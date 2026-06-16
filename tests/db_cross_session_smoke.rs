//! DB streaming smoke: cross-session dual-pool, same-session partitioned, memory, cancel.
//!
//! Policy tests run in CI without MySQL. Integration tests are `#[ignore]` locally/CI:
//! ```text
//! docker compose up -d matchers-mysql
//! set MYSQL_IMPORT_TEST_URL=<mysql-url-for-local-test-database>
//! cargo test --test db_cross_session_smoke -- --ignored
//! ```
//!
//! `MYSQL_SMOKE_ROWS` defaults to 1000; use 10000 for the full G1 gate ladder.

use name_matcher::db::get_person_rows;
use name_matcher::matching::{
    stream_match_csv_dual, stream_match_csv_partitioned, ComputeBackend, MatchOptions,
    MatchPair, MatchingAlgorithm, PartitioningConfig, ProgressConfig, StreamControl,
    StreamingConfig, match_all_with_opts,
};
use name_matcher::metrics::memory_stats_mb;
use name_matcher::run_service::dto::{
    AlgorithmDto, DataSourceKindDto, ExportOptionsDto, GpuOptionsDto, MatchOptionsDto,
    RunConfigDto, RunModeDto, StreamingOptionsDto, TableSelectionDto,
};
use name_matcher::run_service::scale::{
    should_use_db_streaming_worker, should_use_two_pool_db_streaming_worker,
    SUPPORTS_CROSS_SESSION_STREAMING,
};
use sqlx::mysql::MySqlPoolOptions;
use sqlx::MySqlPool;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

const ALGO: MatchingAlgorithm = MatchingAlgorithm::IdUuidYasIsMatchedInfnbd;

struct SmokeTables {
    table_a: String,
    table_b: String,
}

impl SmokeTables {
    fn new(prefix: &str) -> Self {
        Self {
            table_a: format!("{prefix}_a"),
            table_b: format!("{prefix}_b"),
        }
    }
}

fn smoke_rows() -> usize {
    std::env::var("MYSQL_SMOKE_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_000)
}

fn mysql_url() -> Option<String> {
    if let Ok(url) = std::env::var("MYSQL_IMPORT_TEST_URL") {
        return Some(url);
    }
    Some("mysql://root:rootpass@127.0.0.1:3307/matcher".to_string())
}

async fn connect_pool(url: &str) -> anyhow::Result<MySqlPool> {
    Ok(MySqlPoolOptions::new()
        .max_connections(4)
        .connect(url)
        .await?)
}

struct PeakMemTracker {
    peak_used_mb: AtomicUsize,
}

impl PeakMemTracker {
    fn new() -> Self {
        Self {
            peak_used_mb: AtomicUsize::new(0),
        }
    }

    fn sample(&self) {
        let used = memory_stats_mb().used_mb as usize;
        let mut current = self.peak_used_mb.load(Ordering::Relaxed);
        while used > current {
            match self.peak_used_mb.compare_exchange_weak(
                current,
                used,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(next) => current = next,
            }
        }
    }

    fn peak_mb(&self) -> u64 {
        self.peak_used_mb.load(Ordering::Relaxed) as u64
    }
}

async fn ensure_paired_smoke_tables(
    pool: &MySqlPool,
    tables: &SmokeTables,
    rows: usize,
) -> anyhow::Result<()> {
    for table in [&tables.table_a, &tables.table_b] {
        sqlx::query(&format!("DROP TABLE IF EXISTS `{table}`"))
            .execute(pool)
            .await?;
        sqlx::query(&format!(
            "CREATE TABLE `{table}` (
                id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                uuid VARCHAR(36) NULL,
                first_name VARCHAR(100) NOT NULL,
                middle_name VARCHAR(100) NULL,
                last_name VARCHAR(100) NOT NULL,
                birthdate DATE NOT NULL,
                hh_id VARCHAR(50) NULL,
                INDEX idx_name_bd (last_name, first_name, birthdate)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci"
        ))
        .execute(pool)
        .await?;
    }

    const CHUNK: usize = 500;
    for chunk_start in (0..rows).step_by(CHUNK) {
        let chunk_end = (chunk_start + CHUNK).min(rows);
        let mut values_a = Vec::new();
        let mut values_b = Vec::new();
        for i in chunk_start..chunk_end {
            let first = format!("First{i}");
            let last = format!("Last{i}");
            let day = (i % 28) + 1;
            let birthdate = format!("1990-01-{day:02}");
            values_a.push(format!(
                "('{uuid}','{first}',NULL,'{last}','{birthdate}',NULL)",
                uuid = uuid::Uuid::new_v4(),
                first = first,
                last = last,
                birthdate = birthdate
            ));
            values_b.push(format!(
                "('{uuid}','{first}',NULL,'{last}','{birthdate}',NULL)",
                uuid = uuid::Uuid::new_v4(),
                first = first,
                last = last,
                birthdate = birthdate
            ));
        }
        sqlx::query(&format!(
            "INSERT INTO `{}` (uuid, first_name, middle_name, last_name, birthdate, hh_id) VALUES {}",
            tables.table_a,
            values_a.join(",")
        ))
        .execute(pool)
        .await?;
        sqlx::query(&format!(
            "INSERT INTO `{}` (uuid, first_name, middle_name, last_name, birthdate, hh_id) VALUES {}",
            tables.table_b,
            values_b.join(",")
        ))
        .execute(pool)
        .await?;
    }
    Ok(())
}

async fn drop_smoke_tables(pool: &MySqlPool, tables: &SmokeTables) -> anyhow::Result<()> {
    for table in [&tables.table_a, &tables.table_b] {
        sqlx::query(&format!("DROP TABLE IF EXISTS `{table}`"))
            .execute(pool)
            .await?;
    }
    Ok(())
}

fn pair_keys(pairs: &[MatchPair]) -> BTreeSet<(i64, i64)> {
    pairs
        .iter()
        .map(|pair| (pair.person1.id, pair.person2.id))
        .collect()
}

fn in_memory_pairs(table1: &[name_matcher::models::Person], table2: &[name_matcher::models::Person]) -> Vec<MatchPair> {
    let mem_opts = MatchOptions {
        backend: ComputeBackend::Cpu,
        gpu: None,
        progress: ProgressConfig::default(),
        allow_birthdate_swap: false,
    };
    match_all_with_opts(table1, table2, ALGO, mem_opts, |_| {})
}

fn streaming_config(batch_size: i64) -> StreamingConfig {
    StreamingConfig {
        batch_size,
        ..Default::default()
    }
}

fn partitioned_config() -> PartitioningConfig {
    PartitioningConfig {
        strategy: "last_initial".into(),
        enabled: true,
    }
}

fn db_streaming_config(source_session: &str, target_session: &str, rows: u64) -> RunConfigDto {
    RunConfigDto {
        source: TableSelectionDto {
            source_kind: DataSourceKindDto::Database,
            session_id: source_session.into(),
            table: "smoke_policy_a".into(),
            row_count: Some(rows),
            ..Default::default()
        },
        target: TableSelectionDto {
            source_kind: DataSourceKindDto::Database,
            session_id: target_session.into(),
            table: "smoke_policy_b".into(),
            row_count: Some(rows),
            ..Default::default()
        },
        algorithm: AlgorithmDto::DeterministicFnLnBd,
        streaming: StreamingOptionsDto {
            mode: RunModeDto::Streaming,
            batch_size: 5_000,
            ..Default::default()
        },
        options: MatchOptionsDto::default(),
        gpu: GpuOptionsDto::default(),
        export: ExportOptionsDto::default(),
        cascade: None,
        review_band: None,
    }
}

async fn run_dual_stream(
    pool_a: &MySqlPool,
    pool_b: &MySqlPool,
    tables: &SmokeTables,
    mem: &PeakMemTracker,
    ctrl: Option<StreamControl>,
) -> anyhow::Result<(usize, Vec<MatchPair>)> {
    mem.sample();
    let mut stream_pairs = Vec::new();
    let count = stream_match_csv_dual(
        pool_a,
        pool_b,
        &tables.table_a,
        &tables.table_b,
        ALGO,
        |pair| {
            stream_pairs.push(pair.clone());
            Ok(())
        },
        streaming_config(5_000),
        |_| mem.sample(),
        ctrl,
    )
    .await?;
    mem.sample();
    Ok((count, stream_pairs))
}

async fn run_partitioned_stream(
    pool: &MySqlPool,
    tables: &SmokeTables,
    mem: &PeakMemTracker,
    ctrl: Option<StreamControl>,
) -> anyhow::Result<(usize, Vec<MatchPair>)> {
    mem.sample();
    let mut stream_pairs = Vec::new();
    let count = stream_match_csv_partitioned(
        pool,
        &tables.table_a,
        &tables.table_b,
        ALGO,
        |pair| {
            stream_pairs.push(pair.clone());
            Ok(())
        },
        streaming_config(5_000),
        |_| mem.sample(),
        ctrl,
        None,
        None,
        partitioned_config(),
    )
    .await?;
    mem.sample();
    Ok((count, stream_pairs))
}

fn assert_pair_parity(mem_pairs: &[MatchPair], stream_pairs: &[MatchPair]) {
    assert_eq!(mem_pairs.len(), stream_pairs.len());
    assert!(!mem_pairs.is_empty(), "fixture must produce matches");
    assert_eq!(pair_keys(mem_pairs), pair_keys(stream_pairs));
    for (mem, streamed) in mem_pairs.iter().zip(stream_pairs.iter()) {
        assert!(
            (mem.confidence - streamed.confidence).abs() < 1e-6,
            "confidence mismatch for pair {}:{}",
            mem.person1.id,
            mem.person2.id
        );
    }
}

#[test]
fn cross_session_policy_selects_two_pool_not_single_pool() {
    assert!(SUPPORTS_CROSS_SESSION_STREAMING);
    let cfg = db_streaming_config("session-a", "session-b", 150_000);
    assert!(should_use_two_pool_db_streaming_worker(&cfg));
    assert!(!should_use_db_streaming_worker(&cfg));
}

#[test]
fn same_session_policy_selects_partitioned_not_two_pool() {
    let cfg = db_streaming_config("session-a", "session-a", 150_000);
    assert!(should_use_db_streaming_worker(&cfg));
    assert!(!should_use_two_pool_db_streaming_worker(&cfg));
}

#[tokio::test]
#[ignore = "requires MYSQL_IMPORT_TEST_URL and running MySQL"]
async fn db_cross_session_streaming_matches_in_memory_baseline() -> anyhow::Result<()> {
    let url = mysql_url().expect("MYSQL_IMPORT_TEST_URL or default docker URL");
    let rows = smoke_rows();
    let tables = SmokeTables::new("smoke_cross_parity");
    let pool_a = connect_pool(&url).await?;
    let pool_b = connect_pool(&url).await?;

    ensure_paired_smoke_tables(&pool_a, &tables, rows).await?;

    let table1 = get_person_rows(&pool_a, &tables.table_a).await?;
    let table2 = get_person_rows(&pool_b, &tables.table_b).await?;
    let mem_pairs = in_memory_pairs(&table1, &table2);

    let mem = PeakMemTracker::new();
    let (_, stream_pairs) = run_dual_stream(&pool_a, &pool_b, &tables, &mem, None).await?;
    assert_pair_parity(&mem_pairs, &stream_pairs);

    drop_smoke_tables(&pool_a, &tables).await?;
    pool_a.close().await;
    pool_b.close().await;
    Ok(())
}

#[tokio::test]
#[ignore = "requires MYSQL_IMPORT_TEST_URL and running MySQL"]
async fn db_same_session_partitioned_matches_in_memory_baseline() -> anyhow::Result<()> {
    let url = mysql_url().expect("MYSQL_IMPORT_TEST_URL or default docker URL");
    let rows = smoke_rows();
    let tables = SmokeTables::new("smoke_ss_part");
    let pool = connect_pool(&url).await?;

    ensure_paired_smoke_tables(&pool, &tables, rows).await?;

    let table1 = get_person_rows(&pool, &tables.table_a).await?;
    let table2 = get_person_rows(&pool, &tables.table_b).await?;
    let mem_pairs = in_memory_pairs(&table1, &table2);

    let mem = PeakMemTracker::new();
    let (_, stream_pairs) = run_partitioned_stream(&pool, &tables, &mem, None).await?;
    assert_pair_parity(&mem_pairs, &stream_pairs);

    drop_smoke_tables(&pool, &tables).await?;
    pool.close().await;
    Ok(())
}

#[tokio::test]
#[ignore = "requires MYSQL_IMPORT_TEST_URL and running MySQL"]
async fn db_streaming_memory_cross_session_within_baseline() -> anyhow::Result<()> {
    let url = mysql_url().expect("MYSQL_IMPORT_TEST_URL or default docker URL");
    let rows = smoke_rows().max(1_000);
    let tables = SmokeTables::new("smoke_mem_gate");
    let pool_a = connect_pool(&url).await?;
    let pool_b = connect_pool(&url).await?;
    let pool_same = connect_pool(&url).await?;

    ensure_paired_smoke_tables(&pool_a, &tables, rows).await?;

    let baseline_mem = PeakMemTracker::new();
    let (baseline_count, _) =
        run_partitioned_stream(&pool_same, &tables, &baseline_mem, None).await?;

    let cross_mem = PeakMemTracker::new();
    let (cross_count, _) = run_dual_stream(&pool_a, &pool_b, &tables, &cross_mem, None).await?;

    assert_eq!(baseline_count, cross_count);
    assert!(baseline_count > 0);

    let max_allowed = ((baseline_mem.peak_mb() as f64) * 1.2).ceil() as u64;
    assert!(
        cross_mem.peak_mb() <= max_allowed,
        "cross-session peak {} MB exceeds same-session baseline {} MB + 20% (max {})",
        cross_mem.peak_mb(),
        baseline_mem.peak_mb(),
        max_allowed
    );

    drop_smoke_tables(&pool_a, &tables).await?;
    pool_a.close().await;
    pool_b.close().await;
    pool_same.close().await;
    Ok(())
}

#[tokio::test]
#[ignore = "requires MYSQL_IMPORT_TEST_URL and running MySQL"]
async fn db_cross_session_streaming_cancels_with_partial_results() -> anyhow::Result<()> {
    let url = mysql_url().expect("MYSQL_IMPORT_TEST_URL or default docker URL");
    let rows = smoke_rows().max(2_000);
    let tables = SmokeTables::new("smoke_cancel");
    let pool_a = connect_pool(&url).await?;
    let pool_b = connect_pool(&url).await?;

    ensure_paired_smoke_tables(&pool_a, &tables, rows).await?;

    let table1 = get_person_rows(&pool_a, &tables.table_a).await?;
    let table2 = get_person_rows(&pool_b, &tables.table_b).await?;
    let full_count = in_memory_pairs(&table1, &table2).len();

    let cancel = Arc::new(AtomicBool::new(false));
    let pause = Arc::new(AtomicBool::new(false));
    let ctrl = StreamControl {
        cancel: cancel.clone(),
        pause,
    };
    let seen = Arc::new(AtomicUsize::new(0));

    let mem = PeakMemTracker::new();
    mem.sample();
    let partial_count = stream_match_csv_dual(
        &pool_a,
        &pool_b,
        &tables.table_a,
        &tables.table_b,
        ALGO,
        |pair| {
            let n = seen.fetch_add(1, Ordering::Relaxed) + 1;
            if n >= 50 {
                cancel.store(true, Ordering::Relaxed);
            }
            let _ = pair;
            Ok(())
        },
        streaming_config(5_000),
        |_| mem.sample(),
        Some(ctrl),
    )
    .await?;
    mem.sample();

    assert!(partial_count >= 50, "expected at least 50 matches before cancel");
    assert!(
        partial_count < full_count,
        "cancel should stop before full result set ({partial_count} vs {full_count})"
    );

    drop_smoke_tables(&pool_a, &tables).await?;
    pool_a.close().await;
    pool_b.close().await;
    Ok(())
}
