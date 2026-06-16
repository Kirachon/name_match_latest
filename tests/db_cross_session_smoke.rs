//! Cross-session dual-pool DB streaming smoke (Gate G1 / G8).
//!
//! Run manually when Docker MySQL is available:
//! ```text
//! docker compose up -d matchers-mysql
//! set MYSQL_IMPORT_TEST_URL=mysql://root:root@127.0.0.1:3307/duplicate_checker
//! set CARGO_HOME=D:\GitProjects\name_match_latest\.cargo-test-home
//! cargo test --test db_cross_session_smoke -- --ignored
//! ```
//!
//! Optional: `MYSQL_SMOKE_ROWS=10000` for the full gate ladder (default 1000 for faster runs).

use name_matcher::db::get_person_rows;
use name_matcher::matching::{
    stream_match_csv_dual, ComputeBackend, MatchOptions, MatchPair, MatchingAlgorithm,
    ProgressConfig, StreamingConfig, match_all_with_opts,
};
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

const TABLE_A: &str = "smoke_cross_a";
const TABLE_B: &str = "smoke_cross_b";

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
    // Local `matchers-mysql-1` container defaults (inspect with `docker inspect`).
    Some("mysql://root:rootpass@127.0.0.1:3307/matcher".to_string())
}

async fn connect_pool(url: &str) -> anyhow::Result<MySqlPool> {
    Ok(MySqlPoolOptions::new()
        .max_connections(4)
        .connect(url)
        .await?)
}

async fn ensure_paired_smoke_tables(pool: &MySqlPool, rows: usize) -> anyhow::Result<()> {
    for table in [TABLE_A, TABLE_B] {
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
            "INSERT INTO `{TABLE_A}` (uuid, first_name, middle_name, last_name, birthdate, hh_id) VALUES {}",
            values_a.join(",")
        ))
        .execute(pool)
        .await?;
        sqlx::query(&format!(
            "INSERT INTO `{TABLE_B}` (uuid, first_name, middle_name, last_name, birthdate, hh_id) VALUES {}",
            values_b.join(",")
        ))
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

fn demographic_keys(pairs: &[MatchPair]) -> BTreeSet<(String, String, String)> {
    pairs
        .iter()
        .map(|pair| {
            (
                pair.person1.first_name.clone().unwrap_or_default(),
                pair.person1.last_name.clone().unwrap_or_default(),
                pair.person1
                    .birthdate
                    .map(|d| d.format("%Y-%m-%d").to_string())
                    .unwrap_or_default(),
            )
        })
        .collect()
}

fn cross_session_streaming_config(rows: u64) -> RunConfigDto {
    RunConfigDto {
        source: TableSelectionDto {
            source_kind: DataSourceKindDto::Database,
            session_id: "session-a".into(),
            table: TABLE_A.into(),
            row_count: Some(rows),
            ..Default::default()
        },
        target: TableSelectionDto {
            source_kind: DataSourceKindDto::Database,
            session_id: "session-b".into(),
            table: TABLE_B.into(),
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

#[test]
fn cross_session_policy_selects_two_pool_not_single_pool() {
    assert!(SUPPORTS_CROSS_SESSION_STREAMING);
    let cfg = cross_session_streaming_config(150_000);
    assert!(should_use_two_pool_db_streaming_worker(&cfg));
    assert!(!should_use_db_streaming_worker(&cfg));
}

#[tokio::test]
#[ignore = "requires MYSQL_IMPORT_TEST_URL and running MySQL (docker compose matchers-mysql)"]
async fn db_cross_session_streaming_matches_in_memory_baseline() -> anyhow::Result<()> {
    let url = mysql_url().expect("MYSQL_IMPORT_TEST_URL or default docker URL");
    let rows = smoke_rows();
    let pool_a = connect_pool(&url).await?;
    let pool_b = connect_pool(&url).await?;

    ensure_paired_smoke_tables(&pool_a, rows).await?;

    let table1 = get_person_rows(&pool_a, TABLE_A).await?;
    let table2 = get_person_rows(&pool_b, TABLE_B).await?;
    assert_eq!(table1.len(), rows);
    assert_eq!(table2.len(), rows);

    let mem_opts = MatchOptions {
        backend: ComputeBackend::Cpu,
        gpu: None,
        progress: ProgressConfig::default(),
        allow_birthdate_swap: false,
    };
    let mem_pairs = match_all_with_opts(
        &table1,
        &table2,
        MatchingAlgorithm::IdUuidYasIsMatchedInfnbd,
        mem_opts,
        |_| {},
    );

    let mut stream_pairs = Vec::new();
    let stream_count = stream_match_csv_dual(
        &pool_a,
        &pool_b,
        TABLE_A,
        TABLE_B,
        MatchingAlgorithm::IdUuidYasIsMatchedInfnbd,
        |pair| {
            stream_pairs.push(pair.clone());
            Ok(())
        },
        StreamingConfig {
            batch_size: 5_000,
            ..Default::default()
        },
        |_| {},
        None,
    )
    .await?;

    assert_eq!(stream_count, mem_pairs.len());
    assert_eq!(stream_pairs.len(), mem_pairs.len());
    assert!(!mem_pairs.is_empty(), "fixture must produce matches");

    assert_eq!(pair_keys(&mem_pairs), pair_keys(&stream_pairs));
    assert_eq!(
        demographic_keys(&mem_pairs),
        demographic_keys(&stream_pairs),
        "stable demographic pair keys must match"
    );

    for (mem, streamed) in mem_pairs.iter().zip(stream_pairs.iter()) {
        assert!(
            (mem.confidence - streamed.confidence).abs() < 1e-6,
            "confidence mismatch for pair {}:{}",
            mem.person1.id,
            mem.person2.id
        );
    }

    sqlx::query(&format!("DROP TABLE IF EXISTS `{TABLE_A}`"))
        .execute(&pool_a)
        .await?;
    sqlx::query(&format!("DROP TABLE IF EXISTS `{TABLE_B}`"))
        .execute(&pool_a)
        .await?;
    pool_a.close().await;
    pool_b.close().await;
    Ok(())
}
