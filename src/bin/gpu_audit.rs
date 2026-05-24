use anyhow::{Context, Result, bail};
use log::LevelFilter;
use sqlx::{MySqlPool, mysql::MySqlPoolOptions};
use std::cmp::Ordering;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use name_matcher::db::schema::{get_person_count, get_person_rows};
use name_matcher::matching::{
    self, ComputeBackend, GpuConfig, MatchOptions, MatchPair, MatchingAlgorithm, ProgressConfig,
    ProgressUpdate, match_all_with_opts,
};

// Minimal logger to capture library logs without external deps.
struct SimpleLogger;
impl log::Log for SimpleLogger {
    fn enabled(&self, _metadata: &log::Metadata) -> bool {
        true
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            println!("[{}] {}", record.level(), record.args());
        }
    }

    fn flush(&self) {}
}

static LOGGER: SimpleLogger = SimpleLogger;

#[derive(Debug, Clone)]
struct AuditConfig {
    source_a: String,
    source_b: String,
    row_cap: usize,
    gpu_mem_budget_mb: u64,
    no_clone: bool,
}

#[derive(Debug, Clone)]
struct ScratchTables {
    a: String,
    b: String,
}

#[derive(Debug)]
struct RunResult {
    label: String,
    elapsed: Duration,
    pairs: Vec<MatchPair>,
}

fn parse_usize_env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

fn parse_u64_env(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

fn build_database_url() -> String {
    if let Ok(url) = std::env::var("DATABASE_URL") {
        return url;
    }

    let host = std::env::var("DB_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port = std::env::var("DB_PORT")
        .ok()
        .and_then(|v| v.parse::<u16>().ok())
        .unwrap_or(3306);
    let user = std::env::var("DB_USER").unwrap_or_else(|_| "root".to_string());
    let pass = std::env::var("DB_PASSWORD")
        .or_else(|_| std::env::var("DB_PASS"))
        .unwrap_or_else(|_| "secret".to_string());
    let db = std::env::var("DB_NAME").unwrap_or_else(|_| "duplicate_checker".to_string());

    format!("mysql://{}:{}@{}:{}/{}", user, pass, host, port, db)
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_millis()
}

fn canonical_signature(pair: &MatchPair) -> String {
    let matched_fields = if pair.matched_fields.is_empty() {
        String::new()
    } else {
        pair.matched_fields.join("|")
    };
    format!(
        "{}:{}:{}:{}:{}:{}",
        pair.person1.id,
        pair.person2.id,
        pair.confidence.to_bits(),
        matched_fields,
        pair.is_matched_infnbd,
        pair.is_matched_infnmnbd
    )
}

fn canonical_signatures(mut pairs: Vec<MatchPair>) -> Vec<String> {
    pairs.sort_by(|a, b| {
        let ord = a.person1.id.cmp(&b.person1.id);
        if ord != Ordering::Equal {
            return ord;
        }
        let ord = a.person2.id.cmp(&b.person2.id);
        if ord != Ordering::Equal {
            return ord;
        }
        let ord = a.confidence.to_bits().cmp(&b.confidence.to_bits());
        if ord != Ordering::Equal {
            return ord;
        }
        a.matched_fields.join("|").cmp(&b.matched_fields.join("|"))
    });
    pairs.iter().map(canonical_signature).collect()
}

fn progress_printer(label: &'static str) -> impl Fn(ProgressUpdate) + Sync {
    move |u: ProgressUpdate| {
        if u.processed == 0 || u.processed == u.total || u.processed % 50_000 == 0 {
            println!(
                "[gpu_audit] {label}: stage={} processed={}/{} {:.1}% gpu_active={} mem_used={}MB mem_avail={}MB",
                u.stage,
                u.processed,
                u.total,
                u.percent,
                u.gpu_active,
                u.mem_used_mb,
                u.mem_avail_mb
            );
        }
    }
}

fn validate_sql_identifier(name: &str) -> Result<&str> {
    if name.is_empty() {
        bail!("empty SQL identifier");
    }
    if name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        Ok(name)
    } else {
        bail!("invalid SQL identifier: {name}");
    }
}

async fn clone_source_tables(pool: &MySqlPool, cfg: &AuditConfig) -> Result<ScratchTables> {
    let source_a = validate_sql_identifier(&cfg.source_a)?;
    let source_b = validate_sql_identifier(&cfg.source_b)?;
    let count_a = get_person_count(pool, &cfg.source_a)
        .await
        .with_context(|| format!("counting source table {}", cfg.source_a))?;
    let count_b = get_person_count(pool, &cfg.source_b)
        .await
        .with_context(|| format!("counting source table {}", cfg.source_b))?;
    let source_cap = std::cmp::min(count_a, count_b).max(0) as usize;
    let limit = cfg.row_cap.min(source_cap);
    if limit == 0 {
        bail!(
            "No rows available for canary run: {} has {}, {} has {}",
            cfg.source_a,
            count_a,
            cfg.source_b,
            count_b
        );
    }

    let stamp = format!("{}_{}", std::process::id(), now_millis());
    let scratch = ScratchTables {
        a: format!("gpu_audit_a_{}", stamp),
        b: format!("gpu_audit_b_{}", stamp),
    };

    println!(
        "[gpu_audit] Cloning {} rows into disposable scratch tables {} / {}",
        limit, scratch.a, scratch.b
    );

    drop_scratch_tables(pool, &scratch).await.ok();

    sqlx::query(&format!("CREATE TABLE `{}` LIKE `{}`", scratch.a, source_a))
        .execute(pool)
        .await
        .with_context(|| format!("creating scratch table {}", scratch.a))?;
    sqlx::query(&format!("CREATE TABLE `{}` LIKE `{}`", scratch.b, source_b))
        .execute(pool)
        .await
        .with_context(|| format!("creating scratch table {}", scratch.b))?;

    sqlx::query(&format!(
        "INSERT INTO `{scratch_a}` SELECT * FROM `{source_a}` ORDER BY id LIMIT {limit}",
        scratch_a = scratch.a,
        source_a = source_a,
        limit = limit
    ))
    .execute(pool)
    .await
    .with_context(|| format!("copying rows into scratch table {}", scratch.a))?;
    sqlx::query(&format!(
        "INSERT INTO `{scratch_b}` SELECT * FROM `{source_b}` ORDER BY id LIMIT {limit}",
        scratch_b = scratch.b,
        source_b = source_b,
        limit = limit
    ))
    .execute(pool)
    .await
    .with_context(|| format!("copying rows into scratch table {}", scratch.b))?;

    let scratch_a_count = get_person_count(pool, &scratch.a).await?;
    let scratch_b_count = get_person_count(pool, &scratch.b).await?;
    println!(
        "[gpu_audit] Scratch rows: {}={} and {}={}",
        scratch.a, scratch_a_count, scratch.b, scratch_b_count
    );

    Ok(scratch)
}

async fn drop_scratch_tables(pool: &MySqlPool, tables: &ScratchTables) -> Result<()> {
    let _ = sqlx::query(&format!("DROP TABLE IF EXISTS `{}`", tables.a))
        .execute(pool)
        .await;
    let _ = sqlx::query(&format!("DROP TABLE IF EXISTS `{}`", tables.b))
        .execute(pool)
        .await;
    Ok(())
}

async fn load_people(pool: &MySqlPool, table: &str) -> Result<Vec<name_matcher::models::Person>> {
    get_person_rows(pool, table)
        .await
        .with_context(|| format!("fetching rows from {}", table))
}

async fn run_cpu_equivalent_variant(
    label: &'static str,
    t1: &[name_matcher::models::Person],
    t2: &[name_matcher::models::Person],
) -> RunResult {
    matching::set_gpu_fuzzy_direct_prep(false);
    matching::set_gpu_fuzzy_metrics(false);
    matching::set_gpu_fuzzy_force(false);
    matching::set_gpu_fuzzy_disable(false);

    let start = Instant::now();
    let pairs =
        name_matcher::matching::match_fuzzy_cpu_gpu_equivalent(t1, t2, &progress_printer(label));
    let elapsed = start.elapsed();

    RunResult {
        label: label.to_string(),
        elapsed,
        pairs,
    }
}

async fn run_gpu_prefilter_variant(
    label: &'static str,
    t1: &[name_matcher::models::Person],
    t2: &[name_matcher::models::Person],
    gpu_mem_budget_mb: u64,
) -> RunResult {
    matching::set_gpu_fuzzy_direct_prep(true);
    matching::set_gpu_fuzzy_metrics(false);
    matching::set_gpu_fuzzy_force(false);
    matching::set_gpu_fuzzy_disable(false);

    let opts = MatchOptions {
        backend: ComputeBackend::Gpu,
        gpu: Some(GpuConfig {
            device_id: None,
            mem_budget_mb: gpu_mem_budget_mb,
        }),
        progress: ProgressConfig {
            update_every: 5_000,
            ..Default::default()
        },
        allow_birthdate_swap: false,
    };

    let start = Instant::now();
    let pairs = match_all_with_opts(
        t1,
        t2,
        MatchingAlgorithm::Fuzzy,
        opts,
        progress_printer(label),
    );
    let elapsed = start.elapsed();

    RunResult {
        label: label.to_string(),
        elapsed,
        pairs,
    }
}

fn compare_runs(baseline: &RunResult, experimental: &RunResult) -> Result<()> {
    let base_sig = canonical_signatures(baseline.pairs.clone());
    let exp_sig = canonical_signatures(experimental.pairs.clone());
    if base_sig == exp_sig {
        return Ok(());
    }

    let mut diff_idx = None;
    let min_len = base_sig.len().min(exp_sig.len());
    for i in 0..min_len {
        if base_sig[i] != exp_sig[i] {
            diff_idx = Some(i);
            break;
        }
    }

    if let Some(i) = diff_idx {
        bail!(
            "parity mismatch at index {}: baseline={} experimental={}",
            i,
            base_sig[i],
            exp_sig[i]
        );
    }

    bail!(
        "parity mismatch: baseline_count={} experimental_count={}",
        base_sig.len(),
        exp_sig.len()
    );
}

fn print_run_summary(run: &RunResult) {
    println!(
        "[gpu_audit] {}: {} pairs in {:.3}s",
        run.label,
        run.pairs.len(),
        run.elapsed.as_secs_f32()
    );
}

fn print_shortlist_summary(shortlists: &[Vec<usize>], t1_len: usize, t2_len: usize) {
    let candidate_count: usize = shortlists.iter().map(|v| v.len()).sum();
    let possible_pairs = (t1_len as u128) * (t2_len as u128);
    let reduction = if possible_pairs == 0 {
        0.0
    } else {
        (candidate_count as f64 / possible_pairs as f64) * 100.0
    };
    println!(
        "[gpu_audit] GPU shortlist: {} candidate pairs across {} x {} input rows ({:.4}% of brute force)",
        candidate_count, t1_len, t2_len, reduction
    );
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = log::set_logger(&LOGGER).map(|_| log::set_max_level(LevelFilter::Info));

    let cfg = AuditConfig {
        source_a: std::env::var("NAME_MATCHER_GPU_AUDIT_SOURCE_A")
            .unwrap_or_else(|_| "sample_a".to_string()),
        source_b: std::env::var("NAME_MATCHER_GPU_AUDIT_SOURCE_B")
            .unwrap_or_else(|_| "sample_b".to_string()),
        row_cap: parse_usize_env("NAME_MATCHER_GPU_AUDIT_ROWS", 50_000),
        gpu_mem_budget_mb: parse_u64_env("NAME_MATCHER_GPU_AUDIT_GPU_MEM_MB", 1024),
        no_clone: std::env::var("NAME_MATCHER_GPU_AUDIT_NO_CLONE")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"))
            .unwrap_or(false),
    };
    validate_sql_identifier(&cfg.source_a)?;
    validate_sql_identifier(&cfg.source_b)?;

    let db_url = build_database_url();
    println!("[gpu_audit] Connecting to MySQL target (credentials redacted)");

    let pool = MySqlPoolOptions::new()
        .max_connections(1)
        .connect(&db_url)
        .await?;

    let scratch = if cfg.no_clone {
        println!("[gpu_audit] Using source tables directly (NAME_MATCHER_GPU_AUDIT_NO_CLONE=1)");
        ScratchTables {
            a: cfg.source_a.clone(),
            b: cfg.source_b.clone(),
        }
    } else {
        clone_source_tables(&pool, &cfg).await?
    };
    let run_result = async {
        let t1 = load_people(&pool, &scratch.a).await?;
        let t2 = load_people(&pool, &scratch.b).await?;
        println!(
            "[gpu_audit] Loaded canary data: {} rows from {} and {} rows from {}",
            t1.len(),
            scratch.a,
            t2.len(),
            scratch.b
        );

        let baseline = run_cpu_equivalent_variant("baseline_cpu_equiv", &t1, &t2).await;
        print_run_summary(&baseline);

        let experimental = run_gpu_prefilter_variant(
            "experimental_gpu_prefilter",
            &t1,
            &t2,
            cfg.gpu_mem_budget_mb,
        )
        .await;
        print_run_summary(&experimental);

        compare_runs(&baseline, &experimental)?;

        let shortlist_started = Instant::now();
        let shortlists = name_matcher::matching::gpu_fuzzy_direct_hash_prefilter_indices(
            &t1,
            &t2,
            "last_initial",
        )?;
        let shortlist_elapsed = shortlist_started.elapsed();
        println!(
            "[gpu_audit] shortlist pass completed in {:.3}s",
            shortlist_elapsed.as_secs_f32()
        );
        print_shortlist_summary(&shortlists, t1.len(), t2.len());

        let speedup = if experimental.elapsed.as_secs_f64() > 0.0 {
            baseline.elapsed.as_secs_f64() / experimental.elapsed.as_secs_f64()
        } else {
            0.0
        };
        println!(
            "[gpu_audit] speedup: baseline_cpu_equiv / experimental_gpu_prefilter = {:.2}x",
            speedup
        );

        Ok::<_, anyhow::Error>(())
    }
    .await;

    if !cfg.no_clone {
        if let Err(err) = drop_scratch_tables(&pool, &scratch).await {
            eprintln!("[gpu_audit] cleanup failed: {}", err);
        }
    }

    run_result
}
