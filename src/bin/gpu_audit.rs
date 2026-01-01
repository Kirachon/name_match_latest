use anyhow::{Context, Result};
use log::LevelFilter;
use sqlx::{MySql, MySqlPool, mysql::MySqlPoolOptions};
use std::time::Instant;

use name_matcher::db::schema::get_person_rows;
use name_matcher::matching::{
    ComputeBackend, GpuConfig, MatchOptions, MatchingAlgorithm, ProgressConfig, ProgressUpdate,
    match_all_with_opts,
};
use name_matcher::models::Person;

// Minimal logger to capture `log::warn!` from library without external deps
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

// Minimal linear congruential generator
struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as u32
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // init logger
    let _ = log::set_logger(&LOGGER).map(|_| log::set_max_level(LevelFilter::Info));

    // 1) Connect to DB
    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "mysql://root:root@localhost:3307/duplicate_checker".to_string());

    println!("[gpu_audit] Connecting to {}", db_url);
    let pool = MySqlPoolOptions::new()
        .max_connections(8)
        .connect(&db_url)
        .await?;

    // 2) Create/seed tables
    create_schema(&pool).await?;
    seed_if_needed(&pool, "sample_a").await?;
    seed_if_needed(&pool, "sample_b").await?;

    // 3) Load data to memory (GUI does this too)
    let t1: Vec<Person> = get_person_rows(&pool, "sample_a")
        .await
        .context("fetch A")?;
    let t2: Vec<Person> = get_person_rows(&pool, "sample_b")
        .await
        .context("fetch B")?;
    println!(
        "[gpu_audit] Loaded: A={} rows, B={} rows",
        t1.len(),
        t2.len()
    );

    // 4) Run fuzzy with GPU (with GPU fuzzy metrics enabled)
    name_matcher::matching::set_gpu_fuzzy_metrics(true);
    let opts = MatchOptions {
        backend: ComputeBackend::Gpu,
        gpu: Some(GpuConfig {
            device_id: None,
            mem_budget_mb: 1024,
        }),
        progress: ProgressConfig {
            update_every: 5_000,
            ..Default::default()
        },
        allow_birthdate_swap: false,
    };
    let start = Instant::now();
    let pairs = match_all_with_opts(
        &t1,
        &t2,
        MatchingAlgorithm::Fuzzy,
        opts,
        |u: ProgressUpdate| {
            println!(
                "[progress] stage={} processed={}/{} {:.1}% gpu_active={} gpu_total_mb={} gpu_free_mb={}",
                u.stage,
                u.processed,
                u.total,
                u.percent,
                u.gpu_active,
                u.gpu_total_mb,
                u.gpu_free_mb
            );
        },
    );
    let dur = start.elapsed();
    println!(
        "[gpu_audit] Fuzzy GPU done: {} pairs, elapsed: {:.3}s",
        pairs.len(),
        dur.as_secs_f32()
    );

    // 6) CPU baseline
    let start2 = Instant::now();
    let _pairs_cpu = match_all_with_opts(
        &t1,
        &t2,
        MatchingAlgorithm::Fuzzy,
        MatchOptions {
            backend: ComputeBackend::Cpu,
            gpu: None,
            progress: ProgressConfig::default(),
            allow_birthdate_swap: false,
        },
        |_| {},
    );
    let dur2 = start2.elapsed();
    println!(
        "[gpu_audit] Fuzzy CPU baseline elapsed: {:.3}s",
        dur2.as_secs_f32()
    );

    // 7) Streaming GPU hash-join microbench (Algorithms 1 & 2)
    let mut discard = |_: &name_matcher::matching::MatchPair| -> Result<()> { Ok(()) };
    let on_prog = |_u: ProgressUpdate| { /* suppress */ };
    let mut scfg = name_matcher::matching::StreamingConfig::default();
    scfg.use_gpu_hash_join = true;
    scfg.use_gpu_build_hash = true;
    scfg.use_gpu_probe_hash = true;
    scfg.gpu_probe_batch_mb = 256;
    scfg.batch_size = 20000;

    // Single stream (no overlap)
    scfg.gpu_streams = 1;
    let t_start = Instant::now();
    let _ = name_matcher::matching::stream_match_csv(
        &pool,
        "sample_a",
        "sample_b",
        MatchingAlgorithm::IdUuidYasIsMatchedInfnbd,
        &mut discard,
        scfg.clone(),
        &on_prog,
        None,
    )
    .await?;
    let dur_s1 = t_start.elapsed();

    // Double stream (overlap on)
    scfg.gpu_streams = 2;
    scfg.gpu_buffer_pool = true;
    let t_start2 = Instant::now();
    let _ = name_matcher::matching::stream_match_csv(
        &pool,
        "sample_a",
        "sample_b",
        MatchingAlgorithm::IdUuidYasIsMatchedInfnbd,
        &mut discard,
        scfg.clone(),
        &on_prog,
        None,
    )
    .await?;
    let dur_s2 = t_start2.elapsed();

    println!(
        "[gpu_audit] GPU hash join A1: streams=1 {:.3}s | streams=2 {:.3}s (batch {} rows)",
        dur_s1.as_secs_f32(),
        dur_s2.as_secs_f32(),
        scfg.batch_size
    );

    Ok(())
}

async fn create_schema(pool: &MySqlPool) -> Result<()> {
    sqlx::query("CREATE TABLE IF NOT EXISTS sample_a (\n        id BIGINT PRIMARY KEY AUTO_INCREMENT,\n        uuid VARCHAR(64) NOT NULL,\n        first_name VARCHAR(64) NOT NULL,\n        middle_name VARCHAR(64) NULL,\n        last_name VARCHAR(64) NOT NULL,\n        birthdate DATE NOT NULL\n    )").execute(pool).await?;
    sqlx::query("CREATE TABLE IF NOT EXISTS sample_b (\n        id BIGINT PRIMARY KEY AUTO_INCREMENT,\n        uuid VARCHAR(64) NOT NULL,\n        first_name VARCHAR(64) NOT NULL,\n        middle_name VARCHAR(64) NULL,\n        last_name VARCHAR(64) NOT NULL,\n        birthdate DATE NOT NULL\n    )").execute(pool).await?;
    Ok(())
}

async fn seed_if_needed(pool: &MySqlPool, table: &str) -> Result<()> {
    let row: (i64,) = sqlx::query_as::<MySql, (i64,)>(&format!("SELECT COUNT(*) FROM `{}`", table))
        .fetch_one(pool)
        .await?;
    if row.0 >= 1000 {
        println!("[gpu_audit] {} already has {} rows", table, row.0);
        return Ok(());
    }
    println!("[gpu_audit] Seeding {}...", table);

    // simple LCG PRNG to avoid external deps
    let mut rng = Lcg(42);
    let firsts = vec![
        "John", "Jon", "Jonathan", "Jane", "Jan", "Anne", "Ann", "Ana", "Michael", "Micheal",
        "Michel", "Mikael", "Robert", "Rob", "Bob", "Mary", "Maria", "Mariya", "Luis", "Louis",
        "Alicia", "Alisha", "Alex", "Aleks", "Sofia", "Sophia", "Zoe", "Zoey", "Noah", "Noa",
    ];
    let middles = vec!["", "A.", "B.", "C.", "Anne", "Lee", "Marie", "Jo", "Ray"];
    let lasts = vec![
        "Smith",
        "Smyth",
        "Smythe",
        "Johnson",
        "Jonson",
        "Johnsen",
        "Williams",
        "Wiliams",
        "Brown",
        "Browne",
        "Taylor",
        "Tailor",
        "Anderson",
        "Andersen",
        "Martin",
        "Martins",
        "Lee",
        "Li",
        "Kim",
        "Kimm",
        "Garcia",
        "Garza",
        "Gonzalez",
        "Gonzales",
        "Rodriguez",
        "Rodrigez",
        "Lopez",
        "Lopes",
    ];

    let mut tx = pool.begin().await?;
    for _ in 0..1100 {
        let f = firsts[(rng.next_u32() as usize) % firsts.len()];
        let mut l = lasts[(rng.next_u32() as usize) % lasts.len()].to_string();
        // randomly mutate last name slightly
        if rng.next_u32() % 100 < 15 {
            l = l.replace('i', "y");
        }
        if rng.next_u32() % 100 < 10 {
            l.push('e');
        }
        let m = middles[(rng.next_u32() as usize) % middles.len()].to_string();
        let year = 1970 + ((rng.next_u32() % 40) as i32);
        let month = 1 + ((rng.next_u32() % 12) as i32);
        let day = 1 + ((rng.next_u32() % 28) as i32);
        let bd = format!("{:04}-{:02}-{:02}", year, month, day);
        let uuid = format!("{:08x}", rng.next_u32());
        sqlx::query(&format!(
            "INSERT INTO `{}` (uuid,first_name,middle_name,last_name,birthdate) VALUES (?,?,?,?,?)",
            table
        ))
        .bind(&uuid)
        .bind(f)
        .bind(if m.is_empty() {
            None::<String>
        } else {
            Some(m)
        })
        .bind(&l)
        .bind(&bd)
        .execute(&mut *tx)
        .await?;
    }
    tx.commit().await?;
    println!("[gpu_audit] Seeded {} with ~1100 rows", table);
    Ok(())
}
