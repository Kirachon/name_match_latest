use crate::config::DatabaseConfig;
use anyhow::Result;
use sqlx::MySqlPool;
use sqlx::mysql::MySqlPoolOptions;
use std::time::Duration;

pub async fn make_pool(cfg: &DatabaseConfig) -> Result<MySqlPool> {
    make_pool_with_size(cfg, None).await
}

pub async fn make_pool_with_size(cfg: &DatabaseConfig, max: Option<u32>) -> Result<MySqlPool> {
    let url = cfg.to_url();
    let max_conn: u32 = if let Some(m) = max {
        m
    } else if let Ok(s) = std::env::var("NAME_MATCHER_POOL_SIZE") {
        match s.parse::<u32>() {
            Ok(v) if v > 0 => v,
            _ => {
                log::warn!(
                    "Invalid NAME_MATCHER_POOL_SIZE='{}'; using computed default",
                    s
                );
                compute_default_max_conns()
            }
        }
    } else {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8) as u32;
        let multiplier: u32 = if cores > 32 {
            4
        } else if cores > 16 {
            3
        } else {
            2
        };
        let computed = cores.saturating_mul(multiplier);
        let capped = computed.min(128);
        log::info!(
            "DB pool sizing: cores={}, multiplier={}x, computed={}, cap=128, final_max={}",
            cores,
            multiplier,
            computed,
            capped
        );
        capped
    };
    let max_conn = if max_conn == 0 { 16 } else { max_conn };
    let min_conn: u32 = std::env::var("NAME_MATCHER_POOL_MIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let acquire_ms: u64 = std::env::var("NAME_MATCHER_ACQUIRE_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30_000);
    let idle_ms: u64 = std::env::var("NAME_MATCHER_IDLE_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30_000);
    let life_ms: u64 = std::env::var("NAME_MATCHER_LIFETIME_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(600_000);

    let pool = MySqlPoolOptions::new()
        .max_connections(max_conn)
        .min_connections(min_conn.min(max_conn))
        .acquire_timeout(Duration::from_millis(acquire_ms))
        .idle_timeout(Some(Duration::from_millis(idle_ms)))
        .max_lifetime(Some(Duration::from_millis(life_ms)))
        .connect(&url)
        .await?;
    Ok(pool)
}

fn compute_default_max_conns() -> u32 {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8) as u32;
    let multiplier: u32 = if cores > 32 {
        4
    } else if cores > 16 {
        3
    } else {
        2
    };
    let computed = cores.saturating_mul(multiplier);
    let capped = computed.min(128);
    log::info!(
        "DB pool sizing: cores={}, multiplier={}x, computed={}, cap=128, final_max={}",
        cores,
        multiplier,
        computed,
        capped
    );
    capped
}
