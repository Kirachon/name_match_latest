//! Legacy argument parsing from CLI positional args and environment variables.
//!
//! This module handles the traditional CLI interface where arguments are passed
//! positionally and can be overridden by environment variables.

use anyhow::{Context, Result};
use std::collections::HashMap;

use crate::config::DatabaseConfig;
use crate::matching::MatchingAlgorithm;
use crate::util::envfile::parse_env_file;

/// Parsed CLI arguments for the matching run.
#[derive(Debug, Clone)]
pub struct ParsedArgs {
    /// Database configuration
    pub db_config: DatabaseConfig,
    /// Optional secondary database configuration (for dual-DB mode)
    pub db2_config: Option<DatabaseConfig>,
    /// Source table name
    pub table1: String,
    /// Target table name
    pub table2: String,
    /// Selected matching algorithm
    pub algorithm: MatchingAlgorithm,
    /// Algorithm number (1-7)
    pub algo_num: u8,
    /// Output file path
    pub out_path: String,
    /// Output format (csv, xlsx, both)
    pub format: String,
}

/// Parse database configuration from environment variables and CLI args.
///
/// Priority: AppConfig > process env > .env file > CLI positional args
pub fn parse_db_config(
    app_cfg_opt: Option<&crate::config::AppConfig>,
    env_map: &HashMap<String, String>,
    args: &[String],
) -> Result<DatabaseConfig> {
    if let Some(cfg) = app_cfg_opt {
        return Ok(DatabaseConfig {
            host: cfg.database.host.clone(),
            port: cfg.database.port,
            username: cfg.database.username.clone(),
            password: cfg.database.password.clone(),
            database: cfg.database.database.clone(),
        });
    }

    // Prefer process environment over .env to allow per-run overrides; fallback to CLI args
    let host = std::env::var("DB_HOST")
        .ok()
        .or_else(|| env_map.get("DB_HOST").cloned())
        .unwrap_or_else(|| args.get(1).cloned().unwrap_or_default());
    let port: u16 = std::env::var("DB_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .or_else(|| env_map.get("DB_PORT").and_then(|s| s.parse().ok()))
        .or_else(|| args.get(2).and_then(|s| s.parse().ok()))
        .context("Invalid port")?;
    let user = std::env::var("DB_USER")
        .ok()
        .or_else(|| env_map.get("DB_USER").cloned())
        .unwrap_or_else(|| args.get(3).cloned().unwrap_or_default());
    let pass = std::env::var("DB_PASSWORD")
        .ok()
        .or_else(|| env_map.get("DB_PASSWORD").cloned())
        .unwrap_or_else(|| args.get(4).cloned().unwrap_or_default());
    let dbname = std::env::var("DB_NAME")
        .ok()
        .or_else(|| env_map.get("DB_NAME").cloned())
        .unwrap_or_else(|| args.get(5).cloned().unwrap_or_default());

    Ok(DatabaseConfig {
        host,
        port,
        username: user,
        password: pass,
        database: dbname,
    })
}

/// Parse optional secondary database configuration for dual-DB mode.
pub fn parse_db2_config(primary: &DatabaseConfig) -> Option<DatabaseConfig> {
    let host2 = std::env::var("DB2_HOST").ok()?;
    let port2 = std::env::var("DB2_PORT")
        .ok()
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(primary.port);
    let user2 = std::env::var("DB2_USER")
        .ok()
        .unwrap_or_else(|| primary.username.clone());
    let pass2 = std::env::var("DB2_PASS")
        .ok()
        .unwrap_or_else(|| primary.password.clone());
    let db2 = std::env::var("DB2_DATABASE")
        .ok()
        .unwrap_or_else(|| primary.database.clone());
    Some(DatabaseConfig {
        host: host2,
        port: port2,
        username: user2,
        password: pass2,
        database: db2,
    })
}

/// Parse table names from environment/CLI args.
pub fn parse_table_names(env_map: &HashMap<String, String>, args: &[String]) -> (String, String) {
    let table1 = args
        .get(6)
        .cloned()
        .or_else(|| env_map.get("TABLE1").cloned())
        .unwrap_or_else(|| std::env::var("TABLE1").unwrap_or_else(|_| "table1".into()));
    let table2 = args
        .get(7)
        .cloned()
        .or_else(|| env_map.get("TABLE2").cloned())
        .unwrap_or_else(|| std::env::var("TABLE2").unwrap_or_else(|_| "table2".into()));
    (table1, table2)
}

/// Parse algorithm number from environment/CLI args.
pub fn parse_algo_num(
    app_cfg_opt: Option<&crate::config::AppConfig>,
    env_map: &HashMap<String, String>,
    args: &[String],
) -> u8 {
    if let Some(cfg) = app_cfg_opt {
        if let Some(a) = cfg.matching.algorithm {
            return a;
        }
    }
    args.get(8)
        .and_then(|s| s.parse().ok())
        .or_else(|| env_map.get("ALGO").and_then(|s| s.parse().ok()))
        .or_else(|| std::env::var("ALGO").ok().and_then(|s| s.parse().ok()))
        .unwrap_or(1)
}

/// Convert algorithm number to MatchingAlgorithm enum.
pub fn algo_from_num(algo_num: u8) -> Option<MatchingAlgorithm> {
    match algo_num {
        1 => Some(MatchingAlgorithm::IdUuidYasIsMatchedInfnbd),
        2 => Some(MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd),
        3 => Some(MatchingAlgorithm::Fuzzy),
        4 => Some(MatchingAlgorithm::FuzzyNoMiddle),
        5 => Some(MatchingAlgorithm::HouseholdGpu),
        6 => Some(MatchingAlgorithm::HouseholdGpuOpt6),
        7 => Some(MatchingAlgorithm::LevenshteinWeighted),
        _ => None,
    }
}

/// Parse output path from environment/CLI args.
pub fn parse_out_path(
    app_cfg_opt: Option<&crate::config::AppConfig>,
    env_map: &HashMap<String, String>,
    args: &[String],
) -> String {
    if let Some(cfg) = app_cfg_opt {
        if let Some(path) = &cfg.export.out_path {
            return path.clone();
        }
    }
    args.get(9)
        .cloned()
        .or_else(|| env_map.get("OUT_PATH").cloned())
        .unwrap_or_else(|| std::env::var("OUT_PATH").unwrap_or_else(|_| "matches.csv".into()))
}

/// Parse output format from environment/CLI args.
pub fn parse_format(app_cfg_opt: Option<&crate::config::AppConfig>, args: &[String]) -> String {
    if let Some(cfg) = app_cfg_opt {
        if let Some(fmt) = &cfg.export.format {
            return fmt.to_ascii_lowercase();
        }
    }
    args.get(10)
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_else(|| "csv".to_string())
}

/// Check if minimum required arguments are present.
pub fn has_required_args(args: &[String]) -> bool {
    args.len() >= 10
        || (std::env::var("DB_HOST").is_ok()
            && std::env::var("DB_PORT").is_ok()
            && std::env::var("DB_USER").is_ok()
            && std::env::var("DB_PASSWORD").is_ok()
            && std::env::var("DB_NAME").is_ok())
}

/// Print usage help message.
pub fn print_usage(program_name: &str) {
    eprintln!(
        "Usage: {} <host> <port> <user> <password> <database> <table1> <table2> <algo:1|2|3|4|5|6> <out_path> [format: csv|xlsx|both] [--gpu-hash-join] [--gpu-fuzzy-direct-hash] [--direct-fuzzy-normalization] [--gpu-fuzzy-metrics]",
        program_name
    );
    eprintln!(
        "       {} env-template [path]   # generate a .env.template",
        program_name
    );
    eprintln!("Notes:");
    eprintln!(
        "  --gpu-hash-join                  Enable GPU-accelerated hash join prefilter for Algorithms 1/2 (falls back to CPU if unavailable)"
    );
    eprintln!("  NAME_MATCHER_GPU_HASH_JOIN=1     can also enable this feature");
    eprintln!(
        "  --gpu-fuzzy-direct-hash          GPU hash pre-pass for Fuzzy's direct phase (candidate filter only; behavior preserved)"
    );
    eprintln!("  NAME_MATCHER_GPU_FUZZY_DIRECT_HASH=1 to enable the above");
    eprintln!(
        "  --gpu-levenshtein-prepass       GPU pre-pass for Option 7 (LevenshteinWeighted) candidate filtering with strict CPU parity"
    );
    eprintln!("  NAME_MATCHER_GPU_LEVENSHTEIN_PREPASS=1 to enable the above");
    eprintln!(
        "  --gpu-levenshtein-full-scoring  GPU full scoring for Option 7 (LevenshteinWeighted) with strict CPU parity"
    );
    eprintln!("  NAME_MATCHER_GPU_LEVENSHTEIN_FULL_SCORING=1 to enable the above");
    eprintln!(
        "  --direct-fuzzy-normalization     Apply Fuzzy-style normalization to Algorithms 1 & 2 before equality checks"
    );
    eprintln!("  NAME_MATCHER_DIRECT_FUZZY_NORMALIZATION=1 to enable the above");
    eprintln!("  --gpu-streams <N>                Number of CUDA streams for overlap (default 1)");
    eprintln!("  NAME_MATCHER_GPU_STREAMS=<N>     set via environment");
    eprintln!(
        "  --gpu-buffer-pool | --no-gpu-buffer-pool   Reuse device buffers within a run (default on)"
    );
    eprintln!("  NAME_MATCHER_GPU_BUFFER_POOL=0/1 configure via environment");
    eprintln!(
        "  --gpu-pinned-host                Use pinned host memory for transfers when available"
    );
    eprintln!("  NAME_MATCHER_GPU_PINNED_HOST=1   enable via environment (best-effort)");
    eprintln!(
        "  --gpu-fuzzy-metrics              Use GPU kernels for Levenshtein/Jaro/Jaro-Winkler scoring (Algo 3/4)"
    );
    eprintln!("  NAME_MATCHER_GPU_FUZZY_METRICS=1 enable via environment");
    eprintln!(
        "  --gpu-fuzzy-force                Force GPU fuzzy metrics even if heuristics say it's slower"
    );
    eprintln!(
        "  --gpu-fuzzy-disable              Disable GPU fuzzy metrics regardless of other flags"
    );
    eprintln!("  NAME_MATCHER_GPU_FUZZY_FORCE=1  force via environment");
    eprintln!("  NAME_MATCHER_GPU_FUZZY_DISABLE=1 disable via environment");
    eprintln!("  --use-gpu                       Force GPU backend for Option 5 in-memory path");
    eprintln!("  NAME_MATCHER_USE_GPU=1          same as above");
    eprintln!(
        "  --auto-optimize                 Detect hardware and apply optimized settings (streaming + in-memory)"
    );
    eprintln!("  NAME_MATCHER_AUTO_OPTIMIZE=1    same as above");
    eprintln!("Examples:");
    eprintln!(
        "  {} 127.0.0.1 3306 root secret db t1 t2 1 D:/out/matches.csv --gpu-hash-join",
        program_name
    );
    eprintln!(
        "  {} 127.0.0.1 3306 root secret db t1 t2 1 D:/out/matches.xlsx xlsx",
        program_name
    );
    eprintln!(
        "  {} 127.0.0.1 3306 root secret db t1 t2 1 D:/out/matches both",
        program_name
    );
}
