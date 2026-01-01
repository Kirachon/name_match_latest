use anyhow::{Context, Result, bail};
use env_logger::Env;
use log::{error, info, warn};
use std::env;

mod config;
mod db;
mod export;
mod matching;
mod metrics;
mod models;
mod normalize;
mod optimization;
mod util;

mod cli;
#[cfg(feature = "new_engine")]
mod engine;
mod error;
#[cfg(feature = "new_cli")]
mod logging;
mod orchestrator;

use crate::config::DatabaseConfig;
use crate::db::schema::get_all_table_columns;
use crate::db::{discover_table_columns, get_person_count, get_person_rows, make_pool};
use crate::export::csv_export::{
    AdvCsvStreamWriter, CsvStreamWriter, HouseholdCsvWriter, export_summary_csv, export_to_csv,
};
use crate::export::xlsx_export::{
    SummaryContext, XlsxStreamWriter, export_households_xlsx, export_to_xlsx,
};
use crate::matching::HouseholdAggRow;
use crate::matching::advanced_matcher::{AdvColumns, AdvConfig, AdvLevel};
use crate::matching::{
    ComputeBackend, GpuConfig, MatchOptions, MatchingAlgorithm, PartitioningConfig, ProgressConfig,
    ProgressUpdate, StreamingConfig, match_all_progress, match_households_gpu_inmemory,
    match_households_gpu_inmemory_opt6, match_households_inmemory_opt6_streaming,
    stream_match_advanced, stream_match_advanced_dual, stream_match_advanced_l12,
    stream_match_csv_dual, stream_match_csv_partitioned,
};
fn algo_label_summary(algo: MatchingAlgorithm) -> &'static str {
    match algo {
        MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => {
            "Option 1: Deterministic Match (First + Last + Birthdate)"
        }
        MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => {
            "Option 2: Deterministic Match (First + Middle + Last + Birthdate)"
        }
        MatchingAlgorithm::Fuzzy => "Option 3: Fuzzy Match (with Middle Name)",
        MatchingAlgorithm::FuzzyNoMiddle => "Option 4: Fuzzy Match (without Middle Name)",
        MatchingAlgorithm::HouseholdGpu => {
            "Option 5: Household Matching (Table1→Table2, uuid→hh_id)"
        }
        MatchingAlgorithm::HouseholdGpuOpt6 => {
            "Option 6: Household Matching (Table2→Table1, hh_id→uuid)"
        }
        MatchingAlgorithm::LevenshteinWeighted => "Option 7: Levenshtein-Weighted (SQL Equivalent)",
    }
}
use crate::util::envfile::{load_dotenv_if_present, parse_env_file, write_env_template};

#[tokio::main]
async fn main() {
    #[cfg(feature = "new_cli")]
    {
        let use_new_cli = std::env::var("NAME_MATCHER_NEW_CLI")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if use_new_cli {
            crate::logging::init_tracing_from_env();
        } else {
            env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
        }
    }
    #[cfg(not(feature = "new_cli"))]
    {
        env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    }

    #[cfg(feature = "new_cli")]
    if std::env::var("NAME_MATCHER_NEW_CLI")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        match crate::cli::parse_cli_to_app_config() {
            Ok(_cfg) => {
                log::info!("New CLI parsed and validated; using legacy engine for execution");
            }
            Err(e) => {
                eprintln!("New CLI parse error: {}", e);
                std::process::exit(2);
            }
        }
    }

    #[cfg(feature = "new_cli")]
    let mut app_cfg_opt: Option<crate::config::AppConfig> = None;

    #[cfg(feature = "new_cli")]
    if std::env::var("NAME_MATCHER_NEW_CLI")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        match crate::cli::parse_cli_to_app_config() {
            Ok(cfg) => {
                app_cfg_opt = Some(cfg);
                log::info!("New CLI parsed and validated; using legacy engine for execution");
            }
            Err(e) => {
                eprintln!("New CLI parse error: {}", e);
                std::process::exit(2);
            }
        }
    }

    #[cfg(feature = "new_cli")]
    {
        if let Err(e) = run(app_cfg_opt).await {
            error!("{:#}", e);
            std::process::exit(1);
        }
    }
    #[cfg(not(feature = "new_cli"))]
    {
        if let Err(e) = run(None).await {
            error!("{:#}", e);
            std::process::exit(1);
        }
    }
}

async fn run(app_cfg_opt: Option<crate::config::AppConfig>) -> Result<()> {
    load_dotenv_if_present()?;
    let env_map = parse_env_file().unwrap_or_default();
    let args: Vec<String> = env::args().collect();

    // Utility subcommand: generate .env.template
    if args.get(1).map(|s| s.as_str()) == Some("env-template") {
        let path = args
            .get(2)
            .cloned()
            .unwrap_or_else(|| ".env.template".to_string());
        write_env_template(&path)?;
        println!("Wrote {}. Copy to .env and edit values as needed.", path);
        return Ok(());
    }

    if args.len() < 10
        && !(std::env::var("DB_HOST").is_ok()
            && std::env::var("DB_PORT").is_ok()
            && std::env::var("DB_USER").is_ok()
            && std::env::var("DB_PASSWORD").is_ok()
            && std::env::var("DB_NAME").is_ok())
    {
        eprintln!(
            "Usage: {} <host> <port> <user> <password> <database> <table1> <table2> <algo:1|2|3|4|5|6> <out_path> [format: csv|xlsx|both] [--gpu-hash-join] [--gpu-fuzzy-direct-hash] [--direct-fuzzy-normalization] [--gpu-fuzzy-metrics]",
            args.get(0).map(String::as_str).unwrap_or("name_matcher")
        );
        eprintln!(
            "       {} env-template [path]   # generate a .env.template",
            args.get(0).map(String::as_str).unwrap_or("name_matcher")
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
        eprintln!(
            "  --gpu-streams <N>                Number of CUDA streams for overlap (default 1)"
        );
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
        eprintln!(
            "  --use-gpu                       Force GPU backend for Option 5 in-memory path"
        );
        eprintln!("  NAME_MATCHER_USE_GPU=1          same as above");
        eprintln!(
            "  --auto-optimize                 Detect hardware and apply optimized settings (streaming + in-memory)"
        );
        eprintln!("  NAME_MATCHER_AUTO_OPTIMIZE=1    same as above");
        eprintln!();
        eprintln!("Cascade Matching (run L1-L11 sequentially):");
        eprintln!(
            "  --cascade                       Run cascade matching (L1-L11). L12 (Household) is EXCLUDED."
        );
        eprintln!(
            "  --cascade-missing-columns MODE  Handle missing geo columns: auto-skip (default), manual, abort"
        );
        eprintln!(
            "  --levels 1,2,3,10,11            Run specific levels only (comma-separated, 1-11)"
        );
        eprintln!(
            "  Output: {{base_path}}_L01.csv, {{base_path}}_L02.csv, ..., {{base_path}}_summary.txt"
        );

        eprintln!("Examples:");
        eprintln!(
            "  {} 127.0.0.1 3306 root secret db t1 t2 1 D:/out/matches.csv --gpu-hash-join",
            args.get(0).unwrap_or(&"name_matcher".to_string())
        );
        eprintln!(
            "  {} 127.0.0.1 3306 root secret db t1 t2 1 D:/out/matches.xlsx xlsx",
            args.get(0).unwrap_or(&"name_matcher".to_string())
        );
        eprintln!(
            "  {} 127.0.0.1 3306 root secret db t1 t2 1 D:/out/matches both",
            args.get(0).unwrap_or(&"name_matcher".to_string())
        );
        std::process::exit(2);
    }

    // If provided via new_cli AppConfig, prefer it; else prefer env, then CLI args
    let (host, port, user, pass, dbname) = if let Some(cfg) = &app_cfg_opt {
        (
            cfg.database.host.clone(),
            cfg.database.port,
            cfg.database.username.clone(),
            cfg.database.password.clone(),
            cfg.database.database.clone(),
        )
    } else {
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
        (host, port, user, pass, dbname)
    };

    let cfg = DatabaseConfig {
        host,
        port,
        username: user,
        password: pass,
        database: dbname,
    };
    // Global run start captured at CLI invocation (before DB connection/fetch/match/export)
    let run_start_utc = chrono::Utc::now();

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

    // Algorithm, out_path, and format may come from AppConfig when using new_cli
    let mut algo_num: u8 = args
        .get(8)
        .and_then(|s| s.parse().ok())
        .or_else(|| env_map.get("ALGO").and_then(|s| s.parse().ok()))
        .or_else(|| std::env::var("ALGO").ok().and_then(|s| s.parse().ok()))
        .unwrap_or(1);
    let mut out_path = args
        .get(9)
        .cloned()
        .or_else(|| env_map.get("OUT_PATH").cloned())
        .unwrap_or_else(|| std::env::var("OUT_PATH").unwrap_or_else(|_| "matches.csv".into()));
    let mut format = args
        .get(10)
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_else(|| "csv".to_string());

    if let Some(cfg) = &app_cfg_opt {
        if let Some(a) = cfg.matching.algorithm {
            algo_num = a;
        }
        if let Some(path) = &cfg.export.out_path {
            out_path = path.clone();
        }
        if let Some(fmt) = &cfg.export.format {
            format = fmt.to_ascii_lowercase();
        }
    }

    let algorithm = match algo_num {
        1 => MatchingAlgorithm::IdUuidYasIsMatchedInfnbd,
        2 => MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd,
        3 => MatchingAlgorithm::Fuzzy,
        4 => MatchingAlgorithm::FuzzyNoMiddle,
        5 => MatchingAlgorithm::HouseholdGpu,
        6 => MatchingAlgorithm::HouseholdGpuOpt6,
        7 => MatchingAlgorithm::LevenshteinWeighted,
        _ => {
            eprintln!("algo must be 1, 2, 3, 4, 5, 6 or 7");
            std::process::exit(2);
        }
    };

    info!(
        "Connecting to MySQL at {}:{} / db {}",
        cfg.host, cfg.port, cfg.database
    );
    let cfg2_opt = {
        let host2 = std::env::var("DB2_HOST").ok();
        if let Some(h2) = host2 {
            let port2 = std::env::var("DB2_PORT")
                .ok()
                .and_then(|s| s.parse::<u16>().ok())
                .unwrap_or(cfg.port);
            let user2 = std::env::var("DB2_USER")
                .ok()
                .unwrap_or_else(|| cfg.username.clone());
            let pass2 = std::env::var("DB2_PASS")
                .ok()
                .unwrap_or_else(|| cfg.password.clone());
            let db2 = std::env::var("DB2_DATABASE")
                .ok()
                .unwrap_or_else(|| cfg.database.clone());
            Some(DatabaseConfig {
                host: h2,
                port: port2,
                username: user2,
                password: pass2,
                database: db2,
            })
        } else {
            None
        }
    };
    let pool1 = make_pool(&cfg).await?;
    let (pool2_opt, db_label) = if let Some(cfg2) = &cfg2_opt {
        info!(
            "Connecting to second MySQL at {}:{} / db {}",
            cfg2.host, cfg2.port, cfg2.database
        );
        let p2 = make_pool(cfg2).await?;
        (Some(p2), format!("{} | {}", cfg.database, cfg2.database))
    } else {
        (None, cfg.database.clone())
    };

    if matches!(
        algorithm,
        MatchingAlgorithm::Fuzzy
            | MatchingAlgorithm::FuzzyNoMiddle
            | MatchingAlgorithm::LevenshteinWeighted
    ) && format != "csv"
    {
        eprintln!("Selected algorithm supports CSV format only. Use format=csv.");
        std::process::exit(2);
    }
    let db2_name = cfg2_opt
        .as_ref()
        .map(|c| c.database.clone())
        .unwrap_or_else(|| cfg.database.clone());

    // Validate schemas
    let cols1 = discover_table_columns(&pool1, &cfg.database, &table1).await?;
    let cols2 =
        discover_table_columns(pool2_opt.as_ref().unwrap_or(&pool1), &db2_name, &table2).await?;
    cols1.validate_basic()?;
    if !(cols2.has_id && cols2.has_first_name && cols2.has_last_name && cols2.has_birthdate) {
        bail!(
            "Table {} missing required columns: requires id, first_name, last_name, birthdate (uuid optional)",
            table2
        );
    }
    info!("{} columns: {:?}", table1, cols1);
    info!("{} columns: {:?}", table2, cols2);
    // Auto-optimization flag/env (after early CSV algo guard)
    let auto_optimize = args.iter().any(|a| a == "--auto-optimize")
        || std::env::var("NAME_MATCHER_AUTO_OPTIMIZE")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
    if auto_optimize {
        // Apply in-memory rayon threads early (before pool build) so global pool uses desired size
        if let Ok(profile) = crate::optimization::SystemProfile::detect() {
            let inm = crate::optimization::calculate_inmemory_config(&profile, algorithm, false);
            if inm.rayon_threads > 0 {
                unsafe {
                    std::env::set_var("RAYON_NUM_THREADS", inm.rayon_threads.to_string());
                }
                info!(
                    "Auto-Optimize: setting RAYON_NUM_THREADS={} based on {}",
                    inm.rayon_threads, profile
                );
            }
        } else {
            warn!("Auto-Optimize: system detection failed; continuing without rayon thread tuning");
        }
    }

    if std::env::var("CHECK_ONLY").is_ok() {
        info!("Schema OK; exiting due to CHECK_ONLY");
        return Ok(());
    }

    // Decide execution mode (streaming vs in-memory)
    let c1 = get_person_count(&pool1, &table1).await?;
    let c2 = get_person_count(pool2_opt.as_ref().unwrap_or(&pool1), &table2).await?;
    let streaming_env = std::env::var("NAME_MATCHER_STREAMING")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    // Auto-streaming threshold: enable streaming automatically for large datasets
    const AUTO_STREAM_THRESHOLD: i64 = 100_000;

    let part_strategy =
        std::env::var("NAME_MATCHER_PARTITION").unwrap_or_else(|_| "last_initial".to_string());
    // Command-line flags can be placed after required args; scan whole argv
    let gpu_hash_flag = args.iter().any(|a| a == "--gpu-hash-join");
    let gpu_fuzzy_direct_flag = args.iter().any(|a| a == "--gpu-fuzzy-direct-hash");
    let direct_norm_flag = args.iter().any(|a| a == "--direct-fuzzy-normalization");
    let gpu_streams_flag: Option<u32> = args
        .windows(2)
        .find(|w| w[0] == "--gpu-streams")
        .and_then(|w| w.get(1))
        .and_then(|s| s.parse().ok());
    let gpu_buffer_pool_flag = args.iter().any(|a| a == "--gpu-buffer-pool");
    let no_gpu_buffer_pool_flag = args.iter().any(|a| a == "--no-gpu-buffer-pool");
    let gpu_pinned_host_flag = args.iter().any(|a| a == "--gpu-pinned-host");
    let gpu_fuzzy_metrics_flag = args.iter().any(|a| a == "--gpu-fuzzy-metrics");
    let gpu_fuzzy_force_flag = args.iter().any(|a| a == "--gpu-fuzzy-force");
    let gpu_lev_full_flag = args.iter().any(|a| a == "--gpu-levenshtein-full-scoring");

    let gpu_fuzzy_disable_flag = args.iter().any(|a| a == "--gpu-fuzzy-disable");
    let gpu_lev_prepass_flag = args.iter().any(|a| a == "--gpu-levenshtein-prepass");

    // GPU environment variables (needed for Advanced matching)
    let gpu_hash_env = std::env::var("NAME_MATCHER_GPU_HASH_JOIN")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let gpu_fuzzy_metrics_env = std::env::var("NAME_MATCHER_GPU_FUZZY_METRICS")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    // Advanced Matching CLI flags (opt-in)
    let adv_level_num: Option<u8> = args
        .windows(2)
        .find(|w| w[0] == "--advanced-level")
        .and_then(|w| w.get(1))
        .and_then(|s| s.parse().ok());
    let adv_code_col: Option<String> = args
        .windows(2)
        .find(|w| w[0] == "--advanced-code-col")
        .and_then(|w| w.get(1))
        .cloned();

    // Cascade Matching CLI flags
    let cascade_flag = args.iter().any(|a| a == "--cascade");
    let cascade_missing_mode = args
        .windows(2)
        .find(|w| w[0] == "--cascade-missing-columns")
        .and_then(|w| w.get(1))
        .map(|s| match s.to_lowercase().as_str() {
            "auto-skip" | "auto_skip" | "autoskip" => {
                crate::matching::cascade::MissingColumnMode::AutoSkip
            }
            "manual" | "manual-select" => crate::matching::cascade::MissingColumnMode::ManualSelect,
            "abort" | "abort-on-missing" => {
                crate::matching::cascade::MissingColumnMode::AbortOnMissing
            }
            _ => crate::matching::cascade::MissingColumnMode::AutoSkip,
        })
        .unwrap_or(crate::matching::cascade::MissingColumnMode::AutoSkip);
    let cascade_levels: Vec<u8> = args
        .windows(2)
        .find(|w| w[0] == "--levels")
        .and_then(|w| w.get(1))
        .map(|s| {
            s.split(',')
                .filter_map(|n| n.trim().parse::<u8>().ok())
                .filter(|&n| n >= 1 && n <= 11) // L12 excluded from cascade
                .collect()
        })
        .unwrap_or_default();

    // Advanced threshold: check --advanced-threshold flag first (override), then fall back to
    // NAME_MATCHER_HOUSEHOLD_THRESHOLD env var (same as Option 6), finally default to 0.95.
    // This ensures L12 and Option 6 use identical threshold values when run with the same configuration.
    let adv_threshold: f32 = if let Some(flag_val) = args
        .windows(2)
        .find(|w| w[0] == "--advanced-threshold")
        .and_then(|w| w.get(1))
        .and_then(|s| s.parse::<f32>().ok())
    {
        flag_val.clamp(0.5, 1.0)
    } else {
        // Fall back to same env var as Option 6 (lines 759-765)
        let hh_thr_env =
            std::env::var("NAME_MATCHER_HOUSEHOLD_THRESHOLD").unwrap_or_else(|_| "95".to_string());
        (|| {
            let s = hh_thr_env.trim();
            if let Some(p) = s.strip_suffix('%') {
                p.parse::<f32>().ok().map(|v| (v / 100.0).clamp(0.5, 1.0))
            } else if s.contains('.') {
                s.parse::<f32>().ok().map(|v| v.clamp(0.5, 1.0))
            } else {
                s.parse::<f32>().ok().map(|v| (v / 100.0).clamp(0.5, 1.0))
            }
        })()
        .unwrap_or(0.95)
    };

    // Cascade Matching mode (runs L1-L11 sequentially, L12 excluded)
    if cascade_flag {
        use crate::matching::cascade::{
            CascadeConfig, CascadePhase, CascadeProgress, GeoColumnStatus, run_cascade_inmemory,
            summary_output_path,
        };

        info!("Starting cascade run for levels L1-L11 (L12 excluded)");

        // Ensure CSV output
        if format != "csv" {
            warn!("Cascade Matching outputs CSV only; switching format to CSV");
            format = "csv".into();
            if !out_path.to_ascii_lowercase().ends_with(".csv") {
                out_path.push_str(".csv");
            }
        }

        // Detect geographic columns
        let all_cols_t1 = get_all_table_columns(&pool1, &cfg.database, &table1).await?;
        let all_cols_t2 =
            get_all_table_columns(pool2_opt.as_ref().unwrap_or(&pool1), &db2_name, &table2).await?;

        let geo_status = GeoColumnStatus {
            has_barangay_code: all_cols_t1.iter().any(|c| c == "barangay_code")
                || all_cols_t2.iter().any(|c| c == "barangay_code"),
            has_city_code: all_cols_t1.iter().any(|c| c == "city_code")
                || all_cols_t2.iter().any(|c| c == "city_code"),
        };

        info!("{}", geo_status.summary());

        // Warn about missing columns
        if !geo_status.has_barangay_code || !geo_status.has_city_code {
            let missing: Vec<&str> = [
                (!geo_status.has_barangay_code).then_some("barangay_code"),
                (!geo_status.has_city_code).then_some("city_code"),
            ]
            .into_iter()
            .flatten()
            .collect();

            let affected_levels: Vec<&str> = [
                (!geo_status.has_barangay_code).then_some("L4, L5, L6"),
                (!geo_status.has_city_code).then_some("L7, L8, L9"),
            ]
            .into_iter()
            .flatten()
            .collect();

            warn!(
                "Missing geographic columns: {}. Affected levels: {}",
                missing.join(", "),
                affected_levels.join(", ")
            );

            match cascade_missing_mode {
                crate::matching::cascade::MissingColumnMode::AutoSkip => {
                    info!("Auto-skip mode: Will skip levels with missing columns");
                }
                crate::matching::cascade::MissingColumnMode::ManualSelect => {
                    info!("Manual mode: Running only levels specified with --levels");
                }
                crate::matching::cascade::MissingColumnMode::AbortOnMissing => {
                    bail!(
                        "Cascade aborted: Missing required geographic columns. Use --cascade-missing-columns=auto-skip to skip unavailable levels."
                    );
                }
            }
        }

        // Load data
        info!("Loading Table1: {}", table1);
        let t1 = get_person_rows(&pool1, &table1).await?;
        info!("Loaded {} records from Table1", t1.len());

        info!("Loading Table2: {}", table2);
        let t2 = get_person_rows(pool2_opt.as_ref().unwrap_or(&pool1), &table2).await?;
        info!("Loaded {} records from Table2", t2.len());

        // Build cascade config
        let cascade_cfg = CascadeConfig {
            levels: cascade_levels.clone(),
            threshold: adv_threshold,
            allow_birthdate_swap: crate::matching::birthdate_matcher::allow_birthdate_swap(),
            missing_column_mode: cascade_missing_mode,
            base_output_path: out_path.clone(),
            exclusion_mode: crate::matching::cascade::CascadeExclusionMode::Exclusive,
            // GPU acceleration for cascade (L10-L11 fuzzy matching)
            // Currently defaults to CPU; GPU support to be added in Phase 2
            compute_backend: crate::matching::ComputeBackend::Cpu,
            gpu_device_id: None,
        };

        // Run cascade
        let result = run_cascade_inmemory(
            t1.as_slice(),
            t2.as_slice(),
            &cascade_cfg,
            &geo_status,
            |progress: CascadeProgress| match progress.phase {
                CascadePhase::Starting => {
                    info!(
                        "Starting Level {} of {}: {}",
                        progress.current_level, progress.total_levels, progress.level_description
                    );
                }
                CascadePhase::Running => {
                    info!(
                        "Running Level {}: {}",
                        progress.current_level, progress.level_description
                    );
                }
                CascadePhase::WritingOutput => {
                    info!("Writing output for Level {}", progress.current_level);
                }
                CascadePhase::Completed => {
                    info!("Level {} complete", progress.current_level);
                }
                CascadePhase::Skipped(ref reason) => {
                    warn!("Skipping Level {}: {}", progress.current_level, reason);
                }
            },
        );

        // Write summary
        let summary_path = summary_output_path(&out_path);
        match result.write_summary(&summary_path) {
            Ok(_) => info!("Cascade summary written to: {}", summary_path),
            Err(e) => error!("Failed to write cascade summary: {}", e),
        }

        // Final summary
        info!("=== Cascade Complete ===");
        info!("Total matches: {}", result.total_matches);
        info!("Duration: {:.2}s", result.total_duration_ms as f64 / 1000.0);
        for entry in &result.entries {
            match &entry.status {
                crate::matching::cascade::CascadeLevelStatus::Completed => {
                    info!(
                        "  L{}: {} matches -> {}",
                        entry.level,
                        entry.match_count,
                        entry.output_path.as_deref().unwrap_or("N/A")
                    );
                }
                crate::matching::cascade::CascadeLevelStatus::Skipped(reason) => {
                    info!("  L{}: SKIPPED ({})", entry.level, reason);
                }
                crate::matching::cascade::CascadeLevelStatus::Failed(err) => {
                    error!("  L{}: FAILED ({})", entry.level, err);
                }
            }
        }

        return Ok(());
    }

    if let Some(lvl_num) = adv_level_num {
        // Enforce CSV output for Advanced for now
        if format != "csv" {
            warn!("Advanced Matching currently outputs CSV only; switching format to CSV");
            format = "csv".into();
            if !out_path.to_ascii_lowercase().ends_with(".csv") {
                out_path.push_str(".csv");
            }
        }
        // Map level number to AdvLevel
        let level = match lvl_num {
            1 => AdvLevel::L1BirthdateFullMiddle,
            2 => AdvLevel::L2BirthdateMiddleInitial,
            3 => AdvLevel::L3BirthdateNoMiddle,
            4 => AdvLevel::L4BarangayFullMiddle,
            5 => AdvLevel::L5BarangayMiddleInitial,
            6 => AdvLevel::L6BarangayNoMiddle,
            7 => AdvLevel::L7CityFullMiddle,
            8 => AdvLevel::L8CityMiddleInitial,
            9 => AdvLevel::L9CityNoMiddle,
            10 => AdvLevel::L10FuzzyBirthdateFullMiddle,
            11 => AdvLevel::L11FuzzyBirthdateNoMiddle,
            12 => AdvLevel::L12HouseholdMatching,
            _ => {
                bail!("--advanced-level must be one of 1,2,3,4,5,6,7,8,9,10,11,12");
            }
        };
        // Geographic code columns are fixed: 'barangay_code' (L4-L6) and 'city_code' (L7-L9).
        // Any provided --advanced-code-col is deprecated and ignored.
        if let Some(ref c) = adv_code_col {
            if !c.trim().is_empty() {
                log::warn!(
                    "--advanced-code-col is deprecated and ignored; using fixed field names 'barangay_code'/'city_code'"
                );
            }
        }
        let cols = AdvColumns::default();
        // L12 household path: stream when heuristics say streaming, else in-memory
        if matches!(level, AdvLevel::L12HouseholdMatching) {
            let adv_cfg = AdvConfig {
                level,
                threshold: adv_threshold,
                cols: cols.clone(),
                allow_birthdate_swap:
                    name_matcher::matching::birthdate_matcher::allow_birthdate_swap(),
            };
            // Enable GPU fuzzy metrics for L12
            if streaming_env {
                let mut scfg_l12 = StreamingConfig {
                    ..Default::default()
                };
                scfg_l12.use_gpu_fuzzy_metrics =
                    auto_optimize || gpu_fuzzy_metrics_env || gpu_fuzzy_metrics_flag;
                let gpu_used_l12 = scfg_l12.use_gpu_fuzzy_metrics; // Capture before move
                let compute_backend = if scfg_l12.use_gpu_fuzzy_metrics {
                    "GPU"
                } else {
                    "CPU"
                };
                let gpu_features = if scfg_l12.use_gpu_fuzzy_metrics {
                    "GPU Fuzzy Metrics"
                } else {
                    ""
                };
                let gpu_model = if scfg_l12.use_gpu_fuzzy_metrics {
                    crate::matching::try_gpu_name()
                } else {
                    None
                };
                let mut w = HouseholdCsvWriter::create_with_meta(
                    &out_path,
                    compute_backend,
                    gpu_model.as_deref(),
                    gpu_features,
                )?;
                let emitted = stream_match_advanced_l12(&pool1, &table1, &table2, &adv_cfg, scfg_l12, |row| { w.write(row)?; Ok(()) }, |u| {
                    info!(
                        "[adv L12] Progress: {:.1}% | ETA: {}s | Mem used: {} MB | Avail: {} MB ({} / {}) | GPU {} MB free / {} MB total",
                        u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total, u.gpu_free_mb, u.gpu_total_mb
                    );
                }, None).await?;
                w.flush()?;
                info!(
                    "Advanced L12 completed (streaming). Wrote {} household matches to {}",
                    emitted, out_path
                );

                // Generate summary report
                let sum_path = if out_path.to_ascii_lowercase().ends_with(".csv") {
                    format!("{}_summary.csv", out_path.trim_end_matches(".csv"))
                } else {
                    format!("{}.summary.csv", out_path)
                };
                let gpu_used = gpu_used_l12;
                let run_end_utc = chrono::Utc::now();
                let duration_secs =
                    (run_end_utc - run_start_utc).num_milliseconds() as f64 / 1000.0;
                let summary = SummaryContext {
                    db_name: db_label.clone(),
                    table1: table1.clone(),
                    table2: table2.clone(),
                    total_table1: c1 as usize,
                    total_table2: c2 as usize,
                    matches_algo1: 0,
                    matches_algo2: 0,
                    matches_fuzzy: emitted,
                    overlap_count: 0,
                    unique_algo1: 0,
                    unique_algo2: 0,
                    fetch_time: std::time::Duration::from_secs(0),
                    match1_time: std::time::Duration::from_secs(0),
                    match2_time: std::time::Duration::from_secs(0),
                    export_time: std::time::Duration::from_secs(0),
                    mem_used_start_mb: 0,
                    mem_used_end_mb: 0,
                    started_utc: run_start_utc,
                    ended_utc: run_end_utc,
                    duration_secs,
                    exec_mode_algo1: None,
                    exec_mode_algo2: None,
                    exec_mode_fuzzy: Some(if gpu_used { "GPU".into() } else { "CPU".into() }),
                    algo_used: format!("Advanced {:?}", level),
                    gpu_used,
                    gpu_total_mb: 0,
                    gpu_free_mb_end: 0,
                    adv_level: Some(level),
                    adv_level_description: Some("L12: Household Matching".into()),
                };
                info!("Writing CSV summary to {}", sum_path);
                let _ = export_summary_csv(&sum_path, &summary);

                return Ok(());
            } else {
                let t1 = get_person_rows(&pool1, &table1).await?;
                let t2 = get_person_rows(pool2_opt.as_ref().unwrap_or(&pool1), &table2).await?;
                let backend = if (gpu_fuzzy_metrics_env || gpu_fuzzy_metrics_flag) || auto_optimize
                {
                    ComputeBackend::Gpu
                } else {
                    ComputeBackend::Cpu
                };
                let gpu_cfg = if backend == ComputeBackend::Gpu {
                    Some(GpuConfig {
                        device_id: None,
                        mem_budget_mb: 512,
                    })
                } else {
                    None
                };

                let compute_backend = if matches!(backend, ComputeBackend::Gpu) {
                    "GPU"
                } else {
                    "CPU"
                };
                let gpu_features = if matches!(backend, ComputeBackend::Gpu) {
                    "GPU Fuzzy Metrics"
                } else {
                    ""
                };
                let gpu_model = if matches!(backend, ComputeBackend::Gpu) {
                    crate::matching::try_gpu_name()
                } else {
                    None
                };

                // Progressive export for in-memory L12
                let mut emitted = 0usize;
                if out_path.to_ascii_lowercase().ends_with(".csv") {
                    let mut w = HouseholdCsvWriter::create_with_meta(
                        &out_path,
                        compute_backend,
                        gpu_model.as_deref(),
                        gpu_features,
                    )?;
                    emitted = match_households_inmemory_opt6_streaming(
                        &t1,
                        &t2,
                        MatchOptions {
                            backend,
                            gpu: gpu_cfg,
                            progress: ProgressConfig::default(),
                            allow_birthdate_swap: false,
                        },
                        adv_threshold,
                        |row: &HouseholdAggRow| {
                            w.write(row)?;
                            Ok(())
                        },
                        |u: ProgressUpdate| {
                            info!(
                                "[adv L12 inmem] Progress: {:.1}% | ETA: {}s | Mem used: {} MB | Avail: {} MB ({} / {}) | GPU {} MB free / {} MB total",
                                u.percent,
                                u.eta_secs,
                                u.mem_used_mb,
                                u.mem_avail_mb,
                                u.processed,
                                u.total,
                                u.gpu_free_mb,
                                u.gpu_total_mb
                            );
                        },
                    )?;
                    w.flush()?;
                } else {
                    // Fallback to buffered path for non-CSV exports (e.g., XLSX).
                    let rows = match_households_gpu_inmemory_opt6(
                        &t1,
                        &t2,
                        MatchOptions {
                            backend,
                            gpu: gpu_cfg,
                            progress: ProgressConfig::default(),
                            allow_birthdate_swap: false,
                        },
                        adv_threshold,
                        |u| {
                            info!(
                                "[adv L12 inmem] Progress: {:.1}% | ETA: {}s | Mem used: {} MB | Avail: {} MB ({} / {}) | GPU {} MB free / {} MB total",
                                u.percent,
                                u.eta_secs,
                                u.mem_used_mb,
                                u.mem_avail_mb,
                                u.processed,
                                u.total,
                                u.gpu_free_mb,
                                u.gpu_total_mb
                            );
                        },
                    );
                    let mut w = HouseholdCsvWriter::create_with_meta(
                        &out_path,
                        compute_backend,
                        gpu_model.as_deref(),
                        gpu_features,
                    )?;
                    for r in &rows {
                        w.write(r)?;
                    }
                    w.flush()?;
                    emitted = rows.len();
                }
                info!(
                    "Advanced L12 completed (in-memory, backend={:?}). Wrote {} household matches to {}",
                    backend, emitted, out_path
                );

                // Generate summary report
                let sum_path = if out_path.to_ascii_lowercase().ends_with(".csv") {
                    format!("{}_summary.csv", out_path.trim_end_matches(".csv"))
                } else {
                    format!("{}.summary.csv", out_path)
                };
                let gpu_used = matches!(backend, ComputeBackend::Gpu);
                let run_end_utc = chrono::Utc::now();
                let duration_secs =
                    (run_end_utc - run_start_utc).num_milliseconds() as f64 / 1000.0;
                let summary = SummaryContext {
                    db_name: db_label.clone(),
                    table1: table1.clone(),
                    table2: table2.clone(),
                    total_table1: c1 as usize,
                    total_table2: c2 as usize,
                    matches_algo1: 0,
                    matches_algo2: 0,
                    matches_fuzzy: emitted,
                    overlap_count: 0,
                    unique_algo1: 0,
                    unique_algo2: 0,
                    fetch_time: std::time::Duration::from_secs(0),
                    match1_time: std::time::Duration::from_secs(0),
                    match2_time: std::time::Duration::from_secs(0),
                    export_time: std::time::Duration::from_secs(0),
                    mem_used_start_mb: 0,
                    mem_used_end_mb: 0,
                    started_utc: run_start_utc,
                    ended_utc: run_end_utc,
                    duration_secs,
                    exec_mode_algo1: None,
                    exec_mode_algo2: None,
                    exec_mode_fuzzy: Some(if gpu_used { "GPU".into() } else { "CPU".into() }),
                    algo_used: format!("Advanced {:?}", level),
                    gpu_used,
                    gpu_total_mb: 0,
                    gpu_free_mb_end: 0,
                    adv_level: Some(level),
                    adv_level_description: Some("L12: Household Matching".into()),
                };
                info!("Writing CSV summary to {}", sum_path);
                let _ = export_summary_csv(&sum_path, &summary);

                return Ok(());
            }
        }
        let adv_cfg = AdvConfig {
            level,
            threshold: adv_threshold,
            cols,
            allow_birthdate_swap: name_matcher::matching::birthdate_matcher::allow_birthdate_swap(),
        };
        // Build streaming config (reuse env heuristics)
        let mut scfg = StreamingConfig {
            ..Default::default()
        };
        // Enable GPU for Advanced matching based on level type and CLI flags
        match level {
            // L1-L9: Exact matching - use GPU hash join
            AdvLevel::L1BirthdateFullMiddle
            | AdvLevel::L2BirthdateMiddleInitial
            | AdvLevel::L3BirthdateNoMiddle
            | AdvLevel::L4BarangayFullMiddle
            | AdvLevel::L5BarangayMiddleInitial
            | AdvLevel::L6BarangayNoMiddle
            | AdvLevel::L7CityFullMiddle
            | AdvLevel::L8CityMiddleInitial
            | AdvLevel::L9CityNoMiddle => {
                scfg.use_gpu_hash_join = auto_optimize || gpu_hash_env || gpu_hash_flag;
                scfg.use_gpu_build_hash = scfg.use_gpu_hash_join;
                scfg.use_gpu_probe_hash = scfg.use_gpu_hash_join;
            }
            // L10-L11: Fuzzy matching - use GPU fuzzy metrics
            AdvLevel::L10FuzzyBirthdateFullMiddle | AdvLevel::L11FuzzyBirthdateNoMiddle => {
                scfg.use_gpu_fuzzy_metrics =
                    auto_optimize || gpu_fuzzy_metrics_env || gpu_fuzzy_metrics_flag;
            }
            // L12: Handled separately above
            _ => {}
        }
        // Capture GPU usage flags before scfg is moved
        let gpu_hash_used = scfg.use_gpu_hash_join;
        let gpu_fuzzy_used = scfg.use_gpu_fuzzy_metrics;
        scfg.direct_use_fuzzy_normalization = false;

        // Discover extra fields for Table 2
        let standard = [
            "id",
            "uuid",
            "first_name",
            "middle_name",
            "last_name",
            "birthdate",
            "hh_id",
        ];
        let extra_field_names: Vec<String> =
            get_all_table_columns(pool2_opt.as_ref().unwrap_or(&pool1), &db2_name, &table2)
                .await?
                .into_iter()
                .filter(|c| !standard.contains(&c.as_str()))
                .collect();
        // Compute export metadata
        let compute_backend = if scfg.use_gpu_hash_join || scfg.use_gpu_fuzzy_metrics {
            "GPU"
        } else {
            "CPU"
        };
        let gpu_features = if scfg.use_gpu_hash_join {
            "GPU Hash Join"
        } else if scfg.use_gpu_fuzzy_metrics {
            "GPU Fuzzy Metrics"
        } else {
            ""
        };
        let gpu_model = if compute_backend == "GPU" {
            crate::matching::try_gpu_name()
        } else {
            None
        };
        // Debug logging for metadata determination
        log::info!("[METADATA-DEBUG] Advanced Level: {:?}", level);
        log::info!(
            "[METADATA-DEBUG] gpu_hash_env: {}, gpu_hash_flag: {}",
            gpu_hash_env,
            gpu_hash_flag
        );
        log::info!(
            "[METADATA-DEBUG] gpu_fuzzy_metrics_env: {}, gpu_fuzzy_metrics_flag: {}",
            gpu_fuzzy_metrics_env,
            gpu_fuzzy_metrics_flag
        );
        log::info!(
            "[METADATA-DEBUG] scfg.use_gpu_hash_join: {}",
            scfg.use_gpu_hash_join
        );
        log::info!(
            "[METADATA-DEBUG] scfg.use_gpu_fuzzy_metrics: {}",
            scfg.use_gpu_fuzzy_metrics
        );
        log::info!("[METADATA-DEBUG] compute_backend: '{}'", compute_backend);
        log::info!("[METADATA-DEBUG] gpu_features: '{}'", gpu_features);
        log::info!("[METADATA-DEBUG] gpu_model: {:?}", gpu_model);
        // Use Advanced CSV writer (includes metadata columns)
        let mut w = AdvCsvStreamWriter::create_with_extra_fields(
            &out_path,
            extra_field_names.clone(),
            compute_backend,
            gpu_model.as_deref(),
            gpu_features,
        )?;
        let mut kept = 0usize;
        let flush_every = 2000usize;
        if let Some(pool2) = pool2_opt.as_ref() {
            stream_match_advanced_dual(&pool1, pool2, &table1, &table2, &adv_cfg, scfg, |p| {
                if matches!(level, AdvLevel::L10FuzzyBirthdateFullMiddle | AdvLevel::L11FuzzyBirthdateNoMiddle) && (p.confidence / 100.0) < adv_threshold { return Ok(()); }
                kept += 1; w.write(p, level)?; if kept % flush_every == 0 { w.flush_partial()?; }
                Ok(())
            }, |u| {
                info!(
                    "[adv] Progress: {:.1}% | ETA: {}s | Mem used: {} MB | Avail: {} MB ({} / {}) | GPU {} MB free / {} MB total",
                    u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total, u.gpu_free_mb, u.gpu_total_mb
                );
            }, None).await?;
        } else {
            stream_match_advanced(&pool1, &table1, &table2, &adv_cfg, scfg, |p| {
                if matches!(level, AdvLevel::L10FuzzyBirthdateFullMiddle | AdvLevel::L11FuzzyBirthdateNoMiddle) && (p.confidence / 100.0) < adv_threshold { return Ok(()); }
                kept += 1; w.write(p, level)?; if kept % flush_every == 0 { w.flush_partial()?; }
                Ok(())
            }, |u| {
                info!(
                    "[adv] Progress: {:.1}% | ETA: {}s | Mem used: {} MB | Avail: {} MB ({} / {}) | GPU {} MB free / {} MB total",
                    u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total, u.gpu_free_mb, u.gpu_total_mb
                );
            }, None).await?;
        }
        w.flush()?;
        info!(
            "Advanced level {:?} completed. Exported {} records to {}",
            level, kept, out_path
        );

        // Generate summary report
        let sum_path = if out_path.to_ascii_lowercase().ends_with(".csv") {
            format!("{}_summary.csv", out_path.trim_end_matches(".csv"))
        } else {
            format!("{}.summary.csv", out_path)
        };
        let gpu_used = match level {
            AdvLevel::L1BirthdateFullMiddle
            | AdvLevel::L2BirthdateMiddleInitial
            | AdvLevel::L3BirthdateNoMiddle
            | AdvLevel::L4BarangayFullMiddle
            | AdvLevel::L5BarangayMiddleInitial
            | AdvLevel::L6BarangayNoMiddle
            | AdvLevel::L7CityFullMiddle
            | AdvLevel::L8CityMiddleInitial
            | AdvLevel::L9CityNoMiddle => gpu_hash_used,
            AdvLevel::L10FuzzyBirthdateFullMiddle | AdvLevel::L11FuzzyBirthdateNoMiddle => {
                gpu_fuzzy_used
            }
            _ => false,
        };
        let run_end_utc = chrono::Utc::now();
        let duration_secs = (run_end_utc - run_start_utc).num_milliseconds() as f64 / 1000.0;
        let level_desc: String = match level {
            AdvLevel::L1BirthdateFullMiddle => "L1: Birthdate + Full Middle Name".into(),
            AdvLevel::L2BirthdateMiddleInitial => "L2: Birthdate + Middle Initial".into(),
            AdvLevel::L3BirthdateNoMiddle => "L3: Birthdate (No Middle)".into(),
            AdvLevel::L4BarangayFullMiddle => "L4: Barangay + Full Middle Name".into(),
            AdvLevel::L5BarangayMiddleInitial => "L5: Barangay + Middle Initial".into(),
            AdvLevel::L6BarangayNoMiddle => "L6: Barangay (No Middle)".into(),
            AdvLevel::L7CityFullMiddle => "L7: City + Full Middle Name".into(),
            AdvLevel::L8CityMiddleInitial => "L8: City + Middle Initial".into(),
            AdvLevel::L9CityNoMiddle => "L9: City (No Middle)".into(),
            AdvLevel::L10FuzzyBirthdateFullMiddle => "L10: Fuzzy + Birthdate + Full Middle".into(),
            AdvLevel::L11FuzzyBirthdateNoMiddle => "L11: Fuzzy + Birthdate (No Middle)".into(),
            AdvLevel::L12HouseholdMatching => "L12: Household Matching".into(),
        };
        let summary = SummaryContext {
            db_name: db_label.clone(),
            table1: table1.clone(),
            table2: table2.clone(),
            total_table1: c1 as usize,
            total_table2: c2 as usize,
            matches_algo1: 0,
            matches_algo2: 0,
            matches_fuzzy: kept,
            overlap_count: 0,
            unique_algo1: 0,
            unique_algo2: 0,
            fetch_time: std::time::Duration::from_secs(0),
            match1_time: std::time::Duration::from_secs(0),
            match2_time: std::time::Duration::from_secs(0),
            export_time: std::time::Duration::from_secs(0),
            mem_used_start_mb: 0,
            mem_used_end_mb: 0,
            started_utc: run_start_utc,
            ended_utc: run_end_utc,
            duration_secs,
            exec_mode_algo1: None,
            exec_mode_algo2: None,
            exec_mode_fuzzy: Some(if gpu_used { "GPU".into() } else { "CPU".into() }),
            algo_used: format!("Advanced {:?}", level),
            gpu_used,
            gpu_total_mb: 0,
            gpu_free_mb_end: 0,
            adv_level: Some(level),
            adv_level_description: Some(level_desc),
        };
        info!("Writing CSV summary to {}", sum_path);
        let _ = export_summary_csv(&sum_path, &summary);

        return Ok(());
    }

    // Auto-enable streaming for large datasets (> 100K total records)
    let auto_stream_triggered = !streaming_env && (c1 + c2 > AUTO_STREAM_THRESHOLD);
    let use_streaming = streaming_env || auto_stream_triggered;

    if auto_stream_triggered {
        info!(
            "Auto-enabling streaming mode: {} total records exceeds threshold of {}",
            c1 + c2,
            AUTO_STREAM_THRESHOLD
        );
    }

    use crate::metrics::memory_stats_mb;
    let cfgp = ProgressConfig {
        update_every: 1000,
        ..Default::default()
    };

    if use_streaming {
        let gpu_hash_env = std::env::var("NAME_MATCHER_GPU_HASH_JOIN")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let gpu_fuzzy_direct_env = std::env::var("NAME_MATCHER_GPU_FUZZY_DIRECT_HASH")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let direct_norm_env = std::env::var("NAME_MATCHER_DIRECT_FUZZY_NORMALIZATION")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let gpu_streams_env: Option<u32> = std::env::var("NAME_MATCHER_GPU_STREAMS")
            .ok()
            .and_then(|s| s.parse().ok());
        let gpu_buffer_pool_env: Option<bool> = std::env::var("NAME_MATCHER_GPU_BUFFER_POOL")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"));
        let gpu_pinned_host_env: Option<bool> = std::env::var("NAME_MATCHER_GPU_PINNED_HOST")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"));
        let gpu_fuzzy_metrics_env: Option<bool> = std::env::var("NAME_MATCHER_GPU_FUZZY_METRICS")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"));

        info!("Streaming mode enabled ({} + {} rows).", c1, c2);
        let mut s_cfg = StreamingConfig::default();
        s_cfg.checkpoint_path = Some(format!("{}.nmckpt", out_path));
        // Apply global GPU fuzzy overrides (heuristic force/disable)
        {
            // Auto-Optimize streaming config if requested
            if auto_optimize {
                match crate::optimization::SystemProfile::detect() {
                    Ok(profile) => {
                        let before = s_cfg.clone();
                        let opt = crate::optimization::calculate_streaming_config(
                            &profile, algorithm, true,
                        );
                        // Apply optimized fields (override legacy flags if different)
                        if before.batch_size != opt.batch_size {
                            info!(
                                "Auto-Optimize: batch_size {} -> {}",
                                before.batch_size, opt.batch_size
                            );
                            s_cfg.batch_size = opt.batch_size;
                        }
                        if before.memory_soft_min_mb != opt.memory_soft_min_mb {
                            info!(
                                "Auto-Optimize: memory_soft_min_mb {} -> {}",
                                before.memory_soft_min_mb, opt.memory_soft_min_mb
                            );
                            s_cfg.memory_soft_min_mb = opt.memory_soft_min_mb;
                        }
                        if before.prefetch_pool_size != opt.prefetch_pool_size {
                            info!(
                                "Auto-Optimize: prefetch_pool_size {} -> {}",
                                before.prefetch_pool_size, opt.prefetch_pool_size
                            );
                            s_cfg.prefetch_pool_size = opt.prefetch_pool_size;
                        }
                        s_cfg.async_prefetch = opt.async_prefetch;
                        s_cfg.parallel_normalize = opt.parallel_normalize;
                        s_cfg.gpu_probe_batch_mb = opt.gpu_probe_batch_mb;
                        s_cfg.enable_dynamic_gpu_tuning = opt.enable_dynamic_gpu_tuning;
                        // GPU features per algorithm
                        s_cfg.use_gpu_hash_join = opt.use_gpu_hash_join;
                        s_cfg.use_gpu_build_hash = opt.use_gpu_hash_join;
                        s_cfg.use_gpu_probe_hash = opt.use_gpu_hash_join;
                        s_cfg.use_gpu_fuzzy_metrics = opt.use_gpu_fuzzy_metrics;
                        info!("Auto-Optimize applied: {}", profile);
                    }
                    Err(e) => {
                        warn!(
                            "Auto-Optimize: detection failed ({}); proceeding with legacy flags",
                            e
                        );
                    }
                }
            }

            let gpu_fuzzy_force_env = std::env::var("NAME_MATCHER_GPU_FUZZY_FORCE")
                .ok()
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            let gpu_fuzzy_disable_env = std::env::var("NAME_MATCHER_GPU_FUZZY_DISABLE")
                .ok()
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            crate::matching::set_gpu_fuzzy_force(gpu_fuzzy_force_env || gpu_fuzzy_force_flag);
            crate::matching::set_gpu_fuzzy_disable(gpu_fuzzy_disable_env || gpu_fuzzy_disable_flag);
        }
        s_cfg.use_gpu_hash_join = if matches!(
            algorithm,
            MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle
        ) {
            false
        } else {
            gpu_hash_env || gpu_hash_flag
        }; // legacy master switch
        // Granular GPU controls default to the legacy switch for backward compatibility
        s_cfg.use_gpu_build_hash = s_cfg.use_gpu_hash_join;
        s_cfg.use_gpu_probe_hash = s_cfg.use_gpu_hash_join;
        s_cfg.use_gpu_fuzzy_direct_hash = gpu_fuzzy_direct_env || gpu_fuzzy_direct_flag;
        s_cfg.direct_use_fuzzy_normalization = direct_norm_env || direct_norm_flag;
        if let Some(n) = gpu_streams_flag.or(gpu_streams_env) {
            s_cfg.gpu_streams = n;
        }
        if let Some(b) = gpu_buffer_pool_env {
            s_cfg.gpu_buffer_pool = b;
        }
        if gpu_buffer_pool_flag {
            s_cfg.gpu_buffer_pool = true;
        }
        if no_gpu_buffer_pool_flag {
            s_cfg.gpu_buffer_pool = false;
        }
        if let Some(b) = gpu_pinned_host_env {
            s_cfg.gpu_use_pinned_host = b;
        }
        if gpu_pinned_host_flag {
            s_cfg.gpu_use_pinned_host = true;
        }
        if let Some(b) = gpu_fuzzy_metrics_env {
            s_cfg.use_gpu_fuzzy_metrics = b;
        }
        if gpu_fuzzy_metrics_flag {
            s_cfg.use_gpu_fuzzy_metrics = true;
        }

        // SPECIAL CASE: Streaming path for original Option 5/6 (Household aggregation)
        // The generic streaming join emits person-pair rows and does not handle household aggregation.
        // To maintain parity with the in-memory Option 5/6 implementation and avoid zero-results,
        // route Option 5/6 to their dedicated household streaming implementations which mirror
        // the in-memory semantics exactly (Option 5: T1->T2, denom=T1 uuid size; Option 6: T2->T1, denom=T2 hh size).
        if matches!(
            algorithm,
            MatchingAlgorithm::HouseholdGpu | MatchingAlgorithm::HouseholdGpuOpt6
        ) {
            // Build household fuzzy threshold from env (same logic as in-memory path)
            let hh_thr_env = std::env::var("NAME_MATCHER_HOUSEHOLD_THRESHOLD")
                .unwrap_or_else(|_| "95".to_string());
            let hh_min_conf = (|| {
                let s = hh_thr_env.trim();
                if let Some(p) = s.strip_suffix('%') {
                    p.parse::<f32>().ok().map(|v| (v / 100.0).clamp(0.5, 1.0))
                } else if s.contains('.') {
                    s.parse::<f32>().ok().map(|v| v.clamp(0.5, 1.0))
                } else {
                    s.parse::<f32>().ok().map(|v| (v / 100.0).clamp(0.5, 1.0))
                }
            })()
            .unwrap_or(0.95);

            // Household streaming uses fuzzy person-level metrics; decide backend label from current s_cfg
            let compute_backend = if s_cfg.use_gpu_fuzzy_metrics {
                "GPU"
            } else {
                "CPU"
            };
            let gpu_features = if s_cfg.use_gpu_fuzzy_metrics {
                "GPU Fuzzy Metrics"
            } else {
                ""
            };
            let gpu_model = if s_cfg.use_gpu_fuzzy_metrics {
                crate::matching::try_gpu_name()
            } else {
                None
            };

            // Household output writer (CSV only in streaming mode)
            let mut w = HouseholdCsvWriter::create_with_meta(
                &out_path,
                compute_backend,
                gpu_model.as_deref(),
                gpu_features,
            )?;

            // Single-DB vs Dual-DB routing
            let allow_birthdate_swap =
                name_matcher::matching::birthdate_matcher::allow_birthdate_swap();
            let emitted = if let Some(ref pool2) = pool2_opt {
                if matches!(algorithm, MatchingAlgorithm::HouseholdGpuOpt6) {
                    crate::matching::stream_match_option6_dual(&pool1, pool2, &table1, &table2, hh_min_conf, allow_birthdate_swap, s_cfg.clone(), |row| { w.write(row)?; Ok(()) }, |u| {
                        info!(
                            "[opt6 stream dual] {:.1}% | ETA: {}s | Mem used: {} MB | Avail: {} MB ({} / {}) | GPU {} MB free / {} MB total",
                            u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total, u.gpu_free_mb, u.gpu_total_mb
                        );
                    }, None).await?
                } else {
                    crate::matching::stream_match_option5_dual(&pool1, pool2, &table1, &table2, hh_min_conf, s_cfg.clone(), |row| { w.write(row)?; Ok(()) }, |u| {
                        info!(
                            "[opt5 stream dual] {:.1}% | ETA: {}s | Mem used: {} MB | Avail: {} MB ({} / {}) | GPU {} MB free / {} MB total",
                            u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total, u.gpu_free_mb, u.gpu_total_mb
                        );
                    }, None).await?
                }
            } else {
                if matches!(algorithm, MatchingAlgorithm::HouseholdGpuOpt6) {
                    crate::matching::stream_match_option6(&pool1, &table1, &table2, hh_min_conf, allow_birthdate_swap, s_cfg.clone(), |row| { w.write(row)?; Ok(()) }, |u| {
                        info!(
                            "[opt6 stream] {:.1}% | ETA: {}s | Mem used: {} MB | Avail: {} MB ({} / {}) | GPU {} MB free / {} MB total",
                            u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total, u.gpu_free_mb, u.gpu_total_mb
                        );
                    }, None).await?
                } else {
                    crate::matching::stream_match_option5(&pool1, &table1, &table2, hh_min_conf, s_cfg.clone(), |row| { w.write(row)?; Ok(()) }, |u| {
                        info!(
                            "[opt5 stream] {:.1}% | ETA: {}s | Mem used: {} MB | Avail: {} MB ({} / {}) | GPU {} MB free / {} MB total",
                            u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total, u.gpu_free_mb, u.gpu_total_mb
                        );
                    }, None).await?
                }
            };
            w.flush()?;

            // Sidecar CSV summary aligned with legacy Option 5/6 summary
            let run_end_utc = chrono::Utc::now();
            let duration_secs = (run_end_utc - run_start_utc).num_milliseconds() as f64 / 1000.0;
            let sum_path = if out_path.to_ascii_lowercase().ends_with(".csv") {
                format!("{}_summary.csv", out_path.trim_end_matches(".csv"))
            } else {
                format!("{}.summary.csv", out_path)
            };
            let summary = SummaryContext {
                db_name: db_label.clone(),
                table1: table1.clone(),
                table2: table2.clone(),
                total_table1: c1 as usize,
                total_table2: c2 as usize,
                matches_algo1: 0,
                matches_algo2: 0,
                matches_fuzzy: emitted,
                overlap_count: 0,
                unique_algo1: 0,
                unique_algo2: 0,
                fetch_time: std::time::Duration::from_secs(0),
                match1_time: std::time::Duration::from_secs(0),
                match2_time: std::time::Duration::from_secs(0),
                export_time: std::time::Duration::from_secs(0),
                mem_used_start_mb: 0,
                mem_used_end_mb: 0,
                started_utc: run_start_utc,
                ended_utc: run_end_utc,
                duration_secs,
                exec_mode_algo1: None,
                exec_mode_algo2: None,
                exec_mode_fuzzy: Some(compute_backend.into()),
                algo_used: algo_label_summary(algorithm).to_string(),
                gpu_used: s_cfg.use_gpu_fuzzy_metrics,
                gpu_total_mb: 0,
                gpu_free_mb_end: 0,
                adv_level: None,
                adv_level_description: None,
            };
            info!("Writing CSV summary to {}", sum_path);
            let _ = export_summary_csv(&sum_path, &summary);

            info!(
                "Household streaming (Option {}) completed. Wrote {} household matches to {}",
                if matches!(algorithm, MatchingAlgorithm::HouseholdGpu) {
                    5
                } else {
                    6
                },
                emitted,
                out_path
            );
            return Ok(());
        }

        // Apply normalization alignment globally for this run (affects in-memory too if used)
        crate::matching::set_direct_normalization_fuzzy(s_cfg.direct_use_fuzzy_normalization);

        // Pre-scan Table 2 extra field names for streaming writers (so headers include dynamic columns)
        let standard_cols = [
            "id",
            "uuid",
            "first_name",
            "middle_name",
            "last_name",
            "birthdate",
            "hh_id",
        ];
        let extra_field_names: Vec<String> = {
            let pool_for_t2 = pool2_opt.as_ref().unwrap_or(&pool1);
            let db_for_t2 = db2_name.clone();
            match get_all_table_columns(pool_for_t2, &db_for_t2, &table2).await {
                Ok(cols) => cols
                    .into_iter()
                    .filter(|c| !standard_cols.contains(&c.as_str()))
                    .collect(),
                Err(e) => {
                    log::warn!(
                        "Could not discover extra columns for {}.{}: {}",
                        db_for_t2,
                        table2,
                        e
                    );
                    Vec::new()
                }
            }
        };

        if format == "csv" || format == "both" {
            info!("Streaming CSV export to {} using {:?}", out_path, algorithm);
            // Read fuzzy threshold from env (supports 0.95, 95, or 95%)
            let fuzzy_thr_env =
                std::env::var("NAME_MATCHER_FUZZY_THRESHOLD").unwrap_or_else(|_| "95".to_string());
            let fuzzy_min_conf = (|| {
                let s = fuzzy_thr_env.trim();
                if let Some(p) = s.strip_suffix('%') {
                    p.parse::<f32>().ok().map(|v| (v / 100.0).clamp(0.6, 1.0))
                } else if s.contains('.') {
                    s.parse::<f32>().ok().map(|v| {
                        if v > 1.0 {
                            (v / 100.0).clamp(0.6, 1.0)
                        } else {
                            v.clamp(0.6, 1.0)
                        }
                    })
                } else {
                    s.parse::<f32>().ok().map(|v| (v / 100.0).clamp(0.6, 1.0))
                }
            })()
            .unwrap_or(0.95);
            let mut writer = CsvStreamWriter::create_with_extra_fields(
                &out_path,
                algorithm,
                fuzzy_min_conf,
                extra_field_names.clone(),
            )?;
            let t_match = std::time::Instant::now();
            let flush_every = s_cfg.flush_every;
            let mut n_flushed = 0usize;
            // Summary tracking for CSV (Options 3–6)
            // using global run_start_utc captured earlier
            use std::sync::Arc;
            use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
            let gpu_used_flag = Arc::new(AtomicBool::new(false));
            let gpu_total_mb = Arc::new(AtomicU64::new(0));
            let gpu_free_mb = Arc::new(AtomicU64::new(0));

            if let Some(p2) = &pool2_opt {
                if s_cfg.use_gpu_hash_join {
                    warn!(
                        "GPU hash-join requested but cross-database GPU path is not yet available; proceeding with CPU for dual-db"
                    );
                }
                let count = {
                    let use_new_engine = cfg!(feature = "new_engine")
                        && std::env::var("NAME_MATCHER_NEW_ENGINE")
                            .ok()
                            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                            .unwrap_or(false);
                    if use_new_engine {
                        #[cfg(feature = "new_engine")]
                        {
                            let used_c = gpu_used_flag.clone();
                            let tot_c = gpu_total_mb.clone();
                            let free_c = gpu_free_mb.clone();
                            crate::engine::db_pipeline::db_pipeline::stream_new_engine_dual(&pool1, p2, &table1, &table2, algorithm, |pair| {
                                writer.write(pair)?; n_flushed += 1; if n_flushed % flush_every == 0 { writer.flush_partial()?; } Ok(())
                            }, s_cfg.clone(), move |u: ProgressUpdate| {
                                if u.gpu_active { used_c.store(true, Ordering::Relaxed); }
                                tot_c.store(u.gpu_total_mb as u64, Ordering::Relaxed);
                                free_c.store(u.gpu_free_mb as u64, Ordering::Relaxed);
                                info!("[dual:new] {:.1}% | ETA: {}s | Mem used: {} MB | Avail: {} MB ({} / {})",
                                    u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total);
                            }, None).await?
                        }
                        #[cfg(not(feature = "new_engine"))]
                        {
                            unreachable!()
                        }
                    } else {
                        let used_c = gpu_used_flag.clone();
                        let tot_c = gpu_total_mb.clone();
                        let free_c = gpu_free_mb.clone();
                        stream_match_csv_dual(&pool1, p2, &table1, &table2, algorithm, |pair| {
                            writer.write(pair)?; n_flushed += 1; if n_flushed % flush_every == 0 { writer.flush_partial()?; } Ok(())
                        }, s_cfg.clone(), move |u: ProgressUpdate| {
                            if u.gpu_active { used_c.store(true, Ordering::Relaxed); }
                            tot_c.store(u.gpu_total_mb as u64, Ordering::Relaxed);
                            free_c.store(u.gpu_free_mb as u64, Ordering::Relaxed);
                            info!("[dual] {:.1}% | ETA: {}s | Mem used: {} MB | Avail: {} MB ({} / {})",
                                u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total);
                        }, None).await?
                    }
                };
                writer.flush()?;
                info!(
                    "Wrote {} matches (streaming, dual-db) in {:?}",
                    count,
                    t_match.elapsed()
                );
                {
                    let run_end_utc = chrono::Utc::now();
                    let duration_secs =
                        (run_end_utc - run_start_utc).num_milliseconds() as f64 / 1000.0;
                    let gpu_any = gpu_used_flag.load(Ordering::Relaxed);
                    let sum_path = if out_path.to_ascii_lowercase().ends_with(".csv") {
                        format!("{}_summary.csv", out_path.trim_end_matches(".csv"))
                    } else {
                        format!("{}.summary.csv", out_path)
                    };
                    let fuzzy_like = matches!(
                        algorithm,
                        MatchingAlgorithm::Fuzzy
                            | MatchingAlgorithm::FuzzyNoMiddle
                            | MatchingAlgorithm::HouseholdGpu
                            | MatchingAlgorithm::HouseholdGpuOpt6
                    );
                    let summary = SummaryContext {
                        db_name: db_label.clone(),
                        table1: table1.clone(),
                        table2: table2.clone(),
                        total_table1: c1 as usize,
                        total_table2: c2 as usize,
                        matches_algo1: if fuzzy_like { 0 } else { count },
                        matches_algo2: 0,
                        matches_fuzzy: if fuzzy_like { count } else { 0 },
                        overlap_count: 0,
                        unique_algo1: 0,
                        unique_algo2: 0,
                        fetch_time: std::time::Duration::from_secs(0),
                        match1_time: t_match.elapsed(),
                        match2_time: std::time::Duration::from_secs(0),
                        export_time: std::time::Duration::from_secs(0),
                        mem_used_start_mb: 0,
                        mem_used_end_mb: 0,
                        started_utc: run_start_utc,
                        ended_utc: run_end_utc,
                        duration_secs,
                        exec_mode_algo1: None,
                        exec_mode_algo2: None,
                        exec_mode_fuzzy: Some(if gpu_any { "GPU".into() } else { "CPU".into() }),
                        algo_used: algo_label_summary(algorithm).to_string(),
                        gpu_used: gpu_any,
                        gpu_total_mb: gpu_total_mb.load(Ordering::Relaxed),
                        gpu_free_mb_end: gpu_free_mb.load(Ordering::Relaxed),
                        adv_level: None,
                        adv_level_description: None,
                    };
                    info!("Writing CSV summary to {}", sum_path);
                    let _ = export_summary_csv(&sum_path, &summary);
                }
            } else {
                // If GPU hash-join is enabled and single DB, use the GPU-accelerated streaming path
                if s_cfg.use_gpu_hash_join {
                    info!("GPU hash-join requested; using single-DB accelerated path");
                    let used_c = gpu_used_flag.clone();
                    let tot_c = gpu_total_mb.clone();
                    let free_c = gpu_free_mb.clone();
                    let count = crate::matching::stream_match_csv(&pool1, &table1, &table2, algorithm, |pair| {
                        writer.write(pair)?; n_flushed += 1; if n_flushed % flush_every == 0 { writer.flush_partial()?; } Ok(())
                    }, s_cfg.clone(), move |u: ProgressUpdate| {
                        if u.gpu_active { used_c.store(true, Ordering::Relaxed); }
                        tot_c.store(u.gpu_total_mb as u64, Ordering::Relaxed);
                        free_c.store(u.gpu_free_mb as u64, Ordering::Relaxed);
                        info!("[gpu] {:.1}% | ETA: {}s | Mem used: {} MB | Avail: {} MB ({} / {})",
                            u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total);
                    }, None).await?;
                    writer.flush()?;
                    info!(
                        "Wrote {} matches (streaming, gpu-accelerated) in {:?}",
                        count,
                        t_match.elapsed()
                    );
                    {
                        let run_end_utc = chrono::Utc::now();
                        let duration_secs =
                            (run_end_utc - run_start_utc).num_milliseconds() as f64 / 1000.0;
                        let gpu_any = gpu_used_flag.load(Ordering::Relaxed);
                        let sum_path = if out_path.to_ascii_lowercase().ends_with(".csv") {
                            format!("{}_summary.csv", out_path.trim_end_matches(".csv"))
                        } else {
                            format!("{}.summary.csv", out_path)
                        };
                        let fuzzy_like = matches!(
                            algorithm,
                            MatchingAlgorithm::Fuzzy
                                | MatchingAlgorithm::FuzzyNoMiddle
                                | MatchingAlgorithm::HouseholdGpu
                                | MatchingAlgorithm::HouseholdGpuOpt6
                        );
                        let summary = SummaryContext {
                            db_name: db_label.clone(),
                            table1: table1.clone(),
                            table2: table2.clone(),
                            total_table1: c1 as usize,
                            total_table2: c2 as usize,
                            matches_algo1: if fuzzy_like { 0 } else { count },
                            matches_algo2: 0,
                            matches_fuzzy: if fuzzy_like { count } else { 0 },
                            overlap_count: 0,
                            unique_algo1: 0,
                            unique_algo2: 0,
                            fetch_time: std::time::Duration::from_secs(0),
                            match1_time: t_match.elapsed(),
                            match2_time: std::time::Duration::from_secs(0),
                            export_time: std::time::Duration::from_secs(0),
                            mem_used_start_mb: 0,
                            mem_used_end_mb: 0,
                            started_utc: run_start_utc,
                            ended_utc: run_end_utc,
                            duration_secs,
                            exec_mode_algo1: None,
                            exec_mode_algo2: None,
                            exec_mode_fuzzy: Some(if gpu_any {
                                "GPU".into()
                            } else {
                                "CPU".into()
                            }),
                            algo_used: algo_label_summary(algorithm).to_string(),
                            gpu_used: gpu_any,
                            gpu_total_mb: gpu_total_mb.load(Ordering::Relaxed),
                            gpu_free_mb_end: gpu_free_mb.load(Ordering::Relaxed),
                            adv_level: None,
                            adv_level_description: None,
                        };
                        let _ = export_summary_csv(&sum_path, &summary);
                    }
                } else {
                    let pc = PartitioningConfig {
                        enabled: true,
                        strategy: part_strategy.clone(),
                    };
                    let count = {
                        let use_new_engine = cfg!(feature = "new_engine")
                            && std::env::var("NAME_MATCHER_NEW_ENGINE")
                                .ok()
                                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                                .unwrap_or(false);
                        if use_new_engine {
                            #[cfg(feature = "new_engine")]
                            {
                                let used_c = gpu_used_flag.clone();
                                let tot_c = gpu_total_mb.clone();
                                let free_c = gpu_free_mb.clone();
                                crate::engine::db_pipeline::db_pipeline::stream_new_engine_partitioned(&pool1, &table1, &table2, algorithm, |pair| { writer.write(pair)?; n_flushed += 1; if n_flushed % flush_every == 0 { writer.flush_partial()?; } Ok(()) }, s_cfg.clone(), move |u: ProgressUpdate| {
                                    if u.gpu_active { used_c.store(true, Ordering::Relaxed); }
                                    tot_c.store(u.gpu_total_mb as u64, Ordering::Relaxed);
                                    free_c.store(u.gpu_free_mb as u64, Ordering::Relaxed);
                                    info!("[part:new] {:.1}% | ETA: {}s | Mem used: {} MB | Avail: {} MB ({} / {})",
                                        u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total);
                                }, None, None, None, pc).await?
                            }
                            #[cfg(not(feature = "new_engine"))]
                            {
                                unreachable!()
                            }
                        } else {
                            let used_c = gpu_used_flag.clone();
                            let tot_c = gpu_total_mb.clone();
                            let free_c = gpu_free_mb.clone();
                            stream_match_csv_partitioned(&pool1, &table1, &table2, algorithm, |pair| { writer.write(pair)?; n_flushed += 1; if n_flushed % flush_every == 0 { writer.flush_partial()?; } Ok(()) }, s_cfg.clone(), move |u: ProgressUpdate| {
                                if u.gpu_active { used_c.store(true, Ordering::Relaxed); }
                                tot_c.store(u.gpu_total_mb as u64, Ordering::Relaxed);
                                free_c.store(u.gpu_free_mb as u64, Ordering::Relaxed);
                                info!("[part] {:.1}% | ETA: {}s | Mem used: {} MB | Avail: {} MB ({} / {})",
                                    u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total);
                            }, None, None, None, pc).await?
                        }
                    };
                    writer.flush()?;
                    info!(
                        "Wrote {} matches (streaming) in {:?}",
                        count,
                        t_match.elapsed()
                    );
                    {
                        let run_end_utc = chrono::Utc::now();
                        let duration_secs =
                            (run_end_utc - run_start_utc).num_milliseconds() as f64 / 1000.0;
                        let gpu_any = gpu_used_flag.load(Ordering::Relaxed);
                        let sum_path = if out_path.to_ascii_lowercase().ends_with(".csv") {
                            format!("{}_summary.csv", out_path.trim_end_matches(".csv"))
                        } else {
                            format!("{}.summary.csv", out_path)
                        };
                        let fuzzy_like = matches!(
                            algorithm,
                            MatchingAlgorithm::Fuzzy
                                | MatchingAlgorithm::FuzzyNoMiddle
                                | MatchingAlgorithm::HouseholdGpu
                                | MatchingAlgorithm::HouseholdGpuOpt6
                        );
                        let summary = SummaryContext {
                            db_name: db_label.clone(),
                            table1: table1.clone(),
                            table2: table2.clone(),
                            total_table1: c1 as usize,
                            total_table2: c2 as usize,
                            matches_algo1: if fuzzy_like { 0 } else { count },
                            matches_algo2: 0,
                            matches_fuzzy: if fuzzy_like { count } else { 0 },
                            overlap_count: 0,
                            unique_algo1: 0,
                            unique_algo2: 0,
                            fetch_time: std::time::Duration::from_secs(0),
                            match1_time: t_match.elapsed(),
                            match2_time: std::time::Duration::from_secs(0),
                            export_time: std::time::Duration::from_secs(0),
                            mem_used_start_mb: 0,
                            mem_used_end_mb: 0,
                            started_utc: run_start_utc,
                            ended_utc: run_end_utc,
                            duration_secs,
                            exec_mode_algo1: None,
                            exec_mode_algo2: None,
                            exec_mode_fuzzy: Some(if gpu_any {
                                "GPU".into()
                            } else {
                                "CPU".into()
                            }),
                            algo_used: algo_label_summary(algorithm).to_string(),
                            gpu_used: gpu_any,
                            gpu_total_mb: gpu_total_mb.load(Ordering::Relaxed),
                            gpu_free_mb_end: gpu_free_mb.load(Ordering::Relaxed),
                            adv_level: None,
                            adv_level_description: None,
                        };
                        info!("Writing CSV summary to {}", sum_path);
                        let _ = export_summary_csv(&sum_path, &summary);
                    }
                }
            }
        }
        if format == "xlsx" || format == "both" {
            // Streaming both algorithms into an XLSX workbook
            let xlsx_path = if out_path.to_ascii_lowercase().ends_with(".xlsx") {
                out_path.clone()
            } else {
                out_path.replace(".csv", ".xlsx")
            };
            let mem_start = memory_stats_mb().used_mb;
            let mut xw =
                XlsxStreamWriter::create_with_extra_fields(&xlsx_path, extra_field_names.clone())?;
            // using global run_start_utc captured earlier
            let gpu_used_a1 = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let gpu_used_a2 = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let gpu_total_mb = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
            let gpu_free_mb_end = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));

            let t1 = std::time::Instant::now();
            let mut algo1_count = 0usize;
            let a1_used = gpu_used_a1.clone();
            let total1 = gpu_total_mb.clone();
            let free1 = gpu_free_mb_end.clone();

            if let Some(p2) = &pool2_opt {
                let use_new_engine = cfg!(feature = "new_engine")
                    && std::env::var("NAME_MATCHER_NEW_ENGINE")
                        .ok()
                        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                        .unwrap_or(false);
                if use_new_engine {
                    #[cfg(feature = "new_engine")]
                    {
                        crate::engine::db_pipeline::db_pipeline::stream_new_engine_dual(
                            &pool1,
                            p2,
                            &table1,
                            &table2,
                            MatchingAlgorithm::IdUuidYasIsMatchedInfnbd,
                            |pair| {
                                algo1_count += 1;
                                xw.append_algo1(pair)
                            },
                            s_cfg.clone(),
                            move |u: ProgressUpdate| {
                                if u.gpu_active {
                                    a1_used.store(true, std::sync::atomic::Ordering::Relaxed);
                                }
                                total1.store(
                                    u.gpu_total_mb as u64,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                                free1.store(
                                    u.gpu_free_mb as u64,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                            },
                            None,
                        )
                        .await?;
                    }
                    #[cfg(not(feature = "new_engine"))]
                    {
                        unreachable!()
                    }
                } else {
                    stream_match_csv_dual(
                        &pool1,
                        p2,
                        &table1,
                        &table2,
                        MatchingAlgorithm::IdUuidYasIsMatchedInfnbd,
                        |pair| {
                            algo1_count += 1;
                            xw.append_algo1(pair)
                        },
                        s_cfg.clone(),
                        move |u: ProgressUpdate| {
                            if u.gpu_active {
                                a1_used.store(true, std::sync::atomic::Ordering::Relaxed);
                            }
                            total1
                                .store(u.gpu_total_mb as u64, std::sync::atomic::Ordering::Relaxed);
                            free1.store(u.gpu_free_mb as u64, std::sync::atomic::Ordering::Relaxed);
                        },
                        None,
                    )
                    .await?;
                }
            } else {
                let pc = PartitioningConfig {
                    enabled: true,
                    strategy: part_strategy.clone(),
                };
                let use_new_engine = cfg!(feature = "new_engine")
                    && std::env::var("NAME_MATCHER_NEW_ENGINE")
                        .ok()
                        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                        .unwrap_or(false);
                if use_new_engine {
                    #[cfg(feature = "new_engine")]
                    {
                        crate::engine::db_pipeline::db_pipeline::stream_new_engine_partitioned(
                            &pool1,
                            &table1,
                            &table2,
                            MatchingAlgorithm::IdUuidYasIsMatchedInfnbd,
                            |pair| {
                                algo1_count += 1;
                                xw.append_algo1(pair)
                            },
                            s_cfg.clone(),
                            move |u: ProgressUpdate| {
                                if u.gpu_active {
                                    a1_used.store(true, std::sync::atomic::Ordering::Relaxed);
                                }
                                total1.store(
                                    u.gpu_total_mb as u64,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                                free1.store(
                                    u.gpu_free_mb as u64,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                            },
                            None,
                            None,
                            None,
                            pc.clone(),
                        )
                        .await?;
                    }
                    #[cfg(not(feature = "new_engine"))]
                    {
                        unreachable!()
                    }
                } else {
                    stream_match_csv_partitioned(
                        &pool1,
                        &table1,
                        &table2,
                        MatchingAlgorithm::IdUuidYasIsMatchedInfnbd,
                        |pair| {
                            algo1_count += 1;
                            xw.append_algo1(pair)
                        },
                        s_cfg.clone(),
                        move |u: ProgressUpdate| {
                            if u.gpu_active {
                                a1_used.store(true, std::sync::atomic::Ordering::Relaxed);
                            }
                            total1
                                .store(u.gpu_total_mb as u64, std::sync::atomic::Ordering::Relaxed);
                            free1.store(u.gpu_free_mb as u64, std::sync::atomic::Ordering::Relaxed);
                        },
                        None,
                        None,
                        None,
                        pc.clone(),
                    )
                    .await?;
                }
            }
            let took_a1 = t1.elapsed();
            let t2 = std::time::Instant::now();
            let mut algo2_count = 0usize;
            let a2_used = gpu_used_a2.clone();
            let total2 = gpu_total_mb.clone();
            let free2 = gpu_free_mb_end.clone();

            if let Some(p2) = &pool2_opt {
                let use_new_engine = cfg!(feature = "new_engine")
                    && std::env::var("NAME_MATCHER_NEW_ENGINE")
                        .ok()
                        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                        .unwrap_or(false);
                if use_new_engine {
                    #[cfg(feature = "new_engine")]
                    {
                        crate::engine::db_pipeline::db_pipeline::stream_new_engine_dual(
                            &pool1,
                            p2,
                            &table1,
                            &table2,
                            MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd,
                            |pair| {
                                algo2_count += 1;
                                xw.append_algo2(pair)
                            },
                            s_cfg.clone(),
                            move |u: ProgressUpdate| {
                                if u.gpu_active {
                                    a2_used.store(true, std::sync::atomic::Ordering::Relaxed);
                                }
                                total2.store(
                                    u.gpu_total_mb as u64,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                                free2.store(
                                    u.gpu_free_mb as u64,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                            },
                            None,
                        )
                        .await?;
                    }
                    #[cfg(not(feature = "new_engine"))]
                    {
                        unreachable!()
                    }
                } else {
                    stream_match_csv_dual(
                        &pool1,
                        p2,
                        &table1,
                        &table2,
                        MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd,
                        |pair| {
                            algo2_count += 1;
                            xw.append_algo2(pair)
                        },
                        s_cfg.clone(),
                        move |u: ProgressUpdate| {
                            if u.gpu_active {
                                a2_used.store(true, std::sync::atomic::Ordering::Relaxed);
                            }
                            total2
                                .store(u.gpu_total_mb as u64, std::sync::atomic::Ordering::Relaxed);
                            free2.store(u.gpu_free_mb as u64, std::sync::atomic::Ordering::Relaxed);
                        },
                        None,
                    )
                    .await?;
                }
            } else {
                let pc = PartitioningConfig {
                    enabled: true,
                    strategy: part_strategy.clone(),
                };
                let use_new_engine = cfg!(feature = "new_engine")
                    && std::env::var("NAME_MATCHER_NEW_ENGINE")
                        .ok()
                        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                        .unwrap_or(false);
                if use_new_engine {
                    #[cfg(feature = "new_engine")]
                    {
                        crate::engine::db_pipeline::db_pipeline::stream_new_engine_partitioned(
                            &pool1,
                            &table1,
                            &table2,
                            MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd,
                            |pair| {
                                algo2_count += 1;
                                xw.append_algo2(pair)
                            },
                            s_cfg.clone(),
                            move |u: ProgressUpdate| {
                                if u.gpu_active {
                                    a2_used.store(true, std::sync::atomic::Ordering::Relaxed);
                                }
                                total2.store(
                                    u.gpu_total_mb as u64,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                                free2.store(
                                    u.gpu_free_mb as u64,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                            },
                            None,
                            None,
                            None,
                            pc,
                        )
                        .await?;
                    }
                    #[cfg(not(feature = "new_engine"))]
                    {
                        unreachable!()
                    }
                } else {
                    stream_match_csv_partitioned(
                        &pool1,
                        &table1,
                        &table2,
                        MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd,
                        |pair| {
                            algo2_count += 1;
                            xw.append_algo2(pair)
                        },
                        s_cfg.clone(),
                        move |u: ProgressUpdate| {
                            if u.gpu_active {
                                a2_used.store(true, std::sync::atomic::Ordering::Relaxed);
                            }
                            total2
                                .store(u.gpu_total_mb as u64, std::sync::atomic::Ordering::Relaxed);
                            free2.store(u.gpu_free_mb as u64, std::sync::atomic::Ordering::Relaxed);
                        },
                        None,
                        None,
                        None,
                        pc,
                    )
                    .await?;
                }
            }
            let took_a2 = t2.elapsed();
            let mem_end = memory_stats_mb().used_mb;
            let run_end_utc = chrono::Utc::now();
            let duration_secs = (run_end_utc - run_start_utc).num_milliseconds() as f64 / 1000.0;

            let gpu_used_a1_b = gpu_used_a1.load(std::sync::atomic::Ordering::Relaxed);
            let gpu_used_a2_b = gpu_used_a2.load(std::sync::atomic::Ordering::Relaxed);
            let gpu_total_mb_v = gpu_total_mb.load(std::sync::atomic::Ordering::Relaxed);
            let gpu_free_mb_end_v = gpu_free_mb_end.load(std::sync::atomic::Ordering::Relaxed);

            let summary = SummaryContext {
                db_name: db_label.clone(),
                table1: table1.clone(),
                table2: table2.clone(),
                total_table1: c1 as usize,

                total_table2: c2 as usize,
                matches_algo1: algo1_count,
                matches_algo2: algo2_count,
                matches_fuzzy: 0,
                overlap_count: 0, // not tracked in streaming mode to save memory
                unique_algo1: algo1_count,
                unique_algo2: algo2_count,
                fetch_time: std::time::Duration::from_secs(0),
                match1_time: took_a1,
                match2_time: took_a2,
                export_time: std::time::Duration::from_secs(0),
                mem_used_start_mb: mem_start,
                mem_used_end_mb: mem_end,
                started_utc: run_start_utc,
                ended_utc: run_end_utc,
                duration_secs,
                exec_mode_algo1: Some(if gpu_used_a1_b {
                    "GPU".into()
                } else {
                    "CPU".into()
                }),
                exec_mode_algo2: Some(if gpu_used_a2_b {
                    "GPU".into()
                } else {
                    "CPU".into()
                }),
                exec_mode_fuzzy: None,
                algo_used: "Both (1,2)".into(),
                gpu_used: gpu_used_a1_b || gpu_used_a2_b,
                gpu_total_mb: gpu_total_mb_v,
                gpu_free_mb_end: gpu_free_mb_end_v,
                adv_level: None,
                adv_level_description: None,
            };
            xw.finalize(&summary)?;
            info!(
                "XLSX written (streaming) to {} | a1={} a2={}",
                xlsx_path, algo1_count, algo2_count
            );
        }
    } else {
        // Honor normalization alignment option in in-memory mode as well
        let direct_norm_env = std::env::var("NAME_MATCHER_DIRECT_FUZZY_NORMALIZATION")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        crate::matching::set_direct_normalization_fuzzy(direct_norm_env || direct_norm_flag);
        // Apply GPU fuzzy direct pre-pass toggle in in-memory mode too (affects algo 3/4 via match_all_with_opts)
        let gpu_fuzzy_direct_env = std::env::var("NAME_MATCHER_GPU_FUZZY_DIRECT_HASH")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        crate::matching::set_gpu_fuzzy_direct_prep(gpu_fuzzy_direct_env || gpu_fuzzy_direct_flag);
        // Apply GPU fuzzy metrics toggle in in-memory mode as well
        let gpu_fuzzy_metrics_env = std::env::var("NAME_MATCHER_GPU_FUZZY_METRICS")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        crate::matching::set_gpu_fuzzy_metrics(gpu_fuzzy_metrics_env || gpu_fuzzy_metrics_flag);
        // Apply global GPU fuzzy overrides (heuristic force/disable)
        {
            let gpu_fuzzy_force_env = std::env::var("NAME_MATCHER_GPU_FUZZY_FORCE")
                .ok()
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            // Option 7 GPU full scoring toggle (in-memory only)
            let gpu_lev_full_env = std::env::var("NAME_MATCHER_GPU_LEVENSHTEIN_FULL_SCORING")
                .ok()
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            crate::matching::set_gpu_levenshtein_full_scoring(
                gpu_lev_full_env || gpu_lev_full_flag,
            );

            let gpu_fuzzy_disable_env = std::env::var("NAME_MATCHER_GPU_FUZZY_DISABLE")
                .ok()
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            // Option 7 GPU pre-pass toggle (in-memory only)
            let gpu_lev_prepass_env = std::env::var("NAME_MATCHER_GPU_LEVENSHTEIN_PREPASS")
                .ok()
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            crate::matching::set_gpu_levenshtein_prepass(
                gpu_lev_prepass_env || gpu_lev_prepass_flag,
            );

            crate::matching::set_gpu_fuzzy_force(gpu_fuzzy_force_env || gpu_fuzzy_force_flag);
            crate::matching::set_gpu_fuzzy_disable(gpu_fuzzy_disable_env || gpu_fuzzy_disable_flag);
        }
        // If a GPU backend is requested globally, force-enable GPU fuzzy metrics unless explicitly disabled
        let use_gpu_global = std::env::var("NAME_MATCHER_USE_GPU")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
            || args.iter().any(|a| a == "--use-gpu");
        // Respect explicit disable via environment/flag
        let gpu_fuzzy_disable_global = std::env::var("NAME_MATCHER_GPU_FUZZY_DISABLE")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
            || args.iter().any(|a| a == "--gpu-fuzzy-disable");
        if use_gpu_global && !gpu_fuzzy_disable_global {
            crate::matching::set_gpu_fuzzy_metrics(true);
            crate::matching::set_gpu_fuzzy_force(true);
        }
        // Auto-Optimize: force-enable fuzzy metrics for fuzzy/household algorithms
        if auto_optimize {
            if matches!(
                algorithm,
                MatchingAlgorithm::Fuzzy
                    | MatchingAlgorithm::FuzzyNoMiddle
                    | MatchingAlgorithm::HouseholdGpu
                    | MatchingAlgorithm::HouseholdGpuOpt6
            ) {
                crate::matching::set_gpu_fuzzy_metrics(true);
                crate::matching::set_gpu_fuzzy_force(true);
            }
        }

        // In-memory fallback (previous behavior)
        info!("Fetching rows from {} and {}", table1, table2);
        let t_fetch = std::time::Instant::now();
        let people1 = get_person_rows(&pool1, &table1).await?;

        let people2 = get_person_rows(pool2_opt.as_ref().unwrap_or(&pool1), &table2).await?;
        let took_fetch = t_fetch.elapsed();
        if took_fetch.as_secs() >= 30 {
            info!("Fetching took {:?}", took_fetch);
        }
        // Special handling for Option 5/6 (Household GPU aggregation path)
        if matches!(
            algorithm,
            MatchingAlgorithm::HouseholdGpu | MatchingAlgorithm::HouseholdGpuOpt6
        ) {
            // Determine backend (GPU/CPU) and GPU memory budget from CLI flag or environment
            let use_gpu_env = std::env::var("NAME_MATCHER_USE_GPU")
                .ok()
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            let use_gpu_cli = args.iter().any(|a| a == "--use-gpu");
            let use_gpu = use_gpu_cli || use_gpu_env;
            // Use 0 as default to trigger auto-calculation of adaptive budget (75% of free VRAM)
            // User can override with NAME_MATCHER_GPU_MEM_MB env var
            let gpu_mem_mb = std::env::var("NAME_MATCHER_GPU_MEM_MB")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(0);
            let opts = if use_gpu {
                MatchOptions {
                    backend: ComputeBackend::Gpu,
                    gpu: Some(GpuConfig {
                        device_id: None,
                        mem_budget_mb: gpu_mem_mb,
                    }),
                    progress: cfgp,
                    allow_birthdate_swap: false,
                }
            } else {
                MatchOptions {
                    backend: ComputeBackend::Cpu,
                    gpu: None,
                    progress: cfgp,
                    allow_birthdate_swap: false,
                }
            };
            // Align GPU fuzzy toggles for Algo 5 to ensure parity with CPU semantics
            let metrics_auto = std::env::var("NAME_MATCHER_GPU_FUZZY_METRICS")
                .ok()
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
                || args.iter().any(|a| a == "--gpu-fuzzy-metrics");
            let metrics_force = std::env::var("NAME_MATCHER_GPU_FUZZY_FORCE")
                .ok()
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
                || args.iter().any(|a| a == "--gpu-fuzzy-force");
            let metrics_off = std::env::var("NAME_MATCHER_GPU_FUZZY_DISABLE")
                .ok()
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
                || args.iter().any(|a| a == "--gpu-fuzzy-disable");
            crate::matching::apply_gpu_enhancements_for_algo(
                algorithm,
                false,
                false,
                metrics_auto,
                metrics_force,
                metrics_off,
            );

            // Household threshold (percentage in [0.5,1.0], allow 60, 80, 95, or 0.95)
            let hh_thr_env = std::env::var("NAME_MATCHER_HOUSEHOLD_THRESHOLD")
                .unwrap_or_else(|_| "95".to_string());
            let hh_min_conf = (|| {
                let s = hh_thr_env.trim();
                if let Some(p) = s.strip_suffix('%') {
                    p.parse::<f32>().ok().map(|v| (v / 100.0).clamp(0.5, 1.0))
                } else if s.contains('.') {
                    s.parse::<f32>().ok().map(|v| {
                        if v > 1.0 {
                            (v / 100.0).clamp(0.5, 1.0)
                        } else {
                            v.clamp(0.5, 1.0)
                        }
                    })
                } else {
                    s.parse::<f32>().ok().map(|v| (v / 100.0).clamp(0.5, 1.0))
                }
            })()
            .unwrap_or(0.95);

            info!(
                "Computing household aggregation (Option {}) with {} backend ...",
                if matches!(algorithm, MatchingAlgorithm::HouseholdGpu) {
                    5
                } else {
                    6
                },
                if use_gpu { "GPU" } else { "CPU" }
            );
            let mem_start = memory_stats_mb().used_mb;
            // using global run_start_utc captured earlier
            let t_hh = std::time::Instant::now();
            use std::sync::Arc;
            use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
            let gpu_used_flag = Arc::new(AtomicBool::new(false));
            let gpu_total_mb = Arc::new(AtomicU64::new(0));
            let gpu_free_mb = Arc::new(AtomicU64::new(0));
            let rows = if matches!(algorithm, MatchingAlgorithm::HouseholdGpu) {
                let used_c = gpu_used_flag.clone();
                let tot_c = gpu_total_mb.clone();
                let free_c = gpu_free_mb.clone();
                match_households_gpu_inmemory(
                    &people1,
                    &people2,
                    opts,
                    hh_min_conf,
                    move |u: ProgressUpdate| {
                        if u.gpu_active {
                            used_c.store(true, Ordering::Relaxed);
                        }
                        tot_c.store(u.gpu_total_mb as u64, Ordering::Relaxed);
                        free_c.store(u.gpu_free_mb as u64, Ordering::Relaxed);
                        info!(
                            "Progress: {:.1}% | ETA: {}s | Mem used: {} MB | Avail: {} MB ({} / {}) | GPU {} MB free / {} MB total",
                            u.percent,
                            u.eta_secs,
                            u.mem_used_mb,
                            u.mem_avail_mb,
                            u.processed,
                            u.total,
                            u.gpu_free_mb,
                            u.gpu_total_mb
                        );
                    },
                )
            } else {
                let used_c = gpu_used_flag.clone();
                let tot_c = gpu_total_mb.clone();
                let free_c = gpu_free_mb.clone();
                match_households_gpu_inmemory_opt6(
                    &people1,
                    &people2,
                    opts,
                    hh_min_conf,
                    move |u: ProgressUpdate| {
                        if u.gpu_active {
                            used_c.store(true, Ordering::Relaxed);
                        }
                        tot_c.store(u.gpu_total_mb as u64, Ordering::Relaxed);
                        free_c.store(u.gpu_free_mb as u64, Ordering::Relaxed);
                        info!(
                            "Progress: {:.1}% | ETA: {}s | Mem used: {} MB | Avail: {} MB ({} / {}) | GPU {} MB free / {} MB total",
                            u.percent,
                            u.eta_secs,
                            u.mem_used_mb,
                            u.mem_avail_mb,
                            u.processed,
                            u.total,
                            u.gpu_free_mb,
                            u.gpu_total_mb
                        );
                    },
                )
            };
            let took_hh = t_hh.elapsed();

            if format == "csv" || format == "both" {
                // Compute CSV metadata to match L12's export format
                let compute_backend = if use_gpu { "GPU" } else { "CPU" };
                let gpu_model = if use_gpu {
                    crate::matching::try_gpu_name()
                } else {
                    None
                };
                let gpu_features = if use_gpu { "GPU Fuzzy Metrics" } else { "" };
                let mut w = HouseholdCsvWriter::create_with_meta(
                    &out_path,
                    compute_backend,
                    gpu_model.as_deref(),
                    gpu_features,
                )?;
                for r in &rows {
                    w.write(r)?;
                }
                w.flush()?;
                info!(
                    "Household CSV written to {} ({} rows) in {:?}",
                    out_path,
                    rows.len(),
                    took_hh
                );
                // Sidecar CSV summary (Options 5	6)
                let run_end_utc = chrono::Utc::now();
                let duration_secs =
                    (run_end_utc - run_start_utc).num_milliseconds() as f64 / 1000.0;
                let gpu_any = gpu_used_flag.load(Ordering::Relaxed);
                let sum_path = if out_path.to_ascii_lowercase().ends_with(".csv") {
                    format!("{}_summary.csv", out_path.trim_end_matches(".csv"))
                } else {
                    format!("{}.summary.csv", out_path)
                };
                let summary = SummaryContext {
                    db_name: db_label.clone(),
                    table1: table1.clone(),
                    table2: table2.clone(),
                    total_table1: people1.len(),
                    total_table2: people2.len(),
                    matches_algo1: 0,
                    matches_algo2: 0,
                    matches_fuzzy: rows.len(),
                    overlap_count: 0,
                    unique_algo1: 0,
                    unique_algo2: 0,
                    fetch_time: took_fetch,
                    match1_time: took_hh,
                    match2_time: std::time::Duration::from_secs(0),
                    export_time: std::time::Duration::from_secs(0),
                    mem_used_start_mb: mem_start,
                    mem_used_end_mb: memory_stats_mb().used_mb,
                    started_utc: run_start_utc,
                    ended_utc: run_end_utc,
                    duration_secs,
                    exec_mode_algo1: None,
                    exec_mode_algo2: None,
                    exec_mode_fuzzy: Some(if gpu_any { "GPU".into() } else { "CPU".into() }),
                    algo_used: algo_label_summary(algorithm).to_string(),
                    gpu_used: gpu_any,
                    gpu_total_mb: gpu_total_mb.load(Ordering::Relaxed),
                    gpu_free_mb_end: gpu_free_mb.load(Ordering::Relaxed),
                    adv_level: None,
                    adv_level_description: None,
                };
                let _ = export_summary_csv(&sum_path, &summary);
            }
            if format == "xlsx" || format == "both" {
                export_households_xlsx(&out_path, &rows)?;
                info!(
                    "Household XLSX written to {} ({} rows) in {:?}",
                    out_path,
                    rows.len(),
                    took_hh
                );
            }

            return Ok(());
        }

        // Choose engine based on feature + env switch; legacy remains default
        let use_new_engine = cfg!(feature = "new_engine")
            && std::env::var("NAME_MATCHER_NEW_ENGINE")
                .ok()
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
        info!(
            "Matching using {:?} (engine: {})",
            algorithm,
            if use_new_engine { "new" } else { "legacy" }
        );
        let start = std::time::Instant::now();
        // using global run_start_utc captured earlier
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
        let mut fuzzy_gpu_used_opt: Option<(Arc<AtomicBool>, Arc<AtomicU64>, Arc<AtomicU64>)> =
            None;
        let matches_requested = if use_new_engine {
            #[cfg(feature = "new_engine")]
            {
                crate::engine::person_pipeline::run_new_engine_in_memory(
                    &people1, &people2, algorithm,
                )
            }
            #[cfg(not(feature = "new_engine"))]
            {
                unreachable!()
            }
        } else {
            let fuzzy_gpu_used = Arc::new(AtomicBool::new(false));
            let gpu_total_mb = Arc::new(AtomicU64::new(0));
            let gpu_free_mb = Arc::new(AtomicU64::new(0));
            let used_c = fuzzy_gpu_used.clone();
            let tot_c = gpu_total_mb.clone();
            let free_c = gpu_free_mb.clone();
            let res = match_all_progress(
                &people1,
                &people2,
                algorithm,
                cfgp,
                move |u: ProgressUpdate| {
                    if u.gpu_active {
                        used_c.store(true, Ordering::Relaxed);
                    }
                    tot_c.store(u.gpu_total_mb as u64, Ordering::Relaxed);
                    free_c.store(u.gpu_free_mb as u64, Ordering::Relaxed);
                    info!(
                        "Progress: {:.1}% | ETA: {}s | Mem used: {} MB | Avail: {} MB ({} / {})",
                        u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total
                    );
                },
            );
            fuzzy_gpu_used_opt = Some((fuzzy_gpu_used, gpu_total_mb, gpu_free_mb));
            res
        };
        let took_requested = start.elapsed();
        if took_requested.as_secs() >= 30 {
            info!("Matching stage took {:?}", took_requested);
        }

        // CSV export if requested or both
        if format == "csv" || format == "both" {
            info!(
                "Exporting {} match rows to {}",
                matches_requested.len(),
                out_path
            );
            let t_export = std::time::Instant::now();
            let fuzzy_thr_env =
                std::env::var("NAME_MATCHER_FUZZY_THRESHOLD").unwrap_or_else(|_| "95".to_string());
            let fuzzy_min_conf = (|| {
                let s = fuzzy_thr_env.trim();
                if let Some(p) = s.strip_suffix('%') {
                    p.parse::<f32>().ok().map(|v| (v / 100.0).clamp(0.6, 1.0))
                } else if s.contains('.') {
                    s.parse::<f32>().ok().map(|v| {
                        if v > 1.0 {
                            (v / 100.0).clamp(0.6, 1.0)
                        } else {
                            v.clamp(0.6, 1.0)
                        }
                    })
                } else {
                    s.parse::<f32>().ok().map(|v| (v / 100.0).clamp(0.6, 1.0))
                }
            })()
            .unwrap_or(0.95);
            export_to_csv(&matches_requested, &out_path, algorithm, fuzzy_min_conf)?;
            let took_export = t_export.elapsed();
            if took_export.as_secs() >= 30 {
                info!("Export took {:?}", took_export);
            }
            // Write sidecar summary CSV for all algorithms and modes (in-memory)
            {
                let run_end_utc = chrono::Utc::now();
                let duration_secs =
                    (run_end_utc - run_start_utc).num_milliseconds() as f64 / 1000.0;
                // GPU usage from legacy in-memory progress (if available)
                let (gpu_any, gpu_tot, gpu_free) =
                    if let Some((ref used, ref tot, ref free)) = fuzzy_gpu_used_opt {
                        (
                            used.load(Ordering::Relaxed),
                            tot.load(Ordering::Relaxed),
                            free.load(Ordering::Relaxed),
                        )
                    } else {
                        (false, 0, 0)
                    };
                let sum_path = if out_path.to_ascii_lowercase().ends_with(".csv") {
                    format!("{}_summary.csv", out_path.trim_end_matches(".csv"))
                } else {
                    format!("{}.summary.csv", out_path)
                };
                let fuzzy_like = matches!(
                    algorithm,
                    MatchingAlgorithm::Fuzzy
                        | MatchingAlgorithm::FuzzyNoMiddle
                        | MatchingAlgorithm::HouseholdGpu
                        | MatchingAlgorithm::HouseholdGpuOpt6
                );
                let summary = SummaryContext {
                    db_name: db_label.clone(),
                    table1: table1.clone(),
                    table2: table2.clone(),
                    total_table1: people1.len(),
                    total_table2: people2.len(),
                    matches_algo1: 0,
                    matches_algo2: 0,
                    matches_fuzzy: if fuzzy_like {
                        matches_requested.len()
                    } else {
                        0
                    },
                    overlap_count: 0,
                    unique_algo1: 0,
                    unique_algo2: 0,
                    fetch_time: took_fetch,
                    match1_time: took_requested,
                    match2_time: std::time::Duration::from_secs(0),
                    export_time: took_export,
                    mem_used_start_mb: 0,
                    mem_used_end_mb: 0,
                    started_utc: run_start_utc,
                    ended_utc: run_end_utc,
                    duration_secs,
                    exec_mode_algo1: None,
                    exec_mode_algo2: None,
                    exec_mode_fuzzy: Some(if gpu_any { "GPU".into() } else { "CPU".into() }),
                    algo_used: algo_label_summary(algorithm).to_string(),
                    gpu_used: gpu_any,
                    gpu_total_mb: gpu_tot,
                    gpu_free_mb_end: gpu_free,
                    adv_level: None,
                    adv_level_description: None,
                };
                let _ = export_summary_csv(&sum_path, &summary);
            }
        }

        // XLSX export implies computing both algorithms and writing 3 sheets
        if format == "xlsx" || format == "both" {
            info!("Preparing XLSX export (both algorithms) ...");
            let mem_start = memory_stats_mb().used_mb;

            // Algo 1 & 2 (in-memory), track GPU activity via progress callbacks
            // using global run_start_utc captured earlier
            let a1_used = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let a2_used = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let total1 = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
            let free1 = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
            let total2 = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
            let free2 = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));

            // Algo 1
            let t1 = std::time::Instant::now();
            let a1 = {
                let a1_used_c = std::sync::Arc::clone(&a1_used);
                let total1_c = std::sync::Arc::clone(&total1);
                let free1_c = std::sync::Arc::clone(&free1);
                match_all_progress(
                    &people1,
                    &people2,
                    MatchingAlgorithm::IdUuidYasIsMatchedInfnbd,
                    cfgp,
                    move |u: ProgressUpdate| {
                        if u.gpu_active {
                            a1_used_c.store(true, std::sync::atomic::Ordering::Relaxed);
                        }
                        total1_c.store(u.gpu_total_mb as u64, std::sync::atomic::Ordering::Relaxed);
                        free1_c.store(u.gpu_free_mb as u64, std::sync::atomic::Ordering::Relaxed);
                    },
                )
            };
            let took_a1 = t1.elapsed();

            // Algo 2
            let t2 = std::time::Instant::now();
            let a2 = {
                let a2_used_c = std::sync::Arc::clone(&a2_used);
                let total2_c = std::sync::Arc::clone(&total2);
                let free2_c = std::sync::Arc::clone(&free2);
                match_all_progress(
                    &people1,
                    &people2,
                    MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd,
                    cfgp,
                    move |u: ProgressUpdate| {
                        if u.gpu_active {
                            a2_used_c.store(true, std::sync::atomic::Ordering::Relaxed);
                        }
                        total2_c.store(u.gpu_total_mb as u64, std::sync::atomic::Ordering::Relaxed);
                        free2_c.store(u.gpu_free_mb as u64, std::sync::atomic::Ordering::Relaxed);
                    },
                )
            };
            let took_a2 = t2.elapsed();

            // Overlap/unique counts by (id1,id2)
            use std::collections::HashSet;
            let set1: HashSet<(i64, i64)> =
                a1.iter().map(|m| (m.person1.id, m.person2.id)).collect();
            let set2: HashSet<(i64, i64)> =
                a2.iter().map(|m| (m.person1.id, m.person2.id)).collect();
            let overlap = set1.intersection(&set2).count();
            let unique1 = set1.len().saturating_sub(overlap);
            let unique2 = set2.len().saturating_sub(overlap);

            let t_export = std::time::Instant::now();
            let mem_end = memory_stats_mb().used_mb;
            let run_end_utc = chrono::Utc::now();
            let duration_secs = (run_end_utc - run_start_utc).num_milliseconds() as f64 / 1000.0;

            // Derive exec modes and GPU summary
            let a1_gpu = a1_used.load(std::sync::atomic::Ordering::Relaxed);
            let a2_gpu = a2_used.load(std::sync::atomic::Ordering::Relaxed);
            let gpu_used_any = a1_gpu || a2_gpu;
            let gpu_total_mb = std::cmp::max(
                total1.load(std::sync::atomic::Ordering::Relaxed),
                total2.load(std::sync::atomic::Ordering::Relaxed),
            );
            // Prefer the last algorithm's free reading when available
            let gpu_free_mb_end = if a2_gpu || total2.load(std::sync::atomic::Ordering::Relaxed) > 0
            {
                free2.load(std::sync::atomic::Ordering::Relaxed)
            } else {
                free1.load(std::sync::atomic::Ordering::Relaxed)
            };

            let summary = SummaryContext {
                db_name: db_label.clone(),
                table1: table1.clone(),
                table2: table2.clone(),
                total_table1: people1.len(),
                total_table2: people2.len(),
                matches_algo1: a1.len(),
                matches_algo2: a2.len(),
                matches_fuzzy: 0,
                overlap_count: overlap,
                unique_algo1: unique1,
                unique_algo2: unique2,
                fetch_time: took_fetch,
                match1_time: took_a1,
                match2_time: took_a2,
                export_time: std::time::Duration::from_secs(0),
                mem_used_start_mb: mem_start,
                mem_used_end_mb: mem_end,
                started_utc: run_start_utc,
                ended_utc: run_end_utc,
                duration_secs,
                exec_mode_algo1: Some(if a1_gpu { "GPU".into() } else { "CPU".into() }),
                exec_mode_algo2: Some(if a2_gpu { "GPU".into() } else { "CPU".into() }),
                exec_mode_fuzzy: None,
                algo_used: "Both (1,2)".into(),
                gpu_used: gpu_used_any,
                adv_level: None,
                adv_level_description: None,
                gpu_total_mb,
                gpu_free_mb_end,
            };

            // Decide xlsx path
            let xlsx_path = if out_path.to_ascii_lowercase().ends_with(".xlsx") {
                out_path.clone()
            } else if out_path.to_ascii_lowercase().ends_with(".csv") {
                out_path.replace(".csv", ".xlsx")
            } else {
                out_path.clone() + ".xlsx"
            };

            export_to_xlsx(&a1, &a2, &xlsx_path, &summary)?;
            let took_export = t_export.elapsed();
            if took_export.as_secs() >= 30 {
                info!("XLSX export took {:?}", took_export);
            }
            info!("XLSX written to {}", xlsx_path);
        }
    }

    info!("Done.");
    #[cfg(feature = "gpu")]
    {
        // Ensure background GPU dynamic tuner thread is stopped on CLI exit
        crate::matching::dyn_tuner_stop();
    }
    Ok(())
}
