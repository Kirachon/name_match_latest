//! Orchestrator module: high-level workflow coordination.
//!
//! This module provides the main run logic that coordinates:
//! - Database connections
//! - Schema validation
//! - Algorithm dispatch
//! - Export generation
//! - Summary reporting
//!
//! ## Integration Path
//!
//! These modules are ready for integration into `src/main.rs`. To adopt them
//! incrementally:
//!
//! ### Step 1: Use `validate_schemas()` (Low Risk)
//!
//! Replace schema validation in `main.rs` lines 538-550 with:
//!
//! ```rust,ignore
//! use crate::orchestrator::validate_schemas;
//! validate_schemas(&pool1, pool2_opt.as_ref(), &cfg.database, &db2_name, &table1, &table2).await?;
//! ```
//!
//! This consolidates schema discovery and validation into a single call.
//!
//! ### Step 2: Use `should_use_streaming()` (Low Risk)
//!
//! Replace streaming decision logic in `main.rs` lines 580-590, 1163-1174 with:
//!
//! ```rust,ignore
//! use crate::orchestrator::should_use_streaming;
//! let (use_streaming, c1, c2) = should_use_streaming(
//!     &pool1, pool2_opt.as_ref(), &table1, &table2, streaming_env
//! ).await?;
//! ```
//!
//! This encapsulates the auto-streaming threshold logic (100K records).
//!
//! ### Step 3: Use `apply_auto_optimize()` (Low Risk)
//!
//! Replace auto-optimization in `main.rs` lines 557-573 with:
//!
//! ```rust,ignore
//! use crate::orchestrator::apply_auto_optimize;
//! if auto_optimize {
//!     apply_auto_optimize(algorithm);
//! }
//! ```
//!
//! ### Step 4: Use `RunConfig` struct (Medium Risk)
//!
//! For a complete refactoring, create a `RunConfig` struct at the start of `run()`:
//!
//! ```rust,ignore
//! let run_config = RunConfig {
//!     db_config: cfg,
//!     db2_config: cfg2_opt,
//!     table1, table2, algorithm, algo_num,
//!     out_path, format,
//!     flags: cli_flags,  // requires Step 1 from cli/mod.rs
//!     run_start_utc: chrono::Utc::now(),
//! };
//! ```
//!
//! Then pass `&run_config` to functions instead of individual parameters.
//!
//! ### Step 5: Use `SummaryBuilder` (Low Risk)
//!
//! Replace summary generation throughout `main.rs` with:
//!
//! ```rust,ignore
//! use crate::orchestrator::summary::SummaryBuilder;
//! let summary = SummaryBuilder::new()
//!     .algorithm(algo_label_summary(algorithm))
//!     .tables(&table1, &table2, &db_label)
//!     .counts(c1, c2)
//!     .matches(match_count)
//!     .duration(elapsed)
//!     .build();
//! summary.write_csv(&summary_path)?;
//! ```
//!
//! ### Validation After Each Step
//!
//! After each integration step, run:
//! ```bash
//! cargo test --all
//! cargo test streaming_parity --release
//! ```

pub mod summary;

use anyhow::{Result, bail};
use log::{info, warn};

use crate::cli::args;
use crate::cli::flags::CliFlags;
use crate::config::DatabaseConfig;
use crate::db::{discover_table_columns, get_person_count, make_pool};
use crate::matching::MatchingAlgorithm;

/// Configuration for a matching run, parsed from CLI and environment.
#[derive(Debug, Clone)]
pub struct RunConfig {
    /// Primary database configuration
    pub db_config: DatabaseConfig,
    /// Optional secondary database configuration
    pub db2_config: Option<DatabaseConfig>,
    /// Source table name
    pub table1: String,
    /// Target table name
    pub table2: String,
    /// Selected algorithm
    pub algorithm: MatchingAlgorithm,
    /// Algorithm number
    pub algo_num: u8,
    /// Output path
    pub out_path: String,
    /// Output format
    pub format: String,
    /// CLI flags
    pub flags: CliFlags,
    /// Run start timestamp
    pub run_start_utc: chrono::DateTime<chrono::Utc>,
}

/// Auto-streaming threshold: enable streaming automatically for large datasets
pub const AUTO_STREAM_THRESHOLD: i64 = 100_000;

/// Validate table schemas for the matching run.
pub async fn validate_schemas(
    pool1: &sqlx::MySqlPool,
    pool2_opt: Option<&sqlx::MySqlPool>,
    db1_name: &str,
    db2_name: &str,
    table1: &str,
    table2: &str,
) -> Result<()> {
    let cols1 = discover_table_columns(pool1, db1_name, table1).await?;
    let cols2 = discover_table_columns(pool2_opt.unwrap_or(pool1), db2_name, table2).await?;

    cols1.validate_basic()?;
    if !(cols2.has_id && cols2.has_first_name && cols2.has_last_name && cols2.has_birthdate) {
        bail!(
            "Table {} missing required columns: requires id, first_name, last_name, birthdate (uuid optional)",
            table2
        );
    }

    info!("{} columns: {:?}", table1, cols1);
    info!("{} columns: {:?}", table2, cols2);
    Ok(())
}

/// Determine if streaming mode should be used.
pub async fn should_use_streaming(
    pool1: &sqlx::MySqlPool,
    pool2_opt: Option<&sqlx::MySqlPool>,
    table1: &str,
    table2: &str,
    streaming_env: bool,
) -> Result<(bool, i64, i64)> {
    let c1 = get_person_count(pool1, table1).await?;
    let c2 = get_person_count(pool2_opt.unwrap_or(pool1), table2).await?;

    let auto_stream_triggered = !streaming_env && (c1 + c2 > AUTO_STREAM_THRESHOLD);
    let use_streaming = streaming_env || auto_stream_triggered;

    if auto_stream_triggered {
        info!(
            "Auto-enabling streaming mode: {} total records exceeds threshold of {}",
            c1 + c2,
            AUTO_STREAM_THRESHOLD
        );
    }

    Ok((use_streaming, c1, c2))
}

/// Get human-readable algorithm label for summaries.
pub fn algo_label_summary(algo: MatchingAlgorithm) -> &'static str {
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

/// Apply auto-optimization settings based on system profile.
pub fn apply_auto_optimize(algorithm: MatchingAlgorithm) {
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
