//! CLI flag parsing for GPU, streaming, and advanced matching options.
//!
//! This module parses command-line flags that can be placed anywhere in argv
//! and environment variables that control GPU acceleration and streaming behavior.

use crate::matching::cascade::MissingColumnMode;

/// Parsed CLI flags for GPU and streaming configuration.
#[derive(Debug, Clone, Default)]
pub struct CliFlags {
    // GPU flags
    pub gpu_hash_join: bool,
    pub gpu_fuzzy_direct: bool,
    pub direct_norm: bool,
    pub gpu_streams: Option<u32>,
    pub gpu_buffer_pool: bool,
    pub no_gpu_buffer_pool: bool,
    pub gpu_pinned_host: bool,
    pub gpu_fuzzy_metrics: bool,
    pub gpu_fuzzy_force: bool,
    pub gpu_fuzzy_disable: bool,
    pub gpu_lev_prepass: bool,
    pub gpu_lev_full: bool,
    pub use_gpu: bool,

    // Streaming/optimization flags
    pub auto_optimize: bool,
    pub streaming: bool,

    // Advanced matching flags
    pub adv_level: Option<u8>,
    pub adv_code_col: Option<String>,
    pub adv_threshold: f32,

    // Cascade matching flags
    /// Enable cascade mode (run L1-L11 sequentially, L12 excluded)
    pub cascade: bool,
    /// How to handle missing geographic columns in cascade mode
    pub cascade_missing_columns: MissingColumnMode,
    /// Specific levels to run in cascade (comma-separated, e.g., "1,2,3,10,11")
    pub cascade_levels: Vec<u8>,
}

impl CliFlags {
    /// Parse CLI flags from command-line arguments.
    pub fn from_args(args: &[String]) -> Self {
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
        let gpu_fuzzy_disable_flag = args.iter().any(|a| a == "--gpu-fuzzy-disable");
        let gpu_lev_prepass_flag = args.iter().any(|a| a == "--gpu-levenshtein-prepass");
        let gpu_lev_full_flag = args.iter().any(|a| a == "--gpu-levenshtein-full-scoring");
        let use_gpu_flag = args.iter().any(|a| a == "--use-gpu");
        let auto_optimize_flag = args.iter().any(|a| a == "--auto-optimize");

        // Advanced matching flags
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
        let adv_threshold_flag: Option<f32> = args
            .windows(2)
            .find(|w| w[0] == "--advanced-threshold")
            .and_then(|w| w.get(1))
            .and_then(|s| s.parse::<f32>().ok());

        // Cascade matching flags
        let cascade_flag = args.iter().any(|a| a == "--cascade");
        let cascade_missing_columns = args
            .windows(2)
            .find(|w| w[0] == "--cascade-missing-columns")
            .and_then(|w| w.get(1))
            .map(|s| match s.to_lowercase().as_str() {
                "auto-skip" | "auto_skip" | "autoskip" => MissingColumnMode::AutoSkip,
                "manual" | "manual-select" => MissingColumnMode::ManualSelect,
                "abort" | "abort-on-missing" => MissingColumnMode::AbortOnMissing,
                _ => MissingColumnMode::AutoSkip,
            })
            .unwrap_or(MissingColumnMode::AutoSkip);
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

        Self {
            gpu_hash_join: gpu_hash_flag,
            gpu_fuzzy_direct: gpu_fuzzy_direct_flag,
            direct_norm: direct_norm_flag,
            gpu_streams: gpu_streams_flag,
            gpu_buffer_pool: gpu_buffer_pool_flag,
            no_gpu_buffer_pool: no_gpu_buffer_pool_flag,
            gpu_pinned_host: gpu_pinned_host_flag,
            gpu_fuzzy_metrics: gpu_fuzzy_metrics_flag,
            gpu_fuzzy_force: gpu_fuzzy_force_flag,
            gpu_fuzzy_disable: gpu_fuzzy_disable_flag,
            gpu_lev_prepass: gpu_lev_prepass_flag,
            gpu_lev_full: gpu_lev_full_flag,
            use_gpu: use_gpu_flag,
            auto_optimize: auto_optimize_flag,
            streaming: false, // Set from env
            adv_level: adv_level_num,
            adv_code_col,
            adv_threshold: adv_threshold_flag
                .map(|v| v.clamp(0.5, 1.0))
                .unwrap_or(0.95),
            cascade: cascade_flag,
            cascade_missing_columns,
            cascade_levels,
        }
    }

    /// Merge environment variable settings into flags.
    pub fn merge_env(&mut self) {
        // GPU env vars
        if parse_bool_env("NAME_MATCHER_GPU_HASH_JOIN") {
            self.gpu_hash_join = true;
        }
        if parse_bool_env("NAME_MATCHER_GPU_FUZZY_DIRECT_HASH") {
            self.gpu_fuzzy_direct = true;
        }
        if parse_bool_env("NAME_MATCHER_DIRECT_FUZZY_NORMALIZATION") {
            self.direct_norm = true;
        }
        if parse_bool_env("NAME_MATCHER_GPU_FUZZY_METRICS") {
            self.gpu_fuzzy_metrics = true;
        }
        if parse_bool_env("NAME_MATCHER_GPU_FUZZY_FORCE") {
            self.gpu_fuzzy_force = true;
        }
        if parse_bool_env("NAME_MATCHER_GPU_FUZZY_DISABLE") {
            self.gpu_fuzzy_disable = true;
        }
        if parse_bool_env("NAME_MATCHER_USE_GPU") {
            self.use_gpu = true;
        }
        if parse_bool_env("NAME_MATCHER_AUTO_OPTIMIZE") {
            self.auto_optimize = true;
        }
        if parse_bool_env("NAME_MATCHER_STREAMING") {
            self.streaming = true;
        }

        // GPU streams from env
        if self.gpu_streams.is_none() {
            if let Ok(s) = std::env::var("NAME_MATCHER_GPU_STREAMS") {
                self.gpu_streams = s.parse().ok();
            }
        }

        // Advanced threshold from env (fallback to household threshold)
        if self.adv_threshold == 0.95 {
            if let Ok(s) = std::env::var("NAME_MATCHER_HOUSEHOLD_THRESHOLD") {
                self.adv_threshold = parse_threshold_string(&s);
            }
        }

        // Cascade mode from env
        if parse_bool_env("NAME_MATCHER_CASCADE") {
            self.cascade = true;
        }
    }

    /// Print cascade-specific help text
    pub fn print_cascade_help() {
        eprintln!("\nCascade Matching Options:");
        eprintln!("  --cascade                       Run cascade matching (L1-L11 sequentially)");
        eprintln!(
            "                                  Note: L12 (Household Matching) is EXCLUDED from cascade."
        );
        eprintln!(
            "                                  Household matching has different semantics and should be"
        );
        eprintln!("                                  run separately using --advanced-level 12.");
        eprintln!("  --cascade-missing-columns MODE  How to handle missing geographic columns:");
        eprintln!(
            "                                    auto-skip (default) - Skip unavailable levels"
        );
        eprintln!(
            "                                    manual - Use only levels specified with --levels"
        );
        eprintln!(
            "                                    abort - Abort if any required columns are missing"
        );
        eprintln!(
            "  --levels 1,2,3,10,11            Run only specific levels (comma-separated, 1-11)"
        );
        eprintln!("                                  L4-L6 require 'barangay_code' column");
        eprintln!("                                  L7-L9 require 'city_code' column");
        eprintln!("\nCascade Output:");
        eprintln!("  Per-level CSV files: {{base_output}}_L01.csv, {{base_output}}_L02.csv, ...");
        eprintln!("  Summary file: {{base_output}}_summary.txt");
        eprintln!("\nExample:");
        eprintln!("  name_matcher ... --cascade --output matches.csv");
        eprintln!("    Generates: matches_L01.csv, matches_L02.csv, ..., matches_summary.txt");
    }
}

/// Parse a boolean environment variable (accepts "1" or "true").
fn parse_bool_env(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Parse a threshold string (supports percentage or decimal).
fn parse_threshold_string(s: &str) -> f32 {
    let s = s.trim();
    if let Some(p) = s.strip_suffix('%') {
        p.parse::<f32>().ok().map(|v| (v / 100.0).clamp(0.5, 1.0))
    } else if s.contains('.') {
        s.parse::<f32>().ok().map(|v| v.clamp(0.5, 1.0))
    } else {
        s.parse::<f32>().ok().map(|v| (v / 100.0).clamp(0.5, 1.0))
    }
    .unwrap_or(0.95)
}
