// SRS-II Name Matching Application GUI
// Creator/Author: Matthias Tangonan
// This file implements the desktop GUI using eframe/egui.

use eframe::egui::{self, ComboBox, Context, ProgressBar, TextEdit};
use eframe::{App, Frame, NativeOptions};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use std::thread;

use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::fs;
/// Global Tokio runtime for all GUI async operations to prevent per-click runtime leaks
fn gui_runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("nm-gui-rt")
            .build()
            .expect("failed to build global GUI Tokio runtime")
    })
}

use name_matcher::config::DatabaseConfig;
use name_matcher::db::make_pool_with_size;
use name_matcher::db::{get_person_count, get_person_rows};
use name_matcher::export::csv_export::{AdvCsvStreamWriter, CsvStreamWriter, HouseholdCsvWriter};
use name_matcher::export::xlsx_export::{SummaryContext, XlsxStreamWriter, export_households_xlsx};
use name_matcher::matching::advanced_matcher::{AdvColumns, AdvConfig, AdvLevel};
use name_matcher::matching::{
    ComputeBackend, GpuConfig, MatchOptions, MatchingAlgorithm, ProgressConfig, ProgressUpdate,
    StreamControl, StreamingConfig, match_all_progress, match_all_with_opts,
    match_households_gpu_inmemory, match_households_gpu_inmemory_opt6, stream_match_advanced,
    stream_match_advanced_dual, stream_match_advanced_l12, stream_match_advanced_l12_dual,
    stream_match_csv, stream_match_csv_dual, stream_match_option5, stream_match_option5_dual,
    stream_match_option6, stream_match_option6_dual,
};
use sqlx::MySqlPool;

#[derive(Clone, Copy, PartialEq, Debug)]
enum ModeSel {
    Auto,
    Streaming,
    InMemory,
}
use name_matcher::db::schema::{
    discover_table_columns, get_all_table_columns, get_person_count_fast,
};
use name_matcher::models::TableColumns;
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq)]
enum FormatSel {
    Csv,
    Xlsx,
    Both,
}

#[derive(Debug)]
enum Msg {
    Progress(ProgressUpdate),
    Info(String),
    Tables(Vec<String>),
    DbPools {
        pool1: MySqlPool,
        pool2: Option<MySqlPool>,
    },

    Tables2(Vec<String>),
    Done {
        a1: usize,
        a2: usize,
        csv: usize,
        path: String,
        gpu_hash_used: bool,
        gpu_fuzzy_used: bool,
    },
    Error(String),
    ErrorRich {
        display: String,
        sqlstate: Option<String>,
        chain: String,
        operation: Option<String>,
    },
}

// GUI log forwarding: capture log::info!/warn!/error! and stream to GUI console via Msg
use log::{Level, LevelFilter, Metadata, Record};
use std::io::BufWriter;
use std::io::Write;
use std::sync::{Mutex, OnceLock};

static GUI_LOG_SENDER: OnceLock<Mutex<Option<Sender<Msg>>>> = OnceLock::new();
static GUI_LOG_FILE: OnceLock<Mutex<Option<BufWriter<std::fs::File>>>> = OnceLock::new();
static GUI_LOG_FILE_FLUSH_STATE: OnceLock<Mutex<(std::time::Instant, u32)>> = OnceLock::new();

struct GuiForwardLogger;
impl log::Log for GuiForwardLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Level::Info
    }
    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let line = format!("{}", record.args());
        // Forward to GUI console
        if let Some(mtx) = GUI_LOG_SENDER.get() {
            if let Ok(mut guard) = mtx.lock() {
                if let Some(tx) = guard.as_ref() {
                    let _ = tx.send(Msg::Info(line.clone()));
                }
            }
        }
        // Mirror to optional GUI log file (for external console tail)
        if let Some(fm) = GUI_LOG_FILE.get() {
            if let Ok(mut g) = fm.lock() {
                if let Some(f) = g.as_mut() {
                    let _ = writeln!(f, "{}", line);
                    // Avoid flushing on every line (disk I/O can dominate runtime). Flush periodically.
                    if let Some(st) = GUI_LOG_FILE_FLUSH_STATE.get() {
                        if let Ok(mut s) = st.lock() {
                            s.1 = s.1.saturating_add(1);
                            if s.0.elapsed() >= std::time::Duration::from_millis(250)
                                || (s.1 % 200) == 0
                            {
                                let _ = f.flush();
                                s.0 = std::time::Instant::now();
                            }
                        }
                    }
                }
            }
        }
    }
    fn flush(&self) {}
}

fn init_gui_logger_once() {
    // Initialize global sender holder if not set
    let _ = GUI_LOG_SENDER.set(Mutex::new(None));

    // Try to install our logger; ignore if another logger is already set
    if log::set_boxed_logger(Box::new(GuiForwardLogger)).is_ok() {
        log::set_max_level(LevelFilter::Info);
    }
}

fn set_gui_log_sender(tx: Sender<Msg>) {
    if let Some(mtx) = GUI_LOG_SENDER.get() {
        if let Ok(mut guard) = mtx.lock() {
            *guard = Some(tx);
        }
    }
}

// Schema metadata cache helper: fetch TableColumns and all column names with TTL-based caching
async fn schema_cached_or_fetch(
    cache: &std::sync::Arc<
        std::sync::Mutex<HashMap<(String, String), (Option<TableColumns>, Vec<String>)>>,
    >,
    ts: &std::sync::Arc<std::sync::Mutex<HashMap<(String, String), Instant>>>,
    pool: &MySqlPool,
    database: &str,
    table: &str,
    ttl: Duration,
) -> anyhow::Result<(Option<TableColumns>, Vec<String>)> {
    let key = (database.to_string(), table.to_string());

    // Fast path: TTL cache hit
    if let Ok(ts_guard) = ts.lock() {
        if let Some(stamp) = ts_guard.get(&key) {
            if stamp.elapsed() < ttl {
                if let Ok(cache_guard) = cache.lock() {
                    if let Some((cols_opt, extras)) = cache_guard.get(&key) {
                        return Ok((cols_opt.clone(), extras.clone()));
                    }
                }
            }
        }
    }

    // Fetch in parallel (discover_table_columns + get_all_table_columns)
    let (cols_res, extras_res) = tokio::join!(
        discover_table_columns(pool, database, table),
        get_all_table_columns(pool, database, table),
    );

    let cols_opt = cols_res.ok();
    let extras = extras_res?;

    // Update cache
    if let Ok(mut cache_guard) = cache.lock() {
        cache_guard.insert(key.clone(), (cols_opt.clone(), extras.clone()));
    }
    if let Ok(mut ts_guard) = ts.lock() {
        ts_guard.insert(key, Instant::now());
    }

    Ok((cols_opt, extras))
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ReportFormat {
    Text,
    Json,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
enum ErrorCategory {
    DbConnection,
    TableValidation,
    SchemaValidation,
    DataFormat,
    ResourceConstraint,

    Configuration,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum FuzzyGpuMode {
    Off,
    Auto,
    Force,
}

#[derive(Debug, Clone, Serialize)]
struct DiagEvent {
    ts_utc: String,
    category: ErrorCategory,
    message: String,
    sqlstate: Option<String>,
    chain: Option<String>,
    operation: Option<String>,
    source_action: String,

    db1_host: String,
    db1_database: String,
    db2_host: Option<String>,
    db2_database: Option<String>,
    table1: Option<String>,
    table2: Option<String>,
    mem_avail_mb: u64,
    pool_size_cfg: u32,
    env_overrides: Vec<(String, String)>,
}

fn categorize_error(msg: &str) -> ErrorCategory {
    categorize_error_with(None, msg)
}

fn categorize_error_with(sqlstate: Option<&str>, msg: &str) -> ErrorCategory {
    if let Some(state) = sqlstate {
        match state {
            // Schema/table
            "42S02" => return ErrorCategory::TableValidation, // table doesn't exist
            "42S22" => return ErrorCategory::SchemaValidation, // column not found
            // Privilege / access
            "28000" => return ErrorCategory::DbConnection, // access denied
            "42000" => {
                let m = msg.to_ascii_lowercase();
                if m.contains("permission") || m.contains("denied") {
                    return ErrorCategory::TableValidation;
                }
                return ErrorCategory::Configuration; // syntax or access rule violation
            }
            // Connection errors
            "08001" | "08004" | "08S01" => return ErrorCategory::DbConnection,
            // Timeouts
            "HYT00" => return ErrorCategory::ResourceConstraint,
            // Data problems
            "22001" | "22003" | "22007" => return ErrorCategory::DataFormat, // truncation, out of range, invalid datetime
            // Integrity / FK
            "23000" => return ErrorCategory::DataFormat, // integrity constraint violation
            _ => {}
        }
    }
    let m = msg.to_ascii_lowercase();
    if m.contains("access denied")
        || m.contains("authentication")
        || m.contains("unknown database")
        || (m.contains("host") && m.contains("unreach"))
        || m.contains("timed out")
        || m.contains("connection") && m.contains("fail")
    {
        ErrorCategory::DbConnection
    } else if m.contains("doesn't exist")
        || m.contains("no such table")
        || (m.contains("table") && m.contains("permission"))
    {
        ErrorCategory::TableValidation
    } else if m.contains("unknown column")
        || (m.contains("column") && m.contains("type"))
        || (m.contains("index") && m.contains("missing"))
    {
        ErrorCategory::SchemaValidation
    } else if m.contains("incorrect date value")
        || m.contains("invalid date")
        || (m.contains("parse") && m.contains("date"))
        || (m.contains("null") && m.contains("required"))
        || m.contains("truncation")
        || m.contains("data too long")
        || m.contains("foreign key constraint fails")
    {
        ErrorCategory::DataFormat
    } else if m.contains("out of memory")
        || (m.contains("memory") && m.contains("insufficient"))
        || (m.contains("disk") && m.contains("space"))
        || m.contains("lock wait timeout")
    {
        ErrorCategory::ResourceConstraint
    } else if (m.contains("invalid") && m.contains("environment"))
        || (m.contains("env") && m.contains("invalid"))
        || m.contains("malformed")
        || m.contains("configuration")
        || m.contains("syntax error")
    {
        ErrorCategory::Configuration
    } else {
        ErrorCategory::Unknown
    }
}

fn extract_sqlstate_and_chain(e: &anyhow::Error) -> (Option<String>, String) {
    let sqlstate = e.downcast_ref::<sqlx::Error>().and_then(|se| match se {
        sqlx::Error::Database(db) => db.code().map(|c| c.to_string()),
        _ => None,
    });
    let chain = format!("{:?}", e);
    (sqlstate, chain)
}

struct GuiApp {
    host: String,
    port: String,
    user: String,
    pass: String,
    // Cross-Database (optional)
    enable_dual: bool,
    host2: String,
    port2: String,
    user2: String,
    pass2: String,
    db2: String,
    tables2: Vec<String>,

    db: String,
    // Cached pools from last successful 'Load Tables' to reduce start-up latency
    pool1_cache: Option<MySqlPool>,
    pool2_cache: Option<MySqlPool>,

    // Schema metadata caches (TTL-based) to avoid repeated INFORMATION_SCHEMA and COUNT(*) on startup
    schema_cache: std::sync::Arc<
        std::sync::Mutex<HashMap<(String, String), (Option<TableColumns>, Vec<String>)>>,
    >,
    schema_cache_timestamp: std::sync::Arc<std::sync::Mutex<HashMap<(String, String), Instant>>>,

    tables: Vec<String>,
    table1_idx: usize,
    table2_idx: usize,
    algo: MatchingAlgorithm,
    // Advanced Matching (opt-in)
    advanced_enabled: bool,
    adv_level: Option<AdvLevel>,
    adv_threshold: f32,

    // Cascade Matching (run L1-L11 sequentially, L12 excluded)
    cascade_enabled: bool,
    cascade_missing_column_mode: name_matcher::matching::cascade::MissingColumnMode,
    cascade_geo_status: name_matcher::matching::cascade::GeoColumnStatus,
    cascade_status_message: String,
    // Note: Birthdate swap for cascade uses the unified `allow_birthdate_swap` field (Option B fix)
    path: String,
    fmt: FormatSel,
    mode: ModeSel,
    // Mode override tracking (Fix #6): shows when streaming is forced to in-memory
    effective_mode_override: Option<String>,
    pool_size: String,
    batch_size: String,
    mem_thresh: String,
    rayon_threads: String,
    gpu_streams: String,
    gpu_buffer_pool: bool,
    gpu_pinned_host: bool,
    use_gpu: bool,
    use_gpu_hash_join: bool,
    // Granular GPU hash-join controls
    use_gpu_build_hash: bool,
    use_gpu_probe_hash: bool,
    // Fuzzy GPU mode (Off/Auto/Force)
    fuzzy_gpu_mode: FuzzyGpuMode,

    // New options
    use_gpu_fuzzy_direct_hash: bool,
    use_gpu_levenshtein_full_scoring: bool, // GPU full scoring for Option 7
    direct_norm_fuzzy: bool,
    gpu_mem_mb: String, // fuzzy/in-memory GPU mem budget (metrics kernels and tiling)
    gpu_probe_mem_mb: String, // advisory GPU mem for probe batches (A1/A2 streaming)
    gpu_fuzzy_prep_mem_mb: String, // dedicated VRAM budget for fuzzy pre-pass hashing (A3/A4)

    // Dynamic GPU Auto-Tuning (GUI)
    enable_dynamic_gpu_tuning: bool,

    // Fuzzy threshold (percent 60..100) persisted across sessions
    fuzzy_threshold_pct: i32,
    // Allow month/day birthdate swap (env-driven)
    allow_birthdate_swap: bool,

    // Runtime GPU indicators
    gpu_build_active_now: bool,
    gpu_probe_active_now: bool,

    // Storage and system hints
    ssd_storage: bool,

    running: bool,
    progress: f32,
    eta_secs: u64,
    mem_used: u64,
    mem_avail: u64,
    processed: usize,
    total: usize,
    stage: String,
    batch_current: i64,
    rps: f32,
    last_tick: Option<std::time::Instant>,
    last_processed_prev: usize,
    // GPU status
    gpu_total_mb: u64,
    gpu_free_mb: u64,
    gpu_active: bool,

    a1_count: usize,
    a2_count: usize,
    csv_count: usize,
    status: String,

    // Diagnostics
    error_events: Vec<DiagEvent>,
    report_format: ReportFormat,
    last_action: String,
    // Advanced diagnostics
    schema_analysis_enabled: bool,
    log_buffer: Vec<String>,

    // CUDA diagnostics panel state
    cuda_diag_open: bool,
    cuda_diag_text: String,

    ctrl_cancel: Option<Arc<AtomicBool>>,
    ctrl_pause: Option<Arc<AtomicBool>>,

    tx: Option<Sender<Msg>>,
    rx: Receiver<Msg>,
    // Global run timing captured at button click and completion
    run_started_utc: Option<chrono::DateTime<chrono::Utc>>,
    run_ended_utc: Option<chrono::DateTime<chrono::Utc>>,

    // Optional external console tail child (Windows)
    console_child: Option<std::process::Child>,
}
impl GuiApp {
    fn read_fuzzy_threshold_pref() -> Option<i32> {
        let path = ".nm_fuzzy_threshold";
        match std::fs::read_to_string(path) {
            Ok(s) => {
                let s = s.trim();
                if let Some(p) = s.strip_suffix('%') {
                    p.parse::<i32>().ok().and_then(|v| {
                        if (60..=100).contains(&v) {
                            Some(v)
                        } else {
                            None
                        }
                    })
                } else {
                    s.parse::<i32>().ok().and_then(|v| {
                        if (60..=100).contains(&v) {
                            Some(v)
                        } else {
                            None
                        }
                    })
                }
            }
            Err(_) => None,
        }
    }
    fn save_fuzzy_threshold_pref(&self) {
        let _ = std::fs::write(
            ".nm_fuzzy_threshold",
            format!("{}%", self.fuzzy_threshold_pct),
        );
    }

    fn invalidate_db_caches(&mut self, reason: &str) {
        // Clear any cached database connection pools when settings change
        self.pool1_cache = None;
        self.pool2_cache = None;
        // Also clear schema caches to avoid stale metadata after DB settings change
        if let Ok(mut g) = self.schema_cache.lock() {
            g.clear();
        }
        if let Ok(mut g) = self.schema_cache_timestamp.lock() {
            g.clear();
        }
        self.status = format!("{} - connection & schema cache cleared", reason);
    }

    /// Ultra Performance Mode: Intelligent hardware-aware optimization with VRAM-aware execution mode selection
    /// This function performs comprehensive hardware detection, dataset size analysis, and applies maximum safe
    /// performance settings including aggressive GPU optimizations and intelligent execution mode selection.
    fn ultra_performance_mode(&mut self) {
        // Step 1: Detect hardware profile
        let profile_res = name_matcher::optimization::SystemProfile::detect();
        let profile = match profile_res {
            Ok(p) => p,
            Err(e) => {
                self.status = format!(
                    "Ultra Performance Mode failed: Hardware detection error: {}",
                    e
                );
                return;
            }
        };

        // Step 2: Query dataset size from selected tables
        let table1_name = if self.table1_idx < self.tables.len() {
            self.tables[self.table1_idx].clone()
        } else {
            self.status = "Ultra Performance Mode failed: No Table 1 selected".into();
            return;
        };

        let table2_name = if self.table2_idx < self.tables.len() {
            self.tables[self.table2_idx].clone()
        } else {
            self.status = "Ultra Performance Mode failed: No Table 2 selected".into();
            return;
        };

        // Get or create database pool
        let pool1 = match &self.pool1_cache {
            Some(p) => p.clone(),
            None => {
                self.status = "Ultra Performance Mode failed: No database connection. Click 'Load Tables' first.".into();
                return;
            }
        };

        // Query row counts (use fast estimate for speed)
        let rt = gui_runtime();
        let (table1_rows, table2_rows) = match rt.block_on(async {
            let c1 = get_person_count_fast(&pool1, &table1_name).await?;
            let c2 = get_person_count_fast(&pool1, &table2_name).await?;
            Ok::<(i64, i64), anyhow::Error>((c1, c2))
        }) {
            Ok((c1, c2)) => (c1, c2),
            Err(e) => {
                self.status = format!(
                    "Ultra Performance Mode failed: Could not query table sizes: {}",
                    e
                );
                return;
            }
        };

        // Step 3: Calculate dataset size (4KB per row estimate)
        let dataset_size_mb = ((table1_rows + table2_rows) * 4) / 1024; // 4KB per row → MB

        // Step 4: VRAM-aware execution mode selection
        let available_ram_mb = profile.ram.available_mb;
        let free_vram_mb = profile.gpu.as_ref().map(|g| g.vram_free_mb).unwrap_or(0);
        let total_available_memory_mb = available_ram_mb + free_vram_mb;

        // Apply 80% safety factor
        let safe_memory_threshold_mb = (total_available_memory_mb as f64 * 0.80) as u64;

        let selected_mode = if dataset_size_mb < safe_memory_threshold_mb as i64 {
            ModeSel::InMemory
        } else {
            ModeSel::Streaming
        };

        // Step 5: Apply aggressive GPU optimizations
        let cores = profile.cpu.cores.max(1);

        // Database connection pool:
        // - For remote MySQL, large pools often hurt (server thrash / timeouts).
        // - For local MySQL, we can safely scale higher.
        let db_is_local = Self::is_local_db_host(&self.host)
            && (!self.enable_dual || Self::is_local_db_host(&self.host2));
        let pool_cap = if db_is_local { 32 } else { 16 };
        let pool_size = (cores.saturating_mul(2)).clamp(8, pool_cap);
        self.pool_size = pool_size.to_string();

        // Apply mode-specific settings
        match selected_mode {
            ModeSel::InMemory => {
                // In-Memory mode: Maximize Rayon threads and GPU memory budget
                let icfg = name_matcher::optimization::calculate_inmemory_config(
                    &profile, self.algo, true,
                );
                if icfg.rayon_threads > 0 {
                    self.rayon_threads = icfg.rayon_threads.to_string();
                }

                // GPU settings
                if let Some(g) = &profile.gpu {
                    self.use_gpu = true;
                    self.gpu_total_mb = g.vram_total_mb;
                    self.gpu_free_mb = g.vram_free_mb;
                    self.gpu_streams = if g.vram_total_mb >= 6144 { "2" } else { "1" }.into();
                    self.gpu_buffer_pool = true;
                    self.gpu_pinned_host = g.vram_total_mb >= 4096;

                    // Ultra aggressive: 90% of free VRAM
                    let reserve_mb = if g.vram_free_mb >= 4096 { 128 } else { 64 };
                    let mut budget =
                        ((g.vram_free_mb.saturating_sub(reserve_mb)) as f64 * 0.90) as u64;
                    budget = budget.clamp(512, g.vram_total_mb.saturating_sub(reserve_mb));
                    self.gpu_mem_mb = budget.to_string();

                    // Probe batch: 90% of free VRAM (for streaming hash join if needed)
                    let mut probe = (g.vram_free_mb as f64 * 0.90) as u64;
                    probe = probe.clamp(512, 8192);
                    self.gpu_probe_mem_mb = probe.to_string();

                    // Fuzzy pre-pass: 40% of free VRAM
                    let mut prepass = ((g.vram_free_mb as f64) * 0.40) as u64;
                    prepass = prepass.clamp(128, 2048);
                    self.gpu_fuzzy_prep_mem_mb = prepass.to_string();

                    // Enable dynamic GPU tuning if VRAM >= 4GB
                    self.enable_dynamic_gpu_tuning = g.vram_total_mb >= 4096;
                }

                // Batch size and mem_thresh are not used in In-Memory mode, but set reasonable defaults
                self.batch_size = "100000".to_string();
                self.mem_thresh = "512".to_string();
            }
            ModeSel::Streaming => {
                // Streaming mode: Calculate maximum safe batch size based on 90% of available RAM
                let scfg = name_matcher::optimization::calculate_streaming_config(
                    &profile, self.algo, true,
                );
                if scfg.prefetch_pool_size > 0 {
                    // Streaming config doesn't expose rayon threads directly; keep current UI value.
                }

                // Ultra aggressive batch size: 90% of available RAM
                let target_batch_mem_mb = (available_ram_mb as f64 * 0.90) as u64;
                let batch_rows_est = ((target_batch_mem_mb * 256) as i64).clamp(10_000, 500_000);
                self.batch_size = batch_rows_est.to_string();

                // Memory soft minimum: 10% of total RAM
                let mem_soft_min = ((profile.ram.total_mb as f64 * 0.10) as u64).max(256);
                self.mem_thresh = mem_soft_min.to_string();

                // GPU settings
                if let Some(g) = &profile.gpu {
                    self.use_gpu = true;
                    self.gpu_total_mb = g.vram_total_mb;
                    self.gpu_free_mb = g.vram_free_mb;
                    self.gpu_streams = if g.vram_total_mb >= 6144 { "2" } else { "1" }.into();
                    self.gpu_buffer_pool = true;
                    self.gpu_pinned_host = g.vram_total_mb >= 4096;

                    // Ultra aggressive: 90% of free VRAM
                    let reserve_mb = if g.vram_free_mb >= 4096 { 128 } else { 64 };
                    let mut budget =
                        ((g.vram_free_mb.saturating_sub(reserve_mb)) as f64 * 0.90) as u64;
                    budget = budget.clamp(512, g.vram_total_mb.saturating_sub(reserve_mb));
                    self.gpu_mem_mb = budget.to_string();

                    // Probe batch: 90% of free VRAM
                    let mut probe = (g.vram_free_mb as f64 * 0.90) as u64;
                    probe = probe.clamp(512, 8192);
                    self.gpu_probe_mem_mb = probe.to_string();

                    // Fuzzy pre-pass: 40% of free VRAM
                    let mut prepass = ((g.vram_free_mb as f64) * 0.40) as u64;
                    prepass = prepass.clamp(128, 2048);
                    self.gpu_fuzzy_prep_mem_mb = prepass.to_string();

                    // Enable dynamic GPU tuning if VRAM >= 4GB
                    self.enable_dynamic_gpu_tuning = g.vram_total_mb >= 4096;
                }
            }
            ModeSel::Auto => {
                // Should not reach here, but handle gracefully
                self.status = "Ultra Performance Mode: Unexpected Auto mode selected".into();
                return;
            }
        }

        // Step 6: Enable all applicable GPU features for the selected algorithm
        let is_exact_algo = matches!(
            self.algo,
            MatchingAlgorithm::IdUuidYasIsMatchedInfnbd
                | MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd
        );
        let is_fuzzy_algo = matches!(
            self.algo,
            MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle
        );
        let is_house_algo = matches!(
            self.algo,
            MatchingAlgorithm::HouseholdGpu | MatchingAlgorithm::HouseholdGpuOpt6
        );
        let is_lev_algo = matches!(self.algo, MatchingAlgorithm::LevenshteinWeighted);
        let is_adv_exact = self.advanced_enabled
            && matches!(
                self.adv_level,
                Some(AdvLevel::L1BirthdateFullMiddle)
                    | Some(AdvLevel::L2BirthdateMiddleInitial)
                    | Some(AdvLevel::L3BirthdateNoMiddle)
                    | Some(AdvLevel::L4BarangayFullMiddle)
                    | Some(AdvLevel::L5BarangayMiddleInitial)
                    | Some(AdvLevel::L6BarangayNoMiddle)
                    | Some(AdvLevel::L7CityFullMiddle)
                    | Some(AdvLevel::L8CityMiddleInitial)
                    | Some(AdvLevel::L9CityNoMiddle)
            );
        let is_adv_fuzzy = self.advanced_enabled
            && matches!(
                self.adv_level,
                Some(AdvLevel::L10FuzzyBirthdateFullMiddle)
                    | Some(AdvLevel::L11FuzzyBirthdateNoMiddle)
            );
        let is_adv_house =
            self.advanced_enabled && matches!(self.adv_level, Some(AdvLevel::L12HouseholdMatching));

        // Cascade mode runs L1-L9 (exact) + L10-L11 (fuzzy), so needs both GPU feature sets
        let is_cascade_mode = self.advanced_enabled && self.cascade_enabled;

        // Enable GPU features based on algorithm
        if profile.gpu.is_some() {
            // Hash join for exact algorithms, Advanced L1-L9, OR Cascade mode (L1-L9)
            self.use_gpu_hash_join = is_exact_algo || is_adv_exact || is_cascade_mode;
            self.use_gpu_build_hash = self.use_gpu_hash_join;
            self.use_gpu_probe_hash = self.use_gpu_hash_join;

            // Fuzzy GPU features for fuzzy algorithms, Advanced L10-L11, OR Cascade mode (L10-L11)
            self.use_gpu_fuzzy_direct_hash =
                is_fuzzy_algo || is_lev_algo || is_adv_fuzzy || is_cascade_mode;
            self.use_gpu_levenshtein_full_scoring = is_lev_algo;

            // Enable GPU for fuzzy metrics in cascade mode (needed for L10-L11)
            if is_cascade_mode {
                self.use_gpu = true;
                self.fuzzy_gpu_mode = FuzzyGpuMode::Force;
                log::info!(
                    "[GUI] Ultra Performance: Cascade mode - enabled both hash join (L1-L9) and fuzzy GPU (L10-L11)"
                );
            }

            // Force fuzzy GPU mode for maximum throughput
            if is_fuzzy_algo || is_house_algo || is_adv_fuzzy || is_adv_house {
                self.fuzzy_gpu_mode = FuzzyGpuMode::Force;
            }
        }

        // Step 7: Apply selected execution mode
        self.mode = selected_mode;

        // Step 8: Generate detailed user feedback
        let mode_text = match selected_mode {
            ModeSel::InMemory => "In-Memory",
            ModeSel::Streaming => "Streaming",
            ModeSel::Auto => "Auto",
        };

        let mode_reasoning = if selected_mode == ModeSel::InMemory {
            format!(
                "Selected In-Memory mode because dataset ({:.2} GB) fits in available memory ({:.2} GB RAM + {:.2} GB VRAM = {:.2} GB total, 80% threshold = {:.2} GB)",
                dataset_size_mb as f64 / 1024.0,
                available_ram_mb as f64 / 1024.0,
                free_vram_mb as f64 / 1024.0,
                total_available_memory_mb as f64 / 1024.0,
                safe_memory_threshold_mb as f64 / 1024.0
            )
        } else {
            format!(
                "Selected Streaming mode because dataset ({:.2} GB) exceeds available memory ({:.2} GB RAM + {:.2} GB VRAM = {:.2} GB total, 80% threshold = {:.2} GB)",
                dataset_size_mb as f64 / 1024.0,
                available_ram_mb as f64 / 1024.0,
                free_vram_mb as f64 / 1024.0,
                total_available_memory_mb as f64 / 1024.0,
                safe_memory_threshold_mb as f64 / 1024.0
            )
        };

        let gpu_info = if let Some(g) = &profile.gpu {
            format!(
                "GPU: {} | {:.2} GB free / {:.2} GB total | Compute {}.{} | Budget: {} MB (90% of free VRAM)",
                g.device_name,
                g.vram_free_mb as f64 / 1024.0,
                g.vram_total_mb as f64 / 1024.0,
                g.compute_major,
                g.compute_minor,
                self.gpu_mem_mb
            )
        } else {
            "GPU: Not available (CPU-only mode)".to_string()
        };

        let gpu_features = if profile.gpu.is_some() {
            let mut feats: Vec<&str> = Vec::new();
            if self.use_gpu_hash_join {
                feats.push("GPU Hash Join");
            }
            if self.use_gpu_build_hash {
                feats.push("BuildHash");
            }
            if self.use_gpu_probe_hash {
                feats.push("ProbeHash");
            }
            if matches!(self.fuzzy_gpu_mode, FuzzyGpuMode::Force) {
                feats.push("FuzzyMetrics(Force)");
            }
            if self.use_gpu_fuzzy_direct_hash {
                feats.push("FuzzyPrepass");
            }
            if self.use_gpu_levenshtein_full_scoring {
                feats.push("LevFullScoring");
            }
            if self.enable_dynamic_gpu_tuning {
                feats.push("DynamicTuning");
            }
            if feats.is_empty() {
                "None enabled for this algorithm".into()
            } else {
                feats.join(", ")
            }
        } else {
            "N/A (no GPU)".to_string()
        };

        let gpu_utilization_target = if profile.gpu.is_some() {
            "Target GPU Utilization: 70-90% average (Note: 90-100% sustained is not achievable due to batch processing architecture)"
        } else {
            "N/A (CPU-only mode)"
        };

        self.status = format!(
            "🚀 Ultra Performance Mode Enabled\n\n\
            📊 Hardware Profile:\n\
            - CPU: {} cores\n\
            - RAM: {:.2} GB total, {:.2} GB available\n\
            - {}\n\n\
            📁 Dataset Analysis:\n\
            - Table 1: {} ({} rows)\n\
            - Table 2: {} ({} rows)\n\
            - Estimated size: {:.2} GB\n\n\
            ⚙️ Execution Mode: {}\n\
            {}\n\n\
            🔧 Performance Settings:\n\
            - DB Pool Size: {} connections\n\
            - Batch Size: {} rows (streaming)\n\
            - Memory Threshold: {} MB (streaming)\n\
            - GPU Memory Budget: {} MB\n\
            - GPU Probe Batch: {} MB\n\
            - GPU Fuzzy Prepass: {} MB\n\n\
            🎮 GPU Features Enabled:\n\
            {}\n\n\
            📈 Performance Expectations:\n\
            {}",
            cores,
            profile.ram.total_mb as f64 / 1024.0,
            profile.ram.available_mb as f64 / 1024.0,
            gpu_info,
            table1_name,
            table1_rows,
            table2_name,
            table2_rows,
            dataset_size_mb as f64 / 1024.0,
            mode_text,
            mode_reasoning,
            self.pool_size,
            self.batch_size,
            self.mem_thresh,
            self.gpu_mem_mb,
            self.gpu_probe_mem_mb,
            self.gpu_fuzzy_prep_mem_mb,
            gpu_features,
            gpu_utilization_target
        );

        log::info!(
            "Ultra Performance Mode applied: mode={}, dataset_size_mb={}, available_memory_mb={}",
            mode_text,
            dataset_size_mb,
            total_available_memory_mb
        );
    }

    fn cleanup_after_run(&mut self, reason: &str) {
        // NOTE: Database connection pools and schema caches are intentionally NOT cleared here.
        // Pools are expensive to recreate (TCP handshake, authentication, connection initialization).
        // Schema caches have a 300-second TTL and will expire naturally.
        // Stale connections are prevented by reducing idle_timeout to 30 seconds in src/db/connection.rs.
        // Clearing these resources would cause 3-5x performance regression on subsequent runs.

        // Truncate log buffer (keep last 50 entries)
        const LOG_BUFFER_KEEP: usize = 50;
        if self.log_buffer.len() > LOG_BUFFER_KEEP {
            let drop = self.log_buffer.len() - LOG_BUFFER_KEEP;
            self.log_buffer.drain(0..drop);
        }

        // Truncate error events (keep last 20)
        const ERROR_EVENTS_KEEP: usize = 20;
        if self.error_events.len() > ERROR_EVENTS_KEEP {
            let drop = self.error_events.len() - ERROR_EVENTS_KEEP;
            self.error_events.drain(0..drop);
        }

        log::info!("Post-run cleanup completed: {}", reason);
    }

    fn compute_cuda_diagnostics() -> String {
        let mut out = String::new();
        out.push_str("CUDA Diagnostics\n");
        #[cfg(feature = "gpu")]
        {
            use cudarc::driver::sys as cu;
            use std::ffi::CStr;
            unsafe {
                let mut init_ok = true;
                let r = cu::cuInit(0);
                if r != cu::CUresult::CUDA_SUCCESS {
                    out.push_str(&format!("cuInit failed: {:?}\n", r));
                    init_ok = false;
                }
                // Driver version
                let mut drv_ver: i32 = 0;
                let r = cu::cuDriverGetVersion(&mut drv_ver as *mut i32);
                if r == cu::CUresult::CUDA_SUCCESS {
                    out.push_str(&format!("Driver Version: {}\n", drv_ver));
                } else {
                    out.push_str(&format!("Driver Version: <error {:?}>\n", r));
                }
                // Device count
                let mut count: i32 = 0;
                let r = cu::cuDeviceGetCount(&mut count as *mut i32);
                if r == cu::CUresult::CUDA_SUCCESS {
                    out.push_str(&format!("Device Count: {}\n", count));
                } else {
                    out.push_str(&format!("Device Count: <error {:?}>\n", r));
                }
                for i in 0..count {
                    let mut dev: cu::CUdevice = 0;
                    let r = cu::cuDeviceGet(&mut dev as *mut _, i);
                    out.push_str(&format!("\nDevice {}:\n", i));
                    if r != cu::CUresult::CUDA_SUCCESS {
                        out.push_str(&format!("  cuDeviceGet error {:?}\n", r));
                        continue;
                    }
                    let mut name_buf = [0i8; 256];
                    let r = cu::cuDeviceGetName(name_buf.as_mut_ptr(), name_buf.len() as i32, dev);
                    if r == cu::CUresult::CUDA_SUCCESS {
                        let cstr = CStr::from_ptr(name_buf.as_ptr());
                        out.push_str(&format!("  Name: {}\n", cstr.to_string_lossy()));
                    } else {
                        out.push_str(&format!("  Name: <error {:?}>\n", r));
                    }
                    // Compute capability
                    let mut major: i32 = 0;
                    let mut minor: i32 = 0;
                    let r1 = cu::cuDeviceGetAttribute(
                        &mut major as *mut i32,
                        cu::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                        dev,
                    );
                    let r2 = cu::cuDeviceGetAttribute(
                        &mut minor as *mut i32,
                        cu::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                        dev,
                    );
                    if r1 == cu::CUresult::CUDA_SUCCESS && r2 == cu::CUresult::CUDA_SUCCESS {
                        out.push_str(&format!("  Compute Capability: {}.{}\n", major, minor));
                    } else {
                        out.push_str("  Compute Capability: <error>\n");
                    }
                    // Total memory
                    let mut total_mem: usize = 0;
                    let r = cu::cuDeviceTotalMem_v2(&mut total_mem as *mut usize, dev);
                    if r == cu::CUresult::CUDA_SUCCESS {
                        out.push_str(&format!("  Total Memory: {} MB\n", total_mem / 1024 / 1024));
                    } else {
                        out.push_str(&format!("  Total Memory: <error {:?}>\n", r));
                    }
                    // Try create context to get free/used
                    if init_ok {
                        if let Ok(ctx) = cudarc::driver::CudaContext::new(i as usize) {
                            let mut free: usize = 0;
                            let mut total: usize = 0;
                            let _ = cu::cuMemGetInfo_v2(
                                &mut free as *mut usize,
                                &mut total as *mut usize,
                            );
                            let used = total.saturating_sub(free);
                            out.push_str(&format!(
                                "  Free: {} MB | Used: {} MB\n",
                                free / 1024 / 1024,
                                used / 1024 / 1024
                            ));
                            drop(ctx);
                        } else {
                            out.push_str("  Context: unavailable (cannot create)\n");
                        }
                    }
                }
                out.push_str("\nTroubleshooting:\n - Ensure NVIDIA driver is installed and matches CUDA toolkit version.\n - Reboot after installing drivers.\n - If multiple GPUs, verify CUDA_VISIBLE_DEVICES.\n - On Windows, install latest Studio driver; on Linux, install kernel modules.\n - If this binary lacks GPU support, rebuild with `--features gpu`.\n");
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            out.push_str(
                "This build was compiled without GPU support. Rebuild with `--features gpu`.\n",
            );
        }
        out
    }
}

impl Default for GuiApp {
    fn default() -> Self {
        let (_tx, rx) = mpsc::channel();
        let thr = Self::read_fuzzy_threshold_pref().unwrap_or(95);
        let allow_birth_swap = name_matcher::matching::birthdate_matcher::allow_birthdate_swap();
        let suggested_rayon_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8)
            .min(12)
            .max(1);
        // Keep environment in sync with initial GUI state
        unsafe {
            std::env::set_var(
                "NAME_MATCHER_ALLOW_BIRTHDATE_SWAP",
                if allow_birth_swap { "1" } else { "0" },
            );
        }
        Self {
            host: "127.0.0.1".into(),
            port: "3306".into(),
            user: "root".into(),
            pass: "".into(),
            // dual-db defaults
            enable_dual: false,
            pool1_cache: None,
            pool2_cache: None,
            schema_cache: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
            schema_cache_timestamp: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),

            host2: "127.0.0.1".into(),
            port2: "3306".into(),
            user2: "".into(),
            pass2: "".into(),
            db2: "".into(),
            tables2: vec![],
            // primary db
            db: "duplicate_checker".into(),
            tables: vec![],
            table1_idx: 0,
            // CUDA diag defaults
            cuda_diag_open: false,
            cuda_diag_text: String::new(),

            table2_idx: 0,
            algo: MatchingAlgorithm::IdUuidYasIsMatchedInfnbd,
            // Advanced Matching defaults
            advanced_enabled: false,
            adv_level: None,
            adv_threshold: 0.95,

            // Cascade Matching defaults
            cascade_enabled: false,
            cascade_missing_column_mode:
                name_matcher::matching::cascade::MissingColumnMode::AutoSkip,
            cascade_geo_status: name_matcher::matching::cascade::GeoColumnStatus::default(),
            cascade_status_message: String::new(),
            // Note: Cascade uses unified `allow_birthdate_swap` field (set above)
            path: "matches.csv".into(),
            fmt: FormatSel::Csv,
            mode: ModeSel::Auto,
            effective_mode_override: None,
            pool_size: "16".into(),
            batch_size: "50000".into(),
            mem_thresh: "800".into(),
            rayon_threads: suggested_rayon_threads.to_string(),
            gpu_streams: "2".into(),
            gpu_buffer_pool: true,
            gpu_pinned_host: false,
            use_gpu: false,
            fuzzy_gpu_mode: FuzzyGpuMode::Off,

            use_gpu_hash_join: false,
            use_gpu_build_hash: true,

            use_gpu_probe_hash: true,
            // new options
            use_gpu_fuzzy_direct_hash: false,
            use_gpu_levenshtein_full_scoring: false,
            direct_norm_fuzzy: false,

            gpu_mem_mb: "512".into(),
            gpu_probe_mem_mb: "256".into(),
            gpu_fuzzy_prep_mem_mb: "256".into(),
            enable_dynamic_gpu_tuning: false,
            fuzzy_threshold_pct: thr,
            allow_birthdate_swap: allow_birth_swap,
            gpu_build_active_now: false,
            gpu_probe_active_now: false,
            ssd_storage: false,
            running: false,
            progress: 0.0,
            eta_secs: 0,
            mem_used: 0,
            mem_avail: 0,
            processed: 0,
            total: 0,
            stage: String::from("idle"),
            batch_current: 0,
            rps: 0.0,
            last_tick: None,
            last_processed_prev: 0,

            gpu_total_mb: 0,
            gpu_free_mb: 0,
            gpu_active: false,
            a1_count: 0,
            a2_count: 0,
            csv_count: 0,
            status: "Idle".into(),
            // Diagnostics
            error_events: Vec::new(),
            report_format: ReportFormat::Text,
            last_action: "idle".into(),
            schema_analysis_enabled: false,
            run_started_utc: None,
            run_ended_utc: None,

            log_buffer: Vec::with_capacity(200),

            ctrl_cancel: None,
            console_child: None,
            ctrl_pause: None,
            tx: None,
            rx,
        }
    }
}

impl GuiApp {
    fn is_local_db_host(host: &str) -> bool {
        let h = host.trim().to_ascii_lowercase();
        h == "localhost" || h == "127.0.0.1" || h == "::1"
    }

    /// Returns a user-friendly label for the given algorithm
    fn algorithm_label(algo: MatchingAlgorithm) -> &'static str {
        match algo {
            MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => {
                "Option 1: Deterministic Match (First + Last + Birthdate)"
            }
            MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => {
                "Option 2: Deterministic Match (First + Middle + Last + Birthdate)"
            }
            MatchingAlgorithm::Fuzzy => "Option 3: Fuzzy Match (with Middle Name)",
            MatchingAlgorithm::FuzzyNoMiddle => "Option 4: Fuzzy Match (without Middle Name)",
            MatchingAlgorithm::HouseholdGpu => "Option 5: Household Matching (Table 1 → Table 2)",
            MatchingAlgorithm::HouseholdGpuOpt6 => {
                "Option 6: Household Matching (Table 2 → Table 1)"
            }
            MatchingAlgorithm::LevenshteinWeighted => {
                "Option 7: Levenshtein-Weighted (SQL Equivalent)"
            }
        }
    }

    /// Returns a tooltip description for the given algorithm
    fn algorithm_tooltip(algo: MatchingAlgorithm) -> &'static str {
        match algo {
            MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => {
                "Exact match on first name, last name, and birthdate. Fast and deterministic."
            }
            MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => {
                "Exact match on first name, middle name, last name, and birthdate. Most precise deterministic option."
            }
            MatchingAlgorithm::Fuzzy => {
                "Fuzzy matching using Levenshtein, Jaro-Winkler, and Soundex on full names (including middle). Birthdate must match exactly. Use slider to set confidence threshold."
            }
            MatchingAlgorithm::FuzzyNoMiddle => {
                "Fuzzy matching on first and last names only (excludes middle name). Birthdate must match exactly. Use slider to set confidence threshold."
            }
            MatchingAlgorithm::HouseholdGpu => {
                "Household-level matching: groups Table 1 by UUID, Table 2 by hh_id. Fuzzy name matching with exact birthdate. Keeps households where >50% of members match. Denominator = Table 1 household size."
            }
            MatchingAlgorithm::HouseholdGpuOpt6 => {
                "Role-swapped household matching: groups Table 2 by hh_id, Table 1 by UUID. Fuzzy name matching with exact birthdate. Keeps households where >50% of members match. Denominator = Table 2 household size."
            }
            MatchingAlgorithm::LevenshteinWeighted => {
                "SQL-equivalent: Birthdate equality + blocking by (Soundex F+L) OR (3-char prefix F+L) OR (Soundex Middle), score = avg Levenshtein% over present fields. Threshold via slider."
            }
        }
    }

    fn ui_top(&mut self, ui: &mut egui::Ui) {
        // Spacing constants for consistent visual rhythm
        const SPACING_TINY: f32 = 2.0; // Between tightly related elements
        const SPACING_SMALL: f32 = 4.0; // Within sections
        const SPACING_MEDIUM: f32 = 6.0; // Between form rows / after headers
        const SPACING_LARGE: f32 = 8.0; // Between subsections / before separators
        const SPACING_SECTION: f32 = 12.0; // Between major sections / after separators
        const GRID_SPACING: [f32; 2] = [10.0, 6.0]; // Horizontal and vertical spacing for grids

        ui.heading("🔎 SRS-II Name Matching Application");
        ui.separator();
        ui.add_space(SPACING_SECTION);

        // Database Connection Section (Collapsible)
        egui::CollapsingHeader::new("📊 Database Connection")
            .default_open(true)
            .show(ui, |ui| {
                ui.add_space(SPACING_SMALL);
                ui.strong("Database 1 (Primary)");
                ui.add_space(SPACING_TINY);
                egui::Grid::new("db1_grid")
                    .num_columns(2)
                    .spacing(GRID_SPACING)
                    .show(ui, |ui| {
                        ui.label("Host:");
                        let r = ui.add(TextEdit::singleline(&mut self.host).hint_text("127.0.0.1"));
                        if r.changed() {
                            self.invalidate_db_caches("Database settings changed (DB1 Host)");
                        }
                        ui.end_row();

                        ui.label("Port:");
                        let r = ui.add(TextEdit::singleline(&mut self.port).hint_text("3306"));
                        if r.changed() {
                            self.invalidate_db_caches("Database settings changed (DB1 Port)");
                        }
                        ui.end_row();

                        ui.label("Username:");
                        let r = ui.add(TextEdit::singleline(&mut self.user).hint_text("root"));
                        if r.changed() {
                            self.invalidate_db_caches("Database settings changed (DB1 Username)");
                        }
                        ui.end_row();

                        ui.label("Password:");
                        let r = ui.add(
                            TextEdit::singleline(&mut self.pass)
                                .hint_text("password")
                                .password(true),
                        );
                        if r.changed() {
                            self.invalidate_db_caches("Database settings changed (DB1 Password)");
                        }
                        ui.end_row();

                        ui.label("Database:");
                        let r =
                            ui.add(TextEdit::singleline(&mut self.db).hint_text("database name"));
                        if r.changed() {
                            self.invalidate_db_caches("Database settings changed (DB1 Database)");
                        }
                        ui.end_row();
                    });
                ui.add_space(SPACING_LARGE);
                let resp = ui
                    .checkbox(&mut self.enable_dual, "Enable Cross-Database Matching")
                    .on_hover_text("Match Table 1 from Database 1 with Table 2 from Database 2");
                if resp.changed() {
                    self.invalidate_db_caches("Cross-database setting changed");
                }

                if self.enable_dual {
                    ui.add_space(SPACING_LARGE);
                    ui.strong("Database 2 (Secondary)");
                    ui.add_space(SPACING_TINY);
                    egui::Grid::new("db2_grid")
                        .num_columns(2)
                        .spacing(GRID_SPACING)
                        .show(ui, |ui| {
                            ui.label("Host:");
                            let r = ui
                                .add(TextEdit::singleline(&mut self.host2).hint_text("127.0.0.1"));
                            if r.changed() {
                                self.invalidate_db_caches("Database settings changed (DB2 Host)");
                            }
                            ui.end_row();

                            ui.label("Port:");
                            let r = ui.add(TextEdit::singleline(&mut self.port2).hint_text("3306"));
                            if r.changed() {
                                self.invalidate_db_caches("Database settings changed (DB2 Port)");
                            }
                            ui.end_row();

                            ui.label("Username:");
                            let r = ui.add(TextEdit::singleline(&mut self.user2).hint_text("root"));
                            if r.changed() {
                                self.invalidate_db_caches(
                                    "Database settings changed (DB2 Username)",
                                );
                            }
                            ui.end_row();

                            ui.label("Password:");
                            let r = ui.add(
                                TextEdit::singleline(&mut self.pass2)
                                    .hint_text("password")
                                    .password(true),
                            );
                            if r.changed() {
                                self.invalidate_db_caches(
                                    "Database settings changed (DB2 Password)",
                                );
                            }
                            ui.end_row();

                            ui.label("Database:");
                            let r = ui.add(
                                TextEdit::singleline(&mut self.db2).hint_text("database name"),
                            );
                            if r.changed() {
                                self.invalidate_db_caches(
                                    "Database settings changed (DB2 Database)",
                                );
                            }
                            ui.end_row();
                        });
                }
                ui.add_space(SPACING_SMALL);
            });

        ui.add_space(SPACING_LARGE);

        // Action toolbar
        ui.horizontal_wrapped(|ui| {
            ui.strong("Actions:");
            if ui
                .button("🔌 Test Connection")
                .on_hover_text("Verify database connectivity")
                .clicked()
            {
                self.test_connection();
            }
            if ui
                .button("📋 Load Tables")
                .on_hover_text("Query INFORMATION_SCHEMA to list available tables")
                .clicked()
            {
                self.load_tables();
            }
            if ui
                .button("📊 Estimate")
                .on_hover_text("Estimate memory usage and recommend execution mode")
                .clicked()
            {
                self.estimate();
            }
            ui.separator();
            if ui
                .button("📄 Generate .env Template")
                .on_hover_text("Create a .env.template file with all configurable keys")
                .clicked()
            {
                let dialog = rfd::FileDialog::new()
                    .add_filter("Template", &["template", "env", "txt"])
                    .set_file_name(".env.template");
                if let Some(path) = dialog.save_file() {
                    match name_matcher::util::envfile::write_env_template(
                        &path.display().to_string(),
                    ) {
                        Ok(_) => {
                            self.status = format!(".env template saved to {}", path.display());
                        }
                        Err(e) => {
                            self.status = format!("Failed to save .env template: {}", e);
                        }
                    }
                }
            }
            if ui
                .button("📂 Load .env File...")
                .on_hover_text("Load database credentials from a .env file")
                .clicked()
            {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("Env", &["env", "txt", "template"])
                    .pick_file()
                {
                    match name_matcher::util::envfile::load_env_file_from(
                        &path.display().to_string(),
                    ) {
                        Ok(map) => {
                            if let Some(v) = map.get("DB_HOST") {
                                self.host = v.clone();
                            }
                            if let Some(v) = map.get("DB_PORT") {
                                self.port = v.clone();
                            }
                            if let Some(v) = map.get("DB_USER") {
                                self.user = v.clone();
                            }
                            if let Some(v) = map.get("DB_PASSWORD") {
                                self.pass = v.clone();
                            }
                            if let Some(v) = map.get("DB_NAME") {
                                self.db = v.clone();
                            }
                            if let Some(v) = map.get("DB2_HOST") {
                                self.host2 = v.clone();
                                self.enable_dual = true;
                            }
                            if let Some(v) = map.get("DB2_PORT") {
                                self.port2 = v.clone();
                            }
                            if let Some(v) = map.get("DB2_USER") {
                                self.user2 = v.clone();
                            }
                            if let Some(v) = map.get("DB2_PASS") {
                                self.pass2 = v.clone();
                            }
                            if let Some(v) = map.get("DB2_DATABASE") {
                                self.db2 = v.clone();
                                self.enable_dual = true;
                            }
                            self.invalidate_db_caches(&format!(
                                "Loaded .env from {}",
                                path.display()
                            ));
                        }
                        Err(e) => {
                            self.status = format!("Failed to load .env: {}", e);
                        }
                    }
                }
            }
        });

        ui.add_space(SPACING_LARGE);
        ui.separator();
        ui.add_space(SPACING_SECTION);

        // Table Selection Section
        ui.strong("📋 Table Selection");
        ui.add_space(SPACING_MEDIUM);
        ui.horizontal_wrapped(|ui| {
            if self.enable_dual {
                // Dual DB: pick table1 from DB1 list, table2 from DB2 list
                if self.tables.is_empty() {
                    ui.label("⚠ Load DB1 tables first");
                } else {
                    ComboBox::from_label("Table 1 (DB1)")
                        .selected_text(
                            self.tables
                                .get(self.table1_idx)
                                .cloned()
                                .unwrap_or_default(),
                        )
                        .show_ui(ui, |ui| {
                            for (i, t) in self.tables.iter().enumerate() {
                                ui.selectable_value(&mut self.table1_idx, i, t);
                            }
                        });
                }
                if self.tables2.is_empty() {
                    ui.label("⚠ Load DB2 tables first");
                } else {
                    ComboBox::from_label("Table 2 (DB2)")
                        .selected_text(
                            self.tables2
                                .get(self.table2_idx)
                                .cloned()
                                .unwrap_or_default(),
                        )
                        .show_ui(ui, |ui| {
                            for (i, t) in self.tables2.iter().enumerate() {
                                ui.selectable_value(&mut self.table2_idx, i, t);
                            }
                        });
                }
            } else {
                if !self.tables.is_empty() {
                    ComboBox::from_label("Table 1")
                        .selected_text(
                            self.tables
                                .get(self.table1_idx)
                                .cloned()
                                .unwrap_or_default(),
                        )
                        .show_ui(ui, |ui| {
                            for (i, t) in self.tables.iter().enumerate() {
                                ui.selectable_value(&mut self.table1_idx, i, t);
                            }
                        });
                    ComboBox::from_label("Table 2")
                        .selected_text(
                            self.tables
                                .get(self.table2_idx)
                                .cloned()
                                .unwrap_or_default(),
                        )
                        .show_ui(ui, |ui| {
                            for (i, t) in self.tables.iter().enumerate() {
                                ui.selectable_value(&mut self.table2_idx, i, t);
                            }
                        });
                } else {
                    ui.label("⚠ Load tables first using the 'Load Tables' button above");
                }
            }
        });

        ui.add_space(SPACING_LARGE);
        ui.separator();
        ui.add_space(SPACING_SECTION);

        // Matching Configuration Section
        ui.strong("⚙️ Matching Configuration");
        ui.add_space(SPACING_MEDIUM);

        // Current Mode Visual Banner (Phase 3 UI Enhancement)
        {
            let (banner_color, banner_text, banner_icon) = if self.advanced_enabled {
                if self.cascade_enabled {
                    (
                        egui::Color32::from_rgb(255, 193, 7),
                        "CASCADE MODE: L1 → L2 → ... → L11",
                        "🔄",
                    )
                } else {
                    match self.adv_level {
                        Some(level) => {
                            let level_name = Self::level_short_name(level);
                            (
                                egui::Color32::from_rgb(175, 82, 222),
                                format!("ADVANCED: {}", level_name).leak() as &str,
                                "🎯",
                            )
                        }
                        None => (
                            egui::Color32::from_rgb(255, 152, 0),
                            "ADVANCED: Select a level...",
                            "⚠",
                        ),
                    }
                }
            } else {
                (egui::Color32::from_rgb(0, 174, 239), "STANDARD MODE", "📊")
            };

            egui::Frame::default()
                .fill(banner_color.gamma_multiply(0.15))
                .stroke(egui::Stroke::new(1.5, banner_color))
                .corner_radius(egui::CornerRadius::same(6))
                .inner_margin(egui::Margin::symmetric(12, 8))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.colored_label(banner_color, format!("{} {}", banner_icon, banner_text));
                    });
                });
            ui.add_space(SPACING_MEDIUM);
        }

        // Advanced Matching Toggle (prominently placed)
        ui.checkbox(&mut self.advanced_enabled, "Use Advanced Matching (12 Levels)")
            .on_hover_text("Enable Advanced Matching with 12 specialized levels (L1-L12) instead of the standard Options 1-7. Provides more granular control over exact and fuzzy matching criteria.");

        // Cascade Mode Toggle (only available when Advanced Matching is enabled)
        if self.advanced_enabled {
            ui.horizontal(|ui| {
                ui.add_space(20.0); // Indent to show it's a sub-option
                let cascade_changed = ui.checkbox(&mut self.cascade_enabled, "Run Cascade (L1-L11)")
                    .on_hover_text("Run all matching levels L1-L11 sequentially. Level 12 (Household Matching) is excluded. Generates separate output files per level (e.g., output_L01.csv, output_L02.csv, ...).")
                    .changed();

                if cascade_changed && self.cascade_enabled {
                    // Reset to a sensible default when enabling cascade
                    self.adv_level = None;
                    self.cascade_status_message = "Cascade mode: Will run L1-L11 sequentially (L12 excluded)".to_string();
                }
            });

            // Cascade Configuration (shown when cascade is enabled)
            if self.cascade_enabled {
                ui.add_space(SPACING_SMALL);
                ui.horizontal(|ui| {
                    ui.add_space(20.0);
                    ui.colored_label(
                        egui::Color32::from_rgb(255, 193, 7),
                        "⚡ Cascade Mode Active",
                    );
                    ui.label("- Runs L1→L2→...→L11 (L12 excluded)");
                });

                ui.horizontal(|ui| {
                    ui.add_space(20.0);
                    ui.label("Missing Column Handling:");
                    ComboBox::from_id_salt("cascade_missing_mode")
                        .selected_text(match self.cascade_missing_column_mode {
                            name_matcher::matching::cascade::MissingColumnMode::AutoSkip => "Auto-skip unavailable",
                            name_matcher::matching::cascade::MissingColumnMode::ManualSelect => "Manual level selection",
                            name_matcher::matching::cascade::MissingColumnMode::AbortOnMissing => "Abort if missing",
                        })
                        .width(200.0)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.cascade_missing_column_mode,
                                name_matcher::matching::cascade::MissingColumnMode::AutoSkip,
                                "Auto-skip unavailable"
                            ).on_hover_text("Skip levels that require missing columns (L4-L6 need barangay_code, L7-L9 need city_code)");
                            ui.selectable_value(
                                &mut self.cascade_missing_column_mode,
                                name_matcher::matching::cascade::MissingColumnMode::ManualSelect,
                                "Manual level selection"
                            ).on_hover_text("Only run levels that are available based on schema");
                            ui.selectable_value(
                                &mut self.cascade_missing_column_mode,
                                name_matcher::matching::cascade::MissingColumnMode::AbortOnMissing,
                                "Abort if missing"
                            ).on_hover_text("Abort the cascade if any geographic columns are missing");
                        });
                });

                // Display geo column status if we have info
                if !self.cascade_geo_status.has_barangay_code
                    || !self.cascade_geo_status.has_city_code
                {
                    ui.horizontal(|ui| {
                        ui.add_space(20.0);
                        ui.colored_label(egui::Color32::from_rgb(255, 152, 0), "⚠");
                        if !self.cascade_geo_status.has_barangay_code {
                            ui.label("barangay_code missing (L4-L6 will be skipped)");
                        }
                        if !self.cascade_geo_status.has_city_code {
                            ui.label("city_code missing (L7-L9 will be skipped)");
                        }
                    });
                }

                if !self.cascade_status_message.is_empty() {
                    ui.horizontal(|ui| {
                        ui.add_space(20.0);
                        ui.label(&self.cascade_status_message);
                    });
                }
            }
        } else {
            // Disable cascade when Advanced Matching is disabled
            self.cascade_enabled = false;
        }
        ui.add_space(SPACING_MEDIUM);

        // Mutually exclusive UI: Show either Options 1-7 OR Advanced Matching level selector
        // When cascade is enabled, hide the single-level selector
        if !self.advanced_enabled {
            // Original Mode: Algorithm Selection with improved labels
            ComboBox::from_label("Algorithm")
                .selected_text(Self::algorithm_label(self.algo))
                .width(500.0)
                .show_ui(ui, |ui| {
                    for algo in [
                        MatchingAlgorithm::IdUuidYasIsMatchedInfnbd,
                        MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd,
                        MatchingAlgorithm::Fuzzy,
                        MatchingAlgorithm::FuzzyNoMiddle,
                        MatchingAlgorithm::HouseholdGpu,
                        MatchingAlgorithm::HouseholdGpuOpt6,
                        MatchingAlgorithm::LevenshteinWeighted,
                    ] {
                        ui.selectable_value(&mut self.algo, algo, Self::algorithm_label(algo))
                            .on_hover_text(Self::algorithm_tooltip(algo));
                    }
                });
        } else if self.cascade_enabled {
            // Cascade mode is active - show info instead of level selector
            ui.horizontal(|ui| {
                ui.label("Running levels:");
                ui.colored_label(
                    egui::Color32::from_rgb(0, 191, 255),
                    "L1 → L2 → L3 → ... → L11",
                );
                ui.label("(L12 excluded)");
            });
        } else {
            // Advanced Mode: Level Selection (L1-L12)
            // Store previous level to detect changes
            let prev_level = self.adv_level;

            ComboBox::from_label("Advanced Matching Level")
                .selected_text(match self.adv_level {
                    Some(AdvLevel::L1BirthdateFullMiddle) => "L1: Exact (Full Middle + Birthdate)",
                    Some(AdvLevel::L2BirthdateMiddleInitial) => "L2: Exact (Middle Initial + Birthdate)",
                    Some(AdvLevel::L3BirthdateNoMiddle) => "L3: Exact (First+Last + Birthdate)",
                    Some(AdvLevel::L4BarangayFullMiddle) => "L4: Exact (Full Middle + Barangay Code)",
                    Some(AdvLevel::L5BarangayMiddleInitial) => "L5: Exact (Middle Initial + Barangay Code)",
                    Some(AdvLevel::L6BarangayNoMiddle) => "L6: Exact (First+Last + Barangay Code)",
                    Some(AdvLevel::L7CityFullMiddle) => "L7: Exact (Full Middle + City Code)",
                    Some(AdvLevel::L8CityMiddleInitial) => "L8: Exact (Middle Initial + City Code)",
                    Some(AdvLevel::L9CityNoMiddle) => "L9: Exact (First+Last + City Code)",
                    Some(AdvLevel::L10FuzzyBirthdateFullMiddle) => "L10: Fuzzy (Full Middle + Birthdate)",
                    Some(AdvLevel::L11FuzzyBirthdateNoMiddle) => "L11: Fuzzy (No Middle + Birthdate)",
                    Some(AdvLevel::L12HouseholdMatching) => "L12: Household Matching (Table2 → Table1)",
                    None => "Select Level...",
                })
                .width(500.0)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.adv_level, Some(AdvLevel::L1BirthdateFullMiddle), "L1: Exact (Full Middle + Birthdate)")
                        .on_hover_text("Exact match on first name, last name, full middle name, and birthdate");
                    ui.selectable_value(&mut self.adv_level, Some(AdvLevel::L2BirthdateMiddleInitial), "L2: Exact (Middle Initial + Birthdate)")
                        .on_hover_text("Exact match on first name, last name, middle initial(s), and birthdate");
                    ui.selectable_value(&mut self.adv_level, Some(AdvLevel::L3BirthdateNoMiddle), "L3: Exact (First+Last + Birthdate)")
                        .on_hover_text("Exact match on first name, last name, and birthdate (no middle name required)");
                    ui.selectable_value(&mut self.adv_level, Some(AdvLevel::L4BarangayFullMiddle), "L4: Exact (Full Middle + Barangay Code)")
                        .on_hover_text("Exact match on first name, last name, full middle name, and barangay code");
                    ui.selectable_value(&mut self.adv_level, Some(AdvLevel::L5BarangayMiddleInitial), "L5: Exact (Middle Initial + Barangay Code)")
                        .on_hover_text("Exact match on first name, last name, middle initial(s), and barangay code");
                    ui.selectable_value(&mut self.adv_level, Some(AdvLevel::L6BarangayNoMiddle), "L6: Exact (First+Last + Barangay Code)")
                        .on_hover_text("Exact match on first name, last name, and barangay code (no middle name required)");
                    ui.selectable_value(&mut self.adv_level, Some(AdvLevel::L7CityFullMiddle), "L7: Exact (Full Middle + City Code)")
                        .on_hover_text("Exact match on first name, last name, full middle name, and city code");
                    ui.selectable_value(&mut self.adv_level, Some(AdvLevel::L8CityMiddleInitial), "L8: Exact (Middle Initial + City Code)")
                        .on_hover_text("Exact match on first name, last name, middle initial(s), and city code");
                    ui.selectable_value(&mut self.adv_level, Some(AdvLevel::L9CityNoMiddle), "L9: Exact (First+Last + City Code)")
                        .on_hover_text("Exact match on first name, last name, and city code (no middle name required)");
                    ui.selectable_value(&mut self.adv_level, Some(AdvLevel::L10FuzzyBirthdateFullMiddle), "L10: Fuzzy (Full Middle + Birthdate)")
                        .on_hover_text("Fuzzy matching on names with exact birthdate match. Requires full middle name on both sides.");
                    ui.selectable_value(&mut self.adv_level, Some(AdvLevel::L11FuzzyBirthdateNoMiddle), "L11: Fuzzy (No Middle + Birthdate)")
                        .on_hover_text("Fuzzy matching on first and last names only with exact birthdate match. Excludes middle names.");
                    ui.selectable_value(&mut self.adv_level, Some(AdvLevel::L12HouseholdMatching), "L12: Household Matching (Table2 → Table1)")
                        .on_hover_text("Household-level fuzzy matching from Table 2 to Table 1. Groups by household and applies fuzzy person-level matching.");
                });

            // State management: Reset GPU flags when switching between exact (L1-L9) and fuzzy (L10-L12) levels
            if prev_level != self.adv_level {
                let is_exact_level = matches!(
                    self.adv_level,
                    Some(AdvLevel::L1BirthdateFullMiddle)
                        | Some(AdvLevel::L2BirthdateMiddleInitial)
                        | Some(AdvLevel::L3BirthdateNoMiddle)
                        | Some(AdvLevel::L4BarangayFullMiddle)
                        | Some(AdvLevel::L5BarangayMiddleInitial)
                        | Some(AdvLevel::L6BarangayNoMiddle)
                        | Some(AdvLevel::L7CityFullMiddle)
                        | Some(AdvLevel::L8CityMiddleInitial)
                        | Some(AdvLevel::L9CityNoMiddle)
                );
                let is_fuzzy_level = matches!(
                    self.adv_level,
                    Some(AdvLevel::L10FuzzyBirthdateFullMiddle)
                        | Some(AdvLevel::L11FuzzyBirthdateNoMiddle)
                        | Some(AdvLevel::L12HouseholdMatching)
                );

                // When switching to exact levels (L1-L9): disable fuzzy GPU, keep hash join as-is
                if is_exact_level {
                    self.use_gpu = false;
                    self.fuzzy_gpu_mode = FuzzyGpuMode::Off;
                    log::info!("[GUI] Switched to exact level - disabled GPU fuzzy metrics");
                }
                // When switching to fuzzy levels (L10-L12): disable hash join, keep fuzzy GPU as-is
                else if is_fuzzy_level {
                    self.use_gpu_hash_join = false;
                    self.use_gpu_build_hash = false;
                    self.use_gpu_probe_hash = false;
                    log::info!("[GUI] Switched to fuzzy level - disabled GPU hash join");
                }
            }
        }

        // Advanced Matching Configuration Controls (shown when Advanced mode is enabled)
        if self.advanced_enabled {
            ui.add_space(SPACING_SMALL);

            // Fuzzy Threshold for L10-L12 (Fuzzy levels) or Cascade mode (which includes L10-L11)
            let needs_threshold = self.cascade_enabled
                || matches!(
                    self.adv_level,
                    Some(AdvLevel::L10FuzzyBirthdateFullMiddle)
                        | Some(AdvLevel::L11FuzzyBirthdateNoMiddle)
                        | Some(AdvLevel::L12HouseholdMatching)
                );
            if needs_threshold {
                ui.horizontal(|ui| {
                    ui.label("Fuzzy Threshold:");
                    let mut pct = (self.adv_threshold * 100.0).round() as i32;
                    if ui
                        .add(
                            egui::Slider::new(&mut pct, 60..=100)
                                .suffix("%")
                                .text("Confidence"),
                        )
                        .changed()
                    {
                        self.adv_threshold = (pct as f32) / 100.0;
                    }
                    ui.label(format!("{}%", pct));
                });
                // Contextual help text for threshold applicability
                ui.horizontal(|ui| {
                    ui.add_space(20.0);
                    if self.cascade_enabled {
                        ui.colored_label(
                            egui::Color32::from_rgb(152, 152, 157),
                            "ℹ Applies to L10-L11 (fuzzy matching levels in cascade)",
                        );
                    } else {
                        ui.colored_label(
                            egui::Color32::from_rgb(152, 152, 157),
                            "ℹ Applies to this level",
                        );
                    }
                });
                ui.add_space(SPACING_TINY);
            }
        }

        ui.add_space(SPACING_MEDIUM);

        // Algorithm-specific options (Original mode only)
        if !self.advanced_enabled {
            let a1a2 = matches!(
                self.algo,
                MatchingAlgorithm::IdUuidYasIsMatchedInfnbd
                    | MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd
            );
            if a1a2 {
                ui.checkbox(&mut self.direct_norm_fuzzy, "Apply Fuzzy-style normalization")
                    .on_hover_text("Lowercases and strips punctuation; treats hyphens as spaces. Aligns A1/A2 normalization with Fuzzy algorithms.");
            }
        }

        // Fuzzy threshold slider (Original mode only - Advanced mode has its own threshold controls)
        if !self.advanced_enabled {
            let fuzzy_enabled = matches!(
                self.algo,
                MatchingAlgorithm::Fuzzy
                    | MatchingAlgorithm::FuzzyNoMiddle
                    | MatchingAlgorithm::HouseholdGpu
                    | MatchingAlgorithm::HouseholdGpuOpt6
                    | MatchingAlgorithm::LevenshteinWeighted
            );
            ui.horizontal(|ui| {
                ui.label("Fuzzy Confidence Threshold:");
                ui.add_enabled(
                    fuzzy_enabled,
                    egui::Slider::new(&mut self.fuzzy_threshold_pct, 60..=100)
                        .suffix("%")
                        .text("Confidence"),
                );
                if !fuzzy_enabled {
                    ui.weak("(not applicable to deterministic algorithms)");
                } else {
                    ui.label(format!("{}%", self.fuzzy_threshold_pct));
                }
            });
        }

        // Birthdate swap toggle (month/day) used by fuzzy birthdate levels
        // Single unified toggle for both standalone and cascade modes (Option B fix)
        let swap_applicable = if self.advanced_enabled {
            // In cascade mode, swap applies to L10-L11; in single-level mode, check selected level
            self.cascade_enabled
                || matches!(
                    self.adv_level,
                    Some(AdvLevel::L10FuzzyBirthdateFullMiddle)
                        | Some(AdvLevel::L11FuzzyBirthdateNoMiddle)
                        | Some(AdvLevel::L12HouseholdMatching)
                )
        } else {
            matches!(
                self.algo,
                MatchingAlgorithm::Fuzzy
                    | MatchingAlgorithm::FuzzyNoMiddle
                    | MatchingAlgorithm::LevenshteinWeighted
                    | MatchingAlgorithm::HouseholdGpu
                    | MatchingAlgorithm::HouseholdGpuOpt6
            )
        };
        ui.horizontal(|ui| {
            let mut val = self.allow_birthdate_swap;
            let resp = ui
                .add_enabled(
                    swap_applicable,
                    egui::Checkbox::new(&mut val, "Allow birthdate month/day swap (12/04 <-> 04/12)"),
                )
                .on_hover_text("Permits matching dates with swapped month/day when valid. In cascade mode, applies to L10-L11 fuzzy levels only.");
            if resp.changed() {
                self.allow_birthdate_swap = val;
                unsafe {
                    std::env::set_var("NAME_MATCHER_ALLOW_BIRTHDATE_SWAP", if val { "1" } else { "0" });
                }
                log::info!("[GUI] Birthdate swap {}", if val { "enabled" } else { "disabled" });
            }
            if !swap_applicable {
                ui.weak(" (only affects fuzzy birthdate algorithms)");
            } else if self.cascade_enabled {
                ui.weak(" (applies to L10-L11 in cascade)");
            }
        });

        ui.add_space(SPACING_MEDIUM);

        // Execution mode with override detection (Fix #6)
        // Check if mode will be overridden based on algorithm selection
        let mode_override_reason = if matches!(self.mode, ModeSel::Streaming) {
            if !self.advanced_enabled {
                // Original mode algorithms
                if matches!(
                    self.algo,
                    MatchingAlgorithm::Fuzzy
                        | MatchingAlgorithm::FuzzyNoMiddle
                        | MatchingAlgorithm::LevenshteinWeighted
                ) {
                    Some("Fuzzy algorithms require In-Memory mode".to_string())
                } else {
                    None
                }
            } else if !self.cascade_enabled {
                // Advanced mode - check selected level
                None // Advanced streaming is generally supported
            } else {
                None // Cascade mode
            }
        } else {
            None
        };
        self.effective_mode_override = mode_override_reason.clone();

        ui.horizontal(|ui| {
            ui.label("Execution Mode:");
            ComboBox::from_id_salt("mode_combo")
                .selected_text(match self.mode { ModeSel::Auto=>"Auto", ModeSel::Streaming=>"Streaming", ModeSel::InMemory=>"In-Memory" })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.mode, ModeSel::Auto, "Auto")
                        .on_hover_text("Auto defaults to In‑Memory (no row-count threshold). Explicit Streaming/In‑Memory selections override.");
                    ui.selectable_value(&mut self.mode, ModeSel::Streaming, "Streaming")
                        .on_hover_text("Index smaller table, stream larger table in chunks (memory-efficient)");
                    ui.selectable_value(&mut self.mode, ModeSel::InMemory, "In-Memory")
                        .on_hover_text("Load both tables into memory (faster but requires more RAM)");
                });

            // Show override warning when mode will be forced (Fix #6)
            if let Some(reason) = &self.effective_mode_override {
                ui.colored_label(egui::Color32::from_rgb(255, 193, 7), format!("⚠ {}", reason));
            }
        });

        ui.add_space(SPACING_LARGE);
        ui.separator();
        ui.add_space(SPACING_SECTION);

        // Output Settings Section
        ui.strong("💾 Output Settings");
        ui.add_space(SPACING_MEDIUM);
        ui.horizontal(|ui| {
            ui.label("Output File:");
            ui.add(
                TextEdit::singleline(&mut self.path)
                    .hint_text("matches.csv")
                    .desired_width(300.0),
            );
            if ui.button("📁 Browse").clicked() {
                let mut dialog = rfd::FileDialog::new();
                match self.fmt {
                    FormatSel::Csv => {
                        dialog = dialog.add_filter("CSV", &["csv"]);
                    }
                    FormatSel::Xlsx => {
                        dialog = dialog.add_filter("Excel", &["xlsx"]);
                    }
                    FormatSel::Both => {
                        dialog = dialog
                            .add_filter("CSV", &["csv"])
                            .add_filter("Excel", &["xlsx"]);
                    }
                }
                if let Some(path) = dialog.set_file_name(&self.path).save_file() {
                    self.path = path.display().to_string();
                }
            }
        });
        ui.horizontal(|ui| {
            ui.label("Output Format:");
            ComboBox::from_id_salt("format_combo")
                .selected_text(match self.fmt {
                    FormatSel::Csv => "CSV",
                    FormatSel::Xlsx => "XLSX",
                    FormatSel::Both => "Both (CSV + XLSX)",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.fmt, FormatSel::Csv, "CSV");
                    ui.selectable_value(&mut self.fmt, FormatSel::Xlsx, "XLSX");
                    ui.selectable_value(&mut self.fmt, FormatSel::Both, "Both (CSV + XLSX)");
                });
        });
        ui.add_space(SPACING_LARGE);
        ui.separator();
        ui.add_space(SPACING_SECTION);

        // Configuration Summary Panel (Phase 3 UI Enhancement)
        egui::CollapsingHeader::new("📋 Configuration Summary")
            .default_open(true)
            .show(ui, |ui| {
                self.render_config_summary(ui);
            });

        ui.add_space(SPACING_LARGE);
        ui.separator();
        ui.add_space(SPACING_SECTION);

        // Advanced Settings (Collapsible)
        egui::CollapsingHeader::new("⚡ Advanced Settings")
            .default_open(false)
            .show(ui, |ui| {
                ui.add_space(SPACING_MEDIUM);

                // GPU Hash Join (Options 1 & 2 OR Advanced L1-L9)
                // Explicitly exclude fuzzy/household levels (L10-L12) in Advanced mode
                let is_household_level = self.advanced_enabled && matches!(self.adv_level,
                    Some(AdvLevel::L10FuzzyBirthdateFullMiddle) | Some(AdvLevel::L11FuzzyBirthdateNoMiddle) | Some(AdvLevel::L12HouseholdMatching));
                let show_gpu_hash = !is_household_level && (
                    matches!(self.algo, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd | MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd) ||
                    (self.advanced_enabled && matches!(self.adv_level,
                        Some(AdvLevel::L1BirthdateFullMiddle) | Some(AdvLevel::L2BirthdateMiddleInitial) | Some(AdvLevel::L3BirthdateNoMiddle) |
                        Some(AdvLevel::L4BarangayFullMiddle) | Some(AdvLevel::L5BarangayMiddleInitial) | Some(AdvLevel::L6BarangayNoMiddle) |
                        Some(AdvLevel::L7CityFullMiddle) | Some(AdvLevel::L8CityMiddleInitial) | Some(AdvLevel::L9CityNoMiddle))));
                if show_gpu_hash {
                    let header_text = if self.advanced_enabled {
                        "🚀 GPU Hash Join (Advanced L1-L9)"
                    } else {
                        "🚀 GPU Hash Join (Options 1 & 2)"
                    };
                    ui.strong(header_text);
                    ui.add_space(SPACING_SMALL);
                    let prev_gpu_hash_join = self.use_gpu_hash_join;
                    if ui.checkbox(&mut self.use_gpu_hash_join, "Enable GPU Hash Join")
                        .on_hover_text("Accelerate deterministic joins via GPU pre-hash + CPU verification. Automatically enables GPU build and probe hashing. Requires CUDA build. Falls back to CPU automatically if unavailable.")
                        .changed() && self.use_gpu_hash_join && !prev_gpu_hash_join {
                        // Auto-enable GPU build and probe hash when GPU Hash Join is enabled
                        self.use_gpu_build_hash = true;
                        self.use_gpu_probe_hash = true;
                        log::info!("[GUI] GPU Hash Join enabled - auto-enabled build and probe hashing");
                    }

                    // Show visual feedback when GPU Hash Join is enabled
                    if self.use_gpu_hash_join {
                        ui.horizontal(|ui| {
                            ui.colored_label(egui::Color32::from_rgb(76, 217, 100), "✓ GPU Hash Join Enabled");
                            if self.use_gpu_build_hash {
                                ui.colored_label(egui::Color32::from_rgb(0, 174, 239), "• Build Hash");
                            }
                            if self.use_gpu_probe_hash {
                                ui.colored_label(egui::Color32::from_rgb(0, 174, 239), "• Probe Hash");
                            }
                        });
                    }

                    if self.use_gpu_hash_join {
                        ui.add_space(SPACING_SMALL);
                        egui::Grid::new("gpu_hash_join_grid").num_columns(2).spacing(GRID_SPACING).show(ui, |ui| {
                            ui.label("GPU for Build Hash:");
                            ui.checkbox(&mut self.use_gpu_build_hash, "")
                                .on_hover_text("Use GPU to hash the smaller (build) table");
                            ui.end_row();

                            ui.label("GPU for Probe Hash:");
                            ui.checkbox(&mut self.use_gpu_probe_hash, "")
                                .on_hover_text("Use GPU to hash the larger (probe) table");
                            ui.end_row();

                            ui.label("Probe GPU Mem (MB):");
                            ui.add(TextEdit::singleline(&mut self.gpu_probe_mem_mb).desired_width(80.0))
                                .on_hover_text("VRAM budget for probe-side hashing batches");
                            ui.end_row();
                        });

                        if self.gpu_total_mb > 0 {
                            ui.label(format!("GPU Status: Build {} | Probe {}",
                                if self.gpu_build_active_now { "✓ active" } else { "○ idle" },
                                if self.gpu_probe_active_now { "✓ active" } else { "○ idle" }
                            ));
                        }
                    }
                    ui.add_space(SPACING_LARGE);
                    ui.separator();
                }

                // Performance & Streaming Settings
                ui.strong("⚙️ Performance & Streaming");
                ui.add_space(SPACING_SMALL);

                let streaming_enabled = !matches!(self.mode, ModeSel::InMemory);
                egui::Grid::new("perf_stream_grid").num_columns(2).spacing(GRID_SPACING).show(ui, |ui| {
                    ui.label("Database Pool Size:");
                    ui.add(TextEdit::singleline(&mut self.pool_size).desired_width(80.0))
                        .on_hover_text("Maximum number of concurrent database connections in the connection pool");
                    ui.end_row();

                    ui.label("Rayon Threads:");
                    ui.add(TextEdit::singleline(&mut self.rayon_threads).desired_width(80.0))
                        .on_hover_text("CPU parallelism for matching. 0 = leave default. Too high can hurt performance on remote MySQL workloads (oversubscription).");
                    ui.end_row();

                    ui.label("GPU Streams:");
                    ui.add_enabled(self.use_gpu && cfg!(feature = "gpu"), TextEdit::singleline(&mut self.gpu_streams).desired_width(80.0))
                        .on_hover_text("CUDA streams for overlap (1 = off). 2 is a safe start for RTX 4050; try 3 if stable.");
                    if !(self.use_gpu && cfg!(feature = "gpu")) { ui.weak("(requires GPU build)"); }
                    ui.end_row();

                    ui.label("GPU Buffer Pool:");
                    ui.add_enabled(
                        self.use_gpu && cfg!(feature = "gpu"),
                        egui::Checkbox::new(&mut self.gpu_buffer_pool, ""),
                    )
                    .on_hover_text("Reuse GPU buffers to reduce allocations (usually faster).");
                    if !(self.use_gpu && cfg!(feature = "gpu")) { ui.weak("(requires GPU build)"); }
                    ui.end_row();

                    ui.label("GPU Pinned Host:");
                    ui.add_enabled(
                        self.use_gpu && cfg!(feature = "gpu"),
                        egui::Checkbox::new(&mut self.gpu_pinned_host, ""),
                    )
                    .on_hover_text("Use pinned host staging to improve transfers/overlap (may increase RAM pressure).");
                    if !(self.use_gpu && cfg!(feature = "gpu")) { ui.weak("(requires GPU build)"); }
                    ui.end_row();

                    ui.label("Batch Size:");
                    let resp_batch = ui.add_enabled(streaming_enabled, TextEdit::singleline(&mut self.batch_size).desired_width(80.0));
                    if !streaming_enabled {
                        resp_batch.on_hover_text("⚠ Only applies in Streaming mode. In-Memory mode loads entire dataset at once.");
                    } else {
                        resp_batch.on_hover_text("Number of rows fetched per chunk in streaming mode");
                    }
                    ui.end_row();

                    ui.label("Memory Threshold (MB):");
                    let resp_mem = ui.add_enabled(streaming_enabled, TextEdit::singleline(&mut self.mem_thresh).desired_width(80.0));
                    if !streaming_enabled {
                        resp_mem.on_hover_text("⚠ Only applies in Streaming mode. In-Memory mode loads entire dataset at once.");
                    } else {
                        resp_mem.on_hover_text("Soft minimum free memory before reducing batch size dynamically");
                    }
                    ui.end_row();

                    ui.label("Storage Type:");
                    ui.checkbox(&mut self.ssd_storage, "SSD Storage")
                        .on_hover_text("Optimize flush frequency for SSD (uses larger buffered writes)");
                    ui.end_row();
                });

                ui.add_space(SPACING_LARGE);
                ui.separator();

                // GPU Acceleration for Fuzzy Algorithms (Options 3-7 OR Advanced L10-L12)
                // Ensure controls are shown only for compatible levels
                let is_adv_fuzzy = self.advanced_enabled && matches!(self.adv_level,
                    Some(AdvLevel::L10FuzzyBirthdateFullMiddle) | Some(AdvLevel::L11FuzzyBirthdateNoMiddle));
                let is_adv_house = self.advanced_enabled && matches!(self.adv_level, Some(AdvLevel::L12HouseholdMatching));
                let supports_fuzzy_gpu_algs = matches!(self.algo,
                    MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle | MatchingAlgorithm::HouseholdGpu | MatchingAlgorithm::HouseholdGpuOpt6);
                let supports_fuzzy_prepass_algs = matches!(self.algo,
                    MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle | MatchingAlgorithm::LevenshteinWeighted);

                // Show fuzzy GPU section iff:
                // - Advanced mode: L10/L11/L12
                // - Original mode: fuzzy-capable algorithms
                let show_fuzzy_gpu_section = if self.advanced_enabled {
                    is_adv_fuzzy || is_adv_house
                } else {
                    supports_fuzzy_gpu_algs || supports_fuzzy_prepass_algs
                };

                if show_fuzzy_gpu_section {
                    let header_text = if self.advanced_enabled {
                        "🎮 GPU Acceleration (Advanced L10-L12)"
                    } else {
                        "🎮 GPU Acceleration (Fuzzy Algorithms)"
                    };
                    ui.strong(header_text);
                    ui.add_space(SPACING_SMALL);

                    let prev_use_gpu = self.use_gpu;
                    if ui.checkbox(&mut self.use_gpu, "Enable GPU (CUDA)")
                        .on_hover_text("Enable CUDA acceleration for fuzzy matching algorithms. Automatically enables all GPU-accelerated features. Falls back to CPU if unavailable.")
                        .changed() && self.use_gpu && !prev_use_gpu {
                        // Auto-enable all GPU features when GPU is enabled
                        self.use_gpu_fuzzy_direct_hash = true;
                        if matches!(self.algo, MatchingAlgorithm::LevenshteinWeighted) {
                            self.use_gpu_levenshtein_full_scoring = true;
                        }
                        self.fuzzy_gpu_mode = FuzzyGpuMode::Auto;
                        log::info!("[GUI] GPU enabled - auto-enabled all GPU features");
                    }
                    // If GPU was just disabled, ensure dynamic tuner is stopped
                    if prev_use_gpu && !self.use_gpu {
                        #[cfg(feature = "gpu")]
                        {
                            name_matcher::matching::dyn_tuner_ensure_started(false);
                            name_matcher::matching::dyn_tuner_stop();
                        }
                    }

                    // Show visual feedback when GPU is enabled
                    if self.use_gpu {
                        ui.horizontal(|ui| {
                            ui.colored_label(egui::Color32::from_rgb(76, 217, 100), "✓ GPU Enabled");
                            if self.use_gpu_fuzzy_direct_hash {
                                ui.colored_label(egui::Color32::from_rgb(175, 82, 222), "• Pre-pass Active");
                            }
                            if self.use_gpu_levenshtein_full_scoring {
                                ui.colored_label(egui::Color32::from_rgb(175, 82, 222), "• Full Scoring Active");
                            }
                        });
                    }

                    if !cfg!(feature = "gpu") {
                        ui.colored_label(egui::Color32::from_rgb(255, 165, 0), "⚠ This build was compiled without GPU support. Rebuild with `--features gpu` to enable.");
                    }

                    ui.add_space(SPACING_SMALL);

                    egui::Grid::new("gpu_accel_grid").num_columns(2).spacing(GRID_SPACING).show(ui, |ui| {
                        ui.label("GPU Memory Budget (MB):");
                        let enable_metrics = self.use_gpu && (if self.advanced_enabled { is_adv_fuzzy || is_adv_house } else { supports_fuzzy_gpu_algs });
                        ui.add_enabled(enable_metrics, TextEdit::singleline(&mut self.gpu_mem_mb).desired_width(80.0))
                            .on_hover_text("VRAM budget for GPU kernels during fuzzy matching");
                        if !enable_metrics { ui.weak("(not applicable to this level)"); }
                        ui.end_row();

                        ui.label("Fuzzy GPU Metrics Mode:");
                        ui.add_enabled_ui(enable_metrics, |ui| {
                            ComboBox::from_id_salt("fuzzy_gpu_mode")
                                .selected_text(match self.fuzzy_gpu_mode {
                                    FuzzyGpuMode::Off => "Off",
                                    FuzzyGpuMode::Auto => "Auto",
                                    FuzzyGpuMode::Force => "Force"
                                })
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut self.fuzzy_gpu_mode, FuzzyGpuMode::Off, "Off")
                                        .on_hover_text("Disable GPU metrics computation");
                                    ui.selectable_value(&mut self.fuzzy_gpu_mode, FuzzyGpuMode::Auto, "Auto")
                                        .on_hover_text("Automatically decide based on data size");
                                    ui.selectable_value(&mut self.fuzzy_gpu_mode, FuzzyGpuMode::Force, "Force")
                                        .on_hover_text("Always use GPU for metrics computation");
                                });
                        });
                        if !enable_metrics { ui.weak("(GPU metrics apply only to Advanced L10-L12 and fuzzy algorithms)"); }
                        ui.end_row();
                    });

                    // Fuzzy GPU pre-pass (Options 3, 4 & 7, plus Advanced L10/L11 only)
                    let show_prepass = if self.advanced_enabled { is_adv_fuzzy } else { supports_fuzzy_prepass_algs };
                    if show_prepass {
                        ui.add_space(6.0);
                        ui.checkbox(&mut self.use_gpu_fuzzy_direct_hash, "Enable GPU Pre-pass for Candidate Filtering")
                            .on_hover_text("Use GPU hash filter to reduce candidate pairs before scoring. For Options 3/4 uses birthdate + last-initial; for Option 7 uses OR of (date+SNDX(F/L), date+F3/L3, date+SNDX(M)). CPU fallback available.");

                        if self.use_gpu_fuzzy_direct_hash {
                            ui.add_space(4.0);
                            ui.horizontal(|ui| {
                                ui.label("Pre-pass VRAM (MB):");
                                let prepass_enabled = self.use_gpu && self.use_gpu_fuzzy_direct_hash;
                                ui.add_enabled(prepass_enabled, TextEdit::singleline(&mut self.gpu_fuzzy_prep_mem_mb).desired_width(80.0))
                                    .on_hover_text("VRAM budget for GPU hashing during fuzzy candidate pre-pass");
                            });
                        }
                    }

                        if show_fuzzy_gpu_section {
                            // Dynamic GPU Auto-Tuning (HIDDEN - slows down matching, not exposed to users)
                            // Kept in code for potential future use, but UI is disabled
                            /*
                            ui.add_space(6.0);
                            let prev_dyn = self.enable_dynamic_gpu_tuning;
                            let mut chk = self.enable_dynamic_gpu_tuning;
                            ui.checkbox(&mut chk, "Dynamic GPU Auto-Tuning (beta)")
                                .on_hover_text("Automatically adjusts GPU tile size and streams based on VRAM usage and load.");
                            self.enable_dynamic_gpu_tuning = chk;
                            if self.enable_dynamic_gpu_tuning != prev_dyn {
                                #[cfg(feature = "gpu")]
                                {
                                    if self.enable_dynamic_gpu_tuning {
                                        name_matcher::matching::set_dynamic_gpu_tuning(true);
                                        name_matcher::matching::dyn_tuner_ensure_started(true);
                                    } else {
                                        name_matcher::matching::set_dynamic_gpu_tuning(false);
                                        name_matcher::matching::dyn_tuner_ensure_started(false);
                                        name_matcher::matching::dyn_tuner_stop();
                                    }
                                }
                            }

                            if self.enable_dynamic_gpu_tuning {
                                // Pastel blue highlight group with live metrics (2s cadence)
                                ui.add_space(4.0);
                                #[cfg(feature = "gpu")]
                                {
                                    egui::Frame::default()
                                        .fill(egui::Color32::from_rgb(173, 216, 230))
                                        .stroke(egui::Stroke::new(1.0, egui::Color32::from_gray(140)))
                                        .corner_radius(egui::CornerRadius::same(6))
                                        .inner_margin(egui::Margin::symmetric(8, 6))
                                        .show(ui, |ui| {
                                            ui.horizontal(|ui| {
                                                ui.monospace(format!(
                                                    "VRAM: {:.0}% free  |  Tile: {} pairs  |  Streams: {}",
                                                    name_matcher::matching::dyn_tuner_vram_free_pct(),
                                                    name_matcher::matching::dyn_tuner_tile_size(),
                                                    name_matcher::matching::dyn_tuner_streams()
                                                ));
                                            });
                                        });
                                    ui.ctx().request_repaint_after(std::time::Duration::from_secs(2));
                                }
                            }
                            */
                        }

                    // GPU Full Scoring (Option 7 only)
                    if matches!(self.algo, MatchingAlgorithm::LevenshteinWeighted) {
                        ui.add_space(6.0);
                        let prev_full_scoring = self.use_gpu_levenshtein_full_scoring;
                        if ui.checkbox(&mut self.use_gpu_levenshtein_full_scoring, "Enable GPU Full Scoring (Option 7)")
                            .on_hover_text("Compute Levenshtein distances on GPU for all candidate pairs. Requires GPU pre-pass to be enabled. Provides significant speedup for large datasets. Experimental feature.")
                            .changed() && self.use_gpu_levenshtein_full_scoring && !prev_full_scoring {
                            // Auto-enable GPU Pre-pass when Full Scoring is enabled (Fix #5)
                            if !self.use_gpu_fuzzy_direct_hash {
                                self.use_gpu_fuzzy_direct_hash = true;
                                log::info!("[GUI] Auto-enabled GPU Pre-pass (required for Full Scoring)");
                            }
                        }

                        if self.use_gpu_levenshtein_full_scoring && self.use_gpu_fuzzy_direct_hash {
                            ui.colored_label(egui::Color32::from_rgb(76, 217, 100), "✓ GPU Full Scoring Active (Pre-pass enabled)");
                        }
                    }

                    ui.add_space(8.0);
                    ui.separator();
                }


                ui.horizontal_wrapped(|ui| {
                    if ui.button("🚀 Ultra Performance Mode").on_hover_text("Intelligent hardware-aware optimization with VRAM-aware execution mode selection, aggressive GPU settings (90% VRAM), and maximum safe performance. Analyzes dataset size and automatically selects In-Memory or Streaming mode. Target: 70-90% GPU utilization.").clicked() {
                        self.ultra_performance_mode();
                    }

                    if self.gpu_total_mb > 0 { ui.label(format!("GPU: {} MB free / {} MB total | {}", self.gpu_free_mb, self.gpu_total_mb, if self.gpu_active { "active" } else { "idle" })); }
                });

                ui.add_space(8.0);
                ui.separator();

                // System Information
                ui.strong("💻 System Information");
                ui.add_space(4.0);
                let mem = name_matcher::metrics::memory_stats_mb();
                let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
                ui.horizontal_wrapped(|ui| {
                    if ui.button("CUDA Diagnostics").on_hover_text("Show CUDA devices, versions, and memory info").clicked() {
                        self.cuda_diag_text = Self::compute_cuda_diagnostics();
                        self.cuda_diag_open = true;
                    }
                    ui.label(format!("System: {} cores | free mem {} MB", cores, mem.avail_mb));
                });
            }); // End of Advanced Settings CollapsingHeader

        ui.add_space(12.0);
        ui.separator();

        // Status Section
        ui.strong("📊 Status");
        ui.add_space(4.0);
        if self.running {
            ui.label(&self.status);
            ui.add_space(4.0);
            ui.add(ProgressBar::new(self.progress / 100.0).text(format!(
                "{:.1}% | ETA {}s | Used {} MB | Avail {} MB",
                self.progress, self.eta_secs, self.mem_used, self.mem_avail
            )));
            ui.label(format!(
                "Stage: {} | Records: {}/{} | Batch: {} | Throughput: {:.0} rec/s",
                self.stage, self.processed, self.total, self.batch_current, self.rps
            ));
        } else {
            ui.label(&self.status);
        }

        ui.add_space(12.0);
        ui.separator();

        // Action Buttons
        ui.horizontal_wrapped(|ui| {
            if !self.running {

                if ui.button("Start").on_hover_text("Run matching with selected mode and format").clicked() { self.start(); }
            } else {
                if let Some(p) = self.ctrl_pause.as_ref() {
                    let paused = p.load(Ordering::Relaxed);
                    if ui.button(if paused { "Resume" } else { "Pause" }).clicked() { p.store(!paused, Ordering::Relaxed); }
                }
                if let Some(c) = self.ctrl_cancel.as_ref() {
                    if ui.button("Cancel").clicked() { c.store(true, Ordering::Relaxed); }
                }
            }
            if ui.button("Reset").on_hover_text("Clear the form and state").clicked() { self.reset_state(); }
            if !self.error_events.is_empty() {
                ui.separator();
                ComboBox::from_label("Report Format")
                    .selected_text(match self.report_format { ReportFormat::Text=>"TXT", ReportFormat::Json=>"JSON" })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.report_format, ReportFormat::Text, "TXT");
                        ui.selectable_value(&mut self.report_format, ReportFormat::Json, "JSON");
                    });
                ui.checkbox(&mut self.schema_analysis_enabled, "Include schema analysis in report")
                    .on_hover_text("Runs INFORMATION_SCHEMA queries to suggest missing columns/types/indexes; metadata-only, no row data");
                if ui.button("Export Error Report").on_hover_text("Export a sanitized diagnostic report for technical support").clicked() {
                    match self.export_error_report() {
                        Ok(p) => { self.status = format!("Error report saved to {}", p); }
                        Err(e) => { self.status = format!("Failed to export report: {}", e); }
                    }
                }
                ui.label(format!("Errors captured: {} (log tail {} entries)", self.error_events.len(), self.log_buffer.len()));
            }
        });

        ui.separator();
        ui.label(format!(
            "Results: Algo1={} | Algo2={} | CSV={}  @ {}",
            self.a1_count,
            self.a2_count,
            self.csv_count,
            Utc::now()
        ));
    }

    fn validate(&self) -> Result<()> {
        if self.host.trim().is_empty() {
            anyhow::bail!("Host is required");
        }
        if self.port.parse::<u16>().is_err() {
            anyhow::bail!("Port must be a number");
        }
        if self.user.trim().is_empty() {
            anyhow::bail!("Username is required");
        }
        if self.db.trim().is_empty() {
            anyhow::bail!("Database is required");
        }
        if self.path.trim().is_empty() {
            anyhow::bail!("Output path is required");
        }
        if self.enable_dual {
            if self.host2.trim().is_empty() {
                anyhow::bail!("DB2 host is required");
            }
            if self.port2.parse::<u16>().is_err() {
                anyhow::bail!("DB2 port must be a number");
            }
            if self.user2.trim().is_empty() {
                anyhow::bail!("DB2 username is required");
            }
            if self.db2.trim().is_empty() {
                anyhow::bail!("DB2 database is required");
            }
            if self.tables.is_empty() {
                anyhow::bail!("Please load DB1 tables and select Table 1");
            }
            if self.tables2.is_empty() {
                anyhow::bail!("Please load DB2 tables and select Table 2");
            }
        } else {
            if self.tables.is_empty() {
                anyhow::bail!("Please load tables and select Table 1/2");
            }
        }

        // Validate Advanced Matching configuration
        if self.advanced_enabled && !self.cascade_enabled && self.adv_level.is_none() {
            anyhow::bail!(
                "Please select an Advanced Matching level (L1-L12) or enable Cascade mode"
            );
        }

        // Validate GPU Full Scoring dependency (warn but don't block)
        // This is handled in the UI with auto-enable, but double-check here
        if self.use_gpu_levenshtein_full_scoring && !self.use_gpu_fuzzy_direct_hash {
            log::warn!("[GUI] GPU Full Scoring requires GPU Pre-pass - auto-enabling");
        }

        Ok(())
    }

    fn load_tables(&mut self) {
        let (tx, rx) = mpsc::channel::<Msg>();
        self.tx = Some(tx.clone());
        // Forward subsequent log::info!/warn!/error! to the current GUI session
        set_gui_log_sender(tx.clone());
        self.last_action = "Load Tables".into();
        let host = self.host.clone();
        let port = self.port.clone();
        let user = self.user.clone();
        let pass = self.pass.clone();
        let dbname = self.db.clone();
        let enable_dual = self.enable_dual;
        let host2 = self.host2.clone();
        let port2 = self.port2.clone();
        let user2 = self.user2.clone();
        let pass2 = self.pass2.clone();
        let dbname2 = self.db2.clone();
        thread::spawn(move || {
            let rt = gui_runtime();
            if enable_dual {
                let res: Result<(Vec<String>, Vec<String>, MySqlPool, MySqlPool)> = rt.block_on(async move {
                    // DB1
                    let cfg1 = DatabaseConfig { host, port: port.parse().unwrap_or(3306), username: user, password: pass, database: dbname.clone() };
                    let pool1 = make_pool_with_size(&cfg1, Some(8)).await?;
                    let rows1 = sqlx::query_scalar::<_, String>("SELECT CAST(TABLE_NAME AS CHAR) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = ? ORDER BY TABLE_NAME")
                        .bind(dbname).fetch_all(&pool1).await?;
                    // DB2
                    let cfg2 = DatabaseConfig { host: host2, port: port2.parse().unwrap_or(3306), username: user2, password: pass2, database: dbname2.clone() };
                    let pool2 = make_pool_with_size(&cfg2, Some(8)).await?;
                    let rows2 = sqlx::query_scalar::<_, String>("SELECT CAST(TABLE_NAME AS CHAR) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = ? ORDER BY TABLE_NAME")
                        .bind(dbname2).fetch_all(&pool2).await?;
                    Ok((rows1, rows2, pool1, pool2))
                });
                match res {
                    Ok((t1, t2, p1, p2)) => {
                        let _ = tx.send(Msg::Info(format!(
                            "Loaded DB1: {} tables | DB2: {} tables",
                            t1.len(),
                            t2.len()
                        )));
                        let _ = tx.send(Msg::Tables(t1));
                        let _ = tx.send(Msg::Tables2(t2));
                        let _ = tx.send(Msg::DbPools {
                            pool1: p1,
                            pool2: Some(p2),
                        });
                    }
                    Err(e) => {
                        let (sqlstate, chain) = extract_sqlstate_and_chain(&e);
                        let _ = tx.send(Msg::ErrorRich {
                            display: format!("Failed to load tables: {}", e),
                            sqlstate,
                            chain,
                            operation: Some("Load Tables (DB1+DB2)".into()),
                        });
                    }
                }
            } else {
                let res: Result<(Vec<String>, MySqlPool)> = rt.block_on(async move {
                    let cfg = DatabaseConfig { host, port: port.parse().unwrap_or(3306), username: user, password: pass, database: dbname.clone() };
                    let pool = make_pool_with_size(&cfg, Some(8)).await?;
                    let rows = sqlx::query_scalar::<_, String>("SELECT CAST(TABLE_NAME AS CHAR) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = ? ORDER BY TABLE_NAME")
                        .bind(dbname).fetch_all(&pool).await?;
                    Ok((rows, pool))
                });
                match res {
                    Ok((tables, p)) => {
                        let _ = tx.send(Msg::Info(format!("Loaded {} tables", tables.len())));
                        let _ = tx.send(Msg::Tables(tables));
                        let _ = tx.send(Msg::DbPools {
                            pool1: p,
                            pool2: None,
                        });
                    }
                    Err(e) => {
                        let (sqlstate, chain) = extract_sqlstate_and_chain(&e);
                        let _ = tx.send(Msg::ErrorRich {
                            display: format!("Failed to load tables: {}", e),
                            sqlstate,
                            chain,
                            operation: Some("Load Tables (DB1)".into()),
                        });
                    }
                }
            }
        });
        self.rx = rx; // switch to listen on this job
        self.status = "Loading tables...".into();
    }

    fn start(&mut self) {
        if let Err(e) = self.validate() {
            self.status = format!("Error: {}", e);
            return;
        }
        self.running = true;
        self.progress = 0.0;
        self.status = "Running...".into();
        self.a1_count = 0;
        self.a2_count = 0;
        self.csv_count = 0;
        // Capture global run start at button click
        let run_started_utc = chrono::Utc::now();
        self.run_started_utc = Some(run_started_utc);

        self.gpu_build_active_now = false;
        self.gpu_probe_active_now = false;
        self.last_action = "Run Matching".into();
        let (tx, rx) = mpsc::channel::<Msg>();
        self.tx = Some(tx.clone());
        self.rx = rx;
        // Forward subsequent log::info!/warn!/error! to the current GUI session
        set_gui_log_sender(tx.clone());
        // Optionally mirror logs to file and spawn external console tail (Windows)
        let enable_console = std::env::var("NAME_MATCHER_GUI_PROGRESS_CONSOLE")
            .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false")))
            .unwrap_or(true);
        if enable_console {
            let _ = std::fs::create_dir_all("target/gui_logs");
            let ts = chrono::Utc::now().format("%Y%m%d_%H%M%S");
            let path_str = format!("target/gui_logs/run-{}.log", ts);
            if let Ok(file) = std::fs::File::create(&path_str) {
                let m = GUI_LOG_FILE.get_or_init(|| Mutex::new(None));
                if let Ok(mut g) = m.lock() {
                    *g = Some(BufWriter::new(file));
                }
                let st = GUI_LOG_FILE_FLUSH_STATE
                    .get_or_init(|| Mutex::new((std::time::Instant::now(), 0)));
                if let Ok(mut s) = st.lock() {
                    *s = (std::time::Instant::now(), 0);
                }
                if let Ok(ch) = spawn_console_tail(std::path::Path::new(&path_str)) {
                    self.console_child = Some(ch);
                }
            }
        }

        let cfg1 = DatabaseConfig {
            host: self.host.clone(),
            port: self.port.parse().unwrap_or(3306),
            username: self.user.clone(),
            password: self.pass.clone(),
            database: self.db.clone(),
        };
        let enable_dual = self.enable_dual;
        let cfg2 = if enable_dual {
            Some(DatabaseConfig {
                host: self.host2.clone(),
                port: self.port2.parse().unwrap_or(3306),
                username: self.user2.clone(),
                password: self.pass2.clone(),
                database: self.db2.clone(),
            })
        } else {
            None
        };
        let table1 = self
            .tables
            .get(self.table1_idx)
            .cloned()
            .unwrap_or_default();
        let table2 = if enable_dual {
            self.tables2
                .get(self.table2_idx)
                .cloned()
                .unwrap_or_default()
        } else {
            self.tables
                .get(self.table2_idx)
                .cloned()
                .unwrap_or_default()
        };
        let algo = self.algo;
        let path = self.path.clone();
        let fmt = self.fmt;
        let mode = self.mode;
        let use_gpu = self.use_gpu;
        let gpu_mem = self.gpu_mem_mb.parse::<u64>().unwrap_or(512);
        let use_gpu_hash_join = self.use_gpu_hash_join;
        // Cache handles captured for potential reuse (pre-warmed in Load Tables)
        let pool1_cache = self.pool1_cache.clone();
        let pool2_cache = self.pool2_cache.clone();
        let fuzzy_threshold_pct_val = self.fuzzy_threshold_pct;

        let use_gpu_build_hash = self.use_gpu_build_hash;
        let use_gpu_probe_hash = self.use_gpu_probe_hash;
        let gpu_probe_mem_mb_val = self.gpu_probe_mem_mb.parse::<u64>().unwrap_or(256);
        // Schema caches cloned for async usage
        let schema_cache = self.schema_cache.clone();
        let schema_cache_ts = self.schema_cache_timestamp.clone();

        let gpu_fuzzy_prep_mem_mb_val = self.gpu_fuzzy_prep_mem_mb.parse::<u64>().unwrap_or(256);

        let use_gpu_fuzzy_direct_hash = self.use_gpu_fuzzy_direct_hash;
        let use_gpu_levenshtein_full_scoring = self.use_gpu_levenshtein_full_scoring;
        let direct_norm_fuzzy = self.direct_norm_fuzzy;

        let pool_sz = self.pool_size.parse::<u32>().unwrap_or(16);
        let batch = self.batch_size.parse::<i64>().unwrap_or(50_000);
        let fuzzy_gpu_mode = self.fuzzy_gpu_mode;
        let rayon_threads = self.rayon_threads.parse::<usize>().unwrap_or(0);
        let gpu_streams_val = self
            .gpu_streams
            .parse::<u32>()
            .ok()
            .unwrap_or(2)
            .clamp(1, 16);
        let gpu_buffer_pool = self.gpu_buffer_pool;
        let gpu_pinned_host = self.gpu_pinned_host;

        // Advanced Matching selections
        let advanced_enabled = self.advanced_enabled;
        let adv_level = self.adv_level;
        let adv_threshold = self.adv_threshold;
        // Allow birthdate swap toggle (captured by value for thread spawn)
        // This single unified field is used for both standalone and cascade modes (Option B fix)
        let allow_birthdate_swap = self.allow_birthdate_swap;

        // Cascade Matching selections
        let cascade_enabled = self.cascade_enabled;
        let cascade_missing_column_mode = self.cascade_missing_column_mode;
        // Note: cascade now uses the unified allow_birthdate_swap field above

        let enable_dynamic_gpu_tuning = self.enable_dynamic_gpu_tuning;

        let fuzzy_thr: f32 = (self.fuzzy_threshold_pct as f32) / 100.0;

        if matches!(
            algo,
            MatchingAlgorithm::Fuzzy
                | MatchingAlgorithm::FuzzyNoMiddle
                | MatchingAlgorithm::LevenshteinWeighted
        ) && !matches!(fmt, FormatSel::Csv)
        {
            self.running = false;
            self.status = "Selected algorithm supports CSV format only. Please select CSV.".into();
            return;
        }

        let mem_thr = self.mem_thresh.parse::<u64>().unwrap_or(800);

        // create control flags for pause/cancel
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let pause_flag = Arc::new(AtomicBool::new(false));
        self.ctrl_cancel = Some(cancel_flag.clone());
        self.ctrl_pause = Some(pause_flag.clone());

        let ssd_hint = self.ssd_storage;
        thread::spawn(move || {
            let tx_for_async = tx.clone();
            let ctrl = Some(StreamControl {
                cancel: cancel_flag.clone(),
                pause: pause_flag.clone(),
            });

            if rayon_threads > 0 {
                unsafe {
                    std::env::set_var("RAYON_NUM_THREADS", rayon_threads.to_string());
                }
                let _ = rayon::ThreadPoolBuilder::new()
                    .num_threads(rayon_threads)
                    .build_global();
            }

            // Persist preference off the GUI thread to avoid blocking UI
            let _ = std::fs::write(
                ".nm_fuzzy_threshold",
                format!("{}%", fuzzy_threshold_pct_val),
            );

            // Propagate global run start into async context
            let global_run_start_utc = run_started_utc;
            let rt = gui_runtime();
            let res: Result<(usize,usize,usize,String,bool,bool)> = rt.block_on(async move {
                // Use the globally captured start time for all summary finalizations
                let run_start_utc = global_run_start_utc;
                // Announce initialization start
                let _ = tx_for_async.send(Msg::Info("Initializing resources...".into()));
                // Try to reuse cached DB pools when available and compatible
                let (pool1, pool2_opt) = if let Some(p1_cached) = pool1_cache.clone() {
                    let _ = tx_for_async.send(Msg::Info("Reusing cached database connections".into()));
                    let p2 = if enable_dual { pool2_cache.clone() } else { None };
                    (p1_cached, p2)
                } else if enable_dual {
                    let _ = tx_for_async.send(Msg::Info("Connecting to databases...".into()));
                    let p1 = make_pool_with_size(&cfg1, Some(pool_sz)).await?;
                    let p2 = make_pool_with_size(cfg2.as_ref().unwrap(), Some(pool_sz)).await?;
                    (p1, Some(p2))
                } else {
                    let _ = tx_for_async.send(Msg::Info("Connecting to database...".into()));
                    (make_pool_with_size(&cfg1, Some(pool_sz)).await?, None)
                };

                // StreamingConfig: Performance Characteristics and Trade-offs
                // ============================================================
                // Streaming mode is designed for MEMORY EFFICIENCY, not raw speed. It processes data in batches
                // to handle datasets larger than available RAM, but this comes with inherent overhead:
                //
                // Streaming Overhead Sources:
                // - Multiple database queries (one per batch) vs single query in in-memory mode
                // - Network latency per batch fetch
                // - Checkpoint writing/reading for resume capability
                // - Progressive disk flushing during processing
                //
                // In-Memory Advantages:
                // - Single database query to load all data
                // - All data in RAM for instant access (no I/O during matching)
                // - No checkpoint overhead
                // - Batch export at the end (single write operation)
                //
                // Batch Size Selection Guidance:
                // - Smaller batches (e.g., 10,000): More database round trips, slower but more memory-efficient
                // - Larger batches (e.g., 100,000+): Fewer round trips, faster but requires more RAM
                // - Default (50,000): Balanced for most systems
                // - Optimal: Set based on available RAM (see "Max Performance Settings" button)
                //
                // Expected Performance:
                // - Streaming will typically be 2-5x slower than in-memory for datasets that fit in RAM
                // - This is NORMAL and EXPECTED behavior - streaming prioritizes memory efficiency over speed
                // - Use streaming when: Dataset is too large for RAM, or you need resume capability
                // - Use in-memory when: Dataset fits in RAM and you want maximum speed
                let mut scfg = StreamingConfig { batch_size: batch, memory_soft_min_mb: mem_thr, enable_dynamic_gpu_tuning, ..Default::default() };
                // progressive saving/resume: write checkpoint next to output file
                let db_label = if enable_dual { format!("{} | {}", cfg1.database, cfg2.as_ref().unwrap().database) } else { cfg1.database.clone() };
                scfg.use_gpu_hash_join = use_gpu_hash_join;
                scfg.use_gpu_build_hash = use_gpu_build_hash;
                scfg.use_gpu_probe_hash = use_gpu_probe_hash;
                scfg.gpu_probe_batch_mb = gpu_probe_mem_mb_val;
                scfg.gpu_streams = gpu_streams_val;
                scfg.gpu_buffer_pool = gpu_buffer_pool;
                scfg.gpu_use_pinned_host = gpu_pinned_host;
                // CPU-GPU parity is always-on for Options 3–6 on CPU; no toggle call required.
                scfg.use_gpu_fuzzy_direct_hash = use_gpu_fuzzy_direct_hash;
                scfg.direct_use_fuzzy_normalization = direct_norm_fuzzy;
                // Apply global normalization alignment for in-memory comparators as well
                name_matcher::matching::set_direct_normalization_fuzzy(direct_norm_fuzzy);

                // Apply global GPU pre-pass toggles and VRAM budget
                name_matcher::matching::set_gpu_fuzzy_direct_prep(use_gpu_fuzzy_direct_hash && (matches!(algo, MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle | MatchingAlgorithm::LevenshteinWeighted) || (advanced_enabled && matches!(adv_level, Some(AdvLevel::L10FuzzyBirthdateFullMiddle) | Some(AdvLevel::L11FuzzyBirthdateNoMiddle)))));
                name_matcher::matching::set_gpu_levenshtein_prepass(use_gpu_fuzzy_direct_hash && matches!(algo, MatchingAlgorithm::LevenshteinWeighted));
                name_matcher::matching::set_gpu_levenshtein_full_scoring(use_gpu_levenshtein_full_scoring && matches!(algo, MatchingAlgorithm::LevenshteinWeighted));
                name_matcher::matching::set_gpu_fuzzy_prepass_budget_mb(gpu_fuzzy_prep_mem_mb_val);

                // Log comprehensive GPU settings for verification
                if use_gpu {
                    log::info!("[GUI] GPU Settings Applied:");
                    log::info!("[GUI]   Memory Budget: {} MB (in-memory algorithms)", gpu_mem);
                    log::info!("[GUI]   Pre-pass VRAM: {} MB (fuzzy candidate filtering)", gpu_fuzzy_prep_mem_mb_val);
                    log::info!("[GUI]   Probe Memory: {} MB (streaming hash join)", gpu_probe_mem_mb_val);
                    log::info!("[GUI]   GPU Pre-pass: {} (Options 3,4,7)", use_gpu_fuzzy_direct_hash);
                    log::info!("[GUI]   GPU Full Scoring: {} (Option 7 only)", use_gpu_levenshtein_full_scoring);
                    log::info!("[GUI]   Fuzzy Metrics Mode: {:?}", fuzzy_gpu_mode);
                    log::info!("[GUI]   Hash Join: {} (Build: {}, Probe: {})", use_gpu_hash_join, use_gpu_build_hash, use_gpu_probe_hash);
                }

                // Log StreamingConfig for verification (Issue #2: Verify configuration is applied)
                log::info!("[GUI] Execution Configuration:");
                log::info!("[GUI]   Mode: {:?}", mode);
                log::info!("[GUI]   Database Pool Size: {}", pool_sz);
                log::info!("[GUI]   Batch Size: {} rows", scfg.batch_size);
                log::info!("[GUI]   Memory Soft Min: {} MB", scfg.memory_soft_min_mb);
                log::info!("[GUI]   Flush Every: {} matches", scfg.flush_every);
                log::info!("[GUI]   Dynamic GPU Tuning: {}", scfg.enable_dynamic_gpu_tuning);
                log::info!("[GUI]   GPU Hash Join: {}", scfg.use_gpu_hash_join);
                log::info!("[GUI]   GPU Build Hash: {}", scfg.use_gpu_build_hash);
                log::info!("[GUI]   GPU Probe Hash: {}", scfg.use_gpu_probe_hash);
                log::info!("[GUI]   GPU Probe Batch: {} MB", scfg.gpu_probe_batch_mb);
                log::info!("[GUI]   GPU Streams: {}", scfg.gpu_streams);
                log::info!("[GUI]   GPU Buffer Pool: {}", scfg.gpu_buffer_pool);
                log::info!("[GUI]   GPU Pinned Host: {}", scfg.gpu_use_pinned_host);
                log::info!("[GUI]   GPU Fuzzy Direct Hash: {}", scfg.use_gpu_fuzzy_direct_hash);
                log::info!("[GUI]   GPU Fuzzy Metrics: {}", scfg.use_gpu_fuzzy_metrics);
                log::info!("[GUI]   Direct Use Fuzzy Normalization: {}", scfg.direct_use_fuzzy_normalization);

                // Apply Fuzzy GPU mode to both streaming and global toggles
                scfg.use_gpu_fuzzy_metrics = matches!(fuzzy_gpu_mode, FuzzyGpuMode::Auto | FuzzyGpuMode::Force);
                name_matcher::matching::set_gpu_fuzzy_metrics(scfg.use_gpu_fuzzy_metrics);
                name_matcher::matching::set_gpu_fuzzy_force(matches!(fuzzy_gpu_mode, FuzzyGpuMode::Force));
                name_matcher::matching::set_gpu_fuzzy_disable(matches!(fuzzy_gpu_mode, FuzzyGpuMode::Off));

                scfg.checkpoint_path = Some(format!("{}.nmckpt", path));
                // Pre-scan Table 2 extra field names for CSV/XLSX stream writers
                // If Advanced Matching is enabled, route here and return early
                if advanced_enabled {
                    // Enforce CSV for Advanced (for now)
                    if !matches!(fmt, FormatSel::Csv) {
                        let _ = tx_for_async.send(Msg::Info("Advanced Matching currently outputs CSV only; falling back to CSV export".into()));
                    }

                    // Cascade Matching: run L1-L11 sequentially (L12 excluded)
                    if cascade_enabled {
                        use name_matcher::matching::cascade::{
                            CascadeConfig, CascadeProgress, CascadePhase, GeoColumnStatus,
                            run_cascade_inmemory, summary_output_path,
                        };

                        let _ = tx_for_async.send(Msg::Info("Starting cascade run for levels L1-L11 (L12 excluded)".into()));

                        // Detect geographic columns
                        let all_cols_t1 = get_all_table_columns(&pool1, &cfg1.database, &table1).await?;
                        let all_cols_t2 = if let Some(pool2) = pool2_opt.as_ref() {
                            get_all_table_columns(pool2, cfg2.as_ref().unwrap().database.as_str(), &table2).await?
                        } else {
                            get_all_table_columns(&pool1, &cfg1.database, &table2).await?
                        };

                        let geo_status = GeoColumnStatus {
                            has_barangay_code: all_cols_t1.iter().any(|c| c == "barangay_code")
                                || all_cols_t2.iter().any(|c| c == "barangay_code"),
                            has_city_code: all_cols_t1.iter().any(|c| c == "city_code")
                                || all_cols_t2.iter().any(|c| c == "city_code"),
                        };

                        let _ = tx_for_async.send(Msg::Info(geo_status.summary()));

                        // Load data
                        let _ = tx_for_async.send(Msg::Info(format!("Loading Table1: {}", table1)));
                        let t1 = get_person_rows(&pool1, &table1).await?;
                        let _ = tx_for_async.send(Msg::Info(format!("Loaded {} records from Table1", t1.len())));

                        let _ = tx_for_async.send(Msg::Info(format!("Loading Table2: {}", table2)));
                        let t2 = if let Some(pool2) = pool2_opt.as_ref() {
                            get_person_rows(pool2, &table2).await?
                        } else {
                            get_person_rows(&pool1, &table2).await?
                        };
                        let _ = tx_for_async.send(Msg::Info(format!("Loaded {} records from Table2", t2.len())));

                        // Build cascade config
                        // Determine GPU backend: use GPU if either hash join (L1-L9) or fuzzy GPU (L10-L11) is enabled
                        let cascade_use_gpu = use_gpu || use_gpu_hash_join;

                        // Check if GPU is requested but the binary was compiled without GPU feature
                        #[cfg(not(feature = "gpu"))]
                        if cascade_use_gpu {
                            let _ = tx_for_async.send(Msg::Info(
                                "⚠ GPU requested for cascade but this binary lacks GPU support. \
                                L10-L11 will run on CPU. Rebuild with `--features gpu` for GPU acceleration.".into()
                            ));
                            log::warn!("[GUI] Cascade GPU requested but binary compiled without 'gpu' feature");
                        }

                        // When gpu feature is not compiled, always use CPU backend to avoid confusion
                        #[cfg(not(feature = "gpu"))]
                        let cascade_compute_backend = name_matcher::matching::ComputeBackend::Cpu;

                        #[cfg(feature = "gpu")]
                        let cascade_compute_backend = if cascade_use_gpu {
                            name_matcher::matching::ComputeBackend::Gpu
                        } else {
                            name_matcher::matching::ComputeBackend::Cpu
                        };

                        log::info!("[GUI] Cascade compute backend: {:?} (use_gpu={}, use_gpu_hash_join={}, gpu_feature={})",
                            cascade_compute_backend, use_gpu, use_gpu_hash_join, cfg!(feature = "gpu"));

                        let cascade_cfg = CascadeConfig {
                            levels: vec![], // Run all L1-L11
                            threshold: adv_threshold,
                            // Use unified birthdate swap setting (Option B fix - single toggle for all modes)
                            allow_birthdate_swap,
                            missing_column_mode: cascade_missing_column_mode,
                            base_output_path: path.clone(),
                            exclusion_mode: name_matcher::matching::cascade::CascadeExclusionMode::Exclusive,
                            // GPU acceleration for cascade: uses hash join for L1-L9 and fuzzy metrics for L10-L11
                            compute_backend: cascade_compute_backend,
                            gpu_device_id: None,
                        };

                        // Run cascade with progress reporting
                        let tx_progress = tx_for_async.clone();
                        let result = run_cascade_inmemory(&t1, &t2, &cascade_cfg, &geo_status, move |progress: CascadeProgress| {
                            let msg = match progress.phase {
                                CascadePhase::Starting => format!("Starting Level {} of {}: {}",
                                    progress.current_level, progress.total_levels, progress.level_description),
                                CascadePhase::Running => format!("Running Level {}: {}",
                                    progress.current_level, progress.level_description),
                                CascadePhase::WritingOutput => format!("Writing output for Level {}", progress.current_level),
                                CascadePhase::Completed => format!("Level {} complete", progress.current_level),
                                CascadePhase::Skipped(ref reason) => format!("Skipping Level {}: {}", progress.current_level, reason),
                            };
                            let _ = tx_progress.send(Msg::Info(msg));

                            // Send progress update
                            let percent = ((progress.current_level as f32 - 1.0) / progress.total_levels as f32) * 100.0;
                            let _ = tx_progress.send(Msg::Progress(ProgressUpdate {
                                percent,
                                processed: progress.current_level as usize,
                                total: progress.total_levels,
                                mem_used_mb: 0,
                                mem_avail_mb: 0,
                                eta_secs: 0,
                                stage: "cascade",
                                batch_size_current: None,
                                gpu_total_mb: 0,
                                gpu_free_mb: 0,
                                gpu_active: false,
                            }));
                        });

                        // Write summary
                        let summary_path = summary_output_path(&path);
                        match result.write_summary(&summary_path) {
                            Ok(_) => { let _ = tx_for_async.send(Msg::Info(format!("Cascade summary written to: {}", summary_path))); }
                            Err(e) => { let _ = tx_for_async.send(Msg::Info(format!("Failed to write cascade summary: {}", e))); }
                        }

                        let _ = tx_for_async.send(Msg::Info(format!("=== Cascade Complete: {} total matches ===", result.total_matches)));

                        return Ok((0, 0, result.total_matches, path.clone(), false, false));
                    }

                    // Build AdvConfig: fixed geographic field names; no external mapping
                    let adv_cols = AdvColumns::default();
                    if let Some(level) = adv_level {
                        // L12: Household path — stream when allowed, else in-memory
                        if matches!(level, AdvLevel::L12HouseholdMatching) {
                            let adv_cfg = AdvConfig { level, threshold: fuzzy_thr, cols: adv_cols.clone(), allow_birthdate_swap };
                            // Determine execution mode for L12; Auto defaults to In-Memory (no row-count threshold)
                            let use_streaming = match mode {
                                ModeSel::Streaming => true,
                                ModeSel::InMemory => false,
                                ModeSel::Auto => false,
                            };
                            // Row count acquisition policy:
                            // - Streaming: exact COUNT(*) for progress tracking
                            // - Auto: fast estimate via EXPLAIN (fallback to 0 on failure)
                            // - InMemory: skip counts at startup (defer to post-run summary)
                            let (c1, c2) = if use_streaming {
                                let c1 = get_person_count(&pool1, &table1).await?;
                                let c2 = if let Some(pool2) = pool2_opt.as_ref() { get_person_count(pool2, &table2).await? } else { get_person_count(&pool1, &table2).await? };
                                (c1, c2)
                            } else if matches!(mode, ModeSel::Auto) {
                                let c1 = get_person_count_fast(&pool1, &table1).await.unwrap_or(0);
                                let c2 = if let Some(pool2) = pool2_opt.as_ref() { get_person_count_fast(pool2, &table2).await.unwrap_or(0) } else { get_person_count_fast(&pool1, &table2).await.unwrap_or(0) };
                                (c1, c2)
                            } else { (0, 0) };
                            if c1 > 0 || c2 > 0 {
                                log::info!(
                                    "[GUI] L12 execution mode: {} (table1: {} rows, table2: {} rows)",
                                    if use_streaming { "Streaming" } else { "In-Memory" }, c1, c2
                                );
                            } else {
                                log::info!("[GUI] L12 execution mode: {}", if use_streaming { "Streaming" } else { "In-Memory" });
                            }
                            let compute_backend = if scfg.use_gpu_fuzzy_metrics { "GPU" } else { "CPU" };
                            let gpu_features = if scfg.use_gpu_fuzzy_metrics { "GPU Fuzzy Metrics" } else { "" };
                            let gpu_model = if scfg.use_gpu_fuzzy_metrics { name_matcher::matching::try_gpu_name() } else { None };
                            let mut w = HouseholdCsvWriter::create_with_meta(&path, compute_backend, gpu_model.as_deref(), gpu_features)?;
                            let txp = tx_for_async.clone();
                            if use_streaming {
                                if let Some(pool2) = pool2_opt.as_ref() {
                                    log::info!("[GUI] L12 cross-DB streaming route ({} -> {}): pool1 for {} , pool2 for {}", table1, table2, table1, table2);
                                    let emitted = stream_match_advanced_l12_dual(&pool1, pool2, &table1, &table2, &adv_cfg, scfg.clone(), |row| { w.write(row)?; Ok(()) }, move |u| {
                                        log::info!("[adv-stream] {:.1}% | ETA: {}s | Mem: {} MB | Avail: {} MB | {}/{}", u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total);
                                        let _ = txp.send(Msg::Progress(u));
                                    }, ctrl.clone()).await?;
                                    w.flush()?;
                                    return Ok((0,0, emitted, path.clone(), false, scfg.use_gpu_fuzzy_metrics));
                                } else {
                                    log::info!("[GUI] L12 single-DB streaming route for {} vs {}", table1, table2);
                                    let emitted = stream_match_advanced_l12(&pool1, &table1, &table2, &adv_cfg, scfg.clone(), |row| { w.write(row)?; Ok(()) }, move |u| {
                                        log::info!("[adv-stream] {:.1}% | ETA: {}s | Mem: {} MB | Avail: {} MB | {}/{}", u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total);
                                        let _ = txp.send(Msg::Progress(u));
                                    }, ctrl.clone()).await?;
                                    w.flush()?;
                                    return Ok((0,0, emitted, path.clone(), false, scfg.use_gpu_fuzzy_metrics));
                                }
                            } else {
                                // L12 in-memory: delegate to Option 6's implementation (source of truth)
                                let t1 = get_person_rows(&pool1, &table1).await?;
                                let t2 = if let Some(pool2) = pool2_opt.as_ref() { get_person_rows(pool2, &table2).await? } else { get_person_rows(&pool1, &table2).await? };
                                let cfgp = ProgressConfig { update_every: 10_000, ..Default::default() };
                                let mo = MatchOptions { backend: if use_gpu { ComputeBackend::Gpu } else { ComputeBackend::Cpu }, gpu: Some(GpuConfig { device_id: None, mem_budget_mb: gpu_mem }), progress: cfgp, allow_birthdate_swap };
                                let txp2 = tx_for_async.clone();
                                // Use Option 6's function (not the streaming variant) and threshold
                                let rows = match_households_gpu_inmemory_opt6(&t1, &t2, mo, fuzzy_thr, move |u| {
                                    log::info!("[adv-inmem-l12] {:.1}% | ETA: {}s | Mem: {} MB | Avail: {} MB | {}/{}", u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total);
                                    let _ = txp2.send(Msg::Progress(u));
                                });
                                // Recompute CSV metadata based on actual execution backend (matching Option 6's logic)
                                let compute_backend = if matches!(mo.backend, ComputeBackend::Gpu) { "GPU" } else { "CPU" };
                                let gpu_features = if matches!(mo.backend, ComputeBackend::Gpu) { "GPU Fuzzy Metrics" } else { "" };
                                let gpu_model = if matches!(mo.backend, ComputeBackend::Gpu) { name_matcher::matching::try_gpu_name() } else { None };
                                let mut w = HouseholdCsvWriter::create_with_meta(&path, compute_backend, gpu_model.as_deref(), gpu_features)?;
                                for r in &rows { w.write(r)?; }
                                w.flush()?;
                                return Ok((0,0, rows.len(), path.clone(), false, false));
                            }
                        }

                        let adv_cfg = AdvConfig { level, threshold: adv_threshold, cols: adv_cols, allow_birthdate_swap };
                        // Route by execution mode: try streaming unless explicitly forced to in-memory
                        let mut use_streaming = match mode { ModeSel::Streaming => true, ModeSel::InMemory => false, ModeSel::Auto => true };
                        // For Advanced L10/L11, the in-memory GUI path is CPU-only; force streaming when GPU is enabled to keep GPU parity.
                        if use_gpu && matches!(level, AdvLevel::L10FuzzyBirthdateFullMiddle | AdvLevel::L11FuzzyBirthdateNoMiddle) {
                            use_streaming = true;
                        }
                        // streaming implemented for exact and L10
                        if matches!(level,
                            AdvLevel::L10FuzzyBirthdateFullMiddle |
                            AdvLevel::L11FuzzyBirthdateNoMiddle |
                            AdvLevel::L1BirthdateFullMiddle |
                            AdvLevel::L2BirthdateMiddleInitial |
                            AdvLevel::L3BirthdateNoMiddle |
                            AdvLevel::L4BarangayFullMiddle |
                            AdvLevel::L5BarangayMiddleInitial |
                            AdvLevel::L6BarangayNoMiddle |
                            AdvLevel::L7CityFullMiddle |
                            AdvLevel::L8CityMiddleInitial |
                            AdvLevel::L9CityNoMiddle) {
                            // ok
                        } else {
                            use_streaming = false;
                        }
                        // CSV writer
                        let standard_cols_adv = ["id","uuid","first_name","middle_name","last_name","birthdate","hh_id"];
                        let db_for_t2_adv = if let Some(ref c2) = cfg2 { c2.database.clone() } else { cfg1.database.clone() };
                        let cols_adv = match schema_cached_or_fetch(&schema_cache, &schema_cache_ts, pool2_opt.as_ref().unwrap_or(&pool1), &db_for_t2_adv, &table2, Duration::from_secs(300)).await {
                            Ok((_tc, cols)) => cols,
                            Err(_) => Vec::new(),
                        };
                        let extra_field_names_adv: Vec<String> = cols_adv.into_iter().filter(|c| !standard_cols_adv.contains(&c.as_str())).collect();
                        let compute_backend = if scfg.use_gpu_hash_join || scfg.use_gpu_fuzzy_metrics { "GPU" } else { "CPU" };
                        let gpu_features = if scfg.use_gpu_hash_join { "GPU Hash Join" } else if scfg.use_gpu_fuzzy_metrics { "GPU Fuzzy Metrics" } else { "" };
                        let gpu_model = if compute_backend == "GPU" { name_matcher::matching::try_gpu_name() } else { None };
                        let mut w = AdvCsvStreamWriter::create_with_extra_fields(&path, extra_field_names_adv.clone(), compute_backend, gpu_model.as_deref(), gpu_features)?;
                        let flush_every = scfg.flush_every.max(1000);
                        let mut kept = 0usize; let mut seen = 0usize;
                        let txp = tx_for_async.clone();
                        if use_streaming {
                            let txp1 = txp.clone();
                            if let Some(pool2) = pool2_opt.as_ref() {
                                let _ = stream_match_advanced_dual(&pool1, pool2, &table1, &table2, &adv_cfg, scfg.clone(), |p| {
                                    seen += 1;
                                    if matches!(level, AdvLevel::L10FuzzyBirthdateFullMiddle | AdvLevel::L11FuzzyBirthdateNoMiddle) {
                                        if (p.confidence / 100.0) < adv_threshold { return Ok(()); }
                                    }
                                    kept += 1; w.write(p, level)?; if kept % flush_every == 0 { w.flush_partial()?; }
                                    Ok(())
                                }, move |u| { let _ = txp1.send(Msg::Progress(u)); }, ctrl.clone()).await?;
                            } else {
                                let _ = stream_match_advanced(&pool1, &table1, &table2, &adv_cfg, scfg.clone(), |p| {
                                    seen += 1;
                                    if matches!(level, AdvLevel::L10FuzzyBirthdateFullMiddle | AdvLevel::L11FuzzyBirthdateNoMiddle) {
                                        if (p.confidence / 100.0) < adv_threshold { return Ok(()); }
                                    }
                                    kept += 1; w.write(p, level)?; if kept % flush_every == 0 { w.flush_partial()?; }
                                    Ok(())
                                }, move |u| { let _ = txp.send(Msg::Progress(u)); }, ctrl.clone()).await?;
                            }
                            w.flush()?;
                            let (gpu_hash, gpu_fuzzy) = match level {
                                AdvLevel::L1BirthdateFullMiddle | AdvLevel::L2BirthdateMiddleInitial | AdvLevel::L3BirthdateNoMiddle |
                                AdvLevel::L4BarangayFullMiddle | AdvLevel::L5BarangayMiddleInitial | AdvLevel::L6BarangayNoMiddle |
                                AdvLevel::L7CityFullMiddle | AdvLevel::L8CityMiddleInitial | AdvLevel::L9CityNoMiddle => (scfg.use_gpu_hash_join, false),
                                AdvLevel::L10FuzzyBirthdateFullMiddle | AdvLevel::L11FuzzyBirthdateNoMiddle => (false, scfg.use_gpu_fuzzy_metrics),
                                _ => (false, false),
                            };
                            return Ok((0,0, kept, path.clone(), gpu_hash, gpu_fuzzy));
                        } else {
                            let t1 = get_person_rows(&pool1, &table1).await?;
                            let t2 = if let Some(pool2) = pool2_opt.as_ref() { get_person_rows(pool2, &table2).await? } else { get_person_rows(&pool1, &table2).await? };
                            let pairs = name_matcher::matching::advanced_matcher::advanced_match_inmemory(&t1, &t2, &adv_cfg);
                            for p in &pairs {
                                if matches!(level, AdvLevel::L10FuzzyBirthdateFullMiddle | AdvLevel::L11FuzzyBirthdateNoMiddle) && (p.confidence / 100.0) < adv_threshold { continue; }
                                w.write(p, level)?;
                            }
                            w.flush()?;
                            let count = if matches!(level, AdvLevel::L10FuzzyBirthdateFullMiddle | AdvLevel::L11FuzzyBirthdateNoMiddle) { pairs.iter().filter(|p| (p.confidence / 100.0) >= adv_threshold).count() } else { pairs.len() };
                            let (gpu_hash, gpu_fuzzy) = match level {
                                AdvLevel::L1BirthdateFullMiddle | AdvLevel::L2BirthdateMiddleInitial | AdvLevel::L3BirthdateNoMiddle |
                                AdvLevel::L4BarangayFullMiddle | AdvLevel::L5BarangayMiddleInitial | AdvLevel::L6BarangayNoMiddle |
                                AdvLevel::L7CityFullMiddle | AdvLevel::L8CityMiddleInitial | AdvLevel::L9CityNoMiddle => (scfg.use_gpu_hash_join, false),
                                AdvLevel::L10FuzzyBirthdateFullMiddle | AdvLevel::L11FuzzyBirthdateNoMiddle => (false, scfg.use_gpu_fuzzy_metrics),
                                _ => (false, false),
                            };
                            return Ok((0,0, count, path.clone(), gpu_hash, gpu_fuzzy));
                        }
                    } else {
                        anyhow::bail!("Advanced Matching enabled but no level selected");
                    }
                }

                let standard_cols = ["id","uuid","first_name","middle_name","last_name","birthdate","hh_id"];
                let db_for_t2 = if let Some(ref c2) = cfg2 { c2.database.clone() } else { cfg1.database.clone() };
                let cols_t2 = match schema_cached_or_fetch(&schema_cache, &schema_cache_ts, pool2_opt.as_ref().unwrap_or(&pool1), &db_for_t2, &table2, Duration::from_secs(300)).await {
                    Ok((_tc, cols)) => cols,
                    Err(e) => { let _ = tx_for_async.send(Msg::Info(format!("Could not discover extra columns for {}.{}: {}", db_for_t2, table2, e))); Vec::new() }
                };
                let extra_field_names: Vec<String> = cols_t2.into_iter().filter(|c| !standard_cols.contains(&c.as_str())).collect();
                match fmt {

                    FormatSel::Csv => {
                // Tune flush frequency based on batch size to prevent spikes
                if ssd_hint {
                    scfg.flush_every = (batch as usize / 6).max(1000);
                } else {
                    scfg.flush_every = (batch as usize / 12).max(1000);
                }

                        let mut use_streaming = match mode { ModeSel::Streaming => true, ModeSel::InMemory => false, ModeSel::Auto => true };
                        if matches!(algo, MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle | MatchingAlgorithm::LevenshteinWeighted) { use_streaming = false; let _ = tx_for_async.send(Msg::Info("Selected algorithm uses in-memory mode (streaming disabled)".into())); }
                        if use_gpu && !cfg!(feature = "gpu") {
                            let _ = tx_for_async.send(Msg::Info("GPU requested but this binary lacks GPU support; will run on CPU. Rebuild with --features gpu".into()));
                        }

                        if use_streaming {
                            // GPU availability info (best-effort)
                            #[cfg(feature = "gpu")]
                            if use_gpu {
                                if let Ok(ctx) = cudarc::driver::CudaContext::new(0) {
                                    let mut free: usize = 0; let mut total: usize = 0;
                                    unsafe { let _ = cudarc::driver::sys::cuMemGetInfo_v2(&mut free as *mut _ as *mut _, &mut total as *mut _ as *mut _); }
                                    let _ = tx_for_async.send(Msg::Info(format!("CUDA active | Free {} MB / Total {} MB", (free/1024/1024), (total/1024/1024))));
                                    drop(ctx);
                                } else {
                                    let _ = tx_for_async.send(Msg::Info("CUDA requested but unavailable; falling back to CPU".into()));
                                }
                            }

                            // SPECIAL CASE: Option 6 (HouseholdGpuOpt6) streaming routes to Advanced L12 household aggregator
                            if matches!(algo, MatchingAlgorithm::HouseholdGpuOpt6) {
                                // Option 6 uses dedicated streaming path matching in-memory semantics exactly; no AdvConfig.
                                let compute_backend = if scfg.use_gpu_fuzzy_metrics { "GPU" } else { "CPU" };
                                let gpu_features = if scfg.use_gpu_fuzzy_metrics { "GPU Fuzzy Metrics" } else { "" };
                                let gpu_model = if scfg.use_gpu_fuzzy_metrics { name_matcher::matching::try_gpu_name() } else { None };
                                let mut w = HouseholdCsvWriter::create_with_meta(&path, compute_backend, gpu_model.as_deref(), gpu_features)?;
                                let txp = tx_for_async.clone();
                                let emitted = if let Some(pool2) = pool2_opt.as_ref() {
                                    log::info!("[GUI] Option 6 cross-DB streaming route ({} -> {}): pool1 for {} , pool2 for {}", table1, table2, table1, table2);
                                    stream_match_option6_dual(&pool1, pool2, &table1, &table2, fuzzy_thr, allow_birthdate_swap, scfg.clone(), |row| { w.write(row)?; Ok(()) }, move |u| {
                                        log::info!("[opt6-stream] {:.1}% | ETA: {}s | Mem: {} MB | Avail: {} MB | {}/{}", u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total);
                                        let _ = txp.send(Msg::Progress(u));
                                    }, ctrl.clone()).await?
                                } else {
                                    log::info!("[GUI] Option 6 single-DB streaming route for {} vs {}", table1, table2);
                                    stream_match_option6(&pool1, &table1, &table2, fuzzy_thr, allow_birthdate_swap, scfg.clone(), |row| { w.write(row)?; Ok(()) }, move |u| {
                                        log::info!("[opt6-stream] {:.1}% | ETA: {}s | Mem: {} MB | Avail: {} MB | {}/{}", u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total);
                                        let _ = txp.send(Msg::Progress(u));
                                    }, ctrl.clone()).await?
                                };
                                w.flush()?;
                                return Ok((0,0,emitted, path.clone(), false, scfg.use_gpu_fuzzy_metrics));
                            }

                            // SPECIAL CASE: Option 5 (HouseholdGpu) dedicated streaming path mirroring in-memory semantics
                            if matches!(algo, MatchingAlgorithm::HouseholdGpu) {
                                let compute_backend = if scfg.use_gpu_fuzzy_metrics { "GPU" } else { "CPU" };
                                let gpu_features = if scfg.use_gpu_fuzzy_metrics { "GPU Fuzzy Metrics" } else { "" };
                                let gpu_model = if scfg.use_gpu_fuzzy_metrics { name_matcher::matching::try_gpu_name() } else { None };
                                let mut w = HouseholdCsvWriter::create_with_meta(&path, compute_backend, gpu_model.as_deref(), gpu_features)?;
                                let txp = tx_for_async.clone();
                                let emitted = if let Some(pool2) = pool2_opt.as_ref() {
                                    log::info!("[GUI] Option 5 cross-DB streaming route ({} -> {}): pool1 for {} , pool2 for {}", table1, table2, table1, table2);
                                    stream_match_option5_dual(&pool1, pool2, &table1, &table2, fuzzy_thr, scfg.clone(), |row| { w.write(row)?; Ok(()) }, move |u| {
                                        log::info!("[opt5-stream] {:.1}% | ETA: {}s | Mem: {} MB | Avail: {} MB | {}/{}", u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total);
                                        let _ = txp.send(Msg::Progress(u));
                                    }, ctrl.clone()).await?
                                } else {
                                    log::info!("[GUI] Option 5 single-DB streaming route for {} vs {}", table1, table2);
                                    stream_match_option5(&pool1, &table1, &table2, fuzzy_thr, scfg.clone(), |row| { w.write(row)?; Ok(()) }, move |u| {
                                        log::info!("[opt5-stream] {:.1}% | ETA: {}s | Mem: {} MB | Avail: {} MB | {}/{}", u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total);
                                        let _ = txp.send(Msg::Progress(u));
                                    }, ctrl.clone()).await?
                                };
                                w.flush()?;
                                return Ok((0,0,emitted, path.clone(), false, scfg.use_gpu_fuzzy_metrics));
                            }


                            let mut w = CsvStreamWriter::create_with_extra_fields(&path, algo, fuzzy_thr, extra_field_names.clone())?;
                            let mut cnt = 0usize;
                            let mut kept = 0usize;
                            let flush_every = scfg.flush_every;
                            let txp = tx_for_async.clone();
                            if let Some(pool2) = pool2_opt.as_ref() {
                                let _ = stream_match_csv_dual(&pool1, pool2, &table1, &table2, algo, |p| {
                                    cnt += 1;
                                    if matches!(algo, MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle | MatchingAlgorithm::LevenshteinWeighted) {
                                        if p.confidence >= 0.95 { kept += 1; }
                                    }
                                    w.write(p)?;
                                    if cnt % flush_every == 0 { w.flush_partial()?; }
                                    Ok(())
                                }, scfg.clone(), move |u| {
                                    log::info!("[stream-dual] {:.1}% | ETA: {}s | Mem: {} MB | Avail: {} MB | {}/{}", u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total);
                                    let _ = txp.send(Msg::Progress(u));
                                }, ctrl.clone()).await?;
                            } else {
                                let _ = stream_match_csv(&pool1, &table1, &table2, algo, |p| {
                                    cnt += 1;
                                    if matches!(algo, MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle | MatchingAlgorithm::LevenshteinWeighted) {
                                        if p.confidence >= 0.95 { kept += 1; }
                                    }
                                    w.write(p)?;
                                    if cnt % flush_every == 0 { w.flush_partial()?; }
                                    Ok(())
                                }, scfg.clone(), move |u| {
                                    log::info!("[stream] {:.1}% | ETA: {}s | Mem: {} MB | Avail: {} MB | {}/{}", u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total);
                                    let _ = txp.send(Msg::Progress(u));
                                }, ctrl.clone()).await?;
                            }
                            w.flush()?;
                            let csv_val = if matches!(algo, MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle | MatchingAlgorithm::LevenshteinWeighted) { kept } else { cnt };
                            Ok((0,0,csv_val,path.clone(),false,false))
                        } else {
                            // In-memory path
                            let t1 = get_person_rows(&pool1, &table1).await?;
                            let t2 = if let Some(pool2) = pool2_opt.as_ref() { get_person_rows(pool2, &table2).await? } else { get_person_rows(&pool1, &table2).await? };
                            let cfgp = ProgressConfig { update_every: 10_000, ..Default::default() };
                            let txp = tx_for_async.clone();
                                // Also reflect GPU fuzzy pre-pass in in-memory path
                                name_matcher::matching::set_gpu_fuzzy_direct_prep(use_gpu_fuzzy_direct_hash);

                                // Apply normalization alignment globally for in-memory deterministics as well
                                name_matcher::matching::set_direct_normalization_fuzzy(direct_norm_fuzzy);
                                // Apply GPU enhancement preset per selected algorithm to keep controls consistent
                                name_matcher::matching::apply_gpu_enhancements_for_algo(
                                    algo,
                                    use_gpu_fuzzy_direct_hash,
                                    use_gpu_levenshtein_full_scoring,
                                    matches!(fuzzy_gpu_mode, FuzzyGpuMode::Auto | FuzzyGpuMode::Force),
                                    matches!(fuzzy_gpu_mode, FuzzyGpuMode::Force),
                                    matches!(fuzzy_gpu_mode, FuzzyGpuMode::Off)
                                );


                            if matches!(algo, MatchingAlgorithm::HouseholdGpu) {
                                let mo = MatchOptions { backend: if use_gpu { ComputeBackend::Gpu } else { ComputeBackend::Cpu }, gpu: Some(GpuConfig { device_id: None, mem_budget_mb: gpu_mem }), progress: cfgp, allow_birthdate_swap };
                                let rows = match_households_gpu_inmemory(&t1, &t2, mo, fuzzy_thr, move |u| {
                                    log::info!("[inmem-household] {:.1}% | ETA: {}s | Mem: {} MB | Avail: {} MB | {}/{}", u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total);
                                    let _ = txp.send(Msg::Progress(u));
                                });
                                let compute_backend = if matches!(mo.backend, ComputeBackend::Gpu) { "GPU" } else { "CPU" };
                                let gpu_features = if matches!(mo.backend, ComputeBackend::Gpu) { "GPU Fuzzy Metrics" } else { "" };
                                let gpu_model = if matches!(mo.backend, ComputeBackend::Gpu) { name_matcher::matching::try_gpu_name() } else { None };
                                let mut w = HouseholdCsvWriter::create_with_meta(&path, compute_backend, gpu_model.as_deref(), gpu_features)?;
                                for r in &rows { w.write(r)?; }
                                w.flush()?;
                                Ok((0,0, rows.len(), path.clone(),false,false))
                            } else if matches!(algo, MatchingAlgorithm::HouseholdGpuOpt6) {
                                let mo = MatchOptions { backend: if use_gpu { ComputeBackend::Gpu } else { ComputeBackend::Cpu }, gpu: Some(GpuConfig { device_id: None, mem_budget_mb: gpu_mem }), progress: cfgp, allow_birthdate_swap };
                                let rows = match_households_gpu_inmemory_opt6(&t1, &t2, mo, fuzzy_thr, move |u| {
                                    log::info!("[inmem-household-opt6] {:.1}% | ETA: {}s | Mem: {} MB | Avail: {} MB | {}/{}", u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total);
                                    let _ = txp.send(Msg::Progress(u));
                                });
                                let compute_backend = if matches!(mo.backend, ComputeBackend::Gpu) { "GPU" } else { "CPU" };
                                let gpu_features = if matches!(mo.backend, ComputeBackend::Gpu) { "GPU Fuzzy Metrics" } else { "" };
                                let gpu_model = if matches!(mo.backend, ComputeBackend::Gpu) { name_matcher::matching::try_gpu_name() } else { None };
                                let mut w = HouseholdCsvWriter::create_with_meta(&path, compute_backend, gpu_model.as_deref(), gpu_features)?;
                                for r in &rows { w.write(r)?; }
                                w.flush()?;
                                Ok((0,0, rows.len(), path.clone(),false,false))
                            } else {
                                let mo = MatchOptions { backend: if use_gpu { ComputeBackend::Gpu } else { ComputeBackend::Cpu }, gpu: Some(GpuConfig { device_id: None, mem_budget_mb: gpu_mem }), progress: cfgp, allow_birthdate_swap };
                                let pairs = match_all_with_opts(&t1, &t2, algo, mo, move |u| {
                                    log::info!("[inmem] {:.1}% | ETA: {}s | Mem: {} MB | Avail: {} MB | {}/{}", u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total);
                                    let _ = txp.send(Msg::Progress(u));
                                });
                                let mut w = CsvStreamWriter::create_with_extra_fields(&path, algo, fuzzy_thr, extra_field_names.clone())?;
                                for p in &pairs { w.write(p)?; }
                                w.flush()?;
                                let kept = if matches!(algo, MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle | MatchingAlgorithm::LevenshteinWeighted) { let thr = fuzzy_thr; pairs.iter().filter(|p| p.confidence >= thr).count() } else { pairs.len() };
                                Ok((0,0, kept, path.clone(),false,false))
                            }
                        }
                    }
                    FormatSel::Xlsx => {
                        let mut use_streaming = match mode { ModeSel::Streaming => true, ModeSel::InMemory => false, ModeSel::Auto => true };
                            if matches!(algo, MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle | MatchingAlgorithm::HouseholdGpu | MatchingAlgorithm::HouseholdGpuOpt6 | MatchingAlgorithm::LevenshteinWeighted) { use_streaming = false; let _ = tx_for_async.send(Msg::Info("Selected algorithm uses in-memory mode (streaming disabled)".into())); }
                        if use_streaming {
                            let mut xw = XlsxStreamWriter::create_with_extra_fields(&path, extra_field_names.clone())?;
                            // using global run_start_utc captured earlier
                            use std::sync::{Arc};
                            use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
                            let mut a1 = 0usize; let mut a2 = 0usize;
                            let a1_used = Arc::new(AtomicBool::new(false));
                            let a2_used = Arc::new(AtomicBool::new(false));
                            let total1 = Arc::new(AtomicU64::new(0));
                            let free1 = Arc::new(AtomicU64::new(0));
                            let total2 = Arc::new(AtomicU64::new(0));
                            let free2 = Arc::new(AtomicU64::new(0));

                            let txp1 = tx_for_async.clone();
                            if let Some(pool2) = pool2_opt.as_ref() {
                                let a1_used_c = a1_used.clone(); let total1_c = total1.clone(); let free1_c = free1.clone();
                                let _ = stream_match_csv_dual(&pool1, pool2, &table1, &table2, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, |p| { a1+=1; xw.append_algo1(p) }, scfg.clone(), move |u| { if u.gpu_active { a1_used_c.store(true, Ordering::Relaxed); } total1_c.store(u.gpu_total_mb as u64, Ordering::Relaxed); free1_c.store(u.gpu_free_mb as u64, Ordering::Relaxed); let _ = txp1.send(Msg::Progress(u)); }, ctrl.clone()).await?;
                            } else {
                                let a1_used_c = a1_used.clone(); let total1_c = total1.clone(); let free1_c = free1.clone();
                                let _ = stream_match_csv(&pool1, &table1, &table2, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, |p| { a1+=1; xw.append_algo1(p) }, scfg.clone(), move |u| { if u.gpu_active { a1_used_c.store(true, Ordering::Relaxed); } total1_c.store(u.gpu_total_mb as u64, Ordering::Relaxed); free1_c.store(u.gpu_free_mb as u64, Ordering::Relaxed); let _ = txp1.send(Msg::Progress(u)); }, ctrl.clone()).await?;
                            }
                            let txp2 = tx_for_async.clone();
                            if let Some(pool2) = pool2_opt.as_ref() {
                                let a2_used_c = a2_used.clone(); let total2_c = total2.clone(); let free2_c = free2.clone();
                                let _ = stream_match_csv_dual(&pool1, pool2, &table1, &table2, MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd, |p| { a2+=1; xw.append_algo2(p) }, scfg.clone(), move |u| { if u.gpu_active { a2_used_c.store(true, Ordering::Relaxed); } total2_c.store(u.gpu_total_mb as u64, Ordering::Relaxed); free2_c.store(u.gpu_free_mb as u64, Ordering::Relaxed); let _ = txp2.send(Msg::Progress(u)); }, ctrl.clone()).await?;
                            } else {
                                let a2_used_c = a2_used.clone(); let total2_c = total2.clone(); let free2_c = free2.clone();
                                let _ = stream_match_csv(&pool1, &table1, &table2, MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd, |p| { a2+=1; xw.append_algo2(p) }, scfg.clone(), move |u| { if u.gpu_active { a2_used_c.store(true, Ordering::Relaxed); } total2_c.store(u.gpu_total_mb as u64, Ordering::Relaxed); free2_c.store(u.gpu_free_mb as u64, Ordering::Relaxed); let _ = txp2.send(Msg::Progress(u)); }, ctrl.clone()).await?;
                            }
                            // Fetch row counts for summary
                            let c1 = get_person_count(&pool1, &table1).await?;
                            let c2 = if let Some(pool2) = pool2_opt.as_ref() { get_person_count(pool2, &table2).await? } else { get_person_count(&pool1, &table2).await? };

                            let run_end_utc = chrono::Utc::now();
                            let duration_secs = (run_end_utc - run_start_utc).num_milliseconds() as f64 / 1000.0;
                            let a1_gpu = a1_used.load(Ordering::Relaxed);
                            let a2_gpu = a2_used.load(Ordering::Relaxed);
                            let gpu_used = a1_gpu || a2_gpu;
                            let gpu_total_mb = std::cmp::max(total1.load(Ordering::Relaxed), total2.load(Ordering::Relaxed));
                            let gpu_free_mb_end = if a2_gpu || total2.load(Ordering::Relaxed) > 0 { free2.load(Ordering::Relaxed) } else { free1.load(Ordering::Relaxed) };

                            xw.finalize(&SummaryContext {
                                db_name: db_label.clone(), table1: table1.clone(), table2: table2.clone(), total_table1: c1 as usize, total_table2: c2 as usize,
                                matches_algo1: a1, matches_algo2: a2, matches_fuzzy: 0, overlap_count: 0, unique_algo1: a1, unique_algo2: a2,
                                fetch_time: std::time::Duration::from_secs(0), match1_time: std::time::Duration::from_secs(0), match2_time: std::time::Duration::from_secs(0),
                                export_time: std::time::Duration::from_secs(0), mem_used_start_mb: 0, mem_used_end_mb: 0,
                                started_utc: run_start_utc, ended_utc: run_end_utc, duration_secs,
                                exec_mode_algo1: Some(if a1_gpu { "GPU".into() } else { "CPU".into() }), exec_mode_algo2: Some(if a2_gpu { "GPU".into() } else { "CPU".into() }), exec_mode_fuzzy: None,
                                algo_used: "Both (1,2)".into(), gpu_used, gpu_total_mb, gpu_free_mb_end,
                                adv_level: None,
                                adv_level_description: None,
                            })?;
                            Ok((a1,a2,0,path.clone(),false,false))
                        } else {
                            let t1 = get_person_rows(&pool1, &table1).await?;
                            let t2 = if let Some(pool2) = pool2_opt.as_ref() { get_person_rows(pool2, &table2).await? } else { get_person_rows(&pool1, &table2).await? };
                                // Apply normalization alignment for in-memory deterministic algorithms
                                name_matcher::matching::set_direct_normalization_fuzzy(direct_norm_fuzzy);

                            let cfgp = ProgressConfig { update_every: 10_000, ..Default::default() };
                            let mut xw = XlsxStreamWriter::create_with_extra_fields(&path, extra_field_names.clone())?;
                            let mut a1: usize = 0; let mut a2: usize = 0;
                            if matches!(algo, MatchingAlgorithm::HouseholdGpu) {
                                let txp = tx_for_async.clone();
                                let rows = match_households_gpu_inmemory(&t1, &t2, MatchOptions { backend: if use_gpu { ComputeBackend::Gpu } else { ComputeBackend::Cpu }, gpu: Some(GpuConfig { device_id: None, mem_budget_mb: gpu_mem }), progress: cfgp, allow_birthdate_swap }, fuzzy_thr, move |u| {
                                    log::info!("[inmem-household-xlsx] {:.1}% | ETA: {}s | Mem: {} MB | Avail: {} MB | {}/{}", u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total);
                                    let _ = txp.send(Msg::Progress(u));
                                });
                                export_households_xlsx(&path, &rows)?;
                                return Ok((0, 0, 0, path.clone(),false,false));
                            } else if matches!(algo, MatchingAlgorithm::HouseholdGpuOpt6) {
                                let txp = tx_for_async.clone();
                                let rows = match_households_gpu_inmemory_opt6(&t1, &t2, MatchOptions { backend: if use_gpu { ComputeBackend::Gpu } else { ComputeBackend::Cpu }, gpu: Some(GpuConfig { device_id: None, mem_budget_mb: gpu_mem }), progress: cfgp, allow_birthdate_swap }, fuzzy_thr, move |u| {
                                    log::info!("[inmem-household-opt6-xlsx] {:.1}% | ETA: {}s | Mem: {} MB | Avail: {} MB | {}/{}", u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total);
                                    let _ = txp.send(Msg::Progress(u));
                                });
                                export_households_xlsx(&path, &rows)?;
                                return Ok((0, 0, 0, path.clone(),false,false));
                            } else {
                                let txp1 = tx_for_async.clone();
                                // Also reflect GPU fuzzy pre-pass for in-memory runs
                                name_matcher::matching::set_gpu_fuzzy_direct_prep(use_gpu_fuzzy_direct_hash);

                                // using global run_start_utc captured earlier
                                use std::sync::{Arc}; use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
                                let a1_used = Arc::new(AtomicBool::new(false));
                                let a2_used = Arc::new(AtomicBool::new(false));
                                let total1 = Arc::new(AtomicU64::new(0));
                                let free1 = Arc::new(AtomicU64::new(0));
                                let total2 = Arc::new(AtomicU64::new(0));
                                let free2 = Arc::new(AtomicU64::new(0));

                                let a1_used_c = a1_used.clone(); let total1_c = total1.clone(); let free1_c = free1.clone();
                                let pairs1 = match_all_progress(&t1, &t2, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, cfgp, move |u| { if u.gpu_active { a1_used_c.store(true, Ordering::Relaxed); } total1_c.store(u.gpu_total_mb as u64, Ordering::Relaxed); free1_c.store(u.gpu_free_mb as u64, Ordering::Relaxed); let _ = txp1.send(Msg::Progress(u)); });
                                for p in &pairs1 { xw.append_algo1(p)?; }
                                let txp2 = tx_for_async.clone();
                                let a2_used_c = a2_used.clone(); let total2_c = total2.clone(); let free2_c = free2.clone();
                                let pairs2 = match_all_progress(&t1, &t2, MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd, cfgp, move |u| { if u.gpu_active { a2_used_c.store(true, Ordering::Relaxed); } total2_c.store(u.gpu_total_mb as u64, Ordering::Relaxed); free2_c.store(u.gpu_free_mb as u64, Ordering::Relaxed); let _ = txp2.send(Msg::Progress(u)); });
                                for p in &pairs2 { xw.append_algo2(p)?; }
                                a1 = pairs1.len(); a2 = pairs2.len();

                                let run_end_utc = chrono::Utc::now();
                                let duration_secs = (run_end_utc - run_start_utc).num_milliseconds() as f64 / 1000.0;
                                let a1_gpu = a1_used.load(Ordering::Relaxed);
                                let a2_gpu = a2_used.load(Ordering::Relaxed);
                                let gpu_used = a1_gpu || a2_gpu;
                                let gpu_total_mb = std::cmp::max(total1.load(Ordering::Relaxed), total2.load(Ordering::Relaxed));
                                let gpu_free_mb_end = if a2_gpu || total2.load(Ordering::Relaxed) > 0 { free2.load(Ordering::Relaxed) } else { free1.load(Ordering::Relaxed) };

                                xw.finalize(&SummaryContext {
                                    db_name: db_label.clone(), table1: table1.clone(), table2: table2.clone(), total_table1: t1.len(), total_table2: t2.len(),
                                    matches_algo1: a1, matches_algo2: a2, matches_fuzzy: 0, overlap_count: 0, unique_algo1: a1, unique_algo2: a2,
                                    fetch_time: std::time::Duration::from_secs(0), match1_time: std::time::Duration::from_secs(0), match2_time: std::time::Duration::from_secs(0),
                                    export_time: std::time::Duration::from_secs(0), mem_used_start_mb: 0, mem_used_end_mb: 0,
                                    started_utc: run_start_utc, ended_utc: run_end_utc, duration_secs,
                                    exec_mode_algo1: Some(if a1_gpu { "GPU".into() } else { "CPU".into() }), exec_mode_algo2: Some(if a2_gpu { "GPU".into() } else { "CPU".into() }), exec_mode_fuzzy: None,
                                    algo_used: "Both (1,2)".into(), gpu_used, gpu_total_mb, gpu_free_mb_end,
                                    adv_level: None,
                                    adv_level_description: None,
                                })?;
                                return Ok((a1,a2,0,path.clone(),false,false));
                            }
                        }
                    }
                    FormatSel::Both => {
                        let mut use_streaming = match mode { ModeSel::Streaming => true, ModeSel::InMemory => false, ModeSel::Auto => true };
                            if matches!(algo, MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle | MatchingAlgorithm::LevenshteinWeighted) { use_streaming = false; let _ = tx_for_async.send(Msg::Info("Selected algorithm uses in-memory mode (streaming disabled)".into())); }
                        let mut csv_path = path.clone(); if !csv_path.to_ascii_lowercase().ends_with(".csv") { csv_path.push_str(".csv"); }
                        let mut csv_count = 0usize;
                        if use_streaming {
                            let mut w = CsvStreamWriter::create_with_extra_fields(&csv_path, algo, fuzzy_thr, extra_field_names.clone())?;
                            let flush_every = scfg.flush_every;
                            let txp3 = tx_for_async.clone();
                            if let Some(pool2) = pool2_opt.as_ref() {
                                let _ = stream_match_csv_dual(&pool1, pool2, &table1, &table2, algo, |p| {
                                    if matches!(algo, MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle) {
                                        if p.confidence >= 0.95 { csv_count += 1; }
                                    } else { csv_count += 1; }
                                    w.write(p)?;
                                    if csv_count % flush_every == 0 { w.flush_partial()?; }
                                    Ok(())
                                }, scfg.clone(), move |u| { let _ = txp3.send(Msg::Progress(u)); }, ctrl.clone()).await?;
                            } else {
                                let _ = stream_match_csv(&pool1, &table1, &table2, algo, |p| {
                                    if matches!(algo, MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle) {
                                        if p.confidence >= 0.95 { csv_count += 1; }
                                    } else { csv_count += 1; }
                                    w.write(p)?;
                                    if csv_count % flush_every == 0 { w.flush_partial()?; }
                                    Ok(())
                                }, scfg.clone(), move |u| { let _ = txp3.send(Msg::Progress(u)); }, ctrl.clone()).await?;
                            }
                            w.flush()?;
                        } else {
                            let t1 = get_person_rows(&pool1, &table1).await?;
                                // Apply normalization alignment for in-memory A1/A2
                                name_matcher::matching::set_direct_normalization_fuzzy(direct_norm_fuzzy);

                            let t2 = if let Some(pool2) = pool2_opt.as_ref() { get_person_rows(pool2, &table2).await? } else { get_person_rows(&pool1, &table2).await? };
                            let cfgp = ProgressConfig { update_every: 10_000, ..Default::default() };
                            let txp = tx_for_async.clone();
                            let pairs = if matches!(algo, MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle) {
                                let mo = MatchOptions { backend: if use_gpu { ComputeBackend::Gpu } else { ComputeBackend::Cpu }, gpu: Some(GpuConfig { device_id: None, mem_budget_mb: gpu_mem }), progress: cfgp, allow_birthdate_swap };
                                match_all_with_opts(&t1, &t2, algo, mo, move |u| {
                                    log::info!("[inmem-both] {:.1}% | ETA: {}s | Mem: {} MB | Avail: {} MB | {}/{}", u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total);
                                    let _ = txp.send(Msg::Progress(u));
                                })
                            } else {
                                match_all_progress(&t1, &t2, algo, cfgp, move |u| {
                                    log::info!("[inmem-both] {:.1}% | ETA: {}s | Mem: {} MB | Avail: {} MB | {}/{}", u.percent, u.eta_secs, u.mem_used_mb, u.mem_avail_mb, u.processed, u.total);
                                    let _ = txp.send(Msg::Progress(u));
                                })
                            };
                            if matches!(algo, MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle) {
                                let thr = fuzzy_thr;
                                csv_count = pairs.iter().filter(|p| p.confidence >= thr).count();
                            } else { csv_count = pairs.len(); }
                            let mut w = CsvStreamWriter::create_with_extra_fields(&csv_path, algo, fuzzy_thr, extra_field_names.clone())?;
                            for p in &pairs { w.write(p)?; }
                            w.flush()?;
                        }
                        let xlsx_path = if path.to_ascii_lowercase().ends_with(".xlsx") { path.clone() } else { path.replace(".csv", ".xlsx") };
                        if use_streaming {
                            let mut xw = XlsxStreamWriter::create_with_extra_fields(&xlsx_path, extra_field_names.clone())?;
                            let mut a1 = 0usize; let mut a2 = 0usize;
                            // Track timing and GPU for summary (uses outer run_start_utc)
                            use std::sync::{Arc}; use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
                            let a1_used = Arc::new(AtomicBool::new(false));
                            let a2_used = Arc::new(AtomicBool::new(false));
                            let total1 = Arc::new(AtomicU64::new(0));
                            let free1 = Arc::new(AtomicU64::new(0));
                            let total2 = Arc::new(AtomicU64::new(0));
                            let free2 = Arc::new(AtomicU64::new(0));

                            let txp4 = tx_for_async.clone();
                            if let Some(pool2) = pool2_opt.as_ref() {
                                let a1_used_c = a1_used.clone(); let total1_c = total1.clone(); let free1_c = free1.clone();
                                let _ = stream_match_csv_dual(&pool1, pool2, &table1, &table2, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, |p| { a1+=1; xw.append_algo1(p) }, scfg.clone(), move |u| {
                                    if u.gpu_active { a1_used_c.store(true, Ordering::Relaxed); }
                                    total1_c.store(u.gpu_total_mb as u64, Ordering::Relaxed);
                                    free1_c.store(u.gpu_free_mb as u64, Ordering::Relaxed);
                                    let _ = txp4.send(Msg::Progress(u));
                                }, ctrl.clone()).await?;
                            } else {
                                let a1_used_c = a1_used.clone(); let total1_c = total1.clone(); let free1_c = free1.clone();
                                let _ = stream_match_csv(&pool1, &table1, &table2, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, |p| { a1+=1; xw.append_algo1(p) }, scfg.clone(), move |u| {
                                    if u.gpu_active { a1_used_c.store(true, Ordering::Relaxed); }
                                    total1_c.store(u.gpu_total_mb as u64, Ordering::Relaxed);
                                    free1_c.store(u.gpu_free_mb as u64, Ordering::Relaxed);
                                    let _ = txp4.send(Msg::Progress(u));
                                }, ctrl.clone()).await?;
                            }
                            let txp5 = tx_for_async.clone();
                            if let Some(pool2) = pool2_opt.as_ref() {
                                let a2_used_c = a2_used.clone(); let total2_c = total2.clone(); let free2_c = free2.clone();
                                let _ = stream_match_csv_dual(&pool1, pool2, &table1, &table2, MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd, |p| { a2+=1; xw.append_algo2(p) }, scfg.clone(), move |u| {
                                    if u.gpu_active { a2_used_c.store(true, Ordering::Relaxed); }
                                    total2_c.store(u.gpu_total_mb as u64, Ordering::Relaxed);
                                    free2_c.store(u.gpu_free_mb as u64, Ordering::Relaxed);
                                    let _ = txp5.send(Msg::Progress(u));
                                }, ctrl.clone()).await?;
                            } else {
                                let a2_used_c = a2_used.clone(); let total2_c = total2.clone(); let free2_c = free2.clone();
                                let _ = stream_match_csv(&pool1, &table1, &table2, MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd, |p| { a2+=1; xw.append_algo2(p) }, scfg.clone(), move |u| {
                                    if u.gpu_active { a2_used_c.store(true, Ordering::Relaxed); }
                                    total2_c.store(u.gpu_total_mb as u64, Ordering::Relaxed);
                                    free2_c.store(u.gpu_free_mb as u64, Ordering::Relaxed);
                                    let _ = txp5.send(Msg::Progress(u));
                                }, ctrl.clone()).await?;
                            }
                            // Fetch row counts for summary
                            let c1 = get_person_count(&pool1, &table1).await?;
                            let c2 = if let Some(pool2) = pool2_opt.as_ref() { get_person_count(pool2, &table2).await? } else { get_person_count(&pool1, &table2).await? };

                            // Compute summary timing and GPU fields
                            let run_end_utc = chrono::Utc::now();
                            let duration_secs = (run_end_utc - run_start_utc).num_milliseconds() as f64 / 1000.0;
                            let a1_gpu = a1_used.load(Ordering::Relaxed);
                            let a2_gpu = a2_used.load(Ordering::Relaxed);
                            let gpu_used = a1_gpu || a2_gpu;
                            let gpu_total_mb = std::cmp::max(total1.load(Ordering::Relaxed), total2.load(Ordering::Relaxed));
                            let gpu_free_mb_end = if a2_gpu || total2.load(Ordering::Relaxed) > 0 { free2.load(Ordering::Relaxed) } else { free1.load(Ordering::Relaxed) };

                            xw.finalize(&SummaryContext {
                                db_name: db_label.clone(), table1: table1.clone(), table2: table2.clone(), total_table1: c1 as usize, total_table2: c2 as usize,
                                matches_algo1: a1, matches_algo2: a2, matches_fuzzy: 0, overlap_count: 0, unique_algo1: a1, unique_algo2: a2,
                                fetch_time: std::time::Duration::from_secs(0), match1_time: std::time::Duration::from_secs(0), match2_time: std::time::Duration::from_secs(0),
                                export_time: std::time::Duration::from_secs(0), mem_used_start_mb: 0, mem_used_end_mb: 0,
                                started_utc: run_start_utc, ended_utc: run_end_utc, duration_secs,
                                exec_mode_algo1: Some(if a1_gpu { "GPU".into() } else { "CPU".into() }), exec_mode_algo2: Some(if a2_gpu { "GPU".into() } else { "CPU".into() }), exec_mode_fuzzy: None,
                                algo_used: "Both (1,2)".into(), gpu_used, gpu_total_mb, gpu_free_mb_end,
                                adv_level: None,
                                adv_level_description: None,
                            })?;
                            Ok((a1,a2,csv_count,path.clone(),false,false))
                        } else {
                            let t1 = get_person_rows(&pool1, &table1).await?;
                            let t2 = if let Some(pool2) = pool2_opt.as_ref() { get_person_rows(pool2, &table2).await? } else { get_person_rows(&pool1, &table2).await? };
                            let cfgp = ProgressConfig { update_every: 10_000, ..Default::default() };
                            let mut xw = XlsxStreamWriter::create_with_extra_fields(&xlsx_path, extra_field_names.clone())?;
                            // Track timing and GPU for summary (in-memory Both; uses outer run_start_utc)
                            use std::sync::{Arc}; use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
                            let a1_used = Arc::new(AtomicBool::new(false));
                            let a2_used = Arc::new(AtomicBool::new(false));
                            let total1 = Arc::new(AtomicU64::new(0));
                            let free1 = Arc::new(AtomicU64::new(0));
                            let total2 = Arc::new(AtomicU64::new(0));
                            let free2 = Arc::new(AtomicU64::new(0));
                            let txp1 = tx_for_async.clone();
                            let a1_used_c = a1_used.clone(); let total1_c = total1.clone(); let free1_c = free1.clone();
                            let pairs1 = match_all_progress(&t1, &t2, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, cfgp, move |u| {
                                if u.gpu_active { a1_used_c.store(true, Ordering::Relaxed); }
                                total1_c.store(u.gpu_total_mb as u64, Ordering::Relaxed);
                                free1_c.store(u.gpu_free_mb as u64, Ordering::Relaxed);
                                let _ = txp1.send(Msg::Progress(u));
                            });
                            for p in &pairs1 { xw.append_algo1(p)?; }
                            let txp2 = tx_for_async.clone();
                            let a2_used_c = a2_used.clone(); let total2_c = total2.clone(); let free2_c = free2.clone();
                            let pairs2 = match_all_progress(&t1, &t2, MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd, cfgp, move |u| {
                                if u.gpu_active { a2_used_c.store(true, Ordering::Relaxed); }
                                total2_c.store(u.gpu_total_mb as u64, Ordering::Relaxed);
                                free2_c.store(u.gpu_free_mb as u64, Ordering::Relaxed);
                                let _ = txp2.send(Msg::Progress(u));
                            });
                            for p in &pairs2 { xw.append_algo2(p)?; }
                            let a1 = pairs1.len(); let a2 = pairs2.len();
                            // Compute timing and GPU fields
                            let run_end_utc = chrono::Utc::now();
                            let duration_secs = (run_end_utc - run_start_utc).num_milliseconds() as f64 / 1000.0;
                            let a1_gpu = a1_used.load(Ordering::Relaxed);
                            let a2_gpu = a2_used.load(Ordering::Relaxed);
                            let gpu_used = a1_gpu || a2_gpu;
                            let gpu_total_mb = std::cmp::max(total1.load(Ordering::Relaxed), total2.load(Ordering::Relaxed));
                            let gpu_free_mb_end = if a2_gpu || total2.load(Ordering::Relaxed) > 0 { free2.load(Ordering::Relaxed) } else { free1.load(Ordering::Relaxed) };
                            xw.finalize(&SummaryContext {
                                db_name: db_label.clone(), table1: table1.clone(), table2: table2.clone(), total_table1: t1.len(), total_table2: t2.len(),
                                matches_algo1: a1, matches_algo2: a2, matches_fuzzy: 0, overlap_count: 0, unique_algo1: a1, unique_algo2: a2,
                                fetch_time: std::time::Duration::from_secs(0), match1_time: std::time::Duration::from_secs(0), match2_time: std::time::Duration::from_secs(0),
                                export_time: std::time::Duration::from_secs(0), mem_used_start_mb: 0, mem_used_end_mb: 0,
                                started_utc: run_start_utc, ended_utc: run_end_utc, duration_secs,
                                exec_mode_algo1: Some(if a1_gpu { "GPU".into() } else { "CPU".into() }), exec_mode_algo2: Some(if a2_gpu { "GPU".into() } else { "CPU".into() }), exec_mode_fuzzy: None,
                                algo_used: "Both (1,2)".into(), gpu_used, gpu_total_mb, gpu_free_mb_end,
                                adv_level: None,
                                adv_level_description: None,
                            })?;
                            Ok((a1,a2,csv_count,path.clone(),false,false))
                        }
                    }
                }
            });
            match res {
                Ok((a1, a2, csv, out_path, gpu_hash_used, gpu_fuzzy_used)) => {
                    let _ = tx.send(Msg::Done {
                        a1,
                        a2,
                        csv,
                        path: out_path,
                        gpu_hash_used,
                        gpu_fuzzy_used,
                    });
                }
                Err(e) => {
                    let (sqlstate, chain) = extract_sqlstate_and_chain(&e);
                    let _ = tx.send(Msg::ErrorRich {
                        display: format!("{}", e),
                        sqlstate,
                        chain,
                        operation: Some("Run Matching".into()),
                    });
                }
            }
        });
    }

    fn reset_state(&mut self) {
        *self = GuiApp::default();
    }
    fn test_connection(&mut self) {
        let (tx, rx) = mpsc::channel::<Msg>();
        self.tx = Some(tx.clone());
        // Forward subsequent log::info!/warn!/error! to the current GUI session
        set_gui_log_sender(tx.clone());
        // Forward subsequent log::info!/warn!/error! to the current GUI session
        set_gui_log_sender(tx.clone());
        self.last_action = "Test Connection".into();
        let enable_dual = self.enable_dual;
        // DB1
        let host = self.host.clone();
        let port = self.port.clone();
        let user = self.user.clone();
        let pass = self.pass.clone();
        let dbname = self.db.clone();
        let t1 = self.tables.get(self.table1_idx).cloned();
        // DB2 (optional)
        let host2 = self.host2.clone();
        let port2 = self.port2.clone();
        let user2 = self.user2.clone();
        let pass2 = self.pass2.clone();
        let dbname2 = self.db2.clone();
        let t2 = if enable_dual {
            self.tables2.get(self.table2_idx).cloned()
        } else {
            self.tables.get(self.table2_idx).cloned()
        };
        thread::spawn(move || {
            let rt = gui_runtime();
            let res: Result<(Option<i64>, Option<i64>)> = rt.block_on(async move {
                if enable_dual {
                    // connect to both DBs
                    let cfg1 = DatabaseConfig {
                        host,
                        port: port.parse().unwrap_or(3306),
                        username: user,
                        password: pass,
                        database: dbname,
                    };
                    let pool1 = make_pool_with_size(&cfg1, Some(4)).await?;
                    let _pong1: i32 = sqlx::query_scalar("SELECT 1").fetch_one(&pool1).await?;
                    let cfg2 = DatabaseConfig {
                        host: host2,
                        port: port2.parse().unwrap_or(3306),
                        username: user2,
                        password: pass2,
                        database: dbname2,
                    };
                    let pool2 = make_pool_with_size(&cfg2, Some(4)).await?;
                    let _pong2: i32 = sqlx::query_scalar("SELECT 1").fetch_one(&pool2).await?;
                    let c1 = if let Some(t) = t1.as_ref() {
                        Some(get_person_count(&pool1, t).await?)
                    } else {
                        None
                    };
                    let c2 = if let Some(t) = t2.as_ref() {
                        Some(get_person_count(&pool2, t).await?)
                    } else {
                        None
                    };
                    Ok((c1, c2))
                } else {
                    let cfg = DatabaseConfig {
                        host,
                        port: port.parse().unwrap_or(3306),
                        username: user,
                        password: pass,
                        database: dbname,
                    };
                    let pool = make_pool_with_size(&cfg, Some(4)).await?;
                    let _pong: i32 = sqlx::query_scalar("SELECT 1").fetch_one(&pool).await?;
                    let c1 = if let Some(t) = t1.as_ref() {
                        Some(get_person_count(&pool, t).await?)
                    } else {
                        None
                    };
                    let c2 = if let Some(t) = t2.as_ref() {
                        Some(get_person_count(&pool, t).await?)
                    } else {
                        None
                    };
                    Ok((c1, c2))
                }
            });
            match res {
                Ok((c1, c2)) => {
                    let _ = tx.send(Msg::Info(format!(
                        "Connected. Row counts: t1={:?}, t2={:?}",
                        c1, c2
                    )));
                }
                Err(e) => {
                    let (sqlstate, chain) = extract_sqlstate_and_chain(&e);
                    let _ = tx.send(Msg::ErrorRich {
                        display: format!("Connection failed: {}", e),
                        sqlstate,
                        chain,
                        operation: Some("Connect/Test".into()),
                    });
                }
            }
        });
        self.rx = rx;
        self.status = "Testing connection...".into();
    }

    fn estimate(&mut self) {
        let (tx, rx) = mpsc::channel::<Msg>();
        self.tx = Some(tx.clone());
        self.last_action = "Estimate".into();
        let host = self.host.clone();
        let port = self.port.clone();
        let user = self.user.clone();
        let pass = self.pass.clone();
        let dbname = self.db.clone();
        let t1 = self.tables.get(self.table1_idx).cloned();
        let t2 = self.tables.get(self.table2_idx).cloned();
        let mem_thr_mb = self.mem_thresh.parse::<u64>().unwrap_or(800);
        thread::spawn(move || {
            let rt = gui_runtime();
            let res: Result<String> = rt.block_on(async move {
                let cfg = DatabaseConfig {
                    host,
                    port: port.parse().unwrap_or(3306),
                    username: user,
                    password: pass,
                    database: dbname,
                };
                let pool = make_pool_with_size(&cfg, Some(4)).await?;
                let (c1, c2) = match (t1.as_ref(), t2.as_ref()) {
                    (Some(a), Some(b)) => (
                        get_person_count(&pool, a).await?,
                        get_person_count(&pool, b).await?,
                    ),
                    _ => (0, 0),
                };
                let small = c1.min(c2) as u64;
                // very rough estimate: ~96 bytes per index row; batch overhead ~ 64 bytes/row
                let index_bytes = small.saturating_mul(96);
                let index_mb = (index_bytes as f64 / (1024.0 * 1024.0)).ceil() as u64;
                let suggestion = if index_mb > mem_thr_mb {
                    "Streaming"
                } else {
                    "In-memory"
                };
                Ok(format!(
                    "Estimated index ~{} MB ({} vs {}). Suggested mode: {} (threshold {} MB)",
                    index_mb, c1, c2, suggestion, mem_thr_mb
                ))
            });
            match res {
                Ok(s) => {
                    let _ = tx.send(Msg::Info(s));
                }
                Err(e) => {
                    let (sqlstate, chain) = extract_sqlstate_and_chain(&e);
                    let _ = tx.send(Msg::ErrorRich {
                        display: format!("Estimate failed: {}", e),
                        sqlstate,
                        chain,
                        operation: Some("Estimate Resources".into()),
                    });
                }
            }
        });
        self.rx = rx;
        self.status = "Estimating...".into();
    }

    fn poll_messages(&mut self) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                Msg::Progress(u) => {
                    self.progress = u.percent;
                    self.eta_secs = u.eta_secs;
                    self.mem_used = u.mem_used_mb;
                    self.mem_avail = u.mem_avail_mb;
                    self.gpu_total_mb = u.gpu_total_mb;
                    self.gpu_free_mb = u.gpu_free_mb;
                    self.gpu_active = u.gpu_active;
                    self.processed = u.processed;
                    self.total = u.total;
                    self.batch_current = u.batch_size_current.unwrap_or(0);
                    self.stage = u.stage.to_string();
                    // Runtime monitoring: warn if free memory dips below threshold; streaming will reduce batch size automatically
                    let mem_thr = self.mem_thresh.parse::<u64>().unwrap_or(800);
                    if self.mem_avail < mem_thr {
                        self.status = format!(
                            "Warning: low free memory ({} MB < {} MB). Auto-throttling batches.",
                            self.mem_avail, mem_thr
                        );
                    }
                    // Separate GPU build/probe active indicators by stage hint
                    match u.stage {
                        "gpu_hash" => {
                            self.gpu_build_active_now = true;
                        }
                        "gpu_hash_done" => {
                            self.gpu_build_active_now = true;
                        }
                        "gpu_probe_hash" => {
                            self.gpu_probe_active_now = true;
                        }
                        "gpu_probe_hash_done" => {
                            self.gpu_probe_active_now = true;
                        }
                        _ => {}
                    }
                    let now = std::time::Instant::now();

                    if let Some(t0) = self.last_tick {
                        let dt = now.duration_since(t0).as_secs_f32().max(1e-3);
                        self.rps =
                            ((self.processed.saturating_sub(self.last_processed_prev)) as f32 / dt)
                                .max(0.0);
                    } else {
                        self.rps = 0.0;
                    }
                    self.last_tick = Some(now);
                    self.last_processed_prev = self.processed;
                    self.status = format!(
                        "{} | {:.1}% | {} / {} recs | {:.0} rec/s",
                        self.stage, u.percent, self.processed, self.total, self.rps
                    );

                    // log tail
                    let line = format!(
                        "{} PROGRESS stage={} percent={:.1} processed={}/{} rps={:.0}",
                        chrono::Utc::now().to_rfc3339(),
                        self.stage,
                        u.percent,
                        self.processed,
                        self.total,
                        self.rps
                    );
                    self.log_buffer.push(line);
                    if self.log_buffer.len() > 200 {
                        let drop = self.log_buffer.len() - 200;
                        self.log_buffer.drain(0..drop);
                    }
                }
                Msg::Info(s) => {
                    self.status = s.clone();
                    let line = format!("{} INFO {}", chrono::Utc::now().to_rfc3339(), s);
                    self.log_buffer.push(line);
                    if self.log_buffer.len() > 200 {
                        let drop = self.log_buffer.len() - 200;
                        self.log_buffer.drain(0..drop);
                    }
                }
                Msg::Tables(v) => {
                    self.tables = v;
                    self.table1_idx = 0;
                    if self.table1_idx >= self.tables.len() {
                        self.table1_idx = 0;
                    }
                    self.status = format!("Loaded {} tables (DB1)", self.tables.len());
                }
                Msg::Tables2(v2) => {
                    self.tables2 = v2;
                    self.table2_idx = 0;
                    if self.table2_idx >= self.tables2.len() {
                        self.table2_idx = 0;
                    }
                    self.status = format!("Loaded {} tables (DB2)", self.tables2.len());
                }
                Msg::DbPools { pool1, pool2 } => {
                    self.pool1_cache = Some(pool1);
                    self.pool2_cache = pool2;
                    self.status = "Connection ready".into();
                }

                Msg::Done {
                    a1,
                    a2,
                    csv,
                    path,
                    gpu_hash_used,
                    gpu_fuzzy_used,
                } => {
                    self.running = false;
                    self.a1_count = a1;
                    self.a2_count = a2;
                    self.csv_count = csv;
                    self.progress = 100.0;
                    // Stop external console tail and close log file
                    if let Some(mut ch) = self.console_child.take() {
                        let _ = ch.kill();
                    }
                    if let Some(m) = GUI_LOG_FILE.get() {
                        let _ = m.lock().map(|mut g| *g = None);
                    }
                    if self.use_gpu
                        && matches!(
                            self.algo,
                            MatchingAlgorithm::Fuzzy
                                | MatchingAlgorithm::FuzzyNoMiddle
                                | MatchingAlgorithm::LevenshteinWeighted
                        )
                        && self.gpu_total_mb == 0
                    {
                        self.status = format!(
                            "Done (CPU fallback — no GPU activity detected). Output: {}",
                            path
                        );
                    } else {
                        // Write standalone summary CSV/XLSX alongside output
                        let db_label = if self.enable_dual {
                            format!("{} | {}", self.db, self.db2)
                        } else {
                            self.db.clone()
                        };
                        let t1 = self
                            .tables
                            .get(self.table1_idx)
                            .cloned()
                            .unwrap_or_default();
                        let t2 = if self.enable_dual {
                            self.tables2
                                .get(self.table2_idx)
                                .cloned()
                                .unwrap_or_default()
                        } else {
                            self.tables
                                .get(self.table2_idx)
                                .cloned()
                                .unwrap_or_default()
                        };
                        // Determine matches_fuzzy based on algorithm type:
                        // - Standard fuzzy algorithms (Fuzzy, FuzzyNoMiddle, LevenshteinWeighted): use csv count
                        // - Advanced L10-L12 (fuzzy levels): use csv count
                        // - Advanced L1-L9 (exact levels): use csv count (displayed as "Direct Match" in summary)
                        // - Options 1-7 (exact algorithms): use 0 (they use matches_algo1/algo2 instead)
                        let matches_fuzzy = if matches!(
                            self.algo,
                            MatchingAlgorithm::Fuzzy
                                | MatchingAlgorithm::FuzzyNoMiddle
                                | MatchingAlgorithm::LevenshteinWeighted
                        ) {
                            csv
                        } else if self.advanced_enabled
                            && matches!(
                                self.adv_level,
                                Some(AdvLevel::L10FuzzyBirthdateFullMiddle)
                                    | Some(AdvLevel::L11FuzzyBirthdateNoMiddle)
                                    | Some(AdvLevel::L12HouseholdMatching)
                            )
                        {
                            csv
                        } else if self.advanced_enabled
                            && matches!(
                                self.adv_level,
                                Some(AdvLevel::L1BirthdateFullMiddle)
                                    | Some(AdvLevel::L2BirthdateMiddleInitial)
                                    | Some(AdvLevel::L3BirthdateNoMiddle)
                                    | Some(AdvLevel::L4BarangayFullMiddle)
                                    | Some(AdvLevel::L5BarangayMiddleInitial)
                                    | Some(AdvLevel::L6BarangayNoMiddle)
                                    | Some(AdvLevel::L7CityFullMiddle)
                                    | Some(AdvLevel::L8CityMiddleInitial)
                                    | Some(AdvLevel::L9CityNoMiddle)
                            )
                        {
                            csv
                        } else {
                            0
                        };
                        // Derive accurate table totals for summary using lightweight COUNT(*)
                        let (sum_c1, sum_c2) = {
                            let host = self.host.clone();
                            let port = self.port.clone();
                            let user = self.user.clone();
                            let pass = self.pass.clone();
                            let dbname = self.db.clone();
                            let host2 = self.host2.clone();
                            let port2 = self.port2.clone();
                            let user2 = self.user2.clone();
                            let pass2 = self.pass2.clone();
                            let dbname2 = self.db2.clone();
                            let t1n = self
                                .tables
                                .get(self.table1_idx)
                                .cloned()
                                .unwrap_or_default();
                            let t2n = if self.enable_dual {
                                self.tables2
                                    .get(self.table2_idx)
                                    .cloned()
                                    .unwrap_or_default()
                            } else {
                                self.tables
                                    .get(self.table2_idx)
                                    .cloned()
                                    .unwrap_or_default()
                            };
                            let enable_dual = self.enable_dual;
                            let rt = gui_runtime();
                            rt.block_on(async move {
                                use name_matcher::config::DatabaseConfig;
                                use name_matcher::db::connection::make_pool;
                                use name_matcher::db::get_person_count;
                                if enable_dual {
                                    let cfg1 = DatabaseConfig {
                                        host,
                                        port: port.parse().unwrap_or(3306),
                                        username: user,
                                        password: pass,
                                        database: dbname,
                                    };
                                    let cfg2 = DatabaseConfig {
                                        host: host2,
                                        port: port2.parse().unwrap_or(3306),
                                        username: user2,
                                        password: pass2,
                                        database: dbname2,
                                    };
                                    if let (Ok(p1), Ok(p2)) =
                                        (make_pool(&cfg1).await, make_pool(&cfg2).await)
                                    {
                                        let c1 = get_person_count(&p1, &t1n).await.unwrap_or(0);
                                        let c2 = get_person_count(&p2, &t2n).await.unwrap_or(0);
                                        (c1, c2)
                                    } else {
                                        (0, 0)
                                    }
                                } else {
                                    let cfg = DatabaseConfig {
                                        host,
                                        port: port.parse().unwrap_or(3306),
                                        username: user,
                                        password: pass,
                                        database: dbname,
                                    };
                                    if let Ok(p) = make_pool(&cfg).await {
                                        let c1 = get_person_count(&p, &t1n).await.unwrap_or(0);
                                        let c2 = get_person_count(&p, &t2n).await.unwrap_or(0);
                                        (c1, c2)
                                    } else {
                                        (0, 0)
                                    }
                                }
                            })
                        };

                        let started_utc =
                            self.run_started_utc.unwrap_or_else(|| chrono::Utc::now());
                        let ended_utc = chrono::Utc::now();
                        self.run_ended_utc = Some(ended_utc);
                        let duration_secs =
                            (ended_utc - started_utc).num_milliseconds() as f64 / 1000.0;

                        // Determine GPU usage based on Advanced Matching level and GPU flags
                        let gpu_used = if self.advanced_enabled {
                            match self.adv_level {
                                Some(AdvLevel::L1BirthdateFullMiddle)
                                | Some(AdvLevel::L2BirthdateMiddleInitial)
                                | Some(AdvLevel::L3BirthdateNoMiddle)
                                | Some(AdvLevel::L4BarangayFullMiddle)
                                | Some(AdvLevel::L5BarangayMiddleInitial)
                                | Some(AdvLevel::L6BarangayNoMiddle)
                                | Some(AdvLevel::L7CityFullMiddle)
                                | Some(AdvLevel::L8CityMiddleInitial)
                                | Some(AdvLevel::L9CityNoMiddle) => gpu_hash_used,
                                Some(AdvLevel::L10FuzzyBirthdateFullMiddle)
                                | Some(AdvLevel::L11FuzzyBirthdateNoMiddle)
                                | Some(AdvLevel::L12HouseholdMatching) => gpu_fuzzy_used,
                                _ => false,
                            }
                        } else {
                            self.gpu_active // For non-Advanced matching, use existing logic
                        };

                        let is_adv_fuzzy = self.advanced_enabled
                            && matches!(
                                self.adv_level,
                                Some(AdvLevel::L10FuzzyBirthdateFullMiddle)
                                    | Some(AdvLevel::L11FuzzyBirthdateNoMiddle)
                            );
                        let (exec_mode_algo1, exec_mode_algo2, exec_mode_fuzzy) = if matches!(
                            self.algo,
                            MatchingAlgorithm::Fuzzy
                                | MatchingAlgorithm::FuzzyNoMiddle
                                | MatchingAlgorithm::LevenshteinWeighted
                        )
                            || is_adv_fuzzy
                        {
                            (
                                None,
                                None,
                                Some(if gpu_used { "GPU".into() } else { "CPU".into() }),
                            )
                        } else {
                            (Some("CPU".into()), Some("CPU".into()), None)
                        };
                        // Friendly algorithm label for summaries; Advanced uses level label, others use Option labels
                        let algo_used = if self.advanced_enabled {
                            if let Some(level) = self.adv_level {
                                format!("Advanced {:?}", level)
                            } else {
                                Self::algorithm_label(self.algo).to_string()
                            }
                        } else {
                            Self::algorithm_label(self.algo).to_string()
                        };
                        let summary = SummaryContext {
                            db_name: db_label,
                            table1: t1,
                            table2: t2,
                            total_table1: sum_c1 as usize,
                            total_table2: sum_c2 as usize,
                            matches_algo1: a1,
                            matches_algo2: a2,
                            matches_fuzzy,
                            overlap_count: 0,
                            unique_algo1: a1,
                            unique_algo2: a2,
                            fetch_time: std::time::Duration::from_secs(0),
                            match1_time: std::time::Duration::from_secs(0),
                            match2_time: std::time::Duration::from_secs(0),
                            export_time: std::time::Duration::from_secs(0),
                            mem_used_start_mb: 0,
                            mem_used_end_mb: 0,
                            started_utc: started_utc,
                            ended_utc: ended_utc,
                            duration_secs,
                            exec_mode_algo1,
                            exec_mode_algo2,
                            exec_mode_fuzzy,
                            algo_used,
                            gpu_used,
                            gpu_total_mb: self.gpu_total_mb,
                            gpu_free_mb_end: self.gpu_free_mb,
                            adv_level: self.adv_level,
                            adv_level_description: self.adv_level.map(|lvl| match lvl {
                                AdvLevel::L1BirthdateFullMiddle => {
                                    "L1: Birthdate + Full Middle Name".to_string()
                                }
                                AdvLevel::L2BirthdateMiddleInitial => {
                                    "L2: Birthdate + Middle Initial".to_string()
                                }
                                AdvLevel::L3BirthdateNoMiddle => {
                                    "L3: Birthdate (No Middle)".to_string()
                                }
                                AdvLevel::L4BarangayFullMiddle => {
                                    "L4: Barangay + Full Middle Name".to_string()
                                }
                                AdvLevel::L5BarangayMiddleInitial => {
                                    "L5: Barangay + Middle Initial".to_string()
                                }
                                AdvLevel::L6BarangayNoMiddle => {
                                    "L6: Barangay (No Middle)".to_string()
                                }
                                AdvLevel::L7CityFullMiddle => {
                                    "L7: City + Full Middle Name".to_string()
                                }
                                AdvLevel::L8CityMiddleInitial => {
                                    "L8: City + Middle Initial".to_string()
                                }
                                AdvLevel::L9CityNoMiddle => "L9: City (No Middle)".to_string(),
                                AdvLevel::L10FuzzyBirthdateFullMiddle => {
                                    "L10: Fuzzy + Birthdate + Full Middle".to_string()
                                }
                                AdvLevel::L11FuzzyBirthdateNoMiddle => {
                                    "L11: Fuzzy + Birthdate (No Middle)".to_string()
                                }
                                AdvLevel::L12HouseholdMatching => {
                                    "L12: Household Matching".to_string()
                                }
                            }),
                        };
                        let out_dir = std::path::Path::new(&path)
                            .parent()
                            .unwrap_or(std::path::Path::new("."));
                        let ts = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
                        let sum_csv = out_dir.join(format!("summary_report_{}.csv", ts));
                        let sum_xlsx = out_dir.join(format!("summary_report_{}.xlsx", ts));
                        if let Err(e) = name_matcher::export::csv_export::export_summary_csv(
                            sum_csv.to_string_lossy().as_ref(),
                            &summary,
                        ) {
                            self.status =
                                format!("Finished, but failed to write summary CSV: {}", e);
                        }
                        if let Err(e) = name_matcher::export::xlsx_export::export_summary_xlsx(
                            sum_xlsx.to_string_lossy().as_ref(),
                            &summary,
                        ) {
                            self.status =
                                format!("Finished, but failed to write summary XLSX: {}", e);
                        }

                        self.cleanup_after_run("Run completed successfully");
                        self.status = format!("Done. Output: {}", path);
                    }
                }
                Msg::Error(e) => {
                    // Stop external console tail and close log file
                    if let Some(mut ch) = self.console_child.take() {
                        let _ = ch.kill();
                    }
                    if let Some(m) = GUI_LOG_FILE.get() {
                        let _ = m.lock().map(|mut g| *g = None);
                    }
                    self.running = false;
                    self.status = format!("Error: {}", e);
                    self.record_error(e);
                }
                Msg::ErrorRich {
                    display,
                    sqlstate,
                    chain,
                    operation,
                } => {
                    // Stop external console tail and close log file
                    if let Some(mut ch) = self.console_child.take() {
                        let _ = ch.kill();
                    }
                    if let Some(m) = GUI_LOG_FILE.get() {
                        let _ = m.lock().map(|mut g| *g = None);
                    }
                    self.running = false;
                    self.status = format!("Error: {}", display);
                    self.record_error_with_details(display, sqlstate, Some(chain), operation);
                }
            }
        }
    }

    fn record_error(&mut self, message: String) {
        let sanitize = |s: &str| -> String {
            if let Some(pos) = s.find("mysql://") {
                if let Some(at) = s[pos..].find('@') {
                    let mut out = s.to_string();
                    out.replace_range(pos..pos + at, "mysql://[REDACTED]");
                    return out;
                }
            }
            s.to_string()
        };
        let mstats = name_matcher::metrics::memory_stats_mb();
        let pool_sz = self.pool_size.parse::<u32>().unwrap_or(0);
        let envs = [
            "NAME_MATCHER_POOL_SIZE",
            "NAME_MATCHER_POOL_MIN",
            "NAME_MATCHER_STREAMING",
            "NAME_MATCHER_PARTITION",
            "NAME_MATCHER_ALLOW_BIRTHDATE_SWAP",
        ];
        let mut env_overrides = Vec::new();
        for k in envs {
            if let Ok(v) = std::env::var(k) {
                env_overrides.push((k.to_string(), v));
            }
        }
        let evt = DiagEvent {
            ts_utc: Utc::now().to_rfc3339(),
            category: categorize_error(&message),
            message: sanitize(&message),
            sqlstate: None,
            chain: None,
            operation: None,
            source_action: self.last_action.clone(),
            db1_host: self.host.clone(),
            db1_database: self.db.clone(),
            db2_host: if self.enable_dual {
                Some(self.host2.clone())
            } else {
                None
            },
            db2_database: if self.enable_dual {
                Some(self.db2.clone())
            } else {
                None
            },
            table1: self.tables.get(self.table1_idx).cloned(),
            table2: if self.enable_dual {
                self.tables2.get(self.table2_idx).cloned()
            } else {
                self.tables.get(self.table2_idx).cloned()
            },
            mem_avail_mb: mstats.avail_mb,
            pool_size_cfg: pool_sz,
            env_overrides,
        };
        self.error_events.push(evt);
    }

    fn export_error_report(&mut self) -> Result<String> {
        let ts = Utc::now().format("%Y%m%d_%H%M%S").to_string();
        let default_name = match self.report_format {
            ReportFormat::Text => format!("error_report_{}.txt", ts),
            ReportFormat::Json => format!("error_report_{}.json", ts),
        };
        let mut dialog = rfd::FileDialog::new().set_file_name(&default_name);
        match self.report_format {
            ReportFormat::Text => {
                dialog = dialog.add_filter("Text", &["txt"]);
            }
            ReportFormat::Json => {
                dialog = dialog.add_filter("JSON", &["json"]);
            }
        }
        let path = dialog
            .save_file()
            .map(|p| p.display().to_string())
            .unwrap_or(default_name);

        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0);
        let mem = name_matcher::metrics::memory_stats_mb();
        let ver = env!("CARGO_PKG_VERSION");
        let os = std::env::consts::OS;
        let arch = std::env::consts::ARCH;
        // Collect selected environment variables with sanitization
        let mut env_selected: Vec<(String, String)> = Vec::new();
        for (k, v) in std::env::vars() {
            if k.starts_with("SQLX_") || k == "RUST_LOG" || k == "RUST_BACKTRACE" {
                let vv = if v.contains("mysql://") {
                    v.replacen("mysql://", "mysql://[REDACTED]@", 1)
                } else {
                    v.clone()
                };
                env_selected.push((k, vv));
            }
        }

        let suggestions_for = |cat: ErrorCategory| -> Vec<&'static str> {
            match cat {
                ErrorCategory::DbConnection => vec![
                    "Verify host/port reachability (ping, firewall)",
                    "Check username/password and privileges",
                    "Confirm database name exists and user has access",
                ],
                ErrorCategory::TableValidation => vec![
                    "Ensure selected tables exist and user has SELECT permission",
                    "Click 'Load Tables' to refresh the list",
                ],
                ErrorCategory::SchemaValidation => vec![
                    "Add required columns and indexes (see README: Required Indexes)",
                    "Verify column types match expected formats",
                ],
                ErrorCategory::DataFormat => vec![
                    "Normalize/cleanse date formats (YYYY-MM-DD)",
                    "Fill required fields or exclude nulls where needed",
                ],
                ErrorCategory::ResourceConstraint => vec![
                    "Use Streaming mode and reduce batch size",
                    "Close other apps to free RAM; ensure sufficient disk space",
                ],
                ErrorCategory::Configuration => vec![
                    "Check environment variables (NAME_MATCHER_*)",
                    "Re-enter GUI settings and retry",
                ],
                ErrorCategory::Unknown => vec!["Review logs and contact support with this report"],
            }
        };

        match self.report_format {
            ReportFormat::Text => {
                let mut out = String::new();
                out.push_str(&format!(
                    "SRS-II Name Matching - Diagnostic Report\nVersion: {}\nTimestamp: {}\n\n",
                    ver,
                    Utc::now().to_rfc3339()
                ));
                out.push_str(&format!(
                    "System: os={} arch={} cores={} | mem_avail={} MB\n",
                    os, arch, cores, mem.avail_mb
                ));
                let mode_str = match self.mode {
                    ModeSel::Auto => "Auto",
                    ModeSel::Streaming => "Streaming",
                    ModeSel::InMemory => "InMemory",
                };
                let fmt_str = match self.fmt {
                    FormatSel::Csv => "CSV",
                    FormatSel::Xlsx => "XLSX",
                    FormatSel::Both => "Both",
                };
                let algo_str = match self.algo {
                    MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => "Algo1",
                    MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => "Algo2",
                    MatchingAlgorithm::Fuzzy => "Fuzzy",
                    MatchingAlgorithm::FuzzyNoMiddle => "FuzzyNoMiddle",
                    MatchingAlgorithm::HouseholdGpu => "Household5",
                    MatchingAlgorithm::HouseholdGpuOpt6 => "Household6",
                    MatchingAlgorithm::LevenshteinWeighted => "LevWeighted7",
                };
                out.push_str(&format!("Config: db1_host={} db1_db={} enable_dual={} pool_size={} ssd_storage={} mode={} algo={} fmt={}\n\n",
                    self.host, self.db, self.enable_dual, self.pool_size, self.ssd_storage, mode_str, algo_str, fmt_str));
                for (i, evt) in self.error_events.iter().enumerate() {
                    out.push_str(&format!("[{}] {} | Category: {:?}\nAction: {}\nDB1: {}.{}\nDB2: {}.{}\nTables: {:?} vs {:?}\nMem avail: {} MB | Pool: {}\nMessage: {}\n",
                        i+1, evt.ts_utc, evt.category, evt.source_action,
                        evt.db1_host, evt.db1_database,
                        evt.db2_host.clone().unwrap_or("-".into()), evt.db2_database.clone().unwrap_or("-".into()),
                        evt.table1, evt.table2, evt.mem_avail_mb, evt.pool_size_cfg, evt.message));
                    if let Some(ref ss) = evt.sqlstate {
                        out.push_str(&format!("SQLSTATE: {}\n", ss));
                    }
                    if let Some(ref op) = evt.operation {
                        out.push_str(&format!("Operation: {}\n", op));
                    }
                    if let Some(ref ch) = evt.chain {
                        out.push_str("Chain:\n");
                        out.push_str(ch);
                        out.push_str("\n");
                    }
                    if !evt.env_overrides.is_empty() {
                        out.push_str("Env overrides:\n");
                        for (k, v) in &evt.env_overrides {
                            out.push_str(&format!("  {}={}\n", k, v));
                        }
                    }
                    // Specific remediation hints from message
                    let mut specific: Vec<String> = Vec::new();
                    if evt.message.contains("Unknown column '") {
                        if let Some(s) = evt.message.find("Unknown column '") {
                            let rest = &evt.message[s + "Unknown column '".len()..];
                            if let Some(e) = rest.find('\'') {
                                let col = &rest[..e];
                                let tbl = evt
                                    .table1
                                    .clone()
                                    .or(evt.table2.clone())
                                    .unwrap_or("<table>".into());
                                specific
                                    .push(format!("ALTER TABLE {} ADD COLUMN {} <TYPE>", tbl, col));
                            }
                        }
                    }
                    if evt.message.to_ascii_lowercase().contains("doesn't exist")
                        && evt.message.to_ascii_lowercase().contains("table")
                    {
                        let tbl = evt
                            .table1
                            .clone()
                            .or(evt.table2.clone())
                            .unwrap_or("<table>".into());
                        specific.push(format!(
                            "CREATE TABLE {} (...) or select an existing table",
                            tbl
                        ));
                    }
                    if evt.message.contains("Incorrect date value")
                        || evt.message.contains("invalid date")
                    {
                        specific.push(
                            "Normalize date format to YYYY-MM-DD and ensure column type is DATE"
                                .into(),
                        );
                    }
                    if !specific.is_empty() {
                        out.push_str("Specific remediation:\n");
                        for s in &specific {
                            out.push_str(&format!("  - {}\n", s));
                        }
                    }
                    out.push_str("Remediation:\n");
                    for s in suggestions_for(evt.category) {
                        out.push_str(&format!("  - {}\n", s));
                    }
                    out.push_str("\n");
                    // Optional schema analysis
                    if self.schema_analysis_enabled {
                        let t1 = self.tables.get(self.table1_idx).cloned();
                        let t2 = if self.enable_dual {
                            self.tables2.get(self.table2_idx).cloned()
                        } else {
                            None
                        };
                        let host = self.host.clone();
                        let port = self.port.clone();
                        let user = self.user.clone();
                        let pass = self.pass.clone();
                        let dbname = self.db.clone();
                        let host2 = self.host2.clone();
                        let port2 = self.port2.clone();
                        let user2 = self.user2.clone();
                        let pass2 = self.pass2.clone();
                        let dbname2 = self.db2.clone();
                        let mut summary = String::new();
                        let mut index_suggestions: Vec<String> = Vec::new();
                        let mut grant_suggestions: Vec<String> = Vec::new();
                        let mut charset_notes: Vec<String> = Vec::new();
                        // derive hints from last error
                        if let Some(last) = self.error_events.last() {
                            let mm = last.message.to_ascii_lowercase()
                                + "\n"
                                + &last.chain.clone().unwrap_or_default().to_ascii_lowercase();
                            if mm.contains("command denied") || mm.contains("access denied") {
                                grant_suggestions.push(format!(
                                    "GRANT SELECT ON `{}`.* TO '{}'@'%';",
                                    self.db, self.user
                                ));
                            }
                            if mm.contains("incorrect string value") || mm.contains("collation") {
                                charset_notes.push("Consider aligning character set/collation: ALTER TABLE `<db>.<table>` CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;".into());
                            }
                            if mm.contains("foreign key constraint fails") {
                                charset_notes.push("Verify parent rows exist and consider indexing FK columns in child table to speed checks.".into());
                            }
                        }
                        let rt = gui_runtime();
                        let res: anyhow::Result<()> = rt.block_on(async {
                            let cfg1 = DatabaseConfig { host: host.clone(), port: port.parse().unwrap_or(3306), username: user.clone(), password: pass.clone(), database: dbname.clone() };
                            let pool1 = make_pool_with_size(&cfg1, Some(4)).await?;
                            if let Some(table) = t1.as_ref() {
                                summary.push_str(&format!("[DB1:{}] Table `{}`\n", dbname, table));
                                // Columns
                                let cols = sqlx::query(
                                    "SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, CHARACTER_MAXIMUM_LENGTH FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? ORDER BY ORDINAL_POSITION"
                                ).bind(&dbname).bind(table).fetch_all(&pool1).await?;
                                let mut actual: std::collections::HashMap<String,(String,bool,Option<i64>)> = std::collections::HashMap::new();
                                for row in cols {
                                    use sqlx::Row;
                                    let cname: String = row.get::<String,_>("COLUMN_NAME");
                                    let dtype: String = row.get::<String,_>("DATA_TYPE");
                                    let is_null: String = row.get::<String,_>("IS_NULLABLE");
                                    let clen: Option<i64> = row.try_get::<i64,_>("CHARACTER_MAXIMUM_LENGTH").ok();
                                    actual.insert(cname.to_lowercase(), (dtype.to_lowercase(), is_null == "YES", clen));
                                }
                                let expected = [
                                    ("id","bigint", false, None),
                                    ("uuid","varchar", false, Some(64)),
                                    ("first_name","varchar", false, Some(255)),
                                    ("middle_name","varchar", true, Some(255)),
                                    ("last_name","varchar", false, Some(255)),
                                    ("birthdate","date", false, None),
                                ];
                                for (name, ty, nullable, len) in expected {
                                    match actual.get(name) {
                                        None => summary.push_str(&format!("  - Missing column `{}` (expected {}{})\n", name, ty, len.map(|n| format!("({})", n)).unwrap_or_default())),
                                        Some((aty, anull, alen)) => {
                                            // type check (contains to allow varchar vs varchar)
                                            if !aty.contains(ty) { summary.push_str(&format!("  - Type mismatch `{}` actual {} vs expected {}\n", name, aty, ty)); }
                                            if *anull && !nullable { summary.push_str(&format!("  - Nullability mismatch `{}` is NULL but expected NOT NULL\n", name)); }
                                            if let (Some(exp), Some(act)) = (len, *alen) { if act < exp as i64 { summary.push_str(&format!("  - Length `{}` actual {} < expected {}\n", name, act, exp)); } }
                                        }
                                    }
                                }
                                // Index check on id
                                let idx = sqlx::query(
                                    "SELECT INDEX_NAME, COLUMN_NAME FROM INFORMATION_SCHEMA.STATISTICS WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?"
                                ).bind(&dbname).bind(table).fetch_all(&pool1).await?;
                                let has_id_idx = idx.iter().any(|r| { use sqlx::Row; r.get::<String,_>("COLUMN_NAME").to_lowercase()=="id" });
                                if !has_id_idx { index_suggestions.push(format!("ALTER TABLE `{}`.`{}` ADD INDEX idx_{}_id (id);", dbname, table, table)); }
                            }
                            if let (true, Some(table2)) = (self.enable_dual, t2.as_ref()) {
                                let cfg2 = DatabaseConfig { host: host2.clone(), port: port2.parse().unwrap_or(3306), username: user2.clone(), password: pass2.clone(), database: dbname2.clone() };
                                let pool2 = make_pool_with_size(&cfg2, Some(4)).await?;
                                summary.push_str(&format!("[DB2:{}] Table `{}`\n", dbname2, table2));
                                let cols2 = sqlx::query(
                                    "SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, CHARACTER_MAXIMUM_LENGTH FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? ORDER BY ORDINAL_POSITION"
                                ).bind(&dbname2).bind(table2).fetch_all(&pool2).await?;
                                let mut actual2: std::collections::HashMap<String,(String,bool,Option<i64>)> = std::collections::HashMap::new();
                                for row in cols2 {
                                    use sqlx::Row;
                                    let cname: String = row.get::<String,_>("COLUMN_NAME");
                                    let dtype: String = row.get::<String,_>("DATA_TYPE");
                                    let is_null: String = row.get::<String,_>("IS_NULLABLE");
                                    let clen: Option<i64> = row.try_get::<i64,_>("CHARACTER_MAXIMUM_LENGTH").ok();
                                    actual2.insert(cname.to_lowercase(), (dtype.to_lowercase(), is_null=="YES", clen));
                                }
                                for name in ["id","uuid","first_name","last_name","birthdate"] {
                                    if !actual2.contains_key(name) { summary.push_str(&format!("  - Missing column `{}`\n", name)); }
                                }
                                let idx2 = sqlx::query(
                                    "SELECT INDEX_NAME, COLUMN_NAME FROM INFORMATION_SCHEMA.STATISTICS WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?"
                                ).bind(&dbname2).bind(table2).fetch_all(&pool2).await?;
                                let has_id_idx2 = idx2.iter().any(|r| { use sqlx::Row; r.get::<String,_>("COLUMN_NAME").to_lowercase()=="id" });
                                if !has_id_idx2 { index_suggestions.push(format!("ALTER TABLE `{}`.`{}` ADD INDEX idx_{}_id (id);", dbname2, table2, table2)); }
                            }
                            anyhow::Ok(())
                        });
                        let _ = res; // ignore analysis errors silently in report generation
                        out.push_str("Schema Analysis (metadata-only):\n");
                        if summary.is_empty() {
                            out.push_str(
                                "  OK: No obvious schema issues detected for selected tables.\n",
                            );
                        } else {
                            out.push_str(&summary);
                        }
                        if !index_suggestions.is_empty() {
                            out.push_str("Index Suggestions:\n");
                            for s in &index_suggestions {
                                out.push_str(&format!("  {}\n", s));
                            }
                        }
                        if !grant_suggestions.is_empty() {
                            out.push_str("Privilege Suggestions:\n");
                            for s in &grant_suggestions {
                                out.push_str(&format!("  {}\n", s));
                            }
                        }
                        if !charset_notes.is_empty() {
                            out.push_str("Charset/Collation Notes:\n");
                            for s in &charset_notes {
                                out.push_str(&format!("  {}\n", s));
                            }
                        }
                    }
                    // Env (selected)
                    if !env_selected.is_empty() {
                        out.push_str("Env selected (sanitized):\n");
                        for (k, v) in &env_selected {
                            out.push_str(&format!("  {}={}\n", k, v));
                        }
                    }
                    // Log tail
                    if !self.log_buffer.is_empty() {
                        out.push_str("Log tail (most recent first):\n");
                        for line in self.log_buffer.iter().rev().take(200) {
                            out.push_str(&format!("  {}\n", line));
                        }
                    }
                }
                fs::write(&path, out)?;
                Ok(path)
            }
            ReportFormat::Json => {
                let escape = |s: &str| s.replace('"', "\\\"");
                let mut out = String::new();
                out.push_str("{\n");
                out.push_str(&format!(
                    "  \"app\": \"SRS-II Name Matching Application\",\n"
                ));
                out.push_str(&format!("  \"version\": \"{}\",\n", ver));
                out.push_str(&format!(
                    "  \"timestamp\": \"{}\",\n",
                    Utc::now().to_rfc3339()
                ));
                out.push_str(&format!("  \"system\": {{ \"os\": \"{}\", \"arch\": \"{}\", \"cores\": {}, \"mem_avail_mb\": {} }},\n", os, arch, cores, mem.avail_mb));
                out.push_str("  \"config\": {\n");
                out.push_str(&format!("    \"db1_host\": \"{}\",\n", escape(&self.host)));
                out.push_str(&format!(
                    "    \"db1_database\": \"{}\",\n",
                    escape(&self.db)
                ));
                out.push_str(&format!(
                    "    \"enable_dual\": {},\n",
                    if self.enable_dual { "true" } else { "false" }
                ));
                out.push_str(&format!(
                    "    \"pool_size_cfg\": \"{}\",\n",
                    escape(&self.pool_size)
                ));
                out.push_str(&format!(
                    "    \"ssd_storage\": {},\n",
                    if self.ssd_storage { "true" } else { "false" }
                ));
                let mode_str = match self.mode {
                    ModeSel::Auto => "Auto",
                    ModeSel::Streaming => "Streaming",
                    ModeSel::InMemory => "InMemory",
                };
                let fmt_str = match self.fmt {
                    FormatSel::Csv => "CSV",
                    FormatSel::Xlsx => "XLSX",
                    FormatSel::Both => "Both",
                };
                let algo_str = match self.algo {
                    MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => "Algo1",
                    MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => "Algo2",
                    MatchingAlgorithm::Fuzzy => "Fuzzy",
                    MatchingAlgorithm::FuzzyNoMiddle => "FuzzyNoMiddle",
                    MatchingAlgorithm::HouseholdGpu => "Household5",
                    MatchingAlgorithm::HouseholdGpuOpt6 => "Household6",
                    MatchingAlgorithm::LevenshteinWeighted => "LevWeighted7",
                };
                out.push_str(&format!("    \"mode\": \"{}\",\n", mode_str));
                out.push_str(&format!("    \"algo\": \"{}\",\n", algo_str));
                out.push_str(&format!("    \"fmt\": \"{}\"\n", fmt_str));
                out.push_str("  },\n");
                // env_selected
                out.push_str("  \"env_selected\": [\n");
                for (i, (k, v)) in env_selected.iter().enumerate() {
                    if i > 0 {
                        out.push_str(",\n");
                    }
                    out.push_str(&format!(
                        "    {{ \"key\": \"{}\", \"value\": \"{}\" }}",
                        escape(k),
                        escape(v)
                    ));
                }
                out.push_str("\n  ],\n");
                // log_tail
                out.push_str("  \"log_tail\": [\n");
                for (i, line) in self.log_buffer.iter().rev().take(200).enumerate() {
                    if i > 0 {
                        out.push_str(",\n");
                    }
                    out.push_str(&format!("    \"{}\"", escape(line)));
                }
                out.push_str("\n  ],\n");
                // optional schema analysis
                if self.schema_analysis_enabled {
                    let t1 = self.tables.get(self.table1_idx).cloned();
                    let t2 = if self.enable_dual {
                        self.tables2.get(self.table2_idx).cloned()
                    } else {
                        None
                    };
                    let host = self.host.clone();
                    let port = self.port.clone();
                    let user = self.user.clone();
                    let pass = self.pass.clone();
                    let dbname = self.db.clone();
                    let host2 = self.host2.clone();
                    let port2 = self.port2.clone();
                    let user2 = self.user2.clone();
                    let pass2 = self.pass2.clone();
                    let dbname2 = self.db2.clone();
                    let mut summary = String::new();
                    let mut index_suggestions: Vec<String> = Vec::new();
                    let mut grant_suggestions: Vec<String> = Vec::new();
                    let mut charset_notes: Vec<String> = Vec::new();
                    if let Some(last) = self.error_events.last() {
                        let mm = last.message.to_ascii_lowercase()
                            + "\n"
                            + &last.chain.clone().unwrap_or_default().to_ascii_lowercase();
                        if mm.contains("command denied") || mm.contains("access denied") {
                            grant_suggestions.push(format!(
                                "GRANT SELECT ON `{}`.* TO '{}'@'%';",
                                self.db, self.user
                            ));
                        }
                        if mm.contains("incorrect string value") || mm.contains("collation") {
                            charset_notes.push("Consider aligning character set/collation: ALTER TABLE `<db>.<table>` CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;".into());
                        }
                        if mm.contains("foreign key constraint fails") {
                            charset_notes.push("Verify parent rows exist and consider indexing FK columns in child table to speed checks.".into());
                        }
                    }
                    let rt = gui_runtime();
                    let _ = rt.block_on(async {
                        let cfg1 = DatabaseConfig { host: host.clone(), port: port.parse().unwrap_or(3306), username: user.clone(), password: pass.clone(), database: dbname.clone() };
                        let pool1 = make_pool_with_size(&cfg1, Some(4)).await?;
                        if let Some(table) = t1.as_ref() {
                            summary.push_str(&format!("[DB1:{}] Table `{}`\n", dbname, table));
                            let cols = sqlx::query("SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, CHARACTER_MAXIMUM_LENGTH FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? ORDER BY ORDINAL_POSITION").bind(&dbname).bind(table).fetch_all(&pool1).await?;
                            let mut actual: std::collections::HashMap<String,(String,bool,Option<i64>)> = std::collections::HashMap::new();
                            for row in cols { use sqlx::Row; let cname: String = row.get("COLUMN_NAME"); let dtype: String = row.get("DATA_TYPE"); let is_null: String = row.get("IS_NULLABLE"); let clen: Option<i64> = row.try_get("CHARACTER_MAXIMUM_LENGTH").ok(); actual.insert(cname.to_lowercase(), (dtype.to_lowercase(), is_null=="YES", clen)); }
                            let expected = [("id","bigint", false, None),("uuid","varchar", false, Some(64)),("first_name","varchar", false, Some(255)),("middle_name","varchar", true, Some(255)),("last_name","varchar", false, Some(255)),("birthdate","date", false, None)];
                            for (name, ty, nullable, len) in expected { match actual.get(name) { None => summary.push_str(&format!("  - Missing column `{}` (expected {}{})\n", name, ty, len.map(|n| format!("({})", n)).unwrap_or_default())), Some((aty, anull, alen)) => { if !aty.contains(ty) { summary.push_str(&format!("  - Type mismatch `{}` actual {} vs expected {}\n", name, aty, ty)); } if *anull && !nullable { summary.push_str(&format!("  - Nullability mismatch `{}` is NULL but expected NOT NULL\n", name)); } if let (Some(exp), Some(act)) = (len, *alen) { if act < exp as i64 { summary.push_str(&format!("  - Length `{}` actual {} < expected {}\n", name, act, exp)); } } } } }
                            let idx = sqlx::query("SELECT INDEX_NAME, COLUMN_NAME FROM INFORMATION_SCHEMA.STATISTICS WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?").bind(&dbname).bind(table).fetch_all(&pool1).await?; let has_id_idx = idx.iter().any(|r| { use sqlx::Row; r.get::<String,_>("COLUMN_NAME").to_lowercase()=="id" }); if !has_id_idx { index_suggestions.push(format!("ALTER TABLE `{}`.`{}` ADD INDEX idx_{}_id (id);", dbname, table, table)); }
                        }
                        if let (true, Some(table2)) = (self.enable_dual, t2.as_ref()) {
                            let cfg2 = DatabaseConfig { host: host2.clone(), port: port2.parse().unwrap_or(3306), username: user2.clone(), password: pass2.clone(), database: dbname2.clone() };
                            let pool2 = make_pool_with_size(&cfg2, Some(4)).await?;
                            summary.push_str(&format!("[DB2:{}] Table `{}`\n", dbname2, table2));
                            let cols2 = sqlx::query("SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, CHARACTER_MAXIMUM_LENGTH FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? ORDER BY ORDINAL_POSITION").bind(&dbname2).bind(table2).fetch_all(&pool2).await?;
                            let mut actual2: std::collections::HashMap<String,(String,bool,Option<i64>)> = std::collections::HashMap::new();
                            for row in cols2 { use sqlx::Row; let cname: String = row.get("COLUMN_NAME"); let dtype: String = row.get("DATA_TYPE"); let is_null: String = row.get("IS_NULLABLE"); let clen: Option<i64> = row.try_get("CHARACTER_MAXIMUM_LENGTH").ok(); actual2.insert(cname.to_lowercase(), (dtype.to_lowercase(), is_null=="YES", clen)); }
                            for name in ["id","uuid","first_name","last_name","birthdate"] { if !actual2.contains_key(name) { summary.push_str(&format!("  - Missing column `{}`\n", name)); } }
                            let idx2 = sqlx::query("SELECT INDEX_NAME, COLUMN_NAME FROM INFORMATION_SCHEMA.STATISTICS WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?").bind(&dbname2).bind(table2).fetch_all(&pool2).await?; let has_id_idx2 = idx2.iter().any(|r| { use sqlx::Row; r.get::<String,_>("COLUMN_NAME").to_lowercase()=="id" }); if !has_id_idx2 { index_suggestions.push(format!("ALTER TABLE `{}`.`{}` ADD INDEX idx_{}_id (id);", dbname2, table2, table2)); }
                        }
                        anyhow::Ok(())
                    });
                    out.push_str("  \"schema_analysis\": {\n");
                    out.push_str(&format!("    \"summary\": \"{}\",\n", escape(&summary)));
                    out.push_str("    \"index_suggestions\": [\n");
                    for (i, s) in index_suggestions.iter().enumerate() {
                        if i > 0 {
                            out.push_str(",\n");
                        }
                        out.push_str(&format!("      \"{}\"", escape(s)));
                    }
                    out.push_str("\n    ],\n");
                    out.push_str("    \"grant_suggestions\": [\n");
                    for (i, s) in grant_suggestions.iter().enumerate() {
                        if i > 0 {
                            out.push_str(",\n");
                        }
                        out.push_str(&format!("      \"{}\"", escape(s)));
                    }
                    out.push_str("\n    ],\n");
                    out.push_str("    \"charset_notes\": [\n");
                    for (i, s) in charset_notes.iter().enumerate() {
                        if i > 0 {
                            out.push_str(",\n");
                        }
                        out.push_str(&format!("      \"{}\"", escape(s)));
                    }
                    out.push_str("\n    ]\n");
                    out.push_str("  },\n");
                }
                out.push_str("  \"events\": [\n");
                for (i, e) in self.error_events.iter().enumerate() {
                    if i > 0 {
                        out.push_str(",\n");
                    }
                    out.push_str("    {\n");
                    out.push_str(&format!("      \"ts_utc\": \"{}\",\n", escape(&e.ts_utc)));
                    out.push_str(&format!("      \"category\": \"{:?}\",\n", e.category));
                    out.push_str(&format!("      \"message\": \"{}\",\n", escape(&e.message)));
                    out.push_str(&format!(
                        "      \"sqlstate\": \"{}\",\n",
                        escape(&e.sqlstate.clone().unwrap_or_default())
                    ));
                    out.push_str(&format!(
                        "      \"operation\": \"{}\",\n",
                        escape(&e.operation.clone().unwrap_or_default())
                    ));
                    out.push_str(&format!(
                        "      \"chain\": \"{}\",\n",
                        escape(&e.chain.clone().unwrap_or_default())
                    ));
                    out.push_str(&format!(
                        "      \"source_action\": \"{}\",\n",
                        escape(&e.source_action)
                    ));
                    out.push_str(&format!(
                        "      \"db1_host\": \"{}\",\n",
                        escape(&e.db1_host)
                    ));
                    out.push_str(&format!(
                        "      \"db1_database\": \"{}\",\n",
                        escape(&e.db1_database)
                    ));
                    out.push_str(&format!(
                        "      \"db2_host\": \"{}\",\n",
                        escape(&e.db2_host.clone().unwrap_or_default())
                    ));

                    out.push_str(&format!(
                        "      \"db2_database\": \"{}\",\n",
                        escape(&e.db2_database.clone().unwrap_or_default())
                    ));
                    out.push_str(&format!(
                        "      \"table1\": \"{}\",\n",
                        escape(&e.table1.clone().unwrap_or_default())
                    ));
                    out.push_str(&format!(
                        "      \"table2\": \"{}\",\n",
                        escape(&e.table2.clone().unwrap_or_default())
                    ));
                    out.push_str(&format!("      \"mem_avail_mb\": {},\n", e.mem_avail_mb));
                    out.push_str(&format!("      \"pool_size_cfg\": {}\n", e.pool_size_cfg));
                    out.push_str("    }");
                }
                out.push_str("\n  ],\n");
                out.push_str("  \"suggestions_by_event\": [\n");
                for (i, e) in self.error_events.iter().enumerate() {
                    if i > 0 {
                        out.push_str(",\n");
                    }
                    out.push_str("    [\n");
                    let sugg = suggestions_for(e.category);
                    for (j, s) in sugg.iter().enumerate() {
                        if j > 0 {
                            out.push_str(",\n");
                        }
                        out.push_str(&format!("      \"{}\"", escape(s)));
                    }
                    out.push_str("\n    ]");
                }
                out.push_str("\n  ]\n");
                out.push_str("}\n");
                fs::write(&path, out)?;
                Ok(path)
            }
        }
    }

    fn record_error_with_details(
        &mut self,
        message: String,
        sqlstate: Option<String>,
        chain: Option<String>,
        operation: Option<String>,
    ) {
        let sanitize = |s: &str| -> String {
            if let Some(pos) = s.find("mysql://") {
                if let Some(at) = s[pos..].find('@') {
                    let mut out = s.to_string();
                    out.replace_range(pos..pos + at, "mysql://[REDACTED]");
                    return out;
                }
            }
            s.to_string()
        };
        let mstats = name_matcher::metrics::memory_stats_mb();
        let pool_sz = self.pool_size.parse::<u32>().unwrap_or(0);
        let envs = [
            "NAME_MATCHER_POOL_SIZE",
            "NAME_MATCHER_POOL_MIN",
            "NAME_MATCHER_STREAMING",
            "NAME_MATCHER_PARTITION",
            "NAME_MATCHER_ALLOW_BIRTHDATE_SWAP",
        ];
        let mut env_overrides = Vec::new();
        for k in envs {
            if let Ok(v) = std::env::var(k) {
                env_overrides.push((k.to_string(), v));
            }
        }
        let cat = categorize_error_with(sqlstate.as_deref(), &message);

        let evt = DiagEvent {
            ts_utc: Utc::now().to_rfc3339(),
            category: cat,
            message: sanitize(&message),
            sqlstate,
            chain,
            operation,
            source_action: self.last_action.clone(),
            db1_host: self.host.clone(),
            db1_database: self.db.clone(),
            db2_host: if self.enable_dual {
                Some(self.host2.clone())
            } else {
                None
            },
            db2_database: if self.enable_dual {
                Some(self.db2.clone())
            } else {
                None
            },
            table1: self.tables.get(self.table1_idx).cloned(),
            table2: if self.enable_dual {
                self.tables2.get(self.table2_idx).cloned()
            } else {
                self.tables.get(self.table2_idx).cloned()
            },
            mem_avail_mb: mstats.avail_mb,
            pool_size_cfg: pool_sz,
            env_overrides,
        };
        // Automatic parameter reduction on memory-related resource constraints
        if matches!(cat, ErrorCategory::ResourceConstraint)
            && message.to_ascii_lowercase().contains("memory")
        {
            if let Ok(cur_batch) = self.batch_size.parse::<i64>() {
                let new_batch = (cur_batch / 2).max(10_000);
                self.batch_size = new_batch.to_string();
            }
            if let Ok(cur_thr) = self.mem_thresh.parse::<u64>() {
                let new_thr = (((cur_thr as f64) * 1.25) as u64).max(1024);
                self.mem_thresh = new_thr.to_string();
            }
            if let Ok(gm) = self.gpu_mem_mb.parse::<u64>() {
                let gm_new = ((gm as f64) * 0.80).max(256.0) as u64;
                self.gpu_mem_mb = gm_new.to_string();
            }
            if let Ok(pp) = self.gpu_probe_mem_mb.parse::<u64>() {
                let pp_new = ((pp as f64) * 0.75).max(256.0) as u64;
                self.gpu_probe_mem_mb = pp_new.to_string();
            }
            self.status = "Resource constraint detected; auto-reduced batch and adjusted thresholds. Please retry.".into();
        }

        self.error_events.push(evt);
    }

    /// Render the Configuration Summary Panel (Phase 3 UI Enhancement)
    fn render_config_summary(&self, ui: &mut egui::Ui) {
        ui.add_space(4.0);

        // Mode line with visual indicator
        let (mode_icon, mode_text, mode_color) = if self.advanced_enabled {
            if self.cascade_enabled {
                (
                    "🔄",
                    "CASCADE MODE (L1→L11)",
                    egui::Color32::from_rgb(255, 193, 7),
                ) // Amber
            } else {
                match self.adv_level {
                    Some(level) => {
                        let level_name = Self::level_short_name(level);
                        (
                            "🎯",
                            format!("Advanced: {}", level_name).leak() as &str,
                            egui::Color32::from_rgb(175, 82, 222),
                        ) // Purple
                    }
                    None => (
                        "⚠",
                        "Advanced (no level selected)",
                        egui::Color32::from_rgb(255, 69, 58),
                    ), // Red
                }
            }
        } else {
            (
                "📊",
                Self::algorithm_label(self.algo),
                egui::Color32::from_rgb(0, 174, 239),
            ) // Cyan
        };

        ui.horizontal(|ui| {
            ui.colored_label(mode_color, format!("{} {}", mode_icon, mode_text));
        });

        // GPU Status
        ui.horizontal(|ui| {
            ui.label("GPU:");
            if self.use_gpu_hash_join || self.use_gpu || self.use_gpu_fuzzy_direct_hash {
                let mut features = Vec::new();
                if self.use_gpu_hash_join {
                    features.push("Hash Join");
                }
                if self.use_gpu {
                    features.push("Fuzzy Metrics");
                }
                if self.use_gpu_fuzzy_direct_hash {
                    features.push("Pre-pass");
                }
                if self.use_gpu_levenshtein_full_scoring {
                    features.push("Full Scoring");
                }
                ui.colored_label(
                    egui::Color32::from_rgb(76, 217, 100),
                    format!("✓ {}", features.join(" + ")),
                );
            } else {
                ui.colored_label(egui::Color32::from_rgb(152, 152, 157), "○ CPU only");
            }
        });

        // Execution Mode
        ui.horizontal(|ui| {
            ui.label("Execution:");
            let mode_str = match self.mode {
                ModeSel::Auto => "Auto",
                ModeSel::Streaming => "Streaming",
                ModeSel::InMemory => "In-Memory",
            };
            if let Some(ref override_reason) = self.effective_mode_override {
                ui.label(format!("{} →", mode_str));
                ui.colored_label(
                    egui::Color32::from_rgb(255, 193, 7),
                    format!("In-Memory ({})", override_reason),
                );
            } else {
                ui.label(mode_str);
            }
        });

        // Output
        ui.horizontal(|ui| {
            ui.label("Output:");
            let fmt_str = match self.fmt {
                FormatSel::Csv => "CSV",
                FormatSel::Xlsx => "XLSX",
                FormatSel::Both => "Both",
            };
            ui.label(format!("{} ({})", self.path, fmt_str));
        });

        // Threshold (if applicable)
        let needs_threshold = if self.advanced_enabled {
            self.cascade_enabled
                || matches!(
                    self.adv_level,
                    Some(AdvLevel::L10FuzzyBirthdateFullMiddle)
                        | Some(AdvLevel::L11FuzzyBirthdateNoMiddle)
                        | Some(AdvLevel::L12HouseholdMatching)
                )
        } else {
            matches!(
                self.algo,
                MatchingAlgorithm::Fuzzy
                    | MatchingAlgorithm::FuzzyNoMiddle
                    | MatchingAlgorithm::HouseholdGpu
                    | MatchingAlgorithm::HouseholdGpuOpt6
                    | MatchingAlgorithm::LevenshteinWeighted
            )
        };
        if needs_threshold {
            ui.horizontal(|ui| {
                ui.label("Threshold:");
                let thr = if self.advanced_enabled {
                    format!("{:.0}%", self.adv_threshold * 100.0)
                } else {
                    format!("{}%", self.fuzzy_threshold_pct)
                };
                ui.label(thr);
            });
        }

        ui.add_space(4.0);
        ui.separator();

        // Validation warnings
        let warnings = self.get_validation_warnings();
        if warnings.is_empty() {
            ui.horizontal(|ui| {
                ui.colored_label(egui::Color32::from_rgb(76, 217, 100), "✓ Ready to run");
            });
        } else {
            ui.colored_label(
                egui::Color32::from_rgb(255, 193, 7),
                format!("⚠ {} issue(s):", warnings.len()),
            );
            for warning in &warnings {
                ui.horizontal(|ui| {
                    ui.add_space(12.0);
                    ui.colored_label(
                        egui::Color32::from_rgb(255, 193, 7),
                        format!("• {}", warning),
                    );
                });
            }
        }
    }

    /// Get a short name for an advanced level
    fn level_short_name(level: AdvLevel) -> &'static str {
        match level {
            AdvLevel::L1BirthdateFullMiddle => "L1: Birthdate + Full Middle",
            AdvLevel::L2BirthdateMiddleInitial => "L2: Birthdate + Middle Initial",
            AdvLevel::L3BirthdateNoMiddle => "L3: Birthdate (No Middle)",
            AdvLevel::L4BarangayFullMiddle => "L4: Barangay + Full Middle",
            AdvLevel::L5BarangayMiddleInitial => "L5: Barangay + Middle Initial",
            AdvLevel::L6BarangayNoMiddle => "L6: Barangay (No Middle)",
            AdvLevel::L7CityFullMiddle => "L7: City + Full Middle",
            AdvLevel::L8CityMiddleInitial => "L8: City + Middle Initial",
            AdvLevel::L9CityNoMiddle => "L9: City (No Middle)",
            AdvLevel::L10FuzzyBirthdateFullMiddle => "L10: Fuzzy + Birthdate",
            AdvLevel::L11FuzzyBirthdateNoMiddle => "L11: Fuzzy (No Middle)",
            AdvLevel::L12HouseholdMatching => "L12: Household",
        }
    }

    /// Get validation warnings for current configuration
    fn get_validation_warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        // Advanced mode without level selected
        if self.advanced_enabled && !self.cascade_enabled && self.adv_level.is_none() {
            warnings.push("Select an Advanced level (L1-L12) or enable Cascade mode".into());
        }

        // GPU Full Scoring without Pre-pass (should be auto-corrected, but check)
        if self.use_gpu_levenshtein_full_scoring && !self.use_gpu_fuzzy_direct_hash {
            warnings.push("GPU Full Scoring requires GPU Pre-pass".into());
        }

        // Mode override warning
        if self.effective_mode_override.is_some() {
            warnings.push("Streaming mode will be overridden to In-Memory".into());
        }

        // Empty tables
        if self.tables.is_empty() {
            warnings.push("Load tables using 'Load Tables' button".into());
        }

        // Empty output path
        if self.path.trim().is_empty() {
            warnings.push("Specify an output file path".into());
        }

        warnings
    }
}

impl App for GuiApp {
    fn update(&mut self, ctx: &Context, _frame: &mut Frame) {
        self.poll_messages();
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::both()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    self.ui_top(ui);
                });
        });

        // CUDA Diagnostics Window
        egui::Window::new("CUDA Diagnostics")
            .open(&mut self.cuda_diag_open)
            .resizable(true)
            .show(ctx, |ui| {
                ui.label("Comprehensive CUDA system information:");
                ui.add(
                    egui::TextEdit::multiline(&mut self.cuda_diag_text)
                        .desired_width(600.0)
                        .desired_rows(20),
                );
            });
    }
}

/// Apply modern dark theme with color accents
fn apply_modern_dark_theme(ctx: &egui::Context) {
    use egui::{Color32, CornerRadius, Stroke, Visuals, style::WidgetVisuals, style::Widgets};

    let mut visuals = Visuals::dark();

    // Background colors - deep dark with subtle variations
    visuals.window_fill = Color32::from_rgb(20, 22, 26); // Main window background
    visuals.panel_fill = Color32::from_rgb(25, 27, 31); // Panel background
    visuals.faint_bg_color = Color32::from_rgb(30, 32, 36); // Subtle background
    visuals.extreme_bg_color = Color32::from_rgb(15, 17, 21); // Extreme contrast background

    // Primary accent color - Cyan/Blue for technology/performance
    let primary_accent = Color32::from_rgb(0, 174, 239); // Bright cyan-blue
    let primary_accent_hover = Color32::from_rgb(41, 196, 255); // Lighter cyan on hover
    let primary_accent_dim = Color32::from_rgb(0, 139, 191); // Dimmer cyan

    // Secondary accent - Green for success/active states
    let success_color = Color32::from_rgb(76, 217, 100); // Bright green
    let success_dim = Color32::from_rgb(52, 199, 89); // Dimmer green

    // Tertiary accent - Purple for GPU/advanced features
    let gpu_accent = Color32::from_rgb(175, 82, 222); // Vibrant purple
    let gpu_accent_hover = Color32::from_rgb(191, 108, 230); // Lighter purple

    // Warning and error colors
    let warning_color = Color32::from_rgb(255, 204, 0); // Amber
    let error_color = Color32::from_rgb(255, 69, 58); // Red

    // Text colors with proper contrast
    let text_color = Color32::from_rgb(235, 235, 245); // Primary text (high contrast)
    visuals.override_text_color = Some(text_color);
    visuals.weak_text_color = Some(Color32::from_rgb(152, 152, 157)); // Secondary text
    visuals.hyperlink_color = primary_accent; // Links use primary accent

    // Widget styling
    visuals.widgets = Widgets {
        noninteractive: WidgetVisuals {
            bg_fill: Color32::from_rgb(35, 37, 41),
            weak_bg_fill: Color32::from_rgb(30, 32, 36),
            bg_stroke: Stroke::new(1.0, Color32::from_rgb(50, 52, 56)),
            corner_radius: CornerRadius::same(4),
            fg_stroke: Stroke::new(1.0, text_color),
            expansion: 0.0,
        },
        inactive: WidgetVisuals {
            bg_fill: Color32::from_rgb(40, 42, 46),
            weak_bg_fill: Color32::from_rgb(35, 37, 41),
            bg_stroke: Stroke::new(1.0, Color32::from_rgb(60, 62, 66)),
            corner_radius: CornerRadius::same(4),
            fg_stroke: Stroke::new(1.0, text_color),
            expansion: 0.0,
        },
        hovered: WidgetVisuals {
            bg_fill: Color32::from_rgb(50, 52, 56),
            weak_bg_fill: Color32::from_rgb(45, 47, 51),
            bg_stroke: Stroke::new(1.0, primary_accent_dim),
            corner_radius: CornerRadius::same(4),
            fg_stroke: Stroke::new(1.5, primary_accent_hover),
            expansion: 1.0,
        },
        active: WidgetVisuals {
            bg_fill: primary_accent_dim,
            weak_bg_fill: Color32::from_rgb(0, 139, 191),
            bg_stroke: Stroke::new(1.0, primary_accent),
            corner_radius: CornerRadius::same(4),
            fg_stroke: Stroke::new(2.0, Color32::WHITE),
            expansion: 1.0,
        },
        open: WidgetVisuals {
            bg_fill: Color32::from_rgb(45, 47, 51),
            weak_bg_fill: Color32::from_rgb(40, 42, 46),
            bg_stroke: Stroke::new(1.0, primary_accent),
            corner_radius: CornerRadius::same(4),
            fg_stroke: Stroke::new(1.0, text_color),
            expansion: 0.0,
        },
    };

    // Selection colors
    visuals.selection.bg_fill = primary_accent.linear_multiply(0.3);
    visuals.selection.stroke = Stroke::new(1.0, primary_accent);

    // Window styling
    visuals.window_corner_radius = CornerRadius::same(8);
    visuals.window_shadow.color = Color32::from_black_alpha(80);
    visuals.window_stroke = Stroke::new(1.0, Color32::from_rgb(50, 52, 56));

    // Popup styling
    visuals.popup_shadow.color = Color32::from_black_alpha(100);

    ctx.set_visuals(visuals);
}

impl Drop for GuiApp {
    fn drop(&mut self) {
        #[cfg(feature = "gpu")]
        {
            // Ensure background GPU dynamic tuner thread is stopped on GUI shutdown
            name_matcher::matching::dyn_tuner_stop();
        }
    }
}

fn main() -> eframe::Result<()> {
    // Initialize GUI log forwarder so GPU/engine logs appear in the GUI console
    init_gui_logger_once();
    // Eager GPU context pre-warm in the background (non-blocking)
    #[cfg(feature = "gpu")]
    std::thread::spawn(|| {
        name_matcher::matching::prewarm_gpu_contexts();
    });

    let opts = NativeOptions::default();
    eframe::run_native(
        "SRS-II Name Matching Application",
        opts,
        Box::new(|cc| {
            // Apply modern dark theme
            apply_modern_dark_theme(&cc.egui_ctx);
            Ok::<Box<dyn App>, Box<(dyn std::error::Error + Send + Sync + 'static)>>(Box::new(
                GuiApp::default(),
            ))
        }),
    )
}

#[cfg(windows)]
fn spawn_console_tail(path: &std::path::Path) -> std::io::Result<std::process::Child> {
    use std::os::windows::process::CommandExt;
    const CREATE_NEW_CONSOLE: u32 = 0x00000010;
    std::process::Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            &format!("Get-Content -Path '{}' -Wait -Tail 0", path.display()),
        ])
        .creation_flags(CREATE_NEW_CONSOLE)
        .spawn()
}

#[cfg(not(windows))]
fn spawn_console_tail(_path: &std::path::Path) -> std::io::Result<std::process::Child> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Other,
        "console tail only supported on Windows",
    ))
}
