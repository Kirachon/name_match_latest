//! Frozen DTOs shared between the CLI, the legacy egui binary, and the Tauri
//! shell.
//!
//! These types are the **single source of truth** for the cross-language
//! contract documented in `docs/tauri-migration-plan.md` (T3.5 — Backend
//! Contract Freeze). Any change here must be mirrored in
//! `ui/src/shared/tauri/types.ts` and the matching Zod schema.
//!
//! Design rules:
//! * All variants are `#[serde(rename_all = "kebab-case")]` so the wire form is
//!   deterministic.
//! * Numeric fields are typed precisely (`u32`/`u64`/`usize`) and use the
//!   serde defaults so missing fields fall back to safe defaults instead of
//!   rejecting the payload.
//! * No raw `Person` rows are exposed across the IPC boundary — only
//!   `MatchPairDto` summary rows, which the result store paginates.
//! * Secrets are never serialised in events — `DbCredentialsDto::password`
//!   exists only when calling `connect_db`; the registry stores connection
//!   metadata without the password after the pool is built.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Algorithm choices exposed to the UI. The numeric "Option N" naming used in
/// the legacy egui GUI is preserved in the variant docs so operators can
/// cross-reference release notes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AlgorithmDto {
    /// Option 1 — deterministic match on first + last + birthdate.
    DeterministicFnLnBd,
    /// Option 2 — deterministic match on first + middle + last + birthdate.
    DeterministicFnMnLnBd,
    /// Option 3 — fuzzy match (with middle name).
    Fuzzy,
    /// Option 4 — fuzzy match (without middle name).
    FuzzyNoMiddle,
    /// Option 5 — household matching (uuid → hh_id).
    HouseholdGpu,
    /// Option 6 — role-swapped household matching (hh_id → uuid).
    HouseholdGpuOpt6,
    /// Option 7 — Levenshtein-weighted (SQL equivalent).
    LevenshteinWeighted,
}

impl AlgorithmDto {
    /// Numeric option label (1..=7) used in summaries and exports.
    pub fn option_number(self) -> u8 {
        match self {
            AlgorithmDto::DeterministicFnLnBd => 1,
            AlgorithmDto::DeterministicFnMnLnBd => 2,
            AlgorithmDto::Fuzzy => 3,
            AlgorithmDto::FuzzyNoMiddle => 4,
            AlgorithmDto::HouseholdGpu => 5,
            AlgorithmDto::HouseholdGpuOpt6 => 6,
            AlgorithmDto::LevenshteinWeighted => 7,
        }
    }

    /// Map the DTO to the legacy enum used inside `name_matcher::matching`.
    pub fn to_engine(self) -> crate::matching::MatchingAlgorithm {
        use crate::matching::MatchingAlgorithm as M;
        match self {
            AlgorithmDto::DeterministicFnLnBd => M::IdUuidYasIsMatchedInfnbd,
            AlgorithmDto::DeterministicFnMnLnBd => M::IdUuidYasIsMatchedInfnmnbd,
            AlgorithmDto::Fuzzy => M::Fuzzy,
            AlgorithmDto::FuzzyNoMiddle => M::FuzzyNoMiddle,
            AlgorithmDto::HouseholdGpu => M::HouseholdGpu,
            AlgorithmDto::HouseholdGpuOpt6 => M::HouseholdGpuOpt6,
            AlgorithmDto::LevenshteinWeighted => M::LevenshteinWeighted,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum ComputeModeDto {
    /// Force CPU even if GPU resources are present.
    #[default]
    Cpu,
    /// Auto-select: prefer GPU when available, fall back to CPU on OOM.
    Auto,
    /// Force GPU; fail fast if CUDA is unavailable.
    ForceGpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum RunModeDto {
    /// Auto-pick streaming vs in-memory based on row counts.
    #[default]
    Auto,
    /// Force streaming (large datasets, low RAM).
    Streaming,
    /// Force in-memory (faster for small datasets).
    InMemory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum ExportFormatDto {
    #[default]
    Csv,
    Xlsx,
    Both,
}

/// Database credentials accepted by `connect_db`. Passwords are not stored in
/// the registry once the pool is built; subsequent commands operate on the
/// session id only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbCredentialsDto {
    pub host: String,
    pub port: u16,
    pub username: String,
    /// Optional. Empty string is treated as "no password".
    #[serde(default)]
    pub password: String,
    pub database: String,
}

/// A connected database session, returned to the frontend without secrets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbSessionDto {
    /// Opaque server-assigned session id.
    pub session_id: String,
    pub host: String,
    pub port: u16,
    pub username: String,
    pub database: String,
    /// Latency of the test_connection ping (in ms).
    #[serde(default)]
    pub latency_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableInfoDto {
    pub name: String,
    pub schema: String,
    /// `None` until `get_row_count` is called for the table.
    #[serde(default)]
    pub row_count: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TableColumnsDto {
    pub has_id: bool,
    pub has_uuid: bool,
    pub has_first_name: bool,
    pub has_middle_name: bool,
    pub has_last_name: bool,
    pub has_birthdate: bool,
    pub has_hh_id: bool,
    /// All raw column names returned by INFORMATION_SCHEMA, in declaration
    /// order. The frontend uses this for cascade/advanced helper hints.
    #[serde(default)]
    pub raw_columns: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableSelectionDto {
    pub session_id: String,
    pub table: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuOptionsDto {
    pub mode: ComputeModeDto,
    /// Use GPU hash-join for deterministic algorithms 1/2.
    #[serde(default)]
    pub use_hash_join: bool,
    /// GPU fuzzy direct prefilter (Option 3 / 4 GPU acceleration).
    #[serde(default)]
    pub use_direct_prefilter: bool,
    /// GPU Levenshtein scoring (Option 7).
    #[serde(default)]
    pub use_levenshtein_full_scoring: bool,
    /// Memory budget in MB for GPU tiling. `None` = auto.
    #[serde(default)]
    pub vram_budget_mb: Option<u32>,
    /// Enable dynamic GPU tuning (auto-adjust batch sizes).
    #[serde(default)]
    pub dynamic_tuning: bool,
}

impl Default for GpuOptionsDto {
    fn default() -> Self {
        Self {
            mode: ComputeModeDto::Cpu,
            use_hash_join: false,
            use_direct_prefilter: false,
            use_levenshtein_full_scoring: false,
            vram_budget_mb: None,
            dynamic_tuning: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingOptionsDto {
    pub mode: RunModeDto,
    /// Streaming batch size (rows). Range 1_000..=200_000.
    pub batch_size: u32,
    /// Optional partition strategy: `last_initial` | `birthyear5`.
    #[serde(default)]
    pub partition_strategy: Option<String>,
    /// Resume from existing checkpoint file if present.
    #[serde(default)]
    pub resume: bool,
    /// Optional checkpoint path. `None` means default `<output>.nmckpt`.
    #[serde(default)]
    pub checkpoint_path: Option<String>,
}

impl Default for StreamingOptionsDto {
    fn default() -> Self {
        Self {
            mode: RunModeDto::Auto,
            batch_size: 10_000,
            partition_strategy: None,
            resume: false,
            checkpoint_path: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportOptionsDto {
    pub format: ExportFormatDto,
    pub output_directory: String,
    /// File stem; the engine appends the format extension and timestamp.
    pub file_stem: String,
    /// Minimum confidence (0..=100) required to include a row in the export.
    #[serde(default)]
    pub min_confidence: Option<f32>,
}

impl Default for ExportOptionsDto {
    fn default() -> Self {
        Self {
            format: ExportFormatDto::Csv,
            output_directory: ".".into(),
            file_stem: "matches".into(),
            min_confidence: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchOptionsDto {
    /// Allow month/day swap on birthdates (L10/L11 parity).
    #[serde(default)]
    pub allow_birthdate_swap: bool,
    /// Auto-optimise rayon threads / batch sizes from system profile.
    #[serde(default)]
    pub auto_optimize: bool,
    /// Run in "ultra" preset (max resources). Mutually exclusive with manual
    /// pool / batch overrides which the UI greys out when this is on.
    #[serde(default)]
    pub ultra_performance: bool,
    /// Override rayon thread count (`None` = auto).
    #[serde(default)]
    pub rayon_threads: Option<u32>,
    /// DB connection pool size override (`None` = auto).
    #[serde(default)]
    pub pool_size: Option<u32>,
    /// Memory threshold in MB to trigger streaming (`None` = engine default).
    #[serde(default)]
    pub memory_threshold_mb: Option<u32>,
}

impl Default for MatchOptionsDto {
    fn default() -> Self {
        Self {
            allow_birthdate_swap: false,
            auto_optimize: true,
            ultra_performance: false,
            rayon_threads: None,
            pool_size: None,
            memory_threshold_mb: None,
        }
    }
}

/// Cascade matching options. When `enabled` is true, the run service routes
/// the job to `cascade::run_cascade_inmemory` instead of the single-pass
/// `match_all_with_opts`. Cascade runs L1–L11 sequentially; L12 (Household)
/// is intentionally excluded — use Options 5/6 for household matching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeOptionsDto {
    pub enabled: bool,
    /// Levels to run (1..=11). Empty defaults to all 11.
    pub levels: Vec<u8>,
    /// Fuzzy threshold (0.0..=1.0) applied to the fuzzy levels (L10/L11).
    pub fuzzy_threshold: f32,
    /// `exclusive` — pairs matched at earlier levels are removed from later
    /// level inputs. `independent` — every level runs against the full set.
    pub exclusion_mode: String,
    /// Geographic column availability detected on the source table. Drives
    /// auto-skip behaviour for L4–L9.
    #[serde(default)]
    pub has_barangay_code: bool,
    #[serde(default)]
    pub has_city_code: bool,
}

impl Default for CascadeOptionsDto {
    fn default() -> Self {
        Self {
            enabled: false,
            levels: Vec::new(),
            fuzzy_threshold: 0.95,
            exclusion_mode: "exclusive".into(),
            has_barangay_code: false,
            has_city_code: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunConfigDto {
    pub source: TableSelectionDto,
    pub target: TableSelectionDto,
    pub algorithm: AlgorithmDto,
    #[serde(default)]
    pub options: MatchOptionsDto,
    #[serde(default)]
    pub gpu: GpuOptionsDto,
    #[serde(default)]
    pub streaming: StreamingOptionsDto,
    pub export: ExportOptionsDto,
    /// When set with `enabled = true`, the job runs in cascade (Deep Match)
    /// mode and `algorithm` is ignored.
    #[serde(default)]
    pub cascade: Option<CascadeOptionsDto>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum JobStateDto {
    Idle,
    Validating,
    Starting,
    Running,
    Pausing,
    Paused,
    Resuming,
    Cancelling,
    Cancelled,
    Failed,
    Completed,
}

impl JobStateDto {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            JobStateDto::Cancelled | JobStateDto::Failed | JobStateDto::Completed
        )
    }
    pub fn is_active(self) -> bool {
        matches!(
            self,
            JobStateDto::Starting
                | JobStateDto::Validating
                | JobStateDto::Running
                | JobStateDto::Pausing
                | JobStateDto::Paused
                | JobStateDto::Resuming
                | JobStateDto::Cancelling
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PipelineStageDto {
    Load,
    Hash,
    Match,
    Fuzzy,
    Export,
    Idle,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressEventDto {
    pub job_id: String,
    pub state: JobStateDto,
    pub stage: PipelineStageDto,
    pub processed: u64,
    pub total: u64,
    pub percent: f32,
    pub eta_secs: u64,
    pub mem_used_mb: u64,
    pub mem_avail_mb: u64,
    /// 0 when CPU-only. Populated when a GPU run is in progress.
    #[serde(default)]
    pub gpu_total_mb: u64,
    #[serde(default)]
    pub gpu_free_mb: u64,
    #[serde(default)]
    pub gpu_active: bool,
    /// Records-per-second running estimate.
    #[serde(default)]
    pub records_per_sec: f32,
    /// Match count discovered so far.
    #[serde(default)]
    pub matches_found: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LogLevelDto {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntryDto {
    pub job_id: String,
    pub timestamp_ms: u64,
    pub level: LogLevelDto,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobStateEventDto {
    pub job_id: String,
    pub state: JobStateDto,
    /// Optional structured detail for `failed` / `cancelled` transitions.
    #[serde(default)]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ErrorKindDto {
    Validation,
    Database,
    Io,
    Engine,
    Cancelled,
    Internal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppErrorDto {
    pub kind: ErrorKindDto,
    pub message: String,
    /// True when retrying may succeed (e.g. transient DB errors).
    #[serde(default)]
    pub recoverable: bool,
}

/// Slim row delivered to the Results table. Every row carries enough
/// columns for the standard view; raw `Person` records stay in the backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchPairDto {
    pub row_id: u64,
    pub source_id: i64,
    pub source_uuid: Option<String>,
    pub source_full_name: String,
    pub source_birthdate: Option<String>,
    #[serde(default)]
    pub source_region_name: Option<String>,
    #[serde(default)]
    pub source_province_name: Option<String>,
    #[serde(default)]
    pub source_city_name: Option<String>,
    #[serde(default)]
    pub source_barangay_name: Option<String>,
    /// Optional full set of non-standard columns from the source table.
    #[serde(default)]
    pub source_extra_fields: BTreeMap<String, String>,
    pub target_id: i64,
    pub target_uuid: Option<String>,
    pub target_full_name: String,
    pub target_birthdate: Option<String>,
    #[serde(default)]
    pub target_region_name: Option<String>,
    #[serde(default)]
    pub target_province_name: Option<String>,
    #[serde(default)]
    pub target_city_name: Option<String>,
    #[serde(default)]
    pub target_barangay_name: Option<String>,
    /// Optional full set of non-standard columns from the target table.
    #[serde(default)]
    pub target_extra_fields: BTreeMap<String, String>,
    pub confidence: f32,
    pub matched_fields: Vec<String>,
    /// Plain-language explanation for why the pair was exported.
    #[serde(default)]
    pub remarks: Option<String>,
    /// In cascade (Deep Match) runs, the level (1..=11) at which this pair
    /// was first matched. `None` for single-pass (Quick Match) runs.
    #[serde(default)]
    pub matched_at_level: Option<u8>,
    /// Human-readable cascade method label for Deep Match rows.
    /// `None` for single-pass (Quick Match) runs.
    #[serde(default)]
    pub match_method: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultPageRequestDto {
    pub job_id: String,
    /// Cursor-based page index. 0 = first page.
    pub page: u32,
    /// 1..=10000.
    pub limit: u32,
    /// Optional confidence floor (0..=100).
    #[serde(default)]
    pub min_confidence: Option<f32>,
    /// Optional substring filter applied to source/target full-name columns.
    #[serde(default)]
    pub query: Option<String>,
    /// Optional sort key. One of `confidence`, `source_name`, `target_name`,
    /// `row_id`. Default: `row_id`.
    #[serde(default)]
    pub sort_by: Option<String>,
    /// Sort direction: `asc` | `desc`. Default: `desc` for confidence,
    /// `asc` otherwise.
    #[serde(default)]
    pub sort_dir: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultPageDto {
    pub job_id: String,
    pub page: u32,
    pub limit: u32,
    pub total: u64,
    pub rows: Vec<MatchPairDto>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobSummaryDto {
    pub job_id: String,
    pub state: JobStateDto,
    pub algorithm: AlgorithmDto,
    pub source_table: String,
    pub target_table: String,
    pub matches_found: u64,
    pub elapsed_secs: u64,
    pub started_at_unix_ms: u64,
    pub finished_at_unix_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportRequestDto {
    pub job_id: String,
    pub format: ExportFormatDto,
    pub output_directory: String,
    pub file_stem: String,
    /// Minimum confidence (0..=100) for export rows.
    #[serde(default)]
    pub min_confidence: Option<f32>,
    /// When true, append all non-standard source/target table columns.
    #[serde(default)]
    pub include_extra_fields: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportResultDto {
    pub job_id: String,
    pub format: ExportFormatDto,
    pub written_paths: Vec<String>,
    pub rows_exported: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfoDto {
    pub os: String,
    pub cpu_brand: String,
    pub cpu_cores_logical: u32,
    pub cpu_cores_physical: u32,
    pub memory_total_mb: u64,
    pub memory_avail_mb: u64,
    pub gpu_available: bool,
    pub gpu_devices: Vec<String>,
    pub rayon_threads: u32,
    pub app_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaDiagnosticsDto {
    pub gpu_feature_compiled: bool,
    pub device_count: u32,
    pub devices: Vec<String>,
    pub driver_version: Option<String>,
    pub error: Option<String>,
    /// Total VRAM in MB on device 0 (when gpu_feature_compiled and probe succeeded).
    #[serde(default)]
    pub vram_total_mb: Option<u64>,
    /// Free VRAM in MB at probe time on device 0.
    #[serde(default)]
    pub vram_free_mb: Option<u64>,
}
