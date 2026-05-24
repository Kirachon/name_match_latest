// =============================================================================
// Frozen TypeScript mirror of `name_matcher::run_service::dto`.
//
// This file is the single source of truth on the front-end side. Whenever the
// Rust DTOs change, update this file AND `zod-schemas.ts` AND the relevant
// component types. CI (T13) will diff this file against the Rust source as
// part of the contract gate.
//
// The wire format uses kebab-case for enum variants (matching the Rust
// `serde(rename_all = "kebab-case")` attribute). Field names stay snake_case
// to match Rust struct field serialisation.
// =============================================================================

/* ---------- Algorithm + mode enums ---------- */

export type AlgorithmDto =
  | "deterministic-fn-ln-bd"
  | "deterministic-fn-mn-ln-bd"
  | "fuzzy"
  | "fuzzy-no-middle"
  | "household-gpu"
  | "household-gpu-opt6"
  | "levenshtein-weighted";

export type ComputeModeDto = "cpu" | "auto" | "force-gpu";
export type RunModeDto = "auto" | "streaming" | "in-memory";
export type ExportFormatDto = "csv" | "xlsx" | "both";
export type CsvEncodingDto = "utf8" | "utf8-bom" | "windows1252" | "latin1";
export type CsvDelimiterDto = "comma" | "semicolon" | "tab";
export type DataSourceKindDto = "database" | "file";

/* ---------- Database ---------- */

export interface DbCredentialsDto {
  host: string;
  port: number;
  username: string;
  password: string;
  database: string;
}

export interface DbSessionDto {
  session_id: string;
  host: string;
  port: number;
  username: string;
  database: string;
  latency_ms?: number | null;
}

export interface TableInfoDto {
  name: string;
  schema: string;
  row_count?: number | null;
}

export interface TableColumnsDto {
  has_id: boolean;
  has_uuid: boolean;
  has_first_name: boolean;
  has_middle_name: boolean;
  has_last_name: boolean;
  has_birthdate: boolean;
  has_hh_id: boolean;
  raw_columns: string[];
}

export interface ColumnMappingDto {
  id: string;
  uuid?: string | null;
  first_name: string;
  middle_name?: string | null;
  last_name: string;
  birthdate: string;
  hh_id?: string | null;
}

export interface TableSelectionDto {
  source_kind?: DataSourceKindDto;
  session_id: string;
  table: string;
  column_mapping?: ColumnMappingDto | null;
  file?: FileSelectionDto | null;
}

export interface CsvPreviewRequestDto {
  path: string;
  encoding?: CsvEncodingDto | null;
  delimiter?: CsvDelimiterDto | null;
  date_format?: string | null;
}

export interface FileSelectionDto {
  path: string;
  encoding?: CsvEncodingDto | null;
  delimiter?: CsvDelimiterDto | null;
  date_format?: string | null;
}

export interface CsvPreviewDto {
  path: string;
  encoding: CsvEncodingDto;
  delimiter: CsvDelimiterDto;
  headers: string[];
  rows: string[][];
  warnings: string[];
  date_format: string;
  total_preview_rows: number;
}

/* ---------- Run config ---------- */

export interface GpuOptionsDto {
  mode: ComputeModeDto;
  use_hash_join: boolean;
  use_direct_prefilter: boolean;
  use_levenshtein_full_scoring: boolean;
  vram_budget_mb?: number | null;
  dynamic_tuning: boolean;
}

export interface StreamingOptionsDto {
  mode: RunModeDto;
  batch_size: number;
  partition_strategy?: string | null;
  resume: boolean;
  checkpoint_path?: string | null;
}

export interface ExportOptionsDto {
  format: ExportFormatDto;
  output_directory: string;
  file_stem: string;
  min_confidence?: number | null;
  include_extra_fields: boolean;
}

export interface MatchOptionsDto {
  allow_birthdate_swap: boolean;
  auto_optimize: boolean;
  ultra_performance: boolean;
  rayon_threads?: number | null;
  pool_size?: number | null;
  memory_threshold_mb?: number | null;
}

export interface RunConfigDto {
  source: TableSelectionDto;
  target: TableSelectionDto;
  algorithm: AlgorithmDto;
  options: MatchOptionsDto;
  gpu: GpuOptionsDto;
  streaming: StreamingOptionsDto;
  export: ExportOptionsDto;
  /** Set with `enabled: true` to run cascade (Deep Match). */
  cascade?: CascadeOptionsDto | null;
}

/* ---------- Cascade (Deep Match) ---------- */

export type CascadeExclusionMode = "exclusive" | "independent";
export type CascadePresetId = "standard" | "extended" | "full" | "custom";

export interface CascadeOptionsDto {
  enabled: boolean;
  /** Levels (1..=11). Empty = all 11. L12 (Household) is intentionally excluded. */
  levels: number[];
  fuzzy_threshold: number;
  exclusion_mode: CascadeExclusionMode;
  has_barangay_code: boolean;
  has_city_code: boolean;
}

export interface CascadeLevelMeta {
  id: number;
  label: string;
  description: string;
  group: "name" | "barangay" | "city" | "fuzzy";
  /** Required raw column on source/target tables. `null` for name-only. */
  requiresColumn: "barangay_code" | "city_code" | null;
}

export const CASCADE_LEVELS: CascadeLevelMeta[] = [
  {
    id: 1,
    label: "L1 — Birthdate + Full Middle",
    description: "Exact match on first/middle/last + birthdate.",
    group: "name",
    requiresColumn: null,
  },
  {
    id: 2,
    label: "L2 — Birthdate + Middle Initial",
    description: "Same as L1 but matches on middle-name initial only.",
    group: "name",
    requiresColumn: null,
  },
  {
    id: 3,
    label: "L3 — Birthdate + No Middle",
    description: "Exact match on first/last + birthdate; ignores middle name.",
    group: "name",
    requiresColumn: null,
  },
  {
    id: 4,
    label: "L4 — Barangay + Full Middle",
    description: "Adds barangay_code grouping to L1.",
    group: "barangay",
    requiresColumn: "barangay_code",
  },
  {
    id: 5,
    label: "L5 — Barangay + Middle Initial",
    description: "Adds barangay_code grouping to L2.",
    group: "barangay",
    requiresColumn: "barangay_code",
  },
  {
    id: 6,
    label: "L6 — Barangay + No Middle",
    description: "Adds barangay_code grouping to L3.",
    group: "barangay",
    requiresColumn: "barangay_code",
  },
  {
    id: 7,
    label: "L7 — City + Full Middle",
    description: "Adds city_code grouping to L1.",
    group: "city",
    requiresColumn: "city_code",
  },
  {
    id: 8,
    label: "L8 — City + Middle Initial",
    description: "Adds city_code grouping to L2.",
    group: "city",
    requiresColumn: "city_code",
  },
  {
    id: 9,
    label: "L9 — City + No Middle",
    description: "Adds city_code grouping to L3.",
    group: "city",
    requiresColumn: "city_code",
  },
  {
    id: 10,
    label: "L10 — Fuzzy (with middle)",
    description: "Jaro-Winkler / Levenshtein fuzzy with middle name.",
    group: "fuzzy",
    requiresColumn: null,
  },
  {
    id: 11,
    label: "L11 — Fuzzy (no middle)",
    description: "Jaro-Winkler / Levenshtein fuzzy without middle name.",
    group: "fuzzy",
    requiresColumn: null,
  },
];

export const CASCADE_PRESETS: Record<
  CascadePresetId,
  { label: string; levels: number[]; description: string }
> = {
  standard: {
    label: "Standard",
    levels: [1, 2, 3, 10, 11],
    description: "Name-only matching. L1–L3 exact + L10–L11 fuzzy.",
  },
  extended: {
    label: "Extended",
    levels: [1, 2, 3, 4, 5, 6, 10, 11],
    description: "Adds barangay_code grouping to Standard.",
  },
  full: {
    label: "Full",
    levels: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    description: "Every available level, L1–L11.",
  },
  custom: {
    label: "Custom",
    levels: [],
    description: "Pick exactly which levels to run.",
  },
};

export function autoSelectCascadePreset(opts: {
  hasBarangay: boolean;
  hasCity: boolean;
}): CascadePresetId {
  if (opts.hasBarangay && opts.hasCity) return "full";
  if (opts.hasBarangay) return "extended";
  return "standard";
}

/* ---------- Job + events ---------- */

export type JobStateDto =
  | "idle"
  | "validating"
  | "starting"
  | "running"
  | "pausing"
  | "paused"
  | "resuming"
  | "cancelling"
  | "cancelled"
  | "failed"
  | "completed";

export const JOB_STATE_TERMINAL: JobStateDto[] = [
  "cancelled",
  "failed",
  "completed",
];
export const JOB_STATE_ACTIVE: JobStateDto[] = [
  "starting",
  "validating",
  "running",
  "pausing",
  "paused",
  "resuming",
  "cancelling",
];

export type PipelineStageDto =
  | "load"
  | "hash"
  | "match"
  | "fuzzy"
  | "export"
  | "idle";

export interface ProgressEventDto {
  job_id: string;
  state: JobStateDto;
  stage: PipelineStageDto;
  processed: number;
  total: number;
  percent: number;
  eta_secs: number;
  mem_used_mb: number;
  mem_avail_mb: number;
  gpu_total_mb: number;
  gpu_free_mb: number;
  gpu_active: boolean;
  records_per_sec: number;
  matches_found: number;
}

export type LogLevelDto = "trace" | "debug" | "info" | "warn" | "error";

export interface LogEntryDto {
  job_id: string;
  timestamp_ms: number;
  level: LogLevelDto;
  message: string;
}

export interface JobStateEventDto {
  job_id: string;
  state: JobStateDto;
  detail?: string | null;
}

export type ErrorKindDto =
  | "validation"
  | "database"
  | "io"
  | "engine"
  | "cancelled"
  | "internal";

export interface AppErrorDto {
  kind: ErrorKindDto;
  message: string;
  recoverable: boolean;
}

/* ---------- Results ---------- */

export interface MatchPairDto {
  row_id: number;
  source_id: number;
  source_uuid?: string | null;
  source_full_name: string;
  source_birthdate?: string | null;
  source_region_name?: string | null;
  source_province_name?: string | null;
  source_city_name?: string | null;
  source_barangay_name?: string | null;
  source_extra_fields?: Record<string, string>;
  target_id: number;
  target_uuid?: string | null;
  target_full_name: string;
  target_birthdate?: string | null;
  target_region_name?: string | null;
  target_province_name?: string | null;
  target_city_name?: string | null;
  target_barangay_name?: string | null;
  target_extra_fields?: Record<string, string>;
  confidence: number;
  matched_fields: string[];
  /** Plain-language reason for why the pair matched. */
  remarks?: string | null;
  /** Cascade level (1..=11) that produced this pair, when applicable. */
  matched_at_level?: number | null;
  /** Human-readable cascade method label, when applicable. */
  match_method?: string | null;
}

export interface ExplainPairRequestDto {
  job_id: string;
  source_id: number;
  target_id: number;
}

export interface ScoreBreakdownDto {
  supported: boolean;
  algorithm: string;
  case_label?: string | null;
  confidence?: number | null;
  levenshtein_pct?: number | null;
  jaro_winkler_pct?: number | null;
  metaphone_pct?: number | null;
  birthdate_match?: boolean | null;
  birthdate_swap_used: boolean;
  message?: string | null;
}

export interface ResultPageRequestDto {
  job_id: string;
  page: number;
  limit: number;
  min_confidence?: number | null;
  query?: string | null;
  sort_by?: string | null;
  sort_dir?: string | null;
  levels?: number[];
}

export interface ResultPageDto {
  job_id: string;
  page: number;
  limit: number;
  total: number;
  available_levels: number[];
  level_counts: Record<string, number>;
  rows: MatchPairDto[];
}

export type ReviewDecisionValue = "accepted" | "rejected" | "pending";

export interface SaveDecisionRequestDto {
  job_id: string;
  row_id: number;
  source_id: number;
  target_id: number;
  decision: ReviewDecisionValue;
  note?: string | null;
}

export interface ReviewDecisionDto extends SaveDecisionRequestDto {
  updated_at_unix_ms: number;
}

export interface JobSummaryDto {
  job_id: string;
  state: JobStateDto;
  algorithm: AlgorithmDto;
  source_table: string;
  target_table: string;
  matches_found: number;
  elapsed_secs: number;
  started_at_unix_ms: number;
  finished_at_unix_ms?: number | null;
}

export interface ExportRequestDto {
  job_id: string;
  format: ExportFormatDto;
  output_directory: string;
  file_stem: string;
  min_confidence?: number | null;
  levels?: number[];
  include_extra_fields?: boolean;
}

export interface ExportResultDto {
  job_id: string;
  format: ExportFormatDto;
  written_paths: string[];
  rows_exported: number;
}

export interface SystemInfoDto {
  os: string;
  cpu_brand: string;
  cpu_cores_logical: number;
  cpu_cores_physical: number;
  memory_total_mb: number;
  memory_avail_mb: number;
  gpu_available: boolean;
  gpu_devices: string[];
  rayon_threads: number;
  app_version: string;
}

export interface CudaDiagnosticsDto {
  gpu_feature_compiled: boolean;
  device_count: number;
  devices: string[];
  driver_version?: string | null;
  error?: string | null;
  vram_total_mb?: number | null;
  vram_free_mb?: number | null;
}

/* ---------- Algorithm metadata for the Configure UI ---------- */

export interface AlgorithmMeta {
  id: AlgorithmDto;
  optionNumber: number;
  label: string;
  description: string;
  /** Required column names on both source and target tables. */
  requiresColumns: Array<keyof TableColumnsDto>;
  /** True when the algorithm benefits from / can use the GPU panel. */
  gpuApplicable: boolean;
  /** True when fuzzy thresholds (`min_confidence`) are meaningful. */
  fuzzyTuneable: boolean;
}

export const ALGORITHMS: AlgorithmMeta[] = [
  {
    id: "deterministic-fn-ln-bd",
    optionNumber: 1,
    label: "Deterministic — first + last + birthdate",
    description:
      "Exact match on normalised first name, last name, and birthdate. Fastest path; ideal when both tables are clean and complete.",
    requiresColumns: [
      "has_id",
      "has_first_name",
      "has_last_name",
      "has_birthdate",
    ],
    gpuApplicable: true,
    fuzzyTuneable: false,
  },
  {
    id: "deterministic-fn-mn-ln-bd",
    optionNumber: 2,
    label: "Deterministic — first + middle + last + birthdate",
    description:
      "Exact match incorporating middle name. Use when middle names are reliably populated to break ties.",
    requiresColumns: [
      "has_id",
      "has_first_name",
      "has_middle_name",
      "has_last_name",
      "has_birthdate",
    ],
    gpuApplicable: true,
    fuzzyTuneable: false,
  },
  {
    id: "fuzzy",
    optionNumber: 3,
    label: "Fuzzy — Levenshtein + Jaro-Winkler + Metaphone",
    description:
      "Similarity-based match with middle name. Combines Levenshtein, Jaro-Winkler, and Double Metaphone for resilient matching.",
    requiresColumns: [
      "has_id",
      "has_first_name",
      "has_last_name",
      "has_birthdate",
    ],
    gpuApplicable: true,
    fuzzyTuneable: true,
  },
  {
    id: "fuzzy-no-middle",
    optionNumber: 4,
    label: "Fuzzy — without middle name",
    description:
      "Same fuzzy stack as Option 3 but ignores middle name. Use when middle names are sparse or unreliable.",
    requiresColumns: [
      "has_id",
      "has_first_name",
      "has_last_name",
      "has_birthdate",
    ],
    gpuApplicable: true,
    fuzzyTuneable: true,
  },
  {
    id: "household-gpu",
    optionNumber: 5,
    label: "Household — uuid → hh_id",
    description:
      "Aggregated household matching from Table 1 → Table 2. Requires hh_id on the target table.",
    requiresColumns: [
      "has_id",
      "has_uuid",
      "has_first_name",
      "has_last_name",
      "has_birthdate",
    ],
    gpuApplicable: true,
    fuzzyTuneable: true,
  },
  {
    id: "household-gpu-opt6",
    optionNumber: 6,
    label: "Household — hh_id → uuid (role-swapped)",
    description:
      "Role-swapped household aggregation. Denominator switches to Table 2 size for completeness scoring.",
    requiresColumns: [
      "has_id",
      "has_uuid",
      "has_first_name",
      "has_last_name",
      "has_birthdate",
    ],
    gpuApplicable: true,
    fuzzyTuneable: true,
  },
  {
    id: "levenshtein-weighted",
    optionNumber: 7,
    label: "Levenshtein-weighted (SQL equivalent)",
    description:
      "Weighted Levenshtein scoring matching the SQL reference implementation. GPU acceleration available for prefilter and full scoring.",
    requiresColumns: [
      "has_id",
      "has_first_name",
      "has_last_name",
      "has_birthdate",
    ],
    gpuApplicable: true,
    fuzzyTuneable: true,
  },
];

export function algorithmMeta(id: AlgorithmDto): AlgorithmMeta {
  const found = ALGORITHMS.find((a) => a.id === id);
  if (!found) throw new Error(`Unknown algorithm id: ${id}`);
  return found;
}

/* ---------- Defaults ---------- */

export const DEFAULT_GPU_OPTIONS: GpuOptionsDto = {
  mode: "cpu",
  use_hash_join: false,
  use_direct_prefilter: false,
  use_levenshtein_full_scoring: false,
  vram_budget_mb: null,
  dynamic_tuning: false,
};

export const DEFAULT_STREAMING_OPTIONS: StreamingOptionsDto = {
  mode: "auto",
  batch_size: 10_000,
  partition_strategy: null,
  resume: false,
  checkpoint_path: null,
};

export const DEFAULT_MATCH_OPTIONS: MatchOptionsDto = {
  allow_birthdate_swap: false,
  auto_optimize: true,
  ultra_performance: false,
  rayon_threads: null,
  pool_size: null,
  memory_threshold_mb: null,
};

export const DEFAULT_EXPORT_OPTIONS: ExportOptionsDto = {
  format: "csv",
  output_directory: "",
  file_stem: "matches",
  min_confidence: null,
  include_extra_fields: false,
};
