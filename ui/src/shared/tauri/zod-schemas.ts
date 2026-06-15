import { z } from "zod";
import type {
  CsvImportDryRunResultDto,
  CsvImportJobDto,
  CsvImportRequestDto,
  RunConfigDto,
} from "./types";

// ---------- Database credentials ----------

export const DbCredentialsSchema = z.object({
  host: z.string().min(1, "Host is required"),
  port: z.coerce
    .number()
    .int()
    .min(1, "Port must be > 0")
    .max(65535, "Port must be ≤ 65535"),
  username: z.string().min(1, "Username is required"),
  password: z.string(),
  database: z.string().min(1, "Database name is required"),
});

// ---------- Run config (deep validation, dependency-aware) ----------

const ColumnMappingSchema = z.object({
  id: z.string().min(1, "Map an ID column"),
  uuid: z.string().min(1).nullable().optional(),
  first_name: z.string().min(1, "Map a first name column"),
  middle_name: z.string().min(1).nullable().optional(),
  last_name: z.string().min(1, "Map a last name column"),
  birthdate: z.string().min(1, "Map a birthdate column"),
  hh_id: z.string().min(1).nullable().optional(),
});

const FileSelectionSchema = z.object({
  path: z.string().min(1, "Choose a file"),
  sheet_name: z.string().nullable().optional(),
  encoding: z
    .enum(["utf8", "utf8-bom", "windows1252", "latin1"])
    .nullable()
    .optional(),
  delimiter: z.enum(["comma", "semicolon", "tab"]).nullable().optional(),
  date_format: z.string().nullable().optional(),
});

const CsvImportTargetSchema = z.object({
  session_id: z.string().min(1, "Connect to a database first"),
  database: z.string().min(1, "Database is required"),
  table: z
    .string()
    .min(1, "Target table is required")
    .regex(/^[A-Za-z0-9_$]+$/, "Use a safe table name"),
  mode: z.enum(["create", "append", "replace"]),
});

const CsvImportPolicySchema = z
  .object({
    id_behavior: z.enum([
      "use-csv-id",
      "generate-id",
      "db-auto-increment",
      "use-csv-uuid",
      "generate-uuid",
    ]),
    duplicate_behavior: z.enum(["skip", "update", "insert-anyway", "fail"]),
    duplicate_key: z.enum(["id", "uuid", "matcher-fields"]),
    batch_size: z.number().int().min(1).max(200_000),
    create_indexes: z.boolean(),
    confirmed_destructive: z.boolean(),
  })
  .superRefine((policy, ctx) => {
    if (
      policy.duplicate_behavior === "update" &&
      policy.duplicate_key === "matcher-fields"
    ) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["duplicate_key"],
        message: "Update duplicates requires ID or UUID as the duplicate key",
      });
    }
  });

const CsvImportDuplicateProbeStatusSchema = z.enum([
  "complete",
  "sampled",
  "failed",
  "blocked-needs-index",
]);

const CsvImportLoadMethodSchema = z.enum([
  "batched-insert",
  "load-data-infile",
]);

const CsvImportInvalidRowSchema = z.object({
  row_number: z.number().int().min(1),
  reason: z.string().min(1),
});

const CsvImportIndexPlanSchema = z.object({
  name: z.string().min(1),
  columns: z.array(z.string().min(1)),
  unique: z.boolean(),
});

export const CsvImportDryRunResultSchema = z.object({
  total_rows: z.number().int().min(0),
  valid_rows: z.number().int().min(0),
  invalid_rows: z.number().int().min(0),
  duplicate_rows: z.number().int().min(0),
  new_rows: z.number().int().min(0),
  skipped_rows: z.number().int().min(0),
  updated_rows: z.number().int().min(0),
  estimated_batches: z.number().int().min(0),
  table_exists: z.boolean(),
  will_create_table: z.boolean(),
  will_replace_table: z.boolean(),
  warnings: z.array(z.string()),
  invalid_samples: z.array(CsvImportInvalidRowSchema),
  planned_columns: z.array(z.string()),
  planned_indexes: z.array(CsvImportIndexPlanSchema),
  plan_hash: z.string().min(1),
  duplicate_probe_status: CsvImportDuplicateProbeStatusSchema.optional(),
  staging_table: z.string().nullable().optional(),
  load_method: CsvImportLoadMethodSchema.optional(),
});

export const CsvImportJobSchema = z.object({
  job_id: z.string().min(1),
  phase: z.enum([
    "creating-table",
    "importing",
    "creating-indexes",
    "validating",
    "refreshing-source",
    "complete",
    "failed",
    "cancelled",
  ]),
  total_rows: z.number().int().min(0),
  processed_rows: z.number().int().min(0),
  inserted_rows: z.number().int().min(0),
  updated_rows: z.number().int().min(0),
  skipped_rows: z.number().int().min(0),
  failed_rows: z.number().int().min(0),
  current_batch: z.number().int().min(0),
  total_batches: z.number().int().min(0),
  table: z.string().min(1),
  message: z.string().nullable().optional(),
  error: z.string().nullable().optional(),
  dry_run: CsvImportDryRunResultSchema.nullable().optional(),
  partial_commit: z.boolean().optional(),
  destructive_step_completed: z.boolean().optional(),
  staging_table: z.string().nullable().optional(),
  load_method: CsvImportLoadMethodSchema.optional(),
});

export const CsvImportRequestSchema = z
  .object({
    target: CsvImportTargetSchema,
    file: FileSelectionSchema,
    mapping: ColumnMappingSchema,
    policy: CsvImportPolicySchema,
    plan_hash: z.string().min(1).nullable().optional(),
  })
  .superRefine((request, ctx) => {
    if (
      request.target.mode === "replace" &&
      !request.policy.confirmed_destructive
    ) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["policy", "confirmed_destructive"],
        message: "Replace mode requires explicit confirmation",
      });
    }
    if (request.policy.id_behavior === "use-csv-id" && !request.mapping.id) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["mapping", "id"],
        message: "CSV ID mode requires an ID mapping",
      });
    }
  });

const TableSelectionSchema = z
  .object({
    source_kind: z.enum(["database", "file"]).default("database"),
    session_id: z.string().optional().default(""),
    table: z.string().optional().default(""),
    column_mapping: ColumnMappingSchema.nullable().optional(),
    file: FileSelectionSchema.nullable().optional(),
  })
  .superRefine((selection, ctx) => {
    if (selection.source_kind === "database") {
      if (!selection.session_id) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["session_id"],
          message: "Connect to a database first",
        });
      }
      if (!selection.table) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["table"],
          message: "Select a table",
        });
      }
    } else if (!selection.file?.path) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["file", "path"],
        message: "Choose a file",
      });
    }
  });

const GpuOptionsSchema = z.object({
  mode: z.enum(["cpu", "auto", "force-gpu"]),
  use_hash_join: z.boolean(),
  use_direct_prefilter: z.boolean(),
  use_levenshtein_full_scoring: z.boolean(),
  vram_budget_mb: z.number().int().min(64).max(65536).nullable().optional(),
  dynamic_tuning: z.boolean(),
  fuzzy_gate_mode: z.enum(["off", "shadow", "gate-only"]).default("off"),
});

const StreamingOptionsSchema = z.object({
  mode: z.enum(["auto", "streaming", "in-memory"]),
  batch_size: z
    .number()
    .int()
    .min(1_000, "Batch size must be ≥ 1,000")
    .max(200_000, "Batch size must be ≤ 200,000"),
  partition_strategy: z.string().nullable().optional(),
  resume: z.boolean(),
  checkpoint_path: z.string().nullable().optional(),
});

const ExportOptionsSchema = z.object({
  format: z.enum(["csv", "xlsx", "both"]),
  output_directory: z.string().min(1, "Choose an output folder"),
  file_stem: z
    .string()
    .min(1, "File stem is required")
    .regex(
      /^[A-Za-z0-9._-]+$/,
      "Use letters, digits, dot, underscore, or hyphen",
    ),
  min_confidence: z.number().min(0).max(100).nullable().optional(),
  review_band: z
    .object({
      min_confidence: z.number().min(0).max(100),
      max_confidence: z.number().min(0).max(100),
    })
    .refine((band) => band.min_confidence <= band.max_confidence, {
      message: "Review band minimum must be <= maximum",
      path: ["min_confidence"],
    })
    .nullable()
    .optional(),
  include_extra_fields: z.boolean(),
});

const MatchOptionsSchema = z.object({
  allow_birthdate_swap: z.boolean(),
  auto_optimize: z.boolean(),
  ultra_performance: z.boolean(),
  rayon_threads: z.number().int().min(1).max(256).nullable().optional(),
  pool_size: z.number().int().min(1).max(256).nullable().optional(),
  memory_threshold_mb: z
    .number()
    .int()
    .min(256)
    .max(262_144)
    .nullable()
    .optional(),
  persist_result_history: z.boolean(),
});

const CascadeSchema = z
  .object({
    enabled: z.boolean(),
    levels: z.array(z.number().int().min(1).max(11)),
    fuzzy_threshold: z.number().min(0).max(1),
    exclusion_mode: z.enum(["exclusive", "independent"]),
    has_barangay_code: z.boolean(),
    has_city_code: z.boolean(),
  })
  .nullable()
  .optional();

export const SaveDecisionSchema = z.object({
  job_id: z.string().min(1),
  row_id: z.number().int().min(0),
  source_id: z.number().int(),
  target_id: z.number().int(),
  decision: z.enum(["accepted", "rejected", "pending"]),
  note: z.string().nullable().optional(),
});

export const DiffJobsSchema = z
  .object({
    base_job_id: z.string().min(1),
    compare_job_id: z.string().min(1),
  })
  .refine((value) => value.base_job_id !== value.compare_job_id, {
    message: "Choose two different jobs",
    path: ["compare_job_id"],
  });

export const RunConfigSchema = z
  .object({
    source: TableSelectionSchema,
    target: TableSelectionSchema,
    algorithm: z.enum([
      "deterministic-fn-ln-bd",
      "deterministic-fn-mn-ln-bd",
      "fuzzy",
      "fuzzy-no-middle",
      "household-gpu",
      "household-gpu-opt6",
      "levenshtein-weighted",
    ]),
    options: MatchOptionsSchema,
    gpu: GpuOptionsSchema,
    streaming: StreamingOptionsSchema,
    export: ExportOptionsSchema,
    review_band: ExportOptionsSchema.shape.review_band,
    cascade: CascadeSchema,
  })
  .superRefine((cfg, ctx) => {
    // GPU memory fields require a non-CPU mode.
    if (cfg.gpu.mode === "cpu") {
      if (cfg.gpu.use_hash_join) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: "GPU hash-join requires GPU mode",
          path: ["gpu", "use_hash_join"],
        });
      }
      if (cfg.gpu.use_direct_prefilter) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: "GPU prefilter requires GPU mode",
          path: ["gpu", "use_direct_prefilter"],
        });
      }
      if (cfg.gpu.use_levenshtein_full_scoring) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: "GPU full Levenshtein scoring requires GPU mode",
          path: ["gpu", "use_levenshtein_full_scoring"],
        });
      }
      if (cfg.gpu.fuzzy_gate_mode !== "off") {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: "GPU fuzzy gate requires GPU mode",
          path: ["gpu", "fuzzy_gate_mode"],
        });
      }
    }
    // Ultra performance is mutually exclusive with manual overrides.
    if (cfg.options.ultra_performance) {
      if (cfg.options.rayon_threads != null) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: "Disable Ultra Performance to set rayon threads manually",
          path: ["options", "rayon_threads"],
        });
      }
      if (cfg.options.pool_size != null) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: "Disable Ultra Performance to set pool size manually",
          path: ["options", "pool_size"],
        });
      }
    }
    // Cascade requires at least one level.
    if (cfg.cascade && cfg.cascade.enabled && cfg.cascade.levels.length === 0) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "Pick at least one cascade level",
        path: ["cascade", "levels"],
      });
    }
  });

export type RunConfigInput = z.input<typeof RunConfigSchema>;
export type RunConfigParsed = z.output<typeof RunConfigSchema>;

export function parseRunConfig(
  value: unknown,
): { ok: true; value: RunConfigDto } | { ok: false; issues: string[] } {
  const r = RunConfigSchema.safeParse(value);
  if (r.success) return { ok: true, value: r.data as RunConfigDto };
  return {
    ok: false,
    issues: r.error.issues.map(
      (i) => `${i.path.join(".") || "root"}: ${i.message}`,
    ),
  };
}

export function parseCsvImportRequest(
  value: unknown,
): { ok: true; value: CsvImportRequestDto } | { ok: false; issues: string[] } {
  const r = CsvImportRequestSchema.safeParse(value);
  if (r.success) return { ok: true, value: r.data as CsvImportRequestDto };
  return {
    ok: false,
    issues: r.error.issues.map(
      (i) => `${i.path.join(".") || "root"}: ${i.message}`,
    ),
  };
}

export function parseCsvImportDryRunResult(
  value: unknown,
):
  | { ok: true; value: CsvImportDryRunResultDto }
  | { ok: false; issues: string[] } {
  const r = CsvImportDryRunResultSchema.safeParse(value);
  if (r.success) return { ok: true, value: r.data as CsvImportDryRunResultDto };
  return {
    ok: false,
    issues: r.error.issues.map(
      (i) => `${i.path.join(".") || "root"}: ${i.message}`,
    ),
  };
}

export function parseCsvImportJob(
  value: unknown,
): { ok: true; value: CsvImportJobDto } | { ok: false; issues: string[] } {
  const r = CsvImportJobSchema.safeParse(value);
  if (r.success) return { ok: true, value: r.data as CsvImportJobDto };
  return {
    ok: false,
    issues: r.error.issues.map(
      (i) => `${i.path.join(".") || "root"}: ${i.message}`,
    ),
  };
}
