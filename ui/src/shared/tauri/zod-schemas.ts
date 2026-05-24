import { z } from "zod";
import type { RunConfigDto } from "./types";

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

const TableSelectionSchema = z.object({
  session_id: z.string().min(1, "Connect to a database first"),
  table: z.string().min(1, "Select a table"),
  column_mapping: ColumnMappingSchema.nullable().optional(),
});

const GpuOptionsSchema = z.object({
  mode: z.enum(["cpu", "auto", "force-gpu"]),
  use_hash_join: z.boolean(),
  use_direct_prefilter: z.boolean(),
  use_levenshtein_full_scoring: z.boolean(),
  vram_budget_mb: z.number().int().min(64).max(65536).nullable().optional(),
  dynamic_tuning: z.boolean(),
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
