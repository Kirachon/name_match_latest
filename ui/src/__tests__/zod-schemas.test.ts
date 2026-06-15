import { describe, expect, it } from "vitest";
import {
  parseCsvImportDryRunResult,
  parseCsvImportJob,
  parseCsvImportRequest,
  parseRunConfig,
} from "@/shared/tauri/zod-schemas";
import type {
  CsvImportDryRunResultDto,
  CsvImportJobDto,
  CsvImportRequestDto,
  RunConfigDto,
} from "@/shared/tauri/types";

function validRunConfig(): RunConfigDto {
  return {
    source: { session_id: "source-session", table: "source_table" },
    target: { session_id: "target-session", table: "target_table" },
    algorithm: "fuzzy",
    options: {
      allow_birthdate_swap: false,
      auto_optimize: true,
      ultra_performance: false,
      rayon_threads: null,
      pool_size: null,
      memory_threshold_mb: null,
      persist_result_history: false,
    },
    gpu: {
      mode: "auto",
      use_hash_join: false,
      use_direct_prefilter: false,
      use_levenshtein_full_scoring: false,
      vram_budget_mb: null,
      dynamic_tuning: true,
      fuzzy_gate_mode: "off",
    },
    streaming: {
      mode: "auto",
      batch_size: 50_000,
      partition_strategy: null,
      resume: false,
      checkpoint_path: null,
    },
    export: {
      format: "csv",
      output_directory: "D:\\exports",
      file_stem: "matches",
      min_confidence: null,
      include_extra_fields: true,
    },
    cascade: null,
  };
}

describe("RunConfigSchema", () => {
  it("accepts the baseline quick-match config", () => {
    expect(parseRunConfig(validRunConfig()).ok).toBe(true);
  });

  it("rejects GPU flags when compute mode is CPU", () => {
    const result = parseRunConfig({
      ...validRunConfig(),
      gpu: {
        ...validRunConfig().gpu,
        mode: "cpu",
        use_hash_join: true,
        use_direct_prefilter: true,
        use_levenshtein_full_scoring: true,
        fuzzy_gate_mode: "shadow",
      },
    });

    expect(result).toMatchObject({ ok: false });
    if (!result.ok) {
      expect(result.issues).toEqual(
        expect.arrayContaining([
          "gpu.use_hash_join: GPU hash-join requires GPU mode",
          "gpu.use_direct_prefilter: GPU prefilter requires GPU mode",
          "gpu.use_levenshtein_full_scoring: GPU full Levenshtein scoring requires GPU mode",
          "gpu.fuzzy_gate_mode: GPU fuzzy gate requires GPU mode",
        ]),
      );
    }
  });

  it("rejects manual thread overrides while ultra performance is enabled", () => {
    const result = parseRunConfig({
      ...validRunConfig(),
      options: {
        ...validRunConfig().options,
        ultra_performance: true,
        rayon_threads: 4,
        pool_size: 12,
      },
    });

    expect(result).toMatchObject({ ok: false });
    if (!result.ok) {
      expect(result.issues).toEqual(
        expect.arrayContaining([
          "options.rayon_threads: Disable Ultra Performance to set rayon threads manually",
          "options.pool_size: Disable Ultra Performance to set pool size manually",
        ]),
      );
    }
  });

  it("requires at least one cascade level when cascade mode is enabled", () => {
    const result = parseRunConfig({
      ...validRunConfig(),
      cascade: {
        enabled: true,
        levels: [],
        fuzzy_threshold: 0.95,
        exclusion_mode: "exclusive",
        has_barangay_code: true,
        has_city_code: false,
      },
    });

    expect(result).toMatchObject({ ok: false });
    if (!result.ok) {
      expect(result.issues).toContain(
        "cascade.levels: Pick at least one cascade level",
      );
    }
  });
});

function validCsvImportRequest(): CsvImportRequestDto {
  return {
    target: {
      session_id: "session-1",
      database: "matcher",
      table: "imported_people",
      mode: "create",
    },
    file: {
      path: "D:\\data\\people.csv",
      encoding: null,
      delimiter: null,
      date_format: "%Y-%m-%d",
    },
    mapping: {
      id: "id",
      uuid: "uuid",
      first_name: "first_name",
      middle_name: "middle_name",
      last_name: "last_name",
      birthdate: "birthdate",
      hh_id: "hh_id",
    },
    policy: {
      id_behavior: "use-csv-id",
      duplicate_behavior: "skip",
      duplicate_key: "id",
      batch_size: 5000,
      create_indexes: true,
      confirmed_destructive: false,
    },
  };
}

describe("CsvImportRequestSchema", () => {
  it("accepts the baseline CSV import request", () => {
    expect(parseCsvImportRequest(validCsvImportRequest()).ok).toBe(true);
  });

  it("rejects unsafe target table names", () => {
    const result = parseCsvImportRequest({
      ...validCsvImportRequest(),
      target: { ...validCsvImportRequest().target, table: "people;DROP" },
    });

    expect(result).toMatchObject({ ok: false });
  });

  it("accepts plan_hash from dry-run binding", () => {
    const result = parseCsvImportRequest({
      ...validCsvImportRequest(),
      plan_hash: "abc123deadbeef",
    });
    expect(result.ok).toBe(true);
  });

  it("requires explicit confirmation for replace mode", () => {
    const result = parseCsvImportRequest({
      ...validCsvImportRequest(),
      target: { ...validCsvImportRequest().target, mode: "replace" },
    });

    expect(result).toMatchObject({ ok: false });
    if (!result.ok) {
      expect(result.issues).toContain(
        "policy.confirmed_destructive: Replace mode requires explicit confirmation",
      );
    }
  });
});

function validCsvImportDryRun(): CsvImportDryRunResultDto {
  return {
    total_rows: 1000,
    valid_rows: 990,
    invalid_rows: 10,
    duplicate_rows: 5,
    new_rows: 985,
    skipped_rows: 5,
    updated_rows: 0,
    estimated_batches: 2,
    table_exists: false,
    will_create_table: true,
    will_replace_table: false,
    warnings: [],
    invalid_samples: [],
    planned_columns: ["id", "first_name", "last_name", "birthdate"],
    planned_indexes: [
      {
        name: "idx_name_bd",
        columns: ["last_name", "birthdate"],
        unique: false,
      },
    ],
    plan_hash: "plan-abc",
    duplicate_probe_status: "complete",
    staging_table: "_nm_import_staging",
    load_method: "load-data-infile",
  };
}

describe("CsvImportDryRunResultSchema", () => {
  it("accepts scale metadata from dry-run", () => {
    expect(parseCsvImportDryRunResult(validCsvImportDryRun()).ok).toBe(true);
  });

  it("rejects unknown duplicate_probe_status", () => {
    const result = parseCsvImportDryRunResult({
      ...validCsvImportDryRun(),
      duplicate_probe_status: "pending",
    });
    expect(result.ok).toBe(false);
  });
});

function validCsvImportJob(): CsvImportJobDto {
  return {
    job_id: "job-1",
    phase: "importing",
    total_rows: 1000,
    processed_rows: 500,
    inserted_rows: 480,
    updated_rows: 10,
    skipped_rows: 10,
    failed_rows: 0,
    current_batch: 1,
    total_batches: 2,
    table: "imported_people",
    message: "Importing batch 1",
    partial_commit: true,
    destructive_step_completed: false,
    staging_table: "_nm_import_staging",
    load_method: "batched-insert",
  };
}

describe("CsvImportJobSchema", () => {
  it("accepts scale metadata on import jobs", () => {
    expect(parseCsvImportJob(validCsvImportJob()).ok).toBe(true);
  });

  it("accepts jobs without optional scale fields", () => {
    const minimal = validCsvImportJob();
    delete minimal.partial_commit;
    delete minimal.destructive_step_completed;
    delete minimal.staging_table;
    delete minimal.load_method;
    expect(parseCsvImportJob(minimal).ok).toBe(true);
  });
});
