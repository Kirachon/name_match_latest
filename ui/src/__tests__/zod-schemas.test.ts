import { describe, expect, it } from "vitest";
import { parseRunConfig } from "@/shared/tauri/zod-schemas";
import type { RunConfigDto } from "@/shared/tauri/types";

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
      },
    });

    expect(result).toMatchObject({ ok: false });
    if (!result.ok) {
      expect(result.issues).toEqual(
        expect.arrayContaining([
          "gpu.use_hash_join: GPU hash-join requires GPU mode",
          "gpu.use_direct_prefilter: GPU prefilter requires GPU mode",
          "gpu.use_levenshtein_full_scoring: GPU full Levenshtein scoring requires GPU mode",
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
