import { describe, expect, it } from "vitest";
import {
  algorithmSupportsDbStreaming,
  resolveEffectiveRunMode,
  scaleBlockReason,
  streamingBackendActive,
} from "@/shared/runScalePolicy";
import type { RunConfigDto } from "@/shared/tauri/types";

function dbConfig(rows: number, algorithm: RunConfigDto["algorithm"]): RunConfigDto {
  const side = {
    source_kind: "database" as const,
    session_id: "s",
    table: "t",
    row_count: rows,
  };
  return {
    source: side,
    target: { ...side, table: "t2" },
    algorithm,
    streaming: {
      mode: "auto",
      batch_size: 10_000,
      resume: false,
      partition_strategy: "last_initial",
    },
    options: {
      allow_birthdate_swap: false,
      auto_optimize: false,
      ultra_performance: false,
      persist_result_history: true,
      rayon_threads: null,
    },
    gpu: {
      mode: "cpu",
      use_hash_join: false,
      use_direct_prefilter: false,
      use_levenshtein_full_scoring: false,
      dynamic_tuning: false,
      fuzzy_gate_mode: "off",
    },
    export: {
      output_directory: "out",
      file_stem: "run",
      format: "csv",
      include_extra_fields: false,
    },
  };
}

describe("runScalePolicy", () => {
  it("auto resolves to streaming at 100k", () => {
    expect(resolveEffectiveRunMode("auto", 120_000, 50_000)).toBe("streaming");
  });

  it("blocks million-row fuzzy", () => {
    const cfg = dbConfig(1_500_000, "fuzzy");
    expect(scaleBlockReason(cfg)).toBe("million-row-fuzzy");
  });

  it("detects active streaming for deterministic db runs", () => {
    const cfg = dbConfig(200_000, "deterministic-fn-ln-bd");
    cfg.streaming.mode = "streaming";
    expect(algorithmSupportsDbStreaming(cfg.algorithm)).toBe(true);
    expect(streamingBackendActive(cfg)).toBe(true);
  });

  it("blocks million-row file sources", () => {
    const cfg = dbConfig(1_200_000, "deterministic-fn-ln-bd");
    cfg.source.source_kind = "file";
    expect(scaleBlockReason(cfg)).toBe("million-row-file-source");
  });
});
