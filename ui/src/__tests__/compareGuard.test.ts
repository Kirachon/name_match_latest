import { describe, expect, it } from "vitest";
import {
  compareBlockedReason,
  crossSessionDbStreamingMessage,
  DIFF_TOO_LARGE_MESSAGE,
  MAX_DIFF_ROWS,
  needsCrossSessionDbStreamingNotice,
} from "@/shared/runScalePolicy";
import type { RunConfigDto } from "@/shared/tauri/types";

function dbConfig(
  rows: number,
  algorithm: RunConfigDto["algorithm"],
  sessionId = "s",
): RunConfigDto {
  const side = {
    source_kind: "database" as const,
    session_id: sessionId,
    table: "t",
    row_count: rows,
  };
  return {
    source: side,
    target: { ...side, table: "t2", session_id: "other" },
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

describe("compareBlockedReason", () => {
  it("allows comparisons at or below the diff cap", () => {
    expect(compareBlockedReason(MAX_DIFF_ROWS, 50_000)).toBeNull();
    expect(compareBlockedReason(10_000, MAX_DIFF_ROWS)).toBeNull();
  });

  it("blocks comparisons above the diff cap", () => {
    expect(compareBlockedReason(MAX_DIFF_ROWS + 1, 10)).toBe(
      DIFF_TOO_LARGE_MESSAGE,
    );
    expect(compareBlockedReason(10, MAX_DIFF_ROWS + 1)).toBe(
      DIFF_TOO_LARGE_MESSAGE,
    );
  });
});

describe("cross-session DB streaming notice", () => {
  it("returns stable policy copy", () => {
    expect(crossSessionDbStreamingMessage()).toContain("same DB session");
  });

  it("is shown for large cross-session deterministic DB runs", () => {
    const cfg = dbConfig(200_000, "deterministic-fn-ln-bd");
    cfg.streaming.mode = "streaming";
    expect(needsCrossSessionDbStreamingNotice(cfg)).toBe(true);
  });

  it("is hidden for same-session DB runs", () => {
    const cfg = dbConfig(200_000, "deterministic-fn-ln-bd");
    cfg.target.session_id = cfg.source.session_id;
    cfg.streaming.mode = "streaming";
    expect(needsCrossSessionDbStreamingNotice(cfg)).toBe(false);
  });

  it("is hidden when cascade is enabled", () => {
    const cfg = dbConfig(200_000, "deterministic-fn-ln-bd");
    cfg.streaming.mode = "streaming";
    cfg.cascade = {
      enabled: true,
      levels: [1],
      fuzzy_threshold: 80,
      exclusion_mode: "exclusive",
      has_barangay_code: false,
      has_city_code: false,
    };
    expect(needsCrossSessionDbStreamingNotice(cfg)).toBe(false);
  });

  it("is hidden when effective mode is in-memory", () => {
    const cfg = dbConfig(10_000, "deterministic-fn-ln-bd");
    cfg.streaming.mode = "in-memory";
    expect(needsCrossSessionDbStreamingNotice(cfg)).toBe(false);
  });
});
