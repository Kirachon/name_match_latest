import type {
  AlgorithmDto,
  RunConfigDto,
  RunModeDto,
  TableSelectionDto,
} from "@/shared/tauri/types";

export const SCALE_WARN_ROWS = 100_000;
export const SCALE_STRONG_WARN_ROWS = 500_000;
export const SCALE_BLOCK_ROWS = 1_000_000;
export const LARGE_RESULTS_BANNER_ROWS = 100_000;
export const LARGE_RESULTS_DEFAULT_PAGE_SIZE = 50;
export const MAX_DIFF_ROWS = 100_000;
export const MAX_XLSX_EXPORT_ROWS = 100_000;

export const DIFF_TOO_LARGE_MESSAGE =
  "This comparison is too large to load safely in memory. Export both runs as CSV and compare externally, or rerun with narrower filters.";

export type EffectiveRunMode = "in-memory" | "streaming";

export type ScaleBlockReason =
  | "million-row-file-source"
  | "million-row-cascade"
  | "million-row-fuzzy"
  | "million-row-unsupported-algorithm";

export function rowCountForSide(selection: TableSelectionDto): number {
  return selection.row_count ?? 0;
}

export function maxSideRows(
  config: Pick<RunConfigDto, "source" | "target">,
): number {
  return Math.max(
    rowCountForSide(config.source),
    rowCountForSide(config.target),
  );
}

export function resolveEffectiveRunMode(
  requested: RunModeDto,
  sourceRows: number,
  targetRows: number,
): EffectiveRunMode {
  const maxRows = Math.max(sourceRows, targetRows);
  if (requested === "in-memory") return "in-memory";
  if (requested === "streaming") return "streaming";
  return maxRows >= SCALE_WARN_ROWS ? "streaming" : "in-memory";
}

export function algorithmSupportsDbStreaming(algorithm: AlgorithmDto): boolean {
  return (
    algorithm === "deterministic-fn-ln-bd" ||
    algorithm === "deterministic-fn-mn-ln-bd"
  );
}

export function sameDbSession(
  config: Pick<RunConfigDto, "source" | "target">,
): boolean {
  return config.source.session_id === config.target.session_id;
}

export function scaleBlockReason(
  config: Pick<RunConfigDto, "source" | "target" | "algorithm" | "cascade">,
): ScaleBlockReason | null {
  const maxRows = maxSideRows(config);
  if (maxRows < SCALE_BLOCK_ROWS) return null;
  if (
    config.source.source_kind === "file" ||
    config.target.source_kind === "file"
  ) {
    return "million-row-file-source";
  }
  if (config.cascade?.enabled) return "million-row-cascade";
  if (!algorithmSupportsDbStreaming(config.algorithm)) {
    if (
      config.algorithm === "fuzzy" ||
      config.algorithm === "fuzzy-no-middle"
    ) {
      return "million-row-fuzzy";
    }
    return "million-row-unsupported-algorithm";
  }
  return null;
}

export function scaleWarningLevel(maxRows: number): "none" | "warn" | "strong" {
  if (maxRows >= SCALE_STRONG_WARN_ROWS) return "strong";
  if (maxRows >= SCALE_WARN_ROWS) return "warn";
  return "none";
}

export function scaleBlockMessage(reason: ScaleBlockReason): string {
  switch (reason) {
    case "million-row-file-source":
      return "At 1M+ rows, import CSV to MySQL first. Direct file matching is blocked.";
    case "million-row-cascade":
      return "Deep Match is in-memory only and cannot run at 1M+ rows.";
    case "million-row-fuzzy":
      return "Fuzzy matching at 1M+ rows requires a smaller dataset or deterministic algorithms.";
    case "million-row-unsupported-algorithm":
      return "This algorithm is not supported for million-row database streaming runs.";
  }
}

type DbStreamingConfig = Pick<
  RunConfigDto,
  "source" | "target" | "streaming" | "algorithm" | "cascade"
>;

function dbStreamingEligibleIgnoringSession(
  config: DbStreamingConfig,
): boolean {
  if (config.cascade?.enabled) return false;
  if (
    config.source.source_kind !== "database" ||
    config.target.source_kind !== "database"
  ) {
    return false;
  }
  if (!algorithmSupportsDbStreaming(config.algorithm)) return false;
  const effective = resolveEffectiveRunMode(
    config.streaming.mode,
    rowCountForSide(config.source),
    rowCountForSide(config.target),
  );
  return effective === "streaming";
}

export function streamingBackendActive(config: DbStreamingConfig): boolean {
  return dbStreamingEligibleIgnoringSession(config) && sameDbSession(config);
}

export function compareBlockedReason(
  baseMatches: number,
  compareMatches: number,
): string | null {
  if (baseMatches > MAX_DIFF_ROWS || compareMatches > MAX_DIFF_ROWS) {
    return DIFF_TOO_LARGE_MESSAGE;
  }
  return null;
}

export function crossSessionDbStreamingMessage(): string {
  return "Large database streaming currently requires source and target tables from the same DB session. Reconnect both tables through one session or run a smaller job.";
}

export function needsCrossSessionDbStreamingNotice(
  config: DbStreamingConfig,
): boolean {
  return (
    dbStreamingEligibleIgnoringSession(config) && !sameDbSession(config)
  );
}
