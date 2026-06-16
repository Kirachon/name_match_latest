import { useEffect, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { open as shellOpen } from "@tauri-apps/plugin-shell";
import {
  diffJobs,
  explainPair,
  exportResults,
  forgetMatchingJob,
  getDecisions,
  getMatchingStatus,
  getResultsPage,
  listMatchingJobs,
  saveDecision,
} from "@/shared/tauri/commands";
import { useJobStore } from "@/shared/stores/jobStore";
import { useConfigStore } from "@/shared/stores/configStore";
import {
  DEFAULT_RESULTS_VIEW_STATE,
  useResultsStore,
  type ResultsSortDir,
  type ResultsSortKey,
} from "@/shared/stores/resultsStore";
import { useToastStore } from "@/shared/stores/toastStore";
import {
  Button,
  Card,
  Field,
  Pill,
  SectionHeader,
} from "@/shared/components/primitives";
import { formatDuration, formatNumber } from "@/shared/lib/format";
import { inReviewBand } from "@/shared/lib/reviewBand";
import {
  compareBlockedReason,
  LARGE_RESULTS_BANNER_ROWS,
  MAX_DIFF_ROWS,
} from "@/shared/runScalePolicy";
import { JOB_STATE_TERMINAL } from "@/shared/tauri/types";
import type {
  ExportFormatDto,
  DiffResultDto,
  JobSummaryDto,
  MatchPairDto,
  ResultPageDto,
  ReviewDecisionDto,
  ReviewDecisionValue,
  ScoreBreakdownDto,
} from "@/shared/tauri/types";
import { useDebounced } from "@/shared/hooks";
import { ResultsTable } from "./ResultsTable";
import { ExplanationPanel } from "./ExplanationPanel";
import { ReviewToolbar } from "./ReviewToolbar";
import { DiffView } from "./DiffView";

const PAGE_LIMIT = 1000;

export function ResultsTab() {
  const activeJobId = useJobStore((s) => s.activeJobId);
  const setActiveJob = useJobStore((s) => s.setActive);
  const viewState = useResultsStore((s) =>
    activeJobId
      ? (s.jobs[activeJobId] ?? DEFAULT_RESULTS_VIEW_STATE)
      : DEFAULT_RESULTS_VIEW_STATE,
  );
  const patchResultView = useResultsStore((s) => s.patchJob);
  const resetResultView = useResultsStore((s) => s.resetJob);
  const [summary, setSummary] = useState<JobSummaryDto | null>(null);
  const [page, setPage] = useState<ResultPageDto | null>(null);
  const [jobs, setJobs] = useState<JobSummaryDto[]>([]);
  const [baseJobId, setBaseJobId] = useState("");
  const [compareJobId, setCompareJobId] = useState("");
  const [diff, setDiff] = useState<DiffResultDto | null>(null);
  const [diffLoading, setDiffLoading] = useState(false);
  const [compareError, setCompareError] = useState<string | null>(null);
  const [exportingFormat, setExportingFormat] = useState<ExportFormatDto | null>(
    null,
  );
  const [selectedRow, setSelectedRow] = useState<MatchPairDto | null>(null);
  const [decisions, setDecisions] = useState<Record<string, ReviewDecisionDto>>(
    {},
  );
  const [savingRows, setSavingRows] = useState<Set<number>>(() => new Set());
  const [breakdown, setBreakdown] = useState<ScoreBreakdownDto | null>(null);
  const [explainLoading, setExplainLoading] = useState(false);
  const [explainError, setExplainError] = useState<string | null>(null);
  const debouncedSearch = useDebounced(viewState.search, 200);
  const debouncedConf = useDebounced(viewState.minConf, 200);
  const [loading, setLoading] = useState(false);
  const pushToast = useToastStore((s) => s.push);
  const exportCfg = useConfigStore((s) => s.export);
  const { pageIndex, search, sortBy, sortDir, minConf, levels } = viewState;

  function patchView(patch: Partial<typeof viewState>) {
    if (!activeJobId) return;
    patchResultView(activeJobId, patch);
  }

  useEffect(() => {
    if (!activeJobId) return;
    let cancelled = false;
    getMatchingStatus(activeJobId)
      .then((s) => {
        if (!cancelled) setSummary(s);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [activeJobId]);

  useEffect(() => {
    if (!activeJobId) {
      setJobs([]);
      setBaseJobId("");
      setCompareJobId("");
      setDiff(null);
      return;
    }
    let cancelled = false;
    listMatchingJobs()
      .then((items) => {
        if (cancelled) return;
        setJobs(items);
        setBaseJobId((current) => current || activeJobId);
        setCompareJobId((current) => {
          if (current && current !== activeJobId) return current;
          return (
            items.find((item) => item.job_id !== activeJobId)?.job_id ?? ""
          );
        });
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [activeJobId]);

  useEffect(() => {
    if (!activeJobId) {
      setPage(null);
      setDecisions({});
      return;
    }
    let cancelled = false;
    setLoading(true);
    getResultsPage({
      job_id: activeJobId,
      page: pageIndex,
      limit: PAGE_LIMIT,
      query: debouncedSearch || null,
      sort_by: sortBy,
      sort_dir: sortDir,
      min_confidence: debouncedConf > 0 ? debouncedConf : null,
      levels,
    })
      .then((p) => {
        if (!cancelled) {
          setPage(p);
          const maxPage = Math.max(0, Math.ceil(p.total / PAGE_LIMIT) - 1);
          if (pageIndex > maxPage) {
            patchResultView(activeJobId, { pageIndex: maxPage });
          }
        }
      })
      .catch((err: unknown) => {
        pushToast({
          tone: "error",
          title: "Could not load results",
          message:
            typeof err === "object" && err && "message" in err
              ? String((err as { message: unknown }).message)
              : String(err),
        });
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [
    activeJobId,
    pageIndex,
    debouncedSearch,
    sortBy,
    sortDir,
    debouncedConf,
    levels,
    pushToast,
    patchResultView,
  ]);

  useEffect(() => {
    if (!activeJobId) {
      setDecisions({});
      return;
    }
    let cancelled = false;
    getDecisions(activeJobId)
      .then((items) => {
        if (cancelled) return;
        setDecisions(
          Object.fromEntries(items.map((item) => [decisionKey(item), item])),
        );
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        pushToast({
          tone: "error",
          title: "Could not load review decisions",
          message:
            typeof err === "object" && err && "message" in err
              ? String((err as { message: unknown }).message)
              : String(err),
        });
      });
    return () => {
      cancelled = true;
    };
  }, [activeJobId, pushToast]);

  useEffect(() => {
    if (!activeJobId || !selectedRow) {
      setBreakdown(null);
      setExplainError(null);
      return;
    }
    let cancelled = false;
    setExplainLoading(true);
    setExplainError(null);
    explainPair({
      job_id: activeJobId,
      source_id: selectedRow.source_id,
      target_id: selectedRow.target_id,
    })
      .then((result) => {
        if (!cancelled) setBreakdown(result);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setExplainError(
          typeof err === "object" && err && "message" in err
            ? String((err as { message: unknown }).message)
            : String(err),
        );
      })
      .finally(() => {
        if (!cancelled) setExplainLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [activeJobId, selectedRow]);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (!activeJobId || isTypingTarget(event.target)) return;
      if (event.key === "ArrowDown") {
        event.preventDefault();
        onNextPending();
        return;
      }
      if (!selectedRow || savingRows.has(selectedRow.row_id)) return;
      if (!inReviewBand(selectedRow.confidence, exportCfg.review_band)) return;
      if (event.key.toLowerCase() === "a") {
        event.preventDefault();
        void onDecision(selectedRow, "accepted");
      } else if (event.key.toLowerCase() === "r") {
        event.preventDefault();
        void onDecision(selectedRow, "rejected");
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activeJobId, selectedRow, savingRows, decisions, page, exportCfg.review_band]);

  if (!activeJobId) {
    return (
      <Card className="dot-grid">
        <div className="text-center py-10 text-ink-300">
          No completed run yet. Start a run from the{" "}
          <span className="text-ink-100 font-medium">Run</span> tab — results
          appear here automatically.
        </div>
      </Card>
    );
  }

  const total = page?.total ?? 0;
  const lastPage = Math.max(0, Math.ceil(total / PAGE_LIMIT) - 1);
  const availableLevels = page?.available_levels ?? [];
  const levelCounts = page?.level_counts ?? {};
  const pageRows = page?.rows ?? [];
  const reviewCounts = pageRows.reduce(
    (acc, row) => {
      const decision = decisions[decisionKey(row)]?.decision;
      if (decision === "accepted") acc.accepted += 1;
      else if (decision === "rejected") acc.rejected += 1;
      else if (inReviewBand(row.confidence, exportCfg.review_band)) {
        acc.pending += 1;
      }
      return acc;
    },
    { accepted: 0, rejected: 0, pending: 0 },
  );
  const decisionValues = Object.fromEntries(
    Object.entries(decisions).map(([key, value]) => [key, value.decision]),
  ) as Record<string, ReviewDecisionValue>;

  const baseJob = jobs.find((job) => job.job_id === baseJobId);
  const compareJob = jobs.find((job) => job.job_id === compareJobId);
  const compareBlocked = compareBlockedReason(
    baseJob?.matches_found ?? 0,
    compareJob?.matches_found ?? 0,
  );
  const compareStatusMessage = compareError ?? compareBlocked;
  const compareStatusTone = compareError ? "error" : "warn";

  function toggleLevel(level: number) {
    const next = levels.includes(level)
      ? levels.filter((value) => value !== level)
      : [...levels, level].sort((a, b) => a - b);
    patchView({ levels: next, pageIndex: 0 });
  }

  async function onDecision(row: MatchPairDto, decision: ReviewDecisionValue) {
    if (!activeJobId) return;
    if (!inReviewBand(row.confidence, exportCfg.review_band)) return;
    setSavingRows((current) => new Set(current).add(row.row_id));
    try {
      const saved = await saveDecision({
        job_id: activeJobId,
        row_id: row.row_id,
        source_id: row.source_id,
        target_id: row.target_id,
        decision,
      });
      setDecisions((current) => ({ ...current, [decisionKey(saved)]: saved }));
    } catch (err: unknown) {
      pushToast({
        tone: "error",
        title: "Could not save review decision",
        message:
          typeof err === "object" && err && "message" in err
            ? String((err as { message: unknown }).message)
            : String(err),
      });
    } finally {
      setSavingRows((current) => {
        const next = new Set(current);
        next.delete(row.row_id);
        return next;
      });
    }
  }

  function onNextPending() {
    const startIndex = selectedRow
      ? pageRows.findIndex((row) => row.row_id === selectedRow.row_id) + 1
      : 0;
    const orderedRows = [
      ...pageRows.slice(startIndex),
      ...pageRows.slice(0, startIndex),
    ];
    const next = orderedRows.find((row) => {
      const decision = decisions[decisionKey(row)]?.decision;
      return (
        decision !== "accepted" &&
        decision !== "rejected" &&
        inReviewBand(row.confidence, exportCfg.review_band)
      );
    });
    if (next) setSelectedRow(next);
  }

  async function onCompareJobs() {
    if (!baseJobId || !compareJobId || baseJobId === compareJobId) return;
    if (compareBlocked) {
      setCompareError(compareBlocked);
      return;
    }
    setDiffLoading(true);
    setCompareError(null);
    try {
      const result = await diffJobs({
        base_job_id: baseJobId,
        compare_job_id: compareJobId,
      });
      setDiff(result);
    } catch (err: unknown) {
      const message =
        typeof err === "object" && err && "message" in err
          ? String((err as { message: unknown }).message)
          : String(err);
      setCompareError(message);
      pushToast({
        tone: "error",
        title: "Could not compare runs",
        message,
      });
    } finally {
      setDiffLoading(false);
    }
  }

  async function onExport(format: ExportFormatDto) {
    if (!activeJobId || exportingFormat) return;
    let dir = exportCfg.output_directory;
    if (!dir) {
      const picked = await open({ directory: true, multiple: false }).catch(
        () => null,
      );
      if (typeof picked !== "string") return;
      dir = picked;
    }
    try {
      setExportingFormat(format);
      const r = await exportResults({
        job_id: activeJobId,
        format,
        output_directory: dir,
        file_stem: exportCfg.file_stem || "matches",
        min_confidence: minConf > 0 ? minConf : null,
        levels,
        include_extra_fields: exportCfg.include_extra_fields,
      });
      pushToast({
        tone: "success",
        title: `Exported ${formatNumber(r.rows_exported)} rows`,
        message: r.written_paths.join("\n"),
      });
    } catch (err: unknown) {
      const message =
        typeof err === "object" && err && "message" in err
          ? String((err as { message: unknown }).message)
          : String(err);
      pushToast({
        tone: "error",
        title: "Export failed",
        message,
        ttlMs: null,
      });
    } finally {
      setExportingFormat(null);
    }
  }

  async function onOpenFolder() {
    if (!exportCfg.output_directory) return;
    try {
      await shellOpen(exportCfg.output_directory);
    } catch (err: unknown) {
      pushToast({
        tone: "warn",
        title: "Could not open folder",
        message:
          typeof err === "object" && err && "message" in err
            ? String((err as { message: unknown }).message)
            : String(err),
      });
    }
  }

  async function onForgetJob() {
    if (
      !activeJobId ||
      !summary ||
      !JOB_STATE_TERMINAL.includes(summary.state)
    ) {
      return;
    }
    try {
      await forgetMatchingJob(activeJobId);
      resetResultView(activeJobId);
      setActiveJob(null);
      setSummary(null);
      setPage(null);
      pushToast({
        tone: "success",
        title: "Run removed",
        message: "The completed run was removed from local history.",
      });
    } catch (err: unknown) {
      pushToast({
        tone: "error",
        title: "Could not remove run",
        message:
          typeof err === "object" && err && "message" in err
            ? String((err as { message: unknown }).message)
            : String(err),
      });
    }
  }

  return (
    <div className="space-y-5">
      <Card>
        <SectionHeader
          title="Results"
          description={
            summary
              ? `Job ${summary.job_id.slice(0, 8)} · ${summary.source_table} → ${summary.target_table} · ${formatNumber(summary.matches_found)} matches in ${formatDuration(summary.elapsed_secs)}`
              : "Loading run summary…"
          }
          action={
            <div className="flex items-center gap-2 flex-wrap justify-end">
              <Button
                tone="secondary"
                onClick={() => onExport("csv")}
                loading={exportingFormat === "csv"}
                disabled={!!exportingFormat}
              >
                Export CSV
              </Button>
              <Button
                tone="secondary"
                onClick={() => onExport("xlsx")}
                loading={exportingFormat === "xlsx"}
                disabled={!!exportingFormat}
              >
                Export XLSX
              </Button>
              <Button
                tone="primary"
                onClick={() => onExport("both")}
                loading={exportingFormat === "both"}
                disabled={!!exportingFormat}
              >
                Export both
              </Button>
              <Button
                tone="ghost"
                onClick={onOpenFolder}
                disabled={!exportCfg.output_directory}
              >
                Open folder
              </Button>
              <Button
                tone="ghost"
                onClick={onForgetJob}
                disabled={
                  !summary || !JOB_STATE_TERMINAL.includes(summary.state)
                }
              >
                Forget
              </Button>
            </div>
          }
        />
        {(page?.total ?? summary?.matches_found ?? 0) >=
          LARGE_RESULTS_BANNER_ROWS && (
          <div
            className="mb-4 rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm text-amber-100"
            role="status"
          >
            Large result set (
            {formatNumber(page?.total ?? summary?.matches_found ?? 0)} rows).
            Paging is for review only — use Export above for the full dataset.
            Match explanations may be unavailable when person snapshots were
            trimmed.
          </div>
        )}
        <div className="grid lg:grid-cols-4 gap-3">
          <Field label="Search">
            <input
              className="input"
              placeholder="filter source / target full name…"
              value={search}
              onChange={(e) => {
                patchView({ search: e.target.value, pageIndex: 0 });
              }}
            />
          </Field>
          <Field label="Min confidence">
            <input
              type="number"
              className="input tabular"
              min={0}
              max={100}
              step={1}
              value={minConf}
              onChange={(e) => {
                patchView({
                  minConf: Number(e.target.value || 0),
                  pageIndex: 0,
                });
              }}
            />
          </Field>
          <Field label="Sort by">
            <select
              className="select"
              value={sortBy}
              onChange={(e) => {
                patchView({
                  sortBy: e.target.value as ResultsSortKey,
                  pageIndex: 0,
                });
              }}
            >
              <option value="confidence">Confidence</option>
              <option value="source_name">Source name</option>
              <option value="target_name">Target name</option>
              <option value="row_id">Row ID</option>
            </select>
          </Field>
          <Field label="Direction">
            <select
              className="select"
              value={sortDir}
              onChange={(e) => {
                patchView({
                  sortDir: e.target.value as ResultsSortDir,
                  pageIndex: 0,
                });
              }}
            >
              <option value="desc">Descending</option>
              <option value="asc">Ascending</option>
            </select>
          </Field>
          <label className="flex items-start gap-3 rounded-lg border border-white/10 bg-white/5 px-3 py-2 lg:col-span-2 cursor-pointer">
            <input
              type="checkbox"
              className="mt-1 h-4 w-4 accent-primary-400"
              checked={exportCfg.include_extra_fields}
              onChange={(e) =>
                useConfigStore
                  .getState()
                  .setExport({ include_extra_fields: e.target.checked })
              }
            />
            <span className="min-w-0">
              <span className="block text-sm text-ink-100">
                Include all extra fields on export
              </span>
              <span className="block text-xs text-ink-400">
                Adds every non-standard source and target table column to
                CSV/XLSX.
              </span>
            </span>
          </label>
        </div>
        {availableLevels.length > 0 && (
          <div className="mt-4 flex flex-wrap items-center gap-2">
            <Button
              tone={levels.length === 0 ? "primary" : "ghost"}
              size="sm"
              onClick={() => patchView({ levels: [], pageIndex: 0 })}
            >
              All
            </Button>
            {availableLevels.map((level) => {
              const active = levels.includes(level);
              return (
                <Button
                  key={level}
                  tone={active ? "primary" : "ghost"}
                  size="sm"
                  onClick={() => toggleLevel(level)}
                >
                  L{String(level).padStart(2, "0")}{" "}
                  {formatNumber(levelCounts[String(level)] ?? 0)}
                </Button>
              );
            })}
          </div>
        )}
        {jobs.length >= 2 && (
          <div className="mt-4 grid lg:grid-cols-[1fr_1fr_auto] gap-3 border-t border-white/10 pt-4">
            <Field label="Base run">
              <select
                className="select"
                value={baseJobId}
                onChange={(e) => {
                  setBaseJobId(e.target.value);
                  setDiff(null);
                  setCompareError(null);
                }}
              >
                {jobs.map((job) => (
                  <option key={job.job_id} value={job.job_id}>
                    {job.job_id.slice(0, 8)} · {job.source_table} →{" "}
                    {job.target_table}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Compare run">
              <select
                className="select"
                value={compareJobId}
                onChange={(e) => {
                  setCompareJobId(e.target.value);
                  setDiff(null);
                  setCompareError(null);
                }}
              >
                {jobs
                  .filter((job) => job.job_id !== baseJobId)
                  .map((job) => (
                    <option key={job.job_id} value={job.job_id}>
                      {job.job_id.slice(0, 8)} · {job.source_table} →{" "}
                      {job.target_table}
                    </option>
                  ))}
              </select>
            </Field>
            <div className="flex items-end">
              <Button
                tone="secondary"
                onClick={onCompareJobs}
                loading={diffLoading}
                disabled={
                  !compareJobId ||
                  baseJobId === compareJobId ||
                  !!compareBlocked
                }
                title={
                  compareBlocked ??
                  (baseJobId === compareJobId
                    ? "Choose two different runs"
                    : `Compare runs with up to ${formatNumber(MAX_DIFF_ROWS)} matches each`)
                }
              >
                Compare
              </Button>
            </div>
            {compareStatusMessage && (
              <div
                className={
                  compareStatusTone === "error"
                    ? "lg:col-span-3 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-100"
                    : "lg:col-span-3 rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm text-amber-100"
                }
                role="status"
              >
                {compareStatusMessage}
              </div>
            )}
          </div>
        )}
        {diff && <DiffView diff={diff} onClose={() => setDiff(null)} />}
      </Card>

      <Card padded={false}>
        <div className="flex items-center justify-between p-4 pb-2">
          <div className="flex items-center gap-3">
            <Pill tone="info">{formatNumber(total)} rows</Pill>
            {loading && <Pill tone="mute">Loading…</Pill>}
          </div>
          <div className="flex items-center gap-2 text-xs text-ink-300 tabular">
            <Button
              tone="ghost"
              size="sm"
              onClick={() =>
                patchView({ pageIndex: Math.max(0, pageIndex - 1) })
              }
              disabled={pageIndex === 0}
            >
              ← Prev
            </Button>
            <span>
              Page {pageIndex + 1} / {lastPage + 1}
            </span>
            <Button
              tone="ghost"
              size="sm"
              onClick={() =>
                patchView({ pageIndex: Math.min(lastPage, pageIndex + 1) })
              }
              disabled={pageIndex >= lastPage}
            >
              Next →
            </Button>
          </div>
        </div>
        <ReviewToolbar
          total={pageRows.length}
          accepted={reviewCounts.accepted}
          rejected={reviewCounts.rejected}
          pending={reviewCounts.pending}
          disabled={loading}
          onNextPending={onNextPending}
        />
        <div className="flex">
          <div className="min-w-0 flex-1">
            <ResultsTable
              rows={pageRows}
              selectedRowId={selectedRow?.row_id ?? null}
              decisions={decisionValues}
              savingRowIds={savingRows}
              reviewBand={exportCfg.review_band}
              onSelectRow={setSelectedRow}
              onDecision={onDecision}
            />
          </div>
          {selectedRow && (
            <ExplanationPanel
              row={selectedRow}
              breakdown={breakdown}
              loading={explainLoading}
              error={explainError}
              onClose={() => setSelectedRow(null)}
            />
          )}
        </div>
      </Card>
    </div>
  );
}

function decisionKey(row: { source_id: number; target_id: number }) {
  return `${row.source_id}:${row.target_id}`;
}

function isTypingTarget(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName.toLowerCase();
  return (
    target.isContentEditable ||
    tag === "input" ||
    tag === "textarea" ||
    tag === "select"
  );
}
