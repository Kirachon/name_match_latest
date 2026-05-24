import { useEffect, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { open as shellOpen } from "@tauri-apps/plugin-shell";
import {
  explainPair,
  exportResults,
  forgetMatchingJob,
  getMatchingStatus,
  getResultsPage,
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
import { JOB_STATE_TERMINAL } from "@/shared/tauri/types";
import type {
  ExportFormatDto,
  JobSummaryDto,
  MatchPairDto,
  ResultPageDto,
  ScoreBreakdownDto,
} from "@/shared/tauri/types";
import { useDebounced } from "@/shared/hooks";
import { ResultsTable } from "./ResultsTable";
import { ExplanationPanel } from "./ExplanationPanel";

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
  const [selectedRow, setSelectedRow] = useState<MatchPairDto | null>(null);
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
      setPage(null);
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

  function toggleLevel(level: number) {
    const next = levels.includes(level)
      ? levels.filter((value) => value !== level)
      : [...levels, level].sort((a, b) => a - b);
    patchView({ levels: next, pageIndex: 0 });
  }

  async function onExport(format: ExportFormatDto) {
    if (!activeJobId) return;
    let dir = exportCfg.output_directory;
    if (!dir) {
      const picked = await open({ directory: true, multiple: false }).catch(
        () => null,
      );
      if (typeof picked !== "string") return;
      dir = picked;
    }
    try {
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
      pushToast({
        tone: "error",
        title: "Export failed",
        message:
          typeof err === "object" && err && "message" in err
            ? String((err as { message: unknown }).message)
            : String(err),
        ttlMs: null,
      });
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
            <div className="flex items-center gap-2">
              <Button tone="secondary" onClick={() => onExport("csv")}>
                Export CSV
              </Button>
              <Button tone="secondary" onClick={() => onExport("xlsx")}>
                Export XLSX
              </Button>
              <Button tone="primary" onClick={() => onExport("both")}>
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
        <div className="flex">
          <div className="min-w-0 flex-1">
            <ResultsTable
              rows={page?.rows ?? []}
              selectedRowId={selectedRow?.row_id ?? null}
              onSelectRow={setSelectedRow}
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
