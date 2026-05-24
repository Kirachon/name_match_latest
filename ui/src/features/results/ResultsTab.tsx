import { useEffect, useRef, useState } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { open } from "@tauri-apps/plugin-dialog";
import { open as shellOpen } from "@tauri-apps/plugin-shell";
import {
  exportResults,
  getMatchingStatus,
  getResultsPage,
} from "@/shared/tauri/commands";
import { useJobStore } from "@/shared/stores/jobStore";
import { useConfigStore } from "@/shared/stores/configStore";
import { useToastStore } from "@/shared/stores/toastStore";
import {
  Button,
  Card,
  Field,
  Pill,
  SectionHeader,
} from "@/shared/components/primitives";
import {
  cx,
  formatDuration,
  formatNumber,
  formatPercent,
} from "@/shared/lib/format";
import type {
  ExportFormatDto,
  JobSummaryDto,
  MatchPairDto,
  ResultPageDto,
} from "@/shared/tauri/types";
import { useDebounced } from "@/shared/hooks";

const PAGE_LIMIT = 1000;

export function ResultsTab() {
  const activeJobId = useJobStore((s) => s.activeJobId);
  const [summary, setSummary] = useState<JobSummaryDto | null>(null);
  const [page, setPage] = useState<ResultPageDto | null>(null);
  const [pageIndex, setPageIndex] = useState(0);
  const [search, setSearch] = useState("");
  const debouncedSearch = useDebounced(search, 200);
  const [sortBy, setSortBy] = useState<
    "row_id" | "confidence" | "source_name" | "target_name"
  >("confidence");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [minConf, setMinConf] = useState<number>(0);
  const debouncedConf = useDebounced(minConf, 200);
  const [loading, setLoading] = useState(false);
  const pushToast = useToastStore((s) => s.push);
  const exportCfg = useConfigStore((s) => s.export);

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
    })
      .then((p) => {
        if (!cancelled) setPage(p);
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
  }, [activeJobId, pageIndex, debouncedSearch, sortBy, sortDir, debouncedConf, pushToast]);

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

  async function onExport(format: ExportFormatDto) {
    if (!activeJobId) return;
    let dir = exportCfg.output_directory;
    if (!dir) {
      const picked = await open({ directory: true, multiple: false }).catch(() => null);
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
                setSearch(e.target.value);
                setPageIndex(0);
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
                setMinConf(Number(e.target.value || 0));
                setPageIndex(0);
              }}
            />
          </Field>
          <Field label="Sort by">
            <select
              className="select"
              value={sortBy}
              onChange={(e) => {
                setSortBy(e.target.value as typeof sortBy);
                setPageIndex(0);
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
                setSortDir(e.target.value as typeof sortDir);
                setPageIndex(0);
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
                Adds every non-standard source and target table column to CSV/XLSX.
              </span>
            </span>
          </label>
        </div>
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
              onClick={() => setPageIndex(Math.max(0, pageIndex - 1))}
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
              onClick={() => setPageIndex(Math.min(lastPage, pageIndex + 1))}
              disabled={pageIndex >= lastPage}
            >
              Next →
            </Button>
          </div>
        </div>
        <ResultsTable rows={page?.rows ?? []} />
      </Card>
    </div>
  );
}

const COL_TEMPLATE = "70px 90px 1fr 110px 90px 1fr 110px 90px 130px 90px";

function ResultsTable({ rows }: { rows: MatchPairDto[] }) {
  const parentRef = useRef<HTMLDivElement>(null);
  const rowVirtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 32,
    overscan: 12,
  });

  return (
    <div
      ref={parentRef}
      className="h-[520px] overflow-auto border-t border-ink-800/70"
      style={{ ["--data-cols" as string]: COL_TEMPLATE }}
      role="table"
      aria-label="Match results"
    >
      <div className="data-row data-header h-9" role="row">
        <div role="columnheader" className="font-mono">#</div>
        <div role="columnheader">Source ID</div>
        <div role="columnheader">Source name</div>
        <div role="columnheader">Source DOB</div>
        <div role="columnheader">Target ID</div>
        <div role="columnheader">Target name</div>
        <div role="columnheader">Target DOB</div>
        <div role="columnheader" className="text-right">
          Confidence
        </div>
        <div role="columnheader">Level / method</div>
        <div role="columnheader">Fields</div>
      </div>
      <div
        style={{
          height: `${rowVirtualizer.getTotalSize()}px`,
          width: "100%",
          position: "relative",
        }}
      >
        {rowVirtualizer.getVirtualItems().map((vi) => {
          const r = rows[vi.index];
          if (!r) return null;
          return (
            <div
              key={r.row_id}
              role="row"
              className="data-row"
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                right: 0,
                transform: `translateY(${vi.start}px)`,
                height: `${vi.size}px`,
              }}
            >
              <div role="cell" className="font-mono text-ink-500 tabular">
                {r.row_id}
              </div>
              <div role="cell" className="tabular text-ink-400">
                {r.source_id}
              </div>
              <div role="cell" className="truncate" title={r.source_full_name}>
                {r.source_full_name}
              </div>
              <div role="cell" className="tabular text-ink-400">
                {r.source_birthdate ?? "—"}
              </div>
              <div role="cell" className="tabular text-ink-400">
                {r.target_id}
              </div>
              <div role="cell" className="truncate" title={r.target_full_name}>
                {r.target_full_name}
              </div>
              <div role="cell" className="tabular text-ink-400">
                {r.target_birthdate ?? "—"}
              </div>
              <div role="cell" className="text-right tabular">
                <ConfidencePill value={r.confidence} />
              </div>
              <div
                role="cell"
                className="text-2xs text-ink-300 truncate"
                title={r.match_method ?? undefined}
              >
                {r.matched_at_level
                  ? `L${String(r.matched_at_level).padStart(2, "0")}${
                      r.match_method ? ` · ${r.match_method.replace(/^L\d+\s*-\s*/, "")}` : ""
                    }`
                  : "—"}
              </div>
              <div role="cell" className="text-2xs text-ink-400 truncate" title={r.matched_fields.join(", ")}>
                {r.matched_fields.join(", ")}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ConfidencePill({ value }: { value: number }) {
  const tone =
    value >= 95 ? "ok" : value >= 85 ? "info" : value >= 70 ? "warn" : "danger";
  return (
    <span
      className={cx(
        "tabular px-1.5 py-0.5 rounded text-xs",
        tone === "ok" && "bg-ok-500/15 text-ok-400",
        tone === "info" && "bg-accent-500/15 text-accent-400",
        tone === "warn" && "bg-warn-500/15 text-warn-400",
        tone === "danger" && "bg-danger-500/15 text-danger-400",
      )}
    >
      {formatPercent(value)}
    </span>
  );
}
