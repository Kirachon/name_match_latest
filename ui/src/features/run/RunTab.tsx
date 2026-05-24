import { useEffect, useMemo, useRef } from "react";
import {
  cancelMatching,
  pauseMatching,
  resumeMatching,
  startMatching,
} from "@/shared/tauri/commands";
import {
  useJobStore,
  useLogStore,
  useProgressStore,
} from "@/shared/stores/jobStore";
import { useConfigStore } from "@/shared/stores/configStore";
import {
  useConnectionStore,
  readinessForRun,
} from "@/shared/stores/connectionStore";
import { useToastStore } from "@/shared/stores/toastStore";
import {
  Button,
  Card,
  Pill,
  SectionHeader,
  StatusDot,
} from "@/shared/components/primitives";
import { parseRunConfig } from "@/shared/tauri/zod-schemas";
import {
  cx,
  formatDuration,
  formatNumber,
  formatPercent,
  formatTimestamp,
} from "@/shared/lib/format";
import type {
  JobStateDto,
  LogEntryDto,
  PipelineStageDto,
  TableSelectionDto,
} from "@/shared/tauri/types";

const STAGES: Array<{ id: PipelineStageDto; label: string }> = [
  { id: "load", label: "Load" },
  { id: "hash", label: "Hash" },
  { id: "match", label: "Match" },
  { id: "fuzzy", label: "Fuzzy" },
  { id: "export", label: "Export" },
];

export function RunTab({ onComplete }: { onComplete: () => void }) {
  const jobState = useJobStore((s) => s.state);
  const detail = useJobStore((s) => s.detail);
  const activeJobId = useJobStore((s) => s.activeJobId);
  const setActive = useJobStore((s) => s.setActive);
  const progress = useProgressStore();
  const pushToast = useToastStore((s) => s.push);

  // Auto-advance to Results when run completes successfully.
  useEffect(() => {
    if (jobState === "completed") {
      const t = window.setTimeout(() => onComplete(), 800);
      return () => window.clearTimeout(t);
    }
  }, [jobState, onComplete]);

  async function onStart() {
    const ready = readinessForRun(useConnectionStore.getState());
    if (!ready.ready) {
      pushToast({
        tone: "warn",
        title: "Not ready",
        message: ready.reason ?? "Open Connect tab",
      });
      return;
    }
    const cfg = useConfigStore.getState();
    const conn = useConnectionStore.getState();
    const srcRaw = conn.source.columns?.raw_columns ?? [];
    const tgtRaw = conn.target.columns?.raw_columns ?? [];
    const draft = cfg.buildRunConfig(
      selectionFromSide(conn.source),
      selectionFromSide(conn.target),
      {
        hasBarangay:
          srcRaw.includes("barangay_code") && tgtRaw.includes("barangay_code"),
        hasCity: srcRaw.includes("city_code") && tgtRaw.includes("city_code"),
      },
    );
    const parsed = parseRunConfig(draft);
    if (!parsed.ok) {
      pushToast({
        tone: "warn",
        title: "Configuration is invalid",
        message: parsed.issues.slice(0, 3).join("\n"),
      });
      return;
    }
    try {
      const id = await startMatching(parsed.value);
      setActive(id);
      pushToast({
        tone: "info",
        title: "Run started",
        message: `Job ${id.slice(0, 8)}`,
        ttlMs: 1500,
      });
    } catch (err: unknown) {
      pushToast({
        tone: "error",
        title: "Failed to start",
        message:
          typeof err === "object" && err && "message" in err
            ? String((err as { message: unknown }).message)
            : String(err),
        ttlMs: null,
      });
    }
  }

  async function onCancel() {
    if (!activeJobId) return;
    // Cancel-from-paused warrants a confirmation since partial results exist.
    if (jobState === "paused") {
      const ok = window.confirm(
        `Cancel matching? ${progress.matchesFound.toLocaleString()} pairs already matched will be kept in results.`,
      );
      if (!ok) return;
    }
    await cancelMatching(activeJobId).catch((err: unknown) => {
      pushToast({
        tone: "error",
        title: "Cancel failed",
        message:
          typeof err === "object" && err && "message" in err
            ? String((err as { message: unknown }).message)
            : String(err),
      });
    });
  }

  async function onPause() {
    if (!activeJobId) return;
    await pauseMatching(activeJobId).catch((err: unknown) => {
      pushToast({
        tone: "error",
        title: "Pause failed",
        message:
          typeof err === "object" && err && "message" in err
            ? String((err as { message: unknown }).message)
            : String(err),
      });
    });
  }

  async function onResume() {
    if (!activeJobId) return;
    await resumeMatching(activeJobId).catch((err: unknown) => {
      pushToast({
        tone: "error",
        title: "Resume failed",
        message:
          typeof err === "object" && err && "message" in err
            ? String((err as { message: unknown }).message)
            : String(err),
      });
    });
  }

  const isRunning =
    jobState === "starting" ||
    jobState === "validating" ||
    jobState === "running" ||
    jobState === "cancelling";
  const isPausing = jobState === "pausing";
  const isPaused = jobState === "paused";
  const isResuming = jobState === "resuming";
  const isActive = isRunning || isPausing || isPaused || isResuming;
  const isTerminal =
    jobState === "completed" ||
    jobState === "failed" ||
    jobState === "cancelled";

  // Expose pause/resume to the global Ctrl+P shortcut handler via a stable
  // ref on `window` (set once per render). Cheap and avoids a context.
  useEffect(() => {
    const w = window as unknown as {
      __nmRunControls?: {
        pause: () => void;
        resume: () => void;
        state: string;
      };
    };
    w.__nmRunControls = { pause: onPause, resume: onResume, state: jobState };
    return () => {
      delete w.__nmRunControls;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeJobId, jobState]);

  return (
    <div className="grid xl:grid-cols-3 gap-5">
      <div className="xl:col-span-2 space-y-5">
        <Card>
          <SectionHeader
            title="Run"
            description={
              isPaused
                ? "Paused — no batches will start until you resume. Cancel still works."
                : isPausing
                  ? "Pausing after the current batch finishes…"
                  : "Cancellation is cooperative — the engine stops at the next safe boundary."
            }
            action={
              <div className="flex items-center gap-2">
                {!isActive && (
                  <Button tone="primary" onClick={onStart}>
                    {isTerminal ? "Run again" : "Start matching"}
                  </Button>
                )}
                {isRunning && (
                  <>
                    <Button
                      tone="secondary"
                      onClick={onPause}
                      title="Pause (Ctrl+P)"
                      leadingIcon={<PauseIcon />}
                    >
                      Pause
                    </Button>
                    <Button tone="danger" onClick={onCancel}>
                      Cancel
                    </Button>
                  </>
                )}
                {isPausing && (
                  <>
                    <Button tone="secondary" disabled loading>
                      Pausing…
                    </Button>
                    <Button tone="danger" onClick={onCancel}>
                      Cancel
                    </Button>
                  </>
                )}
                {isPaused && (
                  <>
                    <Button
                      tone="primary"
                      onClick={onResume}
                      title="Resume (Ctrl+P)"
                      leadingIcon={<PlayIcon />}
                    >
                      Resume
                    </Button>
                    <Button tone="danger" onClick={onCancel}>
                      Cancel
                    </Button>
                  </>
                )}
                {isResuming && (
                  <Button tone="primary" disabled loading>
                    Resuming…
                  </Button>
                )}
              </div>
            }
          />

          <PipelineRow stage={progress.stage} state={jobState} />

          <div className="mt-4 space-y-3">
            <div className="flex items-baseline justify-between">
              <div className="flex items-baseline gap-3">
                <span
                  className={cx(
                    "text-3xl font-semibold tabular",
                    isPaused || isPausing ? "text-warn-400" : "text-ink-50",
                  )}
                >
                  {formatPercent(progress.percent || 0)}
                </span>
                <span className="text-sm text-ink-400 tabular">
                  {formatNumber(progress.processed)} /{" "}
                  {formatNumber(progress.total || 0)}
                </span>
                {isPaused && (
                  <span className="text-xs text-warn-400 uppercase tracking-wider ml-2">
                    Paused
                  </span>
                )}
              </div>
              <div
                className="text-right"
                role="status"
                aria-live="polite"
                aria-atomic="true"
              >
                <div className="text-2xs uppercase tracking-wider text-ink-500">
                  ETA
                </div>
                <div className="text-sm tabular text-ink-100">
                  {progress.etaSecs > 0
                    ? formatDuration(progress.etaSecs)
                    : "—"}
                </div>
              </div>
            </div>
            <div className="progress-track">
              {progress.percent > 0 ? (
                <div
                  className={cx(
                    "progress-bar",
                    isPaused &&
                      "!bg-gradient-to-r !from-warn-500 !to-warn-400 !animate-none",
                    isPausing &&
                      "!bg-gradient-to-r !from-warn-500 !to-warn-400 !animate-pulse-soft",
                  )}
                  style={{ width: `${Math.min(100, progress.percent)}%` }}
                />
              ) : isRunning ? (
                <div className="progress-bar-indeterminate" />
              ) : null}
            </div>
            {detail && jobState === "failed" && (
              <div className="help-error">{detail}</div>
            )}
          </div>
        </Card>

        <LogConsole />
      </div>

      <div className="xl:col-span-1 space-y-5">
        <MetricsCard />
        <GpuCard />
      </div>
    </div>
  );
}

function selectionFromSide(
  side: ReturnType<typeof useConnectionStore.getState>["source"],
): TableSelectionDto {
  if (side.mode === "file") {
    return {
      source_kind: "file",
      session_id: "",
      table: "",
      column_mapping: side.columnMapping,
      file: {
        path: side.file.path,
        sheet_name: side.file.sheetName,
        encoding: side.file.encoding,
        delimiter: side.file.delimiter,
        date_format: side.file.dateFormat,
      },
    };
  }
  return {
    source_kind: "database",
    session_id: side.session!.session_id,
    table: side.selectedTable!,
    column_mapping: side.columnMapping,
  };
}

function PipelineRow({
  stage,
  state,
}: {
  stage: PipelineStageDto;
  state: JobStateDto;
}) {
  const idx = STAGES.findIndex((s) => s.id === stage);
  return (
    <div className="flex items-center gap-1.5 flex-wrap mt-2">
      {STAGES.map((s, i) => {
        const isActive = s.id === stage;
        const isDone = idx > i;
        return (
          <span
            key={s.id}
            className="chip"
            data-active={isActive ? "true" : "false"}
            data-done={isDone ? "true" : "false"}
          >
            <StatusDot
              tone={isActive ? "info" : isDone ? "ok" : "mute"}
              pulse={isActive && state !== "idle"}
            />
            {s.label}
          </span>
        );
      })}
    </div>
  );
}

function MetricsCard() {
  const p = useProgressStore();
  return (
    <Card>
      <SectionHeader title="Metrics" />
      <div className="grid grid-cols-2 gap-3 text-sm">
        <Stat label="Matches" value={formatNumber(p.matchesFound)} />
        <Stat
          label="Records / sec"
          value={formatNumber(Math.round(p.recordsPerSec))}
        />
        <Stat label="Memory used" value={`${formatNumber(p.memUsedMb)} MB`} />
        <Stat label="Memory free" value={`${formatNumber(p.memAvailMb)} MB`} />
      </div>
    </Card>
  );
}

function GpuCard() {
  const p = useProgressStore();
  const tone = p.gpuActive ? "ok" : "mute";
  return (
    <Card>
      <SectionHeader
        title="GPU"
        action={
          <Pill tone={tone}>
            <StatusDot tone={tone} pulse={p.gpuActive} />{" "}
            {p.gpuActive ? "Active" : "Idle"}
          </Pill>
        }
      />
      <div className="grid grid-cols-2 gap-3 text-sm">
        <Stat
          label="VRAM total"
          value={p.gpuTotalMb ? `${formatNumber(p.gpuTotalMb)} MB` : "—"}
        />
        <Stat
          label="VRAM free"
          value={p.gpuFreeMb ? `${formatNumber(p.gpuFreeMb)} MB` : "—"}
        />
      </div>
    </Card>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="surface-soft px-3 py-2">
      <div className="text-2xs uppercase tracking-wider text-ink-500">
        {label}
      </div>
      <div className="text-base text-ink-50 tabular mt-0.5">{value}</div>
    </div>
  );
}

function LogConsole() {
  const entries = useLogStore((s) => s.entries);
  const clear = useLogStore((s) => s.clear);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    // Auto-scroll only if user is near the bottom.
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
    if (nearBottom) el.scrollTop = el.scrollHeight;
  }, [entries.length]);

  const sliced = useMemo(() => entries.slice(-1000), [entries]);

  return (
    <Card padded={false}>
      <div className="flex items-center justify-between p-4 pb-2">
        <SectionHeader
          title="Run log"
          description="Capped to the latest 5,000 entries; oldest are dropped."
        />
        <Button
          tone="ghost"
          size="sm"
          onClick={clear}
          disabled={entries.length === 0}
        >
          Clear
        </Button>
      </div>
      <div
        ref={containerRef}
        role="log"
        aria-live="polite"
        aria-label="Run log"
        className="font-mono text-xs h-[320px] overflow-auto px-4 pb-4 space-y-1 border-t border-ink-800/70"
      >
        {sliced.length === 0 && (
          <div className="text-ink-500 italic py-6 text-center">
            No log entries yet. Start a run to see engine output here.
          </div>
        )}
        {sliced.map((e, i) => (
          <LogLine key={`${e.timestamp_ms}-${i}`} e={e} />
        ))}
      </div>
    </Card>
  );
}

function LogLine({ e }: { e: LogEntryDto }) {
  const cls =
    e.level === "error"
      ? "text-danger-400"
      : e.level === "warn"
        ? "text-warn-400"
        : e.level === "info"
          ? "text-ink-200"
          : "text-ink-400";
  return (
    <div className={cx("flex gap-3", cls)}>
      <span className="text-ink-500 tabular shrink-0">
        {formatTimestamp(e.timestamp_ms)}
      </span>
      <span className="uppercase text-2xs shrink-0 w-10">{e.level}</span>
      <span className="break-words">{e.message}</span>
    </div>
  );
}

function PauseIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="currentColor"
      aria-hidden
    >
      <rect x="6" y="5" width="4" height="14" rx="1" />
      <rect x="14" y="5" width="4" height="14" rx="1" />
    </svg>
  );
}

function PlayIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="currentColor"
      aria-hidden
    >
      <path d="M8 5v14l11-7L8 5z" />
    </svg>
  );
}
