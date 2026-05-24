import {
  useConnectionStore,
  readinessForRun,
} from "@/shared/stores/connectionStore";
import { useConfigStore } from "@/shared/stores/configStore";
import { useJobStore } from "@/shared/stores/jobStore";
import { Pill } from "@/shared/components/primitives";
import { cx } from "@/shared/lib/format";

export type TabId = "connect" | "configure" | "run" | "results";

const TABS: Array<{ id: TabId; label: string; hint: string }> = [
  { id: "connect", label: "1. Connect", hint: "Database & tables" },
  { id: "configure", label: "2. Configure", hint: "Algorithm & options" },
  { id: "run", label: "3. Run", hint: "Execute & monitor" },
  { id: "results", label: "4. Results", hint: "Review & export" },
];

export function TabBar({
  active,
  onChange,
}: {
  active: TabId;
  onChange: (next: TabId) => void;
}) {
  const connState = useConnectionStore((s) => s);
  const exportDir = useConfigStore((s) => s.export.output_directory);
  const jobState = useJobStore((s) => s.state);
  const activeJobId = useJobStore((s) => s.activeJobId);

  const conn = readinessForRun(connState);
  const configReady = exportDir.trim().length > 0;
  const runHasJob = activeJobId != null;
  const resultsReady =
    jobState === "completed" || jobState === "failed" || jobState === "cancelled";

  const lockReason: Record<TabId, string | null> = {
    connect: null,
    configure: conn.ready ? null : conn.reason ?? "Connect databases first",
    run: !conn.ready
      ? conn.reason ?? "Connect databases first"
      : !configReady
        ? "Pick an output directory first"
        : null,
    results: resultsReady
      ? null
      : runHasJob
        ? "Run is still in progress"
        : "Start a run first",
  };

  return (
    <div
      role="tablist"
      aria-label="Workflow tabs"
      className="flex items-center px-3 border-b border-ink-800 bg-ink-900/60 h-[var(--tabbar-h)] shrink-0 overflow-x-auto"
    >
      {TABS.map((t) => {
        const locked = lockReason[t.id] != null && t.id !== active;
        return (
          <button
            key={t.id}
            role="tab"
            id={`tab-${t.id}`}
            aria-selected={active === t.id}
            aria-controls={`tabpanel-${t.id}`}
            data-active={active === t.id}
            data-locked={locked}
            disabled={locked && active !== t.id}
            tabIndex={active === t.id ? 0 : -1}
            onClick={() => !locked && onChange(t.id)}
            className={cx("tab-btn shrink-0")}
            title={locked ? lockReason[t.id]! : t.hint}
            onKeyDown={(e) => {
              if (e.key === "ArrowRight") {
                const idx = TABS.findIndex((x) => x.id === active);
                const next = TABS[Math.min(idx + 1, TABS.length - 1)];
                if (next && !lockReason[next.id]) onChange(next.id);
              } else if (e.key === "ArrowLeft") {
                const idx = TABS.findIndex((x) => x.id === active);
                const prev = TABS[Math.max(idx - 1, 0)];
                if (prev) onChange(prev.id);
              }
            }}
          >
            <span>{t.label}</span>
            <span
              className={cx(
                "text-2xs hidden lg:inline",
                active === t.id ? "text-ink-400" : "text-ink-500",
              )}
            >
              {t.hint}
            </span>
            {locked && (
              <Pill tone="mute" className="ml-1 hidden md:inline-flex">
                Locked
              </Pill>
            )}
          </button>
        );
      })}
      <div className="ml-auto flex items-center gap-2 pr-2 text-2xs text-ink-500">
        <kbd className="kbd">Ctrl</kbd>
        <span>+</span>
        <kbd className="kbd">Enter</kbd>
        <span className="text-ink-600">to start</span>
        <span className="text-ink-700 mx-1">·</span>
        <kbd className="kbd">Esc</kbd>
        <span className="text-ink-600">to cancel</span>
      </div>
    </div>
  );
}
