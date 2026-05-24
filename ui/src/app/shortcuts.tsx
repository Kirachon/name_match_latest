import { useEffect } from "react";
import type { TabId } from "@/app/TabBar";
import {
  useConnectionStore,
  readinessForRun,
} from "@/shared/stores/connectionStore";
import { useConfigStore } from "@/shared/stores/configStore";
import { useJobStore } from "@/shared/stores/jobStore";
import { useToastStore } from "@/shared/stores/toastStore";
import { cancelMatching, startMatching } from "@/shared/tauri/commands";
import { parseRunConfig } from "@/shared/tauri/zod-schemas";

interface ShortcutOpts {
  activeTab: TabId;
  setTab: (t: TabId) => void;
}

export function useGlobalShortcuts({ activeTab, setTab }: ShortcutOpts) {
  const pushToast = useToastStore((s) => s.push);
  const setActiveJob = useJobStore((s) => s.setActive);
  const jobState = useJobStore((s) => s.state);
  const activeJobId = useJobStore((s) => s.activeJobId);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName?.toLowerCase();
      const isEditable =
        tag === "input" ||
        tag === "textarea" ||
        tag === "select" ||
        target?.isContentEditable;

      // Ctrl+Enter (or Cmd+Enter) — start matching from any tab if ready.
      if (
        (e.ctrlKey || e.metaKey) &&
        e.key === "Enter" &&
        !e.shiftKey &&
        !e.altKey
      ) {
        e.preventDefault();
        triggerStart(setTab, pushToast, setActiveJob);
        return;
      }

      // Escape — request cancel of running job (only when not editing).
      if (e.key === "Escape" && !isEditable) {
        if (
          activeJobId &&
          (jobState === "running" ||
            jobState === "starting" ||
            jobState === "validating")
        ) {
          e.preventDefault();
          void cancelMatching(activeJobId).catch((err: unknown) => {
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
      }

      // Ctrl+1..4 — jump to tab.
      if ((e.ctrlKey || e.metaKey) && ["1", "2", "3", "4"].includes(e.key)) {
        e.preventDefault();
        const idx = Number(e.key) - 1;
        const tabs: TabId[] = ["connect", "configure", "run", "results"];
        const t = tabs[idx];
        if (t) setTab(t);
      }

      // Ctrl+P — pause/resume toggle (when there's an active job).
      if (
        (e.ctrlKey || e.metaKey) &&
        (e.key === "p" || e.key === "P") &&
        !e.shiftKey &&
        !e.altKey
      ) {
        e.preventDefault();
        const w = window as unknown as {
          __nmRunControls?: {
            pause: () => void;
            resume: () => void;
            state: string;
          };
        };
        const c = w.__nmRunControls;
        if (!c) return;
        if (
          c.state === "running" ||
          c.state === "starting" ||
          c.state === "validating"
        ) {
          c.pause();
        } else if (c.state === "paused") {
          c.resume();
        }
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab, jobState, activeJobId]);
}

async function triggerStart(
  setTab: (t: TabId) => void,
  pushToast: ReturnType<typeof useToastStore.getState>["push"],
  setActiveJob: ReturnType<typeof useJobStore.getState>["setActive"],
) {
  const conn = readinessForRun(useConnectionStore.getState());
  if (!conn.ready) {
    pushToast({
      tone: "warn",
      title: "Not ready to run",
      message: conn.reason ?? "Open the Connect tab to finish setup.",
    });
    setTab("connect");
    return;
  }
  const cfgState = useConfigStore.getState();
  const connState = useConnectionStore.getState();
  const srcRaw = connState.source.columns?.raw_columns ?? [];
  const tgtRaw = connState.target.columns?.raw_columns ?? [];
  const draft = cfgState.buildRunConfig(
    {
      session_id: connState.source.session!.session_id,
      table: connState.source.selectedTable!,
      column_mapping: connState.source.columnMapping,
    },
    {
      session_id: connState.target.session!.session_id,
      table: connState.target.selectedTable!,
      column_mapping: connState.target.columnMapping,
    },
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
      title: "Configuration is incomplete",
      message: parsed.issues.slice(0, 3).join("\n"),
    });
    setTab("configure");
    return;
  }
  try {
    const jobId = await startMatching(parsed.value);
    setActiveJob(jobId);
    setTab("run");
    pushToast({
      tone: "info",
      title: "Run started",
      message: `Job ${jobId.slice(0, 8)} queued`,
      ttlMs: 2000,
    });
  } catch (err: unknown) {
    pushToast({
      tone: "error",
      title: "Could not start matching",
      message:
        typeof err === "object" && err && "message" in err
          ? String((err as { message: unknown }).message)
          : String(err),
      ttlMs: null,
    });
  }
}
