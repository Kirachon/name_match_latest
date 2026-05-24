import { useEffect } from "react";
import { onLog, onProgress, onState } from "@/shared/tauri/events";
import {
  useJobStore,
  useLogStore,
  useProgressStore,
} from "@/shared/stores/jobStore";
import { useToastStore } from "@/shared/stores/toastStore";

/**
 * EventBridge mounts the Tauri event listeners exactly once at the app root
 * and dispatches them to the high/low-frequency stores.
 *
 * Stale-job protection: every payload carries `job_id`; we ignore events that
 * do not match the currently active job once one has been registered.
 */
export function EventBridge() {
  const setProgress = useProgressStore((s) => s.apply);
  const pushLog = useLogStore((s) => s.push);
  const setJobState = useJobStore((s) => s.setState);
  const activeJobId = useJobStore((s) => s.activeJobId);
  const pushToast = useToastStore((s) => s.push);

  useEffect(() => {
    let unlistenProgress: (() => void) | null = null;
    let unlistenState: (() => void) | null = null;
    let unlistenLog: (() => void) | null = null;
    let cancelled = false;

    onProgress((p) => {
      if (cancelled) return;
      const currentActive = useJobStore.getState().activeJobId;
      if (currentActive && p.job_id !== currentActive) return;
      setProgress(p);
    }).then((u) => {
      if (cancelled) u();
      else unlistenProgress = u;
    });

    onState((e) => {
      if (cancelled) return;
      const currentActive = useJobStore.getState().activeJobId;
      if (currentActive && e.job_id !== currentActive) return;
      setJobState(e.state, e.detail ?? null);
      if (e.state === "completed") {
        pushToast({
          tone: "success",
          title: "Run completed",
          message: "Open the Results tab to review.",
        });
      } else if (e.state === "failed") {
        pushToast({
          tone: "error",
          title: "Run failed",
          message: e.detail ?? "See the run log for details.",
          ttlMs: null,
        });
      } else if (e.state === "cancelled") {
        pushToast({
          tone: "warn",
          title: "Run cancelled",
          message: "Cancellation completed cleanly.",
        });
      }
    }).then((u) => {
      if (cancelled) u();
      else unlistenState = u;
    });

    onLog((entry) => {
      if (cancelled) return;
      const currentActive = useJobStore.getState().activeJobId;
      if (currentActive && entry.job_id !== currentActive) return;
      pushLog(entry);
    }).then((u) => {
      if (cancelled) u();
      else unlistenLog = u;
    });

    return () => {
      cancelled = true;
      unlistenProgress?.();
      unlistenState?.();
      unlistenLog?.();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeJobId]);

  return null;
}
