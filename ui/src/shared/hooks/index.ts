import { useEffect, useState } from "react";
import type { UnlistenFn } from "@tauri-apps/api/event";

/**
 * Subscribes to a Tauri event for the lifetime of a component. The
 * subscriber receives the (already-unwrapped) payload — the wrapper here
 * automatically unlistens on unmount and on dependency change.
 */
export function useTauriListener<T>(
  factory: (cb: (payload: T) => void) => Promise<UnlistenFn>,
  cb: (payload: T) => void,
  deps: ReadonlyArray<unknown> = [],
) {
  useEffect(() => {
    let cancelled = false;
    let unlisten: UnlistenFn | null = null;
    factory((payload) => {
      if (!cancelled) cb(payload);
    }).then((u) => {
      if (cancelled) {
        u();
      } else {
        unlisten = u;
      }
    });
    return () => {
      cancelled = true;
      if (unlisten) unlisten();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}

export function useDebounced<T>(value: T, ms = 250): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const id = window.setTimeout(() => setV(value), ms);
    return () => window.clearTimeout(id);
  }, [value, ms]);
  return v;
}
