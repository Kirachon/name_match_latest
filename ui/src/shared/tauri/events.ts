import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type { JobStateEventDto, LogEntryDto, ProgressEventDto } from "./types";

export const EVENT_PROGRESS = "match-progress" as const;
export const EVENT_STATE = "job-state" as const;
export const EVENT_LOG = "log-entry" as const;
export const EVENT_ERROR = "job-error" as const;

export type ProgressHandler = (e: ProgressEventDto) => void;
export type StateHandler = (e: JobStateEventDto) => void;
export type LogHandler = (e: LogEntryDto) => void;

function isTauriRuntime(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

async function listenIfAvailable<T>(
  event: string,
  cb: (payload: T) => void,
): Promise<UnlistenFn> {
  if (!isTauriRuntime()) {
    return () => {};
  }
  return listen<T>(event, (evt) => cb(evt.payload));
}

export async function onProgress(cb: ProgressHandler): Promise<UnlistenFn> {
  return listenIfAvailable<ProgressEventDto>(EVENT_PROGRESS, cb);
}
export async function onState(cb: StateHandler): Promise<UnlistenFn> {
  return listenIfAvailable<JobStateEventDto>(EVENT_STATE, cb);
}
export async function onLog(cb: LogHandler): Promise<UnlistenFn> {
  return listenIfAvailable<LogEntryDto>(EVENT_LOG, cb);
}
