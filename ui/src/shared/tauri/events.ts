import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type { JobStateEventDto, LogEntryDto, ProgressEventDto } from "./types";

export const EVENT_PROGRESS = "match-progress" as const;
export const EVENT_STATE = "job-state" as const;
export const EVENT_LOG = "log-entry" as const;
export const EVENT_ERROR = "job-error" as const;

export type ProgressHandler = (e: ProgressEventDto) => void;
export type StateHandler = (e: JobStateEventDto) => void;
export type LogHandler = (e: LogEntryDto) => void;

export async function onProgress(cb: ProgressHandler): Promise<UnlistenFn> {
  return listen<ProgressEventDto>(EVENT_PROGRESS, (evt) => cb(evt.payload));
}
export async function onState(cb: StateHandler): Promise<UnlistenFn> {
  return listen<JobStateEventDto>(EVENT_STATE, (evt) => cb(evt.payload));
}
export async function onLog(cb: LogHandler): Promise<UnlistenFn> {
  return listen<LogEntryDto>(EVENT_LOG, (evt) => cb(evt.payload));
}
