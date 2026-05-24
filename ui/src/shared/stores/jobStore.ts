import { create } from "zustand";
import type {
  JobStateDto,
  JobSummaryDto,
  LogEntryDto,
  PipelineStageDto,
  ProgressEventDto,
} from "@/shared/tauri/types";

/** High-frequency: progress + stage. Splitting prevents the rest of the app
 *  from re-rendering at 20 Hz. */
interface ProgressStore {
  jobId: string | null;
  state: JobStateDto;
  stage: PipelineStageDto;
  processed: number;
  total: number;
  percent: number;
  etaSecs: number;
  memUsedMb: number;
  memAvailMb: number;
  gpuTotalMb: number;
  gpuFreeMb: number;
  gpuActive: boolean;
  recordsPerSec: number;
  matchesFound: number;
  apply: (e: ProgressEventDto) => void;
  reset: () => void;
}

const emptyProgress: Omit<ProgressStore, "apply" | "reset"> = {
  jobId: null,
  state: "idle",
  stage: "idle",
  processed: 0,
  total: 0,
  percent: 0,
  etaSecs: 0,
  memUsedMb: 0,
  memAvailMb: 0,
  gpuTotalMb: 0,
  gpuFreeMb: 0,
  gpuActive: false,
  recordsPerSec: 0,
  matchesFound: 0,
};

export const useProgressStore = create<ProgressStore>((set) => ({
  ...emptyProgress,
  apply: (e) =>
    set({
      jobId: e.job_id,
      state: e.state,
      stage: e.stage,
      processed: e.processed,
      total: e.total,
      percent: e.percent,
      etaSecs: e.eta_secs,
      memUsedMb: e.mem_used_mb,
      memAvailMb: e.mem_avail_mb,
      gpuTotalMb: e.gpu_total_mb,
      gpuFreeMb: e.gpu_free_mb,
      gpuActive: e.gpu_active,
      recordsPerSec: e.records_per_sec,
      matchesFound: e.matches_found,
    }),
  reset: () => set({ ...emptyProgress }),
}));

/** Low-frequency state changes. Listeners on this store re-render only on
 *  major lifecycle transitions. */
interface JobStore {
  activeJobId: string | null;
  state: JobStateDto;
  detail: string | null;
  summary: JobSummaryDto | null;
  setActive: (jobId: string | null) => void;
  setState: (state: JobStateDto, detail?: string | null) => void;
  setSummary: (summary: JobSummaryDto | null) => void;
  reset: () => void;
}

export const useJobStore = create<JobStore>((set) => ({
  activeJobId: null,
  state: "idle",
  detail: null,
  summary: null,
  setActive: (activeJobId) => set({ activeJobId }),
  setState: (state, detail = null) => set({ state, detail }),
  setSummary: (summary) => set({ summary }),
  reset: () =>
    set({ activeJobId: null, state: "idle", detail: null, summary: null }),
}));

/** Bounded ring buffer for log lines. The frontend drops oldest entries past
 *  `MAX_LOG_LINES` to keep the DOM tree small. */
const MAX_LOG_LINES = 5_000;

interface LogStore {
  entries: LogEntryDto[];
  push: (entry: LogEntryDto) => void;
  clear: () => void;
}

export const useLogStore = create<LogStore>((set) => ({
  entries: [],
  push: (entry) =>
    set((s) => {
      const next =
        s.entries.length >= MAX_LOG_LINES
          ? [...s.entries.slice(s.entries.length - MAX_LOG_LINES + 1), entry]
          : [...s.entries, entry];
      return { entries: next };
    }),
  clear: () => set({ entries: [] }),
}));
