import { create } from "zustand";
import { persist } from "zustand/middleware";

export type ResultsSortKey =
  | "row_id"
  | "confidence"
  | "source_name"
  | "target_name";
export type ResultsSortDir = "asc" | "desc";

export interface ResultsViewState {
  pageIndex: number;
  search: string;
  sortBy: ResultsSortKey;
  sortDir: ResultsSortDir;
  minConf: number;
  levels: number[];
}

export const DEFAULT_RESULTS_VIEW_STATE: ResultsViewState = {
  pageIndex: 0,
  search: "",
  sortBy: "confidence",
  sortDir: "desc",
  minConf: 0,
  levels: [],
};

interface ResultsStore {
  jobs: Record<string, ResultsViewState>;
  patchJob: (jobId: string, patch: Partial<ResultsViewState>) => void;
  resetJob: (jobId: string) => void;
}

export const useResultsStore = create<ResultsStore>()(
  persist(
    (set) => ({
      jobs: {},
      patchJob: (jobId, patch) =>
        set((state) => ({
          jobs: {
            ...state.jobs,
            [jobId]: {
              ...DEFAULT_RESULTS_VIEW_STATE,
              ...state.jobs[jobId],
              ...patch,
            },
          },
        })),
      resetJob: (jobId) =>
        set((state) => {
          const jobs = { ...state.jobs };
          delete jobs[jobId];
          return { jobs };
        }),
    }),
    {
      name: "nm-results-view-v1",
      version: 1,
    },
  ),
);
