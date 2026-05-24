import { beforeEach, describe, expect, it } from "vitest";
import {
  DEFAULT_RESULTS_VIEW_STATE,
  useResultsStore,
} from "@/shared/stores/resultsStore";

describe("resultsStore", () => {
  beforeEach(() => {
    localStorage.clear();
    useResultsStore.setState({ jobs: {} });
  });

  it("patches a job from the default result view state", () => {
    useResultsStore.getState().patchJob("job-1", {
      pageIndex: 3,
      search: "delacruz",
      levels: [1, 10],
    });

    expect(useResultsStore.getState().jobs["job-1"]).toEqual({
      ...DEFAULT_RESULTS_VIEW_STATE,
      pageIndex: 3,
      search: "delacruz",
      levels: [1, 10],
    });
  });

  it("keeps persisted view state scoped per job id", () => {
    useResultsStore.getState().patchJob("job-1", {
      pageIndex: 2,
      levels: [2],
    });
    useResultsStore.getState().patchJob("job-2", {
      pageIndex: 0,
      levels: [11],
      sortDir: "asc",
    });

    expect(useResultsStore.getState().jobs["job-1"]?.levels).toEqual([2]);
    expect(useResultsStore.getState().jobs["job-2"]?.levels).toEqual([11]);
    expect(useResultsStore.getState().jobs["job-2"]?.sortDir).toBe("asc");
  });

  it("forgets only the requested job view state", () => {
    useResultsStore.getState().patchJob("job-1", { levels: [1] });
    useResultsStore.getState().patchJob("job-2", { levels: [2] });

    useResultsStore.getState().resetJob("job-1");

    expect(useResultsStore.getState().jobs["job-1"]).toBeUndefined();
    expect(useResultsStore.getState().jobs["job-2"]?.levels).toEqual([2]);
  });
});
