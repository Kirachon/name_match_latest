import { beforeEach, describe, expect, it } from "vitest";
import { useConfigStore } from "@/shared/stores/configStore";
import type { TableSelectionDto } from "@/shared/tauri/types";

const source: TableSelectionDto = {
  session_id: "source-session",
  table: "source_table",
};
const target: TableSelectionDto = {
  session_id: "target-session",
  table: "target_table",
};

describe("configStore", () => {
  beforeEach(() => {
    useConfigStore.getState().reset();
  });

  it("omits cascade settings for quick-match runs", () => {
    useConfigStore.getState().setMode("quick");

    expect(
      useConfigStore.getState().buildRunConfig(source, target).cascade,
    ).toBe(null);
  });

  it("includes geo availability flags for deep cascade runs", () => {
    useConfigStore.getState().setMode("deep");
    useConfigStore.getState().setCascade({
      levels: [1, 4, 10],
      exclusion_mode: "independent",
    });

    expect(
      useConfigStore.getState().buildRunConfig(source, target, {
        hasBarangay: true,
        hasCity: false,
      }).cascade,
    ).toMatchObject({
      enabled: true,
      levels: [1, 4, 10],
      exclusion_mode: "independent",
      has_barangay_code: true,
      has_city_code: false,
    });
  });

  it("restores the default algorithm and mode on reset", () => {
    useConfigStore.getState().setMode("deep");
    useConfigStore.getState().setAlgorithm("levenshtein-weighted");

    useConfigStore.getState().reset();

    expect(useConfigStore.getState().mode).toBe("quick");
    expect(useConfigStore.getState().algorithm).toBe("fuzzy");
  });
});
