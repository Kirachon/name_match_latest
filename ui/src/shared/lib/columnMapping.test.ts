import { describe, expect, it } from "vitest";
import {
  bothSidesHaveGeoColumn,
  BARANGAY_CODE_HINTS,
  CITY_CODE_HINTS,
  pickColumn,
  sideMappingReady,
  suggestColumnMapping,
} from "./columnMapping";
import type { TableColumnsDto } from "@/shared/tauri/types";

describe("columnMapping", () => {
  it("suggests renamed columns from header hints", () => {
    const mapping = suggestColumnMapping([
      "person_id",
      "fname",
      "lname",
      "dob",
      "household_id",
    ]);
    expect(mapping.id).toBe("person_id");
    expect(mapping.first_name).toBe("fname");
    expect(mapping.last_name).toBe("lname");
    expect(mapping.birthdate).toBe("dob");
    expect(mapping.hh_id).toBe("household_id");
  });

  it("detects geographic aliases on both sides", () => {
    expect(
      bothSidesHaveGeoColumn(
        ["brgy_code", "id"],
        ["barangay", "id"],
        BARANGAY_CODE_HINTS,
      ),
    ).toBe(true);
    expect(
      bothSidesHaveGeoColumn(
        ["municipality_code"],
        ["city_code"],
        CITY_CODE_HINTS,
      ),
    ).toBe(true);
    expect(pickColumn(["city_muni_code"], CITY_CODE_HINTS)).toBe(
      "city_muni_code",
    );
  });

  it("requires explicit mapping for non-standard database tables", () => {
    const columns: TableColumnsDto = {
      has_id: false,
      has_uuid: false,
      has_first_name: false,
      has_middle_name: false,
      has_last_name: false,
      has_birthdate: false,
      has_hh_id: false,
      raw_columns: ["fname", "lname", "dob", "person_id"],
    };
    const blocked = sideMappingReady("source", {
      mode: "database",
      columns,
      columnMapping: null,
      file: { preview: null },
    });
    expect(blocked.ok).toBe(false);
    expect(blocked.reason).toContain("Map Source");

    const ready = sideMappingReady("source", {
      mode: "database",
      columns,
      columnMapping: suggestColumnMapping(columns.raw_columns),
      file: { preview: null },
    });
    expect(ready.ok).toBe(true);
  });

  it("blocks database runs when mapped columns are missing from the table", () => {
    const ready = sideMappingReady("target", {
      mode: "database",
      columns: {
        has_id: true,
        has_uuid: false,
        has_first_name: true,
        has_middle_name: false,
        has_last_name: true,
        has_birthdate: true,
        has_hh_id: false,
        raw_columns: ["id", "first_name", "last_name", "birthdate"],
      },
      columnMapping: {
        id: "missing_id",
        uuid: null,
        first_name: "first_name",
        middle_name: null,
        last_name: "last_name",
        birthdate: "birthdate",
        hh_id: null,
      },
      file: { preview: null },
    });
    expect(ready.ok).toBe(false);
    expect(ready.reason).toContain("missing_id");
  });
});
