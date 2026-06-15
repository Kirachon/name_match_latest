import type { ColumnMappingDto, TableColumnsDto } from "@/shared/tauri/types";

const REQUIRED_MAPPING_FIELDS: Array<{
  key: keyof Pick<
    ColumnMappingDto,
    "id" | "first_name" | "last_name" | "birthdate"
  >;
  label: string;
  hints: string[];
}> = [
  { key: "id", label: "ID", hints: ["id", "person_id", "beneficiary_id"] },
  {
    key: "first_name",
    label: "First name",
    hints: ["first_name", "firstname", "fname", "given_name"],
  },
  {
    key: "last_name",
    label: "Last name",
    hints: ["last_name", "lastname", "lname", "surname"],
  },
  {
    key: "birthdate",
    label: "Birthdate",
    hints: ["birthdate", "birth_date", "birthday", "dob"],
  },
];

export const BARANGAY_CODE_HINTS = [
  "barangay_code",
  "brgy_code",
  "barangay",
  "brgy",
];
export const CITY_CODE_HINTS = [
  "city_code",
  "city_muni_code",
  "municipality_code",
  "city",
  "municipality",
];

export function normalizeColumnName(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]/g, "");
}

export function pickColumn(columns: string[], hints: string[]): string {
  const normalized = new Map(
    columns.map((column) => [normalizeColumnName(column), column]),
  );
  for (const hint of hints) {
    const found = normalized.get(normalizeColumnName(hint));
    if (found) return found;
  }
  for (const hint of hints) {
    const found = columns.find((column) =>
      normalizeColumnName(column).includes(normalizeColumnName(hint)),
    );
    if (found) return found;
  }
  return "";
}

export function hasColumnAlias(columns: string[], hints: string[]): boolean {
  return pickColumn(columns, hints) !== "";
}

export function suggestColumnMapping(columns: string[]): ColumnMappingDto {
  return {
    id: pickColumn(columns, REQUIRED_MAPPING_FIELDS[0].hints),
    uuid: pickColumn(columns, ["uuid"]) || null,
    first_name: pickColumn(columns, REQUIRED_MAPPING_FIELDS[1].hints),
    middle_name:
      pickColumn(columns, [
        "middle_name",
        "middlename",
        "mname",
        "middle_initial",
      ]) || null,
    last_name: pickColumn(columns, REQUIRED_MAPPING_FIELDS[2].hints),
    birthdate: pickColumn(columns, REQUIRED_MAPPING_FIELDS[3].hints),
    hh_id: pickColumn(columns, ["hh_id", "household_id"]) || null,
  };
}

export function missingRequiredMapping(mapping: ColumnMappingDto): string[] {
  return REQUIRED_MAPPING_FIELDS.filter((field) => !mapping[field.key]).map(
    (field) => field.label,
  );
}

function usesStandardColumnNames(columns: TableColumnsDto): boolean {
  return (
    columns.has_id &&
    columns.has_first_name &&
    columns.has_last_name &&
    columns.has_birthdate
  );
}

function invalidMappedColumn(
  mapping: ColumnMappingDto,
  rawColumns: string[],
): string | null {
  const raw = new Set(rawColumns);
  for (const field of REQUIRED_MAPPING_FIELDS) {
    const mapped = mapping[field.key];
    if (mapped && !raw.has(mapped)) {
      return mapped;
    }
  }
  return null;
}

export function sideMappingReady(
  label: "source" | "target",
  side: {
    mode: "database" | "file";
    columns: TableColumnsDto | null;
    columnMapping: ColumnMappingDto | null;
    file: { preview: unknown | null };
  },
): { ok: boolean; reason: string | null } {
  const name = label === "source" ? "Source" : "Target";

  if (side.mode === "file") {
    if (!side.file.preview) {
      return { ok: false, reason: `Preview a ${label} file` };
    }
    if (!side.columnMapping) {
      return { ok: false, reason: `Map the ${label} file columns` };
    }
    const missing = missingRequiredMapping(side.columnMapping);
    if (missing.length > 0) {
      return {
        ok: false,
        reason: `${name} mapping missing ${missing.join(", ")}`,
      };
    }
    return { ok: true, reason: null };
  }

  if (!side.columns) {
    return { ok: false, reason: `${name} table schema not loaded` };
  }
  if (side.columns.raw_columns.length === 0) {
    return {
      ok: false,
      reason: `${name} table columns could not be discovered`,
    };
  }

  const needsCustomMapping = !usesStandardColumnNames(side.columns);
  if (needsCustomMapping && !side.columnMapping) {
    return { ok: false, reason: `Map ${name} table columns` };
  }

  if (side.columnMapping) {
    const missing = missingRequiredMapping(side.columnMapping);
    if (missing.length > 0) {
      return { ok: false, reason: `Map ${name} ${missing.join(", ")}` };
    }
    const invalid = invalidMappedColumn(
      side.columnMapping,
      side.columns.raw_columns,
    );
    if (invalid) {
      return {
        ok: false,
        reason: `${name}: mapped column '${invalid}' not in table`,
      };
    }
  }

  return { ok: true, reason: null };
}

export function bothSidesHaveGeoColumn(
  sourceColumns: string[],
  targetColumns: string[],
  hints: string[],
): boolean {
  return (
    hasColumnAlias(sourceColumns, hints) && hasColumnAlias(targetColumns, hints)
  );
}
