import { useEffect, useMemo } from "react";
import { Field, Pill } from "@/shared/components/primitives";
import type { ColumnMappingDto, TableColumnsDto } from "@/shared/tauri/types";

const REQUIRED_FIELDS: Array<{
  key: "id" | "first_name" | "last_name" | "birthdate";
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

const OPTIONAL_FIELDS: Array<{
  key: "uuid" | "middle_name" | "hh_id";
  label: string;
  hints: string[];
}> = [
  { key: "uuid", label: "UUID", hints: ["uuid"] },
  {
    key: "middle_name",
    label: "Middle name",
    hints: ["middle_name", "middlename", "mname", "middle_initial"],
  },
  { key: "hh_id", label: "Household ID", hints: ["hh_id", "household_id"] },
];

export function ColumnMapper({
  columns,
  value,
  onChange,
}: {
  columns: TableColumnsDto;
  value: ColumnMappingDto | null;
  onChange: (mapping: ColumnMappingDto | null) => void;
}) {
  const rawColumns = columns.raw_columns;
  const suggested = useMemo(() => suggestMapping(rawColumns), [rawColumns]);
  const mapping = value ?? suggested;

  useEffect(() => {
    if (!value && missingRequired(suggested).length === 0) {
      onChange(suggested);
    }
  }, [onChange, suggested, value]);

  function setRequired(
    key: (typeof REQUIRED_FIELDS)[number]["key"],
    next: string,
  ) {
    onChange({ ...mapping, [key]: next });
  }

  function setOptional(
    key: (typeof OPTIONAL_FIELDS)[number]["key"],
    next: string,
  ) {
    onChange({ ...mapping, [key]: next || null });
  }

  return (
    <div className="space-y-3 border-t border-ink-800/70 pt-3">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="section-title">Column mapping</div>
          <p className="text-xs text-ink-400">
            Map non-standard table columns to the fields the matcher expects.
          </p>
        </div>
        <button
          type="button"
          className="text-xs text-accent-300 hover:text-accent-200"
          onClick={() => onChange(suggested)}
        >
          Auto
        </button>
      </div>
      <div className="grid md:grid-cols-2 gap-3">
        {REQUIRED_FIELDS.map((field) => (
          <MapperSelect
            key={field.key}
            label={field.label}
            required
            columns={rawColumns}
            value={mapping[field.key]}
            onChange={(next) => setRequired(field.key, next)}
          />
        ))}
        {OPTIONAL_FIELDS.map((field) => (
          <MapperSelect
            key={field.key}
            label={field.label}
            columns={rawColumns}
            value={mapping[field.key] ?? ""}
            onChange={(next) => setOptional(field.key, next)}
          />
        ))}
      </div>
      <div className="flex flex-wrap gap-2">
        {missingRequired(mapping).length === 0 ? (
          <Pill tone="ok">Mapping ready</Pill>
        ) : (
          <Pill tone="danger">
            Missing {missingRequired(mapping).join(", ")}
          </Pill>
        )}
        {value ? (
          <Pill tone="info">Custom mapping</Pill>
        ) : (
          <Pill tone="mute">Auto-suggested</Pill>
        )}
      </div>
    </div>
  );
}

function MapperSelect({
  label,
  columns,
  value,
  onChange,
  required,
}: {
  label: string;
  columns: string[];
  value: string;
  onChange: (value: string) => void;
  required?: boolean;
}) {
  return (
    <Field label={label} required={required}>
      <select
        className="select"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {!required && <option value="">Not mapped</option>}
        {required && !value && <option value="">Select a column</option>}
        {columns.map((column) => (
          <option key={column} value={column}>
            {column}
          </option>
        ))}
      </select>
    </Field>
  );
}

function suggestMapping(columns: string[]): ColumnMappingDto {
  return {
    id: pick(columns, REQUIRED_FIELDS[0].hints),
    uuid: pick(columns, OPTIONAL_FIELDS[0].hints) || null,
    first_name: pick(columns, REQUIRED_FIELDS[1].hints),
    middle_name: pick(columns, OPTIONAL_FIELDS[1].hints) || null,
    last_name: pick(columns, REQUIRED_FIELDS[2].hints),
    birthdate: pick(columns, REQUIRED_FIELDS[3].hints),
    hh_id: pick(columns, OPTIONAL_FIELDS[2].hints) || null,
  };
}

function pick(columns: string[], hints: string[]): string {
  const normalized = new Map(columns.map((c) => [normalize(c), c]));
  for (const hint of hints) {
    const found = normalized.get(normalize(hint));
    if (found) return found;
  }
  for (const hint of hints) {
    const found = columns.find((column) =>
      normalize(column).includes(normalize(hint)),
    );
    if (found) return found;
  }
  return "";
}

function normalize(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]/g, "");
}

function missingRequired(mapping: ColumnMappingDto): string[] {
  return REQUIRED_FIELDS.filter((field) => !mapping[field.key]).map(
    (field) => field.label,
  );
}
