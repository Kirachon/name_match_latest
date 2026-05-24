import { open } from "@tauri-apps/plugin-dialog";
import { loadCsvPreview } from "@/shared/tauri/commands";
import {
  type SessionSide,
  useConnectionStore,
} from "@/shared/stores/connectionStore";
import { useToastStore } from "@/shared/stores/toastStore";
import {
  Button,
  Card,
  Field,
  Pill,
  SectionHeader,
} from "@/shared/components/primitives";
import type {
  ColumnMappingDto,
  CsvDelimiterDto,
  CsvEncodingDto,
  CsvPreviewDto,
  TableColumnsDto,
} from "@/shared/tauri/types";
import { ColumnMapper } from "./ColumnMapper";

const ENCODINGS: Array<{ id: CsvEncodingDto | ""; label: string }> = [
  { id: "", label: "Auto" },
  { id: "utf8", label: "UTF-8" },
  { id: "utf8-bom", label: "UTF-8 BOM" },
  { id: "windows1252", label: "Windows-1252" },
  { id: "latin1", label: "Latin-1" },
];

const DELIMITERS: Array<{ id: CsvDelimiterDto | ""; label: string }> = [
  { id: "", label: "Auto" },
  { id: "comma", label: "Comma" },
  { id: "semicolon", label: "Semicolon" },
  { id: "tab", label: "Tab" },
];

export function FileSourceCard({ side }: { side: SessionSide }) {
  const file = useConnectionStore((s) => s[side].file);
  const columnMapping = useConnectionStore((s) => s[side].columnMapping);
  const setFileSource = useConnectionStore((s) => s.setFileSource);
  const setColumnMapping = useConnectionStore((s) => s.setColumnMapping);
  const pushToast = useToastStore((s) => s.push);

  async function pickFile() {
    const picked = await open({
      directory: false,
      multiple: false,
      filters: [{ name: "CSV", extensions: ["csv", "txt"] }],
    }).catch(() => null);
    if (typeof picked === "string") {
      setFileSource(side, { path: picked, preview: null, error: null });
      await preview(picked);
    }
  }

  async function preview(path = file.path) {
    if (!path.trim()) {
      setFileSource(side, { error: "Choose a CSV file first" });
      return;
    }
    setFileSource(side, { loading: true, error: null });
    try {
      const result = await loadCsvPreview({
        path,
        encoding: file.encoding,
        delimiter: file.delimiter,
        date_format: file.dateFormat,
      });
      setFileSource(side, {
        path,
        preview: result,
        loading: false,
        error: null,
        encoding: result.encoding,
        delimiter: result.delimiter,
        dateFormat: result.date_format,
      });
      setColumnMapping(side, inferColumnMapping(result.headers));
      pushToast({
        tone: "success",
        title: `${sideLabel(side)} CSV preview loaded`,
        message: `${result.headers.length} columns detected`,
        ttlMs: 1800,
      });
    } catch (err: unknown) {
      const message =
        typeof err === "object" && err && "message" in err
          ? String((err as { message: unknown }).message)
          : String(err);
      setFileSource(side, { loading: false, error: message });
      pushToast({ tone: "error", title: "CSV preview failed", message });
    }
  }

  return (
    <Card className="space-y-4">
      <SectionHeader
        title={side === "source" ? "Source File" : "Target File"}
        description="Preview a CSV file before mapping its columns for matching."
        action={
          file.preview ? (
            <Pill tone="info">{file.preview.headers.length} columns</Pill>
          ) : (
            <Pill tone="mute">CSV</Pill>
          )
        }
      />
      <Field label="CSV path" required>
        <div className="flex gap-2">
          <input
            className="input flex-1"
            value={file.path}
            onChange={(e) =>
              setFileSource(side, {
                path: e.target.value,
                preview: null,
                error: null,
              })
            }
            placeholder="C:/data/beneficiaries.csv"
          />
          <Button tone="secondary" onClick={pickFile}>
            Browse...
          </Button>
        </div>
      </Field>
      <div className="grid md:grid-cols-3 gap-3">
        <Field label="Encoding">
          <select
            className="select"
            value={file.encoding ?? ""}
            onChange={(e) =>
              setFileSource(side, {
                encoding: (e.target.value || null) as CsvEncodingDto | null,
                preview: null,
              })
            }
          >
            {ENCODINGS.map((encoding) => (
              <option key={encoding.id || "auto"} value={encoding.id}>
                {encoding.label}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Delimiter">
          <select
            className="select"
            value={file.delimiter ?? ""}
            onChange={(e) =>
              setFileSource(side, {
                delimiter: (e.target.value || null) as CsvDelimiterDto | null,
                preview: null,
              })
            }
          >
            {DELIMITERS.map((delimiter) => (
              <option key={delimiter.id || "auto"} value={delimiter.id}>
                {delimiter.label}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Date format">
          <input
            className="input font-mono"
            value={file.dateFormat}
            onChange={(e) =>
              setFileSource(side, {
                dateFormat: e.target.value,
                preview: null,
              })
            }
            placeholder="%Y-%m-%d"
          />
        </Field>
      </div>
      <div className="flex flex-wrap gap-2">
        <Button tone="primary" loading={file.loading} onClick={() => preview()}>
          Preview CSV
        </Button>
        {file.preview && (
          <Pill tone="ok">
            {delimiterLabel(file.preview.delimiter)} / {file.preview.encoding}
          </Pill>
        )}
      </div>
      {file.error && (
        <div role="alert" className="help-error">
          {file.error}
        </div>
      )}
      {file.preview && (
        <>
          <ColumnMapper
            columns={columnsFromHeaders(file.preview.headers)}
            value={columnMapping}
            onChange={(mapping) => setColumnMapping(side, mapping)}
          />
          <PreviewTable preview={file.preview} />
        </>
      )}
    </Card>
  );
}

function columnsFromHeaders(headers: string[]): TableColumnsDto {
  return {
    has_id: headers.includes("id"),
    has_uuid: headers.includes("uuid"),
    has_first_name: headers.includes("first_name"),
    has_middle_name: headers.includes("middle_name"),
    has_last_name: headers.includes("last_name"),
    has_birthdate: headers.includes("birthdate"),
    has_hh_id: headers.includes("hh_id"),
    raw_columns: headers,
  };
}

function inferColumnMapping(headers: string[]): ColumnMappingDto | null {
  const columns = columnsFromHeaders(headers);
  const id = pick(headers, ["id", "person_id", "beneficiary_id"]);
  const first_name = pick(headers, [
    "first_name",
    "firstname",
    "fname",
    "given_name",
  ]);
  const last_name = pick(headers, [
    "last_name",
    "lastname",
    "lname",
    "surname",
  ]);
  const birthdate = pick(headers, [
    "birthdate",
    "birth_date",
    "birthday",
    "dob",
  ]);
  if (!id || !first_name || !last_name || !birthdate) return null;
  return {
    id,
    uuid: columns.has_uuid ? "uuid" : null,
    first_name,
    middle_name: pick(headers, ["middle_name", "middlename", "mname"]) || null,
    last_name,
    birthdate,
    hh_id: columns.has_hh_id ? "hh_id" : null,
  };
}

function pick(headers: string[], hints: string[]): string {
  const normalized = new Map(
    headers.map((header) => [normalize(header), header]),
  );
  for (const hint of hints) {
    const found = normalized.get(normalize(hint));
    if (found) return found;
  }
  for (const hint of hints) {
    const found = headers.find((header) =>
      normalize(header).includes(normalize(hint)),
    );
    if (found) return found;
  }
  return "";
}

function normalize(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]/g, "");
}

function PreviewTable({ preview }: { preview: CsvPreviewDto }) {
  return (
    <div className="space-y-3">
      {preview.warnings.length > 0 && (
        <div className="surface-soft p-3 text-xs text-warn-300 space-y-1">
          {preview.warnings.slice(0, 3).map((warning) => (
            <div key={warning}>{warning}</div>
          ))}
        </div>
      )}
      <div className="overflow-auto border border-ink-800 rounded-lg">
        <table className="min-w-full text-xs">
          <thead className="bg-ink-900/80 text-ink-300">
            <tr>
              {preview.headers.map((header) => (
                <th
                  key={header}
                  className="px-2 py-2 text-left font-medium whitespace-nowrap"
                >
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.rows.map((row, index) => (
              <tr key={index} className="border-t border-ink-800/70">
                {preview.headers.map((header, colIndex) => (
                  <td
                    key={`${header}-${colIndex}`}
                    className="px-2 py-1.5 text-ink-300 whitespace-nowrap"
                  >
                    {row[colIndex] ?? ""}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function sideLabel(side: SessionSide) {
  return side === "source" ? "Source" : "Target";
}

function delimiterLabel(delimiter: CsvDelimiterDto) {
  return delimiter === "comma"
    ? "Comma"
    : delimiter === "semicolon"
      ? "Semicolon"
      : "Tab";
}
