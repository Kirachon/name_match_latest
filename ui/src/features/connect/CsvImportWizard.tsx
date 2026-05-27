import { open } from "@tauri-apps/plugin-dialog";
import {
  cancelCsvImport,
  getCsvImportStatus,
  getRowCount,
  getTableColumns,
  listTables,
  previewCsvImport,
  startCsvImport,
  validateCsvImportPlan,
} from "@/shared/tauri/commands";
import type {
  ColumnMappingDto,
  CsvDelimiterDto,
  CsvEncodingDto,
  CsvImportJobDto,
  CsvImportJobPhaseDto,
  CsvImportRequestDto,
  CsvPreviewDto,
  TableColumnsDto,
} from "@/shared/tauri/types";
import { useConnectionStore } from "@/shared/stores/connectionStore";
import { useToastStore } from "@/shared/stores/toastStore";
import {
  Button,
  Card,
  Field,
  Pill,
  SectionHeader,
  Toggle,
} from "@/shared/components/primitives";
import { ColumnMapper } from "./ColumnMapper";
import { useCsvImportStore } from "./csvImportStore";
import { loadPersistedConnection, savePersistedConnection } from "./persistence";

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

const TERMINAL_IMPORT_PHASES: CsvImportJobPhaseDto[] = [
  "complete",
  "failed",
  "cancelled",
];

export function CsvImportWizard() {
  const wizard = useCsvImportStore();
  const patch = useCsvImportStore((s) => s.patch);
  const close = useCsvImportStore((s) => s.close);
  const sideState = useConnectionStore((s) => s[wizard.side]);
  const setMode = useConnectionStore((s) => s.setMode);
  const setTables = useConnectionStore((s) => s.setTables);
  const setSelectedTable = useConnectionStore((s) => s.setSelectedTable);
  const setColumns = useConnectionStore((s) => s.setColumns);
  const setColumnMapping = useConnectionStore((s) => s.setColumnMapping);
  const setRowCount = useConnectionStore((s) => s.setRowCount);
  const pushToast = useToastStore((s) => s.push);

  if (!wizard.open) return null;

  const session = sideState.session;
  const database = session?.database ?? "";
  const canUseDb = !!session;

  async function chooseCsv() {
    const picked = await open({
      directory: false,
      multiple: false,
      filters: [{ name: "CSV", extensions: ["csv", "txt"] }],
    }).catch(() => null);
    if (typeof picked === "string") {
      patch({
        filePath: picked,
        preview: null,
        mapping: null,
        dryRun: null,
        error: null,
      });
    }
  }

  async function preview() {
    if (!session) return;
    const request = buildRequest(session.session_id, database, {
      requireMapping: false,
    });
    if (!request) return;
    patch({ previewLoading: true, error: null });
    try {
      const result = await previewCsvImport(request);
      patch({
        preview: result,
        mapping: inferColumnMapping(result.headers),
        previewLoading: false,
        step: "mapping",
      });
      pushToast({
        tone: "success",
        title: "CSV preview ready",
        message: `${result.headers.length} columns detected`,
      });
    } catch (err) {
      patch({ previewLoading: false, error: errMsg(err) });
      pushToast({ tone: "error", title: "Preview failed", message: errMsg(err) });
    }
  }

  async function dryRun() {
    if (!session) return;
    const request = buildRequest(session.session_id, database);
    if (!request) return;
    patch({ dryRunLoading: true, error: null });
    try {
      const result = await validateCsvImportPlan(request);
      patch({ dryRun: result, dryRunLoading: false, step: "dry-run" });
      pushToast({
        tone: result.invalid_rows > 0 ? "warn" : "success",
        title: "Dry run complete",
        message: `${result.valid_rows.toLocaleString()} valid rows`,
      });
    } catch (err) {
      patch({ dryRunLoading: false, error: errMsg(err) });
      pushToast({ tone: "error", title: "Dry run failed", message: errMsg(err) });
    }
  }

  async function syncConnectionAfterImport() {
    if (!session) return;
    const [tables, columns, count] = await Promise.all([
      listTables(session.session_id),
      getTableColumns(session.session_id, wizard.targetTable),
      getRowCount(session.session_id, wizard.targetTable).catch(() => null),
    ]);
    setMode(wizard.side, "database");
    setTables(wizard.side, tables);
    setSelectedTable(wizard.side, wizard.targetTable);
    setColumns(wizard.side, columns);
    setColumnMapping(wizard.side, wizard.mapping);
    setRowCount(wizard.side, count ?? null);
    loadPersistedConnection(wizard.side)
      .then((rec) => {
        if (!rec) return;
        rec.table = wizard.targetTable;
        rec.column_mapping = wizard.mapping;
        return savePersistedConnection(wizard.side, rec);
      })
      .catch(() => {});
  }

  async function pollImportJob(jobId: string): Promise<CsvImportJobDto> {
    const delays = [500, 750, 1000, 1500, 2000];
    let tick = 0;
    for (;;) {
      const job = await getCsvImportStatus(jobId);
      patch({ job });
      if (TERMINAL_IMPORT_PHASES.includes(job.phase)) {
        return job;
      }
      const delay = delays[Math.min(tick, delays.length - 1)] ?? 2000;
      tick += 1;
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }

  async function commit() {
    if (!session || !wizard.dryRun) return;
    const request = buildRequest(session.session_id, database, {
      planHash: wizard.dryRun.plan_hash,
    });
    if (!request) return;
    patch({
      importRunning: true,
      importStatus: "running",
      error: null,
      step: "commit",
      job: null,
      jobId: null,
    });
    try {
      const started = await startCsvImport(request);
      patch({ job: started, jobId: started.job_id });
      const finalJob = await pollImportJob(started.job_id);
      patch({
        job: finalJob,
        importRunning: false,
        importStatus: "terminal",
        step: "done",
      });
      if (finalJob.phase === "complete") {
        await syncConnectionAfterImport();
        pushToast({
          tone: "success",
          title: "CSV imported",
          message: `${wizard.targetTable} is selected for ${wizard.side}`,
        });
      } else if (finalJob.phase === "cancelled") {
        pushToast({
          tone: "warn",
          title: "Import cancelled",
          message: finalJob.message ?? "Import stopped before completion.",
        });
      } else {
        pushToast({
          tone: "error",
          title: "Import failed",
          message: finalJob.error ?? "Import failed before completion.",
        });
      }
    } catch (err) {
      patch({
        importRunning: false,
        importStatus: "terminal",
        error: errMsg(err),
      });
      pushToast({ tone: "error", title: "Import failed", message: errMsg(err) });
    }
  }

  async function cancelImport() {
    if (!wizard.jobId) return;
    patch({ importStatus: "cancelling" });
    try {
      await cancelCsvImport(wizard.jobId);
      const job = await getCsvImportStatus(wizard.jobId);
      patch({
        job,
        importRunning: false,
        importStatus: "terminal",
      });
      pushToast({
        tone: "warn",
        title: "Cancelling import",
        message: job.message ?? "Stopping after the current batch.",
      });
    } catch (err) {
      patch({ importStatus: "running" });
      pushToast({ tone: "error", title: "Cancel failed", message: errMsg(err) });
    }
  }

  function handleClose() {
    if (wizard.importRunning) {
      const ok = window.confirm(
        "An import is still running. Cancel the import and close the wizard?",
      );
      if (!ok) return;
      void cancelImport().finally(() => close());
      return;
    }
    close();
  }

  function buildRequest(
    sessionId: string,
    db: string,
    options: { requireMapping?: boolean; planHash?: string } = {
      requireMapping: true,
    },
  ): CsvImportRequestDto | null {
    if (!wizard.targetTable.trim()) {
      patch({ error: "Target table is required" });
      return null;
    }
    if (!wizard.filePath.trim()) {
      patch({ error: "CSV file is required" });
      return null;
    }
    if (!isCsvPath(wizard.filePath)) {
      patch({ error: "Choose a .csv or .txt file, not a folder" });
      return null;
    }
    if (!wizard.mapping && options.requireMapping !== false) {
      patch({ error: "Preview CSV first, then map the detected columns" });
      return null;
    }
    return {
      target: {
        session_id: sessionId,
        database: db,
        table: wizard.targetTable.trim(),
        mode: wizard.targetMode,
      },
      file: {
        path: wizard.filePath,
        encoding: wizard.encoding,
        delimiter: wizard.delimiter,
        date_format: wizard.dateFormat,
      },
      mapping: wizard.mapping ?? blankMapping(),
      policy: {
        id_behavior: wizard.idBehavior,
        duplicate_behavior: wizard.duplicateBehavior,
        duplicate_key: wizard.duplicateKey,
        batch_size: wizard.batchSize,
        create_indexes: wizard.createIndexes,
        confirmed_destructive: wizard.confirmedDestructive,
      },
      plan_hash: options.planHash ?? null,
    };
  }

  const busy =
    wizard.previewLoading || wizard.dryRunLoading || wizard.importRunning;
  const commitDisabled =
    !wizard.dryRun ||
    wizard.dryRun.invalid_rows > 0 ||
    (wizard.targetMode === "replace" && !wizard.confirmedDestructive) ||
    busy;

  return (
    <div
      className="fixed inset-0 z-50 bg-black/70 p-4 overflow-y-auto"
      role="presentation"
    >
      <Card
        className="max-w-5xl mx-auto space-y-5 shadow-2xl"
        role="dialog"
        aria-modal="true"
        aria-label={`Import CSV to ${capitalize(wizard.side)} database`}
      >
        <div className="flex items-start justify-between gap-3">
          <SectionHeader
            title={`Import CSV to ${capitalize(wizard.side)} Database`}
            description="Create or update a MySQL table, then select it for matching."
          />
          <Button tone="ghost" onClick={handleClose} disabled={wizard.importStatus === "cancelling"}>
            Close
          </Button>
        </div>

        {!canUseDb && (
          <div className="help-error">
            Connect the {wizard.side} database first. Import uses the active
            session and never stores the database password.
          </div>
        )}

        <div className="grid lg:grid-cols-3 gap-4">
          <Card className="space-y-3" padded={false}>
            <SectionHeader
              title="1. Target"
              description={`Database: ${database || "not connected"}`}
            />
            <Field label="Target table" required>
              <input
                className="input"
                value={wizard.targetTable}
                onChange={(e) =>
                  patch({
                    targetTable: e.target.value,
                    dryRun: null,
                    job: null,
                    error: null,
                  })
                }
                placeholder="imported_beneficiaries"
              />
            </Field>
            <Field label="Table mode">
              <select
                className="select"
                value={wizard.targetMode}
                onChange={(e) =>
                  patch({
                    targetMode: e.target.value as typeof wizard.targetMode,
                    dryRun: null,
                    job: null,
                  })
                }
              >
                <option value="create">Create new table</option>
                <option value="append">Use existing table</option>
                <option value="replace">Replace existing rows</option>
              </select>
            </Field>
            {wizard.targetMode === "replace" && (
              <Toggle
                checked={wizard.confirmedDestructive}
                onChange={(confirmedDestructive) =>
                  patch({ confirmedDestructive, dryRun: null })
                }
                label="I understand this will truncate the destination table"
              />
            )}
          </Card>

          <Card className="space-y-3" padded={false}>
            <SectionHeader
              title="2. CSV"
              description="Preview is read-only and does not change the database."
            />
            <Field label="CSV file" required>
              <div className="flex gap-2">
                <input
                  className="input flex-1"
                  value={wizard.filePath}
                  onChange={(e) =>
                    patch({
                      filePath: e.target.value,
                      preview: null,
                      mapping: null,
                      dryRun: null,
                    })
                  }
                  placeholder="C:/data/beneficiaries.csv"
                />
                <Button onClick={chooseCsv}>Browse</Button>
              </div>
            </Field>
            <div className="grid grid-cols-3 gap-2">
              <Field label="Encoding">
                <select
                  className="select"
                  value={wizard.encoding ?? ""}
                  onChange={(e) =>
                    patch({
                      encoding: (e.target.value || null) as CsvEncodingDto | null,
                      preview: null,
                      dryRun: null,
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
                  value={wizard.delimiter ?? ""}
                  onChange={(e) =>
                    patch({
                      delimiter: (e.target.value || null) as CsvDelimiterDto | null,
                      preview: null,
                      dryRun: null,
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
                  value={wizard.dateFormat}
                  onChange={(e) =>
                    patch({
                      dateFormat: e.target.value,
                      preview: null,
                      dryRun: null,
                    })
                  }
                />
              </Field>
            </div>
            <Button
              tone="primary"
              loading={wizard.previewLoading}
              disabled={!canUseDb}
              onClick={preview}
            >
              Preview CSV
            </Button>
          </Card>

          <Card className="space-y-3" padded={false}>
            <SectionHeader
              title="3. Policies"
              description="Dry run validates these choices before commit."
            />
            <Field label="ID behavior">
              <select
                className="select"
                value={wizard.idBehavior}
                onChange={(e) =>
                  patch({
                    idBehavior: e.target.value as typeof wizard.idBehavior,
                    dryRun: null,
                  })
                }
              >
                <option value="use-csv-id">Use CSV ID</option>
                <option value="generate-id">Generate numeric IDs</option>
                <option value="db-auto-increment">Use DB auto-increment</option>
                <option value="use-csv-uuid">Use CSV UUID</option>
                <option value="generate-uuid">Generate UUID</option>
              </select>
            </Field>
            <Field label="Duplicate handling">
              <select
                className="select"
                value={wizard.duplicateBehavior}
                onChange={(e) =>
                  patch({
                    duplicateBehavior:
                      e.target.value as typeof wizard.duplicateBehavior,
                    dryRun: null,
                  })
                }
              >
                <option value="skip">Skip duplicates</option>
                <option value="update">Update duplicates</option>
                <option value="insert-anyway">Insert anyway</option>
                <option value="fail">Fail import</option>
              </select>
            </Field>
            <Field label="Duplicate key">
              <select
                className="select"
                value={wizard.duplicateKey}
                onChange={(e) =>
                  patch({
                    duplicateKey: e.target.value as typeof wizard.duplicateKey,
                    dryRun: null,
                  })
                }
              >
                <option value="id">ID</option>
                <option value="uuid">UUID</option>
                <option value="matcher-fields">First + last + birthdate</option>
              </select>
            </Field>
            <Field label="Batch size">
              <input
                type="number"
                className="input tabular"
                min={1}
                max={200000}
                value={wizard.batchSize}
                onChange={(e) =>
                  patch({
                    batchSize: Number(e.target.value || 1),
                    dryRun: null,
                  })
                }
              />
            </Field>
            <Toggle
              checked={wizard.createIndexes}
              onChange={(createIndexes) => patch({ createIndexes, dryRun: null })}
              label="Create matching indexes after import"
            />
          </Card>
        </div>

        {wizard.preview && (
          <Card className="space-y-4">
            <div className="flex items-center justify-between">
              <SectionHeader
                title="4. Column Mapping"
                description="Map CSV columns to the matcher fields used by database matching."
              />
              <Pill tone="info">
                {wizard.preview.headers.length} columns ·{" "}
                {wizard.preview.rows.length} preview rows
              </Pill>
            </div>
            <ColumnMapper
              columns={columnsFromHeaders(wizard.preview.headers)}
              value={wizard.mapping}
              onChange={(mapping) =>
                patch({ mapping, dryRun: null, job: null, error: null })
              }
            />
            <PreviewTable preview={wizard.preview} />
          </Card>
        )}

        {wizard.error && (
          <div role="alert" className="help-error">
            {wizard.error}
          </div>
        )}

        <div className="flex flex-wrap gap-2">
          <Button
            tone="secondary"
            loading={wizard.dryRunLoading}
            disabled={!wizard.preview || !wizard.mapping || !canUseDb || busy}
            onClick={dryRun}
          >
            Run Dry Run
          </Button>
          <Button
            tone="primary"
            loading={wizard.importRunning}
            disabled={commitDisabled}
            onClick={commit}
          >
            Commit Import
          </Button>
          {wizard.importRunning && wizard.jobId && (
            <Button
              tone="secondary"
              disabled={wizard.importStatus === "cancelling"}
              onClick={cancelImport}
            >
              Cancel Import
            </Button>
          )}
          {wizard.step === "done" && !wizard.importRunning && (
            <Button tone="secondary" onClick={handleClose}>
              Done
            </Button>
          )}
        </div>

        {wizard.dryRun && <DryRunSummary result={wizard.dryRun} />}
        {wizard.job && <JobSummary job={wizard.job} />}
      </Card>
    </div>
  );
}

function DryRunSummary({
  result,
}: {
  result: NonNullable<ReturnType<typeof useCsvImportStore.getState>["dryRun"]>;
}) {
  return (
    <Card className="space-y-3">
      <SectionHeader
        title="Dry Run Summary"
        description="Commit is enabled only when invalid rows are cleared."
      />
      <div className="grid sm:grid-cols-4 gap-2 text-sm">
        <Metric label="Total" value={result.total_rows} />
        <Metric label="Valid" value={result.valid_rows} />
        <Metric label="Invalid" value={result.invalid_rows} danger />
        <Metric label="Duplicates" value={result.duplicate_rows} />
        <Metric label="New" value={result.new_rows} />
        <Metric label="Skipped" value={result.skipped_rows} />
        <Metric label="Updated" value={result.updated_rows} />
        <Metric label="Batches" value={result.estimated_batches} />
      </div>
      <div className="flex flex-wrap gap-2">
        {result.planned_indexes.map((idx) => (
          <Pill key={idx.name} tone={idx.unique ? "warn" : "info"}>
            {idx.name}: {idx.columns.join(", ")}
          </Pill>
        ))}
      </div>
      {(result.duplicate_probe_status ||
        result.staging_table ||
        result.load_method) && (
        <dl className="text-xs text-ink-300 grid sm:grid-cols-2 gap-2">
          {result.duplicate_probe_status && (
            <div>
              <dt className="text-ink-500">Duplicate probe</dt>
              <dd>{result.duplicate_probe_status}</dd>
            </div>
          )}
          {result.staging_table && (
            <div>
              <dt className="text-ink-500">Staging table</dt>
              <dd className="font-mono">{result.staging_table}</dd>
            </div>
          )}
          {result.load_method && (
            <div>
              <dt className="text-ink-500">Load method</dt>
              <dd>{formatLoadMethod(result.load_method)}</dd>
            </div>
          )}
        </dl>
      )}
      {result.warnings.length > 0 && (
        <ul className="text-sm text-warn-300 list-disc list-inside">
          {result.warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}
      {result.invalid_samples.length > 0 && (
        <ul className="text-sm text-danger-300 list-disc list-inside">
          {result.invalid_samples.map((row) => (
            <li key={`${row.row_number}-${row.reason}`}>
              Row {row.row_number}: {row.reason}
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

function JobSummary({ job }: { job: NonNullable<ReturnType<typeof useCsvImportStore.getState>["job"]> }) {
  const batchPct =
    job.total_batches > 0
      ? Math.min(100, Math.round((job.current_batch / job.total_batches) * 100))
      : 0;
  const rowPct =
    job.total_rows > 0
      ? Math.min(100, Math.round((job.processed_rows / job.total_rows) * 100))
      : 0;
  const tone =
    job.phase === "complete"
      ? "ok"
      : job.phase === "failed"
        ? "danger"
        : job.phase === "cancelled"
          ? "warn"
          : "info";
  return (
    <Card className="space-y-3">
      <SectionHeader title="Import Progress" description={job.message ?? undefined} />
      <div className="grid sm:grid-cols-4 gap-2 text-sm">
        <Metric label="Processed" value={job.processed_rows} />
        <Metric label="Inserted" value={job.inserted_rows} />
        <Metric label="Updated" value={job.updated_rows} />
        <Metric label="Skipped" value={job.skipped_rows} />
      </div>
      {job.total_batches > 0 && (
        <div className="space-y-1">
          <div className="text-2xs text-ink-500">
            Batch {job.current_batch} / {job.total_batches} ({batchPct}%)
          </div>
          <div className="h-2 rounded bg-ink-900 overflow-hidden">
            <div
              className="h-full bg-accent-500 transition-all"
              style={{ width: `${batchPct}%` }}
            />
          </div>
        </div>
      )}
      {job.total_rows > 0 && (
        <div className="text-2xs text-ink-500">
          Rows {job.processed_rows.toLocaleString()} / {job.total_rows.toLocaleString()} ({rowPct}%)
        </div>
      )}
      <Pill tone={tone}>{job.phase}</Pill>
      {(job.partial_commit != null ||
        job.destructive_step_completed != null ||
        job.staging_table ||
        job.load_method) && (
        <dl className="text-xs text-ink-300 grid sm:grid-cols-2 gap-2">
          {job.partial_commit != null && (
            <div>
              <dt className="text-ink-500">Partial commit</dt>
              <dd>{job.partial_commit ? "Yes" : "No"}</dd>
            </div>
          )}
          {job.destructive_step_completed != null && (
            <div>
              <dt className="text-ink-500">Destructive step done</dt>
              <dd>{job.destructive_step_completed ? "Yes" : "No"}</dd>
            </div>
          )}
          {job.staging_table && (
            <div>
              <dt className="text-ink-500">Staging table</dt>
              <dd className="font-mono">{job.staging_table}</dd>
            </div>
          )}
          {job.load_method && (
            <div>
              <dt className="text-ink-500">Load method</dt>
              <dd>{formatLoadMethod(job.load_method)}</dd>
            </div>
          )}
        </dl>
      )}
      {job.error && <div className="help-error">{job.error}</div>}
    </Card>
  );
}

function formatLoadMethod(method: "batched-insert" | "load-data-infile"): string {
  return method === "load-data-infile" ? "LOAD DATA INFILE" : "Batched INSERT";
}

function Metric({
  label,
  value,
  danger,
}: {
  label: string;
  value: number;
  danger?: boolean;
}) {
  return (
    <div className="surface-soft p-2">
      <div className="text-2xs text-ink-500">{label}</div>
      <div className={danger && value > 0 ? "text-danger-300" : "text-ink-100"}>
        {value.toLocaleString()}
      </div>
    </div>
  );
}

function PreviewTable({ preview }: { preview: CsvPreviewDto }) {
  return (
    <div className="overflow-x-auto border border-ink-800 rounded-lg">
      <table className="min-w-full text-xs">
        <thead>
          <tr className="bg-ink-900/70">
            {preview.headers.map((header) => (
              <th key={header} className="text-left px-2 py-1 text-ink-300">
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {preview.rows.map((row, idx) => (
            <tr key={idx} className="border-t border-ink-800/70">
              {preview.headers.map((header, colIdx) => (
                <td key={`${header}-${colIdx}`} className="px-2 py-1">
                  {row[colIdx] ?? ""}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
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
  if (!first_name || !last_name || !birthdate) return null;
  return {
    id,
    uuid: pick(headers, ["uuid"]) || null,
    first_name,
    middle_name: pick(headers, ["middle_name", "middlename", "mname"]) || null,
    last_name,
    birthdate,
    hh_id: pick(headers, ["hh_id", "household_id"]) || null,
  };
}

function blankMapping(): ColumnMappingDto {
  return {
    id: "",
    uuid: null,
    first_name: "",
    middle_name: null,
    last_name: "",
    birthdate: "",
    hh_id: null,
  };
}

function isCsvPath(path: string): boolean {
  const clean = path.trim().toLowerCase();
  return clean.endsWith(".csv") || clean.endsWith(".txt");
}

function pick(headers: string[], hints: string[]): string {
  const normalized = new Map(headers.map((header) => [normalize(header), header]));
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

function capitalize(value: string) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function errMsg(err: unknown): string {
  if (typeof err === "object" && err && "message" in err) {
    return String((err as { message: unknown }).message);
  }
  return String(err);
}
