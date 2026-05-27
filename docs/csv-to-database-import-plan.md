# CSV-to-Database Import Setup Plan

**Status:** Recommended implementation plan
**Date:** 2026-05-25
**Scope:** Tauri desktop shell plus Rust import service for MySQL-backed matching sources
**Reviewed by:** Backend architecture, frontend implementation, and security/privacy subagents

## Executive Recommendation

Build CSV-to-database import as a dedicated "prepare database table from CSV" wizard, not as an extension of the current direct file matching path.

The existing app already has these useful pieces:

- MySQL connection/session commands in `src-tauri/src/commands/database.rs`.
- Password-safe session storage in `src-tauri/src/state.rs`, where the password is used to build a pool but is not kept in `DbSession`.
- CSV preview and column inference in `src/loaders/csv_loader.rs`.
- Source/target setup UI in `ui/src/features/connect/ConnectTab.tsx`.
- Database connection/table selection in `ui/src/features/connect/ConnectionCard.tsx`.
- File preview and matcher column mapping in `ui/src/features/connect/FileSourceCard.tsx` and `ColumnMapper.tsx`.
- Frontend command/type mirrors in `ui/src/shared/tauri/commands.ts` and `ui/src/shared/tauri/types.ts`.

The missing feature is a safe, guided, database-mutating import flow that turns a CSV into a reusable MySQL matching source.

## Feature Goals

The flow must support:

- Database host, port, database name, username, and password entry.
- Create-new-table or use-existing-table target choice.
- CSV upload/selection.
- Preview of detected headers, sample rows, parse warnings, inferred field types, and duplicate/invalid-row diagnostics.
- Column mapping from CSV columns to matcher fields: `id`, `uuid`, `first_name`, `middle_name`, `last_name`, `birthdate`, `hh_id`, and future optional fields.
- ID behavior: use CSV ID, generate app IDs, use database auto-increment, or use/generate UUID.
- Duplicate handling: skip, update, insert anyway, or fail.
- Mandatory dry-run/test preview before commit.
- Batch import with clear progress and error reporting.
- Matching indexes created after import, or validated/created explicitly for existing tables.
- Imported table saved/selected as a source or target matching table.
- No password exposure in logs, events, saved profiles, errors, progress, import jobs, or persisted source metadata.

## Recommended User Flow

1. Open `Connect` and choose `Import CSV to Database` for either Source or Target.
2. Connect to MySQL using host, port, database, username, and password.
3. Choose target mode:
   - Create a new table.
   - Append/use an existing table.
   - Replace/truncate only as an explicit destructive option, not a default.
4. Select CSV file.
5. Preview detected columns and sample rows.
6. Map CSV columns to matcher fields and decide what happens to unmapped CSV columns.
7. Choose ID behavior.
8. Choose duplicate behavior and duplicate key basis.
9. Run dry-run validation.
10. Review dry-run summary: row counts, invalid rows, duplicates, schema plan, index plan, and batch estimate.
11. Commit import in batches.
12. Create or validate indexes.
13. Refresh table list and set the imported table as the selected matching source/target.

## UX Architecture

Add a separate import wizard state machine instead of extending `FileSourceCard`.

Reason: `FileSourceCard` currently means "use this CSV/Excel file directly for matching." The new feature means "materialize this CSV into MySQL and then use the resulting table." Those are different operator mental models and different safety risks.

Recommended wizard steps:

- `connect`
- `target`
- `file`
- `mapping`
- `policies`
- `dry-run`
- `commit`
- `done`

Recommended UI integration:

- Add an `Import CSV to Database` action in `ConnectTab`.
- Reuse connection UI patterns from `ConnectionCard`.
- Reuse/extract CSV preview controls from `FileSourceCard`.
- Reuse/extend `ColumnMapper` for matcher fields.
- Add import-specific controls for target table, ID behavior, duplicate policy, batch size, and index options.
- Keep temporary import state separate from `connectionStore`; write back to `connectionStore` only after successful import.

## Frontend State Recommendation

Add a dedicated store or slice for import wizard state.

```ts
type ImportWizardStep =
  | "connect"
  | "target"
  | "file"
  | "mapping"
  | "policies"
  | "dry-run"
  | "commit"
  | "done";

interface CsvImportWizardState {
  side: "source" | "target";
  open: boolean;
  step: ImportWizardStep;
  sessionId: string | null;
  targetDatabase: string;
  targetTable: string;
  targetMode: "create" | "append" | "replace";
  filePath: string;
  encoding: CsvEncodingDto | null;
  delimiter: CsvDelimiterDto | null;
  dateFormat: string;
  mapping: ColumnMappingDto | null;
  idBehavior: "use-csv-id" | "generate-id" | "db-auto-increment" | "use-csv-uuid" | "generate-uuid";
  duplicateBehavior: "skip" | "update" | "insert-anyway" | "fail";
  duplicateKey: "id" | "uuid" | "matcher-fields";
  createIndexes: boolean;
  dryRun: CsvImportDryRunResultDto | null;
  job: CsvImportJobDto | null;
  loading: boolean;
  error: string | null;
}
```

Rules:

- Do not store raw password in import wizard state.
- If credentials are needed, keep password local to the connection form and pass it only to `connectDb`.
- Import commands must use `session_id` after connection.
- On success, update the existing side state with `mode: "database"`, selected table, columns, row count, and final column mapping.

## Backend Architecture

Split the backend into three contracts:

1. Import preview.
2. Import plan/dry-run validation.
3. Commit job.

Do not put all import logic directly in `src-tauri/src/commands/database.rs`. Keep Tauri commands thin and place import logic in core Rust so it can be unit-tested.

Recommended modules:

- `src/import/mod.rs`
- `src/import/csv_stream.rs`
- `src/import/schema_plan.rs`
- `src/import/duplicate_policy.rs`
- `src/import/index_plan.rs`
- `src-tauri/src/commands/import.rs`

Recommended DTO flow:

- Define Rust DTOs first in `src/run_service/dto.rs` or a dedicated import DTO module.
- Mirror DTOs in `ui/src/shared/tauri/types.ts`.
- Add wrappers in `ui/src/shared/tauri/commands.ts`.
- Update zod schemas if the project continues to use frontend runtime validation.

Recommended Tauri commands:

- `preview_csv_import(request)`.
- `validate_csv_import_plan(request)`.
- `start_csv_import(request)`.
- `get_csv_import_status(job_id)`.
- `cancel_csv_import(job_id)` if cancellation is supported.

All import commands after connection should take `session_id`, not credentials or password.

## Import Plan Semantics

Preview should be read-only.

It should not create tables, indexes, temp permanent objects, or partial data.

Dry-run validation should return:

- Total row estimate.
- Rows scanned in sample.
- Valid row count.
- Invalid row count.
- Duplicate estimate.
- New rows.
- Rows that would be skipped or updated.
- Sample invalid rows with row number and reason, without storing full row content.
- Proposed destination schema.
- Existing-table compatibility result.
- Planned indexes.
- Estimated batch count.
- Warnings requiring explicit acknowledgement.

Commit should run as a job with progress events:

- `creating-table`
- `importing`
- `creating-indexes`
- `validating`
- `refreshing-source`
- `complete`
- `failed`
- `cancelled`

Progress should include counts and phase details, not raw row values.

## ID Behavior

Support these behaviors:

- Use CSV ID column.
- Generate numeric IDs in the app.
- Use database auto-increment.
- Use CSV UUID.
- Generate UUID.

Validation rules:

- CSV ID mode requires mapped `id`.
- Database auto-increment requires a compatible target table or generated schema.
- Existing-table append must validate target column type, nullability, and uniqueness.
- Generated numeric IDs must define deterministic or database-owned behavior before implementation.
- Duplicate handling must declare which key is authoritative.

## Duplicate Handling

Support these modes:

- `skip`: ignore rows that conflict with the selected duplicate key.
- `update`: update existing matching rows.
- `insert-anyway`: insert all rows, only valid when target constraints permit it.
- `fail`: reject the import when duplicates are detected.

Duplicate basis must be explicit:

- `id`
- `uuid`
- matcher composite fields such as `first_name`, `last_name`, and `birthdate`

For update mode, the dry run must show which columns can be overwritten. Destructive update/replace behavior must require an explicit confirmation showing database, table, row count estimate, duplicate policy, and affected fields.

## Schema and Index Strategy

For new tables, generate a matcher-compatible schema:

- `id`
- `uuid`
- `first_name`
- `middle_name`
- `last_name`
- `birthdate`
- `hh_id`
- optional imported extra columns
- import metadata columns only if useful and approved, such as `imported_at`

For existing tables, validate:

- Required mapped fields exist or can be added if the operator chose schema modification.
- Types are compatible.
- Nullability is compatible with required fields.
- ID/UUID/unique behavior is compatible with selected policies.
- Existing indexes and constraints do not conflict with import behavior.

Recommended indexes:

- Primary/unique key for selected ID behavior.
- `uuid` if present.
- `birthdate`.
- `last_name`.
- `first_name`.
- `hh_id` if present.
- Composite index for common matching paths, such as `(last_name, first_name, birthdate)`.

Index creation timing:

- New table: import first, then create indexes for faster bulk load.
- Existing table: recommend indexes and require explicit opt-in for new index creation.
- If index creation fails after import succeeds, report the table as imported but degraded, not fully successful.

## Transaction and Batch Strategy

Use batch transactions by default for practical large-file behavior.

The UI and final report must make partial commit semantics clear:

- Total rows processed.
- Inserted count.
- Updated count.
- Skipped count.
- Failed count.
- Failed batch number.
- Whether prior batches were committed.
- Whether rollback occurred.

For high-safety mode, consider staging table import:

1. Create a staging table.
2. Load CSV rows into staging.
3. Validate staging.
4. Merge into destination according to duplicate policy.
5. Drop staging table only after success or operator review.

This reduces partial-write risk and makes duplicate handling more auditable.

## Security and Privacy Requirements

Password handling:

- Password may exist only in the connection form and `DbCredentialsDto` during connect/validate.
- Import commands must not accept or return password fields.
- `DbSessionDto`, import DTOs, events, logs, saved profiles, matching sources, and errors must never contain passwords.
- Existing optional password persistence should remain opt-in and warned; do not add password persistence to import profiles.
- Longer-term recommendation: replace plaintext `tauri-plugin-store` password persistence with OS keychain storage.

Logging:

- Add or reuse shared redaction for `password`, `passwd`, `pwd`, `secret`, `token`, and connection URLs.
- Do not log SQLx connection URLs or full connect options.
- Do not log raw row values from CSV imports.
- Database errors shown to the UI should be useful but sanitized.

Database safety:

- Centralize identifier validation/quoting for database, table, column, and index names.
- Use bound parameters for row values.
- Use strict identifier validation for dynamic SQL.
- Never default to destructive behavior.
- `DROP`, `TRUNCATE`, replace, overwrite, and schema alteration require explicit confirmation.
- Document least-privilege database permissions for operators.

PII:

- Names, birthdates, household IDs, UUIDs, file paths, and row samples are sensitive.
- Keep preview row count limited.
- Do not persist preview contents unless explicitly needed.
- Saved matching source metadata should reference table/mapping, not copied PII.

## Implementation Phases

### Phase 1: Contracts and Safety Foundation

- Add import DTOs in Rust.
- Mirror import DTOs in TypeScript.
- Add redaction helpers and tests.
- Centralize identifier validation/quoting for generated SQL.
- Define dry-run and commit result structures.

### Phase 2: Backend Preview and Dry Run

- Extend CSV parsing with streaming import diagnostics.
- Implement `preview_csv_import`.
- Implement `validate_csv_import_plan`.
- Validate target table mode, ID policy, duplicate policy, schema compatibility, and index plan.
- Add unit tests for invalid identifiers, missing required fields, date parsing, duplicate detection, and schema validation.

### Phase 3: Backend Commit Job

- Implement import job registry or reuse the existing job/event pattern.
- Implement batched insert/upsert/skip/fail logic.
- Implement create-table and index creation plans.
- Add progress events.
- Add failure reporting with partial commit semantics.
- Add tests for duplicate modes, batch failures, and index failure reporting.

### Phase 4: Frontend Wizard

- Add `CsvImportWizard` under `ui/src/features/connect/`.
- Add import wizard store/slice.
- Add command wrappers and DTOs.
- Reuse preview and mapping components where practical.
- Add ID, duplicate, target table, dry-run, and commit UI.
- Keep import state separate from matching source state until success.

### Phase 5: Source Registration and Workflow Integration

- On successful import, refresh `listTables(session_id)`.
- Select imported table on the chosen source/target side.
- Load `getTableColumns`.
- Load `getRowCount`.
- Apply final `columnMapping`.
- Confirm `readinessForRun` allows continuing to Configure.

### Phase 6: Validation and Documentation

- Add frontend unit/component tests.
- Add backend unit/integration tests.
- Add a small local MySQL test fixture if practical.
- Update user documentation with least-privilege DB setup and import workflow.
- Document password non-persistence and logging guarantees.

## Acceptance Criteria

- User can connect to MySQL with host, port, database, username, and password.
- Password is never present in import DTOs, saved import profiles, logs, events, progress, source metadata, or UI errors.
- User can create a new target table or select an existing table.
- User can select CSV, preview headers/sample rows/warnings, and map required matcher fields.
- User can choose ID behavior and incompatible choices are blocked before commit.
- User can choose duplicate handling and see the duplicate key basis.
- Dry-run is mandatory before commit.
- Dry-run reports row estimate, invalid rows, duplicates, schema issues, planned indexes, and batch estimate.
- Commit imports in batches and reports inserted, updated, skipped, failed, and partial/rollback status.
- Unsafe table/column/index names are rejected.
- Generated SQL binds values and validates/quotes identifiers.
- Index creation is explicit and reported.
- Imported table appears in table list and becomes selected as source/target.
- Existing direct CSV matching still works unchanged.

## Test Plan

Backend:

- DTO serialization round-trips.
- Identifier validation rejects unsafe database/table/column/index names.
- Preview is read-only and does not mutate database state.
- CSV parser handles encoding, delimiter, duplicate headers, invalid dates, and empty required fields.
- Dry-run validates schema compatibility and duplicate policy.
- Batch insert works for create-new table.
- Append with `skip`, `update`, `insert-anyway`, and `fail` behaves deterministically.
- Password redaction tests cover errors/loggable payloads.
- Failed batch reports partial commit or rollback state.

Frontend:

- Wizard step gating.
- Password input is masked.
- Import store does not contain password.
- Commit disabled until dry-run succeeds or required warnings are acknowledged.
- Duplicate required mappings are rejected.
- ID behavior validation works.
- Dry-run summary renders key counts and warnings.
- Commit progress renders phase and counts.
- Successful import updates selected table, columns, row count, and mapping.
- Cancel/close does not corrupt existing source/target selections.

E2E:

- Connect to test MySQL.
- Create table from CSV.
- Preview, map, dry-run, commit.
- Confirm table appears in `listTables`.
- Confirm table is selected and `Continue to Configure` is enabled.

## Party Mode Review Notes Incorporated

Backend review refined the plan into three backend-owned contracts: preview, validation plan, and commit job. It also added explicit large-file streaming, identifier safety, duplicate-key clarity, transaction semantics, and repo-specific integration points.

Frontend review recommended a dedicated import wizard state machine rather than overloading direct file matching state. It added the post-import source registration flow that updates the existing `connectionStore` only after success.

Security/privacy review added strict password boundaries, log redaction, read-only preview requirements, destructive action confirmation, PII handling, and least-privilege database guidance.

## Open Decisions Before Coding

- Whether Excel-to-database import is in scope now or CSV-only for the first release.
- Whether generated numeric IDs are app-owned or database-owned.
- Whether commit uses staging-table merge by default or only for high-safety mode.
- ~~Whether cancellation/resume is required in v1 or deferred.~~ **Resolved (2026-05-27):** v1 supports cooperative cancel between batches; see `docs/post-audit-remediation-plan.md`.
- Whether to remove existing plaintext password persistence or migrate it to OS keychain storage.
