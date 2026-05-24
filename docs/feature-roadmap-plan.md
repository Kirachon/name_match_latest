# Plan: Post-Migration Feature Roadmap

**Generated**: 2026-05-24
**Plan ID**: `feature-roadmap-2026-05-24`
**Status**: Finalized after party-mode review; dependency-safe for parallel execution
**Reviewers**: Backend Architect, Frontend Developer, Sprint Prioritizer (Product), party-mode synthesis by Codex
**Estimated Effort**: ~2-3 weeks for v1.1, ~4-5 weeks for v1.2 (one developer)

---

## Council Critical Decisions

1. **Feature D (Embedded HTTP API) DEFERRED** — flagged overengineered; if real demand, ship as a headless CLI binary (D1) instead of embedded server (D2).
2. **Feature B SPLIT** — CSV (B1) ships in v1.1, Excel (B2) deferred to v1.2 (formula handling + memory + serial dates is its own complexity).
3. **Feature F SCOPED** — Quick Match explanation (F1) ships in v1.1 with on-demand recompute; Cascade/GPU explanation (F2) deferred indefinitely (CUDA kernels don't return per-pair breakdowns).
4. **ResultStore → SQLite migration ADDED as prerequisite** — Phase 2 work mentioned in store.rs comments is now an explicit hard dependency for Features C and E.
5. **Component extraction ADDED as prerequisite** — ConnectTab (571 LOC), ResultsTab (430 LOC), ConfigureTab (723 LOC) are too large; extract before piling on more features. Wave 0 is now split into dependency-safe subwaves so parallel-task execution does not create same-file collisions.
6. **Persistence ownership clarified** — Column mapping persists through the existing frontend `PersistedConnection` JSON path, not new backend save/load commands.
7. **File-source lifecycle clarified** — CSV/Excel runs must reuse the shared `RunService` lifecycle so cancellation, progress, events, result storage, export, and future SQLite behavior do not drift from database-backed runs.

---

## Release Waves

| Wave | Features | Sprint Goal | Effort |
|------|----------|-------------|--------|
| **v1.1** | A, B1, F1 | Usable for any analyst regardless of data source; results are explainable | ~2-3 weeks |
| **v1.2** | (P1-P4 prereq), C, E, B2 | Human-in-the-loop review, run-over-run validation, Excel support | ~4-5 weeks |
| **Deferred** | D (flagged overengineered), F2 (GPU kernels opaque) | Validate demand before building | — |

---

## Scope Lock

**In scope (v1.1)**

- Feature A — Column Mapping UI exposing existing `ColumnMapping` struct
- Feature B1 — CSV import (UTF-8 / Windows-1252, comma/semicolon/tab, configurable date formats)
- Feature F1 — Match Explanation for Quick Match path (on-demand recompute)
- Component extraction prerequisite (ConnectTab/ResultsTab/ConfigureTab/SchemaQuality)

**In scope (v1.2)**

- ResultStore → app-scoped SQLite migration with job-keyed tables (P1-P3)
- Stable ID strategy for file-sourced jobs (P4)
- Feature B2 — Excel import (.xlsx/.xls, cached formula values, sheet picker)
- Feature C — Match Review Workflow (per-job decisions in same SQLite)
- Feature E — Diff Between Runs (reads from SQLite)

**Deferred**

- Feature D — Embedded HTTP API (revisit only if ≥3 users request and CLI JSON mode insufficient)
- Feature F2 — Cascade/GPU match explanation (would require CUDA kernel rewrite)
- Cross-job carry-forward of review decisions (v1.3+ if requested)

---

## Prerequisites

### Tooling

- Rust 1.89.0 (already pinned)
- pnpm 10, Node 20+
- `$env:CARGO_HOME = "C:\cargo_nm_temp"` workaround on this host

### Component Extraction (Wave 0 — before any feature work)

All three reviewers flagged existing files as too large; extract first to prevent the new feature work from making them unmaintainable.

| # | Task | depends_on | Location | Validation |
|---|------|-----------|----------|------------|
| X1 | Extract `ConnectionCard` from ConnectTab | — | `ui/src/features/connect/ConnectionCard.tsx` (new) | ConnectTab < 150 lines, renders correctly, no behavior change |
| X2 | Extract `SchemaQuality` from ConnectTab | X1 | `ui/src/features/connect/SchemaQuality.tsx` (new) | No behavior change; avoids same-file merge conflict with X1 |
| X3 | Extract `ResultsTable` from ResultsTab | — | `ui/src/features/results/ResultsTable.tsx` (new) | ResultsTab < 250 lines, virtualization intact |
| X4 | Extract `MatchRow` from ResultsTable (reusable for Diff view later) | X3 | `ui/src/features/results/MatchRow.tsx` (new) | Used in both Results and (future) Diff |
| X5 | Extract ConfigureTab cards (`AlgorithmCard`, `StreamingCard`, `ExportCard`, `GpuCard`, `SummaryCard`) | — | `ui/src/features/configure/` (new card files) | ConfigureTab < 250 lines, no behavior change |

---

## v1.1 Features

---

### Feature A: Column Mapping UI

**Release**: v1.1 (ship first)
**Effort**: ~1-2 days

**Refined Approach**

- Expose `ColumnMapping` through `Option<ColumnMapping>` per `TableSelectionDto` (source + target independently). Backend reviewer's recommendation prevents a breaking-change issue when Feature B lands.
- UI lives inside ConnectTab below the table picker. Frontend reviewer's placement: keeps the existing IA, no new tab.
- Dropdown per logical name with fuzzy auto-suggest from raw columns (e.g. typing "fn" suggests `fname`, `firstname`, `first_name`).
- Persist mapping inside the existing frontend `PersistedConnection` JSON (not ephemeral). Do not add redundant backend save/load commands for this path.
- Validation: keep DB mapping on strict SQL identifier validation (`validate_ident` / mapped row fetch). CSV header validation belongs to Feature B1 because file headers are data keys, not SQL identifiers.

**Disagreements & Resolution**

- None. All three reviewers agreed: lowest risk, highest ROI, ship first.

**Atomic Tasks**

| # | Task | depends_on | Location | Validation |
|---|------|-----------|----------|------------|
| A1 | Add `column_mapping: Option<ColumnMapping>` to `TableSelectionDto` and mirror DTO/Zod/types | [] | `src/run_service/dto.rs`, `ui/src/shared/tauri/types.ts`, `ui/src/shared/tauri/zod-schemas.ts` | Existing tests pass; serde + Zod round-trip test |
| A2 | Version and hydrate column mapping in existing persisted connection JSON | [A1] | `ui/src/features/connect/persistence.ts`, `ui/src/shared/stores/connectionStore.ts` | Automated persistence round-trip; mapping survives app restart |
| A4 | Build `ColumnMapper.tsx` component | [X1] | `ui/src/features/connect/ColumnMapper.tsx` (new) | Renders dropdowns with fuzzy suggest, emits mapping change |
| A5 | Integrate ColumnMapper into ConnectTab + connectionStore | [A2, A4] | `ui/src/features/connect/ConnectTab.tsx`, `ui/src/shared/stores/connectionStore.ts` | Mapping persists across app restart |
| A6 | Wire `TableSelectionDto.column_mapping` through `start_matching` into mapped row fetch paths while preserving `extra_fields` | [A1] | `src-tauri/src/commands/matching.rs`, `src/db/schema.rs` | Run with remapped columns produces correct rows; integration test |



---

### Feature B1: CSV Import (Excel deferred to v1.2)

**Release**: v1.1
**Effort**: ~3-5 days

**Refined Approach**

- **CSV only in v1.1**. Excel (B2) deferred to v1.2 — Backend reviewer's split recommendation accepted by all three reviewers. Excel formulas, memory caps, and Excel serial dates are their own complexity.
- Add a `DataSourceSwitcher` (Database | File) at the top of ConnectTab. **"Database" remains default** — mitigates Frontend reviewer's IA-disruption concern.
- Tab label stays "1. Connect" in v1.1 (rename to "1. Data Source" only when Excel ships).
- **Encoding**: auto-detect with `chardetng`, decode with `encoding_rs`, and show detected encoding in preview with override dropdown (UTF-8 / UTF-8 BOM / Windows-1252 / Latin-1).
- **Delimiter**: auto-detect (try comma, semicolon, tab); show in preview with override.
- **Date format**: configurable per-file with common presets (`YYYY-MM-DD`, `MM/DD/YYYY`, `DD/MM/YYYY`) + custom strftime pattern.
- **ID generation**: user-designated ID column is the stable path. Row index is allowed in v1.1 with a warning, but not valid for run-over-run diff in v1.2.
- Preview first 5 rows after file drop (Frontend reviewer's UX pattern), with browse fallback, keyboard-accessible drop zone, parse warnings, invalid-file/permission states, loading/cancel states, and preview overflow handling.
- File-source state uses a discriminated source model (Database | File) or dedicated file source store, not DB-only `SideState` fields. Readiness, SummaryCard, and run-config building must understand both source types.
- CSV mappings validate selected headers against the parsed header list. Reject formula-injection export cells (`=`, `+`, `-`, `@`) through the existing export safety path before writing CSV/XLSX.
- Full CSV runs reuse the shared `RunService` lifecycle via a file-backed loader/table source. Do not create a separate matching lifecycle.
- Reuses `ColumnMapper.tsx` from Feature A.

**Disagreements & Resolution**

- Product reviewer scored B as Risk 2; Backend reviewer scored it HIGH. **Resolution**: accept Backend's assessment — scope to CSV-only for v1.1, which brings risk back to Medium. Excel complexity moves to B2/v1.2.

**Atomic Tasks**

| # | Task | depends_on | Location | Validation |
|---|------|-----------|----------|------------|
| B1.1 | Add `chardetng` + `encoding_rs` dependencies (`csv` already present) | [] | `Cargo.toml` | Compiles |
| B1.2 | Implement `CsvLoader` (encoding detect/decode, delimiter detect, date parse, header validation, parse warnings) | [B1.1] | `src/loaders/csv_loader.rs` (new module) | Unit tests: UTF-8, Windows-1252, semicolon-delimited, MM/DD/YYYY dates, invalid headers, formula-injection values |
| B1.3 | Add `load_csv_preview` Tauri command and register frontend/backend mirrors | [B1.2] | `src-tauri/src/commands/files.rs` (new), `src-tauri/src/commands/mod.rs`, `src-tauri/src/main.rs`, `ui/src/shared/tauri/commands.ts`, `ui/src/shared/tauri/types.ts`, `ui/src/shared/tauri/zod-schemas.ts` | Returns preview JSON with headers, warnings, detected encoding/delimiter |
| B1.4 | Add file-backed run source that reuses `RunService::start` lifecycle | [B1.2, A1] | `src/run_service/`, `src-tauri/src/commands/files.rs` | Produces MatchPairDto results identical to DB-sourced run on equivalent data; cancel/progress/store behavior matches DB path |
| B1.5 | Build `FileSourceCard.tsx` + `DataSourceSwitcher.tsx` | [A5, B1.6] | `ui/src/features/connect/` | Drag-drop + browse work, preview renders, overrides functional, warnings/errors/loading/cancel states covered |
| B1.6 | Add discriminated source state for Database vs File | [] | `ui/src/shared/stores/connectionStore.ts`, `ui/src/features/configure/ConfigureTab.tsx`, run config builders | State persists across navigation; readiness and summary support DB/file independently for source and target |
| B1.7 | End-to-end integration test: CSV file → column map → run → export | [A5, A6, B1.4, B1.5, B1.6] | `tests/csv_e2e.rs` (new) | Pass on Windows-1252 + UTF-8 fixtures with mismatched columns |

---

### Feature F1: Match Explanation (Quick Match only — on-demand recompute)

**Release**: v1.1
**Effort**: ~3-4 days

**Refined Approach**

- **On-demand recompute**, not pre-stored — Backend reviewer's recommendation. When user clicks a row, invoke `explain_pair(source_id, target_id, job_id)` Tauri command that:
  1. Resolves the two `Person` records from job-time person snapshots / lookup index plus stored job config, mapping, and file metadata. Do not rely on a still-live DB connection.
  2. Calls a public explanation API that mirrors the actual Quick Match scoring for `Fuzzy` and `FuzzyNoMiddle`.
  3. Returns the breakdown to the frontend with a tested UI SLA of <100 ms.
- Scoped to **Quick Match fuzzy paths only** (F1). Deterministic/Opt7 rows either get a separate simple explanation or a clear unsupported state. Cascade/GPU explanation (F2) deferred — Backend reviewer confirms GPU kernels can't return per-pair breakdowns without rewrite.
- UI: right-side panel (320 px wide) on row click — Frontend reviewer's design. Flex layout, table shrinks; close button restores full width. Row selection must be keyboard-accessible and include selected-row state, loading/error state, stale-request cancellation, focus return on close, and `aria-controls` / `aria-expanded`.
- No memory tax on bulk runs. `ScoreBreakdown` is computed for one pair at a time.

**Disagreements & Resolution**

- Frontend reviewer ranked F as priority #2 (above B). Product reviewer ranked it #3 (in v1.1 alongside A+B). Backend reviewer flagged it as HIGH complexity. **Resolution**: all agree it belongs in v1.1, but scope to F1 (Quick Match on-demand only). This drops complexity from HIGH to MEDIUM. F2 (Cascade/GPU) is explicitly deferred indefinitely.
- Product reviewer said "data already computed internally" (Risk 2). Backend reviewer correctly noted it's *discarded* after scoring. **Resolution**: accept Backend's on-demand recompute approach — avoids the 80-200 MB memory tax of pre-storing breakdowns on every pair.

**Atomic Tasks**

| # | Task | depends_on | Location | Validation |
|---|------|-----------|----------|------------|
| F1.1 | Define `ScoreBreakdown` struct (Lev/JW/Metaphone scores, case triggered, swap used) | [] | `src/matching/mod.rs` | Compiles, serde round-trip test |
| F1.2 | Add public explanation API for `Fuzzy` / `FuzzyNoMiddle` that returns `ScoreBreakdown` | [F1.1] | `src/matching/mod.rs` | Existing match tests still pass; breakdown values match expected scorer output |
| F1.3 | Keep job-time person snapshots / lookup index plus config/mapping metadata for explainability | [B1.4] | `src/run_service/store.rs`, `src/run_service/dto.rs` | Persons retrievable by source/target ID after run completes until job is evicted or loaded from SQLite |
| F1.4 | Add `explain_pair` Tauri command and frontend/backend mirrors | [F1.2, F1.3] | `src-tauri/src/commands/results.rs`, `src-tauri/src/commands/mod.rs`, `src-tauri/src/main.rs`, `ui/src/shared/tauri/commands.ts`, `ui/src/shared/tauri/types.ts`, `ui/src/shared/tauri/zod-schemas.ts` | Returns correct breakdown for known fuzzy pair; returns clear unsupported state for unsupported algorithms |
| F1.5 | Build `ExplanationPanel.tsx` + `ScoreBreakdown.tsx` | [X3] | `ui/src/features/results/` | Panel renders mock data, loading, error, unsupported, and close/focus states |
| F1.6 | Wire keyboard/click row selection → `explain_pair` → panel | [F1.4, F1.5] | `ui/src/features/results/ResultsTab.tsx`, `ui/src/features/results/ResultsTable.tsx` | Click/Enter row → panel shows real breakdown within 100 ms; stale responses ignored |



---

## v1.2 Features

---

### Prerequisite: ResultStore → SQLite Migration (P1-P3 + P4)

**Release**: v1.2 (first task in the wave)
**Effort**: ~3-4 days

All three reviewers agreed this is a hard dependency for Features C and E. Backend reviewer noted it's already mentioned in `store.rs` comments as Phase 2 work. Use one app-scoped SQLite database under the app data directory with job-keyed tables, not separate per-job database files; cross-job diff and job history become simpler and consistent. Without it:
- Review decisions (Feature C) are orphaned on app restart.
- Diff between runs (Feature E) requires both jobs in memory simultaneously.

| # | Task | depends_on | Location | Validation |
|---|------|-----------|----------|------------|
| P1 | Design app-scoped SQLite schema (jobs, results, result_person_lookup, decisions) with schema versioning, WAL, busy timeout, indexes, rollback/delete semantics | [] | `src/run_service/store.rs` | Schema reviewed; migrations idempotent; paging/sort/diff indexes documented |
| P2 | Implement transactional SQLite write-through when jobs complete | [P1] | `src/run_service/store.rs` | Restart app → previous job loadable via `summary()`, `list_summaries()`, and `page()` |
| P3 | Migrate all ResultStore read/delete/export surfaces to SQLite fallback when evicted from memory | [P2] | `src/run_service/store.rs`, `src-tauri/src/commands/results.rs` | 51st job remains accessible; `summary`, `list_summaries`, `page`, `export_results`, `forget_job`, and restart listing work |
| P4 | Add stable ID strategy for all file-sourced jobs (user-designated ID column or deterministic content key with collision checks) | [B1.2] | `src/loaders/csv_loader.rs`, `src/loaders/excel_loader.rs` | Re-import same file → same IDs; row-index mode warns and disables run-over-run diff |

---

### Feature B2: Excel Import

**Release**: v1.2
**Effort**: ~2-3 days

**Refined Approach**

- Add `calamine` crate. Handle: sheet selection UI, formula cached-value reads, Excel serial date conversion, memory cap (reject files > 200 MB with warning). Do not promise formula evaluation; warn/reject when cached values are missing or stale.
- Reuses `FileSourceCard` + `ColumnMapper` from B1.
- Sheet picker dropdown added to `FileSourceCard` when file extension is `.xlsx` or `.xls`.

**Atomic Tasks**

| # | Task | depends_on | Location | Validation |
|---|------|-----------|----------|------------|
| B2.1 | Add `calamine` dependency | [] | `Cargo.toml` | Compiles |
| B2.2 | Implement `ExcelLoader` (sheet select, cached formula values, serial-date → NaiveDate, memory guard) | [B1.2, P4] | `src/loaders/excel_loader.rs` (new) | Tests: cached formula values, missing-cache warning/reject, serial dates, > 200 MB rejection |
| B2.3 | Add sheet picker to `FileSourceCard` | [B2.2] | `ui/src/features/connect/FileSourceCard.tsx` | Shows sheets when .xlsx selected; selection works |
| B2.4 | Integration test: .xlsx → map → run → export | [B2.2] | `tests/excel_e2e.rs` (new) | Pass on a multi-sheet fixture with formulas + dates |

---

### Feature C: Match Review Workflow

**Release**: v1.2
**Effort**: ~3-4 days

**Refined Approach**

- Review decisions stored in the **same app-scoped SQLite DB** as job results (not a separate sidecar) — Backend reviewer's recommendation. Eliminates the two-source-of-truth problem.
- Decisions are **per-job** (PK includes `job_id`). Cross-job carry-forward is a v1.3+ enhancement if users request it.
- Confidence band is **configurable** (default 70-85%, adjustable in ConfigureTab Export card) — Backend reviewer's recommendation. Static 70-85% may capture zero pairs on some datasets.
- UI: inline Accept/Reject buttons in ResultsTab with a `ReviewToolbar` (Frontend reviewer's design). Keyboard shortcuts: A (accept), R (reject), Down (next pending) are scoped only when form inputs are not focused.
- Row click opens explanation; action-cell buttons handle Accept/Reject with `stopPropagation`, labels, and keyboard-safe event ownership.
- Export path filters by decision inside the Tauri `export_results` flow before calling `csv_export`/`xlsx_export`.
- Decision semantics: `accepted`, `rejected`, `pending`; upsert by `(job_id, source_id, target_id)`; validate the job exists before saving.

**Disagreements & Resolution**

- Backend reviewer wanted full Phase 2 store migration before C. Product reviewer didn't mention it. **Resolution**: accept Backend — ResultStore→SQLite (P1-P3) is an explicit prerequisite in v1.2, scheduled before C.
- Backend reviewer raised "70-85 % may capture zero pairs." **Resolution**: make the band configurable + show count preview ("12 pairs need review") before entering review mode.

**Atomic Tasks**

| # | Task | depends_on | Location | Validation |
|---|------|-----------|----------|------------|
| C1 | Add `decisions` table to app-scoped SQLite schema | [P2] | `src/run_service/store.rs` | Migration runs, table exists, indexes on (job_id, source_id, target_id) |
| C2 | Add `save_decision` / `get_decisions` Tauri commands and frontend/backend mirrors | [C1] | `src-tauri/src/commands/results.rs`, `src-tauri/src/commands/mod.rs`, `src-tauri/src/main.rs`, `ui/src/shared/tauri/commands.ts`, `ui/src/shared/tauri/types.ts`, `ui/src/shared/tauri/zod-schemas.ts` | Round-trip test; unknown job rejected |
| C3 | Add configurable review band to `RunConfigDto` (`review_band: Option<(f32, f32)>`) and TS/Zod mirrors | [] | `src/run_service/dto.rs`, `ui/src/shared/tauri/types.ts`, `ui/src/shared/tauri/zod-schemas.ts` | Serde + Zod test; default 70-85 |
| C4 | Build `ReviewToolbar.tsx` + `ReviewActions.tsx` (✓/✗ buttons, keyboard handler) | [X3] | `ui/src/features/results/` | Renders, emits events, keyboard A/R/↓ work |
| C5 | Build `reviewStore.ts` (Zustand, syncs to SQLite via commands) | [C2, C4] | `ui/src/shared/stores/reviewStore.ts` (new) | Decisions persist across app restart |
| C6 | Filter Tauri export by decisions before invoking CSV/XLSX writers | [C2, C5] | `src-tauri/src/commands/results.rs`, `src/export/csv_export.rs`, `src/export/xlsx_export.rs` | Export excludes rejected pairs; auto-accepts above band included |
| C7 | Add review band config UI to ConfigureTab Export card | [C3, X5] | `ui/src/features/configure/ExportCard.tsx` | Band sliders persist |

---

### Feature E: Diff Between Runs

**Release**: v1.2
**Effort**: ~2-3 days

**Refined Approach**

- `ResultStore::diff(job_a, job_b) -> DiffResult` reads from SQLite (post-migration). No need to hold both jobs in memory.
- Identity key: `(source_id, target_id)`. Changed threshold: ±2 % confidence delta.
- UI: Compare button in ResultsTab header (visible when ≥ 2 jobs exist). Two job pickers → 3-column DiffView (added/removed/changed). Reuses extracted `MatchRow` component from X4.
- Stable IDs (from P4) ensure cross-source diffs are meaningful.

**Disagreements & Resolution**

- Backend reviewer flagged CSV ID instability breaking diffs. **Resolution**: P4 (stable ID strategy) is an explicit prerequisite. Users must designate an ID column for CSV, or accept row-index IDs with a warning that re-imports invalidate prior diffs.

**Atomic Tasks**

| # | Task | depends_on | Location | Validation |
|---|------|-----------|----------|------------|
| E1 | Implement `ResultStore::diff()` over SQLite | [P3, P4] | `src/run_service/store.rs` | Unit test: known diff produces correct added/removed/changed counts; row-index file jobs rejected |
| E2 | Add `diff_jobs` Tauri command and frontend/backend mirrors | [E1] | `src-tauri/src/commands/results.rs`, `src-tauri/src/commands/mod.rs`, `src-tauri/src/main.rs`, `ui/src/shared/tauri/commands.ts`, `ui/src/shared/tauri/types.ts`, `ui/src/shared/tauri/zod-schemas.ts` | Returns DiffResult JSON |
| E3 | Build `DiffView.tsx` + `DiffTable.tsx` | [X4] | `ui/src/features/results/` | Renders added/removed/changed with text/icons plus color; stacks below tablet width |
| E4 | Wire Compare button + job pickers in ResultsTab | [E2, E3] | `ui/src/features/results/ResultsTab.tsx`, `ui/src/shared/stores/resultsStore.ts` | End-to-end: pick 2 jobs → see diff; disabled reason shown when <2 comparable jobs |

---

## Deferred Features

---

### Feature D: Embedded HTTP API — flagged overengineered

**Release**: Deferred (v1.3+ pending demand validation)

All three reviewers agreed this is the lowest priority. Product reviewer explicitly flags it as overengineered. Backend reviewer recommends splitting into D1 (headless CLI) + D2 (embedded in Tauri) and notes the real need is a standalone binary, not an embedded server.

**Resolution**: do not build now. The existing CLI already supports headless execution. If pipeline integration demand materializes (validated by user interviews), implement as:

- **D1 only**: `name_matcher --headless --port 8080 --token <auto>` — standalone binary, no Tauri dependency, binds 127.0.0.1, bearer-token auth, opt-in.
- **D2 (embedded in Tauri)** explicitly killed — lifecycle coupling and security surface aren't worth the convenience.

**Revisit criteria**: ≥ 3 users request programmatic access AND cannot use CLI JSON mode.

---

### Feature F2: Cascade / GPU Match Explanation

**Release**: Deferred indefinitely

GPU scoring happens in CUDA kernels that don't return per-pair breakdowns. Implementing F2 requires kernel rewrites with unclear ROI. Document GPU matches as "opaque confidence" in the UI. Revisit only if auditors explicitly require per-algorithm breakdown for cascade/GPU paths.

---

## Dependency Graph

```
[X1-X5 Component Extraction] ──────────────────────────────────────┐
        │                                                          │
        ▼                                                          ▼
┌──── v1.1 ────────────────────────────────────┐    ┌──── v1.2 ────────────────────┐
│                                              │    │                              │
│  A1─A2─A5 (Column Mapping persistence/UI)    │    │  P1─P2─P3 (Store→SQLite)     │
│  A1─A6 (Column Mapping backend wiring)       │    │      │         │             │
│  A4─A5 (Column Mapping frontend)             │    │      │         │             │
│      │                                       │    │      ▼         ▼             │
│      ▼                                       │    │  C1─C2─C5─C6  E1─E2          │
│  B1.1─B1.2─B1.3─B1.4 (CSV backend)           │    │  C3─C4─C7     E3─E4          │
│  B1.6─B1.5 (CSV frontend)                    │    │                              │
│  B1.7 (integration)                          │    │  P4 (stable IDs)             │
│                                              │    │      │                       │
│  F1.1─F1.2─F1.4 (Explanation backend)        │    │  B2.1─B2.2─B2.3─B2.4 (Excel) │
│  B1.4─F1.3 (person retention)                │    │                              │
│  F1.5─F1.6 (Explanation frontend)            │    └──────────────────────────────┘
│                                              │
└──────────────────────────────────────────────┘
```

---

## Parallel Execution Waves

| Wave | Tasks | Can Start When | Notes |
|------|-------|----------------|-------|
| 0a | X1, X3, X5 | Immediately | Independent component extraction; no same-file collisions |
| 0b | X2, X4 | X1 / X3 done | Follow-up extraction from files/components touched in 0a |
| 1.1a | A1, B1.1, B1.6, F1.1 | X-wave done | DTO/dependency/state foundations; parallel |
| 1.1b | A2, A4, A6, B1.2, F1.2 | A1 / X1 / B1.1 / F1.1 done | Backend wiring + UI scaffolds |
| 1.1c | A5, B1.3, B1.4, F1.3, F1.5 | upstream done | Commands, file lifecycle, person retention, UI panel |
| 1.1d | B1.5, F1.4, F1.6 | upstream done | File UI + explanation command/wiring |
| 1.1e | B1.7 | all v1.1 feature tasks done | Integration and fixture proof |
| **v1.1 ship** | | All v1.1 tasks done | ~2-3 weeks single dev |
| 1.2a | P1, P4 | v1.1 done | SQLite design and stable file IDs can proceed in parallel |
| 1.2b | P2 | P1 done | Transactional write-through |
| 1.2c | P3, B2.1, C3 | P2 / P4 as applicable | Store read fallback + backend foundations |
| 1.2d | B2.2, C1, C4, E1 | P3 / P4 / C3 as applicable | Excel loader, decisions, diff backend, review UI |
| 1.2e | B2.3, C2, C5, C6, C7, E2, E3 | upstream done | Commands + UI |
| 1.2f | B2.4, E4 | upstream done | Integration |
| **v1.2 ship** | | All v1.2 tasks done | ~4-5 weeks single dev |

---

## Testing Strategy

- **Unit (Rust)**: B1.2 adds CSV loader tests (encodings, delimiters, dates, header validation); F1.2 adds breakdown-correctness tests; P1-P3 add SQLite migration tests; E1 adds diff-correctness tests; A1/A6 add mapping DTO and mapped-row-fetch tests.
- **Unit (TS)**: extends the test scaffold from optimization-sweep-plan T11. Add Vitest/Testing Library tests for `ColumnMapper` fuzzy suggest, `FileSourceCard` drag-drop/browse/warnings, `ExplanationPanel` states/focus, `ReviewToolbar` keyboard scoping, `reviewStore` decision persistence, `DiffView` responsive rendering, and Tauri command mocks.
- **Integration (E2E manual)**:
  - v1.1: CSV file → map columns → run Quick Match → click row → see breakdown → export
  - v1.2: Run twice → compare → reject borderline pairs → re-export with decisions applied
- **Static / command gates**:
  - `$env:CARGO_HOME="C:\cargo_nm_temp"; cargo test --no-default-features`
  - `$env:CARGO_HOME="C:\cargo_nm_temp"; cargo clippy --no-default-features -- -D warnings`
  - `cargo test --manifest-path src-tauri/Cargo.toml`
  - `pnpm --dir ui test`
  - `pnpm --dir ui run lint`
  - `pnpm --dir ui run build`
  - Tauri smoke/build when command surfaces change.
- **Evidence demanded by ship gate**:
  - All Wave 0 component extractions verified visually identical to before (screenshot diff)
  - v1.1: default DB run still works; remapped DB columns work; CSV run works; mapping persists after restart; explanation panel works for DB and CSV Quick Match; unsupported algorithms show a clear explanation-disabled state; full E2E pass on UTF-8 + Windows-1252 fixtures with mismatched column names
  - v1.2: SQLite survives restart; evicted jobs remain readable; review decisions persist and affect export; diff works across re-imports using stable IDs; Excel formula/date/large-file cases are proven
  - Responsive screenshot/browser checks for desktop and narrow width, especially Results + explanation panel and Diff view

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|---------|-----------|
| SQLite store scope is inconsistent across per-job vs cross-job needs | **High** | Use one app-scoped SQLite DB with job-keyed tables, versioning, WAL, busy timeout, and indexed result access |
| File ID instability breaks diffs across re-imports | Medium | P4: require ID column or deterministic content key with collision checks; row-index mode disables diff |
| F1 requires Person data post-run; DB connection may close | Medium | F1.3: store job-time person snapshots / lookup index plus job config and mappings |
| CSV encoding/date edge cases generate support load | Medium | Ship with auto-detect + manual override; log parse warnings visibly |
| ResultsTab touched by C + E + F simultaneously | Medium | Component extraction (Wave 0) prerequisite; feature-flag each addition |
| Excel formula cells return formula text or stale values | Medium | B2.2: use cached values only; warn/reject when cached values are absent or stale |
| `chardetng` mis-detects encoding | Low | Manual override dropdown always available in preview |
| Building D wastes effort without demand | Low | Deferred; revisit criteria defined |

---

## Council Feedback Summary

**Backend Architect**
- Flagged CSV/Excel as HIGH complexity (not Risk 2 as Product had it); resolution split into B1 (v1.1) and B2 (v1.2)
- Identified Phase 2 SQLite migration as hard prerequisite for C and E; added as P1-P4 prereq
- Recommended on-demand recompute for F1 instead of pre-storing breakdowns; saves 80-200 MB per run
- Flagged file ID instability as Feature E blocker; introduced P4 stable-ID strategy
- Corrected persistence ownership: use existing frontend `PersistedConnection` JSON for mappings instead of backend save/load commands
- Corrected CSV/excel lifecycle: file runs must reuse `RunService` and shared result/export behavior
- Pushed for component extraction before piling on more features
- Recommended killing F2 (CUDA kernels can't return breakdowns)
- Recommended killing D2 (embedded HTTP), keeping only D1 (headless CLI) if demand validates

**Frontend Developer**
- Recommended `DataSourceSwitcher` instead of replacing the Connect tab (preserves IA)
- Designed `ExplanationPanel` as right-side flex panel (not modal/tooltip)
- Designed inline ✓/✗ + `ReviewToolbar` for review workflow with A/R/↓ keyboard shortcuts
- Added event ownership and accessibility requirements for row-click explanation vs inline review actions
- Added responsive and keyboard/focus requirements for FileSource, Explanation, Review, and Diff surfaces
- Re-prioritized F above B from a UX impact perspective; resolution is they ship together in v1.1
- Pushed for component extraction (Wave 0) before adding any feature work to oversized files, including ConfigureTab cards

**Sprint Prioritizer (Product)**
- Confirmed v1.1 = A + B + F is the right sprint scope; v1.2 = C + E + (B2)
- **Flagged D as overengineered** — defer until validated demand
- Confirmed F2 (Cascade/GPU explanation) has unclear ROI for the audit use case
- Cross-job decision carry-forward deferred to v1.3+ unless requested
- Tightened dependency graph and parallel waves so `/parallel-task` can run the plan without out-of-order X4 or same-file Wave 0 conflicts

**Cross-cutting consensus**
- All three: ResultStore → SQLite migration is the missing prerequisite that makes v1.2 viable
- All three: component extraction must happen before feature work
- All three: D should be a CLI flag on the existing binary, not a Tauri-embedded server
- All three: command/DTO tasks must update Rust registration plus frontend command, type, and Zod mirrors
- All three: v1.1/v1.2 ship gates need real restart, E2E, responsive, and unsupported-state evidence

---

## Operator Quick-Start

If you only have one week, ship **v1.1 core** and treat F1 as stretch unless Wave 0/A/B1 finish early:

1. **Wave 0** (X1-X5 split into 0a/0b) — component extraction, ~1-1.5 days
2. **Feature A** — column mapping, ~1-2 days
3. **Feature B1** — CSV import, ~3 days
4. **Feature F1** — match explanation, ~2 days if capacity remains
5. **Verification** — DB default + remapped DB + CSV E2E pass on at least one UTF-8 and one Windows-1252 fixture with mismatched columns, ~half day

That single week delivers the highest-impact features: anyone with any data source can use the tool, and every match is explainable.

---

**Plan saved to**: `docs/feature-roadmap-plan.md`
**Plan ID**: `feature-roadmap-2026-05-24`
**Council format**: parallel-task / swarm-planner compatible — execute Wave 0a first, then Wave 0b, then continue through the dependency-safe waves above.
