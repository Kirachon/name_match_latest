# Plan: Post-Migration Feature Roadmap

**Generated**: 2026-05-24
**Plan ID**: `feature-roadmap-2026-05-24`
**Status**: Council-reviewed, ready for execution
**Reviewers**: Backend Architect, Frontend Developer, Sprint Prioritizer (Product)
**Estimated Effort**: ~2-3 weeks for v1.1, ~4-5 weeks for v1.2 (one developer)

---

## Council Critical Decisions

1. **Feature D (Embedded HTTP API) DEFERRED** — flagged overengineered; if real demand, ship as a headless CLI binary (D1) instead of embedded server (D2).
2. **Feature B SPLIT** — CSV (B1) ships in v1.1, Excel (B2) deferred to v1.2 (formula handling + memory + serial dates is its own complexity).
3. **Feature F SCOPED** — Quick Match explanation (F1) ships in v1.1 with on-demand recompute; Cascade/GPU explanation (F2) deferred indefinitely (CUDA kernels don't return per-pair breakdowns).
4. **ResultStore → SQLite migration ADDED as prerequisite** — Phase 2 work mentioned in store.rs comments is now an explicit hard dependency for Features C and E.
5. **Component extraction ADDED as prerequisite** — ConnectTab (571 LOC), ResultsTab (430 LOC), ConfigureTab (723 LOC) are too large; extract before piling on more features.

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
- Component extraction prerequisite (ConnectTab/ResultsTab/SchemaQuality)

**In scope (v1.2)**

- ResultStore → per-job SQLite migration (P1-P3)
- Stable ID strategy for file-sourced jobs (P4)
- Feature B2 — Excel import (.xlsx/.xls, formula fallback, sheet picker)
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
| X2 | Extract `SchemaQuality` from ConnectTab | — | `ui/src/features/connect/SchemaQuality.tsx` (new) | No behavior change |
| X3 | Extract `ResultsTable` from ResultsTab | — | `ui/src/features/results/ResultsTable.tsx` (new) | ResultsTab < 250 lines, virtualization intact |
| X4 | Extract `MatchRow` from ResultsTable (reusable for Diff view later) | X3 | `ui/src/features/results/MatchRow.tsx` (new) | Used in both Results and (future) Diff |

---

## v1.1 Features

---

### Feature A: Column Mapping UI

**Release**: v1.1 (ship first)
**Effort**: ~1-2 days

**Refined Approach**

- Expose `ColumnMapping` via Tauri command with `Option<ColumnMapping>` per `TableSelectionDto` (source + target independently). Backend reviewer's recommendation prevents a breaking-change issue when Feature B lands.
- UI lives inside ConnectTab below the table picker. Frontend reviewer's placement: keeps the existing IA, no new tab.
- Dropdown per logical name with fuzzy auto-suggest from raw columns (e.g. typing "fn" suggests `fname`, `firstname`, `first_name`).
- Persist mapping inside the existing `PersistedConnection` JSON (not ephemeral). Resolves Backend reviewer's persistence question without needing the SQLite migration yet.
- Validation: keep existing `val()` for DB columns; add a separate `val_csv()` path that allows spaces/special chars (but sanitizes injection chars). This unblocks Feature B without retrofitting later.

**Disagreements & Resolution**

- None. All three reviewers agreed: lowest risk, highest ROI, ship first.

**Atomic Tasks**

| # | Task | depends_on | Location | Validation |
|---|------|-----------|----------|------------|
| A1 | Add `column_mapping: Option<ColumnMapping>` to `TableSelectionDto` | [] | `src/run_service/dto.rs` | Existing tests pass; serde round-trip test |
| A2 | Add `save_column_mapping` / `load_column_mapping` Tauri commands | [A1] | `src-tauri/src/commands/database.rs` | Invoke from devtools returns Ok; persistence round-trips |
| A3 | Add `val_csv()` validator (allows spaces, rejects injection chars) | [] | `src/db/schema.rs` | Unit tests for edge cases (spaces, special chars, injection attempts) |
| A4 | Build `ColumnMapper.tsx` component | [X1] | `ui/src/features/connect/ColumnMapper.tsx` (new) | Renders dropdowns with fuzzy suggest, emits mapping change |
| A5 | Integrate ColumnMapper into ConnectTab + connectionStore | [A2, A4] | `ui/src/features/connect/ConnectTab.tsx`, `ui/src/shared/stores/connectionStore.ts` | Mapping persists across app restart |
| A6 | Wire mapping into `build_select_list()` SQL builder | [A1] | `src/db/schema.rs` | Run with remapped columns produces correct SQL; integration test |



---

### Feature B1: CSV Import (Excel deferred to v1.2)

**Release**: v1.1
**Effort**: ~3-5 days

**Refined Approach**

- **CSV only in v1.1**. Excel (B2) deferred to v1.2 — Backend reviewer's split recommendation accepted by all three reviewers. Excel formulas, memory caps, and Excel serial dates are their own complexity.
- Add a `DataSourceSwitcher` (Database | File) at the top of ConnectTab. **"Database" remains default** — mitigates Frontend reviewer's IA-disruption concern.
- Tab label stays "1. Connect" in v1.1 (rename to "1. Data Source" only when Excel ships).
- **Encoding**: auto-detect with `chardetng` crate; show detected encoding in preview with override dropdown (UTF-8 / UTF-8 BOM / Windows-1252 / Latin-1).
- **Delimiter**: auto-detect (try comma, semicolon, tab); show in preview with override.
- **Date format**: configurable per-file with common presets (`YYYY-MM-DD`, `MM/DD/YYYY`, `DD/MM/YYYY`) + custom strftime pattern.
- **ID generation**: row index (1-based) by default, but allow user to designate an ID column. Resolves Backend reviewer's ID-stability concern for Feature E (diffs require stable IDs across re-imports).
- Preview first 5 rows after file drop (Frontend reviewer's UX pattern).
- Reuses `ColumnMapper.tsx` from Feature A.

**Disagreements & Resolution**

- Product reviewer scored B as Risk 2; Backend reviewer scored it HIGH. **Resolution**: accept Backend's assessment — scope to CSV-only for v1.1, which brings risk back to Medium. Excel complexity moves to B2/v1.2.

**Atomic Tasks**

| # | Task | depends_on | Location | Validation |
|---|------|-----------|----------|------------|
| B1.1 | Add `csv` (already a dep) + `chardetng` crate dependencies | [] | `Cargo.toml` | Compiles |
| B1.2 | Implement `CsvLoader` (encoding detect, delimiter detect, date parse) | [A3] | `src/loaders/csv_loader.rs` (new module) | Unit tests: UTF-8, Windows-1252, semicolon-delimited, MM/DD/YYYY dates |
| B1.3 | Add `load_csv_preview` Tauri command (first 5 rows + headers) | [B1.2] | `src-tauri/src/commands/files.rs` (new) | Returns preview JSON |
| B1.4 | Add `start_csv_run` Tauri command (full load → `Vec<Person>` → engine) | [B1.2, A1] | `src-tauri/src/commands/files.rs` | Produces MatchPairDto results identical to DB-sourced run on equivalent data |
| B1.5 | Build `FileSourceCard.tsx` + `DataSourceSwitcher.tsx` | [A5] | `ui/src/features/connect/` | Drag-drop works, preview renders, encoding/delimiter override functional |
| B1.6 | Add `dataSource` / `filePath` / `idColumn` to SideState in connectionStore | [] | `ui/src/shared/stores/connectionStore.ts` | State persists across navigation |
| B1.7 | End-to-end integration test: CSV file → column map → run → export | [B1.4, B1.5] | `tests/csv_e2e.rs` (new) | Pass on Windows-1252 + UTF-8 fixtures |

---

### Feature F1: Match Explanation (Quick Match only — on-demand recompute)

**Release**: v1.1
**Effort**: ~3-4 days

**Refined Approach**

- **On-demand recompute**, not pre-stored — Backend reviewer's recommendation. When user clicks a row, invoke `explain_pair(source_id, target_id, job_id)` Tauri command that:
  1. Retrieves the two `Person` records (from DB connection if still alive, or from cached `Vec<Person>` if file-sourced).
  2. Re-runs `fuzzy_compare_names_new` with a new `ScoreBreakdown` return variant.
  3. Returns the breakdown to the frontend in <1 ms.
- Scoped to **Quick Match path only** (F1). Cascade/GPU explanation (F2) deferred — Backend reviewer confirms GPU kernels can't return per-pair breakdowns without rewrite.
- UI: right-side panel (320 px wide) on row click — Frontend reviewer's design. Flex layout, table shrinks; close button restores full width.
- No memory tax on bulk runs. `ScoreBreakdown` is computed for one pair at a time.

**Disagreements & Resolution**

- Frontend reviewer ranked F as priority #2 (above B). Product reviewer ranked it #3 (in v1.1 alongside A+B). Backend reviewer flagged it as HIGH complexity. **Resolution**: all agree it belongs in v1.1, but scope to F1 (Quick Match on-demand only). This drops complexity from HIGH to MEDIUM. F2 (Cascade/GPU) is explicitly deferred indefinitely.
- Product reviewer said "data already computed internally" (Risk 2). Backend reviewer correctly noted it's *discarded* after scoring. **Resolution**: accept Backend's on-demand recompute approach — avoids the 80-200 MB memory tax of pre-storing breakdowns on every pair.

**Atomic Tasks**

| # | Task | depends_on | Location | Validation |
|---|------|-----------|----------|------------|
| F1.1 | Define `ScoreBreakdown` struct (Lev/JW/Metaphone scores, case triggered, swap used) | [] | `src/matching/mod.rs` | Compiles, serde round-trip test |
| F1.2 | Modify `fuzzy_compare_names_new` to optionally return `ScoreBreakdown` (via new fn) | [F1.1] | `src/matching/mod.rs` | Existing match tests still pass; new test verifies breakdown values match expected |
| F1.3 | Add `explain_pair` Tauri command (loads 2 persons, calls scorer, returns breakdown) | [F1.2] | `src-tauri/src/commands/results.rs` | Returns correct breakdown for known test pair |
| F1.4 | Keep source `Vec<Person>` accessible post-run for file-sourced jobs | [B1.4] | `src/run_service/store.rs` | Persons retrievable by ID after run completes (cache survives until job evicted) |
| F1.5 | Build `ExplanationPanel.tsx` + `ScoreBreakdown.tsx` | [X3] | `ui/src/features/results/` | Panel renders with mock data; close button works |
| F1.6 | Wire row click → `explain_pair` → panel | [F1.3, F1.5] | `ui/src/features/results/ResultsTab.tsx` | Click row → panel shows real breakdown within 100 ms |



---

## v1.2 Features

---

### Prerequisite: ResultStore → SQLite Migration (P1-P3 + P4)

**Release**: v1.2 (first task in the wave)
**Effort**: ~3-4 days

All three reviewers agreed this is a hard dependency for Features C and E. Backend reviewer noted it's already mentioned in `store.rs` comments as Phase 2 work. Without it:
- Review decisions (Feature C) are orphaned on app restart.
- Diff between runs (Feature E) requires both jobs in memory simultaneously.

| # | Task | depends_on | Location | Validation |
|---|------|-----------|----------|------------|
| P1 | Design per-job SQLite schema (results + summary metadata + decisions table for C) | [] | `src/run_service/store.rs` | Schema reviewed; migration script idempotent |
| P2 | Implement SQLite write-through (results persist on job complete) | [P1] | `src/run_service/store.rs` | Restart app → previous job results loadable via `summary()` |
| P3 | Migrate `ResultStore::snapshot()` to read from SQLite when evicted from memory | [P2] | `src/run_service/store.rs` | 51st job still accessible after the first 50 evict from memory |
| P4 | Add stable ID strategy for CSV-sourced jobs (user-designated ID column or content-hash) | [B1.2] | `src/loaders/csv_loader.rs` | Re-import same file → same IDs (deterministic) |

---

### Feature B2: Excel Import

**Release**: v1.2
**Effort**: ~2-3 days

**Refined Approach**

- Add `calamine` crate. Handle: sheet selection UI, formula → cached-value fallback, Excel serial date conversion, memory cap (reject files > 200 MB with warning).
- Reuses `FileSourceCard` + `ColumnMapper` from B1.
- Sheet picker dropdown added to `FileSourceCard` when file extension is `.xlsx` or `.xls`.

**Atomic Tasks**

| # | Task | depends_on | Location | Validation |
|---|------|-----------|----------|------------|
| B2.1 | Add `calamine` dependency | [] | `Cargo.toml` | Compiles |
| B2.2 | Implement `ExcelLoader` (sheet select, formula fallback, serial-date → NaiveDate, memory guard) | [B1.2] | `src/loaders/excel_loader.rs` (new) | Tests: formula cells, serial dates, > 200 MB rejection |
| B2.3 | Add sheet picker to `FileSourceCard` | [B2.2] | `ui/src/features/connect/FileSourceCard.tsx` | Shows sheets when .xlsx selected; selection works |
| B2.4 | Integration test: .xlsx → map → run → export | [B2.2] | `tests/excel_e2e.rs` (new) | Pass on a multi-sheet fixture with formulas + dates |

---

### Feature C: Match Review Workflow

**Release**: v1.2
**Effort**: ~3-4 days

**Refined Approach**

- Review decisions stored as a table in the **same per-job SQLite DB** (not a separate sidecar) — Backend reviewer's recommendation. Eliminates the two-source-of-truth problem.
- Decisions are **per-job** (PK includes `job_id`). Cross-job carry-forward is a v1.3+ enhancement if users request it.
- Confidence band is **configurable** (default 70-85%, adjustable in ConfigureTab Export card) — Backend reviewer's recommendation. Static 70-85% may capture zero pairs on some datasets.
- UI: inline ✓/✗ buttons in ResultsTab with a `ReviewToolbar` (Frontend reviewer's design). Keyboard shortcuts: A (accept), R (reject), ↓ (next pending).
- Export path filters by decision before passing to `csv_export`/`xlsx_export` — no signature change needed, just pre-filter the `&[MatchPair]` slice.

**Disagreements & Resolution**

- Backend reviewer wanted full Phase 2 store migration before C. Product reviewer didn't mention it. **Resolution**: accept Backend — ResultStore→SQLite (P1-P3) is an explicit prerequisite in v1.2, scheduled before C.
- Backend reviewer raised "70-85 % may capture zero pairs." **Resolution**: make the band configurable + show count preview ("12 pairs need review") before entering review mode.

**Atomic Tasks**

| # | Task | depends_on | Location | Validation |
|---|------|-----------|----------|------------|
| C1 | Add `decisions` table to per-job SQLite schema | [P2] | `src/run_service/store.rs` | Migration runs, table exists, indexes on (job_id, source_id, target_id) |
| C2 | Add `save_decision` / `get_decisions` Tauri commands | [C1] | `src-tauri/src/commands/results.rs` | Round-trip test |
| C3 | Add configurable review band to `RunConfigDto` (`review_band: Option<(f32, f32)>`) | [] | `src/run_service/dto.rs` | Serde test; default 70-85 |
| C4 | Build `ReviewToolbar.tsx` + `ReviewActions.tsx` (✓/✗ buttons, keyboard handler) | [X3] | `ui/src/features/results/` | Renders, emits events, keyboard A/R/↓ work |
| C5 | Build `reviewStore.ts` (Zustand, syncs to SQLite via commands) | [C2, C4] | `ui/src/shared/stores/reviewStore.ts` (new) | Decisions persist across app restart |
| C6 | Filter export by decisions (pre-filter slice in CSV/XLSX writers) | [C1] | `src/export/csv_export.rs`, `src/export/xlsx_export.rs` | Export excludes rejected pairs; auto-accepts above band included |
| C7 | Add review band config UI to ConfigureTab Export card | [C3] | `ui/src/features/configure/ConfigureTab.tsx` | Band sliders persist |

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
| E1 | Implement `ResultStore::diff()` over SQLite | [P3] | `src/run_service/store.rs` | Unit test: known diff produces correct added/removed/changed counts |
| E2 | Add `diff_jobs` Tauri command | [E1] | `src-tauri/src/commands/results.rs` | Returns DiffResult JSON |
| E3 | Build `DiffView.tsx` + `DiffTable.tsx` | [X4] | `ui/src/features/results/` | Renders 3-column layout with green/red/amber color coding |
| E4 | Wire Compare button + job pickers in ResultsTab | [E2, E3] | `ui/src/features/results/ResultsTab.tsx` | End-to-end: pick 2 jobs → see diff |

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
[X1-X4 Component Extraction] ──────────────────────────────────────┐
        │                                                          │
        ▼                                                          ▼
┌──── v1.1 ────────────────────────────────────┐    ┌──── v1.2 ────────────────────┐
│                                              │    │                              │
│  A1─A2─A3─A6 (Column Mapping backend)        │    │  P1─P2─P3 (Store→SQLite)     │
│  A4─A5 (Column Mapping frontend)             │    │      │         │             │
│      │                                       │    │      ▼         ▼             │
│      ▼                                       │    │  C1─C2─C5─C6  E1─E2          │
│  B1.1─B1.2─B1.3─B1.4 (CSV backend)           │    │  C3─C4─C7     E3─E4          │
│  B1.5─B1.6 (CSV frontend)                    │    │                              │
│  B1.7 (integration)                          │    │  P4 (stable IDs)             │
│                                              │    │      │                       │
│  F1.1─F1.2─F1.3 (Explanation backend)        │    │  B2.1─B2.2─B2.3─B2.4 (Excel) │
│  F1.4 (person retention)                     │    │                              │
│  F1.5─F1.6 (Explanation frontend)            │    └──────────────────────────────┘
│                                              │
└──────────────────────────────────────────────┘
```

---

## Parallel Execution Waves

| Wave | Tasks | Can Start When | Notes |
|------|-------|----------------|-------|
| 0 | X1, X2, X3, X4 | Immediately | Component extraction prerequisites; ~1 day |
| 1.1a | A1, A3, B1.1 | X-wave done | Backend foundations; parallel |
| 1.1b | A2, A4, B1.2, F1.1 | A1 / X1 / B1.1 done | More backend + UI scaffolds |
| 1.1c | A5, A6, B1.3, B1.4, F1.2, F1.3 | upstream done | Wiring + commands |
| 1.1d | B1.5, B1.6, B1.7, F1.4, F1.5, F1.6 | upstream done | Frontend + integration |
| **v1.1 ship** | | All v1.1 tasks done | ~2-3 weeks single dev |
| 1.2a | P1, P2, P3, P4 | v1.1 done | Store migration; sequential |
| 1.2b | B2.1, B2.2, C1, C3, E1 | P-wave done | Backend foundations |
| 1.2c | B2.3, C2, C4, C5, C6, C7, E2, E3 | upstream done | Commands + UI |
| 1.2d | B2.4, E4 | upstream done | Integration |
| **v1.2 ship** | | All v1.2 tasks done | ~4-5 weeks single dev |

---

## Testing Strategy

- **Unit (Rust)**: B1.2 adds CSV loader tests (encodings, delimiters, dates); F1.2 adds breakdown-correctness tests; P1-P3 add SQLite migration tests; E1 adds diff-correctness tests; A3 adds validator edge cases.
- **Unit (TS)**: extends the 5-test scaffold from optimization-sweep-plan T11. Add tests for `ColumnMapper` fuzzy suggest, `FileSourceCard` drag-drop, `reviewStore` decision persistence, `DiffView` rendering.
- **Integration (E2E manual)**:
  - v1.1: CSV file → map columns → run Quick Match → click row → see breakdown → export
  - v1.2: Run twice → compare → reject borderline pairs → re-export with decisions applied
- **Static**: ESLint + clippy gates from optimization-sweep-plan must stay green.
- **Evidence demanded by ship gate**:
  - All Wave 0 component extractions verified visually identical to before (screenshot diff)
  - v1.1: At least one full E2E pass on UTF-8 + Windows-1252 fixtures with mismatched column names
  - v1.2: At least one full E2E pass with review decisions persisting across app restart

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|---------|-----------|
| Phase 2 SQLite store is unlisted work blocking C+E | **High** | Explicitly scheduled as first v1.2 task (P1-P3) |
| CSV ID instability breaks diffs across re-imports | Medium | P4: require ID column or content-hash; warn on row-index mode |
| F1 requires Person data post-run; DB connection may close | Medium | F1.4: cache source persons in memory for file-sourced jobs; re-query for DB jobs |
| CSV encoding/date edge cases generate support load | Medium | Ship with auto-detect + manual override; log parse warnings visibly |
| ResultsTab touched by C + E + F simultaneously | Medium | Component extraction (Wave 0) prerequisite; feature-flag each addition |
| Excel formula cells return formula text instead of value | Medium | B2.2: prefer cached-value, fall back to formula-eval, warn user when fallback used |
| `chardetng` mis-detects encoding | Low | Manual override dropdown always available in preview |
| Building D wastes effort without demand | Low | Deferred; revisit criteria defined |

---

## Council Feedback Summary

**Backend Architect**
- Flagged CSV/Excel as HIGH complexity (not Risk 2 as Product had it); resolution split into B1 (v1.1) and B2 (v1.2)
- Identified Phase 2 SQLite migration as hard prerequisite for C and E; added as P1-P4 prereq
- Recommended on-demand recompute for F1 instead of pre-storing breakdowns; saves 80-200 MB per run
- Flagged CSV ID instability as Feature E blocker; introduced P4 stable-ID strategy
- Pushed for component extraction before piling on more features
- Recommended killing F2 (CUDA kernels can't return breakdowns)
- Recommended killing D2 (embedded HTTP), keeping only D1 (headless CLI) if demand validates

**Frontend Developer**
- Recommended `DataSourceSwitcher` instead of replacing the Connect tab (preserves IA)
- Designed `ExplanationPanel` as right-side flex panel (not modal/tooltip)
- Designed inline ✓/✗ + `ReviewToolbar` for review workflow with A/R/↓ keyboard shortcuts
- Re-prioritized F above B from a UX impact perspective; resolution is they ship together in v1.1
- Pushed for component extraction (Wave 0) before adding any feature work to oversized files

**Sprint Prioritizer (Product)**
- Confirmed v1.1 = A + B + F is the right sprint scope; v1.2 = C + E + (B2)
- **Flagged D as overengineered** — defer until validated demand
- Confirmed F2 (Cascade/GPU explanation) has unclear ROI for the audit use case
- Cross-job decision carry-forward deferred to v1.3+ unless requested

**Cross-cutting consensus**
- All three: ResultStore → SQLite migration is the missing prerequisite that makes v1.2 viable
- All three: component extraction must happen before feature work
- All three: D should be a CLI flag on the existing binary, not a Tauri-embedded server

---

## Operator Quick-Start

If you only have one week, ship **v1.1** (Features A + B1 + F1 + Wave 0 prerequisites):

1. **Wave 0** (X1-X4) — component extraction, ~1 day
2. **Feature A** — column mapping, ~1-2 days
3. **Feature B1** — CSV import, ~3 days
4. **Feature F1** — match explanation, ~2 days
5. **Verification** — E2E pass on at least one CSV fixture with mismatched columns, ~half day

That single week delivers the highest-impact features: anyone with any data source can use the tool, and every match is explainable.

---

**Plan saved to**: `docs/feature-roadmap-plan.md`
**Plan ID**: `feature-roadmap-2026-05-24`
**Council format**: parallel-task / swarm-planner compatible — feed this file directly to `/parallel-task` to execute Wave 0.
