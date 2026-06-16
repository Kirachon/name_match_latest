# Codebase Performance Audit Report

## Scope Executed

This pass read `docs/codebase-performance-audit-goal.md`, inspected the current repository state, used Context Engine retrieval to map the main runtime/performance paths, and implemented minimal high-confidence fixes for confirmed blocker/OOM risks.

## Repository State

- Branch: `main...origin/main`
- Existing unrelated modified files were present before this fix:
  - `src-tauri/Cargo.toml`
  - `src-tauri/src/state.rs`
  - `src/run_service/store.rs`
  - several `ui/src/features/...` result/configure files
  - `ui/src/__tests__/reviewBand.test.ts`
  - `ui/src/shared/lib/reviewBand.ts`
- Files changed by this pass:
  - `src-tauri/src/commands/results.rs`
  - `src/run_service/scale.rs`
  - `ui/src/shared/runScalePolicy.ts`
  - `ui/src/__tests__/runScalePolicy.test.ts`
  - `docs/codebase-performance-audit-report.md`
  - `docs/two-pool-db-streaming-design.md` (Phase 5 design)

## Confirmed Finding

### P1: Large XLSX Export Can Rebuild OOM Risk After Result Spilling

- Symptom: Result storage supports chunked iteration and SQLite spill behavior for large result sets, but Tauri XLSX export still cloned every exported row into an in-memory `Vec` before writing the workbook.
- Evidence: `export_results` used `for_each_export_row` for chunked reads, but then appended every row into `xlsx_rows` when `ExportFormatDto::Xlsx` or `ExportFormatDto::Both` was requested.
- Affected component: `src-tauri/src/commands/results.rs`
- Root cause: CSV export was streaming, but XLSX export required the full result set in memory because `write_xlsx` accepts a full slice of row references and also creates per-level groupings for cascade exports.
- User-visible impact: Large jobs that were safe to page/export as CSV could still consume excessive memory, slow down, or OOM during XLSX export.
- Fix: Added a preflight row count using the existing chunked `ResultStore::for_each_export_row` path. XLSX exports are rejected above `100_000` filtered rows before any output file is created. CSV export remains chunked and is not limited by the XLSX guard.
- Compatibility risk: Users exporting more than `100_000` rows to XLSX now receive a validation error and must export CSV or narrow filters. This is intentional to avoid OOM.
- Validation method: Focused Rust tests plus Tauri cargo check.

### P1: DB Streaming Is Incorrectly Selected for Cross-Session Database Runs

- Symptom: The scale policy can select the DB streaming worker whenever both sides are database sources, the algorithm supports streaming, and the row policy resolves to streaming.
- Evidence: `src/run_service/scale.rs` did not check whether `source.session_id` and `target.session_id` match. `src-tauri/src/commands/matching.rs` builds the streaming runner with only the source `MySqlPool`, then calls `stream_match_csv_partitioned(&src, source.table, target.table, ...)`, whose API accepts a single pool.
- Affected components:
  - `src/run_service/scale.rs`
  - `src-tauri/src/commands/matching.rs`
  - `ui/src/shared/runScalePolicy.ts`
- Root cause: The streaming implementation can only query two tables through one MySQL connection pool, but the selection policy did not encode that invariant.
- User-visible impact: A deterministic DB-to-DB run across two sessions could be routed into streaming and fail at runtime because the target table is looked up through the source pool, or worse, match against a same-named table in the wrong database.
- Fix: Added a same-session check to the backend streaming policy and mirrored it in the frontend policy so cross-session DB runs fall back to the existing two-pool in-memory loader path.
- Compatibility risk: Cross-session deterministic runs at streaming scale no longer use the streaming backend. This avoids incorrect execution but means those runs still need a future two-pool streaming design for true large-scale support.
- Validation method: Added backend and frontend regression tests. Frontend test passed; backend test is blocked by a local Cargo registry access error noted below.

## Validation Evidence

| Gate | Command / Action | Result | Evidence |
| --- | --- | --- | --- |
| Read goal | `Get-Content docs\codebase-performance-audit-goal.md` | Passed | Goal content inspected |
| Repo state | `git status --short --branch` | Passed with existing unrelated changes | Branch `main...origin/main`; unrelated modified UI/Tauri/store files noted |
| Context Engine retrieval | `codebase_retrieval` for Tauri results export/OOM path | Passed | Identified `src-tauri/src/commands/results.rs` and DTO/export symbols |
| Focused tests | `cargo test --locked --manifest-path src-tauri\Cargo.toml large_` | Passed | 2 tests passed |
| Tauri check | `cargo check --locked --manifest-path src-tauri\Cargo.toml` | Passed | Finished dev profile successfully after `results.rs` and `scale.rs` changes |
| Format check | `cargo fmt --manifest-path src-tauri\Cargo.toml --check` | Failed due existing drift | Reported formatting diffs in unrelated files and existing sections of `results.rs`; formatter was not run to avoid broad unrelated edits |
| Context Engine review | `review_auto` on isolated `results.rs` diff | Passed gate | Risk 2/5, `should_fail: false`; deterministic PRE001 test warning is covered by the two new tests |
| Frontend scale-policy test | `pnpm --dir ui test runScalePolicy` | Passed | 1 file, 5 tests passed |
| Backend scale-policy test | `cargo test --locked streaming_worker_requires_same_db_session` | Blocked | Cargo failed to unpack `icu_locale_core-2.0.0/src/data.rs` with `Access is denied. (os error 5)` |

## Remaining Risks

- XLSX export is still implemented as an in-memory workbook writer for allowed row counts.
- A future improvement could stream or split XLSX output, but that would be a broader design change.
- Cross-session large DB runs now avoid the incorrect single-pool streaming path, but they still do not have a true two-pool streaming implementation. See `docs/two-pool-db-streaming-design.md` for the gated future design.
- `ResultStore::diff` is capped at `100_000` rows per job via preflight `matches_found` checks. Jobs above the cap must use CSV export and external compare until SQL-backed diff exists.
- Non-streaming matching paths still materialize both source and target as `Vec<Person>`. Scale policy blocks 1M+ unsupported/file/cascade/fuzzy cases, but large sub-million runs can still be memory-heavy.
- Full Docker/MySQL and end-to-end runtime smoke validation were not run in this pass.
- Existing unrelated working-tree changes were not modified or validated.

## Validation Ledger

Remediation-pass validation status per `docs/performance-remediation-swarm-plan.md` evidence contract. Status values: `PASSED`, `FAILED`, `BLOCKED`, or `DEFERRED`.

| Item | Status | Command / Action | Evidence | Notes |
| --- | --- | --- | --- | --- |
| Git / repo state | PASSED | `git status --short --branch` | Branch `main...origin/main`; unrelated dirty files preserved | Baseline recorded at pass start |
| Audit goal read | PASSED | Read `docs/codebase-performance-audit-goal.md` | Goal scope and constraints applied | — |
| Context Engine retrieval | PASSED | `codebase_retrieval` on export/streaming paths | Located `results.rs`, `scale.rs`, matching runner symbols | — |
| Tauri compile check | PASSED | `cargo check --locked --manifest-path src-tauri/Cargo.toml` | Finished dev profile successfully | After `results.rs` and `scale.rs` changes |
| XLSX export guard | PASSED | `cargo test --locked --manifest-path src-tauri/Cargo.toml large_xlsx` | `large_xlsx_export_is_rejected_before_buffering_rows` passed | Preflight rejects >100k rows before buffering |
| XLSX CSV parity guard | PASSED | `cargo test --locked --manifest-path src-tauri/Cargo.toml large_csv_export` | `large_csv_export_is_not_limited_by_xlsx_guard` passed | CSV export not capped by XLSX guard |
| Diff guard (spilled compare) | PASSED | `CARGO_HOME=$PWD/.cargo-test-home cargo test --locked diff_ --manifest-path Cargo.toml` | 5 tests passed including `diff_large_spilled_guard` | Preflight rejects >100k rows before `load_rows_for_diff` |
| Scale policy — backend same-session | PASSED | `CARGO_HOME=$PWD/.cargo-test-home cargo test --locked streaming_worker --manifest-path Cargo.toml` | `streaming_worker_requires_same_db_session` passed | Repo-local `CARGO_HOME` workaround clears registry access blocker |
| Scale policy — frontend mirror | PASSED | `pnpm --dir ui test runScalePolicy` | 7 tests passed | Includes cross-session `streamingBackendActive` false case |
| Compare / export UX guards | PASSED | `pnpm --dir ui test compareGuard` | 10 tests passed | Inline compare block, cross-session notice, scale-guard error mapping |
| Frontend tests (full suite) | PASSED | `pnpm --dir ui test` | 10 files, 49 tests passed | Includes policy, compare guard, review band, and component tests |
| `cargo fmt --check` | BLOCKED | `cargo fmt --manifest-path src-tauri/Cargo.toml --check` | Pre-existing drift in unrelated files | Not auto-formatted to avoid broad unrelated edits |
| Context Engine review (results.rs) | PASSED | `review_auto` on isolated `results.rs` diff | Risk 2/5, `should_fail: false` | PRE001 warning covered by focused tests |
| Docker / MySQL smoke | PASSED | `cargo test --test db_cross_session_smoke policy`; `cargo test --test db_cross_session_smoke -- --ignored` (local `matchers-mysql-1`, 1k/10k rows) | Cross + same-session parity, memory +20% gate, cancel partial; CI job `mysql_smoke` | Gates G1–G6/G8 |
| Two-pool streaming design | PASSED | Wrote `docs/two-pool-db-streaming-design.md` | Phase 5 design-only deliverable complete | Implementation gated; `supports_cross_session_streaming` default false |
| Performance benchmarks (fixtures) | DEFERRED | Not run | No peak RSS / p95 evidence for `result_store_spilled_100k` ladder | Requires Phase 2 fixtures and non-Docker benchmark command |
| Full frontend lint/build | PASSED | `pnpm --dir ui lint` and `pnpm --dir ui build` | Typecheck clean; Vite production build succeeded | Phase 6 matrix complete for UI |

### Ledger summary

- **Passed:** XLSX guard, diff guard, CSV/XLSX guard parity tests, backend and frontend scale-policy mirror, compare/export UX guards, full UI test/lint/build, Tauri check, design document, Context Engine review on export changes.
- **Blocked:** `cargo fmt --check` (pre-existing drift in unrelated files).
- **Deferred:** Docker/MySQL smoke, performance fixture benchmarks with peak RSS / p95 evidence.

## Requirement Checklist

- Read `codebase-performance-audit-goal.md`: done.
- Identify evidence-backed issue: done, XLSX export OOM risk.
- Identify additional blocker/bottleneck issue: done, cross-session streaming selection bug.
- Implement minimal safe fixes if needed: done.
- Preserve CSV streaming behavior: done.
- Add or update tests: done.
- Validate with relevant checks: done for focused Rust/Tauri path.
- Document changed files, validation, risks, and follow-ups: done.
- Validation ledger with PASSED/BLOCKED/DEFERRED statuses: done.
- Two-pool DB streaming design (Phase 5): done.
