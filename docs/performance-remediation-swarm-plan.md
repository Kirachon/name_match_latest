# Performance Remediation Swarm Plan

## Objective

Finish the codebase performance and reliability remediation for `name_match_latest` by making unsafe large-result paths impossible to select, validating the remaining safe paths with measurable evidence, and preparing a correctness-first design for future cross-session database streaming.

This plan incorporates feedback from three review subagents:

- Backend architecture review: make large-result safety an explicit invariant and audit every full-materialization consumer.
- Frontend review: make risky actions predictable and recoverable in the UI, not only rejected by backend errors.
- Performance review: define datasets, memory gates, and pass/fail thresholds before deeper implementation.

## Current Confirmed Fixes Already Applied

These fixes are already present in the working tree and must be preserved:

- `src-tauri/src/commands/results.rs`
  - Added XLSX export row guard: reject XLSX exports over `100_000` filtered rows before buffering.
  - CSV export remains chunked/streamed.
- `src/run_service/scale.rs`
  - Added same-session requirement before selecting DB streaming worker.
- `ui/src/shared/runScalePolicy.ts`
  - Mirrored same-session streaming policy in the UI.
- `ui/src/__tests__/runScalePolicy.test.ts`
  - Added frontend regression coverage for cross-session DB streaming policy.
- `docs/codebase-performance-audit-goal.md`
  - Saved audit goal.
- `docs/codebase-performance-audit-report.md`
  - Captures confirmed findings, fixes, validation, and remaining risks.

## Non-Negotiable Safety Invariants

- Spilled or large jobs must not call APIs that materialize all result rows unless a named cap allows it.
- Every large-result feature must be classified as one of:
  - SQL-backed
  - bounded by a hard cap
  - blocked with a clear user-facing error
- Cross-session DB runs must not select the current single-pool streaming worker.
- CSV export must remain available for large result sets.
- XLSX export over `100_000` filtered rows must fail before partial file creation.
- No secrets, credentials, or connection strings may be printed in logs, reports, or test output.

## Evidence And Benchmark Contract

Every validation run must record:

- Git commit or working-tree identifier
- Command and working directory
- Dataset name and row counts
- Dataset manifest hash where practical
- Wall-clock runtime
- Peak RSS or process memory where practical
- p50/p95/p99 runtime for repeated benchmarks
- Result count and stable pair-key correctness checks
- MySQL container/host/port readiness without exposing secrets
- Status: `PASSED`, `FAILED`, `BLOCKED`, or `DEFERRED`

Initial dataset ladder:

- `result_store_spilled_100k`
- `large_spilled_results_250k`
- `db_same_session_10k`
- `db_same_session_100k`
- `db_cross_session_blocked`
- `non_streaming_vec_policy_250k`
- `high_collision_birthdate`
- `1m_manual` for explicit manual-only stress validation

Initial pass/fail thresholds:

- Peak RSS must not regress more than `10%` for local non-DB paths.
- Peak RSS must not regress more than `20%` for DB streaming/import paths.
- Runtime p95 must not regress more than `10%` unless a memory-safety improvement is explicitly accepted.
- Candidate counts for fuzzy/cascade L10/L11 must not exceed baseline by more than `20%`.
- Result counts and stable pair keys must match baseline fixtures.
- GPU/cascade/fuzzy changes cannot be promoted without recall/parity evidence.

## Dependency Map

```text
P0 -> P1 -> P2 -> P3 -> P4 -> P5 -> P6
P1 -> P2A
P1 -> P2B
P2A -> P3
P2B -> P3
P3 -> P4
P4 -> P5
P0,P1,P2,P3,P4,P5 -> P6
```

Parallelizable work:

- P1 backend diff guard and P1 frontend compare UX can run in parallel after shared DTO/error shape is agreed.
- P2 local result-store fixtures and frontend policy/message tests can run in parallel.
- P3 MySQL smoke setup and UI smoke harness setup can run in parallel.
- P4 backend scale policy and frontend copy/tests can run in parallel after policy matrix is fixed.
- P5 design document can begin in parallel with P4, but implementation must wait for P3/P4 gates.

## Phase 0: Baseline, Inventory, And Gate Contract

Goal: freeze the current state and define how remediation will be judged.

Tasks:

- Record `git status --short --branch` and identify unrelated dirty files.
- Preserve current user changes; do not revert unrelated work.
- Record current fixes from `docs/codebase-performance-audit-report.md`.
- Inventory all large-result consumers:
  - `ResultStore::diff`
  - `load_rows_for_diff`
  - `load_all_result_rows`
  - `page`
  - `for_each_export_row`
  - `snapshot`
  - `explain_pair`
  - review decisions
  - result history reload
  - `DiffView`
  - `ResultsTab`
- Classify each consumer as SQL-backed, bounded, or blocked.
- Define named constants for new caps where needed.
- Decide whether to use a repo-local Cargo cache workaround:
  - Preferred command prefix while global cache is broken:
    - PowerShell: `$env:CARGO_HOME="$PWD\.cargo-test-home"; cargo test --locked ...`
- Save validation ledger template in the remediation report or plan appendix.

Acceptance:

- A large-result feature matrix exists.
- All caps and error states are named.
- Cargo cache blocker is either fixed or a repo-local workaround is documented.

Rollback:

- No production code changes in this phase.

## Phase 1: Make Compare Safe And Intelligible

Goal: eliminate the remaining confirmed OOM risk where diff loads all rows from spilled jobs.

Backend tasks:

- Add a diff safety cap, initially `100_000` rows per compared job unless SQL-backed diff is implemented.
- Modify `ResultStore::diff` to fail before `load_rows_for_diff` materializes all spilled rows above the cap.
- Prefer a typed validation error message such as:
  - `This comparison is too large to load safely in memory. Export both runs as CSV and compare externally, or rerun with narrower filters.`
- Add tests for:
  - in-memory diff below cap still works
  - spilled diff below cap remains deterministic
  - spilled diff above cap fails before loading all rows
  - same job id still returns the existing validation error

Frontend tasks:

- Add persistent inline compare status near Compare controls in `ResultsTab`.
- Do not rely only on toast for known compare-blocked errors.
- Disable or warn before Compare when job summaries indicate risky row counts.
- Make `DiffView` explicitly say when it shows only the first N rows.
- Map known scale guard errors to friendly copy, with technical detail secondary.

Suggested UI copy:

- `This comparison is too large to load safely in memory. Export both runs as CSV and compare them outside the app, or rerun with narrower filters.`

Acceptance:

- Large spilled diff cannot OOM through all-row materialization.
- Users get a stable inline explanation and recovery path.
- Context Engine review passes on diff guard changes.

Rollback:

- Remove or lower-risk toggle the diff cap if it blocks an urgent small-job workflow unexpectedly.

## Phase 2: Local Memory Fixtures, Tests, And Tooling

Goal: create fast local proof that large-result paths are bounded without requiring full matcher or MySQL runtime.

Tasks:

- Add synthetic result-store fixtures:
  - `result_store_spilled_100k`
  - `large_spilled_results_250k`
  - `diff_large_spilled_guard`
  - `xlsx_100001_guard`
- Add frontend tests:
  - same-session DB streaming allowed
  - cross-session DB streaming not active
  - file/cascade/fuzzy million-row blocks
  - sub-million warning levels
  - blocked diff UX
  - export failure messaging
- Add DTO/schema tests if `DiffResultDto`, error DTOs, or status fields change.
- Add a non-Docker smoke benchmark command for result-store paging/export/diff guard.
- Resolve or work around Cargo registry blocker:
  - observed blocker: `icu_locale_core-2.0.0/src/data.rs` access denied
  - use repo-local `CARGO_HOME=.cargo-test-home` if global cache cannot be repaired safely

Validation commands:

- `cargo test --locked <focused_result_store_filter>`
- `cargo check --locked --manifest-path src-tauri/Cargo.toml`
- `pnpm --dir ui test runScalePolicy`
- `pnpm --dir ui test <results-or-export-filter>`
- `pnpm --dir ui lint`

Acceptance:

- Local tests prove caps and policy decisions without DB dependencies.
- Cargo blocker is marked `PASSED` or `BLOCKED` with exact next action.

Rollback:

- Fixture additions are test-only and can be reverted independently.

## Phase 3: Database And Tauri Runtime Smoke

Goal: prove real DB workflows match policy and do not route unsafe paths.

Preconditions:

- Docker/MySQL status checked:
  - `docker ps --filter "name=matchers-mysql-1"`
  - `docker compose ps`
- Use `docker-compose.yml` service `matchers-mysql` if local container is absent.
- Do not print secrets beyond repo-documented disposable fixture values.

Backend smoke tasks:

- `db_same_session_10k`
  - same MySQL session
  - deterministic streaming algorithm
  - expected match count equals fixture baseline
- `db_same_session_100k`
  - required pre-merge scale smoke if fixture generation is practical
- `db_cross_session_blocked`
  - prove cross-session DB run does not select single-pool streaming
  - verify fallback/block behavior is intentional and logged
- Cancellation/progress sanity:
  - cancel mid-run
  - ensure terminal state is honest
  - progress events continue until terminal state

Frontend smoke tasks:

- Start same-session DB run from UI.
- Verify result paging after streamed run.
- Verify export CSV remains available.
- Verify XLSX over cap gives friendly message.
- Verify cross-session large DB run gives friendly policy message.

Acceptance:

- Same-session streaming works on small deterministic fixture.
- Cross-session single-pool streaming is impossible.
- Results page and export remain usable after streamed runs.

Rollback:

- DB smoke fixture changes must use disposable `smoke_*` tables only.
- Cleanup requires explicit safe target confirmation if destructive.

## Phase 4: Scale Policy Tightening

Goal: make unsafe large modes hard to trigger and explain why.

Policy tasks:

- Backend `src/run_service/scale.rs`
  - add pair-count or mode-specific risk helpers
  - keep constants mirrored with frontend
  - include same-session and algorithm gates
- Frontend `ui/src/shared/runScalePolicy.ts`
  - mirror backend decisions
  - expose warning level and block reason for UX
- UI copy and behavior:
  - `100k+`: non-blocking warning for file, cascade, fuzzy, cross-session DB
  - `500k+`: strong confirmation with mode-specific text
  - `1M+`: hard block for file/cascade/fuzzy/non-streamable paths
  - cross-session DB at large scale:
    - `Large database streaming currently requires source and target tables from the same DB session. Reconnect both tables through one session or run a smaller job.`
- Export buttons:
  - add per-format loading/disabled state
  - guide users toward CSV when XLSX is blocked
  - explain that export uses current confidence and level filters

Tests:

- Rust scale policy tests.
- Frontend `runScalePolicy` tests.
- Results UI message tests where practical.

Acceptance:

- Backend and frontend policy tests agree.
- No unsupported mode silently starts a known OOM-prone path.

Rollback:

- Policy thresholds are constants and can be tuned without altering matching algorithms.

## Phase 5: Two-Pool DB Streaming Design And Optional Implementation

Goal: define a correctness-first path for true cross-session large DB matching.

Default scope: design only unless implementation is explicitly approved.

Design tasks:

- Write `docs/two-pool-db-streaming-design.md`.
- Define API contract:
  - source pool cursor
  - target pool cursor or partition lookup
  - join semantics
  - deterministic ordering
  - cancellation
  - pause/resume
  - progress events
  - duplicate handling
  - result row id assignment
  - error mapping
- Define future capability flag:
  - `supports_cross_session_streaming`
- Define UI contract:
  - current state: blocked or fallback
  - future state: allowed only when backend capability is true
  - diagnostics show same-session equality as boolean, not secrets

Optional implementation tasks, only after approval:

- Add two-pool streaming runner separate from current single-pool runner.
- Compare against existing in-memory two-pool path on small fixtures.
- Add DB smoke with asymmetric row counts.

Acceptance for implementation:

- Correctness matches existing two-pool in-memory path on small fixture.
- Peak RSS stays within `+20%` of same-session streaming baseline for equivalent row counts.
- Cross-session deterministic match count and stable pair keys match baseline.

Rollback:

- Keep current same-session guard until two-pool streaming passes all gates.

## Phase 6: Full Validation, Review, And Final Report

Goal: bundle evidence and decide what is passed, blocked, or deferred.

Validation matrix:

- `git status --short --branch`
- `cargo fmt --check` or documented existing drift
- `cargo check --locked --manifest-path src-tauri/Cargo.toml`
- focused Rust tests:
  - result-store diff guard
  - XLSX guard
  - scale policy
- `pnpm --dir ui lint`
- `pnpm --dir ui test`
- `pnpm --dir ui build`
- Docker/MySQL smoke or exact blocked reason
- Context Engine `review_auto` on all code changes
- Performance benchmark evidence for local fixtures

Final report must include:

- Findings fixed
- Findings guarded
- Findings deferred
- Validation ledger
- Blocked gates and exact next command
- Memory-risk status by feature
- User-visible behavior changes
- Rollback notes

Acceptance:

- Every requirement from `docs/codebase-performance-audit-goal.md` maps to evidence.
- Unsafe large-result paths are guarded or explicitly deferred.
- No known high-risk path remains silently selectable.

## Swarm Work Slices

### Slice A: Backend Safety

Owner: backend worker

Files:

- `src/run_service/store.rs`
- `src/run_service/dto.rs` if error/status DTOs change
- `src-tauri/src/commands/results.rs`

Tasks:

- Implement diff guard.
- Add result-store fixtures/tests.
- Preserve export and paging behavior.

### Slice B: Frontend Recovery UX

Owner: frontend worker

Files:

- `ui/src/features/results/ResultsTab.tsx`
- `ui/src/features/results/DiffView.tsx`
- `ui/src/shared/runScalePolicy.ts`
- `ui/src/__tests__/...`

Tasks:

- Inline blocked compare state.
- Friendly known-error mapping.
- Export button loading/disabled states.
- Policy tests and UI tests.

### Slice C: Scale Policy And Contracts

Owner: backend/frontend pair

Files:

- `src/run_service/scale.rs`
- `ui/src/shared/runScalePolicy.ts`
- `ui/src/__tests__/runScalePolicy.test.ts`
- `docs/codebase-performance-audit-report.md`

Tasks:

- Tighten policy thresholds.
- Keep constants and decisions aligned.
- Document compatibility and user-facing changes.

### Slice D: DB Smoke And Benchmarks

Owner: performance worker

Files:

- `tests/csv_import_mysql.rs`
- `tests/csv_e2e.rs`
- `docs/mysql-benchmark-fixture.md`
- new benchmark docs/scripts if needed

Tasks:

- Create/verify smoke fixtures.
- Record memory and runtime evidence.
- Mark Docker/MySQL status honestly.

### Slice E: Two-Pool Streaming Design

Owner: backend architect

Files:

- `docs/two-pool-db-streaming-design.md`
- optional API sketches under `docs/`

Tasks:

- Produce design and UI/backend contract.
- Define implementation gates.
- Do not implement until approved.

## Final Notes

- Guarding unsafe paths comes before optimizing them.
- Two-pool streaming is not the universal fix; it is a later capability after current unsafe paths are impossible to trigger.
- Every new block/guard must include a recovery path: CSV export, narrower filters, same-session connection, or explicit future unsupported status.
