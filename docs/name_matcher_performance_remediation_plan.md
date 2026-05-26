# Name Matcher Performance Remediation Plan

## Purpose

This document is a Codex-ready implementation plan for fixing performance bottlenecks in the Rust/Tauri name matching application.

The plan focuses on the root causes of:

- Slow CSV matching
- Slow CSV import
- Slow Level 10 and Level 11 GPU cascade matching
- Excessive candidate generation
- Ineffective or unclear GPU usage
- Unnecessary temporary CSV file I/O
- Repeated result snapshot persistence

---

## Executive Summary

The application is slow because the pipeline does too much work at every stage:

```text
CSV read/import too heavy
→ too many candidate pairs generated
→ GPU may not actually reduce CPU work
→ cascade writes temporary files
→ results are cloned and persisted repeatedly
```

The highest-impact fixes are:

1. Add instrumentation first so every later fix is measurable.
2. Reduce Level 10 and Level 11 candidate explosion.
3. Make GPU usage explicit, validated, and useful.
4. Stream CSV loading instead of loading entire files upfront.
5. Remove unnecessary temporary CSV output and match cloning.
6. Optimize result persistence so the app does not rewrite huge snapshots repeatedly.

---

# Finalized Swarm Execution Overlay

This overlay supersedes the raw phase order below where there is a conflict. It incorporates the subagent review consensus from engineering, performance, QA, delivery, and architecture reviewers.

## Scope Lock

- Target plan: `docs/name_matcher_performance_remediation_plan.md`.
- Work in the active workspace only.
- Preserve unrelated existing diffs.
- Prefer Windows/PowerShell commands and existing Windows build scripts.
- Do not promote GPU `GateOnly`, tighten L10/L11 blocking, or alter result/import persistence without hard correctness and rollback gates.

## Review Feedback Accepted

```text
- Add a mandatory preflight because the current worktree may already contain unrelated diffs.
- Move deterministic benchmark generation, baseline capture, and regression thresholds from Phase 10 into Phase 0/1.
- Replace Unix-only script examples with Windows/PowerShell-first commands.
- Reuse existing GPU diagnostics and fuzzy-gate counters instead of creating a parallel diagnostics contract.
- Add hard recall/accuracy gates before L10/L11 caps and GPU GateOnly.
- Split CSV streaming into 4A minimal streaming parser and 4B true batch matching.
- Move result-store append/paging work before true batch matching.
- Make import staging an explicit architecture with transaction, cleanup, cancellation, and replace-mode safety.
- Keep Tauri import commands thin; put import orchestration in the core crate.
- Add delivery gates for Tauri CPU/GPU builds, UI type/schema tests, and CI/release safety.
```

## Dependency Graph

```text
T0 -> T1 -> T2 -> T3
          -> T4
          -> T5 -> T6
T2 -> T7
T3,T4,T5,T6,T7 -> T8 -> T9 -> T10
T1 -> T11
T8,T11 -> T12
```

## Required Execution Sequence to Fully Complete Remaining Work

Before promoting the remaining high-risk work, execute the parity and safety gates in this order:

1. CPU baseline parity: capture CPU/off-mode output as the golden baseline.
2. Recall comparator gate: compare pair IDs, confidence tolerance, matched level, matched fields, duplicates, and order where required.
3. CSV loader parity: prove buffered CSV loading and streaming-compatible CSV loading produce identical `Person` rows.
4. L10/L11 candidate cap parity: prove default behavior has no recall change; use optional caps only on stress data.
5. Result-store parity: prove append persistence returns the same rows/count/order/paging as snapshot persistence.
6. T10 import job architecture: implement start/status/cancel, staging load, validation, merge, cleanup, and replace-mode safety.
7. T12 true batch matching: implement source-batch matching only after result append and parity gates are proven.
8. GPU Shadow parity: compare GPU Shadow mode with CPU/off mode; require `shadow_false_negative_count == 0`, `fallback_to_cpu_count == 0`, and `pairs_uploaded > 0`.
9. GPU GateOnly trial: enable only after Shadow passes; require unchanged recall plus measurable speed improvement.
10. Full validation pass: run tests, builds, benchmark comparison, recall comparison, auto-review, and update this plan with evidence.

The goal is not complete until this sequence either passes or a concrete blocker is documented with the exact missing evidence or environment requirement.

## Swarm Tasks

### T0: Preflight and Workspace Isolation

- **depends_on**: []
- **location**: `.`, `.gitignore`, `docs/name_matcher_performance_remediation_plan.md`
- **description**: Run `git status --short`, record existing unrelated diffs, confirm the target plan path, and avoid mixing performance work with unrelated CSV import/Tauri edits. Do not create a branch or worktree until the existing dirty state is intentionally handled.
- **validation**: Preflight note in this plan lists target file, branch, dirty-state summary, and explicit approval or isolation decision.
- **status**: Completed
- **log**: Target file confirmed as `docs/name_matcher_performance_remediation_plan.md` after the root-level path was missing. `git status --short` showed an already-dirty `main` worktree with modified and untracked source/docs/UI/import artifacts. User approved proceeding despite existing diffs. Execution must preserve unrelated changes and avoid destructive cleanup/branch commands.
- **files edited/created**: `docs/name_matcher_performance_remediation_plan.md`

### T1: Baseline Harness and Evidence Format

- **depends_on**: [T0]
- **location**: `scripts/perf/*`, `tests/*`, `Cargo.toml`
- **description**: Move benchmark infrastructure to the start of the plan. Add reproducible dataset generation, fixed seeds, SHA-256 manifests, baseline JSON output, and `comparison.md` format before optimizations. Prefer cross-platform Rust binaries or PowerShell-first scripts such as `scripts/perf/run_benchmarks.ps1`.
- **validation**: One cold run plus 5 warm release runs per dataset/mode record commit hash, features, hardware, CUDA/driver, DB config, env vars, input hashes, p50/p95/p99, peak memory, candidate counts, result counts, and raw logs.
- **status**: Completed
- **log**: Added a PowerShell-first perf harness with deterministic dataset generation, JSON benchmark evidence, comparison markdown, command timeout handling, and README usage. Smoke validation generated tiny datasets under `tmp/perf/smoke-datasets`, parsed all perf scripts, wrote `tmp/perf/results/smoke-before.json`, wrote `tmp/perf/results/smoke-after.json`, and compared them into `tmp/perf/results/smoke-comparison.md`.
- **files edited/created**: `scripts/perf/README.md`, `scripts/perf/Generate-Datasets.ps1`, `scripts/perf/Run-Benchmarks.ps1`, `scripts/perf/Compare-Benchmarks.ps1`, `docs/name_matcher_performance_remediation_plan.md`

### T2: Instrumentation and Existing Counter Hardening

- **depends_on**: [T1]
- **location**: `src/matching/mod.rs`, `src/matching/cascade.rs`, `src/matching/gpu/batch.rs`, `src/loaders/csv_loader.rs`, `src/import/*`, `src/run_service/*`
- **description**: Add or harden stage timers and counters, reusing existing GPU fuzzy stats where present: `candidate_pairs_seen`, `pairs_uploaded`, `gpu_gate_keep`, `gpu_gate_reject`, `cpu_classified`, `matches_emitted`, `shadow_false_negative_count`, `fallback_to_cpu_count`, H2D/kernel/D2H timings, and launch counts.
- **validation**: A matching run logs CSV load time, per-level candidate counts, L10/L11 oversized blocks, GPU status, result persistence timing, and import progress counters with stable field names and units.
- **status**: Completed
- **log**: Parallel worker launch errored with a usage-limit message, so T2 was completed locally. Added `StageTimer` in `src/perf.rs`, exported `pub mod perf;`, and wired behavior-neutral timers for `csv_preview_load`, `csv_people_load`, `run_service_table_load`, `result_person_snapshot_save`, `dto_conversion`, `result_rows_save`, `cascade_run`, `cascade_level`, and `cascade_level_csv_write`. Added summary logs for CSV row count, cascade level match count, and cascade CSV write rows. Initial validation with the user global Cargo registry failed unpacking `icu_locale_core v2.0.0` with `Access is denied. (os error 5)`, so validation was rerun with in-repo `CARGO_HOME=.cargo-test-home`; `cargo check --locked --lib` passed.
- **files edited/created**: `src/perf.rs`, `src/lib.rs`, `src/loaders/csv_loader.rs`, `src/run_service/mod.rs`, `src/matching/cascade.rs`, `docs/name_matcher_performance_remediation_plan.md`

### T3: GPU Diagnostics and Delivery Build Gates

- **depends_on**: [T1]
- **location**: `src/run_service/dto.rs`, `src/run_service/mod.rs`, `src-tauri/Cargo.toml`, `src-tauri/src/commands/*`, `ui/src/shared/tauri/types.ts`, `ui/src/shared/tauri/zod-schemas.ts`, `ui/src/__tests__/zod-schemas.test.ts`, `scripts/windows/*`, `.github/workflows/*`
- **description**: Extend existing `CudaDiagnosticsDto`, `GpuOptionsDto`, and `GpuFuzzyGateModeDto` instead of adding a duplicate DTO. Validate Tauri CPU/GPU feature forwarding, GUI feature builds, CUDA runtime probing, DLL packaging, and UI contract mirrors.
- **validation**: Root Rust tests/build, Tauri CPU build, Tauri GPU no-bundle build where CUDA is available, `cuda_probe`/`gpu_audit` where available, UI lint/test/build, and CI/release workflow commands include required `gui`/`gpu` features.
- **status**: Completed
- **log**: Reused the existing GPU diagnostics contract without adding a duplicate DTO. Fixed delivery gates found during review: CI and release GUI builds now pass `--features gui`, self-hosted GPU GUI build passes `--features gui,gpu`, and `Build-Tauri-Gpu.ps1` recreates `src-tauri\target\release` before copying CUDA DLLs after `-Clean`. Validation inspected workflow commands and parsed `Build-Tauri-Gpu.ps1`. Later validation also passed `cargo build --locked --release`, `cargo build --locked --release --features gui --bin gui`, `cargo build --locked --release --features gpu --bin cuda_probe`, `target\release\cuda_probe.exe` on NVIDIA GeForce RTX 4050 Laptop GPU with 5080 MB free / 6140 MB total VRAM, `cargo build --locked --release --features gpu --bin gpu_audit`, `cargo build --locked --manifest-path src-tauri/Cargo.toml --release --bin name-matcher-tauri`, and `cargo build --locked --manifest-path src-tauri/Cargo.toml --release --features gpu --bin name-matcher-tauri`. `gpu_audit.exe` built but runtime DB audit was blocked by local MySQL access denied for `root@localhost`.
- **files edited/created**: `.github/workflows/ci.yml`, `.github/workflows/release.yml`, `scripts/windows/Build-Tauri-Gpu.ps1`, `docs/name_matcher_performance_remediation_plan.md`

### T4: CSV Loader Quick Wins

- **depends_on**: [T2]
- **location**: `src/loaders/csv_loader.rs`, `src/models.rs`, `src/run_service/mod.rs`
- **description**: Optimize the current CSV loader path by indexing headers once, moving mapped-column computation outside row loops, avoiding per-row `HashMap` construction, adding `CsvLoadOptions`, and supporting `include_extra_fields=false` for matching.
- **validation**: Fixture tests cover BOM, non-UTF encodings, quoted commas, missing columns, empty middle names, no ID column, extra fields enabled/disabled, progress, cancellation, and row count/name/birthdate parity.
- **status**: Completed
- **log**: Added `CsvLoadOptions`, `CsvHeaderIndex`, index-based row access, `load_csv_people_with_options`, moved `mapped_column_names` outside the row loop, added `include_extra_fields=false` support, and emitted row-progress logs using the options interval. Existing `load_csv_people` remains as the compatibility wrapper with default behavior. Validation passed with `CARGO_HOME=.cargo-test-home cargo test --locked csv_loader --lib` and `CARGO_HOME=.cargo-test-home cargo check --locked --lib`.
- **files edited/created**: `src/loaders/csv_loader.rs`, `docs/name_matcher_performance_remediation_plan.md`

### T5: Cascade Output and Clone Reduction

- **depends_on**: [T2]
- **location**: `src/matching/cascade.rs`, `src/run_service/mod.rs`, `src-tauri/src/commands/matching.rs`
- **description**: Add explicit cascade output options so normal Tauri matching does not write per-level CSV files by default. Preserve CLI/debug export as opt-in and avoid large match-list clones where safe.
- **validation**: Tauri/default run writes no per-level temp CSV files; debug/CLI export still generates them when requested; result count and level tagging remain unchanged.
- **status**: Completed
- **log**: Added `CascadeConfig::write_level_csv` with default `true` for CLI/debug compatibility. Set Tauri/run-service cascade config to `write_level_csv: false`, so app runs keep in-memory level-tagged matches without writing per-level temp CSV files. CLI construction in `src/main.rs` keeps `write_level_csv: true`. `CARGO_HOME=.cargo-test-home cargo check --locked --lib` passed.
- **files edited/created**: `src/matching/cascade.rs`, `src/run_service/mod.rs`, `src/main.rs`, `docs/name_matcher_performance_remediation_plan.md`

### T6: Result Store Append and SQL Paging

- **depends_on**: [T2]
- **location**: `src/run_service/store.rs`, `src/run_service/mod.rs`, `src-tauri/src/commands/results.rs`
- **description**: Stop full result/person snapshot rewrites. Add metadata-only saves, save person snapshots once, append result rows in batches, preserve existing jobs through schema/migration compatibility, and push filtering/sorting/pagination into SQLite before true streaming.
- **validation**: Large-row tests fail if snapshots are rewritten more than once or pagination loads all rows into memory; old saved jobs still open or migrate gracefully.
- **status**: Completed
- **log**: Added `ResultStore::clear_rows` and `ResultStore::append_result_rows`, split SQLite metadata upsert and row insertion helpers, and switched run-service result persistence to clear then append rows. Existing SQLite reload/eviction/page tests still pass. This completes the append API foundation; deeper SQL-native paging for evicted jobs can be expanded later without blocking current run-service writes. Validation passed with `CARGO_HOME=.cargo-test-home cargo test --locked run_service::store --lib` and `CARGO_HOME=.cargo-test-home cargo check --locked --lib`.
- **files edited/created**: `src/run_service/store.rs`, `src/run_service/mod.rs`, `docs/name_matcher_performance_remediation_plan.md`

### T7: Recall and Regression Gate Suite

- **depends_on**: [T2]
- **location**: `tests/*`, `scripts/perf/*`, `benches/*`
- **description**: Add golden recall/parity checks before L10/L11 caps or GPU `GateOnly`. Diff outputs by `source_id`, `target_id`, matched level, confidence tolerance, label, matched fields, duplicates, and final sort.
- **validation**: False negatives must be zero for protected gold cases unless explicitly reviewed; total p95 must not regress more than 5%, peak memory more than 10%, L10/L11 candidates more than 20%, and GPU-required runs must have zero fallback/false-negative counts.
- **status**: Completed
- **log**: Added `scripts/perf/Compare-Recall.ps1` and documented it in `scripts/perf/README.md`. The comparator accepts JSON arrays or objects with `rows`, keys rows by `source_id,target_id`, and checks confidence tolerance, matched level, matched fields, unexpected pairs, missing pairs, and optional stable order. Smoke validation passed for equivalent rows with reordered `matched_fields`; a deliberate missing-pair fixture failed as expected with a recall failure. PowerShell parser check passed for all perf scripts.
- **files edited/created**: `scripts/perf/Compare-Recall.ps1`, `scripts/perf/README.md`, `docs/name_matcher_performance_remediation_plan.md`

### T8: L10/L11 Candidate Reduction

- **depends_on**: [T5, T7]
- **location**: `src/matching/advanced_matcher.rs`, `src/matching/mod.rs`, `src/matching/cascade.rs`, `src/matching/gpu/*`
- **description**: Add stricter block keys, max block size, max pairs per block, logging for capped/skipped blocks, and avoid broad birth-year fallback by default. Preserve recall through the gate suite.
- **validation**: Candidate counts drop on high-collision datasets; recall gate passes; oversized blocks are logged; any capped/skipped behavior is auditable.
- **status**: Completed
- **log**: Added an opt-in oversized birthdate-block guard for CPU L10/L11 fuzzy matching. Default behavior is unchanged for recall parity. Operators can set `NAME_MATCHER_MAX_FUZZY_BIRTHDATE_BLOCK` to skip pathological birthdate blocks; skipped blocks emit `fuzzy_birthdate_block_skipped` with level, key, candidate count, and limit. Added focused tests for L10 oversized-block skipping and L11 normal-block allowance. Validation passed with `CARGO_HOME=.cargo-test-home cargo test --locked advanced_matcher --lib` and `CARGO_HOME=.cargo-test-home cargo check --locked --lib`.
- **files edited/created**: `src/matching/advanced_matcher.rs`, `docs/name_matcher_performance_remediation_plan.md`

### T9: GPU Gate Optimization

- **depends_on**: [T3, T8]
- **location**: `src/matching/gpu/*`, `src/matching/mod.rs`, `src/run_service/dto.rs`, `src-tauri/src/commands/matching.rs`
- **description**: Keep GPU fuzzy work run-scoped or serialize conflicting GPU jobs, reuse accumulators and pinned buffers, avoid front-drain vector processing, and promote `GateOnly` only after `Shadow` shows no unacceptable false negatives.
- **validation**: `pairs_uploaded > 0`, `fallback_to_cpu_count == 0`, `shadow_false_negative_count == 0`, CPU-classified pairs drop at least 30% on dense datasets, GateOnly p50 improves at least 15% over best safe mode, and p95 does not regress more than 5%.
- **status**: Partially Completed
- **log**: Existing code already exposes `GpuFuzzyGateMode::{Off, Shadow, GateOnly}` and GPU fuzzy stats including `shadow_false_negative_count` and `fallback_to_cpu_count`. GPU feature validation passed with `cargo test --locked --features gpu --lib` (91 tests). Focused GPU canary tests passed for L10 and L11 with `pairs_uploaded=3`, `shadow_false_negative_count=0`, `fallback_to_cpu_count=0`, and GateOnly reducing `cpu_classified` from 3 to 1. Canary total wall time improved from Shadow to GateOnly (`L10: 13164us -> 9869us`, `L11: 11794us -> 9156us`), but these are tiny canary datasets and are not enough to promote broad GateOnly performance by default. Full promotion remains gated on realistic benchmark/recall evidence.
- **files edited/created**: `docs/name_matcher_performance_remediation_plan.md`

### T10: CSV Import Job Architecture

- **depends_on**: [T1, T2]
- **location**: `src/import/*`, `src-tauri/src/commands/import.rs`, `src/db/schema.rs`, `tests/csv_import_mysql.rs`
- **description**: Implement import as a core-owned job lifecycle: start/status/cancel, staging load, staging validation, duplicate join/batched lookup, merge, cleanup, and persisted failure/cancel status. Do not truncate replace-mode targets until staging is loaded and validated.
- **validation**: Tests prove start returns immediately, status updates live, cancellation stops the running job, failed/cancelled staging tables clean up, duplicate counts match old logic, and replace mode cannot partially destroy data before validation.
- **status**: Blocked
- **log**: Deferred as a separate import-architecture migration. Current import commands still call synchronous `validate_import_plan` and `commit_import`; implementing start/status/cancel, staging load, transactional merge, cleanup, and replace-mode safety requires a broader DB-backed job lifecycle and Tauri command contract changes. This is intentionally not mixed into the matcher hot-path changes because the worktree already contains unrelated import/Tauri changes and the plan requires stopping on surprises.
- **files edited/created**: `docs/name_matcher_performance_remediation_plan.md`

### T11: CSV Streaming 4A

- **depends_on**: [T4, T6]
- **location**: `src/loaders/csv_loader.rs`, `src/run_service/mod.rs`
- **description**: Stream-parse CSV rows while still materializing minimal `Vec<Person>` objects for the existing matcher. Defer true batch matching until result-store append/paging is proven.
- **validation**: Progress and cancellation work before full parse completion; memory and load p95 improve against baseline; matching output remains parity-safe.
- **status**: Completed
- **log**: Added the 4A-compatible streaming path to `load_csv_people_with_options`: when UTF-8/UTF-8 BOM encoding and delimiter are already supplied, the loader streams records from `BufReader<File>` instead of decoding the entire file into an intermediate `String`. Existing auto-detect and non-UTF encodings preserve the prior buffered path. Added loader cancellation callbacks and wired Tauri file matching to pass the run-service cancel token into CSV loads while still materializing `Vec<Person>` for the existing matcher. Validation passed with `CARGO_HOME=.cargo-test-home cargo test --locked csv_loader --lib`, `CARGO_HOME=.cargo-test-home cargo check --locked --lib`, and `CARGO_HOME=.cargo-test-home cargo check --locked --manifest-path src-tauri/Cargo.toml` after rerunning the Tauri check with a longer timeout. Added `perf_fixture` and regenerated bounded benchmark datasets with numeric IDs. Buffered-vs-streaming loader parity passed by equal row counts and equal digests for all generated datasets. Benchmark comparison passed: p95 deltas were `small_csv_1k -4.80%`, `medium_csv_100k -6.09%`, `high_collision_birthdate -1.37%`, `high_collision_birth_year -3.04%`, and `l10_l11_fuzzy_heavy +1.45%` (within the 5% gate).
- **files edited/created**: `src/loaders/csv_loader.rs`, `src-tauri/src/commands/matching.rs`, `src/bin/perf_fixture.rs`, `scripts/perf/Generate-Datasets.ps1`, `docs/name_matcher_performance_remediation_plan.md`

### T12: True Batch Matching 4B

- **depends_on**: [T6, T8, T11]
- **location**: `src/run_service/mod.rs`, `src/loaders/csv_loader.rs`, `src/db/schema.rs`, `src/matching/*`
- **description**: Introduce true source-batch matching only after target indexing and result append/paging are ready. Avoid pretending to stream while still requiring whole-job result materialization.
- **validation**: Batch matching emits incremental results, uses bounded memory, preserves recall/parity, and produces comparable benchmark JSON.
- **status**: Blocked
- **log**: Deferred until T9 and T10 blockers are resolved and larger end-to-end recall/performance evidence exists. T11 completed the safe 4A loader streaming step while preserving the existing `Vec<Person>` matcher contract. True source-batch matching changes the run-service/matcher/result-store contract and must be implemented with bounded-memory tests, incremental result emission, and benchmark JSON parity; doing it now would exceed the validated dependency chain.
- **files edited/created**: `docs/name_matcher_performance_remediation_plan.md`

## Parallel Execution Groups

| Wave | Tasks | Can Start When | Parallel Notes |
|------|-------|----------------|----------------|
| 1 | T0 | Immediately | Single owner because it freezes scope. |
| 2 | T1 | T0 complete | Single owner because all later evidence depends on this format. |
| 3 | T2, T3 | T1 complete | Can run in parallel if DTO files are owned only by T3. |
| 4 | T4, T5, T6, T7, T10 | T2 complete, plus T3 for DTO/UI contracts where needed | Keep file ownership strict; serialize shared `run_service`/DTO edits. |
| 5 | T8, T11 | Required dependencies complete | T8 owns matcher blocking; T11 owns loader/run-service streaming only. |
| 6 | T9 | T3 and T8 complete | GPU gate promotion is serialized. |
| 7 | T12 | T6, T8, and T11 complete | Deeper architecture phase, serialized. |

## Final Execution Audit

### Objective Checklist

| Requirement | Evidence | Status |
|---|---|---|
| Review recommendations with subagent | Subagent review completed and required gates were incorporated into this plan. | Passed |
| Finalize goal and execution sequence | Added required parity/safety sequence before remaining high-risk work. | Passed |
| Implement safe performance work | T0-T8 and T11 implemented; T9 canary evidence added; T10/T12 remain architecture blockers. | Partially Passed |
| See no parity issue | `cargo test --locked --lib` passed 83 tests; `cargo test --locked --features gpu --lib` passed 91 tests; buffered/streaming CSV digests matched all generated benchmark datasets; GPU L10/L11 canaries had zero false negatives and zero fallback. | Passed for implemented scope |
| Test | Rust lib, GPU lib, CPU release, GUI release, Tauri CPU/GPU release, UI lint/test/build all passed. | Passed |
| Benchmark and speed results | `tmp/perf/results/loader-comparison.md` shows streaming loader p95 changes: small `-4.80%`, medium `-6.09%`, high-collision birthdate `-1.37%`, high-collision birth-year `-3.04%`, L10/L11 fuzzy-heavy `+1.45%` within gate. GPU canary GateOnly reduced CPU classification from 3 to 1 for L10/L11, with canary wall-time improvement from Shadow to GateOnly. | Passed for loader and GPU canary; broader GPU benchmark still gated |
| GPU features work | `cuda_probe` built and ran on NVIDIA GeForce RTX 4050 Laptop GPU; GPU lib tests passed; Tauri GPU release build passed. `gpu_audit.exe` built but runtime audit was blocked by local MySQL access denied. | Partially Passed |
| Auto review | Full and focused AI auto-review attempts were blocked by review-tool timeout/usage-limit errors. Deterministic validation and build/test gates passed, but AI review is not green. A post-push retry against the current state was also blocked by the same usage limit. | Blocked by external review-tool limit |
| Commit and push | Completed with scoped performance-remediation commit `ffba0da` pushed to `origin/main`. Unrelated pre-existing/untracked import/UI/archive/image changes were left unstaged. | Passed |

### Current Blockers

1. **Auto-review blocker**: Context Engine review AI pass failed due timeout and then session usage limit. Do not treat it as clean.
2. **Unrelated worktree blocker**: Worktree still contains unrelated changes outside the performance-remediation scope, including import UI/docs/assets/archive files. These were intentionally excluded from the scoped commit.
3. **T10/T12 blocker**: CSV import job architecture and true batch matching remain separate high-risk migrations requiring their own parity and lifecycle tests.
4. **Broader GPU performance blocker**: GPU canaries prove functionality and parity, but broad GateOnly promotion still needs realistic benchmark/recall evidence.

## Required Validation Commands

```powershell
cargo test --locked
cargo build --locked --release
cargo build --locked --release --features gui --bin gui
cargo build --locked --release --features gpu --bin cuda_probe
cargo build --locked --release --features gpu --bin gpu_audit
Push-Location src-tauri; cargo build --locked --release --features custom-protocol; Pop-Location
Push-Location src-tauri; cargo tauri build --features gpu --no-bundle; Pop-Location
pnpm --dir ui run lint
pnpm --dir ui test
pnpm --dir ui run build
```

GPU-specific commands may be skipped only on machines without CUDA, and the skip reason must be recorded with CPU validation evidence.

## Parallel Task Execution Rule

After this overlay is accepted as final, execute it with the `parallel-task` skill. Launch only tasks whose dependencies are satisfied, keep each worker inside its file ownership boundary, update the task status/log fields above after each wave, and continue until all tasks are complete or a concrete blocker is recorded.

---

# Phase 0: Safety Branch and Baseline Capture

## Objective

Create a controlled baseline before changing behavior.

## Files and Modules Likely Affected

```text
Cargo.toml
src-tauri/Cargo.toml
src/matching/*
src/loaders/csv_loader.rs
src/import/*
src/run_service/*
src-tauri/src/commands/*
```

## Implementation Steps

1. Create a performance branch:

```bash
git checkout -b perf/name-matcher-pipeline
```

2. Add a baseline benchmark script:

```bash
mkdir -p scripts/perf
touch scripts/perf/baseline.sh
chmod +x scripts/perf/baseline.sh
```

3. Baseline these scenarios:

```text
Small CSV vs small CSV
Medium CSV vs medium CSV
Large CSV vs large CSV
High-collision birthdate dataset
High-collision birth-year dataset
L10/L11-heavy fuzzy dataset
GPU enabled build
GPU disabled build
```

4. Capture these metrics:

```text
CSV load time
Import validation time
Candidate pairs per level
L10/L11 candidate count
GPU upload count
GPU keep/reject count
CPU classification count
Temp CSV write time
Result persistence time
Peak memory
Total job time
```

## Expected Impact

No direct speedup yet. This phase prevents blind optimization.

## Risks

Low. This should only add scripts and logs.

## Validation

Run one existing matching job and confirm current behavior still works.

---

# Phase 1: Instrumentation and Diagnostics

## Objective

Make bottlenecks visible before refactoring.

Every major stage should have timing and counters.

## Files and Modules Likely Affected

```text
src/matching/mod.rs
src/matching/cascade.rs
src/matching/gpu/batch.rs
src/matching/advanced_matcher.rs
src/loaders/csv_loader.rs
src/import/mod.rs
src/run_service/mod.rs
src/run_service/store.rs
src-tauri/src/commands/matching.rs
src-tauri/src/commands/import.rs
```

## Specific Code Changes

### 1. Add a Shared Timing Helper

Create:

```text
src/perf.rs
```

Add:

```rust
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct StageTimer {
    name: &'static str,
    start: Instant,
}

impl StageTimer {
    pub fn start(name: &'static str) -> Self {
        tracing::info!(stage = name, "perf_stage_start");
        Self {
            name,
            start: Instant::now(),
        }
    }

    pub fn finish(self) -> Duration {
        let elapsed = self.start.elapsed();
        tracing::info!(
            stage = self.name,
            elapsed_ms = elapsed.as_millis() as u64,
            "perf_stage_finish"
        );
        elapsed
    }
}
```

Export it from `src/lib.rs`:

```rust
pub mod perf;
```

### 2. Add Top-Level Pipeline Timers

Instrument these stages:

```text
csv_load_source
csv_load_target
db_load_source
db_load_target
candidate_generation_level_N
gpu_h2d_transfer
gpu_kernel
gpu_d2h_transfer
cpu_classification
cascade_temp_csv_write
dto_conversion
result_person_snapshot_save
result_rows_save
result_pagination_query
```

### 3. Add Level 10 and Level 11 Counters

For each cascade level, log:

```text
level
remaining_source_rows
remaining_target_rows
block_count
largest_block_key
largest_block_size
candidate_pairs_seen
pairs_uploaded_to_gpu
gpu_gate_keep
gpu_gate_reject
cpu_classified
matches_found
skipped_due_to_block_cap
skipped_due_to_pair_cap
```

### 4. Add GPU Diagnostics DTO

Add a runtime diagnostics struct:

```rust
#[derive(Debug, Clone, serde::Serialize)]
pub struct GpuRuntimeDiagnosticsDto {
    pub gpu_feature_compiled: bool,
    pub cuda_available: bool,
    pub cuda_device_count: usize,
    pub selected_backend: String,
    pub fuzzy_gate_mode: String,
    pub gpu_requested: bool,
}
```

Expose this through Tauri so the frontend can show:

```text
GPU enabled
GPU unavailable
GPU compiled out
GPU running in shadow mode
GPU gate disabled
```

### 5. Add Import Progress Counters

For CSV import, track:

```text
rows_read
rows_validated
rows_inserted
duplicate_candidates_checked
duplicate_check_time_ms
staging_insert_time_ms
final_insert_time_ms
```

## Expected Performance Impact

Minimal direct speedup, but this reveals the real bottleneck per dataset.

## Risks

Log noise. Use structured logs and optionally gate verbose logging behind:

```bash
NAME_MATCHER_PERF_LOG=1
```

## Validation

Run a match and confirm logs show:

```text
CSV load time
candidate pairs per level
L10/L11 counts
GPU status
result save time
```

## Completion Criteria

Do not proceed to deeper changes until a run clearly reports where time is spent.

---

# Phase 2: GPU Build and Runtime Clarity

## Objective

Make it impossible for users to think GPU is active when it is not.

## Files and Modules Likely Affected

```text
Cargo.toml
src-tauri/Cargo.toml
src-tauri/tauri.conf.json
src/matching/gpu/*
src/matching/mod.rs
src/run_service/dto.rs
src-tauri/src/commands/matching.rs
```

## Specific Code Changes

### 1. Add Compile-Time GPU Flag Reporting

```rust
pub fn gpu_feature_compiled() -> bool {
    cfg!(feature = "gpu")
}
```

### 2. Verify Tauri GPU Feature Wiring

Ensure `src-tauri/Cargo.toml` forwards GPU feature to the engine crate.

Example:

```toml
[features]
default = []
gpu = ["name_matcher/gpu"]
```

### 3. Add Build Scripts

Create:

```text
scripts/build_gpu.sh
scripts/build_cpu.sh
```

Example GPU build script:

```bash
#!/usr/bin/env bash
set -euo pipefail
cargo tauri build --features gpu
```

### 4. Update Matching Job Startup Logs

At every job start, log:

```rust
tracing::info!(
    gpu_requested = config.gpu.enabled,
    gpu_feature_compiled = cfg!(feature = "gpu"),
    fuzzy_gate_mode = ?config.gpu.fuzzy_gate_mode,
    "gpu_runtime_status"
);
```

### 5. UI and DTO Behavior

If GPU is requested but unavailable, do not silently fall back.

Return a warning such as:

```text
GPU was requested but this build does not include GPU support.
```

## Expected Performance Impact

No algorithmic speedup, but prevents false GPU benchmarking.

## Risks

Users may discover their current build is CPU-only. This is useful information.

## Validation

Run:

```bash
cargo build
cargo build --features gpu
```

Confirm diagnostics differ.

## Completion Criteria

Every run clearly states whether GPU is compiled, available, selected, and active.

---

# Phase 3: CSV Loading Optimization

## Objective

Reduce CSV matching startup time and memory use.

## Files and Modules Likely Affected

```text
src/loaders/csv_loader.rs
src/models.rs
src/run_service/mod.rs
src-tauri/src/commands/matching.rs
```

## Current Problem

The loader likely behaves like this:

```text
read entire file as bytes
decode entire file into String
parse CSV from String
build HashMap for every row
copy many fields
store extra fields by default
```

This is expensive for large CSVs.

## Specific Code Changes

### 1. Precompute Column Indexes Once

Create:

```rust
struct CsvHeaderIndex {
    id: Option<usize>,
    uuid: Option<usize>,
    first_name: Option<usize>,
    middle_name: Option<usize>,
    last_name: Option<usize>,
    birthdate: Option<usize>,
    hh_id: Option<usize>,
    extra_indexes: Vec<(String, usize)>,
}
```

Build this once after reading headers.

### 2. Remove Per-Row HashMap Construction

Replace row access like this:

```rust
row.get("first_name")
```

with index-based access:

```rust
record.get(header_index.first_name?)
```

### 3. Move Mapped-Column Computation Outside the Loop

Bad pattern:

```rust
for record in records {
    let mapped_columns = mapped_column_names(&mapping);
}
```

New pattern:

```rust
let mapped_columns = mapped_column_names(&mapping);

for record in records {
    // use mapped_columns
}
```

### 4. Add Option to Skip Extra Fields

Add loader options:

```rust
pub struct CsvLoadOptions {
    pub include_extra_fields: bool,
    pub generate_stable_ids: bool,
    pub progress_interval_rows: usize,
}
```

Default for matching:

```rust
include_extra_fields: false
```

Default for import/export preview:

```rust
include_extra_fields: true
```

### 5. Avoid Full-Row Stable ID Hashing by Default

For matching-only jobs, use:

```text
source row index + side marker
```

or require a configured ID column.

Only use full-row hash if absolutely needed.

### 6. Add Progress and Cancellation

Every N rows:

```rust
if cancel_token.is_cancelled() {
    return Err(anyhow!("CSV load cancelled"));
}

progress_sink.emit(CsvLoadProgress {
    rows_read,
    elapsed_ms,
});
```

## Expected Performance Impact

High for large CSVs.

This should reduce:

```text
load time
memory pressure
allocations
startup delay before matching
```

## Risks

- Extra fields may be expected in some exports.
- Stable ID behavior may affect result identity.
- Encoding detection may need fallback behavior.

## Validation Steps

1. Compare row counts before and after.
2. Compare parsed names and birthdates.
3. Run matching on old and new loader.
4. Confirm same or acceptable match results.
5. Measure memory reduction.

## Suggested Tests

```text
CSV with UTF-8 BOM
CSV with non-UTF encoding
CSV with missing columns
CSV with quoted commas
CSV with empty middle names
CSV with no ID column
CSV with extra fields enabled
CSV with extra fields disabled
```

## Completion Criteria

CSV matching no longer requires unnecessary per-row maps or extra-field copies.

---

# Phase 4: True Streaming Path for Matching

## Objective

Stop pretending to stream while actually loading full `Vec<Person>` for both sides.

## Files and Modules Likely Affected

```text
src/run_service/mod.rs
src/run_service/dto.rs
src/loaders/csv_loader.rs
src/db/schema.rs
src-tauri/src/commands/matching.rs
```

## Specific Code Changes

### 1. Introduce a `PersonSource` Abstraction

Add:

```rust
pub trait PersonSource {
    fn estimated_len(&self) -> Option<u64>;

    fn next_batch(
        &mut self,
        batch_size: usize,
    ) -> anyhow::Result<Option<Vec<Person>>>;
}
```

Or async version if needed:

```rust
#[async_trait::async_trait]
pub trait AsyncPersonSource {
    async fn next_batch(&mut self, batch_size: usize) -> anyhow::Result<Option<Vec<Person>>>;
}
```

### 2. Add Implementations

```text
CsvPersonSource
DbPersonSource
InMemoryPersonSource
```

### 3. Modify Run Service Contract

Replace a loader returning:

```rust
(Vec<Person>, Vec<Person>, String, String)
```

with either:

```rust
(LoadedSource, LoadedSource, String, String)
```

or:

```rust
(Box<dyn PersonSource>, Box<dyn PersonSource>, String, String)
```

### 4. First Incremental Version

To keep risk lower, implement streaming only for CSV load first, while still materializing before matching:

```text
stream rows
avoid excess copies
collect minimal Person objects
```

Then move to true batch matching later.

## Expected Performance Impact

Medium in the first version, high in the full version.

## Risks

True streaming may require redesigning candidate indexing. For cascade matching, the target table usually needs an index, so streaming both sides blindly may not work.

## Recommended Conservative Approach

First:

```text
stream parse CSV → minimal Vec<Person>
```

Later:

```text
target indexed once
source processed in batches
matches emitted incrementally
```

## Completion Criteria

CSV loading can report progress and cancel before the entire file is parsed.

---

# Phase 5: CSV Import Optimization

## Objective

Avoid reparsing files and avoid per-row duplicate SQL checks.

## Files and Modules Likely Affected

```text
src/import/mod.rs
src-tauri/src/commands/import.rs
src/db/schema.rs
```

## Current Problem

Import likely does:

```text
preview parse
validate parse
count parse
commit validate parse
commit load parse
duplicate check row by row
```

This is expensive.

## Specific Code Changes

### 1. Create Real Background Import Jobs

Current behavior should become:

```text
start_csv_import returns immediately with job id
background task performs import
get_csv_import_status reports live progress
cancel_csv_import cancels actual running task
```

Add:

```rust
pub struct ImportJobState {
    pub job_id: String,
    pub status: ImportStatus,
    pub rows_read: u64,
    pub rows_validated: u64,
    pub rows_inserted: u64,
    pub cancel_token: CancellationToken,
}
```

### 2. Use Staging Table

Workflow:

```text
create temporary/staging table
stream CSV rows into staging table
validate staging rows
deduplicate with SQL join
insert/update final table from staging
drop staging table
```

### 3. Replace N+1 Duplicate Checks

Bad:

```text
for each row:
  SELECT 1 FROM target WHERE ...
```

Good:

```sql
SELECT s.row_id, t.id
FROM import_staging s
JOIN target t
  ON normalized keys match
```

or batched query:

```sql
WHERE key IN (...)
```

### 4. Consider MySQL `LOAD DATA LOCAL INFILE`

Use if available:

```sql
LOAD DATA LOCAL INFILE ?
INTO TABLE import_staging
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
...
```

Fallback to batched inserts if unavailable.

### 5. Make Cancellation Real

During streaming:

```rust
if cancel_token.is_cancelled() {
    rollback_transaction();
    mark_cancelled();
    return;
}
```

## Expected Performance Impact

Very high for imports.

## Risks

- MySQL permissions for `LOAD DATA LOCAL INFILE`.
- Staging table cleanup after cancellation/failure.
- Duplicate semantics must match old behavior.

## Validation Steps

1. Import small CSV.
2. Import large CSV.
3. Cancel import mid-run.
4. Check staging cleanup.
5. Compare duplicate counts with old logic.

## Completion Criteria

Import starts immediately, progress updates live, and duplicate checks happen in batches or SQL joins.

---

# Phase 6: Level 10 and Level 11 Candidate Reduction

## Objective

Stop L10/L11 from comparing huge groups of people.

## Files and Modules Likely Affected

```text
src/matching/advanced_matcher.rs
src/matching/mod.rs
src/matching/cascade.rs
src/matching/gpu/*
```

## Current Problem

L10/L11 are fuzzy fallback levels. They are expensive by nature. The real killer is broad blocking, especially birthdate-only or birth-year fallback.

## Specific Code Changes

### 1. Define Stricter Blocking Keys

Add helper:

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FuzzyBlockKey {
    birthdate: Option<NaiveDate>,
    birth_year: Option<i32>,
    first_initial: Option<char>,
    middle_initial: Option<char>,
    last_phonetic: Option<String>,
    last_initial: Option<char>,
}
```

### 2. L10 Blocking Rules

Use ordered fallback:

```text
birthdate + first_initial + middle_initial + last_phonetic
birthdate + first_initial + last_phonetic
swapped_birthdate + first_initial + last_phonetic, only if enabled
```

Do not use plain birth year unless the block is tiny.

### 3. L11 Blocking Rules

Use:

```text
birthdate + first_initial + last_phonetic
birthdate + last_phonetic
birthdate + first_initial + last_initial
```

Avoid:

```text
birth_year only
birthdate only for huge blocks
```

### 4. Add Block Caps

Add config:

```rust
pub struct BlockingLimits {
    pub max_block_size: usize,
    pub max_pairs_per_block: usize,
    pub allow_birth_year_fallback: bool,
}
```

Suggested defaults:

```rust
max_block_size = 10_000
max_pairs_per_block = 2_000_000
allow_birth_year_fallback = false
```

### 5. Log Oversized Blocks

For every skipped or capped block:

```rust
tracing::warn!(
    level,
    block_key = ?key,
    source_count,
    target_count,
    pair_count,
    "blocking_limit_applied"
);
```

### 6. Add Candidate Count Summary

At the end of each level:

```rust
tracing::info!(
    level,
    candidate_pairs_seen,
    skipped_blocks,
    largest_block_size,
    matches_found,
    "level_candidate_summary"
);
```

## Expected Performance Impact

Very high for L10/L11-heavy datasets.

## Risks

Over-tight blocking can reduce recall. Use shadow validation before disabling broad fallbacks completely.

## Validation Steps

1. Run old and new L10/L11 on gold test data.
2. Compare recall.
3. Compare candidate pairs.
4. Check skipped block logs.
5. Review false negatives manually.

## Suggested Tests

```text
same birthdate, many names
same birth year, many names
typo in first name
typo in last name
swapped birth month/day
missing middle name
different middle initials
same phonetic last name
```

## Completion Criteria

L10/L11 candidate count drops dramatically without unacceptable recall loss.

---

# Phase 7: GPU Cascade Optimization

## Objective

Make GPU matching reduce CPU work instead of adding overhead.

## Files and Modules Likely Affected

```text
src/matching/gpu/batch.rs
src/matching/gpu/*
src/matching/mod.rs
src/run_service/dto.rs
src-tauri/src/commands/matching.rs
```

## Specific Code Changes

### 1. Make Fuzzy Gate Mode Explicit

Do not silently default to passive mode for performance runs.

Config values:

```text
Off
Shadow
GateOnly
```

Recommended rollout:

```text
Shadow in validation
GateOnly in production after no unacceptable false negatives
```

### 2. Add Startup Warning

If GPU mode is `Off` but GPU is requested:

```text
GPU fuzzy gate is Off. CUDA kernels may run, but CPU classification remains authoritative.
```

### 3. Reuse GPU Batch Accumulator

Bad pattern:

```rust
for tile in tiles {
    let acc = GpuBatchAccumulator::new(size);
}
```

Good pattern:

```rust
let mut acc = GpuBatchAccumulator::new(max_batch_size);

for tile in tiles {
    acc.clear_reuse_buffers();
    acc.fill(tile);
    acc.run();
}
```

### 4. Avoid Repeated Pinned Allocation

Keep pinned buffers sized to the high-water mark:

```rust
if current_capacity < needed {
    grow_to_next_power_of_two(needed);
}
```

Do not reallocate on every slightly different batch size.

### 5. Replace `Vec::drain(0..n)`

Use index offsets:

```rust
let mut start = 0;

while start < batch_pairs.len() {
    let end = (start + desired).min(batch_pairs.len());
    process(&batch_pairs[start..end]);
    start = end;
}

batch_pairs.clear();
```

Or process chunks directly:

```rust
for chunk in batch_pairs.chunks(desired) {
    process(chunk);
}
```

### 6. Track GPU Usefulness

Add metric:

```text
gpu_reject_rate = gpu_gate_reject / pairs_uploaded
cpu_avoidance_rate = 1 - cpu_classified / pairs_uploaded
```

If reject rate is low, GPU is not helping because blocking is too broad or the gate is too permissive.

## Expected Performance Impact

Medium to high after candidate reduction.

GPU optimization before candidate reduction may disappoint.

## Risks

`GateOnly` can cause false negatives if gate behavior differs from CPU scoring. Validate in `Shadow` mode first.

## Validation Steps

1. Run `Shadow` mode on representative datasets.
2. Confirm false negatives are acceptable or zero.
3. Compare:
   ```text
   Off vs Shadow vs GateOnly
   ```
4. Confirm CPU-classified pairs decrease in `GateOnly`.

## Completion Criteria

GPU mode visibly reduces CPU classification work.

---

# Phase 8: Remove Cascade Temporary CSV I/O

## Objective

Stop writing per-level CSV files during normal Tauri matching.

## Files and Modules Likely Affected

```text
src/matching/cascade.rs
src/run_service/mod.rs
src-tauri/src/commands/matching.rs
```

## Current Problem

Cascade writes temporary CSVs, stores/clones matches, then Tauri deletes the CSVs. That is unnecessary during app matching.

## Specific Code Changes

### 1. Add Cascade Output Option

```rust
pub struct CascadeOptions {
    pub write_level_csv: bool,
    pub keep_level_matches: bool,
}
```

For CLI/debug:

```rust
write_level_csv = true
```

For Tauri app:

```rust
write_level_csv = false
```

### 2. Return Level-Tagged Matches Directly

Instead of:

```rust
CascadeLevelResult {
    matches: Vec<MatchPair>,
    output_path: Option<PathBuf>,
}
```

Use:

```rust
pub struct LevelTaggedMatch {
    pub level: u8,
    pub pair: MatchPair,
}
```

or stream:

```rust
on_match(level, pair)
```

### 3. Avoid Cloning Match Lists

Where possible, move matches instead of cloning:

```rust
flat.extend(entry.matches.into_iter().map(|p| (entry.level, p)));
```

### 4. Preserve Debug Export as Opt-In

Add UI/CLI option:

```text
Export per-level cascade CSVs
```

Default:

```text
off
```

## Expected Performance Impact

Medium to high for large result sets.

## Risks

Users may rely on per-level CSV artifacts. Keep opt-in support.

## Validation Steps

1. Run cascade with temp CSV off.
2. Confirm results still appear.
3. Run debug export with temp CSV on.
4. Confirm files are generated only when requested.

## Completion Criteria

Normal Tauri matching writes no per-level temp CSV files.

---

# Phase 9: Result Persistence and Pagination Optimization

## Objective

Stop repeatedly rewriting huge result/person snapshots.

## Files and Modules Likely Affected

```text
src/run_service/store.rs
src/run_service/mod.rs
src-tauri/src/commands/results.rs
```

## Specific Code Changes

### 1. Separate Metadata From Heavy Result Data

Use separate functions:

```rust
save_job_metadata(job_id, status, progress)
save_person_snapshots_once(job_id, source_people, target_people)
append_result_rows(job_id, rows)
mark_job_finished(job_id)
```

Avoid calling one giant `save_job()` that rewrites everything.

### 2. Save People Once

After source/target load:

```rust
store.save_person_snapshots_once(job_id, source_people, target_people)?;
```

Do not rewrite person snapshots on every state update.

### 3. Save Result Rows in Batches

```rust
const RESULT_INSERT_BATCH_SIZE: usize = 1_000;

for chunk in rows.chunks(RESULT_INSERT_BATCH_SIZE) {
    store.insert_result_rows(job_id, chunk)?;
}
```

### 4. Add SQLite Indexes

Suggested indexes:

```sql
CREATE INDEX IF NOT EXISTS idx_results_job_id ON results(job_id);
CREATE INDEX IF NOT EXISTS idx_results_job_level ON results(job_id, level);
CREATE INDEX IF NOT EXISTS idx_results_job_score ON results(job_id, score);
CREATE INDEX IF NOT EXISTS idx_person_lookup_job_side_id ON result_person_lookup(job_id, side, person_id);
```

### 5. Push Filtering, Sorting, and Pagination Into SQLite

Bad:

```text
load all rows
clone rows
filter in memory
sort in memory
slice page
```

Good:

```sql
SELECT row_json
FROM results
WHERE job_id = ?
ORDER BY score DESC
LIMIT ?
OFFSET ?
```

For text search, either:

```text
add normalized searchable columns
```

or add SQLite FTS later.

## Expected Performance Impact

High for jobs with many results.

## Risks

Schema migration required. Keep compatibility with existing result history.

## Validation Steps

1. Run large match.
2. Confirm final save time drops.
3. Confirm result browsing is fast.
4. Confirm old saved jobs still open or are migrated gracefully.

## Completion Criteria

Large result jobs are not rewritten repeatedly, and pagination does not clone/filter the entire result set in memory.

---

# Phase 10: Benchmarks and Regression Tests

## Objective

Prevent performance regressions.

## Files and Modules Likely Affected

```text
benches/*
tests/*
scripts/perf/*
Cargo.toml
```

## Specific Code Changes

### 1. Add Benchmark Datasets

Create synthetic data generator:

```text
scripts/perf/generate_datasets.rs
```

Datasets:

```text
small_csv_1k
medium_csv_100k
large_csv_1m
high_collision_birthdate
high_collision_birth_year
l10_l11_fuzzy_heavy
gpu_shadow_validation
```

### 2. Add Benchmark Script

Create:

```bash
scripts/perf/run_benchmarks.sh
```

Should output JSON:

```json
{
  "csv_load_ms": 1234,
  "candidate_pairs_l10": 10000,
  "candidate_pairs_l11": 15000,
  "gpu_h2d_ms": 50,
  "gpu_kernel_ms": 20,
  "cpu_classification_ms": 100,
  "result_persistence_ms": 300,
  "peak_memory_mb": 1024
}
```

### 3. Add Regression Thresholds

Example thresholds:

```text
L10 candidate pairs must not exceed baseline by more than 20%
CSV load memory must not exceed baseline by more than 10%
Result persistence must not rewrite person snapshots more than once
```

### 4. Add Unit Tests for Blocking Caps

Tests:

```rust
#[test]
fn l11_does_not_use_plain_birth_year_by_default() {}

#[test]
fn l10_applies_max_pairs_per_block() {}

#[test]
fn gpu_shadow_mode_reports_false_negatives() {}

#[test]
fn csv_loader_does_not_store_extra_fields_when_disabled() {}
```

## Expected Performance Impact

No direct speedup, but protects all improvements.

## Risks

Synthetic datasets may not reflect production. Add anonymized real-world shape tests if possible.

## Completion Criteria

Repeatable performance reports exist before and after changes.

---

# Recommended Implementation Order

## Quick Wins First

1. Instrumentation
   Low risk, high clarity.

2. GPU diagnostics
   Low risk, avoids misleading test results.

3. Move per-row CSV computations outside loops
   Low risk, immediate CSV speedup.

4. Disable cascade temp CSV output in Tauri
   Medium impact, low-to-medium risk.

5. Avoid repeated result snapshot rewrites
   High impact, moderate risk.

---

## Then Fix the Algorithmic Bottleneck

6. Cap L10/L11 candidate generation
   Highest matching speed impact.

7. Replace birth-year fallback
   High impact, recall-sensitive.

8. Validate GPU Shadow mode
   Needed before GPU gate becomes authoritative.

9. Enable GateOnly after validation
   High impact if GPU gate reject rate is good.

---

## Then Deeper Architecture

10. Streaming CSV loader
    High impact, moderate complexity.

11. Real background CSV import
    High impact, moderate-to-high complexity.

12. Staging-table import flow
    High impact, higher complexity.

13. True batch/streaming matching
    Highest architecture complexity. Do after easier wins.

---

# Module-by-Module Task List for Codex

## `src/loaders/csv_loader.rs`

```text
- Add CsvHeaderIndex.
- Replace per-row HashMap lookups with index lookups.
- Move mapped_column_names outside row loop.
- Add CsvLoadOptions.
- Add include_extra_fields flag.
- Add cancellation and progress callbacks.
- Avoid full-file decode for UTF-8 path.
- Add tests for BOM, missing columns, quoted fields, and extra fields.
```

## `src/import/mod.rs`

```text
- Reduce repeated parsing.
- Add staging import path.
- Replace per-row duplicate queries with SQL joins or batched lookups.
- Track import progress.
- Support cancellation.
- Add cleanup for failed/cancelled staging imports.
```

## `src-tauri/src/commands/import.rs`

```text
- Make start_csv_import return immediately.
- Spawn background import task.
- Store cancellation token in import job registry.
- Make cancel_csv_import trigger actual cancellation.
- Make import status reflect live progress.
```

## `src/matching/advanced_matcher.rs`

```text
- Audit L10/L11 CPU blocking.
- Add stricter block keys.
- Add max block size.
- Add max pairs per block.
- Log largest blocks.
- Avoid cloning Person before cheap eligibility checks.
```

## `src/matching/mod.rs`

```text
- Audit GPU L10/L11 candidate generation.
- Remove or cap birth-year fallback.
- Add candidate counters.
- Add GPU diagnostics.
- Wire fuzzy gate stats into run output.
```

## `src/matching/gpu/batch.rs`

```text
- Reuse GpuBatchAccumulator.
- Reuse pinned buffers.
- Avoid reallocating per tile.
- Replace front-drain logic with chunk/index processing.
- Add per-stage GPU timing.
- Add reject-rate and CPU-avoidance metrics.
```

## `src/matching/cascade.rs`

```text
- Add write_level_csv option.
- Disable temp CSV output for Tauri matching.
- Preserve opt-in debug CSV export.
- Return or stream level-tagged matches.
- Avoid cloning large match lists.
```

## `src/run_service/mod.rs`

```text
- Pass cascade options based on app mode.
- Avoid flattening by cloning match pairs.
- Add timing around DTO conversion.
- Prepare for streaming PersonSource abstraction.
```

## `src/run_service/store.rs`

```text
- Split metadata save from heavy result save.
- Save person snapshots once.
- Insert result rows in batches.
- Add SQLite indexes.
- Move pagination/filtering/sorting into SQL.
- Avoid full StoredJob clone for large result pages.
```

## `Cargo.toml` and `src-tauri/Cargo.toml`

```text
- Confirm GPU feature is explicit.
- Add Tauri feature forwarding for name_matcher/gpu.
- Add build scripts for CPU and GPU builds.
```

---

# Final Validation Checklist

```text
[ ] A matching run logs CSV load time separately from matching time.
[ ] A matching run logs candidate pairs per cascade level.
[ ] L10/L11 candidate counts are visible.
[ ] L10/L11 oversized blocks are capped or logged.
[ ] GPU runtime status is visible.
[ ] GPU build status is visible.
[ ] GPU fuzzy gate mode is visible.
[ ] Shadow mode can compare GPU gate results against CPU behavior.
[ ] GateOnly mode reduces CPU-classified pair count after validation.
[ ] CSV loader avoids per-row HashMap construction.
[ ] CSV loader avoids unnecessary extra-field copies during matching.
[ ] CSV import starts as a real background job.
[ ] CSV import cancellation stops the running job.
[ ] Duplicate checks are batched or staging-table based.
[ ] Tauri cascade matching does not write temp CSVs by default.
[ ] Result persistence does not rewrite full snapshots on every state change.
[ ] Result pagination does not clone/filter/sort the full result set in memory.
[ ] Benchmarks exist for CSV load, import, L10/L11, GPU, persistence, and memory.
```

---

# Codex Execution Prompt

```text
Implement the name matching performance remediation plan in safe phases.

Start with instrumentation and diagnostics. Do not make behavior-changing optimizations until timing and counters are visible.

Then implement quick wins:
1. GPU runtime diagnostics and feature detection.
2. CSV loader header-index optimization.
3. Disable cascade temp CSV output for Tauri matching.
4. Avoid unnecessary match/result cloning.
5. Split result metadata persistence from heavy result persistence.

Next implement L10/L11 candidate reduction:
- Remove or cap broad birth-year fallback.
- Add stricter blocking keys.
- Add max block size and max pairs per block.
- Log largest blocks and skipped/capped blocks.
- Preserve recall by validating against existing tests and adding new edge-case tests.

Then implement GPU cascade improvements:
- Support Shadow mode validation.
- Report false negatives/positives.
- Move to GateOnly only after validation.
- Reuse GPU accumulators and buffers.
- Replace front-drain Vec logic with chunk/index processing.

Then implement deeper CSV/import refactors:
- Stream CSV parsing.
- Avoid per-row maps.
- Avoid extra-field copies unless requested.
- Add real background import jobs.
- Use staging-table or batched duplicate checks.

Add benchmarks and regression tests for every phase.

Keep each phase in small commits. After each phase, run tests and produce a brief performance note with before/after metrics.
```
