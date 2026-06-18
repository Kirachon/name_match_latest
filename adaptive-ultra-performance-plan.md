# Plan: Adaptive Ultra Performance

**Generated**: 2026-06-18

## Overview

Turn Ultra Performance from a mostly fixed/manual-lockout preset into a safe adaptive planner. The app should choose CPU vs GPU settings from the machine profile and the matching workload before a run starts, while keeping StringZilla CUDA out of the production path.

The implementation should be conservative: improve defaults, emit the chosen plan in logs for auditability, preserve CPU/GPU parity, and avoid changing result semantics.

## Scope Lock

- In scope:
  - Resolve Ultra Performance inside the Rust run-service boundary before engine execution.
  - Use available system profile, CUDA diagnostics, selected algorithm, cascade mode, row counts, and user GPU settings.
  - Prefer GPU for GPU-applicable workloads when CUDA is available and the workload is large enough to benefit.
  - Auto-enable safe GPU helpers for Ultra where they already exist.
  - Auto-pick a bounded VRAM budget when the user leaves it blank.
  - Improve UI wording so Ultra is described as adaptive safe maximum, not an unconditional hardware saturation promise.
  - Add tests for the adaptive planner and frontend validation contract.
  - Rebuild the Tauri app with `--features gpu`.
- Out of scope:
  - Production StringZilla CUDA integration.
  - RAPIDS integration.
  - Changing matching result semantics.
  - Rewriting cascade or fuzzy scoring algorithms.
  - Changing database schema or installer identity.

## Target Files

- `src/run_service/mod.rs`
- `src/run_service/dto.rs`
- `src/matching/gpu_config.rs`
- `ui/src/features/configure/ConfigureCards.tsx`
- `ui/src/shared/tauri/types.ts`
- `ui/src/shared/tauri/zod-schemas.ts`
- `ui/src/__tests__/zod-schemas.test.ts`
- `docs/adaptive-ultra-benchmark-evidence.md`
- `scripts/windows/Build-Tauri-Gpu.ps1`
- Optional focused Rust tests in `src/run_service/mod.rs` or a new small module if needed.

## Dependency Graph

```text
T1 -> T11 -> T2 -> T4 -> T6A -> T6B -> T7A -> T7B -> T9 -> T8 -> T10
T1 -> T11 -> T3 -> T5 -> T6B
T8 -> T8B -> T10
```

## Tasks

### T1: Freeze Current Behavior And Decision Inputs
- **depends_on**: []
- **location**: `src/run_service/mod.rs`, `src/run_service/dto.rs`, `ui/src/shared/tauri/types.ts`
- **description**: Confirm current Ultra, GPU, dynamic tuning, fuzzy gate, row-count, and system-info wiring. End with a contract decision: adaptive resolver stays internal unless a specific missing input requires a DTO field.
- **validation**: File inspection confirms Rust DTOs and TS DTOs remain mirrored; contract decision says "no new DTO" unless proven otherwise.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T11: Plan Review And Party Mode Refinement
- **depends_on**: [T1]
- **location**: `adaptive-ultra-performance-plan.md`
- **description**: Ask subagents to review this plan, including a multi-role discussion. Incorporate actionable feedback before implementation.
- **validation**: Final plan includes accepted feedback or a short rejection rationale.
- **status**: Completed
- **log**: Incorporated backend, frontend/contract, and benchmark feedback: internal resolver, Rust validation, explicit GPU precedence, strict benchmark evidence, global GPU reset, review before packaging, release script/runtime validation.
- **files edited/created**: `adaptive-ultra-performance-plan.md`

### T2: Add A Pure Adaptive Plan Resolver
- **depends_on**: [T11]
- **location**: `src/run_service/mod.rs`
- **description**: Add a small pure helper that takes `RunConfigDto`, algorithm/cascade context, row counts, CUDA diagnostics, and system profile output, then returns an internal `ResolvedPerformancePlan`. Keep it internal; do not expose it through `RunConfigDto`. The helper must compile without the `gpu` feature.
- **validation**: Unit tests cover CPU-only, CUDA unavailable, small workload, medium fuzzy workload, duplicate-heavy/large workload, manual GPU force, missing row counts, and user-provided VRAM budget.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T3: Define Safe Ultra Rules
- **depends_on**: [T11]
- **location**: `src/run_service/mod.rs`, `src/matching/gpu_config.rs`
- **description**: Encode conservative rules:
  - `gpu.mode=cpu`: always CPU and all GPU helpers off.
  - `gpu.mode=auto`: GPU only if compiled with `gpu`, CUDA probe succeeds, workload is eligible, and row scale passes threshold.
  - `gpu.mode=force-gpu`: fail fast if CUDA is unavailable; do not silently CPU fallback except where existing engine OOM fallback is explicitly documented.
  - Ultra can derive effective settings only for blank/auto fields and must not mutate stored frontend state.
  - Missing row counts use conservative CPU/auto-safe behavior unless force-gpu is selected.
  - Fuzzy and fuzzy-no-middle are GPU eligible; cascade uses GPU only for fuzzy-capable levels; deterministic/Levenshtein/household preserve existing behavior unless separately proven.
  - Prefer parity-proven fuzzy gate behavior; do not auto-select risky gate-only unless already proven for that path.
  - VRAM budget should reuse existing GPU budget helpers where practical instead of duplicating formulas.
- **validation**: Tests verify precedence, eligibility, fallback, and budget behavior.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T4: Apply Resolved Plan In Run Service
- **depends_on**: [T2, T3]
- **location**: `src/run_service/mod.rs`
- **description**: Replace scattered direct config reads with the resolved plan before setting global GPU flags and constructing `MatchOptions`. Explicitly set both true and false values every run for direct prefilter, Levenshtein full scoring, dynamic tuning, fuzzy metrics/force, fuzzy gate mode, and prepass budget so one run cannot leak settings into the next. Emit one concise log line with ultra flag, requested GPU mode, resolved backend, CUDA status/error, row counts, algorithm/cascade, Rayon threads, direct prefilter, fuzzy gate mode, dynamic tuning, VRAM budget, and fallback reason.
- **validation**: Existing run-service behavior is preserved when Ultra is off; sequential GPU-on then CPU-off tests prove globals reset; compile and tests pass.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T5: Update UI Copy Without Adding More Controls
- **depends_on**: [T3]
- **location**: `ui/src/features/configure/ConfigureCards.tsx`
- **description**: Change Ultra Performance copy from fixed "maxes out threads and pool" wording to: "Chooses the fastest safe CPU/GPU settings for this machine and workload. Respects CPU, Auto, and Force GPU choices." Keep controls minimal and do not mutate Zustand GPU toggles for effective per-run choices.
- **validation**: TypeScript build and UI tests pass.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T6A: Rust Contract And Validation
- **depends_on**: [T4, T5]
- **location**: `src/run_service/dto.rs`, `src/run_service/mod.rs`
- **description**: Add Rust-side run config validation before resolving performance. Enforce Ultra/manual override exclusivity and GPU-helper restrictions so non-UI callers cannot bypass Zod. Update Rust comments from "max resources" to adaptive safe maximum.
- **validation**: Rust tests mirror the frontend matrix and assert Ultra-off behavior remains unchanged.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T6B: TypeScript/Zod Contract Alignment
- **depends_on**: [T6A]
- **location**: `src/run_service/dto.rs`, `ui/src/shared/tauri/types.ts`, `ui/src/shared/tauri/zod-schemas.ts`, `ui/src/__tests__/zod-schemas.test.ts`
- **description**: Keep DTO fields unchanged unless T1 proves one is required. Add TS/Zod tests for Ultra + CPU, Ultra + Auto blank VRAM, Ultra + manual overrides, and CPU mode with GPU helpers. Add or reuse a complete config fixture that includes every options/gpu/streaming/export/cascade field.
- **validation**: `pnpm --dir ui test`, `pnpm --dir ui build`, and Rust compile pass.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T7A: Current GPU Path Parity Gate
- **depends_on**: [T6B]
- **location**: `src/bin/gpu_string_bench.rs`, `src/benchmarking/mod.rs`, `docs/adaptive-ultra-benchmark-evidence.md`
- **description**: Run CPU oracle vs Current GPU for `small`, `messy`, `medium`, `duplicate-heavy`, and `large` if runtime is acceptable. Keep StringZilla CUDA out of this gate.
- **validation**: Each report has `false_negatives_vs_current=0`, `false_positives_vs_current=0`, `pair_ids_match_current=true`, `canonical_output_hash_matches_current=true`, and `blocking_failure=false`. Ordered hash drift is non-blocking only when canonical hash matches and the report labels it order-only.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T7B: Current GPU Path Performance And App Ultra Evidence
- **depends_on**: [T7A]
- **location**: `docs/adaptive-ultra-benchmark-evidence.md`, `src/run_service/mod.rs`
- **description**: Run repeated CPU/GPU benchmarks for `medium` and `duplicate-heavy` with identical fixture, git SHA, features, CUDA device, warmups, measured runs, and VRAM mode. Record p50/p95/p99 and throughput. Also run an end-to-end run-service/app-path check with Ultra enabled and assert the resolved-plan log appears. Keep any StringZilla shadow benchmark separate and label it non-production.
- **validation**: Evidence doc records commands, JSON output paths, hardware summary, feature flags, dataset hashes, parity fields, speed ratios, Ultra resolved log, and release decision. Parity is a hard gate; performance is a hard gate on the target release machine with repeated runs, otherwise documented as measured ratio/regression risk.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T9: Thermo-Nuclear Code Quality Review
- **depends_on**: [T7B]
- **location**: Changed files
- **description**: Run the strict maintainability review requested by the user before expensive packaging. Fix any blocker findings before proceeding.
- **validation**: Review findings are either resolved or explicitly documented as non-blocking.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T8: Build Tauri GPU App
- **depends_on**: [T9]
- **location**: `scripts/windows/Build-Tauri-Gpu.ps1`, `src-tauri/target/release`, `src-tauri/target/release/bundle`
- **description**: Use or update the Windows GPU build script if valid so dependency checks, UI build, Tauri GPU build, and CUDA runtime DLL handling are covered. Fall back to direct `cargo tauri build --features gpu` only with documented reason.
- **validation**: Release exe plus MSI/NSIS installers are produced; `cargo tree --manifest-path src-tauri/Cargo.toml --features gpu` includes `cudarc` and excludes `experimental-stringzilla-cuda`/StringZilla production features.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T8B: Release Artifact Runtime Smoke
- **depends_on**: [T8]
- **location**: `src-tauri/target/release/name-matcher-tauri.exe`, release bundle directory
- **description**: Smoke the release artifact enough to verify GPU feature packaging: release build exists, CUDA runtime DLLs are present where expected, and CUDA diagnostics report `gpu_feature_compiled=true` through a release-compatible path.
- **validation**: Smoke evidence captured; packaging blocker documented if GUI runtime automation is unavailable.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T10: Commit, Push, And Create Release
- **depends_on**: [T8B]
- **location**: Git branch, remote, release artifacts
- **description**: Commit the completed changes, push the branch, and create a new release containing the GPU-enabled installer artifacts. Include a rollback note for CPU-only build fallback and disabling adaptive Ultra by returning to CPU/manual settings.
- **validation**: `git status` clean except intentional artifacts if not tracked; commit hash exists on remote; release exists and includes expected installer paths/assets.
- **status**: Not Completed
- **log**:
- **files edited/created**:

## Parallel Execution Groups

| Wave | Tasks | Can Start When |
|------|-------|----------------|
| 1 | T1 | Immediately |
| 2 | T11 | T1 complete |
| 3 | T2, T3 | T11 feedback incorporated |
| 4 | T4, T5 | T2/T3 complete for T4, T3 complete for T5 |
| 5 | T6A, T6B | T4/T5 complete, then T6A before T6B |
| 6 | T7A, T7B | T6B complete |
| 7 | T9 | T7B complete |
| 8 | T8, T8B | T9 complete, then T8 before T8B |
| 9 | T10 | T8B complete |

## Testing Strategy

- Rust:
  - Unit tests for adaptive resolver.
  - Rust-side run config validation tests.
  - Sequential GPU-on then CPU-off reset test.
  - `cargo check --locked --bin gpu_string_bench --features "gpu gpu-bench"`
  - Optional separate `cargo check --locked --bin gpu_string_bench --features "gpu gpu-bench experimental-stringzilla-cuda"` for shadow-only research.
  - Focused release-machine benchmark runs on `medium` and `duplicate-heavy` CPU/GPU.
- Frontend:
  - `pnpm --dir ui test`
  - `pnpm --dir ui build`
- Tauri:
  - `scripts/windows/Build-Tauri-Gpu.ps1` if usable, otherwise documented direct build.
  - Confirm `cudarc` in feature tree and StringZilla CUDA absent from production feature tree.

## Risks And Mitigations

- **Risk**: Ultra silently changes semantics.
  - **Mitigation**: Only change backend/settings; verify CPU/GPU parity.
- **Risk**: GPU selected for tiny workloads and becomes slower.
  - **Mitigation**: Small-workload threshold keeps CPU unless user forces GPU.
- **Risk**: VRAM budget too aggressive.
  - **Mitigation**: Use bounded fraction and preserve dynamic tuning/fallback.
- **Risk**: Global GPU flags leak between runs.
  - **Mitigation**: Resolve and set all relevant flags explicitly per run.
- **Risk**: Release cannot be created due missing remote/auth.
  - **Mitigation**: Stop with exact missing credential/remote evidence.

## Final Readiness Gate

The work is ready only when:

- The adaptive resolver is implemented and covered by tests.
- Ultra off behavior is unchanged.
- Ultra on logs a concrete adaptive plan.
- CPU/GPU parity remains clean on representative datasets.
- GPU performance evidence is recorded with repeated-run p50/p95/p99 comparisons, and any target-machine regression is treated as a blocker or documented risk.
- Tauri GPU release build succeeds.
- Thermo-nuclear review has no unresolved blockers.
- Commit, push, and release creation succeed or the exact external blocker is documented.
