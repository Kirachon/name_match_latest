# Tauri v2 Migration Plan - name_matcher

**Plan ID:** `tauri-migration-final-2026-05-23`
**Status:** Final reviewed plan
**Confidence:** 92%
**Reviewed by:** Backend architecture, frontend implementation, release/build council, plus local UX pass
**Estimated Effort:** 7-9 weeks for one developer, shorter with parallel frontend/backend work

## Executive Decision

Migrate the user-facing desktop app to **Tauri v2 + React + TypeScript**, while keeping the Rust matching engine as the source of truth. Do not rewrite matching logic in JavaScript. Tauri should become a polished desktop shell around the existing Rust database, matching, GPU, export, and configuration code.

The migration must avoid creating a second matching path. The CLI, legacy egui GUI, and Tauri commands should converge on a shared Rust run service so CPU/GPU parity, cancellation, streaming, export, and performance behavior do not drift.

## Current Codebase Reality

- Existing desktop GUI lives in `src/bin/gui.rs` using `eframe`/`egui`.
- Core matching, GPU, streaming, and export logic already lives in the root `name_matcher` crate.
- The current crate has feature flags for `gpu`, `new_cli`, and `new_engine`, but `eframe`, `egui`, and `rfd` are not yet optional.
- The project already has a growing orchestrator layer in `src/orchestrator/mod.rs`, but much run orchestration is still concentrated in `src/main.rs` and `src/bin/gui.rs`.
- GPU builds require CUDA-capable Windows runners; CPU builds can run on normal GitHub-hosted Windows runners.

## Target Architecture

```text
Tauri v2 Desktop Shell
  ui/ React + TypeScript + Tailwind
    app shell
    Connect tab
    Configure tab
    Run tab
    Results tab
    EventBridge
    generated Rust bindings

  src-tauri/
    typed Tauri commands
    AppState
    DbSession registry
    JobRegistry
    ResultStore
    EventSink
    capability permissions

  name_matcher crate
    shared RunService
    db/
    matching/
    engine/
    export/
    config/
    optional GPU feature
```

## Scope

### Included

- Tauri v2 desktop shell under `src-tauri/`.
- React + TypeScript + Tailwind frontend under `ui/`.
- Shared Rust backend service contract for CLI, legacy egui, and Tauri.
- Typed Tauri commands for database, matching, configuration, diagnostics, results, and export.
- `tauri-specta`/`specta` type generation or equivalent generated TypeScript binding workflow.
- Tauri v2 capabilities/permissions from the first scaffold step.
- Event bridge for progress, logs, job state, and errors.
- Run-scoped result pagination so the frontend never receives huge result payloads.
- Windows CPU and GPU release lanes.
- Legacy egui coexistence behind a `gui` feature during rollout.
- UI/UX improvements for an operational data-matching tool.

### Excluded

- Rewriting matching algorithms.
- Removing the CLI.
- Deleting the old egui GUI during the initial Tauri rollout.
- Auto-updater endpoint implementation.
- Linux/macOS Tauri packaging unless explicitly added later.
- Moving to a Cargo workspace in the first migration.

## Non-Negotiable Contracts

### Backend Boundary

- Tauri commands must accept typed DTOs only.
- No Tauri command may simulate CLI argv parsing or rely on per-run process environment mutation.
- Convert `MatchParams` into shared Rust structs such as `RunConfig`, `StreamingConfig`, `MatchOptions`, and `GpuConfig`.
- CLI and Tauri should call the same shared run service or adapter.
- CPU classification remains the final authority for fuzzy/GPU paths unless a separate parity-approved rollout changes that.

### Runtime And Locking

- Tauri async commands may hold `RwLock` guards only long enough to clone handles or update metadata.
- No lock guard may be held across engine execution, DB streaming loops, export writes, GPU work, or event emission.
- Long matching jobs run on dedicated OS threads, not normal `tokio::spawn`.
- Do not create a second Tokio runtime inside Tauri.
- Rayon global pool should be initialized before `tauri::Builder`, leaving cores for Tauri/tokio/OS.

### Cancellation

- Cancellation is cooperative but must be bounded.
- Cancellation must be checked at DB page boundaries, matching batch/tile boundaries, GPU flush boundaries, and before export/result-store writes.
- Job state must transition through `cancelling` and end in `cancelled` or `failed`.
- Cancellation must close result writers safely and release the active job slot.

### Results

- `start_matching` returns `job_id` only.
- Results are read through `get_results_page(job_id, cursor/page, limit <= 10000, sort, filter)`.
- Result storage must be run-scoped: SQLite sidecar, temp database table, or file-backed index.
- The frontend must not accumulate complete result sets in React state.
- Export must read from the same result store, not from frontend memory.

### Events

- Backend emits progress at a capped rate, initially 20Hz.
- Frontend has one app-root listener per event type and cleans up every `listen()` subscription.
- Events include `job_id` so stale progress from old jobs cannot update the active UI.
- Logs use a bounded ring buffer and do not print secrets.
- The progress forwarder must terminate when the job finishes/cancels and the channel is disconnected.

## UI/UX Direction

This is an operational desktop tool, so the UI should be dense, clear, and work-focused rather than a marketing-style app.

### Primary Workflow

1. Connect to database or databases.
2. Select source/target tables and verify schema.
3. Configure algorithm, matching mode, GPU/performance options, and export target.
4. Review a configuration summary.
5. Run matching with visible progress, logs, and cancellation controls.
6. Review results, export outputs, and diagnostics.

### Layout

- Persistent tabs: `Connect`, `Configure`, `Run`, `Results`.
- Sequential unlocks are allowed, but locked tabs must show the missing prerequisite.
- A compact status rail should show connection, selected tables, algorithm, GPU mode, output path, and active job state.
- Advanced settings should be collapsible and grouped by purpose: Matching, GPU, Streaming, Export, Diagnostics.
- GPU controls should be visible but disabled with clear helper text when GPU mode is off or CUDA is unavailable.

### Required Job States

Use explicit states in both Rust DTOs and frontend stores:

- `idle`
- `validating`
- `starting`
- `running`
- `pausing`
- `paused`
- `resuming`
- `cancelling`
- `cancelled`
- `failed`
- `completed`

### Results Table

- Paginated and virtualized.
- Stable sort keys.
- Cursor/page parameters.
- Total count.
- Selected columns.
- Sticky header.
- Column resize.
- Keyboard row navigation.
- Copy cell/row.
- Empty/error/loading states.
- Export-state indicators.

### Accessibility

Accessibility is not a final polish-only task. Each UI step must include:

- Keyboard tab navigation.
- Visible focus rings.
- ARIA tab semantics.
- Accessible validation errors.
- Progress announcements that do not spam screen readers.
- Log console with keyboard navigation and filtering.

## Recommended Tech Stack

### Backend

- Tauri v2
- `tauri-plugin-dialog`
- `tauri-plugin-store`
- `tauri-specta` + `specta` or a comparable generated-binding workflow
- `tokio`
- `tokio-util` for `CancellationToken`
- `serde`, `serde_json`, `thiserror`
- root `name_matcher` crate as a path dependency

### Frontend

- Vite
- React + TypeScript
- Tailwind CSS
- Zustand, split by update frequency
- `@tanstack/react-virtual`
- `react-hook-form`
- `zod`
- `@tauri-apps/api` v2
- Vitest + Testing Library
- Playwright smoke tests for the web UI shell

## Frontend Architecture Contract

```text
ui/src/app/
  App.tsx
  routes-or-tabs.tsx
  EventBridge.tsx

ui/src/features/connect/
ui/src/features/configure/
ui/src/features/run/
ui/src/features/results/

ui/src/shared/components/
ui/src/shared/tauri/
  commands.ts
  events.ts
  generated-bindings.ts

ui/src/shared/stores/
  connectionStore.ts
  settingsStore.ts
  jobStore.ts
  progressStore.ts
  logStore.ts
  resultsStore.ts
```

Store guidance:

- Connection/settings stores update infrequently.
- Job/progress/log stores update frequently and must avoid broad rerenders.
- Logs should keep raw entries in a ring buffer outside heavy React render paths.
- Progress updates should be batched with `requestAnimationFrame` or throttled store actions.
- Zod schemas should be generated from or kept adjacent to Rust-generated types, with explicit mappings for UI-only fields.

## Implementation Plan

### Step 0: Feature Gates, Scaffold, And Permissions

**Priority:** Critical
**Effort:** 3-5 days

Tasks:

- Make `eframe`, `egui`, and `rfd` optional behind a `gui` feature.
- Add `[[bin]] name = "gui" required-features = ["gui"]`.
- Create `src-tauri/Cargo.toml`.
- Add Tauri v2 dependencies and plugins.
- Create `src-tauri/tauri.conf.json`.
- Create `src-tauri/capabilities/default.json`.
- Add minimal commands: `system_info`, `load_config`, `save_config`.
- Add structured `AppError` with `Serialize` and generated TypeScript type support.
- Initialize Rayon pool before `tauri::Builder`.
- Add first CI checks for root crate CPU, `gui`, and frontend build.

Acceptance:

- `cargo tauri dev` opens a blank window.
- Tauri commands are explicitly allowlisted in capabilities.
- Legacy `gui` binary still builds with `--features gui`.

### Step 1: Frontend Shell And Baseline UX

**Priority:** High
**Effort:** 2-4 days
**Depends on:** Step 0

Tasks:

- Create `ui/` with Vite, React, TypeScript, and Tailwind.
- Implement the folder architecture contract.
- Build the tab shell: Connect, Configure, Run, Results.
- Add status rail and locked-tab prerequisite messages.
- Add first Zustand stores.
- Add baseline keyboard navigation, ARIA tab semantics, and focus rings.
- Add dark-first theme with restrained operational styling.

Acceptance:

- `pnpm dev` renders the shell.
- Keyboard navigation works across tabs.
- No marketing hero page or decorative landing screen.

### Step 2: Database Commands And Connect Tab

**Priority:** High
**Effort:** 1 week
**Depends on:** Step 1

Tasks:

- Implement `DbSession` handles with `MySqlPool`, sanitized identity, schema cache, selected tables, and timestamps.
- Avoid storing passwords after pool construction unless encrypted storage is explicitly enabled.
- Commands: `connect_db`, `test_connection`, `list_tables`, `get_table_columns`, `get_row_count`, `estimate_tables`, `disconnect_db`.
- Add TTL schema metadata cache.
- Add dual DB support.
- Build Source/Target connection cards and table selectors.
- Add schema quality view and estimate row counts.

Acceptance:

- User can connect to MySQL, list tables, see columns, and estimate counts.
- Repeated schema calls hit cache.
- Secrets are not printed in logs or stored in plain frontend state.

### Step 2.5: Shared Run Service Contract

**Priority:** Critical
**Effort:** 1 week
**Depends on:** Step 2

Tasks:

- Extract a shared backend service boundary before wiring Tauri matching.
- Define `MatchParams`, `RunConfig`, `DbSessionRef`, `JobRegistry`, `ResultStore`, and `EventSink`.
- Convert UI DTOs into `RunConfig`, `StreamingConfig`, `MatchOptions`, and `GpuConfig`.
- Keep CLI behavior by adapting existing CLI inputs into the same service.
- Keep legacy egui behavior by adapting existing GUI inputs into the same service where feasible.
- Add parity tests around config conversion for major algorithms and GPU flags.

Acceptance:

- Tauri does not call CLI argv parsing.
- Per-run options do not require process-wide environment mutation.
- CLI and Tauri can share the same run-service path.

### Step 3: Matching Bridge, Events, And Job Lifecycle

**Priority:** Critical
**Effort:** 1-2 weeks
**Depends on:** Step 2.5

Tasks:

- Commands: `start_matching`, `cancel_matching`, `pause_matching`, `resume_matching`, `get_matching_status`.
- `start_matching` returns `job_id`.
- Run matching on a dedicated OS thread.
- Use `std::sync::mpsc` from engine callbacks to a Tauri event forwarder.
- Add minimal `EventBridge` now, not later.
- Emit `match-progress`, `job-state`, `log-entry`, and `job-error` events.
- Add concrete forwarder termination condition when channel disconnects and job reaches terminal state.
- Implement cancellation checks at chunk, batch, GPU flush, and export boundaries.

Acceptance:

- User can start, pause, resume, and cancel a run.
- Events update only the active `job_id`.
- No orphan event forwarder remains after completion/cancel.

### Step 4: Generated Types And Validation

**Priority:** High
**Effort:** 2-4 days
**Depends on:** Step 3

Tasks:

- Derive generated TypeScript types for DTOs.
- Generate frontend bindings during build or with a checked command.
- Keep Zod validation schemas adjacent to generated types.
- Explicitly map UI-only fields to Rust DTOs.
- Add validation rules for dependent options:
  - GPU memory fields disabled unless GPU mode is enabled.
  - Output directory required before run.
  - Cascade/geographic options disabled when required columns are unavailable.
  - Streaming fields range-checked.
  - Export format required.

Acceptance:

- `pnpm build` uses generated types.
- Invalid config cannot start a run.
- Generated bindings fail CI if stale.

### Step 5: Configure Tab

**Priority:** High
**Effort:** 1 week
**Depends on:** Step 4

Tasks:

- Algorithm selector for all `MatchingAlgorithm` variants.
- Mode selector: Auto, Streaming, InMemory.
- Advanced matching controls.
- Cascade controls and missing-column handling.
- Fuzzy threshold slider.
- Birthdate swap toggle.
- Export format and output directory picker.
- GPU controls:
  - Off, Auto, Force.
  - GPU hash join.
  - GPU fuzzy/direct/prepass controls.
  - GPU memory budgets.
  - Dynamic tuning.
  - Pinned host/buffer pool toggles.
  - CUDA diagnostics.
- Performance controls:
  - Pool size.
  - Batch size.
  - Rayon threads.
  - Memory threshold.
  - Partition strategy.
- Utility actions:
  - Ultra Performance Mode.
  - Configuration Summary.
  - Estimate.
  - System Information.

Acceptance:

- All existing important egui advanced settings have Tauri equivalents.
- Settings are grouped, validated, and dependency-aware.
- User sees a pre-run summary before starting.

### Step 6: Run And Results Tabs

**Priority:** High
**Effort:** 1 week
**Depends on:** Steps 3 and 4

Run tab tasks:

- Pipeline chips: Load, Hash, Fuzzy, Export.
- Progress bar, ETA, records/sec, batch count, elapsed time.
- GPU active/VRAM indicators.
- Memory used/available.
- Virtualized log console with level filters.
- Pause, Resume, Cancel controls with explicit state handling.

Results tab tasks:

- `get_results_page` command with `limit <= 10000`.
- Stable sort/filter/cursor contract.
- Virtualized table with sticky headers and column resize.
- Match count summary.
- Run summary.
- Export buttons.
- Open output folder.
- Diagnostics export.

Acceptance:

- Full workflow is visible from run start through result review.
- Results stay responsive with large result sets.
- Frontend never stores the full result set.

### Step 7: Event And UI Performance Polish

**Priority:** High
**Effort:** 3-5 days
**Depends on:** Step 6

Tasks:

- Harden app-root `EventBridge`.
- One listener per event type.
- Listener cleanup verified in tests.
- Progress capped at 20Hz.
- Logs capped at 100/sec.
- Frontend update batching with `requestAnimationFrame` or equivalent throttling.
- Ring buffers drop oldest entries.
- Add stale-job event protection.

Acceptance:

- UI remains smooth during heavy matching.
- No duplicate listeners after tab navigation or hot reload.
- Log filtering stays responsive with 10K entries.

### Step 8: Build, Release, GPU, And Coexistence

**Priority:** High
**Effort:** 1 week
**Depends on:** Steps 5-7

Release lanes:

- `windows-tauri-cpu` on GitHub-hosted `windows-latest`.
- `windows-tauri-gpu` on self-hosted `[self-hosted, windows, cuda]`.
- `legacy-egui-cpu` during coexistence.
- `legacy-egui-gpu` on CUDA runner during coexistence.

Tasks:

- Add `Build-Tauri-Cpu.ps1`.
- Add `Build-Tauri-Gpu.ps1`.
- Forward `gpu` feature from `src-tauri` to `name_matcher/gpu`.
- List exact CUDA DLL/resource requirements for the packaged GPU build.
- Add packaging inspection that fails if required GPU resources are missing.
- Add `cuda_diagnostics` command and packaged-app smoke check.
- Add WebView2 policy:
  - document bootstrapper vs fixed runtime choice.
  - smoke test clean Windows profile behavior.
  - provide clear failure messaging.
- Add signing lanes:
  - unsigned CI.
  - optional test-signed.
  - production-signed release.
- Verify signature after packaging when signing is enabled.
- Keep separate artifact names:
  - `name-matcher-tauri-windows-cpu-<tag>.msi`
  - `name-matcher-tauri-windows-gpu-<tag>.msi`
  - `name-matcher-egui-windows-cpu-<tag>.zip`
  - `name-matcher-egui-windows-gpu-<tag>.zip`

Acceptance:

- `cargo tauri build` produces an installable Windows artifact.
- Packaged app launches and reaches ready window.
- `system_info`, `load_config/save_config`, and `cuda_diagnostics` smoke commands pass as applicable.
- Legacy egui artifact still builds during coexistence.
- If Tauri packaging/signing fails, legacy egui release can still be published and Tauri is marked prerelease/blocked.

### Step 9: Accessibility, Error UX, And Persistence

**Priority:** Medium
**Effort:** 1 week
**Depends on:** Step 8

Tasks:

- App icon and window title.
- Keyboard shortcuts:
  - Ctrl+Enter to start.
  - Escape to request cancel.
- Graceful shutdown:
  - close window.
  - request cancel.
  - wait for safe stop or ask user to force close.
- Connection persistence with `tauri-plugin-store`.
- Password persistence only if encrypted or explicitly opted in.
- Loading states for all async commands.
- Error taxonomy:
  - inline validation errors.
  - non-fatal log/badge errors.
  - fatal modal errors.
- Toasts only for confirmations.

Acceptance:

- Keyboard-only workflow is usable.
- Fatal and non-fatal failures are understandable.
- Shutdown during a run is safe.

### Step 10: Documentation And Migration Cleanup

**Priority:** Medium
**Effort:** 3-5 days
**Depends on:** Step 9

Tasks:

- README: Tauri dev, CPU build, GPU build, troubleshooting.
- Document WebView2 and CUDA prerequisites.
- Document release lanes and artifact names.
- Mark `src/bin/gui.rs` as deprecated, but do not delete it.
- Document environment variables and which ones are replaced by typed run options.
- Document auto-updater endpoint design without implementing it.

Acceptance:

- New developer can build and run Tauri from README alone.
- Operator can choose CPU/GPU/legacy artifacts correctly.

## Test And Verification Gates

### Backend

- `cargo check --locked`
- `cargo check --locked --features gui`
- `cargo check --locked --features gpu` on CUDA runner
- Unit tests for DTO-to-run-config conversion
- Cancellation tests around job state transitions
- Result pagination tests
- CPU/GPU parity tests before exposing GPU controls

### Frontend

- `pnpm build`
- `pnpm test`
- Vitest/Testing Library for:
  - tab unlock flow
  - invalid config blocking run
  - dependent option disabling
  - event progress updates
  - log filtering
  - cancellation state
  - results pagination

### Tauri

- `cargo tauri dev` smoke
- `cargo tauri build` CPU
- `cargo tauri build --features gpu` on CUDA runner
- Packaged app launches
- Command smoke:
  - `system_info`
  - `load_config`
  - `save_config`
  - `cuda_diagnostics` for GPU build
- Capability permission check for every command

### Release

- Artifact exists.
- MSI/installer opens.
- WebView2 behavior verified.
- Required CUDA resources present in GPU artifact.
- Signature verified for signed release.
- No secrets in logs.
- Legacy egui artifact still available until Tauri release is proven.

## Timeline

| Week | Work | Milestone |
|---|---|---|
| 1 | Steps 0-1 | Tauri scaffold and frontend shell |
| 2 | Step 2 | Database connection works |
| 3 | Step 2.5 | Shared backend service contract |
| 4 | Step 3 | Matching bridge, events, cancellation |
| 5 | Step 4 | Generated types and validation |
| 6 | Steps 5-6 | Configure, Run, Results functional |
| 7 | Step 7 | Event and UI performance polish |
| 8 | Step 8 | Windows CPU/GPU packaging |
| 9 | Steps 9-10 | Accessibility, docs, cleanup |

Parallel option:

- Frontend shell and backend service extraction can run in parallel after Step 0.
- Configure tab and Results tab can run in parallel after generated DTOs are stable.
- Release scripts can start once Step 0 scaffold exists, then harden after Step 6.

## Swarm Planner Execution Layer

This section is the dependency-aware handoff for custom subagents. It freezes the plan into atomic work packets with clear ownership, dependencies, validation, and merge gates.

### Scope Lock

In scope for the swarm:

- Build a Tauri v2 desktop app around the existing Rust engine.
- Preserve CLI behavior and keep the legacy egui binary available during rollout.
- Extract shared run orchestration before Tauri matching commands call engine code.
- Add typed commands, generated frontend bindings, event streaming, cancellation, paginated results, release lanes, documentation, and smoke tests.

Out of scope for the first swarm:

- Rewriting matching algorithms in TypeScript.
- Removing CLI or egui.
- Implementing an auto-update server.
- Adding Linux/macOS production packaging.
- Moving the project into a Cargo workspace unless a task explicitly proves it is needed.

### Dependency Graph

```text
T0 -> T0.5 -> T1 -> T3 -> T3.5 -> T5 -> T8 -> T10 -> T11 -> T13
T0 -> T0.5 -> T2 -> T4 -> T6 ----^      \-> T9 -/
T0 -> T0.5 -> T14 -> T5/T7/T9/T13
T3.5 -> T7 -> T8/T9
T2 -> T12 -> T13
```

### Atomic Tasks

#### T0: Baseline Inventory And Branch Guard

- **depends_on**: []
- **owner**: repo lead / integrator
- **location**: `Cargo.toml`, `Cargo.lock`, `.github/workflows/*`, `src/bin/gui.rs`, `src/main.rs`, `src/orchestrator/mod.rs`, `README.md`
- **description**: Capture current build commands, feature flags, binaries, release artifacts, GUI settings, CLI flags, and matching entrypoints. Record which behavior must be preserved before any scaffold work starts.
- **validation**: `git status --short`, `cargo check --locked`, current GUI build command, and a short inventory note committed with the migration branch.
- **status**: Not Completed
- **log**:
- **files edited/created**:

#### T0.5: Toolchain, Runtime, And Lockfile Baseline

- **depends_on**: [T0]
- **owner**: repo lead / release engineering agent
- **location**: `README.md`, `docs/installation.md`, `Cargo.toml`, `Cargo.lock`, future `ui/package.json`, future `ui/pnpm-lock.yaml`, future `rust-toolchain.toml`
- **description**: Pin or document required Rust, MSVC Build Tools, Node, pnpm, Tauri CLI or `cargo-tauri`, WebView2 runtime policy, CUDA Toolkit/runtime assumptions, and lockfile ownership before scaffold work begins.
- **validation**: fresh shell can print the pinned versions, `cargo metadata --locked` succeeds, frontend package manager policy is documented, and CI uses the same versions.
- **status**: Not Completed
- **log**:
- **files edited/created**:

#### T1: Cargo Feature Split And Legacy GUI Guard

- **depends_on**: [T0.5]
- **owner**: Rust platform agent
- **location**: `Cargo.toml`, `src/bin/gui.rs`, `scripts/windows/Build-Release-Gui.ps1`, `.github/workflows/*`
- **description**: Make egui dependencies optional behind a `gui` feature, add `required-features = ["gui"]` for the legacy `gui` binary, and update build scripts/workflows to request the feature explicitly.
- **validation**: `cargo check --locked`, `cargo check --locked --features gui`, `cargo build --release --locked --features gui --bin gui`, and `cargo tree -e features` or equivalent proof that CPU CLI builds no longer pull egui/eframe/rfd unless `--features gui` is set.
- **status**: Not Completed
- **log**:
- **files edited/created**:

#### T2: Tauri v2 Scaffold And Permission Baseline

- **depends_on**: [T0.5]
- **owner**: Tauri platform agent
- **location**: `src-tauri/Cargo.toml`, `src-tauri/tauri.conf.json`, `src-tauri/capabilities/default.json`, `src-tauri/src/main.rs`
- **description**: Add the Tauri v2 shell, minimal commands, app state, capability allowlist, plugin setup, and CPU-only launch path.
- **validation**: `cargo tauri dev` opens a window, `cargo tauri build --debug` validates `tauri.conf.json`, capability JSON is schema-valid, capabilities include every command, and `system_info`, `load_config`, `save_config` command smoke tests pass.
- **status**: Not Completed
- **log**:
- **files edited/created**:

#### T3: Shared Run Service Extraction

- **depends_on**: [T1]
- **owner**: backend architecture agent
- **location**: `src/orchestrator/mod.rs`, `src/main.rs`, `src/bin/gui.rs`, `src/config.rs`, `src/matching/*`, `src/export/*`
- **description**: Extract shared `RunService`, `RunConfig`, `MatchParams`, `JobConfig`, `EventSink`, and result-writing contracts so CLI, egui, and Tauri do not fork matching behavior.
- **validation**: CLI smoke still runs, legacy GUI still builds, config conversion unit tests cover all algorithms and GPU flags, no Tauri command calls CLI argv parsing.
- **status**: Not Completed
- **log**:
- **files edited/created**:

#### T3.5: Backend Contract Freeze And API Review

- **depends_on**: [T3, T4]
- **owner**: integrator / backend architecture agent
- **location**: `src/orchestrator/mod.rs`, `src-tauri/src/dto/*`, `src-tauri/src/jobs/*`, `ui/src/shared/tauri/*`, `docs/tauri-migration-plan.md`
- **description**: Freeze the cross-agent contracts before feature work fans out: `MatchParams`, `RunConfig`, `DbSessionRef`, `JobRegistry`, `EventSink`, `ResultStore`, error DTOs, event names, generated binding output paths, and checked-in/generated file policy.
- **validation**: API review checklist is recorded, generated binding command is named, output ownership is documented, CI has a stale-binding diff check, and T5/T6/T7/T9 can implement against the same contract without inventing parallel DTOs.
- **status**: Not Completed
- **log**:
- **files edited/created**:

#### T4: React TypeScript Frontend Shell

- **depends_on**: [T2]
- **owner**: frontend agent
- **location**: `ui/package.json`, `ui/src/app/*`, `ui/src/features/*`, `ui/src/shared/*`
- **description**: Create Vite React TypeScript shell with `Connect`, `Configure`, `Run`, and `Results` tabs, status rail, locked-tab prerequisites, accessible tab semantics, and initial Zustand stores.
- **validation**: `pnpm install`, `pnpm build`, `pnpm test`, keyboard tab flow smoke, no full-result-set state in frontend stores.
- **status**: Not Completed
- **log**:
- **files edited/created**:

#### T5: Database Session Commands

- **depends_on**: [T3.5, T4, T14]
- **owner**: backend/database agent
- **location**: `src-tauri/src/commands/*`, `src-tauri/src/state/*`, `src/db/*`, `ui/src/features/connect/*`
- **description**: Implement `connect_db`, `test_connection`, `list_tables`, `get_table_columns`, `get_row_count`, `estimate_tables`, and `disconnect_db` using run-scoped handles and sanitized metadata.
- **validation**: MySQL connection smoke, schema cache TTL test, dual-source table selection test, no secrets in logs or frontend state snapshots.
- **status**: Not Completed
- **log**:
- **files edited/created**:

#### T6: Generated Types And Config Validation

- **depends_on**: [T3.5]
- **owner**: contract/type agent
- **location**: `src-tauri/src/dto/*`, `ui/src/shared/tauri/*`, `ui/src/features/configure/*`
- **description**: Add generated TypeScript bindings, adjacent Zod schemas, DTO-to-run-config mapping, and dependency-aware validation for algorithm, GPU, streaming, export, cascade, and birthdate-swap options. Define the exact generation command, output path, checked-in versus generated policy, and CI diff check.
- **validation**: generated bindings are reproducible, stale bindings fail CI, invalid configs cannot call `start_matching`, DTO conversion tests cover every matching mode, and generated files have one owning task.
- **status**: Not Completed
- **log**:
- **files edited/created**:

#### T7: Job Registry, Events, And Cancellation

- **depends_on**: [T3.5, T4, T14]
- **owner**: runtime agent
- **location**: `src-tauri/src/jobs/*`, `src-tauri/src/events/*`, `src/orchestrator/mod.rs`, `ui/src/app/EventBridge.tsx`, `ui/src/features/run/*`
- **description**: Implement `start_matching`, `pause_matching`, `resume_matching`, `cancel_matching`, `get_matching_status`, dedicated worker threads, capped event forwarding, stale-job protection, and bounded cooperative cancellation.
- **validation**: lifecycle tests for `idle` through terminal states, cancellation stops at DB/batch/GPU/export boundaries, event listener cleanup test, no lock guard held across long-running work, and IPC serialization tests cover errors, progress events, log entries, and maximum allowed result-page metadata.
- **status**: Not Completed
- **log**:
- **files edited/created**:

#### T8: Configure And Run Workflow

- **depends_on**: [T5, T6, T7]
- **owner**: product frontend agent
- **location**: `ui/src/features/configure/*`, `ui/src/features/run/*`, `ui/src/shared/stores/*`
- **description**: Build full settings UI, GPU/performance panels, pre-run summary, run progress, ETA, logs, pause/resume/cancel controls, and visible disabled states.
- **validation**: Testing Library coverage for tab unlock, validation blocking, dependent disabled controls, progress updates, cancellation UI, and keyboard operation.
- **status**: Not Completed
- **log**:
- **files edited/created**:

#### T9: Result Store, Pagination, Export, And Diagnostics

- **depends_on**: [T6, T7, T14]
- **owner**: results/export agent
- **location**: `src-tauri/src/results/*`, `src/export/*`, `ui/src/features/results/*`
- **description**: Add run-scoped result storage, `get_results_page`, stable sort/filter/cursor contracts, export-from-result-store commands, diagnostics export, and output-folder integration.
- **validation**: pagination tests with large result sets, export reads backend result store only, frontend memory profile confirms no full result accumulation, diagnostics redact secrets, and Tauri serialization stays under the maximum page-size IPC budget.
- **status**: Not Completed
- **log**:
- **files edited/created**:

#### T10: UI Performance And Accessibility Hardening

- **depends_on**: [T8, T9]
- **owner**: UX quality agent
- **location**: `ui/src/app/*`, `ui/src/shared/components/*`, `ui/src/features/*`
- **description**: Harden the completed feature screens with event batching, log ring buffers, virtualized tables, focus rings, ARIA labels, screen-reader-safe progress announcements, and keyboard shortcuts. Shared primitives can be prepared earlier, but feature-screen edits wait until T8/T9 are merged.
- **validation**: Playwright keyboard smoke, listener leak test, log filtering with 10K entries, virtualized result table stays responsive.
- **status**: Not Completed
- **log**:
- **files edited/created**:

#### T11: Build, Release, Signing, And GPU Packaging

- **depends_on**: [T8, T9, T10]
- **owner**: release engineering agent
- **location**: `.github/workflows/ci.yml`, `.github/workflows/release.yml`, `scripts/windows/*`, `src-tauri/tauri.conf.json`, `README.md`
- **description**: Add CPU/GPU Tauri build scripts, update existing CI/release workflows, preserve legacy egui Windows and Linux artifacts during coexistence, define WebView2 policy, artifact naming, optional signing, packaged-app smoke tests, installer install/uninstall/reinstall validation, and explicit rollback/prerelease publishing rules.
- **validation**: CPU installer builds on GitHub-hosted Windows, GPU artifact builds on CUDA runner, Linux egui artifact remains published if currently supported, packaged smoke commands pass, signing verification runs when signing is enabled, install/uninstall/reinstall over a previous version works, app data/config migration is verified, and legacy egui artifacts remain publishable if Tauri is blocked.
- **status**: Not Completed
- **log**:
- **files edited/created**:

#### T12: Documentation And Operator Handoff

- **depends_on**: [T2, T3]
- **owner**: documentation agent
- **location**: `README.md`, `docs/installation.md`, `docs/performance.md`, `docs/usage_guide.md`, `docs/tauri-migration-plan.md`
- **description**: Document Tauri dev setup, CPU/GPU builds, WebView2, CUDA, release lanes, troubleshooting, typed option migration, legacy egui deprecation, and rollback procedure.
- **validation**: Fresh-machine runbook review, commands match CI scripts, operator can choose correct artifact from docs alone.
- **status**: Not Completed
- **log**:
- **files edited/created**:

#### T14: Database Integration Fixtures And Parity Corpus

- **depends_on**: [T0.5]
- **owner**: QA/data agent
- **location**: `tests/*`, `docker/*`, `docs/matching_algorithms.md`, `src/bin/seed.rs`, future CI service-container config`
- **description**: Define a repeatable MySQL/MariaDB fixture strategy with schema, seed rows, expected outputs, GPU-sensitive fuzzy cases, cascade/advanced cases, and CI/local runner instructions.
- **validation**: service-container or local-runner fixture can seed data, CLI produces expected outputs, CPU/GPU/direct-prefilter parity cases are captured, and Tauri smoke tests can reuse the same fixture without embedding credentials.
- **status**: Not Completed
- **log**:
- **files edited/created**:

#### T13: Final Integration And Release Readiness Gate

- **depends_on**: [T11, T12, T14]
- **owner**: integrator / QA agent
- **location**: whole repo
- **description**: Merge swarm outputs, resolve conflicts, run full verification, compare CLI/egui/Tauri outputs on seeded datasets, and decide whether Tauri is stable, prerelease, or blocked while legacy egui remains primary.
- **validation**: full checklist below is green, CPU/GPU parity evidence attached, packaged app smoke proof captured, release artifacts named correctly, `git status --short` is clean before tagging.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### Parallel Execution Waves

| Wave | Tasks | Can Start When | Merge Rule |
|---|---|---|---|
| 1 | T0 | Immediately | Must finish before any code scaffold |
| 2 | T0.5 | T0 complete | Toolchain, runtime, and lockfile decisions become shared constraints |
| 3 | T1, T2, T14 | T0.5 complete | Cargo split, Tauri scaffold, and fixtures proceed independently |
| 4 | T3, T4 | T1/T2 complete as applicable | Backend service and frontend shell can proceed independently |
| 5 | T3.5 | T3 and T4 complete | Freeze DTO/job/result/event contracts before feature fan-out |
| 6 | T5, T6, T7, T12 | T3.5 complete; T14 ready where needed | Contract changes must land before UI uses them |
| 7 | T8, T9 | T5/T6/T7 complete | Run and results UI integrate against stable commands |
| 8 | T10 | T8/T9 complete | Accessibility/performance hardening edits finished feature screens |
| 9 | T11 | T8/T9/T10 complete | Release lanes package the proven app |
| 10 | T13 | T11/T12/T14 complete | Final integrator owns proof and release decision |

### Swarm Review Checklist

- Every task declares `depends_on`, owner, location, description, validation, status, log, and files edited/created.
- No task requires another agent to edit the same files at the same time without a dependency.
- Any task touching runtime behavior has an explicit parity or smoke validation.
- Any task touching secrets has a redaction validation.
- Any task touching Tauri commands has a capabilities validation.
- Any task touching frontend events has a listener cleanup validation.
- Any task touching results has a bounded-memory validation.
- Any release task covers existing workflows, legacy artifact policy, installer upgrade behavior, and rollback criteria.
- Any GPU packaging task explains resource discovery, bundle copy rules, artifact inspection, and clean-machine validation without assuming a developer CUDA Toolkit.
- Any database-dependent task uses the shared fixture corpus instead of ad hoc local data.
- Final readiness requires direct evidence, not proxy success only.

### Completion Gate For The Whole Migration

The migration is not complete until all of these are true:

- CLI still works for existing documented commands.
- Legacy egui still builds behind the `gui` feature.
- Tauri CPU app builds, launches, and completes command smoke tests.
- Tauri GPU app builds on a CUDA runner and passes CUDA diagnostics.
- GPU runtime resources are discovered, copied, inspected, and tested on a clean machine without requiring a developer CUDA Toolkit.
- Matching output parity is proven on seeded data for CPU, GPU prefilter, streaming, advanced, and cascade paths.
- The seeded database fixture has documented schema, data, expected outputs, and local/CI run instructions.
- Cancel, pause, resume, and shutdown behavior are tested with long-running jobs.
- Results pagination and export are backend-owned and memory-bounded.
- Tauri IPC serialization stays within tested payload limits for logs, progress, errors, and max result pages.
- Docs explain install, run, build, GPU prerequisites, WebView2, release lanes, and rollback.
- Release artifacts are named, signed when configured, inspected, install/uninstall/reinstall tested, and labeled prerelease or stable according to explicit decision criteria.
- Legacy egui Windows and existing Linux release support remain documented until intentionally retired.
- Final `git status --short` is clean before release tagging.

## Key Risks And Mitigations

| Risk | Mitigation |
|---|---|
| Tauri becomes a second matching path | Shared `RunService` contract before matching UI |
| Tauri permissions block commands | `capabilities/default.json` in Step 0 and command smoke tests |
| Long jobs block async runtime | Dedicated OS thread for matching work |
| Cancellation is too slow | Check cancel at DB page, batch, GPU flush, and export boundaries |
| Results exceed IPC limits | Run-scoped result store and paginated result commands |
| Frontend rerenders during progress/log floods | Split stores, 20Hz backend cap, frontend batching, log ring buffer |
| GPU controls drift from real engine flags | Typed DTO conversion tests and GPU parity gates |
| CUDA packaging missing DLLs/resources | Explicit resource list and packaged artifact inspection |
| WebView2 missing on target PC | Bootstrapper/fixed runtime policy and clean-profile smoke |
| Signing breaks release | Separate unsigned/test-signed/production-signed lanes |
| Tauri release fails late | Keep legacy egui artifacts and rollback publishing rule |

## Council Review Outcomes Applied

Backend architecture feedback applied:

- Added shared backend service contract.
- Added typed command boundary.
- Added stronger cancellation rules.
- Added explicit result store and pagination contract.
- Added GPU parity gate before exposing controls.

Frontend implementation feedback applied:

- Added frontend folder architecture.
- Moved minimal `EventBridge` into Step 3.
- Split Zustand stores by update frequency.
- Added listener cleanup and stale-job protection.
- Strengthened results table contract.
- Added long-running job states and frontend test gates.

Release/build feedback applied:

- Split Windows CPU/GPU and legacy egui release lanes.
- Added CUDA runner requirements.
- Added deterministic CUDA resource packaging.
- Added WebView2 gates.
- Added signing lanes and artifact naming.
- Added packaged app smoke tests and rollback rules.

Local UX pass applied:

- Kept the app operational and dense, not marketing-style.
- Added status rail, configuration summary, explicit disabled states, and grouped advanced settings.
- Moved accessibility requirements into early steps instead of leaving all accessibility to final polish.

Swarm planner review applied:

- Added toolchain/runtime baseline before scaffold work.
- Added backend contract freeze before database, generated types, jobs, and result-store work fan out.
- Fixed dependencies for database UI, results UI, and accessibility/performance hardening to avoid parallel file ownership conflicts.
- Added generated-binding pipeline ownership, CI stale-binding checks, IPC serialization limits, and Tauri config/capability validation.
- Added release workflow migration requirements for existing CI/release lanes, legacy Linux egui support, installer upgrade behavior, GPU resource reproducibility, and rollback/prerelease criteria.
- Added database fixture and parity corpus task so CLI, egui, and Tauri validation use the same seeded evidence.

## Final Recommendation

Proceed with Tauri, but do it as a shell-and-service migration rather than a UI rewrite that duplicates behavior. The safest path is:

1. Preserve the existing Rust engine.
2. Extract a shared run service.
3. Add Tauri typed commands around that service.
4. Build the React UI around job state, event streams, and paginated results.
5. Keep legacy egui artifacts until Tauri CPU and GPU releases pass real smoke tests.
