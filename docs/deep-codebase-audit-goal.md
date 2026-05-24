# Deep Codebase Audit Goal

## Goal Statement

Audit and harden the current Tauri v2 desktop application for `name_match_latest`, including its React/TypeScript frontend, Tauri Rust backend, shared Rust matching engine, Docker-backed MySQL testing workflow, and build/release paths, until each required path is either verified working with evidence or documented as blocked with root cause, impact, and next fix.

The audit is based on `docs/deep-codebase-audit-prompt.md` and must cover blockers, bottlenecks, slowdowns, parity issues, placeholders, incomplete or non-working features, compatibility gaps, missing tests, and missing operational requirements.

## Success Criteria

- Confirm the repository state, active branch, uncommitted files, and relevant recent Tauri changes before auditing.
- Inspect the main app surfaces:
  - Rust CLI and matching engine under `src/`
  - Run-service bridge under `src/run_service/`
  - Tauri backend under `src-tauri/`
  - React/TypeScript frontend under `ui/`
  - Windows build scripts under `scripts/windows/`
  - GitHub Actions workflows under `.github/workflows/`
  - Documentation under `README.md` and `docs/`
- Verify Docker testing context, especially the `matchers-mysql-1` container and its exposed MySQL port.
- Identify confirmed issues and classify each by severity:
  - P0: security, data loss, crash, or totally broken build/runtime path
  - P1: major feature broken, release blocker, serious correctness/performance issue
  - P2: important but not immediately blocking
  - P3: cleanup, documentation, maintainability
- For every confirmed issue, provide:
  - Symptom
  - Evidence
  - Affected files/components
  - Root cause
  - User-visible impact
  - Recommended fix
  - Compatibility/regression risk
  - Validation method
- Separate confirmed findings from suspected risks.
- Prefer small, targeted fixes over broad rewrites.
- Implement high-confidence minimal fixes when safe and in scope.
- Identify existing tests and add or recommend missing unit, integration, and E2E tests for confirmed blockers.
- Validate with concrete commands and evidence.
- Save a final audit report in Markdown.

## Execution Sequence

1. Create/refine this goal from `docs/deep-codebase-audit-prompt.md`.
2. Run subagent/party-mode review of the goal and incorporate only useful feedback.
3. Execute the audit using Context Engine first, then targeted file/command inspection.
4. Split implementation into safe subagent/skill workstreams when confirmed issues are independent.
5. Implement only minimal, high-confidence fixes for confirmed blockers.
6. Validate Tauri, Rust, frontend, Docker/MySQL, and report-generation paths.
7. Save final findings and evidence to `docs/deep-codebase-audit-report.md`.
8. Complete a final requirement-to-evidence checklist.

## Fix Authority And Non-Goals

Fixes are allowed only for confirmed issues that are small, local, reversible, and validated. Broad rewrites, dependency upgrades, database migrations, destructive Docker actions, release/deploy actions, matching algorithm rewrites, broad UI redesigns, and secret-bearing diagnostics require separate approval.

Do not print secrets. Do not make destructive Docker/database changes. Only clean up disposable smoke-test objects when the cleanup target is explicit and safe.

## Required Validation Gates

Run or explicitly justify skipping these checks:

- `git status --short --branch`
- Dependency/manifest inspection for Rust, Tauri, and frontend packages
- `pnpm run build` from `ui/`
- `cargo check` from `src-tauri/`
- Root Rust check/build if the local Cargo cache permits it
- `pnpm install --frozen-lockfile` from `ui/` or a documented reason for skipping
- `pnpm run lint` from `ui/`
- `pnpm run test` from `ui/`
- `pnpm run build` from `ui/`
- `cargo check --locked` from the repo root
- `cargo check --locked --features gui` from the repo root
- `cargo test --locked` or the narrowest relevant Rust tests available in the repo
- `cargo check --locked --manifest-path src-tauri/Cargo.toml`
- `cargo test --locked --manifest-path src-tauri/Cargo.toml`
- `cargo build --locked --bin name_matcher`
- `cargo build --locked --bin seed`
- Tauri build smoke, preferably `cargo tauri build --no-bundle` from `src-tauri`; if blocked, document the exact blocker
- Docker container inspection for `matchers-mysql-1`
- MySQL connectivity smoke against the exposed Docker port, if credentials can be derived safely from repo docs/config without exposing secrets
- Tauri build-script and CI workflow review
- Context Engine `review_auto` on any code changes made during the audit

A validation gate is not considered passed unless the command/action, working directory, exit result, and relevant evidence are recorded in the final report. Skipped gates must include reason, risk, and next command to run.

## Audit Focus Areas

### Tauri Runtime And Frontend

- Tauri command wiring, IPC boundaries, validation, and error mapping
- React state stores, command schemas, event handling, status/progress lifecycle, and export flows
- Tauri config paths, frontend build integration, icons/assets, and bundle readiness
- Produce an IPC contract matrix covering command name, DTO input/output, validation, error kind, frontend wrapper/schema, capability permission, and event side effects.
- Prove or block the minimum user-visible runtime flow: launch Tauri, connect to MySQL, list tables, list columns, run a small match, observe progress, view results, export CSV/XLSX, and validate at least one error path.

### Rust Engine And Run Service

- CLI/Tauri parity for algorithms, cascade/deep-match behavior, GPU flags, exports, cancellation, pause/resume, and progress events
- Database loading, table/column validation, result storage, paging, export correctness, and memory behavior
- CPU/GPU feature gating, CUDA detection, and fallback behavior
- Produce a matching parity table for CLI vs legacy egui vs Tauri covering Options 1-7, cascade/deep match, birthdate swap, GPU flags, exports, progress, cancellation, and result shape.
- Check session lifecycle, pool sizing, identifier safety, table/schema validation, row-count behavior, connection cleanup, credential non-retention, and whether full-table loads still violate intended streaming behavior.
- Check whether result storage can grow without bound, whether pagination/sorting/filtering can block IPC, and whether export reads backend store rather than frontend memory.

### Docker And Database Testing

- Confirm the `matchers-mysql-1` container exists and is reachable
- Identify test data expectations and safe smoke-test commands
- Verify database-related docs match actual code behavior
- Inspect `matchers-mysql-1` with `docker ps --filter "name=matchers-mysql-1"` and `docker inspect` for running state, exposed host port, and readiness without printing secrets.
- Document whether the container is repo-owned or external. If no repo-owned Docker/compose definition exists, record that dependency and the safest repeatable setup/smoke command.
- Run a disposable fixture smoke if credentials can be derived safely: build/use `seed`, create or verify small `smoke_*` tables, run at least one CLI or Tauri-backed matching path, and verify output/results. Cleanup only disposable `smoke_*` tables when explicitly safe.

### Build, CI, And Release

- Windows CPU/GPU build scripts
- Tauri MSI/NSIS bundle assumptions
- GitHub Actions path filters and artifact contents
- Rust toolchain, Node, pnpm, npm, Tauri CLI, and CUDA compatibility
- Compare PowerShell Tauri scripts against `.github/workflows/tauri-build.yml` for Node version, pnpm version, frozen lockfile usage, cargo `--locked`, feature flags, artifact paths, duplicate frontend builds, and CUDA DLL handling.
- Verify `src-tauri/tauri.conf.json` `beforeDevCommand` and `beforeBuildCommand` from the actual `cargo tauri dev/build` working directory.
- Separate Tauri EXE validation from MSI/NSIS installer validation. If only `--no-bundle` is run, mark installer release readiness as unverified.
- Verify `.github/workflows/release.yml` release assets against README/docs artifact promises, especially whether Tauri CPU/GPU artifacts are published.
- Verify `dist/gpu-dlls` existence and clean-machine GPU runtime expectations.
- Produce a compatibility matrix covering Rust toolchain, Tauri v2, WebView2, Node, pnpm, Windows MSVC, CUDA DLL packaging, and CPU/GPU script parity.

### Performance And Reliability

- Slow database reads, large result memory growth, long fuzzy/cascade runs, GPU/CPU oversubscription, blocking UI operations, and inefficient export/page paths
- Cancellation/pause responsiveness and progress throttling
- Error handling and retry behavior

## Deliverables

- `docs/deep-codebase-audit-goal.md`: this refined goal document
- `docs/deep-codebase-audit-report.md`: final audit findings, root causes, fixes/recommendations, validation evidence, and remaining risks
- A validation evidence ledger in the final report:

| Gate | Command / Action | Expected Result | Actual Result | Evidence Path / Snippet | Status |
|------|------------------|-----------------|---------------|--------------------------|--------|

- A skipped-gates table with reason, risk, and next command to run
- A final requirement-to-evidence checklist
- Code fixes only if they are minimal, high-confidence, and validated
- A final completion audit mapping this goal to actual evidence

## Done Definition

This goal is complete only when:

- Subagent review/party-mode feedback has been gathered and useful feedback has been incorporated into this goal.
- The codebase audit has been executed against the current Tauri-based repo state.
- Docker testing context has been checked, including `matchers-mysql-1`.
- Confirmed blockers and risks are documented with root causes and recommended fixes.
- Safe fixes, if any, have been implemented and reviewed.
- Validation gates have been run or explicitly justified as skipped.
- The final audit report exists and includes evidence, not just summaries.
- A completion checklist confirms every requirement in this goal is covered.
