# Deep Codebase Audit Report

Date: 2026-05-24

## Executive Summary

The audit prompt and execution goal were saved, reviewed in party mode with four specialist subagents, refined, and then executed against the current Tauri-first codebase.

Three concrete blockers were found and fixed:

1. Tauri v2 build hooks pointed at `../ui`, but local Tauri CLI execution runs hooks from the repository root. This made the Tauri build look for `D:\GitProjects\ui` and fail. Fixed in `src-tauri/tauri.conf.json`.
2. The frontend lint command used `tsc -b --noEmit`, which conflicts with project references that disable emit. Fixed in `ui/package.json`.
3. `pnpm run test` failed because Vitest had no test files. Added focused tests for shared format helpers in `ui/src/shared/lib/format.test.ts`.

The Tauri no-bundle Windows executable build now succeeds and produced:

```text
D:\GitProjects\name_match_latest\src-tauri\target\release\name-matcher-tauri.exe
```

The remaining full-green blocker is not a code defect found in this pass: root Rust cargo gates currently fail on this machine because Cargo cannot unpack/read `icu_locale_core v2.0.0` from the local registry cache with `Access is denied. (os error 5)`. The `src-tauri` Rust gates pass, but root CLI/library gates and seed binary rebuilds cannot be considered verified until the Cargo registry permission/cache issue is fixed and rerun.

## Saved Artifacts

- `docs/deep-codebase-audit-prompt.md`: enhanced Context Engine audit prompt.
- `docs/deep-codebase-audit-goal.md`: execution goal refined after party-mode review.
- `docs/deep-codebase-audit-report.md`: this evidence report.

## Party-Mode Review Incorporated

Four specialist reviews were requested and incorporated into the goal before execution:

- Project shipping review: require Tauri-first objective, fixed execution sequence, no silent pass, and explicit fix authority.
- API/testing review: require evidence ledger, exact validation gates, Docker MySQL smoke proof, and skipped-gate explanations.
- Backend architecture review: require IPC matrix, DB/session lifecycle checks, matching parity analysis, result/export coverage, and root-cause format.
- DevOps review: require `--locked` cargo gates, Tauri hook cwd verification, script/CI parity, EXE versus installer separation, release workflow parity, GPU DLL checks, and external Docker dependency proof.

## Validation Evidence Ledger

| Area | Command / Check | Result |
| --- | --- | --- |
| Git state | `git status --short --branch` | On `main...origin/main`; local modifications and new audit/test files are uncommitted. |
| Frontend deps | `pnpm install --frozen-lockfile` in `ui/` | Passed. Warning: build scripts such as `esbuild` were ignored by pnpm approval policy. |
| Frontend lint before fix | `pnpm run lint` in `ui/` | Failed with `TS6310` because referenced project `tsconfig.node.json` may not disable emit under `tsc -b --noEmit`. |
| Frontend lint after fix | `pnpm run lint` in `ui/` | Passed after changing lint to separate `tsc -p ... --noEmit` checks. |
| Frontend tests before fix | `pnpm run test` in `ui/` | Failed because Vitest found no test files. |
| Frontend tests after fix | `pnpm run test` in `ui/` | Passed: `src/shared/lib/format.test.ts`, 3 tests. |
| Frontend build | `pnpm run build` in `ui/` | Passed; Vite generated `ui/dist`. |
| Tauri Rust check | `cargo check --locked --manifest-path src-tauri\Cargo.toml` | Passed with warnings from shared/root code. |
| Tauri Rust tests | `cargo test --locked --manifest-path src-tauri\Cargo.toml` | Passed as a compile-level gate; no Tauri tests were defined. |
| Tauri CLI availability | `pnpm tauri --version` in `ui/` | Passed: local Tauri CLI is available through pnpm. |
| Tauri build before fix | local Tauri CLI from `src-tauri/` | Failed because `beforeBuildCommand` resolved `../ui` from repo root. |
| Tauri build after fix | `..\ui\node_modules\.bin\tauri.cmd build --no-bundle` from `src-tauri/` | Passed and built `src-tauri\target\release\name-matcher-tauri.exe`. |
| Root Rust gates | `cargo check --locked`, `cargo check --locked --features gui`, `cargo test --locked --lib`, root binary builds | Blocked by Cargo registry/cache permission error on `icu_locale_core v2.0.0`. |
| Docker container | `docker ps` / `docker inspect matchers-mysql-1` | `matchers-mysql-1` is running on host port `3307`; no healthcheck is configured. |
| Repo Docker fixtures | search for compose/Dockerfile/SQL fixtures | No repo-owned Docker compose, Dockerfile, or SQL fixture files found. The live MySQL container is an external test dependency. |
| Docker smoke data | disposable MySQL tables inside `matchers-mysql-1` | Created `smoke_a` and `smoke_b`; verified 3 rows in each. |
| Existing compiled Rust tests | previously built `target\debug\deps\name_matcher-*.exe` test harness | 26 tests passed, 0 failed, 18 filtered out. This is stale-binary evidence, not a fresh root cargo gate. |
| GPU runtime payload | `dist/gpu-dlls` inspection | CUDA runtime DLLs are present locally: `nvrtc64_*` and `nvrtc-builtins64_*`. |
| Release workflow parity | `.github/workflows/release.yml` | Publishes legacy `gui` CPU assets only, not Tauri MSI/NSIS or Tauri CPU/GPU artifacts. |
| Tauri workflow parity | `.github/workflows/tauri-build.yml` | Builds no-bundle Tauri EXE artifacts for CPU and GPU paths; does not publish MSI/NSIS installers. |
| Context Engine review | deterministic `review_diff` with LLM disabled | CI gate passed. It flagged `PRE001` because the supplied two-file diff omitted the untracked test file; this is mitigated by `ui/src/shared/lib/format.test.ts`. |

## Root Cause Findings

### Fixed: Tauri Hook Path

Root cause: `src-tauri/tauri.conf.json` used `pnpm --dir ../ui ...` for Tauri build hooks. With the local Tauri v2 CLI, the hook ran from the repository root, so `../ui` resolved outside the repo.

Fix:

```json
"beforeDevCommand": "pnpm --dir ui dev",
"beforeBuildCommand": "pnpm --dir ui build"
```

Compatibility impact: this aligns local validation with the documented Tauri v2 hook behavior in `docs/tauri-migration-plan.md`. CI should use the same path behavior when running from `src-tauri`.

### Fixed: Frontend Lint Script

Root cause: TypeScript project references are incompatible with `tsc -b --noEmit` when a referenced project disables emit.

Fix:

```json
"lint": "tsc -p tsconfig.json --noEmit && tsc -p tsconfig.node.json --noEmit"
```

Compatibility impact: lint now validates both app and Node config files without forcing build-mode emit behavior.

### Fixed: Empty Vitest Gate

Root cause: the package declared `vitest run`, but no test files existed, so the test gate failed before checking behavior.

Fix: added `ui/src/shared/lib/format.test.ts` with coverage for invalid/missing numbers, durations, bytes, and percentages.

Compatibility impact: the test gate now has a stable smoke target and passed locally.

### Blocked: Root Cargo Registry Permission

Root cause: root cargo gates fail while trying to unpack/read `icu_locale_core v2.0.0` in the local Cargo registry cache. The error is environmental and repeatable:

```text
Access is denied. (os error 5)
```

Impact: root library checks, GUI feature checks, root tests, and fresh seed/name_matcher binary builds are not fully verified in this run.

Recommended fix:

1. Stop any process holding files under the affected Cargo registry cache.
2. Clear or repair the specific `icu_locale_core-2.0.0` registry cache directory with normal user permissions.
3. Rerun:

```powershell
cargo check --locked
cargo check --locked --features gui
cargo test --locked --lib
cargo build --locked --bin seed
cargo build --locked --bin name_matcher
```

### Gap: Release Asset Parity

Root cause: `release.yml` still builds and uploads legacy `gui` assets only. The Tauri workflow builds no-bundle EXEs, but release publishing does not yet include Tauri CPU/GPU assets or MSI/NSIS installers.

Impact: documentation/configuration says Tauri bundle targets are `msi` and `nsis`, but release automation does not prove or publish those installers.

Recommended fix: either update release automation to publish Tauri CPU/GPU assets and installer bundles, or update release docs to state that current published release assets are legacy GUI-only until Tauri packaging is wired into release publishing.

### Gap: Docker Test Fixture Ownership

Root cause: the repo does not currently include Docker compose or SQL fixtures for the MySQL smoke path. The `matchers-mysql-1` container works as a live external dependency, but the setup is not reproducible from the repo alone.

Impact: another machine cannot recreate the exact smoke environment without out-of-band setup.

Recommended fix: add a minimal compose file or documented smoke fixture script that creates disposable test tables without exposing credentials.

## IPC And Feature Surface Snapshot

The Tauri command surface registered in `src-tauri/src/main.rs` includes:

- Database/session: `connect_db`, `test_connection`, `disconnect_db`, `list_sessions`, `list_tables`, `get_table_columns`, `get_row_count`.
- Matching lifecycle: `start_matching`, `cancel_matching`, `pause_matching`, `resume_matching`, `get_matching_status`, `list_matching_jobs`.
- Results/export: `get_results_page`, `export_results`.
- System/support: `cuda_diagnostics`, `system_snapshot`, `choose_directory`.

Static audit focus:

- Database access validates required fields and avoids retaining the raw password in the stored session descriptor.
- Table-sensitive commands use backend identifier validation before row-count and matching operations.
- Matching start validates session, table names, output directory, selected columns, and strategy config before dispatch.
- Results and export are backend-mediated through result store/session state rather than frontend-only placeholders.

Runtime caveat: a full interactive Tauri UI flow was not browser-automated in this run. The executable build passed, but end-to-end user-click validation should be run once the root Cargo cache blocker is cleared and a reproducible database fixture exists.

## Matching Parity Snapshot

The audit goal requires parity across CLI, legacy GUI, and Tauri. This run verified the Tauri build path and static command surface, but did not freshly rebuild root CLI/GUI binaries because of the Cargo registry permission blocker.

Current parity status:

| Surface | Status |
| --- | --- |
| Tauri frontend/build | Verified by successful no-bundle EXE build. |
| Tauri backend commands | Static surface present; compile-level Tauri check passed. |
| Root CLI binary | Not freshly verified; root cargo build blocked by Cargo registry permission. |
| Legacy GUI binary | Not freshly verified; root `--features gui` check blocked by Cargo registry permission. |
| Docker-backed matching smoke | Smoke data created, but fresh binary execution blocked by root cargo issue. |

## Compatibility Matrix

| Component | Observed |
| --- | --- |
| Rust toolchain | `rust-toolchain.toml` pins `1.89.0` with `rustfmt` and `clippy`. |
| Node | `ui/package.json` requires `>=20`; Tauri CI uses Node `20`. |
| pnpm | Tauri CI uses pnpm `10`; local lockfile install passed. |
| Tauri CLI | Local pnpm CLI reported `tauri-cli 2.11.2`; package range is `^2.1.0`. |
| Tauri bundle config | MSI and NSIS configured, but only no-bundle EXE was validated. |
| GPU payload | Local `dist/gpu-dlls` contains required CUDA runtime DLL patterns used by CI. |
| MySQL smoke | External live container `matchers-mysql-1`, host port `3307`, no healthcheck. |

## Completion Checklist

- [x] Enhanced prompt saved.
- [x] Execution goal saved.
- [x] Party-mode review completed and incorporated.
- [x] Context Engine used for codebase context and deterministic review.
- [x] Tauri hook blocker fixed.
- [x] Frontend lint blocker fixed.
- [x] Frontend test gate blocker fixed.
- [x] Frontend install, lint, test, and build validated.
- [x] Tauri Rust check/test gates validated.
- [x] Tauri no-bundle EXE build validated.
- [x] Docker container status and disposable smoke data validated.
- [x] Release parity gap identified.
- [x] Root Cargo gate blocker identified with root cause and rerun plan.
- [x] Final audit report saved.

## Recommended Next Fixes

1. Repair the local Cargo registry permission/cache issue, then rerun all root cargo gates and seed/name_matcher binary builds.
2. Add repo-owned Docker MySQL smoke setup so `matchers-mysql-1` is reproducible instead of an external dependency.
3. Decide release direction: publish Tauri CPU/GPU/installer artifacts from release workflow, or document legacy GUI-only release status.
4. Add a small Tauri command integration test around the MySQL smoke fixture after the root cargo gate is unblocked.
5. Run a clean-machine GPU artifact validation on the self-hosted CUDA runner to prove the bundled DLL set outside this workstation.
