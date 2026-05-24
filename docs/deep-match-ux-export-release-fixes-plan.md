# Plan: Deep Match UX, Export, And Release Readiness Fixes

## Summary

Implement the user-facing Deep Match fixes first, then harden export behavior and validation. The reviewed subagent feedback is incorporated: keep Quick Match behavior stable, make Deep Match controls truthful, restore cascade-aware exports from `ResultStore`, and treat release/GPU packaging as a separate hardening lane.

## Key Changes

- **Deep Match configuration clarity**
  - Move `Allow birthdate month/day swap` into a shared Match Options section visible in both Quick Match and Deep Match.
  - Add it to the configuration summary as `Birthdate swap: On/Off`.
  - In Deep Match summary, show `Deep Match - N levels` instead of stale Quick Match `Option N` wording.
  - Add layman text for exclusion mode:
    - `Exclusive`: recommended, faster, matched pairs skip later levels.
    - `Independent`: slower, every level checks the full dataset and duplicates may appear.
  - Show a visible warning when `Independent` is selected.

- **GPU panel mode awareness**
  - Keep the existing GPU DTO flags unchanged.
  - In Quick Match, keep option-specific GPU wording where useful.
  - In Deep Match, replace Quick Match wording like `Options 1-2` and `Option 7` with cascade wording:
    - GPU applies mainly to fuzzy cascade levels `L10/L11`.
    - Exact levels `L1-L9` remain CPU-style matching.
  - Keep GPU default as `CPU` for now. Do not switch default to `Auto` until Tauri CUDA fallback behavior is tested end to end.

- **Cascade-aware results and export**
  - Add optional `match_method` beside existing `matched_at_level`.
    - Rust DTO: `match_method: Option<String>` with serde default.
    - TS DTO: `match_method?: string | null`.
  - Populate `matched_at_level` and `match_method` when flattening cascade results, using labels like `L10 - Fuzzy Birthdate Full Middle`.
  - Results table shows level/method for Deep Match rows and hides or shows dash for Quick Match rows.
  - Quick Match CSV/XLSX export stays unchanged.
  - Deep Match CSV export writes:
    - combined file: `matches.csv`
    - per-level files for levels with exported rows: `matches_L01.csv`, `matches_L10.csv`, etc.
  - Deep Match XLSX export writes one workbook:
    - `Matches` sheet for combined rows
    - one sheet per exported level, such as `L01`, `L10`, `L11`
    - summary sheet remains compatible.
  - `min_confidence` filtering applies before grouping, so per-level files/sheets match the exported combined set.

- **Release and environment hardening**
  - Update release planning so Tauri CPU/GPU artifacts are published, not only legacy GUI assets.
  - Add real MSI/NSIS bundle verification after the current no-bundle EXE path.
  - Add repo-owned Docker MySQL fixture setup for reproducible smoke tests.
  - Keep root Rust warning cleanup as non-blocking unless warnings hide real CI failures.

## Public Interfaces / Types

- Extend `MatchPairDto` only:
  - Add optional `match_method`.
  - Keep `matched_at_level` as the numeric cascade level.
- No new Tauri command is required.
- `export_results` keeps the same request/response shape.
- `written_paths` will include multiple paths for Deep Match CSV/Both exports.

## Test Plan

- **Frontend**
  - Deep Match shows birthdate swap control and Independent warning.
  - Quick Match still shows Algorithm Options 1-7.
  - Deep Match GPU panel has no Quick Match option wording.
  - Configuration summary changes by mode and preserves export, GPU, streaming, and selected Quick Match algorithm state.

- **Backend/export**
  - Quick Match CSV/XLSX headers remain unchanged.
  - Deep Match CSV/XLSX include `matched_at_level` and `match_method`.
  - Deep Match per-level exports contain only that level's rows.
  - `min_confidence` filters combined and per-level exports consistently.
  - Independent mode preserves duplicate pair rows when they match multiple levels.

- **Validation gates**
  - `pnpm install --frozen-lockfile`
  - `pnpm run lint`
  - `pnpm run test`
  - `pnpm run build`
  - `cargo fmt --check`
  - `cargo check --locked --manifest-path src-tauri/Cargo.toml`
  - `cargo test --locked --manifest-path src-tauri/Cargo.toml`
  - Root cargo gates when local Cargo registry permissions are healthy.
  - GPU build smoke with `scripts/windows/Build-Tauri-Gpu.ps1`.
  - Docker fixture smoke once compose/fixtures are added.

## Assumptions

- Quick Match export compatibility is more important than adding new columns globally.
- Deep Match per-level CSV files are created only for levels that have exported rows after filtering.
- XLSX uses per-level sheets instead of many per-level `.xlsx` files.
- GPU default stays CPU until CUDA fallback is proven from the Tauri app path.
- Release workflow, installer verification, Docker fixture, and warning cleanup can be implemented after the core UI/export repair if we want a smaller first patch.
