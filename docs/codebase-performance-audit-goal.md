# Codebase Performance Audit Goal

## Objective

Audit the `D:\GitProjects\name_match_latest` codebase for blockers, performance bottlenecks, OOM risks, slow execution paths, and reliability issues. Identify evidence-backed root causes, propose or implement minimal safe fixes for confirmed high-impact issues, validate with relevant checks, and report changed files, validation results, remaining risks, and follow-up recommendations.

## Scope

Inspect the current codebase with emphasis on:

- Runtime entry points and critical user workflows
- Matching and batch-processing paths
- Database loading, streaming, paging, and export paths
- File import/export behavior
- Memory retention, buffering, spill behavior, and large-result handling
- Concurrency, cancellation, pause/resume, and progress reporting
- Frontend operations that can block, freeze, or fetch too much data
- Build, test, and runtime blockers

## Evidence Requirements

Do not assume the cause of a slowdown, OOM, or blocker before collecting evidence from:

- Code inspection
- Tests
- Logs
- Runtime behavior
- Profiling or targeted measurement where feasible
- Existing documentation or prior audit notes

For every confirmed issue, capture:

- Symptom
- Evidence
- Affected files or components
- Root cause
- User-visible impact
- Recommended fix
- Compatibility or regression risk
- Validation method

Separate confirmed findings from suspected risks.

## Constraints

- Keep fixes minimal, targeted, and reversible.
- Prefer small, low-risk fixes before larger architectural changes.
- Do not perform broad rewrites, dependency upgrades, formatting sweeps, or unrelated cleanup unless required to fix a confirmed issue.
- Preserve existing behavior unless the behavior is clearly part of the bug.
- Do not run destructive commands, migrations, deploys, mass edits, or cleanup without explicit confirmation.
- Do not print secrets or sensitive connection details.

## Proposed Execution Plan

1. Confirm repository state, active branch, and uncommitted files.
2. Inspect project manifests, runtime entry points, tests, and existing audit/performance docs.
3. Use Context Engine retrieval first to locate relevant code paths.
4. Identify likely blockers, bottlenecks, OOM risks, and slow paths.
5. Verify suspected issues with targeted evidence.
6. Prioritize confirmed root causes by severity and user impact.
7. Implement only minimal safe fixes for high-confidence, high-impact confirmed issues.
8. Run relevant validation checks such as tests, typecheck, lint, build, smoke tests, or memory/performance comparisons.
9. Produce a final report with findings, fixes, validation evidence, remaining risks, and follow-up recommendations.

## Validation Checklist

Run or explicitly justify skipping relevant checks:

- Git status and branch inspection
- Rust `cargo check`, `cargo test`, or targeted Rust tests
- Frontend typecheck, lint, test, or build commands
- Tauri/backend checks where relevant
- Runtime smoke tests for affected workflows
- Memory or performance comparison where feasible
- Context Engine review on any code changes made during the audit

If a validation step cannot be run, document why, the residual risk, and the next command to run.

## Done Definition

This goal is complete when:

- Blockers, bottlenecks, OOM risks, slow paths, and reliability concerns have been investigated with evidence.
- Confirmed root causes are documented.
- Minimal safe fixes are implemented for confirmed high-impact issues where appropriate.
- Relevant validation has been run or explicitly documented as skipped.
- A final audit report lists changed files, validation results, remaining risks, and follow-up recommendations.
