# Deep Codebase Audit Prompt

## Objective

Perform a deep audit of the codebase to identify blockers, bottlenecks, slowdowns, parity issues, placeholders, incomplete or non-working features, compatibility problems, missing pieces, and hidden risks. Analyze each finding to determine the likely root cause, then recommend minimal, safe fixes with clear validation steps.

## Critical Context

Before making any changes, inspect the current codebase structure, configuration, dependencies, runtime paths, and existing implementation patterns. Identify the specific files, components, routes, services, scripts, database objects, environment variables, or build/deployment settings affected by each issue.

Separate findings into clear categories:

- Blockers preventing the app or feature from working
- Performance bottlenecks or slow user flows
- Local vs staging/production parity issues
- Placeholder, mock, stub, or incomplete code
- Broken or partially wired frontend/backend features
- Compatibility issues with dependencies, runtime versions, APIs, browsers, or deployment targets
- Missing tests, scripts, configuration, documentation, or operational requirements

## Assumptions

Assume the goal is to improve reliability and readiness without large rewrites unless the evidence shows a rewrite is necessary. Prefer existing project patterns and small targeted fixes over introducing new architecture.

If any required information is missing, state the assumption clearly and continue with the safest evidence-backed path.

## Constraints

Do not make broad or risky changes before identifying the affected files/components and explaining the intended change. Implement minimal, safe changes first. Avoid unrelated refactors, formatting churn, dependency upgrades, or behavior changes unless required to fix a confirmed issue.

Do not expose secrets, tokens, private keys, `.env` contents, or sensitive configuration values in the report.

## Proposed Plan

1. Inspect the repository structure, dependency manifests, configuration files, app entry points, routes, services, database/migration files, frontend wiring, and test/build scripts.
2. Identify affected files/components for each suspected issue before proposing or making changes.
3. Audit for blockers, bottlenecks, parity mismatches, placeholders, incomplete features, broken integrations, compatibility issues, and missing operational pieces.
4. For each confirmed issue, document:
   - Symptom
   - Evidence
   - Affected files/components
   - Root cause
   - Impact
   - Recommended fix
   - Risk level
   - Validation method
5. Prioritize fixes by severity and blast radius.
6. Implement only the smallest safe changes needed for high-confidence fixes, unless instructed to keep this audit read-only.
7. Validate with concrete checks such as tests, lint, typecheck, build, runtime smoke tests, API checks, browser checks, or relevant manual verification.
8. Report what changed, what was validated, remaining risks, and recommended follow-up actions.

## Test Requirement

Tests are required where practical. Identify existing test coverage, run relevant tests, add or recommend missing unit, integration, and E2E tests for confirmed blockers, and clearly state any untested risk before completion.

When fixes are implemented, validation should include the narrowest relevant tests first, then broader build/runtime checks where the blast radius justifies it.

## Validation Checklist

Validate using the strongest available project-specific checks:

- Dependency install or lockfile integrity check, if needed
- Lint/static analysis
- Typecheck
- Unit tests
- Integration/API tests
- Build command
- Runtime smoke test
- Browser/UI verification for user-facing flows
- Environment/config parity check
- Database/migration consistency check, if applicable
- Performance-sensitive path review or benchmark, where practical

If any check cannot be run, state exactly why and what should be run next.

## Risks and Mitigations

Watch for:

- False positives from unused or legacy code
- Local-only behavior that differs from staging/production
- Dependency upgrades causing regressions
- Fixes that solve symptoms but not root causes
- Hidden coupling between frontend, backend, database, and deployment configuration
- Missing environment variables or external service assumptions

Mitigate by grounding every finding in file evidence, preferring small changes, validating after each meaningful fix, and clearly separating confirmed issues from suspected risks.

## Open Questions

List any unresolved questions that affect confidence, such as:

- Which environment is the source of truth: local, staging, or production?
- Are there known broken flows or priority features to audit first?
- Are dependency upgrades allowed, or should fixes stay within current versions?
- Should this audit be read-only, or should confirmed fixes be implemented?
- Which validation commands are considered required before completion?

## Done Definition

The task is complete when:

- The codebase has been audited for blockers, bottlenecks, slowdowns, parity issues, placeholders, non-working features, compatibility problems, and missing pieces.
- Each confirmed issue includes evidence, affected files/components, root cause, impact, and recommended fix.
- Existing test coverage has been identified, relevant tests have been run, and missing tests for confirmed blockers have been added or recommended.
- Minimal safe fixes have been implemented where appropriate and allowed.
- Concrete validation checks have been run and reported.
- Remaining risks, skipped checks, and follow-up actions are clearly documented.
