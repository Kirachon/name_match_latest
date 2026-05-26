# Name Matcher Performance Harness

This folder contains the baseline evidence harness required before performance
optimizations are made.

## Generate Datasets

```powershell
powershell -ExecutionPolicy Bypass -File scripts\perf\Generate-Datasets.ps1
```

Output goes to `tmp/perf/datasets` by default. Each generated dataset includes:

- deterministic seed
- source and target CSV files
- SHA-256 hashes
- manifest JSON
- expected row counts and collision profile

## Run Benchmarks

The runner is command-template based so it can wrap the CLI, Tauri smoke
commands, or later benchmark binaries without changing the evidence format.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\perf\Run-Benchmarks.ps1 `
  -CommandTemplate 'target\release\name_matcher --source "{source}" --target "{target}"' `
  -BuildLabel cpu-release `
  -RepeatCount 5
```

Placeholders available in `-CommandTemplate`:

```text
{source}
{target}
{dataset}
{output}
```

The runner records one cold run and N warm runs per dataset. JSON output
includes command, exit code, elapsed time, commit hash, branch, build label,
environment metadata, input hashes, and raw stdout/stderr snippets.

## Compare Runs

```powershell
powershell -ExecutionPolicy Bypass -File scripts\perf\Compare-Benchmarks.ps1 `
  -Before tmp\perf\results\baseline.json `
  -After tmp\perf\results\after.json
```

Default regression gates:

```text
p95 elapsed time must not regress by more than 5%
peak memory must not regress by more than 10% when available
L10/L11 candidate counts must not regress by more than 20% when available
```

GPU `GateOnly` promotion still requires the stricter gates in
`docs/name_matcher_performance_remediation_plan.md`.

## Compare Recall

Use this before accepting L10/L11 blocking changes or GPU `GateOnly` promotion.
Inputs can be a JSON array of match rows or an object with a `rows` array.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\perf\Compare-Recall.ps1 `
  -Before tmp\perf\results\baseline-rows.json `
  -After tmp\perf\results\candidate-rows.json `
  -AllowedFalseNegatives 0
```

The comparison keys rows by `source_id,target_id` and checks confidence,
matched level, matched fields, and optionally row order.
