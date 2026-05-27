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

## Scale ladder and 250k pre-merge gate

Datasets from `Generate-Datasets.ps1` (million-row scale plan):

| Dataset | Rows/side | Gate |
|---------|-----------|------|
| `medium_csv_100k` | 100k | CI smoke |
| `gate_csv_250k` | 250k | **Required local pre-merge** |
| `large_csv_1m` | 1M | Manual only (`-SkipLarge` omits by default) |

Pre-merge workflow:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\perf\Generate-Datasets.ps1
powershell -ExecutionPolicy Bypass -File scripts\perf\Run-Benchmarks.ps1 `
  -DatasetNames gate_csv_250k `
  -CommandTemplate 'your-command-here'
powershell -ExecutionPolicy Bypass -File scripts\perf\Compare-Benchmarks.ps1 `
  -Before tmp\perf\results\baseline.json `
  -After tmp\perf\results\after.json
```

Pass when `peak_rss_mb` stays under **1.5 GB** or within **+20%** of the 100k baseline (whichever is stricter). See `docs/million-row-scale-plan.md` and `docs/mysql-benchmark-fixture.md` (operator runbook).

## RSS fields in benchmark JSON

Each dataset entry in `Run-Benchmarks.ps1` output includes:

- `peak_rss_mb` — max of PowerShell host working set before/after warm runs (MB)
- Per-run `elapsed_ms`, exit code, stdout/stderr snippets
- `summary.warm_p50_elapsed_ms`, `warm_p95_elapsed_ms`, `warm_p99_elapsed_ms`

`Compare-Benchmarks.ps1` enforces p95 regression by default; extend comparisons manually for RSS if you store multiple runs.

## MySQL config capture (scale evidence)

Before import/match gate runs against Docker MySQL, snapshot server settings:

```powershell
$env:MYSQL_IMPORT_TEST_URL = "<mysql-url-for-local-test-database>"
powershell -ExecutionPolicy Bypass -File scripts\perf\Capture-MysqlConfig.ps1 `
  -Url $env:MYSQL_IMPORT_TEST_URL `
  -OutputPath tmp\perf\mysql-config.json
```

The script records `VERSION()`, `innodb_buffer_pool_size`, `local_infile`, and related variables. Attach `mysql-config.json` next to benchmark JSON for audit trails.

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
