# MySQL benchmark fixture (Docker `matchers`)

Use the local Docker container **`matchers`** for import/match scale gates and integration tests.

## Connection

| Setting | Repo `docker-compose` | Common local `matchers` container |
|---------|----------------------|-----------------------------------|
| Service / name | `matchers-mysql` | `matchers-mysql-1` |
| Host port | `3307` → container `3306` | `3307` |
| Database | `duplicate_checker` | often `matcher` |
| Password | configured locally | inspect local container/env |
| Test URL env | `MYSQL_IMPORT_TEST_URL` | same |

Repo compose example:

```text
<mysql-url-for-repo-compose>
```

Local `matchers` container example (inspect with `docker inspect matchers-mysql-1`):

```text
<mysql-url-for-local-matchers-container>
```

Set before Rust integration tests (database name is inferred from the URL path):

```powershell
$env:MYSQL_IMPORT_TEST_URL = "<mysql-url-for-local-test-database>"
$env:CARGO_HOME = "D:\GitProjects\name_match_latest\.cargo-test-home"
cargo test --test csv_import_mysql -- --ignored
```

## Recommended server settings for scale work

Document in benchmark JSON evidence:

- MySQL version (`SELECT VERSION()`)
- `innodb_buffer_pool_size`
- `local_infile` (for optional `LOAD DATA LOCAL INFILE` v1.1)
- Indexes on destination tables before duplicate probes

## Scale ladder (from million-row plan)

| Profile | Rows | Gate |
|---------|------|------|
| CI smoke | 100k | required in CI |
| Pre-merge | 250k | local gate (`gate_csv_250k` dataset) |
| Manual | 1M | `#[ignore]` integration |
| Exploratory | 5M | operator-only |

## Evidence artifacts

Each gate run must attach:

- `scripts/perf` JSON output (`peak_rss_mb`, p50/p95/p99)
- Dataset manifest SHA-256 from `tmp/perf/datasets/*/manifest.json`
- Git commit + command log
- Comparison markdown from `Compare-Benchmarks.ps1`

## Staging import verification

Replace mode must leave the original table intact when cancel happens **before** the `RENAME TABLE` swap. Use dry-run + commit cancel tests in `tests/csv_import_mysql.rs`.

---

## Operator runbook (scale gates)

### RAM expectations (16–32 GB workstation)

| Phase | Typical peak process RSS | Notes |
|-------|--------------------------|-------|
| CSV import (staging) | &lt; 1.5 GB at 250k; scale vs 100k baseline (+20% max) | Streamed batches; no full `Vec<Person>` in plan cache |
| DB match (streaming) | Bounded vs full in-memory load | `RunService` uses partitioned DB load + spill above `RESULT_SPILL_ROWS` (100k) |
| Results review | SQLite-backed paging | Browser should not fetch all rows; use Export for large jobs |

Record `peak_rss_mb` from `scripts/perf/Run-Benchmarks.ps1` JSON for every gate run.

### Import time vs match time

At **1M × 1M**, **import usually finishes before match**. Set expectations separately in reports:

- Import gate: staging load + swap/merge + indexes (`gate_csv_250k` or `large_csv_1m` datasets).
- Match gate: partitioned streaming on DB tables; wall time often dominates.

**Import complete ≠ match-ready:** wait for indexes/`ANALYZE` and confirm row counts on both sides before starting a run.

### Partial append (non-atomic)

**Append** mode merges staging in chunks. If the job is **cancelled mid-commit**, some rows may already be inserted (`partial_commit: true` on the job DTO). There is no full rollback in v1 — inspect `inserted_rows` / `processed_rows` and clean up manually if needed.

**Replace** is safer: cancel **before** the `RENAME TABLE` swap leaves the live table unchanged (see integration test).

### Scale gates (evidence required)

| Profile | Dataset | Command / test | Pass criteria |
|---------|---------|----------------|---------------|
| CI smoke | `medium_csv_100k` | perf harness or import integration | Manifest SHA-256 + JSON evidence |
| Pre-merge | `gate_csv_250k` | `Generate-Datasets.ps1` + local import/match smoke | RSS within gate; no OOM |
| Manual | `large_csv_1m` | `cargo test --test csv_import_mysql -- --ignored` | Operator-only; attach MySQL config snapshot |
| Exploratory | 5M | operator-only | Same evidence schema |

Each gate must attach: git commit, exact command, dataset manifest SHA-256, `peak_rss_mb`, p50/p95/p99, MySQL version/config (see below), exit code, raw logs, comparison markdown from `Compare-Benchmarks.ps1`.

### Environment variables

**MySQL pool** (see [performance.md](performance.md)):

- `NAME_MATCHER_POOL_SIZE`, `NAME_MATCHER_POOL_MIN`, `NAME_MATCHER_ACQUIRE_MS`, `NAME_MATCHER_IDLE_MS`, `NAME_MATCHER_LIFETIME_MS`

**Import integration tests:**

```powershell
$env:MYSQL_IMPORT_TEST_URL = "mysql://root:root@127.0.0.1:3307/duplicate_checker"
$env:CARGO_HOME = "D:\GitProjects\name_match_latest\.cargo-test-home"
cargo test --test csv_import_mysql -- --ignored
cargo test --test db_cross_session_smoke -- --ignored
```

**Cross-session dual-pool streaming smoke (`tests/db_cross_session_smoke.rs`):**

- Creates disposable `smoke_cross_a` / `smoke_cross_b` tables with paired rows.
- Compares `stream_match_csv_dual` pair set to in-memory `match_all_with_opts` baseline.
- Default `MYSQL_SMOKE_ROWS=1000`; set `10000` for the full G1 gate.
- Pure Rust policy test `cross_session_policy_selects_two_pool_not_single_pool` runs without MySQL.

**Match / perf (optional tuning):**

- `NAME_MATCHER_STREAMING=1` — legacy CLI streaming hint
- `NAME_MATCHER_AUTO_OPTIMIZE=1`, `RAYON_NUM_THREADS=12`
- `NAME_MATCHER_USE_GPU=1` and related GPU vars for fuzzy workloads (not million-row DB streaming v1)

**Capture MySQL server settings** before a gate run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\perf\Capture-MysqlConfig.ps1 `
  -Url $env:MYSQL_IMPORT_TEST_URL `
  -OutputPath tmp\perf\mysql-config.json
```

Attach `tmp/perf/mysql-config.json` to benchmark evidence (version, `innodb_buffer_pool_size`, `local_infile`, key indexes).
