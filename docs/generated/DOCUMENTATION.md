# Name Matcher Technical Documentation

## Objective and Scope

This document describes the `name_matcher` application in `D:\GitProjects\name_match_latest`. It focuses on the end-to-end matching process flow and the fuzzy matching logic used for person-name comparison.

The documentation is grounded in the current source code and existing project docs. It covers the CLI and GUI-facing workflow, matching algorithm choices, cascade matching, CPU/GPU execution paths, output behavior, and pseudocode for the fuzzy matching implementation.

Screenshots were not captured for this documentation pass because the requested deliverable is code-backed technical documentation and no running UI target was provided. Image proof unavailable.

## Application Overview

`name_matcher` is a Rust application for matching person records between two MySQL tables. It supports:

- Deterministic exact matching on normalized name and birthdate fields.
- Fuzzy name matching using Levenshtein similarity, Jaro-Winkler similarity, and Double Metaphone phonetic comparison.
- Household-level matching that aggregates matched people into household match percentages.
- A cascade workflow that runs levels L1-L11 in order and can exclude already-matched records from later levels.
- CPU execution, optional CUDA/GPU acceleration, streaming/partitioned execution, CSV/XLSX export, and a GUI binary.

## Name Matching App Description

The Name Matcher app is a desktop and command-line data-matching tool designed to compare person records from two database tables and produce match results for review or downstream processing. It is especially useful when records may contain spelling differences, missing middle names, inconsistent household identifiers, or large datasets that need batch processing.

| Item | Description |
| --- | --- |
| Application name | `name_matcher` |
| Primary language | Rust, edition 2024 |
| Main package | `Cargo.toml` package `name_matcher`, version `0.1.0` |
| User interfaces | CLI binary `name_matcher` and GUI binary `gui` |
| Database | MySQL through `sqlx` |
| Output formats | CSV, XLSX, or both |
| Matching styles | Deterministic exact matching, fuzzy name matching, household matching, weighted Levenshtein matching, and cascade matching |
| Performance tools | Rayon parallelism, streaming/partitioned execution, optional CUDA/GPU acceleration |
| Target users | Operators or developers who need to reconcile person records across source and target tables |

### Tools and Frameworks

| Tool or crate | Purpose |
| --- | --- |
| Rust/Cargo | Build system, dependency management, CLI/GUI binaries. |
| Tokio | Async runtime used with database and long-running operations. |
| SQLx | MySQL access and row loading. |
| Clap | CLI argument parsing, including environment-aware arguments. |
| eframe/egui | Desktop GUI framework. |
| Rayon | Parallel CPU processing for in-memory matching and fuzzy cache preparation. |
| cudarc | Optional CUDA support when the `gpu` feature is enabled. |
| csv | CSV export. |
| rust_xlsxwriter | XLSX export. |
| strsim | Levenshtein, normalized Levenshtein, and Jaro-Winkler similarity. |
| rphonetic | Double Metaphone phonetic encoding. |
| unicode-normalization | Text normalization for phonetic handling. |
| chrono | Date handling, especially birthdate matching. |
| serde | Serialization for configuration/result data structures. |
| sysinfo | Memory and system statistics for execution tuning. |
| log/env_logger/tracing | Runtime diagnostics and progress logging. |

Primary entrypoints and configuration surfaces:

| Area | File | Purpose |
| --- | --- | --- |
| CLI runtime | `src/main.rs` | Parses top-level flags, chooses algorithm, configures execution mode, calls matching/export paths. |
| CLI argument parsing | `src/cli/args.rs` | Legacy positional/environment argument parser. |
| CLI flags | `src/cli/flags.rs` | GPU, streaming, cascade, threshold, and advanced option flags. |
| Clap parser | `src/cli/clap_parser.rs` | Modern CLI definition for the `name_matcher` binary. |
| Matching core | `src/matching/mod.rs` | Algorithms, fuzzy comparison functions, streaming/in-memory matching, GPU toggles, match pair structures. |
| Advanced levels | `src/matching/advanced_matcher.rs` | L1-L12 advanced matching levels and in-memory matching for exact/fuzzy levels. |
| Cascade workflow | `src/matching/cascade.rs` | Sequential L1-L11 cascade, level output files, skipping rules, and summary output. |
| GPU batching | `src/matching/gpu/batch.rs` | Candidate batching and CUDA-backed fuzzy work when the `gpu` feature is enabled. |
| Normalization helpers | `src/matching/helpers.rs` | Simple text normalization, Levenshtein percentage, Double Metaphone, and Soundex helpers. |
| Export | `src/export/*.rs` | CSV/XLSX export and summary output. |

## Installation Guide

### Option A: Pre-Built Release

Use this route when you only need to run the app.

1. Open the repository's GitHub Releases page.
2. Download the correct archive for your platform:
   - `gui-windows-cpu-<tag>.zip` for Windows GUI CPU builds.
   - `gui-linux-cpu-<tag>.tar.gz` for Linux GUI CPU builds.
3. Extract the archive to a local folder.
4. Run the `gui` executable directly.

No Rust toolchain is required for pre-built GUI releases.

### Option B: Build from Source

Use this route when you need the CLI, need to modify the app, or need a local build with specific feature flags.

#### 1. Install Rust

Windows:

```text
Download and install Rust from https://rustup.rs
```

Linux/macOS:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

#### 2. Clone or Open the Repository

```bash
git clone <repository-url>
cd name_match_latest
```

If the repository is already present locally, open:

```text
D:\GitProjects\name_match_latest
```

#### 3. Build the CPU CLI Release

```powershell
cargo build --release
```

The CLI binary is generated at:

```text
target/release/name_matcher.exe
```

On Linux/macOS, the binary is:

```text
target/release/name_matcher
```

#### 4. Build with GPU Support

GPU support requires CUDA Toolkit and `nvcc` on `PATH`.

```powershell
cargo build --release --features gpu
```

Use GPU builds when you want CUDA-backed hash joins, fuzzy metric acceleration, or GPU-assisted household matching.

#### 5. Build the GUI

Windows helper script:

```powershell
# CPU GUI build
powershell -File scripts\windows\Build-Release-Gui.ps1

# GPU GUI build
powershell -File scripts\windows\Build-Release-Gui.ps1 -Gpu
```

Manual build:

```powershell
cargo build --release --bin gui
cargo build --release --features gpu --bin gui
```

The GUI binary is generated at:

```text
target/release/gui.exe
```

On Linux/macOS, the binary is:

```text
target/release/gui
```

### Option C: CI or Release Builds

CPU builds run on GitHub-hosted runners. GPU builds require a self-hosted Windows runner with CUDA installed. The repo references this setup in `docs/self_hosted_runner_windows_cuda.md`.

### Database Setup

Both input tables should provide the fields needed by the selected algorithm.

| Column | Required For | Notes |
| --- | --- | --- |
| `id` | All workflows | Primary row identifier. |
| `uuid` | Most workflows, household matching | Used as a source/person or household identifier. |
| `first_name` | All name matching | Normalized before comparison. |
| `last_name` | All name matching | Normalized before comparison. |
| `birthdate` | Deterministic and fuzzy person matching | Hard gate for fuzzy matching. |
| `middle_name` | Algorithm 2, Algorithm 3, L1, L2, L4, L5, L7, L8, L10 | Optional for no-middle flows. |
| `hh_id` | Algorithms 5 and 6 | Required for household matching. |
| `barangay_code` | Cascade L4-L6 | Missing-column behavior depends on cascade mode. |
| `city_code` | Cascade L7-L9 | Missing-column behavior depends on cascade mode. |

Database connection values may come from environment variables, a `.env` file, CLI positional arguments, or GUI form fields.

Common environment variables:

```text
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=root
DB_PASSWORD=<password>
DB_NAME=matching_db
TABLE1=source_table
TABLE2=target_table
ALGO=3
OUT_PATH=D:/out/matches.csv
```

Dual-database mode uses a second set of database values, such as:

```text
DB2_HOST=remote-host
DB2_PORT=3306
DB2_USER=reader
DB2_PASS=<password>
DB2_DATABASE=other_db
```

## Build and Runtime Requirements

The project uses Rust 2024 and the `name_matcher` package in `Cargo.toml`.

### Dependency Summary

Runtime and build dependencies are declared in `Cargo.toml`.

| Dependency | Version | Use |
| --- | --- | --- |
| `anyhow` | `1.0.99` | Error handling. |
| `chrono` | `0.4.42` | Date/birthdate handling with Serde support. |
| `clap` | `4` | CLI parsing with derive/env support. |
| `csv` | `1.3.1` | CSV output. |
| `cudarc` | `0.17.3`, optional | CUDA GPU support with NVRTC. |
| `eframe` / `egui` | `0.32.3` | Desktop GUI. |
| `env_logger`, `log`, `tracing`, `tracing-log`, `tracing-subscriber` | see `Cargo.toml` | Logging and diagnostics. |
| `rayon` | `1.11.0` | Parallel CPU processing. |
| `rfd` | `0.15.4` | GUI file dialogs. |
| `rphonetic` | `3.0.4` | Double Metaphone phonetic matching. |
| `rust_xlsxwriter` | `0.90.1` | XLSX output. |
| `serde` | `1.0.225` | Serialization. |
| `sqlx` | `0.8.6` | MySQL database access over Tokio/Rustls. |
| `strsim` | `0.11.1` | String similarity metrics. |
| `sysinfo` | `0.37.0` | Memory/system stats. |
| `thiserror` | `2.0.16` | Structured errors. |
| `tokio` | `1.47.1` | Async runtime. |
| `unicode-normalization` | `0.1.24` | Unicode normalization for phonetic matching. |
| `uuid` | `1.18.1` | UUID support. |

Feature flags:

| Feature | Purpose |
| --- | --- |
| `gpu` | Enables optional `cudarc` CUDA acceleration. |
| `new_cli` | Enables newer CLI code paths where compiled. |
| `new_engine` | Enables newer matching engine adapters where compiled. |

CPU-only build:

```powershell
cargo build --release
```

GPU-enabled build:

```powershell
cargo build --release --features gpu
```

GUI build examples:

```powershell
cargo build --release --bin gui
cargo build --release --features gpu --bin gui
```

GPU support is optional and depends on the `gpu` feature and CUDA tooling. When GPU paths fail due to CUDA errors or out-of-memory conditions, several paths fall back to CPU through `with_oom_cpu_fallback` in `src/matching/gpu_config.rs`.

## CLI Usage

Basic CLI shape from `docs/usage_guide.md` and `src/main.rs`:

```text
name_matcher <host> <port> <user> <password> <database> <table1> <table2> <algo> <out_path> [format] [flags...]
```

Parameters:

| Position | Name | Description |
| --- | --- | --- |
| 1 | `host` | MySQL host. |
| 2 | `port` | MySQL port. |
| 3 | `user` | MySQL username. |
| 4 | `password` | MySQL password. Do not paste real credentials into documentation or logs. |
| 5 | `database` | Database name. |
| 6 | `table1` | Source table. |
| 7 | `table2` | Target table. |
| 8 | `algo` | Algorithm number, 1-7. |
| 9 | `out_path` | Output path. |
| 10 | `format` | `csv`, `xlsx`, or `both`; defaults to CSV. |

Examples:

```powershell
# Deterministic match, CSV output
name_matcher 127.0.0.1 3306 root secret mydb persons targets 1 D:/out/matches.csv

# Fuzzy match with middle name, XLSX output
name_matcher 127.0.0.1 3306 root secret mydb persons targets 3 D:/out/matches.xlsx xlsx

# Deterministic match with GPU hash join
name_matcher 127.0.0.1 3306 root secret mydb persons targets 1 D:/out/matches both --gpu-hash-join
```

Common fuzzy and cascade flags:

| Flag | Purpose |
| --- | --- |
| `--gpu-fuzzy-direct-hash` | Uses a GPU hash pre-pass for the fuzzy direct phase as candidate filtering. |
| `--direct-fuzzy-normalization` | Applies fuzzy-style normalization to algorithms 1 and 2 before equality checks. |
| `--gpu-fuzzy-metrics` | Uses GPU kernels for Levenshtein/Jaro/Jaro-Winkler scoring when enabled and beneficial. |
| `--gpu-fuzzy-force` | Forces GPU fuzzy metrics even if the heuristic says CPU is likely faster. |
| `--gpu-fuzzy-disable` | Disables GPU fuzzy metrics. |
| `--cascade` | Runs cascade matching L1-L11. L12 is excluded. |
| `--cascade-missing-columns MODE` | Handles missing geo columns with `auto-skip`, `manual`, or `abort`. |
| `--cascade-levels` | Restricts cascade execution to selected levels 1-11. |

## High-Level Process Flow

```text
User selects CLI/GUI inputs
  -> database connection and table names are resolved
  -> algorithm number and flags are parsed
  -> records are loaded in in-memory, streaming, or partitioned mode
  -> selected matching path runs:
       algorithms 1-2: deterministic normalized exact matching
       algorithms 3-4: fuzzy person matching with birthdate gate
       algorithms 5-6: household aggregation over fuzzy person matches
       algorithm 7: weighted Levenshtein matching
       cascade: advanced levels L1-L11 in sequence
  -> MatchPair or HouseholdAggRow results are generated
  -> CSV/XLSX exporters write output files
  -> summary metadata and progress are reported
```

### Data Flow

1. The runtime parses database settings, table names, algorithm, output path, output format, and optional GPU/cascade flags.
2. Database rows are mapped into `Person` records, including core fields such as `id`, `uuid`, names, birthdate, and `extra_fields`.
3. Matching logic normalizes relevant text values before comparison.
4. The matching function emits `MatchPair` records for person-to-person matches or `HouseholdAggRow` records for household algorithms.
5. Export code writes rows to CSV/XLSX and includes confidence and matched-field metadata.

## Matching Algorithms

| Algorithm | Name | Core Behavior |
| --- | --- | --- |
| 1 | Deterministic first + last + birthdate | Exact match after normalization; middle name ignored. |
| 2 | Deterministic first + middle + last + birthdate | Strict exact match after normalization. |
| 3 | Fuzzy with middle name | First + middle + last fuzzy scoring, gated by birthdate. |
| 4 | Fuzzy without middle name | First + last fuzzy scoring, gated by birthdate. |
| 5 | Household Table1 -> Table2 | Person fuzzy matches are aggregated from Table 1 UUIDs to Table 2 households. |
| 6 | Household Table2 -> Table1 | Reverse household direction; denominator follows Table 2 household size. |
| 7 | Levenshtein weighted | SQL-equivalent weighted Levenshtein scoring with optional GPU paths. |

`MatchingAlgorithm` is defined in `src/matching/mod.rs` and mapped from CLI algorithm numbers in `src/main.rs` and `src/cli/args.rs`.

## Fuzzy Matching Overview

The primary fuzzy functions live in `src/matching/mod.rs`:

- `fuzzy_compare_names_new`: compares first + middle + last.
- `fuzzy_compare_names_no_mid`: compares first + last.
- `compare_persons_new_with_swap`: applies birthdate gating before full-name fuzzy comparison.
- `compare_persons_no_mid_with_swap`: applies birthdate gating before no-middle fuzzy comparison.
- `build_cpu_fuzzy_cache`: precomputes normalized strings and Double Metaphone codes.
- `classify_cached_full`: classifies cached first + middle + last matches.
- `classify_cached_no_mid`: classifies cached first + last matches.

Supporting helpers in `src/matching/helpers.rs`:

- `normalize_simple`: trims text, removes dots, replaces dashes with spaces, and lowercases.
- `sim_levenshtein_pct`: converts Levenshtein edit distance into a 0-100 similarity percentage.
- `normalize_for_phonetic`: normalizes text for phonetic encoding.
- `metaphone_pct`: returns 100 when Double Metaphone encodings match, otherwise 0.
- `soundex4_ascii`: builds a 4-character Soundex key used in some GPU blocking paths.

### Fuzzy Match Rules

Fuzzy comparison first builds normalized names:

- Algorithm 3/L10: `first middle last`
- Algorithm 4/L11: `first last`

Then it computes:

| Metric | Implementation | Range |
| --- | --- | --- |
| Levenshtein similarity | `sim_levenshtein_pct` | 0-100 |
| Jaro-Winkler | `strsim::jaro_winkler(...) * 100.0` | 0-100 |
| Double Metaphone | `metaphone_pct` or cached Double Metaphone equality | 0 or 100 |

Classification:

| Label | Rule | Score |
| --- | --- | --- |
| `DIRECT MATCH` | Normalized full strings are equal. | 100 |
| `CASE 1` | Levenshtein >= 85, Jaro-Winkler >= 85, Metaphone = 100. | Average of all three metrics. |
| `CASE 2` | At least two of the three metric checks pass. | Average of all three metrics. |
| `CASE 3` | Case 2 with average >= 88 and component Levenshtein distances <= 2. | Average of all three metrics. |
| No match | None of the above. | No `MatchPair`. |

For full-name fuzzy matching, Case 3 checks first, last, and middle component edit distances. For no-middle matching, it checks first and last only.

Birthdate is a hard gate for person-level fuzzy matches. The code requires both records to have birthdates and rejects pairs when the birthdates do not match, except where optional month/day swapping is enabled.

## Fuzzy Matching Pseudocode

### Core Name Classification

```text
function fuzzy_compare_names(first1, middle1, last1, first2, middle2, last2, include_middle):
    if include_middle:
        full1 = normalize_simple(first1 + " " + middle1 + " " + last1)
        full2 = normalize_simple(first2 + " " + middle2 + " " + last2)
    else:
        full1 = normalize_simple(first1 + " " + last1)
        full2 = normalize_simple(first2 + " " + last2)

    if full1 is empty or full2 is empty:
        return no match

    if full1 == full2:
        return score 100, label "DIRECT MATCH"

    lev = levenshtein_similarity_percent(full1, full2)
    jw = jaro_winkler(full1, full2) * 100
    mp = 100 if double_metaphone(full1) == double_metaphone(full2) else 0

    if lev >= 85 and jw >= 85 and mp == 100:
        return average(lev, jw, mp), label "CASE 1"

    pass_count = 0
    if lev >= 85: pass_count += 1
    if jw >= 85: pass_count += 1
    if mp == 100: pass_count += 1

    if pass_count >= 2:
        avg = average(lev, jw, mp)

        if avg >= 88:
            first_dist = levenshtein(normalize_simple(first1), normalize_simple(first2))
            last_dist = levenshtein(normalize_simple(last1), normalize_simple(last2))

            if include_middle:
                middle_dist = levenshtein(normalize_simple(middle1), normalize_simple(middle2))
                if first_dist <= 2 and last_dist <= 2 and middle_dist <= 2:
                    return avg, label "CASE 3"
            else:
                if first_dist <= 2 and last_dist <= 2:
                    return avg, label "CASE 3"

        return avg, label "CASE 2"

    return no match
```

### Person-Level Fuzzy Matching

```text
function compare_persons_fuzzy(person1, person2, include_middle, allow_birthdate_swap):
    if person1.birthdate is missing or person2.birthdate is missing:
        return no match

    if person1.birthdate != person2.birthdate:
        if not birthdate_matches_with_optional_swap(person1.birthdate, person2.birthdate, allow_birthdate_swap):
            return no match

    return fuzzy_compare_names(
        person1.first_name,
        person1.middle_name,
        person1.last_name,
        person2.first_name,
        person2.middle_name,
        person2.last_name,
        include_middle
    )
```

### Cached Advanced Fuzzy Matching

`advanced_match_inmemory` builds per-person fuzzy caches for L10 and L11 before scoring. This avoids repeated normalization and Double Metaphone work.

```text
function advanced_fuzzy_level(table1, table2, level, threshold, allow_birthdate_swap):
    cache1 = build_cpu_fuzzy_cache for every person in table1
    cache2 = build_cpu_fuzzy_cache for every person in table2

    by_birthdate = map from birthdate key to table2 indexes
    for each person2 in table2:
        for each key from birthdate_keys(person2.birthdate, allow_birthdate_swap):
            by_birthdate[key].append(person2 index)

    output = []
    for each person1 in table1:
        if person1.birthdate is missing:
            continue

        seen_inner_ids = empty set
        for each key from birthdate_keys(person1.birthdate, allow_birthdate_swap):
            for each person2 index in by_birthdate[key]:
                if person2 was already seen for this person1:
                    continue

                if level is L10:
                    if not match_level_10(person1.birthdate, person2.birthdate, allow_birthdate_swap):
                        continue
                    result = classify_cached_full(cache1[person1], cache2[person2])
                    require both middle names to have at least 2 non-dot, non-space characters

                if level is L11:
                    if not match_level_11(person1.birthdate, person2.birthdate, allow_birthdate_swap):
                        continue
                    result = classify_cached_no_mid(cache1[person1], cache2[person2])

                if result exists and result.score / 100 >= threshold:
                    emit MatchPair with confidence and matched_fields

    return output sorted by outer record order
```

## Cascade Matching Process Flow

The cascade is implemented in `src/matching/cascade.rs`. It runs levels L1-L11 and explicitly excludes L12 household matching.

Levels:

| Level | Type | Fields |
| --- | --- | --- |
| L1 | Exact | Last + First + Full Middle + Birthdate |
| L2 | Exact | Last + First + Middle Initial(s) + Birthdate |
| L3 | Exact | Last + First + Birthdate |
| L4 | Exact | Last + First + Full Middle + `barangay_code` |
| L5 | Exact | Last + First + Middle Initial(s) + `barangay_code` |
| L6 | Exact | Last + First + `barangay_code` |
| L7 | Exact | Last + First + Full Middle + `city_code` |
| L8 | Exact | Last + First + Middle Initial(s) + `city_code` |
| L9 | Exact | Last + First + `city_code` |
| L10 | Fuzzy | Last + First + Full Middle + Birthdate |
| L11 | Fuzzy | Last + First + Birthdate |

Cascade pseudocode:

```text
function run_cascade(table1, table2, config, geo_status):
    levels = config.levels if provided else [1..11]
    remaining_table1 = copy(table1)
    remaining_table2 = copy(table2)
    results = []

    for level in levels:
        if geo_status says level cannot run:
            if config.missing_column_mode is AutoSkip:
                record skipped level
                continue
            otherwise record failure or omit according to mode

        adv_config = build AdvConfig(level, threshold, allow_birthdate_swap)

        if config.exclusion_mode is Exclusive:
            input1 = remaining_table1
            input2 = remaining_table2
        else:
            input1 = table1
            input2 = table2

        if level is L10 or L11:
            matches = run_fuzzy_level(input1, input2, adv_config, backend, gpu_device)
        else:
            matches = advanced_match_inmemory(input1, input2, adv_config)
            sort matches by person IDs

        write level CSV to base_output_Lxx.csv
        record match count and output path

        if config.exclusion_mode is Exclusive:
            remove matched table1 IDs from remaining_table1
            remove matched table2 IDs from remaining_table2

    write summary with level statuses, output files, total matches, and duration
```

Missing geographic columns are handled by `GeoColumnStatus`:

- L4-L6 require `barangay_code`.
- L7-L9 require `city_code`.
- L1-L3 and L10-L11 do not require geographic columns.
- L12 is rejected for cascade.


## Sample Performance Metrics: GPU vs CPU 10M x 10M

This section gives planning-level runtime estimates for fuzzy matching 10 million records against 10 million records. The estimates are based on the local summary reports currently present in the repository, plus explicit assumptions where no direct CPU benchmark exists.

### Observed Local GPU Runs

The repo contains local summary reports for Advanced L10 fuzzy matching (`Advanced L10FuzzyBirthdateFullMiddle`). These are not 10M x 10M runs, but they provide real measured GPU timings from this workspace.

| Report File | Table 1 Rows | Table 2 Rows | Algorithm | Fuzzy Mode | GPU Total MB | Duration | Matches |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| `summary_report_2026-01-01_18-20-41.csv` | 208,069 | 3,017,093 | Advanced L10 fuzzy | GPU | 6,140 | 00:01:37 | 354 |
| `summary_report_2026-01-01_18-25-46.csv` | 208,069 | 3,017,093 | Advanced L10 fuzzy | GPU | 6,140 | 00:03:54 | 2,457 |
| `summary_report_2026-01-04_21-21-10.csv` | 208,069 | 3,017,093 | Advanced L10 fuzzy | GPU | 6,140 | 00:01:04 | 354 |

The observed input-pair scale is approximately:

```text
208,069 x 3,017,093 = 627,763,950,917 possible raw record pairs
```

Fuzzy matching does not usually evaluate every raw pair directly. The implementation uses birthdate gates, blocking, candidate filtering, partitioning, and optional GPU prepasses. The raw-pair count is still useful as a coarse scale reference.

### 10M x 10M Scale Reference

```text
10,000,000 x 10,000,000 = 100,000,000,000,000 possible raw record pairs
```

Compared with the observed local GPU reports, a 10M x 10M run is about 159.3x larger by raw-pair scale:

```text
100,000,000,000,000 / 627,763,950,917 ~= 159.3
```

### Planning Estimate for GPU 10M x 10M

If the 10M x 10M data has similar blocking selectivity, similar name lengths, similar birthdate distribution, similar match thresholds, and the same approximate GPU class, a simple scale-up from observed GPU runs gives this rough planning range:

| Basis | Observed Duration | Scale Factor | Estimated 10M x 10M GPU Runtime |
| --- | ---: | ---: | ---: |
| Fast observed GPU run | 00:01:04 | 159.3x | ~2.8 hours |
| Middle observed GPU run | 00:01:37 | 159.3x | ~4.3 hours |
| Slow observed GPU run | 00:03:54 | 159.3x | ~10.4 hours |

Practical GPU planning estimate:

```text
GPU fuzzy matching, 10M x 10M: roughly 3 to 11 hours under similar blocking/selectivity conditions.
```

Use the high end when data is less clean, birthdates are dense or duplicated, names are longer, GPU memory is tight, output volume is high, or the run falls back to CPU for some chunks.

### Planning Estimate for CPU 10M x 10M

No direct local CPU fuzzy summary report for the same workload was found in the repository. Because of that, the CPU estimate below is not measured; it is a planning assumption based on GPU-vs-CPU multiplier ranges.

| Assumed CPU Slowdown vs GPU | Estimated CPU Runtime for 10M x 10M |
| ---: | ---: |
| 5x slower than GPU | ~14 to 52 hours |
| 10x slower than GPU | ~28 to 104 hours |
| 20x slower than GPU | ~56 to 208 hours |
| 30x slower than GPU | ~84 to 312 hours |

Practical CPU planning estimate:

```text
CPU fuzzy matching, 10M x 10M: likely measured in days, not hours, unless blocking is extremely selective.
A cautious planning range is ~1 to 13 days depending on CPU hardware and candidate count.
```

### Recommended Benchmark Before a Full 10M x 10M Run

Run a representative subset first, then extrapolate from actual candidate volume and throughput.

Recommended sample sizes:

| Sample | Purpose |
| --- | --- |
| 100k x 100k | Fast smoke test for config, output, and GPU activation. |
| 500k x 500k | Better estimate of birthdate/name blocking behavior. |
| 1M x 1M | Stronger basis for production runtime planning. |

Record these values for each benchmark:

- Table 1 row count.
- Table 2 row count.
- Algorithm and level, especially Algorithm 3/4 or Advanced L10/L11.
- GPU mode and GPU memory total/free.
- Candidate count if logged.
- Match count.
- Fetch time, matching time, export time, and total duration.
- Whether GPU was actually used or CPU fallback occurred.

Recommended GPU command shape for a full fuzzy run:

```powershell
name_matcher <host> <port> <user> <password> <database> <table1> <table2> 3 D:/out/fuzzy_10m_gpu.csv csv --gpu-fuzzy-metrics --gpu-fuzzy-direct-hash
```

Recommended CPU comparison command shape:

```powershell
name_matcher <host> <port> <user> <password> <database> <table1> <table2> 3 D:/out/fuzzy_10m_cpu.csv csv --gpu-fuzzy-disable
```

For 10M x 10M runs, prefer partitioned or streaming execution and monitor output size. Export time can become significant if the match count is large.

## GPU and CPU Fuzzy Paths

GPU support is optional and feature-gated. The fuzzy matching design uses GPU acceleration for candidate generation or metric scoring, while preserving CPU-compatible final behavior.

Important paths:

- `set_gpu_fuzzy_direct_prep` controls the in-memory GPU fuzzy direct pre-pass.
- `gpu_fuzzy_direct_hash_prefilter_indices` builds candidate lists for fuzzy matching.
- `match_all_with_opts` routes Algorithm 3/4 to GPU or CPU based on `MatchOptions`, feature flags, runtime toggles, and heuristics.
- `should_enable_gpu_fuzzy_by_heuristic` decides whether GPU fuzzy metrics are likely worthwhile unless forced.
- `cascade_match_fuzzy_gpu` and `cascade_match_fuzzy_no_mid_gpu` expose GPU fuzzy matching for cascade L10/L11.
- `with_oom_cpu_fallback` catches CUDA OOM cases and falls back to CPU.

CPU remains the required fallback path. For advanced/cascade fuzzy levels, the code sorts matches by IDs after GPU/CPU paths to preserve deterministic output ordering.

## Output Artifacts

Person-level matching emits `MatchPair`:

| Field | Meaning |
| --- | --- |
| `person1` | Source-side person record. |
| `person2` | Target-side person record. |
| `confidence` | Match confidence. Some paths store 0-100, while export/runtime paths may normalize or threshold according to the selected mode. |
| `matched_fields` | Labels such as `fuzzy`, `CASE 1`, `birthdate`, or exact field names. |
| `is_matched_infnbd` | Deterministic algorithm 1 flag. |
| `is_matched_infnmnbd` | Deterministic algorithm 2 flag. |

Household matching emits `HouseholdAggRow`, which includes `uuid`, `hh_id`, `match_percentage`, and optional household metadata.

Cascade output uses per-level CSV paths:

```text
base_output.csv -> base_output_L01.csv
base_output.csv -> base_output_L02.csv
...
```

The cascade summary path is:

```text
base_output.csv -> base_output_summary.txt
```

## Validation Notes

Documentation generation validation performed:

- Inspected repository structure and git status.
- Used Context Engine MCP first for codebase retrieval and targeted file reads.
- Inspected matching source files:
  - `src/matching/mod.rs`
  - `src/matching/helpers.rs`
  - `src/matching/advanced_matcher.rs`
  - `src/matching/cascade.rs`
  - `src/matching/gpu/batch.rs`
  - `src/main.rs`
  - `src/cli/args.rs`
  - `src/cli/flags.rs`
- Inspected existing docs:
  - `README.md`
  - `docs/usage_guide.md`
  - `docs/matching_algorithms.md`
  - `DOCUMENTATION.md`
- Created this generated documentation under `docs/generated/`, inside the workspace.

Code build/tests were not run as part of this documentation-only pass. The current working tree already had many modified and untracked files before the generated docs were added, so this pass avoided touching existing source files or existing documentation.

Suggested validation command for a future build check:

```powershell
cargo test --all
cargo build --release
```

For GPU-specific validation:

```powershell
cargo build --release --features gpu
```

## Known Limitations and Assumptions

- This document describes the source as inspected in the current working tree, which already contained local modifications.
- Screenshots were not captured because no running app URL, login flow, or required UI evidence target was provided.
- Fuzzy confidence representation differs across some paths: core comparison functions return 0-100 scores, while some `MatchPair` construction paths divide by 100 before export/threshold handling. Consumers should verify the expected scale at the export boundary for the selected execution mode.
- GPU fuzzy metrics are optional and may be disabled by feature flags, runtime flags, heuristic decisions, or CUDA fallback behavior.
- L12 household matching is intentionally excluded from cascade and should be run through the dedicated household matching algorithms.
