# SRS-II Name Matching Application

**Author:** Matthias Tangonan
**Version:** 0.1.0
**Language:** Rust (Edition 2024)

---

## I. Objective and Goals

The SRS-II Name Matching Application is a high-performance tool for matching person records across two MySQL database tables. It supports deterministic (exact-field) matching, fuzzy scoring (Levenshtein, Jaro-Winkler, Metaphone), household-level GPU-accelerated matching, and a cascading multi-level workflow that progressively relaxes matching criteria.

**Primary goals:**
- Match person records between a source table (Table 1) and a target table (Table 2) using configurable algorithms
- Support both CLI batch processing and an interactive desktop GUI
- Leverage CUDA GPU acceleration for large-scale workloads
- Export results to CSV and/or XLSX with summary reports

---

## II. Hardware and Software Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Windows 10/11 (primary), Linux (supported) |
| CPU | Any x86_64 processor (multi-core recommended) |
| RAM | 4 GB minimum, 8+ GB recommended for large datasets |
| Disk | SSD recommended for streaming mode |
| Database | MySQL 5.7+ or MariaDB 10.3+ |

### For GPU Acceleration (Optional)

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA GPU with CUDA support (Compute Capability 5.0+) |
| Driver | NVIDIA driver with CUDA 11.0+ |
| Toolkit | CUDA Toolkit (nvcc) for building from source |
| VRAM | 2+ GB recommended |

### Build Requirements (from source)

| Component | Requirement |
|-----------|-------------|
| Rust | Stable toolchain (edition 2024) |
| Cargo | Included with Rust |
| CUDA Toolkit | Required only for `--features gpu` builds |

---

## III. Installation Guide

### Option A: Pre-built Release (Recommended)

1. Go to the GitHub Releases page for this repository.
2. Download the appropriate archive:
   - `gui-windows-cpu-<tag>.zip` — Windows GUI (CPU only)
   - `gui-linux-cpu-<tag>.tar.gz` — Linux GUI (CPU only)
3. Extract the archive to your desired location.
4. Run the `gui` executable directly (no installation required).

### Option B: Build from Source

#### 1. Install Rust

```bash
# Windows: download from https://rustup.rs
# Linux/macOS:
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

#### 2. Clone the Repository

```bash
git clone <repository-url>
cd name_match_latest
```

#### 3. Build CPU-Only Release

```bash
cargo build --release
```

The CLI binary is at `target/release/name_matcher(.exe)`.

#### 4. Build with GPU Support

Requires CUDA Toolkit (nvcc) on PATH:

```bash
cargo build --release --features gpu
```

#### 5. Build GUI

**Windows (PowerShell):**
```powershell
# CPU only
powershell -File scripts\windows\Build-Release-Gui.ps1

# With GPU
powershell -File scripts\windows\Build-Release-Gui.ps1 -Gpu
```

**Manual:**
```bash
cargo build --release --bin gui
# or with GPU:
cargo build --release --features gpu --bin gui
```

The GUI binary is at `target/release/gui(.exe)`.

### Option C: Docker / CI

CPU builds run on GitHub-hosted runners. GPU builds require a self-hosted Windows runner with CUDA. See `docs/self_hosted_runner_windows_cuda.md`.

---

## IV. Database Setup

### Required Table Schema

Both Table 1 (source) and Table 2 (target) must have these columns:

| Column | Type | Required |
|--------|------|----------|
| `id` | INT (auto-increment) | Yes |
| `uuid` | VARCHAR | Yes |
| `first_name` | VARCHAR | Yes |
| `last_name` | VARCHAR | Yes |
| `birthdate` | DATE | Yes |
| `middle_name` | VARCHAR | No (used by Algo 2, 3) |
| `hh_id` | VARCHAR | No (required for Algo 5, 6) |
| `barangay_code` | VARCHAR | No (required for Cascade L4-L6) |
| `city_code` | VARCHAR | No (required for Cascade L7-L9) |

### Connection Configuration

Database credentials can be provided via:

1. **Environment variables** (highest priority):
   ```
   DB_HOST=127.0.0.1
   DB_PORT=3306
   DB_USER=root
   DB_PASSWORD=secret
   DB_NAME=matching_db
   ```

2. **`.env` file** in the working directory

3. **CLI positional arguments** (lowest priority)

4. **GUI form fields** (for the desktop application)

#### Dual-Database Mode

For matching across two separate databases, set:
```
DB2_HOST=remote-host
DB2_PORT=3306
DB2_USER=reader
DB2_PASS=secret
DB2_DATABASE=other_db
```

---

## V. Usage Guide — CLI

### Basic Syntax

```
name_matcher <host> <port> <user> <password> <database> <table1> <table2> <algo> <out_path> [format] [flags...]
```

### Parameters

| Position | Name | Description |
|----------|------|-------------|
| 1 | host | MySQL host address |
| 2 | port | MySQL port (e.g., 3306) |
| 3 | user | Database username |
| 4 | password | Database password |
| 5 | database | Database name |
| 6 | table1 | Source table name |
| 7 | table2 | Target table name |
| 8 | algo | Algorithm number (1–7) |
| 9 | out_path | Output file path |
| 10 | format | Output format: `csv`, `xlsx`, or `both` (default: csv) |

### Examples

```bash
# Deterministic match (Algo 1), CSV output
name_matcher 127.0.0.1 3306 root secret mydb persons targets 1 D:/out/matches.csv

# Fuzzy match (Algo 3), XLSX output
name_matcher 127.0.0.1 3306 root secret mydb persons targets 3 D:/out/matches.xlsx xlsx

# Both formats with GPU hash join
name_matcher 127.0.0.1 3306 root secret mydb persons targets 1 D:/out/matches both --gpu-hash-join
```

### GPU Flags

| Flag | Environment Variable | Description |
|------|---------------------|-------------|
| `--gpu-hash-join` | `NAME_MATCHER_GPU_HASH_JOIN=1` | GPU hash join prefilter (Algo 1/2) |
| `--gpu-fuzzy-direct-hash` | `NAME_MATCHER_GPU_FUZZY_DIRECT_HASH=1` | GPU hash pre-pass for Fuzzy direct phase |
| `--gpu-levenshtein-prepass` | `NAME_MATCHER_GPU_LEVENSHTEIN_PREPASS=1` | GPU pre-pass for Algo 7 |
| `--gpu-levenshtein-full-scoring` | `NAME_MATCHER_GPU_LEVENSHTEIN_FULL_SCORING=1` | GPU full scoring for Algo 7 |
| `--gpu-fuzzy-metrics` | `NAME_MATCHER_GPU_FUZZY_METRICS=1` | GPU Levenshtein/Jaro/Jaro-Winkler (Algo 3/4) |
| `--gpu-fuzzy-force` | `NAME_MATCHER_GPU_FUZZY_FORCE=1` | Force GPU fuzzy even if heuristics say slower |
| `--gpu-fuzzy-disable` | `NAME_MATCHER_GPU_FUZZY_DISABLE=1` | Disable GPU fuzzy regardless |
| `--use-gpu` | `NAME_MATCHER_USE_GPU=1` | Force GPU backend for Algo 5 in-memory |
| `--auto-optimize` | `NAME_MATCHER_AUTO_OPTIMIZE=1` | Auto-detect hardware and apply optimal settings |

### Streaming and Performance Flags

| Flag / Env Var | Description |
|----------------|-------------|
| `NAME_MATCHER_STREAMING=1` | Enable streaming mode (reduces memory) |
| `NAME_MATCHER_POOL_SIZE=12` | MySQL connection pool size |
| `NAME_MATCHER_POOL_MIN=4` | Minimum pool connections |
| `RAYON_NUM_THREADS=12` | CPU parallelism threads |
| `NAME_MATCHER_GPU_STREAMS=2` | CUDA streams for overlap |
| `NAME_MATCHER_GPU_BUFFER_POOL=1` | Reuse GPU device buffers |
| `NAME_MATCHER_GPU_PINNED_HOST=1` | Use pinned host memory |

### Generate .env Template

```bash
name_matcher env-template [path]
```

---

## VI. Usage Guide — GUI

The GUI provides a visual interface for all matching operations.

### Launching

Run the `gui` binary directly:
```bash
./target/release/gui
# or on Windows:
target\release\gui.exe
```

### GUI Panels

1. **Database Configuration**
   - Host, port, username, password, database name
   - Optional dual-database mode (enable checkbox, fill second DB fields)
   - "Load Tables" button to connect and discover available tables

2. **Table Selection**
   - Dropdown selectors for Table 1 (source) and Table 2 (target)

3. **Algorithm Selection**
   - Dropdown for algorithms 1–7
   - Advanced matching toggle (cascade L1-L11)
   - Cascade configuration: missing column mode, geographic column status

4. **Execution Mode**
   - Auto / Streaming / In-Memory selector
   - Batch size, pool size, memory threshold configuration

5. **GPU Configuration**
   - Enable/disable GPU
   - Hash join, fuzzy direct hash, Levenshtein pre-pass toggles
   - Fuzzy GPU mode: Off / Auto / Force
   - GPU streams, buffer pool, pinned host memory
   - VRAM budget fields
   - Dynamic GPU auto-tuning toggle
   - Ultra Performance Mode button

6. **Output Configuration**
   - Output file path (with file picker via `rfd`)
   - Format: CSV / XLSX / Both
   - Fuzzy threshold slider (60–100%)

7. **Progress & Monitoring**
   - Progress bar with ETA
   - Records processed / total, records per second
   - Memory usage (used/available)
   - GPU status (total/free VRAM, active indicator)
   - Stage indicator, batch counter

8. **Diagnostics Panel**
   - Error event log with categorization
   - CUDA diagnostics panel
   - Log console (mirrors `log::info!` output)
   - Report export (Text/JSON)

9. **Controls**
   - Start / Cancel / Pause buttons
   - Birthdate swap toggle

---

## VII. Matching Algorithms Explained

### Algorithm 1 — Deterministic (First + Last + Birthdate)

Performs exact string comparison on normalized `first_name`, `last_name`, and `birthdate`. A match is recorded when all three fields are identical after normalization.

**Use case:** High-confidence deduplication where middle name data is unreliable or missing.

### Algorithm 2 — Deterministic (First + Middle + Last + Birthdate)

Same as Algorithm 1 but also requires exact match on `middle_name`.

**Use case:** Strictest matching when middle name data is reliable.

### Algorithm 3 — Fuzzy Match (with Middle Name)

Compares full names (first + middle + last) using a composite scoring system:
- **Levenshtein similarity** (percentage)
- **Jaro-Winkler similarity** (×100)
- **Double Metaphone** phonetic match (percentage)

**Scoring rules:**
- **Direct Match:** All fields identical → 100% confidence
- **Case 1:** Levenshtein ≥ 85%, Jaro-Winkler ≥ 85%, Metaphone = 100% → average of three scores
- **Case 2:** At least 2 of 3 metrics pass their thresholds → average score
- **Case 3:** Case 2 + average ≥ 88% + per-component Levenshtein distance ≤ 2 for each name part

Birthdate must match exactly (with optional month/day swap tolerance).

**Use case:** Matching records with minor spelling variations, transliteration differences, or typos.

### Algorithm 4 — Fuzzy Match (without Middle Name)

Same scoring logic as Algorithm 3 but excludes middle name from comparison. Only first + last names are scored.

**Use case:** When middle name data is sparse or inconsistent across tables.

### Algorithm 5 — Household Matching (Table1 → Table2)

GPU-accelerated matching that groups records by household ID (`hh_id`). Matches persons from Table 1 to households in Table 2 using `uuid` → `hh_id` linkage.

**Use case:** Linking individual records to household groupings.

### Algorithm 6 — Household Matching (Table2 → Table1)

Reverse direction of Algorithm 5: matches from Table 2 households to Table 1 individuals using `hh_id` → `uuid`.

**Use case:** When the denominator should be Table 2 size.

### Algorithm 7 — Levenshtein Weighted

A weighted Levenshtein distance algorithm that applies SQL-equivalent scoring logic with configurable thresholds. Supports GPU pre-pass for candidate filtering and full GPU scoring.

**Use case:** Fine-grained similarity scoring with weighted edit distances.

---

## VIII. Cascade Matching (Advanced — L1 through L11)

Cascade matching runs levels sequentially, progressively relaxing criteria. Records matched at earlier levels are excluded from later levels (configurable).

| Level | Match Criteria | Geographic Key |
|-------|---------------|----------------|
| L1 | Last + First + Full Middle + Birthdate | — |
| L2 | Last + First + Middle Initial(s) + Birthdate | — |
| L3 | Last + First + Birthdate (no middle) | — |
| L4 | Last + First + Full Middle + Barangay Code | barangay_code |
| L5 | Last + First + Middle Initial(s) + Barangay Code | barangay_code |
| L6 | Last + First + Barangay Code (no middle) | barangay_code |
| L7 | Last + First + Full Middle + City Code | city_code |
| L8 | Last + First + Middle Initial(s) + City Code | city_code |
| L9 | Last + First + City Code (no middle) | city_code |
| L10 | Fuzzy: Last + First + Full Middle + Exact Birthdate | — |
| L11 | Fuzzy: Last + First (no middle) + Exact Birthdate | — |

**Missing column handling modes:**
- **AutoSkip** (default): Skip levels requiring missing columns
- **ManualSelect**: Run only user-specified levels
- **AbortOnMissing**: Fail if geographic columns are absent

**Note:** L12 (Household Matching) is excluded from the cascade and must be run separately via Algorithms 5/6.

---

## IX. Normalization

All name comparisons are performed on normalized text:

1. **Unicode NFD decomposition** — decomposes characters into base + combining marks
2. **Diacritics removal** — strips combining marks (e.g., `Álvaro` → `alvaro`)
3. **Lowercase conversion**
4. **Whitespace trimming**

This ensures matching is accent-insensitive and case-insensitive.

---

## X. Output and Export

### CSV Export

- One row per match pair
- Headers include Table1 and Table2 fields, dynamic extra fields from Table 2, match flag, confidence score, and matched fields list
- Fuzzy algorithms respect the minimum confidence threshold (configurable, default 60%)

### XLSX Export

- Same data as CSV in a worksheet
- Additional summary worksheet with:
  - Database and table metadata
  - Record counts (Table 1, Table 2, matches per algorithm)
  - Timing breakdown (fetch, match, export durations)
  - Memory and GPU usage statistics
  - Execution mode and algorithm used

### Summary Report

A text summary (`matches_summary.txt`) is generated with run statistics.

---

## XI. Performance Tuning

See `docs/performance.md` for the full guide. Key recommendations:

1. **Always use release builds** (`cargo build --release`)
2. **Enable streaming** for large datasets (`NAME_MATCHER_STREAMING=1`)
3. **Enable auto-optimize** (`NAME_MATCHER_AUTO_OPTIMIZE=1`)
4. **Tune pool size** based on MySQL latency (`NAME_MATCHER_POOL_SIZE=12`)
5. **Cap Rayon threads** to avoid oversubscription (`RAYON_NUM_THREADS=12`)
6. **GPU:** Enable buffer pool and pinned host for throughput
7. **Windows TDR:** Use streaming mode to avoid long GPU kernel timeouts

### Safe Baseline (PowerShell)

```powershell
$env:NAME_MATCHER_STREAMING="1"
$env:NAME_MATCHER_AUTO_OPTIMIZE="1"
$env:RAYON_NUM_THREADS="12"
$env:NAME_MATCHER_POOL_SIZE="12"
$env:NAME_MATCHER_POOL_MIN="4"
$env:NAME_MATCHER_USE_GPU="1"
$env:NAME_MATCHER_GPU_STREAMS="2"
$env:NAME_MATCHER_GPU_BUFFER_POOL="1"
$env:NAME_MATCHER_GPU_PINNED_HOST="1"
```

---

## XII. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                         │
│  ┌──────────────┐              ┌──────────────────────┐ │
│  │   CLI        │              │   GUI (egui/eframe)  │ │
│  │  (main.rs)   │              │   (bin/gui.rs)       │ │
│  └──────┬───────┘              └──────────┬───────────┘ │
└─────────┼──────────────────────────────────┼────────────┘
          │                                  │
          ▼                                  ▼
┌─────────────────────────────────────────────────────────┐
│                   Core Library (lib.rs)                   │
│                                                          │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────┐  │
│  │  matching/  │  │    db/     │  │    export/       │  │
│  │ • mod.rs   │  │ • pool     │  │ • csv_export     │  │
│  │ • cascade  │  │ • schema   │  │ • xlsx_export    │  │
│  │ • advanced │  │ • fetch    │  │                  │  │
│  │ • gpu/     │  │            │  │                  │  │
│  └────────────┘  └────────────┘  └──────────────────┘  │
│                                                          │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────┐  │
│  │ normalize  │  │   config   │  │  optimization/   │  │
│  │ (Unicode)  │  │ (AppConfig)│  │ (auto-tuning)    │  │
│  └────────────┘  └────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────┐     ┌─────────────────────┐
│   MySQL Database    │     │   NVIDIA GPU (CUDA)  │
│   (sqlx async)      │     │   (cudarc, optional) │
└─────────────────────┘     └─────────────────────┘
```

---

## XIII. Project Structure

```
name_match_latest/
├── src/
│   ├── main.rs              # CLI entry point
│   ├── lib.rs               # Library root (re-exports all modules)
│   ├── bin/
│   │   ├── gui.rs           # Desktop GUI (egui/eframe)
│   │   ├── gpu_audit.rs     # GPU parity audit tool
│   │   └── seed.rs          # Database seeding utility
│   ├── cli/
│   │   ├── args.rs          # Legacy positional arg parsing
│   │   ├── clap_parser.rs   # Modern clap-based parsing
│   │   ├── flags.rs         # GPU/streaming flag parsing
│   │   └── mod.rs
│   ├── matching/
│   │   ├── mod.rs           # Algorithms 1-7, fuzzy scoring
│   │   ├── cascade.rs       # Cascade L1-L11 workflow
│   │   ├── advanced_matcher.rs  # Geographic-level matching
│   │   ├── birthdate_matcher.rs # Birthdate comparison logic
│   │   ├── helpers.rs       # Scoring utilities
│   │   ├── gpu_config.rs    # GPU configuration
│   │   └── gpu/             # CUDA batch processing
│   ├── db/
│   │   ├── connection.rs    # Pool creation
│   │   ├── schema.rs        # Table discovery, row fetching
│   │   └── mod.rs
│   ├── export/
│   │   ├── csv_export.rs    # CSV writer (streaming + batch)
│   │   ├── xlsx_export.rs   # XLSX writer with summary sheet
│   │   └── mod.rs
│   ├── config.rs            # All configuration structs
│   ├── models.rs            # Person, TableColumns, ColumnMapping
│   ├── normalize.rs         # Unicode normalization
│   ├── metrics.rs           # Memory stats
│   ├── error.rs             # Error types
│   ├── optimization/        # Hardware profiling
│   ├── orchestrator/        # Run orchestration
│   ├── engine/              # New streaming engine (feature-gated)
│   └── util/                # Checkpoint, env parsing, partitioning
├── scripts/windows/
│   ├── Build-Release-Gpu.ps1
│   ├── Build-Release-Gui.ps1
│   └── Run-NameMatcher.ps1
├── docs/
│   ├── performance.md
│   └── self_hosted_runner_windows_cuda.md
├── Cargo.toml
└── README.md
```

---

## XIV. Troubleshooting

| Issue | Solution |
|-------|----------|
| "Table missing required columns" | Ensure table has `id`, `uuid`, `first_name`, `last_name`, `birthdate` |
| GPU not detected | Verify NVIDIA driver installed, binary built with `--features gpu` |
| TDR timeout on Windows | Enable streaming mode (`NAME_MATCHER_STREAMING=1`), reduce GPU streams |
| Connection timeout | Check MySQL host/port, increase `NAME_MATCHER_ACQUIRE_MS` |
| Out of memory | Use streaming mode, reduce batch size, or increase system RAM |
| Cascade skips L4-L9 | Tables lack `barangay_code` / `city_code` columns (expected with AutoSkip) |

---

*Generated: 2026-05-19 | Tool: codebase-docs-with-screenshots (no screenshots — no running UI)*
