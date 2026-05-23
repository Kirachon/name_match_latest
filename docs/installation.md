# Installation Guide

## Option A: Pre-built Release (Recommended)

1. Go to the GitHub Releases page for this repository.
2. Download the appropriate archive:
   - `gui-windows-cpu-<tag>.zip` — Windows GUI (CPU only)
   - `gui-linux-cpu-<tag>.tar.gz` — Linux GUI (CPU only)
3. Extract the archive to your desired location.
4. Run the `gui` executable directly (no installation required).

## Option B: Build from Source

### 1. Install Rust

```bash
# Windows: download from https://rustup.rs
# Linux/macOS:
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 2. Clone the Repository

```bash
git clone <repository-url>
cd name_match_latest
```

### 3. Build CPU-Only Release

```bash
cargo build --release
```

The CLI binary is at `target/release/name_matcher(.exe)`.

### 4. Build with GPU Support

Requires CUDA Toolkit (nvcc) on PATH:

```bash
cargo build --release --features gpu
```

### 5. Build GUI

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

## Option C: Docker / CI

CPU builds run on GitHub-hosted runners. GPU builds require a self-hosted Windows runner with CUDA. See `self_hosted_runner_windows_cuda.md`.

## Database Setup

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

For matching across two separate databases:
```
DB2_HOST=remote-host
DB2_PORT=3306
DB2_USER=reader
DB2_PASS=secret
DB2_DATABASE=other_db
```
