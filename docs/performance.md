# Performance Guide (No Algorithm Changes)

This document focuses on **performance improvements that do not change matching results**, i.e.:
build configuration, runtime configuration, GPU/Windows tuning, and MySQL/infra tuning.

## 1) Build for speed

Always use a release build:

```bash
cargo build --release --features gpu
```

This repo enables additional release profile optimizations in `Cargo.toml` (`lto="thin"`, `codegen-units=1`, `strip="symbols"`).

### Optional: Profile-Guided Optimization (PGO)

PGO can produce large wins for CPU-heavy Rust binaries.

High-level workflow:
1) Build with profile generation enabled
2) Run representative workloads to generate profile data
3) Rebuild using the collected profile

(Exact commands differ by platform/toolchain; keep the workload realistic.)

## 2) Runtime knobs (safe defaults)

### MySQL pool (remote-safe)

These env vars are read in `src/db/connection.rs`:
- `NAME_MATCHER_POOL_SIZE` (max connections)
- `NAME_MATCHER_POOL_MIN`
- `NAME_MATCHER_ACQUIRE_MS`
- `NAME_MATCHER_IDLE_MS`
- `NAME_MATCHER_LIFETIME_MS`

Recommended starting point for remote MySQL:
- `NAME_MATCHER_POOL_SIZE=12`
- `NAME_MATCHER_POOL_MIN=4`

If the app is frequently waiting on DB I/O (CPU low, DB “acquire” waits), try `POOL_SIZE=16`.

### CPU parallelism (avoid oversubscription)

If you are using CPU parallelism heavily (and also doing DB + GPU work), it can help to cap Rayon:
- `RAYON_NUM_THREADS=12` (safe baseline for i5-13500HX)

If CPU is consistently underutilized, try `16`.

### Streaming mode (recommended on Windows + GPU)

Streaming reduces memory pressure and helps avoid long GPU kernels.

Enable:
- `NAME_MATCHER_STREAMING=1`

### Auto-Optimize (safe to enable)

The app can auto-tune some knobs based on detected hardware:
- `NAME_MATCHER_AUTO_OPTIMIZE=1`

## 3) GPU knobs (Windows laptop-friendly)

Enable GPU usage:
- `NAME_MATCHER_USE_GPU=1`

Throughput knobs supported by the app:
- `NAME_MATCHER_GPU_STREAMS=2` (try `3` if stable and VRAM allows)
- `NAME_MATCHER_GPU_BUFFER_POOL=1` (reuses buffers; usually improves throughput)
- `NAME_MATCHER_GPU_PINNED_HOST=1` (can help transfer overlap; uses pinned memory)

Feature toggles (enable only what you need):
- `NAME_MATCHER_GPU_HASH_JOIN=1`
- `NAME_MATCHER_GPU_FUZZY_DIRECT_HASH=1`
- `NAME_MATCHER_GPU_FUZZY_METRICS=1`

## 4) Windows stability (important for GPU)

On Windows, long-running GPU kernels can trigger TDR (driver timeout). If you see random GPU resets:
- Prefer `NAME_MATCHER_STREAMING=1`
- Keep `NAME_MATCHER_GPU_STREAMS` modest (`2` first)
- Only adjust TDR registry settings if this is a dedicated compute machine and you understand the tradeoffs

## 5) Remote MySQL: the biggest “hidden” bottleneck

If MySQL is remote, performance is often dominated by latency and server capacity:
- Run the app as close to MySQL as possible (same region/VPC/subnet when possible)
- Avoid VPN/public internet paths for high-throughput workloads
- Ensure MySQL has enough headroom (`max_connections`, CPU, IOPS, buffer pool sizing)

## 6) Safe baseline (PowerShell)

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
