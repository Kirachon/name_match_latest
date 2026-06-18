# Adaptive Ultra Benchmark Evidence

Generated: 2026-06-18

## Scope

This evidence covers the production Current matcher GPU path only. StringZilla CUDA is intentionally excluded from these pass/fail gates and remains experimental/shadow-only.

## Commands

Build release benchmark binary:

```powershell
cargo build --locked --release --bin gpu_string_bench --features "gpu gpu-bench"
```

Parity sweep:

```powershell
target/release/gpu_string_bench.exe --dataset small --backend gpu --warmup-runs 0 --measured-runs 1 --current-only --json-only
target/release/gpu_string_bench.exe --dataset messy --backend gpu --warmup-runs 0 --measured-runs 1 --current-only --json-only
target/release/gpu_string_bench.exe --dataset medium --backend gpu --warmup-runs 0 --measured-runs 1 --current-only --json-only
target/release/gpu_string_bench.exe --dataset duplicate-heavy --backend gpu --warmup-runs 0 --measured-runs 1 --current-only --json-only
target/release/gpu_string_bench.exe --dataset large --backend gpu --warmup-runs 0 --measured-runs 1 --current-only --json-only
```

Repeated CPU/GPU comparison:

```powershell
target/release/gpu_string_bench.exe --dataset medium --backend cpu --warmup-runs 2 --measured-runs 10 --current-only --json-only
target/release/gpu_string_bench.exe --dataset medium --backend gpu --warmup-runs 2 --measured-runs 10 --current-only --json-only
target/release/gpu_string_bench.exe --dataset duplicate-heavy --backend cpu --warmup-runs 2 --measured-runs 10 --current-only --json-only
target/release/gpu_string_bench.exe --dataset duplicate-heavy --backend gpu --warmup-runs 2 --measured-runs 10 --current-only --json-only
target/release/gpu_string_bench.exe --dataset large --backend cpu --warmup-runs 2 --measured-runs 5 --current-only --json-only
target/release/gpu_string_bench.exe --dataset large --backend gpu --warmup-runs 2 --measured-runs 5 --current-only --json-only
```

JSON outputs are stored in `docs/adaptive-ultra-benchmark-json/`.

## Parity Results

| Dataset | JSON | Matches | GPU p50 us | False negatives | False positives | Pair IDs match | Canonical hash match | Ordered hash match | Blocking failure |
|---|---|---:|---:|---:|---:|---|---|---|---|
| small | `parity-small-gpu.json` | 21 | 7,735 | 0 | 0 | true | true | true | false |
| messy | `parity-messy-gpu.json` | 4 | 8,180 | 0 | 0 | true | true | true | false |
| medium | `parity-medium-gpu.json` | 164 | 17,958 | 0 | 0 | true | true | true | false |
| duplicate-heavy | `parity-duplicate-heavy-gpu.json` | 600 | 19,323 | 0 | 0 | true | true | false | false |
| large | `parity-large-gpu.json` | 1,735 | 103,356 | 0 | 0 | true | true | false | false |

Ordered hash drift on duplicate-heavy and large is non-blocking because the canonical hash matches and the benchmark reports it as order-only drift.

## Performance Results

| Dataset | CPU p50 us | GPU p50 us | CPU p95 us | GPU p95 us | CPU/GPU p50 ratio | Decision |
|---|---:|---:|---:|---:|---:|---|
| medium | 5,264 | 15,038 | 5,762 | 21,043 | 0.35x | CPU faster |
| duplicate-heavy | 13,480 | 19,579 | 16,720 | 22,466 | 0.69x | CPU faster |
| large | 52,807 | 65,796 | 55,234 | 66,536 | 0.80x | CPU faster |

## Decision

The production GPU path has clean parity, but release-mode CPU is faster for the tested fuzzy workloads on this machine. Adaptive Ultra should therefore avoid GPU for these workload sizes in Auto mode and use CPU unless the user explicitly selects Force GPU.

The implementation uses a conservative Auto threshold of 50,000 rows per side before choosing GPU for eligible fuzzy/cascade workloads. This protects users from slower GPU runs while preserving Force GPU for manual testing and future larger workloads.

## Validation Summary

- `cargo check --locked --features gpu`: passed.
- `cargo test --locked --features gpu run_service::tests -- --nocapture`: passed, 5 tests.
- `pnpm --dir ui test`: passed, 49 tests.
- Production Current GPU parity: passed on small, messy, medium, duplicate-heavy, and large.
- Production StringZilla CUDA: not included in this gate.
