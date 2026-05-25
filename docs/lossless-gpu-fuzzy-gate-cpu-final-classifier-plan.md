# Lossless GPU Fuzzy Gate + CPU Final Classifier Plan

## Summary

Implement the L10/L11 speedup as a staged, parity-first GPU gate. The GPU will
only decide whether a candidate pair is safe to skip; the existing CPU
classifier remains the final source of truth for match score, label, and output.

Party-mode reviewer feedback is included: structured stats, a real gate mode
enum, no silent GPU fallback in required tests, and no default gate-only
behavior until shadow testing proves zero false negatives.

## Key Changes

- Add an internal `GpuFuzzyGateMode` with `Off`, `Shadow`, and `GateOnly`.
  Env/test control comes first; normal UI exposure comes only after shadow
  parity passes. Final UI safety switch can live in GPU settings as
  `Off / Shadow verify / Fast gate`.
- Add structured `GpuFuzzyStats` for each L10/L11 run:
  `candidate_pairs_seen`, `pairs_uploaded`, `gpu_gate_keep`,
  `gpu_gate_reject`, `cpu_classified`, `matches_emitted`,
  `shadow_false_negative_count`, `fallback_to_cpu_count`, plus phase timings.
- Refactor `GpuBatchAccumulator::flush_to_gpu` into clear stages:
  batch staging, GPU gate/metric launch, candidate selection, CPU final
  classification, result emission.
- Add `fuzzy_gate_kernel` and `keep_flags`.
  The gate uses a conservative threshold, starting at `84.0`, while CPU final
  classification keeps the existing `>= 85.0` rule.
- Preserve CPU authority:
  `classify_pair_cached` and `classify_pair_cached_no_mid` still produce final
  confidence, labels, and `matched_fields`.
- Lock L10/L11 rules:
  L10 may honor birthdate month/day swap. L11 must keep swap disabled to match
  CPU behavior. L11 must use `simple_full_no_mid` and `dmeta_code_no_mid`.
- Add shadow mode first:
  GPU produces keep/reject flags, but CPU still classifies every candidate. Any
  CPU match with `keep=0` is a false negative and blocks gate-only.
- Add gate-only mode second:
  CPU classifies only GPU-kept pairs. Production fallback may return to CPU, but
  required GPU tests must fail if fallback occurs.
- After repeated clean tests, make Fast Gate automatic for L10/L11 when GPU is
  enabled, while keeping an emergency off switch.

## Test Plan

- Add CPU semantic tests for L10/L11 middle-name rules, no-middle behavior,
  direct matches, empty names, duplicates, swapped birthdates,
  Unicode/diacritics, punctuation, long names, and near-threshold fuzzy scores.
- Add required CUDA tests that assert real GPU execution:
  CUDA context created, gate kernel launched, `keep_flags` read back,
  `pairs_uploaded > 0`, `gpu_gate_reject > 0`, and
  `fallback_to_cpu_count = 0`.
- Shadow acceptance:
  `shadow_false_negative_count = 0`, full CPU/GPU canonical output equality,
  and `gpu_gate_keep + gpu_gate_reject = candidate pairs after safe prechecks`.
- Gate-only acceptance:
  exact pair IDs, confidence tolerance, labels, matched fields, duplicate count,
  and final sorted order must match CPU baseline.
- Benchmark with release builds only:
  same machine, same GPU driver, same data, same row order, same settings, one
  cold run reported separately, then 5-10 warm runs with p50/p95/min/max.
- Speed claim gate:
  parity must pass, fallback count must be `0`, CPU-classified pairs should
  drop by at least `30%` on canaries, and median wall-clock should improve by at
  least `15%` with no p95 regression over `5%`.
- Rebuild and verify final release:
  Rust tests, GPU-feature tests, UI lint/tests/build, Tauri GPU release build,
  deterministic review, then commit/push.

## Assumptions And Defaults

- Default before proof: `Shadow` for diagnostics or `Off` for normal runs;
  never default `GateOnly` before zero-false-negative evidence.
- GPU-resident name tables are future work after this goal; they should not
  block the lossless gate delivery.
- Performance success means faster end-to-end L10/L11 matching, not just faster
  CUDA kernel timing.
- Any parity failure, false negative, or silent CPU fallback blocks making Fast
  Gate the default.
