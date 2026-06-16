# GPU Fuzzy Matching Speedup Plan

## Summary

Finalize the GPU fuzzy path as a staged, parity-first speedup: GPU may only
reject obviously impossible candidates after shadow validation; CPU remains the
final classifier and score authority.

The plan incorporates the subagent review feedback: add instrumentation before
optimization, fix real L10/L11 parity bugs first, use shadow mode before gate
authority, avoid unmeasured speed claims, and make GPU CI tests impossible to
pass through CPU fallback.

## Key Changes

### Stage 0: Baseline and instrumentation

- Add GPU fuzzy counters before optimization: `candidate_pairs_seen`,
  `pairs_uploaded`, `bd_pass`, `bd_fail`, `middle_precheck_skips`,
  `empty_name_skips`, `direct_shortcuts`, `gpu_gate_keep`, `gpu_gate_reject`,
  `cpu_classified`, `matches_emitted`, `false_negative_shadow_count`.
- Add timing for candidate generation, H2D bytes/time, kernel time, D2H
  bytes/time, CPU classification, sorting, and wall-clock p50/p95 over warm
  repeated runs.
- No performance claim is accepted unless exact CPU/GPU output parity is also
  reported.

### Stage 1: Parity fixes

- In `src/matching/advanced_matcher.rs`, force L11 GPU dispatch to use
  `allow_birthdate_swap: false`; L10 keeps `cfg.allow_birthdate_swap`.
- In `src/matching/gpu/batch.rs`, use `simple_full_no_mid` and
  `dmeta_code_no_mid` whenever no-middle mode is active.
- Correct the contradictory L11 swap test so L10 with swap may match, but L11
  swapped birthdates must not match.

### Stage 2: Low-risk waste removal

- Remove the final rescore loop in `match_fuzzy_gpu` only after Stage 1 parity
  tests pass.
- Hoist L10 middle-name precheck, empty-name skip, and direct-match shortcut
  before GPU upload.
- Direct-match shortcuts must preserve birthdate/swap rules, dedupe behavior,
  confidence scale, `matched_fields`, and canonical final sort.

### Stage 3: Shadow GPU gate

- Add `fuzzy_gate_kernel` and `func_gate` to `GpuFuzzyContext`.
- Refactor `GpuBatchAccumulator::flush_to_gpu` to receive the GPU fuzzy context
  instead of adding more loose kernel/function arguments.
- Add `NAME_MATCHER_GPU_FUZZY_GATE_SHADOW=1`: launch the gate and read
  `keep_flags`, but still CPU-classify every candidate. Fail/log if any CPU
  match has `keep=0`.
- Keep `NAME_MATCHER_GPU_FUZZY_READBACK=1` as legacy metric-array audit only.

### Stage 4: Gate authority behind env flag

- Add `NAME_MATCHER_GPU_FUZZY_GATE_ONLY=1`: CPU classifies only pairs where
  `keep_flags[k] == 1`.
- Default remains off until shadow canaries show
  `false_negative_shadow_count == 0`.
- Conservative gate rule stays `84.0` for GPU f32 margin while CPU final
  threshold remains `>= 85.0`.

### Stage 5: Default-on canary

- Flip gate-only default only in a separate rollout after repeated clean shadow
  and opt-in runs.
- Keep legacy kernels available for audit/regression mode.

### Stage 6: GPU-resident name tables — **done**

- `ResidentNamePool` in `src/matching/gpu/resident.rs` uploads both table sides once
  per L10/L11 fuzzy run; batches send only pair indices to
  `fuzzy_gate_kernel_resident`.
- Legacy per-pair string staging remains when `NAME_MATCHER_GPU_RESIDENT_TABLES=0`,
  gate mode is Off, or resident upload exceeds 30% of VRAM budget / OOMs.
- Default: resident tables **on** (opt out with `NAME_MATCHER_GPU_RESIDENT_TABLES=0`);
  fuzzy gate default **Off** until explicit Shadow/GateOnly rollout. Batch VRAM reserve
  applies only after a successful resident upload (not on estimate alone).
- Parity gates: shadow `false_negative_count == 0`, candidate-set equality vs
  batch-copy path, CI job `gpu_resident_parity` on self-hosted CUDA runner.

## Tests And Gates

- Add CPU-only semantic tests for L10/L11 swap policy, middle-name rules, empty
  and whitespace-only names, Unicode/diacritics, missing birthdates, and current
  CPU behavior for `Cristina Santos` vs `Kristina Santos`.
- Add `#[cfg(feature = "gpu")]` hardware parity tests that run on a CUDA runner
  and do not silently fall back to CPU.
- Compare full canonical outputs: pair IDs, confidence tolerance `<= 1e-5`,
  `matched_fields`, labels, and final sorted order after canonical sort.
- Add fixed-seed canaries for dense, sparse, duplicate-heavy,
  missing-middle-heavy, Unicode/empty-name-heavy, and
  swapped-birthdate-heavy data.
- CI GPU job must run `cargo test --locked --features gpu parity` on the
  self-hosted CUDA runner; non-CUDA CI should not run CUDA-required tests.

## Assumptions

- No blocking-policy change is part of this plan; if CPU currently rejects a
  pair, GPU must match that behavior.
- Stage B/C/D speedups are hypotheses until counters prove them.
- `NAME_MATCHER_GPU_FUZZY_GATE_ONLY` must not become default-on until shadow
  mode shows zero false negatives and exact output parity.
- Implementation remains local to fuzzy matching, GPU batch/runtime plumbing,
  CI, and docs.
