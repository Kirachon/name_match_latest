# GPU Fuzzy Matching Speedup Plan Without Breaking CPU Parity

> **Revision note (2026-05-23):** This plan was rewritten against the current
> codebase. File and symbol references below point at code that exists today.
> The original plan's direction is preserved; what changed is the concrete
> targets, the staging order, and the bugs that need to land before anything
> faster ships.

## Goal

Make GPU fuzzy matching faster than CPU on dense workloads while keeping the
final match set identical to the CPU-only path.

The guiding rule remains:

> GPU may reduce work, but CPU must remain the official final judge.

GPU acts as a fast filter; the CPU final classifier still decides every match.

---

## Current Codebase State (what is already done, what is broken)

Before doing anything new, internalize what is already in the repo. Treat this
as the source of truth, not the previous plan.

### Already implemented (do not redo)
- Adaptive memory budget and block sizing — `src/matching/gpu_config.rs`
  (`calculate_gpu_memory_budget`, `calculate_optimal_block_size`,
  `calculate_tile_soft_cap`).
- OOM CPU fallback wrapper — `gpu_config::with_oom_cpu_fallback`.
- Dynamic VRAM tuner thread — `src/matching/gpu/dynamic_tuner.rs`.
- Blocking by `(birth_year, first_initial, last_initial, soundex(last))` —
  built inside `gpu::match_fuzzy_gpu` (`src/matching/mod.rs`,
  ≈line 5051+). The GPU path uses the **same** block index as the CPU path,
  so blocking-level parity is fine.
- Per-person `FuzzyCache` with both `simple_full` and `simple_full_no_mid`
  plus precomputed Double Metaphone codes — `src/matching/mod.rs`
  (`FuzzyCache`, `build_cache_from_person`, ≈line 3170+).
- Pinned-host staging buffers (`PinnedHostSlice`) reused across flushes —
  `src/matching/gpu/batch.rs` (`GpuBatchAccumulator`).
- Cross-outer batch accumulation up to `tile_max` pairs — same file.
- Authoritative CPU classification reused unchanged from CPU path —
  `classify_pair_cached` and `classify_pair_cached_no_mid` in
  `src/matching/mod.rs` (≈lines 3221, 3269). These are called from inside
  `flush_to_gpu`.
- Conditional GPU readback gated by `NAME_MATCHER_GPU_FUZZY_READBACK`
  (default off). Today the GPU output is computed but never read.

### Broken or wasteful in current code
1. **Wasted GPU work on every pair.** `GpuBatchAccumulator::flush_to_gpu` in
   `src/matching/gpu/batch.rs` runs `lev_kernel`, `jaro_kernel`, `jw_kernel`,
   `max3_kernel`, allocates `d_lev`/`d_j`/`d_w`/`d_final`, then **discards**
   the device output (default: no readback). Right after, the post-processing
   loop calls `classify_pair_cached*` for every surviving pair, which
   recomputes Levenshtein, Jaro-Winkler, and metaphone on the CPU. The GPU
   pass adds latency and contributes nothing to the result.
2. **Redundant final rescore loop.** `gpu::match_fuzzy_gpu`
   (`src/matching/mod.rs`, end of the function near line ≈5440) iterates
   over `results` and calls `classify_pair_cached*` again on every pair.
   The same call already ran inside `flush_to_gpu`. This is a third pass on
   the same metrics for every match.
3. **L11 birthdate-swap forwarding bug.**
   `advanced_match_inmemory_gpu` in `src/matching/advanced_matcher.rs`
   (≈line 615) forwards `cfg.allow_birthdate_swap` into `MatchOptions` for
   **both** L10 and L11. CPU L11 (`advanced_match_inmemory`, ≈line 362)
   hardcodes `let allow_swap = false`. If a caller enables swap globally,
   GPU L11 returns swapped-birthdate matches that CPU L11 will not.
4. **L11 GPU kernel input uses the wrong string.** `flush_to_gpu` always
   reads `cache1[i].simple_full` and `cache2[j].simple_full`, even when
   `super::gpu_no_mid_mode() == true`. The classifier branch later uses
   `simple_full_no_mid` correctly, but the GPU is scoring on the wrong
   string. Latent today (output discarded), real bug the moment the keep
   flag becomes authoritative.
5. **L10 middle-name precheck runs *after* GPU work.** The
   `l1 < 2 || l2 < 2 ⇒ continue` short-circuit lives inside the post-kernel
   loop in `flush_to_gpu`. Pairs with a middle initial only on either side
   are still uploaded, scored on GPU, and birthdate-checked before being
   thrown away.
6. **Per-flush re-encoding of the same names.** `flush_to_gpu` rebuilds
   `a_bytes`/`b_bytes` from `cache1[pair.outer_idx].simple_full` for every
   pair on every flush. With ~1.25% selectivity at 25k×25k, each person's
   ~25-byte normalized name is copied to the H2D buffer ~1500× per run.
   Pinned host memory speeds this up; it does not eliminate the waste.

The original plan's diagnosis matches items 1, 4, 5, and 6 directly. Items 2
and 3 are additional concrete bugs uncovered while tracing the code.

---

## Strategy: Lossless GPU Gate + CPU Final Classifier

```text
CPU normalize/cache  ->  CPU block + safe prechecks  ->  GPU keep/reject gate
                                                                  |
                                                                  v
                                              CPU classify_pair_cached* on survivors
                                                                  |
                                                                  v
                                                          Sort by IDs, return
```

The GPU never decides a match. It returns one byte per pair: keep or reject.
CPU classification on the kept set produces the canonical score and label.

This is the same direction the original plan took, but tied to the symbols
that already exist (`classify_pair_cached`, `classify_pair_cached_no_mid`,
`FuzzyCache`).

### Lossless keep rule (conservative, f32-safe)

```rust
// Pseudocode for the fused gate kernel.
// All inputs already normalized on CPU; metaphone equality precomputed and
// uploaded as a u8 mask alongside the index pairs.
let lev = lev_pct(a, b);          // 0..=100 in f32
let jw  = jw_pct(a, b);           // 0..=100 in f32
let mp  = metaphone_eq_mask[i];   // 0 or 1

let keep = if mp == 1 {
    lev >= 84.0 || jw >= 84.0
} else {
    lev >= 84.0 && jw >= 84.0
};
```

`84.0` is the safety margin against f32 vs f64 rounding; the CPU final
threshold stays at the existing `>= 85.0` rule inside `classify_pair_cached*`.

---

## Stage A — Parity hardening (no perf change, must land first)

**Risk: low. No kernel or buffer-layout change.**

A1. **Force L11 birthdate-swap to false in the GPU dispatch.**
- Edit: `src/matching/advanced_matcher.rs`, function
  `advanced_match_inmemory_gpu` (≈line 590).
- Where the function builds `MatchOptions` (`opts`), set
  `allow_birthdate_swap: false` for `AdvLevel::L11FuzzyBirthdateNoMiddle`.
  Keep `cfg.allow_birthdate_swap` for L10.
- Do **not** push the override down into `match_fuzzy_gpu`; keep
  `match_fuzzy_gpu` honest about the flag it received.

A2. **Use `simple_full_no_mid` in the GPU kernel input when in no-mid mode.**
- Edit: `src/matching/gpu/batch.rs`, `GpuBatchAccumulator::flush_to_gpu`,
  both the pinned-host fill loop and the pageable fallback fill loop.
- Replace each `&cache1[pair.outer_idx].simple_full` with
  `if super::gpu_no_mid_mode() { &cache1[pair.outer_idx].simple_full_no_mid } else { &cache1[pair.outer_idx].simple_full }`.
  Same for `cache2`.
- This must land in the same change as A1 so L11 GPU is consistent
  end-to-end.

A3. **Add a CPU vs GPU parity test matrix** (compile-time gated on
    `feature = "gpu"`):
- Location: `src/matching/mod.rs` test module or a new `tests/` file.
- Cases: identical names, single-char typo, transposed order, missing middle,
  full middle, L11 no-middle, swapped birthdate, borderline score (CASE 2 at
  ~85), `Cristina Santos` vs `Kristina Santos`, ASCII-only and accented
  variants, empty strings on each side.
- Assertions: same number of matches, same `(person1.id, person2.id)` set,
  same confidence within `1e-5`, same `matched_fields`, same sort order.

**Validation:**
- `cargo check --locked --features gpu`
- `cargo test --locked --features gpu parity`
- Re-run the existing audit binary at 5k and 25k rows; expect identical pair
  counts to the experimental plan's recorded baselines (`1,814` at 5k,
  `44,710` at 25k from the prior canary).

---

## Stage B — Remove wasted work (no kernel change, single-digit % win)

**Risk: low. Pure deletion / reordering.**

B1. **Delete the redundant final rescore loop in `match_fuzzy_gpu`.**
- Edit: `src/matching/mod.rs`, end of `gpu::match_fuzzy_gpu`
  (≈line 5440-5462). The `for pair in &mut results { let rescored = ... }`
  block is dead work — `flush_to_gpu` already wrote the canonical confidence
  via the same `classify_pair_cached*` calls.
- Verify by running the Stage A parity tests; output must be bit-identical.

B2. **Move L10 middle-name precheck before GPU upload.**
- Edit: `src/matching/mod.rs`, inside `match_fuzzy_gpu` candidate loop where
  pairs are pushed via `batch_pairs.push((i, j_idx));` (≈line 5260+).
- Before pushing, when `!gpu_no_mid_mode()`, compute `l1`/`l2` from
  `t1[i].middle_name` / `t2[j_idx].middle_name` using the same
  `trim_matches('.')` + non-whitespace char count rule already in
  `flush_to_gpu`. If `l1 < 2 || l2 < 2`, `continue` without pushing.
- Remove the corresponding post-kernel `continue` from `flush_to_gpu` so the
  rule lives in exactly one place.

B3. **Move empty-name and direct-equality shortcuts before GPU upload.**
- Same loop as B2. If the chosen string (`simple_full` or
  `simple_full_no_mid`) is empty on either side, skip. If both strings are
  byte-equal, build a `MatchPair` with `confidence = 100.0` and
  `label = "DIRECT MATCH"` directly (matches `classify_pair_cached*`'s
  short-circuit) and skip GPU upload.

**Validation:**
- Re-run Stage A parity tests; output must remain identical.
- Audit run at 25k should show fewer total candidates pushed to GPU on L10
  (anywhere middle-initial-only pairs exist) — log
  `total_candidates` already exists at the end of `match_fuzzy_gpu`.

---

## Stage C — Fused lossless keep/reject kernel (biggest CPU win)

**Risk: medium. New kernel, new readback path. All previous parity tests
must still pass.**

C1. **Add `fuzzy_gate_kernel` to the fuzzy GPU module.**
- Edit: `src/matching/mod.rs` `LEV_KERNEL_SRC` block (≈line 3000+ inside
  `mod gpu`).
- New device function that, per pair, computes Levenshtein similarity and
  Jaro-Winkler similarity in registers (reuse the existing DP and Jaro
  routines), takes a `const unsigned char* mp_eq` mask, and writes a single
  `uint8_t` to `out`:
  ```c
  uint8_t keep = mp_eq[i]
      ? (lev >= 84.0f) || (jw >= 84.0f)
      : (lev >= 84.0f) && (jw >= 84.0f);
  out[i] = keep ? 1u : 0u;
  ```
- Register and load the new function into `GpuFuzzyContext` alongside
  `func_lev` etc. Keep the existing kernels compiled — the audit binary
  uses them, and they're cheap to keep around.

C2. **Upload metaphone equality mask alongside the pair batch.**
- Edit: `src/matching/gpu/batch.rs`. In `flush_to_gpu`, build a
  `Vec<u8>` of length `n_pairs` where `mask[k] = 1` iff
  `cache1[outer].dmeta_code` (or `dmeta_code_no_mid` in no-mid mode) is
  non-empty and equal to the corresponding cache2 code. Reuse a pinned slot
  the same way `pinned_a_offsets` is reused.

C3. **Switch the hot path to gate-only.**
- New `gpu_fuzzy_gate_only` env var (default on) controls this.
- When on: launch only `fuzzy_gate_kernel`. Read back `keep_flags: Vec<u8>`.
  Run `classify_pair_cached*` only on pairs where `keep_flags[k] == 1`.
- When off (audit / regression mode): keep current behavior, run all four
  kernels, optionally read back metric arrays via the existing
  `NAME_MATCHER_GPU_FUZZY_READBACK` switch.

C4. **Strict parity gate, not just spot tests.**
- Add a test that compares CPU-only output against gate-only output on a
  fixed seeded dataset of ~50k rows containing the borderline cases from
  Stage A. The test fails if **any** keep=0 pair would have produced a CPU
  match. Only after this test is green for two consecutive CI runs is the
  margin allowed to tighten beyond `84.0`.

**Validation:**
- `cargo test --locked --features gpu` (full suite + new strict test).
- 25k canary: pair count must equal the Stage A baseline. Wall time should
  drop materially because `classify_pair_cached*` now runs on roughly the
  small fraction of pairs the gate keeps, not on all candidates.

---

## Stage D — GPU-resident name table (biggest H2D win)

**Risk: higher. Kernel signature change. Stages A-C must already be in
production with parity tests green.**

D1. **Build per-table name buffers once per `match_fuzzy_gpu` call.**
- Edit: `src/matching/mod.rs` `match_fuzzy_gpu` (and the no-mid wrapper).
- After `cache1` / `cache2` are built, materialize four `Vec<u8>` /
  `Vec<i32>` blocks (`buf_a`, `off_a`, `len_a`, `buf_b`, `off_b`, `len_b`)
  using `simple_full` or `simple_full_no_mid` as appropriate.
- Upload once with `stream.memcpy_stod(...)` and keep the `CudaSlice`
  handles for the lifetime of the call.

D2. **Switch `flush_to_gpu` to upload only index pairs.**
- Replace the per-pair byte-blob construction with two `Vec<u32>`:
  `idx_a[k] = pair.outer_idx as u32`, `idx_b[k] = pair.inner_idx as u32`.
  Keep these in pinned host memory exactly like the offsets/lengths today.
- Update `fuzzy_gate_kernel` (and the legacy lev/jaro/jw kernels) to take
  `(name_buf, name_off, name_len, idx_a, idx_b, n)` and look up
  `A = name_buf_a + off_a[idx_a[i]]` per thread.

D3. **Drop the per-pair byte buffers and their pinned slots.**
- `pinned_a_bytes`, `pinned_b_bytes`, `a_bytes`, `b_bytes` and their
  offset/length pinned slots are no longer needed. Replace with the index
  pinned slots only.

**Validation:**
- All existing parity tests must still pass.
- 25k canary H2D byte volume (instrumentable via cudarc query or
  `NAME_MATCHER_GPU_BATCH_LOG`) must drop by roughly a factor of average
  candidates per person, ~1500× on the dense dataset.

---

## Stage E — Parity matrix + rollout discipline

**Risk: documentation + CI only.**

E1. **Promote the parity tests to a CI matrix.**
- Run all combinations: `{stage A, A+B, A+B+C, A+B+C+D}` × `{L10, L11}` ×
  `{swap=false, swap=true}` on a fixed seed.
- Block merges on any divergence from the Stage A baseline output.

E2. **Tighten the keep margin only after the strict gate runs clean.**
- Move from `84.0` to `84.5` to `85.0` in separate PRs; each step requires a
  green run of C4's strict test.

E3. **Document the operator-facing rules.**
- Update `docs/usage_guide.md` and the existing experimental plan log
  (`experimental-gpu-assisted-matching-plan.md`) with the new gate env vars
  and the parity tests they imply.

---

## Lossless gate logic, in detail

The CPU rule encoded by `classify_pair_cached` and
`classify_pair_cached_no_mid` (`src/matching/mod.rs`) is:

```text
DIRECT MATCH  if simple_full == simple_full
CASE 1        if lev >= 85 && jw >= 85 && metaphone == 100  -> avg of three
CASE 2        if at least 2 of {lev >= 85, jw >= 85, metaphone == 100} -> avg
CASE 3        CASE 2 + avg >= 88 + per-component levenshtein <= 2
NO MATCH      otherwise
```

The conservative GPU keep rule cannot drop any pair the CPU rule could keep:

```text
metaphone matches  ->  keep if lev >= 84 || jw >= 84
otherwise          ->  keep if lev >= 84 && jw >= 84
```

Why this is safe:
- CASE 1 needs all three at >= 85, so both lev and jw are >= 85 > 84. Kept.
- CASE 2 with metaphone match needs at least one of lev/jw >= 85 > 84.
  Matches the OR branch.
- CASE 2 without metaphone match needs both lev >= 85 and jw >= 85.
  Matches the AND branch.
- CASE 3 is a refinement of CASE 2, same coverage.
- The 1-point margin absorbs f32 rounding around the boundary.

Empty-string and direct-equality shortcuts are CPU-only (Stage B3), so they
never enter the kernel.

---

## What not to do (parity hazards)

- **Do not add GPU-only blockers** beyond the existing
  `(year, fi, li, soundex(last))` block. The `Cristina vs Kristina Santos`
  case in the parity matrix exists precisely to catch first-initial blocking
  drift.
- **Do not let GPU produce the final score.** A previous experimental
  full-GPU scorer returned `0` pairs at 5k where CPU returned `1,814`
  (`experimental-gpu-assisted-matching-plan.md`). The keep/reject
  contract exists to prevent that recurrence.
- **Do not use top-K as the gate output.** Borderline matches the CPU keeps
  may sit outside top-K under f32 ordering.
- **Do not raise the GPU threshold above 85** without re-running the strict
  parity gate (Stage C4) and the per-component CASE 3 test specifically.
- **Do not auto-enable Stage D before Stage C is in production for at least
  one full canary cycle.** D's kernel signature change is the riskiest
  surface in this plan.

---

## Required parity tests (all stages)

Add as `#[cfg(feature = "gpu")] #[test]` cases that compare
`advanced_match_inmemory(...)` against `advanced_match_inmemory_gpu(...)`
on the same inputs, asserting:

```text
- same number of matches
- same set of (person1.id, person2.id) pairs
- same matched_fields per pair
- same confidence within 1e-5
- same sorted output order after cascade::sort_matches_by_id
```

Test fixture cases:
1. Exact same names, exact same birthdate.
2. Single-character typo in last name.
3. Transposed first/last name characters.
4. Middle initial on one side, full middle on the other (L10 must drop;
   L11 must keep).
5. Same name, swapped month/day in birthdate (with `allow_swap=true` in the
   advanced config: L10 must keep, L11 must drop — this catches the
   forwarding bug from A1).
6. Borderline scores around 85: hand-crafted pairs that produce
   `lev=84.9 / jw=85.2` and `lev=85.1 / jw=84.8` on f64 to stress the
   margin.
7. `Cristina Santos` vs `Kristina Santos` (different first initial, same
   sound). Must match in both paths.
8. Names that contain Unicode beyond ASCII (e.g. `José`). Must match in both
   paths and not panic.
9. Empty `first_name` or `last_name` on one side. Must produce no match in
   both paths.
10. Identical `simple_full` on both sides (DIRECT MATCH shortcut).
11. Pair that passes lev/jw but fails metaphone (CASE 2 AND-branch).
12. Pair that passes metaphone + lev only (CASE 2 OR-branch).

Additional canary thresholds (from existing audit baselines, do not
regress):

```text
5k rows  : 1,814 pairs, GPU prefilter under 1.0s on RTX 4050.
10k rows : 7,235 pairs, ~5x speedup over CPU.
25k rows : 44,710 pairs, ~6x speedup over CPU.
```

---

## Implementation order summary

| Stage | Scope                                                | Risk   | Expected win |
|-------|------------------------------------------------------|--------|--------------|
| A     | L11 swap fix, L11 string fix, parity test matrix     | Low    | None (safety) |
| B     | Delete redundant rescore, hoist prechecks            | Low    | Single-digit % |
| C     | Fused keep/reject kernel + CPU classify on survivors | Medium | Largest CPU win |
| D     | GPU-resident name table, index-pair upload           | Higher | Largest H2D win |
| E     | CI parity matrix, margin tightening, docs            | None   | Sustains the rest |

If only one stage ships in a release: ship Stage A. It removes the latent
parity defects so every later optimization can land safely.

If two stages ship: A then C. That is where the plan's headline benefit
(`Lossless GPU fuzzy gate + CPU final classifier`) becomes real.

---

## Final recommendation

The architecture the original plan proposed is correct. The work to do is
not "design a gate" — it is to:

1. Fix two real parity bugs (`allow_swap` forwarding, GPU input string in
   no-mid mode).
2. Stop running the same classifier three times per pair.
3. Stop sending names to the GPU only to throw the GPU output away.
4. When the gate is real and tested, stop sending the same name 1500 times.

Every change is local to `src/matching/advanced_matcher.rs`,
`src/matching/mod.rs`, and `src/matching/gpu/batch.rs`. The CPU path,
blocking, OOM fallback, dynamic tuner, and pinned-host plumbing stay as
they are.

> GPU mode should be faster, but CPU and GPU modes must still return the
> same final matching results.
