# Post-Migration Enhancement Plan — Tauri v2 Name Matcher

**Plan ID:** `tauri-enhancements-2026-05-24`
**Status:** Council-reviewed, ready for implementation
**Reviewed by:** Backend Architect, Frontend Developer, UX Researcher (subagent council)
**Estimated Effort:** ~8 hours total
**Priority Order:** Feature 1 → 3 → 4 → 2

---

## Priority Order (Final)

```
Feature 1: CUDA Diagnostics Fix (~30 min)
Feature 3: Connection Persistence (~1 hr)
Feature 4: Pause/Resume (~2 hr)
Feature 2: Cascade Mode (~4–5 hr)
```

**Rationale:** All three reviewers converge on this order. Feature 3 has highest
daily friction (UX). Feature 4 must exist before Feature 2 ships (cascade runs
are long). Feature 2 is last due to integration complexity and design risk.

---

## Feature 1: CUDA Diagnostics Fix

### Problem

`src-tauri/src/commands/system.rs` `cuda_diagnostics()` returns hardcoded
`device_count: 0` even when compiled with `gpu` feature. The Status Rail shows
"No CUDA" on a machine with an RTX 4050.

### Final Approach

- In the `#[cfg(feature = "gpu")]` block of `cuda_diagnostics_inner()`, call
  `cudarc::driver::CudaContext::new(0)` to probe the device.
- On success: populate `device_count: 1`, push device name string, read VRAM
  via `cuMemGetInfo_v2`.
- On failure: set `error` field with the failure reason.
- `system_info()` already reads from `cuda_diagnostics_inner()` so
  `gpu_available` will automatically become `true`.
- First call may take 200–500ms (driver init) — acceptable for one-time
  startup probe.

### File Changes

| File | Change |
|------|--------|
| `src-tauri/src/commands/system.rs` | Real CUDA probe replacing placeholder |
| `ui/src/app/StatusRail.tsx` | Show device name when GPU available |

### Risk

Low — same pattern as the proven `cuda_probe.rs` binary.

---

## Feature 3: Connection Persistence

### Problem

Users must re-enter DB credentials every time they launch the app.

### Final Approach

- On successful connect, save credentials to `tauri-plugin-store`.
- Password storage: **off by default**. Checkbox label: "Save password locally
  (stored unencrypted on this machine)" with visible amber ⚠️ icon inline.
- Helper text below checkbox (also `aria-describedby`): "Password saved in
  plaintext in app data directory. Use only on trusted machines."
- Store keys: `connections.source` / `connections.target` — include
  `table_name` for full restore.
- Schema versioning: `{ version: 1 }` in persisted JSON for future
  encrypted-storage migration.
- On app launch, load saved connections and auto-populate the Connect tab forms.
- **Credential rotation:** If saved credentials fail on reconnect, clear saved
  password, pre-fill form with everything except password, show toast "Saved
  credentials expired — please re-enter password."
- **"Restored from last session" badge:** Show subtle timestamp on Connect tab
  ("Last connected: May 23, 2026") that fades after 5s.

### File Changes

| File | Change |
|------|--------|
| `ui/src/features/connect/ConnectTab.tsx` | Load/save logic + "Remember" checkbox with ⚠️ icon |
| `ui/src/shared/stores/connectionStore.ts` | Add `persistCredentials` / `loadPersistedCredentials` actions |
| `ui/src/shared/tauri/types.ts` | Add `PersistedConnectionDto` type with `version` field |

### Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| IT audit flags plaintext password | Document in release notes; fast-follow with OS keychain integration |
| Stale credentials cause retry loops | Clear on first failure; never auto-retry with saved password |
| Store file corruption | Wrap load in try/catch; fall back to empty state on parse error |

---

## Feature 4: Pause/Resume

### Problem

The plan specifies pause/resume states but they're not wired end-to-end.

### Final Approach

- Add `pause_flag: Arc<AtomicBool>` to `JobHandle` alongside the existing
  `cancel` token.
- In `run_worker`, check order: **cancel first, then pause.** Prevents deadlock
  where paused job ignores cancel.
- When paused: `thread::sleep(50ms)` loop until unset or cancelled.
- Emit `JobStateDto::Paused` from **inside the worker's pause loop** (not from
  the command handler). Ensures frontend only shows "Paused" after engine has
  actually stopped processing.
- Add `pause_matching` / `resume_matching` Tauri commands that flip the flag
  and emit `pausing`/`resuming` transitional states.

### Frontend State Machine

| Engine State | Button | Progress Bar | Detail Text |
|---|---|---|---|
| `running` | ⏸ Pause (enabled) | Animated blue | — |
| `pausing` | ⏸ Pausing… (disabled, spinner) | Animated amber pulse | "Pausing after current batch…" |
| `paused` | ▶ Resume + ✕ Cancel | Static amber | "Paused — 4,200 / 12,000 pairs (batch 7/20)" |
| `resuming` | ▶ Resuming… (disabled, spinner) | Animated blue | "Resuming…" |

### Keyboard Shortcut

`Ctrl+P` for pause/resume toggle.

### Cancel from Paused

Show confirmation dialog: "Cancel matching? N pairs already matched will be
kept in results."

### File Changes

| File | Change |
|------|--------|
| `src/run_service/mod.rs` | Add pause check in `run_worker` loop: cancel → pause → process |
| `src-tauri/src/commands/matching.rs` | Add `pause_matching` / `resume_matching` commands |
| `ui/src/features/run/RunTab.tsx` | 4-state button rendering; amber progress bar styles |
| `ui/src/app/shortcuts.tsx` | Add `Ctrl+P` handler for pause/resume toggle |
| `ui/src/shared/tauri/commands.ts` | Add `pauseMatching()` / `resumeMatching()` wrappers |

### Accessibility

- `aria-live="polite"` on job state indicator for screen reader announcements.
- Button label changes announced via live region.

### Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Large batches make pause feel unresponsive | Show record-level counter within current batch |
| State race: frontend shows "Paused" before engine stops | Emit state from worker loop, not command handler |
| Cancel ignored while paused | Check cancel before pause in loop |

---

## Feature 2: Cascade Mode (L1–L12 Level Picker)

### Problem

The legacy egui GUI exposes cascade/advanced matching (12 sequential levels) but
the Tauri UI only shows Options 1–7 (single-pass algorithms).

### Final Approach

#### User-Facing Naming

- **"Quick Match"** → existing 7-algorithm radio group (single-pass)
- **"Deep Match"** → cascade level picker (L1–L11)

Internal code stays "cascade." User sees intent-based labels, no jargon.

#### Mode Toggle (top of Configure tab)

```
┌─────────────────────────────────────────┐
│  Matching Mode                          │
│  ┌──────────────┐ ┌──────────────────┐  │
│  │ Quick Match  │ │  Deep Match      │  │
│  │ (single alg) │ │  (cascade)       │  │
│  └──────────────┘ └──────────────────┘  │
│                                         │
│  "Use Quick Match for exploratory runs. │
│   Use Deep Match for production."       │
└─────────────────────────────────────────┘
```

When "Deep Match" is selected, the single-pass algorithm radio group is hidden
and the cascade level picker is shown.

#### Cascade Level Picker (preset-first)

| Preset | Levels | Description |
|--------|--------|-------------|
| Standard | L1–L3 | Name-only matching |
| Extended | L1–L6 | Includes barangay codes |
| Full | L1–L11 | All available levels |
| Custom | User picks | Reveals checkbox grid |

- **Auto-select preset** based on detected columns: if `barangay_code` present
  → default "Extended"; if `city_code` also present → default "Full"; otherwise
  → "Standard".
- **Grouped checkboxes** (Custom mode): Name-based (L1–L3) | Barangay (L4–L6,
  lock icon if missing) | City (L7–L9, lock icon if missing) | Fuzzy (L10–L11,
  threshold slider inline).
- **Disabled levels show reason:** "Requires `barangay_code` in source table —
  not found."
- **Exclusion mode:** Radio pair — "Exclusive (recommended)" / "Independent
  (allows duplicates)".
- **Time estimate:** Show "~3× longer than Quick Match with 6 levels" before
  Run.
- **L12 excluded:** L12 (Household) runs separately via Options 5/6. Footnote
  in UI: "L12 (Household) runs separately via Quick Match Options 5–6."

#### Backend: RunService Adapter Layer

1. **Derive `GeoColumnStatus`** from `TableColumnsDto.raw_columns`.
2. **Map `CascadeOptionsDto` → `CascadeConfig`:**
   - Expose in DTO: `levels`, `fuzzy_threshold`, `exclusion_mode`
   - Hardcode: `missing_column_mode: AutoSkip`, `compute_backend: inherit from
     GpuOptionsDto`, `allow_birthdate_swap: inherit from MatchOptionsDto`,
     `base_output_path: temp_dir()`
3. **Suppress file export:** Pass `std::env::temp_dir()` as `base_output_path`.
   Tauri handles export separately. Temp files cleaned on job completion.
4. **Flatten `CascadeResult` → `Vec<MatchPairDto>`:** Add
   `matched_at_level: Option<u8>` to `MatchPairDto`. Flatten all level results
   into one combined set.
5. **CancelToken between levels:** Thread `CancelToken` into
   `run_cascade_inmemory`. Check between levels. Cancellation granularity is
   per-level (documented limitation).
6. **Progress mapping:** Map `CascadeProgress` to `ProgressEventDto`. Frontend
   shows "Level L3 — batch 4/12" when cascade is active.

#### configStore Shape

```typescript
cascade: {
  enabled: boolean;
  preset: 'standard' | 'extended' | 'full' | 'custom';
  levels: number[];        // 1–11 only
  fuzzy_threshold: number;
  exclusion_mode: 'exclusive' | 'independent';
}
```

#### StatusRail

When cascade active: show "Deep Match (6 levels)" instead of algorithm name.

### File Changes

| File | Change |
|------|--------|
| `src/run_service/mod.rs` | Add cascade routing in `run_worker` |
| `src/run_service/dto.rs` | Add `CascadeOptionsDto`; add `matched_at_level` to `MatchPairDto`; add cascade field to `RunConfigDto` |
| `src/matching/cascade.rs` | Thread `CancelToken` parameter; check between levels |
| `ui/src/features/configure/ConfigureTab.tsx` | Add mode toggle (Quick Match / Deep Match) at top |
| `ui/src/features/configure/CascadePicker.tsx` | **New file.** Preset selector + grouped checkbox grid + exclusion mode + threshold slider |
| `ui/src/shared/stores/configStore.ts` | Add `cascade` state shape + `setCascade` action |
| `ui/src/shared/tauri/types.ts` | Add `CascadeOptionsDto` type; update `MatchPairDto` |
| `ui/src/shared/tauri/zod-schemas.ts` | Add cascade validation |
| `ui/src/app/StatusRail.tsx` | Show "Deep Match (N levels)" when cascade enabled |
| `ui/src/features/run/RunTab.tsx` | Show level-aware progress |

### Accessibility

- Mode toggle: `role="radiogroup"` with `aria-selected`.
- Level checkboxes: `aria-disabled` + `aria-describedby` pointing to reason text.
- Threshold slider: `aria-valuemin`, `aria-valuemax`, `aria-valuenow`,
  `aria-valuetext`.

### Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Cascade writes CSV files as side-effect | Medium | Direct to temp_dir; clean up on job complete |
| No mid-level cancellation | Medium | Document per-level granularity; show "Cancelling after current level…" |
| Memory pressure (exclusive mode clones vectors) | Low | Document in-memory-only constraint; warn on datasets > 500K rows |
| Preset auto-selection wrong for edge cases | Low | Always allow Custom override |
| Time estimate inaccurate | Low | Label as "Estimated" with disclaimer |

---

## Cross-Cutting Concerns

### Testing Strategy

| Feature | Test Type | What to Verify |
|---------|-----------|---------------|
| CUDA | Manual | StatusRail shows device name; no crash on machines without GPU |
| Connection Persistence | Unit + Manual | Store/load round-trip; credential failure clears password |
| Pause/Resume | Integration | Cancel-from-paused works; state transitions emit correctly; 50ms loop doesn't spin CPU |
| Cascade | Integration + Manual | GeoColumnStatus derived correctly; results flatten with level attribution; L12 never sent; temp files cleaned |

### Open Questions (Non-Blocking)

1. Should the Results tab support filtering by `matched_at_level`? (Deferred to
   post-ship iteration.)
2. Should cascade time estimates be calibrated per-machine using CUDA probe
   data? (Nice-to-have for v2.)
3. Should we expose `MissingColumnMode` dropdown for power users? (Gather
   feedback first.)

---

## Acceptance Criteria

1. ✅ Status Rail shows "GPU available" + device name on a CUDA-capable host.
2. ✅ Closing and reopening the app restores the last-used connection details.
3. ✅ Pause button pauses progress between batches; Resume continues; Cancel
   works from paused state.
4. ✅ Cascade section shows L1–L11 with correct column requirements; disabled
   levels show why; presets auto-select based on detected columns.
5. ✅ "Quick Match" / "Deep Match" toggle is clear to non-developer operators.
