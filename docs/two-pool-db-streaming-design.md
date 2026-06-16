# Two-Pool DB Streaming Design

## Status

Design-only document produced in Phase 5 of `docs/performance-remediation-swarm-plan.md`. No implementation is approved until all gates in [Implementation Gates](#implementation-gates) pass.

## Problem Statement

### Current single-pool limitation

The production Tauri streaming path routes DB-to-DB deterministic runs through `stream_match_csv_partitioned`, which accepts **one** `MySqlPool` and queries both `source.table` and `target.table` through that pool:

```172:175:src-tauri/src/commands/matching.rs
                    let count = rt.block_on(name_matcher::matching::stream_match_csv_partitioned(
                        &src,
                        &run_config.source.table,
                        &run_config.target.table,
```

That API is only correct when both tables are reachable on the **same** MySQL connection (same host, credentials, database, and session). The app models separate connections as distinct `session_id` values in `AppState.db`, each with its own pool.

### Failure modes when the invariant is violated

If cross-session runs were routed into single-pool streaming:

1. **Runtime error** — target table not found on the source connection.
2. **Silent wrong-database match** — a same-named table exists on the source database and produces incorrect pairs.
3. **OOM on fallback** — cross-session runs that miss streaming fall back to loading both tables into `Vec<Person>` via separate pools (`load_selection_rows`), which does not scale past memory limits.

### Current mitigation (Phase 4)

`should_use_db_streaming_worker` and the mirrored frontend `streamingBackendActive` require `source.session_id === target.session_id`. Cross-session DB runs use the in-memory two-pool loader until a dedicated two-pool streaming runner exists.

### Existing but unwired building blocks

The matching engine already exposes lower-level dual-pool primitives that are **not** selected by scale policy or the Tauri runner today:

- `stream_match_csv_dual(pool1, pool2, ...)` — non-partitioned hash-join streaming across two pools.
- `stream_match_csv_internal(pool1, pool2, ...)` — shared implementation used by single-pool wrappers when `pool1 == pool2`.

Partitioned multi-pass streaming (`stream_match_csv_partitioned`) remains single-pool only and would need a dual-pool variant for cross-session large jobs.

---

## Goals and Non-Goals

### Goals

- Enable **correct** deterministic DB-to-DB matching when source and target use different `session_id` values.
- Preserve **parity** with the existing in-memory two-pool path on small fixtures (match count, stable pair keys, confidence ordering).
- Keep peak RSS within **+20%** of same-session streaming baseline for equivalent row counts.
- Reuse existing cancellation, pause/resume, progress, and result-store append contracts.

### Non-Goals (this design)

- Fuzzy, cascade, or GPU-only algorithms on cross-session streaming (remain blocked at scale).
- Automatic session merging or credential sharing across connections.
- Replacing the same-session partitioned runner; two-pool is an **additional** runner.
- SQL-backed diff, XLSX streaming export, or other large-result consumers (separate remediation tracks).

---

## API Contract

### Runner selection (backend)

Introduce a separate scale-policy branch and Tauri runner hook:

| Condition | Runner |
| --- | --- |
| DB-to-DB, same session, streaming mode, supported algorithm | `stream_match_csv_partitioned` (current) |
| DB-to-DB, **different** sessions, streaming mode, supported algorithm, capability enabled | `stream_match_csv_dual_partitioned` (new) or `stream_match_csv_dual` (MVP) |
| DB-to-DB, different sessions, capability **disabled** | In-memory two-pool loader (current fallback) |
| DB-to-DB, different sessions, row count ≥ in-memory safe threshold | Block or strong-warn with recovery copy (policy constant) |

Policy function sketch:

```rust
pub fn should_use_two_pool_db_streaming_worker(config: &RunConfigDto, caps: &RuntimeCapabilitiesDto) -> bool {
    caps.supports_cross_session_streaming
        && is_db_to_db(config)
        && !is_same_db_session(config)
        && !config.cascade.as_ref().is_some_and(|c| c.enabled)
        && algorithm_supports_db_streaming(config.algorithm)
        && matches!(resolve_effective_run_mode(...), EffectiveRunMode::Streaming)
}
```

The Tauri layer resolves **two** pools before spawning the worker thread (same pattern as the current loader):

```rust
let src_pool = state.db.get(&config.source.session_id)?.pool;
let tgt_pool = state.db.get(&config.target.session_id)?.pool;
```

### Source pool cursor

**Role:** drive the **outer** table in hash-join streaming (smaller cardinality side becomes inner; outer is scanned in keyset order).

**Mechanism:**

- Keyset pagination: `WHERE id > :last_id AND id <= :watermark ORDER BY id LIMIT :batch`.
- `watermark` captured at run start from `get_max_id(outer_pool, outer_table)` so concurrent inserts during the run are excluded consistently with the in-memory path.
- `last_id` monotonic per outer scan; persisted in checkpoint on pause/resume.
- Column mapping (`ColumnMapping`) applied on read via existing `fetch_person_rows_chunk_keyset` helpers.

**Contract:**

```rust
struct OuterCursor {
    pool: MySqlPool,
    table: String,
    mapping: Option<ColumnMapping>,
    last_id: i64,
    watermark: i64,
    batch_size: i64,
}
```

### Target pool cursor / partition lookup

**Role:** build and probe the **inner** hash index from the target pool (or source pool when target is smaller).

**Non-partitioned MVP (`stream_match_csv_dual`):**

- Inner table loaded in keyset chunks from `inner_pool`.
- Each chunk normalized and inserted into `HashMap<u64, Vec<IndexedPerson>>` keyed by deterministic concat key.
- Outer batches probe the in-memory index; index retained for the full pass (same as current `stream_match_csv_internal`).

**Partitioned v2 (`stream_match_csv_dual_partitioned`):**

- Apply the same `PartitionStrategy` (`last_initial`, `birthyear5`, etc.) independently to source and target mappings.
- Require `parts_source.len() == parts_target.len()`; bail with validation error if mismatched.
- Per partition:
  - Build inner index from `inner_pool` + `inner_where` + `inner_binds`.
  - Scan outer via `outer_pool` + `outer_where` + outer keyset cursor.
  - Drop inner index before next partition to bound RSS.

**Partition lookup contract:**

```rust
struct PartitionSlice {
    source_where_sql: String,
    source_binds: Vec<BindValue>,
    target_where_sql: String,
    target_binds: Vec<BindValue>,
    partition_index: usize,
}
```

### Join semantics

- **Algorithm:** deterministic hash join on normalized person keys (`DeterministicFnLnBd`, `DeterministicFnMnLnBd` only).
- **Normalization:** reuse `normalize_person` + `concat_key_for_np` — same functions as in-memory and same-session streaming.
- **Birthdate swap:** honor `config.options.allow_birthdate_swap`; when enabled, emit lookup variants consistent with the in-memory path (may produce multiple probe keys per person; pairs deduplicated before `on_match`).
- **Direction:** table1 = source selection, table2 = target selection; `MatchPair.person1` = source person, `MatchPair.person2` = target person (unchanged).
- **Confidence:** identical scoring path as `stream_match_csv_internal`; no approximate shortcuts.

### Deterministic ordering

Total order for emitted matches (required for reproducible `row_id` assignment and regression baselines):

1. **Partition index** ascending (partitioned mode) or single implicit partition `0` (dual MVP).
2. **Outer person `id`** ascending within partition.
3. **Inner probe bucket iteration** in stable `HashMap` key order (documented as insertion order per chunk for a given fixture).
4. **Within bucket**, persons sorted by `id` ascending before pair emission.
5. **Duplicate suppression** before emit (see below) keeps first occurrence in this order.

Progress `processed` counts outer rows examined; `matches_found` counts pairs **after** deduplication.

### Cancellation

**Layers:**

| Layer | Signal | Behavior |
| --- | --- | --- |
| `CancelToken` | `cancel.is_cancelled()` in Tauri runner closure | Stop appending rows; return cancelled terminal state |
| `StreamControl.cancel` | `AtomicBool` inside matching engine | Checked between inner/outer chunks and partition boundaries |
| Match callback | `bail!("__name_match_cancelled__")` | Propagate out of `on_match`; runner maps to `JobStateDto::Cancelled` |

**Requirements:**

- No partial checkpoint promotion on cancel (checkpoint may remain for explicit resume).
- Terminal state must be `Cancelled`, not `Failed`, when user-initiated.
- Result rows already appended remain persisted (no rollback of spilled rows).

### Pause / resume

**Mechanism:** existing `StreamControl.pause` + `StreamCheckpoint` files (`cfg.checkpoint_path`, `cfg.resume`).

**Checkpoint payload** (extend for two-pool):

```rust
struct StreamCheckpoint {
    // existing fields...
    pool_mode: "single" | "dual",
    source_session_fingerprint: String,  // hash of session_id only, not credentials
    target_session_fingerprint: String,
    partition_idx: i32,
    last_id: Option<i64>,
    watermark_id: Option<i64>,
    table_inner: String,
    table_outer: String,
    algorithm: String,
    batch_size: i64,
}
```

**Resume validation:**

- Refuse resume if source/target table names, session fingerprints, or algorithm differ from checkpoint.
- Refuse resume if either session is no longer registered in `AppState.db`.

**Pause behavior:**

- Block at chunk boundary when `pause` is set; emit `JobStateDto::Paused` via sink.
- Save checkpoint before sleeping.

### Progress events

Map `ProgressUpdate` → `ProgressEventDto` identically to the current streaming runner:

| Field | Source |
| --- | --- |
| `processed` / `total` | Outer rows in current partition or full scan |
| `percent` | `processed / total`, clamped 0–100 |
| `stage` | `"indexing"`, `"indexing_hash"`, `"matching"`, `"partition"` |
| `mem_used_mb` / `mem_avail_mb` | `memory_stats_mb()` |
| `matches_found` | `AtomicU64` in runner, updated after dedupe |
| `eta_secs` | Existing estimator from chunk throughput |

Emit at most once per 500 ms unless stage changes (match current partitioned runner).

### Duplicate handling

Duplicates can arise from birthdate-swap multi-keys, symmetric key collisions, or partition boundary overlap.

**Stable pair key** (for dedupe and diff baselines):

```text
pair_key = (source_person_id, target_person_id)
```

Optional tie-breaker when IDs are unstable: `(source_uuid, target_uuid)` if both non-empty.

**Policy:**

- Maintain `HashSet<PairKey>` per job (or per partition with merge at end for partitioned mode).
- On collision, keep the **first** emitted pair per deterministic ordering; drop subsequent identical keys.
- Do not increment `matches_found` for dropped duplicates.
- Dedupe set is **not** persisted across resume unless checkpoint stores its hash/size for sanity check; on resume, rebuild set from `ResultStore` row scan if needed for exactly-once semantics (implementation gate).

### Result row id assignment

**Contract unchanged from current streaming runner:**

```rust
let row_id = next_row_id.fetch_add(1, Ordering::Relaxed);
let dto = match_pair_to_dto(row_id, pair);
```

- `row_id` is job-local, monotonic `u64`, assigned **at emit time** after dedupe.
- Rows buffered in batches of 1 000 then `append_result_rows`.
- Spill to SQLite follows existing `RESULT_SPILL_ROWS` policy.
- `row_id` is not derived from person IDs and must not be reused after cancel/resume unless resume continues the same `AtomicU64` counter from checkpoint.

### Error mapping

| Error class | Example | User-facing mapping |
| --- | --- | --- |
| Validation | Session not found, table empty, partition count mismatch | `AppError::Validation` with actionable copy |
| Policy block | Cross-session streaming disabled | Friendly policy message + link to same-session reconnect |
| Cancelled | `__name_match_cancelled__` | `JobStateDto::Cancelled`, no toast as error |
| OOM / internal | Driver errors, pool timeout | `AppError::Internal` or `Failed` with generic retry guidance |
| Algorithm | Fuzzy requested on streaming path | Existing algorithm bail message |

**No secrets in errors:** never include passwords, DSNs, or connection strings in `Display` or emitted events.

---

## Capability Flag

### `supports_cross_session_streaming`

Add to backend DTO (proposed):

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeCapabilitiesDto {
    pub supports_cross_session_streaming: bool,
    // existing CUDA fields remain on CudaDiagnosticsDto or migrate here
}
```

| Value | Meaning |
| --- | --- |
| `false` (default until gates pass) | UI and scale policy keep cross-session streaming blocked; in-memory fallback or policy block. |
| `true` | `should_use_two_pool_db_streaming_worker` may select the dual-pool runner when other conditions hold. |

Expose via Tauri command `runtime_capabilities()` or extend `system_info()`. Frontend reads once at configure/run time and caches in a store.

**Rollout:** compile-time or runtime flag default `false`; flip to `true` only in builds that pass DB smoke gates.

---

## UI Contract

### Current state (capability false or absent)

| Scenario | UI behavior |
| --- | --- |
| Same-session DB, 100k+, deterministic | `streamingBackendActive` = true; show streaming indicator. |
| Cross-session DB, any scale | `streamingBackendActive` = false; run uses in-memory loader. |
| Cross-session DB, 100k+ | Non-blocking warning: *Large database streaming currently requires source and target tables from the same DB session. Reconnect both tables through one session or run a smaller job.* |
| Cross-session DB, 500k+ | Strong confirmation before start; explain memory risk of in-memory fallback. |
| Cross-session DB, 1M+ | Hard block for non-streamable paths (same as million-row policy). |

Compare, export, and paging behave as today; no change until capability is true.

### Future state (`supports_cross_session_streaming: true`)

| Scenario | UI behavior |
| --- | --- |
| Cross-session DB, streaming mode, supported algorithm | `streamingBackendActive` = true; label: *Two-pool DB streaming*. |
| Cross-session DB, in-memory mode | Warn that user forced in-memory across sessions. |
| Pause/resume | Show checkpoint path controls when `streaming.resume` enabled (same as same-session). |
| Export | CSV always; XLSX subject to 100k guard regardless of streaming mode. |

### Diagnostics (no secrets)

Extend diagnostics panel / run detail with booleans and counts only:

```typescript
interface DbStreamingDiagnostics {
  supports_cross_session_streaming: boolean;
  same_db_session: boolean;           // source.session_id === target.session_id
  streaming_backend_active: boolean;  // effective policy decision
  effective_run_mode: "in-memory" | "streaming";
  source_row_count: number;
  target_row_count: number;
}
```

**Must not display:** `session_id` values, hostnames, usernames, passwords, ports, or connection strings in user-facing diagnostics. Session equality is a **boolean** only.

---

## Implementation Gates

Implementation must not merge until **all** gates pass:

| Gate | Evidence required |
| --- | --- |
| G1 — Small fixture parity | Cross-session dual-pool match count and stable pair keys equal in-memory two-pool baseline on `db_cross_session_10k` fixture. |
| G2 — Asymmetric counts | Source 10k / target 100k (and reverse) produces same pairs as in-memory on deterministic algo. |
| G3 — Memory ceiling | Peak RSS ≤ same-session streaming baseline + 20% at 100k rows per side. |
| G4 — Cancellation honesty | Mid-run cancel → `Cancelled` terminal state; no panic; partial results retained. |
| G5 — Progress continuity | Progress events monotonic until terminal; `matches_found` matches stored row count. |
| G6 — Pause/resume | Resume from checkpoint continues row_id sequence and does not duplicate pairs. |
| G7 — Policy alignment | Backend `should_use_two_pool_db_streaming_worker` and frontend `streamingBackendActive` agree for all matrix cases. |
| G8 — DB smoke | `db_cross_session_blocked` proves old single-pool path cannot activate; new path activates only when flag true. |
| G9 — Review | Context Engine `review_auto` on runner and scale changes. |

### Rollback posture

- Keep `supports_cross_session_streaming` default **false** in release builds until G1–G8 pass.
- Same-session guard remains in `should_use_db_streaming_worker` regardless of two-pool status.
- If dual-pool runner regresses, disable capability flag; no schema migration required.
- Checkpoints from dual-pool runs invalidated when rolling back (document in release notes).

---

## Correctness Requirements vs In-Memory Two-Pool Path

The in-memory path loads `Vec<Person>` from each pool via `load_selection_rows`, then runs synchronous deterministic matching in the worker thread. The two-pool streaming path must produce **equivalent match sets** on the same frozen table snapshots:

| Dimension | Requirement |
| --- | --- |
| Match count | Exact equality on baseline fixtures. |
| Pair set | `{(source_id, target_id)}` sets equal after dedupe. |
| Confidence | Per-pair confidence within float epsilon of in-memory (`1e-6`) or bit-identical if same code path. |
| Matched fields | String-equal per pair. |
| Ordering | Streaming order may differ from in-memory iteration order; **pair set** must match. `row_id` order follows streaming emit order, not in-memory order. |
| Filters after match | Paging/export filters apply at read time; not part of streaming parity. |
| Person snapshots | Large streamed jobs may omit snapshots (existing behavior); explain-pair may be unsupported. |
| Birthdate swap | Same pairs as in-memory when `allow_birthdate_swap` enabled. |
| Column mapping | Mapped columns only; identical normalization. |
| Concurrent DB writes | Undefined during run; watermarks freeze outer/inner visibility at start (same as same-session streaming). |

**Regression test plan:**

1. Fixed MySQL smoke tables `smoke_cross_a`, `smoke_cross_b` in disposable databases.
2. Run in-memory cross-session job → capture pair set hash.
3. Run two-pool streaming job → assert pair set hash equality.
4. Repeat with `allow_birthdate_swap: true` and partition strategies `last_initial`, `birthyear5`.

---

## Architecture Sketch

```text
┌─────────────────────────────────────────────────────────────┐
│                        Tauri start_run                       │
└────────────────────────────┬────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │     scale policy decision    │
              └──────────────┬──────────────┘
         same session        │         cross session
              │              │              │
              v              │              v
   stream_match_csv_         │    capability false → TableLoader
   partitioned (1 pool)     │              │         (in-memory)
              │              │              v
              │              │    capability true → stream_match_csv_
              │              │              dual[_partitioned] (2 pools)
              v              v              v
         ┌────────────────────────────────────────┐
         │  on_match → match_pair_to_dto → store  │
         │  ProgressUpdate → ProgressEventDto      │
         └────────────────────────────────────────┘
```

---

## Open Questions

1. **Partitioned dual-pool priority:** Ship `stream_match_csv_dual` MVP first (simpler, higher peak RSS on large inner tables) or wait for partitioned dual-pool before enabling the flag?
2. **Cross-session million-row policy:** Should 1M+ cross-session runs hard-block even with two-pool streaming until G3 evidence exists at 250k+?
3. **Dedupe on resume:** Rebuild dedupe set from SQLite on resume vs persist bloom filter in checkpoint?
4. **Session fingerprint:** Hash of `session_id` only, or include database name from table selection for stricter resume validation?

---

## Related Documents

- `docs/performance-remediation-swarm-plan.md` — Phase 5 scope and acceptance criteria.
- `docs/codebase-performance-audit-report.md` — Validation ledger and confirmed same-session guard fix.
- `docs/million-row-scale-plan.md` — Scale ladder and CI smoke expectations.
