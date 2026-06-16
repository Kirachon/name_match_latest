//! Run-scoped result store.
//!
//! In Phase 1 (this PR) the store keeps results in process memory inside an
//! indexed `Vec<MatchPairDto>` per job. Reads happen through
//! `get_results_page` which builds a deterministic ordered slice.
//!
//! Phase 2 adds an app-scoped SQLite sidecar. Memory remains the hot path, and
//! completed jobs are written through so evicted/restarted jobs can still be
//! paged, exported, explained, reviewed, and diffed by later features.

use super::dto::{
    AlgorithmDto, DiffChangedRowDto, DiffJobsRequestDto, DiffResultDto, JobStateDto, JobSummaryDto,
    MatchPairDto, ResultPageDto, ResultPageRequestDto, ReviewDecisionDto, SaveDecisionRequestDto,
};
use crate::models::Person;
use anyhow::{Result, bail};
use parking_lot::Mutex;
use rusqlite::{Connection, OptionalExtension, params};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy)]
pub struct ResultStoreConfig {
    pub max_retained: usize,
}

impl Default for ResultStoreConfig {
    fn default() -> Self {
        Self { max_retained: 50 }
    }
}

#[derive(Debug, Clone)]
pub struct StoredJob {
    pub summary: JobSummaryDto,
    pub allow_birthdate_swap: bool,
    pub persist_result_history: bool,
    pub rows: Arc<Vec<MatchPairDto>>,
    pub source_people: Arc<Vec<Person>>,
    pub target_people: Arc<Vec<Person>>,
    pub last_accessed_unix_ms: u64,
    /// When true, match rows live in SQLite only (bounded RAM for million-row jobs).
    pub spilled: bool,
}

type DecisionKey = (String, i64, i64);

pub struct ResultStore {
    inner: Mutex<HashMap<String, StoredJob>>,
    config: ResultStoreConfig,
    sqlite: Option<SqliteStore>,
    /// Lightweight sidecar so review decisions survive a broken or huge results DB.
    decisions_sqlite: Option<DecisionsSqliteStore>,
    decisions_memory: Mutex<HashMap<DecisionKey, ReviewDecisionDto>>,
}

/// Filters applied when exporting match rows (confidence, cascade levels, review rejections).
pub struct ExportRowFilter<'a> {
    pub min_confidence: f32,
    pub levels: &'a [u8],
    pub rejected_pairs: &'a HashSet<(i64, i64)>,
}

impl Default for ResultStore {
    fn default() -> Self {
        Self::new()
    }
}

impl ResultStore {
    pub fn new() -> Self {
        Self::with_config(ResultStoreConfig::default())
    }

    pub fn with_config(config: ResultStoreConfig) -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
            config,
            sqlite: None,
            decisions_sqlite: None,
            decisions_memory: Mutex::new(HashMap::new()),
        }
    }

    pub fn with_sqlite_path(config: ResultStoreConfig, path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        Ok(Self::open_with_paths(
            config,
            path.clone(),
            decisions_path_for(&path),
        ))
    }

    /// Open the Tauri app-scoped result store. Results and review decisions use
    /// separate SQLite files so a huge or damaged results DB cannot block review.
    pub fn open_at(config: ResultStoreConfig, app_dir: &Path) -> Self {
        Self::open_with_paths(
            config,
            app_dir.join("result_store.sqlite3"),
            app_dir.join("review_decisions.sqlite3"),
        )
    }

    fn open_with_paths(
        config: ResultStoreConfig,
        results_path: PathBuf,
        decisions_path: PathBuf,
    ) -> Self {
        let decisions_sqlite = DecisionsSqliteStore::open(&decisions_path)
            .map_err(|err| {
                log::error!(
                    "review decisions SQLite unavailable at {}: {err}",
                    decisions_path.display()
                );
                err
            })
            .ok();
        let sqlite = SqliteStore::open(&results_path)
            .map_err(|err| {
                log::error!(
                    "SQLite result store unavailable at {}; using memory only for results: {err}",
                    results_path.display()
                );
                err
            })
            .ok();
        let store = Self {
            inner: Mutex::new(HashMap::new()),
            config,
            sqlite,
            decisions_sqlite,
            decisions_memory: Mutex::new(HashMap::new()),
        };
        if let Err(err) = store.maybe_import_decisions_from_results() {
            log::warn!("failed to import legacy review decisions: {err}");
        }
        store
    }

    pub fn decisions_persisted(&self) -> bool {
        self.decisions_sqlite.is_some() || self.sqlite.is_some()
    }

    /// Reserve a slot at job-start time so the frontend can resolve `job_id`
    /// before the engine has produced any rows.
    pub fn reserve(
        &self,
        job_id: String,
        algorithm: AlgorithmDto,
        source_table: String,
        target_table: String,
        started_at_unix_ms: u64,
    ) {
        self.reserve_with_options(
            job_id,
            algorithm,
            source_table,
            target_table,
            started_at_unix_ms,
            false,
            true,
        );
    }

    pub fn reserve_with_options(
        &self,
        job_id: String,
        algorithm: AlgorithmDto,
        source_table: String,
        target_table: String,
        started_at_unix_ms: u64,
        allow_birthdate_swap: bool,
        persist_result_history: bool,
    ) {
        let mut g = self.inner.lock();
        g.insert(
            job_id.clone(),
            StoredJob {
                summary: JobSummaryDto {
                    job_id,
                    state: JobStateDto::Starting,
                    algorithm,
                    source_table,
                    target_table,
                    matches_found: 0,
                    elapsed_secs: 0,
                    started_at_unix_ms,
                    finished_at_unix_ms: None,
                },
                allow_birthdate_swap,
                persist_result_history,
                rows: Arc::new(Vec::new()),
                source_people: Arc::new(Vec::new()),
                target_people: Arc::new(Vec::new()),
                last_accessed_unix_ms: started_at_unix_ms,
                spilled: false,
            },
        );
        self.evict_terminal_locked(&mut g);
    }

    pub fn set_rows(&self, job_id: &str, rows: Vec<MatchPairDto>) -> Result<()> {
        let mut g = self.inner.lock();
        match g.get_mut(job_id) {
            Some(slot) => {
                slot.summary.matches_found = rows.len() as u64;
                slot.rows = Arc::new(rows);
                slot.last_accessed_unix_ms = now_ms();
                if slot.persist_result_history
                    && let Some(sqlite) = &self.sqlite
                {
                    sqlite.save_job(slot)?;
                }
                Ok(())
            }
            None => bail!("Unknown job id: {}", job_id),
        }
    }

    pub fn clear_rows(&self, job_id: &str) -> Result<()> {
        self.set_rows(job_id, Vec::new())
    }

    pub fn append_result_rows(&self, job_id: &str, rows: &[MatchPairDto]) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let mut g = self.inner.lock();
        match g.get_mut(job_id) {
            Some(slot) => {
                if slot.spilled {
                    slot.summary.matches_found =
                        slot.summary.matches_found.saturating_add(rows.len() as u64);
                    let sqlite = self
                        .sqlite
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("Spilled results require SQLite storage"))?;
                    sqlite.upsert_job_metadata(slot)?;
                    sqlite.append_result_rows(&slot.summary.job_id, rows)?;
                } else if slot.rows.len() + rows.len() >= super::scale::RESULT_SPILL_ROWS {
                    slot.summary.matches_found =
                        slot.summary.matches_found.saturating_add(rows.len() as u64);
                    if let Some(sqlite) = &self.sqlite {
                        sqlite.upsert_job_metadata(slot)?;
                        if !slot.rows.is_empty() {
                            sqlite.append_result_rows(&slot.summary.job_id, &slot.rows)?;
                            Arc::make_mut(&mut slot.rows).clear();
                        }
                        sqlite.append_result_rows(&slot.summary.job_id, rows)?;
                        slot.spilled = true;
                    } else {
                        Arc::make_mut(&mut slot.rows).extend_from_slice(rows);
                        slot.summary.matches_found = slot.rows.len() as u64;
                    }
                } else {
                    Arc::make_mut(&mut slot.rows).extend_from_slice(rows);
                    slot.summary.matches_found = slot.rows.len() as u64;
                    if slot.persist_result_history
                        && let Some(sqlite) = &self.sqlite
                    {
                        sqlite.upsert_job_metadata(slot)?;
                        sqlite.append_result_rows(&slot.summary.job_id, rows)?;
                    }
                }
                slot.last_accessed_unix_ms = now_ms();
                Ok(())
            }
            None => bail!("Unknown job id: {}", job_id),
        }
    }

    pub fn enable_spill_mode(&self, job_id: &str) -> Result<()> {
        let mut g = self.inner.lock();
        let Some(slot) = g.get_mut(job_id) else {
            bail!("Unknown job id: {}", job_id);
        };
        slot.spilled = true;
        Arc::make_mut(&mut slot.rows).clear();
        Ok(())
    }

    pub fn set_person_snapshots(
        &self,
        job_id: &str,
        source_people: Arc<Vec<Person>>,
        target_people: Arc<Vec<Person>>,
    ) -> Result<()> {
        let mut g = self.inner.lock();
        match g.get_mut(job_id) {
            Some(slot) => {
                slot.source_people = source_people;
                slot.target_people = target_people;
                slot.last_accessed_unix_ms = now_ms();
                if slot.persist_result_history
                    && let Some(sqlite) = &self.sqlite
                {
                    sqlite.save_job(slot)?;
                }
                // Bound RAM for large jobs: once rows have spilled to SQLite, keep
                // person snapshots in SQLite only. Explain/diff reload them from
                // result_person_lookup, or degrade gracefully for trimmed large runs.
                if slot.spilled {
                    slot.source_people = Arc::new(Vec::new());
                    slot.target_people = Arc::new(Vec::new());
                }
                Ok(())
            }
            None => bail!("Unknown job id: {}", job_id),
        }
    }

    pub fn mark_finished(&self, job_id: &str, matches: u64, unix_ms: u64) {
        let mut g = self.inner.lock();
        if let Some(slot) = g.get_mut(job_id) {
            slot.summary.matches_found = matches;
            slot.summary.finished_at_unix_ms = Some(unix_ms);
            slot.summary.state = JobStateDto::Completed;
            slot.summary.elapsed_secs =
                ((unix_ms.saturating_sub(slot.summary.started_at_unix_ms)) / 1000) as u64;
            slot.last_accessed_unix_ms = unix_ms;
            if slot.persist_result_history
                && let Some(sqlite) = &self.sqlite
            {
                if let Err(err) = sqlite.save_job(slot) {
                    log::error!("failed to persist finished job {job_id}: {err}");
                }
            }
        }
        self.evict_terminal_locked(&mut g);
    }

    pub fn set_state(&self, job_id: &str, state: JobStateDto) {
        let unix_ms = now_ms();
        let mut g = self.inner.lock();
        if let Some(slot) = g.get_mut(job_id) {
            slot.summary.state = state;
            slot.last_accessed_unix_ms = unix_ms;
            if state.is_terminal() {
                slot.summary.finished_at_unix_ms.get_or_insert(unix_ms);
                slot.summary.elapsed_secs =
                    ((unix_ms.saturating_sub(slot.summary.started_at_unix_ms)) / 1000) as u64;
            }
            if slot.persist_result_history
                && let Some(sqlite) = &self.sqlite
            {
                if let Err(err) = sqlite.save_job(slot) {
                    log::error!("failed to persist job state {job_id}: {err}");
                }
            }
        }
        self.evict_terminal_locked(&mut g);
    }

    pub fn summary(&self, job_id: &str) -> Option<JobSummaryDto> {
        let mut g = self.inner.lock();
        if let Some(summary) = g.get_mut(job_id).map(|s| {
            s.last_accessed_unix_ms = now_ms();
            s.summary.clone()
        }) {
            return Some(summary);
        }
        drop(g);
        self.sqlite
            .as_ref()
            .and_then(|sqlite| sqlite.load_summary(job_id).ok().flatten())
    }

    pub fn list_summaries(&self) -> Vec<JobSummaryDto> {
        let g = self.inner.lock();
        let mut by_id: HashMap<String, JobSummaryDto> = self
            .sqlite
            .as_ref()
            .and_then(|sqlite| sqlite.list_summaries().ok())
            .unwrap_or_default()
            .into_iter()
            .map(|summary| (summary.job_id.clone(), summary))
            .collect();
        for summary in g.values().map(|s| s.summary.clone()) {
            by_id.insert(summary.job_id.clone(), summary);
        }
        let mut v: Vec<JobSummaryDto> = by_id.into_values().collect();
        v.sort_by_key(|s| std::cmp::Reverse(s.started_at_unix_ms));
        v
    }

    /// Borrow rows + a copy of summary under one lock.
    pub fn snapshot(&self, job_id: &str) -> Option<StoredJob> {
        let mut g = self.inner.lock();
        if let Some(job) = g.get_mut(job_id).map(|s| {
            s.last_accessed_unix_ms = now_ms();
            s.clone()
        }) {
            return Some(job);
        }
        drop(g);
        self.sqlite
            .as_ref()
            .and_then(|sqlite| sqlite.load_job(job_id).ok().flatten())
    }

    pub fn forget_job(&self, job_id: &str) -> Result<Option<StoredJob>> {
        let mut g = self.inner.lock();
        if let Some(slot) = g.get(job_id) {
            if slot.summary.state.is_active() {
                bail!("Cannot forget active job: {}", job_id);
            }
        }
        let removed = g.remove(job_id);
        drop(g);
        if let Some(sqlite) = &self.sqlite {
            sqlite.delete_job(job_id)?;
        }
        Ok(removed)
    }

    pub fn drop_job(&self, job_id: &str) -> Result<Option<StoredJob>> {
        self.forget_job(job_id)
    }

    pub fn save_decision(&self, request: SaveDecisionRequestDto) -> Result<ReviewDecisionDto> {
        validate_decision(&request.decision)?;
        if self.summary(&request.job_id).is_none() {
            bail!("Unknown job id: {}", request.job_id);
        }
        if let Some(store) = &self.decisions_sqlite {
            return store.save_decision(&request);
        }
        if let Some(sqlite) = &self.sqlite {
            return sqlite.save_decision(&request);
        }
        let dto = review_decision_from_request(&request);
        self.decisions_memory.lock().insert(
            (
                request.job_id.clone(),
                request.source_id,
                request.target_id,
            ),
            dto.clone(),
        );
        Ok(dto)
    }

    pub fn get_decisions(&self, job_id: &str) -> Result<Vec<ReviewDecisionDto>> {
        if self.summary(job_id).is_none() {
            bail!("Unknown job id: {}", job_id);
        }
        if let Some(store) = &self.decisions_sqlite {
            return store.get_decisions(job_id);
        }
        if let Some(sqlite) = &self.sqlite {
            return sqlite.get_decisions(job_id);
        }
        let mut decisions = self
            .decisions_memory
            .lock()
            .values()
            .filter(|decision| decision.job_id == job_id)
            .cloned()
            .collect::<Vec<_>>();
        decisions.sort_by_key(|decision| decision.row_id);
        Ok(decisions)
    }

    fn maybe_import_decisions_from_results(&self) -> Result<()> {
        let Some(main) = &self.sqlite else {
            return Ok(());
        };
        let Some(side) = &self.decisions_sqlite else {
            return Ok(());
        };
        if !side.is_empty()? {
            return Ok(());
        }
        for decision in main.list_all_decisions()? {
            side.save_decision_dto(&decision)?;
        }
        Ok(())
    }

    pub fn diff(&self, request: &DiffJobsRequestDto) -> Result<DiffResultDto> {
        if request.base_job_id == request.compare_job_id {
            bail!("Choose two different jobs to compare");
        }
        let Some(base_summary) = self.summary(&request.base_job_id) else {
            bail!("Unknown job id: {}", request.base_job_id);
        };
        let Some(compare_summary) = self.summary(&request.compare_job_id) else {
            bail!("Unknown job id: {}", request.compare_job_id);
        };
        if base_summary.matches_found > super::scale::MAX_DIFF_ROWS
            || compare_summary.matches_found > super::scale::MAX_DIFF_ROWS
        {
            bail!(super::scale::DIFF_TOO_LARGE_MESSAGE);
        }
        let base_rows = self.load_rows_for_diff(&request.base_job_id)?;
        let compare_rows = self.load_rows_for_diff(&request.compare_job_id)?;
        Ok(diff_jobs(&base_rows, &compare_rows, request))
    }

    /// Iterate exportable rows in bounded chunks without loading the full job into RAM.
    /// Returns the number of rows delivered to the callback after filters.
    pub fn for_each_export_row(
        &self,
        job_id: &str,
        filter: &ExportRowFilter<'_>,
        chunk_size: usize,
        mut on_chunk: impl FnMut(&[MatchPairDto]) -> Result<()>,
    ) -> Result<u64> {
        validate_levels(filter.levels)?;
        if self.summary(job_id).is_none() {
            bail!("Unknown job id: {}", job_id);
        }
        let chunk_size = chunk_size.clamp(1, 10_000);
        let mut total = 0u64;
        let mut deliver = |rows: Vec<MatchPairDto>| -> Result<()> {
            let filtered = filter_export_rows_in_memory(
                &rows,
                filter.min_confidence,
                filter.levels,
                filter.rejected_pairs,
            );
            if filtered.is_empty() {
                return Ok(());
            }
            total += filtered.len() as u64;
            let owned: Vec<MatchPairDto> = filtered.into_iter().cloned().collect();
            on_chunk(&owned)
        };

        if self.job_rows_spilled(job_id)? {
            let sqlite = self
                .sqlite
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Spilled results require SQLite result storage"))?;
            sqlite.for_each_result_rows(
                job_id,
                filter.min_confidence,
                filter.levels,
                chunk_size,
                deliver,
            )?;
        } else {
            let rows = self.load_in_memory_rows(job_id)?;
            for chunk in rows.chunks(chunk_size) {
                deliver(chunk.to_vec())?;
            }
        }
        Ok(total)
    }

    /// Whether any exported rows would include cascade level metadata.
    pub fn export_has_cascade_levels(&self, job_id: &str) -> Result<bool> {
        if self.summary(job_id).is_none() {
            bail!("Unknown job id: {}", job_id);
        }
        if self.job_rows_spilled(job_id)? {
            let sqlite = self
                .sqlite
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Spilled results require SQLite result storage"))?;
            return sqlite.has_cascade_levels(job_id);
        }
        Ok(self
            .load_in_memory_rows(job_id)?
            .iter()
            .any(|row| row.matched_at_level.is_some()))
    }

    fn job_rows_spilled(&self, job_id: &str) -> Result<bool> {
        let g = self.inner.lock();
        if let Some(slot) = g.get(job_id) {
            return Ok(slot.spilled || job_rows_live_in_sqlite(slot));
        }
        drop(g);
        if let Some(sqlite) = &self.sqlite {
            if let Some(job) = sqlite.load_job(job_id)? {
                return Ok(job.spilled || job_rows_live_in_sqlite(&job));
            }
        }
        Ok(false)
    }

    fn load_in_memory_rows(&self, job_id: &str) -> Result<Arc<Vec<MatchPairDto>>> {
        let g = self.inner.lock();
        if let Some(slot) = g.get(job_id) {
            return Ok(Arc::clone(&slot.rows));
        }
        drop(g);
        if let Some(sqlite) = &self.sqlite {
            if let Some(job) = sqlite.load_job(job_id)? {
                if job.spilled || job_rows_live_in_sqlite(&job) {
                    bail!("Spilled job rows are not available in memory");
                }
                return Ok(job.rows);
            }
        }
        bail!("Unknown job id: {}", job_id)
    }

    fn load_rows_for_diff(&self, job_id: &str) -> Result<Arc<Vec<MatchPairDto>>> {
        let g = self.inner.lock();
        if let Some(slot) = g.get(job_id) {
            if !job_rows_live_in_sqlite(slot) {
                return Ok(Arc::clone(&slot.rows));
            }
        }
        drop(g);
        let sqlite = self.sqlite.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Diff for large spilled jobs requires SQLite storage")
        })?;
        sqlite.load_all_result_rows(job_id).map(Arc::new)
    }

    /// Implementation for the Tauri `get_results_page` command.
    pub fn page(&self, req: &ResultPageRequestDto) -> Result<ResultPageDto> {
        if req.limit == 0 || req.limit > 10_000 {
            bail!("limit must be between 1 and 10000");
        }
        validate_levels(&req.levels)?;
        let mut g = self.inner.lock();
        let spilled = g.get(&req.job_id).map(|s| s.spilled).unwrap_or(false);
        let Some(slot) = g.get_mut(&req.job_id) else {
            drop(g);
            if let Some(sqlite) = self.sqlite.as_ref() {
                if let Some(job) = sqlite.load_job(&req.job_id)? {
                    if job.spilled || job.rows.len() >= super::scale::RESULT_SPILL_ROWS {
                        return sqlite.page_results(req);
                    }
                    return page_from_rows(req, &job.rows);
                }
            }
            bail!("Unknown job id: {}", req.job_id);
        };
        slot.last_accessed_unix_ms = now_ms();
        if slot.spilled || spilled {
            let sqlite = self
                .sqlite
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Spilled results require SQLite storage"))?;
            return sqlite.page_results(req);
        }
        page_from_rows(req, &slot.rows)
    }

    fn evict_terminal_locked(&self, jobs: &mut HashMap<String, StoredJob>) {
        let active_count = jobs
            .values()
            .filter(|job| job.summary.state.is_active())
            .count();
        if active_count > 100 {
            log::warn!(
                "ResultStore has {} active jobs; active jobs are never evicted",
                active_count
            );
        }

        let mut terminal: Vec<(u64, u64, String)> = jobs
            .iter()
            .filter(|(_, job)| job.summary.state.is_terminal())
            .map(|(job_id, job)| {
                (
                    job.last_accessed_unix_ms,
                    job.summary.started_at_unix_ms,
                    job_id.clone(),
                )
            })
            .collect();

        if terminal.len() <= self.config.max_retained {
            return;
        }

        terminal.sort_by(|a, b| (a.0, a.1, &a.2).cmp(&(b.0, b.1, &b.2)));
        let evict_count = terminal.len().saturating_sub(self.config.max_retained);
        for (_, _, job_id) in terminal.into_iter().take(evict_count) {
            jobs.remove(&job_id);
        }
    }
}

fn page_from_rows(req: &ResultPageRequestDto, rows: &[MatchPairDto]) -> Result<ResultPageDto> {
    let base_filtered: Vec<&MatchPairDto> = rows
        .iter()
        .filter(|r| match req.min_confidence {
            Some(min) => r.confidence >= min,
            None => true,
        })
        .filter(|r| match req.query.as_deref() {
            Some(q) if !q.trim().is_empty() => {
                let q = q.to_lowercase();
                r.source_full_name.to_lowercase().contains(&q)
                    || r.target_full_name.to_lowercase().contains(&q)
            }
            _ => true,
        })
        .collect();

    let mut level_counts: BTreeMap<u8, u64> = BTreeMap::new();
    for row in &base_filtered {
        if let Some(level) = row.matched_at_level {
            *level_counts.entry(level).or_insert(0) += 1;
        }
    }
    let available_levels = level_counts.keys().copied().collect();

    let mut filtered: Vec<&MatchPairDto> = if req.levels.is_empty() {
        base_filtered
    } else {
        base_filtered
            .into_iter()
            .filter(|row| {
                row.matched_at_level
                    .map(|level| req.levels.contains(&level))
                    .unwrap_or(false)
            })
            .collect()
    };

    let sort_by = req.sort_by.as_deref().unwrap_or("row_id");
    let sort_dir = req.sort_dir.as_deref().unwrap_or(match sort_by {
        "confidence" => "desc",
        _ => "asc",
    });
    filtered.sort_by(|a, b| match sort_by {
        "confidence" => a
            .confidence
            .partial_cmp(&b.confidence)
            .unwrap_or(std::cmp::Ordering::Equal),
        "source_name" => a.source_full_name.cmp(&b.source_full_name),
        "target_name" => a.target_full_name.cmp(&b.target_full_name),
        _ => a.row_id.cmp(&b.row_id),
    });
    if sort_dir == "desc" {
        filtered.reverse();
    }
    let total = filtered.len() as u64;
    let start = (req.page as usize).saturating_mul(req.limit as usize);
    let end = start.saturating_add(req.limit as usize).min(filtered.len());
    let page_rows = if start >= filtered.len() {
        Vec::new()
    } else {
        filtered[start..end].iter().map(|r| (*r).clone()).collect()
    };
    Ok(ResultPageDto {
        job_id: req.job_id.clone(),
        page: req.page,
        limit: req.limit,
        total,
        available_levels,
        level_counts,
        rows: page_rows,
    })
}

fn validate_decision(decision: &str) -> Result<()> {
    match decision {
        "accepted" | "rejected" | "pending" => Ok(()),
        other => bail!("Invalid review decision: {}", other),
    }
}

fn diff_jobs(
    base_rows: &[MatchPairDto],
    compare_rows: &[MatchPairDto],
    request: &DiffJobsRequestDto,
) -> DiffResultDto {
    let base_by_key = base_rows
        .iter()
        .map(|row| (pair_key(row), row))
        .collect::<HashMap<_, _>>();
    let compare_by_key = compare_rows
        .iter()
        .map(|row| (pair_key(row), row))
        .collect::<HashMap<_, _>>();

    let mut added = compare_rows
        .iter()
        .filter(|row| !base_by_key.contains_key(&pair_key(row)))
        .cloned()
        .collect::<Vec<_>>();
    let mut removed = base_rows
        .iter()
        .filter(|row| !compare_by_key.contains_key(&pair_key(row)))
        .cloned()
        .collect::<Vec<_>>();
    let mut changed = compare_rows
        .iter()
        .filter_map(|after| {
            let before = base_by_key.get(&pair_key(after))?;
            let delta = after.confidence - before.confidence;
            (delta.abs() >= 2.0).then(|| DiffChangedRowDto {
                before: (*before).clone(),
                after: after.clone(),
                confidence_delta: delta,
            })
        })
        .collect::<Vec<_>>();

    added.sort_by_key(pair_key);
    removed.sort_by_key(pair_key);
    changed.sort_by_key(|row| pair_key(&row.after));

    DiffResultDto {
        base_job_id: request.base_job_id.clone(),
        compare_job_id: request.compare_job_id.clone(),
        added,
        removed,
        changed,
    }
}

fn pair_key(row: &MatchPairDto) -> (i64, i64) {
    (row.source_id, row.target_id)
}

fn job_rows_live_in_sqlite(job: &StoredJob) -> bool {
    job.spilled || (job.rows.is_empty() && job.summary.matches_found > 0)
}

fn filter_export_rows_in_memory<'a>(
    rows: &'a [MatchPairDto],
    min_confidence: f32,
    levels: &[u8],
    rejected_pairs: &HashSet<(i64, i64)>,
) -> Vec<&'a MatchPairDto> {
    let selected_levels: BTreeSet<u8> = levels.iter().copied().collect();
    rows.iter()
        .filter(|r| r.confidence >= min_confidence)
        .filter(|r| !rejected_pairs.contains(&(r.source_id, r.target_id)))
        .filter(|r| {
            selected_levels.is_empty()
                || r.matched_at_level
                    .map(|level| selected_levels.contains(&level))
                    .unwrap_or(false)
        })
        .collect()
}

fn decisions_path_for(results_path: &Path) -> PathBuf {
    let stem = results_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("results");
    results_path.with_file_name(format!("{stem}_decisions.sqlite3"))
}

fn review_decision_from_request(request: &SaveDecisionRequestDto) -> ReviewDecisionDto {
    ReviewDecisionDto {
        job_id: request.job_id.clone(),
        row_id: request.row_id,
        source_id: request.source_id,
        target_id: request.target_id,
        decision: request.decision.clone(),
        note: request.note.clone(),
        updated_at_unix_ms: now_ms(),
    }
}

struct DecisionsSqliteStore {
    conn: Mutex<Connection>,
}

impl DecisionsSqliteStore {
    fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "busy_timeout", 5_000i64)?;
        ensure_decisions_schema(&conn, false)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    fn is_empty(&self) -> Result<bool> {
        let count: i64 = self
            .conn
            .lock()
            .query_row("SELECT COUNT(*) FROM decisions", [], |row| row.get(0))?;
        Ok(count == 0)
    }

    fn save_decision(&self, request: &SaveDecisionRequestDto) -> Result<ReviewDecisionDto> {
        let dto = review_decision_from_request(request);
        self.save_decision_dto(&dto)?;
        Ok(dto)
    }

    fn save_decision_dto(&self, decision: &ReviewDecisionDto) -> Result<()> {
        self.conn.lock().execute(
            "INSERT INTO decisions(job_id, source_id, target_id, row_id, decision, note, updated_at_unix_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(job_id, source_id, target_id) DO UPDATE SET
                row_id = excluded.row_id,
                decision = excluded.decision,
                note = excluded.note,
                updated_at_unix_ms = excluded.updated_at_unix_ms",
            params![
                decision.job_id,
                decision.source_id,
                decision.target_id,
                decision.row_id as i64,
                decision.decision,
                decision.note,
                decision.updated_at_unix_ms as i64,
            ],
        )?;
        Ok(())
    }

    fn get_decisions(&self, job_id: &str) -> Result<Vec<ReviewDecisionDto>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT job_id, row_id, source_id, target_id, decision, note, updated_at_unix_ms
             FROM decisions WHERE job_id = ?1 ORDER BY row_id ASC",
        )?;
        let rows = stmt
            .query_map(params![job_id], |row| {
                Ok(ReviewDecisionDto {
                    job_id: row.get(0)?,
                    row_id: row.get::<_, i64>(1)? as u64,
                    source_id: row.get(2)?,
                    target_id: row.get(3)?,
                    decision: row.get(4)?,
                    note: row.get(5)?,
                    updated_at_unix_ms: row.get::<_, i64>(6)? as u64,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }
}

struct SqliteStore {
    conn: Mutex<Connection>,
}

impl SqliteStore {
    fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "busy_timeout", 5_000i64)?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS schema_version (
                key TEXT PRIMARY KEY,
                value INTEGER NOT NULL
            );
            INSERT OR IGNORE INTO schema_version(key, value) VALUES ('result_store', 1);
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                summary_json TEXT NOT NULL,
                allow_birthdate_swap INTEGER NOT NULL DEFAULT 0,
                state TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                started_at_unix_ms INTEGER NOT NULL,
                finished_at_unix_ms INTEGER,
                matches_found INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS results (
                job_id TEXT NOT NULL,
                row_id INTEGER NOT NULL,
                confidence REAL NOT NULL,
                matched_at_level INTEGER,
                source_name TEXT NOT NULL DEFAULT '',
                target_name TEXT NOT NULL DEFAULT '',
                row_json TEXT NOT NULL,
                PRIMARY KEY (job_id, row_id),
                FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_results_job_confidence ON results(job_id, confidence);
            CREATE INDEX IF NOT EXISTS idx_results_job_level ON results(job_id, matched_at_level);
            CREATE INDEX IF NOT EXISTS idx_results_job_source_name ON results(job_id, source_name);
            CREATE INDEX IF NOT EXISTS idx_results_job_target_name ON results(job_id, target_name);
            CREATE TABLE IF NOT EXISTS result_person_lookup (
                job_id TEXT NOT NULL,
                side TEXT NOT NULL,
                person_ordinal INTEGER NOT NULL,
                person_id INTEGER NOT NULL,
                person_json TEXT NOT NULL,
                PRIMARY KEY (job_id, side, person_ordinal),
                FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_result_person_lookup_person
                ON result_person_lookup(job_id, side, person_id);
            "#,
        )?;
        ensure_jobs_schema(&conn)?;
        ensure_results_schema(&conn)?;
        ensure_person_lookup_schema(&conn)?;
        ensure_decisions_schema(&conn, true)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    fn save_job(&self, job: &StoredJob) -> Result<()> {
        if job.spilled || job_rows_live_in_sqlite(job) {
            return self.upsert_job_metadata(job);
        }
        let mut conn = self.conn.lock();
        let tx = conn.transaction()?;
        upsert_job_metadata_tx(&tx, job)?;
        tx.execute(
            "DELETE FROM results WHERE job_id = ?1",
            params![job.summary.job_id],
        )?;
        insert_result_rows_tx(&tx, &job.summary.job_id, &job.rows)?;
        tx.execute(
            "DELETE FROM result_person_lookup WHERE job_id = ?1",
            params![job.summary.job_id],
        )?;
        for (side, people) in [
            ("source", &job.source_people),
            ("target", &job.target_people),
        ] {
            for (person_ordinal, person) in people.iter().enumerate() {
                tx.execute(
                    "INSERT INTO result_person_lookup(job_id, side, person_ordinal, person_id, person_json) VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![
                        job.summary.job_id,
                        side,
                        person_ordinal as i64,
                        person.id,
                        serde_json::to_string(person)?,
                    ],
                )?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    fn upsert_job_metadata(&self, job: &StoredJob) -> Result<()> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction()?;
        upsert_job_metadata_tx(&tx, job)?;
        tx.commit()?;
        Ok(())
    }

    fn append_result_rows(&self, job_id: &str, rows: &[MatchPairDto]) -> Result<()> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction()?;
        insert_result_rows_tx(&tx, job_id, rows)?;
        tx.commit()?;
        Ok(())
    }

    fn load_summary(&self, job_id: &str) -> Result<Option<JobSummaryDto>> {
        self.conn
            .lock()
            .query_row(
                "SELECT summary_json FROM jobs WHERE job_id = ?1",
                params![job_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .map(|json| serde_json::from_str(&json).map_err(Into::into))
            .transpose()
    }

    fn list_summaries(&self) -> Result<Vec<JobSummaryDto>> {
        let conn = self.conn.lock();
        let mut stmt =
            conn.prepare("SELECT summary_json FROM jobs ORDER BY started_at_unix_ms DESC")?;
        let rows = stmt
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        rows.into_iter()
            .map(|json| serde_json::from_str(&json).map_err(Into::into))
            .collect()
    }

    fn load_job(&self, job_id: &str) -> Result<Option<StoredJob>> {
        let Some(summary) = self.load_summary(job_id)? else {
            return Ok(None);
        };
        let conn = self.conn.lock();
        let allow_birthdate_swap = conn
            .query_row(
                "SELECT allow_birthdate_swap FROM jobs WHERE job_id = ?1",
                params![job_id],
                |row| row.get::<_, i64>(0),
            )
            .optional()?
            .map(|value| value != 0)
            .unwrap_or(false);
        let mut rows_stmt =
            conn.prepare("SELECT row_json FROM results WHERE job_id = ?1 ORDER BY row_id ASC")?;
        let rows = rows_stmt
            .query_map(params![job_id], |row| row.get::<_, String>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?
            .into_iter()
            .map(|json| serde_json::from_str(&json).map_err(Into::into))
            .collect::<Result<Vec<MatchPairDto>>>()?;
        let load_people = |side: &str| -> Result<Vec<Person>> {
            let mut stmt = conn.prepare(
                "SELECT person_json FROM result_person_lookup WHERE job_id = ?1 AND side = ?2 ORDER BY person_ordinal ASC",
            )?;
            stmt.query_map(params![job_id, side], |row| row.get::<_, String>(0))?
                .collect::<std::result::Result<Vec<_>, _>>()?
                .into_iter()
                .map(|json| serde_json::from_str(&json).map_err(Into::into))
                .collect()
        };
        let spilled = rows.len() >= super::scale::RESULT_SPILL_ROWS;
        Ok(Some(StoredJob {
            summary,
            allow_birthdate_swap,
            persist_result_history: true,
            rows: Arc::new(if spilled { Vec::new() } else { rows }),
            source_people: Arc::new(load_people("source")?),
            target_people: Arc::new(load_people("target")?),
            last_accessed_unix_ms: now_ms(),
            spilled,
        }))
    }

    fn load_all_result_rows(&self, job_id: &str) -> Result<Vec<MatchPairDto>> {
        let conn = self.conn.lock();
        let mut stmt =
            conn.prepare("SELECT row_json FROM results WHERE job_id = ?1 ORDER BY row_id ASC")?;
        stmt.query_map(params![job_id], |row| row.get::<_, String>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?
            .into_iter()
            .map(|json| serde_json::from_str(&json).map_err(Into::into))
            .collect()
    }

    fn has_cascade_levels(&self, job_id: &str) -> Result<bool> {
        let conn = self.conn.lock();
        let exists: Option<i64> = conn
            .query_row(
                "SELECT 1 FROM results WHERE job_id = ?1 AND matched_at_level IS NOT NULL LIMIT 1",
                params![job_id],
                |row| row.get(0),
            )
            .optional()?;
        Ok(exists.is_some())
    }

    fn for_each_result_rows(
        &self,
        job_id: &str,
        min_confidence: f32,
        levels: &[u8],
        chunk_size: usize,
        mut on_chunk: impl FnMut(Vec<MatchPairDto>) -> Result<()>,
    ) -> Result<()> {
        let conn = self.conn.lock();
        let mut where_parts = vec!["job_id = ?1".to_string()];
        let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(job_id.to_string())];
        if min_confidence > 0.0 {
            where_parts.push("confidence >= ?".to_string());
            params.push(Box::new(min_confidence));
        }
        if !levels.is_empty() {
            let placeholders = std::iter::repeat_n("?", levels.len())
                .collect::<Vec<_>>()
                .join(", ");
            where_parts.push(format!("matched_at_level IN ({placeholders})"));
            for level in levels {
                params.push(Box::new(*level as i64));
            }
        }
        let sql = format!(
            "SELECT row_json FROM results WHERE {} ORDER BY row_id ASC",
            where_parts.join(" AND ")
        );
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = stmt.query(rusqlite::params_from_iter(
            params.iter().map(|p| p.as_ref()),
        ))?;
        let mut chunk = Vec::with_capacity(chunk_size);
        while let Some(row) = rows.next()? {
            let json: String = row.get(0)?;
            let pair: MatchPairDto = serde_json::from_str(&json)?;
            chunk.push(pair);
            if chunk.len() >= chunk_size {
                on_chunk(std::mem::take(&mut chunk))?;
                chunk = Vec::with_capacity(chunk_size);
            }
        }
        if !chunk.is_empty() {
            on_chunk(chunk)?;
        }
        Ok(())
    }

    fn page_results(&self, req: &ResultPageRequestDto) -> Result<ResultPageDto> {
        validate_levels(&req.levels)?;
        let conn = self.conn.lock();
        let mut where_parts = vec!["job_id = ?1".to_string()];
        let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(req.job_id.clone())];
        if let Some(min) = req.min_confidence {
            where_parts.push("confidence >= ?".to_string());
            params.push(Box::new(min));
        }
        if let Some(q) = req.query.as_deref().filter(|q| !q.trim().is_empty()) {
            where_parts.push("(source_name LIKE ? OR target_name LIKE ?)".to_string());
            let like = format!("%{}%", q.trim());
            params.push(Box::new(like.clone()));
            params.push(Box::new(like));
        }
        let base_where_sql = where_parts.join(" AND ");
        let level_counts_sql = format!(
            "SELECT matched_at_level, COUNT(*) FROM results WHERE {base_where_sql} AND matched_at_level IS NOT NULL GROUP BY matched_at_level ORDER BY matched_at_level ASC"
        );
        let mut level_counts = BTreeMap::new();
        let mut level_stmt = conn.prepare(&level_counts_sql)?;
        let level_rows = level_stmt.query_map(
            rusqlite::params_from_iter(params.iter().map(|p| p.as_ref())),
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
        )?;
        for row in level_rows {
            let (level, count) = row?;
            if let Ok(level) = u8::try_from(level) {
                level_counts.insert(level, count.max(0) as u64);
            }
        }
        let available_levels = level_counts.keys().copied().collect();
        if !req.levels.is_empty() {
            let placeholders = std::iter::repeat_n("?", req.levels.len())
                .collect::<Vec<_>>()
                .join(", ");
            where_parts.push(format!("matched_at_level IN ({placeholders})"));
            for level in &req.levels {
                params.push(Box::new(*level as i64));
            }
        }
        let where_sql = where_parts.join(" AND ");
        let count_sql = format!("SELECT COUNT(*) FROM results WHERE {where_sql}");
        let total: i64 = conn.query_row(
            &count_sql,
            rusqlite::params_from_iter(params.iter().map(|p| p.as_ref())),
            |row| row.get(0),
        )?;
        let sort_col = match req.sort_by.as_deref() {
            Some("confidence") => "confidence",
            Some("source_name") => "source_name",
            Some("target_name") => "target_name",
            _ => "row_id",
        };
        let sort_dir = if req.sort_dir.as_deref() == Some("desc") {
            "DESC"
        } else {
            "ASC"
        };
        let offset = (req.page as i64) * (req.limit as i64);
        let data_sql = format!(
            "SELECT row_json FROM results WHERE {where_sql} ORDER BY {sort_col} {sort_dir}, row_id ASC LIMIT ? OFFSET ?"
        );
        let mut data_params: Vec<Box<dyn rusqlite::types::ToSql>> = params;
        data_params.push(Box::new(req.limit as i64));
        data_params.push(Box::new(offset));
        let mut stmt = conn.prepare(&data_sql)?;
        let rows = stmt
            .query_map(
                rusqlite::params_from_iter(data_params.iter().map(|p| p.as_ref())),
                |row| row.get::<_, String>(0),
            )?
            .collect::<std::result::Result<Vec<_>, _>>()?
            .into_iter()
            .map(|json| serde_json::from_str::<MatchPairDto>(&json).map_err(anyhow::Error::from))
            .collect::<Result<Vec<_>>>()?;
        Ok(ResultPageDto {
            job_id: req.job_id.clone(),
            rows,
            total: total.max(0) as u64,
            page: req.page,
            limit: req.limit,
            available_levels,
            level_counts,
        })
    }

    fn delete_job(&self, job_id: &str) -> Result<()> {
        self.conn
            .lock()
            .execute("DELETE FROM jobs WHERE job_id = ?1", params![job_id])?;
        Ok(())
    }

    fn save_decision(&self, request: &SaveDecisionRequestDto) -> Result<ReviewDecisionDto> {
        let updated_at = now_ms();
        self.conn.lock().execute(
            "INSERT INTO decisions(job_id, source_id, target_id, row_id, decision, note, updated_at_unix_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(job_id, source_id, target_id) DO UPDATE SET
                row_id = excluded.row_id,
                decision = excluded.decision,
                note = excluded.note,
                updated_at_unix_ms = excluded.updated_at_unix_ms",
            params![
                request.job_id,
                request.source_id,
                request.target_id,
                request.row_id as i64,
                request.decision,
                request.note,
                updated_at as i64,
            ],
        )?;
        Ok(ReviewDecisionDto {
            job_id: request.job_id.clone(),
            row_id: request.row_id,
            source_id: request.source_id,
            target_id: request.target_id,
            decision: request.decision.clone(),
            note: request.note.clone(),
            updated_at_unix_ms: updated_at,
        })
    }

    fn get_decisions(&self, job_id: &str) -> Result<Vec<ReviewDecisionDto>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT job_id, row_id, source_id, target_id, decision, note, updated_at_unix_ms
             FROM decisions WHERE job_id = ?1 ORDER BY row_id ASC",
        )?;
        let rows = stmt
            .query_map(params![job_id], |row| {
                Ok(ReviewDecisionDto {
                    job_id: row.get(0)?,
                    row_id: row.get::<_, i64>(1)? as u64,
                    source_id: row.get(2)?,
                    target_id: row.get(3)?,
                    decision: row.get(4)?,
                    note: row.get(5)?,
                    updated_at_unix_ms: row.get::<_, i64>(6)? as u64,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    fn list_all_decisions(&self) -> Result<Vec<ReviewDecisionDto>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT job_id, row_id, source_id, target_id, decision, note, updated_at_unix_ms
             FROM decisions ORDER BY job_id ASC, row_id ASC",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok(ReviewDecisionDto {
                    job_id: row.get(0)?,
                    row_id: row.get::<_, i64>(1)? as u64,
                    source_id: row.get(2)?,
                    target_id: row.get(3)?,
                    decision: row.get(4)?,
                    note: row.get(5)?,
                    updated_at_unix_ms: row.get::<_, i64>(6)? as u64,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }
}

fn upsert_job_metadata_tx(tx: &rusqlite::Transaction<'_>, job: &StoredJob) -> Result<()> {
    tx.execute(
        "INSERT INTO jobs(job_id, summary_json, allow_birthdate_swap, state, algorithm, started_at_unix_ms, finished_at_unix_ms, matches_found)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
         ON CONFLICT(job_id) DO UPDATE SET
            summary_json = excluded.summary_json,
            allow_birthdate_swap = excluded.allow_birthdate_swap,
            state = excluded.state,
            algorithm = excluded.algorithm,
            started_at_unix_ms = excluded.started_at_unix_ms,
            finished_at_unix_ms = excluded.finished_at_unix_ms,
            matches_found = excluded.matches_found",
        params![
            job.summary.job_id,
            serde_json::to_string(&job.summary)?,
            if job.allow_birthdate_swap { 1_i64 } else { 0_i64 },
            serde_json::to_string(&job.summary.state)?,
            serde_json::to_string(&job.summary.algorithm)?,
            job.summary.started_at_unix_ms as i64,
            job.summary.finished_at_unix_ms.map(|v| v as i64),
            job.summary.matches_found as i64,
        ],
    )?;
    Ok(())
}

fn insert_result_rows_tx(
    tx: &rusqlite::Transaction<'_>,
    job_id: &str,
    rows: &[MatchPairDto],
) -> Result<()> {
    for row in rows {
        tx.execute(
            "INSERT OR REPLACE INTO results(job_id, row_id, confidence, matched_at_level, source_name, target_name, row_json) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                job_id,
                row.row_id as i64,
                row.confidence,
                row.matched_at_level.map(i64::from),
                row.source_full_name,
                row.target_full_name,
                serde_json::to_string(row)?,
            ],
        )?;
    }
    Ok(())
}

fn ensure_jobs_schema(conn: &Connection) -> Result<()> {
    let mut stmt = conn.prepare("PRAGMA table_info(jobs)")?;
    let columns = stmt
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if !columns
        .iter()
        .any(|column| column == "allow_birthdate_swap")
    {
        conn.execute(
            "ALTER TABLE jobs ADD COLUMN allow_birthdate_swap INTEGER NOT NULL DEFAULT 0",
            [],
        )?;
    }
    Ok(())
}

fn ensure_results_schema(conn: &Connection) -> Result<()> {
    let mut stmt = conn.prepare("PRAGMA table_info(results)")?;
    let columns = stmt
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if !columns.iter().any(|column| column == "source_name") {
        conn.execute(
            "ALTER TABLE results ADD COLUMN source_name TEXT NOT NULL DEFAULT ''",
            [],
        )?;
        conn.execute(
            "UPDATE results SET source_name = COALESCE(json_extract(row_json, '$.source_full_name'), '')",
            [],
        )?;
    }
    if !columns.iter().any(|column| column == "target_name") {
        conn.execute(
            "ALTER TABLE results ADD COLUMN target_name TEXT NOT NULL DEFAULT ''",
            [],
        )?;
        conn.execute(
            "UPDATE results SET target_name = COALESCE(json_extract(row_json, '$.target_full_name'), '')",
            [],
        )?;
    }
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_results_job_source_name ON results(job_id, source_name)",
        [],
    )?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_results_job_target_name ON results(job_id, target_name)",
        [],
    )?;
    Ok(())
}

fn ensure_person_lookup_schema(conn: &Connection) -> Result<()> {
    let mut stmt = conn.prepare("PRAGMA table_info(result_person_lookup)")?;
    let columns = stmt
        .query_map([], |row| {
            Ok((row.get::<_, String>(1)?, row.get::<_, i64>(5)?))
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let has_person_ordinal = columns.iter().any(|(column, _)| column == "person_ordinal");
    let ordinal_is_primary = columns
        .iter()
        .any(|(column, pk)| column == "person_ordinal" && *pk > 0);
    if columns.is_empty() || (has_person_ordinal && ordinal_is_primary) {
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS result_person_lookup (
                job_id TEXT NOT NULL,
                side TEXT NOT NULL,
                person_ordinal INTEGER NOT NULL,
                person_id INTEGER NOT NULL,
                person_json TEXT NOT NULL,
                PRIMARY KEY (job_id, side, person_ordinal),
                FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_result_person_lookup_person
                ON result_person_lookup(job_id, side, person_id);
            "#,
        )?;
        return Ok(());
    }

    let existing_rows = {
        let mut stmt = conn.prepare(
            "SELECT job_id, side, person_id, person_json FROM result_person_lookup ORDER BY rowid ASC",
        )?;
        stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, String>(3)?,
            ))
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?
    };

    conn.execute("DROP TABLE IF EXISTS result_person_lookup_old", [])?;
    conn.execute(
        "ALTER TABLE result_person_lookup RENAME TO result_person_lookup_old",
        [],
    )?;
    conn.execute_batch(
        r#"
        CREATE TABLE result_person_lookup (
            job_id TEXT NOT NULL,
            side TEXT NOT NULL,
            person_ordinal INTEGER NOT NULL,
            person_id INTEGER NOT NULL,
            person_json TEXT NOT NULL,
            PRIMARY KEY (job_id, side, person_ordinal),
            FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_result_person_lookup_person
            ON result_person_lookup(job_id, side, person_id);
        "#,
    )?;
    let mut next_ordinal: HashMap<(String, String), i64> = HashMap::new();
    for (job_id, side, person_id, person_json) in existing_rows {
        let ordinal = next_ordinal
            .entry((job_id.clone(), side.clone()))
            .and_modify(|value| *value += 1)
            .or_insert(0);
        conn.execute(
            "INSERT INTO result_person_lookup(job_id, side, person_ordinal, person_id, person_json) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![job_id, side, *ordinal, person_id, person_json],
        )?;
    }
    conn.execute("DROP TABLE result_person_lookup_old", [])?;
    Ok(())
}

fn ensure_decisions_schema(conn: &Connection, enforce_job_fk: bool) -> Result<()> {
    let mut stmt = conn.prepare("PRAGMA table_info(decisions)")?;
    let columns = stmt
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if !columns.is_empty() && !columns.iter().any(|column| column == "source_id") {
        conn.execute("DROP TABLE decisions", [])?;
    }
    let fk_clause = if enforce_job_fk {
        ",\n            FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE"
    } else {
        ""
    };
    conn.execute_batch(&format!(
        r#"
        CREATE TABLE IF NOT EXISTS decisions (
            job_id TEXT NOT NULL,
            source_id INTEGER NOT NULL,
            target_id INTEGER NOT NULL,
            row_id INTEGER NOT NULL,
            decision TEXT NOT NULL,
            note TEXT,
            updated_at_unix_ms INTEGER NOT NULL,
            PRIMARY KEY (job_id, source_id, target_id){fk_clause}
        );
        CREATE INDEX IF NOT EXISTS idx_decisions_job_row ON decisions(job_id, row_id);
        "#
    ))?;
    Ok(())
}

fn validate_levels(levels: &[u8]) -> Result<()> {
    if let Some(level) = levels
        .iter()
        .copied()
        .find(|level| !(1..=11).contains(level))
    {
        bail!("level must be between 1 and 11: {}", level);
    }
    Ok(())
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run_service::dto::AlgorithmDto;

    fn mk_pair(idx: u64, conf: f32, source: &str, target: &str) -> MatchPairDto {
        MatchPairDto {
            row_id: idx,
            source_id: idx as i64,
            source_uuid: None,
            source_full_name: source.to_string(),
            source_birthdate: None,
            source_region_name: None,
            source_province_name: None,
            source_city_name: None,
            source_barangay_name: None,
            source_extra_fields: Default::default(),
            target_id: (idx + 1000) as i64,
            target_uuid: None,
            target_full_name: target.to_string(),
            target_birthdate: None,
            target_region_name: None,
            target_province_name: None,
            target_city_name: None,
            target_barangay_name: None,
            target_extra_fields: Default::default(),
            confidence: conf,
            matched_fields: Vec::new(),
            remarks: None,
            matched_at_level: None,
            match_method: None,
        }
    }

    fn sqlite_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("nm_result_store_{name}_{}.sqlite3", now_ms()))
    }

    fn mk_person(id: i64, first_name: &str, last_name: &str) -> Person {
        Person {
            id,
            uuid: None,
            first_name: Some(first_name.to_string()),
            middle_name: None,
            last_name: Some(last_name.to_string()),
            birthdate: None,
            hh_id: None,
            extra_fields: Default::default(),
        }
    }

    #[test]
    fn pagination_basic() {
        let store = ResultStore::new();
        store.reserve(
            "job-1".into(),
            AlgorithmDto::Fuzzy,
            "t1".into(),
            "t2".into(),
            0,
        );
        let rows = (0..30u64)
            .map(|i| mk_pair(i, 50.0 + i as f32, "a", "b"))
            .collect();
        store.set_rows("job-1", rows).unwrap();

        let page0 = store
            .page(&ResultPageRequestDto {
                job_id: "job-1".into(),
                page: 0,
                limit: 10,
                min_confidence: None,
                query: None,
                sort_by: Some("row_id".into()),
                sort_dir: Some("asc".into()),
                levels: Vec::new(),
            })
            .unwrap();
        assert_eq!(page0.rows.len(), 10);
        assert_eq!(page0.total, 30);
        assert_eq!(page0.rows[0].row_id, 0);

        let page1 = store
            .page(&ResultPageRequestDto {
                job_id: "job-1".into(),
                page: 2,
                limit: 10,
                min_confidence: None,
                query: None,
                sort_by: Some("row_id".into()),
                sort_dir: Some("asc".into()),
                levels: Vec::new(),
            })
            .unwrap();
        assert_eq!(page1.rows.len(), 10);
        assert_eq!(page1.rows[0].row_id, 20);
    }

    #[test]
    fn pagination_filters() {
        let store = ResultStore::new();
        store.reserve(
            "job-2".into(),
            AlgorithmDto::Fuzzy,
            "t1".into(),
            "t2".into(),
            0,
        );
        let rows = vec![
            mk_pair(0, 60.0, "Alice Smith", "Bob Jones"),
            mk_pair(1, 95.0, "Carol Lee", "Dan Park"),
            mk_pair(2, 99.0, "Eve Black", "Frank White"),
        ];
        store.set_rows("job-2", rows).unwrap();

        let pg = store
            .page(&ResultPageRequestDto {
                job_id: "job-2".into(),
                page: 0,
                limit: 10,
                min_confidence: Some(90.0),
                query: None,
                sort_by: Some("confidence".into()),
                sort_dir: Some("desc".into()),
                levels: Vec::new(),
            })
            .unwrap();
        assert_eq!(pg.rows.len(), 2);
        assert!(pg.rows[0].confidence >= pg.rows[1].confidence);

        let pg = store
            .page(&ResultPageRequestDto {
                job_id: "job-2".into(),
                page: 0,
                limit: 10,
                min_confidence: None,
                query: Some("alice".into()),
                sort_by: None,
                sort_dir: None,
                levels: Vec::new(),
            })
            .unwrap();
        assert_eq!(pg.rows.len(), 1);
        assert_eq!(pg.rows[0].row_id, 0);
    }

    #[test]
    fn filters_by_cascade_level_with_stable_metadata() {
        let store = ResultStore::new();
        store.reserve(
            "job-levels".into(),
            AlgorithmDto::Fuzzy,
            "t1".into(),
            "t2".into(),
            0,
        );
        let mut rows = vec![
            mk_pair(0, 91.0, "Alice Smith", "Bob Jones"),
            mk_pair(1, 92.0, "Alice Smith", "Dan Park"),
            mk_pair(2, 93.0, "Eve Black", "Frank White"),
        ];
        rows[0].matched_at_level = Some(1);
        rows[1].matched_at_level = Some(2);
        rows[2].matched_at_level = Some(2);
        store.set_rows("job-levels", rows).unwrap();

        let pg = store
            .page(&ResultPageRequestDto {
                job_id: "job-levels".into(),
                page: 0,
                limit: 10,
                min_confidence: None,
                query: Some("alice".into()),
                sort_by: Some("row_id".into()),
                sort_dir: Some("asc".into()),
                levels: vec![2],
            })
            .unwrap();

        assert_eq!(pg.rows.len(), 1);
        assert_eq!(pg.rows[0].row_id, 1);
        assert_eq!(pg.available_levels, vec![1, 2]);
        assert_eq!(pg.level_counts.get(&1), Some(&1));
        assert_eq!(pg.level_counts.get(&2), Some(&1));

        let err = store
            .page(&ResultPageRequestDto {
                job_id: "job-levels".into(),
                page: 0,
                limit: 10,
                min_confidence: None,
                query: None,
                sort_by: None,
                sort_dir: None,
                levels: vec![12],
            })
            .unwrap_err();
        assert!(err.to_string().contains("between 1 and 11"));
    }

    #[test]
    fn spilled_sqlite_page_matches_in_memory_filters() {
        let path = sqlite_path("spill_filters");
        let store = ResultStore::with_sqlite_path(ResultStoreConfig::default(), &path).unwrap();
        store.reserve(
            "spill-filter".into(),
            AlgorithmDto::Fuzzy,
            "t1".into(),
            "t2".into(),
            0,
        );
        store.enable_spill_mode("spill-filter").unwrap();
        let mut rows = vec![
            mk_pair(0, 91.0, "Ana Ramos", "Bob Jones"),
            mk_pair(1, 92.0, "Ana Santos", "Dan Park"),
            mk_pair(2, 93.0, "Ana Cruz", "Frank White"),
        ];
        rows[0].matched_at_level = Some(1);
        rows[1].matched_at_level = Some(2);
        rows[2].matched_at_level = Some(2);
        store.append_result_rows("spill-filter", &rows).unwrap();

        let pg = store
            .page(&ResultPageRequestDto {
                job_id: "spill-filter".into(),
                page: 0,
                limit: 10,
                min_confidence: None,
                query: Some("ana".into()),
                sort_by: Some("source_name".into()),
                sort_dir: Some("asc".into()),
                levels: vec![2],
            })
            .unwrap();

        assert_eq!(pg.rows.len(), 2);
        assert_eq!(
            pg.rows
                .iter()
                .map(|row| row.source_full_name.as_str())
                .collect::<Vec<_>>(),
            vec!["Ana Cruz", "Ana Santos"]
        );
        assert_eq!(pg.total, 2);
        assert_eq!(pg.available_levels, vec![1, 2]);
        assert_eq!(pg.level_counts.get(&1), Some(&1));
        assert_eq!(pg.level_counts.get(&2), Some(&2));
    }

    #[test]
    fn eviction_skips_active_jobs() {
        let store = ResultStore::with_config(ResultStoreConfig { max_retained: 1 });
        store.reserve(
            "active".into(),
            AlgorithmDto::Fuzzy,
            "t1".into(),
            "t2".into(),
            1,
        );
        store.reserve(
            "done-1".into(),
            AlgorithmDto::Fuzzy,
            "t1".into(),
            "t2".into(),
            2,
        );
        store.mark_finished("done-1", 0, 2);
        store.reserve(
            "done-2".into(),
            AlgorithmDto::Fuzzy,
            "t1".into(),
            "t2".into(),
            3,
        );
        store.mark_finished("done-2", 0, 3);

        assert!(store.summary("active").is_some());
        assert!(store.summary("done-1").is_none());
        assert!(store.summary("done-2").is_some());
    }

    #[test]
    fn eviction_keeps_n_most_recently_accessed() {
        let store = ResultStore::with_config(ResultStoreConfig { max_retained: 2 });
        for (idx, job_id) in ["old", "touched", "new"].into_iter().enumerate() {
            let ts = (idx + 1) as u64;
            store.reserve(
                job_id.into(),
                AlgorithmDto::Fuzzy,
                "t1".into(),
                "t2".into(),
                ts,
            );
            store.mark_finished(job_id, 0, ts);
        }
        let _ = store.summary("touched");
        store.reserve(
            "trigger".into(),
            AlgorithmDto::Fuzzy,
            "t1".into(),
            "t2".into(),
            4,
        );
        store.mark_finished("trigger", 0, 4);

        assert!(store.summary("old").is_none());
        assert!(store.summary("touched").is_some());
        assert!(store.summary("trigger").is_some());
    }

    #[test]
    fn forget_job_blocks_active() {
        let store = ResultStore::new();
        store.reserve(
            "active".into(),
            AlgorithmDto::Fuzzy,
            "t1".into(),
            "t2".into(),
            0,
        );

        assert!(store.forget_job("active").is_err());
        store.set_state("active", JobStateDto::Cancelled);
        assert!(store.forget_job("active").unwrap().is_some());
    }

    #[test]
    fn forget_job_returns_none_for_unknown_job() {
        let store = ResultStore::new();

        assert!(store.forget_job("missing").unwrap().is_none());
    }

    #[test]
    fn drop_job_alias_removes_terminal_job() {
        let store = ResultStore::new();
        store.reserve(
            "done".into(),
            AlgorithmDto::Fuzzy,
            "t1".into(),
            "t2".into(),
            0,
        );
        store.mark_finished("done", 7, 2_000);

        let removed = store.drop_job("done").unwrap();

        assert_eq!(removed.unwrap().summary.matches_found, 7);
        assert!(store.summary("done").is_none());
    }

    #[test]
    fn terminal_state_sets_finished_at_and_elapsed_secs() {
        let store = ResultStore::new();
        store.reserve(
            "failed".into(),
            AlgorithmDto::Fuzzy,
            "t1".into(),
            "t2".into(),
            1_000,
        );

        store.set_state("failed", JobStateDto::Failed);

        let summary = store.summary("failed").unwrap();
        assert_eq!(summary.state, JobStateDto::Failed);
        assert!(summary.finished_at_unix_ms.is_some());
        assert!(summary.elapsed_secs <= summary.finished_at_unix_ms.unwrap());
    }

    #[test]
    fn list_summaries_sorts_newest_jobs_first() {
        let store = ResultStore::new();
        for (job_id, started_at) in [("old", 1), ("new", 3), ("middle", 2)] {
            store.reserve(
                job_id.into(),
                AlgorithmDto::Fuzzy,
                "t1".into(),
                "t2".into(),
                started_at,
            );
        }

        let ids: Vec<String> = store
            .list_summaries()
            .into_iter()
            .map(|summary| summary.job_id)
            .collect();

        assert_eq!(ids, vec!["new", "middle", "old"]);
    }

    #[test]
    fn page_rejects_zero_and_oversized_limits() {
        let store = ResultStore::new();
        store.reserve(
            "job".into(),
            AlgorithmDto::Fuzzy,
            "t1".into(),
            "t2".into(),
            0,
        );

        for limit in [0, 10_001] {
            let err = store
                .page(&ResultPageRequestDto {
                    job_id: "job".into(),
                    page: 0,
                    limit,
                    min_confidence: None,
                    query: None,
                    sort_by: None,
                    sort_dir: None,
                    levels: Vec::new(),
                })
                .unwrap_err();
            assert!(err.to_string().contains("limit must be"));
        }
    }

    #[test]
    fn for_each_export_row_exports_spilled_job_from_sqlite() {
        let path = sqlite_path("export_spill");
        let store = ResultStore::with_sqlite_path(ResultStoreConfig::default(), &path).unwrap();
        store.reserve_with_options(
            "spill-export".into(),
            AlgorithmDto::DeterministicFnLnBd,
            "t1".into(),
            "t2".into(),
            0,
            false,
            true,
        );
        let rows: Vec<MatchPairDto> = (0..crate::run_service::scale::RESULT_SPILL_ROWS + 5)
            .map(|i| mk_pair(i as u64, 90.0, "a", "b"))
            .collect();
        let expected = rows.len() as u64;
        store.append_result_rows("spill-export", &rows).unwrap();
        store.mark_finished("spill-export", expected, 1);
        let page = store
            .page(&ResultPageRequestDto {
                job_id: "spill-export".into(),
                page: 0,
                limit: 10,
                min_confidence: None,
                query: None,
                sort_by: Some("row_id".into()),
                sort_dir: Some("asc".into()),
                levels: Vec::new(),
            })
            .unwrap();
        assert_eq!(page.total, expected, "rows must survive mark_finished");

        let rejected = HashSet::new();
        let filter = ExportRowFilter {
            min_confidence: 0.0,
            levels: &[],
            rejected_pairs: &rejected,
        };
        let mut seen = 0u64;
        let exported = store
            .for_each_export_row("spill-export", &filter, 2_000, |chunk| {
                seen += chunk.len() as u64;
                Ok(())
            })
            .unwrap();
        assert_eq!(exported, expected);
        assert_eq!(seen, expected);
        assert!(store.snapshot("spill-export").unwrap().rows.len() < rows.len());
    }

    #[test]
    fn diff_loads_spilled_rows_from_sqlite() {
        let path = sqlite_path("diff_spill");
        let store = ResultStore::with_sqlite_path(ResultStoreConfig::default(), &path).unwrap();
        for job_id in ["base-spill", "compare-spill"] {
            store.reserve_with_options(
                job_id.into(),
                AlgorithmDto::Fuzzy,
                "source".into(),
                "target".into(),
                1,
                false,
                true,
            );
        }
        let spill_rows: Vec<MatchPairDto> =
            (0..crate::run_service::scale::RESULT_SPILL_ROWS)
                .map(|i| mk_pair(i as u64, 90.0, "a", "b"))
                .collect();
        store.append_result_rows("base-spill", &spill_rows).unwrap();
        let mut added_row = mk_pair(999, 95.0, "added", "target");
        added_row.source_id = 9_000_000;
        added_row.target_id = 9_000_001;
        store
            .append_result_rows("compare-spill", std::slice::from_ref(&added_row))
            .unwrap();
        store.mark_finished("base-spill", spill_rows.len() as u64, 2);
        store.mark_finished("compare-spill", 1, 2);

        let diff = store
            .diff(&DiffJobsRequestDto {
                base_job_id: "base-spill".into(),
                compare_job_id: "compare-spill".into(),
            })
            .unwrap();
        assert_eq!(diff.added.len(), 1);
        assert_eq!(diff.added[0].source_id, 9_000_000);
        assert!(diff.removed.len() > 1_000);
    }

    #[test]
    fn append_spills_to_sqlite_without_retaining_all_rows_in_memory() {
        let path = sqlite_path("spill");
        let store = ResultStore::with_sqlite_path(ResultStoreConfig::default(), &path).unwrap();
        store.reserve(
            "spill".into(),
            AlgorithmDto::DeterministicFnLnBd,
            "t1".into(),
            "t2".into(),
            0,
        );
        let rows: Vec<MatchPairDto> = (0..crate::run_service::scale::RESULT_SPILL_ROWS + 10)
            .map(|i| mk_pair(i as u64, 90.0, "a", "b"))
            .collect();
        store.append_result_rows("spill", &rows).unwrap();
        let snapshot = store.snapshot("spill").unwrap();
        assert!(snapshot.spilled);
        assert!(snapshot.rows.len() < rows.len());
        let page = store
            .page(&ResultPageRequestDto {
                job_id: "spill".into(),
                page: 0,
                limit: 25,
                min_confidence: None,
                query: None,
                sort_by: Some("row_id".into()),
                sort_dir: Some("asc".into()),
                levels: Vec::new(),
            })
            .unwrap();
        assert_eq!(page.total, rows.len() as u64);
        assert_eq!(page.rows.len(), 25);
    }

    fn page_beyond_total_returns_empty_rows_with_total() {
        let store = ResultStore::new();
        store.reserve(
            "job".into(),
            AlgorithmDto::Fuzzy,
            "t1".into(),
            "t2".into(),
            0,
        );
        store
            .set_rows("job", (0..3).map(|i| mk_pair(i, 90.0, "a", "b")).collect())
            .unwrap();

        let page = store
            .page(&ResultPageRequestDto {
                job_id: "job".into(),
                page: 2,
                limit: 10,
                min_confidence: None,
                query: None,
                sort_by: Some("row_id".into()),
                sort_dir: Some("asc".into()),
                levels: Vec::new(),
            })
            .unwrap();

        assert_eq!(page.total, 3);
        assert!(page.rows.is_empty());
    }

    #[test]
    fn concurrent_get_results_does_not_evict() {
        let store = std::sync::Arc::new(ResultStore::with_config(ResultStoreConfig {
            max_retained: 1,
        }));
        store.reserve(
            "job".into(),
            AlgorithmDto::Fuzzy,
            "t1".into(),
            "t2".into(),
            0,
        );
        store
            .set_rows("job", (0..10).map(|i| mk_pair(i, 90.0, "a", "b")).collect())
            .unwrap();
        store.mark_finished("job", 10, 1);

        let mut threads = Vec::new();
        for _ in 0..8 {
            let store = std::sync::Arc::clone(&store);
            threads.push(std::thread::spawn(move || {
                store
                    .page(&ResultPageRequestDto {
                        job_id: "job".into(),
                        page: 0,
                        limit: 5,
                        min_confidence: None,
                        query: None,
                        sort_by: None,
                        sort_dir: None,
                        levels: Vec::new(),
                    })
                    .unwrap()
                    .rows
                    .len()
            }));
        }
        for thread in threads {
            assert_eq!(thread.join().unwrap(), 5);
        }
        assert!(store.summary("job").is_some());
    }

    #[test]
    fn sqlite_store_reloads_completed_job_after_restart() {
        let path = sqlite_path("reload");
        let store = ResultStore::with_sqlite_path(ResultStoreConfig::default(), &path).unwrap();
        store.reserve_with_options(
            "persisted".into(),
            AlgorithmDto::Fuzzy,
            "source".into(),
            "target".into(),
            10,
            true,
            true,
        );
        store
            .set_rows(
                "persisted",
                vec![mk_pair(0, 99.0, "Ana Santos", "Ana Santos")],
            )
            .unwrap();
        store.mark_finished("persisted", 1, 2_010);
        drop(store);

        let reloaded = ResultStore::with_sqlite_path(ResultStoreConfig::default(), &path).unwrap();
        let summary = reloaded.summary("persisted").unwrap();
        assert_eq!(summary.matches_found, 1);
        assert_eq!(summary.state, JobStateDto::Completed);
        assert!(
            reloaded
                .snapshot("persisted")
                .expect("persisted snapshot")
                .allow_birthdate_swap
        );

        let page = reloaded
            .page(&ResultPageRequestDto {
                job_id: "persisted".into(),
                page: 0,
                limit: 10,
                min_confidence: None,
                query: None,
                sort_by: None,
                sort_dir: None,
                levels: Vec::new(),
            })
            .unwrap();
        assert_eq!(page.total, 1);
        assert_eq!(page.rows[0].source_full_name, "Ana Santos");
    }

    #[test]
    fn sqlite_store_preserves_person_snapshots_with_duplicate_ids() {
        let path = sqlite_path("duplicate_person_ids");
        let store = ResultStore::with_sqlite_path(ResultStoreConfig::default(), &path).unwrap();
        store.reserve(
            "duplicates".into(),
            AlgorithmDto::Fuzzy,
            "source".into(),
            "target".into(),
            10,
        );

        store
            .set_person_snapshots(
                "duplicates",
                Arc::new(vec![
                    mk_person(7, "Ana", "Santos"),
                    mk_person(7, "Ana Maria", "Santos"),
                ]),
                Arc::new(vec![
                    mk_person(9, "Ben", "Reyes"),
                    mk_person(9, "Benjamin", "Reyes"),
                ]),
            )
            .unwrap();
        drop(store);

        let reloaded = ResultStore::with_sqlite_path(ResultStoreConfig::default(), &path).unwrap();
        let snapshot = reloaded.snapshot("duplicates").expect("snapshot");
        assert_eq!(snapshot.source_people.len(), 2);
        assert_eq!(snapshot.target_people.len(), 2);
        assert_eq!(snapshot.source_people[0].first_name.as_deref(), Some("Ana"));
        assert_eq!(
            snapshot.source_people[1].first_name.as_deref(),
            Some("Ana Maria")
        );
    }

    #[test]
    fn sqlite_store_skips_disk_writes_when_history_persistence_is_off() {
        let path = sqlite_path("history_off");
        let store = ResultStore::with_sqlite_path(ResultStoreConfig::default(), &path).unwrap();
        store.reserve_with_options(
            "fast".into(),
            AlgorithmDto::Fuzzy,
            "source".into(),
            "target".into(),
            10,
            false,
            false,
        );
        store
            .set_person_snapshots(
                "fast",
                Arc::new(vec![mk_person(1, "Ana", "Santos")]),
                Arc::new(vec![mk_person(2, "Ana", "Santos")]),
            )
            .unwrap();
        store
            .set_rows("fast", vec![mk_pair(0, 99.0, "Ana Santos", "Ana Santos")])
            .unwrap();
        store.mark_finished("fast", 1, 20);
        assert!(store.summary("fast").is_some());
        drop(store);

        let reloaded = ResultStore::with_sqlite_path(ResultStoreConfig::default(), &path).unwrap();
        assert!(reloaded.summary("fast").is_none());
    }

    #[test]
    fn sqlite_store_pages_evicted_jobs() {
        let path = sqlite_path("evicted");
        let store =
            ResultStore::with_sqlite_path(ResultStoreConfig { max_retained: 1 }, &path).unwrap();
        for (idx, job_id) in ["old", "new"].into_iter().enumerate() {
            let started_at = (idx + 1) as u64;
            store.reserve(
                job_id.into(),
                AlgorithmDto::Fuzzy,
                "source".into(),
                "target".into(),
                started_at,
            );
            store
                .set_rows(job_id, vec![mk_pair(0, 91.0, job_id, "target")])
                .unwrap();
            store.mark_finished(job_id, 1, started_at + 10);
        }

        assert!(store.inner.lock().get("old").is_none());
        assert!(store.summary("old").is_some());
        let page = store
            .page(&ResultPageRequestDto {
                job_id: "old".into(),
                page: 0,
                limit: 10,
                min_confidence: None,
                query: None,
                sort_by: None,
                sort_dir: None,
                levels: Vec::new(),
            })
            .unwrap();
        assert_eq!(page.rows[0].source_full_name, "old");
    }

    #[test]
    fn sqlite_forget_job_removes_persisted_job() {
        let path = sqlite_path("forget");
        let store = ResultStore::with_sqlite_path(ResultStoreConfig::default(), &path).unwrap();
        store.reserve(
            "gone".into(),
            AlgorithmDto::Fuzzy,
            "source".into(),
            "target".into(),
            1,
        );
        store
            .set_rows("gone", vec![mk_pair(0, 95.0, "a", "b")])
            .unwrap();
        store.mark_finished("gone", 1, 2);
        assert!(store.forget_job("gone").unwrap().is_some());
        drop(store);

        let reloaded = ResultStore::with_sqlite_path(ResultStoreConfig::default(), &path).unwrap();
        assert!(reloaded.summary("gone").is_none());
        assert!(
            reloaded
                .page(&ResultPageRequestDto {
                    job_id: "gone".into(),
                    page: 0,
                    limit: 10,
                    min_confidence: None,
                    query: None,
                    sort_by: None,
                    sort_dir: None,
                    levels: Vec::new(),
                })
                .is_err()
        );
    }

    #[test]
    fn sqlite_review_decisions_round_trip_and_upsert() {
        let path = sqlite_path("decisions");
        let store = ResultStore::with_sqlite_path(ResultStoreConfig::default(), &path).unwrap();
        store.reserve(
            "review".into(),
            AlgorithmDto::Fuzzy,
            "source".into(),
            "target".into(),
            1,
        );
        store
            .set_rows("review", vec![mk_pair(7, 82.0, "a", "b")])
            .unwrap();
        store.mark_finished("review", 1, 2);

        let saved = store
            .save_decision(SaveDecisionRequestDto {
                job_id: "review".into(),
                row_id: 7,
                source_id: 7,
                target_id: 1007,
                decision: "accepted".into(),
                note: Some("checked".into()),
            })
            .unwrap();
        assert_eq!(saved.decision, "accepted");

        store
            .save_decision(SaveDecisionRequestDto {
                job_id: "review".into(),
                row_id: 7,
                source_id: 7,
                target_id: 1007,
                decision: "rejected".into(),
                note: None,
            })
            .unwrap();
        let decisions = store.get_decisions("review").unwrap();
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].decision, "rejected");
        assert_eq!(decisions[0].note, None);
    }

    #[test]
    fn review_decisions_validate_job_and_value() {
        let path = sqlite_path("decision_validation");
        let store = ResultStore::with_sqlite_path(ResultStoreConfig::default(), &path).unwrap();
        let request = SaveDecisionRequestDto {
            job_id: "missing".into(),
            row_id: 0,
            source_id: 1,
            target_id: 2,
            decision: "accepted".into(),
            note: None,
        };
        assert!(
            store
                .save_decision(request)
                .unwrap_err()
                .to_string()
                .contains("Unknown job id")
        );

        store.reserve(
            "review".into(),
            AlgorithmDto::Fuzzy,
            "source".into(),
            "target".into(),
            1,
        );
        store.mark_finished("review", 0, 2);
        let bad = SaveDecisionRequestDto {
            job_id: "review".into(),
            row_id: 0,
            source_id: 1,
            target_id: 2,
            decision: "maybe".into(),
            note: None,
        };
        assert!(bad.decision == "maybe");
        assert!(
            store
                .save_decision(bad)
                .unwrap_err()
                .to_string()
                .contains("Invalid review decision")
        );
    }

    #[test]
    fn review_decisions_use_sidecar_when_results_db_missing() {
        let dir = std::env::temp_dir().join(format!(
            "nm_review_sidecar_{}_{}",
            std::process::id(),
            now_ms()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let results_path = dir.join("missing_results.sqlite3");
        let decisions_path = dir.join("review_decisions.sqlite3");
        let store = ResultStore::open_with_paths(
            ResultStoreConfig::default(),
            results_path,
            decisions_path.clone(),
        );
        store.reserve(
            "review".into(),
            AlgorithmDto::Fuzzy,
            "source".into(),
            "target".into(),
            1,
        );
        store.mark_finished("review", 1, 1);
        let saved = store
            .save_decision(SaveDecisionRequestDto {
                job_id: "review".into(),
                row_id: 1,
                source_id: 10,
                target_id: 20,
                decision: "accepted".into(),
                note: Some("sidecar".into()),
            })
            .unwrap();
        assert_eq!(saved.decision, "accepted");
        let decisions = store.get_decisions("review").unwrap();
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].note.as_deref(), Some("sidecar"));
        assert!(decisions_path.exists());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn diff_same_job_still_errors() {
        let store = ResultStore::new();
        store.reserve(
            "solo".into(),
            AlgorithmDto::Fuzzy,
            "source".into(),
            "target".into(),
            1,
        );
        store
            .set_rows("solo", vec![mk_pair(1, 90.0, "a", "b")])
            .unwrap();
        store.mark_finished("solo", 1, 2);

        let err = store
            .diff(&DiffJobsRequestDto {
                base_job_id: "solo".into(),
                compare_job_id: "solo".into(),
            })
            .unwrap_err();
        assert!(err.to_string().contains("Choose two different jobs"));
    }

    #[test]
    fn diff_in_memory_below_cap_still_works() {
        let store = ResultStore::new();
        for job_id in ["mem-base", "mem-compare"] {
            store.reserve(
                job_id.into(),
                AlgorithmDto::Fuzzy,
                "source".into(),
                "target".into(),
                1,
            );
        }
        store
            .set_rows("mem-base", vec![mk_pair(1, 90.0, "only-base", "target")])
            .unwrap();
        store
            .set_rows(
                "mem-compare",
                vec![
                    mk_pair(1, 90.0, "only-base", "target"),
                    mk_pair(2, 91.0, "added", "target"),
                ],
            )
            .unwrap();
        store.mark_finished("mem-base", 1, 2);
        store.mark_finished("mem-compare", 2, 2);

        let diff = store
            .diff(&DiffJobsRequestDto {
                base_job_id: "mem-base".into(),
                compare_job_id: "mem-compare".into(),
            })
            .unwrap();
        assert_eq!(diff.added.len(), 1);
        assert_eq!(diff.removed.len(), 0);
    }

    #[test]
    fn diff_large_spilled_guard() {
        let path = sqlite_path("diff_large_guard");
        let store = ResultStore::with_sqlite_path(ResultStoreConfig::default(), &path).unwrap();
        for job_id in ["large-base", "small-compare"] {
            store.reserve_with_options(
                job_id.into(),
                AlgorithmDto::Fuzzy,
                "source".into(),
                "target".into(),
                1,
                false,
                true,
            );
        }
        let large_rows: Vec<MatchPairDto> =
            (0..crate::run_service::scale::MAX_DIFF_ROWS + 1)
                .map(|i| mk_pair(i, 90.0, "a", "b"))
                .collect();
        store
            .append_result_rows("large-base", &large_rows)
            .unwrap();
        store
            .append_result_rows(
                "small-compare",
                std::slice::from_ref(&mk_pair(1, 90.0, "a", "b")),
            )
            .unwrap();
        store.mark_finished("large-base", large_rows.len() as u64, 2);
        store.mark_finished("small-compare", 1, 2);

        let err = store
            .diff(&DiffJobsRequestDto {
                base_job_id: "large-base".into(),
                compare_job_id: "small-compare".into(),
            })
            .unwrap_err();
        assert!(err.to_string().to_lowercase().contains("too large"));
    }

    #[test]
    fn diff_reports_added_removed_and_changed_pairs() {
        let path = sqlite_path("diff");
        let store = ResultStore::with_sqlite_path(ResultStoreConfig::default(), &path).unwrap();
        for job_id in ["base", "compare"] {
            store.reserve(
                job_id.into(),
                AlgorithmDto::Fuzzy,
                "source".into(),
                "target".into(),
                1,
            );
        }
        store
            .set_rows(
                "base",
                vec![
                    mk_pair(1, 90.0, "same", "target"),
                    mk_pair(2, 88.0, "changed", "target"),
                    mk_pair(3, 92.0, "removed", "target"),
                ],
            )
            .unwrap();
        let mut same = mk_pair(1, 90.5, "same", "target");
        same.target_id = 1001;
        let mut changed = mk_pair(2, 94.0, "changed", "target");
        changed.target_id = 1002;
        let mut added = mk_pair(4, 91.0, "added", "target");
        added.target_id = 1004;
        store
            .set_rows("compare", vec![same, changed, added])
            .unwrap();
        store.mark_finished("base", 3, 2);
        store.mark_finished("compare", 3, 2);

        let diff = store
            .diff(&DiffJobsRequestDto {
                base_job_id: "base".into(),
                compare_job_id: "compare".into(),
            })
            .unwrap();

        assert_eq!(diff.added.len(), 1);
        assert_eq!(diff.removed.len(), 1);
        assert_eq!(diff.changed.len(), 1);
        assert_eq!(diff.changed[0].confidence_delta, 6.0);
    }
}
