//! Run-scoped result store.
//!
//! In Phase 1 (this PR) the store keeps results in process memory inside an
//! indexed `Vec<MatchPairDto>` per job. Reads happen through
//! `get_results_page` which builds a deterministic ordered slice.
//!
//! Phase 2 (post-Tauri-1) is expected to replace this with a SQLite sidecar
//! per job. The trait shape here is designed for that swap.

use super::dto::{
    AlgorithmDto, JobStateDto, JobSummaryDto, MatchPairDto, ResultPageDto, ResultPageRequestDto,
};
use crate::models::Person;
use anyhow::{Result, bail};
use parking_lot::Mutex;
use std::collections::{BTreeMap, HashMap};
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
    pub rows: Vec<MatchPairDto>,
    pub source_people: Vec<Person>,
    pub target_people: Vec<Person>,
    pub last_accessed_unix_ms: u64,
}

pub struct ResultStore {
    inner: Mutex<HashMap<String, StoredJob>>,
    config: ResultStoreConfig,
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
        }
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
                rows: Vec::new(),
                source_people: Vec::new(),
                target_people: Vec::new(),
                last_accessed_unix_ms: started_at_unix_ms,
            },
        );
        self.evict_terminal_locked(&mut g);
    }

    pub fn set_rows(&self, job_id: &str, rows: Vec<MatchPairDto>) -> Result<()> {
        let mut g = self.inner.lock();
        match g.get_mut(job_id) {
            Some(slot) => {
                slot.summary.matches_found = rows.len() as u64;
                slot.rows = rows;
                slot.last_accessed_unix_ms = now_ms();
                Ok(())
            }
            None => bail!("Unknown job id: {}", job_id),
        }
    }

    pub fn set_person_snapshots(
        &self,
        job_id: &str,
        source_people: Vec<Person>,
        target_people: Vec<Person>,
    ) -> Result<()> {
        let mut g = self.inner.lock();
        match g.get_mut(job_id) {
            Some(slot) => {
                slot.source_people = source_people;
                slot.target_people = target_people;
                slot.last_accessed_unix_ms = now_ms();
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
        }
        self.evict_terminal_locked(&mut g);
    }

    pub fn summary(&self, job_id: &str) -> Option<JobSummaryDto> {
        let mut g = self.inner.lock();
        g.get_mut(job_id).map(|s| {
            s.last_accessed_unix_ms = now_ms();
            s.summary.clone()
        })
    }

    pub fn list_summaries(&self) -> Vec<JobSummaryDto> {
        let g = self.inner.lock();
        let mut v: Vec<JobSummaryDto> = g.values().map(|s| s.summary.clone()).collect();
        v.sort_by_key(|s| std::cmp::Reverse(s.started_at_unix_ms));
        v
    }

    /// Borrow rows + a copy of summary under one lock.
    pub fn snapshot(&self, job_id: &str) -> Option<StoredJob> {
        let mut g = self.inner.lock();
        g.get_mut(job_id).map(|s| {
            s.last_accessed_unix_ms = now_ms();
            s.clone()
        })
    }

    pub fn forget_job(&self, job_id: &str) -> Result<Option<StoredJob>> {
        let mut g = self.inner.lock();
        if let Some(slot) = g.get(job_id) {
            if slot.summary.state.is_active() {
                bail!("Cannot forget active job: {}", job_id);
            }
        }
        Ok(g.remove(job_id))
    }

    pub fn drop_job(&self, job_id: &str) -> Result<Option<StoredJob>> {
        self.forget_job(job_id)
    }

    /// Implementation for the Tauri `get_results_page` command.
    pub fn page(&self, req: &ResultPageRequestDto) -> Result<ResultPageDto> {
        if req.limit == 0 || req.limit > 10_000 {
            bail!("limit must be between 1 and 10000");
        }
        validate_levels(&req.levels)?;
        let mut g = self.inner.lock();
        let slot = g
            .get_mut(&req.job_id)
            .ok_or_else(|| anyhow::anyhow!("Unknown job id: {}", req.job_id))?;
        slot.last_accessed_unix_ms = now_ms();

        let base_filtered: Vec<&MatchPairDto> = slot
            .rows
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
        let rows: Vec<MatchPairDto> = if start >= filtered.len() {
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
            rows,
        })
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
}
