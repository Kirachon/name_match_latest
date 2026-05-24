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
use anyhow::{Result, bail};
use std::collections::HashMap;
use std::sync::Mutex;

#[derive(Debug, Clone)]
pub struct StoredJob {
    pub summary: JobSummaryDto,
    pub rows: Vec<MatchPairDto>,
}

#[derive(Default)]
pub struct ResultStore {
    inner: Mutex<HashMap<String, StoredJob>>,
}

impl ResultStore {
    pub fn new() -> Self {
        Self::default()
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
        let mut g = self.inner.lock().expect("result store poisoned");
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
            },
        );
    }

    pub fn set_rows(&self, job_id: &str, rows: Vec<MatchPairDto>) -> Result<()> {
        let mut g = self.inner.lock().expect("result store poisoned");
        match g.get_mut(job_id) {
            Some(slot) => {
                slot.summary.matches_found = rows.len() as u64;
                slot.rows = rows;
                Ok(())
            }
            None => bail!("Unknown job id: {}", job_id),
        }
    }

    pub fn mark_finished(&self, job_id: &str, matches: u64, unix_ms: u64) {
        let mut g = self.inner.lock().expect("result store poisoned");
        if let Some(slot) = g.get_mut(job_id) {
            slot.summary.matches_found = matches;
            slot.summary.finished_at_unix_ms = Some(unix_ms);
            slot.summary.state = JobStateDto::Completed;
            slot.summary.elapsed_secs =
                ((unix_ms.saturating_sub(slot.summary.started_at_unix_ms)) / 1000) as u64;
        }
    }

    pub fn set_state(&self, job_id: &str, state: JobStateDto) {
        let mut g = self.inner.lock().expect("result store poisoned");
        if let Some(slot) = g.get_mut(job_id) {
            slot.summary.state = state;
        }
    }

    pub fn summary(&self, job_id: &str) -> Option<JobSummaryDto> {
        let g = self.inner.lock().expect("result store poisoned");
        g.get(job_id).map(|s| s.summary.clone())
    }

    pub fn list_summaries(&self) -> Vec<JobSummaryDto> {
        let g = self.inner.lock().expect("result store poisoned");
        let mut v: Vec<JobSummaryDto> = g.values().map(|s| s.summary.clone()).collect();
        v.sort_by_key(|s| std::cmp::Reverse(s.started_at_unix_ms));
        v
    }

    /// Borrow rows + a copy of summary under one lock.
    pub fn snapshot(&self, job_id: &str) -> Option<StoredJob> {
        let g = self.inner.lock().expect("result store poisoned");
        g.get(job_id).cloned()
    }

    pub fn drop_job(&self, job_id: &str) -> Option<StoredJob> {
        let mut g = self.inner.lock().expect("result store poisoned");
        g.remove(job_id)
    }

    /// Implementation for the Tauri `get_results_page` command.
    pub fn page(&self, req: &ResultPageRequestDto) -> Result<ResultPageDto> {
        if req.limit == 0 || req.limit > 10_000 {
            bail!("limit must be between 1 and 10000");
        }
        let g = self.inner.lock().expect("result store poisoned");
        let slot = g
            .get(&req.job_id)
            .ok_or_else(|| anyhow::anyhow!("Unknown job id: {}", req.job_id))?;

        let mut filtered: Vec<&MatchPairDto> = slot
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
            rows,
        })
    }
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
            target_id: (idx + 1000) as i64,
            target_uuid: None,
            target_full_name: target.to_string(),
            target_birthdate: None,
            confidence: conf,
            matched_fields: Vec::new(),
            matched_at_level: None,
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
            })
            .unwrap();
        assert_eq!(pg.rows.len(), 1);
        assert_eq!(pg.rows[0].row_id, 0);
    }
}
