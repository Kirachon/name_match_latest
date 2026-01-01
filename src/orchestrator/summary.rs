//! Summary report generation utilities.
//!
//! This module provides helpers for generating run summary reports.

use crate::export::xlsx_export::SummaryContext;
use crate::matching::advanced_matcher::AdvLevel;

/// Builder for SummaryContext to simplify summary creation.
#[derive(Debug, Clone)]
pub struct SummaryBuilder {
    pub db_name: String,
    pub table1: String,
    pub table2: String,
    pub total_table1: usize,
    pub total_table2: usize,
    pub matches_algo1: usize,
    pub matches_algo2: usize,
    pub matches_fuzzy: usize,
    pub overlap_count: usize,
    pub unique_algo1: usize,
    pub unique_algo2: usize,
    pub fetch_time: std::time::Duration,
    pub match1_time: std::time::Duration,
    pub match2_time: std::time::Duration,
    pub export_time: std::time::Duration,
    pub mem_used_start_mb: u64,
    pub mem_used_end_mb: u64,
    pub started_utc: chrono::DateTime<chrono::Utc>,
    pub ended_utc: chrono::DateTime<chrono::Utc>,
    pub exec_mode_algo1: Option<String>,
    pub exec_mode_algo2: Option<String>,
    pub exec_mode_fuzzy: Option<String>,
    pub algo_used: String,
    pub gpu_used: bool,
    pub gpu_total_mb: u64,
    pub gpu_free_mb_end: u64,
    pub adv_level: Option<AdvLevel>,
    pub adv_level_description: Option<String>,
}

impl Default for SummaryBuilder {
    fn default() -> Self {
        let now = chrono::Utc::now();
        Self {
            db_name: String::new(),
            table1: String::new(),
            table2: String::new(),
            total_table1: 0,
            total_table2: 0,
            matches_algo1: 0,
            matches_algo2: 0,
            matches_fuzzy: 0,
            overlap_count: 0,
            unique_algo1: 0,
            unique_algo2: 0,
            fetch_time: std::time::Duration::ZERO,
            match1_time: std::time::Duration::ZERO,
            match2_time: std::time::Duration::ZERO,
            export_time: std::time::Duration::ZERO,
            mem_used_start_mb: 0,
            mem_used_end_mb: 0,
            started_utc: now,
            ended_utc: now,
            exec_mode_algo1: None,
            exec_mode_algo2: None,
            exec_mode_fuzzy: None,
            algo_used: String::new(),
            gpu_used: false,
            gpu_total_mb: 0,
            gpu_free_mb_end: 0,
            adv_level: None,
            adv_level_description: None,
        }
    }
}

impl SummaryBuilder {
    /// Create a new summary builder with database and table info.
    pub fn new(db_name: &str, table1: &str, table2: &str) -> Self {
        Self {
            db_name: db_name.to_string(),
            table1: table1.to_string(),
            table2: table2.to_string(),
            ..Default::default()
        }
    }

    /// Set table counts.
    pub fn with_counts(mut self, total1: usize, total2: usize) -> Self {
        self.total_table1 = total1;
        self.total_table2 = total2;
        self
    }

    /// Set run timestamps.
    pub fn with_timestamps(
        mut self,
        started: chrono::DateTime<chrono::Utc>,
        ended: chrono::DateTime<chrono::Utc>,
    ) -> Self {
        self.started_utc = started;
        self.ended_utc = ended;
        self
    }

    /// Set algorithm info.
    pub fn with_algo(mut self, algo_used: &str, gpu_used: bool) -> Self {
        self.algo_used = algo_used.to_string();
        self.gpu_used = gpu_used;
        self
    }

    /// Set advanced level info.
    pub fn with_adv_level(mut self, level: AdvLevel, description: &str) -> Self {
        self.adv_level = Some(level);
        self.adv_level_description = Some(description.to_string());
        self
    }

    /// Build the final SummaryContext.
    pub fn build(self) -> SummaryContext {
        let duration_secs = (self.ended_utc - self.started_utc).num_milliseconds() as f64 / 1000.0;
        SummaryContext {
            db_name: self.db_name,
            table1: self.table1,
            table2: self.table2,
            total_table1: self.total_table1,
            total_table2: self.total_table2,
            matches_algo1: self.matches_algo1,
            matches_algo2: self.matches_algo2,
            matches_fuzzy: self.matches_fuzzy,
            overlap_count: self.overlap_count,
            unique_algo1: self.unique_algo1,
            unique_algo2: self.unique_algo2,
            fetch_time: self.fetch_time,
            match1_time: self.match1_time,
            match2_time: self.match2_time,
            export_time: self.export_time,
            mem_used_start_mb: self.mem_used_start_mb,
            mem_used_end_mb: self.mem_used_end_mb,
            started_utc: self.started_utc,
            ended_utc: self.ended_utc,
            duration_secs,
            exec_mode_algo1: self.exec_mode_algo1,
            exec_mode_algo2: self.exec_mode_algo2,
            exec_mode_fuzzy: self.exec_mode_fuzzy,
            algo_used: self.algo_used,
            gpu_used: self.gpu_used,
            gpu_total_mb: self.gpu_total_mb,
            gpu_free_mb_end: self.gpu_free_mb_end,
            adv_level: self.adv_level,
            adv_level_description: self.adv_level_description,
        }
    }
}
