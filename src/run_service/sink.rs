//! `EventSink` abstraction. The Tauri shell wraps a sink that forwards events
//! to `AppHandle::emit`. The CLI wraps a sink that pretty-prints to stdout.

use super::dto::{JobStateEventDto, LogEntryDto, PipelineStageDto, ProgressEventDto};
use std::sync::Mutex;

pub trait EventSink: Send + Sync + 'static {
    fn emit_progress(&self, evt: ProgressEventDto);
    fn emit_log(&self, entry: LogEntryDto);
    fn emit_state(&self, evt: JobStateEventDto);

    /// Convenience helper for stage transitions before / between engine phases.
    fn emit_progress_with_stage(
        &self,
        job_id: &str,
        stage: PipelineStageDto,
        processed: u64,
        total: u64,
        message: &str,
    ) {
        let _ = message;
        self.emit_progress(ProgressEventDto {
            job_id: job_id.to_string(),
            state: super::dto::JobStateDto::Running,
            stage,
            processed,
            total,
            percent: if total == 0 {
                0.0
            } else {
                (processed as f32 / total as f32) * 100.0
            },
            eta_secs: 0,
            mem_used_mb: 0,
            mem_avail_mb: 0,
            gpu_total_mb: 0,
            gpu_free_mb: 0,
            gpu_active: false,
            records_per_sec: 0.0,
            matches_found: 0,
        });
    }
}

/// Sink that drops every event. Useful for unit tests / dry-runs.
pub struct NullEventSink;

impl EventSink for NullEventSink {
    fn emit_progress(&self, _evt: ProgressEventDto) {}
    fn emit_log(&self, _entry: LogEntryDto) {}
    fn emit_state(&self, _evt: JobStateEventDto) {}
}

/// Sink that pretty-prints events to `log::*` so the CLI can reuse the same
/// run service.
pub struct ConsoleEventSink;

impl EventSink for ConsoleEventSink {
    fn emit_progress(&self, evt: ProgressEventDto) {
        log::info!(
            "[{}] stage={:?} {}/{} ({:.1}%) ETA {}s",
            &evt.job_id[..8.min(evt.job_id.len())],
            evt.stage,
            evt.processed,
            evt.total,
            evt.percent,
            evt.eta_secs
        );
    }
    fn emit_log(&self, entry: LogEntryDto) {
        match entry.level {
            super::dto::LogLevelDto::Trace => log::trace!("{}", entry.message),
            super::dto::LogLevelDto::Debug => log::debug!("{}", entry.message),
            super::dto::LogLevelDto::Info => log::info!("{}", entry.message),
            super::dto::LogLevelDto::Warn => log::warn!("{}", entry.message),
            super::dto::LogLevelDto::Error => log::error!("{}", entry.message),
        }
    }
    fn emit_state(&self, evt: JobStateEventDto) {
        log::info!(
            "[{}] state -> {:?}{}",
            &evt.job_id[..8.min(evt.job_id.len())],
            evt.state,
            evt.detail
                .as_deref()
                .map(|d| format!(": {}", d))
                .unwrap_or_default()
        );
    }
}

/// Multiplexer that fans events out to a list of sinks. Used in tests where we
/// want both a console sink and a recording sink.
pub struct MultiEventSink {
    inner: Mutex<Vec<Box<dyn EventSink>>>,
}

impl MultiEventSink {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Vec::new()),
        }
    }
    pub fn push(&self, sink: Box<dyn EventSink>) {
        self.inner.lock().expect("multisink poisoned").push(sink);
    }
}

impl Default for MultiEventSink {
    fn default() -> Self {
        Self::new()
    }
}

impl EventSink for MultiEventSink {
    fn emit_progress(&self, evt: ProgressEventDto) {
        if let Ok(g) = self.inner.lock() {
            for s in g.iter() {
                s.emit_progress(evt.clone());
            }
        }
    }
    fn emit_log(&self, entry: LogEntryDto) {
        if let Ok(g) = self.inner.lock() {
            for s in g.iter() {
                s.emit_log(entry.clone());
            }
        }
    }
    fn emit_state(&self, evt: JobStateEventDto) {
        if let Ok(g) = self.inner.lock() {
            for s in g.iter() {
                s.emit_state(evt.clone());
            }
        }
    }
}
