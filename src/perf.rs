use std::time::{Duration, Instant};

/// Small structured timer for performance remediation instrumentation.
///
/// The timer is intentionally lightweight and behavior-neutral. Stage callers
/// decide where to place it; this helper only standardizes log field names.
#[derive(Debug, Clone)]
pub struct StageTimer {
    name: &'static str,
    start: Instant,
}

impl StageTimer {
    pub fn start(name: &'static str) -> Self {
        tracing::info!(stage = name, "perf_stage_start");
        Self {
            name,
            start: Instant::now(),
        }
    }

    pub fn finish(self) -> Duration {
        let elapsed = self.start.elapsed();
        tracing::info!(
            stage = self.name,
            elapsed_ms = elapsed.as_millis() as u64,
            "perf_stage_finish"
        );
        elapsed
    }
}
