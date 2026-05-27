//! Shared run service used by the CLI, the legacy egui binary, and the Tauri
//! shell.
//!
//! The Tauri front-end calls `RunService::start(...)` through a typed Tauri
//! command. The CLI calls the same entry point after parsing argv into a
//! `RunConfigDto`. The legacy egui binary continues to call existing matching
//! functions, but new callers must come through this service so we do not
//! grow a second matching path.
//!
//! Architectural rules (see `docs/tauri-migration-plan.md` §"Backend Boundary"):
//! * No `RwLock` guard is held across engine work.
//! * Long matching jobs run on a dedicated OS thread, not `tokio::spawn`.
//! * Cancellation is checked at DB / batch / GPU flush / export boundaries.
//! * Progress events are emitted via an `EventSink` impl, which the Tauri
//!   shell wraps with throttling + `AppHandle::emit`.

pub mod dto;
pub mod scale;
pub mod sink;
pub mod store;

use crate::perf::StageTimer;
pub use dto::*;
pub use sink::{ConsoleEventSink, EventSink, MultiEventSink, NullEventSink};
pub use store::{ExportRowFilter, ResultStore, ResultStoreConfig, StoredJob};

/// Runtime CUDA probe used by the Tauri `cuda_diagnostics` command and by
/// the legacy egui GPU section. Returns populated `CudaDiagnosticsDto`
/// with device count, name, and VRAM telemetry. Gated by the `gpu` Cargo
/// feature — when the feature is off, returns a placeholder struct that
/// surfaces the build-time decision to the UI.
#[cfg(feature = "gpu")]
pub fn probe_cuda() -> CudaDiagnosticsDto {
    use cudarc::driver::CudaContext;
    match CudaContext::new(0) {
        Ok(ctx) => {
            let mut devices = Vec::new();
            let mut total_mb: u64 = 0;
            let mut free_mb: u64 = 0;
            #[allow(unused_unsafe)]
            unsafe {
                let mut free: usize = 0;
                let mut total: usize = 0;
                let r = cudarc::driver::sys::cuMemGetInfo_v2(
                    &mut free as *mut usize,
                    &mut total as *mut usize,
                );
                if r == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    free_mb = (free / (1024 * 1024)) as u64;
                    total_mb = (total / (1024 * 1024)) as u64;
                }
            }
            // cudarc 0.17 doesn't expose device-name on CudaContext directly;
            // synthesize a label from the VRAM size so the StatusRail can show
            // something meaningful without bringing in a second crate.
            let label = if total_mb > 0 {
                format!("CUDA Device 0 ({} MB VRAM)", total_mb)
            } else {
                "CUDA Device 0".to_string()
            };
            devices.push(label);
            drop(ctx);
            CudaDiagnosticsDto {
                gpu_feature_compiled: true,
                device_count: 1,
                devices,
                driver_version: cuda_driver_version_string(),
                error: None,
                vram_total_mb: Some(total_mb),
                vram_free_mb: Some(free_mb),
            }
        }
        Err(e) => CudaDiagnosticsDto {
            gpu_feature_compiled: true,
            device_count: 0,
            devices: Vec::new(),
            driver_version: None,
            error: Some(format!("CUDA init failed: {e}")),
            vram_total_mb: None,
            vram_free_mb: None,
        },
    }
}

#[cfg(not(feature = "gpu"))]
pub fn probe_cuda() -> CudaDiagnosticsDto {
    CudaDiagnosticsDto {
        gpu_feature_compiled: false,
        device_count: 0,
        devices: Vec::new(),
        driver_version: None,
        error: Some("Built without the `gpu` feature; rebuild with --features gpu".into()),
        vram_total_mb: None,
        vram_free_mb: None,
    }
}

/// Best-effort driver version probe via `cuDriverGetVersion`. Returns
/// `Some("12080")` style strings (CUDA's packed `1000*major + 10*minor`).
#[cfg(feature = "gpu")]
fn cuda_driver_version_string() -> Option<String> {
    let mut v: i32 = 0;
    let r = unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut v as *mut i32) };
    if r == cudarc::driver::sys::CUresult::CUDA_SUCCESS && v > 0 {
        let major = v / 1000;
        let minor = (v % 1000) / 10;
        Some(format!("{major}.{minor}"))
    } else {
        None
    }
}

use crate::matching::{ComputeBackend, MatchOptions, ProgressConfig, ProgressUpdate};
use anyhow::Context;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Cancellation token shared between the orchestrator thread and the engine
/// callbacks. Cooperative cancellation: the engine polls this at safe
/// boundaries.
#[derive(Debug, Default, Clone)]
pub struct CancelToken {
    flag: Arc<AtomicBool>,
}

impl CancelToken {
    pub fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
        }
    }
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
    }
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }
}

/// Pause token shared between the Tauri command layer and the worker
/// progress callback. When `paused` is set, the next progress callback
/// blocks in a 50 ms sleep loop until the flag is cleared or cancellation
/// is requested. Pause granularity is therefore "between progress emissions"
/// — typically batch boundaries — which is the same granularity used for
/// cancellation.
#[derive(Debug, Default, Clone)]
pub struct PauseToken {
    flag: Arc<AtomicBool>,
}

impl PauseToken {
    pub fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
        }
    }
    pub fn pause(&self) {
        self.flag.store(true, Ordering::SeqCst);
    }
    pub fn resume(&self) {
        self.flag.store(false, Ordering::SeqCst);
    }
    pub fn is_paused(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }
}

/// Handle returned by `RunService::start`. The handle owns the worker thread
/// `JoinHandle` so callers can wait for completion.
pub struct JobHandle {
    pub job_id: String,
    pub cancel: CancelToken,
    pub pause: PauseToken,
    state: Arc<Mutex<JobStateDto>>,
    started_at_unix_ms: u64,
    started_at: Instant,
    join: Mutex<Option<JoinHandle<()>>>,
}

impl JobHandle {
    pub fn state(&self) -> JobStateDto {
        *self.state.lock().expect("state mutex poisoned")
    }
    pub fn started_at_unix_ms(&self) -> u64 {
        self.started_at_unix_ms
    }
    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }
    pub fn cancel(&self) {
        let mut s = self.state.lock().expect("state mutex poisoned");
        if !s.is_terminal() && *s != JobStateDto::Cancelling {
            *s = JobStateDto::Cancelling;
        }
        drop(s);
        self.cancel.cancel();
        // If we're paused, also clear pause so the worker exits the wait loop
        // and observes the cancel signal immediately.
        self.pause.resume();
    }
    /// Request a pause. Transitions state to `Pausing` immediately; the
    /// worker will land on `Paused` when the next progress callback runs and
    /// observes the pause flag.
    pub fn request_pause(&self) -> Result<(), &'static str> {
        let mut s = self.state.lock().expect("state mutex poisoned");
        if s.is_terminal() {
            return Err("Cannot pause a terminal job");
        }
        match *s {
            JobStateDto::Running | JobStateDto::Starting | JobStateDto::Validating => {
                *s = JobStateDto::Pausing;
            }
            JobStateDto::Paused | JobStateDto::Pausing => {
                // Already paused/pausing — idempotent.
            }
            _ => return Err("Job not in a pausable state"),
        }
        drop(s);
        self.pause.pause();
        Ok(())
    }
    /// Request a resume from `Paused`. Transitions to `Resuming`; the worker
    /// transitions to `Running` once the wait loop exits.
    pub fn request_resume(&self) -> Result<(), &'static str> {
        let mut s = self.state.lock().expect("state mutex poisoned");
        if !matches!(*s, JobStateDto::Paused | JobStateDto::Pausing) {
            return Err("Job is not paused");
        }
        *s = JobStateDto::Resuming;
        drop(s);
        self.pause.resume();
        Ok(())
    }
    /// Wait for the worker thread to finish. Safe to call multiple times.
    pub fn join(&self) {
        if let Ok(mut slot) = self.join.lock() {
            if let Some(handle) = slot.take() {
                let _ = handle.join();
            }
        }
    }
}

/// Registry of active and recently-completed jobs. Lives behind a `Mutex` and
/// is the only place that holds `Arc<JobHandle>`s.
#[derive(Default)]
pub struct JobRegistry {
    jobs: Mutex<HashMap<String, Arc<JobHandle>>>,
}

impl JobRegistry {
    pub fn insert(&self, handle: Arc<JobHandle>) {
        let mut g = self.jobs.lock().expect("registry poisoned");
        g.insert(handle.job_id.clone(), handle);
    }
    pub fn get(&self, job_id: &str) -> Option<Arc<JobHandle>> {
        self.jobs
            .lock()
            .expect("registry poisoned")
            .get(job_id)
            .cloned()
    }
    pub fn remove(&self, job_id: &str) -> Option<Arc<JobHandle>> {
        self.jobs.lock().expect("registry poisoned").remove(job_id)
    }
    pub fn iter_summaries(&self) -> Vec<(String, JobStateDto)> {
        self.jobs
            .lock()
            .expect("registry poisoned")
            .iter()
            .map(|(id, h)| (id.clone(), h.state()))
            .collect()
    }
    pub fn prune_terminal(&self) -> usize {
        let mut g = self.jobs.lock().expect("registry poisoned");
        let before = g.len();
        g.retain(|_, handle| !handle.state().is_terminal());
        before.saturating_sub(g.len())
    }
}

/// The shared run service. Stateless wrapper that takes the registry, store,
/// and a `LoadFn` callback so the Tauri layer can resolve `session_id` →
/// `MySqlPool` and load `Vec<Person>` for the source / target tables.
pub struct RunService;

/// Closure invoked on the worker thread to load the two tables. Returning
/// `(source_rows, target_rows, source_label, target_label)`.
pub type TableLoader = Arc<
    dyn Fn(
            &TableSelectionDto,
            &TableSelectionDto,
            &CancelToken,
            &dyn EventSink,
        ) -> anyhow::Result<(
            Vec<crate::models::Person>,
            Vec<crate::models::Person>,
            String,
            String,
        )> + Send
        + Sync,
>;

/// Optional DB streaming runner used when scale policy selects streaming mode.
pub type DbStreamRunner = Arc<
    dyn Fn(
            &RunConfigDto,
            &str,
            &CancelToken,
            &dyn EventSink,
            Arc<ResultStore>,
        ) -> anyhow::Result<u64>
        + Send
        + Sync,
>;

impl RunService {
    /// Start a matching job. Returns a populated `JobHandle` immediately —
    /// matching runs on a dedicated OS thread.
    pub fn start(
        config: RunConfigDto,
        registry: Arc<JobRegistry>,
        store: Arc<ResultStore>,
        sink: Arc<dyn EventSink>,
        loader: TableLoader,
    ) -> Arc<JobHandle> {
        Self::start_with_streaming(config, registry, store, sink, loader, None)
    }

    pub fn start_with_streaming(
        config: RunConfigDto,
        registry: Arc<JobRegistry>,
        store: Arc<ResultStore>,
        sink: Arc<dyn EventSink>,
        loader: TableLoader,
        stream_runner: Option<DbStreamRunner>,
    ) -> Arc<JobHandle> {
        let job_id = uuid::Uuid::new_v4().to_string();
        let cancel = CancelToken::new();
        let pause = PauseToken::new();
        let state = Arc::new(Mutex::new(JobStateDto::Starting));
        let started_at_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        // Reserve store slot up-front so the frontend can poll immediately.
        store.reserve_with_options(
            job_id.clone(),
            config.algorithm,
            selection_label(&config.source),
            selection_label(&config.target),
            started_at_unix_ms,
            config.options.allow_birthdate_swap,
            config.options.persist_result_history,
        );

        let handle_state = Arc::clone(&state);
        let handle_cancel = cancel.clone();
        let handle_pause = pause.clone();
        let handle_sink = Arc::clone(&sink);
        let handle_store = Arc::clone(&store);
        let handle_id = job_id.clone();
        let handle_config = config.clone();
        let handle_loader = Arc::clone(&loader);
        let handle_stream = stream_runner;

        let join = thread::Builder::new()
            .name(format!("nm-runner-{}", &job_id[..8]))
            .spawn(move || {
                run_worker(
                    handle_id,
                    handle_config,
                    handle_state,
                    handle_cancel,
                    handle_pause,
                    handle_sink,
                    handle_store,
                    handle_loader,
                    handle_stream,
                );
            })
            .expect("failed to spawn run service worker thread");

        let handle = Arc::new(JobHandle {
            job_id,
            cancel,
            pause,
            state,
            started_at_unix_ms,
            started_at: Instant::now(),
            join: Mutex::new(Some(join)),
        });
        registry.insert(Arc::clone(&handle));
        handle
    }
}

fn selection_label(selection: &TableSelectionDto) -> String {
    match selection.source_kind {
        DataSourceKindDto::Database => selection.table.clone(),
        DataSourceKindDto::File => selection
            .file
            .as_ref()
            .map(|file| file.path.clone())
            .unwrap_or_else(|| "csv".to_string()),
    }
}

fn set_state(
    state: &Arc<Mutex<JobStateDto>>,
    sink: &dyn EventSink,
    store: Option<&ResultStore>,
    job_id: &str,
    new: JobStateDto,
) {
    {
        let mut g = state.lock().expect("state mutex poisoned");
        *g = new;
    }
    if let Some(store) = store {
        store.set_state(job_id, new);
    }
    sink.emit_state(JobStateEventDto {
        job_id: job_id.to_string(),
        state: new,
        detail: None,
    });
}

fn fail_state(
    state: &Arc<Mutex<JobStateDto>>,
    sink: &dyn EventSink,
    store: Option<&ResultStore>,
    job_id: &str,
    msg: String,
) {
    {
        let mut g = state.lock().expect("state mutex poisoned");
        *g = JobStateDto::Failed;
    }
    if let Some(store) = store {
        store.set_state(job_id, JobStateDto::Failed);
    }
    sink.emit_log(LogEntryDto {
        job_id: job_id.to_string(),
        timestamp_ms: now_ms(),
        level: LogLevelDto::Error,
        message: msg.clone(),
    });
    sink.emit_state(JobStateEventDto {
        job_id: job_id.to_string(),
        state: JobStateDto::Failed,
        detail: Some(msg),
    });
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[allow(clippy::too_many_arguments)]
fn run_worker(
    job_id: String,
    config: RunConfigDto,
    state: Arc<Mutex<JobStateDto>>,
    cancel: CancelToken,
    pause: PauseToken,
    sink: Arc<dyn EventSink>,
    store: Arc<ResultStore>,
    loader: TableLoader,
    stream_runner: Option<DbStreamRunner>,
) {
    let sink_ref: &dyn EventSink = sink.as_ref();
    if let Some(runner) = stream_runner {
        if scale::should_use_db_streaming_worker(&config) {
            sink.emit_log(LogEntryDto {
                job_id: job_id.clone(),
                timestamp_ms: now_ms(),
                level: LogLevelDto::Info,
                message: "Effective mode: streaming (partitioned DB load)".into(),
            });
            let _ = store.enable_spill_mode(&job_id);
            set_state(
                &state,
                sink_ref,
                Some(store.as_ref()),
                &job_id,
                JobStateDto::Running,
            );
            match runner(&config, &job_id, &cancel, sink_ref, Arc::clone(&store)) {
                Ok(matches) => {
                    store.mark_finished(&job_id, matches, now_ms());
                    set_state(
                        &state,
                        sink_ref,
                        Some(store.as_ref()),
                        &job_id,
                        JobStateDto::Completed,
                    );
                }
                Err(err) => {
                    if err.to_string().contains("__name_match_cancelled__") {
                        set_state(
                            &state,
                            sink_ref,
                            Some(store.as_ref()),
                            &job_id,
                            JobStateDto::Cancelled,
                        );
                    } else {
                        fail_state(
                            &state,
                            sink_ref,
                            Some(store.as_ref()),
                            &job_id,
                            format!("Streaming match failed: {err}"),
                        );
                    }
                }
            }
            return;
        }
    }
    set_state(
        &state,
        sink_ref,
        Some(store.as_ref()),
        &job_id,
        JobStateDto::Validating,
    );
    sink.emit_log(LogEntryDto {
        job_id: job_id.clone(),
        timestamp_ms: now_ms(),
        level: LogLevelDto::Info,
        message: format!(
            "Starting job {} with algorithm {:?} on {} → {}",
            &job_id[..8],
            config.algorithm,
            selection_label(&config.source),
            selection_label(&config.target)
        ),
    });

    // 1) Load tables.
    set_state(
        &state,
        sink_ref,
        Some(store.as_ref()),
        &job_id,
        JobStateDto::Running,
    );
    sink.emit_progress_with_stage(&job_id, PipelineStageDto::Load, 0, 100, "loading tables");
    let load_timer = StageTimer::start("run_service_table_load");
    let load_result = (loader)(&config.source, &config.target, &cancel, sink_ref);
    if cancel.is_cancelled() {
        sink.emit_log(LogEntryDto {
            job_id: job_id.clone(),
            timestamp_ms: now_ms(),
            level: LogLevelDto::Warn,
            message: "Cancellation requested during table load".to_string(),
        });
        set_state(
            &state,
            sink_ref,
            Some(store.as_ref()),
            &job_id,
            JobStateDto::Cancelled,
        );
        return;
    }
    let (t1, t2, src_label, tgt_label) = match load_result {
        Ok(v) => v,
        Err(e) => {
            fail_state(
                &state,
                sink_ref,
                Some(store.as_ref()),
                &job_id,
                format!("Table load failed: {e}"),
            );
            return;
        }
    };
    load_timer.finish();
    sink.emit_log(LogEntryDto {
        job_id: job_id.clone(),
        timestamp_ms: now_ms(),
        level: LogLevelDto::Info,
        message: format!(
            "Loaded {} rows from {} and {} rows from {}",
            t1.len(),
            src_label,
            t2.len(),
            tgt_label
        ),
    });
    if config.options.persist_result_history {
        let snapshot_timer = StageTimer::start("result_person_snapshot_save");
        if let Err(e) = store
            .set_person_snapshots(&job_id, t1.clone(), t2.clone())
            .context("person snapshot store write")
        {
            fail_state(
                &state,
                sink_ref,
                Some(store.as_ref()),
                &job_id,
                format!("Result store failed: {e}"),
            );
            return;
        }
        snapshot_timer.finish();
    } else {
        sink.emit_log(LogEntryDto {
            job_id: job_id.clone(),
            timestamp_ms: now_ms(),
            level: LogLevelDto::Info,
            message: "Result history persistence is off; skipping pre-match person snapshot write"
                .into(),
        });
    }

    // 2) Resolve job-local matching options. Runtime jobs use a scoped Rayon
    // pool so one run cannot mutate process-wide thread settings for another.
    let scoped_rayon_threads = resolve_rayon_threads(&config.options, config.algorithm.to_engine());
    if let Some(threads) = scoped_rayon_threads {
        sink.emit_log(LogEntryDto {
            job_id: job_id.clone(),
            timestamp_ms: now_ms(),
            level: LogLevelDto::Info,
            message: format!("Using scoped Rayon pool with {threads} worker threads"),
        });
    }

    let backend = match config.gpu.mode {
        ComputeModeDto::Cpu => ComputeBackend::Cpu,
        ComputeModeDto::Auto | ComputeModeDto::ForceGpu => ComputeBackend::Gpu,
    };

    if config.gpu.use_direct_prefilter {
        crate::matching::set_gpu_fuzzy_direct_prep(true);
    }
    if config.gpu.use_levenshtein_full_scoring {
        crate::matching::set_gpu_levenshtein_full_scoring(true);
    }
    if config.gpu.dynamic_tuning {
        crate::matching::set_dynamic_gpu_tuning(true);
    }
    let fuzzy_gate_mode = match config.gpu.fuzzy_gate_mode {
        GpuFuzzyGateModeDto::Off => crate::matching::GpuFuzzyGateMode::Off,
        GpuFuzzyGateModeDto::Shadow => crate::matching::GpuFuzzyGateMode::Shadow,
        GpuFuzzyGateModeDto::GateOnly => crate::matching::GpuFuzzyGateMode::GateOnly,
    };
    crate::matching::set_gpu_fuzzy_gate_mode(fuzzy_gate_mode);
    if matches!(fuzzy_gate_mode, crate::matching::GpuFuzzyGateMode::Off) {
        crate::matching::set_gpu_fuzzy_force(false);
    } else {
        crate::matching::set_gpu_fuzzy_metrics(true);
        crate::matching::set_gpu_fuzzy_force(true);
        sink.emit_log(LogEntryDto {
            job_id: job_id.clone(),
            timestamp_ms: now_ms(),
            level: LogLevelDto::Info,
            message: format!(
                "GPU fuzzy gate mode for L10/L11 is {}",
                fuzzy_gate_mode.as_str()
            ),
        });
    }
    if let Some(mb) = config.gpu.vram_budget_mb {
        crate::matching::set_gpu_fuzzy_prepass_budget_mb(mb as u64);
    }

    let mo = MatchOptions {
        backend,
        gpu: None,
        progress: ProgressConfig::default(),
        allow_birthdate_swap: config.options.allow_birthdate_swap,
    };

    // 3) Run the engine. Engine callbacks emit throttled progress through the
    //    sink. Cancellation and pause are both checked inside this callback
    //    at safe boundaries (between progress emissions / batches).
    set_state(
        &state,
        sink_ref,
        Some(store.as_ref()),
        &job_id,
        JobStateDto::Running,
    );
    sink.emit_log(LogEntryDto {
        job_id: job_id.clone(),
        timestamp_ms: now_ms(),
        level: LogLevelDto::Info,
        message: format!(
            "Starting match engine for {} source rows and {} target rows",
            t1.len(),
            t2.len()
        ),
    });
    sink.emit_progress_with_stage(
        &job_id,
        PipelineStageDto::Match,
        0,
        100,
        "starting match engine",
    );
    let engine_algo = config.algorithm.to_engine();
    let job_id_for_progress = job_id.clone();
    let sink_for_progress = Arc::clone(&sink);
    let state_for_progress = Arc::clone(&state);
    let cancel_for_progress = cancel.clone();
    let pause_for_progress = pause.clone();
    let last_emit = Arc::new(Mutex::new(Instant::now()));
    let total_matches = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let total_matches_progress = Arc::clone(&total_matches);

    // ---- Cascade (Deep Match) routing branch ---------------------------
    let cascade_enabled = config.cascade.as_ref().map(|c| c.enabled).unwrap_or(false);

    // Per-pair level info, populated only on cascade runs.
    let mut worker_cascade_levels: Option<Vec<u8>> = None;

    let pairs: Vec<crate::matching::MatchPair> = if cascade_enabled {
        // Mode B: cascade run via crate::matching::cascade::run_cascade_inmemory
        let casc = config.cascade.as_ref().expect("cascade option present");
        sink.emit_log(LogEntryDto {
            job_id: job_id.clone(),
            timestamp_ms: now_ms(),
            level: LogLevelDto::Info,
            message: format!(
                "Cascade (Deep Match) — running L{:?} with threshold {:.2}",
                casc.levels, casc.fuzzy_threshold
            ),
        });

        let geo_status = crate::matching::cascade::GeoColumnStatus {
            has_barangay_code: casc.has_barangay_code,
            has_city_code: casc.has_city_code,
        };
        let exclusion = if casc.exclusion_mode == "independent" {
            crate::matching::cascade::CascadeExclusionMode::Independent
        } else {
            crate::matching::cascade::CascadeExclusionMode::Exclusive
        };
        let cascade_cfg = crate::matching::cascade::CascadeConfig {
            levels: casc.levels.clone(),
            threshold: casc.fuzzy_threshold,
            allow_birthdate_swap: config.options.allow_birthdate_swap,
            missing_column_mode: crate::matching::cascade::MissingColumnMode::AutoSkip,
            base_output_path: std::env::temp_dir()
                .join(format!("nm-cascade-{}.csv", &job_id[..8]))
                .to_string_lossy()
                .into_owned(),
            exclusion_mode: exclusion,
            compute_backend: backend,
            gpu_device_id: None,
            gpu_mem_budget_mb: config.gpu.vram_budget_mb.map(|mb| mb as u64),
            write_level_csv: false,
        };

        // Throttled progress callback for cascade level transitions.
        let cascade_job_id = job_id.clone();
        let cascade_sink = Arc::clone(&sink);
        let cascade_cancel = cancel.clone();
        let cascade_pause = pause.clone();
        let cascade_state = Arc::clone(&state);
        let cascade_total_levels = if casc.levels.is_empty() {
            11usize
        } else {
            casc.levels.len()
        };
        let cascade_engine_job_id = job_id.clone();
        let cascade_engine_sink = Arc::clone(&sink);
        let cascade_engine_cancel = cancel.clone();
        let cascade_engine_pause = pause.clone();
        let cascade_engine_state = Arc::clone(&state);
        let cascade_engine_last_emit = Arc::new(Mutex::new(Instant::now()));
        let cascade_engine_progress = move |u: ProgressUpdate| {
            if cascade_engine_cancel.is_cancelled() {
                panic!("__name_match_cancelled__");
            }
            if cascade_engine_pause.is_paused() {
                {
                    let mut g = cascade_engine_state.lock().expect("state mutex poisoned");
                    if matches!(*g, JobStateDto::Pausing | JobStateDto::Running) {
                        *g = JobStateDto::Paused;
                        drop(g);
                        cascade_engine_sink.emit_state(JobStateEventDto {
                            job_id: cascade_engine_job_id.clone(),
                            state: JobStateDto::Paused,
                            detail: Some(format!(
                                "Paused during {} at {} / {} ({:.1}%)",
                                u.stage, u.processed, u.total, u.percent
                            )),
                        });
                    }
                }
                while cascade_engine_pause.is_paused() && !cascade_engine_cancel.is_cancelled() {
                    thread::sleep(Duration::from_millis(50));
                }
                if cascade_engine_cancel.is_cancelled() {
                    panic!("__name_match_cancelled__");
                }
                {
                    let mut g = cascade_engine_state.lock().expect("state mutex poisoned");
                    if matches!(*g, JobStateDto::Resuming | JobStateDto::Paused) {
                        *g = JobStateDto::Running;
                        drop(g);
                        cascade_engine_sink.emit_state(JobStateEventDto {
                            job_id: cascade_engine_job_id.clone(),
                            state: JobStateDto::Running,
                            detail: None,
                        });
                    }
                }
            }
            if let Ok(mut last) = cascade_engine_last_emit.lock() {
                if last.elapsed() < Duration::from_millis(50) && u.processed != u.total {
                    return;
                }
                *last = Instant::now();
            }
            cascade_engine_sink.emit_progress(ProgressEventDto {
                job_id: cascade_engine_job_id.clone(),
                state: JobStateDto::Running,
                stage: if u.stage.contains("gpu") {
                    PipelineStageDto::Fuzzy
                } else {
                    PipelineStageDto::Match
                },
                processed: u.processed as u64,
                total: u.total as u64,
                percent: u.percent.clamp(0.0, 100.0),
                eta_secs: u.eta_secs,
                mem_used_mb: u.mem_used_mb,
                mem_avail_mb: u.mem_avail_mb,
                gpu_total_mb: u.gpu_total_mb,
                gpu_free_mb: u.gpu_free_mb,
                gpu_active: u.gpu_active,
                records_per_sec: 0.0,
                matches_found: total_matches_progress.load(Ordering::Relaxed),
            });
        };
        let cascade_progress = move |p: crate::matching::cascade::CascadeProgress| {
            if cascade_cancel.is_cancelled() {
                panic!("__name_match_cancelled__");
            }
            if cascade_pause.is_paused() {
                {
                    let mut g = cascade_state.lock().expect("state mutex poisoned");
                    if matches!(*g, JobStateDto::Pausing | JobStateDto::Running) {
                        *g = JobStateDto::Paused;
                        drop(g);
                        cascade_sink.emit_state(JobStateEventDto {
                            job_id: cascade_job_id.clone(),
                            state: JobStateDto::Paused,
                            detail: Some(format!("Paused at cascade L{}", p.current_level)),
                        });
                    }
                }
                while cascade_pause.is_paused() && !cascade_cancel.is_cancelled() {
                    thread::sleep(Duration::from_millis(50));
                }
                if cascade_cancel.is_cancelled() {
                    panic!("__name_match_cancelled__");
                }
                {
                    let mut g = cascade_state.lock().expect("state mutex poisoned");
                    if matches!(*g, JobStateDto::Resuming | JobStateDto::Paused) {
                        *g = JobStateDto::Running;
                        drop(g);
                        cascade_sink.emit_state(JobStateEventDto {
                            job_id: cascade_job_id.clone(),
                            state: JobStateDto::Running,
                            detail: None,
                        });
                    }
                }
            }
            let pct = if p.total_levels > 0 {
                ((p.current_level as f32 - 0.5) / p.total_levels as f32) * 100.0
            } else {
                0.0
            };
            cascade_sink.emit_log(LogEntryDto {
                job_id: cascade_job_id.clone(),
                timestamp_ms: now_ms(),
                level: LogLevelDto::Info,
                message: format!(
                    "Cascade L{} ({:?}): {}",
                    p.current_level, p.phase, p.level_description
                ),
            });
            cascade_sink.emit_progress(ProgressEventDto {
                job_id: cascade_job_id.clone(),
                state: JobStateDto::Running,
                stage: PipelineStageDto::Match,
                processed: p.current_level as u64,
                total: cascade_total_levels as u64,
                percent: pct.clamp(0.0, 100.0),
                eta_secs: 0,
                mem_used_mb: 0,
                mem_avail_mb: 0,
                gpu_total_mb: 0,
                gpu_free_mb: 0,
                gpu_active: matches!(backend, ComputeBackend::Gpu),
                records_per_sec: 0.0,
                matches_found: 0,
            });
        };

        let cascade_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_in_scoped_rayon(scoped_rayon_threads, || {
                crate::matching::cascade::run_cascade_inmemory_with_engine_progress(
                    &t1,
                    &t2,
                    &cascade_cfg,
                    &geo_status,
                    cascade_progress,
                    cascade_engine_progress,
                )
            })
        }));

        if cancel.is_cancelled() {
            set_state(
                &state,
                sink_ref,
                Some(store.as_ref()),
                &job_id,
                JobStateDto::Cancelled,
            );
            return;
        }
        match cascade_result {
            Ok(Ok(cr)) => {
                // Flatten level entries into MatchPair list, attaching the
                // level number to each pair via a parallel `level_for_pair`
                // vec we'll re-use when building DTOs.
                let mut flat: Vec<(u8, crate::matching::MatchPair)> = Vec::new();
                for entry in &cr.entries {
                    if let crate::matching::cascade::CascadeLevelStatus::Completed = entry.status {
                        for p in &entry.matches {
                            flat.push((entry.level, p.clone()));
                        }
                    }
                }
                // Cleanup temp output files cascade may have written.
                for entry in &cr.entries {
                    if let Some(path) = &entry.output_path {
                        let _ = std::fs::remove_file(path);
                    }
                }
                // Stash the level info on the side; we re-attach during DTO
                // build by walking the same indices.
                worker_cascade_levels = Some(flat.iter().map(|(l, _)| *l).collect());
                flat.into_iter().map(|(_, p)| p).collect()
            }
            Ok(Err(e)) => {
                fail_state(
                    &state,
                    sink_ref,
                    Some(store.as_ref()),
                    &job_id,
                    format!("Scoped Rayon failed: {e}"),
                );
                return;
            }
            Err(panic) => {
                let msg = panic_message(&panic);
                if cancel.is_cancelled() || msg.contains("__name_match_cancelled__") {
                    set_state(
                        &state,
                        sink_ref,
                        Some(store.as_ref()),
                        &job_id,
                        JobStateDto::Cancelled,
                    );
                    return;
                }
                fail_state(
                    &state,
                    sink_ref,
                    Some(store.as_ref()),
                    &job_id,
                    format!("Cascade engine panicked: {msg}"),
                );
                return;
            }
        }
    } else {
        // Mode A: single-pass (Quick Match)
        let progress_cb = move |u: ProgressUpdate| {
            // Cancel takes priority over pause — never wait if cancelling.
            if cancel_for_progress.is_cancelled() {
                panic!("__name_match_cancelled__");
            }

            // Cooperative pause loop. When the pause flag is set, we transition
            // the public state from Pausing -> Paused (so the UI knows the engine
            // has actually stopped processing) and sleep in 50 ms increments
            // until either resume is requested (flag cleared) or cancel arrives.
            if pause_for_progress.is_paused() {
                {
                    let mut g = state_for_progress.lock().expect("state mutex poisoned");
                    if matches!(*g, JobStateDto::Pausing | JobStateDto::Running) {
                        *g = JobStateDto::Paused;
                        drop(g);
                        sink_for_progress.emit_state(JobStateEventDto {
                            job_id: job_id_for_progress.clone(),
                            state: JobStateDto::Paused,
                            detail: Some(format!(
                                "Paused at {} / {} ({:.1}%)",
                                u.processed, u.total, u.percent
                            )),
                        });
                    }
                }
                while pause_for_progress.is_paused() && !cancel_for_progress.is_cancelled() {
                    thread::sleep(Duration::from_millis(50));
                }
                if cancel_for_progress.is_cancelled() {
                    panic!("__name_match_cancelled__");
                }
                {
                    let mut g = state_for_progress.lock().expect("state mutex poisoned");
                    if matches!(*g, JobStateDto::Resuming | JobStateDto::Paused) {
                        *g = JobStateDto::Running;
                        drop(g);
                        sink_for_progress.emit_state(JobStateEventDto {
                            job_id: job_id_for_progress.clone(),
                            state: JobStateDto::Running,
                            detail: None,
                        });
                    }
                }
            }

            // Throttle to ~20Hz to match the plan’s event budget.
            if let Ok(mut last) = last_emit.lock() {
                if last.elapsed() < Duration::from_millis(50) && u.processed != u.total {
                    return;
                }
                *last = Instant::now();
            }
            let evt = ProgressEventDto {
                job_id: job_id_for_progress.clone(),
                state: JobStateDto::Running,
                stage: PipelineStageDto::Match,
                processed: u.processed as u64,
                total: u.total as u64,
                percent: u.percent,
                eta_secs: u.eta_secs,
                mem_used_mb: u.mem_used_mb,
                mem_avail_mb: u.mem_avail_mb,
                gpu_total_mb: u.gpu_total_mb,
                gpu_free_mb: u.gpu_free_mb,
                gpu_active: u.gpu_active,
                records_per_sec: 0.0,
                matches_found: total_matches_progress.load(Ordering::Relaxed),
            };
            sink_for_progress.emit_progress(evt);
        };

        let engine_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_in_scoped_rayon(scoped_rayon_threads, || {
                crate::matching::match_all_with_opts(&t1, &t2, engine_algo, mo, progress_cb)
            })
        }));
        if cancel.is_cancelled() {
            set_state(
                &state,
                sink_ref,
                Some(store.as_ref()),
                &job_id,
                JobStateDto::Cancelled,
            );
            sink.emit_log(LogEntryDto {
                job_id: job_id.clone(),
                timestamp_ms: now_ms(),
                level: LogLevelDto::Warn,
                message: "Job cancelled".to_string(),
            });
            return;
        }
        match engine_result {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => {
                fail_state(
                    &state,
                    sink_ref,
                    Some(store.as_ref()),
                    &job_id,
                    format!("Scoped Rayon failed: {e}"),
                );
                return;
            }
            Err(panic) => {
                let msg = panic_message(&panic);
                if cancel.is_cancelled() || msg.contains("__name_match_cancelled__") {
                    set_state(
                        &state,
                        sink_ref,
                        Some(store.as_ref()),
                        &job_id,
                        JobStateDto::Cancelled,
                    );
                    sink.emit_log(LogEntryDto {
                        job_id: job_id.clone(),
                        timestamp_ms: now_ms(),
                        level: LogLevelDto::Warn,
                        message: "Job cancelled".to_string(),
                    });
                    return;
                }
                fail_state(
                    &state,
                    sink_ref,
                    Some(store.as_ref()),
                    &job_id,
                    format!("Engine panicked: {msg}"),
                );
                return;
            }
        }
    };
    total_matches.store(pairs.len() as u64, Ordering::Relaxed);

    // 4) Persist into result store.
    let dto_timer = StageTimer::start("dto_conversion");
    let dtos: Vec<MatchPairDto> = pairs
        .iter()
        .enumerate()
        .map(|(idx, p)| MatchPairDto {
            row_id: idx as u64,
            source_id: p.person1.id,
            source_uuid: p.person1.uuid.clone(),
            source_full_name: full_name(&p.person1),
            source_birthdate: p.person1.birthdate.map(|d| d.to_string()),
            source_region_name: location_field(&p.person1, &["region_name", "region"]),
            source_province_name: location_field(&p.person1, &["province_name", "province"]),
            source_city_name: location_field(
                &p.person1,
                &["city_name", "city", "municipality_name", "municipality"],
            ),
            source_barangay_name: location_field(
                &p.person1,
                &["barangay_name", "barangay", "brgy_name", "brgy"],
            ),
            source_extra_fields: extra_fields_map(&p.person1),
            target_id: p.person2.id,
            target_uuid: p.person2.uuid.clone(),
            target_full_name: full_name(&p.person2),
            target_birthdate: p.person2.birthdate.map(|d| d.to_string()),
            target_region_name: location_field(&p.person2, &["region_name", "region"]),
            target_province_name: location_field(&p.person2, &["province_name", "province"]),
            target_city_name: location_field(
                &p.person2,
                &["city_name", "city", "municipality_name", "municipality"],
            ),
            target_barangay_name: location_field(
                &p.person2,
                &["barangay_name", "barangay", "brgy_name", "brgy"],
            ),
            target_extra_fields: extra_fields_map(&p.person2),
            confidence: p.confidence,
            matched_fields: p.matched_fields.clone(),
            remarks: Some(match_remarks(
                p,
                worker_cascade_levels
                    .as_ref()
                    .and_then(|v| v.get(idx).copied()),
            )),
            matched_at_level: worker_cascade_levels
                .as_ref()
                .and_then(|v| v.get(idx).copied()),
            match_method: worker_cascade_levels
                .as_ref()
                .and_then(|v| v.get(idx).copied())
                .map(|level| crate::matching::cascade::level_description(level).replace(':', " -")),
        })
        .collect();
    dto_timer.finish();
    let count = dtos.len() as u64;
    let rows_timer = StageTimer::start("result_rows_save");
    if let Err(e) = store
        .clear_rows(&job_id)
        .and_then(|_| store.append_result_rows(&job_id, &dtos))
        .context("result store write")
    {
        fail_state(
            &state,
            sink_ref,
            Some(store.as_ref()),
            &job_id,
            format!("Result store failed: {e}"),
        );
        return;
    }
    rows_timer.finish();
    sink.emit_log(LogEntryDto {
        job_id: job_id.clone(),
        timestamp_ms: now_ms(),
        level: LogLevelDto::Info,
        message: format!("Matching produced {} pairs", count),
    });

    // 5) Final progress event.
    sink.emit_progress(ProgressEventDto {
        job_id: job_id.clone(),
        state: JobStateDto::Completed,
        stage: PipelineStageDto::Idle,
        processed: count,
        total: count,
        percent: 100.0,
        eta_secs: 0,
        mem_used_mb: 0,
        mem_avail_mb: 0,
        gpu_total_mb: 0,
        gpu_free_mb: 0,
        gpu_active: false,
        records_per_sec: 0.0,
        matches_found: count,
    });
    store.mark_finished(&job_id, count, now_ms());
    set_state(
        &state,
        sink_ref,
        Some(store.as_ref()),
        &job_id,
        JobStateDto::Completed,
    );
}

pub fn match_pair_to_dto(row_id: u64, p: &crate::matching::MatchPair) -> MatchPairDto {
    MatchPairDto {
        row_id,
        source_id: p.person1.id,
        source_uuid: p.person1.uuid.clone(),
        source_full_name: full_name(&p.person1),
        source_birthdate: p.person1.birthdate.map(|d| d.to_string()),
        source_region_name: location_field(&p.person1, &["region_name", "region"]),
        source_province_name: location_field(&p.person1, &["province_name", "province"]),
        source_city_name: location_field(
            &p.person1,
            &["city_name", "city", "municipality_name", "municipality"],
        ),
        source_barangay_name: location_field(
            &p.person1,
            &["barangay_name", "barangay", "brgy_name", "brgy"],
        ),
        source_extra_fields: extra_fields_map(&p.person1),
        target_id: p.person2.id,
        target_uuid: p.person2.uuid.clone(),
        target_full_name: full_name(&p.person2),
        target_birthdate: p.person2.birthdate.map(|d| d.to_string()),
        target_region_name: location_field(&p.person2, &["region_name", "region"]),
        target_province_name: location_field(&p.person2, &["province_name", "province"]),
        target_city_name: location_field(
            &p.person2,
            &["city_name", "city", "municipality_name", "municipality"],
        ),
        target_barangay_name: location_field(
            &p.person2,
            &["barangay_name", "barangay", "brgy_name", "brgy"],
        ),
        target_extra_fields: extra_fields_map(&p.person2),
        confidence: p.confidence,
        matched_fields: p.matched_fields.clone(),
        matched_at_level: None,
        match_method: None,
        remarks: Some(match_remarks(p, None)),
    }
}

fn full_name(p: &crate::models::Person) -> String {
    let parts = [
        p.first_name.as_deref().unwrap_or("").trim(),
        p.middle_name.as_deref().unwrap_or("").trim(),
        p.last_name.as_deref().unwrap_or("").trim(),
    ];
    parts
        .iter()
        .filter(|s| !s.is_empty())
        .copied()
        .collect::<Vec<_>>()
        .join(" ")
}

fn location_field(p: &crate::models::Person, aliases: &[&str]) -> Option<String> {
    aliases.iter().find_map(|alias| {
        p.extra_fields
            .get(*alias)
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
    })
}

fn extra_fields_map(p: &crate::models::Person) -> BTreeMap<String, String> {
    p.extra_fields
        .iter()
        .filter_map(|(key, value)| {
            let value = value.trim();
            if value.is_empty() {
                None
            } else {
                Some((key.clone(), value.to_string()))
            }
        })
        .collect()
}

fn match_remarks(p: &crate::matching::MatchPair, level: Option<u8>) -> String {
    let fields = if p.matched_fields.is_empty() {
        "the available comparison fields".to_string()
    } else {
        p.matched_fields.join(", ")
    };
    if let Some(level) = level {
        let method = crate::matching::cascade::level_description(level).replace(':', " -");
        format!(
            "{method}; similar because {fields} matched or scored close enough ({:.2}% confidence).",
            p.confidence
        )
    } else if p.confidence >= 100.0 {
        format!("Direct match because {fields} matched exactly.")
    } else {
        format!(
            "Similarity match because {fields} matched or scored close enough ({:.2}% confidence).",
            p.confidence
        )
    }
}

fn run_in_scoped_rayon<T, F>(threads: Option<usize>, f: F) -> anyhow::Result<T>
where
    T: Send,
    F: FnOnce() -> T + Send,
{
    if let Some(threads) = threads.filter(|threads| *threads > 0) {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("nm-job-rayon-{i}"))
            .build()
            .context("build scoped Rayon pool")?;
        Ok(pool.install(f))
    } else {
        Ok(f())
    }
}

/// Inline copy of `orchestrator::apply_auto_optimize` so the lib does not need
/// to expose the bin-only `orchestrator` and `cli` modules.
fn auto_optimize_rayon_threads(algorithm: crate::matching::MatchingAlgorithm) -> Option<usize> {
    if let Ok(profile) = crate::optimization::SystemProfile::detect() {
        let inm = crate::optimization::calculate_inmemory_config(&profile, algorithm, false);
        if inm.rayon_threads > 0 {
            log::info!(
                "Auto-Optimize: selected {} scoped Rayon threads based on {}",
                inm.rayon_threads,
                profile
            );
            return Some(inm.rayon_threads);
        }
    } else {
        log::warn!(
            "Auto-Optimize: system detection failed; continuing without rayon thread tuning"
        );
    }
    None
}

fn resolve_rayon_threads(
    options: &MatchOptionsDto,
    algorithm: crate::matching::MatchingAlgorithm,
) -> Option<usize> {
    if let Some(threads) = options.rayon_threads {
        return Some(threads as usize);
    }
    if options.auto_optimize {
        return auto_optimize_rayon_threads(algorithm);
    }
    configured_rayon_threads_from_env()
}

fn configured_rayon_threads_from_env() -> Option<usize> {
    std::env::var("RAYON_NUM_THREADS")
        .ok()
        .or_else(|| std::env::var("NAME_MATCHER_RAYON_THREADS").ok())
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|threads| *threads > 0)
}

fn panic_message(panic: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = panic.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = panic.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}
