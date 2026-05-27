use name_matcher::import::staging::{commit_staged, drop_orphan_staging_tables, StagingPlan};
use name_matcher::import::{compute_plan_hash, validate_import_plan_staged};
use name_matcher::run_service::dto::{
    CsvImportDryRunResultDto, CsvImportJobDto, CsvImportJobPhaseDto, CsvImportRequestDto,
};
use name_matcher::run_service::CancelToken;
use sqlx::MySqlPool;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter};

const PLAN_CACHE_TTL: Duration = Duration::from_secs(3600);

struct CachedImportPlan {
    session_id: String,
    staging: StagingPlan,
    dry_run: CsvImportDryRunResultDto,
    created_at: Instant,
}

pub struct ImportJobHandle {
    pub job: SharedImportJob,
    pub cancel: CancelToken,
}

pub type SharedImportJob = Arc<Mutex<CsvImportJobDto>>;

#[derive(Default)]
pub struct ImportJobRegistry {
    handles: Mutex<HashMap<String, ImportJobHandle>>,
    active_by_session: Mutex<HashMap<String, String>>,
    plan_cache: Mutex<HashMap<String, CachedImportPlan>>,
}

impl ImportJobRegistry {
    pub fn prune_plan_cache(&self) {
        let mut cache = self.plan_cache.lock().expect("import plan cache poisoned");
        let now = Instant::now();
        cache.retain(|_, entry| now.duration_since(entry.created_at) < PLAN_CACHE_TTL);
    }

    pub fn store_plan_cache(
        &self,
        plan_hash: String,
        session_id: String,
        staging: StagingPlan,
        dry_run: CsvImportDryRunResultDto,
    ) {
        self.prune_plan_cache();
        let mut cache = self.plan_cache.lock().expect("import plan cache poisoned");
        cache.insert(
            plan_hash,
            CachedImportPlan {
                session_id,
                staging,
                dry_run,
                created_at: Instant::now(),
            },
        );
    }

    pub fn take_cached_plan(
        &self,
        plan_hash: &str,
        session_id: &str,
    ) -> AppResult<(StagingPlan, CsvImportDryRunResultDto)> {
        self.prune_plan_cache();
        let mut cache = self.plan_cache.lock().expect("import plan cache poisoned");
        let entry = cache
            .remove(plan_hash)
            .ok_or_else(|| AppError::Validation("import plan expired; run dry-run again".into()))?;
        if entry.session_id != session_id {
            return Err(AppError::Validation(
                "import plan session mismatch".into(),
            ));
        }
        Ok((entry.staging, entry.dry_run))
    }

    pub fn session_has_active(&self, session_id: &str) -> bool {
        self.active_by_session
            .lock()
            .expect("import active sessions poisoned")
            .contains_key(session_id)
    }

    pub fn register_handle(&self, job_id: String, session_id: String, handle: ImportJobHandle) {
        self.handles
            .lock()
            .expect("import handles poisoned")
            .insert(job_id.clone(), handle);
        self.active_by_session
            .lock()
            .expect("import active sessions poisoned")
            .insert(session_id, job_id);
    }

    pub fn get_job(&self, job_id: &str) -> Option<CsvImportJobDto> {
        self.handles
            .lock()
            .expect("import handles poisoned")
            .get(job_id)
            .map(|h| h.job.lock().expect("import job poisoned").clone())
    }

    pub fn cancel_job(&self, job_id: &str) -> AppResult<()> {
        let handles = self.handles.lock().expect("import handles poisoned");
        let Some(handle) = handles.get(job_id) else {
            return Err(AppError::Validation(format!(
                "Unknown CSV import job: {job_id}"
            )));
        };
        handle.cancel.cancel();
        Ok(())
    }

    pub fn clear_active_session(&self, session_id: &str) {
        self.active_by_session
            .lock()
            .expect("import active sessions poisoned")
            .remove(session_id);
    }

    pub fn finish_job(&self, job_id: &str, session_id: &str) {
        self.clear_active_session(session_id);
        let handles = self.handles.lock().expect("import handles poisoned");
        if let Some(handle) = handles.get(job_id) {
            let job = handle.job.lock().expect("import job poisoned").clone();
            if matches!(
                job.phase,
                CsvImportJobPhaseDto::Complete
                    | CsvImportJobPhaseDto::Failed
                    | CsvImportJobPhaseDto::Cancelled
            ) {
                // Keep terminal jobs for status polling.
            }
        }
    }
}

pub fn enforce_session_database(session_db: &str, target_db: &str) -> AppResult<()> {
    if session_db.trim() != target_db.trim() {
        return Err(AppError::Validation(format!(
            "target database must match connected session database ({session_db})"
        )));
    }
    Ok(())
}

pub async fn run_dry_run(
    pool: &MySqlPool,
    session_id: &str,
    session_database: &str,
    request: &CsvImportRequestDto,
    registry: &ImportJobRegistry,
) -> AppResult<CsvImportDryRunResultDto> {
    enforce_session_database(session_database, &request.target.database)?;
    let _ = drop_orphan_staging_tables(pool, session_database).await;
    let (mut dry_run, staging) = validate_import_plan_staged(pool, request)
        .await
        .map_err(|e| AppError::Validation(e.to_string()))?;
    let plan_hash = compute_plan_hash(request);
    dry_run.plan_hash = plan_hash.clone();
    registry.store_plan_cache(plan_hash, session_id.to_string(), staging, dry_run.clone());
    Ok(dry_run)
}

pub fn spawn_import(
    pool: MySqlPool,
    session_id: String,
    request: CsvImportRequestDto,
    staging: StagingPlan,
    dry_run: CsvImportDryRunResultDto,
    registry: Arc<ImportJobRegistry>,
    app: AppHandle,
) -> AppResult<CsvImportJobDto> {
    let plan_hash = request
        .plan_hash
        .clone()
        .ok_or_else(|| AppError::Validation("plan_hash is required to start import".into()))?;
    if plan_hash != dry_run.plan_hash {
        return Err(AppError::Validation(
            "plan_hash does not match dry-run result".into(),
        ));
    }

    let job_id = staging.job_id.clone();
    let shared: SharedImportJob = Arc::new(Mutex::new(CsvImportJobDto {
        job_id: job_id.clone(),
        phase: CsvImportJobPhaseDto::Importing,
        total_rows: dry_run.valid_rows,
        processed_rows: 0,
        inserted_rows: 0,
        updated_rows: 0,
        skipped_rows: 0,
        failed_rows: 0,
        current_batch: 0,
        total_batches: dry_run.estimated_batches,
        table: request.target.table.clone(),
        message: None,
        error: None,
        dry_run: Some(dry_run.clone()),
        partial_commit: false,
        destructive_step_completed: false,
        staging_table: dry_run.staging_table.clone(),
        load_method: dry_run.load_method,
    }));

    let cancel = CancelToken::new();
    let cancel_worker = cancel.clone();
    let shared_worker = Arc::clone(&shared);
    let registry_worker = Arc::clone(&registry);
    let session_id_worker = session_id.clone();
    let request_worker = request.clone();
    let staging_worker = staging.clone();
    let dry_run_worker = dry_run.clone();

    registry.register_handle(
        job_id.clone(),
        session_id.clone(),
        ImportJobHandle {
            job: Arc::clone(&shared),
            cancel: cancel.clone(),
        },
    );

    std::thread::spawn(move || {
        let rt = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(err) => {
                let mut job = shared_worker.lock().expect("import job poisoned");
                job.phase = CsvImportJobPhaseDto::Failed;
                job.error = Some(format!("import runtime failed: {err}"));
                registry_worker.finish_job(&job_id, &session_id_worker);
                return;
            }
        };
        let result = rt.block_on(commit_staged(
            &pool,
            &request_worker,
            &staging_worker,
            &dry_run_worker,
            &cancel_worker,
            |progress| {
                let mut job = shared_worker.lock().expect("import job poisoned");
                *job = progress.clone();
                let _ = app.emit("csv-import-progress", progress);
            },
        ));
        let mut job = shared_worker.lock().expect("import job poisoned");
        match result {
            Ok(final_job) => *job = final_job,
            Err(err) => {
                job.phase = CsvImportJobPhaseDto::Failed;
                job.error = Some(err.to_string());
            }
        }
        let _ = app.emit("csv-import-progress", job.clone());
        registry_worker.finish_job(&job_id, &session_id_worker);
    });

    Ok(shared.lock().expect("import job poisoned").clone())
}

use crate::error::{AppError, AppResult};
