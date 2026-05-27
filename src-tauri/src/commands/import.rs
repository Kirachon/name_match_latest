use crate::error::{AppError, AppResult};
use crate::import_jobs::{enforce_session_database, run_dry_run, spawn_import};
use crate::state::AppState;
use name_matcher::loaders::csv_loader::{load_csv_preview, CsvPreviewDto, CsvPreviewRequestDto};
use name_matcher::run_service::dto::{
    CsvImportDryRunResultDto, CsvImportJobDto, CsvImportRequestDto,
};
use std::sync::Arc;
use tauri::State;

#[tauri::command]
pub async fn preview_csv_import(request: CsvImportRequestDto) -> AppResult<CsvPreviewDto> {
    let csv_request = CsvPreviewRequestDto {
        path: request.file.path,
        encoding: request.file.encoding,
        delimiter: request.file.delimiter,
        date_format: request.file.date_format,
    };
    tauri::async_runtime::spawn_blocking(move || load_csv_preview(&csv_request))
        .await
        .map_err(|e| AppError::Internal(format!("CSV import preview task failed: {e}")))?
        .map_err(|e| AppError::Validation(e.to_string()))
}

#[tauri::command]
pub async fn validate_csv_import_plan(
    request: CsvImportRequestDto,
    state: State<'_, Arc<AppState>>,
) -> AppResult<CsvImportDryRunResultDto> {
    let session = state
        .db
        .get(&request.target.session_id)
        .ok_or_else(|| AppError::Validation("database session not found".into()))?;
    run_dry_run(
        &session.pool,
        &session.session_id,
        &session.database,
        &request,
        &state.import_jobs,
    )
    .await
}

#[tauri::command]
pub async fn start_csv_import(
    request: CsvImportRequestDto,
    state: State<'_, Arc<AppState>>,
) -> AppResult<CsvImportJobDto> {
    let session = state
        .db
        .get(&request.target.session_id)
        .ok_or_else(|| AppError::Validation("database session not found".into()))?;
    enforce_session_database(&session.database, &request.target.database)?;
    if state.import_jobs.session_has_active(&session.session_id) {
        return Err(AppError::Validation(
            "another CSV import is already running for this database session".into(),
        ));
    }
    let plan_hash = request
        .plan_hash
        .as_deref()
        .ok_or_else(|| AppError::Validation("plan_hash is required; run dry-run first".into()))?;
    let (staging, dry_run) = state
        .import_jobs
        .take_cached_plan(plan_hash, &session.session_id)?;
    spawn_import(
        session.pool.clone(),
        session.session_id.clone(),
        request,
        staging,
        dry_run,
        Arc::clone(&state.import_jobs),
        state.app_handle.clone(),
    )
}

#[tauri::command]
pub fn get_csv_import_status(
    job_id: String,
    state: State<'_, Arc<AppState>>,
) -> AppResult<CsvImportJobDto> {
    state
        .import_jobs
        .get_job(&job_id)
        .ok_or_else(|| AppError::Validation(format!("Unknown CSV import job: {job_id}")))
}

#[tauri::command]
pub fn cancel_csv_import(job_id: String, state: State<'_, Arc<AppState>>) -> AppResult<()> {
    state.import_jobs.cancel_job(&job_id)
}
