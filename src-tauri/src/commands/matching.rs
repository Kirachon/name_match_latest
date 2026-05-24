use crate::commands::database::is_safe_ident;
use crate::error::{AppError, AppResult};
use crate::state::{AppState, TauriEventSink};
use name_matcher::db::get_person_rows;
use name_matcher::models::Person;
use name_matcher::run_service::dto::{JobStateDto, JobSummaryDto, RunConfigDto, TableSelectionDto};
use name_matcher::run_service::{CancelToken, EventSink, RunService};
use std::sync::Arc;
use tauri::State;

#[tauri::command]
pub async fn start_matching(
    config: RunConfigDto,
    state: State<'_, Arc<AppState>>,
) -> AppResult<String> {
    // Validate selection up-front so we never spawn a worker on garbage input.
    if config.source.session_id.is_empty() || config.source.table.is_empty() {
        return Err(AppError::Validation("source selection is required".into()));
    }
    if config.target.session_id.is_empty() || config.target.table.is_empty() {
        return Err(AppError::Validation("target selection is required".into()));
    }
    if !is_safe_ident(&config.source.table) {
        return Err(AppError::Validation(format!(
            "source table name contains unsafe characters: {}",
            config.source.table
        )));
    }
    if !is_safe_ident(&config.target.table) {
        return Err(AppError::Validation(format!(
            "target table name contains unsafe characters: {}",
            config.target.table
        )));
    }
    if config.export.output_directory.trim().is_empty() {
        return Err(AppError::Validation("output directory is required".into()));
    }

    // Resolve sessions -> pools NOW so the worker thread does not need to
    // hold a State<'_> reference (which is not Send across .await in Tauri).
    let src_pool = state
        .db
        .get(&config.source.session_id)
        .ok_or_else(|| {
            AppError::Validation(format!(
                "source session not found: {}",
                config.source.session_id
            ))
        })?
        .pool
        .clone();
    let tgt_pool = state
        .db
        .get(&config.target.session_id)
        .ok_or_else(|| {
            AppError::Validation(format!(
                "target session not found: {}",
                config.target.session_id
            ))
        })?
        .pool
        .clone();

    let sink: Arc<dyn EventSink> = Arc::new(TauriEventSink::new(state.app_handle.clone()));
    let registry = Arc::clone(&state.jobs);
    let store = Arc::clone(&state.results);

    // Loader closure runs on the worker thread. We block_on a private tokio
    // runtime just for the load step so we don't need a tokio handle from the
    // Tauri app. The engine itself is sync.
    let loader: name_matcher::run_service::TableLoader = Arc::new(
        move |src: &TableSelectionDto,
              tgt: &TableSelectionDto,
              _cancel: &CancelToken,
              _sink: &dyn EventSink| {
            let src_pool = src_pool.clone();
            let tgt_pool = tgt_pool.clone();
            let src_table = src.table.clone();
            let tgt_table = tgt.table.clone();
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| anyhow::anyhow!("tokio runtime: {e}"))?;
            let (t1, t2): (Vec<Person>, Vec<Person>) = rt.block_on(async move {
                let t1 = get_person_rows(&src_pool, &src_table).await?;
                let t2 = get_person_rows(&tgt_pool, &tgt_table).await?;
                anyhow::Ok((t1, t2))
            })?;
            Ok((t1, t2, src.table.clone(), tgt.table.clone()))
        },
    );

    let handle = RunService::start(config, registry, store, sink, loader);
    Ok(handle.job_id.clone())
}

#[tauri::command]
pub fn cancel_matching(job_id: String, state: State<'_, Arc<AppState>>) -> AppResult<()> {
    let handle = state
        .jobs
        .get(&job_id)
        .ok_or_else(|| AppError::Validation(format!("unknown job_id: {job_id}")))?;
    handle.cancel();
    state.results.set_state(&job_id, JobStateDto::Cancelling);
    Ok(())
}

#[tauri::command]
pub fn pause_matching(job_id: String, state: State<'_, Arc<AppState>>) -> AppResult<()> {
    let handle = state
        .jobs
        .get(&job_id)
        .ok_or_else(|| AppError::Validation(format!("unknown job_id: {job_id}")))?;
    handle
        .request_pause()
        .map_err(|e| AppError::Validation(e.to_string()))?;
    // Optimistically reflect the transitional state in the registry; the
    // worker emits the final `Paused` from inside its progress callback.
    state.results.set_state(&job_id, JobStateDto::Pausing);
    Ok(())
}

#[tauri::command]
pub fn resume_matching(job_id: String, state: State<'_, Arc<AppState>>) -> AppResult<()> {
    let handle = state
        .jobs
        .get(&job_id)
        .ok_or_else(|| AppError::Validation(format!("unknown job_id: {job_id}")))?;
    handle
        .request_resume()
        .map_err(|e| AppError::Validation(e.to_string()))?;
    state.results.set_state(&job_id, JobStateDto::Resuming);
    Ok(())
}

#[tauri::command]
pub fn get_matching_status(
    job_id: String,
    state: State<'_, Arc<AppState>>,
) -> AppResult<JobSummaryDto> {
    let summary = state
        .results
        .summary(&job_id)
        .ok_or_else(|| AppError::Validation(format!("unknown job_id: {job_id}")))?;
    let live_state = state
        .jobs
        .get(&job_id)
        .map(|h| h.state())
        .unwrap_or(summary.state);
    Ok(JobSummaryDto {
        state: live_state,
        ..summary
    })
}

#[tauri::command]
pub fn list_matching_jobs(state: State<'_, Arc<AppState>>) -> Vec<JobSummaryDto> {
    state.jobs.prune_terminal();
    state.results.list_summaries()
}

#[tauri::command]
pub fn forget_matching_job(job_id: String, state: State<'_, Arc<AppState>>) -> AppResult<()> {
    if let Some(handle) = state.jobs.get(&job_id) {
        let live_state = handle.state();
        if live_state.is_active() {
            return Err(AppError::Validation(format!(
                "cannot forget active job: {job_id}"
            )));
        }
    }

    state
        .results
        .forget_job(&job_id)
        .map_err(|e| AppError::Validation(e.to_string()))?
        .ok_or_else(|| AppError::Validation(format!("unknown job_id: {job_id}")))?;

    if let Some(handle) = state.jobs.remove(&job_id) {
        handle.join();
    }

    Ok(())
}
