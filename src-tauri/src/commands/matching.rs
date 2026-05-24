use crate::commands::database::is_safe_ident;
use crate::error::{AppError, AppResult};
use crate::state::{AppState, TauriEventSink};
use name_matcher::db::get_person_rows;
use name_matcher::db::schema::get_person_rows_mapped;
use name_matcher::loaders::csv_loader::{load_csv_people, CsvPreviewRequestDto};
use name_matcher::models::Person;
use name_matcher::run_service::dto::{
    DataSourceKindDto, JobStateDto, JobSummaryDto, RunConfigDto, TableSelectionDto,
};
use name_matcher::run_service::{CancelToken, EventSink, RunService};
use std::sync::Arc;
use tauri::State;

#[tauri::command]
pub async fn start_matching(
    config: RunConfigDto,
    state: State<'_, Arc<AppState>>,
) -> AppResult<String> {
    // Validate selection up-front so we never spawn a worker on garbage input.
    validate_selection("source", &config.source)?;
    validate_selection("target", &config.target)?;
    if matches!(config.source.source_kind, DataSourceKindDto::Database)
        && !is_safe_ident(&config.source.table)
    {
        return Err(AppError::Validation(format!(
            "source table name contains unsafe characters: {}",
            config.source.table
        )));
    }
    if matches!(config.target.source_kind, DataSourceKindDto::Database)
        && !is_safe_ident(&config.target.table)
    {
        return Err(AppError::Validation(format!(
            "target table name contains unsafe characters: {}",
            config.target.table
        )));
    }
    validate_mapping_idents("source", config.source.column_mapping.as_ref())?;
    validate_mapping_idents("target", config.target.column_mapping.as_ref())?;
    if config.export.output_directory.trim().is_empty() {
        return Err(AppError::Validation("output directory is required".into()));
    }

    // Resolve DB sessions -> pools NOW so the worker thread does not need to
    // hold a State<'_> reference (which is not Send across .await in Tauri).
    let src_pool = if matches!(config.source.source_kind, DataSourceKindDto::Database) {
        Some(
            state
                .db
                .get(&config.source.session_id)
                .ok_or_else(|| {
                    AppError::Validation(format!(
                        "source session not found: {}",
                        config.source.session_id
                    ))
                })?
                .pool
                .clone(),
        )
    } else {
        None
    };
    let tgt_pool = if matches!(config.target.source_kind, DataSourceKindDto::Database) {
        Some(
            state
                .db
                .get(&config.target.session_id)
                .ok_or_else(|| {
                    AppError::Validation(format!(
                        "target session not found: {}",
                        config.target.session_id
                    ))
                })?
                .pool
                .clone(),
        )
    } else {
        None
    };

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
            let src = src.clone();
            let tgt = tgt.clone();
            let src_label = selection_label(&src);
            let tgt_label = selection_label(&tgt);
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| anyhow::anyhow!("tokio runtime: {e}"))?;
            let (t1, t2): (Vec<Person>, Vec<Person>) = rt.block_on(async move {
                let t1 = load_selection_rows(src_pool.as_ref(), &src).await?;
                let t2 = load_selection_rows(tgt_pool.as_ref(), &tgt).await?;
                anyhow::Ok((t1, t2))
            })?;
            Ok((t1, t2, src_label, tgt_label))
        },
    );

    let handle = RunService::start(config, registry, store, sink, loader);
    Ok(handle.job_id.clone())
}

fn validate_selection(side: &str, selection: &TableSelectionDto) -> AppResult<()> {
    match selection.source_kind {
        DataSourceKindDto::Database => {
            if selection.session_id.is_empty() || selection.table.is_empty() {
                return Err(AppError::Validation(format!("{side} database selection is required")));
            }
        }
        DataSourceKindDto::File => {
            let Some(file) = selection.file.as_ref() else {
                return Err(AppError::Validation(format!("{side} CSV file is required")));
            };
            if file.path.trim().is_empty() {
                return Err(AppError::Validation(format!("{side} CSV file is required")));
            }
        }
    }
    Ok(())
}

async fn load_selection_rows(
    pool: Option<&sqlx::MySqlPool>,
    selection: &TableSelectionDto,
) -> anyhow::Result<Vec<Person>> {
    match selection.source_kind {
        DataSourceKindDto::Database => {
            let pool = pool.ok_or_else(|| anyhow::anyhow!("database pool missing"))?;
            if selection.column_mapping.is_some() {
                get_person_rows_mapped(pool, &selection.table, selection.column_mapping.as_ref()).await
            } else {
                get_person_rows(pool, &selection.table).await
            }
        }
        DataSourceKindDto::File => {
            let file = selection
                .file
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("CSV file selection missing"))?;
            load_csv_people(
                &CsvPreviewRequestDto {
                    path: file.path.clone(),
                    encoding: file.encoding.clone(),
                    delimiter: file.delimiter.clone(),
                    date_format: file.date_format.clone(),
                },
                selection.column_mapping.as_ref(),
            )
        }
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

fn validate_mapping_idents(
    side: &str,
    mapping: Option<&name_matcher::models::ColumnMapping>,
) -> AppResult<()> {
    let Some(mapping) = mapping else {
        return Ok(());
    };
    let required = [
        ("id", mapping.id.as_str()),
        ("first_name", mapping.first_name.as_str()),
        ("last_name", mapping.last_name.as_str()),
        ("birthdate", mapping.birthdate.as_str()),
    ];
    for (field, value) in required {
        if !is_safe_ident(value) {
            return Err(AppError::Validation(format!(
                "{side} mapping field {field} contains unsafe characters: {value}"
            )));
        }
    }
    let optional = [
        ("uuid", mapping.uuid.as_deref()),
        ("middle_name", mapping.middle_name.as_deref()),
        ("hh_id", mapping.hh_id.as_deref()),
    ];
    for (field, value) in optional {
        if let Some(value) = value {
            if !is_safe_ident(value) {
                return Err(AppError::Validation(format!(
                    "{side} mapping field {field} contains unsafe characters: {value}"
                )));
            }
        }
    }
    Ok(())
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
