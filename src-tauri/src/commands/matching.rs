use crate::commands::database::is_safe_ident;
use crate::error::{AppError, AppResult};
use crate::state::{AppState, TauriEventSink};
use name_matcher::db::get_person_rows;
use name_matcher::db::schema::get_person_rows_mapped;
use name_matcher::loaders::csv_loader::{
    load_csv_people_with_options, CsvLoadOptions, CsvPreviewRequestDto,
};
use name_matcher::loaders::excel_loader::{load_excel_people, ExcelPreviewRequestDto};
use name_matcher::models::Person;
use name_matcher::matching::{PartitioningConfig, ProgressUpdate};
use name_matcher::run_service::dto::{
    DataSourceKindDto, JobStateDto, JobSummaryDto, PipelineStageDto, ProgressEventDto,
    RunConfigDto, TableSelectionDto,
};
use name_matcher::run_service::scale::{self, ScaleBlockReason};
use name_matcher::run_service::{CancelToken, DbStreamRunner, EventSink, RunService};
use std::sync::atomic::{AtomicU64, Ordering};
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
    if let Some(reason) = scale::scale_block_reason(&config) {
        let msg = match reason {
            ScaleBlockReason::MillionRowFileSource => {
                "At 1M+ rows, import CSV to MySQL first; direct file matching is not supported."
            }
            ScaleBlockReason::MillionRowCascade => {
                "Deep Match / cascade is in-memory only and is not supported for million-row runs."
            }
            ScaleBlockReason::MillionRowFuzzy => {
                "Fuzzy matching at 1M+ rows is not supported in streaming mode; use deterministic algorithms or reduce row counts."
            }
            ScaleBlockReason::MillionRowUnsupportedAlgorithm => {
                "This algorithm is not supported for million-row DB streaming runs."
            }
        };
        return Err(AppError::Validation(msg.into()));
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
    let src_pool_loader = src_pool.clone();
    let src_pool_stream = src_pool.clone();

    // Loader closure runs on the worker thread. We block_on a private tokio
    // runtime just for the load step so we don't need a tokio handle from the
    // Tauri app. The engine itself is sync.
    let loader: name_matcher::run_service::TableLoader = Arc::new(
        move |src: &TableSelectionDto,
              tgt: &TableSelectionDto,
              cancel: &CancelToken,
              _sink: &dyn EventSink| {
            let src_pool = src_pool_loader.clone();
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
                let t1 = load_selection_rows(src_pool.as_ref(), &src, cancel).await?;
                let t2 = load_selection_rows(tgt_pool.as_ref(), &tgt, cancel).await?;
                anyhow::Ok((t1, t2))
            })?;
            Ok((t1, t2, src_label, tgt_label))
        },
    );

    let stream_runner: Option<DbStreamRunner> =
        if scale::should_use_db_streaming_worker(&config) {
            let src = src_pool_stream.expect("source pool for streaming");
            let job_sink = Arc::clone(&sink);
            Some(Arc::new(
                move |run_config: &RunConfigDto,
                      job_id: &str,
                      cancel: &CancelToken,
                      _sink: &dyn EventSink,
                      store: Arc<name_matcher::run_service::ResultStore>| {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map_err(|e| anyhow::anyhow!("tokio runtime: {e}"))?;
                    let algo = run_config.algorithm.to_engine();
                    let scfg = scale::streaming_config_from_dto(&run_config.streaming);
                    let strategy = run_config
                        .streaming
                        .partition_strategy
                        .clone()
                        .unwrap_or_else(|| "last_initial".to_string());
                    let part_cfg = PartitioningConfig {
                        strategy,
                        enabled: true,
                    };
                    let next_row_id = AtomicU64::new(0);
                    let job_id_owned = job_id.to_string();
                    let cancel_flag = cancel.clone();
                    let job_sink = Arc::clone(&job_sink);
                    let progress_job_id = job_id.to_string();
                    let matches_found = Arc::new(AtomicU64::new(0));
                    let matches_found_progress = Arc::clone(&matches_found);
                    let mut pending_rows = Vec::with_capacity(1_000);
                    let count = rt.block_on(name_matcher::matching::stream_match_csv_partitioned(
                        &src,
                        &run_config.source.table,
                        &run_config.target.table,
                        algo,
                        |pair| {
                            if cancel_flag.is_cancelled() {
                                anyhow::bail!("__name_match_cancelled__");
                            }
                            let row_id = next_row_id.fetch_add(1, Ordering::Relaxed);
                            let dto = name_matcher::run_service::match_pair_to_dto(row_id, pair);
                            pending_rows.push(dto);
                            if pending_rows.len() >= 1_000 {
                                store.append_result_rows(&job_id_owned, &pending_rows)?;
                                pending_rows.clear();
                            }
                            matches_found.fetch_add(1, Ordering::Relaxed);
                            Ok(())
                        },
                        scfg,
                        move |u: ProgressUpdate| {
                            job_sink.emit_progress(ProgressEventDto {
                                job_id: progress_job_id.clone(),
                                state: JobStateDto::Running,
                                stage: PipelineStageDto::Match,
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
                                matches_found: matches_found_progress.load(Ordering::Relaxed),
                            });
                        },
                        None,
                        run_config.source.column_mapping.as_ref(),
                        run_config.target.column_mapping.as_ref(),
                        part_cfg,
                    ))?;
                    if !pending_rows.is_empty() {
                        store.append_result_rows(&job_id_owned, &pending_rows)?;
                    }
                    Ok(count as u64)
                },
            ))
        } else {
            None
        };

    let handle = RunService::start_with_streaming(config, registry, store, sink, loader, stream_runner);
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
                return Err(AppError::Validation(format!("{side} file is required")));
            };
            if file.path.trim().is_empty() {
                return Err(AppError::Validation(format!("{side} file is required")));
            }
        }
    }
    Ok(())
}

async fn load_selection_rows(
    pool: Option<&sqlx::MySqlPool>,
    selection: &TableSelectionDto,
    cancel: &CancelToken,
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
                .ok_or_else(|| anyhow::anyhow!("file selection missing"))?;
            if is_excel_path(&file.path) {
                load_excel_people(
                    &ExcelPreviewRequestDto {
                        path: file.path.clone(),
                        sheet_name: file.sheet_name.clone(),
                        date_format: file.date_format.clone(),
                    },
                    selection.column_mapping.as_ref(),
                )
            } else {
                let cancel = cancel.clone();
                let options = CsvLoadOptions {
                    should_cancel: Some(Arc::new(move || cancel.is_cancelled())),
                    ..CsvLoadOptions::default()
                };
                load_csv_people_with_options(
                    &CsvPreviewRequestDto {
                        path: file.path.clone(),
                        encoding: file.encoding.clone(),
                        delimiter: file.delimiter.clone(),
                        date_format: file.date_format.clone(),
                    },
                    selection.column_mapping.as_ref(),
                    &options,
                )
            }
        }
    }
}

fn is_excel_path(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    lower.ends_with(".xlsx") || lower.ends_with(".xls")
}

fn selection_label(selection: &TableSelectionDto) -> String {
    match selection.source_kind {
        DataSourceKindDto::Database => selection.table.clone(),
        DataSourceKindDto::File => selection
            .file
            .as_ref()
            .map(|file| file.path.clone())
            .unwrap_or_else(|| "file".to_string()),
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
        if let Some(value) = value
            && !is_safe_ident(value)
        {
            return Err(AppError::Validation(format!(
                "{side} mapping field {field} contains unsafe characters: {value}"
            )));
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
