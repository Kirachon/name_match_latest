use crate::error::{AppError, AppResult};
use crate::state::AppState;
use name_matcher::run_service::dto::{
    ExportFormatDto, ExportRequestDto, ExportResultDto, ResultPageDto, ResultPageRequestDto,
};
use std::sync::Arc;
use tauri::State;

#[tauri::command]
pub fn get_results_page(
    request: ResultPageRequestDto,
    state: State<'_, Arc<AppState>>,
) -> AppResult<ResultPageDto> {
    let page = state
        .results
        .page(&request)
        .map_err(|e| AppError::Validation(e.to_string()))?;
    Ok(page)
}

#[tauri::command]
pub fn export_results(
    request: ExportRequestDto,
    state: State<'_, Arc<AppState>>,
) -> AppResult<ExportResultDto> {
    let snap = state
        .results
        .snapshot(&request.job_id)
        .ok_or_else(|| AppError::Validation(format!("unknown job_id: {}", request.job_id)))?;
    if snap.rows.is_empty() {
        return Err(AppError::Validation("No results to export".into()));
    }

    std::fs::create_dir_all(&request.output_directory)
        .map_err(|e| AppError::Io(format!("create export dir: {e}")))?;

    let stem = if request.file_stem.trim().is_empty() {
        "matches"
    } else {
        request.file_stem.trim()
    };
    validate_file_stem(stem)?;
    let dir = std::path::Path::new(&request.output_directory);
    let mut written = Vec::new();
    let min = request.min_confidence.unwrap_or(0.0);
    let rows: Vec<&name_matcher::run_service::dto::MatchPairDto> =
        snap.rows.iter().filter(|r| r.confidence >= min).collect();
    let row_count = rows.len() as u64;

    if matches!(request.format, ExportFormatDto::Csv | ExportFormatDto::Both) {
        let path = dir.join(format!("{stem}.csv"));
        write_csv(&path, &rows)?;
        written.push(path.to_string_lossy().into_owned());
    }
    if matches!(
        request.format,
        ExportFormatDto::Xlsx | ExportFormatDto::Both
    ) {
        let path = dir.join(format!("{stem}.xlsx"));
        write_xlsx(&path, &rows, &snap.summary)?;
        written.push(path.to_string_lossy().into_owned());
    }

    Ok(ExportResultDto {
        job_id: request.job_id,
        format: request.format,
        written_paths: written,
        rows_exported: row_count,
    })
}

fn validate_file_stem(stem: &str) -> AppResult<()> {
    let is_safe = !stem.is_empty()
        && !stem.contains("..")
        && !stem.contains(std::path::MAIN_SEPARATOR)
        && !stem.contains('/')
        && !stem.contains('\\')
        && !stem.contains(':')
        && stem
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'));
    if is_safe {
        Ok(())
    } else {
        Err(AppError::Validation(
            "file_stem may only contain letters, numbers, dot, dash, and underscore".into(),
        ))
    }
}

fn write_csv(
    path: &std::path::Path,
    rows: &[&name_matcher::run_service::dto::MatchPairDto],
) -> AppResult<()> {
    let mut wtr =
        csv::Writer::from_path(path).map_err(|e| AppError::Io(format!("csv writer: {e}")))?;
    wtr.write_record([
        "row_id",
        "source_id",
        "source_uuid",
        "source_full_name",
        "source_birthdate",
        "target_id",
        "target_uuid",
        "target_full_name",
        "target_birthdate",
        "confidence",
        "matched_fields",
    ])
    .map_err(|e| AppError::Io(format!("csv header: {e}")))?;
    for r in rows {
        wtr.write_record([
            r.row_id.to_string(),
            r.source_id.to_string(),
            r.source_uuid.clone().unwrap_or_default(),
            r.source_full_name.clone(),
            r.source_birthdate.clone().unwrap_or_default(),
            r.target_id.to_string(),
            r.target_uuid.clone().unwrap_or_default(),
            r.target_full_name.clone(),
            r.target_birthdate.clone().unwrap_or_default(),
            format!("{:.2}", r.confidence),
            r.matched_fields.join("|"),
        ])
        .map_err(|e| AppError::Io(format!("csv row: {e}")))?;
    }
    wtr.flush()
        .map_err(|e| AppError::Io(format!("csv flush: {e}")))?;
    Ok(())
}

fn write_xlsx(
    path: &std::path::Path,
    rows: &[&name_matcher::run_service::dto::MatchPairDto],
    summary: &name_matcher::run_service::dto::JobSummaryDto,
) -> AppResult<()> {
    use rust_xlsxwriter::{Format, Workbook};
    let mut wb = Workbook::new();
    let sheet = wb
        .add_worksheet()
        .set_name("Matches")
        .map_err(|e| AppError::Internal(format!("xlsx sheet: {e}")))?;
    let header = Format::new().set_bold().set_background_color(0xE6E6E6);
    let cols = [
        "row_id",
        "source_id",
        "source_uuid",
        "source_full_name",
        "source_birthdate",
        "target_id",
        "target_uuid",
        "target_full_name",
        "target_birthdate",
        "confidence",
        "matched_fields",
    ];
    for (i, c) in cols.iter().enumerate() {
        sheet
            .write_with_format(0, i as u16, *c, &header)
            .map_err(|e| AppError::Internal(format!("xlsx header: {e}")))?;
    }
    for (idx, r) in rows.iter().enumerate() {
        let row = (idx + 1) as u32;
        sheet.write(row, 0, r.row_id as f64).ok();
        sheet.write(row, 1, r.source_id as f64).ok();
        sheet
            .write(row, 2, r.source_uuid.clone().unwrap_or_default())
            .ok();
        sheet.write(row, 3, &r.source_full_name).ok();
        sheet
            .write(row, 4, r.source_birthdate.clone().unwrap_or_default())
            .ok();
        sheet.write(row, 5, r.target_id as f64).ok();
        sheet
            .write(row, 6, r.target_uuid.clone().unwrap_or_default())
            .ok();
        sheet.write(row, 7, &r.target_full_name).ok();
        sheet
            .write(row, 8, r.target_birthdate.clone().unwrap_or_default())
            .ok();
        sheet.write(row, 9, r.confidence as f64).ok();
        sheet.write(row, 10, r.matched_fields.join("|")).ok();
    }

    // Run summary sheet.
    let summary_sheet = wb
        .add_worksheet()
        .set_name("Summary")
        .map_err(|e| AppError::Internal(format!("xlsx summary sheet: {e}")))?;
    summary_sheet.write(0, 0, "Job ID").ok();
    summary_sheet.write(0, 1, &summary.job_id).ok();
    summary_sheet.write(1, 0, "Algorithm").ok();
    summary_sheet
        .write(1, 1, format!("{:?}", summary.algorithm))
        .ok();
    summary_sheet.write(2, 0, "Source").ok();
    summary_sheet.write(2, 1, &summary.source_table).ok();
    summary_sheet.write(3, 0, "Target").ok();
    summary_sheet.write(3, 1, &summary.target_table).ok();
    summary_sheet.write(4, 0, "Matches found").ok();
    summary_sheet.write(4, 1, summary.matches_found as f64).ok();
    summary_sheet.write(5, 0, "Elapsed (s)").ok();
    summary_sheet.write(5, 1, summary.elapsed_secs as f64).ok();

    wb.save(path)
        .map_err(|e| AppError::Io(format!("xlsx save: {e}")))?;
    Ok(())
}
