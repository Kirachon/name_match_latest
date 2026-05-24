use crate::error::{AppError, AppResult};
use crate::state::AppState;
use name_matcher::run_service::dto::{
    AlgorithmDto, ExplainPairRequestDto, ExportFormatDto, ExportRequestDto, ExportResultDto,
    ResultPageDto, ResultPageRequestDto, ReviewDecisionDto, SaveDecisionRequestDto,
    ScoreBreakdownDto,
};
use std::collections::{BTreeMap, BTreeSet, HashSet};
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
pub fn explain_pair(
    request: ExplainPairRequestDto,
    state: State<'_, Arc<AppState>>,
) -> AppResult<ScoreBreakdownDto> {
    let snap = state
        .results
        .snapshot(&request.job_id)
        .ok_or_else(|| AppError::Validation(format!("unknown job_id: {}", request.job_id)))?;
    let source = snap
        .source_people
        .iter()
        .find(|person| person.id == request.source_id)
        .ok_or_else(|| AppError::Validation("source person snapshot not found".into()))?;
    let target = snap
        .target_people
        .iter()
        .find(|person| person.id == request.target_id)
        .ok_or_else(|| AppError::Validation("target person snapshot not found".into()))?;
    match snap.summary.algorithm {
        AlgorithmDto::Fuzzy | AlgorithmDto::FuzzyNoMiddle => {
            let b = name_matcher::matching::explain_pair_fuzzy(
                source,
                target,
                matches!(snap.summary.algorithm, AlgorithmDto::FuzzyNoMiddle),
                true,
            );
            Ok(ScoreBreakdownDto {
                supported: b.supported,
                algorithm: b.algorithm,
                case_label: b.case_label,
                confidence: b.confidence,
                levenshtein_pct: b.levenshtein_pct,
                jaro_winkler_pct: b.jaro_winkler_pct,
                metaphone_pct: b.metaphone_pct,
                birthdate_match: b.birthdate_match,
                birthdate_swap_used: b.birthdate_swap_used,
                message: b.message,
            })
        }
        _ => Ok(ScoreBreakdownDto {
            supported: false,
            algorithm: format!("{:?}", snap.summary.algorithm),
            case_label: None,
            confidence: None,
            levenshtein_pct: None,
            jaro_winkler_pct: None,
            metaphone_pct: None,
            birthdate_match: None,
            birthdate_swap_used: false,
            message: Some("Explanation is currently available for Quick Match fuzzy modes only.".into()),
        }),
    }
}

#[tauri::command]
pub fn save_decision(
    request: SaveDecisionRequestDto,
    state: State<'_, Arc<AppState>>,
) -> AppResult<ReviewDecisionDto> {
    state
        .results
        .save_decision(request)
        .map_err(|e| AppError::Validation(e.to_string()))
}

#[tauri::command]
pub fn get_decisions(
    job_id: String,
    state: State<'_, Arc<AppState>>,
) -> AppResult<Vec<ReviewDecisionDto>> {
    state
        .results
        .get_decisions(&job_id)
        .map_err(|e| AppError::Validation(e.to_string()))
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
    validate_levels(&request.levels)?;
    let selected_levels: BTreeSet<u8> = request.levels.iter().copied().collect();
    let rejected_pairs = state
        .results
        .get_decisions(&request.job_id)
        .unwrap_or_default()
        .into_iter()
        .filter(|decision| decision.decision == "rejected")
        .map(|decision| (decision.source_id, decision.target_id))
        .collect::<HashSet<_>>();
    let rows = filter_export_rows(&snap.rows, min, &selected_levels, &rejected_pairs);
    let row_count = rows.len() as u64;
    let is_cascade = rows.iter().any(|r| r.matched_at_level.is_some());

    if matches!(request.format, ExportFormatDto::Csv | ExportFormatDto::Both) {
        let path = dir.join(format!("{stem}.csv"));
        write_csv(&path, &rows, request.include_extra_fields)?;
        written.push(path.to_string_lossy().into_owned());
        if is_cascade {
            for (level, level_rows) in group_rows_by_level(&rows) {
                let path = dir.join(format!("{stem}_L{level:02}.csv"));
                write_csv(&path, &level_rows, request.include_extra_fields)?;
                written.push(path.to_string_lossy().into_owned());
            }
        }
    }
    if matches!(
        request.format,
        ExportFormatDto::Xlsx | ExportFormatDto::Both
    ) {
        let path = dir.join(format!("{stem}.xlsx"));
        write_xlsx(
            &path,
            &rows,
            &snap.summary,
            is_cascade,
            request.include_extra_fields,
        )?;
        written.push(path.to_string_lossy().into_owned());
    }

    Ok(ExportResultDto {
        job_id: request.job_id,
        format: request.format,
        written_paths: written,
        rows_exported: row_count,
    })
}

fn group_rows_by_level<'a>(
    rows: &[&'a name_matcher::run_service::dto::MatchPairDto],
) -> BTreeMap<u8, Vec<&'a name_matcher::run_service::dto::MatchPairDto>> {
    let mut grouped: BTreeMap<u8, Vec<&name_matcher::run_service::dto::MatchPairDto>> =
        BTreeMap::new();
    for row in rows {
        if let Some(level) = row.matched_at_level {
            grouped.entry(level).or_default().push(*row);
        }
    }
    grouped
}

fn filter_export_rows<'a>(
    rows: &'a [name_matcher::run_service::dto::MatchPairDto],
    min_confidence: f32,
    selected_levels: &BTreeSet<u8>,
    rejected_pairs: &HashSet<(i64, i64)>,
) -> Vec<&'a name_matcher::run_service::dto::MatchPairDto> {
    rows.iter()
        .filter(|r| r.confidence >= min_confidence)
        .filter(|r| !rejected_pairs.contains(&(r.source_id, r.target_id)))
        .filter(|r| {
            selected_levels.is_empty()
                || r.matched_at_level
                    .map(|level| selected_levels.contains(&level))
                    .unwrap_or(false)
        })
        .collect()
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

fn validate_levels(levels: &[u8]) -> AppResult<()> {
    if let Some(level) = levels.iter().copied().find(|level| !(1..=11).contains(level)) {
        return Err(AppError::Validation(format!(
            "level must be between 1 and 11: {level}"
        )));
    }
    Ok(())
}

fn write_csv(
    path: &std::path::Path,
    rows: &[&name_matcher::run_service::dto::MatchPairDto],
    include_extra_fields: bool,
) -> AppResult<()> {
    let mut wtr =
        csv::Writer::from_path(path).map_err(|e| AppError::Io(format!("csv writer: {e}")))?;
    let is_cascade = rows.iter().any(|r| r.matched_at_level.is_some());
    let extra_headers = extra_field_headers(rows, include_extra_fields);
    let headers = export_headers(is_cascade, &extra_headers);
    wtr.write_record(headers)
        .map_err(|e| AppError::Io(format!("csv header: {e}")))?;
    for r in rows {
        let record = export_record(r, is_cascade, &extra_headers);
        wtr.write_record(record)
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
    is_cascade: bool,
    include_extra_fields: bool,
) -> AppResult<()> {
    use rust_xlsxwriter::{Format, Workbook};
    let mut wb = Workbook::new();
    let header = Format::new().set_bold().set_background_color(0xE6E6E6);
    let sheet = wb
        .add_worksheet()
        .set_name("Matches")
        .map_err(|e| AppError::Internal(format!("xlsx sheet: {e}")))?;
    let extra_headers = extra_field_headers(rows, include_extra_fields);
    write_xlsx_rows(sheet, rows, &header, is_cascade, &extra_headers)?;

    if is_cascade {
        for (level, level_rows) in group_rows_by_level(rows) {
            let sheet = wb
                .add_worksheet()
                .set_name(format!("L{level:02}"))
                .map_err(|e| AppError::Internal(format!("xlsx level sheet: {e}")))?;
            let level_extra_headers = extra_field_headers(&level_rows, include_extra_fields);
            write_xlsx_rows(sheet, &level_rows, &header, true, &level_extra_headers)?;
        }
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

fn write_xlsx_rows(
    sheet: &mut rust_xlsxwriter::Worksheet,
    rows: &[&name_matcher::run_service::dto::MatchPairDto],
    header: &rust_xlsxwriter::Format,
    is_cascade: bool,
    extra_headers: &[ExtraFieldHeader],
) -> AppResult<()> {
    let cols = export_headers(is_cascade, extra_headers);
    for (i, c) in cols.iter().enumerate() {
        sheet
            .write_with_format(0, i as u16, c.as_str(), header)
            .map_err(|e| AppError::Internal(format!("xlsx header: {e}")))?;
    }
    for (idx, r) in rows.iter().enumerate() {
        let row = (idx + 1) as u32;
        for (col, value) in export_record(r, is_cascade, extra_headers).iter().enumerate() {
            sheet.write(row, col as u16, value).ok();
        }
    }
    Ok(())
}

fn export_headers(is_cascade: bool, extra_headers: &[ExtraFieldHeader]) -> Vec<String> {
    let mut cols = vec![
        "row_id".to_string(),
        "source_id".to_string(),
        "source_uuid".to_string(),
        "source_full_name".to_string(),
        "source_birthdate".to_string(),
        "source_region_name".to_string(),
        "source_province_name".to_string(),
        "source_city_name".to_string(),
        "source_barangay_name".to_string(),
        "target_id".to_string(),
        "target_uuid".to_string(),
        "target_full_name".to_string(),
        "target_birthdate".to_string(),
        "target_region_name".to_string(),
        "target_province_name".to_string(),
        "target_city_name".to_string(),
        "target_barangay_name".to_string(),
        "confidence".to_string(),
        "matched_fields".to_string(),
        "remarks".to_string(),
    ];
    cols.extend(extra_headers.iter().map(|header| header.export_name.clone()));
    if is_cascade {
        cols.push("matched_at_level".to_string());
        cols.push("match_method".to_string());
    }
    cols
}

fn export_record(
    r: &name_matcher::run_service::dto::MatchPairDto,
    is_cascade: bool,
    extra_headers: &[ExtraFieldHeader],
) -> Vec<String> {
    let mut record = vec![
        r.row_id.to_string(),
        r.source_id.to_string(),
        r.source_uuid.clone().unwrap_or_default(),
        r.source_full_name.clone(),
        r.source_birthdate.clone().unwrap_or_default(),
        r.source_region_name.clone().unwrap_or_default(),
        r.source_province_name.clone().unwrap_or_default(),
        r.source_city_name.clone().unwrap_or_default(),
        r.source_barangay_name.clone().unwrap_or_default(),
        r.target_id.to_string(),
        r.target_uuid.clone().unwrap_or_default(),
        r.target_full_name.clone(),
        r.target_birthdate.clone().unwrap_or_default(),
        r.target_region_name.clone().unwrap_or_default(),
        r.target_province_name.clone().unwrap_or_default(),
        r.target_city_name.clone().unwrap_or_default(),
        r.target_barangay_name.clone().unwrap_or_default(),
        format!("{:.2}", r.confidence),
        r.matched_fields.join("|"),
        r.remarks.clone().unwrap_or_default(),
    ];
    for header in extra_headers {
        let value = match header.side {
            ExtraFieldSide::Source => r.source_extra_fields.get(&header.raw_name),
            ExtraFieldSide::Target => r.target_extra_fields.get(&header.raw_name),
        };
        record.push(value.cloned().unwrap_or_default());
    }
    if is_cascade {
        record.push(r.matched_at_level.map(|l| l.to_string()).unwrap_or_default());
        record.push(r.match_method.clone().unwrap_or_default());
    }
    record
}

#[derive(Clone, Copy)]
enum ExtraFieldSide {
    Source,
    Target,
}

struct ExtraFieldHeader {
    side: ExtraFieldSide,
    raw_name: String,
    export_name: String,
}

fn extra_field_headers(
    rows: &[&name_matcher::run_service::dto::MatchPairDto],
    include_extra_fields: bool,
) -> Vec<ExtraFieldHeader> {
    if !include_extra_fields {
        return Vec::new();
    }
    let mut source_keys = BTreeSet::new();
    let mut target_keys = BTreeSet::new();
    for row in rows {
        source_keys.extend(row.source_extra_fields.keys().cloned());
        target_keys.extend(row.target_extra_fields.keys().cloned());
    }
    source_keys
        .into_iter()
        .map(|raw_name| ExtraFieldHeader {
            export_name: format!("source_extra_{raw_name}"),
            raw_name,
            side: ExtraFieldSide::Source,
        })
        .chain(target_keys.into_iter().map(|raw_name| ExtraFieldHeader {
            export_name: format!("target_extra_{raw_name}"),
            raw_name,
            side: ExtraFieldSide::Target,
        }))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use name_matcher::run_service::dto::{AlgorithmDto, JobStateDto, JobSummaryDto, MatchPairDto};

    fn row(idx: u64, level: Option<u8>) -> MatchPairDto {
        MatchPairDto {
            row_id: idx,
            source_id: idx as i64,
            source_uuid: Some(format!("source-{idx}")),
            source_full_name: format!("Source {idx}"),
            source_birthdate: Some("1990-04-12".into()),
            source_region_name: Some("Region A".into()),
            source_province_name: Some("Province A".into()),
            source_city_name: Some("City A".into()),
            source_barangay_name: Some("Barangay A".into()),
            source_extra_fields: [
                ("region_name".to_string(), "Region A".to_string()),
                ("custom_source_note".to_string(), "Source note".to_string()),
            ]
            .into(),
            target_id: (idx + 100) as i64,
            target_uuid: Some(format!("target-{idx}")),
            target_full_name: format!("Target {idx}"),
            target_birthdate: Some("1990-12-04".into()),
            target_region_name: Some("Region B".into()),
            target_province_name: Some("Province B".into()),
            target_city_name: Some("City B".into()),
            target_barangay_name: Some("Barangay B".into()),
            target_extra_fields: [
                ("barangay_name".to_string(), "Barangay B".to_string()),
                ("custom_target_note".to_string(), "Target note".to_string()),
            ]
            .into(),
            confidence: 90.0 + idx as f32,
            matched_fields: vec!["birthdate".into()],
            remarks: Some("Birthday matched after month/day swap.".into()),
            matched_at_level: level,
            match_method: level.map(|l| format!("L{l} - test method")),
        }
    }

    fn summary() -> JobSummaryDto {
        JobSummaryDto {
            job_id: "job-test".into(),
            state: JobStateDto::Completed,
            algorithm: AlgorithmDto::Fuzzy,
            source_table: "source".into(),
            target_table: "target".into(),
            matches_found: 2,
            elapsed_secs: 1,
            started_at_unix_ms: 0,
            finished_at_unix_ms: Some(1000),
        }
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "name-matcher-tauri-export-test-{}-{name}",
            std::process::id()
        ))
    }

    #[test]
    fn quick_match_csv_headers_stay_unchanged() {
        let path = temp_path("quick.csv");
        let _ = std::fs::remove_file(&path);
        let rows = [row(1, None)];
        let refs: Vec<&MatchPairDto> = rows.iter().collect();

        write_csv(&path, &refs, false).expect("quick csv export");
        let text = std::fs::read_to_string(&path).expect("csv text");
        let header = text.lines().next().expect("header");
        assert_eq!(
            header,
            "row_id,source_id,source_uuid,source_full_name,source_birthdate,source_region_name,source_province_name,source_city_name,source_barangay_name,target_id,target_uuid,target_full_name,target_birthdate,target_region_name,target_province_name,target_city_name,target_barangay_name,confidence,matched_fields,remarks"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn cascade_csv_includes_level_metadata() {
        let path = temp_path("cascade.csv");
        let _ = std::fs::remove_file(&path);
        let rows = [row(1, Some(10))];
        let refs: Vec<&MatchPairDto> = rows.iter().collect();

        write_csv(&path, &refs, false).expect("cascade csv export");
        let text = std::fs::read_to_string(&path).expect("csv text");
        let mut lines = text.lines();
        let header = lines.next().expect("header");
        assert!(header.ends_with(",matched_at_level,match_method"));
        assert!(header.contains("source_region_name"));
        assert!(header.contains("target_barangay_name"));
        assert!(header.contains("remarks"));
        let data = lines.next().expect("data");
        assert!(data.contains(",10,L10 - test method"));
        assert!(data.contains("Birthday matched after month/day swap."));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn cascade_xlsx_writes_workbook_with_level_sheets() {
        let path = temp_path("cascade.xlsx");
        let _ = std::fs::remove_file(&path);
        let rows = [row(1, Some(1)), row(2, Some(10))];
        let refs: Vec<&MatchPairDto> = rows.iter().collect();

        write_xlsx(&path, &refs, &summary(), true, false).expect("cascade xlsx export");
        let meta = std::fs::metadata(&path).expect("xlsx metadata");
        assert!(meta.len() > 0);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn csv_includes_extra_fields_when_requested() {
        let path = temp_path("extra.csv");
        let _ = std::fs::remove_file(&path);
        let rows = [row(1, None)];
        let refs: Vec<&MatchPairDto> = rows.iter().collect();

        write_csv(&path, &refs, true).expect("csv export");
        let text = std::fs::read_to_string(&path).expect("csv text");
        let mut lines = text.lines();
        let header = lines.next().expect("header");
        assert!(header.contains("source_extra_custom_source_note"));
        assert!(header.contains("target_extra_custom_target_note"));
        let data = lines.next().expect("data");
        assert!(data.contains("Source note"));
        assert!(data.contains("Target note"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn export_filter_excludes_rejected_pairs() {
        let rows = [row(1, None), row(2, None)];
        let rejected = [(2_i64, 102_i64)].into_iter().collect();
        let selected_levels = BTreeSet::new();

        let filtered = filter_export_rows(&rows, 0.0, &selected_levels, &rejected);

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].row_id, 1);
    }
}
