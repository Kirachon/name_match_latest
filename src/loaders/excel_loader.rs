use crate::models::{ColumnMapping, Person};
use anyhow::{Context, Result, bail};
use calamine::{Data, Reader, open_workbook_auto};
use chrono::{Duration, NaiveDate};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;

const PREVIEW_LIMIT: usize = 5;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExcelPreviewRequestDto {
    pub path: String,
    #[serde(default)]
    pub sheet_name: Option<String>,
    #[serde(default)]
    pub date_format: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExcelSheetDto {
    pub name: String,
    pub rows: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExcelPreviewDto {
    pub path: String,
    pub sheets: Vec<ExcelSheetDto>,
    pub selected_sheet: String,
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
    pub warnings: Vec<String>,
    pub date_format: String,
    pub total_preview_rows: usize,
}

pub fn load_excel_preview(request: &ExcelPreviewRequestDto) -> Result<ExcelPreviewDto> {
    let path = Path::new(&request.path);
    if !path.is_file() {
        bail!("Excel file not found: {}", request.path);
    }
    let date_format = normalized_date_format(request);
    let mut workbook =
        open_workbook_auto(path).with_context(|| format!("Failed to open {}", request.path))?;
    let sheet_names = workbook.sheet_names().to_vec();
    if sheet_names.is_empty() {
        bail!("Excel workbook has no sheets");
    }

    let mut sheets = Vec::new();
    for name in &sheet_names {
        let rows = workbook
            .worksheet_range(name)
            .ok()
            .map(|range| range.height().saturating_sub(1) as u64)
            .unwrap_or(0);
        sheets.push(ExcelSheetDto {
            name: name.clone(),
            rows,
        });
    }

    let selected_sheet = select_sheet(&sheet_names, request.sheet_name.as_deref())?;
    let range = workbook
        .worksheet_range(&selected_sheet)
        .with_context(|| format!("Excel sheet not found: {selected_sheet}"))?;
    let mut row_iter = range.rows();
    let header_row = row_iter
        .next()
        .context("Excel sheet has no readable header row")?;
    let headers = header_row
        .iter()
        .map(cell_to_header)
        .collect::<Vec<String>>();
    let mut warnings = Vec::new();
    validate_headers(&headers, &mut warnings)?;
    let date_columns = date_column_indexes(&headers);

    let mut rows = Vec::new();
    for row in row_iter.take(PREVIEW_LIMIT) {
        let values = row_to_values(row, headers.len(), &date_columns, &date_format);
        validate_date_values(&values, &date_columns, &date_format, &mut warnings);
        rows.push(values);
    }
    if rows_have_formula_like_cells(&rows) {
        warnings.push("Potential spreadsheet formula values detected; export will sanitize cells that start with =, +, -, or @.".to_string());
    }
    warnings.sort();
    warnings.dedup();
    let total_preview_rows = rows.len();

    Ok(ExcelPreviewDto {
        path: request.path.clone(),
        sheets,
        selected_sheet,
        headers,
        rows,
        warnings,
        date_format,
        total_preview_rows,
    })
}

pub fn load_excel_people(
    request: &ExcelPreviewRequestDto,
    mapping: Option<&ColumnMapping>,
) -> Result<Vec<Person>> {
    let path = Path::new(&request.path);
    if !path.is_file() {
        bail!("Excel file not found: {}", request.path);
    }
    let date_format = normalized_date_format(request);
    let mut workbook =
        open_workbook_auto(path).with_context(|| format!("Failed to open {}", request.path))?;
    let sheet_names = workbook.sheet_names().to_vec();
    if sheet_names.is_empty() {
        bail!("Excel workbook has no sheets");
    }
    let selected_sheet = select_sheet(&sheet_names, request.sheet_name.as_deref())?;
    let range = workbook
        .worksheet_range(&selected_sheet)
        .with_context(|| format!("Excel sheet not found: {selected_sheet}"))?;
    let mut row_iter = range.rows();
    let header_row = row_iter
        .next()
        .context("Excel sheet has no readable header row")?;
    let headers = header_row
        .iter()
        .map(cell_to_header)
        .collect::<Vec<String>>();
    validate_headers(&headers, &mut Vec::new())?;
    let date_columns = date_column_indexes(&headers);
    let mapping = mapping.cloned().unwrap_or_else(|| infer_mapping(&headers));
    if mapping.first_name.is_empty() || mapping.last_name.is_empty() || mapping.birthdate.is_empty()
    {
        bail!("Excel column mapping is missing first_name, last_name, or birthdate");
    }

    let mut people = Vec::new();
    let mut seen_ids = HashSet::new();
    let mapped_columns = mapped_column_names(&mapping);
    for (row_index, row) in row_iter.enumerate() {
        let values = row_to_values(row, headers.len(), &date_columns, &date_format);
        if values.iter().all(|value| value.trim().is_empty()) {
            continue;
        }
        let row_map = headers
            .iter()
            .cloned()
            .zip(values.into_iter().map(|value| value.trim().to_string()))
            .collect::<HashMap<_, _>>();
        let id = stable_row_id(&headers, &row_map, &mapping)
            .with_context(|| format!("Invalid Excel id on row {}", row_index + 2))?;
        if !seen_ids.insert(id) {
            bail!("Duplicate Excel stable id {} on row {}", id, row_index + 2);
        }
        let birthdate = value_for(&row_map, &mapping.birthdate)
            .filter(|value| !value.is_empty())
            .map(|value| NaiveDate::parse_from_str(value, &date_format))
            .transpose()
            .with_context(|| format!("Invalid Excel birthdate for id {}", id))?;

        let mut extra_fields = HashMap::new();
        for (header, value) in &row_map {
            if !mapped_columns.contains(header.as_str()) {
                extra_fields.insert(header.clone(), value.clone());
            }
        }

        people.push(Person {
            id,
            uuid: mapping
                .uuid
                .as_ref()
                .and_then(|name| value_for(&row_map, name))
                .map(str::to_string),
            first_name: value_for(&row_map, &mapping.first_name).map(str::to_string),
            middle_name: mapping
                .middle_name
                .as_ref()
                .and_then(|name| value_for(&row_map, name))
                .map(str::to_string),
            last_name: value_for(&row_map, &mapping.last_name).map(str::to_string),
            birthdate,
            hh_id: mapping
                .hh_id
                .as_ref()
                .and_then(|name| value_for(&row_map, name))
                .map(str::to_string),
            extra_fields,
        });
    }

    Ok(people)
}

fn normalized_date_format(request: &ExcelPreviewRequestDto) -> String {
    request
        .date_format
        .clone()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "%Y-%m-%d".to_string())
}

fn select_sheet(sheet_names: &[String], requested: Option<&str>) -> Result<String> {
    if let Some(requested) = requested.filter(|value| !value.trim().is_empty()) {
        if sheet_names.iter().any(|name| name == requested) {
            return Ok(requested.to_string());
        }
        bail!("Excel sheet not found: {requested}");
    }
    sheet_names
        .first()
        .cloned()
        .context("Excel workbook has no sheets")
}

fn cell_to_header(cell: &Data) -> String {
    cell_to_string(cell, false, "%Y-%m-%d").trim().to_string()
}

fn row_to_values(
    row: &[Data],
    width: usize,
    date_columns: &HashSet<usize>,
    date_format: &str,
) -> Vec<String> {
    (0..width)
        .map(|idx| {
            row.get(idx)
                .map(|cell| cell_to_string(cell, date_columns.contains(&idx), date_format))
                .unwrap_or_default()
        })
        .collect()
}

fn cell_to_string(cell: &Data, date_cell: bool, date_format: &str) -> String {
    match cell {
        Data::Empty => String::new(),
        Data::String(value) => value.trim().to_string(),
        Data::Float(value) if date_cell => {
            excel_serial_to_date(*value, date_format).unwrap_or_else(|| trim_float(*value))
        }
        Data::Float(value) => trim_float(*value),
        Data::Int(value) if date_cell => {
            excel_serial_to_date(*value as f64, date_format).unwrap_or_else(|| value.to_string())
        }
        Data::Int(value) => value.to_string(),
        Data::Bool(value) => value.to_string(),
        Data::DateTime(value) => {
            let serial = value.as_f64();
            excel_serial_to_date(serial, date_format).unwrap_or_else(|| trim_float(serial))
        }
        Data::DateTimeIso(value) | Data::DurationIso(value) => value.trim().to_string(),
        Data::Error(value) => value.to_string(),
    }
}

fn trim_float(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.0}")
    } else {
        value.to_string()
    }
}

fn excel_serial_to_date(value: f64, date_format: &str) -> Option<String> {
    if !value.is_finite() || value <= 0.0 {
        return None;
    }
    let base = NaiveDate::from_ymd_opt(1899, 12, 30)?;
    let date = base.checked_add_signed(Duration::days(value.trunc() as i64))?;
    Some(date.format(date_format).to_string())
}

fn validate_headers(headers: &[String], warnings: &mut Vec<String>) -> Result<()> {
    if headers.is_empty() {
        bail!("Excel header row is empty");
    }
    let mut seen = HashSet::new();
    for header in headers {
        if header.trim().is_empty() {
            bail!("Excel contains an empty header");
        }
        if !seen.insert(header.to_ascii_lowercase()) {
            warnings.push(format!("Duplicate header detected: {}", header));
        }
    }
    let inferred = infer_mapping(headers);
    if inferred.id.is_empty() {
        warnings.push(
            "No ID column detected; file runs will use deterministic content-based row IDs. Re-imported unchanged rows keep the same IDs, but edited row content changes IDs."
                .to_string(),
        );
    }
    Ok(())
}

fn date_column_indexes(headers: &[String]) -> HashSet<usize> {
    headers
        .iter()
        .enumerate()
        .filter_map(|(idx, header)| {
            let normalized = header.to_ascii_lowercase();
            (normalized.contains("date") || normalized == "dob" || normalized.contains("birthday"))
                .then_some(idx)
        })
        .collect()
}

fn validate_date_values(
    row: &[String],
    date_columns: &HashSet<usize>,
    date_format: &str,
    warnings: &mut Vec<String>,
) {
    for idx in date_columns {
        if let Some(value) = row.get(*idx) {
            if !value.trim().is_empty()
                && NaiveDate::parse_from_str(value.trim(), date_format).is_err()
            {
                warnings.push(format!(
                    "Date value '{}' does not match format {}",
                    value, date_format
                ));
            }
        }
    }
}

fn rows_have_formula_like_cells(rows: &[Vec<String>]) -> bool {
    rows.iter().flatten().any(|cell| {
        matches!(
            cell.trim_start().chars().next(),
            Some('=' | '+' | '-' | '@')
        )
    })
}

fn infer_mapping(headers: &[String]) -> ColumnMapping {
    ColumnMapping {
        id: pick(headers, &["id", "person_id", "beneficiary_id"]),
        uuid: optional_pick(headers, &["uuid"]),
        first_name: pick(headers, &["first_name", "firstname", "fname", "given_name"]),
        middle_name: optional_pick(headers, &["middle_name", "middlename", "mname"]),
        last_name: pick(headers, &["last_name", "lastname", "lname", "surname"]),
        birthdate: pick(headers, &["birthdate", "birth_date", "birthday", "dob"]),
        hh_id: optional_pick(headers, &["hh_id", "household_id"]),
    }
}

fn pick(headers: &[String], hints: &[&str]) -> String {
    optional_pick(headers, hints).unwrap_or_default()
}

fn optional_pick(headers: &[String], hints: &[&str]) -> Option<String> {
    let normalized = headers
        .iter()
        .map(|header| (normalize(header), header.clone()))
        .collect::<HashMap<_, _>>();
    for hint in hints {
        if let Some(header) = normalized.get(&normalize(hint)) {
            return Some(header.clone());
        }
    }
    for hint in hints {
        if let Some(header) = headers
            .iter()
            .find(|header| normalize(header).contains(&normalize(hint)))
        {
            return Some(header.clone());
        }
    }
    None
}

fn normalize(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn value_for<'a>(row: &'a HashMap<String, String>, column: &str) -> Option<&'a str> {
    row.get(column)
        .map(String::as_str)
        .filter(|value| !value.is_empty())
}

fn stable_row_id(
    headers: &[String],
    row: &HashMap<String, String>,
    mapping: &ColumnMapping,
) -> Result<i64> {
    if let Some(id_text) = value_for(row, &mapping.id) {
        return id_text
            .parse::<i64>()
            .with_context(|| format!("Invalid Excel id value '{}'", id_text));
    }

    let mut hash = 0xcbf29ce484222325u64;
    for header in headers {
        fnv1a_update(&mut hash, header.as_bytes());
        fnv1a_update(&mut hash, b"=");
        if let Some(value) = row.get(header) {
            fnv1a_update(&mut hash, value.as_bytes());
        }
        fnv1a_update(&mut hash, b"\x1f");
    }

    Ok((hash & 0x7fff_ffff_ffff_ffff) as i64)
}

fn fnv1a_update(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(0x100000001b3);
    }
}

fn mapped_column_names(mapping: &ColumnMapping) -> HashSet<&str> {
    [
        Some(mapping.id.as_str()),
        mapping.uuid.as_deref(),
        Some(mapping.first_name.as_str()),
        mapping.middle_name.as_deref(),
        Some(mapping.last_name.as_str()),
        Some(mapping.birthdate.as_str()),
        mapping.hh_id.as_deref(),
    ]
    .into_iter()
    .flatten()
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_xlsxwriter::Workbook;

    fn write_fixture(name: &str) -> String {
        let path = std::env::temp_dir().join(format!("nm_excel_loader_{name}.xlsx"));
        let mut workbook = Workbook::new();
        let sheet = workbook.add_worksheet();
        sheet.set_name("People").unwrap();
        sheet.write_string(0, 0, "person_id").unwrap();
        sheet.write_string(0, 1, "given_name").unwrap();
        sheet.write_string(0, 2, "surname").unwrap();
        sheet.write_string(0, 3, "dob").unwrap();
        sheet.write_string(0, 4, "note").unwrap();
        sheet.write_number(1, 0, 1).unwrap();
        sheet.write_string(1, 1, "Ana").unwrap();
        sheet.write_string(1, 2, "Santos").unwrap();
        sheet.write_number(1, 3, 32875).unwrap();
        sheet.write_string(1, 4, "source-extra").unwrap();
        workbook.save(&path).unwrap();
        path.to_string_lossy().to_string()
    }

    #[test]
    fn previews_excel_sheet_and_serial_dates() {
        let path = write_fixture("preview");
        let preview = load_excel_preview(&ExcelPreviewRequestDto {
            path,
            sheet_name: None,
            date_format: None,
        })
        .unwrap();

        assert_eq!(preview.selected_sheet, "People");
        assert_eq!(
            preview.headers,
            vec!["person_id", "given_name", "surname", "dob", "note"]
        );
        assert_eq!(preview.rows[0][3], "1990-01-02");
        assert_eq!(preview.sheets[0].rows, 1);
    }

    #[test]
    fn loads_people_from_excel_with_mapping() {
        let path = write_fixture("people");
        let people = load_excel_people(
            &ExcelPreviewRequestDto {
                path,
                sheet_name: Some("People".to_string()),
                date_format: None,
            },
            None,
        )
        .unwrap();

        assert_eq!(people.len(), 1);
        assert_eq!(people[0].id, 1);
        assert_eq!(people[0].first_name.as_deref(), Some("Ana"));
        assert_eq!(people[0].birthdate, NaiveDate::from_ymd_opt(1990, 1, 2));
        assert_eq!(
            people[0].extra_fields.get("note").map(String::as_str),
            Some("source-extra")
        );
    }
}
