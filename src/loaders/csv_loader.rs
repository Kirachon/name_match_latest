use crate::models::{ColumnMapping, Person};
use crate::perf::StageTimer;
use anyhow::{Context, Result, bail};
use chrono::NaiveDate;
use encoding_rs::{Encoding, UTF_8, WINDOWS_1252};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Cursor, Read};
use std::path::Path;
use std::sync::Arc;

const PREVIEW_LIMIT: usize = 5;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CsvEncodingDto {
    Utf8,
    Utf8Bom,
    Windows1252,
    Latin1,
}

impl CsvEncodingDto {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Utf8 => "UTF-8",
            Self::Utf8Bom => "UTF-8 BOM",
            Self::Windows1252 => "Windows-1252",
            Self::Latin1 => "Latin-1",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CsvDelimiterDto {
    Comma,
    Semicolon,
    Tab,
}

impl CsvDelimiterDto {
    fn byte(&self) -> u8 {
        match self {
            Self::Comma => b',',
            Self::Semicolon => b';',
            Self::Tab => b'\t',
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsvPreviewRequestDto {
    pub path: String,
    #[serde(default)]
    pub encoding: Option<CsvEncodingDto>,
    #[serde(default)]
    pub delimiter: Option<CsvDelimiterDto>,
    #[serde(default)]
    pub date_format: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsvPreviewDto {
    pub path: String,
    pub encoding: CsvEncodingDto,
    pub delimiter: CsvDelimiterDto,
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
    pub warnings: Vec<String>,
    pub date_format: String,
    pub total_preview_rows: usize,
}

#[derive(Clone)]
pub struct CsvLoadOptions {
    pub include_extra_fields: bool,
    pub generate_stable_ids: bool,
    pub progress_interval_rows: usize,
    pub should_cancel: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
    pub on_progress_rows: Option<Arc<dyn Fn(usize) + Send + Sync>>,
}

impl Default for CsvLoadOptions {
    fn default() -> Self {
        Self {
            include_extra_fields: true,
            generate_stable_ids: true,
            progress_interval_rows: 10_000,
            should_cancel: None,
            on_progress_rows: None,
        }
    }
}

#[derive(Debug, Clone)]
struct CsvHeaderIndex {
    id: Option<usize>,
    uuid: Option<usize>,
    first_name: Option<usize>,
    middle_name: Option<usize>,
    last_name: Option<usize>,
    birthdate: Option<usize>,
    hh_id: Option<usize>,
}

impl CsvHeaderIndex {
    fn new(headers: &[String], mapping: &ColumnMapping) -> Self {
        Self {
            id: header_pos(headers, &mapping.id),
            uuid: mapping.uuid.as_ref().and_then(|name| header_pos(headers, name)),
            first_name: header_pos(headers, &mapping.first_name),
            middle_name: mapping
                .middle_name
                .as_ref()
                .and_then(|name| header_pos(headers, name)),
            last_name: header_pos(headers, &mapping.last_name),
            birthdate: header_pos(headers, &mapping.birthdate),
            hh_id: mapping.hh_id.as_ref().and_then(|name| header_pos(headers, name)),
        }
    }
}

pub fn load_csv_preview(request: &CsvPreviewRequestDto) -> Result<CsvPreviewDto> {
    let timer = StageTimer::start("csv_preview_load");
    let path = Path::new(&request.path);
    if !path.is_file() {
        bail!("CSV file not found: {}", request.path);
    }
    let bytes = std::fs::read(path).with_context(|| format!("Failed to read {}", request.path))?;
    let detected_encoding = request
        .encoding
        .clone()
        .unwrap_or_else(|| detect_encoding(&bytes));
    let text = decode_bytes(&bytes, &detected_encoding)?;
    let delimiter = request
        .delimiter
        .clone()
        .unwrap_or_else(|| detect_delimiter(&text));
    let date_format = request
        .date_format
        .clone()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "%Y-%m-%d".to_string());

    let mut warnings = Vec::new();
    if has_formula_injection_risk(&text) {
        warnings.push("Potential spreadsheet formula values detected; export will sanitize cells that start with =, +, -, or @.".to_string());
    }

    let mut reader = csv::ReaderBuilder::new()
        .delimiter(delimiter.byte())
        .flexible(true)
        .from_reader(text.as_bytes());

    let headers = reader
        .headers()
        .context("CSV file has no readable header row")?
        .iter()
        .map(|v| v.trim().to_string())
        .collect::<Vec<_>>();
    validate_headers(&headers, &mut warnings)?;

    let date_columns = headers
        .iter()
        .enumerate()
        .filter_map(|(idx, header)| {
            let h = header.to_ascii_lowercase();
            (h.contains("date") || h == "dob" || h.contains("birthday")).then_some(idx)
        })
        .collect::<Vec<_>>();

    let mut rows = Vec::new();
    for record in reader.records().take(PREVIEW_LIMIT) {
        let record = record.context("Failed to parse CSV preview row")?;
        let row = record.iter().map(|v| v.to_string()).collect::<Vec<_>>();
        for idx in &date_columns {
            if let Some(value) = row.get(*idx) {
                if !value.trim().is_empty()
                    && NaiveDate::parse_from_str(value.trim(), &date_format).is_err()
                {
                    warnings.push(format!(
                        "Date value '{}' does not match format {}",
                        value, date_format
                    ));
                }
            }
        }
        rows.push(row);
    }

    warnings.sort();
    warnings.dedup();

    let preview = CsvPreviewDto {
        path: request.path.clone(),
        encoding: detected_encoding,
        delimiter,
        headers,
        total_preview_rows: rows.len(),
        rows,
        warnings,
        date_format,
    };
    timer.finish();
    Ok(preview)
}

pub fn load_csv_people(
    request: &CsvPreviewRequestDto,
    mapping: Option<&ColumnMapping>,
) -> Result<Vec<Person>> {
    load_csv_people_with_options(request, mapping, &CsvLoadOptions::default())
}

pub fn load_csv_people_with_options(
    request: &CsvPreviewRequestDto,
    mapping: Option<&ColumnMapping>,
    options: &CsvLoadOptions,
) -> Result<Vec<Person>> {
    let timer = StageTimer::start("csv_people_load");
    let path = Path::new(&request.path);
    if !path.is_file() {
        bail!("CSV file not found: {}", request.path);
    }
    let (delimiter, input): (CsvDelimiterDto, Box<dyn Read>) =
        if let (Some(encoding), Some(delimiter)) = (&request.encoding, &request.delimiter) {
            if matches!(encoding, CsvEncodingDto::Utf8 | CsvEncodingDto::Utf8Bom) {
                tracing::info!(
                    path = %request.path,
                    encoding = encoding.label(),
                    delimiter = ?delimiter,
                    "csv_people_streaming_reader"
                );
                let file =
                    File::open(path).with_context(|| format!("Failed to open {}", request.path))?;
                (delimiter.clone(), Box::new(BufReader::new(file)))
            } else {
                let bytes = std::fs::read(path)
                    .with_context(|| format!("Failed to read {}", request.path))?;
                let text = decode_bytes(&bytes, encoding)?;
                (delimiter.clone(), Box::new(Cursor::new(text.into_bytes())))
            }
        } else {
            let bytes =
                std::fs::read(path).with_context(|| format!("Failed to read {}", request.path))?;
            let encoding = request
                .encoding
                .clone()
                .unwrap_or_else(|| detect_encoding(&bytes));
            let text = decode_bytes(&bytes, &encoding)?;
            let delimiter = request
                .delimiter
                .clone()
                .unwrap_or_else(|| detect_delimiter(&text));
            (delimiter, Box::new(Cursor::new(text.into_bytes())))
        };
    let date_format = request
        .date_format
        .clone()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "%Y-%m-%d".to_string());
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(delimiter.byte())
        .flexible(true)
        .from_reader(input);
    let headers = reader
        .headers()
        .context("CSV file has no readable header row")?
        .iter()
        .map(|v| v.trim().to_string())
        .collect::<Vec<_>>();
    validate_headers(&headers, &mut Vec::new())?;
    let mapping = mapping.cloned().unwrap_or_else(|| infer_mapping(&headers));
    if mapping.first_name.is_empty() || mapping.last_name.is_empty() || mapping.birthdate.is_empty()
    {
        bail!("CSV column mapping is missing first_name, last_name, or birthdate");
    }
    let header_index = CsvHeaderIndex::new(&headers, &mapping);
    let mapped_columns = mapped_column_names(&mapping);
    let mut people = Vec::new();
    let mut seen_ids = HashSet::new();
    for (row_index, record) in reader.records().enumerate() {
        if let Some(should_cancel) = &options.should_cancel {
            if should_cancel() {
                bail!("CSV load cancelled");
            }
        }
        let record = record.context("Failed to parse CSV row")?;
        let id = if options.generate_stable_ids {
            stable_row_id_indexed(&headers, &record, &mapping, &header_index)
        } else {
            record
                .get(header_index.id.context("CSV column mapping is missing id")?)
                .map(|value| value.trim().parse::<i64>())
                .transpose()
                .context("Invalid CSV id")?
                .context("CSV id value is empty")
        }
            .with_context(|| format!("Invalid CSV id on row {}", row_index + 2))?;
        if !seen_ids.insert(id) {
            bail!("Duplicate CSV stable id {} on row {}", id, row_index + 2);
        }
        let birthdate = value_at(&record, header_index.birthdate)
            .filter(|v| !v.is_empty())
            .map(|value| NaiveDate::parse_from_str(value, &date_format))
            .transpose()
            .with_context(|| format!("Invalid CSV birthdate for id {}", id))?;

        let mut extra_fields = HashMap::new();
        if options.include_extra_fields {
            for (idx, header) in headers.iter().enumerate() {
                if !mapped_columns.contains(header.as_str()) {
                    if let Some(value) = value_at(&record, Some(idx)) {
                        extra_fields.insert(header.clone(), value.to_string());
                    }
                }
            }
        }
        people.push(Person {
            id,
            uuid: mapping
                .uuid
                .as_ref()
                .and_then(|_| value_at(&record, header_index.uuid))
                .map(str::to_string),
            first_name: value_at(&record, header_index.first_name).map(str::to_string),
            middle_name: mapping
                .middle_name
                .as_ref()
                .and_then(|_| value_at(&record, header_index.middle_name))
                .map(str::to_string),
            last_name: value_at(&record, header_index.last_name).map(str::to_string),
            birthdate,
            hh_id: mapping
                .hh_id
                .as_ref()
                .and_then(|_| value_at(&record, header_index.hh_id))
                .map(str::to_string),
            extra_fields,
        });
        if options.progress_interval_rows > 0 && (row_index + 1) % options.progress_interval_rows == 0 {
            if let Some(on_progress_rows) = &options.on_progress_rows {
                on_progress_rows(row_index + 1);
            }
            tracing::info!(rows = (row_index + 1) as u64, "csv_people_load_progress");
        }
    }
    tracing::info!(rows = people.len() as u64, "csv_people_load_summary");
    timer.finish();
    Ok(people)
}

fn header_pos(headers: &[String], name: &str) -> Option<usize> {
    headers.iter().position(|header| header == name)
}

fn value_at(record: &csv::StringRecord, idx: Option<usize>) -> Option<&str> {
    record.get(idx?).map(str::trim)
}

fn detect_encoding(bytes: &[u8]) -> CsvEncodingDto {
    if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        return CsvEncodingDto::Utf8Bom;
    }
    let mut detector = chardetng::EncodingDetector::new();
    detector.feed(bytes, true);
    let encoding = detector.guess(None, true);
    if encoding == UTF_8 {
        CsvEncodingDto::Utf8
    } else {
        CsvEncodingDto::Windows1252
    }
}

fn decode_bytes(bytes: &[u8], encoding: &CsvEncodingDto) -> Result<String> {
    let enc: &'static Encoding = match encoding {
        CsvEncodingDto::Utf8 | CsvEncodingDto::Utf8Bom => UTF_8,
        CsvEncodingDto::Windows1252 | CsvEncodingDto::Latin1 => WINDOWS_1252,
    };
    let (text, _, had_errors) = enc.decode(bytes);
    if had_errors && matches!(encoding, CsvEncodingDto::Utf8 | CsvEncodingDto::Utf8Bom) {
        bail!("CSV is not valid {}", encoding.label());
    }
    Ok(text.trim_start_matches('\u{feff}').to_string())
}

fn detect_delimiter(text: &str) -> CsvDelimiterDto {
    let sample = text.lines().take(10).collect::<Vec<_>>();
    let score = |needle: char| {
        sample
            .iter()
            .map(|line| line.matches(needle).count())
            .sum::<usize>()
    };
    let comma = score(',');
    let semicolon = score(';');
    let tab = score('\t');
    if tab > comma && tab > semicolon {
        CsvDelimiterDto::Tab
    } else if semicolon > comma {
        CsvDelimiterDto::Semicolon
    } else {
        CsvDelimiterDto::Comma
    }
}

fn validate_headers(headers: &[String], warnings: &mut Vec<String>) -> Result<()> {
    if headers.is_empty() {
        bail!("CSV header row is empty");
    }
    let mut seen = std::collections::HashSet::new();
    for header in headers {
        if header.trim().is_empty() {
            bail!("CSV contains an empty header");
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

fn value_for<'a>(row: &'a HashMap<&str, &str>, column: &str) -> Option<&'a str> {
    row.get(column).copied().filter(|value| !value.is_empty())
}

fn stable_row_id(
    headers: &[String],
    row: &HashMap<&str, &str>,
    mapping: &ColumnMapping,
) -> Result<i64> {
    if let Some(id_text) = value_for(row, &mapping.id) {
        return id_text
            .parse::<i64>()
            .with_context(|| format!("Invalid CSV id value '{}'", id_text));
    }

    let mut hash = 0xcbf29ce484222325u64;
    for header in headers {
        fnv1a_update(&mut hash, header.as_bytes());
        fnv1a_update(&mut hash, b"=");
        if let Some(value) = row.get(header.as_str()) {
            fnv1a_update(&mut hash, value.as_bytes());
        }
        fnv1a_update(&mut hash, b"\x1f");
    }

    Ok((hash & 0x7fff_ffff_ffff_ffff) as i64)
}

fn stable_row_id_indexed(
    headers: &[String],
    record: &csv::StringRecord,
    mapping: &ColumnMapping,
    header_index: &CsvHeaderIndex,
) -> Result<i64> {
    if let Some(id_text) = value_at(record, header_index.id) {
        if !id_text.is_empty() {
            return id_text
                .parse::<i64>()
                .with_context(|| format!("Invalid CSV id value '{}'", id_text));
        }
    }

    let mut hash = 0xcbf29ce484222325u64;
    for (idx, header) in headers.iter().enumerate() {
        fnv1a_update(&mut hash, header.as_bytes());
        fnv1a_update(&mut hash, b"=");
        if let Some(value) = value_at(record, Some(idx)) {
            fnv1a_update(&mut hash, value.as_bytes());
        }
        fnv1a_update(&mut hash, b"\x1f");
    }

    tracing::debug!(
        id_column = mapping.id.as_str(),
        "csv_stable_id_generated_from_row_hash"
    );
    Ok((hash & 0x7fff_ffff_ffff_ffff) as i64)
}

fn fnv1a_update(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(0x100000001b3);
    }
}

fn mapped_column_names(mapping: &ColumnMapping) -> std::collections::HashSet<&str> {
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

fn has_formula_injection_risk(text: &str) -> bool {
    text.lines()
        .skip(1)
        .flat_map(|line| line.split([',', ';', '\t']))
        .any(|cell| {
            matches!(
                cell.trim_start().chars().next(),
                Some('=' | '+' | '-' | '@')
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_fixture(name: &str, bytes: &[u8]) -> String {
        let path = std::env::temp_dir().join(format!("nm_csv_loader_{name}.csv"));
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(bytes).unwrap();
        path.to_string_lossy().to_string()
    }

    #[test]
    fn previews_utf8_comma_csv() {
        let path = write_fixture(
            "utf8",
            b"id,first_name,last_name,birthdate\n1,Ana,Santos,1990-01-02\n",
        );
        let preview = load_csv_preview(&CsvPreviewRequestDto {
            path,
            encoding: None,
            delimiter: None,
            date_format: None,
        })
        .unwrap();
        assert_eq!(
            preview.headers,
            vec!["id", "first_name", "last_name", "birthdate"]
        );
        assert_eq!(preview.delimiter.byte(), b',');
        assert_eq!(preview.rows.len(), 1);
    }

    #[test]
    fn previews_windows_1252_semicolon_csv() {
        let path = write_fixture(
            "windows1252",
            b"id;first_name;last_name;birthdate\n1;Jose\xe9;Reyes;01/02/1990\n",
        );
        let preview = load_csv_preview(&CsvPreviewRequestDto {
            path,
            encoding: Some(CsvEncodingDto::Windows1252),
            delimiter: None,
            date_format: Some("%m/%d/%Y".to_string()),
        })
        .unwrap();
        assert!(preview.rows[0][1].contains("Jose"));
        assert_eq!(preview.delimiter.byte(), b';');
        assert!(preview.warnings.is_empty());
    }

    #[test]
    fn warns_on_formula_like_cells() {
        let path = write_fixture("formula", b"id,first_name,last_name\n1,=cmd,Reyes\n");
        let preview = load_csv_preview(&CsvPreviewRequestDto {
            path,
            encoding: None,
            delimiter: None,
            date_format: None,
        })
        .unwrap();
        assert!(preview.warnings.iter().any(|w| w.contains("formula")));
    }

    #[test]
    fn loads_people_from_csv_with_mapping() {
        let path = write_fixture(
            "people",
            b"person_id,given_name,surname,dob,note\n1,Ana,Santos,1990-01-02,ok\n",
        );
        let people = load_csv_people(
            &CsvPreviewRequestDto {
                path,
                encoding: None,
                delimiter: None,
                date_format: None,
            },
            None,
        )
        .unwrap();
        assert_eq!(people.len(), 1);
        assert_eq!(people[0].id, 1);
        assert_eq!(people[0].first_name.as_deref(), Some("Ana"));
        assert_eq!(
            people[0].extra_fields.get("note").map(String::as_str),
            Some("ok")
        );
    }

    #[test]
    fn generates_stable_content_ids_when_no_id_column_exists() {
        let path_a = write_fixture(
            "no_id_a",
            b"given_name,surname,dob,note\nAna,Santos,1990-01-02,ok\nBen,Cruz,1991-02-03,ok\n",
        );
        let path_b = write_fixture(
            "no_id_b",
            b"given_name,surname,dob,note\nAna,Santos,1990-01-02,ok\nBen,Cruz,1991-02-03,ok\n",
        );

        let first = load_csv_people(
            &CsvPreviewRequestDto {
                path: path_a,
                encoding: None,
                delimiter: None,
                date_format: None,
            },
            None,
        )
        .unwrap();
        let second = load_csv_people(
            &CsvPreviewRequestDto {
                path: path_b,
                encoding: None,
                delimiter: None,
                date_format: None,
            },
            None,
        )
        .unwrap();

        assert_eq!(first.len(), 2);
        assert_ne!(first[0].id, first[1].id);
        assert_eq!(first[0].id, second[0].id);
        assert_eq!(first[1].id, second[1].id);
    }

    #[test]
    fn rejects_duplicate_explicit_csv_ids() {
        let path = write_fixture(
            "duplicate_ids",
            b"id,first_name,last_name,birthdate\n1,Ana,Santos,1990-01-02\n1,Ben,Cruz,1991-02-03\n",
        );

        let err = load_csv_people(
            &CsvPreviewRequestDto {
                path,
                encoding: None,
                delimiter: None,
                date_format: None,
            },
            None,
        )
        .unwrap_err();

        assert!(err.to_string().contains("Duplicate CSV stable id"));
    }

    #[test]
    fn preview_warns_when_no_id_column_exists() {
        let path = write_fixture(
            "no_id_preview",
            b"given_name,surname,dob\nAna,Santos,1990-01-02\n",
        );

        let preview = load_csv_preview(&CsvPreviewRequestDto {
            path,
            encoding: None,
            delimiter: None,
            date_format: None,
        })
        .unwrap();

        assert!(preview.warnings.iter().any(|w| w.contains("No ID column")));
    }
}
