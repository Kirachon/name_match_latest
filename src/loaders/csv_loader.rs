use anyhow::{Context, Result, bail};
use chrono::NaiveDate;
use encoding_rs::{Encoding, UTF_8, WINDOWS_1252};
use serde::{Deserialize, Serialize};
use std::path::Path;

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

pub fn load_csv_preview(request: &CsvPreviewRequestDto) -> Result<CsvPreviewDto> {
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

    Ok(CsvPreviewDto {
        path: request.path.clone(),
        encoding: detected_encoding,
        delimiter,
        headers,
        total_preview_rows: rows.len(),
        rows,
        warnings,
        date_format,
    })
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
    Ok(())
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
}
