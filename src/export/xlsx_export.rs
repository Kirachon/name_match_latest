use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::time::Duration;

use anyhow::Result;
use chrono::{DateTime, FixedOffset, Utc};
use rust_xlsxwriter::{Color, Format, FormatAlign, Workbook, Worksheet};

use crate::matching::MatchPair;
use crate::matching::advanced_matcher::AdvLevel;

/// Collect all unique extra field names from person2 across all match pairs (sorted for consistency)
fn collect_extra_field_names(results: &[MatchPair]) -> Vec<String> {
    let mut field_set = BTreeSet::new();
    for pair in results {
        for key in pair.person2.extra_fields.keys() {
            field_set.insert(key.clone());
        }
    }
    field_set.into_iter().collect()
}

#[derive(Debug, Clone)]
pub struct SummaryContext {
    pub db_name: String,
    pub table1: String,
    pub table2: String,

    pub total_table1: usize,
    pub total_table2: usize,

    pub matches_algo1: usize,
    pub matches_algo2: usize,
    pub matches_fuzzy: usize,

    // Legacy optional metrics (may be 0 when not applicable)
    pub overlap_count: usize,
    pub unique_algo1: usize,
    pub unique_algo2: usize,

    // Timing
    pub fetch_time: Duration,
    pub match1_time: Duration,
    pub match2_time: Duration,
    pub export_time: Duration,

    // Memory
    pub mem_used_start_mb: u64,
    pub mem_used_end_mb: u64,

    // Run window (UTC)
    pub started_utc: DateTime<Utc>,
    pub ended_utc: DateTime<Utc>,
    pub duration_secs: f64,

    // Execution modes per algorithm ("CPU"|"GPU"). None when not run.
    pub exec_mode_algo1: Option<String>,
    pub exec_mode_algo2: Option<String>,
    pub exec_mode_fuzzy: Option<String>,

    // Overall algorithm selection (kept for backward-compat display)
    pub algo_used: String,

    // Actual GPU usage (true only if GPU kernels ran during this operation)
    pub gpu_used: bool,
    pub gpu_total_mb: u64,
    pub gpu_free_mb_end: u64,

    // Advanced Matching context (optional; used to specialize summary labels)
    pub adv_level: Option<AdvLevel>,
    pub adv_level_description: Option<String>,
}

fn ensure_parent_dir(path: &str) -> Result<()> {
    let p = Path::new(path);
    if let Some(parent) = p.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

fn header_format() -> Format {
    Format::new().set_bold().set_align(FormatAlign::Center)
}

fn row_format_even() -> Format {
    Format::new().set_background_color(Color::RGB(0xF2F2F2))
}

/// Write only headers for algo1 sheet (for streaming)
fn write_algo1_sheet_headers(ws: &mut Worksheet, extra_field_names: &[String]) -> Result<()> {
    let mut headers: Vec<String> = vec![
        "Table1_ID".to_string(),
        "Table1_UUID".to_string(),
        "Table1_FirstName".to_string(),
        "Table1_LastName".to_string(),
        "Table1_Birthdate".to_string(),
        "Table2_ID".to_string(),
        "Table2_UUID".to_string(),
        "Table2_FirstName".to_string(),
        "Table2_LastName".to_string(),
        "Table2_Birthdate".to_string(),
    ];
    for field_name in extra_field_names {
        headers.push(format!("Table2_{}", field_name));
    }
    headers.push("is_matched_Infnbd".to_string());
    headers.push("Confidence".to_string());
    headers.push("MatchedFields".to_string());

    let hfmt = header_format();
    for (c, h) in headers.iter().enumerate() {
        ws.write_string_with_format(0, c as u16, h, &hfmt)?;
    }
    Ok(())
}

/// Write only headers for algo2 sheet (for streaming)
fn write_algo2_sheet_headers(ws: &mut Worksheet, extra_field_names: &[String]) -> Result<()> {
    let mut headers: Vec<String> = vec![
        "Table1_ID".to_string(),
        "Table1_UUID".to_string(),
        "Table1_FirstName".to_string(),
        "Table1_MiddleName".to_string(),
        "Table1_LastName".to_string(),
        "Table1_Birthdate".to_string(),
        "Table2_ID".to_string(),
        "Table2_UUID".to_string(),
        "Table2_FirstName".to_string(),
        "Table2_MiddleName".to_string(),
        "Table2_LastName".to_string(),
        "Table2_Birthdate".to_string(),
    ];
    for field_name in extra_field_names {
        headers.push(format!("Table2_{}", field_name));
    }
    headers.push("is_matched_Infnmnbd".to_string());
    headers.push("Confidence".to_string());
    headers.push("MatchedFields".to_string());

    let hfmt = header_format();
    for (c, h) in headers.iter().enumerate() {
        ws.write_string_with_format(0, c as u16, h, &hfmt)?;
    }
    Ok(())
}

fn write_algo1_sheet(ws: &mut Worksheet, matches: &[MatchPair]) -> Result<()> {
    let extra_field_names = collect_extra_field_names(matches);

    let mut headers: Vec<String> = vec![
        "Table1_ID".to_string(),
        "Table1_UUID".to_string(),
        "Table1_FirstName".to_string(),
        "Table1_LastName".to_string(),
        "Table1_Birthdate".to_string(),
        "Table2_ID".to_string(),
        "Table2_UUID".to_string(),
        "Table2_FirstName".to_string(),
        "Table2_LastName".to_string(),
        "Table2_Birthdate".to_string(),
    ];

    // Add extra Table2 field headers
    for field_name in &extra_field_names {
        headers.push(format!("Table2_{}", field_name));
    }

    // Add final columns
    headers.push("is_matched_Infnbd".to_string());
    headers.push("Confidence".to_string());
    headers.push("MatchedFields".to_string());

    let hfmt = header_format();
    for (c, h) in headers.iter().enumerate() {
        ws.write_string_with_format(0, c as u16, h, &hfmt)?;
    }

    let even = row_format_even();
    for (i, m) in matches.iter().enumerate() {
        let r = (i + 1) as u32;
        if i % 2 == 0 {
            ws.set_row_format(r, &even)?;
        }
        let mut col: u16 = 0;
        ws.write_number(r, col, m.person1.id as f64)?;
        col += 1;
        ws.write_string(r, col, m.person1.uuid.as_deref().unwrap_or(""))?;
        col += 1;
        ws.write_string(r, col, m.person1.first_name.as_deref().unwrap_or(""))?;
        col += 1;
        ws.write_string(r, col, m.person1.last_name.as_deref().unwrap_or(""))?;
        col += 1;
        ws.write_string(
            r,
            col,
            &m.person1
                .birthdate
                .as_ref()
                .map(|d| d.to_string())
                .unwrap_or_default(),
        )?;
        col += 1;
        ws.write_number(r, col, m.person2.id as f64)?;
        col += 1;
        ws.write_string(r, col, m.person2.uuid.as_deref().unwrap_or(""))?;
        col += 1;
        ws.write_string(r, col, m.person2.first_name.as_deref().unwrap_or(""))?;
        col += 1;
        ws.write_string(r, col, m.person2.last_name.as_deref().unwrap_or(""))?;
        col += 1;
        ws.write_string(
            r,
            col,
            &m.person2
                .birthdate
                .as_ref()
                .map(|d| d.to_string())
                .unwrap_or_default(),
        )?;
        col += 1;

        // Write extra fields from person2
        for field_name in &extra_field_names {
            ws.write_string(
                r,
                col,
                m.person2
                    .extra_fields
                    .get(field_name)
                    .map(|s| s.as_str())
                    .unwrap_or(""),
            )?;
            col += 1;
        }

        ws.write_string(r, col, if m.is_matched_infnbd { "true" } else { "false" })?;
        col += 1;
        ws.write_number(r, col, m.confidence as f64)?;
        col += 1;
        ws.write_string(r, col, &m.matched_fields.join(";"))?;
    }
    Ok(())
}

fn write_algo2_sheet(ws: &mut Worksheet, matches: &[MatchPair]) -> Result<()> {
    let extra_field_names = collect_extra_field_names(matches);

    let mut headers: Vec<String> = vec![
        "Table1_ID".to_string(),
        "Table1_UUID".to_string(),
        "Table1_FirstName".to_string(),
        "Table1_MiddleName".to_string(),
        "Table1_LastName".to_string(),
        "Table1_Birthdate".to_string(),
        "Table2_ID".to_string(),
        "Table2_UUID".to_string(),
        "Table2_FirstName".to_string(),
        "Table2_MiddleName".to_string(),
        "Table2_LastName".to_string(),
        "Table2_Birthdate".to_string(),
    ];

    // Add extra Table2 field headers
    for field_name in &extra_field_names {
        headers.push(format!("Table2_{}", field_name));
    }

    // Add final columns
    headers.push("is_matched_Infnmnbd".to_string());
    headers.push("Confidence".to_string());
    headers.push("MatchedFields".to_string());

    let hfmt = header_format();
    for (c, h) in headers.iter().enumerate() {
        ws.write_string_with_format(0, c as u16, h, &hfmt)?;
    }

    let even = row_format_even();
    for (i, m) in matches.iter().enumerate() {
        let r = (i + 1) as u32;
        if i % 2 == 0 {
            ws.set_row_format(r, &even)?;
        }
        let mut col: u16 = 0;
        ws.write_number(r, col, m.person1.id as f64)?;
        col += 1;
        ws.write_string(r, col, m.person1.uuid.as_deref().unwrap_or(""))?;
        col += 1;
        ws.write_string(r, col, m.person1.first_name.as_deref().unwrap_or(""))?;
        col += 1;
        ws.write_string(r, col, m.person1.middle_name.as_deref().unwrap_or(""))?;
        col += 1;
        ws.write_string(r, col, m.person1.last_name.as_deref().unwrap_or(""))?;
        col += 1;
        ws.write_string(
            r,
            col,
            &m.person1
                .birthdate
                .as_ref()
                .map(|d| d.to_string())
                .unwrap_or_default(),
        )?;
        col += 1;
        ws.write_number(r, col, m.person2.id as f64)?;
        col += 1;
        ws.write_string(r, col, m.person2.uuid.as_deref().unwrap_or(""))?;
        col += 1;
        ws.write_string(r, col, m.person2.first_name.as_deref().unwrap_or(""))?;
        col += 1;
        ws.write_string(r, col, m.person2.middle_name.as_deref().unwrap_or(""))?;
        col += 1;
        ws.write_string(r, col, m.person2.last_name.as_deref().unwrap_or(""))?;
        col += 1;
        ws.write_string(
            r,
            col,
            &m.person2
                .birthdate
                .as_ref()
                .map(|d| d.to_string())
                .unwrap_or_default(),
        )?;
        col += 1;

        // Write extra fields from person2
        for field_name in &extra_field_names {
            ws.write_string(
                r,
                col,
                m.person2
                    .extra_fields
                    .get(field_name)
                    .map(|s| s.as_str())
                    .unwrap_or(""),
            )?;
            col += 1;
        }

        ws.write_string(
            r,
            col,
            if m.is_matched_infnmnbd {
                "true"
            } else {
                "false"
            },
        )?;
        col += 1;
        ws.write_number(r, col, m.confidence as f64)?;
        col += 1;
        ws.write_string(r, col, &m.matched_fields.join(";"))?;
    }
    Ok(())
}

fn write_summary_sheet(ws: &mut Worksheet, ctx: &SummaryContext) -> Result<()> {
    let hfmt = header_format();
    let mut row: u32 = 0;

    ws.write_string_with_format(row, 0, "Summary", &hfmt)?;
    row += 2;

    let kv = |ws: &mut Worksheet, r: &mut u32, k: &str, v: &str| -> Result<()> {
        ws.write_string(*r, 0, k)?;
        ws.write_string(*r, 1, v)?;
        *r += 1;
        Ok(())
    };

    // Core identifiers
    kv(ws, &mut row, "Database", &ctx.db_name)?;
    kv(ws, &mut row, "Table 1", &ctx.table1)?;
    kv(ws, &mut row, "Table 2", &ctx.table2)?;
    kv(ws, &mut row, "Algorithm", &ctx.algo_used)?;

    // Totals
    kv(
        ws,
        &mut row,
        "Total records (Table1)",
        &ctx.total_table1.to_string(),
    )?;
    kv(
        ws,
        &mut row,
        "Total records (Table2)",
        &ctx.total_table2.to_string(),
    )?;

    // Match counts and execution modes per algorithm
    if let Some(level) = ctx.adv_level {
        match level {
            // Advanced exact levels (L1-L9): use Direct Match template and level description
            AdvLevel::L1BirthdateFullMiddle
            | AdvLevel::L2BirthdateMiddleInitial
            | AdvLevel::L3BirthdateNoMiddle
            | AdvLevel::L4BarangayFullMiddle
            | AdvLevel::L5BarangayMiddleInitial
            | AdvLevel::L6BarangayNoMiddle
            | AdvLevel::L7CityFullMiddle
            | AdvLevel::L8CityMiddleInitial
            | AdvLevel::L9CityNoMiddle => {
                kv(
                    ws,
                    &mut row,
                    "Matches (Direct Match)",
                    &ctx.matches_fuzzy.to_string(),
                )?;
                if let Some(desc) = ctx.adv_level_description.as_deref() {
                    kv(ws, &mut row, "Direct Match Mode", desc)?;
                }
            }
            // Advanced fuzzy levels (L10-L12): keep using Fuzzy labels
            AdvLevel::L10FuzzyBirthdateFullMiddle
            | AdvLevel::L11FuzzyBirthdateNoMiddle
            | AdvLevel::L12HouseholdMatching => {
                if ctx.matches_fuzzy > 0 {
                    kv(
                        ws,
                        &mut row,
                        "Matches (Fuzzy)",
                        &ctx.matches_fuzzy.to_string(),
                    )?;
                }
                if let Some(mode) = ctx.exec_mode_fuzzy.as_deref() {
                    kv(ws, &mut row, "Fuzzy Mode", mode)?;
                }
            }
        }
    } else {
        // Standard Options 1-7 path (backward compatible)
        let is_deterministic_family = ctx.exec_mode_fuzzy.is_none();
        if is_deterministic_family {
            kv(
                ws,
                &mut row,
                "Matches (Algorithm 1)",
                &ctx.matches_algo1.to_string(),
            )?;
            if let Some(mode) = ctx.exec_mode_algo1.as_deref() {
                kv(ws, &mut row, "Algorithm 1 Mode", mode)?;
            }
            kv(
                ws,
                &mut row,
                "Matches (Algorithm 2)",
                &ctx.matches_algo2.to_string(),
            )?;
            if let Some(mode) = ctx.exec_mode_algo2.as_deref() {
                kv(ws, &mut row, "Algorithm 2 Mode", mode)?;
            }
        }
        if ctx.matches_fuzzy > 0 {
            kv(
                ws,
                &mut row,
                "Matches (Fuzzy)",
                &ctx.matches_fuzzy.to_string(),
            )?;
        }
        if let Some(mode) = ctx.exec_mode_fuzzy.as_deref() {
            kv(ws, &mut row, "Fuzzy Mode", mode)?;
        }
    }

    // Timing (GMT+8, human readable)
    let fmt_time = |dt: &DateTime<Utc>| -> String {
        let tz = FixedOffset::east_opt(8 * 3600).unwrap();
        let local = dt.with_timezone(&tz);
        format!("{} GMT+8", local.format("%Y-%m-%d %H:%M:%S"))
    };
    // Human-readable HH:MM:SS duration (hours may exceed 23)
    let fmt_duration = |secs: f64| -> String {
        let total = secs.floor() as u64;
        let h = total / 3600;
        let m = (total % 3600) / 60;
        let s = total % 60;
        format!("{:02}:{:02}:{:02}", h, m, s)
    };
    kv(ws, &mut row, "Started (GMT+8)", &fmt_time(&ctx.started_utc))?;
    kv(ws, &mut row, "Ended (GMT+8)", &fmt_time(&ctx.ended_utc))?;
    kv(ws, &mut row, "Duration", &fmt_duration(ctx.duration_secs))?;

    // GPU (only when actually used)
    kv(
        ws,
        &mut row,
        "GPU Used",
        if ctx.gpu_used { "true" } else { "false" },
    )?;
    if ctx.gpu_used {
        kv(
            ws,
            &mut row,
            "GPU Total (MB)",
            &ctx.gpu_total_mb.to_string(),
        )?;
        kv(
            ws,
            &mut row,
            "GPU Free End (MB)",
            &ctx.gpu_free_mb_end.to_string(),
        )?;
    }

    Ok(())
}

pub struct XlsxStreamWriter {
    workbook: Workbook,
    next_r1: u32,
    next_r2: u32,
    out_path: String,
    extra_field_names: Vec<String>,
}

impl XlsxStreamWriter {
    pub fn create(out_path: &str) -> Result<Self> {
        let mut workbook = Workbook::new();
        {
            let mut ws1 = workbook.add_worksheet();
            ws1.set_name("Algorithm_1_Results")?;
            write_algo1_sheet(&mut ws1, &[])?;
        }
        {
            let mut ws2 = workbook.add_worksheet();
            ws2.set_name("Algorithm_2_Results")?;
            write_algo2_sheet(&mut ws2, &[])?;
        }
        {
            let ws3 = workbook.add_worksheet();
            ws3.set_name("Summary")?;
        }
        Ok(Self {
            workbook,
            next_r1: 1,
            next_r2: 1,
            out_path: out_path.to_string(),
            extra_field_names: Vec::new(),
        })
    }

    /// Create with known extra field names (for streaming with pre-scanned schema)
    pub fn create_with_extra_fields(
        out_path: &str,
        extra_field_names: Vec<String>,
    ) -> Result<Self> {
        let mut workbook = Workbook::new();
        {
            let mut ws1 = workbook.add_worksheet();
            ws1.set_name("Algorithm_1_Results")?;
            // Write headers with extra fields
            write_algo1_sheet_headers(&mut ws1, &extra_field_names)?;
        }
        {
            let mut ws2 = workbook.add_worksheet();
            ws2.set_name("Algorithm_2_Results")?;
            // Write headers with extra fields
            write_algo2_sheet_headers(&mut ws2, &extra_field_names)?;
        }
        {
            let ws3 = workbook.add_worksheet();
            ws3.set_name("Summary")?;
        }
        Ok(Self {
            workbook,
            next_r1: 1,
            next_r2: 1,
            out_path: out_path.to_string(),
            extra_field_names,
        })
    }
    pub fn append_algo1(&mut self, m: &MatchPair) -> Result<()> {
        let r = self.next_r1;
        self.next_r1 += 1;
        let even = row_format_even();
        {
            let sheets = self.workbook.worksheets_mut();
            let ws = &mut sheets[0];
            if (r as usize - 1) % 2 == 0 {
                ws.set_row_format(r, &even)?;
            }
            let mut col: u16 = 0;
            ws.write_number(r, col, m.person1.id as f64)?;
            col += 1;
            ws.write_string(r, col, m.person1.uuid.as_deref().unwrap_or(""))?;
            col += 1;
            ws.write_string(r, col, m.person1.first_name.as_deref().unwrap_or(""))?;
            col += 1;
            ws.write_string(r, col, m.person1.last_name.as_deref().unwrap_or(""))?;
            col += 1;
            ws.write_string(
                r,
                col,
                &m.person1
                    .birthdate
                    .as_ref()
                    .map(|d| d.to_string())
                    .unwrap_or_default(),
            )?;
            col += 1;
            ws.write_number(r, col, m.person2.id as f64)?;
            col += 1;
            ws.write_string(r, col, m.person2.uuid.as_deref().unwrap_or(""))?;
            col += 1;
            ws.write_string(r, col, m.person2.first_name.as_deref().unwrap_or(""))?;
            col += 1;
            ws.write_string(r, col, m.person2.last_name.as_deref().unwrap_or(""))?;
            col += 1;
            ws.write_string(
                r,
                col,
                &m.person2
                    .birthdate
                    .as_ref()
                    .map(|d| d.to_string())
                    .unwrap_or_default(),
            )?;
            col += 1;

            // Write extra fields from person2
            for field_name in &self.extra_field_names {
                ws.write_string(
                    r,
                    col,
                    m.person2
                        .extra_fields
                        .get(field_name)
                        .map(|s| s.as_str())
                        .unwrap_or(""),
                )?;
                col += 1;
            }

            ws.write_string(r, col, if m.is_matched_infnbd { "true" } else { "false" })?;
            col += 1;
            ws.write_number(r, col, m.confidence as f64)?;
            col += 1;
            ws.write_string(r, col, &m.matched_fields.join(";"))?;
        }
        Ok(())
    }
    pub fn append_algo2(&mut self, m: &MatchPair) -> Result<()> {
        let r = self.next_r2;
        self.next_r2 += 1;
        let even = row_format_even();
        {
            let sheets = self.workbook.worksheets_mut();
            let ws = &mut sheets[1];
            if (r as usize - 1) % 2 == 0 {
                ws.set_row_format(r, &even)?;
            }
            let mut col: u16 = 0;
            ws.write_number(r, col, m.person1.id as f64)?;
            col += 1;
            ws.write_string(r, col, m.person1.uuid.as_deref().unwrap_or(""))?;
            col += 1;
            ws.write_string(r, col, m.person1.first_name.as_deref().unwrap_or(""))?;
            col += 1;
            ws.write_string(r, col, m.person1.middle_name.as_deref().unwrap_or(""))?;
            col += 1;
            ws.write_string(r, col, m.person1.last_name.as_deref().unwrap_or(""))?;
            col += 1;
            ws.write_string(
                r,
                col,
                &m.person1
                    .birthdate
                    .as_ref()
                    .map(|d| d.to_string())
                    .unwrap_or_default(),
            )?;
            col += 1;
            ws.write_number(r, col, m.person2.id as f64)?;
            col += 1;
            ws.write_string(r, col, m.person2.uuid.as_deref().unwrap_or(""))?;
            col += 1;
            ws.write_string(r, col, m.person2.first_name.as_deref().unwrap_or(""))?;
            col += 1;
            ws.write_string(r, col, m.person2.middle_name.as_deref().unwrap_or(""))?;
            col += 1;
            ws.write_string(r, col, m.person2.last_name.as_deref().unwrap_or(""))?;
            col += 1;
            ws.write_string(
                r,
                col,
                &m.person2
                    .birthdate
                    .as_ref()
                    .map(|d| d.to_string())
                    .unwrap_or_default(),
            )?;
            col += 1;

            // Write extra fields from person2
            for field_name in &self.extra_field_names {
                ws.write_string(
                    r,
                    col,
                    m.person2
                        .extra_fields
                        .get(field_name)
                        .map(|s| s.as_str())
                        .unwrap_or(""),
                )?;
                col += 1;
            }

            ws.write_string(
                r,
                col,
                if m.is_matched_infnmnbd {
                    "true"
                } else {
                    "false"
                },
            )?;
            col += 1;
            ws.write_number(r, col, m.confidence as f64)?;
            col += 1;
            ws.write_string(r, col, &m.matched_fields.join(";"))?;
        }
        Ok(())
    }
    pub fn finalize(mut self, summary: &SummaryContext) -> Result<()> {
        {
            let sheets = self.workbook.worksheets_mut();
            let ws = &mut sheets[2];
            write_summary_sheet(ws, summary)?;
        }
        self.workbook.save(&self.out_path)?;
        Ok(())
    }
}

pub fn export_to_xlsx(
    algo1_matches: &[MatchPair],
    algo2_matches: &[MatchPair],
    out_path: &str,
    summary: &SummaryContext,
) -> Result<()> {
    ensure_parent_dir(out_path)?;

    let mut workbook = Workbook::new();

    // Algorithm 1 sheet
    let mut sheet1 = workbook.add_worksheet();
    sheet1.set_name("Algorithm_1_Results")?;
    write_algo1_sheet(&mut sheet1, algo1_matches)?;

    // Algorithm 2 sheet
    let mut sheet2 = workbook.add_worksheet();
    sheet2.set_name("Algorithm_2_Results")?;
    write_algo2_sheet(&mut sheet2, algo2_matches)?;

    // Summary sheet
    let mut sheet3 = workbook.add_worksheet();
    sheet3.set_name("Summary")?;
    write_summary_sheet(&mut sheet3, summary)?;

    workbook.save(out_path)?;
    Ok(())
}

pub fn export_summary_xlsx(out_path: &str, summary: &SummaryContext) -> Result<()> {
    ensure_parent_dir(out_path)?;
    let mut workbook = Workbook::new();
    let mut sheet = workbook.add_worksheet();
    sheet.set_name("Summary")?;
    write_summary_sheet(&mut sheet, summary)?;
    workbook.save(out_path)?;
    Ok(())
}

use crate::matching::HouseholdAggRow;

pub fn export_households_xlsx(out_path: &str, rows: &[HouseholdAggRow]) -> Result<()> {
    ensure_parent_dir(out_path)?;
    let mut workbook = Workbook::new();
    let sheet = workbook.add_worksheet();
    sheet.set_name("Household_Matches")?;

    let hfmt = header_format();
    sheet.write_string_with_format(0, 0, "id", &hfmt)?;
    sheet.write_string_with_format(0, 1, "uuid", &hfmt)?;
    sheet.write_string_with_format(0, 2, "hh_id", &hfmt)?;
    sheet.write_string_with_format(0, 3, "match_percentage", &hfmt)?;
    sheet.write_string_with_format(0, 4, "region_code", &hfmt)?;
    sheet.write_string_with_format(0, 5, "poor_hat_0", &hfmt)?;
    sheet.write_string_with_format(0, 6, "poor_hat_10", &hfmt)?;

    let even = row_format_even();
    for (i, r) in rows.iter().enumerate() {
        let row = (i as u32) + 1;
        if i % 2 == 0 {
            sheet.set_row_format(row, &even)?;
        }
        sheet.write_number(row, 0, r.row_id as f64)?;
        sheet.write_string(row, 1, &r.uuid)?;
        sheet.write_string(row, 2, &r.hh_id.to_string())?;
        sheet.write_number(row, 3, r.match_percentage as f64)?;
        sheet.write_string(row, 4, r.region_code.as_deref().unwrap_or(""))?;
        sheet.write_string(row, 5, r.poor_hat_0.as_deref().unwrap_or(""))?;
        sheet.write_string(row, 6, r.poor_hat_10.as_deref().unwrap_or(""))?;
    }

    workbook.save(out_path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Person;
    use chrono::NaiveDate;

    fn p(id: i64, f: &str, m: Option<&str>, l: &str, d: (i32, u32, u32)) -> Person {
        Person {
            id,
            uuid: Some(format!("u{}", id)),
            first_name: Some(f.into()),
            middle_name: m.map(|s| s.to_string()),
            last_name: Some(l.into()),
            birthdate: NaiveDate::from_ymd_opt(d.0, d.1, d.2),
            hh_id: None,
            extra_fields: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn write_xlsx_basic() {
        let a1 = vec![MatchPair {
            person1: p(1, "A", None, "Z", (2000, 1, 1)),
            person2: p(2, "A", None, "Z", (2000, 1, 1)),
            confidence: 1.0,
            matched_fields: vec!["first_name".into(), "last_name".into(), "birthdate".into()],
            is_matched_infnbd: true,
            is_matched_infnmnbd: false,
        }];
        let a2: Vec<MatchPair> = vec![];
        let out = "./target/test_matches.xlsx";
        let _ = std::fs::remove_file(out);
        let summary = SummaryContext {
            db_name: "db".into(),
            table1: "t1".into(),
            table2: "t2".into(),
            total_table1: 1,
            total_table2: 1,
            matches_algo1: 1,
            matches_algo2: 0,
            matches_fuzzy: 0,
            overlap_count: 0,
            unique_algo1: 1,
            unique_algo2: 0,
            fetch_time: std::time::Duration::from_millis(1),
            match1_time: std::time::Duration::from_millis(1),
            match2_time: std::time::Duration::from_millis(1),
            export_time: std::time::Duration::from_millis(0),
            mem_used_start_mb: 0,
            mem_used_end_mb: 0,
            started_utc: chrono::Utc::now(),
            ended_utc: chrono::Utc::now(),
            duration_secs: 0.0,
            exec_mode_algo1: Some("CPU".into()),
            exec_mode_algo2: Some("CPU".into()),
            exec_mode_fuzzy: None,
            algo_used: "Both (1,2)".into(),
            gpu_used: false,
            gpu_total_mb: 0,
            gpu_free_mb_end: 0,
            adv_level: None,
            adv_level_description: None,
        };
        let res = export_to_xlsx(&a1, &a2, out, &summary);
        assert!(res.is_ok());
        let meta = std::fs::metadata(out).unwrap();
        assert!(meta.len() > 0);
    }
}
