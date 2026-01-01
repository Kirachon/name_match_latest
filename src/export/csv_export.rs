use crate::export::xlsx_export::SummaryContext;
use crate::matching::advanced_matcher::AdvLevel;
use crate::matching::{MatchPair, MatchingAlgorithm};
use anyhow::Result;
use csv::{Writer, WriterBuilder};
use std::collections::BTreeSet;
use std::fs::File;
use std::io::BufWriter;

pub fn export_to_csv(
    results: &[MatchPair],
    path: &str,
    algorithm: MatchingAlgorithm,
    fuzzy_min_confidence: f32,
) -> Result<()> {
    // Collect all extra field names from Table 2 (person2) across all results
    let extra_field_names = collect_extra_field_names(results);

    let file = File::create(path)?;
    let buf_writer = BufWriter::with_capacity(512 * 1024, file);
    let mut w = WriterBuilder::new().from_writer(buf_writer);
    write_headers(&mut w, algorithm, &extra_field_names)?;
    for pair in results {
        write_pair(
            &mut w,
            pair,
            algorithm,
            fuzzy_min_confidence,
            &extra_field_names,
        )?;
    }
    w.flush()?;
    Ok(())
}

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

fn write_headers<W: std::io::Write>(
    w: &mut Writer<W>,
    algorithm: MatchingAlgorithm,
    extra_field_names: &[String],
) -> Result<()> {
    let mut headers: Vec<String> = match algorithm {
        MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => vec![
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
        ],
        MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => vec![
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
        ],
        MatchingAlgorithm::Fuzzy
        | MatchingAlgorithm::FuzzyNoMiddle
        | MatchingAlgorithm::HouseholdGpu
        | MatchingAlgorithm::HouseholdGpuOpt6
        | MatchingAlgorithm::LevenshteinWeighted => vec![
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
        ],
    };

    // Add extra Table2 field headers
    for field_name in extra_field_names {
        headers.push(format!("Table2_{}", field_name));
    }

    // Add final columns
    match algorithm {
        MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => {
            headers.push("is_matched_Infnbd".to_string());
            headers.push("Confidence".to_string());
            headers.push("MatchedFields".to_string());
        }
        MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => {
            headers.push("is_matched_Infnmnbd".to_string());
            headers.push("Confidence".to_string());
            headers.push("MatchedFields".to_string());
        }
        MatchingAlgorithm::Fuzzy
        | MatchingAlgorithm::FuzzyNoMiddle
        | MatchingAlgorithm::LevenshteinWeighted
        | MatchingAlgorithm::HouseholdGpu
        | MatchingAlgorithm::HouseholdGpuOpt6 => {
            headers.push("is_matched_Fuzzy".to_string());
            headers.push("Confidence".to_string());
            headers.push("MatchedFields".to_string());
        }
    }

    w.write_record(&headers)?;
    Ok(())
}

fn write_pair<W: std::io::Write>(
    w: &mut Writer<W>,
    pair: &MatchPair,
    algorithm: MatchingAlgorithm,
    fuzzy_min_confidence: f32,
    extra_field_names: &[String],
) -> Result<()> {
    match algorithm {
        MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => {
            // Pre-format numeric/computed fields to avoid mixing String and &str in Vec
            let id1 = pair.person1.id.to_string();
            let bd1 = pair
                .person1
                .birthdate
                .map(|d| d.to_string())
                .unwrap_or_default();
            let id2 = pair.person2.id.to_string();
            let bd2 = pair
                .person2
                .birthdate
                .map(|d| d.to_string())
                .unwrap_or_default();
            let matched = pair.is_matched_infnbd.to_string();
            let conf = pair.confidence.to_string();
            let fields = pair.matched_fields.join(",");

            let mut record: Vec<&str> = vec![
                &id1,
                pair.person1.uuid.as_deref().unwrap_or(""),
                pair.person1.first_name.as_deref().unwrap_or(""),
                pair.person1.last_name.as_deref().unwrap_or(""),
                &bd1,
                &id2,
                pair.person2.uuid.as_deref().unwrap_or(""),
                pair.person2.first_name.as_deref().unwrap_or(""),
                pair.person2.last_name.as_deref().unwrap_or(""),
                &bd2,
            ];
            // Add extra fields from person2
            for field_name in extra_field_names {
                record.push(
                    pair.person2
                        .extra_fields
                        .get(field_name)
                        .map(|s| s.as_str())
                        .unwrap_or(""),
                );
            }
            // Add final columns
            record.push(&matched);
            record.push(&conf);
            record.push(&fields);
            w.write_record(&record)?;
        }
        MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => {
            // Pre-format numeric/computed fields
            let id1 = pair.person1.id.to_string();
            let bd1 = pair
                .person1
                .birthdate
                .map(|d| d.to_string())
                .unwrap_or_default();
            let id2 = pair.person2.id.to_string();
            let bd2 = pair
                .person2
                .birthdate
                .map(|d| d.to_string())
                .unwrap_or_default();
            let matched = pair.is_matched_infnmnbd.to_string();
            let conf = pair.confidence.to_string();
            let fields = pair.matched_fields.join(",");

            let mut record: Vec<&str> = vec![
                &id1,
                pair.person1.uuid.as_deref().unwrap_or(""),
                pair.person1.first_name.as_deref().unwrap_or(""),
                pair.person1.middle_name.as_deref().unwrap_or(""),
                pair.person1.last_name.as_deref().unwrap_or(""),
                &bd1,
                &id2,
                pair.person2.uuid.as_deref().unwrap_or(""),
                pair.person2.first_name.as_deref().unwrap_or(""),
                pair.person2.middle_name.as_deref().unwrap_or(""),
                pair.person2.last_name.as_deref().unwrap_or(""),
                &bd2,
            ];
            // Add extra fields from person2
            for field_name in extra_field_names {
                record.push(
                    pair.person2
                        .extra_fields
                        .get(field_name)
                        .map(|s| s.as_str())
                        .unwrap_or(""),
                );
            }
            // Add final columns
            record.push(&matched);
            record.push(&conf);
            record.push(&fields);
            w.write_record(&record)?;
        }
        MatchingAlgorithm::Fuzzy
        | MatchingAlgorithm::FuzzyNoMiddle
        | MatchingAlgorithm::LevenshteinWeighted => {
            if pair.confidence < fuzzy_min_confidence {
                // Skip writing fuzzy matches below selected threshold
                return Ok(());
            }
            // Pre-format numeric/computed fields
            let id1 = pair.person1.id.to_string();
            let bd1 = pair
                .person1
                .birthdate
                .map(|d| d.to_string())
                .unwrap_or_default();
            let id2 = pair.person2.id.to_string();
            let bd2 = pair
                .person2
                .birthdate
                .map(|d| d.to_string())
                .unwrap_or_default();
            let conf = pair.confidence.to_string();
            let fields = pair.matched_fields.join(",");

            let mut record: Vec<&str> = vec![
                &id1,
                pair.person1.uuid.as_deref().unwrap_or(""),
                pair.person1.first_name.as_deref().unwrap_or(""),
                pair.person1.middle_name.as_deref().unwrap_or(""),
                pair.person1.last_name.as_deref().unwrap_or(""),
                &bd1,
                &id2,
                pair.person2.uuid.as_deref().unwrap_or(""),
                pair.person2.first_name.as_deref().unwrap_or(""),
                pair.person2.middle_name.as_deref().unwrap_or(""),
                pair.person2.last_name.as_deref().unwrap_or(""),
                &bd2,
            ];
            // Add extra fields from person2
            for field_name in extra_field_names {
                record.push(
                    pair.person2
                        .extra_fields
                        .get(field_name)
                        .map(|s| s.as_str())
                        .unwrap_or(""),
                );
            }
            // Add final columns
            record.push("true");
            record.push(&conf);
            record.push(&fields);
            w.write_record(&record)?;
        }
        MatchingAlgorithm::HouseholdGpu | MatchingAlgorithm::HouseholdGpuOpt6 => {
            // Not applicable for person-level writer
            return Ok(());
        }
    }
    Ok(())
}

use crate::matching::HouseholdAggRow;

pub struct HouseholdCsvWriter {
    writer: Writer<BufWriter<File>>,
    compute_backend: String,
    gpu_model: String,
    gpu_features: String,
}

impl HouseholdCsvWriter {
    /// Backward-compatible constructor (no metadata columns)
    pub fn create(path: &str) -> Result<Self> {
        Self::create_with_meta(path, "", None, "")
    }
    /// New constructor with compute backend metadata
    pub fn create_with_meta(
        path: &str,
        compute_backend: &str,
        gpu_model: Option<&str>,
        gpu_features: &str,
    ) -> Result<Self> {
        let file = File::create(path)?;
        let buf_writer = BufWriter::with_capacity(512 * 1024, file);
        let mut w = WriterBuilder::new().from_writer(buf_writer);
        w.write_record(&[
            "id",
            "uuid",
            "hh_id",
            "match_percentage",
            "region_code",
            "poor_hat_0",
            "poor_hat_10",
            "ComputeBackend",
            "GpuModel",
            "GpuFeatures",
        ])?;
        Ok(Self {
            writer: w,
            compute_backend: compute_backend.to_string(),
            gpu_model: gpu_model.unwrap_or_default().to_string(),
            gpu_features: gpu_features.to_string(),
        })
    }
    pub fn write(&mut self, row: &HouseholdAggRow) -> Result<()> {
        // Pre-format numeric/computed fields
        let id = row.row_id.to_string();
        let hh = row.hh_id.to_string();
        let pct = format!("{:.2}", row.match_percentage);

        self.writer.write_record(&[
            &id,
            row.uuid.as_str(),
            &hh,
            &pct,
            row.region_code.as_deref().unwrap_or(""),
            row.poor_hat_0.as_deref().unwrap_or(""),
            row.poor_hat_10.as_deref().unwrap_or(""),
            self.compute_backend.as_str(),
            self.gpu_model.as_str(),
            self.gpu_features.as_str(),
        ])?;
        Ok(())
    }
    pub fn flush(mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

pub struct CsvStreamWriter {
    writer: Writer<BufWriter<File>>,
    algo: MatchingAlgorithm,
    fuzzy_min_confidence: f32,
    extra_field_names: Vec<String>,
}

impl CsvStreamWriter {
    pub fn create(
        path: &str,
        algorithm: MatchingAlgorithm,
        fuzzy_min_confidence: f32,
    ) -> Result<Self> {
        let file = File::create(path)?;
        let buf_writer = BufWriter::with_capacity(512 * 1024, file);
        let mut writer = WriterBuilder::new().from_writer(buf_writer);
        let extra_field_names = Vec::new(); // Will be populated on first write
        write_headers(&mut writer, algorithm, &extra_field_names)?;
        Ok(Self {
            writer,
            algo: algorithm,
            fuzzy_min_confidence,
            extra_field_names,
        })
    }

    /// Create with known extra field names (for streaming with pre-scanned schema)
    pub fn create_with_extra_fields(
        path: &str,
        algorithm: MatchingAlgorithm,
        fuzzy_min_confidence: f32,
        extra_field_names: Vec<String>,
    ) -> Result<Self> {
        let file = File::create(path)?;
        let buf_writer = BufWriter::with_capacity(512 * 1024, file);
        let mut writer = WriterBuilder::new().from_writer(buf_writer);
        write_headers(&mut writer, algorithm, &extra_field_names)?;
        Ok(Self {
            writer,
            algo: algorithm,
            fuzzy_min_confidence,
            extra_field_names,
        })
    }

    pub fn write(&mut self, pair: &MatchPair) -> Result<()> {
        write_pair(
            &mut self.writer,
            pair,
            self.algo,
            self.fuzzy_min_confidence,
            &self.extra_field_names,
        )
    }
    pub fn flush_partial(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
    pub fn flush(mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

pub struct AdvCsvStreamWriter {
    writer: Writer<BufWriter<File>>,
    extra_field_names: Vec<String>,
    compute_backend: String,
    gpu_model: String,
    gpu_features: String,
}

impl AdvCsvStreamWriter {
    pub fn create_with_extra_fields(
        path: &str,
        extra_field_names: Vec<String>,
        compute_backend: &str,
        gpu_model: Option<&str>,
        gpu_features: &str,
    ) -> Result<Self> {
        let file = File::create(path)?;
        let buf_writer = BufWriter::with_capacity(512 * 1024, file);
        let mut writer = WriterBuilder::new().from_writer(buf_writer);
        // Always include middle name columns for both tables to support all levels uniformly
        let mut headers: Vec<String> = vec![
            "Table1_ID".into(),
            "Table1_UUID".into(),
            "Table1_FirstName".into(),
            "Table1_MiddleName".into(),
            "Table1_LastName".into(),
            "Table1_Birthdate".into(),
            "Table2_ID".into(),
            "Table2_UUID".into(),
            "Table2_FirstName".into(),
            "Table2_MiddleName".into(),
            "Table2_LastName".into(),
            "Table2_Birthdate".into(),
        ];
        for f in &extra_field_names {
            headers.push(format!("Table2_{}", f));
        }
        headers.push("AdvancedLevel".into());
        headers.push("Confidence".into());
        headers.push("MatchedFields".into());
        headers.push("ComputeBackend".into());
        headers.push("GpuModel".into());
        headers.push("GpuFeatures".into());
        writer.write_record(&headers)?;
        Ok(Self {
            writer,
            extra_field_names,
            compute_backend: compute_backend.to_string(),
            gpu_model: gpu_model.unwrap_or_default().to_string(),
            gpu_features: gpu_features.to_string(),
        })
    }

    #[inline]
    fn level_code(level: AdvLevel) -> &'static str {
        match level {
            AdvLevel::L1BirthdateFullMiddle => "L1",
            AdvLevel::L2BirthdateMiddleInitial => "L2",
            AdvLevel::L3BirthdateNoMiddle => "L3",
            AdvLevel::L4BarangayFullMiddle => "L4",
            AdvLevel::L5BarangayMiddleInitial => "L5",
            AdvLevel::L6BarangayNoMiddle => "L6",
            AdvLevel::L7CityFullMiddle => "L7",
            AdvLevel::L8CityMiddleInitial => "L8",
            AdvLevel::L9CityNoMiddle => "L9",
            AdvLevel::L10FuzzyBirthdateFullMiddle => "L10",
            AdvLevel::L11FuzzyBirthdateNoMiddle => "L11",
            AdvLevel::L12HouseholdMatching => "L12",
        }
    }

    pub fn write(&mut self, pair: &MatchPair, level: AdvLevel) -> Result<()> {
        // Pre-format numeric/computed fields
        let id1 = pair.person1.id.to_string();
        let bd1 = pair
            .person1
            .birthdate
            .map(|d| d.to_string())
            .unwrap_or_default();
        let id2 = pair.person2.id.to_string();
        let bd2 = pair
            .person2
            .birthdate
            .map(|d| d.to_string())
            .unwrap_or_default();
        let conf = format!("{:.2}", pair.confidence);
        let fields = pair.matched_fields.join(",");

        let mut record: Vec<&str> = vec![
            &id1,
            pair.person1.uuid.as_deref().unwrap_or(""),
            pair.person1.first_name.as_deref().unwrap_or(""),
            pair.person1.middle_name.as_deref().unwrap_or(""),
            pair.person1.last_name.as_deref().unwrap_or(""),
            &bd1,
            &id2,
            pair.person2.uuid.as_deref().unwrap_or(""),
            pair.person2.first_name.as_deref().unwrap_or(""),
            pair.person2.middle_name.as_deref().unwrap_or(""),
            pair.person2.last_name.as_deref().unwrap_or(""),
            &bd2,
        ];
        for field_name in &self.extra_field_names {
            record.push(
                pair.person2
                    .extra_fields
                    .get(field_name)
                    .map(|s| s.as_str())
                    .unwrap_or(""),
            );
        }
        record.push(Self::level_code(level));
        // Confidence is already normalized to [0,100]; do not rescale here
        record.push(&conf);
        record.push(&fields);
        record.push(self.compute_backend.as_str());
        record.push(self.gpu_model.as_str());
        record.push(self.gpu_features.as_str());
        self.writer.write_record(&record)?;
        Ok(())
    }
    pub fn flush_partial(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
    pub fn flush(mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

pub fn export_summary_csv(path: &str, ctx: &SummaryContext) -> Result<()> {
    let file = File::create(path)?;
    let buf_writer = BufWriter::with_capacity(512 * 1024, file);
    let mut w = WriterBuilder::new().from_writer(buf_writer);
    w.write_record(["Key", "Value"])?;

    let mut write_kv = |k: &str, v: String| -> Result<()> {
        w.write_record(&[k, v.as_str()])?;
        Ok(())
    };

    // Core identifiers
    write_kv("Database", ctx.db_name.clone())?;
    write_kv("Table 1", ctx.table1.clone())?;
    write_kv("Table 2", ctx.table2.clone())?;
    write_kv("Algorithm", ctx.algo_used.clone())?;

    // Totals
    write_kv("Total records (Table1)", ctx.total_table1.to_string())?;
    write_kv("Total records (Table2)", ctx.total_table2.to_string())?;

    // Match counts + execution modes
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
                write_kv("Matches (Direct Match)", ctx.matches_fuzzy.to_string())?;
                if let Some(desc) = ctx.adv_level_description.as_deref() {
                    write_kv("Direct Match Mode", desc.to_string())?;
                }
            }
            // Advanced fuzzy levels (L10-L12): keep using Fuzzy labels
            AdvLevel::L10FuzzyBirthdateFullMiddle
            | AdvLevel::L11FuzzyBirthdateNoMiddle
            | AdvLevel::L12HouseholdMatching => {
                if ctx.matches_fuzzy > 0 {
                    write_kv("Matches (Fuzzy)", ctx.matches_fuzzy.to_string())?;
                }
                if let Some(mode) = ctx.exec_mode_fuzzy.as_deref() {
                    write_kv("Fuzzy Mode", mode.to_string())?;
                }
            }
        }
    } else {
        let is_deterministic_family = ctx.exec_mode_fuzzy.is_none();
        if is_deterministic_family {
            write_kv("Matches (Algorithm 1)", ctx.matches_algo1.to_string())?;
            if let Some(mode) = ctx.exec_mode_algo1.as_deref() {
                write_kv("Algorithm 1 Mode", mode.to_string())?;
            }
            write_kv("Matches (Algorithm 2)", ctx.matches_algo2.to_string())?;
            if let Some(mode) = ctx.exec_mode_algo2.as_deref() {
                write_kv("Algorithm 2 Mode", mode.to_string())?;
            }
        }
        if ctx.matches_fuzzy > 0 {
            write_kv("Matches (Fuzzy)", ctx.matches_fuzzy.to_string())?;
        }
        if let Some(mode) = ctx.exec_mode_fuzzy.as_deref() {
            write_kv("Fuzzy Mode", mode.to_string())?;
        }
    }

    // Timing (GMT+8, human readable)
    let fmt_time = |dt: &chrono::DateTime<chrono::Utc>| -> String {
        let tz = chrono::FixedOffset::east_opt(8 * 3600).unwrap();
        let local = dt.with_timezone(&tz);
        format!("{} GMT+8", local.format("%Y-%m-%d %H:%M:%S"))
    };
    // Human-readable HH:MM:SS (hours may exceed 23)
    let fmt_duration = |secs: f64| -> String {
        let total = secs.floor() as u64;
        let h = total / 3600;
        let m = (total % 3600) / 60;
        let s = total % 60;
        format!("{:02}:{:02}:{:02}", h, m, s)
    };
    write_kv("Started (GMT+8)", fmt_time(&ctx.started_utc))?;
    write_kv("Ended (GMT+8)", fmt_time(&ctx.ended_utc))?;
    write_kv("Duration", fmt_duration(ctx.duration_secs))?;

    // GPU (only when actually used)
    write_kv(
        "GPU Used",
        if ctx.gpu_used {
            "true".to_string()
        } else {
            "false".to_string()
        },
    )?;
    if ctx.gpu_used {
        write_kv("GPU Total (MB)", ctx.gpu_total_mb.to_string())?;
        write_kv("GPU Free End (MB)", ctx.gpu_free_mb_end.to_string())?;
    }

    w.flush()?;
    Ok(())
}
