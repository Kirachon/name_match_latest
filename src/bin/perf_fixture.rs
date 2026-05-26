use anyhow::{bail, Context, Result};
use name_matcher::loaders::csv_loader::{
    load_csv_people, load_csv_people_with_options, CsvDelimiterDto, CsvEncodingDto, CsvLoadOptions,
    CsvPreviewRequestDto,
};
use name_matcher::matching::advanced_matcher::{
    advanced_match_inmemory, AdvColumns, AdvConfig, AdvLevel,
};
use serde::Serialize;
use std::env;
use std::time::Instant;

#[derive(Debug, Serialize)]
struct PerfFixtureOutput {
    mode: String,
    source_rows: usize,
    target_rows: usize,
    matches: usize,
    elapsed_ms: u128,
    digest: u64,
}

fn main() -> Result<()> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() < 4 {
        bail!(
            "Usage: {} <mode:loader-buffered|loader-streaming|l10|l11> <source.csv> <target.csv>",
            args.first().map(String::as_str).unwrap_or("perf_fixture")
        );
    }

    let mode = &args[1];
    let source = &args[2];
    let target = &args[3];
    let start = Instant::now();

    let load_buffered = |path: &str| {
        load_csv_people(
            &CsvPreviewRequestDto {
                path: path.to_string(),
                encoding: None,
                delimiter: None,
                date_format: Some("%Y-%m-%d".to_string()),
            },
            None,
        )
    };
    let load_streaming = |path: &str| {
        load_csv_people_with_options(
            &CsvPreviewRequestDto {
                path: path.to_string(),
                encoding: Some(CsvEncodingDto::Utf8),
                delimiter: Some(CsvDelimiterDto::Comma),
                date_format: Some("%Y-%m-%d".to_string()),
            },
            None,
            &CsvLoadOptions {
                include_extra_fields: false,
                ..CsvLoadOptions::default()
            },
        )
    };

    let (source_people, target_people) = match mode.as_str() {
        "loader-buffered" => (load_buffered(source)?, load_buffered(target)?),
        "loader-streaming" | "l10" | "l11" => (load_streaming(source)?, load_streaming(target)?),
        other => bail!("Unknown perf_fixture mode: {other}"),
    };

    let matches = match mode.as_str() {
        "l10" => advanced_match_inmemory(
            &source_people,
            &target_people,
            &AdvConfig {
                level: AdvLevel::L10FuzzyBirthdateFullMiddle,
                threshold: 0.80,
                cols: AdvColumns::default(),
                allow_birthdate_swap: false,
            },
        ),
        "l11" => advanced_match_inmemory(
            &source_people,
            &target_people,
            &AdvConfig {
                level: AdvLevel::L11FuzzyBirthdateNoMiddle,
                threshold: 0.80,
                cols: AdvColumns::default(),
                allow_birthdate_swap: false,
            },
        ),
        _ => Vec::new(),
    };

    let elapsed_ms = start.elapsed().as_millis();
    let mut digest = 0xcbf29ce484222325u64;
    for p in source_people.iter().chain(target_people.iter()) {
        digest = digest.wrapping_mul(0x100000001b3);
        digest ^= p.id as u64;
    }
    for m in &matches {
        digest = digest.wrapping_mul(0x100000001b3);
        digest ^= m.person1.id as u64;
        digest = digest.wrapping_mul(0x100000001b3);
        digest ^= m.person2.id as u64;
        digest ^= m.confidence.to_bits() as u64;
    }

    let output = PerfFixtureOutput {
        mode: mode.clone(),
        source_rows: source_people.len(),
        target_rows: target_people.len(),
        matches: matches.len(),
        elapsed_ms,
        digest,
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&output).context("serialize perf output")?
    );
    Ok(())
}
