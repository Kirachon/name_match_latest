use crate::matching::{
    ComputeBackend, GpuConfig, MatchOptions, MatchingAlgorithm, ProgressConfig,
    last_gpu_fuzzy_stats, match_all_with_opts, set_gpu_fuzzy_disable, set_gpu_fuzzy_force,
    set_gpu_fuzzy_metrics,
};
use crate::models::Person;
use anyhow::{Result, anyhow, bail};
use chrono::NaiveDate;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

const REPORT_SCHEMA_VERSION: u32 = 1;
#[cfg(feature = "experimental-stringzilla-cuda")]
const STRINGZILLA_SHADOW_CHUNK_SIZE: usize = 4_096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchBackend {
    Cpu,
    Gpu,
}

impl BenchBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            BenchBackend::Cpu => "cpu",
            BenchBackend::Gpu => "gpu",
        }
    }
}

#[derive(Debug, Clone)]
pub struct BenchConfig {
    pub dataset: BenchDatasetKind,
    pub backend: BenchBackend,
    pub warmup_runs: usize,
    pub measured_runs: usize,
    pub include_unavailable_engines: bool,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            dataset: BenchDatasetKind::Small,
            backend: BenchBackend::Cpu,
            warmup_runs: 2,
            measured_runs: 5,
            include_unavailable_engines: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum BenchDatasetKind {
    Small,
    Medium,
    Large,
    DuplicateHeavy,
    Sparse,
    Messy,
}

impl BenchDatasetKind {
    pub fn as_str(self) -> &'static str {
        match self {
            BenchDatasetKind::Small => "small",
            BenchDatasetKind::Medium => "medium",
            BenchDatasetKind::Large => "large",
            BenchDatasetKind::DuplicateHeavy => "duplicate-heavy",
            BenchDatasetKind::Sparse => "sparse",
            BenchDatasetKind::Messy => "messy",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "small" => Some(Self::Small),
            "medium" => Some(Self::Medium),
            "large" => Some(Self::Large),
            "duplicate-heavy" | "duplicate_heavy" | "duplicates" => Some(Self::DuplicateHeavy),
            "sparse" => Some(Self::Sparse),
            "messy" => Some(Self::Messy),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkReport {
    pub schema_version: u32,
    pub generated_unix_ms: u128,
    pub git_sha: Option<String>,
    pub feature_flags: FeatureFlagsReport,
    pub dataset: DatasetManifest,
    pub config: RunConfigReport,
    pub hardware: HardwareReport,
    pub engines: Vec<EngineReport>,
    pub summary: DecisionSummary,
}

#[derive(Debug, Clone, Serialize)]
pub struct FeatureFlagsReport {
    pub gpu: bool,
    pub gpu_bench: bool,
    pub experimental_stringzilla_cuda: bool,
    pub experimental_rapids_bench: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct RunConfigReport {
    pub backend: String,
    pub warmup_runs: usize,
    pub measured_runs: usize,
    pub timing_unit: &'static str,
    pub primary_gate: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct HardwareReport {
    pub nvidia_smi: TelemetryAvailability,
}

#[derive(Debug, Clone, Serialize)]
pub struct TelemetryAvailability {
    pub available: bool,
    pub reason: Option<String>,
    pub raw_summary: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DatasetManifest {
    pub id: String,
    pub hash_sha256: String,
    pub seed: u64,
    pub left_rows: usize,
    pub right_rows: usize,
    pub expected_current_matches: Option<usize>,
    pub notes: Vec<String>,
    pub left_string_lengths: LengthStats,
    pub right_string_lengths: LengthStats,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct LengthStats {
    pub min: usize,
    pub p50: usize,
    pub p95: usize,
    pub max: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct EngineReport {
    pub engine_id: String,
    pub availability: EngineAvailability,
    pub timings: Option<TimingSummary>,
    pub throughput: Option<ThroughputSummary>,
    pub output: Option<OutputSummary>,
    pub gpu_stats: Option<GpuStatsReport>,
    pub parity: Option<ParityReport>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "status", rename_all = "kebab-case")]
pub enum EngineAvailability {
    Available,
    Unavailable { reason: String },
    Error { reason: String },
}

#[derive(Debug, Clone, Serialize)]
pub struct TimingSummary {
    pub cold_total_us: u128,
    pub measured_total_us: Distribution,
}

#[derive(Debug, Clone, Serialize)]
pub struct Distribution {
    pub min: u128,
    pub p50: u128,
    pub p95: u128,
    pub p99: u128,
    pub max: u128,
    pub mean: f64,
    pub stddev: f64,
    pub samples: Vec<u128>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ThroughputSummary {
    pub matches_per_sec_p50: f64,
    pub left_rows_per_sec_p50: f64,
    pub right_rows_per_sec_p50: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct OutputSummary {
    pub matches_emitted: usize,
    pub output_hash_sha256: String,
    pub canonical_output_hash_sha256: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct GpuStatsReport {
    pub input_rows_left: u64,
    pub input_rows_right: u64,
    pub candidate_pairs_seen: u64,
    pub pairs_uploaded: u64,
    pub matches_emitted: u64,
    pub normalization_time_us: u128,
    pub candidate_blocking_time_us: u128,
    pub cache_build_time_us: u128,
    pub candidate_scan_time_us: u128,
    pub h2d_time_us: u128,
    pub kernel_time_us: u128,
    pub d2h_time_us: u128,
    pub cpu_classification_time_us: u128,
    pub total_wall_time_us: u128,
    pub gpu_tile_count: u64,
    pub gpu_mem_query_count: u64,
    pub tile_size_min: u64,
    pub tile_size_max: u64,
    pub candidates_per_source_p50: u64,
    pub candidates_per_source_p95: u64,
    pub candidates_per_source_max: u64,
    pub resident_tables_enabled: bool,
    pub resident_table_bytes: u64,
    pub resident_table_upload_us: u128,
    pub batch_index_h2d_us: u128,
}

#[derive(Debug, Clone, Serialize)]
pub struct ParityReport {
    pub oracle_engine_id: String,
    pub false_negatives_vs_current: u64,
    pub false_positives_vs_current: u64,
    pub false_negative_pair_ids: Vec<PairIdSample>,
    pub false_positive_pair_ids: Vec<PairIdSample>,
    pub pair_ids_match_current: bool,
    pub canonical_output_hash_matches_current: bool,
    pub output_hash_matches_current: bool,
    pub blocking_failure: bool,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PairIdSample {
    pub left_id: i64,
    pub right_id: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct DecisionSummary {
    pub recommendation: String,
    pub reason: String,
}

pub fn run_benchmark(config: BenchConfig) -> Result<BenchmarkReport> {
    if config.measured_runs == 0 {
        bail!("measured_runs must be greater than zero");
    }

    let fixture = build_dataset(config.dataset);
    let dataset = dataset_manifest(config.dataset, &fixture);
    let current = run_current_engine(&fixture, &config)?;
    let mut engines = vec![current];

    if config.include_unavailable_engines {
        engines.push(stringzilla_engine_report(&fixture, &config));
        engines.push(rapids_engine_report());
    }

    let summary = summarize_decision(&engines);
    Ok(BenchmarkReport {
        schema_version: REPORT_SCHEMA_VERSION,
        generated_unix_ms: unix_ms(),
        git_sha: git_sha(),
        feature_flags: FeatureFlagsReport {
            gpu: cfg!(feature = "gpu"),
            gpu_bench: cfg!(feature = "gpu-bench"),
            experimental_stringzilla_cuda: cfg!(feature = "experimental-stringzilla-cuda"),
            experimental_rapids_bench: cfg!(feature = "experimental-rapids-bench"),
        },
        dataset,
        config: RunConfigReport {
            backend: config.backend.as_str().to_string(),
            warmup_runs: config.warmup_runs,
            measured_runs: config.measured_runs,
            timing_unit: "microseconds",
            primary_gate: "full-production-path-runtime",
        },
        hardware: HardwareReport {
            nvidia_smi: query_nvidia_smi(),
        },
        engines,
        summary,
    })
}

fn run_current_engine(fixture: &BenchFixture, config: &BenchConfig) -> Result<EngineReport> {
    let opts = match_options(config.backend);
    let _gpu_force_guard = GpuForceGuard::new(config.backend);
    let cold_start = Instant::now();
    let cold_output = run_current_once(fixture, opts);
    let cold_total_us = cold_start.elapsed().as_micros();
    let current_count = cold_output.len();

    for _ in 0..config.warmup_runs {
        let _ = run_current_once(fixture, opts);
    }

    let mut samples = Vec::with_capacity(config.measured_runs);
    let mut final_output = cold_output.clone();
    for _ in 0..config.measured_runs {
        let start = Instant::now();
        final_output = run_current_once(fixture, opts);
        samples.push(start.elapsed().as_micros());
    }

    let distribution = Distribution::from_samples(samples)?;
    let p50_secs = (distribution.p50.max(1) as f64) / 1_000_000.0;
    let output = OutputSummary {
        matches_emitted: final_output.len(),
        output_hash_sha256: output_hash(&final_output),
        canonical_output_hash_sha256: canonical_output_hash(&final_output),
    };
    let parity = match config.backend {
        BenchBackend::Cpu => current_vs_current_parity(&cold_output, &final_output),
        BenchBackend::Gpu => {
            let oracle = run_current_once(fixture, match_options(BenchBackend::Cpu));
            compare_to_cpu_oracle(&oracle, &final_output)
        }
    };

    Ok(EngineReport {
        engine_id: "current".to_string(),
        availability: EngineAvailability::Available,
        timings: Some(TimingSummary {
            cold_total_us,
            measured_total_us: distribution,
        }),
        throughput: Some(ThroughputSummary {
            matches_per_sec_p50: current_count as f64 / p50_secs,
            left_rows_per_sec_p50: fixture.left.len() as f64 / p50_secs,
            right_rows_per_sec_p50: fixture.right.len() as f64 / p50_secs,
        }),
        output: Some(output),
        gpu_stats: last_gpu_fuzzy_stats().map(|stats| GpuStatsReport {
            input_rows_left: stats.input_rows_left,
            input_rows_right: stats.input_rows_right,
            candidate_pairs_seen: stats.candidate_pairs_seen,
            pairs_uploaded: stats.pairs_uploaded,
            matches_emitted: stats.matches_emitted,
            normalization_time_us: stats.normalization_time_us,
            candidate_blocking_time_us: stats.candidate_blocking_time_us,
            cache_build_time_us: stats.cache_build_time_us,
            candidate_scan_time_us: stats.candidate_scan_time_us,
            h2d_time_us: stats.h2d_time_us,
            kernel_time_us: stats.kernel_time_us,
            d2h_time_us: stats.d2h_time_us,
            cpu_classification_time_us: stats.cpu_classification_time_us,
            total_wall_time_us: stats.total_wall_time_us,
            gpu_tile_count: stats.gpu_tile_count,
            gpu_mem_query_count: stats.gpu_mem_query_count,
            tile_size_min: stats.tile_size_min,
            tile_size_max: stats.tile_size_max,
            candidates_per_source_p50: stats.candidates_per_source_p50,
            candidates_per_source_p95: stats.candidates_per_source_p95,
            candidates_per_source_max: stats.candidates_per_source_max,
            resident_tables_enabled: stats.resident_tables_enabled,
            resident_table_bytes: stats.resident_table_bytes,
            resident_table_upload_us: stats.resident_table_upload_us,
            batch_index_h2d_us: stats.batch_index_h2d_us,
        }),
        parity: Some(parity),
    })
}

fn current_vs_current_parity(
    cold_output: &[crate::matching::MatchPair],
    final_output: &[crate::matching::MatchPair],
) -> ParityReport {
    let ordered_hashes_match = output_hash(final_output) == output_hash(cold_output);
    let canonical_hashes_match =
        canonical_output_hash(final_output) == canonical_output_hash(cold_output);
    ParityReport {
        oracle_engine_id: "current".to_string(),
        false_negatives_vs_current: 0,
        false_positives_vs_current: 0,
        false_negative_pair_ids: Vec::new(),
        false_positive_pair_ids: Vec::new(),
        pair_ids_match_current: true,
        canonical_output_hash_matches_current: canonical_hashes_match,
        output_hash_matches_current: ordered_hashes_match,
        blocking_failure: !canonical_hashes_match,
        notes: vec![
            "Current-vs-Current nondeterminism control".to_string(),
            "Ordered hash mismatch with canonical hash match means ordering drift only".to_string(),
        ],
    }
}

fn compare_to_cpu_oracle(
    cpu_output: &[crate::matching::MatchPair],
    candidate_output: &[crate::matching::MatchPair],
) -> ParityReport {
    let cpu_keys = pair_key_set(cpu_output);
    let candidate_keys = pair_key_set(candidate_output);
    let false_negative_keys = sorted_pair_keys(cpu_keys.difference(&candidate_keys));
    let false_positive_keys = sorted_pair_keys(candidate_keys.difference(&cpu_keys));
    let false_negatives = false_negative_keys.len() as u64;
    let false_positives = false_positive_keys.len() as u64;
    let false_negative_pair_ids = pair_key_samples(false_negative_keys);
    let false_positive_pair_ids = pair_key_samples(false_positive_keys);
    let ordered_hashes_match = output_hash(cpu_output) == output_hash(candidate_output);
    let canonical_hashes_match =
        canonical_output_hash(cpu_output) == canonical_output_hash(candidate_output);
    ParityReport {
        oracle_engine_id: "current-cpu".to_string(),
        false_negatives_vs_current: false_negatives,
        false_positives_vs_current: false_positives,
        false_negative_pair_ids,
        false_positive_pair_ids,
        pair_ids_match_current: false_negatives == 0 && false_positives == 0,
        canonical_output_hash_matches_current: canonical_hashes_match,
        output_hash_matches_current: ordered_hashes_match,
        blocking_failure: false_negatives > 0 || false_positives > 0 || !canonical_hashes_match,
        notes: vec![
            "Forced GPU Current output compared with CPU Current oracle".to_string(),
            "False negative means the candidate output missed a CPU Current pair".to_string(),
            "Ordered hash mismatch with canonical hash match means ordering drift only".to_string(),
        ],
    }
}

fn pair_key_set(rows: &[crate::matching::MatchPair]) -> HashSet<(i64, i64)> {
    rows.iter()
        .map(|pair| (pair.person1.id, pair.person2.id))
        .collect()
}

fn sorted_pair_keys<'a>(keys: impl Iterator<Item = &'a (i64, i64)>) -> Vec<(i64, i64)> {
    let mut keys: Vec<(i64, i64)> = keys.copied().collect();
    keys.sort_unstable();
    keys
}

fn pair_key_samples(keys: Vec<(i64, i64)>) -> Vec<PairIdSample> {
    keys.into_iter()
        .take(20)
        .map(|(left_id, right_id)| PairIdSample { left_id, right_id })
        .collect()
}

struct GpuForceGuard {
    enabled: bool,
}

impl GpuForceGuard {
    fn new(backend: BenchBackend) -> Self {
        let enabled = matches!(backend, BenchBackend::Gpu) && cfg!(feature = "gpu");
        if enabled {
            set_gpu_fuzzy_disable(false);
            set_gpu_fuzzy_metrics(true);
            set_gpu_fuzzy_force(true);
        }
        Self { enabled }
    }
}

impl Drop for GpuForceGuard {
    fn drop(&mut self) {
        if self.enabled {
            set_gpu_fuzzy_force(false);
            set_gpu_fuzzy_metrics(false);
        }
    }
}

fn run_current_once(fixture: &BenchFixture, opts: MatchOptions) -> Vec<crate::matching::MatchPair> {
    match_all_with_opts(
        &fixture.left,
        &fixture.right,
        MatchingAlgorithm::Fuzzy,
        opts,
        |_| {},
    )
}

fn match_options(backend: BenchBackend) -> MatchOptions {
    MatchOptions {
        backend: match backend {
            BenchBackend::Cpu => ComputeBackend::Cpu,
            BenchBackend::Gpu => ComputeBackend::Gpu,
        },
        gpu: match backend {
            BenchBackend::Cpu => None,
            BenchBackend::Gpu => Some(GpuConfig {
                device_id: Some(0),
                mem_budget_mb: 0,
            }),
        },
        progress: ProgressConfig::default(),
        allow_birthdate_swap: false,
    }
}

#[cfg(feature = "experimental-stringzilla-cuda")]
fn stringzilla_engine_report(fixture: &BenchFixture, config: &BenchConfig) -> EngineReport {
    match run_stringzilla_shadow_engine(fixture, config) {
        Ok(report) => report,
        Err(err) => EngineReport {
            engine_id: "stringzilla-shadow".to_string(),
            availability: EngineAvailability::Error {
                reason: err.to_string(),
            },
            timings: None,
            throughput: None,
            output: None,
            gpu_stats: None,
            parity: None,
        },
    }
}

#[cfg(not(feature = "experimental-stringzilla-cuda"))]
fn stringzilla_engine_report(_fixture: &BenchFixture, _config: &BenchConfig) -> EngineReport {
    unavailable_engine(
        "stringzilla-shadow",
        "feature experimental-stringzilla-cuda is not enabled",
    )
}

#[cfg(feature = "experimental-stringzilla-cuda")]
fn run_stringzilla_shadow_engine(
    fixture: &BenchFixture,
    config: &BenchConfig,
) -> Result<EngineReport> {
    use stringzilla::szs::{DeviceScope, LevenshteinDistances};

    let (left_names, right_names, pair_ids) = build_shadow_name_pairs(fixture);
    let expected: Vec<usize> = left_names
        .iter()
        .zip(right_names.iter())
        .map(|(left, right)| levenshtein_bytes(left.as_bytes(), right.as_bytes()))
        .collect();

    let device = DeviceScope::gpu_device(0)
        .map_err(|err| anyhow!("StringZilla CUDA device init failed: {err}"))?;
    let engine = LevenshteinDistances::new(&device, 0, 1, 1, 1)
        .map_err(|err| anyhow!("StringZilla CUDA Levenshtein init failed: {err}"))?;

    let cold_start = Instant::now();
    let cold_distances = compute_stringzilla_distances(&engine, &device, &left_names, &right_names)?;
    let cold_total_us = cold_start.elapsed().as_micros();

    for _ in 0..config.warmup_runs {
        let _ = compute_stringzilla_distances(&engine, &device, &left_names, &right_names)?;
    }

    let mut samples = Vec::with_capacity(config.measured_runs);
    let mut final_distances = cold_distances;
    for _ in 0..config.measured_runs {
        let start = Instant::now();
        final_distances = compute_stringzilla_distances(&engine, &device, &left_names, &right_names)?;
        samples.push(start.elapsed().as_micros());
    }

    let distribution = Distribution::from_samples(samples)?;
    let p50_secs = (distribution.p50.max(1) as f64) / 1_000_000.0;
    let mismatches = distance_mismatches(&expected, &final_distances, &pair_ids);
    let distances_match = mismatches.is_empty();
    let hash = usize_hash(&final_distances);

    Ok(EngineReport {
        engine_id: "stringzilla-shadow".to_string(),
        availability: EngineAvailability::Available,
        timings: Some(TimingSummary {
            cold_total_us,
            measured_total_us: distribution,
        }),
        throughput: Some(ThroughputSummary {
            matches_per_sec_p50: final_distances.len() as f64 / p50_secs,
            left_rows_per_sec_p50: fixture.left.len() as f64 / p50_secs,
            right_rows_per_sec_p50: fixture.right.len() as f64 / p50_secs,
        }),
        output: Some(OutputSummary {
            matches_emitted: final_distances.len(),
            output_hash_sha256: hash.clone(),
            canonical_output_hash_sha256: hash,
        }),
        gpu_stats: None,
        parity: Some(ParityReport {
            oracle_engine_id: "rust-levenshtein-distance-oracle".to_string(),
            false_negatives_vs_current: mismatches.len() as u64,
            false_positives_vs_current: 0,
            false_negative_pair_ids: mismatches,
            false_positive_pair_ids: Vec::new(),
            pair_ids_match_current: distances_match,
            canonical_output_hash_matches_current: distances_match,
            output_hash_matches_current: distances_match,
            blocking_failure: !distances_match,
            notes: vec![
                "Distance-only shadow engine; does not change candidate generation or production matching".to_string(),
                "matches_emitted is the number of scored name pairs, not application matches".to_string(),
                "Parity compares StringZilla CUDA byte-level Levenshtein distances against a Rust CPU oracle".to_string(),
            ],
        }),
    })
}

#[cfg(feature = "experimental-stringzilla-cuda")]
fn compute_stringzilla_distances(
    engine: &stringzilla::szs::LevenshteinDistances,
    device: &stringzilla::szs::DeviceScope,
    left_names: &[String],
    right_names: &[String],
) -> Result<Vec<usize>> {
    let mut output = Vec::with_capacity(left_names.len());
    for (left_chunk, right_chunk) in left_names
        .chunks(STRINGZILLA_SHADOW_CHUNK_SIZE)
        .zip(right_names.chunks(STRINGZILLA_SHADOW_CHUNK_SIZE))
    {
        let distances = engine
            .compute(device, left_chunk, right_chunk)
            .map_err(|err| anyhow!("StringZilla CUDA Levenshtein compute failed: {err}"))?;
        output.extend(distances.iter().copied());
    }
    Ok(output)
}

#[cfg(feature = "experimental-stringzilla-cuda")]
fn build_shadow_name_pairs(fixture: &BenchFixture) -> (Vec<String>, Vec<String>, Vec<(i64, i64)>) {
    let pair_count = fixture.left.len().saturating_mul(fixture.right.len());
    let mut left_names = Vec::with_capacity(pair_count);
    let mut right_names = Vec::with_capacity(pair_count);
    let mut pair_ids = Vec::with_capacity(pair_count);
    for left in &fixture.left {
        let left_name = shadow_name(left);
        for right in &fixture.right {
            left_names.push(left_name.clone());
            right_names.push(shadow_name(right));
            pair_ids.push((left.id, right.id));
        }
    }
    (left_names, right_names, pair_ids)
}

#[cfg(feature = "experimental-stringzilla-cuda")]
fn shadow_name(person: &Person) -> String {
    normalize_shadow_name(&format!(
        "{} {} {}",
        person.first_name.as_deref().unwrap_or(""),
        person.middle_name.as_deref().unwrap_or(""),
        person.last_name.as_deref().unwrap_or("")
    ))
}

#[cfg(feature = "experimental-stringzilla-cuda")]
fn normalize_shadow_name(value: &str) -> String {
    value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

#[cfg(feature = "experimental-stringzilla-cuda")]
fn distance_mismatches(
    expected: &[usize],
    actual: &[usize],
    pair_ids: &[(i64, i64)],
) -> Vec<PairIdSample> {
    expected
        .iter()
        .zip(actual.iter())
        .zip(pair_ids.iter())
        .filter_map(|((expected, actual), (left_id, right_id))| {
            (expected != actual).then_some(PairIdSample {
                left_id: *left_id,
                right_id: *right_id,
            })
        })
        .take(20)
        .collect()
}

#[cfg(feature = "experimental-rapids-bench")]
fn rapids_engine_report() -> EngineReport {
    unavailable_engine(
        "rapids-shadow",
        "RAPIDS feature is enabled, but RAPIDS is process-bound/not-decisionable in this slice",
    )
}

#[cfg(not(feature = "experimental-rapids-bench"))]
fn rapids_engine_report() -> EngineReport {
    unavailable_engine(
        "rapids-shadow",
        "feature experimental-rapids-bench is not enabled",
    )
}

fn unavailable_engine(engine_id: &str, reason: &str) -> EngineReport {
    EngineReport {
        engine_id: engine_id.to_string(),
        availability: EngineAvailability::Unavailable {
            reason: reason.to_string(),
        },
        timings: None,
        throughput: None,
        output: None,
        gpu_stats: None,
        parity: None,
    }
}

fn summarize_decision(engines: &[EngineReport]) -> DecisionSummary {
    if let Some(blocking) = engines.iter().find(|engine| {
        engine
            .parity
            .as_ref()
            .is_some_and(|parity| parity.blocking_failure)
    }) {
        return DecisionSummary {
            recommendation: "block-integration".to_string(),
            reason: format!(
                "Parity failure in engine {}; do not optimize or integrate external engines until Current parity is understood.",
                blocking.engine_id
            ),
        };
    }
    let external_available = engines.iter().any(|engine| {
        engine.engine_id != "current"
            && matches!(engine.availability, EngineAvailability::Available)
    });
    if external_available {
        DecisionSummary {
            recommendation: "continue-analysis".to_string(),
            reason: "At least one shadow engine reported availability; compare parity and full-path speedup before integration.".to_string(),
        }
    } else {
        DecisionSummary {
            recommendation: "baseline-only".to_string(),
            reason: "Current engine baseline completed; external engines are unavailable or not implemented in this slice.".to_string(),
        }
    }
}

impl Distribution {
    fn from_samples(mut samples: Vec<u128>) -> Result<Self> {
        if samples.is_empty() {
            return Err(anyhow!("cannot summarize empty samples"));
        }
        samples.sort_unstable();
        let sum: u128 = samples.iter().sum();
        let mean = sum as f64 / samples.len() as f64;
        let variance = samples
            .iter()
            .map(|sample| {
                let delta = *sample as f64 - mean;
                delta * delta
            })
            .sum::<f64>()
            / samples.len() as f64;
        Ok(Self {
            min: samples[0],
            p50: percentile(&samples, 50),
            p95: percentile(&samples, 95),
            p99: percentile(&samples, 99),
            max: *samples.last().unwrap_or(&samples[0]),
            mean,
            stddev: variance.sqrt(),
            samples,
        })
    }
}

fn percentile(samples: &[u128], pct: usize) -> u128 {
    let idx = ((samples.len().saturating_sub(1) * pct) + 99) / 100;
    samples[idx.min(samples.len().saturating_sub(1))]
}

struct BenchFixture {
    left: Vec<Person>,
    right: Vec<Person>,
}

fn build_dataset(kind: BenchDatasetKind) -> BenchFixture {
    match kind {
        BenchDatasetKind::Small => generated_dataset(24, 28, 3, false),
        BenchDatasetKind::Medium => generated_dataset(500, 560, 7, false),
        BenchDatasetKind::Large => generated_dataset(2_000, 2_200, 11, false),
        BenchDatasetKind::DuplicateHeavy => generated_dataset(800, 900, 5, true),
        BenchDatasetKind::Sparse => generated_sparse_dataset(),
        BenchDatasetKind::Messy => generated_messy_dataset(),
    }
}

fn generated_dataset(
    left_count: usize,
    right_count: usize,
    skew: usize,
    duplicates: bool,
) -> BenchFixture {
    let first_names = [
        "Ana", "Maria", "Jose", "Juan", "Rosa", "Luis", "Elena", "Pedro", "Mila", "Nora", "Carlo",
        "Sofia",
    ];
    let last_names = [
        "Santos",
        "Reyes",
        "Cruz",
        "Garcia",
        "Dela Cruz",
        "Ramos",
        "Mendoza",
        "Torres",
        "Bautista",
        "Navarro",
        "Aquino",
        "Castillo",
    ];
    let mut left = Vec::with_capacity(left_count);
    let mut right = Vec::with_capacity(right_count);
    for i in 0..left_count {
        let base = i % first_names.len();
        left.push(person(
            i as i64 + 1,
            first_names[base],
            Some(if i % 3 == 0 { "Mae" } else { "Luis" }),
            last_names[(i / 2) % last_names.len()],
            1980 + (i % 32) as i32,
            1 + (i % 12) as u32,
            1 + (i % 27) as u32,
        ));
    }
    for j in 0..right_count {
        let source_idx = if duplicates { j / 2 } else { j };
        let base = (source_idx + skew) % first_names.len();
        let mut first = first_names[base].to_string();
        if j % 17 == 0 {
            first.push('h');
        }
        right.push(person(
            j as i64 + 10_001,
            &first,
            Some(if j % 3 == 0 { "Mae" } else { "Luis" }),
            last_names[((source_idx + skew) / 2) % last_names.len()],
            1980 + ((source_idx + skew) % 32) as i32,
            1 + ((source_idx + skew) % 12) as u32,
            1 + ((source_idx + skew) % 27) as u32,
        ));
    }
    BenchFixture { left, right }
}

fn generated_sparse_dataset() -> BenchFixture {
    let mut fixture = generated_dataset(400, 420, 19, false);
    for (idx, person) in fixture.right.iter_mut().enumerate() {
        if idx % 3 != 0 {
            person.birthdate = NaiveDate::from_ymd_opt(1960 + (idx % 30) as i32, 12, 28);
            person.last_name = Some(format!("SparseLast{idx}"));
        }
    }
    fixture
}

fn generated_messy_dataset() -> BenchFixture {
    let left = vec![
        person(1, "Maria", Some("Cristina"), "Santos", 1990, 1, 2),
        person(2, "Kristina", None, "De la Cruz", 1988, 5, 9),
        person(3, "Jose", Some("L"), "Reyes", 1975, 7, 14),
        person(4, "Ana  ", Some("Mae"), "Garcia", 1992, 3, 4),
        person(5, "Sofia", Some("Isabel"), "Nunez", 1981, 11, 20),
        person(6, "Carlo", Some(""), "Ocampo", 1999, 9, 17),
    ];
    let right = vec![
        person(101, "Cristina", Some("Maria"), "Santos", 1990, 1, 2),
        person(102, "Kristine", None, "Dela Cruz", 1988, 5, 9),
        person(103, "Joseph", Some("L"), "Reyes", 1975, 7, 14),
        person(104, "Ana", Some("Mae"), "Garcia", 1992, 3, 4),
        person(105, "Sofia", Some("Isabel"), "Nunez", 1981, 11, 20),
        person(106, "", None, "Ocampo", 1999, 9, 17),
    ];
    BenchFixture { left, right }
}

fn person(
    id: i64,
    first: &str,
    middle: Option<&str>,
    last: &str,
    year: i32,
    month: u32,
    day: u32,
) -> Person {
    Person {
        id,
        uuid: Some(format!("uuid-{id}")),
        first_name: Some(first.to_string()),
        middle_name: middle.map(|value| value.to_string()),
        last_name: Some(last.to_string()),
        birthdate: NaiveDate::from_ymd_opt(year, month, day),
        hh_id: Some(format!("{}", id / 10)),
        extra_fields: Arc::new(HashMap::new()),
    }
}

fn dataset_manifest(kind: BenchDatasetKind, fixture: &BenchFixture) -> DatasetManifest {
    DatasetManifest {
        id: kind.as_str().to_string(),
        hash_sha256: dataset_hash(fixture),
        seed: 42,
        left_rows: fixture.left.len(),
        right_rows: fixture.right.len(),
        expected_current_matches: None,
        notes: vec![
            "Generated deterministic in-memory fixture; no external data required".to_string(),
            "Candidate count is observed from Current engine GPU stats when GPU backend is used"
                .to_string(),
        ],
        left_string_lengths: length_stats(&fixture.left),
        right_string_lengths: length_stats(&fixture.right),
    }
}

fn length_stats(rows: &[Person]) -> LengthStats {
    let mut lengths: Vec<usize> = rows
        .iter()
        .map(|p| {
            p.first_name.as_deref().unwrap_or("").len()
                + p.middle_name.as_deref().unwrap_or("").len()
                + p.last_name.as_deref().unwrap_or("").len()
        })
        .collect();
    if lengths.is_empty() {
        return LengthStats::default();
    }
    lengths.sort_unstable();
    LengthStats {
        min: lengths[0],
        p50: lengths[((lengths.len() - 1) * 50) / 100],
        p95: lengths[((lengths.len() - 1) * 95) / 100],
        max: *lengths.last().unwrap_or(&lengths[0]),
    }
}

fn dataset_hash(fixture: &BenchFixture) -> String {
    let mut hasher = Sha256::new();
    for p in fixture.left.iter().chain(fixture.right.iter()) {
        hash_person(&mut hasher, p);
    }
    hex_digest(hasher.finalize().as_slice())
}

fn output_hash(rows: &[crate::matching::MatchPair]) -> String {
    let mut hasher = Sha256::new();
    for pair in rows {
        hash_match_pair(&mut hasher, pair);
    }
    hex_digest(hasher.finalize().as_slice())
}

fn canonical_output_hash(rows: &[crate::matching::MatchPair]) -> String {
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by_key(|idx| (rows[*idx].person1.id, rows[*idx].person2.id));
    let mut hasher = Sha256::new();
    for idx in order {
        hash_match_pair(&mut hasher, &rows[idx]);
    }
    hex_digest(hasher.finalize().as_slice())
}

#[cfg(feature = "experimental-stringzilla-cuda")]
fn usize_hash(values: &[usize]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    hex_digest(hasher.finalize().as_slice())
}

#[cfg(feature = "experimental-stringzilla-cuda")]
fn levenshtein_bytes(left: &[u8], right: &[u8]) -> usize {
    let mut previous: Vec<usize> = (0..=right.len()).collect();
    let mut current = vec![0; right.len() + 1];
    for (i, &left_byte) in left.iter().enumerate() {
        current[0] = i + 1;
        for (j, &right_byte) in right.iter().enumerate() {
            let substitution = previous[j] + usize::from(left_byte != right_byte);
            let deletion = previous[j + 1] + 1;
            let insertion = current[j] + 1;
            current[j + 1] = substitution.min(deletion).min(insertion);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[right.len()]
}

fn hash_match_pair(hasher: &mut Sha256, pair: &crate::matching::MatchPair) {
    hash_person(hasher, &pair.person1);
    hash_person(hasher, &pair.person2);
    hasher.update(format!("{:.6}", pair.confidence).as_bytes());
    for field in &pair.matched_fields {
        hasher.update(field.as_bytes());
        hasher.update([0]);
    }
}

fn hash_person(hasher: &mut Sha256, p: &Person) {
    hasher.update(p.id.to_le_bytes());
    for value in [
        p.uuid.as_deref(),
        p.first_name.as_deref(),
        p.middle_name.as_deref(),
        p.last_name.as_deref(),
        p.hh_id.as_deref(),
    ] {
        hasher.update(value.unwrap_or("").as_bytes());
        hasher.update([0]);
    }
    if let Some(date) = p.birthdate {
        hasher.update(date.to_string().as_bytes());
    }
    hasher.update([0xff]);
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn query_nvidia_smi() -> TelemetryAvailability {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output();
    match output {
        Ok(out) if out.status.success() => TelemetryAvailability {
            available: true,
            reason: None,
            raw_summary: Some(String::from_utf8_lossy(&out.stdout).trim().to_string()),
        },
        Ok(out) => TelemetryAvailability {
            available: false,
            reason: Some(format!("nvidia-smi exited with {}", out.status)),
            raw_summary: Some(String::from_utf8_lossy(&out.stderr).trim().to_string()),
        },
        Err(e) => TelemetryAvailability {
            available: false,
            reason: Some(e.to_string()),
            raw_summary: None,
        },
    }
}

fn git_sha() -> Option<String> {
    let out = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()?;
    if out.status.success() {
        Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
    } else {
        None
    }
}

fn unix_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_hash_is_stable() {
        let first = build_dataset(BenchDatasetKind::Small);
        let second = build_dataset(BenchDatasetKind::Small);
        assert_eq!(dataset_hash(&first), dataset_hash(&second));
    }

    #[test]
    fn benchmark_report_serializes_with_unavailable_engines() {
        let report = run_benchmark(BenchConfig {
            measured_runs: 1,
            warmup_runs: 0,
            ..BenchConfig::default()
        })
        .expect("benchmark report");
        let json = serde_json::to_string(&report).expect("serialize report");
        assert!(json.contains("\"schema_version\":1"));
        assert!(json.contains("\"engine_id\":\"current\""));
        assert!(json.contains("\"stringzilla-shadow\""));
        assert!(json.contains("\"rapids-shadow\""));
    }
}
