//! Cascading Advanced Matching Workflow
//!
//! This module implements a sequential cascade of matching levels (L1-L11).
//!
//! **Level 12 Exclusion:** L12 (Household Matching) is NOT included in the cascade.
//! Household matching has different semantics (aggregates by household) and should
//! be run separately via the dedicated household matching commands.
//!
//! **Geographic Levels (L4-L9):** These levels require specific columns:
//! - L4-L6: Require `barangay_code` column
//! - L7-L9: Require `city_code` column
//!
//! If these columns are missing, the cascade will skip those levels automatically
//! (default behavior) or prompt the user for handling options.

use crate::matching::advanced_matcher::{AdvColumns, AdvConfig, AdvLevel, advanced_match_inmemory};
use crate::matching::{
    ComputeBackend, GpuConfig, MatchOptions, MatchPair, ProgressConfig, ProgressUpdate,
};
use crate::models::Person;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

/// Result of running a single level in the cascade
#[derive(Debug, Clone)]
pub struct LevelResult {
    pub level: AdvLevel,
    pub level_number: u8,
    pub matches: Vec<MatchPair>,
    pub match_count: usize,
    pub skipped: bool,
    pub skip_reason: Option<String>,
}

/// Summary of the entire cascade run
#[derive(Debug, Clone)]
pub struct CascadeSummary {
    pub levels_run: Vec<u8>,
    pub levels_skipped: Vec<(u8, String)>, // (level_num, reason)
    pub total_matches: usize,
    pub matches_per_level: HashMap<u8, usize>,
    pub duration_ms: u64,
}

/// Mode for handling missing geographic columns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MissingColumnMode {
    /// Option A: Run all available levels, skip levels with missing columns (default)
    #[default]
    AutoSkip,
    /// Option B: Use manually specified levels only
    ManualSelect,
    /// Option C: Abort if any geographic columns are missing
    AbortOnMissing,
}

/// Mode for handling record exclusion across cascade levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CascadeExclusionMode {
    /// True cascade: records matched at L1 are excluded from L2, L1+L2 excluded from L3, etc.
    /// Each record pair appears in at most ONE level output (the first level that matched it).
    #[default]
    Exclusive,
    /// Independent levels: all levels run on the full dataset (allows duplicate matches).
    /// Same pair can appear in multiple level outputs.
    Independent,
}

/// Configuration for a cascade run
#[derive(Debug, Clone)]
pub struct CascadeConfig {
    /// Which levels to run (1-11). Empty = all available L1-L11.
    /// Note: L12 is never included in cascade per design.
    pub levels: Vec<u8>,
    /// Fuzzy matching threshold (0.0-1.0)
    pub threshold: f32,
    /// Allow month/day birthdate swaps
    pub allow_birthdate_swap: bool,
    /// How to handle missing geographic columns
    pub missing_column_mode: MissingColumnMode,
    /// Base output path (e.g., "output.csv" -> "output_L01.csv", etc.)
    pub base_output_path: String,
    /// How to handle already-matched records across levels
    pub exclusion_mode: CascadeExclusionMode,
    /// Compute backend for fuzzy matching levels (L10-L11).
    /// L1-L9 always use CPU (HashMap-based exact matching).
    /// GPU acceleration is only applied to L10-L11 fuzzy matching.
    pub compute_backend: ComputeBackend,
    /// GPU device ID for multi-GPU systems.
    /// None = use default device (typically device 0).
    /// Only used when compute_backend = Gpu.
    pub gpu_device_id: Option<usize>,
}

impl Default for CascadeConfig {
    fn default() -> Self {
        Self {
            levels: vec![],
            threshold: 0.95,
            allow_birthdate_swap: false,
            missing_column_mode: MissingColumnMode::AutoSkip,
            base_output_path: "cascade_output.csv".to_string(),
            exclusion_mode: CascadeExclusionMode::Exclusive,
            compute_backend: ComputeBackend::Cpu,
            gpu_device_id: None,
        }
    }
}

/// Geographic column availability in the database schema
#[derive(Debug, Clone, Default)]
pub struct GeoColumnStatus {
    pub has_barangay_code: bool,
    pub has_city_code: bool,
}

impl GeoColumnStatus {
    /// Check which levels can run given the available columns
    pub fn can_run_level(&self, level: u8) -> Result<(), String> {
        match level {
            1 | 2 | 3 | 10 | 11 => Ok(()), // Non-geographic levels always available
            4 | 5 | 6 => {
                if self.has_barangay_code {
                    Ok(())
                } else {
                    Err("Missing required column: barangay_code".to_string())
                }
            }
            7 | 8 | 9 => {
                if self.has_city_code {
                    Ok(())
                } else {
                    Err("Missing required column: city_code".to_string())
                }
            }
            12 => Err("L12 (Household Matching) is excluded from cascade runs".to_string()),
            _ => Err(format!("Invalid level: {}", level)),
        }
    }

    /// Get list of unavailable levels due to missing columns
    pub fn get_unavailable_levels(&self) -> Vec<(u8, String)> {
        let mut unavailable = Vec::new();
        for level in 4..=9 {
            if let Err(reason) = self.can_run_level(level) {
                unavailable.push((level, reason));
            }
        }
        unavailable
    }

    /// Get a human-readable summary of column status
    pub fn summary(&self) -> String {
        format!(
            "Geographic columns: barangay_code={}, city_code={}",
            if self.has_barangay_code {
                "present"
            } else {
                "MISSING"
            },
            if self.has_city_code {
                "present"
            } else {
                "MISSING"
            }
        )
    }
}

/// Convert level number (1-12) to AdvLevel enum
pub fn level_num_to_adv_level(num: u8) -> Option<AdvLevel> {
    match num {
        1 => Some(AdvLevel::L1BirthdateFullMiddle),
        2 => Some(AdvLevel::L2BirthdateMiddleInitial),
        3 => Some(AdvLevel::L3BirthdateNoMiddle),
        4 => Some(AdvLevel::L4BarangayFullMiddle),
        5 => Some(AdvLevel::L5BarangayMiddleInitial),
        6 => Some(AdvLevel::L6BarangayNoMiddle),
        7 => Some(AdvLevel::L7CityFullMiddle),
        8 => Some(AdvLevel::L8CityMiddleInitial),
        9 => Some(AdvLevel::L9CityNoMiddle),
        10 => Some(AdvLevel::L10FuzzyBirthdateFullMiddle),
        11 => Some(AdvLevel::L11FuzzyBirthdateNoMiddle),
        12 => Some(AdvLevel::L12HouseholdMatching),
        _ => None,
    }
}

/// Convert AdvLevel enum to level number (1-12)
pub fn adv_level_to_num(level: AdvLevel) -> u8 {
    match level {
        AdvLevel::L1BirthdateFullMiddle => 1,
        AdvLevel::L2BirthdateMiddleInitial => 2,
        AdvLevel::L3BirthdateNoMiddle => 3,
        AdvLevel::L4BarangayFullMiddle => 4,
        AdvLevel::L5BarangayMiddleInitial => 5,
        AdvLevel::L6BarangayNoMiddle => 6,
        AdvLevel::L7CityFullMiddle => 7,
        AdvLevel::L8CityMiddleInitial => 8,
        AdvLevel::L9CityNoMiddle => 9,
        AdvLevel::L10FuzzyBirthdateFullMiddle => 10,
        AdvLevel::L11FuzzyBirthdateNoMiddle => 11,
        AdvLevel::L12HouseholdMatching => 12,
    }
}

/// Get human-readable description for a level
pub fn level_description(level: u8) -> &'static str {
    match level {
        1 => "L1: Exact Match (Full Middle + Birthdate)",
        2 => "L2: Exact Match (Middle Initial + Birthdate)",
        3 => "L3: Exact Match (First+Last + Birthdate)",
        4 => "L4: Exact Match (Full Middle + Barangay Code)",
        5 => "L5: Exact Match (Middle Initial + Barangay Code)",
        6 => "L6: Exact Match (First+Last + Barangay Code)",
        7 => "L7: Exact Match (Full Middle + City Code)",
        8 => "L8: Exact Match (Middle Initial + City Code)",
        9 => "L9: Exact Match (First+Last + City Code)",
        10 => "L10: Fuzzy Match (Full Middle + Birthdate)",
        11 => "L11: Fuzzy Match (No Middle + Birthdate)",
        12 => "L12: Household Matching (excluded from cascade)",
        _ => "Unknown Level",
    }
}

/// All cascade levels (L1-L11, explicitly excludes L12)
pub const CASCADE_LEVELS: [u8; 11] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

/// Sort match pairs by (person1.id, person2.id) for deterministic ordering.
///
/// This is critical for GPU/CPU parity: GPU kernels may return matches in
/// non-deterministic order due to parallel execution. Sorting ensures:
/// 1. Identical results across multiple runs with the same input
/// 2. Consistent exclusion behavior in cascade mode (same pairs excluded at same levels)
/// 3. Reproducible CSV output for verification and debugging
///
/// # Arguments
/// * `matches` - Mutable vector of match pairs to sort in-place
///
/// # Ordering
/// Primary sort: person1.id ascending
/// Secondary sort: person2.id ascending (for ties in person1.id)
pub fn sort_matches_by_id(matches: &mut Vec<MatchPair>) {
    matches.sort_by(|a, b| match a.person1.id.cmp(&b.person1.id) {
        std::cmp::Ordering::Equal => a.person2.id.cmp(&b.person2.id),
        other => other,
    });
}

/// Sort match pairs and return a new sorted vector (non-mutating variant).
///
/// See [`sort_matches_by_id`] for details on ordering and purpose.
pub fn sorted_matches_by_id(mut matches: Vec<MatchPair>) -> Vec<MatchPair> {
    sort_matches_by_id(&mut matches);
    matches
}

/// Run fuzzy matching for L10 or L11 with compute backend routing.
///
/// - L1-L9: Always CPU (exact matching, not handled here)
/// - L10-L11: Routes to GPU when `backend == ComputeBackend::Gpu` and `gpu` feature is enabled
/// - Graceful fallback to CPU if GPU fails or feature is disabled
///
/// Results are always sorted by `(person1.id, person2.id)` for deterministic ordering.
fn run_fuzzy_level(
    table1: &[Person],
    table2: &[Person],
    cfg: &AdvConfig,
    backend: ComputeBackend,
    gpu_device_id: Option<usize>,
) -> Vec<MatchPair> {
    let use_gpu = matches!(backend, ComputeBackend::Gpu)
        && matches!(
            cfg.level,
            AdvLevel::L10FuzzyBirthdateFullMiddle | AdvLevel::L11FuzzyBirthdateNoMiddle
        );

    if use_gpu {
        #[cfg(feature = "gpu")]
        {
            let opts = MatchOptions {
                backend: ComputeBackend::Gpu,
                gpu: Some(GpuConfig {
                    device_id: gpu_device_id,
                    mem_budget_mb: 512, // Conservative default
                }),
                progress: ProgressConfig::default(),
                allow_birthdate_swap: cfg.allow_birthdate_swap,
            };

            let on_progress = |_u: ProgressUpdate| {};
            let gpu_result = match cfg.level {
                AdvLevel::L10FuzzyBirthdateFullMiddle => {
                    crate::matching::gpu_config::with_oom_cpu_fallback(
                        || {
                            crate::matching::cascade_match_fuzzy_gpu(
                                table1,
                                table2,
                                opts,
                                on_progress,
                            )
                        },
                        || {
                            log::info!("L10 GPU fallback to CPU");
                            advanced_match_inmemory(table1, table2, cfg)
                        },
                        "Cascade L10 GPU fuzzy",
                    )
                }
                AdvLevel::L11FuzzyBirthdateNoMiddle => {
                    crate::matching::gpu_config::with_oom_cpu_fallback(
                        || {
                            crate::matching::cascade_match_fuzzy_no_mid_gpu(
                                table1,
                                table2,
                                opts,
                                on_progress,
                            )
                        },
                        || {
                            log::info!("L11 GPU fallback to CPU");
                            advanced_match_inmemory(table1, table2, cfg)
                        },
                        "Cascade L11 GPU fuzzy (no-mid)",
                    )
                }
                _ => Ok(advanced_match_inmemory(table1, table2, cfg)),
            };

            match gpu_result {
                Ok(mut matches) => {
                    // Apply threshold filtering (GPU may return all candidates)
                    matches.retain(|m| (m.confidence / 100.0) >= cfg.threshold);
                    sort_matches_by_id(&mut matches);
                    return matches;
                }
                Err(e) => {
                    log::warn!("GPU matching failed: {}. Falling back to CPU.", e);
                }
            }
        }

        #[cfg(not(feature = "gpu"))]
        {
            log::info!(
                "GPU backend requested but 'gpu' feature not enabled. Using CPU for {:?}.",
                cfg.level
            );
        }
    }

    // CPU path (default or fallback)
    let mut matches = advanced_match_inmemory(table1, table2, cfg);
    sort_matches_by_id(&mut matches);
    matches
}

/// Generate output file path for a specific level
/// Example: "output.csv" + level 1 -> "output_L01.csv"
pub fn level_output_path(base_path: &str, level: u8) -> String {
    let path = Path::new(base_path);
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("csv");
    let parent = path.parent().and_then(|p| p.to_str()).unwrap_or("");

    if parent.is_empty() {
        format!("{}_L{:02}.{}", stem, level, ext)
    } else {
        format!("{}/{}_L{:02}.{}", parent, stem, level, ext)
    }
}

/// Generate summary file path from base path
/// Example: "output.csv" -> "output_summary.txt"
pub fn summary_output_path(base_path: &str) -> String {
    let path = Path::new(base_path);
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let parent = path.parent().and_then(|p| p.to_str()).unwrap_or("");

    if parent.is_empty() {
        format!("{}_summary.txt", stem)
    } else {
        format!("{}/{}_summary.txt", parent, stem)
    }
}

/// Result entry for cascade summary
#[derive(Debug, Clone)]
pub struct CascadeLevelEntry {
    pub level: u8,
    pub description: String,
    pub status: CascadeLevelStatus,
    pub match_count: usize,
    pub output_path: Option<String>,
}

/// Status of a cascade level
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CascadeLevelStatus {
    Completed,
    Skipped(String), // reason
    Failed(String),  // error message
}

impl std::fmt::Display for CascadeLevelStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Completed => write!(f, "Completed"),
            Self::Skipped(reason) => write!(f, "Skipped: {}", reason),
            Self::Failed(err) => write!(f, "Failed: {}", err),
        }
    }
}

/// Full cascade result including all level entries and timing
#[derive(Debug, Clone)]
pub struct CascadeResult {
    pub entries: Vec<CascadeLevelEntry>,
    pub total_matches: usize,
    pub total_duration_ms: u64,
}

impl CascadeResult {
    /// Write summary to a text file
    pub fn write_summary(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;

        writeln!(f, "Cascade Matching Summary")?;
        writeln!(f, "========================")?;
        writeln!(
            f,
            "Note: L12 (Household Matching) is excluded from cascade runs.\n"
        )?;
        writeln!(
            f,
            "{:<6} {:<50} {:<20} {:>12} {}",
            "Level", "Description", "Status", "Matches", "Output File"
        )?;
        writeln!(f, "{}", "-".repeat(110))?;

        for entry in &self.entries {
            let output = entry.output_path.as_deref().unwrap_or("-");
            let status_str = match &entry.status {
                CascadeLevelStatus::Completed => "Completed".to_string(),
                CascadeLevelStatus::Skipped(r) => format!("Skipped: {}", r),
                CascadeLevelStatus::Failed(e) => format!("Failed: {}", e),
            };
            writeln!(
                f,
                "L{:<5} {:<50} {:<20} {:>12} {}",
                entry.level, entry.description, status_str, entry.match_count, output
            )?;
        }

        writeln!(f, "{}", "-".repeat(110))?;
        writeln!(f, "Total Matches: {}", self.total_matches)?;
        writeln!(
            f,
            "Total Duration: {:.2}s",
            self.total_duration_ms as f64 / 1000.0
        )?;

        Ok(())
    }
}

/// Determine which levels to run based on config and geo column status
pub fn determine_runnable_levels(
    cfg: &CascadeConfig,
    geo_status: &GeoColumnStatus,
) -> Result<Vec<u8>, String> {
    let requested_levels = if cfg.levels.is_empty() {
        CASCADE_LEVELS.to_vec()
    } else {
        // Filter out L12 if user accidentally included it
        cfg.levels
            .iter()
            .copied()
            .filter(|&l| l != 12 && l >= 1 && l <= 11)
            .collect()
    };

    match cfg.missing_column_mode {
        MissingColumnMode::AutoSkip => {
            // Return all requested levels; unavailable ones will be skipped during execution
            Ok(requested_levels)
        }
        MissingColumnMode::ManualSelect => {
            // Only include levels that can actually run
            let runnable: Vec<u8> = requested_levels
                .into_iter()
                .filter(|&l| geo_status.can_run_level(l).is_ok())
                .collect();
            Ok(runnable)
        }
        MissingColumnMode::AbortOnMissing => {
            // Check if any requested geographic levels are unavailable
            for level in &requested_levels {
                if let Err(reason) = geo_status.can_run_level(*level) {
                    return Err(format!(
                        "Cannot run cascade: Level {} requires missing column. {}",
                        level, reason
                    ));
                }
            }
            Ok(requested_levels)
        }
    }
}

/// Progress callback for cascade execution
pub type CascadeProgressFn = Box<dyn Fn(CascadeProgress) + Send + Sync>;

/// Progress update during cascade execution
#[derive(Debug, Clone)]
pub struct CascadeProgress {
    pub current_level: u8,
    pub total_levels: usize,
    pub level_description: String,
    pub phase: CascadePhase,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CascadePhase {
    Starting,
    Running,
    WritingOutput,
    Completed,
    Skipped(String),
}

/// Run cascade matching in-memory for all configured levels
///
/// ## Exclusion Mode Behavior
/// - `Exclusive` (default): Records matched at earlier levels are excluded from later levels.
///   Each record pair appears in at most ONE level output.
/// - `Independent`: All levels run on the full dataset (legacy behavior). Same pair can appear
///   in multiple level outputs.
pub fn run_cascade_inmemory<F>(
    table1: &[Person],
    table2: &[Person],
    cfg: &CascadeConfig,
    geo_status: &GeoColumnStatus,
    on_progress: F,
) -> CascadeResult
where
    F: Fn(CascadeProgress),
{
    let start = Instant::now();
    let mut entries = Vec::new();
    let mut total_matches = 0usize;

    // Collect all extra field names from Table 2 for CSV export
    // Use BTreeSet for consistent, sorted ordering across all output files
    let extra_field_names: Vec<String> = {
        let mut field_set = BTreeSet::new();
        for person in table2 {
            for key in person.extra_fields.keys() {
                field_set.insert(key.clone());
            }
        }
        field_set.into_iter().collect()
    };
    log::info!(
        "Collected {} extra fields from Table 2 for CSV export",
        extra_field_names.len()
    );

    // Track matched IDs for exclusive cascade mode
    // We track by (table1_id, table2_id) pair to ensure uniqueness
    let mut matched_pairs: HashSet<(i64, i64)> = HashSet::new();
    let use_exclusion = cfg.exclusion_mode == CascadeExclusionMode::Exclusive;

    log::info!("Starting cascade run for levels L1-L11 (L12 excluded)");
    log::info!("Exclusion mode: {:?}", cfg.exclusion_mode);
    log::info!("{}", geo_status.summary());

    // Determine which levels to run
    let levels_to_run = if cfg.levels.is_empty() {
        CASCADE_LEVELS.to_vec()
    } else {
        cfg.levels
            .iter()
            .copied()
            .filter(|&l| l >= 1 && l <= 11)
            .collect()
    };

    let total_levels = levels_to_run.len();

    for (idx, level_num) in levels_to_run.iter().enumerate() {
        let level_num = *level_num;
        let desc = level_description(level_num).to_string();

        on_progress(CascadeProgress {
            current_level: level_num,
            total_levels,
            level_description: desc.clone(),
            phase: CascadePhase::Starting,
        });

        // Check if level can run
        if let Err(reason) = geo_status.can_run_level(level_num) {
            if cfg.missing_column_mode == MissingColumnMode::AutoSkip {
                log::warn!("Skipping Level {}: {}", level_num, reason);
                on_progress(CascadeProgress {
                    current_level: level_num,
                    total_levels,
                    level_description: desc.clone(),
                    phase: CascadePhase::Skipped(reason.clone()),
                });
                entries.push(CascadeLevelEntry {
                    level: level_num,
                    description: desc,
                    status: CascadeLevelStatus::Skipped(reason),
                    match_count: 0,
                    output_path: None,
                });
                continue;
            }
        }

        log::info!("Running Level {}: {}", level_num, desc);
        on_progress(CascadeProgress {
            current_level: level_num,
            total_levels,
            level_description: desc.clone(),
            phase: CascadePhase::Running,
        });

        // Get AdvLevel enum
        let adv_level = match level_num_to_adv_level(level_num) {
            Some(l) => l,
            None => {
                entries.push(CascadeLevelEntry {
                    level: level_num,
                    description: desc,
                    status: CascadeLevelStatus::Failed("Invalid level number".to_string()),
                    match_count: 0,
                    output_path: None,
                });
                continue;
            }
        };

        // Build config for this level
        let adv_cfg = AdvConfig {
            level: adv_level,
            threshold: cfg.threshold,
            cols: AdvColumns::default(),
            allow_birthdate_swap: cfg.allow_birthdate_swap,
        };

        // In exclusive mode, filter input tables to exclude already-matched records
        let (filtered_t1, filtered_t2): (Vec<Person>, Vec<Person>) = if use_exclusion
            && !matched_pairs.is_empty()
        {
            // Build sets of IDs that have already been matched
            let matched_t1_ids: HashSet<i64> = matched_pairs.iter().map(|(id1, _)| *id1).collect();
            let matched_t2_ids: HashSet<i64> = matched_pairs.iter().map(|(_, id2)| *id2).collect();

            let t1_remaining: Vec<Person> = table1
                .iter()
                .filter(|p| !matched_t1_ids.contains(&p.id))
                .cloned()
                .collect();
            let t2_remaining: Vec<Person> = table2
                .iter()
                .filter(|p| !matched_t2_ids.contains(&p.id))
                .cloned()
                .collect();

            log::info!(
                "Exclusive mode: {} of {} records remain in T1, {} of {} in T2",
                t1_remaining.len(),
                table1.len(),
                t2_remaining.len(),
                table2.len()
            );

            (t1_remaining, t2_remaining)
        } else {
            // First level or independent mode: use full tables
            (table1.to_vec(), table2.to_vec())
        };

        // Run matching on (potentially filtered) tables
        // L10-L11 use GPU when configured; L1-L9 always use CPU
        let matches = if matches!(
            adv_level,
            AdvLevel::L10FuzzyBirthdateFullMiddle | AdvLevel::L11FuzzyBirthdateNoMiddle
        ) {
            run_fuzzy_level(
                &filtered_t1,
                &filtered_t2,
                &adv_cfg,
                cfg.compute_backend,
                cfg.gpu_device_id,
            )
        } else {
            // L1-L9: exact matching, always CPU
            let mut m = advanced_match_inmemory(&filtered_t1, &filtered_t2, &adv_cfg);
            sort_matches_by_id(&mut m);
            m
        };
        let match_count = matches.len();
        total_matches += match_count;

        // In exclusive mode, track the newly matched pairs
        if use_exclusion {
            for m in &matches {
                matched_pairs.insert((m.person1.id, m.person2.id));
            }
        }

        log::info!(
            "Level {} complete: {} matches found",
            level_num,
            match_count
        );

        // Write output file
        let output_path = level_output_path(&cfg.base_output_path, level_num);
        on_progress(CascadeProgress {
            current_level: level_num,
            total_levels,
            level_description: desc.clone(),
            phase: CascadePhase::WritingOutput,
        });

        match write_level_csv(&output_path, &matches, adv_level, &extra_field_names) {
            Ok(_) => {
                log::info!("Level {} output written to: {}", level_num, output_path);
                on_progress(CascadeProgress {
                    current_level: level_num,
                    total_levels,
                    level_description: desc.clone(),
                    phase: CascadePhase::Completed,
                });
                entries.push(CascadeLevelEntry {
                    level: level_num,
                    description: desc,
                    status: CascadeLevelStatus::Completed,
                    match_count,
                    output_path: Some(output_path),
                });
            }
            Err(e) => {
                log::error!("Failed to write Level {} output: {}", level_num, e);
                entries.push(CascadeLevelEntry {
                    level: level_num,
                    description: desc,
                    status: CascadeLevelStatus::Failed(format!("Write error: {}", e)),
                    match_count,
                    output_path: None,
                });
            }
        }
    }

    let duration_ms = start.elapsed().as_millis() as u64;
    log::info!(
        "Cascade complete: {} unique matches across {} levels in {:.2}s (exclusion mode: {:?})",
        total_matches,
        entries.len(),
        duration_ms as f64 / 1000.0,
        cfg.exclusion_mode
    );

    CascadeResult {
        entries,
        total_matches,
        total_duration_ms: duration_ms,
    }
}

/// Write matches for a single level to CSV, including all Table 2 extra fields
fn write_level_csv(
    path: &str,
    matches: &[MatchPair],
    level: AdvLevel,
    extra_field_names: &[String],
) -> std::io::Result<()> {
    use std::io::Write;

    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);

    // Build header with core fields + extra Table 2 fields
    let mut headers = vec![
        "Table1_ID",
        "Table1_UUID",
        "Table1_FirstName",
        "Table1_MiddleName",
        "Table1_LastName",
        "Table1_Birthdate",
        "Table2_ID",
        "Table2_UUID",
        "Table2_FirstName",
        "Table2_MiddleName",
        "Table2_LastName",
        "Table2_Birthdate",
    ];
    // Add extra field headers for Table 2
    let extra_headers: Vec<String> = extra_field_names
        .iter()
        .map(|f| format!("Table2_{}", f))
        .collect();
    let extra_header_refs: Vec<&str> = extra_headers.iter().map(|s| s.as_str()).collect();
    headers.extend(extra_header_refs.iter().copied());
    headers.push("AdvancedLevel");
    headers.push("Confidence");
    headers.push("MatchedFields");

    writeln!(writer, "{}", headers.join(","))?;

    // Level code for output
    let level_code = match level {
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
    };

    for pair in matches {
        let p1 = &pair.person1;
        let p2 = &pair.person2;

        // Build the row with core fields
        let mut row = vec![
            p1.id.to_string(),
            p1.uuid.as_deref().unwrap_or("").to_string(),
            escape_csv(p1.first_name.as_deref().unwrap_or("")),
            escape_csv(p1.middle_name.as_deref().unwrap_or("")),
            escape_csv(p1.last_name.as_deref().unwrap_or("")),
            p1.birthdate.map(|d| d.to_string()).unwrap_or_default(),
            p2.id.to_string(),
            p2.uuid.as_deref().unwrap_or("").to_string(),
            escape_csv(p2.first_name.as_deref().unwrap_or("")),
            escape_csv(p2.middle_name.as_deref().unwrap_or("")),
            escape_csv(p2.last_name.as_deref().unwrap_or("")),
            p2.birthdate.map(|d| d.to_string()).unwrap_or_default(),
        ];

        // Add extra fields from Table 2 (person2)
        for field_name in extra_field_names {
            let value = p2
                .extra_fields
                .get(field_name)
                .map(|s| escape_csv(s))
                .unwrap_or_default();
            row.push(value);
        }

        // Add level, confidence, and matched fields
        row.push(level_code.to_string());
        row.push(format!("{:.2}", pair.confidence));
        row.push(format!("\"{}\"", pair.matched_fields.join(", ")));

        writeln!(writer, "{}", row.join(","))?;
    }

    writer.flush()?;
    Ok(())
}

/// Escape a string for CSV output
fn escape_csv(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}
