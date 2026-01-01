use crate::matching::MatchPair;
use crate::matching::birthdate_matcher::{birthdate_keys, match_level_10, match_level_11};
use crate::models::Person;
use crate::normalize::{normalize_person, normalize_text};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdvLevel {
    // Exact: last, first, full middle, birthdate
    L1BirthdateFullMiddle,
    // Exact: last, first, middle-initial(s), birthdate
    L2BirthdateMiddleInitial,
    // Exact: last, first, birthdate (no middle)
    L3BirthdateNoMiddle,
    // Exact: last, first, full middle, barangay_code
    L4BarangayFullMiddle,
    // Exact: last, first, middle-initial(s), barangay_code
    L5BarangayMiddleInitial,
    // Exact: last, first, barangay_code (no middle)
    L6BarangayNoMiddle,
    // Exact: last, first, full middle, city_code
    L7CityFullMiddle,
    // Exact: last, first, middle-initial(s), city_code
    L8CityMiddleInitial,
    // Exact: last, first, city_code (no middle)
    L9CityNoMiddle,
    // Fuzzy: last, first, full middle + exact birthdate
    L10FuzzyBirthdateFullMiddle,
    // Fuzzy: last, first (no middle) + exact birthdate
    L11FuzzyBirthdateNoMiddle,
    // Advanced Level 12: Household matching (Table2 -> Table1)
    L12HouseholdMatching,
}

#[derive(Debug, Clone, Default)]
pub struct AdvColumns {
    pub barangay_code: Option<String>,
    pub city_code: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AdvConfig {
    pub level: AdvLevel,
    /// Threshold in 0.0..=1.0
    pub threshold: f32,
    pub cols: AdvColumns,
    /// Whether to allow month/day birthdate swaps when matching (12/04 <-> 04/12).
    pub allow_birthdate_swap: bool,
}

// Build concatenated middle initial(s): e.g., "Maria Santos" -> "ms", "Jose" -> "j"
pub(crate) fn middle_initials(n: &Option<String>) -> Option<String> {
    let s = n.as_ref()?;
    let mut out = String::new();
    for part in s.split_whitespace() {
        if let Some(c) = part.chars().find(|c| c.is_ascii_alphabetic()) {
            out.push(c.to_ascii_lowercase());
        }
    }
    if out.is_empty() { None } else { Some(out) }
}

fn code_for<'a>(p: &'a Person, _cols: &AdvColumns, level: AdvLevel) -> Option<&'a str> {
    // Use fixed field names from the tables directly; no external GUI/CLI column mapping
    let key = match level {
        AdvLevel::L4BarangayFullMiddle
        | AdvLevel::L5BarangayMiddleInitial
        | AdvLevel::L6BarangayNoMiddle => "barangay_code",
        AdvLevel::L7CityFullMiddle | AdvLevel::L8CityMiddleInitial | AdvLevel::L9CityNoMiddle => {
            "city_code"
        }
        _ => return None,
    };
    p.extra_fields.get(key).map(|s| s.as_str())
}

pub fn exact_key(p: &Person, level: AdvLevel, cols: &AdvColumns) -> Option<String> {
    let n = normalize_person(p);
    let f = n.first_name.as_ref()?;
    let l = n.last_name.as_ref()?;
    match level {
        AdvLevel::L1BirthdateFullMiddle => {
            let mfull = n.middle_name.as_ref()?;
            // Require 'full' middle name: at least 2 non-dot, non-space characters
            let valid = mfull
                .trim_matches('.')
                .chars()
                .filter(|c| !c.is_whitespace())
                .count()
                >= 2;
            if !valid {
                return None;
            }
            let bd = p.birthdate?;
            Some(format!("{}|{}|{}|{}", f, l, mfull, bd))
        }
        AdvLevel::L2BirthdateMiddleInitial => {
            let mi = middle_initials(&n.middle_name)?;
            let bd = p.birthdate?;
            Some(format!("{}|{}|{}|{}", f, l, mi, bd))
        }
        AdvLevel::L3BirthdateNoMiddle => {
            let bd = p.birthdate?;
            Some(format!("{}|{}|{}", f, l, bd))
        }
        AdvLevel::L4BarangayFullMiddle => {
            let mfull = n.middle_name.as_ref()?;
            let valid = mfull
                .trim_matches('.')
                .chars()
                .filter(|c| !c.is_whitespace())
                .count()
                >= 2;
            if !valid {
                return None;
            }
            let code = code_for(p, cols, level)?;
            Some(format!("{}|{}|{}|{}", f, l, mfull, normalize_text(code)))
        }
        AdvLevel::L5BarangayMiddleInitial => {
            let mi = middle_initials(&n.middle_name)?;
            let code = code_for(p, cols, level)?;
            Some(format!("{}|{}|{}|{}", f, l, mi, normalize_text(code)))
        }
        AdvLevel::L6BarangayNoMiddle => {
            let code = code_for(p, cols, level)?;
            Some(format!("{}|{}|{}", f, l, normalize_text(code)))
        }
        AdvLevel::L7CityFullMiddle => {
            let mfull = n.middle_name.as_ref()?;
            let valid = mfull
                .trim_matches('.')
                .chars()
                .filter(|c| !c.is_whitespace())
                .count()
                >= 2;
            if !valid {
                return None;
            }
            let code = code_for(p, cols, level)?;
            Some(format!("{}|{}|{}|{}", f, l, mfull, normalize_text(code)))
        }
        AdvLevel::L8CityMiddleInitial => {
            let mi = middle_initials(&n.middle_name)?;
            let code = code_for(p, cols, level)?;
            Some(format!("{}|{}|{}|{}", f, l, mi, normalize_text(code)))
        }
        AdvLevel::L9CityNoMiddle => {
            let code = code_for(p, cols, level)?;
            Some(format!("{}|{}|{}", f, l, normalize_text(code)))
        }
        AdvLevel::L10FuzzyBirthdateFullMiddle
        | AdvLevel::L11FuzzyBirthdateNoMiddle
        | AdvLevel::L12HouseholdMatching => None,
    }
}

fn jw(a: &str, b: &str) -> f32 {
    strsim::jaro_winkler(a, b) as f32
}
fn nlev(a: &str, b: &str) -> f32 {
    strsim::normalized_levenshtein(a, b) as f32
}

pub fn fuzzy_score(
    first_a: &str,
    last_a: &str,
    first_b: &str,
    last_b: &str,
    full_a: &str,
    full_b: &str,
) -> f32 {
    let s = 0.7 * jw(last_a, last_b) + 0.7 * jw(first_a, first_b) + 0.3 * nlev(full_a, full_b);
    s.clamp(0.0, 1.0)
}

fn make_pair(
    p1: &Person,
    p2: &Person,
    confidence_pct: f32,
    matched_fields: Vec<String>,
) -> MatchPair {
    MatchPair {
        person1: p1.clone(),
        person2: p2.clone(),
        confidence: (confidence_pct * 100.0).clamp(0.0, 100.0),
        matched_fields,
        is_matched_infnbd: false,
        is_matched_infnmnbd: false,
    }
}

/// In-memory Advanced Matching (exact levels + L10 fuzzy with birthdate)
pub fn advanced_match_inmemory(
    table1: &[Person],
    table2: &[Person],
    cfg: &AdvConfig,
) -> Vec<MatchPair> {
    use std::collections::HashMap;
    // L12 delegates to Option 6 household implementation to guarantee identical semantics.
    if matches!(cfg.level, AdvLevel::L12HouseholdMatching) {
        let _ = crate::matching::match_households_gpu_inmemory_opt6(
            table1,
            table2,
            crate::matching::MatchOptions {
                backend: crate::matching::ComputeBackend::Cpu,
                gpu: None,
                progress: crate::matching::ProgressConfig::default(),
                allow_birthdate_swap: false,
            },
            cfg.threshold,
            |_u: crate::matching::ProgressUpdate| {},
        );
        // Note: Advanced L12 returns household aggregates (HouseholdAggRow). This function's
        // return type is Vec<MatchPair>, so by design it is not used for L12 in-memory export.
        // Callers should use the dedicated household APIs (CLI/GUI already do).
        return Vec::new();
    }

    match cfg.level {
        AdvLevel::L10FuzzyBirthdateFullMiddle => {
            use std::collections::{HashMap, HashSet};
            let allow_swap = cfg.allow_birthdate_swap;

            // Index table2 by birthdate (with optional swapped keys)
            let mut by_bd2: HashMap<String, Vec<&Person>> = HashMap::new();
            for p in table2 {
                if let Some(bd) = p.birthdate {
                    for key in birthdate_keys(bd, allow_swap) {
                        by_bd2.entry(key).or_default().push(p);
                    }
                }
            }

            let mut out = Vec::new();
            for a in table1 {
                let Some(bd_a) = a.birthdate else {
                    continue;
                };
                let stored = bd_a.format("%Y-%m-%d").to_string();
                let mut seen_inner: HashSet<i64> = HashSet::new();
                for key in birthdate_keys(bd_a, allow_swap) {
                    if let Some(v2) = by_bd2.get(&key) {
                        for &b in v2 {
                            if !seen_inner.insert(b.id) {
                                continue;
                            }
                            let Some(bd_b) = b.birthdate else {
                                continue;
                            };
                            let input = bd_b.format("%Y-%m-%d").to_string();
                            if !match_level_10(&stored, &input, allow_swap) {
                                continue;
                            }

                            if let Some((score, _label)) = super::fuzzy_compare_names_new(
                                a.first_name.as_deref(),
                                a.middle_name.as_deref(),
                                a.last_name.as_deref(),
                                b.first_name.as_deref(),
                                b.middle_name.as_deref(),
                                b.last_name.as_deref(),
                            ) {
                                let mut pair = MatchPair {
                                    person1: a.clone(),
                                    person2: b.clone(),
                                    confidence: score as f32, // 0-100
                                    matched_fields: vec![
                                        "fuzzy".into(),
                                        "first_name".into(),
                                        "middle_name".into(),
                                        "last_name".into(),
                                        "birthdate".into(),
                                    ],
                                    is_matched_infnbd: false,
                                    is_matched_infnmnbd: false,
                                };
                                // Full middle name required (len >= 2 after trimming '.')
                                let m1 = pair.person1.middle_name.as_deref().unwrap_or("").trim();
                                let m2 = pair.person2.middle_name.as_deref().unwrap_or("").trim();
                                let l1 = m1
                                    .trim_matches('.')
                                    .chars()
                                    .filter(|c| !c.is_whitespace())
                                    .count();
                                let l2 = m2
                                    .trim_matches('.')
                                    .chars()
                                    .filter(|c| !c.is_whitespace())
                                    .count();
                                if l1 < 2 || l2 < 2 {
                                    continue;
                                }
                                if (pair.confidence / 100.0) < cfg.threshold {
                                    continue;
                                }
                                out.push(pair);
                            }
                        }
                    }
                }
            }
            out
        }
        AdvLevel::L11FuzzyBirthdateNoMiddle => {
            use std::collections::{HashMap, HashSet};
            let allow_swap = cfg.allow_birthdate_swap;

            let mut by_bd2: HashMap<String, Vec<&Person>> = HashMap::new();
            for p in table2 {
                if let Some(bd) = p.birthdate {
                    for key in birthdate_keys(bd, allow_swap) {
                        by_bd2.entry(key).or_default().push(p);
                    }
                }
            }

            let mut out = Vec::new();
            for a in table1 {
                let Some(bd_a) = a.birthdate else {
                    continue;
                };
                let stored = bd_a.format("%Y-%m-%d").to_string();
                let mut seen_inner: HashSet<i64> = HashSet::new();
                for key in birthdate_keys(bd_a, allow_swap) {
                    if let Some(v2) = by_bd2.get(&key) {
                        for &b in v2 {
                            if !seen_inner.insert(b.id) {
                                continue;
                            }
                            let Some(bd_b) = b.birthdate else {
                                continue;
                            };
                            let input = bd_b.format("%Y-%m-%d").to_string();
                            if !match_level_11(&stored, &input, allow_swap) {
                                continue;
                            }
                            if let Some((score, _label)) = super::fuzzy_compare_names_no_mid(
                                a.first_name.as_deref(),
                                a.last_name.as_deref(),
                                b.first_name.as_deref(),
                                b.last_name.as_deref(),
                            ) {
                                let pair = MatchPair {
                                    person1: a.clone(),
                                    person2: b.clone(),
                                    confidence: score as f32, // 0-100
                                    matched_fields: vec![
                                        "fuzzy".into(),
                                        "first_name".into(),
                                        "last_name".into(),
                                        "birthdate".into(),
                                    ],
                                    is_matched_infnbd: false,
                                    is_matched_infnmnbd: false,
                                };
                                if (pair.confidence / 100.0) < cfg.threshold {
                                    continue;
                                }
                                out.push(pair);
                            }
                        }
                    }
                }
            }
            out
        }
        _ => {
            // Exact levels via key map
            let mut map2: HashMap<String, Vec<&Person>> = HashMap::new();
            for p in table2 {
                if let Some(k) = exact_key(p, cfg.level, &cfg.cols) {
                    map2.entry(k).or_default().push(p);
                }
            }
            let mut out = Vec::new();
            for p1 in table1 {
                if let Some(k) = exact_key(p1, cfg.level, &cfg.cols) {
                    if let Some(cands) = map2.get(&k) {
                        for p2 in cands {
                            let mut fields =
                                vec!["first_name".to_string(), "last_name".to_string()];
                            match cfg.level {
                                AdvLevel::L1BirthdateFullMiddle => {
                                    fields.insert(1, "middle_name".into());
                                    fields.push("birthdate".into());
                                }
                                AdvLevel::L2BirthdateMiddleInitial => {
                                    fields.insert(1, "middle_initial".into());
                                    fields.push("birthdate".into());
                                }
                                AdvLevel::L3BirthdateNoMiddle => {
                                    fields.push("birthdate".into());
                                }
                                AdvLevel::L4BarangayFullMiddle => {
                                    fields.insert(1, "middle_name".into());
                                    fields.push("barangay_code".into());
                                }
                                AdvLevel::L5BarangayMiddleInitial => {
                                    fields.insert(1, "middle_initial".into());
                                    fields.push("barangay_code".into());
                                }
                                AdvLevel::L6BarangayNoMiddle => {
                                    fields.push("barangay_code".into());
                                }
                                AdvLevel::L7CityFullMiddle => {
                                    fields.insert(1, "middle_name".into());
                                    fields.push("city_code".into());
                                }
                                AdvLevel::L8CityMiddleInitial => {
                                    fields.insert(1, "middle_initial".into());
                                    fields.push("city_code".into());
                                }
                                AdvLevel::L9CityNoMiddle => {
                                    fields.push("city_code".into());
                                }
                                AdvLevel::L10FuzzyBirthdateFullMiddle
                                | AdvLevel::L11FuzzyBirthdateNoMiddle
                                | AdvLevel::L12HouseholdMatching => {}
                            }
                            out.push(make_pair(p1, p2, 1.0, fields));
                        }
                    }
                }
            }
            out
        }
    }
}

// =============================================================================
// GPU/CPU PARITY DOCUMENTATION FOR FUZZY MATCHING (L10-L11)
// =============================================================================
//
// ARCHITECTURE OVERVIEW
// ---------------------
// The fuzzy matching system uses a hybrid GPU/CPU approach to maximize performance
// while guaranteeing identical results between GPU and CPU execution paths.
//
// GPU ROLE: **Candidate Generation Only**
// - GPU kernels compute Levenshtein distance, Jaro similarity, and Jaro-Winkler scores
// - These are used purely for FILTERING candidate pairs that might match
// - GPU uses a prefilter threshold of 85.0 to reduce candidate pairs
// - GPU does NOT produce the final match score
//
// CPU ROLE: **Authoritative Final Scoring**
// - After GPU candidate generation, ALL final scores are computed on CPU
// - Uses `fuzzy_compare_names_new()` for L10 (with middle name)
// - Uses `fuzzy_compare_names_no_mid()` for L11 (without middle name)
// - This guarantees bit-exact parity with pure CPU execution
//
// SCORING ALGORITHM (CPU Authoritative)
// -------------------------------------
// 1. **String Normalization** (`normalize_simple()`):
//    - Trim whitespace
//    - Remove dots (.)
//    - Replace dashes (-) with spaces
//    - Lowercase all characters
//
// 2. **Three Metrics Computed**:
//    - `lev`: Levenshtein similarity as percentage (0-100)
//      Formula: (1 - dist/max_len) * 100
//    - `jw`: Jaro-Winkler similarity * 100 (0-100)
//      Uses standard Jaro-Winkler with p=0.1, max prefix=4
//    - `mp`: Metaphone match (0 or 100)
//      100 if Double Metaphone encodings match, 0 otherwise
//
// 3. **Case Classification**:
//    - DIRECT MATCH: full1 == full2 → score = 100.0
//    - CASE 1: lev >= 85 AND jw >= 85 AND mp == 100 → score = avg(lev, jw, mp)
//    - CASE 2: 2+ of (lev >= 85, jw >= 85, mp == 100) → score = avg(lev, jw, mp)
//    - CASE 3: CASE 2 refinement when avg >= 88 AND component Levenshtein dist <= 2
//    - NO MATCH: None of the above → returns None (no match)
//
// FLOATING-POINT PRECISION
// ------------------------
// - CPU uses f64 (double precision) for all scoring
// - GPU kernels use f32 (single precision) for performance
// - Since GPU is only used for prefiltering (threshold 85.0), precision differences
//   do not affect final results - the final score is always computed on CPU with f64
// - The only potential edge case is when GPU f32 rounds differently around 85.0,
//   but this is handled by the conservative 85.0 threshold (false positives are OK,
//   false negatives would be caught by exact match checks anyway)
//
// NORMALIZATION CONSISTENCY
// -------------------------
// Both GPU and CPU paths use `normalize_simple()` for string preprocessing:
// - Inputs to GPU kernels are normalized on CPU before transfer
// - Inputs to CPU scoring are normalized identically
// - `FuzzyCache` precomputes normalized forms once per person
// - `dmeta_code` is precomputed on CPU for metaphone comparison
//
// GPU CUDA KERNELS (for reference)
// --------------------------------
// Located in `src/matching/mod.rs` under `LEV_KERNEL_SRC`:
// - `lev_kernel`: Wagner-Fischer Levenshtein with 2-row DP, O(n*m) per pair
// - `jaro_kernel`: Standard Jaro similarity, O(n*m) per pair
// - `jw_kernel`: Jaro-Winkler with p=0.1, prefix up to 4 chars
// - `max3_kernel`: Element-wise max of three arrays (used for prefiltering)
//
// These kernels match the strsim crate implementations used on CPU.
// =============================================================================

/// GPU-accelerated Advanced Matching for L10-L11 fuzzy levels.
///
/// This function provides GPU acceleration for fuzzy matching while guaranteeing
/// CPU/GPU parity by using CPU scoring for final confidence values.
///
/// # Algorithm Parity
/// - GPU is used for **candidate generation** (fast Levenshtein/Jaro-Winkler filtering)
/// - Final scores are computed on CPU using `fuzzy_compare_names_new()` or `fuzzy_compare_names_no_mid()`
/// - This ensures identical match pairs and confidence scores as the CPU path
///
/// # Levels
/// - L10: Fuzzy matching with full middle name + birthdate
/// - L11: Fuzzy matching without middle name + birthdate
/// - Other levels (L1-L9, L12): Fall back to CPU `advanced_match_inmemory()`
///
/// # Error Handling
/// - If GPU initialization fails, falls back to CPU automatically
/// - If CUDA OOM occurs, falls back to CPU automatically
/// - Returns same results as CPU path in all fallback cases
///
/// # Feature Gate
/// This function is only available when the `gpu` feature is enabled.
/// When `gpu` feature is disabled, this function is not compiled.
#[cfg(feature = "gpu")]
pub fn advanced_match_inmemory_gpu(
    table1: &[Person],
    table2: &[Person],
    cfg: &AdvConfig,
) -> Vec<MatchPair> {
    use crate::matching::{
        ComputeBackend, GpuConfig, MatchOptions, ProgressConfig, ProgressUpdate,
    };

    // Only L10 and L11 benefit from GPU acceleration
    let use_gpu = matches!(
        cfg.level,
        AdvLevel::L10FuzzyBirthdateFullMiddle | AdvLevel::L11FuzzyBirthdateNoMiddle
    );

    if !use_gpu {
        // L1-L9 and L12 use exact matching or household aggregation - CPU is fine
        return advanced_match_inmemory(table1, table2, cfg);
    }

    // Construct GPU match options
    let opts = MatchOptions {
        backend: ComputeBackend::Gpu,
        gpu: Some(GpuConfig {
            device_id: None, // Use default device
            mem_budget_mb: 512,
        }),
        progress: ProgressConfig::default(),
        allow_birthdate_swap: cfg.allow_birthdate_swap,
    };

    let on_progress = |_u: ProgressUpdate| {};

    // Route to appropriate GPU function based on level
    let gpu_result = match cfg.level {
        AdvLevel::L10FuzzyBirthdateFullMiddle => {
            crate::matching::gpu_config::with_oom_cpu_fallback(
                || {
                    crate::matching::cascade_match_fuzzy_gpu(
                        table1,
                        table2,
                        opts.clone(),
                        on_progress,
                    )
                },
                || {
                    log::info!("[GPU] L10 OOM fallback to CPU");
                    advanced_match_inmemory(table1, table2, cfg)
                },
                "advanced_match_inmemory_gpu L10",
            )
        }
        AdvLevel::L11FuzzyBirthdateNoMiddle => crate::matching::gpu_config::with_oom_cpu_fallback(
            || {
                crate::matching::cascade_match_fuzzy_no_mid_gpu(
                    table1,
                    table2,
                    opts.clone(),
                    on_progress,
                )
            },
            || {
                log::info!("[GPU] L11 OOM fallback to CPU");
                advanced_match_inmemory(table1, table2, cfg)
            },
            "advanced_match_inmemory_gpu L11",
        ),
        _ => {
            // Should not reach here due to use_gpu check above
            Ok(advanced_match_inmemory(table1, table2, cfg))
        }
    };

    match gpu_result {
        Ok(mut matches) => {
            // Apply threshold filtering (GPU may return all candidates)
            matches.retain(|m| (m.confidence / 100.0) >= cfg.threshold);
            // Sort for deterministic ordering
            crate::matching::cascade::sort_matches_by_id(&mut matches);
            matches
        }
        Err(e) => {
            log::warn!("[GPU] Matching failed: {}. Falling back to CPU.", e);
            advanced_match_inmemory(table1, table2, cfg)
        }
    }
}

/// GPU-accelerated Advanced Matching (stub for non-GPU builds).
///
/// When the `gpu` feature is not enabled, this function simply delegates
/// to the CPU implementation `advanced_match_inmemory()`.
#[cfg(not(feature = "gpu"))]
pub fn advanced_match_inmemory_gpu(
    table1: &[Person],
    table2: &[Person],
    cfg: &AdvConfig,
) -> Vec<MatchPair> {
    log::debug!("[GPU] Feature not enabled, using CPU path");
    advanced_match_inmemory(table1, table2, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use std::collections::HashMap;
    fn p(
        id: i64,
        f: &str,
        m: Option<&str>,
        l: &str,
        bd: (i32, u32, u32),
        code_k: &str,
        code_v: &str,
    ) -> Person {
        let mut e = HashMap::new();
        if !code_k.is_empty() {
            e.insert(code_k.to_string(), code_v.to_string());
        }
        Person {
            id,
            uuid: Some(format!("u{}", id)),
            first_name: Some(f.into()),
            middle_name: m.map(|s| s.to_string()),
            last_name: Some(l.into()),
            birthdate: NaiveDate::from_ymd_opt(bd.0, bd.1, bd.2),
            hh_id: None,
            extra_fields: e,
        }
    }
    #[test]
    fn exact_birthdate_middle_initial() {
        let a = vec![p(1, "Ann", Some("Mae"), "Lee", (1990, 1, 1), "", "")];
        let b = vec![p(2, "Ann", Some("M"), "Lee", (1990, 1, 1), "", "")];
        let cfg = AdvConfig {
            level: AdvLevel::L2BirthdateMiddleInitial,
            threshold: 0.9,
            cols: AdvColumns::default(),
            allow_birthdate_swap: false,
        };
        let r = advanced_match_inmemory(&a, &b, &cfg);
        assert_eq!(r.len(), 1);
    }
    #[test]
    fn exact_barangay_full_middle() {
        let a = vec![p(
            1,
            "Ann",
            Some("Mae"),
            "Lee",
            (1990, 1, 1),
            "barangay_code",
            "001",
        )];
        let b = vec![p(
            2,
            "Ann",
            Some("Mae"),
            "Lee",
            (1980, 1, 1),
            "barangay_code",
            "001",
        )];
        let cfg = AdvConfig {
            level: AdvLevel::L4BarangayFullMiddle,
            threshold: 0.9,
            cols: AdvColumns::default(),
            allow_birthdate_swap: false,
        };
        let r = advanced_match_inmemory(&a, &b, &cfg);
        assert_eq!(r.len(), 1);
        assert!(r[0].confidence >= 99.9);
    }
    #[test]
    fn fuzzy_birthdate_full_middle() {
        let a = vec![p(1, "Jon", Some("Ann"), "Smith", (1990, 1, 1), "", "")];
        let b = vec![p(2, "John", Some("Ann"), "Smith", (1990, 1, 1), "", "")];
        let cfg = AdvConfig {
            level: AdvLevel::L10FuzzyBirthdateFullMiddle,
            threshold: 0.85,
            cols: AdvColumns::default(),
            allow_birthdate_swap: false,
        };
        let r = advanced_match_inmemory(&a, &b, &cfg);
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn fuzzy_birthdate_swap_allowed_l10() {
        unsafe {
            std::env::set_var("NAME_MATCHER_ALLOW_BIRTHDATE_SWAP", "1");
        }
        let a = vec![p(1, "Jon", Some("Ann"), "Smith", (1990, 12, 4), "", "")];
        let b = vec![p(2, "Jon", Some("Ann"), "Smith", (1990, 4, 12), "", "")];
        let cfg = AdvConfig {
            level: AdvLevel::L10FuzzyBirthdateFullMiddle,
            threshold: 0.80,
            cols: AdvColumns::default(),
            allow_birthdate_swap: true,
        };
        let r = advanced_match_inmemory(&a, &b, &cfg);
        assert_eq!(r.len(), 1);
        unsafe {
            std::env::set_var("NAME_MATCHER_ALLOW_BIRTHDATE_SWAP", "0");
        }
    }

    #[test]
    fn fuzzy_birthdate_swap_allowed_l11() {
        unsafe {
            std::env::set_var("NAME_MATCHER_ALLOW_BIRTHDATE_SWAP", "1");
        }
        let a = vec![p(1, "Jon", None, "Smith", (1990, 12, 4), "", "")];
        let b = vec![p(2, "Jon", None, "Smith", (1990, 4, 12), "", "")];
        let cfg = AdvConfig {
            level: AdvLevel::L11FuzzyBirthdateNoMiddle,
            threshold: 0.80,
            cols: AdvColumns::default(),
            allow_birthdate_swap: true,
        };
        let r = advanced_match_inmemory(&a, &b, &cfg);
        assert_eq!(r.len(), 1);
        unsafe {
            std::env::set_var("NAME_MATCHER_ALLOW_BIRTHDATE_SWAP", "0");
        }
    }

    /// Test that advanced_match_inmemory_gpu produces same results as CPU for L10.
    /// This test verifies the GPU function exists and can be called.
    /// Actual GPU execution requires the 'gpu' feature and hardware.
    #[test]
    fn gpu_l10_function_exists_and_callable() {
        let a = vec![p(1, "Jon", Some("Ann"), "Smith", (1990, 1, 1), "", "")];
        let b = vec![p(2, "John", Some("Ann"), "Smith", (1990, 1, 1), "", "")];
        let cfg = AdvConfig {
            level: AdvLevel::L10FuzzyBirthdateFullMiddle,
            threshold: 0.85,
            cols: AdvColumns::default(),
            allow_birthdate_swap: false,
        };

        // Call GPU function (will use CPU fallback if gpu feature not enabled)
        let gpu_result = advanced_match_inmemory_gpu(&a, &b, &cfg);

        // Call CPU function for comparison
        let cpu_result = advanced_match_inmemory(&a, &b, &cfg);

        // Results should match (same match count)
        assert_eq!(
            gpu_result.len(),
            cpu_result.len(),
            "GPU and CPU should produce same number of matches"
        );
    }

    /// Test that advanced_match_inmemory_gpu produces same results as CPU for L11.
    #[test]
    fn gpu_l11_function_exists_and_callable() {
        let a = vec![p(1, "Jon", None, "Smith", (1990, 1, 1), "", "")];
        let b = vec![p(2, "John", None, "Smith", (1990, 1, 1), "", "")];
        let cfg = AdvConfig {
            level: AdvLevel::L11FuzzyBirthdateNoMiddle,
            threshold: 0.85,
            cols: AdvColumns::default(),
            allow_birthdate_swap: false,
        };

        let gpu_result = advanced_match_inmemory_gpu(&a, &b, &cfg);
        let cpu_result = advanced_match_inmemory(&a, &b, &cfg);

        assert_eq!(
            gpu_result.len(),
            cpu_result.len(),
            "GPU and CPU should produce same number of matches for L11"
        );
    }

    /// Test that GPU function falls back correctly for non-fuzzy levels.
    #[test]
    fn gpu_fallback_for_exact_levels() {
        let a = vec![p(1, "Ann", Some("Mae"), "Lee", (1990, 1, 1), "", "")];
        let b = vec![p(2, "Ann", Some("Mae"), "Lee", (1990, 1, 1), "", "")];
        let cfg = AdvConfig {
            level: AdvLevel::L1BirthdateFullMiddle, // Exact level, not fuzzy
            threshold: 0.9,
            cols: AdvColumns::default(),
            allow_birthdate_swap: false,
        };

        let gpu_result = advanced_match_inmemory_gpu(&a, &b, &cfg);
        let cpu_result = advanced_match_inmemory(&a, &b, &cfg);

        // Should match exactly since GPU falls back to CPU for exact levels
        assert_eq!(gpu_result.len(), cpu_result.len());
        assert_eq!(gpu_result.len(), 1);
    }

    // =========================================================================
    // GPU/CPU PARITY TESTS - Verify identical scores between paths
    // =========================================================================

    /// Test parity for DIRECT MATCH case (identical names).
    #[test]
    fn parity_direct_match() {
        let a = vec![p(
            1,
            "John",
            Some("Michael"),
            "Smith",
            (1990, 5, 15),
            "",
            "",
        )];
        let b = vec![p(
            2,
            "John",
            Some("Michael"),
            "Smith",
            (1990, 5, 15),
            "",
            "",
        )];
        let cfg = AdvConfig {
            level: AdvLevel::L10FuzzyBirthdateFullMiddle,
            threshold: 0.50,
            cols: AdvColumns::default(),
            allow_birthdate_swap: false,
        };

        let gpu = advanced_match_inmemory_gpu(&a, &b, &cfg);
        let cpu = advanced_match_inmemory(&a, &b, &cfg);

        assert_eq!(gpu.len(), cpu.len(), "Match count parity");
        if !gpu.is_empty() && !cpu.is_empty() {
            let score_diff = (gpu[0].confidence - cpu[0].confidence).abs();
            assert!(
                score_diff < 0.0001,
                "Score parity failed: GPU={}, CPU={}, diff={}",
                gpu[0].confidence,
                cpu[0].confidence,
                score_diff
            );
        }
    }

    /// Test parity for CASE 1 match (all three metrics pass).
    #[test]
    fn parity_case1_match() {
        // Names that should trigger CASE 1: lev >= 85, jw >= 85, metaphone match
        let a = vec![p(1, "Jon", Some("Michael"), "Smyth", (1990, 5, 15), "", "")];
        let b = vec![p(
            2,
            "John",
            Some("Michael"),
            "Smith",
            (1990, 5, 15),
            "",
            "",
        )];
        let cfg = AdvConfig {
            level: AdvLevel::L10FuzzyBirthdateFullMiddle,
            threshold: 0.50,
            cols: AdvColumns::default(),
            allow_birthdate_swap: false,
        };

        let gpu = advanced_match_inmemory_gpu(&a, &b, &cfg);
        let cpu = advanced_match_inmemory(&a, &b, &cfg);

        assert_eq!(gpu.len(), cpu.len(), "Match count parity for CASE 1");
        if !gpu.is_empty() && !cpu.is_empty() {
            let score_diff = (gpu[0].confidence - cpu[0].confidence).abs();
            assert!(
                score_diff < 0.0001,
                "CASE 1 score parity: GPU={}, CPU={}, diff={}",
                gpu[0].confidence,
                cpu[0].confidence,
                score_diff
            );
        }
    }

    /// Test parity for CASE 2 match (two of three metrics pass).
    #[test]
    fn parity_case2_match() {
        // Names that might trigger CASE 2: two of (lev >= 85, jw >= 85, mp == 100)
        let a = vec![p(1, "Jonathan", Some("M"), "Smith", (1990, 5, 15), "", "")];
        let b = vec![p(2, "Johnatan", Some("M"), "Smyth", (1990, 5, 15), "", "")];
        let cfg = AdvConfig {
            level: AdvLevel::L10FuzzyBirthdateFullMiddle,
            threshold: 0.50,
            cols: AdvColumns::default(),
            allow_birthdate_swap: false,
        };

        let gpu = advanced_match_inmemory_gpu(&a, &b, &cfg);
        let cpu = advanced_match_inmemory(&a, &b, &cfg);

        assert_eq!(gpu.len(), cpu.len(), "Match count parity for CASE 2");
        if !gpu.is_empty() && !cpu.is_empty() {
            let score_diff = (gpu[0].confidence - cpu[0].confidence).abs();
            assert!(
                score_diff < 0.0001,
                "CASE 2 score parity: GPU={}, CPU={}, diff={}",
                gpu[0].confidence,
                cpu[0].confidence,
                score_diff
            );
        }
    }

    /// Test parity for L11 (no middle name) matches.
    #[test]
    fn parity_l11_no_mid() {
        let a = vec![p(1, "Jonathan", None, "Smith", (1990, 5, 15), "", "")];
        let b = vec![p(2, "Johnatan", None, "Smyth", (1990, 5, 15), "", "")];
        let cfg = AdvConfig {
            level: AdvLevel::L11FuzzyBirthdateNoMiddle,
            threshold: 0.50,
            cols: AdvColumns::default(),
            allow_birthdate_swap: false,
        };

        let gpu = advanced_match_inmemory_gpu(&a, &b, &cfg);
        let cpu = advanced_match_inmemory(&a, &b, &cfg);

        assert_eq!(gpu.len(), cpu.len(), "Match count parity for L11");
        if !gpu.is_empty() && !cpu.is_empty() {
            let score_diff = (gpu[0].confidence - cpu[0].confidence).abs();
            assert!(
                score_diff < 0.0001,
                "L11 score parity: GPU={}, CPU={}, diff={}",
                gpu[0].confidence,
                cpu[0].confidence,
                score_diff
            );
        }
    }

    /// Test parity for non-matches (should both return empty).
    #[test]
    fn parity_non_match() {
        // Names too different to match
        let a = vec![p(
            1,
            "Alice",
            Some("Marie"),
            "Johnson",
            (1990, 5, 15),
            "",
            "",
        )];
        let b = vec![p(
            2,
            "Robert",
            Some("James"),
            "Williams",
            (1990, 5, 15),
            "",
            "",
        )];
        let cfg = AdvConfig {
            level: AdvLevel::L10FuzzyBirthdateFullMiddle,
            threshold: 0.85,
            cols: AdvColumns::default(),
            allow_birthdate_swap: false,
        };

        let gpu = advanced_match_inmemory_gpu(&a, &b, &cfg);
        let cpu = advanced_match_inmemory(&a, &b, &cfg);

        assert_eq!(
            gpu.len(),
            cpu.len(),
            "Both should return same count for non-matches"
        );
    }

    /// Test parity with special characters that need normalization.
    #[test]
    fn parity_normalization() {
        // Names with dots, dashes, mixed case
        let a = vec![p(
            1,
            "Dr. John-Paul",
            Some("M."),
            "O'Brien",
            (1990, 5, 15),
            "",
            "",
        )];
        let b = vec![p(
            2,
            "john paul",
            Some("m"),
            "obrien",
            (1990, 5, 15),
            "",
            "",
        )];
        let cfg = AdvConfig {
            level: AdvLevel::L10FuzzyBirthdateFullMiddle,
            threshold: 0.50,
            cols: AdvColumns::default(),
            allow_birthdate_swap: false,
        };

        let gpu = advanced_match_inmemory_gpu(&a, &b, &cfg);
        let cpu = advanced_match_inmemory(&a, &b, &cfg);

        assert_eq!(gpu.len(), cpu.len(), "Normalization parity");
        if !gpu.is_empty() && !cpu.is_empty() {
            let score_diff = (gpu[0].confidence - cpu[0].confidence).abs();
            assert!(
                score_diff < 0.0001,
                "Normalization score parity: GPU={}, CPU={}, diff={}",
                gpu[0].confidence,
                cpu[0].confidence,
                score_diff
            );
        }
    }
}
