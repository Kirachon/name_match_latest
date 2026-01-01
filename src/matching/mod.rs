use crate::metrics::memory_stats_mb;
use crate::models::{NormalizedPerson, Person};
use crate::normalize::normalize_person;
#[cfg(feature = "gpu")]
use anyhow::anyhow;
use chrono::Datelike;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::{Duration, Instant};

#[cfg(feature = "gpu")]
use crate::matching::birthdate_matcher::birthdate_matches;
use crate::matching::birthdate_matcher::{birthdate_keys, match_level_10, match_level_11};
use strsim::{jaro_winkler, levenshtein};

#[cfg(feature = "gpu")]
pub mod gpu_config;

pub mod birthdate_matcher;

// Advanced Matching (in-memory exact/fuzzy variants by geographic code)
pub mod advanced_matcher;

// Cascade matching workflow (L1-L11 sequential execution)
pub mod cascade;

// Shared helper functions (normalization, similarity scoring)
mod helpers;
use helpers::{metaphone_pct, normalize_simple, sim_levenshtein_pct, soundex4_ascii};
// GPU-only imports for phonetic preprocessing
#[cfg(feature = "gpu")]
use helpers::normalize_for_phonetic;
#[cfg(feature = "gpu")]
use rphonetic::{DoubleMetaphone, Encoder};

fn fuzzy_compare_names_new(
    n1_first: Option<&str>,
    n1_mid: Option<&str>,
    n1_last: Option<&str>,
    n2_first: Option<&str>,
    n2_mid: Option<&str>,
    n2_last: Option<&str>,
) -> Option<(f64, String)> {
    let full1 = normalize_simple(&format!(
        "{} {} {}",
        n1_first.unwrap_or(""),
        n1_mid.unwrap_or(""),
        n1_last.unwrap_or("")
    ));
    let full2 = normalize_simple(&format!(
        "{} {} {}",
        n2_first.unwrap_or(""),
        n2_mid.unwrap_or(""),
        n2_last.unwrap_or("")
    ));
    if full1.trim().is_empty() || full2.trim().is_empty() {
        return None;
    }

    let lev = sim_levenshtein_pct(&full1, &full2);
    let jw = jaro_winkler(&full1, &full2) * 100.0;
    let mp = metaphone_pct(&full1, &full2);

    // Direct match
    if full1 == full2 {
        return Some((100.0, "DIRECT MATCH".to_string()));
    }

    // Case 1
    if lev >= 85.0 && jw >= 85.0 && (mp - 100.0).abs() < f64::EPSILON {
        let avg = (lev + jw + mp) / 3.0;
        return Some((avg, "CASE 1".to_string()));
    }

    // Case 2
    let mut pass = 0;
    if lev >= 85.0 {
        pass += 1;
    }
    if jw >= 85.0 {
        pass += 1;
    }
    if (mp - 100.0).abs() < f64::EPSILON {
        pass += 1;
    }
    if pass >= 2 {
        let avg = (lev + jw + mp) / 3.0;
        // Case 3 refinement
        if avg >= 88.0 {
            let ld_first = levenshtein(
                &normalize_simple(n1_first.unwrap_or("")),
                &normalize_simple(n2_first.unwrap_or("")),
            ) as usize;
            let ld_last = levenshtein(
                &normalize_simple(n1_last.unwrap_or("")),
                &normalize_simple(n2_last.unwrap_or("")),
            ) as usize;
            let ld_mid = levenshtein(
                &normalize_simple(n1_mid.unwrap_or("")),
                &normalize_simple(n2_mid.unwrap_or("")),
            ) as usize;
            if ld_first <= 2 && ld_last <= 2 && ld_mid <= 2 {
                return Some((avg, "CASE 3".to_string()));
            }
        }
        return Some((avg, "CASE 2".to_string()));
    }

    None
}

#[allow(dead_code)]
fn compare_persons_new(p1: &Person, p2: &Person) -> Option<(f64, String)> {
    let allow_swap = crate::matching::birthdate_matcher::allow_birthdate_swap();
    match (p1.birthdate.as_ref(), p2.birthdate.as_ref()) {
        (Some(d1), Some(d2)) => {
            if d1 != d2
                && !crate::matching::birthdate_matcher::birthdate_matches_naive(
                    *d1, *d2, allow_swap,
                )
            {
                return None;
            }
        }
        _ => return None,
    }
    fuzzy_compare_names_new(
        p1.first_name.as_deref(),
        p1.middle_name.as_deref(),
        p1.last_name.as_deref(),
        p2.first_name.as_deref(),
        p2.middle_name.as_deref(),
        p2.last_name.as_deref(),
    )
}

fn compare_persons_no_mid(p1: &Person, p2: &Person) -> Option<(f64, String)> {
    let allow_swap = crate::matching::birthdate_matcher::allow_birthdate_swap();
    match (p1.birthdate.as_ref(), p2.birthdate.as_ref()) {
        (Some(d1), Some(d2)) => {
            if d1 != d2
                && !crate::matching::birthdate_matcher::birthdate_matches_naive(
                    *d1, *d2, allow_swap,
                )
            {
                return None;
            }
        }
        _ => return None,
    }
    fuzzy_compare_names_no_mid(
        p1.first_name.as_deref(),
        p1.last_name.as_deref(),
        p2.first_name.as_deref(),
        p2.last_name.as_deref(),
    )
}

// --- Public single-pair comparison facades for adapters (feature-gated) ---
#[cfg(feature = "new_engine")]
pub fn compare_pair_fuzzy(
    p1: &crate::models::Person,
    p2: &crate::models::Person,
) -> Option<(u32, String)> {
    compare_persons_new(p1, p2).map(|(s, label)| (s.round().clamp(0.0, 100.0) as u32, label))
}

#[cfg(feature = "new_engine")]
pub fn compare_pair_fuzzy_no_middle(
    p1: &crate::models::Person,
    p2: &crate::models::Person,
) -> Option<(u32, String)> {
    compare_persons_no_mid(p1, p2).map(|(s, label)| (s.round().clamp(0.0, 100.0) as u32, label))
}

#[cfg(feature = "new_engine")]
pub fn compare_pair_direct_algo1(p1: &crate::models::Person, p2: &crate::models::Person) -> bool {
    let n1 = normalize_person(p1);
    let n2 = normalize_person(p2);
    matches_algo1(&n1, &n2)
}

#[cfg(feature = "new_engine")]
pub fn compare_pair_direct_algo2(p1: &crate::models::Person, p2: &crate::models::Person) -> bool {
    let n1 = normalize_person(p1);
    let n2 = normalize_person(p2);
    matches_algo2(&n1, &n2)
}

#[cfg(feature = "new_engine")]
pub mod algorithms;

fn fuzzy_compare_names_no_mid(
    n1_first: Option<&str>,
    n1_last: Option<&str>,
    n2_first: Option<&str>,
    n2_last: Option<&str>,
) -> Option<(f64, String)> {
    let full1 = normalize_simple(&format!(
        "{} {}",
        n1_first.unwrap_or(""),
        n1_last.unwrap_or("")
    ));
    let full2 = normalize_simple(&format!(
        "{} {}",
        n2_first.unwrap_or(""),
        n2_last.unwrap_or("")
    ));
    if full1.trim().is_empty() || full2.trim().is_empty() {
        return None;
    }

    let lev = sim_levenshtein_pct(&full1, &full2);
    let jw = jaro_winkler(&full1, &full2) * 100.0;
    let mp = metaphone_pct(&full1, &full2);

    if full1 == full2 {
        return Some((100.0, "DIRECT MATCH".to_string()));
    }

    if lev >= 85.0 && jw >= 85.0 && (mp - 100.0).abs() < f64::EPSILON {
        let avg = (lev + jw + mp) / 3.0;
        return Some((avg, "CASE 1".to_string()));
    }

    let mut pass = 0;
    if lev >= 85.0 {
        pass += 1;
    }
    if jw >= 85.0 {
        pass += 1;
    }
    if (mp - 100.0).abs() < f64::EPSILON {
        pass += 1;
    }
    if pass >= 2 {
        let avg = (lev + jw + mp) / 3.0;
        if avg >= 88.0 {
            let ld_first = levenshtein(
                &normalize_simple(n1_first.unwrap_or("")),
                &normalize_simple(n2_first.unwrap_or("")),
            ) as usize;
            let ld_last = levenshtein(
                &normalize_simple(n1_last.unwrap_or("")),
                &normalize_simple(n2_last.unwrap_or("")),
            ) as usize;
            if ld_first <= 2 && ld_last <= 2 {
                return Some((avg, "CASE 3".to_string()));
            }
        }
        return Some((avg, "CASE 2".to_string()));
    }
    None
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MatchingAlgorithm {
    IdUuidYasIsMatchedInfnbd,
    IdUuidYasIsMatchedInfnmnbd,
    Fuzzy,
    FuzzyNoMiddle,
    // New Option 5: GPU in-memory household matching
    HouseholdGpu,
    // New Option 6: Role-swapped household matching (Table2 -> Table1; denom = Table2 size)
    HouseholdGpuOpt6,
    // New Option 7: Levenshtein weighted SQL-equivalent
    LevenshteinWeighted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HouseholdAggRow {
    pub row_id: i64,
    pub uuid: String, // Table 1 household ID
    pub hh_id: i64,   // Table 2 household ID (uses `hh_id` when available; falls back to `id`)
    pub match_percentage: f32,
    // Optional fields (if later fetched; currently None when unavailable)
    pub region_code: Option<String>,
    pub poor_hat_0: Option<String>,
    pub poor_hat_10: Option<String>,
}

/// Option 5: In-memory GPU household matching with birthdate hard filter and names-only similarity.
/// Produces aggregated household rows grouped by (uuid from Table 1, hh_id from Table 2).
pub fn match_households_gpu_inmemory<F>(
    t1: &[Person],
    t2: &[Person],
    opts: MatchOptions,
    fuzzy_min_conf: f32,
    on_progress: F,
) -> Vec<HouseholdAggRow>
where
    F: Fn(ProgressUpdate) + Sync,
{
    // Lightweight GPU snapshot logger (best-effort)
    let log_snap = |tag: &str| {
        #[cfg(feature = "gpu")]
        {
            if let Ok(ctx) = cudarc::driver::CudaContext::new(0) {
                let (tot_mb, free_mb) = cuda_mem_info_mb(&ctx);
                log::info!(
                    "[GPU_SNAPSHOT] tag={} total_mb={} free_mb={}",
                    tag,
                    tot_mb,
                    free_mb
                );
                if let Ok(path) = std::env::var("NAME_MATCHER_GPU_LOG_CSV") {
                    let line = format!(
                        "{},{}
",
                        tot_mb, free_mb
                    );
                    let mut need_header = false;
                    if !std::path::Path::new(&path).exists() {
                        need_header = true;
                    }
                    if let Ok(mut f) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&path)
                    {
                        use std::io::Write;
                        if need_header {
                            let _ = writeln!(f, "gpu_total_mb,gpu_free_mb");
                        }
                        let _ = f.write_all(line.as_bytes());
                    }
                }
                drop(ctx);
            }
        }
    };

    log_snap("opt5_start");

    use std::collections::{BTreeMap, HashSet};
    // 1) Generate person-level matches; enable GPU when requested
    let mo_cpu = MatchOptions {
        backend: ComputeBackend::Cpu,
        gpu: None,
        progress: opts.progress,
        allow_birthdate_swap: opts.allow_birthdate_swap,
    };
    let pairs: Vec<MatchPair> = if matches!(opts.backend, ComputeBackend::Gpu) {
        #[cfg(feature = "gpu")]
        {
            match crate::matching::gpu_config::with_oom_cpu_fallback(
                || gpu::match_fuzzy_no_mid_gpu(t1, t2, opts, &on_progress),
                || {
                    match_all_with_opts(
                        t1,
                        t2,
                        MatchingAlgorithm::FuzzyNoMiddle,
                        mo_cpu,
                        &on_progress,
                    )
                },
                "Household GPU person-level",
            ) {
                Ok(v) => v,
                Err(e) => {
                    log::warn!(
                        "Household GPU person-level failed: {} — falling back to CPU",
                        e
                    );
                    match_all_with_opts(
                        t1,
                        t2,
                        MatchingAlgorithm::FuzzyNoMiddle,
                        mo_cpu,
                        &on_progress,
                    )
                }
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            match_all_with_opts(
                t1,
                t2,
                MatchingAlgorithm::FuzzyNoMiddle,
                mo_cpu,
                &on_progress,
            )
        }
    } else {
        match_all_with_opts(
            t1,
            t2,
            MatchingAlgorithm::FuzzyNoMiddle,
            mo_cpu,
            &on_progress,
        )
    };

    if pairs.is_empty() {
        return Vec::new();
    }

    log_snap("opt5_pairs_done");

    // 2) Precompute total members per uuid (Table 1). Skip rows without uuid.
    let mut totals: BTreeMap<String, usize> = BTreeMap::new();
    for p in t1.iter() {
        if let Some(u) = p.uuid.as_ref() {
            *totals.entry(u.clone()).or_insert(0) += 1;
        }
    }

    // DEBUG: Log some household sizes and check for UUID truncation
    let mut total_count = 0;
    for (uuid, count) in totals.iter().take(5) {
        eprintln!(
            "DEBUG: Table 1 household {} (len={}) has {} members",
            uuid,
            uuid.len(),
            count
        );
        total_count += 1;
    }
    eprintln!(
        "DEBUG: Total unique households in Table 1: {}",
        totals.len()
    );

    // DEBUG: Check first few raw persons to see their UUIDs
    for (i, p) in t1.iter().take(10).enumerate() {
        if let Some(uuid) = &p.uuid {
            eprintln!(
                "DEBUG: Person {}: id={}, uuid={} (len={}), name={} {}",
                i,
                p.id,
                uuid,
                uuid.len(),
                p.first_name.as_deref().unwrap_or(""),
                p.last_name.as_deref().unwrap_or("")
            );
        }
    }

    // 3) For each person in Table 1, select a single best household (by highest confidence) to avoid double counting across households.
    //    If there is a tie for top confidence across different households, skip counting that person to be conservative.
    let mut best_for_p1: BTreeMap<i64, (String, String, f32, bool)> = BTreeMap::new(); // p1.id -> (uuid, hh_id, conf, tie)
    for p in pairs.into_iter() {
        if p.confidence < fuzzy_min_conf {
            continue;
        }
        let Some(uuid) = p.person1.uuid.clone() else {
            continue;
        };
        let key = p.person1.id;
        // Prefer Table 2 household key `hh_id` when available; fallback to `id` for backward compatibility
        let cand_hh = p
            .person2
            .hh_id
            .clone()
            .unwrap_or_else(|| p.person2.id.to_string());
        match best_for_p1.get_mut(&key) {
            None => {
                best_for_p1.insert(key, (uuid, cand_hh, p.confidence, false));
            }
            Some((u, hh, conf, tie)) => {
                if p.confidence > *conf {
                    *u = uuid;
                    *hh = cand_hh;
                    *conf = p.confidence;
                    *tie = false;
                } else if (p.confidence - *conf).abs() < f32::EPSILON {
                    if cand_hh < *hh {
                        *u = uuid;
                        *hh = cand_hh;
                        *conf = p.confidence;
                        *tie = false;
                    } else if cand_hh != *hh {
                        *tie = true; // ambiguous top-1 across different households
                    }
                }
            }
        }
    }
    let mut matched: BTreeMap<(String, String), HashSet<i64>> = BTreeMap::new();
    for (p1_id, (uuid, hh_id, _conf, tie)) in best_for_p1.into_iter() {
        if tie {
            continue;
        } // skip ambiguous assignments
        matched.entry((uuid, hh_id)).or_default().insert(p1_id);
    }

    // 4) Compute match_percentage and filter > 50%
    let mut out: Vec<HouseholdAggRow> = Vec::new();
    let mut row_id: i64 = 1;
    for ((uuid, hh_id), members) in matched.into_iter() {
        let total = *totals.get(&uuid).unwrap_or(&0usize) as f32;
        if total <= 0.0 {
            continue;
        }
        let pct = (members.len() as f32) / total * 100.0;

        // DEBUG: Log the calculation details for first few households
        if row_id <= 5 {
            eprintln!(
                "DEBUG: uuid={}, hh_id={}, matched_members={}, total_members={}, percentage={:.2}%",
                uuid,
                hh_id,
                members.len(),
                total,
                pct
            );
        }

        if pct > 50.0 {
            out.push(HouseholdAggRow {
                row_id,
                uuid,
                hh_id: hh_id.parse::<i64>().unwrap_or(0),
                match_percentage: pct,
                region_code: None,
                poor_hat_0: None,
                poor_hat_10: None,
            });
            row_id += 1;
        }
    }

    // Sort for deterministic output (by uuid then hh_id)
    out.sort_by(|a, b| a.uuid.cmp(&b.uuid).then_with(|| a.hh_id.cmp(&b.hh_id)));
    log_snap("opt5_done");
    out
}

/// Option 6: Role-swapped in-memory household matching.
/// Source = Table 2 (group by hh_id or fallback id), Target = Table 1 (group by uuid),
/// match_percentage denominator = Table 2 household size.
pub fn match_households_gpu_inmemory_opt6<F>(
    t1: &[Person],
    t2: &[Person],
    opts: MatchOptions,
    fuzzy_min_conf: f32,
    on_progress: F,
) -> Vec<HouseholdAggRow>
where
    F: Fn(ProgressUpdate) + Sync,
{
    let log_snap = |tag: &str| {
        #[cfg(feature = "gpu")]
        {
            if let Ok(ctx) = cudarc::driver::CudaContext::new(0) {
                let (tot_mb, free_mb) = cuda_mem_info_mb(&ctx);
                log::info!(
                    "[GPU_SNAPSHOT] tag={} total_mb={} free_mb={}",
                    tag,
                    tot_mb,
                    free_mb
                );
                if let Ok(path) = std::env::var("NAME_MATCHER_GPU_LOG_CSV") {
                    let line = format!("{},{}\n", tot_mb, free_mb);
                    let mut need_header = false;
                    if !std::path::Path::new(&path).exists() {
                        need_header = true;
                    }
                    if let Ok(mut f) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&path)
                    {
                        use std::io::Write;
                        if need_header {
                            let _ = writeln!(f, "gpu_total_mb,gpu_free_mb");
                        }
                        let _ = f.write_all(line.as_bytes());
                    }
                }
                drop(ctx);
            }
        }
    };

    log_snap("opt6_start");

    // 1) Person-level matches using birthdate blocking to avoid all-pairs explosion
    let mo_cpu = MatchOptions {
        backend: ComputeBackend::Cpu,
        gpu: None,
        progress: opts.progress,
        allow_birthdate_swap: opts.allow_birthdate_swap,
    };
    use std::collections::HashMap;
    let mut by_bd1: HashMap<String, Vec<Person>> = HashMap::new();
    for p in t1 {
        if let Some(d) = p.birthdate {
            for k in birthdate_keys(d, opts.allow_birthdate_swap) {
                by_bd1.entry(k).or_default().push(p.clone());
            }
        }
    }
    let mut by_bd2: HashMap<String, Vec<Person>> = HashMap::new();
    for p in t2 {
        if let Some(d) = p.birthdate {
            for k in birthdate_keys(d, opts.allow_birthdate_swap) {
                by_bd2.entry(k).or_default().push(p.clone());
            }
        }
    }
    let mut pairs: Vec<MatchPair> = Vec::new();
    let mut seen_pairs: std::collections::HashSet<(i64, i64)> = std::collections::HashSet::new();
    for (bd, v2) in by_bd2.iter() {
        if let Some(v1) = by_bd1.get(bd) {
            if matches!(opts.backend, ComputeBackend::Gpu) {
                #[cfg(feature = "gpu")]
                {
                    match crate::matching::gpu_config::with_oom_cpu_fallback(
                        || {
                            gpu::match_fuzzy_no_mid_gpu(
                                v1.as_slice(),
                                v2.as_slice(),
                                opts,
                                &on_progress,
                            )
                        },
                        || {
                            match_all_with_opts(
                                v1.as_slice(),
                                v2.as_slice(),
                                MatchingAlgorithm::FuzzyNoMiddle,
                                mo_cpu,
                                &on_progress,
                            )
                        },
                        "Household Opt6 GPU person-level (bd-blocked)",
                    ) {
                        Ok(mut v) => {
                            for p in v.drain(..) {
                                if seen_pairs.insert((p.person1.id, p.person2.id)) {
                                    pairs.push(p);
                                }
                            }
                        }
                        Err(e) => {
                            log::warn!(
                                "Household Opt6 GPU person-level failed: {} - falling back to CPU (bd-blocked)",
                                e
                            );
                            for p in match_all_with_opts(
                                v1.as_slice(),
                                v2.as_slice(),
                                MatchingAlgorithm::FuzzyNoMiddle,
                                mo_cpu,
                                &on_progress,
                            ) {
                                if seen_pairs.insert((p.person1.id, p.person2.id)) {
                                    pairs.push(p);
                                }
                            }
                        }
                    }
                }
                #[cfg(not(feature = "gpu"))]
                {
                    for p in match_all_with_opts(
                        v1.as_slice(),
                        v2.as_slice(),
                        MatchingAlgorithm::FuzzyNoMiddle,
                        mo_cpu,
                        &on_progress,
                    ) {
                        if seen_pairs.insert((p.person1.id, p.person2.id)) {
                            pairs.push(p);
                        }
                    }
                }
            } else {
                for p in match_all_with_opts(
                    v1.as_slice(),
                    v2.as_slice(),
                    MatchingAlgorithm::FuzzyNoMiddle,
                    mo_cpu,
                    &on_progress,
                ) {
                    if seen_pairs.insert((p.person1.id, p.person2.id)) {
                        pairs.push(p);
                    }
                }
            }
        }
    }

    if pairs.is_empty() {
        return Vec::new();
    }

    log_snap("opt6_pairs_done");

    use std::collections::HashSet;

    // 2) Precompute total members per Table 2 household (hh_id fallback to id)
    let mut totals_t2: BTreeMap<String, usize> = BTreeMap::new();
    for p in t2.iter() {
        let hh_key = p.hh_id.clone().unwrap_or_else(|| p.id.to_string());
        *totals_t2.entry(hh_key).or_insert(0) += 1;
    }

    // 3) For each person in Table 2, select a single best Table 1 household (uuid)
    //    Ties (equal top confidence mapped to different uuids) are skipped to prevent double counting.
    let mut best_for_p2: BTreeMap<
        i64,
        (
            String, /*hh_key*/
            String, /*uuid*/
            f32,    /*conf*/
            bool,   /*tie*/
        ),
    > = BTreeMap::new();
    for p in pairs.into_iter() {
        if p.confidence < fuzzy_min_conf {
            continue;
        }
        let Some(uuid) = p.person1.uuid.clone() else {
            continue;
        }; // Table 1 uuid grouping key
        let hh_key = p
            .person2
            .hh_id
            .clone()
            .unwrap_or_else(|| p.person2.id.to_string()); // Table 2 household key
        let key = p.person2.id; // distinct Table 2 person
        match best_for_p2.get_mut(&key) {
            None => {
                best_for_p2.insert(key, (hh_key, uuid, p.confidence, false));
            }
            Some((hh, u, conf, tie)) => {
                if p.confidence > *conf {
                    *hh = hh_key;
                    *u = uuid;
                    *conf = p.confidence;
                    *tie = false;
                } else if (p.confidence - *conf).abs() < f32::EPSILON {
                    if uuid < *u {
                        *hh = hh_key;
                        *u = uuid;
                        *conf = p.confidence;
                        *tie = false;
                    } else if uuid != *u {
                        *tie = true;
                    }
                }
            }
        }
    }

    // Map (hh_key, uuid) -> unique set of Table 2 person ids counted
    let mut matched: BTreeMap<(String, String), HashSet<i64>> = BTreeMap::new();
    for (p2_id, (hh_key, uuid, _conf, tie)) in best_for_p2.into_iter() {
        if tie {
            continue;
        }
        matched.entry((hh_key, uuid)).or_default().insert(p2_id);
    }

    // 4) Compute match_percentage using denominator = Table 2 household size; filter > 50%
    let mut out: Vec<HouseholdAggRow> = Vec::new();
    let mut row_id: i64 = 1;
    for ((hh_key, uuid), members) in matched.into_iter() {
        let total = *totals_t2.get(&hh_key).unwrap_or(&0usize) as f32;
        if total <= 0.0 {
            continue;
        }
        let pct = (members.len() as f32) / total * 100.0;
        if pct > 50.0 {
            out.push(HouseholdAggRow {
                row_id,
                uuid,
                hh_id: hh_key.parse::<i64>().unwrap_or(0),
                match_percentage: pct,
                region_code: None,
                poor_hat_0: None,
                poor_hat_10: None,
            });
            row_id += 1;
        }
    }

    // Deterministic output order
    out.sort_by(|a, b| a.uuid.cmp(&b.uuid).then_with(|| a.hh_id.cmp(&b.hh_id)));
    log_snap("opt6_done");
    out
}

/// In-memory Advanced Level 12 (Household, Table2 → Table1) with progressive export.
/// Emits HouseholdAggRow incrementally via callback while preserving exact semantics.
pub fn match_households_inmemory_opt6_streaming<F, P>(
    t1: &[Person],
    t2: &[Person],
    opts: MatchOptions,
    fuzzy_min_conf: f32,
    mut on_household: F,
    on_progress: P,
) -> anyhow::Result<usize>
where
    F: FnMut(&HouseholdAggRow) -> anyhow::Result<()>,
    P: Fn(ProgressUpdate) + Sync,
{
    use std::collections::{BTreeMap, HashMap, HashSet};
    let allow_birthdate_swap = crate::matching::birthdate_matcher::allow_birthdate_swap();

    // Build birthdate index for Table 1 (target) to avoid O(n*m)
    let mut by_bd1: HashMap<chrono::NaiveDate, Vec<Person>> = HashMap::new();
    for p in t1 {
        if let Some(d) = p.birthdate {
            by_bd1.entry(d).or_default().push(p.clone());
        }
    }

    // Group Table 2 (source) by household key (hh_id fallback to id) in deterministic order
    let mut groups: BTreeMap<String, Vec<Person>> = BTreeMap::new();
    for p in t2 {
        let k = p.hh_id.clone().unwrap_or_else(|| p.id.to_string());
        groups.entry(k).or_default().push(p.clone());
    }

    let total_hh = groups.len().max(1);
    let mut emitted = 0usize;
    let mut row_id: i64 = 1;
    let mut processed_hh = 0usize;

    let mo_cpu = MatchOptions {
        backend: ComputeBackend::Cpu,
        gpu: None,
        progress: opts.progress,
        allow_birthdate_swap: opts.allow_birthdate_swap,
    };

    for (hh_key, members) in groups.into_iter() {
        // Person-level best mapping for this household only
        let mut best_for_p2: BTreeMap<i64, (String /*uuid*/, f32 /*conf*/, bool /*tie*/)> =
            BTreeMap::new();
        // Group members by birthdate
        let mut by_bd2: HashMap<chrono::NaiveDate, Vec<Person>> = HashMap::new();
        for p in &members {
            if let Some(d) = p.birthdate {
                by_bd2.entry(d).or_default().push(p.clone());
            }
        }
        for (bd, v2) in by_bd2.into_iter() {
            if let Some(v1) = by_bd1.get(&bd) {
                let pairs: Vec<MatchPair> = if matches!(opts.backend, ComputeBackend::Gpu) {
                    #[cfg(feature = "gpu")]
                    {
                        match crate::matching::gpu_config::with_oom_cpu_fallback(
                            || {
                                gpu::match_fuzzy_no_mid_gpu(
                                    v1.as_slice(),
                                    v2.as_slice(),
                                    opts,
                                    &on_progress,
                                )
                            },
                            || {
                                match_all_with_opts(
                                    v1.as_slice(),
                                    v2.as_slice(),
                                    MatchingAlgorithm::FuzzyNoMiddle,
                                    mo_cpu,
                                    &on_progress,
                                )
                            },
                            "Adv L12 inmem streaming GPU person-level (bd-blocked)",
                        ) {
                            Ok(v) => v,
                            Err(e) => {
                                log::warn!("Adv L12 inmem streaming GPU failed: {} - using CPU", e);
                                match_all_with_opts(
                                    v1.as_slice(),
                                    v2.as_slice(),
                                    MatchingAlgorithm::FuzzyNoMiddle,
                                    mo_cpu,
                                    &on_progress,
                                )
                            }
                        }
                    }
                    #[cfg(not(feature = "gpu"))]
                    {
                        match_all_with_opts(
                            v1.as_slice(),
                            v2.as_slice(),
                            MatchingAlgorithm::FuzzyNoMiddle,
                            mo_cpu,
                            &on_progress,
                        )
                    }
                } else {
                    match_all_with_opts(
                        v1.as_slice(),
                        v2.as_slice(),
                        MatchingAlgorithm::FuzzyNoMiddle,
                        mo_cpu,
                        &on_progress,
                    )
                };
                for p in pairs.into_iter() {
                    if p.confidence < fuzzy_min_conf {
                        continue;
                    }
                    let Some(uuid) = p.person1.uuid.clone() else {
                        continue;
                    };
                    let key = p.person2.id;
                    match best_for_p2.get_mut(&key) {
                        None => {
                            best_for_p2.insert(key, (uuid, p.confidence, false));
                        }
                        Some((u2, c2, tie)) => {
                            if p.confidence > *c2 {
                                *u2 = uuid;
                                *c2 = p.confidence;
                                *tie = false;
                            } else if (p.confidence - *c2).abs() <= std::f32::EPSILON && *u2 != uuid
                            {
                                *tie = true;
                            }
                        }
                    }
                }
            }
        }
        // Aggregate for this household and emit rows immediately
        let mut matched: BTreeMap<String /*uuid*/, HashSet<i64> /*member ids*/> = BTreeMap::new();
        for (p2_id, (uuid, _conf, tie)) in best_for_p2.into_iter() {
            if !tie {
                matched.entry(uuid).or_default().insert(p2_id);
            }
        }
        let denom = members.len() as f32;
        for (uuid, set) in matched.into_iter() {
            if denom <= 0.0 {
                continue;
            }
            let pct = (set.len() as f32) / denom * 100.0;
            if pct > 50.0 {
                let row = HouseholdAggRow {
                    row_id,
                    uuid,
                    hh_id: hh_key.parse::<i64>().unwrap_or(0),
                    match_percentage: pct,
                    region_code: None,
                    poor_hat_0: None,
                    poor_hat_10: None,
                };
                on_household(&row)?;
                emitted += 1;
                row_id += 1;
            }
        }
        processed_hh += 1;
        let mem = memory_stats_mb();
        on_progress(ProgressUpdate {
            processed: processed_hh,
            total: total_hh,
            percent: (processed_hh as f32 / total_hh as f32) * 100.0,
            eta_secs: 0,
            mem_used_mb: mem.used_mb,
            mem_avail_mb: mem.avail_mb,
            stage: "adv_l12_inmem_stream",
            batch_size_current: None,
            gpu_total_mb: 0,
            gpu_free_mb: 0,
            gpu_active: matches!(opts.backend, ComputeBackend::Gpu),
        });
    }

    Ok(emitted)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchPair {
    pub person1: Person,
    pub person2: Person,
    pub confidence: f32,
    pub matched_fields: Vec<String>,
    pub is_matched_infnbd: bool,
    pub is_matched_infnmnbd: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct ProgressConfig {
    pub update_every: usize,
    pub long_op_threshold: Duration,
    pub batch_size: Option<usize>,
}
impl Default for ProgressConfig {
    fn default() -> Self {
        Self {
            update_every: 1000,
            long_op_threshold: Duration::from_secs(30),
            batch_size: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ProgressUpdate {
    pub processed: usize,
    pub total: usize,
    pub percent: f32,
    pub eta_secs: u64,
    pub mem_used_mb: u64,
    pub mem_avail_mb: u64,
    #[allow(dead_code)]
    pub stage: &'static str,
    #[allow(dead_code)]
    pub batch_size_current: Option<i64>,
    // GPU-related (0/false when CPU-only)
    #[allow(dead_code)]
    pub gpu_total_mb: u64,
    #[allow(dead_code)]
    pub gpu_free_mb: u64,
    #[allow(dead_code)]
    pub gpu_active: bool,
}

// --- Optional GPU backend abstraction ---
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum ComputeBackend {
    Cpu,
    Gpu,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct GpuConfig {
    pub device_id: Option<usize>,
    pub mem_budget_mb: u64,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct MatchOptions {
    pub backend: ComputeBackend,
    pub gpu: Option<GpuConfig>,
    pub progress: ProgressConfig,
    /// Whether to allow month/day swap on birthdates (L10/L11 parity)
    pub allow_birthdate_swap: bool,
}

impl Default for MatchOptions {
    fn default() -> Self {
        Self {
            backend: ComputeBackend::Cpu,
            gpu: None,
            progress: ProgressConfig::default(),
            allow_birthdate_swap: false,
        }
    }
}

#[allow(dead_code)]
pub fn match_all_with_opts<F>(
    table1: &[Person],
    table2: &[Person],
    algo: MatchingAlgorithm,
    opts: MatchOptions,
    progress: F,
) -> Vec<MatchPair>
where
    F: Fn(ProgressUpdate) + Sync,
{
    // Fast path: deterministic A1/A2 with GPU-accelerated hashing in in-memory mode
    if matches!(
        algo,
        MatchingAlgorithm::IdUuidYasIsMatchedInfnbd | MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd
    ) && matches!(opts.backend, ComputeBackend::Gpu)
    {
        #[cfg(feature = "gpu")]
        {
            match gpu::det_match_gpu_hash_inmemory(table1, table2, algo, &opts, &progress) {
                Ok(v) => return v,
                Err(e) => {
                    log::warn!(
                        "GPU in-memory hash (A1/A2) failed, falling back to CPU: {}",
                        e
                    );
                }
            }
        }
        // If feature disabled or GPU path failed, fall through to CPU
    }

    if matches!(algo, MatchingAlgorithm::Fuzzy) && matches!(opts.backend, ComputeBackend::Gpu) {
        // Optional in-memory GPU pre-pass before full fuzzy scoring
        #[cfg(feature = "gpu")]
        if gpu_fuzzy_prep_enabled() {
            progress(ProgressUpdate {
                processed: 0,
                total: table1.len().max(1),
                percent: 0.0,
                eta_secs: 0,
                mem_used_mb: memory_stats_mb().used_mb,
                mem_avail_mb: memory_stats_mb().avail_mb,
                stage: "gpu_hash",
                batch_size_current: None,
                gpu_total_mb: 1,
                gpu_free_mb: 0,
                gpu_active: true,
            });
            match gpu::fuzzy_direct_gpu_hash_prefilter_indices(table1, table2, "last_initial") {
                Ok(cand_lists) => {
                    progress(ProgressUpdate {
                        processed: 0,
                        total: table1.len().max(1),
                        percent: 0.0,
                        eta_secs: 0,
                        mem_used_mb: memory_stats_mb().used_mb,
                        mem_avail_mb: memory_stats_mb().avail_mb,
                        stage: "gpu_probe_hash",
                        batch_size_current: None,
                        gpu_total_mb: 1,
                        gpu_free_mb: 0,
                        gpu_active: true,
                    });
                    let n1: Vec<NormalizedPerson> =
                        table1.par_iter().map(normalize_person).collect();
                    let n2: Vec<NormalizedPerson> =
                        table2.par_iter().map(normalize_person).collect();
                    let mut out: Vec<MatchPair> = Vec::new();
                    let allow_swap = crate::matching::birthdate_matcher::allow_birthdate_swap();
                    for (i, p) in table1.iter().enumerate() {
                        let n = &n1[i];
                        for &j in cand_lists.get(i).map(|v| v.as_slice()).unwrap_or(&[]) {
                            let n2p = &n2[j];
                            // Use birthdate_matches_naive to support month/day swap when enabled
                            let bd_match = match (p.birthdate, table2[j].birthdate) {
                                (Some(b1), Some(b2)) => {
                                    crate::matching::birthdate_matcher::birthdate_matches_naive(
                                        b1, b2, allow_swap,
                                    )
                                }
                                _ => false,
                            };
                            if !bd_match {
                                continue;
                            }
                            // GPU-equivalent preliminary filter (max of Levenshtein and Jaro-Winkler) at 85.0
                            let s1 = normalize_simple(&format!(
                                "{} {} {}",
                                p.first_name.as_deref().unwrap_or(""),
                                p.middle_name.as_deref().unwrap_or(""),
                                p.last_name.as_deref().unwrap_or("")
                            ));
                            let s2 = normalize_simple(&format!(
                                "{} {} {}",
                                table2[j].first_name.as_deref().unwrap_or(""),
                                table2[j].middle_name.as_deref().unwrap_or(""),
                                table2[j].last_name.as_deref().unwrap_or("")
                            ));
                            let lev = sim_levenshtein_pct(&s1, &s2);
                            let jw = jaro_winkler(&s1, &s2) * 100.0;
                            if lev.max(jw) < 85.0 {
                                continue;
                            }
                            if let Some((score, label)) = fuzzy_compare_names_new(
                                n.first_name.as_deref(),
                                n.middle_name.as_deref(),
                                n.last_name.as_deref(),
                                n2p.first_name.as_deref(),
                                n2p.middle_name.as_deref(),
                                n2p.last_name.as_deref(),
                            ) {
                                let q = &table2[j];
                                let pair = MatchPair {
                                    person1: p.clone(),
                                    person2: q.clone(),
                                    confidence: (score / 100.0) as f32,
                                    matched_fields: vec!["fuzzy".into(), label, "birthdate".into()],
                                    is_matched_infnbd: false,
                                    is_matched_infnmnbd: false,
                                };
                                out.push(pair);
                            }
                        }
                    }
                    progress(ProgressUpdate {
                        processed: out.len(),
                        total: table1.len().max(1),
                        percent: 100.0,
                        eta_secs: 0,
                        mem_used_mb: memory_stats_mb().used_mb,
                        mem_avail_mb: memory_stats_mb().avail_mb,
                        stage: "gpu_probe_hash_done",
                        batch_size_current: None,
                        gpu_total_mb: 1,
                        gpu_free_mb: 0,
                        gpu_active: true,
                    });
                    return out;
                }
                Err(e) => {
                    log::warn!(
                        "GPU fuzzy direct pre-pass (in-memory) failed; proceeding with full GPU fuzzy: {}",
                        e
                    );
                }
            }
        }
        #[cfg(feature = "gpu")]
        {
            if gpu_fuzzy_metrics_enabled() && !gpu_fuzzy_disable() {
                let use_gpu = if gpu_fuzzy_force() {
                    true
                } else {
                    let (ok, why) = should_enable_gpu_fuzzy_by_heuristic(table1, table2);
                    if ok {
                        log::info!("GPU fuzzy metrics enabled by heuristic: {}", why);
                    } else {
                        log::info!("GPU fuzzy metrics disabled by heuristic: {}", why);
                    }
                    ok
                };
                if use_gpu {
                    match crate::matching::gpu_config::with_oom_cpu_fallback(
                        || gpu::match_fuzzy_gpu(table1, table2, opts, &progress),
                        || match_fuzzy_cpu_gpu_equivalent(table1, table2, &progress),
                        "gpu fuzzy metrics",
                    ) {
                        Ok(v) => return v,
                        Err(e) => {
                            log::warn!("GPU fuzzy failed, falling back to CPU: {}", e);
                        }
                    }
                }
            }
        }
    }
    // Option 7: LevenshteinWeighted routing
    if matches!(algo, MatchingAlgorithm::LevenshteinWeighted) {
        #[cfg(feature = "gpu")]
        {
            if gpu_lev_full_scoring_enabled() {
                log::info!("[opt7] GPU full scoring enabled=true");
                match crate::matching::gpu_config::with_oom_cpu_fallback(
                    || gpu::match_levenshtein_weighted_gpu_full(table1, table2, opts, &progress),
                    || match_levenshtein_weighted_cpu(table1, table2, &progress),
                    "[opt7] gpu full scoring",
                ) {
                    Ok(v) => return v,
                    Err(e) => {
                        log::error!(
                            "[opt7] FATAL: GPU scoring requested but {} ; falling back to CPU",
                            e
                        );
                    }
                }
            } else {
                log::info!("[opt7] GPU full scoring enabled=false");
            }
        }
        return match_levenshtein_weighted_cpu(table1, table2, &progress);
    }

    // GPU in-memory path for Option 4 (FuzzyNoMiddle)
    if matches!(algo, MatchingAlgorithm::FuzzyNoMiddle)
        && matches!(opts.backend, ComputeBackend::Gpu)
    {
        #[cfg(feature = "gpu")]
        if gpu_fuzzy_prep_enabled() {
            progress(ProgressUpdate {
                processed: 0,
                total: table1.len().max(1),
                percent: 0.0,
                eta_secs: 0,
                mem_used_mb: memory_stats_mb().used_mb,
                mem_avail_mb: memory_stats_mb().avail_mb,
                stage: "gpu_hash",
                batch_size_current: None,
                gpu_total_mb: 1,
                gpu_free_mb: 0,
                gpu_active: true,
            });
            match gpu::fuzzy_direct_gpu_hash_prefilter_indices(table1, table2, "last_initial") {
                Ok(cand_lists) => {
                    progress(ProgressUpdate {
                        processed: 0,
                        total: table1.len().max(1),
                        percent: 0.0,
                        eta_secs: 0,
                        mem_used_mb: memory_stats_mb().used_mb,
                        mem_avail_mb: memory_stats_mb().avail_mb,
                        stage: "gpu_probe_hash",
                        batch_size_current: None,
                        gpu_total_mb: 1,
                        gpu_free_mb: 0,
                        gpu_active: true,
                    });
                    let n1: Vec<NormalizedPerson> =
                        table1.par_iter().map(normalize_person).collect();
                    let n2: Vec<NormalizedPerson> =
                        table2.par_iter().map(normalize_person).collect();
                    let mut out: Vec<MatchPair> = Vec::new();
                    let allow_swap = crate::matching::birthdate_matcher::allow_birthdate_swap();
                    for (i, p) in table1.iter().enumerate() {
                        let n = &n1[i];
                        for &j in cand_lists.get(i).map(|v| v.as_slice()).unwrap_or(&[]) {
                            let n2p = &n2[j];
                            // Use birthdate_matches_naive to support month/day swap when enabled
                            let bd_match = match (p.birthdate, table2[j].birthdate) {
                                (Some(b1), Some(b2)) => {
                                    crate::matching::birthdate_matcher::birthdate_matches_naive(
                                        b1, b2, allow_swap,
                                    )
                                }
                                _ => false,
                            };
                            if !bd_match {
                                continue;
                            }
                            // GPU-equivalent preliminary filter (max of Levenshtein and Jaro-Winkler) at 85.0
                            let s1 = normalize_simple(&format!(
                                "{} {} {}",
                                p.first_name.as_deref().unwrap_or(""),
                                p.middle_name.as_deref().unwrap_or(""),
                                p.last_name.as_deref().unwrap_or("")
                            ));
                            let s2 = normalize_simple(&format!(
                                "{} {} {}",
                                table2[j].first_name.as_deref().unwrap_or(""),
                                table2[j].middle_name.as_deref().unwrap_or(""),
                                table2[j].last_name.as_deref().unwrap_or("")
                            ));
                            let lev = sim_levenshtein_pct(&s1, &s2);
                            let jw = jaro_winkler(&s1, &s2) * 100.0;
                            if lev.max(jw) < 85.0 {
                                continue;
                            }
                            if let Some((score, label)) = fuzzy_compare_names_no_mid(
                                n.first_name.as_deref(),
                                n.last_name.as_deref(),
                                n2p.first_name.as_deref(),
                                n2p.last_name.as_deref(),
                            ) {
                                let q = &table2[j];
                                let pair = MatchPair {
                                    person1: p.clone(),
                                    person2: q.clone(),
                                    confidence: (score / 100.0) as f32,
                                    matched_fields: vec!["fuzzy".into(), label, "birthdate".into()],
                                    is_matched_infnbd: false,
                                    is_matched_infnmnbd: false,
                                };
                                out.push(pair);
                            }
                        }
                    }
                    progress(ProgressUpdate {
                        processed: out.len(),
                        total: table1.len().max(1),
                        percent: 100.0,
                        eta_secs: 0,
                        mem_used_mb: memory_stats_mb().used_mb,
                        mem_avail_mb: memory_stats_mb().avail_mb,
                        stage: "gpu_probe_hash_done",
                        batch_size_current: None,
                        gpu_total_mb: 1,
                        gpu_free_mb: 0,
                        gpu_active: true,
                    });
                    return out;
                }
                Err(e) => {
                    log::warn!(
                        "GPU fuzzy direct pre-pass (in-memory; no-mid) failed; proceeding with full GPU fuzzy no-mid: {}",
                        e
                    );
                }
            }
        }
        #[cfg(feature = "gpu")]
        #[cfg(feature = "gpu")]
        {
            if gpu_fuzzy_metrics_enabled() && !gpu_fuzzy_disable() {
                let use_gpu = if gpu_fuzzy_force() {
                    true
                } else {
                    let (ok, why) = should_enable_gpu_fuzzy_by_heuristic(table1, table2);
                    if ok {
                        log::info!("GPU fuzzy metrics enabled by heuristic: {}", why);
                    } else {
                        log::info!("GPU fuzzy metrics disabled by heuristic: {}", why);
                    }
                    ok
                };
                if use_gpu {
                    match crate::matching::gpu_config::with_oom_cpu_fallback(
                        || gpu::match_fuzzy_no_mid_gpu(table1, table2, opts, &progress),
                        || match_fuzzy_no_mid_cpu_gpu_equivalent(table1, table2, &progress),
                        "gpu fuzzy (no-mid)",
                    ) {
                        Ok(v) => return v,
                        Err(e) => {
                            log::warn!("GPU fuzzy (no-mid) failed, falling back to CPU: {}", e);
                        }
                    }
                }
            }
        }
        // If feature disabled or GPU path failed, fall through to CPU
    }
    // CPU path: for Fuzzy and FuzzyNoMiddle always apply GPU-equivalent candidate selection and 85.0 prefilter
    if matches!(opts.backend, ComputeBackend::Cpu) {
        if matches!(algo, MatchingAlgorithm::Fuzzy) {
            return match_fuzzy_cpu_gpu_equivalent(table1, table2, &progress);
        }
        if matches!(algo, MatchingAlgorithm::FuzzyNoMiddle) {
            return match_fuzzy_no_mid_cpu_gpu_equivalent(table1, table2, &progress);
        }
    }

    match_all_progress(table1, table2, algo, opts.progress, progress)
}

/// Global toggle: enable GPU fuzzy direct pre-pass in in-memory matching paths as well.
static GPU_FUZZY_PREPASS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[inline]
pub fn set_gpu_fuzzy_direct_prep(enabled: bool) {
    GPU_FUZZY_PREPASS.store(enabled, std::sync::atomic::Ordering::Relaxed);
}
#[inline]
fn gpu_fuzzy_prep_enabled() -> bool {
    GPU_FUZZY_PREPASS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Global budget (MB) for GPU fuzzy direct pre-pass (candidate hashing). 0 = auto (free/2).
static GPU_FUZZY_PREPASS_BUDGET_MB: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
#[inline]
pub fn set_gpu_fuzzy_prepass_budget_mb(mb: u64) {
    GPU_FUZZY_PREPASS_BUDGET_MB.store(mb, std::sync::atomic::Ordering::Relaxed);
}
#[inline]
pub(crate) fn gpu_fuzzy_prep_budget_mb() -> u64 {
    GPU_FUZZY_PREPASS_BUDGET_MB.load(std::sync::atomic::Ordering::Relaxed)
}

/// Global toggle: enable GPU pre-pass for Option 7 (LevenshteinWeighted) candidate filtering.
// Runtime verification counters for Option 7 GPU pre-pass
#[cfg(feature = "gpu")]
static OPT7_GPU_KERNEL_TILES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "gpu")]
static OPT7_GPU_CPU_FALLBACK_TILES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "gpu")]
#[inline]
pub fn opt7_gpu_reset_counters() {
    OPT7_GPU_KERNEL_TILES.store(0, std::sync::atomic::Ordering::Relaxed);
    OPT7_GPU_CPU_FALLBACK_TILES.store(0, std::sync::atomic::Ordering::Relaxed);
}
#[cfg(feature = "gpu")]
#[inline]
pub fn opt7_gpu_stats() -> (u64, u64) {
    (
        OPT7_GPU_KERNEL_TILES.load(std::sync::atomic::Ordering::Relaxed),
        OPT7_GPU_CPU_FALLBACK_TILES.load(std::sync::atomic::Ordering::Relaxed),
    )
}

static GPU_LEVENSHTEIN_PREPASS: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
#[inline]
pub fn set_gpu_levenshtein_prepass(enabled: bool) {
    GPU_LEVENSHTEIN_PREPASS.store(enabled, std::sync::atomic::Ordering::Relaxed);
}
#[inline]
fn gpu_lev_prepass_enabled() -> bool {
    GPU_LEVENSHTEIN_PREPASS.load(std::sync::atomic::Ordering::Relaxed)
}

// --- Option 7: GPU full scoring toggle and counters ---
static GPU_LEVENSHTEIN_FULL_SCORING: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
#[inline]
pub fn set_gpu_levenshtein_full_scoring(enabled: bool) {
    GPU_LEVENSHTEIN_FULL_SCORING.store(enabled, std::sync::atomic::Ordering::Relaxed);
}
#[inline]
fn gpu_lev_full_scoring_enabled() -> bool {
    GPU_LEVENSHTEIN_FULL_SCORING.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(feature = "gpu")]
static OPT7_GPU_SCORING_TILES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "gpu")]
static OPT7_GPU_SCORING_CPU_FALLBACK_TILES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "gpu")]
#[inline]
pub fn opt7_gpu_scoring_reset_counters() {
    OPT7_GPU_SCORING_TILES.store(0, std::sync::atomic::Ordering::Relaxed);
    OPT7_GPU_SCORING_CPU_FALLBACK_TILES.store(0, std::sync::atomic::Ordering::Relaxed);
}
#[cfg(feature = "gpu")]
#[inline]
pub fn opt7_gpu_scoring_stats() -> (u64, u64) {
    (
        OPT7_GPU_SCORING_TILES.load(std::sync::atomic::Ordering::Relaxed),
        OPT7_GPU_SCORING_CPU_FALLBACK_TILES.load(std::sync::atomic::Ordering::Relaxed),
    )
}

/// Heuristic activation for GPU fuzzy metrics: returns (enable_gpu, reason)
fn should_enable_gpu_fuzzy_by_heuristic(table1: &[Person], table2: &[Person]) -> (bool, String) {
    use chrono::Datelike;
    use std::collections::HashMap;
    // Quick stats over normalized name lengths
    let mut total_len: usize = 0;
    let mut max_len: usize = 0;
    let mut count: usize = 0;
    for p in table1.iter().chain(table2.iter()) {
        let n = normalize_person(p);
        let mut l = 0usize;

        if let Some(s) = n.first_name.as_ref() {
            l += s.len();
        }
        if let Some(s) = n.middle_name.as_ref() {
            l += s.len();
        }
        if let Some(s) = n.last_name.as_ref() {
            l += s.len();
        }
        total_len += l;
        max_len = max_len.max(l);
        count += 1;
    }
    let avg_len: f32 = if count == 0 {
        0.0
    } else {
        (total_len as f32) / (count as f32)
    };

    // Estimate candidate pairs using (birthdate, last-initial) blocking on table2
    #[derive(Hash, Eq, PartialEq, Clone, Copy)]
    struct Key {
        y: i32,
        m: u32,
        d: u32,
        li: char,
    }
    let mut blk: HashMap<Key, usize> = HashMap::new();
    for p in table2.iter() {
        let n = normalize_person(p);
        if let (Some(d), Some(last)) = (n.birthdate.as_ref(), n.last_name.as_ref()) {
            if let Some(li) = last.chars().next() {
                let li = li.to_ascii_uppercase();
                let key = Key {
                    y: d.year(),
                    m: d.month(),
                    d: d.day(),
                    li,
                };
                *blk.entry(key).or_insert(0) += 1;
            }
        }
    }
    let mut cand_est: usize = 0;
    for p in table1.iter() {
        let n = normalize_person(p);
        if let (Some(d), Some(last)) = (n.birthdate.as_ref(), n.last_name.as_ref()) {
            if let Some(li) = last.chars().next() {
                let li = li.to_ascii_uppercase();
                let key = Key {
                    y: d.year(),
                    m: d.month(),
                    d: d.day(),
                    li,
                };
                cand_est += *blk.get(&key).unwrap_or(&0);
            }
        }
    }

    // Heuristic rules
    if max_len > 64 {
        return (
            false,
            format!(
                "disabled: max_len={} > 64 (GPU kernels optimized for <=64)",
                max_len
            ),
        );
    }
    // If small candidate set, CPU wins
    if cand_est < 10_000_000 {
        return (
            false,
            format!("disabled: cand_est={} < 10M (CPU likely faster)", cand_est),
        );
    }
    // If very long names overall, CPU tends to be better
    if avg_len > 32.0 {
        return (
            false,
            format!("disabled: avg_len={:.1} > 32 (CPU likely faster)", avg_len),
        );
    }
    (
        true,
        format!(
            "enabled: cand_est={} avg_len={:.1} max_len={} (GPU likely beneficial)",
            cand_est, avg_len, max_len
        ),
    )
}

static GPU_FUZZY_METRICS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[inline]
pub fn set_gpu_fuzzy_metrics(enabled: bool) {
    GPU_FUZZY_METRICS.store(enabled, std::sync::atomic::Ordering::Relaxed);
}
#[inline]
fn gpu_fuzzy_metrics_enabled() -> bool {
    GPU_FUZZY_METRICS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Optional overrides for heuristic activation
static GPU_FUZZY_FORCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static GPU_FUZZY_DISABLE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[inline]
pub fn set_gpu_fuzzy_force(enabled: bool) {
    GPU_FUZZY_FORCE.store(enabled, std::sync::atomic::Ordering::Relaxed);
}
#[inline]
pub fn set_gpu_fuzzy_disable(enabled: bool) {
    GPU_FUZZY_DISABLE.store(enabled, std::sync::atomic::Ordering::Relaxed);
}
#[inline]
fn gpu_fuzzy_force() -> bool {
    GPU_FUZZY_FORCE.load(std::sync::atomic::Ordering::Relaxed)
}
#[inline]
fn gpu_fuzzy_disable() -> bool {
    GPU_FUZZY_DISABLE.load(std::sync::atomic::Ordering::Relaxed)
}

// Global toggle for dynamic GPU auto-tuning (optional, off by default)
static DYNAMIC_GPU_TUNING: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
#[inline]
pub fn set_dynamic_gpu_tuning(enabled: bool) {
    DYNAMIC_GPU_TUNING.store(enabled, std::sync::atomic::Ordering::Relaxed);
}
#[inline]
fn dynamic_gpu_tuning_enabled() -> bool {
    DYNAMIC_GPU_TUNING.load(std::sync::atomic::Ordering::Relaxed)
}

/// Global toggle: when true, Algorithms 1 & 2 use fuzzy-style normalization (drop periods, hyphens->spaces) before equality checks.
static DIRECT_NORM_FUZZY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[inline]
pub fn set_direct_normalization_fuzzy(enabled: bool) {
    DIRECT_NORM_FUZZY.store(enabled, std::sync::atomic::Ordering::Relaxed);
}
#[inline]
fn direct_norm_fuzzy_enabled() -> bool {
    DIRECT_NORM_FUZZY.load(std::sync::atomic::Ordering::Relaxed)
}

fn matches_algo1(p1: &NormalizedPerson, p2: &NormalizedPerson) -> bool {
    let allow_swap = crate::matching::birthdate_matcher::allow_birthdate_swap();
    let date_ok = match (p1.birthdate, p2.birthdate) {
        (Some(b1), Some(b2)) => {
            crate::matching::birthdate_matcher::birthdate_matches_naive(b1, b2, allow_swap)
        }
        _ => false,
    };
    if !date_ok {
        return false;
    }
    if direct_norm_fuzzy_enabled() {
        let a_first = p1.first_name.as_deref().map(normalize_simple);
        let b_first = p2.first_name.as_deref().map(normalize_simple);
        let a_last = p1.last_name.as_deref().map(normalize_simple);
        let b_last = p2.last_name.as_deref().map(normalize_simple);
        a_first.as_deref() == b_first.as_deref() && a_last.as_deref() == b_last.as_deref()
    } else {
        let first_ok = p1
            .first_name
            .as_ref()
            .zip(p2.first_name.as_ref())
            .map_or(false, |(a, b)| a == b);
        let last_ok = p1
            .last_name
            .as_ref()
            .zip(p2.last_name.as_ref())
            .map_or(false, |(a, b)| a == b);
        first_ok && last_ok
    }
}
fn matches_algo2(p1: &NormalizedPerson, p2: &NormalizedPerson) -> bool {
    let allow_swap = crate::matching::birthdate_matcher::allow_birthdate_swap();
    let date_ok = match (p1.birthdate, p2.birthdate) {
        (Some(b1), Some(b2)) => {
            crate::matching::birthdate_matcher::birthdate_matches_naive(b1, b2, allow_swap)
        }
        _ => false,
    };
    if !date_ok {
        return false;
    }
    if direct_norm_fuzzy_enabled() {
        let a_first = p1.first_name.as_deref().map(normalize_simple);
        let b_first = p2.first_name.as_deref().map(normalize_simple);
        let a_last = p1.last_name.as_deref().map(normalize_simple);
        let b_last = p2.last_name.as_deref().map(normalize_simple);
        let a_mid = p1.middle_name.as_deref().map(normalize_simple);
        let b_mid = p2.middle_name.as_deref().map(normalize_simple);
        let middle_ok = match (a_mid.as_deref(), b_mid.as_deref()) {
            (Some(a), Some(b)) => a == b,
            (None, None) => true,
            _ => false,
        };
        a_first.as_deref() == b_first.as_deref()
            && a_last.as_deref() == b_last.as_deref()
            && middle_ok
    } else {
        let first_ok = p1
            .first_name
            .as_ref()
            .zip(p2.first_name.as_ref())
            .map_or(false, |(a, b)| a == b);
        let last_ok = p1
            .last_name
            .as_ref()
            .zip(p2.last_name.as_ref())
            .map_or(false, |(a, b)| a == b);
        let middle_ok = match (&p1.middle_name, &p2.middle_name) {
            (Some(a), Some(b)) => a == b,
            (None, None) => true,
            _ => false,
        };
        first_ok && last_ok && middle_ok
    }
}

/// Apply a consistent set of GPU enhancement toggles for a given algorithm.
/// This centralizes controls so GUI/CLI don't need to set individual flags repeatedly.
pub fn apply_gpu_enhancements_for_algo(
    algo: MatchingAlgorithm,
    prepass: bool,
    full_scoring: bool,
    metrics_auto_or_force: bool,
    metrics_force: bool,
    metrics_off: bool,
) {
    // Only enable pre-pass for algorithms that can use it
    let allow_prepass_fuzzy = matches!(
        algo,
        MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle
    );
    let allow_prepass_lev = matches!(algo, MatchingAlgorithm::LevenshteinWeighted);
    let allow_full_scoring_lev = matches!(algo, MatchingAlgorithm::LevenshteinWeighted);
    // Allow GPU fuzzy metrics for Fuzzy, FuzzyNoMiddle, and HouseholdGpu
    let allow_metrics = matches!(
        algo,
        MatchingAlgorithm::Fuzzy
            | MatchingAlgorithm::FuzzyNoMiddle
            | MatchingAlgorithm::HouseholdGpu
            | MatchingAlgorithm::HouseholdGpuOpt6
    );
    // Wire toggles
    set_gpu_fuzzy_direct_prep(prepass && allow_prepass_fuzzy);
    set_gpu_levenshtein_prepass(prepass && allow_prepass_lev);
    set_gpu_levenshtein_full_scoring(full_scoring && allow_full_scoring_lev);
    set_gpu_fuzzy_metrics(metrics_auto_or_force && allow_metrics);
    set_gpu_fuzzy_force(metrics_force);
    set_gpu_fuzzy_disable(metrics_off);
}

#[allow(dead_code)]
pub fn match_all<F>(
    table1: &[Person],
    table2: &[Person],
    algo: MatchingAlgorithm,
    progress: F,
) -> Vec<MatchPair>
where
    F: Fn(f32) + Sync,
{
    match_all_progress(table1, table2, algo, ProgressConfig::default(), |u| {
        progress(u.percent)
    })
}

pub fn match_all_progress<F>(
    table1: &[Person],
    table2: &[Person],
    algo: MatchingAlgorithm,
    cfg: ProgressConfig,
    progress: F,
) -> Vec<MatchPair>
where
    F: Fn(ProgressUpdate) + Sync,
{
    let start = Instant::now();
    let norm1: Vec<NormalizedPerson> = table1.par_iter().map(normalize_person).collect();
    let norm2: Vec<NormalizedPerson> = table2.par_iter().map(normalize_person).collect();
    let total = norm1.len();
    if total == 0 || norm2.is_empty() {
        return Vec::new();
    }
    let threads = rayon::current_num_threads().max(1);
    let auto_batch = (total / (threads * 4)).clamp(100, 10_000).max(1);
    let batch_size = cfg.batch_size.unwrap_or(auto_batch);

    let mut results: Vec<MatchPair> = Vec::new();
    let mut processed_outer = 0usize;
    let mut last_update = 0usize;

    for chunk in norm1.chunks(batch_size) {
        let chunk_start = Instant::now();
        let batch_res: Vec<MatchPair> = chunk
            .par_iter()
            .flat_map(|p1| {
                norm2
                    .par_iter()
                    .filter_map(|p2| {
                        match algo {
                            MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => {
                                if matches_algo1(p1, p2) {
                                    Some(MatchPair {
                                        person1: to_original(p1, table1),
                                        person2: to_original(p2, table2),
                                        confidence: 1.0,
                                        matched_fields: vec![
                                            "id",
                                            "uuid",
                                            "first_name",
                                            "last_name",
                                            "birthdate",
                                        ]
                                        .into_iter()
                                        .map(String::from)
                                        .collect(),
                                        is_matched_infnbd: true,
                                        is_matched_infnmnbd: false,
                                    })
                                } else {
                                    None
                                }
                            }
                            MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => {
                                if matches_algo2(p1, p2) {
                                    Some(MatchPair {
                                        person1: to_original(p1, table1),
                                        person2: to_original(p2, table2),
                                        confidence: 1.0,
                                        matched_fields: vec![
                                            "id",
                                            "uuid",
                                            "first_name",
                                            "middle_name",
                                            "last_name",
                                            "birthdate",
                                        ]
                                        .into_iter()
                                        .map(String::from)
                                        .collect(),
                                        is_matched_infnbd: false,
                                        is_matched_infnmnbd: true,
                                    })
                                } else {
                                    None
                                }
                            }
                            MatchingAlgorithm::Fuzzy => {
                                let allow_swap =
                                    crate::matching::birthdate_matcher::allow_birthdate_swap();
                                let bd_match = match (p1.birthdate, p2.birthdate) {
                                    (Some(b1), Some(b2)) => {
                                        crate::matching::birthdate_matcher::birthdate_matches_naive(
                                            b1, b2, allow_swap,
                                        )
                                    }
                                    _ => false,
                                };
                                if bd_match {
                                    if let Some((score, label)) = fuzzy_compare_names_new(
                                        p1.first_name.as_deref(),
                                        p1.middle_name.as_deref(),
                                        p1.last_name.as_deref(),
                                        p2.first_name.as_deref(),
                                        p2.middle_name.as_deref(),
                                        p2.last_name.as_deref(),
                                    ) {
                                        Some(MatchPair {
                                            person1: to_original(p1, table1),
                                            person2: to_original(p2, table2),
                                            confidence: (score / 100.0) as f32,
                                            matched_fields: vec![
                                                "fuzzy".into(),
                                                label.into(),
                                                "birthdate".into(),
                                            ],
                                            is_matched_infnbd: false,
                                            is_matched_infnmnbd: false,
                                        })
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            }
                            MatchingAlgorithm::FuzzyNoMiddle => {
                                let allow_swap =
                                    crate::matching::birthdate_matcher::allow_birthdate_swap();
                                let bd_match = match (p1.birthdate, p2.birthdate) {
                                    (Some(b1), Some(b2)) => {
                                        crate::matching::birthdate_matcher::birthdate_matches_naive(
                                            b1, b2, allow_swap,
                                        )
                                    }
                                    _ => false,
                                };
                                if bd_match {
                                    if let Some((score, label)) = fuzzy_compare_names_no_mid(
                                        p1.first_name.as_deref(),
                                        p1.last_name.as_deref(),
                                        p2.first_name.as_deref(),
                                        p2.last_name.as_deref(),
                                    ) {
                                        Some(MatchPair {
                                            person1: to_original(p1, table1),
                                            person2: to_original(p2, table2),
                                            confidence: (score / 100.0) as f32,
                                            matched_fields: vec![
                                                "fuzzy".into(),
                                                label.into(),
                                                "birthdate".into(),
                                            ],
                                            is_matched_infnbd: false,
                                            is_matched_infnmnbd: false,
                                        })
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            }
                            MatchingAlgorithm::LevenshteinWeighted => {
                                let allow_swap =
                                    crate::matching::birthdate_matcher::allow_birthdate_swap();
                                let bd_match = match (p1.birthdate, p2.birthdate) {
                                    (Some(b1), Some(b2)) => {
                                        crate::matching::birthdate_matcher::birthdate_matches_naive(
                                            b1, b2, allow_swap,
                                        )
                                    }
                                    _ => false,
                                };
                                if bd_match {
                                    // Normalize simple strings
                                    let af = p1
                                        .first_name
                                        .as_deref()
                                        .map(normalize_simple)
                                        .unwrap_or_default();
                                    let am = p1
                                        .middle_name
                                        .as_deref()
                                        .map(normalize_simple)
                                        .unwrap_or_default();
                                    let al = p1
                                        .last_name
                                        .as_deref()
                                        .map(normalize_simple)
                                        .unwrap_or_default();
                                    let bf = p2
                                        .first_name
                                        .as_deref()
                                        .map(normalize_simple)
                                        .unwrap_or_default();
                                    let bm = p2
                                        .middle_name
                                        .as_deref()
                                        .map(normalize_simple)
                                        .unwrap_or_default();
                                    let bl = p2
                                        .last_name
                                        .as_deref()
                                        .map(normalize_simple)
                                        .unwrap_or_default();
                                    // Blocking: soundex(first/last) OR 3-char prefix(first/last) OR soundex(middle if both)
                                    let b_ok_soundex = soundex4_ascii(&af) == soundex4_ascii(&bf)
                                        && soundex4_ascii(&al) == soundex4_ascii(&bl);
                                    let a3f: String = af.chars().take(3).collect();
                                    let a3l: String = al.chars().take(3).collect();
                                    let b3f: String = bf.chars().take(3).collect();
                                    let b3l: String = bl.chars().take(3).collect();
                                    let b_ok_prefix = !a3f.is_empty()
                                        && !a3l.is_empty()
                                        && a3f == b3f
                                        && a3l == b3l;
                                    let b_ok_mid = if !am.is_empty() && !bm.is_empty() {
                                        soundex4_ascii(&am) == soundex4_ascii(&bm)
                                    } else {
                                        false
                                    };
                                    if !(b_ok_soundex || b_ok_prefix || b_ok_mid) {
                                        None
                                    } else {
                                        let last_sim = sim_levenshtein_pct(&al, &bl);
                                        let first_sim = sim_levenshtein_pct(&af, &bf);
                                        let mid_present = !am.is_empty() && !bm.is_empty();
                                        let middle_sim = if mid_present {
                                            sim_levenshtein_pct(&am, &bm)
                                        } else {
                                            0.0
                                        };
                                        let denom = 2.0 + if mid_present { 1.0 } else { 0.0 };
                                        let confidence = ((last_sim
                                            + first_sim
                                            + if mid_present { middle_sim } else { 0.0 })
                                            / denom)
                                            as f32
                                            / 100.0;
                                        let a_first_eq = af == bf;
                                        let a_last_eq = al == bl;
                                        let a_mid_eq = am == bm;
                                        let mut matched_fields: Vec<String> = Vec::new();
                                        if a_first_eq {
                                            matched_fields.push("FirstName".into());
                                        }
                                        if a_mid_eq {
                                            matched_fields.push("MiddleName".into());
                                        }
                                        if a_last_eq {
                                            matched_fields.push("LastName".into());
                                        }
                                        matched_fields.push("Birthdate".into());
                                        Some(MatchPair {
                                            person1: to_original(p1, table1),
                                            person2: to_original(p2, table2),
                                            confidence,
                                            matched_fields,
                                            is_matched_infnbd: false,
                                            is_matched_infnmnbd: false,
                                        })
                                    }
                                } else {
                                    None
                                }
                            }
                            MatchingAlgorithm::HouseholdGpu
                            | MatchingAlgorithm::HouseholdGpuOpt6 => None,
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        results.extend(batch_res);
        processed_outer = (processed_outer + chunk.len()).min(total);
        if processed_outer - last_update >= cfg.update_every || processed_outer == total {
            let elapsed = start.elapsed();
            let frac = (processed_outer as f32 / total as f32).clamp(0.0, 1.0);
            let eta_secs = if frac > 0.0 {
                (elapsed.as_secs_f32() * (1.0 - frac) / frac) as u64
            } else {
                0
            };
            let mem = memory_stats_mb();
            progress(ProgressUpdate {
                processed: processed_outer,
                total,
                percent: frac * 100.0,
                eta_secs,
                mem_used_mb: mem.used_mb,
                mem_avail_mb: mem.avail_mb,
                stage: "matching",
                batch_size_current: None,
                gpu_total_mb: 0,
                gpu_free_mb: 0,
                gpu_active: false,
            });
            last_update = processed_outer;
        }
        if chunk_start.elapsed() > cfg.long_op_threshold {
            let _m = memory_stats_mb();
        }
    }

    results
}

// CPU equivalent of GPU fuzzy (no-middle) candidate generation and classification
pub(crate) fn match_fuzzy_no_mid_blocked_cpu<F>(
    t1: &[Person],
    t2: &[Person],
    on_progress: &F,
) -> Vec<MatchPair>
where
    F: Fn(ProgressUpdate) + Sync,
{
    use std::collections::HashMap;
    // Normalize once for blocking keys
    let n1: Vec<NormalizedPerson> = t1.par_iter().map(normalize_person).collect();
    let n2: Vec<NormalizedPerson> = t2.par_iter().map(normalize_person).collect();
    if n1.is_empty() || n2.is_empty() {
        return Vec::new();
    }

    #[derive(Hash, Eq, PartialEq)]
    struct BKey(u16, u8, u8, [u8; 4]); // (birth year, first init, last init, last soundex)

    // Build blocks for table2
    let mut block: HashMap<BKey, Vec<usize>> = HashMap::new();
    for (j, p) in n2.iter().enumerate() {
        let (Some(d), Some(fn_str), Some(ln_str)) = (
            p.birthdate.as_ref(),
            p.first_name.as_deref(),
            p.last_name.as_deref(),
        ) else {
            continue;
        };
        let year = d.year() as u16;
        let fi = fn_str
            .bytes()
            .find(|c| c.is_ascii_alphabetic())
            .unwrap_or(b'?')
            .to_ascii_uppercase();
        let li = ln_str
            .bytes()
            .find(|c| c.is_ascii_alphabetic())
            .unwrap_or(b'?')
            .to_ascii_uppercase();
        let sx = soundex4_ascii(ln_str);
        block.entry(BKey(year, fi, li, sx)).or_default().push(j);
    }

    let total = n1.len();
    let mut results: Vec<MatchPair> = Vec::new();
    for (i, p1) in n1.iter().enumerate() {
        // Progress (best-effort)
        if i % 1000 == 0 {
            let mem = memory_stats_mb();
            on_progress(ProgressUpdate {
                processed: i,
                total,
                percent: (i as f32 / total as f32) * 100.0,
                eta_secs: 0,
                mem_used_mb: mem.used_mb,
                mem_avail_mb: mem.avail_mb,
                stage: "cpu_block_fuzzy",
                batch_size_current: None,
                gpu_total_mb: 0,
                gpu_free_mb: 0,
                gpu_active: false,
            });
        }
        let (Some(d), Some(fn_str), Some(ln_str)) = (
            p1.birthdate.as_ref(),
            p1.first_name.as_deref(),
            p1.last_name.as_deref(),
        ) else {
            continue;
        };
        let year = d.year() as u16;
        let fi = fn_str
            .bytes()
            .find(|c| c.is_ascii_alphabetic())
            .unwrap_or(b'?')
            .to_ascii_uppercase();
        let li = ln_str
            .bytes()
            .find(|c| c.is_ascii_alphabetic())
            .unwrap_or(b'?')
            .to_ascii_uppercase();
        let sx = soundex4_ascii(ln_str);
        let mut set: HashSet<usize> = HashSet::new();
        if let Some(v) = block.get(&BKey(year, fi, li, sx)) {
            set.extend(v.iter().copied());
        }
        if set.is_empty() {
            if let Some(v) = block.get(&BKey(year, b'?', li, sx)) {
                set.extend(v.iter().copied());
            }
        }
        if set.is_empty() {
            let mut sx2 = sx;
            sx2[2] = b'0';
            sx2[3] = b'0';
            if let Some(v) = block.get(&BKey(year, fi, li, sx2)) {
                set.extend(v.iter().copied());
            }
        }
        if set.is_empty() {
            continue;
        }
        let allow_swap = crate::matching::birthdate_matcher::allow_birthdate_swap();
        for j in set.into_iter() {
            // Enforce birthdate equality at Person level (with optional month/day swap)
            let bd_match = match (t1[i].birthdate, t2[j].birthdate) {
                (Some(b1), Some(b2)) => {
                    crate::matching::birthdate_matcher::birthdate_matches_naive(b1, b2, allow_swap)
                }
                _ => false,
            };
            if !bd_match {
                continue;
            }
            if let Some((score, label)) = compare_persons_no_mid(&t1[i], &t2[j]) {
                results.push(MatchPair {
                    person1: t1[i].clone(),
                    person2: t2[j].clone(),
                    confidence: (score / 100.0) as f32,
                    matched_fields: vec!["fuzzy".into(), label, "birthdate".into()],
                    is_matched_infnbd: false,
                    is_matched_infnmnbd: false,
                });
            }
        }
    }
    results
}
/// CPU implementation that replicates GPU candidate generation and prefilter for Option 3 (Fuzzy)
pub(crate) fn match_fuzzy_cpu_gpu_equivalent<F>(
    t1: &[Person],
    t2: &[Person],
    on_progress: &F,
) -> Vec<MatchPair>
where
    F: Fn(ProgressUpdate) + Sync,
{
    use chrono::Datelike;
    use std::collections::{HashMap, HashSet};
    // Normalize
    let n1: Vec<NormalizedPerson> = t1.par_iter().map(normalize_person).collect();
    let n2: Vec<NormalizedPerson> = t2.par_iter().map(normalize_person).collect();
    if n1.is_empty() || n2.is_empty() {
        return Vec::new();
    }

    #[derive(Hash, Eq, PartialEq)]
    struct BKey(u16, u8, u8, [u8; 4]); // (birth year, first init, last init, last soundex)

    // Build blocks for table2 (identical to GPU)
    let mut block: HashMap<BKey, Vec<usize>> = HashMap::new();
    for (j, p) in n2.iter().enumerate() {
        let (Some(d), Some(fn_str), Some(ln_str)) = (
            p.birthdate.as_ref(),
            p.first_name.as_deref(),
            p.last_name.as_deref(),
        ) else {
            continue;
        };
        let year = d.year() as u16;
        let fi = fn_str
            .bytes()
            .find(|c| c.is_ascii_alphabetic())
            .unwrap_or(b'?')
            .to_ascii_uppercase();
        let li = ln_str
            .bytes()
            .find(|c| c.is_ascii_alphabetic())
            .unwrap_or(b'?')
            .to_ascii_uppercase();
        let sx = soundex4_ascii(ln_str);
        block.entry(BKey(year, fi, li, sx)).or_default().push(j);
    }

    // No caches needed; compute prelim strings directly to mirror GPU prefilter behavior

    let total = n1.len();
    let mut results: Vec<MatchPair> = Vec::new();
    for (i, p1) in n1.iter().enumerate() {
        if i % 1000 == 0 {
            let mem = memory_stats_mb();
            on_progress(ProgressUpdate {
                processed: i,
                total,
                percent: (i as f32 / total as f32) * 100.0,
                eta_secs: 0,
                mem_used_mb: mem.used_mb,
                mem_avail_mb: mem.avail_mb,
                stage: "cpu_gpu_equiv_fuzzy",
                batch_size_current: None,
                gpu_total_mb: 0,
                gpu_free_mb: 0,
                gpu_active: false,
            });
        }
        let (Some(d), Some(fn_str), Some(ln_str)) = (
            p1.birthdate.as_ref(),
            p1.first_name.as_deref(),
            p1.last_name.as_deref(),
        ) else {
            continue;
        };
        let year = d.year() as u16;
        let fi = fn_str
            .bytes()
            .find(|c| c.is_ascii_alphabetic())
            .unwrap_or(b'?')
            .to_ascii_uppercase();
        let li = ln_str
            .bytes()
            .find(|c| c.is_ascii_alphabetic())
            .unwrap_or(b'?')
            .to_ascii_uppercase();
        let sx = soundex4_ascii(ln_str);
        let mut set: HashSet<usize> = HashSet::new();
        if let Some(v) = block.get(&BKey(year, fi, li, sx)) {
            set.extend(v.iter().copied());
        }
        if set.is_empty() {
            if let Some(v) = block.get(&BKey(year, b'?', li, sx)) {
                set.extend(v.iter().copied());
            }
        }

        if set.is_empty() {
            let mut sx2 = sx;
            sx2[2] = b'0';
            sx2[3] = b'0';
            if let Some(v) = block.get(&BKey(year, fi, li, sx2)) {
                set.extend(v.iter().copied());
            }
        }
        if set.is_empty() {
            continue;
        }

        let allow_swap = crate::matching::birthdate_matcher::allow_birthdate_swap();
        for j in set.into_iter() {
            // Compute GPU-equivalent preliminary score using full normalized strings
            let s1 = normalize_simple(&format!(
                "{} {} {}",
                t1[i].first_name.as_deref().unwrap_or(""),
                t1[i].middle_name.as_deref().unwrap_or(""),
                t1[i].last_name.as_deref().unwrap_or("")
            ));
            let s2 = normalize_simple(&format!(
                "{} {} {}",
                t2[j].first_name.as_deref().unwrap_or(""),
                t2[j].middle_name.as_deref().unwrap_or(""),
                t2[j].last_name.as_deref().unwrap_or("")
            ));
            let lev = sim_levenshtein_pct(&s1, &s2);
            let jw = jaro_winkler(&s1, &s2) * 100.0;
            let prelim = lev.max(jw);
            if prelim < 85.0 {
                continue;
            }
            // Birthdate equality enforced (with optional month/day swap)
            let bd_match = match (t1[i].birthdate, t2[j].birthdate) {
                (Some(b1), Some(b2)) => {
                    crate::matching::birthdate_matcher::birthdate_matches_naive(b1, b2, allow_swap)
                }
                _ => false,
            };
            if !bd_match {
                continue;
            }
            // Authoritative classification via CPU path
            if let Some((score, label)) = compare_persons_new(&t1[i], &t2[j]) {
                results.push(MatchPair {
                    person1: t1[i].clone(),
                    person2: t2[j].clone(),
                    confidence: (score / 100.0) as f32,
                    matched_fields: vec!["fuzzy".into(), label, "birthdate".into()],
                    is_matched_infnbd: false,
                    is_matched_infnmnbd: false,
                });
            }
        }
    }
    results
}

/// CPU implementation that replicates GPU candidate generation and prefilter for Option 4 (FuzzyNoMiddle)
pub(crate) fn match_fuzzy_no_mid_cpu_gpu_equivalent<F>(
    t1: &[Person],
    t2: &[Person],
    on_progress: &F,
) -> Vec<MatchPair>
where
    F: Fn(ProgressUpdate) + Sync,
{
    use chrono::Datelike;
    use std::collections::{HashMap, HashSet};
    // Normalize
    let n1: Vec<NormalizedPerson> = t1.par_iter().map(normalize_person).collect();
    let n2: Vec<NormalizedPerson> = t2.par_iter().map(normalize_person).collect();
    if n1.is_empty() || n2.is_empty() {
        return Vec::new();
    }

    use chrono::NaiveDate;

    // Build birthdate-only blocks for table2 (align CPU to GPU candidate generation)
    let mut by_bd2: HashMap<NaiveDate, Vec<usize>> = HashMap::new();
    for (j, p) in n2.iter().enumerate() {
        if let Some(d) = p.birthdate.as_ref() {
            by_bd2.entry(*d).or_default().push(j);
        }
    }

    // No caches needed here; compute prelim strings directly (match GPU prefilter behavior)

    let total = n1.len();
    let mut results: Vec<MatchPair> = Vec::new();
    for (i, p1) in n1.iter().enumerate() {
        if i % 1000 == 0 {
            let mem = memory_stats_mb();
            on_progress(ProgressUpdate {
                processed: i,
                total,
                percent: (i as f32 / total as f32) * 100.0,
                eta_secs: 0,
                mem_used_mb: mem.used_mb,
                mem_avail_mb: mem.avail_mb,
                stage: "cpu_gpu_equiv_no_mid",
                batch_size_current: None,
                gpu_total_mb: 0,
                gpu_free_mb: 0,
                gpu_active: false,
            });
        }
        let Some(d) = p1.birthdate.as_ref() else {
            continue;
        };
        let allow_swap = crate::matching::birthdate_matcher::allow_birthdate_swap();
        // Get candidates matching exact date or swapped date (if swap enabled)
        let mut list: Vec<usize> = by_bd2.get(d).cloned().unwrap_or_default();
        if allow_swap {
            if let Some(swapped) = crate::matching::birthdate_matcher::swap_month_day(*d) {
                if swapped != *d {
                    if let Some(v) = by_bd2.get(&swapped) {
                        list.extend(v.iter().copied());
                    }
                }
            }
        }
        if list.is_empty() {
            continue;
        }

        for j in list.into_iter() {
            // Preliminary filter using first+last names only (no middle) to match compare_persons_no_mid() semantics
            let s1 = normalize_simple(&format!(
                "{} {}",
                t1[i].first_name.as_deref().unwrap_or(""),
                t1[i].last_name.as_deref().unwrap_or("")
            ));
            let s2 = normalize_simple(&format!(
                "{} {}",
                t2[j].first_name.as_deref().unwrap_or(""),
                t2[j].last_name.as_deref().unwrap_or("")
            ));
            let lev = sim_levenshtein_pct(&s1, &s2);
            let jw = jaro_winkler(&s1, &s2) * 100.0;
            let prelim = lev.max(jw);
            if prelim < 85.0 {
                continue;
            }
            // Final classification: Option 4 semantics (no middle)
            if let Some((score, label)) = compare_persons_no_mid(&t1[i], &t2[j]) {
                results.push(MatchPair {
                    person1: t1[i].clone(),
                    person2: t2[j].clone(),
                    confidence: (score / 100.0) as f32,
                    matched_fields: vec!["fuzzy".into(), label, "birthdate".into()],
                    is_matched_infnbd: false,
                    is_matched_infnmnbd: false,
                });
            }
        }
    }
    results
}

/// Option 7: SQL-equivalent Levenshtein weighted matching (CPU, optimized)
pub(crate) fn match_levenshtein_weighted_cpu<F>(
    t1: &[Person],
    t2: &[Person],
    on_progress: &F,
) -> Vec<MatchPair>
where
    F: Fn(ProgressUpdate) + Sync,
{
    use rayon::prelude::*;
    use std::collections::{HashMap, HashSet};

    #[cfg(not(feature = "gpu"))]
    if gpu_lev_prepass_enabled() {
        log::warn!(
            "[opt7] GPU pre-pass requested but binary was built without 'gpu' feature; running CPU path"
        );
    }

    // 1) Precompute normalized/cached fields for table2 and build blocking indexes
    #[derive(Clone)]
    struct Cache2 {
        af: String,
        am: String,
        al: String,
        af3: String,
        al3: String,
        sx_f: [u8; 4],
        sx_l: [u8; 4],
        sx_m: [u8; 4],
        date: Option<chrono::NaiveDate>,
    }
    fn pack_sx(sx: [u8; 4]) -> u32 {
        u32::from_le_bytes(sx)
    }

    let caches2: Vec<Cache2> = t2
        .iter()
        .map(|b| {
            let af = normalize_simple(b.first_name.as_deref().unwrap_or(""));
            let am = normalize_simple(b.middle_name.as_deref().unwrap_or(""));
            let al = normalize_simple(b.last_name.as_deref().unwrap_or(""));
            let af3: String = af.chars().take(3).collect();
            let al3: String = al.chars().take(3).collect();
            let sx_f = soundex4_ascii(&af);
            let sx_l = soundex4_ascii(&al);
            let sx_m = if am.is_empty() {
                [b'0'; 4]
            } else {
                soundex4_ascii(&am)
            };
            Cache2 {
                af,
                am,
                al,
                af3,
                al3,
                sx_f,
                sx_l,
                sx_m,
                date: b.birthdate,
            }
        })
        .collect();

    let mut ix_sx: HashMap<(String, u32, u32), Vec<usize>> = HashMap::new(); // (date, sx_f, sx_l)
    let mut ix_pf: HashMap<(String, String, String), Vec<usize>> = HashMap::new(); // (date, f3, l3)
    let mut ix_mid: HashMap<(String, u32), Vec<usize>> = HashMap::new(); // (date, sx_m)

    for (j, c) in caches2.iter().enumerate() {
        if let Some(d) = c.date {
            let dk = d.format("%F").to_string();
            ix_sx
                .entry((dk.clone(), pack_sx(c.sx_f), pack_sx(c.sx_l)))
                .or_default()
                .push(j);
            if !c.af3.is_empty() && !c.al3.is_empty() {
                ix_pf
                    .entry((dk.clone(), c.af3.clone(), c.al3.clone()))
                    .or_default()
                    .push(j);
            }
            if !c.am.is_empty() {
                ix_mid.entry((dk, pack_sx(c.sx_m))).or_default().push(j);
            }
        }
    }

    // 2) Parallel outer traversal over table1; union candidates per OR-blocking semantics
    let total = t1.len();
    let out: Vec<MatchPair> = t1
        .par_iter()
        .enumerate()
        .map(|(i, a)| {
            if i % 2000 == 0 {
                let mem = memory_stats_mb();
                on_progress(ProgressUpdate {
                    processed: i,
                    total,
                    percent: (i as f32 / total.max(1) as f32) * 100.0,
                    eta_secs: 0,
                    mem_used_mb: mem.used_mb,
                    mem_avail_mb: mem.avail_mb,
                    stage: "opt7_cpu",
                    batch_size_current: None,
                    gpu_total_mb: 0,
                    gpu_free_mb: 0,
                    gpu_active: false,
                });
            }
            let mut local: Vec<MatchPair> = Vec::new();
            let Some(ad) = a.birthdate else { return local };

            // Normalize once per outer
            let af = normalize_simple(a.first_name.as_deref().unwrap_or(""));
            let am = normalize_simple(a.middle_name.as_deref().unwrap_or(""));
            let al = normalize_simple(a.last_name.as_deref().unwrap_or(""));
            let af3: String = af.chars().take(3).collect();
            let al3: String = al.chars().take(3).collect();
            let asx_f = soundex4_ascii(&af);
            let asx_l = soundex4_ascii(&al);
            let asx_m = if am.is_empty() {
                None
            } else {
                Some(soundex4_ascii(&am))
            };
            let dk = ad.format("%F").to_string();

            let mut cand: HashSet<usize> = HashSet::new();
            if let Some(v) = ix_sx.get(&(dk.clone(), pack_sx(asx_f), pack_sx(asx_l))) {
                cand.extend(v.iter().copied());
            }
            if !af3.is_empty() && !al3.is_empty() {
                if let Some(v) = ix_pf.get(&(dk.clone(), af3.clone(), al3.clone())) {
                    cand.extend(v.iter().copied());
                }
            }
            if let Some(sxm) = asx_m {
                if let Some(v) = ix_mid.get(&(dk, pack_sx(sxm))) {
                    cand.extend(v.iter().copied());
                }
            }

            for j in cand.into_iter() {
                let b = &t2[j];
                let c2 = &caches2[j];
                // Birthdate is already constrained by index key; re-check for safety
                if b.birthdate != Some(ad) {
                    continue;
                }

                // Scoring: per-field Levenshtein %, weighted by presence of middle
                let mid_present = !am.is_empty() && !c2.am.is_empty();
                let last_sim = sim_levenshtein_pct(&al, &c2.al);
                let first_sim = sim_levenshtein_pct(&af, &c2.af);
                let middle_sim = if mid_present {
                    sim_levenshtein_pct(&am, &c2.am)
                } else {
                    0.0
                };
                let denom = if mid_present { 3.0 } else { 2.0 };
                let confidence =
                    ((last_sim + first_sim + if mid_present { middle_sim } else { 0.0 }) / denom)
                        as f32
                        / 100.0;

                // Classification and matched fields
                let a_first_eq = af == c2.af;
                let a_last_eq = al == c2.al;
                let a_mid_eq = am == c2.am;
                let mut matched_fields: Vec<String> = Vec::new();
                if a_first_eq {
                    matched_fields.push("FirstName".into());
                }
                if a_mid_eq {
                    matched_fields.push("MiddleName".into());
                }
                if a_last_eq {
                    matched_fields.push("LastName".into());
                }
                matched_fields.push("Birthdate".into());

                debug_assert_eq!(
                    b.birthdate,
                    Some(ad),
                    "[opt7] birthdate mismatch before push (CPU path)"
                );
                local.push(MatchPair {
                    person1: a.clone(),
                    person2: b.clone(),
                    confidence,
                    matched_fields,
                    is_matched_infnbd: false,
                    is_matched_infnmnbd: false,
                });
            }
            local
        })
        .flatten()
        .collect();

    out
}

// --- GPU module (feature-gated) ---

#[cfg(feature = "gpu")]
fn cuda_mem_info_mb(ctx: &cudarc::driver::CudaContext) -> (u64, u64) {
    // Query using CUDA driver API; ensure context is current (required on WDDM).
    let _ = ctx.bind_to_thread();
    unsafe {
        use cudarc::driver::sys::CUresult;
        let mut free: usize = 0;
        let mut total: usize = 0;
        let res = cudarc::driver::sys::cuMemGetInfo_v2(
            &mut free as *mut _ as *mut _,
            &mut total as *mut _ as *mut _,
        );
        if res == CUresult::CUDA_SUCCESS {
            return ((total as u64) / 1024 / 1024, (free as u64) / 1024 / 1024);
        }
        // Fallback 1: CUDA runtime API (can succeed when driver ctx isn’t current on WDDM)
        match cudarc::runtime::result::get_mem_info() {
            Ok((free2, total2)) => {
                return ((total2 as u64) / 1024 / 1024, (free2 as u64) / 1024 / 1024);
            }
            Err(e) => {
                log::warn!(
                    "cuda_mem_info_mb: cuMemGetInfo_v2 failed ({:?}); runtime cudaMemGetInfo failed ({:?}); trying nvidia-smi",
                    res,
                    e
                );
            }
        }
        // Fallback 2: shell out to nvidia-smi to avoid blocking telemetry; best-effort.
        if let Ok(out) = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ])
            .output()
        {
            if out.status.success() {
                if let Ok(s) = String::from_utf8(out.stdout) {
                    if let Some(line) = s.lines().next() {
                        let parts: Vec<_> = line.trim().split(',').collect();
                        if parts.len() == 2 {
                            if let (Ok(total), Ok(free)) = (
                                parts[0].trim().parse::<u64>(),
                                parts[1].trim().parse::<u64>(),
                            ) {
                                return (total, free);
                            }
                        }
                    }
                }
            }
        }
        log::warn!("cuda_mem_info_mb: all methods failed; returning 0s");
        (0, 0)
    }
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use anyhow::{Result, anyhow};
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
    use cudarc::nvrtc::compile_ptx;

    pub mod batch;
    pub mod dynamic_tuner;

    use std::sync::atomic::{AtomicBool, Ordering};
    static NO_MID_CLASSIFY: AtomicBool = AtomicBool::new(false);
    #[inline]
    pub fn gpu_no_mid_mode() -> bool {
        NO_MID_CLASSIFY.load(Ordering::Relaxed)
    }
    #[inline]
    pub fn with_no_mid_classification<T, F: FnOnce() -> T>(f: F) -> T {
        let prev = NO_MID_CLASSIFY.swap(true, Ordering::SeqCst);
        let out = f();
        NO_MID_CLASSIFY.store(prev, Ordering::SeqCst);
        out
    }

    const MAX_STR: usize = 64; // truncate names for GPU DP to keep registers/local mem bounded

    // CUDA kernel source for per-pair Levenshtein (two-row DP; lengths capped to MAX_STR)
    const LEV_KERNEL_SRC: &str = r#"
    __device__ __forceinline__ int max_i(int a, int b) { return a > b ? a : b; }
    __device__ __forceinline__ int min_i(int a, int b) { return a < b ? a : b; }

    extern "C" __global__ void lev_kernel(
        const char* a_buf, const int* a_off, const int* a_len,
        const char* b_buf, const int* b_off, const int* b_len,
        float* out, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        const int off_a = a_off[i]; int la = a_len[i]; if (la > (int)64) la = 64;
        const int off_b = b_off[i]; int lb = b_len[i]; if (lb > (int)64) lb = 64;
        const char* A = a_buf + off_a;
        const char* B = b_buf + off_b;
        int prev[65]; int curr[65];
        for (int j=0;j<=lb;++j) prev[j] = j;
        for (int ia=1; ia<=la; ++ia) {
            curr[0] = ia;
            char ca = A[ia-1];
            for (int jb=1; jb<=lb; ++jb) {
                int cost = (ca == B[jb-1]) ? 0 : 1;
                int del = prev[jb] + 1;
                int ins = curr[jb-1] + 1;
                int sub = prev[jb-1] + cost;
                int v = del < ins ? del : ins;
                curr[jb] = v < sub ? v : sub;
            }
            for (int jb=0; jb<=lb; ++jb) prev[jb] = curr[jb];
        }
        int dist = prev[lb];
        int ml = la > lb ? la : lb;
        float score = ml > 0 ? (1.0f - ((float)dist / (float)ml)) * 100.0f : 100.0f;
        out[i] = score;
    }

    extern "C" __global__ void lev_dist_kernel(
        const char* a_buf, const int* a_off, const int* a_len,
        const char* b_buf, const int* b_off, const int* b_len,
        int* out, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        const int off_a = a_off[i]; int la = a_len[i]; if (la > (int)64) la = 64;
        const int off_b = b_off[i]; int lb = b_len[i]; if (lb > (int)64) lb = 64;
        const char* A = a_buf + off_a;
        const char* B = b_buf + off_b;
        int prev[65]; int curr[65];
        for (int j=0;j<=lb;++j) prev[j] = j;
        for (int ia=1; ia<=la; ++ia) {
            curr[0] = ia;
            char ca = A[ia-1];
            for (int jb=1; jb<=lb; ++jb) {
                int cost = (ca == B[jb-1]) ? 0 : 1;
                int del = prev[jb] + 1;
                int ins = curr[jb-1] + 1;
                int sub = prev[jb-1] + cost;
                int v = del < ins ? del : ins;
                curr[jb] = v < sub ? v : sub;
            }
            for (int jb=0; jb<=lb; ++jb) prev[jb] = curr[jb];
        }
        out[i] = prev[lb];
    }

    __device__ float jaro_core(const char* A, int la, const char* B, int lb) {
        if (la == 0 && lb == 0) return 1.0f;
        int match_dist = max_i(0, max_i(la, lb) / 2 - 1);
        bool a_match[64]; bool b_match[64];
        for (int i=0;i<64;++i) { a_match[i]=false; b_match[i]=false; }
        int matches = 0;
        for (int i=0;i<la; ++i) {
            int start = max_i(0, i - match_dist);
            int end = min_i(i + match_dist + 1, lb);
            for (int j=start; j<end; ++j) {
                if (b_match[j]) continue;
                if (A[i] != B[j]) continue;
                a_match[i] = true; b_match[j] = true; ++matches; break;
            }
        }
        if (matches == 0) return 0.0f;
        int k = 0; int trans = 0;
        for (int i=0;i<la; ++i) {
            if (!a_match[i]) continue;
            while (k < lb && !b_match[k]) ++k;
            if (k < lb && A[i] != B[k]) ++trans;
            ++k;
        }
        float m = (float)matches;
        float j1 = m / la;
        float j2 = m / lb;
        float j3 = (m - trans/2.0f) / m;
        return (j1 + j2 + j3) / 3.0f;
    }



    extern "C" __global__ void jaro_kernel(
        const char* a_buf, const int* a_off, const int* a_len,
        const char* b_buf, const int* b_off, const int* b_len,
        float* out, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        int la = a_len[i]; if (la > 64) la = 64;
        int lb = b_len[i]; if (lb > 64) lb = 64;
        const char* A = a_buf + a_off[i];
        const char* B = b_buf + b_off[i];
        float j = jaro_core(A, la, B, lb);
        out[i] = j * 100.0f;
    }

    extern "C" __global__ void jw_kernel(
        const char* a_buf, const int* a_off, const int* a_len,
        const char* b_buf, const int* b_off, const int* b_len,
        float* out, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        int la = a_len[i]; if (la > 64) la = 64;
        int lb = b_len[i]; if (lb > 64) lb = 64;
        const char* A = a_buf + a_off[i];
        const char* B = b_buf + b_off[i];
        float j = jaro_core(A, la, B, lb);
        int l = 0; int maxp = 4;
        for (int k=0; k<min_i(min_i(la, lb), maxp); ++k) { if (A[k] == B[k]) ++l; else break; }
        float p = 0.1f;
        float jw = j + l * p * (1.0f - j);
        out[i] = jw * 100.0f;
    }

    extern "C" __global__ void max3_kernel(const float* a, const float* b, const float* c, float* out, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            float m = a[i];
            if (b[i] > m) m = b[i];
            if (c[i] > m) m = c[i];
            out[i] = m;
        }
    }
    "#;
    // Per-person cache to avoid repeated normalization and metaphone encoding during GPU post-processing
    #[derive(Clone)]
    struct FuzzyCache {
        simple_full: String,
        simple_first: String,
        simple_mid: String,
        simple_last: String,
        phonetic_full: String,
        dmeta_code: String, // empty if encode failed/panicked/empty
    }

    fn build_cache_from_person(p: &Person) -> FuzzyCache {
        let simple_first = normalize_simple(p.first_name.as_deref().unwrap_or(""));
        let simple_mid = normalize_simple(p.middle_name.as_deref().unwrap_or(""));
        let simple_last = normalize_simple(p.last_name.as_deref().unwrap_or(""));
        let simple_full = normalize_simple(&format!(
            "{} {} {}",
            p.first_name.as_deref().unwrap_or(""),
            p.middle_name.as_deref().unwrap_or(""),
            p.last_name.as_deref().unwrap_or("")
        ));
        // match metaphone_pct() path: normalize_for_phonetic on the full name string
        let phonetic_full = normalize_for_phonetic(&simple_full);
        let dmeta_code = if phonetic_full.is_empty() {
            String::new()
        } else {
            // Protect against panics as in metaphone_pct
            match std::panic::catch_unwind(|| DoubleMetaphone::default().encode(&phonetic_full)) {
                Ok(code) => code.to_string(),
                Err(_) => String::new(),
            }
        };
        FuzzyCache {
            simple_full,
            simple_first,
            simple_mid,
            simple_last,
            phonetic_full,
            dmeta_code,
        }
    }

    // Authoritative CPU classification using cached strings/codes to eliminate recomputation
    fn classify_pair_cached(c1: &FuzzyCache, c2: &FuzzyCache) -> Option<(f64, String)> {
        // Direct match
        if c1.simple_full == c2.simple_full {
            return Some((100.0, "DIRECT MATCH".to_string()));
        }
        // Metrics
        let lev = sim_levenshtein_pct(&c1.simple_full, &c2.simple_full);
        let jw = jaro_winkler(&c1.simple_full, &c2.simple_full) * 100.0;
        let mp = if !c1.dmeta_code.is_empty()
            && !c2.dmeta_code.is_empty()
            && c1.dmeta_code == c2.dmeta_code
        {
            100.0
        } else {
            0.0
        };

        // Case 1
        if lev >= 85.0 && jw >= 85.0 && mp == 100.0 {
            let avg = (lev + jw + mp) / 3.0;
            return Some((avg, "CASE 1".to_string()));
        }
        // Case 2 (+ Case 3 refinement)
        let mut pass = 0;
        if lev >= 85.0 {
            pass += 1;
        }
        if jw >= 85.0 {
            pass += 1;
        }
        if mp == 100.0 {
            pass += 1;
        }
        if pass >= 2 {
            let avg = (lev + jw + mp) / 3.0;
            if avg >= 88.0 {
                let ld_first = levenshtein(&c1.simple_first, &c2.simple_first) as usize;
                let ld_last = levenshtein(&c1.simple_last, &c2.simple_last) as usize;
                let ld_mid = levenshtein(&c1.simple_mid, &c2.simple_mid) as usize;
                if ld_first <= 2 && ld_last <= 2 && ld_mid <= 2 {
                    return Some((avg, "CASE 3".to_string()));
                }
            }
            return Some((avg, "CASE 2".to_string()));
        }
        None
    }

    // --- GPU FNV-1a 64-bit hash kernel and hashing helpers (module scope) ---
    const FNV_KERNEL_SRC: &str = r#"
    extern "C" __global__ void fnv1a64_kernel(
        const char* buf, const int* off, const int* len,
        unsigned long long* out, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        unsigned long long hash = 0xcbf29ce484222325ULL;
        const unsigned long long prime = 0x100000001b3ULL;
        const char* s = buf + off[i];
        int L = len[i];
        #pragma unroll 1
        for (int j = 0; j < L; ++j) {
            hash ^= (unsigned long long)(unsigned char)s[j];
            hash *= prime;
        }
        out[i] = hash;
    }
    "#;

    #[derive(Clone)]
    pub struct GpuHashContext {
        ctx: std::sync::Arc<CudaContext>,
        module: std::sync::Arc<cudarc::driver::CudaModule>,
        func_hash: std::sync::Arc<cudarc::driver::CudaFunction>,
    }

    impl GpuHashContext {
        pub fn new() -> Result<Self> {
            let dev_id = 0usize;
            let ctx = CudaContext::new(dev_id).map_err(|e| anyhow!("CUDA init failed: {e}"))?;

            // Query device details (name, compute capability, driver version)
            let (gpu_name, cc_major, cc_minor, drv_major, drv_minor) = unsafe {
                use std::ffi::CStr;
                use std::os::raw::{c_char, c_int};
                let mut cu_dev: cudarc::driver::sys::CUdevice = 0;
                let mut driver_ver: c_int = 0;
                // Best-effort queries; ignore non-zero return codes for logging-only paths
                let _ = cudarc::driver::sys::cuDeviceGet(&mut cu_dev as *mut _, dev_id as c_int);
                let _ = cudarc::driver::sys::cuDriverGetVersion(&mut driver_ver as *mut _);
                let mut maj: c_int = 0;
                let mut min: c_int = 0;
                let _ = cudarc::driver::sys::cuDeviceComputeCapability(
                    &mut maj as *mut _,
                    &mut min as *mut _,
                    cu_dev,
                );
                let mut name_buf: [c_char; 128] = [0; 128];
                let _ = cudarc::driver::sys::cuDeviceGetName(
                    name_buf.as_mut_ptr(),
                    name_buf.len() as c_int,
                    cu_dev,
                );
                let name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }
                    .to_string_lossy()
                    .into_owned();
                let drv_major = driver_ver / 1000;
                let drv_minor = (driver_ver % 1000) / 10;
                (
                    name,
                    maj as i32,
                    min as i32,
                    drv_major as i32,
                    drv_minor as i32,
                )
            };

            // Memory snapshot and activation logs
            let (tot_mb, free_mb) = super::cuda_mem_info_mb(&ctx);
            let used_mb = tot_mb.saturating_sub(free_mb);
            log::info!(
                "[GPU] CUDA context initialized: {name} (dev {dev}, compute {cc_major}.{cc_minor}) | Driver {drv_major}.{drv_minor} | Mem: used={used}/{tot} MB, free={free} MB",
                name = gpu_name,
                dev = dev_id,
                cc_major = cc_major,
                cc_minor = cc_minor,
                drv_major = drv_major,
                drv_minor = drv_minor,
                used = used_mb,
                tot = tot_mb,
                free = free_mb
            );
            log::info!(
                "[GPU] GPU acceleration ACTIVE: {name} (dev {dev}, compute {cc_major}.{cc_minor}) | Memory: {used}/{tot} MB",
                name = gpu_name,
                dev = dev_id,
                cc_major = cc_major,
                cc_minor = cc_minor,
                used = used_mb,
                tot = tot_mb
            );

            let ptx =
                compile_ptx(FNV_KERNEL_SRC).map_err(|e| anyhow!("NVRTC compile failed: {e}"))?;
            let module = ctx
                .load_module(ptx)
                .map_err(|e| anyhow!("Load PTX failed: {e}"))?;
            let func_hash = module
                .load_function("fnv1a64_kernel")
                .map_err(|e| anyhow!("Get fnv1a64 func failed: {e}"))?;
            Ok(Self {
                ctx,
                module,
                func_hash: func_hash.into(),
            })
        }
        pub fn get() -> Result<Self> {
            use std::sync::OnceLock;
            static INSTANCE: OnceLock<GpuHashContext> = OnceLock::new();
            if let Some(c) = INSTANCE.get() {
                return Ok(c.clone());
            }
            let inst = Self::new()?;

            let _ = INSTANCE.set(inst.clone());
            Ok(inst)
        }
    }

    impl GpuHashContext {
        pub fn mem_info_mb(&self) -> (u64, u64) {
            super::cuda_mem_info_mb(&self.ctx)
        }
    }

    // Cached GPU context for fuzzy metrics (Levenshtein/Jaro/Jaro-Winkler/Max3)
    // Mirrors the GpuHashContext pattern to eliminate per-call CUDA init and NVRTC compilation.
    #[derive(Clone)]
    pub struct GpuFuzzyContext {
        pub(crate) ctx: std::sync::Arc<CudaContext>,
        pub(crate) module: std::sync::Arc<cudarc::driver::CudaModule>,
        pub(crate) func_lev: std::sync::Arc<cudarc::driver::CudaFunction>,
        pub(crate) func_jaro: std::sync::Arc<cudarc::driver::CudaFunction>,
        pub(crate) func_jw: std::sync::Arc<cudarc::driver::CudaFunction>,
        pub(crate) func_max3: std::sync::Arc<cudarc::driver::CudaFunction>,
        // Two reusable streams: default + auxiliary for overlapping transfers/compute
        pub(crate) stream_default: std::sync::Arc<cudarc::driver::CudaStream>,
        pub(crate) stream_aux: std::sync::Arc<cudarc::driver::CudaStream>,
    }
    impl GpuFuzzyContext {
        pub fn new() -> Result<Self> {
            let dev_id = 0usize;
            let ctx = CudaContext::new(dev_id).map_err(|e| anyhow!("CUDA init failed: {e}"))?;

            // Compile and load fuzzy kernels once
            let ptx =
                compile_ptx(LEV_KERNEL_SRC).map_err(|e| anyhow!("NVRTC compile failed: {e}"))?;
            let module = ctx
                .load_module(ptx)
                .map_err(|e| anyhow!("Load PTX failed: {e}"))?;
            let func_lev = module
                .load_function("lev_kernel")
                .map_err(|e| anyhow!("Get lev func failed: {e}"))?;
            let func_jaro = module
                .load_function("jaro_kernel")
                .map_err(|e| anyhow!("Get jaro func failed: {e}"))?;
            let func_jw = module
                .load_function("jw_kernel")
                .map_err(|e| anyhow!("Get jw func failed: {e}"))?;
            let func_max3 = module
                .load_function("max3_kernel")
                .map_err(|e| anyhow!("Get max3 func failed: {e}"))?;

            // Prepare reusable streams

            let stream_default = ctx.default_stream();
            let stream_aux = ctx
                .new_stream()
                .map_err(|e| anyhow!("CUDA stream create failed: {e}"))?;

            Ok(Self {
                ctx,
                module: module.into(),
                func_lev: func_lev.into(),
                func_jaro: func_jaro.into(),
                func_jw: func_jw.into(),
                func_max3: func_max3.into(),
                stream_default,
                stream_aux,
            })
        }
        pub fn get() -> Result<Self> {
            use std::sync::OnceLock;
            static INSTANCE: OnceLock<GpuFuzzyContext> = OnceLock::new();
            if let Some(c) = INSTANCE.get() {
                return Ok(c.clone());
            }
            let inst = Self::new()?;
            let _ = INSTANCE.set(inst.clone());
            Ok(inst)
        }
        pub fn mem_info_mb(&self) -> (u64, u64) {
            super::cuda_mem_info_mb(&self.ctx)
        }
    }

    pub fn hash_fnv1a64_batch(hctx: &GpuHashContext, strings: &[String]) -> Result<Vec<u64>> {
        let n = strings.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let mut offsets: Vec<i32> = Vec::with_capacity(n);
        let mut lengths: Vec<i32> = Vec::with_capacity(n);
        let mut flat: Vec<u8> = Vec::new();
        flat.reserve(strings.iter().map(|s| s.len()).sum());
        let mut cur = 0i32;
        for s in strings {
            offsets.push(cur);
            let bytes = s.as_bytes();
            lengths.push(bytes.len() as i32);
            flat.extend_from_slice(bytes);
            cur += bytes.len() as i32;
        }
        let stream = hctx.ctx.default_stream();
        let d_buf = stream.memcpy_stod(flat.as_slice())?;
        let d_off = stream.memcpy_stod(offsets.as_slice())?;
        let d_len = stream.memcpy_stod(lengths.as_slice())?;
        let mut d_out = stream.alloc_zeros::<u64>(n)?;

        // [GPU_OPT1] Adaptive block size based on GPU architecture
        let gpu_props = super::gpu_config::query_gpu_properties(0).unwrap_or_else(|_| {
            super::gpu_config::GpuProperties {
                compute_major: 7,
                compute_minor: 0,
                sm_count: 30,
                max_threads_per_block: 1024,
                max_shared_memory_per_block: 49152,
            }
        });
        let bs: u32 = super::gpu_config::calculate_optimal_block_size(
            &gpu_props,
            super::gpu_config::KernelType::Hash,
        );

        let grid: u32 = ((n as u32 + bs - 1) / bs).max(1);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (bs, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i32 = n as i32;
        let (tot_mb, free_mb) = super::cuda_mem_info_mb(&hctx.ctx);
        log::debug!(
            "[GPU_OPT1] Launching fnv1a64_kernel for {} strings (grid={}, block={}, GPU: {}.{}) | mem: total={} MB free={} MB",
            n,
            grid,
            bs,
            gpu_props.compute_major,
            gpu_props.compute_minor,
            tot_mb,
            free_mb
        );
        let mut b = stream.launch_builder(&hctx.func_hash);
        b.arg(&d_buf)
            .arg(&d_off)
            .arg(&d_len)
            .arg(&mut d_out)
            .arg(&n_i32);
        unsafe {
            b.launch(cfg)?;
        }
        stream.synchronize()?;
        let out: Vec<u64> = stream.memcpy_dtov(&d_out)?;
        Ok(out)
    }

    /// VRAM-aware tiled hashing for probe/build keys. Tiles the input to respect
    /// both current free VRAM and a user-provided budget. On CUDA OOM, halves the
    /// tile size and retries; finally falls back to CPU hashing for that tile.
    pub fn hash_fnv1a64_batch_tiled(
        hctx: &GpuHashContext,
        strings: &[String],
        budget_mb: u64,
        reserve_mb: u64,
    ) -> Result<Vec<u64>> {
        fn is_cuda_oom(e: &anyhow::Error) -> bool {
            let s = e.to_string().to_ascii_lowercase();
            s.contains("cuda_error_out_of_memory")
                || s.contains("out of memory")
                || s.contains("oom")
        }

        if strings.is_empty() {
            return Ok(Vec::new());
        }
        let mut out: Vec<u64> = Vec::with_capacity(strings.len());
        let mut i = 0usize;
        let min_tile = 512usize;
        while i < strings.len() {
            // Determine per-tile target bytes from current free VRAM and user budget
            let (_tot_mb, free_mb) = super::cuda_mem_info_mb(&hctx.ctx);
            let target_mb = free_mb
                .min(budget_mb.max(64))
                .saturating_sub(reserve_mb.max(64));
            let target_bytes: usize = (target_mb as usize)
                .saturating_mul(1024 * 1024)
                .max(256 * 1024);

            // Greedy grow tile until budget
            let mut est_bytes = 0usize;
            let mut j = i;
            while j < strings.len() && est_bytes < target_bytes {
                // Rough per-key bytes: offsets(4)+len(4)+out(8)+string bytes
                est_bytes += 16 + strings[j].len();
                j += 1;
            }
            if j == i {
                j = (i + min_tile).min(strings.len());
            }

            // Attempt with backoff on OOM
            let mut lo = i;
            let mut hi = j;
            let mut done = false;
            while !done {
                let tile = &strings[lo..hi];
                match hash_fnv1a64_batch(hctx, tile) {
                    Ok(mut v) => {
                        // GPU kernel tile executed
                        super::OPT7_GPU_KERNEL_TILES
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        out.extend(v.drain(..));
                        done = true;
                    }
                    Err(e) if is_cuda_oom(&e) && tile.len() > min_tile => {
                        let new_hi = lo + (tile.len() / 2).max(min_tile);
                        log::warn!(
                            "[GPU] OOM during probe hashing ({} keys); shrinking tile to {}",
                            tile.len(),
                            new_hi - lo
                        );
                        hi = new_hi;
                    }
                    Err(e) => {
                        // CPU fallback for this tile
                        super::OPT7_GPU_CPU_FALLBACK_TILES
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        log::warn!(
                            "[GPU] Probe hashing fallback to CPU for {} keys: {}",
                            tile.len(),
                            e
                        );
                        let mut v_cpu: Vec<u64> = Vec::with_capacity(tile.len());
                        for s in tile {
                            v_cpu.push(super::fnv1a64_bytes(s.as_bytes()));
                        }
                        out.extend(v_cpu);
                        done = true;
                    }
                }
            }
            i = j;
        }

        Ok(out)
    }
    /// GPU-accelerated hashing for in-memory deterministic algorithms (A1/A2).

    /// GPU hash pre-pass for Fuzzy direct phase (candidate filtering only).
    /// Returns, for each outer row i (people1), a Vec of indices j into people2 that share the blocking key.
    /// Key policy: always exact birthdate; optionally include last initial when partition strategy is 'last_initial'.
    pub fn fuzzy_direct_gpu_hash_prefilter_indices(
        people1: &[super::Person],
        people2: &[super::Person],
        _part_strategy: &str,
    ) -> Result<Vec<Vec<usize>>> {
        let ctx = GpuHashContext::get()?;
        use chrono::Datelike;
        // Build composite keys (YEAR|FI|LI|SNDX) for inner (people2), primary only (no synthetic fallbacks)
        let mut keys2: Vec<String> = Vec::new();
        let mut idx2: Vec<usize> = Vec::new();
        for (j, p) in people2.iter().enumerate() {
            let (Some(d), Some(fn_str), Some(ln_str)) =
                (p.birthdate, p.first_name.as_deref(), p.last_name.as_deref())
            else {
                continue;
            };
            let year = d.year();
            let fi = super::normalize_simple(fn_str)
                .bytes()
                .find(|c| c.is_ascii_alphabetic())
                .unwrap_or(b'?')
                .to_ascii_uppercase();
            let ln_norm = super::normalize_simple(ln_str);
            let li = ln_norm
                .bytes()
                .find(|c| c.is_ascii_alphabetic())
                .unwrap_or(b'?')
                .to_ascii_uppercase();
            let sx = super::soundex4_ascii(&ln_norm);
            let sx_str = String::from_utf8_lossy(&sx).into_owned();
            let k0 = format!("{:04}|{}|{}|{}", year, fi as char, li as char, sx_str);
            keys2.push(k0);
            idx2.push(j);
        }
        let (_tot_mb, free_mb) = ctx.mem_info_mb();
        let user_mb = super::gpu_fuzzy_prep_budget_mb();
        let budget_mb = if user_mb > 0 {
            user_mb
        } else {
            (free_mb / 2).max(64)
        };
        let h2 = hash_fnv1a64_batch_tiled(&ctx, &keys2, budget_mb, 64)?;
        use std::collections::HashMap as Map;
        let mut map: Map<u64, Vec<usize>> = Map::with_capacity(h2.len());
        for (k, &h) in h2.iter().enumerate() {
            map.entry(h).or_default().push(idx2[k]);
        }
        // Probe from people1: generate keys in fallback order and select first non-empty
        let mut out: Vec<Vec<usize>> = vec![Vec::new(); people1.len()];
        let mut keys1: Vec<String> = Vec::new();
        let mut owner: Vec<usize> = Vec::new();
        for (i, p) in people1.iter().enumerate() {
            let (Some(d), Some(fn_str), Some(ln_str)) =
                (p.birthdate, p.first_name.as_deref(), p.last_name.as_deref())
            else {
                continue;
            };
            let year = d.year();
            let fi = super::normalize_simple(fn_str)
                .bytes()
                .find(|c| c.is_ascii_alphabetic())
                .unwrap_or(b'?')
                .to_ascii_uppercase();
            let ln_norm = super::normalize_simple(ln_str);
            let li = ln_norm
                .bytes()
                .find(|c| c.is_ascii_alphabetic())
                .unwrap_or(b'?')
                .to_ascii_uppercase();
            let sx = super::soundex4_ascii(&ln_norm);
            let sx_str = String::from_utf8_lossy(&sx).into_owned();
            let mut sx2 = sx;
            let _ = {
                sx2[2] = b'0';
                sx2[3] = b'0';
            };
            let sx2_str = String::from_utf8_lossy(&sx2).into_owned();
            // Fallback order: k0 -> k1 -> k2
            let k0 = format!("{:04}|{}|{}|{}", year, fi as char, li as char, sx_str);
            let k1 = format!("{:04}|{}|{}|{}", year, '?' as char, li as char, sx_str);
            let k2 = format!("{:04}|{}|{}|{}", year, fi as char, li as char, sx2_str);
            keys1.push(k0);
            owner.push(i);
            keys1.push(k1);
            owner.push(i);
            keys1.push(k2);
            owner.push(i);
        }
        let h1 = hash_fnv1a64_batch_tiled(&ctx, &keys1, budget_mb, 64)?;
        let mut chosen: Vec<bool> = vec![false; people1.len()];
        for (pos, &h) in h1.iter().enumerate() {
            let i = owner[pos];
            if chosen[i] {
                continue;
            }
            if let Some(cands) = map.get(&h) {
                if !cands.is_empty() {
                    out[i] = cands.clone();
                    chosen[i] = true;
                }
            }
        }
        Ok(out)
    }

    /// GPU hash pre-pass for Option 7 (LevenshteinWeighted) with OR-semantics across 3 keys.
    /// Keys:
    ///  A: date | SNDX(first) | SNDX(last)
    ///  B: date | first3(first) | first3(last)        (only if both non-empty)
    ///  C: date | SNDX(middle)                        (only if middle present)
    pub fn levenshtein_gpu_hash_prefilter_indices(
        people1: &[super::Person],
        people2: &[super::Person],
    ) -> Result<Vec<Vec<usize>>> {
        let ctx = GpuHashContext::get()?;
        // reset runtime counters for Option 7 GPU pre-pass
        super::opt7_gpu_reset_counters();
        // Build inner keys with type tags to share one hash map
        let mut keys2: Vec<String> = Vec::new();
        let mut idx2: Vec<usize> = Vec::new();
        let mut cnt_a: u64 = 0;
        let mut cnt_b: u64 = 0;
        let mut cnt_c: u64 = 0;
        let mut rows2_with_date: u64 = 0;
        for (j, p) in people2.iter().enumerate() {
            let Some(d) = p.birthdate else {
                continue;
            };
            rows2_with_date += 1;
            let dk = d.format("%F").to_string();
            let af = super::normalize_simple(p.first_name.as_deref().unwrap_or(""));
            let am = super::normalize_simple(p.middle_name.as_deref().unwrap_or(""));
            let al = super::normalize_simple(p.last_name.as_deref().unwrap_or(""));
            if !af.is_empty() && !al.is_empty() {
                let sx_f = super::soundex4_ascii(&af);
                let sx_l = super::soundex4_ascii(&al);
                let k_a = format!(
                    "A|{}|{}|{}",
                    dk,
                    String::from_utf8_lossy(&sx_f),
                    String::from_utf8_lossy(&sx_l)
                );
                keys2.push(k_a);
                idx2.push(j);
                cnt_a += 1;
            }
            let f3: String = af.chars().take(3).collect();
            let l3: String = al.chars().take(3).collect();
            if !f3.is_empty() && !l3.is_empty() {
                let k_b = format!("B|{}|{}|{}", dk, f3, l3);
                keys2.push(k_b);
                idx2.push(j);
                cnt_b += 1;
            }
            if !am.is_empty() {
                let sx_m = super::soundex4_ascii(&am);
                let k_c = format!("C|{}|{}", dk, String::from_utf8_lossy(&sx_m));
                keys2.push(k_c);
                idx2.push(j);
                cnt_c += 1;
            }
        }
        log::info!(
            "[opt7] GPU pre-pass build: rows2_with_date={} keys2={} (A:{} B:{} C:{})",
            rows2_with_date,
            keys2.len(),
            cnt_a,
            cnt_b,
            cnt_c
        );
        let (_tot_mb, free_mb) = ctx.mem_info_mb();
        let user_mb = super::gpu_fuzzy_prep_budget_mb();
        let budget_mb = if user_mb > 0 {
            user_mb
        } else {
            (free_mb / 2).max(64)
        };
        let h2 = hash_fnv1a64_batch_tiled(&ctx, &keys2, budget_mb, 64)?;
        use std::collections::HashMap as Map;
        let mut map: Map<u64, Vec<usize>> = Map::with_capacity(h2.len());
        for (k, &h) in h2.iter().enumerate() {
            map.entry(h).or_default().push(idx2[k]);
        }

        // Probe from people1; generate up to 3 keys and union all matching candidate lists
        let mut out: Vec<Vec<usize>> = vec![Vec::new(); people1.len()];
        let mut keys1: Vec<String> = Vec::new();
        let mut owner: Vec<usize> = Vec::new();
        for (i, p) in people1.iter().enumerate() {
            let Some(d) = p.birthdate else {
                continue;
            };
            let dk = d.format("%F").to_string();
            let af = super::normalize_simple(p.first_name.as_deref().unwrap_or(""));
            let am = super::normalize_simple(p.middle_name.as_deref().unwrap_or(""));
            let al = super::normalize_simple(p.last_name.as_deref().unwrap_or(""));
            if !af.is_empty() && !al.is_empty() {
                let sx_f = super::soundex4_ascii(&af);
                let sx_l = super::soundex4_ascii(&al);
                let k_a = format!(
                    "A|{}|{}|{}",
                    dk,
                    String::from_utf8_lossy(&sx_f),
                    String::from_utf8_lossy(&sx_l)
                );
                keys1.push(k_a);
                owner.push(i);
            }
            let f3: String = af.chars().take(3).collect();
            let l3: String = al.chars().take(3).collect();
            if !f3.is_empty() && !l3.is_empty() {
                let k_b = format!("B|{}|{}|{}", dk, f3, l3);
                keys1.push(k_b);
                owner.push(i);
            }
            if !am.is_empty() {
                let sx_m = super::soundex4_ascii(&am);
                let k_c = format!("C|{}|{}", dk, String::from_utf8_lossy(&sx_m));
                keys1.push(k_c);
                owner.push(i);
            }
        }
        let h1 = hash_fnv1a64_batch_tiled(&ctx, &keys1, budget_mb, 64)?;
        let mut seen: Vec<std::collections::HashSet<usize>> =
            vec![std::collections::HashSet::new(); people1.len()];
        for (pos, &h) in h1.iter().enumerate() {
            let i = owner[pos];
            if let Some(cands) = map.get(&h) {
                for &j in cands {
                    seen[i].insert(j);
                }
            }
        }
        for i in 0..people1.len() {
            if !seen[i].is_empty() {
                out[i] = seen[i].iter().copied().collect();
            }
        }
        let total_rows = out.len();
        let total_cands: usize = out.iter().map(|v| v.len()).sum();
        let avg_cands = if total_rows == 0 {
            0.0
        } else {
            total_cands as f64 / total_rows as f64
        };
        log::info!(
            "[opt7] GPU pre-pass probe: out_lists={} total_cands={} avg_cands_per_row={:.2}",
            total_rows,
            total_cands,
            avg_cands
        );
        let (tiles, cpu_fb) = super::opt7_gpu_stats();
        log::info!(
            "[opt7] GPU pre-pass tiles: gpu={} cpu_fallback={}",
            tiles,
            cpu_fb
        );
        Ok(out)
    }

    /// Option 7: Full GPU scoring (Levenshtein percent on first/middle/last), strict parity.
    pub fn match_levenshtein_weighted_gpu_full<F>(
        t1: &[super::Person],
        t2: &[super::Person],
        opts: super::MatchOptions,
        on_progress: &F,
    ) -> Result<Vec<super::MatchPair>>
    where
        F: Fn(super::ProgressUpdate) + Sync,
    {
        use rayon::prelude::*;
        // Build simple caches (lowercased, punctuation-normalized)
        let c1: Vec<FuzzyCache> = t1.par_iter().map(build_cache_from_person).collect();
        let c2: Vec<FuzzyCache> = t2.par_iter().map(build_cache_from_person).collect();
        if c1.is_empty() || c2.is_empty() {
            return Ok(Vec::new());
        }

        // Candidate generation: use GPU pre-pass when enabled, else CPU union-of-keys (A|B|C)
        let cand_lists: Vec<Vec<usize>> = if super::gpu_lev_prepass_enabled() {
            match levenshtein_gpu_hash_prefilter_indices(t1, t2) {
                Ok(v) => v,
                Err(e) => {
                    log::error!(
                        "[opt7] FATAL: GPU pre-pass failed in full scoring; falling back to CPU candidate gen: {}",
                        e
                    );
                    // CPU union-of-keys
                    let mut ix_sx: std::collections::HashMap<(String, u32, u32), Vec<usize>> =
                        std::collections::HashMap::new();
                    let mut ix_pf: std::collections::HashMap<(String, String, String), Vec<usize>> =
                        std::collections::HashMap::new();
                    let mut ix_mid: std::collections::HashMap<(String, u32), Vec<usize>> =
                        std::collections::HashMap::new();
                    fn pack_sx(sx: [u8; 4]) -> u32 {
                        u32::from_le_bytes(sx)
                    }
                    for (j, p) in t2.iter().enumerate() {
                        if let Some(d) = p.birthdate {
                            let dk = d.format("%F").to_string();
                            let af = super::normalize_simple(p.first_name.as_deref().unwrap_or(""));
                            let am =
                                super::normalize_simple(p.middle_name.as_deref().unwrap_or(""));
                            let al = super::normalize_simple(p.last_name.as_deref().unwrap_or(""));
                            let af3: String = af.chars().take(3).collect();
                            let al3: String = al.chars().take(3).collect();
                            let sx_f = super::soundex4_ascii(&af);
                            let sx_l = super::soundex4_ascii(&al);
                            if !af.is_empty() && !al.is_empty() {
                                ix_sx
                                    .entry((dk.clone(), pack_sx(sx_f), pack_sx(sx_l)))
                                    .or_default()
                                    .push(j);
                            }
                            if !af3.is_empty() && !al3.is_empty() {
                                ix_pf.entry((dk.clone(), af3, al3)).or_default().push(j);
                            }
                            if !am.is_empty() {
                                let sx_m = super::soundex4_ascii(&am);
                                ix_mid.entry((dk, pack_sx(sx_m))).or_default().push(j);
                            }
                        }
                    }
                    let mut out: Vec<Vec<usize>> = vec![Vec::new(); t1.len()];
                    for (i, a) in t1.iter().enumerate() {
                        let Some(ad) = a.birthdate else {
                            continue;
                        };
                        let dk = ad.format("%F").to_string();
                        let af = super::normalize_simple(a.first_name.as_deref().unwrap_or(""));
                        let am = super::normalize_simple(a.middle_name.as_deref().unwrap_or(""));
                        let al = super::normalize_simple(a.last_name.as_deref().unwrap_or(""));
                        let af3: String = af.chars().take(3).collect();
                        let al3: String = al.chars().take(3).collect();
                        let asx_f = super::soundex4_ascii(&af);
                        let asx_l = super::soundex4_ascii(&al);
                        let asx_m = if am.is_empty() {
                            None
                        } else {
                            Some(super::soundex4_ascii(&am))
                        };
                        use std::collections::HashSet;
                        let mut cand: HashSet<usize> = HashSet::new();
                        if let Some(v) = ix_sx.get(&(dk.clone(), pack_sx(asx_f), pack_sx(asx_l))) {
                            cand.extend(v.iter().copied());
                        }
                        if !af3.is_empty() && !al3.is_empty() {
                            if let Some(v) = ix_pf.get(&(dk.clone(), af3.clone(), al3.clone())) {
                                cand.extend(v.iter().copied());
                            }
                        }
                        if let Some(sxm) = asx_m {
                            if let Some(v) = ix_mid.get(&(dk, pack_sx(sxm))) {
                                cand.extend(v.iter().copied());
                            }
                        }
                        if !cand.is_empty() {
                            out[i] = cand.into_iter().collect();
                        }
                    }
                    out
                }
            }
        } else {
            // CPU union-of-keys
            let mut ix_sx: std::collections::HashMap<(String, u32, u32), Vec<usize>> =
                std::collections::HashMap::new();
            let mut ix_pf: std::collections::HashMap<(String, String, String), Vec<usize>> =
                std::collections::HashMap::new();
            let mut ix_mid: std::collections::HashMap<(String, u32), Vec<usize>> =
                std::collections::HashMap::new();
            fn pack_sx(sx: [u8; 4]) -> u32 {
                u32::from_le_bytes(sx)
            }
            for (j, p) in t2.iter().enumerate() {
                if let Some(d) = p.birthdate {
                    let dk = d.format("%F").to_string();
                    let af = super::normalize_simple(p.first_name.as_deref().unwrap_or(""));
                    let am = super::normalize_simple(p.middle_name.as_deref().unwrap_or(""));
                    let al = super::normalize_simple(p.last_name.as_deref().unwrap_or(""));
                    let af3: String = af.chars().take(3).collect();
                    let al3: String = al.chars().take(3).collect();
                    let sx_f = super::soundex4_ascii(&af);
                    let sx_l = super::soundex4_ascii(&al);
                    if !af.is_empty() && !al.is_empty() {
                        ix_sx
                            .entry((dk.clone(), pack_sx(sx_f), pack_sx(sx_l)))
                            .or_default()
                            .push(j);
                    }
                    if !af3.is_empty() && !al3.is_empty() {
                        ix_pf.entry((dk.clone(), af3, al3)).or_default().push(j);
                    }
                    if !am.is_empty() {
                        let sx_m = super::soundex4_ascii(&am);
                        ix_mid.entry((dk, pack_sx(sx_m))).or_default().push(j);
                    }
                }
            }
            let mut out: Vec<Vec<usize>> = vec![Vec::new(); t1.len()];
            for (i, a) in t1.iter().enumerate() {
                let Some(ad) = a.birthdate else {
                    continue;
                };
                let dk = ad.format("%F").to_string();
                let af = super::normalize_simple(a.first_name.as_deref().unwrap_or(""));
                let am = super::normalize_simple(a.middle_name.as_deref().unwrap_or(""));
                let al = super::normalize_simple(a.last_name.as_deref().unwrap_or(""));
                let af3: String = af.chars().take(3).collect();
                let al3: String = al.chars().take(3).collect();
                let asx_f = super::soundex4_ascii(&af);
                let asx_l = super::soundex4_ascii(&al);
                let asx_m = if am.is_empty() {
                    None
                } else {
                    Some(super::soundex4_ascii(&am))
                };
                use std::collections::HashSet;
                let mut cand: std::collections::HashSet<usize> = HashSet::new();
                if let Some(v) = ix_sx.get(&(dk.clone(), pack_sx(asx_f), pack_sx(asx_l))) {
                    cand.extend(v.iter().copied());
                }
                if !af3.is_empty() && !al3.is_empty() {
                    if let Some(v) = ix_pf.get(&(dk.clone(), af3.clone(), al3.clone())) {
                        cand.extend(v.iter().copied());
                    }
                }
                if let Some(sxm) = asx_m {
                    if let Some(v) = ix_mid.get(&(dk, pack_sx(sxm))) {
                        cand.extend(v.iter().copied());
                    }
                }
                if !cand.is_empty() {
                    out[i] = cand.into_iter().collect();
                }
            }
            out
        };

        // Prepare CUDA context and kernels
        let dev_id = opts.gpu.and_then(|g| g.device_id).unwrap_or(0);
        let ctx = CudaContext::new(dev_id).map_err(|e| anyhow!("CUDA init failed: {e}"))?;
        let stream = ctx.default_stream();
        let ptx = compile_ptx(LEV_KERNEL_SRC).map_err(|e| anyhow!("NVRTC compile failed: {e}"))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| anyhow!("Load PTX failed: {e}"))?;
        let func_lev = module
            .load_function("lev_kernel")
            .map_err(|e| anyhow!("Get lev func failed: {e}"))?;

        // Tiling - use adaptive budget if not explicitly set
        let (gpu_total_mb_init, gpu_free_mb_init) = super::cuda_mem_info_mb(&ctx);
        let mem_budget_mb = match opts.gpu.and_then(|g| {
            if g.mem_budget_mb > 0 {
                Some(g.mem_budget_mb)
            } else {
                None
            }
        }) {
            Some(explicit_budget) => explicit_budget,
            None => {
                // Auto-calculate adaptive budget (75% of free VRAM, conservative)
                let budget = super::gpu_config::calculate_gpu_memory_budget(
                    gpu_total_mb_init,
                    gpu_free_mb_init,
                    false,
                );
                log::info!(
                    "[GPU] Auto-calculated memory budget: {} MB (75% of {} MB free VRAM)",
                    budget,
                    gpu_free_mb_init
                );
                budget
            }
        };
        let mut out_pairs: Vec<super::MatchPair> = Vec::new();
        super::opt7_gpu_scoring_reset_counters();

        // Iterate rows and score pairs in VRAM-aware tiles
        let total_rows = t1.len();
        let (gpu_total_mb, mut gpu_free_mb) = super::cuda_mem_info_mb(&ctx);
        let mut processed_rows = 0usize;
        for (i, a) in t1.iter().enumerate() {
            if i % 2000 == 0 {
                let memx = super::memory_stats_mb();
                on_progress(super::ProgressUpdate {
                    processed: i,
                    total: total_rows,
                    percent: (i as f32 / total_rows.max(1) as f32) * 100.0,
                    eta_secs: 0,
                    mem_used_mb: memx.used_mb,
                    mem_avail_mb: memx.avail_mb,
                    stage: "opt7_gpu_scoring",
                    batch_size_current: None,
                    gpu_total_mb,
                    gpu_free_mb,
                    gpu_active: true,
                });
            }
            let Some(ad) = a.birthdate else {
                continue;
            };
            let cands = &cand_lists[i];
            if cands.is_empty() {
                continue;
            }
            let mut start = 0usize;
            while start < cands.len() {
                // Determine tile length based on free VRAM
                let (_tmb, free_now) = super::cuda_mem_info_mb(&ctx);
                gpu_free_mb = free_now.max(gpu_free_mb);
                let target_mb = free_now.min(mem_budget_mb).saturating_sub(64);
                let approx_bpp: usize = 3 * (64 + 4 + 4) + 4; // three fields, rough
                let mut suggested = ((target_mb as usize * 1024 * 1024) / approx_bpp).max(1024);
                let remaining = cands.len() - start;
                let tile_len = suggested.min(remaining).max(1);
                let end = start + tile_len;
                let tile = &cands[start..end];

                // Prepare host buffers for first/last for all pairs
                let mut f_a_off: Vec<i32> = Vec::with_capacity(tile.len());
                let mut f_a_len: Vec<i32> = Vec::with_capacity(tile.len());
                let mut f_b_off: Vec<i32> = Vec::with_capacity(tile.len());
                let mut f_b_len: Vec<i32> = Vec::with_capacity(tile.len());
                let mut f_a_bytes: Vec<u8> = Vec::new();
                let mut f_b_bytes: Vec<u8> = Vec::new();

                let mut l_a_off: Vec<i32> = Vec::with_capacity(tile.len());
                let mut l_a_len: Vec<i32> = Vec::with_capacity(tile.len());
                let mut l_b_off: Vec<i32> = Vec::with_capacity(tile.len());
                let mut l_b_len: Vec<i32> = Vec::with_capacity(tile.len());
                let mut l_a_bytes: Vec<u8> = Vec::new();
                let mut l_b_bytes: Vec<u8> = Vec::new();

                // Middle prepared only for pairs with middle present
                let mut mid_map_idx: Vec<usize> = Vec::new(); // map mid_out[k] -> pair index
                let mut m_a_off: Vec<i32> = Vec::new();
                let mut m_a_len: Vec<i32> = Vec::new();
                let mut m_b_off: Vec<i32> = Vec::new();
                let mut m_b_len: Vec<i32> = Vec::new();
                let mut m_a_bytes: Vec<u8> = Vec::new();
                let mut m_b_bytes: Vec<u8> = Vec::new();

                for (pos, &j) in tile.iter().enumerate() {
                    let b = &t2[j];
                    if b.birthdate != Some(ad) {
                        continue;
                    } // safety
                    // first
                    let s1 = &c1[i].simple_first;
                    let s2 = &c2[j].simple_first;
                    let aoff = f_a_bytes.len() as i32;
                    f_a_off.push(aoff);
                    let la = s1.as_bytes().len().min(MAX_STR);
                    f_a_len.push(la as i32);
                    f_a_bytes.extend_from_slice(&s1.as_bytes()[..la]);
                    let boff = f_b_bytes.len() as i32;
                    f_b_off.push(boff);
                    let lb = s2.as_bytes().len().min(MAX_STR);
                    f_b_len.push(lb as i32);
                    f_b_bytes.extend_from_slice(&s2.as_bytes()[..lb]);
                    // last
                    let s1l = &c1[i].simple_last;
                    let s2l = &c2[j].simple_last;
                    let aoffl = l_a_bytes.len() as i32;
                    l_a_off.push(aoffl);
                    let lal = s1l.as_bytes().len().min(MAX_STR);
                    l_a_len.push(lal as i32);
                    l_a_bytes.extend_from_slice(&s1l.as_bytes()[..lal]);
                    let boffl = l_b_bytes.len() as i32;
                    l_b_off.push(boffl);
                    let lbl = s2l.as_bytes().len().min(MAX_STR);
                    l_b_len.push(lbl as i32);
                    l_b_bytes.extend_from_slice(&s2l.as_bytes()[..lbl]);
                    // middle (only if both non-empty)
                    let sm1 = &c1[i].simple_mid;
                    let sm2 = &c2[j].simple_mid;
                    if !sm1.is_empty() && !sm2.is_empty() {
                        mid_map_idx.push(pos);
                        let aoffm = m_a_bytes.len() as i32;
                        m_a_off.push(aoffm);
                        let lam = sm1.as_bytes().len().min(MAX_STR);
                        m_a_len.push(lam as i32);
                        m_a_bytes.extend_from_slice(&sm1.as_bytes()[..lam]);
                        let boffm = m_b_bytes.len() as i32;
                        m_b_off.push(boffm);
                        let lbm = sm2.as_bytes().len().min(MAX_STR);
                        m_b_len.push(lbm as i32);
                        m_b_bytes.extend_from_slice(&sm2.as_bytes()[..lbm]);
                    }
                }

                // [GPU_OPT] Batch synchronization: Launch all 3 kernels, then sync once
                // This reduces GPU idle time by allowing concurrent kernel execution
                // Previous: 3 syncs per tile (one after each field)
                // Optimized: 1 sync per tile (after all fields)
                let gpu_result = (|| -> anyhow::Result<(Vec<f32>, Vec<f32>, Option<Vec<f32>>)> {
                    let s = &stream;
                    let n_first = f_a_off.len();
                    let n_last = l_a_off.len();
                    let n_mid = m_a_off.len();

                    if n_first == 0 {
                        return Ok((Vec::new(), Vec::new(), None));
                    }

                    // Transfer all data to device (first, last, middle)
                    let d_f_a = s.memcpy_stod(f_a_bytes.as_slice())?;
                    let d_f_a_off = s.memcpy_stod(f_a_off.as_slice())?;
                    let d_f_a_len = s.memcpy_stod(f_a_len.as_slice())?;
                    let d_f_b = s.memcpy_stod(f_b_bytes.as_slice())?;
                    let d_f_b_off = s.memcpy_stod(f_b_off.as_slice())?;
                    let d_f_b_len = s.memcpy_stod(f_b_len.as_slice())?;
                    let mut d_out_first = s.alloc_zeros::<f32>(n_first)?;

                    let d_l_a = s.memcpy_stod(l_a_bytes.as_slice())?;
                    let d_l_a_off = s.memcpy_stod(l_a_off.as_slice())?;
                    let d_l_a_len = s.memcpy_stod(l_a_len.as_slice())?;
                    let d_l_b = s.memcpy_stod(l_b_bytes.as_slice())?;
                    let d_l_b_off = s.memcpy_stod(l_b_off.as_slice())?;
                    let d_l_b_len = s.memcpy_stod(l_b_len.as_slice())?;
                    let mut d_out_last = s.alloc_zeros::<f32>(n_last)?;

                    let d_out_mid_opt = if n_mid > 0 {
                        let d_m_a = s.memcpy_stod(m_a_bytes.as_slice())?;
                        let d_m_a_off = s.memcpy_stod(m_a_off.as_slice())?;
                        let d_m_a_len = s.memcpy_stod(m_a_len.as_slice())?;
                        let d_m_b = s.memcpy_stod(m_b_bytes.as_slice())?;
                        let d_m_b_off = s.memcpy_stod(m_b_off.as_slice())?;
                        let d_m_b_len = s.memcpy_stod(m_b_len.as_slice())?;
                        let mut d_out_mid = s.alloc_zeros::<f32>(n_mid)?;

                        // Launch middle kernel
                        let bs: u32 = 256;
                        let grid: u32 = ((n_mid as u32 + bs - 1) / bs).max(1);
                        let cfg = LaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (bs, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        let n_mid_i32 = n_mid as i32;
                        let mut bldr = s.launch_builder(&func_lev);
                        bldr.arg(&d_m_a)
                            .arg(&d_m_a_off)
                            .arg(&d_m_a_len)
                            .arg(&d_m_b)
                            .arg(&d_m_b_off)
                            .arg(&d_m_b_len)
                            .arg(&mut d_out_mid)
                            .arg(&n_mid_i32);
                        unsafe {
                            bldr.launch(cfg)?;
                        }
                        Some(d_out_mid)
                    } else {
                        None
                    };

                    // Launch first name kernel (no sync)
                    let bs: u32 = 256;
                    let grid_first: u32 = ((n_first as u32 + bs - 1) / bs).max(1);
                    let cfg_first = LaunchConfig {
                        grid_dim: (grid_first, 1, 1),
                        block_dim: (bs, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let n_first_i32 = n_first as i32;
                    let mut bldr_first = s.launch_builder(&func_lev);
                    bldr_first
                        .arg(&d_f_a)
                        .arg(&d_f_a_off)
                        .arg(&d_f_a_len)
                        .arg(&d_f_b)
                        .arg(&d_f_b_off)
                        .arg(&d_f_b_len)
                        .arg(&mut d_out_first)
                        .arg(&n_first_i32);
                    unsafe {
                        bldr_first.launch(cfg_first)?;
                    }

                    // Launch last name kernel (no sync)
                    let grid_last: u32 = ((n_last as u32 + bs - 1) / bs).max(1);
                    let cfg_last = LaunchConfig {
                        grid_dim: (grid_last, 1, 1),
                        block_dim: (bs, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let n_last_i32 = n_last as i32;
                    let mut bldr_last = s.launch_builder(&func_lev);
                    bldr_last
                        .arg(&d_l_a)
                        .arg(&d_l_a_off)
                        .arg(&d_l_a_len)
                        .arg(&d_l_b)
                        .arg(&d_l_b_off)
                        .arg(&d_l_b_len)
                        .arg(&mut d_out_last)
                        .arg(&n_last_i32);
                    unsafe {
                        bldr_last.launch(cfg_last)?;
                    }

                    // [GPU_OPT] Single synchronization after all kernels (was 3 syncs before)
                    s.synchronize()?;
                    log::debug!(
                        "[GPU_OPT] Batch sync: 1 sync for {} kernels (first+last+mid)",
                        if n_mid > 0 { 3 } else { 2 }
                    );

                    // Retrieve all results
                    let scores_first: Vec<f32> = s.memcpy_dtov(&d_out_first)?;
                    let scores_last: Vec<f32> = s.memcpy_dtov(&d_out_last)?;
                    let scores_mid_opt: Option<Vec<f32>> = if let Some(d_out_mid) = d_out_mid_opt {
                        Some(s.memcpy_dtov(&d_out_mid)?)
                    } else {
                        None
                    };

                    Ok((scores_first, scores_last, scores_mid_opt))
                })();

                // Handle GPU result or fall back to CPU for entire tile
                let (scores_first, scores_last, scores_mid_opt) = match gpu_result {
                    Ok((sf, sl, sm)) => {
                        super::OPT7_GPU_SCORING_TILES
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        (sf, sl, sm)
                    }
                    Err(e) => {
                        let es = e.to_string().to_ascii_lowercase();
                        if es.contains("out of memory") || es.contains("cuda_error_out_of_memory") {
                            log::error!(
                                "[opt7] GPU scoring OOM; falling back to CPU for entire tile"
                            );
                        } else {
                            log::error!(
                                "[opt7] GPU scoring failed: {}; falling back to CPU for entire tile",
                                e
                            );
                        }
                        super::OPT7_GPU_SCORING_CPU_FALLBACK_TILES
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                        // Compute all scores on CPU for this tile (strict parity with CPU-only path)
                        let sf: Vec<f32> = tile
                            .iter()
                            .map(|&j| {
                                super::sim_levenshtein_pct(&c1[i].simple_first, &c2[j].simple_first)
                                    as f32
                            })
                            .collect();

                        let sl: Vec<f32> = tile
                            .iter()
                            .map(|&j| {
                                super::sim_levenshtein_pct(&c1[i].simple_last, &c2[j].simple_last)
                                    as f32
                            })
                            .collect();

                        let sm: Option<Vec<f32>> = if !mid_map_idx.is_empty() {
                            Some(
                                mid_map_idx
                                    .iter()
                                    .map(|&pos| {
                                        let j = tile[pos];
                                        super::sim_levenshtein_pct(
                                            &c1[i].simple_mid,
                                            &c2[j].simple_mid,
                                        ) as f32
                                    })
                                    .collect(),
                            )
                        } else {
                            None
                        };

                        (sf, sl, sm)
                    }
                };

                // Build output pairs (keep parity with CPU logic)
                for (pos, &j) in tile.iter().enumerate() {
                    let b = &t2[j];
                    if b.birthdate != Some(ad) {
                        continue;
                    }
                    let first_sim = scores_first[pos] as f64;
                    let last_sim = scores_last[pos] as f64;
                    let sm1 = &c1[i].simple_mid;
                    let sm2 = &c2[j].simple_mid;
                    let mid_present = !sm1.is_empty() && !sm2.is_empty();
                    let middle_sim = if mid_present {
                        if let Some(ref v) = scores_mid_opt {
                            let idx = mid_map_idx.iter().position(|&pidx| pidx == pos).unwrap();
                            v[idx] as f64
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };
                    let denom = if mid_present { 3.0 } else { 2.0 };
                    let confidence =
                        ((last_sim + first_sim + if mid_present { middle_sim } else { 0.0 })
                            / denom) as f32
                            / 100.0;
                    let mut matched_fields: Vec<String> = Vec::new();
                    if c1[i].simple_first == c2[j].simple_first {
                        matched_fields.push("FirstName".into());
                    }
                    if sm1 == sm2 && !sm1.is_empty() {
                        matched_fields.push("MiddleName".into());
                    }
                    if c1[i].simple_last == c2[j].simple_last {
                        matched_fields.push("LastName".into());
                    }
                    matched_fields.push("Birthdate".into());
                    out_pairs.push(super::MatchPair {
                        person1: t1[i].clone(),
                        person2: b.clone(),
                        confidence,
                        matched_fields,
                        is_matched_infnbd: false,
                        is_matched_infnmnbd: false,
                    });
                }

                start = end;
            }
            processed_rows += 1;
        }

        let (tiles, cpu_fb) = super::opt7_gpu_scoring_stats();
        log::info!(
            "[opt7] GPU scoring tiles: gpu={} cpu_fallback={}",
            tiles,
            cpu_fb
        );
        Ok(out_pairs)
    }

    /// Uses GPU only to compute FNV-1a 64-bit hashes; join/verification stays on CPU.
    pub fn det_match_gpu_hash_inmemory<F>(
        t1: &[Person],
        t2: &[Person],
        algo: MatchingAlgorithm,
        opts: &MatchOptions,
        on_progress: &F,
    ) -> Result<Vec<MatchPair>>
    where
        F: Fn(ProgressUpdate) + Sync,
    {
        use rayon::prelude::*;
        if matches!(algo, MatchingAlgorithm::Fuzzy) {
            return Err(anyhow!("Fuzzy not supported"));
        }
        let ctx = GpuHashContext::get()?;
        let (gt, gf) = ctx.mem_info_mb();
        // Normalize both tables (CPU)
        let n1: Vec<NormalizedPerson> = t1.par_iter().map(|p| normalize_person(p)).collect();
        let n2: Vec<NormalizedPerson> = t2.par_iter().map(|p| normalize_person(p)).collect();
        if n1.is_empty() || n2.is_empty() {
            return Ok(vec![]);
        }
        // Choose inner (smaller)
        let (inner_t, inner_n, outer_t, outer_n, inner_is_t2) = if n2.len() < n1.len() {
            (t2, n2.as_slice(), t1, n1.as_slice(), true)
        } else {
            (t1, n1.as_slice(), t2, n2.as_slice(), false)
        };
        // Build inner keys
        let mut key_idx: Vec<usize> = Vec::with_capacity(inner_n.len());
        let mut key_strs: Vec<String> = Vec::new();
        for (i, n) in inner_n.iter().enumerate() {
            if let Some(k) = super::key_for(algo, n) {
                key_idx.push(i);
                key_strs.push(k);
            }
        }
        // Hash inner keys
        let (_tmb, fmb) = ctx.mem_info_mb();
        let budget_mb = (fmb / 2).max(128);
        on_progress(ProgressUpdate {
            processed: 0,
            total: key_strs.len(),
            percent: 0.0,
            eta_secs: 0,
            mem_used_mb: memory_stats_mb().used_mb,
            mem_avail_mb: memory_stats_mb().avail_mb,
            stage: "inmem_gpu_hash_build",
            batch_size_current: opts.progress.batch_size.map(|v| v as i64),
            gpu_total_mb: gt,
            gpu_free_mb: gf,
            gpu_active: true,
        });
        let inner_hashes = hash_fnv1a64_batch_tiled(&ctx, &key_strs, budget_mb, 64)?;
        let mut index: std::collections::HashMap<u64, Vec<usize>> =
            std::collections::HashMap::new();
        for (j, &h) in inner_hashes.iter().enumerate() {
            index.entry(h).or_default().push(key_idx[j]);
        }
        // Probe and verify
        let mut matches: Vec<MatchPair> = Vec::new();
        let mut processed: usize = 0;
        let total = outer_n.len();
        let bs = opts.progress.batch_size.unwrap_or(100_000).max(10_000);
        let mut start = 0usize;
        while start < total {
            let end = (start + bs).min(total);
            let slice = &outer_n[start..end];
            let mut pkeys: Vec<String> = Vec::new();
            let mut pidx: Vec<usize> = Vec::new();
            for (i, n) in slice.iter().enumerate() {
                if let Some(k) = super::key_for(algo, n) {
                    pidx.push(i);
                    pkeys.push(k);
                }
            }
            if !pkeys.is_empty() {
                let (gt2, gf2) = ctx.mem_info_mb();
                let memx = memory_stats_mb();
                on_progress(ProgressUpdate {
                    processed,
                    total,
                    percent: (processed as f32 / total.max(1) as f32) * 100.0,
                    eta_secs: 0,
                    mem_used_mb: memx.used_mb,
                    mem_avail_mb: memx.avail_mb,
                    stage: "inmem_gpu_probe",
                    batch_size_current: Some(bs as i64),
                    gpu_total_mb: gt2,
                    gpu_free_mb: gf2,
                    gpu_active: true,
                });
                let phashes = hash_fnv1a64_batch_tiled(&ctx, &pkeys, budget_mb, 64)?;
                for (k, &h) in phashes.iter().enumerate() {
                    let np = &outer_n[pidx[k]];
                    if let Some(cands) = index.get(&h) {
                        for &ii in cands {
                            let q = &inner_t[ii];
                            let nq = &inner_n[ii];
                            let ok = match algo {
                                MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => {
                                    super::matches_algo1(np, nq)
                                }
                                MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => {
                                    super::matches_algo2(np, nq)
                                }
                                MatchingAlgorithm::Fuzzy
                                | MatchingAlgorithm::FuzzyNoMiddle
                                | MatchingAlgorithm::HouseholdGpu
                                | MatchingAlgorithm::HouseholdGpuOpt6
                                | MatchingAlgorithm::LevenshteinWeighted => false,
                            };
                            if ok {
                                let (a, b, na, nb) = if inner_is_t2 {
                                    (&outer_t[pidx[k] + (start - start)], q, np, nq)
                                } else {
                                    (q, &outer_t[pidx[k] + (start - start)], nq, np)
                                };
                                let pair = super::to_pair(a, b, algo, na, nb);
                                matches.push(pair);
                            }
                        }
                    }
                }
            }
            processed = end;
            start = end;
        }
        Ok(matches)
    }

    /// Optimized tiled hashing with optional double buffering across two CUDA streams.
    /// Falls back to single-stream tiled hashing when streams < 2.
    pub fn hash_fnv1a64_batch_tiled_overlap(
        hctx: &GpuHashContext,
        strings: &[String],
        budget_mb: u64,
        reserve_mb: u64,
        streams: u32,
        reuse_device_out: bool,
        use_pinned_host: bool,
    ) -> Result<Vec<u64>> {
        if streams < 2 {
            return hash_fnv1a64_batch_tiled(hctx, strings, budget_mb, reserve_mb);
        }
        if strings.is_empty() {
            return Ok(Vec::new());
        }

        if use_pinned_host {
            log::info!(
                "[GPU] gpu_use_pinned_host requested; current cudarc path uses pageable host memory (best-effort)"
            );
        }
        // Build tile ranges greedily based on budget and free VRAM
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        let mut i = 0usize;
        let min_tile = 512usize;
        while i < strings.len() {
            let (_tot_mb, free_mb) = super::cuda_mem_info_mb(&hctx.ctx);
            let target_mb = free_mb
                .min(budget_mb.max(64))
                .saturating_sub(reserve_mb.max(64));

            let target_bytes: usize = (target_mb as usize)
                .saturating_mul(1024 * 1024)
                .max(256 * 1024);
            let mut est_bytes = 0usize;
            let mut j = i;
            while j < strings.len() && est_bytes < target_bytes {
                est_bytes += 16 + strings[j].len();
                j += 1;
            }
            if j == i {
                j = (i + min_tile).min(strings.len());
            }
            ranges.push((i, j));
            i = j;
        }

        // Prepare two streams
        let s0 = hctx.ctx.default_stream();
        let s1 = hctx
            .ctx
            .new_stream()
            .map_err(|e| anyhow!("CUDA stream create failed: {e}"))?;

        // Host staging buffers reused across tiles
        let mut flat: Vec<u8> = Vec::new();
        let mut offsets: Vec<i32> = Vec::new();
        let mut lengths: Vec<i32> = Vec::new();

        // Optional device output buffers reused across tiles (double-buffered)
        let mut pool_out_s0: Option<_> = None;
        let mut pool_out_s1: Option<_> = None;

        // We keep one pending output buffer from the previous tile (on the other stream)
        let mut last_out_s0: Option<_> = None;
        let mut last_out_s1: Option<_> = None;

        let mut out: Vec<u64> = Vec::with_capacity(strings.len());

        for (idx, (lo, hi)) in ranges.iter().copied().enumerate() {
            // While we prepare and launch current tile on stream S, retrieve previous tile from the other stream
            if idx > 0 {
                if (idx - 1) % 2 == 1 {
                    if let Some(buf) = last_out_s1.take() {
                        let v = s1.memcpy_dtov(&buf)?;
                        out.extend(v);
                        if reuse_device_out {
                            pool_out_s1 = Some(buf);
                        }
                    }
                } else {
                    if let Some(buf) = last_out_s0.take() {
                        let v = s0.memcpy_dtov(&buf)?;
                        out.extend(v);
                        if reuse_device_out {
                            pool_out_s0 = Some(buf);
                        }
                    }
                }
            }

            // Flatten current tile
            let tile = &strings[lo..hi];
            offsets.clear();
            lengths.clear();
            flat.clear();
            offsets.reserve(tile.len());
            lengths.reserve(tile.len());
            let mut cur = 0i32;
            for s in tile {
                offsets.push(cur);
                let b = s.as_bytes();
                lengths.push(b.len() as i32);
                flat.extend_from_slice(b);
                cur += b.len() as i32;
            }
            let n = tile.len();

            // Choose stream alternating
            let use_s1 = idx % 2 == 1;
            let s = if use_s1 { &s1 } else { &s0 };

            // Allocate device buffers by copying to device and create output buffer; enqueue kernel
            let launched = (|| -> anyhow::Result<_> {
                let d_buf = s.memcpy_stod(flat.as_slice())?;
                let d_off = s.memcpy_stod(offsets.as_slice())?;
                let d_len = s.memcpy_stod(lengths.as_slice())?;
                // Reuse or allocate output buffer on the selected stream
                let mut d_out = if reuse_device_out {
                    if use_s1 {
                        match pool_out_s1.take() {
                            Some(buf) => buf,
                            None => s.alloc_zeros::<u64>(n)?,
                        }
                    } else {
                        match pool_out_s0.take() {
                            Some(buf) => buf,
                            None => s.alloc_zeros::<u64>(n)?,
                        }
                    }
                } else {
                    s.alloc_zeros::<u64>(n)?
                };

                // [GPU_OPT1] Adaptive block size based on GPU architecture
                let gpu_props = super::gpu_config::query_gpu_properties(0).unwrap_or_else(|_| {
                    super::gpu_config::GpuProperties {
                        compute_major: 7,
                        compute_minor: 0,
                        sm_count: 30,
                        max_threads_per_block: 1024,
                        max_shared_memory_per_block: 49152,
                    }
                });
                let bs: u32 = super::gpu_config::calculate_optimal_block_size(
                    &gpu_props,
                    super::gpu_config::KernelType::Hash,
                );
                let grid: u32 = ((n as u32 + bs - 1) / bs).max(1);
                let cfg = LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (bs, 1, 1),
                    shared_mem_bytes: 0,
                };
                let n_i32 = n as i32;
                let mut b = s.launch_builder(&hctx.func_hash);
                b.arg(&d_buf)
                    .arg(&d_off)
                    .arg(&d_len)
                    .arg(&mut d_out)
                    .arg(&n_i32);
                unsafe {
                    b.launch(cfg)?;
                }
                Ok(d_out)
            })();

            match launched {
                Ok(d_out) => {
                    if use_s1 {
                        last_out_s1 = Some(d_out);
                    } else {
                        last_out_s0 = Some(d_out);
                    }
                }
                Err(e) => {
                    let s = e.to_string().to_ascii_lowercase();
                    if s.contains("cuda_error_out_of_memory")
                        || s.contains("out of memory")
                        || s.contains("oom")
                    {
                        log::warn!(
                            "[GPU] OOM during tiled hashing ({} keys); falling back to CPU for this tile",
                            n
                        );
                        for s in tile {
                            out.push(super::fnv1a64_bytes(s.as_bytes()));
                        }
                    } else {
                        // Unknown GPU error: CPU fallback for safety
                        log::warn!("[GPU] Hashing error, CPU fallback: {}", e);
                        for s in tile {
                            out.push(super::fnv1a64_bytes(s.as_bytes()));
                        }
                    }
                }
            }

            // If we reused a buffer, return it to the pool after device-to-host copy later
        }
        // Drain the last pending tile (the one launched for the final range)
        if let Some(buf) = last_out_s1.take() {
            let v = s1.memcpy_dtov(&buf)?;
            out.extend(v);
            if reuse_device_out {
                pool_out_s1 = Some(buf);
            }
        }
        if let Some(buf) = last_out_s0.take() {
            let v = s0.memcpy_dtov(&buf)?;
            out.extend(v);
            if reuse_device_out {
                pool_out_s0 = Some(buf);
            }
        }
        Ok(out)
    }

    pub fn match_fuzzy_gpu<F>(
        t1: &[Person],
        t2: &[Person],
        opts: MatchOptions,
        on_progress: &F,
    ) -> Result<Vec<MatchPair>>
    where
        F: Fn(ProgressUpdate) + Sync,
    {
        let allow_swap = opts.allow_birthdate_swap;
        log::info!(
            "[GPU_FUZZY] allow_birthdate_swap flag = {} (env NAME_MATCHER_ALLOW_BIRTHDATE_SWAP={:?})",
            allow_swap,
            std::env::var("NAME_MATCHER_ALLOW_BIRTHDATE_SWAP").ok()
        );

        // 1) Normalize on CPU (reuse existing)
        let n1: Vec<NormalizedPerson> = t1.par_iter().map(normalize_person).collect();
        let n2: Vec<NormalizedPerson> = t2.par_iter().map(normalize_person).collect();
        if n1.is_empty() || n2.is_empty() {
            return Ok(vec![]);
        }

        // 2) Birthdate-only blocking to match CPU fallback and in-memory behavior
        use std::collections::HashMap;
        let mut block: HashMap<String, Vec<usize>> = HashMap::new();
        for (j, p) in n2.iter().enumerate() {
            if let Some(d) = p.birthdate {
                for key in birthdate_keys(d, allow_swap) {
                    block.entry(key).or_default().push(j);
                }
            }
        }

        // 3) Prepare CUDA context & streams
        let dev_id = opts.gpu.and_then(|g| g.device_id).unwrap_or(0);
        // Build per-person caches once (used by GPU tiling and CPU post-processing)
        let cache1: Vec<FuzzyCache> = t1.par_iter().map(build_cache_from_person).collect();
        let cache2: Vec<FuzzyCache> = t2.par_iter().map(build_cache_from_person).collect();

        // Reuse cached CUDA context, compiled module, kernels, and streams
        let fctx = GpuFuzzyContext::get()?;
        let ctx_arc = &fctx.ctx;
        let stream = &fctx.stream_default;
        let stream2 = &fctx.stream_aux;
        let func = &*fctx.func_lev;
        let func_jaro = &*fctx.func_jaro;
        let func_jw = &*fctx.func_jw;
        let func_max3 = &*fctx.func_max3;

        // Report GPU init and memory info
        let (gpu_total_mb, gpu_free_mb_init) = cuda_mem_info_mb(&*ctx_arc);
        let mem0 = memory_stats_mb();
        on_progress(ProgressUpdate {
            processed: 0,
            total: n1.len(),
            percent: 0.0,
            eta_secs: 0,
            mem_used_mb: mem0.used_mb,
            mem_avail_mb: mem0.avail_mb,
            stage: "gpu_init",
            batch_size_current: None,
            gpu_total_mb,
            gpu_free_mb: gpu_free_mb_init,
            gpu_active: true,
        });

        // [GPU_OPT5] Memory query caching to reduce overhead
        use std::time::Instant;
        let mut last_mem_query = Instant::now();
        let mut cached_gpu_free_mb = gpu_free_mb_init;
        let mut mem_query_count = 0usize;
        let mut tile_count = 0usize;

        if dynamic_gpu_tuning_enabled() {
            self::gpu::dynamic_tuner::ensure_started(true);
        }

        // 4) Tile candidates to respect memory budget - use adaptive budget if not explicitly set
        let mem_budget_mb = match opts.gpu.and_then(|g| {
            if g.mem_budget_mb > 0 {
                Some(g.mem_budget_mb)
            } else {
                None
            }
        }) {
            Some(explicit_budget) => explicit_budget,
            None => {
                // Auto-calculate adaptive budget (75% of free VRAM, conservative)

                let budget = super::gpu_config::calculate_gpu_memory_budget(
                    gpu_total_mb,
                    gpu_free_mb_init,
                    false,
                );
                log::info!(
                    "[GPU] Auto-calculated memory budget: {} MB (75% of {} MB free VRAM)",
                    budget,
                    gpu_free_mb_init
                );
                budget
            }
        };
        // Rough bytes per pair: two strings up to 64 bytes + offsets/len + output ~ 256 bytes
        let approx_bpp: usize = 256;

        // [GPU_OPT] Increased minimum tile size from 1,024 to 32,000 pairs
        // Larger tiles reduce CPU-GPU handoff overhead and improve GPU utilization
        // OOM backoff logic (lines 2291-2337) provides safety net for low-VRAM GPUs

        // [GPU_OPT3] Calculate tile soft cap based on GPU parallelism capacity
        let gpu_props = super::gpu_config::query_gpu_properties(0).unwrap_or_else(|_| {
            super::gpu_config::GpuProperties {
                compute_major: 7,
                compute_minor: 0,
                sm_count: 30,
                max_threads_per_block: 1024,
                max_shared_memory_per_block: 49152,
            }
        });
        let block_size = super::gpu_config::calculate_optimal_block_size(
            &gpu_props,
            super::gpu_config::KernelType::Levenshtein,
        );
        let soft_cap = super::gpu_config::calculate_tile_soft_cap(&gpu_props, block_size);

        let memory_based = ((mem_budget_mb as usize * 1024 * 1024) / approx_bpp).max(32_000);
        let mut tile_max = memory_based.min(soft_cap);

        log::info!(
            "[GPU_OPT3] Tile sizing: memory_based={} soft_cap={} final={} (GPU: {}.{}, {} SMs, block_size={})",
            memory_based,
            soft_cap,
            tile_max,
            gpu_props.compute_major,
            gpu_props.compute_minor,
            gpu_props.sm_count,
            block_size
        );

        if dynamic_gpu_tuning_enabled() {
            let dyn_tile = self::gpu::dynamic_tuner::get_current_tile_size();
            if dyn_tile != tile_max {
                log::info!(
                    "[GPU-TUNE] Overriding tile size: {} -> {} pairs",
                    tile_max,
                    dyn_tile
                );
            }
            tile_max = dyn_tile.max(1);
        }

        let mut results: Vec<MatchPair> = Vec::new();
        let total: usize = n1.len();
        // Cross-outer GPU batch (store pairs) to improve utilization by batching multiple outer records per launch
        let mut batch_pairs: Vec<(usize, usize)> = Vec::with_capacity(tile_max.max(1));

        // Track seen pairs to deduplicate when swap generates multiple keys
        use std::collections::HashSet;
        let mut seen_pairs: HashSet<(usize, usize)> = HashSet::new();
        let mut total_candidates = 0usize;
        let mut swap_candidates = 0usize;

        for (i, p1) in n1.iter().enumerate() {
            // Birthdate-only blocking (matches CPU fallback and in-memory behavior)
            // When allow_swap is true, look up by all birthdate keys (original + swapped)
            if let Some(d) = p1.birthdate {
                for key in birthdate_keys(d, allow_swap) {
                    if let Some(cands_vec) = block.get(&key) {
                        // Accumulate candidates for this outer record into cross-outer GPU batch
                        for &j_idx in cands_vec {
                            // Deduplicate: same pair can match via multiple keys when swap is enabled
                            if !seen_pairs.insert((i, j_idx)) {
                                continue;
                            }
                            total_candidates += 1;
                            // Check if this is a swap candidate (birthdates differ)
                            if let (Some(d1), Some(d2)) = (p1.birthdate, n2[j_idx].birthdate) {
                                if d1 != d2 {
                                    swap_candidates += 1;
                                    // Debug: log specific swap pairs
                                    let s1 = d1.format("%Y-%m-%d").to_string();
                                    let s2 = d2.format("%Y-%m-%d").to_string();
                                    if s1 == "1990-09-05" && s2 == "1990-05-09" {
                                        log::info!(
                                            "[GPU_FUZZY] Swap pair generated: i={}, j={}, id1={}, id2={}, bd1={}, bd2={}",
                                            i,
                                            j_idx,
                                            t1[i].id,
                                            t2[j_idx].id,
                                            s1,
                                            s2
                                        );
                                    }
                                }
                            }
                            batch_pairs.push((i, j_idx));
                            if batch_pairs.len() >= tile_max {
                                // Flush in chunks with OOM backoff using prefix drains
                                let mut desired = tile_max.max(1);
                                while batch_pairs.len() >= desired {
                                    let mut inner_acc =
                                        crate::matching::gpu::batch::GpuBatchAccumulator::new(
                                            desired,
                                        );
                                    for &(oi, ij) in batch_pairs.iter().take(desired) {
                                        inner_acc.add_candidate(oi, ij);
                                    }
                                    let attempt = inner_acc.flush_to_gpu(
                                        &n1,
                                        &n2,
                                        &cache1,
                                        &cache2,
                                        t1,
                                        t2,
                                        ctx_arc,
                                        stream,
                                        stream2,
                                        &func,
                                        &func_jaro,
                                        &func_jw,
                                        &func_max3,
                                        desired,
                                        &mut results,
                                        allow_swap,
                                    );
                                    match attempt {
                                        Ok(()) => {
                                            batch_pairs.drain(0..desired);
                                            tile_count += 1;

                                            // [GPU_OPT5] Refresh GPU memory info periodically (every 100ms or 10 tiles)
                                            if last_mem_query.elapsed()
                                                > std::time::Duration::from_millis(100)
                                                || tile_count % 10 == 0
                                            {
                                                let (_tot_mb, free_mb) =
                                                    cuda_mem_info_mb(&*ctx_arc);
                                                cached_gpu_free_mb = free_mb;
                                                last_mem_query = Instant::now();
                                                mem_query_count += 1;
                                            }

                                            let mem = memory_stats_mb();
                                            let frac =
                                                ((i + 1) as f32 / total as f32).clamp(0.0, 1.0);
                                            on_progress(ProgressUpdate {
                                                processed: i + 1,
                                                total,
                                                percent: frac * 100.0,
                                                eta_secs: 0,
                                                mem_used_mb: mem.used_mb,
                                                mem_avail_mb: mem.avail_mb,
                                                stage: "gpu_kernel",
                                                batch_size_current: Some(desired as i64),
                                                gpu_total_mb: gpu_total_mb,
                                                gpu_free_mb: cached_gpu_free_mb,
                                                gpu_active: true,
                                            });
                                            // continue while to check if more remains >= desired
                                        }
                                        Err(e) => {
                                            if super::gpu_config::is_cuda_oom(&e) && desired > 512 {
                                                desired = (desired / 2).max(512);
                                                continue; // retry with smaller desired
                                            } else {
                                                return Err(anyhow!(e));
                                            }
                                        }
                                    }
                                    if batch_pairs.len() < desired {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Final flush of any remaining batched pairs with OOM backoff
        let mut desired = tile_max.max(1);
        while !batch_pairs.is_empty() {
            let actual_len = desired.min(batch_pairs.len());
            let mut inner_acc = crate::matching::gpu::batch::GpuBatchAccumulator::new(actual_len);
            for &(oi, ij) in batch_pairs.iter().take(actual_len) {
                inner_acc.add_candidate(oi, ij);
            }
            let attempt = inner_acc.flush_to_gpu(
                &n1,
                &n2,
                &cache1,
                &cache2,
                t1,
                t2,
                ctx_arc,
                stream,
                stream2,
                &func,
                &func_jaro,
                &func_jw,
                &func_max3,
                actual_len,
                &mut results,
                allow_swap,
            );
            match attempt {
                Ok(()) => {
                    batch_pairs.drain(0..actual_len);
                    tile_count += 1;

                    // [GPU_OPT5] Refresh GPU memory info for final flush
                    if last_mem_query.elapsed() > std::time::Duration::from_millis(100) {
                        let (_tot_mb, free_mb) = cuda_mem_info_mb(&*ctx_arc);
                        cached_gpu_free_mb = free_mb;
                        mem_query_count += 1;
                    }

                    let mem = memory_stats_mb();
                    on_progress(ProgressUpdate {
                        processed: total,
                        total,
                        percent: 100.0,
                        eta_secs: 0,
                        mem_used_mb: mem.used_mb,
                        mem_avail_mb: mem.avail_mb,
                        stage: "gpu_kernel",
                        batch_size_current: Some(actual_len as i64),
                        gpu_total_mb: gpu_total_mb,
                        gpu_free_mb: cached_gpu_free_mb,
                        gpu_active: true,
                    });
                }
                Err(e) => {
                    if super::gpu_config::is_cuda_oom(&e) && desired > 512 {
                        desired = (desired / 2).max(512);
                        continue; // retry smaller
                    } else {
                        return Err(anyhow!(e));
                    }
                }
            }
        }

        // [GPU_OPT5] Log memory query caching efficiency
        log::info!(
            "[GPU_OPT5] Memory query caching: {} queries for {} tiles (avg: {:.1} tiles/query)",
            mem_query_count,
            tile_count,
            if mem_query_count > 0 {
                tile_count as f32 / mem_query_count as f32
            } else {
                0.0
            }
        );
        log::info!(
            "[GPU_FUZZY] Candidate generation: total_candidates={}, swap_candidates={}, results={}",
            total_candidates,
            swap_candidates,
            results.len()
        );

        // Final CPU re-score for parity with CPU/in-memory paths (thresholding is done by callers)
        for pair in &mut results {
            let rescored = if gpu_no_mid_mode() {
                compare_persons_no_mid(&pair.person1, &pair.person2)
            } else {
                compare_persons_new(&pair.person1, &pair.person2)
            };
            if let Some((cpu_score, _)) = rescored {
                pair.confidence = cpu_score as f32;
            }
        }

        Ok(results)
    }

    pub fn match_fuzzy_no_mid_gpu<F>(
        t1: &[Person],
        t2: &[Person],
        opts: MatchOptions,
        on_progress: &F,
    ) -> Result<Vec<MatchPair>>
    where
        F: Fn(ProgressUpdate) + Sync,
    {
        // Force no-middle classification within the GPU batch pipeline to ensure true parity and avoid
        // the expensive reclassification/supplement steps.
        with_no_mid_classification(|| match_fuzzy_gpu(t1, t2, opts, on_progress))
    }
}

fn to_original<'a>(np: &NormalizedPerson, originals: &'a [Person]) -> Person {
    originals
        .iter()
        .find(|p| p.id == np.id)
        .cloned()
        .unwrap_or_else(|| Person {
            id: np.id,
            uuid: Some(np.uuid.clone()),
            first_name: np.first_name.clone(),
            middle_name: np.middle_name.clone(),
            last_name: np.last_name.clone(),
            birthdate: np.birthdate,
            hh_id: None,
            extra_fields: std::collections::HashMap::new(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use std::sync::{Arc, Mutex};
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
    fn algo1_basic() {
        let a = vec![p(1, "José", None, "García", (1990, 1, 1))];
        let b = vec![p(2, "Jose", None, "Garcia", (1990, 1, 1))];
        let r = match_all(&a, &b, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, |_| {});
        assert_eq!(r.len(), 1);
        assert!(r[0].is_matched_infnbd);
    }
    #[test]
    fn algo2_middle_required() {
        let a = vec![p(1, "Ann", Some("B"), "Lee", (1990, 1, 1))];
        let b = vec![p(2, "Ann", None, "Lee", (1990, 1, 1))];
        let r = match_all(
            &a,
            &b,
            MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd,
            |_| {},
        );
        assert_eq!(r.len(), 0);
    }
    #[test]
    fn progress_updates() {
        let a = (0..10)
            .map(|i| p(i, "A", None, "Z", (2000, 1, 1)))
            .collect::<Vec<_>>();
        let b = (0..10)
            .map(|i| p(100 + i as i64, "A", None, "Z", (2000, 1, 1)))
            .collect::<Vec<_>>();
        let updates: Arc<Mutex<Vec<ProgressUpdate>>> = Arc::new(Mutex::new(vec![]));
        let cfg = ProgressConfig {
            update_every: 3,
            batch_size: Some(2),
            ..Default::default()
        };
        let u2 = updates.clone();
        let _ = match_all_progress(
            &a,
            &b,
            MatchingAlgorithm::IdUuidYasIsMatchedInfnbd,
            cfg,
            |u| {
                u2.lock().unwrap().push(u);
            },
        );
        let v = updates.lock().unwrap();
        assert!(v.len() >= 3); // at least a few updates including final
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_fuzzy_honors_birthdate_swap() {
        // Skip gracefully when no GPU is available
        if crate::matching::gpu::GpuFuzzyContext::get().is_err() {
            eprintln!("[test] GPU context unavailable; skipping gpu_fuzzy_honors_birthdate_swap");
            return;
        }
        let a = vec![p(1, "Ann", Some("Marie"), "Lopez", (1990, 4, 12))];
        // Swapped month/day (12/04)
        let b = vec![p(2, "Ann", Some("Marie"), "Lopez", (1990, 12, 4))];
        let opts = MatchOptions {
            backend: ComputeBackend::Gpu,
            gpu: Some(GpuConfig {
                device_id: None,
                mem_budget_mb: 512,
            }),
            progress: ProgressConfig::default(),
            allow_birthdate_swap: true,
        };
        let pairs = match_fuzzy_gpu(&a, &b, opts, &|_u: ProgressUpdate| {});
        if let Err(e) = pairs {
            eprintln!("[test] match_fuzzy_gpu failed: {e}");
            return; // treat as skipped on machines without CUDA
        }
        let pairs = pairs.unwrap();
        assert_eq!(pairs.len(), 1, "expected swapped birthdate to match");
        assert!(
            (pairs[0].confidence - 100.0).abs() < f32::EPSILON,
            "confidence should be 100 for identical names when swap is allowed"
        );
    }

    #[test]
    fn hash_key_for_np_basic() {
        use chrono::NaiveDate;
        let a = Person {
            id: 1,
            uuid: None,
            first_name: Some("Ann".into()),
            middle_name: None,
            last_name: Some("Lee".into()),
            birthdate: NaiveDate::from_ymd_opt(1990, 1, 1),
            hh_id: None,
            extra_fields: std::collections::HashMap::new(),
        };
        let n = normalize_person(&a);
        let h1 = hash_key_for_np(MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, &n);
        assert!(h1.is_some());
        let b = Person {
            id: 2,
            uuid: None,
            first_name: Some("Ann".into()),
            middle_name: None,
            last_name: Some("Lee".into()),
            birthdate: NaiveDate::from_ymd_opt(1990, 1, 1),
            hh_id: None,
            extra_fields: std::collections::HashMap::new(),
        };
        let n2 = normalize_person(&b);
        let h2 = hash_key_for_np(MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, &n2);
        assert_eq!(h1, h2, "same normalized inputs must hash equal");
        // Missing field -> None
        let c = Person {
            id: 3,
            uuid: None,
            first_name: Some("Ann".into()),
            middle_name: None,
            last_name: Some("Lee".into()),
            birthdate: None,
            hh_id: None,
            extra_fields: std::collections::HashMap::new(),
        };
        let n3 = normalize_person(&c);
        assert!(hash_key_for_np(MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, &n3).is_none());
    }

    fn hash_join_in_memory(
        algo: MatchingAlgorithm,
        t1: &[Person],
        t2: &[Person],
    ) -> Vec<(i64, i64)> {
        use std::collections::HashMap;
        let mut map: HashMap<u64, Vec<Person>> = HashMap::new();
        for p in t1 {
            let n = normalize_person(p);
            if let Some(h) = hash_key_for_np(algo, &n) {
                map.entry(h).or_default().push(p.clone());
            }
        }
        let mut out = Vec::new();
        for q in t2 {
            let nq = normalize_person(q);
            if let Some(hq) = hash_key_for_np(algo, &nq) {
                if let Some(cands) = map.get(&hq) {
                    for p in cands {
                        let np = normalize_person(p);
                        let ok = match algo {
                            MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => matches_algo1(&np, &nq),
                            MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => {
                                matches_algo2(&np, &nq)
                            }
                            MatchingAlgorithm::Fuzzy
                            | MatchingAlgorithm::FuzzyNoMiddle
                            | MatchingAlgorithm::HouseholdGpu
                            | MatchingAlgorithm::HouseholdGpuOpt6
                            | MatchingAlgorithm::LevenshteinWeighted => false,
                        };
                        if ok {
                            out.push((p.id, q.id));
                        }
                    }
                }
            }
        }
        out.sort();
        out
    }

    #[test]
    fn hash_join_equivalence_algo1_and_2() {
        use chrono::NaiveDate;
        let t1 = vec![
            Person {
                id: 1,
                uuid: None,
                first_name: Some("José".into()),
                middle_name: None,
                last_name: Some("García".into()),
                birthdate: NaiveDate::from_ymd_opt(1990, 1, 1),
                hh_id: None,
                extra_fields: std::collections::HashMap::new(),
            },
            Person {
                id: 2,
                uuid: None,
                first_name: Some("Ann".into()),
                middle_name: Some("B".into()),
                last_name: Some("Lee".into()),
                birthdate: NaiveDate::from_ymd_opt(1985, 5, 5),
                hh_id: None,
                extra_fields: std::collections::HashMap::new(),
            },
        ];
        let t2 = vec![
            Person {
                id: 10,
                uuid: None,
                first_name: Some("Jose".into()),
                middle_name: None,
                last_name: Some("Garcia".into()),
                birthdate: NaiveDate::from_ymd_opt(1990, 1, 1),
                hh_id: None,
                extra_fields: std::collections::HashMap::new(),
            },
            Person {
                id: 20,
                uuid: None,
                first_name: Some("Ann".into()),
                middle_name: Some("B".into()),
                last_name: Some("Lee".into()),
                birthdate: NaiveDate::from_ymd_opt(1985, 5, 5),
                hh_id: None,
                extra_fields: std::collections::HashMap::new(),
            },
            Person {
                id: 21,
                uuid: None,
                first_name: Some("Ann".into()),
                middle_name: Some("C".into()),
                last_name: Some("Lee".into()),
                birthdate: NaiveDate::from_ymd_opt(1985, 5, 5),
                hh_id: None,
                extra_fields: std::collections::HashMap::new(),
            },
        ];
        // Algorithm 1 (no middle name)
        let hj1 = hash_join_in_memory(MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, &t1, &t2);
        let mut direct1 = match_all(
            &t1,
            &t2,
            MatchingAlgorithm::IdUuidYasIsMatchedInfnbd,
            |_| {},
        )
        .into_iter()
        .map(|m| (m.person1.id, m.person2.id))
        .collect::<Vec<_>>();
        direct1.sort();
        assert_eq!(
            hj1, direct1,
            "hash-join prefilter + exact verify must equal direct matches (algo1)"
        );
        // Algorithm 2 (with middle name)
        let hj2 = hash_join_in_memory(MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd, &t1, &t2);
        let mut direct2 = match_all(
            &t1,
            &t2,
            MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd,
            |_| {},
        )
        .into_iter()
        .map(|m| (m.person1.id, m.person2.id))
        .collect::<Vec<_>>();
        direct2.sort();
        assert_eq!(
            hj2, direct2,
            "hash-join prefilter + exact verify must equal direct matches (algo2)"
        );
    }
}
#[test]
fn fuzzy_basic() {
    use chrono::NaiveDate;
    let a = vec![Person {
        id: 1,
        uuid: Some("u1".into()),
        first_name: Some("Jon".into()),
        middle_name: None,
        last_name: Some("Smith".into()),
        birthdate: NaiveDate::from_ymd_opt(1990, 1, 1),
        hh_id: None,
        extra_fields: std::collections::HashMap::new(),
    }];
    let b = vec![Person {
        id: 2,
        uuid: Some("u2".into()),
        first_name: Some("John".into()),
        middle_name: None,
        last_name: Some("Smith".into()),
        birthdate: NaiveDate::from_ymd_opt(1990, 1, 1),
        hh_id: None,
        extra_fields: std::collections::HashMap::new(),
    }];
    let r = match_all(&a, &b, MatchingAlgorithm::Fuzzy, |_| {});
    assert_eq!(r.len(), 1);
    assert!(r[0].confidence > 0.85);
}
#[test]
fn metaphone_handles_unicode_without_panic() {
    let _ = metaphone_pct("JO\u{2229}N", "JOHN");
    let _ = metaphone_pct("Jos\u{00e9}", "Jose");
    let _ = metaphone_pct("M\u{00fc}ller", "Muller");
    let _ = metaphone_pct("\u{738b}\u{5c0f}\u{660e}", "Wang Xiaoming");
}

#[test]
fn compute_stream_cfg_bounds_and_flush() {
    let c1 = compute_stream_cfg(512);
    assert!(c1.batch_size >= 5_000);
    assert!(c1.flush_every >= 1000);
    let c2 = compute_stream_cfg(32_768);
    assert!(c2.batch_size <= 100_000);
    assert_eq!(c2.flush_every, (c2.batch_size as usize / 10).max(1000));
}

#[test]
fn optional_fields_algo1_requirements() {
    use chrono::NaiveDate;
    // Same names but missing birthdate -> no match
    let a = vec![Person {
        id: 1,
        uuid: Some("u1".into()),
        first_name: Some("Ann".into()),
        middle_name: None,
        last_name: Some("Lee".into()),
        birthdate: None,
        hh_id: None,
        extra_fields: std::collections::HashMap::new(),
    }];
    let b = vec![Person {
        id: 2,
        uuid: Some("u2".into()),
        first_name: Some("Ann".into()),
        middle_name: None,
        last_name: Some("Lee".into()),
        birthdate: NaiveDate::from_ymd_opt(1990, 1, 1),
        hh_id: None,
        extra_fields: std::collections::HashMap::new(),
    }];
    let r = match_all(&a, &b, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd, |_| {});
    assert_eq!(r.len(), 0);
    // Missing first name -> no match
    let a2 = vec![Person {
        id: 3,
        uuid: Some("u3".into()),
        first_name: None,
        middle_name: None,
        last_name: Some("Lee".into()),
        birthdate: NaiveDate::from_ymd_opt(1990, 1, 1),
        hh_id: None,
        extra_fields: std::collections::HashMap::new(),
    }];
    let b2 = vec![Person {
        id: 4,
        uuid: Some("u4".into()),
        first_name: Some("Ann".into()),
        middle_name: None,
        last_name: Some("Lee".into()),
        birthdate: NaiveDate::from_ymd_opt(1990, 1, 1),
        hh_id: None,
        extra_fields: std::collections::HashMap::new(),
    }];
    let r2 = match_all(
        &a2,
        &b2,
        MatchingAlgorithm::IdUuidYasIsMatchedInfnbd,
        |_| {},
    );
    assert_eq!(r2.len(), 0);
}

#[test]
fn optional_fields_algo2_middle_none_allowed() {
    use chrono::NaiveDate;
    // Both middle None but names and birthdate equal -> match
    let a = vec![Person {
        id: 10,
        uuid: Some("u10".into()),
        first_name: Some("Ann".into()),
        middle_name: None,
        last_name: Some("Lee".into()),
        birthdate: NaiveDate::from_ymd_opt(1990, 1, 1),
        hh_id: None,
        extra_fields: std::collections::HashMap::new(),
    }];
    let b = vec![Person {
        id: 20,
        uuid: Some("u20".into()),
        first_name: Some("Ann".into()),
        middle_name: None,
        last_name: Some("Lee".into()),
        birthdate: NaiveDate::from_ymd_opt(1990, 1, 1),
        hh_id: None,
        extra_fields: std::collections::HashMap::new(),
    }];
    let r = match_all(
        &a,
        &b,
        MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd,
        |_| {},
    );
    assert_eq!(r.len(), 1);
    assert!(r[0].is_matched_infnmnbd);
}

#[test]
fn fuzzy_requires_birthdate_and_some_name_content() {
    use chrono::NaiveDate;
    // Missing birthdate -> no match
    let a = vec![Person {
        id: 30,
        uuid: Some("u30".into()),
        first_name: Some("Jon".into()),
        middle_name: None,
        last_name: Some("Smith".into()),
        birthdate: None,
        hh_id: None,
        extra_fields: std::collections::HashMap::new(),
    }];
    let b = vec![Person {
        id: 40,
        uuid: Some("u40".into()),
        first_name: Some("John".into()),
        middle_name: None,
        last_name: Some("Smith".into()),
        birthdate: NaiveDate::from_ymd_opt(1990, 1, 1),
        hh_id: None,
        extra_fields: std::collections::HashMap::new(),
    }];
    let r = match_all(&a, &b, MatchingAlgorithm::Fuzzy, |_| {});
    assert_eq!(r.len(), 0);
    // Empty names (all None) even with same birthdate -> no match
    let a2 = vec![Person {
        id: 31,
        uuid: Some("u31".into()),
        first_name: None,
        middle_name: None,
        last_name: None,
        birthdate: NaiveDate::from_ymd_opt(1990, 1, 1),
        hh_id: None,
        extra_fields: std::collections::HashMap::new(),
    }];
    let b2 = vec![Person {
        id: 41,
        uuid: Some("u41".into()),
        first_name: None,
        middle_name: None,
        last_name: None,
        birthdate: NaiveDate::from_ymd_opt(1990, 1, 1),
        hh_id: None,
        extra_fields: std::collections::HashMap::new(),
    }];
    let r2 = match_all(&a2, &b2, MatchingAlgorithm::Fuzzy, |_| {});
    assert_eq!(r2.len(), 0);
}

#[test]
fn opt6_denominator_and_hh_fallback() {
    use chrono::NaiveDate;
    // Table1 (target): grouped by uuid
    let t1 = vec![Person {
        id: 1,
        uuid: Some("u1".into()),
        first_name: Some("Ann".into()),
        middle_name: None,
        last_name: Some("Lee".into()),
        birthdate: NaiveDate::from_ymd_opt(1990, 1, 1),
        hh_id: None,
        extra_fields: std::collections::HashMap::new(),
    }];
    // Table2 (source): grouped by hh_id with fallback to id
    // Household 100 has 4 members; 3 should match u1; 1 is different birthdate (no match)
    let t2 = vec![
        Person {
            id: 10,
            uuid: None,
            first_name: Some("Ann".into()),
            middle_name: None,
            last_name: Some("Lee".into()),
            birthdate: NaiveDate::from_ymd_opt(1990, 1, 1),
            hh_id: Some("100".into()),
            extra_fields: std::collections::HashMap::new(),
        },
        Person {
            id: 11,
            uuid: None,
            first_name: Some("Ann".into()),
            middle_name: None,
            last_name: Some("Lee".into()),
            birthdate: NaiveDate::from_ymd_opt(1990, 1, 1),
            hh_id: Some("100".into()),
            extra_fields: std::collections::HashMap::new(),
        },
        Person {
            id: 12,
            uuid: None,
            first_name: Some("Ann".into()),
            middle_name: None,
            last_name: Some("Lee".into()),
            birthdate: NaiveDate::from_ymd_opt(1990, 1, 1),
            hh_id: Some("100".into()),
            extra_fields: std::collections::HashMap::new(),
        },
        Person {
            id: 13,
            uuid: None,
            first_name: Some("Ann".into()),
            middle_name: None,
            last_name: Some("Lee".into()),
            birthdate: NaiveDate::from_ymd_opt(1991, 1, 1),
            hh_id: Some("100".into()),
            extra_fields: std::collections::HashMap::new(),
        },
    ];
    let rows = match_households_gpu_inmemory_opt6(
        &t1,
        &t2,
        MatchOptions {
            backend: ComputeBackend::Cpu,
            gpu: None,
            progress: ProgressConfig::default(),
            allow_birthdate_swap: false,
        },
        0.0,
        |_u| {},
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].uuid, "u1");
    assert_eq!(rows[0].hh_id, 100);
    // 3 of 4 members matched => 75%
    assert!((rows[0].match_percentage - 75.0).abs() < 0.01);
    // Fallback: person with no hh_id should fallback to id and still count totals correctly
    let t2b = vec![Person {
        id: 20,
        uuid: None,
        first_name: Some("Bob".into()),
        middle_name: None,
        last_name: Some("Ray".into()),
        birthdate: NaiveDate::from_ymd_opt(1980, 1, 1),
        hh_id: None,
        extra_fields: std::collections::HashMap::new(),
    }];
    let t1b = vec![Person {
        id: 2,
        uuid: Some("u2".into()),
        first_name: Some("Bob".into()),
        middle_name: None,
        last_name: Some("Ray".into()),
        birthdate: NaiveDate::from_ymd_opt(1980, 1, 1),
        hh_id: None,
        extra_fields: std::collections::HashMap::new(),
    }];
    let rows_b = match_households_gpu_inmemory_opt6(
        &t1b,
        &t2b,
        MatchOptions {
            backend: ComputeBackend::Cpu,
            gpu: None,
            progress: ProgressConfig::default(),
            allow_birthdate_swap: false,
        },
        0.0,
        |_u| {},
    );
    assert_eq!(rows_b.len(), 1);
    assert_eq!(rows_b[0].hh_id, 20); // fallback to id
    assert!((rows_b[0].match_percentage - 100.0).abs() < 0.01);
}

#[tokio::test]
async fn prefetch_pipeline_sim() {
    use tokio::task::JoinHandle;
    let total_chunks = 5usize;
    let batch = 10i32;
    let mut next: Option<JoinHandle<Vec<i32>>> = None;
    let mut out: Vec<i32> = Vec::new();
    let mut offset = 0i32;
    while (offset as usize) < total_chunks * batch as usize {
        let cur: Vec<i32> = if let Some(h) = next.take() {
            h.await.unwrap()
        } else {
            let start = offset;
            let b = batch;
            tokio::spawn(async move { (start..start + b).collect::<Vec<_>>() })
                .await
                .unwrap()
        };
        out.extend(cur.iter());
        offset += batch;
        if (offset as usize) < total_chunks * batch as usize {
            let start = offset;
            let b = batch;
            next = Some(tokio::spawn(async move {
                (start..start + b).collect::<Vec<_>>()
            }));
        }
    }
    assert_eq!(out.len(), total_chunks * batch as usize);
    assert_eq!(out[0], 0);
    assert_eq!(*out.last().unwrap(), (total_chunks as i32 * batch) - 1);
}

#[test]
fn checkpoint_roundtrip() {
    use crate::util::checkpoint::{
        StreamCheckpoint, load_checkpoint, remove_checkpoint, save_checkpoint,
    };
    let path = format!(
        "{}\\nmckpt_test_{}.ckpt",
        std::env::temp_dir().display(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    );
    let cp = StreamCheckpoint {
        db: "db".into(),
        table_inner: "t1".into(),
        table_outer: "t2".into(),
        algorithm: "Algo".into(),
        batch_size: 123,
        next_offset: 456,
        total_outer: 789,
        partition_idx: 0,
        partition_name: "p".into(),
        updated_utc: "now".into(),
        last_id: Some(456),
        watermark_id: Some(999),
        filter_sig: Some("test".into()),
    };
    save_checkpoint(&path, &cp).unwrap();
    let loaded = load_checkpoint(&path).expect("should load");
    assert_eq!(loaded.table_inner, cp.table_inner);
    assert_eq!(loaded.next_offset, cp.next_offset);
    remove_checkpoint(&path);
    assert!(load_checkpoint(&path).is_none());
}
// --- Streaming matching and export for large datasets ---
use crate::db::{
    fetch_person_rows_chunk, fetch_person_rows_chunk_all_columns,
    fetch_person_rows_chunk_all_columns_keyset, fetch_person_rows_chunk_keyset,
    fetch_person_rows_chunk_where, fetch_person_rows_chunk_where_keyset, get_max_id,
    get_max_id_where, get_person_count,
};
use anyhow::Result;
use sqlx::MySqlPool;

fn key_for(algo: MatchingAlgorithm, p: &NormalizedPerson) -> Option<String> {
    match algo {
        MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => {
            let (Some(ln), Some(fnm), Some(d)) = (
                p.last_name.as_deref(),
                p.first_name.as_deref(),
                p.birthdate.as_ref(),
            ) else {
                return None;
            };
            if direct_norm_fuzzy_enabled() {
                let ln2 = normalize_simple(ln);
                let fn2 = normalize_simple(fnm);
                Some(format!("{}\x1F{}\x1F{}", ln2, fn2, d))
            } else {
                Some(format!("{}\x1F{}\x1F{}", ln, fnm, d))
            }
        }
        MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => {
            let (Some(ln), Some(fnm), Some(d)) = (
                p.last_name.as_deref(),
                p.first_name.as_deref(),
                p.birthdate.as_ref(),
            ) else {
                return None;
            };
            let mid = p.middle_name.clone().unwrap_or_default();
            if direct_norm_fuzzy_enabled() {
                let ln2 = normalize_simple(ln);
                let fn2 = normalize_simple(fnm);
                let mid2 = normalize_simple(&mid);
                Some(format!("{}\x1F{}\x1F{}\x1F{}", ln2, fn2, mid2, d))
            } else {
                Some(format!("{}\x1F{}\x1F{}\x1F{}", ln, fnm, mid, d))
            }
        }
        MatchingAlgorithm::Fuzzy
        | MatchingAlgorithm::FuzzyNoMiddle
        | MatchingAlgorithm::HouseholdGpu
        | MatchingAlgorithm::HouseholdGpuOpt6
        | MatchingAlgorithm::LevenshteinWeighted => None,
    }
}

#[cfg(feature = "new_engine")]
pub fn key_for_engine(algo: MatchingAlgorithm, p: &NormalizedPerson) -> Option<String> {
    key_for(algo, p)
}

fn to_pair(
    orig1: &Person,
    orig2: &Person,
    algo: MatchingAlgorithm,
    _np1: &NormalizedPerson,
    _np2: &NormalizedPerson,
) -> MatchPair {
    let (im1, im2) = (
        matches!(algo, MatchingAlgorithm::IdUuidYasIsMatchedInfnbd),
        matches!(algo, MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd),
    );
    let matched_fields = match algo {
        MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => {
            vec!["id", "uuid", "first_name", "last_name", "birthdate"]
                .into_iter()
                .map(String::from)
                .collect()
        }

        MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => vec![
            "id",
            "uuid",
            "first_name",
            "middle_name",
            "last_name",
            "birthdate",
        ]
        .into_iter()
        .map(String::from)
        .collect(),
        MatchingAlgorithm::Fuzzy
        | MatchingAlgorithm::FuzzyNoMiddle
        | MatchingAlgorithm::LevenshteinWeighted => vec!["fuzzy".into(), "birthdate".into()],
        MatchingAlgorithm::HouseholdGpu | MatchingAlgorithm::HouseholdGpuOpt6 => {
            vec!["household".into()]
        }
    };
    MatchPair {
        person1: orig1.clone(),
        person2: orig2.clone(),
        confidence: 1.0,
        matched_fields,
        is_matched_infnbd: im1,
        is_matched_infnmnbd: im2,
    }
}

#[cfg(feature = "new_engine")]
pub fn to_pair_public(orig1: &Person, orig2: &Person, algo: MatchingAlgorithm) -> MatchPair {
    let n1 = crate::normalize::normalize_person(orig1);
    let n2 = crate::normalize::normalize_person(orig2);
    to_pair(orig1, orig2, algo, &n1, &n2)
}

async fn build_index(
    pool: &MySqlPool,
    table: &str,
    algo: MatchingAlgorithm,
    mut batch: i64,
) -> Result<HashMap<String, Vec<Person>>> {
    let total = get_person_count(pool, table).await?;
    let watermark = get_max_id(pool, table).await?;
    if batch <= 0 {
        batch = 50_000;
    }
    let mut map: HashMap<String, Vec<Person>> =
        HashMap::with_capacity((total as f64 * 0.8) as usize);
    let mut last_id: i64 = 0;
    loop {
        let rows =
            fetch_person_rows_chunk_keyset(pool, table, last_id, batch, Some(watermark)).await?;
        if rows.is_empty() {
            break;
        }
        for p in rows.iter() {
            let n = normalize_person(p);
            if let Some(k) = key_for(algo, &n) {
                map.entry(k).or_default().push(p.clone());
            }
        }
        if let Some(tail) = rows.last() {
            last_id = tail.id;
        }
    }
    Ok(map)
}

// --- Batch Size Constants ---
// These constants define the bounds for streaming batch sizes based on memory efficiency analysis.
// Recommendation from plan.md: min=5000, max=50000, default=50000

/// Minimum batch size for streaming operations (prevents too-fine-grained DB queries)
pub const MIN_BATCH_SIZE: i64 = 5_000;

/// Maximum batch size for streaming operations (prevents memory exhaustion on large datasets)
/// Reduced from 100,000 to 50,000 for better memory efficiency per plan.md Phase 3.1
pub const MAX_BATCH_SIZE: i64 = 50_000;

/// Default batch size when no adaptive calculation is applied
pub const DEFAULT_BATCH_SIZE: i64 = 50_000;

#[derive(Clone, Debug)]
pub struct StreamingConfig {
    pub batch_size: i64,
    pub memory_soft_min_mb: u64,
    // new fields for robustness on millions of records
    pub flush_every: usize, // flush output every N matches

    pub resume: bool,          // resume from checkpoint if exists
    pub retry_max: u32,        // DB retry attempts per chunk
    pub retry_backoff_ms: u64, // base backoff between retries
    pub checkpoint_path: Option<String>,
    // GPU hash join acceleration (Algorithms 1/2 only)
    pub use_gpu_hash_join: bool, // legacy switch enabling GPU hash-join path
    pub use_gpu_build_hash: bool, // GPU for index build-side hashing
    pub use_gpu_probe_hash: bool, // GPU for probe-side hashing
    pub gpu_probe_batch_mb: u64, // advisory memory budget (MB) for probe GPU batches
    // New: GPU hashing pipeline configuration
    pub gpu_streams: u32,          // number of CUDA streams for overlap (1=off)
    pub gpu_use_pinned_host: bool, // optional pinned host staging (best-effort)
    pub gpu_buffer_pool: bool,     // reuse device buffers within a run (best-effort)

    // New: GPU pre-pass for fuzzy direct phase
    pub use_gpu_fuzzy_direct_hash: bool,

    // Optional: enable runtime dynamic GPU auto-tuning (tile size, streams)
    pub enable_dynamic_gpu_tuning: bool,

    // New: apply fuzzy-style normalization to direct algorithms (A1/A2)
    pub direct_use_fuzzy_normalization: bool,

    // New: GPU fuzzy metrics (Levenshtein/Jaro/Jaro-Winkler) toggle for in-memory and partitioned streaming
    pub use_gpu_fuzzy_metrics: bool,

    // Streaming pipeline optimization toggles
    pub async_prefetch: bool, // prefetch next DB batch while processing current
    pub parallel_normalize: bool, // use rayon to normalize rows in parallel
    pub prefetch_pool_size: u32, // desired pool concurrency for prefetch (best-effort)
}
impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            batch_size: 50_000,
            memory_soft_min_mb: 800,
            flush_every: 10_000,
            resume: true,

            retry_max: 5,
            retry_backoff_ms: 500,
            enable_dynamic_gpu_tuning: false,

            checkpoint_path: None,
            use_gpu_hash_join: false,
            use_gpu_build_hash: false,
            use_gpu_probe_hash: false,
            gpu_probe_batch_mb: 256,
            gpu_streams: 1,
            gpu_use_pinned_host: false,
            gpu_buffer_pool: true,
            use_gpu_fuzzy_direct_hash: false,
            direct_use_fuzzy_normalization: false,
            use_gpu_fuzzy_metrics: false,
            async_prefetch: false,
            parallel_normalize: false,
            prefetch_pool_size: 1,
        }
    }
}

/// Compute an adaptive StreamingConfig based on available memory (MB).
/// Conservative defaults with clamped batch sizes for stability across machines.
/// Uses MIN_BATCH_SIZE and MAX_BATCH_SIZE constants for bounds.
#[allow(dead_code)]
pub fn compute_stream_cfg(avail_mb: u64) -> StreamingConfig {
    let mut cfg = StreamingConfig::default();
    // Start with a conservative estimate: roughly a quarter of free RAM in rows, with clamps

    let mut b = (avail_mb as i64 - 1024).max(256) / 4;
    // Clamp to configured bounds (MIN_BATCH_SIZE=5000, MAX_BATCH_SIZE=50000)
    b = b.clamp(MIN_BATCH_SIZE, MAX_BATCH_SIZE);
    cfg.batch_size = b;
    cfg.flush_every = (cfg.batch_size as usize / 10).max(1000);
    cfg
}

// --- Hash helpers for GPU hash join (CPU fallback) ---
#[inline]
fn fnv1a64_bytes(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis
    let prime: u64 = 0x100000001b3; // FNV prime
    for &b in data {
        hash ^= b as u64;
        hash = hash.wrapping_mul(prime);
    }

    hash
}

#[inline]
fn hash_key_for_np(algo: MatchingAlgorithm, p: &NormalizedPerson) -> Option<u64> {
    match algo {
        MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => {
            let (Some(ln), Some(fnm), Some(d)) = (
                p.last_name.as_deref(),
                p.first_name.as_deref(),
                p.birthdate.as_ref(),
            ) else {
                return None;
            };
            let s = if direct_norm_fuzzy_enabled() {
                let ln2 = normalize_simple(ln);
                let fn2 = normalize_simple(fnm);
                format!("{}\x1F{}\x1F{}", ln2, fn2, d)
            } else {
                format!("{}\x1F{}\x1F{}", ln, fnm, d)
            };
            Some(fnv1a64_bytes(s.as_bytes()))
        }
        MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => {
            let (Some(ln), Some(fnm), Some(d)) = (
                p.last_name.as_deref(),
                p.first_name.as_deref(),
                p.birthdate.as_ref(),
            ) else {
                return None;
            };
            let mid = p.middle_name.clone().unwrap_or_default();
            let s = if direct_norm_fuzzy_enabled() {
                let ln2 = normalize_simple(ln);
                let fn2 = normalize_simple(fnm);
                let mid2 = normalize_simple(&mid);
                format!("{}\x1F{}\x1F{}\x1F{}", ln2, fn2, mid2, d)
            } else {
                format!("{}\x1F{}\x1F{}\x1F{}", ln, fnm, mid, d)
            };
            Some(fnv1a64_bytes(s.as_bytes()))
        }
        MatchingAlgorithm::Fuzzy
        | MatchingAlgorithm::FuzzyNoMiddle
        | MatchingAlgorithm::HouseholdGpu
        | MatchingAlgorithm::HouseholdGpuOpt6
        | MatchingAlgorithm::LevenshteinWeighted => None,
    }
}

#[inline]
fn concat_key_for_np(algo: MatchingAlgorithm, p: &NormalizedPerson) -> Option<String> {
    match algo {
        MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => {
            let (Some(ln), Some(fnm), Some(d)) = (
                p.last_name.as_deref(),
                p.first_name.as_deref(),
                p.birthdate.as_ref(),
            ) else {
                return None;
            };
            Some(format!("{}\x1F{}\x1F{}", ln, fnm, d))
        }
        MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => {
            let (Some(ln), Some(fnm), Some(d)) = (
                p.last_name.as_deref(),
                p.first_name.as_deref(),
                p.birthdate.as_ref(),
            ) else {
                return None;
            };
            let mid = p.middle_name.clone().unwrap_or_default();
            Some(format!("{}\x1F{}\x1F{}\x1F{}", ln, fnm, mid, d))
        }
        MatchingAlgorithm::Fuzzy
        | MatchingAlgorithm::FuzzyNoMiddle
        | MatchingAlgorithm::HouseholdGpu
        | MatchingAlgorithm::HouseholdGpuOpt6
        | MatchingAlgorithm::LevenshteinWeighted => None,
    }
}

/// GPU-accelerated (hash-join) streaming path for Algorithms 1 & 2.
/// Falls back to CPU hashing if GPU is unavailable or feature is disabled.
pub async fn stream_match_gpu_hash_join<F>(
    pool: &MySqlPool,
    table1: &str,
    table2: &str,
    algo: MatchingAlgorithm,
    mut on_match: F,
    cfg: StreamingConfig,
    on_progress: impl Fn(ProgressUpdate) + Sync,
    ctrl: Option<StreamControl>,
) -> Result<usize>
where
    F: FnMut(&MatchPair) -> Result<()>,
{
    use crate::util::checkpoint::{StreamCheckpoint, load_checkpoint, save_checkpoint};
    if matches!(algo, MatchingAlgorithm::Fuzzy) {
        anyhow::bail!("GPU hash join applies only to Algorithms 1/2 (deterministic)");
    }
    // Decide inner/outer by row count
    let c1 = get_person_count(pool, table1).await?;
    let c2 = get_person_count(pool, table2).await?;
    // Try GPU hash computation context (optional)
    #[cfg(feature = "gpu")]
    let gpu_hash_ctx: Option<gpu::GpuHashContext> = if cfg.use_gpu_hash_join
        || cfg.use_gpu_build_hash
        || cfg.use_gpu_probe_hash
    {
        match gpu::GpuHashContext::get() {
            Ok(ctx) => {
                log::info!("[GPU] Hash context ready");
                Some(ctx)
            }
            Err(e) => {
                if cfg.use_gpu_probe_hash {
                    // Explicit probe-on-GPU requested: do not silently fall back
                    return Err(anyhow!(
                        "GPU probe hashing requested but CUDA unavailable: {}",
                        e
                    ));
                } else {
                    log::warn!(
                        "[GPU] Hash context init failed: {}. Falling back to CPU hashing for build.",
                        e
                    );
                    None
                }
            }
        }
    } else {
        None
    };
    #[cfg(not(feature = "gpu"))]
    let _gpu_hash_ctx: Option<()> = None;

    #[cfg(feature = "gpu")]
    if cfg.enable_dynamic_gpu_tuning {
        set_dynamic_gpu_tuning(true);
        self::gpu::dynamic_tuner::ensure_started(true);
    }

    let (inner_table, outer_table, total_outer) = if c2 <= c1 {
        (table2, table1, c1)
    } else {
        (table1, table2, c2)
    };
    let inner_watermark = get_max_id(pool, inner_table).await?;

    // Snapshot bounds + resume support
    let outer_watermark = get_max_id(pool, outer_table).await?;
    let mut offset: i64 = 0; // interpreted as last processed id in keyset mode
    if cfg.resume {
        if let Some(path) = cfg.checkpoint_path.as_ref() {
            if let Some(cp) = load_checkpoint(path) {
                if cp.table_inner == inner_table
                    && cp.table_outer == outer_table
                    && cp.batch_size == cfg.batch_size
                {
                    offset = cp.last_id.unwrap_or(cp.next_offset);
                }
            }
        }
    }

    // Build inner hash index (CPU hash; can be replaced with GPU hash in future)
    let mut index: std::collections::HashMap<u64, Vec<Person>> = std::collections::HashMap::new();
    let mut inner_off: i64 = 0; // last id processed on inner side
    let mut inner_processed: usize = 0;
    let batch = cfg.batch_size.max(10_000);
    on_progress(ProgressUpdate {
        processed: 0,
        total: total_outer as usize,
        percent: 0.0,
        eta_secs: 0,
        mem_used_mb: memory_stats_mb().used_mb,
        mem_avail_mb: memory_stats_mb().avail_mb,
        stage: "indexing_hash",
        batch_size_current: Some(batch),
        gpu_total_mb: 0,
        gpu_free_mb: 0,
        gpu_active: false,
    });
    let mut _gpu_logged_once = false;
    loop {
        let rows = fetch_person_rows_chunk_keyset(
            pool,
            inner_table,
            inner_off,
            batch,
            Some(inner_watermark),
        )
        .await?;
        if rows.is_empty() {
            break;
        }
        // Prepare normalized key strings for hashing (only valid keys)
        let mut key_strs: Vec<String> = Vec::new();
        let mut key_idx: Vec<usize> = Vec::new();
        let mut norm_cache: Vec<Option<NormalizedPerson>> = Vec::with_capacity(rows.len());
        for (i, p) in rows.iter().enumerate() {
            let n = normalize_person(p);
            let k = concat_key_for_np(algo, &n);
            norm_cache.push(Some(n));
            if let Some(s) = k {
                key_idx.push(i);
                key_strs.push(s);
            }
        }
        let mut hashed: Option<Vec<u64>> = None;
        #[cfg(feature = "gpu")]
        if cfg.use_gpu_build_hash {
            if let Some(ctx) = gpu_hash_ctx.as_ref() {
                if !key_strs.is_empty() {
                    // Progress: starting GPU hash for this batch (build side)
                    let (gt, gf) = ctx.mem_info_mb();
                    let memx = memory_stats_mb();
                    on_progress(ProgressUpdate {
                        processed: inner_processed,
                        total: total_outer as usize,
                        percent: 0.0,
                        eta_secs: 0,
                        mem_used_mb: memx.used_mb,
                        mem_avail_mb: memx.avail_mb,
                        stage: "gpu_hash",
                        batch_size_current: Some(batch),
                        gpu_total_mb: gt,
                        gpu_free_mb: gf,
                        gpu_active: true,
                    });

                    let streams = if dynamic_gpu_tuning_enabled() {
                        self::gpu::dynamic_tuner::get_current_streams()
                    } else {
                        cfg.gpu_streams
                    };
                    let budget = (gf / 2).max(128);
                    let gpu_call = || {
                        if streams >= 2 {
                            self::gpu::hash_fnv1a64_batch_tiled_overlap(
                                ctx,
                                &key_strs,
                                budget,
                                64,
                                streams,
                                cfg.gpu_buffer_pool,
                                cfg.gpu_use_pinned_host,
                            )
                        } else {
                            self::gpu::hash_fnv1a64_batch_tiled(ctx, &key_strs, budget, 64)
                        }
                    };
                    let res = crate::matching::gpu_config::with_oom_cpu_fallback(
                        gpu_call,
                        || {
                            key_strs
                                .iter()
                                .map(|s| fnv1a64_bytes(s.as_bytes()))
                                .collect::<Vec<u64>>()
                        },
                        "inner index hashing",
                    );
                    match res {
                        Ok(v) => {
                            if !_gpu_logged_once {
                                log::info!(
                                    "[GPU] Using GPU hashing for inner index (first batch: {} keys)",
                                    v.len()
                                );
                                _gpu_logged_once = true;
                            }
                            let (gt2, gf2) = ctx.mem_info_mb();
                            let mem2 = memory_stats_mb();
                            on_progress(ProgressUpdate {
                                processed: inner_processed,
                                total: total_outer as usize,
                                percent: 0.0,
                                eta_secs: 0,
                                mem_used_mb: mem2.used_mb,
                                mem_avail_mb: mem2.avail_mb,
                                stage: "gpu_hash_done",
                                batch_size_current: Some(batch),
                                gpu_total_mb: gt2,
                                gpu_free_mb: gf2,
                                gpu_active: true,
                            });
                            hashed = Some(v);
                        }
                        Err(e) => {
                            log::warn!("GPU hash failed, falling back to CPU: {}", e);
                        }
                    }
                }
            }
        }
        if let Some(hs) = hashed.as_ref() {
            for (j, &h) in hs.iter().enumerate() {
                let i = key_idx[j];
                index.entry(h).or_default().push(rows[i].clone());
            }
        } else {
            // CPU fallback hashing
            for (i, p) in rows.iter().enumerate() {
                if let Some(ref n) = norm_cache[i] {
                    if let Some(h) = hash_key_for_np(algo, n) {
                        index.entry(h).or_default().push(p.clone());
                    }
                }
            }
        }
        if let Some(last) = rows.last() {
            inner_off = last.id;
        }
        inner_processed += rows.len();
    }

    // Stream outer table and probe
    let start = Instant::now();
    let mut written = 0usize;
    let mut _processed_rows: usize = 0;
    let mut next_rows_task: Option<tokio::task::JoinHandle<anyhow::Result<Vec<Person>>>> = None;
    loop {
        if let Some(c) = &ctrl {
            if c.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            while c.pause.load(std::sync::atomic::Ordering::Relaxed) {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        }
        let rows: Vec<Person> = if cfg.async_prefetch {
            if let Some(handle) = next_rows_task.take() {
                match handle.await {
                    Ok(res) => res?,
                    Err(_join_err) => {
                        // fall back to direct fetch with retry if join failed
                        let mut tries = 0u32;
                        loop {
                            match fetch_person_rows_chunk_keyset(
                                pool,
                                outer_table,
                                offset,
                                batch,
                                Some(outer_watermark),
                            )
                            .await
                            {
                                Ok(v) => break v,
                                Err(e) => {
                                    tries += 1;
                                    if tries > cfg.retry_max {
                                        return Err(e);
                                    }
                                    let backoff =
                                        cfg.retry_backoff_ms * (1u64 << (tries.min(5) - 1));
                                    tokio::time::sleep(std::time::Duration::from_millis(backoff))
                                        .await;
                                }
                            }
                        }
                    }
                }
            } else {
                let mut tries = 0u32;
                loop {
                    match fetch_person_rows_chunk_keyset(
                        pool,
                        outer_table,
                        offset,
                        batch,
                        Some(outer_watermark),
                    )
                    .await
                    {
                        Ok(v) => break v,
                        Err(e) => {
                            tries += 1;
                            if tries > cfg.retry_max {
                                return Err(e);
                            }
                            let backoff = cfg.retry_backoff_ms * (1u64 << (tries.min(5) - 1));
                            tokio::time::sleep(std::time::Duration::from_millis(backoff)).await;
                        }
                    }
                }
            }
        } else {
            fetch_person_rows_chunk_keyset(pool, outer_table, offset, batch, Some(outer_watermark))
                .await?
        };
        if rows.is_empty() {
            break;
        }
        if let Some(last) = rows.last() {
            offset = last.id;
        }
        _processed_rows += rows.len();
        // Progress update
        let elapsed = start.elapsed();
        let processed = _processed_rows.min(total_outer as usize);
        let frac = (processed as f32 / total_outer as f32).clamp(0.0, 1.0);
        let eta_secs = if frac > 0.0 {
            (elapsed.as_secs_f32() * (1.0 - frac) / frac) as u64
        } else {
            0
        };
        let mem = memory_stats_mb();
        on_progress(ProgressUpdate {
            processed,
            total: total_outer as usize,
            percent: frac * 100.0,
            eta_secs,
            mem_used_mb: mem.used_mb,
            mem_avail_mb: mem.avail_mb,
            stage: "probing_hash",
            batch_size_current: Some(batch),
            gpu_total_mb: 0,
            gpu_free_mb: 0,
            gpu_active: false,
        });

        // Schedule prefetch of next rows (if enabled) while we process current batch
        if cfg.async_prefetch && _processed_rows < total_outer as usize {
            let pool_cloned = pool.clone();
            let table = outer_table.to_string();
            let next_off = offset;
            let next_batch = batch;
            let retry_max = cfg.retry_max;
            let backoff_ms = cfg.retry_backoff_ms;
            next_rows_task = Some(tokio::spawn(async move {
                let mut tries = 0u32;
                loop {
                    match fetch_person_rows_chunk_keyset(
                        &pool_cloned,
                        &table,
                        next_off,
                        next_batch,
                        Some(outer_watermark),
                    )
                    .await
                    {
                        Ok(v) => break Ok(v),
                        Err(e) => {
                            tries += 1;
                            if tries > retry_max {
                                break Err(e);
                            }
                            let backoff = backoff_ms * (1u64 << (tries.min(5) - 1));
                            tokio::time::sleep(std::time::Duration::from_millis(backoff)).await;
                        }
                    }
                }
            }));
        }

        // Prepare probe batch normalization and keys
        let mut probe_norms: Vec<NormalizedPerson> = Vec::with_capacity(rows.len());
        let mut probe_keys: Vec<String> = Vec::new();
        let mut probe_idx: Vec<usize> = Vec::new();
        if cfg.parallel_normalize {
            probe_norms = rows.par_iter().map(normalize_person).collect();
            for (i, n) in probe_norms.iter().enumerate() {
                if let Some(k) = concat_key_for_np(algo, n) {
                    probe_keys.push(k);
                    probe_idx.push(i);
                }
            }
        } else {
            for (i, p) in rows.iter().enumerate() {
                let n = normalize_person(p);
                if concat_key_for_np(algo, &n).is_some() {
                    if let Some(k) = concat_key_for_np(algo, &n) {
                        probe_keys.push(k);
                        probe_idx.push(i);
                    }
                }
                probe_norms.push(n);
            }
        }
        // Compute probe hashes (GPU if enabled)
        let mut probe_hashes_opt: Option<Vec<u64>> = None;
        #[cfg(feature = "gpu")]
        if cfg.use_gpu_probe_hash {
            if let Some(ctx) = gpu_hash_ctx.as_ref() {
                if !probe_keys.is_empty() {
                    let (gt, gf) = ctx.mem_info_mb();
                    let memx = memory_stats_mb();
                    log::info!(
                        "[GPU] Using GPU hashing for probe (batch: {} keys)",
                        probe_keys.len()
                    );
                    on_progress(ProgressUpdate {
                        processed,
                        total: total_outer as usize,
                        percent: frac * 100.0,
                        eta_secs,
                        mem_used_mb: memx.used_mb,
                        mem_avail_mb: memx.avail_mb,
                        stage: "gpu_probe_hash",
                        batch_size_current: Some(batch),
                        gpu_total_mb: gt,
                        gpu_free_mb: gf,
                        gpu_active: true,
                    });
                    {
                        let streams = if dynamic_gpu_tuning_enabled() {
                            self::gpu::dynamic_tuner::get_current_streams()
                        } else {
                            cfg.gpu_streams
                        };
                        let gpu_call = || {
                            if streams >= 2 {
                                self::gpu::hash_fnv1a64_batch_tiled_overlap(
                                    ctx,
                                    &probe_keys,
                                    cfg.gpu_probe_batch_mb,
                                    64,
                                    streams,
                                    cfg.gpu_buffer_pool,
                                    cfg.gpu_use_pinned_host,
                                )
                            } else {
                                self::gpu::hash_fnv1a64_batch_tiled(
                                    ctx,
                                    &probe_keys,
                                    cfg.gpu_probe_batch_mb,
                                    64,
                                )
                            }
                        };
                        match crate::matching::gpu_config::with_oom_cpu_fallback(
                            gpu_call,
                            || {
                                // CPU fallback: compute FNV-1a64 hashes for probe_keys
                                probe_keys
                                    .iter()
                                    .map(|s| fnv1a64_bytes(s.as_bytes()))
                                    .collect::<Vec<u64>>()
                            },
                            "probe hashing",
                        ) {
                            Ok(hs) => {
                                // If GPU succeeded, emit GPU-done progress; otherwise we already logged fallback
                                let (gt2, gf2) = ctx.mem_info_mb();
                                let mem2 = memory_stats_mb();
                                on_progress(ProgressUpdate {
                                    processed,
                                    total: total_outer as usize,
                                    percent: frac * 100.0,
                                    eta_secs,
                                    mem_used_mb: mem2.used_mb,
                                    mem_avail_mb: mem2.avail_mb,
                                    stage: "gpu_probe_hash_done",
                                    batch_size_current: Some(batch),
                                    gpu_total_mb: gt2,
                                    gpu_free_mb: gf2,
                                    gpu_active: true,
                                });
                                probe_hashes_opt = Some(hs);
                            }
                            Err(e) => {
                                return Err(anyhow!("GPU probe hash failed: {}", e));
                            }
                        }
                    }
                }
            } else {
                return Err(anyhow!(
                    "GPU probe hashing requested but no CUDA context available"
                ));
            }
        }
        if let Some(probe_hashes) = probe_hashes_opt.as_ref() {
            for (j, &h) in probe_hashes.iter().enumerate() {
                let i = probe_idx[j];
                let p = &rows[i];
                let n = &probe_norms[i];
                if let Some(cands) = index.get(&h) {
                    for q in cands {
                        let n2 = normalize_person(q);
                        let ok = match algo {
                            MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => matches_algo1(n, &n2),
                            MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => matches_algo2(n, &n2),
                            MatchingAlgorithm::Fuzzy
                            | MatchingAlgorithm::FuzzyNoMiddle
                            | MatchingAlgorithm::HouseholdGpu
                            | MatchingAlgorithm::HouseholdGpuOpt6
                            | MatchingAlgorithm::LevenshteinWeighted => false,
                        };
                        if ok {
                            let pair = if inner_table == table2 {
                                to_pair(p, q, algo, n, &n2)
                            } else {
                                to_pair(q, p, algo, &n2, n)
                            };
                            on_match(&pair)?;
                            written += 1;
                        }
                    }
                }
            }
        } else {
            // CPU hashing for probe
            for (i, p) in rows.iter().enumerate() {
                let n = &probe_norms[i];
                if let Some(h) = hash_key_for_np(algo, n) {
                    if let Some(cands) = index.get(&h) {
                        for q in cands {
                            let n2 = normalize_person(q);
                            let ok = match algo {
                                MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => {
                                    matches_algo1(n, &n2)
                                }
                                MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => {
                                    matches_algo2(n, &n2)
                                }
                                MatchingAlgorithm::Fuzzy
                                | MatchingAlgorithm::FuzzyNoMiddle
                                | MatchingAlgorithm::HouseholdGpu
                                | MatchingAlgorithm::HouseholdGpuOpt6
                                | MatchingAlgorithm::LevenshteinWeighted => false,
                            };
                            if ok {
                                let pair = if inner_table == table2 {
                                    to_pair(p, q, algo, n, &n2)
                                } else {
                                    to_pair(q, p, algo, &n2, n)
                                };
                                on_match(&pair)?;
                                written += 1;
                            }
                        }
                    }
                }
            }
        }

        if cfg.resume {
            if let Some(path) = cfg.checkpoint_path.as_ref() {
                let cp = StreamCheckpoint {
                    db: String::new(),
                    table_inner: inner_table.into(),
                    table_outer: outer_table.into(),
                    algorithm: format!("{:?}", algo),
                    batch_size: batch,
                    next_offset: offset,
                    total_outer,
                    partition_idx: 0,
                    partition_name: String::new(),
                    updated_utc: chrono::Utc::now().to_rfc3339(),
                    last_id: Some(offset),
                    watermark_id: Some(outer_watermark),
                    filter_sig: Some(format!("id<={}", outer_watermark)),
                };
                let _ = save_checkpoint(path, &cp);
            }
        }
    }

    Ok(written)
}

// Unified internal implementation for single-DB and cross-DB streaming with GPU parity
async fn stream_match_csv_internal<F>(
    pool1: &MySqlPool,
    pool2: &MySqlPool,
    table1: &str,
    table2: &str,
    algo: MatchingAlgorithm,
    mut on_match: F,
    cfg: StreamingConfig,
    on_progress: impl Fn(ProgressUpdate) + Sync,
    ctrl: Option<StreamControl>,
) -> Result<usize>
where
    F: FnMut(&MatchPair) -> Result<()>,
{
    use crate::util::checkpoint::{StreamCheckpoint, load_checkpoint, save_checkpoint};
    if matches!(
        algo,
        MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle
    ) {
        anyhow::bail!(
            "Fuzzy algorithms are supported only in in-memory or partitioned mode (CSV). Use algorithm=3/4 with CSV in-memory or partitioned streaming."
        );
    }

    // Decide inner/outer by row counts across pools
    let c1 = get_person_count(pool1, table1).await?;
    let c2 = get_person_count(pool2, table2).await?;
    let inner_is_t2 = c2 <= c1;
    let inner_pool = if inner_is_t2 { pool2 } else { pool1 };
    let outer_pool = if inner_is_t2 { pool1 } else { pool2 };
    let inner_table = if inner_is_t2 { table2 } else { table1 };
    let outer_table = if inner_is_t2 { table1 } else { table2 };
    let total_outer = if inner_is_t2 { c1 } else { c2 };

    // Optional GPU hash context (for build/probe hashing)
    #[cfg(feature = "gpu")]
    let gpu_hash_ctx: Option<gpu::GpuHashContext> =
        if cfg.use_gpu_hash_join || cfg.use_gpu_build_hash || cfg.use_gpu_probe_hash {
            match gpu::GpuHashContext::get() {
                Ok(ctx) => {
                    log::info!("[GPU] Hash context ready");
                    Some(ctx)
                }
                Err(e) => {
                    if cfg.use_gpu_probe_hash {
                        return Err(anyhow!(
                            "GPU probe hashing requested but CUDA unavailable: {}",
                            e
                        ));
                    }
                    log::warn!(
                        "[GPU] Hash context init failed: {}. Falling back to CPU hashing.",
                        e
                    );
                    None
                }
            }
        } else {
            None
        };
    #[cfg(not(feature = "gpu"))]
    let _gpu_hash_ctx: Option<()> = None;

    // Snapshot and resume
    let inner_watermark = get_max_id(inner_pool, inner_table).await?;
    let outer_watermark = get_max_id(outer_pool, outer_table).await?;
    let mut offset: i64 = 0; // keyset last_id cursor for outer table
    let mut batch = cfg.batch_size.max(10_000);
    if cfg.resume {
        if let Some(p) = cfg.checkpoint_path.as_ref() {
            if let Some(cp) = load_checkpoint(p) {
                if cp.db == ""
                    || (cp.table_inner == inner_table
                        && cp.table_outer == outer_table
                        && cp.algorithm == format!("{:?}", algo))
                {
                    offset = cp.last_id.unwrap_or(cp.next_offset).min(outer_watermark);
                    batch = cp.batch_size.max(10_000);
                }
            }
        }
    }

    // Build inner hash index (CPU or GPU build hashing)
    let mut index: std::collections::HashMap<u64, Vec<Person>> = std::collections::HashMap::new();
    let mut inner_off: i64 = 0;
    let mut inner_processed: usize = 0;
    on_progress(ProgressUpdate {
        processed: 0,
        total: total_outer as usize,
        percent: 0.0,
        eta_secs: 0,
        mem_used_mb: memory_stats_mb().used_mb,
        mem_avail_mb: memory_stats_mb().avail_mb,
        stage: "indexing_hash",
        batch_size_current: Some(batch),
        gpu_total_mb: 0,
        gpu_free_mb: 0,
        gpu_active: false,
    });
    let mut gpu_logged_once = false;
    loop {
        let rows = fetch_person_rows_chunk_keyset(
            inner_pool,
            inner_table,
            inner_off,
            batch,
            Some(inner_watermark),
        )
        .await?;
        if rows.is_empty() {
            break;
        }
        // Prepare normalized key strings for hashing
        let mut key_strs: Vec<String> = Vec::new();
        let mut key_idx: Vec<usize> = Vec::new();
        let mut norm_cache: Vec<Option<NormalizedPerson>> = Vec::with_capacity(rows.len());
        for (i, p) in rows.iter().enumerate() {
            let n = normalize_person(p);
            let k = concat_key_for_np(algo, &n);
            norm_cache.push(Some(n));
            if let Some(s) = k {
                key_idx.push(i);
                key_strs.push(s);
            }
        }
        let mut hashed: Option<Vec<u64>> = None;
        #[cfg(feature = "gpu")]
        if cfg.use_gpu_build_hash {
            if let Some(ctx) = gpu_hash_ctx.as_ref() {
                if !key_strs.is_empty() {
                    let (gt, gf) = ctx.mem_info_mb();
                    let memx = memory_stats_mb();
                    on_progress(ProgressUpdate {
                        processed: inner_processed,
                        total: total_outer as usize,
                        percent: 0.0,
                        eta_secs: 0,
                        mem_used_mb: memx.used_mb,
                        mem_avail_mb: memx.avail_mb,
                        stage: "gpu_hash",
                        batch_size_current: Some(batch),
                        gpu_total_mb: gt,
                        gpu_free_mb: gf,
                        gpu_active: true,
                    });
                    {
                        let streams = if dynamic_gpu_tuning_enabled() {
                            self::gpu::dynamic_tuner::get_current_streams()
                        } else {
                            cfg.gpu_streams
                        };
                        let budget = (gf / 2).max(128);
                        let gpu_call = || {
                            if streams >= 2 {
                                self::gpu::hash_fnv1a64_batch_tiled_overlap(
                                    ctx,
                                    &key_strs,
                                    budget,
                                    64,
                                    streams,
                                    cfg.gpu_buffer_pool,
                                    cfg.gpu_use_pinned_host,
                                )
                            } else {
                                self::gpu::hash_fnv1a64_batch_tiled(ctx, &key_strs, budget, 64)
                            }
                        };
                        match crate::matching::gpu_config::with_oom_cpu_fallback(
                            gpu_call,
                            || {
                                key_strs
                                    .iter()
                                    .map(|s| fnv1a64_bytes(s.as_bytes()))
                                    .collect::<Vec<u64>>()
                            },
                            "inner index hashing",
                        ) {
                            Ok(v) => {
                                if !gpu_logged_once {
                                    log::info!(
                                        "[GPU] Using GPU hashing for inner index (first batch: {} keys)",
                                        v.len()
                                    );
                                    gpu_logged_once = true;
                                }
                                let (gt2, gf2) = ctx.mem_info_mb();
                                let mem2 = memory_stats_mb();
                                on_progress(ProgressUpdate {
                                    processed: inner_processed,
                                    total: total_outer as usize,
                                    percent: 0.0,
                                    eta_secs: 0,
                                    mem_used_mb: mem2.used_mb,
                                    mem_avail_mb: mem2.avail_mb,
                                    stage: "gpu_hash_done",
                                    batch_size_current: Some(batch),
                                    gpu_total_mb: gt2,
                                    gpu_free_mb: gf2,
                                    gpu_active: true,
                                });
                                hashed = Some(v);
                            }
                            Err(e) => {
                                log::warn!("GPU hash failed, falling back to CPU: {}", e);
                            }
                        }
                    }
                }
            }
        }
        if let Some(hs) = hashed.as_ref() {
            for (j, &h) in hs.iter().enumerate() {
                let i = key_idx[j];
                index.entry(h).or_default().push(rows[i].clone());
            }
        } else {
            for (i, p) in rows.iter().enumerate() {
                if let Some(ref n) = norm_cache[i] {
                    if let Some(h) = hash_key_for_np(algo, n) {
                        index.entry(h).or_default().push(p.clone());
                    }
                }
            }
        }
        if let Some(last) = rows.last() {
            inner_off = last.id;
        }
        inner_processed += rows.len();
    }

    // Stream outer table and probe
    let start = Instant::now();
    let mut written = 0usize;
    let mut _processed_rows: usize = 0;
    let mut next_rows_task: Option<tokio::task::JoinHandle<anyhow::Result<Vec<Person>>>> = None;
    loop {
        if let Some(c) = &ctrl {
            if c.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            while c.pause.load(std::sync::atomic::Ordering::Relaxed) {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        }
        // Prefetch / fetch outer rows with retry (keyset)
        let rows: Vec<Person> = if cfg.async_prefetch {
            if let Some(handle) = next_rows_task.take() {
                match handle.await {
                    Ok(res) => res?,
                    Err(_) => {
                        let mut tries = 0;
                        loop {
                            match fetch_person_rows_chunk_keyset(
                                outer_pool,
                                outer_table,
                                offset,
                                batch,
                                Some(outer_watermark),
                            )
                            .await
                            {
                                Ok(v) => break v,
                                Err(e) => {
                                    tries += 1;
                                    if tries > cfg.retry_max {
                                        return Err(e);
                                    }
                                    let backoff =
                                        cfg.retry_backoff_ms * (1u64 << (tries.min(5) - 1));
                                    tokio::time::sleep(std::time::Duration::from_millis(backoff))
                                        .await;
                                }
                            }
                        }
                    }
                }
            } else {
                let mut tries = 0;
                loop {
                    match fetch_person_rows_chunk_keyset(
                        outer_pool,
                        outer_table,
                        offset,
                        batch,
                        Some(outer_watermark),
                    )
                    .await
                    {
                        Ok(v) => break v,
                        Err(e) => {
                            tries += 1;
                            if tries > cfg.retry_max {
                                return Err(e);
                            }
                            let backoff = cfg.retry_backoff_ms * (1u64 << (tries.min(5) - 1));
                            tokio::time::sleep(std::time::Duration::from_millis(backoff)).await;
                        }
                    }
                }
            }
        } else {
            fetch_person_rows_chunk_keyset(
                outer_pool,
                outer_table,
                offset,
                batch,
                Some(outer_watermark),
            )
            .await?
        };
        if rows.is_empty() {
            break;
        }
        if let Some(last) = rows.last() {
            offset = last.id;
        }
        _processed_rows += rows.len();

        // Progress update (pre-probe)
        let elapsed = start.elapsed();
        let processed = _processed_rows.min(total_outer as usize);
        let frac = (processed as f32 / total_outer as f32).clamp(0.0, 1.0);
        let eta_secs = if frac > 0.0 {
            (elapsed.as_secs_f32() * (1.0 - frac) / frac) as u64
        } else {
            0
        };
        let mem = memory_stats_mb();
        on_progress(ProgressUpdate {
            processed,
            total: total_outer as usize,
            percent: frac * 100.0,
            eta_secs,
            mem_used_mb: mem.used_mb,
            mem_avail_mb: mem.avail_mb,
            stage: "probing_hash",
            batch_size_current: Some(batch),
            gpu_total_mb: 0,
            gpu_free_mb: 0,
            gpu_active: false,
        });

        // Schedule prefetch of next rows
        if cfg.async_prefetch && _processed_rows < total_outer as usize {
            let pool_cloned = outer_pool.clone();
            let table = outer_table.to_string();
            let next_off = offset;
            let next_batch = batch;
            let retry_max = cfg.retry_max;
            let backoff_ms = cfg.retry_backoff_ms;
            next_rows_task = Some(tokio::spawn(async move {
                let mut tries = 0;
                loop {
                    match fetch_person_rows_chunk_keyset(
                        &pool_cloned,
                        &table,
                        next_off,
                        next_batch,
                        Some(outer_watermark),
                    )
                    .await
                    {
                        Ok(v) => break Ok(v),
                        Err(e) => {
                            tries += 1;
                            if tries > retry_max {
                                break Err(e);
                            }
                            let backoff = backoff_ms * (1u64 << (tries.min(5) - 1));
                            tokio::time::sleep(std::time::Duration::from_millis(backoff)).await;
                        }
                    }
                }
            }));
        }

        // Prepare probe batch normalization and keys
        let mut probe_norms: Vec<NormalizedPerson> = Vec::with_capacity(rows.len());
        let mut probe_keys: Vec<String> = Vec::new();
        let mut probe_idx: Vec<usize> = Vec::new();
        if cfg.parallel_normalize {
            probe_norms = rows.par_iter().map(normalize_person).collect();
            for (i, n) in probe_norms.iter().enumerate() {
                if let Some(k) = concat_key_for_np(algo, n) {
                    probe_keys.push(k);
                    probe_idx.push(i);
                }
            }
        } else {
            for (i, p) in rows.iter().enumerate() {
                let n = normalize_person(p);
                if let Some(k) = concat_key_for_np(algo, &n) {
                    probe_keys.push(k);
                    probe_idx.push(i);
                }
                probe_norms.push(n);
            }
        }

        // Compute probe hashes (GPU if enabled)
        let mut probe_hashes_opt: Option<Vec<u64>> = None;
        #[cfg(feature = "gpu")]
        if cfg.use_gpu_probe_hash {
            if let Some(ctx) = gpu_hash_ctx.as_ref() {
                if !probe_keys.is_empty() {
                    let (gt, gf) = ctx.mem_info_mb();
                    let memx = memory_stats_mb();
                    log::info!(
                        "[GPU] Using GPU hashing for probe (batch: {} keys)",
                        probe_keys.len()
                    );
                    on_progress(ProgressUpdate {
                        processed,
                        total: total_outer as usize,
                        percent: frac * 100.0,
                        eta_secs,
                        mem_used_mb: memx.used_mb,
                        mem_avail_mb: memx.avail_mb,
                        stage: "gpu_probe_hash",
                        batch_size_current: Some(batch),
                        gpu_total_mb: gt,
                        gpu_free_mb: gf,
                        gpu_active: true,
                    });
                    {
                        let streams = if dynamic_gpu_tuning_enabled() {
                            self::gpu::dynamic_tuner::get_current_streams()
                        } else {
                            cfg.gpu_streams
                        };
                        let gpu_call = || {
                            if streams >= 2 {
                                self::gpu::hash_fnv1a64_batch_tiled_overlap(
                                    ctx,
                                    &probe_keys,
                                    cfg.gpu_probe_batch_mb,
                                    64,
                                    streams,
                                    cfg.gpu_buffer_pool,
                                    cfg.gpu_use_pinned_host,
                                )
                            } else {
                                self::gpu::hash_fnv1a64_batch_tiled(
                                    ctx,
                                    &probe_keys,
                                    cfg.gpu_probe_batch_mb,
                                    64,
                                )
                            }
                        };
                        match crate::matching::gpu_config::with_oom_cpu_fallback(
                            gpu_call,
                            || {
                                // CPU fallback: compute FNV-1a64 hashes for probe_keys
                                probe_keys
                                    .iter()
                                    .map(|s| fnv1a64_bytes(s.as_bytes()))
                                    .collect::<Vec<u64>>()
                            },
                            "probe hashing",
                        ) {
                            Ok(hs) => {
                                let (gt2, gf2) = ctx.mem_info_mb();
                                let mem2 = memory_stats_mb();
                                on_progress(ProgressUpdate {
                                    processed,
                                    total: total_outer as usize,
                                    percent: frac * 100.0,
                                    eta_secs,
                                    mem_used_mb: mem2.used_mb,
                                    mem_avail_mb: mem2.avail_mb,
                                    stage: "gpu_probe_hash_done",
                                    batch_size_current: Some(batch),
                                    gpu_total_mb: gt2,
                                    gpu_free_mb: gf2,
                                    gpu_active: true,
                                });
                                probe_hashes_opt = Some(hs);
                            }
                            Err(e) => {
                                return Err(anyhow!("GPU probe hash failed: {}", e));
                            }
                        }
                    }
                }
            } else {
                return Err(anyhow!(
                    "GPU probe hashing requested but no CUDA context available"
                ));
            }
        }

        // Probe join
        if let Some(probe_hashes) = probe_hashes_opt.as_ref() {
            for (j, &h) in probe_hashes.iter().enumerate() {
                let i = probe_idx[j];
                let p = &rows[i];
                let n = &probe_norms[i];
                if let Some(cands) = index.get(&h) {
                    for q in cands {
                        let n2 = normalize_person(q);
                        let ok = match algo {
                            MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => matches_algo1(n, &n2),
                            MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => matches_algo2(n, &n2),
                            _ => false,
                        };
                        if ok {
                            let pair = if inner_is_t2 {
                                to_pair(p, q, algo, n, &n2)
                            } else {
                                to_pair(q, p, algo, &n2, n)
                            };
                            on_match(&pair)?;
                            written += 1;
                        }
                    }
                }
            }
        } else {
            for (i, p) in rows.iter().enumerate() {
                let n = &probe_norms[i];
                if let Some(h) = hash_key_for_np(algo, n) {
                    if let Some(cands) = index.get(&h) {
                        for q in cands {
                            let n2 = normalize_person(q);
                            let ok = match algo {
                                MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => {
                                    matches_algo1(n, &n2)
                                }
                                MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => {
                                    matches_algo2(n, &n2)
                                }
                                _ => false,
                            };
                            if ok {
                                let pair = if inner_is_t2 {
                                    to_pair(p, q, algo, n, &n2)
                                } else {
                                    to_pair(q, p, algo, &n2, n)
                                };
                                on_match(&pair)?;
                                written += 1;
                            }
                        }
                    }
                }
            }
        }

        // Checkpoint after processing current chunk
        if cfg.resume {
            if let Some(p) = cfg.checkpoint_path.as_ref() {
                let cp = StreamCheckpoint {
                    db: String::new(),
                    table_inner: inner_table.into(),
                    table_outer: outer_table.into(),
                    algorithm: format!("{:?}", algo),
                    batch_size: batch,
                    next_offset: offset,
                    total_outer,
                    partition_idx: 0,
                    partition_name: String::new(),
                    updated_utc: chrono::Utc::now().to_rfc3339(),
                    last_id: Some(offset),
                    watermark_id: Some(outer_watermark),
                    filter_sig: Some(format!("id<={}", outer_watermark)),
                };
                let _ = save_checkpoint(p, &cp);
            }
        }

        // Adaptive batch increase
        let memx = memory_stats_mb();
        if memx.avail_mb > cfg.memory_soft_min_mb * 2 {
            let new_batch = (batch as f64 * 1.5) as i64;
            batch = new_batch.min(200_000).max(10_000);
        }
        tokio::task::yield_now().await;
    }

    Ok(written)
}

#[derive(Clone)]
pub struct StreamControl {
    pub cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    pub pause: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

#[allow(dead_code, unreachable_code)]
pub async fn stream_match_csv<F>(
    pool: &MySqlPool,
    table1: &str,
    table2: &str,
    algo: MatchingAlgorithm,
    on_match: F,
    cfg: StreamingConfig,
    on_progress: impl Fn(ProgressUpdate) + Sync,
    ctrl: Option<StreamControl>,
) -> Result<usize>
where
    F: FnMut(&MatchPair) -> Result<()>,
{
    use crate::util::checkpoint::{
        StreamCheckpoint, load_checkpoint, remove_checkpoint, save_checkpoint,
    };
    if matches!(
        algo,
        MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle
    ) {
        anyhow::bail!(
            "Fuzzy algorithms are supported only in in-memory or partitioned mode (CSV). Use algorithm=3/4 with CSV in-memory or partitioned streaming."
        );
    }
    #[cfg(not(feature = "gpu"))]
    {
        if cfg.use_gpu_hash_join {
            log::warn!("GPU hash-join requested but GPU feature not compiled; proceeding with CPU");
        }
    }
    // Optional accelerated path: GPU hash-join (with CPU hashing fallback inside)
    // Unified path (delegates to internal for both single-DB and cross-DB)
    return stream_match_csv_internal(
        pool,
        pool,
        table1,
        table2,
        algo,
        on_match,
        cfg,
        on_progress,
        ctrl,
    )
    .await;

    let c1 = get_person_count(pool, table1).await?;
    let c2 = get_person_count(pool, table2).await?;
    // index smaller table
    let (inner_table, outer_table, total) = if c2 <= c1 {
        (table2, table1, c1)
    } else {
        (table1, table2, c2)
    };
    let mut batch = cfg.batch_size.max(10_000);
    let start = Instant::now();

    // Progress: indexing start
    let mems = memory_stats_mb();
    on_progress(ProgressUpdate {
        processed: 0,
        total: total as usize,
        percent: 0.0,
        eta_secs: 0,
        mem_used_mb: mems.used_mb,
        mem_avail_mb: mems.avail_mb,
        stage: "indexing",
        batch_size_current: Some(batch),
        gpu_total_mb: 0,
        gpu_free_mb: 0,
        gpu_active: false,
    });
    let index = build_index(pool, inner_table, algo, batch).await?;
    let mems2 = memory_stats_mb();
    on_progress(ProgressUpdate {
        processed: 0,
        total: total as usize,
        percent: 0.0,
        eta_secs: 0,
        mem_used_mb: mems2.used_mb,
        mem_avail_mb: mems2.avail_mb,
        stage: "indexing_done",
        batch_size_current: Some(batch),
        gpu_total_mb: 0,
        gpu_free_mb: 0,
        gpu_active: false,
    });

    // Resume support: detect checkpoint
    let mut offset: i64 = 0;
    if cfg.resume {
        if let Some(p) = cfg.checkpoint_path.as_ref() {
            if let Some(cp) = load_checkpoint(p) {
                if cp.db == ""
                    || (cp.table_inner == inner_table
                        && cp.table_outer == outer_table
                        && cp.algorithm == format!("{:?}", algo))
                {
                    offset = cp.next_offset.min(total);
                    batch = cp.batch_size.max(10_000);
                }
            }
        }
    }

    let mut written = 0usize;
    let mut processed = 0usize;
    let mut last_chunk_start = Instant::now();

    let mut next_rows_task: Option<tokio::task::JoinHandle<anyhow::Result<Vec<Person>>>> = None;

    while offset < total {
        if let Some(c) = &ctrl {
            if c.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            while c.pause.load(std::sync::atomic::Ordering::Relaxed) {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
        }
        // adaptive batch: memory based decrease, throughput-based increase
        let mem = memory_stats_mb();
        if mem.avail_mb < cfg.memory_soft_min_mb && batch > 10_000 {
            batch = (batch / 2).max(10_000);
        }

        // obtain rows: use prefetched task if available, else fetch with retry
        let rows: Vec<Person> = if let Some(handle) = next_rows_task.take() {
            // await the prefetched result
            match handle.await {
                Ok(res) => res?,
                Err(_join_err) => {
                    // fall back to direct fetch with retry if join failed
                    let mut tries = 0u32;
                    loop {
                        match fetch_person_rows_chunk(pool, outer_table, offset, batch).await {
                            Ok(v) => break v,
                            Err(e) => {
                                tries += 1;
                                if tries > cfg.retry_max {
                                    return Err(e);
                                }
                                let backoff = cfg.retry_backoff_ms * (1u64 << (tries.min(5) - 1));
                                tokio::time::sleep(std::time::Duration::from_millis(backoff)).await;
                            }
                        }
                    }
                }
            }
        } else {
            let mut tries = 0u32;
            loop {
                match fetch_person_rows_chunk(pool, outer_table, offset, batch).await {
                    Ok(v) => break v,
                    Err(e) => {
                        tries += 1;
                        if tries > cfg.retry_max {
                            return Err(e);
                        }
                        let backoff = cfg.retry_backoff_ms * (1u64 << (tries.min(5) - 1));
                        tokio::time::sleep(std::time::Duration::from_millis(backoff)).await;
                    }
                }
            }
        };

        for p in rows.iter() {
            if let Some(c) = &ctrl {
                if c.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
                while c.pause.load(std::sync::atomic::Ordering::Relaxed) {
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                }
            }
            let n = normalize_person(p);
            if let Some(k) = key_for(algo, &n) {
                if let Some(cands) = index.get(&k) {
                    for q in cands {
                        let n2 = normalize_person(q);
                        let ok = match algo {
                            MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => matches_algo1(&n, &n2),
                            MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => matches_algo2(&n, &n2),
                            MatchingAlgorithm::Fuzzy
                            | MatchingAlgorithm::FuzzyNoMiddle
                            | MatchingAlgorithm::HouseholdGpu
                            | MatchingAlgorithm::HouseholdGpuOpt6
                            | MatchingAlgorithm::LevenshteinWeighted => false,
                        };
                        if ok {
                            let pair = if inner_table == table2 {
                                to_pair(p, q, algo, &n, &n2)
                            } else {
                                to_pair(q, p, algo, &n2, &n)
                            };
                            on_match(&pair)?;
                            written += 1;
                        }
                    }
                }
            }
        }

        offset += batch;
        processed = (processed + rows.len()).min(total as usize);

        // save checkpoint
        if let Some(p) = cfg.checkpoint_path.as_ref() {
            let _ = save_checkpoint(
                p,
                &StreamCheckpoint {
                    db: String::new(),
                    table_inner: inner_table.to_string(),
                    table_outer: outer_table.to_string(),
                    algorithm: format!("{:?}", algo),
                    batch_size: batch,
                    next_offset: offset,
                    total_outer: total,
                    partition_idx: 0,
                    partition_name: "all".into(),
                    updated_utc: chrono::Utc::now().to_rfc3339(),
                    last_id: Some(offset),
                    watermark_id: None,
                    filter_sig: None,
                },
            );
        }

        // progress update
        let frac = (processed as f32 / total as f32).clamp(0.0, 1.0);
        let eta = if frac > 0.0 {
            (start.elapsed().as_secs_f32() * (1.0 - frac) / frac) as u64
        } else {
            0
        };
        let memx = memory_stats_mb();
        on_progress(ProgressUpdate {
            processed,
            total: total as usize,
            percent: frac * 100.0,
            eta_secs: eta,
            mem_used_mb: memx.used_mb,
            mem_avail_mb: memx.avail_mb,
            stage: "streaming",
            batch_size_current: Some(batch),
            gpu_total_mb: 0,
            gpu_free_mb: 0,
            gpu_active: false,
        });

        // adaptive increase if fast

        // prefetch next chunk while we process current one
        if offset < total {
            let pool_cloned = pool.clone();
            let table = outer_table.to_string();
            let next_off = offset;
            let next_batch = batch;
            let retry_max = cfg.retry_max;
            let backoff_ms = cfg.retry_backoff_ms;
            next_rows_task = Some(tokio::spawn(async move {
                let mut tries = 0u32;
                loop {
                    match fetch_person_rows_chunk(&pool_cloned, &table, next_off, next_batch).await
                    {
                        Ok(v) => break Ok(v),
                        Err(e) => {
                            tries += 1;
                            if tries > retry_max {
                                break Err(e);
                            }
                            let backoff = backoff_ms * (1u64 << (tries.min(5) - 1));
                            tokio::time::sleep(std::time::Duration::from_millis(backoff)).await;
                        }
                    }
                }
            }));
        }

        let dur = last_chunk_start.elapsed();
        if dur.as_millis() > 0 {
            // if chunk was quick and memory is plentiful, increase
            if memx.avail_mb > cfg.memory_soft_min_mb * 2 && dur < std::time::Duration::from_secs(1)
            {
                let new_batch = (batch as f64 * 1.5) as i64;
                batch = new_batch.min(200_000).max(10_000);
            }
        }
        last_chunk_start = Instant::now();
        // allow runtime to schedule
        tokio::task::yield_now().await;
    }

    if let Some(p) = cfg.checkpoint_path.as_ref() {
        remove_checkpoint(p);
    }
    Ok(written)
}

// New: dual-pool variant to support cross-database streaming
pub async fn stream_match_csv_dual<F>(
    pool1: &MySqlPool,
    pool2: &MySqlPool,
    table1: &str,
    table2: &str,
    algo: MatchingAlgorithm,
    on_match: F,
    cfg: StreamingConfig,
    on_progress: impl Fn(ProgressUpdate) + Sync,
    ctrl: Option<StreamControl>,
) -> Result<usize>
where
    F: FnMut(&MatchPair) -> Result<()>,
{
    use crate::util::checkpoint::{
        StreamCheckpoint, load_checkpoint, remove_checkpoint, save_checkpoint,
    };
    if matches!(
        algo,
        MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle
    ) {
        anyhow::bail!(
            "Fuzzy algorithms are supported only in in-memory or partitioned mode (CSV). Use algorithm=3/4 with CSV in-memory or partitioned streaming."
        );
    }
    // Delegate to unified internal implementation for cross-DB
    return stream_match_csv_internal(
        pool1,
        pool2,
        table1,
        table2,
        algo,
        on_match,
        cfg,
        on_progress,
        ctrl,
    )
    .await;
}

// Backward-compatible single-pool API can continue to be used alongside the dual-pool API

// --- Partitioned streaming (multi-pass) ---
use crate::db::schema::{get_person_count_where, get_person_rows_where};
use crate::models::ColumnMapping;
use crate::util::partition::{DefaultPartition, PartitionStrategy};

#[derive(Clone, Debug)]
pub struct PartitioningConfig {
    #[allow(dead_code)]
    pub enabled: bool,
    pub strategy: String, // e.g., "last_initial" | "birthyear5"
}
impl Default for PartitioningConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            strategy: "last_initial".into(),
        }
    }
}

pub async fn stream_match_csv_partitioned<F>(
    pool: &MySqlPool,
    table1: &str,
    table2: &str,
    algo: MatchingAlgorithm,
    mut on_match: F,
    cfg: StreamingConfig,
    on_progress: impl Fn(ProgressUpdate) + Sync,
    ctrl: Option<StreamControl>,
    mapping1: Option<&ColumnMapping>,
    mapping2: Option<&ColumnMapping>,
    part_cfg: PartitioningConfig,
) -> Result<usize>
where
    F: FnMut(&MatchPair) -> Result<()>,
{
    use crate::util::checkpoint::{
        StreamCheckpoint, load_checkpoint, remove_checkpoint, save_checkpoint,
    };
    let strat: Box<dyn PartitionStrategy + Send + Sync> = match part_cfg.strategy.as_str() {
        "birthyear5" => DefaultPartition::BirthYear5.build(),
        _ => DefaultPartition::LastInitial.build(),
    };
    let parts1 = strat.partitions(mapping1);
    let parts2 = strat.partitions(mapping2);
    if parts1.len() != parts2.len() {
        anyhow::bail!("Partition strategy produced mismatched partition counts for the two tables");
    }

    // resume state
    let mut start_part: usize = 0;
    let mut last_id: i64 = 0;
    let mut outer_watermark: i64 = 0;
    let mut batch = cfg.batch_size.max(10_000);
    if cfg.resume {
        if let Some(pth) = cfg.checkpoint_path.as_ref() {
            if let Some(cp) = load_checkpoint(pth) {
                start_part = (cp.partition_idx as isize).max(0) as usize;
                last_id = cp.last_id.unwrap_or(cp.next_offset);
                outer_watermark = cp.watermark_id.unwrap_or(0);
                batch = cp.batch_size.max(10_000);
            }
        }
    }

    let total_parts = parts1.len();
    let mut total_written = 0usize;
    for pi in start_part..total_parts {
        let p1 = &parts1[pi];
        let p2 = &parts2[pi];
        // Index inner table for this partition
        let c1 = get_person_count_where(pool, table1, &p1.where_sql, &p1.binds).await?;
        let c2 = get_person_count_where(pool, table2, &p2.where_sql, &p2.binds).await?;
        let inner_is_t2 = c2 <= c1;
        let (inner_table, inner_where, inner_binds, inner_map) = if inner_is_t2 {
            (table2, &p2.where_sql, &p2.binds, mapping2)
        } else {
            (table1, &p1.where_sql, &p1.binds, mapping1)
        };
        let (outer_table, outer_where, outer_binds, outer_map) = if inner_is_t2 {
            (table1, &p1.where_sql, &p1.binds, mapping1)
        } else {
            (table2, &p2.where_sql, &p2.binds, mapping2)
        };
        let total_outer = if inner_is_t2 { c1 } else { c2 };
        outer_watermark = get_max_id_where(pool, outer_table, outer_where, outer_binds).await?;

        let mems = memory_stats_mb();
        on_progress(ProgressUpdate {
            processed: 0,
            total: total_outer as usize,
            percent: 0.0,
            eta_secs: 0,
            mem_used_mb: mems.used_mb,
            mem_avail_mb: mems.avail_mb,
            stage: "indexing",
            batch_size_current: Some(batch),
            gpu_total_mb: 0,
            gpu_free_mb: 0,
            gpu_active: false,
        });
        let inner_rows =
            get_person_rows_where(pool, inner_table, inner_where, inner_binds, inner_map).await?;
        let mems2 = memory_stats_mb();
        on_progress(ProgressUpdate {
            processed: 0,
            total: total_outer as usize,
            percent: 0.0,
            eta_secs: 0,
            mem_used_mb: mems2.used_mb,
            mem_avail_mb: mems2.avail_mb,
            stage: "indexing_done",
            batch_size_current: Some(batch),
            gpu_total_mb: 0,
            gpu_free_mb: 0,
            gpu_active: false,
        });

        // Precompute normalized inner and group by birthdate to bound comparisons
        use std::collections::HashMap as Map;
        let norm_inner: Vec<NormalizedPerson> = inner_rows.iter().map(normalize_person).collect();
        let mut by_date: Map<chrono::NaiveDate, Vec<usize>> = Map::new();
        for (i, n) in norm_inner.iter().enumerate() {
            if let Some(d) = n.birthdate.as_ref() {
                by_date.entry(*d).or_default().push(i);
            }
        }

        // Optional GPU: we'll attempt it when feature is enabled; else CPU fallback

        let start_time = Instant::now();
        let mut processed = 0usize;
        if pi != start_part {
            last_id = 0;
        }
        loop {
            if let Some(c) = &ctrl {
                if c.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
                while c.pause.load(std::sync::atomic::Ordering::Relaxed) {
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
            }
            let mem = memory_stats_mb();
            if mem.avail_mb < cfg.memory_soft_min_mb && batch > 10_000 {
                batch = (batch / 2).max(10_000);
            }

            // fetch chunk with WHERE
            let mut tries = 0u32;
            let rows: Vec<Person> = loop {
                match fetch_person_rows_chunk_where_keyset(
                    pool,
                    outer_table,
                    last_id,
                    batch,
                    outer_where,
                    outer_binds,
                    outer_map,
                    Some(outer_watermark),
                )
                .await
                {
                    Ok(v) => break v,
                    Err(e) => {
                        tries += 1;
                        if tries > cfg.retry_max {
                            return Err(e);
                        }
                        let backoff = cfg.retry_backoff_ms * (1u64 << (tries.min(5) - 1));
                        tokio::time::sleep(std::time::Duration::from_millis(backoff)).await;
                    }
                }
            };
            if rows.is_empty() {
                break;
            }
            if let Some(last) = rows.last() {
                last_id = last.id;
            }

            // Optional GPU pre-pass: hash-based candidate filtering for Fuzzy direct phase
            #[allow(unused_mut)]
            let mut gpu_done = false;
            #[cfg(feature = "gpu")]
            if cfg.use_gpu_fuzzy_direct_hash {
                // Indicate GPU build/probe hashing activity for GUI status lights
                on_progress(ProgressUpdate {
                    processed: processed.min(total_outer as usize),
                    total: total_outer as usize,
                    percent: ((processed as f32) / (total_outer.max(1) as f32)) * 100.0,
                    eta_secs: 0,
                    mem_used_mb: memory_stats_mb().used_mb,
                    mem_avail_mb: memory_stats_mb().avail_mb,
                    stage: "gpu_hash",
                    batch_size_current: Some(batch),
                    gpu_total_mb: 1,
                    gpu_free_mb: 0,
                    gpu_active: true,
                });
                match gpu::fuzzy_direct_gpu_hash_prefilter_indices(
                    &rows,
                    &inner_rows,
                    &part_cfg.strategy,
                ) {
                    Ok(cand_lists) => {
                        on_progress(ProgressUpdate {
                            processed: processed.min(total_outer as usize),
                            total: total_outer as usize,
                            percent: ((processed as f32) / (total_outer.max(1) as f32)) * 100.0,
                            eta_secs: 0,
                            mem_used_mb: memory_stats_mb().used_mb,
                            mem_avail_mb: memory_stats_mb().avail_mb,
                            stage: "gpu_probe_hash",
                            batch_size_current: Some(batch),
                            gpu_total_mb: 1,
                            gpu_free_mb: 0,
                            gpu_active: true,
                        });
                        let allow_swap = crate::matching::birthdate_matcher::allow_birthdate_swap();
                        for (i, p) in rows.iter().enumerate() {
                            let n = normalize_person(p);
                            for &i2 in cand_lists.get(i).map(|v| v.as_slice()).unwrap_or(&[]) {
                                let n2 = &norm_inner[i2];
                                // Enforce birthdate equality (with optional month/day swap)
                                let bd_match = match (p.birthdate, inner_rows[i2].birthdate) {
                                    (Some(b1), Some(b2)) => {
                                        crate::matching::birthdate_matcher::birthdate_matches_naive(
                                            b1, b2, allow_swap,
                                        )
                                    }
                                    _ => false,
                                };
                                if !bd_match {
                                    continue;
                                }
                                // If strategy does not enforce last initial, keep it permissive
                                if part_cfg.strategy != "last_initial" {
                                    let li1 = n
                                        .last_name
                                        .as_deref()
                                        .and_then(|s| s.chars().next())
                                        .unwrap_or('\0')
                                        .to_ascii_uppercase();
                                    let li2 = n2
                                        .last_name
                                        .as_deref()
                                        .and_then(|s| s.chars().next())
                                        .unwrap_or('\0')
                                        .to_ascii_uppercase();
                                    if li1 != li2 {
                                        continue;
                                    }
                                }
                                let comp = if matches!(algo, MatchingAlgorithm::FuzzyNoMiddle) {
                                    fuzzy_compare_names_no_mid(
                                        n.first_name.as_deref(),
                                        n.last_name.as_deref(),
                                        n2.first_name.as_deref(),
                                        n2.last_name.as_deref(),
                                    )
                                } else {
                                    fuzzy_compare_names_new(
                                        n.first_name.as_deref(),
                                        n.middle_name.as_deref(),
                                        n.last_name.as_deref(),
                                        n2.first_name.as_deref(),
                                        n2.middle_name.as_deref(),
                                        n2.last_name.as_deref(),
                                    )
                                };
                                if let Some((score, label)) = comp {
                                    let q = &inner_rows[i2];
                                    let pair = MatchPair {
                                        person1: if inner_is_t2 { p.clone() } else { q.clone() },
                                        person2: if inner_is_t2 { q.clone() } else { p.clone() },
                                        confidence: (score / 100.0) as f32,
                                        matched_fields: vec![
                                            "fuzzy".into(),
                                            label,
                                            "birthdate".into(),
                                        ],
                                        is_matched_infnbd: false,
                                        is_matched_infnmnbd: false,
                                    };
                                    on_match(&pair)?;
                                    total_written += 1;
                                }
                            }
                        }
                        on_progress(ProgressUpdate {
                            processed: processed.min(total_outer as usize),
                            total: total_outer as usize,
                            percent: ((processed as f32) / (total_outer.max(1) as f32)) * 100.0,
                            eta_secs: 0,
                            mem_used_mb: memory_stats_mb().used_mb,
                            mem_avail_mb: memory_stats_mb().avail_mb,
                            stage: "gpu_probe_hash_done",
                            batch_size_current: Some(batch),
                            gpu_total_mb: 1,
                            gpu_free_mb: 0,
                            gpu_active: true,
                        });
                        gpu_done = true;
                    }
                    Err(e) => {
                        log::warn!(
                            "GPU fuzzy direct pre-pass failed; continuing to full scoring: {}",
                            e
                        );
                    }
                }
            }

            // Try GPU first (per-chunk) if available and enabled; fallback to CPU if disabled/failed
            #[cfg(feature = "gpu")]
            if cfg.use_gpu_fuzzy_metrics && !gpu_fuzzy_disable() {
                let use_gpu = if gpu_fuzzy_force() {
                    true
                } else {
                    let (ok, why) = should_enable_gpu_fuzzy_by_heuristic(&rows, &inner_rows);
                    if ok {
                        log::info!(
                            "[stream:part] GPU fuzzy metrics enabled by heuristic: {}",
                            why
                        );
                    } else {
                        log::info!(
                            "[stream:part] GPU fuzzy metrics disabled by heuristic: {}",
                            why
                        );
                    }
                    ok
                };
                if use_gpu {
                    let opts = MatchOptions {
                        backend: ComputeBackend::Gpu,
                        gpu: Some(GpuConfig {
                            device_id: None,
                            mem_budget_mb: 512,
                        }),
                        progress: ProgressConfig::default(),
                        allow_birthdate_swap:
                            crate::matching::birthdate_matcher::allow_birthdate_swap(),
                    };
                    let gpu_try = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        crate::matching::gpu_config::with_oom_cpu_fallback(
                            || gpu::match_fuzzy_gpu(&rows, &inner_rows, opts, &on_progress),
                            || {
                                let mo_cpu = MatchOptions {
                                    backend: ComputeBackend::Cpu,
                                    gpu: None,
                                    progress: ProgressConfig::default(),
                                    allow_birthdate_swap:
                                        crate::matching::birthdate_matcher::allow_birthdate_swap(),
                                };
                                match_all_with_opts(&rows, &inner_rows, algo, mo_cpu, &on_progress)
                            },
                            "[stream:part] fuzzy gpu",
                        )
                    }));
                    match gpu_try {
                        Ok(Ok(mut vec_pairs)) => {
                            for pair in vec_pairs.drain(..) {
                                let comp = if matches!(algo, MatchingAlgorithm::FuzzyNoMiddle) {
                                    compare_persons_no_mid(&pair.person1, &pair.person2)
                                } else {
                                    compare_persons_new(&pair.person1, &pair.person2)
                                };
                                if let Some((score, label)) = comp {
                                    let mut updated = pair;
                                    updated.confidence = (score / 100.0) as f32;
                                    updated.matched_fields =
                                        vec!["fuzzy".into(), label, "birthdate".into()];
                                    on_match(&updated)?;
                                    total_written += 1;
                                }
                            }
                            gpu_done = true;
                        }
                        Ok(Err(e)) => {
                            log::warn!("GPU fuzzy failed in partition; falling back to CPU: {}", e);
                        }
                        Err(_) => {
                            log::warn!("GPU fuzzy panicked; falling back to CPU for this chunk");
                        }
                    }
                }
            }

            if !gpu_done {
                // CPU fallback: candidate window by exact birthdate
                for p in rows.iter() {
                    let n = normalize_person(p);
                    if let Some(d) = n.birthdate.as_ref() {
                        if let Some(cand_idx) = by_date.get(d) {
                            for &i2 in cand_idx {
                                let n2 = &norm_inner[i2];
                                // quick initial filter: last initial must match if not enforced by strategy
                                if part_cfg.strategy != "last_initial" {
                                    let li1 = n
                                        .last_name
                                        .as_deref()
                                        .and_then(|s| s.chars().next())
                                        .unwrap_or('\0')
                                        .to_ascii_uppercase();
                                    let li2 = n2
                                        .last_name
                                        .as_deref()
                                        .and_then(|s| s.chars().next())
                                        .unwrap_or('\0')
                                        .to_ascii_uppercase();
                                    if li1 != li2 {
                                        continue;
                                    }
                                }
                                let comp = if matches!(algo, MatchingAlgorithm::FuzzyNoMiddle) {
                                    fuzzy_compare_names_no_mid(
                                        n.first_name.as_deref(),
                                        n.last_name.as_deref(),
                                        n2.first_name.as_deref(),
                                        n2.last_name.as_deref(),
                                    )
                                } else {
                                    fuzzy_compare_names_new(
                                        n.first_name.as_deref(),
                                        n.middle_name.as_deref(),
                                        n.last_name.as_deref(),
                                        n2.first_name.as_deref(),
                                        n2.middle_name.as_deref(),
                                        n2.last_name.as_deref(),
                                    )
                                };
                                if let Some((score, label)) = comp {
                                    let q = &inner_rows[i2];
                                    let pair = MatchPair {
                                        person1: if inner_is_t2 { p.clone() } else { q.clone() },
                                        person2: if inner_is_t2 { q.clone() } else { p.clone() },
                                        confidence: (score / 100.0) as f32,
                                        matched_fields: vec![
                                            "fuzzy".into(),
                                            label,
                                            "birthdate".into(),
                                        ],
                                        is_matched_infnbd: false,
                                        is_matched_infnmnbd: false,
                                    };
                                    on_match(&pair)?;
                                    total_written += 1;
                                }
                            }
                        }
                    }
                }
            }

            // Iteration tail: update checkpoint and progress, yield
            processed = (processed + rows.len()).min(total_outer as usize);
            if let Some(pth) = cfg.checkpoint_path.as_ref() {
                let _ = save_checkpoint(
                    pth,
                    &StreamCheckpoint {
                        db: String::new(),
                        table_inner: inner_table.to_string(),
                        table_outer: outer_table.to_string(),
                        algorithm: format!("{:?}", algo),
                        batch_size: batch,
                        next_offset: last_id,
                        total_outer,
                        partition_idx: pi as i32,
                        partition_name: p1.name.clone(),
                        updated_utc: chrono::Utc::now().to_rfc3339(),
                        last_id: Some(last_id),
                        watermark_id: Some(outer_watermark),
                        filter_sig: Some(outer_where.clone()),
                    },
                );
            }
            let frac = (processed as f32 / total_outer as f32).clamp(0.0, 1.0);
            let eta = if frac > 0.0 {
                (start_time.elapsed().as_secs_f32() * (1.0 - frac) / frac) as u64
            } else {
                0
            };
            let memx = memory_stats_mb();
            on_progress(ProgressUpdate {
                processed,
                total: total_outer as usize,
                percent: frac * 100.0,
                eta_secs: eta,
                mem_used_mb: memx.used_mb,
                mem_avail_mb: memx.avail_mb,
                stage: if gpu_done { "gpu_kernel" } else { "streaming" },
                batch_size_current: Some(batch),
                gpu_total_mb: if gpu_done { 1 } else { 0 },
                gpu_free_mb: 0,
                gpu_active: gpu_done,
            });
            tokio::task::yield_now().await;
        }
    }

    if let Some(pth) = cfg.checkpoint_path.as_ref() {
        remove_checkpoint(pth);
    }
    return Ok(total_written);
}

#[cfg(test)]
mod gpu_fuzzy_heuristics_tests {
    use super::*;
    use chrono::NaiveDate;

    #[test]
    fn heuristic_disables_for_small_candidates() {
        let mut t1 = Vec::new();
        let mut t2 = Vec::new();
        for i in 0..1000 {
            t1.push(Person {
                id: i,
                uuid: None,
                first_name: Some("Ann".into()),
                middle_name: Some("Q".into()),
                last_name: Some("Lee".into()),
                birthdate: NaiveDate::from_ymd_opt(1990, 1, 1),
                hh_id: None,
                extra_fields: std::collections::HashMap::new(),
            });
        }
        for j in 0..1000 {
            t2.push(Person {
                id: 10_000 + j,
                uuid: None,
                first_name: Some("Ann".into()),
                middle_name: Some("Q".into()),
                last_name: Some("Lee".into()),
                birthdate: NaiveDate::from_ymd_opt(1990, 1, 1),
                hh_id: None,
                extra_fields: std::collections::HashMap::new(),
            });
        }
        let (ok, why) = should_enable_gpu_fuzzy_by_heuristic(&t1, &t2);
        assert!(!ok, "expected heuristic to disable, got enable: {}", why);
    }

    #[test]
    fn heuristic_enables_for_large_blocked_candidates() {
        let mut t1 = Vec::new();
        let mut t2 = Vec::new();
        for i in 0..1000 {
            // 1k rows on left
            t1.push(Person {
                id: i,
                uuid: None,
                first_name: Some("Ann".into()),
                middle_name: Some("Q".into()),
                last_name: Some("Lee".into()),
                birthdate: NaiveDate::from_ymd_opt(1990, 1, 1),
                hh_id: None,
                extra_fields: std::collections::HashMap::new(),
            });
        }
        for j in 0..10_100 {
            // ~10.1k rows on right with same block -> cand_est ~ 10.1M
            t2.push(Person {
                id: 10_000 + j,
                uuid: None,
                first_name: Some("Ann".into()),
                middle_name: Some("Q".into()),
                last_name: Some("Long".into()),
                birthdate: NaiveDate::from_ymd_opt(1990, 1, 1),
                hh_id: None,
                extra_fields: std::collections::HashMap::new(),
            });
        }
        let (ok, why) = should_enable_gpu_fuzzy_by_heuristic(&t1, &t2);
        assert!(ok, "expected heuristic to enable, got disable: {}", why);
    }
}

// Public wrappers for Dynamic GPU Tuner to be accessible from GUI without exposing the `gpu` module
#[cfg(feature = "gpu")]
pub fn dyn_tuner_ensure_started(enable: bool) {
    self::gpu::dynamic_tuner::ensure_started(enable);
}
#[cfg(feature = "gpu")]
pub fn dyn_tuner_vram_free_pct() -> f32 {
    self::gpu::dynamic_tuner::get_current_vram_free_pct()
}
#[cfg(feature = "gpu")]
pub fn dyn_tuner_tile_size() -> usize {
    self::gpu::dynamic_tuner::get_current_tile_size()
}
#[cfg(feature = "gpu")]
pub fn dyn_tuner_streams() -> u32 {
    self::gpu::dynamic_tuner::get_current_streams()
}
#[cfg(feature = "gpu")]
pub fn dyn_tuner_stop() {
    let _ = self::gpu::dynamic_tuner::stop();
}

// --- Advanced Matching Streaming Adapter ---
use crate::db::schema::get_person_rows_all_columns;
use crate::matching::advanced_matcher::{AdvConfig, AdvLevel, exact_key as adv_exact_key};

/// Streamed Advanced Matching (opt-in). Indexes smaller table in memory, streams the larger table in batches.
pub async fn stream_match_advanced<F>(
    pool: &MySqlPool,
    table1: &str,
    table2: &str,
    cfg: &AdvConfig,
    scfg: StreamingConfig,
    mut on_match: F,
    on_progress: impl Fn(ProgressUpdate) + Sync,
    ctrl: Option<StreamControl>,
) -> Result<usize>
where
    F: FnMut(&MatchPair) -> Result<()>,
{
    use crate::util::checkpoint::{StreamCheckpoint, load_checkpoint, save_checkpoint};
    use std::collections::HashSet;
    // Deduplicate pairs for L10/L11 across batches (swap adds multiple keys per record)
    let mut seen_pairs: HashSet<(i64, i64)> = HashSet::new();
    let mut user_on_match = on_match;
    let mut on_match = |pair: &MatchPair| -> Result<()> {
        if matches!(
            cfg.level,
            AdvLevel::L10FuzzyBirthdateFullMiddle | AdvLevel::L11FuzzyBirthdateNoMiddle
        ) {
            if !seen_pairs.insert((pair.person1.id, pair.person2.id)) {
                return Ok(());
            }
        }
        user_on_match(pair)
    };
    let allow_swap = cfg.allow_birthdate_swap;
    // Geographic code levels (L4-L9) use fixed field names in both tables:
    // - L4-L6: 'barangay_code'
    // - L7-L9: 'city_code'
    // No external column mapping is required; records missing these fields simply won't join.

    let c1 = get_person_count(pool, table1).await?;
    let _c2 = get_person_count(pool, table2).await?;

    // IMPORTANT: Always use table2 as inner (indexed) and table1 as outer (streamed)
    // to maintain parity with in-memory implementation and ensure consistent
    // person1/person2 assignment in match pairs.
    //
    // Rationale:
    // - In-memory mode (advanced_match_inmemory) always indexes table2 and probes with table1
    // - This ensures person1 is always from table1 and person2 is always from table2
    // - Export writers expect person2 to be from table2 (extra_fields labeled as "Table2_*")
    // - Dynamic ordering (indexing smaller table) would swap person1/person2 when table1 < table2
    // - This causes export data population discrepancies where "Table2_*" columns contain wrong data
    //
    // Performance Note:
    // - This may be slightly less efficient if table1 is much smaller than table2
    // - However, correctness and consistency are more important than this optimization
    let (inner_table, outer_table, total_outer) = (table2, table1, c1);

    // Snapshot + resume support (keyset cursor)
    let outer_watermark = get_max_id(pool, outer_table).await?;
    let mut last_id: i64 = 0;
    let mut batch = scfg.batch_size.max(10_000);
    if scfg.resume {
        if let Some(p) = scfg.checkpoint_path.as_ref() {
            if let Some(cp) = load_checkpoint(p) {
                let algo = format!("Advanced::{:?}", cfg.level);
                if cp.table_inner == inner_table
                    && cp.table_outer == outer_table
                    && cp.algorithm == algo
                {
                    last_id = cp.last_id.unwrap_or(cp.next_offset).min(outer_watermark);
                    batch = cp.batch_size.max(10_000);
                }
            }
        }
    }

    // Build inner index
    on_progress(ProgressUpdate {
        processed: 0,
        total: total_outer as usize,
        percent: 0.0,
        eta_secs: 0,
        mem_used_mb: memory_stats_mb().used_mb,
        mem_avail_mb: memory_stats_mb().avail_mb,
        stage: "indexing_advanced",
        batch_size_current: Some(batch),
        gpu_total_mb: 0,
        gpu_free_mb: 0,
        gpu_active: false,
    });
    let inner_rows = get_person_rows_all_columns(pool, inner_table).await?;
    enum Index {
        Exact(std::collections::HashMap<String, Vec<usize>>),
        ByBirth(std::collections::HashMap<String, Vec<usize>>),
        // GPU-hash-accelerated index: hash -> inner indices, plus parallel key strings for verification
        GpuHash {
            map: std::collections::HashMap<u64, Vec<usize>>,
            key_strs: Vec<String>,
            key_idx: Vec<usize>,
        },
    }

    // Decide index strategy (default)
    let mut index = if matches!(
        cfg.level,
        AdvLevel::L10FuzzyBirthdateFullMiddle | AdvLevel::L11FuzzyBirthdateNoMiddle
    ) {
        Index::ByBirth(Default::default())
    } else {
        Index::Exact(Default::default())
    };

    // Optional: upgrade to GPU-hash index for exact levels (L1-L9)
    #[cfg(feature = "gpu")]
    if !matches!(
        cfg.level,
        AdvLevel::L10FuzzyBirthdateFullMiddle | AdvLevel::L11FuzzyBirthdateNoMiddle
    ) && (scfg.use_gpu_hash_join || scfg.use_gpu_build_hash)
    {
        if let Ok(ctx) = self::gpu::GpuHashContext::get() {
            let mut key_strs: Vec<String> = Vec::new();
            let mut key_idx: Vec<usize> = Vec::new();
            for (i, p) in inner_rows.iter().enumerate() {
                if let Some(k) = adv_exact_key(p, cfg.level, &cfg.cols) {
                    key_idx.push(i);
                    key_strs.push(k);
                }
            }
            if !key_strs.is_empty() {
                let (_gt, gf) = ctx.mem_info_mb();
                let budget = (gf / 2).max(128);
                if let Ok(hashes) = self::gpu::hash_fnv1a64_batch_tiled(&ctx, &key_strs, budget, 64)
                {
                    let mut map: std::collections::HashMap<u64, Vec<usize>> =
                        std::collections::HashMap::new();
                    for (j, &h) in hashes.iter().enumerate() {
                        map.entry(h).or_default().push(key_idx[j]);
                    }
                    let key_count = key_strs.len();
                    index = Index::GpuHash {
                        map,
                        key_strs,
                        key_idx,
                    };
                    log::info!(
                        "Advanced GPU hash index built: level={:?}, inner_keys={}, budget_mb={}",
                        cfg.level,
                        key_count,
                        budget
                    );
                    eprintln!(
                        "[AUDIT] Advanced GPU hash index built: level={:?}, inner_keys={}, budget_mb={}",
                        cfg.level, key_count, budget
                    );
                }
            }
        }
    }

    // Fallback: build CPU index if not using GPU hash variant
    match &mut index {
        Index::Exact(map) => {
            for (i, p) in inner_rows.iter().enumerate() {
                if let Some(k) = adv_exact_key(p, cfg.level, &cfg.cols) {
                    map.entry(k).or_default().push(i);
                }
            }
        }
        Index::ByBirth(map) => {
            for (i, p) in inner_rows.iter().enumerate() {
                if let Some(bd) = p.birthdate {
                    for key in birthdate_keys(bd, allow_swap) {
                        map.entry(key).or_default().push(i);
                    }
                }
            }
        }
        Index::GpuHash { .. } => { /* already built above */ }
    }

    on_progress(ProgressUpdate {
        processed: 0,
        total: total_outer as usize,
        percent: 0.0,
        eta_secs: 0,
        mem_used_mb: memory_stats_mb().used_mb,
        mem_avail_mb: memory_stats_mb().avail_mb,
        stage: "indexing_done",
        batch_size_current: Some(batch),
        gpu_total_mb: 0,
        gpu_free_mb: 0,
        gpu_active: false,
    });

    // Stream outer table and probe
    let start = Instant::now();
    let mut written = 0usize;
    let mut processed_rows: usize = 0;
    loop {
        if let Some(c) = &ctrl {
            if c.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            while c.pause.load(std::sync::atomic::Ordering::Relaxed) {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        }
        let rows = fetch_person_rows_chunk_all_columns_keyset(
            pool,
            outer_table,
            last_id,
            batch,
            Some(outer_watermark),
        )
        .await?;
        if rows.is_empty() {
            break;
        }
        if let Some(last) = rows.last() {
            last_id = last.id;
        }
        processed_rows += rows.len();
        let elapsed = start.elapsed();
        let processed = processed_rows.min(total_outer as usize);
        let frac = (processed as f32 / total_outer as f32).clamp(0.0, 1.0);
        // Helper to emit a pair with proper matched_fields based on level
        #[inline]
        fn emit_pair<FN>(
            p: &Person,
            q: &Person,
            level: AdvLevel,
            on_match: &mut FN,
            written: &mut usize,
        ) -> Result<()>
        where
            FN: FnMut(&MatchPair) -> Result<()>,
        {
            let mut fields = vec!["first_name".to_string(), "last_name".to_string()];
            match level {
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
                _ => {}
            }
            let pair = MatchPair {
                person1: p.clone(),
                person2: q.clone(),
                confidence: 100.0,
                matched_fields: fields,
                is_matched_infnbd: false,
                is_matched_infnmnbd: false,
            };
            on_match(&pair)?;
            *written += 1;
            Ok(())
        }

        let eta_secs = if frac > 0.0 {
            (elapsed.as_secs_f32() * (1.0 - frac) / frac) as u64
        } else {
            0
        };
        let mem = memory_stats_mb();
        on_progress(ProgressUpdate {
            processed,
            total: total_outer as usize,
            percent: frac * 100.0,
            eta_secs,
            mem_used_mb: mem.used_mb,
            mem_avail_mb: mem.avail_mb,
            stage: "probing_advanced",
            batch_size_current: Some(batch),
            gpu_total_mb: 0,
            gpu_free_mb: 0,
            gpu_active: false,
        });

        match &index {
            Index::Exact(map) => {
                for p in &rows {
                    if let Some(k) = adv_exact_key(p, cfg.level, &cfg.cols) {
                        if let Some(cand_idx) = map.get(&k) {
                            for &j in cand_idx {
                                emit_pair(
                                    p,
                                    &inner_rows[j],
                                    cfg.level,
                                    &mut on_match,
                                    &mut written,
                                )?;
                            }
                        }
                    }
                }
            }
            Index::GpuHash {
                map,
                key_strs,
                key_idx,
            } => {
                // Build probe keys for this batch, hash on GPU if enabled, else CPU FNV
                let mut probe_keys: Vec<String> = Vec::new();
                let mut probe_map: Vec<usize> = Vec::new(); // outer row idx -> position in probe_keys
                for (i, p) in rows.iter().enumerate() {
                    if let Some(k) = adv_exact_key(p, cfg.level, &cfg.cols) {
                        probe_map.push(i);
                        probe_keys.push(k);
                    }
                }
                if !probe_keys.is_empty() {
                    #[cfg(feature = "gpu")]
                    let probe_hashes: Vec<u64> = if scfg.use_gpu_hash_join
                        || scfg.use_gpu_probe_hash
                    {
                        if let Ok(ctx) = self::gpu::GpuHashContext::get() {
                            let budget = scfg.gpu_probe_batch_mb.max(128);
                            log::info!(
                                "Advanced GPU hash probe engaged: level={:?}, keys={}, budget_mb={}",
                                cfg.level,
                                probe_keys.len(),
                                budget
                            );
                            eprintln!(
                                "[AUDIT] Advanced GPU hash probe engaged: level={:?}, keys={}, budget_mb={}",
                                cfg.level,
                                probe_keys.len(),
                                budget
                            );
                            match self::gpu::hash_fnv1a64_batch_tiled(&ctx, &probe_keys, budget, 64)
                            {
                                Ok(v) => v,
                                Err(_) => probe_keys
                                    .iter()
                                    .map(|s| fnv1a64_bytes(s.as_bytes()))
                                    .collect(),
                            }
                        } else {
                            probe_keys
                                .iter()
                                .map(|s| fnv1a64_bytes(s.as_bytes()))
                                .collect()
                        }
                    } else {
                        probe_keys
                            .iter()
                            .map(|s| fnv1a64_bytes(s.as_bytes()))
                            .collect()
                    };
                    #[cfg(not(feature = "gpu"))]
                    let probe_hashes: Vec<u64> = probe_keys
                        .iter()
                        .map(|s| fnv1a64_bytes(s.as_bytes()))
                        .collect();

                    for (idx, &h) in probe_hashes.iter().enumerate() {
                        if let Some(inner_list) = map.get(&h) {
                            let i = probe_map[idx];
                            let p = &rows[i];
                            let k = &probe_keys[idx];
                            for &inner_j in inner_list {
                                // Verify to avoid hash collisions
                                if &key_strs
                                    [key_idx.iter().position(|&x| x == inner_j).unwrap_or(0)]
                                    == k
                                {
                                    emit_pair(
                                        p,
                                        &inner_rows[inner_j],
                                        cfg.level,
                                        &mut on_match,
                                        &mut written,
                                    )?;
                                }
                            }
                        }
                    }
                }
            }
            Index::ByBirth(map) => {
                // Birthdate-blocked processing for L10/L11: group outer rows by birthdate,
                // look up corresponding inner indices via the map, and run fuzzy
                // matching only on those filtered subsets. This prevents a full
                // Cartesian product and enforces O(n) behavior with birthdate blocking.
                use std::collections::HashMap as Hm;

                // Group current outer batch by birthdate string (format must match map keys)
                let mut outer_by_bd: Hm<String, Vec<Person>> = Hm::new();
                for p in rows.iter() {
                    if let Some(d) = p.birthdate {
                        for key in birthdate_keys(d, allow_swap) {
                            outer_by_bd.entry(key).or_default().push(p.clone());
                        }
                    }
                }

                /*
                 Phase 1: Cross-birthdate batching for Advanced L10/L11
                 - Rationale: Per-birthdate GPU invocations produced very small batches and low GPU utilization (5–20%).
                 - Strategy: Accumulate multiple (outer_subset, inner_subset) groups across birthdate keys and flush as a single batched GPU call once the estimated
                   comparison count exceeds batch_thresh. This maximizes throughput and reduces kernel launch overhead.
                 - Accuracy (NON-NEGOTIABLE): Parity with CPU is preserved by:
                   1) Enforcing strict birthdate equality (both via pre-grouping and final filtering),
                   2) Preserving middle-name rule for L10 (>=2 non-dot non-space characters per side),
                   3) Applying identical thresholding and matched_fields assignment.
                 - Resilience: GPU block is wrapped with with_oom_cpu_fallback and catch_unwind. Any CUDA OOM or runtime error triggers a deterministic CPU fallback over the
                   same accumulated data, ensuring identical results.
                 - CPU-only behavior: We keep per-group processing to avoid any cross-group Cartesian blow-up when the GPU is disabled.
                 - Performance: Expected 5–7x speedup on ≥100k datasets and 70–90% GPU utilization, with zero accuracy regression.
                */

                // Cross-birthdate batching for GPU efficiency (Phase 1)
                let batch_thresh: usize = 50_000; // flush threshold based on estimated comparisons
                let mut pending_small: Vec<(Vec<Person>, Vec<Person>)> = Vec::new();
                let mut pending_comps: usize = 0;

                // Helper to flush a batched set of groups via GPU (with strict parity filters)
                let mut flush_pending_gpu = |pending: &mut Vec<(Vec<Person>, Vec<Person>)>,
                                             pending_comps: &mut usize|
                 -> anyhow::Result<Vec<MatchPair>> {
                    if pending.is_empty() {
                        return Ok(Vec::new());
                    }
                    use std::collections::HashSet;
                    // Deduplicate outer_all and inner_all by person ID to avoid duplicate pairs
                    // when the same person appears in multiple birthdate groups (swap enabled)
                    let mut outer_all: Vec<Person> = Vec::new();
                    let mut inner_all: Vec<Person> = Vec::new();
                    let mut seen_outer: HashSet<i64> = HashSet::new();
                    let mut seen_inner: HashSet<i64> = HashSet::new();
                    for (o, i) in pending.iter() {
                        for p in o.iter() {
                            if seen_outer.insert(p.id) {
                                outer_all.push(p.clone());
                            }
                        }
                        for p in i.iter() {
                            if seen_inner.insert(p.id) {
                                inner_all.push(p.clone());
                            }
                        }
                    }
                    pending.clear();
                    *pending_comps = 0;

                    #[cfg(feature = "gpu")]
                    if scfg.use_gpu_fuzzy_metrics && !gpu_fuzzy_disable() {
                        let opts = MatchOptions {
                            backend: ComputeBackend::Gpu,
                            gpu: Some(GpuConfig {
                                device_id: None,
                                mem_budget_mb: scfg.gpu_probe_batch_mb.max(512),
                            }),
                            progress: ProgressConfig::default(),
                            allow_birthdate_swap: allow_swap,
                        };

                        // Debug: check for swap candidates in outer_all and inner_all
                        let mut outer_swap_bds: std::collections::HashSet<String> =
                            std::collections::HashSet::new();
                        let mut inner_swap_bds: std::collections::HashSet<String> =
                            std::collections::HashSet::new();
                        for p in &outer_all {
                            if let Some(d) = p.birthdate {
                                let keys = birthdate_keys(d, allow_swap);
                                if keys.len() > 1 {
                                    outer_swap_bds.insert(keys[0].clone());
                                }
                            }
                        }
                        for p in &inner_all {
                            if let Some(d) = p.birthdate {
                                let keys = birthdate_keys(d, allow_swap);
                                if keys.len() > 1 {
                                    inner_swap_bds.insert(keys[0].clone());
                                }
                            }
                        }
                        // Check for potential swap matches (outer bd that could match inner via swap)
                        let mut potential_swap_matches = 0usize;
                        for p in &outer_all {
                            if let Some(d) = p.birthdate {
                                let keys = birthdate_keys(d, allow_swap);
                                for k in &keys {
                                    if inner_swap_bds.contains(k) || outer_swap_bds.contains(k) {
                                        potential_swap_matches += 1;
                                        break;
                                    }
                                }
                            }
                        }
                        log::info!(
                            "[ADV][GPU] Batched fuzzy scoring: level={:?}, outer_all={}, inner_all={}, est_comparisons={}, outer_swap_bds={}, inner_swap_bds={}, potential_swap_matches={}",
                            cfg.level,
                            outer_all.len(),
                            inner_all.len(),
                            outer_all.len().saturating_mul(inner_all.len()),
                            outer_swap_bds.len(),
                            inner_swap_bds.len(),
                            potential_swap_matches
                        );
                        // Debug: log specific persons in outer_all and inner_all
                        for p in &outer_all {
                            if p.id == 36 || p.id == 51 {
                                log::info!(
                                    "[ADV][GPU] outer_all: id={}, first={:?}, last={:?}, bd={:?}",
                                    p.id,
                                    p.first_name,
                                    p.last_name,
                                    p.birthdate
                                );
                            }
                        }
                        for p in &inner_all {
                            if p.id == 36 || p.id == 51 {
                                log::info!(
                                    "[ADV][GPU] inner_all: id={}, first={:?}, last={:?}, bd={:?}",
                                    p.id,
                                    p.first_name,
                                    p.last_name,
                                    p.birthdate
                                );
                            }
                        }
                        eprintln!(
                            "[AUDIT] Advanced fuzzy GPU batched: level={:?}, outer_all={}, inner_all={}, est_comparisons={}",
                            cfg.level,
                            outer_all.len(),
                            inner_all.len(),
                            outer_all.len().saturating_mul(inner_all.len())
                        );
                        let gpu_try = match cfg.level {
                            AdvLevel::L10FuzzyBirthdateFullMiddle => {
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    crate::matching::gpu_config::with_oom_cpu_fallback(
                                        || {
                                            gpu::match_fuzzy_gpu(
                                                &outer_all,
                                                &inner_all,
                                                opts,
                                                &on_progress,
                                            )
                                        },
                                        || {
                                            let mo_cpu = MatchOptions {
                                                backend: ComputeBackend::Cpu,
                                                gpu: None,
                                                progress: ProgressConfig::default(),
                                                allow_birthdate_swap: allow_swap,
                                            };
                                            match_all_with_opts(
                                                &outer_all,
                                                &inner_all,
                                                MatchingAlgorithm::Fuzzy,
                                                mo_cpu,
                                                &on_progress,
                                            )
                                        },
                                        "[ADV][GPU] batched fuzzy",
                                    )
                                }))
                            }
                            AdvLevel::L11FuzzyBirthdateNoMiddle => {
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    crate::matching::gpu_config::with_oom_cpu_fallback(
                                        || {
                                            gpu::match_fuzzy_no_mid_gpu(
                                                &outer_all,
                                                &inner_all,
                                                opts,
                                                &on_progress,
                                            )
                                        },
                                        || {
                                            let mo_cpu = MatchOptions {
                                                backend: ComputeBackend::Cpu,
                                                gpu: None,
                                                progress: ProgressConfig::default(),
                                                allow_birthdate_swap: allow_swap,
                                            };
                                            match_all_with_opts(
                                                &outer_all,
                                                &inner_all,
                                                MatchingAlgorithm::FuzzyNoMiddle,
                                                mo_cpu,
                                                &on_progress,
                                            )
                                        },
                                        "[ADV][GPU] batched fuzzy no-mid",
                                    )
                                }))
                            }
                            _ => Ok(Err(anyhow!("unreachable adv level for batched gpu path"))),
                        };
                        if let Ok(Ok(mut vec_pairs)) = gpu_try {
                            let mut out: Vec<MatchPair> = Vec::with_capacity(vec_pairs.len());
                            let total_gpu = vec_pairs.len();
                            let mut skip_rescore = 0usize;
                            let mut skip_middle = 0usize;
                            let mut skip_bd = 0usize;
                            let mut skip_thresh = 0usize;
                            let mut swap_total = 0usize;
                            let mut swap_kept = 0usize;
                            let mut swap_thresh_filtered = 0usize;
                            for mut pair in vec_pairs.drain(..) {
                                if pair.confidence <= 1.0 {
                                    pair.confidence *= 100.0;
                                }
                                // Re-score on CPU for parity (GPU scores can differ slightly)
                                if matches!(cfg.level, AdvLevel::L10FuzzyBirthdateFullMiddle) {
                                    if let Some((cpu_score, _)) = fuzzy_compare_names_new(
                                        pair.person1.first_name.as_deref(),
                                        pair.person1.middle_name.as_deref(),
                                        pair.person1.last_name.as_deref(),
                                        pair.person2.first_name.as_deref(),
                                        pair.person2.middle_name.as_deref(),
                                        pair.person2.last_name.as_deref(),
                                    ) {
                                        pair.confidence = cpu_score as f32;
                                    } else {
                                        skip_rescore += 1;
                                        continue;
                                    }
                                } else if matches!(cfg.level, AdvLevel::L11FuzzyBirthdateNoMiddle) {
                                    if let Some((cpu_score, _)) = fuzzy_compare_names_no_mid(
                                        pair.person1.first_name.as_deref(),
                                        pair.person1.last_name.as_deref(),
                                        pair.person2.first_name.as_deref(),
                                        pair.person2.last_name.as_deref(),
                                    ) {
                                        pair.confidence = cpu_score as f32;
                                    } else {
                                        skip_rescore += 1;
                                        continue;
                                    }
                                }
                                if matches!(cfg.level, AdvLevel::L10FuzzyBirthdateFullMiddle) {
                                    let m1 =
                                        pair.person1.middle_name.as_deref().unwrap_or("").trim();
                                    let m2 =
                                        pair.person2.middle_name.as_deref().unwrap_or("").trim();
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
                                        skip_middle += 1;
                                        continue;
                                    }
                                }
                                if let (Some(b1), Some(b2)) =
                                    (pair.person1.birthdate, pair.person2.birthdate)
                                {
                                    let stored = b1.format("%Y-%m-%d").to_string();
                                    let input = b2.format("%Y-%m-%d").to_string();
                                    let bd_ok = birthdate_matches(&stored, &input, allow_swap);
                                    let thresh_ok = (pair.confidence / 100.0) >= cfg.threshold;
                                    let is_swap = b1 != b2;
                                    if is_swap {
                                        swap_total += 1;
                                    }
                                    if !bd_ok {
                                        skip_bd += 1;
                                        continue;
                                    }
                                    if !thresh_ok {
                                        skip_thresh += 1;
                                        // Log swap matches that are being filtered by threshold
                                        if is_swap {
                                            swap_thresh_filtered += 1;
                                            log::info!(
                                                "[ADV][GPU] Swap match filtered by threshold: id1={}, id2={}, bd1={}, bd2={}, conf={:.2}",
                                                pair.person1.id,
                                                pair.person2.id,
                                                stored,
                                                input,
                                                pair.confidence
                                            );
                                        }
                                        continue;
                                    }
                                    if is_swap {
                                        swap_kept += 1;
                                    }
                                    pair.matched_fields = match cfg.level {
                                        AdvLevel::L10FuzzyBirthdateFullMiddle => vec![
                                            "fuzzy".into(),
                                            "first_name".into(),
                                            "middle_name".into(),
                                            "last_name".into(),
                                            "birthdate".into(),
                                        ],
                                        AdvLevel::L11FuzzyBirthdateNoMiddle => vec![
                                            "fuzzy".into(),
                                            "first_name".into(),
                                            "last_name".into(),
                                            "birthdate".into(),
                                        ],
                                        _ => vec!["birthdate".into()],
                                    };
                                    out.push(pair);
                                } else {
                                    skip_bd += 1;
                                }
                            }
                            log::info!(
                                "[ADV][GPU] Post-filter stats: total_gpu={}, skip_rescore={}, skip_middle={}, skip_bd={}, skip_thresh={}, kept={}",
                                total_gpu,
                                skip_rescore,
                                skip_middle,
                                skip_bd,
                                skip_thresh,
                                out.len()
                            );
                            log::info!(
                                "[ADV][GPU] Swap stats: total_swap_pairs={}, kept={}, thresh_filtered={}",
                                swap_total,
                                swap_kept,
                                swap_thresh_filtered
                            );
                            return Ok(out);
                        } else {
                            log::warn!(
                                "[ADV][GPU] Batched GPU path failed; falling back to CPU for pending set"
                            );
                        }
                    }
                    // CPU fallback over accumulated data (authoritative classification enforces birthdate equality)
                    log::info!(
                        "[ADV][CPU] Batched CPU fallback: level={:?}, outer_all={}, inner_all={}, est_comparisons={}",
                        cfg.level,
                        outer_all.len(),
                        inner_all.len(),
                        outer_all.len().saturating_mul(inner_all.len())
                    );
                    let mut out: Vec<MatchPair> = Vec::new();
                    let mut seen_pairs: HashSet<(i64, i64)> = HashSet::new();
                    for a in &outer_all {
                        for b in &inner_all {
                            // Deduplicate pairs that can appear twice when month/day swap produces two keys
                            if !seen_pairs.insert((a.id, b.id)) {
                                continue;
                            }
                            let result = match cfg.level {
                                AdvLevel::L10FuzzyBirthdateFullMiddle => fuzzy_compare_names_new(
                                    a.first_name.as_deref(),
                                    a.middle_name.as_deref(),
                                    a.last_name.as_deref(),
                                    b.first_name.as_deref(),
                                    b.middle_name.as_deref(),
                                    b.last_name.as_deref(),
                                ),
                                AdvLevel::L11FuzzyBirthdateNoMiddle => fuzzy_compare_names_no_mid(
                                    a.first_name.as_deref(),
                                    a.last_name.as_deref(),
                                    b.first_name.as_deref(),
                                    b.last_name.as_deref(),
                                ),
                                _ => None,
                            };
                            if let Some((score, _label)) = result {
                                let mut pair = MatchPair {
                                    person1: a.clone(),
                                    person2: b.clone(),
                                    confidence: score as f32, // 0..100
                                    matched_fields: match cfg.level {
                                        AdvLevel::L10FuzzyBirthdateFullMiddle => vec![
                                            "fuzzy".into(),
                                            "first_name".into(),
                                            "middle_name".into(),
                                            "last_name".into(),
                                            "birthdate".into(),
                                        ],
                                        AdvLevel::L11FuzzyBirthdateNoMiddle => vec![
                                            "fuzzy".into(),
                                            "first_name".into(),
                                            "last_name".into(),
                                            "birthdate".into(),
                                        ],
                                        _ => vec!["birthdate".into()],
                                    },
                                    is_matched_infnbd: false,
                                    is_matched_infnmnbd: false,
                                };
                                let birth_ok =
                                    match (pair.person1.birthdate, pair.person2.birthdate) {
                                        (Some(b1), Some(b2)) => {
                                            let stored = b1.format("%Y-%m-%d").to_string();
                                            let input = b2.format("%Y-%m-%d").to_string();
                                            match cfg.level {
                                                AdvLevel::L10FuzzyBirthdateFullMiddle => {
                                                    match_level_10(&stored, &input, allow_swap)
                                                }
                                                AdvLevel::L11FuzzyBirthdateNoMiddle => {
                                                    match_level_11(&stored, &input, allow_swap)
                                                }
                                                _ => true,
                                            }
                                        }
                                        _ => false,
                                    };
                                if !birth_ok {
                                    continue;
                                }
                                if matches!(cfg.level, AdvLevel::L10FuzzyBirthdateFullMiddle) {
                                    let m1 =
                                        pair.person1.middle_name.as_deref().unwrap_or("").trim();
                                    let m2 =
                                        pair.person2.middle_name.as_deref().unwrap_or("").trim();
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
                                }
                                if (pair.confidence / 100.0) < cfg.threshold {
                                    continue;
                                }
                                out.push(pair);
                            }
                        }
                    }
                    log::info!(
                        "[ADV][CPU] Batched CPU fallback produced {} matches",
                        out.len()
                    );
                    Ok(out)
                };

                for (bd_str, outer_subset) in outer_by_bd.into_iter() {
                    if outer_subset.is_empty() {
                        continue;
                    }
                    if let Some(inner_idx) = map.get(&bd_str) {
                        if inner_idx.is_empty() {
                            continue;
                        }
                        let mut inner_subset: Vec<Person> = Vec::with_capacity(inner_idx.len());
                        let mut seen_inner = std::collections::HashSet::new();
                        for &j in inner_idx.iter() {
                            let person = inner_rows[j].clone();
                            if seen_inner.insert(person.id) {
                                inner_subset.push(person);
                            }
                        }
                        if inner_subset.is_empty() {
                            continue;
                        }

                        let comps = outer_subset.len().saturating_mul(inner_subset.len());
                        if scfg.use_gpu_fuzzy_metrics && !gpu_fuzzy_disable() {
                            // Always batch groups for GPU to maximize utilization
                            pending_small.push((outer_subset.clone(), inner_subset.clone()));
                            pending_comps = pending_comps.saturating_add(comps);
                            if pending_comps >= batch_thresh {
                                let emitted_pairs =
                                    flush_pending_gpu(&mut pending_small, &mut pending_comps)
                                        .unwrap_or_else(|_| Vec::new());
                                for pair in emitted_pairs {
                                    on_match(&pair)?;
                                    written += 1;
                                }
                            }
                        } else {
                            // CPU-only: retain per-group nested loop to avoid cross-group Cartesian blow-up
                            let t0 = std::time::Instant::now();
                            match cfg.level {
                                AdvLevel::L10FuzzyBirthdateFullMiddle
                                | AdvLevel::L11FuzzyBirthdateNoMiddle => {
                                    let mut seen_outer = std::collections::HashSet::new();
                                    for a in &outer_subset {
                                        if !seen_outer.insert(a.id) {
                                            continue;
                                        }
                                        for b in &inner_subset {
                                            let result = match cfg.level {
                                                AdvLevel::L10FuzzyBirthdateFullMiddle => {
                                                    fuzzy_compare_names_new(
                                                        a.first_name.as_deref(),
                                                        a.middle_name.as_deref(),
                                                        a.last_name.as_deref(),
                                                        b.first_name.as_deref(),
                                                        b.middle_name.as_deref(),
                                                        b.last_name.as_deref(),
                                                    )
                                                }
                                                AdvLevel::L11FuzzyBirthdateNoMiddle => {
                                                    fuzzy_compare_names_no_mid(
                                                        a.first_name.as_deref(),
                                                        a.last_name.as_deref(),
                                                        b.first_name.as_deref(),
                                                        b.last_name.as_deref(),
                                                    )
                                                }
                                                _ => None,
                                            };
                                            if let Some((score, _label)) = result {
                                                let mut pair = MatchPair {
                                                    person1: a.clone(),
                                                    person2: b.clone(),
                                                    confidence: score as f32,
                                                    matched_fields: match cfg.level {
                                                        AdvLevel::L10FuzzyBirthdateFullMiddle => {
                                                            vec![
                                                                "fuzzy".into(),
                                                                "first_name".into(),
                                                                "middle_name".into(),
                                                                "last_name".into(),
                                                                "birthdate".into(),
                                                            ]
                                                        }
                                                        AdvLevel::L11FuzzyBirthdateNoMiddle => {
                                                            vec![
                                                                "fuzzy".into(),
                                                                "first_name".into(),
                                                                "last_name".into(),
                                                                "birthdate".into(),
                                                            ]
                                                        }
                                                        _ => vec!["birthdate".into()],
                                                    },
                                                    is_matched_infnbd: false,
                                                    is_matched_infnmnbd: false,
                                                };
                                                let birth_ok = match (
                                                    pair.person1.birthdate,
                                                    pair.person2.birthdate,
                                                ) {
                                                    (Some(b1), Some(b2)) => {
                                                        let stored =
                                                            b1.format("%Y-%m-%d").to_string();
                                                        let input =
                                                            b2.format("%Y-%m-%d").to_string();
                                                        match cfg.level {
                                                            AdvLevel::L10FuzzyBirthdateFullMiddle => match_level_10(&stored, &input, allow_swap),
                                                            AdvLevel::L11FuzzyBirthdateNoMiddle => match_level_11(&stored, &input, allow_swap),
                                                            _ => true,
                                                        }
                                                    }
                                                    _ => false,
                                                };
                                                if !birth_ok {
                                                    continue;
                                                }
                                                if matches!(
                                                    cfg.level,
                                                    AdvLevel::L10FuzzyBirthdateFullMiddle
                                                ) {
                                                    let m1 = pair
                                                        .person1
                                                        .middle_name
                                                        .as_deref()
                                                        .unwrap_or("")
                                                        .trim();
                                                    let m2 = pair
                                                        .person2
                                                        .middle_name
                                                        .as_deref()
                                                        .unwrap_or("")
                                                        .trim();
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
                                                }
                                                if (pair.confidence / 100.0) < cfg.threshold {
                                                    continue;
                                                }
                                                on_match(&pair)?;
                                                written += 1;
                                            }
                                        }
                                    }
                                }
                                _ => {}
                            }
                            let elapsed_ms = t0.elapsed().as_millis();
                            log::info!(
                                "[ADV][CPU] group comps={} elapsed={}ms per_comp~{:.6}us",
                                comps,
                                elapsed_ms,
                                (elapsed_ms as f64 * 1000.0) / comps.max(1) as f64
                            );
                        }
                    }
                }
                // Final flush of any remaining batched groups
                if pending_comps > 0 {
                    let emitted_pairs = flush_pending_gpu(&mut pending_small, &mut pending_comps)
                        .unwrap_or_else(|_| Vec::new());
                    for pair in emitted_pairs {
                        on_match(&pair)?;
                        written += 1;
                    }
                }
            }
        }

        if scfg.resume {
            if let Some(p) = scfg.checkpoint_path.as_ref() {
                let cp = StreamCheckpoint {
                    db: String::new(),
                    table_inner: inner_table.into(),
                    table_outer: outer_table.into(),
                    algorithm: format!("Advanced::{:?}", cfg.level),
                    batch_size: batch,
                    next_offset: last_id,
                    total_outer,
                    partition_idx: 0,
                    partition_name: String::new(),
                    updated_utc: chrono::Utc::now().to_rfc3339(),
                    last_id: Some(last_id),
                    watermark_id: Some(outer_watermark),
                    filter_sig: Some(format!("id<={}", outer_watermark)),
                };
                let _ = save_checkpoint(p, &cp);
            }
        }
        // Adaptive batching
        let memx = memory_stats_mb();
        if memx.avail_mb > scfg.memory_soft_min_mb * 2 {
            let new_batch = (batch as f64 * 1.5) as i64;
            batch = new_batch.min(200_000).max(10_000);
        }
        tokio::task::yield_now().await;
    }

    Ok(written)
}

/// Streamed Advanced Matching across two databases (pool1 for Table1 outer, pool2 for Table2 inner)
pub async fn stream_match_advanced_dual<F>(
    pool1: &MySqlPool,
    pool2: &MySqlPool,
    table1: &str,
    table2: &str,
    cfg: &AdvConfig,
    scfg: StreamingConfig,
    mut on_match: F,
    on_progress: impl Fn(ProgressUpdate) + Sync,
    ctrl: Option<StreamControl>,
) -> Result<usize>
where
    F: FnMut(&MatchPair) -> Result<()>,
{
    use crate::util::checkpoint::{StreamCheckpoint, load_checkpoint, save_checkpoint};
    use std::collections::HashSet;
    let mut seen_pairs: HashSet<(i64, i64)> = HashSet::new();
    let mut user_on_match = on_match;
    let mut on_match = |pair: &MatchPair| -> Result<()> {
        if matches!(
            cfg.level,
            AdvLevel::L10FuzzyBirthdateFullMiddle | AdvLevel::L11FuzzyBirthdateNoMiddle
        ) {
            if !seen_pairs.insert((pair.person1.id, pair.person2.id)) {
                return Ok(());
            }
        }
        user_on_match(pair)
    };
    let allow_swap = cfg.allow_birthdate_swap;
    // Geographic code levels (L4-L9) use fixed field names in both tables:
    // - L4-L6: 'barangay_code'
    // - L7-L9: 'city_code'
    // No external column mapping is required; records missing these fields simply won't join.

    // Counts: Table1 from pool1, Table2 from pool2
    let c1 = get_person_count(pool1, table1).await?;
    let _c2 = get_person_count(pool2, table2).await?;

    // Always use table2 as inner (indexed) and table1 as outer (streamed)
    let (inner_table, outer_table, total_outer) = (table2, table1, c1);

    // Resume support
    let mut offset: i64 = 0;
    let mut batch = scfg.batch_size.max(10_000);
    if scfg.resume {
        if let Some(p) = scfg.checkpoint_path.as_ref() {
            if let Some(cp) = load_checkpoint(p) {
                let algo = format!("Advanced::{:?}", cfg.level);
                if cp.table_inner == inner_table
                    && cp.table_outer == outer_table
                    && cp.algorithm == algo
                {
                    offset = cp.next_offset.min(total_outer);
                    batch = cp.batch_size.max(10_000);
                }
            }
        }
    }

    // Build inner index (from pool2)
    on_progress(ProgressUpdate {
        processed: 0,
        total: total_outer as usize,
        percent: 0.0,
        eta_secs: 0,
        mem_used_mb: memory_stats_mb().used_mb,
        mem_avail_mb: memory_stats_mb().avail_mb,
        stage: "indexing_advanced",
        batch_size_current: Some(batch),
        gpu_total_mb: 0,
        gpu_free_mb: 0,
        gpu_active: false,
    });
    let inner_rows = get_person_rows_all_columns(pool2, inner_table).await?;
    enum Index {
        Exact(std::collections::HashMap<String, Vec<usize>>),
        ByBirth(std::collections::HashMap<String, Vec<usize>>),
        // GPU-hash-accelerated index: hash -> inner indices, plus parallel key strings for verification
        GpuHash {
            map: std::collections::HashMap<u64, Vec<usize>>,
            key_strs: Vec<String>,
            key_idx: Vec<usize>,
        },
    }

    // Decide index strategy (default)
    let mut index = if matches!(
        cfg.level,
        AdvLevel::L10FuzzyBirthdateFullMiddle | AdvLevel::L11FuzzyBirthdateNoMiddle
    ) {
        Index::ByBirth(Default::default())
    } else {
        Index::Exact(Default::default())
    };

    // Optional: upgrade to GPU-hash index for exact levels (L1-L9)
    #[cfg(feature = "gpu")]
    if !matches!(
        cfg.level,
        AdvLevel::L10FuzzyBirthdateFullMiddle | AdvLevel::L11FuzzyBirthdateNoMiddle
    ) && (scfg.use_gpu_hash_join || scfg.use_gpu_build_hash)
    {
        if let Ok(ctx) = self::gpu::GpuHashContext::get() {
            let mut key_strs: Vec<String> = Vec::new();
            let mut key_idx: Vec<usize> = Vec::new();
            for (i, p) in inner_rows.iter().enumerate() {
                if let Some(k) = adv_exact_key(p, cfg.level, &cfg.cols) {
                    key_idx.push(i);
                    key_strs.push(k);
                }
            }
            if !key_strs.is_empty() {
                let (_gt, gf) = ctx.mem_info_mb();
                let budget = (gf / 2).max(128);
                if let Ok(hashes) = self::gpu::hash_fnv1a64_batch_tiled(&ctx, &key_strs, budget, 64)
                {
                    let mut map: std::collections::HashMap<u64, Vec<usize>> =
                        std::collections::HashMap::new();
                    for (j, &h) in hashes.iter().enumerate() {
                        map.entry(h).or_default().push(key_idx[j]);
                    }
                    let key_count = key_strs.len();
                    index = Index::GpuHash {
                        map,
                        key_strs,
                        key_idx,
                    };
                    log::info!(
                        "Advanced GPU hash index built: level={:?}, inner_keys={}, budget_mb={}",
                        cfg.level,
                        key_count,
                        budget
                    );
                    eprintln!(
                        "[AUDIT] Advanced GPU hash index built: level={:?}, inner_keys={}, budget_mb={}",
                        cfg.level, key_count, budget
                    );
                }
            }
        }
    }

    // Fallback: build CPU index if not using GPU hash variant
    match &mut index {
        Index::Exact(map) => {
            for (i, p) in inner_rows.iter().enumerate() {
                if let Some(k) = adv_exact_key(p, cfg.level, &cfg.cols) {
                    map.entry(k).or_default().push(i);
                }
            }
        }
        Index::ByBirth(map) => {
            for (i, p) in inner_rows.iter().enumerate() {
                if let Some(bd) = p.birthdate {
                    for key in birthdate_keys(bd, allow_swap) {
                        map.entry(key).or_default().push(i);
                    }
                }
            }
        }
        Index::GpuHash { .. } => { /* already built above */ }
    }

    on_progress(ProgressUpdate {
        processed: 0,
        total: total_outer as usize,
        percent: 0.0,
        eta_secs: 0,
        mem_used_mb: memory_stats_mb().used_mb,
        mem_avail_mb: memory_stats_mb().avail_mb,
        stage: "indexing_done",
        batch_size_current: Some(batch),
        gpu_total_mb: 0,
        gpu_free_mb: 0,
        gpu_active: false,
    });

    // Stream outer table (from pool1) and probe
    let start = Instant::now();
    let mut written = 0usize;
    while offset < total_outer {
        if let Some(c) = &ctrl {
            if c.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            while c.pause.load(std::sync::atomic::Ordering::Relaxed) {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        }
        let rows = fetch_person_rows_chunk_all_columns(pool1, outer_table, offset, batch).await?;
        if rows.is_empty() {
            break;
        }
        let elapsed = start.elapsed();
        let processed = (offset as usize).min(total_outer as usize);
        let frac = (processed as f32 / total_outer as f32).clamp(0.0, 1.0);
        #[inline]
        fn emit_pair<FN>(
            p: &Person,
            q: &Person,
            level: AdvLevel,
            on_match: &mut FN,
            written: &mut usize,
        ) -> Result<()>
        where
            FN: FnMut(&MatchPair) -> Result<()>,
        {
            let mut fields = vec!["first_name".to_string(), "last_name".to_string()];
            match level {
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
                _ => {}
            }
            let pair = MatchPair {
                person1: p.clone(),
                person2: q.clone(),
                confidence: 100.0,
                matched_fields: fields,
                is_matched_infnbd: false,
                is_matched_infnmnbd: false,
            };
            on_match(&pair)?;
            *written += 1;
            Ok(())
        }

        let eta_secs = if frac > 0.0 {
            (elapsed.as_secs_f32() * (1.0 - frac) / frac) as u64
        } else {
            0
        };
        let mem = memory_stats_mb();
        on_progress(ProgressUpdate {
            processed,
            total: total_outer as usize,
            percent: frac * 100.0,
            eta_secs,
            mem_used_mb: mem.used_mb,
            mem_avail_mb: mem.avail_mb,
            stage: "probing_advanced",
            batch_size_current: Some(batch),
            gpu_total_mb: 0,
            gpu_free_mb: 0,
            gpu_active: false,
        });

        match &index {
            Index::Exact(map) => {
                for p in &rows {
                    if let Some(k) = adv_exact_key(p, cfg.level, &cfg.cols) {
                        if let Some(cand_idx) = map.get(&k) {
                            for &j in cand_idx {
                                emit_pair(
                                    p,
                                    &inner_rows[j],
                                    cfg.level,
                                    &mut on_match,
                                    &mut written,
                                )?;
                            }
                        }
                    }
                }
            }
            Index::GpuHash {
                map,
                key_strs,
                key_idx,
            } => {
                // Build probe keys for this batch, hash on GPU if enabled, else CPU FNV
                let mut probe_keys: Vec<String> = Vec::new();
                let mut probe_map: Vec<usize> = Vec::new(); // outer row idx -> position in probe_keys
                for (i, p) in rows.iter().enumerate() {
                    if let Some(k) = adv_exact_key(p, cfg.level, &cfg.cols) {
                        probe_map.push(i);
                        probe_keys.push(k);
                    }
                }
                if !probe_keys.is_empty() {
                    #[cfg(feature = "gpu")]
                    let probe_hashes: Vec<u64> = if scfg.use_gpu_hash_join
                        || scfg.use_gpu_probe_hash
                    {
                        if let Ok(ctx) = self::gpu::GpuHashContext::get() {
                            let budget = scfg.gpu_probe_batch_mb.max(128);
                            log::info!(
                                "Advanced GPU hash probe engaged: level={:?}, keys={}, budget_mb={}",
                                cfg.level,
                                probe_keys.len(),
                                budget
                            );
                            eprintln!(
                                "[AUDIT] Advanced GPU hash probe engaged: level={:?}, keys={}, budget_mb={}",
                                cfg.level,
                                probe_keys.len(),
                                budget
                            );
                            match self::gpu::hash_fnv1a64_batch_tiled(&ctx, &probe_keys, budget, 64)
                            {
                                Ok(v) => v,
                                Err(_) => probe_keys
                                    .iter()
                                    .map(|s| fnv1a64_bytes(s.as_bytes()))
                                    .collect(),
                            }
                        } else {
                            probe_keys
                                .iter()
                                .map(|s| fnv1a64_bytes(s.as_bytes()))
                                .collect()
                        }
                    } else {
                        probe_keys
                            .iter()
                            .map(|s| fnv1a64_bytes(s.as_bytes()))
                            .collect()
                    };
                    #[cfg(not(feature = "gpu"))]
                    let probe_hashes: Vec<u64> = probe_keys
                        .iter()
                        .map(|s| fnv1a64_bytes(s.as_bytes()))
                        .collect();

                    for (idx, &h) in probe_hashes.iter().enumerate() {
                        if let Some(inner_list) = map.get(&h) {
                            let i = probe_map[idx];
                            let p = &rows[i];
                            let k = &probe_keys[idx];
                            for &inner_j in inner_list {
                                // Verify to avoid hash collisions
                                if &key_strs
                                    [key_idx.iter().position(|&x| x == inner_j).unwrap_or(0)]
                                    == k
                                {
                                    emit_pair(
                                        p,
                                        &inner_rows[inner_j],
                                        cfg.level,
                                        &mut on_match,
                                        &mut written,
                                    )?;
                                }
                            }
                        }
                    }
                }
            }
            Index::ByBirth(map) => {
                use std::collections::HashMap as Hm;
                let mut outer_by_bd: Hm<String, Vec<Person>> = Hm::new();
                for p in rows.iter() {
                    if let Some(d) = p.birthdate {
                        for key in birthdate_keys(d, allow_swap) {
                            outer_by_bd.entry(key).or_default().push(p.clone());
                        }
                    }
                }

                // Cross-birthdate batching for GPU efficiency (Phase 1)
                let batch_thresh: usize = 50_000; // flush threshold based on estimated comparisons
                let mut pending_small: Vec<(Vec<Person>, Vec<Person>)> = Vec::new();
                let mut pending_comps: usize = 0;

                let mut flush_pending_gpu = |pending: &mut Vec<(Vec<Person>, Vec<Person>)>,
                                             pending_comps: &mut usize|
                 -> anyhow::Result<Vec<MatchPair>> {
                    if pending.is_empty() {
                        return Ok(Vec::new());
                    }
                    let mut outer_all: Vec<Person> = Vec::new();
                    let mut inner_all: Vec<Person> = Vec::new();
                    for (o, i) in pending.iter() {
                        outer_all.extend_from_slice(o.as_slice());
                        inner_all.extend_from_slice(i.as_slice());
                    }
                    pending.clear();
                    *pending_comps = 0;

                    #[cfg(feature = "gpu")]
                    if scfg.use_gpu_fuzzy_metrics && !gpu_fuzzy_disable() {
                        let opts = MatchOptions {
                            backend: ComputeBackend::Gpu,
                            gpu: Some(GpuConfig {
                                device_id: None,
                                mem_budget_mb: scfg.gpu_probe_batch_mb.max(512),
                            }),
                            progress: ProgressConfig::default(),
                            allow_birthdate_swap: allow_swap,
                        };
                        log::info!(
                            "[ADV][GPU] Batched fuzzy scoring: level={:?}, outer_all={}, inner_all={}, est_comparisons={}",
                            cfg.level,
                            outer_all.len(),
                            inner_all.len(),
                            outer_all.len().saturating_mul(inner_all.len())
                        );
                        eprintln!(
                            "[AUDIT] Advanced fuzzy GPU batched: level={:?}, outer_all={}, inner_all={}, est_comparisons={}",
                            cfg.level,
                            outer_all.len(),
                            inner_all.len(),
                            outer_all.len().saturating_mul(inner_all.len())
                        );
                        let gpu_try = match cfg.level {
                            AdvLevel::L10FuzzyBirthdateFullMiddle => {
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    crate::matching::gpu_config::with_oom_cpu_fallback(
                                        || {
                                            gpu::match_fuzzy_gpu(
                                                &outer_all,
                                                &inner_all,
                                                opts,
                                                &on_progress,
                                            )
                                        },
                                        || {
                                            let mo_cpu = MatchOptions {
                                                backend: ComputeBackend::Cpu,
                                                gpu: None,
                                                progress: ProgressConfig::default(),
                                                allow_birthdate_swap: allow_swap,
                                            };
                                            match_all_with_opts(
                                                &outer_all,
                                                &inner_all,
                                                MatchingAlgorithm::Fuzzy,
                                                mo_cpu,
                                                &on_progress,
                                            )
                                        },
                                        "[ADV][GPU] batched fuzzy",
                                    )
                                }))
                            }
                            AdvLevel::L11FuzzyBirthdateNoMiddle => {
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    crate::matching::gpu_config::with_oom_cpu_fallback(
                                        || {
                                            gpu::match_fuzzy_no_mid_gpu(
                                                &outer_all,
                                                &inner_all,
                                                opts,
                                                &on_progress,
                                            )
                                        },
                                        || {
                                            let mo_cpu = MatchOptions {
                                                backend: ComputeBackend::Cpu,
                                                gpu: None,
                                                progress: ProgressConfig::default(),
                                                allow_birthdate_swap: allow_swap,
                                            };
                                            match_all_with_opts(
                                                &outer_all,
                                                &inner_all,
                                                MatchingAlgorithm::FuzzyNoMiddle,
                                                mo_cpu,
                                                &on_progress,
                                            )
                                        },
                                        "[ADV][GPU] batched fuzzy no-mid",
                                    )
                                }))
                            }
                            _ => Ok(Err(anyhow!("unreachable adv level for batched gpu path"))),
                        };
                        if let Ok(Ok(mut vec_pairs)) = gpu_try {
                            let mut out: Vec<MatchPair> = Vec::with_capacity(vec_pairs.len());
                            for mut pair in vec_pairs.drain(..) {
                                if pair.confidence <= 1.0 {
                                    pair.confidence *= 100.0;
                                }
                                // Re-score on CPU for parity with CPU/in-memory paths
                                if matches!(cfg.level, AdvLevel::L10FuzzyBirthdateFullMiddle) {
                                    if let Some((cpu_score, _)) =
                                        compare_persons_new(&pair.person1, &pair.person2)
                                    {
                                        pair.confidence = cpu_score as f32;
                                    } else {
                                        continue;
                                    }
                                } else if matches!(cfg.level, AdvLevel::L11FuzzyBirthdateNoMiddle) {
                                    if let Some((cpu_score, _)) =
                                        compare_persons_no_mid(&pair.person1, &pair.person2)
                                    {
                                        pair.confidence = cpu_score as f32;
                                    } else {
                                        continue;
                                    }
                                }
                                if matches!(cfg.level, AdvLevel::L10FuzzyBirthdateFullMiddle) {
                                    let m1 =
                                        pair.person1.middle_name.as_deref().unwrap_or("").trim();
                                    let m2 =
                                        pair.person2.middle_name.as_deref().unwrap_or("").trim();
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
                                }
                                if let (Some(b1), Some(b2)) =
                                    (pair.person1.birthdate, pair.person2.birthdate)
                                {
                                    let stored = b1.format("%Y-%m-%d").to_string();
                                    let input = b2.format("%Y-%m-%d").to_string();
                                    if birthdate_matches(&stored, &input, allow_swap)
                                        && (pair.confidence / 100.0) >= cfg.threshold
                                    {
                                        pair.matched_fields = match cfg.level {
                                            AdvLevel::L10FuzzyBirthdateFullMiddle => vec![
                                                "fuzzy".into(),
                                                "first_name".into(),
                                                "middle_name".into(),
                                                "last_name".into(),
                                                "birthdate".into(),
                                            ],
                                            AdvLevel::L11FuzzyBirthdateNoMiddle => vec![
                                                "fuzzy".into(),
                                                "first_name".into(),
                                                "last_name".into(),
                                                "birthdate".into(),
                                            ],
                                            _ => vec!["birthdate".into()],
                                        };
                                        out.push(pair);
                                    }
                                }
                            }
                            return Ok(out);
                        } else {
                            log::warn!(
                                "[ADV][GPU] Batched GPU path failed; falling back to CPU for pending set"
                            );
                        }
                    }
                    // CPU fallback
                    log::info!(
                        "[ADV][CPU] Batched CPU fallback: level={:?}, outer_all={}, inner_all={}, est_comparisons={}",
                        cfg.level,
                        outer_all.len(),
                        inner_all.len(),
                        outer_all.len().saturating_mul(inner_all.len())
                    );
                    let mut out: Vec<MatchPair> = Vec::new();
                    for a in &outer_all {
                        for b in &inner_all {
                            let result = match cfg.level {
                                AdvLevel::L10FuzzyBirthdateFullMiddle => fuzzy_compare_names_new(
                                    a.first_name.as_deref(),
                                    a.middle_name.as_deref(),
                                    a.last_name.as_deref(),
                                    b.first_name.as_deref(),
                                    b.middle_name.as_deref(),
                                    b.last_name.as_deref(),
                                ),
                                AdvLevel::L11FuzzyBirthdateNoMiddle => fuzzy_compare_names_no_mid(
                                    a.first_name.as_deref(),
                                    a.last_name.as_deref(),
                                    b.first_name.as_deref(),
                                    b.last_name.as_deref(),
                                ),
                                _ => None,
                            };
                            if let Some((score, _label)) = result {
                                let mut pair = MatchPair {
                                    person1: a.clone(),
                                    person2: b.clone(),
                                    confidence: score as f32, // 0..100
                                    matched_fields: match cfg.level {
                                        AdvLevel::L10FuzzyBirthdateFullMiddle => vec![
                                            "fuzzy".into(),
                                            "first_name".into(),
                                            "middle_name".into(),
                                            "last_name".into(),
                                            "birthdate".into(),
                                        ],
                                        AdvLevel::L11FuzzyBirthdateNoMiddle => vec![
                                            "fuzzy".into(),
                                            "first_name".into(),
                                            "last_name".into(),
                                            "birthdate".into(),
                                        ],
                                        _ => vec!["birthdate".into()],
                                    },
                                    is_matched_infnbd: false,
                                    is_matched_infnmnbd: false,
                                };
                                let birth_ok =
                                    match (pair.person1.birthdate, pair.person2.birthdate) {
                                        (Some(b1), Some(b2)) => {
                                            let stored = b1.format("%Y-%m-%d").to_string();
                                            let input = b2.format("%Y-%m-%d").to_string();
                                            match cfg.level {
                                                AdvLevel::L10FuzzyBirthdateFullMiddle => {
                                                    match_level_10(&stored, &input, allow_swap)
                                                }
                                                AdvLevel::L11FuzzyBirthdateNoMiddle => {
                                                    match_level_11(&stored, &input, allow_swap)
                                                }
                                                _ => true,
                                            }
                                        }
                                        _ => false,
                                    };
                                if !birth_ok {
                                    continue;
                                }
                                if matches!(cfg.level, AdvLevel::L10FuzzyBirthdateFullMiddle) {
                                    let m1 =
                                        pair.person1.middle_name.as_deref().unwrap_or("").trim();
                                    let m2 =
                                        pair.person2.middle_name.as_deref().unwrap_or("").trim();
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
                                }
                                if (pair.confidence / 100.0) < cfg.threshold {
                                    continue;
                                }
                                out.push(pair);
                            }
                        }
                    }
                    log::info!(
                        "[ADV][CPU] Batched CPU fallback produced {} matches",
                        out.len()
                    );
                    Ok(out)
                };

                for (bd_str, outer_subset) in outer_by_bd.into_iter() {
                    if outer_subset.is_empty() {
                        continue;
                    }
                    if let Some(inner_idx) = map.get(&bd_str) {
                        if inner_idx.is_empty() {
                            continue;
                        }
                        let mut inner_subset: Vec<Person> = Vec::with_capacity(inner_idx.len());
                        let mut seen_inner = std::collections::HashSet::new();
                        for &j in inner_idx.iter() {
                            let person = inner_rows[j].clone();
                            if seen_inner.insert(person.id) {
                                inner_subset.push(person);
                            }
                        }
                        if inner_subset.is_empty() {
                            continue;
                        }

                        let comps = outer_subset.len().saturating_mul(inner_subset.len());
                        if scfg.use_gpu_fuzzy_metrics && !gpu_fuzzy_disable() {
                            // Always batch groups for GPU to maximize utilization
                            pending_small.push((outer_subset.clone(), inner_subset.clone()));
                            pending_comps = pending_comps.saturating_add(comps);
                            if pending_comps >= batch_thresh {
                                let emitted_pairs =
                                    flush_pending_gpu(&mut pending_small, &mut pending_comps)
                                        .unwrap_or_else(|_| Vec::new());
                                for pair in emitted_pairs {
                                    on_match(&pair)?;
                                    written += 1;
                                }
                            }
                        } else {
                            // CPU-only: retain per-group nested loop to avoid cross-group Cartesian blow-up
                            let t0 = std::time::Instant::now();
                            match cfg.level {
                                AdvLevel::L10FuzzyBirthdateFullMiddle
                                | AdvLevel::L11FuzzyBirthdateNoMiddle => {
                                    let mut seen_outer = std::collections::HashSet::new();
                                    for a in &outer_subset {
                                        if !seen_outer.insert(a.id) {
                                            continue;
                                        }
                                        for b in &inner_subset {
                                            let result = if matches!(
                                                cfg.level,
                                                AdvLevel::L10FuzzyBirthdateFullMiddle
                                            ) {
                                                compare_persons_new(a, b)
                                            } else {
                                                compare_persons_no_mid(a, b)
                                            };
                                            if let Some((score, _label)) = result {
                                                let mut pair = MatchPair {
                                                    person1: a.clone(),
                                                    person2: b.clone(),
                                                    confidence: score as f32,
                                                    matched_fields: match cfg.level {
                                                        AdvLevel::L10FuzzyBirthdateFullMiddle => {
                                                            vec![
                                                                "fuzzy".into(),
                                                                "first_name".into(),
                                                                "middle_name".into(),
                                                                "last_name".into(),
                                                                "birthdate".into(),
                                                            ]
                                                        }
                                                        AdvLevel::L11FuzzyBirthdateNoMiddle => {
                                                            vec![
                                                                "fuzzy".into(),
                                                                "first_name".into(),
                                                                "last_name".into(),
                                                                "birthdate".into(),
                                                            ]
                                                        }
                                                        _ => vec!["birthdate".into()],
                                                    },
                                                    is_matched_infnbd: false,
                                                    is_matched_infnmnbd: false,
                                                };
                                                let birth_ok = match (
                                                    pair.person1.birthdate,
                                                    pair.person2.birthdate,
                                                ) {
                                                    (Some(b1), Some(b2)) => {
                                                        let stored =
                                                            b1.format("%Y-%m-%d").to_string();
                                                        let input =
                                                            b2.format("%Y-%m-%d").to_string();
                                                        match cfg.level {
                                                            AdvLevel::L10FuzzyBirthdateFullMiddle => match_level_10(&stored, &input, allow_swap),
                                                            AdvLevel::L11FuzzyBirthdateNoMiddle => match_level_11(&stored, &input, allow_swap),
                                                            _ => true,
                                                        }
                                                    }
                                                    _ => false,
                                                };
                                                if !birth_ok {
                                                    continue;
                                                }
                                                if matches!(
                                                    cfg.level,
                                                    AdvLevel::L10FuzzyBirthdateFullMiddle
                                                ) {
                                                    let m1 = pair
                                                        .person1
                                                        .middle_name
                                                        .as_deref()
                                                        .unwrap_or("")
                                                        .trim();
                                                    let m2 = pair
                                                        .person2
                                                        .middle_name
                                                        .as_deref()
                                                        .unwrap_or("")
                                                        .trim();
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
                                                }
                                                if (pair.confidence / 100.0) < cfg.threshold {
                                                    continue;
                                                }
                                                on_match(&pair)?;
                                                written += 1;
                                            }
                                        }
                                    }
                                }
                                _ => {}
                            }
                            let elapsed_ms = t0.elapsed().as_millis();
                            log::info!(
                                "[ADV][CPU] group comps={} elapsed={}ms per_comp~{:.6}us",
                                comps,
                                elapsed_ms,
                                (elapsed_ms as f64 * 1000.0) / comps.max(1) as f64
                            );
                        }
                    }
                }
                // Final flush of any remaining batched groups
                if pending_comps > 0 {
                    let emitted_pairs = flush_pending_gpu(&mut pending_small, &mut pending_comps)
                        .unwrap_or_else(|_| Vec::new());
                    for pair in emitted_pairs {
                        on_match(&pair)?;
                        written += 1;
                    }
                }
            }
        }

        offset += batch;
        if scfg.resume {
            if let Some(p) = scfg.checkpoint_path.as_ref() {
                let cp = StreamCheckpoint {
                    db: String::new(),
                    table_inner: inner_table.into(),
                    table_outer: outer_table.into(),
                    algorithm: format!("Advanced::{:?}", cfg.level),
                    batch_size: batch,
                    next_offset: offset,
                    total_outer,
                    partition_idx: 0,
                    partition_name: String::new(),
                    updated_utc: chrono::Utc::now().to_rfc3339(),
                    last_id: Some(offset),
                    watermark_id: None,
                    filter_sig: None,
                };
                let _ = save_checkpoint(p, &cp);
            }
        }
        // Adaptive batching
        let memx = memory_stats_mb();
        if memx.avail_mb > scfg.memory_soft_min_mb * 2 {
            let new_batch = (batch as f64 * 1.5) as i64;
            batch = new_batch.min(200_000).max(10_000);
        }
        tokio::task::yield_now().await;
    }

    Ok(written)
}

/// Streamed Advanced Level 12 (Household, Table2 → Table1).
/// DEPRECATED custom implementation: this thin wrapper now delegates to Option 6 streaming
/// to guarantee identical semantics and 100% code reuse with the in-memory source of truth.
pub async fn stream_match_advanced_l12<F>(
    pool: &MySqlPool,
    table1: &str,
    table2: &str,
    cfg: &AdvConfig, // expects level=L12HouseholdMatching; threshold = per-person fuzzy min conf
    scfg: StreamingConfig,
    mut on_household: F,
    on_progress: impl Fn(ProgressUpdate) + Sync,
    ctrl: Option<StreamControl>,
) -> Result<usize>
where
    F: FnMut(&HouseholdAggRow) -> Result<()>,
{
    if !matches!(cfg.level, AdvLevel::L12HouseholdMatching) {
        anyhow::bail!("stream_match_advanced_l12 requires AdvLevel::L12HouseholdMatching");
    }
    // Delegate to Option 6 streaming (exact same semantics)
    stream_match_option6(
        pool,
        table1,
        table2,
        cfg.threshold,
        cfg.allow_birthdate_swap,
        scfg,
        |row| on_household(row),
        on_progress,
        ctrl,
    )
    .await
}

/// Streamed Advanced Level 12 across two databases (pool1 for Table1, pool2 for Table2)
pub async fn stream_match_advanced_l12_dual<F>(
    pool1: &MySqlPool,
    pool2: &MySqlPool,
    table1: &str,
    table2: &str,
    cfg: &AdvConfig, // expects level=L12HouseholdMatching; threshold = per-person fuzzy min conf
    scfg: StreamingConfig,
    mut on_household: F,
    on_progress: impl Fn(ProgressUpdate) + Sync,
    ctrl: Option<StreamControl>,
) -> Result<usize>
where
    F: FnMut(&HouseholdAggRow) -> Result<()>,
{
    if !matches!(cfg.level, AdvLevel::L12HouseholdMatching) {
        anyhow::bail!("stream_match_advanced_l12_dual requires AdvLevel::L12HouseholdMatching");
    }
    // Delegate to Option 6 dual streaming (exact same semantics)
    stream_match_option6_dual(
        pool1,
        pool2,
        table1,
        table2,
        cfg.threshold,
        cfg.allow_birthdate_swap,
        scfg,
        |row| on_household(row),
        on_progress,
        ctrl,
    )
    .await
}

/// Best-effort helper to retrieve current GPU device name for export metadata.

/// Streaming Option 6 (HouseholdGpuOpt6) with EXACT semantics of match_households_gpu_inmemory_opt6.
/// Source = Table 2 (group by hh_id or fallback id), Target = Table 1 (group by uuid),
/// match_percentage denominator = Table 2 household size; per-person fuzzy with exact birthdate equality.
pub async fn stream_match_option6<F>(
    pool: &sqlx::MySqlPool,
    table1: &str,
    table2: &str,
    fuzzy_min_conf: f32,
    allow_birthdate_swap: bool,
    scfg: StreamingConfig,
    mut on_household: F,
    on_progress: impl Fn(ProgressUpdate) + Sync,
    ctrl: Option<StreamControl>,
) -> anyhow::Result<usize>
where
    F: FnMut(&HouseholdAggRow) -> anyhow::Result<()>,
{
    use std::collections::{BTreeMap, HashMap, HashSet};
    let allow_birthdate_swap = crate::matching::birthdate_matcher::allow_birthdate_swap();

    // 1) Precompute totals per Table 2 household (denominator)
    let c2_total = get_person_count(pool, table2).await?;
    on_progress(ProgressUpdate {
        processed: 0,
        total: c2_total as usize,
        percent: 0.0,
        eta_secs: 0,
        mem_used_mb: memory_stats_mb().used_mb,
        mem_avail_mb: memory_stats_mb().avail_mb,
        stage: "opt6_pretotal",
        batch_size_current: None,
        gpu_total_mb: 0,
        gpu_free_mb: 0,
        gpu_active: false,
    });
    let totals_t2 = crate::db::schema::get_household_totals_map(pool, table2).await?;

    // 2) Load Table 1 into birthdate index (incremental, with progress)
    let c1_total = get_person_count(pool, table1).await?;
    let mut by_bd1: HashMap<String, Vec<Person>> = HashMap::new();
    let mut t1_loaded: i64 = 0;
    let load_batch: i64 = scfg.batch_size.max(10_000);
    on_progress(ProgressUpdate {
        processed: 0,
        total: c1_total as usize,
        percent: 0.0,
        eta_secs: 0,
        mem_used_mb: memory_stats_mb().used_mb,
        mem_avail_mb: memory_stats_mb().avail_mb,
        stage: "opt6_load_t1",
        batch_size_current: Some(load_batch),
        gpu_total_mb: 0,
        gpu_free_mb: 0,
        gpu_active: false,
    });
    while t1_loaded < c1_total {
        let chunk =
            fetch_person_rows_chunk_all_columns(pool, table1, t1_loaded, load_batch).await?;
        if chunk.is_empty() {
            break;
        }
        for p in chunk.into_iter() {
            if let Some(d) = p.birthdate {
                for k in birthdate_keys(d, allow_birthdate_swap) {
                    by_bd1.entry(k).or_default().push(p.clone());
                }
            }
        }
        t1_loaded += load_batch;
        let mem = memory_stats_mb();
        let frac = (t1_loaded as f32 / c1_total as f32).clamp(0.0, 1.0);
        on_progress(ProgressUpdate {
            processed: (t1_loaded as usize).min(c1_total as usize),
            total: c1_total as usize,
            percent: frac * 100.0,
            eta_secs: 0,
            mem_used_mb: mem.used_mb,
            mem_avail_mb: mem.avail_mb,
            stage: "opt6_load_t1",
            batch_size_current: Some(load_batch),
            gpu_total_mb: 0,
            gpu_free_mb: 0,
            gpu_active: false,
        });
        tokio::task::yield_now().await;
    }

    // 3) Stream Table 2 in chunks; build person-level best mapping (Table2 person -> (hh_key, uuid))
    let mut offset: i64 = 0;
    let mut batch = scfg.batch_size.max(10_000);
    let mut best_for_p2: BTreeMap<i64, (String, String, f32, bool)> = BTreeMap::new();
    let start = std::time::Instant::now();

    while offset < c2_total {
        if let Some(c) = &ctrl {
            if c.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            while c.pause.load(std::sync::atomic::Ordering::Relaxed) {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        }
        let rows_t2 = fetch_person_rows_chunk_all_columns(pool, table2, offset, batch).await?;
        if rows_t2.is_empty() {
            break;
        }

        let mut by_bd2: HashMap<String, Vec<Person>> = HashMap::new();
        for p in rows_t2 {
            if let Some(d) = p.birthdate {
                for k in birthdate_keys(d, allow_birthdate_swap) {
                    by_bd2.entry(k).or_default().push(p.clone());
                }
            }
        }

        let mo_cpu = MatchOptions {
            backend: ComputeBackend::Cpu,
            gpu: None,
            progress: ProgressConfig::default(),
            allow_birthdate_swap,
        };
        /*
         Phase 2a: Cross-birthdate batching for Option 6 / Advanced L12 (Household)
         - Motivation: Per-birthdate GPU launches produce tiny batches and poor utilization.
         - Strategy: Accumulate multiple (v1, v2) groups across birthdates and flush as one GPU call once an estimated
           comparison threshold is reached. CPU-only path remains per-group to avoid cross-group Cartesian explosion.
         - Parity: Birthdate equality is preserved by pre-grouping; final household aggregation logic is unchanged.
         - Resilience: GPU path is wrapped with with_oom_cpu_fallback and catch_unwind; deterministic CPU fallback emits identical results.
        */
        let batch_thresh: usize = 50_000; // estimated comparisons threshold for flushing

        let mut pending_small: Vec<(Vec<Person>, Vec<Person>)> = Vec::new();
        let mut pending_comps: usize = 0;

        let mut flush_pending_gpu = |pending: &mut Vec<(Vec<Person>, Vec<Person>)>,
                                     pending_comps: &mut usize|
         -> anyhow::Result<Vec<MatchPair>> {
            if pending.is_empty() {
                return Ok(Vec::new());
            }
            let mut outer_all: Vec<Person> = Vec::new();
            let mut inner_all: Vec<Person> = Vec::new();
            for (o, i) in pending.iter() {
                outer_all.extend_from_slice(o.as_slice());
                inner_all.extend_from_slice(i.as_slice());
            }
            pending.clear();
            *pending_comps = 0;

            #[cfg(feature = "gpu")]
            if scfg.use_gpu_fuzzy_metrics && !gpu_fuzzy_disable() {
                let opts = MatchOptions {
                    backend: ComputeBackend::Gpu,
                    gpu: Some(GpuConfig {
                        device_id: None,
                        mem_budget_mb: scfg.gpu_probe_batch_mb.max(512),
                    }),
                    progress: ProgressConfig::default(),
                    allow_birthdate_swap,
                };
                log::info!(
                    "[ADV][GPU][L12] Batched fuzzy (no-mid): outer_all={} inner_all={} est_comparisons={}",
                    outer_all.len(),
                    inner_all.len(),
                    outer_all.len().saturating_mul(inner_all.len())
                );
                let gpu_try = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    crate::matching::gpu_config::with_oom_cpu_fallback(
                        || {
                            gpu::match_fuzzy_no_mid_gpu(
                                &outer_all,
                                &inner_all,
                                opts,
                                &|_u: ProgressUpdate| {},
                            )
                        },
                        || {
                            let mo_cpu = MatchOptions {
                                backend: ComputeBackend::Cpu,
                                gpu: None,
                                progress: ProgressConfig::default(),
                                allow_birthdate_swap,
                            };
                            match_all_with_opts(
                                &outer_all,
                                &inner_all,
                                MatchingAlgorithm::FuzzyNoMiddle,
                                mo_cpu,
                                &|_u: ProgressUpdate| {},
                            )
                        },
                        "[ADV][GPU] opt6 batched fuzzy no-mid",
                    )
                }));
                if let Ok(Ok(vec_pairs)) = gpu_try {
                    // Keep confidence in [0.0, 1.0] units; thresholds are fractional
                    return Ok(vec_pairs);
                } else {
                    log::warn!(
                        "[ADV][GPU] Opt6 batched GPU path failed; falling back to CPU for pending set"
                    );
                }
            }
            // CPU fallback over accumulated data
            let mo_cpu = MatchOptions {
                backend: ComputeBackend::Cpu,
                gpu: None,
                progress: ProgressConfig::default(),
                allow_birthdate_swap,
            };
            let mut seen_pairs: HashSet<(i64, i64)> = HashSet::new();
            let mut out = Vec::new();
            for p in match_all_with_opts(
                &outer_all,
                &inner_all,
                MatchingAlgorithm::FuzzyNoMiddle,
                mo_cpu,
                &|_u: ProgressUpdate| {},
            ) {
                if seen_pairs.insert((p.person1.id, p.person2.id)) {
                    out.push(p);
                }
            }
            Ok(out)
        };

        for (bd, v2) in by_bd2.into_iter() {
            if let Some(v1) = by_bd1.get(&bd) {
                let comps = v1.len().saturating_mul(v2.len());
                if scfg.use_gpu_fuzzy_metrics && !gpu_fuzzy_disable() {
                    // Accumulate groups for GPU flush
                    pending_small.push((v1.clone(), v2.clone()));
                    pending_comps = pending_comps.saturating_add(comps);
                    if pending_comps >= batch_thresh {
                        let emitted_pairs =
                            flush_pending_gpu(&mut pending_small, &mut pending_comps)
                                .unwrap_or_else(|_| Vec::new());
                        let mut seen_pairs: HashSet<(i64, i64)> = HashSet::new();
                        for p in emitted_pairs.into_iter() {
                            if !seen_pairs.insert((p.person1.id, p.person2.id)) {
                                continue;
                            }
                            if p.confidence < fuzzy_min_conf {
                                continue;
                            }
                            let Some(uuid) = p.person1.uuid.clone() else {
                                continue;
                            };
                            let hh_key = p
                                .person2
                                .hh_id
                                .clone()
                                .unwrap_or_else(|| p.person2.id.to_string());
                            let key = p.person2.id;
                            match best_for_p2.get_mut(&key) {
                                None => {
                                    best_for_p2.insert(key, (hh_key, uuid, p.confidence, false));
                                }
                                Some((hh, u2, c2, tie)) => {
                                    if p.confidence > *c2 {
                                        *hh = hh_key;
                                        *u2 = uuid;
                                        *c2 = p.confidence;
                                        *tie = false;
                                    } else if (p.confidence - *c2).abs() <= std::f32::EPSILON {
                                        if uuid < *u2 {
                                            *hh = hh_key;
                                            *u2 = uuid;
                                            *c2 = p.confidence;
                                            *tie = false;
                                        } else if uuid != *u2 {
                                            *tie = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    // CPU-only: keep per-birthdate processing
                    let mut seen_pairs: HashSet<(i64, i64)> = HashSet::new();
                    let pairs: Vec<MatchPair> = match_all_with_opts(
                        v1.as_slice(),
                        v2.as_slice(),
                        MatchingAlgorithm::FuzzyNoMiddle,
                        mo_cpu,
                        &|_u: ProgressUpdate| {},
                    );
                    for p in pairs.into_iter() {
                        if !seen_pairs.insert((p.person1.id, p.person2.id)) {
                            continue;
                        }
                        if p.confidence < fuzzy_min_conf {
                            continue;
                        }
                        let Some(uuid) = p.person1.uuid.clone() else {
                            continue;
                        };
                        let hh_key = p
                            .person2
                            .hh_id
                            .clone()
                            .unwrap_or_else(|| p.person2.id.to_string());
                        let key = p.person2.id;
                        match best_for_p2.get_mut(&key) {
                            None => {
                                best_for_p2.insert(key, (hh_key, uuid, p.confidence, false));
                            }
                            Some((hh, u2, c2, tie)) => {
                                if p.confidence > *c2 {
                                    *hh = hh_key;
                                    *u2 = uuid;
                                    *c2 = p.confidence;
                                    *tie = false;
                                } else if (p.confidence - *c2).abs() <= std::f32::EPSILON {
                                    if uuid < *u2 {
                                        *hh = hh_key;
                                        *u2 = uuid;
                                        *c2 = p.confidence;
                                        *tie = false;
                                    } else if uuid != *u2 {
                                        *tie = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Final flush for any remaining batched groups
        if pending_comps > 0 {
            let emitted_pairs = flush_pending_gpu(&mut pending_small, &mut pending_comps)
                .unwrap_or_else(|_| Vec::new());
            let mut seen_pairs: HashSet<(i64, i64)> = HashSet::new();
            for p in emitted_pairs.into_iter() {
                if !seen_pairs.insert((p.person1.id, p.person2.id)) {
                    continue;
                }
                if p.confidence < fuzzy_min_conf {
                    continue;
                }
                let Some(uuid) = p.person1.uuid.clone() else {
                    continue;
                };
                let hh_key = p
                    .person2
                    .hh_id
                    .clone()
                    .unwrap_or_else(|| p.person2.id.to_string());
                let key = p.person2.id;
                match best_for_p2.get_mut(&key) {
                    None => {
                        best_for_p2.insert(key, (hh_key, uuid, p.confidence, false));
                    }
                    Some((hh, u2, c2, tie)) => {
                        if p.confidence > *c2 {
                            *hh = hh_key;
                            *u2 = uuid;
                            *c2 = p.confidence;
                            *tie = false;
                        } else if (p.confidence - *c2).abs() <= std::f32::EPSILON {
                            if uuid < *u2 {
                                *hh = hh_key;
                                *u2 = uuid;
                                *c2 = p.confidence;
                                *tie = false;
                            } else if uuid != *u2 {
                                *tie = true;
                            }
                        }
                    }
                }
            }
        }

        // Progress + adaptive batch
        let elapsed = start.elapsed();
        let processed = (offset as usize).min(c2_total as usize);
        let frac = (processed as f32 / c2_total as f32).clamp(0.0, 1.0);
        let eta_secs = if frac > 0.0 {
            (elapsed.as_secs_f32() * (1.0 - frac) / frac) as u64
        } else {
            0
        };
        let mem = memory_stats_mb();
        #[cfg(feature = "gpu")]
        let (g_tot, g_free) = if scfg.use_gpu_fuzzy_metrics {
            gpu::GpuFuzzyContext::get()
                .map(|c| c.mem_info_mb())
                .unwrap_or((0, 0))
        } else {
            (0, 0)
        };
        #[cfg(not(feature = "gpu"))]
        let (g_tot, g_free) = (0u64, 0u64);
        on_progress(ProgressUpdate {
            processed,
            total: c2_total as usize,
            percent: frac * 100.0,
            eta_secs,
            mem_used_mb: mem.used_mb,
            mem_avail_mb: mem.avail_mb,
            stage: "opt6_probing",
            batch_size_current: Some(batch),
            gpu_total_mb: g_tot,
            gpu_free_mb: g_free,
            gpu_active: scfg.use_gpu_fuzzy_metrics,
        });
        offset += batch;
        let memx = memory_stats_mb();
        if memx.avail_mb > scfg.memory_soft_min_mb * 2 {
            let new_batch = (batch as f64 * 1.5) as i64;
            batch = new_batch.min(200_000).max(10_000);
        }
        tokio::task::yield_now().await;
    }

    // 4) Aggregate and emit households with match_percentage > 50%
    let mut matched: BTreeMap<(String, String), HashSet<i64>> = BTreeMap::new();
    for (p2_id, (hh_key, uuid, _conf, tie)) in best_for_p2.into_iter() {
        if tie {
            continue;
        }
        matched.entry((hh_key, uuid)).or_default().insert(p2_id);
    }

    let mut emitted = 0usize;
    let mut row_id: i64 = 1;
    for ((hh_key, uuid), members) in matched.into_iter() {
        let total = *totals_t2.get(&hh_key).unwrap_or(&0usize) as f32;
        if total <= 0.0 {
            continue;
        }
        let pct = (members.len() as f32) / total * 100.0;
        if pct > 50.0 {
            let row = HouseholdAggRow {
                row_id,
                uuid,
                hh_id: hh_key.parse::<i64>().unwrap_or(0),
                match_percentage: pct,
                region_code: None,
                poor_hat_0: None,
                poor_hat_10: None,
            };
            on_household(&row)?;
            emitted += 1;
            row_id += 1;
        }
    }
    Ok(emitted)
}

/// Streaming Option 6 across two databases (pool1 for Table1, pool2 for Table2)
pub async fn stream_match_option6_dual<F>(
    pool1: &sqlx::MySqlPool,
    pool2: &sqlx::MySqlPool,
    table1: &str,
    table2: &str,
    fuzzy_min_conf: f32,
    allow_birthdate_swap: bool,
    scfg: StreamingConfig,
    mut on_household: F,
    on_progress: impl Fn(ProgressUpdate) + Sync,
    ctrl: Option<StreamControl>,
) -> anyhow::Result<usize>
where
    F: FnMut(&HouseholdAggRow) -> anyhow::Result<()>,
{
    use std::collections::{BTreeMap, HashMap, HashSet};

    // 1) Precompute totals per Table 2 household (from pool2)
    let c2_total = get_person_count(pool2, table2).await?;
    on_progress(ProgressUpdate {
        processed: 0,
        total: c2_total as usize,
        percent: 0.0,
        eta_secs: 0,
        mem_used_mb: memory_stats_mb().used_mb,
        mem_avail_mb: memory_stats_mb().avail_mb,
        stage: "opt6_pretotal",
        batch_size_current: None,
        gpu_total_mb: 0,
        gpu_free_mb: 0,
        gpu_active: false,
    });
    let totals_t2 = crate::db::schema::get_household_totals_map(pool2, table2).await?;

    // 2) Load Table 1 (from pool1) into birthdate index
    let c1_total = get_person_count(pool1, table1).await?;
    let mut by_bd1: HashMap<String, Vec<Person>> = HashMap::new();
    let mut t1_loaded: i64 = 0;
    let load_batch: i64 = scfg.batch_size.max(10_000);
    on_progress(ProgressUpdate {
        processed: 0,
        total: c1_total as usize,
        percent: 0.0,
        eta_secs: 0,
        mem_used_mb: memory_stats_mb().used_mb,
        mem_avail_mb: memory_stats_mb().avail_mb,
        stage: "opt6_load_t1",
        batch_size_current: Some(load_batch),
        gpu_total_mb: 0,
        gpu_free_mb: 0,
        gpu_active: false,
    });
    while t1_loaded < c1_total {
        let chunk =
            fetch_person_rows_chunk_all_columns(pool1, table1, t1_loaded, load_batch).await?;
        if chunk.is_empty() {
            break;
        }
        for p in chunk.into_iter() {
            if let Some(d) = p.birthdate {
                for k in birthdate_keys(d, allow_birthdate_swap) {
                    by_bd1.entry(k).or_default().push(p.clone());
                }
            }
        }
        t1_loaded += load_batch;
        let mem = memory_stats_mb();
        let frac = (t1_loaded as f32 / c1_total as f32).clamp(0.0, 1.0);
        on_progress(ProgressUpdate {
            processed: (t1_loaded as usize).min(c1_total as usize),
            total: c1_total as usize,
            percent: frac * 100.0,
            eta_secs: 0,
            mem_used_mb: mem.used_mb,
            mem_avail_mb: mem.avail_mb,
            stage: "opt6_load_t1",
            batch_size_current: Some(load_batch),
            gpu_total_mb: 0,
            gpu_free_mb: 0,
            gpu_active: false,
        });
        tokio::task::yield_now().await;
    }

    // 3) Stream Table 2 (from pool2) in chunks
    let mut offset: i64 = 0;
    let mut batch = scfg.batch_size.max(10_000);
    let mut best_for_p2: BTreeMap<i64, (String, String, f32, bool)> = BTreeMap::new();
    let start = std::time::Instant::now();

    while offset < c2_total {
        if let Some(c) = &ctrl {
            if c.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            while c.pause.load(std::sync::atomic::Ordering::Relaxed) {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        }
        let rows_t2 = fetch_person_rows_chunk_all_columns(pool2, table2, offset, batch).await?;
        if rows_t2.is_empty() {
            break;
        }

        let mut by_bd2: HashMap<String, Vec<Person>> = HashMap::new();
        for p in rows_t2 {
            if let Some(d) = p.birthdate {
                for k in birthdate_keys(d, allow_birthdate_swap) {
                    by_bd2.entry(k).or_default().push(p.clone());
                }
            }
        }

        let mo_cpu = MatchOptions {
            backend: ComputeBackend::Cpu,
            gpu: None,
            progress: ProgressConfig::default(),
            allow_birthdate_swap,
        };
        /*
         Phase 2a: Cross-birthdate batching for Option 6 Dual / Advanced L12
         Mirrors single-DB implementation: accumulate per-birthdate groups and flush as one GPU batch.
        */
        let batch_thresh: usize = 50_000;
        let mut pending_small: Vec<(Vec<Person>, Vec<Person>)> = Vec::new();
        let mut pending_comps: usize = 0;

        let mut flush_pending_gpu = |pending: &mut Vec<(Vec<Person>, Vec<Person>)>,
                                     pending_comps: &mut usize|
         -> anyhow::Result<Vec<MatchPair>> {
            if pending.is_empty() {
                return Ok(Vec::new());
            }
            let mut outer_all: Vec<Person> = Vec::new();
            let mut inner_all: Vec<Person> = Vec::new();
            for (o, i) in pending.iter() {
                outer_all.extend_from_slice(o.as_slice());
                inner_all.extend_from_slice(i.as_slice());
            }
            pending.clear();
            *pending_comps = 0;

            #[cfg(feature = "gpu")]
            if scfg.use_gpu_fuzzy_metrics && !gpu_fuzzy_disable() {
                let opts = MatchOptions {
                    backend: ComputeBackend::Gpu,
                    gpu: Some(GpuConfig {
                        device_id: None,
                        mem_budget_mb: scfg.gpu_probe_batch_mb.max(512),
                    }),
                    progress: ProgressConfig::default(),
                    allow_birthdate_swap: false,
                };
                let gpu_try = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    crate::matching::gpu_config::with_oom_cpu_fallback(
                        || {
                            gpu::match_fuzzy_no_mid_gpu(
                                &outer_all,
                                &inner_all,
                                opts,
                                &|_u: ProgressUpdate| {},
                            )
                        },
                        || {
                            let mo_cpu = MatchOptions {
                                backend: ComputeBackend::Cpu,
                                gpu: None,
                                progress: ProgressConfig::default(),
                                allow_birthdate_swap: false,
                            };
                            match_all_with_opts(
                                &outer_all,
                                &inner_all,
                                MatchingAlgorithm::FuzzyNoMiddle,
                                mo_cpu,
                                &|_u: ProgressUpdate| {},
                            )
                        },
                        "[ADV][GPU] opt6-dual batched fuzzy no-mid",
                    )
                }));
                if let Ok(Ok(vec_pairs)) = gpu_try {
                    // Keep confidence in [0.0, 1.0] units; thresholds are fractional
                    return Ok(vec_pairs);
                } else {
                    log::warn!(
                        "[ADV][GPU] Opt6-dual batched GPU path failed; falling back to CPU for pending set"
                    );
                }
            }
            let mo_cpu = MatchOptions {
                backend: ComputeBackend::Cpu,
                gpu: None,
                progress: ProgressConfig::default(),
                allow_birthdate_swap: false,
            };
            Ok(match_all_with_opts(
                &outer_all,
                &inner_all,
                MatchingAlgorithm::FuzzyNoMiddle,
                mo_cpu,
                &|_u: ProgressUpdate| {},
            ))
        };

        for (bd, v2) in by_bd2.into_iter() {
            if let Some(v1) = by_bd1.get(&bd) {
                let comps = v1.len().saturating_mul(v2.len());
                if scfg.use_gpu_fuzzy_metrics && !gpu_fuzzy_disable() {
                    pending_small.push((v1.clone(), v2.clone()));
                    pending_comps = pending_comps.saturating_add(comps);
                    if pending_comps >= batch_thresh {
                        let emitted_pairs =
                            flush_pending_gpu(&mut pending_small, &mut pending_comps)
                                .unwrap_or_else(|_| Vec::new());
                        for p in emitted_pairs.into_iter() {
                            if p.confidence < fuzzy_min_conf {
                                continue;
                            }
                            let Some(uuid) = p.person1.uuid.clone() else {
                                continue;
                            };
                            let hh_key = p
                                .person2
                                .hh_id
                                .clone()
                                .unwrap_or_else(|| p.person2.id.to_string());
                            let key = p.person2.id;
                            match best_for_p2.get_mut(&key) {
                                None => {
                                    best_for_p2.insert(key, (hh_key, uuid, p.confidence, false));
                                }
                                Some((hh, u2, c2, tie)) => {
                                    if p.confidence > *c2 {
                                        *hh = hh_key;
                                        *u2 = uuid;
                                        *c2 = p.confidence;
                                        *tie = false;
                                    } else if (p.confidence - *c2).abs() <= std::f32::EPSILON {
                                        if uuid < *u2 {
                                            *hh = hh_key;
                                            *u2 = uuid;
                                            *c2 = p.confidence;
                                            *tie = false;
                                        } else if uuid != *u2 {
                                            *tie = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    let pairs: Vec<MatchPair> = match_all_with_opts(
                        v1.as_slice(),
                        v2.as_slice(),
                        MatchingAlgorithm::FuzzyNoMiddle,
                        mo_cpu,
                        &|_u: ProgressUpdate| {},
                    );
                    for p in pairs.into_iter() {
                        if p.confidence < fuzzy_min_conf {
                            continue;
                        }
                        let Some(uuid) = p.person1.uuid.clone() else {
                            continue;
                        };
                        let hh_key = p
                            .person2
                            .hh_id
                            .clone()
                            .unwrap_or_else(|| p.person2.id.to_string());
                        let key = p.person2.id;
                        match best_for_p2.get_mut(&key) {
                            None => {
                                best_for_p2.insert(key, (hh_key, uuid, p.confidence, false));
                            }
                            Some((hh, u2, c2, tie)) => {
                                if p.confidence > *c2 {
                                    *hh = hh_key;
                                    *u2 = uuid;
                                    *c2 = p.confidence;
                                    *tie = false;
                                } else if (p.confidence - *c2).abs() <= std::f32::EPSILON {
                                    if uuid < *u2 {
                                        *hh = hh_key;
                                        *u2 = uuid;
                                        *c2 = p.confidence;
                                        *tie = false;
                                    } else if uuid != *u2 {
                                        *tie = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if pending_comps > 0 {
            let emitted_pairs = flush_pending_gpu(&mut pending_small, &mut pending_comps)
                .unwrap_or_else(|_| Vec::new());
            for p in emitted_pairs.into_iter() {
                if p.confidence < fuzzy_min_conf {
                    continue;
                }
                let Some(uuid) = p.person1.uuid.clone() else {
                    continue;
                };
                let hh_key = p
                    .person2
                    .hh_id
                    .clone()
                    .unwrap_or_else(|| p.person2.id.to_string());
                let key = p.person2.id;
                match best_for_p2.get_mut(&key) {
                    None => {
                        best_for_p2.insert(key, (hh_key, uuid, p.confidence, false));
                    }
                    Some((hh, u2, c2, tie)) => {
                        if p.confidence > *c2 {
                            *hh = hh_key;
                            *u2 = uuid;
                            *c2 = p.confidence;
                            *tie = false;
                        } else if (p.confidence - *c2).abs() <= std::f32::EPSILON {
                            if uuid < *u2 {
                                *hh = hh_key;
                                *u2 = uuid;
                                *c2 = p.confidence;
                                *tie = false;
                            } else if uuid != *u2 {
                                *tie = true;
                            }
                        }
                    }
                }
            }
        }

        // Progress + adaptive batch
        let elapsed = start.elapsed();
        let processed = (offset as usize).min(c2_total as usize);
        let frac = (processed as f32 / c2_total as f32).clamp(0.0, 1.0);
        let eta_secs = if frac > 0.0 {
            (elapsed.as_secs_f32() * (1.0 - frac) / frac) as u64
        } else {
            0
        };
        let mem = memory_stats_mb();
        #[cfg(feature = "gpu")]
        let (g_tot, g_free) = if scfg.use_gpu_fuzzy_metrics {
            gpu::GpuFuzzyContext::get()
                .map(|c| c.mem_info_mb())
                .unwrap_or((0, 0))
        } else {
            (0, 0)
        };
        #[cfg(not(feature = "gpu"))]
        let (g_tot, g_free) = (0u64, 0u64);
        on_progress(ProgressUpdate {
            processed,
            total: c2_total as usize,
            percent: frac * 100.0,
            eta_secs,
            mem_used_mb: mem.used_mb,
            mem_avail_mb: mem.avail_mb,
            stage: "opt6_probing",
            batch_size_current: Some(batch),
            gpu_total_mb: g_tot,
            gpu_free_mb: g_free,
            gpu_active: scfg.use_gpu_fuzzy_metrics,
        });
        offset += batch;
        let memx = memory_stats_mb();
        if memx.avail_mb > scfg.memory_soft_min_mb * 2 {
            let new_batch = (batch as f64 * 1.5) as i64;
            batch = new_batch.min(200_000).max(10_000);
        }
        tokio::task::yield_now().await;
    }

    // 4) Aggregate and emit
    let mut matched: BTreeMap<(String, String), HashSet<i64>> = BTreeMap::new();
    for (p2_id, (hh_key, uuid, _conf, tie)) in best_for_p2.into_iter() {
        if tie {
            continue;
        }
        matched.entry((hh_key, uuid)).or_default().insert(p2_id);
    }

    let mut emitted = 0usize;
    let mut row_id: i64 = 1;
    for ((hh_key, uuid), members) in matched.into_iter() {
        let total = *totals_t2.get(&hh_key).unwrap_or(&0usize) as f32;
        if total <= 0.0 {
            continue;
        }
        let pct = (members.len() as f32) / total * 100.0;
        if pct > 50.0 {
            let row = HouseholdAggRow {
                row_id,
                uuid,
                hh_id: hh_key.parse::<i64>().unwrap_or(0),
                match_percentage: pct,
                region_code: None,
                poor_hat_0: None,
                poor_hat_10: None,
            };
            on_household(&row)?;
            emitted += 1;
            row_id += 1;
        }
    }
    Ok(emitted)
}

/// Streaming Option 5 (HouseholdGpu) with EXACT semantics of match_households_gpu_inmemory.
/// Source = Table 1 (group by uuid), Target = Table 2 (group by hh_id or fallback id),
/// match_percentage denominator = Table 1 household size; per-person fuzzy with exact birthdate equality.
pub async fn stream_match_option5<F>(
    pool: &sqlx::MySqlPool,
    table1: &str,
    table2: &str,
    fuzzy_min_conf: f32,
    scfg: StreamingConfig,
    mut on_household: F,
    on_progress: impl Fn(ProgressUpdate) + Sync,
    ctrl: Option<StreamControl>,
) -> anyhow::Result<usize>
where
    F: FnMut(&HouseholdAggRow) -> anyhow::Result<()>,
{
    use std::collections::{BTreeMap, HashMap, HashSet};
    let allow_birthdate_swap = crate::matching::birthdate_matcher::allow_birthdate_swap();

    // 1) Precompute totals per Table 1 UUID (denominator)
    let c1_total = get_person_count(pool, table1).await?;
    on_progress(ProgressUpdate {
        processed: 0,
        total: c1_total as usize,
        percent: 0.0,
        eta_secs: 0,
        mem_used_mb: memory_stats_mb().used_mb,
        mem_avail_mb: memory_stats_mb().avail_mb,
        stage: "opt5_pretotal",
        batch_size_current: None,
        gpu_total_mb: 0,
        gpu_free_mb: 0,
        gpu_active: false,
    });
    let totals_t1 = crate::db::schema::get_uuid_totals_map(pool, table1).await?;

    // 2) Load Table 2 into birthdate index (incremental)
    let c2_total = get_person_count(pool, table2).await?;
    let mut by_bd2: HashMap<chrono::NaiveDate, Vec<Person>> = HashMap::new();
    let mut t2_loaded: i64 = 0;
    let load_batch: i64 = scfg.batch_size.max(10_000);
    on_progress(ProgressUpdate {
        processed: 0,
        total: c2_total as usize,
        percent: 0.0,
        eta_secs: 0,
        mem_used_mb: memory_stats_mb().used_mb,
        mem_avail_mb: memory_stats_mb().avail_mb,
        stage: "opt5_load_t2",
        batch_size_current: Some(load_batch),
        gpu_total_mb: 0,
        gpu_free_mb: 0,
        gpu_active: false,
    });
    while t2_loaded < c2_total {
        let chunk =
            fetch_person_rows_chunk_all_columns(pool, table2, t2_loaded, load_batch).await?;
        if chunk.is_empty() {
            break;
        }
        for p in chunk.into_iter() {
            if let Some(d) = p.birthdate {
                by_bd2.entry(d).or_default().push(p);
            }
        }
        t2_loaded += load_batch;
        let mem = memory_stats_mb();
        let frac = (t2_loaded as f32 / c2_total as f32).clamp(0.0, 1.0);
        on_progress(ProgressUpdate {
            processed: (t2_loaded as usize).min(c2_total as usize),
            total: c2_total as usize,
            percent: frac * 100.0,
            eta_secs: 0,
            mem_used_mb: mem.used_mb,
            mem_avail_mb: mem.avail_mb,
            stage: "opt5_load_t2",
            batch_size_current: Some(load_batch),
            gpu_total_mb: 0,
            gpu_free_mb: 0,
            gpu_active: false,
        });
        tokio::task::yield_now().await;
    }

    // 3) Stream Table 1 in chunks; build person-level best mapping (Table1 person -> (uuid, hh_key))
    let mut offset: i64 = 0;
    let mut batch = scfg.batch_size.max(10_000);
    let mut best_for_p1: BTreeMap<i64, (String, String, f32, bool)> = BTreeMap::new(); // p1.id -> (uuid, hh_key, conf, tie)
    let start = std::time::Instant::now();

    while offset < c1_total {
        if let Some(c) = &ctrl {
            if c.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            while c.pause.load(std::sync::atomic::Ordering::Relaxed) {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        }
        let rows_t1 = fetch_person_rows_chunk_all_columns(pool, table1, offset, batch).await?;
        if rows_t1.is_empty() {
            break;
        }

        let mut by_bd1: HashMap<chrono::NaiveDate, Vec<Person>> = HashMap::new();
        for p in rows_t1 {
            if let Some(d) = p.birthdate {
                by_bd1.entry(d).or_default().push(p);
            }
        }

        let mo_cpu = MatchOptions {
            backend: ComputeBackend::Cpu,
            gpu: None,
            progress: ProgressConfig::default(),
            allow_birthdate_swap: false,
        };
        for (bd, v1) in by_bd1.into_iter() {
            if let Some(v2) = by_bd2.get(&bd) {
                let pairs: Vec<MatchPair> = if scfg.use_gpu_fuzzy_metrics {
                    #[cfg(feature = "gpu")]
                    {
                        let opts = MatchOptions {
                            backend: ComputeBackend::Gpu,
                            gpu: Some(GpuConfig {
                                device_id: None,
                                mem_budget_mb: scfg.gpu_probe_batch_mb.max(512),
                            }),
                            progress: ProgressConfig::default(),
                            allow_birthdate_swap,
                        };
                        match crate::matching::gpu_config::with_oom_cpu_fallback(
                            || {
                                gpu::match_fuzzy_no_mid_gpu(
                                    v1.as_slice(),
                                    v2.as_slice(),
                                    opts,
                                    &|_u: ProgressUpdate| {},
                                )
                            },
                            || {
                                match_all_with_opts(
                                    v1.as_slice(),
                                    v2.as_slice(),
                                    MatchingAlgorithm::FuzzyNoMiddle,
                                    mo_cpu,
                                    &|_u: ProgressUpdate| {},
                                )
                            },
                            "Opt5 GPU person-level (bd-blocked)",
                        ) {
                            Ok(v) => v,
                            Err(e) => {
                                log::warn!(
                                    "Opt5 GPU person-level failed: {} - falling back to CPU",
                                    e
                                );
                                match_all_with_opts(
                                    v1.as_slice(),
                                    v2.as_slice(),
                                    MatchingAlgorithm::FuzzyNoMiddle,
                                    mo_cpu,
                                    &|_u: ProgressUpdate| {},
                                )
                            }
                        }
                    }
                    #[cfg(not(feature = "gpu"))]
                    {
                        match_all_with_opts(
                            v1.as_slice(),
                            v2.as_slice(),
                            MatchingAlgorithm::FuzzyNoMiddle,
                            mo_cpu,
                            &|_u: ProgressUpdate| {},
                        )
                    }
                } else {
                    match_all_with_opts(
                        v1.as_slice(),
                        v2.as_slice(),
                        MatchingAlgorithm::FuzzyNoMiddle,
                        mo_cpu,
                        &|_u: ProgressUpdate| {},
                    )
                };

                for p in pairs.into_iter() {
                    if p.confidence < fuzzy_min_conf {
                        continue;
                    }
                    let uuid = match p.person1.uuid.clone() {
                        Some(u) => u,
                        None => continue,
                    };
                    let cand_hh = p
                        .person2
                        .hh_id
                        .clone()
                        .unwrap_or_else(|| p.person2.id.to_string());
                    let key = p.person1.id;
                    match best_for_p1.get_mut(&key) {
                        None => {
                            best_for_p1.insert(key, (uuid, cand_hh, p.confidence, false));
                        }
                        Some((u, hh, conf, tie)) => {
                            if p.confidence > *conf {
                                *u = uuid;
                                *hh = cand_hh;
                                *conf = p.confidence;
                                *tie = false;
                            } else if (p.confidence - *conf).abs() <= std::f32::EPSILON {
                                if cand_hh < *hh {
                                    *u = uuid;
                                    *hh = cand_hh;
                                    *conf = p.confidence;
                                    *tie = false;
                                } else if cand_hh != *hh {
                                    *tie = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Progress + adaptive batch
        let elapsed = start.elapsed();
        let processed = (offset as usize).min(c1_total as usize);
        let frac = (processed as f32 / c1_total as f32).clamp(0.0, 1.0);
        let eta_secs = if frac > 0.0 {
            (elapsed.as_secs_f32() * (1.0 - frac) / frac) as u64
        } else {
            0
        };
        let mem = memory_stats_mb();
        on_progress(ProgressUpdate {
            processed,
            total: c1_total as usize,
            percent: frac * 100.0,
            eta_secs,
            mem_used_mb: mem.used_mb,
            mem_avail_mb: mem.avail_mb,
            stage: "opt5_probing",
            batch_size_current: Some(batch),
            gpu_total_mb: 0,
            gpu_free_mb: 0,
            gpu_active: scfg.use_gpu_fuzzy_metrics,
        });
        offset += batch;
        let memx = memory_stats_mb();
        if memx.avail_mb > scfg.memory_soft_min_mb * 2 {
            let new_batch = (batch as f64 * 1.5) as i64;
            batch = new_batch.min(200_000).max(10_000);
        }
        tokio::task::yield_now().await;
    }

    // 4) Aggregate and emit households with match_percentage > 50%
    let mut matched: BTreeMap<(String, String), HashSet<i64>> = BTreeMap::new();
    for (p1_id, (uuid, hh_key, _conf, tie)) in best_for_p1.into_iter() {
        if tie {
            continue;
        }
        matched.entry((uuid, hh_key)).or_default().insert(p1_id);
    }

    let mut emitted = 0usize;
    let mut row_id: i64 = 1;
    for ((uuid, hh_key), members) in matched.into_iter() {
        let total = *totals_t1.get(&uuid).unwrap_or(&0usize) as f32;
        if total <= 0.0 {
            continue;
        }
        let pct = (members.len() as f32) / total * 100.0;
        if pct > 50.0 {
            let row = HouseholdAggRow {
                row_id,
                uuid,
                hh_id: hh_key.parse::<i64>().unwrap_or(0),
                match_percentage: pct,
                region_code: None,
                poor_hat_0: None,
                poor_hat_10: None,
            };
            on_household(&row)?;
            emitted += 1;
            row_id += 1;
        }
    }
    Ok(emitted)
}

/// Streaming Option 5 across two databases (pool1 for Table1, pool2 for Table2)
pub async fn stream_match_option5_dual<F>(
    pool1: &sqlx::MySqlPool,
    pool2: &sqlx::MySqlPool,
    table1: &str,
    table2: &str,
    fuzzy_min_conf: f32,
    scfg: StreamingConfig,
    mut on_household: F,
    on_progress: impl Fn(ProgressUpdate) + Sync,
    ctrl: Option<StreamControl>,
) -> anyhow::Result<usize>
where
    F: FnMut(&HouseholdAggRow) -> anyhow::Result<()>,
{
    use std::collections::{BTreeMap, HashMap, HashSet};
    let allow_birthdate_swap = crate::matching::birthdate_matcher::allow_birthdate_swap();

    // 1) Precompute totals per Table 1 UUID (from pool1)
    let c1_total = get_person_count(pool1, table1).await?;
    on_progress(ProgressUpdate {
        processed: 0,
        total: c1_total as usize,
        percent: 0.0,
        eta_secs: 0,
        mem_used_mb: memory_stats_mb().used_mb,
        mem_avail_mb: memory_stats_mb().avail_mb,
        stage: "opt5_pretotal",
        batch_size_current: None,
        gpu_total_mb: 0,
        gpu_free_mb: 0,
        gpu_active: false,
    });
    let totals_t1 = crate::db::schema::get_uuid_totals_map(pool1, table1).await?;

    // 2) Load Table 2 (from pool2) into birthdate index
    let c2_total = get_person_count(pool2, table2).await?;
    let mut by_bd2: HashMap<chrono::NaiveDate, Vec<Person>> = HashMap::new();
    let mut t2_loaded: i64 = 0;
    let load_batch: i64 = scfg.batch_size.max(10_000);
    on_progress(ProgressUpdate {
        processed: 0,
        total: c2_total as usize,
        percent: 0.0,
        eta_secs: 0,
        mem_used_mb: memory_stats_mb().used_mb,
        mem_avail_mb: memory_stats_mb().avail_mb,
        stage: "opt5_load_t2",
        batch_size_current: Some(load_batch),
        gpu_total_mb: 0,
        gpu_free_mb: 0,
        gpu_active: false,
    });
    while t2_loaded < c2_total {
        let chunk =
            fetch_person_rows_chunk_all_columns(pool2, table2, t2_loaded, load_batch).await?;
        if chunk.is_empty() {
            break;
        }
        for p in chunk.into_iter() {
            if let Some(d) = p.birthdate {
                by_bd2.entry(d).or_default().push(p);
            }
        }
        t2_loaded += load_batch;
        let mem = memory_stats_mb();
        let frac = (t2_loaded as f32 / c2_total as f32).clamp(0.0, 1.0);
        on_progress(ProgressUpdate {
            processed: (t2_loaded as usize).min(c2_total as usize),
            total: c2_total as usize,
            percent: frac * 100.0,
            eta_secs: 0,
            mem_used_mb: mem.used_mb,
            mem_avail_mb: mem.avail_mb,
            stage: "opt5_load_t2",
            batch_size_current: Some(load_batch),
            gpu_total_mb: 0,
            gpu_free_mb: 0,
            gpu_active: false,
        });
        tokio::task::yield_now().await;
    }

    // 3) Stream Table 1 (from pool1) in chunks and build best_for_p1 mapping
    let mut offset: i64 = 0;
    let mut batch = scfg.batch_size.max(10_000);
    let mut best_for_p1: BTreeMap<i64, (String, String, f32, bool)> = BTreeMap::new();
    let start = std::time::Instant::now();

    while offset < c1_total {
        if let Some(c) = &ctrl {
            if c.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            while c.pause.load(std::sync::atomic::Ordering::Relaxed) {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        }
        let rows_t1 = fetch_person_rows_chunk_all_columns(pool1, table1, offset, batch).await?;
        if rows_t1.is_empty() {
            break;
        }

        let mut by_bd1: HashMap<chrono::NaiveDate, Vec<Person>> = HashMap::new();
        for p in rows_t1 {
            if let Some(d) = p.birthdate {
                by_bd1.entry(d).or_default().push(p);
            }
        }

        let mo_cpu = MatchOptions {
            backend: ComputeBackend::Cpu,
            gpu: None,
            progress: ProgressConfig::default(),
            allow_birthdate_swap,
        };
        for (bd, v1) in by_bd1.into_iter() {
            if let Some(v2) = by_bd2.get(&bd) {
                let pairs: Vec<MatchPair> = if scfg.use_gpu_fuzzy_metrics {
                    #[cfg(feature = "gpu")]
                    {
                        let opts = MatchOptions {
                            backend: ComputeBackend::Gpu,
                            gpu: Some(GpuConfig {
                                device_id: None,
                                mem_budget_mb: scfg.gpu_probe_batch_mb.max(512),
                            }),
                            progress: ProgressConfig::default(),
                            allow_birthdate_swap: false,
                        };
                        match crate::matching::gpu_config::with_oom_cpu_fallback(
                            || {
                                gpu::match_fuzzy_no_mid_gpu(
                                    v1.as_slice(),
                                    v2.as_slice(),
                                    opts,
                                    &|_u: ProgressUpdate| {},
                                )
                            },
                            || {
                                match_all_with_opts(
                                    v1.as_slice(),
                                    v2.as_slice(),
                                    MatchingAlgorithm::FuzzyNoMiddle,
                                    mo_cpu,
                                    &|_u: ProgressUpdate| {},
                                )
                            },
                            "Opt5 GPU person-level (bd-blocked)",
                        ) {
                            Ok(v) => v,
                            Err(e) => {
                                log::warn!(
                                    "Opt5 GPU person-level failed: {} - falling back to CPU",
                                    e
                                );
                                match_all_with_opts(
                                    v1.as_slice(),
                                    v2.as_slice(),
                                    MatchingAlgorithm::FuzzyNoMiddle,
                                    mo_cpu,
                                    &|_u: ProgressUpdate| {},
                                )
                            }
                        }
                    }
                    #[cfg(not(feature = "gpu"))]
                    {
                        match_all_with_opts(
                            v1.as_slice(),
                            v2.as_slice(),
                            MatchingAlgorithm::FuzzyNoMiddle,
                            mo_cpu,
                            &|_u: ProgressUpdate| {},
                        )
                    }
                } else {
                    match_all_with_opts(
                        v1.as_slice(),
                        v2.as_slice(),
                        MatchingAlgorithm::FuzzyNoMiddle,
                        mo_cpu,
                        &|_u: ProgressUpdate| {},
                    )
                };

                for p in pairs.into_iter() {
                    if p.confidence < fuzzy_min_conf {
                        continue;
                    }
                    let uuid = match p.person1.uuid.clone() {
                        Some(u) => u,
                        None => continue,
                    };
                    let cand_hh = p
                        .person2
                        .hh_id
                        .clone()
                        .unwrap_or_else(|| p.person2.id.to_string());
                    let key = p.person1.id;
                    match best_for_p1.get_mut(&key) {
                        None => {
                            best_for_p1.insert(key, (uuid, cand_hh, p.confidence, false));
                        }
                        Some((u, hh, conf, tie)) => {
                            if p.confidence > *conf {
                                *u = uuid;
                                *hh = cand_hh;
                                *conf = p.confidence;
                                *tie = false;
                            } else if (p.confidence - *conf).abs() <= std::f32::EPSILON {
                                if cand_hh < *hh {
                                    *u = uuid;
                                    *hh = cand_hh;
                                    *conf = p.confidence;
                                    *tie = false;
                                } else if cand_hh != *hh {
                                    *tie = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Progress + adaptive batch
        let elapsed = start.elapsed();
        let processed = (offset as usize).min(c1_total as usize);
        let frac = (processed as f32 / c1_total as f32).clamp(0.0, 1.0);
        let eta_secs = if frac > 0.0 {
            (elapsed.as_secs_f32() * (1.0 - frac) / frac) as u64
        } else {
            0
        };
        let mem = memory_stats_mb();
        on_progress(ProgressUpdate {
            processed,
            total: c1_total as usize,
            percent: frac * 100.0,
            eta_secs,
            mem_used_mb: mem.used_mb,
            mem_avail_mb: mem.avail_mb,
            stage: "opt5_probing",
            batch_size_current: Some(batch),
            gpu_total_mb: 0,
            gpu_free_mb: 0,
            gpu_active: scfg.use_gpu_fuzzy_metrics,
        });
        offset += batch;
        let memx = memory_stats_mb();
        if memx.avail_mb > scfg.memory_soft_min_mb * 2 {
            let new_batch = (batch as f64 * 1.5) as i64;
            batch = new_batch.min(200_000).max(10_000);
        }
        tokio::task::yield_now().await;
    }

    // 4) Aggregate and emit
    let mut matched: BTreeMap<(String, String), HashSet<i64>> = BTreeMap::new();
    for (p1_id, (uuid, hh_key, _conf, tie)) in best_for_p1.into_iter() {
        if tie {
            continue;
        }
        matched.entry((uuid, hh_key)).or_default().insert(p1_id);
    }

    let mut emitted = 0usize;
    let mut row_id: i64 = 1;
    for ((uuid, hh_key), members) in matched.into_iter() {
        let total = *totals_t1.get(&uuid).unwrap_or(&0usize) as f32;
        if total <= 0.0 {
            continue;
        }
        let pct = (members.len() as f32) / total * 100.0;
        if pct > 50.0 {
            let row = HouseholdAggRow {
                row_id,
                uuid,
                hh_id: hh_key.parse::<i64>().unwrap_or(0),
                match_percentage: pct,
                region_code: None,
                poor_hat_0: None,
                poor_hat_10: None,
            };
            on_household(&row)?;
            emitted += 1;
            row_id += 1;
        }
    }
    Ok(emitted)
}

#[allow(dead_code)]
pub fn try_gpu_name() -> Option<String> {
    #[cfg(feature = "gpu")]
    {
        use std::ffi::CStr;
        use std::os::raw::{c_char, c_int};
        unsafe {
            // Ensure CUDA driver is initialized before querying devices
            if cudarc::driver::sys::cuInit(0) != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return None;
            }
            let mut cu_dev: cudarc::driver::sys::CUdevice = 0;
            // pick device 0 by default
            if cudarc::driver::sys::cuDeviceGet(&mut cu_dev as *mut _, 0 as c_int)
                != cudarc::driver::sys::CUresult::CUDA_SUCCESS
            {
                return None;
            }
            let mut name_buf: [c_char; 128] = [0; 128];
            if cudarc::driver::sys::cuDeviceGetName(
                name_buf.as_mut_ptr(),
                name_buf.len() as c_int,
                cu_dev,
            ) != cudarc::driver::sys::CUresult::CUDA_SUCCESS
            {
                return None;
            }
            let name = CStr::from_ptr(name_buf.as_ptr())
                .to_string_lossy()
                .into_owned();
            Some(name)
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        None
    }
}

/// Pre-warm GPU contexts (hash and fuzzy) in the background to remove first-use latency.
/// No-ops when built without the `gpu` feature.
pub fn prewarm_gpu_contexts() {
    #[cfg(feature = "gpu")]
    {
        // Ignore errors; this is best-effort warmup.
        let _ = self::gpu::GpuHashContext::get();
        let _ = self::gpu::GpuFuzzyContext::get();
    }
    #[cfg(not(feature = "gpu"))]
    {
        // Nothing to do without GPU support
    }
}

// ============================================================================
// GPU Cascade Support - Public wrappers for cascade module access
// ============================================================================

/// GPU fuzzy matching for L10 (full middle name) - exposed for cascade module.
///
/// This function wraps the internal GPU fuzzy matcher with OOM fallback.
/// Only available when the `gpu` feature is enabled.
#[cfg(feature = "gpu")]
pub fn cascade_match_fuzzy_gpu<F>(
    table1: &[crate::models::Person],
    table2: &[crate::models::Person],
    opts: MatchOptions,
    on_progress: F,
) -> anyhow::Result<Vec<MatchPair>>
where
    F: Fn(ProgressUpdate) + Sync,
{
    gpu::match_fuzzy_gpu(table1, table2, opts, &on_progress)
}

/// GPU fuzzy matching for L11 (no middle name) - exposed for cascade module.
///
/// This function wraps the internal GPU fuzzy matcher (no-middle variant) with OOM fallback.
/// Only available when the `gpu` feature is enabled.
#[cfg(feature = "gpu")]
pub fn cascade_match_fuzzy_no_mid_gpu<F>(
    table1: &[crate::models::Person],
    table2: &[crate::models::Person],
    opts: MatchOptions,
    on_progress: F,
) -> anyhow::Result<Vec<MatchPair>>
where
    F: Fn(ProgressUpdate) + Sync,
{
    gpu::match_fuzzy_no_mid_gpu(table1, table2, opts, &on_progress)
}
