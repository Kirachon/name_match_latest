use super::*;

// Per-person cache to avoid repeated normalization and metaphone encoding during GPU post-processing
#[derive(Clone)]
pub(crate) struct FuzzyCache {
    pub(crate) simple_full: String,
    pub(crate) simple_full_no_mid: String,
    pub(crate) simple_first: String,
    pub(crate) simple_mid: String,
    pub(crate) simple_last: String,
    pub(crate) phonetic_full: String,
    pub(crate) dmeta_code: String, // empty if encode failed/panicked/empty
    pub(crate) dmeta_code_no_mid: String, // empty if encode failed/panicked/empty
}

pub(crate) fn build_cache_from_person(p: &Person) -> FuzzyCache {
    let raw_first = p.first_name.as_deref().unwrap_or("");
    let raw_mid = p.middle_name.as_deref().unwrap_or("");
    let raw_last = p.last_name.as_deref().unwrap_or("");

    let simple_first = normalize_simple(raw_first);
    let simple_mid = normalize_simple(raw_mid);
    let simple_last = normalize_simple(raw_last);
    let simple_full = normalize_simple(&format!("{} {} {}", raw_first, raw_mid, raw_last));
    let simple_full_no_mid = normalize_simple(&format!("{} {}", raw_first, raw_last));
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
    let phonetic_no_mid = normalize_for_phonetic(&simple_full_no_mid);
    let dmeta_code_no_mid = if phonetic_no_mid.is_empty() {
        String::new()
    } else {
        match std::panic::catch_unwind(|| DoubleMetaphone::default().encode(&phonetic_no_mid)) {
            Ok(code) => code.to_string(),
            Err(_) => String::new(),
        }
    };
    FuzzyCache {
        simple_full,
        simple_full_no_mid,
        simple_first,
        simple_mid,
        simple_last,
        phonetic_full,
        dmeta_code,
        dmeta_code_no_mid,
    }
}

// Authoritative CPU classification using cached strings/codes to eliminate recomputation
pub(crate) fn classify_pair_cached(c1: &FuzzyCache, c2: &FuzzyCache) -> Option<(f64, String)> {
    // Direct match
    if c1.simple_full == c2.simple_full {
        return Some((100.0, "DIRECT MATCH".to_string()));
    }
    // Metrics
    let lev = sim_levenshtein_pct(&c1.simple_full, &c2.simple_full);
    let jw = jaro_winkler(&c1.simple_full, &c2.simple_full) * 100.0;
    let mp =
        if !c1.dmeta_code.is_empty() && !c2.dmeta_code.is_empty() && c1.dmeta_code == c2.dmeta_code
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

pub(crate) fn classify_pair_cached_no_mid(
    c1: &FuzzyCache,
    c2: &FuzzyCache,
) -> Option<(f64, String)> {
    if c1.simple_full_no_mid.trim().is_empty() || c2.simple_full_no_mid.trim().is_empty() {
        return None;
    }
    if c1.simple_full_no_mid == c2.simple_full_no_mid {
        return Some((100.0, "DIRECT MATCH".to_string()));
    }
    let lev = sim_levenshtein_pct(&c1.simple_full_no_mid, &c2.simple_full_no_mid);
    let jw = jaro_winkler(&c1.simple_full_no_mid, &c2.simple_full_no_mid) * 100.0;
    let mp = if !c1.dmeta_code_no_mid.is_empty()
        && !c2.dmeta_code_no_mid.is_empty()
        && c1.dmeta_code_no_mid == c2.dmeta_code_no_mid
    {
        100.0
    } else {
        0.0
    };

    if lev >= 85.0 && jw >= 85.0 && mp == 100.0 {
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
    if mp == 100.0 {
        pass += 1;
    }
    if pass >= 2 {
        let avg = (lev + jw + mp) / 3.0;
        if avg >= 88.0 {
            let ld_first = levenshtein(&c1.simple_first, &c2.simple_first) as usize;
            let ld_last = levenshtein(&c1.simple_last, &c2.simple_last) as usize;
            if ld_first <= 2 && ld_last <= 2 {
                return Some((avg, "CASE 3".to_string()));
            }
        }
        return Some((avg, "CASE 2".to_string()));
    }
    None
}
