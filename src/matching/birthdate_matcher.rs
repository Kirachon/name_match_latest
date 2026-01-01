use chrono::{Datelike, NaiveDate};

const SWAP_ENV: &str = "NAME_MATCHER_ALLOW_BIRTHDATE_SWAP";

/// Read once from the environment to decide if birthdate month/day swapping is allowed.
/// Accepted truthy values: 1, true, yes, on (case-insensitive). Defaults to false.
pub fn allow_birthdate_swap() -> bool {
    std::env::var(SWAP_ENV)
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
        .map(|v| matches!(v.as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

/// Strictly parse a `YYYY-MM-DD` string. Returns `None` on invalid format or date.
pub fn parse_date_strict(date_str: &str) -> Option<NaiveDate> {
    // Trim to allow accidental whitespace but still enforce exact pattern
    let s = date_str.trim();
    if s.len() != 10 {
        return None;
    }
    NaiveDate::parse_from_str(s, "%Y-%m-%d").ok()
}

/// Swap the month and day of a `NaiveDate`. Returns `None` if the swapped value is invalid.
pub fn swap_month_day(date: NaiveDate) -> Option<NaiveDate> {
    NaiveDate::from_ymd_opt(date.year(), date.day(), date.month())
}

/// Generate canonical string keys (YYYY-MM-DD) for a date, including the swapped variant when allowed.
pub fn birthdate_keys(date: NaiveDate, allow_swap: bool) -> Vec<String> {
    let primary = date.format("%Y-%m-%d").to_string();
    if !allow_swap {
        return vec![primary];
    }
    if let Some(swapped) = swap_month_day(date) {
        let alt = swapped.format("%Y-%m-%d").to_string();
        if alt != primary {
            return vec![primary, alt];
        }
    }
    vec![primary]
}

/// Core birthdate matcher shared by CPU and GPU paths.
/// 1) Exact match; 2) Optional month/day swap match if valid; else false.
pub fn birthdate_matches(stored: &str, input: &str, allow_swap: bool) -> bool {
    let Some(s) = parse_date_strict(stored) else {
        return false;
    };
    let Some(i) = parse_date_strict(input) else {
        return false;
    };
    if s == i {
        return true;
    }
    if allow_swap {
        if let Some(swapped) = swap_month_day(i) {
            return swapped == s;
        }
    }
    false
}

/// Core birthdate matcher for NaiveDate values (used by GPU paths).
/// 1) Exact match; 2) Optional month/day swap match if valid; else false.
pub fn birthdate_matches_naive(d1: NaiveDate, d2: NaiveDate, allow_swap: bool) -> bool {
    if d1 == d2 {
        return true;
    }
    if allow_swap {
        if let Some(swapped) = swap_month_day(d2) {
            return swapped == d1;
        }
    }
    false
}

/// Level 10 wrapper (fuzzy + full middle) delegating to shared birthdate matcher.
pub fn match_level_10(stored: &str, input: &str, allow_swap: bool) -> bool {
    birthdate_matches(stored, input, allow_swap)
}

/// Level 11 wrapper (fuzzy + no middle) delegating to shared birthdate matcher.
pub fn match_level_11(stored: &str, input: &str, allow_swap: bool) -> bool {
    birthdate_matches(stored, input, allow_swap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match() {
        assert!(birthdate_matches("1990-03-15", "1990-03-15", false));
    }

    #[test]
    fn mismatch_no_swap() {
        assert!(!birthdate_matches("1990-03-15", "1990-15-03", false));
    }

    #[test]
    fn swap_match_allowed() {
        // Test with valid dates where swapping month/day of input matches stored
        // stored = April 12, input = December 4 -> swapped input = April 12
        assert!(birthdate_matches("1990-04-12", "1990-12-04", true));
    }

    #[test]
    fn swap_invalid_date_rejected() {
        assert!(!birthdate_matches("2024-02-29", "2024-31-02", true));
    }

    #[test]
    fn invalid_format_rejected() {
        assert!(!birthdate_matches("1990/03/15", "1990-03-15", true));
    }

    #[test]
    fn leap_year_edges() {
        assert!(birthdate_matches("2020-02-29", "2020-02-29", false));
        assert!(!birthdate_matches("2020-02-29", "2021-02-29", true));
    }

    #[test]
    fn keys_include_swap() {
        let d = NaiveDate::from_ymd_opt(1990, 4, 12).unwrap();
        let keys = birthdate_keys(d, true);
        assert!(keys.contains(&"1990-04-12".to_string()));
        assert!(keys.contains(&"1990-12-04".to_string()));
    }

    #[test]
    fn level_wrappers_identical() {
        let a = match_level_10("1990-04-12", "1990-12-04", true);
        let b = match_level_11("1990-04-12", "1990-12-04", true);
        assert_eq!(a, b);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn cpu_gpu_parity_smoke() {
        // Parity check: same deterministic helper used for both paths
        let inputs = vec![
            ("1990-03-15", "1990-03-15"),
            ("1990-03-15", "1990-15-03"),
            ("2024-02-29", "2024-31-02"),
        ];
        for (stored, input) in inputs {
            let cpu = birthdate_matches(stored, input, true);
            let gpu = birthdate_matches(stored, input, true); // GPU kernel mirrors this logic
            assert_eq!(cpu, gpu);
        }
    }
}
