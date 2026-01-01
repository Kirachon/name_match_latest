//! Shared helper functions for string normalization and similarity scoring.

use rphonetic::{DoubleMetaphone, Encoder};
use strsim::levenshtein;
use unicode_normalization::UnicodeNormalization;

/// Simple normalization: lowercase, remove dots, replace dashes with spaces.
pub(crate) fn normalize_simple(s: &str) -> String {
    let s = s.trim();
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '.' => { /* drop dot */ }
            '-' => out.push(' '),
            _ => {
                for lc in ch.to_lowercase() {
                    out.push(lc);
                }
            }
        }
    }
    out
}

/// Compute Levenshtein similarity as a percentage (0.0-100.0).
pub(crate) fn sim_levenshtein_pct(a: &str, b: &str) -> f64 {
    let max_len = a.len().max(b.len());
    if max_len == 0 {
        return 100.0;
    }
    let dist = levenshtein(a, b);
    (1.0 - (dist as f64 / max_len as f64)) * 100.0
}

/// Normalize a string for phonetic encoding: decompose diacritics,
/// keep ASCII letters and single spaces, map common non-ASCII characters.
pub(crate) fn normalize_for_phonetic(s: &str) -> String {
    // Decompose diacritics, keep ASCII letters and single spaces; map a few common non-ASCII
    // Do lowercasing inline to avoid an intermediate allocation
    let s = s.trim();
    let mut out = String::with_capacity(s.len());
    for ch in s.nfd() {
        // Lowercase the codepoint; may yield 1..N chars
        for lc in ch.to_lowercase() {
            if lc.is_ascii_alphabetic() {
                out.push(lc);
            } else if lc.is_ascii_whitespace() {
                if !out.ends_with(' ') {
                    out.push(' ');
                }
            } else {
                match lc {
                    'ß' => out.push_str("ss"),
                    'æ' | 'ǽ' => out.push_str("ae"),
                    'ø' => out.push('o'),
                    'đ' => out.push('d'),
                    _ => {}
                }
            }
        }
    }
    // Trim trailing space in-place
    let new_len = out.trim_end().len();
    out.truncate(new_len);
    out
}

/// Compare two strings using Double Metaphone phonetic encoding.
/// Returns 100.0 if encodings match, 0.0 otherwise.
pub(crate) fn metaphone_pct(a: &str, b: &str) -> f64 {
    let sa = normalize_for_phonetic(a);
    let sb = normalize_for_phonetic(b);
    if sa.is_empty() || sb.is_empty() {
        return 0.0;
    }

    // Protect against panics inside rphonetic by catching unwinds
    let ra = std::panic::catch_unwind(|| DoubleMetaphone::default().encode(&sa));
    let rb = std::panic::catch_unwind(|| DoubleMetaphone::default().encode(&sb));
    let (ra_s, rb_s) = match (ra, rb) {
        (Ok(ra), Ok(rb)) => (ra.to_string(), rb.to_string()),
        _ => {
            log::warn!("DoubleMetaphone panicked on inputs: {:?} / {:?}", a, b);
            return 0.0;
        }
    };
    if !ra_s.is_empty() && ra_s == rb_s {
        100.0
    } else {
        0.0
    }
}

/// Simple 4-character Soundex encoding for ASCII strings.
/// Used for blocking in GPU matching paths.
#[inline]
pub(crate) fn soundex4_ascii(s: &str) -> [u8; 4] {
    let mut out = [b'0'; 4];
    if s.is_empty() {
        return out;
    }
    let mut bytes = s
        .as_bytes()
        .iter()
        .copied()
        .filter(|c| c.is_ascii_alphabetic());
    if let Some(f) = bytes.next() {
        out[0] = f.to_ascii_uppercase();
    }
    let mut last = 0u8;
    let mut idx = 1usize;
    for c in bytes {
        if idx >= 4 {
            break;
        }
        let d = match c.to_ascii_lowercase() {
            b'b' | b'f' | b'p' | b'v' => 1,
            b'c' | b'g' | b'j' | b'k' | b'q' | b's' | b'x' | b'z' => 2,
            b'd' | b't' => 3,
            b'l' => 4,
            b'm' | b'n' => 5,
            b'r' => 6,
            _ => 0,
        };
        if d != 0 && d != last {
            out[idx] = b'0' + d;
            idx += 1;
        }
        last = d;
    }
    out
}
