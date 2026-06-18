use super::*;
use std::cell::Cell;

thread_local! {
    static NO_MID_CLASSIFY: Cell<bool> = Cell::new(false);
}

#[inline]
pub(crate) fn gpu_no_mid_mode() -> bool {
    NO_MID_CLASSIFY.with(|flag| flag.get())
}

#[inline]
pub(crate) fn with_no_mid_classification<T, F: FnOnce() -> T>(f: F) -> T {
    NO_MID_CLASSIFY.with(|flag| {
        let prev = flag.replace(true);
        let out = f();
        flag.set(prev);
        out
    })
}

#[inline]
pub(crate) fn fuzzy_cache_name(cache: &FuzzyCache) -> &str {
    if gpu_no_mid_mode() {
        &cache.simple_full_no_mid
    } else {
        &cache.simple_full
    }
}

#[inline]
pub(crate) fn full_middle_len(p: &Person) -> usize {
    p.middle_name
        .as_deref()
        .unwrap_or("")
        .trim()
        .trim_matches('.')
        .chars()
        .filter(|c| !c.is_whitespace())
        .count()
}
