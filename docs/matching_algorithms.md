# Matching Algorithms Reference

## Overview

The application provides 7 matching algorithms plus a cascade workflow (L1–L11). All algorithms operate on normalized text (Unicode NFD, diacritics removed, lowercased, trimmed).

---

## Algorithm 1 — Deterministic (First + Last + Birthdate)

**Type:** Exact match
**Fields compared:** `first_name`, `last_name`, `birthdate`
**GPU support:** Hash join prefilter (`--gpu-hash-join`)

A match is recorded when all three fields are identical after normalization. Middle name is ignored.

---

## Algorithm 2 — Deterministic (First + Middle + Last + Birthdate)

**Type:** Exact match
**Fields compared:** `first_name`, `middle_name`, `last_name`, `birthdate`
**GPU support:** Hash join prefilter (`--gpu-hash-join`)

Strictest algorithm — requires all four fields to match exactly.

---

## Algorithm 3 — Fuzzy Match (with Middle Name)

**Type:** Fuzzy scoring
**Fields compared:** Full name (first + middle + last) + birthdate
**GPU support:** Fuzzy metrics (`--gpu-fuzzy-metrics`), direct hash pre-pass

### Scoring System

Three metrics are computed on the concatenated full name:

1. **Levenshtein similarity %** — `(1 - edit_distance/max_len) × 100`
2. **Jaro-Winkler similarity** — `jaro_winkler × 100`
3. **Double Metaphone %** — phonetic encoding match percentage

### Match Cases

| Case | Criteria | Score |
|------|----------|-------|
| Direct Match | Normalized names identical | 100% |
| Case 1 | Lev ≥ 85% AND JW ≥ 85% AND Metaphone = 100% | Average of 3 |
| Case 2 | At least 2 of 3 metrics pass thresholds | Average of 3 |
| Case 3 | Case 2 + avg ≥ 88% + per-part Levenshtein ≤ 2 | Average of 3 |

Birthdate must match exactly (with optional month/day swap: `12/04` ↔ `04/12`).

---

## Algorithm 4 — Fuzzy Match (without Middle Name)

**Type:** Fuzzy scoring
**Fields compared:** First + Last name only + birthdate
**GPU support:** Same as Algorithm 3

Same scoring logic as Algorithm 3 but excludes middle name from the full-name concatenation.

---

## Algorithm 5 — Household Matching (Table1 → Table2)

**Type:** GPU-accelerated household grouping
**Fields:** `uuid` (Table 1) → `hh_id` (Table 2)
**Requires:** `hh_id` column in Table 2, GPU recommended

Matches individual persons from Table 1 to household groups in Table 2. Denominator is Table 1 size.

---

## Algorithm 6 — Household Matching (Table2 → Table1)

**Type:** GPU-accelerated household grouping
**Fields:** `hh_id` (Table 2) → `uuid` (Table 1)
**Requires:** `hh_id` column in Table 2, GPU recommended

Reverse direction — denominator is Table 2 size.

---

## Algorithm 7 — Levenshtein Weighted

**Type:** Weighted fuzzy scoring
**GPU support:** Pre-pass (`--gpu-levenshtein-prepass`), full scoring (`--gpu-levenshtein-full-scoring`)

Applies SQL-equivalent weighted Levenshtein scoring with configurable thresholds. Supports GPU acceleration for both candidate filtering and full scoring passes.

---

## Cascade Matching (L1–L11)

Sequential execution of progressively relaxed matching levels. Records matched at earlier levels are excluded from later levels.

| Level | Type | Fields | Geographic Key |
|-------|------|--------|----------------|
| L1 | Exact | Last + First + Full Middle + Birthdate | — |
| L2 | Exact | Last + First + Middle Initial(s) + Birthdate | — |
| L3 | Exact | Last + First + Birthdate | — |
| L4 | Exact | Last + First + Full Middle | barangay_code |
| L5 | Exact | Last + First + Middle Initial(s) | barangay_code |
| L6 | Exact | Last + First | barangay_code |
| L7 | Exact | Last + First + Full Middle | city_code |
| L8 | Exact | Last + First + Middle Initial(s) | city_code |
| L9 | Exact | Last + First | city_code |
| L10 | Fuzzy | Last + First + Full Middle + Birthdate | — |
| L11 | Fuzzy | Last + First + Birthdate (no middle) | — |

### Missing Column Modes

- **AutoSkip** (default): Levels requiring missing columns are skipped silently
- **ManualSelect**: Only user-specified levels are executed
- **AbortOnMissing**: Execution fails if required geographic columns are absent

### Birthdate Swap

Optional tolerance for month/day transposition (e.g., `2000-12-04` matches `2000-04-12`). Controlled via environment variable or GUI toggle.
