#![allow(
    dead_code,
    unused_assignments,
    unused_comparisons,
    unused_variables,
    clippy::absurd_extreme_comparisons,
    clippy::collapsible_if,
    clippy::comparison_to_empty,
    clippy::derivable_impls,
    clippy::empty_line_after_doc_comments,
    clippy::explicit_counter_loop,
    clippy::legacy_numeric_constants,
    clippy::manual_clamp,
    clippy::manual_range_contains,
    clippy::manual_range_patterns,
    clippy::needless_borrow,
    clippy::needless_borrows_for_generic_args,
    clippy::needless_lifetimes,
    clippy::needless_return,
    clippy::print_literal,
    clippy::ptr_arg,
    clippy::too_many_arguments,
    clippy::unnecessary_cast,
    clippy::unnecessary_map_or,
    clippy::useless_conversion,
    clippy::write_literal
)]
// Legacy lint surface tracked in docs/optimization-sweep-plan.md T12.
// Keep strict clippy running while deferring the large matching/export refactor.

pub mod config;
pub mod db;
pub mod export;
pub mod loaders;
pub mod matching;
pub mod metrics;
pub mod models;
pub mod normalize;
pub mod util;

pub mod optimization;

pub mod run_service;

#[cfg(feature = "new_engine")]
pub mod engine;
#[cfg(feature = "new_engine")]
pub mod matching_algorithms {
    pub use crate::matching::algorithms::*;
}

pub mod error;
