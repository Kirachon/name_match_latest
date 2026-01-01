//! GPU module for Name_Matcher matching subsystem.
//! Exposes GPU batching and related helpers.

pub mod batch;

// Re-export commonly used items from parent matching module as needed
pub use crate::matching::*;

