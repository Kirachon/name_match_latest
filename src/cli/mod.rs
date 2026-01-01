//! CLI module: argument parsing, configuration, and algorithm dispatch.
//!
//! This module provides:
//! - `clap_parser`: Modern clap-based CLI parsing
//! - `args`: Legacy argument parsing and environment variable handling
//! - `flags`: CLI flag parsing for GPU, streaming, and advanced options
//!
//! ## Integration Path
//!
//! These modules are ready for integration into `src/main.rs`. To adopt them
//! incrementally:
//!
//! ### Step 1: Integrate `CliFlags` (Low Risk)
//!
//! Replace the manual flag parsing in `main.rs` lines 594-627 with:
//!
//! ```rust,ignore
//! use crate::cli::flags::CliFlags;
//! let cli_flags = CliFlags::from_args(&args);
//! // Then use cli_flags.gpu_hash_join instead of gpu_hash_flag, etc.
//! ```
//!
//! This consolidates 15+ flag variables into a single struct.
//!
//! ### Step 2: Integrate `args` utilities (Low Risk)
//!
//! Replace argument parsing in `main.rs` lines 430-478 with:
//!
//! ```rust,ignore
//! use crate::cli::args::{parse_db_config, parse_table_names, parse_algo_num, parse_out_path, parse_format};
//! let db_config = parse_db_config()?;
//! let (table1, table2) = parse_table_names(&args)?;
//! let algo_num = parse_algo_num(&args)?;
//! let algorithm = args::algo_from_num(algo_num)?;
//! let out_path = parse_out_path(&args)?;
//! let format = parse_format(&args)?;
//! ```
//!
//! ### Step 3: Adopt clap-based parsing (Medium Risk)
//!
//! For a complete CLI overhaul, replace all manual parsing with:
//!
//! ```rust,ignore
//! use crate::cli::{Cli, parse_cli_to_app_config};
//! let cli = Cli::parse(); // clap does all the work
//! let (db_config, table1, table2, algorithm, out_path, format) = parse_cli_to_app_config(&cli)?;
//! ```
//!
//! This requires updating all downstream code that references individual variables.
//!
//! ### Validation After Each Step
//!
//! After each integration step, run:
//! ```bash
//! cargo test --all
//! cargo test streaming_parity --release
//! ```

pub mod args;
mod clap_parser;
pub mod flags;

// Re-export clap-based parser types
pub use clap_parser::{Cli, FormatOpt, parse_cli_to_app_config};
