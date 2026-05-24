// Hide the console window on Windows release builds. Dev / debug builds keep
// it so the engine logs are visible for ops triage.
#![allow(dead_code, clippy::wrong_self_convention)]
#![cfg_attr(all(not(debug_assertions), windows), windows_subsystem = "windows")]

mod commands;
mod error;
mod state;

use crate::state::AppState;
use std::sync::Arc;
use tauri::Manager;

fn main() {
    // Initialise env_logger so engine `log::*` calls reach the console in
    // dev mode. The Tauri shell additionally pipes structured `LogEntryDto`
    // events to the front-end via the `log-entry` channel.
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Pre-init Rayon with a sensible global pool so Tauri / Tokio still have
    // CPU left over for IPC + system work. The engine relies on the global
    // Rayon pool internally.
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    let rayon_threads = configured_rayon_threads(cpus);
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(rayon_threads)
        .thread_name(|i| format!("nm-rayon-{i}"))
        .build_global();
    log::info!(
        "Rayon global pool initialised with {} worker threads (host has {} logical CPUs)",
        rayon_threads,
        cpus
    );

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_store::Builder::default().build())
        .plugin(tauri_plugin_window_state::Builder::default().build())
        .setup(|app| {
            let state = Arc::new(AppState::new(app.handle().clone()));
            app.manage(state);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            // System / config (T2)
            commands::system_info,
            commands::cuda_diagnostics,
            commands::load_config,
            commands::save_config,
            // Database session (T5)
            commands::connect_db,
            commands::validate_db_credentials,
            commands::test_connection,
            commands::list_tables,
            commands::get_table_columns,
            commands::get_row_count,
            commands::disconnect_db,
            commands::list_sessions,
            // Matching lifecycle (T7)
            commands::start_matching,
            commands::cancel_matching,
            commands::pause_matching,
            commands::resume_matching,
            commands::get_matching_status,
            commands::list_matching_jobs,
            commands::forget_matching_job,
            // Results / export (T9)
            commands::get_results_page,
            commands::export_results,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

fn configured_rayon_threads(cpus: usize) -> usize {
    std::env::var("RAYON_NUM_THREADS")
        .ok()
        .or_else(|| std::env::var("NAME_MATCHER_RAYON_THREADS").ok())
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|threads| *threads > 0)
        .unwrap_or_else(|| cpus.saturating_sub(2).max(1))
}
