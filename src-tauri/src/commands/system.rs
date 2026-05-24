use crate::error::{AppError, AppResult};
use name_matcher::run_service::dto::{CudaDiagnosticsDto, SystemInfoDto};
use serde_json::Value as JsonValue;
use std::path::PathBuf;
use sysinfo::System;

#[tauri::command]
pub fn system_info() -> AppResult<SystemInfoDto> {
    let mut sys = System::new_all();
    sys.refresh_all();

    let memory_total_mb = sys.total_memory() / 1_048_576;
    let memory_avail_mb = sys.available_memory() / 1_048_576;

    let cpu_brand = sys
        .cpus()
        .first()
        .map(|c| c.brand().to_string())
        .unwrap_or_else(|| "unknown".into());

    let cpu_cores_logical = sys.cpus().len() as u32;
    let cpu_cores_physical = System::physical_core_count().unwrap_or(0) as u32;

    let gpu = cuda_diagnostics_inner();
    let gpu_available = gpu.gpu_feature_compiled && gpu.device_count > 0;

    let rayon_threads = std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(0);

    Ok(SystemInfoDto {
        os: format!(
            "{} {}",
            std::env::consts::OS,
            System::os_version().unwrap_or_else(|| "unknown".into())
        ),
        cpu_brand,
        cpu_cores_logical,
        cpu_cores_physical,
        memory_total_mb,
        memory_avail_mb,
        gpu_available,
        gpu_devices: gpu.devices.clone(),
        rayon_threads,
        app_version: env!("CARGO_PKG_VERSION").into(),
    })
}

#[tauri::command]
pub fn cuda_diagnostics() -> CudaDiagnosticsDto {
    cuda_diagnostics_inner()
}

fn cuda_diagnostics_inner() -> CudaDiagnosticsDto {
    // Delegate to the engine's runtime probe — this respects the `gpu`
    // feature forwarding from src-tauri to name_matcher and avoids a second
    // cudarc dependency.
    name_matcher::run_service::probe_cuda()
}

fn config_path() -> AppResult<PathBuf> {
    // Use a stable per-user config dir. We avoid tauri-plugin-store for the
    // raw load/save here so the format is interoperable with the legacy egui
    // binary and the CLI.
    let home = dirs_home();
    Ok(home.join(".name_matcher").join("config.json"))
}

fn dirs_home() -> PathBuf {
    if let Ok(p) = std::env::var("APPDATA") {
        PathBuf::from(p).join("name_matcher")
    } else if let Ok(p) = std::env::var("HOME") {
        PathBuf::from(p)
    } else {
        PathBuf::from(".")
    }
}

#[tauri::command]
pub fn load_config() -> AppResult<JsonValue> {
    let path = config_path()?;
    if !path.exists() {
        return Ok(JsonValue::Object(serde_json::Map::new()));
    }
    let bytes = std::fs::read(&path)
        .map_err(|e| AppError::Io(format!("read {}: {e}", path.display())))?;
    let v: JsonValue = serde_json::from_slice(&bytes)
        .map_err(|e| AppError::Internal(format!("invalid config json: {e}")))?;
    Ok(v)
}

#[tauri::command]
pub fn save_config(config: JsonValue) -> AppResult<()> {
    let path = config_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| AppError::Io(format!("mkdir {}: {e}", parent.display())))?;
    }
    let bytes = serde_json::to_vec_pretty(&config)?;
    std::fs::write(&path, bytes)
        .map_err(|e| AppError::Io(format!("write {}: {e}", path.display())))?;
    Ok(())
}
