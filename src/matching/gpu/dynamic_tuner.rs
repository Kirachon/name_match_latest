use std::sync::{OnceLock, RwLock};
use std::thread;
use std::time::Duration;

use anyhow::{Result, anyhow};

use crate::matching::gpu_config::calculate_gpu_memory_budget;

// Internal shared state for dynamic GPU tuning
#[derive(Debug, Clone)]
pub struct DynamicState {
    pub enabled: bool,
    pub tile_target_pairs: usize,
    pub gpu_streams: u32,
    pub min_tile_pairs: usize,
    pub max_tile_pairs: usize,
    pub last_budget_mb: u64,
    pub last_total_mb: u64,
    pub last_free_mb: u64,
}

static STATE: OnceLock<RwLock<DynamicState>> = OnceLock::new();
use std::thread::JoinHandle;

// Track the background thread so we can shut it down cleanly
static HANDLE: OnceLock<RwLock<Option<JoinHandle<()>>>> = OnceLock::new();
#[inline]
fn handle_lock() -> &'static RwLock<Option<JoinHandle<()>>> {
    HANDLE.get_or_init(|| RwLock::new(None))
}

fn init_state() -> RwLock<DynamicState> {
    RwLock::new(DynamicState {
        enabled: false,
        // Conservative initial defaults; will be adapted after first query
        tile_target_pairs: 32_000,
        gpu_streams: 1,
        min_tile_pairs: 32_000,
        max_tile_pairs: 1_000_000, // soft upper bound; actual capped by VRAM budget
        last_budget_mb: 0,
        last_total_mb: 0,
        last_free_mb: 0,
    })
}

#[inline]
pub fn get_state() -> &'static RwLock<DynamicState> {
    STATE.get_or_init(init_state)
}

/// Ensure the dynamic tuner background thread is started once per process when enabled.
/// Safe to call multiple times. If `enable` is false, this will also request shutdown.
pub fn ensure_started(enable: bool) {
    let lock = get_state();
    {
        // Acquire write lock defensively to avoid panics on poisoning
        let guard = match lock.write() {
            Ok(g) => g,
            Err(_) => {
                log::error!(
                    "[GPU-TUNE] Dynamic tuner state lock poisoned; skipping ensure_started"
                );
                return;
            }
        };
        let mut s = guard;
        if !enable {
            s.enabled = false;
            // request shutdown and join the thread if running
            if let Ok(mut h) = handle_lock().write() {
                if let Some(handle) = h.take() {
                    let _ = handle.join();
                }
            }
            return;
        }
        if s.enabled {
            // already running
            return;
        }
        s.enabled = true;
    }
    // Spawn background sampler thread and retain handle for graceful shutdown
    let handle = thread::spawn(|| {
        // Lazily create a CUDA context; use device 0 for now
        #[cfg(feature = "gpu")]
        let ctx = match cudarc::driver::CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                log::warn!(
                    "[GPU-TUNE] Failed to init CUDA context: {}. Dynamic tuning disabled.",
                    e
                );
                let mut s = get_state().write().unwrap();
                s.enabled = false;
                return;
            }
        };

        loop {
            // Sampling interval
            thread::sleep(Duration::from_secs(2));
            let mut s = match get_state().write() {
                Ok(guard) => guard,
                Err(_) => break,
            };
            if !s.enabled {
                break;
            }

            #[cfg(feature = "gpu")]
            {
                let (total_mb, free_mb) = super::super::cuda_mem_info_mb(&ctx);
                if total_mb == 0 {
                    continue;
                }
                // Conservative budget (75% of free VRAM)
                let budget_mb = calculate_gpu_memory_budget(total_mb, free_mb, false);
                s.last_budget_mb = budget_mb;
                s.last_total_mb = total_mb;
                s.last_free_mb = free_mb;
                // Approx bytes per pair used by fuzzy GPU pipeline
                let approx_bpp: usize = 256;
                let mut target =
                    ((budget_mb as usize * 1024 * 1024) / approx_bpp).max(s.min_tile_pairs);
                // If lots of headroom, gently increase; if low headroom, reduce aggressively
                let free_frac = if total_mb > 0 {
                    free_mb as f64 / total_mb as f64
                } else {
                    0.0
                };
                if free_frac > 0.50 {
                    target = (target as f64 * 1.25).round() as usize; // grow 25%
                } else if free_frac < 0.15 {
                    target = (target / 2).max(s.min_tile_pairs); // back off fast near OOM
                }
                // Clamp
                if target > s.max_tile_pairs {
                    target = s.max_tile_pairs;
                }
                if target != s.tile_target_pairs {
                    log::info!(
                        "[GPU-TUNE] Adjusting tile target: {} -> {} pairs (free={}MB/{}, budget={}MB)",
                        s.tile_target_pairs,
                        target,
                        free_mb,
                        total_mb,
                        budget_mb
                    );
                    s.tile_target_pairs = target;
                }
                // Streams heuristic: enable 2 streams when enough VRAM headroom
                let new_streams = if free_frac > 0.30 { 2 } else { 1 };
                if new_streams != s.gpu_streams {
                    log::info!(
                        "[GPU-TUNE] Adjusting CUDA streams: {} -> {}",
                        s.gpu_streams,
                        new_streams
                    );
                    s.gpu_streams = new_streams;
                }
            }
        }
        // ctx dropped here (on thread exit) to release CUDA resources
    });
    if let Ok(mut h) = handle_lock().write() {
        *h = Some(handle);
    }
}

/// Explicitly request shutdown of the tuner thread and wait for it to exit.
/// Safe to call multiple times.
pub fn stop() -> Result<()> {
    {
        let mut s = get_state()
            .write()
            .map_err(|_| anyhow!("dynamic tuner poisoned"))?;
        s.enabled = false;
    }
    if let Ok(mut h) = handle_lock().write() {
        if let Some(handle) = h.take() {
            let _ = handle.join();
        }
    }
    Ok(())
}

#[inline]
pub fn get_current_tile_size() -> usize {
    match get_state().read() {
        Ok(s) => s.tile_target_pairs,
        Err(_) => {
            log::error!("[GPU-TUNE] Dynamic tuner state lock poisoned; using default tile size");
            32_000
        }
    }
}

#[inline]
pub fn get_current_vram_free_pct() -> f32 {
    match get_state().read() {
        Ok(s) => {
            if s.last_total_mb == 0 {
                0.0
            } else {
                (s.last_free_mb as f32 / s.last_total_mb as f32) * 100.0
            }
        }
        Err(_) => {
            log::error!("[GPU-TUNE] Dynamic tuner state lock poisoned; using 0% VRAM free");
            0.0
        }
    }
}

#[inline]
pub fn get_current_streams() -> u32 {
    match get_state().read() {
        Ok(s) => s.gpu_streams,
        Err(_) => {
            log::error!("[GPU-TUNE] Dynamic tuner state lock poisoned; defaulting to 1 stream");
            1
        }
    }
}
