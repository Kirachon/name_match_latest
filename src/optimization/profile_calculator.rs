use crate::matching::{
    MAX_BATCH_SIZE, MIN_BATCH_SIZE, MatchingAlgorithm, StreamingConfig as StreamCfg,
};
use crate::optimization::system_profiler::SystemProfile;
use anyhow::{Result, anyhow};

/// Maximum batch size for aggressive mode (high-RAM systems)
/// Allows up to 5x the standard MAX_BATCH_SIZE for systems with ample memory
const AGGRESSIVE_MAX_BATCH_SIZE: i64 = MAX_BATCH_SIZE * 5; // 250,000

#[derive(Debug, Clone, Copy)]
pub struct InMemoryConfig {
    pub rayon_threads: usize,
    pub use_gpu_fuzzy_metrics: bool,
    pub use_gpu_hash_join: bool,
    pub gpu_mem_budget_mb: u64,
}

impl Default for InMemoryConfig {
    fn default() -> Self {
        Self {
            rayon_threads: 0,
            use_gpu_fuzzy_metrics: false,
            use_gpu_hash_join: false,
            gpu_mem_budget_mb: 0,
        }
    }
}

/// Calculate optimal StreamingConfig based on detected hardware and algorithm.
/// Uses MIN_BATCH_SIZE and MAX_BATCH_SIZE from matching module as base bounds.
/// Aggressive mode allows up to AGGRESSIVE_MAX_BATCH_SIZE for high-RAM systems.
pub fn calculate_streaming_config(
    profile: &SystemProfile,
    algo: MatchingAlgorithm,
    aggressive: bool,
) -> StreamCfg {
    let mut cfg = StreamCfg::default();

    // Heuristics for batch size and memory thresholds
    let avail = profile.ram.available_mb.max(256);
    let total = profile.ram.total_mb.max(1024);
    let target_frac = if aggressive { 0.85 } else { 0.75 };
    let target_batch_mem_mb = (avail as f64 * target_frac) as u64;
    // Convert MB to rows: (MB × 1024 KB/MB) / 4 KB/row = MB × 256 rows/MB
    // Assumes ~4 KB per row (consistent with validation logic at line 98)
    let max_for_mode = if aggressive {
        AGGRESSIVE_MAX_BATCH_SIZE
    } else {
        MAX_BATCH_SIZE * 2 // 100,000 for conservative mode with good RAM
    };
    let batch_rows_est = ((target_batch_mem_mb * 256) as i64).clamp(MIN_BATCH_SIZE, max_for_mode);
    cfg.batch_size = batch_rows_est;
    log::info!(
        "Batch size calculation: avail_ram={} MB, target_frac={:.2}, target_mem={} MB, raw_batch={}, final_batch={} (aggressive={})",
        avail,
        target_frac,
        target_batch_mem_mb,
        target_batch_mem_mb * 256,
        batch_rows_est,
        aggressive
    );

    cfg.memory_soft_min_mb = if aggressive {
        ((total as f64 * 0.12) as u64).max(512)
    } else {
        ((total as f64 * 0.18) as u64).max(1024)
    };

    // Streaming pipeline toggles
    cfg.async_prefetch = true;
    cfg.parallel_normalize = true;
    // Prefetch pool proportional to cores
    let cores = profile.cpu.cores.max(1) as u32;
    let divisor: u32 = if cores > 16 { 3 } else { 2 };
    let prefetch = (cores / divisor).max(1).min(16);
    log::info!(
        "Prefetch pool sizing: cores={}, divisor={}, computed={}, cap=16, final_prefetch={}",
        cores,
        divisor,
        cores / divisor,
        prefetch
    );
    cfg.prefetch_pool_size = prefetch;

    // Algorithm-specific GPU toggles
    if let Some(gpu) = &profile.gpu {
        // Enable dynamic tuning on sufficiently sized GPUs
        cfg.enable_dynamic_gpu_tuning = gpu.vram_total_mb >= 4096;
        // Fuzzy GPU metrics only for fuzzy algorithms
        cfg.use_gpu_fuzzy_metrics = matches!(
            algo,
            MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle
        );
        // Hash join only for deterministic algos 1 & 2
        cfg.use_gpu_hash_join = matches!(
            algo,
            MatchingAlgorithm::IdUuidYasIsMatchedInfnbd
                | MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd
        );
        // Derive probe batch from free VRAM
        let frac = if aggressive { 0.40 } else { 0.30 };
        let mut probe = (gpu.vram_free_mb as f64 * frac) as u64;
        probe = if aggressive {
            probe.clamp(512, 8192)
        } else {
            probe.clamp(256, 4096)
        };
        cfg.gpu_probe_batch_mb = probe;
        // Streams: 1-4
        cfg.gpu_streams = if aggressive { (cores.min(8)).max(2) } else { 1 };
        // Buffer pool and pinned host best-effort
        cfg.gpu_buffer_pool = true;
        cfg.gpu_use_pinned_host = aggressive;
        // Direct fuzzy normalization for A1/A2 optional (keep default false for back-compat)
    } else {
        // No GPU: ensure GPU flags are off
        cfg.use_gpu_hash_join = false;
        cfg.use_gpu_fuzzy_metrics = false;
        cfg.enable_dynamic_gpu_tuning = false;
        cfg.gpu_probe_batch_mb = 0;
        cfg.gpu_streams = 1;
        cfg.gpu_buffer_pool = true;
        cfg.gpu_use_pinned_host = false;
    }

    cfg
}

/// Calculate optimal in-memory settings
pub fn calculate_inmemory_config(
    profile: &SystemProfile,
    algo: MatchingAlgorithm,
    aggressive: bool,
) -> InMemoryConfig {
    let mut cfg = InMemoryConfig::default();
    let cores = profile.cpu.cores.max(1);
    cfg.rayon_threads = if aggressive {
        cores
    } else {
        cores.saturating_sub(if cores > 16 { 1 } else { 0 })
    };

    if let Some(gpu) = &profile.gpu {
        cfg.use_gpu_hash_join = matches!(
            algo,
            MatchingAlgorithm::IdUuidYasIsMatchedInfnbd
                | MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd
        );
        cfg.use_gpu_fuzzy_metrics = matches!(
            algo,
            MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle
        ) && gpu.vram_total_mb >= 2048;
        let frac = if aggressive { 0.90 } else { 0.75 };
        let mut budget = (gpu.vram_free_mb as f64 * frac) as u64;
        let max_cap = gpu
            .vram_total_mb
            .saturating_sub(if aggressive { 128 } else { 256 });
        budget = budget.clamp(if aggressive { 1024 } else { 512 }, max_cap);
        cfg.gpu_mem_budget_mb = budget;
    } else {
        cfg.use_gpu_hash_join = false;
        cfg.use_gpu_fuzzy_metrics = false;
        cfg.gpu_mem_budget_mb = 0;
    }
    cfg
}

/// Validate streaming config against system profile
pub fn validate_streaming_config(profile: &SystemProfile, cfg: &StreamCfg) -> Result<()> {
    // Basic RAM safety: assume ~4KB/row equivalent
    let est_mb = (cfg.batch_size.max(1) as u64).saturating_mul(4) / 1024;
    if est_mb > profile.ram.available_mb.saturating_add(128) {
        return Err(anyhow!(
            "Batch size likely exceeds available RAM: est {} MB > avail {} MB",
            est_mb,
            profile.ram.available_mb
        ));
    }
    // DB pool safety is enforced elsewhere; here we clamp prefetch pool
    if cfg.prefetch_pool_size > 64 {
        return Err(anyhow!(
            "prefetch_pool_size too high: {}",
            cfg.prefetch_pool_size
        ));
    }
    // GPU safety
    if let Some(gpu) = &profile.gpu {
        if cfg.gpu_probe_batch_mb > gpu.vram_free_mb.saturating_sub(64) {
            return Err(anyhow!(
                "GPU probe batch exceeds free VRAM: {} > {} (free)",
                cfg.gpu_probe_batch_mb,
                gpu.vram_free_mb
            ));
        }
    }
    Ok(())
}
