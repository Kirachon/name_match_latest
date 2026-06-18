use super::dto::*;
use super::{probe_cuda, resolve_rayon_threads};
use crate::matching::{ComputeBackend, GpuConfig};

pub fn validate_run_config(config: &RunConfigDto) -> anyhow::Result<()> {
    if config.options.ultra_performance {
        if config.options.rayon_threads.is_some() {
            anyhow::bail!(
                "options.rayon_threads: Disable Ultra Performance to set rayon threads manually"
            );
        }
        if config.options.pool_size.is_some() {
            anyhow::bail!(
                "options.pool_size: Disable Ultra Performance to set pool size manually"
            );
        }
    }
    if matches!(config.gpu.mode, ComputeModeDto::Cpu) {
        if config.gpu.use_hash_join {
            anyhow::bail!("gpu.use_hash_join: GPU hash-join requires GPU mode");
        }
        if config.gpu.use_direct_prefilter {
            anyhow::bail!("gpu.use_direct_prefilter: GPU prefilter requires GPU mode");
        }
        if config.gpu.use_levenshtein_full_scoring {
            anyhow::bail!(
                "gpu.use_levenshtein_full_scoring: GPU full Levenshtein scoring requires GPU mode"
            );
        }
        if !matches!(config.gpu.fuzzy_gate_mode, GpuFuzzyGateModeDto::Off) {
            anyhow::bail!("gpu.fuzzy_gate_mode: GPU fuzzy gate requires GPU mode");
        }
    }
    if let Some(cascade) = &config.cascade
        && cascade.enabled
        && cascade.levels.is_empty()
    {
        anyhow::bail!("cascade.levels: Pick at least one cascade level");
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub(super) struct ResolvedPerformancePlan {
    pub(super) scoped_rayon_threads: Option<usize>,
    pub(super) backend: ComputeBackend,
    pub(super) gpu_config: Option<GpuConfig>,
    pub(super) direct_prefilter: bool,
    pub(super) levenshtein_full_scoring: bool,
    pub(super) dynamic_tuning: bool,
    pub(super) fuzzy_gate_mode: crate::matching::GpuFuzzyGateMode,
    pub(super) fuzzy_metrics: bool,
    pub(super) fuzzy_force: bool,
    pub(super) vram_budget_mb: Option<u64>,
    pub(super) cuda_available: bool,
    pub(super) cuda_error: Option<String>,
    pub(super) reason: String,
}

impl ResolvedPerformancePlan {
    pub(super) fn log_summary(
        &self,
        config: &RunConfigDto,
        left_rows: usize,
        right_rows: usize,
    ) -> String {
        format!(
            "Ultra resolved: ultra={} requested_gpu={:?} backend={:?} cuda_available={} cuda_error={} rows=({}, {}) algorithm={:?} cascade={} rayon_threads={} direct_prefilter={} levenshtein_full_scoring={} fuzzy_gate={} fuzzy_metrics={} fuzzy_force={} dynamic_tuning={} vram_budget_mb={} reason={}",
            config.options.ultra_performance,
            config.gpu.mode,
            self.backend,
            self.cuda_available,
            self.cuda_error.as_deref().unwrap_or("none"),
            left_rows,
            right_rows,
            config.algorithm,
            config.cascade.as_ref().map(|c| c.enabled).unwrap_or(false),
            self.scoped_rayon_threads
                .map(|n| n.to_string())
                .unwrap_or_else(|| "global/default".to_string()),
            self.direct_prefilter,
            self.levenshtein_full_scoring,
            self.fuzzy_gate_mode.as_str(),
            self.fuzzy_metrics,
            self.fuzzy_force,
            self.dynamic_tuning,
            self.vram_budget_mb
                .map(|mb| mb.to_string())
                .unwrap_or_else(|| "auto".to_string()),
            self.reason
        )
    }
}

pub(super) fn resolve_performance_plan(
    config: &RunConfigDto,
    left_rows: usize,
    right_rows: usize,
) -> anyhow::Result<ResolvedPerformancePlan> {
    validate_run_config(config)?;
    let cuda = probe_cuda();
    let cuda_available = cuda.gpu_feature_compiled && cuda.device_count > 0;
    let cuda_error = cuda.error.clone();
    let algorithm = config.algorithm.to_engine();
    let scoped_rayon_threads = if config.options.ultra_performance {
        ultra_rayon_threads().or_else(|| resolve_rayon_threads(&config.options, algorithm))
    } else {
        resolve_rayon_threads(&config.options, algorithm)
    };

    let requested_backend = match config.gpu.mode {
        ComputeModeDto::Cpu => ComputeBackend::Cpu,
        ComputeModeDto::Auto | ComputeModeDto::ForceGpu => ComputeBackend::Gpu,
    };
    let cascade_enabled = config.cascade.as_ref().map(|c| c.enabled).unwrap_or(false);
    let workload_rows = left_rows.max(right_rows);
    let gpu_eligible = gpu_eligible_for_config(config);
    // Release benchmarks on the target RTX 4050 laptop show the current fuzzy GPU
    // path loses to optimized CPU at <= 2.2k rows/side. Keep Auto conservative
    // until the workload is large enough to amortize CUDA setup and CPU post-work.
    let large_enough_for_gpu = workload_rows >= 50_000;

    let mut reason_parts = Vec::new();
    let backend = if matches!(config.gpu.mode, ComputeModeDto::Cpu) {
        reason_parts.push("gpu-mode-cpu".to_string());
        ComputeBackend::Cpu
    } else if matches!(config.gpu.mode, ComputeModeDto::ForceGpu) {
        if !cuda_available {
            anyhow::bail!(
                "Force GPU requested but CUDA is unavailable: {}",
                cuda_error.as_deref().unwrap_or("no CUDA device detected")
            );
        }
        if !gpu_eligible {
            anyhow::bail!(
                "Force GPU requested but {:?} is not GPU eligible",
                config.algorithm
            );
        }
        reason_parts.push("force-gpu".to_string());
        ComputeBackend::Gpu
    } else if config.options.ultra_performance {
        if !cuda_available {
            reason_parts.push("auto-cpu-cuda-unavailable".to_string());
            ComputeBackend::Cpu
        } else if !gpu_eligible {
            reason_parts.push("auto-cpu-algorithm-not-gpu-eligible".to_string());
            ComputeBackend::Cpu
        } else if !large_enough_for_gpu {
            reason_parts.push("auto-cpu-small-workload".to_string());
            ComputeBackend::Cpu
        } else {
            reason_parts.push("auto-gpu-ultra-workload".to_string());
            ComputeBackend::Gpu
        }
    } else {
        reason_parts.push("ultra-off-preserve-requested-backend".to_string());
        requested_backend
    };

    let using_gpu = matches!(backend, ComputeBackend::Gpu);
    let requested_gate_mode = match config.gpu.fuzzy_gate_mode {
        GpuFuzzyGateModeDto::Off => crate::matching::GpuFuzzyGateMode::Off,
        GpuFuzzyGateModeDto::Shadow => crate::matching::GpuFuzzyGateMode::Shadow,
        GpuFuzzyGateModeDto::GateOnly => crate::matching::GpuFuzzyGateMode::GateOnly,
    };
    let fuzzy_path = matches!(
        config.algorithm,
        AlgorithmDto::Fuzzy | AlgorithmDto::FuzzyNoMiddle
    ) || cascade_uses_fuzzy(config);
    let ultra_gpu_fuzzy = config.options.ultra_performance && using_gpu && fuzzy_path;
    let vram_budget_mb = if using_gpu {
        config.gpu.vram_budget_mb.map(|mb| mb as u64).or_else(|| {
            match (cuda.vram_total_mb, cuda.vram_free_mb) {
                (Some(total), Some(free)) if total > 0 && free > 0 => {
                    calculate_vram_budget(total, free)
                }
                _ => None,
            }
        })
    } else {
        None
    };

    let fuzzy_gate_mode = if using_gpu {
        requested_gate_mode
    } else {
        crate::matching::GpuFuzzyGateMode::Off
    };
    if cascade_enabled {
        reason_parts.push("cascade-aware".to_string());
    }

    Ok(ResolvedPerformancePlan {
        scoped_rayon_threads,
        backend,
        gpu_config: if using_gpu {
            Some(GpuConfig {
                device_id: Some(0),
                mem_budget_mb: vram_budget_mb.unwrap_or(0),
            })
        } else {
            None
        },
        direct_prefilter: using_gpu && (config.gpu.use_direct_prefilter || ultra_gpu_fuzzy),
        levenshtein_full_scoring: using_gpu && config.gpu.use_levenshtein_full_scoring,
        dynamic_tuning: using_gpu
            && (config.gpu.dynamic_tuning || config.options.ultra_performance),
        fuzzy_gate_mode,
        fuzzy_metrics: using_gpu
            && (ultra_gpu_fuzzy
                || !matches!(fuzzy_gate_mode, crate::matching::GpuFuzzyGateMode::Off)),
        fuzzy_force: using_gpu
            && (ultra_gpu_fuzzy
                || !matches!(fuzzy_gate_mode, crate::matching::GpuFuzzyGateMode::Off)),
        vram_budget_mb,
        cuda_available,
        cuda_error,
        reason: reason_parts.join(","),
    })
}

pub(super) fn apply_performance_plan(plan: &ResolvedPerformancePlan) {
    crate::matching::set_gpu_fuzzy_direct_prep(plan.direct_prefilter);
    crate::matching::set_gpu_levenshtein_full_scoring(plan.levenshtein_full_scoring);
    crate::matching::set_dynamic_gpu_tuning(plan.dynamic_tuning);
    crate::matching::set_gpu_fuzzy_gate_mode(plan.fuzzy_gate_mode);
    crate::matching::set_gpu_fuzzy_disable(false);
    crate::matching::set_gpu_fuzzy_metrics(plan.fuzzy_metrics);
    crate::matching::set_gpu_fuzzy_force(plan.fuzzy_force);
    crate::matching::set_gpu_fuzzy_prepass_budget_mb(plan.vram_budget_mb.unwrap_or(0));
}

fn ultra_rayon_threads() -> Option<usize> {
    std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(1).max(1))
        .ok()
}

#[cfg(feature = "gpu")]
fn calculate_vram_budget(total_mb: u64, free_mb: u64) -> Option<u64> {
    Some(crate::matching::gpu_config::calculate_gpu_memory_budget(
        total_mb, free_mb, true,
    ))
}

#[cfg(not(feature = "gpu"))]
fn calculate_vram_budget(_total_mb: u64, _free_mb: u64) -> Option<u64> {
    None
}

fn gpu_eligible_for_config(config: &RunConfigDto) -> bool {
    if cascade_uses_fuzzy(config) {
        return true;
    }
    matches!(
        config.algorithm,
        AlgorithmDto::Fuzzy
            | AlgorithmDto::FuzzyNoMiddle
            | AlgorithmDto::HouseholdGpu
            | AlgorithmDto::HouseholdGpuOpt6
    )
}

fn cascade_uses_fuzzy(config: &RunConfigDto) -> bool {
    config
        .cascade
        .as_ref()
        .filter(|cascade| cascade.enabled)
        .map(|cascade| {
            cascade.levels.is_empty()
                || cascade
                    .levels
                    .iter()
                    .any(|level| matches!(*level, 10 | 11))
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_config() -> RunConfigDto {
        RunConfigDto {
            source: TableSelectionDto {
                source_kind: DataSourceKindDto::Database,
                session_id: "source".to_string(),
                table: "source_table".to_string(),
                column_mapping: None,
                file: None,
                row_count: Some(1_000),
            },
            target: TableSelectionDto {
                source_kind: DataSourceKindDto::Database,
                session_id: "target".to_string(),
                table: "target_table".to_string(),
                column_mapping: None,
                file: None,
                row_count: Some(1_000),
            },
            algorithm: AlgorithmDto::Fuzzy,
            options: MatchOptionsDto::default(),
            gpu: GpuOptionsDto {
                mode: ComputeModeDto::Auto,
                dynamic_tuning: true,
                ..GpuOptionsDto::default()
            },
            streaming: StreamingOptionsDto::default(),
            export: ExportOptionsDto::default(),
            review_band: None,
            cascade: None,
        }
    }

    #[test]
    fn rust_validation_rejects_ultra_manual_overrides() {
        let mut config = base_config();
        config.options.ultra_performance = true;
        config.options.rayon_threads = Some(4);
        config.options.pool_size = Some(8);

        let err = validate_run_config(&config).unwrap_err().to_string();
        assert!(err.contains("options.rayon_threads"));
    }

    #[test]
    fn rust_validation_rejects_cpu_mode_gpu_helpers() {
        let mut config = base_config();
        config.gpu.mode = ComputeModeDto::Cpu;
        config.gpu.use_direct_prefilter = true;

        let err = validate_run_config(&config).unwrap_err().to_string();
        assert!(err.contains("gpu.use_direct_prefilter"));
    }

    #[test]
    fn ultra_respects_explicit_cpu_mode() {
        let mut config = base_config();
        config.options.ultra_performance = true;
        config.options.auto_optimize = false;
        config.gpu.mode = ComputeModeDto::Cpu;

        let plan = resolve_performance_plan(&config, 10_000, 10_000).unwrap();
        assert!(matches!(plan.backend, ComputeBackend::Cpu));
        assert!(!plan.direct_prefilter);
        assert!(!plan.dynamic_tuning);
        assert!(matches!(
            plan.fuzzy_gate_mode,
            crate::matching::GpuFuzzyGateMode::Off
        ));
    }

    #[test]
    fn ultra_auto_keeps_small_workloads_on_cpu() {
        let mut config = base_config();
        config.options.ultra_performance = true;
        config.options.auto_optimize = false;
        config.gpu.mode = ComputeModeDto::Auto;

        let plan = resolve_performance_plan(&config, 20, 20).unwrap();
        assert!(matches!(plan.backend, ComputeBackend::Cpu));
        assert!(plan.reason.contains("small-workload") || !plan.cuda_available);
    }

    #[test]
    fn applying_cpu_plan_resets_gate_and_budget_after_gpu_plan() {
        let gpu_plan = ResolvedPerformancePlan {
            scoped_rayon_threads: Some(4),
            backend: ComputeBackend::Gpu,
            gpu_config: Some(GpuConfig {
                device_id: Some(0),
                mem_budget_mb: 1024,
            }),
            direct_prefilter: true,
            levenshtein_full_scoring: true,
            dynamic_tuning: true,
            fuzzy_gate_mode: crate::matching::GpuFuzzyGateMode::Shadow,
            fuzzy_metrics: true,
            fuzzy_force: true,
            vram_budget_mb: Some(1024),
            cuda_available: true,
            cuda_error: None,
            reason: "test-gpu".to_string(),
        };
        apply_performance_plan(&gpu_plan);
        assert!(matches!(
            crate::matching::current_gpu_fuzzy_gate_mode(),
            crate::matching::GpuFuzzyGateMode::Shadow
        ));
        assert_eq!(crate::matching::gpu_fuzzy_prep_budget_mb(), 1024);

        let cpu_plan = ResolvedPerformancePlan {
            backend: ComputeBackend::Cpu,
            gpu_config: None,
            direct_prefilter: false,
            levenshtein_full_scoring: false,
            dynamic_tuning: false,
            fuzzy_gate_mode: crate::matching::GpuFuzzyGateMode::Off,
            fuzzy_metrics: false,
            fuzzy_force: false,
            vram_budget_mb: None,
            reason: "test-cpu".to_string(),
            ..gpu_plan
        };
        apply_performance_plan(&cpu_plan);
        assert!(matches!(
            crate::matching::current_gpu_fuzzy_gate_mode(),
            crate::matching::GpuFuzzyGateMode::Off
        ));
        assert_eq!(crate::matching::gpu_fuzzy_prep_budget_mb(), 0);
    }
}
