/// GPU configuration and adaptive resource allocation
///
/// This module provides hardware-adaptive GPU memory budgeting and configuration
/// to maximize GPU utilization across different hardware configurations.
use anyhow::Result;

/// Calculate optimal GPU memory budget based on available VRAM
///
/// This function queries the GPU's free VRAM and calculates an appropriate
/// memory budget that scales with available resources:
/// - Uses 75% of free VRAM (conservative to avoid OOM)
/// - Scales reserve memory with total VRAM (128 MB to 1 GB)
/// - Clamps to safe range (256 MB minimum, total - reserve maximum)
///
/// # Arguments
/// * `total_mb` - Total GPU VRAM in MB
/// * `free_mb` - Currently free GPU VRAM in MB
/// * `aggressive` - If true, uses 85% of free VRAM instead of 75%
///
/// # Returns
/// Recommended memory budget in MB
pub fn calculate_gpu_memory_budget(total_mb: u64, free_mb: u64, aggressive: bool) -> u64 {
    // Adaptive reserve based on total VRAM
    let reserve_mb = match total_mb {
        t if t >= 16384 => 1024, // ≥16 GB: reserve 1 GB
        t if t >= 8192 => 512,   // 8-16 GB: reserve 512 MB
        t if t >= 4096 => 256,   // 4-8 GB: reserve 256 MB
        _ => 128,                // <4 GB: reserve 128 MB
    };

    // Use 75% (normal) or 85% (aggressive) of free VRAM
    let usage_fraction = if aggressive { 0.85 } else { 0.75 };
    let usable_mb = free_mb.saturating_sub(reserve_mb);
    let budget = ((usable_mb as f64) * usage_fraction) as u64;

    // Clamp to reasonable range
    // Minimum: 256 MB (enough for small tiles)
    // Maximum: total - reserve (prevent OOM)
    budget.clamp(256, total_mb.saturating_sub(reserve_mb))
}

/// GPU properties queried from CUDA device
#[derive(Debug, Clone)]
pub struct GpuProperties {
    pub sm_count: i32,
    pub compute_major: i32,
    pub compute_minor: i32,
    pub max_threads_per_block: i32,
    pub max_shared_memory_per_block: usize,
}

/// Query GPU compute capability and properties
///
/// # Arguments
/// * `device_id` - CUDA device ID (usually 0)
///
/// # Returns
/// GPU properties including SM count, compute capability, etc.
#[cfg(feature = "gpu")]
pub fn query_gpu_properties(device_id: i32) -> Result<GpuProperties> {
    use cudarc::driver::sys as cu;

    unsafe {
        let mut sm_count: i32 = 0;
        let mut compute_major: i32 = 0;
        let mut compute_minor: i32 = 0;
        let mut max_threads: i32 = 0;
        let mut max_shared_mem: i32 = 0;

        cu::cuDeviceGetAttribute(
            &mut sm_count,
            cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            device_id,
        );

        cu::cuDeviceGetAttribute(
            &mut compute_major,
            cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            device_id,
        );

        cu::cuDeviceGetAttribute(
            &mut compute_minor,
            cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            device_id,
        );

        cu::cuDeviceGetAttribute(
            &mut max_threads,
            cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            device_id,
        );

        cu::cuDeviceGetAttribute(
            &mut max_shared_mem,
            cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
            device_id,
        );

        Ok(GpuProperties {
            sm_count,
            compute_major,
            compute_minor,
            max_threads_per_block: max_threads,
            max_shared_memory_per_block: max_shared_mem as usize,
        })
    }
}

/// Kernel type for adaptive block size calculation
#[derive(Debug, Clone, Copy)]
pub enum KernelType {
    Hash,        // Low register usage
    Levenshtein, // High register usage
    Jaro,        // Moderate register usage
    JaroWinkler, // Moderate register usage
}

/// Calculate optimal block size based on GPU properties and kernel type
///
/// Different kernels have different register usage patterns:
/// - Hash kernels: Low register usage, can use larger blocks
/// - Levenshtein: High register usage, needs smaller blocks
/// - Jaro/JaroWinkler: Moderate register usage
///
/// # Arguments
/// * `props` - GPU properties from `query_gpu_properties`
/// * `kernel_type` - Type of kernel being launched
///
/// # Returns
/// Recommended block size (number of threads per block)
pub fn calculate_optimal_block_size(props: &GpuProperties, kernel_type: KernelType) -> u32 {
    match kernel_type {
        KernelType::Hash => {
            // Hash kernels have low register usage, can use larger blocks
            if props.compute_major >= 8 {
                512 // Ampere+ (RTX 30xx, 40xx)
            } else if props.compute_major >= 7 {
                384 // Turing/Volta (RTX 20xx, V100)
            } else {
                256 // Pascal and older (GTX 10xx)
            }
        }
        KernelType::Levenshtein => {
            // High register usage - keep blocks smaller
            if props.compute_major >= 8 {
                128 // Ampere+
            } else {
                64 // Older architectures
            }
        }
        KernelType::Jaro | KernelType::JaroWinkler => {
            // Moderate register usage
            if props.compute_major >= 8 {
                256 // Ampere+
            } else {
                128 // Older architectures
            }
        }
    }
}

/// Calculate soft cap for tile size based on GPU parallelism
///
/// This prevents creating tiles that are too large for the GPU to process
/// efficiently. The cap is based on the number of SMs and block size.
///
/// # Arguments
/// * `props` - GPU properties from `query_gpu_properties`
/// * `block_size` - Block size being used for the kernel
///
/// # Returns
/// Recommended maximum tile size (soft cap)
pub fn calculate_tile_soft_cap(props: &GpuProperties, block_size: u32) -> usize {
    // Heuristic: aim for 16 blocks per SM for good occupancy
    let blocks_per_sm = 16;
    let max_concurrent_blocks = props.sm_count as usize * blocks_per_sm;
    let soft_cap = max_concurrent_blocks * block_size as usize;

    // Clamp to reasonable range
    // Minimum: 10K elements (for small GPUs)
    // Maximum: 10M elements (for very large GPUs)
    soft_cap.clamp(10_000, 10_000_000)
}

/// Check if an error is a CUDA out-of-memory error
///
/// # Arguments
/// * `e` - Error to check
///
/// # Returns
/// true if the error is an OOM error, false otherwise
pub fn is_cuda_oom(e: &anyhow::Error) -> bool {
    let s = e.to_string().to_ascii_lowercase();
    s.contains("out of memory") || s.contains("cuda_error_out_of_memory")
}

/// Execute a GPU call with automatic CPU fallback on CUDA OOM.
/// Returns Ok(result) when either GPU succeeds or CPU fallback is used for OOM.
/// Propagates non-OOM errors unchanged.
pub fn with_oom_cpu_fallback<T, F, C>(
    gpu_call: F,
    cpu_fallback: C,
    context: &str,
) -> anyhow::Result<T>
where
    F: FnOnce() -> anyhow::Result<T>,
    C: FnOnce() -> T,
{
    match gpu_call() {
        Ok(v) => Ok(v),
        Err(e) => {
            if is_cuda_oom(&e) {
                log::warn!("[GPU-OOM] {} — falling back to CPU: {}", context, e);
                Ok(cpu_fallback())
            } else {
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_gpu_memory_budget() {
        // Test high-VRAM GPU (24 GB)
        let budget = calculate_gpu_memory_budget(24576, 20000, false);
        assert!(
            budget >= 14000,
            "High-VRAM GPU should have budget ≥14 GB, got {} MB",
            budget
        );
        assert!(
            budget <= 18000,
            "Budget should not exceed 18 GB, got {} MB",
            budget
        );

        // Test mid-VRAM GPU (8 GB)
        let budget = calculate_gpu_memory_budget(8192, 7000, false);
        assert!(
            budget >= 4500,
            "Mid-VRAM GPU should have budget ≥4.5 GB, got {} MB",
            budget
        );
        assert!(
            budget <= 6000,
            "Budget should not exceed 6 GB, got {} MB",
            budget
        );

        // Test low-VRAM GPU (4 GB)
        let budget = calculate_gpu_memory_budget(4096, 3500, false);
        assert!(
            budget >= 2000,
            "Low-VRAM GPU should have budget ≥2 GB, got {} MB",
            budget
        );
        assert!(
            budget <= 3000,
            "Budget should not exceed 3 GB, got {} MB",
            budget
        );

        // Test aggressive mode
        let normal = calculate_gpu_memory_budget(8192, 7000, false);
        let aggressive = calculate_gpu_memory_budget(8192, 7000, true);
        assert!(aggressive > normal, "Aggressive mode should use more VRAM");
    }

    #[test]
    fn test_calculate_optimal_block_size() {
        let props_ampere = GpuProperties {
            sm_count: 80,
            compute_major: 8,
            compute_minor: 6,
            max_threads_per_block: 1024,
            max_shared_memory_per_block: 49152,
        };

        let props_pascal = GpuProperties {
            sm_count: 20,
            compute_major: 6,
            compute_minor: 1,
            max_threads_per_block: 1024,
            max_shared_memory_per_block: 49152,
        };

        // Hash kernels should use larger blocks on newer GPUs
        assert_eq!(
            calculate_optimal_block_size(&props_ampere, KernelType::Hash),
            512
        );
        assert_eq!(
            calculate_optimal_block_size(&props_pascal, KernelType::Hash),
            256
        );

        // Levenshtein should use smaller blocks due to high register usage
        assert_eq!(
            calculate_optimal_block_size(&props_ampere, KernelType::Levenshtein),
            128
        );
        assert_eq!(
            calculate_optimal_block_size(&props_pascal, KernelType::Levenshtein),
            64
        );
    }

    #[test]
    fn test_is_cuda_oom() {
        let oom_err = anyhow::anyhow!("CUDA_ERROR_OUT_OF_MEMORY: out of memory");
        assert!(is_cuda_oom(&oom_err));

        let other_err = anyhow::anyhow!("Some other error");
        assert!(!is_cuda_oom(&other_err));
    }

    #[test]
    fn test_with_oom_cpu_fallback() {
        // GPU closure returns OOM -> should execute CPU fallback
        let gpu_fail = || -> anyhow::Result<i32> {
            Err(anyhow::anyhow!("CUDA_ERROR_OUT_OF_MEMORY: out of memory"))
        };
        let cpu_ok = || 123;
        let v = with_oom_cpu_fallback(gpu_fail, cpu_ok, "unit-test").unwrap();
        assert_eq!(v, 123);

        // Non-OOM error should propagate
        let gpu_other = || -> anyhow::Result<i32> { Err(anyhow::anyhow!("kernel launch failure")) };
        let cpu_x = || 0;
        let err = with_oom_cpu_fallback(gpu_other, cpu_x, "unit-test").unwrap_err();
        assert!(err.to_string().contains("kernel launch failure"));
    }
}
