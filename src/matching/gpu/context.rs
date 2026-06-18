use anyhow::{Result, anyhow};
use cudarc::driver::CudaContext;
use cudarc::nvrtc::compile_ptx;

use super::{FNV_KERNEL_SRC, LEV_KERNEL_SRC};

#[derive(Clone)]
pub struct GpuHashContext {
    pub(crate) ctx: std::sync::Arc<CudaContext>,
    pub(crate) module: std::sync::Arc<cudarc::driver::CudaModule>,
    pub(crate) func_hash: std::sync::Arc<cudarc::driver::CudaFunction>,
}

impl GpuHashContext {
    pub fn new() -> Result<Self> {
        let dev_id = 0usize;
        let ctx = CudaContext::new(dev_id).map_err(|e| anyhow!("CUDA init failed: {e}"))?;

        // Query device details (name, compute capability, driver version)
        let (gpu_name, cc_major, cc_minor, drv_major, drv_minor) = unsafe {
            use std::ffi::CStr;
            use std::os::raw::{c_char, c_int};
            let mut cu_dev: cudarc::driver::sys::CUdevice = 0;
            let mut driver_ver: c_int = 0;
            // Best-effort queries; ignore non-zero return codes for logging-only paths
            let _ = cudarc::driver::sys::cuDeviceGet(&mut cu_dev as *mut _, dev_id as c_int);
            let _ = cudarc::driver::sys::cuDriverGetVersion(&mut driver_ver as *mut _);
            let mut maj: c_int = 0;
            let mut min: c_int = 0;
            let _ = cudarc::driver::sys::cuDeviceComputeCapability(
                &mut maj as *mut _,
                &mut min as *mut _,
                cu_dev,
            );
            let mut name_buf: [c_char; 128] = [0; 128];
            let _ = cudarc::driver::sys::cuDeviceGetName(
                name_buf.as_mut_ptr(),
                name_buf.len() as c_int,
                cu_dev,
            );
            let name = CStr::from_ptr(name_buf.as_ptr())
                .to_string_lossy()
                .into_owned();
            let drv_major = driver_ver / 1000;
            let drv_minor = (driver_ver % 1000) / 10;
            (
                name,
                maj as i32,
                min as i32,
                drv_major as i32,
                drv_minor as i32,
            )
        };

        // Memory snapshot and activation logs
        let (tot_mb, free_mb) = super::super::cuda_mem_info_mb(&ctx);
        let used_mb = tot_mb.saturating_sub(free_mb);
        log::info!(
            "[GPU] CUDA context initialized: {name} (dev {dev}, compute {cc_major}.{cc_minor}) | Driver {drv_major}.{drv_minor} | Mem: used={used}/{tot} MB, free={free} MB",
            name = gpu_name,
            dev = dev_id,
            cc_major = cc_major,
            cc_minor = cc_minor,
            drv_major = drv_major,
            drv_minor = drv_minor,
            used = used_mb,
            tot = tot_mb,
            free = free_mb
        );
        log::info!(
            "[GPU] GPU acceleration ACTIVE: {name} (dev {dev}, compute {cc_major}.{cc_minor}) | Memory: {used}/{tot} MB",
            name = gpu_name,
            dev = dev_id,
            cc_major = cc_major,
            cc_minor = cc_minor,
            used = used_mb,
            tot = tot_mb
        );

        let ptx = compile_ptx(FNV_KERNEL_SRC).map_err(|e| anyhow!("NVRTC compile failed: {e}"))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| anyhow!("Load PTX failed: {e}"))?;
        let func_hash = module
            .load_function("fnv1a64_kernel")
            .map_err(|e| anyhow!("Get fnv1a64 func failed: {e}"))?;
        Ok(Self {
            ctx,
            module,
            func_hash: func_hash.into(),
        })
    }

    pub fn get() -> Result<Self> {
        use std::sync::OnceLock;
        static INSTANCE: OnceLock<GpuHashContext> = OnceLock::new();
        if let Some(c) = INSTANCE.get() {
            return Ok(c.clone());
        }
        let inst = Self::new()?;

        let _ = INSTANCE.set(inst.clone());
        Ok(inst)
    }

    pub fn mem_info_mb(&self) -> (u64, u64) {
        super::super::cuda_mem_info_mb(&self.ctx)
    }
}

// Cached GPU context for fuzzy metrics (Levenshtein/Jaro/Jaro-Winkler/Max3)
// Mirrors the GpuHashContext pattern to eliminate per-call CUDA init and NVRTC compilation.
#[derive(Clone)]
pub struct GpuFuzzyContext {
    pub(crate) ctx: std::sync::Arc<CudaContext>,
    pub(crate) module: std::sync::Arc<cudarc::driver::CudaModule>,
    pub(crate) func_lev: std::sync::Arc<cudarc::driver::CudaFunction>,
    pub(crate) func_jaro: std::sync::Arc<cudarc::driver::CudaFunction>,
    pub(crate) func_jw: std::sync::Arc<cudarc::driver::CudaFunction>,
    pub(crate) func_max3: std::sync::Arc<cudarc::driver::CudaFunction>,
    pub(crate) func_gate: std::sync::Arc<cudarc::driver::CudaFunction>,
    pub(crate) func_gate_resident: std::sync::Arc<cudarc::driver::CudaFunction>,
    // Two reusable streams: default + auxiliary for overlapping transfers/compute
    pub(crate) stream_default: std::sync::Arc<cudarc::driver::CudaStream>,
    pub(crate) stream_aux: std::sync::Arc<cudarc::driver::CudaStream>,
}

impl GpuFuzzyContext {
    pub fn new() -> Result<Self> {
        let dev_id = 0usize;
        let ctx = CudaContext::new(dev_id).map_err(|e| anyhow!("CUDA init failed: {e}"))?;

        // Compile and load fuzzy kernels once
        let ptx = compile_ptx(LEV_KERNEL_SRC).map_err(|e| anyhow!("NVRTC compile failed: {e}"))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| anyhow!("Load PTX failed: {e}"))?;
        let func_lev = module
            .load_function("lev_kernel")
            .map_err(|e| anyhow!("Get lev func failed: {e}"))?;
        let func_jaro = module
            .load_function("jaro_kernel")
            .map_err(|e| anyhow!("Get jaro func failed: {e}"))?;
        let func_jw = module
            .load_function("jw_kernel")
            .map_err(|e| anyhow!("Get jw func failed: {e}"))?;
        let func_max3 = module
            .load_function("max3_kernel")
            .map_err(|e| anyhow!("Get max3 func failed: {e}"))?;
        let func_gate = module
            .load_function("fuzzy_gate_kernel")
            .map_err(|e| anyhow!("Get fuzzy gate func failed: {e}"))?;
        let func_gate_resident = module
            .load_function("fuzzy_gate_kernel_resident")
            .map_err(|e| anyhow!("Get fuzzy gate resident func failed: {e}"))?;

        // Prepare reusable streams

        let stream_default = ctx.default_stream();
        let stream_aux = ctx
            .new_stream()
            .map_err(|e| anyhow!("CUDA stream create failed: {e}"))?;

        Ok(Self {
            ctx,
            module: module.into(),
            func_lev: func_lev.into(),
            func_jaro: func_jaro.into(),
            func_jw: func_jw.into(),
            func_max3: func_max3.into(),
            func_gate: func_gate.into(),
            func_gate_resident: func_gate_resident.into(),
            stream_default,
            stream_aux,
        })
    }

    pub fn get() -> Result<Self> {
        use std::sync::OnceLock;
        static INSTANCE: OnceLock<GpuFuzzyContext> = OnceLock::new();
        if let Some(c) = INSTANCE.get() {
            return Ok(c.clone());
        }
        let inst = Self::new()?;
        let _ = INSTANCE.set(inst.clone());
        Ok(inst)
    }

    pub fn mem_info_mb(&self) -> (u64, u64) {
        super::super::cuda_mem_info_mb(&self.ctx)
    }
}
