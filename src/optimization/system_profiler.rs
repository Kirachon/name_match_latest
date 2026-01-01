use anyhow::Result;

#[derive(Debug, Clone)]
pub struct CpuInfo {
    pub cores: usize,
    pub threads: usize,
    pub arch: String,
}

#[derive(Debug, Clone)]
pub struct RamInfo {
    pub total_mb: u64,
    pub available_mb: u64,
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub device_name: String,
    pub vram_total_mb: u64,
    pub vram_free_mb: u64,
    pub compute_major: i32,
    pub compute_minor: i32,
}

#[derive(Debug, Clone)]
pub struct OsInfo {
    pub name: String,
    pub arch: String,
}

#[derive(Debug, Clone)]
pub struct SystemProfile {
    pub cpu: CpuInfo,
    pub ram: RamInfo,
    pub gpu: Option<GpuInfo>,
    pub os: OsInfo,
}

impl core::fmt::Display for SystemProfile {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if let Some(gpu) = &self.gpu {
            write!(
                f,
                "CPU: {} cores ({} threads) | RAM: {} MB total, {} MB avail | GPU: {} ({} MB VRAM, cc {}.{}) | OS: {} ({})",
                self.cpu.cores,
                self.cpu.threads,
                self.ram.total_mb,
                self.ram.available_mb,
                gpu.device_name,
                gpu.vram_total_mb,
                gpu.compute_major,
                gpu.compute_minor,
                self.os.name,
                self.os.arch
            )
        } else {
            write!(
                f,
                "CPU: {} cores ({} threads) | RAM: {} MB total, {} MB avail | GPU: none | OS: {} ({})",
                self.cpu.cores,
                self.cpu.threads,
                self.ram.total_mb,
                self.ram.available_mb,
                self.os.name,
                self.os.arch
            )
        }
    }
}

impl SystemProfile {
    pub fn detect() -> Result<Self> {
        // CPU
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let cores = threads; // Best-effort (logical cores)
        let arch = std::env::consts::ARCH.to_string();
        let cpu = CpuInfo {
            cores,
            threads,
            arch: arch.clone(),
        };

        // RAM
        let mem = crate::metrics::memory_stats_mb();
        let ram = RamInfo {
            total_mb: mem.total_mb,
            available_mb: mem.avail_mb,
        };

        // OS
        let os = OsInfo {
            name: std::env::consts::OS.to_string(),
            arch,
        };

        // GPU (optional)
        let gpu = {
            #[cfg(feature = "gpu")]
            {
                // Use CUDA driver API via cudarc
                use cudarc::driver::{CudaContext, sys as cu};
                unsafe {
                    let init_rc = cu::cuInit(0);
                    if init_rc == cu::CUresult::CUDA_SUCCESS {
                        if let Ok(ctx) = CudaContext::new(0) {
                            // Device name via cuDeviceGetName
                            let mut free_b: usize = 0;
                            let mut total_b: usize = 0;
                            let rc = cu::cuMemGetInfo_v2(
                                &mut free_b as *mut usize,
                                &mut total_b as *mut usize,
                            );
                            // Compute capability via cuDeviceComputeCapability
                            let mut dev = 0;
                            let _ = cu::cuCtxGetDevice(&mut dev as *mut _);
                            let mut maj: i32 = 0;
                            let mut min: i32 = 0;
                            let _ = cu::cuDeviceComputeCapability(
                                &mut maj as *mut _,
                                &mut min as *mut _,
                                dev,
                            );
                            // Device name (best-effort)
                            let mut name_buf = [0i8; 128];
                            let _ = cu::cuDeviceGetName(name_buf.as_mut_ptr(), 128, dev);
                            let cstr = unsafe { core::ffi::CStr::from_ptr(name_buf.as_ptr()) };
                            let device_name = cstr.to_string_lossy().into_owned();
                            let vram_total_mb = (total_b as u64) / 1024 / 1024;
                            let vram_free_mb = (free_b as u64) / 1024 / 1024;
                            drop(ctx);
                            Some(GpuInfo {
                                device_name,
                                vram_total_mb,
                                vram_free_mb,
                                compute_major: maj,
                                compute_minor: min,
                            })
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
            }
            #[cfg(not(feature = "gpu"))]
            {
                None
            }
        };

        Ok(SystemProfile { cpu, ram, gpu, os })
    }
}
