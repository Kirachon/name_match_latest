//! Standalone CUDA smoke binary.
//!
//! Probes the CUDA runtime via `cudarc` and prints device count, VRAM, and
//! any failure reason. Exits 0 when at least one device is reachable, 1
//! otherwise. Useful for:
//!
//! * Confirming a release build was actually compiled with the `gpu` feature.
//! * Verifying that the linked CUDA runtime DLLs are reachable at runtime
//!   (Tauri release inspection).
//! * Quick smoke after deploying to a new machine without depending on a
//!   MySQL fixture (`gpu_audit` does that, but it's heavier).
//!
//! Build:
//! ```
//! cargo build --release --features gpu --bin cuda_probe
//! ```

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "cuda_probe was built without the `gpu` feature.\n\
         Rebuild with: cargo build --release --features gpu --bin cuda_probe"
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() {
    use cudarc::driver::CudaContext;

    println!("cuda_probe — CUDA runtime smoke");
    println!("==============================");

    // Probe device 0 first.
    match CudaContext::new(0) {
        Ok(ctx) => {
            println!("Device 0: initialised");
            // Free / total VRAM via the runtime API.
            #[allow(unused_unsafe)]
            unsafe {
                let mut free: usize = 0;
                let mut total: usize = 0;
                let r = cudarc::driver::sys::cuMemGetInfo_v2(
                    &mut free as *mut usize,
                    &mut total as *mut usize,
                );
                if r == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    let free_mb = (free / (1024 * 1024)) as u64;
                    let total_mb = (total / (1024 * 1024)) as u64;
                    println!("VRAM: {} MB free / {} MB total", free_mb, total_mb);
                } else {
                    println!("VRAM info: cuMemGetInfo returned non-success: {:?}", r);
                }
            }
            drop(ctx);
            println!("Result: OK");
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("Device 0 init failed: {e}");
            eprintln!("Result: FAIL");
            std::process::exit(1);
        }
    }
}
