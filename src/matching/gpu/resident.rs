//! GPU-resident name string pools for L10/L11 fuzzy gate batches.
//!
//! Uploads normalized name bytes once per fuzzy run; each batch flush sends
//! only pair indices instead of re-copying string payloads.

use super::{FuzzyCache, MAX_STR};
use anyhow::Result;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use std::sync::Arc;
use std::time::Instant;

fn gpu_name_for_pool(cache: &FuzzyCache, use_no_mid: bool) -> &str {
    if use_no_mid {
        &cache.simple_full_no_mid
    } else {
        &cache.simple_full
    }
}

/// Device-resident concatenated name strings for one table side.
pub struct ResidentNamePool {
    pub d_bytes: CudaSlice<u8>,
    pub d_offsets: CudaSlice<i32>,
    pub d_lengths: CudaSlice<i32>,
    pub n_rows: usize,
    pub byte_len: usize,
}

impl ResidentNamePool {
    /// Host-side byte estimate before CUDA allocation.
    pub fn estimate_host_bytes(caches: &[FuzzyCache], use_no_mid: bool) -> usize {
        let strings: usize = caches
            .iter()
            .map(|c| gpu_name_for_pool(c, use_no_mid).as_bytes().len().min(MAX_STR))
            .sum();
        strings + caches.len() * 8
    }

    pub fn build(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        caches: &[FuzzyCache],
        use_no_mid: bool,
    ) -> Result<Self> {
        let n_rows = caches.len();
        let mut bytes: Vec<u8> = Vec::new();
        let mut offsets: Vec<i32> = Vec::with_capacity(n_rows);
        let mut lengths: Vec<i32> = Vec::with_capacity(n_rows);

        for cache in caches {
            let s = gpu_name_for_pool(cache, use_no_mid);
            let sb = s.as_bytes();
            let la = sb.len().min(MAX_STR);
            offsets.push(bytes.len() as i32);
            lengths.push(la as i32);
            if la > 0 {
                bytes.extend_from_slice(&sb[..la]);
            }
        }

        let byte_len = bytes.len();
        let d_bytes = if byte_len > 0 {
            stream.memcpy_stod(bytes.as_slice())?
        } else {
            stream.alloc_zeros::<u8>(1)?
        };
        let d_offsets = if n_rows > 0 {
            stream.memcpy_stod(offsets.as_slice())?
        } else {
            stream.alloc_zeros::<i32>(1)?
        };
        let d_lengths = if n_rows > 0 {
            stream.memcpy_stod(lengths.as_slice())?
        } else {
            stream.alloc_zeros::<i32>(1)?
        };
        stream.synchronize()?;

        Ok(Self {
            d_bytes,
            d_offsets,
            d_lengths,
            n_rows,
            byte_len,
        })
    }
}

/// Build both resident pools when enabled and within VRAM budget; otherwise None.
pub fn try_build_resident_pair(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    cache1: &[FuzzyCache],
    cache2: &[FuzzyCache],
    budget_mb: u64,
) -> Result<Option<(ResidentNamePool, ResidentNamePool, u128)>> {
    if !crate::matching::gpu_resident_tables_enabled() {
        return Ok(None);
    }

    let use_no_mid = super::gpu_no_mid_mode();
    let est1 = ResidentNamePool::estimate_host_bytes(cache1, use_no_mid);
    let est2 = ResidentNamePool::estimate_host_bytes(cache2, use_no_mid);
    let total_bytes = est1 + est2;
    let budget_bytes = (budget_mb as u128).saturating_mul(1024 * 1024);
    let threshold = budget_bytes.saturating_mul(30) / 100;
    if total_bytes as u128 > threshold {
        log::info!(
            "[GPU_RESIDENT] Skipping resident tables: est={} bytes exceeds 30% of budget ({} MB)",
            total_bytes,
            budget_mb
        );
        return Ok(None);
    }

    let upload_start = Instant::now();
    let pool1 = match ResidentNamePool::build(ctx, stream, cache1, use_no_mid) {
        Ok(p) => p,
        Err(e) => {
            log::warn!("[GPU_RESIDENT] pool1 upload failed ({}); using legacy staging", e);
            return Ok(None);
        }
    };
    let pool2 = match ResidentNamePool::build(ctx, stream, cache2, use_no_mid) {
        Ok(p) => p,
        Err(e) => {
            log::warn!("[GPU_RESIDENT] pool2 upload failed ({}); using legacy staging", e);
            return Ok(None);
        }
    };
    let upload_us = upload_start.elapsed().as_micros();

    log::info!(
        "[GPU_RESIDENT] Uploaded pools: t1={} rows ({} bytes), t2={} rows ({} bytes) in {} us",
        pool1.n_rows,
        pool1.byte_len,
        pool2.n_rows,
        pool2.byte_len,
        upload_us
    );

    Ok(Some((pool1, pool2, upload_us)))
}
