use super::*;

pub fn match_fuzzy_gpu<F>(
    t1: &[Person],
    t2: &[Person],
    opts: MatchOptions,
    on_progress: &F,
) -> Result<Vec<MatchPair>>
where
    F: Fn(ProgressUpdate) + Sync,
{
    let wall_start = std::time::Instant::now();
    let allow_swap = opts.allow_birthdate_swap;
    let gate_mode = current_gpu_fuzzy_gate_mode();
    log::info!(
        "[GPU_FUZZY] allow_birthdate_swap flag = {} gate_mode={} (env NAME_MATCHER_ALLOW_BIRTHDATE_SWAP={:?})",
        allow_swap,
        gate_mode.as_str(),
        std::env::var("NAME_MATCHER_ALLOW_BIRTHDATE_SWAP").ok()
    );

    // 1) Normalize on CPU (reuse existing)
    let normalization_start = std::time::Instant::now();
    let n1: Vec<NormalizedPerson> = t1.par_iter().map(normalize_person).collect();
    let n2: Vec<NormalizedPerson> = t2.par_iter().map(normalize_person).collect();
    let normalization_time_us = normalization_start.elapsed().as_micros();
    if n1.is_empty() || n2.is_empty() {
        record_last_gpu_fuzzy_stats(&GpuFuzzyStats {
            input_rows_left: n1.len() as u64,
            input_rows_right: n2.len() as u64,
            normalization_time_us,
            total_wall_time_us: wall_start.elapsed().as_micros(),
            ..GpuFuzzyStats::default()
        });
        return Ok(vec![]);
    }
    let mut stats = GpuFuzzyStats {
        input_rows_left: n1.len() as u64,
        input_rows_right: n2.len() as u64,
        normalization_time_us,
        ..GpuFuzzyStats::default()
    };

    // 2) Build the same blocking index used by the CPU-equivalent path so
    // the GPU scorer operates on the exact same candidate set.
    use chrono::Datelike;
    use std::collections::{HashMap, HashSet};

    #[derive(Hash, Eq, PartialEq)]
    enum BKey {
        Specific(u16, u8, u8, [u8; 4]), // (birth year, first init, last init, last soundex)
        Year(u16),
    }

    let blocking_start = std::time::Instant::now();
    let mut block: HashMap<BKey, Vec<usize>> = HashMap::new();
    for (j, p) in n2.iter().enumerate() {
        let (Some(d), Some(fn_str), Some(ln_str)) = (
            p.birthdate.as_ref(),
            p.first_name.as_deref(),
            p.last_name.as_deref(),
        ) else {
            continue;
        };
        let year = d.year() as u16;
        let fi = fn_str
            .bytes()
            .find(|c| c.is_ascii_alphabetic())
            .unwrap_or(b'?')
            .to_ascii_uppercase();
        let li = ln_str
            .bytes()
            .find(|c| c.is_ascii_alphabetic())
            .unwrap_or(b'?')
            .to_ascii_uppercase();
        let sx = soundex4_ascii(ln_str);
        block
            .entry(BKey::Specific(year, fi, li, sx))
            .or_default()
            .push(j);
        block.entry(BKey::Year(year)).or_default().push(j);
    }
    stats.candidate_blocking_time_us = blocking_start.elapsed().as_micros();

    // 3) Prepare CUDA context & streams
    let dev_id = opts.gpu.and_then(|g| g.device_id).unwrap_or(0);
    // Build per-person caches once (used by GPU tiling and CPU post-processing)
    let cache_build_start = std::time::Instant::now();
    let cache1: Vec<FuzzyCache> = t1.par_iter().map(build_cache_from_person).collect();
    let cache2: Vec<FuzzyCache> = t2.par_iter().map(build_cache_from_person).collect();
    stats.cache_build_time_us = cache_build_start.elapsed().as_micros();

    // Reuse cached CUDA context, compiled module, kernels, and streams
    let fctx = GpuFuzzyContext::get()?;
    let ctx_arc = &fctx.ctx;
    let stream = &fctx.stream_default;
    let stream2 = &fctx.stream_aux;
    let func = &*fctx.func_lev;
    let func_jaro = &*fctx.func_jaro;
    let func_jw = &*fctx.func_jw;
    let func_max3 = &*fctx.func_max3;
    let func_gate = &*fctx.func_gate;
    let func_gate_resident = &*fctx.func_gate_resident;

    // Report GPU init and memory info
    let (gpu_total_mb, gpu_free_mb_init) = cuda_mem_info_mb(&*ctx_arc);
    let mem0 = memory_stats_mb();
    on_progress(ProgressUpdate {
        processed: 0,
        total: n1.len(),
        percent: 0.0,
        eta_secs: 0,
        mem_used_mb: mem0.used_mb,
        mem_avail_mb: mem0.avail_mb,
        stage: "gpu_init",
        batch_size_current: None,
        gpu_total_mb,
        gpu_free_mb: gpu_free_mb_init,
        gpu_active: true,
    });

    // [GPU_OPT5] Memory query caching to reduce overhead
    use std::time::Instant;
    let mut last_mem_query = Instant::now();
    let mut cached_gpu_free_mb = gpu_free_mb_init;
    let mut mem_query_count = 0usize;
    let mut tile_count = 0usize;

    if dynamic_gpu_tuning_enabled() {
        super::dynamic_tuner::ensure_started(true);
    }

    // 4) Tile candidates to respect memory budget - use adaptive budget if not explicitly set
    let gate_mode_for_resident = current_gpu_fuzzy_gate_mode();
    let want_resident = resident_tables_requested(gate_mode_for_resident);

    let base_budget_mb = match opts.gpu.and_then(|g| {
        if g.mem_budget_mb > 0 {
            Some(g.mem_budget_mb)
        } else {
            None
        }
    }) {
        Some(explicit_budget) => explicit_budget,
        None => {
            let budget = super::gpu_config::calculate_gpu_memory_budget(
                gpu_total_mb,
                gpu_free_mb_init,
                false,
            );
            log::info!(
                "[GPU] Auto-calculated memory budget: {} MB (75% of {} MB free VRAM)",
                budget,
                gpu_free_mb_init
            );
            budget
        }
    };

    let resident_holder = if want_resident {
        resident::try_build_resident_pair(ctx_arc, stream, &cache1, &cache2, base_budget_mb)?
    } else {
        None
    };
    let resident_reserve_mb = resident_holder
        .as_ref()
        .map(|(pool1, pool2, _)| resident::actual_resident_pair_mb(pool1, pool2))
        .unwrap_or(0);
    let mem_budget_mb = if resident_reserve_mb > 0 {
        base_budget_mb.saturating_sub(resident_reserve_mb).max(256)
    } else {
        base_budget_mb
    };
    if resident_reserve_mb > 0 {
        log::info!(
            "[GPU] Batch memory budget after resident upload: {} MB (base={} MB, resident={} MB)",
            mem_budget_mb,
            base_budget_mb,
            resident_reserve_mb
        );
    }
    // Rough bytes per pair: two strings up to 64 bytes + offsets/len + output ~ 256 bytes
    let approx_bpp: usize = 256;

    // [GPU_OPT] Increased minimum tile size from 1,024 to 32,000 pairs
    // Larger tiles reduce CPU-GPU handoff overhead and improve GPU utilization
    // OOM backoff logic (lines 2291-2337) provides safety net for low-VRAM GPUs

    // [GPU_OPT3] Calculate tile soft cap based on GPU parallelism capacity
    let gpu_props = super::gpu_config::query_gpu_properties(0).unwrap_or_else(|_| {
        super::gpu_config::GpuProperties {
            compute_major: 7,
            compute_minor: 0,
            sm_count: 30,
            max_threads_per_block: 1024,
            max_shared_memory_per_block: 49152,
        }
    });
    let block_size = super::gpu_config::calculate_optimal_block_size(
        &gpu_props,
        super::gpu_config::KernelType::Levenshtein,
    );
    let soft_cap = super::gpu_config::calculate_tile_soft_cap(&gpu_props, block_size);

    let memory_based = ((mem_budget_mb as usize * 1024 * 1024) / approx_bpp).max(32_000);
    let mut tile_max = memory_based.min(soft_cap);
    stats.tile_size_max = tile_max as u64;

    log::info!(
        "[GPU_OPT3] Tile sizing: memory_based={} soft_cap={} final={} (GPU: {}.{}, {} SMs, block_size={})",
        memory_based,
        soft_cap,
        tile_max,
        gpu_props.compute_major,
        gpu_props.compute_minor,
        gpu_props.sm_count,
        block_size
    );

    if dynamic_gpu_tuning_enabled() {
        let dyn_tile = super::dynamic_tuner::get_current_tile_size();
        if dyn_tile != tile_max {
            log::info!(
                "[GPU-TUNE] Overriding tile size: {} -> {} pairs",
                tile_max,
                dyn_tile
            );
        }
        tile_max = dyn_tile.max(1);
        stats.tile_size_max = tile_max as u64;
    }

    let resident_arg = resident_holder
        .as_ref()
        .map(|(pool1, pool2, _upload_us)| (pool1, pool2));
    if let Some((pool1, pool2, upload_us)) = resident_holder.as_ref() {
        stats.resident_tables_enabled = true;
        stats.resident_table_bytes = (pool1.byte_len + pool2.byte_len) as u64;
        stats.resident_table_upload_us = *upload_us;
    }

    let mut results: Vec<MatchPair> = Vec::new();
    let total: usize = n1.len();
    // Cross-outer GPU batch (store pairs) to improve utilization by batching multiple outer records per launch
    let mut batch_pairs: Vec<(usize, usize)> = Vec::with_capacity(tile_max.max(1));
    // Reusable accumulator — retains grow-only pinned buffers across tiles
    let mut reusable_acc = crate::matching::gpu::batch::GpuBatchAccumulator::new(tile_max.max(1));

    // Track seen pairs to deduplicate when multiple lookup variants hit the same pair.
    let mut seen_pairs: HashSet<(usize, usize)> = HashSet::new();
    let mut total_candidates = 0usize;
    let mut swap_candidates = 0usize;
    let mut candidates_per_source: Vec<usize> = Vec::with_capacity(n1.len());
    let candidate_scan_start = std::time::Instant::now();

    for (i, p1) in n1.iter().enumerate() {
        let mut source_candidates = 0usize;
        let (Some(d), Some(fn_str), Some(ln_str)) = (
            p1.birthdate.as_ref(),
            p1.first_name.as_deref(),
            p1.last_name.as_deref(),
        ) else {
            candidates_per_source.push(0);
            continue;
        };
        let year = d.year() as u16;
        let fi = fn_str
            .bytes()
            .find(|c| c.is_ascii_alphabetic())
            .unwrap_or(b'?')
            .to_ascii_uppercase();
        let li = ln_str
            .bytes()
            .find(|c| c.is_ascii_alphabetic())
            .unwrap_or(b'?')
            .to_ascii_uppercase();
        let sx = soundex4_ascii(ln_str);

        let mut seen_inner: HashSet<usize> = HashSet::new();
        let mut cand_lists: Vec<&[usize]> = Vec::new();
        if let Some(v) = block.get(&BKey::Specific(year, fi, li, sx)) {
            cand_lists.push(v.as_slice());
        }
        if cand_lists.is_empty() {
            if let Some(v) = block.get(&BKey::Specific(year, b'?', li, sx)) {
                cand_lists.push(v.as_slice());
            }
        }
        if cand_lists.is_empty() {
            let mut sx2 = sx;
            sx2[2] = b'0';
            sx2[3] = b'0';
            if let Some(v) = block.get(&BKey::Specific(year, fi, li, sx2)) {
                cand_lists.push(v.as_slice());
            }
        }
        if cand_lists.is_empty() {
            if let Some(v) = block.get(&BKey::Year(year)) {
                cand_lists.push(v.as_slice());
            }
        }

        for cands_vec in cand_lists {
            for &j_idx in cands_vec {
                if !seen_inner.insert(j_idx) {
                    continue;
                }
                if !seen_pairs.insert((i, j_idx)) {
                    continue;
                }
                total_candidates += 1;
                source_candidates += 1;
                stats.candidate_pairs_seen += 1;
                if let (Some(d1), Some(d2)) = (p1.birthdate, n2[j_idx].birthdate) {
                    if d1 != d2 {
                        swap_candidates += 1;
                    }
                }
                let s1 = fuzzy_cache_name(&cache1[i]);
                let s2 = fuzzy_cache_name(&cache2[j_idx]);
                if s1.trim().is_empty() || s2.trim().is_empty() {
                    stats.pre_gpu_skipped_empty_name += 1;
                    continue;
                }
                if s1 == s2 {
                    stats.direct_shortcuts += 1;
                    let bd_match = match (t1[i].birthdate, t2[j_idx].birthdate) {
                        (Some(b1), Some(b2)) => {
                            b1 == b2
                                || crate::matching::birthdate_matcher::birthdate_matches_naive(
                                    b1, b2, allow_swap,
                                )
                        }
                        _ => false,
                    };
                    if bd_match {
                        results.push(MatchPair {
                            person1: t1[i].clone(),
                            person2: t2[j_idx].clone(),
                            confidence: 1.0,
                            matched_fields: vec![
                                "fuzzy".into(),
                                "DIRECT MATCH".into(),
                                "birthdate".into(),
                            ],
                            is_matched_infnbd: false,
                            is_matched_infnmnbd: false,
                        });
                        stats.matches_emitted += 1;
                    }
                    continue;
                }
                batch_pairs.push((i, j_idx));
                if batch_pairs.len() >= tile_max {
                    // Flush in chunks with OOM backoff using prefix drains
                    let mut desired = tile_max.max(1);
                    while batch_pairs.len() >= desired {
                        reusable_acc.clear();
                        for &(oi, ij) in batch_pairs.iter().take(desired) {
                            reusable_acc.add_candidate(oi, ij);
                        }
                        let attempt = reusable_acc.flush_to_gpu(
                            &n1,
                            &n2,
                            &cache1,
                            &cache2,
                            t1,
                            t2,
                            ctx_arc,
                            stream,
                            stream2,
                            &func,
                            &func_jaro,
                            &func_jw,
                            &func_max3,
                            &func_gate,
                            &func_gate_resident,
                            resident_arg,
                            desired,
                            &mut results,
                            allow_swap,
                            &mut stats,
                        );
                        match attempt {
                            Ok(()) => {
                                batch_pairs.drain(0..desired);
                                tile_count += 1;
                                if stats.tile_size_min == 0 || desired as u64 <= stats.tile_size_min
                                {
                                    stats.tile_size_min = desired as u64;
                                }
                                stats.tile_size_max = stats.tile_size_max.max(desired as u64);

                                // [GPU_OPT5] Refresh GPU memory info periodically (every 100ms or 10 tiles)
                                if last_mem_query.elapsed() > std::time::Duration::from_millis(100)
                                    || tile_count % 10 == 0
                                {
                                    let (_tot_mb, free_mb) = cuda_mem_info_mb(&*ctx_arc);
                                    cached_gpu_free_mb = free_mb;
                                    last_mem_query = Instant::now();
                                    mem_query_count += 1;
                                }

                                let mem = memory_stats_mb();
                                let frac = ((i + 1) as f32 / total as f32).clamp(0.0, 1.0);
                                on_progress(ProgressUpdate {
                                    processed: i + 1,
                                    total,
                                    percent: frac * 100.0,
                                    eta_secs: 0,
                                    mem_used_mb: mem.used_mb,
                                    mem_avail_mb: mem.avail_mb,
                                    stage: "gpu_kernel",
                                    batch_size_current: Some(desired as i64),
                                    gpu_total_mb: gpu_total_mb,
                                    gpu_free_mb: cached_gpu_free_mb,
                                    gpu_active: true,
                                });
                                // continue while to check if more remains >= desired
                            }
                            Err(e) => {
                                if crate::matching::gpu_config::is_cuda_oom(&e) && desired > 512 {
                                    desired = (desired / 2).max(512);
                                    continue; // retry with smaller desired
                                } else {
                                    return Err(anyhow!(e));
                                }
                            }
                        }
                        if batch_pairs.len() < desired {
                            break;
                        }
                    }
                }
            }
        }
        candidates_per_source.push(source_candidates);
    }
    stats.candidate_scan_time_us = candidate_scan_start.elapsed().as_micros();
    // Final flush of any remaining batched pairs with OOM backoff
    let mut desired = tile_max.max(1);
    while !batch_pairs.is_empty() {
        let actual_len = desired.min(batch_pairs.len());
        reusable_acc.clear();
        for &(oi, ij) in batch_pairs.iter().take(actual_len) {
            reusable_acc.add_candidate(oi, ij);
        }
        let attempt = reusable_acc.flush_to_gpu(
            &n1,
            &n2,
            &cache1,
            &cache2,
            t1,
            t2,
            ctx_arc,
            stream,
            stream2,
            &func,
            &func_jaro,
            &func_jw,
            &func_max3,
            &func_gate,
            &func_gate_resident,
            resident_arg,
            actual_len,
            &mut results,
            allow_swap,
            &mut stats,
        );
        match attempt {
            Ok(()) => {
                batch_pairs.drain(0..actual_len);
                tile_count += 1;
                if stats.tile_size_min == 0 || actual_len as u64 <= stats.tile_size_min {
                    stats.tile_size_min = actual_len as u64;
                }
                stats.tile_size_max = stats.tile_size_max.max(actual_len as u64);

                // [GPU_OPT5] Refresh GPU memory info for final flush
                if last_mem_query.elapsed() > std::time::Duration::from_millis(100) {
                    let (_tot_mb, free_mb) = cuda_mem_info_mb(&*ctx_arc);
                    cached_gpu_free_mb = free_mb;
                    mem_query_count += 1;
                }

                let mem = memory_stats_mb();
                on_progress(ProgressUpdate {
                    processed: total,
                    total,
                    percent: 100.0,
                    eta_secs: 0,
                    mem_used_mb: mem.used_mb,
                    mem_avail_mb: mem.avail_mb,
                    stage: "gpu_kernel",
                    batch_size_current: Some(actual_len as i64),
                    gpu_total_mb: gpu_total_mb,
                    gpu_free_mb: cached_gpu_free_mb,
                    gpu_active: true,
                });
            }
            Err(e) => {
                if crate::matching::gpu_config::is_cuda_oom(&e) && desired > 512 {
                    desired = (desired / 2).max(512);
                    continue; // retry smaller
                } else {
                    return Err(anyhow!(e));
                }
            }
        }
    }

    // [GPU_OPT5] Log memory query caching efficiency
    log::info!(
        "[GPU_OPT5] Memory query caching: {} queries for {} tiles (avg: {:.1} tiles/query)",
        mem_query_count,
        tile_count,
        if mem_query_count > 0 {
            tile_count as f32 / mem_query_count as f32
        } else {
            0.0
        }
    );
    log::info!(
        "[GPU_FUZZY] Candidate generation: total_candidates={}, swap_candidates={}, results={}",
        total_candidates,
        swap_candidates,
        results.len()
    );
    sort_match_pairs_deterministically(&mut results);
    stats.matches_emitted = results.len() as u64;
    stats.total_wall_time_us = wall_start.elapsed().as_micros();
    stats.gpu_tile_count = tile_count as u64;
    stats.gpu_mem_query_count = mem_query_count as u64;
    let (p50, p95, max) = summarize_candidate_counts(&mut candidates_per_source);
    stats.candidates_per_source_p50 = p50;
    stats.candidates_per_source_p95 = p95;
    stats.candidates_per_source_max = max;
    let resident_suffix = if stats.resident_tables_enabled
        || stats.resident_table_bytes > 0
        || stats.resident_table_upload_us > 0
        || stats.batch_index_h2d_us > 0
    {
        format!(
            " resident_tables={} resident_table_bytes={} resident_table_upload_us={} batch_index_h2d_us={}",
            stats.resident_tables_enabled,
            stats.resident_table_bytes,
            stats.resident_table_upload_us,
            stats.batch_index_h2d_us
        )
    } else {
        String::new()
    };
    log::info!(
        "[GPU_FUZZY_STATS] mode={} rows=({},{}) candidates={} uploaded={} keep={} reject={} cpu_classified={} matches={} false_negatives={} kernels={} tiles={} mem_queries={} norm_us={} block_us={} cache_us={} scan_us={} cps_p50={} cps_p95={} cps_max={} h2d_us={} kernel_us={} h2d_kernel_ratio={:.3} d2h_us={} cpu_us={} total_us={}{}",
        gate_mode.as_str(),
        stats.input_rows_left,
        stats.input_rows_right,
        stats.candidate_pairs_seen,
        stats.pairs_uploaded,
        stats.gpu_gate_keep,
        stats.gpu_gate_reject,
        stats.cpu_classified,
        stats.matches_emitted,
        stats.shadow_false_negative_count,
        stats.gpu_kernel_launch_count,
        stats.gpu_tile_count,
        stats.gpu_mem_query_count,
        stats.normalization_time_us,
        stats.candidate_blocking_time_us,
        stats.cache_build_time_us,
        stats.candidate_scan_time_us,
        stats.candidates_per_source_p50,
        stats.candidates_per_source_p95,
        stats.candidates_per_source_max,
        stats.h2d_time_us,
        stats.kernel_time_us,
        if stats.kernel_time_us > 0 {
            stats.h2d_time_us as f64 / stats.kernel_time_us as f64
        } else {
            0.0
        },
        stats.d2h_time_us,
        stats.cpu_classification_time_us,
        stats.total_wall_time_us,
        resident_suffix
    );
    record_last_gpu_fuzzy_stats(&stats);

    Ok(results)
}

fn sort_match_pairs_deterministically(results: &mut [MatchPair]) {
    results.sort_by(|a, b| {
        (a.person1.id, a.person2.id)
            .cmp(&(b.person1.id, b.person2.id))
            .then_with(|| a.matched_fields.cmp(&b.matched_fields))
            .then_with(|| a.confidence.to_bits().cmp(&b.confidence.to_bits()))
    });
}

fn summarize_candidate_counts(counts: &mut [usize]) -> (u64, u64, u64) {
    if counts.is_empty() {
        return (0, 0, 0);
    }
    counts.sort_unstable();
    let last = counts.len().saturating_sub(1);
    (
        counts[(last * 50) / 100] as u64,
        counts[(last * 95) / 100] as u64,
        counts[last] as u64,
    )
}

pub fn match_fuzzy_no_mid_gpu<F>(
    t1: &[Person],
    t2: &[Person],
    opts: MatchOptions,
    on_progress: &F,
) -> Result<Vec<MatchPair>>
where
    F: Fn(ProgressUpdate) + Sync,
{
    // Force no-middle classification within the GPU batch pipeline to ensure true parity and avoid
    // the expensive reclassification/supplement steps.
    with_no_mid_classification(|| match_fuzzy_gpu(t1, t2, opts, on_progress))
}
