#[cfg(feature = "new_engine")]
use anyhow::Result;
#[cfg(feature = "new_engine")]
use sqlx::MySqlPool;

#[cfg(feature = "new_engine")]
use crate::matching::MatchPair;
#[cfg(feature = "new_engine")]
use crate::matching::{MatchingAlgorithm, ProgressUpdate, StreamControl, StreamingConfig};
#[cfg(feature = "new_engine")]
use crate::models::ColumnMapping;

/// Compute adaptive partition size for inner table loading based on available memory.
/// Targets 60-70% of available memory with safety bounds.
#[cfg(feature = "new_engine")]
fn compute_adaptive_inner_partition_size(avail_mb: u64, soft_min_mb: u64) -> i64 {
    // Estimate bytes per Person record:
    // - Base fields: ~120 bytes (i64 + 6 Option<String/NaiveDate> + small overhead)
    // - HashMap extra_fields: ~100 bytes average (varies widely)
    // - Vec/String allocations: ~80 bytes overhead
    // Total estimate: ~300 bytes per record
    const BYTES_PER_PERSON: u64 = 300;

    // Target 65% of available memory, but leave at least soft_min_mb free
    let partition_size = if avail_mb > soft_min_mb {
        let target_mb = ((avail_mb - soft_min_mb) as f64 * 0.65) as u64;
        let target_bytes = target_mb * 1024 * 1024;
        let estimated_records = target_bytes / BYTES_PER_PERSON;
        // Apply bounds: minimum 10,000 records, maximum 500,000 records
        estimated_records.max(10_000).min(500_000) as i64
    } else {
        // Very low memory - use minimum partition size
        10_000
    };

    log::info!(
        "[ADAPTIVE-PARTITION] Computed partition size: {} records (avail: {}MB, soft_min: {}MB)",
        partition_size,
        avail_mb,
        soft_min_mb
    );

    partition_size
}

/// Database-backed streaming facades for the new engine.
///
/// Notes:
/// - To preserve GPU acceleration, performance, and exact parity, these
///   facades currently delegate to the legacy streaming implementations.
/// - They provide a stable API surface under `engine::db_pipeline` so we can
///   incrementally replace internals with the new trait-based engine without
///   changing call sites.
/// - All functions are compiled only when the `new_engine` feature is enabled.
#[cfg(feature = "new_engine")]
pub mod db_pipeline {
    use super::*;

    /// Single-database streaming (auto-picks inner/outer by row count) using the trait-based StreamEngine.
    pub async fn stream_new_engine_single<F>(
        pool: &MySqlPool,
        table1: &str,
        table2: &str,
        algo: MatchingAlgorithm,
        mut on_match: F,
        cfg: StreamingConfig,
        on_progress: impl Fn(ProgressUpdate) + Sync,
        ctrl: Option<StreamControl>,
    ) -> Result<usize>
    where
        F: FnMut(&MatchPair) -> Result<()>,
    {
        use crate::db::schema::{fetch_person_rows_chunk, get_person_count};
        use crate::engine::file_checkpointer::FileCheckpointer;
        use crate::engine::legacy_adapters::{
            LegacyAdapterAlgo1, LegacyAdapterAlgo2, LegacyAdapterFuzzy, LegacyAdapterFuzzyNoMiddle,
        };
        use crate::engine::{FnPartitioner, StreamEngine};
        use crate::normalize::normalize_person;
        use std::time::Instant;

        // Preserve GPU accelerated single-DB path by delegating only when requested
        #[cfg(feature = "gpu")]
        if cfg.use_gpu_hash_join {
            return crate::matching::stream_match_csv(
                pool,
                table1,
                table2,
                algo,
                on_match,
                cfg,
                on_progress,
                ctrl,
            )
            .await;
        }

        let c1 = get_person_count(pool, table1).await?;
        let c2 = get_person_count(pool, table2).await?;
        let inner_is_t2 = c2 <= c1;
        let inner_table = if inner_is_t2 { table2 } else { table1 };
        let outer_table = if inner_is_t2 { table1 } else { table2 };
        let total_outer = if inner_is_t2 { c1 } else { c2 };

        // Load inner side using adaptive partitioned streaming
        let mut inner_rows: Vec<crate::models::Person> = Vec::new();
        let mut inner_off: i64 = 0;
        let mem_stats = crate::metrics::memory_stats_mb();
        let mut partition_size =
            compute_adaptive_inner_partition_size(mem_stats.avail_mb, cfg.memory_soft_min_mb);
        let mut partition_num = 1;

        on_progress(ProgressUpdate {
            processed: 0,
            total: total_outer as usize,
            percent: 0.0,
            eta_secs: 0,
            mem_used_mb: mem_stats.used_mb,
            mem_avail_mb: mem_stats.avail_mb,
            stage: "indexing_partition_1",
            batch_size_current: Some(partition_size),
            gpu_total_mb: 0,
            gpu_free_mb: 0,
            gpu_active: false,
        });

        loop {
            // Check memory pressure before loading next partition
            let current_mem = crate::metrics::memory_stats_mb();
            if current_mem.avail_mb < cfg.memory_soft_min_mb {
                partition_size = (partition_size / 2).max(10_000);
                log::warn!(
                    "[ADAPTIVE-PARTITION] Memory pressure detected ({}MB < {}MB), reducing partition size to {}",
                    current_mem.avail_mb,
                    cfg.memory_soft_min_mb,
                    partition_size
                );
            }

            let rows =
                fetch_person_rows_chunk(pool, inner_table, inner_off, partition_size).await?;
            if rows.is_empty() {
                break;
            }

            inner_off += rows.len() as i64;
            inner_rows.extend(rows);

            // Update progress for this partition
            partition_num += 1;
            on_progress(ProgressUpdate {
                processed: inner_rows.len(),
                total: total_outer as usize,
                percent: 0.0,
                eta_secs: 0,
                mem_used_mb: current_mem.used_mb,
                mem_avail_mb: current_mem.avail_mb,
                stage: "indexing_partition",
                batch_size_current: Some(partition_size),
                gpu_total_mb: 0,
                gpu_free_mb: 0,
                gpu_active: false,
            });
        }

        on_progress(ProgressUpdate {
            processed: 0,
            total: total_outer as usize,
            percent: 0.0,
            eta_secs: 0,
            mem_used_mb: crate::metrics::memory_stats_mb().used_mb,
            mem_avail_mb: crate::metrics::memory_stats_mb().avail_mb,
            stage: "indexing_done",
            batch_size_current: Some(partition_size),
            gpu_total_mb: 0,
            gpu_free_mb: 0,
            gpu_active: false,
        });

        // Build partitioner factory
        let key_algo = algo;
        let part_make = || {
            FnPartitioner::<crate::models::Person, _>(
                move |p| {
                    crate::matching::key_for_engine(key_algo, &normalize_person(p))
                        .unwrap_or_default()
                },
                std::marker::PhantomData,
            )
        };

        let ck_path = cfg
            .checkpoint_path
            .clone()
            .unwrap_or_else(|| "engine_ck.db".into());
        let job = format!("single:{}->{}", inner_table, outer_table);
        let mut total_written = 0usize;
        let batch = cfg.batch_size.max(10_000);
        match algo {
            MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => {
                let part_a = part_make();
                let part_b = part_make();
                let mut eng = StreamEngine::new(
                    LegacyAdapterAlgo1,
                    part_a,
                    part_b,
                    FileCheckpointer::new(ck_path.clone()),
                );
                let mut offset: i64 = 0;
                let start = Instant::now();
                while offset < total_outer {
                    if let Some(c) = &ctrl {
                        if c.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                            break;
                        }
                        while c.pause.load(std::sync::atomic::Ordering::Relaxed) {
                            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                        }
                    }
                    let rows = fetch_person_rows_chunk(pool, outer_table, offset, batch).await?;
                    if rows.is_empty() {
                        break;
                    }
                    offset += rows.len() as i64;
                    let processed = (offset as usize).min(total_outer as usize);
                    let wrote = eng.for_each(
                        &job,
                        rows.iter(),
                        inner_rows.iter(),
                        |a, b, score, _expl| {
                            let pair = if inner_is_t2 {
                                crate::matching::to_pair_public(a, b, algo)
                            } else {
                                crate::matching::to_pair_public(b, a, algo)
                            };
                            let mut pair = pair;
                            pair.confidence = (score as f32) / 100.0;
                            on_match(&pair)
                        },
                    )?;
                    total_written += wrote;
                    let frac = (processed as f32 / total_outer as f32).clamp(0.0, 1.0);
                    let eta = if frac > 0.0 {
                        (start.elapsed().as_secs_f32() * (1.0 - frac) / frac) as u64
                    } else {
                        0
                    };
                    let memx = crate::metrics::memory_stats_mb();
                    on_progress(ProgressUpdate {
                        processed,
                        total: total_outer as usize,
                        percent: frac * 100.0,
                        eta_secs: eta,
                        mem_used_mb: memx.used_mb,
                        mem_avail_mb: memx.avail_mb,
                        stage: "streaming",
                        batch_size_current: Some(batch),
                        gpu_total_mb: 0,
                        gpu_free_mb: 0,
                        gpu_active: false,
                    });
                    tokio::task::yield_now().await;
                }
            }
            MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => {
                let part_a = part_make();
                let part_b = part_make();
                let mut eng = StreamEngine::new(
                    LegacyAdapterAlgo2,
                    part_a,
                    part_b,
                    FileCheckpointer::new(ck_path.clone()),
                );
                let mut offset: i64 = 0;
                let start = Instant::now();
                while offset < total_outer {
                    if let Some(c) = &ctrl {
                        if c.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                            break;
                        }
                        while c.pause.load(std::sync::atomic::Ordering::Relaxed) {
                            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                        }
                    }
                    let rows = fetch_person_rows_chunk(pool, outer_table, offset, batch).await?;
                    if rows.is_empty() {
                        break;
                    }
                    offset += rows.len() as i64;
                    let processed = (offset as usize).min(total_outer as usize);
                    let wrote = eng.for_each(
                        &job,
                        rows.iter(),
                        inner_rows.iter(),
                        |a, b, score, _expl| {
                            let pair = if inner_is_t2 {
                                crate::matching::to_pair_public(a, b, algo)
                            } else {
                                crate::matching::to_pair_public(b, a, algo)
                            };
                            let mut pair = pair;
                            pair.confidence = (score as f32) / 100.0;
                            on_match(&pair)
                        },
                    )?;
                    total_written += wrote;
                    let frac = (processed as f32 / total_outer as f32).clamp(0.0, 1.0);
                    let eta = if frac > 0.0 {
                        (start.elapsed().as_secs_f32() * (1.0 - frac) / frac) as u64
                    } else {
                        0
                    };
                    let memx = crate::metrics::memory_stats_mb();
                    on_progress(ProgressUpdate {
                        processed,
                        total: total_outer as usize,
                        percent: frac * 100.0,
                        eta_secs: eta,
                        mem_used_mb: memx.used_mb,
                        mem_avail_mb: memx.avail_mb,
                        stage: "streaming",
                        batch_size_current: Some(batch),
                        gpu_total_mb: 0,
                        gpu_free_mb: 0,
                        gpu_active: false,
                    });
                    tokio::task::yield_now().await;
                }
            }
            MatchingAlgorithm::Fuzzy => {
                let part_a = part_make();
                let part_b = part_make();
                let mut eng = StreamEngine::new(
                    LegacyAdapterFuzzy,
                    part_a,
                    part_b,
                    FileCheckpointer::new(ck_path.clone()),
                );
                let mut offset: i64 = 0;
                let start = Instant::now();
                while offset < total_outer {
                    if let Some(c) = &ctrl {
                        if c.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                            break;
                        }
                        while c.pause.load(std::sync::atomic::Ordering::Relaxed) {
                            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                        }
                    }
                    let rows = fetch_person_rows_chunk(pool, outer_table, offset, batch).await?;
                    if rows.is_empty() {
                        break;
                    }
                    offset += rows.len() as i64;
                    let processed = (offset as usize).min(total_outer as usize);
                    let wrote = eng.for_each(
                        &job,
                        rows.iter(),
                        inner_rows.iter(),
                        |a, b, score, _expl| {
                            let pair = if inner_is_t2 {
                                crate::matching::to_pair_public(a, b, algo)
                            } else {
                                crate::matching::to_pair_public(b, a, algo)
                            };
                            let mut pair = pair;
                            pair.confidence = (score as f32) / 100.0;
                            on_match(&pair)
                        },
                    )?;
                    total_written += wrote;
                    let frac = (processed as f32 / total_outer as f32).clamp(0.0, 1.0);
                    let eta = if frac > 0.0 {
                        (start.elapsed().as_secs_f32() * (1.0 - frac) / frac) as u64
                    } else {
                        0
                    };
                    let memx = crate::metrics::memory_stats_mb();
                    on_progress(ProgressUpdate {
                        processed,
                        total: total_outer as usize,
                        percent: frac * 100.0,
                        eta_secs: eta,
                        mem_used_mb: memx.used_mb,
                        mem_avail_mb: memx.avail_mb,
                        stage: "streaming",
                        batch_size_current: Some(batch),
                        gpu_total_mb: 0,
                        gpu_free_mb: 0,
                        gpu_active: false,
                    });
                    tokio::task::yield_now().await;
                }
            }
            MatchingAlgorithm::FuzzyNoMiddle => {
                let part_a = part_make();
                let part_b = part_make();
                let mut eng = StreamEngine::new(
                    LegacyAdapterFuzzyNoMiddle,
                    part_a,
                    part_b,
                    FileCheckpointer::new(ck_path.clone()),
                );
                let mut offset: i64 = 0;
                let start = Instant::now();
                while offset < total_outer {
                    if let Some(c) = &ctrl {
                        if c.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                            break;
                        }
                        while c.pause.load(std::sync::atomic::Ordering::Relaxed) {
                            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                        }
                    }
                    let rows = fetch_person_rows_chunk(pool, outer_table, offset, batch).await?;
                    if rows.is_empty() {
                        break;
                    }
                    offset += rows.len() as i64;
                    let processed = (offset as usize).min(total_outer as usize);
                    let wrote = eng.for_each(
                        &job,
                        rows.iter(),
                        inner_rows.iter(),
                        |a, b, score, _expl| {
                            let pair = if inner_is_t2 {
                                crate::matching::to_pair_public(a, b, algo)
                            } else {
                                crate::matching::to_pair_public(b, a, algo)
                            };
                            let mut pair = pair;
                            pair.confidence = (score as f32) / 100.0;
                            on_match(&pair)
                        },
                    )?;
                    total_written += wrote;
                    let frac = (processed as f32 / total_outer as f32).clamp(0.0, 1.0);
                    let eta = if frac > 0.0 {
                        (start.elapsed().as_secs_f32() * (1.0 - frac) / frac) as u64
                    } else {
                        0
                    };
                    let memx = crate::metrics::memory_stats_mb();
                    on_progress(ProgressUpdate {
                        processed,
                        total: total_outer as usize,
                        percent: frac * 100.0,
                        eta_secs: eta,
                        mem_used_mb: memx.used_mb,
                        mem_avail_mb: memx.avail_mb,
                        stage: "streaming",
                        batch_size_current: Some(batch),
                        gpu_total_mb: 0,
                        gpu_free_mb: 0,
                        gpu_active: false,
                    });
                    tokio::task::yield_now().await;
                }
            }
            MatchingAlgorithm::LevenshteinWeighted => {
                // Not supported in engine streaming; GUI disables streaming for this algorithm.
            }
            MatchingAlgorithm::HouseholdGpu | MatchingAlgorithm::HouseholdGpuOpt6 => {
                let part_a = part_make();
                let part_b = part_make();
                let mut eng = StreamEngine::new(
                    LegacyAdapterAlgo1,
                    part_a,
                    part_b,
                    FileCheckpointer::new(ck_path.clone()),
                );
                let mut offset: i64 = 0;
                let start = Instant::now();
                while offset < total_outer {
                    if let Some(c) = &ctrl {
                        if c.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                            break;
                        }
                        while c.pause.load(std::sync::atomic::Ordering::Relaxed) {
                            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                        }
                    }
                    let rows = fetch_person_rows_chunk(pool, outer_table, offset, batch).await?;
                    if rows.is_empty() {
                        break;
                    }
                    offset += rows.len() as i64;
                    let processed = (offset as usize).min(total_outer as usize);
                    let wrote = eng.for_each(
                        &job,
                        rows.iter(),
                        inner_rows.iter(),
                        |a, b, score, _expl| {
                            let pair = if inner_is_t2 {
                                crate::matching::to_pair_public(a, b, algo)
                            } else {
                                crate::matching::to_pair_public(b, a, algo)
                            };
                            let mut pair = pair;
                            pair.confidence = (score as f32) / 100.0;
                            on_match(&pair)
                        },
                    )?;
                    total_written += wrote;
                    let frac = (processed as f32 / total_outer as f32).clamp(0.0, 1.0);
                    let eta = if frac > 0.0 {
                        (start.elapsed().as_secs_f32() * (1.0 - frac) / frac) as u64
                    } else {
                        0
                    };
                    let memx = crate::metrics::memory_stats_mb();
                    on_progress(ProgressUpdate {
                        processed,
                        total: total_outer as usize,
                        percent: frac * 100.0,
                        eta_secs: eta,
                        mem_used_mb: memx.used_mb,
                        mem_avail_mb: memx.avail_mb,
                        stage: "streaming",
                        batch_size_current: Some(batch),
                        gpu_total_mb: 0,
                        gpu_free_mb: 0,
                        gpu_active: false,
                    });
                    tokio::task::yield_now().await;
                }
            }
        }
        Ok(total_written)
    }

    /// Cross-database streaming (different pools for table1 and table2) using trait-based engine.
    pub async fn stream_new_engine_dual<F>(
        pool1: &MySqlPool,
        pool2: &MySqlPool,
        table1: &str,
        table2: &str,
        algo: MatchingAlgorithm,
        mut on_match: F,
        cfg: StreamingConfig,
        on_progress: impl Fn(ProgressUpdate) + Sync,
        ctrl: Option<StreamControl>,
    ) -> Result<usize>
    where
        F: FnMut(&MatchPair) -> Result<()>,
    {
        use crate::db::schema::{fetch_person_rows_chunk, get_person_count};
        use crate::engine::file_checkpointer::FileCheckpointer;
        use crate::engine::legacy_adapters::{
            LegacyAdapterAlgo1, LegacyAdapterAlgo2, LegacyAdapterFuzzy, LegacyAdapterFuzzyNoMiddle,
        };
        use crate::engine::{FnPartitioner, StreamEngine};
        use crate::normalize::normalize_person;
        use std::time::Instant;

        let c1 = get_person_count(pool1, table1).await?;
        let c2 = get_person_count(pool2, table2).await?;
        let inner_is_t2 = c2 <= c1;
        let inner_pool = if inner_is_t2 { pool2 } else { pool1 };
        let outer_pool = if inner_is_t2 { pool1 } else { pool2 };
        let inner_table = if inner_is_t2 { table2 } else { table1 };
        let outer_table = if inner_is_t2 { table1 } else { table2 };
        let total_outer = if inner_is_t2 { c1 } else { c2 };

        // Load inner side using adaptive partitioned streaming (dual-DB version)
        let mut inner_rows: Vec<crate::models::Person> = Vec::new();
        let mut inner_off: i64 = 0;
        let mem_stats = crate::metrics::memory_stats_mb();
        let mut partition_size =
            compute_adaptive_inner_partition_size(mem_stats.avail_mb, cfg.memory_soft_min_mb);
        let mut partition_num = 1;

        on_progress(ProgressUpdate {
            processed: 0,
            total: total_outer as usize,
            percent: 0.0,
            eta_secs: 0,
            mem_used_mb: mem_stats.used_mb,
            mem_avail_mb: mem_stats.avail_mb,
            stage: "indexing_partition_1",
            batch_size_current: Some(partition_size),
            gpu_total_mb: 0,
            gpu_free_mb: 0,
            gpu_active: false,
        });

        loop {
            // Check memory pressure before loading next partition
            let current_mem = crate::metrics::memory_stats_mb();
            if current_mem.avail_mb < cfg.memory_soft_min_mb {
                partition_size = (partition_size / 2).max(10_000);
                log::warn!(
                    "[ADAPTIVE-PARTITION] Memory pressure detected ({}MB < {}MB), reducing partition size to {}",
                    current_mem.avail_mb,
                    cfg.memory_soft_min_mb,
                    partition_size
                );
            }

            let rows =
                fetch_person_rows_chunk(inner_pool, inner_table, inner_off, partition_size).await?;
            if rows.is_empty() {
                break;
            }

            inner_off += rows.len() as i64;
            inner_rows.extend(rows);

            // Update progress for this partition
            partition_num += 1;
            on_progress(ProgressUpdate {
                processed: inner_rows.len(),
                total: total_outer as usize,
                percent: 0.0,
                eta_secs: 0,
                mem_used_mb: current_mem.used_mb,
                mem_avail_mb: current_mem.avail_mb,
                stage: "indexing_partition",
                batch_size_current: Some(partition_size),
                gpu_total_mb: 0,
                gpu_free_mb: 0,
                gpu_active: false,
            });
        }

        on_progress(ProgressUpdate {
            processed: 0,
            total: total_outer as usize,
            percent: 0.0,
            eta_secs: 0,
            mem_used_mb: crate::metrics::memory_stats_mb().used_mb,
            mem_avail_mb: crate::metrics::memory_stats_mb().avail_mb,
            stage: "indexing_done",
            batch_size_current: Some(partition_size),
            gpu_total_mb: 0,
            gpu_free_mb: 0,
            gpu_active: false,
        });

        // Build partitioner factory
        let key_algo = algo;
        let part_make = || {
            FnPartitioner::<crate::models::Person, _>(
                move |p| {
                    crate::matching::key_for_engine(key_algo, &normalize_person(p))
                        .unwrap_or_default()
                },
                std::marker::PhantomData,
            )
        };

        let ck_path = cfg
            .checkpoint_path
            .clone()
            .unwrap_or_else(|| "engine_ck.db".into());
        let job = format!("dual:{}->{}", inner_table, outer_table);
        let mut total_written = 0usize;
        let batch = cfg.batch_size.max(10_000);
        match algo {
            MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => {
                let part_a = part_make();
                let part_b = part_make();
                let mut eng = StreamEngine::new(
                    LegacyAdapterAlgo1,
                    part_a,
                    part_b,
                    FileCheckpointer::new(ck_path.clone()),
                );
                let mut offset: i64 = 0;
                let start = Instant::now();
                while offset < total_outer {
                    if let Some(c) = &ctrl {
                        if c.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                            break;
                        }
                        while c.pause.load(std::sync::atomic::Ordering::Relaxed) {
                            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                        }
                    }
                    let rows =
                        fetch_person_rows_chunk(outer_pool, outer_table, offset, batch).await?;
                    if rows.is_empty() {
                        break;
                    }
                    offset += rows.len() as i64;
                    let processed = (offset as usize).min(total_outer as usize);
                    let wrote = eng.for_each(
                        &job,
                        rows.iter(),
                        inner_rows.iter(),
                        |a, b, score, _expl| {
                            let pair = if inner_is_t2 {
                                crate::matching::to_pair_public(a, b, algo)
                            } else {
                                crate::matching::to_pair_public(b, a, algo)
                            };
                            let mut pair = pair;
                            pair.confidence = (score as f32) / 100.0;
                            on_match(&pair)
                        },
                    )?;
                    total_written += wrote;
                    let frac = (processed as f32 / total_outer as f32).clamp(0.0, 1.0);
                    let eta = if frac > 0.0 {
                        (start.elapsed().as_secs_f32() * (1.0 - frac) / frac) as u64
                    } else {
                        0
                    };
                    let memx = crate::metrics::memory_stats_mb();
                    on_progress(ProgressUpdate {
                        processed,
                        total: total_outer as usize,
                        percent: frac * 100.0,
                        eta_secs: eta,
                        mem_used_mb: memx.used_mb,
                        mem_avail_mb: memx.avail_mb,
                        stage: "streaming",
                        batch_size_current: Some(batch),
                        gpu_total_mb: 0,
                        gpu_free_mb: 0,
                        gpu_active: false,
                    });
                    tokio::task::yield_now().await;
                }
            }
            MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => {
                let part_a = part_make();
                let part_b = part_make();
                let mut eng = StreamEngine::new(
                    LegacyAdapterAlgo2,
                    part_a,
                    part_b,
                    FileCheckpointer::new(ck_path.clone()),
                );
                let mut offset: i64 = 0;
                let start = Instant::now();
                while offset < total_outer {
                    if let Some(c) = &ctrl {
                        if c.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                            break;
                        }
                        while c.pause.load(std::sync::atomic::Ordering::Relaxed) {
                            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                        }
                    }
                    let rows =
                        fetch_person_rows_chunk(outer_pool, outer_table, offset, batch).await?;
                    if rows.is_empty() {
                        break;
                    }
                    offset += rows.len() as i64;
                    let processed = (offset as usize).min(total_outer as usize);
                    let wrote = eng.for_each(
                        &job,
                        rows.iter(),
                        inner_rows.iter(),
                        |a, b, score, _expl| {
                            let pair = if inner_is_t2 {
                                crate::matching::to_pair_public(a, b, algo)
                            } else {
                                crate::matching::to_pair_public(b, a, algo)
                            };
                            let mut pair = pair;
                            pair.confidence = (score as f32) / 100.0;
                            on_match(&pair)
                        },
                    )?;
                    total_written += wrote;
                    let frac = (processed as f32 / total_outer as f32).clamp(0.0, 1.0);
                    let eta = if frac > 0.0 {
                        (start.elapsed().as_secs_f32() * (1.0 - frac) / frac) as u64
                    } else {
                        0
                    };
                    let memx = crate::metrics::memory_stats_mb();
                    on_progress(ProgressUpdate {
                        processed,
                        total: total_outer as usize,
                        percent: frac * 100.0,
                        eta_secs: eta,
                        mem_used_mb: memx.used_mb,
                        mem_avail_mb: memx.avail_mb,
                        stage: "streaming",
                        batch_size_current: Some(batch),
                        gpu_total_mb: 0,
                        gpu_free_mb: 0,
                        gpu_active: false,
                    });
                    tokio::task::yield_now().await;
                }
            }
            MatchingAlgorithm::Fuzzy => {
                let part_a = part_make();
                let part_b = part_make();
                let mut eng = StreamEngine::new(
                    LegacyAdapterFuzzy,
                    part_a,
                    part_b,
                    FileCheckpointer::new(ck_path.clone()),
                );
                let mut offset: i64 = 0;
                let start = Instant::now();
                while offset < total_outer {
                    if let Some(c) = &ctrl {
                        if c.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                            break;
                        }
                        while c.pause.load(std::sync::atomic::Ordering::Relaxed) {
                            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                        }
                    }
                    let rows =
                        fetch_person_rows_chunk(outer_pool, outer_table, offset, batch).await?;
                    if rows.is_empty() {
                        break;
                    }
                    offset += rows.len() as i64;
                    let processed = (offset as usize).min(total_outer as usize);
                    let wrote = eng.for_each(
                        &job,
                        rows.iter(),
                        inner_rows.iter(),
                        |a, b, score, _expl| {
                            let pair = if inner_is_t2 {
                                crate::matching::to_pair_public(a, b, algo)
                            } else {
                                crate::matching::to_pair_public(b, a, algo)
                            };
                            let mut pair = pair;
                            pair.confidence = (score as f32) / 100.0;
                            on_match(&pair)
                        },
                    )?;
                    total_written += wrote;
                    let frac = (processed as f32 / total_outer as f32).clamp(0.0, 1.0);
                    let eta = if frac > 0.0 {
                        (start.elapsed().as_secs_f32() * (1.0 - frac) / frac) as u64
                    } else {
                        0
                    };
                    let memx = crate::metrics::memory_stats_mb();
                    on_progress(ProgressUpdate {
                        processed,
                        total: total_outer as usize,
                        percent: frac * 100.0,
                        eta_secs: eta,
                        mem_used_mb: memx.used_mb,
                        mem_avail_mb: memx.avail_mb,
                        stage: "streaming",
                        batch_size_current: Some(batch),
                        gpu_total_mb: 0,
                        gpu_free_mb: 0,
                        gpu_active: false,
                    });
                    tokio::task::yield_now().await;
                }
            }
            MatchingAlgorithm::FuzzyNoMiddle => {
                let part_a = part_make();
                let part_b = part_make();
                let mut eng = StreamEngine::new(
                    LegacyAdapterFuzzyNoMiddle,
                    part_a,
                    part_b,
                    FileCheckpointer::new(ck_path.clone()),
                );
                let mut offset: i64 = 0;
                let start = Instant::now();
                while offset < total_outer {
                    if let Some(c) = &ctrl {
                        if c.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                            break;
                        }
                        while c.pause.load(std::sync::atomic::Ordering::Relaxed) {
                            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                        }
                    }
                    let rows =
                        fetch_person_rows_chunk(outer_pool, outer_table, offset, batch).await?;
                    if rows.is_empty() {
                        break;
                    }
                    offset += rows.len() as i64;
                    let processed = (offset as usize).min(total_outer as usize);
                    let wrote = eng.for_each(
                        &job,
                        rows.iter(),
                        inner_rows.iter(),
                        |a, b, score, _expl| {
                            let pair = if inner_is_t2 {
                                crate::matching::to_pair_public(a, b, algo)
                            } else {
                                crate::matching::to_pair_public(b, a, algo)
                            };
                            let mut pair = pair;
                            pair.confidence = (score as f32) / 100.0;
                            on_match(&pair)
                        },
                    )?;
                    total_written += wrote;
                    let frac = (processed as f32 / total_outer as f32).clamp(0.0, 1.0);
                    let eta = if frac > 0.0 {
                        (start.elapsed().as_secs_f32() * (1.0 - frac) / frac) as u64
                    } else {
                        0
                    };
                    let memx = crate::metrics::memory_stats_mb();
                    on_progress(ProgressUpdate {
                        processed,
                        total: total_outer as usize,
                        percent: frac * 100.0,
                        eta_secs: eta,
                        mem_used_mb: memx.used_mb,
                        mem_avail_mb: memx.avail_mb,
                        stage: "streaming",
                        batch_size_current: Some(batch),
                        gpu_total_mb: 0,
                        gpu_free_mb: 0,
                        gpu_active: false,
                    });
                    tokio::task::yield_now().await;
                }
            }
            MatchingAlgorithm::LevenshteinWeighted => {
                // Not supported in engine streaming; GUI disables streaming for this algorithm.
            }
            MatchingAlgorithm::HouseholdGpu | MatchingAlgorithm::HouseholdGpuOpt6 => {
                let part_a = part_make();
                let part_b = part_make();
                let mut eng = StreamEngine::new(
                    LegacyAdapterAlgo1,
                    part_a,
                    part_b,
                    FileCheckpointer::new(ck_path.clone()),
                );
                let mut offset: i64 = 0;
                let start = Instant::now();
                while offset < total_outer {
                    if let Some(c) = &ctrl {
                        if c.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                            break;
                        }
                        while c.pause.load(std::sync::atomic::Ordering::Relaxed) {
                            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                        }
                    }
                    let rows =
                        fetch_person_rows_chunk(outer_pool, outer_table, offset, batch).await?;
                    if rows.is_empty() {
                        break;
                    }
                    offset += rows.len() as i64;
                    let processed = (offset as usize).min(total_outer as usize);
                    let wrote = eng.for_each(
                        &job,
                        rows.iter(),
                        inner_rows.iter(),
                        |a, b, score, _expl| {
                            let pair = if inner_is_t2 {
                                crate::matching::to_pair_public(a, b, algo)
                            } else {
                                crate::matching::to_pair_public(b, a, algo)
                            };
                            let mut pair = pair;
                            pair.confidence = (score as f32) / 100.0;
                            on_match(&pair)
                        },
                    )?;
                    total_written += wrote;
                    let frac = (processed as f32 / total_outer as f32).clamp(0.0, 1.0);
                    let eta = if frac > 0.0 {
                        (start.elapsed().as_secs_f32() * (1.0 - frac) / frac) as u64
                    } else {
                        0
                    };
                    let memx = crate::metrics::memory_stats_mb();
                    on_progress(ProgressUpdate {
                        processed,
                        total: total_outer as usize,
                        percent: frac * 100.0,
                        eta_secs: eta,
                        mem_used_mb: memx.used_mb,
                        mem_avail_mb: memx.avail_mb,
                        stage: "streaming",
                        batch_size_current: Some(batch),
                        gpu_total_mb: 0,
                        gpu_free_mb: 0,
                        gpu_active: false,
                    });
                    tokio::task::yield_now().await;
                }
            }
        }
        Ok(total_written)
    }

    /// Partition-aware streaming with optional column mappings using trait-based engine.
    pub async fn stream_new_engine_partitioned<F>(
        pool: &MySqlPool,
        table1: &str,
        table2: &str,
        algo: MatchingAlgorithm,
        mut on_match: F,
        cfg: StreamingConfig,
        on_progress: impl Fn(ProgressUpdate) + Sync,
        ctrl: Option<StreamControl>,
        mapping1: Option<&ColumnMapping>,
        mapping2: Option<&ColumnMapping>,
        part_cfg: crate::matching::PartitioningConfig,
    ) -> Result<usize>
    where
        F: FnMut(&MatchPair) -> Result<()>,
    {
        use crate::db::schema::{
            fetch_person_rows_chunk_where, get_person_count_where, get_person_rows_where,
        };
        use crate::engine::file_checkpointer::FileCheckpointer;
        use crate::engine::legacy_adapters::{
            LegacyAdapterAlgo1, LegacyAdapterAlgo2, LegacyAdapterFuzzy, LegacyAdapterFuzzyNoMiddle,
        };
        use crate::engine::{FnPartitioner, StreamEngine};
        use crate::normalize::normalize_person;
        use crate::util::partition::{DefaultPartition, PartitionStrategy};
        use std::collections::HashMap;
        use std::time::Instant;

        let strat: Box<dyn PartitionStrategy + Send + Sync> = match part_cfg.strategy.as_str() {
            "birthyear5" => DefaultPartition::BirthYear5.build(),
            _ => DefaultPartition::LastInitial.build(),
        };
        let parts1 = strat.partitions(mapping1);
        let parts2 = strat.partitions(mapping2);
        if parts1.len() != parts2.len() {
            anyhow::bail!(
                "Partition strategy produced mismatched partition counts for the two tables"
            );
        }

        let matcher_direct = matches!(
            algo,
            MatchingAlgorithm::IdUuidYasIsMatchedInfnbd
                | MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd
                | MatchingAlgorithm::HouseholdGpu
                | MatchingAlgorithm::HouseholdGpuOpt6
        );
        let mut total_written = 0usize;
        let mut start_part: usize = 0;
        let mut offset: i64 = 0;
        let mut batch = cfg.batch_size.max(10_000);
        if cfg.resume {
            if let Some(pth) = cfg.checkpoint_path.as_ref() {
                if let Some(cp) = crate::util::checkpoint::load_checkpoint(pth) {
                    start_part = (cp.partition_idx as isize).max(0) as usize;
                    offset = cp.next_offset;
                    batch = cp.batch_size.max(10_000);
                }
            }
        }

        for pi in start_part..parts1.len() {
            let p1 = &parts1[pi];
            let p2 = &parts2[pi];
            let c1 = get_person_count_where(pool, table1, &p1.where_sql, &p1.binds).await?;
            let c2 = get_person_count_where(pool, table2, &p2.where_sql, &p2.binds).await?;
            let inner_is_t2 = c2 <= c1;
            let (inner_table, inner_where, inner_binds, inner_map) = if inner_is_t2 {
                (table2, &p2.where_sql, &p2.binds, mapping2)
            } else {
                (table1, &p1.where_sql, &p1.binds, mapping1)
            };
            let (outer_table, outer_where, outer_binds, outer_map) = if inner_is_t2 {
                (table1, &p1.where_sql, &p1.binds, mapping1)
            } else {
                (table2, &p2.where_sql, &p2.binds, mapping2)
            };
            let total_outer = if inner_is_t2 { c1 } else { c2 };

            // Load inner rows for this partition
            on_progress(ProgressUpdate {
                processed: 0,
                total: total_outer as usize,
                percent: 0.0,
                eta_secs: 0,
                mem_used_mb: crate::metrics::memory_stats_mb().used_mb,
                mem_avail_mb: crate::metrics::memory_stats_mb().avail_mb,
                stage: "indexing",
                batch_size_current: Some(batch),
                gpu_total_mb: 0,
                gpu_free_mb: 0,
                gpu_active: false,
            });
            let inner_rows =
                get_person_rows_where(pool, inner_table, inner_where, inner_binds, inner_map)
                    .await?;
            on_progress(ProgressUpdate {
                processed: 0,
                total: total_outer as usize,
                percent: 0.0,
                eta_secs: 0,
                mem_used_mb: crate::metrics::memory_stats_mb().used_mb,
                mem_avail_mb: crate::metrics::memory_stats_mb().avail_mb,
                stage: "indexing_done",
                batch_size_current: Some(batch),
                gpu_total_mb: 0,
                gpu_free_mb: 0,
                gpu_active: false,
            });

            // Prepare partitioners
            let ck_path = cfg
                .checkpoint_path
                .clone()
                .unwrap_or_else(|| "engine_ck.db".into());
            let last_initial = !matcher_direct && part_cfg.strategy == "last_initial";
            let key_algo = algo;
            let part = FnPartitioner::<crate::models::Person, _>(
                move |p| {
                    if last_initial {
                        let n = normalize_person(p);
                        n.last_name
                            .as_deref()
                            .and_then(|s| s.chars().next())
                            .unwrap_or('\0')
                            .to_ascii_uppercase()
                            .to_string()
                    } else {
                        crate::matching::key_for_engine(key_algo, &normalize_person(p))
                            .unwrap_or_default()
                    }
                },
                std::marker::PhantomData,
            );

            // Build engine per algorithm at use-site to avoid heterogeneous generic types

            // Fuzzy: group inner by birthdate to enforce exact birthdate equality
            let mut by_date: Option<HashMap<chrono::NaiveDate, Vec<crate::models::Person>>> = None;
            if !matcher_direct {
                let mut map: HashMap<chrono::NaiveDate, Vec<crate::models::Person>> =
                    HashMap::new();
                for p in inner_rows.iter() {
                    if let Some(d) = p.birthdate {
                        map.entry(d).or_default().push(p.clone());
                    }
                }
                by_date = Some(map);
            }

            if pi != start_part {
                offset = 0;
            }
            let start = Instant::now();
            while offset < total_outer {
                if let Some(c) = &ctrl {
                    if c.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                        break;
                    }
                    while c.pause.load(std::sync::atomic::Ordering::Relaxed) {
                        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                    }
                }
                let rows = fetch_person_rows_chunk_where(
                    pool,
                    outer_table,
                    offset,
                    batch,
                    outer_where,
                    outer_binds,
                    outer_map,
                )
                .await?;
                if rows.is_empty() {
                    break;
                }
                offset += rows.len() as i64;
                let processed = (offset as usize).min(total_outer as usize);
                let job = format!("part:{}:{}->{}", pi, inner_table, outer_table);
                let wrote = if matcher_direct {
                    match algo {
                        MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => {
                            let part_a = FnPartitioner::<crate::models::Person, _>(
                                move |p| {
                                    if last_initial {
                                        let n = normalize_person(p);
                                        n.last_name
                                            .as_deref()
                                            .and_then(|s| s.chars().next())
                                            .unwrap_or('\0')
                                            .to_ascii_uppercase()
                                            .to_string()
                                    } else {
                                        crate::matching::key_for_engine(
                                            key_algo,
                                            &normalize_person(p),
                                        )
                                        .unwrap_or_default()
                                    }
                                },
                                std::marker::PhantomData,
                            );
                            let part_b = FnPartitioner::<crate::models::Person, _>(
                                move |p| {
                                    if last_initial {
                                        let n = normalize_person(p);
                                        n.last_name
                                            .as_deref()
                                            .and_then(|s| s.chars().next())
                                            .unwrap_or('\0')
                                            .to_ascii_uppercase()
                                            .to_string()
                                    } else {
                                        crate::matching::key_for_engine(
                                            key_algo,
                                            &normalize_person(p),
                                        )
                                        .unwrap_or_default()
                                    }
                                },
                                std::marker::PhantomData,
                            );
                            let mut eng = StreamEngine::new(
                                LegacyAdapterAlgo1,
                                part_a,
                                part_b,
                                FileCheckpointer::new(ck_path.clone()),
                            );
                            eng.for_each(
                                &job,
                                rows.iter(),
                                inner_rows.iter(),
                                |a, b, score, _e| {
                                    let pair = if inner_is_t2 {
                                        crate::matching::to_pair_public(a, b, algo)
                                    } else {
                                        crate::matching::to_pair_public(b, a, algo)
                                    };
                                    let mut pair = pair;
                                    pair.confidence = (score as f32) / 100.0;
                                    on_match(&pair)
                                },
                            )?
                        }
                        MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => {
                            let part_a = FnPartitioner::<crate::models::Person, _>(
                                move |p| {
                                    if last_initial {
                                        let n = normalize_person(p);
                                        n.last_name
                                            .as_deref()
                                            .and_then(|s| s.chars().next())
                                            .unwrap_or('\0')
                                            .to_ascii_uppercase()
                                            .to_string()
                                    } else {
                                        crate::matching::key_for_engine(
                                            key_algo,
                                            &normalize_person(p),
                                        )
                                        .unwrap_or_default()
                                    }
                                },
                                std::marker::PhantomData,
                            );
                            let part_b = FnPartitioner::<crate::models::Person, _>(
                                move |p| {
                                    if last_initial {
                                        let n = normalize_person(p);
                                        n.last_name
                                            .as_deref()
                                            .and_then(|s| s.chars().next())
                                            .unwrap_or('\0')
                                            .to_ascii_uppercase()
                                            .to_string()
                                    } else {
                                        crate::matching::key_for_engine(
                                            key_algo,
                                            &normalize_person(p),
                                        )
                                        .unwrap_or_default()
                                    }
                                },
                                std::marker::PhantomData,
                            );
                            let mut eng = StreamEngine::new(
                                LegacyAdapterAlgo2,
                                part_a,
                                part_b,
                                FileCheckpointer::new(ck_path.clone()),
                            );
                            eng.for_each(
                                &job,
                                rows.iter(),
                                inner_rows.iter(),
                                |a, b, score, _e| {
                                    let pair = if inner_is_t2 {
                                        crate::matching::to_pair_public(a, b, algo)
                                    } else {
                                        crate::matching::to_pair_public(b, a, algo)
                                    };
                                    let mut pair = pair;
                                    pair.confidence = (score as f32) / 100.0;
                                    on_match(&pair)
                                },
                            )?
                        }
                        MatchingAlgorithm::Fuzzy => {
                            let part_a = FnPartitioner::<crate::models::Person, _>(
                                move |p| {
                                    if last_initial {
                                        let n = normalize_person(p);
                                        n.last_name
                                            .as_deref()
                                            .and_then(|s| s.chars().next())
                                            .unwrap_or('\0')
                                            .to_ascii_uppercase()
                                            .to_string()
                                    } else {
                                        crate::matching::key_for_engine(
                                            key_algo,
                                            &normalize_person(p),
                                        )
                                        .unwrap_or_default()
                                    }
                                },
                                std::marker::PhantomData,
                            );
                            let part_b = FnPartitioner::<crate::models::Person, _>(
                                move |p| {
                                    if last_initial {
                                        let n = normalize_person(p);
                                        n.last_name
                                            .as_deref()
                                            .and_then(|s| s.chars().next())
                                            .unwrap_or('\0')
                                            .to_ascii_uppercase()
                                            .to_string()
                                    } else {
                                        crate::matching::key_for_engine(
                                            key_algo,
                                            &normalize_person(p),
                                        )
                                        .unwrap_or_default()
                                    }
                                },
                                std::marker::PhantomData,
                            );
                            let mut eng = StreamEngine::new(
                                LegacyAdapterFuzzy,
                                part_a,
                                part_b,
                                FileCheckpointer::new(ck_path.clone()),
                            );
                            eng.for_each(
                                &job,
                                rows.iter(),
                                inner_rows.iter(),
                                |a, b, score, _e| {
                                    let pair = if inner_is_t2 {
                                        crate::matching::to_pair_public(a, b, algo)
                                    } else {
                                        crate::matching::to_pair_public(b, a, algo)
                                    };
                                    let mut pair = pair;
                                    pair.confidence = (score as f32) / 100.0;
                                    on_match(&pair)
                                },
                            )?
                        }
                        MatchingAlgorithm::FuzzyNoMiddle => {
                            let part_a = FnPartitioner::<crate::models::Person, _>(
                                move |p| {
                                    if last_initial {
                                        let n = normalize_person(p);
                                        n.last_name
                                            .as_deref()
                                            .and_then(|s| s.chars().next())
                                            .unwrap_or('\0')
                                            .to_ascii_uppercase()
                                            .to_string()
                                    } else {
                                        crate::matching::key_for_engine(
                                            key_algo,
                                            &normalize_person(p),
                                        )
                                        .unwrap_or_default()
                                    }
                                },
                                std::marker::PhantomData,
                            );
                            let part_b = FnPartitioner::<crate::models::Person, _>(
                                move |p| {
                                    if last_initial {
                                        let n = normalize_person(p);
                                        n.last_name
                                            .as_deref()
                                            .and_then(|s| s.chars().next())
                                            .unwrap_or('\0')
                                            .to_ascii_uppercase()
                                            .to_string()
                                    } else {
                                        crate::matching::key_for_engine(
                                            key_algo,
                                            &normalize_person(p),
                                        )
                                        .unwrap_or_default()
                                    }
                                },
                                std::marker::PhantomData,
                            );
                            let mut eng = StreamEngine::new(
                                LegacyAdapterFuzzyNoMiddle,
                                part_a,
                                part_b,
                                FileCheckpointer::new(ck_path.clone()),
                            );
                            eng.for_each(
                                &job,
                                rows.iter(),
                                inner_rows.iter(),
                                |a, b, score, _e| {
                                    let pair = if inner_is_t2 {
                                        crate::matching::to_pair_public(a, b, algo)
                                    } else {
                                        crate::matching::to_pair_public(b, a, algo)
                                    };
                                    let mut pair = pair;
                                    pair.confidence = (score as f32) / 100.0;
                                    on_match(&pair)
                                },
                            )?
                        }
                        MatchingAlgorithm::HouseholdGpu | MatchingAlgorithm::HouseholdGpuOpt6 => {
                            let part_a = FnPartitioner::<crate::models::Person, _>(
                                move |p| {
                                    if last_initial {
                                        let n = normalize_person(p);
                                        n.last_name
                                            .as_deref()
                                            .and_then(|s| s.chars().next())
                                            .unwrap_or('\0')
                                            .to_ascii_uppercase()
                                            .to_string()
                                    } else {
                                        crate::matching::key_for_engine(
                                            key_algo,
                                            &normalize_person(p),
                                        )
                                        .unwrap_or_default()
                                    }
                                },
                                std::marker::PhantomData,
                            );
                            let part_b = FnPartitioner::<crate::models::Person, _>(
                                move |p| {
                                    if last_initial {
                                        let n = normalize_person(p);
                                        n.last_name
                                            .as_deref()
                                            .and_then(|s| s.chars().next())
                                            .unwrap_or('\0')
                                            .to_ascii_uppercase()
                                            .to_string()
                                    } else {
                                        crate::matching::key_for_engine(
                                            key_algo,
                                            &normalize_person(p),
                                        )
                                        .unwrap_or_default()
                                    }
                                },
                                std::marker::PhantomData,
                            );
                            let mut eng = StreamEngine::new(
                                LegacyAdapterAlgo1,
                                part_a,
                                part_b,
                                FileCheckpointer::new(ck_path.clone()),
                            );
                            eng.for_each(
                                &job,
                                rows.iter(),
                                inner_rows.iter(),
                                |a, b, score, _e| {
                                    let pair = if inner_is_t2 {
                                        crate::matching::to_pair_public(a, b, algo)
                                    } else {
                                        crate::matching::to_pair_public(b, a, algo)
                                    };
                                    let mut pair = pair;
                                    pair.confidence = (score as f32) / 100.0;
                                    on_match(&pair)
                                },
                            )?
                        }
                        MatchingAlgorithm::LevenshteinWeighted => 0,
                    }
                } else {
                    let mut wrote = 0usize;
                    if let Some(map) = by_date.as_ref() {
                        use std::collections::HashMap as Hm;
                        let mut out_map: Hm<chrono::NaiveDate, Vec<&crate::models::Person>> =
                            Hm::new();
                        for p in rows.iter() {
                            if let Some(d) = p.birthdate {
                                out_map.entry(d).or_default().push(p);
                            }
                        }
                        match algo {
                            MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => {
                                let part_a = FnPartitioner::<crate::models::Person, _>(
                                    move |p| {
                                        if last_initial {
                                            let n = normalize_person(p);
                                            n.last_name
                                                .as_deref()
                                                .and_then(|s| s.chars().next())
                                                .unwrap_or('\0')
                                                .to_ascii_uppercase()
                                                .to_string()
                                        } else {
                                            crate::matching::key_for_engine(
                                                key_algo,
                                                &normalize_person(p),
                                            )
                                            .unwrap_or_default()
                                        }
                                    },
                                    std::marker::PhantomData,
                                );
                                let part_b = FnPartitioner::<crate::models::Person, _>(
                                    move |p| {
                                        if last_initial {
                                            let n = normalize_person(p);
                                            n.last_name
                                                .as_deref()
                                                .and_then(|s| s.chars().next())
                                                .unwrap_or('\0')
                                                .to_ascii_uppercase()
                                                .to_string()
                                        } else {
                                            crate::matching::key_for_engine(
                                                key_algo,
                                                &normalize_person(p),
                                            )
                                            .unwrap_or_default()
                                        }
                                    },
                                    std::marker::PhantomData,
                                );
                                let mut eng = StreamEngine::new(
                                    LegacyAdapterAlgo1,
                                    part_a,
                                    part_b,
                                    FileCheckpointer::new(ck_path.clone()),
                                );
                                for (d, outs) in out_map.iter() {
                                    if let Some(inners) = map.get(d) {
                                        wrote += eng.for_each(
                                            &job,
                                            outs.iter().copied(),
                                            inners.iter(),
                                            |a, b, score, _e| {
                                                let pair = if inner_is_t2 {
                                                    crate::matching::to_pair_public(a, b, algo)
                                                } else {
                                                    crate::matching::to_pair_public(b, a, algo)
                                                };
                                                let mut pair = pair;
                                                pair.confidence = (score as f32) / 100.0;
                                                on_match(&pair)
                                            },
                                        )?;
                                    }
                                }
                            }
                            MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => {
                                let part_a = FnPartitioner::<crate::models::Person, _>(
                                    move |p| {
                                        if last_initial {
                                            let n = normalize_person(p);
                                            n.last_name
                                                .as_deref()
                                                .and_then(|s| s.chars().next())
                                                .unwrap_or('\0')
                                                .to_ascii_uppercase()
                                                .to_string()
                                        } else {
                                            crate::matching::key_for_engine(
                                                key_algo,
                                                &normalize_person(p),
                                            )
                                            .unwrap_or_default()
                                        }
                                    },
                                    std::marker::PhantomData,
                                );
                                let part_b = FnPartitioner::<crate::models::Person, _>(
                                    move |p| {
                                        if last_initial {
                                            let n = normalize_person(p);
                                            n.last_name
                                                .as_deref()
                                                .and_then(|s| s.chars().next())
                                                .unwrap_or('\0')
                                                .to_ascii_uppercase()
                                                .to_string()
                                        } else {
                                            crate::matching::key_for_engine(
                                                key_algo,
                                                &normalize_person(p),
                                            )
                                            .unwrap_or_default()
                                        }
                                    },
                                    std::marker::PhantomData,
                                );
                                let mut eng = StreamEngine::new(
                                    LegacyAdapterAlgo2,
                                    part_a,
                                    part_b,
                                    FileCheckpointer::new(ck_path.clone()),
                                );
                                for (d, outs) in out_map.iter() {
                                    if let Some(inners) = map.get(d) {
                                        wrote += eng.for_each(
                                            &job,
                                            outs.iter().copied(),
                                            inners.iter(),
                                            |a, b, score, _e| {
                                                let pair = if inner_is_t2 {
                                                    crate::matching::to_pair_public(a, b, algo)
                                                } else {
                                                    crate::matching::to_pair_public(b, a, algo)
                                                };
                                                let mut pair = pair;
                                                pair.confidence = (score as f32) / 100.0;
                                                on_match(&pair)
                                            },
                                        )?;
                                    }
                                }
                            }
                            MatchingAlgorithm::Fuzzy => {
                                let part_a = FnPartitioner::<crate::models::Person, _>(
                                    move |p| {
                                        if last_initial {
                                            let n = normalize_person(p);
                                            n.last_name
                                                .as_deref()
                                                .and_then(|s| s.chars().next())
                                                .unwrap_or('\0')
                                                .to_ascii_uppercase()
                                                .to_string()
                                        } else {
                                            crate::matching::key_for_engine(
                                                key_algo,
                                                &normalize_person(p),
                                            )
                                            .unwrap_or_default()
                                        }
                                    },
                                    std::marker::PhantomData,
                                );
                                let part_b = FnPartitioner::<crate::models::Person, _>(
                                    move |p| {
                                        if last_initial {
                                            let n = normalize_person(p);
                                            n.last_name
                                                .as_deref()
                                                .and_then(|s| s.chars().next())
                                                .unwrap_or('\0')
                                                .to_ascii_uppercase()
                                                .to_string()
                                        } else {
                                            crate::matching::key_for_engine(
                                                key_algo,
                                                &normalize_person(p),
                                            )
                                            .unwrap_or_default()
                                        }
                                    },
                                    std::marker::PhantomData,
                                );
                                let mut eng = StreamEngine::new(
                                    LegacyAdapterFuzzy,
                                    part_a,
                                    part_b,
                                    FileCheckpointer::new(ck_path.clone()),
                                );
                                for (d, outs) in out_map.iter() {
                                    if let Some(inners) = map.get(d) {
                                        wrote += eng.for_each(
                                            &job,
                                            outs.iter().copied(),
                                            inners.iter(),
                                            |a, b, score, _e| {
                                                let pair = if inner_is_t2 {
                                                    crate::matching::to_pair_public(a, b, algo)
                                                } else {
                                                    crate::matching::to_pair_public(b, a, algo)
                                                };
                                                let mut pair = pair;
                                                pair.confidence = (score as f32) / 100.0;
                                                on_match(&pair)
                                            },
                                        )?;
                                    }
                                }
                            }
                            MatchingAlgorithm::FuzzyNoMiddle => {
                                let part_a = FnPartitioner::<crate::models::Person, _>(
                                    move |p| {
                                        if last_initial {
                                            let n = normalize_person(p);
                                            n.last_name
                                                .as_deref()
                                                .and_then(|s| s.chars().next())
                                                .unwrap_or('\0')
                                                .to_ascii_uppercase()
                                                .to_string()
                                        } else {
                                            crate::matching::key_for_engine(
                                                key_algo,
                                                &normalize_person(p),
                                            )
                                            .unwrap_or_default()
                                        }
                                    },
                                    std::marker::PhantomData,
                                );
                                let part_b = FnPartitioner::<crate::models::Person, _>(
                                    move |p| {
                                        if last_initial {
                                            let n = normalize_person(p);
                                            n.last_name
                                                .as_deref()
                                                .and_then(|s| s.chars().next())
                                                .unwrap_or('\0')
                                                .to_ascii_uppercase()
                                                .to_string()
                                        } else {
                                            crate::matching::key_for_engine(
                                                key_algo,
                                                &normalize_person(p),
                                            )
                                            .unwrap_or_default()
                                        }
                                    },
                                    std::marker::PhantomData,
                                );
                                let mut eng = StreamEngine::new(
                                    LegacyAdapterFuzzyNoMiddle,
                                    part_a,
                                    part_b,
                                    FileCheckpointer::new(ck_path.clone()),
                                );
                                for (d, outs) in out_map.iter() {
                                    if let Some(inners) = map.get(d) {
                                        wrote += eng.for_each(
                                            &job,
                                            outs.iter().copied(),
                                            inners.iter(),
                                            |a, b, score, _e| {
                                                let pair = if inner_is_t2 {
                                                    crate::matching::to_pair_public(a, b, algo)
                                                } else {
                                                    crate::matching::to_pair_public(b, a, algo)
                                                };
                                                let mut pair = pair;
                                                pair.confidence = (score as f32) / 100.0;
                                                on_match(&pair)
                                            },
                                        )?;
                                    }
                                }
                            }
                            MatchingAlgorithm::HouseholdGpu
                            | MatchingAlgorithm::HouseholdGpuOpt6 => {
                                let part_a = FnPartitioner::<crate::models::Person, _>(
                                    move |p| {
                                        if last_initial {
                                            let n = normalize_person(p);
                                            n.last_name
                                                .as_deref()
                                                .and_then(|s| s.chars().next())
                                                .unwrap_or('\0')
                                                .to_ascii_uppercase()
                                                .to_string()
                                        } else {
                                            crate::matching::key_for_engine(
                                                key_algo,
                                                &normalize_person(p),
                                            )
                                            .unwrap_or_default()
                                        }
                                    },
                                    std::marker::PhantomData,
                                );
                                let part_b = FnPartitioner::<crate::models::Person, _>(
                                    move |p| {
                                        if last_initial {
                                            let n = normalize_person(p);
                                            n.last_name
                                                .as_deref()
                                                .and_then(|s| s.chars().next())
                                                .unwrap_or('\0')
                                                .to_ascii_uppercase()
                                                .to_string()
                                        } else {
                                            crate::matching::key_for_engine(
                                                key_algo,
                                                &normalize_person(p),
                                            )
                                            .unwrap_or_default()
                                        }
                                    },
                                    std::marker::PhantomData,
                                );
                                let mut eng = StreamEngine::new(
                                    LegacyAdapterAlgo1,
                                    part_a,
                                    part_b,
                                    FileCheckpointer::new(ck_path.clone()),
                                );
                                for (d, outs) in out_map.iter() {
                                    if let Some(inners) = map.get(d) {
                                        wrote += eng.for_each(
                                            &job,
                                            outs.iter().copied(),
                                            inners.iter(),
                                            |a, b, score, _e| {
                                                let pair = if inner_is_t2 {
                                                    crate::matching::to_pair_public(a, b, algo)
                                                } else {
                                                    crate::matching::to_pair_public(b, a, algo)
                                                };
                                                let mut pair = pair;
                                                pair.confidence = (score as f32) / 100.0;
                                                on_match(&pair)
                                            },
                                        )?;
                                    }
                                }
                            }
                            MatchingAlgorithm::LevenshteinWeighted => {
                                // Not supported in engine streaming non-direct path; no-op.
                            }
                        }
                    }
                    wrote
                };
                total_written += wrote;
                let frac = (processed as f32 / total_outer as f32).clamp(0.0, 1.0);
                let eta = if frac > 0.0 {
                    (start.elapsed().as_secs_f32() * (1.0 - frac) / frac) as u64
                } else {
                    0
                };
                let memx = crate::metrics::memory_stats_mb();
                on_progress(ProgressUpdate {
                    processed,
                    total: total_outer as usize,
                    percent: frac * 100.0,
                    eta_secs: eta,
                    mem_used_mb: memx.used_mb,
                    mem_avail_mb: memx.avail_mb,
                    stage: "streaming",
                    batch_size_current: Some(batch),
                    gpu_total_mb: 0,
                    gpu_free_mb: 0,
                    gpu_active: false,
                });
                if let Some(pth) = cfg.checkpoint_path.as_ref() {
                    let _ = crate::util::checkpoint::save_checkpoint(
                        pth,
                        &crate::util::checkpoint::StreamCheckpoint {
                            db: String::new(),
                            table_inner: inner_table.to_string(),
                            table_outer: outer_table.to_string(),
                            algorithm: format!("{:?}", algo),
                            batch_size: batch,
                            next_offset: offset,
                            total_outer,
                            partition_idx: pi as i32,
                            partition_name: p1.name.clone(),
                            updated_utc: chrono::Utc::now().to_rfc3339(),
                            last_id: Some(offset),
                            watermark_id: None,
                            filter_sig: None,
                        },
                    );
                }
                tokio::task::yield_now().await;
            }
        }
        if let Some(pth) = cfg.checkpoint_path.as_ref() {
            crate::util::checkpoint::remove_checkpoint(pth);
        }
        Ok(total_written)
    }
}

#[cfg(all(test, feature = "new_engine"))]
mod tests {
    use super::*;

    #[test]
    fn test_compute_adaptive_inner_partition_size() {
        // Test with high available memory (4GB available, 800MB soft min)
        let partition_size = compute_adaptive_inner_partition_size(4096, 800);
        assert!(
            partition_size >= 10_000,
            "Should respect minimum partition size"
        );
        assert!(
            partition_size <= 500_000,
            "Should respect maximum partition size"
        );

        // Test with low available memory (1GB available, 800MB soft min)
        let partition_size_low = compute_adaptive_inner_partition_size(1024, 800);
        assert!(
            partition_size_low >= 10_000,
            "Should respect minimum partition size even with low memory"
        );
        assert!(
            partition_size_low <= partition_size,
            "Lower memory should result in smaller or equal partition size"
        );

        // Test with very low available memory (500MB available, 800MB soft min)
        let partition_size_very_low = compute_adaptive_inner_partition_size(500, 800);
        assert_eq!(
            partition_size_very_low, 10_000,
            "Should use minimum partition size when memory is very constrained"
        );

        // Test boundary conditions
        let partition_size_zero = compute_adaptive_inner_partition_size(0, 800);
        assert_eq!(
            partition_size_zero, 10_000,
            "Should handle zero available memory gracefully"
        );

        // Test with very high memory (32GB available, 800MB soft min)
        let partition_size_high = compute_adaptive_inner_partition_size(32768, 800);
        assert_eq!(
            partition_size_high, 500_000,
            "Should cap at maximum partition size"
        );
    }
}
