//! Row-scale policy helpers for RunService and the Tauri UI.

use super::dto::{
    AlgorithmDto, DataSourceKindDto, RunConfigDto, RunModeDto, StreamingOptionsDto,
    TableSelectionDto,
};

pub const SCALE_WARN_ROWS: u64 = 100_000;
pub const SCALE_STRONG_WARN_ROWS: u64 = 500_000;
pub const SCALE_BLOCK_ROWS: u64 = 1_000_000;
pub const RESULT_SPILL_ROWS: usize = 100_000;
pub const MAX_DIFF_ROWS: u64 = 100_000;
pub const DIFF_TOO_LARGE_MESSAGE: &str = "This comparison is too large to load safely in memory. Export both runs as CSV and compare externally, or rerun with narrower filters.";
pub const LARGE_RESULTS_BANNER_ROWS: u64 = 100_000;
pub const LARGE_RESULTS_DEFAULT_PAGE_SIZE: u32 = 50;

/// Cross-session dual-pool DB streaming. Set false to fall back to in-memory loader.
pub const SUPPORTS_CROSS_SESSION_STREAMING: bool = true;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectiveRunMode {
    InMemory,
    Streaming,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleBlockReason {
    MillionRowFileSource,
    MillionRowUnsupportedAlgorithm,
    MillionRowCascade,
    MillionRowFuzzy,
}

pub fn row_count_for_side(selection: &TableSelectionDto) -> u64 {
    selection.row_count.unwrap_or(0)
}

pub fn max_side_rows(config: &RunConfigDto) -> u64 {
    row_count_for_side(&config.source).max(row_count_for_side(&config.target))
}

pub fn resolve_effective_run_mode(
    config: &RunConfigDto,
    source_rows: u64,
    target_rows: u64,
) -> EffectiveRunMode {
    let max_rows = source_rows.max(target_rows);
    match config.streaming.mode {
        RunModeDto::InMemory => EffectiveRunMode::InMemory,
        RunModeDto::Streaming => EffectiveRunMode::Streaming,
        RunModeDto::Auto => {
            if max_rows >= SCALE_WARN_ROWS {
                EffectiveRunMode::Streaming
            } else {
                EffectiveRunMode::InMemory
            }
        }
    }
}

pub fn algorithm_supports_db_streaming(algorithm: AlgorithmDto) -> bool {
    matches!(
        algorithm,
        AlgorithmDto::DeterministicFnLnBd | AlgorithmDto::DeterministicFnMnLnBd
    )
}

pub fn is_db_to_db(config: &RunConfigDto) -> bool {
    matches!(config.source.source_kind, DataSourceKindDto::Database)
        && matches!(config.target.source_kind, DataSourceKindDto::Database)
}

pub fn is_same_db_session(config: &RunConfigDto) -> bool {
    config.source.session_id == config.target.session_id
}

fn db_streaming_eligible_ignoring_session(config: &RunConfigDto) -> bool {
    if !is_db_to_db(config) {
        return false;
    }
    if config.cascade.as_ref().is_some_and(|c| c.enabled) {
        return false;
    }
    if !algorithm_supports_db_streaming(config.algorithm) {
        return false;
    }
    let source_rows = row_count_for_side(&config.source);
    let target_rows = row_count_for_side(&config.target);
    matches!(
        resolve_effective_run_mode(config, source_rows, target_rows),
        EffectiveRunMode::Streaming
    )
}

pub fn should_use_db_streaming_worker(config: &RunConfigDto) -> bool {
    db_streaming_eligible_ignoring_session(config) && is_same_db_session(config)
}

pub fn should_use_two_pool_db_streaming_worker(config: &RunConfigDto) -> bool {
    if !SUPPORTS_CROSS_SESSION_STREAMING {
        return false;
    }
    if !db_streaming_eligible_ignoring_session(config) {
        return false;
    }
    if is_same_db_session(config) {
        return false;
    }
    // Dual-pool MVP uses standard table columns; custom mappings require in-memory load.
    config.source.column_mapping.is_none() && config.target.column_mapping.is_none()
}

pub fn should_use_any_db_streaming_worker(config: &RunConfigDto) -> bool {
    should_use_db_streaming_worker(config) || should_use_two_pool_db_streaming_worker(config)
}

pub fn scale_block_reason(config: &RunConfigDto) -> Option<ScaleBlockReason> {
    let max_rows = max_side_rows(config);
    if max_rows < SCALE_BLOCK_ROWS {
        return None;
    }
    if matches!(config.source.source_kind, DataSourceKindDto::File)
        || matches!(config.target.source_kind, DataSourceKindDto::File)
    {
        return Some(ScaleBlockReason::MillionRowFileSource);
    }
    if config.cascade.as_ref().is_some_and(|c| c.enabled) {
        return Some(ScaleBlockReason::MillionRowCascade);
    }
    if !algorithm_supports_db_streaming(config.algorithm) {
        if matches!(
            config.algorithm,
            AlgorithmDto::Fuzzy | AlgorithmDto::FuzzyNoMiddle
        ) {
            return Some(ScaleBlockReason::MillionRowFuzzy);
        }
        return Some(ScaleBlockReason::MillionRowUnsupportedAlgorithm);
    }
    None
}

pub fn streaming_config_from_dto(
    streaming: &StreamingOptionsDto,
) -> crate::matching::StreamingConfig {
    crate::matching::StreamingConfig {
        batch_size: streaming.batch_size as i64,
        resume: streaming.resume,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run_service::dto::{
        ExportOptionsDto, GpuOptionsDto, MatchOptionsDto, RunConfigDto, TableSelectionDto,
    };

    fn db_config(rows: u64, mode: RunModeDto, algo: AlgorithmDto) -> RunConfigDto {
        RunConfigDto {
            source: TableSelectionDto {
                source_kind: DataSourceKindDto::Database,
                session_id: "s".into(),
                table: "a".into(),
                row_count: Some(rows),
                ..Default::default()
            },
            target: TableSelectionDto {
                source_kind: DataSourceKindDto::Database,
                session_id: "s".into(),
                table: "b".into(),
                row_count: Some(rows),
                ..Default::default()
            },
            algorithm: algo,
            streaming: StreamingOptionsDto {
                mode,
                ..Default::default()
            },
            options: MatchOptionsDto::default(),
            gpu: GpuOptionsDto::default(),
            export: ExportOptionsDto::default(),
            cascade: None,
            review_band: None,
        }
    }

    #[test]
    fn auto_resolves_streaming_at_100k() {
        let cfg = db_config(150_000, RunModeDto::Auto, AlgorithmDto::DeterministicFnLnBd);
        assert_eq!(
            resolve_effective_run_mode(&cfg, 150_000, 150_000),
            EffectiveRunMode::Streaming
        );
    }

    #[test]
    fn streaming_worker_requires_same_db_session() {
        let mut cfg = db_config(
            150_000,
            RunModeDto::Streaming,
            AlgorithmDto::DeterministicFnLnBd,
        );
        cfg.target.session_id = "other-session".into();

        assert!(!should_use_db_streaming_worker(&cfg));
    }

    #[test]
    fn two_pool_worker_activates_for_cross_session_streaming() {
        let mut cfg = db_config(
            150_000,
            RunModeDto::Streaming,
            AlgorithmDto::DeterministicFnLnBd,
        );
        cfg.target.session_id = "other-session".into();

        assert!(should_use_two_pool_db_streaming_worker(&cfg));
        assert!(should_use_any_db_streaming_worker(&cfg));
    }

    #[test]
    fn two_pool_worker_not_used_for_same_session() {
        let cfg = db_config(
            150_000,
            RunModeDto::Streaming,
            AlgorithmDto::DeterministicFnLnBd,
        );

        assert!(should_use_db_streaming_worker(&cfg));
        assert!(!should_use_two_pool_db_streaming_worker(&cfg));
    }

    #[test]
    fn two_pool_worker_skips_custom_column_mapping() {
        let mut cfg = db_config(
            150_000,
            RunModeDto::Streaming,
            AlgorithmDto::DeterministicFnLnBd,
        );
        cfg.target.session_id = "other-session".into();
        cfg.source.column_mapping = Some(crate::models::ColumnMapping {
            id: "id".into(),
            first_name: "first_name".into(),
            last_name: "last_name".into(),
            birthdate: "birthdate".into(),
            ..Default::default()
        });

        assert!(!should_use_two_pool_db_streaming_worker(&cfg));
    }

    #[test]
    fn blocks_million_row_fuzzy() {
        let cfg = db_config(1_500_000, RunModeDto::Streaming, AlgorithmDto::Fuzzy);
        assert_eq!(
            scale_block_reason(&cfg),
            Some(ScaleBlockReason::MillionRowFuzzy)
        );
    }
}
