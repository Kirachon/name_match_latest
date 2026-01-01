use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("missing required field: {field}")]
    MissingField { field: &'static str },
    #[error("invalid value for {field}: {reason}")]
    InvalidValue { field: &'static str, reason: String },
}

#[derive(Debug, Error)]
pub enum DbError {
    #[error("connection error: {0}")]
    Connection(String),
    #[error("query error: {0}")]
    Query(String),
}

#[derive(Debug, Error)]
pub enum MatchError {
    #[error("configuration error: {0}")]
    Config(String),
    #[error("internal matching error: {0}")]
    Internal(String),
}

#[derive(Debug, Error)]
pub enum GpuError {
    #[error("cuda unavailable: {0}")]
    CudaUnavailable(String),
    #[error("gpu memory error: {0}")]
    Memory(String),
    #[error("gpu kernel error: {0}")]
    Kernel(String),
}

#[derive(Debug, Error)]
pub enum ExportError {
    #[error("csv export error: {0}")]
    Csv(String),
    #[error("xlsx export error: {0}")]
    Xlsx(String),
}

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("streaming error: {0}")]
    Streaming(String),
    #[error("checkpoint error: {0}")]
    Checkpoint(String),
}
