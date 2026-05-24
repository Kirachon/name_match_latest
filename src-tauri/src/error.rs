use name_matcher::run_service::dto::{AppErrorDto, ErrorKindDto};
use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("validation error: {0}")]
    Validation(String),

    #[error("database error: {0}")]
    Database(String),

    #[error("io error: {0}")]
    Io(String),

    #[error("engine error: {0}")]
    Engine(String),

    #[error("cancelled")]
    Cancelled,

    #[error("internal error: {0}")]
    Internal(String),
}

impl AppError {
    pub fn kind(&self) -> ErrorKindDto {
        match self {
            AppError::Validation(_) => ErrorKindDto::Validation,
            AppError::Database(_) => ErrorKindDto::Database,
            AppError::Io(_) => ErrorKindDto::Io,
            AppError::Engine(_) => ErrorKindDto::Engine,
            AppError::Cancelled => ErrorKindDto::Cancelled,
            AppError::Internal(_) => ErrorKindDto::Internal,
        }
    }

    pub fn into_dto(&self) -> AppErrorDto {
        AppErrorDto {
            kind: self.kind(),
            message: self.to_string(),
            recoverable: matches!(
                self,
                AppError::Database(_) | AppError::Io(_) | AppError::Validation(_)
            ),
        }
    }
}

impl From<anyhow::Error> for AppError {
    fn from(e: anyhow::Error) -> Self {
        AppError::Internal(format!("{e:#}"))
    }
}

impl From<sqlx::Error> for AppError {
    fn from(e: sqlx::Error) -> Self {
        AppError::Database(e.to_string())
    }
}

impl From<std::io::Error> for AppError {
    fn from(e: std::io::Error) -> Self {
        AppError::Io(e.to_string())
    }
}

impl From<serde_json::Error> for AppError {
    fn from(e: serde_json::Error) -> Self {
        AppError::Internal(format!("json: {e}"))
    }
}

impl Serialize for AppError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Serialize as the typed DTO so the frontend can display structured
        // error UX without parsing the message string.
        self.into_dto().serialize(serializer)
    }
}

pub type AppResult<T> = Result<T, AppError>;
