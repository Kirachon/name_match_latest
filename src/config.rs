use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, Default)]
pub struct DatabaseConfig {
    pub username: String,
    pub password: String,
    pub host: String,
    pub port: u16,
    pub database: String,
}

impl DatabaseConfig {
    pub fn to_url(&self) -> String {
        format!(
            "mysql://{}:{}@{}:{}/{}",
            self.username, self.password, self.host, self.port, self.database
        )
    }
}

impl std::fmt::Debug for DatabaseConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DatabaseConfig")
            .field("username", &self.username)
            .field("password", &"<redacted>")
            .field("host", &self.host)
            .field("port", &self.port)
            .field("database", &self.database)
            .finish()
    }
}

// --- Application configuration (Phase A) ---
use crate::error::ConfigError;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct StreamingConfig {
    pub batch_size: usize,
    pub resume: bool,
    pub checkpoint_path: Option<String>,
    pub partition_strategy: Option<String>, // e.g., "last_initial" | "birthyear5"
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            batch_size: 10_000,
            resume: false,
            checkpoint_path: None,
            partition_strategy: None,
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct GpuConfig {
    pub enabled: bool,
    pub use_hash_join: bool,
    pub build_on_gpu: bool,
    pub probe_on_gpu: bool,
    pub vram_mb_budget: Option<u32>, // For tiling/backoff
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            use_hash_join: false,
            build_on_gpu: false,
            probe_on_gpu: false,
            vram_mb_budget: None,
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct MatchingConfig {
    pub algorithm: Option<u8>,         // 1,2,3,4 (and HouseholdGpu via GUI)
    pub min_score_export: Option<f32>, // e.g., 95.0 for fuzzy
}

impl Default for MatchingConfig {
    fn default() -> Self {
        Self {
            algorithm: None,
            min_score_export: None,
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ExportConfig {
    pub out_path: Option<String>,
    pub format: Option<String>, // csv|xlsx|both
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            out_path: None,
            format: Some("csv".into()),
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug, Default)]
pub struct AppConfig {
    pub database: DatabaseConfig,
    #[serde(default)]
    pub streaming: StreamingConfig,
    #[serde(default)]
    pub gpu: GpuConfig,
    #[serde(default)]
    pub matching: MatchingConfig,
    #[serde(default)]
    pub export: ExportConfig,
}

impl AppConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.database.host.trim().is_empty() {
            return Err(ConfigError::MissingField {
                field: "database.host",
            });
        }
        if self.database.username.trim().is_empty() {
            return Err(ConfigError::MissingField {
                field: "database.username",
            });
        }
        if self.database.database.trim().is_empty() {
            return Err(ConfigError::MissingField {
                field: "database.database",
            });
        }
        if self.database.port == 0 || self.database.port > 65535 {
            return Err(ConfigError::InvalidValue {
                field: "database.port",
                reason: format!("{} is out of range", self.database.port),
            });
        }
        if let Some(ref fmt) = self.export.format {
            match fmt.as_str() {
                "csv" | "xlsx" | "both" => {}
                other => {
                    return Err(ConfigError::InvalidValue {
                        field: "export.format",
                        reason: format!("unsupported: {}", other),
                    });
                }
            }
        }
        if let Some(score) = self.matching.min_score_export {
            if !(0.0..=100.0).contains(&score) {
                return Err(ConfigError::InvalidValue {
                    field: "matching.min_score_export",
                    reason: format!("{} not in 0..=100", score),
                });
            }
        }

        if self.streaming.batch_size == 0 {
            return Err(ConfigError::InvalidValue {
                field: "streaming.batch_size",
                reason: "must be > 0".into(),
            });
        }
        Ok(())
    }
}
