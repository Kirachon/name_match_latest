use crate::config::{
    AppConfig, DatabaseConfig, ExportConfig, GpuConfig, MatchingConfig, StreamingConfig,
};
use crate::error::ConfigError;
use clap::{Parser, ValueEnum};

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, ValueEnum, Debug)]
pub enum FormatOpt {
    Csv,
    Xlsx,
    Both,
}

impl FormatOpt {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Csv => "csv",
            Self::Xlsx => "xlsx",
            Self::Both => "both",
        }
    }
}
impl std::fmt::Display for FormatOpt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "name_matcher",
    version,
    about = "High-performance name matching (CLI)",
    disable_help_subcommand = true
)]
pub struct Cli {
    /// DB host (env: DB_HOST)
    #[arg(value_name = "HOST", env = "DB_HOST")]
    pub host: String,
    /// DB port (env: DB_PORT, default 3307)
    #[arg(value_name = "PORT", env = "DB_PORT", default_value_t = 3307)]
    pub port: u16,
    /// DB user (env: DB_USER)
    #[arg(value_name = "USER", env = "DB_USER")]
    pub user: String,
    /// DB password (env: DB_PASSWORD or DB_PASS)
    #[arg(value_name = "PASSWORD", env = "DB_PASSWORD")]
    pub password: String,
    /// Database name (env: DB_NAME)
    #[arg(value_name = "DATABASE", env = "DB_NAME")]
    pub database: String,
    /// Table 1 name
    #[arg(value_name = "TABLE1")]
    pub table1: String,
    /// Table 2 name
    #[arg(value_name = "TABLE2")]
    pub table2: String,
    /// Algorithm (1,2,3,4,5,6)
    #[arg(value_name = "ALGO")]
    pub algo: u8,
    /// Output path
    #[arg(value_name = "OUT_PATH")]
    pub out_path: String,
    /// Output format
    #[arg(value_name = "FORMAT", default_value_t = FormatOpt::Csv)]
    pub format: FormatOpt,
    /// Enable GPU hash-join (env: NAME_MATCHER_GPU_HASH_JOIN)
    #[arg(long = "gpu-hash-join", env = "NAME_MATCHER_GPU_HASH_JOIN")]
    pub gpu_hash_join: bool,
    /// Enable GPU direct hash for fuzzy (env: NAME_MATCHER_GPU_FUZZY_DIRECT_HASH)
    #[arg(
        long = "gpu-fuzzy-direct-hash",
        env = "NAME_MATCHER_GPU_FUZZY_DIRECT_HASH"
    )]
    pub gpu_fuzzy_direct_hash: bool,
    /// Enable direct fuzzy normalization path (env: NAME_MATCHER_DIRECT_FUZZY_NORMALIZATION)
    #[arg(
        long = "direct-fuzzy-normalization",
        env = "NAME_MATCHER_DIRECT_FUZZY_NORMALIZATION"
    )]
    pub direct_fuzzy_normalization: bool,
    /// Advanced matching level (1-12)
    #[arg(long = "advanced-level", value_name = "LEVEL")]
    pub advanced_level: Option<u8>,
    /// Code column required for Advanced levels 4/5 (barangay_code) and 7/8 (city_code)
    #[arg(long = "advanced-code-col", value_name = "COLUMN")]
    pub advanced_code_col: Option<String>,
    /// Fuzzy threshold for Advanced levels 10/11/12 (default 0.95)
    #[arg(
        long = "advanced-threshold",
        value_name = "THRESHOLD",
        default_value_t = 0.95
    )]
    pub advanced_threshold: f32,
}

impl Cli {
    pub fn to_app_config(&self) -> Result<AppConfig, ConfigError> {
        let pass = if self.password.is_empty() {
            std::env::var("DB_PASS").unwrap_or_default()
        } else {
            self.password.clone()
        };

        let cfg = AppConfig {
            database: DatabaseConfig {
                username: self.user.clone(),
                password: pass,
                host: self.host.clone(),
                port: self.port,
                database: self.database.clone(),
            },
            streaming: StreamingConfig::default(),
            gpu: GpuConfig {
                enabled: self.gpu_hash_join || self.gpu_fuzzy_direct_hash,
                use_hash_join: self.gpu_hash_join,
                build_on_gpu: false,
                probe_on_gpu: self.gpu_hash_join,
                vram_mb_budget: None,
            },
            matching: MatchingConfig {
                algorithm: Some(self.algo),
                min_score_export: None,
            },
            export: ExportConfig {
                out_path: Some(self.out_path.clone()),
                format: Some(self.format.as_str().into()),
            },
        };
        cfg.validate()?;
        Ok(cfg)
    }
}

pub fn parse_cli_to_app_config() -> Result<AppConfig, ConfigError> {
    let cli = Cli::parse();
    cli.to_app_config()
}
