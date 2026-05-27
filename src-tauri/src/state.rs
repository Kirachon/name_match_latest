use crate::error::{AppError, AppResult};
use crate::import_jobs::ImportJobRegistry;
use name_matcher::run_service::dto::{
    DbCredentialsDto, DbSessionDto, JobStateEventDto, LogEntryDto, ProgressEventDto,
};
use name_matcher::run_service::{EventSink, JobRegistry, ResultStore, ResultStoreConfig};
use sqlx::MySqlPool;
use sqlx::mysql::{MySqlConnectOptions, MySqlPoolOptions};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter};
use tauri::Manager;
use tokio::time::Instant;

/// One connected MySQL session. We do **not** keep the password around once
/// the pool is built so that subsequent commands can not leak it.
#[derive(Clone)]
pub struct DbSession {
    pub session_id: String,
    pub host: String,
    pub port: u16,
    pub username: String,
    pub database: String,
    pub pool: MySqlPool,
}

impl DbSession {
    pub fn to_dto(&self, latency_ms: Option<u64>) -> DbSessionDto {
        DbSessionDto {
            session_id: self.session_id.clone(),
            host: self.host.clone(),
            port: self.port,
            username: self.username.clone(),
            database: self.database.clone(),
            latency_ms,
        }
    }
}

#[derive(Default)]
pub struct DbRegistry {
    sessions: Mutex<HashMap<String, DbSession>>,
}

impl DbRegistry {
    pub fn insert(&self, session: DbSession) {
        let mut g = self.sessions.lock().expect("db registry poisoned");
        g.insert(session.session_id.clone(), session);
    }
    pub fn get(&self, id: &str) -> Option<DbSession> {
        self.sessions
            .lock()
            .expect("db registry poisoned")
            .get(id)
            .cloned()
    }
    pub fn remove(&self, id: &str) -> Option<DbSession> {
        self.sessions
            .lock()
            .expect("db registry poisoned")
            .remove(id)
    }
    pub fn list(&self) -> Vec<DbSessionDto> {
        self.sessions
            .lock()
            .expect("db registry poisoned")
            .values()
            .map(|s| s.to_dto(None))
            .collect()
    }
}

pub struct AppState {
    pub app_handle: AppHandle,
    pub db: DbRegistry,
    pub import_jobs: Arc<ImportJobRegistry>,
    pub jobs: Arc<JobRegistry>,
    pub results: Arc<ResultStore>,
    pub started_at: Instant,
}

impl AppState {
    pub fn new(app_handle: AppHandle) -> Self {
        let results = app_handle
            .path()
            .app_data_dir()
            .ok()
            .and_then(|dir| {
                ResultStore::with_sqlite_path(
                    ResultStoreConfig::default(),
                    dir.join("result_store.sqlite3"),
                )
                .map_err(|err| {
                    log::error!("SQLite result store unavailable; using memory only: {err}");
                    err
                })
                .ok()
            })
            .unwrap_or_else(ResultStore::new);
        Self {
            app_handle,
            db: DbRegistry::default(),
            import_jobs: Arc::new(ImportJobRegistry::default()),
            jobs: Arc::new(JobRegistry::default()),
            results: Arc::new(results),
            started_at: Instant::now(),
        }
    }

    /// Helper that builds a credentials → pool with sane defaults for the
    /// Tauri shell. Mirrors `name_matcher::db::make_pool` but lets us keep
    /// the password out of the persistent registry.
    pub async fn build_pool(creds: &DbCredentialsDto) -> AppResult<MySqlPool> {
        if creds.host.trim().is_empty() {
            return Err(AppError::Validation("host is required".into()));
        }
        if creds.username.trim().is_empty() {
            return Err(AppError::Validation("username is required".into()));
        }
        if creds.database.trim().is_empty() {
            return Err(AppError::Validation("database is required".into()));
        }
        if creds.port == 0 {
            return Err(AppError::Validation("port must be > 0".into()));
        }
        let options = MySqlConnectOptions::new()
            .host(creds.host.trim())
            .port(creds.port)
            .username(&creds.username)
            .password(&creds.password)
            .database(creds.database.trim());
        let pool = MySqlPoolOptions::new()
            .max_connections(16)
            .acquire_timeout(std::time::Duration::from_secs(30))
            .connect_with(options)
            .await?;
        Ok(pool)
    }
}

/// EventSink that fans `match-progress`, `job-state`, and `log-entry` Tauri
/// events to the front-end. Throttling is already applied upstream in the
/// run service progress callback.
pub struct TauriEventSink {
    handle: AppHandle,
}

impl TauriEventSink {
    pub fn new(handle: AppHandle) -> Self {
        Self { handle }
    }
}

impl EventSink for TauriEventSink {
    fn emit_progress(&self, evt: ProgressEventDto) {
        let _ = self.handle.emit("match-progress", evt);
    }
    fn emit_log(&self, entry: LogEntryDto) {
        let _ = self.handle.emit("log-entry", entry);
    }
    fn emit_state(&self, evt: JobStateEventDto) {
        let _ = self.handle.emit("job-state", evt);
    }
}
