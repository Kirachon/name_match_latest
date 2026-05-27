use crate::error::{AppError, AppResult};
use crate::state::{AppState, DbSession};
use name_matcher::db::schema::discover_table_columns;
use name_matcher::run_service::dto::{
    DbCredentialsDto, DbSessionDto, TableColumnsDto, TableInfoDto,
};
use sqlx::Row;
use std::sync::Arc;
use std::time::Instant;
use tauri::State;
use uuid::Uuid;

#[tauri::command]
pub async fn connect_db(
    creds: DbCredentialsDto,
    state: State<'_, Arc<AppState>>,
) -> AppResult<DbSessionDto> {
    let started = Instant::now();
    let pool = AppState::build_pool(&creds).await?;
    // Quick ping to confirm the pool is live.
    sqlx::query("SELECT 1")
        .fetch_one(&pool)
        .await
        .map_err(|e| AppError::Database(format!("ping failed: {e}")))?;
    let latency_ms = started.elapsed().as_millis() as u64;

    let session_id = Uuid::new_v4().to_string();
    let session = DbSession {
        session_id: session_id.clone(),
        host: creds.host.clone(),
        port: creds.port,
        username: creds.username.clone(),
        database: creds.database.clone(),
        pool,
    };
    state.db.insert(session.clone());
    Ok(session.to_dto(Some(latency_ms)))
}

#[tauri::command]
pub async fn validate_db_credentials(creds: DbCredentialsDto) -> AppResult<u64> {
    let started = Instant::now();
    let pool = AppState::build_pool(&creds).await?;
    let ping = sqlx::query("SELECT 1")
        .fetch_one(&pool)
        .await
        .map_err(|e| AppError::Database(format!("ping failed: {e}")));
    pool.close().await;
    ping?;
    Ok(started.elapsed().as_millis() as u64)
}

#[tauri::command]
pub async fn test_connection(
    session_id: String,
    state: State<'_, Arc<AppState>>,
) -> AppResult<u64> {
    let s = state
        .db
        .get(&session_id)
        .ok_or_else(|| AppError::Validation(format!("Unknown session_id: {session_id}")))?;
    let started = Instant::now();
    sqlx::query("SELECT 1")
        .fetch_one(&s.pool)
        .await
        .map_err(|e| AppError::Database(format!("ping failed: {e}")))?;
    Ok(started.elapsed().as_millis() as u64)
}

#[tauri::command]
pub async fn list_tables(
    session_id: String,
    state: State<'_, Arc<AppState>>,
) -> AppResult<Vec<TableInfoDto>> {
    let s = state
        .db
        .get(&session_id)
        .ok_or_else(|| AppError::Validation(format!("Unknown session_id: {session_id}")))?;
    let rows = sqlx::query(
        "SELECT TABLE_NAME, TABLE_SCHEMA FROM information_schema.TABLES \
         WHERE TABLE_SCHEMA = ? \
         ORDER BY TABLE_NAME",
    )
    .bind(&s.database)
    .fetch_all(&s.pool)
    .await
    .map_err(|e| AppError::Database(format!("list tables: {e}")))?;
    let out = rows
        .into_iter()
        .map(|r| TableInfoDto {
            name: r.try_get::<String, _>("TABLE_NAME").unwrap_or_default(),
            schema: r.try_get::<String, _>("TABLE_SCHEMA").unwrap_or_default(),
            row_count: None,
        })
        .collect();
    Ok(out)
}

#[tauri::command]
pub async fn get_table_columns(
    session_id: String,
    table: String,
    state: State<'_, Arc<AppState>>,
) -> AppResult<TableColumnsDto> {
    let s = state
        .db
        .get(&session_id)
        .ok_or_else(|| AppError::Validation(format!("Unknown session_id: {session_id}")))?;
    let cols = discover_table_columns(&s.pool, &s.database, &table)
        .await
        .map_err(|e| AppError::Database(format!("discover columns for {table}: {e}")))?;

    // Also pull the raw column list for the frontend's hint helper.
    let raw_rows = sqlx::query(
        "SELECT COLUMN_NAME FROM information_schema.COLUMNS \
         WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? \
         ORDER BY ORDINAL_POSITION",
    )
    .bind(&s.database)
    .bind(&table)
    .fetch_all(&s.pool)
    .await
    .map_err(|e| AppError::Database(format!("raw columns: {e}")))?;
    let raw_columns: Vec<String> = raw_rows
        .into_iter()
        .filter_map(|r| r.try_get::<String, _>("COLUMN_NAME").ok())
        .collect();

    Ok(TableColumnsDto {
        has_id: cols.has_id,
        has_uuid: cols.has_uuid,
        has_first_name: cols.has_first_name,
        has_middle_name: cols.has_middle_name,
        has_last_name: cols.has_last_name,
        has_birthdate: cols.has_birthdate,
        has_hh_id: cols.has_hh_id,
        raw_columns,
    })
}

#[tauri::command]
pub async fn get_row_count(
    session_id: String,
    table: String,
    state: State<'_, Arc<AppState>>,
) -> AppResult<u64> {
    let s = state
        .db
        .get(&session_id)
        .ok_or_else(|| AppError::Validation(format!("Unknown session_id: {session_id}")))?;
    if !is_safe_ident(&table) {
        return Err(AppError::Validation(format!(
            "table name contains unsafe characters: {table}"
        )));
    }
    let q = format!("SELECT COUNT(*) AS c FROM `{}`", table);
    let row = sqlx::query(&q)
        .fetch_one(&s.pool)
        .await
        .map_err(|e| AppError::Database(format!("count {table}: {e}")))?;
    let count: i64 = row.try_get("c").unwrap_or(0);
    Ok(count as u64)
}

#[tauri::command]
pub async fn disconnect_db(session_id: String, state: State<'_, Arc<AppState>>) -> AppResult<()> {
    if state.import_jobs.session_has_active(&session_id) {
        return Err(AppError::Validation(
            "cannot disconnect while a CSV import is running for this session".into(),
        ));
    }
    if let Some(s) = state.db.remove(&session_id) {
        s.pool.close().await;
    }
    Ok(())
}

#[tauri::command]
pub fn list_sessions(state: State<'_, Arc<AppState>>) -> Vec<DbSessionDto> {
    state.db.list()
}

pub(crate) fn is_safe_ident(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$')
}
