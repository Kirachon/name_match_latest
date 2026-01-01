use anyhow::{Context, Result, bail};
use sqlx::{MySql, MySqlPool, Row};

use crate::models::{ColumnMapping, Person, TableColumns};

#[derive(Debug, Clone)]
pub enum SqlBind {
    I64(i64),
    Str(String),
}

fn build_select_list(mapping: Option<&ColumnMapping>) -> Result<String> {
    // Validate identifiers to prevent injection; when None, use defaults
    let m = mapping.cloned().unwrap_or_default();
    // Validate each column ident when provided
    fn val(id: &str) -> Result<()> {
        super::schema::validate_ident(id)
    }
    val(&m.id)?;
    val(&m.first_name)?;
    val(&m.last_name)?;
    val(&m.birthdate)?;
    if let Some(ref mid) = m.middle_name {
        val(mid)?;
    }
    if let Some(ref hh) = m.hh_id {
        val(hh)?;
    }
    let mid_sql = if let Some(mid) = m.middle_name.as_ref() {
        format!("`{}` AS middle_name", mid)
    } else {
        "NULL AS middle_name".to_string()
    };
    let uuid_sql = if let Some(ref u) = m.uuid {
        val(u)?;
        format!("`{}` AS uuid", u)
    } else {
        "NULL AS uuid".to_string()
    };
    let hh_sql = if let Some(ref h) = m.hh_id {
        format!("`{}` AS hh_id", h)
    } else {
        "NULL AS hh_id".to_string()
    };
    Ok(format!(
        "`{id}` AS id, {uuid} , `{first}` AS first_name, {mid}, `{last}` AS last_name, DATE(`{bd}`) AS birthdate, {hh}",
        id = m.id,
        uuid = uuid_sql,
        first = m.first_name,
        mid = mid_sql,
        last = m.last_name,
        bd = m.birthdate,
        hh = hh_sql
    ))
}

fn validate_ident(name: &str) -> Result<()> {
    if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        bail!("Invalid identifier: {}", name);
    }
    Ok(())
}

pub async fn discover_table_columns(
    pool: &MySqlPool,
    database: &str,
    table: &str,
) -> Result<TableColumns> {
    validate_ident(database)?;
    validate_ident(table)?;

    let rows = sqlx::query(
        r#"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?"#,
    )
    .bind(database)
    .bind(table)
    .fetch_all(pool)
    .await
    .with_context(|| format!("Failed to query columns for {}.{}", database, table))?;

    // Fallback: some MySQL setups (especially on Windows hosts with case-sensitive filesystems)
    // may return zero rows from INFORMATION_SCHEMA even when the table exists. If that happens,
    // run a DESCRIBE fallback to keep execution unblocked.
    let rows = if rows.is_empty() {
        sqlx::query(&format!("DESCRIBE `{}`.`{}`", database, table))
            .fetch_all(pool)
            .await
            .with_context(|| format!("DESCRIBE fallback failed for {}.{}", database, table))?
    } else {
        rows
    };

    let mut cols = TableColumns {
        has_id: false,
        has_uuid: false,
        has_first_name: false,
        has_middle_name: false,
        has_last_name: false,
        has_birthdate: false,
        has_hh_id: false,
    };
    for r in rows {
        let name: String = r.try_get("COLUMN_NAME")?;
        match name.as_str() {
            "id" => cols.has_id = true,
            "uuid" => cols.has_uuid = true,
            "first_name" => cols.has_first_name = true,
            "middle_name" => cols.has_middle_name = true,
            "last_name" => cols.has_last_name = true,
            "birthdate" => cols.has_birthdate = true,
            "hh_id" => cols.has_hh_id = true,
            _ => {}
        }
    }
    Ok(cols)
}

/// Discover all column names for a table (not just standard ones)
pub async fn get_all_table_columns(
    pool: &MySqlPool,
    database: &str,
    table: &str,
) -> Result<Vec<String>> {
    validate_ident(database)?;
    validate_ident(table)?;

    let rows = sqlx::query(
        r#"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
            ORDER BY ORDINAL_POSITION"#,
    )
    .bind(database)
    .bind(table)
    .fetch_all(pool)
    .await
    .with_context(|| format!("Failed to query all columns for {}.{}", database, table))?;

    let mut columns = Vec::new();
    for r in rows {
        let name: String = r.try_get("COLUMN_NAME")?;
        columns.push(name);
    }
    Ok(columns)
}

pub async fn get_person_rows(pool: &MySqlPool, table: &str) -> Result<Vec<Person>> {
    use std::collections::HashMap;

    validate_ident(table)?;

    // Get database name from pool connection string
    let database = extract_database_from_pool(pool).await?;

    // Try to get all columns (for dynamic field support)
    match get_all_table_columns(pool, &database, table).await {
        Ok(all_columns) if !all_columns.is_empty() => {
            // Dynamic query with all columns
            let standard_cols = [
                "id",
                "uuid",
                "first_name",
                "middle_name",
                "last_name",
                "birthdate",
                "hh_id",
            ];

            // Build SELECT clause
            let mut select_parts = Vec::new();
            for col in &all_columns {
                if col == "birthdate" {
                    select_parts.push(format!("DATE(`{}`) AS birthdate", col));
                } else {
                    select_parts.push(format!("`{}`", col));
                }
            }
            let select_clause = select_parts.join(", ");
            let sql = format!("SELECT {} FROM `{}`", select_clause, table);

            // Fetch rows dynamically
            let rows = sqlx::query(&sql)
                .fetch_all(pool)
                .await
                .with_context(|| format!("Failed to fetch all columns from {}", table))?;

            let mut persons = Vec::new();
            for row in rows {
                // Try to read hh_id as i64 first, then convert to String
                let hh_id: Option<String> = row
                    .try_get::<Option<i64>, _>("hh_id")
                    .ok()
                    .flatten()
                    .map(|v| v.to_string())
                    .or_else(|| row.try_get::<Option<String>, _>("hh_id").ok().flatten());

                let mut person = Person {
                    id: row.try_get("id").unwrap_or(0),
                    uuid: row.try_get("uuid").ok(),
                    first_name: row.try_get("first_name").ok(),
                    middle_name: row.try_get("middle_name").ok(),
                    last_name: row.try_get("last_name").ok(),
                    birthdate: row.try_get("birthdate").ok(),
                    hh_id,
                    extra_fields: HashMap::new(),
                };

                // Populate extra fields (columns not in standard set)
                for col in &all_columns {
                    if !standard_cols.contains(&col.as_str()) {
                        if let Ok(value) = row.try_get::<String, _>(col.as_str()) {
                            person.extra_fields.insert(col.clone(), value);
                        } else if let Ok(value) = row.try_get::<i64, _>(col.as_str()) {
                            person.extra_fields.insert(col.clone(), value.to_string());
                        } else if let Ok(value) = row.try_get::<f64, _>(col.as_str()) {
                            person.extra_fields.insert(col.clone(), value.to_string());
                        } else if let Ok(value) = row.try_get::<bool, _>(col.as_str()) {
                            person.extra_fields.insert(col.clone(), value.to_string());
                        } else if let Ok(Some(value)) =
                            row.try_get::<Option<String>, _>(col.as_str())
                        {
                            person.extra_fields.insert(col.clone(), value);
                        }
                        // If all conversions fail, skip this column (NULL or unsupported type)
                    }
                }

                persons.push(person);
            }

            Ok(persons)
        }
        _ => {
            // Fallback to standard query if schema discovery fails
            let sql1 = format!(
                "SELECT id, uuid, first_name, middle_name, last_name, DATE(birthdate) AS birthdate, hh_id AS hh_id FROM `{}`",
                table
            );
            match sqlx::query_as::<MySql, Person>(&sql1).fetch_all(pool).await {
                Ok(rows) => Ok(rows),
                Err(e) => {
                    let unknown_uuid = matches!(e, sqlx::Error::Database(ref db) if db.message().contains("Unknown column") && db.message().contains("uuid"));
                    let unknown_hh = matches!(e, sqlx::Error::Database(ref db) if db.message().contains("Unknown column") && db.message().contains("hh_id"));
                    let (uuid_sel, hh_sel) = (
                        if unknown_uuid { "NULL AS uuid" } else { "uuid" },
                        if unknown_hh {
                            "NULL AS hh_id"
                        } else {
                            "hh_id AS hh_id"
                        },
                    );
                    if unknown_uuid || unknown_hh {
                        let sql2 = format!(
                            "SELECT id, {uuid_sel}, first_name, middle_name, last_name, DATE(birthdate) AS birthdate, {hh_sel} FROM `{table}`",
                            uuid_sel = uuid_sel,
                            hh_sel = hh_sel,
                            table = table
                        );
                        let rows: Vec<Person> = sqlx::query_as::<MySql, Person>(&sql2)
                            .fetch_all(pool)
                            .await
                            .with_context(|| {
                                format!("Failed to fetch rows from {} (uuid/hh_id fallback)", table)
                            })?;
                        Ok(rows)
                    } else {
                        Err(e).with_context(|| format!("Failed to fetch rows from {}", table))
                    }
                }
            }
        }
    }
}

/// Extract database name from MySqlPool by querying SELECT DATABASE()
async fn extract_database_from_pool(pool: &MySqlPool) -> Result<String> {
    let row = sqlx::query("SELECT DATABASE() as db")
        .fetch_one(pool)
        .await
        .context("Failed to query current database name")?;
    let db_name: String = row
        .try_get("db")
        .context("Failed to get database name from query result")?;
    Ok(db_name)
}

/// Fetch person rows with ALL columns from the table (including extra fields beyond standard schema)
pub async fn get_person_rows_with_all_columns(
    pool: &MySqlPool,
    database: &str,
    table: &str,
) -> Result<Vec<Person>> {
    use std::collections::HashMap;

    validate_ident(table)?;

    // Get all column names
    let all_columns = get_all_table_columns(pool, database, table).await?;

    // Standard columns we expect
    let standard_cols = [
        "id",
        "uuid",
        "first_name",
        "middle_name",
        "last_name",
        "birthdate",
        "hh_id",
    ];

    // Build SELECT clause
    let mut select_parts = Vec::new();
    for col in &all_columns {
        if col == "birthdate" {
            select_parts.push(format!("DATE(`{}`) AS birthdate", col));
        } else {
            select_parts.push(format!("`{}`", col));
        }
    }
    let select_clause = select_parts.join(", ");

    let sql = format!("SELECT {} FROM `{}`", select_clause, table);

    // Fetch rows dynamically
    let rows = sqlx::query(&sql)
        .fetch_all(pool)
        .await
        .with_context(|| format!("Failed to fetch all columns from {}", table))?;

    let mut persons = Vec::new();
    for row in rows {
        // Try to read hh_id as i64 first, then convert to String
        let hh_id: Option<String> = row
            .try_get::<Option<i64>, _>("hh_id")
            .ok()
            .flatten()
            .map(|v| v.to_string())
            .or_else(|| row.try_get::<Option<String>, _>("hh_id").ok().flatten());

        let mut person = Person {
            id: row.try_get("id").unwrap_or(0),
            uuid: row.try_get("uuid").ok(),
            first_name: row.try_get("first_name").ok(),
            middle_name: row.try_get("middle_name").ok(),
            last_name: row.try_get("last_name").ok(),
            birthdate: row.try_get("birthdate").ok(),
            hh_id,
            extra_fields: HashMap::new(),
        };

        // Populate extra fields (columns not in standard set)
        for col in &all_columns {
            if !standard_cols.contains(&col.as_str()) {
                if let Ok(value) = row.try_get::<String, _>(col.as_str()) {
                    person.extra_fields.insert(col.clone(), value);
                } else if let Ok(value) = row.try_get::<i64, _>(col.as_str()) {
                    person.extra_fields.insert(col.clone(), value.to_string());
                } else if let Ok(value) = row.try_get::<f64, _>(col.as_str()) {
                    person.extra_fields.insert(col.clone(), value.to_string());
                } else if let Ok(value) = row.try_get::<bool, _>(col.as_str()) {
                    person.extra_fields.insert(col.clone(), value.to_string());
                } else if let Ok(Some(value)) = row.try_get::<Option<String>, _>(col.as_str()) {
                    person.extra_fields.insert(col.clone(), value);
                }
                // If all conversions fail, skip this column (NULL or unsupported type)
            }
        }

        persons.push(person);
    }

    Ok(persons)
}

/// Convenience wrapper: fetch all columns for a table, inferring the current database
pub async fn get_person_rows_all_columns(pool: &MySqlPool, table: &str) -> Result<Vec<Person>> {
    validate_ident(table)?;
    let db = extract_database_from_pool(pool).await?;
    get_person_rows_with_all_columns(pool, &db, table).await
}

/// Fetch a chunk of rows with ALL columns (including dynamic extra fields)
pub async fn fetch_person_rows_chunk_all_columns(
    pool: &MySqlPool,
    table: &str,
    offset: i64,
    limit: i64,
) -> Result<Vec<Person>> {
    use std::collections::HashMap;
    validate_ident(table)?;
    let database = extract_database_from_pool(pool).await?;
    let all_columns = get_all_table_columns(pool, &database, table).await?;
    let standard_cols = [
        "id",
        "uuid",
        "first_name",
        "middle_name",
        "last_name",
        "birthdate",
        "hh_id",
    ];

    let mut select_parts = Vec::new();
    for col in &all_columns {
        if col == "birthdate" {
            select_parts.push(format!("DATE(`{}`) AS birthdate", col));
        } else {
            select_parts.push(format!("`{}`", col));
        }
    }
    let select_clause = select_parts.join(", ");
    let sql = format!(
        "SELECT {} FROM `{}` ORDER BY id LIMIT ? OFFSET ?",
        select_clause, table
    );
    let rows = sqlx::query(&sql)
        .bind(limit)
        .bind(offset)
        .fetch_all(pool)
        .await
        .with_context(|| {
            format!(
                "Failed to fetch chunk (all columns) from {} (offset {}, limit {})",
                table, offset, limit
            )
        })?;

    let mut persons = Vec::new();
    for row in rows {
        // Try to read hh_id as i64 first, then convert to String
        let hh_id: Option<String> = row
            .try_get::<Option<i64>, _>("hh_id")
            .ok()
            .flatten()
            .map(|v| v.to_string())
            .or_else(|| row.try_get::<Option<String>, _>("hh_id").ok().flatten());

        let mut person = Person {
            id: row.try_get("id").unwrap_or(0),
            uuid: row.try_get("uuid").ok(),
            first_name: row.try_get("first_name").ok(),
            middle_name: row.try_get("middle_name").ok(),
            last_name: row.try_get("last_name").ok(),
            birthdate: row.try_get("birthdate").ok(),
            hh_id,
            extra_fields: HashMap::new(),
        };
        for col in &all_columns {
            if !standard_cols.contains(&col.as_str()) {
                if let Ok(value) = row.try_get::<String, _>(col.as_str()) {
                    person.extra_fields.insert(col.clone(), value);
                } else if let Ok(value) = row.try_get::<i64, _>(col.as_str()) {
                    person.extra_fields.insert(col.clone(), value.to_string());
                } else if let Ok(value) = row.try_get::<f64, _>(col.as_str()) {
                    person.extra_fields.insert(col.clone(), value.to_string());
                } else if let Ok(value) = row.try_get::<bool, _>(col.as_str()) {
                    person.extra_fields.insert(col.clone(), value.to_string());
                } else if let Ok(Some(value)) = row.try_get::<Option<String>, _>(col.as_str()) {
                    person.extra_fields.insert(col.clone(), value);
                }
            }
        }
        persons.push(person);
    }
    Ok(persons)
}

/// Get total member counts per Table 2 household key (hh_id fallback to id)
pub async fn get_household_totals_map(
    pool: &MySqlPool,
    table: &str,
) -> Result<std::collections::BTreeMap<String, usize>> {
    validate_ident(table)?;
    // Use IFNULL(hh_id, CAST(id AS CHAR)) to build a stable string key
    let sql = format!(
        "SELECT IFNULL(CAST(hh_id AS CHAR), CAST(id AS CHAR)) AS hh_key, COUNT(*) AS cnt FROM `{}` GROUP BY IFNULL(CAST(hh_id AS CHAR), CAST(id AS CHAR))",
        table
    );
    let rows = sqlx::query(&sql)
        .fetch_all(pool)
        .await
        .with_context(|| format!("Failed to compute household totals for {}", table))?;
    let mut out = std::collections::BTreeMap::new();
    for r in rows {
        let key: String = r.try_get("hh_key").unwrap_or_default();
        let cnt: i64 = r.try_get("cnt").unwrap_or(0);
        if cnt > 0 {
            out.insert(key, cnt as usize);
        }
    }
    Ok(out)
}

/// Get total member counts per Table 1 UUID (skip NULL UUIDs)
pub async fn get_uuid_totals_map(
    pool: &MySqlPool,
    table: &str,
) -> Result<std::collections::BTreeMap<String, usize>> {
    validate_ident(table)?;
    let sql = format!(
        "SELECT uuid AS uuid_key, COUNT(*) AS cnt FROM `{}` WHERE uuid IS NOT NULL GROUP BY uuid",
        table
    );
    let rows = sqlx::query(&sql)
        .fetch_all(pool)
        .await
        .with_context(|| format!("Failed to compute UUID totals for {}", table))?;
    let mut out = std::collections::BTreeMap::new();
    for r in rows {
        let key: String = r.try_get("uuid_key").unwrap_or_default();
        let cnt: i64 = r.try_get("cnt").unwrap_or(0);
        if !key.is_empty() && cnt > 0 {
            out.insert(key, cnt as usize);
        }
    }
    Ok(out)
}

pub async fn get_person_count(pool: &MySqlPool, table: &str) -> Result<i64> {
    validate_ident(table)?;
    let sql = format!("SELECT COUNT(*) as cnt FROM `{}`", table);
    let row = sqlx::query(&sql).fetch_one(pool).await?;
    let cnt: i64 = row.try_get("cnt")?;
    Ok(cnt)
}

/// Fast row count estimation using EXPLAIN. Falls back to exact COUNT(*) on failure.
pub async fn get_person_count_fast(pool: &MySqlPool, table: &str) -> Result<i64> {
    validate_ident(table)?;
    // MySQL EXPLAIN returns an estimated row count in the `rows` column
    let sql = format!("EXPLAIN SELECT * FROM `{}`", table);
    if let Ok(row) = sqlx::query(&sql).fetch_one(pool).await {
        // Try both lowercase and uppercase variants depending on driver mapping
        if let Ok(v) = row.try_get::<i64, _>("rows") {
            return Ok(v);
        }
        if let Ok(v) = row.try_get::<i32, _>("rows") {
            return Ok(v as i64);
        }
        if let Ok(v) = row.try_get::<u64, _>("rows") {
            return Ok(v as i64);
        }
    }
    // Fallback to exact count if EXPLAIN not available/privileged or does not expose `rows`
    get_person_count(pool, table).await
}

/// Return the current MAX(id) for snapshot-style streaming windows.
pub async fn get_max_id(pool: &MySqlPool, table: &str) -> Result<i64> {
    validate_ident(table)?;
    let sql = format!("SELECT COALESCE(MAX(id), 0) as max_id FROM `{}`", table);
    let row = sqlx::query(&sql).fetch_one(pool).await?;
    let max_id: i64 = row.try_get("max_id")?;
    Ok(max_id)
}

// --- Flexible/mapped selection helpers and WHERE-aware fetchers ---
#[allow(dead_code)]
pub async fn get_person_rows_mapped(
    pool: &MySqlPool,
    table: &str,
    mapping: Option<&ColumnMapping>,
) -> Result<Vec<Person>> {
    super::schema::validate_ident(table)?;
    let select = build_select_list(mapping)?;
    let sql = format!(
        "SELECT {select} FROM `{table}`",
        select = select,
        table = table
    );
    let rows: Vec<Person> = sqlx::query_as::<MySql, Person>(&sql)
        .fetch_all(pool)
        .await
        .with_context(|| format!("Failed to fetch rows from {} (mapped)", table))?;
    Ok(rows)
}

pub async fn get_person_count_where(
    pool: &MySqlPool,
    table: &str,
    where_sql: &str,
    binds: &[SqlBind],
) -> Result<i64> {
    super::schema::validate_ident(table)?;
    let sql = format!(
        "SELECT COUNT(*) as cnt FROM `{}` WHERE {}",
        table, where_sql
    );
    let mut q = sqlx::query(&sql);
    for b in binds {
        q = match b {
            SqlBind::I64(v) => q.bind(*v),
            SqlBind::Str(s) => q.bind(s),
        };
    }
    let row = q.fetch_one(pool).await?;
    let cnt: i64 = row.try_get("cnt")?;
    Ok(cnt)
}

/// Return MAX(id) for a filtered subset (for snapshot/keyset bounds).
pub async fn get_max_id_where(
    pool: &MySqlPool,
    table: &str,
    where_sql: &str,
    binds: &[SqlBind],
) -> Result<i64> {
    super::schema::validate_ident(table)?;
    let sql = format!(
        "SELECT COALESCE(MAX(id), 0) as max_id FROM `{}` WHERE {}",
        table, where_sql
    );
    let mut q = sqlx::query(&sql);
    for b in binds {
        q = match b {
            SqlBind::I64(v) => q.bind(*v),
            SqlBind::Str(s) => q.bind(s),
        };
    }
    let row = q.fetch_one(pool).await?;
    let max_id: i64 = row.try_get("max_id")?;
    Ok(max_id)
}

pub async fn get_person_rows_where(
    pool: &MySqlPool,
    table: &str,
    where_sql: &str,
    binds: &[SqlBind],
    mapping: Option<&ColumnMapping>,
) -> Result<Vec<Person>> {
    super::schema::validate_ident(table)?;
    let select = build_select_list(mapping)?;
    let sql = format!("SELECT {select} FROM `{table}` WHERE {where}", select=select, table=table, where=where_sql);
    let mut q = sqlx::query_as::<MySql, Person>(&sql);
    for b in binds {
        q = match b {
            SqlBind::I64(v) => q.bind(*v),
            SqlBind::Str(s) => q.bind(s),
        };
    }
    match q.fetch_all(pool).await {
        Ok(rows) => Ok(rows),
        Err(e) => {
            let unknown_uuid = matches!(e, sqlx::Error::Database(ref db) if db.message().contains("Unknown column") && db.message().contains("uuid"));
            let unknown_hh = matches!(e, sqlx::Error::Database(ref db) if db.message().contains("Unknown column") && db.message().contains("hh_id"));
            if unknown_uuid || unknown_hh {
                let mut m2 = mapping.cloned().unwrap_or_default();
                if unknown_uuid {
                    m2.uuid = None;
                }
                if unknown_hh {
                    m2.hh_id = None;
                }
                let select2 = build_select_list(Some(&m2))?;
                let sql2 = format!("SELECT {select} FROM `{table}` WHERE {where}", select=select2, table=table, where=where_sql);
                let mut q2 = sqlx::query_as::<MySql, Person>(&sql2);
                for b in binds {
                    q2 = match b {
                        SqlBind::I64(v) => q2.bind(*v),
                        SqlBind::Str(s) => q2.bind(s),
                    };
                }
                let rows = q2.fetch_all(pool).await.with_context(|| {
                    format!(
                        "Failed to fetch rows from {} with filter (uuid/hh_id fallback)",
                        table
                    )
                })?;
                Ok(rows)
            } else {
                Err(e).with_context(|| format!("Failed to fetch rows from {} with filter", table))
            }
        }
    }
}

pub async fn fetch_person_rows_chunk_where(
    pool: &MySqlPool,
    table: &str,
    offset: i64,
    limit: i64,
    where_sql: &str,
    binds: &[SqlBind],
    mapping: Option<&ColumnMapping>,
) -> Result<Vec<Person>> {
    super::schema::validate_ident(table)?;
    let select = build_select_list(mapping)?;
    let sql = format!("SELECT {select} FROM `{table}` WHERE {where} ORDER BY id LIMIT ? OFFSET ?", select=select, table=table, where=where_sql);
    let mut q = sqlx::query_as::<MySql, Person>(&sql);
    for b in binds {
        q = match b {
            SqlBind::I64(v) => q.bind(*v),
            SqlBind::Str(s) => q.bind(s),
        };
    }
    let q = q.bind(limit).bind(offset);
    match q.fetch_all(pool).await {
        Ok(rows) => Ok(rows),
        Err(e) => {
            let unknown_uuid = matches!(e, sqlx::Error::Database(ref db) if db.message().contains("Unknown column") && db.message().contains("uuid"));
            let unknown_hh = matches!(e, sqlx::Error::Database(ref db) if db.message().contains("Unknown column") && db.message().contains("hh_id"));
            if unknown_uuid || unknown_hh {
                let mut m2 = mapping.cloned().unwrap_or_default();
                if unknown_uuid {
                    m2.uuid = None;
                }
                if unknown_hh {
                    m2.hh_id = None;
                }
                let select2 = build_select_list(Some(&m2))?;
                let sql2 = format!("SELECT {select} FROM `{table}` WHERE {where} ORDER BY id LIMIT ? OFFSET ?", select=select2, table=table, where=where_sql);
                let mut q2 = sqlx::query_as::<MySql, Person>(&sql2);
                for b in binds {
                    q2 = match b {
                        SqlBind::I64(v) => q2.bind(*v),
                        SqlBind::Str(s) => q2.bind(s),
                    };
                }
                let rows: Vec<Person> = q2.bind(limit).bind(offset).fetch_all(pool).await
                    .with_context(|| format!("Failed to fetch chunk from {} (offset {}, limit {}) with filter and uuid/hh_id fallback", table, offset, limit))?;
                Ok(rows)
            } else {
                Err(e).with_context(|| {
                    format!(
                        "Failed to fetch chunk from {} (offset {}, limit {}) with filter",
                        table, offset, limit
                    )
                })
            }
        }
    }
}

/// Keyset pagination with WHERE and optional column mapping.
pub async fn fetch_person_rows_chunk_where_keyset(
    pool: &MySqlPool,
    table: &str,
    last_id: i64,
    limit: i64,
    where_sql: &str,
    binds: &[SqlBind],
    mapping: Option<&ColumnMapping>,
    watermark_id: Option<i64>,
) -> Result<Vec<Person>> {
    super::schema::validate_ident(table)?;
    let select = build_select_list(mapping)?;
    let sql = if watermark_id.is_some() {
        format!(
            "SELECT {select} FROM `{table}` WHERE id > ? AND id <= ? AND ({where}) ORDER BY id LIMIT ?",
            select = select,
            table = table,
            where = where_sql
        )
    } else {
        format!(
            "SELECT {select} FROM `{table}` WHERE id > ? AND ({where}) ORDER BY id LIMIT ?",
            select = select,
            table = table,
            where = where_sql
        )
    };
    let mut q = sqlx::query_as::<MySql, Person>(&sql).bind(last_id);
    if let Some(w) = watermark_id {
        q = q.bind(w);
    }
    for b in binds {
        q = match b {
            SqlBind::I64(v) => q.bind(*v),
            SqlBind::Str(s) => q.bind(s),
        };
    }
    q = q.bind(limit);
    match q.fetch_all(pool).await {
        Ok(rows) => Ok(rows),
        Err(e) => {
            let unknown_uuid = matches!(e, sqlx::Error::Database(ref db) if db.message().contains("Unknown column") && db.message().contains("uuid"));
            let unknown_hh = matches!(e, sqlx::Error::Database(ref db) if db.message().contains("Unknown column") && db.message().contains("hh_id"));
            if unknown_uuid || unknown_hh {
                let mut m2 = mapping.cloned().unwrap_or_default();
                if unknown_uuid {
                    m2.uuid = None;
                }
                if unknown_hh {
                    m2.hh_id = None;
                }
                let select2 = build_select_list(Some(&m2))?;
                let mut sql2 = format!(
                    "SELECT {select} FROM `{table}` WHERE id > ? AND ({where}) ORDER BY id LIMIT ?",
                    select = select2,
                    table = table,
                    where = where_sql
                );
                if watermark_id.is_some() {
                    sql2 = sql2.replace("WHERE id > ?", "WHERE id > ? AND id <= ? AND (");
                    sql2 = sql2.replacen("AND ((", "AND (", 1);
                }
                let mut q2 = sqlx::query_as::<MySql, Person>(&sql2).bind(last_id);
                if let Some(w) = watermark_id {
                    q2 = q2.bind(w);
                }
                for b in binds {
                    q2 = match b {
                        SqlBind::I64(v) => q2.bind(*v),
                        SqlBind::Str(s) => q2.bind(s),
                    };
                }
                let rows: Vec<Person> = q2
                    .bind(limit)
                    .fetch_all(pool)
                    .await
                    .with_context(|| {
                        format!(
                            "Failed to fetch chunk from {} (last_id {}, limit {}) with filter and uuid/hh_id fallback",
                            table, last_id, limit
                        )
                    })?;
                Ok(rows)
            } else {
                Err(e).with_context(|| {
                    format!(
                        "Failed to fetch chunk from {} (last_id {}, limit {}) with filter",
                        table, last_id, limit
                    )
                })
            }
        }
    }
}

pub async fn fetch_person_rows_chunk(
    pool: &MySqlPool,
    table: &str,
    offset: i64,
    limit: i64,
) -> Result<Vec<Person>> {
    validate_ident(table)?;
    let sql1 = format!(
        "SELECT id, uuid, first_name, middle_name, last_name, DATE(birthdate) AS birthdate, hh_id AS hh_id FROM `{}` ORDER BY id LIMIT ? OFFSET ?",
        table
    );
    let q1 = sqlx::query_as::<MySql, Person>(&sql1)
        .bind(limit)
        .bind(offset);
    match q1.fetch_all(pool).await {
        Ok(rows) => Ok(rows),
        Err(e) => {
            let unknown_uuid = matches!(e, sqlx::Error::Database(ref db) if db.message().contains("Unknown column") && db.message().contains("uuid"));
            let unknown_hh = matches!(e, sqlx::Error::Database(ref db) if db.message().contains("Unknown column") && db.message().contains("hh_id"));
            if unknown_uuid || unknown_hh {
                let (uuid_sel, hh_sel) = (
                    if unknown_uuid { "NULL AS uuid" } else { "uuid" },
                    if unknown_hh {
                        "NULL AS hh_id"
                    } else {
                        "hh_id AS hh_id"
                    },
                );
                let sql2 = format!(
                    "SELECT id, {uuid_sel}, first_name, middle_name, last_name, DATE(birthdate) AS birthdate, {hh_sel} FROM `{table}` ORDER BY id LIMIT ? OFFSET ?",
                    uuid_sel = uuid_sel,
                    hh_sel = hh_sel,
                    table = table
                );
                let rows: Vec<Person> = sqlx::query_as::<MySql, Person>(&sql2)
                    .bind(limit)
                    .bind(offset)
                    .fetch_all(pool)
                    .await
                    .with_context(|| format!("Failed to fetch chunk from {} (offset {}, limit {}) with uuid/hh_id fallback", table, offset, limit))?;
                Ok(rows)
            } else {
                Err(e).with_context(|| {
                    format!(
                        "Failed to fetch chunk from {} (offset {}, limit {})",
                        table, offset, limit
                    )
                })
            }
        }
    }
}

/// Keyset pagination variant: fetch rows after `last_id`, optionally bounded by `watermark_id`.
pub async fn fetch_person_rows_chunk_keyset(
    pool: &MySqlPool,
    table: &str,
    last_id: i64,
    limit: i64,
    watermark_id: Option<i64>,
) -> Result<Vec<Person>> {
    validate_ident(table)?;
    let mut sql = String::from(
        "SELECT id, uuid, first_name, middle_name, last_name, DATE(birthdate) AS birthdate, hh_id AS hh_id \
         FROM `{table}` WHERE id > ?",
    );
    if watermark_id.is_some() {
        sql.push_str(" AND id <= ?");
    }
    sql.push_str(" ORDER BY id LIMIT ?");
    let sql = sql.replace("{table}", table);
    let mut q = sqlx::query_as::<MySql, Person>(&sql).bind(last_id);
    if let Some(w) = watermark_id {
        q = q.bind(w);
    }
    q = q.bind(limit);
    match q.fetch_all(pool).await {
        Ok(rows) => Ok(rows),
        Err(e) => {
            let unknown_uuid = matches!(e, sqlx::Error::Database(ref db) if db.message().contains("Unknown column") && db.message().contains("uuid"));
            let unknown_hh = matches!(e, sqlx::Error::Database(ref db) if db.message().contains("Unknown column") && db.message().contains("hh_id"));
            if unknown_uuid || unknown_hh {
                let (uuid_sel, hh_sel) = (
                    if unknown_uuid { "NULL AS uuid" } else { "uuid" },
                    if unknown_hh {
                        "NULL AS hh_id"
                    } else {
                        "hh_id AS hh_id"
                    },
                );
                let mut sql2 = format!(
                    "SELECT id, {uuid_sel}, first_name, middle_name, last_name, DATE(birthdate) AS birthdate, {hh_sel} FROM `{table}` WHERE id > ?",
                    uuid_sel = uuid_sel,
                    hh_sel = hh_sel,
                    table = table
                );
                if watermark_id.is_some() {
                    sql2.push_str(" AND id <= ?");
                }
                sql2.push_str(" ORDER BY id LIMIT ?");
                let mut q2 = sqlx::query_as::<MySql, Person>(&sql2).bind(last_id);
                if let Some(w) = watermark_id {
                    q2 = q2.bind(w);
                }
                q2 = q2.bind(limit);
                let rows: Vec<Person> = q2
                    .fetch_all(pool)
                    .await
                    .with_context(|| format!("Failed to fetch chunk from {} (last_id {}, limit {}) with uuid/hh_id fallback", table, last_id, limit))?;
                Ok(rows)
            } else {
                Err(e).with_context(|| {
                    format!(
                        "Failed to fetch chunk from {} (last_id {}, limit {})",
                        table, last_id, limit
                    )
                })
            }
        }
    }
}

/// Keyset pagination: fetch all columns after `last_id`, optional `watermark_id` upper bound.
pub async fn fetch_person_rows_chunk_all_columns_keyset(
    pool: &MySqlPool,
    table: &str,
    last_id: i64,
    limit: i64,
    watermark_id: Option<i64>,
) -> Result<Vec<Person>> {
    use std::collections::HashMap;
    validate_ident(table)?;
    let database = extract_database_from_pool(pool).await?;
    let all_columns = get_all_table_columns(pool, &database, table).await?;
    let standard_cols = [
        "id",
        "uuid",
        "first_name",
        "middle_name",
        "last_name",
        "birthdate",
        "hh_id",
    ];

    let mut select_parts = Vec::new();
    for col in &all_columns {
        if col == "birthdate" {
            select_parts.push(format!("DATE(`{}`) AS birthdate", col));
        } else {
            select_parts.push(format!("`{}`", col));
        }
    }
    let select_clause = select_parts.join(", ");
    let sql = if watermark_id.is_some() {
        format!(
            "SELECT {select} FROM `{table}` WHERE id > ? AND id <= ? ORDER BY id LIMIT ?",
            select = select_clause,
            table = table
        )
    } else {
        format!(
            "SELECT {select} FROM `{table}` WHERE id > ? ORDER BY id LIMIT ?",
            select = select_clause,
            table = table
        )
    };
    let mut q = sqlx::query(&sql).bind(last_id);
    if let Some(w) = watermark_id {
        q = q.bind(w);
    }
    q = q.bind(limit);

    let rows = q.fetch_all(pool).await.with_context(|| {
        format!(
            "Failed to fetch chunk (all columns) from {} (last_id {}, limit {})",
            table, last_id, limit
        )
    })?;

    let mut persons = Vec::new();
    for row in rows {
        // Try to read hh_id as i64 first, then convert to String
        let hh_id: Option<String> = row
            .try_get::<Option<i64>, _>("hh_id")
            .ok()
            .flatten()
            .map(|v| v.to_string())
            .or_else(|| row.try_get::<Option<String>, _>("hh_id").ok().flatten());

        let mut person = Person {
            id: row.try_get("id").unwrap_or(0),
            uuid: row.try_get("uuid").ok(),
            first_name: row.try_get("first_name").ok(),
            middle_name: row.try_get("middle_name").ok(),
            last_name: row.try_get("last_name").ok(),
            birthdate: row.try_get("birthdate").ok(),
            hh_id,
            extra_fields: HashMap::new(),
        };
        for col in &all_columns {
            if !standard_cols.contains(&col.as_str()) {
                if let Ok(value) = row.try_get::<String, _>(col.as_str()) {
                    person.extra_fields.insert(col.clone(), value);
                } else if let Ok(value) = row.try_get::<i64, _>(col.as_str()) {
                    person.extra_fields.insert(col.clone(), value.to_string());
                } else if let Ok(value) = row.try_get::<f64, _>(col.as_str()) {
                    person.extra_fields.insert(col.clone(), value.to_string());
                } else if let Ok(value) = row.try_get::<bool, _>(col.as_str()) {
                    person.extra_fields.insert(col.clone(), value.to_string());
                } else if let Ok(Some(value)) = row.try_get::<Option<String>, _>(col.as_str()) {
                    person.extra_fields.insert(col.clone(), value);
                }
            }
        }
        persons.push(person);
    }

    Ok(persons)
}
