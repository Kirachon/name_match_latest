//! Staging-table CSV import for million-row scale (bounded RAM, safe Replace).

use crate::loaders::csv_loader::{CsvPreviewRequestDto, stream_csv_people_batches};
use crate::models::Person;
use crate::run_service::CancelToken;
use crate::run_service::dto::{
    CsvImportDryRunResultDto, CsvImportDuplicateBehaviorDto, CsvImportDuplicateKeyDto,
    CsvImportDuplicateProbeStatusDto, CsvImportIdBehaviorDto, CsvImportInvalidRowDto,
    CsvImportJobDto, CsvImportJobPhaseDto, CsvImportLoadMethodDto, CsvImportRequestDto,
    CsvImportTargetModeDto,
};
use anyhow::{Context, Result, bail};
use sqlx::{MySql, MySqlPool, QueryBuilder, Row};
use uuid::Uuid;

use super::{
    apply_id_policy, create_indexes, insert_batch, planned_columns, planned_indexes, row_count,
    table_exists, validate_ident, validate_request, validate_target_mode,
};

const STAGING_BATCH_DEFAULT: usize = 10_000;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StagingPlan {
    pub job_id: String,
    pub staging_table: String,
    pub session_id: String,
    pub plan_hash: String,
    pub source_path: String,
    pub source_size: u64,
    pub source_mtime: u64,
}

pub fn staging_table_name(target_table: &str, job_id: &str) -> Result<String> {
    let safe_job: String = job_id
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .take(12)
        .collect();
    if safe_job.len() < 8 {
        bail!("invalid staging job id");
    }
    const MAX_IDENT: usize = 64;
    let suffix = format!("__stg_{safe_job}");
    let max_base = MAX_IDENT.saturating_sub(suffix.len());
    if max_base < 1 {
        bail!("staging table name cannot be derived");
    }
    let base = if target_table.len() > max_base {
        &target_table[..max_base]
    } else {
        target_table
    };
    let name = format!("{base}{suffix}");
    validate_ident(&name)?;
    Ok(name)
}

fn old_table_name(target_table: &str, job_id: &str) -> Result<String> {
    let safe_job = job_id
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect::<String>();
    let name = format!("{target_table}__old_{safe_job}");
    validate_ident(&name)?;
    Ok(name)
}

fn file_metadata(path: &str) -> Result<(u64, u64)> {
    let meta = std::fs::metadata(path).with_context(|| format!("stat {}", path))?;
    Ok((
        meta.len(),
        meta.modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0),
    ))
}

pub async fn dry_run_staged(
    pool: &MySqlPool,
    req: &CsvImportRequestDto,
) -> Result<(CsvImportDryRunResultDto, StagingPlan)> {
    validate_request(req)?;
    let table_exists_flag = table_exists(pool, &req.target.database, &req.target.table).await?;
    validate_target_mode(req, table_exists_flag)?;

    if table_exists_flag && matches!(req.target.mode, CsvImportTargetModeDto::Append) {
        super::validate_existing_table(pool, &req.target.database, &req.target.table, req).await?;
    }

    let job_id = Uuid::new_v4().to_string();
    let staging_table = staging_table_name(&req.target.table, &job_id)?;
    let (source_size, source_mtime) = file_metadata(&req.file.path)?;
    let plan_hash = super::compute_plan_hash(req);

    drop_staging_if_exists(pool, &req.target.database, &staging_table).await?;
    create_staging_table(pool, req, &staging_table).await?;

    let cancel = CancelToken::new();
    let stream_stats = stream_csv_into_staging(pool, req, &staging_table, &cancel).await?;

    let sql_stats = staging_sql_stats(pool, req, &staging_table, table_exists_flag).await?;
    let batch_size = req.policy.batch_size.max(1) as u64;
    let valid_rows = stream_stats.loaded_rows;
    let estimated_batches = valid_rows.div_ceil(batch_size);

    let mut warnings = sql_stats.warnings;
    if matches!(req.target.mode, CsvImportTargetModeDto::Replace) {
        warnings.push(
            "Replace mode will swap staging into the live table only after validation (atomic RENAME).".to_string(),
        );
    } else if matches!(req.target.mode, CsvImportTargetModeDto::Append) {
        warnings.push(
            "Append mode is not fully atomic; a cancelled commit may leave partial rows."
                .to_string(),
        );
    }

    let dry_run = CsvImportDryRunResultDto {
        total_rows: stream_stats.total_rows,
        valid_rows,
        invalid_rows: stream_stats.invalid_rows,
        duplicate_rows: sql_stats.duplicate_rows,
        new_rows: valid_rows.saturating_sub(sql_stats.duplicate_rows),
        skipped_rows: if matches!(
            req.policy.duplicate_behavior,
            CsvImportDuplicateBehaviorDto::Skip
        ) {
            sql_stats.duplicate_rows
        } else {
            0
        },
        updated_rows: if matches!(
            req.policy.duplicate_behavior,
            CsvImportDuplicateBehaviorDto::Update
        ) {
            sql_stats.duplicate_rows
        } else {
            0
        },
        estimated_batches,
        table_exists: table_exists_flag,
        will_create_table: matches!(req.target.mode, CsvImportTargetModeDto::Create),
        will_replace_table: matches!(req.target.mode, CsvImportTargetModeDto::Replace),
        warnings,
        invalid_samples: stream_stats.invalid_samples,
        planned_columns: planned_columns(req),
        planned_indexes: planned_indexes(req),
        plan_hash: plan_hash.clone(),
        duplicate_probe_status: sql_stats.duplicate_probe_status,
        staging_table: Some(staging_table.clone()),
        load_method: CsvImportLoadMethodDto::BatchedInsert,
    };

    let plan = StagingPlan {
        job_id,
        staging_table,
        session_id: req.target.session_id.clone(),
        plan_hash,
        source_path: req.file.path.clone(),
        source_size,
        source_mtime,
    };
    Ok((dry_run, plan))
}

pub async fn commit_staged<F>(
    pool: &MySqlPool,
    req: &CsvImportRequestDto,
    plan: &StagingPlan,
    dry_run: &CsvImportDryRunResultDto,
    cancel: &CancelToken,
    mut on_progress: F,
) -> Result<CsvImportJobDto>
where
    F: FnMut(&CsvImportJobDto),
{
    verify_staging_reuse(req, plan)?;
    if dry_run.invalid_rows > 0 {
        bail!("dry run failed: {} invalid row(s)", dry_run.invalid_rows);
    }
    if dry_run.duplicate_rows > 0
        && matches!(
            req.policy.duplicate_behavior,
            CsvImportDuplicateBehaviorDto::Fail
        )
    {
        bail!(
            "dry run failed: {} duplicate row(s)",
            dry_run.duplicate_rows
        );
    }
    if matches!(req.target.mode, CsvImportTargetModeDto::Replace)
        && !req.policy.confirmed_destructive
    {
        bail!("replace mode requires explicit confirmation");
    }
    if let Some(expected) = req.plan_hash.as_deref() {
        if expected != dry_run.plan_hash {
            bail!("import plan hash mismatch; run dry-run again before commit");
        }
    }

    let mut job = CsvImportJobDto {
        job_id: plan.job_id.clone(),
        phase: CsvImportJobPhaseDto::Importing,
        total_rows: dry_run.valid_rows,
        processed_rows: dry_run.valid_rows,
        inserted_rows: 0,
        updated_rows: 0,
        skipped_rows: dry_run.skipped_rows,
        failed_rows: 0,
        current_batch: dry_run.estimated_batches,
        total_batches: dry_run.estimated_batches,
        table: req.target.table.clone(),
        message: None,
        error: None,
        dry_run: Some(dry_run.clone()),
        partial_commit: false,
        destructive_step_completed: false,
        staging_table: Some(plan.staging_table.clone()),
        load_method: dry_run.load_method,
    };
    on_progress(&job);

    if cancel.is_cancelled() {
        return cancelled_staging_job(job, cancel, req, pool, plan, false).await;
    }

    match req.target.mode {
        CsvImportTargetModeDto::Create => {
            if !table_exists(pool, &req.target.database, &req.target.table).await? {
                rename_staging_to_live(
                    pool,
                    &req.target.database,
                    &plan.staging_table,
                    &req.target.table,
                )
                .await?;
                job.destructive_step_completed = true;
            } else {
                merge_staging_append(pool, req, &plan.staging_table, cancel, &mut job).await?;
            }
        }
        CsvImportTargetModeDto::Replace => {
            if req.policy.create_indexes {
                job.phase = CsvImportJobPhaseDto::CreatingIndexes;
                on_progress(&job);
                create_indexes_on_table(pool, req, &plan.staging_table).await?;
            }
            if cancel.is_cancelled() {
                return cancelled_staging_job(job, cancel, req, pool, plan, false).await;
            }
            swap_replace_tables(pool, &req.target.database, &req.target.table, plan).await?;
            job.destructive_step_completed = true;
            job.inserted_rows = dry_run.valid_rows;
        }
        CsvImportTargetModeDto::Append => {
            merge_staging_append(pool, req, &plan.staging_table, cancel, &mut job).await?;
            job.partial_commit = job.inserted_rows > 0 && cancel.is_cancelled();
        }
    }

    if cancel.is_cancelled() {
        let destructive_done = job.destructive_step_completed;
        return cancelled_staging_job(job, cancel, req, pool, plan, destructive_done).await;
    }

    if req.policy.create_indexes && !matches!(req.target.mode, CsvImportTargetModeDto::Replace) {
        job.phase = CsvImportJobPhaseDto::CreatingIndexes;
        on_progress(&job);
        create_indexes(pool, req).await?;
    }

    job.phase = CsvImportJobPhaseDto::Validating;
    on_progress(&job);
    let imported_count = row_count(pool, &req.target.database, &req.target.table).await?;
    job.inserted_rows = imported_count.saturating_sub(job.skipped_rows);
    job.message = Some(format!(
        "{imported_count} row(s) available in destination table"
    ));
    job.phase = CsvImportJobPhaseDto::Complete;
    on_progress(&job);
    Ok(job)
}

fn verify_staging_reuse(req: &CsvImportRequestDto, plan: &StagingPlan) -> Result<()> {
    if plan.session_id != req.target.session_id {
        bail!("import plan session mismatch");
    }
    let (size, mtime) = file_metadata(&req.file.path)?;
    if plan.source_path != req.file.path || plan.source_size != size || plan.source_mtime != mtime {
        bail!("CSV file changed since dry-run; run dry-run again");
    }
    Ok(())
}

struct StreamStats {
    total_rows: u64,
    loaded_rows: u64,
    invalid_rows: u64,
    invalid_samples: Vec<CsvImportInvalidRowDto>,
}

struct SqlStats {
    duplicate_rows: u64,
    duplicate_probe_status: CsvImportDuplicateProbeStatusDto,
    warnings: Vec<String>,
}

async fn stream_csv_into_staging(
    pool: &MySqlPool,
    req: &CsvImportRequestDto,
    staging_table: &str,
    cancel: &CancelToken,
) -> Result<StreamStats> {
    let pool = pool.clone();
    let req = req.clone();
    let staging_table = staging_table.to_string();
    let cancel = cancel.clone();
    tokio::task::spawn_blocking(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("staging stream runtime")?;
        let preview = CsvPreviewRequestDto {
            path: req.file.path.clone(),
            encoding: req.file.encoding.clone(),
            delimiter: req.file.delimiter.clone(),
            date_format: req.file.date_format.clone(),
        };
        let batch_size = req.policy.batch_size.max(1).min(20_000) as usize;
        let mut total_rows = 0u64;
        let mut loaded_rows = 0u64;
        let mut invalid_rows = 0u64;
        let mut invalid_samples = Vec::new();
        let mut batch_buffer = Vec::with_capacity(batch_size);

        stream_csv_people_batches(
            &preview,
            Some(&req.mapping),
            batch_size,
            &crate::loaders::csv_loader::CsvLoadOptions {
                should_cancel: Some(Arc::new(move || cancel.is_cancelled())),
                ..Default::default()
            },
            |people| {
                total_rows += people.len() as u64;
                for (idx, person) in people.iter().enumerate() {
                    let row_number = loaded_rows + invalid_rows + idx as u64 + 2;
                    if let Some(reason) = invalid_person_reason(req.policy.id_behavior, person) {
                        invalid_rows += 1;
                        if invalid_samples.len() < 10 {
                            invalid_samples.push(CsvImportInvalidRowDto { row_number, reason });
                        }
                        continue;
                    }
                    batch_buffer.push(person.clone());
                }
                if !batch_buffer.is_empty() {
                    let written = rt.block_on(insert_staging_batch(
                        &pool,
                        &req,
                        &staging_table,
                        &batch_buffer,
                    ))?;
                    loaded_rows += written as u64;
                    batch_buffer.clear();
                }
                Ok(())
            },
        )?;

        if !batch_buffer.is_empty() {
            loaded_rows += rt.block_on(insert_staging_batch(
                &pool,
                &req,
                &staging_table,
                &batch_buffer,
            ))? as u64;
        }

        Ok(StreamStats {
            total_rows,
            loaded_rows,
            invalid_rows,
            invalid_samples,
        })
    })
    .await
    .context("staging CSV stream task join")?
}

use std::sync::Arc;

fn invalid_person_reason(id_behavior: CsvImportIdBehaviorDto, person: &Person) -> Option<String> {
    let mut reasons = Vec::new();
    if !matches!(id_behavior, CsvImportIdBehaviorDto::DbAutoIncrement) && person.id <= 0 {
        reasons.push("id must be greater than zero");
    }
    if person.first_name.as_deref().unwrap_or("").trim().is_empty() {
        reasons.push("first_name is required");
    }
    if person.last_name.as_deref().unwrap_or("").trim().is_empty() {
        reasons.push("last_name is required");
    }
    if person.birthdate.is_none() {
        reasons.push("birthdate is required");
    }
    if reasons.is_empty() {
        None
    } else {
        Some(reasons.join(", "))
    }
}

async fn create_staging_table(
    pool: &MySqlPool,
    req: &CsvImportRequestDto,
    staging_table: &str,
) -> Result<()> {
    let id_sql = if matches!(
        req.policy.id_behavior,
        CsvImportIdBehaviorDto::DbAutoIncrement
    ) {
        "`id` BIGINT NOT NULL AUTO_INCREMENT"
    } else {
        "`id` BIGINT NOT NULL"
    };
    let sql = format!(
        "CREATE TABLE `{}`.`{}` (\
         `stg_row_id` BIGINT NOT NULL AUTO_INCREMENT, \
         `source_line` BIGINT NULL, \
         {id_sql}, \
         `uuid` VARCHAR(64) NULL, \
         `first_name` VARCHAR(255) NOT NULL, \
         `middle_name` VARCHAR(255) NULL, \
         `last_name` VARCHAR(255) NOT NULL, \
         `birthdate` DATE NOT NULL, \
         `hh_id` VARCHAR(255) NULL, \
         `imported_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, \
         PRIMARY KEY (`stg_row_id`)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4",
        req.target.database, staging_table
    );
    sqlx::query(&sql).execute(pool).await?;
    Ok(())
}

async fn insert_staging_batch(
    pool: &MySqlPool,
    req: &CsvImportRequestDto,
    staging_table: &str,
    people: &[Person],
) -> Result<usize> {
    if people.is_empty() {
        return Ok(0);
    }
    let include_id = !matches!(
        req.policy.id_behavior,
        CsvImportIdBehaviorDto::DbAutoIncrement
    );
    let cols: Vec<&str> = if include_id {
        vec![
            "id",
            "uuid",
            "first_name",
            "middle_name",
            "last_name",
            "birthdate",
            "hh_id",
        ]
    } else {
        vec![
            "uuid",
            "first_name",
            "middle_name",
            "last_name",
            "birthdate",
            "hh_id",
        ]
    };
    let mut builder: QueryBuilder<MySql> = QueryBuilder::new(format!(
        "INSERT INTO `{}`.`{}` ({}) ",
        req.target.database,
        staging_table,
        cols.iter()
            .map(|c| format!("`{c}`"))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    builder.push_values(people, |mut row, person| {
        if include_id {
            row.push_bind(person.id);
        }
        row.push_bind(person.uuid.as_deref());
        row.push_bind(person.first_name.as_deref().unwrap_or(""));
        row.push_bind(person.middle_name.as_deref());
        row.push_bind(person.last_name.as_deref().unwrap_or(""));
        row.push_bind(person.birthdate);
        row.push_bind(person.hh_id.as_deref());
    });
    let result = builder.build().execute(pool).await?;
    Ok(result.rows_affected() as usize)
}

async fn staging_sql_stats(
    pool: &MySqlPool,
    req: &CsvImportRequestDto,
    staging_table: &str,
    dest_exists: bool,
) -> Result<SqlStats> {
    let mut warnings = Vec::new();
    let in_file_dupes = count_in_file_duplicates(pool, req, staging_table).await?;
    let (existing_dupes, probe_status) = if dest_exists {
        count_existing_duplicates_sql(pool, req, staging_table).await?
    } else {
        (0, CsvImportDuplicateProbeStatusDto::Complete)
    };
    let duplicate_rows = in_file_dupes.max(existing_dupes);
    if matches!(probe_status, CsvImportDuplicateProbeStatusDto::Failed) {
        warnings.push("Could not check existing duplicates in destination table.".to_string());
    }
    Ok(SqlStats {
        duplicate_rows,
        duplicate_probe_status: probe_status,
        warnings,
    })
}

async fn count_in_file_duplicates(
    pool: &MySqlPool,
    req: &CsvImportRequestDto,
    staging_table: &str,
) -> Result<u64> {
    let key_expr = duplicate_key_sql(req);
    let sql = format!(
        "SELECT COALESCE(SUM(extra), 0) AS dupes FROM (\
         SELECT COUNT(*) - 1 AS extra FROM `{}`.`{}` GROUP BY {key_expr} HAVING COUNT(*) > 1\
         ) AS d",
        req.target.database, staging_table
    );
    let row = sqlx::query(&sql).fetch_one(pool).await?;
    let dupes: i64 = row.try_get("dupes").unwrap_or(0);
    Ok(dupes.max(0) as u64)
}

async fn count_existing_duplicates_sql(
    pool: &MySqlPool,
    req: &CsvImportRequestDto,
    staging_table: &str,
) -> Result<(u64, CsvImportDuplicateProbeStatusDto)> {
    let join_cond = match req.policy.duplicate_key {
        CsvImportDuplicateKeyDto::Id => "s.`id` = d.`id`".to_string(),
        CsvImportDuplicateKeyDto::Uuid => "s.`uuid` <=> d.`uuid`".to_string(),
        CsvImportDuplicateKeyDto::MatcherFields => {
            "s.`first_name` <=> d.`first_name` AND s.`last_name` <=> d.`last_name` AND s.`birthdate` <=> d.`birthdate`".to_string()
        }
    };
    let sql = format!(
        "SELECT COUNT(*) AS c FROM `{}`.`{}` s \
         INNER JOIN `{}`.`{}` d ON {join_cond}",
        req.target.database, staging_table, req.target.database, req.target.table
    );
    match sqlx::query(&sql).fetch_one(pool).await {
        Ok(row) => {
            let count: i64 = row.try_get("c").unwrap_or(0);
            Ok((
                count.max(0) as u64,
                CsvImportDuplicateProbeStatusDto::Complete,
            ))
        }
        Err(err) => {
            log::warn!("existing duplicate probe failed: {err}");
            Ok((0, CsvImportDuplicateProbeStatusDto::Failed))
        }
    }
}

fn duplicate_key_sql(req: &CsvImportRequestDto) -> String {
    match req.policy.duplicate_key {
        CsvImportDuplicateKeyDto::Id => "`id`".to_string(),
        CsvImportDuplicateKeyDto::Uuid => "COALESCE(`uuid`, '')".to_string(),
        CsvImportDuplicateKeyDto::MatcherFields => {
            "CONCAT(COALESCE(`first_name`,''), '|', COALESCE(`last_name`,''), '|', COALESCE(`birthdate`,''))".to_string()
        }
    }
}

fn replace_swap_sql(
    database: &str,
    live_table: &str,
    old_table: &str,
    staging_table: &str,
) -> String {
    format!(
        "RENAME TABLE `{database}`.`{live_table}` TO `{database}`.`{old_table}`, \
         `{database}`.`{staging_table}` TO `{database}`.`{live_table}`",
        database = database,
        live_table = live_table,
        old_table = old_table,
        staging_table = staging_table
    )
}

async fn swap_replace_tables(
    pool: &MySqlPool,
    database: &str,
    live_table: &str,
    plan: &StagingPlan,
) -> Result<()> {
    let old_table = old_table_name(live_table, &plan.job_id)?;
    drop_staging_if_exists(pool, database, &old_table).await?;
    let sql = replace_swap_sql(database, live_table, &old_table, &plan.staging_table);
    sqlx::query(&sql).execute(pool).await?;
    let drop_sql = format!("DROP TABLE IF EXISTS `{database}`.`{old_table}`");
    sqlx::query(&drop_sql).execute(pool).await?;
    Ok(())
}

async fn rename_staging_to_live(
    pool: &MySqlPool,
    database: &str,
    staging_table: &str,
    live_table: &str,
) -> Result<()> {
    let sql = format!("RENAME TABLE `{database}`.`{staging_table}` TO `{database}`.`{live_table}`");
    sqlx::query(&sql).execute(pool).await?;
    Ok(())
}

async fn merge_staging_append(
    pool: &MySqlPool,
    req: &CsvImportRequestDto,
    staging_table: &str,
    cancel: &CancelToken,
    job: &mut CsvImportJobDto,
) -> Result<()> {
    let chunk = 50_000i64;
    let mut offset = 0i64;
    let mut batch_idx = 0u64;
    loop {
        if cancel.is_cancelled() {
            job.partial_commit = job.inserted_rows > 0;
            break;
        }
        let mut rows = fetch_staging_chunk(pool, req, staging_table, offset, chunk).await?;
        if rows.is_empty() {
            break;
        }
        batch_idx += 1;
        job.current_batch = batch_idx;
        apply_id_policy(req, &mut rows);
        let written = insert_batch(pool, req, &rows).await?;
        job.processed_rows += rows.len() as u64;
        job.inserted_rows += written;
        offset += rows.len() as i64;
    }
    drop_staging_if_exists(pool, &req.target.database, staging_table).await?;
    Ok(())
}

async fn fetch_staging_chunk(
    pool: &MySqlPool,
    req: &CsvImportRequestDto,
    staging_table: &str,
    offset: i64,
    limit: i64,
) -> Result<Vec<Person>> {
    let sql = format!(
        "SELECT `id`, `uuid`, `first_name`, `middle_name`, `last_name`, `birthdate`, `hh_id` \
         FROM `{}`.`{}` ORDER BY `stg_row_id` LIMIT ? OFFSET ?",
        req.target.database, staging_table
    );
    let rows = sqlx::query(&sql)
        .bind(limit)
        .bind(offset)
        .fetch_all(pool)
        .await?;
    let mut people = Vec::with_capacity(rows.len());
    for row in rows {
        people.push(Person {
            id: row.try_get("id").unwrap_or(0),
            uuid: row.try_get("uuid").ok(),
            first_name: row.try_get("first_name").ok(),
            middle_name: row.try_get("middle_name").ok(),
            last_name: row.try_get("last_name").ok(),
            birthdate: row.try_get("birthdate").ok(),
            hh_id: row.try_get("hh_id").ok(),
            extra_fields: Default::default(),
        });
    }
    Ok(people)
}

async fn create_indexes_on_table(
    pool: &MySqlPool,
    req: &CsvImportRequestDto,
    table: &str,
) -> Result<()> {
    let mut req_copy = req.clone();
    req_copy.target.table = table.to_string();
    create_indexes(pool, &req_copy).await
}

async fn drop_staging_if_exists(pool: &MySqlPool, database: &str, table: &str) -> Result<()> {
    let sql = format!("DROP TABLE IF EXISTS `{database}`.`{table}`");
    sqlx::query(&sql).execute(pool).await?;
    Ok(())
}

async fn cancelled_staging_job(
    mut job: CsvImportJobDto,
    cancel: &CancelToken,
    req: &CsvImportRequestDto,
    pool: &MySqlPool,
    plan: &StagingPlan,
    destructive_done: bool,
) -> Result<CsvImportJobDto> {
    let _ = cancel;
    if !destructive_done {
        let _ = drop_staging_if_exists(pool, &req.target.database, &plan.staging_table).await;
    }
    job.phase = CsvImportJobPhaseDto::Cancelled;
    job.message = Some(if destructive_done {
        "Import cancelled after a destructive step; review the destination table.".to_string()
    } else {
        "Import cancelled; staging dropped and live table unchanged.".to_string()
    });
    Ok(job)
}

pub async fn drop_orphan_staging_tables(pool: &MySqlPool, database: &str) -> Result<u64> {
    let rows = sqlx::query(
        "SELECT TABLE_NAME FROM information_schema.TABLES \
         WHERE TABLE_SCHEMA = ? AND TABLE_NAME LIKE '%\\_\\_stg\\_%' ESCAPE '\\\\'",
    )
    .bind(database)
    .fetch_all(pool)
    .await?;
    let mut dropped = 0u64;
    for row in rows {
        let name: String = row.try_get("TABLE_NAME")?;
        if drop_staging_if_exists(pool, database, &name).await.is_ok() {
            dropped += 1;
        }
    }
    Ok(dropped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staging_table_name_is_safe() {
        let id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890";
        let name = staging_table_name("people", id).unwrap();
        assert!(name.starts_with("people__stg_"));
        assert!(!name.contains('-'));
        assert!(name.len() <= 64);
    }

    #[test]
    fn staging_table_name_truncates_long_targets() {
        let long = "a".repeat(80);
        let name = staging_table_name(&long, "abcd1234efgh5678").unwrap();
        assert!(name.len() <= 64);
    }

    #[test]
    fn rejects_unsafe_staging_identifiers() {
        assert!(staging_table_name("bad;drop", "jobid12345678").is_err());
    }

    #[test]
    fn replace_swap_sql_renames_live_to_old_and_staging_to_live() {
        let sql = replace_swap_sql(
            "duplicate_checker",
            "people",
            "people__old_job12345678",
            "people__stg_job12345678",
        );
        assert!(sql.contains(
            "RENAME TABLE `duplicate_checker`.`people` TO `duplicate_checker`.`people__old_job12345678`"
        ));
        assert!(sql.contains(
            "`duplicate_checker`.`people__stg_job12345678` TO `duplicate_checker`.`people`"
        ));
    }
}
