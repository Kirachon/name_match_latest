pub mod staging;
pub use staging::{StagingPlan, drop_orphan_staging_tables};

use crate::import::staging::{commit_staged, dry_run_staged};
use crate::loaders::csv_loader::{CsvDelimiterDto, CsvPreviewRequestDto, load_csv_preview};
use crate::models::Person;
use crate::run_service::dto::{
    CsvImportDryRunResultDto, CsvImportDuplicateBehaviorDto, CsvImportDuplicateKeyDto,
    CsvImportDuplicateProbeStatusDto, CsvImportIdBehaviorDto, CsvImportIndexPlanDto,
    CsvImportInvalidRowDto, CsvImportJobDto, CsvImportJobPhaseDto, CsvImportLoadMethodDto,
    CsvImportRequestDto, CsvImportTargetModeDto,
};
use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};
use sqlx::{MySql, MySqlPool, QueryBuilder, Row};
use std::collections::HashSet;
use uuid::Uuid;

use crate::run_service::CancelToken;

const STANDARD_COLUMNS: [&str; 7] = [
    "id",
    "uuid",
    "first_name",
    "middle_name",
    "last_name",
    "birthdate",
    "hh_id",
];

#[derive(Debug, Clone)]
pub struct CsvImportOutcome {
    pub job: CsvImportJobDto,
}

/// Stable fingerprint of the import request (excluding `plan_hash`).
pub fn compute_plan_hash(req: &CsvImportRequestDto) -> String {
    let mut snapshot = req.clone();
    snapshot.plan_hash = None;
    let json = serde_json::to_string(&snapshot).unwrap_or_default();
    format!("{:x}", Sha256::digest(json.as_bytes()))
}

pub async fn validate_import_plan(
    pool: &MySqlPool,
    req: &CsvImportRequestDto,
) -> Result<CsvImportDryRunResultDto> {
    let (dry_run, _) = dry_run_staged(pool, req).await?;
    Ok(dry_run)
}

/// Dry-run via staging tables; returns metadata for plan cache (no `Vec<Person>`).
pub async fn validate_import_plan_staged(
    pool: &MySqlPool,
    req: &CsvImportRequestDto,
) -> Result<(CsvImportDryRunResultDto, StagingPlan)> {
    dry_run_staged(pool, req).await
}

/// Dry-run analysis plus parsed rows (single CSV parse).
pub async fn validate_and_load(
    pool: &MySqlPool,
    req: &CsvImportRequestDto,
) -> Result<(CsvImportDryRunResultDto, Vec<Person>)> {
    validate_request(req)?;
    let table_exists = table_exists(pool, &req.target.database, &req.target.table).await?;
    validate_target_mode(req, table_exists)?;

    if table_exists && matches!(req.target.mode, CsvImportTargetModeDto::Append) {
        validate_existing_table(pool, &req.target.database, &req.target.table, req).await?;
    }

    let people = load_import_people(req)?;
    let total_rows = count_csv_rows(&req.file.path, req.file.delimiter.as_ref())?;
    let mut warnings = Vec::new();
    let mut invalid_samples = Vec::new();
    let invalid_rows = collect_invalid_rows(req, &people, &mut invalid_samples);
    let duplicate_rows = count_input_duplicates(req, &people);
    let existing_duplicates = if table_exists {
        match count_existing_duplicates(pool, req, &people).await {
            Ok(n) => n,
            Err(err) => {
                warnings.push(format!(
                    "Could not check existing duplicates in destination table: {err}"
                ));
                0
            }
        }
    } else {
        0
    };
    if people.len() > 10_000 && table_exists {
        warnings.push(
            "Existing-duplicate scan is limited to the first 10,000 parsed rows; counts may be understated.".to_string(),
        );
    }
    let duplicate_rows = duplicate_rows.max(existing_duplicates);
    if duplicate_rows > 0 {
        warnings.push(format!(
            "{duplicate_rows} duplicate row(s) detected for {:?}",
            req.policy.duplicate_key
        ));
    }
    if matches!(req.target.mode, CsvImportTargetModeDto::Replace) {
        warnings
            .push("Replace mode will delete existing destination rows before import.".to_string());
    }
    if matches!(
        req.policy.duplicate_behavior,
        CsvImportDuplicateBehaviorDto::Update
    ) {
        warnings.push("Update mode can overwrite existing destination fields.".to_string());
    }
    let valid_rows = people.len().saturating_sub(invalid_rows as usize) as u64;
    let skipped_rows = if matches!(
        req.policy.duplicate_behavior,
        CsvImportDuplicateBehaviorDto::Skip
    ) {
        duplicate_rows
    } else {
        0
    };
    let updated_rows = if matches!(
        req.policy.duplicate_behavior,
        CsvImportDuplicateBehaviorDto::Update
    ) {
        duplicate_rows
    } else {
        0
    };
    let batch_size = req.policy.batch_size.max(1) as u64;
    let estimated_batches = valid_rows.div_ceil(batch_size);

    let plan_hash = compute_plan_hash(req);
    let dry_run = CsvImportDryRunResultDto {
        total_rows,
        valid_rows,
        invalid_rows,
        duplicate_rows,
        new_rows: valid_rows.saturating_sub(duplicate_rows),
        skipped_rows,
        updated_rows,
        estimated_batches,
        table_exists,
        will_create_table: matches!(req.target.mode, CsvImportTargetModeDto::Create),
        will_replace_table: matches!(req.target.mode, CsvImportTargetModeDto::Replace),
        warnings,
        invalid_samples,
        planned_columns: planned_columns(req),
        planned_indexes: planned_indexes(req),
        plan_hash,
        duplicate_probe_status: CsvImportDuplicateProbeStatusDto::Complete,
        staging_table: None,
        load_method: CsvImportLoadMethodDto::BatchedInsert,
    };
    Ok((dry_run, people))
}

pub async fn commit_import(
    pool: &MySqlPool,
    req: &CsvImportRequestDto,
) -> Result<CsvImportOutcome> {
    let (dry_run, plan) = dry_run_staged(pool, req).await?;
    let cancel = CancelToken::new();
    commit_import_staged(pool, req, &plan, &dry_run, &cancel, |_| {}).await
}

pub async fn commit_import_staged<F>(
    pool: &MySqlPool,
    req: &CsvImportRequestDto,
    plan: &StagingPlan,
    dry_run: &CsvImportDryRunResultDto,
    cancel: &CancelToken,
    on_progress: F,
) -> Result<CsvImportOutcome>
where
    F: FnMut(&CsvImportJobDto),
{
    let job = commit_staged(pool, req, plan, dry_run, cancel, on_progress).await?;
    Ok(CsvImportOutcome { job })
}

/// Run import using rows from a prior dry-run (single parse). Updates `job` via `on_progress`.
pub async fn commit_import_prepared<F>(
    pool: &MySqlPool,
    req: &CsvImportRequestDto,
    mut people: Vec<Person>,
    dry_run: CsvImportDryRunResultDto,
    cancel: &CancelToken,
    mut on_progress: F,
) -> Result<CsvImportOutcome>
where
    F: FnMut(&CsvImportJobDto),
{
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

    apply_id_policy(req, &mut people);

    let job_id = Uuid::new_v4().to_string();
    let mut job = CsvImportJobDto {
        job_id: job_id.clone(),
        phase: CsvImportJobPhaseDto::CreatingTable,
        total_rows: dry_run.valid_rows,
        processed_rows: 0,
        inserted_rows: 0,
        updated_rows: 0,
        skipped_rows: 0,
        failed_rows: 0,
        current_batch: 0,
        total_batches: dry_run.estimated_batches,
        table: req.target.table.clone(),
        message: None,
        error: None,
        dry_run: Some(dry_run.clone()),
        partial_commit: false,
        destructive_step_completed: false,
        staging_table: dry_run.staging_table.clone(),
        load_method: dry_run.load_method,
    };
    on_progress(&job);

    if cancel.is_cancelled() {
        return cancelled_outcome(job, cancel, req);
    }

    if matches!(req.target.mode, CsvImportTargetModeDto::Create) {
        create_table(pool, req).await?;
    } else if matches!(req.target.mode, CsvImportTargetModeDto::Replace) {
        replace_table(pool, req).await?;
    }

    if cancel.is_cancelled() {
        return cancelled_outcome(job, cancel, req);
    }

    job.phase = CsvImportJobPhaseDto::Importing;
    on_progress(&job);
    let batch_size = req.policy.batch_size.max(1) as usize;
    for (batch_idx, chunk) in people.chunks(batch_size).enumerate() {
        if cancel.is_cancelled() {
            return cancelled_outcome(job, cancel, req);
        }
        job.current_batch = (batch_idx as u64) + 1;
        let written = insert_batch(pool, req, chunk).await?;
        job.processed_rows += chunk.len() as u64;
        match req.policy.duplicate_behavior {
            CsvImportDuplicateBehaviorDto::Skip => {
                job.inserted_rows += written;
                job.skipped_rows += (chunk.len() as u64).saturating_sub(written);
            }
            CsvImportDuplicateBehaviorDto::Update => {
                job.inserted_rows += written.min(chunk.len() as u64);
                job.updated_rows +=
                    (chunk.len() as u64).saturating_sub(written.min(chunk.len() as u64));
            }
            CsvImportDuplicateBehaviorDto::InsertAnyway | CsvImportDuplicateBehaviorDto::Fail => {
                job.inserted_rows += written;
            }
        }
        on_progress(&job);
    }

    if cancel.is_cancelled() {
        return cancelled_outcome(job, cancel, req);
    }

    if req.policy.create_indexes {
        job.phase = CsvImportJobPhaseDto::CreatingIndexes;
        on_progress(&job);
        create_indexes(pool, req).await?;
    }

    job.phase = CsvImportJobPhaseDto::Validating;
    on_progress(&job);
    let imported_count = row_count(pool, &req.target.database, &req.target.table).await?;
    job.message = Some(format!(
        "{imported_count} row(s) available in destination table"
    ));
    job.phase = CsvImportJobPhaseDto::Complete;
    on_progress(&job);
    Ok(CsvImportOutcome { job })
}

pub(crate) fn apply_id_policy(req: &CsvImportRequestDto, people: &mut [Person]) {
    if matches!(
        req.policy.id_behavior,
        CsvImportIdBehaviorDto::GenerateId | CsvImportIdBehaviorDto::DbAutoIncrement
    ) {
        for (idx, person) in people.iter_mut().enumerate() {
            person.id = (idx as i64) + 1;
        }
    }
    if matches!(req.policy.id_behavior, CsvImportIdBehaviorDto::GenerateUuid) {
        for person in people {
            person.uuid = Some(Uuid::new_v4().to_string());
        }
    }
}

fn cancelled_outcome(
    mut job: CsvImportJobDto,
    cancel: &CancelToken,
    req: &CsvImportRequestDto,
) -> Result<CsvImportOutcome> {
    let _ = cancel;
    job.phase = CsvImportJobPhaseDto::Cancelled;
    let partial = if matches!(req.target.mode, CsvImportTargetModeDto::Replace) {
        "Import cancelled. Replace mode may have truncated the table; partial batches may remain."
    } else {
        "Import cancelled. Partial batches may remain in the destination table."
    };
    job.message = Some(partial.to_string());
    Ok(CsvImportOutcome { job })
}

fn validate_request(req: &CsvImportRequestDto) -> Result<()> {
    validate_ident(&req.target.database)?;
    validate_ident(&req.target.table)?;
    validate_ident(&req.mapping.first_name)?;
    validate_ident(&req.mapping.last_name)?;
    validate_ident(&req.mapping.birthdate)?;
    if !matches!(
        req.policy.id_behavior,
        CsvImportIdBehaviorDto::DbAutoIncrement
    ) {
        validate_ident(&req.mapping.id)?;
    }
    if let Some(value) = req.mapping.uuid.as_deref().filter(|v| !v.is_empty()) {
        validate_ident(value)?;
    }
    if let Some(value) = req.mapping.middle_name.as_deref().filter(|v| !v.is_empty()) {
        validate_ident(value)?;
    }
    if let Some(value) = req.mapping.hh_id.as_deref().filter(|v| !v.is_empty()) {
        validate_ident(value)?;
    }
    if req.file.path.trim().is_empty() {
        bail!("CSV file path is required");
    }
    if req.policy.batch_size == 0 || req.policy.batch_size > 200_000 {
        bail!("batch size must be between 1 and 200000");
    }
    if matches!(
        req.policy.duplicate_behavior,
        CsvImportDuplicateBehaviorDto::Update
    ) && matches!(
        req.policy.duplicate_key,
        CsvImportDuplicateKeyDto::MatcherFields
    ) {
        bail!("update duplicate handling requires id or uuid duplicate key");
    }
    Ok(())
}

fn validate_target_mode(req: &CsvImportRequestDto, table_exists: bool) -> Result<()> {
    match req.target.mode {
        CsvImportTargetModeDto::Create if table_exists => {
            bail!("target table already exists: {}", req.target.table)
        }
        CsvImportTargetModeDto::Append if !table_exists => {
            bail!("target table does not exist: {}", req.target.table)
        }
        CsvImportTargetModeDto::Replace if table_exists && !req.policy.confirmed_destructive => {
            bail!("replace mode requires explicit confirmation")
        }
        _ => Ok(()),
    }
}

pub(crate) async fn validate_existing_table(
    pool: &MySqlPool,
    database: &str,
    table: &str,
    req: &CsvImportRequestDto,
) -> Result<()> {
    let columns = table_columns(pool, database, table).await?;
    for required in required_destination_columns(req) {
        if !columns.contains(required) {
            bail!("target table missing required column: {required}");
        }
    }
    Ok(())
}

fn load_import_people(req: &CsvImportRequestDto) -> Result<Vec<Person>> {
    use crate::loaders::csv_loader::load_csv_people;
    load_csv_people(
        &CsvPreviewRequestDto {
            path: req.file.path.clone(),
            encoding: req.file.encoding.clone(),
            delimiter: req.file.delimiter.clone(),
            date_format: req.file.date_format.clone(),
        },
        Some(&req.mapping),
    )
}

fn collect_invalid_rows(
    req: &CsvImportRequestDto,
    people: &[Person],
    invalid_samples: &mut Vec<CsvImportInvalidRowDto>,
) -> u64 {
    let mut invalid = 0;
    for (idx, person) in people.iter().enumerate() {
        let mut reasons = Vec::new();
        if !matches!(
            req.policy.id_behavior,
            CsvImportIdBehaviorDto::DbAutoIncrement
        ) && person.id <= 0
        {
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
        if !reasons.is_empty() {
            invalid += 1;
            if invalid_samples.len() < 10 {
                invalid_samples.push(CsvImportInvalidRowDto {
                    row_number: idx as u64 + 2,
                    reason: reasons.join(", "),
                });
            }
        }
    }
    invalid
}

fn count_input_duplicates(req: &CsvImportRequestDto, people: &[Person]) -> u64 {
    let mut seen = HashSet::new();
    let mut duplicates = 0;
    for person in people {
        let key = duplicate_key(req, person);
        if !seen.insert(key) {
            duplicates += 1;
        }
    }
    duplicates
}

async fn count_existing_duplicates(
    pool: &MySqlPool,
    req: &CsvImportRequestDto,
    people: &[Person],
) -> Result<u64> {
    let mut duplicates = 0;
    for person in people.iter().take(10_000) {
        let exists = match req.policy.duplicate_key {
            CsvImportDuplicateKeyDto::Id => {
                let row = sqlx::query(&format!(
                    "SELECT 1 FROM `{}`.`{}` WHERE `id` = ? LIMIT 1",
                    req.target.database, req.target.table
                ))
                .bind(person.id)
                .fetch_optional(pool)
                .await?;
                row.is_some()
            }
            CsvImportDuplicateKeyDto::Uuid => {
                if let Some(uuid) = person.uuid.as_deref() {
                    let row = sqlx::query(&format!(
                        "SELECT 1 FROM `{}`.`{}` WHERE `uuid` = ? LIMIT 1",
                        req.target.database, req.target.table
                    ))
                    .bind(uuid)
                    .fetch_optional(pool)
                    .await?;
                    row.is_some()
                } else {
                    false
                }
            }
            CsvImportDuplicateKeyDto::MatcherFields => {
                let row = sqlx::query(&format!(
                    "SELECT 1 FROM `{}`.`{}` WHERE `first_name` <=> ? AND `last_name` <=> ? AND `birthdate` <=> ? LIMIT 1",
                    req.target.database, req.target.table
                ))
                .bind(person.first_name.as_deref())
                .bind(person.last_name.as_deref())
                .bind(person.birthdate)
                .fetch_optional(pool)
                .await?;
                row.is_some()
            }
        };
        if exists {
            duplicates += 1;
        }
    }
    Ok(duplicates)
}

pub(crate) async fn table_exists(pool: &MySqlPool, database: &str, table: &str) -> Result<bool> {
    let row = sqlx::query(
        "SELECT 1 FROM information_schema.TABLES WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? LIMIT 1",
    )
    .bind(database)
    .bind(table)
    .fetch_optional(pool)
    .await?;
    Ok(row.is_some())
}

async fn table_columns(pool: &MySqlPool, database: &str, table: &str) -> Result<HashSet<String>> {
    let rows = sqlx::query(
        "SELECT COLUMN_NAME FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?",
    )
    .bind(database)
    .bind(table)
    .fetch_all(pool)
    .await?;
    Ok(rows
        .into_iter()
        .filter_map(|row| row.try_get::<String, _>("COLUMN_NAME").ok())
        .collect())
}

pub(crate) async fn create_table(pool: &MySqlPool, req: &CsvImportRequestDto) -> Result<()> {
    let id_sql = if matches!(
        req.policy.id_behavior,
        CsvImportIdBehaviorDto::DbAutoIncrement
    ) {
        "`id` BIGINT NOT NULL AUTO_INCREMENT"
    } else {
        "`id` BIGINT NOT NULL"
    };
    let primary = if matches!(
        req.policy.id_behavior,
        CsvImportIdBehaviorDto::DbAutoIncrement
            | CsvImportIdBehaviorDto::UseCsvId
            | CsvImportIdBehaviorDto::GenerateId
    ) {
        ", PRIMARY KEY (`id`)"
    } else {
        ""
    };
    let sql = format!(
        "CREATE TABLE `{}`.`{}` (\
         {id_sql}, \
         `uuid` VARCHAR(64) NULL, \
         `first_name` VARCHAR(255) NOT NULL, \
         `middle_name` VARCHAR(255) NULL, \
         `last_name` VARCHAR(255) NOT NULL, \
         `birthdate` DATE NOT NULL, \
         `hh_id` VARCHAR(255) NULL, \
         `imported_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP\
         {primary}) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4",
        req.target.database, req.target.table
    );
    sqlx::query(&sql).execute(pool).await?;
    Ok(())
}

async fn replace_table(pool: &MySqlPool, req: &CsvImportRequestDto) -> Result<()> {
    let sql = format!(
        "TRUNCATE TABLE `{}`.`{}`",
        req.target.database, req.target.table
    );
    sqlx::query(&sql).execute(pool).await?;
    Ok(())
}

pub(crate) async fn insert_batch(
    pool: &MySqlPool,
    req: &CsvImportRequestDto,
    people: &[Person],
) -> Result<u64> {
    if people.is_empty() {
        return Ok(0);
    }
    let include_id = !matches!(
        req.policy.id_behavior,
        CsvImportIdBehaviorDto::DbAutoIncrement
    );
    let verb = if matches!(
        req.policy.duplicate_behavior,
        CsvImportDuplicateBehaviorDto::Skip
    ) {
        "INSERT IGNORE"
    } else {
        "INSERT"
    };
    let columns = insert_columns(include_id);
    let mut builder: QueryBuilder<MySql> = QueryBuilder::new(format!(
        "{verb} INTO `{}`.`{}` ({}) ",
        req.target.database,
        req.target.table,
        columns
            .iter()
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
    if matches!(
        req.policy.duplicate_behavior,
        CsvImportDuplicateBehaviorDto::Update
    ) {
        builder.push(
            " ON DUPLICATE KEY UPDATE \
             `uuid` = VALUES(`uuid`), \
             `first_name` = VALUES(`first_name`), \
             `middle_name` = VALUES(`middle_name`), \
             `last_name` = VALUES(`last_name`), \
             `birthdate` = VALUES(`birthdate`), \
             `hh_id` = VALUES(`hh_id`)",
        );
    }
    let result = builder.build().execute(pool).await?;
    Ok(result.rows_affected())
}

pub(crate) async fn create_indexes(pool: &MySqlPool, req: &CsvImportRequestDto) -> Result<()> {
    for idx in planned_indexes(req) {
        let unique = if idx.unique { "UNIQUE " } else { "" };
        let cols = idx
            .columns
            .iter()
            .map(|c| format!("`{c}`"))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "CREATE {unique}INDEX `{}` ON `{}`.`{}` ({cols})",
            idx.name, req.target.database, req.target.table
        );
        if let Err(err) = sqlx::query(&sql).execute(pool).await {
            let msg = err.to_string();
            if !(msg.contains("Duplicate key name") || msg.contains("already exists")) {
                return Err(err.into());
            }
        }
    }
    Ok(())
}

pub(crate) async fn row_count(pool: &MySqlPool, database: &str, table: &str) -> Result<u64> {
    let row = sqlx::query(&format!("SELECT COUNT(*) AS c FROM `{database}`.`{table}`"))
        .fetch_one(pool)
        .await?;
    let count: i64 = row.try_get("c").unwrap_or(0);
    Ok(count.max(0) as u64)
}

fn insert_columns(include_id: bool) -> Vec<&'static str> {
    if include_id {
        STANDARD_COLUMNS.to_vec()
    } else {
        STANDARD_COLUMNS[1..].to_vec()
    }
}

fn required_destination_columns(req: &CsvImportRequestDto) -> Vec<&'static str> {
    if matches!(
        req.policy.id_behavior,
        CsvImportIdBehaviorDto::DbAutoIncrement
    ) {
        STANDARD_COLUMNS[1..].to_vec()
    } else {
        STANDARD_COLUMNS.to_vec()
    }
}

fn planned_columns(req: &CsvImportRequestDto) -> Vec<String> {
    required_destination_columns(req)
        .into_iter()
        .map(str::to_string)
        .collect()
}

fn planned_indexes(req: &CsvImportRequestDto) -> Vec<CsvImportIndexPlanDto> {
    let mut indexes = Vec::new();
    if matches!(
        req.policy.id_behavior,
        CsvImportIdBehaviorDto::UseCsvUuid | CsvImportIdBehaviorDto::GenerateUuid
    ) || matches!(req.policy.duplicate_key, CsvImportDuplicateKeyDto::Uuid)
    {
        indexes.push(CsvImportIndexPlanDto {
            name: format!("idx_{}_uuid", req.target.table),
            columns: vec!["uuid".to_string()],
            unique: matches!(req.policy.duplicate_key, CsvImportDuplicateKeyDto::Uuid),
        });
    }
    indexes.push(CsvImportIndexPlanDto {
        name: format!("idx_{}_name_birthdate", req.target.table),
        columns: vec![
            "last_name".to_string(),
            "first_name".to_string(),
            "birthdate".to_string(),
        ],
        unique: matches!(
            req.policy.duplicate_key,
            CsvImportDuplicateKeyDto::MatcherFields
        ),
    });
    indexes.push(CsvImportIndexPlanDto {
        name: format!("idx_{}_hh_id", req.target.table),
        columns: vec!["hh_id".to_string()],
        unique: false,
    });
    indexes
}

fn duplicate_key(req: &CsvImportRequestDto, person: &Person) -> String {
    match req.policy.duplicate_key {
        CsvImportDuplicateKeyDto::Id => format!("id:{}", person.id),
        CsvImportDuplicateKeyDto::Uuid => format!("uuid:{}", person.uuid.as_deref().unwrap_or("")),
        CsvImportDuplicateKeyDto::MatcherFields => format!(
            "m:{}|{}|{}",
            person.first_name.as_deref().unwrap_or(""),
            person.last_name.as_deref().unwrap_or(""),
            person.birthdate.map(|d| d.to_string()).unwrap_or_default()
        ),
    }
}

fn count_csv_rows(path: &str, delimiter: Option<&CsvDelimiterDto>) -> Result<u64> {
    let preview = load_csv_preview(&CsvPreviewRequestDto {
        path: path.to_string(),
        encoding: None,
        delimiter: delimiter.cloned(),
        date_format: None,
    })
    .with_context(|| format!("failed to preview {path}"))?;
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(delimiter_byte(delimiter.unwrap_or(&preview.delimiter)))
        .flexible(true)
        .from_path(path)?;
    Ok(reader.records().count() as u64)
}

fn delimiter_byte(delimiter: &CsvDelimiterDto) -> u8 {
    match delimiter {
        CsvDelimiterDto::Comma => b',',
        CsvDelimiterDto::Semicolon => b';',
        CsvDelimiterDto::Tab => b'\t',
    }
}

fn validate_ident(name: &str) -> Result<()> {
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$')
    {
        bail!("invalid identifier: {name}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ColumnMapping;
    use crate::run_service::dto::{CsvImportPolicyDto, CsvImportTargetDto, FileSelectionDto};

    fn base_request() -> CsvImportRequestDto {
        CsvImportRequestDto {
            target: CsvImportTargetDto {
                session_id: "session".to_string(),
                database: "matcher".to_string(),
                table: "imported_people".to_string(),
                mode: CsvImportTargetModeDto::Create,
            },
            file: FileSelectionDto {
                path: "people.csv".to_string(),
                sheet_name: None,
                encoding: None,
                delimiter: None,
                date_format: Some("%Y-%m-%d".to_string()),
            },
            mapping: ColumnMapping::default(),
            policy: CsvImportPolicyDto::default(),
            plan_hash: None,
        }
    }

    #[test]
    fn rejects_unsafe_target_identifier() {
        let mut req = base_request();
        req.target.table = "people;DROP".to_string();
        let err = validate_request(&req).unwrap_err();
        assert!(err.to_string().contains("invalid identifier"));
    }

    #[test]
    fn replace_requires_confirmation() {
        let mut req = base_request();
        req.target.mode = CsvImportTargetModeDto::Replace;
        req.policy.confirmed_destructive = false;
        let err = validate_target_mode(&req, true).unwrap_err();
        assert!(err.to_string().contains("explicit confirmation"));
    }

    #[test]
    fn update_rejects_matcher_field_duplicate_key() {
        let mut req = base_request();
        req.policy.duplicate_behavior = CsvImportDuplicateBehaviorDto::Update;
        req.policy.duplicate_key = CsvImportDuplicateKeyDto::MatcherFields;
        let err = validate_request(&req).unwrap_err();
        assert!(err.to_string().contains("requires id or uuid"));
    }
}
