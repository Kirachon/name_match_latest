//! Integration test for CSV → MySQL import.
//!
//! Run manually:
//! ```text
//! set MYSQL_IMPORT_TEST_URL=<mysql-url-for-local-test-database>
//! cargo test --test csv_import_mysql -- --ignored
//! ```
use name_matcher::import::staging::{commit_staged, dry_run_staged};
use name_matcher::import::{commit_import, validate_import_plan};
use name_matcher::models::ColumnMapping;
use name_matcher::run_service::CancelToken;
use name_matcher::run_service::dto::{
    CsvImportDuplicateBehaviorDto, CsvImportDuplicateKeyDto, CsvImportIdBehaviorDto,
    CsvImportJobPhaseDto, CsvImportPolicyDto, CsvImportRequestDto, CsvImportTargetDto,
    CsvImportTargetModeDto, FileSelectionDto,
};
use sqlx::mysql::MySqlPoolOptions;

fn test_database() -> String {
    std::env::var("MYSQL_IMPORT_TEST_DATABASE").unwrap_or_else(|_| {
        std::env::var("MYSQL_IMPORT_TEST_URL")
            .ok()
            .and_then(|url| url.rsplit('/').next().map(str::to_string))
            .filter(|db| !db.is_empty())
            .unwrap_or_else(|| "matcher".to_string())
    })
}

#[tokio::test]
#[ignore = "requires MYSQL_IMPORT_TEST_URL"]
async fn imports_csv_into_mysql_table() -> anyhow::Result<()> {
    let url = std::env::var("MYSQL_IMPORT_TEST_URL")?;
    let pool = MySqlPoolOptions::new()
        .max_connections(4)
        .connect(&url)
        .await?;
    let db = test_database();
    let table = "csv_import_smoke";
    sqlx::query(&format!("DROP TABLE IF EXISTS `{db}`.`{table}`"))
        .execute(&pool)
        .await?;

    let path = std::env::temp_dir().join(format!(
        "name_matcher_import_smoke_{}.csv",
        uuid::Uuid::new_v4()
    ));
    std::fs::write(
        &path,
        "id,uuid,first_name,middle_name,last_name,birthdate,hh_id\n1,u1,Ana,M,Santos,1990-01-02,H1\n2,u2,Ben,,Cruz,1991-02-03,H2\n",
    )?;

    let request = CsvImportRequestDto {
        target: CsvImportTargetDto {
            session_id: "integration".to_string(),
            database: test_database(),
            table: table.to_string(),
            mode: CsvImportTargetModeDto::Create,
        },
        file: FileSelectionDto {
            path: path.to_string_lossy().to_string(),
            sheet_name: None,
            encoding: None,
            delimiter: None,
            date_format: Some("%Y-%m-%d".to_string()),
        },
        mapping: ColumnMapping::default(),
        policy: CsvImportPolicyDto {
            id_behavior: CsvImportIdBehaviorDto::UseCsvId,
            duplicate_behavior: CsvImportDuplicateBehaviorDto::Skip,
            duplicate_key: CsvImportDuplicateKeyDto::Id,
            batch_size: 1,
            create_indexes: true,
            confirmed_destructive: false,
        },
        plan_hash: None,
    };

    let dry_run = validate_import_plan(&pool, &request).await?;
    assert_eq!(dry_run.valid_rows, 2);
    assert!(dry_run.will_create_table);

    let outcome = commit_import(&pool, &request).await?;
    assert_eq!(
        outcome.job.phase,
        name_matcher::run_service::dto::CsvImportJobPhaseDto::Complete
    );
    assert_eq!(outcome.job.processed_rows, 2);

    let count: i64 = sqlx::query_scalar(&format!("SELECT COUNT(*) FROM `{db}`.`{table}`"))
        .fetch_one(&pool)
        .await?;
    assert_eq!(count, 2);
    sqlx::query(&format!("DROP TABLE IF EXISTS `{db}`.`{table}`"))
        .execute(&pool)
        .await?;
    pool.close().await;
    let _ = std::fs::remove_file(path);
    Ok(())
}

#[tokio::test]
#[ignore = "requires MYSQL_IMPORT_TEST_URL"]
async fn replace_cancel_before_swap_preserves_live_table() -> anyhow::Result<()> {
    let url = std::env::var("MYSQL_IMPORT_TEST_URL")?;
    let pool = MySqlPoolOptions::new()
        .max_connections(4)
        .connect(&url)
        .await?;
    let table = "csv_import_replace_preserve";
    let db = test_database();
    sqlx::query(&format!("DROP TABLE IF EXISTS `{db}`.`{table}`"))
        .execute(&pool)
        .await?;
    sqlx::query(&format!(
        "CREATE TABLE `{db}`.`{table}` (\
         `id` BIGINT NOT NULL, `uuid` VARCHAR(64) NULL, \
         `first_name` VARCHAR(255) NOT NULL, `middle_name` VARCHAR(255) NULL, \
         `last_name` VARCHAR(255) NOT NULL, `birthdate` DATE NOT NULL, \
         `hh_id` VARCHAR(255) NULL, PRIMARY KEY (`id`)) ENGINE=InnoDB"
    ))
    .execute(&pool)
    .await?;
    sqlx::query(&format!(
        "INSERT INTO `{db}`.`{table}` VALUES \
         (1,'u1','Keep','','Me','1990-01-01','H1')"
    ))
    .execute(&pool)
    .await?;

    let path = std::env::temp_dir().join(format!(
        "name_matcher_import_replace_{}.csv",
        uuid::Uuid::new_v4()
    ));
    std::fs::write(
        &path,
        "id,uuid,first_name,middle_name,last_name,birthdate,hh_id\n9,u9,New,,Row,1992-03-04,H9\n",
    )?;

    let request = CsvImportRequestDto {
        target: CsvImportTargetDto {
            session_id: "integration".to_string(),
            database: test_database(),
            table: table.to_string(),
            mode: CsvImportTargetModeDto::Replace,
        },
        file: FileSelectionDto {
            path: path.to_string_lossy().to_string(),
            sheet_name: None,
            encoding: None,
            delimiter: None,
            date_format: Some("%Y-%m-%d".to_string()),
        },
        mapping: ColumnMapping::default(),
        policy: CsvImportPolicyDto {
            id_behavior: CsvImportIdBehaviorDto::UseCsvId,
            duplicate_behavior: CsvImportDuplicateBehaviorDto::Skip,
            duplicate_key: CsvImportDuplicateKeyDto::Id,
            batch_size: 500,
            create_indexes: false,
            confirmed_destructive: true,
        },
        plan_hash: None,
    };

    let (dry_run, plan) = dry_run_staged(&pool, &request).await?;
    assert!(dry_run.staging_table.is_some());
    let cancel = CancelToken::new();
    cancel.cancel();
    let job = commit_staged(&pool, &request, &plan, &dry_run, &cancel, |_| {}).await?;
    assert_eq!(job.phase, CsvImportJobPhaseDto::Cancelled);

    let preserved: (String,) = sqlx::query_as(&format!(
        "SELECT first_name FROM `{db}`.`{table}` WHERE id = 1"
    ))
    .fetch_one(&pool)
    .await?;
    assert_eq!(preserved.0, "Keep");

    sqlx::query(&format!("DROP TABLE IF EXISTS `{db}`.`{table}`"))
        .execute(&pool)
        .await?;
    let _ = std::fs::remove_file(path);
    pool.close().await;
    Ok(())
}
