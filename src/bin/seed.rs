use anyhow::Result;
use chrono::{Datelike, NaiveDate};
use sqlx::{MySql, Pool, mysql::MySqlPoolOptions};
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    // Args: host port user pass db table_a table_b rows
    let args: Vec<String> = std::env::args().collect();
    let host = args.get(1).cloned().unwrap_or_else(|| "127.0.0.1".into());
    let port = args
        .get(2)
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(3307);
    let user = args.get(3).cloned().unwrap_or_else(|| "root".into());
    let pass = args.get(4).cloned().unwrap_or_else(|| "root".into());
    let db = args
        .get(5)
        .cloned()
        .unwrap_or_else(|| "duplicate_checker".into());
    let t1 = args.get(6).cloned().unwrap_or_else(|| "sample_a".into());
    let t2 = args.get(7).cloned().unwrap_or_else(|| "sample_b".into());
    let rows = args
        .get(8)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(30000);

    println!("Seeding MySQL {db}::{t1},{t2} on {host}:{port} with {rows} rows each...");

    // 1) Connect to server-level DB to create target database
    let url_server = format!("mysql://{user}:{pass}@{host}:{port}/mysql");
    let pool_server = MySqlPoolOptions::new()
        .max_connections(5)
        .connect(&url_server)
        .await?;
    sqlx::query(&format!(
        "CREATE DATABASE IF NOT EXISTS `{}` CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci",
        db
    ))
    .execute(&pool_server)
    .await?;

    // 2) Connect to target database
    let url_db = format!("mysql://{user}:{pass}@{host}:{port}/{db}");
    let pool = MySqlPoolOptions::new()
        .max_connections(10)
        .connect(&url_db)
        .await?;

    // 3) Create tables
    create_table(&pool, &t1).await?;
    create_table(&pool, &t2).await?;

    // 4) Seed coordinated household data across T1 (uuid groups) and T2 (representatives)
    seed_households(&pool, &t1, &t2, rows, 42).await?;

    println!("Seeding complete.");
    Ok(())
}

async fn create_table(pool: &Pool<MySql>, table: &str) -> Result<()> {
    // Table 2 (sample_b, test_table_b) gets extra columns for testing dynamic field export
    let is_table2 = table.ends_with("_b") || table == "sample_b" || table == "test_table_b";

    // Include dynamic fields for T2 + shared codes for both tables
    let extra_cols = if is_table2 {
        ",\n            region_code VARCHAR(50) NULL,\n            poor_hat_0 VARCHAR(50) NULL,\n            poor_hat_10 VARCHAR(50) NULL,\n            barangay_code VARCHAR(50) NULL,\n            city_code VARCHAR(50) NULL,\n            hh_id VARCHAR(50) NULL"
    } else {
        ",\n            barangay_code VARCHAR(50) NULL,\n            city_code VARCHAR(50) NULL,\n            hh_id VARCHAR(50) NULL"
    };

    // Recreate table to ensure schema is up-to-date for tests
    let _ = sqlx::query(&format!("DROP TABLE IF EXISTS `{}`", table))
        .execute(pool)
        .await;

    let sql = format!(
        "CREATE TABLE IF NOT EXISTS `{table}` (
            id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            uuid VARCHAR(36) NULL,
            first_name VARCHAR(100) NOT NULL,
            middle_name VARCHAR(100) NULL,
            last_name VARCHAR(100) NOT NULL,
            birthdate DATE NOT NULL{extra_cols},
            INDEX idx_name_bd (last_name, first_name, birthdate),
            INDEX idx_bd (birthdate),
            INDEX idx_uuid (uuid),
            INDEX idx_barangay (barangay_code),
            INDEX idx_city (city_code),
            INDEX idx_hh (hh_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci"
    );
    sqlx::query(&sql).execute(pool).await?;
    Ok(())
}

async fn seed_households(
    pool: &Pool<MySql>,
    table_a: &str,
    table_b: &str,
    rows: usize,
    seed: u64,
) -> Result<()> {
    // We will create T1 households (uuid groups) with 2-5 members each.
    // For a subset of households, we create a partial match scenario where ~60-90% of members
    // share the same birthdate and very similar names to a single representative in T2 (by id),
    // so Option 5 aggregation yields realistic percentages > 50% but not all 100%.
    println!(
        "Seeding coordinated households into {table_a} and representatives into {table_b} ..."
    );

    // Clean existing tables
    sqlx::query(&format!("TRUNCATE TABLE `{}`", table_a))
        .execute(pool)
        .await
        .ok();
    sqlx::query(&format!("TRUNCATE TABLE `{}`", table_b))
        .execute(pool)
        .await
        .ok();

    let mut rng = Lcg::new(seed);
    let first_names = sample_first_names();
    let last_names = sample_last_names();

    // Build household specs until we reach approx `rows` members in T1
    #[derive(Clone)]
    struct HhSpec {
        uuid: String,
        size: usize,
        base_first: String,
        base_last: String,
        base_bd: NaiveDate,
        match_count: usize,
        barangay_code: String,
        city_code: String,
        /// If true, the T2 representative will have month/day swapped birthdate (for testing swap feature)
        use_birthdate_swap: bool,
        /// Middle name for T1 members (None for L11 no-middle-name testing)
        base_middle: Option<String>,
    }

    /// Check if a date can be swapped (both month and day must be <= 12)
    fn can_swap_birthdate(date: NaiveDate) -> bool {
        let month = date.month();
        let day = date.day();
        day <= 12 && month <= 12
    }

    /// Swap month and day of a date (only call if can_swap_birthdate returns true)
    fn swap_birthdate(date: NaiveDate) -> Option<NaiveDate> {
        let year = date.year();
        let month = date.month();
        let day = date.day();
        NaiveDate::from_ymd_opt(year, day, month)
    }

    let mut specs: Vec<HhSpec> = Vec::new();
    let mut t1_members = 0usize;
    let mut swap_count = 0usize;
    let mut no_middle_count = 0usize;

    while t1_members < rows {
        let size = 2 + ((rng.next() % 4) as usize); // 2..=5
        let base_first = first_names[(rng.next() as usize) % first_names.len()].to_string();
        let base_last = last_names[(rng.next() as usize) % last_names.len()].to_string();
        let base_bd = random_birthdate(&mut rng);
        let barangay_code = format!("B{:04}", (rng.next() % 500) + 1);
        let city_code = format!("C{:03}", (rng.next() % 120) + 1);

        // For roughly 30% of households, plan to have a majority of members match the same T2 rep.
        let is_matchy = (rng.next() % 10) < 3;
        let match_count = if is_matchy {
            std::cmp::max(2, (size as f32 * 0.6).ceil() as usize)
        } else {
            0
        };

        // For ~40% of matchy households with swappable dates, use birthdate swap
        // This creates test data for the birthdate swap feature (L10/L11 matching)
        let use_birthdate_swap =
            is_matchy && can_swap_birthdate(base_bd) && (rng.next() % 100) < 40;
        if use_birthdate_swap {
            swap_count += 1;
        }

        // For ~25% of matchy households, omit middle name (for L11 no-middle-name testing)
        let use_no_middle = is_matchy && (rng.next() % 100) < 25;
        let base_middle = if use_no_middle {
            no_middle_count += 1;
            None
        } else {
            Some(random_middle(&mut rng))
        };

        specs.push(HhSpec {
            uuid: Uuid::new_v4().to_string(),
            size,
            base_first,
            base_last,
            base_bd,
            match_count,
            barangay_code,
            city_code,
            use_birthdate_swap,
            base_middle,
        });
        t1_members += size;
    }

    println!(
        "  Generated {} households: {} with birthdate swaps, {} without middle names",
        specs.len(),
        swap_count,
        no_middle_count
    );

    // Insert T1 members in batches to avoid placeholder limits
    {
        let batch_cap = 800usize;
        let mut pending = 0usize;
        let mut q = sqlx::QueryBuilder::<MySql>::new("INSERT INTO ");
        q.push("`").push(table_a).push("` (uuid, first_name, middle_name, last_name, birthdate, barangay_code, city_code, hh_id) VALUES ");
        let mut first = true;
        for hh in &specs {
            // Matching members (same name/birthdate for fuzzy strength, birthdate exact)
            // Use base_middle for consistent matching with T2 (supports L11 no-middle-name testing)
            for _ in 0..hh.match_count {
                if !first {
                    q.push(", ");
                }
                first = false;
                q.push("(")
                    .push_bind(Some(hh.uuid.clone()))
                    .push(", ")
                    .push_bind(&hh.base_first)
                    .push(", ")
                    .push_bind(hh.base_middle.clone())
                    .push(", ")
                    .push_bind(&hh.base_last)
                    .push(", ")
                    .push_bind(hh.base_bd)
                    .push(", ")
                    .push_bind(&hh.barangay_code)
                    .push(", ")
                    .push_bind(&hh.city_code)
                    .push(", ")
                    .push_bind(Some(hh.uuid.clone()))
                    .push(")");
                pending += 1;
                if pending >= batch_cap {
                    q.build().execute(pool).await?;
                    pending = 0;
                    first = true;
                    q = sqlx::QueryBuilder::<MySql>::new("INSERT INTO ");
                    q.push("`").push(table_a).push("` (uuid, first_name, middle_name, last_name, birthdate, barangay_code, city_code, hh_id) VALUES ");
                }
            }
            // Non-matching or weakly similar members (vary name and/or birthdate)
            for _ in 0..(hh.size - hh.match_count) {
                let f = &first_names[(rng.next() as usize) % first_names.len()];
                let l = &last_names[(rng.next() as usize) % last_names.len()];
                let bd = if (rng.next() % 2) == 0 {
                    hh.base_bd
                } else {
                    random_birthdate(&mut rng)
                };
                if !first {
                    q.push(", ");
                }
                first = false;
                q.push("(")
                    .push_bind(Some(hh.uuid.clone()))
                    .push(", ")
                    .push_bind(f)
                    .push(", ")
                    .push_bind(Some(random_middle(&mut rng)))
                    .push(", ")
                    .push_bind(l)
                    .push(", ")
                    .push_bind(bd)
                    .push(", ")
                    .push_bind(&hh.barangay_code)
                    .push(", ")
                    .push_bind(&hh.city_code)
                    .push(", ")
                    .push_bind(Some(hh.uuid.clone()))
                    .push(")");
                pending += 1;
                if pending >= batch_cap {
                    q.build().execute(pool).await?;
                    pending = 0;
                    first = true;
                    q = sqlx::QueryBuilder::<MySql>::new("INSERT INTO ");
                    q.push("`").push(table_a).push("` (uuid, first_name, middle_name, last_name, birthdate, barangay_code, city_code, hh_id) VALUES ");
                }
            }
        }
        // flush remaining
        if pending > 0 {
            q.build().execute(pool).await?;
        }
    }

    // Insert T2 representatives: one row per "matchy" household with base name/birthdate (batched)
    // Include extra columns for Table 2
    // For households with use_birthdate_swap=true, use swapped birthdate to test the swap feature
    {
        let batch_cap = 1000usize;
        let mut pending = 0usize;
        let mut q = sqlx::QueryBuilder::<MySql>::new("INSERT INTO ");
        q.push("`").push(table_b).push("` (uuid, first_name, middle_name, last_name, birthdate, region_code, poor_hat_0, poor_hat_10, barangay_code, city_code, hh_id) VALUES ");
        let mut first = true;
        for hh in &specs {
            if hh.match_count == 0 {
                continue;
            }
            if !first {
                q.push(", ");
            }
            first = false;

            // Generate test data for extra columns
            let region_code = format!("R{:02}", (rng.next() % 10) + 1);
            let poor_hat_0 = if (rng.next() % 2) == 0 {
                "YES".to_string()
            } else {
                "NO".to_string()
            };
            let poor_hat_10 = format!("{:.2}", (rng.next() % 100) as f32 / 100.0);

            // Use swapped birthdate if this household is marked for swap testing
            let t2_birthdate = if hh.use_birthdate_swap {
                swap_birthdate(hh.base_bd).unwrap_or(hh.base_bd)
            } else {
                hh.base_bd
            };

            // Use same middle name as T1 for proper matching (None for L11 testing)
            let t2_middle = hh.base_middle.clone();

            q.push("(")
                .push_bind(Some(hh.uuid.clone()))
                .push(", ")
                .push_bind(&hh.base_first)
                .push(", ")
                .push_bind(t2_middle)
                .push(", ")
                .push_bind(&hh.base_last)
                .push(", ")
                .push_bind(t2_birthdate)
                .push(", ")
                .push_bind(region_code)
                .push(", ")
                .push_bind(poor_hat_0)
                .push(", ")
                .push_bind(poor_hat_10)
                .push(", ")
                .push_bind(&hh.barangay_code)
                .push(", ")
                .push_bind(&hh.city_code)
                .push(", ")
                .push_bind(Some(hh.uuid.clone()))
                .push(")");
            pending += 1;
            if pending >= batch_cap {
                q.build().execute(pool).await?;
                pending = 0;
                first = true;
                q = sqlx::QueryBuilder::<MySql>::new("INSERT INTO ");
                q.push("`").push(table_b).push("` (uuid, first_name, middle_name, last_name, birthdate, region_code, poor_hat_0, poor_hat_10, barangay_code, city_code, hh_id) VALUES ");
            }
        }
        if pending > 0 {
            q.build().execute(pool).await?;
        }
    }

    // Add extra noise rows to T2 to reach approximate `rows` total (batched)
    // Include extra columns for Table 2
    {
        // Determine current count in table_b
        let cnt: (i64,) = sqlx::query_as(&format!("SELECT COUNT(*) FROM {}", table_b))
            .fetch_one(pool)
            .await?;
        let have = cnt.0 as usize;
        if have < rows {
            let mut to_add = rows - have;
            let batch_cap = 1000usize;
            while to_add > 0 {
                let mut q = sqlx::QueryBuilder::<MySql>::new("INSERT INTO ");
                q.push("`").push(table_b).push("` (uuid, first_name, middle_name, last_name, birthdate, region_code, poor_hat_0, poor_hat_10, barangay_code, city_code, hh_id) VALUES ");
                let mut first = true;
                let mut pending = 0usize;
                while to_add > 0 && pending < batch_cap {
                    let f = &first_names[(rng.next() as usize) % first_names.len()];
                    let l = &last_names[(rng.next() as usize) % last_names.len()];
                    let bd = random_birthdate(&mut rng);
                    // Generate test data for extra columns
                    let region_code = format!("R{:02}", (rng.next() % 10) + 1);
                    let poor_hat_0 = if (rng.next() % 2) == 0 {
                        "YES".to_string()
                    } else {
                        "NO".to_string()
                    };
                    let poor_hat_10 = format!("{:.2}", (rng.next() % 100) as f32 / 100.0);
                    let barangay_code = format!("B{:04}", (rng.next() % 500) + 1);
                    let city_code = format!("C{:03}", (rng.next() % 120) + 1);

                    if !first {
                        q.push(", ");
                    }
                    first = false;
                    q.push("(")
                        .push_bind(Some(Uuid::new_v4().to_string()))
                        .push(", ")
                        .push_bind(f)
                        .push(", ")
                        .push_bind(Some(random_middle(&mut rng)))
                        .push(", ")
                        .push_bind(l)
                        .push(", ")
                        .push_bind(bd)
                        .push(", ")
                        .push_bind(region_code)
                        .push(", ")
                        .push_bind(poor_hat_0)
                        .push(", ")
                        .push_bind(poor_hat_10)
                        .push(", ")
                        .push_bind(barangay_code)
                        .push(", ")
                        .push_bind(city_code)
                        .push(", ")
                        .push_bind(Option::<String>::None)
                        .push(")");
                    to_add -= 1;
                    pending += 1;
                }
                if pending > 0 {
                    q.build().execute(pool).await?;
                }
            }
        }
    }

    println!(
        "  Seeding complete: {} swap pairs and {} no-middle pairs created for birthdate swap testing",
        swap_count, no_middle_count
    );

    Ok(())
}

async fn seed_table(pool: &Pool<MySql>, table: &str, rows: usize, seed: u64) -> Result<()> {
    println!("Seeding {table} with {rows} rows...");

    // Simple LCG for deterministic pseudo-random without external deps
    let mut rng = Lcg::new(seed);
    let first_names = sample_first_names();
    let last_names = sample_last_names();

    // Insert in batches for speed
    let batch_size = 1000usize;
    let mut inserted = 0usize;
    while inserted < rows {
        let to_insert = std::cmp::min(batch_size, rows - inserted);
        let mut tx = pool.begin().await?;
        let mut q = sqlx::QueryBuilder::<MySql>::new("INSERT INTO ");
        q.push("`")
            .push(table)
            .push("` (uuid, first_name, middle_name, last_name, birthdate) VALUES ");
        for i in 0..to_insert {
            let f = &first_names[(rng.next() as usize) % first_names.len()];
            let l = &last_names[(rng.next() as usize) % last_names.len()];
            let mid = if (rng.next() % 4) == 0 {
                Some(random_middle(&mut rng))
            } else {
                None
            };
            let uuid = Some(Uuid::new_v4().to_string());
            let bd = random_birthdate(&mut rng);

            if i > 0 {
                q.push(", ");
            }
            q.push("(");
            q.push_bind(uuid);
            q.push(", ");
            q.push_bind(f);
            q.push(", ");
            q.push_bind(mid);
            q.push(", ");
            q.push_bind(l);
            q.push(", ");
            q.push_bind(bd);
            q.push(")");
        }
        let query = q.build();
        query.execute(&mut *tx).await?;
        tx.commit().await?;
        inserted += to_insert;
        if inserted % 5000 == 0 || inserted == rows {
            println!("  {inserted}/{rows}...");
        }
    }
    Ok(())
}

fn random_middle(rng: &mut Lcg) -> String {
    // Either single-letter middle initial or short name
    if (rng.next() % 2) == 0 {
        let c = (b'A' + (rng.next() % 26) as u8) as char;
        format!("{c}")
    } else {
        let opts = ["Lee", "Ann", "Kai", "Ray", "Mae", "Jo", "Jay", "Kim"];
        opts[(rng.next() as usize) % opts.len()].to_string()
    }
}

fn random_birthdate(rng: &mut Lcg) -> NaiveDate {
    // Between 1950-01-01 and 2010-12-31
    let year = 1950 + (rng.next() % 61) as i32;
    let month = 1 + (rng.next() % 12) as u32;
    let mut day_max = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => 28,
    };
    // handle leap years simply
    if month == 2 && (year % 4 == 0) {
        day_max = 29;
    }
    let day = 1 + (rng.next() % day_max as u64) as u32;
    NaiveDate::from_ymd_opt(year, month, day)
        .unwrap_or_else(|| NaiveDate::from_ymd_opt(2000, 1, 1).unwrap())
}

struct Lcg {
    state: u64,
}
impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next(&mut self) -> u64 {
        // Numerical Recipes LCG constants
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        self.state
    }
}

fn sample_first_names() -> Vec<&'static str> {
    vec![
        "James",
        "Mary",
        "Robert",
        "Patricia",
        "John",
        "Jennifer",
        "Michael",
        "Linda",
        "William",
        "Elizabeth",
        "David",
        "Barbara",
        "Richard",
        "Susan",
        "Joseph",
        "Jessica",
        "Thomas",
        "Sarah",
        "Charles",
        "Karen",
        "Christopher",
        "Nancy",
        "Daniel",
        "Lisa",
        "Matthew",
        "Betty",
        "Anthony",
        "Margaret",
        "Mark",
        "Sandra",
        "Donald",
        "Ashley",
        "Steven",
        "Kimberly",
        "Paul",
        "Emily",
        "Andrew",
        "Donna",
        "Joshua",
        "Michelle",
    ]
}

fn sample_last_names() -> Vec<&'static str> {
    vec![
        "Smith",
        "Johnson",
        "Williams",
        "Brown",
        "Jones",
        "Garcia",
        "Miller",
        "Davis",
        "Rodriguez",
        "Martinez",
        "Hernandez",
        "Lopez",
        "Gonzalez",
        "Wilson",
        "Anderson",
        "Thomas",
        "Taylor",
        "Moore",
        "Jackson",
        "Martin",
        "Lee",
        "Perez",
        "Thompson",
        "White",
        "Harris",
        "Sanchez",
        "Clark",
        "Ramirez",
        "Lewis",
        "Robinson",
        "Walker",
        "Young",
        "Allen",
        "King",
        "Wright",
        "Scott",
        "Torres",
        "Nguyen",
        "Hill",
        "Flores",
    ]
}
