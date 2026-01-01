use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Person {
    pub id: i64,
    pub uuid: Option<String>,
    pub first_name: Option<String>,
    pub middle_name: Option<String>,
    pub last_name: Option<String>,
    pub birthdate: Option<NaiveDate>,
    pub hh_id: Option<String>, // New: household key for Table 2
    #[sqlx(skip)]
    #[serde(default)]
    pub extra_fields: HashMap<String, String>, // Dynamic fields beyond standard schema
}

#[derive(Debug, Clone)]
pub struct NormalizedPerson {
    pub id: i64,
    pub uuid: String,
    pub first_name: Option<String>,
    pub middle_name: Option<String>,
    pub last_name: Option<String>,
    pub birthdate: Option<NaiveDate>,
}

#[derive(Debug, Clone)]
pub struct TableColumns {
    pub has_id: bool,
    pub has_uuid: bool,
    pub has_first_name: bool,
    pub has_middle_name: bool,
    pub has_last_name: bool,
    pub has_birthdate: bool,
    pub has_hh_id: bool, // New: indicates presence of hh_id
}

impl TableColumns {
    pub fn validate_basic(&self) -> anyhow::Result<()> {
        use anyhow::bail;
        if !(self.has_id
            && self.has_uuid
            && self.has_first_name
            && self.has_last_name
            && self.has_birthdate)
        {
            bail!(
                "Table missing required columns: requires id, uuid, first_name, last_name, birthdate (middle_name optional)"
            );
        }
        Ok(())
    }
}

// Column mapping for flexible schemas; map source column names to expected aliases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnMapping {
    pub id: String,
    pub uuid: Option<String>,
    pub first_name: String,
    pub middle_name: Option<String>,
    pub last_name: String,
    pub birthdate: String,
    pub hh_id: Option<String>, // New: household key column name in Table 2
}

impl Default for ColumnMapping {
    fn default() -> Self {
        Self {
            id: "id".into(),
            uuid: Some("uuid".into()),
            first_name: "first_name".into(),
            middle_name: Some("middle_name".into()),
            last_name: "last_name".into(),
            birthdate: "birthdate".into(),
            hh_id: Some("hh_id".into()),
        }
    }
}

impl ColumnMapping {
    #[allow(dead_code)]
    pub fn required_ok(&self) -> bool {
        !self.id.is_empty()
            && !self.first_name.is_empty()
            && !self.last_name.is_empty()
            && !self.birthdate.is_empty()
    }
}
