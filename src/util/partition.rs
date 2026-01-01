use crate::db::schema::SqlBind;
use crate::models::ColumnMapping;

#[derive(Debug, Clone)]
pub struct Partition {
    pub name: String,
    pub where_sql: String,
    pub binds: Vec<SqlBind>,
}

pub trait PartitionStrategy {
    fn partitions(&self, mapping: Option<&ColumnMapping>) -> Vec<Partition>;
}

#[derive(Debug, Clone, Copy)]
pub struct LastInitial; // A-Z plus non-alpha bucket

impl PartitionStrategy for LastInitial {
    fn partitions(&self, mapping: Option<&ColumnMapping>) -> Vec<Partition> {
        let m = mapping.cloned().unwrap_or_default();
        let last = m.last_name;
        let mut parts = Vec::with_capacity(27);
        for ch in b'A'..=b'Z' {
            let c = ch as char;
            parts.push(Partition {
                name: format!("last_{}", c),
                where_sql: format!("UPPER(LEFT(`{}`,1)) = ?", last),
                binds: vec![SqlBind::Str(c.to_string())],
            });
        }
        // Non-alpha bucket
        parts.push(Partition {
            name: "last_other".into(),
            where_sql: format!("NOT (UPPER(LEFT(`{}`,1)) BETWEEN 'A' AND 'Z')", last),
            binds: vec![],
        });
        parts
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BirthYearRanges {
    pub years_per_shard: i32,
    pub min_year: i32,
    pub max_year: i32,
}

impl Default for BirthYearRanges {
    fn default() -> Self {
        Self {
            years_per_shard: 5,
            min_year: 1900,
            max_year: 2030,
        }
    }
}

impl PartitionStrategy for BirthYearRanges {
    fn partitions(&self, mapping: Option<&ColumnMapping>) -> Vec<Partition> {
        let m = mapping.cloned().unwrap_or_default();
        let bd = m.birthdate;
        let mut v = Vec::new();
        let mut y = self.min_year;
        while y <= self.max_year {
            let y2 = (y + self.years_per_shard - 1).min(self.max_year);
            v.push(Partition {
                name: format!("year_{}_{}", y, y2),
                where_sql: format!("YEAR(`{}`) BETWEEN ? AND ?", bd),
                binds: vec![SqlBind::I64(y as i64), SqlBind::I64(y2 as i64)],
            });
            y += self.years_per_shard;
        }
        v
    }
}

pub enum DefaultPartition {
    LastInitial,
    BirthYear5,
}

impl DefaultPartition {
    pub fn build(&self) -> Box<dyn PartitionStrategy + Send + Sync> {
        match self {
            DefaultPartition::LastInitial => Box::new(LastInitial),
            DefaultPartition::BirthYear5 => Box::new(BirthYearRanges::default()),
        }
    }
}
