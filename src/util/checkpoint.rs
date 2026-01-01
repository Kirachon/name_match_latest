use std::{fs, path::Path};

#[derive(Debug, Clone)]
pub struct StreamCheckpoint {
    pub db: String,
    pub table_inner: String,
    pub table_outer: String,
    pub algorithm: String,
    pub batch_size: i64,
    pub next_offset: i64,
    pub total_outer: i64,
    pub partition_idx: i32,
    pub partition_name: String,
    pub updated_utc: String,
    // New optional fields for keyset pagination + snapshot consistency
    pub last_id: Option<i64>,
    pub watermark_id: Option<i64>,
    pub filter_sig: Option<String>,
}

impl StreamCheckpoint {
    fn serialize(&self) -> String {
        let mut out = format!(
            "db={}\ninner={}\nouter={}\nalgo={}\nbatch={}\noffset={}\ntotal={}\npart_idx={}\npart_name={}\nupdated={}\n",
            self.db,
            self.table_inner,
            self.table_outer,
            self.algorithm,
            self.batch_size,
            self.next_offset,
            self.total_outer,
            self.partition_idx,
            self.partition_name,
            self.updated_utc
        );
        if let Some(v) = self.last_id {
            out.push_str(&format!("last_id={}\n", v));
        }
        if let Some(v) = self.watermark_id {
            out.push_str(&format!("watermark_id={}\n", v));
        }
        if let Some(v) = &self.filter_sig {
            out.push_str(&format!("filter_sig={}\n", v));
        }
        out
    }
    fn deserialize(s: &str) -> Option<Self> {
        let mut db = String::new();
        let mut inner = String::new();
        let mut outer = String::new();
        let mut algo = String::new();
        let mut batch = 0i64;
        let mut offset = 0i64;
        let mut total = 0i64;
        let mut part_idx = 0i32;
        let mut part_name = String::new();
        let mut updated = String::new();
        let mut last_id: Option<i64> = None;
        let mut watermark_id: Option<i64> = None;
        let mut filter_sig: Option<String> = None;
        for line in s.lines() {
            if let Some((k, v)) = line.split_once('=') {
                match k {
                    "db" => db = v.to_string(),
                    "inner" => inner = v.to_string(),
                    "outer" => outer = v.to_string(),
                    "algo" => algo = v.to_string(),
                    "batch" => batch = v.parse().unwrap_or(0),
                    "offset" => offset = v.parse().unwrap_or(0),
                    "total" => total = v.parse().unwrap_or(0),
                    "part_idx" => part_idx = v.parse().unwrap_or(0),
                    "part_name" => part_name = v.to_string(),
                    "updated" => updated = v.to_string(),
                    "last_id" => last_id = v.parse().ok(),
                    "watermark_id" => watermark_id = v.parse().ok(),
                    "filter_sig" => filter_sig = Some(v.to_string()),
                    _ => {}
                }
            }
        }
        if db.is_empty() || inner.is_empty() || outer.is_empty() || algo.is_empty() {
            return None;
        }
        Some(Self {
            db,
            table_inner: inner,
            table_outer: outer,
            algorithm: algo,
            batch_size: batch,
            next_offset: offset,
            total_outer: total,
            partition_idx: part_idx,
            partition_name: part_name,
            updated_utc: updated,
            last_id,
            watermark_id,
            filter_sig,
        })
    }
}

pub fn load_checkpoint(path: &str) -> Option<StreamCheckpoint> {
    if !Path::new(path).exists() {
        return None;
    }
    match fs::read_to_string(path) {
        Ok(s) => StreamCheckpoint::deserialize(&s),
        Err(_) => None,
    }
}

pub fn save_checkpoint(path: &str, cp: &StreamCheckpoint) -> std::io::Result<()> {
    if let Some(parent) = Path::new(path).parent() {
        let _ = fs::create_dir_all(parent);
    }
    let s = cp.serialize();
    // write atomically: write to tmp then rename
    let tmp = format!("{}.tmp", path);
    fs::write(&tmp, s.as_bytes())?;
    fs::rename(&tmp, path)?;
    Ok(())
}

pub fn remove_checkpoint(path: &str) {
    let _ = fs::remove_file(path);
}
