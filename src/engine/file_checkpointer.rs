#![cfg(feature = "new_engine")]

use crate::engine::Checkpointer;
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone)]
pub struct FileCheckpointer {
    base_dir: PathBuf,
}

impl FileCheckpointer {
    pub fn new<P: Into<PathBuf>>(base: P) -> Self {
        Self {
            base_dir: base.into(),
        }
    }
    fn path_for(&self, job: &str, partition: &str) -> PathBuf {
        // Sanitize simple file name components; fall back to hex if empty
        let clean = |s: &str| {
            let mut t: String = s
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                        c
                    } else {
                        '_'
                    }
                })
                .collect();
            if t.is_empty() {
                t = "_".to_string();
            }
            t
        };
        let fname = format!("{}_{}.ckpt", clean(job), clean(partition));
        self.base_dir.join(fname)
    }
}

impl Checkpointer for FileCheckpointer {
    fn save(&mut self, job: &str, partition: &str, token: &str) -> anyhow::Result<()> {
        let path = self.path_for(job, partition);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("tmp");
        fs::write(&tmp, token.as_bytes())?;
        fs::rename(&tmp, &path)?;
        Ok(())
    }
    fn load(&self, job: &str, partition: &str) -> anyhow::Result<Option<String>> {
        let path = self.path_for(job, partition);
        if !Path::new(&path).exists() {
            return Ok(None);
        }
        match fs::read_to_string(&path) {
            Ok(s) => Ok(Some(s)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}
