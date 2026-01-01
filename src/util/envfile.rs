use anyhow::Result;
use std::fs;
use std::io::Write;
use std::path::Path;

/// Parse environment variables from a .env file in the current working directory, if present.
/// Returns a map of key/value pairs. Does not modify the process environment.
pub fn parse_env_file() -> Result<std::collections::HashMap<String, String>> {
    let path = Path::new(".env");
    let mut map = std::collections::HashMap::new();
    if !path.exists() {
        return Ok(map);
    }
    let content = fs::read_to_string(path)?;
    for (idx, line) in content.lines().enumerate() {
        let s = line.trim();
        if s.is_empty() || s.starts_with('#') {
            continue;
        }
        if let Some(eq) = s.find('=') {
            let key = s[..eq].trim();
            let mut val = s[eq + 1..].to_string();
            // Remove surrounding quotes if present
            if (val.starts_with('"') && val.ends_with('"'))
                || (val.starts_with('\'') && val.ends_with('\''))
            {
                val = val[1..val.len() - 1].to_string();
            }
            map.insert(key.to_string(), val);
        } else {
            eprintln!(
                "Warning: ignoring .env line {} without '=': {}",
                idx + 1,
                line
            );
        }
    }
    Ok(map)
}

/// Load `.env` from current working directory into process environment (non-destructive: does not override existing vars).
pub fn load_dotenv_if_present() -> Result<()> {
    if let Ok(map) = parse_env_file() {
        for (k, v) in map {
            if std::env::var_os(&k).is_none() {
                unsafe {
                    std::env::set_var(&k, &v);
                }
            }
        }
    }
    Ok(())
}

/// Load a specific .env file path into process environment and also return the parsed map.
/// Existing environment variables are NOT overridden.
pub fn load_env_file_from(path: &str) -> Result<std::collections::HashMap<String, String>> {
    let p = std::path::Path::new(path);
    let mut map = std::collections::HashMap::new();
    if !p.exists() {
        return Ok(map);
    }
    let content = std::fs::read_to_string(p)?;
    for (idx, line) in content.lines().enumerate() {
        let s = line.trim();
        if s.is_empty() || s.starts_with('#') {
            continue;
        }
        if let Some(eq) = s.find('=') {
            let key = s[..eq].trim();
            let mut val = s[eq + 1..].to_string();
            if (val.starts_with('"') && val.ends_with('"'))
                || (val.starts_with('\'') && val.ends_with('\''))
            {
                val = val[1..val.len() - 1].to_string();
            }
            if std::env::var_os(key).is_none() {
                unsafe {
                    std::env::set_var(key, &val);
                }
            }
            map.insert(key.to_string(), val);
        } else {
            eprintln!(
                "Warning: ignoring .env line {} without '=': {}",
                idx + 1,
                line
            );
        }
    }
    Ok(map)
}

/// Generate a .env.template file with placeholder values and comments.
pub fn write_env_template(path: &str) -> Result<()> {
    let mut f = fs::File::create(path)?;
    let template = r#"# Name_Matcher environment configuration template
# Copy this file to .env and fill in your database connection settings.
# Any of these variables can also be provided via the system environment.

# Primary database (Table 1)
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=root
DB_PASSWORD=secret
DB_NAME=database_name

# Optional: Second database (Table 2). If not provided, Table 2 will be read from the primary database.
#DB2_HOST=127.0.0.1
#DB2_PORT=3306
#DB2_USER=root
#DB2_PASS=secret
#DB2_DATABASE=database_name

# Streaming/Performance (optional)
#NAME_MATCHER_STREAMING=true
#NAME_MATCHER_PARTITION=last_initial
#NAME_MATCHER_POOL_SIZE=16
#NAME_MATCHER_POOL_MIN=4
#NAME_MATCHER_ACQUIRE_MS=30000
#NAME_MATCHER_IDLE_MS=60000
#NAME_MATCHER_LIFETIME_MS=600000
"#;
    f.write_all(template.as_bytes())?;
    Ok(())
}
