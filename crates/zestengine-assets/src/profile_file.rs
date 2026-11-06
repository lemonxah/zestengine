//! Shared RON profile persistence. Domain modules own format-specific validation.
use std::path::Path;
use anyhow::{Context, Result};

pub fn load<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    ron::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

pub fn save<T: serde::Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() { std::fs::create_dir_all(parent)?; }
    let text = ron::ser::to_string_pretty(value, ron::ser::PrettyConfig::default())?;
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))
}
