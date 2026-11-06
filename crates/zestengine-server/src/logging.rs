//! Console log with timestamps and levels, plus an optional file rotated by size.

use std::fs::{File, OpenOptions};
use std::io::{self, IsTerminal as _, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context as _, Result};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::util::SubscriberInitExt as _;

use crate::config::ServerConfig;

struct Inner {
    path: PathBuf,
    file: File,
    size: u64,
    max: u64,
    keep: u32,
}

/// An append-only log file that moves itself to `<name>.1` (shifting older
/// ones up, dropping past `keep`) when it grows past `max` bytes.
#[derive(Clone)]
pub struct RotatingFile {
    inner: Arc<Mutex<Inner>>,
}

impl RotatingFile {
    pub fn open(path: &Path, max_bytes: u64, keep: u32) -> io::Result<Self> {
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        let size = file.metadata()?.len();
        Ok(Self { inner: Arc::new(Mutex::new(Inner { path: path.to_path_buf(), file, size, max: max_bytes.max(1), keep })) })
    }

    /// Push buffered bytes to disk (shutdown, panic).
    pub fn sync(&self) {
        if let Ok(mut i) = self.inner.lock() {
            let _ = i.file.flush();
            let _ = i.file.sync_all();
        }
    }
}

fn numbered(path: &Path, n: u32) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(format!(".{n}"));
    PathBuf::from(s)
}

impl Inner {
    fn rotate(&mut self) -> io::Result<()> {
        let _ = self.file.flush();
        if self.keep == 0 {
            let _ = std::fs::remove_file(&self.path);
        } else {
            let _ = std::fs::remove_file(numbered(&self.path, self.keep));
            for n in (1..self.keep).rev() {
                let _ = std::fs::rename(numbered(&self.path, n), numbered(&self.path, n + 1));
            }
            std::fs::rename(&self.path, numbered(&self.path, 1))?;
        }
        self.file = OpenOptions::new().create(true).append(true).open(&self.path)?;
        self.size = 0;
        Ok(())
    }
}

impl Write for RotatingFile {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut i = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if i.size > 0 && i.size + buf.len() as u64 > i.max {
            i.rotate()?;
        }
        let n = i.file.write(buf)?;
        i.size += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).file.flush()
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for RotatingFile {
    type Writer = RotatingFile;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

/// Install the global subscriber. Returns the file handle (if any) so the
/// caller can [`RotatingFile::sync`] it before exiting.
pub fn init(cfg: &ServerConfig) -> Result<Option<RotatingFile>> {
    let filter = EnvFilter::try_new(&cfg.log_level).with_context(|| format!("log_level {:?}", cfg.log_level))?;
    let console = tracing_subscriber::fmt::layer().with_ansi(io::stdout().is_terminal());
    let file = match &cfg.log_path {
        Some(p) => Some(RotatingFile::open(Path::new(p), u64::from(cfg.log_max_mb) * 1024 * 1024, cfg.log_keep).with_context(|| format!("opening log file {p}"))?),
        None => None,
    };
    let file_layer = file.clone().map(|f| tracing_subscriber::fmt::layer().with_ansi(false).with_writer(f));
    tracing_subscriber::registry().with(filter).with(console).with(file_layer).try_init().context("installing the log subscriber")?;
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotates_by_size_and_keeps_a_bounded_history() {
        let dir = std::env::temp_dir().join(format!("zestengine-server-log-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("logs/server.log");
        let mut f = RotatingFile::open(&path, 100, 2).unwrap();
        for n in 0..12 {
            f.write_all(format!("line {n:02} ................................\n").as_bytes()).unwrap(); // 43 bytes
        }
        f.flush().unwrap();
        let read = |p: &Path| std::fs::read_to_string(p).unwrap();
        assert!(read(&path).contains("line 11"), "newest lines in the live file");
        assert!(numbered(&path, 1).exists() && numbered(&path, 2).exists());
        assert!(!numbered(&path, 3).exists(), "history is bounded");
        assert!(std::fs::metadata(&path).unwrap().len() <= 100);
        assert!(!read(&numbered(&path, 2)).contains("line 11"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
