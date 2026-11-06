//! Safe zip reading for downloaded asset packs (Fab / Megascans archives and
//! `.mhpkg` MetaHuman packages). The archive is untrusted: names are checked
//! against path traversal (zip-slip), sizes against hard limits (a zip bomb
//! declares small and expands huge, so the bytes actually written are counted
//! too), symlinks and encrypted entries are skipped, and repeated names never
//! overwrite each other.

use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};

use anyhow::{Context as _, Result, bail};

/// Hard limits applied when listing or extracting an archive.
#[derive(Clone, Copy, Debug)]
pub struct ZipLimits {
    /// Most entries an archive may hold.
    pub max_entries: usize,
    /// Largest single entry, uncompressed.
    pub max_entry: u64,
    /// Largest sum of all entries, uncompressed.
    pub max_total: u64,
    /// Largest uncompressed : compressed ratio of one entry (above it the entry
    /// is a bomb; tiny entries are exempt, see [`RATIO_FLOOR`]).
    pub max_ratio: u64,
}

impl Default for ZipLimits {
    fn default() -> Self {
        Self { max_entries: 50_000, max_entry: 4 << 30, max_total: 24 << 30, max_ratio: 1000 }
    }
}

/// Entries smaller than this are never judged by their compression ratio
/// (an empty or tiny file compresses to almost nothing legitimately).
const RATIO_FLOOR: u64 = 1 << 20;

/// A relative path with no `..`, root, drive or empty components. Backslashes
/// count as separators (archives written on Windows). `None` = unsafe.
pub fn safe_rel(name: &str) -> Option<PathBuf> {
    if name.contains('\0') {
        return None;
    }
    let name = name.replace('\\', "/");
    let mut out = PathBuf::new();
    for c in Path::new(name.trim()).components() {
        match c {
            Component::Normal(n) => out.push(n),
            Component::CurDir => {}
            // `..`, a leading `/`, a drive prefix: all escape the destination.
            _ => return None,
        }
    }
    (!out.as_os_str().is_empty()).then_some(out)
}

/// One archive entry as the central directory declares it.
#[derive(Clone, Debug)]
pub struct ZipEntryInfo {
    /// Name as stored.
    pub name: String,
    /// Uncompressed size (as declared; extraction re-checks the real count).
    pub size: u64,
    pub compressed: u64,
    pub is_dir: bool,
    /// Stored (0) or deflate (8) are the methods read; others are listed as unreadable.
    pub readable: bool,
    /// The sanitised relative path, `None` when the name is unsafe.
    pub rel: Option<PathBuf>,
    pub symlink: bool,
    /// Never set by a listing (the central directory is read without a password);
    /// an encrypted entry shows up as skipped when extracting.
    pub encrypted: bool,
}

/// What a listing found, with the archive-level checks applied.
#[derive(Clone, Debug, Default)]
pub struct ZipListing {
    pub entries: Vec<ZipEntryInfo>,
    /// Sum of declared uncompressed sizes of the file entries.
    pub total: u64,
}

fn open(path: &Path) -> Result<zip::ZipArchive<std::io::BufReader<std::fs::File>>> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    zip::ZipArchive::new(std::io::BufReader::new(file)).with_context(|| format!("{} is not a readable zip archive", path.display()))
}

/// List `path` without extracting. Fails when the entry count or the declared
/// sizes break `limits`.
pub fn list_zip(path: &Path, limits: &ZipLimits) -> Result<ZipListing> {
    let mut zip = open(path)?;
    if zip.len() > limits.max_entries {
        bail!("{} holds {} entries (limit {})", path.display(), zip.len(), limits.max_entries);
    }
    let mut out = ZipListing::default();
    for i in 0..zip.len() {
        let f = zip.by_index_raw(i).with_context(|| format!("reading entry {i} of {}", path.display()))?;
        let mode = f.unix_mode().unwrap_or(0);
        let info = ZipEntryInfo {
            name: f.name().to_owned(),
            size: f.size(),
            compressed: f.compressed_size(),
            is_dir: f.is_dir(),
            readable: matches!(f.compression(), zip::CompressionMethod::Stored | zip::CompressionMethod::Deflated),
            rel: safe_rel(f.name()),
            symlink: mode & 0o170000 == 0o120000,
            encrypted: false,
        };
        if !info.is_dir {
            out.total = out.total.saturating_add(info.size);
        }
        out.entries.push(info);
    }
    if out.total > limits.max_total {
        bail!("{} would expand to {} bytes (limit {})", path.display(), out.total, limits.max_total);
    }
    Ok(out)
}

/// Read one entry by its stored name into memory (at most `max` bytes).
pub fn read_zip_entry(path: &Path, name: &str, max: u64) -> Result<Vec<u8>> {
    let mut zip = open(path)?;
    let mut f = zip.by_name(name).with_context(|| format!("{name} not found in {}", path.display()))?;
    if f.size() > max {
        bail!("{name} is {} bytes (limit {max})", f.size());
    }
    let mut buf = Vec::new();
    let n = f.by_ref().take(max + 1).read_to_end(&mut buf)?;
    if n as u64 > max {
        bail!("{name} expands past {max} bytes");
    }
    Ok(buf)
}

/// Outcome of an extraction.
#[derive(Clone, Debug, Default)]
pub struct ExtractReport {
    /// Files written.
    pub files: usize,
    pub bytes: u64,
    /// Entries with an unsafe name (not written).
    pub unsafe_names: Vec<String>,
    /// Entries left out for another reason (symlink, encrypted, unreadable
    /// method, repeated name), each with the reason.
    pub skipped: Vec<(String, String)>,
}

/// Extract every safe file of `path` under `dest` (created). `progress(done,
/// total, name)` returns false to cancel (the call then fails). A name that
/// repeats an earlier entry (or differs only by case) is skipped, never
/// overwritten.
pub fn extract_zip(path: &Path, dest: &Path, limits: &ZipLimits, progress: &mut dyn FnMut(usize, usize, &str) -> bool) -> Result<ExtractReport> {
    let listing = list_zip(path, limits)?;
    let mut zip = open(path)?;
    std::fs::create_dir_all(dest).with_context(|| format!("creating {}", dest.display()))?;
    let mut report = ExtractReport::default();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let total = listing.entries.len();
    let mut written_total = 0u64;
    for (i, info) in listing.entries.iter().enumerate() {
        if !progress(i, total, &info.name) {
            bail!("cancelled");
        }
        if info.is_dir {
            continue;
        }
        let Some(rel) = &info.rel else {
            report.unsafe_names.push(info.name.clone());
            continue;
        };
        if info.symlink {
            report.skipped.push((info.name.clone(), "symbolic link".into()));
            continue;
        }
        if info.encrypted {
            report.skipped.push((info.name.clone(), "encrypted".into()));
            continue;
        }
        if !info.readable {
            report.skipped.push((info.name.clone(), "unsupported compression method".into()));
            continue;
        }
        if info.size > limits.max_entry {
            bail!("{} is {} bytes (limit {})", info.name, info.size, limits.max_entry);
        }
        if info.size >= RATIO_FLOOR && info.compressed > 0 && info.size / info.compressed > limits.max_ratio {
            bail!("{} expands {}x (limit {}x): not extracting what looks like a zip bomb", info.name, info.size / info.compressed, limits.max_ratio);
        }
        if !seen.insert(rel.to_string_lossy().to_lowercase()) {
            report.skipped.push((info.name.clone(), "repeats an earlier entry".into()));
            continue;
        }
        let target = dest.join(rel);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut f = match zip.by_index(i) {
            Ok(f) => f,
            // An encrypted entry cannot be opened without a password.
            Err(e) => {
                report.skipped.push((info.name.clone(), format!("unreadable ({e})")));
                continue;
            }
        };
        let mut out = std::fs::File::create(&target).with_context(|| format!("writing {}", target.display()))?;
        // The declared size can lie: count what actually comes out.
        let limit = limits.max_entry.min(limits.max_total.saturating_sub(written_total));
        let n = std::io::copy(&mut f.by_ref().take(limit + 1), &mut out)?;
        out.flush()?;
        if n > limit {
            drop(out);
            let _ = std::fs::remove_file(&target);
            bail!("{} expands past its limit ({limit} bytes)", info.name);
        }
        written_total += n;
        report.files += 1;
        report.bytes += n;
    }
    Ok(report)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    /// A unique scratch directory for one test.
    pub fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("zest_pack_{tag}_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Build a zip with the given (name, data, deflate) entries.
    pub fn make_zip(path: &Path, entries: &[(&str, &[u8], bool)]) {
        let file = std::fs::File::create(path).unwrap();
        let mut w = zip::ZipWriter::new(file);
        for (name, data, deflate) in entries {
            let method = if *deflate { zip::CompressionMethod::Deflated } else { zip::CompressionMethod::Stored };
            w.start_file(*name, zip::write::FileOptions::default().compression_method(method)).unwrap();
            w.write_all(data).unwrap();
        }
        w.finish().unwrap();
    }

    fn no_progress() -> impl FnMut(usize, usize, &str) -> bool {
        |_, _, _| true
    }

    #[test]
    fn safe_rel_rejects_escapes() {
        assert_eq!(safe_rel("a/b.txt"), Some(PathBuf::from("a/b.txt")));
        assert_eq!(safe_rel("./a//b.txt"), Some(PathBuf::from("a/b.txt")));
        assert_eq!(safe_rel("a\\b\\c.png"), Some(PathBuf::from("a/b/c.png")), "windows separators");
        for bad in ["../x", "a/../../x", "/etc/passwd", "a/../b/../../x", "..\\x", "\\abs", "", ".", "x\0y"] {
            assert_eq!(safe_rel(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn extracts_stored_and_deflate_and_blocks_zip_slip() {
        let dir = scratch("slip");
        let z = dir.join("t.zip");
        let big = vec![b'a'; 100_000];
        make_zip(&z, &[("ok/stored.txt", b"hello", false), ("ok/deflate.bin", &big, true), ("../evil.txt", b"x", false), ("/abs.txt", b"x", false), ("dir\\win.txt", b"w", false)]);
        let out = dir.join("out");
        let r = extract_zip(&z, &out, &ZipLimits::default(), &mut no_progress()).unwrap();
        assert_eq!(r.files, 3);
        assert_eq!(r.unsafe_names.len(), 2);
        assert_eq!(std::fs::read(out.join("ok/stored.txt")).unwrap(), b"hello");
        assert_eq!(std::fs::read(out.join("ok/deflate.bin")).unwrap().len(), 100_000);
        assert_eq!(std::fs::read(out.join("dir/win.txt")).unwrap(), b"w");
        assert!(!dir.join("evil.txt").exists());
        assert!(!Path::new("/abs.txt").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn limits_are_enforced() {
        let dir = scratch("limits");
        let z = dir.join("t.zip");
        make_zip(&z, &[("a.bin", &vec![1u8; 5000], false), ("b.bin", &vec![2u8; 5000], false)]);
        let small_entry = ZipLimits { max_entry: 4000, ..Default::default() };
        assert!(extract_zip(&z, &dir.join("o1"), &small_entry, &mut no_progress()).is_err(), "per-entry limit");
        let small_total = ZipLimits { max_total: 8000, ..Default::default() };
        assert!(list_zip(&z, &small_total).is_err(), "total limit");
        let few = ZipLimits { max_entries: 1, ..Default::default() };
        assert!(list_zip(&z, &few).is_err(), "entry-count limit");
        // A highly compressible entry above the floor trips the ratio check.
        let z2 = dir.join("bomb.zip");
        make_zip(&z2, &[("zeros.bin", &vec![0u8; 8 << 20], true)]);
        let strict = ZipLimits { max_ratio: 50, ..Default::default() };
        let e = extract_zip(&z2, &dir.join("o2"), &strict, &mut no_progress()).unwrap_err();
        assert!(format!("{e:#}").contains("zip bomb"), "{e:#}");
        // The same archive extracts under the default ratio.
        assert!(extract_zip(&z2, &dir.join("o3"), &ZipLimits { max_ratio: 100_000, ..Default::default() }, &mut no_progress()).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn repeated_names_do_not_overwrite_and_garbage_errors() {
        let dir = scratch("dup");
        let z = dir.join("t.zip");
        make_zip(&z, &[("A.txt", b"first", false), ("a.TXT", b"second", false)]);
        let out = dir.join("out");
        let r = extract_zip(&z, &out, &ZipLimits::default(), &mut no_progress()).unwrap();
        assert_eq!(r.files, 1);
        assert_eq!(r.skipped.len(), 1);
        assert_eq!(std::fs::read(out.join("A.txt")).unwrap(), b"first");
        std::fs::write(&z, b"not a zip").unwrap();
        assert!(list_zip(&z, &ZipLimits::default()).is_err());
        let cancel = extract_zip(&dir.join("missing.zip"), &out, &ZipLimits::default(), &mut no_progress());
        assert!(cancel.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reads_one_entry_with_a_cap() {
        let dir = scratch("read");
        let z = dir.join("t.zip");
        make_zip(&z, &[("Manifest.json", b"{\"a\":1}", false), ("big.bin", &vec![0u8; 4096], true)]);
        assert_eq!(read_zip_entry(&z, "Manifest.json", 1024).unwrap(), b"{\"a\":1}");
        assert!(read_zip_entry(&z, "big.bin", 100).is_err());
        assert!(read_zip_entry(&z, "nope", 100).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
