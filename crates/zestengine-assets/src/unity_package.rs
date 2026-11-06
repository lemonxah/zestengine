//! `.unitypackage` reader: a gzip-compressed tar holding, per asset, a
//! `<guid>/asset` (the data), `<guid>/asset.meta` and `<guid>/pathname` (its
//! `Assets/...` path). Extracting rebuilds that folder layout with the `.meta`
//! files beside the assets, which is exactly what the Unity material
//! converter indexes GUIDs from.

use std::io::Read;
use std::path::{Component, Path, PathBuf};

use anyhow::{Context as _, Result, bail};

/// Largest single entry accepted (a package can carry big textures / meshes).
const MAX_ENTRY: u64 = 1 << 30;

/// One tar member: (name, data). Streaming reader over any `Read`.
fn read_tar(mut r: impl Read, mut each: impl FnMut(String, Vec<u8>) -> Result<()>) -> Result<()> {
    let mut long_name: Option<String> = None;
    loop {
        let mut h = [0u8; 512];
        match r.read_exact(&mut h) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e.into()),
        }
        if h.iter().all(|b| *b == 0) {
            return Ok(());
        }
        let text = |a: usize, b: usize| -> String {
            let s = &h[a..b];
            let end = s.iter().position(|c| *c == 0).unwrap_or(s.len());
            String::from_utf8_lossy(&s[..end]).into_owned()
        };
        let size_txt = text(124, 136);
        let size = u64::from_str_radix(size_txt.trim().trim_matches('\0'), 8).unwrap_or(0);
        if size > MAX_ENTRY {
            bail!("tar entry of {size} bytes is too large");
        }
        let kind = h[156];
        let mut name = text(0, 100);
        if &h[257..262] == b"ustar" {
            let prefix = text(345, 500);
            if !prefix.is_empty() {
                name = format!("{prefix}/{name}");
            }
        }
        let mut data = vec![0u8; size as usize];
        r.read_exact(&mut data).context("truncated tar entry")?;
        let pad = (512 - (size % 512)) % 512;
        std::io::copy(&mut r.by_ref().take(pad), &mut std::io::sink())?;
        match kind {
            b'L' => long_name = Some(String::from_utf8_lossy(&data).trim_end_matches('\0').to_owned()),
            b'x' | b'g' => {} // pax headers: not needed for Unity's layout
            b'0' | 0 => each(long_name.take().unwrap_or(name), data)?,
            _ => {}
        }
    }
}

/// A relative path with no `..`, root or drive components.
fn safe_rel(p: &str) -> Option<PathBuf> {
    let path = Path::new(p.trim());
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::Normal(n) => out.push(n),
            Component::CurDir => {}
            _ => return None,
        }
    }
    (!out.as_os_str().is_empty()).then_some(out)
}

#[derive(Clone, Debug, Default)]
pub struct PackageExtract {
    /// Number of assets written.
    pub assets: usize,
    /// Entries skipped (unsafe or missing path).
    pub skipped: usize,
}

/// Extract `package` into `dest` as `dest/Assets/...` with `.meta` files.
pub fn extract_unitypackage(package: &Path, dest: &Path) -> Result<PackageExtract> {
    let file = std::fs::File::open(package).with_context(|| format!("opening {}", package.display()))?;
    let gz = flate2::read::GzDecoder::new(std::io::BufReader::new(file));
    let staging = dest.join("_pkg");
    std::fs::create_dir_all(&staging)?;
    let mut members = 0usize;
    read_tar(gz, |name, data| {
        members += 1;
        // `<guid>/asset`, `<guid>/asset.meta`, `<guid>/pathname`, `<guid>/preview.png`.
        let Some(rel) = safe_rel(&name) else { return Ok(()) };
        let parts: Vec<_> = rel.components().collect();
        if parts.len() != 2 {
            return Ok(());
        }
        let leaf = rel.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        if !matches!(leaf.as_str(), "asset" | "asset.meta" | "pathname") {
            return Ok(());
        }
        let target = staging.join(&rel);
        std::fs::create_dir_all(target.parent().unwrap_or(&staging))?;
        std::fs::write(&target, data)?;
        Ok(())
    })
    .context("reading the tar stream (is this a .unitypackage?)")?;
    if members == 0 {
        bail!("{} holds no entries (not a .unitypackage?)", package.display());
    }

    let mut out = PackageExtract::default();
    for e in std::fs::read_dir(&staging)?.flatten() {
        let dir = e.path();
        let Ok(path_txt) = std::fs::read_to_string(dir.join("pathname")) else {
            out.skipped += 1;
            continue;
        };
        // The pathname file's first line is the asset path.
        let Some(rel) = safe_rel(path_txt.lines().next().unwrap_or("")) else {
            out.skipped += 1;
            continue;
        };
        let target = dest.join(&rel);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // Folders have a pathname + meta but no asset file.
        if dir.join("asset").is_file() {
            std::fs::rename(dir.join("asset"), &target).or_else(|_| std::fs::copy(dir.join("asset"), &target).map(|_| ()))?;
            out.assets += 1;
        } else {
            std::fs::create_dir_all(&target)?;
        }
        if dir.join("asset.meta").is_file() {
            let mut m = target.clone().into_os_string();
            m.push(".meta");
            let _ = std::fs::copy(dir.join("asset.meta"), PathBuf::from(m));
        }
    }
    let _ = std::fs::remove_dir_all(&staging);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tar_entry(name: &str, data: &[u8]) -> Vec<u8> {
        let mut h = [0u8; 512];
        h[..name.len()].copy_from_slice(name.as_bytes());
        h[124..135].copy_from_slice(format!("{:011o}", data.len()).as_bytes());
        h[156] = b'0';
        let mut v = h.to_vec();
        v.extend_from_slice(data);
        v.resize(v.len().div_ceil(512) * 512, 0);
        v
    }

    fn package(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut tar = Vec::new();
        for (n, d) in entries {
            tar.extend(tar_entry(n, d));
        }
        tar.extend([0u8; 1024]);
        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        gz.write_all(&tar).unwrap();
        gz.finish().unwrap()
    }

    #[test]
    fn extracts_assets_with_meta_and_blocks_traversal() {
        let dir = std::env::temp_dir().join(format!("zest_upkg_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let g1 = "11111111111111111111111111111111";
        let g2 = "22222222222222222222222222222222";
        let pkg = package(&[
            (&format!("{g1}/asset"), b"MATDATA"),
            (&format!("{g1}/asset.meta"), format!("guid: {g1}\n").as_bytes()),
            (&format!("{g1}/pathname"), b"Assets/Mats/a.mat\n00"),
            (&format!("{g2}/asset"), b"evil"),
            (&format!("{g2}/pathname"), b"../../evil.txt"),
        ]);
        let file = dir.join("t.unitypackage");
        std::fs::write(&file, pkg).unwrap();
        let out = dir.join("out");
        let r = extract_unitypackage(&file, &out).unwrap();
        assert_eq!(r.assets, 1);
        assert_eq!(r.skipped, 1);
        assert_eq!(std::fs::read(out.join("Assets/Mats/a.mat")).unwrap(), b"MATDATA");
        assert!(out.join("Assets/Mats/a.mat.meta").is_file());
        assert!(!dir.join("evil.txt").exists());
        assert!(!out.join("_pkg").exists());
        // Garbage is an error, not a panic.
        std::fs::write(&file, b"not gzip").unwrap();
        assert!(extract_unitypackage(&file, &dir.join("o2")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
