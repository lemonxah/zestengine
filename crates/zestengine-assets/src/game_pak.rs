//! `assets.pak`: a shipped desktop game's cooked assets in one zip archive
//! beside the executable. At startup the game extracts it once into a
//! per-user cache folder named after the archive's id and runs from there,
//! so every loader keeps reading plain files (Android does the same with the
//! assets inside its APK).

use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail};

use crate::package::sha256;

/// File name of the archive next to the game executable.
pub const GAME_PAK_NAME: &str = "assets.pak";

/// Marker written into an extraction folder once every file is in place.
const COMPLETE_MARKER: &str = ".zestengine_pak_complete";

/// Formats that are already compressed: stored as-is instead of deflated.
const STORED_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "ktx2", "ogg", "mp3", "flac", "glb", "vrm", "cbc", "castc", "zip"];

/// Model formats the geometry cook looks for.
/// File extensions of the models the geometry cook builds cluster data for.
pub const MODEL_EXTENSIONS: &[&str] = &["glb", "gltf", "vrm", "fbx"];

/// Build the cluster DAG (see [`crate::geo_cache`]) of every model under
/// `assets_dir`, so the archive carries them in the models' `.zestengine_geocache/`
/// folders and the game never builds them on first load. Models that fail to
/// import, or have no mesh big enough, are skipped. Returns the number of
/// models that got one and the warnings.
pub fn cook_geometry(assets_dir: &Path) -> (usize, Vec<String>) {
    let mut files = Vec::new();
    if collect_files(assets_dir, assets_dir, &mut files).is_err() {
        return (0, vec![format!("could not list {}", assets_dir.display())]);
    }
    let (mut cooked, mut warnings) = (0, Vec::new());
    for rel in files.iter().filter(|r| !r.split('/').any(|p| p.starts_with('.'))) {
        let ext = rel.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase()).unwrap_or_default();
        if !MODEL_EXTENSIONS.contains(&ext.as_str()) {
            continue;
        }
        let path = assets_dir.join(rel);
        match crate::load_model_with_meta(&path) {
            Ok(scene) => {
                if let Some((_, _)) = crate::geo_cache::cook(&path, &scene) {
                    cooked += 1;
                }
            }
            Err(e) => warnings.push(format!("{rel}: not cooked ({e:#})")),
        }
    }
    (cooked, warnings)
}

/// Pack every file under `assets_dir` into a zip at `out`, after
/// [`cook_geometry`] has put each model's cluster data beside it. Returns the
/// number of files packed.
pub fn write_game_pak(assets_dir: &Path, out: &Path) -> Result<usize> {
    let (cooked, warnings) = cook_geometry(assets_dir);
    for w in &warnings {
        tracing::warn!("{w}");
    }
    if cooked > 0 {
        tracing::info!("cooked cluster geometry for {cooked} model(s)");
    }
    // Sprite atlases (2D-4): pack them so `.zestengine_atlascache/` ships beside each atlas.
    let atlases = crate::sprite_atlas::cook_atlases(assets_dir, crate::texture_import::Platform::Desktop);
    for w in &atlases.warnings {
        tracing::warn!("sprite atlas: {w}");
    }
    if atlases.packed + atlases.from_cache > 0 {
        tracing::info!("cooked {} sprite atlas(es) ({} from the cache)", atlases.packed + atlases.from_cache, atlases.from_cache);
    }
    pack_dir(assets_dir, out)
}

/// Pack every file under `assets_dir` into a zip at `out` as it is (no geometry or atlas cook):
/// the dedicated server's `assets.pak` (its models are already collision blobs). Returns the
/// number of files packed; [`extract_game_pak`] reads it back.
pub fn pack_dir(assets_dir: &Path, out: &Path) -> Result<usize> {
    let mut files = Vec::new();
    collect_files(assets_dir, assets_dir, &mut files)?;
    files.sort();
    let file = std::fs::File::create(out).with_context(|| format!("creating {}", out.display()))?;
    let mut zip = zip::ZipWriter::new(std::io::BufWriter::new(file));
    let stored = zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Stored).large_file(true);
    let deflated = zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Deflated).large_file(true);
    for rel in &files {
        let ext = rel.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase()).unwrap_or_default();
        let options = if STORED_EXTENSIONS.contains(&ext.as_str()) { stored } else { deflated };
        zip.start_file(rel.as_str(), options)?;
        let mut src = std::fs::File::open(assets_dir.join(rel)).with_context(|| format!("reading {rel}"))?;
        std::io::copy(&mut src, &mut zip).with_context(|| format!("packing {rel}"))?;
    }
    zip.finish()?.flush()?;
    Ok(files.len())
}

/// Id of an archive: a hash of every entry's name, size and CRC-32, read from
/// the zip directory alone (nothing is decompressed).
pub fn game_pak_id<R: Read + Seek>(archive: &mut zip::ZipArchive<R>) -> Result<String> {
    let mut buf = Vec::new();
    for i in 0..archive.len() {
        let entry = archive.by_index_raw(i)?;
        buf.extend_from_slice(entry.name().as_bytes());
        buf.push(0);
        buf.extend_from_slice(&entry.size().to_le_bytes());
        buf.extend_from_slice(&entry.crc32().to_le_bytes());
    }
    Ok(sha256::hex(&sha256::digest(&buf))[..16].to_owned())
}

/// Extract `pak` into `cache_dir/<id>` unless a complete extraction is already
/// there, remove extractions of other ids, and return the folder.
pub fn extract_game_pak(pak: &Path, cache_dir: &Path) -> Result<PathBuf> {
    let file = std::fs::File::open(pak).with_context(|| format!("opening {}", pak.display()))?;
    let mut archive = zip::ZipArchive::new(std::io::BufReader::new(file)).with_context(|| format!("{} is not a zip archive", pak.display()))?;
    let id = game_pak_id(&mut archive)?;
    let dest = cache_dir.join(&id);
    if !dest.join(COMPLETE_MARKER).is_file() {
        let staging = cache_dir.join(format!("{id}.partial"));
        let _ = std::fs::remove_dir_all(&staging);
        std::fs::create_dir_all(&staging).with_context(|| format!("creating {}", staging.display()))?;
        for i in 0..archive.len() {
            let mut entry = archive.by_index(i)?;
            if entry.is_dir() {
                continue;
            }
            let Some(rel) = entry.enclosed_name().map(Path::to_path_buf) else {
                bail!("{} has an entry outside the archive root: {}", pak.display(), entry.name());
            };
            let out = staging.join(&rel);
            if let Some(dir) = out.parent() {
                std::fs::create_dir_all(dir)?;
            }
            let mut writer = std::fs::File::create(&out).with_context(|| format!("extracting {}", rel.display()))?;
            std::io::copy(&mut entry, &mut writer).with_context(|| format!("extracting {}", rel.display()))?;
        }
        std::fs::write(staging.join(COMPLETE_MARKER), &id)?;
        let _ = std::fs::remove_dir_all(&dest);
        std::fs::rename(&staging, &dest).with_context(|| format!("moving extracted assets to {}", dest.display()))?;
    }
    if let Ok(entries) = std::fs::read_dir(cache_dir) {
        for entry in entries.flatten() {
            if entry.file_name() != id.as_str() && entry.path().is_dir() {
                let _ = std::fs::remove_dir_all(entry.path());
            }
        }
    }
    Ok(dest)
}

fn collect_files(base: &Path, dir: &Path, out: &mut Vec<String>) -> Result<()> {
    for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry?.path();
        if path.is_dir() {
            collect_files(base, &path, out)?;
        } else {
            let rel = path.strip_prefix(base).expect("walked under base");
            out.push(rel.to_string_lossy().replace('\\', "/"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn models_are_cooked_into_the_pak_and_extract_beside_the_model() {
        let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/DamagedHelmet.glb");
        if !src.is_file() {
            return;
        }
        let root = std::env::temp_dir().join(format!("zestengine-game-pak-geo-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let (assets, cache, pak) = (root.join("assets"), root.join("cache"), root.join(GAME_PAK_NAME));
        std::fs::create_dir_all(assets.join("models")).unwrap();
        std::fs::copy(&src, assets.join("models/DamagedHelmet.glb")).unwrap();
        std::fs::write(assets.join("project.zestengine"), "(name: \"pak\")").unwrap();
        let packed = write_game_pak(&assets, &pak).unwrap();
        assert!(packed >= 3, "model, project file and the cluster cache: {packed}");
        let dir = extract_game_pak(&pak, &cache).unwrap();
        let model = dir.join("models/DamagedHelmet.glb");
        let geo_file = crate::geo_cache::geo_path(&model).unwrap();
        let geo = crate::geo_cache::decode(&std::fs::read(&geo_file).expect("the cook shipped the cluster data")).unwrap();
        assert!(geo.clusters() > 100);
        let scene = crate::load_model_with_meta(&model).unwrap();
        assert_eq!(crate::geo_cache::load(&model, crate::geo_cache::model_key(&scene, &crate::cluster_dag::DagOptions::default())).as_ref(), Some(&geo));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn packs_extracts_once_and_replaces_old_versions() {
        let root = std::env::temp_dir().join(format!("zestengine-game-pak-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let (assets, cache, pak) = (root.join("assets"), root.join("cache"), root.join(GAME_PAK_NAME));
        std::fs::create_dir_all(assets.join("scenes")).unwrap();
        std::fs::write(assets.join("project.zestengine"), "(name: \"pak\")").unwrap();
        std::fs::write(assets.join("scenes/main.scene"), "scene text").unwrap();
        std::fs::write(assets.join("logo.png"), [137u8, 80, 78, 71]).unwrap();
        assert_eq!(write_game_pak(&assets, &pak).unwrap(), 3);

        let first = extract_game_pak(&pak, &cache).unwrap();
        assert_eq!(std::fs::read_to_string(first.join("scenes/main.scene")).unwrap(), "scene text");
        assert_eq!(std::fs::read(first.join("logo.png")).unwrap(), [137, 80, 78, 71]);
        // A second start reuses the extraction instead of rewriting it.
        std::fs::write(first.join("scenes/main.scene"), "edited in place").unwrap();
        assert_eq!(extract_game_pak(&pak, &cache).unwrap(), first);
        assert_eq!(std::fs::read_to_string(first.join("scenes/main.scene")).unwrap(), "edited in place");

        // A new build gets a new folder and the old one is removed.
        std::fs::write(assets.join("scenes/main.scene"), "patched").unwrap();
        write_game_pak(&assets, &pak).unwrap();
        let second = extract_game_pak(&pak, &cache).unwrap();
        assert_ne!(second, first);
        assert_eq!(std::fs::read_to_string(second.join("scenes/main.scene")).unwrap(), "patched");
        assert!(!first.exists());
        let _ = std::fs::remove_dir_all(&root);
    }
}
