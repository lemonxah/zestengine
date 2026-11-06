//! Asset references by ID. An asset slot stores `@<id>` where the id is the
//! stable GUID in the asset's `.meta` sidecar (`foo.physmat.meta`), so moving
//! or renaming the file (the editor carries the sidecar along) keeps every
//! slot pointing at it. Plain paths from older files still resolve as paths.
//!
//! The id -> path index is built per project root by scanning its `.meta`
//! files once, kept current as the editor moves files, and rescanned when an
//! id isn't found.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const META_SUFFIX: &str = ".meta";

#[derive(Default)]
struct Index {
    by_id: HashMap<String, PathBuf>,
    by_path: HashMap<PathBuf, String>,
}

static INDEX: Mutex<Option<HashMap<PathBuf, Index>>> = Mutex::new(None);

/// Whether `s` is an id reference (`@<id>`).
pub fn is_ref(s: &str) -> bool {
    s.starts_with('@')
}

fn meta_of(asset: &Path) -> PathBuf {
    let mut name = asset.as_os_str().to_owned();
    name.push(META_SUFFIX);
    PathBuf::from(name)
}

/// The `id: "..."` value of a `.meta` file's text.
fn parse_id(text: &str) -> Option<String> {
    let rest = &text[text.find("id:")? + 3..];
    let start = rest.find('"')? + 1;
    let end = start + rest[start..].find('"')?;
    Some(rest[start..end].to_owned())
}

fn scan(dir: &Path, index: &mut Index) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') || name == "target" {
            continue;
        }
        let path = entry.path();
        if path.is_dir() {
            scan(&path, index);
        } else if let Some(asset) = name.strip_suffix(META_SUFFIX)
            && let Some(id) = std::fs::read_to_string(&path).ok().as_deref().and_then(parse_id)
        {
            let asset = path.with_file_name(asset);
            index.by_path.insert(asset.clone(), id.clone());
            index.by_id.insert(id, asset);
        }
    }
}

fn with_index<R>(root: &Path, rescan: bool, f: impl FnOnce(&mut Index) -> R) -> R {
    let mut guard = INDEX.lock().unwrap_or_else(|e| e.into_inner());
    let all = guard.get_or_insert_with(HashMap::new);
    let root = root.to_path_buf();
    if rescan || !all.contains_key(&root) {
        let mut index = Index::default();
        scan(&root, &mut index);
        all.insert(root.clone(), index);
    }
    f(all.get_mut(&root).expect("indexed above"))
}

/// The project-relative path `s` points at: an `@<id>` resolved through the
/// `.meta` index, or `s` itself when it's a plain path. None for an id no
/// asset carries.
pub fn resolve(root: &Path, s: &str) -> Option<String> {
    let Some(id) = s.strip_prefix('@') else {
        return Some(s.to_owned());
    };
    let rel = |p: &PathBuf| p.strip_prefix(root).unwrap_or(p).to_string_lossy().replace('\\', "/");
    if let Some(p) = with_index(root, false, |i| i.by_id.get(id).map(rel)) {
        return Some(p);
    }
    // Not seen yet (made outside the editor, or a fresh checkout): look again.
    with_index(root, true, |i| i.by_id.get(id).map(rel))
}

/// [`resolve`] as a full path (a missing id resolves to the raw text, which
/// then fails to load like any missing file).
pub fn resolve_path(root: &Path, s: &str) -> PathBuf {
    root.join(resolve(root, s).unwrap_or_else(|| s.to_owned()))
}

/// [`resolve_path`] for a reference held as a path (texture slots).
pub fn resolve_rel_path(root: &Path, rel: &Path) -> PathBuf {
    resolve_path(root, &rel.to_string_lossy())
}

/// An `@<id>` reference to the project-relative file `rel`, writing its
/// `.meta` sidecar (with a new id) when it has none. Falls back to the plain
/// path if the sidecar can't be written.
pub fn reference(root: &Path, rel: &str) -> String {
    if is_ref(rel) {
        return rel.to_owned();
    }
    let asset = root.join(rel);
    if !asset.is_file() {
        return rel.to_owned();
    }
    if let Some(id) = with_index(root, false, |i| i.by_path.get(&asset).cloned()) {
        return format!("@{id}");
    }
    let meta = meta_of(&asset);
    let id = match std::fs::read_to_string(&meta).ok().as_deref().and_then(parse_id) {
        Some(id) => id,
        None => {
            let id = crate::ObjectId::new().to_string();
            if std::fs::write(&meta, format!("(\n    id: \"{id}\",\n)\n")).is_err() {
                return rel.to_owned();
            }
            id
        }
    };
    with_index(root, false, |i| {
        i.by_path.insert(asset.clone(), id.clone());
        i.by_id.insert(id.clone(), asset);
    });
    format!("@{id}")
}

/// A file or folder moved from `old` to `new` (absolute paths): carry its
/// sidecar along and keep every index pointing at the new place.
pub fn moved(old: &Path, new: &Path) {
    let (old_meta, new_meta) = (meta_of(old), meta_of(new));
    if old_meta.is_file() && !new_meta.exists() {
        let _ = std::fs::rename(&old_meta, &new_meta);
    }
    let mut guard = INDEX.lock().unwrap_or_else(|e| e.into_inner());
    for index in guard.get_or_insert_with(HashMap::new).values_mut() {
        let hits: Vec<PathBuf> = index.by_path.keys().filter(|p| p.starts_with(old)).cloned().collect();
        for p in hits {
            let Some(id) = index.by_path.remove(&p) else { continue };
            let to = new.join(p.strip_prefix(old).unwrap_or(Path::new("")));
            let to = if p == old { new.to_path_buf() } else { to };
            index.by_id.insert(id.clone(), to.clone());
            index.by_path.insert(to, id);
        }
    }
}

/// A file was deleted: drop its sidecar too.
pub fn deleted(path: &Path) {
    let _ = std::fs::remove_file(meta_of(path));
    let mut guard = INDEX.lock().unwrap_or_else(|e| e.into_inner());
    for index in guard.get_or_insert_with(HashMap::new).values_mut() {
        let gone: Vec<PathBuf> = index.by_path.keys().filter(|p| p.starts_with(path)).cloned().collect();
        for p in gone {
            if let Some(id) = index.by_path.remove(&p) {
                index.by_id.remove(&id);
            }
        }
    }
}

/// A `.meta` file's text with a new id (for a copied asset).
pub fn with_fresh_id(text: &str) -> String {
    match parse_id(text) {
        Some(old) => text.replacen(&old, &crate::ObjectId::new().to_string(), 1),
        None => text.to_owned(),
    }
}

/// Whether `path` is a `.meta` sidecar (hidden in the file browser).
pub fn is_sidecar(path: &Path) -> bool {
    path.to_string_lossy().ends_with(META_SUFFIX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn references_survive_a_rename() {
        let root = std::env::temp_dir().join(format!("zestengine-asset-ref-{}", std::process::id()));
        std::fs::create_dir_all(root.join("physics")).unwrap();
        std::fs::write(root.join("physics/rubber.physmat"), "()").unwrap();
        let r = reference(&root, "physics/rubber.physmat");
        assert!(is_ref(&r));
        assert_eq!(reference(&root, "physics/rubber.physmat"), r, "same file, same id");
        assert_eq!(resolve(&root, &r).as_deref(), Some("physics/rubber.physmat"));
        let (old, new) = (root.join("physics/rubber.physmat"), root.join("physics/bouncy.physmat"));
        std::fs::rename(&old, &new).unwrap();
        moved(&old, &new);
        assert!(root.join("physics/bouncy.physmat.meta").is_file(), "the sidecar moved with it");
        assert_eq!(resolve(&root, &r).as_deref(), Some("physics/bouncy.physmat"));
        // A fresh index (another session) finds it from the sidecar.
        INDEX.lock().unwrap().as_mut().unwrap().remove(&root);
        assert_eq!(resolve(&root, &r).as_deref(), Some("physics/bouncy.physmat"));
        assert_eq!(resolve(&root, "audio/old.wav").as_deref(), Some("audio/old.wav"), "plain paths still work");
        deleted(&new);
        assert!(!root.join("physics/bouncy.physmat.meta").exists());
        let _ = std::fs::remove_dir_all(&root);
    }
}
