//! Stored automatic LOD chains (GEO-2), beside the model's import-cache entry.
//!
//! The chain is derived from the imported scene by the simplifier, which is
//! slow on big models, so the result is kept in
//! `<dir>/.zestengine_modelcache/<file>.<key>.lod.zmc`. `<key>` is the model's
//! import-cache key, so any change to the source or its importer settings is a
//! different file; the model entry writer already deletes stale `<file>.*zmc`
//! files. The header carries [`LOD_FORMAT`] and the chain's shape (the
//! [`AUTO_LOD_STEPS`] shares), so a changed simplifier or step table is
//! rebuilt. An empty chain is stored too ("this model gets none"), so small
//! models are not re-imported every session to find that out.
//!
//! Layout: `ZML1`, u32 format, u32 steps, f32 shares of the steps, u32 levels,
//! then per level f32 share, u64 length, and an import-cache scene entry.

use std::path::{Path, PathBuf};

use zestengine_core::lod::AUTO_LOD_STEPS;

use crate::Scene;
use crate::import_cache::{CACHE_DIR, CACHE_EXTENSION, decode, encode};

/// Bump when the simplifier or the level layout changes what a chain contains.
pub const LOD_FORMAT: u32 = 3;
const MAGIC: &[u8; 4] = b"ZML1";

/// A chain as the engine installs it: share of triangles kept, level scene.
pub type Chain = Vec<(f32, Scene)>;

/// `<dir>/.zestengine_modelcache/<file>.<key>.lod.zmc`.
pub fn lod_path(source: &Path, key: &str) -> Option<PathBuf> {
    let name = source.file_name()?.to_string_lossy();
    Some(source.parent()?.join(CACHE_DIR).join(format!("{name}.{key}.lod.{CACHE_EXTENSION}")))
}

fn header(levels: usize) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(MAGIC);
    b.extend_from_slice(&LOD_FORMAT.to_le_bytes());
    b.extend_from_slice(&(AUTO_LOD_STEPS.len() as u32).to_le_bytes());
    for (keep, _) in AUTO_LOD_STEPS {
        b.extend_from_slice(&keep.to_le_bytes());
    }
    b.extend_from_slice(&(levels as u32).to_le_bytes());
    b
}

/// Serialise a chain (possibly empty).
pub fn encode_chain(chain: &[(f32, Scene)], source: &Path) -> Vec<u8> {
    let mut b = header(chain.len());
    for (keep, scene) in chain {
        let blob = encode(scene, source, &[], "");
        b.extend_from_slice(&keep.to_le_bytes());
        b.extend_from_slice(&(blob.len() as u64).to_le_bytes());
        b.extend_from_slice(&blob);
    }
    b
}

/// Parse a chain; `None` for anything stale, truncated or from another format.
pub fn decode_chain(bytes: &[u8], source: &Path) -> Option<Chain> {
    let head = header(0);
    // Magic + format + step table must match exactly; the level count follows.
    let fixed = head.len() - 4;
    if bytes.len() < head.len() || bytes[..fixed] != head[..fixed] {
        return None;
    }
    let mut at = fixed;
    let take = |at: &mut usize, n: usize| -> Option<&[u8]> {
        let s = bytes.get(*at..*at + n)?;
        *at += n;
        Some(s)
    };
    let levels = u32::from_le_bytes(take(&mut at, 4)?.try_into().ok()?) as usize;
    if levels > AUTO_LOD_STEPS.len() {
        return None;
    }
    let mut out = Vec::with_capacity(levels);
    for _ in 0..levels {
        let keep = f32::from_le_bytes(take(&mut at, 4)?.try_into().ok()?);
        let len = u64::from_le_bytes(take(&mut at, 8)?.try_into().ok()?) as usize;
        let scene = decode(take(&mut at, len)?, source).ok()?;
        out.push((keep, scene));
    }
    (at == bytes.len()).then_some(out)
}

/// The stored chain for `source` under import-cache `key`. `Some(vec![])` =
/// stored "no chain"; `None` = nothing usable stored.
pub fn load(source: &Path, key: &str) -> Option<Chain> {
    let bytes = std::fs::read(lod_path(source, key)?).ok()?;
    decode_chain(&bytes, source)
}

/// Store a chain (empty = "no chain"). Failures are ignored: a cache that can't
/// be written just means the chain is rebuilt next session.
pub fn store(source: &Path, key: &str, chain: &[(f32, Scene)]) {
    let Some(path) = lod_path(source, key) else { return };
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Err(e) = crate::write_atomic(&path, &encode_chain(chain, source)) {
        tracing::debug!("LOD cache not written: {e:#}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere_scene() -> Scene {
        Scene {
            meshes: vec![crate::procedural::primitive_mesh(crate::PrimitiveShape::Sphere)],
            morphs: Vec::new(),
            springs: Default::default(),
            textures: Vec::new(),
            materials: Vec::new(),
            instances: vec![crate::Instance::single("m", 0, 0, glam::Mat4::IDENTITY)],
            skeletons: Vec::new(),
            animations: Vec::new(),
            collision: Vec::new(),
            identity: Default::default(),
        }
    }

    #[test]
    fn chain_round_trips_and_rejects_stale_headers() {
        let src = Path::new("models/a.glb");
        let chain = vec![(0.5, sphere_scene()), (0.25, sphere_scene())];
        let bytes = encode_chain(&chain, src);
        let back = decode_chain(&bytes, src).expect("decodes");
        assert_eq!(back.len(), 2);
        assert_eq!(back[1].0, 0.25);
        assert_eq!(back[0].1.meshes[0].indices, chain[0].1.meshes[0].indices);
        // Another format version or a truncated file is a miss, not a panic.
        let mut stale = bytes.clone();
        stale[4] ^= 0xff;
        assert!(decode_chain(&stale, src).is_none());
        assert!(decode_chain(&bytes[..bytes.len() - 5], src).is_none());
        assert!(decode_chain(&[], src).is_none());
    }

    #[test]
    fn an_empty_chain_is_a_stored_answer() {
        let src = Path::new("models/small.glb");
        let bytes = encode_chain(&[], src);
        assert_eq!(decode_chain(&bytes, src).map(|c| c.len()), Some(0));
    }

    #[test]
    fn store_then_load_through_the_cache_dir() {
        let dir = std::env::temp_dir().join(format!("zest_lod_cache_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("m.glb");
        std::fs::write(&src, b"x").unwrap();
        assert!(load(&src, "k1").is_none());
        store(&src, "k1", &[(0.5, sphere_scene())]);
        assert_eq!(load(&src, "k1").map(|c| c.len()), Some(1));
        assert!(load(&src, "k2").is_none(), "another import key is another entry");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
