//! Collision-only models for the dedicated-server cook (NET-3).
//!
//! A server never draws a model, so the cook replaces each model file with a
//! blob that keeps what gameplay reads: mesh positions and indices (mesh
//! colliders, ray casts), authored collision pieces, instances, skeletons and
//! animation clips (hitboxes, root motion, events), the sub-asset identity.
//! Dropped: normals, tangents, UVs, vertex colours, skin weights, blendshapes,
//! spring bones, materials' textures, every embedded texture.
//!
//! The blob is `ZSM1` + zstd of the import-cache entry of the stripped scene, and
//! lives at the model's own path: [`crate::load_model_cached`] recognises the
//! magic, so scenes, `.meta` sidecars and sub-asset ids keep working unchanged.

use std::path::Path;

use anyhow::{Context as _, Result, bail};

use crate::Scene;
use crate::import_cache;

/// First four bytes of a server model blob.
pub const MAGIC: &[u8; 4] = b"ZSM1";
const ZSTD_LEVEL: i32 = 9;

/// `scene` reduced to what a server needs.
pub fn strip_scene(mut scene: Scene) -> Scene {
    for m in &mut scene.meshes {
        for v in &mut m.vertices {
            *v = zestengine_render::Vertex { position: v.position, ..Default::default() };
        }
        m.has_lightmap_uv = false;
    }
    scene.morphs.clear();
    scene.springs = Default::default();
    scene.textures.clear();
    for m in &mut scene.materials {
        m.albedo = None;
        m.normal = None;
        m.orm = None;
        m.emission = None;
    }
    scene
}

/// Encode `scene` (stripped by the caller or not) as a blob.
pub fn encode(scene: &Scene) -> Result<Vec<u8>> {
    let entry = import_cache::encode(scene, Path::new("."), &[], "");
    let mut out = MAGIC.to_vec();
    out.extend(zstd::bulk::compress(&entry, ZSTD_LEVEL).context("compressing the server model")?);
    Ok(out)
}

/// Decode a blob.
pub fn decode(bytes: &[u8]) -> Result<Scene> {
    if !bytes.starts_with(MAGIC) {
        bail!("not a server model blob");
    }
    let mut dec = zstd::bulk::Decompressor::new()?;
    let entry = dec.decompress(&bytes[4..], 1 << 30).context("inflating the server model")?;
    import_cache::decode(&entry, Path::new("."))
}

/// Whether the file at `path` is a server model blob (reads 4 bytes).
pub fn is_blob(path: &Path) -> bool {
    use std::io::Read as _;
    let mut head = [0u8; 4];
    std::fs::File::open(path).and_then(|mut f| f.read_exact(&mut head)).is_ok() && &head == MAGIC
}

/// Import `src` (a real model) and write its collision-only blob to `dst`.
/// Returns the sizes `(source bytes, blob bytes)`.
pub fn cook_model(src: &Path, dst: &Path) -> Result<(u64, u64)> {
    let scene = crate::load_model_with_meta(src).with_context(|| format!("importing {}", src.display()))?;
    let blob = encode(&strip_scene(scene))?;
    if let Some(dir) = dst.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(dst, &blob).with_context(|| format!("writing {}", dst.display()))?;
    Ok((std::fs::metadata(src).map(|m| m.len()).unwrap_or(0), blob.len() as u64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blob_keeps_positions_and_drops_the_rest() {
        let mut scene = crate::test_scene();
        // Give the first mesh something to lose.
        for v in &mut scene.meshes[0].vertices {
            v.normal = [0.0, 1.0, 0.0];
            v.uv = [0.5, 0.5];
            v.color = [1.0, 0.0, 0.0, 1.0];
        }
        let positions: Vec<[f32; 3]> = scene.meshes[0].vertices.iter().map(|v| v.position).collect();
        let indices = scene.meshes[0].indices.clone();
        let instances = scene.instances.len();
        let blob = encode(&strip_scene(scene)).unwrap();
        assert!(blob.starts_with(MAGIC));
        let back = decode(&blob).unwrap();
        assert_eq!(back.meshes[0].vertices.iter().map(|v| v.position).collect::<Vec<_>>(), positions);
        assert_eq!(back.meshes[0].indices, indices);
        let plain = zestengine_render::Vertex::default();
        assert!(back.meshes[0].vertices.iter().all(|v| v.normal == plain.normal && v.uv == [0.0; 2] && v.color == plain.color), "normals, uvs and colours are reset");
        assert!(back.textures.is_empty() && back.morphs.is_empty());
        assert_eq!(back.instances.len(), instances);
        assert!(decode(b"not a blob").is_err());
    }
}
