//! ZestSurface round-trip over `<project>/.zestsurface/<object id>/`.
//!
//! `export_object` writes `zestengine.json` and returns the model to open; ZestSurface
//! writes `zestengine_return.json` + maps back; `apply_return` rebinds them. The JSON
//! field names are a shared contract with the ZestSurface project.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use zestengine_engine::scene::{LoadedScene, RenderInfo};

pub const EXCHANGE_DIR: &str = ".zestsurface";
pub const MANIFEST_NAME: &str = "zestengine.json";
pub const RETURN_NAME: &str = "zestengine_return.json";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapKind {
    BaseColor,
    Normal,
    Roughness,
    Metallic,
    Ao,
    Emissive,
    Opacity,
}

impl MapKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BaseColor => "BaseColor",
            Self::Normal => "Normal",
            Self::Roughness => "Roughness",
            Self::Metallic => "Metallic",
            Self::Ao => "Ao",
            Self::Emissive => "Emissive",
            Self::Opacity => "Opacity",
        }
    }

    fn from_str(s: &str) -> Option<Self> {
        Some(match s {
            "BaseColor" => Self::BaseColor,
            "Normal" => Self::Normal,
            "Roughness" => Self::Roughness,
            "Metallic" => Self::Metallic,
            "Ao" => Self::Ao,
            "Emissive" => Self::Emissive,
            "Opacity" => Self::Opacity,
            _ => return None,
        })
    }

    // Slot index in `MaterialTexturePaths` order + whether it loads as sRGB.
    fn zestengine_slot(self) -> (usize, bool) {
        match self {
            Self::BaseColor => (0, true),
            Self::Normal => (1, false),
            Self::Emissive => (3, true),
            Self::Opacity => (4, false),
            Self::Ao => (12, false),
            Self::Roughness => (13, false),
            Self::Metallic => (14, false),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ZestEngineManifest {
    pub object_id: String,
    pub object_name: String,
    /// Number of material slots (ZestSurface texture sets) on the object.
    pub slot_count: usize,
    /// Absolute directory ZestSurface writes `zestengine_return.json` + maps into.
    pub return_dir: PathBuf,
    /// Starting maps (the applied material's textures), shaped like the return.
    #[serde(default)]
    pub textures: Vec<ReturnMap>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct ZestEngineReturn {
    pub object_id: String,
    pub maps: Vec<ReturnMap>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ReturnMap {
    /// Material slot index (0..`slot_count`), in the manifest's order.
    pub slot: usize,
    pub kind: String,
    /// File name relative to the return directory.
    pub file: String,
}

pub fn exchange_dir(project_root: &Path, object_id: &zestengine_core::ObjectId) -> PathBuf {
    project_root.join(EXCHANGE_DIR).join(object_id.to_string())
}

pub fn manifest_path(project_root: &Path, object_id: &zestengine_core::ObjectId) -> PathBuf {
    exchange_dir(project_root, object_id).join(MANIFEST_NAME)
}

pub fn return_path(project_root: &Path, object_id: &zestengine_core::ObjectId) -> PathBuf {
    exchange_dir(project_root, object_id).join(RETURN_NAME)
}

/// Write `zestengine.json` and return the model ZestSurface should open (`None` if the
/// object has no paintable mesh).
pub fn export_object(
    scene: &LoadedScene,
    project_root: &Path,
    object_index: usize,
) -> anyhow::Result<Option<PathBuf>> {
    let obj = &scene.objects[object_index];
    let slots: Vec<RenderInfo> = obj.render_slots().collect();
    if slots.is_empty() {
        return Ok(None);
    }

    let dir = exchange_dir(project_root, &obj.id);
    std::fs::create_dir_all(&dir)?;
    let textures = export_applied_textures(scene, project_root, object_index, &dir);

    let manifest = ZestEngineManifest {
        object_id: obj.id.to_string(),
        object_name: obj.name.clone(),
        slot_count: slots.len(),
        return_dir: dir.clone(),
        textures,
    };
    std::fs::write(dir.join(MANIFEST_NAME), serde_json::to_string_pretty(&manifest)?)?;

    let model = match &obj.source {
        zestengine_assets::ObjectSource::Model { path, .. } => Some(project_root.join(path)),
        zestengine_assets::ObjectSource::Primitive { shape } => {
            let mesh = zestengine_assets::primitive_mesh(*shape);
            let gltf = dir.join("model.gltf");
            write_gltf(&mesh, &gltf)?;
            Some(gltf)
        }
        _ => None,
    };
    Ok(model)
}

pub fn launch(model: &Path, manifest: &Path) -> std::io::Result<std::process::Child> {
    std::process::Command::new("zestsurface")
        .arg(model)
        .arg("--zestengine")
        .arg(manifest)
        .spawn()
}

pub fn poll_return(project_root: &Path, object_id: &zestengine_core::ObjectId) -> Option<ZestEngineReturn> {
    let path = return_path(project_root, object_id);
    let text = std::fs::read_to_string(&path).ok()?;
    match serde_json::from_str(&text) {
        Ok(ret) => Some(ret),
        Err(e) => {
            tracing::warn!("ZestSurface return manifest {} is invalid: {e:#}", path.display());
            None
        }
    }
}

pub fn apply_return(
    scene: &mut LoadedScene,
    renderer: &mut zestengine_render::Renderer,
    shaders: &mut zestengine_engine::shaders::ShaderLibrary,
    project_root: &Path,
    ret: &ZestEngineReturn,
) -> anyhow::Result<usize> {
    let Some(object_index) = scene
        .objects
        .iter()
        .position(|o| o.id.to_string() == ret.object_id)
    else {
        anyhow::bail!("ZestSurface returned for an object no longer in the scene");
    };
    let slots: Vec<RenderInfo> = scene.objects[object_index].render_slots().collect();
    if slots.is_empty() {
        return Ok(0);
    }

    let return_dir = exchange_dir(project_root, &scene.objects[object_index].id);
    let obj_name = sanitize_name(&scene.objects[object_index].name);
    let textures_root = project_root.join("textures").join("zestsurface").join(&obj_name);
    std::fs::create_dir_all(&textures_root)?;

    let mut applied = 0;
    for map in &ret.maps {
        let Some(kind) = MapKind::from_str(&map.kind) else {
            tracing::warn!("ZestSurface returned unknown map kind {:?}", map.kind);
            continue;
        };
        let Some(render) = slots.get(map.slot) else {
            tracing::warn!("ZestSurface returned map for slot {} (out of range)", map.slot);
            continue;
        };
        let (zestengine_slot, _srgb) = kind.zestengine_slot();
        let src = return_dir.join(&map.file);
        let dest = textures_root.join(format!("{}_{}.png", map.slot, kind.as_str()));
        if let Err(e) = std::fs::copy(&src, &dest) {
            tracing::warn!("copying ZestSurface map {}: {e:#}", src.display());
            continue;
        }

        let material = &mut scene.materials[render.material];
        let rel = zestengine_engine::scene::relative_to(&dest, project_root);
        if let Some(slot_path) = material.model.textures.slot_mut(zestengine_slot) {
            *slot_path = Some(PathBuf::from(rel));
        }
        scene.apply_material(renderer, shaders, project_root, render.material);
        applied += 1;
    }
    Ok(applied)
}

fn sanitize_name(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    if cleaned.is_empty() {
        "object".to_owned()
    } else {
        cleaned
    }
}

fn slot_path(paths: &zestengine_core::MaterialTexturePaths, slot: usize) -> Option<&PathBuf> {
    match slot {
        0 => paths.albedo.as_ref(),
        1 => paths.normal.as_ref(),
        2 => paths.orm.as_ref(),
        3 => paths.emission.as_ref(),
        4 => paths.opacity.as_ref(),
        12 => paths.ao.as_ref(),
        13 => paths.roughness.as_ref(),
        14 => paths.metallic.as_ref(),
        _ => None,
    }
}

fn load_texture_file(project_root: &Path, rel: &Path) -> Option<zestengine_render::TextureData> {
    let abs = zestengine_core::asset_ref::resolve_rel_path(project_root, rel);
    let img = image::open(&abs).ok()?.to_rgba8();
    let (w, h) = img.dimensions();
    Some(zestengine_render::TextureData {
        width: w,
        height: h,
        pixels: img.into_raw(),
        srgb: true,
        hdr: false,
    })
}

fn slot_texture(
    project_root: &Path,
    material: &zestengine_engine::scene::MaterialEntry,
    slot: usize,
) -> Option<zestengine_render::TextureData> {
    if let Some(path) = slot_path(&material.model.textures, slot) {
        return load_texture_file(project_root, path);
    }
    material.base_tex_data.get(slot).and_then(|d| d.clone())
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as i32;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let mant = (bits & 0x3ff) as f32;
    let v = if exp == 0 {
        mant / 1024.0
    } else if exp == 31 {
        if mant == 0.0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1.0 + mant / 1024.0) * 2f32.powi(exp - 15)
    };
    if sign == 1 { -v } else { v }
}

fn to_rgba8(data: &zestengine_render::TextureData) -> Vec<u8> {
    if !data.hdr {
        return data.pixels.clone();
    }
    let n = data.pixels.len() / 8;
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        for c in 0..4 {
            let lo = data.pixels[i * 8 + c * 2] as u16;
            let hi = data.pixels[i * 8 + c * 2 + 1] as u16;
            let f = f16_to_f32((hi << 8) | lo);
            out.push((f.clamp(0.0, 1.0) * 255.0).round() as u8);
        }
    }
    out
}

fn extract_channel(data: &zestengine_render::TextureData, channel: usize) -> zestengine_render::TextureData {
    let rgba = to_rgba8(data);
    let n = (data.width as usize) * (data.height as usize);
    let mut out = Vec::with_capacity(n * 4);
    for p in 0..n {
        let v = rgba[p * 4 + channel];
        out.extend_from_slice(&[v, v, v, 255]);
    }
    zestengine_render::TextureData {
        width: data.width,
        height: data.height,
        pixels: out,
        srgb: false,
        hdr: false,
    }
}

fn write_png(path: &Path, data: &zestengine_render::TextureData) -> bool {
    let rgba = to_rgba8(data);
    match image::RgbaImage::from_raw(data.width, data.height, rgba) {
        Some(img) => img.save(path).is_ok(),
        None => false,
    }
}

fn resolve_kind(
    project_root: &Path,
    material: &zestengine_engine::scene::MaterialEntry,
    kind: MapKind,
) -> Option<zestengine_render::TextureData> {
    let (slot, _srgb) = kind.zestengine_slot();
    if let Some(data) = slot_texture(project_root, material, slot) {
        return Some(data);
    }
    match kind {
        MapKind::Ao | MapKind::Roughness | MapKind::Metallic => {
            let orm = slot_texture(project_root, material, 2)?;
            let channel = match kind {
                MapKind::Ao => 0,
                MapKind::Roughness => 1,
                _ => 2,
            };
            Some(extract_channel(&orm, channel))
        }
        _ => None,
    }
}

fn export_applied_textures(
    scene: &LoadedScene,
    project_root: &Path,
    object_index: usize,
    dir: &Path,
) -> Vec<ReturnMap> {
    let mut out = Vec::new();
    let slots: Vec<RenderInfo> = scene.objects[object_index].render_slots().collect();
    for (slot, render) in slots.iter().enumerate() {
        let material = &scene.materials[render.material];
        for kind in [
            MapKind::BaseColor,
            MapKind::Normal,
            MapKind::Emissive,
            MapKind::Opacity,
            MapKind::Ao,
            MapKind::Roughness,
            MapKind::Metallic,
        ] {
            let Some(data) = resolve_kind(project_root, material, kind) else {
                continue;
            };
            let file = format!("{}_{}.png", slot, kind.as_str());
            if write_png(&dir.join(&file), &data) {
                out.push(ReturnMap {
                    slot,
                    kind: kind.as_str().into(),
                    file,
                });
            }
        }
    }
    out
}

/// Minimal glTF (`.gltf` JSON + sidecar `.bin`): positions, normals, UV0.
fn write_gltf(mesh: &zestengine_render::MeshData, path: &Path) -> anyhow::Result<()> {
    use std::io::Write;

    let vertex_count = mesh.vertices.len();
    let index_count = mesh.indices.len();

    // Interleaved vertex stride: position 12B + normal 12B + uv 8B.
    const STRIDE: usize = 32;
    const POSITION_OFF: usize = 0;
    const NORMAL_OFF: usize = 12;
    const UV_OFF: usize = 24;
    let vertices_bytes = vertex_count * STRIDE;
    let indices_offset = vertices_bytes;
    let total_bytes = indices_offset + index_count * 4;

    let mut bin = Vec::with_capacity(total_bytes);
    for v in &mesh.vertices {
        for f in v.position {
            bin.extend_from_slice(&f.to_le_bytes());
        }
        for f in v.normal {
            bin.extend_from_slice(&f.to_le_bytes());
        }
        for f in v.uv {
            bin.extend_from_slice(&f.to_le_bytes());
        }
    }
    for i in &mesh.indices {
        bin.extend_from_slice(&i.to_le_bytes());
    }

    let bin_path = path.with_extension("bin");
    let bin_name = bin_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model.bin");

    let (min, max) = mesh.vertices.iter().map(|v| v.position).fold(
        ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]),
        |(mn, mx), p| {
            (
                [mn[0].min(p[0]), mn[1].min(p[1]), mn[2].min(p[2])],
                [mx[0].max(p[0]), mx[1].max(p[1]), mx[2].max(p[2])],
            )
        },
    );

    let json = serde_json::json!({
        "asset": { "version": "2.0" },
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": [{ "mesh": 0, "name": "ZestEngineExport" }],
        "meshes": [{
            "name": "ZestEngineExport",
            "primitives": [{
                "attributes": { "POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2 },
                "indices": 3,
                "material": 0
            }]
        }],
        "materials": [{
            "name": "ZestEngineExport",
            "pbrMetallicRoughness": { "baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.0, "roughnessFactor": 1.0 }
        }],
        "buffers": [{ "uri": bin_name, "byteLength": total_bytes }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": POSITION_OFF, "byteLength": (vertex_count - 1) * STRIDE + 12, "byteStride": STRIDE, "target": 34962 },
            { "buffer": 0, "byteOffset": NORMAL_OFF, "byteLength": (vertex_count - 1) * STRIDE + 12, "byteStride": STRIDE, "target": 34962 },
            { "buffer": 0, "byteOffset": UV_OFF, "byteLength": (vertex_count - 1) * STRIDE + 8, "byteStride": STRIDE, "target": 34962 },
            { "buffer": 0, "byteOffset": indices_offset, "byteLength": index_count * 4, "target": 34963 }
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": vertex_count, "type": "VEC3", "min": min, "max": max },
            { "bufferView": 1, "componentType": 5126, "count": vertex_count, "type": "VEC3" },
            { "bufferView": 2, "componentType": 5126, "count": vertex_count, "type": "VEC2" },
            { "bufferView": 3, "componentType": 5125, "count": index_count, "type": "SCALAR" }
        ]
    });

    let mut f = std::fs::File::create(path)?;
    f.write_all(serde_json::to_string_pretty(&json)?.as_bytes())?;
    std::fs::write(&bin_path, &bin)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_kind_round_trip_and_slots() {
        for kind in [
            MapKind::BaseColor,
            MapKind::Normal,
            MapKind::Roughness,
            MapKind::Metallic,
            MapKind::Ao,
            MapKind::Emissive,
            MapKind::Opacity,
        ] {
            assert_eq!(MapKind::from_str(kind.as_str()), Some(kind));
        }
        assert_eq!(MapKind::BaseColor.zestengine_slot(), (0, true));
        assert_eq!(MapKind::Normal.zestengine_slot(), (1, false));
        assert_eq!(MapKind::Roughness.zestengine_slot(), (13, false));
        assert_eq!(MapKind::from_str("Bogus"), None);
    }

    #[test]
    fn manifest_round_trip() {
        let m = ZestEngineManifest {
            object_id: "abc".into(),
            object_name: "House".into(),
            slot_count: 2,
            return_dir: PathBuf::from("/tmp/x"),
            textures: vec![],
        };
        let s = serde_json::to_string(&m).unwrap();
        let back: ZestEngineManifest = serde_json::from_str(&s).unwrap();
        assert_eq!(back.object_id, "abc");
        assert_eq!(back.slot_count, 2);
    }

    #[test]
    fn gltf_export_writes_mesh() {
        let dir = std::env::temp_dir().join(format!("zestengine-zest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mesh = zestengine_assets::primitive_mesh(zestengine_assets::PrimitiveShape::Cube);
        let path = dir.join("model.gltf");
        write_gltf(&mesh, &path).unwrap();
        assert!(path.exists());
        assert!(path.with_extension("bin").exists());
        check_gltf_invariants(&path, &mesh);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Every primitive shape's export must satisfy the glTF invariants
    /// loaders (ZestSurface's included) rely on. Also written to
    /// /tmp/engine_gltf_exports so ZestSurface's own loader probe can import
    /// them end-to-end.
    #[test]
    fn gltf_export_all_shapes_are_spec_valid() {
        let out = std::path::Path::new("/tmp/engine_gltf_exports");
        std::fs::create_dir_all(out).unwrap();
        for shape in [
            zestengine_assets::PrimitiveShape::Cube,
            zestengine_assets::PrimitiveShape::Sphere,
            zestengine_assets::PrimitiveShape::Capsule,
            zestengine_assets::PrimitiveShape::Plane,
            zestengine_assets::PrimitiveShape::Bucket,
        ] {
            let mesh = zestengine_assets::primitive_mesh(shape);
            let path = out.join(format!("{}.gltf", shape.label()));
            write_gltf(&mesh, &path).unwrap();
            check_gltf_invariants(&path, &mesh);
        }
    }

    /// The strided bufferViews must span the whole vertex array
    /// ((count - 1) * stride + element size) and fit the buffer; accessors
    /// must match the vertex/index counts. ZestSurface's loader rejected the
    /// old per-attribute byteLength ("glTF scene contains no triangle meshes").
    fn check_gltf_invariants(path: &Path, mesh: &zestengine_render::MeshData) {
        let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let views = json["bufferViews"].as_array().unwrap();
        let count = mesh.vertices.len();
        assert_eq!(views[0]["byteLength"], (count - 1) * 32 + 12);
        assert_eq!(views[1]["byteLength"], (count - 1) * 32 + 12);
        assert_eq!(views[2]["byteLength"], (count - 1) * 32 + 8);
        assert_eq!(views[3]["byteLength"], mesh.indices.len() * 4);
        let buffer_bytes = json["buffers"][0]["byteLength"].as_u64().unwrap();
        let end = json["bufferViews"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v["byteOffset"].as_u64().unwrap() + v["byteLength"].as_u64().unwrap())
            .max()
            .unwrap();
        assert!(end <= buffer_bytes, "views end at {end} past the buffer's {buffer_bytes}");
        let accessors = json["accessors"].as_array().unwrap();
        assert_eq!(accessors[0]["count"].as_u64().unwrap(), count as u64);
        assert_eq!(accessors[3]["count"].as_u64().unwrap(), mesh.indices.len() as u64);
        assert_eq!(accessors[0]["type"], "VEC3");
        assert_eq!(accessors[3]["componentType"], 5125);
    }

    #[test]
    fn sanitize_names() {
        assert_eq!(sanitize_name("House 01"), "House_01");
        assert_eq!(sanitize_name("bad/name:"), "bad_name_");
        assert_eq!(sanitize_name(""), "object");
    }
}
