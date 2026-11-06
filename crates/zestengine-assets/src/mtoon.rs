//! VRM MToon materials from the glTF itself (`VRMC_materials_mtoon` in VRM 1.0,
//! `VRM.materialProperties` with `VRM/MToon` in VRM 0.x), converted through the
//! same MToon property table as a Unity `.mat` (`unity_convert.rs`): the VRM
//! values are written under the serialised Unity property names and run through
//! [`convert_material`], so a VRM and a Unity MToon material import identically
//! and share one report format.
//!
//! The result is a [`Plan`] per material. Its textures carry the placeholder
//! GUID `gltf:texture:<index>` (a glTF texture index) for the importer to
//! resolve against the model's textures; the plan is otherwise complete.
//! Colours in the VRM JSON are linear; they are written back as sRGB so the
//! converter's sRGB to linear step returns them unchanged.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::Value;

use crate::unity_convert::{ConvertOptions, Plan, convert_material};
use crate::unity_mat::{UnityMaterial, UnityShader, UnityTexEnv};

/// One converted VRM MToon material.
#[derive(Clone, Debug)]
pub struct MToonMaterial {
    /// Index into the glTF `materials` array.
    pub material_index: usize,
    pub plan: Plan,
}

/// The GUID placeholder for glTF texture `index`.
pub fn gltf_texture_guid(index: usize) -> String {
    format!("gltf:texture:{index}")
}

/// The glTF texture index in a placeholder GUID from this module.
pub fn parse_gltf_texture_guid(guid: &str) -> Option<usize> {
    guid.strip_prefix("gltf:texture:")?.parse().ok()
}

/// Read every MToon material of a `.vrm` / `.glb` / `.gltf` (empty for anything else).
pub fn read_mtoon(path: &Path, opts: &ConvertOptions) -> Vec<MToonMaterial> {
    crate::vrm::gltf_json(path).map(|j| parse_mtoon(&j, opts)).unwrap_or_default()
}

fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

fn num(v: Option<&Value>) -> Option<f32> {
    v.and_then(Value::as_f64).map(|x| x as f32)
}

fn rgb(v: Option<&Value>) -> Option<[f32; 3]> {
    let a = v?.as_array()?;
    Some([a.first()?.as_f64()? as f32, a.get(1)?.as_f64()? as f32, a.get(2)?.as_f64()? as f32])
}

/// A linear colour as the sRGB colour Unity would serialise.
fn srgb4(c: [f32; 3], a: f32) -> [f32; 4] {
    [linear_to_srgb(c[0]), linear_to_srgb(c[1]), linear_to_srgb(c[2]), a]
}

fn tex(m: &mut UnityMaterial, prop: &str, index: Option<u64>) {
    if let Some(i) = index {
        m.textures.insert(prop.to_owned(), UnityTexEnv { guid: Some(gltf_texture_guid(i as usize)), file_id: 0, scale: [1.0, 1.0], offset: [0.0, 0.0] });
    }
}

/// glTF alpha mode and culling into MToon's `_BlendMode` (0 opaque, 1 cutout, 2 fade) and `_CullMode`.
fn base_state(m: &mut UnityMaterial, gltf_mat: &Value) {
    let mode = match gltf_mat.get("alphaMode").and_then(Value::as_str) {
        Some("MASK") => 1.0,
        Some("BLEND") => 2.0,
        _ => 0.0,
    };
    m.floats.insert("_BlendMode".into(), mode);
    m.floats.insert("_Cutoff".into(), num(gltf_mat.get("alphaCutoff")).unwrap_or(0.5));
    let double = gltf_mat.get("doubleSided").and_then(Value::as_bool).unwrap_or(false);
    m.floats.insert("_CullMode".into(), if double { 0.0 } else { 2.0 });
    let pbr = gltf_mat.get("pbrMetallicRoughness");
    let base = pbr.and_then(|p| p.get("baseColorFactor")).and_then(Value::as_array);
    let c = base.map(|a| [0, 1, 2, 3].map(|i| a.get(i).and_then(Value::as_f64).unwrap_or(1.0) as f32)).unwrap_or([1.0; 4]);
    m.colors.insert("_Color".into(), srgb4([c[0], c[1], c[2]], c[3]));
    tex(m, "_MainTex", pbr.and_then(|p| p.get("baseColorTexture")).and_then(|t| t.get("index")).and_then(Value::as_u64));
    tex(m, "_BumpMap", gltf_mat.get("normalTexture").and_then(|t| t.get("index")).and_then(Value::as_u64));
    if let Some(s) = num(gltf_mat.get("normalTexture").and_then(|t| t.get("scale"))) {
        m.floats.insert("_BumpScale".into(), s);
    }
    if let Some(e) = rgb(gltf_mat.get("emissiveFactor")).filter(|e| *e != [0.0; 3]) {
        m.colors.insert("_EmissionColor".into(), [e[0], e[1], e[2], 1.0]);
    }
    tex(m, "_EmissionMap", gltf_mat.get("emissiveTexture").and_then(|t| t.get("index")).and_then(Value::as_u64));
}

/// VRM 1.0 `VRMC_materials_mtoon` onto the Unity property names.
fn from_vrm1(name: &str, gltf_mat: &Value, ext: &Value) -> UnityMaterial {
    let mut m = UnityMaterial { name: name.to_owned(), shader_name: Some("VRM/MToon".into()), ..Default::default() };
    base_state(&mut m, gltf_mat);
    if let Some(c) = rgb(ext.get("shadeColorFactor")) {
        m.colors.insert("_ShadeColor".into(), srgb4(c, 1.0));
    }
    tex(&mut m, "_ShadeTexture", ext.get("shadeMultiplyTexture").and_then(|t| t.get("index")).and_then(Value::as_u64));
    if let Some(v) = num(ext.get("shadingShiftFactor")) {
        m.floats.insert("_ShadeShift".into(), v);
    }
    if let Some(v) = num(ext.get("shadingToonyFactor")) {
        m.floats.insert("_ShadeToony".into(), v);
    }
    if let Some(c) = rgb(ext.get("parametricRimColorFactor")) {
        m.colors.insert("_RimColor".into(), srgb4(c, 1.0));
    }
    for (key, prop) in [("parametricRimFresnelPowerFactor", "_RimFresnelPower"), ("parametricRimLiftFactor", "_RimLift"), ("rimLightingMixFactor", "_RimLightingMix")] {
        if let Some(v) = num(ext.get(key)) {
            m.floats.insert(prop.into(), v);
        }
    }
    tex(&mut m, "_RimTexture", ext.get("rimMultiplyTexture").and_then(|t| t.get("index")).and_then(Value::as_u64));
    tex(&mut m, "_SphereAdd", ext.get("matcapTexture").and_then(|t| t.get("index")).and_then(Value::as_u64));
    let mode = ext.get("outlineWidthMode").and_then(Value::as_str).unwrap_or("none");
    if mode != "none" {
        // Metres (or screen fraction) to ZestEngine outline units: 0.002 -> 1.
        let w = num(ext.get("outlineWidthFactor")).unwrap_or(0.0) * 500.0;
        m.floats.insert("_OutlineWidth".into(), w);
        m.shader_name = Some("VRM/MToon Outline".into());
        if let Some(c) = rgb(ext.get("outlineColorFactor")) {
            m.colors.insert("_OutlineColor".into(), srgb4(c, 1.0));
        }
    }
    m
}

/// VRM 0.x `materialProperties` entry onto the Unity property names (they are the same names).
fn from_vrm0(name: &str, gltf_mat: &Value, props: &Value) -> UnityMaterial {
    let mut m = UnityMaterial { name: name.to_owned(), shader_name: Some("VRM/MToon".into()), ..Default::default() };
    base_state(&mut m, gltf_mat);
    if let Some(floats) = props.get("floatProperties").and_then(Value::as_object) {
        for (k, v) in floats {
            if let Some(x) = v.as_f64() {
                m.floats.insert(k.clone(), x as f32);
            }
        }
    }
    // Textures first: the `_MainTex` vector below is that texture's tiling / offset.
    if let Some(textures) = props.get("textureProperties").and_then(Value::as_object) {
        for (k, v) in textures {
            tex(&mut m, k, v.as_u64());
        }
    }
    if let Some(vectors) = props.get("vectorProperties").and_then(Value::as_object) {
        for (k, v) in vectors {
            let Some(a) = v.as_array() else { continue };
            let g = |i: usize, d: f32| a.get(i).and_then(Value::as_f64).map_or(d, |x| x as f32);
            if k.starts_with("_MainTex") || k.ends_with("Tex") || k.ends_with("Texture") || k == "_BumpMap" || k == "_SphereAdd" {
                // [offset x, offset y, scale x, scale y]
                if let Some(t) = m.textures.get_mut(k.as_str()) {
                    t.offset = [g(0, 0.0), g(1, 0.0)];
                    t.scale = [g(2, 1.0), g(3, 1.0)];
                }
                continue;
            }
            let c = [g(0, 0.0), g(1, 0.0), g(2, 0.0)];
            // HDR emission stays linear (the converter keeps it); other colours are linear in the file.
            let col = if k == "_EmissionColor" { [c[0], c[1], c[2], g(3, 1.0)] } else { srgb4(c, g(3, 1.0)) };
            m.colors.insert(k.clone(), col);
        }
    }
    // Outline: VRM 0.x `_OutlineWidthMode` 0 none, 1 world, 2 screen; width in centimetres (world) or percent of height.
    let mode = m.floats.get("_OutlineWidthMode").copied().unwrap_or(0.0).round() as i32;
    if mode == 0 {
        m.floats.remove("_OutlineWidth");
    } else {
        let w = m.floats.get("_OutlineWidth").copied().unwrap_or(0.0);
        m.floats.insert("_OutlineWidth".into(), w * 5.0);
        m.shader_name = Some("VRM/MToon Outline".into());
    }
    if let Some(q) = props.get("renderQueue").and_then(Value::as_i64) {
        m.render_queue = Some(q as i32);
    }
    m
}

/// Convert every MToon material of a glTF JSON document.
pub fn parse_mtoon(json: &Value, opts: &ConvertOptions) -> Vec<MToonMaterial> {
    let materials = json.get("materials").and_then(Value::as_array).cloned().unwrap_or_default();
    let vrm0: BTreeMap<usize, &Value> = json
        .get("extensions")
        .and_then(|e| e.get("VRM"))
        .and_then(|v| v.get("materialProperties"))
        .and_then(Value::as_array)
        .map(|a| a.iter().enumerate().filter(|(_, p)| p.get("shader").and_then(Value::as_str) == Some("VRM/MToon")).collect())
        .unwrap_or_default();
    let mut out = Vec::new();
    for (i, gm) in materials.iter().enumerate() {
        let name = gm.get("name").and_then(Value::as_str).unwrap_or("material");
        let um = if let Some(ext) = gm.get("extensions").and_then(|e| e.get("VRMC_materials_mtoon")) {
            from_vrm1(name, gm, ext)
        } else if let Some(props) = vrm0.get(&i) {
            from_vrm0(name, gm, props)
        } else {
            continue;
        };
        let plan = convert_material(&um, UnityShader::MToon, opts);
        out.push(MToonMaterial { material_index: i, plan });
    }
    out
}

/// What [`import_mtoon_materials`] wrote.
#[derive(Clone, Debug, Default)]
pub struct MToonImport {
    /// One entry per MToon material: its `.material`, report and mapping report.
    pub converted: Vec<crate::unity_convert::Converted>,
    /// Remap-table entries added (materials already mapped are left alone).
    pub remapped: usize,
    /// Textures that could not be read out of the model.
    pub texture_failures: Vec<String>,
}

/// Import the MToon materials of a VRM / glTF as ZestEngine Toon materials: the plans of
/// [`read_mtoon`], their glTF textures written out of the model and copied into
/// `opts.texture_dir` (through the Unity converter's own texture step, so the slots get the
/// same import settings), one `.material` plus `.convert.txt` per material in `mat_dir`, and
/// the model's material remap table filled with them (keyed by material name; a name that is
/// already remapped is left alone). Nothing happens for a model without MToon materials.
pub fn import_mtoon_materials(
    model_path: &Path,
    project_root: &Path,
    mat_dir: &Path,
    opts: &ConvertOptions,
    model: &mut crate::ModelImport,
) -> anyhow::Result<MToonImport> {
    use crate::unity_convert::{UnityAssets, add_to_model_remap, finish_plan};
    let mut out = MToonImport::default();
    let Some(json) = crate::vrm::gltf_json(model_path) else { return Ok(out) };
    let mut plans = parse_mtoon(&json, opts);
    plans.retain(|p| !model.material_remap.contains_key(&p.plan.name));
    if plans.is_empty() {
        return Ok(out);
    }
    let (doc, _buffers, images) = gltf::import(model_path).map_err(|e| anyhow::anyhow!("reading {}: {e}", model_path.display()))?;
    // glTF texture index -> image index, for every texture a plan references.
    let tex_image: Vec<usize> = doc.textures().map(|t| t.source().index()).collect();
    let scratch = std::env::temp_dir().join(format!("zest_mtoon_{}_{}", std::process::id(), plans.len()));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch)?;
    let stem = crate::material_remap::file_stem_of(&model_path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(), "model");
    let mut assets = UnityAssets { root: scratch.clone(), ..Default::default() };
    let mut written: BTreeMap<usize, std::path::PathBuf> = BTreeMap::new();
    let wanted: std::collections::BTreeSet<usize> = plans
        .iter()
        .flat_map(|p| p.plan.textures.iter().map(|t| t.guid.clone()).chain(p.plan.repacks.iter().map(|r| r.guid.clone())))
        .filter_map(|g| parse_gltf_texture_guid(&g))
        .collect();
    for ti in wanted {
        let Some(&ii) = tex_image.get(ti) else {
            out.texture_failures.push(format!("texture {ti}: no such glTF texture"));
            continue;
        };
        let path = match written.get(&ii) {
            Some(p) => p.clone(),
            None => {
                let data = match images.get(ii).map(|d| crate::gltf_loader::rgba8_from_gltf(d, true)) {
                    Some(Ok(d)) => d,
                    Some(Err(e)) => {
                        out.texture_failures.push(format!("texture {ti}: {e:#}"));
                        continue;
                    }
                    None => {
                        out.texture_failures.push(format!("texture {ti}: image {ii} missing"));
                        continue;
                    }
                };
                let img = image::RgbaImage::from_raw(data.width, data.height, data.pixels).ok_or_else(|| anyhow::anyhow!("image {ii} has a bad pixel buffer"))?;
                let name = doc.images().nth(ii).and_then(|i| i.name().map(str::to_owned)).filter(|n| !n.is_empty()).unwrap_or_else(|| format!("image{ii}"));
                let p = scratch.join(format!("{stem}_{}.png", crate::material_remap::file_stem_of(&name, "texture")));
                let p = if p.exists() { scratch.join(format!("{stem}_{ii}.png")) } else { p };
                img.save(&p)?;
                written.insert(ii, p.clone());
                p
            }
        };
        assets.guid_path.insert(gltf_texture_guid(ti).to_ascii_lowercase(), path);
    }
    for m in plans {
        let plan = m.plan;
        match finish_plan(plan, &assets, project_root, mat_dir, opts) {
            Ok(c) => out.converted.push(c),
            Err(e) => out.texture_failures.push(format!("{}: {e:#}", m.material_index)),
        }
    }
    out.remapped = add_to_model_remap(model, project_root, &out.converted);
    let _ = std::fs::remove_dir_all(&scratch);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unity_convert::Level;
    use crate::unity_convert::TexTarget;
    use zestengine_render::AlphaMode;

    fn c(p: &Plan, k: &str) -> Vec<f32> {
        p.file.custom.get(k).cloned().unwrap_or_default()
    }

    const VRM1: &str = r#"{
      "materials": [
        {"name":"Skin","alphaMode":"MASK","alphaCutoff":0.4,"doubleSided":true,
         "pbrMetallicRoughness":{"baseColorFactor":[0.5,0.25,0.1,1.0],"baseColorTexture":{"index":3}},
         "normalTexture":{"index":5,"scale":0.8},
         "extensions":{"VRMC_materials_mtoon":{
           "shadeColorFactor":[0.4,0.2,0.1],"shadeMultiplyTexture":{"index":4},
           "shadingShiftFactor":-0.2,"shadingToonyFactor":0.8,
           "parametricRimColorFactor":[1.0,0.5,0.0],"parametricRimFresnelPowerFactor":3.0,"rimLightingMixFactor":0.7,
           "matcapTexture":{"index":6},
           "outlineWidthMode":"worldCoordinates","outlineWidthFactor":0.004,"outlineColorFactor":[0.0,0.0,0.0]}}},
        {"name":"Plain","pbrMetallicRoughness":{}}
      ]}"#;

    #[test]
    fn vrm1_mtoon_maps_through_the_unity_table() {
        let v: Value = serde_json::from_str(VRM1).unwrap();
        let out = parse_mtoon(&v, &ConvertOptions::default());
        assert_eq!(out.len(), 1, "only the MToon material converts");
        let p = &out[0].plan;
        assert_eq!(out[0].material_index, 0);
        assert_eq!(p.report.target, "ZestEngine Toon");
        assert_eq!(p.file.features.alpha_mode, AlphaMode::Cutout);
        assert!((p.file.params.alpha_cutoff - 0.4).abs() < 1e-6);
        assert!(p.file.features.double_sided);
        // Linear colours round trip through the sRGB step.
        let main = c(p, "main_color");
        assert!((main[0] - 0.5).abs() < 1e-3 && (main[1] - 0.25).abs() < 1e-3 && (main[2] - 0.1).abs() < 1e-3, "{main:?}");
        let shade = c(p, "shadow_color");
        assert!((shade[0] - 0.4).abs() < 1e-3, "{shade:?}");
        // shift -0.2 -> border 0.6, toony 0.8 -> blur 0.1 (the Unity MToon rule).
        assert!((c(p, "layer1_border")[0] - 0.6).abs() < 1e-4);
        assert!((c(p, "layer1_blur")[0] - 0.1).abs() < 1e-4);
        assert_eq!(c(p, "RIM"), vec![1.0]);
        assert!((c(p, "rim_strength")[0] - 0.7).abs() < 1e-4);
        assert_eq!(c(p, "MATCAP"), vec![1.0], "the matcap texture maps to Matcap 1");
        assert_eq!(c(p, "OUTLINE"), vec![1.0]);
        assert!((c(p, "outline_width")[0] - 2.0).abs() < 1e-3, "0.004 m -> 2 units: {:?}", c(p, "outline_width"));
        // Textures carry glTF indices.
        let idx = |t: &TexTarget| p.textures.iter().find(|u| &u.target == t).and_then(|u| parse_gltf_texture_guid(&u.guid));
        assert_eq!(idx(&TexTarget::Albedo), Some(3));
        assert_eq!(idx(&TexTarget::Normal), Some(5));
        assert_eq!(idx(&TexTarget::Custom("matcap_tex".into())), Some(6));
        // The per-pixel shade texture is a known gap and says so.
        assert!(p.report.lines.iter().any(|l| l.level == Level::Unsupported && l.property == "_ShadeTexture"), "{}", p.report.to_text());
    }

    #[test]
    fn vrm0_material_properties_map_with_the_same_table() {
        let json = r#"{
          "materials":[{"name":"Hair","alphaMode":"OPAQUE","pbrMetallicRoughness":{"baseColorFactor":[1,1,1,1]}},{"name":"Other"}],
          "extensions":{"VRM":{"materialProperties":[
            {"name":"Hair","shader":"VRM/MToon","renderQueue":2000,
             "floatProperties":{"_ShadeShift":0.0,"_ShadeToony":0.9,"_OutlineWidthMode":1,"_OutlineWidth":0.2,"_RimFresnelPower":1.0},
             "vectorProperties":{"_ShadeColor":[0.5,0.5,0.6,1.0],"_RimColor":[0.0,0.0,0.0,1.0],"_OutlineColor":[0.1,0.1,0.1,1.0],"_MainTex":[0.0,0.0,2.0,2.0]},
             "textureProperties":{"_MainTex":2,"_ShadeTexture":2}},
            {"name":"Other","shader":"Standard"}]}}}"#;
        let v: Value = serde_json::from_str(json).unwrap();
        let out = parse_mtoon(&v, &ConvertOptions::default());
        assert_eq!(out.len(), 1);
        let p = &out[0].plan;
        assert!((c(p, "shadow_color")[2] - 0.6).abs() < 1e-3);
        assert!((c(p, "layer1_blur")[0] - 0.05).abs() < 1e-4);
        assert_eq!(c(p, "OUTLINE"), vec![1.0]);
        assert!((c(p, "outline_width")[0] - 1.0).abs() < 1e-3, "0.2 cm x 5 = 1 unit: {:?}", c(p, "outline_width"));
        assert_eq!(p.file.custom["main_st"][..2], [2.0, 2.0], "tiling carried");
        assert_eq!(p.file.render_queue, Some(2000));
    }

    #[test]
    fn non_vrm_gltf_and_garbage_are_empty() {
        let v: Value = serde_json::from_str(r#"{"materials":[{"name":"M"}]}"#).unwrap();
        assert!(parse_mtoon(&v, &ConvertOptions::default()).is_empty());
        assert!(parse_mtoon(&Value::Null, &ConvertOptions::default()).is_empty());
        assert!(read_mtoon(Path::new("/nonexistent/x.vrm"), &ConvertOptions::default()).is_empty());
    }

    #[test]
    fn real_vrm_in_the_project_converts_when_present() {
        // The repository's own avatars are glTF; this only asserts nothing panics on real files.
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/avatars");
        let Ok(rd) = std::fs::read_dir(&dir) else { return };
        for e in rd.flatten().take(8) {
            let p = e.path();
            if p.extension().is_some_and(|x| x == "vrm" || x == "glb") {
                let n = read_mtoon(&p, &ConvertOptions::default()).len();
                eprintln!("{}: {n} MToon materials", p.display());
            }
        }
    }

    /// A VRM 1.0 document with one MToon material, a base-colour texture and an outline,
    /// as a `.gltf` next to a 2x2 `skin.png` written into `dir`.
    fn synthetic_vrm(dir: &Path) -> String {
        image::RgbaImage::from_pixel(2, 2, image::Rgba([200, 100, 50, 255])).save(dir.join("skin.png")).unwrap();
        r#"{"asset":{"version":"2.0"},"extensionsUsed":["VRMC_materials_mtoon"],
              "images":[{"uri":"skin.png","name":"skin"}],"textures":[{"source":0}],
              "materials":[{"name":"Skin","pbrMetallicRoughness":{"baseColorFactor":[1,1,1,1],"baseColorTexture":{"index":0}},
                "extensions":{"VRMC_materials_mtoon":{"shadeColorFactor":[0.5,0.4,0.4],"parametricRimColorFactor":[1,1,1],
                  "outlineWidthMode":"worldCoordinates","outlineWidthFactor":0.002,"outlineColorFactor":[0,0,0]}}},
                {"name":"Plain"}]}"#
            .to_owned()
    }

    #[test]
    fn mtoon_import_writes_toon_materials_with_resolved_textures() {
        let dir = std::env::temp_dir().join(format!("zest_mtoon_import_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let model_path = dir.join("hero.gltf");
        std::fs::write(&model_path, synthetic_vrm(&dir)).unwrap();
        let opts = ConvertOptions { texture_dir: "Textures".into(), ..Default::default() };
        let mut model = crate::ModelImport::default();
        let r = import_mtoon_materials(&model_path, &dir, &dir.join("Materials"), &opts, &mut model).unwrap();
        assert_eq!(r.converted.len(), 1, "only the MToon material imports: {:?}", r.texture_failures);
        assert_eq!(r.remapped, 1);
        let remap = model.material_remap.get("Skin").expect("remapped by name");
        let mf = crate::load_material_file(dir.join(remap)).expect("loadable .material");
        assert_eq!(mf.shader, "shaders/zestengine_toon.frag");
        let albedo = mf.textures.albedo.as_deref().expect("the glTF texture resolved to a project file");
        assert!(dir.join(albedo).is_file(), "{albedo:?}");
        assert_eq!(image::open(dir.join(albedo)).unwrap().into_rgba8().get_pixel(0, 0).0, [200, 100, 50, 255]);
        assert_eq!(mf.custom.get("OUTLINE"), Some(&vec![1.0]));
        assert_eq!(mf.custom.get("RIM"), Some(&vec![1.0]));
        // A second run leaves the mapped material alone.
        let again = import_mtoon_materials(&model_path, &dir, &dir.join("Materials"), &opts, &mut model).unwrap();
        assert!(again.converted.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A real VRM on this machine (VRM 0.x or 1.0): every MToon material imports with its
    /// textures resolved to project files. Skipped when the file is absent.
    #[test]
    fn real_vrm_on_this_machine_imports_mtoon_materials() {
        let src = Path::new("/home/lemonxah/shared/Nadia/Vivixx Vtube.vrm");
        if !src.is_file() {
            eprintln!("[real data] {} not present: skipped", src.display());
            return;
        }
        let json = crate::vrm::gltf_json(src).expect("a VRM holds glTF JSON");
        let version = if json.get("extensions").is_some_and(|e| e.get("VRMC_vrm").is_some()) { "1.0" } else { "0.x" };
        let total = json.get("materials").and_then(Value::as_array).map_or(0, Vec::len);
        let dir = std::env::temp_dir().join(format!("zest_real_vrm_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let opts = ConvertOptions { texture_dir: "Textures".into(), ..Default::default() };
        let mut model = crate::ModelImport::default();
        let r = import_mtoon_materials(src, &dir, &dir.join("Materials"), &opts, &mut model).unwrap();
        eprintln!("[real data] VRM {version}: {} of {total} materials are MToon; {} textures failed; textures written: {}", r.converted.len(), r.texture_failures.len(), std::fs::read_dir(dir.join("Textures")).map_or(0, |d| d.count()));
        assert!(!r.converted.is_empty(), "the avatar holds MToon materials");
        let mut with_albedo = 0;
        let (mut outlines, mut rims) = (0, 0);
        for c in &r.converted {
            let mf = crate::load_material_file(&c.material_path).unwrap();
            assert_eq!(mf.shader, "shaders/zestengine_toon.frag", "{}", c.name);
            if let Some(a) = &mf.textures.albedo {
                assert!(dir.join(a).is_file(), "{}: {a:?}", c.name);
                with_albedo += 1;
            }
            outlines += usize::from(mf.custom.get("OUTLINE").is_some_and(|v| v[0] != 0.0));
            rims += usize::from(mf.custom.get("RIM").is_some_and(|v| v[0] != 0.0));
        }
        eprintln!("[real data] albedo textures resolved: {with_albedo}; with outline: {outlines}; with rim: {rims}");
        assert!(with_albedo > 0, "some material has a base-colour texture");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
