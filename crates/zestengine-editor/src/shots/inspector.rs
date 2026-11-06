//! Headless screenshots of the object inspector: the pinned header, the Mesh Filter / Mesh
//! Renderer sections with their slot list, the per-material sections and the Model section.
//! `ZESTENGINE_SHOT_DIR=target/shots cargo test -p zestengine-engine --lib shots_inspector`

use zestengine_core::{Component, ComponentRegistry, MaterialSlot, MeshRenderer, ShadowCasting};
use crate::shot::Shot;
use crate::{
    EditorComponents, InspectorContent, InspectorPanel, MaterialSectionModel, MaterialSource, MeshFilterInfo, MixedInfo, ModelChild, ModelRootInfo, ObjectInfoModel, RendererInfo, SlotInfo,
    TransformModel,
};
use zestengine_render::{MaterialFeatures, MaterialParams};

fn material(name: &str) -> crate::MaterialModel {
    zestengine_engine::scene::model_from_material(name, &MaterialParams::default(), &MaterialFeatures::default(), false)
}

fn info(key: &str, name: &str, slots: &[(&str, &str)]) -> ObjectInfoModel {
    ObjectInfoModel {
        mixed: MixedInfo::default(),
        name: name.into(),
        enabled: true,
        static_geometry: true,
        key: key.into(),
        kind: "Mesh",
        transform: TransformModel { translation: [0.0; 3], rotation_deg: [0.0; 3], scale: [1.0; 3], parent: None, world: None },
        mesh: Some((1200, 400)),
        model_root: None,
        filter: Some(MeshFilterInfo { label: "Mesh 0".into(), model_path: Some("models/hero.glb".into()), vertices: 1200, triangles: 400 }),
        renderer: Some(RendererInfo {
            slots: slots.iter().map(|(sub, mat)| SlotInfo { submesh: sub.to_string(), model_material: mat.to_string(), material: mat.to_string() }).collect(),
            ..Default::default()
        }),
        blendshapes: Vec::new().into(),
    }
}

/// Draw the header and the body of one object in a 380 pt inspector; returns (image, header height in points).
/// A material section without its model (lent per frame from `models`).
struct Spec {
    slots: Vec<usize>,
    title: &'static str,
    shared_by: u32,
    source: MaterialSource,
    embedded: bool,
    model: Option<usize>,
}

fn spec(slots: &[usize], title: &'static str, shared_by: u32, source: MaterialSource, embedded: bool, model: Option<usize>) -> Spec {
    Spec { slots: slots.to_vec(), title, shared_by, source, embedded, model }
}

#[allow(clippy::too_many_arguments)]
fn draw(name: &str, kind: crate::theme::ObjectKind, type_line: &str, info: &mut ObjectInfoModel, components: Vec<Box<dyn Component>>, specs: Vec<Spec>, models: &mut [crate::MaterialModel], open: &[usize]) -> f32 {
    draw_with(name, kind, type_line, info, components, specs, models, open, false)
}

#[allow(clippy::too_many_arguments)]
fn draw_with(name: &str, kind: crate::theme::ObjectKind, type_line: &str, info: &mut ObjectInfoModel, mut components: Vec<Box<dyn Component>>, specs: Vec<Spec>, models: &mut [crate::MaterialModel], open: &[usize], open_blend: bool) -> f32 {
    let reg = ComponentRegistry::with_builtins();
    let ed = EditorComponents::with_builtins();
    let mut panel = InspectorPanel::new();
    let mut shot = Shot::new(380.0, 1600.0, 2.0);
    for &k in open {
        crate::set_section_open(&shot.ctx, &info.key, k, true);
    }
    let layers: Vec<String> = ["Default", "TransparentFX", "Ignore Raycast", "Water", "UI"].iter().map(|s| s.to_string()).collect();
    let mut header_h = 0.0;
    let mut n = info.name.clone();
    let mut role = zestengine_core::NetRole::Both;
    let (mut enabled, mut stat, mut layer) = (info.enabled, info.static_geometry, 0u8);
    let img = shot.run_panel(|ui| {
        panel.lock_bar(ui);
        if open_blend {
            let id = ui.make_persistent_id("zestengine-blendshapes-section");
            let mut st = egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false);
            st.set_open(true);
            st.store(ui.ctx());
        }
        let top = ui.cursor().top();
        panel.object_header(ui, kind, type_line, &mut n, &mut enabled, &mut stat, &mut layer, &layers, &mut role, &[]);
        header_h = ui.cursor().top() - top;
        let mut lent: Vec<Option<&mut crate::MaterialModel>> = models.iter_mut().map(Some).collect();
        let secs: Vec<MaterialSectionModel<'_>> = specs
            .iter()
            .enumerate()
            .map(|(k, sp)| MaterialSectionModel {
                slots: sp.slots.clone(),
                material_index: k,
                title: sp.title.into(),
                shader: "Standard".into(),
                shared_by: sp.shared_by,
                source: sp.source.clone(),
                embedded: sp.embedded,
                model: sp.model.and_then(|m| lent[m].take()),
                shader_info: None,
                slot_warnings: Vec::new(),
            })
            .collect();
        panel.ui(ui, InspectorContent::Object { info, sections: secs, components: &mut components, registry: &reg, editor_components: &ed, objects: &[], layer_names: &layers, asset_editor: None }, &["standard"]);
    });
    assert!(img.non_background() > 500, "{name} drew nothing");
    shot.save_named(&img, name);
    header_h
}

#[test]
fn cube_with_one_slot() {
    let mut i = info("cube", "Cube", &[("Cube", "Default")]);
    i.filter = Some(MeshFilterInfo { label: "Cube".into(), model_path: None, vertices: 24, triangles: 12 });
    let mut models = [material("Default")];
    let specs = vec![spec(&[0], "Default", 3, MaterialSource::File("materials/default.material".into()), false, Some(0))];
    let comps: Vec<Box<dyn Component>> = vec![Box::new(MeshRenderer { materials: vec![MaterialSlot::File("materials/default.material".into())], cast_shadows: ShadowCasting::On, ..Default::default() })];
    let h = draw("inspector_cube", crate::theme::ObjectKind::Mesh, "Mesh", &mut i, comps, specs, &mut models, &[0]);
    assert!(h < 80.0, "header block is about two rows: {h} pt");
}

#[test]
fn model_part_with_three_materials() {
    let mut i = info("body", "Body", &[("Body", "Skin"), ("Eyes", "Eyes_mat"), ("Hair", "Hair")]);
    let mut models = [material("Skin"), material("Hair")];
    let specs = vec![
        spec(&[0], "Skin (slot 0)", 4, MaterialSource::File("materials/skin.material".into()), false, Some(0)),
        spec(&[1], "Eyes_mat (slot 1)", 12, MaterialSource::Model { model: "hero.glb".into() }, true, None),
        spec(&[2], "Hair (slot 2)", 1, MaterialSource::File("materials/hair.material".into()), false, Some(1)),
    ];
    let comps: Vec<Box<dyn Component>> = vec![Box::new(MeshRenderer {
        materials: vec![MaterialSlot::File("materials/skin.material".into()), MaterialSlot::Model, MaterialSlot::File("materials/hair.material".into())],
        ..Default::default()
    })];
    // Slot 0 open (editable), slot 1 open (embedded, read-only), slot 2 closed.
    draw("inspector_three_materials", crate::theme::ObjectKind::Mesh, "Mesh \u{b7} child of hero", &mut i, comps, specs, &mut models, &[0, 1]);
}

#[test]
fn model_part_with_an_embedded_material() {
    let mut i = info("part", "Chair", &[("Chair", "Wood")]);
    let specs = vec![spec(&[0], "Wood", 8, MaterialSource::Model { model: "chair.fbx".into() }, true, None)];
    draw("inspector_embedded_material", crate::theme::ObjectKind::Mesh, "Mesh", &mut i, vec![Box::new(MeshRenderer::inheriting(1))], specs, &mut [], &[0]);
}

#[test]
fn model_root_shows_the_model_section() {
    let mut i = info("root", "lemonjuice", &[]);
    i.kind = "Empty";
    i.mesh = None;
    i.filter = None;
    i.renderer = None;
    i.model_root = Some(ModelRootInfo {
        path: "models/lemonjuice.fbx".into(),
        meshes: 3,
        materials: 4,
        clips: 2,
        rig: "Humanoid".into(),
        children: vec![
            ModelChild { object: 1, name: "Body".into(), materials: vec!["Skin".into(), "Eyes".into()] },
            ModelChild { object: 2, name: "Hair".into(), materials: vec!["Hair".into()] },
            ModelChild { object: 3, name: "Armature".into(), materials: vec![] },
        ],
    });
    draw("inspector_model_root", crate::theme::ObjectKind::Model, "Model", &mut i, Vec::new(), Vec::new(), &mut [], &[]);
}

#[test]
fn light_and_mixed_selection_headers() {
    let mut i = info("light", "Sun", &[]);
    i.kind = "Light";
    i.mesh = None;
    i.filter = None;
    i.renderer = None;
    draw("inspector_light", crate::theme::ObjectKind::DirectionalLight, "Directional Light", &mut i, Vec::new(), Vec::new(), &mut [], &[]);
    let mut m = info("multi", "", &[]);
    m.filter = None;
    m.renderer = None;
    m.mixed = MixedInfo { name: true, ..Default::default() };
    draw("inspector_multi", crate::theme::ObjectKind::Mesh, "3 objects selected", &mut m, Vec::new(), Vec::new(), &mut [], &[]);
}

/// Stitch the inspector PNGs written by the tests above into one sheet (run last:
/// `ZESTENGINE_SHOT_DIR=<raw> ZESTENGINE_SHEET_DIR=docs/design/screens cargo test -p zestengine-engine --lib shots_inspector::contact_sheet -- --ignored`).
#[test]
#[ignore]
fn contact_sheet() {
    use crate::shot::Shot8;
    let (Some(raw), Some(out)) = (std::env::var_os("ZESTENGINE_SHOT_DIR"), std::env::var_os("ZESTENGINE_SHEET_DIR")) else { return };
    let (raw, out) = (std::path::PathBuf::from(raw), std::path::PathBuf::from(out));
    let names = ["inspector_cube", "inspector_three_materials", "inspector_embedded_material", "inspector_model_root", "inspector_light", "inspector_multi", "inspector_skinned_510_shapes"];
    let imgs: Vec<Shot8> = names.iter().filter_map(|n| Shot8::load(&raw.join(format!("{n}.png")))).map(|i| i.downscale(2)).collect();
    if !imgs.is_empty() {
        Shot8::sheet(&imgs, 3).save(&out.join("sheet-inspector-mesh.png"));
    }
}

#[test]
fn skinned_part_with_four_materials_and_510_blendshapes() {
    let names: Vec<String> = (0..510)
        .map(|i| match i % 4 {
            0 => format!("Face_{i}_Smile"),
            1 => format!("Eye_{i}_Blink"),
            2 => format!("Mouth_{i}_Open"),
            _ => format!("Brow_{i}_Up"),
        })
        .collect();
    let mut i = info("skin", "Body", &[("Body", "Body"), ("Face", "Face"), ("Eyes", "Eyes"), ("Transparent", "Transparent")]);
    i.blendshapes = names.clone().into();
    if let Some(r) = i.renderer.as_mut() {
        r.skinned = true;
        r.joints = (0..80).map(|k| format!("Bone{k}")).collect();
    }
    let specs = vec![
        spec(&[0], "Body (slot 0)", 1, MaterialSource::Model { model: "avatar.vrm".into() }, true, None),
        spec(&[1], "Face (slot 1)", 1, MaterialSource::Model { model: "avatar.vrm".into() }, true, None),
        spec(&[2], "Eyes (slot 2)", 1, MaterialSource::Model { model: "avatar.vrm".into() }, true, None),
        spec(&[3], "Transparent (slot 3)", 1, MaterialSource::Model { model: "avatar.vrm".into() }, true, None),
    ];
    // The Blend Shapes component exists (storage) but must not get its own section.
    let comps: Vec<Box<dyn Component>> = vec![
        Box::new(MeshRenderer::inheriting(4)),
        Box::new(zestengine_core::BlendShapes { weights: vec![(names[0].clone(), 0.6)], ..Default::default() }),
        Box::new(zestengine_core::FaceTracking::default()),
    ];
    draw_with("inspector_skinned_510_shapes", crate::theme::ObjectKind::Mesh, "Mesh", &mut i, comps, specs, &mut [], &[], true);
}

#[test]
fn child_of_a_rotated_scaled_parent_shows_local_transform() {
    let mut i = info("child", "Wheel", &[("Wheel", "Default")]);
    i.filter = None;
    i.renderer = None;
    // Parent: rotated 30 deg about Y, scale (2, 1, 0.5), at (5, 1, -2). Child local: (1, 2, 3).
    i.transform = TransformModel {
        translation: [1.0, 2.0, 3.0],
        rotation_deg: [-0.0, 0.0, -0.03],
        scale: [1.5, 2.0, 0.75],
        parent: Some("Car Body".into()),
        world: Some(([7.0, 3.0, -1.1], [0.0, 30.0, -0.0], [3.0, 2.0, 0.38])),
    };
    // The World line is collapsed by default; open it for the second image.
    let h = draw("inspector_child_local_transform", crate::theme::ObjectKind::Mesh, "Mesh", &mut i, Vec::new(), Vec::new(), &mut [], &[]);
    assert!(h < 80.0);
    assert_eq!(crate::space_tooltip(Some("Car Body")), "Relative to the parent (Car Body)");
    assert_eq!(crate::space_tooltip(None), "World (no parent)");
    assert_eq!(crate::widgets::fixed_no_neg_zero(-0.0, 1), "0.0");
    assert_eq!(crate::widgets::fixed_no_neg_zero(-0.03, 1), "0.0");
    assert_eq!(crate::widgets::fixed_no_neg_zero(-0.2, 1), "-0.2");
}
