//! `Text3D`: lays 3D text out through the text system and queues it for the
//! scene pass (depth tested, multiview aware). No render target, no mesh.

use glam::{Mat4, Vec3};
use zestengine_core::{Localization, Text3D, Text3DBillboard, UiAlign, UiTextOverflow};
use zestengine_text::gpu::srgb_to_linear;
use zestengine_text::{
    Billboard, Effects, HAlign, LayoutParams, Overflow, TextBlock, TextDraw, TextHandle, TextRequest, TextSpace, VAlign,
};

use crate::scene::LoadedScene;
use std::collections::HashMap;
use zestengine_core::{ObjectId, ecs::{Entity, SystemWorld}};

/// Layout units per metre (text is laid out in millimetres).
pub const PX_PER_M: f32 = 1000.0;

fn params(t: &Text3D) -> LayoutParams {
    let px = |m: f32| m * PX_PER_M;
    LayoutParams {
        size: px(t.size),
        box_size: [px(t.width), px(t.height)],
        wrap: t.wrap,
        halign: match t.align_x {
            UiAlign::Start => HAlign::Start,
            UiAlign::Center => HAlign::Center,
            UiAlign::End => HAlign::End,
        },
        valign: match t.align_y {
            UiAlign::Start => VAlign::Top,
            UiAlign::Center => VAlign::Middle,
            UiAlign::End => VAlign::Bottom,
        },
        overflow: match t.overflow {
            UiTextOverflow::Overflow => Overflow::Overflow,
            UiTextOverflow::Ellipsis => Overflow::Ellipsis,
            UiTextOverflow::Truncate => Overflow::Truncate,
            UiTextOverflow::Masking => Overflow::Masking,
            UiTextOverflow::Page => Overflow::Page,
        },
        auto_size: t.auto_size.then_some((px(t.auto_size_min), px(t.auto_size_max))),
        char_spacing: t.char_spacing,
        line_spacing: t.line_spacing,
        bold: t.bold,
        italic: t.italic,
        ..Default::default()
    }
}

/// The draw for a laid-out block: centred on the object's origin, facing +Z
/// (layout y runs down, world y up), or turned toward the camera.
pub fn make_draw(block: &TextBlock, t: &Text3D, world: Mat4) -> TextDraw {
    let s = 1.0 / PX_PER_M;
    let (w, h) = (if t.width > 0.0 { t.width * PX_PER_M } else { block.width }, if t.height > 0.0 { t.height * PX_PER_M } else { block.height });
    let (transform, billboard) = match t.billboard {
        // The shader flips y itself when billboarding.
        Text3DBillboard::None => (world * Mat4::from_scale(Vec3::new(s, -s, s)), Billboard::None),
        Text3DBillboard::Y => (world * Mat4::from_scale(Vec3::new(s, s, s)), Billboard::Y),
        Text3DBillboard::Full => (world * Mat4::from_scale(Vec3::new(s, s, s)), Billboard::Full),
    };
    TextDraw {
        instances: block.instances.clone(),
        transform: transform.to_cols_array_2d(),
        space: TextSpace::World,
        billboard,
        tint: srgb_to_linear(t.color),
        local_offset: [-w * 0.5, -h * 0.5],
        clip: block.clip,
        supersample: true,
        order: 0,
    }
}

/// Queue every active `Text3D` for this frame.
pub fn queue(handle: &TextHandle, scene: &LoadedScene, active: &[bool], world: &[Mat4], strings: Option<&Localization>) {
    scene.objects.query(|view, order| queue_in(handle, view, order, strings, |_, index| {
        active.get(index).copied().unwrap_or(false).then(|| world.get(index).copied()).flatten()
    }));
}

/// Queue native world text without copying scene transforms or component values.
pub fn queue_scene(handle: &TextHandle, scene: &LoadedScene, strings: Option<&Localization>) {
    scene.objects.query(|world, order| queue_in(handle, world, order, strings, |entity, _| {
        scene.visible_entity_in(world, entity).then(|| world.world_transform(entity)).flatten()
    }));
}

fn queue_in(handle: &TextHandle, world: &SystemWorld<'_>, order: &HashMap<ObjectId, usize>, strings: Option<&Localization>, pose: impl Fn(Entity, usize) -> Option<Mat4>) {
    let Ok(mut sys) = handle.system.lock() else { return };
    if let Some(s) = strings {
        sys.set_language(s.current_language());
        warm_language(&mut sys, s);
    }
    let mut query = world.query::<(&ObjectId, &Text3D)>();
    let mut texts: Vec<_> = query.iter().filter_map(|(entity, (id, text))| Some((*order.get(id)?, entity, text))).collect();
    texts.sort_unstable_by_key(|(index, _, _)| *index);
    for (index, entity, t) in texts {
        let Some(matrix) = pose(entity, index) else { continue };
        let shown: &str = match strings {
            Some(s) if !t.loc_key.is_empty() && s.entries.contains_key(&t.loc_key) => s.tr(&t.loc_key),
            _ => &t.text,
        };
        if shown.is_empty() {
            continue;
        }
        let req = TextRequest {
            text: shown,
            font: &t.font,
            rich: t.rich_text,
            params: params(t),
            effects: Effects {
                outline_em: t.outline_width / 100.0,
                outline_color: t.outline_color.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8),
                ..Default::default()
            },
            rtl: None,
        };
        if let Some(block) = sys.block(&req) {
            handle.push(make_draw(&block, t, matrix));
        }
    }
}

#[cfg(all(test, feature = "client"))]
mod tests {
    use super::*;

    fn system() -> TextHandle {
        let mut sys = zestengine_text::TextSystem::new(zestengine_text::TextConfig { glyph_budget: std::time::Duration::from_secs(60), ..Default::default() });
        let defs = egui::FontDefinitions::default();
        for name in defs.families[&egui::FontFamily::Proportional].clone() {
            sys.add_builtin_font(&name, defs.font_data[&name].font.to_vec());
        }
        zestengine_text::TextShared::new(sys)
    }

    #[test]
    fn native_queue_reads_current_text_pose_and_visibility_after_reordering() {
        use crate::scene::SceneObject;
        use zestengine_core::{Disabled, Transform, ecs::{ComponentSet, Enabled, Parent}};
        let handle = system();
        let mut scene = LoadedScene::empty();
        scene.objects.push(SceneObject::from_components(ComponentSet::new()).with_transform(Transform { translation: Vec3::X * 10.0, ..Default::default() }));
        scene.objects.push(SceneObject::from_components(ComponentSet::from_vec(vec![
            Box::new(Disabled(Box::new(Text3D { text: "disabled".into(), ..Default::default() }))),
            Box::new(Text3D { text: "ab".into(), ..Default::default() }),
            Box::new(Text3D { text: "xyz".into(), ..Default::default() }),
        ])).with_parent(Some(0)).with_transform(Transform { translation: Vec3::Y, ..Default::default() }));
        scene.objects.push(SceneObject::from_components(ComponentSet::from_vec(vec![Box::new(Text3D { text: "q".into(), ..Default::default() })])));
        let id = scene.objects[1].id();
        let parent_id = scene.objects[0].id();
        let draws = || {
            let mut draws = handle.draws.lock().unwrap();
            let out = draws.iter().map(|draw| (draw.instances.len(), draw.transform[3][..3].to_vec())).collect::<Vec<_>>();
            draws.clear();
            out
        };
        queue_scene(&handle, &scene, None);
        assert_eq!(draws(), [(2, vec![10.0, 1.0, 0.0]), (1, vec![0.0; 3])]);
        scene.objects.swap(1, 2);
        scene.objects.query(|world, _| {
            world.get_mut::<Text3D>(world.entity(id).unwrap()).unwrap().text = "live".into();
            world.get_mut::<Transform>(world.entity(parent_id).unwrap()).unwrap().translation = Vec3::Z * 4.0;
        });
        queue_scene(&handle, &scene, None);
        assert_eq!(draws(), [(1, vec![0.0; 3]), (4, vec![0.0, 1.0, 4.0])]);
        scene.objects[2].components.set_enabled(1, false);
        queue_scene(&handle, &scene, None);
        assert_eq!(draws(), [(1, vec![0.0; 3]), (3, vec![0.0, 1.0, 4.0])]);
        scene.scene_view.edit = true;
        scene.set_view_hidden(0, true);
        queue_scene(&handle, &scene, None);
        assert_eq!(draws(), [(1, vec![0.0; 3])]);
        scene.scene_view.edit = false;
        scene.objects.query(|world, _| world.get_mut::<Enabled>(world.entity(parent_id).unwrap()).unwrap().0 = false);
        queue_scene(&handle, &scene, None);
        assert_eq!(draws(), [(1, vec![0.0; 3])]);
        scene.objects.query(|world, _| world.get_mut::<Parent>(world.entity(id).unwrap()).unwrap().0 = None);
        queue_scene(&handle, &scene, None);
        assert_eq!(draws(), [(1, vec![0.0; 3]), (3, vec![0.0, 1.0, 0.0])]);
        // Existing callers supplying their own derived visibility/pose arrays
        // still get that snapshot, independently of the current scene view.
        queue(&handle, &scene, &[false, false, true], &[Mat4::IDENTITY; 3], None);
        assert_eq!(draws(), [(3, vec![0.0; 3])]);
    }

    #[test]
    fn draw_is_centred_on_the_origin_in_metres() {
        let h = system();
        let t = Text3D { text: "Hello".into(), size: 0.2, ..Default::default() };
        let block = h.system.lock().unwrap().block(&TextRequest {
            text: &t.text,
            font: "",
            rich: false,
            params: params(&t),
            effects: Effects::default(),
            rtl: None,
        })
        .unwrap();
        // 0.2 m em = 200 layout px.
        assert_eq!(block.size, 200.0);
        let world = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let d = make_draw(&block, &t, world);
        assert_eq!(d.space, TextSpace::World);
        assert_eq!(d.local_offset, [-block.width * 0.5, -block.height * 0.5]);
        // One layout px is a millimetre; layout y runs down.
        assert!((d.transform[0][0] - 0.001).abs() < 1e-6 && (d.transform[1][1] + 0.001).abs() < 1e-6);
        assert_eq!(d.transform[3][..3], [1.0, 2.0, 3.0]);
        let b = make_draw(&block, &Text3D { billboard: Text3DBillboard::Full, ..Default::default() }, world);
        assert_eq!(b.billboard, Billboard::Full);
        assert!(b.transform[1][1] > 0.0, "billboards flip y in the shader");
    }

    #[test]
    fn explicit_box_sets_the_pivot() {
        let h = system();
        let t = Text3D { text: "x".into(), width: 2.0, height: 1.0, ..Default::default() };
        let block = h.system.lock().unwrap().block(&TextRequest {
            text: "x", font: "", rich: false, params: params(&t), effects: Effects::default(), rtl: None,
        })
        .unwrap();
        let d = make_draw(&block, &t, Mat4::IDENTITY);
        assert_eq!(d.local_offset, [-1000.0, -500.0]);
    }
}

/// Queue the glyphs the current language's strings can draw, so a language switch does not
/// rasterize them on first use (the text system spreads the work over frames inside its glyph
/// budget). Cheap to call every frame: a language it has already queued is skipped.
pub fn warm_language(sys: &mut zestengine_text::TextSystem, strings: &zestengine_core::Localization) {
    let lang = strings.current_language();
    sys.warm_language("", lang, strings.entries.len() as u64, || strings.used_chars(lang));
}
