//! Headless screenshots of the Model Import inspector sections.

use super::*;
use zestengine_assets::TakeInfo;
use crate::shot::Shot;

fn stats() -> Stats {
    let take = |n: &str, a: f32, b: f32| TakeInfo { name: n.into(), start: a / 30.0, stop: b / 30.0, frame_rate: 30.0, first_frame: a, last_frame: b };
    Stats {
        meshes: 6,
        vertices: 48_000,
        max_vertices: 20_000,
        triangles: 90_000,
        materials: vec!["Skin".into(), "Hair".into(), "Cloth".into()],
        embedded: vec!["Skin/albedo".into(), "Skin/normal".into()],
        cameras: 0,
        lights: 0,
        clips: 3,
        blendshapes: 52,
        takes: vec![take("Idle", 0.0, 60.0), take("Walk_Loop", 0.0, 32.0), take("Wave", 10.0, 90.0)],
        clip_rows: vec![("Idle".into(), 0.4, 0), ("Walk_Loop".into(), 6.5, 12), ("Wave".into(), 31.0, 0)],
        warnings: vec!["412 of 12000 skinned vertices have more than 4 bone influences (up to 6); the weakest are dropped and the rest renormalised.".into()],
        constraints: 3,
        custom_props: 2,
        optimize: Default::default(),
        rig: Some(std::sync::Arc::new(stick_figure())),
    }
}

/// A small figure with an arm that waves, for the Clip Preview.
fn stick_figure() -> RigPreview {
    use glam::{Quat, Vec3};
    use zestengine_assets::{AnimChannel, AnimationClip, Joint, Skeleton};
    let joint = |name: &str, parent: Option<usize>, t: Vec3| Joint {
        name: name.into(),
        parent,
        inverse_bind: glam::Mat4::IDENTITY,
        rest_translation: t,
        rest_rotation: Quat::IDENTITY,
        rest_scale: Vec3::ONE,
    };
    let joints = vec![
        joint("Hips", None, Vec3::new(0.0, 1.0, 0.0)),
        joint("Spine", Some(0), Vec3::new(0.0, 0.3, 0.0)),
        joint("Head", Some(1), Vec3::new(0.0, 0.4, 0.0)),
        joint("ArmL", Some(1), Vec3::new(-0.2, 0.3, 0.0)),
        joint("HandL", Some(3), Vec3::new(-0.5, 0.0, 0.0)),
        joint("ArmR", Some(1), Vec3::new(0.2, 0.3, 0.0)),
        joint("HandR", Some(5), Vec3::new(0.5, 0.0, 0.0)),
        joint("LegL", Some(0), Vec3::new(-0.1, -0.1, 0.0)),
        joint("FootL", Some(7), Vec3::new(0.0, -0.9, 0.0)),
        joint("LegR", Some(0), Vec3::new(0.1, -0.1, 0.0)),
        joint("FootR", Some(9), Vec3::new(0.0, -0.9, 0.0)),
    ];
    let wave = AnimChannel::rot(5, vec![0.0, 0.5, 1.0], vec![Quat::IDENTITY, Quat::from_rotation_z(1.2), Quat::IDENTITY]);
    RigPreview {
        skeleton: Skeleton { joints, rig: None, ..Default::default() },
        clips: vec![AnimationClip { name: "Wave".into(), duration: 1.0, channels: vec![wave], morph_tracks: Vec::new(), playback: None }],
    }
}

fn shot_of(name: &str, customize: bool) {
    let path = std::path::PathBuf::from("/proj/art/hero.fbx");
    let mut ui_state = ModelImportUi { loaded: Some(Loaded { path: path.clone(), stats: Ok(stats()) }), ..Default::default() };
    let mut m = ModelImport { import_blend_shapes: true, ..Default::default() };
    if customize {
        let takes = ui_state.stats(&path).unwrap().takes.clone();
        m.animation.customize_from_takes(&takes);
    }
    let mut shot = Shot::new(380.0, 3000.0, 1.5);
    let img = shot.run_panel(|ui| {
        ui_state.show(ui, &path, &mut m);
    });
    shot.save_named(&img, name);
    assert!(img.non_background() > 500);
}

#[test]
fn model_import_sections() {
    shot_of("model_import_takes", false);
    shot_of("model_import_clips", true);
}
