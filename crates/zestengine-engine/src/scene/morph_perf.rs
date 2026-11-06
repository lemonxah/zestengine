//! CPU cost of the real avatar's morph/skin path (headless: no renderer), and
//! the regression guards for it. The avatar test skips when the file is absent.

use std::time::Instant;

use glam::Mat4;
use zestengine_core::{Animator, BlendShapes};

use super::LoadedScene;

const AVATAR: &str = "/home/lemonxah/ZestEngineProjects/my-game-2/models/lemonjuice.fbx";

fn load_avatar() -> Option<LoadedScene> {
    let path = std::path::Path::new(AVATAR);
    if !path.exists() {
        eprintln!("morph_perf: {AVATAR} absent, skipped");
        return None;
    }
    let settings = zestengine_assets::load_asset_meta(path).ok().flatten().and_then(|m| m.importer.as_model().cloned()).unwrap_or_default();
    let asset = zestengine_assets::load_fbx_with(path, &settings).ok()?;
    let mut scene = LoadedScene::empty();
    scene.add_asset_scene_with(None, &asset, Some(path)).ok()?;
    Some(scene)
}

/// Mean ms of `f` over `n` runs after one warm-up.
fn ms(n: usize, mut f: impl FnMut()) -> f64 {
    f();
    let t = Instant::now();
    for _ in 0..n {
        f();
    }
    t.elapsed().as_secs_f64() * 1e3 / n as f64
}

/// The object whose meshes carry the most blendshapes (the Body).
fn body_index(scene: &LoadedScene) -> usize {
    (0..scene.objects.len()).max_by_key(|&i| scene.morph_names(i).len()).unwrap()
}

fn set_weights(scene: &mut LoadedScene, obj: usize, n_active: usize, w: f32) {
    let names = scene.morph_names(obj);
    let comps = &mut scene.objects[obj].components;
    let pos = comps.iter().position(|c| c.as_any().is::<BlendShapes>());
    let pos = pos.unwrap_or_else(|| {
        comps.push(Box::new(BlendShapes::default()));
        comps.len() - 1
    });
    let b = comps[pos].as_any_mut().downcast_mut::<BlendShapes>().unwrap();
    b.weights = names.iter().take(n_active).map(|n| (n.clone(), w)).collect();
}

/// One frame of posing with a BlendShapes component (one active weight) on the Body must
/// stay cheap. The old per-frame name scans cost ~9 ms (release) here; the bound is
/// well below that (the new cost is ~0.1 ms).
#[test]
fn one_active_blendshape_costs_little_per_frame() {
    let Some(mut scene) = load_avatar() else { return };
    let body = body_index(&scene);
    set_weights(&mut scene, body, 1, 0.5);
    let per_frame = ms(10, || scene.pose_skeletons(1.0 / 60.0));
    let bound = 3.0;
    eprintln!("blendshape pose frame: {per_frame:.3} ms (bound {bound})");
    assert!(per_frame < bound, "{per_frame:.3} ms per frame");
    // The inspector's name list is shared, not cloned per frame.
    let a = scene.morph_names_shared(body);
    let b = scene.morph_names_shared(body);
    assert!(std::sync::Arc::ptr_eq(&a, &b));
    assert_eq!(&a[..], &scene.morph_names(body)[..]);
}

/// The fused CPU morph+skin gives the same vertices as the old two-step path on the
/// real Body, weights 0 -> 0.5 -> 0, in the rest pose, an animated pose and the T-pose.
#[test]
fn real_body_morph_skin_is_identical_to_the_two_step_path() {
    let Some(scene) = load_avatar() else { return };
    let mut cpu = crate::skin_cpu::CpuSkin::default();
    for sm in scene.skinned_meshes.iter().filter(|sm| !sm.morphs.is_empty()) {
        let skel = &scene.skeletons[sm.skeleton];
        let rest = skel.rest_locals();
        let animated = match scene.animations.iter().zip(&scene.clip_skeleton).find(|(_, s)| **s == sm.skeleton) {
            Some((clip, _)) => clip.sample(skel, clip.duration * 0.37),
            None => rest.iter().enumerate().map(|(i, m)| *m * Mat4::from_rotation_z(0.05 * (i % 5) as f32)).collect(),
        };
        let palettes = [skel.palette(&rest), skel.palette(&animated), vec![Mat4::IDENTITY; skel.joints.len()]];
        for palette in &palettes {
            for scale in [0.0f32, 0.5, 0.0] {
                let w: Vec<f32> = (0..sm.morphs.len()).map(|i| if i % 3 == 0 { scale } else { 0.0 }).collect();
                let got = cpu.morph_skin(&sm.base_vertices, &sm.morphs, &w, palette);
                let morphed = crate::face_drive::morphed(&sm.base_vertices, &sm.morphs, &w);
                let want = super::skin_vertices(morphed.as_deref().unwrap_or(&sm.base_vertices), palette);
                assert!(got.iter().zip(&want).all(|(a, b)| a.position == b.position && a.normal == b.normal), "mesh {} scale {scale}", sm.mesh_index);
            }
        }
    }
}

#[test]
#[ignore = "profile table: cargo test -p zestengine-engine --lib morph_perf::profile -- --ignored --nocapture"]
fn profile_real_avatar_frame() {
    let Some(mut scene) = load_avatar() else { return };
    let body = body_index(&scene);
    eprintln!("objects {} skinned meshes {} body {} ({} shapes)", scene.objects.len(), scene.skinned_meshes.len(), scene.objects[body].name, scene.morph_names(body).len());
    let mut rows: Vec<(&str, f64)> = Vec::new();
    eprintln!("animations {} skeletons {}", scene.animations.len(), scene.skeletons.len());
    {
        // Shared path with a looping clip: every skinned mesh is re-skinned on the CPU each frame.
        let mut cpu = crate::skin_cpu::CpuSkin::default();
        let verts: usize = scene.skinned_meshes.iter().map(|s| s.base_vertices.len()).sum();
        let t = ms(10, || {
            for sm in &scene.skinned_meshes {
                let skel = &scene.skeletons[sm.skeleton];
                let palette = skel.palette(&skel.rest_locals());
                std::hint::black_box(cpu.morph_skin(&sm.base_vertices, &[], &[], &palette));
            }
        });
        eprintln!("all {} skinned meshes ({verts} verts) CPU skin, one frame: {t:.3} ms", scene.skinned_meshes.len());
    }
    rows.push(("a idle (nothing taken)", ms(20, || scene.pose_skeletons(1.0 / 60.0))));
    set_weights(&mut scene, body, 1, 0.5);
    rows.push(("b BlendShapes on Body, 1 weight", ms(20, || scene.pose_skeletons(1.0 / 60.0))));
    set_weights(&mut scene, body, 200, 0.3);
    rows.push(("c 200 weights", ms(20, || scene.pose_skeletons(1.0 / 60.0))));
    {
        let slots: Vec<usize> = scene.objects[body].render_slots().map(|r| r.mesh).collect();
        let morphs: Vec<&[zestengine_assets::MorphTarget]> = slots.iter().filter_map(|m| scene.skinned_meshes.iter().find(|s| s.mesh_index == *m)).map(|s| s.morphs.as_slice()).collect();
        let mut comps = std::mem::take(&mut scene.objects[body].components);
        rows.push(("c2 face_drive::drive alone (Body)", ms(20, || {
            std::hint::black_box(crate::face_drive::drive(&mut comps, &morphs, None, None, None, 1.0 / 60.0));
        })));
        scene.objects[body].components = comps;
    }
    // CPU skin + morph of the Body mesh as the non-GPU path does it each frame.
    let (sm_i, _) = scene.skinned_meshes.iter().enumerate().filter(|(_, s)| !s.morphs.is_empty()).max_by_key(|(_, s)| s.base_vertices.len()).unwrap();
    eprintln!("body mesh: {} verts, {} shapes, {} deltas", scene.skinned_meshes[sm_i].base_vertices.len(), scene.skinned_meshes[sm_i].morphs.len(), scene.skinned_meshes[sm_i].morphs.iter().map(|m| m.deltas.len()).sum::<usize>());
    let palette = vec![Mat4::IDENTITY; scene.skeletons[scene.skinned_meshes[sm_i].skeleton].joints.len()];
    let w1: Vec<f32> = (0..scene.skinned_meshes[sm_i].morphs.len()).map(|i| if i == 0 { 0.5 } else { 0.0 }).collect();
    let wall: Vec<f32> = w1.iter().map(|_| 0.3).collect();
    rows.push(("d1b CPU morphed+skin Body, all weights", ms(5, || {
        let sm = &scene.skinned_meshes[sm_i];
        let v = crate::face_drive::morphed(&sm.base_vertices, &sm.morphs, &wall);
        std::hint::black_box(super::skin_vertices(v.as_deref().unwrap_or(&sm.base_vertices), &palette));
    })));
    rows.push(("d1 CPU morphed+skin Body, 1 weight", ms(20, || {
        let sm = &scene.skinned_meshes[sm_i];
        let v = crate::face_drive::morphed(&sm.base_vertices, &sm.morphs, &w1);
        std::hint::black_box(super::skin_vertices(v.as_deref().unwrap_or(&sm.base_vertices), &palette));
    })));
    // All parts on the CPU skin path (Animator without GPU skinning on every skinned object).
    for o in scene.objects.iter_mut() {
        if o.render_slots().next().is_some() && !o.components.iter().any(|c| c.as_any().is::<Animator>()) {
            let a = Animator { gpu_skinning: false, ..Default::default() };
            o.components.push(Box::new(a));
        }
    }
    rows.push(("d2 all parts Animator (pose only)", ms(20, || scene.pose_skeletons(1.0 / 60.0))));
    let names = scene.morph_names(body);
    rows.push(("e1 morph_names(Body)", ms(200, || {
        std::hint::black_box(scene.morph_names(body));
    })));
    rows.push(("e2 names clone (info model)", ms(200, || {
        std::hint::black_box(names.clone());
    })));
    for (label, v) in rows {
        eprintln!("{label:40} {v:9.3} ms");
    }
}
