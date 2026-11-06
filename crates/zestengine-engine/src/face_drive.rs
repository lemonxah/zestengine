//! Drives a model's blendshapes from its Face Tracking (and Blend Shapes)
//! components each skinning pass, and applies the weighted morph offsets to
//! the bind-pose vertices before skinning.

use zestengine_assets::MorphTarget;
use zestengine_core::{BlendShapes, Component, FaceFrame, FaceSource, FaceTracking, LipSync, LipSyncMode, ShapeSource, VisemeFrame, viseme_index};
use glam::{Vec2, Vec3};
use std::collections::HashMap;

pub(crate) struct FaceOutput {
    /// Per skinned slot, one weight per morph target.
    pub weights: Vec<Vec<f32>>,
    /// Eye gaze (left, right) and max (yaw, pitch) radians, when the eye
    /// bones follow it.
    pub eyes: Option<(Vec2, Vec2, Vec2)>,
}

/// Resolve this frame's blendshape weights for an object. `morphs` are its
/// skinned slots' blendshapes, `local` this machine's tracking and
/// `network` the (remote?, face) of the Network Avatar above it.
pub(crate) fn drive(
    components: &mut [Box<dyn Component>],
    morphs: &[&[MorphTarget]],
    local: Option<&FaceFrame>,
    network: Option<(bool, Option<FaceFrame>)>,
    visemes: Option<&VisemeFrame>,
    dt: f32,
) -> FaceOutput {
    // Borrowed names: a face model has hundreds, and this runs every frame.
    let names: Vec<&str> = {
        let mut v: Vec<&str> = morphs.iter().flat_map(|m| m.iter().map(|t| t.name.as_str())).collect();
        v.dedup();
        v
    };
    let mut authored: Vec<(String, f32)> = Vec::new();
    for c in components.iter_mut() {
        if let Some(b) = c.as_any_mut().downcast_mut::<BlendShapes>() {
            if !same_names(&b.runtime.names, &names) {
                b.runtime.names = names.iter().map(|n| (*n).to_owned()).collect();
            }
            authored = b.weights.clone();
        }
    }
    let mut eyes = None;
    let mut sources: Vec<Option<ShapeSource>> = Vec::new();
    let mut frame: Option<FaceFrame> = None;
    for c in components.iter_mut() {
        let Some(ft) = c.as_any_mut().downcast_mut::<FaceTracking>() else { continue };
        let remote = network.as_ref().is_some_and(|(r, _)| *r);
        let target = match ft.source {
            FaceSource::Auto if remote => network.as_ref().and_then(|(_, f)| f.clone()),
            FaceSource::Auto | FaceSource::Local => local.cloned(),
            FaceSource::Network => network.as_ref().and_then(|(_, f)| f.clone()),
        };
        ft.runtime.active = target.is_some();
        if let Some(t) = &target {
            let k = 1.0 - ft.smoothing.clamp(0.0, 0.98).powf(dt.max(0.0) * 60.0);
            ft.runtime.frame.approach(t, k);
        }
        if !same_names(&ft.runtime.shapes, &names) {
            ft.runtime.shapes = names.iter().map(|n| (*n).to_owned()).collect();
            ft.runtime.matched = names.iter().filter_map(|n| ft.source_for(n).map(|(_, e)| ((*n).to_owned(), e))).collect();
        }
        if ft.runtime.active {
            sources = names.iter().map(|n| ft.source_for(n).map(|(s, _)| s)).collect();
            frame = Some(ft.runtime.frame.clone());
            let f = &ft.runtime.frame;
            if ft.eye_bones && f.has_gaze {
                let max = Vec2::new(ft.eye_angle[0].to_radians(), ft.eye_angle[1].to_radians());
                eyes = Some((f.gaze_left, f.gaze_right, max));
            }
        }
        break;
    }
    // Lip sync: visemes from the voice, when its mode allows.
    let tracked = frame.is_some();
    let mut lip: Option<(VisemeFrame, f32, bool)> = None;
    for c in components.iter_mut() {
        let Some(ls) = c.as_any_mut().downcast_mut::<LipSync>() else { continue };
        let on = match ls.mode {
            LipSyncMode::Off => false,
            LipSyncMode::WhenUntracked => !tracked,
            LipSyncMode::Always => true,
        };
        ls.runtime.active = on && visemes.is_some_and(|v| v.openness() > 0.01);
        let target = if on { visemes.copied().unwrap_or_default() } else { VisemeFrame::default() };
        let k = 1.0 - ls.smoothing.clamp(0.0, 0.98).powf(dt.max(0.0) * 60.0);
        ls.runtime.frame.approach(&target, k);
        if ls.runtime.matched.len() != names.iter().filter(|n| viseme_index(n).is_some()).count() {
            ls.runtime.matched = names
                .iter()
                .filter_map(|n| viseme_index(n).map(|i| ((*n).to_owned(), zestengine_core::VISEMES[i].to_owned())))
                .collect();
        }
        if on {
            let has_visemes = !ls.runtime.matched.is_empty();
            lip = Some((ls.runtime.frame, ls.gain, ls.drive_face_shapes && !has_visemes));
        }
        break;
    }
    let lip_face = lip.as_ref().filter(|(_, _, drive)| *drive).map(|(v, g, _)| {
        let mut scaled = *v;
        for w in &mut scaled.0 {
            *w = (*w * g).min(1.0);
        }
        scaled.to_face()
    });
    // Nothing drives any shape (no tracked frame, no lip sync, nothing authored): all zero.
    if frame.is_none() && lip.is_none() && authored.is_empty() {
        return FaceOutput { weights: morphs.iter().map(|m| vec![0.0; m.len()]).collect(), eyes };
    }
    // Hashed lookups: the weight of every shape is asked for once per frame (and again per in-between group).
    let mut index: HashMap<&str, usize> = HashMap::with_capacity(names.len());
    for (k, n) in names.iter().enumerate() {
        index.entry(n).or_insert(k);
    }
    let mut authored_by: HashMap<&str, f32> = HashMap::with_capacity(authored.len());
    for (n, w) in &authored {
        authored_by.entry(n.as_str()).or_insert(*w);
    }
    let weight = |name: &str| -> f32 {
        // Name parsing only when a source can use it (it lowercases the name).
        if let Some((v, gain, _)) = &lip
            && let Some(i) = viseme_index(name)
        {
            return (v.get(i) * gain).min(1.0);
        }
        let k = index.get(name).copied();
        let tracked_value = frame.as_ref().zip(k.and_then(|k| sources.get(k)?.as_ref())).map(|(f, src)| src.read(f));
        let lip_value = lip_face.as_ref().zip(lip_face.as_ref().and_then(|_| ShapeSource::resolve(name))).map(|(f, src)| src.read(f));
        match (tracked_value, lip_value) {
            (Some(a), Some(b)) => a.max(b),
            (Some(a), None) | (None, Some(a)) => a,
            (None, None) => authored_by.get(name).copied().unwrap_or(0.0),
        }
    };
    // In-between frames (`Name@50`) share their shape's weight along its piecewise-linear path.
    let weights = morphs.iter().map(|m| zestengine_assets::anim_import::inbetween_weights(m, &weight)).collect();
    FaceOutput { weights, eyes }
}

fn same_names(owned: &[String], names: &[&str]) -> bool {
    owned.len() == names.len() && owned.iter().zip(names).all(|(a, b)| a == b)
}

/// `base` with the weighted morph offsets added, or None when every weight
/// is (near) zero.
pub(crate) fn morphed(base: &[zestengine_render::Vertex], morphs: &[MorphTarget], weights: &[f32]) -> Option<Vec<zestengine_render::Vertex>> {
    if !weights.iter().any(|w| w.abs() > 1e-4) {
        return None;
    }
    let mut out = base.to_vec();
    let mut touched = vec![false; out.len()];
    for (m, &w) in morphs.iter().zip(weights) {
        if w.abs() <= 1e-4 {
            continue;
        }
        for &(i, dp, dn) in &m.deltas {
            let Some(v) = out.get_mut(i as usize) else { continue };
            v.position = (Vec3::from(v.position) + Vec3::from(dp) * w).to_array();
            v.normal = (Vec3::from(v.normal) + Vec3::from(dn) * w).to_array();
            touched[i as usize] = true;
        }
    }
    for (v, t) in out.iter_mut().zip(touched) {
        if t {
            v.normal = Vec3::from(v.normal).normalize_or(Vec3::Y).to_array();
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(name: &str) -> MorphTarget {
        MorphTarget { name: name.into(), deltas: vec![(0, [0.0, 1.0, 0.0], [0.0; 3])] }
    }

    #[test]
    fn the_gpu_morph_reference_matches_the_cpu_morph_path() {
        let base: Vec<zestengine_render::Vertex> = (0..8)
            .map(|i| zestengine_render::Vertex {
                position: [i as f32 * 0.5, 0.0, 1.0],
                normal: Vec3::new(0.2, 1.0, 0.1 * i as f32).normalize().to_array(),
                ..Default::default()
            })
            .collect();
        let morphs = vec![
            MorphTarget { name: "a".into(), deltas: vec![(1, [0.0, 1.0, 0.0], [0.3, 0.0, 0.0]), (5, [0.2, 0.0, 0.1], [0.0, 0.2, 0.0])] },
            MorphTarget { name: "b".into(), deltas: vec![(1, [1.0, 0.0, 0.0], [0.0; 3]), (7, [0.0, 0.0, 2.0], [0.0, 0.0, 0.4])] },
            MorphTarget { name: "c".into(), deltas: vec![(3, [0.5; 3], [0.1; 3])] },
        ];
        let shapes: Vec<zestengine_render::MorphDeltas<'_>> = morphs.iter().map(|m| m.deltas.as_slice()).collect();
        let tables = zestengine_render::MorphTables::build(&base, &shapes);
        for weights in [[0.5, 1.0, 0.0], [1.0, -0.4, 0.7], [0.0, 0.0, 5e-5]] {
            let cpu = morphed(&base, &morphs, &weights).unwrap_or_else(|| base.clone());
            let gpu = zestengine_render::morph_reference(&base, &tables, &weights);
            for (i, (a, b)) in cpu.iter().zip(&gpu).enumerate() {
                let close = |x: [f32; 3], y: [f32; 3]| Vec3::from(x).distance(Vec3::from(y)) < 1e-5;
                assert!(close(a.position, b.position), "vertex {i} position, weights {weights:?}");
                assert!(close(a.normal, b.normal), "vertex {i} normal, weights {weights:?}");
            }
        }
    }

    #[test]
    fn tracked_face_drives_matching_shapes_and_authored_weights_fill_the_rest() {
        let mut components: Vec<Box<dyn Component>> = vec![
            Box::new(FaceTracking { smoothing: 0.0, ..Default::default() }),
            Box::new(BlendShapes { weights: vec![("Hair_Up".into(), 0.3), ("JawOpen".into(), 1.0)], ..Default::default() }),
        ];
        let morphs = [target("JawOpen"), target("eyeBlinkLeft"), target("Hair_Up")];
        let mut face = FaceFrame::default();
        face.set_param("JawOpen", 0.5);
        face.set_param("EyeLidLeft", 0.0);
        let out = drive(&mut components, &[&morphs], Some(&face), None, None, 1.0 / 60.0);
        assert_eq!(out.weights[0], vec![0.5, 1.0, 0.3]);
        let ft = components[0].as_any().downcast_ref::<FaceTracking>().unwrap();
        assert!(ft.runtime.active);
        assert_eq!(ft.runtime.matched.len(), 2, "JawOpen and eyeBlinkLeft match");

        // Without tracking the authored weight shows instead.
        let out = drive(&mut components, &[&morphs], None, None, None, 1.0 / 60.0);
        assert_eq!(out.weights[0], vec![1.0, 0.0, 0.3]);
    }

    #[test]
    fn remote_avatars_use_the_replicated_face() {
        let mut components: Vec<Box<dyn Component>> = vec![Box::new(FaceTracking { smoothing: 0.0, ..Default::default() })];
        let morphs = [target("JawOpen")];
        let mut local = FaceFrame::default();
        local.set_param("JawOpen", 0.2);
        let mut remote = FaceFrame::default();
        remote.set_param("JawOpen", 0.9);
        let out = drive(&mut components, &[&morphs], Some(&local), Some((true, Some(remote))), None, 0.016);
        assert!((out.weights[0][0] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn lip_sync_drives_visemes_and_the_jaw_when_untracked() {
        let mut components: Vec<Box<dyn Component>> =
            vec![Box::new(FaceTracking::default()), Box::new(LipSync { smoothing: 0.0, ..Default::default() })];
        let mut v = VisemeFrame::default();
        v.0[10] = 0.8;
        let morphs = [target("vrc.v_aa"), target("v_oh")];
        let out = drive(&mut components, &[&morphs], None, None, Some(&v), 0.016);
        assert!((out.weights[0][0] - 0.8).abs() < 1e-6 && out.weights[0][1] == 0.0);
        // Face tracking active: this mode steps aside.
        let face = FaceFrame::default();
        let out = drive(&mut components, &[&morphs], Some(&face), None, Some(&v), 0.016);
        assert_eq!(out.weights[0][0], 0.0);
        // A face-only avatar gets its jaw opened instead.
        let mut components: Vec<Box<dyn Component>> = vec![Box::new(LipSync { smoothing: 0.0, ..Default::default() })];
        let morphs = [target("JawOpen")];
        let out = drive(&mut components, &[&morphs], None, None, Some(&v), 0.016);
        assert!(out.weights[0][0] > 0.4, "{:?}", out.weights);
    }

    #[test]
    fn morph_offsets_scale_with_weight() {
        let base = vec![zestengine_render::Vertex::default(); 2];
        let v = morphed(&base, &[target("JawOpen")], &[0.5]).unwrap();
        assert_eq!(v[0].position, [0.0, 0.5, 0.0]);
        assert_eq!(v[1].position, [0.0; 3]);
        assert!(morphed(&base, &[target("JawOpen")], &[0.0]).is_none());
    }
}
