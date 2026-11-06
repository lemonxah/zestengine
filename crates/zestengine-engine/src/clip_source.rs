//! Clips taken from another model: an animation-only FBX (Mixamo "Without
//! Skin") or any model's clip played on an avatar that is not its own.
//!
//! A controller refers to such a clip as `<model>#<clip name>` (the model by
//! project path or `@id`). Loading reads the model through the import cache and
//! turns the clip into an [`AnimClip`] that any rig can play by name:
//!
//! * bone tracks keyed by the source bone's name (translation only where the
//!   bone moves, never the rest value), played on a rig through
//!   [`Rig::index_loose`](crate::anim_graph::Rig::index_loose) (`mixamorig:Hips`
//!   finds `Hips`) with the offset-from-rest translation scaling of
//!   [`AnimClip::source_rest`];
//! * when the source skeleton maps to a humanoid, the same motion baked into
//!   muscle space (`AnimClip::humanoid`), which a humanoid rig plays instead, so
//!   a different naming, rest pose (A against T) or proportions retarget.
//!
//! Animation Type / Avatar Definition > Copy From Other Avatar in the rig
//! section picks which model resolves the bone map of the *source*; the target
//! rig always uses its own.

use std::path::Path;

use anyhow::{Context as _, Result, anyhow};
use glam::{Quat, Vec3};
use zestengine_assets::muscle::{MuscleRig, bake_muscle_clip};
use zestengine_assets::{AnimClip, AnimTrack, AnimationClip, ChannelPath, ClipEvent, EventArgs, KeyInterp, Keyframe, Skeleton, SourceBone, TrackProperty};

/// Model file extensions a clip reference can name.
const MODEL_EXTENSIONS: &[&str] = &["fbx", "glb", "gltf", "vrm"];

/// The reference to clip `clip` of `model` (a project path or `@id`).
pub fn clip_ref(model: &str, clip: &str) -> String {
    format!("{model}#{clip}")
}

/// `(model, clip)` of a `<model>#<clip>` reference; `None` for anything else.
pub fn split_ref(name: &str) -> Option<(&str, &str)> {
    let (model, clip) = name.split_once('#')?;
    if clip.is_empty() {
        return None;
    }
    let by_id = zestengine_core::asset_ref::is_ref(model);
    let by_ext = model.rsplit_once('.').is_some_and(|(_, e)| MODEL_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()));
    (by_id || by_ext).then_some((model, clip))
}

/// Whether `name` is a clip of another model.
pub fn is_model_ref(name: &str) -> bool {
    split_ref(name).is_some()
}

/// Load a clip reference: a `.anim` file, or a clip of another model.
pub fn load_any(root: &Path, name: &str) -> Result<AnimClip> {
    if is_model_ref(name) { load(root, name) } else { zestengine_assets::load_anim_clip(zestengine_core::asset_ref::resolve_path(root, name)) }
}

/// Load the clip a `<model>#<clip>` reference names, resolved against `root`.
pub fn load(root: &Path, name: &str) -> Result<AnimClip> {
    let (model, clip) = split_ref(name).ok_or_else(|| anyhow!("{name} is not a model clip reference"))?;
    let path = zestengine_core::asset_ref::resolve_path(root, model);
    let scene = zestengine_assets::load_model_with_meta(&path).with_context(|| format!("reading {}", path.display()))?;
    let skeleton = scene.skeletons.first().ok_or_else(|| anyhow!("{model} has no skeleton"))?;
    let imported = scene
        .animations
        .iter()
        .find(|c| c.name == clip)
        .ok_or_else(|| anyhow!("{model} has no clip named {clip}"))?;
    let mut skeleton = skeleton.clone();
    crate::humanoid::ensure_rigs(std::slice::from_mut(&mut skeleton));
    Ok(to_asset(&skeleton, imported, 30.0))
}

/// Turn an imported clip of `skeleton` into a clip any rig can play by name.
/// `fps` is the muscle bake rate.
pub fn to_asset(skeleton: &Skeleton, clip: &AnimationClip, fps: f32) -> AnimClip {
    let mut out = AnimClip { length: clip.duration.max(1e-3), looping: clip.playback.as_ref().is_none_or(|p| p.loop_time), frame_rate: fps, ..Default::default() };
    out.source_rest = skeleton.joints.iter().map(|j| SourceBone { name: j.name.clone(), translation: j.rest_translation.to_array() }).collect();
    for ch in &clip.channels {
        let Some(joint) = skeleton.joints.get(ch.joint) else { continue };
        let n = ch.times.len();
        if n == 0 {
            continue;
        }
        // A channel that never leaves the rest value stays out, so the rig keeps its own.
        let (property, values): (TrackProperty, Vec<[f32; 4]>) = match ch.path {
            ChannelPath::Translation => {
                let vals: Vec<Vec3> = (0..n).filter_map(|i| ch.vec_at(i)).collect();
                if vals.iter().all(|v| v.abs_diff_eq(joint.rest_translation, 1e-5)) {
                    continue;
                }
                (TrackProperty::Translation, vals.iter().map(|v| [v.x, v.y, v.z, 0.0]).collect())
            }
            ChannelPath::Scale => {
                let vals: Vec<Vec3> = (0..n).filter_map(|i| ch.vec_at(i)).collect();
                if vals.iter().all(|v| v.abs_diff_eq(joint.rest_scale, 1e-5)) {
                    continue;
                }
                (TrackProperty::Scale, vals.iter().map(|v| [v.x, v.y, v.z, 0.0]).collect())
            }
            ChannelPath::Rotation => {
                let vals: Vec<Quat> = (0..n).filter_map(|i| ch.quat_at(i)).collect();
                if vals.iter().all(|q| q.abs_diff_eq(joint.rest_rotation, 1e-6) || q.abs_diff_eq(-joint.rest_rotation, 1e-6)) {
                    continue;
                }
                (TrackProperty::Rotation, vals.iter().map(|q| q.to_array()).collect())
            }
        };
        let mut track = AnimTrack::new(&joint.name, property);
        track.keys = ch
            .times
            .iter()
            .zip(values)
            .map(|(&time, value)| Keyframe { time, value, interp: KeyInterp::Linear, ..Default::default() })
            .collect();
        out.tracks.push(track);
    }
    if let Some(pb) = clip.playback.as_deref() {
        for (i, e) in pb.events.iter().enumerate() {
            out.events.push(ClipEvent { time: e.time, name: e.name.clone() });
            if e.float != 0.0 || e.int != 0 || !e.string.is_empty() || !e.object.is_empty() {
                out.event_args.push(EventArgs { event: i, float: e.float, int: e.int, string: e.string.clone(), object: e.object.clone() });
            }
        }
    }
    if let Some(rig) = skeleton.rig.as_ref().filter(|r| r.is_configured())
        && let Some(mr) = MuscleRig::new(skeleton, rig)
    {
        // Humanoid Oversampling: the muscle bake runs at 1/2/4/8 times the base rate.
        out.humanoid = Some(Box::new(bake_muscle_clip(skeleton, &mr, clip, fps * rig.oversampling() as f32)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn references_split_into_model_and_clip() {
        assert_eq!(split_ref("anims/Zombie Idle.fbx#mixamo.com"), Some(("anims/Zombie Idle.fbx", "mixamo.com")));
        assert_eq!(split_ref("a/run.glb#Run"), Some(("a/run.glb", "Run")));
        assert_eq!(split_ref("walk.anim"), None);
        assert_eq!(split_ref("notes#1"), None);
        assert_eq!(split_ref("a.fbx#"), None);
        assert_eq!(clip_ref("a.fbx", "Idle"), "a.fbx#Idle");
        assert!(is_model_ref(&clip_ref("a.fbx", "Idle")));
    }

    // ------------------------------------------------------------ real Mixamo files

    use crate::anim_graph::{ClipRef, Pose, Rig};
    use glam::Mat4;
    use std::path::PathBuf;

    const MIXAMO: &str = "/home/lemonxah/Downloads/maximo";

    fn mixamo(file: &str) -> Option<PathBuf> {
        let p = PathBuf::from(MIXAMO).join(file);
        p.exists().then_some(p)
    }

    /// Joint world positions (metres) of the avatar posed by `clip` at `t`.
    fn avatar_positions(avatar: &Skeleton, to_world: Mat4, clip: &AnimClip, rig: &Rig, t: f32) -> Vec<(String, Vec3)> {
        let mut pose = Pose::rest(rig);
        ClipRef::Asset(clip).sample(rig, t, true, &mut pose);
        let world = avatar.world_matrices(&pose.to_locals());
        avatar.joints.iter().zip(world).map(|(j, w)| (j.name.clone(), to_world.transform_point3(w.w_axis.truncate()))).collect()
    }

    use glam::Vec3;

    /// Largest distance between the avatar's bones and the reference evaluation, for bones both have.
    fn worst_gap(got: &[(String, Vec3)], reference: &[(String, [f32; 3])]) -> (f32, usize) {
        let (mut worst, mut n) = (0.0f32, 0usize);
        for (name, p) in got {
            if let Some((_, r)) = reference.iter().find(|(rn, _)| rn == name) {
                worst = worst.max(p.distance(Vec3::from_array(*r)));
                n += 1;
            }
        }
        (worst, n)
    }

    fn off() -> zestengine_assets::ModelImport {
        let mut m = zestengine_assets::ModelImport::default();
        m.animation.compression = zestengine_assets::AnimCompression::Off;
        m
    }

    /// The skinless "Zombie Idle" played on the skinned "T-Pose" avatar (same Mixamo
    /// rig, centimetre skeleton under a 0.01 instance) through the bone-name
    /// tracks puts every bone where ufbx's own evaluation of the file puts it.
    #[test]
    fn skinless_mixamo_clip_plays_on_a_skinned_avatar_by_bone_name() {
        let (Some(idle), Some(tpose)) = (mixamo("Zombie Idle.fbx"), mixamo("T-Pose.fbx")) else {
            eprintln!("skipped: Mixamo files not present");
            return;
        };
        let started = std::time::Instant::now();
        let source = zestengine_assets::load_fbx_with(&idle, &off()).expect("a skinless file imports");
        assert!(source.instances.is_empty() && source.meshes.is_empty(), "no mesh");
        let (skeleton, imported) = (&source.skeletons[0], &source.animations[0]);
        let avatar = zestengine_assets::load_fbx_with(&tpose, &off()).unwrap();
        let (avatar_skel, to_world) = (&avatar.skeletons[0], avatar.instances[0].transform);
        // Generic path: the avatar carries no humanoid map here, so the bone tracks play.
        let asset = to_asset(skeleton, imported, 30.0);
        assert!(asset.humanoid.is_none(), "no rig was attached to the source");
        let rig = Rig::from_skeleton(avatar_skel);
        assert!(rig.muscle.is_none());
        let mut worst = 0.0f32;
        for t in [0.0f32, 0.4, 1.3, 2.2, 3.1, 3.9] {
            let reference = zestengine_assets::reference_bone_positions(&idle, t as f64).unwrap();
            let (gap, n) = worst_gap(&avatar_positions(avatar_skel, to_world, &asset, &rig, t), &reference);
            assert!(n >= 50, "bones matched by name: {n}");
            worst = worst.max(gap);
        }
        eprintln!("skinless clip on skinned avatar: worst bone gap {:.2} mm over 6 times ({:?})", worst * 1000.0, started.elapsed());
        assert!(worst < 0.01, "worst gap {worst} m");
    }

    /// The same clip through the muscle twin (a humanoid avatar), which also retargets,
    /// stays within a few centimetres of the reference on the same-rig pair.
    #[test]
    fn skinless_mixamo_clip_plays_on_a_humanoid_avatar_through_muscles() {
        let (Some(idle), Some(tpose)) = (mixamo("Zombie Idle.fbx"), mixamo("T-Pose.fbx")) else {
            eprintln!("skipped: Mixamo files not present");
            return;
        };
        let started = std::time::Instant::now();
        let source = zestengine_assets::load_fbx_with(&idle, &off()).unwrap();
        let mut skeleton = source.skeletons[0].clone();
        crate::humanoid::ensure_rigs(std::slice::from_mut(&mut skeleton));
        assert!(skeleton.rig.as_ref().is_some_and(|r| r.is_configured()), "Mixamo names map to a humanoid");
        let asset = to_asset(&skeleton, &source.animations[0], 30.0);
        assert!(asset.humanoid.is_some());
        let avatar = zestengine_assets::load_fbx_with(&tpose, &off()).unwrap();
        let mut avatar_skel = avatar.skeletons[0].clone();
        crate::humanoid::ensure_rigs(std::slice::from_mut(&mut avatar_skel));
        let to_world = avatar.instances[0].transform;
        let rig = Rig::from_skeleton(&avatar_skel);
        assert!(rig.muscle.is_some());
        let mut worst = 0.0f32;
        for t in [0.0f32, 1.3, 2.2, 3.9] {
            let reference = zestengine_assets::reference_bone_positions(&idle, t as f64).unwrap();
            let (gap, n) = worst_gap(&avatar_positions(&avatar_skel, to_world, &asset, &rig, t), &reference);
            assert!(n >= 50);
            worst = worst.max(gap);
        }
        eprintln!("skinless clip through muscles: worst bone gap {:.1} mm ({:?})", worst * 1000.0, started.elapsed());
        assert!(worst < 0.08, "worst gap {worst} m");
    }

    /// Humanoid Oversampling multiplies the muscle bake rate (1, 2, 4, 8 times 30 Hz) and the key count with it.
    #[test]
    fn humanoid_oversampling_multiplies_the_muscle_bake_rate() {
        let Some(idle) = mixamo("Zombie Idle.fbx") else {
            eprintln!("skipped: Mixamo files not present");
            return;
        };
        let source = zestengine_assets::load_fbx_with(&idle, &off()).unwrap();
        let mut skeleton = source.skeletons[0].clone();
        crate::humanoid::ensure_rigs(std::slice::from_mut(&mut skeleton));
        let mut rows = Vec::new();
        for n in [1u8, 2, 4, 8] {
            skeleton.rig.as_mut().unwrap().humanoid_oversampling = n;
            let h = *to_asset(&skeleton, &source.animations[0], 30.0).humanoid.expect("a humanoid twin");
            rows.push((h.frame_rate, h.tracks.iter().map(|t| t.keys.len()).max().unwrap_or(0)));
        }
        eprintln!("oversampling 1/2/4/8x: (rate, max keys) {rows:?}");
        assert_eq!(rows.iter().map(|r| r.0 as u32).collect::<Vec<_>>(), [30, 60, 120, 240]);
        for w in rows.windows(2) {
            assert!(w[1].1 as f32 > w[0].1 as f32 * 1.8, "{rows:?}");
        }
    }
}
