//! Play-time glue for avatar parameters: fills the built-in parameters from
//! the engine, pushes changed values into the avatar's Animator, and reads
//! back what state-behaviour drivers wrote.
//!
//! Per avatar and frame this is a handful of float writes into the store and a
//! flush of only the changed slots; the name lookups of `absorb_from` run only
//! on frames a state behaviour actually ran.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use glam::{Quat, Vec3};
use zestengine_core::{
    AvatarDescriptor, Animator, AvatarParameters, BuiltinParam, NetworkAvatar, ObjectId, PhysicsQuery as _, SavedParams, VisemeFrame,
};
use zestengine_assets::ObjectSource;

use crate::physics::PhysicsWorld;
use crate::scene::LoadedScene;

/// Smoothing time of the velocity estimate (matches the Animator drivers').
const VELOCITY_SMOOTHING: f32 = 0.08;

/// Seconds after the last change of a saved parameter before it is written.
pub const SAVE_DEBOUNCE: f32 = 1.0;

/// Per-avatar motion memory for the velocity built-ins, plus the saved-parameter
/// bookkeeping (loaded when an avatar becomes active, written back debounced
/// and on unequip / scene change / drop).
#[derive(Default)]
pub struct AvatarRuntime {
    prev: HashMap<ObjectId, Motion>,
    /// Where `avatar_params/<key>.ron` files live; `None` keeps nothing.
    dir: Option<PathBuf>,
    clock: f32,
    frame: u64,
    tracked: HashMap<ObjectId, Tracked>,
}

/// One worn local avatar's saved parameters.
struct Tracked {
    path: PathBuf,
    /// Store slots flagged `saved`, their last seen values, and the store size
    /// they were resolved against (a rebuilt store re-tracks the avatar).
    slots: Vec<usize>,
    names: Vec<String>,
    last: Vec<f32>,
    store_len: usize,
    changed_at: Option<f32>,
    seen: u64,
}

/// Stable file key of an avatar: its root name (file-name safe) and a hash of
/// the name plus the first model path under it, so two avatars never share
/// state. (The scene carries no content id for an avatar root.)
pub fn avatar_key(name: &str, model_path: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in name.bytes().chain(std::iter::once(0)).chain(model_path.bytes()) {
        h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    let safe: String = name.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).take(24).collect();
    format!("{safe}-{h:016x}")
}

/// A change at `changed_at` is written once `SAVE_DEBOUNCE` has passed.
pub fn save_due(changed_at: Option<f32>, now: f32) -> bool {
    changed_at.is_some_and(|t| now - t >= SAVE_DEBOUNCE)
}

/// Apply the file at `path` over `ap`'s defaults (the store must be built).
/// Unknown / removed names, built-ins and unsaved params are ignored; values
/// are coerced and clamped by the store. A missing or corrupt file changes nothing.
pub fn apply_saved(ap: &mut AvatarParameters, path: &Path) {
    ap.store.load_saved(&SavedParams::load(path));
}

/// Root name plus the first model path found under (or on) the root.
fn identity(scene: &LoadedScene, root: usize) -> String {
    let under = |mut k: usize| loop {
        if k == root {
            return true;
        }
        match scene.objects[k].parent {
            Some(p) => k = p,
            None => return false,
        }
    };
    let model = (0..scene.objects.len())
        .filter(|&k| under(k))
        .find_map(|k| match &scene.objects[k].source {
            ObjectSource::Model { path, .. } => Some(path.as_str()),
            _ => None,
        })
        .unwrap_or("");
    avatar_key(&scene.objects[root].name, model)
}

#[derive(Clone, Copy)]
struct Motion {
    pos: Vec3,
    yaw: f32,
    /// World-space smoothed velocity.
    vel: Vec3,
    /// Smoothed turn rate, radians per second.
    turn: f32,
}

impl Tracked {
    fn write(&self) {
        let mut values = SavedParams::default();
        for (name, &v) in self.names.iter().zip(&self.last) {
            values.values.insert(name.clone(), v);
        }
        if let Err(e) = values.save(&self.path) {
            tracing::warn!("saved avatar parameters not written to {}: {e}", self.path.display());
        }
    }
    /// Final write of an avatar going away: only when something is pending.
    fn flush(&self) {
        if self.changed_at.is_some() {
            self.write();
        }
    }
}

impl Drop for AvatarRuntime {
    /// Session / app exit.
    fn drop(&mut self) {
        self.flush_all();
    }
}

/// The viseme index (0 = silence) and loudness of a lip-sync frame.
pub fn viseme_and_voice(frame: Option<&VisemeFrame>) -> (f32, f32) {
    let Some(f) = frame else {
        return (0.0, 0.0);
    };
    let voice = f.openness();
    if voice < 0.05 {
        return (0.0, voice);
    }
    let best = (1..f.0.len()).max_by(|&a, &b| f.0[a].total_cmp(&f.0[b])).unwrap_or(0);
    (best as f32, voice)
}

/// World velocity as seen from an avatar facing `rot`: X right, Y up, Z forward
/// (forward is the avatar's -Z axis, like the Animator's Forward speed driver).
pub fn local_velocity(rot: Quat, world: Vec3) -> Vec3 {
    let l = rot.inverse() * world;
    Vec3::new(l.x, l.y, -l.z)
}

/// Heading as a rotation about +Y (positive = turning left, seen from above).
fn yaw_of(rot: Quat) -> f32 {
    let fwd = rot * Vec3::NEG_Z;
    (-fwd.x).atan2(-fwd.z)
}

impl AvatarRuntime {
    /// Where saved parameters live (`<dir>/avatar_params/`). Flushes what is
    /// pending to the old place first.
    pub fn set_dir(&mut self, dir: PathBuf) {
        self.flush_all();
        self.dir = Some(dir);
    }

    /// Write every pending change and forget the avatars (scene change).
    pub fn reset(&mut self) {
        self.flush_all();
        self.prev.clear();
    }

    fn flush_all(&mut self) {
        for (_, t) in self.tracked.drain() {
            t.flush();
        }
    }

    /// Load on first sight, note changes, write when the debounce is up.
    fn persist(&mut self, id: ObjectId, key: Option<String>, ap: &mut AvatarParameters) {
        let Some(dir) = self.dir.as_ref() else {
            return;
        };
        if self.tracked.get(&id).is_some_and(|t| t.store_len != ap.store.len()) {
            // The store was rebuilt (defaults are back): save what was pending, load again.
            if let Some(t) = self.tracked.remove(&id) {
                t.flush();
            }
        }
        if !self.tracked.contains_key(&id) {
            let Some(key) = key else {
                return;
            };
            let path = SavedParams::path_for(dir, &key);
            apply_saved(ap, &path);
            let slots = ap.store.saved_slots();
            let last = slots.iter().map(|&s| ap.store.get(s)).collect();
            let names = slots.iter().map(|&s| ap.store.name(s).to_owned()).collect();
            self.tracked.insert(id, Tracked { path, slots, names, last, store_len: ap.store.len(), changed_at: None, seen: self.frame });
            return;
        }
        let Some(t) = self.tracked.get_mut(&id) else {
            return;
        };
        t.seen = self.frame;
        for (k, &slot) in t.slots.iter().enumerate() {
            let v = ap.store.get(slot);
            if v != t.last[k] {
                t.last[k] = v;
                t.changed_at = Some(self.clock);
            }
        }
        if save_due(t.changed_at, self.clock) {
            t.changed_at = None;
            t.write();
        }
    }

    pub fn drive(&mut self, scene: &mut LoadedScene, physics: &PhysicsWorld, dt: f32) {
        self.clock += dt;
        self.frame += 1;
        let roots: Vec<usize> = (0..scene.objects.len())
            .filter(|&i| scene.objects[i].components.iter().any(|c| c.as_any().is::<AvatarParameters>()))
            .collect();
        for i in roots {
            if !scene.is_active(i) {
                continue;
            }
            let id = scene.objects[i].id;
            let remote = std::iter::successors(Some(i), |&k| scene.objects[k].parent)
                .find_map(|k| scene.objects[k].components.iter().find_map(|c| c.as_any().downcast_ref::<NetworkAvatar>()))
                .is_some_and(|n| n.remote);
            let grounded = std::iter::successors(Some(i), |&k| scene.objects[k].parent).any(|k| physics.grounded(scene.objects[k].id));
            let (visemes, eye_height) = (
                scene.visemes_for(id, remote),
                scene.objects[i].components.iter().find_map(|c| c.as_any().downcast_ref::<AvatarDescriptor>()).map(|d| d.eye_height()),
            );
            let (world_rot, pos) = {
                let (_, r, t) = scene.world_transform(i).to_scale_rotation_translation();
                (r, t)
            };
            let yaw = yaw_of(world_rot);
            let m = match self.prev.get(&id) {
                Some(p) if dt > 1e-6 => {
                    let k = 1.0 - (-dt / VELOCITY_SMOOTHING).exp();
                    let mut dyaw = yaw - p.yaw;
                    dyaw = (dyaw + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
                    Motion { pos, yaw, vel: p.vel.lerp((pos - p.pos) / dt, k), turn: p.turn + (dyaw / dt - p.turn) * k }
                }
                Some(p) => Motion { pos, yaw, ..*p },
                None => Motion { pos, yaw, vel: Vec3::ZERO, turn: 0.0 },
            };
            self.prev.insert(id, m);

            let hands = scene.objects.iter().enumerate().find_map(|(k, object)| {
                if !scene.is_active(k) || !std::iter::successors(Some(k), |&p| scene.objects[p].parent).any(|p| p == i) { return None; }
                object.components.iter().find_map(|c| c.as_any().downcast_ref::<zestengine_core::AvatarHands>())
                    .map(|h| (h.selected, h.weight.clamp(0.0, 1.0)))
            });

            // Only the wearer's own avatar keeps saved parameters.
            let key = (!remote && self.dir.is_some() && !self.tracked.contains_key(&id)).then(|| identity(scene, i));
            let Some(ap) = scene.objects[i].components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<AvatarParameters>()) else {
                continue;
            };
            if !ap.ready {
                ap.rebuild_store();
            }
            if !remote {
                self.persist(id, key, ap);
            }
            let local = local_velocity(world_rot, m.vel);
            let (viseme, voice) = viseme_and_voice(visemes.as_ref());
            let s = &mut ap.store;
            s.set_builtin(BuiltinParam::IsLocal, if remote { 0.0 } else { 1.0 });
            s.set_builtin(BuiltinParam::Grounded, grounded as u8 as f32);
            s.set_builtin(BuiltinParam::VelocityX, local.x);
            s.set_builtin(BuiltinParam::VelocityY, local.y);
            s.set_builtin(BuiltinParam::VelocityZ, local.z);
            s.set_builtin(BuiltinParam::VelocityMagnitude, local.length());
            s.set_builtin(BuiltinParam::AngularY, m.turn);
            s.set_builtin(BuiltinParam::Viseme, viseme);
            s.set_builtin(BuiltinParam::Voice, voice);
            if let Some((poses, weight)) = hands {
                s.set_builtin(BuiltinParam::GestureLeft, poses[0] as u8 as f32);
                s.set_builtin(BuiltinParam::GestureRight, poses[1] as u8 as f32);
                s.set_builtin(BuiltinParam::GestureLeftWeight, weight);
                s.set_builtin(BuiltinParam::GestureRightWeight, weight);
            }
            if let Some(h) = eye_height {
                s.set_builtin(BuiltinParam::EyeHeightAsMeters, h);
            }
            self.sync_animator(scene, i);
        }
        // Avatars that were unequipped, hidden or removed: write and let go.
        let frame = self.frame;
        let gone: Vec<ObjectId> = self.tracked.iter().filter(|(_, t)| t.seen != frame).map(|(&k, _)| k).collect();
        for id in gone {
            if let Some(t) = self.tracked.remove(&id) {
                t.flush();
            }
        }
        if self.prev.len() > 64 {
            // Avatars that left: keep memory bounded.
            let live: std::collections::HashSet<ObjectId> = scene.objects.iter().map(|o| o.id).collect();
            self.prev.retain(|id, _| live.contains(id));
        }
    }

    /// The avatar's Animator is the one on its root or, failing that, the
    /// nearest one below it.
    fn sync_animator(&self, scene: &mut LoadedScene, root: usize) {
        let target = if scene.objects[root].components.iter().any(|c| c.as_any().is::<Animator>()) {
            Some(root)
        } else {
            (0..scene.objects.len()).find(|&k| {
                k != root
                    && scene.objects[k].components.iter().any(|c| c.as_any().is::<Animator>())
                    && std::iter::successors(scene.objects[k].parent, |&p| scene.objects[p].parent).any(|p| p == root)
            })
        };
        let Some(t) = target else {
            return;
        };
        // The Animator may be on the same object: lift the store out for the swap.
        let Some(mut store) = scene.objects[root]
            .components
            .iter_mut()
            .find_map(|c| c.as_any_mut().downcast_mut::<AvatarParameters>())
            .map(|ap| std::mem::replace(&mut ap.store, zestengine_core::AvatarParamStore::placeholder()))
        else {
            return;
        };
        if let Some(a) = scene.objects[t].components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<Animator>()) {
            if std::mem::take(&mut a.state.params_touched) {
                store.absorb_from(&a.state.params);
            }
            store.flush_to(&mut a.state.params);
        }
        if let Some(ap) = scene.objects[root].components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<AvatarParameters>()) {
            ap.store = store;
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn hand_poses_feed_existing_avatar_gesture_parameters() {
        use zestengine_core::{AvatarHands, AvatarParameters, HandPose, ObjectId};
        let mut scene = crate::scene::LoadedScene::empty();
        let mut object = crate::scene::SceneObject::new_empty(ObjectId::new(), "Worn avatar");
        let mut hands = AvatarHands::default();
        hands.advance([HandPose::Peace, HandPose::Rock], 0.0);
        object.components.extend([Box::new(AvatarParameters::default()) as Box<dyn zestengine_core::Component>, Box::new(hands)]);
        scene.objects.push(object);
        let physics = crate::physics::PhysicsWorld::build(&scene);
        super::AvatarRuntime::default().drive(&mut scene, &physics, 1.0 / 90.0);
        let store = &scene.objects[0].components[0].as_any().downcast_ref::<AvatarParameters>().unwrap().store;
        assert_eq!(store.get_builtin(zestengine_core::BuiltinParam::GestureLeft), HandPose::Peace as u8 as f32);
        assert_eq!(store.get_builtin(zestengine_core::BuiltinParam::GestureRight), HandPose::Rock as u8 as f32);
        assert_eq!(store.get_builtin(zestengine_core::BuiltinParam::GestureLeftWeight), 1.0);
    }

    use super::*;

    #[test]
    fn silence_and_missing_input_read_zero() {
        assert_eq!(viseme_and_voice(None), (0.0, 0.0));
        assert_eq!(viseme_and_voice(Some(&VisemeFrame::default())), (0.0, 0.0));
        let mut quiet = VisemeFrame::default();
        quiet.0[10] = 0.01;
        assert_eq!(viseme_and_voice(Some(&quiet)).0, 0.0, "below the noise floor stays silent");
    }

    #[test]
    fn loudest_viseme_wins() {
        let mut f = VisemeFrame::default();
        f.0[0] = 0.9;
        f.0[10] = 0.5;
        f.0[13] = 0.2;
        let (v, voice) = viseme_and_voice(Some(&f));
        assert_eq!(v, 10.0, "'sil' never wins while the mouth is open");
        assert!((voice - 0.7).abs() < 1e-6);
    }

    #[test]
    fn velocity_is_in_the_avatars_frame() {
        // Facing -X (turned 90 degrees left): walking along -X is walking forward.
        let rot = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let v = local_velocity(rot, Vec3::new(-2.0, 0.5, 0.0));
        assert!((v - Vec3::new(0.0, 0.5, 2.0)).length() < 1e-5, "{v:?}");
        // Unrotated: +X is right, -Z is forward.
        let v = local_velocity(Quat::IDENTITY, Vec3::new(1.0, 0.0, -3.0));
        assert_eq!(v, Vec3::new(1.0, 0.0, 3.0));
    }

    #[test]
    fn yaw_follows_the_facing_direction() {
        assert!(yaw_of(Quat::IDENTITY).abs() < 1e-6);
        assert!((yaw_of(Quat::from_rotation_y(0.5)) - 0.5).abs() < 1e-5);
    }

    use zestengine_core::{AvatarParam, AvatarParamKind, AvatarParamSync};

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("zest-avrt-{tag}-{}", ObjectId::new().to_string().replace('-', "")));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn avatar(params: &[(&str, AvatarParamKind, f32, bool, AvatarParamSync)]) -> AvatarParameters {
        let mut ap = AvatarParameters::default();
        for &(n, k, d, saved, sync) in params {
            ap.params.push(AvatarParam { name: n.into(), kind: k, default: d, saved, sync, ..Default::default() });
        }
        ap.rebuild_store();
        ap
    }

    use AvatarParamKind::*;
    use AvatarParamSync::*;

    fn slot(ap: &AvatarParameters, n: &str) -> usize {
        ap.store.index_of(n).unwrap()
    }

    #[test]
    fn saved_values_apply_over_defaults_and_clamp() {
        let dir = tmp("apply");
        let path = SavedParams::path_for(&dir, "k");
        let mut saved = SavedParams::default();
        for (n, v) in [("Hat", 1.0), ("Gain", 5.0), ("Mode", 300.0), ("Plain", 1.0), ("Gone", 1.0), ("IsLocal", 0.0), ("Loc", 7.5)] {
            saved.values.insert(n.into(), v);
        }
        saved.save(&path).unwrap();
        let mut ap = avatar(&[
            ("Hat", Bool, 0.0, true, Synced),
            ("Gain", Float, 0.25, true, Synced),
            ("Mode", Int, 2.0, true, Synced),
            ("Plain", Bool, 0.0, false, Synced),
            ("Loc", Float, 0.0, true, LocalOnly),
            ("Untouched", Float, 0.5, true, Synced),
        ]);
        apply_saved(&mut ap, &path);
        let g = |n: &str| ap.store.get(slot(&ap, n));
        assert_eq!(g("Hat"), 1.0);
        assert_eq!(g("Gain"), 1.0, "synced float clamps to -1..1");
        assert_eq!(g("Mode"), 255.0, "synced int clamps to 0..255");
        assert_eq!(g("Plain"), 0.0, "unsaved params are not restored");
        assert_eq!(g("Loc"), 7.5, "local-only keeps its full range and persists");
        assert_eq!(g("Untouched"), 0.5, "no saved value keeps the default");
        assert_eq!(ap.store.get_builtin(BuiltinParam::IsLocal), 1.0, "built-ins never load");
        assert!(ap.store.index_of("Gone").is_none());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn corrupt_or_missing_file_keeps_defaults() {
        let dir = tmp("corrupt");
        let path = SavedParams::path_for(&dir, "k");
        let mut ap = avatar(&[("Hat", Bool, 1.0, true, Synced)]);
        apply_saved(&mut ap, &path);
        assert_eq!(ap.store.get(slot(&ap, "Hat")), 1.0);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "((( not ron \u{0}").unwrap();
        apply_saved(&mut ap, &path);
        assert_eq!(ap.store.get(slot(&ap, "Hat")), 1.0);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn debounce_waits_for_a_quiet_second() {
        assert!(!save_due(None, 100.0));
        assert!(!save_due(Some(10.0), 10.5));
        assert!(save_due(Some(10.0), 11.0));
        assert!(save_due(Some(10.0), 50.0));
    }

    #[test]
    fn key_is_stable_distinct_and_file_safe() {
        let a = avatar_key("Fox Girl", "models/fox.glb");
        assert_eq!(a, avatar_key("Fox Girl", "models/fox.glb"));
        assert_ne!(a, avatar_key("Fox Girl", "models/other.glb"));
        assert_ne!(a, avatar_key("Fox Boy", "models/fox.glb"));
        assert!(a.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'));
        let dir = Path::new("/x");
        assert_eq!(SavedParams::path_for(dir, &a), SavedParams::path_for(dir, &a));
        assert!(SavedParams::path_for(dir, &a).ends_with(format!("avatar_params/{a}.ron")));
    }

    #[test]
    fn changes_round_trip_through_the_store_with_debounce_and_flush() {
        let dir = tmp("round");
        let mut rt = AvatarRuntime::default();
        rt.set_dir(dir.clone());
        let id = ObjectId::new();
        let decl = [("Hat", Bool, 0.0, true, Synced), ("Gain", Float, 0.0, true, LocalOnly), ("Tmp", Float, 0.0, false, Synced)];
        let mut ap = avatar(&decl);
        let path = SavedParams::path_for(&dir, "k");
        // First sight: nothing on disk, defaults stay.
        rt.persist(id, Some("k".into()), &mut ap);
        assert!(!path.exists());
        // A change is not written inside the debounce window.
        let (hat, gain, tmp_slot) = (slot(&ap, "Hat"), slot(&ap, "Gain"), slot(&ap, "Tmp"));
        ap.store.set(hat, 1.0);
        ap.store.set(tmp_slot, 0.5);
        rt.clock = 1.0;
        rt.persist(id, None, &mut ap);
        rt.clock = 1.5;
        rt.persist(id, None, &mut ap);
        assert!(!path.exists());
        // A further change restarts the wait; once quiet the file appears.
        ap.store.set(gain, 3.0);
        rt.clock = 1.6;
        rt.persist(id, None, &mut ap);
        rt.clock = 2.4;
        rt.persist(id, None, &mut ap);
        assert!(!path.exists(), "0.8 s after the last change");
        rt.clock = 2.7;
        rt.persist(id, None, &mut ap);
        let on_disk = SavedParams::load(&path);
        assert_eq!(on_disk.values.get("Hat"), Some(&1.0));
        assert_eq!(on_disk.values.get("Gain"), Some(&3.0));
        assert!(!on_disk.values.contains_key("Tmp") && !on_disk.values.contains_key("IsLocal"));
        // Wear it again: a fresh runtime loads the values over the defaults.
        let mut rt2 = AvatarRuntime::default();
        rt2.set_dir(dir.clone());
        let mut ap2 = avatar(&decl);
        rt2.persist(id, Some("k".into()), &mut ap2);
        assert_eq!(ap2.store.get(slot(&ap2, "Hat")), 1.0);
        assert_eq!(ap2.store.get(slot(&ap2, "Gain")), 3.0);
        // A pending change is written on unequip / drop without waiting.
        ap2.store.set(slot(&ap2, "Hat"), 0.0);
        rt2.clock = 0.1;
        rt2.persist(id, None, &mut ap2);
        drop(rt2);
        assert_eq!(SavedParams::load(&path).values.get("Hat"), Some(&0.0));
        drop(rt);
        let _ = std::fs::remove_dir_all(dir);
    }
}
