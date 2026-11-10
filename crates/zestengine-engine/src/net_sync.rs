//! Scene-level replication: the bridge between a [`NetSession`] and the
//! scene. Runs once per frame after the gameplay tick (`LoadedScene::network_sync`):
//!
//! - objects with authority (owner, or the host when unowned) send delta /
//!   quantized transforms at their send rate, parent / active changes,
//!   Animator parameters and avatar poses;
//! - everything else is written from the snapshot buffers (interpolated
//!   behind by a delay that adapts to the sender's jitter and loss, briefly
//!   extrapolated when packets are late);
//! - a stopped object sends rest snapshots, so starved receivers don't drift;
//! - predicted controllers (`NetworkObject::predicted`) are sent by the
//!   host, which simulates them from their owner's inputs, and left alone on
//!   the owner, which predicts them (`PlaySession::net_predict`);
//! - received synced variables and parent changes land on their objects;
//! - networked dynamic bodies another peer simulates are reported back so the
//!   play session makes them kinematic;
//! - the host records networked hitboxes for lag-compensated hit tests;
//! - the host spawns the Network Manager's player prefab for every peer.
//!
//! Host-loop helpers (`route_command`, `config_for`, `auto_start`) and the
//! deterministic spawn helpers used by the play session live here too.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};

use zestengine_core::{
    Animator, BodyKind, BoxCollider, CapsuleCollider, Character, ColliderAxis, ComponentCommand, HitShape,
    InterpolationMode, NetAutoStart, NetSpawnSource, NetworkAnimator, NetworkAvatar, NetworkManager, NetworkObject,
    NetworkVariables, ObjectId, Pawn, RigidBody, SpawnPoint, SphereCollider, Sync as SyncComp,
};
use glam::{Quat, Vec3};
#[cfg(test)]
use glam::Mat4;
use zestengine_core::ecs::{ComponentIndex, Entity, Parent, SystemWorld};

use crate::net::{NetConfig, NetSession, SnapEntry, XformState, sample_avatar, sample_snapshots};
use crate::net_predict::target_delay;
use crate::prefab::Prefab;
use crate::scene::LoadedScene;

/// Seconds between full (non-delta) transform keyframes.
const KEYFRAME: f64 = 1.0;
/// Rest snapshots sent after an object stops (redundant against loss).
const REST_SENDS: u8 = 2;

#[cfg(test)]
fn component<T: zestengine_core::Component>(o: &crate::scene::SceneObject) -> Option<zestengine_core::ecs::ComponentRef<'_, T>> {
    o.components.get_typed::<T>()
}

#[cfg(test)]
fn component_mut<T: zestengine_core::Component>(o: &crate::scene::SceneObject) -> Option<zestengine_core::ecs::ComponentMut<'_, T>> {
    o.components.get_typed_mut::<T>()
}

/// Select only replication participants. These are identities and authoring
/// order; component settings stay borrowed in the permanent ECS columns.
fn replication_objects(world: &SystemWorld<'_>, indices: &HashMap<ObjectId, usize>) -> Vec<(usize, Entity)> {
    fn include<T: zestengine_core::ecs::DataComponent>(world: &SystemWorld<'_>, indices: &HashMap<ObjectId, usize>, objects: &mut HashMap<Entity, usize>) {
        for (entity, (id, _)) in world.query::<(&ObjectId, &T)>().iter() {
            if let Some(&index) = indices.get(id) { objects.insert(entity, index); }
        }
    }
    let mut objects = HashMap::new();
    include::<NetworkObject>(world, indices, &mut objects);
    include::<SyncComp>(world, indices, &mut objects);
    include::<NetworkVariables>(world, indices, &mut objects);
    include::<NetworkAnimator>(world, indices, &mut objects);
    include::<NetworkAvatar>(world, indices, &mut objects);
    let mut objects: Vec<_> = objects.into_iter().map(|(entity, index)| (index, entity)).collect();
    objects.sort_unstable_by_key(|(index, _)| *index);
    objects
}

/// Borrow explicit settings; only legacy participants synthesize defaults.
fn with_replication_config<R>(world: &SystemWorld<'_>, entity: Entity, read: impl FnOnce(&NetworkObject) -> R) -> R {
    if let Some(config) = world.get::<NetworkObject>(entity) { return read(&config); }
    let fallback = world.get::<SyncComp>(entity).map(|sync| NetworkObject::from_sync(sync.smoothing))
        .unwrap_or_else(|| NetworkObject { sync_transform: false, sync_parent: false, sync_active: false, ..Default::default() });
    read(&fallback)
}

fn has_predictable_in(scene: &LoadedScene, world: &SystemWorld<'_>, entity: Entity) -> bool {
    (0..scene.objects.component_count(entity).unwrap_or(0)).filter_map(|slot| scene.objects.component_slot(entity, slot))
        .any(|slot| slot.read_in(world).is_some_and(|value| value.predictable().is_some()))
}

/// Whether an object carries a controller prediction can replay (Pawn,
/// First / Third Person Character, VR Character).
#[cfg(test)]
fn has_predictable(o: &crate::scene::SceneObject) -> bool {
    o.components.iter().any(|c| zestengine_core::is_predictable(c.as_ref()))
}

/// Whether an object is predicted (server-authoritative, owner-predicted).
#[cfg(test)]
#[allow(dead_code)] // prediction checks used by the play-net tests
pub fn is_predicted(o: &crate::scene::SceneObject) -> bool {
    component::<NetworkObject>(o).is_some_and(|n| n.predicted) && has_predictable(o)
}

/// [`is_predicted`] in a session: on a dedicated server a player-owned controller is predicted
/// even when its Network Object does not ask for it ([`NetSession::forces_prediction`]).
#[cfg(test)]
pub fn is_predicted_in(o: &crate::scene::SceneObject, s: &NetSession) -> bool {
    component::<NetworkObject>(o).is_some_and(|n| n.predicted || s.forces_prediction(s.owner_of_raw(o.id().raw()))) && has_predictable(o)
}

/// An object's hit volumes for lag compensation (local space; triggers excluded).
fn hit_shapes(scene: &LoadedScene, world: &SystemWorld<'_>, entity: Entity) -> Vec<HitShape> {
    let mut out = Vec::new();
    use std::any::TypeId;
    for slot in (0..scene.objects.component_count(entity).unwrap_or(0)).filter_map(|slot| scene.objects.component_slot(entity, slot)) {
        let ty = slot.type_id();
        if let Some(cc) = (ty == TypeId::of::<Character>()).then(|| world.get::<Character>(slot.entity())).flatten().filter(|c| c.use_default_capsule) {
            let half = (cc.height * 0.5 - cc.radius).max(0.0);
            let center = Vec3::from(cc.center);
            out.push(HitShape::Capsule { a: center - Vec3::Y * half, b: center + Vec3::Y * half, radius: cc.radius });
        } else if let Some(c) = (ty == TypeId::of::<CapsuleCollider>()).then(|| world.get::<CapsuleCollider>(slot.entity())).flatten().filter(|c| !c.is_trigger) {
            let axis = match c.direction {
                ColliderAxis::X => Vec3::X,
                ColliderAxis::Y => Vec3::Y,
                ColliderAxis::Z => Vec3::Z,
            };
            let half = (c.height * 0.5 - c.radius).max(0.0);
            let center = Vec3::from(c.center);
            out.push(HitShape::Capsule { a: center - axis * half, b: center + axis * half, radius: c.radius });
        } else if let Some(c) = (ty == TypeId::of::<SphereCollider>()).then(|| world.get::<SphereCollider>(slot.entity())).flatten().filter(|c| !c.is_trigger) {
            out.push(HitShape::Sphere { center: Vec3::from(c.center), radius: c.radius });
        } else if let Some(c) = (ty == TypeId::of::<BoxCollider>()).then(|| world.get::<BoxCollider>(slot.entity())).flatten().filter(|c| !c.is_trigger) {
            out.push(HitShape::Box { center: Vec3::from(c.center), half: Vec3::from(c.size) * 0.5 });
        }
    }
    out
}

/// The interpolation delay for an object whose snapshots come from `origin`.
fn interp_delay(s: &mut NetSession, raw: u128, origin: u64, cfg: &NetworkObject, dt: f32) -> f32 {
    let floor = cfg.interp_delay.max(0.0);
    if !cfg.adaptive_delay {
        return floor;
    }
    let target = target_delay(1.0 / cfg.send_rate.max(0.5), s.jitter_of(origin), s.link_loss(origin), floor);
    s.delays.entry(raw).or_default().update(target, dt)
}

/// Replicate the scene for one frame.
pub fn sync(scene: &mut LoadedScene, s: &mut NetSession, dt: f32) {
    if !s.is_connected() {
        return;
    }
    let now = s.now();
    let is_host = s.is_host();
    scene.objects.query(|world, ids| {
        apply_pending(scene, world, s);
        if is_host {
            s.world.modes.clear();
            s.world.var_anyone.clear();
            s.world.relevancy.clear();
            s.world.always_relevant.clear();
            s.world.predicted.clear();
        }
        let mut entries: Vec<SnapEntry> = Vec::new();
        let mut remote_bodies: HashMap<ObjectId, Vec3> = HashMap::new();
        let mut part_writes: Vec<(Entity, Vec3, Quat, Option<Vec3>)> = Vec::new();
        let mut transform_writes: Vec<(Entity, Vec3, Quat, Option<Vec3>)> = Vec::new();
        let mut present: HashSet<u128> = HashSet::new();
        let mut delays: Vec<f32> = Vec::new();
        let mut hits: Vec<(ObjectId, Vec3, Quat, Vec3, Vec<HitShape>)> = Vec::new();

        for (_, entity) in replication_objects(world, ids) {
            let id = world.object_id(entity).expect("replicated object identity");
            // Resolve reflected controllers before mutably borrowing NetworkObject.
            let predictable = has_predictable_in(scene, world, entity);
            if let Some(mut object) = world.get_mut::<NetworkObject>(entity) {
                object.state.connected = true;
                object.state.owner = s.owner_of_raw(id.raw());
                object.state.authority = s.has_authority_raw(id.raw());
            }
            with_replication_config(world, entity, |cfg| {
                let raw = id.raw();
                present.insert(raw);
                let owner = s.owner_of_raw(raw);
                let authority = s.has_authority_raw(raw);
                let predicted = (cfg.predicted || s.forces_prediction(owner)) && predictable;
                // Who sends the transform: its authority, except for a predicted
                // controller, which the host simulates from its owner's input.
                let sender = if predicted { is_host } else { authority };
                let predicting = predicted && authority && !is_host;
                if is_host {
                    s.world.modes.insert(raw, cfg.ownership);
                    if cfg.relevancy > 0.0 {
                        s.world.relevancy.insert(raw, cfg.relevancy);
                    }
                    if cfg.always_relevant {
                        s.world.always_relevant.insert(raw);
                    }
                    if cfg.despawn_on_owner_leave {
                        s.world.despawn_on_leave.insert(raw);
                    }
                    if predicted {
                        s.world.predicted.insert(raw);
                    }
                    if world.get::<NetworkVariables>(entity).is_some_and(|v| v.anyone_can_write) {
                        s.world.var_anyone.insert(raw);
                    }
                    if owner != 0
                        && (predictable || world.get::<NetworkAvatar>(entity).is_some())
                    {
                        s.world.player_obj.entry(owner).or_insert(raw);
                    }
                }
                let (wscale, wrot, wpos) = world.world_transform(entity).expect("object transform").to_scale_rotation_translation();
                if is_host && cfg.sync_transform {
                    let shapes = hit_shapes(scene, world, entity);
                    if !shapes.is_empty() {
                        // Remote-simulated objects: where their owner last said they
                        // were (what the other peers are shown), not our interpolation.
                        let pose = if sender { (wpos, wrot) } else { s.world.xforms.get(&raw).map_or((wpos, wrot), |x| (x.pos, x.rot)) };
                        hits.push((id, pose.0, pose.1, wscale, shapes));
                    }
                }
                let parent_raw = world.get::<Parent>(entity).and_then(|p| p.0).and_then(|p| world.object_id(p)).map_or(0, |id| id.raw());
                let enabled = world.get::<zestengine_core::ecs::Enabled>(entity).is_some_and(|enabled| enabled.0);
                let dynamic = world.get::<RigidBody>(entity).is_some_and(|r| r.kind == BodyKind::Dynamic);
                let mut st = s.send_state.remove(&raw).unwrap_or_default();
                st.seen = true;
                st.missing_since = None;

                if sender && cfg.sync_transform && now - st.last_send >= 1.0 / cfg.send_rate.max(0.5) as f64 {
                    let vel = match st.prev_pos {
                        Some((p, t)) if now > t => (wpos - p) / (now - t) as f32,
                        _ => Vec3::ZERO,
                    };
                    st.prev_pos = Some((wpos, now));
                    let key = now - st.last_key >= KEYFRAME || st.sent.is_none();
                    let sent = st.sent.unwrap_or_default();
                    let pos_changed = key || sent.pos.distance(wpos) > cfg.position_threshold;
                    let rot_changed = key || sent.rot.angle_between(wrot).to_degrees() > cfg.rotation_threshold;
                    let scale_changed = cfg.sync_scale && (key || sent.scale.distance(wscale) > 1e-4);
                    let want_vel = cfg.interpolation == InterpolationMode::Extrapolate || (cfg.sync_rigidbody && dynamic);
                    if pos_changed || rot_changed || scale_changed {
                        entries.push(SnapEntry {
                            id: raw,
                            pos: pos_changed.then_some(wpos),
                            rot: rot_changed.then_some(wrot),
                            scale: scale_changed.then_some(wscale),
                            vel: (want_vel && (pos_changed || key)).then_some(vel),
                            quantized: cfg.quantize,
                        });
                        if pos_changed && sent.pos.distance(wpos) > cfg.position_threshold {
                            st.rest = REST_SENDS;
                        }
                        st.sent = Some(XformState { pos: wpos, rot: wrot, scale: wscale, vel });
                        st.last_send = now;
                        if key {
                            st.last_key = now;
                        }
                        s.world.xforms.insert(raw, XformState { pos: wpos, rot: wrot, scale: wscale, vel });
                    } else if st.rest > 0 {
                        // Stopped: say so explicitly (zero velocity at the same
                        // pose), so receivers extrapolating over a gap stop too.
                        st.rest -= 1;
                        entries.push(SnapEntry { id: raw, pos: Some(wpos), rot: Some(wrot), scale: None, vel: Some(Vec3::ZERO), quantized: cfg.quantize });
                        st.last_send = now;
                    }
                }
                if authority {
                    if cfg.sync_parent {
                        if st.parent.is_some_and(|p| p != parent_raw) {
                            st.parent = Some(parent_raw);
                            s.set_parent(raw, parent_raw);
                        } else {
                            st.parent = Some(parent_raw);
                        }
                    }
                    if cfg.sync_active {
                        if st.active.is_some_and(|a| a != enabled) {
                            st.active = Some(enabled);
                            s.set_active(raw, enabled);
                        } else {
                            st.active = Some(enabled);
                        }
                    }
                } else {
                    st.parent = None;
                    st.active = None;
                }
                let origin = s.buffers.get(&raw).map(|b| b.origin).or_else(|| s.avatar_in.get(&raw).map(|b| b.origin));
                let delay = match origin {
                    Some(o) if !sender => interp_delay(s, raw, o, cfg, dt),
                    _ => cfg.interp_delay.max(0.0),
                };
                if !sender {
                    st.sent = None;
                    st.prev_pos = None;
                    if predicting {
                        // The play session predicts and reconciles this one.
                    } else if cfg.sync_transform
                        && let Some(buf) = s.buffers.get(&raw)
                    {
                        delays.push(delay);
                        let rt = s.render_time(buf.origin, delay);
                        if let Some(snap) = sample_snapshots(buf, rt, cfg.interpolation) {
                            transform_writes.push((entity, snap.pos, snap.rot, cfg.sync_scale.then_some(snap.scale)));
                            if dynamic && cfg.sync_rigidbody {
                                remote_bodies.insert(id, snap.vel);
                            }
                        }
                    } else if dynamic && cfg.sync_rigidbody && cfg.sync_transform {
                        remote_bodies.insert(id, Vec3::ZERO);
                    }
                }
                s.send_state.insert(raw, st);

                sync_animator(world, s, entity, raw, authority, now);
                sync_avatar(scene, world, ids, s, entity, raw, authority, now, delay, &mut part_writes);
                crate::avatar_net::sync(world, ids, s, entity, raw, authority, now, dt);
            });
        }

        if !delays.is_empty() {
            s.view_delay = delays.iter().sum::<f32>() / delays.len() as f32;
        }
        if is_host {
            // Hit history is queried against the session clock (`net_view_time`
            // and `ray_cast_rewound` use `NetView::clock`), so record on the
            // same base. On a migrated host the local clock (`now`) is offset
            // from the session clock; recording on `now` would misalign every
            // rewind across the handover.
            let clock = s.shared_clock().unwrap_or(now);
            let mut h = s.history.lock().unwrap_or_else(|e| e.into_inner());
            for (id, pos, rot, scale, shapes) in &hits {
                h.record(*id, clock, *pos, *rot, *scale, shapes);
            }
            h.prune(clock);
        }
        apply_pose_writes(world, ids, &mut transform_writes);
        apply_pose_writes(world, ids, &mut part_writes);
        s.send_snapshot(&entries);

        // Bodies whose simulation just came back to this peer keep their momentum.
        let reclaimed: HashMap<ObjectId, Vec3> = s
            .remote_bodies
            .iter()
            .filter(|(id, _)| !remote_bodies.contains_key(id) && present.contains(&id.raw()))
            .map(|(id, v)| (*id, *v))
            .collect();
        s.reclaimed_bodies.extend(reclaimed);
        s.remote_bodies = remote_bodies;

        // An owned runtime spawn that vanished locally (plain destroy) despawns everywhere.
        let gone: Vec<u128> = s.send_state.keys().copied().filter(|r| !present.contains(r)).collect();
        for raw in gone {
            let spawned = s.world.spawns.iter().any(|sp| sp.id == raw);
            let authority = s.has_authority_raw(raw);
            let Some(st) = s.send_state.get_mut(&raw) else {
                continue;
            };
            if !st.seen {
                continue;
            }
            if spawned && authority {
                let since = *st.missing_since.get_or_insert(now);
                if now - since > 0.25 {
                    s.send_state.remove(&raw);
                    s.despawn(ObjectId::from_raw(raw));
                }
            } else if !spawned {
                s.send_state.remove(&raw);
            }
        }

        if is_host {
            spawn_players(world, ids, s);
        }
    });
}

/// Write received parents before their children, including scenes whose tree
/// order differs from hierarchy order. All poses resolve current native links.
fn apply_pose_writes(world: &SystemWorld<'_>, indices: &HashMap<ObjectId, usize>, writes: &mut [(Entity, Vec3, Quat, Option<Vec3>)]) {
    writes.sort_by_cached_key(|(entity, _, _, _)| {
        let mut depth = 0;
        let mut parent = world.get::<Parent>(*entity).and_then(|p| p.0);
        while let Some(entity) = parent {
            depth += 1;
            if depth > indices.len() { break; }
            parent = world.get::<Parent>(entity).and_then(|p| p.0);
        }
        let order = world.object_id(*entity).and_then(|id| indices.get(&id).copied()).unwrap_or(usize::MAX);
        (depth, order)
    });
    for &(entity, pos, rot, scale) in writes.iter() { set_world_pose(world, entity, pos, rot, scale); }
}

fn set_world_pose(world: &SystemWorld<'_>, entity: Entity, pos: Vec3, rot: Quat, scale: Option<Vec3>) {
    let parent = world.get::<Parent>(entity).and_then(|p| p.0).and_then(|p| world.world_transform(p));
    let Some(mut transform) = world.get_mut::<zestengine_core::Transform>(entity) else { return };
    match parent {
        Some(pw) => {
            let (ps, pr, _) = pw.to_scale_rotation_translation();
            transform.translation = pw.inverse().transform_point3(pos);
            transform.rotation = (pr.inverse() * rot).normalize();
            if let Some(s) = scale { transform.scale = s / ps.max(Vec3::splat(1e-6)); }
        }
        None => {
            transform.translation = pos;
            transform.rotation = rot;
            if let Some(s) = scale { transform.scale = s; }
        }
    }
}

fn apply_pending(scene: &LoadedScene, world: &SystemWorld<'_>, s: &mut NetSession) {
    s.pending_vars.retain(|p| {
        let Some(entity) = world.entity(ObjectId::from_raw(p.id)) else { return true };
        if let Some(mut vars) = world.get_mut::<NetworkVariables>(entity) { vars.set(&p.key, p.value.clone()); }
        false
    });
    let mut parents = std::mem::take(&mut s.pending_parents);
    parents.retain(|(id, parent, _)| {
        let Some(entity) = world.entity(ObjectId::from_raw(*id)) else { return true };
        let new_parent = if *parent == 0 { None } else {
            match world.entity(ObjectId::from_raw(*parent)) { Some(parent) => Some(parent), None => return true }
        };
        let mut ancestor = new_parent;
        let mut count = 0;
        while let Some(parent) = ancestor {
            if parent == entity || count > scene.objects.len() { return false; }
            ancestor = world.get::<Parent>(parent).and_then(|p| p.0);
            count += 1;
        }
        let Some(matrix) = world.world_transform(entity) else { return false };
        // This helper also clears any outstanding authoring parent request.
        let index = scene.objects.index_of(ObjectId::from_raw(*id)).expect("indexed scene object");
        scene.objects[index].set_parent_in(world, new_parent);
        let (scale, rot, pos) = matrix.to_scale_rotation_translation();
        set_world_pose(world, entity, pos, rot, Some(scale));
        false
    });
    s.pending_parents.extend(parents);
    for (id, pos, rot) in std::mem::take(&mut s.corrections) {
        if let Some(entity) = world.entity(ObjectId::from_raw(id)) {
            set_world_pose(world, entity, pos, rot, None);
            s.send_state.remove(&id);
            s.reclaimed_bodies.insert(ObjectId::from_raw(id), Vec3::ZERO);
        }
    }
}

fn sync_animator(world: &SystemWorld<'_>, s: &mut NetSession, entity: Entity, raw: u128, authority: bool, now: f64) {
    let Some(na) = world.get::<NetworkAnimator>(entity) else {
        return;
    };
    let Some(mut anim) = world.get_mut::<Animator>(entity) else {
        return;
    };
    if authority {
        let st = s.send_state.entry(raw).or_default();
        if now - st.last_anim < 1.0 / na.send_rate.max(0.5) as f64 {
            return;
        }
        st.last_anim = now;
        let full = now - st.last_anim_full >= KEYFRAME;
        if full {
            st.last_anim_full = now;
        }
        let mut changed = Vec::new();
        for (k, v) in &anim.state.params {
            if !na.parameters.is_empty() && !na.parameters.iter().any(|p| p == k) {
                continue;
            }
            let prev = st.anim.get(k).copied();
            if full || prev.is_none_or(|p| (p - v).abs() > 1e-4) {
                changed.push((k.clone(), *v));
                st.anim.insert(k.clone(), *v);
            }
        }
        s.send_anim(raw, &changed);
    } else if let Some(params) = s.anim_in.get(&raw) {
        for (k, v) in params {
            anim.set_float(k, *v);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn sync_avatar(
    scene: &LoadedScene,
    world: &SystemWorld<'_>,
    ids: &HashMap<ObjectId, usize>,
    s: &mut NetSession,
    entity: Entity,
    raw: u128,
    authority: bool,
    now: f64,
    delay: f32,
    part_writes: &mut Vec<(Entity, Vec3, Quat, Option<Vec3>)>,
) {
    let Some(mut av) = world.get_mut::<NetworkAvatar>(entity) else {
        return;
    };
    let parts = av.parts().map(|reference| world.entity(reference.id()?));
    let rate = av.send_rate;
    let tracked = av.drive_from_vr || av.follow_local_player;
    let scene_face = scene.local_face();
    let body = scene.local_body;
    let poses: [Option<(Vec3, Quat)>; zestengine_core::TRACKER_SLOTS] = if authority {
        // Every tracker the local player has (calibrated bone targets), else
        // the part objects' poses.
        let mut poses = if tracked { scene.local_tracking_world_in(world, ids).map(|t| t.slots()).unwrap_or_default() } else { Default::default() };
        for (k, p) in parts.iter().enumerate() {
            if let (None, Some(pi)) = (poses[k], p) {
                let (_, r, t) = world.world_transform(*pi).expect("avatar part transform").to_scale_rotation_translation();
                poses[k] = Some((t, r));
            }
        }
        let st = s.send_state.entry(raw).or_default();
        let due = now - st.last_avatar >= 1.0 / rate.max(0.5) as f64;
        if due {
            st.last_avatar = now;
            s.send_avatar(raw, &poses);
            if tracked && let Some(face) = scene.local_face() {
                s.send_face(raw, face);
            }
        }
        poses
    } else {
        let Some(buf) = s.avatar_in.get(&raw) else {
            return;
        };
        let rt = s.render_time(buf.origin, delay);
        let Some(poses) = sample_avatar(buf, rt) else {
            return;
        };
        for (k, p) in parts.iter().enumerate() {
            if let (Some(pi), Some((t, r))) = (p, poses[k]) {
                part_writes.push((*pi, t, r, None));
            }
        }
        poses
    };
    {
        av.targets = zestengine_core::TrackerTargets::from_slots(poses);
        av.remote = !authority;
        av.face = if authority {
            if tracked { scene_face.cloned() } else { None }
        } else {
            s.face_in.get(&raw).map(|(_, f)| f.clone())
        };
        // The player's avatar scale and height ride on synced variables too.
        if authority && tracked {
            av.avatar_scale = body.avatar_scale;
            av.player_eye_height = body.eye_height;
            let now = (body.avatar_scale, body.eye_height.unwrap_or(0.0));
            if av.body_sent.is_none_or(|(a, e)| (a - now.0).abs() > 1e-3 || (e - now.1).abs() > 1e-3) {
                s.set_var(ObjectId::from_raw(raw), zestengine_core::AVATAR_SCALE_VAR, zestengine_core::NetValue::Float(now.0));
                s.set_var(ObjectId::from_raw(raw), zestengine_core::PLAYER_EYE_HEIGHT_VAR, zestengine_core::NetValue::Float(now.1));
                av.body_sent = Some(now);
            }
        } else if !authority && let Some(vars) = s.world.vars.get(&raw) {
            let float = |key: &str| {
                vars.iter().find(|(k, _)| k == key).and_then(|(_, v)| match v {
                    zestengine_core::NetValue::Float(f) => Some(*f),
                    _ => None,
                })
            };
            if let Some(scale) = float(zestengine_core::AVATAR_SCALE_VAR).filter(|f| f.is_finite() && *f > 0.0) {
                av.avatar_scale = scale;
            }
            av.player_eye_height = float(zestengine_core::PLAYER_EYE_HEIGHT_VAR).filter(|e| *e > 0.0);
        }
        // The worn avatar package rides on a synced variable (reliable, with
        // late-join catch-up).
        if authority {
            if av.wearing != av.wearing_sent {
                let value = av.wearing.as_ref().map(|w| w.encode()).unwrap_or_default();
                s.set_var(ObjectId::from_raw(raw), zestengine_core::WORN_AVATAR_VAR, zestengine_core::NetValue::Str(value));
                av.wearing_sent = av.wearing.clone();
            }
        } else if let Some(vars) = s.world.vars.get(&raw) {
            let worn = vars.iter().find(|(k, _)| k == zestengine_core::WORN_AVATAR_VAR).and_then(|(_, v)| match v {
                zestengine_core::NetValue::Str(s) => zestengine_core::WornAvatar::decode(s),
                _ => None,
            });
            if worn != av.wearing {
                av.wearing = worn;
            }
        }
    }
}

/// Host: spawn the Network Manager's player prefab for the host and each joiner.
fn spawn_players(world: &SystemWorld<'_>, indices: &HashMap<ObjectId, usize>, s: &mut NetSession) {
    let mut managers = world.query::<(&ObjectId, &NetworkManager)>();
    let manager = managers.iter().filter_map(|(entity, (id, manager))| Some((*indices.get(id)?, entity, manager)))
        .min_by_key(|(order, _, _)| *order);
    let Some((_, manager_entity, manager)) = manager else { s.players_to_spawn.clear(); return };
    if manager.player_prefab.is_empty() { s.players_to_spawn.clear(); s.host_player_done = true; return; }
    let mut todo = std::mem::take(&mut s.players_to_spawn);
    if !s.host_player_done {
        s.host_player_done = true;
        if !manager.is_dedicated() { todo.insert(0, s.local_peer()); }
    }
    let mut points: Vec<_> = world.query::<(&ObjectId, &SpawnPoint)>().iter().filter_map(|(entity, (id, point))| {
        (point.tag == manager.spawn_tag).then(|| indices.get(id).map(|&index| (point.index, index, entity))).flatten()
    }).collect();
    points.sort_unstable_by_key(|(point, index, _)| (*point, *index));
    for peer in todo {
        if s.world.player_obj.get(&peer).is_some_and(|id| s.world.spawns.iter().any(|spawn| spawn.id == *id)) { continue; }
        let at = if points.is_empty() { manager_entity } else { points[s.spawn_counter as usize % points.len()].2 };
        s.spawn_counter += 1;
        let Some(matrix) = world.world_transform(at) else { continue };
        let (_, rot, pos) = matrix.to_scale_rotation_translation();
        let id = s.spawn_everywhere(NetSpawnSource::Prefab(manager.player_prefab.clone()), pos, rot, peer);
        s.world.player_obj.insert(peer, id.raw());
        s.world.despawn_on_leave.insert(id.raw());
    }
}

// ------------------------------------------------------------------ host loop helpers

/// The scene's Network Manager settings as a session config + port.
pub fn config_for(scene: &LoadedScene) -> (NetConfig, u16) {
    let (mut cfg,port)=scene.objects.find_component::<NetworkManager, _>(|_, m, _ecs| Some((NetConfig::from_manager(m), m.port)))
        .unwrap_or_else(|| (NetConfig::default(), 9000));
    cfg.mods=scene.mod_lock.clone();(cfg,port)
}

/// Built games: start hosting / joining once if the scene's Network Manager asks for it.
pub fn auto_start(net: &mut Option<NetSession>, scene: &LoadedScene) {
    static TRIED: AtomicBool = AtomicBool::new(false);
    if net.is_some() || TRIED.load(Ordering::Relaxed) {
        return;
    }
    let Some(m) = scene.objects.find_component::<NetworkManager, _>(|_, m, _ecs| Some(m.clone())) else {
        return;
    };
    TRIED.store(true, Ordering::Relaxed);
    let cfg = config_for(scene).0;
    let result = match m.auto_start {
        NetAutoStart::Off => return,
        NetAutoStart::Host => NetSession::host_with(m.port, cfg),
        NetAutoStart::Join => NetSession::join_with(&m.address, cfg),
    };
    match result {
        Ok(s) => *net = Some(s),
        Err(e) => tracing::error!("network auto-start failed: {e}"),
    }
}

/// Apply a networking command from gameplay. Returns the command back when it
/// isn't a networking one (or, offline, the local fallback it maps to).
pub fn route_command(net: &mut Option<NetSession>, scene: &LoadedScene, cmd: ComponentCommand) -> Option<ComponentCommand> {
    use ComponentCommand as C;
    match cmd {
        C::RequestOwnership(id) => {
            if let Some(n) = net {
                n.request_ownership(id);
            }
        }
        C::ReleaseOwnership(id) => {
            if let Some(n) = net {
                n.release_ownership(id);
            }
        }
        C::TransferOwnership(id, peer) => {
            if let Some(n) = net {
                n.transfer_ownership(id, peer);
            }
        }
        C::NetMessage { to, text } => {
            if let Some(n) = net {
                n.send_message(to, &text);
            }
        }
        C::NetSpawn { source, root_id, translation, rotation, owner } => {
            if let Some(n) = net {
                n.spawn(source, root_id, translation, rotation, owner);
            }
        }
        C::NetDespawn(id) => {
            if let Some(n) = net {
                n.despawn(id);
            }
        }
        C::NetSetVar { id, key, value } => {
            if let Some(n) = net {
                n.set_var(id, &key, value);
            }
        }
        C::NetRpc { object, name, args, target, reliable } => {
            if let Some(n) = net {
                n.send_rpc(object, &name, &args, target, reliable);
            }
        }
        C::NetHost { port } => {
            let (cfg, default_port) = config_for(scene);
            let port = if port == 0 { default_port } else { port };
            match NetSession::host_with(port, cfg) {
                Ok(s) => *net = Some(s),
                Err(e) => tracing::error!("net_host({port}) failed: {e}"),
            }
        }
        C::NetJoin { address } => {
            let (cfg, _) = config_for(scene);
            match NetSession::join_with(&address, cfg) {
                Ok(s) => *net = Some(s),
                Err(e) => tracing::error!("net_join({address}) failed: {e}"),
            }
        }
        C::NetDisconnect => {
            if let Some(mut n) = net.take() {
                n.disconnect();
            }
        }
        C::NetKick { peer, reason, ban } => {
            if let Some(n) = net.as_mut().filter(|n| n.is_host()) {
                if ban && let Some(fp) = n.peer_fingerprint(peer) {
                    n.ban_fingerprint(&fp);
                }
                n.kick(peer, &reason);
            }
        }
        C::NetLoadScene(path) => match net {
            Some(n) if n.is_connected() => {
                if n.is_host() {
                    n.load_scene(&path);
                } else {
                    tracing::warn!("net_load_scene({path}) ignored: only the host switches scenes");
                }
            }
            _ => return Some(C::LoadScene(path)),
        },
        other => return Some(other),
    }
    None
}

// ------------------------------------------------------------------ spawn helpers

/// Deterministic id of the `k`-th object of a networked spawn rooted at
/// `root`, so every peer gives the same children the same ids.
pub fn child_id(root: ObjectId, k: usize) -> ObjectId {
    if k == 0 {
        return root;
    }
    let mut x = root.raw() ^ (k as u128).wrapping_mul(0x9E37_79B9_7F4A_7C15_F39C_C060_5CED_C835);
    x ^= x >> 67;
    x = x.wrapping_mul(0x2545_F491_4F6C_DD1D_9E37_79B9_7F4A_7C15);
    x ^= x >> 59;
    ObjectId::from_raw(x | 1)
}

/// Prefab entries for a networked spawn: deterministic ids (root first) and
/// object references inside the prefab remapped to the new ids.
pub fn spawn_entries(prefab: &Prefab, root_id: ObjectId, translation: Vec3, rotation: Quat) -> Vec<zestengine_assets::SceneEntry> {
    let mut entries = prefab.instantiate(translation.to_array());
    let first_root = prefab.entries.iter().position(|e| e.parent.is_none()).unwrap_or(0);
    let mut order: Vec<usize> = (0..entries.len()).collect();
    order.retain(|&k| k != first_root);
    order.insert(0, first_root);
    let mut new_ids = vec![ObjectId::NIL; entries.len()];
    for (slot, &k) in order.iter().enumerate() {
        new_ids[k] = child_id(root_id, slot);
    }
    // `instantiate` already remapped internal references to its own fresh ids.
    let originals: Vec<String> = entries.iter().map(|e| e.id.clone()).collect();
    for (k, e) in entries.iter_mut().enumerate() {
        e.id = new_ids[k].to_string();
        if e.parent.is_none() {
            e.rotation = rotation.to_array();
        }
        for c in &mut e.components {
            for (old, new) in originals.iter().zip(&new_ids) {
                if !old.is_empty() && c.data.contains(old) {
                    c.data = c.data.replace(old, &new.to_string());
                }
            }
        }
    }
    entries
}

/// Give a freshly cloned subtree (root at `root`) deterministic ids.
pub fn reid_subtree(scene: &mut LoadedScene, root: usize, root_id: ObjectId) {
    let mut subtree = vec![root];
    let mut k = 0;
    while k < subtree.len() {
        let p = subtree[k];
        for c in 0..scene.objects.len() {
            if scene.objects[c].parent() == Some(p) && !subtree.contains(&c) {
                subtree.push(c);
            }
        }
        k += 1;
    }
    for (slot, &i) in subtree.iter().enumerate() {
        scene.objects.set_id(i, child_id(root_id, slot)).expect("unique saved object ID");
    }
}

/// Before a frame simulates: take the smoothing offset back off predicted
/// pawns, so they simulate from their true position.
pub fn restore_prediction_offsets(scene: &mut LoadedScene) {
    scene.objects.query(|world, indices| {
        for (entity, (id, object)) in world.query::<(&ObjectId, &mut NetworkObject)>().iter() {
            let offset = std::mem::take(&mut object.state.visual_offset);
            if offset == Vec3::ZERO || !indices.contains_key(id) { continue; }
            let local = world.get::<zestengine_core::ecs::Parent>(entity).and_then(|p| p.0).and_then(|p| world.world_transform(p))
                .map_or(offset, |parent| parent.inverse().transform_vector3(offset));
            world.get_mut::<zestengine_core::Transform>(entity).expect("object transform").translation -= local;
        }
    });
}

/// Networked pawns follow ownership: a remote player's pawn never reads local input.
pub fn unpossess_remote_pawns(scene: &mut LoadedScene, net: &zestengine_core::NetView) {
    if !net.connected { return; }
    scene.objects.query(|world, _| {
        for (_, (id, pawn, object, sync, vars, animator, avatar)) in world.query::<(
            &ObjectId, &mut Pawn, Option<&NetworkObject>, Option<&SyncComp>,
            Option<&NetworkVariables>, Option<&NetworkAnimator>, Option<&NetworkAvatar>,
        )>().iter() {
            let networked = object.is_some() || sync.is_some() || vars.is_some() || animator.is_some() || avatar.is_some();
            if networked && !net.has_authority(*id) { pawn.possessed = false; }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::MemHub;
    use crate::play::{PlaySession, TickInput, World};
    use crate::scene::SceneObject;
    use crate::shaders::ShaderLibrary;
    use zestengine_assets::ObjectSource;
    use zestengine_core::{
        Component, ComponentRegistry, InputState, NetValue, NetView, ObjectRef, ScreenInfo, TypedComponent,
    };
    use serde::{Deserialize, Serialize};
    use std::path::Path;

    const DT: f32 = 1.0 / 60.0;

    /// Host-side gameplay: clones a template for everyone, then sets a synced variable on it.
    #[derive(Serialize, Deserialize, Default)]
    struct TestSpawner {
        template: ObjectRef,
        #[serde(skip)]
        spawned: Option<ObjectId>,
        #[serde(skip)]
        wrote: bool,
    }
    impl TypedComponent for TestSpawner {
        const NAME: &'static str = "Test Net Spawner";
    }

    impl TestSpawner {
        fn update_native(&mut self, ctx: &mut zestengine_core::ecs::ObjectContext<'_, '_>) {
            if !ctx.is_server() {
                return;
            }
            match self.spawned {
                None => self.spawned = ctx.net_clone(self.template, Vec3::new(3.0, 0.0, 0.0), Quat::IDENTITY),
                Some(id) if !self.wrote => self.wrote = ctx.set_net_var(id, "hp", 42),
                _ => {}
            }
        }
    }

    /// Collects RPCs addressed to its object and answers pings.
    #[derive(Serialize, Deserialize, Default)]
    struct RpcProbe {
        #[serde(skip)]
        got: Vec<String>,
        #[serde(skip)]
        sent: bool,
    }
    impl TypedComponent for RpcProbe {
        const NAME: &'static str = "Test Rpc Probe";
    }

    impl RpcProbe {
        fn update_native(&mut self, ctx: &mut zestengine_core::ecs::ObjectContext<'_, '_>) {
            let names: Vec<String> = ctx.rpcs_for_self().map(|r| r.name.clone()).collect();
            self.got.extend(names);
            if ctx.is_client() && !self.sent && ctx.peers().len() > 1 {
                self.sent = true;
                ctx.rpc_self("Ping", vec![NetValue::from(1)], zestengine_core::RpcTarget::All);
            }
        }
    }

    fn registry() -> ComponentRegistry {
        let mut r = ComponentRegistry::with_builtins();
        use zestengine_core::ecs::{Stage, run_ordered};
        r.register_ecs::<TestSpawner>();
        r.register_ecs::<RpcProbe>();
        r.register_ecs::<Glider>();
        r.register_ecs::<ModPrefabSpawner>();
        r.register_system_for::<TestSpawner>("net_test::spawner", Stage::Gameplay, |world, ctx| run_ordered(world, ctx, TestSpawner::update_native));
        r.register_system_for::<RpcProbe>("net_test::rpc", Stage::Gameplay, |world, ctx| run_ordered(world, ctx, RpcProbe::update_native));
        r.register_system_for::<Glider>("net_test::glider", Stage::Update, |world, ctx| run_ordered(world, ctx, Glider::update_native));
        r.register_system_for::<ModPrefabSpawner>("net_test::mod_spawn", Stage::Gameplay, |world, ctx| run_ordered(world, ctx, ModPrefabSpawner::update_native));
        r
    }

    /// A game's own movement component: not one of the built-in controllers,
    /// predicted because it implements `Predictable`.
    #[derive(Default, serde::Serialize, serde::Deserialize)]
    struct Glider {
        possessed: bool,
        #[serde(skip)]
        vel: Vec3,
    }

    impl Glider {
        const SPEED: f32 = 3.0;

        fn advance(&mut self, obj: &mut dyn zestengine_core::MotionContext, input: &InputState, dt: f32) {
            let axis = input.axis2("Move");
            self.vel = Vec3::new(axis.x, 0.0, -axis.y) * Self::SPEED;
            let t = obj.translation() + self.vel * dt;
            obj.set_translation(t);
        }
    }

    impl zestengine_core::TypedComponent for Glider {
        const NAME: &'static str = "Glider";

        fn predictable(&self) -> Option<&dyn zestengine_core::Predictable> {
            Some(self)
        }
        fn predictable_mut(&mut self) -> Option<&mut dyn zestengine_core::Predictable> {
            Some(self)
        }
    }

    impl Glider {
        fn update_native(&mut self, ctx: &mut zestengine_core::ecs::ObjectContext<'_, '_>) {
            if self.possessed {
                let (input, dt) = (ctx.input, ctx.dt);
                self.advance(ctx, input, dt);
            }
        }
    }

    impl zestengine_core::Predictable for Glider {
        fn motion(&self) -> zestengine_core::MotionState {
            zestengine_core::MotionState { vel: self.vel, ..Default::default() }
        }
        fn set_motion(&mut self, m: &zestengine_core::MotionState, _host: &mut dyn zestengine_core::MotionContext) {
            self.vel = m.vel;
        }
        fn frame_extra(&self) -> Vec<f32> {
            Vec::new()
        }
        fn replay(&mut self, obj: &mut dyn zestengine_core::MotionContext, input: &InputState, _extra: &[f32], dt: f32) {
            self.advance(obj, input, dt);
        }
        fn locally_driven(&self) -> bool {
            self.possessed
        }
    }

    fn obj(scene: &mut LoadedScene, id: ObjectId, parent: Option<usize>, enabled: bool, components: Vec<Box<dyn Component>>) -> usize {
        scene.objects.push(SceneObject::from_components(zestengine_core::ecs::ComponentSet::from_vec(components)).with_parent(parent).with_component_roles(Default::default()).with_extra_render(Vec::new()).with_source(ObjectSource::Empty).with_net_role(Default::default()).with_render(None).with_static_geometry(false).with_layer(0).with_id(id).with_name(String::new()).with_enabled(enabled).with_transform(zestengine_core::Transform { translation: Vec3::ZERO, rotation: Quat::IDENTITY, scale: Vec3::ONE }));
        scene.objects.len() - 1
    }

    #[test]
    fn humanoid_ik_picks_network_targets_for_remote_avatars_and_local_ones_otherwise() {
        use zestengine_core::{HumanoidIk, TrackerTargets, VrCharacter};
        let mut s = LoadedScene::empty();
        let rig = obj(&mut s, ObjectId::new(), None, true, vec![Box::new(VrCharacter::default())]);
        s.objects[rig].transform_mut().translation = Vec3::new(5.0, 0.0, 0.0);
        let remote_targets = TrackerTargets { head: Some((Vec3::new(1.0, 1.6, 0.0), Quat::IDENTITY)), ..Default::default() };
        let remote = obj(
            &mut s,
            ObjectId::new(),
            None,
            true,
            vec![
                Box::new(HumanoidIk::default()),
                Box::new(NetworkAvatar { remote: true, targets: remote_targets, ..Default::default() }),
            ],
        );
        s.objects[remote].transform_mut().translation = Vec3::new(1.0, 0.0, 0.0);
        // A worn avatar under the remote player's Network Avatar follows it too.
        let worn = obj(&mut s, ObjectId::new(), Some(remote), true, vec![Box::new(HumanoidIk::default())]);
        let local = obj(&mut s, ObjectId::new(), None, true, vec![Box::new(HumanoidIk::default())]);
        s.objects[local].transform_mut().translation = Vec3::new(5.0, 0.0, 0.0);
        s.set_ik_targets(Some(TrackerTargets { head: Some((Vec3::new(0.0, 1.5, 0.0), Quat::IDENTITY)), ..Default::default() }));
        let world = s.world_transforms();
        let active = s.active_flags();
        let inputs = s.humanoid_ik_inputs(&active, &world);
        let head = |i: usize| inputs[&i].0.head.unwrap().0;
        assert!(head(remote).distance(Vec3::new(0.0, 1.6, 0.0)) < 1e-4, "remote: replicated, in the avatar's space");
        assert!(head(worn).distance(Vec3::new(0.0, 1.6, 0.0)) < 1e-4, "worn avatar of a remote player: replicated");
        assert!(head(local).distance(Vec3::new(0.0, 1.5, 0.0)) < 1e-4, "local: stage space on the VR Character rig");
    }

    struct Ids {
        cube: ObjectId,
        head: ObjectId,
        template: ObjectId,
        pawn: ObjectId,
        probe: ObjectId,
    }

    /// The same scene on every peer (same object ids, like a loaded .scene).
    fn build_scene(ids: &Ids) -> LoadedScene {
        let mut s = LoadedScene::empty();
        let anim = Animator { play_on_start: false, ..Default::default() };
        obj(
            &mut s,
            ids.cube,
            None,
            true,
            vec![Box::new(NetworkObject::default()), Box::new(anim), Box::new(NetworkAnimator::default())],
        );
        let rig = obj(
            &mut s,
            ObjectId::from_raw(ids.head.raw() ^ 1),
            None,
            true,
            vec![Box::new(NetworkAvatar { head: ObjectRef::new(ids.head), drive_from_vr: false, ..Default::default() })],
        );
        obj(&mut s, ids.head, Some(rig), true, vec![]);
        obj(
            &mut s,
            ids.template,
            None,
            false,
            vec![Box::new(NetworkObject::default()), Box::new(NetworkVariables::default())],
        );
        obj(&mut s, ObjectId::new(), None, true, vec![Box::new(TestSpawner { template: ObjectRef::new(ids.template), ..Default::default() })]);
        obj(&mut s, ids.pawn, None, true, vec![Box::new(NetworkObject::default()), Box::new(Pawn::default())]);
        obj(&mut s, ids.probe, None, true, vec![Box::new(NetworkObject::default()), Box::new(RpcProbe::default())]);
        s
    }

    struct Peer {
        net: Option<NetSession>,
        scene: LoadedScene,
        play: PlaySession,
        input: InputState,
    }

    impl Peer {
        fn new(net: NetSession, ids: &Ids) -> Self {
            Self::with_scene(net, build_scene(ids))
        }

        fn with_scene(net: NetSession, scene: LoadedScene) -> Self {
            let play = PlaySession::new(&scene, Path::new("."), None);
            Self { net: Some(net), scene, play, input: InputState::default() }
        }

        /// One frame of the host loop: pump, tick, route commands, predict, replicate.
        fn frame(&mut self, registry: &ComponentRegistry) {
            let net = self.net.as_mut().unwrap();
            net.advance(DT as f64);
            net.pump();
            let view: NetView = net.view();
            let mut shaders = ShaderLibrary::default();
            let screen = ScreenInfo::default();
            let host = self.play.tick(
                &mut World { scene: &mut self.scene, renderer: None, registry, shaders: &mut shaders },
                &TickInput { dt: DT, input: &self.input, net: &view, listener_fallback: Mat4::IDENTITY, screen: &screen },
            );
            for c in host {
                route_command(&mut self.net, &self.scene, c);
            }
            let net = self.net.as_mut().unwrap();
            self.play.net_predict(&mut self.scene, net, &self.input, DT);
            self.scene.network_sync(net, DT);
        }

        fn net(&mut self) -> &mut NetSession {
            self.net.as_mut().unwrap()
        }

        fn index(&self, id: ObjectId) -> usize {
            self.scene.index_of(id).expect("object exists")
        }

        fn pos(&self, id: ObjectId) -> Vec3 {
            self.scene.world_transform(self.index(id)).w_axis.truncate()
        }
    }

    fn run(peers: &mut [&mut Peer], registry: &ComponentRegistry, frames: usize) {
        for _ in 0..frames {
            for p in peers.iter_mut() {
                p.frame(registry);
            }
        }
    }

    fn session(hub: &MemHub, host: Option<&NetSession>, name: &str) -> NetSession {
        let cfg = NetConfig { player_name: name.into(), discovery: false, secret: String::new(), relay: String::new(), ..Default::default() };
        let mut s = match host {
            None => NetSession::host_on(Box::new(hub.endpoint()), cfg),
            Some(h) => NetSession::join_on(Box::new(hub.endpoint()), h.local_addr().unwrap(), cfg),
        };
        s.use_manual_clock();
        s
    }

    fn ids() -> Ids {
        Ids { cube: ObjectId::new(), head: ObjectId::new(), template: ObjectId::new(), pawn: ObjectId::new(), probe: ObjectId::new() }
    }

    #[derive(Default,Serialize,Deserialize)]
    struct ModPrefabSpawner {prefab:String,#[serde(skip)]spawned:Option<ObjectId>,#[serde(skip)]wrote:bool}
    impl TypedComponent for ModPrefabSpawner {
        const NAME:&'static str="Test Mod Prefab Spawner";

    }

    impl ModPrefabSpawner {
        fn update_native(&mut self,ctx:&mut zestengine_core::ecs::ObjectContext<'_, '_>){
            if !ctx.is_server(){return;}
            match self.spawned{None=>self.spawned=Some(ctx.net_spawn_server(&self.prefab,Vec3::new(3.0,0.0,0.0),Quat::IDENTITY)),Some(id)if !self.wrote=>self.wrote=ctx.set_net_var(id,"hp",42),_=>{}}
        }
    }
    #[test]
    fn mod_package_scene_and_prefab_replicate_across_installations_and_late_join(){
        use zestengine_assets::{SceneFile,ComponentData,package::*,mods};
        use zestengine_core::mods::{ModManifest,ModSettings,ModProfile};
        let dir=std::env::temp_dir().join(format!("zest-mod-network-{}",ObjectId::new()));
        std::fs::create_dir_all(dir.join("author")).unwrap();std::fs::create_dir_all(dir.join("game")).unwrap();
        let mut registry=registry();registry.register::<ModPrefabSpawner>();
        let mut scene:SceneFile=ron::from_str(&format!("(entries:[(id:\"{}\",name:\"Mod spawner\",source:Empty,translation:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),scale:(1.0,1.0,1.0))])",ObjectId::new())).unwrap();
        let mut prefab=scene.clone();prefab.entries[0].name="Mod spawn".into();
        prefab.entries[0].components=vec![ComponentData::new("Network Object",ron::to_string(&NetworkObject::default()).unwrap()),ComponentData::new("Network Variables",ron::to_string(&NetworkVariables::default()).unwrap())];
        std::fs::write(dir.join("author/thing.prefab"),ron::to_string(&prefab).unwrap()).unwrap();
        scene.entries[0].components.push(ComponentData::new(ModPrefabSpawner::NAME,ron::to_string(&ModPrefabSpawner{prefab:"thing.prefab".into(),..Default::default()}).unwrap()));
        std::fs::write(dir.join("author/addon.scene"),ron::to_string(&scene).unwrap()).unwrap();
        std::fs::write(dir.join("game/main.scene"),"(entries:[])").unwrap();
        let mut manifest=ModManifest::new("test",1);manifest.additions.insert("main.scene".into(),"addon.scene".into());
        let options=BuildOptions{kind:zestengine_core::ContentKind::World,name:"Mod".into(),author:"test".into(),version:"1".into(),description:String::new(),project_root:dir.join("author"),scene,entry:"addon.scene".into(),thumbnail:None,classify:&|_|ComponentStatus::Builtin,strip_plugin_components:false,include_texture_cache:false,limits:ContentLimits::for_kind(zestengine_core::ContentKind::World)};
        let archive=dir.join("mod.zestenginepak");write_package(&mods::plan_mod(&options,&manifest).unwrap(),&archive).unwrap();
        let settings=ModSettings{game_id:"test".into(),override_paths:vec!["main.scene".into()],..Default::default()};let profile=ModProfile{enabled:vec![manifest.id.clone()]};
        let hub=MemHub::new();let mut mounts=Vec::new();
        let mut make_peer=|host:Option<&NetSession>,name:&str|{
            let library=dir.join(name);mods::install(&archive,&library).unwrap();
            let mount=mods::mount(&dir.join("game"),&library,&settings,&profile,&|_|true).unwrap().unwrap();
            let data=crate::scene_loader::load_on_worker_cpu_for(&mount.root.join("main.scene"),&mount.root,&[],&|_|{},zestengine_core::NetRole::Both).unwrap();
            let (scene,_)=crate::scene_loader::install(data,None,&registry,&mut ShaderLibrary::default(),&mount.root,|_|{}).unwrap();
            let cfg=NetConfig{mods:config_for(&scene).0.mods,player_name:name.into(),discovery:false,secret:String::new(),relay:String::new(),..Default::default()};
            assert_eq!(cfg.mods,mount.lock);
            let mut net=match host{None=>NetSession::host_on(Box::new(hub.endpoint()),cfg),Some(h)=>NetSession::join_on(Box::new(hub.endpoint()),h.local_addr().unwrap(),cfg)};net.use_manual_clock();
            let play=PlaySession::new(&scene,&mount.root,None);mounts.push(mount);
            Peer{net:Some(net),scene,play,input:InputState::default()}
        };
        let mut host=make_peer(None,"host");let mut client=make_peer(host.net.as_ref(),"client");
        assert_eq!(host.scene.objects[0].id(),client.scene.objects[0].id());
        run(&mut [&mut host,&mut client],&registry,40);
        let spawned=component::<ModPrefabSpawner>(&host.scene.objects[0]).unwrap().spawned.unwrap();
        assert_eq!(component::<NetworkVariables>(&client.scene.objects[client.index(spawned)]).unwrap().get("hp"),Some(&NetValue::Int(42)));
        let i=host.index(spawned);host.scene.objects[i].transform_mut().translation=Vec3::new(8.0,1.0,-2.0);
        run(&mut [&mut host,&mut client],&registry,30);
        let mut late=make_peer(host.net.as_ref(),"late");run(&mut [&mut host,&mut client,&mut late],&registry,40);
        assert_eq!(component::<NetworkVariables>(&late.scene.objects[late.index(spawned)]).unwrap().get("hp"),Some(&NetValue::Int(42)));
        assert!(late.pos(spawned).distance(Vec3::new(8.0,1.0,-2.0))<0.01);
        drop((host,client,late));drop(mounts);let _=std::fs::remove_dir_all(dir);
    }

    #[test]
    fn native_replication_keeps_child_world_poses_correct_when_scene_order_is_reversed() {
        let registry = registry();
        let parent = ObjectId::new();
        let child = ObjectId::new();
        let scene = || {
            let mut scene = LoadedScene::empty();
            let config = || Box::new(NetworkObject { send_rate: 60.0, adaptive_delay: false, interp_delay: 0.0, ..Default::default() }) as Box<dyn Component>;
            obj(&mut scene, parent, None, true, vec![config()]);
            scene.objects[0].transform_mut().rotation = Quat::from_rotation_y(0.4);
            scene.objects[0].transform_mut().scale = Vec3::splat(1.5);
            obj(&mut scene, child, Some(0), true, vec![config()]);
            scene.objects[1].transform_mut().translation = Vec3::X;
            scene
        };
        let hub = MemHub::new();
        let hs = session(&hub, None, "Host");
        let cs = session(&hub, Some(&hs), "Client");
        let mut host = Peer::with_scene(hs, scene());
        let mut client = Peer::with_scene(cs, scene());
        client.scene.objects.swap(0, 1);
        run(&mut [&mut host, &mut client], &registry, 20);
        let offset = host.pos(child) - host.pos(parent);
        let mut received = false;
        for frame in 0..60 {
            let row = host.index(parent);
            host.scene.objects[row].transform_mut().translation.x = frame as f32 * 0.1;
            run(&mut [&mut host, &mut client], &registry, 1);
            if client.pos(parent).x > 0.2 {
                received = true;
                assert!((client.pos(child) - client.pos(parent) - offset).length() < 0.01,
                    "received parent and child share the same snapshot despite reversed rows: parent={:?} child={:?}", client.pos(parent), client.pos(child));
            }
        }
        assert!(received);
    }

    #[test]
    fn transform_animator_avatar_converge_and_pawns_unpossess() {
        let registry = registry();
        let ids = ids();
        let hub = MemHub::new();
        let hs = session(&hub, None, "Host");
        let cs = session(&hub, Some(&hs), "Client");
        let mut host = Peer::new(hs, &ids);
        let mut client = Peer::new(cs, &ids);
        run(&mut [&mut host, &mut client], &registry, 10);
        assert!(client.net().is_connected());

        // The host (authority over unowned objects) moves the cube along a path.
        for k in 0..90 {
            let i = host.index(ids.cube);
            host.scene.objects[i].transform_mut().translation = Vec3::new(k as f32 * 0.05, 1.0, 0.0);
            host.scene.objects[i].transform_mut().rotation = Quat::from_rotation_y(k as f32 * 0.02);
            let h = host.index(ids.head);
            host.scene.objects[h].transform_mut().translation = Vec3::new(0.0, 1.7, k as f32 * 0.01);
            run(&mut [&mut host, &mut client], &registry, 1);
            if k == 60 {
                // Mid-motion the client renders ~interp_delay behind, between samples.
                let (hp, cp) = (host.pos(ids.cube), client.pos(ids.cube));
                assert!(cp.x < hp.x && cp.x > hp.x - 0.6, "interpolating behind: host {hp} client {cp}");
            }
        }
        let i = host.index(ids.cube);
        if let Some(mut a) = component_mut::<Animator>(&mut host.scene.objects[i]) {
            a.set_float("Speed", 3.5);
        }
        run(&mut [&mut host, &mut client], &registry, 40);
        let (hp, cp) = (host.pos(ids.cube), client.pos(ids.cube));
        assert!(hp.distance(cp) < 0.002, "converged: host {hp} client {cp}");
        let hr = host.scene.objects[host.index(ids.cube)].rotation();
        let cr = client.scene.objects[client.index(ids.cube)].rotation();
        assert!(hr.angle_between(cr).to_degrees() < 0.5);
        let speed = component::<Animator>(&client.scene.objects[client.index(ids.cube)]).unwrap().get_float("Speed");
        assert_eq!(speed, 3.5, "animator parameter replicated");
        assert!(host.pos(ids.head).distance(client.pos(ids.head)) < 0.002, "avatar part replicated");
        let rig = client.index(ObjectId::from_raw(ids.head.raw() ^ 1));
        assert!(component::<NetworkAvatar>(&client.scene.objects[rig]).unwrap().targets.head.is_some());
        // Full-body trackers without part objects (chest, knees) replicate too.
        let host_rig = host.index(ObjectId::from_raw(ids.head.raw() ^ 1));
        component_mut::<NetworkAvatar>(&mut host.scene.objects[host_rig]).unwrap().drive_from_vr = true;
        let chest = Vec3::new(0.0, 1.3, 0.1);
        host.scene.set_ik_targets(Some(zestengine_core::TrackerTargets {
            chest: Some((chest, Quat::IDENTITY)),
            left_knee: Some((Vec3::new(0.1, 0.5, 0.1), Quat::IDENTITY)),
            ..Default::default()
        }));
        let mut face = zestengine_core::FaceFrame::default();
        face.set_param("JawOpen", 0.5);
        host.scene.set_local_face(Some(face));
        // The owner's avatar fit (scale, measured eye height).
        host.scene.local_body = crate::scene::LocalBody { arm_reach: 1.0, avatar_scale: 1.25, eye_height: Some(1.7) };
        run(&mut [&mut host, &mut client], &registry, 20);
        let av = component::<NetworkAvatar>(&client.scene.objects[rig]).unwrap();
        assert!(av.remote, "the client sees a remote avatar");
        assert_eq!((av.avatar_scale, av.player_eye_height), (1.25, Some(1.7)), "avatar scale and height replicated");
        let jaw = zestengine_core::expression_index("JawOpen").unwrap();
        assert!(av.face.as_ref().is_some_and(|f| (f.weights[jaw] - 0.5).abs() < 0.01), "face replicated");
        assert!(av.targets.chest.is_some_and(|(p, _)| p.distance(chest) < 0.01), "chest tracker replicated");
        assert!(av.targets.left_knee.is_some());
        drop(av);
        assert!(!component::<NetworkAvatar>(&host.scene.objects[host_rig]).unwrap().remote);
        let no = component::<NetworkObject>(&client.scene.objects[client.index(ids.cube)]).unwrap();
        assert!(no.state.connected && !no.state.authority);
        drop(no);

        // The host-authority pawn must not read the client's input.
        assert!(!component::<Pawn>(&client.scene.objects[client.index(ids.pawn)]).unwrap().possessed);
        assert!(component::<Pawn>(&host.scene.objects[host.index(ids.pawn)]).unwrap().possessed);

        // Ownership moves authority: now the client drives and the host follows.
        client.net().request_ownership(ids.cube);
        run(&mut [&mut host, &mut client], &registry, 10);
        let ci = client.index(ids.cube);
        client.scene.objects[ci].transform_mut().translation = Vec3::new(-4.0, 2.0, 1.0);
        run(&mut [&mut host, &mut client], &registry, 40);
        assert!(host.pos(ids.cube).distance(Vec3::new(-4.0, 2.0, 1.0)) < 0.002, "host follows the new owner");
    }

    #[test]
    fn avatar_parameters_replicate_bit_packed_and_a_different_schema_is_rejected() {
        use zestengine_core::{AvatarParam, AvatarParamKind, AvatarParameters};
        let registry = registry();
        let ids = ids();
        let hub = MemHub::new();
        let hs = session(&hub, None, "Host");
        let cs = session(&hub, Some(&hs), "Client");
        let mut host = Peer::new(hs, &ids);
        let mut client = Peer::new(cs, &ids);
        run(&mut [&mut host, &mut client], &registry, 10);
        let rig_id = ObjectId::from_raw(ids.head.raw() ^ 1);
        let declare = |p: &mut Peer, extra: bool| {
            let i = p.index(rig_id);
            let mut params = vec![AvatarParam::new("Hat", AvatarParamKind::Bool), AvatarParam::new("Smile", AvatarParamKind::Float), AvatarParam::new("Outfit", AvatarParamKind::Int)];
            if extra {
                params.push(AvatarParam::new("Extra", AvatarParamKind::Bool));
            }
            let mut ap = AvatarParameters { params, ..Default::default() };
            ap.rebuild_store();
            p.scene.objects[i].components.retain(|c| !c.as_any().is::<AvatarParameters>());
            p.scene.objects[i].components.push(Box::new(ap));
        };
        let set = |p: &mut Peer, name: &str, v: f32| {
            let i = p.index(rig_id);
            let mut ap = component_mut::<AvatarParameters>(&mut p.scene.objects[i]).unwrap();
            let slot = ap.store.index_of(name).unwrap();
            ap.store.set(slot, v);
        };
        let get = |p: &Peer, name: &str| {
            let ap = component::<AvatarParameters>(&p.scene.objects[p.index(rig_id)]).unwrap();
            ap.store.get(ap.store.index_of(name).unwrap())
        };
        declare(&mut host, false);
        declare(&mut client, false);
        set(&mut host, "Hat", 1.0);
        set(&mut host, "Smile", 0.5);
        set(&mut host, "Outfit", 3.0);
        // The avatar is looked up at most once a second, so allow for one rescan.
        run(&mut [&mut host, &mut client], &registry, 100);
        assert_eq!(get(&client, "Hat"), 1.0);
        assert_eq!(get(&client, "Outfit"), 3.0);
        let smile = get(&client, "Smile");
        assert!((smile - 0.5).abs() < 0.01 && smile != 0.5, "the float arrives quantised to a signed byte: {smile}");
        let rig_raw = rig_id.raw();
        let state = &client.net().avparam_state[&rig_raw];
        assert_eq!(state.rejected, 0);
        assert!(state.schema_hash().is_some());

        // Cached roots follow saved identity even when both scene orders change
        // within the one-second scan interval.
        for peer in [&mut host, &mut client] {
            let row = peer.index(rig_id);
            peer.scene.objects.swap(row, 0);
        }
        set(&mut host, "Outfit", 4.0);
        run(&mut [&mut host, &mut client], &registry, 10);
        assert_eq!(get(&client, "Outfit"), 4.0, "cached avatar parameters survive row reordering without waiting for a rescan");

        // A client whose avatar declares something else drops the host's packets.
        declare(&mut client, true);
        set(&mut host, "Hat", 0.0);
        set(&mut client, "Hat", 1.0);
        run(&mut [&mut host, &mut client], &registry, 150);
        assert!(client.net().avparam_state[&rig_raw].rejected > 0, "the schema hash differs");
        assert_eq!(get(&client, "Hat"), 1.0, "a mismatched packet is not applied");
    }

    #[test]
    fn worn_avatar_package_reaches_peers_and_late_joiners() {
        let registry = registry();
        let ids = ids();
        let hub = MemHub::new();
        let hs = session(&hub, None, "Host");
        let cs = session(&hub, Some(&hs), "Client");
        let mut host = Peer::new(hs, &ids);
        let mut client = Peer::new(cs, &ids);
        run(&mut [&mut host, &mut client], &registry, 10);
        let rig_id = ObjectId::from_raw(ids.head.raw() ^ 1);
        let worn = zestengine_core::WornAvatar { id: "abc123".into(), url: "http://example.com/fox.zestenginepak".into() };
        let hr = host.index(rig_id);
        component_mut::<NetworkAvatar>(&mut host.scene.objects[hr]).unwrap().wearing = Some(worn.clone());
        run(&mut [&mut host, &mut client], &registry, 10);
        let cr = client.index(rig_id);
        assert_eq!(component::<NetworkAvatar>(&client.scene.objects[cr]).unwrap().wearing, Some(worn.clone()));

        let ls = session(&hub, Some(host.net.as_ref().unwrap()), "Late");
        let mut late = Peer::new(ls, &ids);
        run(&mut [&mut host, &mut client, &mut late], &registry, 20);
        let li = late.index(rig_id);
        assert_eq!(component::<NetworkAvatar>(&late.scene.objects[li]).unwrap().wearing, Some(worn), "late joiner catches up");
    }

    /// The same networked lever (Toggle) and a Hold plate on every peer.
    fn interact_scene(lever: ObjectId, plate: ObjectId, user: ObjectId) -> LoadedScene {
        use zestengine_core::{InteractMode, Interactable};
        let mut s = LoadedScene::empty();
        let make = |mode| -> Vec<Box<dyn Component>> {
            vec![
                Box::new(NetworkObject::default()),
                Box::new(NetworkVariables::default()),
                Box::new(Interactable { mode, cooldown: 0.0, ..Default::default() }),
            ]
        };
        obj(&mut s, lever, None, true, make(InteractMode::Toggle));
        obj(&mut s, plate, None, true, make(InteractMode::Hold));
        obj(&mut s, user, None, true, vec![]);
        s
    }

    #[test]
    fn a_client_using_a_networked_interactable_asks_the_authority() {
        use zestengine_core::Interactable;
        let registry = registry();
        let (lever, plate, user) = (ObjectId::new(), ObjectId::new(), ObjectId::new());
        let hub = MemHub::new();
        let hs = session(&hub, None, "Host");
        let cs = session(&hub, Some(&hs), "Client");
        let mut host = Peer::with_scene(hs, interact_scene(lever, plate, user));
        let mut client = Peer::with_scene(cs, interact_scene(lever, plate, user));
        run(&mut [&mut host, &mut client], &registry, 30);
        let get = |p: &Peer, id| component::<Interactable>(&p.scene.objects[p.index(id)]).map(|i| (i.on, i.uses, i.held)).unwrap();
        let ask = |p: &mut Peer, id, release: bool| {
            let i = p.index(id);
            let mut it = component_mut::<Interactable>(&mut p.scene.objects[i]).unwrap();
            if release { it.request_release(user) } else { it.request(user) }
        };
        // Use on the client: the host runs it (uses counts there), `on` replicates back.
        ask(&mut client, lever, false);
        run(&mut [&mut host, &mut client], &registry, 30);
        assert_eq!(get(&host, lever), (true, 1, false), "the authority ran the use");
        assert!(get(&client, lever).0, "`on` is mirrored on the client");
        assert_eq!(get(&client, lever).1, 0, "the client did not run it itself");
        ask(&mut client, lever, false);
        run(&mut [&mut host, &mut client], &registry, 30);
        assert!(!get(&host, lever).0);
        assert!(!get(&client, lever).0, "toggled back");
        // Hold: press then release from the client.
        ask(&mut client, plate, false);
        run(&mut [&mut host, &mut client], &registry, 30);
        assert!(get(&host, plate).2, "pressed on the authority");
        ask(&mut client, plate, true);
        run(&mut [&mut host, &mut client], &registry, 30);
        assert!(!get(&host, plate).2, "released on the authority");
        // The host's own use still works locally.
        ask(&mut host, lever, false);
        run(&mut [&mut host, &mut client], &registry, 30);
        assert_eq!((get(&host, lever).0, get(&client, lever).0), (true, true));
    }

    #[test]
    fn spawn_vars_rpc_and_late_join() {
        let registry = registry();
        let ids = ids();
        let hub = MemHub::new();
        let hs = session(&hub, None, "Host");
        let cs = session(&hub, Some(&hs), "A");
        let mut host = Peer::new(hs, &ids);
        let mut a = Peer::new(cs, &ids);
        run(&mut [&mut host, &mut a], &registry, 30);

        // The host's TestSpawner cloned the template for everyone and set hp.
        let spawned = host
            .scene
            .objects
            .iter()
            .find_map(|o| component::<TestSpawner>(o).and_then(|s| s.spawned))
            .expect("host spawned");
        let ai = a.index(spawned);
        assert!(a.scene.objects[ai].enabled());
        assert_eq!(component::<NetworkVariables>(&a.scene.objects[ai]).unwrap().get("hp"), Some(&NetValue::Int(42)));
        assert!(a.pos(spawned).distance(Vec3::new(3.0, 0.0, 0.0)) < 0.01);

        // The client's RPC on the probe reached the host and (All) itself.
        let got = |p: &Peer| component::<RpcProbe>(&p.scene.objects[p.index(ids.probe)]).unwrap().got.clone();
        assert_eq!(got(&host), vec!["Ping"]);
        assert_eq!(got(&a), vec!["Ping"]);

        // Move the spawned object on the host, then a late joiner catches up.
        let hi = host.index(spawned);
        host.scene.objects[hi].transform_mut().translation = Vec3::new(5.0, 1.0, -2.0);
        run(&mut [&mut host, &mut a], &registry, 30);
        let ls = session(&hub, host.net.as_ref(), "Late");
        let mut late = Peer::new(ls, &ids);
        run(&mut [&mut host, &mut a, &mut late], &registry, 40);
        let li = late.index(spawned);
        assert_eq!(component::<NetworkVariables>(&late.scene.objects[li]).unwrap().get("hp"), Some(&NetValue::Int(42)));
        assert!(late.pos(spawned).distance(Vec3::new(5.0, 1.0, -2.0)) < 0.01, "late joiner has the current transform");
        assert_eq!(late.net().peer_count(), 3);

        // Destroying the owned spawn on the host despawns it everywhere.
        let hi = host.index(spawned);
        host.scene.remove_object(hi);
        run(&mut [&mut host, &mut a, &mut late], &registry, 30);
        assert!(a.scene.index_of(spawned).is_none() && late.scene.index_of(spawned).is_none());
    }

    fn walk(forward: f32, crouch: bool) -> InputState {
        let v = zestengine_core::ActionValue { axis2: glam::Vec2::new(0.0, forward), ..Default::default() };
        let mut actions = vec![("Move".to_string(), v)];
        if crouch {
            actions.push(("Crouch".to_string(), zestengine_core::ActionValue { down: true, value: 1.0, ..Default::default() }));
        }
        InputState::from_actions(actions, glam::Vec2::ZERO)
    }

    /// The controller a prediction round trip drives.
    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Mover {
        Pawn,
        FirstPerson,
        ThirdPerson,
        Vr,
        Custom,
    }

    impl Mover {
        fn component(self) -> Box<dyn Component> {
            match self {
                Mover::Pawn => Box::new(Pawn::default()),
                Mover::FirstPerson => Box::new(zestengine_core::FirstPersonCharacter { lock_cursor: false, ..Default::default() }),
                Mover::ThirdPerson => Box::new(zestengine_core::ThirdPersonCharacter { lock_cursor: false, ..Default::default() }),
                Mover::Vr => Box::new(zestengine_core::VrCharacter::default()),
                Mover::Custom => Box::new(Glider::default()),
            }
        }

        fn speed(self) -> f32 {
            match self {
                Mover::Pawn => Pawn::default().move_speed,
                Mover::FirstPerson => zestengine_core::FirstPersonCharacter::default().walk_speed,
                Mover::ThirdPerson => zestengine_core::ThirdPersonCharacter::default().walk_speed,
                Mover::Vr => zestengine_core::VrCharacter::default().move_speed,
                Mover::Custom => Glider::SPEED,
            }
        }
    }

    /// A predicted pawn over a laggy, lossy link: the owner moves at once,
    /// the host simulates it from the owner's inputs, a server-side shove is
    /// reconciled without a visible snap, and both end up in the same place.
    #[test]
    fn predicted_pawn_moves_at_once_and_reconciles_smoothly() {
        predicted_round_trip(Mover::Pawn, false);
    }

    /// The same with a Character walking on a floor collider
    /// (replays rewind the controller body and re-run its moves).
    #[test]
    fn predicted_character_controller_pawn_reconciles() {
        predicted_round_trip(Mover::Pawn, true);
    }

    /// A First Person Character on a Character, crouching for a
    /// while (the capsule resize is rewound and replayed too).
    #[test]
    fn predicted_first_person_controller_reconciles() {
        predicted_round_trip(Mover::FirstPerson, true);
    }

    #[test]
    fn predicted_third_person_controller_reconciles() {
        predicted_round_trip(Mover::ThirdPerson, true);
    }

    /// A VR Character's smooth locomotion (desktop fallback), replayed from the
    /// stick and head pose its input commands carry.
    #[test]
    fn predicted_vr_player_reconciles() {
        predicted_round_trip(Mover::Vr, false);
    }

    #[test]
    fn predicted_vr_character_capsule_reconciles() {
        predicted_round_trip(Mover::Vr, true);
    }

    /// On a listen server the host sees a remote predicted pawn glide even
    /// though its inputs arrive in bursts, and it settles on the true position.
    #[test]
    fn host_smooths_remote_predicted_pawns_through_input_bursts() {
        let registry = registry();
        let pawn_id = ObjectId::new();
        let scene = || {
            let mut s = LoadedScene::empty();
            let net = NetworkObject { predicted: true, ..Default::default() };
            obj(&mut s, pawn_id, None, true, vec![Box::new(net), Mover::Pawn.component()]);
            s
        };
        let hub = MemHub::new();
        let hs = session(&hub, None, "Host");
        let cs = session(&hub, Some(&hs), "Client");
        let mut host = Peer::with_scene(hs, scene());
        let mut client = Peer::with_scene(cs, scene());
        let frame = |host: &mut Peer, client: &mut Peer| {
            hub.advance(DT as f64);
            host.frame(&registry);
            client.frame(&registry);
        };
        for _ in 0..10 {
            frame(&mut host, &mut client);
        }
        client.net().request_ownership(pawn_id);
        for _ in 0..30 {
            frame(&mut host, &mut client);
        }
        let ci = client.index(pawn_id);
        component_mut::<Pawn>(&mut client.scene.objects[ci]).unwrap().possessed = true;
        // Jitter as large as the latency: input commands reach the host in clumps.
        hub.set_latency(0.04, 0.04);
        let step = Mover::Pawn.speed() * DT;
        let mut prev = host.pos(pawn_id);
        let (mut largest, mut lag) = (0.0f32, 0.0f32);
        for k in 0..180 {
            client.input = walk(1.0, false);
            frame(&mut host, &mut client);
            let shown = host.pos(pawn_id);
            if k >= 30 {
                largest = largest.max((shown - prev).length());
                let hi = host.index(pawn_id);
                lag = lag.max(component::<NetworkObject>(&host.scene.objects[hi]).unwrap().state.visual_offset.length());
            }
            prev = shown;
        }
        assert!(largest < step * 1.6, "the host shows an even walk: largest step {largest} m vs {step} m per frame");
        assert!(lag < Mover::Pawn.speed() * 0.25, "smoothing trails by at most a quarter second: {lag} m");
        client.input = InputState::default();
        for _ in 0..90 {
            frame(&mut host, &mut client);
        }
        let hi = host.index(pawn_id);
        let offset = component::<NetworkObject>(&host.scene.objects[hi]).unwrap().state.visual_offset;
        assert!(offset.length() < 1e-3, "settled on the true position: {offset}");
    }

    /// NET-4: on a dedicated server a player's controller is simulated by the server whatever its
    /// Network Object says; on a listen host the flag still decides.
    #[test]
    fn a_dedicated_server_forces_prediction_for_player_pawns() {
        let registry = registry();
        let pawn_id = ObjectId::new();
        let scene = || {
            let mut s = LoadedScene::empty();
            // `predicted: false`: a careless (or hostile-client-friendly) setup.
            let net = NetworkObject { predicted: false, ..Default::default() };
            obj(&mut s, pawn_id, None, true, vec![Box::new(net), Mover::Pawn.component()]);
            s
        };
        let run_with = |topology: zestengine_core::Topology| {
            let hub = MemHub::new();
            let mk = |host: Option<&NetSession>, name: &str| {
                let cfg = NetConfig { player_name: name.into(), discovery: false, secret: String::new(), relay: String::new(), topology, ..Default::default() };
                let mut s = match host {
                    None => NetSession::host_on(Box::new(hub.endpoint()), cfg),
                    Some(h) => NetSession::join_on(Box::new(hub.endpoint()), h.local_addr().unwrap(), cfg),
                };
                s.use_manual_clock();
                s
            };
            let hs = mk(None, "Host");
            let cs = mk(Some(&hs), "Client");
            let mut host = Peer::with_scene(hs, scene());
            let mut client = Peer::with_scene(cs, scene());
            for _ in 0..30 {
                hub.advance(DT as f64);
                host.frame(&registry);
                client.frame(&registry);
            }
            let me = client.net().local_peer();
            host.net().transfer_ownership(pawn_id, me);
            for _ in 0..30 {
                hub.advance(DT as f64);
                host.frame(&registry);
                client.frame(&registry);
            }
            assert_eq!(host.net().owner_of(pawn_id), me);
            let hi = host.index(pawn_id);
            let ci = client.index(pawn_id);
            component_mut::<Pawn>(&mut client.scene.objects[ci]).unwrap().possessed = true;
            for _ in 0..60 {
                client.input = walk(1.0, false);
                hub.advance(DT as f64);
                host.frame(&registry);
                client.frame(&registry);
            }
            let moved = host.pos(pawn_id).distance(Vec3::ZERO);
            let predicted_on_host = host.net().world.predicted.contains(&pawn_id.raw());
            let predicting_on_client = client.net().predictors.contains_key(&pawn_id.raw());
            let _ = hi;
            (predicted_on_host, predicting_on_client, moved, is_predicted_in(&host.scene.objects[host.index(pawn_id)], host.net.as_ref().unwrap()))
        };
        let (host_flag, client_pred, moved, is_pred) = run_with(zestengine_core::Topology::Dedicated);
        assert!(host_flag && client_pred && is_pred, "dedicated: the server simulates it and the owner predicts it");
        assert!(moved > 1.0, "and the pawn walks ({moved} m)");
        let (host_flag, client_pred, _, is_pred) = run_with(zestengine_core::Topology::ListenHost);
        assert!(!host_flag && !client_pred && !is_pred, "listen host: unchanged, the Network Object decides");
    }

    /// A game's own movement component that implements `Predictable` is
    /// predicted and reconciled like the built-in controllers.
    #[test]
    fn predicted_custom_mover_reconciles() {
        predicted_round_trip(Mover::Custom, false);
    }

    fn predicted_round_trip(mover: Mover, physical: bool) {
        let registry = registry();
        let pawn_id = ObjectId::new();
        let floor_id = ObjectId::new();
        let scene = || {
            let mut s = LoadedScene::empty();
            let net = NetworkObject { predicted: true, ..Default::default() };
            let mut comps: Vec<Box<dyn Component>> = vec![Box::new(net), mover.component()];
            if physical {
                comps.push(Box::new(Character::default()));
            }
            if physical || mover == Mover::Vr {
                let floor = obj(&mut s, floor_id, None, true, vec![Box::new(BoxCollider { size: [200.0, 1.0, 200.0], ..Default::default() })]);
                s.objects[floor].transform_mut().translation = Vec3::new(0.0, -0.5, 0.0);
            }
            obj(&mut s, pawn_id, None, true, comps);
            s
        };
        let hub = MemHub::new();
        let hs = session(&hub, None, "Host");
        let cs = session(&hub, Some(&hs), "Client");
        let mut host = Peer::with_scene(hs, scene());
        let mut client = Peer::with_scene(cs, scene());
        let frame = |host: &mut Peer, client: &mut Peer| {
            hub.advance(DT as f64);
            host.frame(&registry);
            client.frame(&registry);
        };
        for _ in 0..10 {
            frame(&mut host, &mut client);
        }
        // 60 ms each way with jitter, 5% loss both ways.
        hub.set_latency(0.06, 0.02);
        hub.set_loss(0.05);
        client.net().request_ownership(pawn_id);
        for _ in 0..60 {
            frame(&mut host, &mut client);
        }
        assert_eq!(host.net().owner_of(pawn_id), 2);
        let ci = client.index(pawn_id);
        if let Some(mut p) = component_mut::<Pawn>(&mut client.scene.objects[ci]) {
            p.possessed = true;
        }
        if let Some(mut g) = component_mut::<Glider>(&mut client.scene.objects[ci]) {
            g.possessed = true;
        }
        let hi = host.index(pawn_id);
        if mover == Mover::Pawn {
            assert!(!component::<Pawn>(&host.scene.objects[hi]).unwrap().possessed, "the host never reads its own input for it");
        }
        if mover == Mover::FirstPerson {
            let fpc = component::<zestengine_core::FirstPersonCharacter>(&host.scene.objects[hi]).unwrap();
            assert!(!fpc.controlling, "the host never drives the owner's controller from its own input");
        }

        let start = client.pos(pawn_id);
        let mut prev = start;
        let mut max_extra = 0.0f32;
        let speed_step = mover.speed() * DT;
        let crouch = |k: usize| mover == Mover::FirstPerson && (40..80).contains(&k);
        let mut crouched_on_host = false;
        for k in 0..240 {
            client.input = walk(if k < 150 { 1.0 } else { 0.0 }, crouch(k));
            if k == 100 {
                // Server-side knockback the client can't predict.
                let hi = host.index(pawn_id);
                { let ecs_transform_rhs = Vec3::new(0.5, 0.0, 0.0); host.scene.objects[hi].transform_mut().translation += ecs_transform_rhs; };
            }
            frame(&mut host, &mut client);
            let visual = client.pos(pawn_id);
            if k == 0 {
                assert!(visual.z < start.z - 1e-4, "moved on the first input frame, before any reply");
            }
            max_extra = max_extra.max((visual - prev).length() - speed_step);
            prev = visual;
            if mover == Mover::FirstPerson {
                let hi = host.index(pawn_id);
                crouched_on_host |= component::<Character>(&host.scene.objects[hi]).is_some_and(|c| c.height < 1.5);
            }
        }
        client.input = InputState::default();
        for _ in 0..90 {
            frame(&mut host, &mut client);
        }
        let ci = client.index(pawn_id);
        let offset = component::<NetworkObject>(&client.scene.objects[ci]).unwrap().state.visual_offset;
        let client_true = client.scene.objects[ci].translation() - offset;
        let host_pos = host.pos(pawn_id);
        assert!(client_true.distance(host_pos) < 1e-3, "{mover:?}: converged: client {client_true} host {host_pos}");
        assert!(offset.length() < 1e-3, "{mover:?}: smoothing finished");
        assert!((host_pos.x - 0.5).abs() < 0.01, "{mover:?}: the shove stuck: {host_pos}");
        assert!(host_pos.y > -0.01, "{mover:?}: stands on the floor, not in it: {host_pos}");
        let walked = -host_pos.z;
        assert!(walked > mover.speed() * 1.5, "{mover:?}: walked a fair way: {host_pos}");
        assert!(max_extra < 0.15, "{mover:?}: no visible snap: largest extra per-frame move {max_extra} m (the shove is 0.5 m)");
        let pred = &client.net().predictors[&pawn_id.raw()];
        assert!(pred.corrections > 0 && pred.corrections < 20, "{mover:?}: {} corrections", pred.corrections);
        if mover == Mover::FirstPerson {
            assert!(crouched_on_host, "the host replayed the crouch");
            let hi = host.index(pawn_id);
            let cc = component::<Character>(&host.scene.objects[hi]).unwrap();
            assert!((cc.height - 1.8).abs() < 1e-4, "stood back up on the host");
        }
    }

    /// A predicted pawn keeps moving through a host crash: its owner predicts
    /// on, the successor takes over the simulation from the owner's state
    /// (no rubber-banding back to where it was when the host died), and
    /// owner and new host end up agreeing.
    #[test]
    fn predicted_pawn_keeps_moving_through_a_host_migration() {
        let registry = registry();
        let pawn_id = ObjectId::new();
        let scene = || {
            let mut s = LoadedScene::empty();
            let net = NetworkObject { predicted: true, ..Default::default() };
            obj(&mut s, pawn_id, None, true, vec![Box::new(net), Box::new(Pawn::default())]);
            s
        };
        let hub = MemHub::new();
        let quick = |name: &str| NetConfig {
            player_name: name.into(),
            discovery: false,
            secret: String::new(),
            relay: String::new(),
            timeout: 2.0,
            migrate_attempt: 1.0,
            identity: crate::net::Identity::generate(),
            ..Default::default()
        };
        let mut hs = NetSession::host_on(Box::new(hub.endpoint()), quick("Host"));
        hs.use_manual_clock();
        let ha = hs.local_addr().unwrap();
        let join = |name: &str| {
            let mut s = NetSession::join_on(Box::new(hub.endpoint()), ha, quick(name));
            s.use_manual_clock();
            s
        };
        let (sa, sb) = (join("A"), join("B"));
        let mut host = Peer::with_scene(hs, scene());
        let mut a = Peer::with_scene(sa, scene());
        let mut b = Peer::with_scene(sb, scene());
        let frame = |peers: &mut [&mut Peer]| {
            hub.advance(DT as f64);
            for p in peers.iter_mut() {
                p.frame(&registry);
            }
        };
        for _ in 0..10 {
            frame(&mut [&mut host, &mut a, &mut b]);
        }
        b.net().request_ownership(pawn_id);
        for _ in 0..30 {
            frame(&mut [&mut host, &mut a, &mut b]);
        }
        let bi = b.index(pawn_id);
        component_mut::<Pawn>(&mut b.scene.objects[bi]).unwrap().possessed = true;
        b.input = walk(1.0, false);
        for _ in 0..60 {
            frame(&mut [&mut host, &mut a, &mut b]);
        }
        host.net().crash();
        hub.cut(ha);
        let at_crash = b.pos(pawn_id);
        let mut prev = at_crash;
        let mut max_step = 0.0f32;
        let step = Pawn::default().move_speed * DT;
        for _ in 0..(5.0 / DT) as usize {
            frame(&mut [&mut a, &mut b]);
            let p = b.pos(pawn_id);
            max_step = max_step.max((p - prev).length());
            prev = p;
        }
        assert!(a.net().is_host() && b.net().host_peer() == 2 && b.net().is_connected());
        assert!(b.pos(pawn_id).z < at_crash.z - Pawn::default().move_speed * 4.0, "kept walking through the migration");
        b.input = InputState::default();
        for _ in 0..90 {
            frame(&mut [&mut a, &mut b]);
            let p = b.pos(pawn_id);
            max_step = max_step.max((p - prev).length());
            prev = p;
        }
        assert!(max_step < step + 0.15, "no snap back: largest step {max_step} m");
        let bi = b.index(pawn_id);
        let offset = component::<NetworkObject>(&b.scene.objects[bi]).unwrap().state.visual_offset;
        let truth = b.scene.objects[bi].translation() - offset;
        let on_new_host = a.pos(pawn_id);
        assert!(truth.distance(on_new_host) < 1e-3, "owner {truth} and new host {on_new_host} agree");
        assert!(b.net().predictors[&pawn_id.raw()].corrections < 20);
    }

    #[test]
    fn deterministic_spawn_ids_and_ref_remap() {
        use zestengine_assets::{ComponentData, MaterialRef, SceneEntry};
        let entry = |id: &str, parent: Option<usize>, data: &str| SceneEntry {
            id: id.into(),
            name: id.into(),
            source: ObjectSource::Empty,
            enabled: true,
            static_geometry: false,
            lightmap_scale: 1.0,
            cast_shadows: zestengine_core::ShadowCasting::On,
            layer: 0,
            material: MaterialRef::File(String::new()),
            extra_materials: vec![],
            renderer_split: false,
            parent,
            components: if data.is_empty() { vec![] } else { vec![ComponentData::new("Network Avatar", data)] },
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
            model: None,
            model_part: None,
            prefab: None,
            prefab_parent: None,
            net_role: Default::default(),
            component_roles: Default::default(),
        };
        let head = "11111111-2222-4333-8444-555555555555";
        let prefab = Prefab::new(vec![entry("aaaaaaaa-2222-4333-8444-555555555555", None, &format!("(head: Some(\"{head}\"))")), entry(head, Some(0), "")]);
        let root = ObjectId::new();
        let a = spawn_entries(&prefab, root, Vec3::X, Quat::IDENTITY);
        let b = spawn_entries(&prefab, root, Vec3::X, Quat::IDENTITY);
        assert_eq!(a[0].id, root.to_string());
        assert_eq!(a[1].id, b[1].id, "children get the same ids on every peer");
        assert_ne!(a[1].id, head);
        assert!(a[0].components[0].data.contains(&a[1].id), "intra-prefab reference remapped");
    }
}

#[cfg(test)]
mod ecs_query_tests {
    use super::*;
    use crate::scene::SceneObject;
    use zestengine_core::{Component, Disabled};

    fn object(components: Vec<Box<dyn Component>>) -> SceneObject {
        let mut object = SceneObject::new_empty(ObjectId::new(), "Object");
        object.components.replace_all(components);
        object
    }

    fn sync(smoothing: f32) -> SyncComp {
        let mut sync = SyncComp::default();
        sync.smoothing = smoothing;
        sync
    }

    fn configs(scene: &LoadedScene) -> Vec<(usize, bool, bool, bool, f32)> {
        scene.objects.query(|world, indices| replication_objects(world, indices).into_iter().map(|(index, entity)|
            with_replication_config(world, entity, |cfg| (index, cfg.sync_transform, cfg.sync_parent, cfg.sync_active, cfg.send_rate))).collect())
    }

    #[test]
    fn replication_queries_keep_precedence_and_ignore_non_networked_or_disabled_data() {
        let mut scene = LoadedScene::empty();
        scene.objects.push(object(vec![Box::new(NetworkVariables::default())]));
        scene.objects.push(object(vec![]));
        scene.objects.push(object(vec![Box::new(sync(0.4)), Box::new(NetworkObject { send_rate: 17.0, ..Default::default() })]));
        scene.objects.push(object(vec![Box::new(Disabled(Box::new(NetworkObject::default())))]));
        scene.objects.push(object(vec![Box::new(sync(0.7))]));
        let values = configs(&scene);
        assert_eq!(values.iter().map(|(i, ..)| *i).collect::<Vec<_>>(), [0, 2, 4]);
        assert!(!values[0].1 && !values[0].2 && !values[0].3);
        assert_eq!(values[1].4, 17.0, "explicit NetworkObject takes precedence over Sync");
        assert!(values[2].1);
        let id = scene.objects[2].id();
        scene.objects.swap(0, 2);
        assert_eq!(scene.index_of(id), Some(0));
        assert_eq!(configs(&scene)[0].4, 17.0);
        scene.objects[0].components.set_enabled(1, false);
        assert_ne!(configs(&scene)[0].4, 17.0, "disabling the explicit config reveals legacy Sync defaults");
    }
}
