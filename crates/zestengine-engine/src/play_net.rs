//! Networking hooks of the play session (a child module of `play` so it can
//! reach the session's physics and prefab cache): applies the world changes
//! the network asks for, hands physics bodies between local simulation and
//! remote control, applies networked spawn / despawn / variable commands
//! locally before they are replicated, and runs client-side prediction and
//! the host's simulation of predicted controllers (`net_predict`).

use zestengine_core::{
    Animator, Character, ComponentCommand, InputState, MotionContext, NetSpawnSource, NetWorldOp,
    NetworkObject, NetworkVariables, ObjectId, PhysicsQuery, Predictable, RayFilter, RayHit, Swimmer, Water,
    as_predictable,
};
use glam::{Mat4, Quat, Vec3};
use zestengine_core::ecs::{ComponentIndex, ComponentSlot, Entity, Parent, SystemWorld};

use super::{PlaySession, World};
use crate::net::NetSession;
use crate::net_predict::{AuthState, InputCmd, MoveState, Replay};
use crate::physics::PhysicsWorld;
use crate::prefab::Prefab;
use crate::scene::LoadedScene;

/// Resolve the first controller from authoring handles, borrowing native data.
fn controller_slot(scene: &LoadedScene, world: &SystemWorld<'_>, entity: Entity) -> Option<ComponentSlot> {
    (0..scene.objects.component_count(entity)?).filter_map(|slot| scene.objects.component_slot(entity, slot))
        .find(|slot| slot.read_in(world).is_some_and(|value| value.predictable().is_some()))
}

fn with_predictable<R>(scene: &LoadedScene, index: usize, read: impl FnOnce(&dyn Predictable) -> R) -> Option<R> {
    let entity = scene.objects[index].entity();
    scene.objects.query(|world, _| {
        let component = controller_slot(scene, world, entity)?.read_in(world)?;
        component.predictable().map(read)
    })
}

fn apply_visual_offset(scene: &LoadedScene, id: ObjectId, offset: Vec3) {
    scene.objects.query(|world, _| {
        let Some(entity) = world.entity(id) else { return };
        let local = world.get::<Parent>(entity).and_then(|p| p.0).and_then(|p| world.world_transform(p))
            .map_or(offset, |parent| parent.inverse().transform_vector3(offset));
        world.get_mut::<zestengine_core::Transform>(entity).expect("object transform").translation += local;
        if let Some(mut object) = world.get_mut::<NetworkObject>(entity) { object.state.visual_offset = offset; }
    });
}

/// Host: seconds over which a remote predicted pawn's display catches up
/// with its simulation after a clump of inputs.
const HOST_SMOOTH_TIME: f32 = 0.08;
/// Host: a display gap larger than this (metres) is a teleport, shown at once.
const HOST_SMOOTH_SNAP: f32 = 2.0;

/// One predicted controller driven outside the component passes: the host's
/// authoritative simulation and the owner's rewind + replay. It is the
/// controller's [`MotionContext`], so it steps exactly like the live update (the
/// same step, the same character controller moves).
struct MoverSim<'a> {
    scene: &'a LoadedScene,
    physics: &'a mut PhysicsWorld,
    index: usize,
    contact: bool,
    water_time: f32,
}

/// Replays movement using native column borrows, including plugin controllers.
/// The outer simulation owns no component lease across a native world scope.
struct NativeMover<'a, 'w> {
    world: &'a SystemWorld<'w>,
    order: &'a std::collections::HashMap<ObjectId, usize>,
    owner: Entity,
    physics: &'a mut PhysicsWorld,
    contact: &'a mut bool,
    water_time: f32,
}

impl<'a> MoverSim<'a> {
    fn new(scene: &'a LoadedScene, physics: &'a mut PhysicsWorld, index: usize, water_time: f32) -> Self {
        let contact = scene.objects.query(|world, _| {
            let entity = scene.objects[index].entity();
            if world.get::<Character>(entity).is_some_and(|c| c.use_default_capsule) {
                physics.grounded(world.object_id(entity).expect("object identity"))
            } else {
                controller_slot(scene, world, entity).and_then(|slot| slot.read_in(world))
                    .and_then(|value| value.predictable().map(|p| p.motion().grounded)).unwrap_or(false)
            }
        });
        Self { scene, physics, index, contact, water_time }
    }

    fn state(&self) -> MoveState {
        self.scene.objects.query(|world, _| {
            let entity = self.scene.objects[self.index].entity();
            let (_, rot, pos) = world.world_transform(entity).expect("object transform").to_scale_rotation_translation();
            let motion = controller_slot(self.scene, world, entity).and_then(|slot| slot.read_in(world))
                .and_then(|value| value.predictable().map(|p| p.motion())).unwrap_or_default();
            MoveState { pos, rot, motion, contact: self.contact }
        })
    }

    fn with_controller(&mut self, f: impl FnOnce(&mut dyn Predictable, &mut NativeMover<'_, '_>)) {
        let scene = self.scene;
        let entity = scene.objects[self.index].entity();
        scene.objects.query(|world, order| {
            let Some(slot) = controller_slot(scene, world, entity) else { return };
            let Some(mut component) = slot.write_in(world) else { return };
            if let Some(controller) = as_predictable(&mut *component) {
                let mut obj = NativeMover { world, order, owner: entity, physics: self.physics, contact: &mut self.contact, water_time: self.water_time };
                f(controller, &mut obj);
            }
        });
    }
}

impl MotionContext for NativeMover<'_, '_> {
    fn self_id(&self) -> ObjectId { self.world.object_id(self.owner).expect("object identity") }
    fn translation(&self) -> Vec3 { self.world.get::<zestengine_core::Transform>(self.owner).expect("object transform").translation }
    fn set_translation(&mut self, t: Vec3) { self.world.get_mut::<zestengine_core::Transform>(self.owner).expect("object transform").translation = t; }
    fn rotation(&self) -> Quat { self.world.get::<zestengine_core::Transform>(self.owner).expect("object transform").rotation }
    fn scale(&self) -> Vec3 { self.world.get::<zestengine_core::Transform>(self.owner).expect("object transform").scale }
    fn set_rotation(&mut self, r: Quat) { self.world.get_mut::<zestengine_core::Transform>(self.owner).expect("object transform").rotation = r; }
    fn parent_world(&self) -> Option<Mat4> { self.world.world_transform(self.world.get::<Parent>(self.owner)?.0?) }
    fn has_character(&self) -> bool { self.world.get::<Character>(self.owner).is_some_and(|c| c.use_default_capsule) }
    fn move_character(&mut self, delta: Vec3) {
        self.physics.move_character_in(self.world, self.owner, delta);
        *self.contact = self.physics.grounded(self.self_id());
    }
    fn grounded(&self) -> bool { *self.contact }
    fn capsule(&self) -> Option<(f32, f32, [f32; 3])> { self.world.get::<Character>(self.owner).map(|c| (c.radius, c.height, c.center)) }
    fn set_capsule(&mut self, height: f32, center: [f32; 3]) {
        if let Some(mut cc) = self.world.get_mut::<Character>(self.owner) { cc.height = height; cc.center = center; }
    }
    fn swimmer(&self) -> Swimmer { self.world.get::<Swimmer>(self.owner).map(|value| *value).unwrap_or_default() }
    fn water_surface(&self, p: Vec3) -> Option<f32> {
        self.world.query::<(&ObjectId, &Water)>().iter().filter_map(|(entity, (id, water))| {
            let order = *self.order.get(id)?;
            water.height_at(self.world.world_transform(entity)?, p, self.water_time).map(|(height, _)| (order, height))
        }).min_by_key(|(order, _)| *order).map(|(_, height)| height)
    }
    fn raycast(&self, origin: Vec3, dir: Vec3, max_dist: f32, filter: &RayFilter) -> Option<RayHit> { self.physics.raycast(origin, dir, max_dist, filter) }
    fn set_animator_bool(&mut self, name: &str, value: bool) {
        if let Some(mut animator) = self.world.get_mut::<Animator>(self.owner) { animator.set_bool(name, value); }
    }
}

impl Replay for MoverSim<'_> {
    fn rewind(&mut self, s: &MoveState) {
        let entity = self.scene.objects[self.index].entity();
        let physical = self.scene.objects.query(|world, _| {
            let (pos, rot) = match world.get::<Parent>(entity).and_then(|p| p.0).and_then(|p| world.world_transform(p)) {
                Some(parent) => {
                    let (_, rotation, _) = parent.to_scale_rotation_translation();
                    (parent.inverse().transform_point3(s.pos), (rotation.inverse() * s.rot).normalize())
                }
                None => (s.pos, s.rot),
            };
            let mut transform = world.get_mut::<zestengine_core::Transform>(entity).expect("object transform");
            transform.translation = pos;
            transform.rotation = rot;
            world.get::<Character>(entity).is_some_and(|c| c.use_default_capsule)
        });
        self.with_controller(|p, obj| p.set_motion(&s.motion, obj));
        self.contact = s.contact;
        if physical { self.physics.place_character(self.scene, self.index, s.contact); }
    }

    fn step(&mut self, cmd: &InputCmd) -> MoveState {
        let input = cmd.input();
        self.with_controller(|p, obj| p.replay(obj, &input, &cmd.extra, cmd.dt));
        self.state()
    }
}

impl PlaySession {
    /// Before the frame runs: remote spawns / despawns / activation, scene
    /// switches, pawn possession and physics hand-over.
    pub(super) fn net_pre_tick(&mut self, w: &mut World, net: &zestengine_core::NetView, host: &mut Vec<ComponentCommand>) {
        crate::net_sync::restore_prediction_offsets(w.scene);
        for op in &net.world_ops {
            match op {
                NetWorldOp::Spawn { source, root_id, translation, rotation, .. } => {
                    self.net_spawn(w, source, *root_id, *translation, *rotation)
                }
                NetWorldOp::Despawn(id) => self.net_remove(w, *id),
                NetWorldOp::SetActive(id, on) => {
                    if let Some(i) = w.scene.index_of(*id)
                        && w.scene.objects[i].enabled() != *on
                    {
                        *w.scene.objects[i].enabled_mut() = *on;
                        self.physics.mark_dirty();
                    }
                }
                NetWorldOp::LoadScene(path) => host.push(ComponentCommand::LoadScene(path.clone())),
            }
        }
        crate::net_sync::unpossess_remote_pawns(w.scene, net);
        if !net.remote_bodies.is_empty() || !net.reclaimed_bodies.is_empty() {
            self.physics.sync_structure(w.scene);
            for id in net.remote_bodies.keys() {
                self.physics.set_net_driven(w.scene, *id, true, Vec3::ZERO);
            }
            for (id, vel) in &net.reclaimed_bodies {
                self.physics.set_net_driven(w.scene, *id, false, *vel);
                // An owner's own prop the server corrected: stopped where it was put back.
                self.physics.set_velocity(*id, *vel);
            }
        }
    }

    /// Networked commands from gameplay: apply locally, then hand to the host
    /// loop for replication.
    pub(super) fn apply_net_command(&mut self, w: &mut World, cmd: ComponentCommand, host: &mut Vec<ComponentCommand>) {
        match &cmd {
            ComponentCommand::NetSpawn { source, root_id, translation, rotation, .. } => {
                self.net_spawn(w, source, *root_id, *translation, *rotation)
            }
            ComponentCommand::NetDespawn(id) => self.net_remove(w, *id),
            ComponentCommand::NetSetVar { id, key, value } => {
                w.scene.objects.query(|world, _| {
                    if let Some(entity) = world.entity(*id) && let Some(mut vars) = world.get_mut::<NetworkVariables>(entity) {
                        vars.set(key, value.clone());
                    }
                });
            }
            _ => {}
        }
        host.push(cmd);
    }

    /// After the frame's tick (before `network_sync`): prediction for the
    /// predicted controllers this peer owns (record the frame's input, send the
    /// unacknowledged inputs, reconcile with the host's newest state, smooth
    /// the correction), and on the host the authoritative simulation of
    /// remote players' predicted controllers from their inputs. `input` and
    /// `frame_dt` are what the tick got.
    pub fn net_predict(&mut self, scene: &mut LoadedScene, net: &mut NetSession, input: &InputState, frame_dt: f32) {
        if !net.is_connected() {
            return;
        }
        let unscaled = frame_dt.clamp(0.0, 0.25);
        let dt = unscaled * self.time_scale;
        let local = net.local_peer();
        let water_time = self.water_time();
        let movers = scene.objects.query(|world, indices| {
            let mut movers: Vec<_> = world.query::<(&ObjectId, &NetworkObject)>().iter().filter_map(|(entity, (id, object))| {
                let &index = indices.get(id)?;
                ((object.predicted || net.forces_prediction(net.owner_of_raw(id.raw()))) && controller_slot(scene, world, entity).is_some())
                    .then_some((index, *id))
            }).collect();
            movers.sort_unstable_by_key(|(index, _)| *index);
            movers
        });
        if movers.is_empty() {
            net.predictors.clear();
            return;
        }
        self.physics.sync_structure(scene);
        // A UI text field had the keyboard: the frame moved nothing.
        let blank = InputState::default();
        let input = if self.typing { &blank } else { input };
        let mut predicting = Vec::new();
        let mut smoothing = Vec::new();
        for (i, id) in movers {
            let raw = id.raw();
            let owner = net.owner_of_raw(raw);
            if net.is_host() {
                if owner == 0 || owner == local {
                    continue;
                }
                let world_pos = |scene: &LoadedScene| scene.world_transform(i).w_axis.truncate();
                // Where this frame starts out showing it (the true position
                // plus last frame's display offset).
                let mut shown = Some(world_pos(scene) + net.host_smoothing.get(&raw).copied().unwrap_or(Vec3::ZERO));
                if let Some(anchor) = net.take_anchor(raw) {
                    // Just took over as host: carry on from where the owner is.
                    MoverSim::new(scene, &mut self.physics, i, water_time).rewind(&anchor.state);
                    shown = None;
                }
                let cmds = net.take_inputs(raw, unscaled);
                if let Some(last) = cmds.last().map(|c| c.seq) {
                    let mut sim = MoverSim::new(scene, &mut self.physics, i, water_time);
                    for c in &cmds {
                        sim.step(c);
                    }
                    let state = sim.state();
                    net.send_auth(raw, owner, &AuthState { ack: last, state });
                }
                // Inputs arrive in clumps: show the pawn easing from where it
                // was shown towards where it now is, instead of stepping.
                let mut offset = shown.map_or(Vec3::ZERO, |s| s - world_pos(scene)) * (-unscaled / HOST_SMOOTH_TIME).exp();
                if offset.length() > HOST_SMOOTH_SNAP || offset.length() < 1e-4 {
                    offset = Vec3::ZERO;
                }
                if offset != Vec3::ZERO {
                    smoothing.push(raw);
                    net.host_smoothing.insert(raw, offset);
                    apply_visual_offset(scene, id, offset);
                }
            } else if owner == local && with_predictable(scene, i, |p| p.locally_driven()).unwrap_or(false) {
                predicting.push(raw);
                let mut pred = net.predictors.remove(&raw).unwrap_or_default();
                let extra = with_predictable(scene, i, |p| p.frame_extra()).unwrap_or_default();
                let mut sim = MoverSim::new(scene, &mut self.physics, i, water_time);
                let after = sim.state();
                pred.record(dt, input, extra, after);
                let cmds: Vec<&InputCmd> = pred.unacked().collect();
                net.send_inputs(raw, &cmds);
                if let Some(auth) = net.take_auth(raw) {
                    pred.reconcile(&auth, &mut sim);
                }
                pred.decay(unscaled);
                let offset = pred.error;
                net.predictors.insert(raw, pred);
                if offset != Vec3::ZERO {
                    apply_visual_offset(scene, id, offset);
                }
            }
        }
        net.predictors.retain(|id, _| predicting.contains(id));
        net.host_smoothing.retain(|id, _| smoothing.contains(id));
    }

    fn net_remove(&mut self, w: &mut World, id: ObjectId) {
        if let Some(i) = w.scene.index_of(id) {
            w.scene.remove_object(i);
            self.physics.mark_dirty();
        }
    }

    /// Instantiate a networked spawn with deterministic ids (same on every peer).
    pub(super) fn net_spawn(&mut self, w: &mut World, source: &NetSpawnSource, root_id: ObjectId, translation: Vec3, rotation: Quat) {
        if w.scene.index_of(root_id).is_some() {
            return;
        }
        match source {
            NetSpawnSource::Prefab(path) => {
                if !self.prefabs.contains_key(path) {
                    match Prefab::load(zestengine_core::asset_ref::resolve_path(&self.project_root, path)) {
                        Ok(p) => {
                            self.prefabs.insert(path.to_owned(), p);
                        }
                        Err(e) => {
                            tracing::error!("net spawn {path:?}: {e:#}");
                            return;
                        }
                    }
                }
                let mut entries = crate::net_sync::spawn_entries(&self.prefabs[path], root_id, translation, rotation);
                crate::prefab::expand_entries(&mut entries, &self.project_root);
                match w.scene.append_entries_with(w.renderer.as_deref_mut(), w.shaders, &self.project_root, w.registry, &entries) {
                    Ok(_) => self.physics.mark_dirty(),
                    Err(e) => tracing::error!("net spawn {path:?}: {e:#}"),
                }
            }
            NetSpawnSource::Clone(original) => {
                if let Some(i) = w.scene.index_of(*original)
                    && let Some(n) = w.scene.duplicate_object(i, w.registry)
                {
                    crate::net_sync::reid_subtree(w.scene, n, root_id);
                    let o = &mut w.scene.objects[n];
                    o.set_parent(None);
                    *o.enabled_mut() = true;
                    o.transform_mut().translation = translation;
                    o.transform_mut().rotation = rotation;
                    self.physics.mark_dirty();
                }
            }
        }
    }
}
