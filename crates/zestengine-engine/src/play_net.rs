//! Networking hooks of the play session (a child module of `play` so it can
//! reach the session's physics and prefab cache): applies the world changes
//! the network asks for, hands physics bodies between local simulation and
//! remote control, applies networked spawn / despawn / variable commands
//! locally before they are replicated, and runs client-side prediction and
//! the host's simulation of predicted controllers (`net_predict`).

use zestengine_core::{
    Animator, Character, Component, ComponentCommand, InputState, MoveHost, NetSpawnSource, NetWorldOp,
    NetworkObject, NetworkVariables, ObjectId, PhysicsQuery, Predictable, RayFilter, RayHit, Swimmer, Water,
    as_predictable,
};
use glam::{Mat4, Quat, Vec3};

use super::{PlaySession, World};
use crate::net::NetSession;
use crate::net_predict::{AuthState, InputCmd, MoveState, Replay};
use crate::physics::PhysicsWorld;
use crate::prefab::Prefab;
use crate::scene::LoadedScene;

fn component<T: 'static>(scene: &LoadedScene, i: usize) -> Option<&T> {
    scene.objects[i].components.iter().find_map(|c| c.as_any().downcast_ref::<T>())
}

/// Slot of the object's predictable controller among its components.
fn controller_slot(scene: &LoadedScene, i: usize) -> Option<usize> {
    scene.objects[i].components.iter().position(|c| zestengine_core::is_predictable(c.as_ref()))
}

/// Host: seconds over which a remote predicted pawn's display catches up
/// with its simulation after a clump of inputs.
const HOST_SMOOTH_TIME: f32 = 0.08;
/// Host: a display gap larger than this (metres) is a teleport, shown at once.
const HOST_SMOOTH_SNAP: f32 = 2.0;

fn predictable(scene: &mut LoadedScene, i: usize) -> Option<&mut dyn Predictable> {
    let slot = controller_slot(scene, i)?;
    as_predictable(scene.objects[i].components[slot].as_mut())
}

/// One predicted controller driven outside the component passes: the host's
/// authoritative simulation and the owner's rewind + replay. It is the
/// controller's [`MoveHost`], so it steps exactly like the live update (the
/// same step, the same character controller moves).
struct MoverSim<'a> {
    scene: &'a mut LoadedScene,
    physics: &'a mut PhysicsWorld,
    index: usize,
    physical: bool,
    /// Ground contact of the controller's last move.
    contact: bool,
    /// A capsule resize waiting for the next move (as `ctx.modify` waits
    /// for the end of the pass in a live frame).
    capsule: Option<(f32, [f32; 3])>,
    water_time: f32,
}

impl<'a> MoverSim<'a> {
    fn new(scene: &'a mut LoadedScene, physics: &'a mut PhysicsWorld, index: usize, water_time: f32) -> Self {
        let physical = component::<Character>(scene, index).is_some_and(|c| c.use_default_capsule);
        let id = scene.objects[index].id;
        let contact = if physical { physics.grounded(id) } else { predictable(scene, index).is_some_and(|p| p.motion().grounded) };
        Self { scene, physics, index, physical, contact, capsule: None, water_time }
    }

    fn state(&mut self) -> MoveState {
        let (_, rot, pos) = self.scene.world_transform(self.index).to_scale_rotation_translation();
        let motion = predictable(self.scene, self.index).map(|p| p.motion()).unwrap_or_default();
        MoveState { pos, rot, motion, contact: self.contact }
    }

    fn apply_capsule(&mut self) {
        let Some((height, center)) = self.capsule.take() else {
            return;
        };
        if let Some(cc) =
            self.scene.objects[self.index].components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<Character>())
        {
            cc.height = height;
            cc.center = center;
        }
    }

    /// Run `f` on the controller with this simulation as its host.
    fn with_controller(&mut self, f: impl FnOnce(&mut dyn Predictable, &mut Self)) {
        let Some(slot) = controller_slot(self.scene, self.index) else {
            return;
        };
        let mut c: Box<dyn Component> = self.scene.objects[self.index].components.remove(slot);
        if let Some(p) = as_predictable(c.as_mut()) {
            f(p, self);
        }
        self.scene.objects[self.index].components.insert(slot, c);
        self.apply_capsule();
    }
}

impl MoveHost for MoverSim<'_> {
    fn self_id(&self) -> ObjectId {
        self.scene.objects[self.index].id
    }
    fn translation(&self) -> Vec3 {
        self.scene.objects[self.index].translation
    }
    fn set_translation(&mut self, t: Vec3) {
        self.scene.objects[self.index].translation = t;
    }
    fn rotation(&self) -> Quat {
        self.scene.objects[self.index].rotation
    }
    fn scale(&self) -> Vec3 { self.scene.objects[self.index].scale }
    fn set_rotation(&mut self, r: Quat) {
        self.scene.objects[self.index].rotation = r;
    }
    fn parent_world(&self) -> Option<Mat4> {
        self.scene.objects[self.index].parent.map(|p| self.scene.world_transform(p))
    }
    fn has_character(&self) -> bool {
        self.physical
    }
    fn move_character(&mut self, delta: Vec3) {
        self.apply_capsule();
        self.physics.move_character(self.scene, self.index, delta);
        self.contact = self.physics.grounded(self.scene.objects[self.index].id);
    }
    fn grounded(&self) -> bool {
        self.contact
    }
    fn capsule(&self) -> Option<(f32, f32, [f32; 3])> {
        component::<Character>(self.scene, self.index).map(|c| (c.radius, c.height, c.center))
    }
    fn set_capsule(&mut self, height: f32, center: [f32; 3]) {
        self.capsule = Some((height, center));
    }
    fn swimmer(&self) -> Swimmer {
        component::<Swimmer>(self.scene, self.index).copied().unwrap_or_default()
    }
    fn water_surface(&self, p: Vec3) -> Option<f32> {
        (0..self.scene.objects.len()).find_map(|i| {
            let w = component::<Water>(self.scene, i)?;
            w.height_at(self.scene.world_transform(i), p, self.water_time).map(|(h, _)| h)
        })
    }
    fn raycast(&self, origin: Vec3, dir: Vec3, max_dist: f32, filter: &RayFilter) -> Option<RayHit> {
        self.physics.raycast(origin, dir, max_dist, filter)
    }
    fn set_animator_bool(&mut self, name: &str, value: bool) {
        if let Some(a) = self.scene.objects[self.index].components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<Animator>()) {
            a.set_bool(name, value);
        }
    }
}

impl Replay for MoverSim<'_> {
    fn rewind(&mut self, s: &MoveState) {
        let i = self.index;
        let (pos, rot) = match self.scene.objects[i].parent {
            Some(p) => {
                let pw = self.scene.world_transform(p);
                let (_, pr, _) = pw.to_scale_rotation_translation();
                (pw.inverse().transform_point3(s.pos), (pr.inverse() * s.rot).normalize())
            }
            None => (s.pos, s.rot),
        };
        self.scene.objects[i].translation = pos;
        self.scene.objects[i].rotation = rot;
        self.with_controller(|p, host| p.set_motion(&s.motion, host));
        self.contact = s.contact;
        if self.physical {
            self.physics.place_character(self.scene, i, s.contact);
        }
    }

    fn step(&mut self, cmd: &InputCmd) -> MoveState {
        let input = cmd.input();
        self.with_controller(|p, host| p.replay(host, &input, &cmd.extra, cmd.dt));
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
                        && w.scene.objects[i].enabled != *on
                    {
                        w.scene.objects[i].enabled = *on;
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
                if let Some(i) = w.scene.index_of(*id)
                    && let Some(v) = w.scene.objects[i]
                        .components
                        .iter_mut()
                        .find_map(|c| c.as_any_mut().downcast_mut::<NetworkVariables>())
                {
                    v.set(key, value.clone());
                }
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
        let movers: Vec<(usize, ObjectId)> = scene
            .objects
            .iter()
            .enumerate()
            .filter(|(_, o)| crate::net_sync::is_predicted_in(o, net))
            .map(|(i, o)| (i, o.id))
            .collect();
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
                    let local_off = crate::net_sync::local_offset(scene, i, offset);
                    scene.objects[i].translation += local_off;
                    if let Some(n) = scene.objects[i].components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<NetworkObject>()) {
                        n.state.visual_offset = offset;
                    }
                }
            } else if owner == local && predictable(scene, i).is_some_and(|p| p.locally_driven()) {
                predicting.push(raw);
                let mut pred = net.predictors.remove(&raw).unwrap_or_default();
                let extra = predictable(scene, i).map(|p| p.frame_extra()).unwrap_or_default();
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
                    let local_off = crate::net_sync::local_offset(scene, i, offset);
                    scene.objects[i].translation += local_off;
                    if let Some(n) = scene.objects[i].components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<NetworkObject>()) {
                        n.state.visual_offset = offset;
                    }
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
                    o.parent = None;
                    o.enabled = true;
                    o.translation = translation;
                    o.rotation = rotation;
                    self.physics.mark_dirty();
                }
            }
        }
    }
}
