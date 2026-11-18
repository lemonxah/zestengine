//! One running game: the per-frame gameplay loop shared by the editor's Play
//! mode and the standalone runtime, so both behave identically.
//!
//! Frame order (Unity-style):
//! 1. `start` for every newly active object (Play start, spawned, re-enabled)
//! 2. fixed steps at [`FIXED_DT`]: `fixed_update`, then one physics step each
//! 3. collision / trigger callbacks
//! 4. `update`, then `late_update`
//! 5. audio spatialization
//!
//! Component commands that touch the world (spawn, destroy, physics, audio …)
//! are applied right after the pass that issued them. The rest (scene switch,
//! window, networking, quit) are returned to the host from [`PlaySession::tick`].

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use glam::{Mat4, Quat, Vec3};

use zestengine_core::{AnyBox, ComponentCommand, ComponentRegistry, InputState, NetView, ObjectId};
use zestengine_render::Renderer;

use crate::audio::AudioEngine;
use crate::navmesh::NavSystem;
use crate::physics::{ContactEvent, FIXED_DT, PhysicsWorld};
use crate::physics2d::{ContactEvent2D, Physics2D, QueryHub};
use crate::prefab::Prefab;
use crate::scene::{LoadedScene, PassEnv};
use crate::shaders::ShaderLibrary;

#[path = "play_net.rs"]
mod play_net;

/// Upper bound on fixed steps per frame, so a long hitch can't snowball.
const MAX_STEPS: u32 = 8;

/// The mutable engine state a tick works on.
pub struct World<'a> {
    pub scene: &'a mut LoadedScene,
    /// Needed only to spawn prefabs (mesh / material upload). None in headless tests.
    pub renderer: Option<&'a mut Renderer>,
    pub registry: &'a ComponentRegistry,
    pub shaders: &'a mut ShaderLibrary,
}

pub struct TickInput<'a> {
    /// Real frame time in seconds.
    pub dt: f32,
    pub input: &'a InputState,
    pub net: &'a NetView,
    /// Listener pose when the scene has no `AudioListener` (the camera's
    /// world transform).
    pub listener_fallback: Mat4,
    /// Game view, pointer, camera and UI clicks for this frame.
    pub screen: &'a zestengine_core::ScreenInfo,
}

/// CPU cost of the last tick, for the profiler overlay.
#[derive(Clone, Copy, Default)]
pub struct PlayTimings {
    pub components_ms: f32,
    pub physics_ms: f32,
    pub audio_ms: f32,
}

pub struct PlaySession {
    suspended: HashMap<ObjectId, Vec<String>>,
    physics: PhysicsWorld,
    /// The 2D world; empty (no cost) until a scene has 2D bodies.
    pub(crate) physics2d: Physics2D,
    /// Scratch for the per-frame 2D stay events.
    stays2d: Vec<ContactEvent2D>,
    physics_settings: zestengine_core::PhysicsSettings,
    /// Water clock minus play time (non-zero when following a shared clock).
    water_offset: f32,
    nav: NavSystem,
    audio: Option<AudioEngine>,
    /// Mix tap -> band analysis for materials using Toon's AUDIO feature.
    audio_fx: crate::audio_tap::AudioReactive,
    project_root: PathBuf,
    started: HashSet<ObjectId>,
    accumulator: f32,
    /// Scaled game time in seconds.
    pub time: f32,
    pub time_scale: f32,
    /// Readable this frame (emitted last frame).
    events_prev: Vec<AnyBox>,
    events_next: Vec<AnyBox>,
    sequences: crate::sequence_play::SequenceAssets,
    prefabs: HashMap<String, Prefab>,
    listener: Mat4,
    pub timings: PlayTimings,
    save: zestengine_core::SaveGame,
    save_dir: PathBuf,
    /// Shared with the UI (`UiText::loc_key`): copy-on-write on language change.
    strings: std::sync::Arc<zestengine_core::Localization>,
    mixer: zestengine_core::AudioMixer,
    /// Object-hierarchy rigs for Animators without a skinned mesh: the rig and
    /// the scene object id of each node (rest = transforms at first use).
    object_rigs: HashMap<ObjectId, (crate::anim_graph::Rig, Vec<ObjectId>)>,
    /// World position of each parameter-driver source last frame.
    /// Last seen position + smoothed velocity per driver source object.
    driver_prev: HashMap<ObjectId, (Vec3, Vec3)>,
    /// Avatar parameter stores: built-ins in, driver writes out.
    avatar_params: crate::avatar_runtime::AvatarRuntime,
    /// Water Containers' water and the streams poured from them.
    carry: crate::water_carry::CarrySim,
    water_interaction: crate::water_interaction::WaterInteractionSim,
    /// Static objects and still colliders as walls in the simulated water.
    water_walls: crate::water_walls::WaterWalls,
    /// A UI text field had the keyboard last tick (prediction records no
    /// gameplay input then).
    typing: bool,
}

impl PlaySession {
    /// Begin playing `scene`. `audio` is the output device to play through
    /// (None = silent); hand it back with [`PlaySession::end`].
    pub fn new(scene: &LoadedScene, project_root: &Path, audio: Option<AudioEngine>) -> Self {
        Self {
            physics: PhysicsWorld::build_with(scene, Some(project_root), &zestengine_core::PhysicsSettings::default()),
            physics2d: Physics2D::new(Some(project_root), &zestengine_core::PhysicsSettings::default()),
            stays2d: Vec::new(),
            physics_settings: zestengine_core::PhysicsSettings::default(),
            suspended: HashMap::new(),
            water_offset: 0.0,
            nav: NavSystem::default(),
            audio,
            audio_fx: Default::default(),
            project_root: project_root.to_owned(),
            started: HashSet::new(),
            accumulator: 0.0,
            time: 0.0,
            time_scale: 1.0,
            events_prev: Vec::new(),
            events_next: Vec::new(),
            sequences: Default::default(),
            prefabs: HashMap::new(),
            listener: Mat4::IDENTITY,
            timings: PlayTimings::default(),
            save: {let mut save=zestengine_core::SaveGame::new();save.mods=scene.mod_lock.clone();save},
            save_dir: zestengine_assets::mods::origin_root(project_root).join(".zestengine").join("saves"),
            strings: std::sync::Arc::new(load_strings(project_root)),
            mixer: zestengine_core::AudioMixer::default(),
            object_rigs: HashMap::new(),
            driver_prev: HashMap::new(),
            // Editor Play keeps saved avatar parameters out of the project's
            // tracked files (`.zest` is git-ignored).
            avatar_params: {
                let mut a = crate::avatar_runtime::AvatarRuntime::default();
                a.set_dir(project_root.join(".zest").join("saved"));
                a
            },
            carry: crate::water_carry::CarrySim::default(),
            water_interaction: crate::water_interaction::WaterInteractionSim::default(),
            water_walls: crate::water_walls::WaterWalls::default(),
            typing: false,
        }
    }

    /// The water clock: play time, or the networked game's shared clock.
    pub fn water_time(&self) -> f32 {
        self.time + self.water_offset
    }

    /// Follow a clock every peer agrees on for the water (the host's, from
    /// [`crate::net::NetSession::shared_clock`]); None = local play time.
    pub fn set_shared_clock(&mut self, clock: Option<f64>) {
        let target = clock.map_or(0.0, |c| (c - self.time as f64) as f32);
        if (target - self.water_offset).abs() > 0.5 {
            self.water_offset = target;
        } else {
            self.water_offset += (target - self.water_offset) * 0.1;
        }
    }

    /// Project physics (gravity, contact stiffness, default material). Rebuilds
    /// the simulation, so call it before the first tick.
    pub fn set_physics(&mut self, settings: &zestengine_core::PhysicsSettings, scene: &LoadedScene) {
        self.physics_settings = settings.clone();
        self.physics = PhysicsWorld::build_with(scene, Some(&self.project_root), settings);
        self.physics2d.set_settings(settings);
    }

    /// Where `ctx.write_save` / `load_save` slots live (default: the
    /// project's `.zestengine/saves`; a built game uses the user's data folder).
    pub fn set_save_dir(&mut self, dir: PathBuf) {
        // Saved avatar parameters live in `<saves>/avatar_params/`.
        self.avatar_params.set_dir(dir.clone());
        self.save_dir = dir;
    }

    /// Audio bus volumes (from the project's Audio Mixer).
    /// Apply the project's Steam Audio settings to the audio device.
    pub fn set_spatial_audio(&mut self, settings: &zestengine_core::SpatialAudioSettings) {
        if let Some(audio) = self.audio.as_mut() {
            audio.configure(settings, &self.project_root);
        }
    }

    pub fn set_mixer(&mut self, mixer: zestengine_core::AudioMixer) {
        self.mixer = mixer;
    }

    fn slot_path(&self, slot: &str) -> PathBuf {
        let safe: String = slot
            .chars()
            .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
            .collect();
        self.save_dir.join(format!("{safe}.save"))
    }

    /// A different scene was loaded mid-play: start it fresh.
    /// The string tables (shared; changes on language switch).
    pub fn strings(&self) -> std::sync::Arc<zestengine_core::Localization> {
        self.strings.clone()
    }

    pub fn scene_changed(&mut self, scene: &LoadedScene) {
        self.physics = PhysicsWorld::build_with(scene, Some(&self.project_root), &self.physics_settings);
        self.physics2d.reset();
        self.nav = NavSystem::default();
        self.started.clear();
        self.accumulator = 0.0;
        self.events_prev.clear();
        self.events_next.clear();
        self.object_rigs.clear();
        self.driver_prev.clear();
        self.avatar_params.reset();
        self.carry.reset();
        self.water_interaction = Default::default();
        if let Some(audio) = self.audio.as_mut() {
            audio.stop_all();
        }
    }

    /// Both physics worlds rebuild their membership on the next sync.
    fn mark_physics_dirty(&mut self) {
        self.physics.mark_dirty();
        self.physics2d.mark_dirty();
    }

    /// World-space outlines of 2D colliders for the Play view: bodies with
    /// debug draw on, or every body when `all`.
        pub fn debug_lines2d(&self, all: bool, out: &mut Vec<(Vec3, Vec3)>) {
        self.physics2d.debug_lines(all, out);
    }

    /// Objects were added / removed outside the session (additive streaming).
    pub fn structure_changed(&mut self) {
        self.mark_physics_dirty();
        self.nav.invalidate();
    }

    /// Deliver an engine-side event (e.g. content loading) to every component
    /// on the next tick via `ctx.events`.
    pub fn post_event<E: std::any::Any>(&mut self, event: E) {
        self.events_prev.push(AnyBox(Box::new(event)));
    }

    /// Feed the mixed output's band analysis to materials that use the AUDIO
    /// feature. While none does, the tap stays off (nothing runs on the audio
    /// thread) and the scene is only probed every few frames.
    fn drive_audio_materials(&mut self, w: &mut World, dt: f32) {
        let (Some(audio), Some(renderer)) = (self.audio.as_ref(), w.renderer.as_deref_mut()) else { return };
        let bands = if self.audio_fx.active {
            audio.set_tap_enabled(true);
            self.audio_fx.analyze_with(|out| audio.read_tap(out), dt)
        } else if self.audio_fx.should_probe() {
            Default::default()
        } else {
            return;
        };
        let fed = w.scene.set_audio_bands(renderer, w.shaders, &self.project_root, &bands);
        self.audio_fx.active = fed > 0;
        audio.set_tap_enabled(self.audio_fx.active);
    }

    /// Stop playing; returns the audio device for reuse.
        pub fn end(mut self) -> Option<AudioEngine> {
        if let Some(audio) = self.audio.as_mut() {
            audio.stop_all();
        }
        self.audio
    }

    /// Fire `binding` once as if an event source on `owner` ran it, with
    /// `instigator` as the event's instigator (the editor's Test button).
    /// Returns the host-only commands it produced; a failure to resolve the
    /// target, component, action or arguments is the error. Failures while the
    /// queued call runs (a guest trap, say) are only logged, like a real firing.
        pub fn test_binding(
        &mut self,
        w: &mut World,
        _inp: &TickInput,
        owner: ObjectId,
        binding: &zestengine_core::EventBinding,
        instigator: Option<ObjectId>,
    ) -> Result<Vec<ComponentCommand>, String> {
        let mut commands = Vec::new();
        w.scene.objects.query(|world, _| {
            let entity = world.entity(owner).ok_or("the object is gone")?;
            if !world.current_active(entity) { return Err("the object is inactive".to_string()); }
            let reader = zestengine_core::ecs::NativeProperties::new(world, &w.scene.objects);
            let source = zestengine_core::FireSource { source: owner, instigator, payload: &[] };
            binding.try_fire_native(&reader, &source, &mut commands)
        })?;
        let mut host = Vec::new();
        self.apply(w, commands, &mut host);
        Ok(host)
    }

    /// Advance one frame. Returns the commands only the host can apply.
    pub fn tick(&mut self, w: &mut World, inp: &TickInput) -> Vec<ComponentCommand> {
        w.scene.water_assets.root = self.project_root.clone();
        w.scene.cloth_cache_assets.root = self.project_root.clone();
        w.scene.water_assets.sync(&w.scene.objects);
        crate::water_connections::sync(w.scene);
        crate::water_bake::update(w.scene);
        let tick_start = Instant::now();
        let unscaled = inp.dt.clamp(0.0, 0.25);
        let dt = unscaled * self.time_scale;
        self.time += dt;
        self.listener = crate::audio::listener_pose(w.scene).unwrap_or(inp.listener_fallback);
        let events = std::mem::take(&mut self.events_prev);
        let mut host = Vec::new();
        self.typing = inp.screen.ui_text_editing;

        self.net_pre_tick(w, inp.net, &mut host);
        self.start_pending(w, inp, dt, unscaled, &events, &mut host);
        self.apply_animation_output(w, inp, dt, unscaled, &events, &mut host);

        let phys_start = Instant::now();
        let mut fixed_ms = 0.0;
        // GameObject tiles: one prefab instance per cell of a tile that names one (play-time spawn).
        if w.scene.sync_tile_objects(w.renderer.as_deref_mut(), w.shaders, &self.project_root, w.registry) > 0 {
            self.physics.mark_dirty();
            self.physics2d.mark_dirty();
        }
        crate::water_connections::sync(w.scene);
        crate::water_terrain::sync(w.scene);
        self.physics.sync_structure(w.scene);
        if self.physics2d.sync_structure(w.scene) {
            self.physics2d.write_measured(w.scene);
        }
        { let world = w.scene.world_transforms(); crate::water_exclusion::sync(w.scene, &world); }
        self.carry.track(w.scene, dt);
        self.accumulator += dt;
        let mut steps = 0;
        while self.accumulator >= FIXED_DT && steps < MAX_STEPS {
            let t = Instant::now();
            self.ecs_pass(w, inp, zestengine_core::ecs::Stage::FixedUpdate, FIXED_DT, unscaled, &events, &mut host);
            fixed_ms += t.elapsed().as_secs_f32() * 1000.0;
            crate::water_connections::sync(w.scene);
        crate::water_terrain::sync(w.scene);
            self.physics.sync_structure(w.scene);
            let world = w.scene.world_transforms();
            crate::water_exclusion::sync(w.scene, &world);
            self.physics.push_poses(w.scene, &world);
            if w.scene.take_shape_collider_dirty() {
                self.physics2d.mark_dirty();
            }
            self.physics2d.sync_structure(w.scene);
            self.physics2d.sync_tilemaps(w.scene);
            self.physics2d.push_poses(w.scene, &world);
            self.physics.drive_vehicles(w.scene, &world);
            let vortices = crate::water_fx::scene_vortices(w.scene, &world, self.water_time());
            let mut water = self.physics.apply_water(w.scene, &world, self.water_time(), &vortices);
            let water_t = self.water_time();
            self.water_interaction.step(w.scene, &world, water_t, FIXED_DT, &mut water);
            water.fx_pushes.append(&mut w.scene.water_fx_pushes);
            w.scene.objects.query(|ecs, _| {
                for &(i, frac) in &water.submerged {
                    if let Some(mut b) = ecs.get_mut::<zestengine_core::Buoyancy>(w.scene.objects[i].entity()) {
                        b.submerged = frac;
                    }
                }
            });
            let water_t = self.water_time();
            self.carry.step(w.scene, &world, Some(&mut self.physics), water_t, FIXED_DT, &mut water);
            crate::physics::step_ripples(w.scene, &world, self.listener.w_axis.truncate(), &water, Some(&self.physics), water_t, FIXED_DT);
            self.water_walls.update(w.scene, &world);
            w.scene.water_splashes.extend(water.splashes.iter().copied());
            // One wake per body per frame is plenty for the effects.
            for wake in &water.wakes {
                if !w.scene.water_wakes.iter().any(|x| x.at.distance_squared(wake.at) < 1e-4) {
                    w.scene.water_wakes.push(crate::water_fx::FxWake { at: wake.at, flow: wake.flow, radius: wake.radius });
                }
            }
            self.physics.step();
            self.physics2d.step();
            self.accumulator -= FIXED_DT;
            steps += 1;
        }
        if steps == MAX_STEPS {
            self.accumulator = self.accumulator.min(FIXED_DT);
        }
        if steps > 0 {
            self.physics.reset_forces();
            // What the inspector shows as each body's real mass and density.
            self.physics.write_measured(w.scene);
        }
        let alpha = (self.accumulator / FIXED_DT).clamp(0.0, 1.0);
        self.physics.write_back(w.scene, alpha);
        self.physics2d.write_back(w.scene, alpha);
        { let world = w.scene.world_transforms(); crate::water_exclusion::sync(w.scene, &world); }
        if self.physics2d.world().is_some() || !w.scene.debug_lines2d.is_empty() {
            w.scene.debug_lines2d.clear();
            self.physics2d.debug_lines(false, &mut w.scene.debug_lines2d);
        }
        w.scene.water_pours = self.carry.snapshot(alpha);
        // Navmeshes bake once the physics world has stepped (its query
        // structure is built by the step).
        if steps > 0 && self.nav.needs_bake() {
            self.nav.bake(w.scene, &self.physics);
        }
        let physics_ms = phys_start.elapsed().as_secs_f32() * 1000.0 - fixed_ms;

        let contacts = self.physics.take_events();
        if !contacts.is_empty() {
            self.deliver_contacts(w, inp, dt, unscaled, &events, &mut host, &contacts);
        }
        self.deliver_contacts2d(w, inp, dt, unscaled, &events, &mut host);

        self.ecs_pass(w, inp, zestengine_core::ecs::Stage::Update, dt, unscaled, &events, &mut host);
        // Movement requests have now been applied. Gameplay systems see the
        // corrected physics position and this frame's camera/tracking poses.
        self.ecs_pass(w, inp, zestengine_core::ecs::Stage::Gameplay, dt, unscaled, &events, &mut host);
        self.nav.move_agents(w.scene, &mut self.physics, dt);
        self.drive_animator_params(w.scene, dt);
        self.avatar_params.drive(w.scene, &self.physics, dt);
        w.scene.anim_assets.unscaled_dt = unscaled;
        self.animate_objects(w.scene, dt);
        let played = crate::sequence_play::tick(w.scene, &mut self.sequences, &self.project_root, dt);
        self.events_next.extend(played.events.into_iter().map(|e| AnyBox(Box::new(e))));
        self.events_next.extend(played.finished.into_iter().map(|e| AnyBox(Box::new(e))));
        if !played.commands.is_empty() {
            self.apply(w, played.commands, &mut host);
        }
        self.ecs_pass(w, inp, zestengine_core::ecs::Stage::LateUpdate, dt, unscaled, &events, &mut host);

        self.listener = crate::audio::listener_pose(w.scene).unwrap_or(inp.listener_fallback);
        let audio_start = Instant::now();
        let water_t = self.water_time();
        if let Some(audio) = self.audio.as_mut() {
            let cues = crate::audio::gather_cues(w.scene);
            audio.update_environment(w.scene, &cues, self.listener, &self.project_root, unscaled);
            audio.sync_geometry(w.scene);
            audio.update(&cues, self.listener, &self.mixer);
            let under = crate::underwater::listener(w.scene, self.listener.w_axis.truncate(), water_t);
            audio.set_underwater(under.as_ref(), unscaled, &self.project_root, &self.mixer);
        }
        self.drive_audio_materials(w, unscaled);
        let audio_ms = audio_start.elapsed().as_secs_f32() * 1000.0;

        self.events_prev = std::mem::take(&mut self.events_next);
        let total = tick_start.elapsed().as_secs_f32() * 1000.0;
        self.timings = PlayTimings {
            components_ms: (total - physics_ms - audio_ms).max(0.0),
            physics_ms,
            audio_ms,
        };
        host
    }

    /// Initialize native instances, including objects spawned by startup
    /// systems, and start each newly active object's audio once.
    fn start_pending(
        &mut self,
        w: &mut World,
        inp: &TickInput,
        dt: f32,
        unscaled: f32,
        events: &[AnyBox],
        host: &mut Vec<ComponentCommand>,
    ) {
        for _ in 0..8 {
            self.ecs_pass(w, inp, zestengine_core::ecs::Stage::Start, dt, unscaled, events, host);
            let active = w.scene.active_flags();
            let pending: HashSet<usize> = (0..w.scene.objects.len())
                .filter(|&i| active[i] && !self.started.contains(&w.scene.objects[i].id()))
                .collect();
            if pending.is_empty() {
                break;
            }
            let ids: HashSet<ObjectId> = pending.iter().map(|&i| w.scene.objects[i].id()).collect();
            self.started.extend(ids.iter().copied());
            if let Some(audio) = self.audio.as_mut() {
                for cue in crate::audio::gather_cues(w.scene) {
                    if cue.play_on_start && ids.contains(&cue.object) {
                        audio.start_cue(&cue, self.listener, &self.project_root, &self.mixer);
                    }
                }
            }
        }
    }

    /// Feed Animator parameter drivers (movement speed, grounded, another
    /// Animator's parameter).
    fn drive_animator_params(&mut self, scene: &mut LoadedScene, dt: f32) {
        use zestengine_core::{DriverSource, PhysicsQuery as _};
        let ids = scene.id_index();
        let mut writes: Vec<(usize, String, f32)> = Vec::new();
        let mut seen: Vec<(ObjectId, Vec3, Vec3)> = Vec::new();
        for (i, o) in scene.objects.iter().enumerate() {
            let Some(a) = o.components.get_typed::<zestengine_core::Animator>() else {
                continue;
            };
            for d in &a.drivers {
                if d.parameter.is_empty() {
                    continue;
                }
                let src_id = d.object.id().unwrap_or(o.id());
                let Some(&src) = ids.get(&src_id) else {
                    continue;
                };
                let value = match d.source {
                    DriverSource::AnimatorParameter => scene.objects[src]
                        .components.get_typed::<zestengine_core::Animator>()
                        .map(|other| other.get_float(&d.source_parameter)),
                    DriverSource::Grounded => Some(if self.physics.grounded(src_id) { 1.0 } else { 0.0 }),
                    _ => {
                        let world = scene.world_transform(src);
                        let pos = world.w_axis.truncate();
                        let (last, prev_vel) = self.driver_prev.get(&src_id).copied().unwrap_or((pos, Vec3::ZERO));
                        // Raw finite difference of the scene position. Kinematic
                        // bodies (the character) update their position once per
                        // fixed step, so under vsync the estimate beats between
                        // 0x and 2x the true speed; smooth it so the Speed /
                        // ForwardSpeed / Strafe drivers stay steady.
                        let vel = if dt > 1e-6 {
                            let raw = (pos - last) / dt;
                            let a = 1.0 - (-dt / 0.08).exp();
                            prev_vel.lerp(raw, a)
                        } else {
                            prev_vel
                        };
                        seen.push((src_id, pos, vel));
                        let fwd = world.transform_vector3(Vec3::NEG_Z).normalize_or_zero();
                        let right = world.transform_vector3(Vec3::X).normalize_or_zero();
                        Some(match d.source {
                            DriverSource::Speed => Vec3::new(vel.x, 0.0, vel.z).length(),
                            DriverSource::VerticalSpeed => vel.y,
                            DriverSource::ForwardSpeed => vel.dot(fwd),
                            _ => vel.dot(right),
                        })
                    }
                };
                if let Some(v) = value {
                    writes.push((i, d.parameter.clone(), v * d.scale));
                }
            }
        }
        if dt > 1e-6 {
            for (id, pos, vel) in seen {
                self.driver_prev.insert(id, (pos, vel));
            }
        }
        for (i, name, v) in writes {
            if let Some(mut a) = scene.objects[i]
                .components.get_typed_mut::<zestengine_core::Animator>()
            {
                a.set_float(&name, v);
            }
        }
    }

    /// Evaluate Animators on objects without a skinned mesh: controllers and
    /// `.anim` clips animate the object and its descendants' transforms.
    fn animate_objects(&mut self, scene: &mut LoadedScene, dt: f32) {
        let targets: Vec<usize> = (0..scene.objects.len())
            .filter(|&i| {
                scene.objects[i].components.contains_type::<zestengine_core::Animator>()
                    && scene.is_active(i)
                    && !scene.has_skinned_mesh(i)
            })
            .collect();
        if targets.is_empty() {
            return;
        }
        scene.anim_assets.set_root(&self.project_root);
        let ids = scene.id_index();
        for i in targets {
            let id = scene.objects[i].id();
            let valid = self
                .object_rigs
                .get(&id)
                .is_some_and(|(_, nodes)| nodes.iter().all(|n| ids.contains_key(n)));
            if !valid {
                let (rig, nodes) = crate::anim_graph::object_rig(scene, i);
                let nodes = nodes.iter().map(|&n| scene.objects[n].id()).collect();
                self.object_rigs.insert(id, (rig, nodes));
            }
            let (rig, node_ids) = &self.object_rigs[&id];
            let nodes: Vec<usize> = node_ids.iter().filter_map(|n| ids.get(n).copied()).collect();
            let LoadedScene { objects, anim_assets, .. } = &mut *scene;
            let Some(mut animator) = objects[i]
                .components.get_typed_mut::<zestengine_core::Animator>()
            else {
                continue;
            };
            if animator.controller.is_empty() && animator.clip.is_empty() && animator.state.current.is_none() {
                continue;
            }
            let ctrl = anim_assets.prepare(&animator);
            let set = crate::anim_graph::ClipSet { imported: &[], indices: &[], assets: &anim_assets.clips };
            let out = crate::anim_graph::advance_timed(&mut animator, ctrl.as_deref(), rig, &set, anim_assets.timing(dt));
            drop(animator);
            crate::anim_graph::apply_pose_to_objects(scene, &nodes, &out.pose);
            crate::anim_props::apply_pose_props(scene, rig, &nodes, &out.pose);
        }
    }

    /// Root motion and animation events produced by the last skinning update.
    fn apply_animation_output(
        &mut self,
        w: &mut World,
        inp: &TickInput,
        dt: f32,
        unscaled: f32,
        events: &[AnyBox],
        host: &mut Vec<ComponentCommand>,
    ) {
        let mut fired: HashMap<usize, Vec<String>> = HashMap::new();
        let mut motion: Vec<(usize, Vec3)> = Vec::new();
        for (i, o) in w.scene.objects.iter_mut().enumerate() {
            let Some(mut a) = o
                .components.get_typed_mut::<zestengine_core::Animator>()
            else {
                continue;
            };
            if !a.state.fired.is_empty() {
                fired.insert(i, std::mem::take(&mut a.state.fired));
            }
            if a.state.root_delta != Vec3::ZERO {
                motion.push((i, std::mem::take(&mut a.state.root_delta)));
            }
            // Root motion rotation: the object turns about its own up axis.
            let yaw = std::mem::take(&mut a.state.root_yaw);
            if yaw != 0.0 {
                o.transform_mut().rotation = (o.rotation() * Quat::from_rotation_y(yaw)).normalize();
            }
            for (layer, from, to) in std::mem::take(&mut a.state.state_changes) {
                self.events_next.push(AnyBox(Box::new(zestengine_core::AnimatorStateChanged { object: o.id(), layer, from, to })));
            }
        }
        for (i, delta) in motion {
            let has_cc = w.scene.objects[i]
                .components
                .iter()
                .any(|c| c.as_any().downcast_ref::<zestengine_core::Character>().is_some_and(|c| c.use_default_capsule));
            if has_cc {
                self.physics.sync_structure(w.scene);
                self.physics.move_character(w.scene, i, delta);
            } else {
                let world = w.scene.world_transform(i).w_axis.truncate() + delta;
                let local = match w.scene.objects[i].parent() {
                    Some(p) => w.scene.world_transform(p).inverse().transform_point3(world),
                    None => world,
                };
                w.scene.objects[i].transform_mut().translation = local;
            }
        }
        if fired.is_empty() {
            return;
        }
        for (&i, names) in &fired {
            let object = w.scene.objects[i].id();
            for entry in names {
                let (name, args) = zestengine_core::AnimEventArgs::decode(entry);
                self.events_next.push(AnyBox(Box::new(zestengine_core::AnimationEvent {
                    object,
                    name: name.to_owned(),
                    args,
                })));
            }
        }
        let scene_events: Vec<_> = fired.iter().flat_map(|(&i, names)| {
            let object = w.scene.objects[i].id();
            names.iter().map(move |entry| zestengine_core::ecs::SceneEvent::Animation { object, name: zestengine_core::AnimEventArgs::decode(entry).0.to_owned() })
        }).collect();
        let commands = self.collect_ecs(w, inp, zestengine_core::ecs::Stage::Events, dt, unscaled, events, &scene_events);
        self.apply(w, commands, host);
    }

    #[allow(clippy::too_many_arguments)]
    fn deliver_contacts(
        &mut self,
        w: &mut World,
        inp: &TickInput,
        dt: f32,
        unscaled: f32,
        events: &[AnyBox],
        host: &mut Vec<ComponentCommand>,
        contacts: &[ContactEvent],
    ) {
        use zestengine_core::ecs::{ContactPhase, SceneEvent};
        let mut scene_events = Vec::with_capacity(contacts.len() * 2);
        for event in contacts {
            for (object, other) in [(event.a, event.b), (event.b, event.a)] {
                if w.scene.index_of(object).is_none() { continue; }
                scene_events.push(SceneEvent::Contact { object, other, trigger: event.trigger,
                    phase: if event.started { ContactPhase::Enter } else { ContactPhase::Exit }, two_dimensional: false });
            }
        }
        let commands = self.collect_ecs(w, inp, zestengine_core::ecs::Stage::Events, dt, unscaled, events, &scene_events);
        self.apply(w, commands, host);
    }

    /// 2D contacts: this step's begin / end events plus one stay per pair still
    /// touching, to the colliders' objects and to the objects owning their
    /// bodies (a child collider reports to its Rigidbody 2D too).
    fn deliver_contacts2d(
        &mut self,
        w: &mut World,
        inp: &TickInput,
        dt: f32,
        unscaled: f32,
        events: &[AnyBox],
        host: &mut Vec<ComponentCommand>,
    ) {
        let began = self.physics2d.take_events();
        let mut stays = std::mem::take(&mut self.stays2d);
        stays.clear();
        if let Some(world) = self.physics2d.world() {
            world.stay_events(&mut stays);
        }
        if began.is_empty() && stays.is_empty() {
            self.stays2d = stays;
            return;
        }
        use zestengine_core::ecs::{ContactPhase, SceneEvent};
        let mut scene_events = Vec::with_capacity((began.len() + stays.len()) * 4);
        let add = |out: &mut Vec<SceneEvent>, phase, event: &ContactEvent2D| {
            for (collider, body, other) in [(event.a, event.a_body, event.b), (event.b, event.b_body, event.a)] {
                for object in if body == collider { [Some(collider), None] } else { [Some(collider), Some(body)] }.into_iter().flatten() {
                    if w.scene.index_of(object).is_none() { continue; }
                    out.push(SceneEvent::Contact { object, other, trigger: event.trigger, phase, two_dimensional: true });
                }
            }
        };
        for event in &began { add(&mut scene_events, if event.started { ContactPhase::Enter } else { ContactPhase::Exit }, event); }
        for event in &stays { add(&mut scene_events, ContactPhase::Stay, event); }
        self.stays2d = stays;
        let commands = self.collect_ecs(w, inp, zestengine_core::ecs::Stage::Events, dt, unscaled, events, &scene_events);
        self.apply(w, commands, host);
    }

    #[allow(clippy::too_many_arguments)]
    fn ecs_pass(&mut self, w: &mut World, inp: &TickInput, stage: zestengine_core::ecs::Stage, dt: f32, unscaled: f32, events: &[AnyBox], host: &mut Vec<ComponentCommand>) {
        let commands = self.collect_ecs(w, inp, stage, dt, unscaled, events, &[]);
        self.apply(w, commands, host);
    }

    #[allow(clippy::too_many_arguments)]
    fn collect_ecs(&self, w: &mut World, inp: &TickInput, stage: zestengine_core::ecs::Stage, dt: f32, unscaled: f32, events: &[AnyBox], scene_events: &[zestengine_core::ecs::SceneEvent]) -> Vec<ComponentCommand> {
        let hub = QueryHub { world3: &self.physics, world2: self.physics2d.world() };
        let env = PassEnv {
            dt, unscaled_dt: unscaled, time: self.time, water_time: self.water_time(), time_scale: self.time_scale,
            input: inp.input, net: inp.net, physics: Some(&hub), nav: (!self.nav.needs_bake()).then_some(&self.nav as &dyn zestengine_core::NavQuery), events, screen: inp.screen,
            save: &self.save, strings: &self.strings,
        };
        match w.scene.run_ecs_event_systems(w.registry, stage, &env, scene_events) {
            Ok(commands) => commands,
            Err(error) => { tracing::error!(?stage, "ECS stage failed: {error}"); Vec::new() },
        }
    }

    fn apply(&mut self, w: &mut World, commands: Vec<ComponentCommand>, host: &mut Vec<ComponentCommand>) {
        // Resolve every event's arguments at this boundary before applying any
        // command. Separate sources in one event batch retain the same deferred
        // property semantics as bindings resolved during a component pass.
        let commands = if commands.iter().any(|command| matches!(command, ComponentCommand::FireActions { .. })) {
            let mut resolved = Vec::with_capacity(commands.len());
            for command in commands {
                match command {
                    ComponentCommand::FireActions { bindings, source, instigator, payload } => {
                        w.scene.fire_native_actions(&bindings, &zestengine_core::FireSource { source, instigator, payload: &payload }, &mut resolved);
                    }
                    command => resolved.push(command),
                }
            }
            resolved
        } else { commands };
        for c in commands {
            match c {
                ComponentCommand::SetActiveCamera(id) => w.scene.set_active_camera(id),
                ComponentCommand::ReleaseActiveCamera(id) => {
                    if w.scene.active_camera_override() == Some(id) {
                        w.scene.restore_active_camera(None);
                    }
                }
                ComponentCommand::SuspendComponents { id, names } => {
                    if let Some(i) = w.scene.index_of(id) {
                        let held = self.suspended.entry(id).or_default();
                        let components = &mut w.scene.objects[i].components;
                        let targets: Vec<(usize, String)> = components
                            .iter()
                            .enumerate()
                            .filter(|(_, c)| zestengine_core::component_enabled(c.as_ref()) && names.iter().any(|n| n == c.type_name()))
                            .map(|(k, c)| (k, c.type_name().to_owned()))
                            .collect();
                        for (k, name) in targets {
                            components.set_enabled(k, false);
                            held.push(name);
                        }
                        self.mark_physics_dirty();
                    }
                }
                ComponentCommand::ResumeComponents(id) => {
                    if let Some(names) = self.suspended.remove(&id)
                        && let Some(i) = w.scene.index_of(id)
                    {
                        let components = &mut w.scene.objects[i].components;
                        let targets: Vec<usize> = components
                            .iter()
                            .enumerate()
                            .filter(|(_, c)| !zestengine_core::component_enabled(c.as_ref()) && names.iter().any(|n| n == c.type_name()))
                            .map(|(k, _)| k)
                            .collect();
                        for k in targets {
                            components.set_enabled(k, true);
                        }
                        self.mark_physics_dirty();
                    }
                }
                ComponentCommand::SetLocalTransform {
                    id,
                    translation,
                    rotation,
                    scale,
                } => w.scene.set_local_transform(id, translation, rotation, scale),
                ComponentCommand::SetWorldTransform {
                    id,
                    translation,
                    rotation,
                } => w.scene.set_world_transform(id, translation, rotation),
                ComponentCommand::ModifyMaterialParams { id, slot, edit } => {
                    if let Some(i) = w.scene.index_of(id) {
                        w.scene.modify_material_params(w.renderer.as_deref_mut(), i, slot, edit);
                    }
                }
                ComponentCommand::AssignMaterial { id, slot, path } => {
                    if let Some(i) = w.scene.index_of(id)
                        && let Some(renderer) = w.renderer.as_deref_mut()
                    {
                        w.scene.assign_material(renderer, w.shaders, i, slot, Path::new(&path), &self.project_root);
                    }
                }
                ComponentCommand::SpawnPrefab {
                    path,
                    root_id,
                    translation,
                    rotation,
                } => self.spawn_prefab(w, &path, root_id, translation, rotation),
                ComponentCommand::SpawnEntity { name, root_id, translation, rotation } => {
                    match w.scene.spawn_entity(w.renderer.as_deref_mut(), w.registry, &name, root_id, translation, rotation) {
                        Ok(_) => self.mark_physics_dirty(),
                        Err(e) => tracing::error!("spawn_entity {name:?}: {e:#}"),
                    }
                }
                ComponentCommand::Instantiate {
                    original,
                    new_id,
                    translation,
                    rotation,
                } => {
                    if let Some(i) = w.scene.index_of(original)
                        && let Some(n) = w.scene.duplicate_object(i, w.registry)
                    {
                        w.scene.objects.set_id(n, new_id).expect("fresh spawned ID");
                        let o = &mut w.scene.objects[n];
                        o.set_parent(None);
                        o.transform_mut().translation = translation;
                        o.transform_mut().rotation = rotation;
                        self.mark_physics_dirty();
                    }
                }
                ComponentCommand::Destroy(id) => {
                    if let Some(i) = w.scene.index_of(id) {
                        w.scene.remove_object(i);
                        self.mark_physics_dirty();
                    }
                }
                ComponentCommand::SetActive(id, on) => {
                    if let Some(i) = w.scene.index_of(id)
                        && w.scene.objects[i].enabled() != on
                    {
                        *w.scene.objects[i].enabled_mut() = on;
                        self.mark_physics_dirty();
                    }
                }
                ComponentCommand::Modify { id, type_id, f } => {
                    if let Some(i) = w.scene.index_of(id)
                        && let Some(mut c) = w.scene.objects[i]
                            .components
                            .iter_mut()
                            .find(|c| c.as_any().type_id() == type_id)
                    {
                        (f.0)(c.as_any_mut());
                    }
                }
                ComponentCommand::Emit(e) => self.events_next.push(e),
                ComponentCommand::InvokeAction { id, slot, component, action, args, instigator } => {
                    let mut follow_up = Vec::new();
                    let result = w.scene.invoke_native_action(zestengine_core::ecs::NativeActionCall { object: id, slot, component, action, args: &args, instigator }, &mut follow_up);
                    match result {
                        Ok(()) => self.apply(w, follow_up, host),
                        Err(e) => tracing::warn!("action {component}.{action}: {e}"),
                    }
                }
                ComponentCommand::WaterImpact(impact) => {w.scene.impact_water(impact,self.water_time());}
                ComponentCommand::AddForce { id, force } => {
                    self.physics.add_force(id, force);
                    self.physics2d.add_force(id, force);
                }
                ComponentCommand::AddImpulse { id, impulse } => {
                    self.physics.add_impulse(id, impulse);
                    self.physics2d.add_impulse(id, impulse);
                }
                ComponentCommand::AddTorque { id, torque } => {
                    self.physics.add_torque(id, torque);
                    self.physics2d.add_torque(id, torque);
                }
                ComponentCommand::SetVelocity { id, linear } => {
                    self.physics.set_velocity(id, linear);
                    self.physics2d.set_velocity(id, linear);
                }
                ComponentCommand::RebakeNavMesh => self.nav.invalidate(),
                ComponentCommand::SetJointMotor { id, speed, max_force } => {
                    self.physics.set_joint_motor(id, speed, max_force);
                    self.physics2d.set_joint_motor(w.scene, id, speed, max_force)
                }
                ComponentCommand::MoveCharacter { id, delta } => {
                    self.physics.sync_structure(w.scene);
                    if let Some(i) = w.scene.index_of(id) {
                        self.physics.move_character(w.scene, i, delta);
                    }
                }
                ComponentCommand::PlaySound { clip, position, volume } => {
                    if let Some(audio) = self.audio.as_mut() {
                        audio.play_oneshot(&self.project_root, &clip, position, volume, self.listener, &self.mixer);
                    }
                }
                ComponentCommand::PlayAudio(id) => {
                    if let Some(audio) = self.audio.as_mut()
                        && let Some(cue) = crate::audio::gather_cues(w.scene).into_iter().find(|c| c.object == id)
                    {
                        audio.start_cue(&cue, self.listener, &self.project_root, &self.mixer);
                    }
                }
                ComponentCommand::StopAudio(id) => {
                    if let Some(audio) = self.audio.as_mut() {
                        audio.stop_object(id);
                    }
                }
                ComponentCommand::SetTimeScale(s) => self.time_scale = s.max(0.0),
                ComponentCommand::SaveSet { key, value } => self.save.set(key, value),
                ComponentCommand::SaveRemove(key) => {
                    self.save.remove(&key);
                }
                ComponentCommand::SaveWrite(slot) => {
                    let path = self.slot_path(&slot);
                    let result = std::fs::create_dir_all(&self.save_dir)
                        .map_err(|e| e.to_string())
                        .and_then(|_| self.save.save(&path).map_err(|e| e.to_string()));
                    if let Err(e) = result {
                        tracing::error!("write_save {}: {e}", path.display());
                    }
                }
                ComponentCommand::SaveLoad(slot) => {
                    let path = self.slot_path(&slot);
                    match zestengine_core::SaveGame::load(&path) {
                        Ok(save) => match self.save.mods.check(&save.mods){
                            Ok(())=>self.save=save,
                            Err(e)=>tracing::error!("load_save {}: {e}",path.display()),
                        },
                        Err(e) => tracing::warn!("load_save {}: {e}", path.display()),
                    }
                }
                ComponentCommand::SetLanguage(lang) => {
                    if !std::sync::Arc::make_mut(&mut self.strings).set_language(&lang) {
                        tracing::warn!("set_language: no {lang:?} column in localization tables");
                    }
                }
                ComponentCommand::SetBusVolume { bus, volume } => self.mixer.set_volume(&bus, volume),
                ComponentCommand::SetBusMuted { bus, muted } => self.mixer.set_muted(&bus, muted),
                c @ (ComponentCommand::NetSpawn { .. } | ComponentCommand::NetDespawn(_) | ComponentCommand::NetSetVar { .. }) => {
                    self.apply_net_command(w, c, host)
                }
                other => host.push(other),
            }
        }
    }

    fn spawn_prefab(&mut self, w: &mut World, path: &str, root_id: ObjectId, translation: Vec3, rotation: Quat) {
        if !self.prefabs.contains_key(path) {
            match Prefab::load(zestengine_core::asset_ref::resolve_path(&self.project_root, path)) {
                Ok(p) => {
                    self.prefabs.insert(path.to_owned(), p);
                }
                Err(e) => {
                    tracing::error!("spawn_prefab {path:?}: {e:#}");
                    return;
                }
            }
        }
        let prefab = &self.prefabs[path];
        let mut entries = prefab.instantiate_rooted(translation.to_array(), Some(root_id));
        for e in entries.iter_mut().filter(|e| e.parent.is_none()) {
            e.rotation = rotation.to_array();
        }
        crate::prefab::expand_entries(&mut entries, &self.project_root);
        // No renderer (headless server): the CPU half only, so collision and
        // gameplay still see the spawned objects.
        match w
            .scene
            .append_entries_with(w.renderer.as_deref_mut(), w.shaders, &self.project_root, w.registry, &entries)
        {
            Ok(_) => self.physics.mark_dirty(),
            Err(e) => tracing::error!("spawn_prefab {path:?}: {e:#}"),
        }
    }
}

/// Merge every `localization/*.loc` table (same language columns) into one.
fn load_strings(project_root: &Path) -> zestengine_core::Localization {
    let mut out = zestengine_core::Localization::default();
    let Ok(dir) = std::fs::read_dir(project_root.join("localization")) else {
        return out;
    };
    let mut files: Vec<PathBuf> = dir
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "loc"))
        .collect();
    files.sort();
    for path in files {
        match zestengine_core::Localization::load(&path) {
            Ok(table) if out.languages.is_empty() => out = table,
            Ok(table) if table.languages == out.languages => out.entries.extend(table.entries),
            Ok(_) => tracing::warn!("{}: language columns differ from the first table; skipped", path.display()),
            Err(e) => tracing::warn!("{}: {e}", path.display()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use zestengine_core::Component;
    use super::*;
    use crate::scene::{MaterialEntry, RenderInfo, SceneObject, TexSlots};
    use zestengine_assets::ObjectSource;
    use zestengine_core::{BoxCollider, MotionContext, ObjectRef, RigidBody, SphereCollider, TypedComponent};
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize)]
    struct Health {
        hp: f32,
    }
    impl Default for Health {
        fn default() -> Self {
            Self { hp: 100.0 }
        }
    }
    impl TypedComponent for Health {
        const NAME: &'static str = "Test Health";
    }

    struct Ping;

    /// The user-facing native plugin pattern: query a built-in light and
    /// edit its actual typed value.
    #[derive(Serialize, Deserialize, Default)]
    struct Flicker {
        target: ObjectRef,
        seen: Vec<f32>,
    }
    impl TypedComponent for Flicker {
        const NAME: &'static str = "Test Flicker";
    }

    impl Flicker {
        fn update_native(&mut self, ctx: &mut zestengine_core::ecs::ObjectContext<'_, '_>) {
            if let Some(base) = ctx.get::<zestengine_core::LightComponent>(self.target).map(|light| light.intensity) {
                self.seen.push(base);
                ctx.edit::<zestengine_core::LightComponent>(self.target, move |l| l.intensity = base * 0.5);
            }
        }
    }

    /// Sits on the lamp itself: reads ITS OWN material (slot 0) and light,
    /// then halves both every frame - the user's flicker pattern, self-driven.
    #[derive(Serialize, Deserialize, Default)]
    struct SelfFlicker {
        seen_emission: Vec<f32>,
        seen_light: Vec<f32>,
    }
    impl TypedComponent for SelfFlicker {
        const NAME: &'static str = "Test Self Flicker";
    }

    impl SelfFlicker {
        fn update_native(&mut self, ctx: &mut zestengine_core::ecs::ObjectContext<'_, '_>) {
            let me: ObjectRef = ctx.self_id().into();
            if let Some(m) = ctx.material_params(me, 0) {
                self.seen_emission.push(m.emission_intensity);
                let base = m.emission_intensity;
                ctx.modify_material(me, 0, move |m| m.emission_intensity = base * 0.5);
            }
            if let Some(base) = ctx.get_self::<zestengine_core::LightComponent>().map(|light| light.intensity) {
                self.seen_light.push(base);
                ctx.edit::<zestengine_core::LightComponent>(me, move |l| l.intensity = base * 0.5);
            }
        }
    }

    /// Drives other objects: teleports a referenced "player", switches the
    /// camera, and reads a referenced object's material from the outside.
    #[derive(Serialize, Deserialize, Default)]
    struct WorldDriver {
        lamp: ObjectRef,
        player: ObjectRef,
        cam: ObjectRef,
        seen_lamp_emission: Vec<f32>,
        seen_extra_emission: Vec<f32>,
        invalid_slot_empty: bool,
        last_player_pos: Option<[f32; 3]>,
    }
    impl TypedComponent for WorldDriver {
        const NAME: &'static str = "Test World Driver";
    }

    impl WorldDriver {
        fn update_native(&mut self, ctx: &mut zestengine_core::ecs::ObjectContext<'_, '_>) {
            ctx.set_world_position_of(self.player, Vec3::new(2.0, 0.0, 3.0));
            ctx.set_active_camera(self.cam);
            if let Some(m) = ctx.material_params(self.lamp, 0) {
                self.seen_lamp_emission.push(m.emission_intensity);
            }
            if let Some(m) = ctx.material_params(self.lamp, 1) { self.seen_extra_emission.push(m.emission_intensity); }
            self.invalid_slot_empty = ctx.material_params(self.lamp, 2).is_none();
            self.last_player_pos = ctx.position_of(self.player).map(|p| p.to_array());
        }
    }

    /// Native systems consume only their chosen event types, before updating
    /// their own state; newly emitted events arrive on the following frame.
    #[test]
    fn a_seated_rider_follows_the_vehicle_without_pushing_it_and_gets_their_controller_back() {
        use zestengine_core::{Character, Disabled, ThirdPersonCharacter, Vehicle, component_enabled};
        let registry = ComponentRegistry::with_builtins();
        let mut scene = LoadedScene::empty();
        let vehicle = Vehicle { requires_driver: true, hide_driver: false, seat_driver: true, driver_seat: [0.0, 0.5, 0.0], exit_offset: [-2.0, 0.0, 0.0], wheels: Vec::new(), ..Default::default() };
        let car = obj(&mut scene, Vec3::new(10.0, 1.0, 0.0), true, vec![
            Box::new(RigidBody { mass: 800.0, gravity_scale: 0.0, ..Default::default() }),
            Box::new(BoxCollider { size: [2.0, 1.0, 4.0], ..Default::default() }),
            Box::new(vehicle),
        ]);
        let rider = obj(&mut scene, Vec3::new(0.0, 0.0, 0.0), true, vec![Box::new(Character::default()), Box::new(Disabled(Box::new(ThirdPersonCharacter::default())))]);
        let mut session = PlaySession::new(&scene, Path::new("."), None);
        tick(&mut session, &mut scene, &registry, 2);
        let car_i = scene.index_of(car).unwrap();
        let rider_i = scene.index_of(rider).unwrap();
        let enabled = |scene: &LoadedScene, k: usize| component_enabled(scene.objects[rider_i].components.get(k).unwrap().as_ref());
        let mut commands = Vec::new();
        assert!(scene.objects[car_i].components.get_typed_mut::<Vehicle>().unwrap().enter(car, rider, &mut commands));
        let mut shaders = ShaderLibrary::default();
        let mut host = Vec::new();
        session.apply(&mut World { scene: &mut scene, renderer: None, registry: &registry, shaders: &mut shaders }, commands, &mut host);
        assert!(!enabled(&scene, 0), "the capsule is switched off while seated");
        tick(&mut session, &mut scene, &registry, 30);
        let car_at = scene.world_transform(car_i).w_axis.truncate();
        let rider_at = scene.world_transform(rider_i).w_axis.truncate();
        assert!((car_at - Vec3::new(10.0, 1.0, 0.0)).length() < 1e-3, "the rider inside the chassis does not push it: {car_at:?}");
        assert!((rider_at - (car_at + Vec3::new(0.0, 0.5, 0.0))).length() < 1e-3, "the rider is held at the seat: {rider_at:?}");
        scene.objects[car_i].components.get_typed_mut::<Vehicle>().unwrap().pending_exit = true;
        tick(&mut session, &mut scene, &registry, 2);
        assert!(enabled(&scene, 0), "the capsule is back after getting out");
        assert!(!enabled(&scene, 1), "a controller that was already off stays off");
        let rider_at = scene.world_transform(rider_i).w_axis.truncate();
        assert!((rider_at.x - 8.0).abs() < 0.2 && rider_at.z.abs() < 0.2, "put down beside the vehicle: {rider_at:?}");
        assert!(scene.objects[car_i].components.get_typed::<Vehicle>().unwrap().driver.is_none());
    }

    #[test]
    fn native_event_consumers_filter_types_and_observe_previous_frames() {
        let registry = registry();
        let mut scene = LoadedScene::empty();
        let listener = obj(&mut scene, Vec3::ZERO, true, vec![Box::new(EventListener::default())]);
        let nosy = obj(&mut scene, Vec3::ZERO, true, vec![Box::new(NosyListener::default())]);
        obj(&mut scene, Vec3::ZERO, true, vec![Box::new(PingSender::default())]);
        let mut session = PlaySession::new(&scene, Path::new("."), None);
        tick(&mut session, &mut scene, &registry, 3);
        let l = get::<EventListener>(&scene, listener);
        assert_eq!(l.got, vec![1, 2], "each frame's event is delivered next frame");
        assert_eq!(l.saw_in_update, vec![0, 1, 2], "delivery runs before update (update sees the event)");
        let n = get::<NosyListener>(&scene, nosy);
        assert_eq!(n.seen, 0, "a system consuming another type never receives TestPing");
        assert_eq!(n.updates, 3, "the differently typed consumer still ran every frame");
    }

    #[derive(Serialize, Deserialize, Default)]
    struct TestPing {
        n: u32,
    }

    #[derive(Serialize, Deserialize, Default)]
    struct PingSender {
        n: u32,
    }
    impl TypedComponent for PingSender { const NAME: &'static str = "Test Ping Sender"; }

    #[derive(Serialize, Deserialize, Default)]
    struct EventListener {
        got: Vec<u32>,
        saw_in_update: Vec<u32>,
    }
    impl TypedComponent for EventListener { const NAME: &'static str = "Test Event Listener"; }

    #[derive(Serialize, Deserialize, Default)]
    struct NosyListener {
        seen: u32,
        updates: u32,
    }
    impl TypedComponent for NosyListener { const NAME: &'static str = "Test Nosy Listener"; }

    #[derive(Serialize, Deserialize, Default)]
    struct Shooter {
        target: ObjectRef,
        seen_hp: Vec<f32>,
        starts: u32,
        #[serde(skip)]
        started: bool,
        events_seen: Vec<usize>,
    }
    impl TypedComponent for Shooter {
        const NAME: &'static str = "Test Shooter";
    }

    impl Shooter {
        fn start_native(&mut self, _ctx: &mut zestengine_core::ecs::ObjectContext<'_, '_>) {
            if self.started { return; }
            self.started = true;
            self.starts += 1;
        }
        fn update_native(&mut self, ctx: &mut zestengine_core::ecs::ObjectContext<'_, '_>) {
            if let Some(h) = ctx.get::<Health>(self.target) {
                self.seen_hp.push(h.hp);
            }
            ctx.edit::<Health>(self.target, |h| h.hp -= 10.0);
            self.events_seen.push(ctx.events::<Ping>().count());
            ctx.emit(Ping);
        }
    }

    #[derive(Serialize, Deserialize, Default)]
    struct Counter {
        #[serde(skip)]
        started: bool,
        starts: u32,
        updates: u32,
        collisions: u32,
        triggers: u32,
    }
    impl TypedComponent for Counter { const NAME: &'static str = "Test Counter"; }

    #[derive(Serialize, Deserialize, Default)]
    struct Spawner {
        template: ObjectRef,
        victim: ObjectRef,
        spawned: Option<ObjectId>,
    }
    impl TypedComponent for Spawner {
        const NAME: &'static str = "Test Spawner";
    }

    impl Spawner {
        fn update_native(&mut self, ctx: &mut zestengine_core::ecs::ObjectContext<'_, '_>) {
            if self.spawned.is_none() {
                let id = ctx.instantiate(self.template, Vec3::new(4.0, 0.0, 0.0), Quat::IDENTITY);
                if let Some(id) = id {
                    ctx.set_active(id, true);
                }
                self.spawned = id;
                ctx.destroy(self.victim);
            }
        }
    }

    fn registry() -> ComponentRegistry {
        let mut r = ComponentRegistry::with_builtins();
        r.register::<Health>();
        use zestengine_core::ecs::{Stage as NativeStage, run_ordered};
        r.register_ecs::<Shooter>();
        r.register_ecs::<Spawner>();
        r.register_ecs::<Flicker>();
        r.register_ecs::<SelfFlicker>();
        r.register_ecs::<WorldDriver>();
        r.register_ecs::<Saver>();
        r.register_system_for::<Shooter>("test::shooter_start", NativeStage::Start, |world, ctx| run_ordered(world, ctx, Shooter::start_native));
        r.register_system_for::<Shooter>("test::shooter", NativeStage::Gameplay, |world, ctx| run_ordered(world, ctx, Shooter::update_native));
        r.register_system_for::<Spawner>("test::spawner", NativeStage::Gameplay, |world, ctx| run_ordered(world, ctx, Spawner::update_native));
        r.register_system_for::<Flicker>("test::flicker", NativeStage::Gameplay, |world, ctx| run_ordered(world, ctx, Flicker::update_native));
        r.register_system_for::<SelfFlicker>("test::self_flicker", NativeStage::Gameplay, |world, ctx| run_ordered(world, ctx, SelfFlicker::update_native));
        r.register_system_for::<WorldDriver>("test::world_driver", NativeStage::Gameplay, |world, ctx| run_ordered(world, ctx, WorldDriver::update_native));
        r.register_system_for::<Saver>("test::save", NativeStage::Gameplay, |world, ctx| run_ordered(world, ctx, Saver::update_native));
        r.register_ecs::<Counter>();
        r.register_system_for::<Counter>("test::counter_start", zestengine_core::ecs::Stage::Start, |world, ctx| {
            for (_, (owner, counter)) in world.query::<(&zestengine_core::ecs::Owner, &mut Counter)>().iter() {
                if !ctx.is_active(world, owner.0) || counter.started { continue; }
                counter.started = true;
                counter.starts += 1;
            }
        });
        r.register_system_for::<Counter>("test::counter_update", zestengine_core::ecs::Stage::Gameplay, |world, ctx| {
            for (_, (owner, counter)) in world.query::<(&zestengine_core::ecs::Owner, &mut Counter)>().iter() {
                if ctx.is_active(world, owner.0) { counter.updates += 1; }
            }
        });
        r.register_system_for::<Counter>("test::counter_contacts", zestengine_core::ecs::Stage::Events, |world, ctx| {
            for (_, (owner, counter)) in world.query::<(&zestengine_core::ecs::Owner, &mut Counter)>().iter() {
                if !ctx.is_active(world, owner.0) { continue; }
                let id = world.object_id(owner.0).unwrap();
                for event in ctx.services.scene_events {
                    if let zestengine_core::ecs::SceneEvent::Contact { object, trigger, phase: zestengine_core::ecs::ContactPhase::Enter, .. } = event
                        && *object == id { if *trigger { counter.triggers += 1; } else { counter.collisions += 1; } }
                }
            }
        });
        r.register::<Spawner>();
        use zestengine_core::ecs::{Owner, Stage};
        r.register_ecs::<PingSender>();
        r.register_ecs::<EventListener>();
        r.register_ecs::<NosyListener>();
        r.register_ecs::<EventLog>();
        r.register_system_for::<EventLog>("test::animation_listener", Stage::Events, |world, ctx| {
            for (_, (owner, log)) in world.query::<(&Owner, &mut EventLog)>().iter() {
                if !ctx.is_active(world, owner.0) { continue; }
                let id = world.object_id(owner.0).unwrap();
                for event in ctx.services.scene_events {
                    if let zestengine_core::ecs::SceneEvent::Animation { object, name } = event
                        && *object == id { log.names.push(name.clone()); }
                }
            }
        });
        r.register_system_for::<PingSender>("test::ping_sender", Stage::Gameplay, |world, ctx| {
            for (_, (owner, sender)) in world.query::<(&Owner, &mut PingSender)>().iter() {
                if !ctx.is_active(world, owner.0) { continue; }
                sender.n += 1;
                ctx.emit(TestPing { n: sender.n });
            }
        });
        r.register_system_for::<EventListener>("test::ping_listener", Stage::Update, |world, ctx| {
            for (_, (owner, listener)) in world.query::<(&Owner, &mut EventListener)>().iter() {
                if !ctx.is_active(world, owner.0) { continue; }
                listener.got.extend(ctx.events::<TestPing>().map(|ping| ping.n));
                listener.saw_in_update.push(listener.got.last().copied().unwrap_or(0));
            }
        });
        r.register_system_for::<NosyListener>("test::other_listener", Stage::Update, |world, ctx| {
            for (_, (owner, listener)) in world.query::<(&Owner, &mut NosyListener)>().iter() {
                if !ctx.is_active(world, owner.0) { continue; }
                listener.updates += 1;
                listener.seen += ctx.events::<u8>().count() as u32;
            }
        });
        r
    }

    #[test]
    fn ecs_motion_runs_once_per_frame_in_the_shared_play_loop() {
        let registry = registry();
        let mut scene = LoadedScene::empty();
        let id = obj(&mut scene, Vec3::ZERO, true, vec![Box::new(zestengine_core::Spin {
            degrees_per_second: 90.0, ..Default::default()
        })]);
        let mut session = PlaySession::new(&scene, Path::new("."), None);
        tick(&mut session, &mut scene, &registry, 60);
        let object = &scene.objects[scene.index_of(id).unwrap()];
        let expected = Quat::from_rotation_y(90.0_f32.to_radians());
        assert!(object.rotation().abs_diff_eq(expected, 1e-4), "ECS motion must not also execute the old update callback");
    }

    #[derive(Default, Serialize, Deserialize)]
    struct EntityDropper {
        dropped: Option<ObjectId>,
    }
    impl TypedComponent for EntityDropper {
        const NAME: &'static str = "Test Entity Dropper";
    }

    #[test]
    fn native_systems_spawn_registered_entities_as_new_root_objects() {
        use zestengine_core::ecs::{EntityShape, EntityTemplate, Stage, run_ordered};
        let mut registry = registry();
        registry.register_ecs::<EntityDropper>();
        registry.register_entity("Test Crate", || EntityTemplate::new(EntityShape::Cube).with(Health { hp: 7.0 }));
        registry.register_system_for::<EntityDropper>("test::entity_dropper", Stage::Gameplay, |world, ctx| {
            run_ordered(world, ctx, |dropper: &mut EntityDropper, obj| {
                if dropper.dropped.is_none() {
                    dropper.dropped = Some(obj.spawn_entity("Test Crate", Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY));
                }
            })
        });
        let mut scene = LoadedScene::empty();
        let dropper = obj(&mut scene, Vec3::ZERO, true, vec![Box::new(EntityDropper::default())]);
        let mut session = PlaySession::new(&scene, Path::new("."), None);
        tick(&mut session, &mut scene, &registry, 3);
        let id = get::<EntityDropper>(&scene, dropper).dropped.expect("the system spawned the entity");
        let index = scene.index_of(id).expect("the new object has the id the system was given");
        let spawned = &scene.objects[index];
        assert_eq!(&*spawned.name(), "Test Crate");
        assert_eq!(spawned.translation(), Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(spawned.components.get_typed::<Health>().unwrap().hp, 7.0);
        assert_eq!(scene.objects.len(), 2, "the entity is spawned exactly once");
    }

    #[test]
    fn spawning_an_unknown_entity_is_logged_and_changes_nothing() {
        let registry = registry();
        let mut scene = LoadedScene::empty();
        let result = scene.spawn_entity(None, &registry, "No Such Entity", ObjectId::new(), Vec3::ZERO, Quat::IDENTITY);
        assert!(result.unwrap_err().to_string().contains("No Such Entity"));
        assert!(scene.objects.is_empty());
    }

    #[derive(Default, Serialize, Deserialize)]
    struct EcsLifecycle {
        fixed: u32,
        update: u32,
        late: u32,
        events: Vec<u32>,
        seen: Vec<usize>,
    }
    impl TypedComponent for EcsLifecycle {
        const NAME: &'static str = "Test ECS Lifecycle";
    }

    #[test]
    fn ecs_stages_replace_frame_callbacks_and_see_events_before_update() {
        use zestengine_core::ecs::{SystemWorld, Stage, SystemContext};
        fn fixed(world: &SystemWorld<'_>, _: &mut SystemContext<'_>) {
            for (_, data) in world.query::<&mut EcsLifecycle>().iter() { data.fixed += 1; }
        }
        fn update(world: &SystemWorld<'_>, ctx: &mut SystemContext<'_>) {
            for (_, data) in world.query::<&mut EcsLifecycle>().iter() {
                data.events.extend(ctx.events::<TestPing>().map(|ping| ping.n));
                data.update += 1;
                data.seen.push(data.events.len());
            }
        }
        fn late(world: &SystemWorld<'_>, _: &mut SystemContext<'_>) {
            for (_, data) in world.query::<&mut EcsLifecycle>().iter() { data.late += 1; }
        }
        let mut registry = registry();
        registry.register_ecs::<EcsLifecycle>();
        registry.register_system("test::fixed", Stage::FixedUpdate, fixed);
        registry.register_system("test::update", Stage::Update, update);
        registry.register_system("test::late", Stage::LateUpdate, late);
        let mut scene = LoadedScene::empty();
        let id = obj(&mut scene, Vec3::ZERO, true, vec![Box::new(EcsLifecycle::default())]);
        obj(&mut scene, Vec3::ZERO, true, vec![Box::new(PingSender::default())]);
        let mut session = PlaySession::new(&scene, Path::new("."), None);
        tick(&mut session, &mut scene, &registry, 3);
        let data = get::<EcsLifecycle>(&scene, id);
        assert_eq!((data.fixed, data.update, data.late), (3, 3, 3));
        assert_eq!(data.events, vec![1, 2]);
        assert_eq!(data.seen, vec![0, 1, 2]);
    }

    fn obj(scene: &mut LoadedScene, pos: Vec3, enabled: bool, components: Vec<Box<dyn Component>>) -> ObjectId {
        let id = ObjectId::new();
        scene.objects.push(SceneObject::from_components(zestengine_core::ecs::ComponentSet::from_vec(components)).with_parent(None).with_component_roles(Default::default()).with_extra_render(Vec::new()).with_source(ObjectSource::Empty).with_net_role(Default::default()).with_render(None).with_static_geometry(false).with_layer(0).with_id(id).with_name(String::new()).with_enabled(enabled).with_transform(zestengine_core::Transform { translation: pos, rotation: Quat::IDENTITY, scale: Vec3::ONE }));
        id
    }

    fn get<T: Component>(scene: &LoadedScene, id: ObjectId) -> zestengine_core::ecs::ComponentRef<'_, T> {
        let i = scene.index_of(id).expect("object exists");
        scene.objects[i]
            .components.get_typed::<T>()
            .expect("component exists")
    }

    fn tick(session: &mut PlaySession, scene: &mut LoadedScene, registry: &ComponentRegistry, frames: usize) {
        let mut shaders = ShaderLibrary::default();
        let input = InputState::default();
        let net = NetView::default();
        let screen = zestengine_core::ScreenInfo::default();
        for _ in 0..frames {
            let host = session.tick(
                &mut World {
                    scene,
                    renderer: None,
                    registry,
                    shaders: &mut shaders,
                },
                &TickInput {
                    dt: 1.0 / 60.0,
                    input: &input,
                    net: &net,
                    listener_fallback: Mat4::IDENTITY,
                    screen: &screen,
                },
            );
            assert!(host.is_empty(), "unexpected host commands {host:?}");
        }
    }

    /// One frame with the given UI input; returns the host commands.
    fn tick_ui(
        session: &mut PlaySession,
        scene: &mut LoadedScene,
        registry: &ComponentRegistry,
        clicks: Vec<ObjectId>,
        cancel: bool,
    ) -> Vec<ComponentCommand> {
        let mut shaders = ShaderLibrary::default();
        let input = InputState::default();
        let net = NetView::default();
        let screen = zestengine_core::ScreenInfo {
            ui_clicks: clicks,
            ui_cancel: cancel,
            ..Default::default()
        };
        session.tick(
            &mut World { scene, renderer: None, registry, shaders: &mut shaders },
            &TickInput { dt: 1.0 / 60.0, input: &input, net: &net, listener_fallback: Mat4::IDENTITY, screen: &screen },
        )
    }

    #[test]
    fn native_test_binding_works_without_legacy_callbacks_and_reads_its_own_source() {
        use zestengine_core::{ArgSource, ArgValue, EventBinding, UiButton, UiText};
        let registry = registry();
        let mut scene = LoadedScene::empty();
        let button = obj(&mut scene, Vec3::ZERO, true, vec![Box::new(UiButton::default())]);
        let text = obj(&mut scene, Vec3::ZERO, true, vec![Box::new(UiText::default())]);
        let mut session = PlaySession::new(&scene, Path::new("."), None);
        let mut shaders = ShaderLibrary::default();
        let input = InputState::default();
        let net = NetView::default();
        let screen = zestengine_core::ScreenInfo::default();
        let mut binding = EventBinding { target: text.into(), component: "UI Text".into(), action: "SetText".into(),
            args: vec![ArgSource::Const(ArgValue::String("Tested".into()))], ..Default::default() };
        let mut run = |scene: &mut LoadedScene, owner, binding: &EventBinding| session.test_binding(
            &mut World { scene, renderer: None, registry: &registry, shaders: &mut shaders },
            &TickInput { dt: 0.0, input: &input, net: &net, listener_fallback: Mat4::IDENTITY, screen: &screen }, owner, binding, None);
        assert!(run(&mut scene, button, &binding).unwrap().is_empty());
        assert_eq!(get::<UiText>(&scene, text).text, "Tested");
        binding.args = vec![ArgSource::Property { object: ObjectRef::NONE, component: "UI Text".into(), property: "text".into() }];
        assert!(run(&mut scene, text, &binding).unwrap().is_empty(), "no unrelated component is needed to execute a test binding");
        assert_eq!(get::<UiText>(&scene, text).text, "Tested", "the test source's own properties are readable");
        binding.action = "Missing".into();
        assert!(run(&mut scene, button, &binding).is_err());
        let button_index = scene.index_of(button).unwrap();
        *scene.objects[button_index].enabled_mut() = false;
        assert!(run(&mut scene, button, &binding).unwrap_err().contains("inactive"));
    }

    #[test]
    fn ui_button_actions_and_pause_menu() {
        use zestengine_core::{UiAction, UiButton, UiPauseMenu};
        let registry = registry();
        let mut scene = LoadedScene::empty();
        let settings = obj(&mut scene, Vec3::ZERO, false, vec![]);
        let panel = obj(&mut scene, Vec3::ZERO, false, vec![]);
        obj(&mut scene, Vec3::ZERO, true, vec![Box::new(UiPauseMenu::default())]);
        let button = obj(
            &mut scene,
            Vec3::ZERO,
            true,
            vec![Box::new(UiButton {
                on_click: vec![UiAction::Show("Settings".into())],
                ..Default::default()
            })],
        );
        *scene.objects[0].name_mut() = "Settings".into();
        *scene.objects[1].name_mut() = "Pause Panel".into();
        let enabled = |scene: &LoadedScene, id: ObjectId| scene.objects[scene.index_of(id).unwrap()].enabled();
        let mut session = PlaySession::new(&scene, Path::new("."), None);
        tick_ui(&mut session, &mut scene, &registry, vec![], false);

        tick_ui(&mut session, &mut scene, &registry, vec![button], false);
        assert!(enabled(&scene, settings), "Show action ran on click");

        tick_ui(&mut session, &mut scene, &registry, vec![], true);
        assert!(enabled(&scene, panel), "cancel opened the pause panel");
        let host = tick_ui(&mut session, &mut scene, &registry, vec![], false);
        assert_eq!(session.time_scale, 0.0, "open pause menu stops time");
        assert!(host.iter().any(|c| matches!(c, ComponentCommand::SetCursorLocked(false))));

        tick_ui(&mut session, &mut scene, &registry, vec![], true);
        tick_ui(&mut session, &mut scene, &registry, vec![], false);
        assert!(!enabled(&scene, panel));
        assert_eq!(session.time_scale, 1.0, "closing resumes");
    }

    /// One frame with live bindings: the runtime sees `raw`, the session
    /// ticks with its view, and binding requests go back to the runtime.
    fn tick_controls(
        session: &mut PlaySession,
        scene: &mut LoadedScene,
        registry: &ComponentRegistry,
        rt: &mut zestengine_core::ControlsRuntime,
        raw: &zestengine_core::RawInput,
        clicks: Vec<ObjectId>,
    ) {
        rt.frame(raw, false);
        let mut shaders = ShaderLibrary::default();
        let input = InputState::default();
        let net = NetView::default();
        let screen = zestengine_core::ScreenInfo { ui_clicks: clicks, controls: rt.info(), ..Default::default() };
        let host = session.tick(
            &mut World { scene, renderer: None, registry, shaders: &mut shaders },
            &TickInput { dt: 1.0 / 60.0, input: &input, net: &net, listener_fallback: Mat4::IDENTITY, screen: &screen },
        );
        for c in host {
            if let ComponentCommand::Input(c) = c {
                rt.apply(c).unwrap();
            }
        }
    }

    #[test]
    fn controls_menu_rebinds_through_capture() {
        use zestengine_core::{InputSource, Key, UiText};
        let registry = registry();
        let mut scene = LoadedScene::empty();
        for node in crate::ui_templates::UiTemplate::Controls.build() {
            obj(&mut scene, Vec3::ZERO, node.active, node.components);
            let o = scene.objects.last_mut().unwrap();
            *o.name_mut() = node.name;
            o.set_parent(node.parent);
        }
        let id = |scene: &LoadedScene, n: &str| scene.objects.iter().find(|o| o.name() == n).unwrap().id();
        let text = |scene: &LoadedScene, n: &str| {
            let o = scene.objects.iter().find(|o| o.name() == n).unwrap();
            o.components.get_typed::<UiText>().unwrap().text.clone()
        };
        let mut rt = zestengine_core::ControlsRuntime::new(zestengine_core::Bindings::default());
        let mut session = PlaySession::new(&scene, Path::new("."), None);
        let none = zestengine_core::RawInput::default();
        tick_controls(&mut session, &mut scene, &registry, &mut rt, &none, vec![]);
        assert_eq!(text(&scene, "Controls Scheme"), "KB+Mouse (in use)");
        assert_eq!(text(&scene, "Controls Row 1 Label"), "Move Up");
        assert_eq!(text(&scene, "Controls Row 6 Label"), "Jump");
        assert_eq!(text(&scene, "Controls Row 6 Bind 1"), "Space");

        let jump = id(&scene, "Controls Row 6 Bind 1");
        tick_controls(&mut session, &mut scene, &registry, &mut rt, &none, vec![jump]);
        assert_eq!(text(&scene, "Controls Row 6 Bind 1"), "...");
        tick_controls(&mut session, &mut scene, &registry, &mut rt, &none, vec![]);
        let mut raw = zestengine_core::RawInput::default();
        raw.keys_down.insert(Key::J);
        tick_controls(&mut session, &mut scene, &registry, &mut rt, &raw, vec![]);
        tick_controls(&mut session, &mut scene, &registry, &mut rt, &none, vec![]);
        let kb = &rt.bindings().schemes[0];
        assert_eq!(kb.action("Jump").unwrap().buttons, vec![InputSource::Key(Key::J)], "rebound by pressing J");
        assert_eq!(text(&scene, "Controls Row 6 Bind 1"), "J");
        assert_eq!(rt.overrides().actions.len(), 1, "saved as a player override");

        // Clicks wait out the capture's mouse release first.
        for _ in 0..20 {
            tick_controls(&mut session, &mut scene, &registry, &mut rt, &none, vec![]);
        }
        let reset = id(&scene, "Controls Reset");
        tick_controls(&mut session, &mut scene, &registry, &mut rt, &none, vec![reset]);
        tick_controls(&mut session, &mut scene, &registry, &mut rt, &none, vec![]);
        assert_eq!(text(&scene, "Controls Row 6 Bind 1"), "Space", "reset to the game's defaults");
    }

    #[test]
    fn get_modify_and_events_between_components() {
        let registry = registry();
        let mut scene = LoadedScene::empty();
        let target = obj(&mut scene, Vec3::ZERO, true, vec![Box::new(Health::default())]);
        let shooter = obj(
            &mut scene,
            Vec3::ZERO,
            true,
            vec![Box::new(Shooter {
                target: target.into(),
                ..Default::default()
            })],
        );
        let mut session = PlaySession::new(&scene, Path::new("."), None);
        tick(&mut session, &mut scene, &registry, 3);
        let s = get::<Shooter>(&scene, shooter);
        assert_eq!(s.starts, 1);
        assert_eq!(s.seen_hp, vec![100.0, 90.0, 80.0]);
        assert_eq!(s.events_seen, vec![0, 1, 1]);
        assert_eq!(get::<Health>(&scene, target).hp, 70.0);
    }

    /// A plugin system reads and edits a built-in typed light.
    #[test]
    fn a_plugin_component_reads_and_modifies_a_builtin_light() {
        let registry = registry();
        let mut scene = LoadedScene::empty();
        let lamp = obj(
            &mut scene,
            Vec3::ZERO,
            true,
            vec![Box::new(zestengine_core::LightComponent {
                intensity: 4.0,
                ..Default::default()
            })],
        );
        let flicker = obj(
            &mut scene,
            Vec3::ZERO,
            true,
            vec![Box::new(Flicker {
                target: lamp.into(),
                ..Default::default()
            })],
        );
        let mut session = PlaySession::new(&scene, Path::new("."), None);
        tick(&mut session, &mut scene, &registry, 3);
        let f = get::<Flicker>(&scene, flicker);
        assert_eq!(f.seen, vec![4.0, 2.0, 1.0], "reads the authored intensity each frame");
        assert_eq!(get::<zestengine_core::LightComponent>(&scene, lamp).intensity, 0.5, "native edit updates the typed value");
    }

    /// The full world-control surface, in the user's flicker terms: a plugin
    /// component on the lamp reads ITS OWN material, halves the material's
    /// emissiveness AND the light's brightness (shader + light control), a
    /// driver component teleports a parented "player" in world space, reads a
    /// referenced object's material from the outside, and switches the active
    /// camera.
    #[test]
    fn a_plugin_flickers_materials_lights_teleports_and_switches_cameras() {
        let registry = registry();
        let mut scene = LoadedScene::empty();
        // The lamp's material (slot 0): an emissive surface to flicker.
        let model = crate::scene::model_from_material(
            "plug",
            &zestengine_core::MaterialParams { emission_intensity: 3.0, ..Default::default() },
            &zestengine_render::MaterialFeatures::default(),
            false,
        );
        scene.materials.push(MaterialEntry {
            model: model.clone(),
            default: model.clone(),
            handle: zestengine_render::MaterialHandle::NONE,
            emission_map_mean: [1.0; 3],
            bake_albedo: [1.0; 3],
            bake_metallic: 0.0,
            file: None,
            embedded_textures: false,
            base_textures: TexSlots::default(),
            bound_textures: TexSlots::default(),
            base_tex_data: Default::default(),
        });
        let lamp = obj(
            &mut scene,
            Vec3::ZERO,
            true,
            vec![
                Box::new(zestengine_core::LightComponent { intensity: 4.0, ..Default::default() }),
                Box::new(SelfFlicker::default()),
            ],
        );
        let lamp_i = scene.index_of(lamp).unwrap();
        *scene.objects[lamp_i].render_mut() = Some(RenderInfo { mesh: 0, material: 0 });
        *scene.objects[lamp_i].extra_render_mut() = vec![RenderInfo { mesh: 0, material: 0 }];
        // A parented "player" to teleport (parent sits at y = 10).
        let parent = obj(&mut scene, Vec3::new(0.0, 10.0, 0.0), true, vec![]);
        let player = obj(&mut scene, Vec3::new(0.0, 1.5, 0.0), true, vec![]);
        let parent_i = scene.index_of(parent).unwrap();
        let player_i = scene.index_of(player).unwrap();
        scene.objects[player_i].set_parent(Some(parent_i));
        // A camera to switch to.
        let cam = obj(&mut scene, Vec3::new(5.0, 2.0, 5.0), true, vec![]);
        let driver = obj(
            &mut scene,
            Vec3::ZERO,
            true,
            vec![Box::new(WorldDriver {
                lamp: lamp.into(),
                player: player.into(),
                cam: cam.into(),
                ..Default::default()
            })],
        );
        let mut session = PlaySession::new(&scene, Path::new("."), None);
        tick(&mut session, &mut scene, &registry, 1);
        scene.objects.swap(0, 4);
        tick(&mut session, &mut scene, &registry, 2);

        // The user's flicker: emissiveness on the material + brightness on the light.
        let f = get::<SelfFlicker>(&scene, lamp);
        assert_eq!(f.seen_emission, vec![3.0, 1.5, 0.75], "reads its own material each frame");
        assert_eq!(f.seen_light, vec![4.0, 2.0, 1.0], "reads its own light each frame");
        assert_eq!(scene.materials[0].model.emission_intensity, 0.375, "deferred material edit lands in the scene material");
        assert_eq!(get::<zestengine_core::LightComponent>(&scene, lamp).intensity, 0.5, "native light edits persist");

        // Materials of PASSED-IN objects, read from another component.
        let d = get::<WorldDriver>(&scene, driver);
        assert_eq!(d.seen_lamp_emission, vec![3.0, 1.5, 0.75], "a component reads a referenced object's material");

        assert_eq!(d.seen_extra_emission, vec![3.0, 1.5, 0.75], "extra slots use native render metadata after row reorder");
        assert!(d.invalid_slot_empty, "missing slots have no material");

        // Teleport: world (2, 0, 3) under a parent at y=10 -> local, then world again.
        let pw = scene.world_transform(scene.index_of(player).unwrap()).w_axis.truncate();
        assert!(pw.distance(Vec3::new(2.0, 0.0, 3.0)) < 1e-4, "world teleport lands at the asked position (got {pw:?})");
        assert_eq!(d.last_player_pos.map(|p| Vec3::from_array(p).distance(Vec3::new(2.0, 0.0, 3.0)) < 1e-4), Some(true), "the ctx view agrees");

        // Camera switching.
        assert_eq!(scene.active_camera_override, Some(cam), "the referenced camera became active");
    }

    #[test]
    fn desktop_fallback_pitch_looks_up_and_down() {
        let registry = registry();
        let mut scene = LoadedScene::empty();
        let head = obj(
            &mut scene,
            Vec3::new(0.0, 1.65, 0.0),
            true,
            vec![Box::new(zestengine_core::CameraComponent::default())],
        );
        let head_i = scene.index_of(head).unwrap();
        obj(
            &mut scene,
            Vec3::ZERO,
            true,
            vec![Box::new(zestengine_core::VrPlayer {
                head: head.into(),
                desktop_fallback: true,
                ..Default::default()
            })],
        );
        let mut session = PlaySession::new(&scene, Path::new("."), None);
        // A pure vertical Look delta for two frames (start + first update).
        let look = InputState::from_actions(
            [(
                "Look".to_owned(),
                zestengine_core::ActionValue { down: false, just_pressed: false, just_released: false, value: 20.0, axis2: glam::Vec2::new(0.0, 20.0) },
            )],
            glam::Vec2::ZERO,
        );
        let mut shaders = ShaderLibrary::default();
        let net = NetView::default();
        let screen = zestengine_core::ScreenInfo::default();
        for _ in 0..2 {
            let host = session.tick(
                &mut World {
                    scene: &mut scene,
                    renderer: None,
                    registry: &registry,
                    shaders: &mut shaders,
                },
                &TickInput { dt: 1.0 / 60.0, input: &look, net: &net, listener_fallback: Mat4::IDENTITY, screen: &screen },
            );
            assert!(
                host.iter().all(|c| matches!(c, ComponentCommand::SetCursorLocked(_))),
                "only the cursor lock is expected, got {host:?}"
            );
        }
        let head_rot = scene.objects[head_i].rotation();
        assert!(
            head_rot.x.abs() > 1e-3,
            "the head/camera pitches on a vertical Look (got {head_rot:?})"
        );
    }

    #[test]
    fn instantiate_destroy_and_activation() {
        let registry = registry();
        let mut scene = LoadedScene::empty();
        let template = obj(&mut scene, Vec3::ZERO, false, vec![Box::new(Counter::default())]);
        let victim = obj(&mut scene, Vec3::ZERO, true, vec![Box::new(Counter::default())]);
        let spawner = obj(
            &mut scene,
            Vec3::ZERO,
            true,
            vec![Box::new(Spawner {
                template: template.into(),
                victim: victim.into(),
                spawned: None,
            })],
        );
        let mut session = PlaySession::new(&scene, Path::new("."), None);
        tick(&mut session, &mut scene, &registry, 3);
        assert!(scene.index_of(victim).is_none(), "victim destroyed");
        let clone = get::<Spawner>(&scene, spawner).spawned.expect("spawned");
        let c = get::<Counter>(&scene, clone);
        assert_eq!(c.starts, 1, "clone started once when activated");
        assert_eq!(c.updates, 2, "clone updates from the frame after it was spawned");
        let t = get::<Counter>(&scene, template);
        assert_eq!((t.starts, t.updates), (0, 0), "disabled template never runs");
        let i = scene.index_of(clone).unwrap();
        assert_eq!(scene.objects[i].translation(), Vec3::new(4.0, 0.0, 0.0));
    }

    #[test]
    fn collision_and_trigger_callbacks() {
        let registry = registry();
        let mut scene = LoadedScene::empty();
        obj(
            &mut scene,
            Vec3::ZERO,
            true,
            vec![Box::new(BoxCollider {
                size: [20.0, 1.0, 20.0],
                ..Default::default()
            })],
        );
        let zone = obj(
            &mut scene,
            Vec3::new(0.0, 2.0, 0.0),
            true,
            vec![
                Box::new(BoxCollider {
                    size: [2.0, 1.0, 2.0],
                    is_trigger: true,
                    ..Default::default()
                }),
                Box::new(Counter::default()),
            ],
        );
        let ball = obj(
            &mut scene,
            Vec3::new(0.0, 4.0, 0.0),
            true,
            vec![
                Box::new(SphereCollider::default()),
                Box::new(RigidBody::default()),
                Box::new(Counter::default()),
            ],
        );
        let mut session = PlaySession::new(&scene, Path::new("."), None);
        tick(&mut session, &mut scene, &registry, 180);
        let b = get::<Counter>(&scene, ball);
        assert_eq!(b.collisions, 1, "ball hit the floor once");
        assert_eq!(b.triggers, 1, "ball passed through the zone once");
        assert_eq!(get::<Counter>(&scene, zone).triggers, 1);
    }

    #[derive(Serialize, Deserialize, Default)]
    struct EventLog {
        names: Vec<String>,
    }
    impl TypedComponent for EventLog {
        const NAME: &'static str = "Test Event Log";
    }

    #[test]
    fn native_binding_arguments_resolve_before_any_command_in_the_batch_is_applied() {
        use zestengine_core::{ArgSource, ArgValue, EventBinding, FireSource, LightComponent};
        let registry = registry();
        let mut scene = LoadedScene::empty();
        let light = obj(&mut scene, Vec3::ZERO, true, vec![Box::new(LightComponent { intensity: 1.0, ..Default::default() })]);
        let first = EventBinding { target: light.into(), component: "Light".into(), action: "SetIntensity".into(), args: vec![ArgSource::Const(ArgValue::Float(5.0))], ..Default::default() };
        let second = EventBinding { args: vec![ArgSource::Property { object: light.into(), component: "Light".into(), property: "intensity".into() }], ..first.clone() };
        let mut commands = Vec::new();
        for binding in [&first, &second] { zestengine_core::queue_actions(std::slice::from_ref(binding), &FireSource { source: light, instigator: None, payload: &[] }, &mut commands); }
        let mut session = PlaySession::new(&scene, Path::new("."), None);
        session.apply(&mut World { scene: &mut scene, renderer: None, registry: &registry, shaders: &mut ShaderLibrary::default() }, commands, &mut Vec::new());
        assert_eq!(get::<LightComponent>(&scene, light).intensity, 1.0, "both sources resolve against the state before command application");
    }

    #[test]
    fn native_event_destruction_waits_until_all_contact_consumers_finish() {
        use zestengine_core::{ActionCtx, ActionDesc, ArgValue, EventBinding, TriggerEvents};
        #[derive(Default, Serialize, Deserialize)]
        struct RemoveVisitor;
        static REMOVE_ACTIONS: &[ActionDesc] = &[ActionDesc::new("Remove", "Remove", &[], true)];
        impl TypedComponent for RemoveVisitor {
            const NAME: &'static str = "Test Remove Visitor";
            fn actions(&self) -> &'static [ActionDesc] { REMOVE_ACTIONS }
            fn invoke_action(&mut self, context: &mut ActionCtx<'_>, _: &str, _: &[ArgValue]) -> Result<(), String> {
                context.commands.push(ComponentCommand::Destroy(context.instigator.ok_or("missing visitor")?));
                Ok(())
            }
        }
        struct ContactSeen(ObjectId);
        #[derive(Default, Serialize, Deserialize)]
        struct Notify;
        impl TypedComponent for Notify {
            const NAME: &'static str = "Test Contact Notify";
        }
        let mut registry = registry();
        registry.register_ecs::<RemoveVisitor>();
        registry.register_ecs::<Notify>();
        registry.register_system_for::<Notify>("test::contact_notify", zestengine_core::ecs::Stage::Events, |world, context| {
            for (_, (owner, _)) in world.query::<(&zestengine_core::ecs::Owner, &Notify)>().iter() {
                if !context.is_active(world, owner.0) { continue; }
                let id = world.object_id(owner.0).unwrap();
                for event in context.services.scene_events {
                    if let zestengine_core::ecs::SceneEvent::Contact { object, trigger: true, phase: zestengine_core::ecs::ContactPhase::Enter, .. } = event
                        && *object == id { context.emit(ContactSeen(id)); }
                }
            }
        });
        let mut scene = LoadedScene::empty();
        let zone = obj(&mut scene, Vec3::ZERO, true, vec![Box::new(RemoveVisitor), Box::new(Notify), Box::new(TriggerEvents {
            on_enter: vec![EventBinding { component: RemoveVisitor::NAME.into(), action: "Remove".into(), ..Default::default() }], ..Default::default()
        })]);
        let visitor = obj(&mut scene, Vec3::ZERO, true, vec![Box::new(Notify)]);
        let survivor = obj(&mut scene, Vec3::ZERO, true, vec![Box::new(Notify)]);
        let mut session = PlaySession::new(&scene, Path::new("."), None);
        let mut shaders = ShaderLibrary::default();
        let input = InputState::default();
        let net = NetView::default();
        let screen = zestengine_core::ScreenInfo::default();
        session.deliver_contacts(&mut World { scene: &mut scene, renderer: None, registry: &registry, shaders: &mut shaders },
            &TickInput { dt: 1.0 / 60.0, input: &input, net: &net, listener_fallback: Mat4::IDENTITY, screen: &screen },
            1.0 / 60.0, 1.0 / 60.0, &[], &mut Vec::new(), &[ContactEvent { a: zone, b: visitor, started: true, trigger: true }]);
        assert!(scene.index_of(visitor).is_none());
        assert!(scene.index_of(survivor).is_some());
        let seen: Vec<_> = session.events_next.iter().filter_map(|event| event.0.downcast_ref::<ContactSeen>().map(|event| event.0)).collect();
        assert!(seen.contains(&zone) && seen.contains(&visitor), "both sides are notified before destruction");
        assert!(!seen.contains(&survivor), "row removal cannot misroute this contact to the next object");
    }

    #[test]
    fn animation_events_and_root_motion_are_applied() {
        let mut registry = registry();
        registry.register_ecs::<EventLog>();
        let mut scene = LoadedScene::empty();
        let mut animator = zestengine_core::Animator::default();
        animator.state.fired = vec!["Footstep".into()];
        animator.state.root_delta = Vec3::new(0.0, 0.0, 1.5);
        let light = obj(&mut scene, Vec3::ZERO, true, vec![Box::new(zestengine_core::LightComponent::default())]);
        let id = obj(&mut scene, Vec3::ZERO, true, vec![Box::new(animator), Box::new(EventLog::default()), Box::new(zestengine_core::AnimationEventBindings {
            events: vec![zestengine_core::AnimationEventBinding { event: "Footstep".into(), actions: vec![zestengine_core::EventBinding {
                target: light.into(), component: "Light".into(), action: "SetIntensity".into(), args: vec![zestengine_core::ArgSource::Const(zestengine_core::ArgValue::Float(4.0))], ..Default::default()
            }] }],
        })]);
        let mut session = PlaySession::new(&scene, Path::new("."), None);
        tick(&mut session, &mut scene, &registry, 1);
        assert_eq!(get::<EventLog>(&scene, id).names, vec!["Footstep".to_string()]);
        assert_eq!(scene.objects[scene.index_of(id).unwrap()].translation(), Vec3::new(0.0, 0.0, 1.5));
        assert_eq!(get::<zestengine_core::LightComponent>(&scene, light).intensity, 4.0, "native bindings run alongside legacy event observers");
        assert!(get::<zestengine_core::Animator>(&scene, id).state.fired.is_empty());
    }

    #[test]
    fn anim_clips_drive_child_transforms_and_params_follow_drivers() {
        use zestengine_assets::{AnimClip, AnimTrack, KeyInterp, Keyframe, TrackProperty};
        let registry = registry();
        let mut scene = LoadedScene::empty();
        let leader = obj(&mut scene, Vec3::ZERO, true, vec![Box::new(zestengine_core::Animator::default())]);
        let follower = zestengine_core::Animator {
            clip: "wave.anim".into(),
            drivers: vec![zestengine_core::ParamDriver {
                parameter: "Mood".into(),
                source: zestengine_core::DriverSource::AnimatorParameter,
                object: ObjectRef::new(leader),
                source_parameter: "Mood".into(),
                scale: 2.0,
            }],
            ..Default::default()
        };
        let root = obj(&mut scene, Vec3::ZERO, true, vec![Box::new(follower)]);
        obj(&mut scene, Vec3::ZERO, true, vec![]);
        *scene.objects[2].name_mut() = "Arm".into();
        scene.objects[2].set_parent(Some(1));
        scene.anim_assets.set_clip(
            "wave.anim",
            AnimClip {
                length: 1.0,
                tracks: vec![AnimTrack {
                    target: "Arm".into(),
                    property: TrackProperty::Translation,
                    name: String::new(),
                    slot: 0,
                    keys: vec![
                        Keyframe { time: 0.0, value: [0.0; 4], interp: KeyInterp::Linear, ..Default::default() },
                        Keyframe { time: 1.0, value: [0.0, 6.0, 0.0, 0.0], interp: KeyInterp::Linear, ..Default::default() },
                    ],
                }],
                ..Default::default()
            },
        );
        let i = scene.index_of(leader).unwrap();
        scene.objects[i]
            .components.get_typed_mut::<zestengine_core::Animator>()
            .unwrap()
            .set_float("Mood", 1.5);
        let mut session = PlaySession::new(&scene, Path::new("."), None);
        tick(&mut session, &mut scene, &registry, 30);
        let y = scene.objects[2].translation().y;
        assert!((y - 3.0).abs() < 0.05, "arm moved by the clip: {y}");
        assert_eq!(scene.objects[1].translation(), Vec3::ZERO, "root not animated by the clip");
        assert_eq!(get::<zestengine_core::Animator>(&scene, root).get_float("Mood"), 3.0);
    }

    #[derive(Serialize, Deserialize, Default)]
    struct Saver {
        frame: u32,
        loaded_score: Option<i64>,
        greeting: String,
    }
    impl TypedComponent for Saver {
        const NAME: &'static str = "Test Saver";
    }

    impl Saver {
        fn update_native(&mut self, ctx: &mut zestengine_core::ecs::ObjectContext<'_, '_>) {
            match self.frame {
                0 => {
                    ctx.save_value("score", 42i64);
                    ctx.write_save("slot1");
                }
                1 => {
                    ctx.save_value("score", 7i64);
                    ctx.load_save("slot1");
                }
                _ => {
                    self.loaded_score = ctx.save_data().and_then(|save| save.int("score"));
                    self.greeting = ctx.tr("hello").to_owned();
                }
            }
            self.frame += 1;
        }
    }

    #[test]
    fn save_slots_and_localization() {
        let mut registry = registry();
        registry.register::<Saver>();
        let dir = std::env::temp_dir().join(format!("zestengine-save-test-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("localization")).unwrap();
        std::fs::write(
            dir.join("localization/strings.loc"),
            r#"Localization(languages: ["en", "fr"], entries: {"hello": ["Hello", "Bonjour"]})"#,
        )
        .unwrap();
        let mut scene = LoadedScene::empty();
        let id = obj(&mut scene, Vec3::ZERO, true, vec![Box::new(Saver::default())]);
        let mut session = PlaySession::new(&scene, &dir, None);
        std::sync::Arc::make_mut(&mut session.strings).set_language("fr");
        tick(&mut session, &mut scene, &registry, 3);
        let s = get::<Saver>(&scene, id);
        assert_eq!(s.loaded_score, Some(42), "slot restored the written value");
        assert_eq!(s.greeting, "Bonjour");
        assert!(dir.join(".zestengine/saves/slot1.save").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn mod_save_load_rejects_incompatible_contents_without_replacing_live_state() {
        use zestengine_core::mods::{ModLock,LockedMod};
        for mismatch in [false,true]{
            let mut registry=registry();registry.register::<Saver>();
            let dir=std::env::temp_dir().join(format!("zest-mod-save-{}",ObjectId::new()));
            let mut scene=LoadedScene::empty();
            scene.mod_lock=ModLock{game_id:"test".into(),game_api:1,mods:vec![LockedMod{id:ObjectId::new().to_string(),version:"1".into(),content_hash:"abc".into()}]};
            let id=obj(&mut scene,Vec3::ZERO,true,vec![Box::new(Saver::default())]);
            let mut session=PlaySession::new(&scene,&dir,None);
            tick(&mut session,&mut scene,&registry,1);
            let path=dir.join(".zestengine/saves/slot1.save");
            let mut saved=zestengine_core::SaveGame::load(&path).unwrap();assert_eq!(saved.mods,scene.mod_lock);
            if mismatch{saved.mods.mods[0].content_hash="different".into();saved.save(&path).unwrap();}
            tick(&mut session,&mut scene,&registry,2);
            assert_eq!(get::<Saver>(&scene,id).loaded_score,Some(if mismatch{7}else{42}));
            let _=std::fs::remove_dir_all(dir);
        }
    }

    #[test]
    fn scene_portal_preloads_then_switches_with_kept_activator() {
        let registry = registry();
        let mut scene = LoadedScene::empty();
        obj(
            &mut scene,
            Vec3::new(0.0, 0.0, 0.0),
            true,
            vec![
                Box::new(BoxCollider { size: [2.0, 2.0, 2.0], is_trigger: true, ..Default::default() }),
                Box::new({
                    let mut p = zestengine_core::ScenePortal::default();
                    p.scene = "scenes/level2.scene".into();
                    p.preload_distance = 10.0;
                    p
                }),
            ],
        );
        let player = obj(
            &mut scene,
            Vec3::new(0.0, 0.0, 20.0),
            true,
            vec![
                Box::new(SphereCollider::default()),
                Box::new(RigidBody { kind: zestengine_core::BodyKind::Kinematic, ..Default::default() }),
                Box::new({
                    let mut p = zestengine_core::Pawn::default();
                    p.possessed = false;
                    p
                }),
            ],
        );
        let mut session = PlaySession::new(&scene, Path::new("."), None);
        let mut shaders = ShaderLibrary::default();
        let input = InputState::default();
        let net = NetView::default();
        let screen = zestengine_core::ScreenInfo::default();
        let mut run = |scene: &mut LoadedScene, session: &mut PlaySession| {
            session.tick(
                &mut World { scene, renderer: None, registry: &registry, shaders: &mut shaders },
                &TickInput { dt: 1.0 / 60.0, input: &input, net: &net, listener_fallback: Mat4::IDENTITY, screen: &screen },
            )
        };
        let host = run(&mut scene, &mut session);
        assert!(host.is_empty(), "too far to preload yet: {host:?}");
        scene.objects[1].transform_mut().translation = Vec3::new(0.0, 0.0, 8.0);
        let host = run(&mut scene, &mut session);
        assert!(matches!(host.as_slice(), [ComponentCommand::PreloadScene(p)] if p == "scenes/level2.scene"), "{host:?}");
        scene.objects[1].transform_mut().translation = Vec3::ZERO;
        let mut all = Vec::new();
        for _ in 0..3 {
            all.extend(run(&mut scene, &mut session));
        }
        assert!(all.iter().any(|c| matches!(c, ComponentCommand::KeepOnSceneChange(id, true) if *id == player)), "{all:?}");
        assert!(all.iter().any(|c| matches!(c, ComponentCommand::LoadScene(p) if p == "scenes/level2.scene")));
    }

    #[test]
    fn time_scale_zero_freezes_physics() {
        let registry = registry();
        let mut scene = LoadedScene::empty();
        obj(
            &mut scene,
            Vec3::new(0.0, 4.0, 0.0),
            true,
            vec![Box::new(SphereCollider::default()), Box::new(RigidBody::default())],
        );
        let mut session = PlaySession::new(&scene, Path::new("."), None);
        session.time_scale = 0.0;
        tick(&mut session, &mut scene, &registry, 30);
        assert_eq!(scene.objects[0].translation().y, 4.0);
        session.time_scale = 1.0;
        tick(&mut session, &mut scene, &registry, 30);
        assert!(scene.objects[0].translation().y < 4.0);
    }
}
