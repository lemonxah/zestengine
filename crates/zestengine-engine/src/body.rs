//! Body tracking for VR players: tracker devices and their roles, the
//! player's measured height and arm span (saved per user in `body.ron` in the
//! user's config folder, shared by every ZestEngine game), fitting avatars to the
//! player (Scale avatar / Scale player / Off), and the guided calibration
//! (stand straight, T-pose, floor check) behind the VR body menu.
//!
//! Each frame the host builds the raw stage-space targets with
//! [`BodyTracking::assemble`] (headset, controllers, every tracker by its
//! role), then [`BodyTracking::update`] runs the calibration steps, fits the
//! scene's avatars and turns the raw poses into calibrated bone targets for
//! Humanoid IK. The tracker offsets are solved from the stored T-pose for
//! whichever avatar is worn, so changing avatars needs no new T-pose.

use std::collections::HashMap;
use std::path::PathBuf;

use zestengine_core::{
    AvatarScaling, BodyCommand, BodyFit, BodyInfo, BodyProfile, FIRST_BODY_SLOT, HeightSampler, ObjectId, TRACKER_SLOTS,
    TrackerTargets,
};
use glam::{Mat4, Quat, Vec3};

use crate::humanoid::{self, AvatarMeasure, BodyCalibration, TrackerPoses};
use crate::scene::LoadedScene;

mod panel;
pub use panel::BodyPanel;

/// Countdown before each capture (s).
const COUNTDOWN: f32 = 3.0;
/// How long the T-pose is held before it is captured (s).
const TPOSE_HOLD: f32 = 0.6;
/// A controller resting on the floor holds its tracked point this high (m).
const CONTROLLER_ON_FLOOR: f32 = 0.035;

/// One tracking device that can drive a body role: a runtime tracker (by its
/// role path) or a SlimeVR tracker (by serial).
#[derive(Clone, Debug, PartialEq)]
pub struct TrackerDevice {
    /// Stable id the role choice is saved under.
    pub id: String,
    pub label: String,
    /// The role the device reports for itself.
    pub native: Option<usize>,
    /// Stage-space pose (floor corrected once assembled).
    pub pose: Option<(Vec3, Quat)>,
}

/// OpenXR runtime role paths (`XR_HTCX_vive_tracker_interaction`) and their slots.
const XR_ROLES: [(&str, &str, usize); 8] = [
    ("waist", "Waist", 3),
    ("left_foot", "Left foot", 4),
    ("right_foot", "Right foot", 5),
    ("chest", "Chest", 6),
    ("left_elbow", "Left elbow", 7),
    ("right_elbow", "Right elbow", 8),
    ("left_knee", "Left knee", 9),
    ("right_knee", "Right knee", 10),
];

/// The runtime's body trackers this frame (one per tracker role path).
pub fn xr_devices(b: &zestengine_xr::BodyPoses) -> Vec<TrackerDevice> {
    let poses = [b.hips, b.left_foot, b.right_foot, b.chest, b.left_elbow, b.right_elbow, b.left_knee, b.right_knee];
    XR_ROLES
        .iter()
        .zip(poses)
        .filter(|(_, p)| p.is_some())
        .map(|(&(path, name, slot), pose)| TrackerDevice {
            id: format!("xr:{path}"),
            label: format!("Tracker ({name})"),
            native: Some(slot),
            pose,
        })
        .collect()
}

/// `body.ron` in the user's config folder (shared by every game).
pub fn body_profile_path() -> Option<PathBuf> {
    if cfg!(test) {
        return None;
    }
    crate::input_engine::user_config_dir().map(|d| d.join("ZestEngine").join("body.ron"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Flow {
    /// Manual alignment to the worn avatar, accepted with both triggers.
    Align,
    /// Stand straight, T-pose (assigns roles), floor check.
    Full,
    /// Stand straight only.
    Height,
    /// T-pose only, keeping the roles.
    Quick,
}

#[derive(Clone, Debug)]
enum Step {
    /// Armed only after both triggers have been released.
    Align(bool),
    StandCountdown(f32),
    Standing(HeightSampler),
    TPoseCountdown(f32),
    TPoseHold(f32),
    /// Waiting for a controller on the floor (its trigger) or Skip.
    Floor,
}

#[derive(Clone, Debug)]
struct Calibration {
    flow: Flow,
    step: Step,
}

/// Offsets solved for one avatar at one scale.
#[derive(Clone, Copy, Debug)]
struct Solved {
    avatar: (ObjectId, usize),
    scale: f32,
    from_tpose: bool,
    cal: BodyCalibration,
}

/// A scale this module wrote onto an object (so the authored one can be
/// recovered, and an outside change noticed).
#[derive(Clone, Copy, Debug)]
struct Applied {
    factor: f32,
    written: Vec3,
}

/// Per-frame input of [`BodyTracking::update`].
#[derive(Clone, Copy, Debug, Default)]
pub struct BodyFrame {
    pub dt: f32,
    /// A headset is running.
    pub tracking: bool,
    /// Avatars and the VR Player may be rescaled (Play / a shipped game).
    pub apply: bool,
    /// Controller triggers held (left, right).
    pub triggers: [bool; 2],
}

/// The local player's body: devices, roles, measurements, calibration and
/// the fit of the scene's avatars.
pub struct BodyTracking {
    pub(crate) profile: BodyProfile,
    path: Option<PathBuf>,
    pub(crate) devices: Vec<TrackerDevice>,
    hmd: [Option<(Vec3, Quat)>; 3],
    calibration: Option<Calibration>,
    /// `ctx.calibrate_trackers()`: capture the T-pose on the next update.
    instant: bool,
    solved: Option<Solved>,
    pub(crate) fit: BodyFit,
    pub(crate) mode: AvatarScaling,
    pub(crate) avatar: Option<AvatarMeasure>,
    applied: HashMap<ObjectId, Applied>,
    /// The VR body menu (Calibrate / Height / Done) is open.
    pub menu_open: bool,
    /// The editor's desktop Body Tracking window is open.
    pub window_open: bool,
    pub(crate) status: Option<String>,
    triggers_prev: [bool; 2],
    pub(crate) last_raw: TrackerTargets,
    pub(crate) panel: BodyPanel,
}

impl Default for BodyTracking {
    fn default() -> Self {
        Self::with_profile(BodyProfile::default(), None)
    }
}

impl BodyTracking {
    /// The player's saved body (or a fresh one).
    pub fn load() -> Self {
        let path = body_profile_path();
        let profile = path.as_deref().and_then(BodyProfile::load).unwrap_or_default();
        Self::with_profile(profile, path)
    }

    pub fn with_profile(profile: BodyProfile, path: Option<PathBuf>) -> Self {
        Self {
            profile,
            path,
            devices: Vec::new(),
            hmd: [None; 3],
            calibration: None,
            instant: false,
            solved: None,
            fit: BodyFit::default(),
            mode: AvatarScaling::default(),
            avatar: None,
            applied: HashMap::new(),
            menu_open: false,
            window_open: false,
            status: None,
            triggers_prev: [false; 2],
            last_raw: TrackerTargets::default(),
            panel: BodyPanel::default(),
        }
    }

    fn save(&self) {
        if let Some(path) = &self.path
            && let Err(e) = self.profile.save(path)
        {
            tracing::warn!("body: saving {}: {e}", path.display());
        }
    }

    /// Drop renderer handles (the renderer's scene resources were reset).
    pub fn reset_render(&mut self) {
        self.panel = BodyPanel::default();
    }

    /// The scene changed: forget scales written onto its objects.
    pub fn scene_changed(&mut self) {
        self.applied.clear();
        self.solved = None;
    }

    pub fn calibrated(&self) -> bool {
        self.profile.tpose.is_some()
    }

    /// The panel shows (menu open or a calibration running).
    pub fn panel_visible(&self) -> bool {
        self.menu_open || self.calibration.is_some()
    }

    pub fn info(&self) -> BodyInfo {
        BodyInfo {
            player_height: self.profile.height(),
            eye_height: self.profile.eye_height(),
            arm_span: self.profile.arm_span,
            measured: self.profile.measured(),
            avatar_scale: self.fit.avatar,
            tracking_scale: self.fit.tracking,
            calibrating: self.calibration.is_some(),
            calibrated: self.calibrated(),
        }
    }

    pub fn command(&mut self, c: BodyCommand) {
        match c {
            BodyCommand::AlignAvatar => self.start(Flow::Align),
            BodyCommand::MeasureHeight => self.start(Flow::Height),
            BodyCommand::Calibrate => self.start(Flow::Full),
            BodyCommand::QuickCalibrate => self.start(Flow::Quick),
            BodyCommand::Menu(open) => {
                self.menu_open = open.unwrap_or(!self.menu_open);
                if !self.menu_open {
                    self.calibration = None;
                }
            }
            BodyCommand::SetHeight(cm) => {
                self.profile.height_override_cm = cm.map(|c| c.clamp(50.0, 260.0));
                self.save();
            }
            BodyCommand::Reset => {
                let auto = self.profile.auto_assign;
                self.profile = BodyProfile { auto_assign: auto, ..Default::default() };
                self.solved = None;
                self.calibration = None;
                self.status = Some("Body measurements and calibration cleared".into());
                self.save();
            }
        }
    }

    /// Capture the T-pose on the next update, keeping the roles
    /// (`ctx.calibrate_trackers()`).
    pub fn calibrate_now(&mut self) {
        self.instant = true;
    }

    /// Forget the T-pose (back to the default offsets).
    pub fn clear_calibration(&mut self) {
        self.profile.tpose = None;
        self.solved = None;
        self.save();
    }

    fn start(&mut self, flow: Flow) {
        let step = match flow {
            Flow::Align => Step::Align(false),
            Flow::Full | Flow::Height => Step::StandCountdown(COUNTDOWN),
            Flow::Quick => Step::TPoseCountdown(COUNTDOWN),
        };
        self.calibration = Some(Calibration { flow, step });
        self.status = None;
    }

    pub(crate) fn cancel(&mut self) {
        self.calibration = None;
        self.status = Some("Calibration cancelled".into());
    }

    pub(crate) fn skip_floor(&mut self) {
        if matches!(self.calibration.as_ref().map(|c| &c.step), Some(Step::Floor)) {
            self.finish("Calibrated".into());
        }
    }

    fn finish(&mut self, msg: String) {
        self.calibration = None;
        self.status = Some(msg);
    }

    /// Raw stage-space targets: head and hands from the headset and
    /// controllers, each tracker in the slot the player gave it (else its own
    /// role; the first device on a slot wins). Heights are floor corrected.
    pub fn assemble(
        &mut self,
        head: Option<(Vec3, Quat)>,
        left: Option<(Vec3, Quat)>,
        right: Option<(Vec3, Quat)>,
        mut devices: Vec<TrackerDevice>,
    ) -> TrackerTargets {
        let floor = self.profile.floor_offset;
        let fix = |p: Option<(Vec3, Quat)>| p.filter(|(p, r)| p.is_finite() && r.is_finite() && r.length_squared() > 1e-6)
            .map(|(p, r)| (p - Vec3::Y * floor, r.normalize()));
        for d in &mut devices {
            d.pose = fix(d.pose);
        }
        self.hmd = [fix(head), fix(left), fix(right)];
        self.devices = devices;
        self.assembled()
    }

    fn role_for(&self, d: &TrackerDevice) -> Option<usize> {
        self.profile.role_of(&d.id).unwrap_or(d.native).filter(|s| (FIRST_BODY_SLOT..TRACKER_SLOTS).contains(s))
    }

    fn assembled(&self) -> TrackerTargets {
        let mut slots = [None; TRACKER_SLOTS];
        slots[0] = self.hmd[0];
        slots[1] = self.hmd[1];
        slots[2] = self.hmd[2];
        for d in &self.devices {
            if let (Some(slot), Some(p)) = (self.role_for(d), d.pose)
                && slots[slot].is_none()
            {
                slots[slot] = Some(p);
            }
        }
        TrackerTargets::from_slots(slots)
    }

    /// Run the calibration steps, fit the scene's avatars and return the
    /// calibrated bone targets for the local avatar (stage space).
    pub fn update(&mut self, scene: &mut LoadedScene, raw: &TrackerTargets, frame: BodyFrame) -> TrackerTargets {
        self.last_raw = *raw;
        let pressed = [frame.triggers[0] && !self.triggers_prev[0], frame.triggers[1] && !self.triggers_prev[1]];
        self.triggers_prev = frame.triggers;
        if frame.tracking {
            let missing_avatar = self.calibration.as_ref().is_some_and(|c| c.flow == Flow::Align)
                && scene.local_avatar().is_none_or(|a| !a.local);
            if missing_avatar { self.status = Some("Choose a humanoid avatar before calibrating".into()); }
            else { self.step(raw, frame.dt, pressed); }
        } else if self.calibration.is_some() {
            self.calibration = None;
            self.status = Some("Calibration needs a headset".into());
        }
        if self.instant {
            self.instant = false;
            match self.capture_tpose(raw, false) {
                Ok(()) => tracing::info!("VR: captured T-pose calibration"),
                Err(e) => tracing::warn!("VR: T-pose calibration failed: {e}"),
            }
        }
        crate::avatar_hands::ensure_defaults(scene, frame.apply);
        let alignment = self.calibration.as_ref().is_some_and(|c| c.flow == Flow::Align);
        if alignment {
            let preview = scene.local_avatar().filter(|a| a.local).map(|av| {
                (scene.objects[av.index].id, zestengine_assets::ref_pose::reference_locals(av.skeleton))
            });
            if preview.is_none() { self.status = Some("Choose a humanoid avatar before calibrating".into()); }
            scene.calibration_pose = preview;
        } else {
            scene.calibration_pose = None;
        }
        let settings = vr_player_settings(scene);
        self.mode = settings.map(|s| s.0).unwrap_or_default();
        let mode = if frame.tracking && frame.apply { self.mode } else { AvatarScaling::Off };
        if !frame.apply {
            self.applied.clear();
        }
        let local = self.fit_scene(scene, mode, frame.apply);
        let reach = match (local, settings, self.profile.arm_span) {
            (Some((_, m, fit)), Some((_, true)), Some(player)) => m
                .arm_span
                .map(|a| zestengine_core::arm_reach(a * fit.model_to_stage, player))
                .unwrap_or(1.0),
            _ => 1.0,
        };
        scene.local_body = crate::scene::LocalBody {
            arm_reach: reach,
            avatar_scale: self.fit.avatar,
            eye_height: self.profile.measured().then(|| self.profile.eye_height()),
        };
        // Without a local avatar the targets still map onto a humanoid's bones
        // (other players see them on the avatar this player wears there).
        let Some(av) = scene.local_avatar() else {
            self.solved = None;
            return *raw;
        };
        let (key, scale) = match local {
            Some((key, _, fit)) => (key, fit.model_to_stage),
            None => ((scene.objects[av.index].id, av.skeleton.joints.len()), 1.0),
        };
        let stale = self
            .solved
            .is_none_or(|s| s.avatar != key || (s.scale - scale).abs() > 1e-3 * scale || s.from_tpose != self.calibrated());
        if stale {
            let cal = match self.profile.tpose_targets() {
                Some(t) => humanoid::calibrate_tpose(av.skeleton, &av.rig, &TrackerPoses::from_targets(&t), scale),
                None => humanoid::default_calibration(av.skeleton, &av.rig, scale),
            };
            self.solved = Some(Solved { avatar: key, scale, from_tpose: self.calibrated(), cal });
        }
        match &self.solved {
            Some(s) => humanoid::targets_from_calibrated(&s.cal, &TrackerPoses::from_targets(raw)),
            None => *raw,
        }
    }

    fn step(&mut self, raw: &TrackerTargets, dt: f32, pressed: [bool; 2]) {
        enum Next {
            Stay,
            Go(Step),
            Measured(Result<f32, &'static str>),
            Capture,
            Floor(f32),
        }
        let Some(cal) = self.calibration.as_mut() else { return };
        let flow = cal.flow;
        let next = match &mut cal.step {
            Step::Align(armed) => {
                if !self.triggers_prev[0] && !self.triggers_prev[1] { *armed = true; }
                if *armed && self.triggers_prev[0] && self.triggers_prev[1] { Next::Capture } else { Next::Stay }
            }
            Step::StandCountdown(t) => {
                *t -= dt;
                if *t <= 0.0 { Next::Go(Step::Standing(HeightSampler::default())) } else { Next::Stay }
            }
            Step::Standing(sampler) => match raw.head {
                Some((head, _)) => {
                    sampler.push(head.y, dt);
                    if sampler.done() { Next::Measured(sampler.result()) } else { Next::Stay }
                }
                None => Next::Stay,
            },
            Step::TPoseCountdown(t) => {
                *t -= dt;
                if *t <= 0.0 { Next::Go(Step::TPoseHold(TPOSE_HOLD)) } else { Next::Stay }
            }
            Step::TPoseHold(t) => {
                *t -= dt;
                if *t <= 0.0 { Next::Capture } else { Next::Stay }
            }
            // The controller whose trigger was pulled, if it is near the floor.
            Step::Floor => [raw.left_hand, raw.right_hand]
                .iter()
                .zip(pressed)
                .find_map(|(hand, down)| hand.filter(|(p, _)| down && p.y.abs() < 0.35).map(|(p, _)| Next::Floor(p.y - CONTROLLER_ON_FLOOR)))
                .unwrap_or(Next::Stay),
        };
        match next {
            Next::Stay => {}
            Next::Go(step) => cal.step = step,
            Next::Measured(Ok(eye)) => {
                self.profile.measured_eye_height = Some(eye);
                self.profile.height_override_cm = None;
                self.save();
                if flow == Flow::Height {
                    self.finish(format!("Height {:.0} cm", (eye + zestengine_core::EYE_TO_CROWN) * 100.0));
                } else if let Some(c) = self.calibration.as_mut() {
                    c.step = Step::TPoseCountdown(COUNTDOWN);
                }
            }
            Next::Measured(Err(e)) => {
                self.calibration = None;
                self.status = Some(format!("Height not measured: {e}"));
            }
            Next::Capture => match if flow == Flow::Align && (raw.head.is_none() || raw.left_hand.is_none() || raw.right_hand.is_none()) {
                Err("Headset and both hands must be tracking before accepting".into())
            } else { self.capture_tpose(raw, flow == Flow::Full || flow == Flow::Align) } {
                Ok(()) if flow == Flow::Full => {
                    if let Some(c) = self.calibration.as_mut() {
                        c.step = Step::Floor;
                    }
                }
                Ok(()) => self.finish("Calibrated".into()),
                Err(e) => {
                    if flow == Flow::Align {
                        if let Some(c) = self.calibration.as_mut() { c.step = Step::Align(false); }
                    } else { self.calibration = None; }
                    self.status = Some(format!("Calibration failed: {e}"));
                }
            },
            Next::Floor(delta) => {
                self.correct_floor(delta);
                self.finish(format!("Calibrated (floor moved {:+.1} cm)", delta * 100.0));
            }
        }
    }

    /// Lower the floor by `delta` (m): the measurements and stored T-pose
    /// were taken against the old one.
    fn correct_floor(&mut self, delta: f32) {
        self.profile.floor_offset += delta;
        if let Some(eye) = self.profile.measured_eye_height.as_mut() {
            *eye -= delta;
        }
        if let Some(t) = self.profile.tpose.as_mut() {
            for (p, _) in t.iter_mut().flatten() {
                p[1] -= delta;
            }
        }
        self.solved = None;
        self.save();
    }

    /// Store the T-pose (assigning roles from it first when `assign`), the
    /// arm span, and re-solve the offsets.
    fn capture_tpose(&mut self, raw: &TrackerTargets, assign: bool) -> Result<(), String> {
        let head = raw.head.ok_or("the headset is not tracking")?;
        let mut targets = *raw;
        if assign && self.profile.auto_assign {
            let tracked: Vec<(String, Vec3)> = self.devices.iter().filter_map(|d| Some((d.id.clone(), d.pose?.0))).collect();
            if !tracked.is_empty() {
                let positions: Vec<Vec3> = tracked.iter().map(|t| t.1).collect();
                let roles = zestengine_core::assign_roles(head, self.profile.eye_height(), self.profile.arm_span, &positions);
                for ((id, _), role) in tracked.iter().zip(roles) {
                    self.profile.set_role(id, role);
                }
                targets = self.assembled();
            }
        }
        if let (Some((l, _)), Some((r, _))) = (targets.left_hand, targets.right_hand)
            && let Some(span) = zestengine_core::arm_span(l, r, self.profile.eye_height())
        {
            self.profile.arm_span = Some(span);
        }
        self.profile.set_tpose(&targets);
        self.solved = None;
        self.save();
        Ok(())
    }

    /// Fit every avatar: local ones to the player (`mode`), remote ones to
    /// the scale their owner replicates. Writes object scales when `apply`.
    /// Returns the primary local avatar's key, measurements and fit.
    fn fit_scene(&mut self, scene: &mut LoadedScene, mode: AvatarScaling, apply: bool) -> Option<((ObjectId, usize), AvatarMeasure, BodyFit)> {
        let player_eye = self.profile.eye_height();
        let mut writes: Vec<(usize, f32)> = Vec::new();
        let mut primary = None;
        let mut rig_write = None;
        {
            let avatars = scene.humanoid_avatars();
            let world = if avatars.is_empty() { Vec::new() } else { scene.world_transforms() };
            for av in &avatars {
                let target = scale_target(scene, av.index, av.network_avatar, av.tracking_root);
                if writes.iter().any(|(t, _)| *t == target) {
                    continue;
                }
                let f_last = self.factor(scene, target);
                if av.local {
                    let Some(m) = humanoid::measure_avatar(av.skeleton, &av.rig) else { continue };
                    let root = av.tracking_root;
                    let k_last = root.map(|r| self.factor(scene, r)).unwrap_or(1.0);
                    let in_chain = root.is_some_and(|r| is_ancestor(scene, r, av.index));
                    let natural = uniform_scale(&world[av.index]) / f_last / if in_chain { k_last } else { 1.0 };
                    let root_base = root.map(|r| uniform_scale(&world[r]) / k_last).unwrap_or(1.0);
                    let fit = zestengine_core::fit_avatar(mode, player_eye, m.eye_height, natural, root_base);
                    let tracking = if in_chain { fit.tracking } else { 1.0 };
                    writes.push((target, fit.avatar / tracking));
                    if primary.is_none() {
                        primary = Some(((scene.objects[av.index].id, av.skeleton.joints.len()), m, fit));
                        rig_write = root.map(|r| (r, fit.tracking));
                    }
                } else if let Some(na) = av.network_avatar.and_then(|k| scene.network_avatar_of(k)).filter(|(_, a)| a.remote) {
                    writes.push((target, na.1.avatar_scale.clamp(0.05, 20.0)));
                }
            }
        }
        match primary {
            Some((_, m, fit)) => {
                self.fit = fit;
                self.avatar = Some(m);
            }
            None => {
                self.fit = BodyFit::default();
                self.avatar = None;
            }
        }
        if apply {
            if let Some(rig) = rig_write {
                writes.push(rig);
            } else if let Some(r) = scene.objects.iter().position(|o| o.components.iter().any(|c| c.as_any().is::<zestengine_core::VrPlayer>())) {
                writes.push((r, 1.0));
            }
            for (i, factor) in writes {
                self.write_scale(scene, i, factor);
            }
        }
        primary
    }

    /// The factor this module last wrote on object `i` (1 when untouched or
    /// changed by something else since).
    fn factor(&self, scene: &LoadedScene, i: usize) -> f32 {
        let o = &scene.objects[i];
        match self.applied.get(&o.id) {
            Some(a) if o.scale.abs_diff_eq(a.written, 1e-6) => a.factor,
            _ => 1.0,
        }
    }

    fn write_scale(&mut self, scene: &mut LoadedScene, i: usize, factor: f32) {
        let o = &mut scene.objects[i];
        let last = match self.applied.get(&o.id) {
            Some(a) if o.scale.abs_diff_eq(a.written, 1e-6) => Some(*a),
            _ => None,
        };
        let authored = last.map(|a| a.written / a.factor).unwrap_or(o.scale);
        if (last.map(|a| a.factor).unwrap_or(1.0) - factor).abs() < 1e-4 {
            return;
        }
        o.scale = authored * factor;
        self.applied.insert(o.id, Applied { factor, written: o.scale });
    }
}

/// The first active VR Player's fit settings (mode, match arm span).
fn vr_player_settings(scene: &LoadedScene) -> Option<(AvatarScaling, bool)> {
    let active = scene.active_flags();
    scene.objects.iter().enumerate().filter(|(i, _)| active[*i]).find_map(|(_, o)| {
        let p = o.components.iter().find_map(|c| c.as_any().downcast_ref::<zestengine_core::VrPlayer>())?;
        Some((p.avatar_scaling, p.match_arm_span))
    })
}

fn uniform_scale(m: &Mat4) -> f32 {
    let s = m.to_scale_rotation_translation().0;
    ((s.x.abs() + s.y.abs() + s.z.abs()) / 3.0).max(1e-6)
}

fn is_ancestor(scene: &LoadedScene, ancestor: usize, mut i: usize) -> bool {
    while let Some(p) = scene.objects[i].parent {
        if p == ancestor {
            return true;
        }
        i = p;
    }
    false
}

/// The object an avatar's fit scales: its Avatar Descriptor root (a worn
/// package), else the Humanoid IK object itself. Never climbs past the
/// player's Network Avatar or tracking root.
fn scale_target(scene: &LoadedScene, i: usize, network_avatar: Option<usize>, root: Option<usize>) -> usize {
    let mut k = i;
    loop {
        if Some(k) == network_avatar || Some(k) == root {
            return i;
        }
        if scene.objects[k].components.iter().any(|c| c.as_any().is::<zestengine_core::AvatarDescriptor>()) {
            return k;
        }
        match scene.objects[k].parent {
            Some(p) => k = p,
            None => return i,
        }
    }
}

#[cfg(test)]
mod tests;
