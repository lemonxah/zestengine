//! The player's body for VR avatars: standing eye height and arm span
//! measured from the headset and controllers (kept per user, shared by every
//! game), how an avatar is fitted to them, and which tracker drives which body
//! part. Pure math and data; the engine owns the devices and the capture flow.

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::{TRACKER_ROLES, TRACKER_SLOTS, TrackerTargets};

/// Average distance from the eyes to the top of the head (m): standing height
/// is the eye height plus this.
pub const EYE_TO_CROWN: f32 = 0.11;
/// Eye height used until the player is measured (a 1.75 m adult).
pub const DEFAULT_EYE_HEIGHT: f32 = 1.64;
/// Seconds of headset samples a height measurement averages.
pub const HEIGHT_CAPTURE_SECONDS: f32 = 1.0;
/// Fit factors stay inside this range (a broken rig must not blow up the world).
const FIT_RANGE: std::ops::RangeInclusive<f32> = 0.05..=20.0;
/// First slot a body tracker can take (head and hands come from the headset
/// and controllers).
pub const FIRST_BODY_SLOT: usize = 3;

/// How an avatar is fitted to the player.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum AvatarScaling {
    /// Uniformly scale the avatar so its eyes sit at the player's eyes.
    #[default]
    ScaleAvatar,
    /// Keep the avatar's size and scale the tracking space instead, so the
    /// player sees from the avatar's eyes (a giant, a child).
    ScalePlayer,
    /// No fitting: both stay at their authored scale.
    Off,
}

impl AvatarScaling {
    /// Every mode, in declaration order.
    pub const ALL: [AvatarScaling; 3] = [Self::ScaleAvatar, Self::ScalePlayer, Self::Off];
    /// Inspector label of this mode.
    pub fn label(self) -> &'static str {
        match self {
            Self::ScaleAvatar => "Scale avatar",
            Self::ScalePlayer => "Scale player",
            Self::Off => "Off",
        }
    }
}

/// How an avatar is fitted to the player this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyFit {
    /// Avatar world scale over its authored (natural) world scale.
    pub avatar: f32,
    /// Tracking space (VR Player rig) scale over its authored scale.
    pub tracking: f32,
    /// Avatar model units to tracking-space meters: the avatar's rest pose
    /// as the player stands in it is `rest * model_to_stage`.
    pub model_to_stage: f32,
}

impl Default for BodyFit {
    fn default() -> Self {
        Self { avatar: 1.0, tracking: 1.0, model_to_stage: 1.0 }
    }
}

/// Fit an avatar to the player. `player_eye` is the player's standing eye
/// height (m), `avatar_eye` the avatar's eye height in its model units,
/// `natural` the avatar's authored world scale and `root` the tracking
/// space's authored world scale.
pub fn fit_avatar(mode: AvatarScaling, player_eye: f32, avatar_eye: f32, natural: f32, root: f32) -> BodyFit {
    let natural = natural.max(1e-6);
    let root = root.max(1e-6);
    let avatar_eye_world = avatar_eye.max(1e-4) * natural;
    let player_eye_world = player_eye.max(0.1) * root;
    let (avatar, tracking) = match mode {
        AvatarScaling::ScaleAvatar => ((player_eye_world / avatar_eye_world).clamp(*FIT_RANGE.start(), *FIT_RANGE.end()), 1.0),
        AvatarScaling::ScalePlayer => (1.0, (avatar_eye_world / player_eye_world).clamp(*FIT_RANGE.start(), *FIT_RANGE.end())),
        AvatarScaling::Off => (1.0, 1.0),
    };
    BodyFit { avatar, tracking, model_to_stage: natural * avatar / (root * tracking) }
}

/// Hand-target reach multiplier that maps the player's arm span onto the
/// avatar's (both in tracking-space meters), so long or short avatar arms
/// reach naturally without non-uniform scaling.
pub fn arm_reach(avatar_span: f32, player_span: f32) -> f32 {
    if avatar_span <= 1e-3 || player_span <= 1e-3 {
        return 1.0;
    }
    (avatar_span / player_span).clamp(0.5, 2.0)
}

/// Robust average of samples: drops everything further than three median
/// absolute deviations (at least 1 cm) from the median, averages the rest.
/// Returns the mean and the inliers' spread (max - min).
pub fn robust_mean(samples: &[f32]) -> Option<(f32, f32)> {
    let mut v: Vec<f32> = samples.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(f32::total_cmp);
    let median = v[v.len() / 2];
    let mut dev: Vec<f32> = v.iter().map(|x| (x - median).abs()).collect();
    dev.sort_by(f32::total_cmp);
    let mad = dev[dev.len() / 2];
    let limit = (mad * 3.0).max(0.01);
    let inliers: Vec<f32> = v.into_iter().filter(|x| (x - median).abs() <= limit).collect();
    let mean = inliers.iter().sum::<f32>() / inliers.len() as f32;
    let spread = inliers.last().copied().unwrap_or(mean) - inliers.first().copied().unwrap_or(mean);
    Some((mean, spread))
}

/// Collects headset heights over [`HEIGHT_CAPTURE_SECONDS`] while the player
/// stands straight, then gives their eye height.
#[derive(Clone, Debug, Default)]
pub struct HeightSampler {
    samples: Vec<f32>,
    elapsed: f32,
}

impl HeightSampler {
    /// One frame: the headset's height above the floor and the frame time.
    pub fn push(&mut self, y: f32, dt: f32) {
        self.samples.push(y);
        self.elapsed += dt.max(0.0);
    }

    /// True when the capture window has elapsed.
    pub fn done(&self) -> bool {
        self.elapsed >= HEIGHT_CAPTURE_SECONDS
    }

    /// 0..1 through the capture.
    pub fn progress(&self) -> f32 {
        (self.elapsed / HEIGHT_CAPTURE_SECONDS).clamp(0.0, 1.0)
    }

    /// The measured eye height, or why it can't be trusted.
    pub fn result(&self) -> Result<f32, &'static str> {
        if self.samples.len() < 10 {
            return Err("not enough headset samples (is the headset tracking?)");
        }
        let (mean, spread) = robust_mean(&self.samples).ok_or("no headset samples")?;
        if spread > 0.06 {
            return Err("the headset moved too much: stand still and look ahead");
        }
        if !(0.5..=2.6).contains(&mean) {
            return Err("that height looks wrong: check the play area floor");
        }
        Ok(mean)
    }
}

/// Distance between the hands in a T-pose (m), if the pose looks like one.
pub fn arm_span(left: Vec3, right: Vec3, eye_height: f32) -> Option<f32> {
    let span = left.distance(right);
    let level = (left.y - right.y).abs() < 0.25 * eye_height;
    (level && (0.5 * eye_height..=1.6 * eye_height).contains(&span)).then_some(span)
}

/// A tracker pose as stored: position and quaternion (xyzw).
pub type StoredPose = ([f32; 3], [f32; 4]);

fn store_pose((p, r): (Vec3, Quat)) -> StoredPose {
    (p.to_array(), r.to_array())
}

fn load_pose((p, r): StoredPose) -> (Vec3, Quat) {
    (Vec3::from(p), Quat::from_array(r).normalize())
}

/// One tracker device's role choice, by device id.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrackerRole {
    /// Device id.
    pub device: String,
    /// Tracker slot (see [`TRACKER_ROLES`]); `None` ignores the device.
    pub role: Option<usize>,
}

/// The player's body, saved per user (shared by every game): measurements,
/// tracker roles and the T-pose the tracker offsets are solved from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BodyProfile {
    /// Standing eye height from the last measurement (m).
    pub measured_eye_height: Option<f32>,
    /// The player's own height entry (cm, to the top of the head); wins over
    /// the measurement.
    pub height_override_cm: Option<f32>,
    /// Hand-to-hand distance in a T-pose (m).
    pub arm_span: Option<f32>,
    /// Play-area floor correction (m), subtracted from every tracked height.
    pub floor_offset: f32,
    /// Assign tracker roles from the T-pose during a full calibration.
    pub auto_assign: bool,
    /// Role choices by device id (devices not listed keep their own role).
    pub roles: Vec<TrackerRole>,
    /// Stage-space tracker poses per slot at the last T-pose. Offsets are
    /// solved from it for whichever avatar is worn, so changing avatars
    /// needs no new T-pose.
    pub tpose: Option<Vec<Option<StoredPose>>>,
}

impl Default for BodyProfile {
    fn default() -> Self {
        Self {
            measured_eye_height: None,
            height_override_cm: None,
            arm_span: None,
            floor_offset: 0.0,
            auto_assign: true,
            roles: Vec::new(),
            tpose: None,
        }
    }
}

impl BodyProfile {
    /// Standing eye height: the override, else the measurement, else a default.
    pub fn eye_height(&self) -> f32 {
        match self.height_override_cm {
            Some(cm) => (cm / 100.0 - EYE_TO_CROWN).max(0.3),
            None => self.measured_eye_height.unwrap_or(DEFAULT_EYE_HEIGHT),
        }
    }

    /// Standing height to the top of the head (m).
    pub fn height(&self) -> f32 {
        self.eye_height() + EYE_TO_CROWN
    }

    /// The player was measured or entered a height.
    pub fn measured(&self) -> bool {
        self.height_override_cm.is_some() || self.measured_eye_height.is_some()
    }

    /// The role a device was given (`Some(None)` = ignored), `None` when the
    /// player never chose one.
    pub fn role_of(&self, device: &str) -> Option<Option<usize>> {
        self.roles.iter().find(|r| r.device == device).map(|r| r.role)
    }

    /// Set (or record) the role of a device; None ignores it.
    pub fn set_role(&mut self, device: &str, role: Option<usize>) {
        match self.roles.iter_mut().find(|r| r.device == device) {
            Some(r) => r.role = role,
            None => self.roles.push(TrackerRole { device: device.to_owned(), role }),
        }
    }

    /// The saved T-pose as tracker targets.
    pub fn tpose_targets(&self) -> Option<TrackerTargets> {
        let poses = self.tpose.as_ref()?;
        let mut slots = [None; TRACKER_SLOTS];
        for (s, p) in slots.iter_mut().zip(poses) {
            *s = p.map(load_pose);
        }
        Some(TrackerTargets::from_slots(slots))
    }

    /// Store the T-pose (the tracker offsets are solved from it).
    pub fn set_tpose(&mut self, t: &TrackerTargets) {
        self.tpose = Some(t.slots().iter().map(|p| p.map(store_pose)).collect());
    }

    /// Serialize to RON.
    pub fn to_ron(&self) -> String {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default()).unwrap_or_default()
    }

    /// Parse RON.
    pub fn from_ron(s: &str) -> Option<Self> {
        ron::from_str(s).ok()
    }

    /// Read a saved profile (`None` when missing or unreadable).
    pub fn load(path: &std::path::Path) -> Option<Self> {
        Self::from_ron(&std::fs::read_to_string(path).ok()?)
    }

    /// Write to `path` (creating parent directories).
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, self.to_ron())
    }
}

/// Where each body role sits in a T-pose, in the player's frame (x to their
/// right, y up from the floor, z forward), for an eye height `e` and hand
/// span `w`.
fn tpose_template(slot: usize, e: f32, w: f32) -> Vec3 {
    let k = e / DEFAULT_EYE_HEIGHT;
    match slot {
        3 => Vec3::new(0.0, 0.57 * e, 0.0),
        4 => Vec3::new(-0.11 * k, 0.07 * k, 0.0),
        5 => Vec3::new(0.11 * k, 0.07 * k, 0.0),
        6 => Vec3::new(0.0, 0.78 * e, 0.0),
        7 => Vec3::new(-0.28 * w, 0.87 * e, 0.0),
        8 => Vec3::new(0.28 * w, 0.87 * e, 0.0),
        9 => Vec3::new(-0.1 * k, 0.3 * e, 0.0),
        10 => Vec3::new(0.1 * k, 0.3 * e, 0.0),
        _ => Vec3::ZERO,
    }
}

/// Body roles for trackers from a T-pose: `head` is the headset pose (stage
/// space, floor at y = 0), `trackers` their positions. Returns a slot per
/// tracker (`None` for one that fits no role). Every tracker is matched to a
/// role's expected spot at once (best total fit, each role used once), so
/// the left foot never steals the right one.
pub fn assign_roles(head: (Vec3, Quat), eye_height: f32, arm_span: Option<f32>, trackers: &[Vec3]) -> Vec<Option<usize>> {
    const ROLES: [usize; 8] = [3, 4, 5, 6, 7, 8, 9, 10];
    const SKIP: f32 = 0.35 * 0.35;
    let e = eye_height.max(0.5);
    let w = arm_span.unwrap_or(e * 1.0);
    let f = head.1 * Vec3::NEG_Z;
    let fwd = Vec3::new(f.x, 0.0, f.z).normalize_or(Vec3::NEG_Z);
    let right = fwd.cross(Vec3::Y);
    let origin = Vec3::new(head.0.x, 0.0, head.0.z);
    let cost = |t: Vec3, slot: usize| {
        let rel = t - origin;
        let local = Vec3::new(rel.dot(right), rel.y, rel.dot(fwd));
        let d = local - tpose_template(slot, e, w);
        d.x * d.x + d.y * d.y + 0.25 * d.z * d.z
    };
    // Exact assignment over role subsets: best[mask] after each tracker.
    let n = trackers.len();
    let masks = 1usize << ROLES.len();
    let mut best = vec![f32::INFINITY; masks];
    best[0] = 0.0;
    let mut choice: Vec<Vec<(usize, Option<usize>)>> = Vec::with_capacity(n);
    for &t in trackers {
        let mut next = vec![f32::INFINITY; masks];
        let mut from = vec![(0usize, None); masks];
        for mask in 0..masks {
            if !best[mask].is_finite() {
                continue;
            }
            let skip = best[mask] + SKIP;
            if skip < next[mask] {
                next[mask] = skip;
                from[mask] = (mask, None);
            }
            for (k, &slot) in ROLES.iter().enumerate() {
                if mask & (1 << k) != 0 {
                    continue;
                }
                let c = cost(t, slot);
                if c >= SKIP {
                    continue;
                }
                let m = mask | (1 << k);
                if best[mask] + c < next[m] {
                    next[m] = best[mask] + c;
                    from[m] = (mask, Some(slot));
                }
            }
        }
        best = next;
        choice.push(from);
    }
    let mut mask = (0..masks).min_by(|a, b| best[*a].total_cmp(&best[*b])).unwrap_or(0);
    let mut out = vec![None; n];
    for i in (0..n).rev() {
        let (prev, slot) = choice[i][mask];
        out[i] = slot;
        mask = prev;
    }
    out
}

/// The player's body as gameplay sees it (`ctx.screen.body`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyInfo {
    /// Standing height to the top of the head (m).
    pub player_height: f32,
    /// Standing eye height (m).
    pub eye_height: f32,
    /// Hand-to-hand distance in a T-pose (m).
    pub arm_span: Option<f32>,
    /// The player was measured (or entered their height).
    pub measured: bool,
    /// Avatar scale over its authored size.
    pub avatar_scale: f32,
    /// Tracking-space scale (Scale player mode).
    pub tracking_scale: f32,
    /// A calibration step is running.
    pub calibrating: bool,
    /// The trackers were calibrated (a T-pose was captured).
    pub calibrated: bool,
}

impl Default for BodyInfo {
    fn default() -> Self {
        Self {
            player_height: DEFAULT_EYE_HEIGHT + EYE_TO_CROWN,
            eye_height: DEFAULT_EYE_HEIGHT,
            arm_span: None,
            measured: false,
            avatar_scale: 1.0,
            tracking_scale: 1.0,
            calibrating: false,
            calibrated: false,
        }
    }
}

/// Requests for the engine's body calibration (see `ComponentCtx`).
#[derive(Clone, Debug, PartialEq)]
pub enum BodyCommand {
    /// Align the worn avatar in a T-pose mirror, then accept with both triggers.
    AlignAvatar,
    /// Measure the player's standing height (a short stand-straight capture).
    MeasureHeight,
    /// Full calibration: stand straight, T-pose (assigns tracker roles),
    /// then an optional floor check.
    Calibrate,
    /// T-pose only, keeping the tracker roles.
    QuickCalibrate,
    /// Open (`Some(true)`), close or toggle (`None`) the VR body menu.
    Menu(Option<bool>),
    /// The player's height in cm (`None` goes back to the measurement).
    SetHeight(Option<f32>),
    /// Forget the measurements, roles and calibration.
    Reset,
}

/// Display name of a tracker slot.
pub fn role_name(slot: usize) -> &'static str {
    TRACKER_ROLES.get(slot).copied().unwrap_or("?")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn robust_mean_rejects_spikes() {
        let mut s: Vec<f32> = (0..60).map(|i| 1.62 + ((i % 5) as f32 - 2.0) * 0.002).collect();
        s[10] = 0.2;
        s[30] = 2.5;
        s[45] = 1.9;
        let (mean, spread) = robust_mean(&s).unwrap();
        assert!((mean - 1.62).abs() < 0.002, "{mean}");
        assert!(spread < 0.01);
        assert!(robust_mean(&[]).is_none());
    }

    #[test]
    fn height_sampler_smooths_and_rejects_movement() {
        let mut h = HeightSampler::default();
        for i in 0..90 {
            let y = 1.70 + (i as f32 * 0.7).sin() * 0.004 + if i == 40 { 0.4 } else { 0.0 };
            h.push(y, 1.0 / 90.0);
        }
        assert!(h.done());
        assert!((h.result().unwrap() - 1.70).abs() < 0.003);

        let mut moving = HeightSampler::default();
        for i in 0..90 {
            moving.push(1.4 + i as f32 * 0.004, 1.0 / 90.0);
        }
        assert!(moving.result().is_err(), "a crouch-to-stand is not a measurement");
        let mut short = HeightSampler::default();
        short.push(1.6, 1.0);
        assert!(short.result().is_err());
    }

    #[test]
    fn scale_avatar_matches_eye_heights() {
        let fit = fit_avatar(AvatarScaling::ScaleAvatar, 1.8, 1.5, 1.0, 1.0);
        assert!((fit.avatar - 1.2).abs() < 1e-5 && fit.tracking == 1.0);
        assert!((1.5 * fit.model_to_stage - 1.8).abs() < 1e-4, "avatar eyes at the player's eyes");
        // A centimeter-unit model (authored at 0.01) still lands at the player's eyes.
        let cm = fit_avatar(AvatarScaling::ScaleAvatar, 1.8, 150.0, 0.01, 1.0);
        assert!((cm.avatar - 1.2).abs() < 1e-4);
        assert!((150.0 * cm.model_to_stage - 1.8).abs() < 1e-3);
    }

    #[test]
    fn scale_player_scales_tracking_and_keeps_the_avatar() {
        let fit = fit_avatar(AvatarScaling::ScalePlayer, 1.6, 3.2, 1.0, 1.0);
        assert_eq!(fit.avatar, 1.0);
        assert!((fit.tracking - 2.0).abs() < 1e-5, "the player becomes a giant");
        // The headset (1.6 m in the room) sits at 3.2 m in the world, the avatar's eyes.
        assert!((1.6 * fit.tracking - 3.2).abs() < 1e-4);
        assert!((3.2 * fit.model_to_stage - 1.6).abs() < 1e-4, "in the room the avatar is player-sized");
    }

    #[test]
    fn off_keeps_everything() {
        let fit = fit_avatar(AvatarScaling::Off, 1.9, 1.4, 2.0, 1.0);
        assert_eq!((fit.avatar, fit.tracking), (1.0, 1.0));
        assert!((fit.model_to_stage - 2.0).abs() < 1e-6);
    }

    #[test]
    fn arm_span_and_reach() {
        let e = 1.64;
        let span = arm_span(Vec3::new(-0.8, 1.45, 0.0), Vec3::new(0.82, 1.43, 0.0), e).unwrap();
        assert!((span - 1.62).abs() < 0.01);
        assert!(arm_span(Vec3::new(-0.1, 1.0, 0.0), Vec3::new(0.1, 1.0, 0.0), e).is_none(), "hands together");
        assert!(arm_span(Vec3::new(-0.8, 0.4, 0.0), Vec3::new(0.8, 1.6, 0.0), e).is_none(), "not level");
        assert!((arm_reach(1.4, 1.75) - 0.8).abs() < 1e-5);
        assert_eq!(arm_reach(0.0, 1.6), 1.0);
    }

    #[test]
    fn profile_eye_height_override_and_round_trip() {
        let mut p = BodyProfile { measured_eye_height: Some(1.7), ..Default::default() };
        assert!((p.eye_height() - 1.7).abs() < 1e-6);
        p.height_override_cm = Some(160.0);
        assert!((p.eye_height() - (1.6 - EYE_TO_CROWN)).abs() < 1e-5);
        assert!((p.height() - 1.6).abs() < 1e-5);
        p.set_role("xr:waist", Some(6));
        p.set_role("slimevr:abc", None);
        p.set_role("xr:waist", Some(3));
        p.arm_span = Some(1.66);
        p.floor_offset = 0.02;
        let t = TrackerTargets {
            hips: Some((Vec3::new(0.0, 0.95, 0.1), Quat::from_rotation_y(0.3))),
            left_foot: Some((Vec3::new(-0.1, 0.08, 0.0), Quat::IDENTITY)),
            ..Default::default()
        };
        p.set_tpose(&t);
        let back = BodyProfile::from_ron(&p.to_ron()).expect("parses");
        assert_eq!(back, p);
        assert_eq!(back.role_of("xr:waist"), Some(Some(3)));
        assert_eq!(back.role_of("slimevr:abc"), Some(None));
        assert_eq!(back.role_of("other"), None);
        let tp = back.tpose_targets().unwrap();
        assert!(tp.hips.unwrap().1.angle_between(Quat::from_rotation_y(0.3)) < 1e-5);
        assert!(tp.chest.is_none());

        let dir = std::env::temp_dir().join(format!("zestengine-body-{}", std::process::id()));
        let path = dir.join("body.ron");
        p.save(&path).unwrap();
        assert_eq!(BodyProfile::load(&path), Some(p));
        let _ = std::fs::remove_dir_all(&dir);
        assert!(BodyProfile::from_ron("garbage(").is_none());
    }

    /// A 1.75 m player in a T-pose facing `yaw`, trackers on every body role.
    fn tpose_trackers(yaw: f32) -> ((Vec3, Quat), Vec<(usize, Vec3)>) {
        let rot = Quat::from_rotation_y(yaw);
        let at = Vec3::new(0.4, 0.0, -0.7);
        let place = |x: f32, y: f32, z: f32| at + rot * Vec3::new(x, y, -z);
        let head = (place(0.0, 1.64, 0.0), rot);
        let trackers = vec![
            (3, place(0.01, 0.96, -0.08)),
            (4, place(-0.12, 0.09, 0.04)),
            (5, place(0.11, 0.1, 0.05)),
            (6, place(0.0, 1.3, 0.1)),
            (7, place(-0.47, 1.44, -0.03)),
            (8, place(0.45, 1.42, -0.02)),
            (9, place(-0.1, 0.5, 0.08)),
            (10, place(0.11, 0.48, 0.07)),
        ];
        (head, trackers)
    }

    #[test]
    fn tpose_auto_assigns_every_role() {
        for yaw in [0.0f32, 1.3, -2.4, std::f32::consts::PI] {
            let (head, mut trackers) = tpose_trackers(yaw);
            // Order as a runtime might list them.
            trackers.swap(0, 5);
            trackers.swap(2, 7);
            trackers.swap(1, 4);
            let positions: Vec<Vec3> = trackers.iter().map(|t| t.1).collect();
            let roles = assign_roles(head, 1.64, Some(1.62), &positions);
            for ((want, _), got) in trackers.iter().zip(&roles) {
                assert_eq!(Some(*want), *got, "yaw {yaw}: {} got {:?}", role_name(*want), got.map(role_name));
            }
        }
    }

    #[test]
    fn three_trackers_become_hips_and_feet_and_strays_stay_unassigned() {
        let (head, all) = tpose_trackers(0.6);
        let mut trackers: Vec<(usize, Vec3)> = all.into_iter().filter(|(s, _)| (3..=5).contains(s)).collect();
        trackers.reverse();
        let mut positions: Vec<Vec3> = trackers.iter().map(|t| t.1).collect();
        // A tracker left on a table across the room.
        positions.push(Vec3::new(4.0, 0.8, 3.0));
        let roles = assign_roles(head, 1.64, None, &positions);
        assert_eq!(roles, vec![Some(5), Some(4), Some(3), None]);
    }
}
