//! Authored animation clip asset (`.anim`): keyframed property tracks made in
//! the editor's Animation tab. A track targets a bone (skinned models) or an
//! object path relative to the Animator's object ("" = the object itself,
//! "Arm/Hand" = a descendant) and animates one property with per-key
//! interpolation (step, linear, or cubic with automatic tangents).
//!
//! Besides transforms, tracks can drive object properties (object active,
//! component enabled, blendshape weight, material colour / float / texture
//! offset). Their address is the same relative object path plus a property
//! `name` and a `slot`:
//!
//! | property          | `name`                          | `slot`              |
//! |-------------------|---------------------------------|---------------------|
//! | `Active`          | (empty)                         | -                   |
//! | `ComponentEnabled`| component type name ("Light")   | -                   |
//! | `BlendShape`      | blendshape name ("vrc.v_aa")    | -                   |
//! | `Material*`       | material property ("base_color")| render slot (0 = primary) |
//!
//! Paths are names of objects below the Animator, never indices or object ids,
//! so a clip is portable between scenes, prefabs and avatar-build clones;
//! blendshapes are addressed by name, so reordering shapes on reimport keeps
//! working. When an object is renamed or moved, [`AnimClip::rename_target`]
//! and [`AnimClip::rebase_targets`] rewrite the paths.

use std::path::Path;

use anyhow::{Context as _, Result};
use glam::Quat;
use serde::{Deserialize, Serialize};

pub const ANIM_CLIP_EXTENSION: &str = "anim";

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum TrackProperty {
    #[default]
    Translation,
    /// Quaternion `[x, y, z, w]`.
    Rotation,
    Scale,
    /// `Light` component intensity (first value lane).
    LightIntensity,
    /// Object active flag (step; value >= 0.5 is on).
    Active,
    /// A component switched on / off by type name (step; value >= 0.5 is on).
    ComponentEnabled,
    /// Blendshape weight by name on the object's skinned mesh (one lane).
    BlendShape,
    /// Material colour property (RGBA lanes) of a render slot.
    MaterialColor,
    /// Material scalar property of a render slot (one lane).
    MaterialFloat,
    /// Material texture offset (UV, two lanes) of a render slot.
    MaterialOffset,
    /// Humanoid muscle value in [-1, 1] (`name` = muscle name, one lane); a
    /// muscle clip ([`ClipKind::Muscle`]) holds these.
    Muscle,
    /// Muscle clip body root: centre of mass over the human scale (three lanes).
    RootPosition,
    /// Muscle clip body root orientation relative to the reference pose (quaternion).
    RootRotation,
    /// Muscle clip limb goal position over the human scale (`name` = `LeftHand`, `RightHand`, `LeftFoot`, `RightFoot`).
    GoalPosition,
    /// Muscle clip limb goal relative to its limb root over the limb length.
    GoalRelative,
    /// A named value of the object's `CustomProperties` (`name` = property,
    /// a vector's lanes are `Name.x`, `Name.y`, `Name.z`; one lane). Ints and
    /// bools round / threshold when written.
    CustomProperty,
    /// 2D sprite swap: the `SpriteResolver` on the object takes the sprite
    /// library entry whose 24-bit hash (`sprite_library::entry_hash`) is the
    /// key value (one lane, always a step). Unknown hashes change nothing.
    SpriteSwap,
}

impl TrackProperty {
    /// The classic transform / light properties offered per object.
    pub const ALL: [TrackProperty; 4] = [Self::Translation, Self::Rotation, Self::Scale, Self::LightIntensity];
    pub fn label(self) -> &'static str {
        match self {
            Self::Translation => "Position",
            Self::Rotation => "Rotation",
            Self::Scale => "Scale",
            Self::LightIntensity => "Light Intensity",
            Self::Active => "Active",
            Self::ComponentEnabled => "Component Enabled",
            Self::BlendShape => "Blend Shape",
            Self::MaterialColor => "Material Colour",
            Self::MaterialFloat => "Material Value",
            Self::MaterialOffset => "Material Offset",
            Self::Muscle => "Muscle",
            Self::RootPosition => "Root Position",
            Self::RootRotation => "Root Rotation",
            Self::GoalPosition => "Goal Position",
            Self::GoalRelative => "Goal Relative",
            Self::SpriteSwap => "Sprite Swap",
            Self::CustomProperty => "Custom Property",
        }
    }
    /// Value lanes used in `Keyframe::value`.
    pub fn lanes(self) -> usize {
        match self {
            Self::Translation | Self::Scale | Self::RootPosition | Self::GoalPosition | Self::GoalRelative => 3,
            Self::Rotation | Self::MaterialColor | Self::RootRotation => 4,
            Self::MaterialOffset => 2,
            Self::LightIntensity | Self::Active | Self::ComponentEnabled | Self::BlendShape | Self::MaterialFloat | Self::Muscle | Self::SpriteSwap | Self::CustomProperty => 1,
        }
    }
    /// A track of a muscle clip (see [`ClipKind::Muscle`]); it drives a humanoid
    /// pose, not a node, so its `target` is unused.
    pub fn is_muscle(self) -> bool {
        matches!(self, Self::Muscle | Self::RootPosition | Self::RootRotation | Self::GoalPosition | Self::GoalRelative)
    }
    /// Drives an object property addressed by `AnimTrack::name` / `slot`
    /// rather than a transform node channel.
    pub fn is_object_prop(self) -> bool {
        matches!(
            self,
            Self::Active | Self::ComponentEnabled | Self::BlendShape | Self::MaterialColor | Self::MaterialFloat | Self::MaterialOffset | Self::SpriteSwap | Self::CustomProperty
        )
    }
    /// A switch: always evaluated as a step whatever the key says.
    pub fn is_step(self) -> bool {
        matches!(self, Self::Active | Self::ComponentEnabled | Self::SpriteSwap)
    }
    /// Interpolation of a track's first key.
    pub fn default_interp(self) -> KeyInterp {
        match self {
            Self::Active | Self::ComponentEnabled | Self::SpriteSwap => KeyInterp::Step,
            Self::BlendShape | Self::MaterialColor | Self::MaterialFloat | Self::MaterialOffset | Self::CustomProperty => KeyInterp::Linear,
            Self::Muscle | Self::RootPosition | Self::RootRotation | Self::GoalPosition | Self::GoalRelative => KeyInterp::Linear,
            _ => KeyInterp::Cubic,
        }
    }
    /// The track needs a `name` (component, shape or material property).
    pub fn needs_name(self) -> bool {
        matches!(
            self,
            Self::ComponentEnabled | Self::BlendShape | Self::MaterialColor | Self::MaterialFloat | Self::MaterialOffset | Self::Muscle | Self::GoalPosition | Self::GoalRelative | Self::CustomProperty
        )
    }
    /// The track addresses a material render slot.
    pub fn uses_slot(self) -> bool {
        matches!(self, Self::MaterialColor | Self::MaterialFloat | Self::MaterialOffset)
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum KeyInterp {
    /// Hold this key's value until the next key.
    Step,
    Linear,
    /// Smooth Hermite curve with automatic (Catmull-Rom) tangents, flat at the ends.
    #[default]
    Cubic,
    /// Smooth Hermite curve with user-edited tangents (falls back to automatic
    /// where a key leaves them unset).
    Custom,
}

impl KeyInterp {
    pub const ALL: [KeyInterp; 4] = [Self::Step, Self::Linear, Self::Cubic, Self::Custom];
    pub fn label(self) -> &'static str {
        match self {
            Self::Step => "Constant",
            Self::Linear => "Linear",
            Self::Cubic => "Auto (cubic)",
            Self::Custom => "Custom (bezier)",
        }
    }
}

/// `interp` shapes the segment from this key to the next.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Default)]
#[serde(default)]
pub struct Keyframe {
    pub time: f32,
    pub value: [f32; 4],
    pub interp: KeyInterp,
    /// Outgoing tangent (derivative w.r.t. time, per lane) for a `Custom`
    /// segment. `None` falls back to the automatic (Catmull-Rom) tangent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tangent_out: Option<[f32; 4]>,
    /// Incoming tangent of the `Custom` segment ending at this key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tangent_in: Option<[f32; 4]>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
#[serde(default)]
pub struct AnimTrack {
    pub target: String,
    pub property: TrackProperty,
    /// Component type, blendshape or material property, for the object
    /// properties that need one (see [`TrackProperty::needs_name`]).
    #[serde(skip_serializing_if = "String::is_empty")]
    pub name: String,
    /// Material render slot (0 = the object's primary material).
    #[serde(skip_serializing_if = "is_zero")]
    pub slot: u8,
    /// Sorted by time.
    pub keys: Vec<Keyframe>,
}

fn is_zero(v: &u8) -> bool {
    *v == 0
}

impl AnimTrack {
    pub fn new(target: &str, property: TrackProperty) -> Self {
        Self { target: target.to_owned(), property, ..Default::default() }
    }

    /// An object-property track (`name` / `slot` as in the module docs).
    pub fn prop(target: &str, property: TrackProperty, name: &str, slot: u8) -> Self {
        Self { target: target.to_owned(), property, name: name.to_owned(), slot, keys: Vec::new() }
    }

    /// Whether this track has the given address.
    pub fn is(&self, target: &str, property: TrackProperty, name: &str, slot: u8) -> bool {
        self.property == property && self.target == target && self.name == name && (self.slot == slot || !property.uses_slot())
    }

    pub fn sort(&mut self) {
        self.keys.sort_by(|a, b| a.time.total_cmp(&b.time));
    }

    /// Insert or replace the key at `time` (within half a millisecond).
    pub fn set_key(&mut self, time: f32, value: [f32; 4]) -> usize {
        if let Some(i) = self.keys.iter().position(|k| (k.time - time).abs() < 5e-4) {
            self.keys[i].value = value;
            return i;
        }
        let interp = if self.property.is_step() {
            KeyInterp::Step
        } else {
            self.keys.last().map(|k| k.interp).unwrap_or(self.property.default_interp())
        };
        self.keys.push(Keyframe { time, value, interp, ..Default::default() });
        self.sort();
        self.keys.iter().position(|k| (k.time - time).abs() < 5e-4).unwrap_or(0)
    }

    /// Value at `t` seconds (no wrapping; clamps to the first / last key).
    pub fn sample(&self, t: f32) -> [f32; 4] {
        let keys = &self.keys;
        let Some(first) = keys.first() else {
            return [0.0; 4];
        };
        if keys.len() == 1 || t <= first.time {
            return first.value;
        }
        let last = keys[keys.len() - 1];
        if t >= last.time {
            return last.value;
        }
        let i = keys.partition_point(|k| k.time <= t).saturating_sub(1).min(keys.len() - 2);
        let (k0, k1) = (keys[i], keys[i + 1]);
        let h = (k1.time - k0.time).max(1e-6);
        let s = ((t - k0.time) / h).clamp(0.0, 1.0);
        let rotation = matches!(self.property, TrackProperty::Rotation | TrackProperty::RootRotation);
        let interp = if self.property.is_step() { KeyInterp::Step } else { k0.interp };
        match interp {
            KeyInterp::Step => k0.value,
            KeyInterp::Linear => {
                if rotation {
                    let (a, b) = (Quat::from_array(k0.value), Quat::from_array(k1.value));
                    a.slerp(b, s).normalize().to_array()
                } else {
                    lerp4(k0.value, k1.value, s)
                }
            }
            KeyInterp::Cubic => {
                // Keep neighbouring quaternions in one hemisphere so the
                // component-wise curve doesn't take the long way round.
                let align = |reference: [f32; 4], v: [f32; 4]| {
                    if rotation && dot4(reference, v) < 0.0 { v.map(|x| -x) } else { v }
                };
                let p0 = k0.value;
                let p1 = align(p0, k1.value);
                let tangent = |j: usize, pj: [f32; 4]| -> [f32; 4] {
                    if j == 0 || j + 1 >= keys.len() {
                        return [0.0; 4];
                    }
                    let prev = align(pj, keys[j - 1].value);
                    let next = align(pj, keys[j + 1].value);
                    let dt = (keys[j + 1].time - keys[j - 1].time).max(1e-6);
                    std::array::from_fn(|c| (next[c] - prev[c]) / dt)
                };
                let m0 = tangent(i, p0);
                let m1 = tangent(i + 1, p1);
                let (s2, s3) = (s * s, s * s * s);
                let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
                let h10 = s3 - 2.0 * s2 + s;
                let h01 = -2.0 * s3 + 3.0 * s2;
                let h11 = s3 - s2;
                let v: [f32; 4] = std::array::from_fn(|c| h00 * p0[c] + h10 * h * m0[c] + h01 * p1[c] + h11 * h * m1[c]);
                if rotation { Quat::from_array(v).normalize().to_array() } else { v }
            }
            KeyInterp::Custom => {
                let align = |reference: [f32; 4], v: [f32; 4]| {
                    if rotation && dot4(reference, v) < 0.0 { v.map(|x| -x) } else { v }
                };
                let p0 = k0.value;
                let p1 = align(p0, k1.value);
                let auto = |j: usize, pj: [f32; 4]| -> [f32; 4] {
                    if j == 0 || j + 1 >= keys.len() {
                        return [0.0; 4];
                    }
                    let prev = align(pj, keys[j - 1].value);
                    let next = align(pj, keys[j + 1].value);
                    let dt = (keys[j + 1].time - keys[j - 1].time).max(1e-6);
                    std::array::from_fn(|c| (next[c] - prev[c]) / dt)
                };
                let m0 = k0.tangent_out.map(|t| align(p0, t)).unwrap_or_else(|| auto(i, p0));
                let m1 = k1.tangent_in.map(|t| align(p1, t)).unwrap_or_else(|| auto(i + 1, p1));
                let (s2, s3) = (s * s, s * s * s);
                let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
                let h10 = s3 - 2.0 * s2 + s;
                let h01 = -2.0 * s3 + 3.0 * s2;
                let h11 = s3 - s2;
                let v: [f32; 4] = std::array::from_fn(|c| h00 * p0[c] + h10 * h * m0[c] + h01 * p1[c] + h11 * h * m1[c]);
                if rotation { Quat::from_array(v).normalize().to_array() } else { v }
            }
        }
    }
}

fn lerp4(a: [f32; 4], b: [f32; 4], s: f32) -> [f32; 4] {
    std::array::from_fn(|c| a[c] + (b[c] - a[c]) * s)
}

fn dot4(a: [f32; 4], b: [f32; 4]) -> f32 {
    a.iter().zip(&b).map(|(x, y)| x * y).sum()
}

/// A named marker fired when playback passes `time`.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
#[serde(default)]
pub struct ClipEvent {
    pub time: f32,
    pub name: String,
}

/// Parameters delivered with a clip event (a model clip's function name takes
/// one of each, like Unity's `AnimationEvent`); unused ones stay empty.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
#[serde(default)]
pub struct EventArgs {
    /// Index into [`AnimClip::events`].
    pub event: usize,
    pub float: f32,
    pub int: i32,
    pub string: String,
    /// An asset or object reference, by project path or name.
    pub object: String,
}

/// Rest translation of one bone of the rig a clip was made on.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
#[serde(default)]
pub struct SourceBone {
    pub name: String,
    pub translation: [f32; 3],
}

/// What a clip's tracks address.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ClipKind {
    /// Nodes and object properties by path (the original kind).
    #[default]
    Generic,
    /// A humanoid pose in muscle space ([`crate::muscle`]): `Muscle` tracks
    /// (one per muscle, named), the body root and optional limb goals. Any
    /// humanoid avatar can play it.
    Muscle,
}

impl ClipKind {
    fn is_generic(&self) -> bool {
        *self == ClipKind::Generic
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct AnimClip {
    /// Seconds.
    pub length: f32,
    pub looping: bool,
    /// Editor snapping only (the bake rate of a muscle clip).
    pub frame_rate: f32,
    pub tracks: Vec<AnimTrack>,
    pub events: Vec<ClipEvent>,
    /// Generic clips address nodes; muscle clips drive any humanoid.
    #[serde(skip_serializing_if = "ClipKind::is_generic")]
    pub kind: ClipKind,
    /// Muscle clip baked from an imported clip: that clip's name, played on
    /// rigs that are not humanoid.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub fallback: String,
    /// A clip taken from another model's skeleton (a clip library such as a
    /// Mixamo animation file): the rest translation of each of that rig's bones.
    /// A bone track is then played as the offset from this rest, scaled by the
    /// ratio of the playing rig's bone length to the source's, which converts
    /// units (centimetres to metres) and proportions at once. Empty for clips
    /// authored on the playing rig.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub source_rest: Vec<SourceBone>,
    /// Parameters of some of `events`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub event_args: Vec<EventArgs>,
    /// The same motion baked into muscle space, when the source rig is a
    /// humanoid: a humanoid rig plays this (so differing bone naming, rest
    /// pose and proportions retarget) instead of the bone tracks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub humanoid: Option<Box<AnimClip>>,
}

impl Default for AnimClip {
    fn default() -> Self {
        Self {
            length: 1.0,
            looping: true,
            frame_rate: 30.0,
            tracks: Vec::new(),
            events: Vec::new(),
            kind: ClipKind::Generic,
            fallback: String::new(),
            source_rest: Vec::new(),
            event_args: Vec::new(),
            humanoid: None,
        }
    }
}

impl AnimClip {
    pub fn track_index(&self, target: &str, property: TrackProperty) -> Option<usize> {
        self.tracks.iter().position(|t| t.target == target && t.property == property)
    }

    /// Rest translation of the source rig's bone `name`, for a clip taken from another model.
    pub fn source_rest_of(&self, name: &str) -> Option<[f32; 3]> {
        self.source_rest.iter().find(|b| b.name == name).map(|b| b.translation)
    }

    /// Track with this full address (object-property tracks).
    pub fn prop_track_index(&self, target: &str, property: TrackProperty, name: &str, slot: u8) -> Option<usize> {
        self.tracks.iter().position(|t| t.is(target, property, name, slot))
    }

    /// Object `old` (a path below the Animator) was renamed or moved to
    /// `new`: rewrite every track on it or below it. Returns how many tracks
    /// changed. Matches whole path components ("Arm" does not touch "Armor").
    pub fn rename_target(&mut self, old: &str, new: &str) -> usize {
        let mut n = 0;
        for t in &mut self.tracks {
            let rest = if t.target == old {
                Some("")
            } else {
                t.target.strip_prefix(old).filter(|r| !old.is_empty() && r.starts_with('/'))
            };
            if let Some(rest) = rest {
                t.target = if new.is_empty() { rest.trim_start_matches('/').to_owned() } else { format!("{new}{rest}") };
                n += 1;
            }
        }
        n
    }

    /// Re-root every track under `prefix` (the clip moves to an Animator that
    /// sits `prefix` above the objects it was authored against; used by
    /// prefab and avatar-build passes). An empty target becomes `prefix`.
    pub fn rebase_targets(&mut self, prefix: &str) {
        if prefix.is_empty() {
            return;
        }
        for t in &mut self.tracks {
            t.target = if t.target.is_empty() { prefix.to_owned() } else { format!("{prefix}/{}", t.target) };
        }
    }

    /// Wrap (looping) or clamp `t` into the clip.
    pub fn local_time(&self, t: f32, looping: bool) -> f32 {
        let len = self.length.max(1e-4);
        if looping { t.rem_euclid(len) } else { t.clamp(0.0, len) }
    }

    /// Sample track `i` at playback time `t`.
    pub fn sample_track(&self, i: usize, t: f32, looping: bool) -> [f32; 4] {
        self.tracks.get(i).map(|tr| tr.sample(self.local_time(t, looping))).unwrap_or([0.0; 4])
    }

    /// Latest key time over all tracks.
    pub fn last_key_time(&self) -> f32 {
        self.tracks.iter().filter_map(|t| t.keys.last()).map(|k| k.time).fold(0.0, f32::max)
    }
}

pub fn load_anim_clip(path: impl AsRef<Path>) -> Result<AnimClip> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut clip: AnimClip = ron::from_str(&text).with_context(|| format!("parsing animation clip {}", path.display()))?;
    for t in &mut clip.tracks {
        t.sort();
    }
    Ok(clip)
}

pub fn save_anim_clip(path: impl AsRef<Path>, clip: &AnimClip) -> Result<()> {
    let path = path.as_ref();
    let text = ron::ser::to_string_pretty(clip, ron::ser::PrettyConfig::default())?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn track(interp: KeyInterp, keys: &[(f32, f32)]) -> AnimTrack {
        AnimTrack {
            target: String::new(),
            property: TrackProperty::Translation,
            name: String::new(),
            slot: 0,
            keys: keys.iter().map(|&(time, v)| Keyframe { time, value: [v, 0.0, 0.0, 0.0], interp, ..Default::default() }).collect(),
        }
    }

    #[test]
    fn step_holds_until_next_key() {
        let t = track(KeyInterp::Step, &[(0.0, 1.0), (1.0, 3.0)]);
        assert_eq!(t.sample(0.99)[0], 1.0);
        assert_eq!(t.sample(1.0)[0], 3.0);
    }

    #[test]
    fn linear_interpolates_and_clamps() {
        let t = track(KeyInterp::Linear, &[(0.0, 0.0), (2.0, 4.0)]);
        assert!((t.sample(0.5)[0] - 1.0).abs() < 1e-5);
        assert_eq!(t.sample(-1.0)[0], 0.0);
        assert_eq!(t.sample(5.0)[0], 4.0);
    }

    #[test]
    fn cubic_is_smooth_through_keys() {
        let t = track(KeyInterp::Cubic, &[(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]);
        assert!((t.sample(1.0)[0] - 1.0).abs() < 1e-5);
        // Flat end tangent, Catmull-Rom middle tangent of 0: symmetric hump.
        let a = t.sample(0.5)[0];
        let b = t.sample(1.5)[0];
        assert!((a - b).abs() < 1e-5 && a > 0.4 && a < 0.6, "{a} {b}");
        // Ease in from the flat start.
        assert!(t.sample(0.1)[0] < 0.1 * 1.0);
    }

    #[test]
    fn custom_tangents_shape_the_segment() {
        let t = AnimTrack {
            target: String::new(),
            property: TrackProperty::Translation,
            name: String::new(),
            slot: 0,
            keys: vec![
                Keyframe { time: 0.0, value: [0.0; 4], interp: KeyInterp::Custom, tangent_out: Some([10.0, 0.0, 0.0, 0.0]), tangent_in: None },
                Keyframe { time: 1.0, value: [0.0; 4], interp: KeyInterp::Custom, tangent_out: None, tangent_in: Some([10.0, 0.0, 0.0, 0.0]) },
            ],
        };
        // Hermite with p0=p1=0 and m0=m1=10: v(s) = 10*(h10+h11) = 10*(2s^3-3s^2+s).
        // At s=0.25 this is 0.9375 (a positive bump), proving the explicit
        // tangents override the flat automatic tangents.
        assert!((t.sample(0.25)[0] - 0.9375).abs() < 1e-4, "{}", t.sample(0.25)[0]);
    }

    #[test]
    fn custom_falls_back_to_auto_tangents() {
        let keys = |interp: KeyInterp| AnimTrack {
            target: String::new(),
            property: TrackProperty::Translation,
            name: String::new(),
            slot: 0,
            keys: (0..3)
                .map(|i| Keyframe {
                    time: i as f32,
                    value: [i as f32, 0.0, 0.0, 0.0],
                    interp,
                    ..Default::default()
                })
                .collect(),
        };
        let cubic = keys(KeyInterp::Cubic);
        let custom = keys(KeyInterp::Custom);
        for t in [0.2, 0.5, 0.8, 1.2, 1.8] {
            let a = cubic.sample(t);
            let b = custom.sample(t);
            assert!((a[0] - b[0]).abs() < 1e-5, "t={t}: {a:?} vs {b:?}");
        }
    }

    #[test]
    fn rotation_linear_slerps() {
        let q0 = Quat::IDENTITY;
        let q1 = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let t = AnimTrack {
            target: String::new(),
            property: TrackProperty::Rotation,
            name: String::new(),
            slot: 0,
            keys: vec![
                Keyframe { time: 0.0, value: q0.to_array(), interp: KeyInterp::Linear, ..Default::default() },
                Keyframe { time: 1.0, value: q1.to_array(), interp: KeyInterp::Linear, ..Default::default() },
            ],
        };
        let mid = Quat::from_array(t.sample(0.5));
        assert!(mid.angle_between(Quat::from_rotation_y(std::f32::consts::FRAC_PI_4)) < 1e-4);
    }

    #[test]
    fn looping_wraps_and_one_shot_clamps() {
        let clip = AnimClip {
            length: 1.0,
            tracks: vec![track(KeyInterp::Linear, &[(0.0, 0.0), (1.0, 1.0)])],
            ..Default::default()
        };
        assert!((clip.sample_track(0, 1.25, true)[0] - 0.25).abs() < 1e-5);
        assert_eq!(clip.sample_track(0, 1.25, false)[0], 1.0);
    }

    #[test]
    fn set_key_replaces_and_sorts() {
        let mut t = track(KeyInterp::Linear, &[(1.0, 1.0)]);
        t.set_key(0.5, [2.0; 4]);
        t.set_key(1.0, [3.0; 4]);
        assert_eq!(t.keys.len(), 2);
        assert_eq!(t.keys[0].time, 0.5);
        assert_eq!(t.keys[1].value[0], 3.0);
    }

    // ------------------------------------------------ object-property tracks

    fn prop_track(property: TrackProperty, name: &str, keys: &[(f32, f32)]) -> AnimTrack {
        let mut t = AnimTrack::prop("Body", property, name, 0);
        for &(time, v) in keys {
            t.set_key(time, [v, 0.0, 0.0, 0.0]);
        }
        t
    }

    #[test]
    fn switches_step_whatever_the_key_says() {
        let mut t = prop_track(TrackProperty::Active, "", &[(0.0, 0.0), (1.0, 1.0)]);
        assert!(t.keys.iter().all(|k| k.interp == KeyInterp::Step));
        t.keys[0].interp = KeyInterp::Linear;
        assert_eq!(t.sample(0.99)[0], 0.0, "a switch never blends");
        assert_eq!(t.sample(1.0)[0], 1.0);
        let c = prop_track(TrackProperty::ComponentEnabled, "Light", &[(0.0, 1.0), (2.0, 0.0)]);
        assert_eq!(c.sample(1.9)[0], 1.0);
        assert_eq!(c.sample(2.0)[0], 0.0);
    }

    #[test]
    fn float_props_interpolate_linearly_by_default() {
        let t = prop_track(TrackProperty::BlendShape, "Smile", &[(0.0, 0.0), (2.0, 1.0)]);
        assert_eq!(t.keys[0].interp, KeyInterp::Linear);
        assert!((t.sample(0.5)[0] - 0.25).abs() < 1e-6);
        let m = prop_track(TrackProperty::MaterialFloat, "roughness", &[(0.0, 1.0), (1.0, 0.0)]);
        assert!((m.sample(0.25)[0] - 0.75).abs() < 1e-6);
        // Transform tracks keep their cubic default.
        let mut tr = AnimTrack::new("", TrackProperty::Translation);
        tr.set_key(0.0, [0.0; 4]);
        assert_eq!(tr.keys[0].interp, KeyInterp::Cubic);
    }

    #[test]
    fn colour_lanes_blend_together() {
        let mut t = AnimTrack::prop("Body", TrackProperty::MaterialColor, "base_color", 1);
        t.set_key(0.0, [1.0, 0.0, 0.0, 1.0]);
        t.set_key(1.0, [0.0, 0.0, 1.0, 0.5]);
        let v = t.sample(0.5);
        assert!((v[0] - 0.5).abs() < 1e-6 && (v[2] - 0.5).abs() < 1e-6 && (v[3] - 0.75).abs() < 1e-6);
        assert_eq!(TrackProperty::MaterialColor.lanes(), 4);
        assert_eq!(TrackProperty::MaterialOffset.lanes(), 2);
    }

    #[test]
    fn property_tracks_round_trip_through_ron() {
        let clip = AnimClip {
            length: 2.0,
            tracks: vec![
                prop_track(TrackProperty::Active, "", &[(0.0, 1.0), (1.0, 0.0)]),
                prop_track(TrackProperty::ComponentEnabled, "Light", &[(0.0, 1.0)]),
                prop_track(TrackProperty::BlendShape, "Smile", &[(0.0, 0.0), (1.0, 1.0)]),
                {
                    let mut t = AnimTrack::prop("Body/Skin", TrackProperty::MaterialColor, "base_color", 2);
                    t.set_key(0.0, [1.0, 0.5, 0.25, 1.0]);
                    t
                },
                prop_track(TrackProperty::MaterialFloat, "metallic", &[(0.0, 0.5)]),
                prop_track(TrackProperty::MaterialOffset, "albedo_offset", &[(0.0, 0.1)]),
            ],
            ..Default::default()
        };
        let text = ron::ser::to_string_pretty(&clip, ron::ser::PrettyConfig::default()).unwrap();
        let back: AnimClip = ron::from_str(&text).unwrap();
        assert_eq!(back, clip);
        assert_eq!(back.tracks[3].slot, 2);
        assert_eq!(back.tracks[3].name, "base_color");
    }

    #[test]
    fn old_clips_load_unchanged_and_save_without_new_fields() {
        let old = r#"(
            length: 1.5,
            looping: false,
            frame_rate: 24.0,
            tracks: [(
                target: "Arm",
                property: Rotation,
                keys: [(time: 0.0, value: (0.0, 0.0, 0.0, 1.0), interp: Linear)],
            ), (
                target: "",
                property: LightIntensity,
                keys: [(time: 0.5, value: (3.0, 0.0, 0.0, 0.0))],
            )],
            events: [(time: 0.25, name: "step")],
        )"#;
        let clip: AnimClip = ron::from_str(old).unwrap();
        assert_eq!(clip.tracks.len(), 2);
        assert_eq!(clip.tracks[0].property, TrackProperty::Rotation);
        assert!(clip.tracks[0].name.is_empty() && clip.tracks[0].slot == 0);
        assert_eq!(clip.tracks[1].keys[0].interp, KeyInterp::Cubic);
        assert_eq!(clip.events[0].name, "step");
        let text = ron::ser::to_string_pretty(&clip, ron::ser::PrettyConfig::default()).unwrap();
        assert!(!text.contains("name: \"\"") && !text.contains("slot"), "{text}");
    }

    #[test]
    fn muscle_clips_round_trip_and_generic_clips_do_not_write_the_new_fields() {
        let mut clip = AnimClip { kind: ClipKind::Muscle, fallback: "Wave".into(), ..Default::default() };
        let mut t = AnimTrack::prop("", TrackProperty::Muscle, "Jaw Close", 0);
        t.set_key(0.0, [0.5, 0.0, 0.0, 0.0]);
        clip.tracks.push(t);
        let mut q = AnimTrack::new("", TrackProperty::RootRotation);
        q.set_key(0.0, [0.0, 0.0, 0.0, 1.0]);
        q.set_key(1.0, Quat::from_rotation_y(1.0).to_array());
        clip.tracks.push(q);
        let text = ron::ser::to_string_pretty(&clip, ron::ser::PrettyConfig::default()).unwrap();
        let back: AnimClip = ron::from_str(&text).unwrap();
        assert_eq!(back, clip);
        assert_eq!(back.kind, ClipKind::Muscle);
        assert_eq!(back.tracks[0].keys[0].interp, KeyInterp::Linear, "muscle tracks interpolate linearly");
        // The root quaternion slerps like a rotation track.
        let mid = Quat::from_array(back.tracks[1].sample(0.5));
        assert!(mid.angle_between(Quat::from_rotation_y(0.5)) < 1e-4);
        let generic = ron::ser::to_string_pretty(&AnimClip::default(), ron::ser::PrettyConfig::default()).unwrap();
        assert!(!generic.contains("kind") && !generic.contains("fallback"), "{generic}");
        assert!(TrackProperty::Muscle.is_muscle() && !TrackProperty::Rotation.is_muscle());
    }

    #[test]
    fn track_addresses_distinguish_name_and_slot() {
        let mut clip = AnimClip::default();
        clip.tracks.push(AnimTrack::prop("Body", TrackProperty::MaterialFloat, "metallic", 0));
        clip.tracks.push(AnimTrack::prop("Body", TrackProperty::MaterialFloat, "metallic", 1));
        clip.tracks.push(AnimTrack::prop("Body", TrackProperty::MaterialFloat, "roughness", 1));
        assert_eq!(clip.prop_track_index("Body", TrackProperty::MaterialFloat, "metallic", 1), Some(1));
        assert_eq!(clip.prop_track_index("Body", TrackProperty::MaterialFloat, "roughness", 1), Some(2));
        assert_eq!(clip.prop_track_index("Body", TrackProperty::MaterialFloat, "roughness", 0), None);
        // Blendshape and active tracks ignore the slot.
        clip.tracks.push(AnimTrack::prop("Body", TrackProperty::BlendShape, "Smile", 0));
        assert_eq!(clip.prop_track_index("Body", TrackProperty::BlendShape, "Smile", 7), Some(3));
    }

    #[test]
    fn renaming_and_rebasing_rewrite_whole_path_components() {
        let mut clip = AnimClip::default();
        for target in ["", "Arm", "Arm/Hand", "Armor", "Leg/Arm"] {
            clip.tracks.push(AnimTrack::prop(target, TrackProperty::Active, "", 0));
        }
        assert_eq!(clip.rename_target("Arm", "Limb"), 2);
        let targets: Vec<&str> = clip.tracks.iter().map(|t| t.target.as_str()).collect();
        assert_eq!(targets, ["", "Limb", "Limb/Hand", "Armor", "Leg/Arm"]);
        clip.rebase_targets("Avatar");
        let targets: Vec<&str> = clip.tracks.iter().map(|t| t.target.as_str()).collect();
        assert_eq!(targets, ["Avatar", "Avatar/Limb", "Avatar/Limb/Hand", "Avatar/Armor", "Avatar/Leg/Arm"]);
    }
}
