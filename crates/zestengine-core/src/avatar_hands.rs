//! Shared hand poses for full-body avatars, independent of camera and tracking device.
use serde::{Deserialize, Serialize};
use crate::{ComponentCtx, NetValue, NetworkAvatar, NetworkVariables, ObjectRef, TypedComponent};

/// Stable gesture values, also used by the avatar's replicated hand parameters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum HandPose {
    #[default]
    Idle = 0,
    Fist = 1,
    Open = 2,
    Point = 3,
    Peace = 4,
    Rock = 5,
}
impl HandPose {
    pub const ALL: [Self; 6] = [Self::Idle, Self::Fist, Self::Open, Self::Point, Self::Peace, Self::Rock];
    pub fn label(self) -> &'static str {
        match self { Self::Idle => "Idle", Self::Fist => "Fist", Self::Open => "Open", Self::Point => "Point", Self::Peace => "Peace", Self::Rock => "Rock and roll" }
    }
    pub fn from_index(index: i64) -> Option<Self> { Self::ALL.get(usize::try_from(index).ok()?).copied() }
    /// Controller fallback when finger tracking is unavailable.
    pub fn from_grip_trigger(grip: f32, trigger: f32, fallback: Self) -> Self {
        if trigger > 0.5 { Self::Fist } else if grip > 0.5 { Self::Point } else { fallback }
    }
}

/// Thumb, index, middle, ring, little: curl 0..1 and spread in degrees.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FingerPose { pub curl: [f32; 5], pub spread: [f32; 5] }
impl FingerPose {
    pub fn blend(self, to: Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self { curl: std::array::from_fn(|i| self.curl[i] + (to.curl[i] - self.curl[i]) * t),
            spread: std::array::from_fn(|i| self.spread[i] + (to.spread[i] - self.spread[i]) * t) }
    }
}

/// Finger animation layered after body IK. Put this on the skinned avatar model.
/// The defaults bind the shipped male/female rigs; names and axes are editable.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AvatarHands {
    pub player: ObjectRef,
    /// Read pose actions and controller grip/trigger; false previews authored poses.
    pub read_input: bool,
    pub left: HandPose,
    pub right: HandPose,
    pub blend_rate: f32,
    pub weight: f32,
    pub poses: [FingerPose; 6],
    /// Bone names: left/right, thumb/index/middle/ring/little, proximal to distal.
    pub bones: [[[String; 3]; 5]; 2],
    /// Derive bend axes from the mapped hand geometry (disable for manual axes).
    pub automatic_axes: bool,
    pub curl_axis: [f32; 3],
    pub spread_axis: [f32; 3],
    pub curl_degrees: [f32; 3],
    pub thumb_curl_degrees: [f32; 3],
    /// Actions use '<prefix> <pose label>', e.g. 'Left Hand Peace'.
    pub action_prefix: [String; 2],
    #[serde(skip)]
    pub current: Option<[FingerPose; 2]>,
    #[serde(skip)]
    pub selected: [HandPose; 2],
}
impl Default for AvatarHands {
    fn default() -> Self {
        Self {
            player: ObjectRef::NONE, read_input: true, left: HandPose::Idle, right: HandPose::Idle,
            blend_rate: 18.0, weight: 1.0,
            poses: [
                FingerPose { curl: [0.15, 0.12, 0.18, 0.23, 0.28], spread: [0.0; 5] },
                FingerPose { curl: [0.85, 1.0, 1.0, 1.0, 1.0], spread: [0.0; 5] },
                FingerPose { curl: [0.0; 5], spread: [8.0, 4.0, 0.0, -3.0, -7.0] },
                FingerPose { curl: [0.6, 0.0, 1.0, 1.0, 1.0], spread: [0.0; 5] },
                FingerPose { curl: [0.8, 0.0, 0.0, 1.0, 1.0], spread: [0.0, 9.0, -4.0, 0.0, 0.0] },
                FingerPose { curl: [0.8, 0.0, 1.0, 1.0, 0.0], spread: [0.0, 5.0, 0.0, 0.0, -8.0] },
            ],
            bones: std::array::from_fn(|side| std::array::from_fn(|finger| std::array::from_fn(|joint|
                format!("{}_{:02}_{}", ["thumb", "index", "middle", "ring", "pinky"][finger], joint + 1, ["l", "r"][side])))),
            automatic_axes: true,
            curl_axis: [1.0, 0.0, 0.0], spread_axis: [0.0, 0.0, 1.0],
            curl_degrees: [65.0, 85.0, 65.0], thumb_curl_degrees: [35.0, 45.0, 45.0],
            action_prefix: ["Left Hand".into(), "Right Hand".into()], current: None,
            selected: [HandPose::Idle; 2],
        }
    }
}
impl AvatarHands {
    pub fn posed(&self) -> [FingerPose; 2] {
        self.current.unwrap_or([self.poses[self.left as usize], self.poses[self.right as usize]])
    }
    pub fn advance(&mut self, selected: [HandPose; 2], dt: f32) {
        self.selected = selected;
        let target = selected.map(|p| self.poses[p as usize]);
        let old = self.current.unwrap_or(target);
        let blend = if self.blend_rate <= 0.0 { 1.0 } else { 1.0 - (-self.blend_rate * dt.max(0.0)).exp() };
        self.current = Some(std::array::from_fn(|i| old[i].blend(target[i], blend)));
    }
}
impl TypedComponent for AvatarHands {
    const NAME: &'static str = "Avatar Hands";
    fn update(&mut self, ctx: &mut ComponentCtx) {
        const VARS: [&str; 2] = ["zestengine.hand_left", "zestengine.hand_right"];
        let mut selected = [self.left, self.right];
        let remote = ctx.get::<NetworkAvatar>(self.player).is_some_and(|a| a.remote);
        if remote {
            for side in 0..2 {
                if let Some(NetValue::Int(value)) = ctx.net_var(self.player, VARS[side]) {
                    selected[side] = HandPose::from_index(*value).unwrap_or(HandPose::Idle);
                }
            }
        } else if self.read_input && !ctx.ui_text_editing() {
            let vr = ctx.screen.vr;
            for side in 0..2 {
                let explicit = HandPose::ALL.into_iter().rev().find(|pose| ctx.input.down(&format!("{} {}", self.action_prefix[side], pose.label())));
                selected[side] = explicit.unwrap_or_else(|| {
                    if vr.active {
                        let (grip, trigger) = if side == 0 { (vr.left_grip, vr.left_trigger_value.max(vr.left_trigger as u8 as f32)) }
                            else { (vr.right_grip, vr.right_trigger_value.max(vr.right_trigger as u8 as f32)) };
                        HandPose::from_grip_trigger(grip, trigger, selected[side])
                    } else { selected[side] }
                });
            }
        }
        if !remote && ctx.get::<NetworkVariables>(self.player).is_some() {
            for side in 0..2 {
                let value = selected[side] as i64;
                if ctx.net_var(self.player, VARS[side]) != Some(&NetValue::Int(value)) {
                    ctx.set_net_var(self.player, VARS[side], value);
                }
            }
        }
        self.advance(selected, ctx.dt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gesture_fingers_and_blending() {
        let mut hands = AvatarHands::default();
        let point = hands.poses[HandPose::Point as usize];
        let peace = hands.poses[HandPose::Peace as usize];
        let rock = hands.poses[HandPose::Rock as usize];
        assert_eq!(&point.curl[1..], &[0.0, 1.0, 1.0, 1.0]);
        assert_eq!(&peace.curl[1..], &[0.0, 0.0, 1.0, 1.0]);
        assert_eq!(&rock.curl[1..], &[0.0, 1.0, 1.0, 0.0]);
        hands.advance([HandPose::Open; 2], 0.0);
        hands.advance([HandPose::Fist, HandPose::Open], 1.0 / 90.0);
        assert!(hands.posed()[0].curl[1] > 0.0 && hands.posed()[0].curl[1] < 1.0);
        assert_eq!(hands.posed()[1].curl[1], 0.0);
        assert_eq!(HandPose::from_grip_trigger(1.0, 0.0, HandPose::Idle), HandPose::Point);
        assert_eq!(HandPose::from_grip_trigger(1.0, 1.0, HandPose::Idle), HandPose::Fist);
        assert_eq!(HandPose::from_index(999), None);
    }
}
