//! Template avatar selection and animation-controller inputs. The Animator
//! owns playback/blending; this component only chooses visuals and parameters.

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::{Animator, ComponentCtx, FirstPersonCharacter, NetworkAvatar, ObjectRef, ThirdPersonCharacter, TypedComponent};

/// Which template visual is selected.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AvatarVariant {
    /// The male template.
    #[default]
    Male,
    /// The female template.
    Female,
}

/// Switches between two template visuals on one player.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct TemplateAvatar {
    /// Which visual is selected.
    pub variant: AvatarVariant,
    /// The male visual.
    pub male: ObjectRef,
    /// The female visual.
    pub female: ObjectRef,
    /// Stable movement/physics root. Visuals can change without replacing it.
    pub player: ObjectRef,
    /// Input action that switches the variant.
    pub switch_action: String,
    /// Network player prefab: copy the local VR rig's selection for the owner.
    pub follow_local_player: bool,
    /// The player's pose last frame (runtime).
    #[serde(skip)]
    pub previous: Option<(Vec3, Quat)>,
    /// The variant the visuals were last set to (runtime).
    #[serde(skip)]
    pub applied: Option<AvatarVariant>,
    /// Smoothed position-delta velocity, for players without a local walker.
    #[serde(skip)]
    pub smoothed_velocity: Vec3,
}

impl Default for TemplateAvatar {
    fn default() -> Self {
        Self {
            variant: AvatarVariant::Male, male: ObjectRef::NONE, female: ObjectRef::NONE,
            player: ObjectRef::NONE, switch_action: "Switch Avatar".into(),
            follow_local_player: false, previous: None, applied: None, smoothed_velocity: Vec3::ZERO,
        }
    }
}

impl TypedComponent for TemplateAvatar {
    const NAME: &'static str = "Template Avatar";

    fn update(&mut self, ctx: &mut ComponentCtx) {
        let remote = ctx.get::<NetworkAvatar>(self.player).is_some_and(|av| av.remote);
        if remote {
            if let Some(crate::NetValue::Bool(female)) = ctx.net_var(self.player, "zestengine.avatar_female") {
                self.variant = if *female { AvatarVariant::Female } else { AvatarVariant::Male };
            }
        } else if self.follow_local_player {
            if let Some(variant) = ctx.object_ids.iter().filter_map(|id| ctx.get::<TemplateAvatar>(*id))
                .find(|a| !a.follow_local_player && ctx.get::<crate::VrCharacter>(a.player).is_some()).map(|a| a.variant)
            {
                self.variant = variant;
            }
            let female = self.variant == AvatarVariant::Female;
            if ctx.net_var(self.player, "zestengine.avatar_female") != Some(&crate::NetValue::Bool(female)) {
                ctx.set_net_var(self.player, "zestengine.avatar_female", female);
            }
        } else if !ctx.ui_text_editing() && ctx.input.pressed(&self.switch_action) {
            self.variant = match self.variant { AvatarVariant::Male => AvatarVariant::Female, AvatarVariant::Female => AvatarVariant::Male };
        }
        if self.applied != Some(self.variant) {
            for (target, on) in [(self.male, self.variant == AvatarVariant::Male), (self.female, self.variant == AvatarVariant::Female)] {
                if target.is_set() {
                    ctx.set_active(target, on);
                }
            }
            self.applied = Some(self.variant);
        }
        let Some(index) = self.player.id().and_then(|id| ctx.index_of(id)) else { return };
        let (_, rotation, position) = ctx.world_transforms[index].to_scale_rotation_translation();
        let previous = self.previous.replace((position, rotation));
        if ctx.dt <= 0.0 { return }
        let (moved, turn) = previous.map(|(p, r)| {
            let velocity = (position - p) / ctx.dt;
            let old = r * Vec3::NEG_Z;
            let new = rotation * Vec3::NEG_Z;
            let angle = old.cross(new).y.atan2(old.dot(new));
            (velocity, angle / ctx.dt)
        }).unwrap_or_default();
        let mut grounded = true;
        let mut crouched = false;
        let mut aiming = false;
        // A locally driven controller knows its velocity. The position delta
        // doesn't: physics steps at a fixed rate, so frames alternate between
        // moving and not, and the locomotion blend jumped every frame.
        let mut walker_velocity = None;
        if let Some(fp) = ctx.get::<FirstPersonCharacter>(self.player) {
            grounded = fp.walker.is_grounded();
            crouched = fp.crouched;
            walker_velocity = Some(fp.walker.velocity());
        }
        if let Some(tp) = ctx.get::<ThirdPersonCharacter>(self.player) {
            grounded = tp.walker.is_grounded();
            aiming = tp.aiming;
            walker_velocity = Some(tp.walker.velocity());
        }
        self.smoothed_velocity = self.smoothed_velocity.lerp(moved, 1.0 - (-12.0 * ctx.dt).exp());
        let velocity = match walker_velocity {
            Some(v) if !remote => v,
            _ => self.smoothed_velocity,
        };
        let local = rotation.inverse() * velocity;
        let speed = velocity.with_y(0.0).length().min(12.0);
        for target in [self.male, self.female] {
            ctx.modify::<Animator>(target, move |animator| {
                animator.set_float("Speed", speed);
                animator.set_float("Forward", -local.z);
                animator.set_float("Right", local.x);
                animator.set_float("Turn", turn.clamp(-8.0, 8.0));
                animator.set_float("Vertical", velocity.y);
                animator.set_bool("Grounded", grounded);
                animator.set_bool("Crouched", crouched);
                animator.set_bool("Aiming", aiming);
            });
        }
    }
}
