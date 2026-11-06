//! Built-in gameplay kit: reusable components the project templates are made
//! of (character controllers, hitscan weapons, health, HUD bindings,
//! click-to-move, RTS camera + selection, VR player rig and grabbing, the
//! in-game controls / remap menu). Every
//! piece is an ordinary [`TypedComponent`], so games can use, tweak or replace
//! them like any other component.

mod combat;
mod avatar;
mod character;
mod character_movement;
mod controls_menu;
mod events;
mod hud;
mod rts;
mod swim;
mod topdown;
mod vr_character;

pub use combat::{
    DamageEvent, DeathAction, DeathEvent, Health, HitscanWeapon, ScoreEvent, TargetDummy, WeaponTrigger, spread_direction,
};
pub use avatar::{AvatarVariant, TemplateAvatar};
pub use character::{FirstPersonCharacter, FollowCamera, ThirdPersonCharacter, boom_position, look_rotation};
pub use controls_menu::ControlsMenu;
pub use hud::{HudBinding, HudValue, format_hud};
pub use rts::{
    ResourceAmount, ResourceStore, RtsCamera, RtsSelection, Selectable, edge_pan, formation_offsets, formation_slots,
    rect_contains, rts_camera_pose, screen_rect,
};
pub use swim::{Climb, Swimmer, breathe, float_accel, stroke, swims};
pub use events::{AnimationEventBinding, AnimationEventBindings, TriggerEvents};
pub use topdown::{ClickToMove, InteractEvent, InteractMode, Interactable};
pub use vr_character::{Grabbable, VrAvatar, VrHand, VrLocomotion, VrTurning, VrCharacter, teleport_arc};

use crate::ComponentRegistry;

pub(crate) fn register_kit(registry: &mut ComponentRegistry) {
    registry.register::<TemplateAvatar>();
    registry.register::<Health>();
    registry.register::<HitscanWeapon>();
    registry.register::<TargetDummy>();
    registry.register::<FirstPersonCharacter>();
    registry.register::<Swimmer>();
    registry.register::<ThirdPersonCharacter>();
    registry.register::<FollowCamera>();
    registry.register::<ClickToMove>();
    registry.register::<Interactable>();
    registry.register::<TriggerEvents>();
    registry.register::<AnimationEventBindings>();
    registry.register::<RtsCamera>();
    registry.register::<RtsSelection>();
    registry.register::<Selectable>();
    registry.register::<ResourceStore>();
    registry.register::<HudBinding>();
    registry.register::<VrCharacter>();
    registry.register::<Grabbable>();
    registry.register::<VrAvatar>();
    registry.register::<ControlsMenu>();
}

/// Ids of every object carrying a component of type `T`.
pub(crate) fn objects_with<T: crate::Component>(ctx: &crate::ComponentCtx) -> Vec<crate::ObjectId> {
    ctx.object_ids
        .iter()
        .copied()
        .filter(|id| ctx.get::<T>(*id).is_some())
        .collect()
}

// Source compatibility for gameplay plugins written before the Character naming.
pub type FirstPersonController = FirstPersonCharacter;
pub type ThirdPersonController = ThirdPersonCharacter;
pub type VrPlayer = VrCharacter;

pub use character_movement::{CharacterBraking, CharacterMovement, CharacterMovementMode};

#[cfg(test)]
mod character_compatibility_tests {
    use crate::*;
    #[test]
    fn old_character_names_load_once_and_save_under_the_new_names() {
        let mut r = ComponentRegistry::with_builtins();
        for (old, new, data) in [
            ("Character Controller", "Character", "(radius: 0.42)"),
            ("First Person Controller", "First Person Character", "(walk_speed: 3.2, acceleration: 12.0)"),
            ("Third Person Controller", "Third Person Character", "(walk_speed: 3.2)"),
            ("VR Player", "VR Character", "(move_speed: 2.0)"),
        ] {
            let c = r.load(old, data).expect(old);
            assert_eq!(c.type_name(), new);
            assert!(r.load(new, &c.save()).is_some());
            assert!(!r.names().any(|n| n == old));
            assert_eq!(r.names().filter(|n| *n == new).count(), 1);
            r.set_role_override(old, Some(NetRole::Server));
            assert_eq!(r.default_role(new), NetRole::Server);
        }
        assert_eq!(r.canonical_key("VR Player#1"), "VR Character#1");
        r.set_role_override("VR Character", None);
        assert_eq!(r.default_role("VR Player"), NetRole::Both);
        let c = r.load("Character Controller", "(radius: 0.42)").unwrap();
        let c = c.as_any().downcast_ref::<Character>().unwrap();
        assert_eq!(c.radius, 0.42);
        assert!(c.use_default_capsule && c.capsule_affects_water);
    }
}
