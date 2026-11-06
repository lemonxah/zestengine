//! Shared avatar visuals and the normal ZestEngine Animator Controller used by
//! every gameplay starter. Gameplay roots/colliders never change on selection.

use anyhow::Result;
use zestengine_assets::{
    AnimCondition, AnimLayer, AnimParamKind, AnimParameter, AnimState, AnimTransition,
    AnimatorController, CondOp, Motion, ObjectSource, StateMachine,
};
use zestengine_core::{
    Animator, AvatarDescriptor, BlendMotion, BlendTree, HumanoidIk, IkSource, TemplateAvatar,
};
use glam::Quat;
use std::path::Path;

use super::builder::Scene;

pub const MALE: &str = "models/avatars/default/male.glb";
pub const FEMALE: &str = "models/avatars/default/female.glb";
pub const CONTROLLER: &str = "animations/default_locomotion.animctrl";

pub fn visuals(s: &mut Scene, parent: usize, player: usize, vr: bool) -> usize {
    let root = s.empty("Avatar", Some(parent), [0.0; 3]);
    let body = s.empty("Avatar Body", Some(root), [0.0; 3]);
    s.character_layer(root);
    s.character_layer(body);
    let mut models = Vec::new();
    for (name, path) in [("Male", MALE), ("Female", FEMALE)] {
        let model = s.model(name, path, Some(body), [0.0; 3], 1.0);
        s.character_layer(model);
        s.rot(model, Quat::from_rotation_y(std::f32::consts::PI));
        // The exported mesh has skin and joint-detail material primitives.
        if let ObjectSource::Model { extra_meshes, .. } = &mut s.entries[model].source {
            extra_meshes.push(1);
        }
        s.with(
            model,
            Animator {
                controller: CONTROLLER.into(),
                ..Default::default()
            },
        );
        let roles = [
            ("hips", "pelvis"),
            ("head", "Head"),
            ("neck", "neck_01"),
            ("chest", "spine_03"),
            ("left_shoulder", "clavicle_l"),
            ("right_shoulder", "clavicle_r"),
            ("left_upper_arm", "upperarm_l"),
            ("right_upper_arm", "upperarm_r"),
            ("left_lower_arm", "lowerarm_l"),
            ("right_lower_arm", "lowerarm_r"),
            ("left_hand", "hand_l"),
            ("right_hand", "hand_r"),
            ("left_upper_leg", "thigh_l"),
            ("right_upper_leg", "thigh_r"),
            ("left_lower_leg", "calf_l"),
            ("right_lower_leg", "calf_r"),
            ("left_foot", "foot_l"),
            ("right_foot", "foot_r"),
        ];
        s.with(
            model,
            AvatarDescriptor {
                bone_map: roles
                    .into_iter()
                    .map(|(r, b)| (r.into(), b.into()))
                    .collect(),
                ..Default::default()
            },
        );
        s.with(model, zestengine_core::AvatarHands { player: s.r(player), read_input: vr, ..Default::default() });
        if vr {
            // Auto resolves remote Network Avatar targets and local tracking.
            s.with(
                model,
                HumanoidIk {
                    tracking_root: s.r(player),
                    source: IkSource::Auto,
                    ..Default::default()
                },
            );
        }
        if name == "Female" {
            s.hidden(model);
        }
        models.push(model);
    }
    s.with(
        root,
        TemplateAvatar {
            male: s.r(models[0]),
            female: s.r(models[1]),
            player: s.r(player),
            ..Default::default()
        },
    );
    root
}

pub fn controller() -> AnimatorController {
    let tree = |name: &str, parameter: &str, motions: &[(&str, f32)]| {
        Motion::BlendTree(BlendTree {
            name: name.into(),
            parameter: parameter.into(),
            parameter_y: String::new(),
            kind: Default::default(),
            normalize_direct: false,
            motions: motions
                .iter()
                .map(|(clip, value)| BlendMotion {
                    clip: (*clip).into(),
                    position: [*value, 0.0],
                    ..Default::default()
                })
                .collect(),
        })
    };
    let directional_tree = |name: &str, motions: &[(&str, f32, f32)]| {
        Motion::BlendTree(BlendTree {
            name: name.into(),
            parameter: "Right".into(),
            parameter_y: "Forward".into(),
            kind: Default::default(),
            normalize_direct: false,
            motions: motions
                .iter()
                .map(|(clip, x, y)| BlendMotion {
                    clip: (*clip).into(),
                    position: [*x, *y],
                    ..Default::default()
                })
                .collect(),
        })
    };
    let mut states = vec![
        AnimState {
            name: "Locomotion".into(),
            motion: directional_tree(
                "Locomotion",
                &[
                    ("Idle_Loop", 0.0, 0.0),
                    ("Walk_Loop", 0.0, 1.6),
                    ("Jog_Fwd_Loop", 0.0, 4.5),
                    ("Sprint_Loop", 0.0, 7.0),
                    ("Walking_Backwards", 0.0, -1.6),
                    ("Running_Strafe_Left", -4.5, 0.0),
                    ("Running_Strafe_Right", 4.5, 0.0),
                ],
            ),
            ..Default::default()
        },
        AnimState {
            name: "Crouch".into(),
            motion: directional_tree(
                "Crouch",
                &[("Crouch_Idle_Loop", 0.0, 0.0), ("Crouch_Fwd_Loop", 0.0, 2.2)],
            ),
            ..Default::default()
        },
        AnimState {
            name: "Air".into(),
            motion: tree(
                "Air",
                "Vertical",
                &[("Jump_Loop", -1.0), ("Jump_Start", 2.0)],
            ),
            ..Default::default()
        },
        AnimState {
            name: "Land".into(),
            motion: Motion::Clip("Jump_Land".into()),
            speed: 1.6,
            looping: false,
            ..Default::default()
        },
    ];
    for (i, state) in states.iter_mut().enumerate() {
        state.position = [i as f32 * 210.0, 0.0];
    }
    let condition = |p: &str, op, threshold| AnimCondition {
        parameter: p.into(),
        op,
        threshold,
    };
    let transition = |from: &str, to: &str, conditions| AnimTransition {
        from: Some(from.into()),
        to: to.into(),
        conditions,
        duration: 0.12,
        interruptible: true,
        ..Default::default()
    };
    let transitions = vec![
        transition(
            "Locomotion",
            "Air",
            vec![condition("Grounded", CondOp::IfNot, 0.0)],
        ),
        transition("Crouch", "Air", vec![condition("Grounded", CondOp::IfNot, 0.0)]),
        transition("Air", "Land", vec![condition("Grounded", CondOp::If, 0.0)]),
        transition(
            "Locomotion",
            "Crouch",
            vec![condition("Crouched", CondOp::If, 0.0)],
        ),
        transition(
            "Crouch",
            "Locomotion",
            vec![condition("Crouched", CondOp::IfNot, 0.0)],
        ),
        // Land back to walk: the moment the player moves, cut the landing
        // short and blend straight into locomotion (no sliding while the
        // landing clip is still playing).
        transition(
            "Land",
            "Locomotion",
            vec![condition("Speed", CondOp::Greater, 0.3)],
        ),
        // Otherwise finish the sped-up landing and return to locomotion.
        AnimTransition {
            has_exit_time: true,
            exit_time: 0.4,
            ..transition("Land", "Locomotion", Vec::new())
        },
    ];
    let parameters = [
        ("Speed", AnimParamKind::Float, 0.0),
        ("Forward", AnimParamKind::Float, 0.0),
        ("Right", AnimParamKind::Float, 0.0),
        ("Turn", AnimParamKind::Float, 0.0),
        ("Vertical", AnimParamKind::Float, 0.0),
        ("Grounded", AnimParamKind::Bool, 1.0),
        ("Crouched", AnimParamKind::Bool, 0.0),
        // Read by a Weapon Hold to switch its weapon to the aim pose.
        ("Aiming", AnimParamKind::Bool, 0.0),
    ]
    .into_iter()
    .map(|(name, kind, default)| AnimParameter {
        name: name.into(),
        kind,
        default,
    })
    .collect();
    AnimatorController {
        parameters,
        layers: vec![AnimLayer {
            machine: StateMachine {
                states,
                transitions,
                default_state: "Locomotion".into(),
                ..Default::default()
            },
            ..Default::default()
        }],
    }
}

pub fn write_controller(root: &Path) -> Result<()> {
    let path = root.join(CONTROLLER);
    std::fs::create_dir_all(path.parent().unwrap())?;
    zestengine_assets::save_anim_controller(path, &controller())
}
