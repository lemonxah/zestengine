//! Inspectors and viewport gizmos for the built-in gameplay kit components
//! (zestengine-core `kit`): controllers, weapons, health, HUD bindings,
//! click-to-move, RTS camera / selection and the VR rig.

use zestengine_core::{
    BoneAttach, ClickToMove, ConflictPolicy, ControlsMenu, DeathAction, FirstPersonCharacter, FollowCamera, Grabbable, Health, HitscanWeapon, HudBinding, HudValue,
    AnimationEventBindings, InteractMode, Interactable, TriggerEvents, ResourceAmount, ResourceStore, RtsCamera, RtsSelection, Selectable, Swimmer, TargetDummy, ThirdPersonCharacter,
    VrAvatar, VrLocomotion, VrCharacter, WeaponFollow, WeaponHold, WeaponPoseRequest, WeaponPoseSlot, WeaponTrigger,
};
use egui::{DragValue, Ui};
use zestengine_core::ActionType;
use glam::Vec3;

use crate::components::{EditorComponents, Gizmo, GizmoEdit, GizmoSpec, Inspect, InspectCtx};
use crate::actions_ui::event_actions_ui;
use crate::theme;
use crate::widgets;

pub(crate) fn register(e: &mut EditorComponents) {
    e.register::<zestengine_core::TemplateAvatar>();
    e.register::<zestengine_core::AvatarHands>();
    e.register::<Health>();
    e.register::<HitscanWeapon>();
    e.register::<TargetDummy>();
    e.register::<FirstPersonCharacter>();
    e.register::<Swimmer>();
    e.register::<ThirdPersonCharacter>();
    e.register::<FollowCamera>();
    e.register::<BoneAttach>();
    e.register::<WeaponHold>();
    e.register::<ClickToMove>();
    e.register::<Interactable>();
    e.register::<TriggerEvents>();
    e.register::<AnimationEventBindings>();
    e.register::<RtsCamera>();
    e.register::<RtsSelection>();
    e.register::<Selectable>();
    e.register::<ResourceStore>();
    e.register::<HudBinding>();
    e.register::<VrCharacter>();
    e.register::<Grabbable>();
    e.register::<VrAvatar>();
    e.register::<ControlsMenu>();
}

impl Inspect for zestengine_core::TemplateAvatar {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        use zestengine_core::AvatarVariant;
        let mut changed = false;
        combo(ui, "avatar-variant", "Avatar", &mut self.variant,
            &[(AvatarVariant::Male, "Male"), (AvatarVariant::Female, "Female")], &mut changed);
        changed |= ctx.object_ref(ui, "Male model", &mut self.male);
        changed |= ctx.object_ref(ui, "Female model", &mut self.female);
        changed |= ctx.object_ref(ui, "Movement root", &mut self.player);
        check(ui, "Follow local VR player", &mut self.follow_local_player, &mut changed);
        widgets::info_note(ui, "F6 switches avatar in the starter projects. Both models share the same animation controller; movement stays on the player root.");
        changed
    }
}
impl Gizmo for zestengine_core::TemplateAvatar {}

impl Inspect for zestengine_core::AvatarHands {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        use zestengine_core::HandPose;
        let mut c = false;
        c |= ctx.object_ref(ui, "Character root", &mut self.player);
        check(ui, "Read hand input", &mut self.read_input, &mut c);
        let options: Vec<_> = HandPose::ALL.iter().map(|p| (*p, p.label())).collect();
        combo(ui, "hand-left-pose", "Left pose", &mut self.left, &options, &mut c);
        combo(ui, "hand-right-pose", "Right pose", &mut self.right, &options, &mut c);
        drag(ui, "Blend rate", &mut self.blend_rate, 0.1, 0.0..=100.0, &mut c);
        drag(ui, "Weight", &mut self.weight, 0.01, 0.0..=1.0, &mut c);
        note(ui, "Grip points; trigger closes the hand. Bind actions such as Left Hand Peace or Right Hand Rock and roll for the other poses. Disable Read hand input for manual poses.");
        for (side, label) in ["Left", "Right"].iter().enumerate() {
            text(ui, &format!("{label} action prefix"), &mut self.action_prefix[side], &mut c);
        }
        ui.collapsing("Pose settings", |ui| {
            for pose in HandPose::ALL {
                ui.collapsing(pose.label(), |ui| {
                    let p = &mut self.poses[pose as usize];
                    for (finger, label) in ["Thumb", "Index", "Middle", "Ring", "Little"].iter().enumerate() {
                        drag(ui, &format!("{label} curl"), &mut p.curl[finger], 0.01, 0.0..=1.0, &mut c);
                        drag(ui, &format!("{label} spread"), &mut p.spread[finger], 0.5, -45.0..=45.0, &mut c);
                    }
                });
            }
        });
        ui.collapsing("Finger rig", |ui| {
            check(ui, "Fit curl axes from hand geometry", &mut self.automatic_axes, &mut c);
            note(ui, "Defaults match both shipped avatars. Bone names and local rotation axes can be changed for another rig.");
            for axis in 0..3 {
                drag(ui, &format!("Curl axis {}", ["X", "Y", "Z"][axis]), &mut self.curl_axis[axis], 0.01, -1.0..=1.0, &mut c);
                drag(ui, &format!("Spread axis {}", ["X", "Y", "Z"][axis]), &mut self.spread_axis[axis], 0.01, -1.0..=1.0, &mut c);
                drag(ui, &format!("Finger joint {} bend", axis + 1), &mut self.curl_degrees[axis], 0.5, -180.0..=180.0, &mut c);
                drag(ui, &format!("Thumb joint {} bend", axis + 1), &mut self.thumb_curl_degrees[axis], 0.5, -180.0..=180.0, &mut c);
            }
            for side in 0..2 {
                ui.collapsing(["Left bones", "Right bones"][side], |ui| {
                    for finger in 0..5 { for joint in 0..3 {
                        text(ui, &format!("{} {}", ["Thumb", "Index", "Middle", "Ring", "Little"][finger], joint + 1), &mut self.bones[side][finger][joint], &mut c);
                    }}
                });
            }
        });
        if c { self.current = None; }
        c
    }
}
impl Gizmo for zestengine_core::AvatarHands {}

/// Header icon for kit components (None = not a kit component).
pub(crate) fn icon(name: &str) -> Option<(crate::icons::Icon, egui::Color32)> {
    use crate::icons::Icon;
    let pick = match name {
        "Health" | "Target Dummy" => (Icon::Target, theme::STATUS_ERROR),
        "Hitscan Weapon" => (Icon::Target, theme::TYPE_LIGHT),
        "First Person Character" | "Third Person Character" | "Follow Camera" | "RTS Camera" => {
            (Icon::Camera, theme::TYPE_CAMERA)
        }
        "Click To Move" | "RTS Selection" | "Selectable" => (Icon::ToolSelect, theme::TYPE_MESH),
        "Interactable" => (Icon::ToolSelect, theme::TYPE_LIGHT),
        "Trigger Events" | "Animation Event Bindings" => (Icon::Target, theme::TYPE_LIGHT),
        "Wasm Script" => (Icon::Gizmo, theme::TYPE_GI),
        "Resource Store" => (Icon::Stats, theme::TYPE_GROUP),
        "HUD Binding" | "Controls Menu" => (Icon::UiText, theme::TYPE_UI),
        "VR Character" | "VR Avatar" | "Grabbable" => (Icon::Gizmo, theme::TYPE_GI),
        _ => return None,
    };
    Some(pick)
}

// ------------------------------------------------------------------ rows

fn drag(ui: &mut Ui, label: &str, v: &mut f32, speed: f64, range: std::ops::RangeInclusive<f32>, changed: &mut bool) {
    widgets::property_row(ui, label, |ui| *changed |= ui.add(DragValue::new(v).speed(speed).range(range)).changed());
}

fn uint(ui: &mut Ui, label: &str, v: &mut u32, max: u32, changed: &mut bool) {
    widgets::property_row(ui, label, |ui| *changed |= ui.add(DragValue::new(v).range(0..=max)).changed());
}

fn check(ui: &mut Ui, label: &str, v: &mut bool, changed: &mut bool) {
    widgets::property_row(ui, label, |ui| *changed |= ui.checkbox(v, "").changed());
}

fn text(ui: &mut Ui, label: &str, v: &mut String, changed: &mut bool) {
    widgets::property_row(ui, label, |ui| *changed |= ui.text_edit_singleline(v).changed());
}

const COMMON_BONES: &[&str] = &[
    "spine_03", "pelvis", "chest", "neck_01", "Head",
    "clavicle_l", "clavicle_r", "upperarm_l", "upperarm_r",
    "lowerarm_l", "lowerarm_r", "hand_l", "hand_r",
    "thigh_l", "thigh_r", "calf_l", "calf_r", "foot_l", "foot_r",
];
const COMMON_PARAMS: &[&str] = &["Aiming", "Grounded", "Crouched", "Speed", "Forward", "Right", "Turn", "Vertical"];

/// A name field that offers the common values as a dropdown, falling back to a
/// text field for custom names (fewer bare text boxes on the kit components).
fn combo_text(ui: &mut Ui, id: &str, label: &str, value: &mut String, options: &[&str], changed: &mut bool) {
    let custom = !value.is_empty() && !options.contains(&value.as_str());
    widgets::property_row(ui, label, |ui| {
        egui::ComboBox::from_id_salt(("zestengine-combo-text", id))
            .selected_text(if value.is_empty() || custom { "Custom…" } else { value.as_str() })
            .width(ui.available_width())
            .show_ui(ui, |ui| {
                for o in options {
                    if ui.selectable_label(value.as_str() == *o, *o).clicked() {
                        *value = (*o).to_owned();
                        *changed = true;
                    }
                }
                ui.separator();
                if ui.selectable_label(value.is_empty() || custom, "Custom…").clicked() {
                    value.clear();
                    *changed = true;
                }
            });
        if value.is_empty() || custom {
            *changed |= ui.text_edit_singleline(value).changed();
        }
    });
}

fn vec3(ui: &mut Ui, label: &str, v: &mut [f32; 3], changed: &mut bool) {
    widgets::property_row(ui, label, |ui| *changed |= widgets::vec3_field(ui, v, 2, 0.02));
}

fn combo<T: Copy + PartialEq>(ui: &mut Ui, id: &str, label: &str, value: &mut T, options: &[(T, &str)], changed: &mut bool) {
    let current = options.iter().find(|(v, _)| v == value).map(|(_, l)| *l).unwrap_or("");
    widgets::property_row(ui, label, |ui| {
        egui::ComboBox::from_id_salt(id).selected_text(current).show_ui(ui, |ui| {
            for (v, l) in options {
                *changed |= ui.selectable_value(value, *v, *l).changed();
            }
        });
    });
}

fn note(ui: &mut Ui, s: &str) {
    ui.label(egui::RichText::new(s).size(theme::SIZE_META).color(theme::TEXT_DIM));
}

fn ring(center: Vec3, r: f32) -> Vec<(Vec3, Vec3)> {
    let n = 32;
    (0..n)
        .map(|k| {
            let a = k as f32 / n as f32 * std::f32::consts::TAU;
            let b = (k + 1) as f32 / n as f32 * std::f32::consts::TAU;
            (center + Vec3::new(a.cos(), 0.0, a.sin()) * r, center + Vec3::new(b.cos(), 0.0, b.sin()) * r)
        })
        .collect()
}

const GIZMO_GREEN: egui::Color32 = theme::STATUS_OK;
const GIZMO_BLUE: egui::Color32 = theme::STATUS_INFO;
const GIZMO_WARM: egui::Color32 = theme::TYPE_LIGHT;

// ------------------------------------------------------------------ combat

impl Inspect for Health {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut c = false;
        drag(ui, "Max", &mut self.max, 1.0, 1.0..=100_000.0, &mut c);
        drag(ui, "Current", &mut self.current, 1.0, 0.0..=self.max.max(1.0), &mut c);
        drag(ui, "Regen / sec", &mut self.regen_per_sec, 0.1, 0.0..=10_000.0, &mut c);
        drag(ui, "Regen delay", &mut self.regen_delay, 0.05, 0.0..=60.0, &mut c);
        check(ui, "Invulnerable", &mut self.invulnerable, &mut c);
        let opts: Vec<(DeathAction, &str)> = DeathAction::ALL.iter().map(|a| (*a, a.label())).collect();
        combo(ui, "zestengine-health-death", "On death", &mut self.on_death, &opts, &mut c);
        c
    }
}
impl Gizmo for Health {}

impl Inspect for HitscanWeapon {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut c = false;
        let opts: Vec<(WeaponTrigger, &str)> = WeaponTrigger::ALL.iter().map(|t| (*t, t.label())).collect();
        combo(ui, "zestengine-weapon-trigger", "Trigger", &mut self.trigger, &opts, &mut c);
        check(ui, "Automatic", &mut self.automatic, &mut c);
        drag(ui, "Fire rate", &mut self.fire_rate, 0.1, 0.1..=100.0, &mut c);
        drag(ui, "Damage", &mut self.damage, 0.5, 0.0..=100_000.0, &mut c);
        drag(ui, "Range", &mut self.range, 1.0, 0.1..=10_000.0, &mut c);
        drag(ui, "Impact force", &mut self.impact_force, 0.1, 0.0..=1000.0, &mut c);
        drag(ui, "Spread (deg)", &mut self.spread_deg, 0.05, 0.0..=45.0, &mut c);
        widgets::eyebrow(ui, "Ammo");
        uint(ui, "Magazine", &mut self.magazine_size, 10_000, &mut c);
        uint(ui, "Loaded", &mut self.ammo, self.magazine_size, &mut c);
        uint(ui, "Reserve", &mut self.reserve, 1_000_000, &mut c);
        check(ui, "Infinite reserve", &mut self.infinite_reserve, &mut c);
        drag(ui, "Reload time", &mut self.reload_time, 0.05, 0.0..=30.0, &mut c);
        widgets::eyebrow(ui, "References");
        c |= ctx.object_ref(ui, "Aim from", &mut self.aim_from);
        c |= ctx.object_ref(ui, "Muzzle", &mut self.muzzle);
        uint(ui, "Muzzle burst", &mut self.muzzle_burst, 1000, &mut c);
        c |= ctx.object_ref(ui, "Hit effect", &mut self.hit_effect);
        uint(ui, "Hit burst", &mut self.hit_burst, 1000, &mut c);
        c |= ctx.file_field(ui, "Fire sound", &mut self.fire_sound, "audio/shot.wav", &["wav", "ogg", "mp3", "flac"]);
        c |= ctx.file_field(ui, "Hit sound", &mut self.hit_sound, "audio/hit.wav", &["wav", "ogg", "mp3", "flac"]);
        c
    }
}
impl Gizmo for HitscanWeapon {
    fn gizmos(&self) -> Vec<GizmoSpec> {
        let len = self.range.min(8.0);
        vec![GizmoSpec::Lines { segments: vec![(Vec3::ZERO, Vec3::NEG_Z * len)], color: GIZMO_WARM, world: false }]
    }
}

impl Inspect for TargetDummy {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut c = false;
        drag(ui, "Respawn delay", &mut self.respawn_delay, 0.05, 0.0..=120.0, &mut c);
        drag(ui, "Fall speed", &mut self.fall_speed, 5.0, 1.0..=3600.0, &mut c);
        uint(ui, "Points", &mut self.points, 100_000, &mut c);
        note(ui, "Needs a Health on the same object.");
        c
    }
}
impl Gizmo for TargetDummy {}

// ------------------------------------------------------------------ controllers

impl Inspect for Swimmer {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut c = false;
        widgets::info_note(
            ui,
            "Tunes how the First / Third Person controllers swim: in water deeper than Swim depth they float with the head out, stroke where they look (dive by looking down; Jump rises, Crouch sinks), hold their breath under and climb out over an edge they swim into.",
        );
        drag(ui, "Swim depth (x height)", &mut self.swim_depth, 0.01, 0.2..=1.5, &mut c);
        drag(ui, "Swim speed", &mut self.swim_speed, 0.05, 0.0..=20.0, &mut c);
        drag(ui, "Sprint speed", &mut self.sprint_speed, 0.05, 0.0..=20.0, &mut c);
        drag(ui, "Stroke acceleration", &mut self.stroke_accel, 0.05, 0.1..=50.0, &mut c);
        drag(ui, "Water drag", &mut self.water_drag, 0.05, 0.0..=50.0, &mut c);
        drag(ui, "Float height (m)", &mut self.float_height, 0.01, -1.0..=1.0, &mut c);
        drag(ui, "Breath (s)", &mut self.breath, 0.5, 0.0..=600.0, &mut c);
        drag(ui, "Climb height (m)", &mut self.climb_height, 0.01, 0.0..=3.0, &mut c);
        drag(ui, "Climb time (s)", &mut self.climb_time, 0.01, 0.05..=5.0, &mut c);
        c
    }
}

impl Gizmo for Swimmer {}

fn character_braking_ui(ui: &mut Ui, braking: &mut zestengine_core::CharacterBraking) -> bool {
    let mut c = false;
    drag(ui, "Ground friction", &mut braking.ground_friction, 0.1, 0.0..=100.0, &mut c);
    drag(ui, "Walking braking (m/s²)", &mut braking.walking_deceleration, 0.1, 0.0..=200.0, &mut c);
    drag(ui, "Falling friction", &mut braking.falling_friction, 0.1, 0.0..=100.0, &mut c);
    drag(ui, "Falling braking (m/s²)", &mut braking.falling_deceleration, 0.1, 0.0..=200.0, &mut c);
    c
}

impl Inspect for FirstPersonCharacter {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut c = false;
        c |= ctx.object_ref(ui, "Camera", &mut self.camera);
        drag(ui, "Walk speed", &mut self.walk_speed, 0.05, 0.0..=100.0, &mut c);
        drag(ui, "Sprint speed", &mut self.sprint_speed, 0.05, 0.0..=100.0, &mut c);
        drag(ui, "Crouch speed", &mut self.crouch_speed, 0.05, 0.0..=100.0, &mut c);
        drag(ui, "Acceleration (m/s²)", &mut self.acceleration, 0.1, 0.0..=200.0, &mut c);
        c |= character_braking_ui(ui, &mut self.braking);
        drag(ui, "Air control", &mut self.air_control, 0.01, 0.0..=1.0, &mut c);
        drag(ui, "Jump height", &mut self.jump_height, 0.02, 0.0..=20.0, &mut c);
        drag(ui, "Gravity", &mut self.gravity, 0.1, 0.0..=200.0, &mut c);
        drag(ui, "Look sensitivity", &mut self.look_sensitivity, 0.005, 0.0..=5.0, &mut c);
        drag(ui, "Eye height", &mut self.eye_height, 0.01, 0.0..=10.0, &mut c);
        drag(ui, "Crouch eye height", &mut self.crouch_eye_height, 0.01, 0.0..=10.0, &mut c);
        drag(ui, "Crouch capsule height", &mut self.crouch_height, 0.01, 0.0..=10.0, &mut c);
        check(ui, "Crouch toggles", &mut self.crouch_toggle, &mut c);
        drag(ui, "Min pitch", &mut self.min_pitch, 0.5, -89.0..=0.0, &mut c);
        drag(ui, "Max pitch", &mut self.max_pitch, 0.5, 0.0..=89.0, &mut c);
        check(ui, "Lock cursor", &mut self.lock_cursor, &mut c);
        c
    }
}
impl Gizmo for FirstPersonCharacter {
    /// Eye and crouch-eye heights.
    fn gizmos(&self) -> Vec<GizmoSpec> {
        let mut s = vec![(Vec3::ZERO, Vec3::Y * self.eye_height)];
        s.extend(ring(Vec3::Y * self.eye_height, 0.15));
        s.extend(ring(Vec3::Y * self.crouch_eye_height, 0.1));
        vec![GizmoSpec::Lines { segments: s, color: GIZMO_BLUE, world: false }]
    }
}

impl Inspect for ThirdPersonCharacter {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut c = false;
        c |= ctx.object_ref(ui, "Camera", &mut self.camera);
        note(ui, "The camera must be a root object.");
        drag(ui, "Walk speed", &mut self.walk_speed, 0.05, 0.0..=100.0, &mut c);
        drag(ui, "Sprint speed", &mut self.sprint_speed, 0.05, 0.0..=100.0, &mut c);
        drag(ui, "Aim speed", &mut self.aim_speed, 0.05, 0.0..=100.0, &mut c);
        drag(ui, "Acceleration (m/s²)", &mut self.acceleration, 0.1, 0.0..=200.0, &mut c);
        c |= character_braking_ui(ui, &mut self.braking);
        drag(ui, "Air control", &mut self.air_control, 0.01, 0.0..=1.0, &mut c);
        drag(ui, "Turn speed", &mut self.turn_speed, 5.0, 0.0..=5000.0, &mut c);
        drag(ui, "Jump height", &mut self.jump_height, 0.02, 0.0..=20.0, &mut c);
        drag(ui, "Gravity", &mut self.gravity, 0.1, 0.0..=200.0, &mut c);
        drag(ui, "Look sensitivity", &mut self.look_sensitivity, 0.005, 0.0..=5.0, &mut c);
        drag(ui, "Min pitch", &mut self.min_pitch, 0.5, -89.0..=0.0, &mut c);
        drag(ui, "Max pitch", &mut self.max_pitch, 0.5, 0.0..=89.0, &mut c);
        widgets::eyebrow(ui, "Camera boom");
        drag(ui, "Pivot height", &mut self.pivot_height, 0.01, 0.0..=10.0, &mut c);
        drag(ui, "Distance", &mut self.distance, 0.05, 0.0..=50.0, &mut c);
        drag(ui, "Aim distance", &mut self.aim_distance, 0.05, 0.0..=50.0, &mut c);
        drag(ui, "Shoulder", &mut self.shoulder_offset, 0.01, -5.0..=5.0, &mut c);
        drag(ui, "Aim shoulder", &mut self.aim_shoulder_offset, 0.01, -5.0..=5.0, &mut c);
        drag(ui, "Camera radius", &mut self.camera_radius, 0.01, 0.0..=2.0, &mut c);
        drag(ui, "Zoom rate", &mut self.zoom_rate, 0.1, 0.1..=100.0, &mut c);
        check(ui, "Lock cursor", &mut self.lock_cursor, &mut c);
        c
    }
}
impl Gizmo for ThirdPersonCharacter {
    /// The boom from the pivot to the hip and aim camera positions.
    fn gizmos(&self) -> Vec<GizmoSpec> {
        let pivot = Vec3::Y * self.pivot_height;
        let hip = pivot + Vec3::new(self.shoulder_offset, 0.0, self.distance);
        let aim = pivot + Vec3::new(self.aim_shoulder_offset, 0.0, self.aim_distance);
        vec![
            GizmoSpec::Lines { segments: vec![(Vec3::ZERO, pivot), (pivot, hip)], color: GIZMO_BLUE, world: false },
            GizmoSpec::Lines { segments: vec![(pivot, aim)], color: GIZMO_WARM, world: false },
        ]
    }
}

impl Inspect for FollowCamera {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut c = false;
        c |= ctx.object_ref(ui, "Target", &mut self.target);
        vec3(ui, "Offset", &mut self.offset, &mut c);
        vec3(ui, "Look offset", &mut self.look_offset, &mut c);
        drag(ui, "Smoothing", &mut self.smoothing, 0.1, 0.0..=100.0, &mut c);
        drag(ui, "Zoom step", &mut self.zoom_step, 0.01, 0.0..=1.0, &mut c);
        drag(ui, "Min zoom", &mut self.min_zoom, 0.01, 0.05..=10.0, &mut c);
        drag(ui, "Max zoom", &mut self.max_zoom, 0.01, 0.05..=10.0, &mut c);
        c
    }
}
impl Gizmo for FollowCamera {}

impl Inspect for BoneAttach {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut c = false;
        c |= ctx.object_ref(ui, "Avatar", &mut self.avatar);
        combo_text(ui, "bone", "Bone", &mut self.bone, COMMON_BONES, &mut c);
        vec3(ui, "Position", &mut self.position, &mut c);
        vec3(ui, "Rotation", &mut self.rotation, &mut c);
        c
    }
}
impl Gizmo for BoneAttach {}

impl Inspect for WeaponHold {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut c = false;
        c |= ctx.object_ref(ui, "Avatar", &mut self.avatar);
        c |= ctx.object_ref(ui, "Aim camera", &mut self.camera);
        c |= ctx.object_ref(ui, "Right hand target", &mut self.right_hand);
        c |= ctx.object_ref(ui, "Left hand target", &mut self.left_hand);
        combo_text(ui, "anchor-bone", "Anchor bone", &mut self.anchor_bone, COMMON_BONES, &mut c);
        combo_text(ui, "aim-param", "Aim parameter", &mut self.aim_parameter, COMMON_PARAMS, &mut c);
        drag(ui, "Blend time", &mut self.blend_time, 0.01, 0.0..=2.0, &mut c);
        ui.add_space(6.0);
        if widgets::secondary_button(ui, None, "Pose editor").on_hover_text("Open the dedicated pose editor window").clicked() {
            ui.data_mut(|d| d.insert_temp(egui::Id::new(crate::components::OPEN_POSE_EDITOR_KEY), true));
        }
        widgets::property_row(ui, "Edit pose", |ui| {
            if widgets::segmented(ui, &mut self.editing, &[(WeaponPoseSlot::Idle, "Idle"), (WeaponPoseSlot::Aim, "Aim")]).changed() {
                // Selecting the pose being edited previews it on the avatar.
                self.request = Some(WeaponPoseRequest::Load);
                c = true;
            }
        });
        widgets::property_row(ui, "", |ui| {
            if widgets::secondary_button(ui, None, "Load pose").on_hover_text("Move the weapon to this pose").clicked() {
                self.request = Some(WeaponPoseRequest::Load);
                c = true;
            }
            if widgets::secondary_button(ui, None, "Store pose").on_hover_text("Save where the weapon is now as this pose").clicked() {
                self.request = Some(WeaponPoseRequest::Store);
                c = true;
            }
        });
        let slot = self.editing;
        let pose = self.pose_mut(slot);
        combo(
            ui,
            "weapon-hold-follow",
            "Follows",
            &mut pose.follow,
            &[(WeaponFollow::Bone, "Anchor bone (sways)"), (WeaponFollow::Avatar, "Avatar (no sway)"), (WeaponFollow::World, "World (fixed)")],
            &mut c,
        );
        vec3(ui, "Position", &mut pose.position, &mut c);
        vec3(ui, "Rotation", &mut pose.rotation, &mut c);
        note(ui, "Outside Play the weapon stays where you put it and the arms reach its hand targets. Move the weapon and the two hand targets with the gizmo, then Store pose. In Play the weapon follows the idle pose, and the aim pose while the aim parameter is on.");
        c
    }
}
impl Gizmo for WeaponHold {}

// ------------------------------------------------------------------ top-down

impl Inspect for ClickToMove {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut c = false;
        check(ui, "Hold to move", &mut self.hold_to_move, &mut c);
        c |= ctx.object_ref(ui, "Marker", &mut self.marker);
        drag(ui, "Max ray", &mut self.max_ray_distance, 1.0, 1.0..=10_000.0, &mut c);
        note(ui, "Needs a Nav Agent here and a NavMesh Surface in the scene.");
        c
    }
}
impl Gizmo for ClickToMove {}

impl Inspect for Interactable {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut c = false;
        text(ui, "Prompt", &mut self.prompt, &mut c);
        drag(ui, "Range", &mut self.range, 0.05, 0.1..=100.0, &mut c);
        widgets::property_row(ui, "Mode", |ui| {
            let options = InteractMode::ALL.map(|m| (m, m.label()));
            c |= widgets::segmented(ui, &mut self.mode, &options).changed();
        });
        c |= ctx.object_ref(ui, "Toggle", &mut self.toggle);
        check(ui, "Rebake navmesh", &mut self.rebake_navmesh, &mut c);
        c |= ctx.file_field(ui, "Sound", &mut self.sound, "audio/click.wav", &["wav", "ogg", "mp3", "flac"]);
        drag(ui, "Cooldown", &mut self.cooldown, 0.05, 0.0..=60.0, &mut c);
        check(ui, "Single use", &mut self.single_use, &mut c);
        let fields = INTERACT_FIELDS;
        c |= event_actions_ui(ui, ctx, "on-interact", "On interact", &mut self.on_interact, fields);
        match self.mode {
            InteractMode::Trigger => {}
            InteractMode::Toggle => {
                check(ui, "Is on", &mut self.on, &mut c);
                c |= event_actions_ui(ui, ctx, "on-on", "When turned on", &mut self.on_on, fields);
                c |= event_actions_ui(ui, ctx, "on-off", "When turned off", &mut self.on_off, fields);
            }
            InteractMode::Hold => {
                c |= event_actions_ui(ui, ctx, "on-press", "On press", &mut self.on_press, fields);
                c |= event_actions_ui(ui, ctx, "on-release", "On release", &mut self.on_release, fields);
                note(ui, "Release comes from code: ctx.modify::<Interactable>(id, |i| i.request_release(user)).");
            }
        }
        c
    }
}
impl Gizmo for Interactable {
    fn gizmos(&self) -> Vec<GizmoSpec> {
        vec![GizmoSpec::Range { center: Vec3::ZERO, radius: self.range, color: GIZMO_WARM }]
    }
    fn apply_gizmo_edit(&mut self, edit: GizmoEdit) {
        if let GizmoEdit::Range { radius, .. } = edit {
            self.range = radius.max(0.1);
        }
    }
}

/// Event fields an Interactable offers to "Event" sources.
const INTERACT_FIELDS: &[(&str, ActionType)] = &[
    ("instigator", ActionType::Object),
    ("object", ActionType::Object),
    ("uses", ActionType::Int),
    ("on", ActionType::Bool),
];

impl Inspect for TriggerEvents {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut c = false;
        text(ui, "Require component", &mut self.require_component, &mut c);
        check(ui, "First in / last out", &mut self.first_and_last_only, &mut c);
        check(ui, "Once", &mut self.once, &mut c);
        let fields: &[(&str, ActionType)] = &[("other", ActionType::Object), ("object", ActionType::Object)];
        c |= event_actions_ui(ui, ctx, "on-enter", "On enter", &mut self.on_enter, fields);
        c |= event_actions_ui(ui, ctx, "on-exit", "On exit", &mut self.on_exit, fields);
        note(ui, "Needs a collider with Is Trigger on this object. The object that entered is the instigator.");
        c
    }
}
impl Gizmo for TriggerEvents {}

impl Inspect for AnimationEventBindings {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut c = false;
        let fields: &[(&str, ActionType)] = &[("name", ActionType::String), ("object", ActionType::Object)];
        let mut remove = None;
        for (k, e) in self.events.iter_mut().enumerate() {
            ui.push_id(("anim-event", k), |ui| {
                ui.horizontal(|ui| {
                    c |= ui.add(egui::TextEdit::singleline(&mut e.event).hint_text("event name")).changed();
                    if widgets::icon_button(ui, crate::icons::Icon::Close, "Remove event", false).clicked() {
                        remove = Some(k);
                    }
                });
                c |= event_actions_ui(ui, ctx, "actions", "Runs", &mut e.actions, fields);
            });
            widgets::divider(ui);
        }
        if let Some(k) = remove {
            self.events.remove(k);
            c = true;
        }
        if widgets::dashed_button(ui, Some(crate::icons::Icon::Plus), "Add event").clicked() {
            self.events.push(Default::default());
            c = true;
        }
        note(ui, "Runs when this object's Animator passes a named event marker.");
        c
    }
}
impl Gizmo for AnimationEventBindings {}

// ------------------------------------------------------------------ strategy

impl Inspect for RtsCamera {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut c = false;
        drag(ui, "Yaw", &mut self.yaw, 0.5, -360.0..=360.0, &mut c);
        drag(ui, "Pitch", &mut self.pitch, 0.5, 5.0..=89.0, &mut c);
        drag(ui, "Distance", &mut self.distance, 0.1, 0.5..=1000.0, &mut c);
        drag(ui, "Min distance", &mut self.min_distance, 0.1, 0.5..=1000.0, &mut c);
        drag(ui, "Max distance", &mut self.max_distance, 0.1, 0.5..=1000.0, &mut c);
        drag(ui, "Pan speed", &mut self.pan_speed, 0.1, 0.0..=500.0, &mut c);
        check(ui, "Edge pan", &mut self.edge_pan, &mut c);
        drag(ui, "Edge margin (px)", &mut self.edge_margin, 0.5, 0.0..=200.0, &mut c);
        drag(ui, "Zoom step", &mut self.zoom_step, 0.05, 0.0..=100.0, &mut c);
        drag(ui, "Rotate speed", &mut self.rotate_speed, 1.0, 0.0..=1000.0, &mut c);
        drag(ui, "Smoothing", &mut self.smoothing, 0.1, 0.0..=100.0, &mut c);
        widgets::property_row(ui, "Bounds min", |ui| {
            c |= ui.add(DragValue::new(&mut self.bounds_min[0]).speed(0.5).prefix("x ")).changed();
            c |= ui.add(DragValue::new(&mut self.bounds_min[1]).speed(0.5).prefix("z ")).changed();
        });
        widgets::property_row(ui, "Bounds max", |ui| {
            c |= ui.add(DragValue::new(&mut self.bounds_max[0]).speed(0.5).prefix("x ")).changed();
            c |= ui.add(DragValue::new(&mut self.bounds_max[1]).speed(0.5).prefix("z ")).changed();
        });
        c
    }
}
impl Gizmo for RtsCamera {
    /// The pan bounds on the ground (world space).
    fn gizmos(&self) -> Vec<GizmoSpec> {
        let (a, b) = (self.bounds_min, self.bounds_max);
        let p = |x: f32, z: f32| Vec3::new(x, 0.05, z);
        let segments = vec![
            (p(a[0], a[1]), p(b[0], a[1])),
            (p(b[0], a[1]), p(b[0], b[1])),
            (p(b[0], b[1]), p(a[0], b[1])),
            (p(a[0], b[1]), p(a[0], a[1])),
        ];
        vec![GizmoSpec::Lines { segments, color: GIZMO_BLUE, world: true }]
    }
}

impl Inspect for RtsSelection {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut c = false;
        c |= ctx.object_ref(ui, "Box visual", &mut self.box_visual);
        drag(ui, "UI ref. height", &mut self.ui_reference_height, 1.0, 1.0..=8192.0, &mut c);
        drag(ui, "Drag threshold", &mut self.drag_threshold, 0.5, 0.0..=100.0, &mut c);
        drag(ui, "Formation spacing", &mut self.formation_spacing, 0.05, 0.1..=50.0, &mut c);
        c |= ctx.object_ref(ui, "Move marker", &mut self.move_marker);
        uint(ui, "Team", &mut self.team, 1000, &mut c);
        c
    }
}
impl Gizmo for RtsSelection {}

impl Inspect for Selectable {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut c = false;
        c |= ctx.object_ref(ui, "Ring", &mut self.ring);
        uint(ui, "Team", &mut self.team, 1000, &mut c);
        c
    }
}
impl Gizmo for Selectable {
    fn gizmos(&self) -> Vec<GizmoSpec> {
        vec![GizmoSpec::Lines { segments: ring(Vec3::Y * 0.05, 0.6), color: GIZMO_GREEN, world: false }]
    }
}

impl Inspect for ResourceStore {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut c = false;
        let mut remove = None;
        for (i, r) in self.resources.iter_mut().enumerate() {
            ui.push_id(i, |ui| {
                ui.horizontal(|ui| {
                    widgets::eyebrow(ui, if r.name.is_empty() { "Resource" } else { &r.name });
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if widgets::icon_button(ui, crate::icons::Icon::Trash, "Remove resource", false).clicked() {
                            remove = Some(i);
                        }
                    });
                });
                text(ui, "Name", &mut r.name, &mut c);
                drag(ui, "Amount", &mut r.amount, 1.0, -1e9..=1e9, &mut c);
                drag(ui, "Income / sec", &mut r.income_per_sec, 0.1, -1e6..=1e6, &mut c);
                drag(ui, "Capacity", &mut r.capacity, 1.0, 0.0..=1e9, &mut c);
            });
        }
        if let Some(i) = remove {
            self.resources.remove(i);
            c = true;
        }
        if widgets::dashed_button(ui, Some(crate::icons::Icon::Plus), "Add resource").clicked() {
            self.resources.push(ResourceAmount::default());
            c = true;
        }
        c
    }
}
impl Gizmo for ResourceStore {}

impl Inspect for HudBinding {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut c = false;
        c |= ctx.object_ref(ui, "Source", &mut self.source);
        widgets::property_row(ui, "Value", |ui| {
            egui::ComboBox::from_id_salt("zestengine-hud-value").selected_text(self.value.label()).show_ui(ui, |ui| {
                for kind in HudValue::kinds() {
                    let same = std::mem::discriminant(&kind) == std::mem::discriminant(&self.value);
                    if ui.selectable_label(same, kind.label()).clicked() && !same {
                        self.value = kind;
                        c = true;
                    }
                }
            });
        });
        if let HudValue::Resource(name) = &mut self.value {
            text(ui, "Resource", name, &mut c);
        }
        text(ui, "Format", &mut self.format, &mut c);
        if self.value == HudValue::Ammo {
            text(ui, "Reloading text", &mut self.reloading_text, &mut c);
        }
        note(ui, "Writes into this object's UI Text ({value}, {max}) and UI Progress Bar.");
        c
    }
}
impl Gizmo for HudBinding {}

// ------------------------------------------------------------------ controls menu

impl Inspect for ControlsMenu {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut c = false;
        text(ui, "Name prefix", &mut self.prefix, &mut c);
        uint(ui, "Rows", &mut self.rows, 32, &mut c);
        uint(ui, "Bindings per row", &mut self.columns, 2, &mut c);
        let opts: Vec<(ConflictPolicy, &str)> = ConflictPolicy::ALL.iter().map(|p| (*p, p.label())).collect();
        combo(ui, "zestengine-controls-conflicts", "On conflict", &mut self.conflicts, &opts, &mut c);
        widgets::eyebrow(ui, "Hidden actions");
        let mut remove = None;
        for (i, name) in self.hidden_actions.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                c |= ui.text_edit_singleline(name).changed();
                if ui.small_button(egui_phosphor::regular::X).on_hover_text("Show this action again").clicked() {
                    remove = Some(i);
                }
            });
        }
        if let Some(i) = remove {
            self.hidden_actions.remove(i);
            c = true;
        }
        if ui.small_button("Hide an action").clicked() {
            self.hidden_actions.push("Look".into());
            c = true;
        }
        note(
            ui,
            "Fills the rows named \"<prefix> Row N ...\" (see the Controls UI template) with the shown scheme's actions. Players click a binding, then press the new input; remaps save to their input.ron.",
        );
        c
    }
}
impl Gizmo for ControlsMenu {}

// ------------------------------------------------------------------ VR

impl Inspect for VrCharacter {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut c = false;
        c |= ctx.object_ref_with_hint(ui, "Head", &mut self.head, "(drag camera here)");
        c |= ctx.object_ref(ui, "Left hand", &mut self.left_hand);
        c |= ctx.object_ref(ui, "Right hand", &mut self.right_hand);
        note(ui, "Head is the camera shared by VR and desktop. Hands are scene anchors. Parent all three directly under this rig; hand visuals can be children of the anchors.");
        let mut smooth = matches!(self.locomotion, VrLocomotion::Smooth | VrLocomotion::Both);
        let mut teleport = matches!(self.locomotion, VrLocomotion::Teleport | VrLocomotion::Both);
        if ui.checkbox(&mut smooth, "Smooth locomotion").changed() | ui.checkbox(&mut teleport, "Teleport locomotion").changed() {
            self.locomotion = match (smooth, teleport) {
                (true, true) => VrLocomotion::Both, (true, false) => VrLocomotion::Smooth,
                (false, true) => VrLocomotion::Teleport, (false, false) => VrLocomotion::Disabled,
            };
            c = true;
        }
        let opts: Vec<_> = zestengine_core::VrTurning::ALL.iter().map(|m| (*m, m.label())).collect();
        combo(ui, "zestengine-vr-turning", "Turning", &mut self.turning, &opts, &mut c);
        drag(ui, "Move speed", &mut self.move_speed, 0.05, 0.0..=50.0, &mut c);
        drag(ui, "Acceleration (m/s²)", &mut self.acceleration, 0.1, 0.0..=200.0, &mut c);
        drag(ui, "Air control", &mut self.air_control, 0.01, 0.0..=1.0, &mut c);
        drag(ui, "Gravity", &mut self.gravity, 0.1, 0.0..=200.0, &mut c);
        c |= character_braking_ui(ui, &mut self.braking);
        check(ui, "Track capsule to head", &mut self.track_capsule, &mut c);
        note(ui, "Add Character on this rig for collision, steps and slopes. Tracking sets capsule height and horizontal centre; head and hands keep their tracked poses.");
        drag(ui, "Snap turn (deg)", &mut self.snap_turn, 1.0, 0.0..=180.0, &mut c);
        drag(ui, "Smooth turn speed", &mut self.smooth_turn_speed, 1.0, 0.0..=720.0, &mut c);
        c |= ctx.object_ref(ui, "Teleport marker", &mut self.teleport_marker);
        drag(ui, "Teleport speed", &mut self.teleport_speed, 0.1, 1.0..=50.0, &mut c);
        drag(ui, "Grab radius", &mut self.grab_radius, 0.01, 0.0..=2.0, &mut c);
        text(ui, "Spawn tag", &mut self.spawn_tag, &mut c);
        widgets::eyebrow(ui, "Desktop fallback");
        check(ui, "Enabled", &mut self.desktop_fallback, &mut c);
        drag(ui, "Eye height", &mut self.eye_height, 0.01, 0.0..=5.0, &mut c);
        drag(ui, "Look sensitivity", &mut self.look_sensitivity, 0.005, 0.0..=5.0, &mut c);
        widgets::eyebrow(ui, "Body");
        let opts: Vec<(zestengine_core::AvatarScaling, &str)> = zestengine_core::AvatarScaling::ALL.iter().map(|m| (*m, m.label())).collect();
        combo(ui, "zestengine-vr-avatar-scaling", "Avatar scaling", &mut self.avatar_scaling, &opts, &mut c);
        check(ui, "Match arm span", &mut self.match_arm_span, &mut c);
        note(ui, match self.avatar_scaling {
            zestengine_core::AvatarScaling::ScaleAvatar => "The avatar is scaled so its eyes sit at the player's measured eye height.",
            zestengine_core::AvatarScaling::ScalePlayer => "The avatar keeps its size; the play space scales so the player sees from the avatar's eyes.",
            zestengine_core::AvatarScaling::Off => "Avatars keep their size and the play space is not scaled.",
        });
        note(ui, "The body menu opens Calibrate / Height / Done in the headset.");
        c
    }
}
impl Gizmo for VrCharacter {
    /// A 2 x 2 m play-space square on the floor plus the desktop eye height.
    fn gizmos(&self) -> Vec<GizmoSpec> {
        let p = |x: f32, z: f32| Vec3::new(x, 0.02, z);
        let mut s = vec![(p(-1.0, -1.0), p(1.0, -1.0)), (p(1.0, -1.0), p(1.0, 1.0)), (p(1.0, 1.0), p(-1.0, 1.0)), (p(-1.0, 1.0), p(-1.0, -1.0))];
        s.push((Vec3::ZERO, Vec3::Y * self.eye_height));
        s.push((Vec3::Y * self.eye_height, Vec3::new(0.0, self.eye_height, -0.4)));
        vec![GizmoSpec::Lines { segments: s, color: GIZMO_BLUE, world: false }]
    }
}

impl Inspect for Grabbable {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut c = false;
        check(ui, "Snap to hand", &mut self.snap_to_hand, &mut c);
        if self.snap_to_hand {
            vec3(ui, "Hold position", &mut self.hold_position, &mut c);
            vec3(ui, "Hold rotation", &mut self.hold_rotation, &mut c);
        }
        drag(ui, "Radius", &mut self.radius, 0.01, 0.0..=5.0, &mut c);
        check(ui, "Throwable", &mut self.throwable, &mut c);
        drag(ui, "Return after", &mut self.return_after, 0.1, 0.0..=600.0, &mut c);
        note(ui, "Must be a root object. Throwing needs a dynamic Rigid Body.");
        c
    }
}
impl Gizmo for Grabbable {
    fn gizmos(&self) -> Vec<GizmoSpec> {
        vec![GizmoSpec::Range { center: Vec3::ZERO, radius: self.radius, color: GIZMO_GREEN }]
    }
    fn apply_gizmo_edit(&mut self, edit: GizmoEdit) {
        if let GizmoEdit::Range { radius, .. } = edit {
            self.radius = radius.max(0.0);
        }
    }
}

impl Inspect for VrAvatar {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut c = false;
        c |= ctx.object_ref(ui, "Head", &mut self.head);
        drag(ui, "Neck drop", &mut self.neck_drop, 0.01, 0.0..=1.0, &mut c);
        drag(ui, "Turn rate", &mut self.turn_rate, 0.1, 0.0..=50.0, &mut c);
        c
    }
}
impl Gizmo for VrAvatar {}
