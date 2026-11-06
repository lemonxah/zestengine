//! Pieces of the rig Configure window that live outside `rig_editor.rs`:
//! dropping a Hierarchy object on a bone row, the embedded Transform editor
//! for the selected joint (write-back to the reference pose), and the Biped
//! Pose command. The data handling is pure and tested; the egui drawing is not
//! verified visually.

use egui::{DragValue, Ui};
use glam::{EulerRot, Quat, Vec3};
use zestengine_assets::human_rig::{HumanBone, PoseBone, RigImport};
use zestengine_assets::Skeleton;
use crate::{theme, widgets};

/// The joint of `skel` a dragged object name stands for: an exact match, else
/// one ignoring case, spaces, `_`, `-` and `.`. A name that fits two joints
/// equally is no match (the user picks from the menu then).
pub(crate) fn joint_for_name(skel: &Skeleton, name: &str) -> Option<String> {
    let key = |s: &str| -> String { s.chars().filter(|c| !matches!(c, ' ' | '_' | '-' | '.')).collect::<String>().to_lowercase() };
    if let Some(j) = skel.joints.iter().find(|j| j.name == name) {
        return Some(j.name.clone());
    }
    let want = key(name);
    if want.is_empty() {
        return None;
    }
    let mut hits = skel.joints.iter().filter(|j| key(&j.name) == want);
    let first = hits.next()?;
    hits.next().is_none().then(|| first.name.clone())
}

/// A joint dropped on `resp` this frame: the pointer was released over it
/// while the Hierarchy was dragging an object whose name matches a joint.
pub(crate) fn dropped_joint(ui: &Ui, skel: &Skeleton, resp: &egui::Response) -> Option<String> {
    if !(resp.hovered() && ui.input(|i| i.pointer.any_released())) {
        return None;
    }
    let name = ui.data(|d| d.get_temp::<String>(egui::Id::new(crate::DRAG_OBJECT_NAME_KEY)))?;
    joint_for_name(skel, &name)
}

/// The reference-pose entry of joint `j`: the stored one, else the file pose.
pub(crate) fn joint_pose(rig: &RigImport, skel: &Skeleton, j: usize) -> PoseBone {
    let name = &skel.joints[j].name;
    rig.reference_pose.iter().find(|p| &p.name == name).cloned().unwrap_or_else(|| {
        let joint = &skel.joints[j];
        PoseBone {
            name: name.clone(),
            translation: joint.rest_translation.to_array(),
            rotation: joint.rest_rotation.to_array(),
            scale: joint.rest_scale.to_array(),
        }
    })
}

/// Store `pose` in the rig's reference pose. A rig without one first gets the
/// whole file pose, so the other joints keep their transforms.
pub(crate) fn set_joint_pose(rig: &mut RigImport, skel: &Skeleton, pose: PoseBone) {
    if rig.reference_pose.is_empty() {
        rig.reference_pose = skel
            .joints
            .iter()
            .map(|j| PoseBone { name: j.name.clone(), translation: j.rest_translation.to_array(), rotation: j.rest_rotation.to_array(), scale: j.rest_scale.to_array() })
            .collect();
    }
    match rig.reference_pose.iter_mut().find(|p| p.name == pose.name) {
        Some(p) => *p = pose,
        None => rig.reference_pose.push(pose),
    }
    rig.auto_generated = false;
}

/// Pose menu: Biped Pose. False when Hips is not mapped.
pub(crate) fn apply_biped_pose(rig: &mut RigImport, skel: &Skeleton) -> bool {
    let pose = zestengine_assets::rig_biped::biped_pose(skel, &rig.human);
    if pose.is_empty() {
        return false;
    }
    rig.reference_pose = pose;
    rig.auto_generated = false;
    true
}

/// The Transform editor of the selected bone's joint: position, rotation (XYZ
/// Euler degrees) and scale, written back to the reference pose. Returns
/// whether the rig changed.
pub(crate) fn transform_section(ui: &mut Ui, skel: &Skeleton, rig: &mut RigImport, selected: Option<HumanBone>) -> bool {
    let Some(bone) = selected else { return false };
    let Some(joint) = rig.joint_name(bone).and_then(|n| skel.joints.iter().position(|j| j.name == n)) else {
        return false;
    };
    let mut pose = joint_pose(rig, skel, joint);
    let mut t = Vec3::from(pose.translation);
    let (ex, ey, ez) = Quat::from_array(pose.rotation).normalize().to_euler(EulerRot::XYZ);
    let mut euler = Vec3::new(ex.to_degrees(), ey.to_degrees(), ez.to_degrees());
    let mut s = Vec3::from(pose.scale);
    let mut changed = false;
    egui::CollapsingHeader::new(format!("Transform: {} ({})", bone.name(), skel.joints[joint].name)).id_salt("zestengine-rig-transform").default_open(true).show(ui, |ui| {
        ui.label(egui::RichText::new("Local transform in the reference pose. Edits change this pose only; Apply saves it to the model.").size(theme::SIZE_META).color(theme::TEXT_DIM));
        let mut row = |ui: &mut Ui, label: &str, v: &mut Vec3, speed: f32| {
            widgets::property_row(ui, label, |ui| {
                ui.horizontal(|ui| {
                    for c in [&mut v.x, &mut v.y, &mut v.z] {
                        changed |= ui.add(DragValue::new(c).speed(speed)).changed();
                    }
                });
            });
        };
        row(ui, "Position", &mut t, 0.001);
        row(ui, "Rotation", &mut euler, 0.5);
        row(ui, "Scale", &mut s, 0.001);
    });
    if changed {
        pose.translation = t.to_array();
        pose.rotation = Quat::from_euler(EulerRot::XYZ, euler.x.to_radians(), euler.y.to_radians(), euler.z.to_radians()).normalize().to_array();
        pose.scale = s.to_array();
        set_joint_pose(rig, skel, pose);
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Mat4;
    use zestengine_assets::Joint;

    fn skel(names: &[&str]) -> Skeleton {
        Skeleton {
            joints: names
                .iter()
                .enumerate()
                .map(|(i, n)| Joint {
                    name: (*n).into(),
                    parent: i.checked_sub(1),
                    inverse_bind: Mat4::IDENTITY,
                    rest_translation: Vec3::new(0.0, i as f32, 0.0),
                    rest_rotation: Quat::IDENTITY,
                    rest_scale: Vec3::ONE,
                })
                .collect(),
            rig: None,
            ..Default::default()
        }
    }

    #[test]
    fn dropped_names_match_joints_loosely_but_not_ambiguously() {
        let s = skel(&["Hips", "Left_Arm", "LeftArm2", "mixamorig:Head"]);
        assert_eq!(joint_for_name(&s, "Hips").as_deref(), Some("Hips"));
        assert_eq!(joint_for_name(&s, "left arm").as_deref(), Some("Left_Arm"));
        assert_eq!(joint_for_name(&s, "HIPS").as_deref(), Some("Hips"));
        assert_eq!(joint_for_name(&s, "Nope"), None);
        assert_eq!(joint_for_name(&s, ""), None);
        let dup = skel(&["Arm.L", "Arm_L"]);
        assert_eq!(joint_for_name(&dup, "arm l"), None, "two equal matches: the user chooses");
        assert_eq!(joint_for_name(&dup, "Arm_L").as_deref(), Some("Arm_L"), "an exact name always wins");
    }

    #[test]
    fn editing_a_joint_writes_the_whole_pose_once_then_updates_it() {
        let s = skel(&["Hips", "Spine", "Head"]);
        let mut rig = RigImport::default();
        assert!(rig.reference_pose.is_empty());
        let mut p = joint_pose(&rig, &s, 1);
        assert_eq!(Vec3::from(p.translation), Vec3::new(0.0, 1.0, 0.0), "falls back to the file pose");
        p.rotation = Quat::from_rotation_z(0.5).to_array();
        set_joint_pose(&mut rig, &s, p.clone());
        assert_eq!(rig.reference_pose.len(), 3, "the other joints keep their file pose");
        assert_eq!(rig.reference_pose[1].rotation, p.rotation);
        assert_eq!(rig.reference_pose[2].translation, [0.0, 2.0, 0.0]);
        let mut again = p;
        again.scale = [2.0; 3];
        set_joint_pose(&mut rig, &s, again);
        assert_eq!(rig.reference_pose.len(), 3);
        assert_eq!(joint_pose(&rig, &s, 1).scale, [2.0; 3]);
        assert!(!rig.auto_generated);
    }

    #[test]
    fn biped_pose_needs_hips_and_replaces_the_reference_pose() {
        let s = skel(&["Hips", "Spine"]);
        let mut rig = RigImport::default();
        assert!(!apply_biped_pose(&mut rig, &s));
        rig.human = vec![(HumanBone::Hips, "Hips".into()), (HumanBone::Spine, "Spine".into())];
        assert!(apply_biped_pose(&mut rig, &s));
        assert_eq!(rig.reference_pose.len(), 2);
    }
}
