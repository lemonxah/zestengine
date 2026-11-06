//! The Muscles & Settings tab of the avatar Configure window.
//!
//! Sliders for the 95 muscles (a master slider per muscle family and one slider
//! per muscle, grouped Body, Head, arms, fingers and legs) pose the avatar's
//! skeleton live in the viewport overlay; each muscle row expands to its limits
//! in degrees (stored in [`RigImport::muscles`], with a Reset). The pose is
//! computed by [`posed_locals`], a plain function over the muscle model.

use std::sync::{Arc, OnceLock};

use egui::RichText;
use glam::Mat4;
use zestengine_assets::human_rig::RigImport;
use zestengine_assets::muscle::{MUSCLE_COUNT, MuscleGroup, MusclePose, MuscleRig, muscles};
use zestengine_assets::ref_pose::reference_locals;
use zestengine_assets::Skeleton;
use crate::icons::Icon;
use crate::{theme, widgets};

use crate::rig_editor::RigOut;

/// Preview sliders snap to 0 inside this band, as the master and muscle sliders do.
pub const DEAD_ZONE: f32 = 0.1;

/// Height of the slider list in the Muscles & Settings tab (it scrolls inside).
pub const SCROLL_HEIGHT: f32 = 400.0;

/// A slider value with the dead zone applied.
pub fn snap_dead_zone(v: f32) -> f32 {
    if v.abs() < DEAD_ZONE { 0.0 } else { v.clamp(-1.0, 1.0) }
}

/// A master slider: one value written to a fixed set of muscles.
pub struct Master {
    pub label: &'static str,
    pub members: Vec<usize>,
}

/// The seven master sliders: Open Close, Left Right, Roll Left Right, In Out,
/// Roll In Out, Finger Open Close, Finger In Out. Each drives the muscles whose
/// names carry its keyword.
pub fn masters() -> &'static [Master] {
    static M: OnceLock<Vec<Master>> = OnceLock::new();
    M.get_or_init(|| {
        let pick = |label: &'static str, f: &dyn Fn(&str) -> bool| Master {
            label,
            members: muscles().iter().enumerate().filter(|(_, m)| f(&m.name)).map(|(i, _)| i).collect(),
        };
        let fingers = |n: &str| n.contains("Stretched") || n.contains("Spread");
        vec![
            pick("Open Close", &|n| {
                !fingers(n)
                    && !n.contains("Eye")
                    && !n.contains("Twist")
                    && (n.contains("Front-Back") || n.contains("Stretch") || n.contains("Nod") || n.contains("Up-Down") || n.contains("Close") || n.contains("Hand Down-Up"))
            }),
            pick("Left Right", &|n| n.contains("Left-Right") && !n.contains("Twist") && !n.contains("Turn")),
            pick("Roll Left Right", &|n| n.contains("Twist Left-Right") || n.contains("Turn Left-Right")),
            pick("In Out", &|n| n.contains("In-Out") && !n.contains("Twist") && !n.contains("Eye")),
            pick("Roll In Out", &|n| n.contains("Twist In-Out")),
            pick("Finger Open Close", &|n| n.contains("Stretched")),
            pick("Finger In Out", &|n| n.contains("Spread")),
        ]
    })
}

/// Local transforms of `skel` with the human bones posed by the muscle `values`
/// (the root stays at the reference pose); other joints keep their reference
/// transform.
pub fn posed_locals(skel: &Skeleton, mr: &MuscleRig, values: &[f32; MUSCLE_COUNT]) -> Vec<Mat4> {
    let mut locals = reference_locals(skel);
    let (mut rot, mut trans) = mr.reference_locals();
    let pose = MusclePose { muscles: *values, root: mr.reference_root(), ..MusclePose::default() };
    mr.pose_from_muscles(&pose, &mut rot, &mut trans);
    for (_, j) in mr.driven_joints(zestengine_assets::muscle::ALL_DRIVEN) {
        let (s, _, _) = locals[j].to_scale_rotation_translation();
        locals[j] = Mat4::from_scale_rotation_translation(s, rot[j], locals[j].to_scale_rotation_translation().2);
    }
    if let Some(h) = mr.joint(zestengine_assets::human_rig::HumanBone::Hips) {
        let (s, _, _) = locals[h].to_scale_rotation_translation();
        locals[h] = Mat4::from_scale_rotation_translation(s, rot[h], trans[h]);
    }
    locals
}

struct Cache {
    rig: RigImport,
    joints: usize,
    last: String,
    mr: Option<Arc<MuscleRig>>,
}

/// State of the Muscles & Settings tab.
#[derive(Default)]
pub struct MuscleUi {
    /// The tab is shown (else the Mapping tab).
    pub active: bool,
    values: Option<Box<[f32; MUSCLE_COUNT]>>,
    masters: [f32; 7],
    /// The muscle whose limits are expanded.
    focus: Option<usize>,
    cache: Option<Cache>,
}

impl MuscleUi {
    /// The muscle model of `skel` under `rig` (cached while neither changes).
    fn model(&mut self, skel: &Skeleton, rig: &RigImport) -> Option<Arc<MuscleRig>> {
        let last = skel.joints.last().map(|j| j.name.clone()).unwrap_or_default();
        let fresh = self.cache.as_ref().is_some_and(|c| c.rig == *rig && c.joints == skel.joints.len() && c.last == last);
        if !fresh {
            self.cache = Some(Cache { rig: rig.clone(), joints: skel.joints.len(), last, mr: MuscleRig::new(skel, rig).map(Arc::new) });
        }
        self.cache.as_ref().and_then(|c| c.mr.clone())
    }

    /// The pose of the tab's sliders for the viewport overlay: `None` unless the tab is open.
    pub fn preview_locals(&mut self, skel: &Skeleton, rig: &RigImport) -> Option<Vec<Mat4>> {
        if !self.active {
            return None;
        }
        let mr = self.model(skel, rig)?;
        let values: [f32; MUSCLE_COUNT] = **self.values.get_or_insert_with(|| Box::new(mr.rest_pose().muscles));
        Some(posed_locals(skel, &mr, &values))
    }

    /// Slider values back to the reference pose (or all zero: the relaxed neutral pose).
    fn reset_values(&mut self, mr: &MuscleRig, neutral: bool) {
        self.values = Some(Box::new(if neutral { [0.0; MUSCLE_COUNT] } else { mr.rest_pose().muscles }));
        self.masters = [0.0; 7];
    }

    pub fn tab(&mut self, ui: &mut egui::Ui, skel: &Skeleton, rig: &mut RigImport, out: &mut RigOut) {
        let Some(mr) = self.model(skel, rig) else {
            widgets::info_note(ui, "The muscle model needs the Hips mapped to a joint of this model. Map the required bones on the Mapping tab first.");
            return;
        };
        let mut vals: [f32; MUSCLE_COUNT] = **self.values.get_or_insert_with(|| Box::new(mr.rest_pose().muscles));
        let mut reset: Option<bool> = None;
        ui.horizontal(|ui| {
            if widgets::secondary_button(ui, Some(Icon::Reset), "Reset All").on_hover_text("Back to the reference pose").clicked() {
                reset = Some(false);
            }
            if widgets::secondary_button(ui, None, "Neutral").on_hover_text("Every muscle at 0: the relaxed pose with bent elbows and knees").clicked() {
                reset = Some(true);
            }
            let menu = widgets::secondary_button(ui, None, "Muscles");
            egui::Popup::menu(&menu).show(|ui| {
                widgets::menu_frame(ui);
                let modified = !rig.muscles.limits.is_empty();
                if ui.add_enabled(modified, egui::Button::new("Reset limits to defaults")).clicked() {
                    rig.muscles.reset_all();
                    rig.auto_generated = false;
                    out.changed = true;
                    ui.close();
                }
            });
        });
        ui.add_space(4.0);
        // A fixed height, never `available_height()`: the Configure window sizes itself to its content
        // and has a footer below this area, so an available-height scroll area grew the window by the
        // footer's height every frame.
        egui::ScrollArea::vertical().auto_shrink([false, false]).min_scrolled_height(SCROLL_HEIGHT).max_height(SCROLL_HEIGHT).show(ui, |ui| {
            widgets::eyebrow(ui, "Muscle Group Preview");
            for (k, m) in masters().iter().enumerate() {
                let mut v = self.masters[k];
                let before = v;
                widgets::property_row(ui, m.label, |ui| widgets::slider(ui, egui::Slider::new(&mut v, -1.0..=1.0).show_value(true)));
                if v != before {
                    v = snap_dead_zone(v);
                    self.masters[k] = v;
                    for &i in &m.members {
                        vals[i] = v;
                    }
                }
            }
            ui.add_space(8.0);
            widgets::eyebrow(ui, "Per-Muscle Settings");
            for group in MuscleGroup::ALL {
                let id = ui.make_persistent_id(("zestengine-muscle-group", group.label()));
                let (open, _) = widgets::section_header_with(ui, id, None, group.label(), None, group == MuscleGroup::Body, |_| {});
                if !open {
                    continue;
                }
                for (i, info) in muscles().iter().enumerate().filter(|(_, m)| m.group == group) {
                    let modified = rig.muscles.is_modified(i);
                    ui.horizontal(|ui| {
                        let color = if modified { theme::accent() } else { theme::TEXT_BODY };
                        let label = ui.add_sized([150.0, theme::H_FIELD], egui::Label::new(RichText::new(short_name(&info.name, group)).color(color).size(theme::SIZE_BODY)).truncate());
                        label.on_hover_text(&info.name);
                        let before = vals[i];
                        let mut v = before;
                        ui.add(egui::Slider::new(&mut v, -1.0..=1.0).show_value(true).fixed_decimals(2));
                        if v != before {
                            vals[i] = snap_dead_zone(v);
                            self.focus = Some(i);
                        }
                        let open = self.focus == Some(i);
                        if widgets::icon_button_sized(ui, Icon::Gear, "Edit the limits of this muscle", open, egui::Vec2::splat(theme::H_BUTTON_SMALL)).clicked() {
                            self.focus = if open { None } else { Some(i) };
                        }
                    });
                    if self.focus == Some(i) {
                        let (mut lo, mut hi) = rig.muscles.limit(i);
                        let (d_lo, d_hi) = (info.min, info.max);
                        let mut changed = false;
                        ui.horizontal(|ui| {
                            ui.add_space(12.0);
                            ui.label(RichText::new("Min").size(theme::SIZE_META).color(theme::TEXT_MUTED));
                            changed |= ui.add(egui::DragValue::new(&mut lo).range(-180.0..=0.0).speed(0.5).suffix(" deg")).changed();
                            ui.label(RichText::new("Max").size(theme::SIZE_META).color(theme::TEXT_MUTED));
                            changed |= ui.add(egui::DragValue::new(&mut hi).range(0.0..=180.0).speed(0.5).suffix(" deg")).changed();
                            ui.label(RichText::new(format!("default {d_lo:.1} to {d_hi:.1}")).size(theme::SIZE_META).color(theme::TEXT_DIM));
                            if ui.add_enabled(modified, egui::Button::new("Reset")).on_hover_text("Back to the default limits").clicked() {
                                rig.muscles.reset(i);
                                rig.auto_generated = false;
                                out.changed = true;
                            }
                        });
                        if changed {
                            rig.muscles.set_limit(i, lo, hi);
                            rig.auto_generated = false;
                            out.changed = true;
                        }
                    }
                }
            }
        });
        if let Some(neutral) = reset {
            self.reset_values(&mr, neutral);
        } else if let Some(v) = self.values.as_mut() {
            **v = vals;
        }
    }
}

/// A muscle's name without its group prefix ("Left Arm Down-Up" in Left Arm: "Arm Down-Up").
fn short_name(name: &str, group: MuscleGroup) -> &str {
    match group {
        MuscleGroup::LeftArm | MuscleGroup::LeftFingers | MuscleGroup::LeftLeg => name.strip_prefix("Left ").unwrap_or(name),
        MuscleGroup::RightArm | MuscleGroup::RightFingers | MuscleGroup::RightLeg => name.strip_prefix("Right ").unwrap_or(name),
        _ => name,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::skeleton;
    use zestengine_assets::muscle::muscle_index;

    fn setup() -> (Skeleton, MuscleRig) {
        let skel = skeleton();
        let mr = MuscleRig::new(&skel, skel.rig.as_ref().unwrap()).unwrap();
        (skel, mr)
    }

    #[test]
    fn the_dead_zone_snaps_small_values_to_zero() {
        assert_eq!(snap_dead_zone(0.05), 0.0);
        assert_eq!(snap_dead_zone(-0.099), 0.0);
        assert_eq!(snap_dead_zone(0.1), 0.1);
        assert_eq!(snap_dead_zone(2.0), 1.0);
    }

    #[test]
    fn masters_cover_their_families() {
        let m = masters();
        assert_eq!(m.iter().map(|m| m.label).collect::<Vec<_>>(), ["Open Close", "Left Right", "Roll Left Right", "In Out", "Roll In Out", "Finger Open Close", "Finger In Out"]);
        assert!(m.iter().all(|m| !m.members.is_empty()));
        assert_eq!(m[5].members.len(), 30, "30 finger stretch muscles");
        assert_eq!(m[6].members.len(), 10, "10 spread muscles");
        let has = |k: usize, n: &str| m[k].members.contains(&muscle_index(n).unwrap());
        assert!(has(0, "Left Forearm Stretch") && has(0, "Spine Front-Back") && has(0, "Left Upper Leg Front-Back"));
        assert!(has(1, "Neck Tilt Left-Right") && has(2, "Head Turn Left-Right") && has(3, "Left Hand In-Out") && has(4, "Left Arm Twist In-Out"));
        assert!(!has(0, "Left Eye Down-Up") && !has(3, "Left Eye In-Out"), "eyes stay out of the body masters");
    }

    #[test]
    fn the_preview_pose_is_the_reference_at_rest_values_and_bends_at_zero() {
        let (skel, mr) = setup();
        let rest = posed_locals(&skel, &mr, &mr.rest_pose().muscles);
        let reference = reference_locals(&skel);
        for (i, (a, b)) in rest.iter().zip(&reference).enumerate() {
            assert!(a.abs_diff_eq(*b, 1e-3), "joint {i} moved at the reference values");
        }
        // Every muscle at 0: the elbow is bent 80 degrees.
        let neutral = posed_locals(&skel, &mr, &[0.0; MUSCLE_COUNT]);
        let world = skel.world_matrices(&neutral);
        let at = |n: &str| world[skel.joints.iter().position(|j| j.name == n).unwrap()].w_axis.truncate();
        let upper = (at("LeftLowerArm") - at("LeftUpperArm")).normalize();
        let lower = (at("LeftHand") - at("LeftLowerArm")).normalize();
        assert!((upper.angle_between(lower).to_degrees() - 80.0).abs() < 0.2);
        assert!(neutral.len() == skel.joints.len());
    }

    #[test]
    fn editing_a_limit_changes_what_a_slider_does() {
        let (skel, mr) = setup();
        let mut rig = skel.rig.clone().unwrap();
        let i = muscle_index("Left Arm Down-Up").unwrap();
        let mut v = mr.rest_pose().muscles;
        v[i] = 1.0;
        let tip = |mr: &MuscleRig| {
            let w = skel.world_matrices(&posed_locals(&skel, mr, &v));
            w[skel.joints.iter().position(|j| j.name == "LeftHand").unwrap()].w_axis.y
        };
        let before = tip(&mr);
        rig.muscles.set_limit(i, -60.0, 30.0);
        let narrow = MuscleRig::new(&skel, &rig).unwrap();
        assert!(tip(&narrow) < before - 0.05, "a +1 slider reaches less far with a smaller max");
        rig.muscles.reset(i);
        assert!((tip(&MuscleRig::new(&skel, &rig).unwrap()) - before).abs() < 1e-5, "Reset restores the default limits");
    }

    #[test]
    fn the_preview_only_exists_while_the_tab_is_open() {
        let (skel, _) = setup();
        let rig = skel.rig.clone().unwrap();
        let mut ui = MuscleUi::default();
        assert!(ui.preview_locals(&skel, &rig).is_none());
        ui.active = true;
        let locals = ui.preview_locals(&skel, &rig).expect("preview");
        assert_eq!(locals.len(), skel.joints.len());
    }
}
