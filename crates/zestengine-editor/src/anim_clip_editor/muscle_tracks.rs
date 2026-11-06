//! Muscle clips in the Animation tab: the Add Property menu for muscle,
//! root and goal curves, track labels, and the bake of an imported humanoid
//! clip into a muscle clip asset.
//!
//! A muscle clip's tracks are ordinary [`AnimTrack`]s (so the timeline, curve
//! editor and key inspector work on them) addressed by muscle name instead of
//! an object path. Viewing the result on the avatar happens in play mode, where
//! the Animator evaluates the clip through the muscle model.

use std::path::{Path, PathBuf};

use zestengine_assets::muscle::{GOAL_NAMES, MuscleGroup, MusclePose, MuscleRig, bake_muscle_clip, muscles};
use zestengine_assets::{AnimClip, AnimTrack, AnimationClip, ClipKind, Skeleton, TrackProperty};
use crate::icons::Icon;
use crate::widgets;

use super::{AnimClipTab, Target};
use crate::anim_controller_editor::{rel_path, unique_path};
use zestengine_engine::scene::LoadedScene;

/// Name shown for a muscle-clip track: the muscle, the root, or the goal.
pub(super) fn label(track: &AnimTrack) -> String {
    match track.property {
        TrackProperty::Muscle => track.name.clone(),
        TrackProperty::GoalPosition | TrackProperty::GoalRelative => format!("{} / {}", track.property.label(), track.name),
        p => p.label().to_owned(),
    }
}

/// Bake `clip` (an imported clip of `skel`) into a new muscle clip asset in
/// `dir`, named after the clip, and return its path. Fails when the skeleton
/// has no usable humanoid map.
pub(super) fn bake_to_file(dir: &Path, skel: &Skeleton, clip: &AnimationClip, fps: f32) -> anyhow::Result<PathBuf> {
    let rig = skel.rig.as_ref().filter(|r| r.is_configured()).ok_or_else(|| anyhow::anyhow!("the model has no humanoid avatar"))?;
    let mr = MuscleRig::new(skel, rig).ok_or_else(|| anyhow::anyhow!("the humanoid map has no Hips joint in this model"))?;
    let baked = bake_muscle_clip(skel, &mr, clip, fps);
    let path = unique_path(dir, &format!("{} Muscles", clip.name), zestengine_assets::ANIM_CLIP_EXTENSION);
    zestengine_assets::save_anim_clip(&path, &baked)?;
    Ok(path)
}

impl AnimClipTab {
    /// A muscle clip is open.
    pub(super) fn is_muscle_clip(&self) -> bool {
        self.clip.kind == ClipKind::Muscle
    }

    /// Key `property` (named `name`) at the playhead with `value`, adding the track when missing.
    fn key_muscle(&mut self, property: TrackProperty, name: &str, value: [f32; 4]) {
        let t = self.snap(self.playhead, false);
        let i = match self.clip.prop_track_index("", property, name, 0) {
            Some(i) => i,
            None => {
                self.clip.tracks.push(AnimTrack::prop("", property, name, 0));
                self.clip.tracks.len() - 1
            }
        };
        self.clip.tracks[i].set_key(t, value);
        self.clip.length = self.clip.length.max(t);
    }

    /// The Add Property menu of a muscle clip. `reference` is the target
    /// avatar's reference pose (new curves start there); without one they start at 0.
    pub(super) fn muscle_add_menu(&mut self, ui: &mut egui::Ui, reference: Option<&MusclePose>) -> bool {
        let mut edited = false;
        let base = reference.copied().unwrap_or_default();
        for group in MuscleGroup::ALL {
            ui.menu_button(group.label(), |ui| {
                widgets::menu_frame(ui);
                egui::ScrollArea::vertical().max_height(360.0).show(ui, |ui| {
                    if widgets::menu_item(ui, None, "Add All", None).clicked() {
                        for (i, m) in muscles().iter().enumerate().filter(|(_, m)| m.group == group) {
                            if self.clip.prop_track_index("", TrackProperty::Muscle, &m.name, 0).is_none() {
                                self.key_muscle(TrackProperty::Muscle, &m.name, [base.muscles[i], 0.0, 0.0, 0.0]);
                            }
                        }
                        edited = true;
                        ui.close();
                    }
                    widgets::menu_separator(ui);
                    for (i, m) in muscles().iter().enumerate().filter(|(_, m)| m.group == group) {
                        let exists = self.clip.prop_track_index("", TrackProperty::Muscle, &m.name, 0).is_some();
                        if ui.add_enabled(!exists, egui::Button::new(&m.name)).clicked() {
                            self.key_muscle(TrackProperty::Muscle, &m.name, [base.muscles[i], 0.0, 0.0, 0.0]);
                            edited = true;
                            ui.close();
                        }
                    }
                });
            });
        }
        ui.separator();
        let has = |c: &AnimClip, p: TrackProperty| c.prop_track_index("", p, "", 0).is_some();
        if ui.add_enabled(!has(&self.clip, TrackProperty::RootPosition), egui::Button::new("Root Position")).clicked() {
            let p = base.root.position;
            self.key_muscle(TrackProperty::RootPosition, "", [p.x, p.y, p.z, 0.0]);
            edited = true;
            ui.close();
        }
        if ui.add_enabled(!has(&self.clip, TrackProperty::RootRotation), egui::Button::new("Root Rotation")).clicked() {
            self.key_muscle(TrackProperty::RootRotation, "", base.root.rotation.to_array());
            edited = true;
            ui.close();
        }
        let goals = base.goals;
        let missing = GOAL_NAMES.iter().any(|g| self.clip.prop_track_index("", TrackProperty::GoalPosition, g, 0).is_none());
        if ui
            .add_enabled(goals.is_some() && missing, egui::Button::new("Limb Goals"))
            .on_hover_text("Hand and foot targets: the other avatar's limbs reach for them by direction and length ratio")
            .clicked()
            && let Some(g) = goals
        {
            for (k, name) in GOAL_NAMES.iter().enumerate() {
                self.key_muscle(TrackProperty::GoalPosition, name, [g.position[k].x, g.position[k].y, g.position[k].z, 0.0]);
                self.key_muscle(TrackProperty::GoalRelative, name, [g.relative[k].x, g.relative[k].y, g.relative[k].z, 0.0]);
            }
            edited = true;
            ui.close();
        }
        edited
    }

    /// The "Bake Muscles" button: a menu of the imported clips of the target's
    /// skeleton; picking one writes a muscle clip asset and opens it.
    pub(super) fn bake_menu(&mut self, ui: &mut egui::Ui, scene: &LoadedScene, root: &Path, target: Option<&Target>) {
        let Some((skel, clips)) = target.and_then(|t| scene.skeleton_with_clips(t.index)) else {
            return;
        };
        if clips.is_empty() || !skel.rig.as_ref().is_some_and(|r| r.is_configured()) {
            return;
        }
        let button = widgets::secondary_button(ui, Some(Icon::Plus), "Bake Muscles")
            .on_hover_text("Bake an imported humanoid clip into a muscle clip that any humanoid avatar can play");
        egui::Popup::menu(&button).show(|ui| {
            widgets::menu_frame(ui);
            for clip in &clips {
                if widgets::menu_item(ui, None, &clip.name, None).clicked() {
                    match bake_to_file(&root.join("animations"), skel, clip, 30.0) {
                        Ok(path) => {
                            self.open(root, &rel_path(root, &path));
                            self.open_request = true;
                        }
                        Err(e) => tracing::error!("baking {} to a muscle clip: {e:#}", clip.name),
                    }
                    ui.close();
                }
            }
        });
    }

    /// The target avatar's reference pose in muscle space (None for targets without a humanoid skeleton).
    pub(super) fn target_reference(&self, scene: &LoadedScene, target: Option<&Target>) -> Option<MusclePose> {
        let skel = scene.object_skeleton(target?.index)?;
        let rig = skel.rig.as_ref().filter(|r| r.is_configured())?;
        Some(MuscleRig::new(skel, rig)?.reference_pose())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::skeleton;
    use glam::{Quat, Vec3};
    use zestengine_assets::{AnimChannel, ChannelPath};

    #[test]
    fn muscle_tracks_are_labelled_by_muscle_and_goal() {
        let m = AnimTrack::prop("", TrackProperty::Muscle, "Left Arm Down-Up", 0);
        assert_eq!(label(&m), "Left Arm Down-Up");
        assert_eq!(label(&AnimTrack::prop("", TrackProperty::GoalPosition, "LeftFoot", 0)), "Goal Position / LeftFoot");
        assert_eq!(label(&AnimTrack::new("", TrackProperty::RootRotation)), "Root Rotation");
    }

    #[test]
    fn baking_an_imported_clip_writes_a_muscle_asset_next_to_the_others() {
        let skel = skeleton();
        let arm = skel.joints.iter().position(|j| j.name == "LeftUpperArm").unwrap();
        let clip = AnimationClip {
            name: "Wave".into(),
            duration: 0.5,
            channels: vec![AnimChannel {
                joint: arm,
                path: ChannelPath::Rotation,
                times: vec![0.0, 0.5],
                vec_values: vec![],
                quat_values: vec![Quat::IDENTITY, Quat::from_rotation_z(0.8)], extra: None
            }],
            morph_tracks: Vec::new(), playback: None,
        };
        let dir = std::env::temp_dir().join(format!("zest-muscle-bake-{}", std::process::id()));
        let path = bake_to_file(&dir, &skel, &clip, 24.0).unwrap();
        assert!(path.file_name().unwrap().to_string_lossy().starts_with("Wave Muscles"));
        let loaded = zestengine_assets::load_anim_clip(&path).unwrap();
        assert_eq!(loaded.kind, ClipKind::Muscle);
        assert_eq!(loaded.fallback, "Wave");
        assert_eq!(loaded.frame_rate, 24.0);
        let arm_curve = loaded.prop_track_index("", TrackProperty::Muscle, "Left Arm Down-Up", 0).unwrap();
        assert_eq!(loaded.tracks[arm_curve].keys.len(), 13, "0.5 s at 24 fps is 12 frames plus the end");
        // A second bake does not overwrite the first.
        let again = bake_to_file(&dir, &skel, &clip, 24.0).unwrap();
        assert_ne!(again, path);
        // A skeleton without a humanoid map cannot bake.
        let mut plain = skel.clone();
        plain.rig = None;
        assert!(bake_to_file(&dir, &plain, &clip, 24.0).is_err());
        let _ = std::fs::remove_dir_all(&dir);
        let _ = Vec3::ZERO;
    }
}
