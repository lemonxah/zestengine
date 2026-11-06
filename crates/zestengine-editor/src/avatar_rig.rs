//! The Avatar Descriptor's view of its model's rig.
//!
//! The humanoid map lives in the model's `.meta` (the Rig section of the model
//! inspector). The editor publishes an [`AvatarRigInfo`] for the selected
//! avatar under [`AVATAR_RIG_KEY`]; the descriptor shows its status, keeps only
//! true overrides in a collapsed section and offers a one-time move of its old
//! bone table into the model. Requests go back through [`OPEN_RIG_KEY`] and
//! [`MOVE_TO_MODEL_KEY`] (egui temp data, like the other inspector hand-offs).

use crate::{theme, widgets};
use egui::{RichText, Ui};

/// Temp-data key: the [`AvatarRigInfo`] of the avatar being inspected.
pub const AVATAR_RIG_KEY: &str = "zestengine-avatar-rig";
/// Temp-data key set (to the model's project-relative path, a `String`) when Open Rig is clicked.
pub const OPEN_RIG_KEY: &str = "zestengine-open-rig";
/// Temp-data key set (to the descriptor's `Vec<(String, String)>` bone table) when
/// Move to model rig is clicked.
pub const MOVE_TO_MODEL_KEY: &str = "zestengine-avatar-move-to-model";

/// What the avatar's model says about its rig.
#[derive(Clone, Debug, Default)]
pub struct AvatarRigInfo {
    /// Project-relative path of the model that carries the skeleton.
    pub model: String,
    /// "Humanoid", "Generic" or "No rig".
    pub kind: String,
    /// Required humanoid bones the model maps (of [`REQUIRED_BONES`]).
    pub required_mapped: usize,
    /// The model's stored human map: Unity-style bone name (`LeftHand`) and joint name.
    pub human: Vec<(String, String)>,
    /// The rig was computed automatically rather than configured by hand.
    pub auto_generated: bool,
}

/// Bones a humanoid must map.
pub const REQUIRED_BONES: usize = 15;

/// A role or bone name without case, spaces, `_` and `-`, so `left_hand`,
/// `Left Hand` and `LeftHand` compare equal.
fn key(name: &str) -> String {
    name.chars().filter(|c| !matches!(c, ' ' | '_' | '-')).collect::<String>().to_lowercase()
}

impl AvatarRigInfo {
    pub fn is_humanoid(&self) -> bool {
        self.kind == "Humanoid" && self.required_mapped == REQUIRED_BONES
    }

    /// "Humanoid, 15/15 required bones, from model".
    pub fn status_line(&self) -> String {
        format!("{}, {}/{REQUIRED_BONES} required bones, from model", self.kind, self.required_mapped)
    }

    /// Whether a descriptor entry (role, joint) differs from what the model
    /// already maps, i.e. is a real override.
    pub fn is_true_override(&self, role: &str, joint: &str) -> bool {
        let role = key(role);
        self.human.iter().find(|(b, _)| key(b) == role).map(|(_, j)| j.as_str()) != Some(joint)
    }
}

/// Indices of `bone_map` entries that really override the model's rig.
pub fn true_overrides(info: Option<&AvatarRigInfo>, bone_map: &[(String, String)]) -> Vec<usize> {
    (0..bone_map.len()).filter(|&i| info.is_none_or(|inf| inf.is_true_override(&bone_map[i].0, &bone_map[i].1))).collect()
}

/// The rig status, Open Rig button, Overrides section and migration for an
/// Avatar Descriptor. Returns true when `bone_map` changed.
pub fn ui(ui: &mut Ui, bone_map: &mut Vec<(String, String)>) -> bool {
    let mut changed = false;
    let info = ui.data(|d| d.get_temp::<AvatarRigInfo>(egui::Id::new(AVATAR_RIG_KEY)));
    ui.horizontal(|ui| match &info {
        Some(i) if i.is_humanoid() => {
            widgets::status_pill(ui, &i.status_line(), theme::STATUS_OK);
        }
        Some(i) => {
            widgets::status_pill(ui, &i.status_line(), theme::STATUS_WARN_FG);
        }
        None => {
            ui.label(RichText::new("No model with a skeleton under this object").size(theme::SIZE_META).color(theme::TEXT_DIM));
        }
    });
    if let Some(i) = &info {
        ui.horizontal(|ui| {
            if widgets::secondary_button(ui, None, "Open Rig").on_hover_text("Open the model's Rig section and Configure window").clicked() {
                ui.data_mut(|d| d.insert_temp(egui::Id::new(OPEN_RIG_KEY), i.model.clone()));
            }
            if i.auto_generated {
                widgets::badge(ui, "automatic");
            }
        });
    }

    let shown = true_overrides(info.as_ref(), bone_map);
    let hidden = bone_map.len() - shown.len();
    egui::CollapsingHeader::new(format!("Overrides ({})", shown.len()))
        .id_salt("avatar-rig-overrides")
        .default_open(false)
        .show(ui, |ui| {
            ui.label(RichText::new("Humanoid role -> joint name, only where it differs from the model rig.").size(theme::SIZE_META).color(theme::TEXT_DIM));
            let mut remove = None;
            for &i in &shown {
                let (role, joint) = &mut bone_map[i];
                ui.horizontal(|ui| {
                    changed |= ui.add(egui::TextEdit::singleline(role).desired_width(110.0).hint_text("left_hand")).changed();
                    changed |= ui.add(egui::TextEdit::singleline(joint).desired_width(130.0).hint_text("joint name")).changed();
                    if widgets::icon_button(ui, crate::icons::Icon::Close, "Remove override", false).clicked() {
                        remove = Some(i);
                    }
                });
            }
            if let Some(i) = remove {
                bone_map.remove(i);
                changed = true;
            }
            if hidden > 0 {
                ui.label(RichText::new(format!("{hidden} entr{} already match the model rig and are hidden.", if hidden == 1 { "y" } else { "ies" })).size(theme::SIZE_META).color(theme::TEXT_DIM));
            }
            if widgets::secondary_button(ui, Some(crate::icons::Icon::Plus), "Add bone override").clicked() {
                bone_map.push((String::new(), String::new()));
                changed = true;
            }
        });

    if info.is_some() && !bone_map.is_empty() {
        let tip = "Write this table into the model's rig (a hand-configured map) and clear it here";
        if widgets::secondary_button(ui, None, "Move to model rig").on_hover_text(tip).clicked() {
            ui.data_mut(|d| d.insert_temp(egui::Id::new(MOVE_TO_MODEL_KEY), bone_map.clone()));
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info() -> AvatarRigInfo {
        AvatarRigInfo {
            model: "avatars/hero.fbx".into(),
            kind: "Humanoid".into(),
            required_mapped: 2,
            human: vec![("Hips".into(), "Hips".into()), ("LeftHand".into(), "Hand_L".into())],
            auto_generated: true,
        }
    }

    #[test]
    fn only_true_overrides_are_listed() {
        let map = vec![
            ("hips".to_string(), "Hips".to_string()),
            ("left_hand".to_string(), "Wrist_L".to_string()),
            ("left_hand".to_string(), "Hand_L".to_string()),
            ("nonsense".to_string(), "X".to_string()),
            ("head".to_string(), "Head".to_string()),
        ];
        assert_eq!(true_overrides(Some(&info()), &map), vec![1, 3, 4], "matching entries hide; unknown roles and unmapped bones stay");
        assert_eq!(true_overrides(None, &map), vec![0, 1, 2, 3, 4], "without a model every entry is an override");
    }

    #[test]
    fn status_line_counts_required_bones() {
        let i = info();
        assert_eq!(i.status_line(), "Humanoid, 2/15 required bones, from model");
        assert!(!i.is_humanoid());
    }
}
