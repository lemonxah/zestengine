//! The player's avatar safety settings (GEO-2): persisted per user, and the
//! in-game overlay (F9) that edits them and shows the performance rank of the
//! other players' avatars. Hidden avatars are drawn as rank-tinted proxies by
//! the scene (`scene/perf.rs`); this is only the settings and the UI.

use std::path::{Path, PathBuf};

use egui::{Align2, Color32, FontId, Pos2, RichText};
use glam::{Mat4, Vec3, Vec4};
use serde::{Deserialize, Serialize};
use zestengine_assets::perf_rank::{AvatarPolicy, PerfRank};

use crate::scene::LoadedScene;

/// What the player chose, saved as `<user data>/<game>/player_settings.ron`
/// next to the saves folder.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct PlayerSettings {
    /// The avatar safety policy; `None` = the device-tier default.
    pub avatar_policy: Option<AvatarPolicy>,
    /// Rank labels floating over other players' avatars.
    pub show_rank_badges: bool,
}


impl PlayerSettings {
    /// Read the settings; a missing or unreadable file gives the defaults.
    pub fn load(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(text) => ron::from_str(&text).unwrap_or_else(|e| {
                tracing::warn!("player settings {}: {e}; using defaults", path.display());
                Self::default()
            }),
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self, path: &Path) {
        let text = match ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default()) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!("player settings not saved: {e}");
                return;
            }
        };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Err(e) = zestengine_assets::write_atomic(path, text.as_bytes()) {
            tracing::warn!("player settings {}: {e:#}", path.display());
        }
    }
}

/// `player_settings.ron` beside a game's saves folder.
pub fn settings_path(save_dir: &Path) -> PathBuf {
    save_dir.parent().unwrap_or(save_dir).join("player_settings.ron")
}

fn rank_color(r: PerfRank) -> Color32 {
    let t = crate::scene::perf::rank_tint(r);
    Color32::from_rgb((t[0] * 255.0) as u8, (t[1] * 255.0) as u8, (t[2] * 255.0) as u8)
}

/// Screen position of a world point under `view_proj` for a `size` screen, or
/// `None` behind the camera.
pub fn project_to_screen(view_proj: &Mat4, world: Vec3, size: egui::Vec2) -> Option<Pos2> {
    let clip = *view_proj * Vec4::new(world.x, world.y, world.z, 1.0);
    (clip.w > 1e-4).then(|| {
        let ndc = clip.truncate() / clip.w;
        // The game projection is y-up (the renderer flips it): screen y grows downward.
        Pos2::new((ndc.x * 0.5 + 0.5) * size.x, (0.5 - ndc.y * 0.5) * size.y)
    })
}

/// Overlay state: open flag, whether the policy was edited since the last save.
#[derive(Default)]
pub struct SafetyMenu {
    pub open: bool,
    /// Camera `proj * view` of this frame, for the rank badges.
    pub view_proj: Option<Mat4>,
    dirty: bool,
}

impl SafetyMenu {
    /// True once after the player edited something and let go of the pointer:
    /// time to save (a drag does not write the file on every step).
    pub fn take_dirty(&mut self, pointer_down: bool) -> bool {
        if self.dirty && !pointer_down {
            self.dirty = false;
            return true;
        }
        false
    }

    #[cfg(test)]
    fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Draw the rank badges (if enabled) and the settings window (if open).
    /// Edits go straight into the scene's policy; `settings` carries the badge
    /// toggle.
    pub fn show(&mut self, ctx: &egui::Context, scene: &mut LoadedScene, settings: &mut PlayerSettings, screen: egui::Vec2) {
        if settings.show_rank_badges
            && let Some(vp) = self.view_proj
        {
            let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Background, egui::Id::new("avatar_rank_badges")));
            for s in scene.avatar_statuses() {
                let Some(i) = scene.index_of(s.object) else { continue };
                let head = scene.frame_world(i).w_axis.truncate() + Vec3::Y * 2.1;
                if let Some(p) = project_to_screen(&vp, head, screen) {
                    painter.text(p, Align2::CENTER_BOTTOM, s.rank.label(), FontId::proportional(13.0), rank_color(s.rank));
                }
            }
        }
        if !self.open {
            return;
        }
        let mut policy = scene.avatar_policy();
        let before = (policy, settings.show_rank_badges);
        let mut open = self.open;
        egui::Window::new("Avatar safety").open(&mut open).resizable(false).anchor(Align2::RIGHT_TOP, [-12.0, 12.0]).show(ctx, |ui| {
            ui.label("Other players' avatars that break these limits are shown as a plain tinted figure.");
            ui.add_space(4.0);
            egui::Grid::new("avatar_safety_grid").num_columns(2).show(ui, |ui| {
                ui.label("Minimum rank");
                egui::ComboBox::from_id_salt("safety_min_rank").selected_text(policy.min_rank.label()).show_ui(ui, |ui| {
                    for r in PerfRank::ALL {
                        ui.selectable_value(&mut policy.min_rank, r, r.label());
                    }
                });
                ui.end_row();
                ui.label("Max shown");
                ui.add(egui::DragValue::new(&mut policy.max_shown).range(0..=256).speed(0.2).custom_formatter(|v, _| if v == 0.0 { "no limit".to_string() } else { format!("{v}") }));
                ui.end_row();
                ui.label("Cull distance");
                ui.add(egui::DragValue::new(&mut policy.cull_distance).range(0.0..=1000.0).speed(0.5).suffix(" m"));
                ui.end_row();
                ui.label("Impostor distance");
                ui.add(egui::DragValue::new(&mut policy.impostor_distance).range(0.0..=1000.0).speed(0.5).suffix(" m").custom_formatter(|v, _| if v == 0.0 { "never".to_string() } else { format!("{v} m") }));
                ui.end_row();
            });
            ui.checkbox(&mut settings.show_rank_badges, "Show rank over avatars");
            if ui.small_button("Reset to the device default").clicked() {
                policy = AvatarPolicy::for_tier(scene.perf_tier());
            }
            let statuses = scene.avatar_statuses();
            if !statuses.is_empty() {
                ui.separator();
                for s in statuses {
                    let name = scene.index_of(s.object).map(|i| scene.objects[i].name.clone()).unwrap_or_default();
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(s.rank.label()).color(rank_color(s.rank)).strong());
                        ui.label(name);
                        match s.hidden {
                            Some(why) => ui.label(RichText::new(why.label()).weak()),
                            None => ui.label(RichText::new(format!("{:.0} m", s.distance)).weak()),
                        };
                    });
                }
            }
        });
        self.open = open;
        scene.set_avatar_policy(policy);
        if (policy, settings.show_rank_badges) != before {
            self.dirty = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_round_trip_and_tolerate_missing_or_broken_files() {
        let dir = std::env::temp_dir().join(format!("zest_player_settings_{}", std::process::id()));
        let path = dir.join("game").join("player_settings.ron");
        assert_eq!(PlayerSettings::load(&path), PlayerSettings::default(), "no file: defaults");
        let s = PlayerSettings {
            avatar_policy: Some(AvatarPolicy { min_rank: PerfRank::Medium, max_shown: 12, cull_distance: 40.0, ..Default::default() }),
            show_rank_badges: true,
        };
        s.save(&path);
        assert_eq!(PlayerSettings::load(&path), s);
        std::fs::write(&path, "not ron {{").unwrap();
        assert_eq!(PlayerSettings::load(&path), PlayerSettings::default(), "broken file: defaults");
        // Older files without the newer fields still load.
        std::fs::write(&path, "(show_rank_badges: true)").unwrap();
        let old = PlayerSettings::load(&path);
        assert!(old.show_rank_badges && old.avatar_policy.is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn settings_live_beside_the_saves_folder() {
        assert_eq!(settings_path(Path::new("/d/My Game/saves")), PathBuf::from("/d/My Game/player_settings.ron"));
    }

    #[test]
    fn save_waits_for_the_pointer_to_be_released() {
        let mut m = SafetyMenu::default();
        assert!(!m.take_dirty(false));
        m.mark_dirty();
        assert!(!m.take_dirty(true), "still dragging");
        assert!(m.take_dirty(false));
        assert!(!m.take_dirty(false), "saved once");
    }

    #[test]
    fn badges_project_in_front_of_the_camera_only() {
        let proj = Mat4::perspective_rh(1.0, 16.0 / 9.0, 0.1, 100.0);
        let view = Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
        let vp = proj * view;
        let size = egui::vec2(1600.0, 900.0);
        let ahead = project_to_screen(&vp, Vec3::new(0.0, 0.0, -5.0), size).unwrap();
        assert!((ahead.x - 800.0).abs() < 1.0 && (ahead.y - 450.0).abs() < 1.0);
        assert!(project_to_screen(&vp, Vec3::new(0.0, 0.0, 5.0), size).is_none(), "behind the camera");
        let above = project_to_screen(&vp, Vec3::new(0.0, 1.0, -5.0), size).unwrap();
        assert!(above.y < ahead.y, "higher in the world is higher on screen");
    }
}
