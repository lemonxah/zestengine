//! The editor's Performance window (GEO-2): the rank of the selected hierarchy
//! (or the whole scene) per metric against the budgets for a content kind and
//! device tier, and the player-facing avatar safety settings with the avatars
//! they currently hide.

use egui::RichText;
use zestengine_assets::perf_rank::{AvatarPolicy, PerfMetric, PerfRank, PerfTier};
use zestengine_core::ContentKind;
use crate::{theme, widgets};

use zestengine_engine::scene::LoadedScene;

/// Window state kept between frames.
pub struct PerfWindowState {
    kind: ContentKind,
    tier: PerfTier,
    whole_scene: bool,
}

impl Default for PerfWindowState {
    fn default() -> Self {
        Self { kind: ContentKind::Avatar, tier: PerfTier::Desktop, whole_scene: false }
    }
}

/// Colour of a rank: green through red.
pub fn rank_color(r: PerfRank) -> egui::Color32 {
    match r {
        PerfRank::Excellent | PerfRank::Good => theme::STATUS_OK,
        PerfRank::Medium => theme::STATUS_INFO,
        PerfRank::Poor => theme::STATUS_WARN_FG,
        PerfRank::VeryPoor => theme::STATUS_ERROR,
    }
}

fn format_value(metric: PerfMetric, v: u64) -> String {
    match metric {
        PerfMetric::TextureMemory => format!("{:.1} MiB", v as f64 / (1024.0 * 1024.0)),
        PerfMetric::Triangles if v >= 10_000 => format!("{:.1}k", v as f64 / 1000.0),
        _ => v.to_string(),
    }
}

/// Draw the window. `selected` is the selected scene object, if any.
pub fn performance_window(ctx: &egui::Context, open: &mut bool, st: &mut PerfWindowState, scene: &mut LoadedScene, selected: Option<usize>) {
    egui::Window::new("Performance").open(open).resizable(false).default_width(340.0).show(ctx, |ui| {
        widgets::eyebrow(ui, "Budget");
        widgets::property_row(ui, "Content", |ui| {
            widgets::segmented(ui, &mut st.kind, &[(ContentKind::Avatar, "Avatar"), (ContentKind::Prop, "Prop"), (ContentKind::World, "World")]);
        });
        widgets::property_row(ui, "Device", |ui| {
            widgets::segmented(ui, &mut st.tier, &[(PerfTier::Desktop, "Desktop"), (PerfTier::Mobile, "Mobile")]);
        });
        let root = if st.whole_scene { None } else { selected };
        widgets::property_row(ui, "Measure", |ui| {
            widgets::segmented(ui, &mut st.whole_scene, &[(false, "Selection"), (true, "Whole scene")]);
        });
        if root.is_none() && !st.whole_scene {
            widgets::info_note(ui, "Select an object to rank its hierarchy, or measure the whole scene.");
            return;
        }
        let report = scene.perf_report(root, st.kind, st.tier);

        widgets::eyebrow(ui, "Rank");
        ui.horizontal(|ui| {
            widgets::status_pill(ui, report.overall.label(), rank_color(report.overall));
            if report.overall != PerfRank::Excellent
                && let Some(w) = report.worst()
            {
                ui.label(RichText::new(format!("limited by {}", w.metric.label().to_lowercase())).size(theme::SIZE_META).color(theme::TEXT_MUTED));
            }
        });
        ui.add_space(4.0);
        for m in &report.metrics {
            widgets::property_row(ui, m.metric.label(), |ui| {
                ui.label(RichText::new(format_value(m.metric, m.value)).font(theme::mono(11.0)).color(theme::TEXT_BODY));
                widgets::status_pill(ui, m.rank.label(), rank_color(m.rank));
                if let Some(better) = m.next_better {
                    ui.label(RichText::new(format!("≤ {} for the next rank", format_value(m.metric, better))).size(theme::SIZE_META).color(theme::TEXT_DIM));
                }
            });
        }
        widgets::info_note(
            ui,
            "Models get automatic screen-size LODs, so triangles cost less at a distance; materials, bones, blendshapes and textures cost the same wherever the avatar stands. Texture memory counts textures embedded in models.",
        );

        widgets::eyebrow(ui, "Player safety");
        let mut policy = scene.avatar_policy();
        let mut tier = scene.perf_tier();
        widgets::property_row(ui, "Device", |ui| {
            widgets::segmented(ui, &mut tier, &[(PerfTier::Desktop, "Desktop"), (PerfTier::Mobile, "Mobile")]);
        });
        widgets::property_row(ui, "Minimum rank", |ui| {
            egui::ComboBox::from_id_salt("perf_min_rank").selected_text(policy.min_rank.label()).show_ui(ui, |ui| {
                widgets::menu_frame(ui);
                for r in PerfRank::ALL {
                    ui.selectable_value(&mut policy.min_rank, r, r.label());
                }
            });
        });
        widgets::property_row(ui, "Max shown", |ui| {
            ui.add(egui::DragValue::new(&mut policy.max_shown).range(0..=256).speed(0.2).custom_formatter(|v, _| if v == 0.0 { "no limit".to_string() } else { format!("{v}") }))
                .on_hover_text("Other players' avatars shown at full detail, nearest first (0 = no limit)")
        });
        widgets::property_row(ui, "Cull distance", |ui| {
            ui.add(egui::DragValue::new(&mut policy.cull_distance).range(0.0..=1000.0).speed(0.5).suffix(" m"))
                .on_hover_text("Other players' avatars past this distance are not drawn (0 = never)")
        });
        if ui.small_button("Reset to the device default").clicked() {
            policy = AvatarPolicy::for_tier(tier);
        }
        scene.set_perf_tier(tier);
        scene.set_avatar_policy(policy);

        let statuses = scene.avatar_statuses();
        if !statuses.is_empty() {
            widgets::eyebrow(ui, "Other players");
            for s in statuses {
                let name = scene.index_of(s.object).map(|i| scene.objects[i].name.clone()).unwrap_or_default();
                widgets::property_row(ui, name, |ui| {
                    widgets::status_pill(ui, s.rank.label(), rank_color(s.rank));
                    match s.hidden {
                        Some(why) => ui.label(RichText::new(format!("hidden: {}", why.label())).size(theme::SIZE_META).color(theme::STATUS_WARN_FG)),
                        None => ui.label(RichText::new(format!("{:.0} m", s.distance)).size(theme::SIZE_META).color(theme::TEXT_DIM)),
                    };
                });
            }
        }
        widgets::info_note(ui, "Hidden avatars are not drawn until a fallback model is configured.");
    });
}
