//! The UI Editor tab: a canvas previewed at a target resolution (or its 3D
//! panel's pixels) where elements are selected, moved and resized directly,
//! plus an Add menu and ready-made templates.

use std::path::Path;

use zestengine_core::{ObjectId, UiCanvas, UiRenderMode};
use crate::UiPreset;
use crate::egui_phosphor::regular as ph;
use egui::{Color32, CornerRadius, Rect, Sense, Stroke, StrokeKind, Vec2};

use zestengine_engine::game_ui::{GameUi, UiScope, layout_rects};
use zestengine_engine::scene::LoadedScene;
use crate::ui_edit::UiRectEditor;
use zestengine_engine::ui_templates::UiTemplate;

const RESOLUTIONS: [(&str, [f32; 2]); 6] = [
    ("1920 x 1080 (16:9)", [1920.0, 1080.0]),
    ("1280 x 720 (16:9)", [1280.0, 720.0]),
    ("2560 x 1440 (16:9)", [2560.0, 1440.0]),
    ("3440 x 1440 (21:9)", [3440.0, 1440.0]),
    ("1024 x 768 (4:3)", [1024.0, 768.0]),
    ("1080 x 1920 (portrait)", [1080.0, 1920.0]),
];

const SNAPS: [f32; 5] = [0.0, 1.0, 5.0, 10.0, 20.0];

/// What the tab asks the editor to do (applied after the dock pass).
pub enum UiEditorRequest {
    Select(Option<usize>),
    Template(UiTemplate),
    Add(UiPreset),
    SetMode(usize, UiRenderMode),
}

#[derive(Default)]
pub struct UiEditorTab {
    canvas: Option<ObjectId>,
    resolution: usize,
    editor: UiRectEditor,
    game_ui: GameUi,
}

fn canvas(scene: &LoadedScene, i: usize) -> Option<&UiCanvas> {
    scene.objects[i].components.iter().find_map(|c| c.as_any().downcast_ref::<UiCanvas>())
}

/// The canvas an object belongs to (itself or its nearest canvas ancestor).
fn canvas_of(scene: &LoadedScene, mut i: usize) -> Option<usize> {
    for _ in 0..64 {
        if canvas(scene, i).is_some() {
            return Some(i);
        }
        i = scene.objects[i].parent?;
    }
    None
}

impl UiEditorTab {
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        scene: &mut LoadedScene,
        selected: Option<usize>,
        root: &Path,
    ) -> Vec<UiEditorRequest> {
        let mut requests = Vec::new();
        let selected = selected.filter(|&i| i < scene.objects.len());
        if let Some(c) = selected.and_then(|s| canvas_of(scene, s)) {
            self.canvas = Some(scene.objects[c].id);
        }
        let canvases: Vec<usize> = (0..scene.objects.len()).filter(|&i| canvas(scene, i).is_some()).collect();
        let current = self
            .canvas
            .and_then(|id| canvases.iter().copied().find(|&i| scene.objects[i].id == id))
            .or(canvases.first().copied());

        ui.horizontal(|ui| {
            match current {
                Some(c) => {
                    egui::ComboBox::from_id_salt("zestengine-ui-editor-canvas")
                        .selected_text(scene.objects[c].name.clone())
                        .show_ui(ui, |ui| {
                            for &k in &canvases {
                                if ui.selectable_label(k == c, scene.objects[k].name.clone()).clicked() {
                                    self.canvas = Some(scene.objects[k].id);
                                    requests.push(UiEditorRequest::Select(Some(k)));
                                }
                            }
                        });
                    let mode = canvas(scene, c).map(|k| k.render_mode).unwrap_or(UiRenderMode::Screen);
                    for (m, label, tip) in [
                        (UiRenderMode::Screen, "2D", "Screen overlay, drawn over the camera view"),
                        (UiRenderMode::World, "3D", "Panel placed in the world (VR menus, in-game screens)"),
                    ] {
                        if ui.selectable_label(mode == m, label).on_hover_text(tip).clicked() && mode != m {
                            requests.push(UiEditorRequest::SetMode(c, m));
                        }
                    }
                    if mode == UiRenderMode::Screen {
                        egui::ComboBox::from_id_salt("zestengine-ui-editor-res")
                            .selected_text(RESOLUTIONS[self.resolution].0)
                            .show_ui(ui, |ui| {
                                for (k, (label, _)) in RESOLUTIONS.iter().enumerate() {
                                    ui.selectable_value(&mut self.resolution, k, *label);
                                }
                            });
                    }
                    ui.separator();
                    ui.menu_button(format!("{} Add", ph::PLUS), |ui| {
                        crate::widgets::menu_frame(ui);
                        for p in UiPreset::ALL {
                            if matches!(p, UiPreset::Canvas | UiPreset::WorldCanvas) {
                                continue;
                            }
                            if ui.button(p.label()).clicked() {
                                // New elements go under the selection when it's in
                                // this canvas, else under the canvas.
                                if selected.and_then(|s| canvas_of(scene, s)) != Some(c) {
                                    requests.push(UiEditorRequest::Select(Some(c)));
                                }
                                requests.push(UiEditorRequest::Add(p));
                                ui.close();
                            }
                        }
                    });
                }
                None => {
                    ui.label(egui::RichText::new("No canvas").weak());
                }
            }
            template_menu(ui, &mut requests);
            ui.separator();
            ui.label(egui::RichText::new("Snap").small());
            egui::ComboBox::from_id_salt("zestengine-ui-editor-snap")
                .width(56.0)
                .selected_text(if self.editor.snap > 0.0 { format!("{}", self.editor.snap) } else { "Off".into() })
                .show_ui(ui, |ui| {
                    for s in SNAPS {
                        let label = if s > 0.0 { format!("{s} px") } else { "Off".into() };
                        ui.selectable_value(&mut self.editor.snap, s, label);
                    }
                });
        });
        ui.separator();

        let area = ui.available_rect_before_wrap();
        let response = ui.allocate_rect(area, Sense::click_and_drag());
        let painter = ui.painter_at(area);
        painter.rect_filled(area, CornerRadius::ZERO, Color32::from_rgb(28, 28, 32));

        let Some(c) = current else {
            empty_state(ui, area, &mut requests);
            return requests;
        };
        let target = match canvas(scene, c) {
            Some(k) if k.render_mode == UiRenderMode::World => {
                let [w, h] = k.world_pixels();
                [w as f32, h as f32]
            }
            _ => RESOLUTIONS[self.resolution].1,
        };
        let margin = 24.0;
        let fit_scale = ((area.width() - 2.0 * margin) / target[0])
            .min((area.height() - 2.0 * margin) / target[1])
            .max(0.01);
        let view = Rect::from_center_size(area.center(), Vec2::new(target[0], target[1]) * fit_scale);
        painter.rect_filled(view, CornerRadius::ZERO, Color32::from_rgb(16, 17, 20));
        let step = view.width() / 12.0;
        for k in 1..12 {
            let x = view.left() + step * k as f32;
            painter.line_segment([egui::pos2(x, view.top()), egui::pos2(x, view.bottom())], Stroke::new(1.0_f32, Color32::from_white_alpha(6)));
        }
        let rows = (view.height() / step).floor() as i32;
        for k in 1..=rows {
            let y = view.top() + step * k as f32;
            painter.line_segment([egui::pos2(view.left(), y), egui::pos2(view.right(), y)], Stroke::new(1.0_f32, Color32::from_white_alpha(6)));
        }

        let vppp = target[0] / view.width().max(1.0);
        self.game_ui.virtual_ppp = Some(vppp);
        self.game_ui.layer = Some(ui.layer_id());
        self.game_ui.reveal = selected;
        let ctx = ui.ctx().clone();
        self.game_ui.run(&ctx, view, scene, root, None, None, UiScope::Canvas(c));
        painter.rect_stroke(view, CornerRadius::ZERO, Stroke::new(1.0_f32, Color32::from_white_alpha(60)), StrokeKind::Outside);

        let nodes = layout_rects(scene, view, vppp, UiScope::Canvas(c), selected);
        let out = self.editor.interact(ui, &response, scene, &nodes, selected, area, true);
        match out.clicked {
            Some(Some(i)) => requests.push(UiEditorRequest::Select(Some(i))),
            Some(None) => requests.push(UiEditorRequest::Select(Some(c))),
            None => {}
        }
        if let Some(s) = selected.filter(|s| nodes.iter().any(|n| n.0 == *s))
            && response.contains_pointer()
        {
            UiRectEditor::nudge(ui, scene, s);
        }
        self.editor.paint(&painter, scene, &nodes, selected);

        painter.text(
            area.left_bottom() + Vec2::new(8.0, -6.0),
            egui::Align2::LEFT_BOTTOM,
            format!(
                "{} x {} · drag to move · handles resize · Ctrl snaps 10 px · arrows nudge · hidden pages show while selected",
                target[0], target[1]
            ),
            egui::FontId::proportional(11.0),
            Color32::from_white_alpha(110),
        );
        requests
    }
}

fn template_menu(ui: &mut egui::Ui, requests: &mut Vec<UiEditorRequest>) {
    ui.menu_button(format!("{} Templates", ph::LAYOUT), |ui| {
        crate::widgets::menu_frame(ui);
        for t in UiTemplate::ALL {
            if ui.button(t.label()).on_hover_text(t.description()).clicked() {
                requests.push(UiEditorRequest::Template(t));
                ui.close();
            }
        }
    });
}

fn empty_state(ui: &mut egui::Ui, area: Rect, requests: &mut Vec<UiEditorRequest>) {
    let card = Rect::from_center_size(area.center(), Vec2::new(420.0, 330.0)).intersect(area);
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(card).layout(egui::Layout::top_down(egui::Align::Center)));
    child.label(egui::RichText::new("No UI canvas in this scene").strong());
    child.label(egui::RichText::new("Start from a template or an empty canvas.").small().weak());
    child.add_space(8.0);
    for t in UiTemplate::ALL {
        if child
            .add_sized([240.0, 26.0], egui::Button::new(t.label()))
            .on_hover_text(t.description())
            .clicked()
        {
            requests.push(UiEditorRequest::Template(t));
        }
    }
    child.add_space(4.0);
    if child.add_sized([240.0, 26.0], egui::Button::new("Empty canvas")).clicked() {
        requests.push(UiEditorRequest::Add(UiPreset::Canvas));
    }
}
