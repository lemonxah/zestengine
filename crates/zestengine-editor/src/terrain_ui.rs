//! Inspector and Scene-view sculpting UI for the `Terrain` component: a
//! Unity-style brush that raises, lowers, sets, or smooths the heightfield.
//! The tool settings live in egui memory so the inspector panel and the
//! engine's Scene-view painter share one state (the same split the Mesh Cloth
//! pin painter uses).

use zestengine_core::{BrushFalloff, ObjectId, Terrain};
use egui::{DragValue, RichText, Ui};

use crate::components::{Gizmo, Inspect, InspectCtx};
use crate::icons::Icon;
use crate::theme;
use crate::widgets;

/// What the terrain brush does to the heights under it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TerrainBrushMode {
    /// Add height (hold or drag to keep raising).
    #[default]
    Raise,
    /// Subtract height (Shift while in Raise does this, and vice versa).
    Lower,
    /// Blend toward a fixed height (the brush's `height` value).
    SetHeight,
    /// Pull each vertex toward the average under the brush.
    Smooth,
}

impl TerrainBrushMode {
    pub const ALL: [TerrainBrushMode; 4] =
        [Self::Raise, Self::Lower, Self::SetHeight, Self::Smooth];

    pub fn label(self) -> &'static str {
        match self {
            Self::Raise => "Raise",
            Self::Lower => "Lower",
            Self::SetHeight => "Set height",
            Self::Smooth => "Smooth",
        }
    }
}

/// Terrain sculpting settings, shared between the inspector and the Scene view.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainPaintTool {
    /// The object being painted (the selection when painting started).
    pub object: Option<ObjectId>,
    pub active: bool,
    pub mode: TerrainBrushMode,
    pub falloff: BrushFalloff,
    /// Brush radius in world units.
    pub radius: f32,
    /// 0..1 raise/set/smooth amount per second.
    pub strength: f32,
    /// What Set height paints (and what "Flatten all" fills with), in metres.
    pub height: f32,
    /// Paint texture-layer weights instead of sculpting heights.
    pub texture_mode: bool,
    /// Which RGBA layer texture painting writes (0..3).
    pub texture_layer: u8,
    /// A stroke is in progress (one undo step).
    pub stroking: bool,
}

impl Default for TerrainPaintTool {
    fn default() -> Self {
        Self {
            object: None,
            active: false,
            mode: TerrainBrushMode::Raise,
            falloff: BrushFalloff::Smooth,
            radius: 4.0,
            strength: 0.5,
            height: 0.0,
            texture_mode: false,
            texture_layer: 0,
            stroking: false,
        }
    }
}

impl TerrainPaintTool {
    fn id() -> egui::Id {
        egui::Id::new("zestengine-terrain-paint-tool")
    }

    pub fn get(ctx: &egui::Context) -> Self {
        ctx.data(|d| d.get_temp(Self::id())).unwrap_or_default()
    }

    pub fn set(self, ctx: &egui::Context) {
        ctx.data_mut(|d| d.insert_temp(Self::id(), self));
    }
}

impl Gizmo for Terrain {}

impl Inspect for Terrain {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        changed |= ui
            .add(
                DragValue::new(&mut self.size)
                    .speed(0.5)
                    .range(1.0..=1_000_000.0)
                    .prefix("Size "),
            )
            .on_hover_text("World-space X/Z extent in meters")
            .changed();
        ui.label(format!(
            "Grid: {} x {} quads ({} vertices)",
            self.resolution,
            self.resolution,
            (self.resolution as usize + 1).pow(2)
        ));

        widgets::eyebrow(ui, "Heightmap");
        changed |= ctx.file_field(ui, "Heightmap", &mut self.heightmap, "import a greyscale image", &["png", "jpg", "jpeg", "tga", "bmp", "exr", "hdr", "gif", "webp"]);
        widgets::property_row(ui, "Height Scale", |ui| {
            changed |= ui
                .add(DragValue::new(&mut self.heightmap_scale).speed(0.1).range(0.0..=1_000_000.0).suffix(" m"))
                .on_hover_text("Metres the heightmap's 0..1 luminance spans")
                .changed();
        });
        widgets::property_row(ui, "Height Offset", |ui| {
            changed |= ui
                .add(DragValue::new(&mut self.heightmap_offset).speed(0.1).suffix(" m"))
                .on_hover_text("Metres added to the imported heights (0..1 black is at this height)")
                .changed();
        });
        if !self.heightmap.is_empty() && widgets::secondary_button(ui, Some(Icon::Reset), "Re-import heightmap")
            .on_hover_text("Reload the heightmap image into the height grid")
            .clicked()
        {
            self.heightmap_imported_path.clear();
            changed = true;
        }

        widgets::eyebrow(ui, "Sculpt");
        let egui_ctx = ui.ctx().clone();
        let mut tool = TerrainPaintTool::get(&egui_ctx);
        let before = tool;
        if tool.active {
            changed |= paint_panel(ui, self, &mut tool);
        } else if widgets::secondary_button(ui, Some(Icon::Target), "Paint terrain")
            .on_hover_text("Sculpt the heights in the Scene view (P)")
            .clicked()
        {
            tool.active = true;
        }
        if tool != before {
            tool.set(&egui_ctx);
        }
        changed
    }
}

fn note(ui: &mut Ui, text: &str) {
    ui.label(
        RichText::new(text)
            .size(theme::SIZE_META)
            .color(theme::TEXT_MUTED),
    );
}

/// The brush settings while sculpting, plus the Set-height controls.
fn paint_panel(ui: &mut Ui, terrain: &mut Terrain, tool: &mut TerrainPaintTool) -> bool {
    let mut changed = false;
    widgets::property_row(ui, "Mode", |ui| {
        ui.horizontal(|ui| {
            let sculpt = ui.selectable_label(!tool.texture_mode, "Sculpt");
            if sculpt.clicked() {
                tool.texture_mode = false;
            }
            let texture = ui.selectable_label(tool.texture_mode, "Texture");
            if texture.clicked() {
                tool.texture_mode = true;
            }
            changed |= sculpt.changed() || texture.changed();
        });
    });
    ui.add_space(2.0);
    if tool.texture_mode {
        widgets::property_row(ui, "Layer", |ui| {
            ui.horizontal(|ui| {
                for (idx, label) in ["R", "G", "B", "A"].into_iter().enumerate() {
                    changed |= ui
                        .selectable_value(&mut tool.texture_layer, idx as u8, label)
                        .changed();
                }
            });
        });
    } else {
    widgets::property_row(ui, "Brush", |ui| {
        egui::ComboBox::from_id_salt("zestengine-terrain-brush")
            .width(ui.available_width())
            .selected_text(tool.mode.label())
            .show_ui(ui, |ui| {
                for m in TerrainBrushMode::ALL {
                    changed |= ui.selectable_value(&mut tool.mode, m, m.label()).changed();
                }
            });
    });
    }
    widgets::property_row(ui, "Radius", |ui| {
        changed |= ui
            .add(
                DragValue::new(&mut tool.radius)
                    .speed(0.1)
                    .range(0.05..=1000.0)
                    .suffix(" m"),
            )
            .changed();
    });
    widgets::property_row(ui, "Strength", |ui| {
        changed |= ui
            .add(
                DragValue::new(&mut tool.strength)
                    .speed(0.01)
                    .range(0.0..=1.0),
            )
            .changed();
    });
    widgets::property_row(ui, "Falloff", |ui| {
        egui::ComboBox::from_id_salt("zestengine-terrain-falloff")
            .width(ui.available_width())
            .selected_text(tool.falloff.label())
            .show_ui(ui, |ui| {
                for f in BrushFalloff::ALL {
                    changed |= ui
                        .selectable_value(&mut tool.falloff, f, f.label())
                        .changed();
                }
            });
    });
    if !tool.texture_mode && tool.mode == TerrainBrushMode::SetHeight {
        widgets::property_row(ui, "Height", |ui| {
            changed |= ui
                .add(DragValue::new(&mut tool.height).speed(0.1).suffix(" m"))
                .changed();
        });
        if widgets::secondary_button(ui, Some(Icon::Reset), "Flatten all to height")
            .on_hover_text("Set every vertex to the Height value (one undo step)")
            .clicked()
        {
            terrain.heights.fill(tool.height);
            changed = true;
        }
    }
    if tool.texture_mode {
        note(
            ui,
            "Drag to paint texture layer weights   [ ] or Ctrl+Scroll size   P / Esc done",
        );
    } else {
        note(
            ui,
            "Drag to sculpt   Shift reverse   Alt smooth   [ ] or Ctrl+Scroll size   P / Esc done",
        );
    }
    changed
}
