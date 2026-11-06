//! Inspectors for the 2D rig components (2D-7): `SpriteBone`, `SpriteSkin`,
//! `SpriteLibraryRef`, `SpriteResolver` and `IkSolver2D`. Plain property rows;
//! the Skinning Editor window itself lives in the engine crate
//! (`engine/src/skinning_editor.rs`) because it needs the project and the texture.
//! GUI not yet seen on screen.

use egui::{DragValue, RichText, Ui};
use zestengine_core::{IkSolver2D, IkSolverKind, SpriteBone, SpriteLibraryRef, SpriteResolver, SpriteSkin, sprite_swap_hash};

use crate::components::{EditorComponents, Gizmo, Inspect, InspectCtx};
use crate::{theme, widgets};

pub(crate) fn register(e: &mut EditorComponents) {
    e.register::<SpriteBone>();
    e.register::<SpriteSkin>();
    e.register::<SpriteLibraryRef>();
    e.register::<SpriteResolver>();
    e.register::<IkSolver2D>();
}

fn text_row(ui: &mut Ui, label: &str, value: &mut String, hint: &str, changed: &mut bool) {
    widgets::property_row(ui, label, |ui| {
        *changed |= ui.add(egui::TextEdit::singleline(value).hint_text(hint).desired_width(ui.available_width())).changed();
    });
}

fn note(ui: &mut Ui, text: &str) {
    ui.label(RichText::new(text).size(theme::SIZE_META).color(theme::TEXT_DIM));
}

impl Inspect for SpriteBone {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        widgets::property_row(ui, "Bone Id", |ui| {
            ui.label(if self.bone_id == 0 { "(match by name)".to_owned() } else { format!("{:016x}", self.bone_id) });
        });
        widgets::property_row(ui, "Limit Rotation", |ui| changed |= ui.checkbox(&mut self.limit_rotation, "").on_hover_text("IK keeps this bone's rotation inside the range").changed());
        if self.limit_rotation {
            widgets::property_row(ui, "Range (deg)", |ui| {
                ui.horizontal(|ui| {
                    changed |= ui.add(DragValue::new(&mut self.min_angle).speed(0.5).range(-360.0..=360.0)).changed();
                    changed |= ui.add(DragValue::new(&mut self.max_angle).speed(0.5).range(-360.0..=360.0)).changed();
                });
            });
        }
        changed
    }
}
impl Gizmo for SpriteBone {}

impl Inspect for SpriteSkin {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        text_row(ui, "Rig", &mut self.rig, "rigs/hero.rig2d", &mut changed);
        changed |= ctx.object_ref(ui, "Root Bone", &mut self.root);
        widgets::property_row(ui, "Auto Rebind", |ui| changed |= ui.checkbox(&mut self.auto_rebind, "").on_hover_text("Find unset bones among the root's children by bone id, then name").changed());
        widgets::property_row(ui, "Always Update", |ui| changed |= ui.checkbox(&mut self.always_update, "").on_hover_text("Skin even when the sprite's layer is not visible").changed());
        let set = self.bones.iter().filter(|b| b.is_set()).count();
        note(ui, &format!("{set} of {} bones set explicitly; the rest bind at runtime.", self.bones.len()));
        if ui.small_button("Clear Bones").on_hover_text("Forget the explicit bone list and rebind automatically").clicked() {
            self.bones.clear();
            changed = true;
        }
        note(ui, "Needs a Sprite Renderer on this object whose sprite has a part in the rig. The sprite draws from its own texture, not an atlas page.");
        changed
    }
}
impl Gizmo for SpriteSkin {}

impl Inspect for SpriteLibraryRef {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        text_row(ui, "Library", &mut self.library, "art/hero.spritelib", &mut changed);
        note(ui, "Resolvers on this object and below swap from this library.");
        changed
    }
}
impl Gizmo for SpriteLibraryRef {}

impl Inspect for SpriteResolver {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        let (mut cat, mut label) = (self.category.clone(), self.label.clone());
        let mut edited = false;
        text_row(ui, "Category", &mut cat, "Head", &mut edited);
        text_row(ui, "Label", &mut label, "Happy", &mut edited);
        if edited {
            self.set_entry(&cat, &label);
            changed = true;
        }
        widgets::property_row(ui, "Hash", |ui| {
            ui.label(RichText::new(format!("{}", self.hash)).monospace());
            if !self.category.is_empty() && sprite_swap_hash(&self.category, &self.label) != self.hash {
                ui.label(RichText::new("animated").size(theme::SIZE_META).color(theme::TEXT_DIM));
            }
        });
        note(ui, "A Sprite Swap animation track writes the hash; the Sprite Renderer on this object shows the matching library entry.");
        changed
    }
}
impl Gizmo for SpriteResolver {}

impl Inspect for IkSolver2D {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        widgets::property_row(ui, "Solver", |ui| {
            egui::ComboBox::from_id_salt("zestengine-ik2d-solver").selected_text(self.solver.label()).show_ui(ui, |ui| {
                for s in IkSolverKind::ALL {
                    if ui.selectable_label(self.solver == s, s.label()).clicked() && self.solver != s {
                        self.solver = s;
                        changed = true;
                    }
                }
            });
        });
        changed |= ctx.object_ref(ui, "Tip", &mut self.tip);
        changed |= ctx.object_ref(ui, "Target", &mut self.target);
        if self.solver != IkSolverKind::Limb {
            widgets::property_row(ui, "Chain Length", |ui| {
                changed |= ui.add(DragValue::new(&mut self.chain_length).range(1..=32)).changed();
            });
            widgets::property_row(ui, "Iterations", |ui| {
                changed |= ui.add(DragValue::new(&mut self.iterations).range(1..=256)).changed();
            });
            widgets::property_row(ui, "Tolerance", |ui| {
                changed |= ui.add(DragValue::new(&mut self.tolerance).speed(0.0005).range(0.00001..=1.0).max_decimals(5)).changed();
            });
        } else {
            widgets::property_row(ui, "Flip", |ui| changed |= ui.checkbox(&mut self.flip, "").on_hover_text("Bend the limb the other way").changed());
        }
        widgets::property_row(ui, "Weight", |ui| {
            changed |= widgets::slider(ui, egui::Slider::new(&mut self.weight, 0.0..=1.0)).changed();
        });
        note(ui, "The chain is the tip bone and its parents; the tip's position is the effector. Planar chains with positive uniform scale.");
        changed
    }
}
impl Gizmo for IkSolver2D {}
