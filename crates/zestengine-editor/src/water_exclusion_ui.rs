//! Water-specific dry-volume authoring using the existing box gizmo.
use super::*;
use zestengine_core::WaterExclusionVolume;

impl Inspect for WaterExclusionVolume {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        axis_row(ui, "Half extents", &mut self.half_extents, &mut changed);
        for v in &mut self.half_extents { *v = v.max(0.01); }
        let mut all = self.waters.is_empty();
        if ui.checkbox(&mut all, "All water bodies").changed() {
            changed = true;
            if all { self.waters.clear(); } else { self.waters.push(ObjectRef::NONE); }
        }
        if !all {
            let mut remove = None;
            for (i, water) in self.waters.iter_mut().enumerate() {
                ui.push_id(i, |ui| {
                    changed |= ctx.object_ref_with_hint(ui, "Water body", water, "(drag a water object here)");
                    if ui.small_button("Remove").clicked() { remove = Some(i); }
                });
            }
            if let Some(i) = remove { self.waters.remove(i); changed = true; }
            if ui.small_button("Add water body").clicked() { self.waters.push(ObjectRef::NONE); changed = true; }
        }
        ui.label(RichText::new("Keeps the box interior dry. Water beneath its floor remains. Parent the volume to a moving boat to follow it.").small().weak());
        changed
    }
}
impl Gizmo for WaterExclusionVolume {
    fn gizmos(&self) -> Vec<GizmoSpec> {
        vec![GizmoSpec::Box { center: glam::Vec3::ZERO, half_extents: self.half_extents.into(), object_anchored: true,
            blend: 0.0, color: egui::Color32::from_rgb(240, 170, 60) }]
    }
    fn apply_gizmo_edit(&mut self, edit: GizmoEdit) {
        if let GizmoEdit::Box { half_extents, .. } = edit { self.half_extents = half_extents.max(glam::Vec3::splat(0.01)).to_array(); }
    }
}
