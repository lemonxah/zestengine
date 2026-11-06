//! Audio volume authoring, kept separate from runtime region evaluation.
use super::*;
use zestengine_core::AudioVolume;

impl Inspect for AudioVolume {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = ctx.file_field(ui, "Environment", &mut self.profile,
            "audio/Cabin.audiofx", &[zestengine_assets::AUDIOFX_EXTENSION]);
        property_row(ui, "Global", &mut changed, |ui| ui.checkbox(&mut self.global, ""));
        property_row(ui, "Priority", &mut changed, |ui| ui.add(DragValue::new(&mut self.priority).speed(1.0)));
        if !self.global {
            axis_row(ui, "Half extents", &mut self.half_extents, &mut changed);
            for v in &mut self.half_extents { *v = v.max(0.01); }
        }
        ui.label(RichText::new("Highest priority wins. Equal priorities use scene order. Give local rooms higher priority than the global volume.").small().weak());
        ui.label(RichText::new("Audio Probe Volumes control bake coverage separately.").small().weak());
        changed
    }
}
impl Gizmo for AudioVolume {
    fn gizmos(&self) -> Vec<GizmoSpec> {
        if self.global { return Vec::new(); }
        vec![GizmoSpec::Box { center: glam::Vec3::ZERO, half_extents: self.half_extents.into(),
            object_anchored: true, blend: 0.0, color: egui::Color32::from_rgb(90, 190, 230) }]
    }
    fn apply_gizmo_edit(&mut self, edit: GizmoEdit) {
        if let GizmoEdit::Box { half_extents, .. } = edit { self.half_extents = half_extents.max(glam::Vec3::splat(0.01)).to_array(); }
    }
}
