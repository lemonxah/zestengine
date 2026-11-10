//! Shared mesh policy and bounds, using the engine's existing volume gizmo.
use super::*;
use crate::widgets;
use zestengine_core::WaterZone;
impl Inspect for WaterZone {
    fn inspector_ui(&mut self,ui:&mut Ui,_ctx:&InspectCtx)->bool {
        let mut changed=false;
        widgets::property_row(ui,"Half extents (X, Z)",|ui|for v in &mut self.half_extents{changed|=ui.add(DragValue::new(v).speed(1.0).range(0.25..=100000.0)).changed();});
        widgets::property_row(ui,"Smallest tile",|ui|changed|=ui.add(DragValue::new(&mut self.tile_size).speed(0.25).range(0.25..=4096.0).suffix(" m")).changed());
        widgets::property_row(ui,"Cells per tile edge",|ui|changed|=ui.add(DragValue::new(&mut self.tile_quads).range(2..=32)).changed());
        widgets::property_row(ui,"Detail distance",|ui|changed|=ui.add(DragValue::new(&mut self.lod_scale).speed(0.05).range(2.0..=8.0)).changed());
        widgets::property_row(ui,"Maximum LOD",|ui|changed|=ui.add(DragValue::new(&mut self.max_lod).range(0..=10)).changed());
        widgets::property_row(ui,"Far ocean extent",|ui|changed|=ui.add(DragValue::new(&mut self.far_distance).speed(100.0).range(0.0..=1000000.0).suffix(" m")).changed());
        widgets::property_row(ui,"Priority",|ui|changed|=ui.add(DragValue::new(&mut self.priority)).changed());
        ui.label(RichText::new("Water bodies inside this zone share adaptive tile boundaries. Higher priority wins when zones overlap. Each water body can also select a zone explicitly.").small().weak());
        changed
    }
}
impl Gizmo for WaterZone {
    fn gizmos(&self)->Vec<GizmoSpec>{vec![GizmoSpec::Box{center:glam::Vec3::ZERO,half_extents:glam::Vec3::new(self.half_extents[0],0.25,self.half_extents[1]),object_anchored:true,blend:0.0,color:egui::Color32::from_rgb(70,180,240)}]}
    fn apply_gizmo_edit(&mut self,edit:GizmoEdit){if let GizmoEdit::Box{half_extents,..}=edit{self.half_extents=[half_extents.x.max(0.25),half_extents.z.max(0.25)];}}
}
