//! Water terrain layer authoring; reuses water's existing outline gizmos.
use super::*;
use crate::widgets;
use zestengine_core::{WaterTerrainBrush,WaterTerrainBlend,WaterIsland};
impl Inspect for WaterTerrainBrush {
    fn inspector_ui(&mut self,ui:&mut Ui,ctx:&InspectCtx)->bool {
        let mut changed=ctx.object_ref_with_hint(ui,"Terrain",&mut self.terrain,"(all terrains, or drag a terrain here)");
        widgets::property_row(ui,"Blend",|ui|changed|=widgets::segmented(ui,&mut self.blend,&[(WaterTerrainBlend::Lower,"Lower"),(WaterTerrainBlend::Raise,"Raise"),(WaterTerrainBlend::Set,"Set")]).changed());
        for(label,value)in[("Bank falloff",&mut self.falloff),("Depth scale",&mut self.depth_scale)] {
            widgets::property_row(ui,label,|ui|changed|=ui.add(DragValue::new(value).speed(0.05).range(0.0..=1000.0)).changed());
        }
        widgets::property_row(ui,"Bed offset",|ui|changed|=ui.add(DragValue::new(&mut self.bed_offset).speed(0.05).suffix(" m")).changed());
        widgets::property_row(ui,"Layer order",|ui|changed|=ui.add(DragValue::new(&mut self.order)).changed());
        ui.label(RichText::new("Uses the Water component on this object. Original terrain heights stay editable; disable this brush to restore them.").small().weak());
        changed
    }
}
impl Gizmo for WaterTerrainBrush {}
impl Inspect for WaterIsland {
    fn inspector_ui(&mut self,ui:&mut Ui,ctx:&InspectCtx)->bool {
        let mut changed=ctx.object_ref_with_hint(ui,"Terrain",&mut self.terrain,"(all terrains, or drag a terrain here)");
        widgets::property_row(ui,"Size (X, Z)",|ui|for v in &mut self.size{changed|=ui.add(DragValue::new(v).speed(0.1).range(0.01..=100000.0)).changed();});
        widgets::property_row(ui,"Bank falloff",|ui|changed|=ui.add(DragValue::new(&mut self.falloff).speed(0.1).range(0.0..=10000.0)).changed());
        widgets::property_row(ui,"Layer order",|ui|changed|=ui.add(DragValue::new(&mut self.order)).changed());
        changed|=ui.checkbox(&mut self.smooth_shape,"Smooth outline").changed();
        ui.horizontal(|ui|{
            if ui.button("Rectangle").clicked(){self.shape.clear();changed=true;}
            for preset in zestengine_core::OutlinePreset::ALL {if ui.button(preset.label()).clicked(){self.shape=preset.points(self.size);changed=true;}}
        });
        let mut remove=None;
        for(i,p)in self.shape.iter_mut().enumerate(){ui.push_id(i,|ui|ui.horizontal(|ui|{
            ui.label(format!("Point {}",i+1));for v in p{changed|=ui.add(DragValue::new(v).speed(0.1)).changed();}
            if ui.small_button("Remove").clicked(){remove=Some(i);}
        }));}
        if let Some(i)=remove{self.shape.remove(i);changed=true;}
        ui.label(RichText::new("Move the island vertically to set its plateau. Island layers apply after water carving; drag the outline with the water gizmos.").small().weak());
        changed
    }
}
impl Gizmo for WaterIsland {
    fn gizmos(&self)->Vec<GizmoSpec>{let mut water=self.footprint();water.depth=0.0;water.gizmos()}
    fn apply_gizmo_edit(&mut self,edit:GizmoEdit){let mut water=self.footprint();water.apply_gizmo_edit(edit);self.size=water.size;self.shape=water.shape;}
}
