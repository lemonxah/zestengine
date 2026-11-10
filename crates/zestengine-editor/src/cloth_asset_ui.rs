//! Clothing asset authoring. Drafts remain in egui memory until Save succeeds.
use std::path::Path;
use egui::{Ui,DragValue};
use zestengine_assets::cloth_file::{ClothingAsset,load_clothing,save_clothing};
use zestengine_core::{ClothPanel,ClothSeam,ClothPanelEdge,ClothPanelSide,PinSource};
#[path="cloth_pattern_ui.rs"]
mod pattern_ui;
#[derive(Clone)]
struct Draft{edit_id:zestengine_core::ObjectId,asset:ClothingAsset,error:String,stamp:Option<std::time::SystemTime>,saved:String}
fn draft(asset:ClothingAsset,path:&Path)->Draft{let saved=ron::to_string(&asset).unwrap_or_default();Draft{edit_id:zestengine_core::ObjectId::new(),asset,error:String::new(),stamp:std::fs::metadata(path).and_then(|m|m.modified()).ok(),saved}}
fn number(ui:&mut Ui,label:&str,value:&mut f32,range:std::ops::RangeInclusive<f32>){ui.horizontal(|ui|{ui.label(label);ui.add(DragValue::new(value).speed(0.01).range(range));});}
fn edge(ui:&mut Ui,label:&str,edge:&mut ClothPanelEdge,names:&[String],panels:&[ClothPanel]){
    ui.push_id(label,|ui|{
        let old=edge.panel;
        ui.horizontal(|ui|{ui.label(label);egui::ComboBox::from_id_salt("panel").selected_text(names.get(edge.panel).map(String::as_str).unwrap_or("Missing panel")).show_ui(ui,|ui|{for(i,name)in names.iter().enumerate(){ui.selectable_value(&mut edge.panel,i,name);}});});
        if old!=edge.panel{edge.vertices.clear();}
        let Some(panel)=panels.get(edge.panel)else{return;};
        if panel.vertices.is_empty(){ui.horizontal(|ui|{egui::ComboBox::from_id_salt("side").selected_text(edge.side.label()).show_ui(ui,|ui|{for side in ClothPanelSide::ALL{if ui.selectable_value(&mut edge.side,side,side.label()).changed(){edge.vertices.clear();}}});if !edge.vertices.is_empty()&&ui.small_button("Use whole side").clicked(){edge.vertices.clear();}});}
        if !edge.vertices.is_empty(){ui.label(format!("Selected boundary: {} vertices",edge.vertices.len()));}
        ui.collapsing("Pick seam endpoints",|ui|pattern_ui::seam_ui(ui,panel,edge));
    });
}
pub(crate) fn file_ui(ui:&mut Ui,path:&Path)->bool{
    let id=egui::Id::new(("clothing-draft",path));
    let mut draft=match ui.data(|d|d.get_temp::<Draft>(id)){Some(d)=>d,None=>match load_clothing(path){Ok(asset)=>draft(asset,path),Err(e)=>{ui.colored_label(crate::theme::STATUS_WARN_FG,format!("Cannot open clothing: {e:#}"));return false;}}};
    let stamp=std::fs::metadata(path).and_then(|m|m.modified()).ok();
    if stamp!=draft.stamp{
        if ron::to_string(&draft.asset).unwrap_or_default()==draft.saved{if let Ok(asset)=load_clothing(path){draft=self::draft(asset,path);}}
        else{draft.error="The file changed while this draft was open. Reload to use that version, or Save garment to keep this draft.".into();}
    }
    ui.heading("Clothing");ui.label(path.display().to_string());ui.text_edit_singleline(&mut draft.asset.name);
    ui.label("Arrange panels around the avatar's bind pose, sew matching boundaries, then save and drag the garment into the scene. Mesh Cloth is added automatically.");
    let canvas_id=draft.edit_id.raw();let mut remove=None;let mut editable=None;
    for (i,panel) in draft.asset.pattern.panels.iter_mut().enumerate(){ui.push_id((canvas_id,i),|ui|{ui.collapsing(format!("Panel {}: {}",i+1,panel.name),|ui|{
        ui.text_edit_singleline(&mut panel.name);
        if panel.vertices.is_empty(){ui.horizontal(|ui|{ui.label("Size (m)");for v in &mut panel.size{ui.add(DragValue::new(v).speed(0.01).range(0.001..=100.0));}});ui.horizontal(|ui|{ui.label("Subdivisions");for v in &mut panel.segments{ui.add(DragValue::new(v).range(1..=256));}});}else{ui.label(format!("Custom pattern: {} vertices, {} triangles",panel.vertices.len(),panel.triangles.len()/3));}
        if panel.vertices.is_empty()&&ui.button("Convert to editable vertices").clicked(){editable=Some(i);}
        ui.collapsing("Panel shape",|ui|pattern_ui::panel_ui(ui,panel));
        ui.horizontal(|ui|{ui.label("Position");for v in &mut panel.translation{ui.add(DragValue::new(v).speed(0.01));}});ui.horizontal(|ui|{ui.label("Rotation (degrees)");for v in &mut panel.rotation{ui.add(DragValue::new(v).speed(1.0));}});
        if ui.button("Remove panel").clicked(){remove=Some(i);}
    });});}
    if let Some(i)=editable{if let Err(e)=draft.asset.pattern.make_panel_editable(i){draft.error=e;}}
    if let Some(i)=remove{match draft.asset.remove_panel(i){Ok(())=>draft.edit_id=zestengine_core::ObjectId::new(),Err(e)=>draft.error=format!("{e:#}")}}
    if ui.button("Add panel").clicked(){if let Err(e)=draft.asset.add_panel(ClothPanel{name:format!("Panel {}",draft.asset.pattern.panels.len()+1),..Default::default()}){draft.error=format!("{e:#}");}}
    ui.collapsing("Draw a new panel",|ui|{if let Some(mut panel)=ui.push_id(canvas_id,pattern_ui::outline_ui).inner{panel.name=format!("Outline panel {}",draft.asset.pattern.panels.len()+1);if let Err(e)=draft.asset.add_panel(panel){draft.error=format!("{e:#}");}}});
    ui.separator();ui.label("Sewing");let names=draft.asset.pattern.panels.iter().enumerate().map(|(i,p)|format!("{}: {}",i+1,p.name)).collect::<Vec<_>>();let mut remove=None;
    for (i,seam) in draft.asset.pattern.seams.iter_mut().enumerate(){ui.push_id(("seam",canvas_id,i),|ui|{edge(ui,"From",&mut seam.a,&names,&draft.asset.pattern.panels);edge(ui,"To",&mut seam.b,&names,&draft.asset.pattern.panels);ui.checkbox(&mut seam.reverse,"Reverse edge direction");number(ui,"Compliance (0 = firm seam)",&mut seam.compliance,0.0..=1.0);if ui.button("Remove seam").clicked(){remove=Some(i);}ui.separator();});}
    if let Some(i)=remove{draft.asset.pattern.seams.remove(i);}
    if ui.add_enabled(names.len()>1,egui::Button::new("Add seam")).clicked(){draft.asset.pattern.seams.push(ClothSeam::default());}
    ui.collapsing("Skin-weight transfer",|ui|{
        ui.label("Optional avatar model. Panels and body mesh must use the same bind-space coordinates. Weights and the selected skeleton are imported with this garment.");
        ui.text_edit_singleline(&mut draft.asset.skin_model);
        if let Some(p)=crate::components::asset_browse_button(ui,"cloth-body",&["gltf","glb","fbx","vrm"],&draft.asset.skin_model){draft.asset.skin_model=zestengine_core::asset_ref::reference(&zestengine_assets::cloth_file::clothing_root(path),&p);}
        ui.horizontal(|ui|{ui.label("Body mesh");ui.add(DragValue::new(&mut draft.asset.skin_mesh));ui.label("Skeleton");ui.add(DragValue::new(&mut draft.asset.skin_skeleton));});
        number(ui,"Maximum transfer distance (m)",&mut draft.asset.transfer_distance,0.001..=100.0);
        if ui.button("Validate garment and skin weights").clicked(){draft.error=match zestengine_assets::cloth_file::build_clothing_model(path,&draft.asset){Ok(scene)=>format!("Valid: {} vertices, {} skeleton(s)",scene.meshes[0].vertices.len(),scene.skeletons.len()),Err(e)=>format!("{e:#}")};}
    });
    ui.collapsing("Simulation defaults",|ui|{
        let settings=&mut draft.asset.settings;egui::ComboBox::from_id_salt("cloth-profile-pins").selected_text(settings.pin.label()).show_ui(ui,|ui|{for pin in PinSource::ALL{ui.selectable_value(&mut settings.pin,pin,pin.label());}});
        number(ui,"Pinned top share",&mut settings.pin_height,0.0..=1.0);number(ui,"Particle spacing",&mut settings.particle_spacing,0.0..=1.0);number(ui,"Frequency",&mut settings.frequency,10.0..=480.0);
        ui.horizontal(|ui|{ui.label("Iterations");ui.add(DragValue::new(&mut settings.iterations).range(1..=64));});ui.checkbox(&mut settings.self_collision,"Self collision");
        if let Some(f)=&mut settings.fabric{number(ui,"Density",&mut f.density,0.001..=100.0);number(ui,"Warp",&mut f.warp,0.0..=1e6);number(ui,"Weft",&mut f.weft,0.0..=1e6);number(ui,"Shear",&mut f.shear,0.0..=1e6);number(ui,"Fold stiffness",&mut f.bend,0.0..=10.0);number(ui,"Animation drive (Hz)",&mut f.animation_hz,0.0..=60.0);}
        ui.label("Paint and tune a placed garment using Mesh Cloth, then use Save settings to garment to reuse those maps and settings.");
    });
    if let Err(e)=draft.asset.refresh_pattern_paint(){draft.error=format!("{e:#}");}
    let mut saved=false;ui.horizontal(|ui|{if ui.button("Save garment").clicked(){match save_clothing(path,&draft.asset){Ok(())=>{draft.error.clear();draft.saved=ron::to_string(&draft.asset).unwrap_or_default();draft.stamp=std::fs::metadata(path).and_then(|m|m.modified()).ok();saved=true;},Err(e)=>draft.error=format!("{e:#}")}}if ui.button("Reload").clicked(){match load_clothing(path){Ok(asset)=>draft=self::draft(asset,path),Err(e)=>draft.error=format!("{e:#}")}}});
    if !draft.error.is_empty(){ui.label(&draft.error);}
    if let Ok(geometry)=draft.asset.pattern.assemble(){ui.label(format!("{} vertices · {} triangles · {} stitches",geometry.positions.len(),geometry.indices.len()/3,geometry.stitches.len()));}
    ui.data_mut(|d|d.insert_temp(id,draft));saved
}
