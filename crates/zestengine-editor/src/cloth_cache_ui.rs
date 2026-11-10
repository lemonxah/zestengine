//! Live cloth takes use the same inspector and asset reference conventions as
//! other reusable simulation assets.
use egui::{Ui,DragValue};
use zestengine_core::MeshCloth;
use crate::components::InspectCtx;
pub(crate) fn inspector(ui:&mut Ui,ctx:&InspectCtx,c:&mut MeshCloth)->bool{
    let mut changed=false;
    ui.collapsing("Simulation recording",|ui|{
        changed|=ctx.file_field(ui,"Recording",&mut c.cache.asset,"choose a .clothcache path",&["clothcache"]);
        let mut path=crate::components::resolve_slot(ui,&c.cache.asset).unwrap_or_else(||c.cache.asset.clone());
        ui.horizontal(|ui|{ui.label("Save path");if ui.add(egui::TextEdit::singleline(&mut path).hint_text("Clothing/take.clothcache")).changed(){c.cache.asset=path;changed=true;}});
        changed|=ui.checkbox(&mut c.cache.playback,"Play recording").changed();
        changed|=ui.checkbox(&mut c.cache.looping,"Loop recording").changed();
        ui.horizontal(|ui|{ui.label("Speed");changed|=ui.add(DragValue::new(&mut c.cache.speed).speed(0.05).range(-16.0..=16.0)).changed();ui.label("Offset (s)");changed|=ui.add(DragValue::new(&mut c.cache.offset).speed(0.05)).changed();});
        if ui.button("Restart playback").clicked(){c.runtime.cache.playback_time=0.0;}
        ui.horizontal(|ui|{ui.label("Capture rate");changed|=ui.add(DragValue::new(&mut c.cache.frame_rate).range(1..=240)).changed();ui.label("Duration (s)");changed|=ui.add(DragValue::new(&mut c.cache.record_seconds).speed(0.1).range(0.01..=600.0)).changed();});
        if let Some(recording)=&c.runtime.cache.recorder{
            ui.label(format!("Recording: {} frames",recording.frame_count()));
            if ui.button("Stop and save recording").clicked(){c.runtime.cache.stop_requested=true;}
        }else if ui.add_enabled(!c.cache.asset.trim().is_empty()&&!c.runtime.hold&&!c.runtime.cache.unsaved,egui::Button::new("Record and save")).clicked(){
            c.cache.playback=false;c.runtime.cache.record_requested=true;changed=true;
        }
        if c.runtime.cache.record_requested{ui.label("Waiting for the next simulated cloth frame…");if ui.button("Cancel recording request").clicked(){c.runtime.cache.record_requested=false;}}
        if let Some(take)=&c.runtime.cache.recorded{
            ui.label(format!("Captured {} frames · {:.2} s · {} vertices",take.frames.len(),take.duration(),take.vertex_count));
            if c.runtime.cache.unsaved{ui.label("This take has not been saved.");}
            ui.horizontal(|ui|{
                if ui.button("Save take again").clicked(){c.runtime.cache.save_requested=true;}
                if ui.button("Discard captured take").clicked(){c.runtime.cache.recorded=None;c.runtime.cache.unsaved=false;c.runtime.cache.error.clear();}
            });
        }
        if let Some(data)=&c.runtime.cache.data{ui.label(format!("Loaded {} frames · {:.2} s",data.frames.len(),data.duration()));}
        if !c.runtime.cache.error.is_empty(){ui.colored_label(crate::theme::STATUS_WARN_FG,&c.runtime.cache.error);}
        ui.label("Records the final cloth pose after collisions and saves automatically. Playback follows this object's transform and replaces live cloth simulation. Recorded cloth does not react to new collisions.");
    });changed
}
