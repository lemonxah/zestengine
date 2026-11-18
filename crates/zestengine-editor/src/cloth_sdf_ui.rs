//! Level-set authoring is an editor concern; only baked samples ship in scenes.
use std::{path::PathBuf,sync::{Arc,Mutex}};
use egui::{Ui,DragValue};
use glam::Vec3;
use zestengine_core::cloth_sdf::{ClothLevelSet,ClothSdf,ClothSdfShape};
use crate::components::InspectCtx;
#[derive(Clone)]
struct Bake{key:String,result:Arc<Mutex<Option<Result<ClothSdf,String>>>>}

pub(crate) fn inspector(ui:&mut Ui,ctx:&InspectCtx,settings:&mut ClothLevelSet)->bool{
    let id=ui.make_persistent_id("cloth-level-set-bake");let message_id=id.with("message");let mut changed=false;
    changed|=ctx.file_field(ui,"Closed proxy mesh",&mut settings.source,"drag a closed model here",&["gltf","glb","fbx","vrm"]);
    ui.horizontal(|ui|{ui.label("Mesh");changed|=ui.add(DragValue::new(&mut settings.mesh).range(0..=65535)).changed();ui.label("Resolution");changed|=ui.add(DragValue::new(&mut settings.resolution).range(8..=128)).changed();});
    ui.label("Proxy vertices are in the selected bone/object's local coordinates. The field follows that frame rigidly. Bake again after changing its source.");
    let path=crate::components::resolve_slot(ui,&settings.source).and_then(|rel|ui.data(|d|d.get_temp::<String>(egui::Id::new(crate::components::PROJECT_ROOT_KEY))).map(|root|PathBuf::from(root).join(rel)));
    let key=format!("{path:?}/{}/{}",settings.mesh,settings.resolution);
    let job=ui.data(|d|d.get_temp::<Bake>(id));let mut busy=job.is_some();
    if let Some(job)=job{
        let result=job.result.lock().unwrap_or_else(|e|e.into_inner()).take();
        if let Some(result)=result{
            busy=false;ui.data_mut(|d|d.remove::<Bake>(id));
            let message=if job.key!=key{"Source settings changed during baking; bake the new settings.".to_string()}else{match result{
                Ok(field)=>{settings.field=Some(field);changed=true;"Level set baked. Save the scene to keep it.".into()},Err(e)=>e,
            }};ui.data_mut(|d|d.insert_temp(message_id,message));
        }
    }
    ui.horizontal(|ui|{
        if ui.add_enabled(!busy&&path.is_some(),egui::Button::new("Bake level set")).clicked(){
            let path=path.unwrap();let mesh=settings.mesh;let resolution=settings.resolution;let result=Arc::new(Mutex::new(None));let output=result.clone();
            std::thread::spawn(move||{
                let baked=(||{
                    let model=zestengine_assets::load_model_with_meta(&path).map_err(|e|format!("Cannot load proxy: {e:#}"))?;
                    let mesh=model.meshes.get(mesh).ok_or_else(||"The source model has no mesh at that index".to_string())?;
                    let positions=mesh.vertices.iter().map(|v|Vec3::from(v.position)).collect::<Vec<_>>();ClothSdf::from_mesh(&positions,&mesh.indices,resolution)
                })();*output.lock().unwrap_or_else(|e|e.into_inner())=Some(baked);
            });
            ui.data_mut(|d|{d.insert_temp(id,Bake{key:key.clone(),result});d.insert_temp(message_id,"Baking closed proxy…".to_string());});busy=true;
        }
        if settings.field.is_some()&&ui.add_enabled(!busy,egui::Button::new("Clear field")).clicked(){settings.field=None;changed=true;}
    });
    if busy{ui.spinner();ui.ctx().request_repaint_after(std::time::Duration::from_millis(50));}
    if let Some(field)=&settings.field{let d=field.dimensions();ui.label(format!("{} × {} × {} samples · {:.1} KiB",d[0],d[1],d[2],field.bytes()as f32/1024.0));}else{ui.label("Bake a field to enable this collider.");}
    if let Some(message)=ui.data(|d|d.get_temp::<String>(message_id)){ui.label(message);}
    changed
}

/// Bounds plus zero-contours through three centre slices. Sampling is bounded
/// independently of bake resolution, so inspecting a dense field stays cheap.
pub(crate) fn lines(shape:&ClothSdfShape)->Vec<(Vec3,Vec3)>{
    let(lo,hi)=shape.field.bounds();let mut lines=Vec::new();let transform=shape.transform();
    let corner=|i:usize|Vec3::new(if i&1==0{lo.x}else{hi.x},if i&2==0{lo.y}else{hi.y},if i&4==0{lo.z}else{hi.z});
    for i in 0..8{for bit in [1,2,4]{if i&bit==0{lines.push((transform.transform_point3(corner(i)),transform.transform_point3(corner(i|bit))));}}}
    let center=(lo+hi)*0.5;
    for axis in 0..3{let u=(axis+1)%3;let v=(axis+2)%3;let dims=shape.field.dimensions();let nx=(dims[u]-1).min(32);let ny=(dims[v]-1).min(32);
        let point=|x:usize,y:usize|{let mut p=center;p[u]=lo[u]+(hi[u]-lo[u])*x as f32/nx as f32;p[v]=lo[v]+(hi[v]-lo[v])*y as f32/ny as f32;p};
        for y in 0..ny{for x in 0..nx{let p=[point(x,y),point(x+1,y),point(x+1,y+1),point(x,y+1)];let values=p.map(|p|shape.field.sample(p).0);let mut hits=Vec::new();
            for i in 0..4{let j=(i+1)%4;if (values[i]<0.0)!=(values[j]<0.0){hits.push(p[i].lerp(p[j],values[i]/(values[i]-values[j])));}}
            for pair in hits.as_chunks::<2>().0{lines.push((transform.transform_point3(pair[0]),transform.transform_point3(pair[1])));}
        }}
    }
    lines
}
