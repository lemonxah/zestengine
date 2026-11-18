//! Cloth recording/replay scheduling. Capture the final pose after shared
//! contacts, and feed playback through the ordinary CPU/GPU cloth upload path.
use std::{collections::HashMap,path::PathBuf,sync::Arc,time::{Duration,Instant,SystemTime}};
use zestengine_core::{MeshCloth,ClothCacheFrame,ClothCacheRecorder,ClothSimulationCache};
use zestengine_render::Vertex;
use crate::scene::SceneObject;

type ClothCacheEntry = (Option<SystemTime>, Option<Arc<ClothSimulationCache>>, String);

#[derive(Default)]
pub(crate) struct ClothCacheAssets {
    pub root:PathBuf,
    entries:HashMap<PathBuf, ClothCacheEntry>,
    next_poll:Option<Instant>,
}
impl ClothCacheAssets {
    pub fn sync(&mut self,objects:&[SceneObject]){
        let now=Instant::now();let poll=self.next_poll.is_none_or(|t|now>=t);
        if poll{self.next_poll=Some(now+Duration::from_millis(250));}
        let mut live=std::collections::HashSet::new();
        for mut component in objects.iter().flat_map(|o|o.components.iter_typed_mut::<MeshCloth>()){
            let c=&mut *component;
            if !c.cache.playback||c.cache.asset.trim().is_empty(){c.runtime.cache.data=None;continue;}
            let path=zestengine_core::asset_ref::resolve_path(&self.root,&c.cache.asset);live.insert(path.clone());
            if poll||!self.entries.contains_key(&path){
                let stamp=path.metadata().and_then(|m|m.modified()).ok();
                if self.entries.get(&path).is_none_or(|(old,_,_)|*old!=stamp){
                    let old=self.entries.get(&path).and_then(|(_,data,_)|data.clone());
                    let(data,error)=match zestengine_assets::cloth_cache_file::load_cloth_cache(&path){Ok(cache)=>(Some(Arc::new(cache)),String::new()),Err(e)=>(old,format!("Cannot load cloth recording: {e:#}"))};
                    self.entries.insert(path.clone(),(stamp,data,error));
                }
            }
            let(_,data,error)=self.entries.get(&path).unwrap();
            let replaced=match(&c.runtime.cache.data,data){(Some(a),Some(b))=>!Arc::ptr_eq(a,b),(None,None)=>false,_=>true};
            if replaced{c.runtime.cache.playback_time=0.0;}
            c.runtime.cache.data=data.clone();
            if !error.is_empty(){c.runtime.cache.error=error.clone();}else if replaced{c.runtime.cache.error.clear();}
        }
        self.entries.retain(|p,_|live.contains(p));
    }
    pub fn flush(&mut self,objects:&[SceneObject]){
        for mut component in objects.iter().flat_map(|o|o.components.iter_typed_mut::<MeshCloth>()){
            let c=&mut *component;
            if std::mem::take(&mut c.runtime.cache.stop_requested){
                c.runtime.cache.record_requested=false;finish_recording(c);
            }
            if !std::mem::take(&mut c.runtime.cache.save_requested){continue;}
            let Some(data)=&c.runtime.cache.recorded else{continue};
            if c.cache.asset.trim().is_empty(){c.runtime.cache.error="Choose a .clothcache path before saving the recording.".into();continue;}
            let path=zestengine_core::asset_ref::resolve_path(&self.root,&c.cache.asset);
            match zestengine_assets::cloth_cache_file::save_cloth_cache(&path,data){
                Ok(())=>{c.runtime.cache.error.clear();c.runtime.cache.unsaved=false;self.entries.remove(&path);},
                Err(e)=>c.runtime.cache.error=format!("Cannot save cloth recording: {e:#}"),
            }
        }
    }
}

/// Called once the bind mesh is known, before live simulation. Cached output
/// is kinematic: it is not modified again by garment or body constraints.
pub(crate) fn prepare(c:&mut MeshCloth,bind:&[Vertex],indices:&[u32],dt:f32){
    let state=&mut c.runtime.cache;
    if !c.cache.playback&&!state.record_requested&&state.recorder.is_none(){
        if state.playing{c.runtime.idle=true;}state.playing=false;return;
    }
    state.signature=zestengine_core::cloth_cache::cloth_cache_signature(&bind.iter().map(|v|v.position).collect::<Vec<_>>(),indices);
    let valid=state.data.as_ref().is_some_and(|data|data.matches(state.signature,bind.len()));
    if c.cache.playback&&state.data.is_some()&&!valid{state.error="Cloth recording belongs to a different mesh. Record again after reimporting or changing topology.".into();}
    let playing=c.cache.playback&&valid&&!state.record_requested&&state.recorder.is_none();
    if playing&&!state.playing{state.playback_time=0.0;}
    if state.playing&&!playing{c.runtime.idle=true;}
    state.playing=playing;
    if playing&&dt.is_finite(){state.playback_time+=dt.max(0.0)as f64;}
}

pub(crate) fn apply_playback(c:&mut MeshCloth,vertices:&mut[Vertex])->bool{
    if !c.runtime.cache.playing{return false;}
    let speed=if c.cache.speed.is_finite(){c.cache.speed.clamp(-16.0,16.0)}else{1.0};
    let offset=if c.cache.offset.is_finite(){c.cache.offset}else{0.0};
    let time=c.runtime.cache.playback_time*speed as f64+offset as f64;
    let Some(frame)=c.runtime.cache.data.as_ref().and_then(|data|data.sample(time,c.cache.looping))else{return false;};
    if frame.positions.len()!=vertices.len(){return false;}
    for((v,p),n)in vertices.iter_mut().zip(frame.positions).zip(frame.normals){v.position=p;v.normal=n;}true
}

fn finish_recording(c:&mut MeshCloth){
    if let Some(recorder)=c.runtime.cache.recorder.take(){match recorder.finish(){
        Ok(data)=>{c.runtime.cache.recorded=Some(Arc::new(data));c.runtime.cache.save_requested=true;c.runtime.cache.unsaved=true;},
        Err(error)=>c.runtime.cache.error=error,
    }}
}

pub(crate) fn capture(c:&mut MeshCloth,vertices:&[Vertex],dt:f32){
    let started=std::mem::take(&mut c.runtime.cache.record_requested);
    if started{
        c.runtime.cache.error.clear();c.runtime.cache.recording_time=0.0;
        if c.cache.asset.trim().is_empty(){c.runtime.cache.error="Choose a .clothcache path before recording.".into();}
        else{match ClothCacheRecorder::new(c.runtime.cache.signature,vertices.len(),c.cache.frame_rate,c.cache.record_seconds){
            Ok(recorder)=>{c.runtime.cache.recorder=Some(recorder);c.runtime.cache.recorded=None;c.runtime.cache.unsaved=false;},Err(error)=>c.runtime.cache.error=error,
        }}
    }
    if c.runtime.cache.recorder.as_ref().is_some_and(|r|!r.matches(c.runtime.cache.signature,vertices.len())){
        finish_recording(c);
        c.runtime.cache.error="The mesh changed during recording. The partial take uses the original mesh.".into();
        tracing::warn!("{}",c.runtime.cache.error);return;
    }
    if !started&&dt.is_finite(){c.runtime.cache.recording_time+=dt.max(0.0)as f64;}
    if let Some(recorder)=&mut c.runtime.cache.recorder{
        let pose=ClothCacheFrame{positions:vertices.iter().map(|v|v.position).collect(),normals:vertices.iter().map(|v|v.normal).collect()};
        let done=match recorder.capture(c.runtime.cache.recording_time,pose){Ok(done)=>done,Err(error)=>{c.runtime.cache.error=error;true}};
        if done{finish_recording(c);}
    }
    if std::mem::take(&mut c.runtime.cache.stop_requested){finish_recording(c);}
}

impl crate::scene::LoadedScene {
    pub fn set_cloth_cache_root(&mut self,root:&std::path::Path){
        if self.cloth_cache_assets.root!=root{self.cloth_cache_assets.root=root.to_owned();}
    }

    /// Finish partial takes before an editor/session tears down its live scene.
    /// A failed save keeps the captured take available for another save attempt.
    pub fn finish_cloth_recordings(&mut self)->Result<(),String>{
        for mut value in self.objects.iter().flat_map(|o|o.components.iter_typed_mut::<MeshCloth>()){
            let c = &mut *value;
            finish_recording(c);
            if c.runtime.cache.unsaved{c.runtime.cache.save_requested=true;}
        }
        self.cloth_cache_assets.flush(&self.objects);
        let errors=self.objects.iter().flat_map(|o|o.components.iter_typed::<MeshCloth>())
            .filter(|c|c.runtime.cache.unsaved&&!c.runtime.cache.error.is_empty()).map(|c|c.runtime.cache.error.clone()).collect::<Vec<_>>();
        if errors.is_empty(){Ok(())}else{Err(errors.join("\n"))}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat4,Vec3};
    use zestengine_core::{MeshClothSim,ObjectId};
    fn vertices(z:f32)->Vec<Vertex>{[[0.0,0.0,z],[1.0,0.0,z],[0.0,1.0,z]].into_iter().map(|position|Vertex{position,normal:[0.0,0.0,1.0],..Default::default()}).collect()}
    fn cloth(o:&SceneObject)->zestengine_core::ecs::ComponentRef<'_, MeshCloth>{o.components.get_typed::<MeshCloth>().unwrap()}
    fn cloth_mut(o:&mut SceneObject)->zestengine_core::ecs::ComponentMut<'_, MeshCloth>{o.components.get_typed_mut::<MeshCloth>().unwrap()}
    struct Temp(PathBuf);
    impl Drop for Temp{fn drop(&mut self){let _=std::fs::remove_dir_all(&self.0);}}
    fn temp()->Temp{let p=std::env::temp_dir().join(format!("cloth-cache-scene-{}",ObjectId::new()));std::fs::create_dir_all(&p).unwrap();Temp(p)}
    #[test]
    fn capture_uses_current_frame_dt_and_rejects_a_same_count_mesh_change(){
        let mut c=MeshCloth::default();c.cache.asset="take.clothcache".into();c.cache.frame_rate=10;c.cache.record_seconds=0.5;c.runtime.cache.record_requested=true;
        let bind=vertices(0.0);let mut elapsed=0.0;
        for dt in [0.016,0.07,0.17,0.06,0.27]{prepare(&mut c,&bind,&[0,1,2],dt);if c.runtime.cache.recorder.is_some(){elapsed+=dt;}capture(&mut c,&vertices(elapsed*2.0),dt);}
        let take=c.runtime.cache.recorded.clone().unwrap();assert!(c.runtime.cache.save_requested&&c.runtime.cache.unsaved);
        for(i,f)in take.frames.iter().enumerate(){assert!((f.positions[0][2]-i as f32*0.2).abs()<1e-5);}
        c.runtime.cache.data=Some(take);c.cache.playback=true;c.cache.speed=0.0;c.cache.offset=0.35;
        prepare(&mut c,&bind,&[0,1,2],0.1);let mut output=vertices(900.0);assert!(apply_playback(&mut c,&mut output));assert!((output[0].position[2]-0.7).abs()<1e-5);
        prepare(&mut c,&vertices(0.1),&[0,1,2],0.1);assert!(!c.runtime.cache.playing);assert!(c.runtime.cache.error.contains("different mesh"));assert!(c.runtime.idle);
        assert!(!apply_playback(&mut c,&mut output));
    }
    #[test]
    fn reimport_during_capture_preserves_only_frames_of_the_original_mesh(){
        let mut c=MeshCloth::default();c.cache.asset="take.clothcache".into();c.runtime.cache.record_requested=true;
        let bind=vertices(0.0);prepare(&mut c,&bind,&[0,1,2],0.0);let signature=c.runtime.cache.signature;
        capture(&mut c,&vertices(0.0),0.0);capture(&mut c,&vertices(0.1),0.1);
        let count=c.runtime.cache.recorder.as_ref().unwrap().frame_count();
        prepare(&mut c,&vertices(0.2),&[0,1,2],0.1);capture(&mut c,&vertices(100.0),0.1);
        assert!(c.runtime.cache.recorder.is_none());assert!(c.runtime.cache.unsaved&&c.runtime.cache.save_requested);
        let take=c.runtime.cache.recorded.as_ref().unwrap();assert_eq!(take.mesh_signature,signature);assert_eq!(take.frames.len(),count);
        assert!(take.frames.iter().all(|f|f.positions.iter().all(|p|p[2]<=0.10001)),"a reimported pose cannot enter the original take");
    }
    #[test]
    fn final_shared_contacts_are_recorded_and_saved_takes_load_for_playback(){
        let dir=temp();let mut assets=ClothCacheAssets{root:dir.0.clone(),..Default::default()};let mut objects=Vec::new();let mut work=Vec::new();
        for i in 0..2{
            let bind=vertices(i as f32*0.01);let mut c=MeshCloth{thickness:0.04,..Default::default()};
            c.cache.asset=format!("take-{i}.clothcache");c.cache.record_seconds=0.01;c.runtime.cache.record_requested=true;
            c.runtime.sim=MeshClothSim::build(&bind.iter().map(|v|Vec3::from(v.position)).collect::<Vec<_>>(),&[0,1,2],&[if i==0{1.0}else{0.0};3],0.0,Mat4::IDENTITY);
            c.runtime.sim.as_mut().unwrap().reset(&bind.iter().map(|v|Vec3::from(v.position)).collect::<Vec<_>>(),Mat4::IDENTITY);
            prepare(&mut c,&bind,&[0,1,2],0.0);
            let mut object=SceneObject::new_empty(ObjectId::new(),"cloth");object.components.push(Box::new(c));objects.push(object);
            work.push(crate::cloth_contacts::ClothRenderWork{mesh:0,object:i,slot:0,handle:zestengine_render::MeshHandle::NONE,animated:bind,world:Mat4::IDENTITY,gpu:false});
        }
        crate::cloth_contacts::finish(&objects,work,None,1.0/60.0,&mut Vec::new());assets.flush(&objects);
        let recorded=cloth(&objects[1]);let frame=&recorded.runtime.cache.recorded.as_ref().unwrap().frames[0];assert!(frame.positions.iter().all(|p|p[2]>=0.0399),"capture runs after shared contacts");drop(recorded);
        assert!(dir.0.join("take-1.clothcache").exists());assert!(!cloth(&objects[1]).runtime.cache.unsaved);
        cloth_mut(&mut objects[1]).cache.playback=true;assets.sync(&objects);
        let mut c=cloth_mut(&mut objects[1]);prepare(&mut c,&vertices(0.01),&[0,1,2],0.0);let mut output=vertices(-100.0);assert!(apply_playback(&mut c,&mut output)); drop(c);assert!(output.iter().all(|p|p.position[2]>=0.0399));
        let mut patch=Vec::new();assert_eq!(zestengine_render::pack_cloth_patch(&vertices(0.01),&output,1e-5,&mut patch),3,"cached poses use the ordinary GPU patch layout");
    }
    #[test]
    fn partial_take_survives_a_failed_save_and_can_be_retried(){
        let dir=temp();let mut scene=crate::scene::LoadedScene::empty();scene.cloth_cache_assets.root=dir.0.clone();std::fs::create_dir(dir.0.join("blocked.clothcache")).unwrap();
        let mut c=MeshCloth::default();c.cache.asset="blocked.clothcache".into();c.runtime.cache.record_requested=true;
        prepare(&mut c,&vertices(0.0),&[0,1,2],0.0);capture(&mut c,&vertices(0.0),0.0);capture(&mut c,&vertices(0.2),0.2);
        let mut object=SceneObject::new_empty(ObjectId::new(),"cloth");object.components.push(Box::new(c));scene.objects.push(object);
        assert!(scene.finish_cloth_recordings().is_err());assert!(cloth(&scene.objects[0]).runtime.cache.unsaved);let frames=cloth(&scene.objects[0]).runtime.cache.recorded.as_ref().unwrap().frames.len();assert!(frames>1);
        cloth_mut(&mut scene.objects[0]).cache.asset="saved.clothcache".into();scene.finish_cloth_recordings().unwrap();
        assert!(!cloth(&scene.objects[0]).runtime.cache.unsaved);assert_eq!(zestengine_assets::cloth_cache_file::load_cloth_cache(dir.0.join("saved.clothcache")).unwrap().frames.len(),frames);
    }
}
