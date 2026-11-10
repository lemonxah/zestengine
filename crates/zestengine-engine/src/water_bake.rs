//! River/lake recording jobs. Scene capture is read-only; simulation and
//! compressed asset writing run off the editor/render thread.
use std::{path::PathBuf,sync::{Arc,mpsc,atomic::{AtomicBool,Ordering}}};
use zestengine_core::{ObjectId,Water,WaterSimulationCache,PipeSim,PipeParams,SimField,WaterCacheSettings};
use crate::scene::LoadedScene;

pub(crate) struct Job {id:ObjectId,asset:String,requested:String,cancel:Arc<AtomicBool>,receiver:mpsc::Receiver<Result<Arc<WaterSimulationCache>,String>>}

pub fn update(scene:&mut LoadedScene) {
    for o in &mut scene.objects {
        let Some(mut w)=o.components.get_typed_mut::<Water>()else{continue;};
        if std::mem::take(&mut w.cache_cancel_requested){
            w.cache_bake_requested=false;w.cache_status="Cancelled".into();
            for job in &scene.water_assets.cache_jobs{if job.id==o.id(){job.cancel.store(true,Ordering::Relaxed);}}
        }
    }
    let jobs=std::mem::take(&mut scene.water_assets.cache_jobs);
    for job in jobs {
        match job.receiver.try_recv(){
            Err(mpsc::TryRecvError::Empty)=>scene.water_assets.cache_jobs.push(job),
            result=>{
                let Some(mut water)=scene.objects.iter_mut().find(|o|o.id()==job.id).and_then(|o|o.components.get_typed_mut::<Water>())else{continue;};
                match result {
                    Ok(Ok(cache))=>{if water.simulation_cache.asset==job.requested {water.simulation_cache.asset=job.asset;water.cache_data=Some(cache);water.cache_status="Recording ready".into();}},
                    Ok(Err(error))=>{if error=="Cancelled"{water.cache_status=error;}else{tracing::error!("water cache: {error}");water.cache_status=format!("Bake failed: {error}");}},
                    Err(_)=>water.cache_status="Bake worker stopped".into(),
                }
            }
        }
    }
    let requests:Vec<_>=scene.objects.iter_mut().enumerate().filter_map(|(i,o)|{
        let mut water=o.components.get_typed_mut::<Water>()?;
        std::mem::take(&mut water.cache_bake_requested).then(||(i,o.id(),water.clone()))
    }).collect();
    for(i,id,water)in requests {
        if scene.water_assets.cache_jobs.iter().any(|j|j.id==id){continue;}
        if scene.water_assets.cache_jobs.len()>=2 {
            if let Some(mut w)=scene.objects[i].components.get_typed_mut::<Water>(){w.cache_bake_requested=true;w.cache_status="Queued recording…".into();}continue;
        }
        let result=(||->anyhow::Result<Job>{
            let settings=&water.simulation_cache;
            let seconds=settings.bake_seconds;let warmup=settings.warmup_seconds;
            anyhow::ensure!(seconds.is_finite() && (0.1..=120.0).contains(&seconds) && warmup.is_finite() && (0.0..=120.0).contains(&warmup),"invalid bake duration");
            let n=zestengine_core::sim_resolution(settings.resolution);let fps=settings.frame_rate.clamp(1,60);
            let frames=(seconds*fps as f32).ceil()as usize+1;
            anyhow::ensure!(frames*n as usize*n as usize*16<=zestengine_core::water_cache::WATER_CACHE_MAX_BYTES,"recording exceeds 128 MiB; lower resolution, duration or frame rate");
            let(lo,hi)=water.extent().ok_or_else(||anyhow::anyhow!("recording needs a finite river, lake or pool"))?;
            let size=(hi-lo).max_element().max(1.0)*1.2;let cell=size/n as f32;
            let window=zestengine_core::SimWindow{origin:(((lo+hi)*0.5)/cell-glam::Vec2::splat(n as f32*0.5)).floor().as_ivec2(),n,cell,closed:false};
            let field=crate::water_walls::recording_field(scene,i,window).ok_or_else(||anyhow::anyhow!("water was removed"))?;
            let asset=if settings.asset.trim().is_empty(){format!("water/Baked-{id}.watercache")}else{settings.asset.clone()};
            let path=zestengine_core::asset_ref::resolve_path(&scene.water_assets.root,&asset);
            let params=PipeParams::for_water(&water);let settings=settings.clone();let cancel=Arc::new(AtomicBool::new(false));let(receiver,worker)=launch(field,params,settings,path,cancel.clone())?;
            drop(worker);Ok(Job{id,asset,requested:water.simulation_cache.asset.clone(),cancel,receiver})
        })();
        if let Some(mut water)=scene.objects[i].components.get_typed_mut::<Water>() {
            match result {Ok(job)=>{water.cache_status="Baking recording…".into();scene.water_assets.cache_jobs.push(job);},Err(e)=>{water.cache_status=format!("Bake failed: {e:#}");tracing::error!("water cache: {e:#}");}}
        }
    }
}

fn launch(field:SimField,params:PipeParams,settings:WaterCacheSettings,path:PathBuf,cancel:Arc<AtomicBool>)->anyhow::Result<(mpsc::Receiver<Result<Arc<WaterSimulationCache>,String>>,std::thread::JoinHandle<()>)> {
    let(tx,rx)=mpsc::channel();let thread=std::thread::Builder::new().name("water-cache-bake".into()).spawn(move||{
        let result=record_cancellable(&field,&params,&settings,&cancel).and_then(|cache|{
            if cancel.load(Ordering::Relaxed){return Err("Cancelled".into());}
            zestengine_assets::water_cache_file::save_water_cache(&path,&cache).map_err(|e|format!("{e:#}"))?;Ok(Arc::new(cache))
        });let _=tx.send(result);
    })?;Ok((rx,thread))
}

pub fn record(field:&SimField,params:&PipeParams,settings:&WaterCacheSettings)->Result<WaterSimulationCache,String> {
    record_cancellable(field,params,settings,&AtomicBool::new(false))
}

fn record_cancellable(field:&SimField,params:&PipeParams,settings:&WaterCacheSettings,cancel:&AtomicBool)->Result<WaterSimulationCache,String> {
    let fps=settings.frame_rate.clamp(1,60)as f32;
    if !settings.bake_seconds.is_finite() || !(0.1..=120.0).contains(&settings.bake_seconds) || !settings.warmup_seconds.is_finite() || !(0.0..=120.0).contains(&settings.warmup_seconds){return Err("invalid recording duration".into());}
    let frames=(settings.bake_seconds*fps).ceil()as usize+1;let n=field.window.n as usize;
    if !(8..=512).contains(&n) || field.cells.len()!=n*n || !field.window.cell.is_finite() || field.window.cell<=0.0{return Err("invalid recording field".into());}
    if frames*n*n*16>zestengine_core::water_cache::WATER_CACHE_MAX_BYTES{return Err("recording exceeds playback budget".into());}
    let mut sim=PipeSim::new(field);
    // Fixed, conservative ticks prevent PipeSim's safety clamp from silently
    // shortening the recording clock on narrow/high-speed channels.
    let max_speed=field.cells.iter().map(|c|glam::Vec2::new(c[1],c[2]).length()).fold(0.0f32,f32::max)+8.0;
    let tick=(1.0/120.0f32).min(zestengine_core::max_sim_dt(field.window.cell,field.max_depth+20.0,max_speed)*0.5).max(1e-5);
    let advance=|sim:&mut PipeSim,seconds:f32|->Result<(),String>{let steps=(seconds/tick).ceil().max(1.0)as u32;let dt=seconds/steps as f32;for _ in 0..steps{if cancel.load(Ordering::Relaxed){return Err("Cancelled".into());}sim.step(dt,params,&[]);}Ok(())};
    advance(&mut sim,settings.warmup_seconds)?;
    let mut output=Vec::with_capacity(frames);
    for i in 0..frames {
        if i>0{advance(&mut sim,1.0/fps)?;}
        let(tex,back)=sim.output();let samples=tex.iter().zip(back).map(|(a,b)|{
            let foam=((a[1]*a[1]+a[2]*a[2]).sqrt()*0.5+b[1].hypot(b[2])*0.04-0.08).clamp(0.0,1.0);
            [a[0],b[1],b[2],foam]
        }).collect();output.push(samples);
    }
    let cache=WaterSimulationCache{n:n as u32,origin:(field.window.lo()+glam::Vec2::splat(field.window.cell*0.5)).to_array(),cell:field.window.cell,frame_rate:fps,frames:output};cache.validate()?;Ok(cache)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn background_bake_saves_a_reloadable_asset_and_cancellation_is_observed(){
        let root=std::env::temp_dir().join(format!("water-recording-{}",ObjectId::new()));
        let mut scene=LoadedScene::empty();scene.water_assets.root=root.clone();
        let mut object=crate::scene::SceneObject::new_empty(ObjectId::new(),"Recorded river");
        object.components.push(Box::new(Water{size:[8.0;2],cache_bake_requested:true,
            simulation_cache:WaterCacheSettings{bake_seconds:0.1,warmup_seconds:0.0,resolution:32,frame_rate:10,..Default::default()},..Default::default()}));
        scene.objects.push(object);update(&mut scene);assert_eq!(scene.water_assets.cache_jobs.len(),1);
        let deadline=std::time::Instant::now()+std::time::Duration::from_secs(10);
        while !scene.water_assets.cache_jobs.is_empty() && std::time::Instant::now()<deadline{std::thread::sleep(std::time::Duration::from_millis(2));update(&mut scene);}
        let water=scene.objects[0].components.get_typed_at::<Water>(0).unwrap();
        assert_eq!(water.cache_status,"Recording ready");let cache=zestengine_assets::water_cache_file::load_water_cache(root.join(&water.simulation_cache.asset)).unwrap();assert_eq!(cache.frames.len(),2);
        let field=SimField::build(&water,zestengine_core::choose_window(&water,glam::Vec2::ZERO,None),None);
        assert_eq!(record_cancellable(&field,&PipeParams::for_water(&water),&water.simulation_cache,&AtomicBool::new(true)).unwrap_err(),"Cancelled"); drop(water);
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn recorded_current_keeps_its_obstacle_response_without_double_counting_live_water(){
        let mut water=Water{size:[10.0;2],depth:1.0,waves:vec![],current:[1.0,0.0,0.0],ripple_resolution:32,ripple_area:20.0,..Default::default()};
        let window=zestengine_core::choose_window(&water,glam::Vec2::ZERO,None);
        let mut blocked=vec![false;32*32];for z in 14..18{for x in 14..18{blocked[z*32+x]=true;}}
        let field=SimField::build(&water,window,Some(&blocked));
        let settings=WaterCacheSettings{bake_seconds:0.5,warmup_seconds:1.0,frame_rate:10,looping:false,..Default::default()};
        let cache=record(&field,&PipeParams::for_water(&water),&settings).unwrap();
        assert!(cache.frames[0].iter().any(|v|v[0]>0.005));assert!(cache.frames[0].iter().any(|v|v[0]< -0.005));
        let again=record(&field,&PipeParams::for_water(&water),&settings).unwrap();assert_eq!(cache.frames,again.frames);
        water.cache_data=Some(Arc::new(cache));water.simulation_cache=settings;
        let overlay=SimField::build(&water,window,Some(&blocked));let mut live=PipeSim::new(&overlay);let params=PipeParams::for_water(&water);
        for _ in 0..120{live.step(1.0/120.0,&params,&[]);}
        assert!(live.output().0.iter().all(|v|v[0].abs()<1e-5),"cached static wakes must not be generated twice");
        let push=zestengine_core::WaterDisplacer::from_push(&zestengine_core::WaterPush{at:glam::Vec2::new(-3.0,0.0),radius:0.7,strength:4.0});
        live.step(1.0/60.0,&params,&[push]);assert!(live.output().0.iter().any(|v|v[0].abs()>0.001),"live impacts still work over a recording");
    }
}
