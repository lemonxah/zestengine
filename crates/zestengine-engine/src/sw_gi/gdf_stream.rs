//! One in-flight geometry/domain build. A caller may keep initial/edited
//! geometry asynchronous when the renderer has complete mesh-detail fallback.
//! While a camera build runs, the previous field and full-range mesh fallback
//! remain valid; camera motion never waits for this worker.
use super::{Gdf,SdfInstance,gdf_cache::Cache,gdf_domain::Domain};

struct Job {geometry:u64,key:u64,generation:u64,worker:std::thread::JoinHandle<Cache>}
#[derive(Default)]
pub(crate) struct Stream {
    cache:Cache,
    geometry:Option<u64>,
    ready:Option<u64>,
    job:Option<Job>,
    generation:u64,
}
impl Stream {
    pub fn pending(&self)->bool {self.job.is_some()}
    pub fn clear(&mut self) {
        self.generation=self.generation.wrapping_add(1);
        self.cache=Cache::default();self.geometry=None;self.ready=None;
        // Keep ownership until completion; rapid scene switches must not spawn
        // arbitrarily many detached workers. The geometry key rejects stale data.
        if self.job.as_ref().is_some_and(|j|j.worker.is_finished()) {
            let _=self.job.take().unwrap().worker.join();
        }
    }
    /// Whole-world fallback cannot publish a camera-local job left over from
    /// the previous coverage policy. Retain its handle, invalidate its result,
    /// and establish the required coarse coverage before dropping mesh detail.
    pub fn request_immediate(&mut self,instances:&[SdfInstance],domain:Domain,geometry:u64,key:u64)->(u64,&Gdf) {
        if self.geometry!=Some(geometry) || self.ready!=Some(key) {
            self.generation=self.generation.wrapping_add(1);
            self.cache.update(instances,domain.min,domain.max,domain.dims);
            self.geometry=Some(geometry);self.ready=Some(key);
        }
        if self.job.as_ref().is_some_and(|j|j.worker.is_finished()) {let _=self.job.take().unwrap().worker.join();}
        (key,self.cache.field().unwrap())
    }
    #[cfg(test)]
    pub fn request(&mut self,instances:&[SdfInstance],domain:Domain,geometry:u64,key:u64)->Option<(u64,&Gdf)> {
        self.request_scheduled(instances,domain,geometry,key,false,true)
    }
    /// The caller shares one worker budget across all levels. Outer levels may
    /// build their initial snapshot asynchronously because mesh detail already
    /// covers that geometry while they are absent.
    pub fn request_scheduled(&mut self,instances:&[SdfInstance],domain:Domain,geometry:u64,key:u64,
        async_geometry:bool,allow_start:bool)->Option<(u64,&Gdf)> {
        let mut completed=false;
        if self.job.as_ref().is_some_and(|j|j.worker.is_finished()) {
            let job=self.job.take().unwrap();
            match job.worker.join() {
                Ok(cache) if job.geometry==geometry && job.generation==self.generation => {
                    self.cache=cache;self.geometry=Some(geometry);self.ready=Some(job.key);completed=true;
                }
                Ok(_)=>{}, // Geometry changed while the old domain was building.
                Err(_)=>{
                    tracing::warn!("Flux camera distance-field worker panicked");
                    if self.geometry==Some(job.geometry) && self.ready.is_none() {self.geometry=None;}
                }
            }
        }
        if self.geometry!=Some(geometry) && !async_geometry {
            self.cache.update(instances,domain.min,domain.max,domain.dims);
            self.geometry=Some(geometry);self.ready=Some(key);completed=true;
        } else if (self.ready!=Some(key) || self.geometry!=Some(geometry)) && !completed && self.job.is_none() && allow_start {
            let mut cache=std::mem::take(&mut self.cache);
            let instances=instances.to_vec();
            self.ready=None;
            match std::thread::Builder::new().name("flux-field-scroll".into()).spawn(move || {
                super::lower_compute_priority();
                cache.update(&instances,domain.min,domain.max,domain.dims);cache
            }) {
                Ok(worker)=>self.job=Some(Job {geometry,key,generation:self.generation,worker}),
                Err(error)=>{self.geometry=None;tracing::warn!("Flux camera distance-field worker: {error}");}
            }
        }
        // Publish a completed intermediate domain even if the camera moved
        // again. It is valid geometry, and starvation during travel is worse
        // than a one-build-old acceleration region. Queue the latest next tick.
        if self.geometry==Some(geometry) && (completed || self.ready==Some(key)) {self.ready.zip(self.cache.field())} else {None}
    }
}

#[cfg(test)] mod tests {
    use super::*;
    use glam::Vec3;
    fn domain(x:f32)->Domain {Domain {min:Vec3::splat(-4.0)+Vec3::X*x,max:Vec3::splat(4.0)+Vec3::X*x,dims:[17;3]}}
    fn wait(stream:&Stream) {
        let start=std::time::Instant::now();
        while stream.job.as_ref().is_some_and(|j|!j.worker.is_finished()) {
            assert!(start.elapsed().as_secs()<10,"field worker did not finish");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
    #[test] fn scrolling_publishes_completed_domains_without_starving_or_waiting() {
        let mut stream=Stream::default();
        let (key,field)=stream.request(&[],domain(0.0),1,10).unwrap();
        assert_eq!(key,10);assert_eq!(field.min,domain(0.0).min);
        assert!(stream.request(&[],domain(1.0),1,11).is_none());
        assert!(stream.pending());wait(&stream);
        let (key,field)=stream.request(&[],domain(2.0),1,12).unwrap();
        assert_eq!(key,11);assert_eq!(field.min,domain(1.0).min);
        assert!(stream.request(&[],domain(2.0),1,12).is_none());wait(&stream);
        let (key,field)=stream.request(&[],domain(2.0),1,12).unwrap();
        assert_eq!(key,12);assert_eq!(field.min,domain(2.0).min);
        assert!(field.dist.iter().all(|&d|d==f32::MAX));
        assert!(!stream.pending());
    }
    #[test] fn outer_initial_build_obeys_shared_budget_and_rejects_stale_geometry() {
        let mut stream=Stream::default();
        assert!(stream.request_scheduled(&[],domain(0.0),1,10,true,false).is_none());
        assert!(!stream.pending(),"a denied budget must not spawn a worker");
        assert!(stream.request_scheduled(&[],domain(0.0),1,10,true,true).is_none());
        assert!(stream.pending());wait(&stream);
        // Reject the completed old scene, even if no new worker is allowed yet.
        assert!(stream.request_scheduled(&[],domain(20.0),2,20,true,false).is_none());
        assert!(!stream.pending());
        assert!(stream.request_scheduled(&[],domain(20.0),2,20,true,true).is_none());wait(&stream);
        let (key,field)=stream.request_scheduled(&[],domain(24.0),2,24,true,false).unwrap();
        assert_eq!(key,20);assert_eq!(field.min,domain(20.0).min);
        assert!(stream.request_scheduled(&[],domain(24.0),2,24,true,false).is_none());
        assert!(!stream.pending());
    }
    #[test] fn scene_changes_and_clear_do_not_publish_old_geometry() {
        let mut stream=Stream::default();
        stream.request(&[],domain(0.0),1,10).unwrap();
        stream.request(&[],domain(1.0),1,11);
        assert_eq!(stream.request(&[],domain(20.0),2,20).unwrap().0,20);
        wait(&stream);
        let (key,field)=stream.request(&[],domain(20.0),2,20).unwrap();
        assert_eq!(key,20);assert_eq!(field.min,domain(20.0).min);
        stream.request(&[],domain(21.0),2,21);
        stream.clear();wait(&stream);
        let (key,field)=stream.request(&[],domain(40.0),3,30).unwrap();
        assert_eq!(key,30);assert_eq!(field.min,domain(40.0).min);
        assert!(!stream.pending());
    }
    #[test] fn whole_world_fallback_and_clear_reject_old_domains_even_with_identical_geometry() {
        let mut stream=Stream::default();
        stream.request(&[],domain(0.0),1,10).unwrap();
        stream.request(&[],domain(1.0),1,11);
        assert_eq!(stream.request_immediate(&[],domain(40.0),1,40).1.min,domain(40.0).min);
        wait(&stream);
        let (key,field)=stream.request_scheduled(&[],domain(40.0),1,40,true,false).unwrap();
        assert_eq!(key,40);assert_eq!(field.min,domain(40.0).min);
        stream.request_scheduled(&[],domain(41.0),1,41,true,true);
        stream.clear();wait(&stream);
        assert!(stream.request_scheduled(&[],domain(50.0),1,50,true,false).is_none());
        assert!(!stream.pending(),"clear must retire the finished stale worker");
    }
}
