//! One mesh-field worker, with bounded main-thread geometry staging. Tokens
//! survive in-flight work so a reimport or scene switch cannot publish an old
//! field into a reused mesh slot.
use std::sync::Arc;
use glam::Vec3;
use zestengine_render::{sdf::SdfVolume,GiMeshSdfLevels};

pub(crate) struct Slot {key:Arc<()>,field:Option<GiMeshSdfLevels>}
impl Default for Slot {fn default()->Self {Self {key:Arc::new(()),field:None}}}
impl Slot {
    pub fn is_none(&self)->bool {self.field.is_none()}
    pub fn as_ref(&self)->Option<&Arc<SdfVolume>> {self.field.as_ref().map(|f|&f.detail)}
    pub fn levels(&self)->Option<&GiMeshSdfLevels> {self.field.as_ref()}
    pub fn reset(&mut self) {*self=Self::default();}
    pub fn set(&mut self,field:GiMeshSdfLevels) {self.field=Some(field);}
    pub fn key(&self)->&Arc<()> {&self.key}
    pub fn accept(&mut self,key:&Arc<()>,field:GiMeshSdfLevels)->bool {
        if !Arc::ptr_eq(&self.key,key) {return false;}
        self.field=Some(field);true
    }
}
struct Staging {key:Arc<()>,positions:Vec<Vec3>,indices:Vec<u32>}
struct Job {key:Arc<()>,worker:std::thread::JoinHandle<GiMeshSdfLevels>}
pub(crate) fn generate(positions:&[Vec3],indices:&[u32])->GiMeshSdfLevels {
    // Generate each level from triangles: filtering the fine signed field can
    // erase thin surfaces or their sign changes before residency even starts.
    GiMeshSdfLevels {
        coarse:Arc::new(zestengine_render::sdf::generate_mesh_sdf(positions,indices,8,0.1)),
        detail:Arc::new(zestengine_render::sdf::generate_mesh_sdf(positions,indices,32,0.1)),
    }
}
#[derive(Default)]
pub(crate) struct Stream {staging:Option<Staging>,job:Option<Job>}
impl Stream {
    pub fn key(&self)->Option<&Arc<()>> {
        self.job.as_ref().map(|j|&j.key).or_else(||self.staging.as_ref().map(|s|&s.key))
    }
    pub fn poll(&mut self)->Option<(Arc<()>,GiMeshSdfLevels)> {
        if !self.job.as_ref().is_some_and(|j|j.worker.is_finished()) {return None;}
        let job=self.job.take().unwrap();
        match job.worker.join() {
            Ok(field)=>Some((job.key,field)),
            Err(_)=>{tracing::warn!("Flux mesh distance-field worker panicked");None},
        }
    }
    /// Copies at most `bytes` this update. Generation itself always runs on the
    /// worker, including tiny primitives. The existing job is never detached.
    pub fn stage(&mut self,key:&Arc<()>,positions:&[Vec3],indices:&[u32],bytes:usize)->usize {
        if self.job.is_some() {return 0;}
        if self.staging.as_ref().is_none_or(|s|!Arc::ptr_eq(&s.key,key)) {
            self.staging=Some(Staging {key:key.clone(),positions:Vec::new(),indices:Vec::new()});
        }
        let s=self.staging.as_mut().unwrap();
        // Reserve the final capacities once to avoid copying an increasingly
        // large partial mesh on every Vec growth. Allocation is uninitialized.
        if s.positions.try_reserve_exact(positions.len()-s.positions.len()).is_err()
            || s.indices.try_reserve_exact(indices.len()-s.indices.len()).is_err() {
            tracing::warn!("Flux mesh distance-field staging allocation failed");return 0;
        }
        let count=(bytes/size_of::<Vec3>()).min(positions.len()-s.positions.len());
        let start=s.positions.len();s.positions.extend_from_slice(&positions[start..start+count]);
        let used=count*size_of::<Vec3>();
        let count=((bytes-used)/size_of::<u32>()).min(indices.len()-s.indices.len());
        let start=s.indices.len();s.indices.extend_from_slice(&indices[start..start+count]);
        let used=used+count*size_of::<u32>();
        if s.positions.len()==positions.len() && s.indices.len()==indices.len() {
            let staging=self.staging.take().unwrap();let key=staging.key.clone();
            match std::thread::Builder::new().name("flux-mesh-field".into()).spawn(move || {
                super::lower_compute_priority();
                generate(&staging.positions,&staging.indices)
            }) {
                Ok(worker)=>self.job=Some(Job {key,worker}),
                Err(error)=>tracing::warn!("Flux mesh distance-field worker: {error}"),
            }
        }
        used
    }
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn staging_is_bounded_and_reimports_cannot_accept_old_fields() {
        let positions=[Vec3::ZERO,Vec3::X,Vec3::Y];let indices=[0,1,2];
        let mut slot=Slot::default();let old=slot.key().clone();let mut stream=Stream::default();
        let mut copied=0;
        while stream.job.is_none() {let n=stream.stage(&old,&positions,&indices,16);assert!(n<=16);copied+=n;}
        assert_eq!(copied,size_of_val(&positions)+size_of_val(&indices));
        slot.reset();assert!(!Arc::ptr_eq(slot.key(),&old));
        assert_eq!(stream.stage(slot.key(),&positions,&indices,1024),0,"a second worker started before retirement");
        let start=std::time::Instant::now();
        let (key,field)=loop {
            if let Some(value)=stream.poll() {break value;}
            assert!(start.elapsed().as_secs()<30);std::thread::sleep(std::time::Duration::from_millis(1));
        };
        assert!(Arc::ptr_eq(&key,&old));assert!(!slot.accept(&key,field));assert!(slot.is_none());
        stream.stage(slot.key(),&positions,&indices,1024);
        let (key,field)=loop {
            if let Some(value)=stream.poll() {break value;}
            assert!(start.elapsed().as_secs()<30);std::thread::sleep(std::time::Duration::from_millis(1));
        };
        let expected=zestengine_render::sdf::generate_mesh_sdf(&positions,&indices,32,0.1);
        assert_eq!(field.detail.data,expected.data);
        let coarse=zestengine_render::sdf::generate_mesh_sdf(&positions,&indices,8,0.1);
        assert_eq!(field.coarse.data,coarse.data);
        assert_eq!(field.coarse.geometry_bounds,field.detail.geometry_bounds);
        assert!(slot.accept(&key,field));assert!(!slot.is_none());
    }
}
