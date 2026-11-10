//! Static meshes use the same cloth solver, with their undeformed vertices as
//! animation targets. Only cloth instances get mutable render buffers. Imported
//! vertex attributes load lazily; ordinary world meshes keep no extra CPU copy.
use super::*;
use std::sync::Arc;
use std::collections::HashSet;
use crate::cloth_contacts::ClothRenderWork;
use zestengine_core::MeshCloth;
use zestengine_render::MeshData;

enum Source {
    Model(PathBuf,usize),
    Primitive(zestengine_assets::PrimitiveShape),
    Memory(Arc<MeshData>),
}
struct Entry {
    source:Source,
    loaded:Option<Arc<MeshData>>,
    revision:u64,
    signature:Option<u64>,
    error:Option<String>,
}
struct Instance {mesh:usize,revision:u64,handle:MeshHandle}

#[derive(Default)]
pub(super) struct StaticCloth {
    sources:HashMap<usize,Entry>,
    instances:HashMap<(ObjectId,usize),Instance>,
    retired:HashSet<usize>,
    revision:u64,
    /// Current cloth render bounds, including skinned cloth, in object space.
    pub bounds:HashMap<(ObjectId,usize),(Vec3,Vec3)>,
}

fn copy_mesh(mesh:&MeshData)->MeshData{MeshData{vertices:mesh.vertices.clone(),indices:mesh.indices.clone(),has_lightmap_uv:mesh.has_lightmap_uv}}
fn signature(mesh:&MeshData)->u64{
    use std::hash::{Hash,Hasher};
    let mut h=std::collections::hash_map::DefaultHasher::new();mesh.indices.hash(&mut h);mesh.has_lightmap_uv.hash(&mut h);
    for v in &mesh.vertices{
        for f in v.position.into_iter().chain(v.normal).chain(v.uv).chain(v.uv1).chain(v.color).chain(v.tangent).chain(v.weights){f.to_bits().hash(&mut h);}
        v.joints.hash(&mut h);
    }h.finish()
}

impl StaticCloth {
    pub(super) fn is_idle(&self)->bool{self.instances.is_empty()&&self.retired.is_empty()&&self.bounds.is_empty()}
    pub(super) fn resolve_paths(&mut self,root:&Path){
        let root=if root.is_absolute(){root.to_owned()}else{std::env::current_dir().unwrap_or_default().join(root)};
        for entry in self.sources.values_mut(){if let Source::Model(path,_)=&mut entry.source{if path.is_relative(){*path=root.join(&*path);}}}
    }
    fn register(&mut self,mesh:usize,source:Source){
        self.revision=self.revision.wrapping_add(1);
        self.sources.insert(mesh,Entry{source,loaded:None,revision:self.revision,error:None,signature:None});
    }
    pub(super) fn imported(&mut self,index:usize,mesh:&MeshData,path:Option<&Path>,local:usize){
        let live=self.sources.get(&index).is_some_and(|e|e.loaded.is_some());
        self.register(index,match path{Some(path)=>Source::Model(path.to_owned(),local),None=>Source::Memory(Arc::new(copy_mesh(mesh)))});
        self.sources.get_mut(&index).unwrap().signature=Some(signature(mesh));
        if live{self.sources.get_mut(&index).unwrap().loaded=Some(Arc::new(copy_mesh(mesh)));}
    }
    pub(super) fn primitive(&mut self,index:usize,shape:zestengine_assets::PrimitiveShape){self.register(index,Source::Primitive(shape));}
    fn data(&mut self,mesh:usize,retry:bool)->Result<(Arc<MeshData>,u64),String>{
        let entry=self.sources.get_mut(&mesh).ok_or_else(||"This mesh has no cloth vertex source. Reimport its model or use a mesh primitive.".to_string())?;
        if retry{entry.error=None;}
        if let Some(error)=&entry.error{return Err(error.clone());}
        if let Some(data)=&entry.loaded{return Ok((data.clone(),entry.revision));}
        let result=match &entry.source{
            Source::Memory(data)=>Ok(data.clone()),
            Source::Primitive(shape)=>Ok(Arc::new(zestengine_assets::primitive_mesh(*shape))),
            Source::Model(path,local)=>zestengine_assets::load_model_with_meta(path).map_err(|e|format!("Cannot load cloth mesh {}: {e:#}",path.display()))
                .and_then(|scene|scene.meshes.into_iter().nth(*local).map(Arc::new).ok_or_else(||"Cloth submesh is missing; reimport the model.".into())),
        };
        let result=result.and_then(|data|{
            if entry.signature.is_some_and(|expected|signature(&data)!=expected){Err("Cloth mesh differs from the loaded model. Reimport the model to refresh its vertex attributes.".into())}else{Ok(data)}
        });
        match result{
            Ok(data)=>{entry.loaded=Some(data.clone());Ok((data,entry.revision))},
            Err(error)=>{entry.error=Some(error.clone());Err(error)},
        }
    }
    fn retire(&mut self,key:(ObjectId,usize)){
        if let Some(instance)=self.instances.remove(&key){if instance.handle!=MeshHandle::NONE{self.retired.insert(instance.handle.index());}}
    }
    pub(super) fn draw(&self,id:ObjectId,slot:usize,mesh:usize)->Option<MeshHandle>{
        self.instances.get(&(id,slot)).filter(|i|i.mesh==mesh&&i.handle!=MeshHandle::NONE).map(|i|i.handle)
    }
}

#[cfg(test)]
#[path="cloth_static_tests.rs"]
mod tests;

impl LoadedScene {
    /// Garment imports are immediately reimportable, including before the first
    /// scene save/reload. All placements share one registered model template.
    pub(super) fn register_clothing_template(&mut self,asset:&zestengine_assets::Scene,path:Option<&Path>,mesh_base:usize,skel_base:usize,material_base:usize){
        if asset.identity.cloth_components.is_empty(){return;}
        let Some(path)=path else{return};let key=path.to_string_lossy().into_owned();
        let mut materials=vec![material_base;asset.meshes.len()];
        for instance in &asset.instances{for slot in &instance.slots{materials[slot.mesh]=material_base+slot.material;}}
        self.model_slots.insert(key.clone(),model_reimport::ModelSlots::installed(asset,skel_base,self.animations.len()-asset.animations.len(),material_base));
        self.model_templates.insert(key,(mesh_base,materials));
    }
    pub(super) fn place_existing_clothing(&mut self,asset:&zestengine_assets::Scene,path:Option<&Path>)->Result<bool>{
        if asset.identity.cloth_components.is_empty(){return Ok(false);}
        let Some(path)=path else{return Ok(false)};let key=path.to_string_lossy().into_owned();
        let Some((base,materials))=self.model_templates.get(&key).cloned()else{return Ok(false)};
        // A topology change must go through reimport before placement so the
        // existing template and the imported seams always describe one mesh.
        if asset.meshes.len()!=materials.len()||asset.meshes.iter().enumerate().any(|(i,m)|{
            let static_changed=self.static_cloth.sources.get(&(base+i)).is_some_and(|e|e.signature.is_some_and(|old|old!=signature(m)));
            static_changed||self.mesh_geometry.get(base+i).is_none_or(|(p,t)|p.len()!=m.vertices.len()||*t!=m.indices||p.iter().zip(&m.vertices).any(|(p,v)|p.to_array()!=v.position))
        }){
            anyhow::bail!("Garment geometry changed; reimport {key} before placing another instance");
        }
        let root=self.objects.len();self.objects.push(SceneObject::new_empty(ObjectId::new(),path.file_stem().and_then(|n|n.to_str()).unwrap_or("Garment")));
        for (i,instance) in asset.instances.iter().enumerate(){
            let Some(first)=instance.slots.first()else{continue};let mut object=SceneObject::new_empty(ObjectId::new(),&instance.name);
            object.set_parent(Some(root));let (scale,rotation,translation)=instance.transform.to_scale_rotation_translation();
            *object.transform_mut()=zestengine_core::Transform{translation,rotation,scale};
            let render=|slot:&zestengine_assets::MeshSlot|RenderInfo{mesh:base+slot.mesh,material:materials[slot.mesh]};
            *object.render_mut()=Some(render(first));(*object.extra_render_mut())=instance.slots[1..].iter().map(render).collect();
            (*object.source_mut())=ObjectSource::Model{path:key.clone(),mesh:first.mesh,extra_meshes:instance.slots[1..].iter().map(|s|s.mesh).collect(),model:asset.identity.guid,mesh_id:asset.identity.mesh_ids.get(first.mesh).copied(),extra_mesh_ids:instance.slots[1..].iter().filter_map(|s|asset.identity.mesh_ids.get(s.mesh).copied()).collect()};
            *object.enabled_mut()=!asset.identity.hidden_instances.contains(&i);object.components.replace_all(instance_components(asset,instance));
            object.components.insert(0,Box::new(zestengine_core::MeshRenderer::inheriting(instance.slots.len())));self.objects.push(object);
        }
        Ok(true)
    }
    pub(super) fn warm_static_cloth(&mut self,models:&HashMap<String,zestengine_assets::Scene>){
        for object in &self.objects {
            let ObjectSource::Model{path,..}=&(*object.source()) else{continue};
            let Some(asset)=models.get(path)else{continue};
            for (slot,r) in object.render_slots().enumerate(){
                if !crate::cloth::has_mesh_cloth_slot(&object.components,slot){continue;}
                let Some(entry)=self.static_cloth.sources.get_mut(&r.mesh)else{continue};
                if let Source::Model(_,local)=&entry.source{
                    if let Some(mesh)=asset.meshes.get(*local){entry.loaded=Some(Arc::new(copy_mesh(mesh)));}
                }
            }
        }
    }
    pub(super) fn queue_static_cloth(&mut self,mut renderer:Option<&mut Renderer>,active:&[bool],world:&[Mat4],work:&mut Vec<ClothRenderWork>){
        let skinned:HashSet<_>=self.skinned_meshes.iter().map(|sm|sm.mesh_index).collect();
        let jobs:Vec<_>=self.objects.iter().enumerate().filter(|(i,_)|active[*i]).flat_map(|(i,o)|{
            o.render_slots().enumerate().filter(|(slot,r)|!skinned.contains(&r.mesh)&&crate::cloth::has_mesh_cloth_slot(&o.components,*slot))
                .map(move |(slot,r)|(i,slot,r.mesh))
        }).collect();
        let mut live=HashSet::new();
        // A constant identity palette uses the existing per-frame GPU patch
        // buffers without requiring authored bones. The CPU fallback remains
        // available through the same switch as skinned garments.
        let gpu=!crate::cloth::cpu_skinning_forced();
        for (object,slot,mesh) in jobs{
            let key=(self.objects[object].id(),slot);
            let retry=self.objects[object].components.iter_typed::<MeshCloth>().find(|c|c.slot==slot).is_some_and(|c|c.runtime.reset_requested);
            let result=self.static_cloth.data(mesh,retry);
            let (data,revision)=match result{
                Ok(value)=>value,
                Err(error)=>{
                    let name=self.objects[object].name();
                    if let Some(mut c)=self.objects[object].components.iter_typed_mut::<MeshCloth>().find(|c|c.slot==slot){
                        if c.runtime.build_error!=error{tracing::warn!("{name}: {error}");}
                        c.runtime.build_error=error;c.runtime.sim=None;c.runtime.reset_requested=false;
                    }
                    continue;
                },
            };
            if self.static_cloth.instances.get(&key).is_some_and(|i|i.mesh!=mesh||i.revision!=revision){self.static_cloth.retire(key);}
            self.static_cloth.instances.entry(key).or_insert(Instance{mesh,revision,handle:MeshHandle::NONE});
            let instance=self.static_cloth.instances.get_mut(&key).unwrap();
            if instance.handle==MeshHandle::NONE {
                if let Some(renderer)=renderer.as_deref_mut(){
                    let mut unweighted=None;
                    if gpu&&data.vertices.iter().any(|v|v.weights.iter().any(|w|*w!=0.0)){
                        let mut copy=copy_mesh(&data);for v in &mut copy.vertices{v.weights=[0.0;4];v.joints=[0;4];}unweighted=Some(copy);
                    }
                    let upload_data=unweighted.as_ref().unwrap_or(&data);
                    let upload=if gpu{renderer.upload_mesh_gpu_skinned_morphed(upload_data,1,&[])}else{renderer.upload_mesh_skinned(upload_data)};
                    match upload{
                        Ok(handle)=>instance.handle=handle,
                        Err(error)=>{tracing::error!("static cloth mesh upload: {error:#}");continue;},
                    }
                }
            }
            if gpu{if let Some(renderer)=renderer.as_deref_mut(){renderer.skin_mesh_gpu(instance.handle,&[Mat4::IDENTITY]);}}
            live.insert(key);
            work.push(ClothRenderWork{mesh,object,slot,handle:instance.handle,animated:data.vertices.clone(),world:world[object],gpu});
        }
        let stale:Vec<_>=self.static_cloth.instances.keys().copied().filter(|key|!live.contains(key)).collect();
        for key in stale{self.static_cloth.retire(key);}
        if let Some(renderer)=renderer{
            if !self.static_cloth.retired.is_empty(){
                match renderer.free_resources(&self.static_cloth.retired,&HashSet::new(),&HashSet::new()){
                    Ok(())=>self.static_cloth.retired.clear(),Err(error)=>tracing::warn!("releasing cloth meshes: {error:#}"),
                }
            }
        }
        // Release decoded model attributes when no cloth instance uses them.
        let used:HashSet<_>=self.static_cloth.instances.values().map(|i|i.mesh).collect();
        for (mesh,entry) in &mut self.static_cloth.sources{if !used.contains(mesh){entry.loaded=None;}}
    }

    pub(super) fn static_cloth_source(&self,mesh:usize)->Option<Arc<MeshData>>{
        self.static_cloth.sources.get(&mesh)?.loaded.clone()
    }

    pub(super) fn finish_cloth(&mut self,work:Vec<ClothRenderWork>,renderer:Option<&mut Renderer>,active:&[bool],world:&[Mat4],winds:&[zestengine_core::WindSample],viewer:Option<Vec3>,dt:f32){
        self.simulate_queued_cloth(&work,active,world,winds,viewer,dt);
        let bounds=crate::cloth_contacts::finish(&self.objects,work,renderer,dt,&mut self.cloth_patch);
        self.cloth_cache_assets.flush(&self.objects);
        self.static_cloth.bounds=bounds.into_iter().map(|(object,slot,bounds)|((self.objects[object].id(),slot),bounds)).collect();
    }
}
