//! Resolve custom water from an imported mesh template, including headless
//! hosts. The template may be hidden; it supplies geometry rather than a body.
use std::hash::{Hash,Hasher};
use zestengine_core::{Water,WaterBody,WaterMesh};
use crate::scene::LoadedScene;
pub fn sync(scene:&mut LoadedScene){
    if !scene.objects.iter().flat_map(|o|&o.components).any(|c|c.as_any().downcast_ref::<Water>().is_some_and(|w|w.body==WaterBody::Custom||w.custom_mesh.is_some())){return;}
    let world=scene.world_transforms();let mut updates=Vec::new();
    for(i,o)in scene.objects.iter().enumerate(){for(k,c)in o.components.iter().enumerate(){let Some(w)=c.as_any().downcast_ref::<Water>()else{continue};
        if w.body!=WaterBody::Custom {if w.custom_mesh.is_some(){updates.push((i,k,0,None));}continue;}
        let Some(source)=w.mesh_source.id().map_or(Some(i),|id|scene.index_of(id))else{updates.push((i,k,0,None));continue;};
        if !world[i].is_finite()||world[i].determinant().abs()<1e-8{updates.push((i,k,0,None));continue;}
        let mut hash=std::collections::hash_map::DefaultHasher::new();let mut slots=Vec::new();
        for(j,object)in scene.objects.iter().enumerate(){
            let mut at=Some(j);let mut inside=false;for _ in 0..64{let Some(node)=at else{break};if node==source{inside=true;break;}at=scene.objects.get(node).and_then(|o|o.parent());}
            if !inside{continue;}let m=world[i].inverse()*world[j];if !m.is_finite()||m.determinant().abs()<1e-8{continue;}
            for r in object.render().iter().chain(&(*object.extra_render())){if let Some((v,indices))=scene.mesh_geometry(r.mesh){
                r.mesh.hash(&mut hash);for x in m.to_cols_array(){x.to_bits().hash(&mut hash);}for p in v{for x in p.to_array(){x.to_bits().hash(&mut hash);}}indices.hash(&mut hash);slots.push((m,v,indices));
            }}
        }
        let key=hash.finish();if key==w.custom_mesh_source_key{continue;}
        let mut vertices=Vec::new();let mut indices=Vec::new();for(m,v,idx)in slots{let offset=vertices.len()as u32;vertices.extend(v.iter().map(|p|m.transform_point3(*p)));indices.extend(idx.iter().map(|i|*i+offset));}
        let mesh=WaterMesh::new(&vertices,&indices);updates.push((i,k,key,(!mesh.triangles.is_empty()).then(||std::sync::Arc::new(mesh))));
    }}
    for(i,k,key,mesh)in updates{if let Some(mut w)=scene.objects[i].components.get_typed_at_mut::<Water>(k){w.custom_mesh_source_key=key;w.custom_mesh=mesh;}}
}
