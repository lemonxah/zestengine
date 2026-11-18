//! `.cloth` garments: authoring data, generated model import and optional skin transfer.
use std::path::{Path,PathBuf};
use anyhow::{Result,ensure,Context};
use serde::{Serialize,Deserialize};
use zestengine_core::{ClothPattern,ClothPanel,MeshCloth,PinSource};
use zestengine_render::{MeshData,Vertex};
use glam::{Vec3,Mat4};
pub const CLOTH_EXTENSION:&str="cloth";
#[derive(Clone,Serialize,Deserialize)]
#[serde(default)]
pub struct ClothingAsset{
    pub version:u32,
    pub name:String,
    pub pattern:ClothPattern,
    pub settings:MeshCloth,
    /// Optional body model, project relative (or relative to this file outside a project).
    pub skin_model:String,
    pub skin_mesh:usize,
    pub skin_skeleton:usize,
    pub transfer_distance:f32,
}
impl Default for ClothingAsset{fn default()->Self{Self{version:1,name:"Garment".into(),pattern:ClothPattern{panels:vec![ClothPanel::default()],seams:Vec::new()},settings:MeshCloth{pin:PinSource::Top,..Default::default()},skin_model:String::new(),skin_mesh:0,skin_skeleton:0,transfer_distance:0.25}}}
impl ClothingAsset{
    pub fn add_panel(&mut self,panel:ClothPanel)->Result<()>{
        let mut pattern=self.pattern.clone();pattern.seams.clear();
        let old=pattern.assemble().ok();pattern.panels.push(panel.clone());let geometry=pattern.assemble().map_err(anyhow::Error::msg)?;
        if let Some(old)=old.filter(|g|self.settings.paint.matches(g.positions.len(),zestengine_core::topology_hash(g.positions.len(),&g.indices))){
            let sources=(0..geometry.positions.len()).map(|i|(i<old.positions.len()).then_some(i)).collect::<Vec<_>>();
            self.settings.paint.reindex(&geometry.positions,&geometry.indices,&sources);
        }
        self.pattern.panels.push(panel);Ok(())
    }
    /// Panel authoring has explicit vertex identities. Moving a panel keeps
    /// its masks attached; a genuinely new topology uses spatial transfer.
    pub fn refresh_pattern_paint(&mut self)->Result<()>{
        if self.settings.paint.is_empty(){return Ok(());}
        let mut pattern=self.pattern.clone();pattern.seams.clear();let geometry=pattern.assemble().map_err(anyhow::Error::msg)?;
        if !self.settings.paint.rebind_positions(&geometry.positions,&geometry.indices){self.settings.paint.fit(&geometry.positions,&geometry.indices);}
        Ok(())
    }
    /// Remove a panel and remap authored global stitches on surviving panels.
    /// Pattern seams use panel-local indices; explicit settings stitches use
    /// assembled vertex indices and must not silently attach to another panel.
    pub fn remove_panel(&mut self,index:usize)->Result<()>{
        ensure!(index<self.pattern.panels.len(),"Missing panel");
        let counts=self.pattern.panels.iter().map(|p|p.mesh().map(|(p,_)|p.len()).map_err(anyhow::Error::msg)).collect::<Result<Vec<_>>>()?;
        let start=counts[..index].iter().sum::<usize>()as u32;let count=counts[index]as u32;let end=start+count;
        let mut geometry_pattern=self.pattern.clone();geometry_pattern.seams.clear();
        let old_geometry=geometry_pattern.assemble().map_err(anyhow::Error::msg)?;
        let paint_matches=self.settings.paint.matches(old_geometry.positions.len(),zestengine_core::topology_hash(old_geometry.positions.len(),&old_geometry.indices));
        geometry_pattern.panels.remove(index);
        let new_geometry=if geometry_pattern.panels.is_empty(){None}else{Some(geometry_pattern.assemble().map_err(anyhow::Error::msg)?)};
        self.settings.stitches.retain(|s|[s.vertex,s.edge[0],s.edge[1]].iter().all(|&v|v<start||v>=end));
        for stitch in &mut self.settings.stitches{for v in std::iter::once(&mut stitch.vertex).chain(stitch.edge.iter_mut()){if *v>=end{*v-=count;}}}
        self.pattern.panels.remove(index);self.pattern.seams.retain(|s|s.a.panel!=index&&s.b.panel!=index);
        for seam in &mut self.pattern.seams{if seam.a.panel>index{seam.a.panel-=1;}
        if seam.b.panel>index{seam.b.panel-=1;}}
        if paint_matches{if let Some(geometry)=new_geometry{
            let retained=(0..old_geometry.positions.len()).filter(|&i|i<start as usize||i>=end as usize).map(Some).collect::<Vec<_>>();
            self.settings.paint.reindex(&geometry.positions,&geometry.indices,&retained);
        }else{self.settings.paint=Default::default();}}
        Ok(())
    }
}
fn validate_stitches(stitches:&[zestengine_core::ClothStitch],count:usize)->Result<()>{
    ensure!(stitches.len()<=1_000_000,"Too many clothing stitches");
    ensure!(stitches.iter().all(|s|[s.vertex,s.edge[0],s.edge[1]].iter().all(|&v|(v as usize)<count)&&s.weight.is_finite()&&(0.0..=1.0).contains(&s.weight)&&s.compliance.is_finite()&&s.compliance>=0.0),"Invalid clothing stitch");Ok(())
}
pub fn load_clothing(path:impl AsRef<Path>)->Result<ClothingAsset>{
    let path=path.as_ref();ensure!(std::fs::metadata(path)?.len()<=64*1024*1024,"Clothing asset exceeds 64 MiB");let asset:ClothingAsset=crate::profile_file::load(path)?;ensure!(asset.version==1,"Unsupported clothing asset version {}",asset.version);let geometry=asset.pattern.assemble().map_err(anyhow::Error::msg)?;validate_stitches(&asset.settings.stitches,geometry.positions.len())?;Ok(asset)
}
pub fn save_clothing(path:impl AsRef<Path>,asset:&ClothingAsset)->Result<()>{
    ensure!(asset.version==1,"Unsupported clothing asset version");let geometry=asset.pattern.assemble().map_err(anyhow::Error::msg)?;validate_stitches(&asset.settings.stitches,geometry.positions.len())?;let path=path.as_ref();if let Some(parent)=path.parent(){std::fs::create_dir_all(parent)?;}
    let temp=path.with_extension(format!("cloth.{}.tmp",zestengine_core::ObjectId::new()));let result=(||{crate::profile_file::save(&temp,asset)?;std::fs::rename(&temp,path).context("installing clothing asset")})();if result.is_err(){let _=std::fs::remove_file(&temp);}result
}
fn component(asset:&ClothingAsset,path:&Path)->Result<MeshCloth>{let mut component=asset.settings.clone();component.slot=0;let root=clothing_root(path);let relative=path.strip_prefix(&root).unwrap_or(path).to_string_lossy().into_owned();component.garment=zestengine_core::asset_ref::reference(&root,&relative);component.stitches.extend(asset.pattern.assemble().map_err(anyhow::Error::msg)?.stitches);component.collider_objects.clear();Ok(component)}
pub fn component_from_clothing(path:impl AsRef<Path>)->Result<MeshCloth>{let path=path.as_ref();component(&load_clothing(path)?,path)}

pub fn clothing_root(path:&Path)->PathBuf{path.ancestors().skip(1).find(|p|p.join(crate::PROJECT_FILE_NAME).is_file()).unwrap_or_else(||path.parent().unwrap_or(Path::new("."))).to_owned()}
pub fn load_clothing_model(path:impl AsRef<Path>)->Result<crate::Scene>{
    let path=path.as_ref();let asset=load_clothing(path)?;build_clothing_model(path,&asset)
}
pub fn build_clothing_model(path:&Path,asset:&ClothingAsset)->Result<crate::Scene>{
    let geometry=asset.pattern.assemble().map_err(anyhow::Error::msg)?;
    let mut vertices:Vec<_>=geometry.positions.iter().zip(&geometry.uv).map(|(p,uv)|Vertex{position:p.to_array(),normal:[0.0;3],uv:uv.to_array(),color:[0.0,0.0,0.0,1.0],..Default::default()}).collect();
    let mut normals=vec![Vec3::ZERO;vertices.len()];for tri in geometry.indices.as_chunks::<3>().0{let[a,b,c]=[tri[0]as usize,tri[1]as usize,tri[2]as usize];let n=(geometry.positions[b]-geometry.positions[a]).cross(geometry.positions[c]-geometry.positions[a]);for i in [a,b,c]{normals[i]+=n;}}
    for (v,n) in vertices.iter_mut().zip(normals){v.normal=n.normalize_or(Vec3::Z).to_array();}
    let mut skeletons=Vec::new();
    if !asset.skin_model.trim().is_empty(){let root=clothing_root(path);let rel=zestengine_core::asset_ref::resolve(&root,&asset.skin_model).context("Missing clothing skin source")?;let source_path=root.join(rel);ensure!(!source_path.extension().is_some_and(|e|e.eq_ignore_ascii_case("cloth")),"Skin source must be an avatar model, not another clothing asset");let source=crate::load_model_with_meta(&source_path)?;let mesh=source.meshes.get(asset.skin_mesh).context("Missing skin source mesh")?;let skeleton=source.skeletons.get(asset.skin_skeleton).context("Missing skin source skeleton")?;
        crate::cloth_skin_transfer::transfer_clothing_weights(&mut vertices,mesh,skeleton.joints.len(),asset.transfer_distance)?;skeletons.push(skeleton.clone());
    }
    Ok(crate::Scene{meshes:vec![MeshData{vertices,indices:geometry.indices,has_lightmap_uv:false}],morphs:Vec::new(),springs:Default::default(),textures:Vec::new(),materials:vec![crate::SceneMaterial{name:"Fabric".into(),params:Default::default(),features:Default::default(),albedo:None,normal:None,orm:None,emission:None}],instances:vec![crate::Instance::single(asset.name.clone(),0,0,Mat4::IDENTITY)],skeletons,animations:Vec::new(),collision:Vec::new(),identity:crate::ModelIdentity{cloth_components:vec![(0,ron::to_string(&component(asset,path)?)?)],..Default::default()}})
}

#[cfg(test)]
mod tests{
    use super::*;
    struct Project(PathBuf);impl Project{fn new()->Self{let p=std::env::temp_dir().join(format!("cloth_asset_{}",zestengine_core::ObjectId::new()));std::fs::create_dir_all(&p).unwrap();Self(p)}}impl Drop for Project{fn drop(&mut self){let _=std::fs::remove_dir_all(&self.0);}}
    fn body(root:&Path){
        let mut bytes=Vec::new();for p in [[0.0f32,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0]]{for v in p{bytes.extend(v.to_le_bytes());}}
        for _ in 0..3{for v in [1.0f32,0.0,0.0,0.0]{bytes.extend(v.to_le_bytes());}}
        for _ in 0..12{bytes.extend(0u16.to_le_bytes());}for v in [0u16,1,2]{bytes.extend(v.to_le_bytes());}
        std::fs::write(root.join("body.bin"),&bytes).unwrap();
        let gltf=serde_json::json!({"asset":{"version":"2.0"},"buffers":[{"uri":"body.bin","byteLength":bytes.len()}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36},{"buffer":0,"byteOffset":36,"byteLength":48},{"buffer":0,"byteOffset":84,"byteLength":24},{"buffer":0,"byteOffset":108,"byteLength":6}],"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]},{"bufferView":1,"componentType":5126,"count":3,"type":"VEC4"},{"bufferView":2,"componentType":5123,"count":3,"type":"VEC4"},{"bufferView":3,"componentType":5123,"count":3,"type":"SCALAR"}],"meshes":[{"primitives":[{"attributes":{"POSITION":0,"WEIGHTS_0":1,"JOINTS_0":2},"indices":3}]}],"nodes":[{"name":"Body","mesh":0,"skin":0},{"name":"Root"}],"skins":[{"joints":[1]}],"scenes":[{"nodes":[0,1]}],"scene":0});
        std::fs::write(root.join("body.gltf"),gltf.to_string()).unwrap();
    }
    #[test]
    fn clothing_asset_roundtrips_imports_skin_weights_and_packages_dependencies(){
        let project=Project::new();body(&project.0);let path=project.0.join("shirt.cloth");let mut asset=ClothingAsset::default();asset.pattern.panels[0].size=[0.25,0.25];asset.pattern.panels[0].segments=[2,2];asset.skin_model="body.gltf".into();
        asset.settings.fabric.as_mut().unwrap().warp=2345.0;save_clothing(&path,&asset).unwrap();let decoded=load_clothing(&path).unwrap();assert_eq!(decoded.settings.fabric.unwrap().warp,2345.0);
        let imported=crate::load_model_with_meta(&path).unwrap();assert_eq!(imported.meshes[0].vertices.len(),9);assert_eq!(imported.skeletons.len(),1);assert_eq!(imported.skeletons[0].joints[0].name,"Root");
        for v in &imported.meshes[0].vertices{assert!((v.weights.iter().sum::<f32>()-1.0).abs()<1e-6);assert_eq!(v.joints[0],0);}
        let component=component_from_clothing(&path).unwrap();assert!(component.runtime.sim.is_none());assert_eq!(component.fabric.unwrap().warp,2345.0);
        let (files,warnings)=crate::package::collect_dependencies(&project.0,"entry.scene","(model:\"shirt.cloth\")");assert!(warnings.is_empty(),"{warnings:?}");for name in ["shirt.cloth","body.gltf","body.bin"]{assert!(files.contains(name),"missing {name}: {files:?}");}
        // Regeneration observes a source/pattern edit instead of a stale cached mesh.
        asset.pattern.panels[0].segments=[3,3];save_clothing(&path,&asset).unwrap();let(imported,report)=crate::load_model_cached(&path,false).unwrap();assert_eq!(imported.meshes[0].vertices.len(),16);assert_eq!(report.outcome,crate::import_cache::CacheOutcome::Disabled);
        let before=std::fs::read(&path).unwrap();asset.version=99;assert!(save_clothing(&path,&asset).is_err());assert_eq!(std::fs::read(&path).unwrap(),before);
    }
    #[test]
    fn custom_panel_edits_roundtrip_and_removal_preserves_surviving_stitches(){
        use zestengine_core::{ClothStitch,ClothSeam,ClothPanelEdge,ClothPanelSide};
        let project=Project::new();let path=project.0.join("custom.cloth");let mut asset=ClothingAsset::default();
        asset.pattern.panels=vec![ClothPanel{segments:[1,1],..Default::default()};3];
        asset.pattern.seams=vec![ClothSeam{a:ClothPanelEdge{side:ClothPanelSide::Right,..Default::default()},b:ClothPanelEdge{panel:2,..Default::default()},..Default::default()}];
        asset.settings.stitches=vec![ClothStitch{vertex:0,edge:[8,9],weight:0.5,compliance:0.0},ClothStitch{vertex:5,edge:[8,9],weight:0.0,compliance:0.0}];
        asset.pattern.make_panel_editable(2).unwrap();asset.pattern.panels[2].move_vertex(0,[-0.1,0.0]).unwrap();asset.remove_panel(1).unwrap();
        assert_eq!(asset.settings.stitches,vec![ClothStitch{vertex:0,edge:[4,5],weight:0.5,compliance:0.0}]);assert_eq!(asset.pattern.seams[0].b.panel,1);
        asset.pattern.panels.push(ClothPanel::from_outline(&[[0.0,0.0],[0.5,0.0],[0.5,0.2],[0.2,0.2],[0.2,0.5],[0.0,0.5]],2).unwrap());
        save_clothing(&path,&asset).unwrap();let loaded=load_clothing(&path).unwrap();assert_eq!(loaded.pattern.panels[1].vertices[0],[-0.1,0.0]);
        let model=load_clothing_model(&path).unwrap();assert_eq!(model.meshes[0].vertices.len(),loaded.pattern.assemble().unwrap().positions.len());
        assert_eq!(component_from_clothing(&path).unwrap().stitches.len(),loaded.settings.stitches.len()+loaded.pattern.assemble().unwrap().stitches.len());
    }
    #[test]
    fn panel_authoring_keeps_paint_identity_even_for_overlapping_panels(){
        use zestengine_core::{ClothPaint,PaintChannel};
        let mut asset=ClothingAsset::default();asset.pattern.panels=vec![ClothPanel{segments:[1,1],..Default::default()};3];
        let g=asset.pattern.assemble().unwrap();asset.settings.paint=ClothPaint::seed(&g.positions,&g.indices,&[0.2,0.2,0.2,0.2,0.5,0.5,0.5,0.5,0.8,0.8,0.8,0.8]);
        asset.settings.paint.set_values(PaintChannel::Warp,&[0.0,0.0,0.0,0.0,0.5,0.5,0.5,0.5,1.0,1.0,1.0,1.0]);
        asset.remove_panel(1).unwrap();assert_eq!(asset.settings.paint.pins.as_slice(),&[51,51,51,51,204,204,204,204]);
        asset.pattern.panels[1].translation=[3.0,0.0,0.0];asset.refresh_pattern_paint().unwrap();let before=asset.settings.paint.clone();
        let g=asset.pattern.assemble().unwrap();asset.settings.paint.fit(&g.positions,&g.indices);assert_eq!(asset.settings.paint.pins,before.pins);assert_eq!(asset.settings.paint.maps,before.maps);
        asset.add_panel(ClothPanel{segments:[1,1],..Default::default()}).unwrap();
        assert_eq!(asset.settings.paint.pins.as_slice(),&[51,51,51,51,204,204,204,204,0,0,0,0]);
        assert_eq!(asset.settings.paint.values(PaintChannel::Warp),vec![0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]);
    }
}
