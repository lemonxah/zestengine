//! Opt-in full-renderer validation. An invisible X11 window supplies the
//! swapchain; viewport readback checks actual deformation and instance isolation.
use super::*;
use winit::{application::ApplicationHandler,event::WindowEvent,event_loop::{ActiveEventLoop,EventLoop},window::{Window,WindowId}};
use winit::platform::{run_on_demand::EventLoopExtRunOnDemand,x11::EventLoopBuilderExtX11};
use zestengine_render::{CameraData,FrameInput};

fn capture(r:&mut Renderer,s:&mut LoadedScene)->anyhow::Result<Vec<u8>>{
    s.sync_draws(None,0.0);
    let eye=Vec3::new(2.0,1.4,6.0);
    let camera=CameraData{view:Mat4::look_at_rh(eye,Vec3::new(2.0,0.5,0.0),Vec3::Y),proj:Mat4::perspective_rh(0.8,2.0,0.1,50.0),position:eye};
    let input=FrameInput{clear_color:[0.0,0.0,0.0,1.0],camera,light:Default::default(),lights:&[],capture_lights:&[],voxel_emitters:&[],camera_preview:None,model_preview:None,draw_skybox:false,
        shadow_pcf_texel:0.0,shadow_bias_scale:0.0,shadow_normal_offset:0.0,shadow_distance:10.0,shadow_cascades:1,shadow_cascade_lambda:0.5,time:0.0,water_time:0.0,water:&[],water_coverage:&[],draws:&s.draws,decals:&[],
        lightmap_preview:false,gi_debug:0,render_viewport:true,viewport_extent:Some([320,160]),postfx:Default::default(),preview_postfx:Default::default(),reflection_probe:None,reflection_probes:Default::default(),reflection_probe_b:None,
        reflection_primary:0,fog:None,voxel_specular:false,planar:&[],camera_targets:&[],egui:None,world_ui:vec![]};
    r.render(&input)?;
    let (w,h,pixels)=r.debug_read_viewport()?.expect("viewport readback");assert_eq!((w,h),(320,160));Ok(pixels)
}

#[derive(Default)]
struct App{result:Option<anyhow::Result<()>>}
impl App{
    fn probe(&mut self,event_loop:&ActiveEventLoop)->anyhow::Result<()>{
        let window=Arc::new(event_loop.create_window(Window::default_attributes().with_visible(false).with_inner_size(winit::dpi::LogicalSize::new(320.0,160.0)))?);
        let mut renderer=Renderer::new(window)?;
        renderer.set_vsync(false);renderer.set_gi_enabled(false);renderer.set_anti_aliasing(zestengine_render::AntiAliasing::None);
        let source_asset=asset();let mut s=LoadedScene::empty();s.add_asset_scene_with(Some(&mut renderer),&source_asset,None)?;
        let shader=renderer.unlit_shader()?;renderer.set_material_shader(s.materials[0].handle,Some(shader));
        let cache_project=Project::new();s.set_cloth_cache_root(&cache_project.0);
        let mut flag=MeshCloth{pin:PinSource::Top,gravity:[0.0,0.0,-9.81],..Default::default()};
        flag.cache.asset="flag.clothcache".into();flag.cache.frame_rate=60;flag.cache.record_seconds=1.5;flag.runtime.cache.record_requested=true;
        s.objects[1].components.push(Box::new(flag));
        s.update_skinning(&mut renderer,0.0);
        let id=s.objects[1].id();let original=s.mesh_handles[0];let deformed=s.draw_mesh(id,0,0);
        assert_ne!(deformed,original,"cloth has its own mutable GPU mesh");
        assert_eq!(s.draw_mesh(s.objects[2].id(),0,0),original,"rigid instance keeps the source buffer");
        let before=capture(&mut renderer,&mut s)?;
        for _ in 0..90{s.update_skinning(&mut renderer,1.0/60.0);}
        let after=capture(&mut renderer,&mut s)?;
        let mut changes=[0usize;2];let mut lit=[0usize;2];
        for (i,(a,b)) in before.chunks_exact(4).zip(after.chunks_exact(4)).enumerate(){
            let side=usize::from(i%320>=160);lit[side]+=usize::from(a[..3].iter().zip(&before[..3]).any(|(v,bg)|v.abs_diff(*bg)>10));
            changes[side]+=usize::from(a[..3].iter().zip(&b[..3]).any(|(a,b)|a.abs_diff(*b)>2));
        }
        eprintln!("[static cloth GPU] visible pixels {lit:?}; changed pixels {changes:?}");
        assert!(lit.iter().all(|n|*n>200),"both shared-mesh instances must be visible");
        assert!(changes[0]>100&&changes[1]<5,"only the cloth instance changes in the actual rendered image");
        if let Some(dir)=std::env::var_os("ZESTENGINE_CLOTH_DUMP_DIR"){
            let dir=PathBuf::from(dir);std::fs::create_dir_all(&dir)?;
            image::save_buffer(dir.join("static-cloth-before.png"),&before,320,160,image::ColorType::Rgba8)?;
            image::save_buffer(dir.join("static-cloth-after.png"),&after,320,160,image::ColorType::Rgba8)?;
        }
        // Replay the saved take through the same real renderer, including
        // its initial pose and final deformed pose. Rigid siblings stay intact.
        let take=zestengine_assets::cloth_cache_file::load_cloth_cache(cache_project.0.join("flag.clothcache"))?;
        {let mut c=cloth_mut(&mut s,1);c.cache.playback=true;c.cache.speed=0.0;c.cache.offset=0.0;}
        s.update_skinning(&mut renderer,0.0);let rewind=capture(&mut renderer,&mut s)?;
        assert!(cloth(&s,1).runtime.cache.playing);
        assert!(before.iter().zip(&rewind).all(|(a,b)|a.abs_diff(*b)<=2),"cached first frame reproduces the initial rendered image");
        cloth_mut(&mut s,1).cache.offset=take.duration()as f32;
        s.update_skinning(&mut renderer,0.0);let replay=capture(&mut renderer,&mut s)?;
        assert!(after.iter().zip(&replay).all(|(a,b)|a.abs_diff(*b)<=2),"cached final frame reproduces live cloth, including its normals");
        eprintln!("[cloth cache GPU] saved/reloaded take reproduced first and final rendered poses; {} frames",take.frames.len());
        s.objects[2].components.push(Box::new(MeshCloth{pin:PinSource::Top,..Default::default()}));s.update_skinning(&mut renderer,1.0/60.0);
        let second=s.draw_mesh(s.objects[2].id(),0,0);assert_ne!(second,original);assert_ne!(second,deformed,"two flags sharing a source have separate GPU buffers");
        s.objects[1].components.retain(|c|!c.as_any().is::<MeshCloth>());s.objects[2].components.retain(|c|!c.as_any().is::<MeshCloth>());
        s.update_skinning(&mut renderer,1.0/60.0);assert!(s.static_cloth.instances.is_empty());assert!(s.static_cloth.retired.is_empty());
        assert_eq!(s.draw_mesh(id,0,0),original);let restored=capture(&mut renderer,&mut s)?;
        assert!(before.iter().zip(&restored).all(|(a,b)|a.abs_diff(*b)<=2),"removing cloth restores the undeformed source");
        // The visible topology and GPU allocation survive resolution switches.
        let mut dense=asset();dense.meshes[0].vertices.clear();dense.meshes[0].indices.clear();
        for y in 0..=12{for x in 0..=12{dense.meshes[0].vertices.push(Vertex{position:[x as f32/12.0,y as f32/12.0,0.0],normal:[0.0,0.0,1.0],uv:[x as f32/12.0,y as f32/12.0],color:[1.0,0.2,0.4,1.0],..Default::default()});}}
        for y in 0..12{for x in 0..12{let a=y*13+x;dense.meshes[0].indices.extend([a,a+1,a+13,a+1,a+14,a+13]);}}
        let mut s=LoadedScene::empty();s.add_asset_scene_with(Some(&mut renderer),&dense,None)?;renderer.set_material_shader(s.materials[0].handle,Some(shader));
        s.objects[1].components.push(Box::new(MeshCloth{pin:PinSource::Top,gravity:[0.0,0.0,-9.81],lod:zestengine_core::ClothLod{enabled:true,levels:vec![zestengine_core::ClothLodLevel{distance:5.0,particle_spacing:0.25,..Default::default()}],..Default::default()},..Default::default()}));
        s.cloth_viewer=Some(Vec3::ZERO);
        for _ in 0..30{s.update_skinning(&mut renderer,1.0/60.0);}
        let near=capture(&mut renderer,&mut s)?;let particles=cloth(&s,1).runtime.sim.as_ref().unwrap().particles.len();let handle=s.draw_mesh(s.objects[1].id(),0,0);
        s.cloth_viewer=Some(Vec3::Z*10.0);s.update_skinning(&mut renderer,0.0);
        let far=capture(&mut renderer,&mut s)?;
        assert_eq!(cloth(&s,1).runtime.lod_level,1);assert!(cloth(&s,1).runtime.sim.as_ref().unwrap().particles.len()<particles/2);
        assert_eq!(handle,s.draw_mesh(s.objects[1].id(),0,0));
        assert!(near.iter().zip(&far).all(|(a,b)|a.abs_diff(*b)<=2),"changing simulation resolution does not pop the rendered cloth");
        for _ in 0..30{s.update_skinning(&mut renderer,1.0/60.0);}
        let settled=capture(&mut renderer,&mut s)?;
        assert!(far.chunks_exact(4).zip(settled.chunks_exact(4)).take(160*320).any(|(a,b)|a[..3].iter().zip(&b[..3]).any(|(a,b)|a.abs_diff(*b)>5)),"coarse cloth continues moving");
        s.cloth_viewer=Some(Vec3::ZERO);s.update_skinning(&mut renderer,0.0);let returned=capture(&mut renderer,&mut s)?;
        assert_eq!(cloth(&s,1).runtime.sim.as_ref().unwrap().particles.len(),particles);
        assert!(settled.iter().zip(&returned).all(|(a,b)|a.abs_diff(*b)<=2),"return to full resolution preserves the visible pose");
        eprintln!("[cloth LOD GPU] preserved pixels through near/far/near transitions with shared rigid instance");
        // A scene-authored level set drives the actual cloth vertex upload.
        // The unrelated rigid instance must keep exactly the same image.
        let source=asset();let mut s=LoadedScene::empty();s.add_asset_scene_with(Some(&mut renderer),&source,None)?;renderer.set_material_shader(s.materials[0].handle,Some(shader));
        let proxy=zestengine_assets::primitive_mesh(zestengine_assets::PrimitiveShape::Cube);
        let points=proxy.vertices.iter().map(|v|Vec3::from(v.position)*Vec3::new(4.0,4.0,0.2)).collect::<Vec<_>>();
        let field=zestengine_core::cloth_sdf::ClothSdf::from_mesh(&points,&proxy.indices,17).unwrap();
        let mut collider=zestengine_core::ClothCollider{shape:zestengine_core::ClothShape::LevelSet,center:[0.5,0.5,0.0],..Default::default()};collider.level_set.field=Some(field);
        let body=s.objects.len();let id=ObjectId::new();let mut object=SceneObject::new_empty(id,"Level-set body");object.transform_mut().translation.z=-0.2;
        object.components.push(Box::new(zestengine_core::ClothColliders{colliders:vec![collider],..Default::default()}));s.objects.push(object);
        let positions=source.meshes[0].vertices.iter().map(|v|Vec3::from(v.position)).collect::<Vec<_>>();
        s.objects[1].components.push(Box::new(MeshCloth{pin:PinSource::Painted,paint:ClothPaint::seed(&positions,&source.meshes[0].indices,&vec![0.0;positions.len()]),gravity:[0.0;3],collider_objects:vec![zestengine_core::ObjectRef(Some(id))],..Default::default()}));
        s.update_skinning(&mut renderer,1.0/60.0);let before=capture(&mut renderer,&mut s)?;
        s.objects[body].transform_mut().translation.z=0.5;s.update_skinning(&mut renderer,1.0/60.0);let after=capture(&mut renderer,&mut s)?;
        assert!(cloth(&s,1).runtime.sim.as_ref().unwrap().particles.iter().all(|p|p.pos.z>=0.6099));
        let mut changes=[0;2];for (i,(a,b)) in before.chunks_exact(4).zip(after.chunks_exact(4)).enumerate(){if a[..3].iter().zip(&b[..3]).any(|(a,b)|a.abs_diff(*b)>5){changes[usize::from(i%320>=160)]+=1;}}
        assert!(changes[0]>30&&changes[1]<5,"level-set displacement must reach the rendered cloth only: {changes:?}");
        eprintln!("[cloth SDF GPU] moving field changed cloth pixels, preserved rigid instance: {changes:?}");
        // Bone cloth reads a later animated object's current collider pose;
        // verify the resulting rotations reach both skinning backends.
        let mut source=asset();
        for v in &mut source.meshes[0].vertices{v.weights=[1.0,0.0,0.0,0.0];v.joints=[if v.position[1]>0.5{0}else{1},0,0,0];}
        let mut proxy=sheet();for v in &mut proxy.vertices{v.position=[100.0;3];v.weights=[1.0,0.0,0.0,0.0];}
        source.meshes.push(proxy);source.instances.push(AssetInstance::single("body proxy",1,0,Mat4::IDENTITY));
        source.skeletons.push(zestengine_assets::Skeleton{joints:vec![
            zestengine_assets::Joint{name:"root".into(),parent:None,inverse_bind:Mat4::from_translation(-Vec3::Y),rest_translation:Vec3::Y,rest_rotation:Quat::IDENTITY,rest_scale:Vec3::ONE},
            zestengine_assets::Joint{name:"tip".into(),parent:Some(0),inverse_bind:Mat4::IDENTITY,rest_translation:-Vec3::Y,rest_rotation:Quat::IDENTITY,rest_scale:Vec3::ONE},
        ],..Default::default()});
        let mut s=LoadedScene::empty();s.add_asset_scene_with(Some(&mut renderer),&source,None)?;
        renderer.set_material_shader(s.materials[0].handle,Some(shader));
        // This comparison object is an independent character, not a sibling
        // body part that intentionally follows its model's Animator.
        s.objects[2].set_parent(None);
        let body_id=s.objects[3].id();
        s.objects[1].components.push(Box::new(zestengine_core::Animator{gpu_skinning:!crate::cloth::cpu_skinning_forced(),..Default::default()}));
        s.objects[1].components.push(Box::new(zestengine_core::BoneCloth{roots:vec!["root".into()],end_tail:false,gravity:[0.0;3],stiffness:[0.0;2],
            collider_objects:vec![zestengine_core::ObjectRef(Some(body_id))],..Default::default()}));
        s.objects[3].components.push(Box::new(zestengine_core::Animator::default()));
        s.objects[3].components.push(Box::new(zestengine_core::ClothColliders{colliders:vec![zestengine_core::ClothCollider{shape:zestengine_core::ClothShape::Plane,
            bone:"root".into(),normal:[0.0,0.0,1.0],..Default::default()}],..Default::default()}));
        s.set_cloth_cache_root(&cache_project.0);
        let mut recorded_skin=MeshCloth{gravity:[0.0;3],..Default::default()};
        recorded_skin.cache.asset="skinned.clothcache".into();recorded_skin.cache.frame_rate=60;recorded_skin.cache.record_seconds=1.0/60.0;recorded_skin.runtime.cache.record_requested=true;
        s.objects[1].components.push(Box::new(recorded_skin));
        s.pose_override=Some((3,vec![Mat4::from_translation(Vec3::Y-Vec3::Z*0.2),Mat4::from_translation(-Vec3::Y)]));
        s.update_skinning(&mut renderer,1.0/60.0);let before=capture(&mut renderer,&mut s)?;
        s.pose_override=Some((3,vec![Mat4::from_translation(Vec3::Y+Vec3::Z*0.4),Mat4::from_translation(-Vec3::Y)]));
        s.update_skinning(&mut renderer,1.0/60.0);let after=capture(&mut renderer,&mut s)?;
        let c=s.objects[1].components.get_typed::<zestengine_core::BoneCloth>().unwrap();
        assert!(c.runtime.sim.as_ref().unwrap().particles[1].pos.z>=0.419);
        let mut changes=[0;2];for (i,(a,b)) in before.chunks_exact(4).zip(after.chunks_exact(4)).enumerate(){if a[..3].iter().zip(&b[..3]).any(|(a,b)|a.abs_diff(*b)>5){changes[usize::from(i%320>=160)]+=1;}}
        assert!(changes[0]>30&&changes[1]<5,"bone-cloth pose reaches its own rendered instance in the same frame: {changes:?}");
        eprintln!("[bone cloth pose GPU] same-frame external animated collider changed cloth only: {changes:?}");
        drop(c);
        {let mut c=cloth_mut(&mut s,1);c.cache.playback=true;c.cache.speed=0.0;c.cache.offset=0.0;}
        s.update_skinning(&mut renderer,0.0);let rewind=capture(&mut renderer,&mut s)?;
        assert!(cloth(&s,1).runtime.cache.playing);
        assert!(before.iter().zip(&rewind).all(|(a,b)|a.abs_diff(*b)<=2),"cached skinned vertices override the current animated pose");
        cloth_mut(&mut s,1).cache.offset=1.0/60.0;s.update_skinning(&mut renderer,0.0);let replay=capture(&mut renderer,&mut s)?;
        assert!(after.iter().zip(&replay).all(|(a,b)|a.abs_diff(*b)<=2),"skinned cache replay reproduces the recorded deformed pose");
        eprintln!("[skinned cloth cache GPU] both recorded animated poses reproduced through the cloth upload path");
        Ok(())
    }
}
impl ApplicationHandler for App{
    fn resumed(&mut self,event_loop:&ActiveEventLoop){if self.result.is_none(){self.result=Some(self.probe(event_loop));event_loop.exit();}}
    fn window_event(&mut self,_:&ActiveEventLoop,_:WindowId,_:WindowEvent){}
}

#[test]
fn static_cloth_gpu_deforms_only_its_instance_and_releases_buffers(){
    if std::env::var("ZESTENGINE_TEST_CLOTH_RENDER").as_deref()!=Ok("1"){eprintln!("[static cloth GPU] opt in with ZESTENGINE_TEST_CLOTH_RENDER=1 and an X11 display");return;}
    let mut event_loop=EventLoop::builder().with_x11().with_any_thread(true).build().expect("X11 event loop");
    let mut app=App::default();event_loop.run_app_on_demand(&mut app).unwrap();app.result.expect("renderer test ran").unwrap();
}
