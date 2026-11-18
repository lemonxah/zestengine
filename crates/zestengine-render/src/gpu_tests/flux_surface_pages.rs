//! Actual variable-resolution captures, GPU history migration and atlas growth.
use glam::{Mat4,Quat,Vec3};
use crate::{CameraData,FluxSettings,GiSource,GpuGiMaterial,Renderer};
use super::planar::{cube,draw,lit};
fn camera(z:f32)->CameraData {
    let position=Vec3::new(0.0,0.0,z);
    CameraData {position,view:Mat4::look_at_rh(position,Vec3::ZERO,Vec3::Y),proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,200.0)}
}
fn step(r:&mut Renderer,input:&crate::FrameInput,n:usize)->anyhow::Result<()> {
    for _ in 0..n {
        r.render(input)?;
        let cache=r.flux_surface.as_ref().unwrap();
        assert!(cache.captures<=12 && cache.capture_texels<=12*64*64);
        assert!(cache.lighting_texels[0]<=24*64*64 && cache.lighting_texels[1]<=12*64*64);
        let (extent,pages)=cache.debug_pages();assert!(extent<=2048);
        for (i,(_,a)) in pages.iter().enumerate() {
            assert!(a[0]+a[2]<=extent && a[1]+a[2]<=extent);
            for (_,b) in &pages[i+1..] {
                assert!(a[0]+a[2]<=b[0] || b[0]+b[2]<=a[0] || a[1]+a[2]<=b[1] || b[1]+b[2]<=a[1],"resident physical pages overlap");
            }
        }
    }Ok(())
}
fn lighting(r:&Renderer,material:usize)->anyhow::Result<[[f32;4];2]> {
    let allocator=r.allocator();r.flux_surface.as_ref().unwrap().debug_lighting(&r.ctx.device,
        &mut allocator.lock().unwrap(),r.command_pool,r.ctx.queue,material,4)
}
pub(super) fn lighting_jobs(r:&mut Renderer)->anyhow::Result<()> {
    r.set_area_lights(&[]);r.set_anti_aliasing(crate::AntiAliasing::None);
    let mesh=r.upload_mesh(&cube())?;
    let colors=[[0.6,0.3,0.2,1.0],[0.2,0.6,0.3,1.0],[0.3,0.2,0.6,1.0]];
    let materials=colors.map(|color|lit(r,color,[0.0;3],1.0,0.0));
    let draws:Vec<_>=[(8.0,0.0),(4.0,4.0),(0.5,8.0)].into_iter().enumerate().map(|(i,(size,x))| {
        let mut d=draw(mesh,materials[i],Mat4::from_scale_rotation_translation(Vec3::new(size,size,0.2),Quat::IDENTITY,Vec3::new(x,0.0,0.0)),-1);
        d.gi_source=Some(GiSource {object:0x71e000+i as u128,slot:0});d
    }).collect();
    let light=[crate::LightInstance {kind:crate::LightKind::Directional,position:Vec3::ZERO,direction:Vec3::NEG_Z,
        color:[1.0;3],intensity:4.0,range:0.0,spot_inner_deg:0.0,spot_outer_deg:0.0,cast_shadows:true,soft_shadows:false,
        radius:0.0,shadow_bias:0.0,shadow_clip:None,baked:false}];
    r.gi_set_mesh_sdfs(&[])?;r.gi_set_emitters(&[]);
    r.gi_set_gdf(&[100.0;512],&[0;512],[8;3],[-30.0;3],[30.0;3],&[GpuGiMaterial {albedo:[0.5;3],emission:[0.0;3],metallic:0.0,roughness:1.0}]);
    let mut input=crate::cascade_tests::input(camera(2.0));input.camera_preview=Some(camera(2.0));input.viewport_extent=Some([323,181]);
    input.lights=&light;input.light.intensity=0.0;input.light.ambient=[0.0;3];input.light.env_intensity=0.0;
    input.postfx.flux.scene_lighting_update_speed=Some(4.0);input.preview_postfx.flux.scene_lighting_update_speed=Some(4.0);
    for gi in [true,false] {for hardware in [false,true] {
        if hardware && !r.ctx.ray_tracing() {continue;}
        r.set_gi_enabled(gi);r.set_flux_settings(FluxSettings {rt_trace:hardware,reflection_mode:2,reflection_screen_traces:false,..Default::default()});
        input.draws=&draws;
        // Both histories must be full before blocks are compared: radiosity follows the
        // setup changes for 32 frames, then needs 31 updates to settle.
        for _ in 0..120 {r.render(&input)?;}
        let sizes:Vec<_>=materials.iter().map(|m|r.flux_surface.as_ref().unwrap().debug_card(m.0,4).2.rect[2]).collect();
        assert_eq!(sizes,[128,64,32],"fixture must exercise all physical page sizes simultaneously");
        let mut timing:[Vec<f32>;2]=Default::default();let mut reference=None;
        for block in 0..8 {
            let legacy=block%2==0;r.flux_surface.as_mut().unwrap().legacy_lighting_dispatch=legacy;
            for frame in 0..16 {
                r.render(&input)?;
                let work=r.flux_surface.as_ref().unwrap().lighting_workgroups;
                if legacy {assert_eq!(work[0],work[1]);} else {assert!(work[0]<work[1],"mixed sizes retained padded workgroups: {work:?}");}
                // Discard timestamp latency and warmup at each switch.
                if frame>=4 {let stats=r.stats();let ms=stats.gpu_gi_surface_ms+stats.gpu_camera_gi_surface_ms;if ms>0.0 {timing[usize::from(!legacy)].push(ms);}}
            }
            let mut pages=Vec::new();
            for (i,m) in materials.iter().enumerate() {
                let allocator=r.allocator();let (size,pixels)=r.flux_surface.as_ref().unwrap().debug_lighting_page(
                    &r.ctx.device,&mut allocator.lock().unwrap(),r.command_pool,r.ctx.queue,m.0,4)?;
                // Every interior receiver, including all tile boundaries, must
                // retain the expected direct irradiance and valid black GI.
                for y in 1..size-1 {for x in 1..size-1 {
                    let pixel=pixels[(y*size+x) as usize];
                    assert!(pixel[0][3]>0.0 && pixel[1][3]>0.0,"unwritten tile at material{i} ({x},{y}): {pixel:?}");
                    for c in 0..3 {
                        assert!((pixel[0][c]-colors[i][c]*4.0/std::f32::consts::PI).abs()<0.02,"wrong receiver at material{i} ({x},{y}): {pixel:?}");
                        assert!(pixel[1][c].abs()<0.002,"unexpected indirect energy: {pixel:?}");
                    }
                }}
                pages.push(pixels);
            }
            if let Some(previous)=&reference {assert_eq!(&pages,previous,"tile packing changed surface lighting/history");}
            else {reference=Some(pages);}
        }
        assert!(timing.iter().all(|samples|!samples.is_empty()));
        let median=|samples:&mut Vec<f32>| {samples.sort_by(f32::total_cmp);samples[samples.len()/2]};
        let old=median(&mut timing[0]);let compact=median(&mut timing[1]);
        let work=r.flux_surface.as_ref().unwrap().lighting_workgroups;
        eprintln!("[Flux lighting tiles] GI={gi} hardware={hardware}: groups {}->{}, cache {old:.4}->{compact:.4}ms, alternating blocks; every interior texel and both lighting histories identical",work[1],work[0]);
        // Both views share physical storage; a low-resolution request must
        // not degrade another active view requesting full detail.
        input.postfx.flux.surface_cache_resolution=Some(0.5);
        for _ in 0..12 {r.render(&input)?;}
        assert_eq!(r.flux_surface.as_ref().unwrap().debug_card(materials[0].0,4).2.rect[2],128);
        input.preview_postfx.flux.surface_cache_resolution=Some(0.5);
        for _ in 0..24 {r.render(&input)?;}
        assert_eq!(r.flux_surface.as_ref().unwrap().debug_card(materials[0].0,4).2.rect[2],64);
        let energy=lighting(r,materials[0].0)?;
        assert!((energy[0][0]-colors[0][0]*4.0/std::f32::consts::PI).abs()<0.02,"resolution control changed physical lighting");
        input.postfx.flux.surface_cache_resolution=None;
        for _ in 0..24 {r.render(&input)?;}
        assert_eq!(r.flux_surface.as_ref().unwrap().debug_card(materials[0].0,4).2.rect[2],128);
        input.preview_postfx.flux.surface_cache_resolution=None;
        eprintln!("[Flux surface resolution control] GI={gi} hardware={hardware}: independent view overrides, shared maximum and128->64->128 preserve lighting");
        input.draws=&[];r.render(&input)?;
    }}
    r.flux_surface.as_mut().unwrap().legacy_lighting_dispatch=false;
    r.set_gi_enabled(true);Ok(())
}
pub(super) fn run(r:&mut Renderer)->anyhow::Result<()> {
    let old_gi=r.gi_enabled;let old_settings=r.flux_settings;
    r.set_gi_enabled(false);r.set_area_lights(&[]);r.set_anti_aliasing(crate::AntiAliasing::None);
    let mesh=r.upload_mesh(&cube())?;let material=lit(r,[0.6,0.3,0.2,1.0],[0.0;3],1.0,0.0);
    let mut receiver=draw(mesh,material,Mat4::from_scale_rotation_translation(Vec3::new(8.0,8.0,0.2),Quat::IDENTITY,Vec3::ZERO),-1);
    receiver.gi_source=Some(GiSource {object:0xada000,slot:0});
    let light=[crate::LightInstance {kind:crate::LightKind::Directional,position:Vec3::ZERO,direction:Vec3::NEG_Z,color:[1.0;3],intensity:4.0,
        range:0.0,spot_inner_deg:0.0,spot_outer_deg:0.0,cast_shadows:true,soft_shadows:false,radius:0.0,shadow_bias:0.0,shadow_clip:None,baked:false}];
    let neutral=GpuGiMaterial {albedo:[0.5;3],emission:[0.0;3],roughness:1.0,metallic:0.0};
    r.gi_set_gdf(&[100.0;512],&[0;512],[8;3],[-100.0;3],[100.0;3],&[neutral]);
    for hardware in [false,true] {
        if hardware && !r.ctx.ray_tracing() {continue;}
        r.set_flux_settings(FluxSettings {rt_trace:hardware,reflection_mode:2,reflection_screen_traces:false,..Default::default()});
        let draws=[receiver];let far=camera(40.0);let near=camera(2.0);
        let mut input=crate::cascade_tests::input(far);input.camera_preview=Some(far);input.viewport_extent=Some([323,181]);
        input.draws=&draws;input.lights=&light;input.light.intensity=0.0;input.light.ambient=[0.0;3];input.light.env_intensity=0.0;
        step(r,&input,20)?;
        let before=r.flux_surface.as_ref().unwrap().debug_card(material.0,4);let energy=lighting(r,material.0)?;
        assert_eq!(before.2.rect[2],32);assert!(energy[0][0]>0.5,"fixture must have cached direct lighting: {energy:?}");
        input.camera=near;let mut migrated=false;
        for _ in 0..8 {
            step(r,&input,1)?;let cache=r.flux_surface.as_ref().unwrap();
            let current=cache.debug_card(material.0,4);assert_eq!(current.0,before.0,"resolution changed logical identity");
            if current.2.rect[2]==128 {
                migrated=true;let now=lighting(r,material.0)?;
                for c in 0..3 {assert!((now[0][c]-energy[0][c]).abs()<0.025,"resolution change lost physical lighting: {energy:?}->{now:?}");}
            }
        }
        assert!(migrated,"near view did not request detail");
        let near_card=r.flux_surface.as_ref().unwrap().debug_card(material.0,4);
        input.camera=far;input.camera_preview=Some(near);step(r,&input,12)?;
        let both=r.flux_surface.as_ref().unwrap().debug_card(material.0,4);
        assert_eq!(both.2.rect[2],128,"Camera demand lost to distant Scene view");assert_eq!(both.1,near_card.1,"view swap recaptured stable detail");
        input.camera_preview=Some(far);step(r,&input,12)?;
        let coarse=r.flux_surface.as_ref().unwrap().debug_card(material.0,4);
        assert_eq!(coarse.2.rect[2],32);assert_eq!(coarse.0,before.0);
        let now=lighting(r,material.0)?;assert!((now[0][0]-energy[0][0]).abs()<0.025);
        // Force atlas growth without touching the original receiver. Growing
        // storage must preserve its pixels, generation and last capture time.
        let extent=r.flux_surface.as_ref().unwrap().debug_pages().0;
        let mut many=vec![receiver];
        for i in 0..80 {
            let mut d=draw(mesh,material,Mat4::from_translation(Vec3::new(15.0+(i%10) as f32*2.0,(i/10) as f32*2.0,0.0)),-1);
            d.gi_source=Some(GiSource {object:0xada001+i as u128,slot:0});many.push(d);
        }
        input.draws=&many;step(r,&input,1)?;
        let cache=r.flux_surface.as_ref().unwrap();let after=cache.debug_card(material.0,4);
        if !hardware {assert!(cache.debug_pages().0>extent,"fixture did not grow atlas");}
        assert_eq!((after.0,after.1,after.2.rect),(coarse.0,coarse.1,coarse.2.rect),"atlas growth discarded or moved resident card");
        assert!((lighting(r,material.0)?[0][0]-energy[0][0]).abs()<0.025,"atlas growth discarded lighting history");
        input.draws=&[];step(r,&input,1)?;
        eprintln!("[Flux adaptive pages] hardware={hardware}: 32->128->32, shared Scene/Camera demand, stable identity, preserved radiance, atlas growth and texel budgets passed");
    }
    r.set_gi_enabled(old_gi);r.set_flux_settings(old_settings);Ok(())
}

/// Real capture/lighting integration, including partial fine-page arrivals.
pub(super) fn virtual_pages(r:&mut Renderer)->anyhow::Result<()> {
    r.set_area_lights(&[]);r.set_anti_aliasing(crate::AntiAliasing::None);
    let mesh=r.upload_mesh(&cube())?;let color=[0.6,0.3,0.2,1.0];
    let material=lit(r,color,[0.0;3],1.0,0.0);
    let mut receiver=draw(mesh,material,Mat4::from_scale(Vec3::new(32.0,32.0,0.2)),-1);
    receiver.gi_source=Some(GiSource {object:0xfee000,slot:0});
    let light=[crate::LightInstance {kind:crate::LightKind::Directional,position:Vec3::ZERO,direction:Vec3::NEG_Z,
        color:[1.0;3],intensity:4.0,range:0.0,spot_inner_deg:0.0,spot_outer_deg:0.0,cast_shadows:true,
        soft_shadows:false,radius:0.0,shadow_bias:0.0,shadow_clip:None,baked:false}];
    r.gi_set_mesh_sdfs(&[])?;r.gi_set_emitters(&[]);
    r.gi_set_gdf(&[100.0;512],&[0;512],[8;3],[-100.0;3],[100.0;3],&[GpuGiMaterial {albedo:[0.5;3],emission:[0.0;3],metallic:0.0,roughness:1.0}]);
    let mut near=camera(2.0);near.proj=Mat4::perspective_rh(3.0,16.0/9.0,0.1,200.0);
    let draws=[receiver];let aliases=[receiver,receiver];
    let moved_model=Mat4::from_translation(Vec3::new(0.3,0.2,0.1))*Mat4::from_rotation_z(0.15)*receiver.transform;
    let mut moved_receiver=receiver;moved_receiver.transform=moved_model;let moved=[moved_receiver];
    let mut input=crate::cascade_tests::input(near);input.camera_preview=Some(near);input.viewport_extent=Some([323,181]);
    input.lights=&light;input.light.intensity=0.0;input.light.ambient=[0.0;3];input.light.env_intensity=0.0;
    let check=|r:&Renderer,expected:[f32;4],model:Mat4,stage:&str|->anyhow::Result<()> {
        let cache=r.flux_surface.as_ref().unwrap();let cards=cache.debug_virtual_cards(material.0,4);
        eprintln!("[Flux virtual stage] {stage}: page counts by face={:?}, total={}",
            (0..6).map(|face|cache.debug_virtual_cards(material.0,face).len()).collect::<Vec<_>>(),cache.debug_pages().1.len());
        assert_eq!(cards.len(),21,"{stage}: expected coarse+four256-chart tiles+sixteen512-chart tiles");
        let mut levels=[0;3];let mut root=None;let mut points=Vec::new();
        for (_,card) in cards {
            let vp=Mat4::from_cols_array_2d(&card.view_proj);
            let depth=vp.transform_point3(model.transform_point3(Vec3::new(0.0,0.0,0.5))).z;
            // Test chart centres and every logical edge/corner through the
            // production filter, using captured material and lighting images.
            for y in [-0.9375,0.0,0.9375] {for x in [-0.9375,0.0,0.9375] {
                points.push((card,vp.inverse().transform_point3(Vec3::new(x,y,depth))));
            }}
            levels[card.virtual_page[1] as usize]+=1;
            assert_eq!(*root.get_or_insert(card.virtual_page[0]),card.virtual_page[0]);
            assert_eq!(card.rect[3]&36,36,"fine page lighting has not become ready");
            let allocator=r.allocator();let (size,pixels)=cache.debug_page_lighting(&r.ctx.device,
                &mut allocator.lock().unwrap(),r.command_pool,r.ctx.queue,card)?;
            for y in 1..size-1 {for x in 1..size-1 {
                let value=pixels[(y*size+x) as usize];
                assert!(value[0][3]>0.0 && value[1][3]>0.0,"lighting hole at mip{} tile{:?}, ({x},{y}): {value:?}",card.virtual_page[1],&card.virtual_page[2..]);
                for c in 0..3 {assert!((value[0][c]-expected[c]*4.0/std::f32::consts::PI).abs()<0.025,"wrong virtual-page direct light: {value:?}");
                    assert!(value[1][c].abs()<0.002,"unexpected bounced energy: {value:?}");}
            }}
        }
        assert_eq!(levels,[1,4,16]);
        let allocator=r.allocator();let samples=cache.debug_virtual_samples(&r.ctx.device,
            &mut allocator.lock().unwrap(),r.command_pool,r.ctx.queue,&points);
        for ((card,position),value) in points.iter().zip(samples) {
            assert_eq!(value[3],1.0,"{stage}: invalid sample at {position:?}, page{:?}: {value:?}",card.virtual_page);
            for c in 0..3 {assert!(value[c].is_finite() && (value[c]-expected[c]*4.0/std::f32::consts::PI).abs()<0.025,
                "{stage}: wrong filtered light at {position:?}, page{:?}: {value:?}",card.virtual_page);}
        }
        Ok(())
    };
    for gi in [true,false] {for hardware in [false,true] {
        if hardware && !r.ctx.ray_tracing() {continue;}
        r.set_gi_enabled(gi);r.set_flux_settings(FluxSettings {rt_trace:hardware,reflection_mode:2,reflection_screen_traces:false,..Default::default()});
        r.material_params_mut(material).base_color=color;
        input.draws=&draws;input.camera=near;input.camera_preview=Some(near);
        let mut coarse_id=None;let mut partial=false;
        for _ in 0..48 {
            step(r,&input,1)?;
            let cache=r.flux_surface.as_ref().unwrap();let cards=cache.debug_virtual_cards(material.0,4);
            if let Some((id,_))=cards.iter().find(|(_,c)|c.virtual_page[1]==0) {
                assert_eq!(*coarse_id.get_or_insert(*id),*id,"fine arrivals evicted coarse coverage");
            }
            if cards.len()>1 && cards.len()<21 {partial=true;}
        }
        assert!(partial,"fixture did not exercise budgeted partial arrivals");check(r,color,receiver.transform,"initial")?;
        // A material response change must reach every mip; inherited history
        // cannot lock fine tiles to their old diffuse response.
        let changed=[0.2,0.5,0.4,1.0];r.material_params_mut(material).base_color=changed;
        step(r,&input,40)?;check(r,changed,receiver.transform,"material")?;
        // Rigid attachment must keep logical identity and sampling aligned,
        // including pages whose captures were recorded in different poses.
        let before_move=r.flux_surface.as_ref().unwrap().debug_virtual_cards(material.0,4);
        input.draws=&moved;step(r,&input,24)?;check(r,changed,moved_model,"rigid move")?;
        let after_move=r.flux_surface.as_ref().unwrap().debug_virtual_cards(material.0,4);
        for (id,card) in &before_move {
            assert!(after_move.iter().any(|(next,c)|next==id && c.virtual_page[..3]==card.virtual_page[..3]),
                "rigid motion replaced a resident virtual page");
        }
        input.draws=&draws;step(r,&input,24)?;check(r,changed,receiver.transform,"rigid return")?;
        // Distance removes fine storage while preserving the logical coarse
        // page. Camera demand alone must retain detail when Scene moves away.
        input.camera=camera(80.0);step(r,&input,8)?;
        assert_eq!(r.flux_surface.as_ref().unwrap().debug_virtual_cards(material.0,4).len(),21);
        input.camera_preview=Some(camera(80.0));step(r,&input,16)?;
        let cards=r.flux_surface.as_ref().unwrap().debug_virtual_cards(material.0,4);
        assert_eq!(cards.len(),1,"fine pages retained without projected demand");assert_eq!(cards[0].0,coarse_id.unwrap());
        assert!((lighting(r,material.0)?[0][0]-changed[0]*4.0/std::f32::consts::PI).abs()<0.025);
        input.camera=near;input.camera_preview=Some(near);step(r,&input,40)?;check(r,changed,receiver.transform,"distance")?;
        // Physical slot pressure may discard detail, never the selected
        // coarse chart. Identical draw aliases share the same tile requests.
        // Hardware ray feedback can also request the opposite chart. Restore
        // its measured full working set, rather than assuming only one face
        // can legitimately receive fine-page demand.
        let full_residency_count=r.flux_surface.as_ref().unwrap().debug_pages().1.len();
        r.flux_surface.as_mut().unwrap().residency_capacity_override=Some(8);
        input.draws=&aliases;step(r,&input,8)?;
        let cards=r.flux_surface.as_ref().unwrap().debug_virtual_cards(material.0,4);
        let cache=r.flux_surface.as_ref().unwrap();
        assert_eq!(cache.debug_pages().1.len(),8,"six coarse cards must leave exactly two fine slots");
        let mut fine_count=0;
        for face in 0..6 {
            let pages=cache.debug_virtual_cards(material.0,face);
            assert_eq!(pages.iter().filter(|(_,c)|c.virtual_page[1]==0).count(),1,"pressure evicted coarse face{face}");
            fine_count+=pages.iter().filter(|(_,c)|c.virtual_page[1]>0).count();
        }
        assert_eq!(fine_count,2,"pressure must retain exactly two captured fine pages across the requested faces");
        assert_eq!(cards.iter().find(|(_,c)|c.virtual_page[1]==0).unwrap().0,coarse_id.unwrap());
        assert!((lighting(r,material.0)?[0][0]-changed[0]*4.0/std::f32::consts::PI).abs()<0.025);
        r.flux_surface.as_mut().unwrap().residency_capacity_override=Some(full_residency_count);
        step(r,&input,40)?;check(r,changed,receiver.transform,"pressure")?;
        r.flux_surface.as_mut().unwrap().residency_capacity_override=None;
        let black=[0.0,0.0,0.0,1.0];r.material_params_mut(material).base_color=black;
        step(r,&input,40)?;check(r,black,receiver.transform,"zero")?;
        let before=r.flux_surface.as_ref().unwrap().debug_lighting_updates(material.0,4).0;
        step(r,&input,16)?;
        assert_eq!(r.flux_surface.as_ref().unwrap().debug_lighting_updates(material.0,4).0,before,
            "settled zero diffuse pages consumed recurring lighting work");
        r.material_params_mut(material).base_color=changed;step(r,&input,40)?;check(r,changed,receiver.transform,"restore")?;
        eprintln!("[Flux virtual pages] GI={gi} hardware={hardware}:512-chart captures, partial arrivals, full-page lighting, material updates, Camera demand, coarse fallback, pressure eviction, alias deduplication, zero-response scheduling and re-admission passed");
        input.draws=&[];step(r,&input,1)?;
    }}
    r.set_gi_enabled(true);Ok(())
}
