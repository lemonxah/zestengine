//! End-to-end Flux tests: actual depth, gather, integrate and forward passes in
//! both editor views. Run with a display and `--test-threads=1 --nocapture`.
use std::sync::Arc;

use glam::{Mat4, Vec3};
use winit::{application::ApplicationHandler, event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop}, window::{Window, WindowId}};
use winit::platform::{run_on_demand::EventLoopExtRunOnDemand, x11::EventLoopBuilderExtX11};

use crate::{CameraData, FluxSettings, GpuGiMaterial, Renderer};
use super::planar::{cube, draw, lit};

#[derive(Default)]
struct App { done: bool, error: Option<String>, backface_only: bool }

fn patch(image: &(u32, u32, Vec<u8>), camera: CameraData, point: Vec3) -> f32 {
    let (w, h, pixels) = image;
    let p = (camera.proj * camera.view).project_point3(point);
    let x = ((p.x * 0.5 + 0.5) * *w as f32) as i32;
    let y = ((0.5 - p.y * 0.5) * *h as f32) as i32;
    let mut sum = 0.0;
    for dy in -2..=2 {
        for dx in -2..=2 {
            assert!((0..*w as i32).contains(&(x + dx)) && (0..*h as i32).contains(&(y + dy)));
            sum += pixels[(((y + dy) as u32 * w + (x + dx) as u32) * 4) as usize] as f32;
        }
    }
    sum / 25.0
}

fn transfer_probe_history(r: &Renderer, buffer: ash::vk::Buffer, size: usize,
    data: Option<&[u8]>) -> anyhow::Result<Vec<u8>> {
    use ash::vk;
    let device = &r.ctx.device;
    let allocator = r.allocator();
    let mut host = crate::alloc::Buffer::new(device, &mut allocator.lock().unwrap(), size as u64,
        vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
        if data.is_some() { gpu_allocator::MemoryLocation::CpuToGpu }
            else { gpu_allocator::MemoryLocation::GpuToCpu }, "Flux history regression")?;
    if let Some(bytes) = data { host.write(0, bytes); }
    crate::alloc::one_time_submit(device, r.command_pool, r.ctx.queue, |cb| unsafe {
        let before = [vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE)];
        device.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(), &before, &[], &[]);
        let (src, dst) = if data.is_some() { (host.handle, buffer) } else { (buffer, host.handle) };
        device.cmd_copy_buffer(cb, src, dst, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: size as u64 }]);
        let after = [vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)];
        device.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(), &after, &[], &[]);
    })?;
    let bytes = host.read();
    host.destroy(device, &mut allocator.lock().unwrap());
    Ok(bytes)
}

pub(super) fn cube_distance_field() -> Arc<crate::sdf::SdfVolume> {
        let mut values=Vec::new();
        for z in 0..24 {for y in 0..24 {for x in 0..24 {
            let p=Vec3::splat(-0.6)+Vec3::new(x as f32,y as f32,z as f32)*(1.2/23.0);
            let q=p.abs()-Vec3::splat(0.5);
            values.push(q.max(Vec3::ZERO).length()+q.max_element().min(0.0));
        }}}
        Arc::new(crate::sdf::SdfVolume {dims:[24;3],min:Vec3::splat(-0.6),max:Vec3::splat(0.6),geometry_bounds:[Vec3::splat(-0.5),Vec3::splat(0.5)],data:values})

}

impl App {
    fn run(&mut self, el: &ActiveEventLoop) -> anyhow::Result<()> {
        let window = Arc::new(el.create_window(Window::default_attributes()
            .with_visible(false).with_inner_size(winit::dpi::LogicalSize::new(320, 180)))?);
        let mut r = Renderer::new(window)?;
        r.set_vsync(false);
        r.set_gpu_zone_timing(true);
        if self.backface_only { return self.backface_occlusion(&mut r); }
        let mesh = r.upload_mesh(&cube())?;
        let mat = lit(&mut r, [0.7, 0.7, 0.7, 1.0], [0.0; 3], 1.0, 0.0);
        let draws = [draw(mesh, mat, Mat4::from_translation(Vec3::new(2.0, 0.0, 0.0)), -1)];
        // Empty software field: an unoccluded surface must see constant sky.
        // RT uses the actual cube; outward rays see the same sky.
        r.gi_set_gdf(&vec![10.0; 8*8*8], &vec![0; 8*8*8], [8; 3], [-10.0; 3], [10.0; 3],
            &[GpuGiMaterial { albedo: [0.7; 3], emission: [0.0; 3], metallic: 0.0, roughness: 1.0 }]);
        anyhow::ensure!(r.gi_has_gdf(), "GPU GI failed to initialize");
        for (rt, detail) in [(false,false), (false,true), (true,false)] {
            if rt && !r.ctx.ray_tracing() {
                eprintln!("[Flux test] hardware ray query unavailable; RT skipped");
                continue;
            }
            r.gi_set_gdf(&vec![10.0;512], &vec![0;512], [8;3], [-10.0;3], [10.0;3],
                &[GpuGiMaterial {albedo:[0.7;3],emission:[0.0;3],metallic:0.0,roughness:1.0}]);
            r.gi_set_emitters(&[]);
            r.gi_set_mesh_sdfs(&[])?;
            if detail {
                r.gi_set_mesh_sdfs(&[crate::GiMeshSdf {
                    gi_source: None,
                    world_to_local:draws[0].transform.inverse(),volume:cube_distance_field(),
                    material:GpuGiMaterial {albedo:[0.7;3],emission:[0.0;3],metallic:0.0,roughness:1.0},
                    in_global_field:false,
                }])?;
            }
            r.set_flux_settings(FluxSettings { rt_trace: rt, samples: 8, reflection_mode: 0,
                ssr_enabled: false, ..Default::default() });
            for size in [[320, 180], [323, 181]] {
                let eye = Vec3::new(0.0, 0.0, 5.0);
                let camera = CameraData { position: eye, view: Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y),
                    proj: Mat4::perspective_rh(1.0, size[0] as f32 / size[1] as f32, 0.1, 100.0) };
                let preview = CameraData { proj: Mat4::perspective_rh(1.0, 16.0/9.0, 0.1, 100.0), ..camera };
                let mut input = crate::cascade_tests::input(camera);
                input.viewport_extent = Some(size);
                input.camera_preview = Some(preview);
                input.draws = &draws;
                input.light.intensity = 0.0;
                input.light.ambient = [0.3, 0.1, 0.05];
                input.gi_debug = 6; // isolate Flux, excluding direct lighting and ambient fallback
                input.postfx.bloom_enabled = false;
                input.preview_postfx.bloom_enabled = false;
                for frame in 0..3 {
                    r.render(&input)?;
                    if !rt && size == [320, 180] && frame == 0 {
                        // Retire live placeholder resources while Scene/Camera
                        // submissions are in flight. Validation must catch any
                        // forgotten descriptor in disabled effects or probes.
                        r.set_baked_lightmaps(&[]);
                        r.set_skybox(None)?;
                    }
                }
                let scene = r.debug_read_viewport()?.expect("Scene target");
                let game = r.debug_read_camera_preview()?.expect("Camera target");
                assert_eq!([scene.0, scene.1], size);
                let point = Vec3::new(2.0, 0.0, 0.5);
                let a = patch(&scene, camera, point);
                let b = patch(&game, preview, point);
                eprintln!("[Flux test] RT={rt}, size={size:?}: Scene={a:.1}, Camera={b:.1}");
                anyhow::ensure!(a > 50.0 && b > 50.0, "missing sky GI: RT={rt}, size={size:?}, Scene={a}, Camera={b}");
                anyhow::ensure!((a-b).abs() < 25.0, "Scene/Camera GI mismatch: {a} vs {b}");
                if detail { self.adaptive_placement(&r)?; }
                self.gather_budget(&mut r, &mut input)?;
                self.history_identity(&mut r, &mut input)?;
                if detail && size==[323,181] {self.importance_budget(&mut r,&mut input)?;}
                self.temporal_response(&mut r, &mut input)?;
                if size == [323, 181] {
                    r.set_anti_aliasing(crate::AntiAliasing::Taa);
                    for _ in 0..8 { r.render(&input)?; }
                    let jittered = Mat4::from_cols_array_2d(&r.last_frame_ubo.as_ref().unwrap().view_proj);
                    assert_ne!(jittered, camera.proj * camera.view, "TAA must jitter this regression's depth");
                    assert_eq!(r.sgi_last_view_proj, Some(jittered), "GI history must match the depth projection");
                    let taa = patch(&r.debug_read_viewport()?.unwrap(), camera, point);
                    anyhow::ensure!((taa-b).abs() < 25.0, "jittered Scene/Camera GI mismatch: {taa} vs {b}");
                    r.set_anti_aliasing(crate::AntiAliasing::None);
                }
                // Resize dragging retains the old target temporarily. Its
                // actual dimensions must still drive GI sampling and jitter.
                r.set_viewport_resize_hold(true);
                input.viewport_extent = Some([size[0] + 100, size[1] + 50]);
                r.render(&input)?;
                let actual = r.viewport_target.as_ref().unwrap().extent;
                let ubo = r.last_frame_ubo.as_ref().unwrap();
                assert_eq!(ubo.postfx2[3], actual.width as f32);
                assert_eq!(ubo.postfx3[3], actual.height as f32);
                r.set_viewport_resize_hold(false);
                input.viewport_extent = Some(size);
                if size == [320, 180] {
                    self.gi_timing_views(&mut r, &mut input)?;
                    // Cross the old idle cutoff, then change lighting without
                    // moving the camera or invoking an invalidation helper.
                    input.camera_preview = None;
                    for _ in 0..125 { r.render(&input)?; }
                    input.light.ambient = [0.0; 3];
                    for _ in 0..100 { r.render(&input)?; }
                    let dark = patch(&r.debug_read_viewport()?.unwrap(), camera, point);
                    eprintln!("[Flux test] RT={rt}: stationary camera after sky off={dark:.1}");
                    anyhow::ensure!(dark < 25.0, "stationary view retained stale GI: {dark}");
                }
            }
            r.gi_set_mesh_sdfs(&[])?;
            self.bounce(&mut r, rt, !detail)?;
        }
        self.surface_materials(&mut r)?;
        for rt in [false,true] { if !rt || r.ctx.ray_tracing() { self.animated_surfaces(&mut r,rt)?; } }
        self.shadow_budget(&mut r)?;
        if r.ctx.ray_tracing() { self.reflected_bounce(&mut r)?; }
        self.mesh_distance_fields(&mut r)?;
        self.single_sided_panel(&mut r)?;
        self.open_emissive_frame(&mut r)?;
        super::flux_attachment::run(&mut r)?;
        super::flux_screen_history::run(&mut r)?;
        super::exposure::run(&mut r)?;
        super::reflection_denoise::run(&mut r)?;
        super::reflection_mask::run(&mut r)?;
        super::flux_light_falloff::run(&mut r)?;
        super::flux_area_light::run(&mut r)?;
        super::flux_ray_lighting::run(&mut r)?;
        super::flux_hit_material::run(&mut r)?;
        super::flux_radiance::run(&mut r)?;
        self.backface_occlusion(&mut r)?;
        Ok(())
    }

    // An opaque panel must block the same incoming light from either winding.
    // Seeing its back from a valid floor receiver does not put that receiver
    // inside geometry, and must not discard its other hemisphere directions.
    fn backface_occlusion(&self,r:&mut Renderer)->anyhow::Result<()> {
        if !r.ctx.ray_tracing() {return Ok(());}
        r.set_gi_enabled(true);r.set_area_lights(&[]);r.set_anti_aliasing(crate::AntiAliasing::None);
        r.gi_set_gdf(&vec![10.0;512],&vec![0;512],[8;3],[-20.0;3],[20.0;3],
            &[GpuGiMaterial {albedo:[0.7;3],emission:[0.0;3],metallic:0.0,roughness:1.0}]);
        r.gi_set_mesh_sdfs(&[])?;r.gi_set_emitters(&[]);
        let saved=r.flux_settings;
        let gi=r.gpu_gi.as_mut().unwrap();
        let saved_rc=gi.radiance_cache.enabled_override;let saved_screen=gi.screen_trace_override;
        let saved_alpha=gi.probe_history_alpha;
        gi.radiance_cache.enabled_override=Some(false);gi.screen_trace_override=Some(false);gi.probe_history_alpha=Some(1.0);
        let mesh=r.upload_mesh(&cube())?;
        let grey=lit(r,[0.7;4],[0.0;3],1.0,0.0);let black=lit(r,[0.0,0.0,0.0,1.0],[0.0;3],1.0,0.0);
        let floor=draw(mesh,grey,Mat4::from_scale_rotation_translation(Vec3::new(6.0,0.1,6.0),glam::Quat::IDENTITY,Vec3::new(0.0,-0.05,0.0)),-1);
        let mut energies=Vec::new();
        // The low panel covers about a quarter of the cosine hemisphere:
        // jitter used to push it across the whole-probe rejection threshold.
        for (height,facing) in [(20.0,-1.0f32),(20.0,1.0),(0.9,-1.0),(0.9,1.0)] {
            let vertices=[Vec3::new(0.5,0.0,-20.0),Vec3::new(0.5,height,-20.0),Vec3::new(0.5,height,20.0),Vec3::new(0.5,0.0,20.0)]
                .map(|p|crate::Vertex {position:p.to_array(),normal:[facing,0.0,0.0],..Default::default()}).to_vec();
            let wall=r.upload_mesh(&crate::MeshData {vertices,indices:if facing>0.0 {vec![0,1,2,0,2,3]} else {vec![0,2,1,0,3,2]},has_lightmap_uv:false})?;
            let draws=[floor,draw(wall,black,Mat4::IDENTITY,-1)];
            let eye=Vec3::new(-0.2,2.0,3.0);
            let camera=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0)};
            let mut input=crate::cascade_tests::input(camera);input.viewport_extent=Some([323,181]);input.camera_preview=Some(camera);
            input.draws=&draws;input.light.ambient=[0.3;3];input.light.intensity=0.0;input.light.env_intensity=0.0;input.gi_debug=6;
            input.postfx.bloom_enabled=false;input.preview_postfx.bloom_enabled=false;
            for samples in [16,64] {
                r.set_flux_settings(FluxSettings {rt_trace:true,samples,bounces:1,reflection_mode:0,ssr_enabled:false,..Default::default()});
                for _ in 0..24 {r.render(&input)?;}
                let mut total=0;let mut invalid=0;let mut incoming=0.0;
                for _ in 0..8 {
                    r.render(&input)?;
                    for (target,parity) in [(&r.viewport_target.as_ref().unwrap().sgi,r.sgi_parity),(&r.camera_preview.as_ref().unwrap().sgi,r.preview_sgi_parity)] {
                        let bytes=transfer_probe_history(r,target.sh_buf[(parity^1)&1].handle,
                            (target.probe_extent.width*target.probe_extent.height) as usize*crate::gpu_gi::SCREEN_PROBE_BYTES as usize,None)?;
                        for probe in bytes.chunks_exact(crate::gpu_gi::SCREEN_PROBE_BYTES as usize) {
                            let f=|offset|f32::from_ne_bytes(probe[offset..offset+4].try_into().unwrap());
                            if f(68)*f(68)+f(72)*f(72)+f(76)*f(76)<1e-6 {continue;} // cleared sky/padding
                            if f(68).abs()>0.2 || f(72).abs()>0.001 || f(76).abs()>0.2 {continue;}
                            total+=1;if f(12)<=0.0 {invalid+=1;}
                            for bin in 0..16 {incoming+=f(80+bin*16)/16.0;}
                        }
                    }
                }
                let energy=[patch(&r.debug_read_viewport()?.unwrap(),camera,Vec3::ZERO),patch(&r.debug_read_camera_preview()?.unwrap(),camera,Vec3::ZERO)];
                incoming/=total.max(1) as f32;
                eprintln!("[Flux backface receiver] height={height} facing={facing} samples={samples} invalid={invalid}/{total} incoming={incoming} energy={energy:?}");
                anyhow::ensure!(total>20 && invalid==0,"opaque back faces discarded valid floor probes: {invalid}/{total}");
                let expected=if height>1.0 {0.15} else {0.225};
                anyhow::ensure!((incoming-expected).abs()<0.025,"panel sky occlusion incorrect: {incoming}, expected ~{expected}");
                anyhow::ensure!(energy.iter().all(|v|*v>30.0),"unblocked hemisphere lost: {energy:?}");
                energies.push(energy);
            }
        }
        for i in [0,1,4,5] {for view in 0..2 {
            anyhow::ensure!((energies[i][view]-energies[i+2][view]).abs()<12.0,"panel winding changed occlusion: {energies:?}");
        }}
        let gi=r.gpu_gi.as_mut().unwrap();gi.radiance_cache.enabled_override=saved_rc;
        gi.screen_trace_override=saved_screen;gi.probe_history_alpha=saved_alpha;r.set_flux_settings(saved);
        Ok(())
    }

    fn gi_timing_views(&self,r:&mut Renderer,input:&mut crate::FrameInput<'_>)->anyhow::Result<()> {
        if r.gpu_ts_period<=0.0 {return Ok(());}
        let extent=input.viewport_extent; let render=input.render_viewport;
        for camera_only in [false,true] {
            input.viewport_extent=if camera_only {None} else {extent};
            input.render_viewport=if camera_only {false} else {render};
            for _ in 0..5 {r.render(input)?;}
            let s=r.stats();
            let camera_sub=s.gpu_camera_gi_gather_ms+s.gpu_camera_gi_denoise_ms
                +s.gpu_camera_gi_integrate_ms+s.gpu_camera_gi_surface_ms;
            eprintln!("[Flux timing] Camera only={camera_only}: SceneGI={} CameraGI={} gather={} cache={}",
                s.gpu_gi_ms,s.gpu_camera_gi_ms,s.gpu_camera_gi_gather_ms,s.gpu_camera_gi_surface_ms);
            anyhow::ensure!(s.gpu_camera_gi_gather_ms>0.0 && s.gpu_camera_gi_ms>0.0
                && camera_sub<=s.gpu_camera_gi_ms+0.01 && s.gpu_camera_gi_ms<=s.gpu_cam_preview_ms+0.01,
                "Camera GI timestamp zones missing, overwritten or outside parent");
            if camera_only {anyhow::ensure!(s.gpu_gi_ms==0.0 && s.gpu_gi_gather_ms==0.0,"hidden Scene retained stale GI timings");}
            else {anyhow::ensure!(s.gpu_gi_ms>0.0 && s.gpu_gi_gather_ms>0.0
                && s.gpu_gi_ms+s.gpu_camera_gi_ms<=s.gpu_frame_ms+0.01,"both GI views must have independent nested timestamps");}
        }
        input.viewport_extent=extent;input.render_viewport=render;
        Ok(())
    }

    // Read actual GPU placement, not a CPU copy of the selection algorithm.
    // Both uniform work reduction and silhouette/normal refinement must occur.
    fn adaptive_placement(&self,r:&Renderer)->anyhow::Result<()> {
        let stride=crate::gpu_gi::SCREEN_PROBE_BYTES as usize;
        for (name,target,parity) in [
            ("Scene",&r.viewport_target.as_ref().unwrap().sgi,r.sgi_parity),
            ("Camera",&r.camera_preview.as_ref().unwrap().sgi,r.preview_sgi_parity),
        ] {
            let count=target.probe_extent.width as usize*target.probe_extent.height as usize;
            let bytes=transfer_probe_history(r,target.sh_buf[(parity^1)&1].handle,count*stride,None)?;
            let f=|i:usize,offset:usize| f32::from_ne_bytes(bytes[i*stride+offset..i*stride+offset+4].try_into().unwrap());
            let owner=|i:usize| u32::from_ne_bytes(bytes[i*stride+444..i*stride+448].try_into().unwrap()) as usize;
            let normal=|i:usize| Vec3::new(f(i,28),f(i,44),f(i,60));
            let pos=|i:usize| Vec3::new(f(i,68),f(i,72),f(i,76));
            let(mut valid,mut traced,mut fine,mut coarse)=(0,0,0,0);
            for i in 0..count {
                if f(i,12)<=0.0 {continue;}
                valid+=1; let o=owner(i);
                anyhow::ensure!(o<count && f(o,12)>0.0 && owner(o)==o,"{name}: invalid trace owner");
                anyhow::ensure!(normal(i).dot(normal(o))>0.995 && (pos(i)-pos(o)).dot(normal(i)).abs()<0.02,
                    "{name}: interpolation crosses a geometry edge");
                traced+=usize::from(o==i); fine+=usize::from(f(i,0)==1.0); coarse+=usize::from(f(i,0)==4.0);
            }
            let jobs=transfer_probe_history(r,target.raw_rays.handle,16+count*17*16,None)?;
            let u=|offset:usize| u32::from_ne_bytes(jobs[offset..offset+4].try_into().unwrap()) as usize;
            anyhow::ensure!(u(0)==traced && u(4)==traced.div_ceil(64) && u(8)==1 && u(12)==16,
                "{name}: indirect work count differs from selected trace owners");
            let mut queued=std::collections::HashSet::new();
            for job in 0..traced {
                let o=u(16+count*16*16+job*16+12);
                anyhow::ensure!(o<count && owner(o)==o && f(o,12)>0.0 && queued.insert(o),
                    "{name}: duplicate or invalid compacted trace job");
            }
            eprintln!("[Flux adaptive] {name}: {traced}/{valid} traced, {fine} fine, {coarse} coarse receivers");
            anyhow::ensure!(valid>20 && traced*10<valid*7 && fine>0 && coarse>0,
                "{name}: adaptive placement failed to reduce work or preserve edges");
        }
        Ok(())
    }

    // No GDF geometry and no analytic light proxies: only the uploaded mesh
    // fields can find the emissive surface and the moving thin occluder.
    fn mesh_distance_fields(&self, r: &mut Renderer) -> anyhow::Result<()> {
        let mesh=r.upload_mesh(&cube())?;
        let grey=lit(r,[0.7,0.7,0.7,1.0],[0.0;3],1.0,0.0);
        let glow=lit(r,[0.0,0.0,0.0,1.0],[8.0,0.2,0.2],1.0,0.0);
        r.set_material_features(glow,crate::MaterialFeatures {emission:true,..Default::default()});
        let material=GpuGiMaterial {albedo:[0.7;3],emission:[0.0;3],metallic:0.0,roughness:1.0};
        let emissive=GpuGiMaterial {albedo:[0.0;3],emission:[8.0,0.2,0.2],..material};
        let volume=cube_distance_field();
        let floor=Mat4::from_scale_rotation_translation(Vec3::new(6.0,0.2,6.0),glam::Quat::IDENTITY,Vec3::new(0.0,-0.1,0.0));
        let source=Mat4::from_scale_rotation_translation(Vec3::new(0.12,2.0,2.0),glam::Quat::IDENTITY,Vec3::new(1.2,1.0,0.0));
        let blocker=Mat4::from_scale_rotation_translation(Vec3::new(0.08,3.0,5.0),glam::Quat::IDENTITY,Vec3::new(0.6,1.5,0.0));
        let mut draws=vec![draw(mesh,grey,floor,-1),draw(mesh,glow,source,-1)];
        for (slot,draw) in draws.iter_mut().enumerate() {draw.gi_source=Some(crate::GiSource {object:10,slot:slot as u32});}
        let mut blocked_draws=draws.clone(); blocked_draws.push(draw(mesh,grey,blocker,-1));
        let moving_draws:Vec<_>=(0..12).map(|step| {
            let mut d=blocked_draws.clone();
            d[2].transform=Mat4::from_translation(Vec3::new(-1.0-step as f32*0.2,0.0,0.0))*blocker;
            d
        }).collect();
        let mut fields=vec![crate::GiMeshSdf { gi_source: draws[0].gi_source,world_to_local:floor.inverse(),volume:volume.clone(),material,in_global_field:false},
            crate::GiMeshSdf { gi_source: draws[1].gi_source,world_to_local:source.inverse(),volume:Arc::new((*volume).clone()),material:emissive,in_global_field:false}];
        r.gi_set_gdf(&vec![10.0;512],&vec![0;512],[8;3],[-10.0;3],[10.0;3],&[material]);
        r.gi_set_emitters(&[]); r.set_gi_enabled(true);
        r.set_flux_settings(FluxSettings {rt_trace:false,samples:64,bounces:2,reflection_mode:0,ssr_enabled:false,..Default::default()});
        let eye=Vec3::new(-2.0,2.5,4.0);
        let camera=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::new(0.0,0.3,0.0),Vec3::Y),
            proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0)};
        let mut input=crate::cascade_tests::input(camera);
        input.viewport_extent=Some([323,181]); input.camera_preview=Some(camera);
        input.light.intensity=0.0; input.light.ambient=[0.0;3]; input.gi_debug=6;
        input.postfx.bloom_enabled=false; input.preview_postfx.bloom_enabled=false;
        input.draws=&draws;
        for _ in 0..20 {r.render(&input)?;}
        let receiver=Vec3::new(0.0,0.0,0.0);
        let absent=patch(&r.debug_read_viewport()?.unwrap(),camera,receiver);
        anyhow::ensure!(absent<2.0,"empty GDF unexpectedly found mesh emission: {absent}");
        r.gi_set_mesh_sdfs(&fields)?;
        let voxels=r.gpu_gi.as_ref().unwrap().mesh_sdfs.current.as_ref().unwrap().resources().1;
        for _ in 0..50 {r.render(&input)?;}
        let lit_scene=patch(&r.debug_read_viewport()?.unwrap(),camera,receiver);
        let lit_camera=patch(&r.debug_read_camera_preview()?.unwrap(),camera,receiver);
        eprintln!("[Flux mesh SDF] absent={absent}, emissive Scene={lit_scene}, Camera={lit_camera}");
        anyhow::ensure!(lit_scene>30.0 && lit_camera>30.0,"mesh-field material emission missing");
        // Exact IDs survive raster reordering and do not rely on fallback
        // material equality. The actual source is captured bright; its SDF
        // fallback is deliberately black. A missing known identity must stay
        // black instead of borrowing this coincident bright capture.
        let reversed=vec![draws[1],draws[0]];
        input.draws=&reversed;
        fields[1].material.emission=[0.0;3];
        fields[1].gi_source=Some(crate::GiSource {object:999,slot:0});
        r.gi_set_mesh_sdfs(&fields)?;
        for _ in 0..64 {r.render(&input)?;}
        let missing=patch(&r.debug_read_viewport()?.unwrap(),camera,receiver);
        let missing_camera=patch(&r.debug_read_camera_preview()?.unwrap(),camera,receiver);
        anyhow::ensure!(missing<2.0 && missing_camera<2.0,"known uncached SDF borrowed a neighbour: {missing}, {missing_camera}");
        fields[1].gi_source=draws[1].gi_source;
        r.gi_set_mesh_sdfs(&fields)?; // identity is the ONLY changed input
        assert_eq!(voxels,r.gpu_gi.as_ref().unwrap().mesh_sdfs.current.as_ref().unwrap().resources().1);
        for _ in 0..64 {r.render(&input)?;}
        let mapped=patch(&r.debug_read_viewport()?.unwrap(),camera,receiver);
        let mapped_camera=patch(&r.debug_read_camera_preview()?.unwrap(),camera,receiver);
        eprintln!("[Flux mesh identity] missing={missing}/{missing_camera}, mapped after reorder={mapped}/{mapped_camera}");
        anyhow::ensure!(mapped>30.0 && mapped_camera>30.0,"software hit lost exact cached source after reorder");
        fields[1].material=emissive;
        input.draws=&blocked_draws;
        fields.push(crate::GiMeshSdf { gi_source: None,world_to_local:blocker.inverse(),volume:volume.clone(),material,in_global_field:false});
        r.gi_set_mesh_sdfs(&fields)?;
        assert_eq!(voxels,r.gpu_gi.as_ref().unwrap().mesh_sdfs.current.as_ref().unwrap().resources().1,"instances must share voxel storage");
        for _ in 0..50 {r.render(&input)?;}
        let blocked=patch(&r.debug_read_viewport()?.unwrap(),camera,receiver);
        let blocked_camera=patch(&r.debug_read_camera_preview()?.unwrap(),camera,receiver);
        eprintln!("[Flux mesh SDF] thin moving blocker Scene={blocked}, Camera={blocked_camera}");
        anyhow::ensure!(blocked<lit_scene*0.5 && blocked_camera<lit_camera*0.5,"mesh-field occluder did not shadow emission");
        // Publish changing snapshots while both views still have submissions
        // in flight. Geometry buffers change; the shared voxel buffer must not.
        for moved in &moving_draws {
            input.draws=moved; fields[2].world_to_local=moved[2].transform.inverse();
            r.gi_set_mesh_sdfs(&fields)?; r.render(&input)?;
        }
        for _ in 0..50 {r.render(&input)?;}
        let recovered=patch(&r.debug_read_viewport()?.unwrap(),camera,receiver);
        anyhow::ensure!(recovered>lit_scene*0.75,"moving blocker left stale GI: {recovered}");
        r.gi_set_mesh_sdfs(&[])?;
        for _ in 0..50 {r.render(&input)?;}
        let cleared=patch(&r.debug_read_viewport()?.unwrap(),camera,receiver);
        anyhow::ensure!(cleared<2.0,"removed mesh fields left stale lighting: {cleared}");

        // Static detail ends at two metres; the global field must take over.
        // Geometry absent from that field must instead trace the full distance.
        let far=Mat4::from_scale_rotation_translation(Vec3::new(1.0,2.0,2.0),glam::Quat::IDENTITY,Vec3::new(3.0,1.0,0.0));
        let far_draws=vec![draw(mesh,grey,floor,-1),draw(mesh,glow,far,-1)];
        input.draws=&far_draws;
        fields.truncate(2); fields[0].gi_source=None; fields[1].gi_source=None; fields[1].world_to_local=far.inverse(); fields[1].in_global_field=true;
        let min=Vec3::new(-6.0,-1.0,-4.0); let max=Vec3::new(6.0,4.0,4.0);
        let dims=[32,16,32]; let mut dist=Vec::new();
        for z in 0..32 {for y in 0..16 {for x in 0..32 {
            let p=min+(max-min)*Vec3::new(x as f32/31.0,y as f32/15.0,z as f32/31.0);
            let q=(p-Vec3::new(3.0,1.0,0.0)).abs()-Vec3::new(0.5,1.0,1.0);
            dist.push(q.max(Vec3::ZERO).length()+q.max_element().min(0.0));
        }}}
        r.gi_set_gdf(&dist,&vec![0;dist.len()],dims,min.to_array(),max.to_array(),&[emissive]);
        r.gi_set_mesh_sdfs(&fields)?;
        for _ in 0..50 {r.render(&input)?;}
        let global=patch(&r.debug_read_viewport()?.unwrap(),camera,receiver);
        // The distant GDF hit must select its own explicit source too. Use
        // a black fallback and change only identity to expose wrong lookup.
        let mut identified_far=far_draws.clone();
        identified_far[1].gi_source=Some(crate::GiSource {object:20,slot:1});
        input.draws=&identified_far;
        r.gi_set_mesh_sdfs(&[])?; // force global-field hits throughout this check
        r.gi_set_gdf_identified(&dist,&vec![0;dist.len()],dims,min.to_array(),max.to_array(),&[material],
            &[Some(crate::GiSource {object:999,slot:1})])?;
        for _ in 0..64 {r.render(&input)?;}
        let missing=patch(&r.debug_read_viewport()?.unwrap(),camera,receiver);
        let missing_camera=patch(&r.debug_read_camera_preview()?.unwrap(),camera,receiver);
        r.gi_set_gdf_identified(&dist,&vec![0;dist.len()],dims,min.to_array(),max.to_array(),&[material],
            &[identified_far[1].gi_source])?;
        for _ in 0..64 {r.render(&input)?;}
        let mapped=patch(&r.debug_read_viewport()?.unwrap(),camera,receiver);
        let mapped_camera=patch(&r.debug_read_camera_preview()?.unwrap(),camera,receiver);
        eprintln!("[Flux global identity] missing={missing}/{missing_camera}, mapped={mapped}/{mapped_camera}");
        anyhow::ensure!(missing<2.0 && missing_camera<2.0 && mapped>20.0 && mapped_camera>20.0,
            "global field lost source identity or borrowed neighbouring light");
        input.draws=&far_draws;
        r.gi_set_mesh_sdfs(&fields)?;
        r.gi_set_gdf(&vec![10.0;512],&vec![0;512],[8;3],[-10.0;3],[10.0;3],&[material]);
        for _ in 0..50 {r.render(&input)?;}
        let near_only=patch(&r.debug_read_viewport()?.unwrap(),camera,receiver);
        fields[1].in_global_field=false; r.gi_set_mesh_sdfs(&fields)?;
        for _ in 0..50 {r.render(&input)?;}
        let dynamic=patch(&r.debug_read_viewport()?.unwrap(),camera,receiver);
        let dynamic_camera=patch(&r.debug_read_camera_preview()?.unwrap(),camera,receiver);
        eprintln!("[Flux mesh hierarchy] global={global}, near only={near_only}, dynamic={dynamic}, Camera={dynamic_camera}");
        anyhow::ensure!(global>20.0 && near_only<2.0 && dynamic>20.0 && dynamic_camera>20.0,
            "detail/global/dynamic trace handoff failed");
        r.gi_set_mesh_sdfs(&[])?;
        Ok(())
    }

    // All directional cells of this convex front face see the same sky.
    // Check actual GPU records across ray budgets, including partial groups
    // and multiple samples per direction: quality must not change energy.
    fn balanced_gather(&self,r:&mut Renderer,input:&mut crate::FrameInput<'_>)->anyhow::Result<()> {
        let saved=r.flux_settings;let hardware=saved.rt_trace;
        let cache_override=r.gpu_gi.as_mut().unwrap().radiance_cache.enabled_override;
        let screen_override=r.gpu_gi.as_mut().unwrap().screen_trace_override;
        r.gpu_gi.as_mut().unwrap().radiance_cache.enabled_override=Some(false);
        r.gpu_gi.as_mut().unwrap().screen_trace_override=Some(false);
        r.gpu_gi.as_mut().unwrap().probe_history_alpha=Some(0.5);
        r.gpu_gi.as_mut().unwrap().gather_seed=Some(0xb41a9ce);
        let stride=crate::gpu_gi::SCREEN_PROBE_BYTES as usize;
        for samples in [1,17,64,256] {
            r.set_flux_settings(FluxSettings {samples,bounces:1,..saved});
            for _ in 0..48 {r.render(input)?;}
            let mut histories=Vec::new();
            for (target,parity) in [(&r.viewport_target.as_ref().unwrap().sgi,r.sgi_parity),
                (&r.camera_preview.as_ref().unwrap().sgi,r.preview_sgi_parity)] {
                let size=(target.probe_extent.width*target.probe_extent.height) as usize*stride;
                let mut bytes=transfer_probe_history(r,target.sh_buf[(parity^1)&1].handle,size,None)?;
                for (index,probe) in bytes[..size].chunks_exact_mut(stride).enumerate() {for bin in 0..16 {
                    let ray=&mut probe[80+bin*16..96+bin*16];
                    let value=if bin==index%16 {100.0f32} else {0.001};
                    for channel in 0..3 {ray[channel*4..channel*4+4].copy_from_slice(&value.to_ne_bytes());}
                    ray[14..16].copy_from_slice(&half::f16::from_f32(16.0).to_bits().to_ne_bytes());
                }}
                histories.push(bytes);
            }
            let mut outputs=Vec::new();
            for (balanced,area) in [(false,false),(true,false),(true,true)] {
                r.gpu_gi.as_mut().unwrap().balanced_gather=Some(balanced);
                r.gpu_gi.as_mut().unwrap().area_gather_variant=Some(area);
                for (view,(target,parity)) in [(&r.viewport_target.as_ref().unwrap().sgi,r.sgi_parity),
                    (&r.camera_preview.as_ref().unwrap().sgi,r.preview_sgi_parity)].into_iter().enumerate() {
                    transfer_probe_history(r,target.sh_buf[(parity^1)&1].handle,histories[view].len(),Some(&histories[view]))?;
                }
                r.render(input)?;
                let mut views=Vec::new();
                for (view,(target,parity)) in [(&r.viewport_target.as_ref().unwrap().sgi,r.sgi_parity),
                    (&r.camera_preview.as_ref().unwrap().sgi,r.preview_sgi_parity)].into_iter().enumerate() {
                    views.push(transfer_probe_history(r,target.sh_buf[(parity^1)&1].handle,histories[view].len(),None)?);
                }
                outputs.push(views);
            }
            for variant in 1..outputs.len() {for view in 0..2 {
                let mut checked=0;let mut max_error=0.0f32;
                for (a,b) in outputs[0][view].chunks_exact(stride).zip(outputs[variant][view].chunks_exact(stride)) {
                    if f32::from_ne_bytes(a[12..16].try_into().unwrap())<=0.0 {continue;}
                    for bin in 0..16 {for channel in 0..3 {
                        let at=80+bin*16+channel*4;
                        let x=f32::from_ne_bytes(a[at..at+4].try_into().unwrap());
                        let y=f32::from_ne_bytes(b[at..at+4].try_into().unwrap());
                        anyhow::ensure!(x.is_finite() && y.is_finite(),"nonfinite balanced ray");
                        max_error=max_error.max((x-y).abs());
                    }
                        assert_eq!(&a[92+bin*16..96+bin*16],&b[92+bin*16..96+bin*16],"balanced dispatch changed ray distances/history");
                    }
                    checked+=1;
                }
                eprintln!("[Flux balanced gather] hardware={hardware} samples={samples} view={view} area_variant={} probes={checked} max_error={max_error}",variant==2);
                anyhow::ensure!(checked>10 && max_error<0.00001,"balanced dispatch changed directional radiance");
            }}
        }
        r.gpu_gi.as_mut().unwrap().area_gather_variant=None;
        let extent=input.viewport_extent;let preview=input.camera_preview;
        input.viewport_extent=Some([1280,720]);input.camera_preview=None;
        r.set_flux_settings(FluxSettings {samples:64,bounces:1,..saved});
        let mut timing=[Vec::new(),Vec::new()];
        for balanced in [false,true,true,false] {
            r.gpu_gi.as_mut().unwrap().balanced_gather=Some(balanced);
            for _ in 0..24 {r.render(input)?;}
            for _ in 0..32 {r.render(input)?;let ms=r.stats().gpu_gi_gather_ms;if ms>0.0 {timing[usize::from(balanced)].push(ms);}}
        }
        for t in &mut timing {t.sort_by(f32::total_cmp);}
        eprintln!("[Flux balanced timing] hardware={hardware} serial={}ms balanced={}ms (alternating blocks,1280x720,64 rays)",
            timing[0][timing[0].len()/2],timing[1][timing[1].len()/2]);
        // The same empty-Area-Light scene must not pay for the textured-light
        // shader path. Alternate pipeline variants in one process to reduce
        // clock drift; external GPU contention still requires an idle rerun.
        r.gpu_gi.as_mut().unwrap().balanced_gather=Some(true);
        let mut area_timing=[Vec::new(),Vec::new()];
        for area in [false,true,true,false] {
            r.gpu_gi.as_mut().unwrap().area_gather_variant=Some(area);
            for _ in 0..24 {r.render(input)?;}
            for _ in 0..32 {r.render(input)?;let ms=r.stats().gpu_gi_gather_ms;if ms>0.0 {area_timing[usize::from(area)].push(ms);}}
        }
        for t in &mut area_timing {t.sort_by(f32::total_cmp);}
        eprintln!("[Flux Area Light specialization timing] hardware={hardware} disabled={}ms enabled={}ms (same scene without Area Lights, alternating blocks)",
            area_timing[0][area_timing[0].len()/2],area_timing[1][area_timing[1].len()/2]);
        r.gpu_gi.as_mut().unwrap().area_gather_variant=None;
        input.viewport_extent=extent;input.camera_preview=preview;
        let gi=r.gpu_gi.as_mut().unwrap();gi.balanced_gather=None;gi.gather_seed=None;gi.probe_history_alpha=None;
        gi.radiance_cache.enabled_override=cache_override;
        gi.screen_trace_override=screen_override;
        r.set_flux_settings(saved);for _ in 0..40 {r.render(input)?;}
        Ok(())
    }

    fn importance_budget(&self,r:&mut Renderer,input:&mut crate::FrameInput<'_>)->anyhow::Result<()> {
        let saved=r.flux_settings;r.set_flux_settings(FluxSettings {samples:64,..saved});
        r.gpu_gi.as_mut().unwrap().probe_history_alpha=Some(0.5);
        for _ in 0..4 {r.render(input)?;}
        let stride=crate::gpu_gi::SCREEN_PROBE_BYTES as usize;
        for mode in 0..3 {
            // Inject a directional guide, then inspect actual dispatched rays.
            // Changing the bright direction must move work without changing the
            // total, starving other cells, or changing raw constant-sky energy.
            for (target,parity) in [(&r.viewport_target.as_ref().unwrap().sgi,r.sgi_parity),
                (&r.camera_preview.as_ref().unwrap().sgi,r.preview_sgi_parity)] {
                let size=(target.probe_extent.width*target.probe_extent.height) as usize*stride;
                let buffer=target.sh_buf[(parity^1)&1].handle;
                let mut bytes=transfer_probe_history(r,buffer,size,None)?;
                for probe in bytes[..size].chunks_exact_mut(stride) {for bin in 0..16 {
                    let ray=&mut probe[80+bin*16..96+bin*16];
                    let value=if bin==mode%2 {100.0f32} else {0.001};
                    for channel in 0..3 {ray[channel*4..channel*4+4].copy_from_slice(&value.to_ne_bytes());}
                    ray[14..16].copy_from_slice(&half::f16::from_f32(if mode==2 {0.0} else {16.0}).to_bits().to_ne_bytes());
                }}
                transfer_probe_history(r,buffer,size,Some(&bytes[..size]))?;
            }
            r.render(input)?;
            for (name,target,parity) in [("Scene",&r.viewport_target.as_ref().unwrap().sgi,r.sgi_parity),
                ("Camera",&r.camera_preview.as_ref().unwrap().sgi,r.preview_sgi_parity)] {
                let probes=(target.probe_extent.width*target.probe_extent.height) as usize;
                let meta=transfer_probe_history(r,target.sh_buf[(parity^1)&1].handle,probes*stride,None)?;
                let raw=transfer_probe_history(r,target.raw_rays.handle,16+probes*17*16,None)?;
                let u=|at|u32::from_ne_bytes(raw[at..at+4].try_into().unwrap()) as usize;
                let jobs=u(0);let mut guided=0;
                for job in 0..jobs {
                    let owner=u(16+probes*16*16+job*16+12);
                    let p=&meta[owner*stride..(owner+1)*stride];
                    let span=f32::from_ne_bytes(p[0..4].try_into().unwrap()) as usize;
                    let uniform=(4*span).clamp(1,16);let mut counts=[0usize;16];
                    for bin in 0..16 {
                        let at=16+(owner*16+bin)*16;
                        counts[bin]=half::f16::from_bits(u16::from_ne_bytes(raw[at+14..at+16].try_into().unwrap())).to_f32() as usize;
                        anyhow::ensure!(counts[bin]>=uniform/2,"importance sampling starved a direction");
                        for channel in 0..3 {
                            let value=f32::from_ne_bytes(raw[at+channel*4..at+channel*4+4].try_into().unwrap());
                            anyhow::ensure!((value-input.light.ambient[channel]*saved.intensity).abs()<0.001,
                                "importance sampling changed constant-sky radiance: {value}");
                        }
                    }
                    assert_eq!(counts.iter().sum::<usize>(),uniform*16,"importance sampling changed the ray budget");
                    if mode==2 {assert!(counts.iter().all(|n|*n==uniform),"invalid history guided rays");}
                    else if counts[mode%2]>uniform {guided+=1;}
                }
                eprintln!("[Flux guided budget] {name} mode={mode}: jobs={jobs}, guided={guided}");
                anyhow::ensure!(jobs>10 && (mode==2 || guided>10),"validated lighting did not guide the actual ray jobs");
            }
        }
        r.gpu_gi.as_mut().unwrap().probe_history_alpha=None;r.set_flux_settings(saved);
        for _ in 0..40 {r.render(input)?;}
        Ok(())
    }

    fn gather_budget(&self, r: &mut Renderer, input: &mut crate::FrameInput<'_>) -> anyhow::Result<()> {
        let saved = r.flux_settings;
        let stride = crate::gpu_gi::SCREEN_PROBE_BYTES as usize;
        for samples in [1, 17, 64, 256] {
            r.set_flux_settings(FluxSettings { samples, ..saved });
            for _ in 0..3 { r.render(input)?; }
            for (name, target, parity) in [
                ("Scene", &r.viewport_target.as_ref().unwrap().sgi, r.sgi_parity),
                ("Camera", &r.camera_preview.as_ref().unwrap().sgi, r.preview_sgi_parity),
            ] {
                let size = target.probe_extent.width as usize * target.probe_extent.height as usize * stride;
                let bytes = transfer_probe_history(r, target.sh_buf[(parity^1)&1].handle, size, None)?;
                let f = |p: &[u8], at: usize| f32::from_ne_bytes(p[at..at+4].try_into().unwrap());
                let mut error = 0.0f32;
                let mut checked = 0;
                for p in bytes[..size].chunks_exact(stride) {
                    if f(p,12) <= 0.0 || f(p,60) < 0.99
                        || (f(p,68)-2.0).abs() > 0.35 || f(p,72).abs() > 0.35 { continue; }
                    for bin in 0..16 {
                        for channel in 0..3 {
                            let v = f(p,80+bin*16+channel*4);
                            anyhow::ensure!(v.is_finite(), "{name} non-finite directional radiance");
                            error = error.max((v-input.light.ambient[channel]*saved.intensity).abs());
                        }
                    }
                    checked += 1;
                }
                eprintln!("[Flux gather budget] {name}, samples={samples}: {checked} receivers, maximum error={error:.6}");
                anyhow::ensure!(checked > 10 && error < 0.001, "{name} gather changed sky energy at {samples} samples: {error}");
            }
        }
        r.set_flux_settings(saved);
        for _ in 0..3 { r.render(input)?; }
        Ok(())
    }

    // Inject a high-frequency history pattern into the actual GPU gather
    // buffers. With a fixed camera, a receiver must reuse its own sample, not
    // blur toward a neighbour because its chosen depth texel is off-centre.
    // This checks both views and both regular/odd target dimensions.
    fn history_identity(&self, r: &mut Renderer, input: &mut crate::FrameInput<'_>) -> anyhow::Result<()> {
        // Exercise Lumen's optional short probe history independently of the
        // selected production temporal configuration.
        r.gpu_gi.as_mut().unwrap().probe_history_alpha = Some(0.5);
        let saved = r.flux_settings;
        r.set_flux_settings(FluxSettings { samples: 64, ..saved });
        for _ in 0..3 { r.render(input)?; }
        let stride = crate::gpu_gi::SCREEN_PROBE_BYTES as usize;
        let mut controls = Vec::new();
        for (name, target, parity) in [
            ("Scene", &r.viewport_target.as_ref().unwrap().sgi, r.sgi_parity),
            ("Camera", &r.camera_preview.as_ref().unwrap().sgi, r.preview_sgi_parity),
        ] {
            let width = target.probe_extent.width as usize;
            let size = width * target.probe_extent.height as usize * stride;
            let mut bytes = transfer_probe_history(r, target.sh_buf[(parity^1)&1].handle, size, None)?;
            for (index, probe) in bytes[..size].chunks_exact_mut(stride).enumerate() {
                let value = ((index % width + index / width) & 1) as f32;
                for bin in 0..16 {
                    let ray = &mut probe[80+bin*16..96+bin*16];
                    // Unequal directional history catches accidental sharing
                    // between ray lanes, including during a lighting change.
                    let direction_value = value * if bin % 4 == 0 { 1.0f32 } else { 0.01 };
                    ray[..4].copy_from_slice(&direction_value.to_ne_bytes());
                    ray[4..12].fill(0);
                    ray[14..16].copy_from_slice(&half::f16::from_f32(16.0).to_bits().to_ne_bytes());
                }
            }
            transfer_probe_history(r, target.sh_buf[(parity^1)&1].handle, size, Some(&bytes[..size]))?;
            controls.push((name, target.sh_buf[parity&1].handle, size, bytes));
        }
        let saved_sky = input.light.ambient;
        input.light.ambient = [0.2, 0.0, 0.0];
        r.render(input)?;
        input.light.ambient = saved_sky;
        let expected_alpha = 0.5;
        for (name, buffer, size, previous) in controls {
            let current = transfer_probe_history(r, buffer, size, None)?;
            let f = |bytes: &[u8], at: usize| f32::from_ne_bytes(bytes[at..at+4].try_into().unwrap());
            let mut checked = 0;
            let mut error = 0.0f32;
            for (old, new) in previous[..size].chunks_exact(stride).zip(current[..size].chunks_exact(stride)) {
                // Interior of the cube's front face: all new rays see red sky.
                if f(old,12) <= 0.0 || f(old,60) < 0.99
                    || (f(old,68)-2.0).abs() > 0.35 || f(old,72).abs() > 0.35 { continue; }
                for bin in 0..16 {
                    let expected = f(old,80+bin*16)*(1.0-expected_alpha)
                        + 0.2*saved.intensity*expected_alpha;
                    error = error.max((f(new,80+bin*16)-expected).abs());
                }
                checked += 1;
            }
            eprintln!("[Flux history identity] {name}: {checked} receivers, maximum error={error:.6}");
            anyhow::ensure!(checked > 10 && error < 0.01, "{name} history moved between fixed receivers: {error}");
        }
        r.gpu_gi.as_mut().unwrap().probe_history_alpha = None;
        r.set_flux_settings(saved);
        r.invalidate_screen_gi();
        for _ in 0..3 { r.render(input)?; }
        Ok(())
    }

    // Isolate final-gather history from world-cache convergence. A uniform
    // sky changes without resetting either view: stale directional radiance
    // must not keep a now-unlit surface bright for dozens of frames.
    fn temporal_response(&self, r: &mut Renderer, input: &mut crate::FrameInput<'_>) -> anyhow::Result<()> {
        // The faster response path remains opt-in pending quality tuning.
        r.gpu_gi.as_mut().unwrap().temporal_override = Some([1.0, 10.0, 0.0, 0.0]);
        r.invalidate_screen_gi();
        let saved = input.light.ambient;
        let receiver = Vec3::new(2.0, 0.0, 0.5);
        input.light.ambient = [0.3;3];
        for _ in 0..20 { r.render(input)?; }
        let lit = patch(&r.debug_read_viewport()?.unwrap(), input.camera, receiver);
        anyhow::ensure!(lit > 50.0, "history response test did not start lit: {lit}");
        input.light.ambient = [0.0;3];
        for _ in 0..4 { r.render(input)?; }
        let scene = patch(&r.debug_read_viewport()?.unwrap(), input.camera, receiver);
        let camera = patch(&r.debug_read_camera_preview()?.unwrap(), input.camera_preview.unwrap(), receiver);
        eprintln!("[Flux temporal response] RT={}: after four dark frames Scene={scene}, Camera={camera}", r.flux_settings.rt_trace);
        anyhow::ensure!(scene < 2.0 && camera < 2.0, "final gather retained stale sky: {scene}, {camera}");
        input.light.ambient = saved;
        r.gpu_gi.as_mut().unwrap().temporal_override = None;
        r.invalidate_screen_gi();
        for _ in 0..4 { r.render(input)?; }
        Ok(())
    }

    fn reflected_bounce(&self, r: &mut Renderer) -> anyhow::Result<()> {
        r.set_gi_enabled(true);
        let mesh = r.upload_mesh(&cube())?;
        let materials = [lit(r, [0.95;4], [0.0;3], 0.02, 1.0),
            lit(r, [0.7;4], [0.0;3], 1.0, 0.0),
            lit(r, [0.8,0.02,0.02,1.0], [0.0;3], 1.0, 0.0)];
        // A mirror floor sees the front of the middle wall. The directional
        // light travels from behind that wall, so only light bounced from the
        // offset red panel can illuminate its reflected front face.
        let boxes = [(Vec3::new(0.0,-0.1,0.0),Vec3::new(4.0,0.1,4.0)),
            (Vec3::new(0.0,1.5,-2.0),Vec3::new(1.0,1.5,0.1)),
            (Vec3::new(2.2,1.5,1.0),Vec3::new(1.0,1.5,0.1))];
        let draws: Vec<_> = boxes.iter().enumerate().map(|(i,(p,h))|
            draw(mesh,materials[i],Mat4::from_scale_rotation_translation(*h*2.0,glam::Quat::IDENTITY,*p),-1)).collect();
        let dims = [48,32,48];
        let min = Vec3::new(-4.0,-1.0,-4.0);
        let max = Vec3::new(4.0,4.0,4.0);
        let mut distances = Vec::new(); let mut indices = Vec::new();
        for z in 0..dims[2] { for y in 0..dims[1] { for x in 0..dims[0] {
            let p = min+(max-min)*Vec3::new(x as f32/47.0,y as f32/31.0,z as f32/47.0);
            let (i,d) = boxes.iter().enumerate().map(|(i,(c,h))| {
                let q = (p-*c).abs()-*h;
                (i,q.max(Vec3::ZERO).length()+q.max_element().min(0.0))
            }).min_by(|a,b| a.1.total_cmp(&b.1)).unwrap();
            distances.push(d); indices.push(i as u32);
        } } }
        r.gi_set_gdf(&distances,&indices,dims,min.to_array(),max.to_array(),&[
            GpuGiMaterial { albedo:[0.95;3],emission:[0.0;3],metallic:1.0,roughness:0.02 },
            GpuGiMaterial { albedo:[0.7;3],emission:[0.0;3],metallic:0.0,roughness:1.0 },
            GpuGiMaterial { albedo:[0.8,0.02,0.02],emission:[0.0;3],metallic:0.0,roughness:1.0 },
        ]);
        r.gi_set_emitters(&[]);
        let eye = Vec3::new(0.0,2.0,4.0);
        let camera = CameraData { position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),
            proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0) };
        let mut input = crate::cascade_tests::input(camera);
        input.viewport_extent=Some([640,360]); input.camera_preview=Some(camera); input.draws=&draws;
        input.light.intensity=0.0; input.light.ambient=[0.0;3]; input.light.env_intensity=0.0;
        input.draw_skybox=false; input.postfx.bloom_enabled=false; input.preview_postfx.bloom_enabled=false;
        let light = [crate::LightInstance {kind:crate::LightKind::Directional,position:Vec3::ZERO,direction:Vec3::Z,
            color:[1.0;3],intensity:8.0,range:0.0,spot_inner_deg:30.0,spot_outer_deg:45.0,
            cast_shadows:true,soft_shadows:false,radius:0.0,shadow_bias:0.0,baked:false}];
        input.lights=&light;
        let emission_map=r.upload_texture(&crate::TextureData {width:1,height:1,
            pixels:vec![128,255,255,255],srgb:false,hdr:false})?;
        let mut emitter_slots=[None;16]; emitter_slots[3]=Some(emission_map);
        r.set_material_textures(materials[2],&emitter_slots);
        for rt in [false,true] {
            let settings=FluxSettings {rt_trace:rt,samples:16,bounces:1,reflection_mode:2,rt_samples:1,rt_bounces:1,
                ssr_enabled:false,..Default::default()};
            r.set_flux_settings(settings);
            for _ in 0..24 {r.render(&input)?;}
            let single=patch(&r.debug_read_viewport()?.unwrap(),camera,Vec3::ZERO);
            r.set_flux_settings(FluxSettings {bounces:4,..settings});
            for _ in 0..90 {r.render(&input)?;}
            let feedback=r.flux_surface.as_ref().unwrap().debug_feedback();
            eprintln!("[Flux GPU surface feedback] RT GI={rt}: GI/reflection hit pages={feedback:?}");
            anyhow::ensure!(feedback.iter().all(|n|*n>0),"GPU view hit feedback was not consumed: {feedback:?}");
            for (material,face) in [(materials[1],4),(materials[2],5)] {
                let allocator=r.allocator();
                let cache=r.flux_surface.as_ref().unwrap().debug_lighting(&r.ctx.device,&mut allocator.lock().unwrap(),
                    r.command_pool,r.ctx.queue,material.0,face)?;
                eprintln!("[Flux reflection cache] RT={rt}, material={material:?}, face={face}: {cache:?}");
            }
            let scene=patch(&r.debug_read_viewport()?.unwrap(),camera,Vec3::ZERO);
            let preview=patch(&r.debug_read_camera_preview()?.unwrap(),camera,Vec3::ZERO);
            eprintln!("[Flux reflection bounce] RT GI={rt}: single={single:.1}, cached Scene={scene:.1}, Camera={preview:.1}");
            anyhow::ensure!(scene>single+15.0 && preview>single+15.0,
                "reflections lost cached bounce lighting: single={single}, Scene={scene}, Camera={preview}");
            anyhow::ensure!((scene-preview).abs()<15.0,"Scene/Camera reflected GI differs: {scene}, {preview}");
            for mode in [crate::RayLightingMode::HitLightingForReflections,crate::RayLightingMode::HitLighting] {
                input.preview_postfx.flux.ray_lighting_mode=Some(mode);
                for _ in 0..20 {r.render(&input)?;}
                let hit=patch(&r.debug_read_camera_preview()?.unwrap(),camera,Vec3::ZERO);
                let inherited=patch(&r.debug_read_viewport()?.unwrap(),camera,Vec3::ZERO);
                eprintln!("[Flux ray lighting secondary] RT={rt}, mode={mode:?}: cached Scene={inherited}, hit Camera={hit}");
                anyhow::ensure!(hit>single+15.0 && inherited>single+15.0,"hit lighting lost cached secondary transport: {hit}, {inherited}");
                assert_eq!(r.gi_hit_lighting[0],false);
                assert_eq!(r.gi_hit_lighting[1],rt && mode==crate::RayLightingMode::HitLighting);
            }
            input.preview_postfx.flux.ray_lighting_mode=None;
            input.preview_postfx.flux.max_roughness_to_trace=Some(0.0);
            for _ in 0..3 {r.render(&input)?;}
            let cutoff=patch(&r.debug_read_camera_preview()?.unwrap(),camera,Vec3::ZERO);
            let unchanged=patch(&r.debug_read_viewport()?.unwrap(),camera,Vec3::ZERO);
            anyhow::ensure!(cutoff<5.0 && unchanged>single+15.0,
                "Camera reflection override leaked into Scene or was ignored: {cutoff}, {unchanged}");
            input.preview_postfx.flux.max_roughness_to_trace=None;
            input.lights=&[];
            for _ in 0..48 {r.render(&input)?;}
            let dark=patch(&r.debug_read_camera_preview()?.unwrap(),camera,Vec3::ZERO);
            anyhow::ensure!(dark<10.0,"reflection retained stale cached lighting after lights off: {dark}");
            // No analytic emitters or scene lights: actual textured material
            // emission must enter radiosity and illuminate the reflected wall.
            r.set_material_features(materials[2],crate::MaterialFeatures {emission:true,..Default::default()});
            let params=r.material_params_mut(materials[2]);
            params.emission_color=[4.0,0.0,0.0]; params.emission_intensity=1.0;
            for _ in 0..90 {r.render(&input)?;}
            let scene=patch(&r.debug_read_viewport()?.unwrap(),camera,Vec3::ZERO);
            let preview=patch(&r.debug_read_camera_preview()?.unwrap(),camera,Vec3::ZERO);
            eprintln!("[Flux material emission bounce] RT GI={rt}: Scene={scene:.1}, Camera={preview:.1}");
            anyhow::ensure!(scene>20.0 && preview>20.0,"textured emissive material failed to light a reflected receiver");
            r.material_params_mut(materials[2]).emission_intensity=0.0;
            input.lights=&light;
        }
        // Reflections remain functional with GI disabled, and stale surface
        // atlases must not suppress a freshly changed emissive material.
        r.set_gi_enabled(false);
        input.lights=&[];
        r.set_material_features(materials[1],crate::MaterialFeatures {emission:true,..Default::default()});
        let params=r.material_params_mut(materials[1]);
        params.emission_color=[2.0,0.0,0.0]; params.emission_intensity=1.0;
        for mode in [2,1] {
            r.set_flux_settings(FluxSettings {reflection_mode:mode,ssr_enabled:true,rt_samples:1,..Default::default()});
            for _ in 0..4 {r.render(&input)?;}
            let scene=patch(&r.debug_read_viewport()?.unwrap(),camera,Vec3::ZERO);
            let preview=patch(&r.debug_read_camera_preview()?.unwrap(),camera,Vec3::ZERO);
            eprintln!("[Flux reflection standalone] mode={mode}, Scene={scene:.1}, Camera={preview:.1}");
            // A missing row of mirror pixels still passed the former >50 test.
            // The constant emissive wall must fill this interior 5x5 patch.
            anyhow::ensure!(scene>235.0 && preview>235.0,"standalone reflection lost coverage: {scene}, {preview}");
        }
        // A source's analytic sampling proxy can have a black average (e.g. a
        // small bright image region). It must never hide visible hit emission.
        r.set_gi_enabled(true);
        r.set_flux_settings(FluxSettings {reflection_mode:2,rt_samples:1,..Default::default()});
        r.gi_set_emitters(&[crate::GpuGiEmitter {center:[0.0,1.5,-2.0],radius:2.0,emission:[0.0;3]}]);
        for _ in 0..24 {r.render(&input)?;}
        let visible_emission=patch(&r.debug_read_camera_preview()?.unwrap(),camera,Vec3::ZERO);
        eprintln!("[Flux reflected source emission] Camera={visible_emission:.1}");
        anyhow::ensure!(visible_emission>230.0,"analytic GI proxy hid the reflected emitting material");
        r.gi_set_emitters(&[]);
        // A valid (dark) realtime cache must not replace a hit's baked lightmap.
        r.material_params_mut(materials[1]).emission_intensity=0.0;
        let mut baked_draws=draws.clone(); baked_draws[1].lightmap_layer=0;
        input.draws=&baked_draws;
        r.set_baked_lightmaps(&[crate::BakedLightmap {size:1,pixels:vec![3.0,0.0,0.0,1.0]}]);
        r.set_gi_enabled(true);
        r.set_flux_settings(FluxSettings {reflection_mode:2,..Default::default()});
        for _ in 0..24 {r.render(&input)?;}
        let baked_scene=patch(&r.debug_read_viewport()?.unwrap(),camera,Vec3::ZERO);
        let baked_camera=patch(&r.debug_read_camera_preview()?.unwrap(),camera,Vec3::ZERO);
        eprintln!("[Flux reflection baked] Scene={baked_scene:.1}, Camera={baked_camera:.1}");
        anyhow::ensure!(baked_scene>50.0 && baked_camera>50.0,"cached reflection replaced baked lighting");
        r.set_baked_lightmaps(&[crate::BakedLightmap {size:1,pixels:vec![0.0,0.0,0.0,1.0]}]);
        for _ in 0..4 {r.render(&input)?;}
        anyhow::ensure!(patch(&r.debug_read_camera_preview()?.unwrap(),camera,Vec3::ZERO)<10.0,
            "reflection retained the replaced lightmap");
        r.set_baked_lightmaps(&[]);
        r.set_gi_enabled(false);
        // Empty TLAS: resolve must write the current background in both views,
        // rather than sampling the previous reflected scene or an undefined image.
        r.set_flux_settings(FluxSettings {reflection_mode:2,..Default::default()});
        input.draws=&[];
        for _ in 0..4 {r.render(&input)?;}
        let empty_scene=r.debug_read_viewport()?.unwrap();
        let empty_camera=r.debug_read_camera_preview()?.unwrap();
        r.set_flux_settings(FluxSettings {reflection_mode:0,..Default::default()});
        for _ in 0..3 {r.render(&input)?;}
        anyhow::ensure!(empty_scene==r.debug_read_viewport()?.unwrap()
            && empty_camera==r.debug_read_camera_preview()?.unwrap(),"empty reflection resolve changed the background");
        Ok(())
    }

    fn surface_materials(&self, r: &mut Renderer) -> anyhow::Result<()> {
        let mesh = r.upload_mesh(&cube())?;
        let mat = lit(r,[0.5,0.25,0.75,1.0],[2.0,3.0,4.0],0.8,0.6);
        r.set_material_features(mat,crate::MaterialFeatures { emission: true, ..Default::default() });
        let texture = |r: &mut Renderer,pixels: [u8;4]| r.upload_texture(&crate::TextureData {
            width: 1,height: 1,pixels: pixels.to_vec(),srgb: false,hdr: false });
        let albedo = texture(r,[255,128,64,255])?;
        let emission = texture(r,[64,128,255,255])?;
        let orm = texture(r,[255,64,128,255])?;
        let mut slots = [None;16]; slots[0]=Some(albedo); slots[2]=Some(orm); slots[3]=Some(emission);
        r.set_material_textures(mat,&slots);
        // Completely off screen: material captures must not depend on depth or
        // visible pixels in either editor camera.
        let draws = [draw(mesh,mat,Mat4::from_translation(Vec3::new(20.0,0.0,0.0)),-1)];
        let camera = CameraData { position: Vec3::new(0.0,0.0,5.0),
            view: Mat4::look_at_rh(Vec3::new(0.0,0.0,5.0),Vec3::ZERO,Vec3::Y),
            proj: Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0) };
        let mut input = crate::cascade_tests::input(camera);
        input.viewport_extent = Some([320,180]); input.camera_preview = Some(camera); input.draws = &draws;
        r.render(&input)?;
        let cache = r.flux_surface.as_ref().unwrap();
        assert_eq!(cache.captures,6,"Scene/Camera duplicated material captures");
        let allocator = r.allocator();
        let sample = cache.debug_sample(&r.ctx.device,&mut allocator.lock().unwrap(),r.command_pool,r.ctx.queue,4)?;
        eprintln!("[Flux surface] position/normal/albedo/emission = {sample:?}");
        anyhow::ensure!((sample[0][0]-20.0).abs()<0.03 && (sample[0][2]-0.5).abs()<0.01 && sample[0][3]>0.9,"surface position capture is wrong");
        anyhow::ensure!(sample[1][2]>0.99 && (sample[1][3]-0.8*64.0/255.0).abs()<0.01,"surface normal/roughness capture is wrong");
        for (actual,expected) in sample[2].iter().zip([0.5,0.25*128.0/255.0,0.75*64.0/255.0,0.6*128.0/255.0]) {
            anyhow::ensure!((*actual-expected).abs()<0.006,"textured surface albedo/metalness is wrong: {sample:?}");
        }
        for (actual,expected) in sample[3][..3].iter().zip([2.0*64.0/255.0,3.0*128.0/255.0,4.0]) {
            anyhow::ensure!((*actual-expected).abs()<0.01,"textured surface emission is wrong: {sample:?}");
        }
        input.camera.position.x=1.0;
        input.camera.view=Mat4::look_at_rh(input.camera.position,Vec3::ZERO,Vec3::Y);
        r.render(&input)?;
        assert_eq!(r.flux_surface.as_ref().unwrap().captures,0,"camera movement recaptured stable surfaces");
        r.material_params_mut(mat).base_color=[0.1,0.2,0.3,1.0];
        r.render(&input)?;
        assert_eq!(r.flux_surface.as_ref().unwrap().captures,6,"material edit did not invalidate capture");
        let moved_draws = [crate::DrawCmd { transform: Mat4::from_translation(Vec3::new(21.0,0.0,0.0)), ..draws[0] }];
        input.draws=&moved_draws;
        r.render(&input)?;
        let cache=r.flux_surface.as_ref().unwrap();
        let moved=cache.debug_sample(&r.ctx.device,&mut allocator.lock().unwrap(),r.command_pool,r.ctx.queue,4)?;
        anyhow::ensure!((moved[0][0]-21.0).abs()<0.03,"moving geometry retained its old capture");
        let many: Vec<_> = (0..13).map(|i| crate::DrawCmd {
            transform: Mat4::from_translation(Vec3::new(20.0+i as f32*3.0,0.0,0.0)), ..draws[0]
        }).collect();
        input.draws=&many;
        for _ in 0..8 {
            r.render(&input)?;
            assert!(r.flux_surface.as_ref().unwrap().captures<=12,"surface capture exceeded its frame budget");
        }
        assert_eq!(r.flux_surface.as_ref().unwrap().resident,13*6,"atlas resize lost surface captures");
        let reverse: Vec<_> = many.iter().rev().copied().collect();
        input.draws=&reverse;
        r.render(&input)?;
        assert_eq!(r.flux_surface.as_ref().unwrap().captures,0,"draw order invalidated surface captures");
        input.draws=&[];
        r.render(&input)?;
        assert_eq!(r.flux_surface.as_ref().unwrap().resident,0,"removed geometry remained in the cache");
        // Many overlapping layers in ONE mesh: exterior captures cannot see
        // the inner planes. Exercise clipping, variable residency, per-card
        // budgets and transformed local projections through real rasterization.
        let layered = r.upload_mesh(&crate::flux_cards::layered_test_mesh(20))?;
        let expected = r.meshes[layered.0].surface_geometry.captures.len();
        assert!(expected>12 && expected<=32);
        let model = Mat4::from_scale_rotation_translation(Vec3::new(0.8,1.2,0.6),
            glam::Quat::from_rotation_y(0.7),Vec3::new(20.0,0.0,0.0));
        let layered_draws = [draw(layered,mat,model,-1)];
        input.draws=&layered_draws;
        for _ in 0..expected.div_ceil(12)+1 {
            r.render(&input)?;
            assert!(r.flux_surface.as_ref().unwrap().captures<=12,"interior cards exceeded capture budget");
        }
        let cache=r.flux_surface.as_ref().unwrap();
        assert_eq!(cache.resident,expected,"interior cards failed to become resident");
        assert_eq!(cache.captures,0,"stable interior cards were recaptured");
        for face in [4,6] {
            let sample=cache.debug_sample(&r.ctx.device,&mut allocator.lock().unwrap(),r.command_pool,r.ctx.queue,face)?;
            let local=model.inverse().transform_point3(Vec3::from_array(sample[0][..3].try_into().unwrap()));
            anyhow::ensure!(sample[0][3]>0.9,"interior capture is empty: {face}");
            if face==4 { anyhow::ensure!((local.z-19.0).abs()<0.01,"exterior capture missed nearest layer"); }
            else { anyhow::ensure!(local.z<18.5,"near plane failed to expose an interior layer"); }
            let n=Vec3::from_array(sample[1][..3].try_into().unwrap());
            anyhow::ensure!(n.dot(model.inverse().transpose().transform_vector3(Vec3::Z).normalize())>0.99,
                "interior capture has incorrect transformed normal");
        }
        input.draws=&[];
        r.render(&input)?;
        assert_eq!(r.flux_surface.as_ref().unwrap().resident,0,"removed interior cards remained resident");
        // Sparse/high source-draw indices must survive the independent TLAS
        // compaction and grow the per-frame lookup buffer safely. Coincident
        // identical draws share their six material pages, not 5000 copies.
        let aliases=vec![crate::DrawCmd { shadow_casting:crate::ShadowCasting::Off, ..draws[0] };5000];
        input.draws=&aliases;
        r.render(&input)?;
        assert_eq!(r.flux_surface.as_ref().unwrap().resident,6,"identical draws duplicated material pages");
        let instances=r.build_rt_scene(&input).0;
        assert_eq!(instances.last().unwrap().source_draw,4999);
        r.render(&input)?;
        assert_eq!(r.flux_surface.as_ref().unwrap().captures,0,"stable alias mappings invalidated material captures");
        input.draws=&[];
        r.render(&input)?;
        Ok(())
    }

    // A five-sided TV housing has negative signed distances in its open
    // interior. Its bounding box must not hide the screen in front of the back
    // panel, either inside the detail radius or through the coarse GDF.
    fn open_emissive_frame(&self,r:&mut Renderer)->anyhow::Result<()> {
        let data=cube();let mesh=r.upload_mesh(&data)?;
        let mut housing=cube();housing.indices.drain(24..30); // remove +Z face
        let positions:Vec<_>=housing.vertices.iter().map(|v|Vec3::from_array(v.position)).collect();
        let housing_sdf=Arc::new(crate::sdf::generate_sdf(&positions,&housing.indices,32,0.1));
        let housing_mesh=r.upload_mesh(&housing)?;
        let mut screen=cube();screen.indices=data.indices[24..30].to_vec();
        // Only referenced vertices define bounds, just like an imported submesh.
        screen.vertices=data.vertices[16..20].to_vec();
        screen.indices.iter_mut().for_each(|i|*i-=16);
        let positions:Vec<_>=screen.vertices.iter().map(|v|Vec3::from_array(v.position)).collect();
        let screen_sdf=Arc::new(crate::sdf::generate_sdf(&positions,&screen.indices,32,0.1));
        let screen_mesh=r.upload_mesh(&screen)?;
        let receiver=lit(r,[0.7;4],[0.0;3],1.0,0.0);
        let black=lit(r,[0.0,0.0,0.0,1.0],[0.0;3],1.0,0.0);
        let source=lit(r,[0.0,0.0,0.0,1.0],[6.0,3.0,1.0],1.0,0.0);
        let mask=r.upload_texture(&crate::TextureData {width:1,height:1,pixels:vec![0,0,0,255],srgb:false,hdr:false})?;
        let grey=GpuGiMaterial {albedo:[0.7;3],emission:[0.0;3],metallic:0.0,roughness:1.0};
        let opaque=GpuGiMaterial {albedo:[0.0;3],..grey};
        let emissive=GpuGiMaterial {emission:[6.0,3.0,1.0],..opaque};
        r.set_flux_settings(FluxSettings {rt_trace:false,samples:64,bounces:1,reflection_mode:0,ssr_enabled:false,firefly_clamp:32.0,..Default::default()});
        r.set_gi_enabled(true);r.gi_set_emitters(&[]);
        let eye=Vec3::new(0.0,2.5,5.0);
        let camera=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0)};
        for distance in [1.25,4.0] {
            let transform=|scale,position|Mat4::from_scale_rotation_translation(scale,glam::Quat::IDENTITY,position);
            let mut draws=[draw(mesh,receiver,transform(Vec3::new(12.0,0.2,12.0),Vec3::new(0.0,-0.1,0.0)),-1),
                draw(housing_mesh,black,transform(Vec3::new(3.0,2.0,0.4),Vec3::new(0.0,1.0,-distance)),-1),
                draw(screen_mesh,source,transform(Vec3::new(2.9,1.9,1.0),Vec3::new(0.0,1.0,-distance-0.5)),-1)];
            for (i,d) in draws.iter_mut().enumerate() {d.gi_source=Some(crate::GiSource {object:0xf0a0+i as u128,slot:0});}
            let volumes=[cube_distance_field(),housing_sdf.clone(),screen_sdf.clone()];let materials=[grey,opaque,emissive];
            let fields:Vec<_>=draws.iter().enumerate().map(|(i,d)|crate::GiMeshSdf {gi_source:d.gi_source,
                world_to_local:d.transform.inverse(),volume:volumes[i].clone(),material:materials[i],in_global_field:i<2}).collect();
            // Match the engine's signed minimum over nonemissive instances.
            let dims=[48,24,64];let min=Vec3::new(-7.0,-1.0,-7.0);let max=Vec3::new(7.0,5.0,5.0);
            let mut dist=Vec::new();let mut ids=Vec::new();
            for z in 0..dims[2] {for y in 0..dims[1] {for x in 0..dims[0] {
                let p=min+(max-min)*Vec3::new(x as f32/47.0,y as f32/23.0,z as f32/63.0);
                let (who,d)=fields[..2].iter().enumerate().map(|(i,f)| {
                    let scale=1.0/f.world_to_local.x_axis.truncate().length().max(f.world_to_local.y_axis.truncate().length()).max(f.world_to_local.z_axis.truncate().length());
                    (i,f.volume.sample(f.world_to_local.transform_point3(p))*scale)
                }).min_by(|a,b|a.1.total_cmp(&b.1)).unwrap();dist.push(d);ids.push(who as u32);
            }}}
            r.gi_set_gdf_identified(&dist,&ids,dims,min.to_array(),max.to_array(),&materials[..2],&[draws[0].gi_source,draws[1].gi_source])?;
            r.gi_set_mesh_sdfs(&fields)?;
            let mut input=crate::cascade_tests::input(camera);input.viewport_extent=Some([320,180]);input.camera_preview=Some(camera);
            input.light.intensity=0.0;input.light.ambient=[0.0;3];input.light.env_intensity=0.0;input.lights=&[];
            input.postfx.bloom_enabled=false;input.preview_postfx.bloom_enabled=false;input.draw_skybox=false;
            input.draws=&draws;
            let mut energy=[[0.0;2];3];
            for state in 0..3 {
                r.gpu_gi.as_mut().unwrap().importance_sampling=Some(state!=2);
                let mut slots=[None;16];if state==1 {slots[5]=Some(mask);} // emission mask (binding 6)
                r.set_material_textures(source,&slots);
                for _ in 0..128 {r.render(&input)?;}
                for _ in 0..8 {r.render(&input)?;for view in 0..2 {energy[state][view]+=super::flux_radiance::hdr(r,view,camera)?/8.0;}}
            }
            eprintln!("[Flux open emissive frame] distance={distance} lit={:?} masked={:?} uniform={:?}",energy[0],energy[1],energy[2]);
            for view in 0..2 {anyhow::ensure!(energy[0][view]>0.01 && energy[1][view]<energy[0][view]*0.05,
                "open housing hid emission or masked source leaked fallback radiance: {energy:?}");
                anyhow::ensure!((energy[0][view]-energy[2][view]).abs()<energy[2][view]*0.1,
                    "guided sampling changed emissive irradiance: {energy:?}");}
        }
        r.gpu_gi.as_mut().unwrap().importance_sampling=None;
        r.gi_set_mesh_sdfs(&[])?;
        Ok(())
    }

    fn animated_surfaces(&self,r:&mut Renderer,rt:bool) -> anyhow::Result<()> {
        let mesh=r.upload_mesh(&cube())?;
        let receiver=lit(r,[0.7;4],[0.0;3],1.0,0.0);
        let source=lit(r,[0.0,0.0,0.0,1.0],[6.0,3.0,1.0],1.0,0.0);
        r.set_material_features(source,crate::MaterialFeatures {emission:true,..Default::default()});
        let texture=r.upload_texture(&crate::TextureData {width:2,height:1,
            pixels:vec![255,255,255,255,0,0,0,255],srgb:false,hdr:false})?;
        let mut slots=[None;16];slots[3]=Some(texture);r.set_material_textures(source,&slots);
        r.material_params_mut(source).emission_scroll=[1.0,0.0];
        // Deliberately dark fallback means: only the actual animated texture
        // captures may illuminate the receiver, with no analytic emitter proxy.
        r.material_params_mut(source).emission_map_mean=[0.0;3];
        r.upload_material_fx(source);
        let boxes=[(Vec3::new(0.0,-0.1,0.0),Vec3::new(4.0,0.1,4.0)),
            (Vec3::new(-1.5,1.2,0.0),Vec3::new(0.4,0.4,0.15)),
            (Vec3::new(1.5,1.2,0.0),Vec3::new(0.4,0.4,0.15)),
            (Vec3::new(0.0,1.2,-2.0),Vec3::new(0.4,0.4,0.15))];
        let draws:Vec<_>=boxes.iter().enumerate().map(|(i,(c,h))|draw(mesh,if i==0 {receiver}else{source},
            Mat4::from_scale_rotation_translation(*h*2.0,glam::Quat::IDENTITY,*c),-1)).collect();
        let dims=[64,32,64];let min=Vec3::new(-4.0,-1.0,-4.0);let max=Vec3::new(4.0,4.0,4.0);
        let mut dist=Vec::new();let mut indices=Vec::new();
        for z in 0..dims[2] {for y in 0..dims[1] {for x in 0..dims[0] {
            let p=min+(max-min)*Vec3::new(x as f32/63.0,y as f32/31.0,z as f32/63.0);
            let (i,d)=boxes.iter().enumerate().map(|(i,(c,h))| {
                let q=(p-*c).abs()-*h;(i,q.max(Vec3::ZERO).length()+q.max_element().min(0.0))
            }).min_by(|a,b|a.1.total_cmp(&b.1)).unwrap();dist.push(d);indices.push(i as u32);
        }}}
        let materials:Vec<_>=(0..4).map(|i|GpuGiMaterial {albedo:if i==0 {[0.7;3]}else{[0.0;3]},
            emission:[0.0;3],metallic:0.0,roughness:1.0}).collect();
        r.gi_set_gdf(&dist,&indices,dims,min.to_array(),max.to_array(),&materials);
        r.gi_set_mesh_sdfs(&[])?;r.gi_set_emitters(&[]);
        r.set_flux_settings(FluxSettings {rt_trace:rt,samples:32,bounces:2,reflection_mode:0,ssr_enabled:false,..Default::default()});
        r.set_gi_enabled(true);
        let eye=Vec3::new(0.0,2.5,5.0);
        let camera=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),
            proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0)};
        let mut input=crate::cascade_tests::input(camera);input.viewport_extent=Some([320,180]);
        input.camera_preview=Some(camera);input.draws=&draws;input.light.ambient=[0.0;3];
        input.light.intensity=0.0;input.lights=&[];input.gi_debug=6;
        input.postfx.bloom_enabled=false;input.preview_postfx.bloom_enabled=false;
        for frame in 0..48 {
            input.time=0.25+frame as f32*0.00001;r.render(&input)?;
            let cache=r.flux_surface.as_ref().unwrap();
            assert!(cache.captures<=12,"animated refresh exceeded capture budget");
            if frame>=4 {assert_eq!(cache.resident,24,"budgeted animated refresh discarded valid captures");}
        }
        let read=|r:&Renderer| -> anyhow::Result<[[f32;4];2]> {
            let allocator=r.allocator();r.flux_surface.as_ref().unwrap().debug_lighting(&r.ctx.device,
                &mut allocator.lock().unwrap(),r.command_pool,r.ctx.queue,receiver.0,2)
        };
        let bright=read(r)?;
        let scene=patch(&r.debug_read_viewport()?.unwrap(),camera,Vec3::ZERO);
        let preview=patch(&r.debug_read_camera_preview()?.unwrap(),camera,Vec3::ZERO);
        eprintln!("[Flux animated texture] RT={rt}: cached={bright:?}, Scene={scene}, Camera={preview}");
        anyhow::ensure!(bright[1][0]>0.001 && scene>8.0 && preview>8.0,
            "animated material refresh starved receiver lighting: {bright:?}, {scene}, {preview}");
        // Main-engine Flux now uses these captured emitters, not CPU sphere
        // proxies. Check actual HDR energy and mixed/static policy in BOTH views.
        let settings=r.flux_settings;
        let mut strengths=Vec::new();
        for boost in [1.0,2.0,0.0] {
            r.set_flux_settings(FluxSettings {emissive_boost:boost,firefly_clamp:32.0,..settings});
            for _ in 0..96 {r.render(&input)?;}
            let mut value=[0.0;2];
            for _ in 0..8 {r.render(&input)?;for view in 0..2 {value[view]+=super::flux_radiance::hdr(r,view,camera)?/8.0;}}
            strengths.push(value);
        }
        eprintln!("[Flux surface emission strength] RT={rt}: one/two/off={strengths:?}");
        for view in 0..2 {
            anyhow::ensure!(strengths[0][view]>0.01 && (strengths[1][view]/strengths[0][view]-2.0).abs()<0.2
                && strengths[2][view]<0.001,"material emission strength is stale/nonlinear: {strengths:?}");
        }
        r.set_flux_settings(FluxSettings {mixed_lighting:true,firefly_clamp:32.0,..settings});
        for _ in 0..96 {r.render(&input)?;}
        let mut dynamic=[0.0;2];for view in 0..2 {dynamic[view]=super::flux_radiance::hdr(r,view,camera)?;}
        let mut baked_draws=draws.clone();for d in &mut baked_draws[1..] {d.static_geometry=true;}
        input.draws=&baked_draws;
        // Material/source identity edits retain the accepted temporal history;
        // allow its existing decay, without manually clearing renderer history.
        for _ in 0..160 {r.render(&input)?;}
        let mut baked=[0.0;2];for view in 0..2 {baked[view]=super::flux_radiance::hdr(r,view,camera)?;}
        eprintln!("[Flux mixed surface emission] RT={rt}: dynamic={dynamic:?}, baked static={baked:?}");
        for view in 0..2 {anyhow::ensure!(dynamic[view]>0.01 && baked[view]<0.001,
            "mixed lighting lost dynamic emission or doubled baked static emission: {dynamic:?} {baked:?}");}
        input.draws=&draws;r.set_flux_settings(settings);
        for _ in 0..48 {r.render(&input)?;}
        for frame in 0..64 {
            input.time=0.75+frame as f32*0.000001;r.render(&input)?;
            assert_eq!(r.flux_surface.as_ref().unwrap().resident,24,"animated dark frames lost residency");
        }
        let dark=read(r)?;
        let dark_scene=patch(&r.debug_read_viewport()?.unwrap(),camera,Vec3::ZERO);
        let dark_preview=patch(&r.debug_read_camera_preview()?.unwrap(),camera,Vec3::ZERO);
        eprintln!("[Flux animated dark] RT={rt}: cached={dark:?}, Scene={dark_scene}, Camera={dark_preview}");
        anyhow::ensure!(dark[1][0]<bright[1][0]*0.1 && dark_scene<scene*0.5 && dark_preview<preview*0.5,
            "animated black texture retained lighting: {bright:?} -> {dark:?}");
        // With no changing texture/time, drain the refresh queue then go idle.
        for _ in 0..4 {r.render(&input)?;}
        assert_eq!(r.flux_surface.as_ref().unwrap().captures,0,"unchanged animated material was recaptured");
        // Changing animated albedo resets that page's accumulation when it is
        // relit, without making it permanently "new" and starving the floor.
        r.material_params_mut(source).base_color=[1.0;4];
        r.material_params_mut(source).base_scroll=[1.0,0.0];
        r.material_params_mut(source).emission_color=[0.0;3];
        slots[0]=Some(texture);r.set_material_textures(source,&slots);r.upload_material_fx(source);
        let lights=[crate::LightInstance {kind:crate::LightKind::Point,position:Vec3::new(0.0,2.0,2.0),
            direction:Vec3::NEG_Y,color:[1.0;3],intensity:10.0,range:10.0,spot_inner_deg:30.0,spot_outer_deg:45.0,
            cast_shadows:false,soft_shadows:false,radius:0.0,shadow_bias:0.0,baked:false}];
        input.lights=&lights;
        input.time=0.25;
        // Stay below the direct-history count cap so an accidental extra GPU
        // update remains observable even when the point light is deterministic.
        for _ in 0..8 {r.render(&input)?;}
        let read_source=|r:&Renderer| -> anyhow::Result<[[f32;4];2]> {
            let allocator=r.allocator();r.flux_surface.as_ref().unwrap().debug_lighting(&r.ctx.device,
                &mut allocator.lock().unwrap(),r.command_pool,r.ctx.queue,source.0,4)
        };
        let white=read_source(r)?;
        // The shared cache runs once per frame, with independent direct and
        // radiosity budgets. Verify actual GPU values survive frames where only
        // the other component is selected, and that view overrides affect work.
        let updates=|r:&Renderer|r.flux_surface.as_ref().unwrap().debug_lighting_updates(receiver.0,2);
        let mut skipped=[0;2];
        for (scene_speed,camera_speed,scene_visible,expected,frames) in [
            (0.5,0.5,true,[12,6],12),
            (0.5,4.0,true,[24,24],2),
            (16.0,0.5,true,[24,24],2),
            (16.0,0.5,false,[12,6],8),
        ] {
            input.postfx.flux.scene_lighting_update_speed=Some(scene_speed);
            input.preview_postfx.flux.scene_lighting_update_speed=Some(camera_speed);
            input.render_viewport=scene_visible;
            for _ in 0..frames {
                let before=updates(r);let old=read(r)?;
                r.render(&input)?;
                let after=updates(r);let new=read(r)?;
                for kind in 0..2 {
                    assert_eq!(after.0[kind]-before.0[kind],expected[kind],"lighting update budget / view aggregation");
                    if after.1[kind]==before.1[kind] {
                        skipped[kind]+=1;
                        assert_eq!(new[kind],old[kind],"unscheduled lighting was overwritten on GPU");
                    } else {
                        assert_eq!(new[kind][3],(old[kind][3]+1.0).min([32.0,4.0][kind]),
                            "scheduled GPU lighting did not accumulate exactly once");
                    }
                }
            }
        }
        assert!(skipped.iter().all(|n|*n>0),"test did not exercise independent lighting preservation");
        eprintln!("[Flux lighting schedule] RT={rt}: budgets and Scene/Camera overrides pass; preserved={skipped:?}");
        input.render_viewport=true;
        input.postfx.flux.scene_lighting_update_speed=Some(0.5);
        input.preview_postfx.flux.scene_lighting_update_speed=Some(0.5);
        let animated_frames:Vec<_>=(0..120).map(|frame| {
            let mut lights=lights;
            lights[0].intensity=if frame<24 {20.0+4.0*(frame as f32*0.17).sin()} else {0.0};
            lights[0].position.x=0.1*(frame as f32*0.11).cos();
            lights
        }).collect();
        let start=updates(r);
        for changing_lights in &animated_frames[..24] {
            input.lights=changing_lights;
            let before=updates(r);
            r.render(&input)?;
            assert_eq!(r.flux_surface.as_ref().unwrap().debug_lighting_coverage(),[24,24],
                "animated light discarded valid surface lighting");
            let after=updates(r);
            assert_eq!([after.0[0]-before.0[0],after.0[1]-before.0[1]],[12,6],
                "animated light reset scheduling or exceeded the lighting budget");
            input.lights=&lights;
        }
        let end=updates(r);
        // View demand changes which pages consume the fixed budget. Every
        // receiver must still advance within the scheduler's bounded wait.
        assert!(end.1[0]>=start.1[0]+2 && end.1[1]>=start.1[1]+1,
            "animated light starved the receiver's scheduled updates: {:?}->{:?}",start.1,end.1);
        let animated=read(r)?;
        assert_eq!([animated[0][3],animated[1][3]],[1.0,4.0],
            "light changes must refresh direct shading while retaining radiosity accumulation");
        let animated_scene=patch(&r.debug_read_viewport()?.unwrap(),camera,Vec3::ZERO);
        let animated_camera=patch(&r.debug_read_camera_preview()?.unwrap(),camera,Vec3::ZERO);
        anyhow::ensure!(animated[0][0]>0.001 && animated[1][0]>0.001 && animated_scene>8.0 && animated_camera>8.0,
            "animated light lost cached direct/bounced illumination: {animated:?}, Scene={animated_scene}, Camera={animated_camera}");
        // Continue changing the transform with zero light power. This must not
        // retain stale energy or repeatedly restart the page update order.
        for changing_lights in &animated_frames[24..] {
            input.lights=changing_lights;
            r.render(&input)?;
            assert_eq!(r.flux_surface.as_ref().unwrap().debug_lighting_coverage(),[24,24]);
            input.lights=&lights;
        }
        let unlit=read(r)?;
        let unlit_scene=patch(&r.debug_read_viewport()?.unwrap(),camera,Vec3::ZERO);
        let unlit_camera=patch(&r.debug_read_camera_preview()?.unwrap(),camera,Vec3::ZERO);
        eprintln!("[Flux animated light] RT={rt}: cached={animated:?}->{unlit:?}; Scene={animated_scene}->{unlit_scene}; Camera={animated_camera}->{unlit_camera}");
        anyhow::ensure!(unlit[0][0]<0.001 && unlit[1][0]<0.001 && unlit_scene<animated_scene*0.1 && unlit_camera<animated_camera*0.1,
            "moving disabled light retained illumination: {unlit:?}");
        input.postfx.flux.scene_lighting_update_speed=None;
        input.preview_postfx.flux.scene_lighting_update_speed=None;
        for frame in 0..8 {input.time=0.75+frame as f32*0.000001;r.render(&input)?;}
        let black=read_source(r)?;
        eprintln!("[Flux animated albedo] RT={rt}: white={white:?}, black={black:?}");
        anyhow::ensure!(white[0][0]>0.001 && black[0][0]<white[0][0]*0.01,
            "animated material retained bright reflected history: {white:?} -> {black:?}");
        // Model a changing GPU-fed TV image: preserve the texture handle,
        // descriptor, material and time while replacing the image contents.
        // No decoder/render-to-texture scene is involved in this focused test.
        let movie=r.camera_target(0xf10cafe,16,16)?;
        let clear_movie=|r:&Renderer,color:[f32;4]| -> anyhow::Result<()> {
            use ash::vk;
            let device=&r.ctx.device;let image=r.textures[movie.0].image;
            crate::alloc::one_time_submit(device,r.command_pool,r.ctx.queue,|cb|unsafe {
                let range=vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
                let before=vk::ImageMemoryBarrier::default().image(image).subresource_range(range)
                    .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::SHADER_READ).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::ALL_COMMANDS,vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),&[],&[],&[before]);
                device.cmd_clear_color_image(cb,image,vk::ImageLayout::TRANSFER_DST_OPTIMAL,&vk::ClearColorValue {float32:color},&[range]);
                let after=vk::ImageMemoryBarrier::default().image(image).subresource_range(range)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL).new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ);
                device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::TRANSFER,vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),&[],&[],&[after]);
            })
        };
        r.material_params_mut(source).base_color=[0.0,0.0,0.0,1.0];
        r.material_params_mut(source).base_scroll=[0.0;2];
        r.material_params_mut(source).emission_scroll=[0.0;2];
        r.material_params_mut(source).emission_color=[6.0,3.0,1.0];
        slots[0]=None;slots[3]=Some(movie);r.set_material_textures(source,&slots);r.upload_material_fx(source);
        input.lights=&[];input.time=1.0;
        clear_movie(r,[1.0;4])?;
        for _ in 0..48 {r.render(&input)?;}
        let movie_lit=read(r)?;
        let movie_scene=patch(&r.debug_read_viewport()?.unwrap(),camera,Vec3::ZERO);
        let movie_preview=patch(&r.debug_read_camera_preview()?.unwrap(),camera,Vec3::ZERO);
        anyhow::ensure!(movie_lit[1][0]>0.001 && movie_scene>8.0 && movie_preview>8.0,
            "dynamic texture failed to illuminate receiver");
        clear_movie(r,[0.0,0.0,0.0,1.0])?;
        for _ in 0..64 {
            r.render(&input)?;
            assert_eq!(r.flux_surface.as_ref().unwrap().resident,24,"dynamic texture refresh discarded pages");
        }
        let movie_dark=read(r)?;
        let movie_scene_dark=patch(&r.debug_read_viewport()?.unwrap(),camera,Vec3::ZERO);
        let movie_preview_dark=patch(&r.debug_read_camera_preview()?.unwrap(),camera,Vec3::ZERO);
        eprintln!("[Flux dynamic image] RT={rt}: cached={movie_lit:?} -> {movie_dark:?}, Scene={movie_scene}->{movie_scene_dark}, Camera={movie_preview}->{movie_preview_dark}");
        anyhow::ensure!(movie_dark[1][0]<movie_lit[1][0]*0.1 && movie_scene_dark<movie_scene*0.5 && movie_preview_dark<movie_preview*0.5,
            "same-handle dynamic texture retained stale illumination");
        // Scripted material parameters (such as the project's flicker component)
        // must refresh captures without discarding geometry or other receivers.
        clear_movie(r,[1.0;4])?;
        for _ in 0..32 {r.render(&input)?;}
        let before_parameters=updates(r);
        for frame in 0..24 {
            r.material_params_mut(source).emission_intensity=1.0+0.1*(frame as f32*0.3).sin();
            r.render(&input)?;
            let cache=r.flux_surface.as_ref().unwrap();
            assert!(cache.captures<=12);
            assert_eq!(cache.resident,24,"emission parameter edit discarded captured geometry");
            assert_eq!(cache.debug_lighting_coverage(),[24,24],"material edit invalidated unrelated lighting");
        }
        let after_parameters=updates(r);
        assert_eq!([after_parameters.1[0]-before_parameters.1[0],after_parameters.1[1]-before_parameters.1[1]],[24,12]);
        let parameter_lit=read(r)?;
        r.material_params_mut(source).emission_intensity=0.0;
        for _ in 0..64 {r.render(&input)?;}
        let parameter_dark=read(r)?;
        anyhow::ensure!(parameter_lit[1][0]>0.01 && parameter_dark[1][0]<parameter_lit[1][0]*0.1,
            "scripted emission failed to light/darken receiver: {parameter_lit:?}->{parameter_dark:?}");
        input.lights=&lights;
        r.material_params_mut(source).base_color=[0.8,0.8,0.8,1.0];
        for _ in 0..8 {r.render(&input)?;}
        for frame in 0..24 {
            let grey=0.5+0.2*(frame as f32*0.3).sin();
            r.material_params_mut(source).base_color=[grey,grey,grey,1.0];
            r.render(&input)?;
            let cache=r.flux_surface.as_ref().unwrap();
            assert_eq!(cache.resident,24,"base colour edits changed card identity");
            assert_eq!(cache.debug_lighting_coverage(),[24,24],"base colour edits restarted lighting scheduling");
        }
        let tint_lit=read_source(r)?;
        r.material_params_mut(source).base_color=[0.0,0.0,0.0,1.0];
        for _ in 0..8 {r.render(&input)?;}
        let tint_dark=read_source(r)?;
        anyhow::ensure!(tint_lit[0][0]>0.001 && tint_dark[0][0]<0.0001,"material response retained stale reflected light");
        eprintln!("[Flux material parameters] RT={rt}: emission {parameter_lit:?}->{parameter_dark:?}; albedo {tint_lit:?}->{tint_dark:?}");
        // A continuously moving extra object must not invalidate stable pages
        // or consume every radiosity update with its newly created cards.
        let blocker_material=lit(r,[0.0,0.0,0.0,1.0],[0.0;3],1.0,0.0);
        let volume=cube_distance_field();
        let motion_frames:Vec<_>=(0..32).map(|i| {
            let mut d=draws.clone();d.push(draw(mesh,blocker_material,
                Mat4::from_translation(Vec3::new(20.0+i as f32*0.01,0.0,0.0)),-1));d
        }).collect();
        input.postfx.flux.scene_lighting_update_speed=Some(0.5);
        input.preview_postfx.flux.scene_lighting_update_speed=Some(0.5);
        let before_motion=updates(r);
        for moved in &motion_frames {
            input.draws=moved;
            r.gi_set_mesh_sdfs(&[crate::GiMeshSdf { gi_source: None,world_to_local:moved[4].transform.inverse(),volume:volume.clone(),
                material:GpuGiMaterial {albedo:[0.0;3],emission:[0.0;3],metallic:0.0,roughness:1.0},in_global_field:false}])?;
            r.render(&input)?;
            let coverage=r.flux_surface.as_ref().unwrap().debug_lighting_coverage();
            assert!(coverage.iter().all(|n|*n>=24),"moving object invalidated stable lighting: {coverage:?}");
        }
        let after_motion=updates(r);
        assert!(after_motion.1[0]>=before_motion.1[0]+4 && after_motion.1[1]>=before_motion.1[1]+4,
            "new moving cards starved the stable receiver: {:?}->{:?}",before_motion.1,after_motion.1);
        // Move an actual shadow caster over the stable floor, then remove it.
        // Read the floor's cached irradiance, not the black blocker's pixels.
        let unobstructed=read(r)?;
        let mut near=draws.clone();
        let mut blocker=draw(mesh,blocker_material,Mat4::from_scale_rotation_translation(
            Vec3::new(3.0,0.2,3.0),glam::Quat::IDENTITY,Vec3::new(0.0,0.6,0.0)),-1);
        blocker.shadow_casting=crate::ShadowCasting::On;near.push(blocker);
        input.draws=&near;
        r.gi_set_mesh_sdfs(&[crate::GiMeshSdf { gi_source: None,world_to_local:blocker.transform.inverse(),volume:volume.clone(),
            material:GpuGiMaterial {albedo:[0.0;3],emission:[0.0;3],metallic:0.0,roughness:1.0},in_global_field:false}])?;
        for _ in 0..32 {r.render(&input)?;}
        let obstructed=read(r)?;
        anyhow::ensure!(unobstructed[0][0]>0.001 && obstructed[0][0]<unobstructed[0][0]*0.05,
            "retained cache ignored moving occluder: {unobstructed:?}->{obstructed:?}");
        input.draws=&draws;r.gi_set_mesh_sdfs(&[])?;
        for _ in 0..16 {r.render(&input)?;}
        let recovered=read(r)?;
        anyhow::ensure!(recovered[0][0]>unobstructed[0][0]*0.8,"removed occluder left stale cache shadows");
        eprintln!("[Flux cached occlusion] RT={rt}: direct {} -> {} -> {}",unobstructed[0][0],obstructed[0][0],recovered[0][0]);
        // A replacement GDF must retain unaffected world-cache lighting too.
        r.gi_set_gdf(&dist,&indices,dims,min.to_array(),max.to_array(),&materials);
        r.gi_set_mesh_sdfs(&[])?;input.draws=&draws;r.render(&input)?;
        assert_eq!(r.flux_surface.as_ref().unwrap().debug_lighting_coverage(),[24,24]);
        eprintln!("[Flux moving geometry] RT={rt}: stable receiver updates {:?}->{:?}",before_motion.1,after_motion.1);
        input.draws=&[];r.render(&input)?;
        Ok(())
    }

    // Opaque single-sided planes occlude from either side, independently of
    // their main-pass face visibility. Exercise real shadow maps and GI rays.
    fn single_sided_panel(&self,r:&mut Renderer)->anyhow::Result<()> {
        let ground=r.upload_mesh(&cube())?;
        let material=lit(r,[0.8;4],[0.0;3],1.0,0.0);
        r.set_material_features(material,crate::MaterialFeatures {double_sided:false,..Default::default()});
        // Black panel prevents legitimate floor->panel->floor bounce around
        // its open edges from obscuring the through-wall occlusion check.
        let panel_material=lit(r,[0.0,0.0,0.0,1.0],[0.0;3],1.0,0.0);
        let sky_wall_material=lit(r,[0.0,0.0,0.0,1.0],[0.0;3],1.0,0.0);
        let floor=draw(ground,material,Mat4::from_scale_rotation_translation(Vec3::new(6.0,0.2,6.0),glam::Quat::IDENTITY,Vec3::new(0.0,-0.1,0.0)),-1);
        let eye=Vec3::new(0.0,0.6,3.0);
        let camera=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0)};
        let lights=[crate::LightInstance {kind:crate::LightKind::Point,position:Vec3::new(0.0,2.0,0.0),direction:Vec3::NEG_Y,
            color:[1.0;3],intensity:15.0,range:10.0,spot_inner_deg:30.0,spot_outer_deg:45.0,
            cast_shadows:true,soft_shadows:false,radius:0.0,shadow_bias:0.0,baked:false}];
        let emitters=[crate::GpuGiEmitter {center:[0.0,2.0,0.0],radius:0.25,emission:[8.0;3]}];
        for facing in [1.0f32,-1.0] {
            let positions=[Vec3::new(-2.0,1.0,-2.0),Vec3::new(2.0,1.0,-2.0),Vec3::new(2.0,1.0,2.0),Vec3::new(-2.0,1.0,2.0)];
            let indices=if facing>0.0 {vec![0,2,1,0,3,2]} else {vec![0,1,2,0,2,3]};
            let data=crate::MeshData {vertices:positions.iter().map(|p|crate::Vertex {position:p.to_array(),normal:[0.0,facing,0.0],color:[1.0;4],..Default::default()}).collect(),indices:indices.clone(),has_lightmap_uv:false};
            let mesh=r.upload_mesh(&data)?;
            let mut panel=draw(mesh,panel_material,Mat4::IDENTITY,-1);panel.shadow_casting=crate::ShadowCasting::On;
            let bare=[floor];let covered=[floor,panel];
            // Side walls stop genuinely escaped near-horizontal sky rays, so
            // closing the ceiling tests coverage rather than finite trace range.
            let wall=|position,scale|draw(ground,sky_wall_material,Mat4::from_scale_rotation_translation(scale,glam::Quat::IDENTITY,position),-1);
            let large=[floor,panel,
                wall(Vec3::new(-5.0,5.0,0.0),Vec3::new(0.2,10.0,10.0)),wall(Vec3::new(5.0,5.0,0.0),Vec3::new(0.2,10.0,10.0)),
                wall(Vec3::new(0.0,5.0,-5.0),Vec3::new(10.0,10.0,0.2)),wall(Vec3::new(0.0,5.0,5.0),Vec3::new(10.0,10.0,0.2))];
            let mut large=large;large[1].transform=Mat4::from_scale(Vec3::new(50.0,1.0,50.0));
            let volume=Arc::new(crate::sdf::generate_sdf(&positions,&indices,32,0.1));
            let fields=[crate::GiMeshSdf {gi_source:None,world_to_local:Mat4::IDENTITY,volume,
                material:GpuGiMaterial {albedo:[0.0;3],emission:[0.0;3],metallic:0.0,roughness:1.0},in_global_field:false}];
            let mut input=crate::cascade_tests::input(camera);
            input.viewport_extent=Some([323,181]);input.camera_preview=Some(camera);
            input.light.intensity=0.0;input.light.ambient=[0.0;3];
            input.postfx.bloom_enabled=false;input.preview_postfx.bloom_enabled=false;
            for mode in [0,1,2] {
                if mode==2 && !r.ctx.ray_tracing() {continue;}
                r.set_flux_settings(FluxSettings {rt_trace:mode==2,samples:64,reflection_mode:0,ssr_enabled:false,..Default::default()});
                r.set_gi_enabled(mode!=0);input.gi_debug=if mode==0 {0} else {6};
                input.lights=if mode==0 {&lights} else {&[]};
                r.gi_set_emitters(if mode==0 {&[]} else {&emitters});
                r.gi_set_gdf(&vec![10.0;512],&vec![0;512],[8;3],[-10.0;3],[10.0;3],&[fields[0].material]);
                r.gi_set_mesh_sdfs(&[])?;input.draws=&bare;
                for _ in 0..24 {r.render(&input)?;}
                let lit=patch(&r.debug_read_viewport()?.unwrap(),camera,Vec3::ZERO);
                let lit_camera=patch(&r.debug_read_camera_preview()?.unwrap(),camera,Vec3::ZERO);
                r.gi_set_mesh_sdfs(&fields)?;input.draws=&covered;
                for _ in 0..32 {r.render(&input)?;}
                let dark=patch(&r.debug_read_viewport()?.unwrap(),camera,Vec3::ZERO);
                let dark_camera=patch(&r.debug_read_camera_preview()?.unwrap(),camera,Vec3::ZERO);
                eprintln!("[Flux single-sided panel] mode={mode}, facing={facing}: Scene {lit}->{dark}, Camera {lit_camera}->{dark_camera}");
                anyhow::ensure!(lit>30.0 && lit_camera>30.0 && dark<if mode==0 {5.0} else {2.0} && dark_camera<if mode==0 {5.0} else {2.0},
                    "single-sided panel leaked light: mode={mode}, facing={facing}, Scene={lit}->{dark}, Camera={lit_camera}->{dark_camera}");
                if mode==2 {
                    let open=r.upload_texture(&crate::TextureData {width:1,height:1,pixels:vec![255,255,255,0],srgb:false,hdr:false})?;
                    let solid=r.upload_texture(&crate::TextureData {width:1,height:1,pixels:vec![255;4],srgb:false,hdr:false})?;
                    let mut slots=[None;16];
                    for (case,texture,base_alpha,cutoff) in [(0,open,1.0,0.5),(1,solid,1.0,0.5),(2,solid,0.25,0.5),(3,solid,0.25,0.1)] {
                        slots[0]=Some(texture);r.set_material_textures(panel_material,&slots);
                        r.set_material_features(panel_material,crate::MaterialFeatures {alpha_mode:crate::AlphaMode::Cutout,..Default::default()});
                        r.material_params_mut(panel_material).base_color[3]=base_alpha;r.material_params_mut(panel_material).alpha_cutoff=cutoff;
                        for _ in 0..48 {r.render(&input)?;}
                        let scene=patch(&r.debug_read_viewport()?.unwrap(),camera,Vec3::ZERO);
                        let camera_value=patch(&r.debug_read_camera_preview()?.unwrap(),camera,Vec3::ZERO);
                        let alloc=r.allocator();let cache=r.flux_surface.as_ref().unwrap().debug_lighting(&r.ctx.device,&mut alloc.lock().unwrap(),r.command_pool,r.ctx.queue,material.0,2)?;
                        eprintln!("[Flux GI mask emitter] facing={facing} case={case}: Scene={scene} Camera={camera_value} cache={cache:?}");
                        let open=case==0 || case==2;
                        anyhow::ensure!(if open {scene>30.0 && camera_value>30.0 && cache[0][0]>0.01} else {scene<2.0 && camera_value<2.0 && cache[0][0]<0.001},
                            "masked emitter visibility/cache mismatch: {case} {scene}/{camera_value} {cache:?}");
                    }
                    // No analytic emitter: sky reaches the receiver only via
                    // primary gather and the cache's separate radiosity rays.
                    r.gi_set_emitters(&[]);input.light.ambient=[1.0;3];
                    input.draws=&large;
                    r.set_flux_settings(FluxSettings {rt_trace:true,samples:64,bounces:2,reflection_mode:0,ssr_enabled:false,..Default::default()});
                    let mut sky_results=Vec::new();
                    for alpha in [0.0,1.0,0.0] {
                        r.material_params_mut(panel_material).base_color[3]=alpha;
                        for _ in 0..if alpha==0.0 {64} else {160} {r.render(&input)?;}
                        let scene=patch(&r.debug_read_viewport()?.unwrap(),camera,Vec3::ZERO);
                        let preview=patch(&r.debug_read_camera_preview()?.unwrap(),camera,Vec3::ZERO);
                        let alloc=r.allocator();let cache=r.flux_surface.as_ref().unwrap().debug_lighting(&r.ctx.device,&mut alloc.lock().unwrap(),r.command_pool,r.ctx.queue,material.0,2)?;
                        eprintln!("[Flux GI mask sky] facing={facing} alpha={alpha}: Scene={scene} Camera={preview} indirect={:?}",cache[1]);
                        sky_results.push([scene,preview,cache[1][0]]);
                    }
                    for view in 0..3 {anyhow::ensure!(sky_results[0][view]>if view==2 {0.1} else {30.0}
                        && sky_results[1][view]<sky_results[0][view]*0.15 && sky_results[2][view]>sky_results[0][view]*0.8,
                        "masked sky/radiosity did not open, close and reopen: {sky_results:?}");}
                    input.draws=&covered;input.light.ambient=[0.0;3];
                    r.set_material_textures(panel_material,&[None;16]);r.material_params_mut(panel_material).base_color[3]=1.0;
                    r.set_material_features(panel_material,crate::MaterialFeatures::default());
                }
            }
        }
        r.gi_set_mesh_sdfs(&[])?;r.gi_set_emitters(&[]);
        Ok(())
    }

    fn shadow_budget(&self, r: &mut Renderer) -> anyhow::Result<()> {
        r.set_gi_enabled(false);
        let mesh = r.upload_mesh(&cube())?;
        let material = lit(r, [0.8; 4], [0.0; 3], 1.0, 0.0);
        let mut draws = [
            draw(mesh, material, Mat4::from_scale_rotation_translation(Vec3::new(8.0, 0.2, 8.0), glam::Quat::IDENTITY, Vec3::new(0.0, -0.1, 0.0)), -1),
            draw(mesh, material, Mat4::from_scale_rotation_translation(Vec3::splat(1.0), glam::Quat::IDENTITY, Vec3::new(0.0, 1.0, 0.0)), -1),
        ];
        for d in &mut draws { d.shadow_casting = crate::ShadowCasting::On; }
        let lights: Vec<_> = (0..7).map(|i| crate::LightInstance {
            kind: crate::LightKind::Point,
            position: if i == 0 { Vec3::new(0.0, 3.0, 0.0) } else { Vec3::new(3.0 + i as f32, 1.0, 2.0) },
            direction: Vec3::NEG_Y, color: [1.0; 3], intensity: 15.0,
            range: if i == 0 { 10.0 } else { 0.5 }, spot_inner_deg: 30.0, spot_outer_deg: 45.0,
            cast_shadows: true, soft_shadows: true, radius: 0.0, shadow_bias: 0.005, baked: false,
        }).collect();
        let camera = |x: f32| {
            let eye = Vec3::new(x, 1.5, 5.0);
            CameraData { position: eye, view: Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y),
                proj: Mat4::perspective_rh(1.0, 16.0/9.0, 0.1, 100.0) }
        };
        let mut input = crate::cascade_tests::input(camera(0.0));
        input.viewport_extent = Some([640, 360]);
        input.draws = &draws;
        input.lights = &lights;
        input.light.ambient = [0.0; 3];
        input.light.intensity = 0.0;
        input.light.env_intensity = 0.0;
        input.shadow_pcf_texel = 1.0 / 2048.0;
        let point = Vec3::new(0.0, 0.001, 0.0);
        for x in [0.0, 6.0, -3.0, 0.0] {
            input.camera = camera(x);
            for _ in 0..3 { r.render(&input)?; }
            let shadow = patch(&r.debug_read_viewport()?.unwrap(), input.camera, point);
            let ubo = r.last_frame_ubo.as_ref().unwrap();
            anyhow::ensure!(ubo.lights[..7].iter().all(|l| l.spot[1] >= 0.0), "a point light lost its shadow allocation");
            anyhow::ensure!(r.shadow_atlas.layers >= 42, "physical shadow maps not allocated");
            eprintln!("[Flux shadow budget] camera x={x}: shadow={shadow:.1}, {} maps at {}px", r.shadow_atlas.layers, r.shadow_atlas.dim);
            anyhow::ensure!(shadow < 20.0, "lamp shadow disappeared while moving: {shadow}");
        }
        let mut unshadowed = lights.clone();
        for l in &mut unshadowed { l.cast_shadows = false; }
        input.lights = &unshadowed;
        for _ in 0..3 { r.render(&input)?; }
        let unshadowed = patch(&r.debug_read_viewport()?.unwrap(), input.camera, point);
        anyhow::ensure!(unshadowed > 50.0, "shadow regression's control is not lit: {unshadowed}");
        Ok(())
    }

    fn bounce(&self, r: &mut Renderer, rt: bool, check_balanced: bool) -> anyhow::Result<()> {
        let mesh = r.upload_mesh(&cube())?;
        let grey = lit(r, [0.7; 4], [0.0; 3], 1.0, 0.0);
        let red = lit(r, [0.8, 0.02, 0.02, 1.0], [0.0; 3], 1.0, 0.0);
        let boxes = [(Vec3::new(0.0, -0.1, 0.0), Vec3::new(4.0, 0.1, 4.0)),
            (Vec3::new(1.5, 1.5, 0.0), Vec3::new(0.1, 1.5, 3.0))];
        let draws: Vec<_> = boxes.iter().enumerate().map(|(i, (p, h))| {
            let mut d = draw(mesh, if i == 0 { grey } else { red },
                Mat4::from_scale_rotation_translation(*h * 2.0, glam::Quat::IDENTITY, *p), -1);
            d.shadow_casting = crate::ShadowCasting::On;
            d
        }).collect();
        let dims = [64, 32, 64];
        let min = Vec3::new(-4.0, -1.0, -4.0);
        let max = Vec3::new(4.0, 4.0, 4.0);
        let mut dist = Vec::new();
        let mut index = Vec::new();
        for z in 0..dims[2] { for y in 0..dims[1] { for x in 0..dims[0] {
            let p = min + (max-min) * Vec3::new(x as f32/63.0, y as f32/31.0, z as f32/63.0);
            let (i, d) = boxes.iter().enumerate().map(|(i, (c, h))| {
                let q = (p-*c).abs()-*h;
                (i, q.max(Vec3::ZERO).length()+q.max_element().min(0.0))
            }).min_by(|a,b| a.1.total_cmp(&b.1)).unwrap();
            dist.push(d); index.push(i as u32);
        } } }
        r.gi_set_gdf(&dist, &index, dims, min.to_array(), max.to_array(), &[
            GpuGiMaterial { albedo: [0.7; 3], emission: [0.0; 3], metallic: 0.0, roughness: 1.0 },
            GpuGiMaterial { albedo: [0.8, 0.02, 0.02], emission: [0.0; 3], metallic: 0.0, roughness: 1.0 },
        ]);
        r.gi_set_emitters(&[]);
        r.set_flux_settings(FluxSettings { rt_trace: rt, samples: 32, bounces: 2,
            reflection_mode: 0, ssr_enabled: false, ..Default::default() });
        let eye = Vec3::new(0.0, 2.5, 5.0);
        let camera = CameraData { position: eye, view: Mat4::look_at_rh(eye, Vec3::new(0.0, 0.3, 0.0), Vec3::Y),
            proj: Mat4::perspective_rh(1.0, 16.0/9.0, 0.1, 100.0) };
        let mut input = crate::cascade_tests::input(camera);
        input.viewport_extent = Some([320, 180]);
        input.camera_preview = Some(camera);
        input.draws = &draws;
        input.light.ambient = [0.0; 3];
        input.light.intensity = 0.0;
        input.gi_debug = 6;
        input.postfx.bloom_enabled = false;
        input.preview_postfx.bloom_enabled = false;
        let lights = [crate::LightInstance { kind: crate::LightKind::Point,
            position: Vec3::new(0.0, 1.5, 0.0), direction: Vec3::NEG_Y,
            color: [1.0; 3], intensity: 15.0, range: 10.0, spot_inner_deg: 30.0,
            spot_outer_deg: 45.0, cast_shadows: false, soft_shadows: false, radius: 0.0, shadow_bias: 0.0, baked: false }];
        input.lights = &lights;
        for _ in 0..20 { r.render(&input)?; }
        let point = Vec3::new(0.5, 0.0, 0.0);
        let a = patch(&r.debug_read_viewport()?.unwrap(), camera, point);
        let b = patch(&r.debug_read_camera_preview()?.unwrap(), camera, point);
        eprintln!("[Flux test] RT={rt}: point-light bounce Scene={a:.1}, Camera={b:.1}");
        anyhow::ensure!(a > 25.0 && b > 25.0, "missing diffuse light bounce: {a}, {b}");
        if check_balanced {self.balanced_gather(r,&mut input)?;}

        input.viewport_extent = Some([1280, 720]);
        input.camera_preview = None;
        for _ in 0..40 { r.render(&input)?; }
        let mut lo = f32::INFINITY;
        let mut hi = 0.0f32;
        for frame in 0..40 {
            let eye = eye + Vec3::X * (frame as f32 * 0.025);
            let moving_camera = CameraData { position: eye,
                view: Mat4::look_at_rh(eye, Vec3::new(0.0, 0.3, 0.0), Vec3::Y), ..camera };
            input.camera = moving_camera;
            r.render(&input)?;
            let value = patch(&r.debug_read_viewport()?.unwrap(), moving_camera, point);
            lo = lo.min(value); hi = hi.max(value);
        }
        eprintln!("[Flux motion] RT={rt}: fixed world surface while panning = {lo:.1}..{hi:.1}");
        anyhow::ensure!(hi-lo < 5.0, "GI shifts with the camera: {lo}..{hi}");
        input.camera = camera;
        input.camera_preview = Some(camera);
        input.viewport_extent = Some([320, 180]);

        let read_light = |r: &Renderer| -> anyhow::Result<[[f32;4];2]> {
            let allocator=r.allocator();
            r.flux_surface.as_ref().unwrap().debug_lighting(&r.ctx.device,&mut allocator.lock().unwrap(),
                r.command_pool,r.ctx.queue,grey.0,2)
        };
        let physical=read_light(r)?;
        eprintln!("[Flux radiosity] RT={rt}: direct={:?}, indirect={:?}",physical[0],physical[1]);
        anyhow::ensure!(physical[0][0]>0.01 && physical[1][0]>0.001,"surface cache did not accumulate bounce radiance");
        let settings=r.flux_settings;
        r.set_flux_settings(FluxSettings {intensity:4.0,..settings});
        for _ in 0..4 {r.render(&input)?;}
        let exposed=read_light(r)?;
        anyhow::ensure!(physical[0]==exposed[0],"display GI intensity fed back into cached physical lighting");
        r.set_flux_settings(settings);
        input.lights=&[];
        for _ in 0..40 {r.render(&input)?;}
        let dark=read_light(r)?;
        anyhow::ensure!(dark[0][..3].iter().chain(&dark[1][..3]).all(|v| *v<0.001),
            "cache retained stale energy after the lights switched off: {dark:?}");
        input.lights=&lights;
        for _ in 0..30 {r.render(&input)?;}

        if rt {
            // glTF stores metallic=1 as a multiplier even on a dielectric ORM
            // texture. Deliberately wrong fallback means verify that the ray
            // hit reads the captured ORM texture, rather than the old average.
            let orm = r.upload_texture(&crate::TextureData { width: 1, height: 1,
                pixels: vec![255,255,0,255], srgb: false, hdr: false })?;
            for material in [grey, red] {
                r.material_params_mut(material).metallic = 1.0;
                r.set_material_metallic_mean(material, 1.0);
                let mut slots = [None;16]; slots[2] = Some(orm);
                r.set_material_textures(material, &slots);
            }
            r.invalidate_screen_gi();
            for _ in 0..20 { r.render(&input)?; }
            let dielectric = patch(&r.debug_read_viewport()?.unwrap(), camera, point);
            for material in [grey, red] {
                r.set_material_metallic_mean(material, 1.0);
                r.set_material_textures(material, &[None;16]);
            }
            r.invalidate_screen_gi();
            for _ in 0..20 { r.render(&input)?; }
            let metal = patch(&r.debug_read_viewport()?.unwrap(), camera, point);
            eprintln!("[Flux test] ORM mean: dielectric={dielectric:.1}, metal={metal:.1}");
            anyhow::ensure!(dielectric > 25.0 && metal < dielectric * 0.25,
                "GI ignored texture metalness: dielectric={dielectric}, metal={metal}");
            for material in [grey, red] { r.material_params_mut(material).metallic = 0.0; }
            r.invalidate_screen_gi();
        }

        // Equal-resolution views must agree on bounce lighting as well as sky.
        input.viewport_extent = Some([1280, 720]);
        for _ in 0..30 { r.render(&input)?; }
        let a = patch(&r.debug_read_viewport()?.unwrap(), camera, point);
        let b = patch(&r.debug_read_camera_preview()?.unwrap(), camera, point);
        eprintln!("[Flux test] RT={rt}: equal-resolution bounce Scene={a:.1}, Camera={b:.1}");
        anyhow::ensure!((a-b).abs() < 20.0, "bounce differs between equal-resolution views: {a}, {b}");
        // Per-camera volume settings must not bleed into the other view. A
        // short trace cannot reach the red wall, while the preview retains it.
        input.postfx.flux.max_trace_distance = Some(0.01);
        input.postfx.flux.final_gather_update_speed = Some(4.0);
        for _ in 0..40 { r.render(&input)?; }
        let short = patch(&r.debug_read_viewport()?.unwrap(), camera, point);
        let long = patch(&r.debug_read_camera_preview()?.unwrap(), camera, point);
        eprintln!("[Flux volumes] RT={rt}: short-trace Scene={short:.1}, inherited Camera={long:.1}");
        anyhow::ensure!(short < 15.0 && long > 100.0, "camera GI overrides leaked: {short}, {long}");
        input.postfx.flux = crate::FluxPostFx::default();
        for _ in 0..40 { r.render(&input)?; }
        input.viewport_extent = Some([320, 180]);

        input.lights = &[];
        r.gi_set_emitters(&[crate::GpuGiEmitter { center: [0.0, 1.0, 0.0], radius: 0.25, emission: [4.0, 0.0, 0.0] }]);
        for _ in 0..8 { r.render(&input)?; }
        let a = patch(&r.debug_read_viewport()?.unwrap(), camera, point);
        let b = patch(&r.debug_read_camera_preview()?.unwrap(), camera, point);
        eprintln!("[Flux test] RT={rt}: emissive lighting Scene={a:.1}, Camera={b:.1}");
        anyhow::ensure!(a > 25.0 && b > 25.0, "missing emissive lighting: {a}, {b}");
        let mut blocked = draws.clone();
        if rt {
            // A moving blocker belongs to the TLAS, not the cached static GDF.
            // FluxRT must shadow the emitter against that actual geometry.
            let mut blocker = draw(mesh, grey, Mat4::from_scale_rotation_translation(
                Vec3::new(1.4, 0.1, 1.4), glam::Quat::IDENTITY, Vec3::new(0.25, 0.5, 0.0)), -1);
            blocker.shadow_casting = crate::ShadowCasting::On;
            blocked.push(blocker);
            input.draws = &blocked;
            for _ in 0..20 { r.render(&input)?; }
            let shadow = patch(&r.debug_read_viewport()?.unwrap(), camera, point);
            eprintln!("[Flux test] RT dynamic emitter occlusion: lit={a:.1}, shadow={shadow:.1}");
            anyhow::ensure!(shadow < a * 0.5, "FluxRT emitter ignores TLAS occluder: lit={a}, shadow={shadow}");
        }
        r.gi_set_emitters(&[]);
        // Measure a visible, continually updating view at a realistic internal
        // resolution. Exclude warmup and the second camera from the GI budget.
        input.draws = &draws;
        input.lights = &lights;
        input.camera_preview = None;
        input.viewport_extent = Some([1280, 720]);
        r.set_flux_settings(FluxSettings { rt_trace: rt, reflection_mode: 0,
            ssr_enabled: false, ..Default::default() });
        let mut timings = Vec::new();
        for frame in 0..30 {
            r.render(&input).map_err(|e| anyhow::anyhow!("Flux perf RT={rt}, frame={frame}: {e:#}"))?;
            if frame >= 10 { timings.push(r.stats().gpu_gi_ms); }
        }
        timings.sort_by(f32::total_cmp);
        let extent = r.viewport_target.as_ref().unwrap().extent;
        assert_eq!([extent.width, extent.height], [1280, 720]);
        anyhow::ensure!(timings[timings.len()/2] > 0.0, "GI timing was not recorded");
        eprintln!("[Flux perf] RT={rt}, 1280x720, 16 rays, 2 bounces: median GI {:.3} ms, max {:.3} ms",
            timings[timings.len()/2], timings.last().unwrap());
        if rt {
            // A light's centre can sit inside a fixture while its finite source
            // extends outside it. The radius must reach the realtime tracer.
            let mut fixtures = draws.clone();
            let mut fixture = draw(mesh, grey, Mat4::from_scale_rotation_translation(
                Vec3::splat(0.3), glam::Quat::IDENTITY, lights[0].position), -1);
            fixture.shadow_casting = crate::ShadowCasting::On;
            fixtures.push(fixture);
            input.draws = &fixtures;
            input.viewport_extent = Some([320, 180]);
            input.camera_preview = None;
            r.invalidate_screen_gi();
            for _ in 0..30 { r.render(&input)?; }
            let point_source = patch(&r.debug_read_viewport()?.unwrap(), camera, point);
            let finite_lights = [crate::LightInstance { radius: 0.9, ..lights[0] }];
            input.lights = &finite_lights;
            r.invalidate_screen_gi();
            for _ in 0..30 { r.render(&input)?; }
            let area_source = patch(&r.debug_read_viewport()?.unwrap(), camera, point);
            eprintln!("[Flux test] finite source in fixture: point={point_source:.1}, area={area_source:.1}");
            anyhow::ensure!(area_source > point_source + 25.0, "source radius was lost before the GI trace");
        }
        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if !self.done {
            self.done = true;
            self.error = self.run(el).err().map(|e| format!("{e:#}"));
            el.exit();
        }
    }
    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, _: WindowEvent) {}
}

#[test]
fn flux_lights_scene_and_camera_at_odd_extents() {
    run_app(false);
}

#[test]
fn flux_backfaces_occlude_without_discarding_valid_receivers() {
    run_app(true);
}

fn run_app(backface_only:bool) {
    if std::env::var("ZESTENGINE_TEST_GPU").is_ok_and(|v| v == "0") || std::env::var_os("DISPLAY").is_none() {
        eprintln!("[Flux test] no display / GPU disabled: skipped");
        return;
    }
    let _gpu = crate::cluster_headless::gpu_lock();
    let validation_before = crate::context::validation_error_count();
    let mut el = EventLoop::builder().with_x11().with_any_thread(true).build().unwrap();
    let mut app = App {backface_only,..Default::default()};
    el.run_app_on_demand(&mut app).unwrap();
    assert!(app.error.is_none(), "{}", app.error.unwrap_or_default());
    assert_eq!(crate::context::validation_error_count(), validation_before,
        "Flux Scene/Camera run produced Vulkan validation errors; see diagnostics above");
}
