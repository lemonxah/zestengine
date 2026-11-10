//! Share material captures and surface transport with standalone world reflections.
use super::*;
use bytemuck::Zeroable;
impl Renderer {
    pub(super) fn flux_lighting_quality(&self, post: FluxPostFx) -> Option<f32> {
        Some(post.scene_lighting_quality.filter(|v|v.is_finite()).unwrap_or(self.flux_settings.scene_lighting_quality))
    }
    pub(super) fn flux_lighting_speed(&self, post: FluxPostFx) -> Option<f32> {
        Some(post.scene_lighting_update_speed.filter(|v|v.is_finite()).unwrap_or(self.flux_settings.scene_lighting_update_speed))
    }
    pub(super) fn capture_flux_surfaces(&mut self, device:&ash::Device, cb:vk::CommandBuffer, input:&FrameInput)
        -> Result<Vec<flux_surface::LightingView>> {
        let views:Vec<_>=(input.render_viewport || input.camera_preview.is_none()).then_some((input.camera,input.postfx.flux)).into_iter()
            .chain(input.camera_preview.map(|camera|(camera,input.preview_postfx.flux)))
            .map(|(camera,post)|flux_surface::LightingView::from(camera).with_surface_resolution(post.surface_cache_resolution).with_lighting_quality(self.flux_lighting_quality(post))).collect();
        let allocator=self.allocator();let mut alloc=allocator.lock().unwrap();
        if self.flux_surface.is_none() {
            self.flux_surface=Some(flux_surface::SurfaceCache::new(device,&mut alloc,self.pipeline_cache.set1_layout)?);
        }
        let mesh_sources=self.gpu_gi.as_ref().and_then(|g|g.mesh_sdfs.current.as_ref()).map_or(&[][..],|s|s.sources.as_slice());
        let gdf_sources=self.gpu_gi.as_ref().map_or(&[][..],|g|g.gdf_sources());
        self.flux_surface.as_mut().unwrap().record(device,&mut alloc,cb,self.frame_index,self.frame_serial,input.time,input.draws,
            &self.meshes,&self.materials,&self.mesh_versions,self.skinner.as_ref(),&self.texture_revisions,mesh_sources,gdf_sources,&views)?;
        Ok(views)
    }
    /// Active world-reflection views, also used to keep off-screen video feeds live.
    pub(super) fn reflection_views<'a>(&self,input:&'a FrameInput)->impl Iterator<Item=(CameraData,FluxSettings)>+'a {
        let settings=self.flux_settings;let supported=self.tier.reflections();
        input.render_viewport.then_some((input.camera,input.postfx.flux)).into_iter()
            .chain(input.camera_preview.map(|c|(c,input.preview_postfx.flux)))
            .map(move |(c,post)|(c,settings.for_view(post)))
            .filter(move |(_,fs)|supported && fs.reflection_mode==2 && fs.ssr_intensity>0.0)
    }
    pub(super) fn surface_lights(input:&FrameInput, mixed:bool)->Vec<BakeLight> {
        let mut lights:Vec<_>=input.lights.iter().filter(|l|!(mixed && l.baked)).map(|l|BakeLight {
            kind:l.kind,position:l.position,direction:l.direction,color:l.color.map(|c|c*l.intensity),
            range:l.range,spot_inner_deg:l.spot_inner_deg,spot_outer_deg:l.spot_outer_deg,radius:l.radius.max(0.0),
        }).collect();
        if input.lights.is_empty() && !mixed && input.light.intensity>0.0 {
            lights.push(BakeLight {kind:LightKind::Directional,position:Vec3::ZERO,direction:input.light.direction,
                color:input.light.color.map(|c|c*input.light.intensity),range:0.0,spot_inner_deg:0.0,spot_outer_deg:0.0,radius:0.0});
        }
        lights
    }
    pub(super) fn packed_surface_lights(input:&FrameInput,mixed:bool,areas:&[GpuLight])->Vec<GpuLight> {
        let mut lights:Vec<_>=Self::surface_lights(input,mixed).iter().map(|l|GpuLight {
            pos_kind:[l.position.x,l.position.y,l.position.z,match l.kind {LightKind::Directional=>0.0,LightKind::Point=>1.0,LightKind::Spot=>2.0}],
            dir_range:[l.direction.x,l.direction.y,l.direction.z,l.range],color:[l.color[0],l.color[1],l.color[2],0.0],
            spot:[l.spot_inner_deg,l.spot_outer_deg,l.radius,0.0],
        }).collect();lights.extend_from_slice(areas);lights
    }
    pub(super) fn record_standalone_reflection_surface(&mut self,device:&ash::Device,cb:vk::CommandBuffer,input:&FrameInput)->Result<()> {
        // The existing diffuse path owns this frame's shared update when active.
        if self.gi_has_gdf() || self.gpu_gi.is_none() {return Ok(());}
        let settings:Vec<_>=self.reflection_views(input).map(|(_,fs)|fs).collect();
        let Some(mut fs)=settings.first().copied() else {return Ok(());};
        fs.bounces=settings.iter().map(|s|s.bounces).max().unwrap_or(fs.bounces);
        let hardware=fs.rt_trace && self.rt_reflect.is_some();
        let scene=if hardware {None} else {self.gpu_gi.as_ref().and_then(|g|g.reflection_scene())};
        if !hardware && scene.is_none() {return Ok(());}
        let rt=if hardware {
            let (instances,_)=self.rt_scene(input);let water=self.rt_water(input,&instances);
            let Some((tlas,buffer))=build_gi_rt_tlas(&self.ctx,self.allocator.clone(),self.command_pool,&self.meshes,
                &mut self.rt_reflect,&instances,&water,cb,self.frame_index) else {return Ok(());};
            let views=self.rt_texture_views();
            let textures=self.gpu_gi.as_mut().unwrap().rt_textures.as_mut().unwrap().prepare(device,self.frame_index,&views);
            self.tex_view_cache=views;
            Some((tlas,buffer,textures))
        } else {None};
        let views=self.capture_flux_surfaces(device,cb,input)?;
        let lights=Self::surface_lights(input,fs.mixed_lighting);
        let sky=if fs.mixed_lighting {[0.0;3]} else {input.light.ambient};
        let gi=self.gpu_gi.as_ref().unwrap();
        let bounds=self.flux_surface.as_ref().unwrap().bounds()
            .unwrap_or([input.camera.position-Vec3::ONE,input.camera.position+Vec3::ONE]);
        let automatic_range=(bounds[1]-bounds[0]).length();
        let trace_distance=settings.iter().map(|s|if s.ssr_max_distance>0.0 {s.ssr_max_distance} else {automatic_range})
            .fold(0.02,f32::max);
        let signature=flux_surface::lighting_signature(&lights,&[],sky,hardware,fs.bounces,fs.mixed_lighting,
            gi.gdf_revision,gi.global_field_revision,gi.material_revision,fs.emissive_boost,trace_distance,&self.packed_area_lights,&self.area_sources);
        let speed=flux_surface::shared_lighting_speed((input.render_viewport || input.viewport_extent.is_some()).then_some(self.flux_lighting_speed(input.postfx.flux)),
            input.camera_preview.map(|_|self.flux_lighting_speed(input.preview_postfx.flux)));
        let resolution=flux_surface::shared_lighting_resolution(
            (input.render_viewport || input.viewport_extent.is_some() || input.camera_preview.is_none()).then_some(self.flux_lighting_quality(input.postfx.flux)),
            input.camera_preview.map(|_|self.flux_lighting_quality(input.preview_postfx.flux)));
        let surface=self.flux_surface.as_mut().unwrap();
        let Some(update)=surface.prepare_lighting(device,cb,self.frame_index,self.frame_serial,signature,speed,resolution,&views) else {return Ok(());};
        let resources=surface.resources(self.frame_index).unwrap();
        let mut params=gpu_gi::ScreenGiParams::zeroed();
        params.gdf_min=(bounds[0]-Vec3::splat(0.1)).extend(0.0).to_array();
        params.gdf_max=(bounds[1]+Vec3::splat(0.1)).extend(0.0).to_array();
        params.sky=[sky[0],sky[1],sky[2],1.0];params.counts[1]=fs.bounces;
        params.march=[0.02,trace_distance,0.0,fs.firefly_clamp];
        params.transport[0]=fs.emissive_boost.max(0.0);
        params.misc[0]=self.frame_serial as u32;
        let packed=Self::packed_surface_lights(input,fs.mixed_lighting,&self.packed_area_lights);
        self.gpu_zone_begin(cb,gpu_zone::GI_SURFACE);
        let allocator=self.allocator();
        let result=self.reflection_surface.record(self.gpu_gi.as_ref().unwrap(),device,&mut allocator.lock().unwrap(),cb,
            self.frame_index,scene,rt,&packed,self.area_atlas.view,resources,update,params);
        self.gpu_zone_end(cb,gpu_zone::GI_SURFACE);
        result
    }
}
