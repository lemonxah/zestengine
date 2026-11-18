//! Content revisions for textures written without replacing their handles.
use super::*;

impl Renderer {
    /// Notify lighting caches after updating an existing texture's pixels.
    /// Call after recording the write and its shader-read barrier, before
    /// recording consumers. Camera targets call this automatically. External
    /// video/GPU producers must also notify after each completed content update.
    pub fn notify_texture_updated(&mut self, texture: TextureHandle) {
        let Some(image)=self.textures.get(texture.0) else {return;};
        let revision=self.texture_revisions.entry(texture.0).or_default();
        *revision=revision.wrapping_add(1).max(1);
        for source in &mut self.area_sources {
            if source.1==image.view {source.2 = *revision;}
        }
    }

    pub(super) fn live_gi_distance(&self,camera:Vec3)->f32 {
        if !self.gi_enabled {return 0.0;}
        if self.flux_settings.march_distance>0.0 {self.flux_settings.march_distance}
        else {self.gpu_gi.as_ref().map_or(0.0,|g|g.gdf_trace_distance(camera))}
    }

    pub(super) fn live_material_needed(&self,draw:&DrawCmd,input:&FrameInput)->bool {
        if !draw.shadow_casting.visible() || (!self.gi_enabled && self.reflection_views(input).next().is_none()) {return false;}
        let Some(material)=self.materials.get(draw.material.0) else {return false;};
        // Reflectance and mask textures change bounced lighting and occlusion
        // too. Keep all supported surface-cache inputs live off-screen.
        if material.features.water || material.custom_shader.is_some()
            || material.features.alpha_mode==AlphaMode::Blend {return false;}
        // Missing bounds disable culling, as in the regular camera pass.
        if draw.bound_radius<=0.0 {return true;}
        let center=draw.transform.transform_point3(draw.mesh_center);
        let scale=[draw.transform.x_axis,draw.transform.y_axis,draw.transform.z_axis]
            .iter().map(|v|v.truncate().length()).fold(0.0f32,f32::max);
        let radius=draw.bound_radius*scale;
        (self.gi_enabled && std::iter::once(input.camera).chain(input.camera_preview)
            .any(|c|c.position.distance(center)<=radius+self.live_gi_distance(c.position)))
            || self.reflection_views(input).any(|(c,fs)|c.position.distance(center)<=radius+fs.ssr_max_distance.max(0.0))
    }
}
