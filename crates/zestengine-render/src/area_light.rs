//! Area lights (rectangle / disc, LTC shading, optional emission texture): the
//! renderer side. The scene hands over [`AreaLightInstance`]s with
//! [`Renderer::set_area_lights`]; each frame the nearest [`MAX_AREA_LIGHTS`] are
//! packed into the frame's light array (two entries each, see `area_light.glsl`)
//! and the textured ones are copied into the emission atlas.

use super::*;

impl Renderer {
    /// The area lights of this frame's scene (replaces the previous list). The
    /// nearest [`MAX_AREA_LIGHTS`] to the camera are drawn; call it every frame the
    /// lights should show (the list persists until the next call).
    pub fn set_area_lights(&mut self, lights: &[AreaLightInstance]) {
        self.area_lights.clear();
        self.area_lights.extend_from_slice(lights);
    }

    /// Point a material set's atlas binding at the shared emission atlas.
    pub(crate) fn write_area_atlas(&self, set: vk::DescriptorSet) {
        let info = [vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(self.area_atlas.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(AREA_ATLAS_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&info);
        let ltc_info = [vk::DescriptorImageInfo::default()
            .sampler(self.ltc_lut.sampler)
            .image_view(self.ltc_lut.tex.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let ltc_write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(LTC_LUT_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&ltc_info);
        unsafe { self.ctx.device.update_descriptor_sets(&[write, ltc_write], &[]) };
    }

    /// Append this frame's area lights to the packed light array (two entries
    /// each) and queue the atlas copies for the textured ones.
    pub(crate) fn pack_area_lights(&mut self, eye: Vec3, lights: &mut [GpuLight; MAX_LIGHTS], count: &mut usize) {
        self.area_sources.clear();
        self.packed_area_lights.clear();
        if self.area_lights.is_empty() {
            return;
        }
        // Nearest first: distance to the emitter's centre less its size, inside the range.
        let mut order: Vec<(f32, usize)> = self
            .area_lights
            .iter()
            .enumerate()
            .filter(|(_, l)| l.intensity > 0.0 && l.range > 0.0)
            .map(|(i, l)| ((l.position.distance(eye) - l.half_size[0].max(l.half_size[1])).max(0.0), i))
            .collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut tile = 0u32;
        for &(_, i) in order.iter().take(MAX_AREA_LIGHTS) {
            if *count + 2 > MAX_LIGHTS {
                break;
            }
            let l = self.area_lights[i];
            let mut slot = 0.0;
            if let Some(h) = l.texture
                && tile < MAX_AREA_LIGHTS as u32
                && let Some(t) = self.textures.get(h.0)
            {
                self.area_sources.push((tile, t.view));
                slot = tile as f32 + 1.0;
                tile += 1;
            }
            let kind = if l.shape == AreaShape::Disc { 5.0 } else { 4.0 };
            let (r, u) = (l.right.normalize_or_zero() * l.half_size[0], l.up.normalize_or_zero() * l.half_size[1]);
            lights[*count] = GpuLight {
                pos_kind: [l.position.x, l.position.y, l.position.z, kind],
                dir_range: [r.x, r.y, r.z, l.range],
                color: [l.color[0] * l.intensity, l.color[1] * l.intensity, l.color[2] * l.intensity, if l.two_sided { 1.0 } else { 0.0 }],
                spot: [slot, -1.0, 0.0, 0.0],
            };
            lights[*count + 1] = GpuLight {
                pos_kind: [l.position.x, l.position.y, l.position.z, 6.0],
                dir_range: [u.x, u.y, u.z, l.range],
                color: [0.0; 4],
                spot: [0.0, -1.0, 0.0, 0.0],
            };
            self.packed_area_lights.extend_from_slice(&lights[*count..*count+2]);
            *count += 2;
        }
    }

    /// Record the atlas upkeep: the first-use layout change, then this frame's
    /// texture copies and their mip chains.
    pub(crate) fn record_area_atlas(&mut self, device: &ash::Device, cb: vk::CommandBuffer) {
        let color = vk::ImageAspectFlags::COLOR;
        let readers = vk::PipelineStageFlags2::VERTEX_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER;
        let blit_stage = vk::PipelineStageFlags2::BLIT | vk::PipelineStageFlags2::COPY;
        let compute=vk::PipelineStageFlags2::COMPUTE_SHADER;
        let atlas = self.area_atlas.image;
        if self.area_atlas.fresh {
            self.area_atlas.fresh = false;
            image_barrier_mips(device,cb,atlas,color,0,AREA_MIPS,
                vk::ImageLayout::UNDEFINED,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::TOP_OF_PIPE,vk::AccessFlags2::empty(),readers,vk::AccessFlags2::SHADER_SAMPLED_READ);
        }
        if self.area_sources.is_empty() { return; }
        // Uploaded, compressed and live render textures all arrive shader-readable.
        // Synchronize their writes without changing source layouts or requiring
        // format-dependent blit support. Sets belong to the retired frame slot.
        unsafe {
            let memory=[vk::MemoryBarrier2::default().src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .src_access_mask(vk::AccessFlags2::MEMORY_WRITE).dst_stage_mask(compute)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)];
            device.cmd_pipeline_barrier2(cb,&vk::DependencyInfo::default().memory_barriers(&memory));
        }
        image_barrier_mips(device,cb,atlas,color,0,1,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,vk::ImageLayout::GENERAL,
            readers,vk::AccessFlags2::SHADER_SAMPLED_READ,compute,vk::AccessFlags2::SHADER_STORAGE_WRITE);
        for &(tile,source) in &self.area_sources {
            let set=self.area_atlas.copy_sets[self.frame_index*MAX_AREA_LIGHTS+tile as usize];
            let src=[vk::DescriptorImageInfo::default().image_view(source).sampler(self.sampler)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let dst=[vk::DescriptorImageInfo::default().image_view(self.area_atlas.base_view).image_layout(vk::ImageLayout::GENERAL)];
            let writes=[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&src),
                vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&dst)];
            unsafe {
                // Descriptor ranges cover the whole atlas; explicitly order
                // dispatch writes even though tiles have disjoint texels.
                if tile>0 {
                    let memory=[vk::MemoryBarrier2::default().src_stage_mask(compute).dst_stage_mask(compute)
                        .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE).dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)];
                    device.cmd_pipeline_barrier2(cb,&vk::DependencyInfo::default().memory_barriers(&memory));
                }
                device.update_descriptor_sets(&writes,&[]);
                device.cmd_bind_pipeline(cb,vk::PipelineBindPoint::COMPUTE,self.area_atlas.copy_pipeline);
                device.cmd_bind_descriptor_sets(cb,vk::PipelineBindPoint::COMPUTE,self.area_atlas.copy_pipeline_layout,0,&[set],&[]);
                device.cmd_push_constants(cb,self.area_atlas.copy_pipeline_layout,vk::ShaderStageFlags::COMPUTE,0,
                    bytemuck::cast_slice(&[(tile%AREA_COLS)*AREA_TILE,(tile/AREA_COLS)*AREA_TILE]));
                device.cmd_dispatch(cb,AREA_TILE/8,AREA_TILE/8,1);
            }
        }
        image_barrier_mips(device,cb,atlas,color,0,1,vk::ImageLayout::GENERAL,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            compute,vk::AccessFlags2::SHADER_STORAGE_WRITE,blit_stage,vk::AccessFlags2::TRANSFER_READ);
        image_barrier_mips(device,cb,atlas,color,1,AREA_MIPS-1,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            readers,vk::AccessFlags2::SHADER_SAMPLED_READ,blit_stage,vk::AccessFlags2::TRANSFER_WRITE);
        let layers = |level: u32| vk::ImageSubresourceLayers::default().aspect_mask(color).mip_level(level).layer_count(1);
        let tile_origin = |tile: u32, level: u32| -> (i32, i32, i32) {
            let side = (AREA_TILE >> level).max(1) as i32;
            ((tile % AREA_COLS) as i32 * side, (tile / AREA_COLS) as i32 * side, side)
        };
        // Each level from the one above, tile by tile.
        for level in 1..AREA_MIPS {
            if level>1 { image_barrier_mips(
                device, cb, atlas, color, level - 1, 1,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                blit_stage, vk::AccessFlags2::TRANSFER_WRITE, blit_stage, vk::AccessFlags2::TRANSFER_READ,
            );
            }
            for &(tile, _) in &self.area_sources {
                let (x0, y0, s0) = tile_origin(tile, level - 1);
                let (x1, y1, s1) = tile_origin(tile, level);
                let blit = vk::ImageBlit::default()
                    .src_subresource(layers(level - 1))
                    .src_offsets([vk::Offset3D { x: x0, y: y0, z: 0 }, vk::Offset3D { x: x0 + s0, y: y0 + s0, z: 1 }])
                    .dst_subresource(layers(level))
                    .dst_offsets([vk::Offset3D { x: x1, y: y1, z: 0 }, vk::Offset3D { x: x1 + s1, y: y1 + s1, z: 1 }]);
                unsafe {
                    device.cmd_blit_image(cb, atlas, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, atlas, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[blit], vk::Filter::LINEAR);
                }
            }
        }
        // Back to shader-read: the upper levels were moved to SRC one by one, the last is still DST.
        image_barrier_mips(
            device, cb, atlas, color, 0, AREA_MIPS - 1,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            blit_stage, vk::AccessFlags2::TRANSFER_READ, readers, vk::AccessFlags2::SHADER_SAMPLED_READ,
        );
        image_barrier_mips(
            device, cb, atlas, color, AREA_MIPS - 1, 1,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            blit_stage, vk::AccessFlags2::TRANSFER_WRITE, readers, vk::AccessFlags2::SHADER_SAMPLED_READ,
        );
    }
}
