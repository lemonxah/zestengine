//! Eye-local opaque copies and the water/transparent pass boundary. Lighting
//! remains in the ordinary scene draw; this module only supplies its inputs.
use anyhow::Result;
use ash::vk;
use gpu_allocator::vulkan::{Allocation, Allocator};
use crate::{FrameInput, Renderer, create_image_layers, create_view_layers, depth_aspect,
    depth_attach_layout, depth_copy_format, image_barrier_layers, post, stencil_attachment};

pub(crate) struct XrWaterCopy {
    color: vk::Image,
    depth: vk::Image,
    color_view: vk::ImageView,
    depth_view: vk::ImageView,
    allocations: Vec<Allocation>,
    extent: vk::Extent2D,
    layers: u32,
    fresh: bool,
    #[cfg(test)]
    timestamps: vk::QueryPool,
}

impl XrWaterCopy {
    pub(crate) fn new(device: &ash::Device, alloc: &mut Allocator, extent: vk::Extent2D, layers: u32) -> Result<Self> {
        let usage = vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED;
        let (color, color_alloc) = create_image_layers(device, alloc, post::HDR_FORMAT, usage, extent, layers, "XR water colour")?;
        let (depth, depth_alloc) = create_image_layers(device, alloc, depth_copy_format(), usage, extent, layers, "XR water depth")?;
        let color_view = create_view_layers(device, color, post::HDR_FORMAT, vk::ImageAspectFlags::COLOR, 0, layers)?;
        let depth_view = create_view_layers(device, depth, depth_copy_format(), vk::ImageAspectFlags::DEPTH, 0, layers)?;
        #[cfg(test)]
        let timestamps = unsafe { device.create_query_pool(&vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP).query_count(crate::XR_SLOTS as u32 * 4), None)? };
        Ok(Self { color, depth, color_view, depth_view, allocations: vec![color_alloc, depth_alloc], extent, layers, fresh: true,
            #[cfg(test)] timestamps })
    }

    pub(crate) fn destroy(&mut self, device: &ash::Device, alloc: &mut Allocator) {
        unsafe {
            #[cfg(test)]
            device.destroy_query_pool(self.timestamps, None);
            device.destroy_image_view(self.color_view, None);
            device.destroy_image_view(self.depth_view, None);
            device.destroy_image(self.color, None);
            device.destroy_image(self.depth, None);
        }
        for a in self.allocations.drain(..) { let _ = alloc.free(a); }
    }

    fn prepare(&mut self, device: &ash::Device, cb: vk::CommandBuffer) {
        if !self.fresh { return; }
        for (image, aspect) in [(self.color, vk::ImageAspectFlags::COLOR), (self.depth, depth_aspect())] {
            image_barrier_layers(device, cb, image, aspect, 0, self.layers,
                vk::ImageLayout::UNDEFINED, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::TOP_OF_PIPE, vk::AccessFlags2::empty(),
                vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_SAMPLED_READ);
        }
        self.fresh = false;
    }

    fn record(&self, device: &ash::Device, cb: vk::CommandBuffer, color: vk::Image, depth: vk::Image) {
        let tests = vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
        for (source, dest, aspect, copy_aspect, layout, stages, access) in [
            (color, self.color, vk::ImageAspectFlags::COLOR, vk::ImageAspectFlags::COLOR,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE),
            (depth, self.depth, depth_aspect(), vk::ImageAspectFlags::DEPTH,
                depth_attach_layout(), tests | vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE),
        ] {
            image_barrier_layers(device, cb, source, aspect, 0, self.layers, layout, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                stages, access, vk::PipelineStageFlags2::COPY, vk::AccessFlags2::TRANSFER_READ);
            // Previous eye submissions can still be reading this shared image;
            // queue order plus this dependency serializes overwrite after them.
            image_barrier_layers(device, cb, dest, aspect, 0, self.layers,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_SAMPLED_READ,
                vk::PipelineStageFlags2::COPY, vk::AccessFlags2::TRANSFER_WRITE);
            let sub = vk::ImageSubresourceLayers::default().aspect_mask(copy_aspect).layer_count(self.layers);
            let region = vk::ImageCopy::default().src_subresource(sub).dst_subresource(sub)
                .extent(vk::Extent3D { width: self.extent.width, height: self.extent.height, depth: 1 });
            unsafe { device.cmd_copy_image(cb, source, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, dest, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region]); }
            image_barrier_layers(device, cb, dest, aspect, 0, self.layers,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::COPY, vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_SAMPLED_READ);
            image_barrier_layers(device, cb, source, aspect, 0, self.layers, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, layout,
                vk::PipelineStageFlags2::COPY, vk::AccessFlags2::TRANSFER_READ, stages,
                access | if copy_aspect == vk::ImageAspectFlags::COLOR { vk::AccessFlags2::COLOR_ATTACHMENT_READ } else { vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ });
        }
    }
}

pub(crate) struct XrWaterPass {
    pub extent: vk::Extent2D,
    pub layers: u32,
    pub color: vk::Image,
    pub depth: vk::Image,
    /// Attachments for the resumed pass, including MSAA resolve destinations.
    pub color_att: vk::RenderingAttachmentInfo<'static>,
    pub depth_att: vk::RenderingAttachmentInfo<'static>,
}

impl Renderer {
    pub(crate) fn xr_has_refraction(&self, input: &FrameInput) -> bool {
        input.draws.iter().any(|d| self.materials.get(d.material.0)
            .is_some_and(|m| !m.error && (m.features.water || m.features.grab)))
    }

    pub(crate) fn prepare_xr_water_copy(&mut self, cb: vk::CommandBuffer, _slot: usize) {
        if let Some(copy) = self.xr_render.as_mut().and_then(|xr| xr.water_copy.as_mut()) {
            copy.prepare(&self.ctx.device, cb);
            #[cfg(test)]
            if self.gpu_ts_period > 0.0 { unsafe { self.ctx.device.cmd_reset_query_pool(cb, copy.timestamps, _slot as u32 * 4, 4); } }
        }
    }

    // Test-only GPU timestamps. Measurements exclude CPU submission and readback.
    pub(crate) fn xr_water_timestamp(&self, _cb: vk::CommandBuffer, _slot: usize, _query: u32) {
        #[cfg(test)]
        if self.gpu_ts_period > 0.0 {
            if let Some(copy) = self.xr_render.as_ref().and_then(|xr| xr.water_copy.as_ref()) {
                unsafe { self.ctx.device.cmd_write_timestamp2(_cb, vk::PipelineStageFlags2::ALL_COMMANDS, copy.timestamps, _slot as u32 * 4 + _query); }
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn xr_water_timing(&self) -> Option<(f64, f64)> {
        let xr = self.xr_render.as_ref()?;
        let copy = xr.water_copy.as_ref()?;
        if self.gpu_ts_period <= 0.0 { return None; }
        let slot = (xr.next + crate::XR_SLOTS - 1) % crate::XR_SLOTS;
        let mut ticks = [[0u64; 2]; 4];
        let result = unsafe { self.ctx.device.get_query_pool_results(copy.timestamps, slot as u32 * 4, &mut ticks,
            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY) };
        if result.is_err_and(|error| error != vk::Result::NOT_READY) || ticks[2][1] == 0 || ticks[3][1] == 0 { return None; }
        let ms = |a: u64, b: u64| (b.wrapping_sub(a) & self.gpu_ts_mask) as f64 * self.gpu_ts_period as f64 / 1e6;
        let copy = if ticks[0][1] != 0 && ticks[1][1] != 0 { ms(ticks[0][0], ticks[1][0]) } else { 0.0 };
        Some((copy, ms(ticks[2][0], ticks[3][0])))
    }

    pub(crate) fn bind_xr_water_copy(&self, set: vk::DescriptorSet) {
        let xr = self.xr_render.as_ref().unwrap();
        let (depth, color) = if let Some(copy) = &xr.water_copy { (copy.depth_view, copy.color_view) }
            else if xr.layers > 1 { (self.lightmaps.view, self.lightmaps.view) }
            else { return; }; // No copy reads; retain the ordinary 2D fallbacks.
        let image = |view, sampler| [vk::DescriptorImageInfo::default().image_view(view).sampler(sampler)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let depth = image(depth, self.lightmap_sampler);
        let color = image(color, self.grab_sampler);
        let write = |binding| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(binding)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER);
        unsafe { self.ctx.device.update_descriptor_sets(&[write(5).image_info(&depth), write(6).image_info(&color)], &[]); }
    }

    /// Called inside the scene rendering scope; leaves the resumed scope open.
    pub(crate) fn record_xr_water_draws(&mut self, cb: vk::CommandBuffer, order: &[usize], input: &FrameInput,
        set: vk::DescriptorSet, pass: &XrWaterPass) -> Result<()> {
        let device = self.ctx.device.clone();
        let slot = (self.xr_render.as_ref().unwrap().next + crate::XR_SLOTS - 1) % crate::XR_SLOTS;
        let split = self.water_split(order, input);
        let first = split.unwrap_or(order.len());
        self.record_scene_draws(&device, cb, &order[..first], input, set, false)?;
        if let Some(unused) = self.gpu_take_split() {
            if unused.xr.is_some() { self.gpu_cull_late_xr(&device, cb, &unused); }
        }
        if let Some(split) = split {
            unsafe { device.cmd_end_rendering(cb); }
            self.xr_water_timestamp(cb, slot, 0);
            self.xr_render.as_ref().unwrap().water_copy.as_ref().unwrap().record(&device, cb, pass.color, pass.depth);
            self.xr_water_timestamp(cb, slot, 1);
            let color = [pass.color_att.load_op(vk::AttachmentLoadOp::LOAD)];
            let depth = pass.depth_att.load_op(vk::AttachmentLoadOp::LOAD);
            let stencil = stencil_attachment(&depth);
            let area = vk::Rect2D { offset: vk::Offset2D::default(), extent: pass.extent };
            unsafe {
                // The copies transition resolved images. MSAA attachments are
                // separate allocations: order their stores before LOAD too.
                let stages = vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
                let writes = vk::AccessFlags2::COLOR_ATTACHMENT_WRITE | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;
                let barrier = [vk::MemoryBarrier2::default().src_stage_mask(stages).src_access_mask(writes)
                    .dst_stage_mask(stages).dst_access_mask(writes | vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ)];
                device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&barrier));
                device.cmd_begin_rendering(cb, &vk::RenderingInfo::default().render_area(area).layer_count(1)
                    .view_mask(if pass.layers == 2 { 3 } else { 0 }).color_attachments(&color).depth_attachment(&depth).stencil_attachment(&stencil));
                device.cmd_set_viewport(cb, 0, &[vk::Viewport { x: 0.0, y: pass.extent.height as f32,
                    width: pass.extent.width as f32, height: -(pass.extent.height as f32), min_depth: 0.0, max_depth: 1.0 }]);
                device.cmd_set_scissor(cb, 0, &[area]);
            }
            self.record_scene_draws(&device, cb, &order[split..], input, set, false)?;
        }
        Ok(())
    }
}
