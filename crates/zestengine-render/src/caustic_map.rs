use crate::pipeline_store::PipelineDeviceExt as _;
use crate::sync::{Access, Range, Stages};
use crate::{CAUSTIC_MAP_REQUESTED, FRAMES_IN_FLIGHT, GpuWater, LightInstance, LightKind, MAX_LIGHTS, sync_check};
use anyhow::Result;
use ash::vk;
use glam::Mat4;
use gpu_allocator::vulkan::{Allocation, Allocator};

pub(crate) const MAP_SIZE: u32 = 1024;
pub(crate) const MAP_LAYERS: u32 = 8;
pub(crate) const FLOOR_LAYERS: u32 = 4;
pub(crate) const MAP_MIPS: u32 = 6;
const LIGHTS_PER_BODY: usize = 4;
pub(crate) const LEARN_SETS: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CausticJob {
    pub body: u32,
    pub light: u32,
    pub layer: u32,
    pub floor: u32,
}

fn relevance(to_local: &Mat4, body: &GpuWater, light: &LightInstance) -> Option<f32> {
    let strength = light.intensity * light.color.iter().copied().fold(0.0f32, f32::max);
    if strength <= 0.0 {
        return None;
    }
    match light.kind {
        LightKind::Directional => (to_local.transform_vector3(light.direction).y < -0.01).then_some(strength * 1.0e4),
        LightKind::Point | LightKind::Spot => {
            let p = to_local.transform_point3(light.position);
            if p.y <= body.bounds[2] + 0.01 {
                return None;
            }
            let dx = (p.x.abs() - body.bounds[0]).max(0.0);
            let dz = (p.z.abs() - body.bounds[1]).max(0.0);
            let dy = p.y - body.bounds[2];
            let distance = (dx * dx + dz * dz + dy * dy).sqrt();
            if light.range > 0.0 && distance > light.range {
                return None;
            }
            Some(strength / (1.0 + distance * distance))
        }
    }
}

pub(crate) fn plan(bodies: &mut [GpuWater], lights: &[LightInstance]) -> Vec<CausticJob> {
    let mut jobs = Vec::new();
    let mut floors = 0u32;
    for (slot, body) in bodies.iter_mut().enumerate() {
        let requested = body.caustic_map[0] & CAUSTIC_MAP_REQUESTED != 0;
        body.caustic_map = [0; 4];
        if !requested || body.bounds[0] <= 0.0 || body.bounds[1] <= 0.0 || body.bounds[3] <= 0.0 || floors >= FLOOR_LAYERS {
            continue;
        }
        let to_local = Mat4::from_cols_array_2d(&body.to_local);
        let mut scored: Vec<(f32, usize)> = lights
            .iter()
            .take(MAX_LIGHTS.min(256))
            .enumerate()
            .filter_map(|(index, light)| relevance(&to_local, body, light).map(|score| (score, index)))
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        let (mut light_bits, mut layer_bits, mut count) = (0u32, 0u32, 0u32);
        for &(_, index) in scored.iter().take(LIGHTS_PER_BODY) {
            if jobs.len() as u32 >= MAP_LAYERS {
                break;
            }
            let layer = jobs.len() as u32;
            light_bits |= (index as u32) << (8 * count);
            layer_bits |= layer << (8 * count);
            jobs.push(CausticJob { body: slot as u32, light: index as u32, layer, floor: floors });
            count += 1;
        }
        if count == 0 {
            continue;
        }
        body.caustic_map = [count | (floors << 8) | CAUSTIC_MAP_REQUESTED, light_bits, layer_bits, MAP_SIZE];
        floors += 1;
    }
    jobs
}

struct Target {
    image: vk::Image,
    sampled: vk::ImageView,
    storage: vk::ImageView,
    allocation: Option<Allocation>,
    mips: u32,
}

impl Target {
    #[allow(clippy::too_many_arguments)]
    fn new(device: &ash::Device, allocator: &mut Allocator, format: vk::Format, size: u32, layers: u32, mips: u32, usage: vk::ImageUsageFlags, name: &'static str) -> Result<Self> {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D { width: size, height: size, depth: 1 })
            .mip_levels(mips)
            .array_layers(layers)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.create_image(&info, None)? };
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let allocation = match allocator.allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name,
            requirements,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        }) {
            Ok(allocation) => allocation,
            Err(e) => {
                unsafe { device.destroy_image(image, None) };
                return Err(e.into());
            }
        };
        let view = |levels: u32| {
            let range = vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(levels).layer_count(layers);
            unsafe { device.create_image_view(&vk::ImageViewCreateInfo::default().image(image).view_type(vk::ImageViewType::TYPE_2D_ARRAY).format(format).subresource_range(range), None) }
        };
        let views = unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset()) }.and_then(|()| Ok((view(mips)?, view(1)?)));
        let (sampled, storage) = match views {
            Ok(views) => views,
            Err(e) => {
                unsafe { device.destroy_image(image, None) };
                let _ = allocator.free(allocation);
                return Err(e.into());
            }
        };
        sync_check::register(image, sync_check::shared(name, vk::ImageAspectFlags::COLOR, mips, layers));
        Ok(Self { image, sampled, storage, allocation: Some(allocation), mips })
    }

    fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_image_view(self.sampled, None);
            device.destroy_image_view(self.storage, None);
            device.destroy_image(self.image, None);
        }
        sync_check::forget(self.image);
        if let Some(allocation) = self.allocation.take() {
            let _ = allocator.free(allocation);
        }
    }
}

struct Compute {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl Compute {
    fn new(device: &ash::Device, frame_layout: vk::DescriptorSetLayout, bindings: &[vk::DescriptorType], push_size: u32, code: &[u8]) -> Result<Self> {
        let entries: Vec<vk::DescriptorSetLayoutBinding<'_>> = bindings
            .iter()
            .enumerate()
            .map(|(index, kind)| {
                vk::DescriptorSetLayoutBinding::default().binding(index as u32).descriptor_type(*kind).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        unsafe {
            let set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&entries), None)?;
            let push = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(push_size)];
            let layout = device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[frame_layout, set_layout]).push_constant_ranges(&push), None)?;
            let module = crate::screen_pass::shader_module(device, code)?;
            let built = device.create_cached_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .layout(layout)
                    .stage(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main"))],
                None,
            );
            device.destroy_shader_module(module, None);
            let pipeline = built.map_err(|(_, e)| e)?[0];
            Ok(Self { set_layout, layout, pipeline })
        }
    }

    fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

pub(crate) struct CausticMaps {
    sampler: vk::Sampler,
    nearest: vk::Sampler,
    placeholder: Target,
    placeholder_ready: bool,
    maps: Option<Target>,
    floor: Option<Target>,
    map_pass: Compute,
    learn_pass: Compute,
    pool: vk::DescriptorPool,
    map_sets: Vec<vk::DescriptorSet>,
    learn_sets: Vec<vk::DescriptorSet>,
    learn_depth: [vk::ImageView; LEARN_SETS],
    jobs: Vec<CausticJob>,
    generation: u64,
    recorded: Option<u64>,
    desktop_serial: Option<u64>,
    xr_cycle_serial: Option<Option<u64>>,
}

impl CausticMaps {
    pub fn new(device: &ash::Device, allocator: &mut Allocator, frame_layout: vk::DescriptorSetLayout) -> Result<Self> {
        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .min_filter(vk::Filter::LINEAR)
                    .mag_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?
        };
        let nearest = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .min_filter(vk::Filter::NEAREST)
                    .mag_filter(vk::Filter::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?
        };
        let placeholder = Target::new(
            device,
            allocator,
            vk::Format::R16G16_SFLOAT,
            1,
            2,
            1,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            "caustic map placeholder",
        )?;
        let map_pass = Compute::new(
            device,
            frame_layout,
            &[vk::DescriptorType::STORAGE_IMAGE, vk::DescriptorType::STORAGE_IMAGE],
            16,
            include_bytes!(concat!(env!("OUT_DIR"), "/caustic_map.comp.spv")),
        )?;
        let learn_pass = Compute::new(
            device,
            frame_layout,
            &[vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::DescriptorType::STORAGE_IMAGE],
            80,
            include_bytes!(concat!(env!("OUT_DIR"), "/caustic_learn.comp.spv")),
        )?;
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets((FRAMES_IN_FLIGHT + LEARN_SETS) as u32).pool_sizes(&[
                    vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: (2 * FRAMES_IN_FLIGHT + LEARN_SETS) as u32 },
                    vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: LEARN_SETS as u32 },
                ]),
                None,
            )?
        };
        let map_sets = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[map_pass.set_layout; FRAMES_IN_FLIGHT]))? };
        let learn_sets = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[learn_pass.set_layout; LEARN_SETS]))? };
        Ok(Self {
            sampler,
            nearest,
            placeholder,
            placeholder_ready: false,
            maps: None,
            floor: None,
            map_pass,
            learn_pass,
            pool,
            map_sets,
            learn_sets,
            learn_depth: [vk::ImageView::null(); LEARN_SETS],
            jobs: Vec::new(),
            generation: 0,
            recorded: None,
            desktop_serial: None,
            xr_cycle_serial: None,
        })
    }

    pub fn binding(&self) -> vk::DescriptorImageInfo {
        let view = self.maps.as_ref().map_or(self.placeholder.sampled, |maps| maps.sampled);
        vk::DescriptorImageInfo::default().sampler(self.sampler).image_view(view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
    }

    pub fn placeholder_binding(&self) -> vk::DescriptorImageInfo {
        vk::DescriptorImageInfo::default().sampler(self.sampler).image_view(self.placeholder.sampled).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
    }

    pub fn active(&self) -> bool {
        self.maps.is_some() && !self.jobs.is_empty()
    }

    pub fn prepare_desktop(&mut self, device: &ash::Device, allocator: &mut Allocator, jobs: Vec<CausticJob>, serial: u64) -> Result<()> {
        self.desktop_serial = Some(serial);
        self.generation += 1;
        self.prepare(device, allocator, jobs)
    }

    pub fn prepare_xr(&mut self, device: &ash::Device, allocator: &mut Allocator, jobs: Vec<CausticJob>, cycle_start: bool) -> Result<()> {
        if !cycle_start {
            return Ok(());
        }
        let desktop_running = self.xr_cycle_serial.is_some_and(|seen| seen != self.desktop_serial);
        self.xr_cycle_serial = Some(self.desktop_serial);
        if desktop_running {
            return Ok(());
        }
        self.generation += 1;
        self.prepare(device, allocator, jobs)
    }

    fn prepare(&mut self, device: &ash::Device, allocator: &mut Allocator, jobs: Vec<CausticJob>) -> Result<()> {
        self.jobs = jobs;
        if self.jobs.is_empty() || self.maps.is_some() {
            return Ok(());
        }
        let maps = Target::new(
            device,
            allocator,
            vk::Format::R16G16_SFLOAT,
            MAP_SIZE,
            MAP_LAYERS,
            MAP_MIPS,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
            "caustic maps",
        )?;
        let floor = match Target::new(
            device,
            allocator,
            vk::Format::R32_UINT,
            MAP_SIZE,
            FLOOR_LAYERS,
            1,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST,
            "caustic floor",
        ) {
            Ok(floor) => floor,
            Err(e) => {
                let mut maps = maps;
                maps.destroy(device, allocator);
                return Err(e);
            }
        };
        for &set in &self.map_sets {
            let floor_info = [vk::DescriptorImageInfo::default().image_view(floor.storage).image_layout(vk::ImageLayout::GENERAL)];
            let map_info = [vk::DescriptorImageInfo::default().image_view(maps.storage).image_layout(vk::ImageLayout::GENERAL)];
            let writes = [
                vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&floor_info),
                vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&map_info),
            ];
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
        for &set in &self.learn_sets {
            let floor_info = [vk::DescriptorImageInfo::default().image_view(floor.storage).image_layout(vk::ImageLayout::GENERAL)];
            unsafe {
                device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&floor_info)],
                    &[],
                )
            };
        }
        self.learn_depth = [vk::ImageView::null(); LEARN_SETS];
        self.maps = Some(maps);
        self.floor = Some(floor);
        self.recorded = None;
        Ok(())
    }

    pub fn record_placeholder(&mut self, device: &ash::Device, cb: vk::CommandBuffer) {
        if self.placeholder_ready {
            return;
        }
        let image = self.placeholder.image;
        sync_check::pass(device, cb, "caustic map placeholder clear", &[(image, Access::TransferDst)]);
        let range = vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(2);
        unsafe { device.cmd_clear_color_image(cb, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &vk::ClearColorValue { float32: [1.0, 0.0, 0.0, 0.0] }, &[range]) };
        sync_check::pass(device, cb, "caustic map placeholder sampled", &[(image, Access::SampledFragment)]);
        self.placeholder_ready = true;
    }

    pub fn record_maps(&mut self, device: &ash::Device, cb: vk::CommandBuffer, frame_set: vk::DescriptorSet) {
        if !self.active() || self.recorded == Some(self.generation) {
            return;
        }
        let (Some(maps), Some(floor)) = (self.maps.as_ref(), self.floor.as_ref()) else { return };
        self.recorded = Some(self.generation);
        let slot = 0;
        let base = Range::Mips { base: 0, count: 1 };
        sync_check::pass_ranges(device, cb, "caustic maps", [(floor.image, Range::All, Access::StorageRead(Stages::COMPUTE)), (maps.image, base, Access::StorageWrite(Stages::COMPUTE))]);
        unsafe {
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.map_pass.pipeline);
            device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, self.map_pass.layout, 0, &[frame_set, self.map_sets[slot]], &[]);
            for job in &self.jobs {
                let push = [job.body as i32, job.light as i32, job.layer as i32, job.floor as i32];
                device.cmd_push_constants(cb, self.map_pass.layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::cast_slice(&push));
                device.cmd_dispatch(cb, MAP_SIZE.div_ceil(8), MAP_SIZE.div_ceil(8), 1);
            }
        }
        let full = |level: u32| vk::Offset3D { x: (MAP_SIZE >> level).max(1) as i32, y: (MAP_SIZE >> level).max(1) as i32, z: 1 };
        for level in 1..maps.mips {
            sync_check::pass_ranges(device, cb, "caustic map mips", [
                (maps.image, Range::Mips { base: level - 1, count: 1 }, Access::TransferSrc),
                (maps.image, Range::Mips { base: level, count: 1 }, Access::TransferDst),
            ]);
            let layers = |mip: u32| vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).mip_level(mip).layer_count(MAP_LAYERS);
            let blit = vk::ImageBlit::default()
                .src_subresource(layers(level - 1))
                .src_offsets([vk::Offset3D::default(), full(level - 1)])
                .dst_subresource(layers(level))
                .dst_offsets([vk::Offset3D::default(), full(level)]);
            unsafe { device.cmd_blit_image(cb, maps.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, maps.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[blit], vk::Filter::LINEAR) };
        }
        sync_check::pass(device, cb, "caustic maps sampled", &[(maps.image, Access::SampledFragment)]);
        sync_check::pass(device, cb, "caustic floor clear", &[(floor.image, Access::TransferDst)]);
        let range = vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(FLOOR_LAYERS);
        unsafe { device.cmd_clear_color_image(cb, floor.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &vk::ClearColorValue { uint32: [0; 4] }, &[range]) };
        sync_check::pass(device, cb, "caustic floor learning", &[(floor.image, Access::StorageReadWrite(Stages::COMPUTE))]);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_learn(&mut self, device: &ash::Device, cb: vk::CommandBuffer, frame_set: vk::DescriptorSet, slot: usize, depth: vk::ImageView, extent: vk::Extent2D, inv_view_proj: Option<Mat4>) {
        if !self.active() || self.recorded.is_none() || depth == vk::ImageView::null() || slot >= LEARN_SETS {
            return;
        }
        if self.learn_depth[slot] != depth {
            let info = [vk::DescriptorImageInfo::default().sampler(self.nearest).image_view(depth).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            unsafe {
                device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default().dst_set(self.learn_sets[slot]).dst_binding(0).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&info)],
                    &[],
                )
            };
            self.learn_depth[slot] = depth;
        }
        let before = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::ALL_TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)];
        let after = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::ALL_TRANSFER)];
        unsafe {
            sync_check::barrier2(device, cb, &vk::DependencyInfo::default().memory_barriers(&before));
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.learn_pass.pipeline);
            device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, self.learn_pass.layout, 0, &[frame_set, self.learn_sets[slot]], &[]);
            let mut push = [0u32; 20];
            push[..16].copy_from_slice(bytemuck::cast_slice(&inv_view_proj.unwrap_or(Mat4::IDENTITY).to_cols_array()));
            push[16] = extent.width;
            push[17] = extent.height;
            push[18] = u32::from(inv_view_proj.is_some());
            let stride = learn_stride(extent);
            push[19] = stride;
            device.cmd_push_constants(cb, self.learn_pass.layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::cast_slice(&push));
            device.cmd_dispatch(cb, extent.width.div_ceil(stride).div_ceil(8), extent.height.div_ceil(stride).div_ceil(8), 1);
            sync_check::barrier2(device, cb, &vk::DependencyInfo::default().memory_barriers(&after));
        }
    }

    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        if let Some(mut maps) = self.maps.take() {
            maps.destroy(device, allocator);
        }
        if let Some(mut floor) = self.floor.take() {
            floor.destroy(device, allocator);
        }
        self.placeholder.destroy(device, allocator);
        self.map_pass.destroy(device);
        self.learn_pass.destroy(device);
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_sampler(self.sampler, None);
            device.destroy_sampler(self.nearest, None);
        }
    }
}

fn learn_stride(extent: vk::Extent2D) -> u32 {
    let pixels = extent.width as f64 * extent.height as f64;
    (pixels / LEARN_SAMPLES).sqrt().ceil().max(1.0) as u32
}

const LEARN_SAMPLES: f64 = 1_048_576.0;

pub(crate) fn write_binding(device: &ash::Device, set: vk::DescriptorSet, info: vk::DescriptorImageInfo) {
    let infos = [info];
    unsafe {
        device.update_descriptor_sets(
            &[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(15).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&infos)],
            &[],
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn pool(requested: bool) -> GpuWater {
        let mut body = GpuWater { to_local: Mat4::IDENTITY.to_cols_array_2d(), bounds: [2.0, 10.0, 0.0, 1.0], ..Default::default() };
        if requested {
            body.caustic_map[0] = CAUSTIC_MAP_REQUESTED;
        }
        body
    }

    fn light(kind: LightKind, position: Vec3, range: f32) -> LightInstance {
        LightInstance {
            kind,
            position,
            direction: Vec3::new(0.2, -1.0, 0.1).normalize(),
            color: [1.0; 3],
            intensity: 5.0,
            range,
            spot_inner_deg: 30.0,
            spot_outer_deg: 45.0,
            cast_shadows: true,
            soft_shadows: true,
            radius: 0.1,
            shadow_bias: 0.002,
            shadow_clip: None,
            baked: false,
        }
    }

    #[test]
    fn only_requested_bounded_bodies_get_jobs_for_lights_above_them_in_range() {
        let lights = [
            light(LightKind::Point, Vec3::new(0.0, 2.0, 0.0), 6.0),
            light(LightKind::Point, Vec3::new(0.0, -1.0, 0.0), 6.0),
            light(LightKind::Point, Vec3::new(40.0, 2.0, 0.0), 6.0),
            light(LightKind::Directional, Vec3::ZERO, 0.0),
        ];
        let mut bodies = [pool(true), pool(false), { let mut endless = pool(true); endless.bounds[0] = 0.0; endless }];
        let jobs = plan(&mut bodies, &lights);
        assert_eq!(jobs, [CausticJob { body: 0, light: 3, layer: 0, floor: 0 }, CausticJob { body: 0, light: 0, layer: 1, floor: 0 }]);
        assert_eq!(bodies[0].caustic_map, [2 | CAUSTIC_MAP_REQUESTED, 3 | (0 << 8), 0 | (1 << 8), MAP_SIZE]);
        assert_eq!(bodies[1].caustic_map, [0; 4]);
        assert_eq!(bodies[2].caustic_map, [0; 4]);
        let again = plan(&mut bodies, &lights);
        assert_eq!(again, jobs, "planning planned bodies again gives the same jobs");
    }

    #[test]
    fn learning_samples_about_a_million_pixels_per_view() {
        let stride = |w, h| learn_stride(vk::Extent2D { width: w, height: h });
        assert_eq!(stride(989, 529), 1);
        assert_eq!(stride(2064, 2272), 3);
        assert_eq!(stride(4128, 4544), 5);
    }

    #[test]
    fn the_layers_run_out_before_the_lights_do() {
        let lights: Vec<LightInstance> = (0..6).map(|i| light(LightKind::Point, Vec3::new(i as f32 * 0.1, 1.0, 0.0), 0.0)).collect();
        let mut bodies = [pool(true), pool(true), pool(true)];
        let jobs = plan(&mut bodies, &lights);
        assert_eq!(jobs.len(), MAP_LAYERS as usize);
        assert_eq!(bodies[0].caustic_map[0] & 0xff, 4);
        assert_eq!(bodies[1].caustic_map[0] & 0xff, 4);
        assert_eq!(bodies[2].caustic_map, [0; 4]);
        assert_eq!((bodies[1].caustic_map[0] >> 8) & 0xff, 1);
    }
}
