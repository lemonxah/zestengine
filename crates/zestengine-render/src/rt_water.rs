//! Moving water in the ray-traced reflections. The water meshes hold the flat
//! rest grid (standard.vert displaces it), so a compute pass
//! (`water_displace.comp`) writes each water mesh's displaced positions, the
//! same Gerstner sum + ripple height as the raster, into a position buffer per
//! frame in flight, and that slot's water BLAS is refit (UPDATE) from it inside
//! the frame command buffer before the TLAS build. The hit shader keeps reading
//! the rest vertices: the wave normal is evaluated at the rest point, as the
//! raster does.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocator;

use crate::alloc::Buffer;
use crate::bake::{self, Accel};
use crate::context::GpuContext;
use crate::rt_reflect::RtInstance;
use crate::types::Vertex;
use crate::{GpuMesh, GpuWater, MAX_WATER_BODIES, frame::FRAMES_IN_FLIGHT};

const SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/water_displace.comp.spv"));

// water_displace.comp reads the 26-float engine vertex.
const _: () = assert!(std::mem::size_of::<Vertex>() == 104);

/// Frames a water mesh may go untraced before its BLASes are freed.
const IDLE_FRAMES: u64 = 240;

const BLAS_FLAGS: vk::BuildAccelerationStructureFlagsKHR = vk::BuildAccelerationStructureFlagsKHR::from_raw(
    vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE.as_raw()
        | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE.as_raw(),
);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Push {
    src: u64,
    dst: u64,
    count: u32,
    slot: u32,
    ripple: u32,
    offset: [f32; 2],
    _pad: u32,
}

/// Mirrors `WaterData` in water_displace.comp (std140).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WaterUbo {
    water: [f32; 4],
    bodies: [GpuWater; MAX_WATER_BODIES],
}

const _: () = assert!(std::mem::size_of::<WaterUbo>() == 16 + MAX_WATER_BODIES * 512);

/// This frame's water state for the RT scene.
pub(crate) struct RtWaterFrame {
    /// Water instances get displaced BLASes (RT reflections on); otherwise
    /// they keep the flat rest-grid BLAS.
    pub animate: bool,
    pub time: f32,
    pub bodies: [GpuWater; MAX_WATER_BODIES],
    /// Ripple field per body slot (null = none).
    pub ripples: [vk::ImageView; MAX_WATER_BODIES],
    /// Shore-depth field per body slot (null = none; sampled through the
    /// body's `extra` shore window to calm the displaced waves).
    pub shores: [vk::ImageView; MAX_WATER_BODIES],
}

impl RtWaterFrame {
    fn key(&self) -> u64 {
        let mut k = fnv(0xcbf29ce484222325, self.time.to_bits() as u64);
        for w in bytemuck::cast_slice::<GpuWater, u32>(&self.bodies) {
            k = fnv(k, *w as u64);
        }
        for v in self.ripples {
            k = fnv(k, vk::Handle::as_raw(v));
        }
        for v in self.shores {
            k = fnv(k, vk::Handle::as_raw(v));
        }
        k
    }
}

fn fnv(s: u64, v: u64) -> u64 {
    (s ^ v).wrapping_mul(0x100000001b3)
}

/// One frame slot's displaced copy of a water mesh and the BLAS over it.
struct Slot {
    positions: Buffer,
    accel: Accel,
    scratch: Buffer,
    scratch_addr: u64,
    vertex_count: u32,
    index_count: u32,
    /// Rest-mesh version the BLAS was last fully built from (None = never).
    built: Option<u32>,
    /// Water + rest-version key of the positions in `positions`.
    key: u64,
}

struct Entry {
    slots: Vec<Option<Slot>>,
    last_used: u64,
}

pub(crate) struct RtWater {
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    sampler: vk::Sampler,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    /// Per frame slot: the water uniforms (created on first use) and the key
    /// of what its UBO + set hold.
    ubos: Vec<Option<Buffer>>,
    set_keys: Vec<u64>,
    /// Keyed by mesh index.
    meshes: HashMap<usize, Entry>,
    epoch: u64,
}

impl RtWater {
    pub fn new(device: &ash::Device) -> Result<Self> {
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(MAX_WATER_BODIES as u32)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(MAX_WATER_BODIES as u32)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let binding_flags = [
            vk::DescriptorBindingFlags::empty(),
            vk::DescriptorBindingFlags::PARTIALLY_BOUND,
            vk::DescriptorBindingFlags::PARTIALLY_BOUND,
        ];
        let mut flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);
        let set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings).push_next(&mut flags_info),
                None,
            )?
        };
        let layouts = [set_layout];
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(std::mem::size_of::<Push>() as u32)];
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&ranges),
                None,
            )?
        };
        let module = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&ash::util::read_spv(&mut std::io::Cursor::new(SPV))?),
                None,
            )?
        };
        let pipeline = unsafe {
            device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::COMPUTE)
                                .module(module)
                                .name(c"main"),
                        )
                        .layout(pipeline_layout)],
                    None,
                )
                .map_err(|(_, e)| e)?[0]
        };
        unsafe { device.destroy_shader_module(module, None) };
        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .max_lod(vk::LOD_CLAMP_NONE)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?
        };
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(FRAMES_IN_FLIGHT as u32).pool_sizes(&[
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::UNIFORM_BUFFER,
                        descriptor_count: FRAMES_IN_FLIGHT as u32,
                    },
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        descriptor_count: (FRAMES_IN_FLIGHT * MAX_WATER_BODIES * 2) as u32,
                    },
                ]),
                None,
            )?
        };
        let set_layouts = vec![set_layout; FRAMES_IN_FLIGHT];
        let sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&set_layouts),
            )?
        };
        Ok(Self {
            set_layout,
            pipeline_layout,
            pipeline,
            sampler,
            pool,
            sets,
            ubos: (0..FRAMES_IN_FLIGHT).map(|_| None).collect(),
            set_keys: vec![0; FRAMES_IN_FLIGHT],
            meshes: HashMap::new(),
            epoch: 0,
        })
    }

    /// The water BLAS for `mesh` in `frame`'s slot and the key of its current
    /// contents (folded into the TLAS signature so a refit refits the TLAS).
    pub fn blas(&self, mesh: usize, frame: usize) -> Option<(u64, u64)> {
        let slot = self.meshes.get(&mesh)?.slots[frame].as_ref()?;
        slot.built.map(|_| (slot.accel.address, slot.key))
    }

    /// Displace + refit the BLAS of every water instance in `instances` whose
    /// waves or rest mesh changed since `frame`'s slot was last updated, all
    /// recorded into `cb`. Replaced resources go to `retired` (freed once this
    /// slot's fence is waited again). Returns whether any BLAS was built.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        ctx: &GpuContext,
        accel: &ash::khr::acceleration_structure::Device,
        allocator: &Arc<Mutex<Allocator>>,
        meshes: &[GpuMesh],
        instances: &[RtInstance],
        water: &RtWaterFrame,
        cb: vk::CommandBuffer,
        frame: usize,
        retired: &mut (Vec<Accel>, Vec<Buffer>),
    ) -> Result<bool> {
        let device = &ctx.device;
        let water_key = water.key();
        let epoch = self.epoch;

        // (mesh, body slot, rest-mesh version, slot key, mesh origin in the
        // body's local XZ) needing new positions this frame.
        let mut work: Vec<(usize, u32, u32, u64, [f32; 2])> = Vec::new();
        for inst in instances {
            if inst.water < 0 || work.iter().any(|w| w.0 == inst.mesh) {
                continue;
            }
            let Some(mesh) = meshes.get(inst.mesh) else { continue };
            if mesh.vertex_count == 0 || mesh.index_count < 3 {
                continue;
            }
            let version = inst.dynamic.unwrap_or(0);
            let body = (inst.water as u32).min(MAX_WATER_BODIES as u32 - 1);
            let local = glam::Mat4::from_cols_array_2d(&water.bodies[body as usize].to_local) * inst.transform;
            let offset = [local.w_axis.x, local.w_axis.z];
            let key = fnv(fnv(fnv(fnv(water_key, version as u64), body as u64), inst.mesh as u64), ((offset[0].to_bits() as u64) << 32) | offset[1].to_bits() as u64);
            let entry = self.meshes.entry(inst.mesh).or_insert_with(|| Entry {
                slots: (0..FRAMES_IN_FLIGHT).map(|_| None).collect(),
                last_used: epoch,
            });
            entry.last_used = epoch;
            let fits = entry.slots[frame]
                .as_ref()
                .is_some_and(|s| s.vertex_count == mesh.vertex_count && s.index_count == mesh.index_count);
            if !fits {
                if let Some(old) = entry.slots[frame].take() {
                    retire(old, retired);
                }
                entry.slots[frame] = Some(new_slot(ctx, accel, allocator, mesh)?);
            }
            if entry.slots[frame].as_ref().is_some_and(|s| s.key == key) {
                continue;
            }
            work.push((inst.mesh, body, version, key, offset));
        }
        if work.is_empty() {
            return Ok(false);
        }

        // This slot's uniforms + ripple bindings, rewritten only when the water
        // changed (never twice within a frame: every caller passes the same
        // frame state, so the set isn't updated after being bound).
        if self.ubos[frame].is_none() {
            let mut a = allocator.lock().unwrap();
            self.ubos[frame] = Some(Buffer::new(
                device,
                &mut a,
                std::mem::size_of::<WaterUbo>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                MemoryLocation::CpuToGpu,
                "rt-water-uniforms",
            )?);
            self.set_keys[frame] = 0;
        }
        let set = self.sets[frame];
        if self.set_keys[frame] != water_key {
            let ubo = self.ubos[frame].as_mut().unwrap();
            ubo.write(0, bytemuck::bytes_of(&WaterUbo { water: [water.time, 0.0, 0.0, 0.0], bodies: water.bodies }));
            let ubo_info = [vk::DescriptorBufferInfo::default().buffer(ubo.handle).range(vk::WHOLE_SIZE)];
            let ripple_infos: Vec<[vk::DescriptorImageInfo; 1]> = water
                .ripples
                .iter()
                .map(|&v| {
                    [vk::DescriptorImageInfo::default()
                        .sampler(self.sampler)
                        .image_view(v)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]
                })
                .collect();
            let shore_infos: Vec<[vk::DescriptorImageInfo; 1]> = water
                .shores
                .iter()
                .map(|&v| {
                    [vk::DescriptorImageInfo::default()
                        .sampler(self.sampler)
                        .image_view(v)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]
                })
                .collect();
            let mut writes = vec![vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&ubo_info)];
            for (i, info) in ripple_infos.iter().enumerate() {
                if info[0].image_view != vk::ImageView::null() {
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .dst_set(set)
                            .dst_binding(1)
                            .dst_array_element(i as u32)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .image_info(info),
                    );
                }
            }
            for (i, info) in shore_infos.iter().enumerate() {
                if info[0].image_view != vk::ImageView::null() {
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .dst_set(set)
                            .dst_binding(2)
                            .dst_array_element(i as u32)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .image_info(info),
                    );
                }
            }
            unsafe { device.update_descriptor_sets(&writes, &[]) };
            self.set_keys[frame] = water_key;
        }

        unsafe {
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &[set], &[]);
        }
        for &(mesh_i, body, _, _, offset) in &work {
            let mesh = &meshes[mesh_i];
            let slot = self.meshes[&mesh_i].slots[frame].as_ref().unwrap();
            let push = Push {
                src: mesh.vertex_buffer.device_address(device),
                dst: slot.positions.device_address(device),
                count: mesh.vertex_count,
                slot: body,
                ripple: (water.ripples[body as usize] != vk::ImageView::null()) as u32,
                offset,
                _pad: 0,
            };
            unsafe {
                device.cmd_push_constants(
                    cb,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::bytes_of(&push),
                );
                device.cmd_dispatch(cb, mesh.vertex_count.div_ceil(64), 1, 1);
            }
        }
        let barrier = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ)];
        unsafe { device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&barrier)) };

        // Refit in place when this slot's BLAS was built from the same rest
        // mesh (same topology, vertices moved by at most the wave amplitude);
        // a new or rewritten rest mesh gets a full build.
        for &(mesh_i, _, version, key, _) in &work {
            let mesh = &meshes[mesh_i];
            let slot = self.meshes.get_mut(&mesh_i).unwrap().slots[frame].as_mut().unwrap();
            let refit = slot.built == Some(version);
            let triangles = triangles(device, mesh, &slot.positions);
            let geometries = [vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
                .flags(vk::GeometryFlagsKHR::OPAQUE)];
            let mut info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                .flags(BLAS_FLAGS)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .geometries(&geometries)
                .dst_acceleration_structure(slot.accel.handle)
                .scratch_data(vk::DeviceOrHostAddressKHR { device_address: slot.scratch_addr });
            if refit {
                info = info
                    .mode(vk::BuildAccelerationStructureModeKHR::UPDATE)
                    .src_acceleration_structure(slot.accel.handle);
            }
            let ranges = [vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(mesh.index_count / 3)];
            unsafe { accel.cmd_build_acceleration_structures(cb, &[info], &[&ranges]) };
            slot.built = Some(version);
            slot.key = key;
        }
        Ok(true)
    }

    /// Advance the frame clock and free the BLASes of water meshes untraced
    /// for a while (long past any frame in flight); call once `frame`'s fence
    /// has been waited.
    pub fn begin_frame(&mut self, ctx: &GpuContext, allocator: &mut Allocator) {
        self.epoch += 1;
        let epoch = self.epoch;
        let idle: Vec<usize> = self
            .meshes
            .iter()
            .filter(|(_, e)| epoch.saturating_sub(e.last_used) > IDLE_FRAMES)
            .map(|(&m, _)| m)
            .collect();
        for m in idle {
            if let Some(e) = self.meshes.remove(&m) {
                for s in e.slots.into_iter().flatten() {
                    destroy_slot(ctx, allocator, s);
                }
            }
        }
    }

    pub fn destroy(&mut self, ctx: &GpuContext, allocator: &mut Allocator) {
        let device = &ctx.device;
        for (_, e) in self.meshes.drain() {
            for s in e.slots.into_iter().flatten() {
                destroy_slot(ctx, allocator, s);
            }
        }
        for mut b in self.ubos.drain(..).flatten() {
            b.destroy(device, allocator);
        }
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_sampler(self.sampler, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

/// BLAS triangle input: the displaced positions (packed vec3) + the mesh's indices.
fn triangles<'a>(device: &ash::Device, mesh: &GpuMesh, positions: &Buffer) -> vk::AccelerationStructureGeometryTrianglesDataKHR<'a> {
    vk::AccelerationStructureGeometryTrianglesDataKHR::default()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .vertex_data(vk::DeviceOrHostAddressConstKHR { device_address: positions.device_address(device) })
        .vertex_stride(12)
        .max_vertex(mesh.vertex_count.saturating_sub(1))
        .index_type(vk::IndexType::UINT32)
        .index_data(vk::DeviceOrHostAddressConstKHR { device_address: mesh.index_buffer.device_address(device) })
}

fn new_slot(
    ctx: &GpuContext,
    accel: &ash::khr::acceleration_structure::Device,
    allocator: &Arc<Mutex<Allocator>>,
    mesh: &GpuMesh,
) -> Result<Slot> {
    let device = &ctx.device;
    let scratch_align = bake::accel_scratch_alignment(ctx);
    let mut a = allocator.lock().unwrap();
    let positions = Buffer::new(
        device,
        &mut a,
        (mesh.vertex_count as u64 * 12).max(16),
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        MemoryLocation::GpuOnly,
        "rt-water-positions",
    )?;
    let geometries = [vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .geometry(vk::AccelerationStructureGeometryDataKHR { triangles: triangles(device, mesh, &positions) })
        .flags(vk::GeometryFlagsKHR::OPAQUE)];
    let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .flags(BLAS_FLAGS)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(&geometries);
    let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe {
        accel.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &info,
            &[mesh.index_count / 3],
            &mut sizes,
        )
    };
    let as_buffer = Buffer::new(
        device,
        &mut a,
        sizes.acceleration_structure_size,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        MemoryLocation::GpuOnly,
        "blas (water)",
    )?;
    let handle = unsafe {
        accel.create_acceleration_structure(
            &vk::AccelerationStructureCreateInfoKHR::default()
                .buffer(as_buffer.handle)
                .size(sizes.acceleration_structure_size)
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL),
            None,
        )?
    };
    let (scratch, scratch_addr) = bake::scratch_buffer(
        device,
        &mut a,
        sizes.build_scratch_size.max(sizes.update_scratch_size),
        scratch_align,
    )?;
    let address = unsafe {
        accel.get_acceleration_structure_device_address(
            &vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(handle),
        )
    };
    Ok(Slot {
        positions,
        accel: Accel { handle, buffer: as_buffer, address },
        scratch,
        scratch_addr,
        vertex_count: mesh.vertex_count,
        index_count: mesh.index_count,
        built: None,
        key: 0,
    })
}

fn retire(slot: Slot, retired: &mut (Vec<Accel>, Vec<Buffer>)) {
    retired.0.push(slot.accel);
    retired.1.push(slot.positions);
    retired.1.push(slot.scratch);
}

fn destroy_slot(ctx: &GpuContext, allocator: &mut Allocator, mut slot: Slot) {
    if let Some(accel) = ctx.accel.as_ref() {
        unsafe { accel.destroy_acceleration_structure(slot.accel.handle, None) };
    }
    slot.accel.buffer.destroy(&ctx.device, allocator);
    slot.positions.destroy(&ctx.device, allocator);
    slot.scratch.destroy(&ctx.device, allocator);
}
