//! GPU shallow-water simulation of the interactive water layer (see
//! `water_sim.comp`; `zestengine_core::PipeSim` is its CPU reference). Each queued
//! body keeps its state in GPU buffers; every frame its passes are recorded
//! into the frame command buffer, writing the body's ripple texture (so the
//! raster, RT reflections and RT displacement show it unchanged) and a
//! readback the CPU picks up once that frame slot's fence has been waited,
//! one or two frames later. Nothing on the frame path waits for the GPU.
//!
//! Further surface layers belong in the shader's OUTPUT pass, which sums
//! into both the texture and the readback. The dispersive layer (eWave,
//! `zestengine_core::EwaveLayer`) does: its sources, mode update and mask are
//! modes of the same shader, its 2D transforms `water_fft.comp` passes over
//! the same descriptor set.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocator;

use crate::alloc::Buffer;
use crate::frame::FRAMES_IN_FLIGHT;
use crate::texture::GpuTexture;
use crate::{MAX_WATER_BODIES, TextureHandle};

const SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/water_sim.comp.spv"));
const FFT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/water_fft.comp.spv"));

/// Displacers one body takes per frame.
pub const MAX_WATER_DISPLACERS: usize = 128;
/// Frames a body may go unqueued before its state is freed.
const IDLE_FRAMES: u64 = 240;
const BINDINGS: u32 = 17;
const IMAGE_BINDING: u32 = 9;
const READBACK_BINDING: u32 = 10;
/// The dispersive layer's buffers start here.
const EW_BINDING: u32 = 11;
/// The rest depth the columns are measured from (`Rest` in water_sim.comp).
const REST_BINDING: u32 = 16;

const MODE_RESET: u32 = 0;
const MODE_SPLAT: u32 = 1;
const MODE_FLUX: u32 = 2;
const MODE_HEIGHT: u32 = 3;
const MODE_ADVECT: u32 = 4;
const MODE_OUTPUT: u32 = 5;
const MODE_EW_SOURCE: u32 = 6;
const MODE_EW_UPDATE: u32 = 7;
const MODE_EW_MASK: u32 = 8;
const MODE_REBASE: u32 = 9;
/// `water_fft.comp` buffers.
const FFT_EW: u32 = 0;
const FFT_SPEC_A: u32 = 1;
const FFT_SPEC_B: u32 = 2;

/// The dispersive layer's constants for a frame (`zestengine_core::EwaveParams`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterSimEwave {
    /// Height per metre of body occupancy change.
    pub gain: f32,
    pub depth: f32,
    /// Damping of every mode (1/s).
    pub damping: f32,
    /// Wave number below which modes fade to the pipe sim (0 = none).
    pub split: f32,
    /// Uniform current carrying the waves (local XZ, m/s).
    pub current: [f32; 2],
    /// Share of a body's occupancy change that starts waves.
    pub body_gain: f32,
    /// Seconds a body's occupancy change is spread over (0 = at once).
    pub smoothing: f32,
    /// Damping (1/s) of the grid's finest waves.
    pub grid_damping: f32,
    /// Share of the layer that shows (0..1).
    pub mix: f32,
}

/// One water body's simulation input for a frame.
#[derive(Clone, Debug)]
pub struct WaterSimBody {
    pub key: u64,
    /// The body's ripple texture ([`crate::Renderer::create_ripple_texture`]).
    pub texture: TextureHandle,
    pub n: u32,
    pub cell: f32,
    /// The window's first cell, in whole cells of the water's local XZ.
    pub origin: [i32; 2],
    /// Edges are walls (the window covers the whole body).
    pub closed: bool,
    /// Absorbing border width in cells.
    pub sponge: u32,
    /// Start again from rest.
    pub reset: bool,
    /// Nothing moving: keep the last result without stepping.
    pub idle: bool,
    pub dt: f32,
    pub substeps: u32,
    pub gravity: f32,
    /// Pipe flow kept per substep after damping.
    pub keep: f32,
    pub momentum: f32,
    pub level_return: f32,
    pub wave_depth: f32,
    /// Scales the bodies' occupancy (not the pushes').
    pub displace: f32,
    /// Kinematic viscosity (m^2/s).
    pub viscosity: f32,
    /// Eddy viscosity rate (1/s; nu_eddy = eddy * cell^2).
    pub eddy: f32,
    /// The dispersive layer, when it runs (`n` a power of two).
    pub ewave: Option<WaterSimEwave>,
    /// Per-cell rest depth, base flow x/z and kind for the window at
    /// `field_origin`; uploaded when `field_version` changes.
    pub field_version: u64,
    pub field_origin: [i32; 2],
    pub field: Arc<Vec<[f32; 4]>>,
    pub flowing: bool,
    pub default_depth: f32,
    /// `zestengine_core::WaterDisplacer::to_gpu` per displacer.
    pub displacers: Vec<[f32; 12]>,
}

/// A finished frame of a body's simulation: per window cell (row-major from
/// `origin`) the height offset, the water velocity x/z and the drag weight.
#[derive(Clone, Debug, Default)]
pub struct WaterSimReadback {
    pub serial: u64,
    pub origin: [i32; 2],
    pub n: u32,
    pub cell: f32,
    pub cells: Vec<[f32; 4]>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Push {
    origin: [i32; 2],
    prev: [i32; 2],
    field_origin: [i32; 2],
    n: u32,
    mode: u32,
    cell: f32,
    dt: f32,
    gravity: f32,
    keep: f32,
    momentum: f32,
    level_return: f32,
    default_depth: f32,
    sponge: u32,
    displacers: u32,
    closed: u32,
    wave_depth: f32,
    displace: f32,
    eddy: f32,
    ew_gain: f32,
    ew_depth: f32,
    ew_damping: f32,
    ew_split: f32,
    ew_current: [f32; 2],
    ew_body_gain: f32,
    ew_follow: f32,
    ew_grid_damping: f32,
    viscosity: f32,
    ew_mix: f32,
}

const _: () = assert!(std::mem::size_of::<Push>() == 128);

/// `water_fft.comp`'s push constants (within the shared range).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FftPush {
    n: u32,
    log2n: u32,
    src: u32,
    dst: u32,
    inverse: u32,
    scale: f32,
}

struct FrameRes {
    set: vk::DescriptorSet,
    displacers: Buffer,
    readback: Buffer,
    staging: Option<Buffer>,
    /// Window of the readback this slot's last submission writes.
    pending: Option<([i32; 2], u32, f32)>,
}

struct Body {
    key: u64,
    n: u32,
    cell: f32,
    origin: [i32; 2],
    initialized: bool,
    field_version: u64,
    last_used: u64,
    /// h, flux, vel, field, couple, src, h_tmp, flux_tmp.
    state: Vec<Buffer>,
    /// ew, ew_prev, spectrum a, spectrum b, ew_in (bindings from `EW_BINDING`).
    ew: Vec<Buffer>,
    /// The rest depth each column is measured from (`REST_BINDING`).
    rest: Buffer,
    /// The dispersive layer ran last frame (its state is live).
    ew_on: bool,
    frames: Vec<FrameRes>,
}

impl Body {
    fn destroy(mut self, device: &ash::Device, allocator: &mut Allocator, pool: vk::DescriptorPool) {
        for mut b in self.state.drain(..).chain(self.ew.drain(..)) {
            b.destroy(device, allocator);
        }
        self.rest.destroy(device, allocator);
        for mut f in self.frames.drain(..) {
            f.displacers.destroy(device, allocator);
            f.readback.destroy(device, allocator);
            if let Some(mut s) = f.staging.take() {
                s.destroy(device, allocator);
            }
            unsafe {
                let _ = device.free_descriptor_sets(pool, &[f.set]);
            }
        }
    }
}

pub(crate) struct GpuWaterSim {
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    fft_pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
    bodies: Vec<Body>,
    /// Bodies freed once no frame in flight can use them (frames left).
    retired: Vec<(Body, usize)>,
    readbacks: HashMap<u64, WaterSimReadback>,
    serial: u64,
    epoch: u64,
}

impl GpuWaterSim {
    pub fn new(device: &ash::Device) -> Result<Self> {
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..BINDINGS)
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(if b == IMAGE_BINDING { vk::DescriptorType::STORAGE_IMAGE } else { vk::DescriptorType::STORAGE_BUFFER })
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let set_layout = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)? };
        let layouts = [set_layout];
        let ranges = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(std::mem::size_of::<Push>() as u32)];
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&ranges), None)?
        };
        let compile = |spv: &[u8]| -> Result<vk::Pipeline> {
            let module =
                unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&ash::util::read_spv(&mut std::io::Cursor::new(spv))?), None)? };
            let pipeline = unsafe {
                device.create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main"))
                        .layout(pipeline_layout)],
                    None,
                )
            };
            unsafe { device.destroy_shader_module(module, None) };
            Ok(pipeline.map_err(|(_, e)| e)?[0])
        };
        let pipeline = compile(SPV)?;
        let fft_pipeline = compile(FFT_SPV)?;
        // Room for every body several times over (a replaced body, as when
        // its resolution is dragged, waits out the frames in flight before
        // its sets are freed).
        let sets = (4 * MAX_WATER_BODIES * FRAMES_IN_FLIGHT) as u32;
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                    .max_sets(sets)
                    .pool_sizes(&[
                        vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: sets * (BINDINGS - 1) },
                        vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: sets },
                    ]),
                None,
            )?
        };
        Ok(Self {
            set_layout,
            pipeline_layout,
            pipeline,
            fft_pipeline,
            pool,
            bodies: Vec::new(),
            retired: Vec::new(),
            readbacks: HashMap::new(),
            serial: 0,
            epoch: 0,
        })
    }

    pub fn readback(&self, key: u64) -> Option<&WaterSimReadback> {
        self.readbacks.get(&key)
    }

    fn new_body(&self, device: &ash::Device, allocator: &mut Allocator, key: u64, n: u32) -> Result<Body> {
        let cells = (n * n) as u64;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC;
        let mut state = Vec::with_capacity(8);
        for (name, stride) in [("h", 4), ("flux", 16), ("vel", 8), ("field", 16), ("couple", 16), ("src", 4), ("h tmp", 4), ("flux tmp", 16)] {
            state.push(Buffer::new(device, allocator, cells * stride, usage, MemoryLocation::GpuOnly, &format!("water sim {name}"))?);
        }
        let mut ew = Vec::with_capacity(5);
        for (name, stride) in [("ew", 8), ("ew prev", 4), ("ew spectrum a", 8), ("ew spectrum b", 8), ("ew in", 8)] {
            ew.push(Buffer::new(device, allocator, cells * stride, usage, MemoryLocation::GpuOnly, &format!("water sim {name}"))?);
        }
        let rest = Buffer::new(device, allocator, cells * 4, usage, MemoryLocation::GpuOnly, "water sim rest")?;
        let layouts = vec![self.set_layout; FRAMES_IN_FLIGHT];
        let sets = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(self.pool).set_layouts(&layouts))? };
        let mut frames = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for set in sets {
            frames.push(FrameRes {
                set,
                displacers: Buffer::new(
                    device,
                    allocator,
                    (MAX_WATER_DISPLACERS * 48) as u64,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    MemoryLocation::CpuToGpu,
                    "water sim displacers",
                )?,
                readback: Buffer::new(device, allocator, cells * 16, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuToCpu, "water sim readback")?,
                staging: None,
                pending: None,
            });
        }
        Ok(Body { key, n, cell: 0.0, origin: [0; 2], initialized: false, field_version: 0, last_used: self.epoch, state, ew, rest, ew_on: false, frames })
    }

    /// Record this frame's simulation of `queued` into `cb` (frame slot
    /// `frame`, whose previous submission has finished). Call every frame,
    /// even with nothing queued, so finished readbacks are collected and
    /// unused bodies freed.
#[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        device: &ash::Device,
        allocator: &Arc<Mutex<Allocator>>,
        cb: vk::CommandBuffer,
        frame: usize,
        queued: &[WaterSimBody],
        textures: &[GpuTexture],
        ripple_mips: &HashMap<usize, (vk::ImageView, u32)>,
    ) -> Result<()> {
        self.epoch += 1;
        let mut alloc = allocator.lock().unwrap();
        for (_, left) in &mut self.retired {
            *left = left.saturating_sub(1);
        }
        let (done, keep): (Vec<_>, Vec<_>) = std::mem::take(&mut self.retired).into_iter().partition(|(_, left)| *left == 0);
        self.retired = keep;
        for (b, _) in done {
            b.destroy(device, &mut alloc, self.pool);
        }

        // Readbacks of this slot's last submission are complete.
        for b in &mut self.bodies {
            let f = &mut b.frames[frame];
            if let Some((origin, n, cell)) = f.pending.take()
                && let Some(bytes) = f.readback.mapped()
            {
                let count = (n * n) as usize;
                if bytes.len() >= count * 16 {
                    self.serial += 1;
                    let entry = self.readbacks.entry(b.key).or_default();
                    entry.serial = self.serial;
                    entry.origin = origin;
                    entry.n = n;
                    entry.cell = cell;
                    entry.cells.clear();
                    entry.cells.extend_from_slice(bytemuck::cast_slice(&bytes[..count * 16]));
                }
            }
        }

        let epoch = self.epoch;
        let stale: Vec<usize> = (0..self.bodies.len())
            .filter(|&i| {
                let b = &self.bodies[i];
                let q = queued.iter().find(|q| q.key == b.key);
                epoch.saturating_sub(b.last_used) > IDLE_FRAMES || q.is_some_and(|q| q.n != b.n)
            })
            .collect();
        for i in stale.into_iter().rev() {
            let b = self.bodies.swap_remove(i);
            self.readbacks.remove(&b.key);
            self.retired.push((b, FRAMES_IN_FLIGHT + 1));
        }

        // (body index, input, full reset, ripple image)
        let mut work: Vec<(usize, &WaterSimBody, bool, vk::Image)> = Vec::new();
        for q in queued.iter().take(MAX_WATER_BODIES) {
            let Some(tex) = textures.get(q.texture.0) else { continue };
            // The simulation writes mip 0; the chain below it is rebuilt after.
            let storage_view = ripple_mips.get(&q.texture.0).map_or(tex.view, |m| m.0);
            if q.n < 8 || q.field.len() != (q.n * q.n) as usize {
                continue;
            }
            let i = match self.bodies.iter().position(|b| b.key == q.key) {
                Some(i) => i,
                None => {
                    if self.bodies.len() >= MAX_WATER_BODIES {
                        continue;
                    }
                    let body = self.new_body(device, &mut alloc, q.key, q.n)?;
                    self.bodies.push(body);
                    self.bodies.len() - 1
                }
            };
            let b = &mut self.bodies[i];
            b.last_used = epoch;
            let full = !b.initialized || q.reset || (b.cell - q.cell).abs() > q.cell * 1e-5;
            if q.idle && !full && b.field_version == q.field_version && b.origin == q.origin {
                continue;
            }
            let f = &mut b.frames[frame];
            let count = q.displacers.len().min(MAX_WATER_DISPLACERS);
            if count > 0 {
                f.displacers.write(0, bytemuck::cast_slice(&q.displacers[..count]));
            }
            if b.field_version != q.field_version {
                let bytes: &[u8] = bytemuck::cast_slice(q.field.as_slice());
                if f.staging.is_none() {
                    f.staging = Some(Buffer::new(
                        device,
                        &mut alloc,
                        bytes.len() as u64,
                        vk::BufferUsageFlags::TRANSFER_SRC,
                        MemoryLocation::CpuToGpu,
                        "water sim field staging",
                    )?);
                }
                f.staging.as_mut().unwrap().write(0, bytes);
            }
            let infos: Vec<[vk::DescriptorBufferInfo; 1]> = b
                .state
                .iter()
                .chain([&f.displacers])
                .map(|buf| [vk::DescriptorBufferInfo::default().buffer(buf.handle).range(vk::WHOLE_SIZE)])
                .collect();
            let readback_info = [vk::DescriptorBufferInfo::default().buffer(f.readback.handle).range(vk::WHOLE_SIZE)];
            let image_info = [vk::DescriptorImageInfo::default().image_view(storage_view).image_layout(vk::ImageLayout::GENERAL)];
            let mut writes: Vec<vk::WriteDescriptorSet> = infos
                .iter()
                .enumerate()
                .map(|(k, info)| {
                    vk::WriteDescriptorSet::default().dst_set(f.set).dst_binding(k as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info)
                })
                .collect();
            writes.push(vk::WriteDescriptorSet::default().dst_set(f.set).dst_binding(IMAGE_BINDING).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&image_info));
            writes.push(
                vk::WriteDescriptorSet::default().dst_set(f.set).dst_binding(READBACK_BINDING).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&readback_info),
            );
            let ew_infos: Vec<[vk::DescriptorBufferInfo; 1]> =
                b.ew.iter().map(|buf| [vk::DescriptorBufferInfo::default().buffer(buf.handle).range(vk::WHOLE_SIZE)]).collect();
            for (k, info) in ew_infos.iter().enumerate() {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(f.set)
                        .dst_binding(EW_BINDING + k as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info),
                );
            }
            let rest_info = [vk::DescriptorBufferInfo::default().buffer(b.rest.handle).range(vk::WHOLE_SIZE)];
            writes.push(
                vk::WriteDescriptorSet::default().dst_set(f.set).dst_binding(REST_BINDING).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&rest_info),
            );
            unsafe { device.update_descriptor_sets(&writes, &[]) };
            work.push((i, q, full, tex.image));
        }
        drop(alloc);
        if work.is_empty() {
            return Ok(());
        }

        let barrier = |src_s: vk::PipelineStageFlags2, src_a: vk::AccessFlags2, dst_s: vk::PipelineStageFlags2, dst_a: vk::AccessFlags2| {
            let b = [vk::MemoryBarrier2::default().src_stage_mask(src_s).src_access_mask(src_a).dst_stage_mask(dst_s).dst_access_mask(dst_a)];
            unsafe { device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&b)) };
        };
        let compute = vk::PipelineStageFlags2::COMPUTE_SHADER;
        let rw = vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE;
        let transfer = vk::PipelineStageFlags2::TRANSFER;
        let compute_to_compute = || barrier(compute, vk::AccessFlags2::SHADER_STORAGE_WRITE, compute, rw);

        // Field uploads, after the previous frames' reads of the field, and
        // a dispersive layer starting up from rest (h = phi = 0, occupancy
        // not seen yet).
        let uploads = work.iter().any(|(i, q, _, _)| self.bodies[*i].field_version != q.field_version);
        // Bodies whose field is replaced this frame under state that carries
        // on: their columns move with the change of rest depth (REBASE).
        let rebase: Vec<bool> = work.iter().map(|(i, q, full, _)| !*full && self.bodies[*i].field_version != q.field_version).collect();
        let ew_starts = work.iter().any(|(i, q, _, _)| q.ewave.is_some() && !self.bodies[*i].ew_on);
        if uploads || ew_starts {
            barrier(compute, vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE, transfer, vk::AccessFlags2::TRANSFER_WRITE);
            for (i, q, _, _) in &work {
                let b = &mut self.bodies[*i];
                if q.ewave.is_some() && !b.ew_on {
                    unsafe {
                        device.cmd_fill_buffer(cb, b.ew[0].handle, 0, vk::WHOLE_SIZE, 0);
                        device.cmd_fill_buffer(cb, b.ew[1].handle, 0, vk::WHOLE_SIZE, (-1.0f32).to_bits());
                    }
                }
                if b.field_version == q.field_version {
                    continue;
                }
                let size = (q.field.len() * 16) as u64;
                let staging = b.frames[frame].staging.as_ref().unwrap();
                unsafe { device.cmd_copy_buffer(cb, staging.handle, b.state[3].handle, &[vk::BufferCopy::default().size(size)]) };
                b.field_version = q.field_version;
            }
            barrier(transfer, vk::AccessFlags2::TRANSFER_WRITE, compute, rw);
        }

        // All simulation modes use the same shader and descriptor set. The
        // storage image must have its declared layout before the first mode,
        // even when only MODE_OUTPUT takes the imageStore branch at runtime.
        let range = vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
        let readers = vk::PipelineStageFlags2::VERTEX_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER;
        let to_general: Vec<vk::ImageMemoryBarrier2> = work
            .iter()
            .map(|(_, _, _, image)| {
                vk::ImageMemoryBarrier2::default()
                    .image(*image)
                    .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_stage_mask(readers)
                    .src_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .dst_stage_mask(compute)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .subresource_range(range)
            })
            .collect();
        unsafe { device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&to_general)) };

        unsafe { device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline) };
        let (sim_pipeline, fft_pipeline) = (self.pipeline, self.fft_pipeline);
        let bind = |pipeline: vk::Pipeline| unsafe { device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline) };
        let pipeline_layout = self.pipeline_layout;
        let bodies = &mut self.bodies;
        let dispatch = |b: &Body, q: &WaterSimBody, mode: u32, prev: [i32; 2], dt: f32| {
            let ew = q.ewave.unwrap_or(WaterSimEwave { gain: 0.0, depth: 1.0, damping: 0.0, split: 0.0, current: [0.0; 2], body_gain: 0.0, smoothing: 0.0, grid_damping: 0.0, mix: 0.0 });
            let push = Push {
                origin: q.origin,
                prev,
                field_origin: q.field_origin,
                n: q.n,
                mode,
                cell: q.cell,
                dt,
                gravity: q.gravity,
                keep: q.keep,
                momentum: q.momentum,
                level_return: q.level_return,
                default_depth: q.default_depth,
                sponge: q.sponge,
                displacers: q.displacers.len().min(MAX_WATER_DISPLACERS) as u32,
                closed: q.closed as u32,
                wave_depth: q.wave_depth,
                displace: q.displace,
                eddy: q.eddy,
                ew_gain: ew.gain,
                ew_depth: ew.depth,
                ew_damping: ew.damping,
                ew_split: ew.split,
                ew_current: ew.current,
                ew_body_gain: ew.body_gain,
                ew_follow: if ew.smoothing <= 1e-4 { 1.0 } else { 1.0 - (-dt.max(0.0) / ew.smoothing).exp() },
                ew_grid_damping: ew.grid_damping,
                viscosity: q.viscosity,
                ew_mix: ew.mix,
            };
            let groups = q.n.div_ceil(8);
            unsafe {
                device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, pipeline_layout, 0, &[b.frames[frame].set], &[]);
                device.cmd_push_constants(cb, pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&push));
                device.cmd_dispatch(cb, groups, groups, 1);
            }
        };
        // One line pass of the dispersive layer's 2D transform.
        let fft = |b: &Body, q: &WaterSimBody, src: u32, dst: u32, inverse: bool| {
            let scale = if inverse && dst == FFT_EW { 1.0 / (q.n * q.n) as f32 } else { 1.0 };
            let push = FftPush { n: q.n, log2n: q.n.trailing_zeros(), src, dst, inverse: inverse as u32, scale };
            unsafe {
                device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, pipeline_layout, 0, &[b.frames[frame].set], &[]);
                device.cmd_push_constants(cb, pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&push));
                device.cmd_dispatch(cb, q.n, 1, 1);
            }
        };

        // Earlier frames' passes are done with the state (same queue, but
        // their writes must land before these reads).
        barrier(compute, vk::AccessFlags2::SHADER_STORAGE_WRITE, compute, rw);
        for (i, q, _, _) in &work {
            dispatch(&bodies[*i], q, MODE_SPLAT, q.origin, 0.0);
        }
        compute_to_compute();
        // Cells new to the window start level around this frame's bodies.
        let mut resets = false;
        for (i, q, full, _) in &work {
            let b = &bodies[*i];
            let prev = if *full {
                [q.origin[0] + 4 * q.n as i32, q.origin[1] + 4 * q.n as i32]
            } else {
                b.origin
            };
            if *full || prev != q.origin {
                dispatch(b, q, MODE_RESET, prev, 0.0);
                resets = true;
            }
        }
        if resets {
            compute_to_compute();
        }
        if rebase.iter().any(|r| *r) {
            for ((i, q, _, _), _) in work.iter().zip(&rebase).filter(|(_, r)| **r) {
                dispatch(&bodies[*i], q, MODE_REBASE, q.origin, 0.0);
            }
            compute_to_compute();
        }

        let stepping: Vec<&(usize, &WaterSimBody, bool, vk::Image)> = work.iter().filter(|(_, q, _, _)| q.dt > 0.0 && !q.idle).collect();
        // The dispersive layer's sources (bodies not stepping only remember
        // their occupancy); nothing the pipe passes touch.
        for (i, q, _, _) in work.iter().filter(|(_, q, _, _)| q.ewave.is_some()) {
            let dt = if q.dt > 0.0 && !q.idle { q.dt } else { 0.0 };
            dispatch(&bodies[*i], q, MODE_EW_SOURCE, q.origin, dt);
        }
        let advecting: Vec<_> = stepping.iter().filter(|(_, q, _, _)| q.flowing).collect();
        if !advecting.is_empty() {
            for (i, q, _, _) in advecting.iter().map(|w| **w) {
                dispatch(&bodies[*i], q, MODE_ADVECT, q.origin, q.dt);
            }
            barrier(compute, vk::AccessFlags2::SHADER_STORAGE_WRITE, transfer, vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE);
            for (i, q, _, _) in advecting.iter().map(|w| **w) {
                let b = &bodies[*i];
                let cells = (q.n * q.n) as u64;
                unsafe {
                    device.cmd_copy_buffer(cb, b.state[6].handle, b.state[0].handle, &[vk::BufferCopy::default().size(cells * 4)]);
                    device.cmd_copy_buffer(cb, b.state[7].handle, b.state[1].handle, &[vk::BufferCopy::default().size(cells * 16)]);
                }
            }
            barrier(transfer, vk::AccessFlags2::TRANSFER_WRITE, compute, rw);
        }
        let max_steps = stepping.iter().map(|(_, q, _, _)| q.substeps.max(1)).max().unwrap_or(0);
        for s in 0..max_steps {
            for mode in [MODE_FLUX, MODE_HEIGHT] {
                for (i, q, _, _) in stepping.iter().map(|w| **w) {
                    let steps = q.substeps.max(1);
                    if s < steps {
                        dispatch(&bodies[i], q, mode, q.origin, q.dt / steps as f32);
                    }
                }
                compute_to_compute();
            }
        }

        // The dispersive layer: forward transform, every mode moved on,
        // inverse transform, then solid cells and the sponge.
        let ew_stepping: Vec<(usize, &WaterSimBody)> = stepping.iter().filter(|(_, q, _, _)| q.ewave.is_some()).map(|(i, q, _, _)| (*i, *q)).collect();
        if !ew_stepping.is_empty() {
            compute_to_compute();
            bind(fft_pipeline);
            for &(i, q) in &ew_stepping {
                fft(&bodies[i], q, FFT_EW, FFT_SPEC_A, false);
            }
            compute_to_compute();
            for &(i, q) in &ew_stepping {
                fft(&bodies[i], q, FFT_SPEC_A, FFT_SPEC_B, false);
            }
            compute_to_compute();
            bind(sim_pipeline);
            for &(i, q) in &ew_stepping {
                dispatch(&bodies[i], q, MODE_EW_UPDATE, q.origin, q.dt);
            }
            compute_to_compute();
            bind(fft_pipeline);
            for &(i, q) in &ew_stepping {
                fft(&bodies[i], q, FFT_SPEC_A, FFT_SPEC_B, true);
            }
            compute_to_compute();
            for &(i, q) in &ew_stepping {
                fft(&bodies[i], q, FFT_SPEC_B, FFT_EW, true);
            }
            compute_to_compute();
            bind(sim_pipeline);
            for &(i, q) in &ew_stepping {
                dispatch(&bodies[i], q, MODE_EW_MASK, q.origin, q.dt);
            }
        }
        // The sources and mask land before the output reads them.
        if work.iter().any(|(_, q, _, _)| q.ewave.is_some()) {
            compute_to_compute();
        }

        for (i, q, _, _) in &work {
            dispatch(&bodies[*i], q, MODE_OUTPUT, q.origin, 0.0);
        }
        let to_host = [vk::MemoryBarrier2::default()
            .src_stage_mask(compute)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)];
        unsafe {
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&to_host));
        }
        // Mips of the new field, read band-limited by the surface mesh and
        // the caustics.
        for (_, q, _, image) in &work {
            let levels = ripple_mips.get(&q.texture.0).map_or(1, |m| m.1);
            crate::texture::record_mip_chain(
                device,
                cb,
                *image,
                q.n,
                levels,
                vk::ImageLayout::GENERAL,
                compute,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                readers,
            );
        }
        for (i, q, _, _) in &work {
            let b = &mut bodies[*i];
            b.initialized = true;
            b.n = q.n;
            b.cell = q.cell;
            b.origin = q.origin;
            b.ew_on = q.ewave.is_some();
            b.frames[frame].pending = Some((q.origin, q.n, q.cell));
        }
        Ok(())
    }

    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        for b in self.bodies.drain(..) {
            b.destroy(device, allocator, self.pool);
        }
        for (b, _) in self.retired.drain(..) {
            b.destroy(device, allocator, self.pool);
        }
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline(self.fft_pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}
