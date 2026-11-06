//! GPU lighting bake (Vulkan ray query). Builds acceleration structures over
//! the scene's static geometry, then:
//!   - per static object, rasters a world pos+normal gbuffer into lightmap
//!     (uv1) space and path-traces diffuse irradiance per texel,
//!   - per light probe, sphere-traces incoming radiance into SH-L1.
//!
//! Everything here is transient: created, submitted, waited on, read back to
//! host data, and torn down. It runs off the hot path (an explicit editor
//! action), so it submits and blocks rather than pipelining. Requires
//! `GpuContext::ray_tracing()`; callers check first.

use crate::hitch::WaitNamed as _;
use std::collections::HashMap;
use std::io::Cursor;
use std::mem::size_of;
use std::sync::{Arc, Mutex};

use anyhow::{Context as _, Result, bail};
use ash::{khr, vk};
use glam::Mat4;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocator;

use crate::alloc::{self, Buffer};
use crate::context::GpuContext;
use crate::types::{BakeInput, BakeOutput, BakedLightmap, LightKind, ProbeSh};
use crate::{GpuMesh, Vertex};

const GBUFFER_VERT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bake_gbuffer.vert.spv"));
const GBUFFER_FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bake_gbuffer.frag.spv"));
const LIGHTMAP_COMP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bake_lightmap.comp.spv"));
const PROBE_COMP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bake_probe.comp.spv"));

/// Rows-per-band (in workgroups; local_size_y = 8, so ×8 texel rows) for the tiled
/// lightmap dispatch. Each band is a separate GPU submission so the OS GPU
/// scheduler can interleave the desktop compositor between bands instead of one
/// giant dispatch hogging the GPU until the whole lightmap finishes.
const LIGHTMAP_BAND_GROUPS: u32 = 8;

/// Workgroups-per-chunk for the tiled probe dispatch (local_size_x = 64, so ×64
/// probes per chunk). Same idea as the lightmap bands — keep each submission short.
const PROBE_CHUNK_GROUPS: u32 = 16;

/// Submit a one-time command buffer, wait for it, then sleep for `idle_frac` of how
/// long it took — deliberately idling the GPU between heavy bake dispatches so the
/// rest of the system (compositor, other apps) gets GPU time. Tiling alone only
/// creates scheduler boundaries; this guarantees the GPU is actually free part of
/// the time, which is what keeps the machine responsive during a bake. `idle_frac`
/// = 0 disables the sleep (the realtime-GI preview path uses 0 so it stays fast).
fn throttled_submit(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    idle_frac: f32,
    f: impl FnOnce(vk::CommandBuffer),
) -> Result<()> {
    let t = std::time::Instant::now();
    alloc::one_time_submit(device, command_pool, queue, f)?;
    if idle_frac > 0.0 {
        let idle = t.elapsed().mul_f32(idle_frac);
        if idle > std::time::Duration::ZERO {
            std::thread::sleep(idle);
        }
    }
    Ok(())
}

const GBUF_FORMAT: vk::Format = vk::Format::R32G32B32A32_SFLOAT;

/// Mirrors the `Instance` struct in the bake shaders (scalar layout, 48 B).
#[repr(C)]
#[derive(Clone, Copy)]
struct GpuInstance {
    vtx_addr: u64,
    idx_addr: u64,
    albedo: [f32; 4],
    emission: [f32; 4],
}

/// Mirrors the `Light` struct in the bake shaders (64 B).
#[repr(C)]
#[derive(Clone, Copy)]
struct GpuLight {
    position: [f32; 4],  // xyz, w = kind
    direction: [f32; 4], // xyz, w = range
    color: [f32; 4],     // rgb × intensity
    spot: [f32; 4],      // x cos(inner/2), y cos(outer/2)
}

fn bytes_of<T: Copy>(slice: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice))
    }
}

/// A built acceleration structure plus its backing buffer. Shared with the
/// runtime ray-traced reflection path (`rt_reflect`).
pub(crate) struct Accel {
    pub handle: vk::AccelerationStructureKHR,
    pub buffer: Buffer,
    pub address: u64,
}

/// TLAS instance masks, matching `BAKE_MASK_*` in the bake shaders: bounce
/// rays test `MASK_VISIBLE`, shadow rays `MASK_CASTER`.
const MASK_VISIBLE: u8 = 0x01;
const MASK_CASTER: u8 = 0x02;

/// Bake TLAS mask for an instance's "Cast Shadows": `Off` stays visible to
/// bounce light but no longer blocks direct light; `ShadowsOnly` blocks direct
/// light but is invisible to bounces.
pub(crate) fn instance_mask(shadow_casting: crate::ShadowCasting) -> u8 {
    let mut mask = 0;
    if shadow_casting.visible() {
        mask |= MASK_VISIBLE;
    }
    if shadow_casting.casts() {
        mask |= MASK_CASTER;
    }
    mask
}

/// Run the full bake synchronously. Thin wrapper over the resumable `BakeJob`
/// (single code path); used by callers that don't need progress/chunking.
pub fn bake(
    ctx: &GpuContext,
    allocator: &Arc<Mutex<Allocator>>,
    command_pool: vk::CommandPool,
    meshes: &[GpuMesh],
    input: &BakeInput<'_>,
) -> Result<BakeOutput> {
    if input.instances.is_empty() && input.probes.is_empty() {
        return Ok(BakeOutput::default());
    }
    let mut job = BakeJob::begin(ctx, allocator, command_pool, meshes, input)?;
    loop {
        match job.step(ctx, allocator, command_pool, meshes) {
            Ok(BakeStep::Complete) => break,
            Ok(_) => {}
            Err(e) => {
                let _ = job.finish(ctx, allocator);
                return Err(e);
            }
        }
    }
    Ok(job.finish(ctx, allocator))
}

/// One unit of bake work completed by `BakeJob::step`.
pub enum BakeStep {
    /// A lightmap finished; `done`/`total` for progress.
    Lightmap { done: usize, total: usize },
    /// The probe batch finished.
    Probes,
    /// All work done — call `finish` to collect the output + free resources.
    Complete,
}

/// A resumable lighting bake: builds the acceleration structure + SSBOs +
/// pipelines once (`begin`), then traces one lightmap (or the probe batch) per
/// `step` so the editor can chunk it across frames and show progress. `finish`
/// tears down every GPU resource and returns the output — call it on completion
/// AND on cancel (a partial output is fine). The GPU resources are held across
/// frames, so the owning renderer's `meshes` must not be reallocated mid-bake
/// (the TLAS embeds their device addresses) — the editor gates that.
pub struct BakeJob {
    accels: Vec<Accel>,
    tlas: Option<Accel>,
    buffers: Vec<Buffer>,
    instance_ssbo: Buffer,
    light_ssbo: Buffer,
    /// Number of lights uploaded to `light_ssbo`. The per-step `BakeInput` is
    /// rebuilt with empty borrowed slices (they can't outlive `begin`), so the
    /// shader's `light_count` push constant MUST come from here, not from the
    /// temporary input — otherwise the bake traces zero direct light.
    light_count: u32,
    gbuffer: Option<GbufferPipeline>,
    lightmap: Option<LightmapPipeline>,
    /// Owned per-instance work list (mesh index, transform, lightmap size) so no
    /// borrow of the scene/`BakeInput` is held across frames.
    units: Vec<(usize, Mat4, u32)>,
    cursor: usize,
    probes: Vec<glam::Vec3>,
    probes_done: bool,
    sky_color: [f32; 3],
    bounces: u32,
    samples: u32,
    probes_only: bool,
    gpu_idle_frac: f32,
    output: BakeOutput,
    destroyed: bool,
}

impl BakeJob {
    pub(crate) fn begin(
        ctx: &GpuContext,
        allocator: &Arc<Mutex<Allocator>>,
        command_pool: vk::CommandPool,
        meshes: &[GpuMesh],
        input: &BakeInput<'_>,
    ) -> Result<BakeJob> {
        let Some(accel_loader) = ctx.accel.as_ref() else {
            bail!("ray tracing unavailable; cannot bake");
        };
        let device = &ctx.device;
        let scratch_align = accel_scratch_alignment(ctx);
        let mut buffers: Vec<Buffer> = Vec::new();
        let mut accels: Vec<Accel> = Vec::new();

        // BLAS per unique mesh referenced by an instance.
        let mut blas_for_mesh: HashMap<usize, usize> = HashMap::new();
        for inst in input.instances {
            let mesh_idx = inst.mesh.0;
            if blas_for_mesh.contains_key(&mesh_idx) {
                continue;
            }
            let mesh = &meshes[mesh_idx];
            let blas = build_blas(ctx, accel_loader, allocator, command_pool, mesh, scratch_align)
                .with_context(|| format!("building BLAS for mesh {mesh_idx}"))?;
            blas_for_mesh.insert(mesh_idx, accels.len());
            accels.push(blas);
        }

        // Instance descriptors (for shading hits) + TLAS instances.
        let mut gpu_instances = Vec::with_capacity(input.instances.len());
        let mut tlas_instances: Vec<vk::AccelerationStructureInstanceKHR> =
            Vec::with_capacity(input.instances.len());
        for (i, inst) in input.instances.iter().enumerate() {
            let mesh = &meshes[inst.mesh.0];
            gpu_instances.push(GpuInstance {
                vtx_addr: mesh.vertex_buffer.device_address(device),
                idx_addr: mesh.index_buffer.device_address(device),
                albedo: [inst.albedo[0], inst.albedo[1], inst.albedo[2], 0.0],
                emission: [inst.emission[0], inst.emission[1], inst.emission[2], 0.0],
            });
            let blas = &accels[blas_for_mesh[&inst.mesh.0]];
            tlas_instances.push(vk::AccelerationStructureInstanceKHR {
                transform: transform_matrix(&inst.transform),
                instance_custom_index_and_mask: vk::Packed24_8::new(i as u32, instance_mask(inst.shadow_casting)),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                    0,
                    vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
                ),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                    device_handle: blas.address,
                },
            });
        }

        let tlas = if tlas_instances.is_empty() {
            None
        } else {
            Some(
                build_tlas(
                    ctx,
                    accel_loader,
                    allocator,
                    command_pool,
                    &tlas_instances,
                    scratch_align,
                    &mut buffers,
                )
                .context("building TLAS")?,
            )
        };

        // Shared SSBOs: instance descriptors + lights.
        let mut alloc = allocator.lock().unwrap();
        let instance_ssbo =
            host_ssbo(device, &mut alloc, bytes_of(&gpu_instances), "bake-instances")?;
        let gpu_lights: Vec<GpuLight> = input.lights.iter().map(gpu_light).collect();
        let light_bytes = if gpu_lights.is_empty() {
            vec![0u8; size_of::<GpuLight>()]
        } else {
            bytes_of(&gpu_lights).to_vec()
        };
        let light_ssbo = host_ssbo(device, &mut alloc, &light_bytes, "bake-lights")?;
        drop(alloc);

        // Lightmap pipelines + work list only when we'll trace lightmaps.
        let do_lightmaps = tlas.is_some() && !input.probes_only;
        let (gbuffer, lightmap) = if do_lightmaps {
            (Some(GbufferPipeline::new(ctx)?), Some(LightmapPipeline::new(ctx)?))
        } else {
            (None, None)
        };
        let units: Vec<(usize, Mat4, u32)> = if do_lightmaps {
            input
                .instances
                .iter()
                .map(|i| (i.mesh.0, i.transform, i.lightmap_size.clamp(8, 2048)))
                .collect()
        } else {
            Vec::new()
        };

        Ok(BakeJob {
            accels,
            tlas,
            buffers,
            instance_ssbo,
            light_ssbo,
            light_count: input.lights.len() as u32,
            gbuffer,
            lightmap,
            units,
            cursor: 0,
            probes: input.probes.to_vec(),
            probes_done: false,
            sky_color: input.sky_color,
            bounces: input.bounces,
            samples: input.samples,
            probes_only: input.probes_only,
            gpu_idle_frac: input.gpu_idle_frac,
            output: BakeOutput::default(),
            destroyed: false,
        })
    }

    /// Do one unit of work (one lightmap, then the probe batch as the final
    /// step). Each per-unit fn takes the TLAS/SSBOs explicitly, so the temporary
    /// `BakeInput` carries only the scalar params (its slices stay empty).
    pub(crate) fn step(
        &mut self,
        ctx: &GpuContext,
        allocator: &Arc<Mutex<Allocator>>,
        command_pool: vk::CommandPool,
        meshes: &[GpuMesh],
    ) -> Result<BakeStep> {
        let (sky_color, bounces, samples, probes_only, gpu_idle_frac) = (
            self.sky_color,
            self.bounces,
            self.samples,
            self.probes_only,
            self.gpu_idle_frac,
        );

        if let Some(lightmap) = self.lightmap.as_ref()
            && self.cursor < self.units.len()
        {
            let (mesh_idx, transform, size) = self.units[self.cursor];
            let mesh = &meshes[mesh_idx];
            let lm = {
                let tlas = self.tlas.as_ref().unwrap();
                let gbuffer = self.gbuffer.as_ref().unwrap();
                let input = BakeInput {
                    instances: &[],
                    lights: &[],
                    probes: &[],
                    sky_color,
                    bounces,
                    samples,
                    probes_only,
                    gpu_idle_frac,
                };
                bake_one_lightmap(
                    ctx,
                    allocator,
                    command_pool,
                    gbuffer,
                    lightmap,
                    tlas,
                    &self.instance_ssbo,
                    &self.light_ssbo,
                    mesh,
                    &transform,
                    size,
                    self.light_count,
                    &input,
                )
                .with_context(|| format!("baking lightmap {}", self.cursor))?
            };
            self.output.lightmaps.push(lm);
            self.cursor += 1;
            return Ok(BakeStep::Lightmap {
                done: self.cursor,
                total: self.units.len(),
            });
        }

        if !self.probes_done {
            let probes_result = if let Some(tlas) = self.tlas.as_ref() {
                if self.probes.is_empty() {
                    None
                } else {
                    let input = BakeInput {
                        instances: &[],
                        lights: &[],
                        probes: &self.probes,
                        sky_color,
                        bounces,
                        samples,
                        probes_only,
                        gpu_idle_frac,
                    };
                    Some(
                        bake_probes(
                            ctx,
                            allocator,
                            command_pool,
                            tlas,
                            &self.instance_ssbo,
                            &self.light_ssbo,
                            self.light_count,
                            &input,
                        )
                        .context("baking probes")?,
                    )
                }
            } else {
                None
            };
            if let Some(p) = probes_result {
                self.output.probes = p;
            }
            self.probes_done = true;
            return Ok(BakeStep::Probes);
        }

        Ok(BakeStep::Complete)
    }

    /// Lightmaps finished so far (one per completed instance, in unit order).
    /// Used to live-preview the bake building up before it finishes.
    pub fn lightmaps_so_far(&self) -> &[crate::types::BakedLightmap] {
        &self.output.lightmaps
    }

    /// Tear down all GPU resources and return the (possibly partial) output.
    pub(crate) fn finish(mut self, ctx: &GpuContext, allocator: &Arc<Mutex<Allocator>>) -> BakeOutput {
        let device = &ctx.device;
        unsafe { device.wait_idle("wait:device_wait_idle@finish").ok() };
        let mut alloc = allocator.lock().unwrap();
        self.instance_ssbo.destroy(device, &mut alloc);
        self.light_ssbo.destroy(device, &mut alloc);
        if let Some(g) = self.gbuffer.take() {
            g.destroy(device);
        }
        if let Some(l) = self.lightmap.take() {
            l.destroy(device);
        }
        let accel_loader = ctx.accel.as_ref();
        if let Some(mut tlas) = self.tlas.take() {
            if let Some(al) = accel_loader {
                unsafe { al.destroy_acceleration_structure(tlas.handle, None) };
            }
            tlas.buffer.destroy(device, &mut alloc);
        }
        for mut a in std::mem::take(&mut self.accels) {
            if let Some(al) = accel_loader {
                unsafe { al.destroy_acceleration_structure(a.handle, None) };
            }
            a.buffer.destroy(device, &mut alloc);
        }
        for mut b in std::mem::take(&mut self.buffers) {
            b.destroy(device, &mut alloc);
        }
        drop(alloc);
        self.destroyed = true;
        std::mem::take(&mut self.output)
    }
}

impl Drop for BakeJob {
    fn drop(&mut self) {
        if !self.destroyed {
            tracing::error!("BakeJob dropped without finish(); GPU bake resources leaked");
        }
    }
}

fn gpu_light(l: &crate::types::BakeLight) -> GpuLight {
    let kind = match l.kind {
        LightKind::Directional => 0.0,
        LightKind::Point => 1.0,
        LightKind::Spot => 2.0,
    };
    let dir = l.direction.normalize_or_zero();
    GpuLight {
        position: [l.position.x, l.position.y, l.position.z, kind],
        direction: [dir.x, dir.y, dir.z, l.range],
        color: [l.color[0], l.color[1], l.color[2], 0.0],
        spot: [
            (l.spot_inner_deg.to_radians() * 0.5).cos(),
            (l.spot_outer_deg.to_radians() * 0.5).cos(),
            l.radius.max(0.0), // soft-shadow source radius
            0.0,
        ],
    }
}

/// glam column-major Mat4 → Vulkan row-major 3×4 instance transform.
pub(crate) fn transform_matrix(m: &Mat4) -> vk::TransformMatrixKHR {
    let c = m.to_cols_array();
    vk::TransformMatrixKHR {
        matrix: [
            c[0], c[4], c[8], c[12], c[1], c[5], c[9], c[13], c[2], c[6], c[10], c[14],
        ],
    }
}

pub(crate) fn accel_scratch_alignment(ctx: &GpuContext) -> u64 {
    let mut as_props = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut as_props);
    unsafe {
        ctx.instance
            .get_physical_device_properties2(ctx.physical_device, &mut props2)
    };
    as_props
        .min_acceleration_structure_scratch_offset_alignment
        .max(1) as u64
}

/// Host-visible SSBO with device-address usage, filled with `data`.
fn host_ssbo(
    device: &ash::Device,
    alloc: &mut Allocator,
    data: &[u8],
    name: &str,
) -> Result<Buffer> {
    let mut buf = Buffer::new(
        device,
        alloc,
        data.len().max(4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        MemoryLocation::CpuToGpu,
        name,
    )?;
    if !data.is_empty() {
        buf.write(0, data);
    }
    Ok(buf)
}

/// Scratch buffer whose device address is aligned for an AS build.
pub(crate) fn scratch_buffer(
    device: &ash::Device,
    alloc: &mut Allocator,
    size: u64,
    align: u64,
) -> Result<(Buffer, u64)> {
    let buf = Buffer::new(
        device,
        alloc,
        size + align,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        MemoryLocation::GpuOnly,
        "bake-scratch",
    )?;
    let base = buf.device_address(device);
    let aligned = base.div_ceil(align) * align;
    Ok((buf, aligned))
}

pub(crate) fn build_blas(
    ctx: &GpuContext,
    accel_loader: &khr::acceleration_structure::Device,
    allocator: &Arc<Mutex<Allocator>>,
    command_pool: vk::CommandPool,
    mesh: &GpuMesh,
    scratch_align: u64,
) -> Result<Accel> {
    let device = &ctx.device;
    let prim_count = mesh.index_count / 3;
    let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .vertex_data(vk::DeviceOrHostAddressConstKHR {
            device_address: mesh.vertex_buffer.device_address(device),
        })
        .vertex_stride(size_of::<Vertex>() as u64)
        .max_vertex(mesh.vertex_count.saturating_sub(1))
        .index_type(vk::IndexType::UINT32)
        .index_data(vk::DeviceOrHostAddressConstKHR {
            device_address: mesh.index_buffer.device_address(device),
        });
    let geometry = vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            triangles,
        })
        .flags(vk::GeometryFlagsKHR::OPAQUE);
    let geometries = [geometry];

    let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(&geometries);

    let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe {
        accel_loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &build_info,
            &[prim_count],
            &mut sizes,
        )
    };

    let mut alloc = allocator.lock().unwrap();
    let as_buffer = Buffer::new(
        device,
        &mut alloc,
        sizes.acceleration_structure_size,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        MemoryLocation::GpuOnly,
        "blas",
    )?;
    let handle = unsafe {
        accel_loader.create_acceleration_structure(
            &vk::AccelerationStructureCreateInfoKHR::default()
                .buffer(as_buffer.handle)
                .size(sizes.acceleration_structure_size)
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL),
            None,
        )?
    };
    let (mut scratch, scratch_addr) =
        scratch_buffer(device, &mut alloc, sizes.build_scratch_size, scratch_align)?;
    drop(alloc);

    build_info = build_info
        .dst_acceleration_structure(handle)
        .scratch_data(vk::DeviceOrHostAddressKHR {
            device_address: scratch_addr,
        });
    let range = vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(prim_count);
    let ranges = [range];
    alloc::one_time_submit(device, command_pool, ctx.queue, |cb| unsafe {
        accel_loader.cmd_build_acceleration_structures(cb, &[build_info], &[&ranges]);
    })?;

    let mut alloc = allocator.lock().unwrap();
    scratch.destroy(device, &mut alloc);
    drop(alloc);

    let address = unsafe {
        accel_loader.get_acceleration_structure_device_address(
            &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                .acceleration_structure(handle),
        )
    };
    Ok(Accel {
        handle,
        buffer: as_buffer,
        address,
    })
}

pub(crate) fn build_tlas(
    ctx: &GpuContext,
    accel_loader: &khr::acceleration_structure::Device,
    allocator: &Arc<Mutex<Allocator>>,
    command_pool: vk::CommandPool,
    instances: &[vk::AccelerationStructureInstanceKHR],
    scratch_align: u64,
    buffers: &mut Vec<Buffer>,
) -> Result<Accel> {
    let device = &ctx.device;
    let inst_bytes = unsafe {
        std::slice::from_raw_parts(
            instances.as_ptr() as *const u8,
            std::mem::size_of_val(instances),
        )
    };
    let mut alloc = allocator.lock().unwrap();
    let mut instance_buf = Buffer::new(
        device,
        &mut alloc,
        inst_bytes.len() as u64,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        MemoryLocation::CpuToGpu,
        "tlas-instances",
    )?;
    instance_buf.write(0, inst_bytes);
    let instance_addr = instance_buf.device_address(device);

    let geometry = vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            instances: vk::AccelerationStructureGeometryInstancesDataKHR::default().data(
                vk::DeviceOrHostAddressConstKHR {
                    device_address: instance_addr,
                },
            ),
        })
        .flags(vk::GeometryFlagsKHR::OPAQUE);
    let geometries = [geometry];
    let count = instances.len() as u32;

    let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(&geometries);
    let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe {
        accel_loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &build_info,
            &[count],
            &mut sizes,
        )
    };
    let as_buffer = Buffer::new(
        device,
        &mut alloc,
        sizes.acceleration_structure_size,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        MemoryLocation::GpuOnly,
        "tlas",
    )?;
    let handle = unsafe {
        accel_loader.create_acceleration_structure(
            &vk::AccelerationStructureCreateInfoKHR::default()
                .buffer(as_buffer.handle)
                .size(sizes.acceleration_structure_size)
                .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL),
            None,
        )?
    };
    let (scratch, scratch_addr) =
        scratch_buffer(device, &mut alloc, sizes.build_scratch_size, scratch_align)?;
    drop(alloc);

    build_info = build_info
        .dst_acceleration_structure(handle)
        .scratch_data(vk::DeviceOrHostAddressKHR {
            device_address: scratch_addr,
        });
    let range = vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(count);
    let ranges = [range];
    alloc::one_time_submit(device, command_pool, ctx.queue, |cb| unsafe {
        accel_loader.cmd_build_acceleration_structures(cb, &[build_info], &[&ranges]);
    })?;

    let address = unsafe {
        accel_loader.get_acceleration_structure_device_address(
            &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                .acceleration_structure(handle),
        )
    };
    // The instance + scratch buffers are dead after the build; queue them for
    // teardown with the rest.
    buffers.push(instance_buf);
    buffers.push(scratch);
    Ok(Accel {
        handle,
        buffer: as_buffer,
        address,
    })
}

fn shader_module(device: &ash::Device, spv: &[u8]) -> Result<vk::ShaderModule> {
    let code = ash::util::read_spv(&mut Cursor::new(spv))?;
    Ok(unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)?
    })
}

/// A transient rgba32f image (gbuffer attachment / storage / lightmap target).
struct BakeImage {
    image: vk::Image,
    view: vk::ImageView,
    allocation: Option<gpu_allocator::vulkan::Allocation>,
}

impl BakeImage {
    fn new(
        device: &ash::Device,
        alloc: &mut Allocator,
        w: u32,
        h: u32,
        usage: vk::ImageUsageFlags,
    ) -> Result<Self> {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(GBUF_FORMAT)
            .extent(vk::Extent3D {
                width: w,
                height: h,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.create_image(&info, None)? };
        let req = unsafe { device.get_image_memory_requirements(image) };
        let allocation = alloc.allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "bake-image",
            requirements: req,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        })?;
        unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset())? };
        let view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(GBUF_FORMAT)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                    ),
                None,
            )?
        };
        Ok(Self {
            image,
            view,
            allocation: Some(allocation),
        })
    }

    fn destroy(&mut self, device: &ash::Device, alloc: &mut Allocator) {
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
        }
        if let Some(a) = self.allocation.take() {
            let _ = alloc.free(a);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn image_barrier(
    device: &ash::Device,
    cb: vk::CommandBuffer,
    image: vk::Image,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
    src_stage: vk::PipelineStageFlags2,
    src_access: vk::AccessFlags2,
    dst_stage: vk::PipelineStageFlags2,
    dst_access: vk::AccessFlags2,
) {
    let barrier = vk::ImageMemoryBarrier2::default()
        .image(image)
        .old_layout(old)
        .new_layout(new)
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1),
        );
    let barriers = [barrier];
    unsafe {
        device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&barriers))
    };
}

/// Graphics pipeline that rasters a mesh into the lightmap's pos+normal
/// gbuffer (two rgba32f attachments, dynamic rendering, model push constant).
struct GbufferPipeline {
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    vert: vk::ShaderModule,
    frag: vk::ShaderModule,
}

impl GbufferPipeline {
    fn new(ctx: &GpuContext) -> Result<Self> {
        let device = &ctx.device;
        let vert = shader_module(device, GBUFFER_VERT)?;
        let frag = shader_module(device, GBUFFER_FRAG)?;
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .size(size_of::<Mat4>() as u32)];
        let layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push),
                None,
            )?
        };

        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert)
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag)
                .name(c"main"),
        ];
        let bindings = [vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)];
        let attrs = [
            vk::VertexInputAttributeDescription { location: 0, binding: 0, format: vk::Format::R32G32B32_SFLOAT, offset: 0 },
            vk::VertexInputAttributeDescription { location: 1, binding: 0, format: vk::Format::R32G32B32_SFLOAT, offset: 12 },
            vk::VertexInputAttributeDescription { location: 2, binding: 0, format: vk::Format::R32G32_SFLOAT, offset: 24 },
            vk::VertexInputAttributeDescription { location: 3, binding: 0, format: vk::Format::R32G32B32A32_SFLOAT, offset: 32 },
            vk::VertexInputAttributeDescription { location: 4, binding: 0, format: vk::Format::R32G32B32A32_SFLOAT, offset: 48 },
            vk::VertexInputAttributeDescription { location: 5, binding: 0, format: vk::Format::R32G32_SFLOAT, offset: 64 },
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attrs);
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        // No culling: lightmap texels for back faces still need their data.
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let blend_attachments = [
            vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA),
            vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA),
        ];
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let formats = [GBUF_FORMAT, GBUF_FORMAT];
        let mut rendering =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);

        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic)
            .layout(layout)
            .push_next(&mut rendering);
        let pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
                .map_err(|(_, e)| e)?[0]
        };
        Ok(Self {
            layout,
            pipeline,
            vert,
            frag,
        })
    }

    fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_shader_module(self.vert, None);
            device.destroy_shader_module(self.frag, None);
        }
    }
}

/// Lightmap path-trace compute pipeline (6-binding set: TLAS, instances,
/// lights, g_pos, g_normal, out_lightmap).
struct LightmapPipeline {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    module: vk::ShaderModule,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct LightmapPush {
    light_count: u32,
    sample_count: u32,
    bounces: u32,
    frame_seed: u32,
    sky_color: [f32; 4],
}

impl LightmapPipeline {
    fn new(ctx: &GpuContext) -> Result<Self> {
        let device = &ctx.device;
        let stage = vk::ShaderStageFlags::COMPUTE;
        let storage_img = |b: u32| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(stage)
        };
        let ssbo = |b: u32| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(stage)
        };
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1)
                .stage_flags(stage),
            ssbo(1),
            ssbo(2),
            storage_img(3),
            storage_img(4),
            storage_img(5),
        ];
        let set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?
        };
        let set_layouts = [set_layout];
        let push = [vk::PushConstantRange::default()
            .stage_flags(stage)
            .size(size_of::<LightmapPush>() as u32)];
        let layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&push),
                None,
            )?
        };
        let module = shader_module(device, LIGHTMAP_COMP)?;
        let pipeline = unsafe {
            device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        // DISPATCH_BASE lets the bake tile the lightmap into row
                        // bands via cmd_dispatch_base (one submit per band) so a
                        // single huge dispatch can't monopolize the GPU and freeze
                        // the whole desktop while baking.
                        .flags(vk::PipelineCreateFlags::DISPATCH_BASE)
                        .stage(
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(stage)
                                .module(module)
                                .name(c"main"),
                        )
                        .layout(layout)],
                    None,
                )
                .map_err(|(_, e)| e)?[0]
        };
        Ok(Self {
            set_layout,
            layout,
            pipeline,
            module,
        })
    }

    fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_shader_module(self.module, None);
        }
    }
}

/// Write a TLAS into a descriptor set binding.
fn write_tlas(device: &ash::Device, set: vk::DescriptorSet, binding: u32, tlas: vk::AccelerationStructureKHR) {
    let accels = [tlas];
    let mut as_write =
        vk::WriteDescriptorSetAccelerationStructureKHR::default().acceleration_structures(&accels);
    let mut write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
        .push_next(&mut as_write);
    write.descriptor_count = 1;
    unsafe { device.update_descriptor_sets(&[write], &[]) };
}

fn write_ssbo(device: &ash::Device, set: vk::DescriptorSet, binding: u32, buffer: vk::Buffer) {
    let info = [vk::DescriptorBufferInfo::default()
        .buffer(buffer)
        .range(vk::WHOLE_SIZE)];
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(&info);
    unsafe { device.update_descriptor_sets(&[write], &[]) };
}

fn write_storage_image(device: &ash::Device, set: vk::DescriptorSet, binding: u32, view: vk::ImageView) {
    let info = [vk::DescriptorImageInfo::default()
        .image_view(view)
        .image_layout(vk::ImageLayout::GENERAL)];
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .image_info(&info);
    unsafe { device.update_descriptor_sets(&[write], &[]) };
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn texel(x: i32, y: i32, size: i32) -> usize {
    ((y * size + x) as usize) * 4
}

/// Largest world-space spacing to a valid axis neighbor `step` away. This is the
/// local texel size, used to scale the denoise position weight (scale-invariant).
fn neighbor_spacing(pos: &[f32], px: &[f32], x: i32, y: i32, step: i32, s: i32) -> f32 {
    let ci = texel(x, y, s);
    let p0 = [pos[ci], pos[ci + 1], pos[ci + 2]];
    let mut best = 0.0f32;
    for (dx, dy) in [(step, 0), (0, step), (-step, 0), (0, -step)] {
        let nx = (x + dx).clamp(0, s - 1);
        let ny = (y + dy).clamp(0, s - 1);
        let ni = texel(nx, ny, s);
        if px[ni + 3] <= 0.0 {
            continue;
        }
        let d = ((pos[ni] - p0[0]).powi(2)
            + (pos[ni + 1] - p0[1]).powi(2)
            + (pos[ni + 2] - p0[2]).powi(2))
        .sqrt();
        best = best.max(d);
    }
    best
}

/// Edge-aware À-Trous denoise (Dammertz et al.): a few wavelet passes with
/// increasing hole size, each a 5×5 B3-spline blur weighted by world-position
/// and normal similarity (from the bake gbuffer) so it smooths Monte-Carlo
/// grain without crossing shadow/geometry edges or UV-chart seams. RGBA32F;
/// `.a` is per-texel validity (0 = no surface).
fn denoise_atrous(pixels: &mut Vec<f32>, pos: &[f32], nrm: &[f32], size: u32, iters: u32) {
    let s = size as i32;
    if s < 3 {
        return;
    }
    const K: [f32; 5] = [1.0, 4.0, 6.0, 4.0, 1.0]; // B3 spline
    let mut src = pixels.clone();
    let mut dst = pixels.clone();
    for it in 0..iters {
        let step = 1i32 << it;
        for y in 0..s {
            for x in 0..s {
                let ci = texel(x, y, s);
                if src[ci + 3] <= 0.0 {
                    dst[ci..ci + 4].copy_from_slice(&src[ci..ci + 4]);
                    continue;
                }
                let p0 = [pos[ci], pos[ci + 1], pos[ci + 2]];
                let n0 = [nrm[ci], nrm[ci + 1], nrm[ci + 2]];
                let lum = |i: usize| 0.2126 * src[i] + 0.7152 * src[i + 1] + 0.0722 * src[i + 2];
                let l0 = lum(ci);
                let h = neighbor_spacing(pos, &src, x, y, step, s);
                let inv2h2 = if h > 1e-6 { 1.0 / (2.0 * h * h) } else { 0.0 };
                let mut sum = [0.0f32; 3];
                let mut wsum = 0.0f32;
                for ky in -2..=2i32 {
                    for kx in -2..=2i32 {
                        let nx = (x + kx * step).clamp(0, s - 1);
                        let ny = (y + ky * step).clamp(0, s - 1);
                        let ni = texel(nx, ny, s);
                        if src[ni + 3] <= 0.0 {
                            continue;
                        }
                        let wk = K[(kx + 2) as usize] * K[(ky + 2) as usize];
                        let d2 = (pos[ni] - p0[0]).powi(2)
                            + (pos[ni + 1] - p0[1]).powi(2)
                            + (pos[ni + 2] - p0[2]).powi(2);
                        let wp = if inv2h2 > 0.0 { (-d2 * inv2h2).exp() } else { 1.0 };
                        let ndot =
                            (nrm[ni] * n0[0] + nrm[ni + 1] * n0[1] + nrm[ni + 2] * n0[2]).max(0.0);
                        // Normal edge-stop. The exponent sets how fast the blur
                        // cuts off as normals diverge. `pow(32)` was far too tight
                        // (it stopped at ~15-20°), so a low-poly surface with
                        // smoothly-varying interpolated normals baked with hard
                        // faceted edges between faces. `pow(8)` blurs across
                        // moderate curvature (≈30-40°) for a smooth lightmap while
                        // still cutting hard at real corners (90° → ndot 0). This
                        // is why the realtime (interpolated-normal) GI looked smooth
                        // but the baked lightmap showed clear polygon edges.
                        let wn = ndot.powf(8.0);
                        // Luminance edge-stop (relative, scale-invariant): smooths
                        // similar-brightness noise but preserves lighting edges
                        // (shadow boundaries, falloff) on flat surfaces. Without
                        // this the denoiser washes shadows/gradients away.
                        let wl = (-(lum(ni) - l0).abs() / (0.6 * (l0 + 0.02))).exp();
                        let w = wk * wp * wn * wl;
                        sum[0] += src[ni] * w;
                        sum[1] += src[ni + 1] * w;
                        sum[2] += src[ni + 2] * w;
                        wsum += w;
                    }
                }
                if wsum > 1e-8 {
                    dst[ci] = sum[0] / wsum;
                    dst[ci + 1] = sum[1] / wsum;
                    dst[ci + 2] = sum[2] / wsum;
                    dst[ci + 3] = src[ci + 3];
                } else {
                    dst[ci..ci + 4].copy_from_slice(&src[ci..ci + 4]);
                }
            }
        }
        std::mem::swap(&mut src, &mut dst);
    }
    *pixels = src;
}

/// Stitch lightmap seams: texels that land on the same world-space point but in
/// different UV charts (the cube's per-face atlas, the sphere's lat-long
/// meridian) get different Monte-Carlo results, so the chart boundary shows as a
/// line. Average co-located texels whose normals agree (so genuine hard-edge
/// lighting discontinuities, where the normals differ, are left intact). Uses
/// the gbuffer world position + normal already read back for the denoiser, so
/// it's mesh-agnostic.
fn stitch_seams(pixels: &mut [f32], pos: &[f32], nrm: &[f32], size: u32) {
    let s = size as i32;
    // World-space AABB of valid texels, giving a quantization cell ~1.5 texels
    // wide so points that coincide in 3D share a cell regardless of mesh scale.
    let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    let mut any = false;
    for t in 0..(size * size) as usize {
        if pixels[t * 4 + 3] <= 0.0 {
            continue;
        }
        any = true;
        for k in 0..3 {
            lo[k] = lo[k].min(pos[t * 4 + k]);
            hi[k] = hi[k].max(pos[t * 4 + k]);
        }
    }
    if !any {
        return;
    }
    let diag = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt();
    let cell = (diag / size as f32 * 1.5).max(1e-5);

    // Bucket valid texels by quantized world cell.
    let key = |t: usize| -> (i32, i32, i32) {
        (
            (pos[t * 4] / cell).floor() as i32,
            (pos[t * 4 + 1] / cell).floor() as i32,
            (pos[t * 4 + 2] / cell).floor() as i32,
        )
    };
    let mut buckets: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for y in 0..s {
        for x in 0..s {
            let t = (y * size as i32 + x) as usize;
            if pixels[t * 4 + 3] > 0.0 {
                buckets.entry(key(t)).or_default().push(t);
            }
        }
    }
    // Within each multi-texel cell, average the group of texels sharing a normal
    // direction (dot > 0.9), so only smooth seams are merged.
    for group in buckets.values() {
        if group.len() < 2 {
            continue;
        }
        for &a in group {
            let na = [nrm[a * 4], nrm[a * 4 + 1], nrm[a * 4 + 2]];
            let mut sum = [0.0f32; 3];
            let mut n = 0u32;
            for &b in group {
                let nb = [nrm[b * 4], nrm[b * 4 + 1], nrm[b * 4 + 2]];
                if na[0] * nb[0] + na[1] * nb[1] + na[2] * nb[2] > 0.9 {
                    sum[0] += pixels[b * 4];
                    sum[1] += pixels[b * 4 + 1];
                    sum[2] += pixels[b * 4 + 2];
                    n += 1;
                }
            }
            if n > 1 {
                pixels[a * 4] = sum[0] / n as f32;
                pixels[a * 4 + 1] = sum[1] / n as f32;
                pixels[a * 4 + 2] = sum[2] / n as f32;
            }
        }
    }
}

/// Dilate valid texels outward into invalid (gutter) texels, `iters` rings.
/// Bilinear sampling at a UV-chart edge then reads the surface colour instead
/// of the black background, which removes lightmap seams.
fn dilate_lightmap(pixels: &mut [f32], size: u32, iters: u32) {
    let s = size as i32;
    for _ in 0..iters {
        let src = pixels.to_vec();
        for y in 0..s {
            for x in 0..s {
                let ci = texel(x, y, s);
                if src[ci + 3] > 0.0 {
                    continue;
                }
                let mut sum = [0.0f32; 3];
                let mut n = 0u32;
                for dy in -1..=1i32 {
                    for dx in -1..=1i32 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = (x + dx).clamp(0, s - 1);
                        let ny = (y + dy).clamp(0, s - 1);
                        let ni = texel(nx, ny, s);
                        if src[ni + 3] > 0.0 {
                            sum[0] += src[ni];
                            sum[1] += src[ni + 1];
                            sum[2] += src[ni + 2];
                            n += 1;
                        }
                    }
                }
                if n > 0 {
                    pixels[ci] = sum[0] / n as f32;
                    pixels[ci + 1] = sum[1] / n as f32;
                    pixels[ci + 2] = sum[2] / n as f32;
                    pixels[ci + 3] = 1.0;
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn bake_one_lightmap(
    ctx: &GpuContext,
    allocator: &Arc<Mutex<Allocator>>,
    command_pool: vk::CommandPool,
    gbuffer: &GbufferPipeline,
    lightmap: &LightmapPipeline,
    tlas: &Accel,
    instance_ssbo: &Buffer,
    light_ssbo: &Buffer,
    mesh: &GpuMesh,
    transform: &Mat4,
    size: u32,
    light_count: u32,
    input: &BakeInput<'_>,
) -> Result<BakedLightmap> {
    let device = &ctx.device;
    let mut alloc = allocator.lock().unwrap();
    let mut g_pos = BakeImage::new(
        device,
        &mut alloc,
        size,
        size,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE,
    )?;
    let mut g_normal = BakeImage::new(
        device,
        &mut alloc,
        size,
        size,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE,
    )?;
    let mut out_img = BakeImage::new(
        device,
        &mut alloc,
        size,
        size,
        vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
    )?;
    let rb_bytes = (size * size * 4 * 4) as u64;
    let readback = Buffer::new(
        device,
        &mut alloc,
        rb_bytes,
        vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuToCpu,
        "lightmap-readback",
    )?;
    // Read the gbuffer back too, for the edge-aware denoise (CPU À-Trous uses
    // per-texel world position + normal to avoid blurring across edges/seams).
    let pos_readback = Buffer::new(
        device, &mut alloc, rb_bytes,
        vk::BufferUsageFlags::TRANSFER_DST, MemoryLocation::GpuToCpu, "lightmap-pos-readback",
    )?;
    let nrm_readback = Buffer::new(
        device, &mut alloc, rb_bytes,
        vk::BufferUsageFlags::TRANSFER_DST, MemoryLocation::GpuToCpu, "lightmap-nrm-readback",
    )?;
    drop(alloc);

    // Transient descriptor pool for this object's compute set.
    let pool_sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            descriptor_count: 1,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 2,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: 3,
        },
    ];
    let pool = unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes),
            None,
        )?
    };
    let set_layouts = [lightmap.set_layout];
    let set = unsafe {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&set_layouts),
        )?[0]
    };
    write_tlas(device, set, 0, tlas.handle);
    write_ssbo(device, set, 1, instance_ssbo.handle);
    write_ssbo(device, set, 2, light_ssbo.handle);
    write_storage_image(device, set, 3, g_pos.view);
    write_storage_image(device, set, 4, g_normal.view);
    write_storage_image(device, set, 5, out_img.view);

    let push = LightmapPush {
        light_count,
        sample_count: input.samples.max(1),
        bounces: input.bounces,
        frame_seed: 1,
        sky_color: [input.sky_color[0], input.sky_color[1], input.sky_color[2], 0.0],
    };
    let model = transform.to_cols_array();
    let groups = size.div_ceil(8);

    alloc::one_time_submit(device, command_pool, ctx.queue, |cb| unsafe {
        // gbuffer attachments → COLOR_ATTACHMENT_OPTIMAL
        for img in [g_pos.image, g_normal.image] {
            image_barrier(
                device, cb, img,
                vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags2::TOP_OF_PIPE, vk::AccessFlags2::empty(),
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            );
        }
        let clear = vk::ClearValue {
            color: vk::ClearColorValue { float32: [0.0; 4] },
        };
        let attachments = [
            vk::RenderingAttachmentInfo::default()
                .image_view(g_pos.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(clear),
            vk::RenderingAttachmentInfo::default()
                .image_view(g_normal.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(clear),
        ];
        let area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width: size, height: size },
        };
        let rendering = vk::RenderingInfo::default()
            .render_area(area)
            .layer_count(1)
            .color_attachments(&attachments);
        device.cmd_begin_rendering(cb, &rendering);
        device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, gbuffer.pipeline);
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: size as f32,
            height: size as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        device.cmd_set_viewport(cb, 0, &[viewport]);
        device.cmd_set_scissor(cb, 0, &[area]);
        device.cmd_push_constants(
            cb,
            gbuffer.layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            bytes_of(&model),
        );
        device.cmd_bind_vertex_buffers(cb, 0, &[mesh.vertex_buffer.handle], &[0]);
        device.cmd_bind_index_buffer(cb, mesh.index_buffer.handle, 0, mesh.index_type);
        device.cmd_draw_indexed(cb, mesh.index_count, 1, 0, 0, 0);
        device.cmd_end_rendering(cb);

        // gbuffers → GENERAL (compute read), out image UNDEFINED → GENERAL
        for img in [g_pos.image, g_normal.image] {
            image_barrier(
                device, cb, img,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_READ,
            );
        }
        image_barrier(
            device, cb, out_img.image,
            vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL,
            vk::PipelineStageFlags2::TOP_OF_PIPE, vk::AccessFlags2::empty(),
            vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_WRITE,
        );

    })?;

    // Path-trace the lightmap in horizontal row bands, each its own GPU
    // submission, so the OS GPU scheduler can interleave the desktop compositor
    // between bands instead of one giant dispatch monopolizing the GPU (which
    // froze the whole system) until the lightmap finished. DISPATCH_BASE offsets
    // gl_WorkGroupID so gl_GlobalInvocationID.y already carries the band origin —
    // no shader / push-constant change needed. Bands write disjoint rows, so no
    // inter-band barrier is required; the copy submit below synchronizes them.
    let mut y0 = 0u32;
    while y0 < groups {
        let bg = LIGHTMAP_BAND_GROUPS.min(groups - y0);
        throttled_submit(device, command_pool, ctx.queue, input.gpu_idle_frac, |cb| unsafe {
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, lightmap.pipeline);
            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                lightmap.layout,
                0,
                &[set],
                &[],
            );
            device.cmd_push_constants(
                cb,
                lightmap.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes_of(std::slice::from_ref(&push)),
            );
            device.cmd_dispatch_base(cb, 0, y0, 0, groups, bg, 1);
        })?;
        y0 += bg;
    }

    alloc::one_time_submit(device, command_pool, ctx.queue, |cb| unsafe {
        // out image → TRANSFER_SRC, copy to readback buffer
        image_barrier(
            device, cb, out_img.image,
            vk::ImageLayout::GENERAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_READ,
        );
        let copy = vk::BufferImageCopy::default()
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D { width: size, height: size, depth: 1 });
        device.cmd_copy_image_to_buffer(
            cb,
            out_img.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            readback.handle,
            &[copy],
        );
        // Gbuffer → readback (for the CPU edge-aware denoise): GENERAL → TRANSFER_SRC.
        for (img, buf) in [(g_pos.image, pos_readback.handle), (g_normal.image, nrm_readback.handle)]
        {
            image_barrier(
                device, cb, img,
                vk::ImageLayout::GENERAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_READ,
                vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_READ,
            );
            device.cmd_copy_image_to_buffer(cb, img, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, buf, &[copy]);
        }
    })?;

    let bytes = readback.read();
    let mut pixels: Vec<f32> =
        bytemuck::cast_slice(&bytes[..(size * size * 16) as usize]).to_vec();
    let pos: Vec<f32> =
        bytemuck::cast_slice(&pos_readback.read()[..(size * size * 16) as usize]).to_vec();
    let nrm: Vec<f32> =
        bytemuck::cast_slice(&nrm_readback.read()[..(size * size * 16) as usize]).to_vec();
    // Edge-aware denoise (smooth MC noise, keep shadow/geometry edges), then
    // dilate valid texels into the gutter so bilinear sampling at chart edges
    // never reads the black background (fixes lightmap seams).
    denoise_atrous(&mut pixels, &pos, &nrm, size, 4);
    // Merge co-located cross-chart texels (cube face-atlas / sphere meridian)
    // before dilation so the stitched values also bleed into the gutter.
    stitch_seams(&mut pixels, &pos, &nrm, size);
    dilate_lightmap(&mut pixels, size, 6);

    let mut alloc = allocator.lock().unwrap();
    let mut readback = readback;
    let mut pos_readback = pos_readback;
    let mut nrm_readback = nrm_readback;
    readback.destroy(device, &mut alloc);
    pos_readback.destroy(device, &mut alloc);
    nrm_readback.destroy(device, &mut alloc);
    g_pos.destroy(device, &mut alloc);
    g_normal.destroy(device, &mut alloc);
    out_img.destroy(device, &mut alloc);
    drop(alloc);
    unsafe { device.destroy_descriptor_pool(pool, None) };

    // Diagnostic: brightness range of the baked lightmap. A near-flat range
    // (min ≈ max) means the bake produced uniform light (no shadow/spot detail),
    // which distinguishes a bake problem from a resolution/display one.
    {
        let lum = |i: usize| 0.2126 * pixels[i] + 0.7152 * pixels[i + 1] + 0.0722 * pixels[i + 2];
        let (mut lo, mut hi, mut acc) = (f32::INFINITY, 0.0f32, 0.0f64);
        let n = (size * size) as usize;
        let mut covered = 0usize;
        for t in 0..n {
            if pixels[t * 4 + 3] <= 0.0 {
                continue;
            }
            covered += 1;
            let l = lum(t * 4);
            lo = lo.min(l);
            hi = hi.max(l);
            acc += l as f64;
        }
        // `covered` = texels the g-buffer actually rasterized (i.e. the object has
        // usable lightmap UVs). covered==0 → missing/degenerate lightmap UVs (the
        // object never gets light), distinct from covered>0 but dark (shadowed/no
        // light reaching). avg is over covered texels only.
        tracing::info!(
            "lightmap {size}²: {covered}/{n} covered, luminance min {:.3} max {:.3} avg {:.3}",
            if lo.is_finite() { lo } else { 0.0 },
            hi,
            acc / covered.max(1) as f64
        );
    }

    Ok(BakedLightmap { size, pixels })
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ProbePush {
    probe_count: u32,
    light_count: u32,
    sample_count: u32,
    frame_seed: u32,
    bounces: u32,
    sky_color: [f32; 4],
}

#[allow(clippy::too_many_arguments)]
fn bake_probes(
    ctx: &GpuContext,
    allocator: &Arc<Mutex<Allocator>>,
    command_pool: vk::CommandPool,
    tlas: &Accel,
    instance_ssbo: &Buffer,
    light_ssbo: &Buffer,
    light_count: u32,
    input: &BakeInput<'_>,
) -> Result<Vec<ProbeSh>> {
    let device = &ctx.device;
    let count = input.probes.len();

    // Probe positions as vec4 (scalar layout), output SH = 4×vec4 per probe.
    let positions: Vec<[f32; 4]> = input
        .probes
        .iter()
        .map(|p| [p.x, p.y, p.z, 0.0])
        .collect();
    let mut alloc = allocator.lock().unwrap();
    let probe_ssbo = host_ssbo(device, &mut alloc, bytes_of(&positions), "bake-probe-pos")?;
    let out_ssbo = Buffer::new(
        device,
        &mut alloc,
        (count * 4 * 16) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::GpuToCpu,
        "bake-probe-sh",
    )?;
    drop(alloc);

    let stage = vk::ShaderStageFlags::COMPUTE;
    let ssbo = |b: u32| {
        vk::DescriptorSetLayoutBinding::default()
            .binding(b)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(stage)
    };
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(1)
            .stage_flags(stage),
        ssbo(1),
        ssbo(2),
        ssbo(3),
        ssbo(4),
    ];
    let set_layout = unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )?
    };
    let set_layouts = [set_layout];
    let push = [vk::PushConstantRange::default()
        .stage_flags(stage)
        .size(size_of::<ProbePush>() as u32)];
    let layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&push),
            None,
        )?
    };
    let module = shader_module(device, PROBE_COMP)?;
    let pipeline = unsafe {
        device
            .create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    // Tiled probe dispatch (cmd_dispatch_base) so a huge probe bake
                    // (e.g. a dense FluxVoxel auto-grid) doesn't monopolize the GPU.
                    .flags(vk::PipelineCreateFlags::DISPATCH_BASE)
                    .stage(
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(stage)
                            .module(module)
                            .name(c"main"),
                    )
                    .layout(layout)],
                None,
            )
            .map_err(|(_, e)| e)?[0]
    };

    let pool_sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            descriptor_count: 1,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 4,
        },
    ];
    let pool = unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes),
            None,
        )?
    };
    let set = unsafe {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&set_layouts),
        )?[0]
    };
    write_tlas(device, set, 0, tlas.handle);
    write_ssbo(device, set, 1, instance_ssbo.handle);
    write_ssbo(device, set, 2, light_ssbo.handle);
    write_ssbo(device, set, 3, probe_ssbo.handle);
    write_ssbo(device, set, 4, out_ssbo.handle);

    let push_data = ProbePush {
        probe_count: count as u32,
        light_count,
        sample_count: input.samples.max(1),
        frame_seed: 1,
        bounces: input.bounces,
        sky_color: [input.sky_color[0], input.sky_color[1], input.sky_color[2], 0.0],
    };
    let groups = (count as u32).div_ceil(64);
    // Tile the probe path-trace into chunks, each its own throttled submission, so
    // a large probe set (a dense FluxVoxel auto-grid can be tens of thousands of
    // probes) can't hog the GPU in one dispatch and freeze the machine.
    let mut g0 = 0u32;
    while g0 < groups {
        let cg = PROBE_CHUNK_GROUPS.min(groups - g0);
        throttled_submit(device, command_pool, ctx.queue, input.gpu_idle_frac, |cb| unsafe {
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                &[set],
                &[],
            );
            device.cmd_push_constants(
                cb,
                layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes_of(std::slice::from_ref(&push_data)),
            );
            device.cmd_dispatch_base(cb, g0, 0, 0, cg, 1, 1);
        })?;
        g0 += cg;
    }

    let bytes = out_ssbo.read();
    let raw: &[f32] = bytemuck::cast_slice(&bytes[..(count * 4 * 4 * 4).min(bytes.len())]);
    let mut probes = Vec::with_capacity(count);
    for p in 0..count {
        let base = p * 16; // 4 coeffs × 4 floats
        let mut sh = ProbeSh::default();
        for c in 0..4 {
            sh.coeffs[c] = [raw[base + c * 4], raw[base + c * 4 + 1], raw[base + c * 4 + 2]];
        }
        probes.push(sh);
    }

    let mut alloc = allocator.lock().unwrap();
    let mut probe_ssbo = probe_ssbo;
    let mut out_ssbo = out_ssbo;
    probe_ssbo.destroy(device, &mut alloc);
    out_ssbo.destroy(device, &mut alloc);
    drop(alloc);
    unsafe {
        device.destroy_descriptor_pool(pool, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(layout, None);
        device.destroy_descriptor_set_layout(set_layout, None);
        device.destroy_shader_module(module, None);
    }
    Ok(probes)
}
