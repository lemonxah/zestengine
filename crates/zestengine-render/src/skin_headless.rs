//! Headless driver for the GPU skinning / blendshape / cloth-override passes
//! (`skin.comp`, `morph.comp`, `cloth_override.comp`) on a real Vulkan device,
//! plus the CPU references they are compared with. It runs the renderer's own
//! `GpuSkinner` (descriptor sets, push constants, barriers, per-frame slots)
//! frame by frame and reads the skinned vertex buffer and the previous-frame
//! position stream back. Used by this crate's tests and by the avatar test in
//! `zestengine-assets`; like the cluster harness it returns nothing when the
//! machine has no Vulkan 1.3 device.

use anyhow::Result;
use ash::vk;
use glam::{Mat4, Vec3};
use gpu_allocator::MemoryLocation;

use crate::alloc::{self, Buffer};
use crate::cluster_headless::HeadlessGpu;
use crate::frame::FRAMES_IN_FLIGHT;
use crate::morph::{MorphDeltas, MorphTables};
use crate::skin::GpuSkinner;
use crate::types::Vertex;

/// What one simulated frame feeds the skinner.
#[derive(Clone, Debug, Default)]
pub struct SkinFrame {
    pub palette: Vec<Mat4>,
    /// Blendshape weights (`None` = not queued this frame).
    pub morph: Option<Vec<f32>>,
    /// Mesh Cloth overrides as packed by [`crate::pack_cloth_patch`] and the entry count.
    pub cloth: Option<(Vec<f32>, u32)>,
    /// Skip queueing the palette (the mesh is not skinned this frame).
    pub skip_skin: bool,
    /// The skinned cluster path draws the mesh this frame: its dispatch is dropped (`record`'s skip list).
    pub cluster_skip: bool,
}

/// What the frame left behind.
#[derive(Clone, Debug)]
pub struct SkinFrameOut {
    /// The mesh's vertex buffer after the passes.
    pub vertices: Vec<Vertex>,
    /// The previous-frame position stream (xyz per vertex).
    pub prev: Vec<[f32; 3]>,
    /// Errors the validation layer reported during the frame (0 without it).
    pub validation_errors: u32,
}

/// CPU twin of `skin.comp`: linear blend skinning with the weight sum
/// renormalised, joints past the palette ignored, normal/tangent rotated by the
/// blended matrix's upper 3x3 and renormalised (the bind normal when the matrix
/// collapses); weightless vertices unchanged.
pub fn skin_reference(base: &[Vertex], palette: &[Mat4]) -> Vec<Vertex> {
    base.iter()
        .map(|v| {
            let mut out = *v;
            let sum: f32 = v.weights.iter().sum();
            if sum <= 1e-5 {
                return out;
            }
            let mut m = Mat4::ZERO;
            for k in 0..4 {
                let j = v.joints[k] as usize;
                if v.weights[k] > 0.0 && j < palette.len() {
                    m += palette[j] * v.weights[k];
                }
            }
            m *= 1.0 / sum;
            let p = m.transform_point3(Vec3::from(v.position));
            let r = glam::Mat3::from_mat4(m);
            let n_in = Vec3::from(v.normal);
            let n = r * n_in;
            let n = if n.dot(n) > 1e-12 { n.normalize() } else { n_in };
            let t = r * Vec3::from_slice(&v.tangent[..3]);
            let t = if t.length() > 1e-6 { t.normalize() } else { t };
            out.position = p.to_array();
            out.normal = n.to_array();
            out.tangent = [t.x, t.y, t.z, v.tangent[3]];
            out
        })
        .collect()
}

/// Largest absolute difference over position, normal and tangent xyz between
/// two vertex lists (NaN counts as infinite).
pub fn max_vertex_error(a: &[Vertex], b: &[Vertex]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut worst = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        let pairs = (0..3)
            .map(|k| (x.position[k], y.position[k]))
            .chain((0..3).map(|k| (x.normal[k], y.normal[k])))
            .chain((0..3).map(|k| (x.tangent[k], y.tangent[k])));
        for (p, q) in pairs {
            let d = (p - q).abs();
            worst = worst.max(if d.is_nan() { f32::INFINITY } else { d });
        }
    }
    worst
}

impl HeadlessGpu {
    /// Register `base` as a GPU-skinned mesh (with optional blendshapes), run
    /// one skinner `record` per `frames` entry (cycling the frame slots like the
    /// renderer does) and read back after each.
    pub fn skin_frames(
        &self,
        base: &[Vertex],
        joint_count: usize,
        shapes: &[MorphDeltas<'_>],
        frames: &[SkinFrame],
    ) -> Result<Vec<SkinFrameOut>> {
        let device = &self.device;
        let mut alloc = self.allocator()?;
        let mut skinner = GpuSkinner::new(device)?;
        let bytes: &[u8] = bytemuck::cast_slice(base);
        // The vertex buffer additionally gets TRANSFER_SRC so the test can copy it out.
        let vertex_buffer = alloc::upload_buffer(
            device,
            &mut alloc,
            self.pool,
            self.queue,
            bytes,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            "test skinned vertices",
        )?;
        let base_buf = alloc::upload_buffer(
            device,
            &mut alloc,
            self.pool,
            self.queue,
            bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "test skin bind pose",
        )?;
        let morph = (!shapes.is_empty()).then(|| MorphTables::build(base, shapes)).filter(|m| !m.is_empty());
        skinner.add(device, &mut alloc, 0, vertex_buffer.handle, base_buf, base.len() as u32, joint_count, morph)?;
        let mut host = Buffer::new(
            device,
            &mut alloc,
            bytes.len().max(4) as u64,
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuToCpu,
            "test skin readback",
        )?;
        let mut results = Vec::new();
        for (i, f) in frames.iter().enumerate() {
            let errors_before = crate::cluster_headless::VALIDATION_ERRORS.load(std::sync::atomic::Ordering::Relaxed);
            if !f.skip_skin {
                skinner.queue(0, &f.palette);
            }
            if let Some(w) = &f.morph {
                skinner.queue_morph(0, w);
            }
            if let Some((packed, count)) = &f.cloth {
                skinner.queue_cloth(device, &mut alloc, 0, packed, *count)?;
            }
            alloc::one_time_submit(device, self.pool, self.queue, |cb| unsafe {
                skinner.record(device, cb, i % FRAMES_IN_FLIGHT, None, if f.cluster_skip { &[0] } else { &[] });
                // Make the compute writes visible to the transfer, then to the host.
                let b = [vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)];
                device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&b));
                device.cmd_copy_buffer(
                    cb,
                    vertex_buffer.handle,
                    host.handle,
                    &[vk::BufferCopy::default().size(bytes.len() as u64)],
                );
                let b = [vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                    .dst_access_mask(vk::AccessFlags2::HOST_READ)];
                device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&b));
            })?;
            let vertices: Vec<Vertex> = bytemuck::pod_collect_to_vec(&host.mapped().unwrap_or(&[])[..bytes.len()]);
            let prev_bytes = skinner.prev_positions(0);
            let prev_v: Vec<Vertex> = bytemuck::pod_collect_to_vec(&prev_bytes[..bytes.len()]);
            results.push(SkinFrameOut {
                vertices,
                prev: prev_v.iter().map(|v| v.position).collect(),
                validation_errors: crate::cluster_headless::VALIDATION_ERRORS.load(std::sync::atomic::Ordering::Relaxed)
                    - errors_before,
            });
        }
        skinner.destroy(device, &mut alloc);
        host.destroy(device, &mut alloc);
        let mut vb = vertex_buffer;
        vb.destroy(device, &mut alloc);
        Ok(results)
    }
}

#[cfg(test)]
impl HeadlessGpu {
    /// Run the renderer's skin (and, when `morph`, blendshape) passes over
    /// `meshes` meshes of `verts` vertices inside a command buffer bracketed
    /// like a renderer frame (pool reset, FRAME zone stamps), and return the raw
    /// timestamp query results (`frame::TS_COUNT` of them, read back with
    /// availability exactly like `Renderer::render`) plus the wall time of the
    /// submit in ms.
    pub(crate) fn skin_timed(&self, verts: u32, meshes: usize, morph: bool) -> Result<(Vec<crate::TsResult>, f64)> {
        let device = &self.device;
        let mut alloc = self.allocator()?;
        let mut skinner = GpuSkinner::new(device)?;
        let base: Vec<Vertex> = (0..verts)
            .map(|i| Vertex {
                position: [i as f32 * 1e-3, 0.0, 0.0],
                joints: [0, 1, 0, 0],
                weights: [0.5, 0.5, 0.0, 0.0],
                ..Default::default()
            })
            .collect();
        let bytes: &[u8] = bytemuck::cast_slice(&base);
        let shape: Vec<(u32, [f32; 3], [f32; 3])> = (0..verts).map(|v| (v, [0.0, 0.01, 0.0], [0.0; 3])).collect();
        let mut outs = Vec::new();
        for m in 0..meshes {
            let out = alloc::upload_buffer(device, &mut alloc, self.pool, self.queue, bytes, vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER, "timed vertices")?;
            let b = alloc::upload_buffer(device, &mut alloc, self.pool, self.queue, bytes, vk::BufferUsageFlags::STORAGE_BUFFER, "timed bind")?;
            let tables = morph.then(|| MorphTables::build(&base, &[shape.as_slice()]));
            skinner.add(device, &mut alloc, m, out.handle, b, verts, 2, tables)?;
            skinner.queue(m, &[Mat4::from_rotation_z(0.3), Mat4::from_rotation_y(1.0)]);
            if morph {
                skinner.queue_morph(m, &[1.0]);
            }
            outs.push(out);
        }
        let pool = unsafe {
            device.create_query_pool(
                &vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(crate::frame::TS_COUNT),
                None,
            )?
        };
        let started = std::time::Instant::now();
        alloc::one_time_submit(device, self.pool, self.queue, |cb| unsafe {
            device.cmd_reset_query_pool(cb, pool, 0, crate::frame::TS_COUNT);
            let stamp = |slot: u32| device.cmd_write_timestamp2(cb, vk::PipelineStageFlags2::ALL_COMMANDS, pool, slot);
            stamp(2 * crate::gpu_zone::FRAME as u32);
            skinner.record(device, cb, 0, Some(pool), &[]);
            stamp(2 * crate::gpu_zone::FRAME as u32 + 1);
        })?;
        let wall_ms = started.elapsed().as_secs_f64() * 1e3;
        let mut res = vec![crate::TsResult::default(); crate::frame::TS_COUNT as usize];
        let r = unsafe {
            device.get_query_pool_results(pool, 0, &mut res, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY)
        };
        // NOT_READY is expected: zones this frame never wrote stay unavailable.
        assert!(matches!(r, Ok(()) | Err(vk::Result::NOT_READY)), "{r:?}");
        unsafe { device.destroy_query_pool(pool, None) };
        skinner.destroy(device, &mut alloc);
        for mut o in outs {
            o.destroy(device, &mut alloc);
        }
        Ok((res, wall_ms))
    }
}
