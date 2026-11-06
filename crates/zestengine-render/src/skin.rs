//! GPU skinning: a compute pass that writes skinned vertices straight into a
//! mesh's vertex buffer, so every other pass (shadows, main, planar, VR) draws
//! it unchanged. Palettes are queued on the CPU and uploaded into the current
//! frame slot's buffer after that slot's fence has retired.
//!
//! Each mesh is skinned once per frame. The same dispatch hands the vertex
//! buffer's previous contents (last frame's skinned position) to a second
//! stream the motion-vector pass reads, so skin motion costs no extra pass.
//! Blendshapes run first as a sparse compute pass over only the vertices a
//! shape touches (see `morph.rs`), rewriting the bind-pose buffer skinning
//! reads.

use std::collections::HashMap;

use anyhow::Result;
use ash::vk;
use glam::Mat4;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocator;

use crate::alloc::Buffer;
use crate::frame::FRAMES_IN_FLIGHT;
use crate::morph::MorphTables;
use crate::types::Vertex;

// skin.comp hard-codes the vertex layout (26 floats, joints at 18, weights at 22).
const _: () = assert!(std::mem::size_of::<Vertex>() == 104);

const SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/skin.comp.spv"));
const MORPH_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/morph.comp.spv"));
const CLOTH_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cloth_override.comp.spv"));

/// Floats per cloth patch entry: vertex index (u32 bits), position, normal, pad.
pub const CLOTH_PATCH_FLOATS: usize = 8;

/// Select the cloth-affected vertices and pack them for `cloth_override.comp`.
///
/// `animated` is the pose the mesh would have without cloth (skinned and
/// morphed), `simulated` the same vertices after the cloth step. Vertices whose
/// position or normal moved by more than `eps` are cloth-driven and get an
/// entry; pinned vertices (unchanged) stay on plain GPU skinning. `out` is
/// cleared and reused (no per-frame allocation once warm). Returns the entry
/// count.
pub fn pack_cloth_patch(animated: &[Vertex], simulated: &[Vertex], eps: f32, out: &mut Vec<f32>) -> u32 {
    out.clear();
    let mut count = 0;
    for (i, (a, s)) in animated.iter().zip(simulated).enumerate() {
        let moved = (0..3).any(|k| (a.position[k] - s.position[k]).abs() > eps || (a.normal[k] - s.normal[k]).abs() > eps);
        if !moved {
            continue;
        }
        out.push(f32::from_bits(i as u32));
        out.extend_from_slice(&s.position);
        out.extend_from_slice(&s.normal);
        out.push(0.0);
        count += 1;
    }
    count
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Push {
    vertex_count: u32,
    stride: u32,
    joint_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MorphPush {
    touched_count: u32,
    stride: u32,
    shape_count: u32,
}

fn mat_floats(m: &[Mat4]) -> Vec<f32> {
    m.iter().flat_map(|m| m.to_cols_array()).collect()
}

/// Per-mesh blendshape state: the packed sparse deltas, a weights buffer per
/// frame slot, and the weights the bind-pose buffer currently reflects.
struct GpuMorph {
    touched: Buffer,
    deltas: Buffer,
    weights: Vec<Buffer>,
    sets: Vec<vk::DescriptorSet>,
    touched_count: u32,
    shape_count: u32,
    /// Weights last applied to the bind-pose buffer (all zero at upload).
    applied: Vec<f32>,
}

/// Lazily created per-mesh Mesh Cloth patch buffers (one per frame slot) and
/// the descriptor sets binding them to the mesh's output vertex buffer.
struct GpuCloth {
    patches: Vec<Buffer>,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
}

/// The cloth patch queued for a mesh this frame (buffers reused every frame).
#[derive(Default)]
struct ClothPending {
    data: Vec<f32>,
    count: u32,
}

pub(crate) struct GpuSkin {
    /// The mesh's vertex buffer this pass rewrites.
    out: vk::Buffer,
    cloth: Option<GpuCloth>,
    base: Buffer,
    palettes: Vec<Buffer>,
    /// Previous frame's skinned positions, written by the skin pass from the
    /// vertex buffer's old contents and bound as the second vertex stream so
    /// the motion-vector pass reprojects skin motion.
    prev_out: Buffer,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    morph: Option<GpuMorph>,
    vertex_count: u32,
    joint_count: u32,
}

impl GpuSkin {
    pub(crate) fn destroy(mut self, device: &ash::Device, allocator: &mut Allocator) {
        self.base.destroy(device, allocator);
        self.prev_out.destroy(device, allocator);
        if let Some(mut c) = self.cloth.take() {
            for mut p in c.patches.drain(..) {
                p.destroy(device, allocator);
            }
            unsafe { device.destroy_descriptor_pool(c.pool, None) };
        }
        for mut p in self.palettes.drain(..) {
            p.destroy(device, allocator);
        }
        if let Some(mut m) = self.morph.take() {
            m.touched.destroy(device, allocator);
            m.deltas.destroy(device, allocator);
            for mut w in m.weights.drain(..) {
                w.destroy(device, allocator);
            }
        }
        unsafe { device.destroy_descriptor_pool(self.pool, None) };
    }
}

pub(crate) struct GpuSkinner {
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    morph_set_layout: vk::DescriptorSetLayout,
    morph_pipeline_layout: vk::PipelineLayout,
    morph_pipeline: vk::Pipeline,
    cloth_set_layout: vk::DescriptorSetLayout,
    cloth_pipeline_layout: vk::PipelineLayout,
    cloth_pipeline: vk::Pipeline,
    /// Keyed by mesh index.
    skins: HashMap<usize, GpuSkin>,
    pending: Vec<(usize, Vec<Mat4>)>,
    /// Index of each mesh's entry in `pending` (a mesh queued twice in a frame replaces its palette; a
    /// linear search here was quadratic in the frame's skinned meshes).
    pending_at: HashMap<usize, usize>,
    /// Palette vectors of drained `pending` entries, reused by `queue` (no per-mesh allocation per frame).
    spare: Vec<Vec<Mat4>>,
    pending_morph: Vec<(usize, Vec<f32>)>,
    pending_cloth: HashMap<usize, ClothPending>,
    /// Latest pose per mesh (kept after `record` drains `pending`) for the
    /// skinned cluster path: this and last frame's palette plus blendshape weights.
    poses: HashMap<usize, MeshPose>,
    /// Meshes whose ordinary skinned vertex buffer lags their latest pose: the
    /// pose was queued in a frame the cluster path skipped the dispatch. The
    /// engine sends a pose that did not change only once more, so the buffer is
    /// skinned from `poses` the first frame something reads it again.
    stale: std::collections::HashSet<usize>,
    /// Rendered frames so far (`record` calls); stamps the blendshape weights.
    frame_no: u64,
}

/// A mesh's latest pose (GEO-5): buffers are reused frame to frame.
#[derive(Default)]
pub(crate) struct MeshPose {
    pub cur: Vec<Mat4>,
    pub prev: Vec<Mat4>,
    pub weights: Vec<f32>,
    /// The weights of the frame before (equal to `weights` when the mesh was not
    /// morphed last frame), so the velocity pass sees blendshape motion.
    pub prev_weights: Vec<f32>,
    /// The frame (`GpuSkinner::frame_no`) the weights were last queued in.
    weights_frame: Option<u64>,
}

impl MeshPose {
    /// Record the weights queued in `frame`: they become current, the previous
    /// call's current weights become `prev_weights` when that call was the
    /// frame before (otherwise there is no motion: prev = current). A second
    /// call in the same frame replaces the current weights only.
    pub(crate) fn queue_weights(&mut self, weights: &[f32], frame: u64) {
        match self.weights_frame {
            Some(f) if f == frame => {}
            Some(f) if f + 1 == frame => std::mem::swap(&mut self.prev_weights, &mut self.weights),
            _ => {
                self.prev_weights.clear();
                self.prev_weights.extend_from_slice(weights);
            }
        }
        self.weights.clear();
        self.weights.extend_from_slice(weights);
        self.weights_frame = Some(frame);
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ClothPush {
    count: u32,
    stride: u32,
    vertex_count: u32,
}

fn storage_bindings(count: u32) -> Vec<vk::DescriptorSetLayoutBinding<'static>> {
    (0..count)
        .map(|b| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect()
}

fn compute_pipeline(
    device: &ash::Device,
    spv: &[u8],
    set_layout: vk::DescriptorSetLayout,
    push_size: u32,
) -> Result<(vk::PipelineLayout, vk::Pipeline)> {
    let layouts = [set_layout];
    let ranges = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(push_size)];
    let pipeline_layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&ranges),
            None,
        )?
    };
    let module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&ash::util::read_spv(&mut std::io::Cursor::new(spv))?),
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
    Ok((pipeline_layout, pipeline))
}

fn write_storage_set(device: &ash::Device, set: vk::DescriptorSet, buffers: &[vk::Buffer]) {
    let infos: Vec<[vk::DescriptorBufferInfo; 1]> =
        buffers.iter().map(|&b| [vk::DescriptorBufferInfo::default().buffer(b).range(vk::WHOLE_SIZE)]).collect();
    let writes: Vec<_> = infos
        .iter()
        .enumerate()
        .map(|(b, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(b as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(info)
        })
        .collect();
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

impl GpuSkinner {
    pub fn new(device: &ash::Device) -> Result<Self> {
        let set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&storage_bindings(4)),
                None,
            )?
        };
        let (pipeline_layout, pipeline) = compute_pipeline(device, SPV, set_layout, std::mem::size_of::<Push>() as u32)?;
        let morph_set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&storage_bindings(4)),
                None,
            )?
        };
        let (morph_pipeline_layout, morph_pipeline) =
            compute_pipeline(device, MORPH_SPV, morph_set_layout, std::mem::size_of::<MorphPush>() as u32)?;
        let cloth_set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&storage_bindings(2)),
                None,
            )?
        };
        let (cloth_pipeline_layout, cloth_pipeline) =
            compute_pipeline(device, CLOTH_SPV, cloth_set_layout, std::mem::size_of::<ClothPush>() as u32)?;
        Ok(Self {
            cloth_set_layout,
            cloth_pipeline_layout,
            cloth_pipeline,
            pending_cloth: HashMap::new(),
            poses: HashMap::new(),
            set_layout,
            pipeline_layout,
            pipeline,
            morph_set_layout,
            morph_pipeline_layout,
            morph_pipeline,
            skins: HashMap::new(),
            pending: Vec::new(),
            pending_at: HashMap::new(),
            spare: Vec::new(),
            pending_morph: Vec::new(),
            stale: Default::default(),
            frame_no: 0,
        })
    }

    /// Register a mesh whose vertex buffer (`out`, STORAGE usage) is rewritten
    /// by the skinning pass from `base` (device-local copy of the bind pose).
    /// `morph` carries the mesh's blendshape deltas, applied to `base` in place
    /// by a compute pass before skinning (see [`Self::queue_morph`]).
    #[allow(clippy::too_many_arguments)]
    pub fn add(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        mesh: usize,
        out: vk::Buffer,
        base: Buffer,
        vertex_count: u32,
        joint_count: usize,
        morph: Option<MorphTables>,
    ) -> Result<()> {
        let joint_count = joint_count.max(1);
        let size = (joint_count * std::mem::size_of::<Mat4>()) as u64;
        let mut palettes = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for _ in 0..FRAMES_IN_FLIGHT {
            let mut b = Buffer::new(
                device,
                allocator,
                size,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::CpuToGpu,
                "skin palette",
            )?;
            b.write(0, bytemuck::cast_slice(&mat_floats(&vec![Mat4::IDENTITY; joint_count])));
            palettes.push(b);
        }
        // Previous-frame positions for the motion-vector pass. Starts as zeros;
        // the first skin dispatch overwrites it from the bind-pose upload.
        let prev_out = Buffer::new(
            device,
            allocator,
            vertex_count as u64 * std::mem::size_of::<Vertex>() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER,
            MemoryLocation::CpuToGpu,
            "prev skin vertices",
        )?;
        let morph = morph.filter(|m| !m.is_empty());
        let skin_sets = FRAMES_IN_FLIGHT as u32;
        let morph_sets = if morph.is_some() { FRAMES_IN_FLIGHT as u32 } else { 0 };
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(skin_sets + morph_sets)
                    .pool_sizes(&[vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::STORAGE_BUFFER,
                        descriptor_count: 4 * (skin_sets + morph_sets),
                    }]),
                None,
            )?
        };
        let layouts = vec![self.set_layout; FRAMES_IN_FLIGHT];
        let sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts),
            )?
        };
        for (f, &set) in sets.iter().enumerate() {
            write_storage_set(device, set, &[base.handle, out, palettes[f].handle, prev_out.handle]);
        }
        let morph = match morph {
            Some(tables) => {
                let shapes = tables.shape_count.max(1) as u64;
                let mut touched = Buffer::new(
                    device,
                    allocator,
                    (tables.touched.len() * 4) as u64,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    MemoryLocation::CpuToGpu,
                    "morph touched",
                )?;
                touched.write(0, bytemuck::cast_slice(&tables.touched));
                let mut deltas = Buffer::new(
                    device,
                    allocator,
                    (tables.deltas.len() * 4).max(4) as u64,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    MemoryLocation::CpuToGpu,
                    "morph deltas",
                )?;
                deltas.write(0, bytemuck::cast_slice(&tables.deltas));
                let mut weights = Vec::with_capacity(FRAMES_IN_FLIGHT);
                for _ in 0..FRAMES_IN_FLIGHT {
                    let mut w = Buffer::new(
                        device,
                        allocator,
                        shapes * 4,
                        vk::BufferUsageFlags::STORAGE_BUFFER,
                        MemoryLocation::CpuToGpu,
                        "morph weights",
                    )?;
                    w.write(0, bytemuck::cast_slice(&vec![0.0f32; shapes as usize]));
                    weights.push(w);
                }
                let layouts = vec![self.morph_set_layout; FRAMES_IN_FLIGHT];
                let msets = unsafe {
                    device.allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts),
                    )?
                };
                for (f, &set) in msets.iter().enumerate() {
                    write_storage_set(device, set, &[base.handle, touched.handle, deltas.handle, weights[f].handle]);
                }
                Some(GpuMorph {
                    touched,
                    deltas,
                    weights,
                    sets: msets,
                    touched_count: tables.touched_count,
                    shape_count: tables.shape_count,
                    applied: vec![0.0; tables.shape_count as usize],
                })
            }
            None => None,
        };
        self.skins.insert(
            mesh,
            GpuSkin {
                out,
                cloth: None,
                base,
                palettes,
                prev_out,
                pool,
                sets,
                morph,
                vertex_count,
                joint_count: joint_count as u32,
            },
        );
        Ok(())
    }

    /// The previous frame's skinned vertex buffer for a mesh, if it is skinned
    /// (bound as the motion-vector pass's second position stream).
    pub fn prev_vertex_buffer(&self, mesh: usize) -> Option<vk::Buffer> {
        self.skins.get(&mesh).map(|s| s.prev_out.handle)
    }

    /// Host copy of a mesh's previous-frame vertex stream (the GPU tests read it).
    pub(crate) fn prev_positions(&self, mesh: usize) -> Vec<u8> {
        self.skins.get(&mesh).map(|s| s.prev_out.read()).unwrap_or_default()
    }

    /// Queue this frame's pose for a mesh (applied by the next `record`).
    pub fn queue(&mut self, mesh: usize, palette: &[Mat4]) {
        if self.skins.contains_key(&mesh) {
            match self.pending_at.get(&mesh) {
                Some(&k) => {
                    let v = &mut self.pending[k].1;
                    v.clear();
                    v.extend_from_slice(palette);
                }
                None => {
                    let mut v = self.spare.pop().unwrap_or_default();
                    v.clear();
                    v.extend_from_slice(palette);
                    self.pending_at.insert(mesh, self.pending.len());
                    self.pending.push((mesh, v));
                }
            }
            let p = self.poses.entry(mesh).or_default();
            std::mem::swap(&mut p.cur, &mut p.prev);
            p.cur.clear();
            p.cur.extend_from_slice(palette);
            if p.prev.is_empty() {
                p.prev.extend_from_slice(palette);
            }
        }
    }

    /// Record a pose for a mesh without a GPU skin (a CPU-skinned shared mesh
    /// the skinned cluster path draws); a GPU-skinned mesh is queued as usual.
    pub fn set_pose(&mut self, mesh: usize, palette: &[Mat4]) {
        if self.skins.contains_key(&mesh) {
            self.queue(mesh, palette);
            return;
        }
        let p = self.poses.entry(mesh).or_default();
        std::mem::swap(&mut p.cur, &mut p.prev);
        p.cur.clear();
        p.cur.extend_from_slice(palette);
        if p.prev.is_empty() {
            p.prev.extend_from_slice(palette);
        }
    }

    /// Mesh Cloth overrides are queued for `mesh` this frame (the cluster path
    /// cannot draw them: the ordinary skinned buffer carries the cloth).
    pub(crate) fn cloth_pending(&self, mesh: usize) -> bool {
        self.pending_cloth.get(&mesh).is_some_and(|p| p.count > 0)
    }

    /// The latest pose queued for a mesh (skinned cluster path).
    pub(crate) fn pose_of(&self, mesh: usize) -> Option<&MeshPose> {
        self.poses.get(&mesh).filter(|p| !p.cur.is_empty())
    }

    /// Last frame's blendshape weights for a mesh's pose: only when the weights
    /// were queued this frame and the frame before (a mesh that stopped being
    /// morphed has no motion, whatever its stale weights say).
    pub(crate) fn prev_weights_of<'a>(&self, pose: &'a MeshPose) -> Option<&'a [f32]> {
        (pose.weights_frame == Some(self.frame_no)).then_some(pose.prev_weights.as_slice())
    }

    /// Queue this frame's Mesh Cloth overrides for a GPU-skinned mesh: `packed`
    /// holds `count` entries from [`pack_cloth_patch`]. The cloth-driven
    /// vertices bypass skinning (an override pass after the skin dispatch
    /// rewrites them); an empty patch leaves the mesh purely skinned. The
    /// per-mesh patch buffers are created on first use.
    pub fn queue_cloth(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        mesh: usize,
        packed: &[f32],
        count: u32,
    ) -> Result<()> {
        let Some(skin) = self.skins.get_mut(&mesh) else { return Ok(()) };
        if count == 0 && skin.cloth.is_none() {
            return Ok(());
        }
        if skin.cloth.is_none() {
            // Worst case every vertex is cloth-driven.
            let bytes = (skin.vertex_count as usize * CLOTH_PATCH_FLOATS * 4).max(16) as u64;
            let mut patches = Vec::with_capacity(FRAMES_IN_FLIGHT);
            for _ in 0..FRAMES_IN_FLIGHT {
                patches.push(Buffer::new(
                    device,
                    allocator,
                    bytes,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    MemoryLocation::CpuToGpu,
                    "cloth patch",
                )?);
            }
            let pool = unsafe {
                device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default().max_sets(FRAMES_IN_FLIGHT as u32).pool_sizes(&[
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 2 * FRAMES_IN_FLIGHT as u32,
                        },
                    ]),
                    None,
                )?
            };
            let layouts = vec![self.cloth_set_layout; FRAMES_IN_FLIGHT];
            let sets = unsafe {
                device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts),
                )?
            };
            for (f, &set) in sets.iter().enumerate() {
                write_storage_set(device, set, &[skin.out, patches[f].handle]);
            }
            skin.cloth = Some(GpuCloth { patches, pool, sets });
        }
        let cap = skin.vertex_count as usize * CLOTH_PATCH_FLOATS;
        let n = packed.len().min(cap);
        let p = self.pending_cloth.entry(mesh).or_default();
        p.data.clear();
        p.data.extend_from_slice(&packed[..n]);
        p.count = (n / CLOTH_PATCH_FLOATS) as u32;
        Ok(())
    }

    /// Queue blendshape weights for a mesh registered with morph tables. The
    /// morph pass only dispatches when they differ from the last applied set.
    pub fn queue_morph(&mut self, mesh: usize, weights: &[f32]) {
        if self.skins.get(&mesh).is_some_and(|s| s.morph.is_some()) {
            self.pending_morph.retain(|(m, _)| *m != mesh);
            self.pending_morph.push((mesh, weights.to_vec()));
            let frame = self.frame_no;
            let p = self.poses.entry(mesh).or_default();
            p.queue_weights(weights, frame);
        }
    }

    /// Upload queued palettes into frame slot `frame` and dispatch, followed by
    /// a barrier so vertex fetch sees the results. Call after the slot's fence.
    ///
    /// `ts` is the frame's timestamp pool when zone timing is active: the morph
    /// and skin dispatches are bracketed by the `MORPH` / `SKIN` zones (a zone
    /// with no dispatch this frame stays unwritten and reads 0).
    ///
    /// `skip`: meshes whose skinned vertex buffer nothing reads this frame (the
    /// cluster path draws them, shadows included, from its own posed pool): their
    /// palette dispatch is dropped. Blendshapes and cloth are never skipped.
    pub fn record(&mut self, device: &ash::Device, cb: vk::CommandBuffer, frame: usize, ts: Option<vk::QueryPool>, skip: &[usize]) {
        self.record_passes(device, cb, frame, ts, skip);
        // One `record` per rendered frame: the poses' notion of "last frame".
        self.frame_no += 1;
    }

    fn record_passes(&mut self, device: &ash::Device, cb: vk::CommandBuffer, frame: usize, ts: Option<vk::QueryPool>, skip: &[usize]) {
        // `skip` can hold every cluster-drawn skinned mesh of a crowd: a set keeps the loops linear.
        let skip: std::collections::HashSet<usize> = skip.iter().copied().collect();
        if self.pending.is_empty()
            && self.pending_morph.is_empty()
            && self.pending_cloth.values().all(|p| p.count == 0)
            && self.stale.iter().all(|m| skip.contains(m))
        {
            return;
        }
        let stamp = |zone: usize, end: bool| {
            if let Some(pool) = ts {
                unsafe {
                    device.cmd_write_timestamp2(cb, vk::PipelineStageFlags2::ALL_COMMANDS, pool, (2 * zone + end as usize) as u32);
                }
            }
        };
        let stride = (std::mem::size_of::<Vertex>() / 4) as u32;
        // Each mesh has one skinned vertex buffer shared by every frame in
        // flight: the previous frame's prepass, shadow, main and RT passes must
        // finish reading it before this frame rewrites it, or the depth-EQUAL
        // main pass sees different vertices than its prepass and drops them.
        let war = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)];
        unsafe {
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&war));
        }

        // Blendshapes first: sparse deltas into the bind-pose buffer.
        let mut morphed = false;
        for (mesh, weights) in self.pending_morph.drain(..) {
            let Some(morph) = self.skins.get_mut(&mesh).and_then(|s| s.morph.as_mut()) else {
                continue;
            };
            let n = morph.shape_count as usize;
            let mut w = weights;
            w.resize(n, 0.0);
            if w == morph.applied {
                continue;
            }
            if !morphed {
                stamp(crate::gpu_zone::MORPH, false);
                unsafe { device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.morph_pipeline) };
                morphed = true;
            }
            morph.weights[frame].write(0, bytemuck::cast_slice(&w));
            let push = MorphPush { touched_count: morph.touched_count, stride, shape_count: morph.shape_count };
            unsafe {
                device.cmd_bind_descriptor_sets(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    self.morph_pipeline_layout,
                    0,
                    &[morph.sets[frame]],
                    &[],
                );
                device.cmd_push_constants(
                    cb,
                    self.morph_pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::bytes_of(&push),
                );
                device.cmd_dispatch(cb, morph.touched_count.div_ceil(64), 1, 1);
            }
            morph.applied = w;
        }
        if morphed {
            stamp(crate::gpu_zone::MORPH, true);
            let b = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)];
            unsafe { device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&b)) };
        }

        // A mesh whose dispatch an earlier frame skipped and that was not sent a new pose
        // since: skin its latest pose now that its ordinary buffer is read again.
        requeue_stale(&self.stale, &self.poses, &self.pending_at, &skip, &mut self.pending, &mut self.spare);
        self.pending_at.clear();
        let skinning = self.pending.iter().any(|(m, _)| !skip.contains(m));
        if skinning {
            stamp(crate::gpu_zone::SKIN, false);
            unsafe { device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline) };
        }
        let mut pending = std::mem::take(&mut self.pending);
        for (mesh, palette) in pending.drain(..) {
            if skip.contains(&mesh) {
                self.stale.insert(mesh);
                self.spare.push(palette);
                continue;
            }
            self.stale.remove(&mesh);
            let Some(skin) = self.skins.get_mut(&mesh) else {
                self.spare.push(palette);
                continue;
            };
            let n = palette.len().min(skin.joint_count as usize);
            skin.palettes[frame].write(0, bytemuck::cast_slice(&mat_floats(&palette[..n])));
            let push = Push {
                vertex_count: skin.vertex_count,
                stride,
                joint_count: n as u32,
            };
            // One dispatch per mesh per frame: skins the current pose and moves
            // last frame's skinned position into the motion-vector stream.
            unsafe {
                device.cmd_bind_descriptor_sets(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    &[skin.sets[frame]],
                    &[],
                );
                device.cmd_push_constants(
                    cb,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::bytes_of(&push),
                );
                device.cmd_dispatch(cb, skin.vertex_count.div_ceil(64), 1, 1);
            }
            self.spare.push(palette);
        }
        // Keep the (now empty) list's capacity for the next frame.
        self.pending = pending;
        // Mesh Cloth: overwrite the cloth-driven vertices of the meshes just
        // skinned (barrier first: the override reads/writes what skinning wrote).
        let mut cloth_bound = false;
        for (mesh, pending) in self.pending_cloth.iter_mut() {
            if pending.count == 0 {
                continue;
            }
            let Some(skin) = self.skins.get_mut(mesh) else { continue };
            let vertex_count = skin.vertex_count;
            let Some(cloth) = skin.cloth.as_mut() else { continue };
            if !cloth_bound {
                let b = [vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE)];
                unsafe {
                    device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&b));
                    device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.cloth_pipeline);
                }
                cloth_bound = true;
            }
            cloth.patches[frame].write(0, bytemuck::cast_slice(&pending.data));
            let push = ClothPush { count: pending.count, stride, vertex_count };
            unsafe {
                device.cmd_bind_descriptor_sets(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    self.cloth_pipeline_layout,
                    0,
                    &[cloth.sets[frame]],
                    &[],
                );
                device.cmd_push_constants(
                    cb,
                    self.cloth_pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::bytes_of(&push),
                );
                device.cmd_dispatch(cb, pending.count.div_ceil(64), 1, 1);
            }
            pending.count = 0;
        }
        if skinning {
            stamp(crate::gpu_zone::SKIN, true);
        }
        let barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT | vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::VERTEX_ATTRIBUTE_READ | vk::AccessFlags2::SHADER_STORAGE_READ);
        let barriers = [barrier];
        unsafe {
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&barriers));
        }
    }

    /// Drop every registered mesh (scene reset); the pipeline stays.
    pub fn clear(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        self.pending.clear();
        self.pending_at.clear();
        self.pending_morph.clear();
        self.pending_cloth.clear();
        self.poses.clear();
        self.stale.clear();
        for (_, skin) in self.skins.drain() {
            skin.destroy(device, allocator);
        }
    }

    /// Drop one registered mesh (its GPU resources were freed). The pipeline
    /// and the other meshes stay alive.
    /// Forget `mesh` and hand back its GPU resources for a deferred destroy
    /// (in-flight frames may still run its skin / morph passes).
    pub(crate) fn take(&mut self, mesh: usize) -> Option<GpuSkin> {
        self.pending.retain(|(m, _)| *m != mesh);
        self.pending_at = self.pending.iter().enumerate().map(|(k, (m, _))| (*m, k)).collect();
        self.pending_morph.retain(|(m, _)| *m != mesh);
        self.pending_cloth.remove(&mesh);
        self.poses.remove(&mesh);
        self.stale.remove(&mesh);
        self.skins.remove(&mesh)
    }

    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        for (_, skin) in self.skins.drain() {
            skin.destroy(device, allocator);
        }
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_pipeline(self.morph_pipeline, None);
            device.destroy_pipeline_layout(self.morph_pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.morph_set_layout, None);
            device.destroy_pipeline(self.cloth_pipeline, None);
            device.destroy_pipeline_layout(self.cloth_pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.cloth_set_layout, None);
        }
    }
}

/// Queue the latest pose of every stale mesh whose ordinary buffer is read again (not in `skip`)
/// and that was not queued a new pose this frame (`pending_at`).
fn requeue_stale(
    stale: &std::collections::HashSet<usize>,
    poses: &HashMap<usize, MeshPose>,
    pending_at: &HashMap<usize, usize>,
    skip: &std::collections::HashSet<usize>,
    pending: &mut Vec<(usize, Vec<Mat4>)>,
    spare: &mut Vec<Vec<Mat4>>,
) {
    for &mesh in stale {
        if skip.contains(&mesh) || pending_at.contains_key(&mesh) {
            continue;
        }
        if let Some(p) = poses.get(&mesh).filter(|p| !p.cur.is_empty()) {
            let mut v = spare.pop().unwrap_or_default();
            v.clear();
            v.extend_from_slice(&p.cur);
            pending.push((mesh, v));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stale_mesh_is_reskinned_from_its_last_pose_once_its_buffer_is_read_again() {
        let pose = |x: f32| MeshPose { cur: vec![Mat4::from_translation(glam::Vec3::new(x, 0.0, 0.0))], ..Default::default() };
        let stale: std::collections::HashSet<usize> = [1, 2, 3, 4].into();
        let poses: HashMap<usize, MeshPose> = [(1, pose(1.0)), (2, pose(2.0)), (3, pose(3.0))].into();
        let (mut pending, mut spare) = (Vec::new(), vec![Vec::new()]);
        // 1 still cluster-drawn (skipped), 2 got a fresh pose this frame, 4 has no pose recorded.
        let skip = [1].into();
        let pending_at = [(2, 0)].into();
        requeue_stale(&stale, &poses, &pending_at, &skip, &mut pending, &mut spare);
        assert_eq!(pending.len(), 1, "only mesh 3: 1 is skipped, 2 is queued already, 4 has no pose");
        assert_eq!(pending[0].0, 3);
        assert_eq!(pending[0].1, poses[&3].cur);
        // Once the cluster path stops drawing 1 it is skinned from the pose it last had.
        pending.clear();
        requeue_stale(&stale, &poses, &HashMap::new(), &Default::default(), &mut pending, &mut spare);
        let mut meshes: Vec<usize> = pending.iter().map(|p| p.0).collect();
        meshes.sort_unstable();
        assert_eq!(meshes, [1, 2, 3]);
    }

    #[test]
    fn previous_weights_follow_consecutive_frames_only() {
        let mut p = MeshPose::default();
        p.queue_weights(&[0.1, 0.2], 5);
        assert_eq!(p.prev_weights, [0.1, 0.2], "first call: no motion");
        p.queue_weights(&[0.5, 0.2], 6);
        assert_eq!((p.weights.as_slice(), p.prev_weights.as_slice()), ([0.5, 0.2].as_slice(), [0.1, 0.2].as_slice()));
        // A second call in the same frame replaces the current weights, keeps last frame's.
        p.queue_weights(&[0.6, 0.2], 6);
        assert_eq!((p.weights.as_slice(), p.prev_weights.as_slice()), ([0.6, 0.2].as_slice(), [0.1, 0.2].as_slice()));
        // A skipped frame: the gap means no motion information.
        p.queue_weights(&[0.9, 0.0], 9);
        assert_eq!(p.prev_weights, [0.9, 0.0]);
    }

    fn vert(x: f32, y: f32) -> Vertex {
        Vertex { position: [x, y, 0.0], normal: [0.0, 0.0, 1.0], ..Default::default() }
    }

    #[test]
    fn cloth_patch_selects_only_moved_vertices() {
        let animated = vec![vert(0.0, 0.0), vert(1.0, 0.0), vert(2.0, 0.0), vert(3.0, 0.0)];
        let mut sim = animated.clone();
        sim[1].position[1] = -0.5; // fell
        sim[3].normal = [1.0, 0.0, 0.0]; // only the normal changed
        sim[2].position[0] += 1e-7; // below eps: pinned
        let mut out = vec![9.0; 3]; // reused buffer is cleared
        let n = pack_cloth_patch(&animated, &sim, 1e-5, &mut out);
        assert_eq!(n, 2);
        assert_eq!(out.len(), 2 * CLOTH_PATCH_FLOATS);
        assert_eq!(out[0].to_bits(), 1);
        assert_eq!(&out[1..4], &[1.0, -0.5, 0.0]);
        assert_eq!(out[CLOTH_PATCH_FLOATS].to_bits(), 3);
        assert_eq!(&out[CLOTH_PATCH_FLOATS + 4..CLOTH_PATCH_FLOATS + 7], &[1.0, 0.0, 0.0]);
    }

    #[test]
    fn untouched_cloth_packs_nothing() {
        let v = vec![vert(0.0, 0.0), vert(1.0, 1.0)];
        let mut out = Vec::new();
        assert_eq!(pack_cloth_patch(&v, &v, 1e-5, &mut out), 0);
        assert!(out.is_empty());
    }
}
