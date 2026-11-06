//! Skinned clusters (GEO-5): the pass that poses only the visible clusters.
//!
//! Skinned and blendshape meshes are installed into the cluster store like
//! static ones, with their skin weights, per-cluster bone lists and sparse
//! blendshape deltas in three extra pools. The cull (`cluster_cull.comp`) works
//! on the *bind pose*: the LOD cut uses the bind-pose error spheres (the cooked
//! DAG already penalised simplification that hurts skinning), and every culling
//! bound is grown by the instance's pose margin (the largest distance the pose
//! and blendshapes can move a bind-pose point, `pose_margin`). No backface cone
//! is used for skinned instances.
//!
//! `cluster_skin.comp` then runs over each cull's visible lists (early, then
//! the late Hi-Z recoveries): it reserves pool space for every visible cluster
//! of a skinned instance, and poses their vertices once - blendshapes first,
//! then the cluster's <= 16 bones blended through the instance's palette - into
//! a per-frame pool that also holds the previous-frame position (posed from the
//! previous palette in the same thread, so motion vectors need no second
//! pass). The clustered prepass and main pass read that one pool, so their
//! positions are bit-identical.
//!
//! The previous position is morphed with the previous frame's blendshape
//! weights (`SkinPose::prev_weights`) as well as posed with the previous
//! palette, so blendshape motion shows in the velocity.
//!
//! Limits, by design: every pass of the frame shares the
//! one pool (no per-view skinning, so shadow views would reuse it); a full pool
//! drops the clusters that did not fit (flagged; the CPU pauses the path like
//! any other overflow).

use anyhow::Result;
use ash::vk;
use bytemuck::Zeroable;
use glam::Mat4;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocator;

use crate::alloc::Buffer;
use crate::cluster::{SKIN_SPV, SkinInstRec, SkinPush};
use crate::gpu_cull::GrowBuffer;

/// Bytes of one posed vertex in the pool (`SkinVert` in cluster_skin.comp).
pub const SKIN_VERT_BYTES: usize = 64;
/// Default capacity of the per-frame skinned pool, in vertices (up to 64 MB per frame slot, allocated only for the demand).
pub const DEFAULT_SKIN_POOL_VERTICES: u32 = 1 << 20;
const CTL_WORDS: usize = 8;
const CTL_BYTES: u64 = (CTL_WORDS * 4) as u64;

/// `ctl[]` word indices (mirror of cluster_skin.comp; a test compares them).
pub mod ctl {
    pub const TOTAL: usize = 0;
    pub const USED: usize = 1;
    pub const FLAGS: usize = 2;
    pub const LIST_BASE: usize = 3;
    pub const N: usize = 4;
    pub const SKINNED: usize = 5;
    /// `ctl[FLAGS]` bits.
    pub const FLAG_POOL_FULL: u32 = 1;
    pub const FLAG_BAD: u32 = 2;
}

/// Slot value of a visible entry that has no skinned vertices.
pub const NO_SLOT: u32 = u32::MAX;

/// What the skin pass did, read back from a retired frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SkinCounts {
    /// Pool vertices reserved (above the capacity means the pool overflowed).
    pub used: u32,
    /// Visible clusters posed.
    pub clusters: u32,
    /// Entries listed over both passes.
    pub listed: u32,
    pub flags: u32,
}

impl SkinCounts {
    pub fn from_words(w: &[u32]) -> Self {
        let g = |i: usize| w.get(i).copied().unwrap_or(0);
        Self { used: g(ctl::USED), clusters: g(ctl::SKINNED), listed: g(ctl::TOTAL), flags: g(ctl::FLAGS) }
    }
}

/// The pose of one skinned instance, as the renderer is given it.
#[derive(Clone, Copy)]
pub struct SkinPose<'a> {
    /// This frame's joint palette (skin matrices, bind -> posed object space).
    pub palette: &'a [Mat4],
    /// Last frame's palette for motion vectors (None = no motion).
    pub prev_palette: Option<&'a [Mat4]>,
    /// Blendshape weights by target index.
    pub weights: &'a [f32],
    /// Last frame's blendshape weights (None, or a different length = no
    /// blendshape motion), so the previous position is morphed like last frame's.
    pub prev_weights: Option<&'a [f32]>,
    /// Object-space distance the pose and blendshapes move a bind-pose point at most.
    pub margin: f32,
}

/// Conservative object-space bound on how far a pose moves any point of the
/// bind-pose sphere (`centre`, `radius`): for every joint matrix `M`,
/// `|M p - p| <= |M c - c| + |A - I|_F * radius` with `A` the linear part; a
/// skin blend is a convex combination, so the largest joint bound holds for
/// every vertex. Blendshapes add `sum |w_t| * reach_t`. Loose for large
/// rotations (it ignores which bones a cluster uses) but never too small.
pub fn pose_margin(centre: [f32; 3], radius: f32, palette: &[Mat4], weights: &[f32], morph_reach: &[f32]) -> f32 {
    let c = glam::Vec3::from(centre);
    let mut margin = 0.0f32;
    for m in palette {
        let linear = glam::Mat3::from_mat4(*m) - glam::Mat3::IDENTITY;
        let frob = linear.to_cols_array().iter().map(|x| x * x).sum::<f32>().sqrt();
        let shift = (m.transform_point3(c) - c).length();
        let bound = shift + frob * radius;
        if !bound.is_finite() {
            return f32::MAX.sqrt(); // f32::max would swallow a NaN
        }
        margin = margin.max(bound);
    }
    let morph: f32 = weights.iter().zip(morph_reach).map(|(w, r)| w.abs() * r).sum();
    let total = margin + morph;
    if total.is_finite() { total } else { f32::MAX.sqrt() }
}

pub(crate) struct SkinPipes {
    pub set_layout: vk::DescriptorSetLayout,
    pub layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
}

impl SkinPipes {
    pub(crate) fn new(device: &ash::Device, cluster_set_layout: vk::DescriptorSetLayout) -> Result<Self> {
        let bindings: Vec<_> = (0..10)
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let set_layout =
            unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)? };
        let layouts = [cluster_set_layout, set_layout];
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(size_of::<SkinPush>() as u32)];
        let layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&ranges),
                None,
            )?
        };
        let pipeline = crate::gpu_cull::compute_pipeline(device, layout, SKIN_SPV)?;
        Ok(Self { set_layout, layout, pipeline })
    }

    pub(crate) fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

/// One frame slot's skin buffers.
pub(crate) struct SkinFrame {
    pub set: vk::DescriptorSet,
    palettes: GrowBuffer,
    weights: GrowBuffer,
    insts: GrowBuffer,
    cum: GrowBuffer,
    list: GrowBuffer,
    ctl: Buffer,
    dispatch: Buffer,
    readback: Buffer,
    /// The pass was recorded into this slot: read `ctl` when it retires.
    pub recorded: bool,
}

impl SkinFrame {
    fn new(device: &ash::Device, allocator: &mut Allocator, set: vk::DescriptorSet) -> Result<Self> {
        let host = |name| GrowBuffer::new(vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, name);
        let gpu = |name| GrowBuffer::new(vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, name);
        Ok(Self {
            set,
            palettes: host("cluster-skin-palettes"),
            weights: host("cluster-skin-weights"),
            insts: host("cluster-skin-insts"),
            cum: gpu("cluster-skin-cum"),
            list: gpu("cluster-skin-list"),
            ctl: Buffer::new(
                device,
                allocator,
                CTL_BYTES,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::GpuOnly,
                "cluster-skin-ctl",
            )?,
            dispatch: Buffer::new(
                device,
                allocator,
                32,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
                MemoryLocation::GpuOnly,
                "cluster-skin-dispatch",
            )?,
            readback: Buffer::new(
                device,
                allocator,
                CTL_BYTES,
                vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuToCpu,
                "cluster-skin-readback",
            )?,
            recorded: false,
        })
    }

    fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        for b in [&mut self.palettes, &mut self.weights, &mut self.insts, &mut self.cum, &mut self.list] {
            b.destroy(device, allocator);
        }
        self.ctl.destroy(device, allocator);
        self.dispatch.destroy(device, allocator);
        self.readback.destroy(device, allocator);
    }
}

/// What a skin recording needs besides the pipelines.
#[derive(Clone, Copy)]
pub(crate) struct SkinTargets {
    pub cluster_set: vk::DescriptorSet,
    pub set: vk::DescriptorSet,
    pub ctl: vk::Buffer,
    pub dispatch: vk::Buffer,
    pub readback: vk::Buffer,
    pub batch_count: u32,
    pub list_cap: u32,
    pub pool_cap: u32,
}

/// The skin stage of the cluster system: pipelines, per-slot buffers and this
/// frame's CPU lists.
pub(crate) struct SkinStage {
    pub pipes: SkinPipes,
    pool: vk::DescriptorPool,
    pub frames: Vec<SkinFrame>,
    /// Matrices of this frame's skinned instances (current then previous palette each).
    palette_cpu: Vec<[f32; 16]>,
    weights_cpu: Vec<f32>,
    pub insts_cpu: Vec<SkinInstRec>,
    /// Capacity of the per-frame pool in vertices.
    pub pool_vertices: u32,
    /// Counters of the last retired frame.
    pub counts: SkinCounts,
    pub pool_overflows: u32,
}

impl SkinStage {
    pub(crate) fn new(device: &ash::Device, allocator: &mut Allocator, cluster_set_layout: vk::DescriptorSetLayout) -> Result<Self> {
        let pipes = SkinPipes::new(device, cluster_set_layout)?;
        let sizes = [vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(10 * crate::cluster::CLUSTER_SETS as u32)];
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(crate::cluster::CLUSTER_SETS as u32).pool_sizes(&sizes),
                None,
            )?
        };
        let layouts = vec![pipes.set_layout; crate::cluster::CLUSTER_SETS];
        let sets = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts))? };
        let frames = sets.into_iter().map(|s| SkinFrame::new(device, allocator, s)).collect::<Result<Vec<_>>>()?;
        let pool_vertices = std::env::var("ZESTENGINE_CLUSTER_SKIN_VERTICES")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|&v| v >= 1024)
            .unwrap_or(DEFAULT_SKIN_POOL_VERTICES);
        Ok(Self {
            pipes,
            pool,
            frames,
            palette_cpu: Vec::new(),
            weights_cpu: Vec::new(),
            insts_cpu: Vec::new(),
            pool_vertices,
            counts: SkinCounts::default(),
            pool_overflows: 0,
        })
    }

    pub(crate) fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        for f in &mut self.frames {
            f.destroy(device, allocator);
        }
        self.pipes.destroy(device);
        unsafe { device.destroy_descriptor_pool(self.pool, None) };
    }

    pub(crate) fn begin_build(&mut self) {
        self.palette_cpu.clear();
        self.weights_cpu.clear();
        self.insts_cpu.clear();
    }

    /// Queue a skinned instance's pose; returns the value for `InstRec::mesh_batch[2]`
    /// (the instance's index + 1).
    pub(crate) fn add(&mut self, pose: &SkinPose) -> u32 {
        let joints = pose.palette.len().max(1);
        let offset = (self.palette_cpu.len()) as u32;
        let cur = |i: usize| pose.palette.get(i).copied().unwrap_or(Mat4::IDENTITY).to_cols_array();
        self.palette_cpu.extend((0..joints).map(cur));
        let prev_offset = match pose.prev_palette.filter(|p| p.len() == pose.palette.len()) {
            Some(prev) => {
                let at = self.palette_cpu.len() as u32;
                self.palette_cpu.extend(prev.iter().map(|m| m.to_cols_array()));
                at
            }
            None => offset,
        };
        let weights_offset = self.weights_cpu.len() as u32;
        self.weights_cpu.extend_from_slice(pose.weights);
        // Last frame's weights follow this frame's; without them the previous
        // position uses the current weights (no blendshape motion).
        let prev_weights_offset = match pose.prev_weights.filter(|p| p.len() == pose.weights.len() && p != &pose.weights) {
            Some(prev) => {
                let at = self.weights_cpu.len() as u32;
                self.weights_cpu.extend_from_slice(prev);
                at
            }
            None => weights_offset,
        };
        self.insts_cpu.push(SkinInstRec {
            a: [offset, prev_offset, pose.palette.len() as u32, weights_offset],
            b: [pose.weights.len() as u32, prev_weights_offset, 0, 0],
        });
        self.insts_cpu.len() as u32
    }

    /// Size the slot's buffers, copy the CPU lists and point its set at the pools.
    /// `pools`: skin stream, bone list, morph entries.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn upload(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        slot: usize,
        pools: [vk::Buffer; 3],
        batches: u32,
        list_cap: u32,
    ) -> Result<()> {
        let f = &mut self.frames[slot];
        f.palettes.ensure(device, allocator, self.palette_cpu.len().max(1) * 64)?;
        f.weights.ensure(device, allocator, self.weights_cpu.len().max(1) * 4)?;
        f.insts.ensure(device, allocator, self.insts_cpu.len().max(1) * size_of::<SkinInstRec>())?;
        f.cum.ensure(device, allocator, (batches as usize + 1) * 4)?;
        f.list.ensure(device, allocator, list_cap.max(1) as usize * 16)?;
        if self.palette_cpu.is_empty() {
            // Never read (no skinned instance), but the binding must be valid.
            f.palettes.write(bytemuck::bytes_of(&Mat4::IDENTITY.to_cols_array()));
        } else {
            f.palettes.write(bytemuck::cast_slice(&self.palette_cpu));
        }
        f.weights.write(bytemuck::cast_slice(&self.weights_cpu));
        f.insts.write(bytemuck::cast_slice(&self.insts_cpu));
        let buffers = [
            pools[0],
            pools[1],
            pools[2],
            f.palettes.handle(),
            f.weights.handle(),
            f.insts.handle(),
            f.cum.handle(),
            f.list.handle(),
            f.ctl.handle,
            f.dispatch.handle,
        ];
        let infos: Vec<[vk::DescriptorBufferInfo; 1]> =
            buffers.iter().map(|&b| [vk::DescriptorBufferInfo::default().buffer(b).range(vk::WHOLE_SIZE)]).collect();
        let writes: Vec<_> = infos
            .iter()
            .enumerate()
            .map(|(b, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(f.set)
                    .dst_binding(b as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
            })
            .collect();
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        Ok(())
    }

    pub(crate) fn targets(&self, slot: usize, cluster_set: vk::DescriptorSet, batches: u32, list_cap: u32, pool_cap: u32) -> SkinTargets {
        let f = &self.frames[slot];
        SkinTargets {
            cluster_set,
            set: f.set,
            ctl: f.ctl.handle,
            dispatch: f.dispatch.handle,
            readback: f.readback.handle,
            batch_count: batches,
            list_cap,
            pool_cap,
        }
    }

    /// The control words of the last skin pass of `slot` (once it has completed).
    pub(crate) fn read_ctl(&self, slot: usize) -> Vec<u32> {
        let m = self.frames[slot].readback.mapped().unwrap_or(&[]);
        bytemuck::pod_collect_to_vec(&m[..m.len().min(CTL_BYTES as usize) & !3])
    }
}

fn compute_barrier(device: &ash::Device, cb: vk::CommandBuffer, src: (vk::PipelineStageFlags2, vk::AccessFlags2), dst: (vk::PipelineStageFlags2, vk::AccessFlags2)) {
    let b = [vk::MemoryBarrier2::default()
        .src_stage_mask(src.0)
        .src_access_mask(src.1)
        .dst_stage_mask(dst.0)
        .dst_access_mask(dst.1)];
    unsafe { device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&b)) };
}

/// Record one half of the skin pass (`late` = the Hi-Z recoveries) after the
/// matching cull has finished its lists. The early half clears the control
/// words; both halves end with the pool visible to the vertex stage.
pub(crate) fn record_skin(device: &ash::Device, cb: vk::CommandBuffer, pipeline: vk::Pipeline, layout: vk::PipelineLayout, t: &SkinTargets, late: bool) {
    const CS: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::COMPUTE_SHADER;
    const RW: vk::AccessFlags2 =
        vk::AccessFlags2::from_raw(vk::AccessFlags2::SHADER_STORAGE_READ.as_raw() | vk::AccessFlags2::SHADER_STORAGE_WRITE.as_raw());
    let mut push = SkinPush::zeroed();
    push.late = late as u32;
    push.batch_count = t.batch_count;
    push.list_cap = t.list_cap;
    push.pool_cap = t.pool_cap;
    unsafe {
        if !late {
            device.cmd_fill_buffer(cb, t.ctl, 0, CTL_BYTES, 0);
        }
        // The cull's lists (compute writes) and the clear (transfer) before the plan reads them.
        compute_barrier(
            device,
            cb,
            (CS | vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::TRANSFER_WRITE),
            (CS, RW),
        );
        device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, layout, 0, &[t.cluster_set, t.set], &[]);
        let step = |push: &SkinPush| {
            device.cmd_push_constants(cb, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(push));
        };
        let between = || {
            compute_barrier(
                device,
                cb,
                (CS, vk::AccessFlags2::SHADER_STORAGE_WRITE),
                (CS | vk::PipelineStageFlags2::DRAW_INDIRECT, RW | vk::AccessFlags2::INDIRECT_COMMAND_READ),
            );
        };
        push.mode = 0;
        step(&push);
        device.cmd_dispatch(cb, 1, 1, 1);
        between();
        push.mode = 1;
        step(&push);
        device.cmd_dispatch_indirect(cb, t.dispatch, 0);
        between();
        push.mode = 2;
        step(&push);
        device.cmd_dispatch_indirect(cb, t.dispatch, 16);
        compute_barrier(
            device,
            cb,
            (CS, vk::AccessFlags2::SHADER_STORAGE_WRITE),
            (
                vk::PipelineStageFlags2::VERTEX_SHADER | vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::TRANSFER_READ,
            ),
        );
        device.cmd_copy_buffer(cb, t.ctl, t.readback, &[vk::BufferCopy::default().size(CTL_BYTES)]);
        compute_barrier(
            device,
            cb,
            (vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE),
            (vk::PipelineStageFlags2::HOST, vk::AccessFlags2::HOST_READ),
        );
    }
}

/// CPU reference of `cluster_skin.comp` for one cluster: the posed vertices of
/// cluster `cluster` of `up` under `pose`. Same arithmetic order as the shader
/// (f32), used by the headless tests; positions agree to float rounding.
pub mod reference {
    use super::SkinPose;
    use crate::cluster::ClusterMeshUpload;
    use glam::{Mat4, Vec3, Vec4};

    /// One posed vertex.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Posed {
        pub pos: [f32; 3],
        pub nrm: [f32; 3],
        pub tan: [f32; 3],
        pub prev: [f32; 3],
    }

    fn snorm16(w: i16) -> f32 {
        (f32::from(w) / 32767.0).clamp(-1.0, 1.0)
    }

    pub fn oct_decode(e: [i16; 2]) -> Vec3 {
        let (x, y) = (snorm16(e[0]), snorm16(e[1]));
        let mut n = Vec3::new(x, y, 1.0 - x.abs() - y.abs());
        if n.z < 0.0 {
            let sign = |v: f32| if v >= 0.0 { 1.0 } else { -1.0 };
            let (sx, sy) = (sign(n.x), sign(n.y));
            n.x = (1.0 - y.abs()) * sx;
            n.y = (1.0 - x.abs()) * sy;
        }
        n / n.length().max(1e-20)
    }

    /// The posed vertices of one cluster, in cluster-local order.
    pub fn pose_cluster(up: &ClusterMeshUpload, cluster: usize, pose: &SkinPose) -> Vec<Posed> {
        let cl = &up.clusters[cluster];
        let mut out = Vec::with_capacity(cl.vertex_count as usize);
        let entries = &up.morphs[cl.morph_offset as usize..][..cl.morph_count as usize];
        let palette = |i: usize, set: Option<&[Mat4]>| set.and_then(|p| p.get(i)).copied().unwrap_or(Mat4::IDENTITY);
        for t in 0..cl.vertex_count as usize {
            let v = &up.vertices[cl.vertex_offset as usize + t];
            let q = |i: usize| f32::from(v.pos[i]);
            let mut p = Vec3::from(up.mesh_min) + Vec3::new(q(0), q(1), q(2)) * up.step;
            let mut n = oct_decode(v.normal);
            let tg = oct_decode(v.tangent);
            let (mut pm, mut nm, mut morphed) = (p, n, false);
            let mut pp = p;
            let prev_weights = pose.prev_weights.filter(|w| w.len() == pose.weights.len()).unwrap_or(pose.weights);
            for e in entries.iter().filter(|e| e.vertex as usize == t) {
                let h = |b: u16| half::f16::from_bits(b).to_f32();
                let d = Vec3::new(h(e.d_pos[0]), h(e.d_pos[1]), h(e.d_pos[2]));
                if let Some(&w) = prev_weights.get(e.target as usize)
                    && w.abs() > 1e-4
                {
                    pp += d * w;
                }
                let Some(&w) = pose.weights.get(e.target as usize) else { continue };
                if w.abs() <= 1e-4 {
                    continue;
                }
                morphed = true;
                pm += d * w;
                nm += Vec3::new(h(e.d_normal[0]), h(e.d_normal[1]), h(e.d_normal[2])) * w;
            }
            if morphed {
                let l = nm.length();
                n = if l > 1e-20 { nm / l } else { Vec3::Y };
                p = pm;
            }
            let mut posed = Posed { pos: p.to_array(), nrm: n.to_array(), tan: tg.to_array(), prev: pp.to_array() };
            let sk = up.skin.get(cl.vertex_offset as usize + t).copied().unwrap_or_default();
            let wv = sk.weights.map(f32::from);
            let sum: f32 = wv.iter().sum();
            if sum > 0.0 {
                let (mut cur, mut prv) = (Mat4::ZERO, Mat4::ZERO);
                for (k, w) in wv.iter().enumerate() {
                    if *w <= 0.0 {
                        continue;
                    }
                    let local = sk.bones[k] as usize;
                    if local >= cl.bone_count as usize {
                        continue;
                    }
                    let joint = up.bones[cl.bone_offset as usize + local] as usize;
                    if joint >= pose.palette.len() {
                        continue;
                    }
                    cur += palette(joint, Some(pose.palette)) * w;
                    prv += palette(joint, Some(pose.prev_palette.unwrap_or(pose.palette))) * w;
                }
                cur *= 1.0 / sum;
                prv *= 1.0 / sum;
                let p4 = Vec4::new(p.x, p.y, p.z, 1.0);
                posed.pos = (cur * p4).truncate().to_array();
                posed.prev = (prv * Vec4::new(pp.x, pp.y, pp.z, 1.0)).truncate().to_array();
                let r = glam::Mat3::from_mat4(cur);
                posed.nrm = (r * n).normalize_or_zero().to_array();
                let tt = r * tg;
                posed.tan = if tt.length() > 1e-6 { tt.normalize().to_array() } else { tt.to_array() };
            }
            out.push(posed);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_constants_match_rust() {
        let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/cluster_skin.comp")).unwrap();
        let get = |name: &str| -> u32 {
            let line = src.lines().find(|l| l.contains(&format!("const uint {name} ="))).unwrap_or_else(|| panic!("no {name}"));
            line.split('=').nth(1).unwrap().split(';').next().unwrap().trim().trim_end_matches('u').parse().unwrap()
        };
        assert_eq!(get("CTL_TOTAL") as usize, ctl::TOTAL);
        assert_eq!(get("CTL_USED") as usize, ctl::USED);
        assert_eq!(get("CTL_FLAGS") as usize, ctl::FLAGS);
        assert_eq!(get("CTL_LIST_BASE") as usize, ctl::LIST_BASE);
        assert_eq!(get("CTL_N") as usize, ctl::N);
        assert_eq!(get("CTL_SKINNED") as usize, ctl::SKINNED);
        assert!(src.contains("const uint NONE = 0xFFFFFFFFu;") && NO_SLOT == u32::MAX);
        assert!(src.contains("flag(1u)") && ctl::FLAG_POOL_FULL == 1 && ctl::FLAG_BAD == 2);
        // 64 bytes per posed vertex = four vec4.
        assert_eq!(SKIN_VERT_BYTES, 64);
    }

    #[test]
    fn pose_margin_bounds_every_blended_displacement() {
        // Random palettes: the margin must bound |M p - p| for p in the sphere, for every joint.
        let mut seed = 1u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        for _ in 0..200 {
            let palette: Vec<Mat4> = (0..6)
                .map(|_| {
                    Mat4::from_translation(glam::Vec3::new(rnd(), rnd(), rnd()))
                        * Mat4::from_axis_angle(glam::Vec3::new(rnd(), rnd(), rnd() + 1.0).normalize(), rnd() * 5.0)
                        * Mat4::from_scale(glam::Vec3::splat(1.0 + rnd()))
                })
                .collect();
            let (c, r) = ([rnd(), rnd(), rnd()], 0.5 + rnd().abs());
            let margin = pose_margin(c, r, &palette, &[], &[]);
            for _ in 0..20 {
                let d = glam::Vec3::new(rnd(), rnd(), rnd()).normalize() * r * (rnd() + 0.5);
                let p = glam::Vec3::from(c) + d;
                // A convex blend of joints moves a point by at most the largest single-joint move.
                let w: Vec<f32> = (0..palette.len()).map(|_| rnd() + 0.5).collect();
                let sum: f32 = w.iter().sum();
                let blended = palette.iter().zip(&w).map(|(m, w)| m.transform_point3(p) * (w / sum)).fold(glam::Vec3::ZERO, |a, b| a + b);
                assert!((blended - p).length() <= margin + 1e-4);
            }
        }
        // Blendshapes add weight * reach.
        let m = pose_margin([0.0; 3], 1.0, &[Mat4::IDENTITY], &[0.5, -2.0], &[1.0, 0.25]);
        assert!((m - 1.0).abs() < 1e-6);
        // A non-finite pose cannot produce a small margin.
        assert!(pose_margin([0.0; 3], 1.0, &[Mat4::from_scale(glam::Vec3::splat(f32::NAN))], &[], &[]) > 1e9);
    }
}
