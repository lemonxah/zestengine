//! GPU-driven culling with indirect draws.
//!
//! Each frame the CPU groups the culled passes' instances into batches (one per
//! material-set + mesh + per-draw push lanes for camera views, one per mesh for
//! shadow views) and uploads the per-draw data once into persistent, growable
//! per-submission buffers. A compute pass (`gpu_cull.comp`) then tests every
//! (view, instance) candidate against the view's frustum, an optional distance
//! / small-object limit and, for the main camera, an optional Hi-Z occlusion
//! test against the previous frame's depth pyramid (`hiz_build.comp`). Visible
//! instances are compacted into each batch's region of a device-local instance
//! stream (the same mat4 stream the vertex shaders already read) while atomics
//! fill one `VkDrawIndexedIndirectCommand` per batch, so the passes draw with
//! `vkCmdDrawIndexedIndirect` and the vertex shaders stay unchanged.
//!
//! Hi-Z is two-phase so nothing coming into view is drawn a frame late. The
//! early pass above draws what last frame's pyramid does not hide and flags
//! the main-view instances it rejected. The first main-camera depth pass (the
//! depth prepass when there is one, else the main pass) draws the early
//! batches, ends, builds a pyramid from that depth with this frame's camera,
//! and the late pass re-tests only the flagged instances against it into a
//! second set of commands / instances (`Late`); the pass resumes (LOAD) and
//! draws them, and every later pass of the main camera draws both sets. The
//! pyramid is rebuilt from the finished depth after the main pass for the
//! next frame's early pass.
//!
//! Every batch binds its own mesh, material-set and push constants, so each one is
//! a single-command indirect draw: a count buffer would only ever say 1. The
//! batch's instance region is bound as the instance vertex buffer offset, so
//! `firstInstance` stays 0 and `drawIndirectFirstInstance` is not required.
//!
//! GPU-culled: opaque + alpha-test draws of the main camera (its depth
//! prepass included), shadow views, VR eyes. Transparent (back-to-front),
//! custom-shader, outlined, water and error draws keep the CPU path, as do the
//! secondary cameras (captures, planar mirrors, previews, render textures).

use crate::hitch::WaitNamed as _;
use std::collections::HashMap;

use anyhow::Result;
use ash::vk;
use ash::vk::Handle;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};

use crate::alloc::Buffer;
use crate::cluster::{ClusterBatch, ClusterSystem};
use crate::frame::FRAMES_IN_FLIGHT;
use crate::pipeline::INSTANCE_STRIDE;
use crate::{
    AlphaMode, BoundDraw, ClusterRefusal, CameraData, DrawCmd, FrameInput, PushData, RENDER_QUEUE_TRANSPARENT, RenderStats, Renderer,
    SceneBatch, XR_SLOTS, frustum_planes, gpu_zone, image_barrier, scene_pipeline_key, sphere_light_mask,
};

const CULL_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gpu_cull.comp.spv"));
/// Storage-buffer bindings of gpu_cull.comp (draws, candidates, batches,
/// views, commands, instances, counters, late commands, late instances,
/// occluded flags) and its pyramid sampler.
const CULL_BUFFER_BINDINGS: [u32; 10] = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10];
const CULL_HIZ_BINDING: u32 = 7;
const HIZ_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/hiz_build.comp.spv"));

/// Views one cull dispatch can serve (main camera + every shadow layer fits).
/// Matches `MAX_VIEWS` in gpu_cull.comp (and its 64-thread workgroup).
pub(crate) const MAX_CULL_VIEWS: usize = 16;
const MAX_HIZ_MIPS: usize = 16;
/// Per view: tested, visible, frustum/distance/size culled, occluded (early
/// pass); then the late pass's block: re-tested, recovered, -, still occluded.
const LATE_COUNTERS: u64 = (MAX_CULL_VIEWS * 16) as u64;
const COUNTER_BYTES: u64 = 2 * LATE_COUNTERS;
/// Pyramid builds per frame slot: after the main pass (next frame's early
/// pass), and inside the first main-camera depth pass (this frame's late pass).
const HIZ_BUILDS: usize = 2;
const HIZ_BUILD_NEXT: usize = 0;
const HIZ_BUILD_LATE: usize = 1;
/// Smallest allocation of a growable cull buffer.
const MIN_BUFFER_BYTES: usize = 64 * 1024;
const INDIRECT_STRIDE: u32 = size_of::<IndirectCmd>() as u32;
/// Cull slots: one per frame in flight, then one per VR eye slot.
const CULL_SLOTS: usize = FRAMES_IN_FLIGHT + XR_SLOTS;
/// Hi-Z from the previous frame is skipped after a camera cut: a jump farther
/// than this (m) or a turn sharper than `HIZ_MIN_FORWARD_DOT` in one frame.
const HIZ_MAX_MOVE: f32 = 2.0;
const HIZ_MIN_FORWARD_DOT: f32 = 0.9;

/// Renderer option: GPU-driven culling and its optional occlusion tests.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuCullSettings {
    /// Cull + draw opaque / alpha-test instances on the GPU (indirect draws).
    /// Off = the CPU frustum cull and instanced draws.
    pub enabled: bool,
    /// Also reject main-camera instances hidden behind the depth (Hi-Z,
    /// two-phase: what last frame's depth hides is re-tested against this
    /// frame's, so disoccluded objects appear the same frame).
    pub hiz_occlusion: bool,
    /// Main camera: skip instances whose projected radius is under this many
    /// pixels. 0 = off.
    pub min_pixel_radius: f32,
    /// Main camera: skip instances farther than this (m). 0 = off.
    pub max_distance: f32,
}

impl Default for GpuCullSettings {
    fn default() -> Self {
        Self { enabled: true, hiz_occlusion: false, min_pixel_radius: 0.0, max_distance: 0.0 }
    }
}

impl GpuCullSettings {
    /// Defaults, overridden by `ZESTENGINE_GPU_CULL=0` (CPU path) and
    /// `ZESTENGINE_HIZ=1` (Hi-Z occlusion on) for testing the game runtime.
    pub(crate) fn from_env() -> Self {
        let flag = |name: &str| std::env::var(name).ok().map(|v| v != "0" && !v.is_empty());
        let mut s = Self::default();
        if let Some(on) = flag("ZESTENGINE_GPU_CULL") {
            s.enabled = on;
        }
        if let Some(on) = flag("ZESTENGINE_HIZ") {
            s.hiz_occlusion = on;
        }
        s
    }
}

/// Per-draw data read by the cull shader (std430 `Draw`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuDraw {
    model: [[f32; 4]; 4],
    /// Object-space bounding sphere: center xyz, radius w (<= 0: never culled).
    sphere: [f32; 4],
    /// Per-instance base colour.
    color: [f32; 4],
    /// Per-instance material parameters.
    param: [f32; 4],
}

impl GpuDraw {
    /// `lightmap`: pack the draw's baked-lightmap layer into the instance
    /// (col0.w = layer + 1; the vertex shaders read it and clear it), so
    /// lightmapped statics batch together instead of one batch per layer.
    fn of(d: &DrawCmd, lightmap: bool, color: [f32; 4], param: [f32; 4]) -> Self {
        let mut model = d.transform.to_cols_array_2d();
        model[0][3] = if lightmap && d.lightmap_layer >= 0 { d.lightmap_layer as f32 + 1.0 } else { 0.0 };
        Self {
            model,
            sphere: d.mesh_center.extend(d.bound_radius).to_array(),
            color,
            param,
        }
    }
}

/// One culled view (std430 `View`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuView {
    pub(crate) planes: [[f32; 4]; 6],
    pub(crate) hiz_vp: [[f32; 4]; 4],
    /// xy pyramid mip-0 size, z mip count, w 1 = Hi-Z test on.
    pub(crate) hiz: [f32; 4],
    /// xyz camera position, w max distance (0 = off).
    pub(crate) cam: [f32; 4],
    /// x min projected radius px (0 = off), y px per unit at distance 1,
    /// z near-plane test (0 for shadow views: casters behind it still cast).
    pub(crate) lod: [f32; 4],
}

impl GpuView {
    pub(crate) fn new(view_proj: Mat4, cam: Vec3, near: bool) -> Self {
        let planes = frustum_planes(view_proj);
        Self {
            planes: planes.map(|p| p.to_array()),
            hiz_vp: Mat4::IDENTITY.to_cols_array_2d(),
            hiz: [0.0; 4],
            cam: cam.extend(0.0).to_array(),
            lod: [0.0, 0.0, if near { 1.0 } else { 0.0 }, 0.0],
        }
    }
}

/// `VkDrawIndexedIndirectCommand` as uploaded (instanceCount starts at 0).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq, Eq)]
pub(crate) struct IndirectCmd {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    vertex_offset: i32,
    first_instance: u32,
}

/// Push constants of gpu_cull.comp.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CullPush {
    /// Late pass: the view-projection its pyramid was built with.
    late_vp: [[f32; 4]; 4],
    count: u32,
    /// 0 = early pass, 1 = late pass.
    late: u32,
}

const _: () = assert!(size_of::<CullPush>() == 72);
const _: () = assert!(size_of::<IndirectCmd>() == size_of::<vk::DrawIndexedIndirectCommand>());
const _: () = assert!(size_of::<GpuView>() == 208);
const _: () = assert!(size_of::<GpuDraw>() == 112);

/// New capacity (bytes) of a growable buffer that must hold `needed`: never
/// shrinks, grows to a power of two so steady growth reallocates rarely.
pub(crate) fn grow_capacity(current: usize, needed: usize) -> usize {
    if needed <= current { current } else { needed.next_power_of_two().max(MIN_BUFFER_BYTES) }
}

/// CPU-side lists for one cull dispatch, reused across frames.
#[derive(Default)]
pub(crate) struct CullBuilder {
    draws: Vec<GpuDraw>,
    /// `FrameInput::draws` index -> `draws` index (u32::MAX = not uploaded).
    draw_slot: Vec<u32>,
    /// (draws index, batch).
    candidates: Vec<[u32; 2]>,
    /// (view, first instance of the batch's region, candidate count, 0).
    batches: Vec<[u32; 4]>,
    /// Pack lightmap layers into the instances (off for the lightmap-UV preview).
    pack_lightmap: bool,
    cmds: Vec<IndirectCmd>,
    views: Vec<GpuView>,
}

impl CullBuilder {
    fn begin(&mut self, draw_count: usize) {
        self.draws.clear();
        self.draw_slot.clear();
        self.draw_slot.resize(draw_count, u32::MAX);
        self.candidates.clear();
        self.batches.clear();
        self.cmds.clear();
        self.views.clear();
    }

    fn add_view(&mut self, view: GpuView) -> u32 {
        self.views.push(view);
        self.views.len() as u32 - 1
    }

    fn add_batch(&mut self, view: u32, index_count: u32) -> u32 {
        self.batches.push([view, 0, 0, 0]);
        self.cmds.push(IndirectCmd { index_count, ..Zeroable::zeroed() });
        self.batches.len() as u32 - 1
    }

    /// Queue draw `index` for culling into `batch`; its data uploads once
    /// however many views reference it.
    fn add_candidate(&mut self, index: usize, d: &DrawCmd, color: [f32; 4], param: [f32; 4], batch: u32) {
        let slot = match self.draw_slot[index] {
            u32::MAX => {
                self.draws.push(GpuDraw::of(d, self.pack_lightmap, color, param));
                let s = self.draws.len() as u32 - 1;
                self.draw_slot[index] = s;
                s
            }
            s => s,
        };
        self.candidates.push([slot, batch]);
        self.batches[batch as usize][2] += 1;
    }

    /// Lay the batches' instance regions out back to back. Returns the total
    /// instance capacity.
    fn finalize(&mut self) -> u32 {
        let mut base = 0u32;
        for b in &mut self.batches {
            b[1] = base;
            base += b[2];
        }
        base
    }

    fn cmd_offset(batch: u32) -> u64 {
        batch as u64 * INDIRECT_STRIDE as u64
    }

    fn instance_offset(&self, batch: u32) -> u64 {
        self.batches[batch as usize][1] as u64 * INSTANCE_STRIDE as u64
    }

    fn instance_count(&self, batch: u32) -> u32 {
        self.batches[batch as usize][2]
    }
}

/// A buffer that only grows (old one dropped when the owning slot's previous
/// submission has retired, i.e. before any command of this one uses it).
pub(crate) struct GrowBuffer {
    buf: Option<Buffer>,
    bytes: usize,
    usage: vk::BufferUsageFlags,
    location: MemoryLocation,
    name: &'static str,
}

impl GrowBuffer {
    pub(crate) fn new(usage: vk::BufferUsageFlags, location: MemoryLocation, name: &'static str) -> Self {
        Self { buf: None, bytes: 0, usage, location, name }
    }

    pub(crate) fn ensure(&mut self, device: &ash::Device, allocator: &mut Allocator, needed: usize) -> Result<()> {
        let bytes = grow_capacity(self.bytes, needed.max(1));
        if self.buf.is_some() && bytes == self.bytes {
            return Ok(());
        }
        if let Some(mut old) = self.buf.take() {
            old.destroy(device, allocator);
        }
        self.buf = Some(Buffer::new(device, allocator, bytes as u64, self.usage, self.location, self.name)?);
        self.bytes = bytes;
        Ok(())
    }

    /// Allocated bytes.
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }

    pub(crate) fn handle(&self) -> vk::Buffer {
        self.buf.as_ref().map_or(vk::Buffer::null(), |b| b.handle)
    }

    pub(crate) fn write(&mut self, data: &[u8]) {
        if !data.is_empty()
            && let Some(b) = self.buf.as_mut()
        {
            b.write(0, data);
        }
    }

    pub(crate) fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        if let Some(mut b) = self.buf.take() {
            b.destroy(device, allocator);
        }
        self.bytes = 0;
    }
}

/// Which views the slot's last submission culled (read back once it retires).
#[derive(Clone, Copy)]
struct Recorded {
    main: Option<u32>,
    shadow_views: (u32, u32),
    eye: bool,
}

/// One submission's cull buffers (a frame slot or a VR eye slot).
struct CullFrame {
    draws: GrowBuffer,
    candidates: GrowBuffer,
    batches: GrowBuffer,
    views: GrowBuffer,
    cmd_src: GrowBuffer,
    cmds: GrowBuffer,
    instances: GrowBuffer,
    /// Late pass: its commands / instances (main-view batches) and the early
    /// pass's per-candidate occluded flags. Minimal while Hi-Z is off.
    late_cmds: GrowBuffer,
    late_instances: GrowBuffer,
    occluded: GrowBuffer,
    counters: Buffer,
    readback: Buffer,
    set: vk::DescriptorSet,
    recorded: Option<Recorded>,
}

impl CullFrame {
    fn new(device: &ash::Device, allocator: &mut Allocator, set: vk::DescriptorSet) -> Result<Self> {
        let host = |name| GrowBuffer::new(vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, name);
        let indirect = |name| {
            GrowBuffer::new(
                vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly,
                name,
            )
        };
        let instances = |name| {
            GrowBuffer::new(
                vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::GpuOnly,
                name,
            )
        };
        Ok(Self {
            draws: host("cull-draws"),
            candidates: host("cull-candidates"),
            batches: host("cull-batches"),
            views: host("cull-views"),
            cmd_src: GrowBuffer::new(vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu, "cull-cmd-src"),
            cmds: indirect("cull-indirect"),
            instances: instances("cull-instances"),
            late_cmds: indirect("cull-late-indirect"),
            late_instances: instances("cull-late-instances"),
            occluded: GrowBuffer::new(vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, "cull-occluded"),
            counters: Buffer::new(
                device,
                allocator,
                COUNTER_BYTES,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::GpuOnly,
                "cull-counters",
            )?,
            readback: Buffer::new(
                device,
                allocator,
                COUNTER_BYTES,
                vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuToCpu,
                "cull-readback",
            )?,
            set,
            recorded: None,
        })
    }

    fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        for b in [
            &mut self.draws,
            &mut self.candidates,
            &mut self.batches,
            &mut self.views,
            &mut self.cmd_src,
            &mut self.cmds,
            &mut self.instances,
            &mut self.late_cmds,
            &mut self.late_instances,
            &mut self.occluded,
        ] {
            b.destroy(device, allocator);
        }
        self.counters.destroy(device, allocator);
        self.readback.destroy(device, allocator);
    }
}

/// Cull slot of VR eye slot `i`.
pub(crate) fn eye_slot(i: usize) -> usize {
    FRAMES_IN_FLIGHT + i
}

/// Mip count of a pyramid whose level 0 is `size`.
pub(crate) fn hiz_mip_count(size: [u32; 2]) -> u32 {
    (32 - size[0].max(size[1]).max(1).leading_zeros()).min(MAX_HIZ_MIPS as u32)
}

/// Level-0 size of the pyramid over a `src` sized depth buffer (half res).
pub(crate) fn hiz_base_size(src: [u32; 2]) -> [u32; 2] {
    [src[0].div_ceil(2).max(1), src[1].div_ceil(2).max(1)]
}

/// The max-depth pyramid of the main camera (R32F, GENERAL layout).
struct HiZ {
    image: vk::Image,
    allocation: Option<Allocation>,
    full_view: vk::ImageView,
    mip_views: Vec<vk::ImageView>,
    size: [u32; 2],
    mips: u32,
    /// Depth-buffer size it is built from.
    src: [u32; 2],
    /// Moved out of UNDEFINED (sampling it is valid).
    ready: bool,
}

impl HiZ {
    fn new(device: &ash::Device, allocator: &mut Allocator, src: [u32; 2]) -> Result<Self> {
        let size = hiz_base_size(src);
        let mips = hiz_mip_count(size);
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R32_SFLOAT)
            .extent(vk::Extent3D { width: size[0], height: size[1], depth: 1 })
            .mip_levels(mips)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.create_image(&info, None)? };
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let allocation = allocator.allocate(&AllocationCreateDesc {
            name: "hiz-pyramid",
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;
        unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset())? };
        let view = |base: u32, count: u32| {
            let info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R32_SFLOAT)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(base)
                        .level_count(count)
                        .layer_count(1),
                );
            unsafe { device.create_image_view(&info, None) }
        };
        let full_view = view(0, mips)?;
        let mip_views = (0..mips).map(|m| view(m, 1)).collect::<Result<Vec<_>, _>>()?;
        Ok(Self { image, allocation: Some(allocation), full_view, mip_views, size, mips, src, ready: false })
    }

    fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe {
            for v in self.mip_views.drain(..) {
                device.destroy_image_view(v, None);
            }
            device.destroy_image_view(self.full_view, None);
            device.destroy_image(self.image, None);
        }
        if let Some(a) = self.allocation.take() {
            let _ = allocator.free(a);
        }
    }

    fn range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(self.mips)
            .layer_count(1)
    }
}

/// This frame's late (second-phase) occlusion pass of the main camera.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Late {
    /// None: Hi-Z off, no main camera, or the early pass had no pyramid to
    /// test against (so it rejected nothing as occluded).
    Off,
    /// The early pass tested occlusion: the first main-camera depth pass runs
    /// the late pass after drawing the early batches.
    Pending,
    /// Recorded: the main camera's passes draw the late batches too.
    Done,
}

impl Late {
    /// State after the early pass was recorded.
    pub(crate) fn after_early(hiz_tested: bool) -> Self {
        if hiz_tested { Late::Pending } else { Late::Off }
    }

    /// Whether a pass drawing the early batches into `split` (a pass to
    /// split for the late pass, if any) runs the late pass after them.
    pub(crate) fn runs_in(self, split: bool) -> bool {
        self == Late::Pending && split
    }
}

/// The attachments of a main-camera pass the late pass can split: ended,
/// then resumed with LOAD after the late cull. Depth is attachment-optimal.
pub(crate) struct LateSplit {
    pub colors: Vec<vk::ImageView>,
    /// The depth image and its depth-aspect (sampled) view.
    pub depth: (vk::Image, vk::ImageView),
    /// The attachment view of the same image (depth + stencil aspects).
    pub depth_att: vk::ImageView,
    pub extent: vk::Extent2D,
    /// A VR multiview pass: the two eyes' depth layers feed two pyramids (see [`XrSplit`]).
    pub xr: Option<XrSplit>,
    /// A per-eye (single layer, non-multiview) VR pass: the eye's pyramid is built from `depth`.
    pub eye: Option<EyeSplit>,
}

/// What the late pass of a per-eye VR pass needs besides the [`LateSplit`] attachments.
pub(crate) struct EyeSplit {
    /// Eye index (0 left, 1 right): which pyramid.
    pub eye: usize,
    /// The eye-render slot (indexes the pyramids' descriptor sets).
    pub slot: usize,
    pub view_proj: Mat4,
    pub position: Vec3,
    pub forward: Vec3,
}

/// What the late pass of a VR stereo pass needs to split it (multiview, 2 layers).
pub(crate) struct XrSplit {
    pub depth_image: vk::Image,
    /// Depth-aspect (sampled) view of each eye's layer of the single-sample depth image.
    pub depth_layers: [vk::ImageView; 2],
    /// The colour attachment (the MSAA image when multisampled) and, with MSAA, the target it resolves into.
    pub color: vk::ImageView,
    pub resolve_color: Option<vk::ImageView>,
    /// The depth attachment (the MSAA depth when multisampled) and the single-sample depth view it resolves into.
    pub depth_att: vk::ImageView,
    pub resolve_depth: Option<vk::ImageView>,
    /// The pass keeps its depth (underwater fog samples it).
    pub keep_depth: bool,
    /// The eye-render slot (indexes the pyramids' descriptor sets).
    pub slot: usize,
    pub view_proj: [Mat4; 2],
    pub position: Vec3,
    pub forward: Vec3,
}

/// The two per-eye Hi-Z pyramids of the VR stereo pass, with their own build sets.
pub(crate) struct XrHiz {
    hz: [HiZ; 2],
    pool: vk::DescriptorPool,
    /// `[slot][eye][mip]`.
    sets: Vec<vk::DescriptorSet>,
    /// Frame serial, the eyes' view-projections, the centre position and forward of the last build.
    built: Option<(u64, [Mat4; 2], Vec3, Vec3)>,
    /// The same per eye for the single-layer per-eye path: frame serial, view-projection, position, forward.
    eye_built: [Option<(u64, Mat4, Vec3, Vec3)>; 2],
}

impl XrHiz {
    fn new(device: &ash::Device, allocator: &mut Allocator, layout: vk::DescriptorSetLayout, src: [u32; 2]) -> Result<Self> {
        let n = XR_SLOTS * 2 * MAX_HIZ_MIPS;
        let sizes = [
            vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: n as u32 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: n as u32 },
        ];
        let pool = unsafe { device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(n as u32).pool_sizes(&sizes), None)? };
        let layouts = vec![layout; n];
        let sets = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts))? };
        let hz = [HiZ::new(device, allocator, src)?, HiZ::new(device, allocator, src)?];
        Ok(Self { hz, pool, sets, built: None, eye_built: [None; 2] })
    }

    pub(crate) fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        for h in &mut self.hz {
            h.destroy(device, allocator);
        }
        unsafe { device.destroy_descriptor_pool(self.pool, None) };
    }
}

/// Build the max-depth pyramid `hz` from the depth view `src` (already readable):
/// one dispatch per mip, `sets` = this build's descriptor sets (one per mip).
#[allow(clippy::too_many_arguments)]
fn build_pyramid(device: &ash::Device, cb: vk::CommandBuffer, pipeline: vk::Pipeline, layout: vk::PipelineLayout, sampler: vk::Sampler, hz: &mut HiZ, sets: &[vk::DescriptorSet], src: vk::ImageView) {
    image_barrier_range(
        device, cb, hz.image, hz.range(),
        vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL,
        vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::empty(),
        vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_WRITE,
    );
    hz.ready = true;
    let mut src_size = hz.src;
    unsafe {
        device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline);
        for (mip, &set) in sets.iter().enumerate().take(hz.mips as usize) {
            let dst_size = [(hz.size[0] >> mip).max(1), (hz.size[1] >> mip).max(1)];
            let (view, image_layout) = if mip == 0 { (src, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) } else { (hz.mip_views[mip - 1], vk::ImageLayout::GENERAL) };
            let src_info = [vk::DescriptorImageInfo::default().sampler(sampler).image_view(view).image_layout(image_layout)];
            let dst_info = [vk::DescriptorImageInfo::default().image_view(hz.mip_views[mip]).image_layout(vk::ImageLayout::GENERAL)];
            let writes = [
                vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&src_info),
                vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&dst_info),
            ];
            device.update_descriptor_sets(&writes, &[]);
            device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[]);
            let push: [i32; 4] = [src_size[0] as i32, src_size[1] as i32, dst_size[0] as i32, dst_size[1] as i32];
            device.cmd_push_constants(cb, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&push));
            device.cmd_dispatch(cb, dst_size[0].div_ceil(8), dst_size[1].div_ceil(8), 1);
            let done = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)];
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&done));
            src_size = dst_size;
        }
    }
}

/// Main-view counts once the late pass folds in: early [tested, visible,
/// frustum, occluded] plus late [re-tested, recovered, -, still occluded]
/// give final [tested, visible, frustum, occluded] and the recovered count.
pub(crate) fn fold_late_counts(early: [u32; 4], late: [u32; 4]) -> ([u32; 4], u32) {
    let recovered = late[1].min(early[3]);
    ([early[0], early[1] + recovered, early[2], early[3] - recovered], recovered)
}

/// A GPU-culled camera batch, drawn by `record_scene_draws`.
pub(crate) struct SceneIndirect {
    material: usize,
    mesh: usize,
    push: PushData,
    /// Light mask covering every instance of the batch (main camera pass).
    mask: [u32; 4],
    batch: u32,
    count: u32,
    cmd_offset: u64,
    inst_offset: u64,
    /// World AABB of the instances' bounding spheres (None = unbounded).
    bounds: Option<(Vec3, Vec3)>,
}

/// One camera-batch key for GPU-cull scene draws. The material handle itself
/// is intentionally absent: variants that resolve to the same descriptor set
/// and push lanes can share an instanced batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SceneBatchKey {
    set: u64,
    mesh: usize,
    base_color: [u32; 4],
    emission: [u32; 4],
    params0: [u32; 4],
    params1: [u32; 4],
}

/// A GPU-culled shadow batch (one mesh in one shadow layer).
pub(crate) struct ShadowIndirect {
    mesh: usize,
    batch: u32,
    cmd_offset: u64,
    inst_offset: u64,
}

/// Last read-back cull counts (FRAMES_IN_FLIGHT submissions old).
#[derive(Clone, Copy, Default)]
struct CullCounts {
    main: [u32; 4],
    late: [u32; 4],
    shadow_visible: u32,
    eye_visible: u32,
}

pub(crate) struct GpuCull {
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    hiz_set_layout: vk::DescriptorSetLayout,
    hiz_pipeline_layout: vk::PipelineLayout,
    hiz_pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
    /// Frame slots `[0, FRAMES_IN_FLIGHT)`, then VR eye slots.
    frames: Vec<CullFrame>,
    /// Hi-Z build sets per frame slot and build (`HIZ_BUILDS`), one per mip.
    hiz_sets: Vec<vk::DescriptorSet>,
    sampler: vk::Sampler,
    hiz: HiZ,
    /// The pyramid's frame serial, view-projection, camera position + forward.
    hiz_built: Option<(u64, Mat4, Vec3, Vec3)>,
    builder: CullBuilder,
    key_map: HashMap<SceneBatchKey, usize>,
    mesh_map: HashMap<usize, usize>,
    /// This submission's camera batches (main view or a VR eye).
    scene: Vec<SceneIndirect>,
    /// This frame's shadow batches per atlas layer.
    shadow: Vec<(usize, Vec<ShadowIndirect>)>,
    cmd_buffer: vk::Buffer,
    inst_buffer: vk::Buffer,
    late_cmd_buffer: vk::Buffer,
    late_inst_buffer: vk::Buffer,
    /// The main camera's late pass this frame: its state, candidate count (the
    /// main view's candidates come first) and camera (`pyramid_camera`).
    late: Late,
    late_count: u32,
    late_cam: (Mat4, Vec3, Vec3),
    /// The pass the next `record_scene_draws` may split for the late pass.
    split: Option<LateSplit>,
    /// VR stereo Hi-Z (GEO-5): the per-eye pyramids, and whether this frame's cluster early cull tested them.
    xr_hiz: Option<XrHiz>,
    xr_tested: bool,
    /// Draws of `FrameInput::draws` the camera batches cover (the CPU part of
    /// the pass skips them).
    handled: Vec<bool>,
    /// The next `record_scene_draws` draws `scene` first.
    pending: bool,
    /// The main pass resolved into an opaque copy before its water: build the
    /// pyramid from that copy (water would hide what lies below it).
    pub hiz_from_copy: bool,
    counts: CullCounts,
    indirect_draws: u32,
}

pub(crate) fn compute_pipeline(device: &ash::Device, layout: vk::PipelineLayout, spv: &[u8]) -> Result<vk::Pipeline> {
    let code = ash::util::read_spv(&mut std::io::Cursor::new(spv))?;
    let module = unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)? };
    let pipeline = unsafe {
        device.create_compute_pipelines(
            vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::default()
                .stage(
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .module(module)
                        .name(c"main"),
                )
                .layout(layout)],
            None,
        )
    };
    unsafe { device.destroy_shader_module(module, None) };
    Ok(pipeline.map_err(|(_, e)| e)?[0])
}

impl GpuCull {
    pub fn new(device: &ash::Device, allocator: &mut Allocator) -> Result<Self> {
        let binding = |b: u32, ty: vk::DescriptorType| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(ty)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        };
        let cull_bindings: Vec<_> = CULL_BUFFER_BINDINGS
            .iter()
            .map(|&b| binding(b, vk::DescriptorType::STORAGE_BUFFER))
            .chain([binding(CULL_HIZ_BINDING, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)])
            .collect();
        let hiz_bindings =
            [binding(0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER), binding(1, vk::DescriptorType::STORAGE_IMAGE)];
        let set_layout = unsafe {
            device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&cull_bindings), None)?
        };
        let hiz_set_layout = unsafe {
            device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&hiz_bindings), None)?
        };
        let layout = |set: vk::DescriptorSetLayout, push: u32| {
            let sets = [set];
            let ranges = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(push)];
            unsafe {
                device.create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&sets).push_constant_ranges(&ranges),
                    None,
                )
            }
        };
        let pipeline_layout = layout(set_layout, size_of::<CullPush>() as u32)?;
        let hiz_pipeline_layout = layout(hiz_set_layout, 16)?;
        let pipeline = compute_pipeline(device, pipeline_layout, CULL_SPV)?;
        let hiz_pipeline = compute_pipeline(device, hiz_pipeline_layout, HIZ_SPV)?;

        let hiz_set_count = FRAMES_IN_FLIGHT * HIZ_BUILDS * MAX_HIZ_MIPS;
        let sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: (CULL_BUFFER_BINDINGS.len() * CULL_SLOTS) as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: (CULL_SLOTS + hiz_set_count) as u32,
            },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: hiz_set_count as u32 },
        ];
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets((CULL_SLOTS + hiz_set_count) as u32)
                    .pool_sizes(&sizes),
                None,
            )?
        };
        let alloc_sets = |layout: vk::DescriptorSetLayout, n: usize| {
            let layouts = vec![layout; n];
            unsafe {
                device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts),
                )
            }
        };
        let cull_sets = alloc_sets(set_layout, CULL_SLOTS)?;
        let hiz_sets = alloc_sets(hiz_set_layout, hiz_set_count)?;
        let frames = cull_sets
            .into_iter()
            .map(|set| CullFrame::new(device, allocator, set))
            .collect::<Result<Vec<_>>>()?;
        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .max_lod(vk::LOD_CLAMP_NONE),
                None,
            )?
        };
        let hiz = HiZ::new(device, allocator, [1, 1])?;
        Ok(Self {
            set_layout,
            pipeline_layout,
            pipeline,
            hiz_set_layout,
            hiz_pipeline_layout,
            hiz_pipeline,
            pool,
            frames,
            hiz_sets,
            sampler,
            hiz,
            hiz_built: None,
            builder: CullBuilder::default(),
            key_map: HashMap::new(),
            mesh_map: HashMap::new(),
            scene: Vec::new(),
            shadow: Vec::new(),
            cmd_buffer: vk::Buffer::null(),
            inst_buffer: vk::Buffer::null(),
            late_cmd_buffer: vk::Buffer::null(),
            late_inst_buffer: vk::Buffer::null(),
            late: Late::Off,
            late_count: 0,
            late_cam: (Mat4::IDENTITY, Vec3::ZERO, Vec3::ZERO),
            split: None,
            xr_hiz: None,
            xr_tested: false,
            handled: Vec::new(),
            pending: false,
            hiz_from_copy: false,
            counts: CullCounts::default(),
            indirect_draws: 0,
        })
    }

    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        for f in &mut self.frames {
            f.destroy(device, allocator);
        }
        self.hiz.destroy(device, allocator);
        if let Some(mut x) = self.xr_hiz.take() {
            x.destroy(device, allocator);
        }
        unsafe {
            device.destroy_sampler(self.sampler, None);
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline(self.hiz_pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_pipeline_layout(self.hiz_pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_descriptor_set_layout(self.hiz_set_layout, None);
        }
    }

    /// Slot `slot`'s previous submission has retired: fold its counters into
    /// the stats and drop the per-submission batch state.
    fn begin_slot(&mut self, slot: usize) {
        let f = &mut self.frames[slot];
        if let Some(rec) = f.recorded.take()
            && let Some(bytes) = f.readback.mapped()
            && bytes.len() >= COUNTER_BYTES as usize
        {
            let view = |v: u32| -> [u32; 4] { bytemuck::pod_read_unaligned(&bytes[v as usize * 16..v as usize * 16 + 16]) };
            if rec.eye {
                self.counts.eye_visible = rec.main.map_or(0, |v| view(v)[1]);
            } else {
                self.counts.main = rec.main.map_or([0; 4], view);
                // Zero when no late pass ran (cleared with the early block).
                self.counts.late = rec.main.map_or([0; 4], |v| view(v + MAX_CULL_VIEWS as u32));
                self.counts.shadow_visible = (rec.shadow_views.0..rec.shadow_views.1).map(|v| view(v)[1]).sum();
            }
        }
        self.scene.clear();
        self.shadow.clear();
        self.handled.clear();
        self.pending = false;
        self.hiz_from_copy = false;
        self.late = Late::Off;
        self.split = None;
        self.xr_tested = false;
    }

    /// Size this slot's buffers for the builder's lists, write them and point
    /// the slot's descriptor set at them. `late`: size the late pass's buffers
    /// for the main view (its candidates and instance regions come first).
    fn upload(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        slot: usize,
        instances: u32,
        late: bool,
    ) -> Result<()> {
        let b = &self.builder;
        let f = &mut self.frames[slot];
        let draws: &[u8] = bytemuck::cast_slice(&b.draws);
        let candidates: &[u8] = bytemuck::cast_slice(&b.candidates);
        let batches: &[u8] = bytemuck::cast_slice(&b.batches);
        let views: &[u8] = bytemuck::cast_slice(&b.views);
        let cmds: &[u8] = bytemuck::cast_slice(&b.cmds);
        f.draws.ensure(device, allocator, draws.len())?;
        f.candidates.ensure(device, allocator, candidates.len())?;
        f.batches.ensure(device, allocator, batches.len())?;
        f.views.ensure(device, allocator, views.len())?;
        f.cmd_src.ensure(device, allocator, cmds.len())?;
        f.cmds.ensure(device, allocator, cmds.len())?;
        f.instances.ensure(device, allocator, instances as usize * INSTANCE_STRIDE as usize)?;
        let late_count = if late { self.late_count as usize } else { 0 };
        f.late_cmds.ensure(device, allocator, if late { cmds.len() } else { 0 })?;
        f.late_instances.ensure(device, allocator, late_count * INSTANCE_STRIDE as usize)?;
        f.occluded.ensure(device, allocator, late_count * size_of::<u32>())?;
        f.draws.write(draws);
        f.candidates.write(candidates);
        f.batches.write(batches);
        f.views.write(views);
        f.cmd_src.write(cmds);

        // In CULL_BUFFER_BINDINGS order.
        let buffers = [
            f.draws.handle(),
            f.candidates.handle(),
            f.batches.handle(),
            f.views.handle(),
            f.cmds.handle(),
            f.instances.handle(),
            f.counters.handle,
            f.late_cmds.handle(),
            f.late_instances.handle(),
            f.occluded.handle(),
        ];
        let infos: Vec<[vk::DescriptorBufferInfo; 1]> =
            buffers.iter().map(|&b| [vk::DescriptorBufferInfo::default().buffer(b).range(vk::WHOLE_SIZE)]).collect();
        let image = [vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(self.hiz.full_view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let mut writes: Vec<vk::WriteDescriptorSet> = infos
            .iter()
            .zip(CULL_BUFFER_BINDINGS)
            .map(|(info, binding)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(f.set)
                    .dst_binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
            })
            .collect();
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(f.set)
                .dst_binding(CULL_HIZ_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&image),
        );
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        self.cmd_buffer = f.cmds.handle();
        self.inst_buffer = f.instances.handle();
        self.late_cmd_buffer = f.late_cmds.handle();
        self.late_inst_buffer = f.late_instances.handle();
        Ok(())
    }

    /// Record the (early) cull: reset the indirect commands (the late pass's
    /// too when `late`) + counters, dispatch, and make the results visible to
    /// indirect / vertex fetch (and copy the counters out for the next
    /// readback of this slot).
    fn record(&mut self, device: &ash::Device, cb: vk::CommandBuffer, slot: usize, rec: Recorded, late: bool) {
        let f = &mut self.frames[slot];
        let cmd_bytes = (self.builder.cmds.len() * size_of::<IndirectCmd>()) as u64;
        let push = CullPush { late_vp: Mat4::IDENTITY.to_cols_array_2d(), count: self.builder.candidates.len() as u32, late: 0 };
        unsafe {
            if !self.hiz.ready {
                image_barrier_range(
                    device, cb, self.hiz.image, self.hiz.range(),
                    vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL,
                    vk::PipelineStageFlags2::TOP_OF_PIPE, vk::AccessFlags2::empty(),
                    vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_SAMPLED_READ,
                );
                self.hiz.ready = true;
            }
            device.cmd_copy_buffer(
                cb,
                f.cmd_src.handle(),
                f.cmds.handle(),
                &[vk::BufferCopy::default().size(cmd_bytes)],
            );
            if late {
                device.cmd_copy_buffer(
                    cb,
                    f.cmd_src.handle(),
                    f.late_cmds.handle(),
                    &[vk::BufferCopy::default().size(cmd_bytes)],
                );
            }
            device.cmd_fill_buffer(cb, f.counters.handle, 0, COUNTER_BYTES, 0);
            // Also orders the previous frame's pyramid build before this read.
            let pre = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE | vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(
                    vk::AccessFlags2::SHADER_STORAGE_READ
                        | vk::AccessFlags2::SHADER_STORAGE_WRITE
                        | vk::AccessFlags2::SHADER_SAMPLED_READ,
                )];
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&pre));
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &[f.set], &[]);
            device.cmd_push_constants(cb, self.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&push));
            device.cmd_dispatch(cb, push.count.div_ceil(64), 1, 1);
            let post = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::DRAW_INDIRECT
                        | vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT
                        | vk::PipelineStageFlags2::TRANSFER,
                )
                .dst_access_mask(
                    vk::AccessFlags2::INDIRECT_COMMAND_READ
                        | vk::AccessFlags2::VERTEX_ATTRIBUTE_READ
                        | vk::AccessFlags2::TRANSFER_READ,
                )];
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&post));
            device.cmd_copy_buffer(
                cb,
                f.counters.handle,
                f.readback.handle,
                &[vk::BufferCopy::default().size(COUNTER_BYTES)],
            );
            let host = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                .dst_access_mask(vk::AccessFlags2::HOST_READ)];
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&host));
        }
        f.recorded = Some(rec);
    }

    /// Move the pyramid out of UNDEFINED so a shader may sample it (the cull
    /// passes do it on first use; the cluster pass can run without them).
    pub(crate) fn hiz_prepare(&mut self, device: &ash::Device, cb: vk::CommandBuffer) {
        if !self.hiz.ready {
            image_barrier_range(
                device, cb, self.hiz.image, self.hiz.range(),
                vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags2::TOP_OF_PIPE, vk::AccessFlags2::empty(),
                vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_SAMPLED_READ,
            );
            self.hiz.ready = true;
        }
    }

    /// Record the late pass (outside rendering, after this frame's pyramid
    /// was built): re-test the main-view candidates the early pass flagged
    /// occluded, fill the late commands / instances, and copy the late
    /// counters out after the early ones.
    fn record_late(&mut self, device: &ash::Device, cb: vk::CommandBuffer, slot: usize) {
        let f = &self.frames[slot];
        let push = CullPush { late_vp: self.late_cam.0.to_cols_array_2d(), count: self.late_count, late: 1 };
        unsafe {
            // The early pass's flags + its counter readback copy, the late
            // commands' reset copy and the pyramid build, before the late pass.
            let pre = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE | vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER)
                .dst_access_mask(
                    vk::AccessFlags2::SHADER_STORAGE_READ
                        | vk::AccessFlags2::SHADER_STORAGE_WRITE
                        | vk::AccessFlags2::SHADER_SAMPLED_READ
                        | vk::AccessFlags2::TRANSFER_WRITE,
                )];
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&pre));
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &[f.set], &[]);
            device.cmd_push_constants(cb, self.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&push));
            device.cmd_dispatch(cb, push.count.div_ceil(64), 1, 1);
            let post = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::DRAW_INDIRECT
                        | vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT
                        | vk::PipelineStageFlags2::TRANSFER,
                )
                .dst_access_mask(
                    vk::AccessFlags2::INDIRECT_COMMAND_READ
                        | vk::AccessFlags2::VERTEX_ATTRIBUTE_READ
                        | vk::AccessFlags2::TRANSFER_READ,
                )];
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&post));
            device.cmd_copy_buffer(
                cb,
                f.counters.handle,
                f.readback.handle,
                &[vk::BufferCopy::default().src_offset(LATE_COUNTERS).dst_offset(LATE_COUNTERS).size(LATE_COUNTERS)],
            );
            let host = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                .dst_access_mask(vk::AccessFlags2::HOST_READ)];
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&host));
        }
    }

    /// Record a pyramid build (outside rendering) into descriptor sets
    /// `build` of frame slot `frame`. `src`: the depth view to reduce;
    /// `depth_image`: its image when it is an attachment (DEPTH_ATTACHMENT_
    /// OPTIMAL, moved to SHADER_READ_ONLY_OPTIMAL and back), else it is
    /// already SHADER_READ_ONLY_OPTIMAL.
    fn record_pyramid(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        frame: usize,
        build: usize,
        src: vk::ImageView,
        depth_image: Option<vk::Image>,
    ) {
        let depth_range = vk::ImageSubresourceRange::default()
            .aspect_mask(crate::depth_aspect())
            .level_count(1)
            .layer_count(1);
        if let Some(image) = depth_image {
            image_barrier(
                device, cb, image, crate::depth_aspect(),
                crate::depth_attach_layout(), vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS, vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_SAMPLED_READ,
            );
        }
        unsafe {
            // The source's last write (depth tests, or the opaque-copy transfer)
            // before the compute read.
            let src_ready = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS | vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE | vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)];
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&src_ready));
        }
        // Rewritten whole: this frame's cull reads are the only prior use.
        image_barrier_range(
            device, cb, self.hiz.image, self.hiz.range(),
            vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL,
            vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::empty(),
            vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_WRITE,
        );
        self.hiz.ready = true;
        let first = (frame * HIZ_BUILDS + build) * MAX_HIZ_MIPS;
        let sets = &self.hiz_sets[first..first + MAX_HIZ_MIPS];
        let mut src_size = self.hiz.src;
        unsafe {
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.hiz_pipeline);
            for (mip, &set) in sets.iter().enumerate().take(self.hiz.mips as usize) {
                let dst_size = [(self.hiz.size[0] >> mip).max(1), (self.hiz.size[1] >> mip).max(1)];
                let (view, layout) = if mip == 0 {
                    (src, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                } else {
                    (self.hiz.mip_views[mip - 1], vk::ImageLayout::GENERAL)
                };
                let src_info = [vk::DescriptorImageInfo::default().sampler(self.sampler).image_view(view).image_layout(layout)];
                let dst_info = [vk::DescriptorImageInfo::default()
                    .image_view(self.hiz.mip_views[mip])
                    .image_layout(vk::ImageLayout::GENERAL)];
                let writes = [
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&src_info),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&dst_info),
                ];
                device.update_descriptor_sets(&writes, &[]);
                device.cmd_bind_descriptor_sets(
                    cb, vk::PipelineBindPoint::COMPUTE, self.hiz_pipeline_layout, 0, &[set], &[],
                );
                let push: [i32; 4] = [src_size[0] as i32, src_size[1] as i32, dst_size[0] as i32, dst_size[1] as i32];
                device.cmd_push_constants(
                    cb, self.hiz_pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&push),
                );
                device.cmd_dispatch(cb, dst_size[0].div_ceil(8), dst_size[1].div_ceil(8), 1);
                let done = [vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)];
                device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&done));
                src_size = dst_size;
            }
        }
        if let Some(image) = depth_image {
            image_barrier_range(
                device, cb, image, depth_range,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, crate::depth_attach_layout(),
                vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::empty(),
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            );
        }
    }

    /// This frame's camera can't leave a pyramid for the next frame's early
    /// pass. One the late pass built this frame (from the early batches'
    /// depth, this frame's camera) still serves.
    fn drop_next_pyramid(&mut self) {
        if self.late != Late::Done {
            self.hiz_built = None;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn image_barrier_range(
    device: &ash::Device,
    cb: vk::CommandBuffer,
    image: vk::Image,
    range: vk::ImageSubresourceRange,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
    src_stage: vk::PipelineStageFlags2,
    src_access: vk::AccessFlags2,
    dst_stage: vk::PipelineStageFlags2,
    dst_access: vk::AccessFlags2,
) {
    let barrier = [vk::ImageMemoryBarrier2::default()
        .image(image)
        .old_layout(old)
        .new_layout(new)
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .subresource_range(range)];
    unsafe { device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&barrier)) };
}

/// A pyramid's camera: view-projection, position, forward.
fn pyramid_camera(cam: &CameraData) -> (Mat4, Vec3, Vec3) {
    (cam.proj * cam.view, cam.position, -cam.view.row(2).truncate().normalize_or_zero())
}

/// Union of two world AABBs (None = unbounded, absorbing).
fn merge_bounds(a: Option<(Vec3, Vec3)>, b: Option<(Vec3, Vec3)>) -> Option<(Vec3, Vec3)> {
    let ((a0, a1), (b0, b1)) = (a?, b?);
    Some((a0.min(b0), a1.max(b1)))
}

/// A draw's world bounding sphere as an AABB (None when it has no bounds).
fn draw_bounds(d: &DrawCmd) -> Option<(Vec3, Vec3)> {
    (d.bound_radius > 0.0).then(|| {
        let s = [d.transform.x_axis, d.transform.y_axis, d.transform.z_axis]
            .iter()
            .map(|c| c.truncate().length())
            .fold(0.0f32, f32::max);
        let c = d.transform.transform_point3(d.mesh_center);
        let r = Vec3::splat(d.bound_radius * s);
        (c - r, c + r)
    })
}

impl Renderer {
    /// Set the GPU-driven culling options (takes effect next frame).
    pub fn set_gpu_cull_settings(&mut self, settings: GpuCullSettings) {
        self.gpu_cull_settings = settings;
    }

    pub fn gpu_cull_settings(&self) -> GpuCullSettings {
        self.gpu_cull_settings
    }

    /// Whether GPU culling can run (its compute pipelines built). When false
    /// the CPU path is used whatever the setting says.
    pub fn gpu_cull_supported(&self) -> bool {
        !self.gpu_cull_failed
    }

    /// Whether `d` can be drawn from a GPU-culled camera batch: opaque /
    /// alpha-test, standard shader, no material outline or fur shells, not water.
    fn gpu_eligible(&self, d: &DrawCmd) -> bool {
        let m = &self.materials[d.material.0];
        d.shadow_casting.visible()
            && !m.error
            && m.custom_shader.is_none()
            && m.outline.is_none()
            && m.fur.is_none()
            && !m.features.water
            // Scene Colour Read (grab pass, transmission): drawn after the opaque copy, which
            // is taken between the CPU-ordered draws, so these keep the CPU path.
            && !m.features.grab
            // Displaced materials need their tessellation stages, which the indirect batches
            // (and the cluster path behind them) do not have: they draw on the CPU path.
            && !m.params.tess.enabled
            && !matches!(m.features.alpha_mode, AlphaMode::Blend | AlphaMode::Transclipping)
            && m.render_queue < RENDER_QUEUE_TRANSPARENT
            // Stencil writers / readers keep the strict queue order of the CPU path.
            && m.features.stencil == crate::StencilState::default()
    }

    /// Whether the current GPU camera batches cover draw `i`.
    pub(crate) fn gpu_handles(&self, i: usize) -> bool {
        self.gpu_cull.as_ref().is_some_and(|g| g.handled.get(i).copied().unwrap_or(false))
    }

    /// `order` minus the draws the GPU camera batches cover.
    pub(crate) fn gpu_cpu_order<'a>(&self, order: &'a [usize]) -> std::borrow::Cow<'a, [usize]> {
        match self.gpu_cull.as_ref() {
            Some(g) if !g.handled.is_empty() => {
                order.iter().copied().filter(|&i| !g.handled.get(i).copied().unwrap_or(false)).collect()
            }
            _ => std::borrow::Cow::Borrowed(order),
        }
    }

    /// Arm (or disarm) the GPU camera batches for the next `record_scene_draws`.
    pub(crate) fn gpu_arm(&mut self, on: bool) {
        let clusters = self.cluster_armed();
        if let Some(g) = self.gpu_cull.as_mut() {
            g.pending = on && (!g.scene.is_empty() || clusters);
        }
    }

    /// The next `record_scene_draws` records into this main-camera pass (a
    /// depth-writing one, rendering begun): when the late pass is pending it
    /// splits the pass after the early batches to run it.
    pub(crate) fn gpu_late_split(&mut self, split: LateSplit) {
        if let Some(g) = self.gpu_cull.as_mut()
            && g.late == Late::Pending
        {
            g.split = Some(split);
        }
    }

    /// The split the scene draws did not use (nothing was armed for it), if any.
    pub(crate) fn gpu_take_split(&mut self) -> Option<LateSplit> {
        self.gpu_cull.as_mut().and_then(|g| g.split.take())
    }

    /// A VR stereo cull with Hi-Z recorded: the multiview pass must be split after its early draws.
    pub(crate) fn gpu_xr_split_pending(&self) -> bool {
        self.gpu_cull.as_ref().is_some_and(|g| g.late == Late::Pending && g.xr_hiz.is_some())
    }

    /// Whether this frame draws the main camera (game view, or the editor
    /// viewport's offscreen target).
    fn main_camera_drawn(&self, input: &FrameInput) -> bool {
        input.render_viewport || (input.viewport_extent.is_some() && self.viewport_target.is_some())
    }

    /// Create the culling pipelines on first use. False when unavailable.
    fn gpu_cull_ready(&mut self) -> bool {
        if !self.gpu_cull_settings.enabled || self.gpu_cull_failed {
            return false;
        }
        if self.gpu_cull.is_none() {
            let allocator = self.allocator();
            match GpuCull::new(&self.ctx.device, &mut allocator.lock().unwrap()) {
                Ok(g) => self.gpu_cull = Some(g),
                Err(e) => {
                    tracing::warn!("GPU culling unavailable, using the CPU path: {e:#}");
                    self.gpu_cull_failed = true;
                    return false;
                }
            }
        }
        true
    }

    /// Cull slot `slot` (a frame slot, or `FRAMES_IN_FLIGHT + eye slot`) has
    /// retired: read its counters back and reset the per-submission state.
    pub(crate) fn gpu_cull_begin(&mut self, slot: usize) {
        if slot < FRAMES_IN_FLIGHT {
            self.cluster_begin(slot);
        }
        if let Some(g) = self.gpu_cull.as_mut() {
            g.begin_slot(slot);
            if slot < FRAMES_IN_FLIGHT {
                g.indirect_draws = 0;
            }
        }
    }

    /// Group the camera-visible eligible draws of `order` into batches of view
    /// `view` (main camera / VR eye).
    fn gpu_build_scene(
        &self,
        g: &mut GpuCull,
        mut cl: Option<&mut ClusterSystem>,
        view: u32,
        order: &[usize],
        input: &FrameInput,
    ) {
        g.key_map.clear();
        g.handled.clear();
        g.handled.resize(input.draws.len(), false);
        let local_lights = self.cull_lights.iter().any(|l| l[3] >= 0.0);
        if view == 0 {
            self.draw_fallbacks.set([0; crate::REFUSAL_REASONS]);
        }
        for &i in order {
            let d = &input.draws[i];
            if d.occluded || !self.gpu_eligible(d) {
                continue;
            }
            let material = &self.materials[d.material.0];
            let mut push = self.draw_push(d, material, input);
            if !input.lightmap_preview {
                // The layer rides in each instance (GpuDraw::of): one batch
                // for all of the mesh's lightmapped copies.
                push.params1[3] = -1.0;
            }
            let key = SceneBatchKey {
                set: material.set.as_raw(),
                mesh: d.mesh.0,
                base_color: push.base_color.map(f32::to_bits),
                emission: push.emission.map(f32::to_bits),
                params0: push.params0.map(f32::to_bits),
                params1: push.params1.map(f32::to_bits),
            };
            // A mesh with an installed cluster DAG draws from clusters (not
            // lightmapped: the pulled vertices carry no second UV set). `fallback` names why a mesh
            // that has a DAG still draws ordinary this frame (the coverage view colours it by that).
            let mut fallback: Option<ClusterRefusal> = None;
            let on_clusters = match cl.as_deref_mut() {
                Some(c) => match c.store.record_of(d.mesh.0) {
                    None => false,
                    Some(id) => {
                        let skinned = c.store.is_skinned(id);
                        if d.lightmap_layer >= 0 {
                            fallback = Some(ClusterRefusal::LightmapUvs);
                        } else if self.skinner.as_ref().is_some_and(|s| s.cloth_pending(d.mesh.0)) {
                            fallback = Some(ClusterRefusal::ClothSim);
                        // A skinned DAG needs the mesh's latest pose; without one the ordinary path draws it.
                        } else if skinned && !self.skinner.as_ref().is_some_and(|s| s.pose_of(d.mesh.0).is_some()) {
                            fallback = Some(ClusterRefusal::NoPose);
                        // A full skinned-vertex pool sends the farthest skinned instances
                        // back to the ordinary skinned draw instead of pausing the path.
                        } else if skinned && !c.admission.allowed(d.mesh.0, (d.transform.transform_point3(d.mesh_center) - input.camera.position).length(), self.frame_serial) {
                            fallback = Some(ClusterRefusal::SkinPoolFull);
                        }
                        fallback.is_none()
                    }
                },
                None => false,
            };
            if let Some(f) = fallback
                && view == 0
            {
                let mut t = self.draw_fallbacks.get();
                t[f.index()] += 1;
                self.draw_fallbacks.set(t);
            }
            let (inst_color, inst_param) = self.instance_values(d, material, input, fallback);
            if on_clusters
                && let Some(cl) = cl.as_deref_mut()
                && let Some(id) = cl.store.record_of(d.mesh.0)
            {
                let batch = *cl.key_map.entry(SceneBatchKey { mesh: usize::MAX, ..key }).or_insert_with(|| {
                    cl.batches.push(ClusterBatch {
                        material: d.material.0,
                        push,
                        mask: [u32::MAX; 4],
                        bounds: draw_bounds(d),
                        bound: 0,
                        early_base: 0,
                        shadow_layer: None,
                    });
                    cl.batches.len() - 1
                });
                if cl.store.is_skinned(id)
                    && let Some(pose) = self.skinner.as_ref().and_then(|s| s.pose_of(d.mesh.0))
                {
                    let margin = cl.store.pose_margin(id, d.mesh_center.to_array(), d.bound_radius, &pose.cur, &pose.weights);
                    let sp = crate::cluster_skin::SkinPose {
                        palette: &pose.cur,
                        prev_palette: Some(&pose.prev),
                        weights: &pose.weights,
                        prev_weights: self.skinner.as_ref().and_then(|s| s.prev_weights_of(pose)),
                        margin,
                    };
                    cl.add_skinned_instance(i, id, batch as u32, d, inst_color, inst_param, &sp);
                } else {
                    cl.add_instance(i, id, batch as u32, d, inst_color, inst_param, input.camera.position);
                }
                if local_lights {
                    let b = &mut cl.batches[batch];
                    b.bounds = merge_bounds(b.bounds, draw_bounds(d));
                }
                cl.mark_main(i);
                g.handled[i] = true;
                continue;
            }
            let k = *g.key_map.entry(key).or_insert_with(|| {
                let batch = g.builder.add_batch(view, self.meshes[d.mesh.0].index_count);
                g.scene.push(SceneIndirect {
                    material: d.material.0,
                    mesh: d.mesh.0,
                    push,
                    mask: [u32::MAX; 4],
                    batch,
                    count: 0,
                    cmd_offset: 0,
                    inst_offset: 0,
                    bounds: draw_bounds(d),
                });
                g.scene.len() - 1
            });
            let s = &mut g.scene[k];
            g.builder.add_candidate(i, d, inst_color, inst_param, s.batch);
            if local_lights {
                s.bounds = merge_bounds(s.bounds, draw_bounds(d));
            }
            g.handled[i] = true;
        }
        for s in &mut g.scene {
            s.mask = if local_lights {
                sphere_light_mask(&self.cull_lights, s.bounds.map(|(lo, hi)| ((lo + hi) * 0.5, (hi - lo).length() * 0.5)))
            } else {
                [u32::MAX; 4]
            };
        }
        if let Some(cl) = cl {
            for b in &mut cl.batches {
                b.mask = if local_lights {
                    sphere_light_mask(&self.cull_lights, b.bounds.map(|(lo, hi)| ((lo + hi) * 0.5, (hi - lo).length() * 0.5)))
                } else {
                    [u32::MAX; 4]
                };
            }
        }
    }

    /// Fill in the batches' buffer offsets once the builder is finalized.
    fn gpu_place_batches(g: &mut GpuCull) {
        for s in &mut g.scene {
            s.cmd_offset = CullBuilder::cmd_offset(s.batch);
            s.inst_offset = g.builder.instance_offset(s.batch);
            s.count = g.builder.instance_count(s.batch);
        }
        for (_, batches) in &mut g.shadow {
            for b in batches {
                b.cmd_offset = CullBuilder::cmd_offset(b.batch);
                b.inst_offset = g.builder.instance_offset(b.batch);
            }
        }
    }

    /// Upload + record the built cull into slot `slot` (`late`: the main view
    /// tests Hi-Z, so a late pass follows). On failure the frame falls back to
    /// the CPU path.
    fn gpu_cull_submit(&mut self, g: &mut GpuCull, cb: vk::CommandBuffer, slot: usize, rec: Recorded, late: bool) {
        let instances = g.builder.finalize();
        Self::gpu_place_batches(g);
        let allocator = self.allocator();
        let uploaded = g.upload(&self.ctx.device, &mut allocator.lock().unwrap(), slot, instances, late);
        match uploaded {
            Ok(()) => {
                g.record(&self.ctx.device, cb, slot, rec, late);
                g.late = Late::after_early(late);
            }
            Err(e) => {
                tracing::warn!("GPU cull upload failed, CPU path this frame: {e:#}");
                g.scene.clear();
                g.shadow.clear();
                g.handled.clear();
            }
        }
    }

    /// Main-camera view, with the Hi-Z test when last frame's pyramid fits.
    fn gpu_main_view(&self, g: &GpuCull, input: &FrameInput, extent: [u32; 2]) -> GpuView {
        let cam = &input.camera;
        let view_proj = cam.proj * cam.view;
        let s = self.gpu_cull_settings;
        let mut v = GpuView::new(view_proj, cam.position, true);
        v.cam[3] = s.max_distance.max(0.0);
        // Perspective only: pixels per world unit at distance 1.
        if cam.proj.w_axis.w == 0.0 {
            v.lod[0] = s.min_pixel_radius.max(0.0);
            v.lod[1] = 0.5 * extent[1] as f32 * cam.proj.y_axis.y.abs();
        }
        let forward = -cam.view.row(2).truncate().normalize_or_zero();
        let hiz_ok = s.hiz_occlusion
            && g.hiz.src == extent
            && g.hiz_built.is_some_and(|(serial, _, pos, fwd)| {
                serial.wrapping_add(1) == self.frame_serial
                    && pos.distance(cam.position) <= HIZ_MAX_MOVE
                    && fwd.dot(forward) >= HIZ_MIN_FORWARD_DOT
            });
        if hiz_ok && let Some((_, vp, _, _)) = g.hiz_built {
            v.hiz_vp = vp.to_cols_array_2d();
            v.hiz = [g.hiz.size[0] as f32, g.hiz.size[1] as f32, g.hiz.mips as f32, 1.0];
        }
        v
    }

    /// Record this frame's cull (main camera + the shadow layers that
    /// re-render) into the frame command buffer, ahead of every pass that
    /// draws from it.
    pub(crate) fn gpu_cull_frame(
        &mut self,
        cb: vk::CommandBuffer,
        order: &[usize],
        input: &FrameInput,
        shadow_work: &[(usize, Mat4, Vec<usize>)],
        main_extent: [u32; 2],
    ) {
        if !self.gpu_cull_ready() {
            return;
        }
        let mut g = self.gpu_cull.take().unwrap();
        // The pyramid follows the main depth buffer's size. Resizes are rare
        // (and already idle the device), so wait rather than track retirement.
        if self.gpu_cull_settings.hiz_occlusion && g.hiz.src != main_extent {
            let device = self.ctx.device.clone();
            let allocator = self.allocator();
            let mut alloc = allocator.lock().unwrap();
            unsafe {
                let _ = device.wait_idle("wait:device_wait_idle@gpu_cull_frame");
            }
            match HiZ::new(&device, &mut alloc, main_extent) {
                Ok(h) => {
                    let mut old = std::mem::replace(&mut g.hiz, h);
                    old.destroy(&device, &mut alloc);
                }
                Err(e) => tracing::warn!("Hi-Z pyramid resize failed: {e:#}"),
            }
            g.hiz_built = None;
        }
        g.builder.begin(input.draws.len());
        g.builder.pack_lightmap = !input.lightmap_preview;
        let mut main = None;
        let mut late = false;
        // Cluster path for this frame's main camera (perspective, not VR).
        let mut cl = if self.cluster_frame_ok(input) { self.cluster.take() } else { None };
        if let Some(c) = cl.as_mut() {
            c.begin_build();
        } else {
            self.cluster_reset();
        }
        let mut main_view = None;
        if self.main_camera_drawn(input) {
            let v = self.gpu_main_view(&g, input, main_extent);
            main_view = Some(v);
            let view = g.builder.add_view(v);
            self.gpu_build_scene(&mut g, cl.as_mut(), view, order, input);
            main = Some(view);
            // The main view's candidates (and instance regions) come first.
            late = v.hiz[3] > 0.5 && (!g.builder.candidates.is_empty() || cl.as_ref().is_some_and(|c| c.has_instances()));
            g.late_count = g.builder.candidates.len() as u32;
            g.late_cam = pyramid_camera(&input.camera);
        }
        crate::hitch::phase("rr_cull_shadow_cpu");
        let first_shadow = g.builder.views.len() as u32;
        let shadow_px = self.cluster_settings.shadow_px;
        let shadow_dim = self.shadow_atlas.dim;
        for (layer, view_proj, casters) in shadow_work {
            if g.builder.views.len() >= MAX_CULL_VIEWS || casters.is_empty() {
                continue;
            }
            let view = g.builder.add_view(GpuView::new(*view_proj, Vec3::ZERO, false));
            g.mesh_map.clear();
            let mut batches = Vec::new();
            // Cluster shadows: casters with a DAG are culled and cut against
            // this light's view (coarser than the camera's) and drawn from the
            // visible list; the rest keep the ordinary batches below.
            if main_view.is_some()
                && shadow_px > 0.0
                && let Some(c) = cl.as_mut()
            {
                let mut slot: Option<(u32, u32)> = None;
                for &i in casters {
                    let d = &input.draws[i];
                    // Vertex-hook and displaced casters draw their own surface (the ordinary mesh).
                    if self.shadow_special(d) {
                        continue;
                    }
                    let Some(id) = c.store.record_of(d.mesh.0) else { continue };
                    if self.skinner.as_ref().is_some_and(|s| s.cloth_pending(d.mesh.0)) {
                        continue;
                    }
                    let pose = if c.store.is_skinned(id) {
                        // A skinned DAG needs the mesh's latest pose, as for the camera.
                        let Some(pose) = self.skinner.as_ref().and_then(|s| s.pose_of(d.mesh.0)) else { continue };
                        // Pushed out of the pool by `SkinAdmission`: the ordinary shadow draw.
                        if c.admission.is_denied(d.mesh.0) {
                            continue;
                        }
                        Some(pose)
                    } else {
                        None
                    };
                    let (cv, batch) = *slot.get_or_insert_with(|| {
                        let cv = c.add_shadow_view(*layer, crate::cluster::shadow_cull_view(*view_proj, shadow_dim, shadow_px));
                        (cv, c.add_shadow_batch(*layer))
                    });
                    match pose {
                        Some(pose) => {
                            let margin = c.store.pose_margin(id, d.mesh_center.to_array(), d.bound_radius, &pose.cur, &pose.weights);
                            let sp = crate::cluster_skin::SkinPose { palette: &pose.cur, prev_palette: Some(&pose.prev), weights: &pose.weights, prev_weights: self.skinner.as_ref().and_then(|s| s.prev_weights_of(pose)), margin };
                            c.add_shadow_instance(i, id, batch, cv, 1.0, d, Some(&sp));
                        }
                        None => c.add_shadow_instance(i, id, batch, cv, 1.0, d, None),
                    }
                    c.mark_shadow_handled(*layer, i);
                }
            }
            for &i in casters {
                if cl.as_ref().is_some_and(|c| c.shadow_is_handled(*layer, i)) {
                    continue;
                }
                let d = &input.draws[i];
                if self.shadow_special(d) {
                    continue;
                }
                let k = *g.mesh_map.entry(d.mesh.0).or_insert_with(|| {
                    let batch = g.builder.add_batch(view, self.meshes[d.mesh.0].index_count);
                    batches.push(ShadowIndirect { mesh: d.mesh.0, batch, cmd_offset: 0, inst_offset: 0 });
                    batches.len() - 1
                });
                g.builder.add_candidate(i, d, [1.0; 4], [0.0; 4], batches[k].batch);
            }
            g.shadow.push((*layer, batches));
        }
        crate::hitch::phase("rr_cull_submit");
        let rec = Recorded { main, shadow_views: (first_shadow, g.builder.views.len() as u32), eye: false };
        if !g.builder.candidates.is_empty() {
            self.gpu_zone_begin(cb, gpu_zone::CULL);
            self.gpu_cull_submit(&mut g, cb, self.frame_index, rec, late);
            self.gpu_zone_end(cb, gpu_zone::CULL);
        } else {
            g.scene.clear();
            g.shadow.clear();
        }
        crate::hitch::phase("rr_cluster_early");
        // The cluster cull follows the instance cull (it reads the same pyramid).
        if let Some(c) = cl {
            self.cluster = Some(c);
            if let Some(v) = main_view.filter(|_| self.cluster.as_ref().is_some_and(|c| c.has_instances())) {
                g.hiz_prepare(&self.ctx.device, cb);
                let mut cv = v;
                cv.lod[0] = 0.0;
                cv.lod[3] = self.cluster_settings.target_px.max(0.01);
                if g.builder.candidates.is_empty() {
                    g.late = Late::after_early(late);
                }
                if !self.cluster_record_early(cb, &cv, (g.sampler, g.hiz.full_view)) {
                    // Hand the draws back to the ordinary CPU path.
                    for i in self.cluster_take_draws() {
                        if let Some(h) = g.handled.get_mut(i) {
                            *h = false;
                        }
                    }
                    if g.builder.candidates.is_empty() {
                        g.late = Late::Off;
                    }
                }
            }
        }
        self.gpu_cull = Some(g);
    }

    /// Record a VR eye's cull (frustum only) into its command buffer. With
    /// `stereo` (the multiview pass of both eyes) the cluster path also runs:
    /// one cut judged from the centre eye and culled against the union of both
    /// eyes' frusta, drawn once for both eyes (no Hi-Z, no backface cone: the
    /// cone test is only valid for one eye position).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gpu_cull_eye(
        &mut self,
        cb: vk::CommandBuffer,
        slot: usize,
        order: &[usize],
        input: &FrameInput,
        view_proj: Mat4,
        cam_pos: Vec3,
        stereo: Option<(&crate::cluster_vr::StereoView, f32)>,
        extent: [u32; 2],
        eye_hiz: Option<(usize, Vec3)>,
    ) {
        let extent_height = extent[1] as f32;
        if !self.gpu_cull_ready() {
            return;
        }
        let mut g = self.gpu_cull.take().unwrap();
        g.builder.begin(input.draws.len());
        g.builder.pack_lightmap = !input.lightmap_preview;
        g.late = Late::Off;
        let view = g.builder.add_view(GpuView::new(view_proj, cam_pos, true));
        // The cluster path draws both shapes of VR frame: the multiview pass (one cut
        // for both eyes) and the per-eye pass (this eye's own cut, a plain camera).
        let mut cl = if self.cluster_xr_ok(input) { self.cluster.take() } else { None };
        if let Some(c) = cl.as_mut() {
            c.begin_build();
        } else {
            self.cluster_reset();
        }
        self.gpu_build_scene(&mut g, cl.as_mut(), view, order, input);
        if !g.builder.candidates.is_empty() {
            self.ctx.checkpoint(cb, c"GPU cull (VR eye)");
            let rec = Recorded { main: Some(view), shadow_views: (0, 0), eye: true };
            self.gpu_cull_submit(&mut g, cb, eye_slot(slot), rec, false);
        } else {
            g.scene.clear();
        }
        if let Some(mut c) = cl {
            // The cone test uses one camera position: off for stereo.
            if stereo.is_some() {
                for i in &mut c.instances {
                    i.eye[3] = 0.0;
                }
            }
            self.cluster = Some(c);
            if self.cluster.as_ref().is_some_and(|c| c.has_instances()) {
                g.hiz_prepare(&self.ctx.device, cb);
                let mut cv = match stereo {
                    Some((sv, height)) => crate::cluster::stereo_cull_view(sv, height, self.cluster_settings.target_px),
                    // One eye on its own: an ordinary camera (frustum, backface cone, no Hi-Z).
                    None => crate::cluster::eye_cull_view(view_proj, cam_pos, extent_height, self.cluster_settings.target_px),
                };
                let mut hiz_left = g.hiz.full_view;
                let mut xr_hiz_on = false;
                if let Some((sv, _)) = stereo
                    && self.gpu_cull_settings.hiz_occlusion
                    && self.xr_hiz_prepare(&mut g, cb, extent)
                {
                    // Two-phase Hi-Z for both eyes: each eye has its own pyramid; the early cull tests
                    // last XR frame's (when it still fits), the pass is split for the late cull.
                    xr_hiz_on = true;
                    let xr = g.xr_hiz.as_ref().expect("prepared");
                    hiz_left = xr.hz[0].full_view;
                    let forward = -sv.view[0].row(2).truncate().normalize_or_zero();
                    if let Some((serial, vps, pos, fwd)) = xr.built
                        && self.frame_serial.wrapping_sub(serial) <= 3
                        && pos.distance(cam_pos) <= HIZ_MAX_MOVE
                        && fwd.dot(forward) >= HIZ_MIN_FORWARD_DOT
                    {
                        let size = [xr.hz[0].size[0] as f32, xr.hz[0].size[1] as f32, xr.hz[0].mips as f32, 1.0];
                        cv.hiz_vp = vps[0].to_cols_array_2d();
                        cv.hiz = size;
                        let mut partner = GpuView::new(vps[1], Vec3::ZERO, false);
                        partner.hiz = size;
                        partner.hiz_vp = vps[1].to_cols_array_2d();
                        if let Some(c) = self.cluster.as_mut() {
                            c.stereo_hiz = Some(crate::cluster::StereoHiz { right: xr.hz[1].full_view, partner });
                        }
                        g.xr_tested = true;
                    }
                }
                // One eye on its own (the per-eye pass): that eye's pyramid, two-phase like the desktop's.
                let mut eye_hiz_on = false;
                if let (None, Some((eye, forward))) = (stereo, eye_hiz)
                    && self.gpu_cull_settings.hiz_occlusion
                    && eye < 2
                    && self.xr_hiz_prepare(&mut g, cb, extent)
                {
                    eye_hiz_on = true;
                    let xr = g.xr_hiz.as_ref().expect("prepared");
                    hiz_left = xr.hz[eye].full_view;
                    if let Some((serial, vp, pos, fwd)) = xr.eye_built[eye]
                        && self.frame_serial.wrapping_sub(serial) <= 3
                        && pos.distance(cam_pos) <= HIZ_MAX_MOVE
                        && fwd.dot(forward) >= HIZ_MIN_FORWARD_DOT
                    {
                        cv.hiz_vp = vp.to_cols_array_2d();
                        cv.hiz = [xr.hz[eye].size[0] as f32, xr.hz[eye].size[1] as f32, xr.hz[eye].mips as f32, 1.0];
                        g.xr_tested = true;
                    }
                    if let Some(c) = self.cluster.as_mut() {
                        c.stereo_hiz = None;
                    }
                }
                if self.cluster_record_early_in(cb, FRAMES_IN_FLIGHT + slot, &cv, (g.sampler, hiz_left)) {
                    if eye_hiz_on {
                        g.late = Late::Pending;
                    }
                    if xr_hiz_on {
                        // The pass splits after its early batches: pyramids from this frame's depth, the late cull.
                        g.late = Late::Pending;
                    }
                } else {
                    g.xr_tested = false;
                    if let Some(c) = self.cluster.as_mut() {
                        c.stereo_hiz = None;
                    }
                    // Hand the draws back to the ordinary CPU path.
                    for i in self.cluster_take_draws() {
                        if let Some(h) = g.handled.get_mut(i) {
                            *h = false;
                        }
                    }
                }
            }
        }
        self.gpu_cull = Some(g);
    }

    /// Make sure the per-eye pyramids exist at the eye image size (replacing ones of another size
    /// without waiting: the old ones go with the frames in flight) and are readable. False when
    /// the pass cannot run the two-phase test (no pipeline); the cull then tests no Hi-Z.
    fn xr_hiz_prepare(&mut self, g: &mut GpuCull, cb: vk::CommandBuffer, extent: [u32; 2]) -> bool {
        if g.xr_hiz.as_ref().is_some_and(|x| x.hz[0].src != extent)
            && let Some(mut old) = g.xr_hiz.take() {
                self.defer_free(move |device, alloc| old.destroy(device, alloc));
            }
        if g.xr_hiz.is_none() {
            let allocator = self.allocator();
            match XrHiz::new(&self.ctx.device, &mut allocator.lock().unwrap(), g.hiz_set_layout, extent) {
                Ok(x) => g.xr_hiz = Some(x),
                Err(e) => {
                    tracing::warn!("VR Hi-Z unavailable: {e:#}");
                    return false;
                }
            }
        }
        let x = g.xr_hiz.as_mut().expect("created");
        for h in &mut x.hz {
            if !h.ready {
                image_barrier_range(
                    &self.ctx.device, cb, h.image, h.range(),
                    vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL,
                    vk::PipelineStageFlags2::TOP_OF_PIPE, vk::AccessFlags2::empty(),
                    vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_SAMPLED_READ,
                );
                h.ready = true;
            }
        }
        true
    }

    /// The late pass of a VR stereo pass, inside the multiview pass `split` (rendering begun, the
    /// early batches drawn): end the pass, build both eyes' pyramids from their depth layers, run
    /// the late cull (re-test what the early pass found occluded in both pyramids), and resume the
    /// pass with LOAD. False when the late batches stay empty.
    pub(crate) fn gpu_cull_late_xr(&mut self, device: &ash::Device, cb: vk::CommandBuffer, split: &LateSplit) -> bool {
        let (Some(xs), Some(mut g)) = (split.xr.as_ref(), self.gpu_cull.take()) else { return false };
        let Some(mut xr) = g.xr_hiz.take() else {
            self.gpu_cull = Some(g);
            return false;
        };
        unsafe { device.cmd_end_rendering(cb) };
        // Both depth layers readable (a resolve writes in the colour output stage).
        crate::image_barrier_layers(
            device, cb, xs.depth_image, crate::depth_aspect(), 0, 2,
            crate::depth_attach_layout(), vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS | vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_SAMPLED_READ,
        );
        for e in 0..2 {
            let first = (xs.slot * 2 + e) * MAX_HIZ_MIPS;
            let sets = xr.sets[first..first + MAX_HIZ_MIPS].to_vec();
            build_pyramid(device, cb, g.hiz_pipeline, g.hiz_pipeline_layout, g.sampler, &mut xr.hz[e], &sets, xs.depth_layers[e]);
        }
        crate::image_barrier_layers(
            device, cb, xs.depth_image, crate::depth_aspect(), 0, 2,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, crate::depth_attach_layout(),
            vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::empty(),
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
        );
        let ran = g.xr_tested && self.cluster_record_late_xr(cb, FRAMES_IN_FLIGHT + xs.slot, xs.view_proj);
        xr.built = Some((self.frame_serial, xs.view_proj, xs.position, xs.forward));
        g.xr_hiz = Some(xr);
        g.late = if ran { Late::Done } else { Late::Off };
        // Resume with LOAD: the colour writes first (depth was ordered by the transition above).
        let color_barrier = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)];
        let mut color = vk::RenderingAttachmentInfo::default()
            .image_view(xs.color)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(if xs.resolve_color.is_some() { vk::AttachmentStoreOp::DONT_CARE } else { vk::AttachmentStoreOp::STORE });
        if let Some(r) = xs.resolve_color {
            color = color.resolve_mode(vk::ResolveModeFlags::AVERAGE).resolve_image_view(r).resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        }
        let mut depth = vk::RenderingAttachmentInfo::default()
            .image_view(xs.depth_att)
            .image_layout(crate::depth_attach_layout())
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(if xs.resolve_depth.is_none() && xs.keep_depth { vk::AttachmentStoreOp::STORE } else { vk::AttachmentStoreOp::DONT_CARE });
        if let (Some(r), true) = (xs.resolve_depth, xs.keep_depth) {
            depth = depth.resolve_mode(vk::ResolveModeFlags::SAMPLE_ZERO).resolve_image_view(r).resolve_image_layout(crate::depth_attach_layout());
        }
        let colors = [color];
        let stencil = crate::stencil_attachment(&depth);
        let area = vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: split.extent };
        let viewport = vk::Viewport { x: 0.0, y: split.extent.height as f32, width: split.extent.width as f32, height: -(split.extent.height as f32), min_depth: 0.0, max_depth: 1.0 };
        unsafe {
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&color_barrier));
            device.cmd_begin_rendering(
                cb,
                &vk::RenderingInfo::default().render_area(area).layer_count(1).view_mask(0b11).color_attachments(&colors).depth_attachment(&depth).stencil_attachment(&stencil),
            );
            device.cmd_set_viewport(cb, 0, &[viewport]);
            device.cmd_set_scissor(cb, 0, &[area]);
        }
        self.gpu_cull = Some(g);
        ran
    }

    /// The late pass of a per-eye VR pass `split` (one layer, no multiview): end the pass, build the eye's
    /// pyramid from its depth, re-test what the early cull found occluded, resume the pass with LOAD.
    pub(crate) fn gpu_cull_late_eye(&mut self, device: &ash::Device, cb: vk::CommandBuffer, split: &LateSplit) -> bool {
        let (Some(es), Some(mut g)) = (split.eye.as_ref(), self.gpu_cull.take()) else { return false };
        let Some(mut xr) = g.xr_hiz.take() else {
            self.gpu_cull = Some(g);
            return false;
        };
        unsafe { device.cmd_end_rendering(cb) };
        crate::image_barrier_layers(
            device, cb, split.depth.0, crate::depth_aspect(), 0, 1,
            crate::depth_attach_layout(), vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_SAMPLED_READ,
        );
        let first = (es.slot * 2 + es.eye) * MAX_HIZ_MIPS;
        let sets = xr.sets[first..first + MAX_HIZ_MIPS].to_vec();
        if std::env::var_os("DBG_NOPYR").is_none() { build_pyramid(device, cb, g.hiz_pipeline, g.hiz_pipeline_layout, g.sampler, &mut xr.hz[es.eye], &sets, split.depth.1); }
        crate::image_barrier_layers(
            device, cb, split.depth.0, crate::depth_aspect(), 0, 1,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, crate::depth_attach_layout(),
            vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::empty(),
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
        );
        let ran = g.xr_tested && std::env::var_os("DBG_NOLATE").is_none() && self.cluster_record_late_eye(cb, FRAMES_IN_FLIGHT + es.slot, es.view_proj);
        xr.eye_built[es.eye] = Some((self.frame_serial, es.view_proj, es.position, es.forward));
        g.xr_hiz = Some(xr);
        g.late = if ran { Late::Done } else { Late::Off };
        let color_barrier = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)];
        let load = |view: vk::ImageView, layout: vk::ImageLayout| {
            vk::RenderingAttachmentInfo::default().image_view(view).image_layout(layout).load_op(vk::AttachmentLoadOp::LOAD).store_op(vk::AttachmentStoreOp::STORE)
        };
        let colors: Vec<_> = split.colors.iter().map(|&v| load(v, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)).collect();
        let depth = load(split.depth_att, crate::depth_attach_layout());
        let stencil = crate::stencil_attachment(&depth);
        let area = vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: split.extent };
        let viewport = vk::Viewport { x: 0.0, y: split.extent.height as f32, width: split.extent.width as f32, height: -(split.extent.height as f32), min_depth: 0.0, max_depth: 1.0 };
        unsafe {
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&color_barrier));
            device.cmd_begin_rendering(
                cb,
                &vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&colors).depth_attachment(&depth).stencil_attachment(&stencil),
            );
            device.cmd_set_viewport(cb, 0, &[viewport]);
            device.cmd_set_scissor(cb, 0, &[area]);
        }
        self.gpu_cull = Some(g);
        ran
    }

    /// Draw the armed GPU camera batches (called by `record_scene_draws` before
    /// its CPU draws, so the pass's state comes from the same key logic):
    /// the early batches, then the late ones once the late pass has run. When
    /// it is pending and this pass was offered as its split, run it here
    /// between the two.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_gpu_scene_batches(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        frame_set: vk::DescriptorSet,
        gbuf: bool,
        bound: &mut BoundDraw,
        stats: &mut RenderStats,
        materials_seen: &mut [bool],
    ) -> Result<()> {
        let Some(g) = self.gpu_cull.as_mut() else {
            return Ok(());
        };
        let split = g.split.take();
        if !std::mem::take(&mut g.pending) {
            return Ok(());
        }
        let batches = std::mem::take(&mut g.scene);
        let early = (g.cmd_buffer, g.inst_buffer);
        let late = (g.late_cmd_buffer, g.late_inst_buffer);
        let late_state = g.late;
        // A VR stereo split has a late cluster half only (the eye's instance cull tests no Hi-Z).
        let xr_split = split.as_ref().is_some_and(|s| s.xr.is_some());
        let mut drawn = 0;
        let mut result =
            self.draw_gpu_batches(device, cb, frame_set, gbuf, &batches, early, bound, stats, materials_seen, &mut drawn);
        if result.is_ok() {
            result = self.draw_cluster_batches(device, cb, frame_set, gbuf, false, bound, stats, materials_seen);
        }
        if result.is_ok() {
            let draw_late = match split {
                Some(split) if late_state.runs_in(true) => {
                    let ran = if split.xr.is_some() {
                        self.gpu_cull_late_xr(device, cb, &split)
                    } else if split.eye.is_some() {
                        self.gpu_cull_late_eye(device, cb, &split)
                    } else {
                        self.gpu_cull_late(device, cb, &split)
                    };
                    // Resumed rendering: rebind everything.
                    *bound = BoundDraw::default();
                    ran
                }
                _ => late_state == Late::Done,
            };
            if draw_late {
                if !xr_split {
                    result = self.draw_gpu_batches(
                        device, cb, frame_set, gbuf, &batches, late, bound, stats, materials_seen, &mut drawn,
                    );
                }
                if result.is_ok() {
                    result = self.draw_cluster_batches(device, cb, frame_set, gbuf, true, bound, stats, materials_seen);
                }
            }
        }
        stats.indirect_draws += drawn;
        if let Some(g) = self.gpu_cull.as_mut() {
            g.scene = batches;
            g.indirect_draws += drawn;
        }
        result
    }

    /// One indirect draw per camera batch from `(commands, instances)`.
    #[allow(clippy::too_many_arguments)]
    fn draw_gpu_batches(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        frame_set: vk::DescriptorSet,
        gbuf: bool,
        batches: &[SceneIndirect],
        (cmds, insts): (vk::Buffer, vk::Buffer),
        bound: &mut BoundDraw,
        stats: &mut RenderStats,
        materials_seen: &mut [bool],
        drawn: &mut u32,
    ) -> Result<()> {
        for b in batches {
            let material = &self.materials[b.material];
            let key = scene_pipeline_key(material, gbuf, self.velocity_active, self.capturing_refl, self.draw_mode, false);
            let pipeline = self.pipeline_cache.get(device, key)?;
            let mut push = b.push;
            push.model[0] = if self.cull_active { b.mask } else { [u32::MAX; 4] }.map(f32::from_bits);
            if let Some(seen) = materials_seen.get_mut(b.material)
                && !*seen
            {
                *seen = true;
                stats.materials_drawn += 1;
            }
            let batch = SceneBatch {
                pipeline,
                material: b.material,
                mesh: b.mesh,
                push,
                first: 0,
                count: b.count,
                batchable: false,
                indirect: Some((cmds, b.cmd_offset, insts, b.inst_offset)),
            };
            self.flush_scene_batch(device, cb, frame_set, &batch, bound, stats);
            *drawn += 1;
        }
        Ok(())
    }

    /// The late pass, inside the main-camera pass `split` (rendering begun,
    /// the early batches drawn): end it, build this frame's pyramid from its
    /// depth, re-test the instances the early pass found occluded, and resume
    /// it (LOAD). False when it could not run (the late batches stay empty).
    fn gpu_cull_late(&mut self, device: &ash::Device, cb: vk::CommandBuffer, split: &LateSplit) -> bool {
        let Some(mut g) = self.gpu_cull.take() else {
            return false;
        };
        // The pyramid follows the main depth's size (the early pass only
        // tested occlusion when it matched).
        let fits = g.hiz.src == [split.extent.width, split.extent.height];
        unsafe { device.cmd_end_rendering(cb) };
        if fits {
            self.gpu_zone_begin(cb, gpu_zone::CULL_LATE);
            g.record_pyramid(device, cb, self.frame_index, HIZ_BUILD_LATE, split.depth.1, Some(split.depth.0));
            if g.late_count > 0 {
                g.record_late(device, cb, self.frame_index);
            }
            let (vp, _, _) = g.late_cam;
            self.cluster_record_late(cb, vp);
            self.gpu_zone_end(cb, gpu_zone::CULL_LATE);
            let (vp, pos, forward) = g.late_cam;
            g.hiz_built = Some((self.frame_serial, vp, pos, forward));
            g.late = Late::Done;
        } else {
            g.late = Late::Off;
        }
        // The pass's colour writes before it resumes with LOAD (depth was
        // ordered by the pyramid build's transitions).
        let color = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)];
        let load = |view: vk::ImageView, layout: vk::ImageLayout| {
            vk::RenderingAttachmentInfo::default()
                .image_view(view)
                .image_layout(layout)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE)
        };
        let colors: Vec<_> =
            split.colors.iter().map(|&v| load(v, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)).collect();
        let depth = load(split.depth_att, crate::depth_attach_layout());
        let stencil = crate::stencil_attachment(&depth);
        let area = vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: split.extent };
        let info = vk::RenderingInfo::default()
            .render_area(area)
            .layer_count(1)
            .color_attachments(&colors)
            .depth_attachment(&depth)
            .stencil_attachment(&stencil);
        // Negative-height viewport, as every main-camera pass.
        let viewport = vk::Viewport {
            x: 0.0,
            y: split.extent.height as f32,
            width: split.extent.width as f32,
            height: -(split.extent.height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        };
        unsafe {
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&color));
            device.cmd_begin_rendering(cb, &info);
            device.cmd_set_viewport(cb, 0, &[viewport]);
            device.cmd_set_scissor(cb, 0, &[area]);
        }
        let ran = g.late == Late::Done;
        self.gpu_cull = Some(g);
        ran
    }

    /// Record the GPU-culled casters of shadow layer `layer` (shadow pipeline
    /// bound, rendering begun). False when the layer has no GPU batches.
    pub(crate) fn record_gpu_shadow_layer(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        layer: usize,
        light_vp: Mat4,
    ) -> bool {
        let Some(g) = self.gpu_cull.as_mut() else {
            return false;
        };
        let Some(pos) = g.shadow.iter().position(|(l, _)| *l == layer) else {
            return false;
        };
        let (cmds, insts) = (g.cmd_buffer, g.inst_buffer);
        let batches = std::mem::take(&mut g.shadow[pos].1);
        let push = crate::ShadowPush { light_vp: light_vp.to_cols_array_2d(), model: Mat4::IDENTITY.to_cols_array_2d() };
        unsafe {
            device.cmd_push_constants(
                cb,
                self.pipeline_cache.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&push),
            );
            for b in &batches {
                let mesh = &self.meshes[b.mesh];
                device.cmd_bind_vertex_buffers(cb, 0, &[mesh.vertex_buffer.handle, insts], &[0, b.inst_offset]);
                device.cmd_bind_index_buffer(cb, mesh.index_buffer.handle, 0, mesh.index_type);
                device.cmd_draw_indexed_indirect(cb, cmds, b.cmd_offset, 1, INDIRECT_STRIDE);
            }
        }
        if let Some(g) = self.gpu_cull.as_mut() {
            g.indirect_draws += batches.len() as u32;
            g.shadow[pos].1 = batches;
        }
        true
    }

    /// Build next frame's Hi-Z pyramid from the main pass depth (recorded
    /// outside rendering). `depth` is the main depth target; `attachment` says
    /// it is still in DEPTH_ATTACHMENT_OPTIMAL (it is returned there), else it
    /// is SHADER_READ_ONLY_OPTIMAL. A pass that resolved an opaque copy before
    /// its water builds from that copy instead.
    pub(crate) fn gpu_hiz_build(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        input: &FrameInput,
        depth: (vk::Image, vk::ImageView),
        attachment: bool,
    ) {
        if !self.gpu_cull_settings.enabled || !self.gpu_cull_settings.hiz_occlusion || !self.main_camera_drawn(input) {
            return;
        }
        let Some(mut g) = self.gpu_cull.take() else {
            return;
        };
        let src = g.hiz.src;
        let copy = if g.hiz_from_copy { self.opaque_copy.as_ref().map(|c| (c.depth_view, c.extent)) } else { None };
        let (src_view, transition) = match copy {
            Some((v, e)) if [e.width, e.height] == src => (v, false),
            // A copy of another size doesn't hold this pass's depth.
            Some(_) => {
                g.drop_next_pyramid();
                self.gpu_cull = Some(g);
                return;
            }
            None => (depth.1, attachment),
        };
        if src[0] <= 1 && src[1] <= 1 {
            self.gpu_cull = Some(g);
            return;
        }
        self.gpu_zone_begin(cb, gpu_zone::HIZ);
        g.record_pyramid(device, cb, self.frame_index, HIZ_BUILD_NEXT, src_view, transition.then_some(depth.0));
        self.gpu_zone_end(cb, gpu_zone::HIZ);
        let (vp, pos, forward) = pyramid_camera(&input.camera);
        g.hiz_built = Some((self.frame_serial, vp, pos, forward));
        self.gpu_cull = Some(g);
    }

    /// The pyramid of this frame can't be built (e.g. the selection outline
    /// wiped the depth): next frame's early pass tests against the late
    /// pass's pyramid when one was built this frame, else culls without
    /// occlusion.
    pub(crate) fn gpu_hiz_invalidate(&mut self) {
        if let Some(g) = self.gpu_cull.as_mut() {
            g.drop_next_pyramid();
        }
    }

    /// Fold the GPU cull counters into this frame's stats.
    pub(crate) fn gpu_cull_stats(&self, stats: &mut RenderStats) {
        stats.gpu_cull_ms = self.gpu_zone_ms[gpu_zone::CULL] + self.gpu_zone_ms[gpu_zone::CULL_LATE];
        stats.gpu_hiz_ms = self.gpu_zone_ms[gpu_zone::HIZ];
        let Some(g) = self.gpu_cull.as_ref().filter(|_| self.gpu_cull_settings.enabled) else {
            return;
        };
        stats.gpu_culling = true;
        stats.indirect_draws = g.indirect_draws;
        let ([tested, visible, frustum, occluded], recovered) = fold_late_counts(g.counts.main, g.counts.late);
        stats.gpu_cull_tested = tested;
        stats.gpu_cull_visible = visible;
        stats.gpu_cull_frustum = frustum;
        stats.gpu_cull_occluded = occluded;
        stats.gpu_cull_recovered = recovered;
        stats.gpu_cull_shadow_visible = g.counts.shadow_visible;
        stats.gpu_cull_eye_visible = g.counts.eye_visible;
        stats.opaque_draws += visible;
    }
}

/// Source texels (inclusive range per axis) a pyramid texel `p` of a `dst`
/// sized level covers in its `src` sized source: every texel it overlaps in
/// UV space. Mirrors `hiz_build.comp`.
#[cfg(test)]
pub(crate) fn hiz_footprint(p: [u32; 2], src: [u32; 2], dst: [u32; 2]) -> ([u32; 2], [u32; 2]) {
    let lo = [0, 1].map(|a| p[a] * src[a] / dst[a]);
    let hi = [0, 1].map(|a| (((p[a] + 1) * src[a]).div_ceil(dst[a]) - 1).min(src[a] - 1));
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sphere_in_planes;
    use glam::Vec4;

    /// LCG for repeatable random cases.
    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 40) as f32 / (1u64 << 24) as f32
        }
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.f()
        }
    }

    fn draw(transform: Mat4, center: Vec3, radius: f32) -> DrawCmd {
        DrawCmd {
            gi_source: None,
            mesh: crate::MeshHandle(0),
            material: crate::MaterialHandle(0),
            transform,
            highlight: 0.0,
            mesh_center: center,
            bound_radius: radius,
            lightmap_layer: -1,
            lightmap_size: 0,
            static_geometry: false,
            planar_index: -1,
            occluded: false,
            shadow_casting: crate::ShadowCasting::On,
            instance_color: None,
            instance_param: None,
        }
    }

    /// `classify` of gpu_cull.comp without the Hi-Z branch.
    fn classify(v: &GpuView, d: &GpuDraw) -> u32 {
        if d.sphere[3] <= 0.0 {
            return 0;
        }
        let model = Mat4::from_cols_array_2d(&d.model);
        let c = model.transform_point3(Vec3::new(d.sphere[0], d.sphere[1], d.sphere[2]));
        let s = model.x_axis.truncate().length().max(model.y_axis.truncate().length()).max(model.z_axis.truncate().length());
        let r = d.sphere[3] * s;
        for k in 0..6 {
            if k == 4 && v.lod[2] < 0.5 {
                continue;
            }
            let p = Vec4::from_array(v.planes[k]);
            if p.truncate().dot(c) + p.w < -r {
                return 1;
            }
        }
        let dist = c.distance(Vec3::new(v.cam[0], v.cam[1], v.cam[2]));
        if v.cam[3] > 0.0 && dist - r > v.cam[3] {
            return 1;
        }
        if v.lod[0] > 0.0 && dist > r && r / dist * v.lod[1] < v.lod[0] {
            return 1;
        }
        0
    }

    fn random_draw(rng: &mut Rng) -> DrawCmd {
        let t = Mat4::from_scale_rotation_translation(
            Vec3::new(rng.range(0.2, 3.0), rng.range(0.2, 3.0), rng.range(0.2, 3.0)),
            glam::Quat::from_euler(glam::EulerRot::XYZ, rng.range(-3.0, 3.0), rng.range(-3.0, 3.0), rng.range(-3.0, 3.0)),
            Vec3::new(rng.range(-60.0, 60.0), rng.range(-20.0, 20.0), rng.range(-80.0, 40.0)),
        );
        let radius = if rng.f() < 0.05 { 0.0 } else { rng.range(0.05, 4.0) };
        draw(t, Vec3::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)), radius)
    }

    fn views() -> Vec<Mat4> {
        let view = Mat4::look_at_rh(Vec3::new(3.0, 4.0, 10.0), Vec3::new(0.0, 1.0, -20.0), Vec3::Y);
        vec![
            Mat4::perspective_rh(1.0, 16.0 / 9.0, 0.1, 200.0) * view,
            Mat4::perspective_rh(0.6, 1.0, 0.5, 60.0) * Mat4::look_at_rh(Vec3::ZERO, Vec3::new(1.0, -0.2, -1.0), Vec3::Y),
            Mat4::orthographic_rh(-30.0, 30.0, -30.0, 30.0, -50.0, 50.0)
                * Mat4::look_at_rh(Vec3::new(0.0, 20.0, 0.0), Vec3::new(0.3, 0.0, -0.2), Vec3::Y),
        ]
    }

    #[test]
    fn gpu_frustum_test_matches_cpu_cull() {
        let mut rng = Rng(7);
        let mut kept = [0usize; 2];
        for vp in views() {
            let planes = frustum_planes(vp);
            for near in [true, false] {
                let v = GpuView::new(vp, Vec3::ZERO, near);
                assert_eq!(v.planes, planes.map(|p| p.to_array()), "uploaded planes are the CPU planes");
                for _ in 0..4000 {
                    let d = random_draw(&mut rng);
                    let cpu = sphere_in_planes(&planes, &d, near);
                    let gpu = classify(&v, &GpuDraw::of(&d, false, [1.0; 4], [0.0; 4])) == 0;
                    assert_eq!(cpu, gpu, "near={near} draw={:?} r={}", d.transform, d.bound_radius);
                    kept[cpu as usize] += 1;
                }
            }
        }
        assert!(kept[0] > 100 && kept[1] > 100, "cases cover both outcomes: {kept:?}");
    }

    #[test]
    fn distance_and_size_hooks_only_cull_when_set() {
        let vp = views()[0];
        let mut v = GpuView::new(vp, Vec3::new(3.0, 4.0, 10.0), true);
        let near_big = GpuDraw::of(&draw(Mat4::from_translation(Vec3::new(0.0, 1.0, -5.0)), Vec3::ZERO, 1.0), false, [1.0; 4], [0.0; 4]);
        let far_small = GpuDraw::of(&draw(Mat4::from_translation(Vec3::new(0.0, 1.0, -150.0)), Vec3::ZERO, 0.05), false, [1.0; 4], [0.0; 4]);
        assert_eq!(classify(&v, &near_big), 0);
        assert_eq!(classify(&v, &far_small), 0);
        v.cam[3] = 100.0;
        assert_eq!(classify(&v, &near_big), 0);
        assert_eq!(classify(&v, &far_small), 1);
        v.cam[3] = 0.0;
        v.lod = [2.0, 540.0, 1.0, 0.0];
        assert_eq!(classify(&v, &near_big), 0);
        assert_eq!(classify(&v, &far_small), 1);
    }

    #[test]
    fn struct_layouts_match_std430_and_vulkan() {
        assert_eq!(std::mem::offset_of!(GpuView, hiz_vp), 96);
        assert_eq!(std::mem::offset_of!(GpuView, hiz), 160);
        assert_eq!(std::mem::offset_of!(GpuView, lod), 192);
        assert_eq!(std::mem::offset_of!(GpuDraw, sphere), 64);
        assert_eq!(std::mem::offset_of!(GpuDraw, color), 80);
        assert_eq!(std::mem::offset_of!(GpuDraw, param), 96);
        assert_eq!(std::mem::offset_of!(IndirectCmd, instance_count), 4);
        assert_eq!(std::mem::offset_of!(IndirectCmd, first_instance), 16);
        assert_eq!(INDIRECT_STRIDE, 20);
        assert_eq!(MAX_CULL_VIEWS * 4, 64, "one shared counter per thread of the 64-wide group");
        assert_eq!(std::mem::offset_of!(CullPush, count), 64);
        assert_eq!(std::mem::offset_of!(CullPush, late), 68);
        assert_eq!(LATE_COUNTERS, MAX_CULL_VIEWS as u64 * 16, "late block starts at counters[MAX_VIEWS * 4]");
        assert!(!CULL_BUFFER_BINDINGS.contains(&CULL_HIZ_BINDING));
    }

    #[test]
    fn late_counts_fold_into_the_main_view() {
        // Early: 100 tested, 60 visible, 25 frustum, 15 occluded; the late
        // pass re-tests the 15 and finds 4 visible now.
        let (main, recovered) = fold_late_counts([100, 60, 25, 15], [15, 4, 0, 11]);
        assert_eq!(main, [100, 64, 25, 11]);
        assert_eq!(recovered, 4);
        // No late pass (its block stays cleared): the early counts stand.
        assert_eq!(fold_late_counts([100, 60, 25, 15], [0; 4]), ([100, 60, 25, 15], 0));
    }

    #[test]
    fn late_pass_runs_once_in_the_offered_split() {
        assert_eq!(Late::after_early(false), Late::Off, "no pyramid tested: nothing to re-test");
        assert_eq!(Late::after_early(true), Late::Pending);
        assert!(Late::Pending.runs_in(true));
        assert!(!Late::Pending.runs_in(false), "a pass that can't be split leaves it pending");
        assert!(!Late::Done.runs_in(true), "the prepass ran it: the main pass only draws its batches");
        assert!(!Late::Off.runs_in(true));
    }

    #[test]
    fn main_view_candidates_lead_the_list() {
        // The late pass dispatches over candidates [0, main count) and writes
        // instances below the main count: the main view must come first.
        let mut b = CullBuilder::default();
        b.begin(4);
        let d: Vec<DrawCmd> = (0..4).map(|i| draw(Mat4::from_translation(Vec3::X * i as f32), Vec3::ZERO, 1.0)).collect();
        let main = b.add_view(GpuView::new(Mat4::IDENTITY, Vec3::ZERO, true));
        let m0 = b.add_batch(main, 36);
        let m1 = b.add_batch(main, 36);
        for (i, cmd) in d.iter().enumerate() {
            b.add_candidate(i, cmd, [1.0; 4], [0.0; 4], if i % 2 == 0 { m0 } else { m1 });
        }
        let main_count = b.candidates.len() as u32;
        let shadow = b.add_view(GpuView::new(Mat4::IDENTITY, Vec3::ZERO, false));
        let s = b.add_batch(shadow, 36);
        for (i, cmd) in d.iter().enumerate() {
            b.add_candidate(i, cmd, [1.0; 4], [0.0; 4], s);
        }
        b.finalize();
        for (k, c) in b.candidates.iter().enumerate() {
            let view = b.batches[c[1] as usize][0];
            assert_eq!(view == main, (k as u32) < main_count, "candidate {k}");
        }
        for batch in [m0, m1] {
            let [_, first, count, _] = b.batches[batch as usize];
            assert!(first + count <= main_count, "main regions fit the late instance buffer");
        }
    }

    #[test]
    fn batches_lay_out_indirect_commands_and_instance_regions() {
        let mut b = CullBuilder::default();
        b.begin(5);
        let d: Vec<DrawCmd> = (0..5).map(|i| draw(Mat4::from_translation(Vec3::X * i as f32), Vec3::ZERO, 1.0)).collect();
        let main = b.add_view(GpuView::new(Mat4::IDENTITY, Vec3::ZERO, true));
        let shadow = b.add_view(GpuView::new(Mat4::IDENTITY, Vec3::ZERO, false));
        let a = b.add_batch(main, 36);
        let c = b.add_batch(main, 600);
        let s = b.add_batch(shadow, 36);
        for i in [0, 2, 4] {
            b.add_candidate(i, &d[i], [1.0; 4], [0.0; 4], a);
        }
        b.add_candidate(1, &d[1], [1.0; 4], [0.0; 4], c);
        for i in [0, 1, 2, 3] {
            b.add_candidate(i, &d[i], [1.0; 4], [0.0; 4], s);
        }
        assert_eq!(b.finalize(), 8);
        // Each draw is uploaded once, whichever views reference it.
        assert_eq!(b.draws.len(), 5);
        assert_eq!(b.candidates.len(), 8);
        assert_eq!(b.batches, vec![[main, 0, 3, 0], [main, 3, 1, 0], [shadow, 4, 4, 0]]);
        assert_eq!(
            b.cmds[1],
            IndirectCmd { index_count: 600, instance_count: 0, first_index: 0, vertex_offset: 0, first_instance: 0 }
        );
        assert_eq!(CullBuilder::cmd_offset(c), 20);
        assert_eq!(b.instance_offset(s), 4 * INSTANCE_STRIDE as u64);
        assert_eq!(b.instance_count(s), 4);
        // Candidates name the uploaded slot, not the frame draw index.
        let slot_of_3 = b.draw_slot[3];
        assert!(b.candidates.contains(&[slot_of_3, s]));
        assert_eq!(b.draws[slot_of_3 as usize].model, d[3].transform.to_cols_array_2d());
        b.begin(2);
        assert!(b.draws.is_empty() && b.batches.is_empty() && b.draw_slot == vec![u32::MAX; 2]);
    }

    #[test]
    fn buffer_growth_only_grows() {
        assert_eq!(grow_capacity(0, 1), MIN_BUFFER_BYTES);
        assert_eq!(grow_capacity(MIN_BUFFER_BYTES, 100), MIN_BUFFER_BYTES);
        assert_eq!(grow_capacity(MIN_BUFFER_BYTES, MIN_BUFFER_BYTES + 1), MIN_BUFFER_BYTES * 2);
        let mut cap = 0;
        let mut reallocs = 0;
        for n in (0..2_000_000).step_by(997) {
            let next = grow_capacity(cap, n);
            assert!(next >= n && next >= cap);
            if next != cap {
                reallocs += 1;
            }
            cap = next;
        }
        assert!(reallocs <= 8, "{reallocs} reallocations");
    }

    #[test]
    fn hiz_footprints_cover_every_overlapping_texel() {
        for src in [[1, 1], [2, 2], [5, 3], [7, 9], [1920, 1080], [1001, 3]] {
            let mut dst = hiz_base_size(src);
            let mut s = src;
            for _ in 0..hiz_mip_count(dst) {
                for a in 0..2 {
                    for p in 0..dst[a] {
                        let (lo, hi) = hiz_footprint([p, p], [s[a], s[a]], [dst[a], dst[a]]);
                        let (lo, hi) = (lo[0], hi[0]);
                        assert!(hi - lo < 4, "footprint {lo}..={hi} of {p} ({s:?} -> {dst:?})");
                        // Texel p spans [p/dst, (p+1)/dst); source j spans [j/src, (j+1)/src).
                        for j in 0..s[a] {
                            let overlaps = (j as u64 * dst[a] as u64) < ((p + 1) as u64 * s[a] as u64)
                                && ((j + 1) as u64 * dst[a] as u64) > (p as u64 * s[a] as u64);
                            assert_eq!(overlaps, (lo..=hi).contains(&j), "src {j} vs dst {p} ({s:?} -> {dst:?})");
                        }
                    }
                }
                s = dst;
                dst = [(dst[0] / 2).max(1), (dst[1] / 2).max(1)];
            }
        }
        assert_eq!(hiz_mip_count([960, 540]), 10);
        assert_eq!(hiz_mip_count([1, 1]), 1);
    }

    /// Mirror of the pyramid build + `occluded` in gpu_cull.comp.
    struct Pyramid {
        levels: Vec<(Vec<f32>, [u32; 2])>,
    }

    impl Pyramid {
        fn build(depth: &[f32], src: [u32; 2]) -> Self {
            let base = hiz_base_size(src);
            let mut levels: Vec<(Vec<f32>, [u32; 2])> = Vec::new();
            for mip in 0..hiz_mip_count(base) {
                let dst = [(base[0] >> mip).max(1), (base[1] >> mip).max(1)];
                let (prev, s) = match levels.last() {
                    Some((d, s)) => (d.as_slice(), *s),
                    None => (depth, src),
                };
                let mut out = vec![0.0; (dst[0] * dst[1]) as usize];
                for y in 0..dst[1] {
                    for x in 0..dst[0] {
                        let (lo, hi) = hiz_footprint([x, y], s, dst);
                        let mut m = 0.0f32;
                        for sy in lo[1]..=hi[1] {
                            for sx in lo[0]..=hi[0] {
                                m = m.max(prev[(sy * s[0] + sx) as usize]);
                            }
                        }
                        out[(y * dst[0] + x) as usize] = m;
                    }
                }
                levels.push((out, dst));
            }
            Self { levels }
        }

        fn occluded(&self, vp: Mat4, c: Vec3, r: f32) -> bool {
            let (mut mn, mut mx, mut zmin) = (glam::Vec2::splat(1e30), glam::Vec2::splat(-1e30), 1e30f32);
            for k in 0..8 {
                let p = c + Vec3::new(
                    if k & 1 != 0 { r } else { -r },
                    if k & 2 != 0 { r } else { -r },
                    if k & 4 != 0 { r } else { -r },
                );
                let clip = vp * p.extend(1.0);
                if clip.w <= 1e-5 {
                    return false;
                }
                let ndc = clip.truncate() / clip.w;
                mn = mn.min(ndc.truncate());
                mx = mx.max(ndc.truncate());
                zmin = zmin.min(ndc.z);
            }
            if zmin <= 0.0 || mn.x < -1.0 || mn.y < -1.0 || mx.x > 1.0 || mx.y > 1.0 {
                return false;
            }
            let uv0 = glam::Vec2::new(mn.x * 0.5 + 0.5, 0.5 - mx.y * 0.5);
            let uv1 = glam::Vec2::new(mx.x * 0.5 + 0.5, 0.5 - mn.y * 0.5);
            let size0 = self.levels[0].1;
            let px = (uv1 - uv0) * glam::Vec2::new(size0[0] as f32, size0[1] as f32);
            let max_lod = self.levels.len() - 1;
            let mut lod = (px.x.max(px.y).max(1.0).log2().ceil() as usize).min(max_lod);
            let (a, b) = loop {
                let dim = self.levels[lod].1;
                let cl = |v: f32, n: u32| (v.floor() as i64).clamp(0, n as i64 - 1) as u32;
                let a = [cl(uv0.x * dim[0] as f32, dim[0]), cl(uv0.y * dim[1] as f32, dim[1])];
                let b = [cl(uv1.x * dim[0] as f32, dim[0]), cl(uv1.y * dim[1] as f32, dim[1])];
                if (b[0] - a[0] <= 1 && b[1] - a[1] <= 1) || lod >= max_lod {
                    break (a, b);
                }
                lod += 1;
            };
            let (data, dim) = &self.levels[lod];
            let at = |x: u32, y: u32| data[(y * dim[0] + x) as usize];
            let zmax = at(a[0], a[1]).max(at(b[0], a[1])).max(at(a[0], b[1])).max(at(b[0], b[1]));
            zmin > zmax
        }
    }

    #[test]
    fn hiz_occlusion_is_conservative() {
        let (w, h) = (161u32, 97u32);
        let proj = Mat4::perspective_rh(1.2, w as f32 / h as f32, 0.1, 100.0);
        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0), Vec3::Y);
        let vp = proj * view;
        let inv = vp.inverse();
        // A wall 10 m ahead covering the middle of the screen, sky elsewhere.
        let wall_z = 5.0 - 10.0;
        let mut depth = vec![1.0f32; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let ndc = glam::Vec2::new((x as f32 + 0.5) / w as f32 * 2.0 - 1.0, 1.0 - (y as f32 + 0.5) / h as f32 * 2.0);
                if ndc.x.abs() < 0.6 && ndc.y.abs() < 0.5 {
                    let near = inv.project_point3(ndc.extend(0.0));
                    let far = inv.project_point3(ndc.extend(1.0));
                    let t = (wall_z - near.z) / (far.z - near.z);
                    depth[(y * w + x) as usize] = vp.project_point3(near + (far - near) * t).z;
                }
            }
        }
        let pyramid = Pyramid::build(&depth, [w, h]);
        assert!(pyramid.occluded(vp, Vec3::new(0.0, 0.0, -20.0), 1.0), "hidden behind the wall");
        assert!(!pyramid.occluded(vp, Vec3::new(0.0, 0.0, -2.0), 1.0), "in front of the wall");
        assert!(!pyramid.occluded(vp, Vec3::new(22.0, 0.0, -20.0), 1.0), "beside the wall");
        assert!(!pyramid.occluded(vp, Vec3::new(0.0, 0.0, 4.95), 1.0), "crosses the near plane");

        // Never occludes a sphere with a visible point: sample its surface and
        // check each on-screen sample lies behind the depth buffer.
        let mut rng = Rng(11);
        let mut occluded = 0;
        for _ in 0..3000 {
            let c = Vec3::new(rng.range(-14.0, 14.0), rng.range(-9.0, 9.0), rng.range(-40.0, 3.0));
            let r = rng.range(0.05, 3.0);
            if !pyramid.occluded(vp, c, r) {
                continue;
            }
            occluded += 1;
            for i in 0..400 {
                let (u, v) = (rng.f() * std::f32::consts::TAU, rng.range(-1.0, 1.0));
                let s = (1.0 - v * v).sqrt();
                let p = c + Vec3::new(s * u.cos(), s * u.sin(), v) * r;
                let ndc = vp.project_point3(p);
                if ndc.x.abs() >= 1.0 || ndc.y.abs() >= 1.0 {
                    continue;
                }
                let x = (((ndc.x * 0.5 + 0.5) * w as f32) as u32).min(w - 1);
                let y = (((0.5 - ndc.y * 0.5) * h as f32) as u32).min(h - 1);
                assert!(ndc.z > depth[(y * w + x) as usize], "visible point {i} of occluded sphere {c} r={r}");
            }
        }
        assert!(occluded > 50, "only {occluded} random spheres were occluded");
    }

    /// Depth of an axis-aligned wall (plane z = `z`, |x| <= hx, |y| <= hy)
    /// rendered with `vp` into a `w` x `h` target, 1 elsewhere.
    fn wall_depth(vp: Mat4, w: u32, h: u32, z: f32, hx: f32, hy: f32) -> Vec<f32> {
        let inv = vp.inverse();
        let mut depth = vec![1.0f32; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let ndc = glam::Vec2::new((x as f32 + 0.5) / w as f32 * 2.0 - 1.0, 1.0 - (y as f32 + 0.5) / h as f32 * 2.0);
                let near = inv.project_point3(ndc.extend(0.0));
                let far = inv.project_point3(ndc.extend(1.0));
                let t = (z - near.z) / (far.z - near.z);
                let p = near + (far - near) * t;
                if (0.0..=1.0).contains(&t) && p.x.abs() <= hx && p.y.abs() <= hy {
                    depth[(y * w + x) as usize] = vp.project_point3(p).z;
                }
            }
        }
        depth
    }

    #[test]
    fn late_pass_draws_disoccluded_instances_the_same_frame() {
        // Two frames: the camera strafes 2 m right (within the early pass's
        // camera-cut gate) past a wall. Instance A was behind the wall last
        // frame and is in view now; B is behind it in both frames.
        // The wall is 2.5 m ahead, A and B 12.5 m: seen through the wall
        // plane, A moves from x = 0 (hidden) to 1.6 (past the 1.2 m edge) and B
        // from -0.8 to 0.8.
        let (w, h) = (321u32, 193u32);
        let proj = Mat4::perspective_rh(1.2, w as f32 / h as f32, 0.1, 100.0);
        let eye = |x: f32| Mat4::look_at_rh(Vec3::new(x, 0.0, 1.25), Vec3::new(x, 0.0, -1.0), Vec3::Y);
        let (prev_vp, vp) = (proj * eye(0.0), proj * eye(2.0));
        assert!(Vec3::new(0.0, 0.0, 1.25).distance(Vec3::new(2.0, 0.0, 1.25)) <= HIZ_MAX_MOVE);
        let wall = |vp| wall_depth(vp, w, h, -1.25, 1.2, 1.0);
        let (a, b, r) = (Vec3::new(0.0, 0.0, -11.25), Vec3::new(-4.0, 0.0, -11.25), 0.25);

        // Early pass: last frame's pyramid (its depth, its camera).
        let last = Pyramid::build(&wall(prev_vp), [w, h]);
        let early_occluded = |c| last.occluded(prev_vp, c, r);
        assert!(early_occluded(a), "a one-phase cull drops A this frame");
        assert!(early_occluded(b));

        // A really is in view this frame: its centre is nearer than the depth.
        let depth = wall(vp);
        let ndc = vp.project_point3(a);
        let (px, py) = (((ndc.x * 0.5 + 0.5) * w as f32) as u32, ((0.5 - ndc.y * 0.5) * h as f32) as u32);
        assert!(ndc.x.abs() < 1.0 && ndc.z < depth[(py * w + px) as usize], "A is visible this frame");

        // Late pass: this frame's pyramid from the early batches' depth (the
        // wall), this frame's camera, re-testing only the flagged instances.
        let now = Pyramid::build(&depth, [w, h]);
        assert!(!now.occluded(vp, a, r), "the late pass draws A the same frame");
        assert!(now.occluded(vp, b, r), "B stays culled");
    }

    #[test]
    fn batch_light_bounds_merge() {
        let a = draw_bounds(&draw(Mat4::from_translation(Vec3::X * 10.0), Vec3::ZERO, 1.0));
        let b = draw_bounds(&draw(Mat4::from_scale(Vec3::splat(2.0)), Vec3::Y, 0.5));
        let (lo, hi) = merge_bounds(a, b).unwrap();
        assert_eq!(lo, Vec3::new(-1.0, -1.0, -1.0));
        assert_eq!(hi, Vec3::new(11.0, 3.0, 1.0));
        assert!(merge_bounds(a, None).is_none(), "an unbounded instance makes the batch unbounded");
    }

    #[test]
    fn scene_batch_key_merges_material_variants_with_same_set_and_push() {
        let mut map: HashMap<SceneBatchKey, usize> = HashMap::new();
        let key_a = SceneBatchKey {
            set: 7,
            mesh: 3,
            base_color: [1, 2, 3, 4],
            emission: [5, 6, 7, 8],
            params0: [9, 10, 11, 12],
            params1: [13, 14, 15, 16],
        };
        let key_b = key_a;
        let slot_a = *map.entry(key_a).or_insert(0);
        let slot_b = *map.entry(key_b).or_insert(1);
        assert_eq!(slot_a, slot_b, "same set + mesh + push lanes should share one batch");
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn scene_batch_key_splits_when_set_or_push_differs() {
        let base = SceneBatchKey {
            set: 7,
            mesh: 3,
            base_color: [1, 2, 3, 4],
            emission: [5, 6, 7, 8],
            params0: [9, 10, 11, 12],
            params1: [13, 14, 15, 16],
        };
        let mut different_set = base;
        different_set.set = 8;
        let mut different_push = base;
        different_push.params1[0] ^= 1;

        let mut map: HashMap<SceneBatchKey, usize> = HashMap::new();
        map.insert(base, 0);
        map.insert(different_set, 1);
        map.insert(different_push, 2);
        assert_eq!(map.len(), 3, "set changes and push-lane changes must split batches");
    }

    #[test]
    fn lightmap_layers_ride_in_the_instance_matrix() {
        let mut d = draw(Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0)), Vec3::ZERO, 1.0);
        d.lightmap_layer = 4;
        let packed = GpuDraw::of(&d, true, [1.0; 4], [0.0; 4]);
        assert_eq!(packed.model[0][3], 5.0, "layer + 1 in col0.w");
        assert_eq!(packed.model[3], [1.0, 2.0, 3.0, 1.0], "the transform is otherwise untouched");
        assert_eq!(GpuDraw::of(&d, false, [1.0; 4], [0.0; 4]).model[0][3], 0.0, "lightmap preview keeps the push lane");
        d.lightmap_layer = -1;
        assert_eq!(GpuDraw::of(&d, true, [1.0; 4], [0.0; 4]).model[0][3], 0.0, "no lightmap: 0, the shader falls back to the push lane");
    }
}
