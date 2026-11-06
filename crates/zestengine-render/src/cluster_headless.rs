//! Headless harness for the cluster cull (GEO-4): runs the real
//! `cluster_cull.comp` (and the shared vertex-pulling decode) on a Vulkan
//! device with no surface, through the same `ClusterSystem` upload/record code
//! the renderer uses, and reads the visible lists back. Used by the tests in
//! this crate and in `zestengine-assets` (which can build real DAGs); every
//! test skips cleanly when `HeadlessGpu::new()` finds no Vulkan device (no
//! loader, no ICD such as a Mesa lavapipe, or no Vulkan 1.3 synchronization2).
//!
//! Out-of-range access is checked three ways: the shader bounds-checks every
//! index itself, the buffers the cull writes are pre-filled with a sentinel and
//! compared afterwards (a write outside a batch's region, a queue or the
//! occluded list shows up as a changed sentinel), and when the Khronos
//! validation layer is installed it is enabled and its error count returned.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::{Context as _, Result, bail};
use ash::vk;
use bytemuck::Zeroable;
use glam::{Mat4, Vec3};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use crate::alloc::{self, Buffer};
use crate::cluster::{
    ClusterBatch, ClusterMeshUpload, ClusterSystem, CullPlan, InstRec, PoolSizes, create_set_layout, record_early, record_late,
};
use crate::gpu_cull::GpuView;

const SENTINEL: u32 = 0xFFFF_FFFF;

/// One instance to cull.
#[derive(Clone, Debug)]
pub struct HeadlessInstance {
    /// Index into the meshes passed to `run`.
    pub mesh: usize,
    pub model: Mat4,
    pub batch: u32,
    /// A skinned instance's pose (the mesh must have been uploaded skinned).
    pub skin: Option<HeadlessSkin>,
    /// Per-instance LOD error scale (the budget's coarsening; 1 = none).
    pub lod_scale: f32,
}

/// The pose of a skinned headless instance.
#[derive(Clone, Debug, Default)]
pub struct HeadlessSkin {
    pub palette: Vec<Mat4>,
    pub prev_palette: Option<Vec<Mat4>>,
    pub weights: Vec<f32>,
    /// Last frame's blendshape weights (None = no blendshape motion).
    pub prev_weights: Option<Vec<f32>>,
    /// Pose margin; None = `cluster_skin::pose_margin` over the mesh's sphere.
    pub margin: Option<f32>,
}

/// The culled view: a perspective camera.
#[derive(Clone, Copy, Debug)]
pub struct HeadlessView {
    pub view_proj: Mat4,
    pub cam: Vec3,
    /// `0.5 * height_px * proj[1][1]`.
    pub pixels_per_unit: f32,
    pub target_px: f32,
}

/// The Hi-Z pyramid of a run: none, or a single texel (one mip) per pass.
#[derive(Clone, Copy, Debug)]
pub enum HeadlessHiz {
    Off,
    /// The early pass tests against a pyramid of the value `early`; the late
    /// pass (with `late_vp`) against a pyramid of the value `late`.
    Uniform { early: f32, late: f32, late_vp: Mat4 },
}

/// Knobs of a run.
#[derive(Clone, Copy, Debug, Default)]
pub struct HeadlessOptions {
    /// Skip the install-time validation (feed the shader corrupt data).
    pub unchecked: bool,
    /// Clamp the queue / list capacities to this many entries (forces overflows).
    pub cap_limit: Option<u32>,
    pub sizes: Option<PoolSizes>,
    /// Capacity of the per-frame skinned-vertex pool (forces a pool overflow).
    pub skin_pool_vertices: Option<u32>,
    /// Triangle budget fair share (estimated triangles; None = off) and the
    /// controller's global error scale.
    pub max_triangles: Option<u32>,
    pub global_scale: Option<f32>,
}

/// What a run produced.
#[derive(Debug, Default)]
pub struct HeadlessRun {
    pub plan: CullPlan,
    /// Per batch: the (instance, cluster) pairs of the early pass.
    pub early: Vec<Vec<(u32, u32)>>,
    /// Per batch: the late pass's pairs (recovered from Hi-Z).
    pub late: Vec<Vec<(u32, u32)>>,
    /// The shader's counters (`cluster::counter::*`).
    pub counters: Vec<u32>,
    /// Occluded-list entries the early pass queued.
    pub occluded: Vec<(u32, u32)>,
    /// Indirect commands: early then late, per batch.
    pub commands: Vec<[u32; 4]>,
    /// Sentinel words that changed where nothing may be written.
    pub out_of_range_writes: Vec<String>,
    /// Errors the validation layer reported during the run (0 without it).
    pub validation_errors: u32,
    /// The skin stage's control words (`cluster_skin::ctl`), the posed-vertex
    /// pool (16 floats per vertex: pos, normal, tangent, previous position)
    /// and each visible entry's pool base (`NO_SLOT` = none).
    pub skin_ctl: Vec<u32>,
    pub skin_pool: Vec<f32>,
    pub skin_slots: Vec<u32>,
    /// Per batch: first entry of its early and late region in the visible list
    /// (index into `skin_slots`).
    pub region_bases: Vec<[u32; 2]>,
    /// The per-instance error scale each instance was drawn with (the budget).
    pub inst_scales: Vec<f32>,
}

pub struct HeadlessGpu {
    _entry: ash::Entry,
    // The device handles are crate-visible so the other headless GPU tests
    // (skin_headless, gpu_tests) share this device setup.
    pub(crate) instance: ash::Instance,
    pub(crate) device: ash::Device,
    pub(crate) pd: vk::PhysicalDevice,
    pub(crate) queue: vk::Queue,
    pub(crate) pool: vk::CommandPool,
    // Read by the GPU timing tests.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) timestamp_period: f32,
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) timestamp_valid_bits: u32,
    pub(crate) multiview: bool,
    name: String,
    max_storage_range: u32,
    messenger: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    validation: bool,
    /// Held for the device's lifetime (dropped after `Drop::drop` destroys it).
    _lock: std::sync::MutexGuard<'static, ()>,
}

static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// One device test at a time: the whole test binary shares one GPU, and
/// concurrent device tests perturb each other's timing and memory. Poison
/// tolerant (a panicking test must not fail every later one). Not reentrant:
/// a test holds at most one `HeadlessGpu` / `gpu_lock()` guard at a time.
pub fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

pub(crate) static VALIDATION_ERRORS: AtomicU32 = AtomicU32::new(0);

unsafe extern "system" fn on_message(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _ty: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut c_void,
) -> vk::Bool32 {
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        VALIDATION_ERRORS.fetch_add(1, Ordering::Relaxed);
        let msg = unsafe { data.as_ref() }
            .and_then(|d| unsafe { d.message_as_c_str() })
            .map(|m| m.to_string_lossy().into_owned())
            .unwrap_or_default();
        eprintln!("[vulkan validation] {msg}");
    }
    vk::FALSE
}

impl HeadlessGpu {
    /// A device for compute work, or None when this machine has no usable
    /// Vulkan 1.3 device. `ZESTENGINE_TEST_GPU=0` forces None.
    pub fn new() -> Result<Option<Self>> {
        if std::env::var("ZESTENGINE_TEST_GPU").is_ok_and(|v| v == "0") {
            return Ok(None);
        }
        let lock = gpu_lock();
        let Ok(entry) = (unsafe { ash::Entry::load() }) else { return Ok(None) };
        let layer = c"VK_LAYER_KHRONOS_validation";
        let have_layer = unsafe { entry.enumerate_instance_layer_properties() }
            .map(|l| l.iter().any(|p| p.layer_name_as_c_str().is_ok_and(|n| n == layer)))
            .unwrap_or(false);
        let want_validation = have_layer && std::env::var("ZESTENGINE_TEST_NO_VALIDATION").is_err();
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let layers = [layer.as_ptr()];
        let exts = [ash::ext::debug_utils::NAME.as_ptr()];
        let mut info = vk::InstanceCreateInfo::default().application_info(&app);
        if want_validation {
            info = info.enabled_layer_names(&layers).enabled_extension_names(&exts);
        }
        let Ok(instance) = (unsafe { entry.create_instance(&info, None) }) else { return Ok(None) };
        let devices = unsafe { instance.enumerate_physical_devices() }.unwrap_or_default();
        let wanted = std::env::var("ZESTENGINE_TEST_GPU_INDEX").ok().and_then(|v| v.parse::<usize>().ok());
        let mut best: Option<(vk::PhysicalDevice, u32, u32)> = None;
        for (i, &pd) in devices.iter().enumerate() {
            let props = unsafe { instance.get_physical_device_properties(pd) };
            if props.api_version < vk::API_VERSION_1_3 {
                continue;
            }
            let Some(family) = unsafe { instance.get_physical_device_queue_family_properties(pd) }
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER))
            else {
                continue;
            };
            let rank = match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                vk::PhysicalDeviceType::CPU => 1,
                _ => 0,
            } + if wanted == Some(i) { 100 } else { 0 };
            if best.is_none_or(|b| rank > b.2) {
                best = Some((pd, family as u32, rank));
            }
        }
        let Some((pd, family, _)) = best else {
            unsafe { instance.destroy_instance(None) };
            return Ok(None);
        };
        let props = unsafe { instance.get_physical_device_properties(pd) };
        let name = props.device_name_as_c_str().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let prio = [1.0];
        let qi = [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&prio)];
        let mut vk13 = vk::PhysicalDeviceVulkan13Features::default().synchronization2(true).dynamic_rendering(true);
        // Multiview (the XR text / scene variants), only when the device has it.
        let mut have13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut have11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut query = vk::PhysicalDeviceFeatures2::default().push_next(&mut have11).push_next(&mut have13);
        unsafe { instance.get_physical_device_features2(pd, &mut query) };
        let mut vk11 = vk::PhysicalDeviceVulkan11Features::default()
            .multiview(have11.multiview == vk::TRUE)
            .multiview_tessellation_shader(have11.multiview_tessellation_shader == vk::TRUE);
        // Tessellation (and multiview tessellation above) are switched on when the device has them, so a
        // headless test can build a displaced material's pipelines.
        let tessellation = unsafe { instance.get_physical_device_features(pd) }.tessellation_shader == vk::TRUE;
        // Standard fragment variants use primitive IDs and demote-to-helper.
        vk13.shader_demote_to_helper_invocation = have13.shader_demote_to_helper_invocation;
        let features = vk::PhysicalDeviceFeatures::default().tessellation_shader(tessellation)
            .geometry_shader(unsafe { instance.get_physical_device_features(pd) }.geometry_shader == vk::TRUE);
        let device = unsafe {
            instance.create_device(
                pd,
                &vk::DeviceCreateInfo::default().queue_create_infos(&qi).enabled_features(&features).push_next(&mut vk13).push_next(&mut vk11),
                None,
            )
        }
        .context("creating the headless device")?;
        let ts_valid_bits = unsafe { instance.get_physical_device_queue_family_properties(pd) }[family as usize].timestamp_valid_bits;
        let queue = unsafe { device.get_device_queue(family, 0) };
        let pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?
        };
        let messenger = if want_validation {
            let du = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let ci = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING)
                .message_type(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION | vk::DebugUtilsMessageTypeFlagsEXT::GENERAL)
                .pfn_user_callback(Some(on_message));
            unsafe { du.create_debug_utils_messenger(&ci, None) }.ok().map(|m| (du, m))
        } else {
            None
        };
        Ok(Some(Self {
            _entry: entry,
            instance,
            device,
            pd,
            queue,
            pool,
            timestamp_period: props.limits.timestamp_period,
            timestamp_valid_bits: ts_valid_bits,
            multiview: have11.multiview == vk::TRUE,
            name,
            max_storage_range: props.limits.max_storage_buffer_range,
            messenger,
            validation: want_validation,
            _lock: lock,
        }))
    }

    pub fn device_name(&self) -> &str {
        &self.name
    }

    pub fn validation_enabled(&self) -> bool {
        self.validation
    }

    /// The scene depth(-stencil) format the renderer would pick on this device
    /// (`D32_SFLOAT_S8_UINT` when available; `D32_SFLOAT` under `ZESTENGINE_STENCIL=0`).
    #[cfg(test)]
    pub(crate) fn depth_stencil_format(&self) -> vk::Format {
        crate::context::pick_depth_format(&self.instance, self.pd)
    }

    pub(crate) fn allocator(&self) -> Result<Allocator> {
        Ok(Allocator::new(&AllocatorCreateDesc {
            instance: self.instance.clone(),
            device: self.device.clone(),
            physical_device: self.pd,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })?)
    }

    /// A 1x1 R32F pyramid in GENERAL layout, and a nearest sampler for it.
    fn hiz(&self, alloc: &mut Allocator) -> Result<HizImage> {
        let device = &self.device;
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R32_SFLOAT)
            .extent(vk::Extent3D { width: 1, height: 1, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.create_image(&info, None)? };
        let req = unsafe { device.get_image_memory_requirements(image) };
        let allocation = alloc.allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "headless-hiz",
            requirements: req,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        })?;
        unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset())? };
        let range = vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
        let view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R32_SFLOAT)
                    .subresource_range(range),
                None,
            )?
        };
        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?
        };
        alloc::one_time_submit(device, self.pool, self.queue, |cb| unsafe {
            let b = [vk::ImageMemoryBarrier2::default()
                .image(image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
                .subresource_range(range)];
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&b));
        })?;
        Ok(HizImage { image, view, sampler, allocation: Some(allocation), range })
    }

    /// Cull `instances` of `meshes` and read everything back.
    pub fn run(
        &self,
        meshes: &[ClusterMeshUpload],
        instances: &[HeadlessInstance],
        batches: u32,
        view: &HeadlessView,
        hiz: HeadlessHiz,
        opts: &HeadlessOptions,
    ) -> Result<HeadlessRun> {
        let device = &self.device;
        let errors_before = VALIDATION_ERRORS.load(Ordering::Relaxed);
        let mut alloc = self.allocator()?;
        let set_layout = create_set_layout(device)?;
        let sizes = opts.sizes.unwrap_or_default().clamped(self.max_storage_range);
        let mut sys = ClusterSystem::new(device, &mut alloc, set_layout, sizes)?;
        sys.cap_limit = opts.cap_limit;
        if let Some(v) = opts.skin_pool_vertices {
            sys.skin.pool_vertices = v;
        }
        sys.max_triangles = opts.max_triangles.unwrap_or(0);
        sys.global_scale = opts.global_scale.unwrap_or(1.0);
        let mut ids = Vec::new();
        for (i, m) in meshes.iter().enumerate() {
            ids.push(if opts.unchecked {
                sys.store.install_unchecked(device, &mut alloc, self.pool, self.queue, i, m)?
            } else {
                sys.store.install(device, &mut alloc, self.pool, self.queue, i, m)?
            });
        }
        let spheres: Vec<[f32; 4]> = meshes.iter().map(mesh_sphere).collect();
        sys.begin_build();
        sys.batches = (0..batches)
            .map(|_| ClusterBatch {
                material: 0,
                push: bytemuck::Zeroable::zeroed(),
                mask: [u32::MAX; 4],
                bounds: None,
                bound: 0,
                early_base: 0,
                shadow_layer: None,
            })
            .collect();
        let reach: Vec<Vec<f32>> = meshes.iter().map(|m| m.morph_reach()).collect();
        for (n, inst) in instances.iter().enumerate() {
            let rec = InstRec {
                model: inst.model.to_cols_array_2d(),
                color: [1.0; 4],
                param: [0.0; 4],
                sphere: spheres[inst.mesh],
                mesh_batch: [ids[inst.mesh], inst.batch, 0, 0],
                eye: crate::cluster_ref::object_space_eye(inst.model, view.cam),
                extra: [inst.lod_scale.max(1.0), 0.0, 0.0, 0.0],
            };
            match &inst.skin {
                None => sys.add_raw_instance(rec, inst.batch, n),
                Some(sk) => {
                    let sph = spheres[inst.mesh];
                    let margin = sk.margin.unwrap_or_else(|| {
                        crate::cluster_skin::pose_margin([sph[0], sph[1], sph[2]], sph[3], &sk.palette, &sk.weights, &reach[inst.mesh])
                    });
                    let pose = crate::cluster_skin::SkinPose {
                        palette: &sk.palette,
                        prev_palette: sk.prev_palette.as_deref(),
                        weights: &sk.weights,
                        prev_weights: sk.prev_weights.as_deref(),
                        margin,
                    };
                    sys.add_raw_skinned_instance(rec, inst.batch, n, &pose);
                }
            }
        }
        let mut gv = GpuView::new(view.view_proj, view.cam, true);
        gv.lod[1] = view.pixels_per_unit;
        gv.lod[3] = view.target_px;
        let hz = self.hiz(&mut alloc)?;
        if let HeadlessHiz::Uniform { .. } = hiz {
            gv.hiz_vp = view.view_proj.to_cols_array_2d();
            gv.hiz = [1.0, 1.0, 1.0, 1.0];
        }
        let plan = sys.upload_frame(device, &mut alloc, 0, &gv, (hz.sampler, hz.view))?;
        let targets = sys.targets(0);
        let outs = sys.frame_outputs(0);
        let mut hosts = Vec::new();
        for (i, (_, bytes)) in outs.iter().enumerate() {
            hosts.push(Buffer::new(
                device,
                &mut alloc,
                (*bytes).max(4) as u64,
                vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuToCpu,
                ["h-visible", "h-cmds", "h-qa", "h-qb", "h-occ", "h-skin-pool", "h-skin-slots"][i],
            )?);
        }
        let clear = |cb: vk::CommandBuffer, v: f32| unsafe {
            device.cmd_clear_color_image(
                cb,
                hz.image,
                vk::ImageLayout::GENERAL,
                &vk::ClearColorValue { float32: [v, 0.0, 0.0, 0.0] },
                &[hz.range],
            );
        };
        alloc::one_time_submit(device, self.pool, self.queue, |cb| unsafe {
            // Sentinel-fill what the cull writes (never the commands: finalize writes all of them).
            for (i, (buf, bytes)) in outs.iter().enumerate() {
                if i != 1 {
                    device.cmd_fill_buffer(cb, *buf, 0, (*bytes as u64) & !3, SENTINEL);
                }
            }
            if let HeadlessHiz::Uniform { early, .. } = hiz {
                clear(cb, early);
            }
            record_early(device, cb, &sys.pipes, &targets, &plan);
            if let HeadlessHiz::Uniform { late, late_vp, .. } = hiz {
                clear(cb, late);
                record_late(device, cb, &sys.pipes, &targets, &plan, late_vp);
            }
            // Early outputs are final once the late pass has run (or there is none).
            for (i, (buf, bytes)) in outs.iter().enumerate() {
                device.cmd_copy_buffer(cb, *buf, hosts[i].handle, &[vk::BufferCopy::default().size(*bytes as u64)]);
            }
            let b = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                .dst_access_mask(vk::AccessFlags2::HOST_READ)];
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&b));
        })?;

        let words = |b: &Buffer, bytes: usize| -> Vec<u32> {
            bytemuck::pod_collect_to_vec(b.mapped().map(|m| &m[..bytes & !3]).unwrap_or(&[]))
        };
        let visible = words(&hosts[0], outs[0].1);
        let cmds = words(&hosts[1], outs[1].1);
        let qa = words(&hosts[2], outs[2].1);
        let qb = words(&hosts[3], outs[3].1);
        let occ = words(&hosts[4], outs[4].1);
        let skin_words = words(&hosts[5], outs[5].1);
        let skin_slots = words(&hosts[6], outs[6].1);
        let counters: Vec<u32> = sys.read_counters(0);
        let nb = plan.batches as usize;
        let mut run = HeadlessRun { plan, counters, ..Default::default() };
        run.inst_scales = sys.instances.iter().map(|i| i.extra[0]).collect();
        run.region_bases = (0..plan.batches as usize).map(|b| [sys.batch_rec(b)[0], sys.batch_rec(b)[2]]).collect();
        run.commands = (0..2 * nb).map(|i| [cmds[i * 4], cmds[i * 4 + 1], cmds[i * 4 + 2], cmds[i * 4 + 3]]).collect();
        let mut claimed = vec![false; visible.len()];
        for b in 0..nb {
            let rec = sys.batch_rec(b);
            for (late, out) in [(0usize, &mut run.early), (1usize, &mut run.late)] {
                let n = run.commands[late * nb + b][1] as usize;
                let base = if late == 0 { rec[0] } else { rec[2] } as usize;
                let mut list = Vec::with_capacity(n);
                for k in 0..n {
                    let i = (base + k) * 2;
                    if i + 1 >= visible.len() {
                        run.out_of_range_writes.push(format!("batch {b} list {late}: entry {k} beyond the buffer"));
                        break;
                    }
                    claimed[i] = true;
                    claimed[i + 1] = true;
                    list.push((visible[i], visible[i + 1]));
                }
                out.push(list);
            }
        }
        for (i, w) in visible.iter().enumerate() {
            if !claimed[i] && *w != SENTINEL {
                run.out_of_range_writes.push(format!("visible list word {i} written outside any batch's counted entries"));
                break;
            }
        }
        let guard = |name: &str, buf: &[u32], used: usize, out: &mut Vec<String>| {
            if let Some(i) = buf.iter().skip(used).position(|w| *w != SENTINEL) {
                out.push(format!("{name}: word {} written past its capacity ({used} words)", used + i));
            }
        };
        guard("queue A", &qa, plan.node_cap as usize * 2, &mut run.out_of_range_writes);
        guard("queue B", &qb, plan.node_cap as usize * 2, &mut run.out_of_range_writes);
        let occ_n = (run.counters[crate::cluster::counter::OCCLUDED] as usize).min(plan.occ_cap as usize);
        guard("occluded list", &occ, plan.occ_cap as usize * 2, &mut run.out_of_range_writes);
        run.occluded = (0..occ_n).filter(|i| i * 2 + 1 < occ.len()).map(|i| (occ[i * 2], occ[i * 2 + 1])).collect();

        run.skin_ctl = sys.skin.read_ctl(0);
        let used = run.skin_ctl.get(crate::cluster_skin::ctl::USED).copied().unwrap_or(0) as usize;
        let reserved = used.min(sys.skin_pool_cap as usize);
        if let Some(i) = skin_words.iter().skip(reserved * 16).position(|w| *w != SENTINEL) {
            run.out_of_range_writes.push(format!("skin pool: word {} written past the reserved {reserved} vertices", reserved * 16 + i));
        }
        run.skin_pool = skin_words.iter().map(|w| f32::from_bits(*w)).collect();
        run.skin_slots = skin_slots;
        for mut h in hosts {
            h.destroy(device, &mut alloc);
        }
        hz.destroy(device, &mut alloc);
        sys.destroy(device, &mut alloc);
        unsafe { device.destroy_descriptor_set_layout(set_layout, None) };
        run.validation_errors = VALIDATION_ERRORS.load(Ordering::Relaxed) - errors_before;
        Ok(run)
    }

    /// Build the clustered graphics pipelines for the variants the renderer
    /// uses (opaque and cutout materials; prepass, main, depth-EQUAL passes;
    /// G-buffer and velocity attachments; 1x and 4x MSAA) on this device, so
    /// the driver links the clustered vertex shader against every fragment
    /// variant. Returns how many were created.
    pub fn check_cluster_pipelines(&self) -> Result<usize> {
        use crate::pipeline::{PipelineCache, PipelineKey};
        let device = &self.device;
        let mut cache = PipelineCache::new(
            device,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::Format::D32_SFLOAT,
            vk::Format::R8G8B8A8_UNORM,
        )?;
        let mut made = 0;
        let mut result = Ok(());
        'outer: for alpha_mode in [0u32, 1] {
            for (toon, normal_map, emission, per_instance) in
                [(false, false, false, false), (true, true, true, false), (false, true, false, true)]
            {
                for (prepass, depth_equal) in [(true, false), (false, true), (false, false)] {
                    for (gbuf, velocity) in [(false, false), (true, false), (false, true), (true, true)] {
                        // Multiview variants only on devices that expose it (GEO-5: both eyes
                        // from one visible list); single view always.
                        for (samples, mv) in [(1u8, false), (4, false), (1, true), (4, true)] {
                            if mv && !self.multiview {
                                continue;
                            }
                            let mut key = PipelineKey::from_features(&crate::MaterialFeatures {
                                toon,
                                normal_map,
                                emission,
                                alpha_mode: if alpha_mode == 0 { crate::AlphaMode::Opaque } else { crate::AlphaMode::Cutout },
                                double_sided: alpha_mode == 1,
                                water: false,
                                per_instance,
                                ..Default::default()
                            });
                            key.cluster = true;
                            key.prepass = prepass;
                            key.depth_equal = depth_equal;
                            key.gbuf = gbuf && !prepass;
                            key.velocity = velocity;
                            cache.set_target_views(mv, samples);
                            match cache.get(device, key) {
                                Ok(_) => made += 1,
                                Err(e) => {
                                    result = Err(e.context(format!("cluster pipeline {key:?}")));
                                    break 'outer;
                                }
                            }
                        }
                    }
                }
            }
        }
        cache.destroy(device);
        result.map(|()| made)
    }

    /// Decode vertices with the shader's `cluster_pull` (the function
    /// standard.vert uses): each request is (cluster, draw vertex index 0..383,
    /// 0, 0) of instance 0 of `mesh`. Returns per request
    /// `[x, y, z, live, nx, ny, nz, _, tx, ty, tz, tw, u, v, level, colour]`, the colour being the
    /// vertex's RGBA8 word as float bits (`f32::to_bits`).
    pub fn decode(&self, mesh: &ClusterMeshUpload, requests: &[[u32; 4]]) -> Result<Vec<[f32; 16]>> {
        let device = &self.device;
        let mut alloc = self.allocator()?;
        let set_layout = create_set_layout(device)?;
        let sizes = PoolSizes::default().clamped(self.max_storage_range);
        let mut sys = ClusterSystem::new(device, &mut alloc, set_layout, sizes)?;
        let id = sys.store.install(device, &mut alloc, self.pool, self.queue, 0, mesh)?;
        sys.begin_build();
        sys.batches.push(ClusterBatch {
            material: 0,
            push: Zeroable::zeroed(),
            mask: [u32::MAX; 4],
            bounds: None,
            bound: 0,
            early_base: 0,
            shadow_layer: None,
        });
        sys.add_raw_instance(
            InstRec {
                model: Mat4::IDENTITY.to_cols_array_2d(),
                color: [1.0; 4],
                param: [0.0; 4],
                sphere: [0.0; 4],
                mesh_batch: [id, 0, 0, 0],
                eye: [0.0; 4],
                extra: [0.0; 4],
            },
            0,
            0,
        );
        let hz = self.hiz(&mut alloc)?;
        let plan = sys.upload_frame(device, &mut alloc, 0, &GpuView::new(Mat4::IDENTITY, Vec3::ZERO, true), (hz.sampler, hz.view))?;
        let _ = plan;
        // Requests are (cluster, vertex index); the shader takes (instance, cluster, vertex index).
        let requests: Vec<[u32; 4]> = requests.iter().map(|r| [0, r[0], r[1], 0]).collect();
        let requests = &requests[..];
        let req_bytes = std::mem::size_of_val(requests).max(16) as u64;
        let out_bytes = (requests.len().max(1) * 64) as u64;
        let mut req = Buffer::new(device, &mut alloc, req_bytes, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "decode-req")?;
        req.write(0, bytemuck::cast_slice(requests));
        let out = Buffer::new(
            device,
            &mut alloc,
            out_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::GpuOnly,
            "decode-out",
        )?;
        let mut host = Buffer::new(device, &mut alloc, out_bytes, vk::BufferUsageFlags::TRANSFER_DST, MemoryLocation::GpuToCpu, "decode-host")?;
        let set = sys.targets(0).set;
        let infos = [
            [vk::DescriptorBufferInfo::default().buffer(req.handle).range(vk::WHOLE_SIZE)],
            [vk::DescriptorBufferInfo::default().buffer(out.handle).range(vk::WHOLE_SIZE)],
        ];
        let writes: Vec<_> = infos
            .iter()
            .zip([11u32, 12])
            .map(|(i, b)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(b)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(i)
            })
            .collect();
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        let count = requests.len() as u32;
        alloc::one_time_submit(device, self.pool, self.queue, |cb| unsafe {
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, sys.pipes.decode);
            device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, sys.pipes.layout, 0, &[set], &[]);
            device.cmd_push_constants(cb, sys.pipes.layout, vk::ShaderStageFlags::COMPUTE, 0, &count.to_le_bytes());
            device.cmd_dispatch(cb, count.div_ceil(64).max(1), 1, 1);
            let b = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)];
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&b));
            device.cmd_copy_buffer(cb, out.handle, host.handle, &[vk::BufferCopy::default().size(out_bytes)]);
            let b = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                .dst_access_mask(vk::AccessFlags2::HOST_READ)];
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&b));
        })?;
        let floats: Vec<f32> = bytemuck::pod_collect_to_vec(host.mapped().map(|m| &m[..requests.len() * 64]).unwrap_or(&[]));
        if floats.len() < requests.len() * 16 {
            bail!("decode read back {} floats", floats.len());
        }
        let result = floats.as_chunks::<16>().0.to_vec();
        req.destroy(device, &mut alloc);
        let mut out = out;
        out.destroy(device, &mut alloc);
        host.destroy(device, &mut alloc);
        hz.destroy(device, &mut alloc);
        sys.destroy(device, &mut alloc);
        unsafe { device.destroy_descriptor_set_layout(set_layout, None) };
        Ok(result)
    }
}


/// What `render_compare` measured.
#[derive(Debug, Default)]
pub struct RenderCompare {
    /// Pixels the ordinary (original mesh) depth prepass covers.
    pub covered_standard: usize,
    /// Pixels the clustered prepass covers.
    pub covered_cluster: usize,
    /// Covered by exactly one of the two.
    pub coverage_mismatch: usize,
    /// Largest |depth difference| over pixels both cover (99.9th percentile).
    pub depth_diff_p999: f32,
    pub depth_diff_max: f32,
    /// Pixels the clustered prepass covers where the second clustered pipeline
    /// (a different fragment/spec variant, depth compare EQUAL) failed the
    /// depth test: 0 means the two passes' positions are bit-identical.
    pub equal_failures: usize,
    pub validation_errors: u32,
}

impl HeadlessGpu {
    /// Render `mesh` twice into 256x256 depth targets with the engine's own
    /// pipelines: once the ordinary way (vertex buffers, `cmd_draw_indexed`,
    /// prepass pipeline) and once through the cluster path (the real cull +
    /// `cmd_draw_indirect` + the clustered vertex shader at the finest level),
    /// then compare. A second clustered draw with another pipeline variant and
    /// depth compare EQUAL over the first's depth proves the prepass/main
    /// position identity the forward renderer relies on.
    pub fn render_compare(&self, mesh: &crate::MeshData, up: &ClusterMeshUpload) -> Result<RenderCompare> {
        self.render_compare_posed(mesh, up, None)
    }

    /// `render_compare` for a skinned DAG: `mesh` is the CPU-posed original,
    /// `skin` the pose the cluster path poses on the GPU (GEO-5). The clustered
    /// prepass and the depth-EQUAL draw both read the one skinned pool.
    pub fn render_compare_posed(&self, mesh: &crate::MeshData, up: &ClusterMeshUpload, skin: Option<&HeadlessSkin>) -> Result<RenderCompare> {
        use crate::pipeline::{PipelineCache, PipelineKey};
        const W: u32 = 256;
        let device = &self.device;
        let errors_before = VALIDATION_ERRORS.load(Ordering::Relaxed);
        let mut alloc = self.allocator()?;
        let mut cache = PipelineCache::new(device, vk::Format::R16G16B16A16_SFLOAT, vk::Format::D32_SFLOAT, vk::Format::R8G8B8A8_UNORM)?;
        cache.set_target_views(false, 1);
        let features = crate::MaterialFeatures {
            toon: false,
            normal_map: false,
            emission: false,
            alpha_mode: crate::AlphaMode::Opaque,
            double_sided: false,
            water: false,
            per_instance: false,
            ..Default::default()
        };
        let mut std_key = PipelineKey::from_features(&features);
        std_key.prepass = true;
        let mut cl_prepass = std_key;
        cl_prepass.cluster = true;
        let mut cl_equal = PipelineKey::error();
        cl_equal.cluster = true;
        cl_equal.depth_equal = true;
        let std_pipe = cache.get(device, std_key)?;
        let cl_prepass_pipe = cache.get(device, cl_prepass)?;
        let cl_equal_pipe = cache.get(device, cl_equal)?;

        // Camera: looking at the mesh's bounding sphere.
        let sph = mesh_sphere(up);
        let centre = Vec3::new(sph[0], sph[1], sph[2]);
        let eye = centre + Vec3::new(0.35, 0.4, 1.0).normalize() * sph[3] * 3.0;
        let proj = Mat4::perspective_rh(0.9, 1.0, sph[3] * 0.1, sph[3] * 20.0);
        let view_proj = proj * Mat4::look_at_rh(eye, centre, Vec3::Y);
        let mut ubo: crate::FrameUbo = bytemuck::Zeroable::zeroed();
        ubo.view_proj = view_proj.to_cols_array_2d();
        ubo.prev_view_proj = view_proj.to_cols_array_2d();
        let mut ubo_buf = Buffer::new(
            device,
            &mut alloc,
            std::mem::size_of::<crate::FrameUbo>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            "rc-frame-ubo",
        )?;
        ubo_buf.write(0, bytemuck::bytes_of(&ubo));
        let mut fx_buf = Buffer::new(device, &mut alloc, 4096, vk::BufferUsageFlags::UNIFORM_BUFFER, MemoryLocation::CpuToGpu, "rc-fx")?;
        fx_buf.write(0, &[0u8; 4096]);

        // A 1x1 white texture for every sampler of the material set.
        let tex = self.hiz(&mut alloc)?;

        // Descriptor sets: frame (set 0), material (set 1).
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(2).pool_sizes(&[
                    vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(8),
                    vk::DescriptorPoolSize::default().ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(48),
                    vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(8),
                ]),
                None,
            )?
        };
        let layouts = [cache.set0_layout, cache.set1_layout];
        let sets = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts))? };
        let ubo_info = [vk::DescriptorBufferInfo::default().buffer(ubo_buf.handle).range(vk::WHOLE_SIZE)];
        let fx_info = [vk::DescriptorBufferInfo::default().buffer(fx_buf.handle).range(vk::WHOLE_SIZE)];
        let img = [vk::DescriptorImageInfo::default().sampler(tex.sampler).image_view(tex.view).image_layout(vk::ImageLayout::GENERAL)];
        let mut writes = vec![
            vk::WriteDescriptorSet::default().dst_set(sets[0]).dst_binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&ubo_info),
            vk::WriteDescriptorSet::default().dst_set(sets[1]).dst_binding(4).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&fx_info),
        ];
        for b in [0u32, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] {
            writes.push(
                vk::WriteDescriptorSet::default().dst_set(sets[1]).dst_binding(b).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&img),
            );
        }
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        // The ordinary mesh.
        let usage = vk::BufferUsageFlags::VERTEX_BUFFER;
        let vbuf = alloc::upload_buffer(device, &mut alloc, self.pool, self.queue, bytemuck::cast_slice(&mesh.vertices), usage, "rc-vertices")?;
        let ibuf = alloc::upload_buffer(device, &mut alloc, self.pool, self.queue, bytemuck::cast_slice(&mesh.indices), vk::BufferUsageFlags::INDEX_BUFFER, "rc-indices")?;
        let inst = crate::pipeline::InstanceData::from_model(Mat4::IDENTITY.to_cols_array_2d());
        let instbuf = alloc::upload_buffer(device, &mut alloc, self.pool, self.queue, bytemuck::bytes_of(&inst), vk::BufferUsageFlags::VERTEX_BUFFER, "rc-instance")?;

        // The cluster path: real install, cull and draw arguments.
        let set_layout = cache.cluster_set_layout;
        let mut sys = ClusterSystem::new(device, &mut alloc, set_layout, PoolSizes::default().clamped(self.max_storage_range))?;
        let id = sys.store.install(device, &mut alloc, self.pool, self.queue, 0, up)?;
        sys.begin_build();
        sys.batches.push(ClusterBatch {
            material: 0,
            push: Zeroable::zeroed(),
            mask: [u32::MAX; 4],
            bounds: None,
            bound: 0,
            early_base: 0,
            shadow_layer: None,
        });
        let rec = InstRec {
            model: Mat4::IDENTITY.to_cols_array_2d(),
            color: [1.0; 4],
            param: [0.0; 4],
            sphere: sph,
            mesh_batch: [id, 0, 0, 0],
            eye: crate::cluster_ref::object_space_eye(Mat4::IDENTITY, eye),
            extra: [1.0, 0.0, 0.0, 0.0],
        };
        match skin {
            None => sys.add_raw_instance(rec, 0, 0),
            Some(sk) => {
                let margin = sk.margin.unwrap_or_else(|| {
                    crate::cluster_skin::pose_margin([sph[0], sph[1], sph[2]], sph[3], &sk.palette, &sk.weights, &up.morph_reach())
                });
                let pose = crate::cluster_skin::SkinPose {
                    palette: &sk.palette,
                    prev_palette: sk.prev_palette.as_deref(),
                    weights: &sk.weights,
                    prev_weights: sk.prev_weights.as_deref(),
                    margin,
                };
                sys.add_raw_skinned_instance(rec, 0, 0, &pose);
            }
        }
        let mut gv = GpuView::new(view_proj, eye, true);
        // The finest level everywhere: a target so small only error 0 is accurate enough.
        gv.lod[1] = 0.5 * W as f32 * proj.y_axis.y;
        gv.lod[3] = 1e-6;
        let plan = sys.upload_frame(device, &mut alloc, 0, &gv, (tex.sampler, tex.view))?;
        let targets = sys.targets(0);
        let cmds = sys.frame_outputs(0)[1].0;

        // Targets.
        let make_image = |alloc: &mut Allocator, format: vk::Format, usage: vk::ImageUsageFlags, aspect: vk::ImageAspectFlags, name: &'static str| -> Result<(vk::Image, vk::ImageView, gpu_allocator::vulkan::Allocation)> {
            let info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D { width: W, height: W, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let image = unsafe { device.create_image(&info, None)? };
            let req = unsafe { device.get_image_memory_requirements(image) };
            let a = alloc.allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name,
                requirements: req,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })?;
            unsafe { device.bind_image_memory(image, a.memory(), a.offset())? };
            let view = unsafe {
                device.create_image_view(
                    &vk::ImageViewCreateInfo::default().image(image).view_type(vk::ImageViewType::TYPE_2D).format(format).subresource_range(
                        vk::ImageSubresourceRange::default().aspect_mask(aspect).level_count(1).layer_count(1),
                    ),
                    None,
                )?
            };
            Ok((image, view, a))
        };
        let depth_usage = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC;
        let (d_std, dv_std, da_std) = make_image(&mut alloc, vk::Format::D32_SFLOAT, depth_usage, vk::ImageAspectFlags::DEPTH, "rc-depth-std")?;
        let (d_cl, dv_cl, da_cl) = make_image(&mut alloc, vk::Format::D32_SFLOAT, depth_usage, vk::ImageAspectFlags::DEPTH, "rc-depth-cl")?;
        let (c_cl, cv_cl, ca_cl) = make_image(
            &mut alloc,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::ImageAspectFlags::COLOR,
            "rc-color-cl",
        )?;
        let px = (W * W) as u64;
        let mk_host = |alloc: &mut Allocator, bytes: u64, name: &str| {
            Buffer::new(device, alloc, bytes, vk::BufferUsageFlags::TRANSFER_DST, MemoryLocation::GpuToCpu, name)
        };
        let h_std = mk_host(&mut alloc, px * 4, "rc-h-std")?;
        let h_cl = mk_host(&mut alloc, px * 4, "rc-h-cl")?;
        let h_col = mk_host(&mut alloc, px * 8, "rc-h-col")?;

        let area = vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: W, height: W } };
        let viewport = vk::Viewport { x: 0.0, y: W as f32, width: W as f32, height: -(W as f32), min_depth: 0.0, max_depth: 1.0 };
        let depth_range = vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::DEPTH).level_count(1).layer_count(1);
        let color_range = vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
        let img_barrier = |cb: vk::CommandBuffer, image: vk::Image, range, old, new, src: (vk::PipelineStageFlags2, vk::AccessFlags2), dst: (vk::PipelineStageFlags2, vk::AccessFlags2)| unsafe {
            let b = [vk::ImageMemoryBarrier2::default()
                .image(image)
                .old_layout(old)
                .new_layout(new)
                .src_stage_mask(src.0)
                .src_access_mask(src.1)
                .dst_stage_mask(dst.0)
                .dst_access_mask(dst.1)
                .subresource_range(range)];
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&b));
        };
        let depth_stages = vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
        let depth_rw = vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;
        let none = (vk::PipelineStageFlags2::TOP_OF_PIPE, vk::AccessFlags2::empty());
        alloc::one_time_submit(device, self.pool, self.queue, |cb| unsafe {
            for (img, range) in [(d_std, depth_range), (d_cl, depth_range)] {
                img_barrier(cb, img, range, vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL, none, (depth_stages, depth_rw));
            }
            img_barrier(
                cb,
                c_cl,
                color_range,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                none,
                (vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE),
            );
            // The cluster cull (needs the pyramid in GENERAL: tex is).
            record_early(device, cb, &sys.pipes, &targets, &plan);
            let depth_att = |view: vk::ImageView, load: vk::AttachmentLoadOp| {
                vk::RenderingAttachmentInfo::default()
                    .image_view(view)
                    .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .load_op(load)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } })
            };
            let begin = |cb: vk::CommandBuffer, depth: &vk::RenderingAttachmentInfo, color: &[vk::RenderingAttachmentInfo]| {
                let info = vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(color).depth_attachment(depth);
                device.cmd_begin_rendering(cb, &info);
                device.cmd_set_viewport(cb, 0, &[viewport]);
                device.cmd_set_scissor(cb, 0, &[area]);
            };
            // 1. The ordinary prepass.
            let d = depth_att(dv_std, vk::AttachmentLoadOp::CLEAR);
            begin(cb, &d, &[]);
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, std_pipe);
            device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, cache.layout, 0, &[sets[0], sets[1]], &[]);
            let push: crate::PushData = Zeroable::zeroed();
            device.cmd_push_constants(cb, cache.layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, bytemuck::bytes_of(&push));
            device.cmd_bind_vertex_buffers(cb, 0, &[vbuf.handle], &[0]);
            device.cmd_bind_vertex_buffers(cb, 1, &[instbuf.handle], &[0]);
            device.cmd_bind_index_buffer(cb, ibuf.handle, 0, vk::IndexType::UINT32);
            device.cmd_draw_indexed(cb, mesh.indices.len() as u32, 1, 0, 0, 0);
            device.cmd_end_rendering(cb);
            // 2. The clustered prepass.
            let d = depth_att(dv_cl, vk::AttachmentLoadOp::CLEAR);
            begin(cb, &d, &[]);
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, cl_prepass_pipe);
            device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, cache.cluster_layout, 0, &[sets[0], sets[1], targets.set], &[]);
            let mut push: crate::PushData = Zeroable::zeroed();
            push.model[1][0] = f32::from_bits(0);
            device.cmd_push_constants(cb, cache.cluster_layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, bytemuck::bytes_of(&push));
            device.cmd_draw_indirect(cb, cmds, 0, 1, 16);
            device.cmd_end_rendering(cb);
            // 3. The clustered main-pass-style draw over that depth, compare EQUAL.
            let b = [vk::MemoryBarrier2::default()
                .src_stage_mask(depth_stages)
                .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .dst_stage_mask(depth_stages)
                .dst_access_mask(depth_rw)];
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&b));
            let d = depth_att(dv_cl, vk::AttachmentLoadOp::LOAD);
            let clear_color = vk::ClearValue { color: vk::ClearColorValue { float32: [0.0; 4] } };
            let col = [vk::RenderingAttachmentInfo::default()
                .image_view(cv_cl)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(clear_color)];
            begin(cb, &d, &col);
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, cl_equal_pipe);
            device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, cache.cluster_layout, 0, &[sets[0], sets[1], targets.set], &[]);
            device.cmd_push_constants(cb, cache.cluster_layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, bytemuck::bytes_of(&push));
            device.cmd_draw_indirect(cb, cmds, 0, 1, 16);
            device.cmd_end_rendering(cb);
            // Read everything back.
            for (img, range, aspect, host) in [
                (d_std, depth_range, vk::ImageAspectFlags::DEPTH, &h_std),
                (d_cl, depth_range, vk::ImageAspectFlags::DEPTH, &h_cl),
            ] {
                img_barrier(cb, img, range, vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, (depth_stages, depth_rw), (vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_READ));
                let copy = vk::BufferImageCopy::default()
                    .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(aspect).layer_count(1))
                    .image_extent(vk::Extent3D { width: W, height: W, depth: 1 });
                device.cmd_copy_image_to_buffer(cb, img, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, host.handle, &[copy]);
            }
            img_barrier(
                cb,
                c_cl,
                color_range,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                (vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE),
                (vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_READ),
            );
            let copy = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
                .image_extent(vk::Extent3D { width: W, height: W, depth: 1 });
            device.cmd_copy_image_to_buffer(cb, c_cl, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, h_col.handle, &[copy]);
            let b = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                .dst_access_mask(vk::AccessFlags2::HOST_READ)];
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&b));
        })?;
        let f32s = |b: &Buffer| -> Vec<f32> { bytemuck::pod_collect_to_vec(b.mapped().unwrap_or(&[])) };
        let (ds, dc) = (f32s(&h_std), f32s(&h_cl));
        let col: Vec<u16> = bytemuck::pod_collect_to_vec(h_col.mapped().unwrap_or(&[]));
        let mut out = RenderCompare::default();
        let mut diffs = Vec::new();
        for i in 0..(W * W) as usize {
            let (a, b) = (ds[i] < 1.0, dc[i] < 1.0);
            out.covered_standard += usize::from(a);
            out.covered_cluster += usize::from(b);
            out.coverage_mismatch += usize::from(a != b);
            if a && b {
                diffs.push((ds[i] - dc[i]).abs());
            }
            // Any non-zero channel of the RGBA16F pixel: the EQUAL draw wrote it.
            let wrote = col[i * 4..i * 4 + 4].iter().any(|&h| h != 0);
            out.equal_failures += usize::from(b && !wrote);
        }
        diffs.sort_by(f32::total_cmp);
        out.depth_diff_max = diffs.last().copied().unwrap_or(0.0);
        out.depth_diff_p999 = diffs.get(diffs.len().saturating_sub(1) * 999 / 1000).copied().unwrap_or(0.0);
        // Clean up.
        for mut b in [h_std, h_cl, h_col] {
            b.destroy(device, &mut alloc);
        }
        for (i, v, a) in [(d_std, dv_std, da_std), (d_cl, dv_cl, da_cl), (c_cl, cv_cl, ca_cl)] {
            unsafe {
                device.destroy_image_view(v, None);
                device.destroy_image(i, None);
            }
            let _ = alloc.free(a);
        }
        let (mut vbuf, mut ibuf, mut instbuf) = (vbuf, ibuf, instbuf);
        for b in [&mut vbuf, &mut ibuf, &mut instbuf, &mut ubo_buf, &mut fx_buf] {
            b.destroy(device, &mut alloc);
        }
        tex.destroy(device, &mut alloc);
        unsafe { device.destroy_descriptor_pool(pool, None) };
        sys.destroy(device, &mut alloc);
        cache.destroy(device);
        out.validation_errors = VALIDATION_ERRORS.load(Ordering::Relaxed) - errors_before;
        Ok(out)
    }
}

/// What `render_stereo_compare` measured (per eye: [left, right]).
#[derive(Debug, Default)]
pub struct StereoCompare {
    /// Pixels the ordinary mesh covers in that eye (its own single-view render).
    pub covered_standard: [usize; 2],
    /// Pixels the clustered multiview pass covers in that eye's layer.
    pub covered_cluster: [usize; 2],
    /// Covered by exactly one of the two.
    pub mismatch: [usize; 2],
    /// Pixels the multiview pass wrote colour for in each layer (the pass really ran per view).
    pub colour_written: [usize; 2],
    /// Pixels covered in exactly one of the two layers of the multiview pass (the eyes really differ).
    pub layers_differ: usize,
    /// Clusters / triangles the one centre-eye cut drew for both eyes.
    pub visible_clusters: u32,
    pub triangles: u32,
    pub validation_errors: u32,
}

impl HeadlessGpu {
    /// The XR cluster path without a headset: ONE cull from the centre eye
    /// (`stereo_cull_view`: union frustum, centre-eye LOD with the IPD slack)
    /// drawn by `standard.vert -DCLUSTER -DMULTIVIEW` into a two-layer target in
    /// a single multiview pass (view mask 0b11), against the ordinary mesh
    /// rendered once per eye with that eye's own matrices. `target_px` = the
    /// LOD error target (a tiny value draws the finest level, so the
    /// silhouettes can be compared exactly). None when the device has no multiview.
    pub fn render_stereo_compare(&self, mesh: &crate::MeshData, up: &ClusterMeshUpload, target_px: f32) -> Result<Option<StereoCompare>> {
        use crate::pipeline::{PipelineCache, PipelineKey};
        if !self.multiview {
            return Ok(None);
        }
        const W: u32 = 256;
        let device = &self.device;
        let errors_before = VALIDATION_ERRORS.load(Ordering::Relaxed);
        let mut alloc = self.allocator()?;
        let mut cache = PipelineCache::new(device, vk::Format::R16G16B16A16_SFLOAT, vk::Format::D32_SFLOAT, vk::Format::R8G8B8A8_UNORM)?;
        let features = crate::MaterialFeatures {
            toon: false,
            normal_map: false,
            emission: false,
            alpha_mode: crate::AlphaMode::Opaque,
            double_sided: false,
            water: false,
            per_instance: false,
            ..Default::default()
        };
        let mut std_key = PipelineKey::from_features(&features);
        std_key.prepass = true;
        let mut cl_key = PipelineKey::error();
        cl_key.cluster = true;
        cache.set_target_views(false, 1);
        let std_pipe = cache.get(device, std_key)?;
        cache.set_target_views(true, 1);
        let mv_pipe = cache.get(device, cl_key)?;
        cache.set_target_views(false, 1);

        // Two eyes 64 mm apart, looking a little past the mesh so it is partly outside each frustum.
        let sph = mesh_sphere(up);
        let centre = Vec3::new(sph[0], sph[1], sph[2]);
        let head = centre + Vec3::new(0.35, 0.3, 1.0).normalize() * sph[3] * 2.6;
        let gaze = centre + Vec3::new(sph[3] * 0.45, 0.0, 0.0);
        let head_view = Mat4::look_at_rh(head, gaze, Vec3::Y);
        // A wide baseline (6 % of the mesh radius each side) so the two views differ visibly.
        let half_ipd = 0.06 * sph[3];
        let eye_view = |dx: f32| Mat4::from_translation(Vec3::new(-dx, 0.0, 0.0)) * head_view;
        let proj = Mat4::perspective_rh(1.5, 1.0, sph[3] * 0.05, sph[3] * 20.0);
        let sv = crate::cluster_vr::StereoView { view: [eye_view(-half_ipd), eye_view(half_ipd)], proj: [proj, proj] };
        let single = |vp: Mat4| -> crate::FrameUbo {
            let mut u: crate::FrameUbo = bytemuck::Zeroable::zeroed();
            u.view_proj = vp.to_cols_array_2d();
            u.prev_view_proj = vp.to_cols_array_2d();
            u
        };
        let ubos = [single(sv.proj[0] * sv.view[0]), single(sv.proj[1] * sv.view[1])];
        let mut mv: crate::FrameUbo = bytemuck::Zeroable::zeroed();
        for e in 0..2 {
            let vp = (sv.proj[e] * sv.view[e]).to_cols_array_2d();
            mv.eye_view[e] = sv.view[e].to_cols_array_2d();
            mv.eye_view_proj[e] = vp;
            let p = sv.view[e].inverse().w_axis;
            mv.eye_camera_pos[e] = [p.x, p.y, p.z, 1.0];
        }
        mv.view_proj = (sv.proj[0] * crate::cluster_vr::centre_eye(&sv)).to_cols_array_2d();
        mv.prev_view_proj = mv.view_proj;
        let mut ubo_bufs = Vec::new();
        for u in [ubos[0], ubos[1], mv] {
            let mut b = Buffer::new(device, &mut alloc, std::mem::size_of::<crate::FrameUbo>() as u64, vk::BufferUsageFlags::UNIFORM_BUFFER, MemoryLocation::CpuToGpu, "sc-ubo")?;
            b.write(0, bytemuck::bytes_of(&u));
            ubo_bufs.push(b);
        }
        let mut fx_buf = Buffer::new(device, &mut alloc, 4096, vk::BufferUsageFlags::UNIFORM_BUFFER, MemoryLocation::CpuToGpu, "sc-fx")?;
        fx_buf.write(0, &[0u8; 4096]);
        let tex = self.hiz(&mut alloc)?;
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(4).pool_sizes(&[
                    vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(8),
                    vk::DescriptorPoolSize::default().ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(48),
                    vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(8),
                ]),
                None,
            )?
        };
        let layouts = [cache.set0_layout, cache.set0_layout, cache.set0_layout, cache.set1_layout];
        let sets = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts))? };
        let ubo_infos: Vec<[vk::DescriptorBufferInfo; 1]> = ubo_bufs.iter().map(|b| [vk::DescriptorBufferInfo::default().buffer(b.handle).range(vk::WHOLE_SIZE)]).collect();
        let fx_info = [vk::DescriptorBufferInfo::default().buffer(fx_buf.handle).range(vk::WHOLE_SIZE)];
        let img = [vk::DescriptorImageInfo::default().sampler(tex.sampler).image_view(tex.view).image_layout(vk::ImageLayout::GENERAL)];
        let mut writes: Vec<vk::WriteDescriptorSet> = (0..3)
            .map(|k| vk::WriteDescriptorSet::default().dst_set(sets[k]).dst_binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&ubo_infos[k]))
            .collect();
        writes.push(vk::WriteDescriptorSet::default().dst_set(sets[3]).dst_binding(4).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&fx_info));
        for b in [0u32, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] {
            writes.push(vk::WriteDescriptorSet::default().dst_set(sets[3]).dst_binding(b).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&img));
        }
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        let vbuf = alloc::upload_buffer(device, &mut alloc, self.pool, self.queue, bytemuck::cast_slice(&mesh.vertices), vk::BufferUsageFlags::VERTEX_BUFFER, "sc-vertices")?;
        let ibuf = alloc::upload_buffer(device, &mut alloc, self.pool, self.queue, bytemuck::cast_slice(&mesh.indices), vk::BufferUsageFlags::INDEX_BUFFER, "sc-indices")?;
        let inst = crate::pipeline::InstanceData::from_model(Mat4::IDENTITY.to_cols_array_2d());
        let instbuf = alloc::upload_buffer(device, &mut alloc, self.pool, self.queue, bytemuck::bytes_of(&inst), vk::BufferUsageFlags::VERTEX_BUFFER, "sc-instance")?;

        // The cluster path: one instance, one cut for both eyes.
        let mut sys = ClusterSystem::new(device, &mut alloc, cache.cluster_set_layout, PoolSizes::default().clamped(self.max_storage_range))?;
        let id = sys.store.install(device, &mut alloc, self.pool, self.queue, 0, up)?;
        sys.begin_build();
        sys.batches.push(ClusterBatch {
            material: 0,
            push: Zeroable::zeroed(),
            mask: [u32::MAX; 4],
            bounds: None,
            bound: 0,
            early_base: 0,
            shadow_layer: None,
        });
        sys.add_raw_instance(
            InstRec {
                model: Mat4::IDENTITY.to_cols_array_2d(),
                color: [1.0; 4],
                param: [0.0; 4],
                sphere: sph,
                mesh_batch: [id, 0, 0, 0],
                // The cone test is for one eye position: off for stereo.
                eye: [0.0; 4],
                extra: [1.0, 0.0, 0.0, 0.0],
            },
            0,
            0,
        );
        let gv = crate::cluster::stereo_cull_view(&sv, W as f32, target_px);
        let plan = sys.upload_frame(device, &mut alloc, 0, &gv, (tex.sampler, tex.view))?;
        let targets = sys.targets(0);
        let cmds = sys.frame_outputs(0)[1].0;

        let make_image = |alloc: &mut Allocator, format: vk::Format, layers: u32, usage: vk::ImageUsageFlags, aspect: vk::ImageAspectFlags| -> Result<(vk::Image, vk::ImageView, gpu_allocator::vulkan::Allocation)> {
            let info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D { width: W, height: W, depth: 1 })
                .mip_levels(1)
                .array_layers(layers)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let image = unsafe { device.create_image(&info, None)? };
            let req = unsafe { device.get_image_memory_requirements(image) };
            let a = alloc.allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: "sc-image",
                requirements: req,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })?;
            unsafe { device.bind_image_memory(image, a.memory(), a.offset())? };
            let view = unsafe {
                device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(if layers > 1 { vk::ImageViewType::TYPE_2D_ARRAY } else { vk::ImageViewType::TYPE_2D })
                        .format(format)
                        .subresource_range(vk::ImageSubresourceRange::default().aspect_mask(aspect).level_count(1).layer_count(layers)),
                    None,
                )?
            };
            Ok((image, view, a))
        };
        let depth_usage = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC;
        let eye_depth = [
            make_image(&mut alloc, vk::Format::D32_SFLOAT, 1, depth_usage, vk::ImageAspectFlags::DEPTH)?,
            make_image(&mut alloc, vk::Format::D32_SFLOAT, 1, depth_usage, vk::ImageAspectFlags::DEPTH)?,
        ];
        let (d_mv, dv_mv, da_mv) = make_image(&mut alloc, vk::Format::D32_SFLOAT, 2, depth_usage, vk::ImageAspectFlags::DEPTH)?;
        let (c_mv, cv_mv, ca_mv) = make_image(
            &mut alloc,
            vk::Format::R16G16B16A16_SFLOAT,
            2,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::ImageAspectFlags::COLOR,
        )?;
        let px = (W * W) as u64;
        let mk_host = |alloc: &mut Allocator, bytes: u64, name: &str| Buffer::new(device, alloc, bytes, vk::BufferUsageFlags::TRANSFER_DST, MemoryLocation::GpuToCpu, name);
        let h_eye = [mk_host(&mut alloc, px * 4, "sc-h0")?, mk_host(&mut alloc, px * 4, "sc-h1")?];
        let h_mv = mk_host(&mut alloc, px * 8, "sc-hmv")?;
        let h_col = mk_host(&mut alloc, px * 16, "sc-hcol")?;

        let area = vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: W, height: W } };
        let viewport = vk::Viewport { x: 0.0, y: W as f32, width: W as f32, height: -(W as f32), min_depth: 0.0, max_depth: 1.0 };
        let depth_range = |layers: u32| vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::DEPTH).level_count(1).layer_count(layers);
        let color_range = vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(2);
        let img_barrier = |cb: vk::CommandBuffer, image: vk::Image, range, old, new, src: (vk::PipelineStageFlags2, vk::AccessFlags2), dst: (vk::PipelineStageFlags2, vk::AccessFlags2)| unsafe {
            let b = [vk::ImageMemoryBarrier2::default()
                .image(image)
                .old_layout(old)
                .new_layout(new)
                .src_stage_mask(src.0)
                .src_access_mask(src.1)
                .dst_stage_mask(dst.0)
                .dst_access_mask(dst.1)
                .subresource_range(range)];
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&b));
        };
        let depth_stages = vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
        let depth_rw = vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;
        let none = (vk::PipelineStageFlags2::TOP_OF_PIPE, vk::AccessFlags2::empty());
        alloc::one_time_submit(device, self.pool, self.queue, |cb| unsafe {
            for img in [eye_depth[0].0, eye_depth[1].0] {
                img_barrier(cb, img, depth_range(1), vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL, none, (depth_stages, depth_rw));
            }
            img_barrier(cb, d_mv, depth_range(2), vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL, none, (depth_stages, depth_rw));
            img_barrier(cb, c_mv, color_range, vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, none, (vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE));
            record_early(device, cb, &sys.pipes, &targets, &plan);
            let depth_att = |view: vk::ImageView| {
                vk::RenderingAttachmentInfo::default()
                    .image_view(view)
                    .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } })
            };
            // The ordinary mesh, once per eye with that eye's own matrices.
            for e in 0..2 {
                let d = depth_att(eye_depth[e].1);
                device.cmd_begin_rendering(cb, &vk::RenderingInfo::default().render_area(area).layer_count(1).depth_attachment(&d));
                device.cmd_set_viewport(cb, 0, &[viewport]);
                device.cmd_set_scissor(cb, 0, &[area]);
                device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, std_pipe);
                device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, cache.layout, 0, &[sets[e], sets[3]], &[]);
                let push: crate::PushData = Zeroable::zeroed();
                device.cmd_push_constants(cb, cache.layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, bytemuck::bytes_of(&push));
                device.cmd_bind_vertex_buffers(cb, 0, &[vbuf.handle], &[0]);
                device.cmd_bind_vertex_buffers(cb, 1, &[instbuf.handle], &[0]);
                device.cmd_bind_index_buffer(cb, ibuf.handle, 0, vk::IndexType::UINT32);
                device.cmd_draw_indexed(cb, mesh.indices.len() as u32, 1, 0, 0, 0);
                device.cmd_end_rendering(cb);
            }
            // The clustered multiview pass: both layers from one visible list.
            let d = depth_att(dv_mv);
            let col = [vk::RenderingAttachmentInfo::default()
                .image_view(cv_mv)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0; 4] } })];
            device.cmd_begin_rendering(cb, &vk::RenderingInfo::default().render_area(area).layer_count(1).view_mask(0b11).color_attachments(&col).depth_attachment(&d));
            device.cmd_set_viewport(cb, 0, &[viewport]);
            device.cmd_set_scissor(cb, 0, &[area]);
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, mv_pipe);
            device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, cache.cluster_layout, 0, &[sets[2], sets[3], targets.set], &[]);
            let mut push: crate::PushData = Zeroable::zeroed();
            push.model[1][0] = f32::from_bits(0);
            device.cmd_push_constants(cb, cache.cluster_layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, bytemuck::bytes_of(&push));
            device.cmd_draw_indirect(cb, cmds, 0, 1, 16);
            device.cmd_end_rendering(cb);
            // Read back.
            let copy = |cb: vk::CommandBuffer, image: vk::Image, aspect: vk::ImageAspectFlags, layers: u32, host: &Buffer| {
                let c = vk::BufferImageCopy::default()
                    .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(aspect).layer_count(layers))
                    .image_extent(vk::Extent3D { width: W, height: W, depth: 1 });
                device.cmd_copy_image_to_buffer(cb, image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, host.handle, &[c]);
            };
            for (img, layers, host) in [(eye_depth[0].0, 1, &h_eye[0]), (eye_depth[1].0, 1, &h_eye[1]), (d_mv, 2, &h_mv)] {
                img_barrier(cb, img, depth_range(layers), vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, (depth_stages, depth_rw), (vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_READ));
                copy(cb, img, vk::ImageAspectFlags::DEPTH, layers, host);
            }
            img_barrier(
                cb,
                c_mv,
                color_range,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                (vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE),
                (vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_READ),
            );
            copy(cb, c_mv, vk::ImageAspectFlags::COLOR, 2, &h_col);
            let b = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                .dst_access_mask(vk::AccessFlags2::HOST_READ)];
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&b));
        })?;
        let f32s = |b: &Buffer| -> Vec<f32> { bytemuck::pod_collect_to_vec(b.mapped().unwrap_or(&[])) };
        let (dmv, d0, d1) = (f32s(&h_mv), f32s(&h_eye[0]), f32s(&h_eye[1]));
        let col: Vec<u16> = bytemuck::pod_collect_to_vec(h_col.mapped().unwrap_or(&[]));
        let mut out = StereoCompare::default();
        let n = (W * W) as usize;
        for e in 0..2 {
            let reference = if e == 0 { &d0 } else { &d1 };
            for i in 0..n {
                let (a, b) = (reference[i] < 1.0, dmv[e * n + i] < 1.0);
                out.covered_standard[e] += usize::from(a);
                out.covered_cluster[e] += usize::from(b);
                out.mismatch[e] += usize::from(a != b);
                out.colour_written[e] += usize::from(col[(e * n + i) * 4..(e * n + i) * 4 + 4].iter().any(|&h| h != 0));
            }
        }
        out.layers_differ = (0..n).filter(|&i| (dmv[i] < 1.0) != (dmv[n + i] < 1.0)).count();
        let counts = crate::cluster::ClusterCounts::from_words(&sys.read_counters(0));
        out.visible_clusters = counts.visible;
        out.triangles = counts.triangles;
        for mut b in h_eye.into_iter().chain([h_mv, h_col]) {
            b.destroy(device, &mut alloc);
        }
        for (i, v, a) in eye_depth.into_iter().chain([(d_mv, dv_mv, da_mv), (c_mv, cv_mv, ca_mv)]) {
            unsafe {
                device.destroy_image_view(v, None);
                device.destroy_image(i, None);
            }
            let _ = alloc.free(a);
        }
        let (mut vbuf, mut ibuf, mut instbuf) = (vbuf, ibuf, instbuf);
        for b in [&mut vbuf, &mut ibuf, &mut instbuf, &mut fx_buf].into_iter().chain(ubo_bufs.iter_mut()) {
            b.destroy(device, &mut alloc);
        }
        tex.destroy(device, &mut alloc);
        unsafe { device.destroy_descriptor_pool(pool, None) };
        sys.destroy(device, &mut alloc);
        cache.destroy(device);
        out.validation_errors = VALIDATION_ERRORS.load(Ordering::Relaxed) - errors_before;
        Ok(Some(out))
    }
}

impl Drop for HeadlessGpu {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            if let Some((du, m)) = self.messenger.take() {
                du.destroy_debug_utils_messenger(m, None);
            }
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

struct HizImage {
    image: vk::Image,
    view: vk::ImageView,
    sampler: vk::Sampler,
    allocation: Option<gpu_allocator::vulkan::Allocation>,
    range: vk::ImageSubresourceRange,
}

impl HizImage {
    fn destroy(mut self, device: &ash::Device, alloc: &mut Allocator) {
        unsafe {
            device.destroy_sampler(self.sampler, None);
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
        }
        if let Some(a) = self.allocation.take() {
            let _ = alloc.free(a);
        }
        let _ = self.range;
    }
}

/// Bounding sphere (object space) of a DAG: centred on the finest clusters,
/// enclosing every cluster of every level (so culling an instance by it can
/// never drop a cluster the sphere test would keep).
pub fn mesh_sphere(m: &ClusterMeshUpload) -> [f32; 4] {
    let leaf: Vec<_> = m.clusters.iter().filter(|c| c.level == 0).collect();
    let n = leaf.len().max(1) as f32;
    let mut c = Vec3::ZERO;
    for cl in &leaf {
        c += Vec3::new(cl.bounds[0], cl.bounds[1], cl.bounds[2]) / n;
    }
    let r = m
        .clusters
        .iter()
        .map(|cl| (Vec3::new(cl.bounds[0], cl.bounds[1], cl.bounds[2]) - c).length() + cl.bounds[3])
        .fold(0.0, f32::max);
    [c.x, c.y, c.z, r * 1.001]
}
