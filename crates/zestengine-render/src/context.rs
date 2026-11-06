//! Vulkan instance, surface, and device bootstrap.

use crate::hitch::WaitNamed as _;
use std::ffi::{CStr, c_char, c_void};

use anyhow::{Context as _, Result, bail};
use ash::vk::Handle as _;
use ash::{amd, ext, khr, nv, vk};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

/// When a VR headset is present, the renderer's `VkInstance`/`VkDevice` MUST be
/// created via OpenXR (`xrCreateVulkanInstanceKHR` / `xrGetVulkanGraphicsDevice2KHR`
/// / `xrCreateVulkanDeviceKHR`) so the runtime records our `vkGetInstanceProcAddr`
/// and the session can share this device. `None` = normal flat creation.
pub struct XrBootstrap<'a> {
    pub instance: &'a openxr::Instance,
    pub system: openxr::SystemId,
}

/// Owns the core Vulkan objects every other renderer module hangs off of.
pub struct GpuContext {
    /// Never read, but must outlive everything: it owns the loaded libvulkan.
    #[allow(dead_code)]
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    debug: Option<(ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    pub surface_loader: khr::surface::Instance,
    pub surface: vk::SurfaceKHR,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue_family: u32,
    pub queue: vk::Queue,
    /// Background asset-upload queue (same family, index 1) for the loader
    /// thread; None when the family exposes only one queue.
    pub transfer_queue: Option<vk::Queue>,
    /// Acceleration-structure loader. Present only when the device supports
    /// ray query (the lighting bake's ray tracing); `None` disables baking.
    // Consumed by the bake's BLAS/TLAS builders (next phase).
    #[allow(dead_code)]
    pub accel: Option<khr::acceleration_structure::Device>,
    /// GPU checkpoints (NVIDIA): named markers in the command stream, read
    /// back after a device loss to say which pass the GPU was in.
    checkpoints: Option<nv::device_diagnostic_checkpoints::Device>,
    /// The same on AMD: marker ids the GPU writes into a host-visible buffer
    /// as it starts and finishes each marked point.
    markers: Option<MarkerBuffer>,
    /// The driver's own fault report after a loss (cross-vendor).
    fault: Option<ext::device_fault::Device>,
    lost_reported: std::sync::atomic::AtomicBool,
    /// Negotiated device API version (1.1 ..= 1.3).
    pub api_version: u32,
    /// `VK_KHR_multiview` (core 1.1) enabled: one pass renders both eyes.
    pub multiview: bool,
    /// `tessellationShader` enabled: displaced materials get tessellation stages.
    pub tessellation: bool,
    /// `multiviewTessellationShader` enabled too: tessellated materials draw in the XR multiview pass.
    pub multiview_tessellation: bool,
    /// Probed GPU-driven-rendering capabilities and limits (see [`GpuCaps`]).
    pub caps: GpuCaps,
}

/// What the device offers for GPU-driven rendering (the Nanite-lite work, GEO-4),
/// probed once at startup. The `*_enabled` style booleans are what was actually
/// switched on at device creation; the rest are recorded only.
///
/// Fallbacks GEO-4 must keep working when a bit is missing:
/// - no `multi_draw_indirect`: one `vkCmdDrawIndexedIndirect` per batch (draw
///   count 1) instead of a single multi-draw over the cluster/draw list;
/// - no `draw_indirect_count`: the culling pass writes fixed-size command
///   arrays (instance count 0 for culled entries) and the draw uses the
///   upper-bound count instead of a GPU-written count buffer;
/// - no `shader_int64` / `buffer_device_address`: vertex pulling goes through
///   storage-buffer descriptors with 32-bit indices (no 64-bit pointers or
///   packed 64-bit visibility keys; split into two u32);
/// - no `shader_buffer_int64_atomics`: the visibility buffer packs depth and
///   id into 32 bits, or uses a two-pass depth-then-id resolve, instead of a
///   single 64-bit atomicMax;
/// - `mesh_shader` is never used (record only); the compute-cull + indirect
///   path is the portable one, and the only one on Adreno / Quest.
/// - Large cluster tables may exceed `max_storage_buffer_range`: split across
///   several buffers/descriptors.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GpuCaps {
    pub api_version: u32,
    pub device_name: String,
    /// `multiDrawIndirect` (core feature) - enabled when supported.
    pub multi_draw_indirect: bool,
    /// `drawIndirectCount` (Vulkan 1.2 feature, or `VK_KHR_draw_indirect_count`
    /// on 1.1 devices) - enabled when supported.
    pub draw_indirect_count: bool,
    /// `shaderInt64` (core feature) - enabled when supported.
    pub shader_int64: bool,
    /// `bufferDeviceAddress` (Vulkan 1.2), independent of ray tracing -
    /// enabled when supported. False on 1.1 devices (extension path not taken).
    pub buffer_device_address: bool,
    /// `shaderBufferInt64Atomics` - recorded only, not enabled.
    pub shader_buffer_int64_atomics: bool,
    /// `VK_EXT_mesh_shader` + its meshShader/taskShader features - recorded
    /// only, never enabled or used.
    pub mesh_shader: bool,
    /// Ray query + acceleration structures enabled (lighting bake).
    pub ray_tracing: bool,
    pub multiview: bool,
    /// `VK_EXT/KHR_present_mode_fifo_latest_ready` enabled (the swapchain may
    /// then use `PRESENT_MODE_FIFO_LATEST_READY`).
    pub present_fifo_latest_ready: bool,
    pub max_draw_indirect_count: u32,
    pub max_storage_buffer_range: u32,
    pub max_compute_work_group_count: [u32; 3],
}

impl GpuCaps {
    /// The single INFO line logged at startup.
    pub fn summary(&self) -> String {
        let b = |v: bool| if v { "yes" } else { "no" };
        format!(
            "GPU caps [{}]: multiDrawIndirect={} drawIndirectCount={} shaderInt64={} bufferDeviceAddress={} \
             shaderBufferInt64Atomics={} meshShader(EXT, unused)={} maxDrawIndirectCount={} \
             maxStorageBufferRange={} maxComputeWorkGroupCount={:?}",
            self.device_name,
            b(self.multi_draw_indirect),
            b(self.draw_indirect_count),
            b(self.shader_int64),
            b(self.buffer_device_address),
            b(self.shader_buffer_int64_atomics),
            b(self.mesh_shader),
            self.max_draw_indirect_count,
            self.max_storage_buffer_range,
            self.max_compute_work_group_count,
        )
    }

    /// Multi-line report for `ZESTENGINE_GPU_CAPS=1` / `--gpu-caps`, with the
    /// GEO-4 path each bit selects.
    pub fn report(&self) -> String {
        let b = |v: bool| if v { "yes" } else { "NO " };
        let (major, minor) = (vk::api_version_major(self.api_version), vk::api_version_minor(self.api_version));
        let mut s = String::new();
        s += &format!("zestengine GPU capability report\ndevice: {} (Vulkan {major}.{minor})\n", self.device_name);
        s += &format!("  multiDrawIndirect          {}  -> else one vkCmdDrawIndexedIndirect per batch\n", b(self.multi_draw_indirect));
        s += &format!("  drawIndirectCount          {}  -> else fixed-size command array, upper-bound count\n", b(self.draw_indirect_count));
        s += &format!("  shaderInt64                {}  -> else 2x u32 emulation\n", b(self.shader_int64));
        s += &format!("  bufferDeviceAddress        {}  -> else vertex pulling via storage-buffer descriptors\n", b(self.buffer_device_address));
        s += &format!("  shaderBufferInt64Atomics   {}  (recorded only) -> else 32-bit visibility key\n", b(self.shader_buffer_int64_atomics));
        s += &format!("  VK_EXT_mesh_shader         {}  (recorded only, never used)\n", b(self.mesh_shader));
        s += &format!("  ray query / accel structs  {}\n", b(self.ray_tracing));
        s += &format!("  multiview                  {}\n", b(self.multiview));
        s += &format!("  present FIFO_LATEST_READY  {}\n", b(self.present_fifo_latest_ready));
        s += &format!("  maxDrawIndirectCount       {}\n", self.max_draw_indirect_count);
        s += &format!("  maxStorageBufferRange      {} ({} MiB)\n", self.max_storage_buffer_range, self.max_storage_buffer_range >> 20);
        s += &format!("  maxComputeWorkGroupCount   {:?}\n", self.max_compute_work_group_count);
        let gpu_driven = self.multi_draw_indirect && self.draw_indirect_count;
        s += &format!(
            "GEO-4 path: {}\n",
            if gpu_driven { "compute cull + multi-draw-indirect-count" } else { "compute cull + per-batch vkCmdDrawIndexedIndirect (VS pulling)" }
        );
        s
    }
}

/// `VK_EXT_present_mode_fifo_latest_ready` / `VK_KHR_present_mode_fifo_latest_ready`
/// raw present-mode value (neither is in the vendored ash). Behaves like FIFO
/// but, when a vblank finds several queued images, shows the newest ready one
/// instead of waiting for each in turn.
const FLR_EXT_NAME: &CStr = c"VK_EXT_present_mode_fifo_latest_ready";
const FLR_KHR_NAME: &CStr = c"VK_KHR_present_mode_fifo_latest_ready";

pub const PRESENT_MODE_FIFO_LATEST_READY: vk::PresentModeKHR = vk::PresentModeKHR::from_raw(1000361000);

/// `VkPhysicalDevicePresentModeFifoLatestReadyFeaturesEXT` (the KHR promotion
/// shares the layout and sType), defined here because ash predates it.
#[repr(C)]
pub struct PresentModeFifoLatestReadyFeatures {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    pub present_mode_fifo_latest_ready: vk::Bool32,
}

impl Default for PresentModeFifoLatestReadyFeatures {
    fn default() -> Self {
        Self { s_type: <Self as vk::TaggedStructure>::STRUCTURE_TYPE, p_next: std::ptr::null_mut(), present_mode_fifo_latest_ready: vk::FALSE }
    }
}

unsafe impl vk::TaggedStructure for PresentModeFifoLatestReadyFeatures {
    const STRUCTURE_TYPE: vk::StructureType = vk::StructureType::from_raw(1000361000);
}
unsafe impl vk::ExtendsPhysicalDeviceFeatures2 for PresentModeFifoLatestReadyFeatures {}
unsafe impl vk::ExtendsDeviceCreateInfo for PresentModeFifoLatestReadyFeatures {}

/// Readable name of a present mode for logs.
pub fn present_mode_name(mode: vk::PresentModeKHR) -> String {
    match mode {
        vk::PresentModeKHR::IMMEDIATE => "IMMEDIATE".into(),
        vk::PresentModeKHR::MAILBOX => "MAILBOX".into(),
        vk::PresentModeKHR::FIFO => "FIFO".into(),
        vk::PresentModeKHR::FIFO_RELAXED => "FIFO_RELAXED".into(),
        vk::PresentModeKHR::SHARED_DEMAND_REFRESH => "SHARED_DEMAND_REFRESH".into(),
        vk::PresentModeKHR::SHARED_CONTINUOUS_REFRESH => "SHARED_CONTINUOUS_REFRESH".into(),
        PRESENT_MODE_FIFO_LATEST_READY => "FIFO_LATEST_READY".into(),
        other => format!("UNKNOWN({})", other.as_raw()),
    }
}

/// True when the user asked for the capability report (`ZESTENGINE_GPU_CAPS=1`
/// or `--gpu-caps`); the renderer prints it after device creation and exits.
fn gpu_caps_requested() -> bool {
    std::env::var_os("ZESTENGINE_GPU_CAPS").is_some_and(|v| v != "0" && !v.is_empty())
        || std::env::args().any(|a| a == "--gpu-caps")
}

/// `VK_AMD_buffer_marker` target: word 0 = last marker the GPU started,
/// word 1 = last one everything before has finished.
struct MarkerBuffer {
    loader: amd::buffer_marker::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *const u32,
    names: std::sync::Mutex<Vec<&'static CStr>>,
}

// The mapping is host-coherent memory only ever read after the GPU has gone.
unsafe impl Send for MarkerBuffer {}
unsafe impl Sync for MarkerBuffer {}

impl MarkerBuffer {
    fn new(instance: &ash::Instance, pd: vk::PhysicalDevice, device: &ash::Device) -> Option<Self> {
        let buffer = unsafe {
            device.create_buffer(&vk::BufferCreateInfo::default().size(8).usage(vk::BufferUsageFlags::TRANSFER_DST), None)
        }
        .ok()?;
        let req = unsafe { device.get_buffer_memory_requirements(buffer) };
        let props = unsafe { instance.get_physical_device_memory_properties(pd) };
        let want = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let Some(ty) = (0..props.memory_type_count)
            .find(|&i| req.memory_type_bits & (1 << i) != 0 && props.memory_types[i as usize].property_flags.contains(want))
        else {
            unsafe { device.destroy_buffer(buffer, None) };
            return None;
        };
        let memory = unsafe { device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty), None) }.ok()?;
        unsafe { device.bind_buffer_memory(buffer, memory, 0) }.ok()?;
        let mapped = unsafe { device.map_memory(memory, 0, 8, vk::MemoryMapFlags::empty()) }.ok()? as *mut u32;
        unsafe { std::ptr::write_bytes(mapped, 0, 2) };
        Some(Self { loader: amd::buffer_marker::Device::new(instance, device), buffer, memory, mapped, names: std::sync::Mutex::new(Vec::new()) })
    }

    /// Marker id for a name (1-based; 0 = nothing yet).
    fn id(&self, name: &'static CStr) -> u32 {
        let mut names = self.names.lock().unwrap_or_else(|e| e.into_inner());
        let i = names.iter().position(|n| std::ptr::eq(*n, name)).unwrap_or_else(|| {
            names.push(name);
            names.len() - 1
        });
        i as u32 + 1
    }

    fn name(&self, id: u32) -> String {
        let names = self.names.lock().unwrap_or_else(|e| e.into_inner());
        match id {
            0 => "(none)".into(),
            i => names.get(i as usize - 1).map_or("?".into(), |n| n.to_string_lossy().into_owned()),
        }
    }
}

impl GpuContext {
    /// Whether the device can ray-trace (acceleration structures + ray query),
    /// which the lighting bake requires.
    pub fn ray_tracing(&self) -> bool {
        self.accel.is_some()
    }
}

impl GpuContext {
    pub fn new(
        display: RawDisplayHandle,
        window: RawWindowHandle,
        app_name: &CStr,
        xr: Option<XrBootstrap<'_>>,
    ) -> Result<Self> {
        let entry = unsafe { ash::Entry::load() }.context("loading Vulkan library")?;
        // The runtime needs our vkGetInstanceProcAddr (passed into the XR-routed
        // instance/device creation below). ash's PFN is ABI-identical to OpenXR's.
        let get_proc: openxr::sys::platform::VkGetInstanceProcAddr =
            unsafe { std::mem::transmute(entry.static_fn().get_instance_proc_addr) };

        // Ask for 1.3 when the loader offers it; a Vulkan 1.1 loader (older
        // Android) caps the instance at what it implements.
        let loader_version = unsafe { entry.try_enumerate_instance_version() }
            .ok()
            .flatten()
            .unwrap_or(vk::API_VERSION_1_0);
        if loader_version < vk::API_VERSION_1_1 {
            bail!("the Vulkan loader only implements {}; zestengine needs Vulkan 1.1 or newer", version_label(loader_version));
        }
        let instance_version = loader_version.min(vk::API_VERSION_1_3);
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"zestengine")
            .api_version(instance_version);

        let mut extensions: Vec<*const c_char> =
            ash_window::enumerate_required_extensions(display)?.to_vec();

        let validation = has_validation_layer(&entry);
        let mut layers: Vec<*const c_char> = Vec::new();
        if validation {
            layers.push(VALIDATION_LAYER.as_ptr());
            extensions.push(ext::debug_utils::NAME.as_ptr());
        }

        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&layers);
        let instance = match &xr {
            // xrCreateVulkanInstanceKHR merges the runtime's required instance
            // extensions with ours and records the proc-addr the runtime needs.
            Some(b) => {
                let raw = unsafe {
                    b.instance.create_vulkan_instance(
                        b.system,
                        get_proc,
                        &instance_info as *const _ as *const c_void,
                    )
                }
                .context("xrCreateVulkanInstanceKHR")?
                .map_err(|e| anyhow::anyhow!("xrCreateVulkanInstanceKHR: VkResult {e}"))?;
                let handle = vk::Instance::from_raw(raw as usize as u64);
                unsafe { ash::Instance::load(entry.static_fn(), handle) }
            }
            None => unsafe { entry.create_instance(&instance_info, None) }
                .context("creating Vulkan instance")?,
        };

        let debug = if validation {
            let loader = ext::debug_utils::Instance::new(&entry, &instance);
            let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_callback));
            let messenger = unsafe { loader.create_debug_utils_messenger(&info, None)? };
            tracing::info!("Vulkan validation layer enabled");
            Some((loader, messenger))
        } else {
            tracing::warn!("VK_LAYER_KHRONOS_validation not found; running without validation");
            None
        };

        let surface_loader = khr::surface::Instance::new(&entry, &instance);
        let surface =
            unsafe { ash_window::create_surface(&entry, &instance, display, window, None) }
                .context("creating window surface")?;

        let (physical_device, queue_family) = match &xr {
            // The runtime dictates the GPU (xrGetVulkanGraphicsDevice2KHR); we just
            // find a graphics+present queue family on it.
            Some(b) => {
                let inst_ptr = instance.handle().as_raw() as usize as *const c_void;
                let pd_raw = unsafe { b.instance.vulkan_graphics_device(b.system, inst_ptr) }
                    .context("xrGetVulkanGraphicsDevice2KHR")?;
                let pd = vk::PhysicalDevice::from_raw(pd_raw as usize as u64);
                let family = present_queue_family(&instance, &surface_loader, surface, pd)
                    .context("OpenXR's required GPU has no graphics+present queue family")?;
                (pd, family)
            }
            None => pick_device(&instance, &surface_loader, surface)?,
        };
        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        let gpu_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) };
        let api_version = props.api_version.min(instance_version);
        if api_version < vk::API_VERSION_1_1 {
            bail!("{gpu_name:?} only offers {}; zestengine needs Vulkan 1.1 or newer", version_label(props.api_version));
        }
        tracing::info!(gpu = ?gpu_name, api = %version_label(api_version), "selected physical device");

        // Request a SECOND queue from the same family (when the family exposes
        // ≥2) to use as a background asset-upload queue: a loader thread can
        // submit transfers on it independently of the render thread (distinct
        // VkQueues need no cross-synchronization; same family → no ownership
        // transfer for the uploaded resources). Falls back to one queue (uploads
        // then stay on the main thread).
        let queue_count = unsafe {
            instance
                .get_physical_device_queue_family_properties(physical_device)
                .get(queue_family as usize)
                .map(|q| q.queue_count)
                .unwrap_or(1)
        };
        let want_transfer = queue_count >= 2;
        let queue_priorities: &[f32] = if want_transfer { &[1.0, 0.5] } else { &[1.0] };
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(queue_priorities)];

        // Ray query (for the lighting bake) needs acceleration structures +
        // ray query + deferred host ops + buffer device address. Enable them
        // only when the device advertises all of it; otherwise the engine runs
        // fine without baking.
        let rt_exts = [
            khr::acceleration_structure::NAME,
            khr::ray_query::NAME,
            khr::deferred_host_operations::NAME,
        ];
        // The RT feature chain uses Vulkan12Features, which needs a 1.2 device.
        let mut has_rt = api_version >= vk::API_VERSION_1_2 && device_supports(&instance, physical_device, &rt_exts)?;
        // Ray-traced hit shading samples every material texture through one
        // runtime-sized array (descriptor indexing). VK_KHR_acceleration_structure
        // requires the extension, so RT devices have it; check the features anyway.
        if has_rt {
            let mut q12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut q12);
            unsafe { instance.get_physical_device_features2(physical_device, &mut f2) };
            let indexing = q12.runtime_descriptor_array == vk::TRUE
                && q12.shader_sampled_image_array_non_uniform_indexing == vk::TRUE
                && q12.descriptor_binding_partially_bound == vk::TRUE
                && q12.scalar_block_layout == vk::TRUE;
            if !indexing {
                tracing::warn!("GPU lacks descriptor indexing or scalar block layout; ray tracing disabled");
                has_rt = false;
            }
        }

        let mut device_extensions = vec![khr::swapchain::NAME.as_ptr()];
        if has_rt {
            for e in rt_exts {
                device_extensions.push(e.as_ptr());
            }
        }
        let has_checkpoints = device_supports(&instance, physical_device, &[nv::device_diagnostic_checkpoints::NAME])?;
        if has_checkpoints {
            device_extensions.push(nv::device_diagnostic_checkpoints::NAME.as_ptr());
        }
        let has_markers = !has_checkpoints && device_supports(&instance, physical_device, &[amd::buffer_marker::NAME])?;
        if has_markers {
            device_extensions.push(amd::buffer_marker::NAME.as_ptr());
        }
        // The driver's own fault report (supported on AMD and recent NVIDIA).
        let has_fault = device_supports(&instance, physical_device, &[ext::device_fault::NAME])? && {
            let mut f = vk::PhysicalDeviceFaultFeaturesEXT::default();
            let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f);
            unsafe { instance.get_physical_device_features2(physical_device, &mut f2) };
            f.device_fault == vk::TRUE
        };
        if has_fault {
            device_extensions.push(ext::device_fault::NAME.as_ptr());
        }
        let mut fault_features = vk::PhysicalDeviceFaultFeaturesEXT::default().device_fault(true);

        // Dynamic rendering + synchronization2: core on 1.3, otherwise the KHR
        // extensions (the device table aliases their entry points onto the core
        // names below, so call sites are the same on both paths).
        let core13 = api_version >= vk::API_VERSION_1_3;
        let astc_hdr_ext = !core13 && device_supports(&instance, physical_device, &[ext::texture_compression_astc_hdr::NAME])?;
        let (multiview, multiview_tess, astc_hdr, shader_demote) = {
            let mut mv = vk::PhysicalDeviceMultiviewFeatures::default();
            let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut hdr = vk::PhysicalDeviceTextureCompressionASTCHDRFeatures::default();
            let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut mv);
            if core13 {
                f2 = f2.push_next(&mut f13);
            } else if astc_hdr_ext {
                f2 = f2.push_next(&mut hdr);
            }
            unsafe { instance.get_physical_device_features2(physical_device, &mut f2) };
            let astc_hdr = if core13 { f13.texture_compression_astc_hdr } else { hdr.texture_compression_astc_hdr };
            (mv.multiview == vk::TRUE, mv.multiview_tessellation_shader == vk::TRUE, astc_hdr == vk::TRUE,
                f13.shader_demote_to_helper_invocation == vk::TRUE)
        };
        if !core13 {
            let mut required = vec![khr::dynamic_rendering::NAME, khr::synchronization2::NAME];
            if api_version < vk::API_VERSION_1_2 {
                // VK_KHR_dynamic_rendering depends on these (core in 1.2).
                required.extend([khr::depth_stencil_resolve::NAME, khr::create_renderpass2::NAME]);
            }
            for name in required {
                if !device_supports(&instance, physical_device, &[name])? {
                    bail!(
                        "{gpu_name:?} offers {} without {}: zestengine needs Vulkan 1.3 or that extension",
                        version_label(api_version),
                        name.to_string_lossy()
                    );
                }
                device_extensions.push(name.as_ptr());
            }
            let mut dr = vk::PhysicalDeviceDynamicRenderingFeatures::default();
            let mut s2 = vk::PhysicalDeviceSynchronization2Features::default();
            let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut dr).push_next(&mut s2);
            unsafe { instance.get_physical_device_features2(physical_device, &mut f2) };
            if dr.dynamic_rendering != vk::TRUE {
                bail!("{gpu_name:?} lists VK_KHR_dynamic_rendering but not its dynamicRendering feature");
            }
            if s2.synchronization2 != vk::TRUE {
                bail!("{gpu_name:?} lists VK_KHR_synchronization2 but not its synchronization2 feature");
            }
            if astc_hdr {
                device_extensions.push(ext::texture_compression_astc_hdr::NAME.as_ptr());
            }
            tracing::info!("Vulkan {} device: dynamic rendering + synchronization2 via KHR extensions", version_label(api_version));
        }
        let mut vk13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true)
            .texture_compression_astc_hdr(astc_hdr)
            .shader_demote_to_helper_invocation(shader_demote);
        let mut dr_khr = vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
        let mut s2_khr = vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
        let mut astc_hdr_khr = vk::PhysicalDeviceTextureCompressionASTCHDRFeatures::default().texture_compression_astc_hdr(true);
        let mut multiview_features = vk::PhysicalDeviceMultiviewFeatures::default().multiview(true).multiview_tessellation_shader(multiview_tess);
        let supported = unsafe { instance.get_physical_device_features(physical_device) };
        let tessellation = supported.tessellation_shader == vk::TRUE;

        // GPU-driven rendering features (GEO-4): enabled when the device has
        // them, all optional. Vulkan 1.2 features come through one struct.
        let mut q12 = vk::PhysicalDeviceVulkan12Features::default();
        if api_version >= vk::API_VERSION_1_2 {
            let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut q12);
            unsafe { instance.get_physical_device_features2(physical_device, &mut f2) };
        }
        let bda = api_version >= vk::API_VERSION_1_2 && q12.buffer_device_address == vk::TRUE;
        let count_core = api_version >= vk::API_VERSION_1_2 && q12.draw_indirect_count == vk::TRUE;
        // 1.1 devices expose it as an extension (no feature struct).
        let count_ext = api_version < vk::API_VERSION_1_2
            && device_supports(&instance, physical_device, &[khr::draw_indirect_count::NAME])?;
        if count_ext {
            device_extensions.push(khr::draw_indirect_count::NAME.as_ptr());
        }
        // Mesh shaders: probed and recorded only, never enabled.
        let mesh_shader = device_supports(&instance, physical_device, &[ext::mesh_shader::NAME])? && {
            let mut m = vk::PhysicalDeviceMeshShaderFeaturesEXT::default();
            let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut m);
            unsafe { instance.get_physical_device_features2(physical_device, &mut f2) };
            m.mesh_shader == vk::TRUE && m.task_shader == vk::TRUE
        };
        // FIFO_LATEST_READY presenting: opt-in at swapchain creation, but the
        // extension + feature must be enabled on the device for it to work.
        let flr_ext = [FLR_EXT_NAME, FLR_KHR_NAME]
            .into_iter()
            .find(|n| device_supports(&instance, physical_device, &[*n]).unwrap_or(false));
        let present_fifo_latest_ready = flr_ext.is_some() && {
            let mut f = PresentModeFifoLatestReadyFeatures::default();
            let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f);
            unsafe { instance.get_physical_device_features2(physical_device, &mut f2) };
            f.present_mode_fifo_latest_ready == vk::TRUE
        };
        if present_fifo_latest_ready && let Some(n) = flr_ext {
            device_extensions.push(n.as_ptr());
        }
        let mut flr_features = PresentModeFifoLatestReadyFeatures { present_mode_fifo_latest_ready: vk::TRUE, ..Default::default() };
        let caps = GpuCaps {
            present_fifo_latest_ready,
            api_version,
            device_name: gpu_name.to_string_lossy().into_owned(),
            multi_draw_indirect: supported.multi_draw_indirect == vk::TRUE,
            draw_indirect_count: count_core || count_ext,
            shader_int64: supported.shader_int64 == vk::TRUE,
            buffer_device_address: bda || has_rt,
            shader_buffer_int64_atomics: q12.shader_buffer_int64_atomics == vk::TRUE,
            mesh_shader,
            ray_tracing: has_rt,
            multiview,
            max_draw_indirect_count: props.limits.max_draw_indirect_count,
            max_storage_buffer_range: props.limits.max_storage_buffer_range,
            max_compute_work_group_count: props.limits.max_compute_work_group_count,
        };
        let mut vk12 = vk::PhysicalDeviceVulkan12Features::default()
            .scalar_block_layout(q12.scalar_block_layout == vk::TRUE)
            .buffer_device_address(caps.buffer_device_address)
            .draw_indirect_count(count_core)
            .runtime_descriptor_array(has_rt)
            .shader_sampled_image_array_non_uniform_indexing(has_rt)
            .descriptor_binding_partially_bound(has_rt);
        let mut as_features = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
            .acceleration_structure(true);
        let mut rq_features = vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(true);

        // BC (S3TC/BPTC) compressed textures for the texture import cache. Core
        // on essentially all desktop GPUs; enable only if reported to avoid
        // failing device creation on the rare GPU without it.
        let bc = supported.texture_compression_bc == vk::TRUE;
        if !bc {
            tracing::warn!("GPU lacks textureCompressionBC; textures upload uncompressed");
        }
        crate::set_bc_supported(bc);
        crate::set_depth_format(pick_depth_format(&instance, physical_device));
        // E5B9G9R9: the compact fallback for baked HDR data on GPUs without BC.
        let e5 = unsafe {
            instance.get_physical_device_format_properties(physical_device, vk::Format::E5B9G9R9_UFLOAT_PACK32)
        }
        .optimal_tiling_features;
        crate::set_rgb9e5_supported(e5.contains(
            vk::FormatFeatureFlags::SAMPLED_IMAGE
                | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR
                | vk::FormatFeatureFlags::TRANSFER_DST,
        ));
        // ASTC: the mobile path for colour / data textures (no BC on Adreno).
        let astc = supported.texture_compression_astc_ldr == vk::TRUE;
        crate::set_astc_supported(astc, astc_hdr);
        if !bc {
            tracing::info!("ASTC textures: LDR={astc} HDR={astc_hdr}");
        }
        let features = vk::PhysicalDeviceFeatures::default()
            .texture_compression_bc(bc)
            .texture_compression_astc_ldr(astc)
            .sampler_anisotropy(supported.sampler_anisotropy == vk::TRUE)
            .multi_draw_indirect(caps.multi_draw_indirect)
            .shader_int64(caps.shader_int64)
            .tessellation_shader(tessellation)
            .geometry_shader(supported.geometry_shader == vk::TRUE)
            .independent_blend(supported.independent_blend == vk::TRUE);

        let mut device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extensions)
            .enabled_features(&features);
        if core13 {
            device_info = device_info.push_next(&mut vk13);
        } else {
            device_info = device_info.push_next(&mut dr_khr).push_next(&mut s2_khr);
            if astc_hdr {
                device_info = device_info.push_next(&mut astc_hdr_khr);
            }
        }
        if multiview {
            device_info = device_info.push_next(&mut multiview_features);
        }
        if api_version >= vk::API_VERSION_1_2 {
            device_info = device_info.push_next(&mut vk12);
        }
        if has_rt {
            device_info = device_info.push_next(&mut as_features).push_next(&mut rq_features);
        }
        if has_fault {
            device_info = device_info.push_next(&mut fault_features);
        }
        if present_fifo_latest_ready {
            device_info = device_info.push_next(&mut flr_features);
        }
        let device = match &xr {
            // xrCreateVulkanDeviceKHR adds the runtime's required device extensions
            // (external memory/fd, timeline semaphores for the compositor hand-off)
            // on top of ours, so the session can present this device's images.
            Some(b) => {
                let raw = unsafe {
                    b.instance.create_vulkan_device(
                        b.system,
                        get_proc,
                        physical_device.as_raw() as usize as *const c_void,
                        &device_info as *const _ as *const c_void,
                    )
                }
                .context("xrCreateVulkanDeviceKHR")?
                .map_err(|e| anyhow::anyhow!("xrCreateVulkanDeviceKHR: VkResult {e}"))?;
                let handle = vk::Device::from_raw(raw as usize as u64);
                unsafe { ash::Device::load(instance.fp_v1_0(), handle) }
            }
            None => unsafe { instance.create_device(physical_device, &device_info, None) }
                .context("creating logical device")?,
        };
        let device = if core13 { device } else { unsafe { load_with_khr_aliases(&instance, device.handle()) } };
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let transfer_queue = if want_transfer {
            tracing::info!("background asset-upload queue enabled");
            Some(unsafe { device.get_device_queue(queue_family, 1) })
        } else {
            tracing::warn!("only one queue available; asset uploads stay on the main thread");
            None
        };

        let accel = if has_rt {
            tracing::info!("ray query supported; lighting bake enabled");
            Some(khr::acceleration_structure::Device::new(&instance, &device))
        } else {
            tracing::warn!("ray query unsupported on this device; lighting bake disabled");
            None
        };

        let checkpoints = has_checkpoints.then(|| nv::device_diagnostic_checkpoints::Device::new(&instance, &device));
        let markers = if has_markers { MarkerBuffer::new(&instance, physical_device, &device) } else { None };
        let fault = has_fault.then(|| ext::device_fault::Device::new(&instance, &device));
        if checkpoints.is_some() || markers.is_some() || fault.is_some() {
            tracing::info!(
                "GPU crash diagnostics: checkpoints={} markers={} fault report={}",
                checkpoints.is_some(),
                markers.is_some(),
                fault.is_some()
            );
        }
        tracing::info!("{}", caps.summary());
        if gpu_caps_requested() {
            println!("{}", caps.report());
            std::process::exit(0);
        }
        Ok(Self {
            entry,
            instance,
            debug,
            surface_loader,
            surface,
            physical_device,
            device,
            queue_family,
            queue,
            transfer_queue,
            accel,
            checkpoints,
            markers,
            fault,
            lost_reported: std::sync::atomic::AtomicBool::new(false),
            api_version,
            multiview,
            tessellation,
            multiview_tessellation: tessellation && multiview && multiview_tess,
            caps,
        })
    }

    /// Mark where `cb` is in its work; after a device loss
    /// [`Self::report_device_lost`] names the last marks the GPU reached.
    pub fn checkpoint(&self, cb: vk::CommandBuffer, marker: &'static CStr) {
        if let Some(c) = &self.checkpoints {
            unsafe { c.cmd_set_checkpoint(cb, marker.as_ptr().cast()) };
        } else if let Some(m) = &self.markers {
            let id = m.id(marker);
            unsafe {
                m.loader.cmd_write_buffer_marker(cb, vk::PipelineStageFlags::TOP_OF_PIPE, m.buffer, 0, id);
                m.loader.cmd_write_buffer_marker(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, m.buffer, 4, id);
            }
        }
    }

    /// If `err` is a lost device, log (once) the last checkpoints the GPU
    /// reached on the main queue: the pass that hung or faulted.
    pub fn report_device_lost(&self, err: &anyhow::Error) {
        let lost = err.chain().any(|e| e.downcast_ref::<vk::Result>() == Some(&vk::Result::ERROR_DEVICE_LOST))
            || err.to_string().contains("device has been lost");
        if !lost || self.lost_reported.swap(true, std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        if let Some(c) = &self.checkpoints {
            let n = unsafe { c.get_queue_checkpoint_data_len(self.queue) };
            let mut data = vec![vk::CheckpointDataNV::default(); n];
            unsafe { c.get_queue_checkpoint_data(self.queue, &mut data) };
            if data.is_empty() {
                tracing::error!("GPU device lost; no checkpoints recorded");
            }
            for d in &data {
                // Markers are &'static CStr names set by `checkpoint`.
                let name = if d.p_checkpoint_marker.is_null() {
                    "?".into()
                } else {
                    unsafe { CStr::from_ptr(d.p_checkpoint_marker.cast()) }.to_string_lossy()
                };
                tracing::error!("GPU device lost: last checkpoint at stage {:?}: {name}", d.stage);
            }
        } else if let Some(m) = &self.markers {
            let (started, finished) = unsafe { (std::ptr::read_volatile(m.mapped), std::ptr::read_volatile(m.mapped.add(1))) };
            tracing::error!(
                "GPU device lost: last marker started: {}; last finished: {} (the work between them hung or faulted)",
                m.name(started),
                m.name(finished)
            );
        } else {
            tracing::error!("GPU device lost (this GPU offers no checkpoints or markers to say where)");
        }
        self.log_fault();
    }
}

impl GpuContext {
    /// The driver's fault report (VK_EXT_device_fault): what went wrong and
    /// at which GPU addresses.
    fn log_fault(&self) {
        let Some(f) = &self.fault else { return };
        let get = f.fp().get_device_fault_info_ext;
        let mut counts = vk::DeviceFaultCountsEXT::default();
        if unsafe { get(self.device.handle(), &mut counts, std::ptr::null_mut()) } != vk::Result::SUCCESS {
            return;
        }
        let mut addresses = vec![vk::DeviceFaultAddressInfoEXT::default(); counts.address_info_count as usize];
        let mut vendors = vec![vk::DeviceFaultVendorInfoEXT::default(); counts.vendor_info_count as usize];
        counts.vendor_binary_size = 0;
        let mut info = vk::DeviceFaultInfoEXT {
            p_address_infos: addresses.as_mut_ptr(),
            p_vendor_infos: vendors.as_mut_ptr(),
            ..Default::default()
        };
        if unsafe { get(self.device.handle(), &mut counts, &mut info) } != vk::Result::SUCCESS {
            return;
        }
        let text = |c: &[std::ffi::c_char]| unsafe { CStr::from_ptr(c.as_ptr()) }.to_string_lossy().into_owned();
        tracing::error!("GPU fault report: {}", text(&info.description));
        for a in addresses.iter().take(counts.address_info_count as usize) {
            tracing::error!("GPU fault address: {:?} at {:#x} (+/- {:#x})", a.address_type, a.reported_address, a.address_precision);
        }
        for v in vendors.iter().take(counts.vendor_info_count as usize) {
            tracing::error!("GPU fault vendor info: {} (code {:#x}, data {:#x})", text(&v.description), v.vendor_fault_code, v.vendor_fault_data);
        }
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.wait_idle("wait:device_wait_idle@drop");
            if let Some(m) = self.markers.take() {
                self.device.destroy_buffer(m.buffer, None);
                self.device.free_memory(m.memory, None);
            }
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            if let Some((loader, messenger)) = self.debug.take() {
                loader.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

/// "Vulkan 1.x" for a packed API version.
fn version_label(v: u32) -> String {
    format!("Vulkan {}.{}", vk::api_version_major(v), vk::api_version_minor(v))
}

/// Load the device table so the core 1.3 names (`cmd_begin_rendering`,
/// `cmd_pipeline_barrier2`, `queue_submit2`, ...) dispatch to the KHR
/// extension entry points on a 1.1 / 1.2 device. The KHR commands have the
/// same signatures and structs as their core promotions; they are preferred
/// because a pre-1.3 driver may resolve the core names to stubs.
unsafe fn load_with_khr_aliases(instance: &ash::Instance, device: vk::Device) -> ash::Device {
    let get = instance.fp_v1_0().get_device_proc_addr;
    unsafe {
        ash::Device::load_with(
            |name| {
                let mut khr = name.to_bytes().to_vec();
                khr.extend_from_slice(b"KHR\0");
                let alias = CStr::from_bytes_with_nul(&khr).ok().and_then(|n| get(device, n.as_ptr()));
                alias
                    .or_else(|| get(device, name.as_ptr()))
                    .map_or(std::ptr::null(), |f| f as *const c_void)
            },
            device,
        )
    }
}

/// Whether `pd` advertises every extension in `wanted`.
fn device_supports(
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
    wanted: &[&CStr],
) -> Result<bool> {
    let exts = unsafe { instance.enumerate_device_extension_properties(pd)? };
    let available: Vec<&CStr> = exts
        .iter()
        .map(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) })
        .collect();
    Ok(wanted.iter().all(|w| available.contains(w)))
}

fn has_validation_layer(entry: &ash::Entry) -> bool {
    let Ok(layers) = (unsafe { entry.enumerate_instance_layer_properties() }) else {
        return false;
    };
    layers.iter().any(|l| {
        let name = unsafe { CStr::from_ptr(l.layer_name.as_ptr()) };
        name == VALIDATION_LAYER
    })
}

/// First queue family on `pd` that supports both graphics and presenting to
/// `surface`. Used when OpenXR already chose the physical device for us.
fn present_queue_family(
    instance: &ash::Instance,
    surface_loader: &khr::surface::Instance,
    surface: vk::SurfaceKHR,
    pd: vk::PhysicalDevice,
) -> Option<u32> {
    let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
    families.iter().enumerate().find_map(|(i, f)| {
        let graphics = f.queue_flags.contains(vk::QueueFlags::GRAPHICS);
        let present = unsafe {
            surface_loader
                .get_physical_device_surface_support(pd, i as u32, surface)
                .unwrap_or(false)
        };
        (graphics && present).then_some(i as u32)
    })
}

/// Pick the best physical device that can render to `surface`, preferring
/// discrete GPUs. Returns the device and a graphics+present queue family.
fn pick_device(
    instance: &ash::Instance,
    surface_loader: &khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32)> {
    let devices = unsafe { instance.enumerate_physical_devices()? };
    let mut best: Option<(vk::PhysicalDevice, u32, u32)> = None;

    for pd in devices {
        let exts = unsafe { instance.enumerate_device_extension_properties(pd)? };
        let has_swapchain = exts.iter().any(|e| {
            let name = unsafe { CStr::from_ptr(e.extension_name.as_ptr()) };
            name == khr::swapchain::NAME
        });
        if !has_swapchain {
            continue;
        }

        let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
        let family = families.iter().enumerate().find_map(|(i, f)| {
            let graphics = f.queue_flags.contains(vk::QueueFlags::GRAPHICS);
            let present = unsafe {
                surface_loader
                    .get_physical_device_surface_support(pd, i as u32, surface)
                    .unwrap_or(false)
            };
            (graphics && present).then_some(i as u32)
        });
        let Some(family) = family else { continue };

        let props = unsafe { instance.get_physical_device_properties(pd) };
        let score = match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 100,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 50,
            _ => 10,
        };
        if best.is_none_or(|(_, _, s)| score > s) {
            best = Some((pd, family, score));
        }
    }

    match best {
        Some((pd, family, _)) => Ok((pd, family)),
        None => bail!("no Vulkan device supports rendering to this surface"),
    }
}

// GPU regressions must observe validation failures even without a tracing
// subscriber. Keep this test-only; production logging remains unchanged.
#[cfg(test)]
static VALIDATION_ERRORS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub(crate) fn validation_error_count() -> usize {
    VALIDATION_ERRORS.load(std::sync::atomic::Ordering::Relaxed)
}

extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut c_void,
) -> vk::Bool32 {
    let message = unsafe {
        let data = &*data;
        if data.p_message.is_null() {
            "<no message>".into()
        } else {
            CStr::from_ptr(data.p_message).to_string_lossy()
        }
    };
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        #[cfg(test)]
        {
            VALIDATION_ERRORS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            eprintln!("[Vulkan validation] {message}");
        }
        tracing::error!(target: "vulkan", "{message}");
        if std::env::var_os("ZESTENGINE_VALIDATION_BACKTRACE").is_some() {
            tracing::error!(target: "vulkan", "{}", std::backtrace::Backtrace::force_capture());
        }
    } else {
        tracing::warn!(target: "vulkan", "{message}");
    }
    vk::FALSE
}

/// Scene depth(-stencil) format for this device: the first of
/// `D32_SFLOAT_S8_UINT`, `D24_UNORM_S8_UINT`, `D16_UNORM_S8_UINT` that supports
/// optimal-tiling depth-stencil attachment + sampled image + transfer source
/// (Hi-Z and SSR sample the depth aspect; readbacks copy it). Falls back to the
/// plain `D32_SFLOAT` fast path (no stencil) when none does, and
/// `ZESTENGINE_STENCIL=0` forces that fast path (a scene that never uses
/// stencil then pays nothing for it).
pub(crate) fn pick_depth_format(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> vk::Format {
    if std::env::var("ZESTENGINE_STENCIL").is_ok_and(|v| v == "0") {
        tracing::info!("scene depth: D32_SFLOAT (stencil disabled by ZESTENGINE_STENCIL=0)");
        return vk::Format::D32_SFLOAT;
    }
    let want = vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT | vk::FormatFeatureFlags::SAMPLED_IMAGE | vk::FormatFeatureFlags::TRANSFER_SRC;
    let fmt = [vk::Format::D32_SFLOAT_S8_UINT, vk::Format::D24_UNORM_S8_UINT, vk::Format::D16_UNORM_S8_UINT]
        .into_iter()
        .find(|f| {
            let p = unsafe { instance.get_physical_device_format_properties(physical_device, *f) };
            p.optimal_tiling_features.contains(want)
        })
        .unwrap_or(vk::Format::D32_SFLOAT);
    tracing::info!("scene depth format: {fmt:?}");
    fmt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn present_mode_names_are_readable() {
        assert_eq!(present_mode_name(vk::PresentModeKHR::MAILBOX), "MAILBOX");
        assert_eq!(present_mode_name(PRESENT_MODE_FIFO_LATEST_READY), "FIFO_LATEST_READY");
        assert_eq!(present_mode_name(vk::PresentModeKHR::from_raw(42)), "UNKNOWN(42)");
    }

    #[test]
    fn caps_report_and_summary_name_every_bit() {
        let mut c = GpuCaps { device_name: "Test GPU".into(), api_version: vk::API_VERSION_1_3, ..Default::default() };
        let r = c.report();
        assert!(r.contains("Test GPU") && r.contains("Vulkan 1.3"));
        assert!(r.contains("per-batch vkCmdDrawIndexedIndirect"), "missing features pick the fallback");
        c.multi_draw_indirect = true;
        c.draw_indirect_count = true;
        c.max_compute_work_group_count = [65535; 3];
        assert!(c.report().contains("multi-draw-indirect-count"));
        let s = c.summary();
        assert!(s.contains("multiDrawIndirect=yes") && s.contains("shaderInt64=no") && s.contains("65535"));
        assert_eq!(s.lines().count(), 1);
    }
}
