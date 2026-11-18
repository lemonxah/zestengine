#![allow(dead_code)]

use std::alloc::{Layout, alloc, dealloc};
use std::ffi::c_void;
use std::ptr;

use anyhow::{Result, bail};
use ash::vk;
use ash::vk::Handle;

use super::UpscaleInput;
use super::ffx::*;

static FFX_MESSAGES: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

fn wide_text(message: *const u32) -> String {
    if message.is_null() {
        return String::new();
    }
    let mut text = String::new();
    for k in 0..512 {
        let unit = unsafe { *message.add(k) };
        if unit == 0 {
            break;
        }
        text.push(char::from_u32(unit).unwrap_or('?'));
    }
    text
}

fn take_ffx_messages() -> String {
    let mut held = FFX_MESSAGES.lock().unwrap_or_else(|e| e.into_inner());
    let text = held.join(" | ");
    held.clear();
    text
}

unsafe extern "C" fn ffx_message(type_: FfxMsgType, message: *const u32) {
    let text = wide_text(message);
    match type_ {
        0 => tracing::error!("FFX: {text}"),
        1 => tracing::warn!("FFX: {text}"),
        _ => tracing::debug!("FFX: {text}"),
    }
    let mut held = FFX_MESSAGES.lock().unwrap_or_else(|e| e.into_inner());
    if held.len() < 8 {
        held.push(text);
    }
}

/// Alignment the FFX backend requires for its host scratch buffer.
const SCRATCH_ALIGN: usize = 256;
/// Contexts sharing one backend interface (upscaler + frame interpolation).
const MAX_CONTEXTS: usize = 2;

/// Formats the upscaler consumes each frame. These are the engine's actual
/// attachment formats; `ffx_format` maps them to the FFX equivalents.
#[derive(Clone, Copy, Debug)]
pub struct Fsr3Formats {
    /// Reduced-resolution HDR colour (e.g. `R16G16B16A16_SFLOAT`).
    pub color: vk::Format,
    /// Reduced-resolution depth (must be a 32-bit float view for FSR).
    pub depth: vk::Format,
    /// Reduced-resolution motion vectors (RG16F).
    pub motion: vk::Format,
    /// Reduced-resolution reactive mask (R8).
    pub reactive: vk::Format,
    /// Full-resolution output the upscaler writes.
    pub output: vk::Format,
}

/// Maps a Vulkan format to the FFX surface format FSR samples. `None` for
/// formats the upscaler cannot consume.
pub fn ffx_format(format: vk::Format) -> Option<FfxSurfaceFormat> {
    use vk::Format as F;
    Some(match format {
        F::R16G16B16A16_SFLOAT => FfxSurfaceFormat_FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT,
        F::R16G16_SFLOAT => FfxSurfaceFormat_FFX_SURFACE_FORMAT_R16G16_FLOAT,
        F::R32_SFLOAT => FfxSurfaceFormat_FFX_SURFACE_FORMAT_R32_FLOAT,
        F::R32G32_SFLOAT => FfxSurfaceFormat_FFX_SURFACE_FORMAT_R32G32_FLOAT,
        F::R8_UNORM => FfxSurfaceFormat_FFX_SURFACE_FORMAT_R8_UNORM,
        F::R8G8B8A8_UNORM => FfxSurfaceFormat_FFX_SURFACE_FORMAT_R8G8B8A8_UNORM,
        F::R8G8B8A8_SRGB => FfxSurfaceFormat_FFX_SURFACE_FORMAT_R8G8B8A8_SRGB,
        F::R8_UINT => FfxSurfaceFormat_FFX_SURFACE_FORMAT_R8_UINT,
        F::R16_UINT => FfxSurfaceFormat_FFX_SURFACE_FORMAT_R16_UINT,
        F::R32_UINT => FfxSurfaceFormat_FFX_SURFACE_FORMAT_R32_UINT,
        F::B10G11R11_UFLOAT_PACK32 => FfxSurfaceFormat_FFX_SURFACE_FORMAT_R11G11B10_FLOAT,
        _ => return None,
    })
}

/// Builds the FSR3 upscaler context flags from the renderer's depth/HDR
/// conventions.
pub fn upscaler_flags(hdr: bool, depth_inverted: bool, depth_infinite: bool) -> u32 {
    let mut flags = 0;
    if hdr {
        flags |= FfxFsr3UpscalerInitializationFlagBits_FFX_FSR3UPSCALER_ENABLE_HIGH_DYNAMIC_RANGE;
    }
    if depth_inverted {
        flags |= FfxFsr3UpscalerInitializationFlagBits_FFX_FSR3UPSCALER_ENABLE_DEPTH_INVERTED;
    }
    if depth_infinite {
        flags |= FfxFsr3UpscalerInitializationFlagBits_FFX_FSR3UPSCALER_ENABLE_DEPTH_INFINITE;
    }
    flags
}

fn dims2(width: u32, height: u32) -> FfxDimensions2D {
    FfxDimensions2D { width, height }
}

fn resource_description(
    format: FfxSurfaceFormat,
    width: u32,
    height: u32,
    usage: FfxResourceUsage,
) -> FfxResourceDescription {
    FfxResourceDescription {
        type_: FfxResourceType_FFX_RESOURCE_TYPE_TEXTURE2D,
        format,
        __bindgen_anon_1: FfxResourceDescription__bindgen_ty_1 { width },
        __bindgen_anon_2: FfxResourceDescription__bindgen_ty_2 { height },
        __bindgen_anon_3: FfxResourceDescription__bindgen_ty_3 { depth: 1 },
        mipCount: 1,
        flags: FfxResourceFlags_FFX_RESOURCE_FLAGS_NONE,
        usage,
    }
}

/// Wraps a `VkImageView` as an FFX resource. A null view (optional inputs)
/// becomes a null resource the backend ignores.
fn resource(
    image: vk::Image,
    format: FfxSurfaceFormat,
    width: u32,
    height: u32,
    usage: FfxResourceUsage,
    state: FfxResourceStates,
) -> FfxResource {
    if image == vk::Image::null() {
        return FfxResource {
            resource: ptr::null_mut(),
            description: resource_description(format, width, height, usage),
            state,
            name: [0; 64],
        };
    }
    unsafe {
        ffxGetResourceVK(
            image.as_raw() as usize as *mut c_void,
            resource_description(format, width, height, usage),
            ptr::null(),
            state,
        )
    }
}

/// Owns the FFX Vulkan backend interface, its host scratch buffer, and the
/// FSR3 upscaler context.
struct SharedImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
    description: FfxResourceDescription,
}

#[allow(non_upper_case_globals)]
fn shared_format(format: FfxSurfaceFormat) -> Option<vk::Format> {
    Some(match format {
        FfxSurfaceFormat_FFX_SURFACE_FORMAT_R32_FLOAT => vk::Format::R32_SFLOAT,
        FfxSurfaceFormat_FFX_SURFACE_FORMAT_R16G16_FLOAT => vk::Format::R16G16_SFLOAT,
        FfxSurfaceFormat_FFX_SURFACE_FORMAT_R32_UINT => vk::Format::R32_UINT,
        FfxSurfaceFormat_FFX_SURFACE_FORMAT_R16G16_SINT => vk::Format::R16G16_SINT,
        FfxSurfaceFormat_FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT => vk::Format::R16G16B16A16_SFLOAT,
        FfxSurfaceFormat_FFX_SURFACE_FORMAT_R8_UNORM => vk::Format::R8_UNORM,
        _ => return None,
    })
}

fn create_shared_image(device: &ash::Device, instance: &ash::Instance, physical_device: vk::PhysicalDevice, description: FfxResourceDescription) -> Result<SharedImage> {
    let (width, height) = unsafe { (description.__bindgen_anon_1.width, description.__bindgen_anon_2.height) };
    let format = shared_format(description.format).ok_or_else(|| anyhow::anyhow!("FSR3: shared resource format {} is not handled", description.format))?;
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D { width: width.max(1), height: height.max(1), depth: 1 })
        .mip_levels(description.mipCount.max(1))
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { device.create_image(&info, None)? };
    let requirements = unsafe { device.get_image_memory_requirements(image) };
    let properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let memory_type = (0..properties.memory_type_count).find(|i| {
        requirements.memory_type_bits & (1 << i) != 0 && properties.memory_types[*i as usize].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
    });
    let Some(memory_type) = memory_type else {
        unsafe { device.destroy_image(image, None) };
        bail!("FSR3: no device-local memory type for a shared resource");
    };
    let memory = match unsafe { device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(requirements.size).memory_type_index(memory_type), None) } {
        Ok(memory) => memory,
        Err(e) => {
            unsafe { device.destroy_image(image, None) };
            bail!("FSR3: allocating a shared resource: {e}");
        }
    };
    if let Err(e) = unsafe { device.bind_image_memory(image, memory, 0) } {
        unsafe {
            device.destroy_image(image, None);
            device.free_memory(memory, None);
        }
        bail!("FSR3: binding a shared resource: {e}");
    }
    Ok(SharedImage { image, memory, description })
}

pub struct Fsr3 {
    interface: FfxInterface,
    device: FfxDevice,
    vk_device: ash::Device,
    vk_context: Box<VkDeviceContext>,
    scratch: *mut u8,
    scratch_layout: Layout,
    context: Box<FfxFsr3UpscalerContext>,
    shared: Vec<SharedImage>,
    shared_ready: bool,
    initialized: bool,
}

// Host-side opaque FFX state + a host-only scratch buffer.
unsafe impl Send for Fsr3 {}

impl Drop for Fsr3 {
    fn drop(&mut self) {
        self.destroy();
    }
}

impl Fsr3 {
    /// Creates the backend interface + upscaler context for a maximum render
    /// size and a display size. `max_render` is the largest reduced-resolution
    /// size the context must support (typically the output size), `display` is
    /// the presentation resolution.
    pub fn new(
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        instance: &ash::Instance,
        max_render: [u32; 2],
        display: [u32; 2],
        flags: u32,
    ) -> Result<Self> {
        // The FFX VK backend needs the device, physical device, and the
        // device-level `vkGetDeviceProcAddr` loader entry point.
        let mut vk_context = Box::new(VkDeviceContext {
            vk_device: device.handle().as_raw() as usize as *mut c_void,
            vk_physical_device: physical_device.as_raw() as usize as *mut c_void,
            vk_device_proc_addr: crate::ffx_trace::install(instance.fp_v1_0().get_device_proc_addr),
        });
        let ffx_device = unsafe { ffxGetDeviceVK(vk_context.as_mut()) };

        // Host scratch the backend uses for its own bookkeeping, 256-aligned.
        let scratch_size =
            unsafe { ffxGetScratchMemorySizeVK(physical_device.as_raw() as usize as *mut c_void, MAX_CONTEXTS) };
        let scratch_layout = Layout::from_size_align(scratch_size.max(1), SCRATCH_ALIGN)?;
        let scratch = unsafe { alloc(scratch_layout) };
        if scratch.is_null() {
            bail!("FSR3: out of memory allocating {scratch_size}-byte scratch buffer");
        }
        unsafe { ptr::write_bytes(scratch, 0, scratch_size) };

        let mut interface = unsafe { std::mem::MaybeUninit::<FfxInterface>::zeroed().assume_init() };
        let res = unsafe {
            ffxGetInterfaceVK(
                &mut interface,
                ffx_device,
                scratch as *mut c_void,
                scratch_size,
                MAX_CONTEXTS,
            )
        };
        if res != 0 {
            unsafe { dealloc(scratch, scratch_layout) };
            bail!("FSR3: ffxGetInterfaceVK failed with code {res} (0x{:08x}), scratch {scratch_size} bytes; failed Vulkan calls: {}", res as u32, crate::ffx_trace::take());
        }

        let desc = FfxFsr3UpscalerContextDescription {
            flags,
            maxRenderSize: dims2(max_render[0], max_render[1]),
            maxUpscaleSize: dims2(display[0], display[1]),
            fpMessage: Some(ffx_message),
            backendInterface: interface,
        };

        let mut context = Box::new(std::mem::MaybeUninit::<FfxFsr3UpscalerContext>::zeroed());
        let res = unsafe { ffxFsr3UpscalerContextCreate(context.as_mut_ptr(), &desc) };
        if res != 0 {
            unsafe { dealloc(scratch, scratch_layout) };
            let calls = crate::ffx_trace::take();
            let cause = if calls.contains("vkCreateShaderModule") && calls.contains("code size 0 bytes") {
                " The FidelityFX libraries this build links (render/third_party/ffx/lib) contain no shader blobs, so FSR3 cannot start until they are rebuilt with their shaders."
            } else {
                ""
            };
            bail!("FSR3: ffxFsr3UpscalerContextCreate failed with code {res} (0x{:08x}); backend said: {}; failed Vulkan calls: {calls}.{cause}", res as u32, take_ffx_messages());
        }
        let context = unsafe { context.assume_init() };
        let mut upscaler = Self {
            interface,
            device: ffx_device,
            vk_device: device.clone(),
            vk_context,
            scratch,
            scratch_layout,
            context,
            shared: Vec::new(),
            shared_ready: false,
            initialized: true,
        };
        let mut wanted = unsafe { std::mem::MaybeUninit::<FfxFsr3UpscalerSharedResourceDescriptions>::zeroed().assume_init() };
        let res = unsafe { ffxFsr3UpscalerGetSharedResourceDescriptions(upscaler.context.as_mut(), &mut wanted) };
        if res != 0 {
            bail!("FSR3: ffxFsr3UpscalerGetSharedResourceDescriptions failed with code {res}");
        }
        for description in [wanted.dilatedDepth.resourceDescription, wanted.dilatedMotionVectors.resourceDescription, wanted.reconstructedPrevNearestDepth.resourceDescription] {
            let image = create_shared_image(device, instance, physical_device, description)?;
            upscaler.shared.push(image);
        }
        Ok(upscaler)
    }

    /// Runs the temporal upscale pass, reading the reduced-resolution inputs
    /// and writing the full-resolution `output`.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &mut self,
        cb: vk::CommandBuffer,
        input: &UpscaleInput,
        output: vk::ImageView,
        formats: &Fsr3Formats,
        sharpness: f32,
        frame_time_delta: f32,
        camera_near: f32,
        camera_far: f32,
        camera_fov_y: f32,
        view_space_to_meters: f32,
        reset: bool,
    ) -> Result<()> {
        let [rw, rh] = input.render_size;
        let [ow, oh] = input.output_size;

        // Required inputs.
        let color_format = ffx_format(formats.color).ok_or_else(|| anyhow::anyhow!("FSR3: unsupported color format {:?}", formats.color))?;
        let depth_format = ffx_format(formats.depth).ok_or_else(|| anyhow::anyhow!("FSR3: unsupported depth format {:?}", formats.depth))?;
        let motion_format = ffx_format(formats.motion).ok_or_else(|| anyhow::anyhow!("FSR3: unsupported motion format {:?}", formats.motion))?;
        let reactive_format = ffx_format(formats.reactive).ok_or_else(|| anyhow::anyhow!("FSR3: unsupported reactive format {:?}", formats.reactive))?;
        let output_format = ffx_format(formats.output).ok_or_else(|| anyhow::anyhow!("FSR3: unsupported output format {:?}", formats.output))?;

        let null = resource(
            vk::Image::null(),
            FfxSurfaceFormat_FFX_SURFACE_FORMAT_UNKNOWN,
            0,
            0,
            FfxResourceUsage_FFX_RESOURCE_USAGE_READ_ONLY,
            FfxResourceStates_FFX_RESOURCE_STATE_COMPUTE_READ,
        );

        if !self.shared_ready {
            let range = vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(vk::REMAINING_MIP_LEVELS).layer_count(1);
            let barriers: Vec<vk::ImageMemoryBarrier2<'_>> = self
                .shared
                .iter()
                .map(|shared| {
                    vk::ImageMemoryBarrier2::default()
                        .image(shared.image)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .src_stage_mask(vk::PipelineStageFlags2::NONE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE)
                        .subresource_range(range)
                })
                .collect();
            unsafe { self.vk_device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&barriers)) };
            self.shared_ready = true;
        }
        let shared = |k: usize| unsafe {
            ffxGetResourceVK(
                self.shared[k].image.as_raw() as usize as *mut c_void,
                self.shared[k].description,
                ptr::null(),
                FfxResourceStates_FFX_RESOURCE_STATE_UNORDERED_ACCESS,
            )
        };
        let (dilated_depth, dilated_motion, previous_depth) = (shared(0), shared(1), shared(2));

        let desc = FfxFsr3UpscalerDispatchDescription {
            commandList: unsafe { ffxGetCommandListVK(cb.as_raw() as usize as *mut c_void) },
            color: resource(
                input.color_image,
                color_format,
                rw,
                rh,
                FfxResourceUsage_FFX_RESOURCE_USAGE_RENDERTARGET,
                FfxResourceStates_FFX_RESOURCE_STATE_COMPUTE_READ,
            ),
            depth: resource(
                input.depth_image,
                depth_format,
                rw,
                rh,
                if input.depth_has_stencil {
                    FfxResourceUsage_FFX_RESOURCE_USAGE_DEPTHTARGET | FfxResourceUsage_FFX_RESOURCE_USAGE_STENCILTARGET
                } else {
                    FfxResourceUsage_FFX_RESOURCE_USAGE_DEPTHTARGET
                },
                FfxResourceStates_FFX_RESOURCE_STATE_COMPUTE_READ,
            ),
            motionVectors: resource(
                input.motion_image,
                motion_format,
                rw,
                rh,
                FfxResourceUsage_FFX_RESOURCE_USAGE_RENDERTARGET,
                FfxResourceStates_FFX_RESOURCE_STATE_COMPUTE_READ,
            ),
            // Exposure is passed as the scalar `preExposure` below.
            exposure: null,
            reactive: resource(
                input.reactive_image,
                reactive_format,
                rw,
                rh,
                FfxResourceUsage_FFX_RESOURCE_USAGE_RENDERTARGET,
                FfxResourceStates_FFX_RESOURCE_STATE_COMPUTE_READ,
            ),
            transparencyAndComposition: null,
            dilatedDepth: dilated_depth,
            dilatedMotionVectors: dilated_motion,
            reconstructedPrevNearestDepth: previous_depth,
            output: resource(
                input.output_image,
                output_format,
                ow,
                oh,
                FfxResourceUsage_FFX_RESOURCE_USAGE_UAV,
                FfxResourceStates_FFX_RESOURCE_STATE_UNORDERED_ACCESS,
            ),
            jitterOffset: FfxFloatCoords2D {
                x: input.jitter[0],
                y: input.jitter[1],
            },
            motionVectorScale: FfxFloatCoords2D { x: -0.5 * rw as f32, y: 0.5 * rh as f32 },
            renderSize: dims2(rw, rh),
            upscaleSize: dims2(ow, oh),
            enableSharpening: sharpness > 0.0,
            sharpness,
            frameTimeDelta: frame_time_delta,
            preExposure: input.exposure,
            reset,
            cameraNear: camera_near,
            cameraFar: camera_far,
            cameraFovAngleVertical: camera_fov_y,
            viewSpaceToMetersFactor: view_space_to_meters,
            flags: 0,
        };
        let _ = output;

        let res = unsafe { ffxFsr3UpscalerContextDispatch(self.context.as_mut(), &desc) };
        if res != 0 {
            bail!("FSR3: ffxFsr3UpscalerContextDispatch failed with code {res}");
        }
        Ok(())
    }

    pub fn destroy(&mut self) {
        if self.initialized {
            unsafe { ffxFsr3UpscalerContextDestroy(self.context.as_mut()) };
            self.initialized = false;
        }
        for shared in self.shared.drain(..) {
            unsafe {
                self.vk_device.destroy_image(shared.image, None);
                self.vk_device.free_memory(shared.memory, None);
            }
        }
        if !self.scratch.is_null() {
            unsafe { dealloc(self.scratch, self.scratch_layout) };
            self.scratch = ptr::null_mut();
        }
    }

    #[allow(dead_code)]
    pub fn device(&self) -> FfxDevice {
        self.device
    }
}

pub const FRAME_GEN_BUILT: bool = true;

pub fn frame_gen_flags(_hdr: bool, depth_inverted: bool, depth_infinite: bool) -> u32 {
    let mut flags = 0;
    if depth_inverted {
        flags |= FfxFrameInterpolationInitializationFlagBits_FFX_FRAMEINTERPOLATION_ENABLE_DEPTH_INVERTED;
    }
    if depth_infinite {
        flags |= FfxFrameInterpolationInitializationFlagBits_FFX_FRAMEINTERPOLATION_ENABLE_DEPTH_INFINITE;
    }
    flags as u32
}

pub fn frame_gen_target_format(swapchain: vk::Format) -> Option<vk::Format> {
    use vk::Format as F;
    Some(match swapchain {
        F::B8G8R8A8_SRGB | F::B8G8R8A8_UNORM => F::B8G8R8A8_UNORM,
        F::R8G8B8A8_SRGB | F::R8G8B8A8_UNORM => F::R8G8B8A8_UNORM,
        F::A2B10G10R10_UNORM_PACK32 => F::A2B10G10R10_UNORM_PACK32,
        _ => return None,
    })
}

fn back_buffer_format(target: vk::Format) -> Option<FfxSurfaceFormat> {
    use vk::Format as F;
    Some(match target {
        F::B8G8R8A8_UNORM => FfxSurfaceFormat_FFX_SURFACE_FORMAT_B8G8R8A8_UNORM,
        F::R8G8B8A8_UNORM => FfxSurfaceFormat_FFX_SURFACE_FORMAT_R8G8B8A8_UNORM,
        F::A2B10G10R10_UNORM_PACK32 => FfxSurfaceFormat_FFX_SURFACE_FORMAT_R10G10B10A2_UNORM,
        _ => return None,
    })
}

struct Backend {
    interface: FfxInterface,
    vk_context: Box<VkDeviceContext>,
    scratch: *mut u8,
    layout: Layout,
}

impl Backend {
    fn new(device: &ash::Device, physical_device: vk::PhysicalDevice, instance: &ash::Instance, contexts: usize) -> Result<Self> {
        let mut vk_context = Box::new(VkDeviceContext {
            vk_device: device.handle().as_raw() as usize as *mut c_void,
            vk_physical_device: physical_device.as_raw() as usize as *mut c_void,
            vk_device_proc_addr: crate::ffx_trace::install(instance.fp_v1_0().get_device_proc_addr),
        });
        let ffx_device = unsafe { ffxGetDeviceVK(vk_context.as_mut()) };
        let size = unsafe { ffxGetScratchMemorySizeVK(physical_device.as_raw() as usize as *mut c_void, contexts) };
        let layout = Layout::from_size_align(size.max(1), SCRATCH_ALIGN)?;
        let scratch = unsafe { alloc(layout) };
        if scratch.is_null() {
            bail!("FSR3 frame generation: out of memory allocating a {size}-byte scratch buffer");
        }
        unsafe { ptr::write_bytes(scratch, 0, size) };
        let mut interface = unsafe { std::mem::MaybeUninit::<FfxInterface>::zeroed().assume_init() };
        let res = unsafe { ffxGetInterfaceVK(&mut interface, ffx_device, scratch as *mut c_void, size, contexts) };
        if res != 0 {
            unsafe { dealloc(scratch, layout) };
            bail!("FSR3 frame generation: ffxGetInterfaceVK failed with code {res} (0x{:08x}); failed Vulkan calls: {}", res as u32, crate::ffx_trace::take());
        }
        Ok(Self { interface, vk_context, scratch, layout })
    }

    fn free(&mut self) {
        if !self.scratch.is_null() {
            unsafe { dealloc(self.scratch, self.layout) };
            self.scratch = ptr::null_mut();
        }
    }
}

#[derive(Clone, Copy)]
struct FrameParams {
    render: [u32; 2],
    frame_time_delta: f32,
    near: f32,
    far: f32,
    fov_y: f32,
    view_space_to_meters: f32,
}

pub struct Fsr3FrameGen {
    upscaler: Fsr3,
    backend: Backend,
    flow: Box<FfxOpticalflowContext>,
    flow_live: bool,
    flow_images: Vec<SharedImage>,
    flow_ready: bool,
    prepared: Vec<SharedImage>,
    prepared_ready: bool,
    interpolation: Box<FfxFrameInterpolationContext>,
    interpolation_live: bool,
    target_format: FfxSurfaceFormat,
    display: [u32; 2],
    frame_id: u64,
    params: Option<FrameParams>,
}

unsafe impl Send for Fsr3FrameGen {}

impl Drop for Fsr3FrameGen {
    fn drop(&mut self) {
        self.destroy();
    }
}

impl Fsr3FrameGen {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        instance: &ash::Instance,
        max_render: [u32; 2],
        upscale_output: [u32; 2],
        display: [u32; 2],
        target_format: vk::Format,
        upscaler_flags: u32,
        flags: u32,
    ) -> Result<Self> {
        let Some(ffx_target) = back_buffer_format(target_format) else {
            bail!("FSR3 frame generation: {target_format:?} is not a backbuffer format it can read");
        };
        let upscaler = Fsr3::new(device, physical_device, instance, max_render, upscale_output, upscaler_flags)?;
        let mut backend = Backend::new(device, physical_device, instance, 2)?;

        let mut flow_description = FfxOpticalflowContextDescription { backendInterface: backend.interface, flags: 0, resolution: dims2(display[0], display[1]) };
        let mut flow = Box::new(std::mem::MaybeUninit::<FfxOpticalflowContext>::zeroed());
        let res = unsafe { ffxOpticalflowContextCreate(flow.as_mut_ptr(), &mut flow_description) };
        if res != 0 {
            backend.free();
            bail!("FSR3 frame generation: ffxOpticalflowContextCreate failed with code {res} (0x{:08x}); backend said: {}; failed Vulkan calls: {}", res as u32, take_ffx_messages(), crate::ffx_trace::take());
        }
        let flow = unsafe { flow.assume_init() };
        let interpolation = Box::new(unsafe { std::mem::MaybeUninit::<FfxFrameInterpolationContext>::zeroed().assume_init() });
        let mut generator = Self {
            upscaler,
            backend,
            flow,
            flow_live: true,
            flow_images: Vec::new(),
            flow_ready: false,
            prepared: Vec::new(),
            prepared_ready: false,
            interpolation,
            interpolation_live: false,
            target_format: ffx_target,
            display,
            frame_id: 0,
            params: None,
        };

        let mut interpolation_description = FfxFrameInterpolationContextDescription {
            flags,
            maxRenderSize: dims2(max_render[0], max_render[1]),
            displaySize: dims2(display[0], display[1]),
            backBufferFormat: ffx_target,
            previousInterpolationSourceFormat: ffx_target,
            backendInterface: generator.backend.interface,
        };
        let res = unsafe { ffxFrameInterpolationContextCreate(generator.interpolation.as_mut(), &mut interpolation_description) };
        if res != 0 {
            bail!("FSR3 frame generation: ffxFrameInterpolationContextCreate failed with code {res} (0x{:08x}); backend said: {}; failed Vulkan calls: {}", res as u32, take_ffx_messages(), crate::ffx_trace::take());
        }
        generator.interpolation_live = true;

        let mut wanted = unsafe { std::mem::MaybeUninit::<FfxOpticalflowSharedResourceDescriptions>::zeroed().assume_init() };
        let res = unsafe { ffxOpticalflowGetSharedResourceDescriptions(generator.flow.as_mut(), &mut wanted) };
        if res != 0 {
            bail!("FSR3 frame generation: ffxOpticalflowGetSharedResourceDescriptions failed with code {res}");
        }
        for description in [wanted.opticalFlowVector.resourceDescription, wanted.opticalFlowSCD.resourceDescription] {
            let image = create_shared_image(device, instance, physical_device, description)?;
            generator.flow_images.push(image);
        }
        let mut own = unsafe { std::mem::MaybeUninit::<FfxFrameInterpolationSharedResourceDescriptions>::zeroed().assume_init() };
        let res = unsafe { ffxFrameInterpolationGetSharedResourceDescriptions(generator.interpolation.as_mut(), &mut own) };
        if res != 0 {
            bail!("FSR3 frame generation: ffxFrameInterpolationGetSharedResourceDescriptions failed with code {res}");
        }
        for description in [own.dilatedDepth.resourceDescription, own.dilatedMotionVectors.resourceDescription, own.reconstructedPrevNearestDepth.resourceDescription] {
            let image = create_shared_image(device, instance, physical_device, description)?;
            generator.prepared.push(image);
        }
        Ok(generator)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_upscale_with(
        &mut self,
        cb: vk::CommandBuffer,
        input: &UpscaleInput,
        output: vk::ImageView,
        formats: &Fsr3Formats,
        sharpness: f32,
        frame_time_delta: f32,
        camera_near: f32,
        camera_far: f32,
        camera_fov_y: f32,
        view_space_to_meters: f32,
        reset: bool,
    ) -> Result<()> {
        self.params = Some(FrameParams {
            render: input.render_size,
            frame_time_delta,
            near: camera_near,
            far: camera_far,
            fov_y: camera_fov_y,
            view_space_to_meters,
        });
        self.upscaler.dispatch(cb, input, output, formats, sharpness, frame_time_delta, camera_near, camera_far, camera_fov_y, view_space_to_meters, reset)?;
        if !self.prepared_ready {
            let range = vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(vk::REMAINING_MIP_LEVELS).layer_count(1);
            let barriers: Vec<vk::ImageMemoryBarrier2<'_>> = self
                .prepared
                .iter()
                .map(|shared| {
                    vk::ImageMemoryBarrier2::default()
                        .image(shared.image)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .src_stage_mask(vk::PipelineStageFlags2::NONE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE)
                        .subresource_range(range)
                })
                .collect();
            unsafe { self.upscaler.vk_device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&barriers)) };
            self.prepared_ready = true;
        }
        let [rw, rh] = input.render_size;
        let depth_format = ffx_format(formats.depth).ok_or_else(|| anyhow::anyhow!("FSR3: unsupported depth format {:?}", formats.depth))?;
        let motion_format = ffx_format(formats.motion).ok_or_else(|| anyhow::anyhow!("FSR3: unsupported motion format {:?}", formats.motion))?;
        let external = |image: &SharedImage| unsafe {
            ffxGetResourceVK(image.image.as_raw() as usize as *mut c_void, image.description, ptr::null(), FfxResourceStates_FFX_RESOURCE_STATE_UNORDERED_ACCESS)
        };
        self.frame_id += 1;
        let prepare = FfxFrameInterpolationPrepareDescription {
            flags: 0,
            commandList: unsafe { ffxGetCommandListVK(cb.as_raw() as usize as *mut c_void) },
            renderSize: dims2(rw, rh),
            jitterOffset: FfxFloatCoords2D { x: input.jitter[0], y: input.jitter[1] },
            motionVectorScale: FfxFloatCoords2D { x: -0.5 * rw as f32, y: 0.5 * rh as f32 },
            frameTimeDelta: frame_time_delta,
            cameraNear: camera_near,
            cameraFar: camera_far,
            viewSpaceToMetersFactor: view_space_to_meters,
            cameraFovAngleVertical: camera_fov_y,
            depth: resource(
                input.depth_image,
                depth_format,
                rw,
                rh,
                if input.depth_has_stencil {
                    FfxResourceUsage_FFX_RESOURCE_USAGE_DEPTHTARGET | FfxResourceUsage_FFX_RESOURCE_USAGE_STENCILTARGET
                } else {
                    FfxResourceUsage_FFX_RESOURCE_USAGE_DEPTHTARGET
                },
                FfxResourceStates_FFX_RESOURCE_STATE_COMPUTE_READ,
            ),
            motionVectors: resource(input.motion_image, motion_format, rw, rh, FfxResourceUsage_FFX_RESOURCE_USAGE_READ_ONLY, FfxResourceStates_FFX_RESOURCE_STATE_COMPUTE_READ),
            frameID: self.frame_id,
            dilatedDepth: external(&self.prepared[0]),
            dilatedMotionVectors: external(&self.prepared[1]),
            reconstructedPrevDepth: external(&self.prepared[2]),
            cameraPosition: [0.0; 3],
            cameraUp: [0.0, 1.0, 0.0],
            cameraRight: [1.0, 0.0, 0.0],
            cameraForward: [0.0, 0.0, -1.0],
        };
        let res = unsafe { ffxFrameInterpolationPrepare(self.interpolation.as_mut(), &prepare) };
        if res != 0 {
            bail!("FSR3 frame generation: ffxFrameInterpolationPrepare failed with code {res} (0x{:08x})", res as u32);
        }
        Ok(())
    }

    pub fn frame_gen_dispatch(&mut self, cb: vk::CommandBuffer, present: vk::Image, output: vk::Image, reset: bool) -> Result<()> {
        let Some(params) = self.params else {
            bail!("FSR3 frame generation: no upscale was dispatched this frame");
        };
        let device = self.upscaler.vk_device.clone();
        if !self.flow_ready {
            let range = vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(vk::REMAINING_MIP_LEVELS).layer_count(1);
            let to_clear: Vec<vk::ImageMemoryBarrier2<'_>> = self
                .flow_images
                .iter()
                .map(|shared| {
                    vk::ImageMemoryBarrier2::default()
                        .image(shared.image)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .src_stage_mask(vk::PipelineStageFlags2::NONE)
                        .dst_stage_mask(vk::PipelineStageFlags2::CLEAR)
                        .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .subresource_range(range)
                })
                .collect();
            unsafe { device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&to_clear)) };
            for shared in &self.flow_images {
                let zero = if shared.description.format == FfxSurfaceFormat_FFX_SURFACE_FORMAT_R32_UINT || shared.description.format == FfxSurfaceFormat_FFX_SURFACE_FORMAT_R16G16_SINT {
                    vk::ClearColorValue { uint32: [0; 4] }
                } else {
                    vk::ClearColorValue { float32: [0.0; 4] }
                };
                unsafe { device.cmd_clear_color_image(cb, shared.image, vk::ImageLayout::GENERAL, &zero, &[range]) };
            }
            let to_compute: Vec<vk::ImageMemoryBarrier2<'_>> = self
                .flow_images
                .iter()
                .map(|shared| {
                    vk::ImageMemoryBarrier2::default()
                        .image(shared.image)
                        .old_layout(vk::ImageLayout::GENERAL)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .src_stage_mask(vk::PipelineStageFlags2::CLEAR)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE)
                        .subresource_range(range)
                })
                .collect();
            unsafe { device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&to_compute)) };
            self.flow_ready = true;
        }
        let external = |image: &SharedImage| unsafe {
            ffxGetResourceVK(image.image.as_raw() as usize as *mut c_void, image.description, ptr::null(), FfxResourceStates_FFX_RESOURCE_STATE_UNORDERED_ACCESS)
        };
        let (flow_vector, flow_scene_change) = (external(&self.flow_images[0]), external(&self.flow_images[1]));
        let (dilated_depth, dilated_motion, previous_depth) = (external(&self.prepared[0]), external(&self.prepared[1]), external(&self.prepared[2]));
        let [width, height] = self.display;
        let back_buffer = resource(
            present,
            self.target_format,
            width,
            height,
            FfxResourceUsage_FFX_RESOURCE_USAGE_RENDERTARGET,
            FfxResourceStates_FFX_RESOURCE_STATE_COMPUTE_READ,
        );
        let interpolated = resource(output, self.target_format, width, height, FfxResourceUsage_FFX_RESOURCE_USAGE_UAV, FfxResourceStates_FFX_RESOURCE_STATE_UNORDERED_ACCESS);
        let none = resource(vk::Image::null(), self.target_format, 0, 0, FfxResourceUsage_FFX_RESOURCE_USAGE_READ_ONLY, FfxResourceStates_FFX_RESOURCE_STATE_COMPUTE_READ);
        let command_list = unsafe { ffxGetCommandListVK(cb.as_raw() as usize as *mut c_void) };
        let transfer = FfxBackbufferTransferFunction_FFX_BACKBUFFER_TRANSFER_FUNCTION_SRGB;

        let flow_dispatch = FfxOpticalflowDispatchDescription {
            commandList: command_list,
            color: back_buffer,
            opticalFlowVector: flow_vector,
            opticalFlowSCD: flow_scene_change,
            reset,
            backbufferTransferFunction: transfer as i32,
            minMaxLuminance: FfxFloatCoords2D { x: 0.0, y: 1.0 },
        };
        let res = unsafe { ffxOpticalflowContextDispatch(self.flow.as_mut(), &flow_dispatch) };
        if res != 0 {
            bail!("FSR3 frame generation: ffxOpticalflowContextDispatch failed with code {res} (0x{:08x})", res as u32);
        }

        let dispatch = FfxFrameInterpolationDispatchDescription {
            flags: 0,
            commandList: command_list,
            displaySize: dims2(width, height),
            renderSize: dims2(params.render[0], params.render[1]),
            currentBackBuffer: back_buffer,
            currentBackBuffer_HUDLess: none,
            output: interpolated,
            interpolationRect: FfxRect2D { left: 0, top: 0, width: width as i32, height: height as i32 },
            opticalFlowVector: flow_vector,
            opticalFlowSceneChangeDetection: flow_scene_change,
            opticalFlowBufferSize: dims2(width.div_ceil(8), height.div_ceil(8)),
            opticalFlowScale: FfxFloatCoords2D { x: 1.0 / width as f32, y: 1.0 / height as f32 },
            opticalFlowBlockSize: 8,
            cameraNear: params.near,
            cameraFar: params.far,
            cameraFovAngleVertical: params.fov_y,
            viewSpaceToMetersFactor: params.view_space_to_meters,
            frameTimeDelta: params.frame_time_delta,
            reset,
            backBufferTransferFunction: transfer,
            minMaxLuminance: [0.0, 1.0],
            frameID: self.frame_id,
            dilatedDepth: dilated_depth,
            dilatedMotionVectors: dilated_motion,
            reconstructedPrevDepth: previous_depth,
            distortionField: none,
        };
        let res = unsafe { ffxFrameInterpolationDispatch(self.interpolation.as_mut(), &dispatch) };
        if res != 0 {
            bail!("FSR3 frame generation: ffxFrameInterpolationDispatch failed with code {res} (0x{:08x})", res as u32);
        }
        Ok(())
    }

    pub fn destroy(&mut self) {
        if self.interpolation_live {
            unsafe { ffxFrameInterpolationContextDestroy(self.interpolation.as_mut()) };
            self.interpolation_live = false;
        }
        if self.flow_live {
            unsafe { ffxOpticalflowContextDestroy(self.flow.as_mut()) };
            self.flow_live = false;
        }
        for shared in self.flow_images.drain(..).chain(self.prepared.drain(..)) {
            unsafe {
                self.upscaler.vk_device.destroy_image(shared.image, None);
                self.upscaler.vk_device.free_memory(shared.memory, None);
            }
        }
        self.backend.free();
        self.upscaler.destroy();
    }
}
