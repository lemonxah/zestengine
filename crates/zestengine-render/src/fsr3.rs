//! AMD FSR 3 (FidelityFX SDK 3.0.4) temporal upscaler, driven through the FFX
//! Vulkan backend bound in [`super::ffx`].
//!
//! This module owns the FFX backend interface + the FSR3 upscaler context and
//! turns the renderer's [`UpscaleInput`] into an `ffxFsr3UpscalerContextDispatch`
//! call each frame. The frame-interpolation pass (frame generation) is wired
//! separately once the upscaler path is proven.
#![allow(dead_code)] // WIP: wired into the renderer's post chain in the next step.

use std::alloc::{Layout, alloc, dealloc};
use std::ffi::c_void;
use std::ptr;

use anyhow::{Result, bail};
use ash::vk;
use ash::vk::Handle;

use super::UpscaleInput;
use super::ffx::*;

/// FFX diagnostics: the backend reports errors and warnings here (message is a
/// 32-bit FFX message code; the names live in `ffx_assert.h`).
unsafe extern "C" fn ffx_message(type_: FfxMsgType, message: *const u32) {
    let code = if message.is_null() { 0 } else { unsafe { *message } };
    // FfxMsgType: 0 = error, 1 = warning (bindgen's constant names trip the
    // non-upper-case lint, so match on the raw values).
    match type_ {
        0 => tracing::error!("FFX: error code 0x{code:08x}"),
        1 => tracing::warn!("FFX: warning code 0x{code:08x}"),
        _ => tracing::debug!("FFX: message code 0x{code:08x}"),
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
    view: vk::ImageView,
    format: FfxSurfaceFormat,
    width: u32,
    height: u32,
    usage: FfxResourceUsage,
    state: FfxResourceStates,
) -> FfxResource {
    if view == vk::ImageView::null() {
        return FfxResource {
            resource: ptr::null_mut(),
            description: resource_description(format, width, height, usage),
            state,
            name: [0; 64],
        };
    }
    unsafe {
        ffxGetResourceVK(
            view.as_raw() as usize as *mut c_void,
            resource_description(format, width, height, usage),
            ptr::null_mut(),
            state,
        )
    }
}

/// Owns the FFX Vulkan backend interface, its host scratch buffer, and the
/// FSR3 upscaler context.
pub struct Fsr3 {
    interface: FfxInterface,
    device: FfxDevice,
    scratch: *mut u8,
    scratch_layout: Layout,
    context: Box<FfxFsr3UpscalerContext>,
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
        let mut vk_ctx = VkDeviceContext {
            vk_device: device.handle().as_raw() as usize as *mut c_void,
            vk_physical_device: physical_device.as_raw() as usize as *mut c_void,
            vk_device_proc_addr: instance.fp_v1_0().get_device_proc_addr as usize as *mut c_void,
        };
        let ffx_device = unsafe { ffxGetDeviceVK(&mut vk_ctx) };

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
            bail!("FSR3: ffxGetInterfaceVK failed with code {res}");
        }

        let desc = FfxFsr3UpscalerContextDescription {
            flags,
            maxRenderSize: dims2(max_render[0], max_render[1]),
            displaySize: dims2(display[0], display[1]),
            backendInterface: interface,
            fpMessage: Some(ffx_message),
        };

        let mut context = Box::new(std::mem::MaybeUninit::<FfxFsr3UpscalerContext>::zeroed());
        let res = unsafe { ffxFsr3UpscalerContextCreate(context.as_mut_ptr(), &desc) };
        if res != 0 {
            unsafe { dealloc(scratch, scratch_layout) };
            bail!("FSR3: ffxFsr3UpscalerContextCreate failed with code {res}");
        }
        let context = unsafe { context.assume_init() };

        Ok(Self {
            interface,
            device: ffx_device,
            scratch,
            scratch_layout,
            context,
            initialized: true,
        })
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
            vk::ImageView::null(),
            FfxSurfaceFormat_FFX_SURFACE_FORMAT_UNKNOWN,
            0,
            0,
            FfxResourceUsage_FFX_RESOURCE_USAGE_READ_ONLY,
            FfxResourceStates_FFX_RESOURCE_STATE_COMPUTE_READ,
        );

        let desc = FfxFsr3UpscalerDispatchDescription {
            commandList: unsafe { ffxGetCommandListVK(cb.as_raw() as usize as *mut c_void) },
            color: resource(
                input.color,
                color_format,
                rw,
                rh,
                FfxResourceUsage_FFX_RESOURCE_USAGE_RENDERTARGET,
                FfxResourceStates_FFX_RESOURCE_STATE_COMPUTE_READ,
            ),
            depth: resource(
                input.depth,
                depth_format,
                rw,
                rh,
                FfxResourceUsage_FFX_RESOURCE_USAGE_DEPTHTARGET,
                FfxResourceStates_FFX_RESOURCE_STATE_COMPUTE_READ,
            ),
            motionVectors: resource(
                input.motion,
                motion_format,
                rw,
                rh,
                FfxResourceUsage_FFX_RESOURCE_USAGE_RENDERTARGET,
                FfxResourceStates_FFX_RESOURCE_STATE_COMPUTE_READ,
            ),
            // Exposure is passed as the scalar `preExposure` below.
            exposure: null,
            reactive: resource(
                input.reactive,
                reactive_format,
                rw,
                rh,
                FfxResourceUsage_FFX_RESOURCE_USAGE_RENDERTARGET,
                FfxResourceStates_FFX_RESOURCE_STATE_COMPUTE_READ,
            ),
            transparencyAndComposition: null,
            dilatedDepth: null,
            dilatedMotionVectors: null,
            reconstructedPrevNearestDepth: null,
            output: resource(
                output,
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
            motionVectorScale: FfxFloatCoords2D { x: 1.0, y: 1.0 },
            renderSize: dims2(rw, rh),
            enableSharpening: sharpness > 0.0,
            sharpness,
            frameTimeDelta: frame_time_delta,
            preExposure: input.exposure,
            reset,
            cameraNear: camera_near,
            cameraFar: camera_far,
            cameraFovAngleVertical: camera_fov_y,
            viewSpaceToMetersFactor: view_space_to_meters,
        };

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

/// Builds the combined FSR3 context flags (temporal upscaling + frame
/// generation, since `UPSCALING_ONLY` is left clear).
pub fn frame_gen_flags(hdr: bool, depth_inverted: bool, depth_infinite: bool) -> u32 {
    let mut flags = 0;
    if hdr {
        flags |= FfxFsr3InitializationFlagBits_FFX_FSR3_ENABLE_HIGH_DYNAMIC_RANGE;
    }
    if depth_inverted {
        flags |= FfxFsr3InitializationFlagBits_FFX_FSR3_ENABLE_DEPTH_INVERTED;
    }
    if depth_infinite {
        flags |= FfxFsr3InitializationFlagBits_FFX_FSR3_ENABLE_DEPTH_INFINITE;
    }
    flags
}

/// The combined FSR3 context (temporal upscaling + frame interpolation) over
/// the FFX Vulkan backend. The upscale runs through `dispatch_upscale`; the
/// frame-interpolation pass then writes the interpolated (or current) frame to
/// `outputs[0]` for the swapchain to present.
pub struct Fsr3FrameGen {
    context: Box<FfxFsr3Context>,
    scratch: *mut u8,
    scratch_layout: Layout,
    initialized: bool,
}

// Host-side opaque FFX state + a host-only scratch buffer.
unsafe impl Send for Fsr3FrameGen {}

impl Drop for Fsr3FrameGen {
    fn drop(&mut self) {
        self.destroy();
    }
}

impl Fsr3FrameGen {
    /// Creates the combined context with three backend interfaces (shared,
    /// upscaling, frame interpolation) over one scratch buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        instance: &ash::Instance,
        max_render: [u32; 2],
        upscale_output: [u32; 2],
        display: [u32; 2],
        back_buffer_format: vk::Format,
        flags: u32,
    ) -> Result<Self> {
        let mut vk_ctx = VkDeviceContext {
            vk_device: device.handle().as_raw() as usize as *mut c_void,
            vk_physical_device: physical_device.as_raw() as usize as *mut c_void,
            vk_device_proc_addr: instance.fp_v1_0().get_device_proc_addr as usize as *mut c_void,
        };
        let ffx_device = unsafe { ffxGetDeviceVK(&mut vk_ctx) };

        // Three contexts (shared + upscaling + frame interpolation) share one
        // 256-aligned host scratch buffer.
        let scratch_size =
            unsafe { ffxGetScratchMemorySizeVK(physical_device.as_raw() as usize as *mut c_void, 3) };
        let scratch_layout = Layout::from_size_align(scratch_size.max(1), SCRATCH_ALIGN)?;
        let scratch = unsafe { alloc(scratch_layout) };
        if scratch.is_null() {
            bail!("FSR3: out of memory allocating {scratch_size}-byte scratch buffer");
        }
        unsafe { ptr::write_bytes(scratch, 0, scratch_size) };

        let mut interfaces = [unsafe { std::mem::MaybeUninit::<FfxInterface>::zeroed().assume_init() }; 3];
        for interface in &mut interfaces {
            let res = unsafe {
                ffxGetInterfaceVK(interface, ffx_device, scratch as *mut c_void, scratch_size, 3)
            };
            if res != 0 {
                unsafe { dealloc(scratch, scratch_layout) };
                bail!("FSR3: ffxGetInterfaceVK failed with code {res}");
            }
        }

        let bb = ffx_format(back_buffer_format)
            .unwrap_or(FfxSurfaceFormat_FFX_SURFACE_FORMAT_R8G8B8A8_UNORM);
        let mut desc = FfxFsr3ContextDescription {
            flags,
            maxRenderSize: dims2(max_render[0], max_render[1]),
            upscaleOutputSize: dims2(upscale_output[0], upscale_output[1]),
            displaySize: dims2(display[0], display[1]),
            backendInterfaceSharedResources: interfaces[0],
            backendInterfaceUpscaling: interfaces[1],
            backendInterfaceFrameInterpolation: interfaces[2],
            fpMessage: Some(ffx_message),
            backBufferFormat: bb,
        };

        let mut context = Box::new(std::mem::MaybeUninit::<FfxFsr3Context>::zeroed());
        let res = unsafe { ffxFsr3ContextCreate(context.as_mut_ptr(), &mut desc) };
        if res != 0 {
            unsafe { dealloc(scratch, scratch_layout) };
            bail!("FSR3: ffxFsr3ContextCreate failed with code {res}");
        }
        let context = unsafe { context.assume_init() };

        Ok(Self {
            context,
            scratch,
            scratch_layout,
            initialized: true,
        })
    }

    /// Runs the temporal upscale pass (the combined-context variant).
    pub fn dispatch_upscale(&mut self, upscale: &FfxFsr3DispatchUpscaleDescription) -> Result<()> {
        let res = unsafe { ffxFsr3ContextDispatchUpscale(self.context.as_mut(), upscale) };
        if res != 0 {
            bail!("FSR3: ffxFsr3ContextDispatchUpscale failed with code {res}");
        }
        Ok(())
    }

    /// Builds the combined-context upscale description from the renderer's
    /// [`UpscaleInput`] + an output view, then dispatches it (the frame-gen
    /// context's upscale; identical inputs to [`Fsr3::dispatch`]).
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
        let [rw, rh] = input.render_size;
        let [ow, oh] = input.output_size;
        let color_format = ffx_format(formats.color).ok_or_else(|| anyhow::anyhow!("FSR3: unsupported color format {:?}", formats.color))?;
        let depth_format = ffx_format(formats.depth).ok_or_else(|| anyhow::anyhow!("FSR3: unsupported depth format {:?}", formats.depth))?;
        let motion_format = ffx_format(formats.motion).ok_or_else(|| anyhow::anyhow!("FSR3: unsupported motion format {:?}", formats.motion))?;
        let reactive_format = ffx_format(formats.reactive).ok_or_else(|| anyhow::anyhow!("FSR3: unsupported reactive format {:?}", formats.reactive))?;
        let output_format = ffx_format(formats.output).ok_or_else(|| anyhow::anyhow!("FSR3: unsupported output format {:?}", formats.output))?;

        let null = resource(
            vk::ImageView::null(),
            FfxSurfaceFormat_FFX_SURFACE_FORMAT_UNKNOWN,
            0,
            0,
            FfxResourceUsage_FFX_RESOURCE_USAGE_READ_ONLY,
            FfxResourceStates_FFX_RESOURCE_STATE_COMPUTE_READ,
        );

        let desc = FfxFsr3DispatchUpscaleDescription {
            commandList: unsafe { ffxGetCommandListVK(cb.as_raw() as usize as *mut c_void) },
            color: resource(
                input.color,
                color_format,
                rw,
                rh,
                FfxResourceUsage_FFX_RESOURCE_USAGE_RENDERTARGET,
                FfxResourceStates_FFX_RESOURCE_STATE_COMPUTE_READ,
            ),
            depth: resource(
                input.depth,
                depth_format,
                rw,
                rh,
                FfxResourceUsage_FFX_RESOURCE_USAGE_DEPTHTARGET,
                FfxResourceStates_FFX_RESOURCE_STATE_COMPUTE_READ,
            ),
            motionVectors: resource(
                input.motion,
                motion_format,
                rw,
                rh,
                FfxResourceUsage_FFX_RESOURCE_USAGE_RENDERTARGET,
                FfxResourceStates_FFX_RESOURCE_STATE_COMPUTE_READ,
            ),
            exposure: null,
            reactive: resource(
                input.reactive,
                reactive_format,
                rw,
                rh,
                FfxResourceUsage_FFX_RESOURCE_USAGE_RENDERTARGET,
                FfxResourceStates_FFX_RESOURCE_STATE_COMPUTE_READ,
            ),
            transparencyAndComposition: null,
            upscaleOutput: resource(
                output,
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
            motionVectorScale: FfxFloatCoords2D { x: 1.0, y: 1.0 },
            renderSize: dims2(rw, rh),
            enableSharpening: sharpness > 0.0,
            sharpness,
            frameTimeDelta: frame_time_delta,
            preExposure: input.exposure,
            reset,
            cameraNear: camera_near,
            cameraFar: camera_far,
            cameraFovAngleVertical: camera_fov_y,
            viewSpaceToMetersFactor: view_space_to_meters,
        };
        self.dispatch_upscale(&desc)
    }

    /// Schedules the frame-interpolation pass: `desc.presentColor` is the
    /// rendered frame; the interpolated (or current) frame lands in
    /// `desc.outputs[0]`.
    pub fn dispatch_frame_generation(&mut self, desc: &FfxFrameGenerationDispatchDescription) -> Result<()> {
        let res = unsafe { ffxFsr3DispatchFrameGeneration(desc) };
        if res != 0 {
            bail!("FSR3: ffxFsr3DispatchFrameGeneration failed with code {res}");
        }
        Ok(())
    }

    /// Builds the frame-generation dispatch from two display-space views
    /// (`present` = the tonemapped scene, `output` = the interpolated/current
    /// frame the swapchain then presents) and dispatches it.
    pub fn frame_gen_dispatch(
        &mut self,
        cb: vk::CommandBuffer,
        present: vk::ImageView,
        output: vk::ImageView,
        format: vk::Format,
        size: [u32; 2],
        reset: bool,
    ) -> Result<()> {
        let fmt = ffx_format(format).unwrap_or(FfxSurfaceFormat_FFX_SURFACE_FORMAT_R8G8B8A8_UNORM);
        let desc = resource_description(
            fmt,
            size[0],
            size[1],
            FfxResourceUsage_FFX_RESOURCE_USAGE_RENDERTARGET,
        );
        let present_res = unsafe {
            ffxGetResourceVK(
                present.as_raw() as usize as *mut c_void,
                desc,
                ptr::null_mut(),
                FfxResourceStates_FFX_RESOURCE_STATE_UNORDERED_ACCESS,
            )
        };
        let output_res = unsafe {
            ffxGetResourceVK(
                output.as_raw() as usize as *mut c_void,
                desc,
                ptr::null_mut(),
                FfxResourceStates_FFX_RESOURCE_STATE_UNORDERED_ACCESS,
            )
        };
        let d = FfxFrameGenerationDispatchDescription {
            commandList: unsafe { ffxGetCommandListVK(cb.as_raw() as usize as *mut c_void) },
            presentColor: present_res,
            outputs: [output_res, FfxResource { resource: ptr::null_mut(), description: desc, state: 0, name: [0; 64] }, FfxResource { resource: ptr::null_mut(), description: desc, state: 0, name: [0; 64] }, FfxResource { resource: ptr::null_mut(), description: desc, state: 0, name: [0; 64] }],
            numInterpolatedFrames: 0,
            reset,
            backBufferTransferFunction: FfxBackbufferTransferFunction_FFX_BACKBUFFER_TRANSFER_FUNCTION_SRGB,
            minMaxLuminance: [0.0; 2],
        };
        self.dispatch_frame_generation(&d)
    }

    pub fn destroy(&mut self) {
        if self.initialized {
            unsafe { ffxFsr3ContextDestroy(self.context.as_mut()) };
            self.initialized = false;
        }
        if !self.scratch.is_null() {
            unsafe { dealloc(self.scratch, self.scratch_layout) };
            self.scratch = ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_the_formats_fsr_consumes() {
        use vk::Format as F;
        assert_eq!(
            ffx_format(F::R16G16B16A16_SFLOAT),
            Some(FfxSurfaceFormat_FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT)
        );
        assert_eq!(
            ffx_format(F::R16G16_SFLOAT),
            Some(FfxSurfaceFormat_FFX_SURFACE_FORMAT_R16G16_FLOAT)
        );
        assert_eq!(
            ffx_format(F::R32_SFLOAT),
            Some(FfxSurfaceFormat_FFX_SURFACE_FORMAT_R32_FLOAT)
        );
        assert_eq!(
            ffx_format(F::R8_UNORM),
            Some(FfxSurfaceFormat_FFX_SURFACE_FORMAT_R8_UNORM)
        );
        // Depth-stencil formats are not a FSR sampleable surface.
        assert_eq!(ffx_format(F::D32_SFLOAT), None);
        assert_eq!(ffx_format(F::D32_SFLOAT_S8_UINT), None);
    }

    #[test]
    fn ffx_backend_entry_points_link() {
        // Taking the address of each entry point forces the linker to pull the
        // symbol out of the vendored static libs, so a broken/missing backend
        // fails the link rather than being silently dead-stripped.
        let entry_points = [
            ffxGetDeviceVK as *const () as usize,
            ffxGetInterfaceVK as *const () as usize,
            ffxGetScratchMemorySizeVK as *const () as usize,
            ffxGetCommandListVK as *const () as usize,
            ffxGetResourceVK as *const () as usize,
            ffxFsr3UpscalerContextCreate as *const () as usize,
            ffxFsr3UpscalerContextDispatch as *const () as usize,
            ffxFsr3UpscalerContextDestroy as *const () as usize,
            ffxFsr3ContextCreate as *const () as usize,
            ffxFsr3ContextDispatchUpscale as *const () as usize,
            ffxFsr3DispatchFrameGeneration as *const () as usize,
            ffxFsr3ContextDestroy as *const () as usize,
        ];
        assert!(entry_points.iter().all(|&p| p != 0));
    }

    #[test]
    fn upscaler_flags_set_hdr_and_depth_conventions() {
        assert_eq!(upscaler_flags(false, false, false), 0);
        let hdr = FfxFsr3UpscalerInitializationFlagBits_FFX_FSR3UPSCALER_ENABLE_HIGH_DYNAMIC_RANGE;
        let inv = FfxFsr3UpscalerInitializationFlagBits_FFX_FSR3UPSCALER_ENABLE_DEPTH_INVERTED;
        assert_eq!(upscaler_flags(true, true, false), hdr | inv);
        let inf = FfxFsr3UpscalerInitializationFlagBits_FFX_FSR3UPSCALER_ENABLE_DEPTH_INFINITE;
        assert_eq!(upscaler_flags(false, true, true), inv | inf);
    }

    #[test]
    fn frame_gen_flags_leave_upscaling_only_clear() {
        assert_eq!(frame_gen_flags(false, false, false), 0);
        let hdr = FfxFsr3InitializationFlagBits_FFX_FSR3_ENABLE_HIGH_DYNAMIC_RANGE;
        let inv = FfxFsr3InitializationFlagBits_FFX_FSR3_ENABLE_DEPTH_INVERTED;
        assert_eq!(frame_gen_flags(true, true, false), hdr | inv);
        // Frame generation is enabled by NOT setting UPSCALING_ONLY.
        assert_eq!(
            frame_gen_flags(true, false, false) & FfxFsr3InitializationFlagBits_FFX_FSR3_ENABLE_UPSCALING_ONLY,
            0
        );
    }

    #[test]
    fn resource_description_records_size_and_format() {
        let desc = resource_description(
            FfxSurfaceFormat_FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT,
            1280,
            720,
            FfxResourceUsage_FFX_RESOURCE_USAGE_RENDERTARGET,
        );
        assert_eq!(desc.type_, FfxResourceType_FFX_RESOURCE_TYPE_TEXTURE2D);
        assert_eq!(desc.format, FfxSurfaceFormat_FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT);
        assert_eq!(unsafe { desc.__bindgen_anon_1.width }, 1280);
        assert_eq!(unsafe { desc.__bindgen_anon_2.height }, 720);
        assert_eq!(unsafe { desc.__bindgen_anon_3.depth }, 1);
        assert_eq!(desc.mipCount, 1);
        assert_eq!(desc.usage, FfxResourceUsage_FFX_RESOURCE_USAGE_RENDERTARGET);
    }
}
