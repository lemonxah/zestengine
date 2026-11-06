//! Device tests for GPU code that was previously only compiled and CPU-tested
//! (skinning, blendshapes, cloth override, text, shadow bias, timestamps, egui
//! texture upload). Every test skips cleanly (returns early) when
//! `HeadlessGpu::new()` finds no Vulkan 1.3 device or `ZESTENGINE_TEST_GPU=0`.
//! Run them with `cargo test -p zestengine-render --lib gpu_tests -- --test-threads=1`
//! (one device at a time; several tests also print measured numbers with
//! `--nocapture`).

mod aa;
mod ao;
mod ao_engine;
mod default_material_cost;
mod editor_grid;
mod flux;
mod flux_math;
mod flux_attachment;
mod exposure;
mod flux_surface_filter;
mod flux_screen_trace;
mod flux_screen_history;
mod reflection_denoise;
mod reflection_mask;
mod flux_radiance;
mod flux_radiance_blend;
mod flux_ray_lighting;
mod flux_light_falloff;
mod flux_area_light;
mod flux_hit_material;
mod egui_upload;
mod impostor;
mod model_preview;
mod offscreen;
pub(crate) mod planar;
mod renderer;
mod selection_outline;
mod transclipping;
mod shadow;
mod sprite;
mod skin;
mod sky_flicker;
mod stencil;
mod timing;
mod xr_cluster;
mod xr_cluster_hiz;
mod zestlet_view;
mod vertex_tess;
mod water;


use crate::cluster_headless::HeadlessGpu;

/// The shared test device, or None (skip) when there is no usable GPU.
pub(crate) fn gpu() -> Option<HeadlessGpu> {
    match HeadlessGpu::new() {
        Ok(Some(g)) => {
            eprintln!("[gpu test] {} (validation layer: {})", g.device_name(), g.validation_enabled());
            Some(g)
        }
        Ok(None) => {
            eprintln!("[gpu test] no Vulkan 1.3 device: skipped");
            None
        }
        Err(e) => panic!("creating the headless device: {e:#}"),
    }
}

use anyhow::Result;
use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};

use crate::alloc::{self, Buffer};

/// All aspects of a test image's format (depth-stencil formats carry both).
pub(crate) fn aspect_of(format: vk::Format) -> vk::ImageAspectFlags {
    match format {
        vk::Format::D32_SFLOAT | vk::Format::D16_UNORM | vk::Format::X8_D24_UNORM_PACK32 => vk::ImageAspectFlags::DEPTH,
        vk::Format::D32_SFLOAT_S8_UINT | vk::Format::D24_UNORM_S8_UINT | vk::Format::D16_UNORM_S8_UINT => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }
        vk::Format::S8_UINT => vk::ImageAspectFlags::STENCIL,
        _ => vk::ImageAspectFlags::COLOR,
    }
}

/// An offscreen image with its default view (all layers).
pub(crate) struct Img {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub w: u32,
    pub h: u32,
    pub layers: u32,
    allocation: Option<Allocation>,
}

impl Img {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        g: &HeadlessGpu,
        alloc: &mut Allocator,
        w: u32,
        h: u32,
        layers: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        samples: vk::SampleCountFlags,
    ) -> Result<Self> {
        let device = &g.device;
        let aspect = aspect_of(format);
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D { width: w, height: h, depth: 1 })
            .mip_levels(1)
            .array_layers(layers)
            .samples(samples)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.create_image(&info, None)? };
        let req = unsafe { device.get_image_memory_requirements(image) };
        let allocation = alloc.allocate(&AllocationCreateDesc {
            name: "gpu-test image",
            requirements: req,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;
        unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset())? };
        let view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(if layers > 1 { vk::ImageViewType::TYPE_2D_ARRAY } else { vk::ImageViewType::TYPE_2D })
                    .format(format)
                    .subresource_range(
                        vk::ImageSubresourceRange::default().aspect_mask(aspect).level_count(1).layer_count(layers),
                    ),
                None,
            )?
        };
        Ok(Self { image, view, format, w, h, layers, allocation: Some(allocation) })
    }

    pub fn destroy(&mut self, g: &HeadlessGpu, alloc: &mut Allocator) {
        unsafe {
            g.device.destroy_image_view(self.view, None);
            g.device.destroy_image(self.image, None);
        }
        if let Some(a) = self.allocation.take() {
            let _ = alloc.free(a);
        }
    }

    pub fn range(&self) -> vk::ImageSubresourceRange {
        let aspect = aspect_of(self.format);
        vk::ImageSubresourceRange::default().aspect_mask(aspect).level_count(1).layer_count(self.layers)
    }
}

/// A full-image layout transition (all layers).
pub(crate) fn transition(
    g: &HeadlessGpu,
    cb: vk::CommandBuffer,
    img: &Img,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
    src: (vk::PipelineStageFlags2, vk::AccessFlags2),
    dst: (vk::PipelineStageFlags2, vk::AccessFlags2),
) {
    let b = [vk::ImageMemoryBarrier2::default()
        .image(img.image)
        .old_layout(old)
        .new_layout(new)
        .src_stage_mask(src.0)
        .src_access_mask(src.1)
        .dst_stage_mask(dst.0)
        .dst_access_mask(dst.1)
        .subresource_range(img.range())];
    unsafe { g.device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&b)) };
}

pub(crate) const NONE: (vk::PipelineStageFlags2, vk::AccessFlags2) =
    (vk::PipelineStageFlags2::TOP_OF_PIPE, vk::AccessFlags2::empty());
pub(crate) const COLOR_OUT: (vk::PipelineStageFlags2, vk::AccessFlags2) =
    (vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE);

/// Copy `layers` layers of an image that is in `layout` into host memory
/// (tightly packed, layer after layer). The image is left in `layout`.
pub(crate) fn read_image(
    g: &HeadlessGpu,
    alloc: &mut Allocator,
    img: &Img,
    layout: vk::ImageLayout,
    src: (vk::PipelineStageFlags2, vk::AccessFlags2),
    bytes_per_px: usize,
) -> Result<Vec<u8>> {
    let size = (img.w * img.h * img.layers) as usize * bytes_per_px;
    let mut host = Buffer::new(&g.device, alloc, size as u64, vk::BufferUsageFlags::TRANSFER_DST, MemoryLocation::GpuToCpu, "gpu-test readback")?;
    let aspect = img.range().aspect_mask;
    alloc::one_time_submit(&g.device, g.pool, g.queue, |cb| unsafe {
        transition(g, cb, img, layout, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, src, (vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_READ));
        let copy = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(aspect).layer_count(img.layers))
            .image_extent(vk::Extent3D { width: img.w, height: img.h, depth: 1 });
        g.device.cmd_copy_image_to_buffer(cb, img.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, host.handle, &[copy]);
        transition(
            g,
            cb,
            img,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            layout,
            (vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::empty()),
            (vk::PipelineStageFlags2::ALL_COMMANDS, vk::AccessFlags2::empty()),
        );
        let b = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)];
        g.device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&b));
    })?;
    let out = host.mapped().map(|m| m[..size].to_vec()).unwrap_or_default();
    host.destroy(&g.device, alloc);
    Ok(out)
}

/// Write a luminance image as a PGM into `$ZESTENGINE_TEST_DUMP_DIR` (when set)
/// for looking at a failing render; `v` is linear, shown with gamma 2.2.
pub(crate) fn dump_gray(name: &str, v: &[f32], w: u32) {
    let Ok(dir) = std::env::var("ZESTENGINE_TEST_DUMP_DIR") else { return };
    let max = v.iter().cloned().fold(1e-6f32, f32::max);
    let mut out = format!("P5\n{w} {}\n255\n", v.len() as u32 / w).into_bytes();
    out.extend(v.iter().map(|x| ((x / max).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8));
    let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{name}.pgm")), out);
}
