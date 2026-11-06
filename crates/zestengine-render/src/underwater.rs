//! Underwater view pass: when a camera is in a water body, fog the lit HDR
//! scene toward the water's colour with distance (by scene depth), darken it
//! with depth below the surface (the caustics are on the lit surfaces already)
//! (`underwater.frag`, two blended fullscreen draws into the HDR target). The
//! post pass adds the wobble (see [`Wobble`]). The shader evaluates the water
//! body's waves per pixel (from a small uniform block), so the waterline where
//! the near plane straddles the surface, the fog's cut at the surface and the
//! depth darkening all follow the moving surface.

use std::collections::HashMap;
use std::io::Cursor;

use anyhow::Result;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocator;

use crate::alloc::Buffer;
use crate::post::HDR_FORMAT;
use crate::{CameraData, LightData, MAX_WATER_WAVES, UnderwaterFx, image_barrier_layers};

const FULLSCREEN_VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fullscreen.vert.spv"));
const UNDERWATER_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/underwater.frag.spv"));

/// Descriptor-set slots: one per view kind and frame in flight (or XR slot),
/// so a set is only rewritten once the frame that used it has finished.
pub(crate) const SLOT_MAIN: u32 = 0;
pub(crate) const SLOT_VIEWPORT: u32 = 8;
pub(crate) const SLOT_PREVIEW: u32 = 16;
pub(crate) const SLOT_XR: u32 = 24;
/// The right eye of a multiview XR pass (`SLOT_XR` is the left).
pub(crate) const SLOT_XR_RIGHT: u32 = 32;
const MAX_SETS: u32 = 40;

/// Near-plane margin (m) inside which a pixel blends between above and below.
const WATERLINE_BLEND: f32 = 0.0015;

/// Bytes per slot in the wave uniform buffer (covers any offset alignment).
const UBO_STRIDE: u64 = 512;

/// The camera's water body for the shader (std140, `Waves` in underwater.frag).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WaterUbo {
    to_local: [[f32; 4]; 4],
    to_world: [[f32; 4]; 4],
    /// x rest surface (local y), y wave count, z water clock.
    info: [f32; 4],
    wave_a: [[f32; 4]; MAX_WATER_WAVES],
    wave_b: [[f32; 4]; MAX_WATER_WAVES],
    absorption: [f32; 4],
    scattering: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct UnderwaterPush {
    inv_view_proj: [[f32; 4]; 4],
    camera: [f32; 4],
    water: [f32; 4],
    deep: [f32; 4],
    shallow: [f32; 4],
}

/// The post pass's share of the underwater look: the screen wobble, masked to
/// the pixels under the waterline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Wobble {
    pub strength: f32,
    /// Blur radius below the waterline (px at 1080p).
    pub blur: f32,
    pub time: f32,
    /// Near-plane height above the surface across the screen:
    /// `line[0] * u + line[1] * v + line[2]` (negative = under water).
    pub line: [f32; 3],
}

/// Near-plane world height minus the surface height, as a linear function of
/// the screen uv (the near plane is flat, so this is exact).
pub(crate) fn waterline(cam: &CameraData, surface: f32) -> [f32; 3] {
    let inv = (cam.proj * cam.view).inverse();
    // uv (0,0) is the top-left; the scene renders Y-flipped (ndc.y = 1 - 2v).
    let near = |u: f32, v: f32| inv.project_point3(Vec3::new(u * 2.0 - 1.0, 1.0 - v * 2.0, 0.0)).y;
    let y00 = near(0.0, 0.0);
    [near(1.0, 0.0) - y00, near(0.0, 1.0) - y00, y00 - surface]
}

/// Whether any pixel's near-plane point is under the surface.
pub(crate) fn any_under(line: [f32; 3]) -> bool {
    let [a, b, c] = line;
    [c, a + c, b + c, a + b + c].into_iter().fold(f32::MAX, f32::min) < WATERLINE_BLEND
}

/// The wobble for a camera, or None when it sees no water from inside.
pub(crate) fn wobble(fx: Option<&UnderwaterFx>, cam: &CameraData, time: f32) -> Option<Wobble> {
    let fx = fx?;
    let line = waterline(cam, fx.surface);
    (any_under(line) && (fx.distortion > 0.0 || fx.blur > 0.0)).then_some(Wobble { strength: fx.distortion, blur: fx.blur, time, line })
}

/// Sun + ambient light just under the surface, for the water's in-scatter.
fn illumination(light: &LightData) -> Vec3 {
    let up = (-light.direction).normalize_or(Vec3::Y).y.max(0.0);
    // A small floor so murk reads as haze, not just darkness, in dim scenes.
    (Vec3::from(light.color) * light.intensity * up * 0.3 + Vec3::from(light.ambient)).max(Vec3::splat(0.06))
}

pub(crate) struct UnderwaterPass {
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    multiply: vk::Pipeline,
    add: vk::Pipeline,
    sampler: vk::Sampler,
    pool: vk::DescriptorPool,
    sets: HashMap<u32, vk::DescriptorSet>,
    ubo: Buffer,
}

impl UnderwaterPass {
    pub fn new(device: &ash::Device, allocator: &mut Allocator) -> Result<Self> {
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
        let set_layout = unsafe {
            device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)?
        };
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .size(size_of::<UnderwaterPush>() as u32)];
        let layouts = [set_layout];
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&push),
                None,
            )?
        };
        let module = |spv: &[u8]| -> Result<vk::ShaderModule> {
            Ok(unsafe {
                device.create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(&ash::util::read_spv(&mut Cursor::new(spv))?),
                    None,
                )?
            })
        };
        let vert = module(FULLSCREEN_VERT_SPV)?;
        let frag = module(UNDERWATER_FRAG_SPV)?;
        // dst * src, then dst + src; alpha left as the scene wrote it.
        let multiply = build_pipeline(device, pipeline_layout, vert, frag, vk::BlendFactor::ZERO, vk::BlendFactor::SRC_COLOR);
        let add = build_pipeline(device, pipeline_layout, vert, frag, vk::BlendFactor::ONE, vk::BlendFactor::ONE);
        unsafe {
            device.destroy_shader_module(vert, None);
            device.destroy_shader_module(frag, None);
        }
        let (multiply, add) = (multiply?, add?);
        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?
        };
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(MAX_SETS).pool_sizes(&[
                    vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: MAX_SETS },
                    vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: MAX_SETS },
                ]),
                None,
            )?
        };
        let ubo = Buffer::new(
            device,
            allocator,
            UBO_STRIDE * MAX_SETS as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            "underwater waves",
        )?;
        Ok(Self { set_layout, pipeline_layout, multiply, add, sampler, pool, sets: HashMap::new(), ubo })
    }

    /// Fog the HDR target (COLOR_ATTACHMENT_OPTIMAL) of one view using its
    /// depth (DEPTH_ATTACHMENT_OPTIMAL, stored); both are back in those layouts
    /// afterwards. Record outside a render pass. `slot` picks this view's
    /// descriptor set (`SLOT_*` + frame / XR slot index). Skipped when no pixel
    /// is under the surface.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        slot: u32,
        hdr: (vk::Image, vk::ImageView),
        depth: (vk::Image, vk::ImageView),
        extent: vk::Extent2D,
        cam: &CameraData,
        fx: &UnderwaterFx,
        light: &LightData,
        time: f32,
        layer: u32,
    ) {
        // Crests can reach the near plane even when the surface over the
        // camera is below it.
        if !any_under(waterline(cam, fx.surface + fx.body.look1[3].max(0.0))) {
            return;
        }
        let Some(set) = self.set(device, slot) else {
            return;
        };
        let info = [vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(depth.1)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&info);
        unsafe { device.update_descriptor_sets(&[write], &[]) };
        let waves = WaterUbo {
            to_local: fx.water_to_world.inverse().to_cols_array_2d(),
            to_world: fx.water_to_world.to_cols_array_2d(),
            info: [fx.rest_surface, fx.body.look1[2], fx.clock, 0.0],
            wave_a: fx.body.wave_a,
            wave_b: fx.body.wave_b,
            absorption: fx.body.absorption(),
            scattering: fx.body.scattering(),
        };
        self.ubo.write((slot as u64 * UBO_STRIDE) as usize, bytemuck::bytes_of(&waves));

        let light = illumination(light);
        let lit = |c: [f32; 3]| Vec3::from(c) * light;
        let (deep, shallow) = if fx.body.absorption()[3] >= 1.0 {
            let a = Vec3::from_array(fx.body.absorption()[..3].try_into().unwrap());
            let s = Vec3::from_array(fx.body.scattering()[..3].try_into().unwrap());
            let albedo = (s / (a + s).max(Vec3::splat(1e-6))).to_array();
            (lit(albedo), lit(albedo))
        } else { (lit(fx.deep_color), lit(fx.shallow_color)) };
        let inv_view_proj: Mat4 = (cam.proj * cam.view).inverse();
        let mut push = UnderwaterPush {
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            camera: cam.position.extend(time).to_array(),
            // Caustics come from the forward pass for every view, not here.
            water: [fx.surface, fx.density.max(0.0), fx.light_falloff.max(0.0), 0.0],
            deep: deep.extend(0.0).to_array(),
            shallow: shallow.extend(0.0).to_array(),
        };

        image_barrier_layers(
            device, cb, depth.0, crate::depth_aspect(), layer, 1,
            crate::depth_attach_layout(), vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_SAMPLED_READ,
        );
        image_barrier_layers(
            device, cb, hdr.0, vk::ImageAspectFlags::COLOR, layer, 1,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        );
        let area = vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent };
        unsafe {
            let att = [vk::RenderingAttachmentInfo::default()
                .image_view(hdr.1)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE)];
            device.cmd_begin_rendering(cb, &vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&att));
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: extent.width as f32,
                height: extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            device.cmd_set_viewport(cb, 0, &[viewport]);
            device.cmd_set_scissor(cb, 0, &[area]);
            for (pipeline, mode) in [(self.multiply, 0.0), (self.add, 1.0)] {
                push.deep[3] = mode;
                device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
                device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline_layout, 0, &[set], &[]);
                device.cmd_push_constants(cb, self.pipeline_layout, vk::ShaderStageFlags::FRAGMENT, 0, bytemuck::bytes_of(&push));
                device.cmd_draw(cb, 3, 1, 0, 0);
            }
            device.cmd_end_rendering(cb);
        }
        image_barrier_layers(
            device, cb, depth.0, crate::depth_aspect(), layer, 1,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, crate::depth_attach_layout(),
            vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_SAMPLED_READ,
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
        );
    }

    fn set(&mut self, device: &ash::Device, slot: u32) -> Option<vk::DescriptorSet> {
        if let Some(&s) = self.sets.get(&slot) {
            return Some(s);
        }
        let layouts = [self.set_layout];
        let set = unsafe {
            device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(self.pool).set_layouts(&layouts))
        }
        .map_err(|e| tracing::error!("underwater descriptor set: {e}"))
        .ok()?[0];
        // Each slot reads its own stretch of the wave buffer.
        let info = [vk::DescriptorBufferInfo::default()
            .buffer(self.ubo.handle)
            .offset(slot as u64 * UBO_STRIDE)
            .range(size_of::<WaterUbo>() as u64)];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&info);
        unsafe { device.update_descriptor_sets(&[write], &[]) };
        self.sets.insert(slot, set);
        Some(set)
    }

    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        self.ubo.destroy(device, allocator);
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_sampler(self.sampler, None);
            device.destroy_pipeline(self.multiply, None);
            device.destroy_pipeline(self.add, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
        self.sets.clear();
    }
}

fn build_pipeline(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    vert: vk::ShaderModule,
    frag: vk::ShaderModule,
    src: vk::BlendFactor,
    dst: vk::BlendFactor,
) -> Result<vk::Pipeline> {
    let stages = [
        vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vert).name(c"main"),
        vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(frag).name(c"main"),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default();
    let blend = [vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(true)
        .src_color_blend_factor(src)
        .dst_color_blend_factor(dst)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ZERO)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE)
        .alpha_blend_op(vk::BlendOp::ADD)
        .color_write_mask(vk::ColorComponentFlags::R | vk::ColorComponentFlags::G | vk::ColorComponentFlags::B)];
    let color_blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let formats = [HDR_FORMAT];
    let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .push_next(&mut rendering);
    let pipeline = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None).map_err(|(_, e)| e)?[0]
    };
    Ok(pipeline)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera(pos: Vec3, look: Vec3) -> CameraData {
        CameraData {
            view: Mat4::look_at_rh(pos, pos + look, Vec3::Y),
            proj: Mat4::perspective_rh(60f32.to_radians(), 16.0 / 9.0, 0.05, 500.0),
            position: pos,
        }
    }

    fn at(line: [f32; 3], u: f32, v: f32) -> f32 {
        line[0] * u + line[1] * v + line[2]
    }

    #[test]
    fn waterline_splits_a_straddling_near_plane() {
        // Level camera with its eye 1 cm under the surface at y = 0.
        let cam = camera(Vec3::new(0.0, -0.01, 0.0), Vec3::NEG_Z);
        let line = waterline(&cam, 0.0);
        assert!(any_under(line));
        assert!(at(line, 0.5, 0.0) > 0.0, "top edge of the near plane is above the water");
        assert!(at(line, 0.5, 1.0) < 0.0, "bottom edge is under it");
        assert!(line[0].abs() < 1e-5, "a level camera's waterline is horizontal");
    }

    #[test]
    fn deep_and_high_cameras() {
        let deep = waterline(&camera(Vec3::new(0.0, -3.0, 0.0), Vec3::NEG_Z), 0.0);
        assert!([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)].iter().all(|&(u, v)| at(deep, u, v) < 0.0));
        let high = waterline(&camera(Vec3::new(0.0, 2.0, 0.0), Vec3::new(0.0, -1.0, -1.0)), 0.0);
        assert!(!any_under(high), "looking down at water from above isn't underwater");
    }

    #[test]
    fn wobble_needs_distortion_and_a_pixel_under() {
        let fx = UnderwaterFx {
            surface: 0.0,
            shallow_color: [0.1; 3],
            deep_color: [0.0; 3],
            density: 0.1,
            light_falloff: 0.05,
            distortion: 1.0,
            blur: 0.0,
            caustics: 0.5,
            water_to_world: Mat4::IDENTITY,
            body: crate::GpuWater::default(),
            rest_surface: 0.0,
            clock: 0.0,
        };
        let cam = camera(Vec3::new(0.0, -1.0, 0.0), Vec3::NEG_Z);
        assert!(wobble(Some(&fx), &cam, 0.0).is_some());
        assert!(wobble(Some(&UnderwaterFx { distortion: 0.0, ..fx }), &cam, 0.0).is_none());
        assert!(wobble(None, &cam, 0.0).is_none());
        assert!(wobble(Some(&fx), &camera(Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Z), 0.0).is_none());
    }
}
