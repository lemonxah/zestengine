//! Editor ground grid of the Scene view.
//!
//! An infinite grid on a world plane through the origin (XZ by default, Y up),
//! drawn by one fullscreen pass over the tonemapped editor viewport, right
//! before the selection outline (`editor_grid.frag`). Every pixel's view ray
//! is intersected with the plane and the hit is tested against the scene
//! depth, so opaque geometry hides the grid; lines are anti-aliased with
//! screen-space derivatives, with powers-of-ten level of detail and distance /
//! grazing-angle fades.
//!
//! It runs in `record_viewport_scene` only: the game, the Camera tab preview,
//! reflection captures, shadow maps and XR eyes never draw it, and the editor
//! turns it off while the Scene view shows the game camera. Drawing after the
//! tonemap keeps the editor's colours exact whatever the exposure, bloom or
//! fog, and keeps the grid out of SSR / RT reflections.

use std::io::Cursor;

use anyhow::Result;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::Mat4;

use crate::frame::FRAMES_IN_FLIGHT;
use crate::selection::is_srgb;

const GRID_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/editor_grid.frag.spv"));
const FULLSCREEN_VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fullscreen.vert.spv"));

/// Minimum pixel spacing of the minor lines before the next level takes over
/// (`MIN_PX` in `editor_grid.frag`).
pub const GRID_MIN_PX: f32 = 3.0;

/// Which world plane the grid lies on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GridPlane {
    /// XZ in perspective; in orthographic views the plane facing the camera
    /// most (Top: XZ, Front: XY, Right: YZ), like other editors' 2D views.
    #[default]
    Auto,
    /// The ground plane, y = 0.
    XZ,
    /// z = 0 (2D games looking down -Z).
    XY,
    /// x = 0.
    YZ,
}

impl GridPlane {
    /// The world axis normal to the plane (0 X, 1 Y, 2 Z) for this view.
    pub fn normal_axis(self, view: Mat4, proj: Mat4) -> usize {
        match self {
            GridPlane::XZ => 1,
            GridPlane::XY => 2,
            GridPlane::YZ => 0,
            GridPlane::Auto => {
                if !is_orthographic(&proj) {
                    return 1;
                }
                // The camera looks down its -Z: that direction in world space.
                let fwd = -view.inverse().z_axis.truncate();
                let a = fwd.abs();
                if a.y >= a.x && a.y >= a.z {
                    1
                } else if a.z >= a.x {
                    2
                } else {
                    0
                }
            }
        }
    }
}

fn is_orthographic(proj: &Mat4) -> bool {
    proj.w_axis.w.abs() > 0.5
}

/// How the editor wants the grid drawn. Colours are sRGB.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EditorGridStyle {
    pub enabled: bool,
    pub plane: GridPlane,
    /// Minor line spacing in metres at the closest level (the move-snap step).
    pub unit: f32,
    /// Overall opacity, 0..1.
    pub opacity: f32,
    /// Colour of the minor and major lines.
    pub color: [f32; 3],
    /// Axis line colours for X, Y and Z (the line along each world axis).
    pub axis_colors: [[f32; 3]; 3],
}

impl Default for EditorGridStyle {
    fn default() -> Self {
        Self {
            enabled: false,
            plane: GridPlane::Auto,
            unit: 1.0,
            opacity: 1.0,
            color: [0.55, 0.565, 0.6],
            axis_colors: [[0.94, 0.42, 0.42], [0.55, 0.83, 0.42], [0.42, 0.66, 0.94]],
        }
    }
}

/// The line spacing the grid shows where one pixel spans `world_per_px`
/// metres of the plane (the length of the plane position's screen-space
/// derivative): `minor = unit * 10^n` for the smallest `n >= 0` that keeps the
/// minor cells at least [`GRID_MIN_PX`] pixels apart, `major = 10 * minor`,
/// and `fade` (0..1) the minor lines' fade toward the next level. The shader
/// evaluates the same formula per pixel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridLod {
    pub minor: f32,
    pub major: f32,
    pub fade: f32,
}

pub fn grid_lod(unit: f32, world_per_px: f32) -> GridLod {
    let unit = unit.max(1e-4);
    let lod = ((world_per_px / unit * GRID_MIN_PX).max(1e-30).log10() + 1.0).max(0.0);
    let level = lod.floor();
    let minor = unit * 10f32.powf(level);
    GridLod { minor, major: minor * 10.0, fade: lod - level }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GridPush {
    inv_view_proj: [[f32; 4]; 4],
    params: [f32; 4],
    line_color: [f32; 4],
    axis_u: [f32; 4],
    axis_v: [f32; 4],
}

const MINOR_ALPHA: f32 = 0.32;
const MAJOR_ALPHA: f32 = 0.6;
const AXIS_ALPHA: f32 = 0.85;

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

pub(crate) struct GridPass {
    pub style: EditorGridStyle,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    sampler: vk::Sampler,
    vert: vk::ShaderModule,
    frag: vk::ShaderModule,
    pool: vk::DescriptorPool,
    /// One set per frame in flight (each frame points it at its depth view).
    sets: Vec<vk::DescriptorSet>,
    /// Pipeline and the colour format it was built for.
    pipeline: Option<(vk::Format, vk::Pipeline)>,
}

impl GridPass {
    pub fn new(device: &ash::Device) -> Result<Self> {
        let module = |spv: &[u8]| -> Result<vk::ShaderModule> {
            Ok(unsafe {
                device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&ash::util::read_spv(&mut Cursor::new(spv))?), None)?
            })
        };
        let vert = module(FULLSCREEN_VERT_SPV)?;
        let frag = module(GRID_FRAG_SPV)?;
        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        let set_layout = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)? };
        let sets_l = [set_layout];
        let push = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::FRAGMENT).size(size_of::<GridPush>() as u32)];
        let layout = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&sets_l).push_constant_ranges(&push), None)? };
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
                &vk::DescriptorPoolCreateInfo::default().max_sets(FRAMES_IN_FLIGHT as u32).pool_sizes(&[vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: FRAMES_IN_FLIGHT as u32,
                }]),
                None,
            )?
        };
        let layouts = vec![set_layout; FRAMES_IN_FLIGHT];
        let sets = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts))? };
        Ok(Self { style: EditorGridStyle::default(), set_layout, layout, sampler, vert, frag, pool, sets, pipeline: None })
    }

    fn build(&self, device: &ash::Device, color_format: vk::Format) -> Result<vk::Pipeline> {
        let stages = [
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(self.vert).name(c"main"),
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(self.frag).name(c"main"),
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
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ZERO)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::R | vk::ColorComponentFlags::G | vk::ColorComponentFlags::B)];
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let formats = [color_format];
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
            .layout(self.layout)
            .push_next(&mut rendering);
        Ok(unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None).map_err(|(_, e)| e)?[0] })
    }

    /// Draw the grid over the colour target currently bound (a rendering
    /// instance of `color_format`, `extent` pixels). `depth_view` is the scene
    /// depth (depth aspect, SHADER_READ_ONLY_OPTIMAL) at the same size and
    /// `view` / `proj` the matrices the scene was drawn with.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        frame_index: usize,
        color_format: vk::Format,
        extent: vk::Extent2D,
        depth_view: vk::ImageView,
        view: Mat4,
        proj: Mat4,
    ) -> Result<bool> {
        let s = self.style;
        if !s.enabled || s.opacity <= 0.0 {
            return Ok(false);
        }
        let inv = (proj * view).inverse();
        if !inv.is_finite() {
            return Ok(false);
        }
        if self.pipeline.is_none_or(|(f, _)| f != color_format) {
            if let Some((_, old)) = self.pipeline.take() {
                // A format change is a swapchain recreate: the device is idle then.
                unsafe { device.destroy_pipeline(old, None) };
            }
            let p = self.build(device, color_format)?;
            self.pipeline = Some((color_format, p));
        }
        let pipeline = self.pipeline.unwrap().1;
        let set = self.sets[frame_index % self.sets.len()];
        let info = [vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(depth_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        unsafe {
            device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&info)],
                &[],
            )
        };
        let axis = s.plane.normal_axis(view, proj);
        // The plane's (u, v) axes, in the shader's order.
        let (u, v) = match axis {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        let linear = is_srgb(color_format);
        let conv = |c: [f32; 3], a: f32| -> [f32; 4] {
            let f = |x: f32| if linear { srgb_to_linear(x) } else { x };
            [f(c[0]), f(c[1]), f(c[2]), a]
        };
        let push = GridPush {
            inv_view_proj: inv.to_cols_array_2d(),
            params: [s.unit.max(1e-3), s.opacity.clamp(0.0, 1.0), axis as f32, if is_orthographic(&proj) { 1.0 } else { 0.0 }],
            line_color: conv(s.color, MINOR_ALPHA),
            axis_u: conv(s.axis_colors[u], MAJOR_ALPHA),
            axis_v: conv(s.axis_colors[v], AXIS_ALPHA),
        };
        unsafe {
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_set_viewport(cb, 0, &[vk::Viewport { x: 0.0, y: 0.0, width: extent.width as f32, height: extent.height as f32, min_depth: 0.0, max_depth: 1.0 }]);
            device.cmd_set_scissor(cb, 0, &[vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent }]);
            device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, self.layout, 0, &[set], &[]);
            device.cmd_push_constants(cb, self.layout, vk::ShaderStageFlags::FRAGMENT, 0, bytemuck::bytes_of(&push));
            device.cmd_draw(cb, 3, 1, 0, 0);
        }
        Ok(true)
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            if let Some((_, p)) = self.pipeline.take() {
                device.destroy_pipeline(p, None);
            }
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_sampler(self.sampler, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_shader_module(self.vert, None);
            device.destroy_shader_module(self.frag, None);
        }
    }
}

/// World position of the grid's view-ray hit for a pixel centre, the CPU twin
/// of the shader's ray / plane step (tests).
#[cfg(test)]
pub(crate) fn plane_hit(inv_view_proj: Mat4, ndc: glam::Vec2, axis: usize) -> Option<glam::Vec3> {
    let near = inv_view_proj.project_point3(ndc.extend(0.0));
    let far = inv_view_proj.project_point3(ndc.extend(1.0));
    let seg = far - near;
    let t = -near[axis] / seg[axis];
    (t > 0.0 && t < 1.0).then(|| near + seg * t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn close_up_uses_the_unit_and_far_away_steps_by_powers_of_ten() {
        // Many pixels per cell: the snap unit itself, no fade.
        let l = grid_lod(0.5, 0.001);
        assert_eq!(l.minor, 0.5);
        assert_eq!(l.major, 5.0);
        assert_eq!(l.fade, 0.0);
        // The minor spacing is always unit * 10^n (snap-aligned) and keeps at
        // least GRID_MIN_PX pixels per cell at its level, never 10x more.
        let mut last = 0.0;
        for i in 0..400 {
            let wpp = 0.0005 * 1.04f32.powi(i);
            let l = grid_lod(0.5, wpp);
            let n = (l.minor / 0.5).log10();
            assert!((n - n.round()).abs() < 1e-3, "minor {} is not unit * 10^n", l.minor);
            assert!(l.minor >= last, "spacing shrinks as the camera moves away");
            assert!((0.0..1.0).contains(&l.fade));
            assert!((l.major / l.minor - 10.0).abs() < 1e-3);
            let px_per_minor = l.minor / wpp;
            if l.minor > 0.5 {
                assert!(px_per_minor >= GRID_MIN_PX * 0.999, "minor cells {px_per_minor} px apart at {wpp} m/px");
                assert!(px_per_minor <= GRID_MIN_PX * 10.001);
            }
            last = l.minor;
        }
        assert!(last >= 50.0, "far away the grid reaches 100x the unit ({last})");
    }

    #[test]
    fn the_fade_is_continuous_across_a_level_change() {
        // Just below / above the threshold where 1 m minor lines hand over to 10 m.
        let unit = 1.0;
        let edge = 10f32.powf(0.0) / GRID_MIN_PX; // lod = 1 exactly
        let below = grid_lod(unit, edge * 0.999);
        let above = grid_lod(unit, edge * 1.001);
        assert_eq!(below.minor, 1.0);
        assert!(below.fade > 0.99, "the old minor lines have faded out ({})", below.fade);
        assert_eq!(above.minor, 10.0);
        assert!(above.fade < 0.01);
    }

    #[test]
    fn plane_choice_follows_the_view() {
        let persp = Mat4::perspective_rh(1.0, 1.5, 0.1, 100.0);
        let ortho = Mat4::orthographic_rh(-5.0, 5.0, -3.0, 3.0, -100.0, 100.0);
        let front = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 10.0), Vec3::ZERO, Vec3::Y);
        let top = Mat4::look_at_rh(Vec3::new(0.0, 10.0, 0.0), Vec3::ZERO, Vec3::NEG_Z);
        let right = Mat4::look_at_rh(Vec3::new(10.0, 0.0, 0.0), Vec3::ZERO, Vec3::Y);
        assert_eq!(GridPlane::Auto.normal_axis(front, persp), 1, "perspective keeps the ground plane");
        assert_eq!(GridPlane::Auto.normal_axis(front, ortho), 2, "2D front view: the XY plane");
        assert_eq!(GridPlane::Auto.normal_axis(top, ortho), 1);
        assert_eq!(GridPlane::Auto.normal_axis(right, ortho), 0);
        assert_eq!(GridPlane::XY.normal_axis(top, persp), 2);
    }

    #[test]
    fn view_rays_hit_the_ground_below_the_horizon_only() {
        let eye = Vec3::new(0.0, 2.0, 5.0);
        let view = Mat4::look_at_rh(eye, Vec3::new(0.0, 0.0, 0.0), Vec3::Y);
        let inv = (Mat4::perspective_rh(1.0, 1.5, 0.1, 500.0) * view).inverse();
        let hit = plane_hit(inv, glam::Vec2::ZERO, 1).expect("the screen centre looks at the origin");
        assert!(hit.length() < 1e-3 && hit.y.abs() < 1e-5, "{hit}");
        assert!(plane_hit(inv, glam::Vec2::new(0.0, 0.99), 1).is_none(), "top of the screen is sky");
    }
}
