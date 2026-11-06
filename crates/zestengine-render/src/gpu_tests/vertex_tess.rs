//! UNITY-11 (vertex hooks) and UNITY-12 (tessellated displacement) on a REAL `Renderer`
//! (invisible X11 window, real device, validation layers when the environment has them).
//!
//! Everything here is checked against numbers worked out from the scene, not against
//! another render of the same code:
//!
//!  * a displaced plane seen edge-on: the silhouette's top row, side columns and span are the
//!    projected height-map plateau;
//!  * the shadow of a displaced wall lands where the displaced surface (not the flat mesh) puts it;
//!  * the shadow of a swaying column (a vertex hook) lands where the swayed column puts it, and a
//!    control shader without the hook still casts the straight column's shadow;
//!  * the depth prepass and the EQUAL-tested main pass agree pixel for pixel (and a deliberately
//!    flat prepass does not, so the check can fail);
//!  * the shadow pass's depth equals the camera pass's depth when the light is the camera, for a
//!    displaced material and for a hooked one;
//!  * the XR multiview pass draws the displaced material and matches the per-eye pass.
//!
//! Contact sheet: `docs/design/vertex-hook-tessellation-contact-sheet.png`.

use std::sync::Arc;

use anyhow::{Result, ensure};
use ash::vk;
use glam::{Mat4, Vec3};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::platform::x11::EventLoopBuilderExtX11;
use winit::window::{Window, WindowId};

use crate::pipeline::InstanceData;
use crate::{
    CameraData, DrawCmd, DrawMode, FrameInput, LightData, LightInstance, LightKind, MaterialDesc, MaterialFeatures, MaterialHandle, MaterialParams, MeshData, MeshHandle,
    PushData, Renderer, ShadowCasting, ShadowPush, TessParams, TextureData, Vertex,
};

const W: u32 = 320;
const H: u32 = 240;
const FOV: f32 = std::f32::consts::FRAC_PI_3;

// ---------------------------------------------------------------- scene helpers

/// A grid of `n x n` quads on the parallelogram `origin + u * s + v * t` (s, t in 0..1), facing
/// `u x v`. UVs run 0..1 along u and v; the tangent follows u with the handedness that makes the
/// bitangent follow +v.
fn face_grid(origin: Vec3, u: Vec3, v: Vec3, n: u32) -> MeshData {
    let normal = u.cross(v).normalize();
    let tangent = u.normalize();
    let w = if normal.cross(tangent).dot(v) >= 0.0 { 1.0 } else { -1.0 };
    let mut vertices = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            let (s, t) = (i as f32 / n as f32, j as f32 / n as f32);
            vertices.push(Vertex {
                position: (origin + u * s + v * t).to_array(),
                normal: normal.to_array(),
                uv: [s, t],
                tangent: [tangent.x, tangent.y, tangent.z, w],
                color: [1.0; 4],
                ..Default::default()
            });
        }
    }
    let mut indices = Vec::new();
    for j in 0..n {
        for i in 0..n {
            let a = j * (n + 1) + i;
            let (b, c, d) = (a + 1, a + n + 1, a + n + 2);
            indices.extend([a, b, c, b, d, c]);
        }
    }
    MeshData { vertices, indices, has_lightmap_uv: false }
}

/// Horizontal plane y = 0 facing +Y, `2 half` across.
fn floor_grid(half: f32, n: u32) -> MeshData {
    face_grid(Vec3::new(-half, 0.0, -half), Vec3::Z * (2.0 * half), Vec3::X * (2.0 * half), n)
}

/// Vertical plane z = 0 facing +Z, `2 half` across.
fn wall_grid(half: f32, n: u32) -> MeshData {
    face_grid(Vec3::new(-half, -half, 0.0), Vec3::X * (2.0 * half), Vec3::Y * (2.0 * half), n)
}

/// An upright column: x, z within `hw`, y from 0 to `h`, `seg` segments tall, four sides and a top.
fn column(hw: f32, h: f32, seg: u32) -> MeshData {
    let mut all = MeshData { vertices: Vec::new(), indices: Vec::new(), has_lightmap_uv: false };
    let mut add = |m: MeshData| {
        let base = all.vertices.len() as u32;
        all.vertices.extend(m.vertices);
        all.indices.extend(m.indices.into_iter().map(|i| i + base));
    };
    let (x, y, z) = (Vec3::X * (2.0 * hw), Vec3::Y * h, Vec3::Z * (2.0 * hw));
    // Each side as a grid `seg` tall (a coarse width: the sway only depends on height).
    let side = |origin: Vec3, u: Vec3, v: Vec3| {
        let mut m = face_grid(origin, u, v, seg);
        // face_grid is square in uv; fine, the columns are thin.
        m.has_lightmap_uv = false;
        m
    };
    add(side(Vec3::new(-hw, 0.0, hw), x, y)); // +Z
    add(side(Vec3::new(hw, 0.0, -hw), -x, y)); // -Z
    add(side(Vec3::new(hw, 0.0, hw), -z, y)); // +X
    add(side(Vec3::new(-hw, 0.0, -hw), z, y)); // -X
    add(side(Vec3::new(-hw, h, hw), x, -z)); // +Y
    all
}

fn height_texture(r: &mut Renderer, size: u32, f: impl Fn(f32, f32) -> f32) -> crate::TextureHandle {
    let mut pixels = Vec::new();
    for y in 0..size {
        for x in 0..size {
            let v = (f((x as f32 + 0.5) / size as f32, (y as f32 + 0.5) / size as f32).clamp(0.0, 1.0) * 255.0).round() as u8;
            pixels.extend([v, v, v, 255]);
        }
    }
    r.upload_texture(&TextureData { width: size, height: size, pixels, srgb: false, hdr: false }).unwrap()
}

fn tess(scale: f32, mid: f32, edge: f32, max: f32, phong: f32) -> TessParams {
    TessParams { enabled: true, scale, midlevel: mid, edge_px: edge, max_factor: max, phong }
}

/// A material shown as flat white (emission), optionally displaced by `height`.
fn white_material(r: &mut Renderer, tess: Option<TessParams>, height: Option<crate::TextureHandle>) -> MaterialHandle {
    r.create_material(&MaterialDesc {
        name: "tess white".into(),
        params: MaterialParams {
            base_color: [0.0, 0.0, 0.0, 1.0],
            emission_color: [1.0; 3],
            emission_intensity: 1.0,
            roughness: 1.0,
            tess: tess.unwrap_or_default(),
            ..Default::default()
        },
        features: MaterialFeatures { emission: true, double_sided: true, ..Default::default() },
        displacement: height,
        ..Default::default()
    })
    .unwrap()
}

/// A lit grey material (no emission), optionally displaced.
fn lit_material(r: &mut Renderer, tess: Option<TessParams>, height: Option<crate::TextureHandle>) -> MaterialHandle {
    r.create_material(&MaterialDesc {
        name: "tess lit".into(),
        params: MaterialParams { base_color: [0.8, 0.8, 0.8, 1.0], roughness: 1.0, tess: tess.unwrap_or_default(), ..Default::default() },
        // Single sided on purpose: the tessellated triangles must keep the patch's winding.
        features: MaterialFeatures::default(),
        displacement: height,
        ..Default::default()
    })
    .unwrap()
}

fn draw(mesh: MeshHandle, material: MaterialHandle, transform: Mat4, casts: bool) -> DrawCmd {
    DrawCmd {
        gi_source: None,
        mesh,
        material,
        transform,
        highlight: 0.0,
        mesh_center: Vec3::ZERO,
        bound_radius: 40.0,
        lightmap_layer: -1,
        lightmap_size: 0,
        static_geometry: false,
        planar_index: -1,
        occluded: false,
        shadow_casting: if casts { ShadowCasting::On } else { ShadowCasting::Off },
        instance_color: None,
        instance_param: None,
    }
}

fn sun(dir: Vec3, intensity: f32, shadows: bool) -> LightInstance {
    LightInstance {
        kind: LightKind::Directional,
        position: Vec3::ZERO,
        direction: dir.normalize(),
        color: [1.0; 3],
        intensity,
        range: 100.0,
        spot_inner_deg: 0.0,
        spot_outer_deg: 0.0,
        cast_shadows: shadows,
        soft_shadows: false, radius: 0.0,
        shadow_bias: 0.0,
        baked: false,
    }
}

#[derive(Clone, Copy)]
struct Cam {
    eye: Vec3,
    target: Vec3,
    up: Vec3,
}

impl Cam {
    fn data(&self) -> CameraData {
        CameraData { view: Mat4::look_at_rh(self.eye, self.target, self.up), proj: Mat4::perspective_rh(FOV, W as f32 / H as f32, 0.1, 100.0), position: self.eye }
    }

    /// Pixel (x, y) of a world point.
    fn project(&self, p: Vec3) -> (f32, f32) {
        let d = self.data();
        let c = d.proj * d.view * p.extend(1.0);
        ((c.x / c.w * 0.5 + 0.5) * W as f32, (0.5 - c.y / c.w * 0.5) * H as f32)
    }
}

struct Img {
    px: Vec<u8>,
}

impl Img {
    fn lum(&self, x: i32, y: i32) -> i32 {
        let (x, y) = (x.clamp(0, W as i32 - 1) as usize, y.clamp(0, H as i32 - 1) as usize);
        let i = (y * W as usize + x) * 4;
        self.px[i] as i32 + self.px[i + 1] as i32 + self.px[i + 2] as i32
    }

    fn lit_count(&self, min: i32) -> usize {
        (0..H as i32).flat_map(|y| (0..W as i32).map(move |x| (x, y))).filter(|&(x, y)| self.lum(x, y) > min).count()
    }
}

/// Render one frame (a few times, so the viewport target exists and the frame slots settle) and read the viewport.
#[allow(clippy::too_many_arguments)]
fn shoot(r: &mut Renderer, sheet: &mut Vec<(String, Vec<u8>)>, label: &str, cam: Cam, draws: &[DrawCmd], lights: &[LightInstance], ambient: f32, time: f32) -> Result<Img> {
    let mut input: FrameInput = crate::cascade_tests::input(cam.data());
    input.render_viewport = false;
    input.viewport_extent = Some([W, H]);
    input.clear_color = [0.0, 0.0, 0.0, 1.0];
    input.light = LightData { ambient: [ambient; 3], intensity: 0.0, env_intensity: 0.0, ..LightData::default() };
    input.lights = lights;
    input.draws = draws;
    input.time = time;
    let mut px = None;
    for _ in 0..5 {
        r.render(&input)?;
        px = r.debug_read_viewport()?;
    }
    let (w, h, px) = px.ok_or_else(|| anyhow::anyhow!("{label}: no viewport target"))?;
    ensure!((w, h) == (W, H), "{label}: viewport is {w}x{h}");
    sheet.push((label.to_owned(), px.clone()));
    // Written as it grows, so a failing assertion still leaves the pictures to look at.
    write_sheet(sheet);
    Ok(Img { px })
}

fn glslc_fragment(src: &str) -> Vec<u8> {
    use std::io::Write as _;
    use std::process::{Command, Stdio};
    let mut child = Command::new("glslc")
        .args(["--target-env=vulkan1.3", "-fshader-stage=fragment", "-O", "-", "-o", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("glslc");
    child.stdin.take().unwrap().write_all(src.as_bytes()).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    out.stdout
}

/// A custom fragment shader that paints a constant yellow (the vertex hook is the test subject).
const YELLOW_FRAG: &str = r#"#version 450
layout(location = 0) out vec4 o_color;
layout(location = 1) out vec4 o_gbuf;
void main() { o_color = vec4(1.0, 0.9, 0.1, 1.0); o_gbuf = vec4(0.0); }
"#;

/// What `zestengine_assets::vertex_hook_glsl` produces for a sway shader, written by hand: the
/// column leans in +x by `sin(time)` metres at its top (object y = 2).
const SWAY_HOOK: &str = r#"
struct VertexData {
    vec3 position; vec3 normal; vec4 tangent; vec2 uv; vec2 uv1; vec4 color;
    mat4 model; vec3 world_position; vec3 world_offset; float time; int vertex_index;
};
void zestengine_vertex(inout VertexData v) {
    float h = clamp(v.position.y / 2.0, 0.0, 1.0);
    v.world_offset.x += sin(v.time) * h;
}
"#;

// ---------------------------------------------------------------- manual passes

/// A colour or depth image we own.
struct Tex {
    image: vk::Image,
    view: vk::ImageView,
    memory: Option<gpu_allocator::vulkan::Allocation>,
    layers: u32,
}

impl Tex {
    fn new(r: &Renderer, format: vk::Format, usage: vk::ImageUsageFlags, layers: u32) -> Result<Self> {
        let device = &r.ctx.device;
        let aspect = if format == vk::Format::R8G8B8A8_UNORM { vk::ImageAspectFlags::COLOR } else if crate::sprite_gpu::has_stencil(format) { vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL } else { vk::ImageAspectFlags::DEPTH };
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D { width: W, height: H, depth: 1 })
            .mip_levels(1)
            .array_layers(layers)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.create_image(&info, None)? };
        let req = unsafe { device.get_image_memory_requirements(image) };
        let memory = r.allocator().lock().unwrap().allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "vertex-tess test image",
            requirements: req,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        })?;
        unsafe { device.bind_image_memory(image, memory.memory(), memory.offset())? };
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
        Ok(Self { image, view, memory: Some(memory), layers })
    }

    fn destroy(mut self, r: &Renderer) {
        unsafe {
            r.ctx.device.destroy_image_view(self.view, None);
            r.ctx.device.destroy_image(self.image, None);
        }
        if let Some(m) = self.memory.take() {
            let _ = r.allocator().lock().unwrap().free(m);
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Passes {
    /// One colour + depth pass, depth LESS_OR_EQUAL.
    Plain,
    /// Depth prepass, then a colour pass that tests EQUAL (what the renderer does with Flux GI on).
    PrepassEqual,
    /// A prepass drawn WITHOUT the tessellation (flat), then the EQUAL colour pass: the control
    /// that shows the comparison can fail.
    FlatPrepassEqual,
}

/// Draw `mesh` with `material` (identity model) into an RGBA8 target with the frame set the last
/// `render` call wrote (so the camera is that frame's), using the renderer's own pipeline keys.
fn raster(r: &mut Renderer, mesh: MeshHandle, material: MaterialHandle, passes: Passes) -> Result<Vec<u8>> {
    let device = r.ctx.device.clone();
    let slot = (r.frame_index + crate::FRAMES_IN_FLIGHT - 1) % crate::FRAMES_IN_FLIGHT;
    let frame_set = r.frame_sets[slot];
    let depth_fmt = crate::depth_format();
    let stencil = crate::sprite_gpu::has_stencil(depth_fmt);
    let color = Tex::new(r, vk::Format::R8G8B8A8_UNORM, vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::SAMPLED, 1)?;
    let depth = Tex::new(r, depth_fmt, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT, 1)?;
    r.pipeline_cache.set_target_format(vk::Format::R8G8B8A8_UNORM);
    r.pipeline_cache.set_target_views(false, 1);
    let key = |r: &Renderer, mode: DrawMode, flat: bool| {
        let mut k = crate::scene_pipeline_key(&r.materials[material.0], false, false, false, mode, false);
        if flat {
            k.tess = false;
        }
        k
    };
    let main_key = key(r, if passes == Passes::Plain { DrawMode::Normal } else { DrawMode::DepthEqual }, false);
    let pre_key = key(r, DrawMode::Prepass, passes == Passes::FlatPrepassEqual);
    let main_pipe = r.pipeline_cache.get(&device, main_key)?;
    let pre_pipe = r.pipeline_cache.get(&device, pre_key)?;
    let inst = InstanceData { model: Mat4::IDENTITY.to_cols_array_2d(), color: [1.0; 4], param: [0.0; 4] };
    let inst_buf = {
        let al = r.allocator();
        let mut a = al.lock().unwrap();
        crate::alloc::upload_buffer(&device, &mut a, r.command_pool, r.ctx.queue, bytemuck::bytes_of(&inst), vk::BufferUsageFlags::VERTEX_BUFFER, "vertex-tess instance")?
    };
    let m = &r.materials[material.0];
    let (set1, set2) = (m.set, if m.custom_shader.is_some() { Some(m.custom_set) } else { None });
    let mesh_ref = &r.meshes[mesh.0];
    let (vbuf, ibuf, itype, icount) = (mesh_ref.vertex_buffer.handle, mesh_ref.index_buffer.handle, mesh_ref.index_type, mesh_ref.index_count);
    let layout = r.pipeline_cache.layout;
    let mut push: PushData = bytemuck::Zeroable::zeroed();
    push.model[0] = [f32::from_bits(u32::MAX); 4];
    if m.custom_shader.is_none() {
        push.base_color = [0.0, 0.0, 0.0, 1.0];
        push.emission = [1.0, 1.0, 1.0, 0.0];
        push.params0 = [0.0, 1.0, 3.0, 1.0];
        push.params1 = [0.5, 1.0, 1.0, -1.0];
    }
    let attach = if stencil { vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL } else { vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL };
    let area = vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: W, height: H } };
    let record_pass = |cb: vk::CommandBuffer, pipe: vk::Pipeline, with_color: bool, load_depth: bool| unsafe {
        let catt = [vk::RenderingAttachmentInfo::default()
            .image_view(color.view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0; 4] } })];
        let datt = vk::RenderingAttachmentInfo::default()
            .image_view(depth.view)
            .image_layout(attach)
            .load_op(if load_depth { vk::AttachmentLoadOp::LOAD } else { vk::AttachmentLoadOp::CLEAR })
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } });
        let satt = crate::stencil_attachment_for(&datt, stencil);
        let colors: &[vk::RenderingAttachmentInfo] = if with_color { &catt } else { &[] };
        device.cmd_begin_rendering(cb, &vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(colors).depth_attachment(&datt).stencil_attachment(&satt));
        device.cmd_set_viewport(cb, 0, &[vk::Viewport { x: 0.0, y: H as f32, width: W as f32, height: -(H as f32), min_depth: 0.0, max_depth: 1.0 }]);
        device.cmd_set_scissor(cb, 0, &[area]);
        device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipe);
        device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, layout, 0, &[frame_set], &[]);
        device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, layout, 1, &[set1], &[]);
        if let Some(s2) = set2 {
            device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, layout, 2, &[s2], &[]);
        }
        device.cmd_push_constants(cb, layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, bytemuck::bytes_of(&push));
        device.cmd_bind_vertex_buffers(cb, 0, &[vbuf], &[0]);
        device.cmd_bind_vertex_buffers(cb, 1, &[inst_buf.handle], &[0]);
        device.cmd_bind_index_buffer(cb, ibuf, 0, itype);
        device.cmd_draw_indexed(cb, icount, 1, 0, 0, 0);
        device.cmd_end_rendering(cb);
    };
    let ds = (vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS, vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE);
    crate::alloc::one_time_submit(&device, r.command_pool, r.ctx.queue, |cb| {
        crate::image_barrier(&device, cb, color.image, vk::ImageAspectFlags::COLOR, vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::PipelineStageFlags2::TOP_OF_PIPE, vk::AccessFlags2::empty(), vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE);
        let depth_aspect = if stencil { vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL } else { vk::ImageAspectFlags::DEPTH };
        crate::image_barrier(&device, cb, depth.image, depth_aspect, vk::ImageLayout::UNDEFINED, attach, vk::PipelineStageFlags2::TOP_OF_PIPE, vk::AccessFlags2::empty(), ds.0, ds.1);
        if passes == Passes::Plain {
            record_pass(cb, main_pipe, true, false);
        } else {
            record_pass(cb, pre_pipe, false, false);
            let mem = [vk::MemoryBarrier2::default().src_stage_mask(ds.0).src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE).dst_stage_mask(ds.0).dst_access_mask(ds.1)];
            unsafe { device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&mem)) };
            record_pass(cb, main_pipe, true, true);
        }
        crate::image_barrier(&device, cb, color.image, vk::ImageAspectFlags::COLOR, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE, vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_SAMPLED_READ);
    })?;
    let (_, _, px) = r.read_back_color(W, H, color.image)?;
    let mut inst_buf = inst_buf;
    inst_buf.destroy(&device, &mut r.allocator().lock().unwrap());
    color.destroy(r);
    depth.destroy(r);
    r.pipeline_cache.set_target_format(crate::post::HDR_FORMAT);
    Ok(px)
}

/// Depth (f32 per pixel, row-major) a draw of `material` leaves: through the CAMERA pipelines
/// (a depth prepass) when `shadow` is false, through the SHADOW pipeline with the camera's own
/// view-projection as the light when it is true. Same viewport, so the two are comparable.
fn depth_of(r: &mut Renderer, mesh: MeshHandle, material: MaterialHandle, shadow: bool, flat_shadow: bool) -> Result<Vec<f32>> {
    let device = r.ctx.device.clone();
    let slot = (r.frame_index + crate::FRAMES_IN_FLIGHT - 1) % crate::FRAMES_IN_FLIGHT;
    let frame_set = r.frame_sets[slot];
    let ubo = r.last_frame_ubo.expect("a frame was rendered");
    let fmt = if shadow { vk::Format::D32_SFLOAT } else { crate::depth_format() };
    let stencil = !shadow && crate::sprite_gpu::has_stencil(fmt);
    let depth = Tex::new(r, fmt, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC, 1)?;
    let pipe = if shadow {
        let mut k = r.shadow_key_for(&r.materials[material.0]);
        if flat_shadow {
            k.tess = false;
        }
        r.pipeline_cache.get(&device, k)?
    } else {
        r.pipeline_cache.set_target_views(false, 1);
        let k = crate::scene_pipeline_key(&r.materials[material.0], false, false, false, DrawMode::Prepass, false);
        r.pipeline_cache.get(&device, k)?
    };
    let inst = InstanceData { model: Mat4::IDENTITY.to_cols_array_2d(), color: [1.0; 4], param: [0.0; 4] };
    let mut inst_buf = {
        let al = r.allocator();
        let mut a = al.lock().unwrap();
        crate::alloc::upload_buffer(&device, &mut a, r.command_pool, r.ctx.queue, bytemuck::bytes_of(&inst), vk::BufferUsageFlags::VERTEX_BUFFER, "vertex-tess instance")?
    };
    let m = &r.materials[material.0];
    let (set1, set2) = (m.set, if m.custom_shader.is_some() { Some(m.custom_set) } else { None });
    let mesh_ref = &r.meshes[mesh.0];
    let (vbuf, ibuf, itype, icount) = (mesh_ref.vertex_buffer.handle, mesh_ref.index_buffer.handle, mesh_ref.index_type, mesh_ref.index_count);
    let layout = r.pipeline_cache.layout;
    let attach = if stencil { vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL } else { vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL };
    let aspect = if stencil { vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL } else { vk::ImageAspectFlags::DEPTH };
    let area = vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: W, height: H } };
    let ds = (vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS, vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE);
    let mut host = {
        let al = r.allocator();
        let mut a = al.lock().unwrap();
        crate::alloc::Buffer::new(&device, &mut a, (W * H * 4) as u64, vk::BufferUsageFlags::TRANSFER_DST, gpu_allocator::MemoryLocation::GpuToCpu, "depth readback")?
    };
    crate::alloc::one_time_submit(&device, r.command_pool, r.ctx.queue, |cb| unsafe {
        crate::image_barrier(&device, cb, depth.image, aspect, vk::ImageLayout::UNDEFINED, attach, vk::PipelineStageFlags2::TOP_OF_PIPE, vk::AccessFlags2::empty(), ds.0, ds.1);
        let datt = vk::RenderingAttachmentInfo::default()
            .image_view(depth.view)
            .image_layout(attach)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } });
        let satt = crate::stencil_attachment_for(&datt, stencil);
        device.cmd_begin_rendering(cb, &vk::RenderingInfo::default().render_area(area).layer_count(1).depth_attachment(&datt).stencil_attachment(&satt));
        device.cmd_set_viewport(cb, 0, &[vk::Viewport { x: 0.0, y: H as f32, width: W as f32, height: -(H as f32), min_depth: 0.0, max_depth: 1.0 }]);
        device.cmd_set_scissor(cb, 0, &[area]);
        device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipe);
        device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, layout, 0, &[frame_set], &[]);
        device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, layout, 1, &[set1], &[]);
        if let Some(s2) = set2 {
            device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, layout, 2, &[s2], &[]);
        }
        if shadow {
            let push = ShadowPush { light_vp: ubo.view_proj, model: Mat4::IDENTITY.to_cols_array_2d() };
            device.cmd_push_constants(cb, layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, bytemuck::bytes_of(&push));
        } else {
            let mut push: PushData = bytemuck::Zeroable::zeroed();
            push.model[0] = [f32::from_bits(u32::MAX); 4];
            device.cmd_push_constants(cb, layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, bytemuck::bytes_of(&push));
        }
        device.cmd_bind_vertex_buffers(cb, 0, &[vbuf], &[0]);
        device.cmd_bind_vertex_buffers(cb, 1, &[inst_buf.handle], &[0]);
        device.cmd_bind_index_buffer(cb, ibuf, 0, itype);
        device.cmd_draw_indexed(cb, icount, 1, 0, 0, 0);
        device.cmd_end_rendering(cb);
        crate::image_barrier(&device, cb, depth.image, aspect, attach, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, ds.0, vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE, vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_READ);
        let c = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::DEPTH).layer_count(1))
            .image_extent(vk::Extent3D { width: W, height: H, depth: 1 });
        device.cmd_copy_image_to_buffer(cb, depth.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, host.handle, &[c]);
    })?;
    let bytes = host.mapped().map(|m| m[..(W * H * 4) as usize].to_vec()).unwrap_or_default();
    host.destroy(&device, &mut r.allocator().lock().unwrap());
    inst_buf.destroy(&device, &mut r.allocator().lock().unwrap());
    depth.destroy(r);
    Ok(bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

/// (pixels covered by either, pixels where the depths differ by more than `tol`).
fn depth_diff(a: &[f32], b: &[f32], tol: f32) -> (usize, usize) {
    let covered = |d: f32| d < 0.999_999;
    let either = a.iter().zip(b).filter(|(x, y)| covered(**x) || covered(**y)).count();
    let differ = a.iter().zip(b).filter(|(x, y)| covered(**x) != covered(**y) || (covered(**x) && (**x - **y).abs() > tol)).count();
    (either, differ)
}

// ---------------------------------------------------------------- XR

/// A 2-layer image `render_xr_stereo` draws into (and a 1-layer one for `render_xr_eye`).
fn xr_images(r: &Renderer, layers: u32) -> Result<Tex> {
    Tex::new(r, vk::Format::R8G8B8A8_UNORM, vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC, layers)
}

fn xr_read(r: &Renderer, t: &Tex) -> Result<Vec<u8>> {
    let device = r.ctx.device.clone();
    unsafe { device.device_wait_idle()? };
    let bytes = (W * H * 4 * t.layers) as u64;
    let mut host = crate::alloc::Buffer::new(&device, &mut r.allocator().lock().unwrap(), bytes, vk::BufferUsageFlags::TRANSFER_DST, gpu_allocator::MemoryLocation::GpuToCpu, "xr readback")?;
    let layers = t.layers;
    crate::alloc::one_time_submit(&device, r.command_pool, r.ctx.queue, |cb| unsafe {
        crate::image_barrier_layers(&device, cb, t.image, vk::ImageAspectFlags::COLOR, 0, layers, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::PipelineStageFlags2::ALL_COMMANDS, vk::AccessFlags2::MEMORY_WRITE, vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_READ);
        let c = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(layers))
            .image_extent(vk::Extent3D { width: W, height: H, depth: 1 });
        device.cmd_copy_image_to_buffer(cb, t.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, host.handle, &[c]);
        crate::image_barrier_layers(&device, cb, t.image, vk::ImageAspectFlags::COLOR, 0, layers, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::empty(), vk::PipelineStageFlags2::ALL_COMMANDS, vk::AccessFlags2::empty());
    })?;
    let px = host.mapped().map(|m| m[..bytes as usize].to_vec()).unwrap_or_default();
    host.destroy(&device, &mut r.allocator().lock().unwrap());
    Ok(px)
}

fn raw(t: &Tex) -> u64 {
    use ash::vk::Handle as _;
    t.image.as_raw()
}

fn lit_mask(px: &[u8], layer: usize) -> Vec<bool> {
    let n = (W * H * 4) as usize;
    px[layer * n..(layer + 1) * n].as_chunks::<4>().0.iter().map(|p| p[0] as u32 + p[1] as u32 + p[2] as u32 > 60).collect()
}

fn iou(a: &[bool], b: &[bool]) -> f64 {
    let inter = a.iter().zip(b).filter(|(x, y)| **x && **y).count();
    let union = a.iter().zip(b).filter(|(x, y)| **x || **y).count();
    inter as f64 / union.max(1) as f64
}

// ---------------------------------------------------------------- the run

pub(super) fn run(r: &mut Renderer) -> Result<String> {
    let mut report = String::new();
    let mut sheet: Vec<(String, Vec<u8>)> = Vec::new();
    let has_tess = r.tessellation_supported();
    report += &format!("[vertex/tess] device tessellation: {has_tess}\n");

    // ======================= UNITY-12: displaced plane, edge-on silhouette =======================
    if has_tess {
        let floor = r.upload_mesh(&floor_grid(2.0, 32))?;
        // White plateau over the middle half of the map, black elsewhere.
        let plateau = height_texture(r, 64, |u, v| if (0.25..0.75).contains(&u) && (0.25..0.75).contains(&v) { 1.0 } else { 0.0 });
        let m_on = white_material(r, Some(tess(1.0, 0.0, 6.0, 16.0, 0.0)), Some(plateau));
        let m_off = white_material(r, None, Some(plateau));
        let side = Cam { eye: Vec3::new(0.0, 0.0, 8.0), target: Vec3::ZERO, up: Vec3::Y };
        let off = shoot(r, &mut sheet, "T00 plane edge-on, no tessellation", side, &[draw(floor, m_off, Mat4::IDENTITY, false)], &[], 0.0, 0.0)?;
        let on = shoot(r, &mut sheet, "T01 plane edge-on, plateau displaced 1 m", side, &[draw(floor, m_on, Mat4::IDENTITY, false)], &[], 0.0, 0.0)?;
        ensure!(off.lit_count(100) == 0, "an edge-on flat plane must draw nothing, drew {}", off.lit_count(100));
        let lit = on.lit_count(100);
        ensure!(lit > 400, "the displaced plateau drew only {lit} pixels");
        // The plateau: x in [-1, 1], height 1, spanning z in [-1, 1]. The near top edge is the highest silhouette.
        let (cx, _) = side.project(Vec3::new(0.0, 0.5, 1.0));
        let col = cx.round() as i32;
        let top = (0..H as i32).find(|&y| on.lum(col, y) > 100).unwrap_or(-1);
        let bottom = (0..H as i32).rev().find(|&y| on.lum(col, y) > 100).unwrap_or(-1);
        let (_, want_top) = side.project(Vec3::new(0.0, 1.0, 0.94));
        let (_, want_bottom) = side.project(Vec3::new(0.0, 0.0, 1.0));
        let row = (want_top.round() as i32 + want_bottom.round() as i32) / 2;
        let left = (0..W as i32).find(|&x| on.lum(x, row) > 100).unwrap_or(-1);
        let right = (0..W as i32).rev().find(|&x| on.lum(x, row) > 100).unwrap_or(-1);
        let (want_left, _) = side.project(Vec3::new(-1.0, 0.5, 1.0));
        let (want_right, _) = side.project(Vec3::new(1.0, 0.5, 1.0));
        report += &format!(
            "[vertex/tess] edge-on plateau: {lit} px lit; top row {top} (projected {want_top:.1}), bottom row {bottom} ({want_bottom:.1}), left {left} ({want_left:.1}), right {right} ({want_right:.1})\n"
        );
        ensure!((top as f32 - want_top).abs() <= 3.0, "plateau top row {top} vs projected {want_top}");
        ensure!((bottom as f32 - want_bottom).abs() <= 3.0, "plateau bottom row {bottom} vs projected {want_bottom}");
        ensure!((left as f32 - want_left).abs() <= 4.0 && (right as f32 - want_right).abs() <= 4.0, "plateau sides {left}..{right} vs projected {want_left}..{want_right}");

        // ---- prepass vs main (EQUAL) on the same displaced mesh, seen from above at an angle
        let above = Cam { eye: Vec3::new(2.5, 3.0, 6.5), target: Vec3::ZERO, up: Vec3::Y };
        shoot(r, &mut sheet, "T02 plateau from above", above, &[draw(floor, m_on, Mat4::IDENTITY, false)], &[], 0.0, 0.0)?;
        let plain = raster(r, floor, m_on, Passes::Plain)?;
        let eq = raster(r, floor, m_on, Passes::PrepassEqual)?;
        let flat_pre = raster(r, floor, m_on, Passes::FlatPrepassEqual)?;
        let count = |a: &[u8]| a.as_chunks::<4>().0.iter().filter(|p| p[0] as u32 + p[1] as u32 + p[2] as u32 > 60).count();
        let differ = |a: &[u8], b: &[u8]| a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0).filter(|(x, y)| x != y).count();
        let (n_plain, d_eq, d_flat) = (count(&plain), differ(&plain, &eq), differ(&plain, &flat_pre));
        report += &format!("[vertex/tess] prepass + EQUAL main vs plain main: {n_plain} px drawn, {d_eq} differ; control (flat prepass): {d_flat} differ\n");
        ensure!(n_plain > 3000, "the displaced plane drew only {n_plain} px");
        ensure!(d_eq == 0, "prepass and main disagree on {d_eq} pixels (the EQUAL test dropped them)");
        ensure!(d_flat > 200, "the flat-prepass control only differs in {d_flat} px: the comparison cannot fail");

        // ---- the shadow pass (light = the camera) lays the same depth as the camera pass
        let cam_depth = depth_of(r, floor, m_on, false, false)?;
        let shadow_depth = depth_of(r, floor, m_on, true, false)?;
        let flat_shadow = depth_of(r, floor, m_on, true, true)?;
        let (covered, bad) = depth_diff(&cam_depth, &shadow_depth, 2e-6);
        let (_, bad_flat) = depth_diff(&cam_depth, &flat_shadow, 2e-6);
        report += &format!("[vertex/tess] shadow-pass depth vs camera depth (light = camera): {covered} px covered, {bad} differ; control (flat shadow caster): {bad_flat} differ\n");
        ensure!(covered > 3000 && bad == 0, "the shadow pass's displaced surface differs from the camera's in {bad} of {covered} px");
        ensure!(bad_flat > 200, "the flat-shadow control only differs in {bad_flat} px");

        // ---- lit, smoothly displaced terrain: the shading follows the displaced surface
        let hills = height_texture(r, 128, |u, v| 0.5 + 0.5 * (std::f32::consts::TAU * 3.0 * u).sin() * (std::f32::consts::TAU * 3.0 * v).sin());
        let terrain = r.upload_mesh(&floor_grid(2.0, 48))?;
        let m_hills = lit_material(r, Some(tess(0.4, 0.5, 10.0, 24.0, 0.5)), Some(hills));
        let m_flat = lit_material(r, None, Some(hills));
        let view = Cam { eye: Vec3::new(2.6, 3.4, 5.4), target: Vec3::new(0.0, 0.0, 0.0), up: Vec3::Y };
        let lights = [sun(Vec3::new(-0.5, -0.7, -0.45), 3.0, false)];
        let flat_img = shoot(r, &mut sheet, "T03 flat terrain, lit (tessellation off)", view, &[draw(terrain, m_flat, Mat4::IDENTITY, false)], &lights, 0.05, 0.0)?;
        let hill_img = shoot(r, &mut sheet, "T04 displaced terrain, lit (tessellation on)", view, &[draw(terrain, m_hills, Mat4::IDENTITY, false)], &lights, 0.05, 0.0)?;
        let spread = |img: &Img| {
            let (mut lo, mut hi) = (i32::MAX, 0);
            let (px, py) = view.project(Vec3::ZERO);
            let (cx, cy) = (px.round() as i32, py.round() as i32);
            for dy in -14..14 {
                for dx in -45..45 {
                    let l = img.lum(cx + dx, cy + dy);
                    lo = lo.min(l);
                    hi = hi.max(l);
                }
            }
            hi - lo
        };
        let (s_flat, s_hill) = (spread(&flat_img), spread(&hill_img));
        report += &format!("[vertex/tess] lit shading spread over the middle: flat {s_flat}, displaced {s_hill}\n");
        ensure!(s_hill > s_flat + 40, "displaced shading does not vary more than the flat plane ({s_hill} vs {s_flat})");

        // ---- shadow of a displaced wall lands where the displaced surface puts it
        let wall = r.upload_mesh(&wall_grid(1.0, 8))?;
        let receiver = r.upload_mesh(&face_grid(Vec3::new(-8.0, -8.0, 0.0), Vec3::X * 16.0, Vec3::Y * 16.0, 2))?;
        let all_white = height_texture(r, 8, |_, _| 1.0);
        let m_wall_on = white_material(r, Some(tess(1.0, 0.0, 10.0, 16.0, 0.0)), Some(all_white));
        let m_wall_off = white_material(r, None, Some(all_white));
        let m_ground = lit_material(r, None, None);
        let front = Cam { eye: Vec3::new(0.0, 0.0, 8.0), target: Vec3::ZERO, up: Vec3::Y };
        let slanted = [sun(Vec3::new(0.5, 0.0, -1.0), 3.0, true)];
        let recv = Mat4::from_translation(Vec3::new(0.0, 0.0, -2.0));
        let mut edges = Vec::new();
        for (label, mat) in [("T05 shadow of the flat wall", m_wall_off), ("T06 shadow of the wall displaced 1 m toward the light", m_wall_on)] {
            let img = shoot(r, &mut sheet, label, front, &[draw(wall, mat, Mat4::IDENTITY, true), draw(receiver, m_ground, recv, false)], &slanted, 0.0, 0.0)?;
            // The receiver is lit except in the shadow: scan the row of the wall's middle from the right edge in.
            let (_, row) = front.project(Vec3::new(2.0, 0.0, -2.0));
            let row = row.round() as i32;
            let lit_level = img.lum(W as i32 - 4, row);
            let edge = (0..W as i32 - 2).rev().find(|&x| img.lum(x, row) < lit_level / 3).unwrap_or(-1);
            ensure!(lit_level > 150, "{label}: the receiver is not lit ({lit_level})");
            edges.push(edge);
        }
        let want_flat = front.project(Vec3::new(2.0, 0.0, -2.0)).0;
        let want_disp = front.project(Vec3::new(2.5, 0.0, -2.0)).0;
        report += &format!("[vertex/tess] wall shadow right edge: flat at px {} (projected {want_flat:.1}), displaced at px {} (projected {want_disp:.1})\n", edges[0], edges[1]);
        ensure!((edges[0] as f32 - want_flat).abs() <= 4.0, "flat wall shadow edge {} vs {want_flat}", edges[0]);
        ensure!((edges[1] as f32 - want_disp).abs() <= 4.0, "displaced wall shadow edge {} vs {want_disp}", edges[1]);
        ensure!(edges[1] > edges[0] + 6, "the displaced wall's shadow did not move ({} vs {})", edges[1], edges[0]);

        // ---- XR multiview: the displaced material draws in both eyes and matches the per-eye pass
        if r.ctx.multiview && r.ctx.multiview_tessellation {
            let slab = r.upload_mesh(&wall_grid(1.5, 24))?;
            let one = height_texture(r, 8, |_, _| 1.0);
            let on = white_material(r, Some(tess(0.8, 0.0, 8.0, 16.0, 0.0)), Some(one));
            let off = white_material(r, None, Some(one));
            let (eyes, cull) = super::xr_cluster::eyes();
            let place = Mat4::from_translation(Vec3::new(0.0, 0.0, -4.0)) * Mat4::from_rotation_y(65f32.to_radians());
            let stereo = xr_images(r, 2)?;
            let eye_img = [xr_images(r, 1)?, xr_images(r, 1)?];
            let mut shots = Vec::new();
            for (name, mat) in [("off", off), ("on", on)] {
                let draws = [draw(slab, mat, place, false)];
                let mut input: FrameInput = crate::cascade_tests::input(CameraData { view: Mat4::IDENTITY, proj: cull.1, position: Vec3::ZERO });
                input.render_viewport = false;
                input.clear_color = [0.0, 0.0, 0.0, 1.0];
                input.draws = &draws;
                for _ in 0..3 {
                    r.render_xr_frame(&input)?;
                    r.render_xr_stereo(raw(&stereo), vk::Format::R8G8B8A8_UNORM.as_raw() as u32, W, H, &eyes, cull, &input)?;
                }
                let both = xr_read(r, &stereo)?;
                let mut per_eye = Vec::new();
                for e in 0..2 {
                    r.render_xr_eye(raw(&eye_img[e]), vk::Format::R8G8B8A8_UNORM.as_raw() as u32, W, H, eyes[e].view, eyes[e].proj, &input, None)?;
                    per_eye.push(xr_read(r, &eye_img[e])?);
                }
                shots.push((name, both, per_eye));
            }
            let layer = (W * H * 4) as usize;
            for e in 0..2 {
                let (_, off_both, _) = &shots[0];
                let (_, on_both, on_eye) = &shots[1];
                let (m_on_l, m_off_l, m_eye) = (lit_mask(on_both, e), lit_mask(off_both, e), lit_mask(&on_eye[e], 0));
                let n_on = m_on_l.iter().filter(|b| **b).count();
                let moved = 1.0 - iou(&m_on_l, &m_off_l);
                let agree = iou(&m_on_l, &m_eye);
                report += &format!("[vertex/tess] XR eye {e}: multiview layer draws {n_on} px; displaced vs flat silhouette differ {:.0}%; multiview vs per-eye IoU {agree:.4}\n", moved * 100.0);
                ensure!(n_on > 2000, "eye {e}: the displaced slab drew {n_on} px in the multiview layer");
                ensure!(moved > 0.15, "eye {e}: displacement barely moved the silhouette ({moved:.3})");
                ensure!(agree > 0.985, "eye {e}: multiview layer vs per-eye pass IoU {agree}");
            }
            let on_both = &shots[1].1;
            sheet.push(("T07 XR multiview, left eye, displaced slab".into(), on_both[..layer].to_vec()));
            sheet.push(("T08 XR multiview, right eye, displaced slab".into(), on_both[layer..].to_vec()));
            stereo.destroy(r);
            for t in eye_img {
                t.destroy(r);
            }
        } else {
            report += "[vertex/tess] XR multiview tessellation: not supported by this device, skipped\n";
        }
    } else {
        report += "[vertex/tess] no tessellation shader support: the displaced-material checks were skipped (such a device draws displaced materials flat)\n";
    }

    // ======================= UNITY-11: vertex hook =======================
    let frag = glslc_fragment(YELLOW_FRAG);
    let hooked = r.register_shader_with_vertex(&frag, SWAY_HOOK)?;
    let plain = r.register_shader(&frag)?;
    let make = |r: &mut Renderer, id| {
        let m = r.create_material(&MaterialDesc { name: "sway".into(), features: MaterialFeatures { double_sided: true, ..Default::default() }, ..Default::default() }).unwrap();
        r.set_material_shader(m, Some(id));
        m
    };
    let m_hook = make(r, hooked);
    let m_ctrl = make(r, plain);
    let col = r.upload_mesh(&column(0.15, 2.0, 16))?;
    let ground = r.upload_mesh(&floor_grid(9.0, 2))?;
    let m_ground = lit_material(r, None, None);
    let front = Cam { eye: Vec3::new(0.0, 1.0, 7.0), target: Vec3::new(0.0, 1.0, 0.0), up: Vec3::Y };
    let still = shoot(r, &mut sheet, "H00 column, hook at rest (time 0)", front, &[draw(col, m_hook, Mat4::IDENTITY, false)], &[], 0.0, 0.0)?;
    let swayed = shoot(r, &mut sheet, "H01 column, hook at time pi/2 (top leans 1 m)", front, &[draw(col, m_hook, Mat4::IDENTITY, false)], &[], 0.0, std::f32::consts::FRAC_PI_2)?;
    let ctrl = shoot(r, &mut sheet, "H02 column, same shader without the hook at time pi/2", front, &[draw(col, m_ctrl, Mat4::IDENTITY, false)], &[], 0.0, std::f32::consts::FRAC_PI_2)?;
    // Centroid of the lit pixels in a band near the top of the column.
    let (_, top_row) = front.project(Vec3::new(0.0, 1.85, 0.0));
    let centroid = |img: &Img| {
        let y = top_row.round() as i32;
        let (mut sum, mut n) = (0i64, 0i64);
        for x in 0..W as i32 {
            if img.lum(x, y) > 100 {
                sum += x as i64;
                n += 1;
            }
        }
        if n == 0 { f32::NAN } else { sum as f32 / n as f32 }
    };
    let (c_still, c_sway, c_ctrl) = (centroid(&still), centroid(&swayed), centroid(&ctrl));
    // At object y = 1.85 the lean is sin(pi/2) * 1.85 / 2 = 0.925 m.
    let want_shift = front.project(Vec3::new(0.925, 1.85, 0.0)).0 - front.project(Vec3::new(0.0, 1.85, 0.0)).0;
    report += &format!("[vertex/tess] hook: column band centroid at rest {c_still:.1}, swayed {c_sway:.1} (shift {:.1}, projected {want_shift:.1}), control without hook {c_ctrl:.1}\n", c_sway - c_still);
    ensure!(c_still.is_finite() && c_sway.is_finite() && c_ctrl.is_finite(), "the column vanished ({c_still} {c_sway} {c_ctrl})");
    ensure!((c_sway - c_still - want_shift).abs() <= 6.0, "hook shift {} vs projected {want_shift}", c_sway - c_still);
    ensure!((c_ctrl - c_still).abs() <= 1.5, "the control shader moved without a hook ({c_ctrl} vs {c_still})");

    // ---- the shadow follows the hook (sun slanted toward -x, camera straight down)
    let down = Cam { eye: Vec3::new(0.0, 8.0, 0.0), target: Vec3::ZERO, up: -Vec3::Z };
    let sun_lights = [sun(Vec3::new(-0.6, -1.0, 0.0), 3.0, true)];
    let mut tips = Vec::new();
    for (label, mat) in [("H03 shadow with the hook (top leans 1 m)", m_hook), ("H04 shadow of the same shader without the hook", m_ctrl)] {
        let img = shoot(r, &mut sheet, label, down, &[draw(col, mat, Mat4::IDENTITY, true), draw(ground, m_ground, Mat4::IDENTITY, false)], &sun_lights, 0.0, std::f32::consts::FRAC_PI_2)?;
        let (_, row) = down.project(Vec3::ZERO);
        let row = row.round() as i32;
        let lit_level = img.lum(W as i32 - 4, row);
        let tip = (0..W as i32).find(|&x| img.lum(x, row) < lit_level / 3).unwrap_or(-1);
        ensure!(lit_level > 150, "{label}: the ground is not lit ({lit_level})");
        tips.push(tip);
    }
    // The column's top shadow lands 1.2 m toward -x (2 m * 0.6); leaning 1 m moves it back to -0.35 (-1.35 unswayed).
    let want_hook = down.project(Vec3::new(-0.35, 0.0, 0.0)).0;
    let want_ctrl = down.project(Vec3::new(-1.35, 0.0, 0.0)).0;
    report += &format!("[vertex/tess] hook shadow: leftmost shadow px {} with the hook (projected {want_hook:.1}), {} without (projected {want_ctrl:.1})\n", tips[0], tips[1]);
    ensure!((tips[0] as f32 - want_hook).abs() <= 6.0, "swayed shadow tip {} vs projected {want_hook}", tips[0]);
    ensure!((tips[1] as f32 - want_ctrl).abs() <= 6.0, "unhooked shadow tip {} vs projected {want_ctrl}", tips[1]);

    // ---- prepass / main EQUAL and shadow == camera depth, for the hooked material
    shoot(r, &mut sheet, "H05 hooked column from the side-above", Cam { eye: Vec3::new(2.0, 3.0, 5.0), target: Vec3::new(0.5, 1.0, 0.0), up: Vec3::Y }, &[draw(col, m_hook, Mat4::IDENTITY, false)], &[], 0.0, std::f32::consts::FRAC_PI_2)?;
    let plain_px = raster(r, col, m_hook, Passes::Plain)?;
    let eq_px = raster(r, col, m_hook, Passes::PrepassEqual)?;
    let drawn = plain_px.as_chunks::<4>().0.iter().filter(|p| p[0] as u32 + p[1] as u32 + p[2] as u32 > 60).count();
    let differ = plain_px.as_chunks::<4>().0.iter().zip(eq_px.as_chunks::<4>().0).filter(|(a, b)| a != b).count();
    report += &format!("[vertex/tess] hooked column, prepass + EQUAL main vs plain main: {drawn} px drawn, {differ} differ\n");
    ensure!(drawn > 500, "the hooked column drew only {drawn} px");
    ensure!(differ == 0, "hooked prepass and main disagree on {differ} px");
    let cam_depth = depth_of(r, col, m_hook, false, false)?;
    let shadow_depth = depth_of(r, col, m_hook, true, false)?;
    let (covered, bad) = depth_diff(&cam_depth, &shadow_depth, 2e-6);
    report += &format!("[vertex/tess] hooked column, shadow-pass depth vs camera depth: {covered} px covered, {bad} differ\n");
    ensure!(covered > 500 && bad == 0, "the hooked shadow pass differs from the camera's in {bad} of {covered} px");

    write_sheet(&sheet);
    report += "[vertex/tess] contact sheet: docs/design/vertex-hook-tessellation-contact-sheet.png\n";
    for (i, (label, _)) in sheet.iter().enumerate() {
        report += &format!("[vertex/tess]   [{} , {}] {label}\n", i / 4, i % 4);
    }
    Ok(report)
}

fn write_sheet(sheet: &[(String, Vec<u8>)]) {
    if std::env::var("ZESTENGINE_FX_SHEET").is_ok_and(|v| v == "0") || sheet.is_empty() {
        return;
    }
    let cols = 4usize;
    let rows = sheet.len().div_ceil(cols);
    let (sw, sh) = (cols * W as usize, rows * H as usize);
    let mut out = vec![0u8; sw * sh * 4];
    for px in out.as_chunks_mut::<4>().0 {
        px.copy_from_slice(&[24, 24, 24, 255]);
    }
    for (i, (_, img)) in sheet.iter().enumerate() {
        let (ox, oy) = ((i % cols) * W as usize, (i / cols) * H as usize);
        for y in 0..H as usize {
            for x in 0..W as usize {
                let s = (y * W as usize + x) * 4;
                let d = ((oy + y) * sw + ox + x) * 4;
                out[d..d + 3].copy_from_slice(&img[s..s + 3]);
            }
        }
    }
    let png = super::stencil::png_rgba(sw as u32, sh as u32, &out);
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/design/vertex-hook-tessellation-contact-sheet.png");
    let _ = std::fs::create_dir_all(path.parent().unwrap());
    if let Err(e) = std::fs::write(&path, png) {
        eprintln!("[vertex/tess] could not write {}: {e}", path.display());
    }
}

// ---------------------------------------------------------------- harness

struct App {
    result: Option<Result<String, String>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.result.is_some() {
            return;
        }
        let attrs = Window::default_attributes().with_visible(false).with_inner_size(winit::dpi::LogicalSize::new(W as f64, H as f64));
        self.result = Some(match event_loop.create_window(attrs).map(Arc::new) {
            Err(e) => Err(format!("window: {e}")),
            Ok(window) => match Renderer::new(window) {
                Err(e) => Err(format!("Renderer::new: {e:#}")),
                Ok(mut r) => {
                    r.set_vsync(false);
                    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&mut r)));
                    match out {
                        Ok(Ok(report)) => Ok(report),
                        Ok(Err(e)) => Err(format!("{e:#}")),
                        Err(p) => Err(p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_else(|| "panic".into())),
                    }
                }
            },
        });
        event_loop.exit();
    }

    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, _: WindowEvent) {}
}

#[test]
fn vertex_hooks_and_tessellated_displacement_on_a_real_renderer() {
    if std::env::var("ZESTENGINE_TEST_GPU").is_ok_and(|v| v == "0") || std::env::var_os("DISPLAY").is_none() {
        eprintln!("[gpu test] no display / GPU disabled: skipped");
        return;
    }
    let _gpu = crate::cluster_headless::gpu_lock();
    let Ok(mut event_loop) = EventLoop::builder().with_x11().with_any_thread(true).build() else {
        eprintln!("[gpu test] no X11 event loop: skipped");
        return;
    };
    let mut app = App { result: None };
    event_loop.run_app_on_demand(&mut app).unwrap();
    match app.result {
        Some(Ok(report)) => eprintln!("{report}"),
        Some(Err(e)) => panic!("{e}"),
        None => panic!("the window never resumed"),
    }
}
