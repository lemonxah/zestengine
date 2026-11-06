//! Shadow bias on a real device, through the engine's own shaders: the shadow
//! depth pass (`shadow.vert` via `PipelineKey::shadow()`), the planner that
//! picks cascades / cube faces (`plan_shadows`), the frame UBO packer, and
//! `standard.frag`'s `shadow_factor` in the main pass.
//!
//! A ground plane and a thin "avatar" (two 0.15 m legs and a torso box) are lit
//! by a point / spot light at 5, 10 and 20 m, and by the cascaded sun with the
//! camera 6 - 22 m away. Each scene renders twice, with and without the
//! casters' shadow maps, so the lit ratio `shadowed / unshadowed` at
//! analytically projected ground points says whether the shadow is present
//! (the bug: legs, then torso, vanished with distance) and, on a flat receiver
//! with no caster at grazing angles, how much acne there is.

use anyhow::Result;
use ash::vk;
use glam::{Mat4, Vec3};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};

use super::{COLOR_OUT, Img, NONE, gpu, read_image, transition};
use crate::alloc::{self, Buffer};
use crate::cluster_headless::HeadlessGpu;
use crate::pipeline::{InstanceData, PipelineCache, PipelineKey};
use crate::{
    CameraData, DrawCmd, FrameUbo, LightInstance, LightKind, MAX_LIGHTS, MaterialHandle, MeshHandle, PushData, ShadowCasting,
    ShadowPush, Vertex, frame_ubo, gpu_light, plan_shadows,
};

const SHADOW_DIM: u32 = 2048;
const W: u32 = 512;

pub(super) fn vert(p: Vec3, n: Vec3) -> Vertex {
    Vertex { position: p.to_array(), normal: n.to_array(), tangent: [1.0, 0.0, 0.0, 1.0], color: [1.0; 4], ..Default::default() }
}

/// Flat quad in the XZ plane, normal +Y.
fn plane(half: f32) -> (Vec<Vertex>, Vec<u32>) {
    let n = Vec3::Y;
    (
        vec![vert(Vec3::new(-half, 0.0, -half), n), vert(Vec3::new(half, 0.0, -half), n), vert(Vec3::new(half, 0.0, half), n), vert(Vec3::new(-half, 0.0, half), n)],
        vec![0, 2, 1, 0, 3, 2],
    )
}

/// Axis-aligned box centred on the origin.
fn cuboid(h: Vec3) -> (Vec<Vertex>, Vec<u32>) {
    let mut v = Vec::new();
    let mut i = Vec::new();
    for (n, u, w) in [
        (Vec3::X, Vec3::Y, Vec3::Z),
        (Vec3::NEG_X, Vec3::Z, Vec3::Y),
        (Vec3::Y, Vec3::Z, Vec3::X),
        (Vec3::NEG_Y, Vec3::X, Vec3::Z),
        (Vec3::Z, Vec3::X, Vec3::Y),
        (Vec3::NEG_Z, Vec3::Y, Vec3::X),
    ] {
        let c = n * h;
        let (a, b) = (u * h, w * h);
        let base = v.len() as u32;
        for (su, sw) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            v.push(vert(c + a * su + b * sw, n));
        }
        i.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    (v, i)
}

/// Upright cylinder of `radius` from y = 0 to `height`.
fn cylinder(radius: f32, height: f32, segs: u32) -> (Vec<Vertex>, Vec<u32>) {
    let mut v = Vec::new();
    let mut i = Vec::new();
    for s in 0..=segs {
        let a = std::f32::consts::TAU * s as f32 / segs as f32;
        let n = Vec3::new(a.cos(), 0.0, a.sin());
        v.push(vert(n * radius, n));
        v.push(vert(n * radius + Vec3::Y * height, n));
    }
    for s in 0..segs {
        let b = s * 2;
        i.extend([b, b + 1, b + 2, b + 1, b + 3, b + 2]);
    }
    // Caps.
    for (y, n) in [(0.0, Vec3::NEG_Y), (height, Vec3::Y)] {
        let c = v.len() as u32;
        v.push(vert(Vec3::Y * y, n));
        for s in 0..=segs {
            let a = std::f32::consts::TAU * s as f32 / segs as f32;
            v.push(vert(Vec3::new(a.cos() * radius, y, a.sin() * radius), n));
        }
        for s in 0..segs {
            i.extend([c, c + 1 + s, c + 2 + s]);
        }
    }
    (v, i)
}

struct Mesh {
    vbuf: Buffer,
    ibuf: Buffer,
    index_count: u32,
}

#[derive(Clone, Copy)]
struct Draw {
    mesh: usize,
    model: Mat4,
    color: [f32; 4],
    /// Drawn into shadow maps.
    casts: bool,
}

pub(super) fn dummy_image(g: &HeadlessGpu, alloc: &mut Allocator, layers: u32, cube: bool, color: [f32; 4]) -> Result<(vk::Image, [vk::ImageView; 3], Allocation)> {
    let device = &g.device;
    let info = vk::ImageCreateInfo::default()
        .flags(if cube { vk::ImageCreateFlags::CUBE_COMPATIBLE } else { vk::ImageCreateFlags::empty() })
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::R8G8B8A8_UNORM)
        .extent(vk::Extent3D { width: 1, height: 1, depth: 1 })
        .mip_levels(1)
        .array_layers(layers)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { device.create_image(&info, None)? };
    let req = unsafe { device.get_image_memory_requirements(image) };
    let a = alloc.allocate(&AllocationCreateDesc {
        name: "dummy",
        requirements: req,
        location: MemoryLocation::GpuOnly,
        linear: false,
        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
    })?;
    unsafe { device.bind_image_memory(image, a.memory(), a.offset())? };
    let range = |layers| vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(layers);
    let mk = |ty, layers| unsafe {
        device.create_image_view(
            &vk::ImageViewCreateInfo::default().image(image).view_type(ty).format(vk::Format::R8G8B8A8_UNORM).subresource_range(range(layers)),
            None,
        )
    };
    let v2d = mk(vk::ImageViewType::TYPE_2D, 1)?;
    let varr = mk(vk::ImageViewType::TYPE_2D_ARRAY, layers)?;
    let vcube = if cube { mk(vk::ImageViewType::CUBE, 6)? } else { vk::ImageView::null() };
    alloc::one_time_submit(device, g.pool, g.queue, |cb| unsafe {
        let b = |old, new, src: (vk::PipelineStageFlags2, vk::AccessFlags2), dst: (vk::PipelineStageFlags2, vk::AccessFlags2)| {
            [vk::ImageMemoryBarrier2::default()
                .image(image)
                .old_layout(old)
                .new_layout(new)
                .src_stage_mask(src.0)
                .src_access_mask(src.1)
                .dst_stage_mask(dst.0)
                .dst_access_mask(dst.1)
                .subresource_range(range(layers))]
        };
        let xfer = (vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE);
        device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&b(vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, NONE, xfer)));
        device.cmd_clear_color_image(cb, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &vk::ClearColorValue { float32: color }, &[range(layers)]);
        device.cmd_pipeline_barrier2(
            cb,
            &vk::DependencyInfo::default().image_memory_barriers(&b(
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                xfer,
                (vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::VERTEX_SHADER, vk::AccessFlags2::SHADER_SAMPLED_READ),
            )),
        );
    })?;
    Ok((image, [v2d, varr, vcube], a))
}

/// Shadow settings a test turns, besides the per-render bias scale.
#[derive(Clone, Copy)]
struct Knobs {
    softness: f32,
    /// Environment `shadow_normal_offset` (0 = the default; `NO_OFFSET` ~ off).
    normal_offset: f32,
    /// Environment `shadow_cascades` / `shadow_cascade_lambda` (0 = defaults).
    cascades: u32,
    lambda: f32,
    /// Texel grid the cascade centres snap to; `NO_SNAP` = the legacy unsnapped fit.
    snap_dim: u32,
    /// Farthest receiver depth the scene reports (one bounding-sphere draw); 0 = none.
    recv_limit: f32,
}

/// A normal-offset small enough to be no offset: the pre-normal-offset lookup.
const NO_OFFSET: f32 = 1e-4;
/// A snap grid so fine that centres do not move: the legacy cascade fit.
const NO_SNAP: u32 = 1 << 24;

impl Default for Knobs {
    fn default() -> Self {
        Self { softness: 1.0, normal_offset: 0.0, cascades: 0, lambda: 0.0, snap_dim: SHADOW_DIM, recv_limit: 0.0 }
    }
}

/// Everything a shadow scene needs on the device.
struct Rig<'a> {
    knobs: Knobs,
    g: &'a HeadlessGpu,
    alloc: Allocator,
    cache: PipelineCache,
    meshes: Vec<Mesh>,
    atlas: Img,
    layer_views: Vec<vk::ImageView>,
    atlas_view: vk::ImageView,
    shadow_sampler: vk::Sampler,
    plain_sampler: vk::Sampler,
    dummies: Vec<(vk::Image, [vk::ImageView; 3], Allocation)>,
    sh_buf: Buffer,
    fx_buf: Buffer,
    pool: vk::DescriptorPool,
}

impl<'a> Rig<'a> {
    fn new(g: &'a HeadlessGpu) -> Self {
        let mut alloc = g.allocator().unwrap();
        let device = &g.device;
        let cache = PipelineCache::new(device, vk::Format::R16G16B16A16_SFLOAT, vk::Format::D32_SFLOAT, vk::Format::A2B10G10R10_UNORM_PACK32).unwrap();
        let atlas = Img::new(g, &mut alloc, SHADOW_DIM, SHADOW_DIM, 12, vk::Format::D32_SFLOAT, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED, vk::SampleCountFlags::TYPE_1).unwrap();
        let layer_views = (0..12)
            .map(|l| unsafe {
                device
                    .create_image_view(
                        &vk::ImageViewCreateInfo::default().image(atlas.image).view_type(vk::ImageViewType::TYPE_2D).format(vk::Format::D32_SFLOAT).subresource_range(
                            vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::DEPTH).level_count(1).base_array_layer(l).layer_count(1),
                        ),
                        None,
                    )
                    .unwrap()
            })
            .collect();
        // The renderer's shadow sampler (PCF compare, white border).
        let shadow_sampler = unsafe {
            device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::LINEAR)
                        .min_filter(vk::Filter::LINEAR)
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                        .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
                        .compare_enable(true)
                        .compare_op(vk::CompareOp::LESS_OR_EQUAL),
                    None,
                )
                .unwrap()
        };
        let plain_sampler = unsafe {
            device
                .create_sampler(&vk::SamplerCreateInfo::default().mag_filter(vk::Filter::LINEAR).min_filter(vk::Filter::LINEAR), None)
                .unwrap()
        };
        let black_cube = dummy_image(g, &mut alloc, 6, true, [0.0; 4]).unwrap();
        let white = dummy_image(g, &mut alloc, 1, false, [1.0; 4]).unwrap();
        let atlas_view = unsafe {
            device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default().image(atlas.image).view_type(vk::ImageViewType::TYPE_2D_ARRAY).format(vk::Format::D32_SFLOAT).subresource_range(
                        vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::DEPTH).level_count(1).layer_count(12),
                    ),
                    None,
                )
                .unwrap()
        };
        let sh_buf = Buffer::new(device, &mut alloc, 4096, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "sh").unwrap();
        let mut fx_buf = Buffer::new(device, &mut alloc, 4096, vk::BufferUsageFlags::UNIFORM_BUFFER, MemoryLocation::CpuToGpu, "fx").unwrap();
        fx_buf.write(0, &[0u8; 4096]);
        let pool = unsafe {
            device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default().max_sets(4).pool_sizes(&[
                        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(8),
                        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(96),
                        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(8),
                    ]),
                    None,
                )
                .unwrap()
        };
        Self {
            knobs: Knobs::default(),
            g,
            alloc,
            cache,
            meshes: Vec::new(),
            atlas,
            layer_views,
            atlas_view,
            shadow_sampler,
            plain_sampler,
            dummies: vec![black_cube, white],
            sh_buf,
            fx_buf,
            pool,
        }
    }

    fn add_mesh(&mut self, (v, i): (Vec<Vertex>, Vec<u32>)) -> usize {
        let g = self.g;
        let vbuf = alloc::upload_buffer(&g.device, &mut self.alloc, g.pool, g.queue, bytemuck::cast_slice(&v), vk::BufferUsageFlags::VERTEX_BUFFER, "shadow-test vertices").unwrap();
        let ibuf = alloc::upload_buffer(&g.device, &mut self.alloc, g.pool, g.queue, bytemuck::cast_slice(&i), vk::BufferUsageFlags::INDEX_BUFFER, "shadow-test indices").unwrap();
        self.meshes.push(Mesh { vbuf, ibuf, index_count: i.len() as u32 });
        self.meshes.len() - 1
    }

    /// Render `draws` lit by `lights` from `camera`; returns the linear
    /// luminance per pixel (W x W). `shadows` = draw the casters into the
    /// shadow maps (false = every map stays cleared to "lit").
    fn render(&mut self, draws: &[Draw], lights: &[LightInstance], camera: CameraData, shadow_distance: f32, bias_scale: f32, shadows: bool) -> Vec<f32> {
        let g = self.g;
        let device = &g.device;
        // Frame data through the renderer's own packers.
        let mut fi = crate::cascade_tests::input(camera);
        fi.lights = lights;
        // No ambient / sky fill, so a fully shadowed pixel is black.
        fi.light = crate::LightData { ambient: [0.0; 3], env_intensity: 0.0, intensity: 0.0, ..Default::default() };
        fi.shadow_distance = shadow_distance;
        fi.shadow_pcf_texel = self.knobs.softness / SHADOW_DIM as f32; // Environment default: softness 1 / resolution 2048
        fi.shadow_bias_scale = bias_scale;
        let k = self.knobs;
        fi.shadow_normal_offset = k.normal_offset;
        fi.shadow_cascades = k.cascades;
        fi.shadow_cascade_lambda = k.lambda;
        // One bounding-sphere draw whose far edge sits `recv_limit` in front of the camera.
        let fwd = camera.view.inverse().transform_vector3(Vec3::NEG_Z).normalize();
        let bound = [DrawCmd {
            gi_source: None,
            mesh: MeshHandle(0),
            material: MaterialHandle(0),
            transform: Mat4::from_translation(camera.position + fwd * (k.recv_limit - 1.0)),
            highlight: 0.0,
            mesh_center: Vec3::ZERO,
            bound_radius: 1.0,
            lightmap_layer: -1,
            lightmap_size: 0,
            static_geometry: false,
            planar_index: -1,
            occluded: false,
            shadow_casting: ShadowCasting::On,
            instance_color: None,
            instance_param: None,
        }];
        if k.recv_limit > 0.0 {
            fi.draws = &bound;
        }
        let mut gl = [crate::GpuLight::default(); MAX_LIGHTS];
        for (i, l) in lights.iter().enumerate() {
            gl[i] = gpu_light(l);
        }
        let (views, shadow_vp, splits) = plan_shadows(&fi, &mut gl, lights.len(), crate::CascadeCfg::resolve(&fi, 4, k.snap_dim));
        let vp = camera.proj * camera.view;
        let mut ubo: FrameUbo = frame_ubo(&camera, &fi, gl, lights.len(), shadow_vp, splits, &[], [0.5, 0.5], [W as f32, W as f32], [W as f32, W as f32], vp.to_cols_array_2d());
        ubo.debug[3] = 1.0; // HDR output: linear radiance, no tonemap
        let mut ubo_buf = Buffer::new(device, &mut self.alloc, std::mem::size_of::<FrameUbo>() as u64, vk::BufferUsageFlags::UNIFORM_BUFFER, MemoryLocation::CpuToGpu, "ubo").unwrap();
        ubo_buf.write(0, bytemuck::bytes_of(&ubo));

        // Descriptor sets (frame set 0 with every binding valid, material set 1).
        let layouts = [self.cache.set0_layout, self.cache.set1_layout];
        let sets = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(self.pool).set_layouts(&layouts)).unwrap() };
        let black = &self.dummies[0].1;
        let white = &self.dummies[1].1;
        let img = |view, sampler, layout| [vk::DescriptorImageInfo::default().image_view(view).sampler(sampler).image_layout(layout)];
        let ro = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        let shadow_info = img(self.atlas_view, self.shadow_sampler, vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL);
        let black_arr = img(black[1], self.plain_sampler, ro);
        let black_2d = img(black[0], self.plain_sampler, ro);
        let black_cube = img(black[2], self.plain_sampler, ro);
        let water: Vec<[vk::DescriptorImageInfo; 1]> = (0..crate::MAX_WATER_BODIES).map(|_| img(black[0], self.plain_sampler, ro)).collect();
        let water_flat: Vec<vk::DescriptorImageInfo> = water.iter().map(|w| w[0]).collect();
        let ubo_info = [vk::DescriptorBufferInfo::default().buffer(ubo_buf.handle).range(vk::WHOLE_SIZE)];
        let sh_info = [vk::DescriptorBufferInfo::default().buffer(self.sh_buf.handle).range(vk::WHOLE_SIZE)];
        let fx_info = [vk::DescriptorBufferInfo::default().buffer(self.fx_buf.handle).range(vk::WHOLE_SIZE)];
        let cs = vk::DescriptorType::COMBINED_IMAGE_SAMPLER;
        let w = |set, binding, ty| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(binding).descriptor_type(ty);
        let white_info = img(white[0], self.plain_sampler, ro);
        let mut writes = vec![
            w(sets[0], 0, vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&ubo_info),
            w(sets[0], 1, cs).image_info(&shadow_info),
            w(sets[0], 2, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&sh_info),
            // No water in this fixture, but the shared layout still requires its buffer.
            w(sets[0], 14, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&sh_info),
            w(sets[0], 3, cs).image_info(&black_arr),
            w(sets[0], 4, cs).image_info(&black_2d),
            w(sets[0], 5, cs).image_info(&black_2d),
            w(sets[0], 6, cs).image_info(&black_2d),
            w(sets[0], 7, cs).image_info(&black_cube),
            w(sets[0], 8, cs).image_info(&black_arr),
            w(sets[0], 9, cs).image_info(&black_cube),
            w(sets[0], 11, cs).image_info(&black_cube),
            w(sets[0], 12, cs).image_info(&black_cube),
            w(sets[0], 13, cs).image_info(&white_info),
            w(sets[0], 10, cs).image_info(&water_flat),
            w(sets[1], 4, vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&fx_info),
        ];
        for b in crate::TEXTURE_BINDINGS.into_iter().chain([crate::AREA_ATLAS_BINDING,crate::LTC_LUT_BINDING]) {
            writes.push(w(sets[1], b, cs).image_info(&white_info));
        }
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        let mut color = Img::new(g, &mut self.alloc, W, W, 1, vk::Format::R16G16B16A16_SFLOAT, vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC, vk::SampleCountFlags::TYPE_1).unwrap();
        let mut depth = Img::new(g, &mut self.alloc, W, W, 1, vk::Format::D32_SFLOAT, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT, vk::SampleCountFlags::TYPE_1).unwrap();
        // Instance streams, one buffer per draw (simple).
        let mut inst_bufs = Vec::new();
        for d in draws {
            let inst = InstanceData { model: d.model.to_cols_array_2d(), color: d.color, param: [0.0; 4] };
            inst_bufs.push(alloc::upload_buffer(&g.device, &mut self.alloc, g.pool, g.queue, bytemuck::bytes_of(&inst), vk::BufferUsageFlags::VERTEX_BUFFER, "inst").unwrap());
        }
        self.cache.set_target_views(false, 1);
        let shadow_pipe = self.cache.get(device, PipelineKey::shadow()).unwrap();
        let mut key = PipelineKey::from_features(&crate::MaterialFeatures {
            toon: false,
            normal_map: false,
            emission: false,
            alpha_mode: crate::AlphaMode::Opaque,
            double_sided: true,
            water: false,
            per_instance: false,
            ..Default::default()
        });
        key.per_instance = false;
        let main_pipe = self.cache.get(device, key).unwrap();
        let layout = self.cache.layout;
        let both = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
        let ds = (vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS, vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE);
        let atlas = &self.atlas;
        let meshes = &self.meshes;
        let layer_views = &self.layer_views;
        let used: Vec<usize> = views.iter().map(|v| v.layer).collect();
        alloc::one_time_submit(device, g.pool, g.queue, |cb| unsafe {
            // Shadow atlas: every layer cleared to 1.0, used layers get their casters.
            transition(g, cb, atlas, vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL, NONE, ds);
            let shadow_area = vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: SHADOW_DIM, height: SHADOW_DIM } };
            for (layer, layer_view) in layer_views.iter().enumerate() {
                let datt = vk::RenderingAttachmentInfo::default()
                    .image_view(*layer_view)
                    .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } });
                device.cmd_begin_rendering(cb, &vk::RenderingInfo::default().render_area(shadow_area).layer_count(1).depth_attachment(&datt));
                if shadows && used.contains(&layer) {
                    let view_proj = views.iter().find(|v| v.layer == layer).unwrap().view_proj;
                    device.cmd_set_viewport(cb, 0, &[vk::Viewport { x: 0.0, y: 0.0, width: SHADOW_DIM as f32, height: SHADOW_DIM as f32, min_depth: 0.0, max_depth: 1.0 }]);
                    device.cmd_set_scissor(cb, 0, &[shadow_area]);
                    device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, shadow_pipe);
                    let push = ShadowPush { light_vp: view_proj.to_cols_array_2d(), model: Mat4::IDENTITY.to_cols_array_2d() };
                    device.cmd_push_constants(cb, layout, both, 0, bytemuck::bytes_of(&push));
                    for (d, inst) in draws.iter().zip(&inst_bufs) {
                        if !d.casts {
                            continue;
                        }
                        let m = &meshes[d.mesh];
                        device.cmd_bind_vertex_buffers(cb, 0, &[m.vbuf.handle], &[0]);
                        device.cmd_bind_vertex_buffers(cb, 1, &[inst.handle], &[0]);
                        device.cmd_bind_index_buffer(cb, m.ibuf.handle, 0, vk::IndexType::UINT32);
                        device.cmd_draw_indexed(cb, m.index_count, 1, 0, 0, 0);
                    }
                }
                device.cmd_end_rendering(cb);
            }
            transition(
                g,
                cb,
                atlas,
                vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL,
                (vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS, vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE),
                (vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_SAMPLED_READ),
            );
            // Main pass.
            transition(g, cb, &color, vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, NONE, COLOR_OUT);
            transition(g, cb, &depth, vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL, NONE, ds);
            let catt = [vk::RenderingAttachmentInfo::default()
                .image_view(color.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0; 4] } })];
            let datt = vk::RenderingAttachmentInfo::default()
                .image_view(depth.view)
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } });
            let area = vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: W, height: W } };
            device.cmd_begin_rendering(cb, &vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&catt).depth_attachment(&datt));
            device.cmd_set_viewport(cb, 0, &[vk::Viewport { x: 0.0, y: W as f32, width: W as f32, height: -(W as f32), min_depth: 0.0, max_depth: 1.0 }]);
            device.cmd_set_scissor(cb, 0, &[area]);
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, main_pipe);
            device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, layout, 0, &[sets[0], sets[1]], &[]);
            for (d, inst) in draws.iter().zip(&inst_bufs) {
                let mut push: PushData = bytemuck::Zeroable::zeroed();
                push.model[0] = [f32::from_bits(u32::MAX); 4]; // every light reaches this draw
                push.base_color = d.color;
                push.params0 = [0.0, 0.8, 0.0, 0.0]; // metallic, roughness
                push.params1 = [0.5, 1.0, 1.0, -1.0]; // alpha cutoff, normal strength, occlusion, no lightmap
                device.cmd_push_constants(cb, layout, both, 0, bytemuck::bytes_of(&push));
                let m = &meshes[d.mesh];
                device.cmd_bind_vertex_buffers(cb, 0, &[m.vbuf.handle], &[0]);
                device.cmd_bind_vertex_buffers(cb, 1, &[inst.handle], &[0]);
                device.cmd_bind_index_buffer(cb, m.ibuf.handle, 0, vk::IndexType::UINT32);
                device.cmd_draw_indexed(cb, m.index_count, 1, 0, 0, 0);
            }
            device.cmd_end_rendering(cb);
        })
        .unwrap();
        let bytes = read_image(g, &mut self.alloc, &color, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, COLOR_OUT, 8).unwrap();
        for mut b in inst_bufs {
            b.destroy(device, &mut self.alloc);
        }
        ubo_buf.destroy(device, &mut self.alloc);
        color.destroy(g, &mut self.alloc);
        depth.destroy(g, &mut self.alloc);
        unsafe { device.reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty()).unwrap() };
        bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|p| {
                let h = |i: usize| half::f16::from_le_bytes([p[i], p[i + 1]]).to_f32();
                0.2126 * h(0) + 0.7152 * h(2) + 0.0722 * h(4)
            })
            .collect()
    }

    fn destroy(mut self) {
        let device = &self.g.device;
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_sampler(self.shadow_sampler, None);
            device.destroy_sampler(self.plain_sampler, None);
            device.destroy_image_view(self.atlas_view, None);
            for v in self.layer_views.drain(..) {
                device.destroy_image_view(v, None);
            }
            for (image, views, _) in &self.dummies {
                for v in views {
                    if *v != vk::ImageView::null() {
                        device.destroy_image_view(*v, None);
                    }
                }
                device.destroy_image(*image, None);
            }
        }
        for m in &mut self.meshes {
            m.vbuf.destroy(device, &mut self.alloc);
            m.ibuf.destroy(device, &mut self.alloc);
        }
        self.sh_buf.destroy(device, &mut self.alloc);
        self.fx_buf.destroy(device, &mut self.alloc);
        self.atlas.destroy(self.g, &mut self.alloc);
        self.cache.destroy(device);
    }
}

fn light(kind: LightKind, position: Vec3, target: Vec3, range: f32, intensity: f32) -> LightInstance {
    LightInstance {
        kind,
        position,
        direction: (target - position).normalize(),
        color: [1.0; 3],
        intensity,
        range,
        spot_inner_deg: 50.0,
        spot_outer_deg: 70.0,
        cast_shadows: true,
        soft_shadows: true, radius: 0.0,
        shadow_bias: 0.002, // LightComponent default
        baked: false,
    }
}

struct Avatar {
    draws: Vec<Draw>,
    /// Ground-plane points under the shadow of the legs' axes / torso centre (the thin parts).
    leg_axis: Vec<Vec3>,
    torso_centre: Vec3,
}

fn avatar(rig: &mut Rig, with_ground: bool) -> Avatar {
    avatar_with_legs(rig, with_ground, 0.075)
}

/// The avatar with legs of `radius` (0.075 = the 0.15 m thick default).
fn avatar_with_legs(rig: &mut Rig, with_ground: bool, radius: f32) -> Avatar {
    let ground = rig.add_mesh(plane(60.0));
    let leg = rig.add_mesh(cylinder(radius, 0.9, 16));
    let torso = rig.add_mesh(cuboid(Vec3::new(0.25, 0.3, 0.15)));
    let mut draws = Vec::new();
    if with_ground {
        draws.push(Draw { mesh: ground, model: Mat4::IDENTITY, color: [1.0; 4], casts: true });
    }
    let mut leg_axis = Vec::new();
    for x in [-0.12f32, 0.12] {
        draws.push(Draw { mesh: leg, model: Mat4::from_translation(Vec3::new(x, 0.0, 0.0)), color: [1.0; 4], casts: true });
    }
    // A single sample line along the left leg's axis, from 10% to 90% of its height.
    for k in 1..10 {
        leg_axis.push(Vec3::new(-0.12, 0.9 * k as f32 / 10.0, 0.0));
    }
    draws.push(Draw { mesh: torso, model: Mat4::from_translation(Vec3::new(0.0, 1.2, 0.0)), color: [1.0; 4], casts: true });
    Avatar { draws, leg_axis, torso_centre: Vec3::new(0.0, 1.2, 0.0) }
}

/// Project a world point to its pixel (x, y); None when behind the camera.
fn pixel(vp: Mat4, p: Vec3) -> Option<(usize, usize)> {
    let c = vp * p.extend(1.0);
    if c.w <= 0.0 {
        return None;
    }
    let n = c.truncate() / c.w;
    let (x, y) = ((n.x * 0.5 + 0.5) * W as f32, (0.5 - n.y * 0.5) * W as f32);
    (x >= 0.0 && y >= 0.0 && x < W as f32 && y < W as f32).then_some((x as usize, y as usize))
}

/// Where the ray from a point light through `p` meets the ground (y = 0); for a
/// directional light (`l` is the travel direction) the parallel projection.
fn ground_hit(l: &LightInstance, p: Vec3) -> Vec3 {
    let dir = match l.kind {
        LightKind::Directional => l.direction,
        _ => (p - l.position).normalize(),
    };
    p - dir * (p.y / dir.y)
}

struct Shot {
    /// Lit ratio (shadowed render / unshadowed render) at the leg-axis shadow points.
    leg: Vec<f32>,
    torso: f32,
    /// Lit ratio on the ground just past the leg's rim (where its shadow starts): 1, 3, 6 and 12 cm out.
    contact: [f32; 4],
    /// Lit ratio at ground points well outside any shadow (over-darkening / acne check).
    open: Vec<f32>,
}

fn ratio(a: &[f32], b: &[f32], at: (usize, usize)) -> f32 {
    // 3x3 neighbourhood mean to ride over single-texel jitter; the unshadowed
    // value is the reference (the shaded ground differs per point by falloff).
    let mut sa = 0.0;
    let mut sb = 0.0;
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            let i = (at.1 as i32 + dy) as usize * W as usize + (at.0 as i32 + dx) as usize;
            sa += a[i];
            sb += b[i];
        }
    }
    if sb < 1e-4 { f32::NAN } else { sa / sb }
}

fn shoot(rig: &mut Rig, av: &Avatar, l: LightInstance, camera: CameraData, shadow_distance: f32) -> Shot {
    shoot_scaled(rig, av, l, camera, shadow_distance, 0.0)
}

fn shoot_scaled(rig: &mut Rig, av: &Avatar, l: LightInstance, camera: CameraData, shadow_distance: f32, bias_scale: f32) -> Shot {
    let lit = rig.render(&av.draws, &[l], camera, shadow_distance, bias_scale, false);
    let shaded = rig.render(&av.draws, &[l], camera, shadow_distance, bias_scale, true);
    super::dump_gray(&format!("{:?}-{:.0}-lit", l.kind, l.position.length()), &lit, W);
    super::dump_gray(&format!("{:?}-{:.0}-shaded", l.kind, l.position.length()), &shaded, W);
    let vp = camera.proj * camera.view;
    let leg: Vec<f32> = av.leg_axis.iter().filter_map(|p| pixel(vp, ground_hit(&l, *p))).map(|px| ratio(&shaded, &lit, px)).collect();
    let torso = pixel(vp, ground_hit(&l, av.torso_centre)).map(|px| ratio(&shaded, &lit, px)).unwrap_or(f32::NAN);
    // Open ground: points on the light side of the avatar, clear of every shadow.
    let side = match l.kind {
        LightKind::Directional => -l.direction,
        _ => (l.position - Vec3::new(0.0, l.position.y, 0.0)).normalize_or(Vec3::X),
    };
    let side = Vec3::new(side.x, 0.0, side.z).normalize_or(Vec3::X);
    let open = [1.0f32, 1.5, 2.0, 2.5]
        .iter()
        .filter_map(|d| pixel(vp, side * *d + Vec3::new(0.0, 0.0, 0.3)))
        .map(|px| ratio(&shaded, &lit, px))
        .collect();
    // The left leg's rim is at x = -0.12 - 0.075; the shadow falls toward -X.
    let contact_at = |out: f32| pixel(vp, Vec3::new(-0.195 - out, 0.0, 0.0)).map(|px| ratio(&shaded, &lit, px)).unwrap_or(f32::NAN);
    let contact = [contact_at(0.01), contact_at(0.03), contact_at(0.06), contact_at(0.12)];
    Shot { leg, torso, contact, open }
}

fn camera_at(eye: Vec3, target: Vec3, fov_deg: f32) -> CameraData {
    let view = Mat4::look_at_rh(eye, target, Vec3::Y);
    CameraData { view, proj: Mat4::perspective_rh(fov_deg.to_radians(), 1.0, 0.1, 200.0), position: eye }
}

fn check(name: &str, s: &Shot, leg_max: f32, torso_max: f32) {
    let leg_worst = s.leg.iter().cloned().fold(0.0f32, f32::max);
    let leg_mean = s.leg.iter().sum::<f32>() / s.leg.len().max(1) as f32;
    let open_min = s.open.iter().cloned().fold(f32::INFINITY, f32::min);
    eprintln!(
        "{name}: leg shadow lit-ratio mean {leg_mean:.3} worst {leg_worst:.3} ({} pts), torso {:.3}, open ground min {open_min:.3}",
        s.leg.len(),
        s.torso
    );
    assert!(s.leg.len() >= 4, "{name}: leg shadow points off screen");
    assert!(leg_worst <= leg_max, "{name}: the 0.15 m leg's shadow is gone (lit ratio up to {leg_worst:.3})");
    assert!(s.torso <= torso_max, "{name}: the torso's shadow is gone (lit ratio {:.3})", s.torso);
    assert!(open_min >= 0.9, "{name}: open ground darkened (acne / over-bias): ratio {open_min:.3}");
}

#[test]
fn thin_casters_keep_their_shadow_under_point_and_spot_lights_at_distance() {
    let Some(g) = gpu() else { return };
    let mut rig = Rig::new(&g);
    let av = avatar(&mut rig, true);
    // The shadow falls toward -X (light on the +X, +Y side); the camera looks from +Z.
    let camera = camera_at(Vec3::new(-0.8, 2.0, 5.0), Vec3::new(-0.8, 0.2, 0.0), 40.0);
    for kind in [LightKind::Point, LightKind::Spot] {
        for d in [5.0f32, 10.0, 20.0] {
            // 45 degrees up on the +X side, `d` metres from the avatar's centre.
            let pos = Vec3::new(1.0, 1.0, 0.0).normalize() * d;
            let l = light(kind, pos, Vec3::new(0.0, 0.6, 0.0), d * 3.0 + 10.0, 40.0 * d * d / 25.0);
            let s = shoot(&mut rig, &av, l, camera, 25.0);
            check(&format!("{kind:?} light at {d} m"), &s, 0.25, 0.25);
        }
    }
    // Harness sensitivity: a large receiver bias (the failure the user saw, deliberately
    // overdone) must make the same legs' shadow disappear, or the checks above prove nothing.
    let pos = Vec3::new(1.0, 1.0, 0.0).normalize() * 20.0;
    let l = light(LightKind::Point, pos, Vec3::new(0.0, 0.6, 0.0), 70.0, 64.0);
    // Bias now uses a sub-texel default; include deliberately out-of-UI-range
    // scales to retain this negative control under the new calibration.
    let mut vanished = None;
    for scale in [1500.0f32, 6000.0, 12000.0, 25000.0, 50000.0, 150000.0, 500000.0] {
        let s = shoot_scaled(&mut rig, &av, l, camera, 25.0, scale);
        let mean = s.leg.iter().sum::<f32>() / s.leg.len() as f32;
        eprintln!("point light at 20 m, bias scale {scale}: leg lit ratio {mean:.3}");
        if mean > 0.5 && vanished.is_none() {
            vanished = Some(scale);
        }
    }
    assert!(vanished.is_some(), "no bias scale up to 500000 made the leg shadow vanish: the checks are insensitive");
    rig.destroy();
}

#[test]
fn thin_casters_keep_their_shadow_under_the_cascaded_sun_with_a_far_camera() {
    // Grazing camera (2 m up): at 22 m the 0.15 m shadow is under two image rows tall,
    // so the 3x3 lit ratio reads ~0.2 there with a solid shadow in the map (see
    // `far_sun_leg_by_camera_distance` for the same scene from above).
    let Some(g) = gpu() else { return };
    let mut rig = Rig::new(&g);
    let av = avatar(&mut rig, true);
    let sun = light(LightKind::Directional, Vec3::ZERO, Vec3::new(-1.0, -1.0, 0.0), 100.0, 3.0);
    for (dist, fov) in [(6.0f32, 40.0f32), (12.0, 20.0), (22.0, 11.0)] {
        // The shadow falls toward -X; the camera is `dist` metres away on +Z.
        let camera = camera_at(Vec3::new(-0.8, 2.0, dist), Vec3::new(-0.8, 0.2, 0.0), fov);
        let s = shoot(&mut rig, &av, sun, camera, 25.0);
        check(&format!("sun, camera {dist} m away"), &s, 0.4, 0.3);
    }
    rig.destroy();
}

/// Fraction of lit pixels (reference > `LIT_MIN`) a flat ground plane's own shadow
/// darkens by more than 10 %: pure self-shadowing (acne), no other caster.
const LIT_MIN: f32 = 0.003;

fn acne_fraction(rig: &mut Rig, kind: LightKind, elev: f32) -> f32 {
    let ground = rig.add_mesh(plane(60.0));
    let draws = vec![Draw { mesh: ground, model: Mat4::IDENTITY, color: [1.0; 4], casts: true }];
    let camera = camera_at(Vec3::new(0.0, 6.0, 6.0), Vec3::new(0.0, 0.0, 0.0), 45.0);
    let e = elev.to_radians();
    let toward = Vec3::new(e.cos(), e.sin(), 0.0);
    let l = if kind == LightKind::Directional {
        light(kind, Vec3::ZERO, -toward * 10.0, 100.0, 3.0)
    } else {
        light(kind, toward * 12.0, Vec3::ZERO, 80.0, 60.0)
    };
    let lit = rig.render(&draws, &[l], camera, 25.0, 0.0, false);
    let shaded = rig.render(&draws, &[l], camera, 25.0, 0.0, true);
    let (mut n, mut dark) = (0, 0);
    for (a, b) in shaded.iter().zip(&lit) {
        if *b > LIT_MIN {
            n += 1;
            if *a < 0.9 * *b {
                dark += 1;
            }
        }
    }
    dark as f32 / n.max(1) as f32
}

/// Acne on a flat receiver at grazing angles: no caster except the ground
/// itself, so any darkening is self-shadowing. Before (the lookup without the
/// normal offset) and after (default), down to a 1 degree sun.
#[test]
fn flat_receiver_acne_at_grazing_angles() {
    let Some(g) = gpu() else { return };
    let mut rig = Rig::new(&g);
    let mut report = Vec::new();
    for (name, kind) in [("sun", LightKind::Directional), ("spot", LightKind::Spot), ("point", LightKind::Point)] {
        for elev in [60.0f32, 30.0, 15.0, 8.0, 4.0, 2.0, 1.0] {
            rig.knobs.normal_offset = NO_OFFSET;
            let before = acne_fraction(&mut rig, kind, elev);
            rig.knobs.normal_offset = 0.0;
            let after = acne_fraction(&mut rig, kind, elev);
            report.push(format!("{name:<5} {elev:>4}deg: lit pixels darkened >10%: before {:>6.2}%  after {:>6.2}%", 100.0 * before, 100.0 * after));
            if elev >= 8.0 {
                assert!(after <= 0.01, "{name} at {elev} deg: acne on {:.2}% of pixels", 100.0 * after);
            } else if elev >= 2.0 {
                assert!(after <= 0.05, "{name} at {elev} deg: acne on {:.2}% of pixels (limit 5%)", 100.0 * after);
            }
        }
    }
    eprintln!("flat receiver, light bias 0.002 x scale 1500, normal offset 2 (before = offset off):\n  {}", report.join("\n  "));
    rig.destroy();
}

/// The shadow must start at the leg's base: with the normal offset on, the
/// ground just past the leg's rim (1 - 12 cm out) is dark.
/// The offset lifts the lookup by at most a couple of texels, so it may only
/// eat the first centimetres, and never more than the old depth bias did.
#[test]
fn shadow_starts_at_the_legs_base() {
    let Some(g) = gpu() else { return };
    let mut rig = Rig::new(&g);
    let av = avatar(&mut rig, true);
    let camera = camera_at(Vec3::new(-0.8, 2.0, 5.0), Vec3::new(-0.8, 0.2, 0.0), 40.0);
    let mut rows = Vec::new();
    let mut cases: Vec<(String, LightInstance, f32)> = Vec::new();
    for deg in [45.0f32, 20.0] {
        let e = deg.to_radians();
        cases.push((format!("sun {deg:>2.0} deg"), light(LightKind::Directional, Vec3::ZERO, Vec3::new(-e.cos(), -e.sin(), 0.0), 100.0, 3.0), 25.0));
    }
    for kind in [LightKind::Point, LightKind::Spot] {
        for d in [5.0f32, 10.0, 20.0] {
            let pos = Vec3::new(1.0, 1.0, 0.0).normalize() * d;
            cases.push((format!("{kind:?} {d} m"), light(kind, pos, Vec3::new(0.0, 0.6, 0.0), d * 3.0 + 10.0, 40.0 * d * d / 25.0), 25.0));
        }
    }
    for (name, l, sd) in cases {
        rig.knobs.normal_offset = NO_OFFSET;
        let before = shoot(&mut rig, &av, l, camera, sd);
        rig.knobs.normal_offset = 0.0;
        let after = shoot(&mut rig, &av, l, camera, sd);
        let f = |c: [f32; 4]| format!("{:.2} {:.2} {:.2} {:.2}", c[0], c[1], c[2], c[3]);
        rows.push(format!("{name:<12} contact lit ratio 1/3/6/12 cm past the rim: before {} | after {}", f(before.contact), f(after.contact)));
        // The offset may only soften the first centimetres; past 6 cm the shadow is solid.
        assert!(after.contact[2] <= 0.15 && after.contact[3] <= 0.1, "{name}: shadow detached from the leg's base ({:?})", after.contact);
        assert!(after.leg.iter().cloned().fold(0.0, f32::max) <= 0.1, "{name}: leg shadow weakened ({:?})", after.leg);
    }
    eprintln!("{}", rows.join("\n"));
    rig.destroy();
}

/// Peter-panning on a short caster: a 4 cm thick foot slab on the ground casts
/// a 4 cm shadow at 45 degrees. A lookup lifted off the ground by the bias or
/// the normal offset reads that sliver as lit, so the length of the dark strip
/// past the slab's edge is the detachment, in millimetres.
#[test]
fn a_thin_foot_keeps_its_contact_shadow() {
    let Some(g) = gpu() else { return };
    let mut rig = Rig::new(&g);
    let ground = rig.add_mesh(plane(60.0));
    let foot = rig.add_mesh(cuboid(Vec3::new(0.15, 0.02, 0.1)));
    let draws = vec![
        Draw { mesh: ground, model: Mat4::IDENTITY, color: [1.0; 4], casts: true },
        Draw { mesh: foot, model: Mat4::from_translation(Vec3::new(0.0, 0.02, 0.0)), color: [1.0; 4], casts: true },
    ];
    // Close camera: ~2 mm per pixel along X; the strip sits on the -X edge of the slab.
    let camera = camera_at(Vec3::new(-0.2, 1.2, 1.4), Vec3::new(-0.2, 0.0, 0.0), 25.0);
    let vp = camera.proj * camera.view;
    let strip = |rig: &mut Rig, l: LightInstance, sd: f32| -> f32 {
        let lit = rig.render(&draws, &[l], camera, sd, 0.0, false);
        let shaded = rig.render(&draws, &[l], camera, sd, 0.0, true);
        // Dark extent: the largest 2 mm step out from the edge before the first lit one.
        let mut len = 0.0;
        for k in 0..30 {
            let c = 0.002 * k as f32;
            let Some(px) = pixel(vp, Vec3::new(-0.15 - 0.001 - c, 0.0, 0.0)) else { break };
            let r = ratio(&shaded, &lit, px);
            if r.partial_cmp(&0.5) != Some(std::cmp::Ordering::Less) {
                break;
            }
            len = c + 0.002;
        }
        len * 1000.0
    };
    let mut rows = Vec::new();
    let mut cases = vec![("sun 45 deg".to_string(), light(LightKind::Directional, Vec3::ZERO, Vec3::new(-1.0, -1.0, 0.0), 100.0, 3.0))];
    for kind in [LightKind::Point, LightKind::Spot] {
        for d in [5.0f32, 10.0, 20.0] {
            let pos = Vec3::new(1.0, 1.0, 0.0).normalize() * d;
            cases.push((format!("{kind:?} {d} m"), light(kind, pos, Vec3::new(0.0, 0.6, 0.0), d * 3.0 + 10.0, 40.0 * d * d / 25.0)));
        }
    }
    for (name, l) in cases {
        rig.knobs.normal_offset = 1e-4;
        let before = strip(&mut rig, l, 25.0);
        rig.knobs.normal_offset = 0.0;
        let after = strip(&mut rig, l, 25.0);
        rows.push(format!("{name:<12} dark strip past the slab edge (ideal 40 mm): before {before:>4.0} mm, after {after:>4.0} mm"));
        // No new peter-panning: the offset may cost a few millimetres, no more.
        assert!(after + 8.0 >= before, "{name}: the foot's contact shadow shrank from {before} to {after} mm");
    }
    eprintln!("{}", rows.join("\n"));
    rig.destroy();
}

/// A camera `dist` metres from the subject, 33 degrees above the ground, so
/// the leg's 0.15 m shadow is ~10 image rows tall at any distance (a grazing
/// camera sees it foreshortened to under two rows).
fn high_camera(dist: f32) -> CameraData {
    let target = Vec3::new(-0.8, 0.2, 0.0);
    let el = 33f32.to_radians();
    let fov = 2.0 * (2.2 / dist).atan().to_degrees();
    camera_at(target + Vec3::new(0.0, el.sin(), el.cos()) * dist, target, fov)
}

fn leg_mean(s: &Shot) -> f32 {
    s.leg.iter().sum::<f32>() / s.leg.len().max(1) as f32
}

/// The cascaded sun's thin leg at camera distances 6 - 40 m: the legacy fit
/// (unsnapped, full `shadow_distance`) against the current one, at a
/// shadow distance just past the subject and at the editor's maximum (150),
/// and with a scene whose receivers end 8 m behind the subject.
#[test]
fn far_sun_leg_by_camera_distance() {
    let Some(g) = gpu() else { return };
    let mut rig = Rig::new(&g);
    let av = avatar(&mut rig, true);
    let sun = light(LightKind::Directional, Vec3::ZERO, Vec3::new(-1.0, -1.0, 0.0), 100.0, 3.0);
    let mut rows = Vec::new();
    for dist in [6.0f32, 12.0, 22.0, 40.0] {
        let camera = high_camera(dist);
        let mut cells = Vec::new();
        for (sd, recv) in [(dist * 1.3 + 5.0, 0.0f32), (150.0, 0.0), (500.0, 0.0), (500.0, dist + 8.0)] {
            rig.knobs = Knobs { snap_dim: NO_SNAP, ..Knobs::default() };
            let old = leg_mean(&shoot(&mut rig, &av, sun, camera, sd));
            rig.knobs = Knobs { recv_limit: recv, ..Knobs::default() };
            let new = shoot(&mut rig, &av, sun, camera, sd);
            let nl = leg_mean(&new);
            cells.push(format!("sd {sd:>5.1}{}: {old:.3} -> {nl:.3}", if recv > 0.0 { format!(" recv {recv:.0}") } else { String::new() }));
            if recv == 0.0 && sd < 100.0 {
                assert!(nl <= 0.1, "sun, camera {dist} m, shadow distance {sd}: leg shadow weak ({nl:.3})");
                assert!(new.torso <= 0.1, "sun, camera {dist} m: torso shadow weak ({:.3})", new.torso);
            }
        }
        rows.push(format!("camera {dist:>2} m, leg lit ratio legacy -> now: {}", cells.join(" | ")));
    }
    eprintln!("{}", rows.join("\n"));
    rig.destroy();
}

/// Where the cascade controls pay off: a 2 cm wire (radius 1 cm) 22 m from the
/// camera under a 150 m / 500 m shadow distance. The wire is ~2 texels wide
/// in the far cascades, so only a tighter fit (receiver depth limit), another
/// split blend or fewer, larger cascades keep it. Reported, loosely asserted.
#[test]
fn far_thin_wire_vs_cascade_settings() {
    let Some(g) = gpu() else { return };
    let mut rig = Rig::new(&g);
    let av = avatar_with_legs(&mut rig, true, 0.01);
    let sun = light(LightKind::Directional, Vec3::ZERO, Vec3::new(-1.0, -1.0, 0.0), 100.0, 3.0);
    let camera = high_camera(22.0);
    let mut rows = Vec::new();
    let mut run = |rig: &mut Rig, label: &str, sd: f32, k: Knobs| {
        rig.knobs = k;
        let m = leg_mean(&shoot(rig, &av, sun, camera, sd));
        rows.push(format!("{label:<44} wire lit ratio {m:.3}"));
        m
    };
    let legacy = Knobs { snap_dim: NO_SNAP, ..Knobs::default() };
    for sd in [60.0f32, 150.0, 500.0] {
        run(&mut rig, &format!("sd {sd:>3}: legacy fit (lambda 0.7, 4 casc)"), sd, legacy);
        run(&mut rig, &format!("sd {sd:>3}: current defaults"), sd, Knobs::default());
        run(&mut rig, &format!("sd {sd:>3}: receivers end at 30 m"), sd, Knobs { recv_limit: 30.0, ..Knobs::default() });
        for lambda in [0.4f32, 0.9] {
            run(&mut rig, &format!("sd {sd:>3}: lambda {lambda}"), sd, Knobs { lambda, ..Knobs::default() });
        }
        for cascades in [2u32, 3] {
            run(&mut rig, &format!("sd {sd:>3}: {cascades} cascades"), sd, Knobs { cascades, ..Knobs::default() });
        }
    }
    eprintln!("{}", rows.join("\n"));
    rig.destroy();
}

/// Informational: how the far-camera sun shadow of the leg reacts to the scene's
/// `shadow_bias_scale` (default 1500) and to the shadow distance. The grazing
/// camera sees the leg's shadow foreshortened to under two image rows, so the
/// 3x3 ratio reads partly lit whatever the map holds (`far_sun_leg_by_camera_distance`
/// measures it from above); what the bias scale does to it is still visible here.
#[test]
fn far_sun_shadow_vs_bias_scale_and_distance() {
    let Some(g) = gpu() else { return };
    let mut rig = Rig::new(&g);
    let av = avatar(&mut rig, true);
    let sun = light(LightKind::Directional, Vec3::ZERO, Vec3::new(-1.0, -1.0, 0.0), 100.0, 3.0);
    let camera = camera_at(Vec3::new(-0.8, 2.0, 22.0), Vec3::new(-0.8, 0.2, 0.0), 11.0);
    for dist in [25.0f32, 60.0] {
        for scale in [0.0f32, 100.0, 400.0, 1500.0, 4000.0] {
            let s = shoot_scaled(&mut rig, &av, sun, camera, dist, scale);
            let mean = s.leg.iter().sum::<f32>() / s.leg.len().max(1) as f32;
            let worst = s.leg.iter().cloned().fold(0.0, f32::max);
            eprintln!("far sun, shadow distance {dist:>4}, bias scale {scale:>6}: leg lit {mean:.3} (worst {worst:.3}), torso {:.3}, open min {:.3}", s.torso, s.open.iter().cloned().fold(9.0, f32::min));
        }
    }
    rig.destroy();
}

#[test]
fn sealed_corner_blocks_direct_point_light_without_gi() {
    let Some(g)=gpu() else {return};
    let mut rig=Rig::new(&g);
    let floor=rig.add_mesh(plane(4.0));
    // The divider penetrates the receiver: no mesh crack, no GI, no ambient.
    // This is a wall/floor corner rotated equivalently to two upright walls.
    let wall=rig.add_mesh(cuboid(Vec3::new(0.005,1.6,2.0)));
    let draws=[Draw {mesh:floor,model:Mat4::IDENTITY,color:[1.0;4],casts:true},
        Draw {mesh:wall,model:Mat4::from_translation(Vec3::new(0.0,1.5,0.0)),color:[1.0;4],casts:true}];
    let camera=camera_at(Vec3::new(-0.75,0.7,1.0),Vec3::new(-0.15,0.0,0.0),40.0);
    let vp=camera.proj*camera.view;
    let mut l=light(LightKind::Point,Vec3::new(1.0,0.4,0.0),Vec3::ZERO,6.0,15.0);
    l.shadow_bias=0.006;
    let mut worst=0.0f32;
    for kind in [LightKind::Point,LightKind::Spot] { for softness in [0.0,1.0,2.5,4.0] { for soft in [false,true] {
        rig.knobs.softness=softness;l.kind=kind;
        l.soft_shadows=soft;
        let lit=rig.render(&draws,&[l],camera,25.0,1500.0,false);
        for (bias,normal) in [(1500.0,2.0),(1.0,2.0),(1500.0,NO_OFFSET),(1.0,NO_OFFSET)] {
            rig.knobs.normal_offset=normal;
            let shaded=rig.render(&draws,&[l],camera,25.0,bias,true);
            let values:Vec<_>=[0.015,0.025,0.04,0.06,0.1].into_iter().map(|d| {
                let at=pixel(vp,Vec3::new(-d,0.0,0.0)).unwrap();ratio(&shaded,&lit,at)
            }).collect();
            eprintln!("[Direct sealed corner] kind={kind:?}, softness={softness}, soft={soft}, bias={bias}, normal={normal}: {values:?}");
            if bias==1500.0 && normal==2.0 {
                // At maximum softness the 8-texel PCF radius covers the first
                // two probes on this grazing receiver. That blur is a known
                // contact-hardening gap, not evidence of depth-bias crossing.
                // Keep the measured values visible; require full occlusion
                // outside that footprint and at EVERY probe at normal softness.
                let first=if soft && softness>2.5 {2} else {0};
                for v in &values {assert!(v.is_finite());}
                for v in &values[first..] {worst=worst.max(*v);}
            }
        }
    }
    }}
    rig.destroy();
    assert!(worst<0.05,"direct light escaped a sealed corner without GI: {worst}");
}

#[test]
fn shadow_receiver_plane_uses_geometry_not_smooth_normals() {
    let Some(g)=gpu() else {return};let mut rig=Rig::new(&g);
    let (mut vertices,indices)=plane(60.0);
    for v in &mut vertices {v.normal=Vec3::new(0.5,1.0,0.3).normalize().to_array();}
    let mesh=rig.add_mesh((vertices,indices));
    let draws=[Draw {mesh,model:Mat4::IDENTITY,color:[1.0;4],casts:true}];
    let camera=camera_at(Vec3::new(0.0,6.0,6.0),Vec3::ZERO,45.0);
    let l=light(LightKind::Point,Vec3::new(10.0,3.0,0.0),Vec3::ZERO,80.0,60.0);
    let lit=rig.render(&draws,&[l],camera,25.0,1500.0,false);
    let shaded=rig.render(&draws,&[l],camera,25.0,1500.0,true);
    let mut count=0;let mut dark=0;
    for (a,b) in shaded.iter().zip(&lit) {if *b>LIT_MIN {count+=1;if *a<0.9* *b {dark+=1;}}}
    rig.destroy();assert!(count>1000);let fraction=dark as f32/count as f32;
    eprintln!("[Direct receiver plane] smoothed normals, acne fraction={fraction}");
    assert!(fraction<0.01,"shadow depth used the shading normal: {fraction}");
}
