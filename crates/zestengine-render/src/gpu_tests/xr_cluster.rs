//! The XR paths of a REAL `Renderer` with the cluster path on, without a headset:
//! the frames render into images we own.
//!
//!  * the multiview stereo pass (`render_xr_stereo`, both eyes in one pass into a
//!    2-layer image): ONE cull for both eyes (centre-eye cut, union frustum),
//!  * the per-eye pass (`render_xr_eye`, one eye per call, no multiview): each
//!    eye is an ordinary camera with its own cut,
//!  * (GEO-5 leftover) Hi-Z in the stereo pass: a wall in front of the clustered
//!    quads occludes some of them; a cluster is culled only when it is occluded
//!    in BOTH eyes' pyramids.
//!
//! The clustered images are compared with the ordinary draw of the same mesh
//! (silhouette IoU) and, for Hi-Z, with the clustered image without Hi-Z (must
//! be identical). The pictures go to `docs/design/` when
//! `ZESTENGINE_XR_SHEET` names a file (the test always writes the default).

use anyhow::Result;
use ash::vk;
use glam::{Mat4, Vec3};

use crate::cluster::tests::quad_grid;
use crate::{CameraData, ClusterMode, DrawCmd, MaterialDesc, MaterialHandle, MeshData, MeshHandle, Renderer, Vertex, XrEyeCamera};

pub(super) const W: u32 = 256;
pub(super) const H: u32 = 256;
pub(super) const FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

/// An image we own, rendered into by the XR code.
pub(super) struct Target {
    image: vk::Image,
    memory: Option<gpu_allocator::vulkan::Allocation>,
    layers: u32,
}

impl Target {
    pub(super) fn new(r: &Renderer, layers: u32) -> Result<Self> {
        let device = &r.ctx.device;
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(FORMAT)
            .extent(vk::Extent3D { width: W, height: H, depth: 1 })
            .mip_levels(1)
            .array_layers(layers)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.create_image(&info, None)? };
        let req = unsafe { device.get_image_memory_requirements(image) };
        let memory = r.allocator().lock().unwrap().allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "xr test image",
            requirements: req,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        })?;
        unsafe { device.bind_image_memory(image, memory.memory(), memory.offset())? };
        Ok(Self { image, memory: Some(memory), layers })
    }

    pub(super) fn raw(&self) -> u64 {
        use ash::vk::Handle as _;
        self.image.as_raw()
    }

    /// Read every layer back (the post pass leaves them in COLOR_ATTACHMENT_OPTIMAL).
    pub(super) fn read(&self, r: &Renderer) -> Result<Vec<u8>> {
        let device = r.ctx.device.clone();
        unsafe { device.device_wait_idle()? };
        let allocator = r.allocator();
        let bytes = (W * H * 4 * self.layers) as u64;
        let mut host = crate::alloc::Buffer::new(&device, &mut allocator.lock().unwrap(), bytes, vk::BufferUsageFlags::TRANSFER_DST, gpu_allocator::MemoryLocation::GpuToCpu, "xr readback")?;
        let layers = self.layers;
        crate::alloc::one_time_submit(&device, r.command_pool, r.ctx.queue, |cb| unsafe {
            crate::image_barrier_layers(
                &device, cb, self.image, vk::ImageAspectFlags::COLOR, 0, layers,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::PipelineStageFlags2::ALL_COMMANDS, vk::AccessFlags2::MEMORY_WRITE,
                vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_READ,
            );
            let c = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(layers))
                .image_extent(vk::Extent3D { width: W, height: H, depth: 1 });
            device.cmd_copy_image_to_buffer(cb, self.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, host.handle, &[c]);
            crate::image_barrier_layers(
                &device, cb, self.image, vk::ImageAspectFlags::COLOR, 0, layers,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::empty(),
                vk::PipelineStageFlags2::ALL_COMMANDS, vk::AccessFlags2::empty(),
            );
        })?;
        let px = host.mapped().map(|m| m[..bytes as usize].to_vec()).unwrap_or_default();
        host.destroy(&device, &mut allocator.lock().unwrap());
        Ok(px)
    }

    pub(super) fn destroy(mut self, r: &Renderer) {
        unsafe { r.ctx.device.destroy_image(self.image, None) };
        if let Some(m) = self.memory.take() {
            let _ = r.allocator().lock().unwrap().free(m);
        }
    }
}

/// The test scene: 32 one-quad clusters in a grid, optionally with an ordinary
/// wall in front of the left part of it.
pub(super) struct Scene {
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
    wall: MeshHandle,
    wall_material: MaterialHandle,
}

fn vtx(p: [f32; 3]) -> Vertex {
    Vertex { position: p, normal: [0.0, 0.0, 1.0], tangent: [1.0, 0.0, 0.0, 1.0], color: [1.0; 4], ..Default::default() }
}

impl Scene {
    pub(super) fn new(r: &mut Renderer) -> Result<Self> {
        // Quads exactly where the DAG puts them (32 clusters of one quad each).
        let up = quad_grid(32);
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for k in 0..32usize {
            let (x0, y0) = ((k % 8) as f32 * 0.5 - 2.0, (k / 8) as f32 * 0.5 - 2.0);
            let base = vertices.len() as u32;
            for (dx, dy) in [(0.0f32, 0.0f32), (0.4, 0.0), (0.4, 0.4), (0.0, 0.4)] {
                vertices.push(vtx([x0 + dx, y0 + dy, 0.0]));
            }
            indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        let mesh = r.upload_mesh(&MeshData { vertices, indices, has_lightmap_uv: false })?;
        let material = r.create_material(&MaterialDesc {
            name: "xr cluster".into(),
            params: crate::MaterialParams { base_color: [0.9, 0.5, 0.3, 1.0], ..Default::default() },
            ..Default::default()
        })?;
        r.set_cluster_settings(crate::ClusterSettings { mode: ClusterMode::On, ..r.cluster_settings() });
        r.install_cluster_dag(mesh, &up)?;
        let wall = r.upload_mesh(&MeshData {
            vertices: [[-1.0f32, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]].iter().map(|p| vtx([p[0], p[1], 0.0])).collect(),
            indices: vec![0, 1, 2, 0, 2, 3],
            has_lightmap_uv: false,
        })?;
        let wall_material = r.create_material(&MaterialDesc {
            name: "xr wall".into(),
            params: crate::MaterialParams { base_color: [0.25, 0.3, 0.9, 1.0], ..Default::default() },
            ..Default::default()
        })?;
        Ok(Self { mesh, material, wall, wall_material })
    }

    pub(super) fn draws(&self, wall: bool) -> Vec<DrawCmd> {
        let draw = |mesh, material, transform, center, radius| DrawCmd {
            gi_source: None,
            mesh,
            material,
            transform,
            highlight: 0.0,
            mesh_center: center,
            bound_radius: radius,
            lightmap_layer: -1,
            lightmap_size: 0,
            static_geometry: false,
            planar_index: -1,
            occluded: false,
            shadow_casting: crate::ShadowCasting::Off,
            instance_color: None,
            instance_param: None,
        };
        let mut v = vec![draw(self.mesh, self.material, Mat4::from_translation(Vec3::new(-1.0, -0.5, -4.0)), Vec3::ZERO, 5.0)];
        if wall {
            // A wall 25 cm in front of the eyes covering the left part of the grid. So close that
            // the eyes' parallax at the grid (4 m away) is about 1 m: the clusters near the wall's
            // edge are hidden in one eye and visible in the other, those further left in both.
            v.push(draw(self.wall_material_mesh(), self.wall_material, Mat4::from_translation(Vec3::new(-0.74, 0.0, -0.25)) * Mat4::from_scale(Vec3::new(0.8, 2.0, 1.0)), Vec3::ZERO, 2.5));
        }
        v
    }

    fn wall_material_mesh(&self) -> MeshHandle {
        self.wall
    }

    /// The ordinary wall's mesh.
    pub(super) fn wall_mesh(&self) -> MeshHandle {
        self.wall
    }
}

/// Eyes 64 mm apart looking down -Z, asymmetric projections.
pub(super) fn eyes() -> ([XrEyeCamera; 2], (Mat4, Mat4)) {
    let proj = |l: f32, rr: f32| {
        let (u, d, n, f) = (0.9f32, -0.9f32, 0.05f32, 100.0f32);
        let (w, h) = (rr - l, u - d);
        Mat4::from_cols_array(&[2.0 / w, 0.0, 0.0, 0.0, 0.0, 2.0 / h, 0.0, 0.0, (rr + l) / w, (u + d) / h, -f / (f - n), -1.0, 0.0, 0.0, -(f * n) / (f - n), 0.0])
    };
    let eyes = [
        XrEyeCamera { view: Mat4::from_translation(Vec3::new(0.032, 0.0, 0.0)), proj: proj(-1.0, 0.8), postfx: crate::PostFx::default() },
        XrEyeCamera { view: Mat4::from_translation(Vec3::new(-0.032, 0.0, 0.0)), proj: proj(-0.8, 1.0), postfx: crate::PostFx::default() },
    ];
    (eyes, (Mat4::IDENTITY, proj(-1.1, 1.1)))
}

pub(super) fn frame_input<'a>(draws: &'a [DrawCmd], cull_proj: Mat4) -> crate::FrameInput<'a> {
    let mut input = crate::cascade_tests::input(CameraData { view: Mat4::IDENTITY, proj: cull_proj, position: Vec3::ZERO });
    input.render_viewport = false;
    input.draws = draws;
    input.clear_color = [0.0, 0.0, 0.0, 1.0];
    input
}

pub(super) fn set_mode(r: &mut Renderer, on: bool, hiz: bool) {
    r.set_cluster_settings(crate::ClusterSettings { mode: if on { ClusterMode::On } else { ClusterMode::Off }, ..r.cluster_settings() });
    r.set_gpu_cull_settings(crate::GpuCullSettings { hiz_occlusion: hiz, ..r.gpu_cull_settings() });
}

/// One multiview stereo frame into `target` (2 layers).
pub(super) fn stereo_frame(r: &mut Renderer, scene: &Scene, target: &Target, wall: bool, on: bool, hiz: bool) -> Result<Vec<u8>> {
    let (eyes, cull) = eyes();
    set_mode(r, on, hiz);
    let draws = scene.draws(wall);
    let input = frame_input(&draws, cull.1);
    r.render_xr_frame(&input)?;
    r.render_xr_stereo(target.raw(), FORMAT.as_raw() as u32, W, H, &eyes, cull, &input)?;
    target.read(r)
}

/// Both eyes, one `render_xr_eye` call each into its own 1-layer image.
pub(super) fn per_eye_frame(r: &mut Renderer, scene: &Scene, targets: &[Target; 2], wall: bool, on: bool) -> Result<[Vec<u8>; 2]> {
    let (eyes, cull) = eyes();
    set_mode(r, on, false);
    let draws = scene.draws(wall);
    let input = frame_input(&draws, cull.1);
    r.render_xr_frame(&input)?;
    for e in 0..2 {
        r.render_xr_eye(targets[e].raw(), FORMAT.as_raw() as u32, W, H, eyes[e].view, eyes[e].proj, &input, None)?;
    }
    Ok([targets[0].read(r)?, targets[1].read(r)?])
}

fn lit(img: &[u8], layer: usize) -> Vec<bool> {
    let n = (W * H * 4) as usize;
    img[layer * n..(layer + 1) * n].as_chunks::<4>().0.iter().map(|p| p[0] as u32 + p[1] as u32 + p[2] as u32 > 24).collect()
}

fn iou(a: &[bool], b: &[bool]) -> (f64, usize, usize) {
    let inter = a.iter().zip(b).filter(|(x, y)| **x && **y).count();
    let union = a.iter().zip(b).filter(|(x, y)| **x || **y).count();
    (inter as f64 / union.max(1) as f64, a.iter().filter(|x| **x).count(), b.iter().filter(|x| **x).count())
}

/// Largest per-channel difference and the number of differing pixels.
/// The counters of the last XR cluster cull.
pub(super) fn per_frame_counts(r: &Renderer) -> Option<crate::cluster::ClusterCounts> {
    r.cluster_xr_counts()
}

pub(super) fn diff(a: &[u8], b: &[u8]) -> (u8, usize) {
    let mut max = 0u8;
    let mut n = 0;
    for (x, y) in a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0) {
        let d = (0..3).map(|c| x[c].abs_diff(y[c])).max().unwrap_or(0);
        max = max.max(d);
        n += (d > 0) as usize;
    }
    (max, n)
}

/// Save layers side by side as a PNG under `docs/design/`.
pub(super) fn save_sheet(name: &str, layers: &[&[u8]]) {
    let (w, h) = (W as usize, H as usize);
    let mut out = vec![0u8; w * layers.len() * h * 4];
    for (k, img) in layers.iter().enumerate() {
        for y in 0..h {
            let src = &img[y * w * 4..(y + 1) * w * 4];
            let dst = (y * w * layers.len() + k * w) * 4;
            out[dst..dst + w * 4].copy_from_slice(src);
            for x in 0..w {
                out[dst + x * 4 + 3] = 255;
            }
        }
    }
    let png = super::stencil::png_rgba(W * layers.len() as u32, H, &out);
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/design").join(name);
    let _ = std::fs::create_dir_all(path.parent().unwrap());
    if let Err(e) = std::fs::write(&path, png) {
        eprintln!("[xr] could not write {}: {e}", path.display());
    }
}

pub(super) fn run(r: &mut Renderer) -> Result<String> {
    if !r.ctx.multiview {
        return Ok("[xr] this device has no multiview: skipped".into());
    }
    let scene = Scene::new(r)?;
    let stereo = Target::new(r, 2)?;
    let mut report = String::new();

    // ---- multiview stereo: ordinary draw vs one centre-eye cut for both layers
    let (mut off, mut on) = (Vec::new(), Vec::new());
    for _ in 0..3 {
        off = stereo_frame(r, &scene, &stereo, false, false, false)?;
    }
    for _ in 0..3 {
        on = stereo_frame(r, &scene, &stereo, false, true, false)?;
    }
    // The last stereo frame really went through the cluster cull (one instance, one batch, one cut).
    let (instances, batches, armed) = r.cluster.as_ref().map(|c| (c.last_instances, c.last_batches, c.drawable_armed())).unwrap_or_default();
    anyhow::ensure!(instances == 1 && batches == 1 && armed, "the stereo frame did not use the cluster path ({instances} instances, {batches} batches, armed {armed})");
    report += "[xr] real Renderer, multiview stereo pass into a 2-layer image, clusters Off vs On:";
    for l in 0..2 {
        let (i, a, b) = iou(&lit(&off, l), &lit(&on, l));
        report += &format!("\n[xr]   eye {l}: {a} px ordinary, {b} px clustered, silhouette IoU {i:.4}");
        anyhow::ensure!(a > 500, "eye {l}: the ordinary stereo pass drew nothing");
        anyhow::ensure!(i > 0.93, "eye {l}: the clustered multiview layer differs from the ordinary one (IoU {i})");
    }
    let (l0, l1) = (lit(&on, 0), lit(&on, 1));
    let differ = l0.iter().zip(&l1).filter(|(a, b)| a != b).count();
    report += &format!("\n[xr]   the eyes differ in {differ} px; the stereo frame culled {instances} instance in {batches} batch with ONE centre-eye cut for both layers");
    anyhow::ensure!(differ > 50, "both layers show the same picture");

    // ---- per-eye (non multiview): each eye is an ordinary camera with its own cut
    let eye_targets = [Target::new(r, 1)?, Target::new(r, 1)?];
    let (mut eoff, mut eon) = ([Vec::new(), Vec::new()], [Vec::new(), Vec::new()]);
    for _ in 0..3 {
        eoff = per_eye_frame(r, &scene, &eye_targets, false, false)?;
    }
    for _ in 0..3 {
        eon = per_eye_frame(r, &scene, &eye_targets, false, true)?;
    }
    let (instances, batches, armed) = r.cluster.as_ref().map(|c| (c.last_instances, c.last_batches, c.drawable_armed())).unwrap_or_default();
    anyhow::ensure!(instances == 1 && batches == 1 && armed, "the per-eye frame did not use the cluster path ({instances} instances, {batches} batches, armed {armed})");
    report += "\n[xr] real Renderer, per-eye passes (no multiview), clusters Off vs On, and per-eye vs the multiview layer:";
    for e in 0..2 {
        let o = lit(&eoff[e], 0);
        let c = lit(&eon[e], 0);
        let (i, a, b) = iou(&o, &c);
        let (im, _, _) = iou(&c, &lit(&on, e));
        report += &format!("\n[xr]   eye {e}: {a} px ordinary, {b} px clustered, IoU vs ordinary {i:.4}, IoU vs the multiview cluster layer {im:.4}");
        anyhow::ensure!(a > 500, "eye {e}: the ordinary per-eye pass drew nothing");
        anyhow::ensure!(i > 0.93, "eye {e}: the clustered per-eye image differs from the ordinary one (IoU {i})");
        anyhow::ensure!(im > 0.93, "eye {e}: the per-eye cluster image differs from the multiview one (IoU {im})");
    }
    let (b0, b1) = (lit(&eon[0], 0), lit(&eon[1], 0));
    anyhow::ensure!(b0.iter().zip(&b1).filter(|(a, b)| a != b).count() > 50, "both eyes show the same picture");
    let layer = (W * H * 4) as usize;
    save_sheet("geo5_xr_clusters.png", &[&off[..layer], &on[..layer], &eoff[0], &eon[0], &off[layer..], &on[layer..], &eoff[1], &eon[1]]);
    report += "\n[xr]   picture: docs/design/geo5_xr_clusters.png (left eye: stereo ordinary, stereo clustered, per-eye ordinary, per-eye clustered; then the right eye)";

    report += &super::xr_cluster_hiz::run(r, &scene, &stereo)?;
    report += &super::xr_cluster_hiz::run_per_eye(r, &scene, &eye_targets)?;

    stereo.destroy(r);
    for t in eye_targets {
        t.destroy(r);
    }
    Ok(report)
}
