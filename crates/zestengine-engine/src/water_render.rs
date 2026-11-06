//! Water surfaces: a generated mesh per active `Water`, drawn through the
//! standard shader's water path. The GPU sums the same Gerstner waves as
//! `zestengine_core::Water::displacement` (packed from `resolved_waves`) on the
//! same clock, so floating bodies ride the waves that are drawn.
//!
//! Oceans use a fixed grid around the camera, fine at its centre and growing
//! coarser toward the horizon, drawn shifted in whole centre cells as the
//! camera moves (the vertices never change); lakes with one zero size axis a
//! uniform grid that follows the camera in whole cells; lakes a grid over
//! their rectangle or polygon; rivers a ribbon along their points. Vertex
//! colour carries the local flow (rg = m/s / 10 + 0.5) and the distance to
//! the bank (b, waves calm there); uv1.x the grid spacing at the vertex,
//! which the shader band-limits the displacement to.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use zestengine_core::{ObjectId, Water, WaterBody as BodyKind};
use zestengine_render::{GpuWater, MeshHandle, MaterialHandle, Renderer, TextureHandle, MAX_WATER_BODIES, MAX_WATER_WAVES};
use glam::{Vec2, Vec3};

use crate::scene::LoadedScene;

/// Side (m) of the grid that follows the camera over endless water.
/// How far the graded mesh of an endless water body reaches from the camera
/// (keep in sync with `WATER_HORIZON_EXTENT` in
/// `crates/zestengine-render/shaders/standard_common.glsl`, which fades the
/// surface out just before this edge).
const ENDLESS_EXTENT: f32 = 240.0;
/// Meshes per surface. A re-centre or edit rewrites the next one, which no
/// frame in flight can still be reading.
const RING: usize = 3;
const MAX_QUADS: u32 = 512;
/// Fewest quads per side of an ocean's graded grid.
const OCEAN_MIN_QUADS: u32 = 192;
/// Spacing (m) at the centre of an ocean's grid without a simulation.
const OCEAN_CENTRE_SPACING: f32 = 0.25;
/// Vertices across a river ribbon, minus one.
const RIVER_ACROSS: u32 = 8;
/// Frames a new mesh layout must hold before it is built (dragging a lake or
/// river point would otherwise make a new mesh every frame).
const STABLE_FRAMES: u32 = 8;

/// What a water's mesh is built from this frame.
#[derive(Clone, Copy, PartialEq)]
enum Shape {
    /// `quads` x `quads` cells spanning `extent` around `offset` (local XZ);
    /// `clip` drops the cells' triangles outside the lake polygon.
    Grid { quads: u32, extent: Vec2, offset: Vec2, clip: bool },
    /// `quads` x `quads` cells spanning `extent` around the origin, `centre`
    /// metres wide in the middle and growing geometrically outward.
    Graded { quads: u32, extent: f32, centre: f32 },
    River,
}

struct Plan {
    shape: Shape,
    /// Vertex count and indices depend only on this.
    layout: u64,
    /// Everything the vertices depend on.
    vertices: u64,
    center: Vec3,
    radius: f32,
    /// Local XZ the mesh is drawn shifted by (an ocean following the camera).
    shift: Vec2,
}

struct Surface {
    meshes: [MeshHandle; RING],
    current: usize,
    material: MaterialHandle,
    layout: u64,
    vertices: u64,
    /// A new layout and how many frames it has held.
    pending: Option<(u64, u32)>,
}

/// Meshes + materials for the scene's water, and the water clock.
#[derive(Default)]
pub struct WaterRenderer {
    surfaces: HashMap<ObjectId, Surface>,
    /// Released mesh rings (layout, meshes, frame released), reused for the
    /// same layout once no frame in flight can still draw them.
    spare_meshes: Vec<(u64, [MeshHandle; RING], u64)>,
    spare_materials: Vec<MaterialHandle>,
    /// Ripple field texture (and its side) bound to each water material.
    ripple_textures: Vec<(MaterialHandle, TextureHandle, u32)>,
    /// Shore-depth texture bound to each water material, its side and the
    /// `ShoreDepth::version` last uploaded to it.
    shore_textures: Vec<(MaterialHandle, TextureHandle, u32, u64)>,
    bodies: Vec<GpuWater>,
    edit_time: f32,
    time: f32,
    frame: u64,
}

impl WaterRenderer {
    /// The water clock: the play clock while playing, else a running editor clock.
    pub fn time(&self) -> f32 {
        self.time
    }

    /// This frame's water bodies, indexed by the water materials' slots.
    pub fn bodies(&self) -> &[GpuWater] {
        &self.bodies
    }

    /// Advance the clock, keep each active water's mesh and material current
    /// and queue its draw on `scene.extra_draws`. `play_time` is the play
    /// session's clock while playing.
    pub fn update(&mut self, renderer: &mut Renderer, scene: &mut LoadedScene, cam_pos: Vec3, play_time: Option<f32>, dt: f32) {
        scene.water_assets.sync(&mut scene.objects);
        self.frame += 1;
        self.edit_time += dt.max(0.0);
        self.time = play_time.unwrap_or(self.edit_time);
        self.bodies.clear();
        let active = scene.visible_flags();
        let world = scene.world_transforms();
        let mut alive = Vec::new();
        let spots = crate::water_fx::foam_spots(scene, &world, self.time);
        for i in 0..scene.objects.len() {
            if !active[i] || self.bodies.len() >= MAX_WATER_BODIES {
                continue;
            }
            let id = scene.objects[i].id;
            let Some(water) = scene.objects[i].components.iter().find_map(|c| c.as_any().downcast_ref::<Water>()) else {
                continue;
            };
            let local_cam = world[i].inverse().transform_point3(cam_pos);
            let Some(plan) = plan(water, local_cam) else {
                continue;
            };
            let mut body = pack_water(water);
            body.to_local = world[i].inverse().to_cols_array_2d();
            // The caustics advect their ripple pattern with the water's real
            // flow here: a river's column flow at the body origin (world, then
            // to local), else the uniform current.
            let flow = water.base_flow_at(world[i], world[i].w_axis.truncate());
            let local_flow = world[i].inverse().transform_vector3(flow);
            body.reflect[1] = local_flow.x;
            body.reflect[2] = local_flow.z;
            body.bounds = caustic_bounds(water);
            if water.project_caustics {
                body.extra[0] = water.projection_strength.max(0.0);
            }
            let to_local = world[i].inverse();
            let mut k = 0;
            for (at, radius, amount) in &spots {
                let l = to_local.transform_point3(*at);
                if k < zestengine_render::MAX_WATER_FOAM && water.covers(Vec2::new(l.x, l.z)) {
                    body.foam[k] = [l.x, l.z, *radius, *amount];
                    k += 1;
                }
            }
            let Some((mesh, material)) = self.surface(renderer, id, water, &plan) else {
                continue;
            };
            let probe = crate::water_probe::enabled().then(|| (world[i], scene.objects[i].name.clone()));
            let Some(water) = scene.objects[i].components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<Water>()) else {
                continue;
            };
            body.ripple = self.ripples(renderer, material, water, id.raw() as u64, local_cam, dt, probe);
            body.ripple[3] = if water.look.foam_on { water.look.foam.max(0.0) } else { 0.0 };
            if let Some(half) = self.shore(renderer, material, water) {
                body.extra[1] = water.shore_depth;
                body.extra[2] = half;
                body.extra[3] = half;
            }
            alive.push(id);
            renderer.set_material_water_slot(material, self.bodies.len() as u32);
            self.bodies.push(body);
            let planar_index = if water.reflection == zestengine_core::SurfaceReflection::Planar {
                scene.water_planar_index(i)
            } else {
                -1
            };
            tracing::debug!("water {}: planar layer {planar_index}", scene.objects[i].name);
            scene.extra_draws.push(zestengine_render::DrawCmd {
                gi_source: None,
                mesh,
                material,
                transform: world[i] * glam::Mat4::from_translation(Vec3::new(plan.shift.x, 0.0, plan.shift.y)),
                highlight: 0.0,
                mesh_center: plan.center,
                bound_radius: plan.radius + body.look1[3] * 2.0,
                lightmap_layer: -1,
                lightmap_size: 0,
                static_geometry: false,
                planar_index,
                occluded: false,
                shadow_casting: zestengine_render::ShadowCasting::On,
                instance_color: None,
                instance_param: None,
            });
        }
        let gone: Vec<ObjectId> = self.surfaces.keys().filter(|id| !alive.contains(id)).copied().collect();
        for id in gone {
            self.release(id);
        }
    }

    /// Keep `water`'s interactive layer current for `material` and return
    /// its window (`GpuWater::ripple`): the GPU simulation when it can run,
    /// else the CPU ripple field uploaded. `cam` is the camera in the
    /// water's local space, `dt` the frame time (0 = paused).
    #[allow(clippy::too_many_arguments)]
    fn ripples(&mut self, renderer: &mut Renderer, material: MaterialHandle, water: &mut Water, key: u64, cam: Vec3, dt: f32, probe: Option<(glam::Mat4, String)>) -> [f32; 4] {
        if !water.interactive {
            water.sim_feed = Default::default();
            return [0.0; 4];
        }
        if water.gpu_sim && renderer.water_sim_supported() {
            return self.gpu_ripples(renderer, material, water, key, cam, dt, probe);
        }
        if water.ripples.as_ref().is_some_and(|r| r.gpu) {
            water.ripples = None;
            water.sim_feed = Default::default();
        }
        let n = water.ripples.as_ref().map_or(water.ripple_resolution as usize, |r| r.n).clamp(8, 512) as u32;
        let Some(texture) = self.ripple_texture(renderer, material, n) else {
            return [0.0; 4];
        };
        let Some(sim) = water.ripples.as_ref() else {
            return [0.0; 4];
        };
        renderer.update_ripple_texture(texture, sim.n, sim.heights(), sim.cell());
        [sim.center.x, sim.center.y, sim.size, 0.0]
    }

    /// The ripple texture bound to `material`, `n` x `n` (made with the
    /// surface, remade when the size changes).
    fn ripple_texture(&mut self, renderer: &mut Renderer, material: MaterialHandle, n: u32) -> Option<TextureHandle> {
        let k = self.ripple_textures.iter().position(|r| r.0 == material);
        match k {
            Some(k) if self.ripple_textures[k].2 == n => Some(self.ripple_textures[k].1),
            _ => {
                let texture = match renderer.create_ripple_texture(n) {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::error!("water ripple texture: {e:#}");
                        return None;
                    }
                };
                match k {
                    Some(k) => self.ripple_textures[k] = (material, texture, n),
                    None => self.ripple_textures.push((material, texture, n)),
                }
                self.bind_textures(renderer, material);
                Some(texture)
            }
        }
    }

    /// Keep `water`'s shore-depth bake uploaded to `material`'s shore
    /// texture; the bake's half extent (local XZ) when its waves calm over
    /// it (`GpuWater::extra`).
    fn shore(&mut self, renderer: &mut Renderer, material: MaterialHandle, water: &Water) -> Option<f32> {
        let bake = water.shore.as_ref().filter(|_| water.shore_depth > 0.0)?;
        let n = bake.n as u32;
        let k = match self.shore_textures.iter().position(|s| s.0 == material) {
            Some(k) if self.shore_textures[k].2 == n => k,
            k => {
                let texture = match renderer.create_ripple_texture(n) {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::error!("water shore texture: {e:#}");
                        return None;
                    }
                };
                let k = match k {
                    Some(k) => {
                        self.shore_textures[k] = (material, texture, n, 0);
                        k
                    }
                    None => {
                        self.shore_textures.push((material, texture, n, 0));
                        self.shore_textures.len() - 1
                    }
                };
                self.bind_textures(renderer, material);
                k
            }
        };
        let s = &mut self.shore_textures[k];
        if s.3 != bake.version {
            // Depth in the red channel (the gradient it fills in goes unused).
            renderer.update_ripple_texture(s.1, bake.n, &bake.depth, 2.0 * bake.half / bake.n as f32);
            s.3 = bake.version;
        }
        Some(bake.half)
    }

    /// Bind `material`'s ripple field (slot 15) and shore depth (slot 14,
    /// which the water path does not read as a metallic map).
    fn bind_textures(&self, renderer: &mut Renderer, material: MaterialHandle) {
        let mut slots = [None; 16];
        slots[14] = self.shore_textures.iter().find(|s| s.0 == material).map(|s| s.1);
        slots[15] = self.ripple_textures.iter().find(|r| r.0 == material).map(|r| r.1);
        renderer.set_material_textures(material, &slots);
    }

    /// Queue this frame of `water`'s GPU simulation (writing its ripple
    /// texture) and take in the newest readback as `water.ripples`, which
    /// physics, Wave Riders and foam read.
    #[allow(clippy::too_many_arguments)]
    fn gpu_ripples(&mut self, renderer: &mut Renderer, material: MaterialHandle, water: &mut Water, key: u64, cam: Vec3, dt: f32, probe: Option<(glam::Mat4, String)>) -> [f32; 4] {
        let window = zestengine_core::choose_window(water, Vec2::new(cam.x, cam.z), water.sim_feed.window);
        water.sim_feed.window = Some(window);
        let Some(texture) = self.ripple_texture(renderer, material, window.n) else {
            return [0.0; 4];
        };
        if water.ripples.as_ref().is_none_or(|r| !r.gpu) {
            water.ripples = Some(zestengine_core::RippleSim::gpu_owned(window.n as usize, window.size()));
        }
        if let Some(rb) = renderer.water_sim_readback(key)
            && rb.serial != water.sim_feed.readback_serial
            && let Some(sim) = water.ripples.as_mut()
        {
            water.sim_feed.readback_serial = rb.serial;
            let n = rb.n as usize;
            let center = (Vec2::new(rb.origin[0] as f32, rb.origin[1] as f32) + Vec2::splat(rb.n as f32 * 0.5)) * rb.cell;
            sim.set_field(n, rb.n as f32 * rb.cell, center, &rb.cells);
        }
        let (max_height, max_speed) = water.ripples.as_ref().map_or((0.0, 0.0), |r| (r.max_height, r.max_speed));
        let mut feed = std::mem::take(&mut water.sim_feed);
        let (field, reset) = feed.field(water, window);
        if let Some((world, name)) = probe {
            // `ZESTENGINE_WATER_PROBE`: the sim dt below is what the solver
            // advances by this frame; the probe clocks itself by it.
            let dt_sim = dt.clamp(0.0, 0.05);
            let bed = feed.bed.as_ref().filter(|(w, _)| *w == window).map(|(_, b)| b.clone());
            crate::water_probe::frame(renderer, key, &name, world, window, &field, bed.as_deref().map(|b| b.as_slice()), dt_sim);
        }
        let displacers = feed.take_displacers(window.cell, zestengine_render::MAX_WATER_DISPLACERS);
        let params = zestengine_core::PipeParams::for_water(water);
        let ewave = zestengine_core::EwaveParams::for_water(water, window, field.max_depth).map(|e| zestengine_render::WaterSimEwave {
            gain: e.gain,
            depth: e.depth,
            damping: e.damping,
            split: e.split,
            current: e.current.to_array(),
            body_gain: e.body_gain,
            smoothing: e.smoothing,
            grid_damping: e.grid_damping,
            mix: e.mix,
        });
        let depth = field.max_depth.min(params.wave_depth) + max_height;
        let dt = dt.clamp(0.0, 0.05).min(zestengine_core::max_sim_dt(window.cell, depth, max_speed));
        let substeps = zestengine_core::pipe_substeps(dt, window.cell, depth, max_speed);
        // Still water stops stepping until disturbed: nothing in it, or only
        // bodies held where they were, once the water round them has settled.
        let held = displacers.iter().all(|d| !d.impulse && d.rate == 0.0 && d.velocity == Vec2::ZERO) && displacers == feed.last_displacers;
        let still = held && max_height < 2e-4 && max_speed < 2e-3 && !reset;
        feed.last_displacers.clone_from(&displacers);
        feed.quiet_frames = if still { feed.quiet_frames.saturating_add(1) } else { 0 };
        renderer.queue_water_sim(zestengine_render::WaterSimBody {
            key,
            texture,
            n: window.n,
            cell: window.cell,
            origin: window.origin.to_array(),
            closed: window.closed,
            sponge: window.sponge(),
            reset,
            idle: feed.quiet_frames > 120,
            dt,
            substeps,
            gravity: params.gravity,
            keep: params.keep(dt / substeps as f32),
            momentum: params.momentum,
            level_return: params.level_return,
            wave_depth: params.wave_depth,
            displace: params.displacement,
            viscosity: params.viscosity,
            eddy: params.eddy,
            ewave,
            field_version: field.version,
            field_origin: field.window.origin.to_array(),
            field: field.cells.clone(),
            flowing: field.flowing,
            default_depth: field.default_depth,
            displacers: displacers.iter().map(|d| d.to_gpu()).collect(),
        });
        water.sim_feed = feed;
        let c = window.center();
        [c.x, c.y, window.size(), 0.0]
    }

    fn release(&mut self, id: ObjectId) {
        if let Some(s) = self.surfaces.remove(&id) {
            self.spare_meshes.push((s.layout, s.meshes, self.frame));
            self.spare_materials.push(s.material);
        }
    }

    /// The mesh + material to draw for `id`, built, rebuilt or rewritten as
    /// the plan asks.
    fn surface(&mut self, renderer: &mut Renderer, id: ObjectId, water: &Water, plan: &Plan) -> Option<(MeshHandle, MaterialHandle)> {
        if let Some(s) = self.surfaces.get_mut(&id)
            && s.layout != plan.layout
        {
            let held = match s.pending {
                Some((layout, n)) if layout == plan.layout => n + 1,
                _ => 0,
            };
            s.pending = Some((plan.layout, held));
            if held < STABLE_FRAMES {
                return Some((s.meshes[s.current], s.material));
            }
            self.release(id);
        }
        if !self.surfaces.contains_key(&id) {
            let frame = self.frame;
            let data = build(water, plan.shape);
            let reuse = self
                .spare_meshes
                .iter()
                .position(|&(layout, _, released)| layout == plan.layout && frame - released >= RING as u64);
            let meshes = match reuse {
                Some(k) => {
                    let (_, meshes, _) = self.spare_meshes.swap_remove(k);
                    for &m in &meshes {
                        renderer.update_mesh_vertices(m, &data.vertices);
                    }
                    meshes
                }
                None => {
                    let mut meshes = Vec::with_capacity(RING);
                    for _ in 0..RING {
                        match renderer.upload_mesh_skinned(&data) {
                            Ok(m) => meshes.push(m),
                            Err(e) => {
                                tracing::error!("water mesh: {e:#}");
                                return None;
                            }
                        }
                    }
                    [meshes[0], meshes[1], meshes[2]]
                }
            };
            let material = match self.spare_materials.pop() {
                Some(m) => m,
                None => match water_material(renderer) {
                    Ok(m) => m,
                    Err(e) => {
                        tracing::error!("water material: {e:#}");
                        return None;
                    }
                },
            };
            let surface = Surface { meshes, current: 0, material, layout: plan.layout, vertices: plan.vertices, pending: None };
            self.surfaces.insert(id, surface);
        }
        let s = self.surfaces.get_mut(&id)?;
        s.pending = None;
        if s.vertices != plan.vertices {
            s.current = (s.current + 1) % RING;
            renderer.update_mesh_vertices(s.meshes[s.current], &build(water, plan.shape).vertices);
            s.vertices = plan.vertices;
        }
        Some((s.meshes[s.current], s.material))
    }
}

fn water_material(renderer: &mut Renderer) -> anyhow::Result<MaterialHandle> {
    let features = zestengine_render::MaterialFeatures {
        alpha_mode: zestengine_render::AlphaMode::Blend,
        double_sided: true,
        water: true,
        ..Default::default()
    };
    let desc = zestengine_render::MaterialDesc { name: "water".to_owned(), features, ..Default::default() };
    renderer.create_material(&desc)
}

fn hash_f32s(h: &mut impl Hasher, values: impl IntoIterator<Item = f32>) {
    for v in values {
        v.to_bits().hash(h);
    }
}

/// The mesh a water draws this frame, None when it has nothing to draw.
/// `cam` is the camera in the water's local space.
fn plan(water: &Water, cam: Vec3) -> Option<Plan> {
    let quads = water.resolution.clamp(1, MAX_QUADS);
    let mut layout = std::collections::hash_map::DefaultHasher::new();
    let mut shift = Vec2::ZERO;
    let shape = match water.body {
        BodyKind::River => {
            if water.river.len() < 2 {
                return None;
            }
            (2u8, river_samples(water).len()).hash(&mut layout);
            Shape::River
        }
        BodyKind::Lake | BodyKind::Pool if water.shape.len() >= 3 => {
            let outline = water.outline();
            let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
            for &p in outline.iter() {
                lo = lo.min(Vec2::from(p));
                hi = hi.max(Vec2::from(p));
            }
            (1u8, quads).hash(&mut layout);
            hash_f32s(&mut layout, outline.iter().flatten().copied());
            Shape::Grid { quads, extent: (hi - lo).max(Vec2::splat(0.01)), offset: (lo + hi) * 0.5, clip: true }
        }
        BodyKind::Ocean | BodyKind::Lake | BodyKind::Pool if water.body == BodyKind::Ocean || (water.size[0] <= 0.0 && water.size[1] <= 0.0) => {
            // Fine where the camera is (two simulation cells, so the
            // interactive layer shows in the shape), coarse at the horizon.
            let quads = quads.max(OCEAN_MIN_QUADS).div_ceil(2) * 2;
            let centre = if water.interactive {
                (2.0 * water.ripple_area.max(1.0) / zestengine_core::sim_resolution(water.ripple_resolution) as f32).max(0.05)
            } else {
                OCEAN_CENTRE_SPACING
            };
            (3u8, quads, centre.to_bits()).hash(&mut layout);
            let snap = |v: f32| if v.is_finite() { (v / centre).round() * centre } else { 0.0 };
            shift = Vec2::new(snap(cam.x), snap(cam.z));
            Shape::Graded { quads, extent: ENDLESS_EXTENT, centre }
        }
        _ => {
            let endless = [water.size[0] <= 0.0, water.size[1] <= 0.0];
            let extent = Vec2::new(
                if endless[0] { ENDLESS_EXTENT } else { water.size[0] },
                if endless[1] { ENDLESS_EXTENT } else { water.size[1] },
            );
            let cell = extent / quads as f32;
            let snap = |follow: bool, v: f32, c: f32| if follow && c > 0.0 && v.is_finite() { (v / c).round() * c } else { 0.0 };
            let offset = Vec2::new(snap(endless[0], cam.x, cell.x), snap(endless[1], cam.z, cell.y));
            (0u8, quads).hash(&mut layout);
            Shape::Grid { quads, extent, offset, clip: false }
        }
    };
    let layout = layout.finish();
    let mut vertices = std::collections::hash_map::DefaultHasher::new();
    layout.hash(&mut vertices);
    hash_f32s(&mut vertices, water.current);
    let (center, radius) = match shape {
        Shape::Grid { extent, offset, .. } => {
            hash_f32s(&mut vertices, [extent.x, extent.y, offset.x, offset.y]);
            (Vec3::new(offset.x, 0.0, offset.y), (extent * 0.5).length())
        }
        Shape::Graded { extent, centre, .. } => {
            hash_f32s(&mut vertices, [extent, centre]);
            (Vec3::ZERO, extent * std::f32::consts::FRAC_1_SQRT_2)
        }
        Shape::River => {
            let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            for p in &water.river {
                hash_f32s(&mut vertices, [p.position[0], p.position[1], p.position[2], p.width, p.depth, p.speed]);
                let pad = Vec3::new(p.width * 0.5, 0.0, p.width * 0.5);
                lo = lo.min(Vec3::from(p.position) - pad);
                hi = hi.max(Vec3::from(p.position) + pad);
            }
            ((lo + hi) * 0.5, ((hi - lo) * 0.5).length())
        }
    };
    Some(Plan { shape, layout, vertices: vertices.finish(), center, radius, shift })
}

/// Vertex colour for a local flow (m/s) and bank distance (0 centre, 1 bank).
fn flow_color(flow: Vec3, edge: f32) -> [f32; 4] {
    [flow.x / 10.0 + 0.5, flow.z / 10.0 + 0.5, edge.clamp(0.0, 1.0), 1.0]
}

/// A rest-grid vertex; `spacing` is the distance to its neighbours (m).
fn vertex(p: Vec3, uv: [f32; 2], color: [f32; 4], spacing: f32) -> zestengine_render::Vertex {
    zestengine_render::Vertex {
        position: p.to_array(),
        normal: [0.0, 1.0, 0.0],
        uv,
        color,
        tangent: [1.0, 0.0, 0.0, 1.0],
        uv1: [spacing, 0.0],
        joints: [0; 4],
        weights: [0.0; 4],
    }
}

fn build(water: &Water, shape: Shape) -> zestengine_render::MeshData {
    match shape {
        Shape::Grid { quads, extent, offset, clip } => grid_mesh(water, quads, extent, offset, clip),
        Shape::Graded { quads, extent, centre } => graded_mesh(water, quads, extent, centre),
        Shape::River => river_mesh(water),
    }
}

/// Cells on the local XZ plane facing +Y (counter-clockwise from above).
fn grid_mesh(water: &Water, quads: u32, extent: Vec2, offset: Vec2, clip: bool) -> zestengine_render::MeshData {
    let n = quads as usize;
    let color = flow_color(Vec3::from(water.current), 0.0);
    let spacing = (extent / n as f32).max_element();
    let mut vertices = Vec::with_capacity((n + 1) * (n + 1));
    for z in 0..=n {
        for x in 0..=n {
            let uv = Vec2::new(x as f32, z as f32) / n as f32;
            let p = offset + (uv - 0.5) * extent;
            vertices.push(vertex(Vec3::new(p.x, 0.0, p.y), uv.to_array(), color, spacing));
        }
    }
    let row = n as u32 + 1;
    let mut indices = Vec::with_capacity(n * n * 6);
    let inside = |tri: [u32; 3]| {
        let c = tri.iter().map(|&i| Vec2::new(vertices[i as usize].position[0], vertices[i as usize].position[2])).sum::<Vec2>() / 3.0;
        water.column(c).is_some()
    };
    for z in 0..n as u32 {
        for x in 0..n as u32 {
            let a = z * row + x;
            let b = a + row;
            for tri in [[a, b, a + 1], [a + 1, b, b + 1]] {
                if !clip || inside(tri) {
                    indices.extend_from_slice(&tri);
                }
            }
        }
    }
    zestengine_render::MeshData { vertices, indices, has_lightmap_uv: false }
}

/// Positions across a graded grid of `quads` cells spanning `extent`: the
/// middle cells `centre` wide, each further out wider by a constant ratio
/// (constant size on screen), reaching the edges.
fn graded_axis(quads: u32, extent: f32, centre: f32) -> Vec<f32> {
    let m = (quads / 2).max(1) as i32;
    let half = extent * 0.5;
    // Distance out to cell boundary `j` for growth ratio `r`, in centre cells.
    let out = |r: f64, j: i32| if (r - 1.0).abs() < 1e-9 { j as f64 } else { (r.powi(j) - 1.0) / (r - 1.0) };
    let target = half as f64 / centre.max(1e-3) as f64;
    let r = if out(1.0, m) >= target {
        1.0
    } else {
        let (mut lo, mut hi) = (1.0f64, 2.0f64);
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if out(mid, m) < target { lo = mid } else { hi = mid }
        }
        0.5 * (lo + hi)
    };
    let scale = half as f64 / out(r, m);
    let at = |j: i32| (scale * out(r, j)) as f32;
    (-m..=m).map(|j| at(j.abs()) * j.signum() as f32).collect()
}

/// An ocean's grid: `graded_axis` both ways, each vertex knowing its spacing.
fn graded_mesh(water: &Water, quads: u32, extent: f32, centre: f32) -> zestengine_render::MeshData {
    let axis = graded_axis(quads, extent, centre);
    let n = axis.len() - 1;
    let gap = |i: usize| {
        let a = if i > 0 { axis[i] - axis[i - 1] } else { axis[1] - axis[0] };
        let b = if i < n { axis[i + 1] - axis[i] } else { axis[n] - axis[n - 1] };
        a.max(b)
    };
    let color = flow_color(Vec3::from(water.current), 0.0);
    let mut vertices = Vec::with_capacity((n + 1) * (n + 1));
    for z in 0..=n {
        for x in 0..=n {
            let uv = [x as f32 / n as f32, z as f32 / n as f32];
            vertices.push(vertex(Vec3::new(axis[x], 0.0, axis[z]), uv, color, gap(x).max(gap(z))));
        }
    }
    let row = n as u32 + 1;
    let mut indices = Vec::with_capacity(n * n * 6);
    for z in 0..n as u32 {
        for x in 0..n as u32 {
            let a = z * row + x;
            let b = a + row;
            indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    zestengine_render::MeshData { vertices, indices, has_lightmap_uv: false }
}

/// One cross-section of a river ribbon.
struct RiverSample {
    pos: Vec3,
    width: f32,
    flow: Vec3,
    /// Distance downstream (m).
    dist: f32,
}

/// Cross-sections every ~half width along the river's points.
fn river_samples(water: &Water) -> Vec<RiverSample> {
    let current = Vec3::from(water.current);
    water
        .river_path()
        .iter()
        .map(|p| RiverSample { pos: p.pos, width: p.width, flow: p.dir * p.speed + current, dist: p.dist })
        .collect()
}

/// A ribbon along the river, `RIVER_ACROSS` cells wide, at the points' heights.
fn river_mesh(water: &Water) -> zestengine_render::MeshData {
    let samples = river_samples(water);
    let cols = RIVER_ACROSS + 1;
    let mut vertices = Vec::with_capacity(samples.len() * cols as usize);
    for (i, s) in samples.iter().enumerate() {
        let prev = samples[i.saturating_sub(1)].pos;
        let next = samples[(i + 1).min(samples.len() - 1)].pos;
        let tangent = Vec2::new(next.x - prev.x, next.z - prev.z).normalize_or(Vec2::X);
        let side = Vec3::new(-tangent.y, 0.0, tangent.x);
        let along = (next - prev).length() / ((i + 1).min(samples.len() - 1) - i.saturating_sub(1)).max(1) as f32;
        let spacing = along.max(s.width / RIVER_ACROSS as f32);
        for j in 0..cols {
            let across = j as f32 / RIVER_ACROSS as f32 * 2.0 - 1.0;
            let p = s.pos + side * (across * s.width * 0.5);
            let uv = [across * 0.5 + 0.5, s.dist / s.width];
            vertices.push(vertex(p, uv, flow_color(s.flow, across.abs()), spacing));
        }
    }
    let mut indices = Vec::with_capacity(samples.len().saturating_sub(1) * RIVER_ACROSS as usize * 6);
    for i in 0..samples.len().saturating_sub(1) as u32 {
        for j in 0..RIVER_ACROSS {
            let a = i * cols + j;
            let b = a + cols;
            indices.extend_from_slice(&[a, a + 1, b, b, a + 1, b + 1]);
        }
    }
    zestengine_render::MeshData { vertices, indices, has_lightmap_uv: false }
}

/// GPU layout of a water's waves (`resolved_waves`) and look.
/// Where a body's caustics fall (see `GpuWater::bounds`): its local XZ half
/// extents (0 = endless), rest surface height and caustic strength.
fn caustic_bounds(water: &Water) -> [f32; 4] {
    let strength = water.underwater.caustics.max(0.0);
    match water.body {
        zestengine_core::WaterBody::Ocean => [0.0, 0.0, 0.0, strength],
        zestengine_core::WaterBody::Lake | zestengine_core::WaterBody::Pool if water.shape.len() >= 3 => {
            let h = water.outline().iter().fold(Vec2::ZERO, |m, p| m.max(Vec2::from(*p).abs()));
            [h.x, h.y, 0.0, strength]
        }
        zestengine_core::WaterBody::Lake | zestengine_core::WaterBody::Pool => [water.size[0] * 0.5, water.size[1] * 0.5, 0.0, strength],
        zestengine_core::WaterBody::River => {
            let path = water.river_path();
            if path.is_empty() {
                return [0.0; 4];
            }
            let h = path.iter().fold(Vec2::ZERO, |m, p| m.max(Vec2::new(p.pos.x.abs(), p.pos.z.abs()) + Vec2::splat(p.width * 0.5)));
            let y = path.iter().map(|p| p.pos.y).sum::<f32>() / path.len() as f32;
            [h.x, h.y, y, strength]
        }
    }
}

pub fn pack_water(water: &Water) -> GpuWater {
    let mut body = GpuWater::default();
    let waves = water.resolved_waves();
    let n = waves.len().min(MAX_WATER_WAVES);
    let mut amp_sum = 0.0;
    for (i, w) in waves.iter().take(n).enumerate() {
        body.wave_a[i] = [w.dir.x, w.dir.y, w.k, w.omega];
        body.wave_b[i] = [w.qa, w.amplitude, (w.qa / w.amplitude.max(1e-6)) * w.k * w.amplitude, 0.0];
        amp_sum += w.amplitude.max(0.0);
    }
    let l = &water.look;
    let optics = l.optics.unwrap_or_else(|| zestengine_core::WaterOptics::from_legacy(l)).sanitized();
    body.set_optics([optics.absorption[0], optics.absorption[1], optics.absorption[2], optics.ior],
        [optics.scattering[0], optics.scattering[1], optics.scattering[2], optics.anisotropy]);
    body.shallow = [l.shallow_color[0], l.shallow_color[1], l.shallow_color[2], l.absorption_depth.max(0.01)];
    body.deep = [l.deep_color[0], l.deep_color[1], l.deep_color[2], l.clarity.clamp(0.0, 1.0)];
    // The shader's scrolling detail is about 15 cm waves (at scale 1):
    // thick fluids (and oil, a little) calm it.
    let detail_k = std::f32::consts::TAU * l.ripple_scale.max(0.01) / 0.15;
    body.look0 = [l.roughness, l.ripple_strength * water.wave_share(detail_k), l.ripple_scale, l.foam_width];
    body.look1 = [l.crest_foam, l.refraction, n as f32, amp_sum];
    body.reflect = [water.reflection as u8 as f32, water.current[0], water.current[2], 0.0];
    body
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `water_waves` in standard_common.glsl, on the packed body.
    fn shader_waves(b: &GpuWater, xz: Vec2, t: f32) -> (Vec3, Vec3) {
        let mut d = Vec3::ZERO;
        let mut n = Vec3::Y;
        for i in 0..(b.look1[2] as usize).min(MAX_WATER_WAVES) {
            let a = b.wave_a[i];
            let w = b.wave_b[i];
            let phase = a[2] * (a[0] * xz.x + a[1] * xz.y) - a[3] * t;
            let (s, c) = phase.sin_cos();
            d.x += w[0] * a[0] * c;
            d.z += w[0] * a[1] * c;
            d.y += w[1] * s;
            let wa = a[2] * w[1];
            n.x -= a[0] * wa * c;
            n.z -= a[1] * wa * c;
            n.y -= w[2] * s;
        }
        (d, n.normalize())
    }

    fn check(water: &Water) {
        let body = pack_water(water);
        for k in 0..64 {
            let xz = Vec2::new(k as f32 * 0.73 - 20.0, (k as f32 * 1.31).sin() * 15.0);
            let t = k as f32 * 0.21;
            let (d, n) = water.displacement(xz, t);
            let (gd, gn) = shader_waves(&body, xz, t);
            assert!((d - gd).length() < 1e-5, "displacement {d} vs {gd}");
            assert!((n - gn).length() < 1e-4, "normal {n} vs {gn}");
        }
    }

    #[test]
    fn packed_waves_match_the_physics_waves() {
        check(&Water::default());
        let mut windy = Water::default();
        windy.runtime.wind_dir = Vec2::new(0.6, 0.8);
        windy.runtime.wind_speed = 9.0;
        check(&windy);
        check(&Water { waves: Vec::new(), ..Default::default() });
    }

    fn faces_up(m: &zestengine_render::MeshData) -> bool {
        let p = |i: u32| Vec3::from(m.vertices[i as usize].position);
        m.indices.chunks(3).all(|t| (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0])).y > 0.0)
    }

    #[test]
    fn lake_grid_counts_bounds_and_winding() {
        let water = Water { size: [20.0, 10.0], resolution: 16, ..Default::default() };
        let plan = plan(&water, Vec3::new(50.0, 3.0, 50.0)).unwrap();
        let m = build(&water, plan.shape);
        assert_eq!(m.vertices.len(), 17 * 17);
        assert_eq!(m.indices.len(), 16 * 16 * 6);
        assert!(m.indices.iter().all(|&i| (i as usize) < m.vertices.len()));
        let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
        for v in &m.vertices {
            let p = Vec2::new(v.position[0], v.position[2]);
            lo = lo.min(p);
            hi = hi.max(p);
            assert_eq!(v.position[1], 0.0);
            assert_eq!(&v.color[..3], &[0.5, 0.5, 0.0], "still water, no bank");
        }
        assert!((lo - Vec2::new(-10.0, -5.0)).length() < 1e-4 && (hi - Vec2::new(10.0, 5.0)).length() < 1e-4);
        assert!(faces_up(&m));
    }

    #[test]
    fn ocean_grid_is_fine_at_the_camera_and_shifts_in_whole_centre_cells() {
        let water = Water { body: BodyKind::Ocean, resolution: 80, ..Default::default() };
        let a = plan(&water, Vec3::new(53.3, 5.0, -12.0)).unwrap();
        let Shape::Graded { quads, extent, centre } = a.shape else { panic!() };
        assert!(quads >= OCEAN_MIN_QUADS && quads % 2 == 0);
        assert_eq!(extent, ENDLESS_EXTENT);
        let cell = water.ripple_area / zestengine_core::sim_resolution(water.ripple_resolution) as f32;
        assert!((centre - 2.0 * cell).abs() < 1e-5, "two simulation cells: {centre}");
        assert!((a.shift.x / centre - (a.shift.x / centre).round()).abs() < 1e-3);
        assert!((a.shift.x - 53.3).abs() <= centre * 0.5 + 1e-4 && (a.shift.y + 12.0).abs() <= centre * 0.5 + 1e-4);
        let b = plan(&water, Vec3::new(80.0, 5.0, 40.0)).unwrap();
        assert_eq!((a.layout, a.vertices), (b.layout, b.vertices), "moving never rebuilds the mesh");
        assert_ne!(a.shift, b.shift);

        let axis = graded_axis(quads, extent, centre);
        assert_eq!(axis.len(), quads as usize + 1);
        assert!((axis[0] + extent * 0.5).abs() < 1e-3 && (axis[quads as usize] - extent * 0.5).abs() < 1e-3);
        let m = quads as usize / 2;
        assert!(axis[m].abs() < 1e-6);
        let gaps: Vec<f32> = axis.windows(2).map(|w| w[1] - w[0]).collect();
        assert!((gaps[m] - centre).abs() < centre * 0.05, "centre cell {}", gaps[m]);
        assert!(gaps[m..].windows(2).all(|g| g[1] >= g[0] - 1e-5), "growing outward");
        assert!(*gaps.last().unwrap() < 8.0, "horizon cells {}", gaps.last().unwrap());

        let mesh = build(&water, a.shape);
        assert_eq!(mesh.vertices.len(), (quads as usize + 1).pow(2));
        assert!(faces_up(&mesh));
        let mid = &mesh.vertices[m * (quads as usize + 1) + m];
        assert_eq!(mid.position, [0.0, 0.0, 0.0]);
        assert!((mid.uv1[0] - centre).abs() < centre * 0.1, "spacing carried in uv1: {}", mid.uv1[0]);
    }

    #[test]
    fn polygon_lake_drops_cells_outside() {
        let water = Water { shape: vec![[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], resolution: 20, ..Default::default() };
        let m = build(&water, plan(&water, Vec3::ZERO).unwrap().shape);
        let full = 20 * 20 * 6;
        assert!(m.indices.len() > full / 3 && m.indices.len() < full * 2 / 3, "about half: {}", m.indices.len());
        assert!(faces_up(&m));
    }

    #[test]
    fn river_ribbon_follows_its_points() {
        let water = Water {
            body: BodyKind::River,
            river: vec![
                zestengine_core::RiverPoint { position: [0.0, 2.0, 0.0], width: 4.0, depth: 1.0, speed: 2.0 },
                zestengine_core::RiverPoint { position: [20.0, 1.0, 0.0], width: 4.0, depth: 1.0, speed: 2.0 },
                zestengine_core::RiverPoint { position: [40.0, 0.0, 8.0], width: 6.0, depth: 1.0, speed: 1.0 },
            ],
            ..Default::default()
        };
        let m = build(&water, plan(&water, Vec3::ZERO).unwrap().shape);
        let rows = river_samples(&water).len();
        assert_eq!(m.vertices.len(), rows * (RIVER_ACROSS as usize + 1));
        assert_eq!(m.indices.len(), (rows - 1) * RIVER_ACROSS as usize * 6);
        assert!(faces_up(&m));
        let first = &m.vertices[RIVER_ACROSS as usize / 2];
        assert_eq!(first.position, [0.0, 2.0, 0.0]);
        assert_eq!(first.color[2], 0.0, "centre line");
        assert!((first.color[0] - (2.0 / 10.0 + 0.5)).abs() < 1e-3, "flows along +X");
        assert_eq!(m.vertices[0].color[2], 1.0, "bank");
        assert!(plan(&Water { body: BodyKind::River, ..Default::default() }, Vec3::ZERO).is_none());
    }

    /// Hypothesis `basin_clip`: the drawn grid is the whole `size`
    /// rectangle at local Y 0. Nothing about colliders, a floor's extent or
    /// the baked walls reaches `plan`/`grid_mesh`, so an 8 x 8 m pool in a
    /// cube whose walls span |x|,|z| = 3.0..3.5 (the `cube_*` repro scenes)
    /// draws water over the wall tops and 0.5 m beyond the walls. Only a
    /// polygon `shape` (what "Fit to basin" writes) clips the triangles.
    #[test]
    fn basin_clip_grid_covers_wall_tops_and_outside_the_cube() {
        let water = Water { body: BodyKind::Pool, size: [8.0, 8.0], depth: 1.3, resolution: 32, ..Default::default() };
        let m = build(&water, plan(&water, Vec3::new(0.0, 3.0, 0.0)).unwrap().shape);
        assert_eq!(m.indices.len(), 32 * 32 * 6, "every cell of the rectangle is drawn");
        assert!(m.vertices.iter().all(|v| v.position[1] == 0.0), "the rest surface is local Y 0 (the basin box top)");
        let p = |i: u32| Vec2::new(m.vertices[i as usize].position[0], m.vertices[i as usize].position[2]);
        let centroids: Vec<Vec2> = m.indices.chunks(3).map(|t| (p(t[0]) + p(t[1]) + p(t[2])) / 3.0).collect();
        let band = |c: &Vec2| {
            let r = c.x.abs().max(c.y.abs());
            if r <= 3.0 { 0 } else if r <= 3.5 { 1 } else { 2 }
        };
        let inside = centroids.iter().filter(|c| band(c) == 0).count();
        let over_walls = centroids.iter().filter(|c| band(c) == 1).count();
        let outside = centroids.iter().filter(|c| band(c) == 2).count();
        eprintln!("[basin_clip] 8x8 grid, 32 quads: {inside} triangles inside the 6 m cube, {over_walls} over the wall tops, {outside} outside the walls");
        assert!(over_walls > 0 && outside > 0, "the mesh is drawn over the rim and outside the cube");
        assert_eq!(inside + over_walls + outside, 32 * 32 * 2);

        // The polygon outline is the only clip: a fitted 6 x 6 shape drops
        // everything past the inner faces.
        let fitted = Water { shape: vec![[-3.0, -3.0], [3.0, -3.0], [3.0, 3.0], [-3.0, 3.0]], ..water.clone() };
        let f = build(&fitted, plan(&fitted, Vec3::new(0.0, 3.0, 0.0)).unwrap().shape);
        let fp = |i: u32| Vec2::new(f.vertices[i as usize].position[0], f.vertices[i as usize].position[2]);
        let worst = f.indices.chunks(3).map(|t| ((fp(t[0]) + fp(t[1]) + fp(t[2])) / 3.0).abs().max_element()).fold(0.0f32, f32::max);
        assert!(worst <= 3.0 + 1e-4, "a polygon shape keeps the mesh inside the walls: {worst}");
    }
}
