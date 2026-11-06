//! Editor-time water interaction: while not playing there is no physics step,
//! so what's in GPU-simulated water is fed to it from here instead. Each
//! Rigid Body (or character), and any other collider while it is being moved,
//! crossing the surface displaces its collider bounds, moving at the speed it
//! was dragged since the last frame; still colliders through the surface are
//! walls the water reflects off. Only objects whose layer collides with the
//! water's count (the Layers matrix).

use std::collections::HashMap;

use zestengine_core::{
    BodyKind, BoxCollider, CapsuleCollider, Character, MeshCollider, ObjectId, RigidBody, SphereCollider, Water,
    WaterDisplacer,
};
use glam::{Vec2, Vec3};

use zestengine_engine::scene::LoadedScene;

/// Time constant (s) a dragged body's speed is smoothed over: gizmo drags
/// move in jerky per-frame steps.
const SPEED_SMOOTHING: f32 = 0.1;
/// Fastest a dragged body moves the water (m/s).
const MAX_DRAG_SPEED: f32 = 4.0;
/// Below this smoothed speed (m/s) a plain collider stands as a wall.
const STILL_SPEED: f32 = 0.02;

/// Feeds the editor's water simulations; keeps last frame's positions and
/// smoothed velocities.
#[derive(Default)]
pub struct EditorWater {
    last: HashMap<ObjectId, (Vec3, Vec3)>,
    walls: zestengine_engine::water_walls::WaterWalls,
}

impl EditorWater {
    pub fn reset(&mut self) {
        self.last.clear();
        self.walls.reset();
    }

    /// Replace each GPU-simulated water's displacers with the bodies in it
    /// now (`dt` the frame time).
    pub fn feed(&mut self, scene: &mut LoadedScene, dt: f32) {
        let world = scene.world_transforms();
        let active = scene.active_flags();
        let gpu_waters: Vec<usize> = (0..scene.objects.len())
            .filter(|&i| active[i])
            .filter(|&i| water_of(scene, i).is_some_and(|w| w.interactive && w.ripples.as_ref().is_some_and(|r| r.gpu)))
            .collect();
        if gpu_waters.is_empty() {
            self.last.clear();
            return;
        }
        let mut per_water: HashMap<usize, Vec<WaterDisplacer>> = gpu_waters.iter().map(|&w| (w, Vec::new())).collect();
        let mut seen = HashMap::new();
        for (i, o) in scene.objects.iter().enumerate() {
            let body = moves(o);
            if !active[i] || !(body || has_collider(o)) {
                continue;
            }
            let Some((lo, hi)) = collider_bounds(scene, i) else { continue };
            let pos = world[i].w_axis.truncate();
            let (prev, prev_vel) = self.last.get(&o.id).copied().unwrap_or((pos, Vec3::ZERO));
            let raw = if dt > 1e-5 { (pos - prev) / dt } else { Vec3::ZERO };
            let velocity = prev_vel.lerp(raw, 1.0 - (-dt.max(0.0) / SPEED_SMOOTHING).exp()).clamp_length_max(MAX_DRAG_SPEED);
            seen.insert(o.id, (pos, velocity));
            let wall = !body && velocity.length() < STILL_SPEED;
            let corners_world: [Vec3; 8] = std::array::from_fn(|k| {
                world[i].transform_point3(Vec3::new(
                    if k & 1 == 0 { lo.x } else { hi.x },
                    if k & 2 == 0 { lo.y } else { hi.y },
                    if k & 4 == 0 { lo.z } else { hi.z },
                ))
            });
            let centre = corners_world.iter().copied().sum::<Vec3>() / 8.0;
            for &wi in &gpu_waters {
                if !zestengine_engine::physics::layers_meet(scene, wi, i) {
                    continue;
                }
                let Some(water) = water_of(scene, wi) else { continue };
                let inv = world[wi].inverse();
                let lc = inv.transform_point3(centre);
                let xz = Vec2::new(lc.x, lc.z);
                if wall {
                    // Still: a wall (`WaterWalls`), not a push.
                    continue;
                }
                if !water.covers(xz) {
                    continue;
                }
                // Held in place in the editor: measured against the rest level, so
                // passing waves don't pump rings out of a still body.
                let Some(col) = water.column(xz) else { continue };
                let surface = col.surface;
                let corners = corners_world.map(|p| inv.transform_point3(p));
                let axis = inv.transform_vector3(world[i].transform_vector3(Vec3::X));
                let v = inv.transform_vector3(velocity);
                let rel = Vec2::new(v.x - col.flow.x, v.z - col.flow.z);
                let round = is_round(scene, i);
                if let Some(d) = WaterDisplacer::from_corners(&corners, axis, surface, water.sim_wave_depth.min(col.depth), rel, round) {
                    per_water.entry(wi).or_default().push(d);
                }
                break;
            }
        }
        self.last = seen;
        for (wi, list) in per_water {
            if let Some(w) = water_mut(scene, wi) {
                w.sim_feed.displacers = list;
            }
        }
        self.walls.update(scene, &world);
    }
}

fn has_collider(o: &zestengine_engine::scene::SceneObject) -> bool {
    o.components.iter().any(|c| {
        let a = c.as_any();
        a.is::<BoxCollider>() || a.is::<SphereCollider>() || a.is::<CapsuleCollider>() || a.downcast_ref::<MeshCollider>().is_some_and(|m| !m.is_trigger)
    })
}

fn water_of(scene: &LoadedScene, i: usize) -> Option<&Water> {
    scene.objects[i].components.iter().find_map(|c| c.as_any().downcast_ref::<Water>())
}

fn water_mut(scene: &mut LoadedScene, i: usize) -> Option<&mut Water> {
    scene.objects[i].components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<Water>())
}

/// Things that move the water: non-fixed Rigid Bodies and characters.
fn moves(o: &zestengine_engine::scene::SceneObject) -> bool {
    o.components.iter().any(|c| {
        let a = c.as_any();
        a.is::<Character>() || a.downcast_ref::<RigidBody>().is_some_and(|r| r.kind != BodyKind::Fixed)
    })
}

/// Round footprint: sphere and capsule colliders, and Mesh Colliders whose
/// mesh is a ball (the cached shape physics builds for it).
fn is_round(scene: &LoadedScene, i: usize) -> bool {
    scene.objects[i].components.iter().any(|c| {
        let a = c.as_any();
        a.is::<SphereCollider>()
            || a.is::<CapsuleCollider>()
            || a.downcast_ref::<MeshCollider>().is_some_and(|m| {
                let mode = zestengine_engine::physics::mesh_shape_mode(scene, i, m, true);
                zestengine_engine::physics::mesh_collider_shape(scene, i, mode)
                    .is_some_and(|s| matches!(*s, zestengine_engine::physics::MeshColliderShape::Ball(..)))
            })
    })
}

/// Local bounds of an object's colliders (mesh bounds for a Mesh Collider).
fn collider_bounds(scene: &LoadedScene, i: usize) -> Option<(Vec3, Vec3)> {
    let o = &scene.objects[i];
    let mut lo = Vec3::splat(f32::MAX);
    let mut hi = Vec3::splat(f32::MIN);
    let mut grow = |a: Vec3, b: Vec3| {
        lo = lo.min(a);
        hi = hi.max(b);
    };
    for c in &o.components {
        let a = c.as_any();
        if let Some(b) = a.downcast_ref::<BoxCollider>() {
            let (c, h) = (Vec3::from(b.center), Vec3::from(b.size) * 0.5);
            grow(c - h, c + h);
        } else if let Some(s) = a.downcast_ref::<SphereCollider>() {
            let c = Vec3::from(s.center);
            grow(c - Vec3::splat(s.radius), c + Vec3::splat(s.radius));
        } else if let Some(cap) = a.downcast_ref::<CapsuleCollider>() {
            let c = Vec3::from(cap.center);
            let half = cap.height * 0.5;
            let along = match cap.direction {
                zestengine_core::ColliderAxis::X => Vec3::X,
                zestengine_core::ColliderAxis::Y => Vec3::Y,
                zestengine_core::ColliderAxis::Z => Vec3::Z,
            };
            let ext = along * (half - cap.radius).max(0.0) + Vec3::splat(cap.radius);
            grow(c - ext, c + ext);
        } else if a.is::<MeshCollider>() {
            for r in o.render_slots() {
                let (a, b) = scene.mesh_aabb(r.mesh);
                grow(a, b);
            }
        } else if let Some(cc) = a.downcast_ref::<Character>().filter(|c| c.use_default_capsule && c.capsule_affects_water) {
            let c = Vec3::from(cc.center);
            let h = Vec3::new(cc.radius, cc.height * 0.5, cc.radius);
            grow(c - h, c + h);
        }
    }
    (lo.x <= hi.x).then_some((lo, hi)).or_else(|| {
        // An opted-out character must not silently fall back to its visual mesh.
        if o.components.iter().any(|c| c.as_any().is::<Character>()) { return None; }
        // A body with no collider yet: its mesh.
        let r = o.render?;
        Some(scene.mesh_aabb(r.mesh))
    })
}
