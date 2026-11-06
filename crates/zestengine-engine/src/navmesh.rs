//! Navigation: a height-aware walkable grid baked from the physics world, A*
//! path search with string pulling, and `NavAgent` steering.
//!
//! Bake (per `NavMeshSurface`): one downward ray per cell finds the ground; a
//! cell is walkable when the surface is flat enough and an upward ray finds
//! `agent_height` of headroom. Walkable cells next to blocked ones are eroded by
//! the agent radius. Neighbouring cells connect when their heights differ by at
//! most `step_height`, so wall tops and ledges stay separate.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use zestengine_core::{NavAgent, NavMeshSurface, NavQuery, ObjectId, PhysicsQuery, RayFilter};
use glam::{Quat, Vec3};

use crate::physics::PhysicsWorld;
use crate::scene::LoadedScene;

/// A baked walkable grid. `heights[z * w + x]` is the ground height, or NaN
/// when the cell isn't walkable.
#[derive(Clone, Debug)]
pub struct NavGrid {
    pub min: Vec3,
    pub cell: f32,
    pub w: i32,
    pub h: i32,
    pub heights: Vec<f32>,
    pub step: f32,
}

impl NavGrid {
    fn idx(&self, x: i32, z: i32) -> Option<usize> {
        (x >= 0 && z >= 0 && x < self.w && z < self.h).then(|| (z * self.w + x) as usize)
    }

    pub fn height(&self, x: i32, z: i32) -> Option<f32> {
        self.idx(x, z).map(|i| self.heights[i]).filter(|h| !h.is_nan())
    }

    pub fn cell_of(&self, p: Vec3) -> (i32, i32) {
        (
            ((p.x - self.min.x) / self.cell).floor() as i32,
            ((p.z - self.min.z) / self.cell).floor() as i32,
        )
    }

    pub fn center(&self, x: i32, z: i32) -> Option<Vec3> {
        let y = self.height(x, z)?;
        Some(Vec3::new(
            self.min.x + (x as f32 + 0.5) * self.cell,
            y,
            self.min.z + (z as f32 + 0.5) * self.cell,
        ))
    }

    fn connected(&self, a: (i32, i32), b: (i32, i32)) -> bool {
        match (self.height(a.0, a.1), self.height(b.0, b.1)) {
            (Some(ha), Some(hb)) => (ha - hb).abs() <= self.step,
            _ => false,
        }
    }

    pub fn contains(&self, p: Vec3) -> bool {
        let (x, z) = self.cell_of(p);
        self.idx(x, z).is_some()
    }

    /// Nearest walkable cell to `p` within `radius` cells (spiral search),
    /// preferring cells whose height is close to `p.y`.
    pub fn nearest(&self, p: Vec3, radius: i32) -> Option<(i32, i32)> {
        let (cx, cz) = self.cell_of(p);
        let mut best: Option<((i32, i32), f32)> = None;
        for r in 0..=radius {
            for dz in -r..=r {
                for dx in -r..=r {
                    if dx.abs() != r && dz.abs() != r {
                        continue;
                    }
                    let (x, z) = (cx + dx, cz + dz);
                    if let Some(c) = self.center(x, z) {
                        let d = c.distance_squared(p);
                        if best.is_none_or(|(_, b)| d < b) {
                            best = Some(((x, z), d));
                        }
                    }
                }
            }
            if best.is_some() {
                break;
            }
        }
        best.map(|(c, _)| c)
    }

    /// A* over 8-connected cells (no corner cutting), then string pulling.
    pub fn find_path(&self, from: Vec3, to: Vec3) -> Option<Vec<Vec3>> {
        let start = self.nearest(from, 8)?;
        let goal = self.nearest(to, 8)?;
        let h = |c: (i32, i32)| (((c.0 - goal.0).pow(2) + (c.1 - goal.1).pow(2)) as f32).sqrt();
        let mut open = BinaryHeap::new();
        let mut came: HashMap<(i32, i32), (i32, i32)> = HashMap::new();
        let mut g: HashMap<(i32, i32), f32> = HashMap::new();
        g.insert(start, 0.0);
        open.push(Node { cell: start, f: h(start) });
        let max_nodes = (self.w * self.h) as usize;
        let mut expanded = 0;
        while let Some(Node { cell, .. }) = open.pop() {
            if cell == goal {
                let mut cells = vec![cell];
                let mut c = cell;
                while let Some(&p) = came.get(&c) {
                    cells.push(p);
                    c = p;
                }
                cells.reverse();
                return Some(self.smooth(from, to, &cells));
            }
            expanded += 1;
            if expanded > max_nodes {
                break;
            }
            let gc = g[&cell];
            for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)] {
                let n = (cell.0 + dx, cell.1 + dz);
                if !self.connected(cell, n) {
                    continue;
                }
                if dx != 0 && dz != 0 && !(self.connected(cell, (cell.0 + dx, cell.1)) && self.connected(cell, (cell.0, cell.1 + dz))) {
                    continue;
                }
                let step = if dx != 0 && dz != 0 { std::f32::consts::SQRT_2 } else { 1.0 };
                let ng = gc + step;
                if g.get(&n).is_none_or(|&old| ng < old) {
                    g.insert(n, ng);
                    came.insert(n, cell);
                    open.push(Node { cell: n, f: ng + h(n) });
                }
            }
        }
        None
    }

    /// Walkable straight line between two cells (sampled per half cell).
    fn line_clear(&self, a: (i32, i32), b: (i32, i32)) -> bool {
        let (Some(pa), Some(pb)) = (self.center(a.0, a.1), self.center(b.0, b.1)) else {
            return false;
        };
        let steps = (((b.0 - a.0).abs().max((b.1 - a.1).abs())) * 2).max(1);
        let mut prev = a;
        for k in 1..=steps {
            let t = k as f32 / steps as f32;
            let p = pa.lerp(pb, t);
            let c = self.cell_of(p);
            if c != prev {
                if !self.connected(prev, c) {
                    return false;
                }
                // Diagonal moves must not clip a blocked corner.
                if c.0 != prev.0 && c.1 != prev.1
                    && !(self.connected(prev, (c.0, prev.1)) && self.connected(prev, (prev.0, c.1)))
                {
                    return false;
                }
                prev = c;
            }
        }
        true
    }

    fn smooth(&self, from: Vec3, to: Vec3, cells: &[(i32, i32)]) -> Vec<Vec3> {
        let mut keep = vec![cells[0]];
        let mut anchor = 0;
        let mut i = 1;
        while i < cells.len() {
            if !self.line_clear(cells[anchor], cells[i]) {
                anchor = i - 1;
                keep.push(cells[anchor]);
            }
            i += 1;
        }
        let last = *cells.last().unwrap();
        if keep.last() != Some(&last) {
            keep.push(last);
        }
        let mut path: Vec<Vec3> = keep.iter().filter_map(|&(x, z)| self.center(x, z)).collect();
        // Exact endpoints (snapped onto the walkable height).
        if let Some(first) = path.first_mut() {
            *first = Vec3::new(from.x, first.y, from.z);
        }
        if let Some(goal) = path.last_mut()
            && self.cell_of(to) == last
        {
            *goal = Vec3::new(to.x, goal.y, to.z);
        }
        path
    }
}

#[derive(PartialEq)]
struct Node {
    cell: (i32, i32),
    f: f32,
}
impl Eq for Node {}
impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        other.f.total_cmp(&self.f)
    }
}
impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Bake a surface's grid by ray casting the physics world.
pub fn bake(physics: &PhysicsWorld, center: Vec3, s: &NavMeshSurface) -> NavGrid {
    let cell = s.cell_size.max(0.05);
    let half = Vec3::from(s.size) * 0.5;
    let min = center - half;
    let w = ((s.size[0] / cell).ceil() as i32).clamp(1, 4096);
    let h = ((s.size[2] / cell).ceil() as i32).clamp(1, 4096);
    let filter = RayFilter {
        layer_mask: s.layer_mask,
        include_triggers: false,
        exclude: None,
    };
    let min_normal_y = s.max_slope.clamp(0.0, 89.0).to_radians().cos();
    let top = center.y + half.y;
    let mut heights = vec![f32::NAN; (w * h) as usize];
    for z in 0..h {
        for x in 0..w {
            let px = min.x + (x as f32 + 0.5) * cell;
            let pz = min.z + (z as f32 + 0.5) * cell;
            let Some(hit) = physics.raycast(Vec3::new(px, top, pz), Vec3::NEG_Y, s.size[1], &filter) else {
                continue;
            };
            if hit.normal.y < min_normal_y {
                continue;
            }
            let up = hit.point + Vec3::Y * 0.02;
            if physics.raycast(up, Vec3::Y, s.agent_height, &filter).is_some() {
                continue;
            }
            heights[(z * w + x) as usize] = hit.point.y;
        }
    }
    let mut grid = NavGrid { min, cell, w, h, heights, step: s.step_height };
    erode(&mut grid, (s.agent_radius / cell).ceil() as i32);
    grid
}

/// Drop walkable cells within `r` cells of a blocked cell or a ledge.
fn erode(grid: &mut NavGrid, r: i32) {
    if r <= 0 {
        return;
    }
    let edge: Vec<bool> = (0..grid.h)
        .flat_map(|z| (0..grid.w).map(move |x| (x, z)))
        .map(|(x, z)| {
            grid.height(x, z).is_some()
                && [(1, 0), (-1, 0), (0, 1), (0, -1)]
                    .iter()
                    .any(|(dx, dz)| !grid.connected((x, z), (x + dx, z + dz)))
        })
        .collect();
    let mut out = grid.heights.clone();
    for z in 0..grid.h {
        for x in 0..grid.w {
            if !edge[(z * grid.w + x) as usize] {
                continue;
            }
            for dz in -(r - 1)..=(r - 1) {
                for dx in -(r - 1)..=(r - 1) {
                    if dx * dx + dz * dz <= (r - 1) * (r - 1)
                        && let Some(i) = grid.idx(x + dx, z + dz)
                    {
                        out[i] = f32::NAN;
                    }
                }
            }
        }
    }
    grid.heights = out;
}

/// Baked grids for the scene's surfaces + agent movement.
#[derive(Default)]
pub struct NavSystem {
    grids: Vec<(ObjectId, NavGrid)>,
    baked: bool,
}

impl NavSystem {
    pub fn needs_bake(&self) -> bool {
        !self.baked
    }

    pub fn invalidate(&mut self) {
        self.baked = false;
    }

    pub fn bake(&mut self, scene: &LoadedScene, physics: &PhysicsWorld) {
        self.baked = true;
        self.grids.clear();
        let active = scene.active_flags();
        let world = scene.world_transforms();
        for (i, o) in scene.objects.iter().enumerate() {
            if !active[i] {
                continue;
            }
            if let Some(s) = o.components.iter().find_map(|c| c.as_any().downcast_ref::<NavMeshSurface>()) {
                self.grids.push((o.id, bake(physics, world[i].w_axis.truncate(), s)));
            }
        }
    }

    pub fn grids(&self) -> impl Iterator<Item = &NavGrid> {
        self.grids.iter().map(|(_, g)| g)
    }

    fn grid_for(&self, p: Vec3) -> Option<&NavGrid> {
        self.grids.iter().map(|(_, g)| g).find(|g| g.contains(p))
    }

    /// Re-plan and steer every active agent by `dt`.
    pub fn move_agents(&self, scene: &mut LoadedScene, physics: &mut PhysicsWorld, dt: f32) {
        if self.grids.is_empty() || dt <= 0.0 {
            return;
        }
        let active = scene.active_flags();
        for (i, &is_active) in active.iter().enumerate() {
            if !is_active {
                continue;
            }
            let pos = scene.world_transform(i).w_axis.truncate();
            let has_cc = scene.objects[i]
                .components
                .iter()
                .any(|c| c.as_any().downcast_ref::<zestengine_core::Character>().is_some_and(|c| c.use_default_capsule));
            let Some(agent) = scene.objects[i]
                .components
                .iter_mut()
                .find_map(|c| c.as_any_mut().downcast_mut::<NavAgent>())
            else {
                continue;
            };
            if agent.state.replan {
                agent.state.replan = false;
                let path = agent.state.destination.and_then(|d| self.find_path(pos, d));
                agent.state.unreachable = path.is_none();
                agent.state.path = path.unwrap_or_default();
                agent.state.next = 1.min(agent.state.path.len());
            }
            let s = &mut agent.state;
            if s.destination.is_none() || s.next >= s.path.len() {
                s.velocity = Vec3::ZERO;
                continue;
            }
            let goal = s.path[s.next];
            let flat = Vec3::new(goal.x - pos.x, 0.0, goal.z - pos.z);
            let last = s.next + 1 == s.path.len();
            let reach = if last { agent.stopping_distance.max(0.02) } else { 0.15 };
            if flat.length() <= reach {
                s.next += 1;
                if s.next >= s.path.len() {
                    s.destination = None;
                    s.velocity = Vec3::ZERO;
                }
                continue;
            }
            let step = flat.normalize() * (agent.speed * dt).min(flat.length());
            s.velocity = step / dt;
            let turn = agent.turn_speed;
            if has_cc {
                // Small downward bias keeps the controller grounded on slopes.
                physics.move_character(scene, i, step + Vec3::NEG_Y * 0.05);
            } else {
                let mut world = pos + step;
                world.y = goal.y + (pos.y - goal.y) * 0.5;
                let local = match scene.objects[i].parent {
                    Some(p) => scene.world_transform(p).inverse().transform_point3(world),
                    None => world,
                };
                scene.objects[i].translation = local;
            }
            if turn > 0.0 {
                let target = Quat::from_rotation_y((-step.x).atan2(-step.z));
                let o = &mut scene.objects[i];
                let angle = o.rotation.angle_between(target);
                let max = (turn * dt).to_radians();
                o.rotation = if angle <= max { target } else { o.rotation.slerp(target, max / angle) };
            }
        }
    }
}

/// Overlay mesh of a grid's walkable cells (editor preview).
pub fn preview_mesh(grid: &NavGrid, color: [f32; 4]) -> zestengine_render::MeshData {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let half = grid.cell * 0.45;
    for z in 0..grid.h {
        for x in 0..grid.w {
            let Some(c) = grid.center(x, z) else {
                continue;
            };
            let base = vertices.len() as u32;
            for (dx, dz) in [(-half, -half), (half, -half), (half, half), (-half, half)] {
                vertices.push(zestengine_render::Vertex {
                    position: [c.x + dx, c.y + 0.03, c.z + dz],
                    normal: [0.0, 1.0, 0.0],
                    uv: [0.5, 0.5],
                    color,
                    tangent: [1.0, 0.0, 0.0, 1.0],
                    uv1: [0.0, 0.0],
                    joints: [0; 4],
                    weights: [0.0; 4],
                });
            }
            indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
        }
    }
    zestengine_render::MeshData { vertices, indices, has_lightmap_uv: false }
}

impl zestengine_core::NavQuery for NavSystem {
    fn find_path(&self, from: Vec3, to: Vec3) -> Option<Vec<Vec3>> {
        self.grid_for(from).or_else(|| self.grid_for(to))?.find_path(from, to)
    }

    fn nearest_walkable(&self, p: Vec3) -> Option<Vec3> {
        let g = self.grid_for(p)?;
        let (x, z) = g.nearest(p, 16)?;
        g.center(x, z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::SceneObject;
    use zestengine_assets::ObjectSource;
    use zestengine_core::{BoxCollider, Component};

    fn obj(scene: &mut LoadedScene, pos: Vec3, components: Vec<Box<dyn Component>>) {
        scene.objects.push(SceneObject {
            net_role: Default::default(),
            component_roles: Default::default(),
            id: ObjectId::new(),
            name: String::new(),
            render: None,
            extra_render: Vec::new(),
            source: ObjectSource::Empty,
            enabled: true,
            static_geometry: false,
            layer: 0,
            parent: None,
            translation: pos,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            components,
        });
    }

    fn boxc(size: [f32; 3]) -> Box<dyn Component> {
        Box::new(BoxCollider { size, ..Default::default() })
    }

    /// 20 x 20 floor with a wall across x = 0 that has a gap near z = +8.
    fn level() -> (LoadedScene, PhysicsWorld) {
        let mut scene = LoadedScene::empty();
        obj(&mut scene, Vec3::new(0.0, -0.5, 0.0), vec![boxc([20.0, 1.0, 20.0])]);
        obj(&mut scene, Vec3::new(0.0, 1.0, -2.0), vec![boxc([0.5, 2.0, 16.0])]);
        obj(
            &mut scene,
            Vec3::ZERO,
            vec![Box::new(NavMeshSurface {
                size: [20.0, 6.0, 20.0],
                ..Default::default()
            })],
        );
        let mut pw = PhysicsWorld::build(&scene);
        pw.step();
        (scene, pw)
    }

    #[test]
    fn bakes_floor_and_blocks_wall() {
        let (scene, pw) = level();
        let mut nav = NavSystem::default();
        nav.bake(&scene, &pw);
        let g = &nav.grids[0].1;
        let (fx, fz) = g.cell_of(Vec3::new(5.0, 0.0, 5.0));
        assert!(g.height(fx, fz).is_some_and(|h| h.abs() < 0.01), "floor walkable at y=0");
        let (wx, wz) = g.cell_of(Vec3::new(0.0, 0.0, -2.0));
        assert!(g.height(wx, wz).is_none(), "wall (and its top) not reachable as floor");
        let (ex, ez) = g.cell_of(Vec3::new(0.0, 0.0, 9.9));
        assert!(g.height(ex, ez).is_none(), "level edge eroded by the agent radius");
    }

    #[test]
    fn path_goes_around_the_wall_through_the_gap() {
        let (scene, pw) = level();
        let mut nav = NavSystem::default();
        nav.bake(&scene, &pw);
        let from = Vec3::new(-5.0, 0.0, -5.0);
        let to = Vec3::new(5.0, 0.0, -5.0);
        let path = nav.find_path(from, to).expect("reachable via the gap");
        assert!(path.iter().any(|p| p.z > 6.0), "detours through the gap: {path:?}");
        assert!(path.len() < 12, "string-pulled to few waypoints: {}", path.len());
        assert!((path[0] - from).length() < 0.6 && (path.last().unwrap() - to).length() < 0.6);
    }

    #[test]
    fn agent_walks_to_destination() {
        let (mut scene, mut pw) = level();
        let mut agent = NavAgent::default();
        agent.set_destination(Vec3::new(5.0, 0.0, -5.0));
        obj(&mut scene, Vec3::new(-5.0, 0.0, -5.0), vec![Box::new(agent)]);
        let mut nav = NavSystem::default();
        nav.bake(&scene, &pw);
        for _ in 0..600 {
            nav.move_agents(&mut scene, &mut pw, 1.0 / 60.0);
        }
        let p = scene.objects[3].translation;
        assert!((p - Vec3::new(5.0, 0.0, -5.0)).length() < 0.3, "arrived at {p}");
    }
}
