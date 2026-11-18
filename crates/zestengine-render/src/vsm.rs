use glam::{Mat4, Vec2, Vec3};

pub const PAGES_PER_LEVEL: usize = 4;
pub const MIN_LEVELS: usize = 2;
pub const MAX_LEVELS: usize = 6;
pub const BASE_RADIUS: f32 = 8.0;
const GUARD_TEXELS: f32 = 8.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PageKey {
    pub level: u8,
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

pub fn level_radius(level: usize) -> f32 {
    BASE_RADIUS * (1u32 << level.min(MAX_LEVELS - 1)) as f32
}

pub fn page_size(level: usize) -> f32 {
    level_radius(level) * 2.0
}

fn depth_half(level: usize) -> f32 {
    (page_size(level) * 4.0).max(128.0)
}

fn depth_step(level: usize) -> f32 {
    depth_half(level) * 0.5
}

fn grid_offset(level: usize) -> f32 {
    page_size(level) * 0.381_966
}

pub fn levels_for(distance: f32) -> usize {
    let wanted = if distance.is_finite() { distance.max(1.0) } else { level_radius(MAX_LEVELS - 1) };
    let mut levels = MIN_LEVELS;
    while levels < MAX_LEVELS && level_radius(levels - 1) < wanted {
        levels += 1;
    }
    levels
}

pub fn layer_count(distance: f32) -> usize {
    levels_for(distance) * PAGES_PER_LEVEL
}

pub fn light_basis(dir: Vec3, up: Vec3) -> Mat4 {
    Mat4::look_to_rh(Vec3::ZERO, dir, up)
}

pub fn page_matrix(basis: Mat4, key: PageKey, dim: u32) -> Mat4 {
    let level = key.level as usize;
    let size = page_size(level);
    let half_depth = depth_half(level);
    let offset = grid_offset(level);
    let centre = Vec3::new((key.x as f32 + 0.5) * size + offset, (key.y as f32 + 0.5) * size + offset, key.z as f32 * depth_step(level));
    let half = size * 0.5 * (1.0 + 2.0 * GUARD_TEXELS / dim.max(64) as f32);
    let view = Mat4::from_translation(Vec3::new(-centre.x, -centre.y, -(centre.z + half_depth))) * basis;
    Mat4::orthographic_rh(-half, half, -half, half, 0.05, half_depth * 2.0) * view
}

#[derive(Clone, Copy, Debug)]
pub struct PageCamera {
    pub position: Vec3,
    pub inv_view: Mat4,
    pub tan_h: f32,
    pub tan_v: f32,
    pub perspective: bool,
}

impl PageCamera {
    pub fn from_matrices(view: Mat4, proj: Mat4, position: Vec3) -> Self {
        let perspective = proj.w_axis.w.abs() < 0.5;
        let safe = |v: f32| if v.is_finite() && v.abs() > 1e-6 { (1.0 / v).abs() } else { 1.0 };
        Self { position, inv_view: view.inverse(), tan_h: safe(proj.x_axis.x), tan_v: safe(proj.y_axis.y), perspective }
    }

    fn footprint(&self, basis: Mat4, depth: f32) -> (Vec2, Vec2) {
        let centre = basis.transform_point3(self.position).truncate();
        if !self.perspective {
            return (centre - Vec2::splat(depth), centre + Vec2::splat(depth));
        }
        let (w, h) = (depth * self.tan_h, depth * self.tan_v);
        let mut min = centre;
        let mut max = centre;
        for corner in [Vec3::new(-w, h, -depth), Vec3::new(w, h, -depth), Vec3::new(w, -h, -depth), Vec3::new(-w, -h, -depth)] {
            let p = basis.transform_point3(self.inv_view.transform_point3(corner)).truncate();
            min = min.min(p);
            max = max.max(p);
        }
        (min, max)
    }
}

pub fn needed_pages(basis: Mat4, camera: &PageCamera, level: usize) -> Vec<PageKey> {
    let radius = level_radius(level);
    let size = page_size(level);
    let centre = basis.transform_point3(camera.position);
    let (seen_min, seen_max) = camera.footprint(basis, radius);
    let min = (centre.truncate() - Vec2::splat(radius)).max(seen_min);
    let max = (centre.truncate() + Vec2::splat(radius)).min(seen_max).max(min);
    let offset = grid_offset(level);
    let cell = |v: f32| ((v - offset) / size).floor() as i32;
    let z = (centre.z / depth_step(level)).round() as i32;
    let (x0, x1) = (cell(min.x), cell(max.x).min(cell(min.x) + 1));
    let (y0, y1) = (cell(min.y), cell(max.y).min(cell(min.y) + 1));
    let mut keys = Vec::with_capacity(PAGES_PER_LEVEL);
    for y in y0..=y1 {
        for x in x0..=x1 {
            keys.push(PageKey { level: level as u8, x, y, z });
        }
    }
    keys
}

#[derive(Default)]
pub struct Pager {
    slots: [[Option<PageKey>; PAGES_PER_LEVEL]; MAX_LEVELS],
    used: [[u64; PAGES_PER_LEVEL]; MAX_LEVELS],
    frame: u64,
    basis: Option<[f32; 16]>,
}

impl Pager {
    pub fn plan(&mut self, basis: Mat4, camera: &PageCamera, levels: usize, dim: u32) -> Vec<Mat4> {
        let levels = levels.clamp(MIN_LEVELS, MAX_LEVELS);
        self.frame += 1;
        let key = basis.to_cols_array();
        if self.basis != Some(key) {
            self.basis = Some(key);
            self.slots = Default::default();
        }
        let mut matrices = vec![Mat4::ZERO; levels * PAGES_PER_LEVEL];
        for level in 0..levels {
            let needed = needed_pages(basis, camera, level);
            let mut slot_of: Vec<Option<usize>> = needed.iter().map(|key| self.slots[level].iter().position(|held| *held == Some(*key))).collect();
            for k in 0..needed.len() {
                if slot_of[k].is_some() {
                    continue;
                }
                let free = (0..PAGES_PER_LEVEL)
                    .filter(|slot| !slot_of.contains(&Some(*slot)))
                    .min_by_key(|slot| (self.slots[level][*slot].is_some(), self.used[level][*slot]));
                if let Some(slot) = free {
                    self.slots[level][slot] = Some(needed[k]);
                    slot_of[k] = Some(slot);
                }
            }
            for (key, slot) in needed.iter().zip(slot_of.iter()) {
                if let Some(slot) = slot {
                    self.used[level][*slot] = self.frame;
                    matrices[level * PAGES_PER_LEVEL + slot] = page_matrix(basis, *key, dim);
                }
            }
        }
        matrices
    }

    #[cfg(test)]
    pub fn resident(&self) -> usize {
        self.slots.iter().flatten().flatten().count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera(position: Vec3, forward: Vec3) -> PageCamera {
        let view = Mat4::look_to_rh(position, forward, Vec3::Y);
        let proj = Mat4::perspective_rh(60f32.to_radians(), 16.0 / 9.0, 0.1, 500.0);
        PageCamera::from_matrices(view, proj, position)
    }

    fn sun() -> Mat4 {
        light_basis(Vec3::new(-0.4, -1.0, -0.3).normalize(), Vec3::Y)
    }

    fn covers(matrix: Mat4, p: Vec3) -> bool {
        let c = matrix * p.extend(1.0);
        c.w > 0.0 && c.x.abs() <= c.w && c.y.abs() <= c.w && c.z >= 0.0 && c.z <= c.w
    }

    #[test]
    fn levels_double_until_the_shadow_distance_is_covered() {
        assert_eq!(levels_for(5.0), MIN_LEVELS);
        assert_eq!(levels_for(60.0), 4);
        assert_eq!(levels_for(64.0), 4);
        assert_eq!(levels_for(65.0), 5);
        assert_eq!(levels_for(10_000.0), MAX_LEVELS);
        assert_eq!(levels_for(f32::NAN), MAX_LEVELS);
        assert_eq!(layer_count(60.0), 16);
        assert_eq!(page_size(0), 16.0);
        assert_eq!(page_size(3), 128.0);
    }

    #[test]
    fn every_visible_point_near_the_camera_is_inside_a_page_and_the_finest_comes_first() {
        let basis = sun();
        let cam = camera(Vec3::new(3.0, 1.7, -41.0), Vec3::new(0.3, -0.1, -1.0).normalize());
        let mut pager = Pager::default();
        let pages = pager.plan(basis, &cam, 5, 2048);
        assert_eq!(pages.len(), 20);
        let inv = cam.inv_view;
        for depth in [0.5f32, 3.0, 7.0, 15.0, 30.0, 60.0, 110.0] {
            for (u, v) in [(0.0f32, 0.0f32), (0.9, 0.0), (-0.9, 0.0), (0.0, 0.9), (0.0, -0.9), (0.9, 0.9), (-0.9, -0.9)] {
                let p = inv.transform_point3(Vec3::new(u * depth * cam.tan_h, v * depth * cam.tan_v, -depth));
                let first = pages.iter().position(|m| *m != Mat4::ZERO && covers(*m, p)).unwrap_or_else(|| panic!("no page covers depth {depth} at ({u}, {v})"));
                let level = first / PAGES_PER_LEVEL;
                let reach = (basis.transform_point3(p) - basis.transform_point3(cam.position)).truncate().abs().max_element();
                let finest_reaching = (0..MAX_LEVELS).find(|k| level_radius(*k) >= reach).unwrap_or(MAX_LEVELS - 1);
                assert!(level <= finest_reaching, "a point {reach} m away was given level {level}, level {finest_reaching} reaches it");
            }
        }
    }

    #[test]
    fn pages_do_not_change_while_the_camera_moves_and_turns_inside_them() {
        let basis = sun();
        let mut pager = Pager::default();
        let start = Vec3::new(100.3, 2.0, 50.2);
        let first = pager.plan(basis, &camera(start, Vec3::NEG_Z), 4, 2048);
        let mut changed_slots = 0usize;
        let mut previous = first.clone();
        for step in 1..=40 {
            let position = start + Vec3::new(0.01, 0.0, -0.01) * step as f32;
            let forward = Vec3::new((step as f32 * 0.002).sin(), 0.0, -1.0).normalize();
            let next = pager.plan(basis, &camera(position, forward), 4, 2048);
            for (slot, (a, b)) in previous.iter().zip(next.iter()).enumerate() {
                if a != b && *a != Mat4::ZERO && *b != Mat4::ZERO {
                    panic!("slot {slot} was given another page while the old one was still needed");
                }
                if a != b {
                    changed_slots += 1;
                }
            }
            previous = next;
        }
        assert!(changed_slots <= 4, "a 0.4 m walk with a small turn touched {changed_slots} slots");
        let kept = first.iter().zip(previous.iter()).filter(|(a, b)| **a != Mat4::ZERO && a == b).count();
        let needed_at_start = first.iter().filter(|m| **m != Mat4::ZERO).count();
        assert!(kept + 2 >= needed_at_start, "{kept} of {needed_at_start} pages survived the walk unchanged");
    }

    #[test]
    fn a_page_left_and_entered_again_returns_to_the_same_slot_with_the_same_matrix() {
        let basis = sun();
        let mut pager = Pager::default();
        let cam_a = camera(Vec3::new(7.9, 2.0, 7.9), Vec3::NEG_Z);
        let cam_b = camera(Vec3::new(40.0, 2.0, -40.0), Vec3::X);
        let a1 = pager.plan(basis, &cam_a, 3, 2048);
        let _ = pager.plan(basis, &cam_b, 3, 2048);
        let a2 = pager.plan(basis, &cam_a, 3, 2048);
        let level0 = |plan: &[Mat4]| plan[..PAGES_PER_LEVEL].iter().filter(|m| **m != Mat4::ZERO).copied().collect::<Vec<_>>();
        for m in level0(&a1) {
            assert!(level0(&a2).contains(&m), "a level-0 page changed its matrix after coming back");
        }
        assert!(pager.resident() <= 3 * PAGES_PER_LEVEL);
    }

    #[test]
    fn turning_the_light_drops_every_page() {
        let mut pager = Pager::default();
        let cam = camera(Vec3::new(1.0, 2.0, 3.0), Vec3::NEG_Z);
        let before = pager.plan(sun(), &cam, 3, 2048);
        let turned = light_basis(Vec3::new(0.5, -1.0, 0.1).normalize(), Vec3::Y);
        let after = pager.plan(turned, &cam, 3, 2048);
        for (a, b) in before.iter().zip(after.iter()) {
            assert!(*a == Mat4::ZERO || a != b);
        }
    }

    #[test]
    fn a_small_caster_is_inside_few_pages() {
        let basis = light_basis(Vec3::new(-1.0, -1.0, 0.0).normalize(), Vec3::Y);
        let eye = Vec3::new(2.5, 3.0, 6.0);
        let view = Mat4::look_at_rh(eye, eye + Vec3::new(-0.35, -0.45, -1.0), Vec3::Y);
        let cam = PageCamera::from_matrices(view, Mat4::perspective_rh(1.0, 16.0 / 9.0, 0.05, 200.0), eye);
        let mut pager = Pager::default();
        let pages = pager.plan(basis, &cam, 4, 1536);
        let needed = pages.iter().filter(|m| **m != Mat4::ZERO).count();
        let (centre, radius) = (Vec3::new(0.1, 0.5, 0.0), 0.9f32);
        assert!((1..=16).contains(&needed));
        let mut touched = Vec::new();
        for (slot, m) in pages.iter().enumerate() {
            if *m == Mat4::ZERO {
                continue;
            }
            let planes = crate::frustum_planes(*m);
            let outside = planes.iter().enumerate().any(|(k, p)| k != 4 && p.truncate().dot(centre) + p.w < -radius);
            if !outside {
                touched.push(slot);
            }
        }
        assert!(touched.len() >= 4, "every level has a page under the caster: {touched:?} of {needed}");
        assert!(touched.len() <= 6, "a 0.9 m caster at the world origin is in {} of {needed} pages: {touched:?}", touched.len());
    }

    #[test]
    fn neighbouring_pages_overlap_by_the_guard_band() {
        let basis = sun();
        let left = page_matrix(basis, PageKey { level: 1, x: 0, y: 0, z: 0 }, 2048);
        let right = page_matrix(basis, PageKey { level: 1, x: 1, y: 0, z: 0 }, 2048);
        let on_edge = basis.inverse().transform_point3(Vec3::new(page_size(1) + grid_offset(1), page_size(1) * 0.5 + grid_offset(1), 0.0));
        let uv = |m: Mat4| {
            let c = m * on_edge.extend(1.0);
            (c.x / c.w) * 0.5 + 0.5
        };
        let texel = 1.0 / 2048.0;
        assert!(uv(left) < 1.0 - 6.0 * texel && uv(left) > 0.9, "the shared edge is inside the left page: {}", uv(left));
        assert!(uv(right) > 6.0 * texel && uv(right) < 0.1, "and inside the right page: {}", uv(right));
    }
}
