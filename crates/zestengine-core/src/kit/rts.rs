//! Strategy (RTS) pieces: the camera rig, box / click selection with move
//! orders in formation, selectable units and a resource store.

use glam::{EulerRot, Quat, Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::{ComponentCtx, ObjectId, ObjectRef, TypedComponent, UiRect};

/// Camera pose orbiting `focus` on the ground: `yaw` / `pitch` in degrees
/// (pitch looks down), `distance` along the view ray.
pub fn rts_camera_pose(focus: Vec3, yaw: f32, pitch: f32, distance: f32) -> (Vec3, Quat) {
    let rot = Quat::from_euler(EulerRot::YXZ, yaw.to_radians(), -pitch.to_radians(), 0.0);
    (focus + rot * Vec3::Z * distance, rot)
}

/// Screen-edge pan direction: x = right, y = forward (top edge), each -1, 0 or 1.
pub fn edge_pan(pointer: Option<Vec2>, size: Vec2, margin: f32) -> Vec2 {
    let Some(p) = pointer else {
        return Vec2::ZERO;
    };
    if size.x < 1.0 || size.y < 1.0 {
        return Vec2::ZERO;
    }
    let x = if p.x <= margin {
        -1.0
    } else if p.x >= size.x - margin {
        1.0
    } else {
        0.0
    };
    let y = if p.y <= margin {
        1.0
    } else if p.y >= size.y - margin {
        -1.0
    } else {
        0.0
    };
    Vec2::new(x, y)
}

/// Normalized (min, max) corners of the rectangle spanned by two drag points.
pub fn screen_rect(a: Vec2, b: Vec2) -> (Vec2, Vec2) {
    (a.min(b), a.max(b))
}

/// True when `p` lies inside the rectangle spanned by `min`..`max` (inclusive).
pub fn rect_contains(min: Vec2, max: Vec2, p: Vec2) -> bool {
    p.x >= min.x && p.x <= max.x && p.y >= min.y && p.y <= max.y
}

/// Grid formation offsets for `count` units: x = right, y = back from the
/// formation center. Rows are as square as possible; a short last row is centered.
pub fn formation_offsets(count: usize, spacing: f32) -> Vec<Vec2> {
    if count == 0 {
        return Vec::new();
    }
    let cols = (count as f32).sqrt().ceil() as usize;
    let rows = count.div_ceil(cols);
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let (r, c) = (i / cols, i % cols);
        let in_row = cols.min(count - r * cols);
        let x = (c as f32 - (in_row as f32 - 1.0) * 0.5) * spacing;
        let y = (r as f32 - (rows as f32 - 1.0) * 0.5) * spacing;
        out.push(Vec2::new(x, y));
    }
    out
}

/// World slots around `target` for a formation facing `facing` (xz).
pub fn formation_slots(target: Vec3, facing: Vec3, count: usize, spacing: f32) -> Vec<Vec3> {
    let f = facing.with_y(0.0).normalize_or(Vec3::NEG_Z);
    let right = Vec3::new(-f.z, 0.0, f.x);
    formation_offsets(count, spacing)
        .into_iter()
        .map(|o| target + right * o.x - f * o.y)
        .collect()
}

/// RTS camera: WASD / arrow + screen-edge panning, wheel zoom, Q/E rotation,
/// clamped to a rectangle of the map. Put it on the camera object (a root).
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct RtsCamera {
    /// Yaw in degrees.
    pub yaw: f32,
    /// Pitch in degrees (looks down at the ground).
    pub pitch: f32,
    /// Distance from `focus` along the view ray (m).
    pub distance: f32,
    /// Closest zoom (m).
    pub min_distance: f32,
    /// Farthest zoom (m).
    pub max_distance: f32,
    /// Meters per second at a distance of 20.
    pub pan_speed: f32,
    /// Pan when the pointer nears the screen edge.
    pub edge_pan: bool,
    /// Pixels from the view edge that start edge panning.
    pub edge_margin: f32,
    /// Zoom change per full axis step (m).
    pub zoom_step: f32,
    /// Degrees per second.
    pub rotate_speed: f32,
    /// Pan clamp: minimum focus x / z.
    pub bounds_min: [f32; 2],
    /// Pan clamp: maximum focus x / z.
    pub bounds_max: [f32; 2],
    /// Follow rate (1/s); 0 snaps.
    pub smoothing: f32,
    /// Input action for pan (Axis2).
    pub move_action: String,
    /// Input action for zoom (Axis1).
    pub zoom_action: String,
    /// Input action for rotation (Axis1).
    pub rotate_action: String,
    /// The point the camera orbits (runtime).
    #[serde(skip)]
    pub focus: Vec3,
    /// Distance being zoomed toward (runtime).
    #[serde(skip)]
    pub target_distance: f32,
    /// Started (runtime).
    #[serde(skip)]
    pub started: bool,
}

impl Default for RtsCamera {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 55.0,
            distance: 24.0,
            min_distance: 8.0,
            max_distance: 50.0,
            pan_speed: 18.0,
            edge_pan: true,
            edge_margin: 12.0,
            zoom_step: 3.0,
            rotate_speed: 90.0,
            bounds_min: [-40.0, -40.0],
            bounds_max: [40.0, 40.0],
            smoothing: 10.0,
            move_action: "Move".into(),
            zoom_action: "Zoom".into(),
            rotate_action: "Rotate".into(),
            focus: Vec3::ZERO,
            target_distance: 24.0,
            started: false,
        }
    }
}

impl TypedComponent for RtsCamera {
    const NAME: &'static str = "RTS Camera";

    fn start(&mut self, ctx: &mut ComponentCtx) {
        let t = ctx.self_transform();
        let f = t.forward();
        self.focus = if f.y < -0.05 {
            let k = -t.translation.y / f.y;
            (t.translation + f * k).with_y(0.0)
        } else {
            t.translation.with_y(0.0)
        };
        self.target_distance = self.distance;
        ctx.set_active_camera(ctx.self_id().into());
        ctx.set_cursor_locked(false);
        self.started = true;
    }

    fn update(&mut self, ctx: &mut ComponentCtx) {
        if !self.started {
            TypedComponent::start(self, ctx);
        }
        let dt = ctx.unscaled_dt.min(0.1);
        let typing = ctx.ui_text_editing();
        let mut mv = if typing { Vec2::ZERO } else { ctx.input.axis2(&self.move_action) };
        if self.edge_pan {
            mv += edge_pan(ctx.pointer_position(), ctx.screen_size(), self.edge_margin);
        }
        let mv = mv.clamp_length_max(1.0);
        let yaw_rot = Quat::from_rotation_y(self.yaw.to_radians());
        let speed = self.pan_speed * (self.distance / 20.0).max(0.25);
        self.focus += (yaw_rot * Vec3::X * mv.x + yaw_rot * Vec3::NEG_Z * mv.y) * speed * dt;
        self.focus.x = self.focus.x.clamp(self.bounds_min[0], self.bounds_max[0]);
        self.focus.z = self.focus.z.clamp(self.bounds_min[1], self.bounds_max[1]);
        if !typing {
            self.yaw += ctx.input.axis(&self.rotate_action) * self.rotate_speed * dt;
            if !ctx.pointer_over_ui() {
                self.target_distance -= ctx.input.axis(&self.zoom_action) * self.zoom_step;
            }
        }
        self.target_distance = self.target_distance.clamp(self.min_distance, self.max_distance);
        let k = 1.0 - (-self.smoothing.max(0.0) * dt).exp();
        self.distance += (self.target_distance - self.distance) * if self.smoothing > 0.0 { k } else { 1.0 };
        let (pos, rot) = rts_camera_pose(self.focus, self.yaw, self.pitch, self.distance);
        ctx.set_world_position(pos);
        *ctx.rotation = rot;
    }
}

/// A unit the [`RtsSelection`] can pick. Shows `ring` while selected.
#[derive(Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Selectable {
    /// Object shown while selected (e.g. a ring mesh).
    pub ring: ObjectRef,
    /// Only selections of the same team pick it (0 = any).
    pub team: u32,
    /// Selected (runtime).
    #[serde(skip)]
    pub selected: bool,
    /// The ring's last shown state (runtime).
    #[serde(skip)]
    pub shown: Option<bool>,
}

impl TypedComponent for Selectable {
    const NAME: &'static str = "Selectable";

    fn update(&mut self, ctx: &mut ComponentCtx) {
        if self.shown != Some(self.selected) {
            if self.ring.is_set() {
                ctx.set_active(self.ring, self.selected);
            }
            self.shown = Some(self.selected);
        }
    }
}

/// RTS selection + orders: left-drag draws a box (`box_visual`, a UI element
/// under a screen canvas) and selects every [`Selectable`] inside it, left click
/// selects one, Shift adds, right click sends the selection to the clicked
/// point in a grid formation through their Nav Agents.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct RtsSelection {
    /// Input action for selecting.
    pub select_action: String,
    /// Input action for move orders.
    pub command_action: String,
    /// Input action that adds to the selection.
    pub add_action: String,
    /// Pixels the pointer must travel before a click becomes a box.
    pub drag_threshold: f32,
    /// UI element drawn as the selection box.
    pub box_visual: ObjectRef,
    /// Reference height of `box_visual`'s canvas (Scale with screen, match height).
    pub ui_reference_height: f32,
    /// Spacing between units in a formation (m).
    pub formation_spacing: f32,
    /// Shown briefly where a move order lands (a root object).
    pub move_marker: ObjectRef,
    /// Only Selectables of the same team (0 = any).
    pub team: u32,
    /// Currently selected units (runtime).
    #[serde(skip)]
    pub selected: Vec<ObjectId>,
    /// Pointer position the drag started at (runtime).
    #[serde(skip)]
    pub drag_start: Option<Vec2>,
    /// A box drag is in progress (runtime).
    #[serde(skip)]
    pub dragging: bool,
    /// Seconds the move marker stays visible (runtime).
    #[serde(skip)]
    pub marker_timer: f32,
}

impl Default for RtsSelection {
    fn default() -> Self {
        Self {
            select_action: "Select".into(),
            command_action: "Command".into(),
            add_action: "AddSelect".into(),
            drag_threshold: 6.0,
            box_visual: ObjectRef::NONE,
            ui_reference_height: 1080.0,
            formation_spacing: 1.6,
            move_marker: ObjectRef::NONE,
            team: 0,
            selected: Vec::new(),
            drag_start: None,
            dragging: false,
            marker_timer: 0.0,
        }
    }
}

impl RtsSelection {
    fn candidates(&self, ctx: &ComponentCtx) -> Vec<ObjectId> {
        super::objects_with::<Selectable>(ctx)
            .into_iter()
            .filter(|id| ctx.is_active(*id))
            .filter(|id| {
                let team = ctx.get::<Selectable>(*id).map(|s| s.team).unwrap_or(0);
                self.team == 0 || team == 0 || team == self.team
            })
            .collect()
    }

    fn set_selection(&mut self, ctx: &mut ComponentCtx, ids: Vec<ObjectId>, add: bool) {
        let mut next = if add { self.selected.clone() } else { Vec::new() };
        for id in ids {
            if !next.contains(&id) {
                next.push(id);
            }
        }
        for id in &self.selected {
            if !next.contains(id) {
                ctx.modify::<Selectable>(*id, |s| s.selected = false);
            }
        }
        for id in &next {
            ctx.modify::<Selectable>(*id, |s| s.selected = true);
        }
        self.selected = next;
    }

    fn show_box(&self, ctx: &mut ComponentCtx, a: Vec2, b: Vec2) {
        if !self.box_visual.is_set() {
            return;
        }
        let (min, max) = screen_rect(a, b);
        let k = self.ui_reference_height / ctx.screen_size().y.max(1.0);
        let (pos, size) = ((min * k).to_array(), ((max - min) * k).to_array());
        ctx.modify::<UiRect>(self.box_visual, move |r| {
            r.anchor_min = [0.0, 0.0];
            r.anchor_max = [0.0, 0.0];
            r.pivot = [0.0, 0.0];
            r.position = pos;
            r.size = size;
        });
        ctx.set_active(self.box_visual, true);
    }

    fn command(&mut self, ctx: &mut ComponentCtx) {
        self.selected.retain(|id| ctx.index_of(*id).is_some());
        if self.selected.is_empty() {
            return;
        }
        let Some((origin, dir)) = ctx.pointer_ray() else {
            return;
        };
        let Some(hit) = ctx.raycast(origin, dir, 1000.0) else {
            return;
        };
        let positions: Vec<(ObjectId, Vec3)> =
            self.selected.iter().filter_map(|id| ctx.position_of((*id).into()).map(|p| (*id, p))).collect();
        if positions.is_empty() {
            return;
        }
        let centroid = positions.iter().map(|(_, p)| *p).sum::<Vec3>() / positions.len() as f32;
        let slots = formation_slots(hit.point, hit.point - centroid, positions.len(), self.formation_spacing);
        let mut free: Vec<(ObjectId, Vec3)> = positions;
        for slot in slots {
            let Some((k, _)) = free
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.1.distance_squared(slot).total_cmp(&b.1.1.distance_squared(slot)))
            else {
                break;
            };
            let (unit, _) = free.swap_remove(k);
            let dest = ctx.nearest_walkable(slot).unwrap_or(slot);
            ctx.set_destination(unit, dest);
        }
        if self.move_marker.is_set() {
            ctx.set_local_transform(self.move_marker, Some(hit.point + Vec3::Y * 0.03), None, None);
            ctx.set_active(self.move_marker, true);
            self.marker_timer = 0.8;
        }
    }
}

impl TypedComponent for RtsSelection {
    const NAME: &'static str = "RTS Selection";

    fn start(&mut self, ctx: &mut ComponentCtx) {
        for r in [self.box_visual, self.move_marker] {
            if r.is_set() {
                ctx.set_active(r, false);
            }
        }
    }

    fn update(&mut self, ctx: &mut ComponentCtx) {
        if self.marker_timer > 0.0 {
            self.marker_timer -= ctx.unscaled_dt;
            if self.marker_timer <= 0.0 && self.move_marker.is_set() {
                ctx.set_active(self.move_marker, false);
            }
        }
        let pointer = ctx.pointer_position();
        if ctx.input.pressed(&self.select_action) && !ctx.pointer_over_ui() {
            self.drag_start = pointer;
            self.dragging = false;
        }
        if let (Some(start), Some(p)) = (self.drag_start, pointer)
            && ctx.input.down(&self.select_action)
        {
            if !self.dragging && start.distance(p) > self.drag_threshold {
                self.dragging = true;
            }
            if self.dragging {
                self.show_box(ctx, start, p);
            }
        }
        if ctx.input.released(&self.select_action)
            && let Some(start) = self.drag_start.take()
        {
            let add = ctx.input.down(&self.add_action);
            let end = pointer.unwrap_or(start);
            if self.dragging {
                let (min, max) = screen_rect(start, end);
                let picked: Vec<ObjectId> = self
                    .candidates(ctx)
                    .into_iter()
                    .filter(|id| {
                        ctx.position_of((*id).into())
                            .and_then(|p| ctx.world_to_screen(p))
                            .is_some_and(|s| rect_contains(min, max, s))
                    })
                    .collect();
                self.set_selection(ctx, picked, add);
            } else {
                let hit = ctx.screen_ray(end).and_then(|(o, d)| ctx.raycast(o, d, 1000.0));
                let candidates = self.candidates(ctx);
                let picked: Vec<ObjectId> =
                    hit.map(|h| h.object).filter(|id| candidates.contains(id)).into_iter().collect();
                self.set_selection(ctx, picked, add);
            }
            self.dragging = false;
            if self.box_visual.is_set() {
                ctx.set_active(self.box_visual, false);
            }
        }
        if ctx.input.pressed(&self.command_action) && !ctx.pointer_over_ui() {
            self.command(ctx);
        }
    }
}

/// One named resource: its amount, passive income and optional capacity.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ResourceAmount {
    /// Resource name.
    pub name: String,
    /// Current amount.
    pub amount: f32,
    /// Amount added per second.
    pub income_per_sec: f32,
    /// 0 = unlimited.
    pub capacity: f32,
}

impl Default for ResourceAmount {
    fn default() -> Self {
        Self { name: "Gold".into(), amount: 0.0, income_per_sec: 0.0, capacity: 0.0 }
    }
}

/// Named amounts (gold, wood, supply) with passive income. Spend from code
/// with `ctx.modify::<ResourceStore>(store, |s| { s.spend("Gold", 50.0); })`.
#[derive(Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ResourceStore {
    /// The resources, in declaration order.
    pub resources: Vec<ResourceAmount>,
}

impl ResourceStore {
    /// The named resource.
    pub fn get(&self, name: &str) -> Option<&ResourceAmount> {
        self.resources.iter().find(|r| r.name == name)
    }

    /// Add `amount` (capped at capacity when set); creates the resource if
    /// missing.
    pub fn add(&mut self, name: &str, amount: f32) {
        match self.resources.iter_mut().find(|r| r.name == name) {
            Some(r) => {
                r.amount += amount;
                if r.capacity > 0.0 {
                    r.amount = r.amount.min(r.capacity);
                }
            }
            None => self.resources.push(ResourceAmount { name: name.into(), amount, ..Default::default() }),
        }
    }

    /// Take `amount` if there is enough; returns whether it was spent.
    pub fn spend(&mut self, name: &str, amount: f32) -> bool {
        match self.resources.iter_mut().find(|r| r.name == name) {
            Some(r) if r.amount >= amount => {
                r.amount -= amount;
                true
            }
            _ => false,
        }
    }
}

impl TypedComponent for ResourceStore {
    const NAME: &'static str = "Resource Store";

    fn update(&mut self, ctx: &mut ComponentCtx) {
        for r in &mut self.resources {
            r.amount += r.income_per_sec * ctx.dt;
            if r.capacity > 0.0 {
                r.amount = r.amount.min(r.capacity);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drag_rect_normalizes_and_contains() {
        let (min, max) = screen_rect(Vec2::new(300.0, 50.0), Vec2::new(100.0, 250.0));
        assert_eq!(min, Vec2::new(100.0, 50.0));
        assert_eq!(max, Vec2::new(300.0, 250.0));
        assert!(rect_contains(min, max, Vec2::new(200.0, 100.0)));
        assert!(rect_contains(min, max, Vec2::new(100.0, 250.0)), "edges count");
        assert!(!rect_contains(min, max, Vec2::new(99.0, 100.0)));
        assert!(!rect_contains(min, max, Vec2::new(200.0, 251.0)));
    }

    #[test]
    fn formation_is_centered_and_spaced() {
        assert_eq!(formation_offsets(0, 2.0), Vec::<Vec2>::new());
        assert_eq!(formation_offsets(1, 2.0), vec![Vec2::ZERO]);
        for n in [2usize, 3, 4, 5, 7, 9, 12] {
            let offs = formation_offsets(n, 1.5);
            assert_eq!(offs.len(), n);
            let center = offs.iter().copied().sum::<Vec2>() / n as f32;
            assert!(center.x.abs() < 1e-4, "n={n} not centered sideways: {center:?}");
            assert!(center.y.abs() <= 0.75 + 1e-4, "n={n} block roughly centered: {center:?}");
            for i in 0..n {
                for j in i + 1..n {
                    assert!(offs[i].distance(offs[j]) >= 1.5 - 1e-4, "n={n} slots {i},{j} overlap");
                }
            }
        }
        let four = formation_offsets(4, 2.0);
        assert!(four.contains(&Vec2::new(-1.0, -1.0)) && four.contains(&Vec2::new(1.0, 1.0)));
    }

    #[test]
    fn formation_slots_face_the_move_direction() {
        let slots = formation_slots(Vec3::new(10.0, 0.0, 0.0), Vec3::X, 2, 2.0);
        // Facing +X: the two units stand side by side along Z.
        assert!((slots[0].x - 10.0).abs() < 1e-4 && (slots[1].x - 10.0).abs() < 1e-4);
        assert!((slots[0].z - slots[1].z).abs() > 1.9);
    }

    #[test]
    fn camera_pose_looks_at_focus() {
        let focus = Vec3::new(3.0, 0.0, -2.0);
        let (pos, rot) = rts_camera_pose(focus, 30.0, 55.0, 20.0);
        assert!((pos.distance(focus) - 20.0).abs() < 1e-3);
        let fwd = rot * Vec3::NEG_Z;
        assert!((pos + fwd * 20.0 - focus).length() < 1e-3);
        assert!(pos.y > 10.0, "camera above the ground");
    }

    #[test]
    fn edge_pan_by_margin() {
        let size = Vec2::new(800.0, 600.0);
        assert_eq!(edge_pan(Some(Vec2::new(400.0, 300.0)), size, 10.0), Vec2::ZERO);
        assert_eq!(edge_pan(Some(Vec2::new(2.0, 2.0)), size, 10.0), Vec2::new(-1.0, 1.0));
        assert_eq!(edge_pan(Some(Vec2::new(799.0, 599.0)), size, 10.0), Vec2::new(1.0, -1.0));
        assert_eq!(edge_pan(None, size, 10.0), Vec2::ZERO);
    }

    #[test]
    fn resources_spend_and_cap() {
        let mut s = ResourceStore::default();
        s.add("Gold", 100.0);
        assert!(s.spend("Gold", 60.0));
        assert!(!s.spend("Gold", 60.0));
        s.resources[0].capacity = 50.0;
        s.add("Gold", 100.0);
        assert_eq!(s.get("Gold").unwrap().amount, 50.0);
    }
}
