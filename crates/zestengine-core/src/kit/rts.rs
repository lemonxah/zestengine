//! Strategy (RTS) pieces: the camera rig, box / click selection with move
//! orders in formation, selectable units and a resource store.

use glam::{EulerRot, Quat, Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::{MotionContext, ObjectId, ObjectRef, TypedComponent, UiRect};
use crate::ecs::{ObjectContext, Owner, Stage, SystemContext, SystemWorld};

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
/// clamped to a rectangle of the map. Put it on the camera object.
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
}

impl RtsCamera {
    fn start_native(&mut self, ctx: &mut ObjectContext<'_, '_>) {
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

    fn update_native(&mut self, ctx: &mut ObjectContext<'_, '_>) {
        if !self.started {
            self.start_native(ctx);
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
        ctx.set_world_rotation(rot);
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
}

impl Selectable {

    fn update_native(&mut self, ctx: &mut ObjectContext<'_, '_>) {
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
    /// Shown briefly at the world position where a move order lands.
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
    /// This instance has initialized its marker/box (runtime).
    #[serde(skip)]
    pub started: bool,
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
            started: false,
        }
    }
}

impl RtsSelection {
    fn candidates(&self, ctx: &ObjectContext<'_, '_>) -> Vec<ObjectId> {
        ctx.objects_with::<Selectable>()
            .into_iter()
            .filter(|id| ctx.is_active(*id))
            .filter(|id| {
                let team = ctx.get::<Selectable>(*id).map(|s| s.team).unwrap_or(0);
                self.team == 0 || team == 0 || team == self.team
            })
            .collect()
    }

    fn set_selection(&mut self, ctx: &mut ObjectContext<'_, '_>, ids: Vec<ObjectId>, add: bool) {
        let mut next = if add { self.selected.clone() } else { Vec::new() };
        for id in ids {
            if !next.contains(&id) {
                next.push(id);
            }
        }
        for id in &self.selected {
            if !next.contains(id) {
                ctx.edit::<Selectable>(*id, |s| s.selected = false);
            }
        }
        for id in &next {
            ctx.edit::<Selectable>(*id, |s| s.selected = true);
        }
        self.selected = next;
    }

    fn show_box(&self, ctx: &mut ObjectContext<'_, '_>, a: Vec2, b: Vec2) {
        if !self.box_visual.is_set() {
            return;
        }
        let (min, max) = screen_rect(a, b);
        let k = self.ui_reference_height / ctx.screen_size().y.max(1.0);
        let (pos, size) = ((min * k).to_array(), ((max - min) * k).to_array());
        ctx.edit::<UiRect>(self.box_visual, move |r| {
            r.anchor_min = [0.0, 0.0];
            r.anchor_max = [0.0, 0.0];
            r.pivot = [0.0, 0.0];
            r.position = pos;
            r.size = size;
        });
        ctx.set_active(self.box_visual, true);
    }

    fn command(&mut self, ctx: &mut ObjectContext<'_, '_>) {
        self.selected.retain(|id| ctx.entity_of(*id).is_some());
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
            ctx.set_world_transform_of(self.move_marker, Some(hit.point + Vec3::Y * 0.03), None);
            ctx.set_active(self.move_marker, true);
            self.marker_timer = 0.8;
        }
    }
}

impl TypedComponent for RtsSelection {
    const NAME: &'static str = "RTS Selection";
}

impl RtsSelection {
    fn start_native(&mut self, ctx: &mut ObjectContext<'_, '_>) {
        for r in [self.box_visual, self.move_marker] {
            if r.is_set() {
                ctx.set_active(r, false);
            }
        }
        self.started = true;
    }

    fn update_native(&mut self, ctx: &mut ObjectContext<'_, '_>) {
        if !self.started { self.start_native(ctx); }
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

/// Named amounts (gold, wood, supply) with passive income. Native systems spend
/// resources through checked mutable ECS borrows.
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
}

impl ResourceStore {

    fn update_native(&mut self, ctx: &mut ObjectContext<'_, '_>) {
        for r in &mut self.resources {
            r.amount += r.income_per_sec * ctx.dt;
            if r.capacity > 0.0 {
                r.amount = r.amount.min(r.capacity);
            }
        }
    }
}

pub(crate) fn register_systems(registry: &mut crate::ComponentRegistry) {
    registry.register_ecs::<RtsCamera>();
    registry.register_ecs::<RtsSelection>();
    registry.register_ecs::<Selectable>();
    registry.register_ecs::<ResourceStore>();
    registry.register_system_if("zestengine::rts_start", Stage::Start, initialize, |world|
        world.query::<&RtsCamera>().iter().any(|(_, value)| !value.started)
        || world.query::<&RtsSelection>().iter().any(|(_, value)| !value.started));
    registry.register_system_for::<RtsCamera>("zestengine::rts_camera", Stage::Update, camera);
    registry.register_system_for::<RtsSelection>("zestengine::rts_selection", Stage::Gameplay, selection);
    registry.register_system_for::<ResourceStore>("zestengine::resources", Stage::Gameplay, resources);
    registry.register_system_for::<Selectable>("zestengine::selection_rings", Stage::LateUpdate, rings);
}

fn initialize(world: &SystemWorld<'_>, context: &mut SystemContext<'_>) {
    macro_rules! start { ($t:ty) => {
        let mut query = world.query::<(&Owner, &mut $t)>();
        let mut actors: Vec<_> = query.iter().filter(|(_, (owner, value))| !value.started && context.is_active(world, owner.0)).collect();
        actors.sort_unstable_by_key(|(instance, (owner, _))| context.instance_order::<$t>(world, owner.0, *instance));
        for (_, (owner, value)) in actors { value.start_native(&mut ObjectContext::new(world, owner.0, context)); }
    }; }
    start!(RtsCamera);
    start!(RtsSelection);
}

fn camera(world: &SystemWorld<'_>, context: &mut SystemContext<'_>) {
    let mut query = world.query::<(&Owner, &mut RtsCamera)>();
    let mut actors: Vec<_> = query.iter().filter(|(_, (owner, _))| context.is_active(world, owner.0)).collect();
    actors.sort_unstable_by_key(|(instance, (owner, _))| context.instance_order::<RtsCamera>(world, owner.0, *instance));
    for (_, (owner, value)) in actors { value.update_native(&mut ObjectContext::new(world, owner.0, context)); }
}

fn selection(world: &SystemWorld<'_>, context: &mut SystemContext<'_>) {
    let mut query = world.query::<(&Owner, &mut RtsSelection)>();
    let mut actors: Vec<_> = query.iter().filter(|(_, (owner, _))| context.is_active(world, owner.0)).collect();
    actors.sort_unstable_by_key(|(instance, (owner, _))| context.instance_order::<RtsSelection>(world, owner.0, *instance));
    for (_, (owner, value)) in actors { value.update_native(&mut ObjectContext::new(world, owner.0, context)); }
}

fn resources(world: &SystemWorld<'_>, context: &mut SystemContext<'_>) {
    for (_, (owner, value)) in world.query::<(&Owner, &mut ResourceStore)>().iter() {
        if context.is_active(world, owner.0) { value.update_native(&mut ObjectContext::new(world, owner.0, context)); }
    }
}

fn rings(world: &SystemWorld<'_>, context: &mut SystemContext<'_>) {
    let mut query = world.query::<(&Owner, &mut Selectable)>();
    let mut actors: Vec<_> = query.iter().filter(|(_, (owner, _))| context.is_active(world, owner.0)).collect();
    actors.sort_unstable_by_key(|(instance, (owner, _))| context.instance_order::<Selectable>(world, owner.0, *instance));
    for (_, (owner, value)) in actors { value.update_native(&mut ObjectContext::new(world, owner.0, context)); }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ActionValue, ComponentCommand, ComponentRegistry, InputState, NavAgent, NavQuery, PhysicsQuery, RayFilter, RayHit, ResourceAmount, Transform};
    use crate::ecs::{Enabled, SceneWorld};
    use std::cell::Cell;

    struct Hits(Cell<RayHit>);
    impl PhysicsQuery for Hits {
        fn raycast(&self, _: Vec3, _: Vec3, _: f32, _: &RayFilter) -> Option<RayHit> { Some(self.0.get()) }
        fn overlap_sphere(&self, _: Vec3, _: f32, _: &RayFilter) -> Vec<ObjectId> { Vec::new() }
        fn velocity(&self, _: ObjectId) -> Option<Vec3> { None }
        fn grounded(&self, _: ObjectId) -> bool { true }
    }
    struct Navigation;
    impl NavQuery for Navigation {
        fn find_path(&self, from: Vec3, to: Vec3) -> Option<Vec<Vec3>> { Some(vec![from, to]) }
        fn nearest_walkable(&self, point: Vec3) -> Option<Vec3> { Some(point.with_y(0.2)) }
    }
    fn run(scene: &mut SceneWorld, stage: Stage, input: &InputState, screen: &crate::ScreenInfo, physics: Option<&dyn PhysicsQuery>, dt: f32) -> Vec<ComponentCommand> {
        scene.propagate().unwrap();
        let mut commands = Vec::new();
        let mut context = SystemContext::new(input, &mut commands);
        context.dt = dt;
        context.unscaled_dt = 0.1;
        context.services.screen = Some(screen);
        context.services.physics = physics;
        context.services.nav = Some(&Navigation);
        ComponentRegistry::with_builtins().run_systems(stage, scene, &mut context);
        assert!(!commands.iter().any(|command| matches!(command, ComponentCommand::Modify { .. })));
        for command in &commands {
            if let ComponentCommand::SetActive(id, enabled) = command {
                scene.get_mut::<Enabled>(scene.entity(*id).unwrap()).unwrap().0 = *enabled;
            }
        }
        commands
    }

    #[test]
    fn native_rts_camera_keeps_world_pose_under_a_parent_and_initializes_once() {
        let mut scene = SceneWorld::new();
        let parent = scene.spawn("Parent");
        *scene.get_mut::<Transform>(parent).unwrap() = Transform { translation: Vec3::new(10.0, 2.0, -3.0), rotation: Quat::from_rotation_y(0.7), scale: Vec3::splat(1.5) };
        let camera = scene.spawn("Camera");
        scene.set_parent(camera, Some(parent), false).unwrap();
        scene.get_mut::<Transform>(camera).unwrap().translation = Vec3::Y * 5.0;
        scene.insert(camera, RtsCamera { smoothing: 0.0, edge_pan: false, ..Default::default() }).unwrap();
        let input = InputState::from_actions(vec![("Move".into(), ActionValue { axis2: Vec2::X, ..Default::default() })], Vec2::ZERO);
        let screen = crate::ScreenInfo::default();
        let first = run(&mut scene, Stage::Start, &input, &screen, None, 0.1);
        assert!(first.iter().any(|command| matches!(command, ComponentCommand::SetCursorLocked(false))));
        run(&mut scene, Stage::Update, &input, &screen, None, 0.1);
        let data = scene.get::<RtsCamera>(camera).unwrap();
        let expected = rts_camera_pose(data.focus, data.yaw, data.pitch, data.distance);
        drop(data);
        scene.with_world(|world| {
            let actual = world.world_transform(camera).unwrap();
            assert!(actual.w_axis.truncate().abs_diff_eq(expected.0, 1e-4));
            assert!(actual.to_scale_rotation_translation().1.dot(expected.1).abs() > 1.0 - 1e-5);
        });
        assert!(run(&mut scene, Stage::Start, &input, &screen, None, 0.1).is_empty());
    }

    #[test]
    fn native_rts_drag_selection_rings_ui_and_formation_orders_write_live_data() {
        let mut scene = SceneWorld::new();
        let parent = scene.spawn("Marker parent");
        scene.get_mut::<Transform>(parent).unwrap().translation = Vec3::X * 10.0;
        let marker = scene.spawn("Marker");
        scene.set_parent(marker, Some(parent), false).unwrap();
        let visual = scene.spawn("Selection box");
        scene.insert(visual, UiRect::default()).unwrap();
        let controller = scene.spawn("Controller");
        scene.insert(controller, RtsSelection { team: 1, box_visual: scene.object_id(visual).unwrap().into(), move_marker: scene.object_id(marker).unwrap().into(), ui_reference_height: 100.0, ..Default::default() }).unwrap();
        let mut units = Vec::new();
        let mut rings = Vec::new();
        for x in [-0.2, 0.2] {
            let ring = scene.spawn("Ring");
            let unit = scene.spawn("Unit");
            scene.get_mut::<Transform>(unit).unwrap().translation = Vec3::X * x;
            scene.insert(unit, Selectable { ring: scene.object_id(ring).unwrap().into(), team: 1, ..Default::default() }).unwrap();
            scene.insert(unit, NavAgent::default()).unwrap();
            units.push(unit); rings.push(ring);
        }
        let enemy = scene.spawn("Enemy");
        scene.insert(enemy, Selectable { team: 2, ..Default::default() }).unwrap();
        let screen = crate::ScreenInfo { size: Vec2::splat(100.0), pointer: Some(Vec2::splat(80.0)), view: glam::Mat4::IDENTITY, proj: glam::Mat4::IDENTITY, ..Default::default() };
        let empty = InputState::default();
        run(&mut scene, Stage::Start, &empty, &screen, None, 0.1);
        scene.get_mut::<RtsSelection>(controller).unwrap().drag_start = Some(Vec2::splat(20.0));
        let dragging = InputState::from_actions(vec![("Select".into(), ActionValue { down: true, ..Default::default() })], Vec2::ZERO);
        run(&mut scene, Stage::Gameplay, &dragging, &screen, None, 0.1);
        assert_eq!(scene.get::<UiRect>(visual).unwrap().size, [60.0, 60.0]);
        let released = InputState::from_actions(vec![("Select".into(), ActionValue { just_released: true, ..Default::default() })], Vec2::ZERO);
        run(&mut scene, Stage::Gameplay, &released, &screen, None, 0.1);
        assert_eq!(scene.get::<RtsSelection>(controller).unwrap().selected.len(), 2);
        for &unit in &units { assert!(scene.get::<Selectable>(unit).unwrap().selected); }
        assert!(!scene.get::<Selectable>(enemy).unwrap().selected);
        run(&mut scene, Stage::LateUpdate, &empty, &screen, None, 0.1);
        for ring in rings { assert!(scene.get::<Enabled>(ring).unwrap().0); }
        assert!(!scene.get::<Enabled>(visual).unwrap().0);
        let point = Vec3::new(3.0, 0.0, 4.0);
        let hits = Hits(Cell::new(RayHit { object: scene.object_id(parent).unwrap(), point, normal: Vec3::Y, distance: 1.0 }));
        let command = InputState::from_actions(vec![("Command".into(), ActionValue { just_pressed: true, down: true, ..Default::default() })], Vec2::ZERO);
        run(&mut scene, Stage::Gameplay, &command, &screen, Some(&hits), 0.1);
        let destinations: Vec<_> = units.iter().map(|&unit| scene.get::<NavAgent>(unit).unwrap().state.destination.unwrap()).collect();
        assert_ne!(destinations[0], destinations[1]);
        assert!((destinations.iter().copied().sum::<Vec3>() / 2.0).abs_diff_eq(point.with_y(0.2), 1e-5));
        scene.with_world(|world| assert!(world.world_transform(marker).unwrap().w_axis.truncate().abs_diff_eq(point + Vec3::Y * 0.03, 1e-5)));
    }

    #[test]
    fn native_resource_income_respects_pause_and_inherited_activation_without_copying_the_store() {
        let mut scene = SceneWorld::new();
        let parent = scene.spawn("Parent");
        let store = scene.spawn("Store");
        scene.set_parent(store, Some(parent), false).unwrap();
        scene.insert(store, ResourceStore { resources: vec![ResourceAmount { amount: 9.0, income_per_sec: 20.0, capacity: 10.0, ..Default::default() }] }).unwrap();
        let allocation = scene.get::<ResourceStore>(store).unwrap().resources.as_ptr();
        let input = InputState::default();
        let screen = crate::ScreenInfo::default();
        run(&mut scene, Stage::Gameplay, &input, &screen, None, 0.0);
        assert_eq!(scene.get::<ResourceStore>(store).unwrap().resources[0].amount, 9.0);
        scene.get_mut::<Enabled>(parent).unwrap().0 = false;
        run(&mut scene, Stage::Gameplay, &input, &screen, None, 0.1);
        assert_eq!(scene.get::<ResourceStore>(store).unwrap().resources[0].amount, 9.0);
        scene.get_mut::<Enabled>(parent).unwrap().0 = true;
        run(&mut scene, Stage::Gameplay, &input, &screen, None, 0.1);
        let data = scene.get::<ResourceStore>(store).unwrap();
        assert_eq!(data.resources[0].amount, 10.0);
        assert_eq!(data.resources.as_ptr(), allocation);
    }

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
