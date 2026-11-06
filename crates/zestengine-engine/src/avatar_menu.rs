//! The avatar's radial expression menu, for desktop and VR.
//!
//! [`RadialMenu`] is the state machine: it reads a thumbstick / mouse
//! direction and select / back presses ([`MenuInput`]) and writes the avatar's
//! parameters through its [`AvatarParamStore`] (Toggle, Button held for at
//! least 0.2 s, Sub Menu and puppet "page open" parameters, and the axes of
//! Radial / Two-Axis / Four-Axis puppets). It touches nothing else and has no
//! UI; [`RadialView`] is what a frame shows, and [`paint_radial`] draws that
//! with egui, shared by the desktop overlay and the VR panel.
//!
//! [`AvatarMenuHost`] finds the local avatar's `Avatar Menu` + parameters in
//! the scene and runs the state machine for it. It costs nothing while the
//! menu is closed except one cached-index check per frame.
//!
//! Slices: a page of `n` entries is `n` equal slices clockwise from the top;
//! a sub page has a Back slice first. The centre circle closes the menu (root)
//! or goes back (sub page).

use egui::{Align2, Color32, FontId, Pos2, Shape, Stroke, Vec2};
use glam::Vec2 as GVec2;
use zestengine_core::{AvatarMenu, AvatarMenuData, AvatarMenuItem, AvatarMenuNode, AvatarParamStore, AvatarParameters, MenuItemKind, ObjectId};

use crate::scene::LoadedScene;

/// Stick / mouse offsets shorter than this select nothing.
const DEAD_ZONE: f32 = 0.35;
/// A Button stays pressed at least this long (seconds).
const MIN_BUTTON_TIME: f64 = 0.2;
/// Puppet radial axis only follows the stick beyond this.
const PUPPET_DEAD: f32 = 0.2;

/// One frame of input, already reduced to what the menu needs.
#[derive(Clone, Copy, Debug, Default)]
pub struct MenuInput {
    /// The menu is held open (desktop key held) or toggled open (VR).
    pub open: bool,
    /// Stick / mouse offset from the centre, x right and y up, each -1..1.
    pub dir: GVec2,
    /// Select was pressed this frame.
    pub select: bool,
    /// Select was released this frame (ends a Button press).
    pub select_released: bool,
    /// Back was pressed this frame.
    pub back: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Held {
    slot: usize,
    since: f64,
    /// Released: clear `slot` once `MIN_BUTTON_TIME` passed.
    released: bool,
}

#[derive(Clone, Debug)]
struct Puppet {
    /// Index of the puppet's entry on the current page.
    idx: usize,
    kind: MenuItemKind,
    /// Slots of the puppet's axes (`usize::MAX` = not declared).
    axes: [usize; 4],
    open_slot: Option<usize>,
}

/// The menu state machine.
#[derive(Default)]
pub struct RadialMenu {
    /// Child indices from the root page to the current page.
    path: Vec<usize>,
    open: bool,
    highlighted: Option<usize>,
    held: Option<Held>,
    puppet: Option<Puppet>,
    /// "Page open" slots set for each open Sub Menu level, to clear on exit.
    open_slots: Vec<Option<usize>>,
}

/// What a slice shows.
#[derive(Clone, Debug, PartialEq)]
pub struct SliceView {
    pub label: String,
    pub badge: &'static str,
    pub on: bool,
    pub highlighted: bool,
    /// A puppet's current value (0..1) for the ring.
    pub value: Option<f32>,
    pub is_back: bool,
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct PuppetView {
    pub label: String,
    pub kind: Option<MenuItemKind>,
    /// Axis values (radial: [0]; two-axis: x, y; four-axis: up, right, down, left).
    pub values: [f32; 4],
}

/// Everything a frame needs to draw the menu.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct RadialView {
    pub slices: Vec<SliceView>,
    pub centre: String,
    pub puppet: Option<PuppetView>,
}

/// Which of `count` slices the offset `dir` (x right, y up) points at, or
/// `None` inside the dead zone. Slice 0 is straight up; they run clockwise.
pub fn slice_at(dir: GVec2, count: usize) -> Option<usize> {
    if count == 0 || dir.length() < DEAD_ZONE {
        return None;
    }
    Some(((angle_cw(dir) / std::f32::consts::TAU * count as f32).round() as usize) % count)
}

/// Clockwise angle from straight up, 0..TAU.
fn angle_cw(dir: GVec2) -> f32 {
    dir.x.atan2(dir.y).rem_euclid(std::f32::consts::TAU)
}

fn kind_badge(k: MenuItemKind) -> &'static str {
    match k {
        MenuItemKind::Button => "button",
        MenuItemKind::Toggle => "toggle",
        MenuItemKind::SubMenu => ">",
        MenuItemKind::RadialPuppet => "radial",
        MenuItemKind::TwoAxisPuppet => "2-axis",
        MenuItemKind::FourAxisPuppet => "4-axis",
    }
}

impl RadialMenu {
    pub fn is_open(&self) -> bool {
        self.open
    }
    /// A puppet is being moved: sync its parameters at the fast rate.
    pub fn fast_sync(&self) -> bool {
        self.puppet.is_some()
    }
    /// Something needs the per-frame update even while closed (a Button still
    /// inside its minimum press time).
    pub fn busy(&self) -> bool {
        self.open || self.held.is_some()
    }

    fn page<'a>(&self, menu: &'a AvatarMenu) -> &'a [AvatarMenuNode] {
        menu.page(&self.path).unwrap_or(&[])
    }

    fn has_back(&self) -> bool {
        !self.path.is_empty()
    }

    fn write(store: &mut AvatarParamStore, name: &str, v: f32) -> Option<usize> {
        let slot = store.index_of(name).filter(|&s| !store.is_builtin(s))?;
        store.set(slot, v);
        Some(slot)
    }

    fn close(&mut self, store: &mut AvatarParamStore, now: f64) {
        self.leave_puppet(store);
        while let Some(slot) = self.open_slots.pop() {
            if let Some(s) = slot {
                store.set(s, 0.0);
            }
        }
        self.path.clear();
        self.open = false;
        self.highlighted = None;
        if let Some(h) = self.held.as_mut() {
            h.released = true;
            h.since = h.since.min(now);
        }
    }

    fn leave_puppet(&mut self, store: &mut AvatarParamStore) {
        if let Some(p) = self.puppet.take()
            && let Some(s) = p.open_slot
        {
            store.set(s, 0.0);
        }
    }

    /// Advance one frame. Returns true when a parameter changed.
    pub fn update(&mut self, menu: &AvatarMenu, store: &mut AvatarParamStore, input: &MenuInput, now: f64) -> bool {
        let mut changed = false;
        if input.select_released
            && let Some(h) = self.held.as_mut()
        {
            h.released = true;
        }
        // A released Button clears once its minimum time is up.
        if let Some(h) = self.held
            && h.released
            && now - h.since >= MIN_BUTTON_TIME
        {
            store.set(h.slot, 0.0);
            self.held = None;
            changed = true;
        }
        if !input.open {
            if self.open {
                self.close(store, now);
                changed = true;
            }
            return changed;
        }
        if !self.open {
            self.open = true;
            self.path.clear();
            self.highlighted = None;
        }

        if let Some(p) = self.puppet.clone() {
            changed |= self.drive_puppet(&p, store, input.dir);
            if input.select || input.back {
                self.leave_puppet(store);
                changed = true;
            }
            return changed;
        }

        let count = self.page(menu).len() + self.has_back() as usize;
        self.highlighted = slice_at(input.dir, count);
        if input.back {
            changed |= self.go_back(store);
        } else if input.select {
            changed |= match self.highlighted {
                // Pointing at the centre: close, or back from a sub page.
                None if input.dir.length() < DEAD_ZONE => {
                    if self.has_back() {
                        self.go_back(store)
                    } else {
                        self.close(store, now);
                        true
                    }
                }
                None => false,
                Some(0) if self.has_back() => self.go_back(store),
                Some(s) => {
                    let idx = s - self.has_back() as usize;
                    self.activate(menu, store, idx, now)
                }
            };
        }
        changed
    }

    fn go_back(&mut self, store: &mut AvatarParamStore) -> bool {
        if self.path.pop().is_none() {
            return false;
        }
        if let Some(Some(s)) = self.open_slots.pop() {
            store.set(s, 0.0);
        }
        true
    }

    fn activate(&mut self, menu: &AvatarMenu, store: &mut AvatarParamStore, idx: usize, now: f64) -> bool {
        let Some(node) = self.page(menu).get(idx) else {
            return false;
        };
        let item = &node.item;
        match item.kind {
            MenuItemKind::Toggle => {
                let Some(slot) = store.index_of(&item.parameter).filter(|&s| !store.is_builtin(s)) else {
                    return false;
                };
                store.set(slot, item.toggled(store.get(slot)));
                true
            }
            MenuItemKind::Button => {
                let Some(slot) = Self::write(store, &item.parameter, item.value) else {
                    return false;
                };
                self.held = Some(Held { slot, since: now, released: false });
                true
            }
            MenuItemKind::SubMenu => {
                let slot = (!item.parameter.is_empty()).then(|| Self::write(store, &item.parameter, item.value)).flatten();
                self.path.push(idx);
                self.open_slots.push(slot);
                true
            }
            k if k.is_puppet() => {
                let open_slot = (!item.parameter.is_empty()).then(|| Self::write(store, &item.parameter, item.value)).flatten();
                let mut axes = [usize::MAX; 4];
                for (a, name) in axes.iter_mut().zip(&item.sub_parameters) {
                    if let Some(s) = store.index_of(name).filter(|&s| !store.is_builtin(s)) {
                        *a = s;
                    }
                }
                self.puppet = Some(Puppet { idx, kind: k, axes, open_slot });
                true
            }
            _ => false,
        }
    }

    /// Stick -> puppet axes: radial = angle around the circle (0..1), two-axis
    /// = x and y (-1..1), four-axis = one 0..1 value per direction.
    fn drive_puppet(&mut self, p: &Puppet, store: &mut AvatarParamStore, dir: GVec2) -> bool {
        let set = |store: &mut AvatarParamStore, k: usize, v: f32| {
            if p.axes[k] != usize::MAX && (store.get(p.axes[k]) - v).abs() > 1e-4 {
                store.set(p.axes[k], v);
                return true;
            }
            false
        };
        match p.kind {
            MenuItemKind::RadialPuppet => {
                if dir.length() < PUPPET_DEAD {
                    return false;
                }
                set(store, 0, angle_cw(dir) / std::f32::consts::TAU)
            }
            MenuItemKind::TwoAxisPuppet => {
                let d = dir.clamp(GVec2::splat(-1.0), GVec2::ONE);
                set(store, 0, d.x) | set(store, 1, d.y)
            }
            MenuItemKind::FourAxisPuppet => {
                let d = dir.clamp(GVec2::splat(-1.0), GVec2::ONE);
                set(store, 0, d.y.max(0.0)) | set(store, 1, d.x.max(0.0)) | set(store, 2, (-d.y).max(0.0)) | set(store, 3, (-d.x).max(0.0))
            }
            _ => false,
        }
    }

    /// What to draw now. Allocates (labels), so call it only while open.
    pub fn view(&self, menu: &AvatarMenu, store: &AvatarParamStore) -> RadialView {
        let mut v = RadialView::default();
        if let Some(p) = &self.puppet {
            let node = self.page(menu).get(p.idx);
            let mut pv = PuppetView { label: node.map(|n| n.item.label.clone()).unwrap_or_default(), kind: Some(p.kind), values: [0.0; 4] };
            for (k, &s) in p.axes.iter().enumerate() {
                if s != usize::MAX {
                    pv.values[k] = store.get(s);
                }
            }
            v.puppet = Some(pv);
            v.centre = "Done".into();
            return v;
        }
        if self.has_back() {
            v.slices.push(SliceView { label: "Back".into(), badge: "", on: false, highlighted: self.highlighted == Some(0), value: None, is_back: true });
        }
        for (i, n) in self.page(menu).iter().enumerate() {
            let slice = i + self.has_back() as usize;
            let it: &AvatarMenuItem = &n.item;
            let on = matches!(it.kind, MenuItemKind::Toggle | MenuItemKind::Button)
                && store.index_of(&it.parameter).is_some_and(|s| it.is_on(store.get(s)));
            let value = (it.kind == MenuItemKind::RadialPuppet).then(|| it.sub_parameters.first().and_then(|p| store.index_of(p)).map(|s| store.get(s))).flatten();
            v.slices.push(SliceView {
                label: it.label.clone(),
                badge: kind_badge(it.kind),
                on,
                highlighted: self.highlighted == Some(slice),
                value,
                is_back: false,
            });
        }
        v.centre = if self.has_back() { "Back".into() } else { "Close".into() };
        v
    }
}

// --------------------------------------------------------------------- paint

const COL_SLICE: Color32 = Color32::from_rgba_premultiplied(26, 29, 36, 230);
const COL_HOT: Color32 = Color32::from_rgb(52, 62, 84);
const COL_ON: Color32 = Color32::from_rgb(48, 128, 92);
const COL_ON_HOT: Color32 = Color32::from_rgb(64, 160, 116);
const COL_TEXT: Color32 = Color32::from_rgb(232, 234, 240);
const COL_DIM: Color32 = Color32::from_rgb(150, 156, 170);

/// A ring sector from angle `a0` to `a1` (clockwise from up), radii `r0..r1`.
pub fn sector_points(centre: Pos2, r0: f32, r1: f32, a0: f32, a1: f32) -> Vec<Pos2> {
    let steps = (((a1 - a0).abs() / 0.12).ceil() as usize).max(2);
    let at = |r: f32, a: f32| Pos2::new(centre.x + r * a.sin(), centre.y - r * a.cos());
    let mut pts = Vec::with_capacity((steps + 1) * 2);
    for i in 0..=steps {
        pts.push(at(r1, a0 + (a1 - a0) * i as f32 / steps as f32));
    }
    for i in (0..=steps).rev() {
        pts.push(at(r0, a0 + (a1 - a0) * i as f32 / steps as f32));
    }
    pts
}

/// Draw `view` as a pie centred at `centre` with outer radius `radius`.
pub fn paint_radial(painter: &egui::Painter, centre: Pos2, radius: f32, view: &RadialView) {
    let inner = radius * 0.34;
    painter.circle_filled(centre, radius * 1.02, Color32::from_black_alpha(90));
    if let Some(p) = &view.puppet {
        paint_puppet(painter, centre, radius, p);
    } else {
        let n = view.slices.len().max(1);
        let step = std::f32::consts::TAU / n as f32;
        let gap = 0.025;
        for (i, s) in view.slices.iter().enumerate() {
            let mid = step * i as f32;
            let fill = match (s.on, s.highlighted) {
                (true, true) => COL_ON_HOT,
                (true, false) => COL_ON,
                (false, true) => COL_HOT,
                (false, false) => COL_SLICE,
            };
            painter.add(Shape::convex_polygon(sector_points(centre, inner, radius, mid - step / 2.0 + gap, mid + step / 2.0 - gap), fill, Stroke::NONE));
            if let Some(v) = s.value {
                // The radial puppet's value as an arc along the slice.
                let a0 = mid - step / 2.0 + gap;
                let a1 = a0 + (step - 2.0 * gap) * v.clamp(0.0, 1.0);
                painter.add(Shape::convex_polygon(sector_points(centre, radius * 0.9, radius * 0.97, a0, a1.max(a0 + 0.02)), COL_ON_HOT, Stroke::NONE));
            }
            let lr = (inner + radius) * 0.5;
            let p = Pos2::new(centre.x + lr * mid.sin(), centre.y - lr * mid.cos());
            let size = (radius * 0.085).clamp(11.0, 22.0);
            painter.text(p, Align2::CENTER_CENTER, &s.label, FontId::proportional(size), COL_TEXT);
            if !s.badge.is_empty() {
                painter.text(p + Vec2::new(0.0, size * 1.2), Align2::CENTER_CENTER, s.badge, FontId::proportional(size * 0.7), COL_DIM);
            }
        }
    }
    painter.circle_filled(centre, inner * 0.92, COL_SLICE);
    painter.text(centre, Align2::CENTER_CENTER, &view.centre, FontId::proportional((radius * 0.08).clamp(11.0, 20.0)), COL_DIM);
}

fn paint_puppet(painter: &egui::Painter, centre: Pos2, radius: f32, p: &PuppetView) {
    let r = radius * 0.8;
    painter.circle_filled(centre, r, COL_SLICE);
    painter.circle_stroke(centre, r, Stroke::new(1.5_f32, COL_DIM));
    painter.text(centre + Vec2::new(0.0, -r - 14.0), Align2::CENTER_CENTER, &p.label, FontId::proportional(18.0), COL_TEXT);
    match p.kind {
        Some(MenuItemKind::RadialPuppet) => {
            let v = p.values[0].clamp(0.0, 1.0);
            let a1 = (std::f32::consts::TAU * v).max(0.02);
            painter.add(Shape::convex_polygon(sector_points(centre, r * 0.7, r * 0.95, 0.0, a1), COL_ON_HOT, Stroke::NONE));
            painter.text(centre, Align2::CENTER_CENTER, format!("{:.0}%", v * 100.0), FontId::proportional(24.0), COL_TEXT);
        }
        Some(MenuItemKind::TwoAxisPuppet) => {
            let dot = centre + Vec2::new(p.values[0], -p.values[1]) * r * 0.9;
            painter.line_segment([centre - Vec2::new(r, 0.0), centre + Vec2::new(r, 0.0)], Stroke::new(1.0_f32, COL_DIM));
            painter.line_segment([centre - Vec2::new(0.0, r), centre + Vec2::new(0.0, r)], Stroke::new(1.0_f32, COL_DIM));
            painter.circle_filled(dot, 9.0, COL_ON_HOT);
        }
        Some(MenuItemKind::FourAxisPuppet) => {
            let [up, right, down, left] = p.values;
            let dot = centre + Vec2::new(right - left, down - up) * r * 0.9;
            painter.line_segment([centre - Vec2::new(r, 0.0), centre + Vec2::new(r, 0.0)], Stroke::new(1.0_f32, COL_DIM));
            painter.line_segment([centre - Vec2::new(0.0, r), centre + Vec2::new(0.0, r)], Stroke::new(1.0_f32, COL_DIM));
            painter.circle_filled(dot, 9.0, COL_ON_HOT);
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------- host

/// Finds the local avatar's menu in the scene and runs the radial menu for it.
#[derive(Default)]
pub struct AvatarMenuHost {
    menu: RadialMenu,
    /// Index and id of the avatar being driven.
    root: Option<(usize, ObjectId)>,
    last_scan: f64,
    view: Option<RadialView>,
    avatar_name: String,
}

impl AvatarMenuHost {
    /// The menu is up and wants a free cursor on desktop.
    pub fn is_open(&self) -> bool {
        self.menu.is_open()
    }

    pub fn view(&self) -> Option<&RadialView> {
        self.view.as_ref().filter(|_| self.menu.is_open())
    }

    pub fn avatar_name(&self) -> &str {
        &self.avatar_name
    }

    /// The local avatar with a menu: its object index.
    fn find_root(&mut self, scene: &LoadedScene, now: f64, force: bool) -> Option<usize> {
        if let Some((i, id)) = self.root
            && scene.objects.get(i).is_some_and(|o| o.id == id)
        {
            return Some(i);
        }
        if !force && now - self.last_scan < 1.0 {
            return None;
        }
        self.last_scan = now;
        let found = (0..scene.objects.len()).find(|&i| {
            let c = &scene.objects[i].components;
            c.iter().any(|x| x.as_any().is::<AvatarMenuData>())
                && c.iter().any(|x| x.as_any().downcast_ref::<AvatarParameters>().is_some_and(|a| a.ready && a.store.get_builtin(zestengine_core::BuiltinParam::IsLocal) > 0.5))
                && scene.is_active(i)
        });
        self.root = found.map(|i| (i, scene.objects[i].id));
        self.avatar_name = found.map(|i| scene.objects[i].name.clone()).unwrap_or_default();
        found
    }

    /// Run the menu for this frame. Returns true while it is up.
    pub fn drive(&mut self, scene: &mut LoadedScene, input: &MenuInput, now: f64) -> bool {
        if !input.open && !self.menu.busy() {
            if let Some((i, _)) = self.root
                && let Some(ap) = scene.objects.get_mut(i).and_then(|o| o.components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<AvatarParameters>()))
                && ap.fast_sync
            {
                ap.fast_sync = false;
            }
            self.view = None;
            // A scan only on the frame the menu opens.
            return false;
        }
        let Some(i) = self.find_root(scene, now, input.open) else {
            self.menu = RadialMenu::default();
            self.view = None;
            return false;
        };
        let obj = &mut scene.objects[i];
        // Lift the menu out of its component for the frame (moves, no copies).
        let Some(mut data) = obj.components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<AvatarMenuData>()).map(|d| std::mem::take(&mut d.menu)) else {
            self.root = None;
            return false;
        };
        if let Some(ap) = obj.components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<AvatarParameters>()) {
            self.menu.update(&data, &mut ap.store, input, now);
            ap.fast_sync = self.menu.fast_sync();
            self.view = self.menu.is_open().then(|| self.menu.view(&data, &ap.store));
        }
        if let Some(d) = obj.components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<AvatarMenuData>()) {
            d.menu = std::mem::take(&mut data);
        }
        self.menu.is_open()
    }

    /// Desktop overlay: the pie at the centre of `rect`.
    pub fn show(&self, ctx: &egui::Context, rect: egui::Rect) {
        let Some(view) = self.view() else { return };
        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("zestengine-avatar-menu"))).with_clip_rect(rect);
        let radius = (rect.width().min(rect.height()) * 0.36).clamp(120.0, 300.0);
        paint_radial(&painter, rect.center(), radius, view);
    }
}

/// Whether any active object in `scene` carries a finished avatar menu
/// (cheap: only asked on a menu-button press).
pub fn scene_has_menu(scene: &LoadedScene) -> bool {
    (0..scene.objects.len()).any(|i| scene.objects[i].components.iter().any(|c| c.as_any().is::<AvatarMenuData>()) && scene.is_active(i))
}

/// Desktop input: `held` is the open key, `pointer` the cursor in the same
/// space as `rect` (None while the cursor is hidden), and the arrow keys /
/// WASD give a compass direction when there is no pointer.
pub fn desktop_input(rect: egui::Rect, held: bool, pointer: Option<Pos2>, keys_dir: GVec2, select: bool, select_released: bool, back: bool) -> MenuInput {
    let radius = (rect.width().min(rect.height()) * 0.36).clamp(120.0, 300.0);
    let dir = match pointer {
        Some(p) if keys_dir == GVec2::ZERO => {
            let d = p - rect.center();
            GVec2::new(d.x, -d.y) / radius
        }
        _ => keys_dir,
    };
    MenuInput { open: held, dir: dir.clamp_length_max(1.0), select, select_released, back }
}

/// The editor's desktop menu input read from egui: hold R to open, arrows
/// pick a slice, click / Enter selects, Backspace / right-click goes back.
/// `cursor_locked` hides the pointer (mouse-look), so only the arrows steer.
///
/// `Context::input` holds the context write lock for its closure, and every
/// other `Context` accessor (`wants_keyboard_input` reads `memory`) takes the
/// same lock; calling one inside `input(..)` deadlocks the main thread. So
/// the focus query happens before the closure.
pub fn egui_desktop_input(ctx: &egui::Context, rect: egui::Rect, cursor_locked: bool) -> MenuInput {
    let typing = ctx.wants_keyboard_input();
    ctx.input(|i| {
        use egui::Key;
        let k = |key: Key| i.key_down(key) as u8 as f32;
        let dir = GVec2::new(k(Key::ArrowRight) - k(Key::ArrowLeft), k(Key::ArrowUp) - k(Key::ArrowDown));
        let pointer = (!cursor_locked).then(|| i.pointer.latest_pos().filter(|p| rect.contains(*p))).flatten();
        desktop_input(
            rect,
            !typing && i.key_down(Key::R),
            pointer,
            dir,
            i.pointer.primary_pressed() || i.key_pressed(Key::Enter),
            i.pointer.primary_released() || i.key_released(Key::Enter),
            i.key_pressed(Key::Backspace) || i.pointer.secondary_pressed(),
        )
    })
}

// ------------------------------------------------------------ desktop keys

/// Raw key state for the standalone player (the editor reads egui's instead).
#[derive(Default)]
pub struct MenuKeys {
    open: bool,
    up: bool,
    down: bool,
    left: bool,
    right: bool,
    enter: bool,
    enter_prev: bool,
    back: bool,
    back_prev: bool,
}

impl MenuKeys {
    pub fn set(&mut self, code: winit::keyboard::KeyCode, pressed: bool) {
        use winit::keyboard::KeyCode as K;
        match code {
            K::KeyR => self.open = pressed,
            K::ArrowUp => self.up = pressed,
            K::ArrowDown => self.down = pressed,
            K::ArrowLeft => self.left = pressed,
            K::ArrowRight => self.right = pressed,
            K::Enter | K::NumpadEnter => self.enter = pressed,
            K::Backspace => self.back = pressed,
            _ => {}
        }
    }

    /// This frame's input; call once per frame (it keeps the previous state
    /// for press / release edges).
    pub fn input(&mut self, rect: egui::Rect, pointer: Option<Pos2>, mouse_pressed: bool, mouse_released: bool) -> MenuInput {
        let k = |b: bool| b as u8 as f32;
        let dir = GVec2::new(k(self.right) - k(self.left), k(self.up) - k(self.down));
        let select = mouse_pressed || (self.enter && !self.enter_prev);
        let released = mouse_released || (!self.enter && self.enter_prev);
        let back = self.back && !self.back_prev;
        self.enter_prev = self.enter;
        self.back_prev = self.back;
        desktop_input(rect, self.open, pointer, dir, select, released, back)
    }
}

// ------------------------------------------------------------------- VR panel

const VR_PANEL_PX: u32 = 768;
/// Panel width in tracking-space meters.
const VR_PANEL_M: f32 = 0.55;

struct VrGpu {
    target: zestengine_render::TextureHandle,
    material: zestengine_render::MaterialHandle,
    quad: zestengine_render::MeshHandle,
}

/// The menu as a world-space panel in front of the player: the pie is painted
/// by egui into a texture that a quad shows. The right thumbstick picks a
/// slice (or moves a puppet), the right trigger selects, B goes back and the
/// menu button opens and closes it.
#[derive(Default)]
pub struct AvatarMenuVr {
    gpu: Option<VrGpu>,
    ctx: egui::Context,
    anchor: Option<glam::Mat4>,
    open: bool,
    menu_prev: bool,
    trigger_prev: bool,
    b_prev: bool,
    time: f64,
    host: AvatarMenuHost,
    failed: bool,
}

impl AvatarMenuVr {
    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Run the menu for this VR frame and queue its quad on
    /// `scene.extra_draws`. Returns the panel's UI draw while it is up.
    pub fn frame(
        &mut self,
        renderer: &mut zestengine_render::Renderer,
        scene: &mut LoadedScene,
        rig: &zestengine_core::VrRig,
        vr: &zestengine_core::VrFrame,
        dt: f32,
    ) -> Option<zestengine_render::WorldUiDraw> {
        if !vr.active {
            if self.open {
                self.open = false;
                let closed = MenuInput::default();
                self.host.drive(scene, &closed, self.time);
            }
            return None;
        }
        self.time += dt as f64;
        let toggled = vr.menu && !self.menu_prev;
        self.menu_prev = vr.menu;
        if toggled {
            self.open = !self.open;
            self.anchor = None;
        }
        let select = vr.right_trigger && !self.trigger_prev;
        let released = !vr.right_trigger && self.trigger_prev;
        let back = vr.b && !self.b_prev;
        self.trigger_prev = vr.right_trigger;
        self.b_prev = vr.b;
        let input = MenuInput { open: self.open, dir: vr.right_stick.clamp_length_max(1.0), select, select_released: released, back };
        let up = self.host.drive(scene, &input, self.time);
        if !self.open {
            return None;
        }
        if !up && self.host.view().is_none() {
            // No avatar with a menu: nothing to show; do not hold the menu open.
            self.open = false;
            return None;
        }
        if self.gpu.is_none() && !self.failed {
            match create_vr_gpu(renderer) {
                Ok(g) => {
                    self.ctx.set_visuals(egui::Visuals::dark());
                    self.gpu = Some(g);
                }
                Err(e) => {
                    tracing::error!("avatar menu panel: {e:#}");
                    self.failed = true;
                }
            }
        }
        let gpu = self.gpu.as_ref()?;
        let head = vr.head.map(|h| rig.stage_pose_to_world(h)).unwrap_or((rig.origin, glam::Quat::IDENTITY));
        let scale = rig.scale.max(1e-3);
        let anchor = *self.anchor.get_or_insert_with(|| {
            let f = head.1 * glam::Vec3::NEG_Z;
            let fwd = glam::Vec3::new(f.x, 0.0, f.z).normalize_or(glam::Vec3::NEG_Z);
            let pos = head.0 + fwd * 0.7 * scale - glam::Vec3::Y * 0.1 * scale;
            glam::Mat4::from_scale_rotation_translation(glam::Vec3::splat(scale), glam::Quat::from_rotation_y((-fwd.x).atan2(-fwd.z)), pos)
        });
        let view = self.host.view()?.clone();
        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(Pos2::ZERO, Vec2::splat(VR_PANEL_PX as f32))),
            time: Some(self.time),
            ..Default::default()
        };
        let out = self.ctx.run(raw, |ctx| {
            egui::CentralPanel::default().frame(egui::Frame::new().fill(Color32::TRANSPARENT)).show(ctx, |ui| {
                let r = ui.max_rect();
                paint_radial(ui.painter(), r.center(), r.width() * 0.46, &view);
            });
        });
        let primitives = self.ctx.tessellate(out.shapes, out.pixels_per_point);
        scene.extra_draws.push(vr_draw(gpu.quad, gpu.material, anchor * glam::Mat4::from_scale(glam::Vec3::new(VR_PANEL_M, VR_PANEL_M, 1.0))));
        Some(zestengine_render::WorldUiDraw {
            target: gpu.target,
            extent: [VR_PANEL_PX, VR_PANEL_PX],
            draw: zestengine_render::EguiDraw { pixels_per_point: out.pixels_per_point, primitives, textures_delta: out.textures_delta },
        })
    }
}

fn create_vr_gpu(renderer: &mut zestengine_render::Renderer) -> anyhow::Result<VrGpu> {
    let target = renderer.create_ui_target(VR_PANEL_PX, VR_PANEL_PX)?;
    let material = crate::game_ui::unlit_material(renderer, Some(target), "avatar menu")?;
    let v = |x: f32, y: f32, u: f32, w: f32| zestengine_render::Vertex {
        position: [x, y, 0.0],
        normal: [0.0, 0.0, 1.0],
        uv: [u, w],
        color: [1.0; 4],
        tangent: [1.0, 0.0, 0.0, 1.0],
        uv1: [u, w],
        joints: [0; 4],
        weights: [0.0; 4],
    };
    let quad = renderer.upload_mesh(&zestengine_render::MeshData {
        vertices: vec![v(-0.5, 0.5, 0.0, 0.0), v(0.5, 0.5, 1.0, 0.0), v(0.5, -0.5, 1.0, 1.0), v(-0.5, -0.5, 0.0, 1.0)],
        indices: vec![0, 2, 1, 0, 3, 2],
        has_lightmap_uv: false,
    })?;
    Ok(VrGpu { target, material, quad })
}

fn vr_draw(mesh: zestengine_render::MeshHandle, material: zestengine_render::MaterialHandle, transform: glam::Mat4) -> zestengine_render::DrawCmd {
    zestengine_render::DrawCmd {
        gi_source: None,
        mesh,
        material,
        transform,
        highlight: 0.0,
        mesh_center: glam::Vec3::ZERO,
        bound_radius: 0.5,
        lightmap_layer: -1,
        lightmap_size: 0,
        static_geometry: false,
        planar_index: -1,
        occluded: false,
        shadow_casting: zestengine_render::ShadowCasting::Off,
        instance_color: None,
        instance_param: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_core::{AvatarParam, AvatarParamKind};

    /// Regression: Play deadlocked the main thread because the desktop menu
    /// input called `ctx.wants_keyboard_input()` (a read lock) inside
    /// `ctx.input(..)` (which holds the write lock). Runs on a worker so a
    /// relapse fails in seconds instead of waiting for egui's 10 s panic.
    #[test]
    fn egui_desktop_input_does_not_deadlock_the_context() {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let ctx = egui::Context::default();
            let rect = egui::Rect::from_min_size(Pos2::ZERO, egui::vec2(800.0, 600.0));
            let r_down = egui::Event::Key {
                key: egui::Key::R,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            };
            let raw = egui::RawInput { screen_rect: Some(rect), events: vec![r_down], ..Default::default() };
            let mut got = None;
            let _ = ctx.run(raw, |ctx| got = Some(egui_desktop_input(ctx, rect, false)));
            let _ = tx.send(got.unwrap().open);
        });
        let open = rx.recv_timeout(std::time::Duration::from_secs(5)).expect("egui_desktop_input deadlocked the egui context");
        assert!(open, "holding R opens the menu");
    }

    fn item(label: &str, kind: MenuItemKind, parameter: &str) -> AvatarMenuNode {
        AvatarMenuNode::new(AvatarMenuItem { label: label.into(), kind, parameter: parameter.into(), ..Default::default() })
    }

    fn setup() -> (AvatarMenu, AvatarParamStore) {
        let mut sub = item("Outfit", MenuItemKind::SubMenu, "OutfitOpen");
        sub.children.push(item("Hat", MenuItemKind::Toggle, "Hat"));
        let mut radial = item("Smile", MenuItemKind::RadialPuppet, "SmileOpen");
        radial.item.sub_parameters = vec!["Smile".into()];
        let mut two = item("Look", MenuItemKind::TwoAxisPuppet, "");
        two.item.sub_parameters = vec!["LookX".into(), "LookY".into()];
        let mut four = item("Pad", MenuItemKind::FourAxisPuppet, "");
        four.item.sub_parameters = vec!["Up".into(), "Right".into(), "Down".into(), "Left".into()];
        let menu = AvatarMenu { root: vec![item("Cape", MenuItemKind::Toggle, "Cape"), sub, item("Wave", MenuItemKind::Button, "Wave"), radial, two, four] };
        let store = AvatarParamStore::new(&[
            AvatarParam::new("Cape", AvatarParamKind::Bool),
            AvatarParam::new("Hat", AvatarParamKind::Bool),
            AvatarParam::new("OutfitOpen", AvatarParamKind::Bool),
            AvatarParam::new("Wave", AvatarParamKind::Bool),
            AvatarParam::new("SmileOpen", AvatarParamKind::Bool),
            AvatarParam::new("Smile", AvatarParamKind::Float),
            AvatarParam::new("LookX", AvatarParamKind::Float),
            AvatarParam::new("LookY", AvatarParamKind::Float),
            AvatarParam::new("Up", AvatarParamKind::Float),
            AvatarParam::new("Right", AvatarParamKind::Float),
            AvatarParam::new("Down", AvatarParamKind::Float),
            AvatarParam::new("Left", AvatarParamKind::Float),
        ]);
        (menu, store)
    }

    /// Direction pointing at slice `i` of `n`.
    fn toward(i: usize, n: usize) -> GVec2 {
        let a = std::f32::consts::TAU * i as f32 / n as f32;
        GVec2::new(a.sin(), a.cos())
    }

    fn open(dir: GVec2) -> MenuInput {
        MenuInput { open: true, dir, ..Default::default() }
    }
    fn select(dir: GVec2) -> MenuInput {
        MenuInput { open: true, dir, select: true, ..Default::default() }
    }
    fn val(s: &AvatarParamStore, n: &str) -> f32 {
        s.get(s.index_of(n).unwrap())
    }

    #[test]
    fn slices_are_clockwise_from_the_top() {
        assert_eq!(slice_at(GVec2::new(0.0, 1.0), 4), Some(0));
        assert_eq!(slice_at(GVec2::new(1.0, 0.0), 4), Some(1));
        assert_eq!(slice_at(GVec2::new(0.0, -1.0), 4), Some(2));
        assert_eq!(slice_at(GVec2::new(-1.0, 0.0), 4), Some(3));
        assert_eq!(slice_at(GVec2::new(-0.05, 1.0), 8), Some(0), "slightly left of up is still the first slice");
        assert_eq!(slice_at(GVec2::new(-0.2, 1.0), 8), Some(0));
        assert_eq!(slice_at(GVec2::new(-0.7, 1.0), 8), Some(7), "further left is the last slice");
        assert_eq!(slice_at(GVec2::new(0.1, 0.1), 4), None, "dead zone");
        assert_eq!(slice_at(GVec2::new(1.0, 0.0), 0), None);
    }

    #[test]
    fn toggles_flip_and_flip_back() {
        let (menu, mut store) = setup();
        let mut m = RadialMenu::default();
        let n = menu.root.len();
        assert!(m.update(&menu, &mut store, &select(toward(0, n)), 0.0));
        assert_eq!(val(&store, "Cape"), 1.0);
        m.update(&menu, &mut store, &select(toward(0, n)), 0.1);
        assert_eq!(val(&store, "Cape"), 0.0);
        // A press that points at no slice changes no parameter.
        assert!(!m.update(&menu, &mut store, &MenuInput { open: true, select: false, dir: GVec2::new(0.1, 0.0), ..Default::default() }, 0.2));
    }

    #[test]
    fn buttons_hold_for_at_least_two_tenths_of_a_second() {
        let (menu, mut store) = setup();
        let mut m = RadialMenu::default();
        let n = menu.root.len();
        m.update(&menu, &mut store, &select(toward(2, n)), 1.0);
        assert_eq!(val(&store, "Wave"), 1.0);
        // Released after 50 ms: stays on until 200 ms are up.
        let rel = MenuInput { open: true, select_released: true, dir: toward(2, n), ..Default::default() };
        m.update(&menu, &mut store, &rel, 1.05);
        assert_eq!(val(&store, "Wave"), 1.0);
        m.update(&menu, &mut store, &open(toward(2, n)), 1.15);
        assert_eq!(val(&store, "Wave"), 1.0);
        assert!(m.busy());
        m.update(&menu, &mut store, &open(toward(2, n)), 1.21);
        assert_eq!(val(&store, "Wave"), 0.0);
        // A long hold ends the moment it is released.
        m.update(&menu, &mut store, &select(toward(2, n)), 2.0);
        m.update(&menu, &mut store, &MenuInput { open: true, select_released: true, ..Default::default() }, 3.0);
        assert_eq!(val(&store, "Wave"), 0.0);
    }

    #[test]
    fn sub_menus_set_their_open_parameter_and_back_clears_it() {
        let (menu, mut store) = setup();
        let mut m = RadialMenu::default();
        let n = menu.root.len();
        m.update(&menu, &mut store, &select(toward(1, n)), 0.0);
        assert_eq!(val(&store, "OutfitOpen"), 1.0, "open while the page is up");
        // The sub page has Back (slice 0) and Hat (slice 1) of 2.
        let v = m.view(&menu, &store);
        assert_eq!(v.slices.iter().map(|s| s.label.as_str()).collect::<Vec<_>>(), vec!["Back", "Hat"]);
        m.update(&menu, &mut store, &select(toward(1, 2)), 0.1);
        assert_eq!(val(&store, "Hat"), 1.0);
        assert!(m.view(&menu, &store).slices[1].on);
        m.update(&menu, &mut store, &select(toward(0, 2)), 0.2);
        assert_eq!(val(&store, "OutfitOpen"), 0.0);
        assert_eq!(m.view(&menu, &store).centre, "Close");
        // Closing the menu from inside a sub page clears the open parameter too.
        m.update(&menu, &mut store, &select(toward(1, n)), 0.3);
        assert_eq!(val(&store, "OutfitOpen"), 1.0);
        m.update(&menu, &mut store, &MenuInput::default(), 0.4);
        assert_eq!(val(&store, "OutfitOpen"), 0.0);
        assert!(!m.is_open());
        // The centre closes the root page.
        m.update(&menu, &mut store, &open(GVec2::ZERO), 0.5);
        assert!(m.is_open());
        m.update(&menu, &mut store, &select(GVec2::ZERO), 0.6);
        assert!(!m.is_open());
    }

    #[test]
    fn puppets_map_the_stick_to_their_axes() {
        let (menu, mut store) = setup();
        let mut m = RadialMenu::default();
        let n = menu.root.len();
        // Radial: the angle around the circle, 0..1; opening sets the open parameter.
        m.update(&menu, &mut store, &select(toward(3, n)), 0.0);
        assert!(m.fast_sync());
        assert_eq!(val(&store, "SmileOpen"), 1.0);
        m.update(&menu, &mut store, &open(GVec2::new(1.0, 0.0)), 0.1);
        assert!((val(&store, "Smile") - 0.25).abs() < 0.01, "quarter turn is 0.25 (quantised to the wire range)");
        m.update(&menu, &mut store, &open(GVec2::new(0.0, -1.0)), 0.2);
        assert!((val(&store, "Smile") - 0.5).abs() < 0.01);
        m.update(&menu, &mut store, &open(GVec2::new(0.05, 0.0)), 0.3);
        assert!((val(&store, "Smile") - 0.5).abs() < 0.01, "inside the dead zone it holds");
        // Select leaves the puppet and keeps the value.
        m.update(&menu, &mut store, &select(GVec2::ZERO), 0.4);
        assert!(!m.fast_sync());
        assert_eq!(val(&store, "SmileOpen"), 0.0);
        assert!((val(&store, "Smile") - 0.5).abs() < 0.01);

        // Two-axis: x and y straight through.
        m.update(&menu, &mut store, &select(toward(4, n)), 1.0);
        m.update(&menu, &mut store, &open(GVec2::new(-0.5, 1.0)), 1.1);
        assert!((val(&store, "LookX") + 0.5).abs() < 0.01 && (val(&store, "LookY") - 1.0).abs() < 0.01);
        m.update(&menu, &mut store, &MenuInput { open: true, back: true, ..Default::default() }, 1.2);
        assert!(!m.fast_sync());

        // Four-axis: one non-negative value per direction.
        m.update(&menu, &mut store, &select(toward(5, n)), 2.0);
        m.update(&menu, &mut store, &open(GVec2::new(0.6, -0.3)), 2.1);
        assert_eq!(val(&store, "Up"), 0.0);
        assert!((val(&store, "Down") - 0.3).abs() < 1e-4);
        assert!((val(&store, "Right") - 0.6).abs() < 0.01 && val(&store, "Left") == 0.0);
        // The view shows the live values.
        let v = m.view(&menu, &store);
        assert_eq!(v.puppet.as_ref().unwrap().label, "Pad");
    }

    #[test]
    fn closing_while_a_puppet_is_open_clears_its_open_parameter() {
        let (menu, mut store) = setup();
        let mut m = RadialMenu::default();
        let n = menu.root.len();
        m.update(&menu, &mut store, &select(toward(3, n)), 0.0);
        m.update(&menu, &mut store, &MenuInput::default(), 0.1);
        assert_eq!(val(&store, "SmileOpen"), 0.0);
        assert!(!m.fast_sync() && !m.is_open());
    }

    #[test]
    fn views_mark_the_highlighted_slice_and_toggle_state() {
        let (menu, mut store) = setup();
        let mut m = RadialMenu::default();
        let n = menu.root.len();
        m.update(&menu, &mut store, &select(toward(0, n)), 0.0);
        m.update(&menu, &mut store, &open(toward(1, n)), 0.1);
        let v = m.view(&menu, &store);
        assert_eq!(v.slices.len(), n);
        assert!(v.slices[0].on && !v.slices[1].on);
        assert!(v.slices[1].highlighted && !v.slices[0].highlighted);
        assert_eq!(v.slices[1].badge, ">");
    }

    #[test]
    fn sector_geometry_is_a_closed_ring_piece() {
        let pts = sector_points(Pos2::new(100.0, 100.0), 20.0, 60.0, 0.0, std::f32::consts::FRAC_PI_2);
        // The outer arc starts straight above the centre at radius 60 and the
        // inner arc ends at radius 20.
        assert!((pts[0].x - 100.0).abs() < 1e-3 && (pts[0].y - 40.0).abs() < 1e-3);
        let last = pts.last().unwrap();
        assert!((last.x - 100.0).abs() < 1e-3 && (last.y - 80.0).abs() < 1e-3);
        assert!(pts.iter().all(|p| {
            let d = ((p.x - 100.0).powi(2) + (p.y - 100.0).powi(2)).sqrt();
            (19.9..=60.1).contains(&d)
        }));
    }

    #[test]
    fn desktop_input_prefers_keys_then_the_pointer() {
        let rect = egui::Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));
        let i = desktop_input(rect, true, Some(Pos2::new(400.0 + 200.0, 300.0)), GVec2::ZERO, false, false, false);
        assert!(i.dir.x > 0.9 && i.dir.y.abs() < 1e-3, "{:?}", i.dir);
        let k = desktop_input(rect, true, Some(Pos2::new(0.0, 0.0)), GVec2::new(0.0, 1.0), false, false, false);
        assert_eq!(k.dir, GVec2::new(0.0, 1.0));
        assert!(desktop_input(rect, false, None, GVec2::ZERO, false, false, false).dir == GVec2::ZERO);
    }
}
