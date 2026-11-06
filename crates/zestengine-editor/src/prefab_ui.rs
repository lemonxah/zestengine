//! Prefab editing UI: override marks in the inspector, the Overrides
//! dropdown with its side-by-side diff, and the Prefab Mode header bar.
//!
//! The editor crate knows nothing about assets, so everything here works on
//! plain strings: the engine fills [`PrefabPanelData`] / [`PrefabInspect`]
//! from the live instance and acts on the [`PrefabRequest`]s the widgets
//! leave in egui's temp data ([`PREFAB_REQUESTS_KEY`]).

use egui::{Color32, RichText, Sense, Ui, Vec2};
use egui_phosphor::regular as ph;

use crate::theme;
use crate::widgets;

/// egui temp-data key: the [`PrefabInspect`] of the object being inspected.
pub const PREFAB_INSPECT_KEY: &str = "zestengine-prefab-inspect";
/// egui temp-data key: `Vec<PrefabRequest>` raised by the widgets this frame.
pub const PREFAB_REQUESTS_KEY: &str = "zestengine-prefab-requests";
/// egui temp-data key: the `(label, path)` pairs the rows being drawn match.
const ACTIVE_KEY: &str = "zestengine-prefab-active";

/// What a Revert / Apply covers.
#[derive(Clone, Debug, PartialEq)]
pub enum PrefabTarget {
    /// Every change on the instance.
    All,
    /// Every override on one object (its source id).
    Object(String),
    Property { target: String, path: String },
    /// An added object (its live id) and what is below it.
    Added(String),
    /// A deleted object (its source id).
    RemovedObject(String),
    RemovedComponent { target: String, key: String },
}

#[derive(Clone, Debug, PartialEq)]
pub enum PrefabRequest {
    /// Write the change into the prefab at level `level` (index into the
    /// inspected object's `levels`).
    Apply { target: PrefabTarget, level: usize },
    Revert { target: PrefabTarget },
    /// Open the instance's asset in Prefab Mode.
    OpenAsset,
    /// Show the asset in the Files panel.
    SelectAsset,
    Unpack,
    UnpackCompletely,
    RemoveUnused,
    /// Apply/Revert several rows at once (the dropdown's "Selected" buttons).
    Batch(Vec<PrefabRequest>),
}

/// Push a request for the engine to act on after this frame.
pub fn raise(ui: &Ui, req: PrefabRequest) {
    let key = egui::Id::new(PREFAB_REQUESTS_KEY);
    ui.data_mut(|d| {
        let mut v = d.get_temp::<Vec<PrefabRequest>>(key).unwrap_or_default();
        v.push(req);
        d.insert_temp(key, v);
    });
}

/// Requests raised since the last call.
pub fn take_requests(ctx: &egui::Context) -> Vec<PrefabRequest> {
    ctx.data_mut(|d| d.remove_temp::<Vec<PrefabRequest>>(egui::Id::new(PREFAB_REQUESTS_KEY))).unwrap_or_default()
}

// ---------------------------------------------------------------- inspector marks

/// One component of the inspected object, as far as the instance differs.
#[derive(Clone, Debug, Default)]
pub struct PrefabComponent {
    /// `Kind` or `Kind#1` (the key overrides use).
    pub key: String,
    /// The component doesn't exist in the asset.
    pub added: bool,
    /// Overridden fields: (normalized field name, override path).
    pub fields: Vec<(String, String)>,
    /// The whole component differs and can't be split into fields.
    pub whole_path: Option<String>,
}

/// A component the instance deleted.
#[derive(Clone, Debug, Default)]
pub struct PrefabRemovedComponent {
    pub key: String,
    pub kind: String,
}

/// Override state of the object the inspector shows.
#[derive(Clone, Debug, Default)]
pub struct PrefabInspect {
    /// Source id of the object in the asset (what overrides target).
    pub target: String,
    /// Apply menu entries, outermost asset first ("Apply to Prefab 'Ship'", ...).
    pub levels: Vec<String>,
    /// Object fields: (normalized label, override path).
    pub entry: Vec<(String, String)>,
    pub components: Vec<PrefabComponent>,
    pub removed_components: Vec<PrefabRemovedComponent>,
    /// The object is not in the asset (it was added to the instance).
    pub added: bool,
    /// Instance root (or nested root): Name/Position/Rotation are placement,
    /// not overrides.
    pub is_root: bool,
}

/// Publish the inspected object's override state for the rows drawn after
/// this call. Pair with [`clear_inspect`] after the inspector.
pub fn publish_inspect(ui: &Ui, inspect: PrefabInspect) {
    ui.data_mut(|d| {
        d.insert_temp(egui::Id::new(ACTIVE_KEY), inspect.entry.clone());
        d.insert_temp(egui::Id::new(PREFAB_INSPECT_KEY), inspect);
    });
}

pub fn clear_inspect(ui: &Ui) {
    ui.data_mut(|d| {
        d.remove::<PrefabInspect>(egui::Id::new(PREFAB_INSPECT_KEY));
        d.remove::<Vec<(String, String)>>(egui::Id::new(ACTIVE_KEY));
    });
}

fn inspect(ui: &Ui) -> Option<PrefabInspect> {
    ui.data(|d| d.get_temp::<PrefabInspect>(egui::Id::new(PREFAB_INSPECT_KEY)))
}

/// Marker for a component section: returns the y to start the margin bar at.
pub struct ComponentMark {
    start: f32,
    overridden: bool,
}

/// Call before drawing a component's header; matches the component by its
/// key and switches the rows that follow to its fields.
pub fn begin_component(ui: &Ui, key: &str) -> ComponentMark {
    let mut overridden = false;
    if let Some(p) = inspect(ui) {
        let comp = p.components.iter().find(|c| c.key == key);
        overridden = comp.is_some_and(|c| c.added || c.whole_path.is_some() || !c.fields.is_empty());
        let fields = comp.map(|c| c.fields.clone()).unwrap_or_default();
        ui.data_mut(|d| d.insert_temp(egui::Id::new(ACTIVE_KEY), fields));
    }
    ComponentMark { start: ui.cursor().top(), overridden }
}

/// Call after the component's body: draws the 2 px margin bar beside an
/// overridden component and puts the object's own fields back.
pub fn end_component(ui: &Ui, mark: ComponentMark) {
    if let Some(p) = inspect(ui) {
        ui.data_mut(|d| d.insert_temp(egui::Id::new(ACTIVE_KEY), p.entry));
    }
    if mark.overridden {
        let x = ui.max_rect().left() - 6.0;
        let rect = egui::Rect::from_min_max(egui::pos2(x, mark.start), egui::pos2(x + 2.0, ui.cursor().top()));
        ui.painter().rect_filled(rect, 0.0, theme::PREFAB_OVERRIDE);
    }
}

/// Ghost headers for components the instance deleted, each with a Revert.
pub fn removed_components_ui(ui: &mut Ui) {
    let Some(p) = inspect(ui) else { return };
    for c in &p.removed_components {
        ui.horizontal(|ui| {
            ui.set_min_height(theme::H_FIELD);
            let bar = ui.max_rect().left() - 6.0;
            let top = ui.cursor().top();
            ui.painter().rect_filled(
                egui::Rect::from_min_max(egui::pos2(bar, top), egui::pos2(bar + 2.0, top + theme::H_FIELD)),
                0.0,
                theme::PREFAB_OVERRIDE,
            );
            ui.label(RichText::new(format!("{} {} (removed)", ph::MINUS_CIRCLE, c.kind)).color(theme::TEXT_DIM));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if widgets::ghost_button(ui, "Revert", theme::H_BUTTON_SMALL).clicked() {
                    raise(ui, PrefabRequest::Revert { target: PrefabTarget::RemovedComponent { target: p.target.clone(), key: c.key.clone() } });
                }
            });
        });
    }
}

/// Inspector rows that show a component field under a label that isn't the
/// field's name: `(component, field, normalized row labels)`. The inspector
/// matches override marks to rows by label, so a field whose row is labelled
/// differently needs an entry here to get its bold label and margin bar. The
/// `field_rows_cover_every_inspector_label` test lists what is missing.
pub const FIELD_ROWS: &[(&str, &str, &[&str])] = &[
    ("Box Collider", "material", &["physicsmaterial"]),
    ("Sphere Collider", "material", &["physicsmaterial"]),
    ("Capsule Collider", "material", &["physicsmaterial"]),
    ("Mesh Collider", "material", &["physicsmaterial"]),
    ("Terrain Collider", "material", &["physicsmaterial"]),
    ("Box Collider 2D", "material", &["physicsmaterial"]),
    ("Capsule Collider 2D", "material", &["physicsmaterial"]),
    ("Circle Collider 2D", "material", &["physicsmaterial"]),
    ("Composite Collider 2D", "material", &["physicsmaterial"]),
    ("Edge Collider 2D", "material", &["physicsmaterial"]),
    ("Polygon Collider 2D", "material", &["physicsmaterial"]),
    ("Rigid Body", "material", &["physicsmaterial"]),
    ("Rigidbody 2D", "material", &["physicsmaterial"]),
    ("Acoustic Material", "dynamic", &["movingoccluder"]),
    ("Animator", "looping", &["loop"]),
    ("Animator", "culling", &["cullingmode"]),
    ("Audio Probe Volume", "reverb", &["bakereverb"]),
    ("Audio Probe Volume", "pathing", &["bakepathing"]),
    ("Audio Source", "looping", &["loop"]),
    ("Audio Source", "bus", &["mixerbus"]),
    ("Audio Source", "directivity_sharpness", &["directivity"]),
    ("Audio Source", "spatialize_via_ambisonics", &["viaambisonics"]),
    ("Avatar Menu Item", "kind", &["type"]),
    ("Avatar Menu Item", "is_default", &["default"]),
    ("Avatar Parameters", "budget_bits", &["bitbudget"]),
    ("Behaviour Tree", "root", &["asset"]),
    ("Camera", "id", &["cameraid"]),
    ("Camera", "fov_deg", &["fieldofview"]),
    ("Camera", "texture_size", &["rendertexture"]),
    ("Camera", "ortho_size", &["size"]),
    ("Click To Move", "max_ray_distance", &["maxray"]),
    ("Cloth Colliders", "colliders", &["bones"]),
    ("Cloth Colliders", "fit", &["coverage"]),
    ("Effector 2D", "kind", &["type"]),
    ("Effector 2D", "layer_mask", &["uselayermask"]),
    ("Effector 2D", "flow_angle", &["forceangle"]),
    ("Effector 2D", "flow_magnitude", &["forcemagnitude"]),
    ("Effector 2D", "flow_variation", &["forcevariation"]),
    ("Face Tracking", "eye_angle", &["eyeyawpitch"]),
    ("First Person Character", "crouch_height", &["crouchcapsuleheight"]),
    ("First Person Character", "crouch_toggle", &["crouchtoggles"]),
    ("Health", "regen_per_sec", &["regensec"]),
    ("Hitscan Weapon", "spread_deg", &["spread"]),
    ("Hitscan Weapon", "magazine_size", &["magazine"]),
    ("Hitscan Weapon", "ammo", &["loaded"]),
    ("Hitscan Weapon", "hit_effect", &["hitburst"]),
    ("Humanoid IK", "plant_feet", &["plantandstep"]),
    ("Joint", "kind", &["type"]),
    ("Joint", "limits_enabled", &["limits"]),
    ("Joint", "motor_enabled", &["motor"]),
    ("Joint 2D", "kind", &["type"]),
    ("Joint 2D", "limits_enabled", &["limits"]),
    ("Joint 2D", "motor_enabled", &["motor"]),
    ("LOD Group", "levels", &["byscreensize"]),
    ("Mesh Cloth", "pin", &["pinby"]),
    ("Mesh Cloth", "pin_channel", &["channel"]),
    ("Mesh Cloth", "shape_stiffness", &["keepshape"]),
    ("Mesh Cloth", "collider_objects", &["allcolliders"]),
    ("Mesh Cloth", "teleport", &["teleportdistance"]),
    ("NavMesh Surface", "size", &["boundssize"]),
    ("Network Manager", "pin_host", &["pinhostidentity"]),
    ("Network Manager", "account_list", &["accounts"]),
    ("Network Manager", "discovery", &["landiscovery"]),
    ("Network Object", "sync_transform", &["transform"]),
    ("Network Object", "sync_scale", &["scale"]),
    ("Network Object", "sync_rigidbody", &["rigidbody"]),
    ("Network Object", "sync_parent", &["parent"]),
    ("Network Object", "sync_active", &["active"]),
    ("Network Object", "interpolation", &["mode"]),
    ("Network Object", "interp_delay", &["delay"]),
    ("Network Object", "position_threshold", &["movethreshold"]),
    ("Network Object", "rotation_threshold", &["turnthreshold"]),
    ("Network Object", "despawn_on_owner_leave", &["despawnonleave"]),
    ("Network Object", "adaptive_delay", &["adaptive"]),
    ("Particle Emitter", "shape_radius", &["spawnradius"]),
    ("RTS Selection", "ui_reference_height", &["uirefheight"]),
    ("Reflection Probe", "lock_box", &["lockboxinworld"]),
    ("Reflection Probe", "update_hz", &["updaterate"]),
    ("Reflection Probe", "include_dynamic", &["movingobjects"]),
    ("Rigid Body", "kind", &["type"]),
    ("Rigidbody 2D", "linear_damping", &["lineardrag"]),
    ("Rigidbody 2D", "angular_damping", &["angulardrag"]),
    ("Rigidbody 2D", "freeze_x", &["freeze"]),
    ("Rigidbody 2D", "freeze_y", &["freeze"]),
    ("Rigidbody 2D", "freeze_rotation", &["freeze"]),
    ("Rigidbody 2D", "sleep_mode", &["sleeping"]),
    ("Rigidbody 2D", "interpolation", &["interpolate"]),
    ("Spin", "degrees_per_second", &["speed"]),
    ("Swimmer", "stroke_accel", &["strokeacceleration"]),
    ("Template Avatar", "follow_local_player", &["followlocalvrplayer"]),
    ("Terrain", "heightmap_scale", &["heightscale"]),
    ("Terrain", "heightmap_offset", &["heightoffset"]),
    ("Text 3D", "loc_key", &["localizationkey"]),
    ("Text 3D", "auto_size_min", &["autosize"]),
    ("Text 3D", "auto_size_max", &["autosize"]),
    ("Text 3D", "char_spacing", &["characterspacing"]),
    ("Trigger Events", "first_and_last_only", &["firstinlastout"]),
    ("UI Canvas", "match_height", &["matchwidthheight"]),
    ("UI Rect", "anchor_min", &["anchor"]),
    ("UI Rect", "anchor_max", &["anchor"]),
    ("UI Slider", "track_color", &["track"]),
    ("UI Slider", "fill_color", &["fill"]),
    ("UI Slider", "handle_color", &["handle"]),
    ("UI Text", "loc_key", &["localizationkey"]),
    ("UI Text", "auto_size_min", &["autosize"]),
    ("UI Text", "auto_size_max", &["autosize"]),
    ("UI Text", "char_spacing", &["characterspacing"]),
    ("UI Text", "margins", &["marginleft", "margintop", "marginright", "marginbottom"]),
    ("UI Toggle", "box_color", &["box"]),
    ("UI Toggle", "check_color", &["check"]),
    ("Voice Chat", "min_distance", &["fullvolumewithin"]),
    ("Voice Chat", "max_distance", &["silentpast"]),
    ("Voice Chat", "activation_db", &["activation"]),
    ("Water", "smooth_shape", &["smoothoutline"]),
    ("Water", "gpu_sim", &["gpusimulation"]),
    ("Water", "ripple_area", &["area"]),
    ("Water", "ripple_resolution", &["resolution"]),
    ("Water", "ripple_damping", &["damping"]),
    ("Water", "sim_wave_depth", &["wavedepth"]),
    ("Water", "sim_displacement", &["displacement"]),
    ("Water", "sim_momentum", &["momentum"]),
    ("Water", "sim_dispersion", &["dispersion"]),
    ("Water", "sim_ripple_mix", &["ripples"]),
    ("Water", "sim_chop_smoothing", &["chopsmoothing"]),
    ("Water", "sim_wake_ripples", &["wakeripples"]),
    ("Water", "sim_ripple_smoothing", &["ripplesmoothing"]),
    ("Water", "sim_fine_damping", &["finerippledamping"]),
    ("Water", "shore_depth", &["shorecalmdepth"]),
    ("Water", "projection_strength", &["projectcaustics"]),
    ("Water Container", "center", &["centre"]),
    ("Water Container", "half_extents", &["halfsize"]),
    ("Water Container", "initial_fill", &["startfill"]),
    ("Water Container", "color", &["colour"]),
    ("Waterfall", "flow_speed", &["exitspeed"]),
    ("Waterfall", "color", &["colour"]),
    ("Weapon Hold", "editing", &["editpose"]),
    ("Wind Zone", "frequency", &["gustfrequency"]),
];

/// Row labels (normalized, see `widgets::mixed_key`) that show `field` of a
/// component of `kind`: its own name plus the explicit entries of [`FIELD_ROWS`].
pub fn rows_of_field(kind: &str, field: &str) -> Vec<String> {
    let mut rows = vec![widgets::mixed_key(field)];
    for (k, f, labels) in FIELD_ROWS {
        if *k == kind && *f == field {
            rows.extend(labels.iter().map(|l| (*l).to_owned()));
        }
    }
    rows
}

/// How a property row labelled `label` should look: Some(path) when it is
/// overridden on the inspected instance.
pub fn row_override(ui: &Ui, label: &str) -> Option<String> {
    let key = widgets::mixed_key(label);
    if key.is_empty() {
        return None;
    }
    let active = ui.data(|d| d.get_temp::<Vec<(String, String)>>(egui::Id::new(ACTIVE_KEY)))?;
    active.into_iter().find(|(k, _)| *k == key).map(|(_, p)| p)
}

/// The 2 px margin line beside an overridden row.
pub fn paint_row_bar(ui: &Ui, row: egui::Rect) {
    let x = ui.max_rect().left() - 6.0;
    ui.painter().rect_filled(
        egui::Rect::from_min_max(egui::pos2(x, row.top()), egui::pos2(x + 2.0, row.bottom())),
        0.0,
        theme::PREFAB_OVERRIDE,
    );
}

/// Right-click menu of an overridden row: Apply per level, Revert.
pub fn property_menu(ui: &mut Ui, path: &str) {
    widgets::menu_frame(ui);
    let Some(p) = inspect(ui) else { return };
    let target = PrefabTarget::Property { target: p.target.clone(), path: path.to_owned() };
    for (i, level) in p.levels.iter().enumerate() {
        if widgets::menu_item(ui, None, level, None).clicked() {
            raise(ui, PrefabRequest::Apply { target: target.clone(), level: i });
            ui.close();
        }
    }
    if widgets::menu_item(ui, None, "Revert", None).clicked() {
        raise(ui, PrefabRequest::Revert { target });
        ui.close();
    }
}

// ---------------------------------------------------------------- Overrides dropdown

/// One field that differs, for the diff.
#[derive(Clone, Debug)]
pub struct PrefabFieldRow {
    pub path: String,
    pub pristine: String,
    pub live: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefabRowKind {
    Modified,
    Added,
    RemovedObject,
}

#[derive(Clone, Debug)]
pub struct PrefabChangeRow {
    pub kind: PrefabRowKind,
    pub name: String,
    /// Source id (modified / removed) or live id (added).
    pub id: String,
    pub fields: Vec<PrefabFieldRow>,
    /// Keys of components deleted from this object.
    pub removed_components: Vec<String>,
}

/// Everything the Overrides dropdown shows.
#[derive(Clone, Debug, Default)]
pub struct PrefabPanelData {
    pub asset_name: String,
    pub asset_path: String,
    /// Apply menu entries for the instance as a whole.
    pub levels: Vec<String>,
    pub rows: Vec<PrefabChangeRow>,
    /// Overrides whose target is gone from the asset.
    pub unused: usize,
    /// The asset is missing or unreadable.
    pub missing: bool,
}

/// State kept between frames by the dropdown.
#[derive(Default)]
pub struct OverridesState {
    /// Waiting for the answer to "Remove Unused Overrides?".
    pub confirm_unused: bool,
    pub shown: Option<usize>,
    pub checked: std::collections::HashSet<usize>,
}

/// The inspector's prefab row: asset chip, Overrides dropdown, Open, Select.
pub fn prefab_header(ui: &mut Ui, data: &PrefabPanelData, state: &mut OverridesState) {
    ui.horizontal(|ui| {
        ui.set_min_height(theme::H_FIELD);
        widgets::row_label(ui, "Prefab");
        let name_color = if data.missing { theme::STATUS_ERROR } else { theme::PREFAB_NAME };
        ui.label(RichText::new(format!("{} {}", ph::CUBE, data.asset_name)).font(theme::medium(theme::SIZE_BODY)).color(name_color)).on_hover_text(&data.asset_path);
        let changes = data.rows.len() + data.unused;
        let label = if changes > 0 { format!("Overrides ({changes})") } else { "Overrides".to_owned() };
        let btn = widgets::secondary_button_sized(ui, None, &label, theme::H_BUTTON_SMALL);
        egui::Popup::menu(&btn).close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside).show(|ui| overrides_popup(ui, data, state));
        if widgets::secondary_button_sized(ui, None, "Open", theme::H_BUTTON_SMALL).on_hover_text("Edit the prefab in Prefab Mode").clicked() {
            raise(ui, PrefabRequest::OpenAsset);
        }
        if widgets::secondary_button_sized(ui, None, "Select", theme::H_BUTTON_SMALL).on_hover_text("Show the prefab asset in Files").clicked() {
            raise(ui, PrefabRequest::SelectAsset);
        }
    });
}

fn field_label(path: &str) -> String {
    match crate::prefab_ui::split_path(path) {
        (Some(kind), Some(field)) => format!("{kind} / {field}"),
        (Some(kind), None) => kind.to_owned(),
        _ => {
            let mut c = path.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + &c.as_str().replace('_', " "),
                None => String::new(),
            }
        }
    }
}

/// `components[Kind].field` -> (Some(Kind), Some(field)).
pub fn split_path(path: &str) -> (Option<&str>, Option<&str>) {
    let Some(rest) = path.strip_prefix("components[") else { return (None, None) };
    let Some(close) = rest.find(']') else { return (None, None) };
    (Some(&rest[..close]), rest[close + 1..].strip_prefix('.'))
}

fn overrides_popup(ui: &mut Ui, data: &PrefabPanelData, state: &mut OverridesState) {
    ui.set_min_width(460.0);
    ui.set_max_width(460.0);
    ui.label(RichText::new("Review, Revert or Apply Overrides").font(theme::semibold(theme::SIZE_TITLE)).color(theme::TEXT_STRONG));
    ui.label(RichText::new(format!("{} {}", ph::CUBE, data.asset_name)).color(theme::PREFAB_NAME));
    widgets::divider(ui);
    if data.rows.is_empty() && data.unused == 0 {
        widgets::info_note(ui, "No overrides: this instance matches the prefab.");
        return;
    }
    egui::ScrollArea::vertical().max_height(260.0).auto_shrink([false, true]).show(ui, |ui| {
        for (i, row) in data.rows.iter().enumerate() {
            ui.horizontal(|ui| {
                let mut on = state.checked.contains(&i);
                if ui.checkbox(&mut on, "").changed() {
                    if on {
                        state.checked.insert(i);
                    } else {
                        state.checked.remove(&i);
                    }
                }
                let (glyph, color) = match row.kind {
                    PrefabRowKind::Modified => (ph::PENCIL_SIMPLE, theme::PREFAB_NAME),
                    PrefabRowKind::Added => (ph::PLUS_CIRCLE, theme::STATUS_OK),
                    PrefabRowKind::RemovedObject => (ph::MINUS_CIRCLE, theme::STATUS_ERROR),
                };
                let selected = state.shown == Some(i);
                let text = RichText::new(format!("{glyph} {}", row.name)).color(if selected { theme::TEXT_STRONG } else { color });
                if ui.selectable_label(selected, text).clicked() {
                    state.shown = if selected { None } else { Some(i) };
                }
                let n = row.fields.len() + row.removed_components.len();
                if n > 0 {
                    widgets::badge(ui, &n.to_string());
                }
            });
        }
        if data.unused > 0 {
            unused_overrides(ui, data.unused, state);
        }
    });
    if let Some(row) = state.shown.and_then(|i| data.rows.get(i)) {
        widgets::divider(ui);
        diff_view(ui, data, row);
    }
    widgets::divider(ui);
    let selected: Vec<usize> = {
        let mut v: Vec<usize> = state.checked.iter().copied().filter(|i| *i < data.rows.len()).collect();
        v.sort_unstable();
        v
    };
    let some = !selected.is_empty();
    ui.horizontal(|ui| {
        let batch = |ui: &Ui, apply: Option<usize>| {
            let pick: Vec<usize> = if some { selected.clone() } else { (0..data.rows.len()).collect() };
            let mut reqs = Vec::new();
            for i in pick {
                let r = &data.rows[i];
                let target = match r.kind {
                    PrefabRowKind::Modified => PrefabTarget::Object(r.id.clone()),
                    PrefabRowKind::Added => PrefabTarget::Added(r.id.clone()),
                    PrefabRowKind::RemovedObject => PrefabTarget::RemovedObject(r.id.clone()),
                };
                reqs.push(match apply {
                    Some(level) => PrefabRequest::Apply { target, level },
                    None => PrefabRequest::Revert { target },
                });
                if let PrefabRowKind::Modified = r.kind {
                    for key in &r.removed_components {
                        let t = PrefabTarget::RemovedComponent { target: r.id.clone(), key: key.clone() };
                        reqs.push(match apply {
                            Some(level) => PrefabRequest::Apply { target: t, level },
                            None => PrefabRequest::Revert { target: t },
                        });
                    }
                }
            }
            if !some {
                // Everything: one request covers it (and keeps unused overrides).
                reqs = vec![match apply {
                    Some(level) => PrefabRequest::Apply { target: PrefabTarget::All, level },
                    None => PrefabRequest::Revert { target: PrefabTarget::All },
                }];
            }
            raise(ui, PrefabRequest::Batch(reqs));
        };
        if widgets::secondary_button_sized(ui, None, if some { "Revert Selected" } else { "Revert All" }, theme::H_BUTTON_SMALL).clicked() {
            batch(ui, None);
            ui.close();
        }
        let apply = widgets::primary_button_sized(ui, None, if some { "Apply Selected" } else { "Apply All" }, theme::H_BUTTON_SMALL);
        if data.levels.len() > 1 {
            egui::Popup::menu(&apply).show(|ui| {
                widgets::menu_frame(ui);
                for (i, level) in data.levels.iter().enumerate() {
                    if widgets::menu_item(ui, None, level, None).clicked() {
                        batch(ui, Some(i));
                        ui.close();
                    }
                }
            });
        } else if apply.clicked() {
            batch(ui, Some(0));
            ui.close();
        }
    });
}

/// The unused-overrides line: what they are, and a two-step Remove (they can't be brought back).
fn unused_overrides(ui: &mut Ui, count: usize, state: &mut OverridesState) {
    let plural = if count == 1 { "" } else { "s" };
    ui.label(RichText::new(format!("{count} unused override{plural}: the object or property is gone from the prefab")).color(theme::STATUS_WARN_FG));
    ui.horizontal(|ui| {
        if state.confirm_unused {
            ui.label(RichText::new(format!("Remove {count} unused override{plural}? This can't be undone from here.")).color(theme::TEXT_SECONDARY));
            if widgets::primary_button_sized(ui, None, "Remove", theme::H_BUTTON_SMALL).clicked() {
                raise(ui, PrefabRequest::RemoveUnused);
                state.confirm_unused = false;
            }
            if widgets::secondary_button_sized(ui, None, "Cancel", theme::H_BUTTON_SMALL).clicked() {
                state.confirm_unused = false;
            }
        } else if widgets::ghost_button(ui, "Remove Unused Overrides", theme::H_BUTTON_SMALL)
            .on_hover_text("Delete the overrides that no longer apply to anything in the prefab")
            .clicked()
        {
            state.confirm_unused = true;
        }
    });
}

/// "Prefab Source" beside "Override" for one changed object.
fn diff_view(ui: &mut Ui, data: &PrefabPanelData, row: &PrefabChangeRow) {
    let width = (ui.available_width() - 8.0) * 0.5;
    let value = |s: &str| {
        let s = s.trim();
        if s.chars().count() > 90 { format!("{}…", s.chars().take(90).collect::<String>()) } else { s.to_owned() }
    };
    match row.kind {
        PrefabRowKind::Modified => {
            ui.horizontal(|ui| {
                ui.allocate_ui(Vec2::new(width, 0.0), |ui| ui.label(RichText::new("Prefab Source").font(theme::semibold(theme::SIZE_META)).color(theme::TEXT_MUTED)));
                ui.allocate_ui(Vec2::new(width, 0.0), |ui| ui.label(RichText::new("Override").font(theme::semibold(theme::SIZE_META)).color(theme::PREFAB_OVERRIDE)));
            });
            for f in &row.fields {
                ui.label(RichText::new(field_label(&f.path)).font(theme::semibold(theme::SIZE_META)).color(theme::TEXT_SECONDARY));
                ui.horizontal(|ui| {
                    ui.allocate_ui(Vec2::new(width, 0.0), |ui| {
                        egui::Frame::new().fill(theme::BG_FIELD).corner_radius(theme::R_FIELD).inner_margin(4).show(ui, |ui| {
                            ui.set_width(width - 8.0);
                            ui.label(RichText::new(if f.pristine.is_empty() { "(none)".to_owned() } else { value(&f.pristine) }).font(theme::mono(theme::SIZE_META)).color(theme::TEXT_MUTED));
                        });
                    });
                    ui.allocate_ui(Vec2::new(width, 0.0), |ui| {
                        egui::Frame::new().fill(theme::BG_FIELD).corner_radius(theme::R_FIELD).inner_margin(4).show(ui, |ui| {
                            ui.set_width(width - 8.0);
                            ui.label(RichText::new(value(&f.live)).font(theme::mono(theme::SIZE_META)).color(theme::TEXT_STRONG));
                        });
                    });
                });
                ui.horizontal(|ui| {
                    if widgets::ghost_button(ui, "Revert", theme::H_BUTTON_SMALL).clicked() {
                        raise(ui, PrefabRequest::Revert { target: PrefabTarget::Property { target: row.id.clone(), path: f.path.clone() } });
                    }
                    for (i, level) in data.levels.iter().enumerate() {
                        let label = if data.levels.len() == 1 { "Apply".to_owned() } else { level.clone() };
                        if widgets::ghost_button(ui, &label, theme::H_BUTTON_SMALL).clicked() {
                            raise(ui, PrefabRequest::Apply { target: PrefabTarget::Property { target: row.id.clone(), path: f.path.clone() }, level: i });
                        }
                    }
                });
            }
            for key in &row.removed_components {
                ui.label(RichText::new(format!("{} {key} removed", ph::MINUS_CIRCLE)).color(theme::STATUS_ERROR));
            }
        }
        PrefabRowKind::Added => {
            ui.label(RichText::new("Added").font(theme::semibold(theme::SIZE_META)).color(theme::STATUS_OK));
            ui.label(RichText::new(format!("{} exists only on this instance.", row.name)).color(theme::TEXT_MUTED));
        }
        PrefabRowKind::RemovedObject => {
            ui.label(RichText::new("Removed").font(theme::semibold(theme::SIZE_META)).color(theme::STATUS_ERROR));
            ui.add_enabled(false, egui::Label::new(RichText::new(format!("{} was deleted from this instance.", row.name))));
        }
    }
}

// ---------------------------------------------------------------- Prefab Mode bar

/// What the Prefab Mode header shows.
pub struct StageBar<'a> {
    /// Crumbs from the main scene down to the prefab being edited.
    pub crumbs: &'a [String],
    pub dirty: bool,
    pub autosave: &'a mut bool,
}

#[derive(Default)]
pub struct StageBarResponse {
    /// Leave Prefab Mode (one level up).
    pub back: bool,
    /// A crumb above the last was clicked.
    pub crumb: Option<usize>,
    pub save: bool,
}

/// The header bar over the viewport in Prefab Mode: back arrow, breadcrumbs,
/// Save and Auto Save.
pub fn stage_bar(ui: &mut Ui, bar: StageBar<'_>) -> StageBarResponse {
    let mut out = StageBarResponse::default();
    egui::Frame::new().fill(theme::BG_PANEL).inner_margin(egui::Margin::symmetric(theme::PAD as i8, 0)).show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.set_height(theme::H_TOOLBAR);
            if widgets::icon_button(ui, crate::icons::Icon::ArrowUp, "Back to the scene", false).clicked() {
                out.back = true;
            }
            let last = bar.crumbs.len().saturating_sub(1);
            for (i, c) in bar.crumbs.iter().enumerate() {
                if i > 0 {
                    ui.label(RichText::new(ph::CARET_RIGHT).color(theme::TEXT_DIM));
                }
                if i == last {
                    ui.label(RichText::new(format!("{} {c}", ph::CUBE)).font(theme::semibold(theme::SIZE_BODY)).color(theme::PREFAB_NAME));
                } else if ui.add(egui::Label::new(RichText::new(c).color(theme::TEXT_MUTED)).sense(Sense::click())).clicked() {
                    out.crumb = Some(i);
                }
            }
            if bar.dirty {
                widgets::status_pill(ui, "Unsaved", theme::STATUS_WARN_FG);
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if widgets::secondary_button_sized(ui, None, "Save", theme::H_BUTTON_SMALL).clicked() {
                    out.save = true;
                }
                ui.checkbox(bar.autosave, "Auto Save");
            });
        });
    });
    out
}

/// Colour for the "added" overlay in the Hierarchy.
pub fn added_color() -> Color32 {
    theme::STATUS_OK
}

/// Test support: records the row labels an inspector draws.
#[cfg(test)]
pub(crate) mod audit {
    use std::cell::RefCell;

    thread_local! {
        static LABELS: RefCell<Option<Vec<String>>> = const { RefCell::new(None) };
    }

    pub fn start() {
        LABELS.with(|l| *l.borrow_mut() = Some(Vec::new()));
    }

    pub fn record(label: &str) {
        LABELS.with(|l| {
            if let Some(v) = l.borrow_mut().as_mut() {
                v.push(crate::widgets::mixed_key(label));
            }
        });
    }

    pub fn finish() -> Vec<String> {
        LABELS.with(|l| l.borrow_mut().take()).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{EditorComponents, InspectCtx};
    use zestengine_core::ComponentRegistry;

    /// Top-level field names of a component's RON state.
    fn field_names(data: &str) -> Vec<String> {
        let s = data.trim();
        let Some(open) = s.find('(') else { return Vec::new() };
        let inner = &s[open + 1..s.rfind(')').unwrap_or(s.len())];
        let (mut depth, mut lit, mut start, mut out) = (0i32, false, 0usize, Vec::new());
        let mut pieces = Vec::new();
        for (i, c) in inner.char_indices() {
            match c {
                '"' => lit = !lit,
                '(' | '[' | '{' if !lit => depth += 1,
                ')' | ']' | '}' if !lit => depth -= 1,
                ',' if !lit && depth == 0 => {
                    pieces.push(&inner[start..i]);
                    start = i + 1;
                }
                _ => {}
            }
        }
        pieces.push(&inner[start..]);
        for p in pieces {
            if let Some((name, _)) = p.split_once(':') {
                let name = name.trim();
                if !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                    out.push(name.to_owned());
                }
            }
        }
        out
    }

    /// Draw every builtin component's inspector once and report, per component,
    /// the fields no row is labelled after (neither by name nor in FIELD_ROWS).
    /// Many are intentionally not shown; the list is for filling FIELD_ROWS.
    fn unmatched() -> Vec<(String, Vec<String>, Vec<String>)> {
        let registry = ComponentRegistry::with_builtins();
        let editors = EditorComponents::with_builtins();
        let mut report = Vec::new();
        let ctx = egui::Context::default();
        crate::install_icon_font(&ctx);
        theme::apply(&ctx);
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        let mut names: Vec<&str> = registry.names().collect();
        names.sort_unstable();
        for name in names {
            let Some(mut comp) = registry.create(name) else { continue };
            let kind = comp.type_name().to_owned();
            let fields = field_names(&comp.save());
            audit::start();
            let _ = ctx.run(egui::RawInput::default(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    let objects = Vec::new();
                    let layers = Vec::new();
                    let ic = InspectCtx { objects: &objects, layer_names: &layers, asset_editor: None, has_body: false };
                    editors.inspect(zestengine_core::component_inner_mut(&mut comp), ui, &ic);
                });
            });
            let labels = audit::finish();
            let missing: Vec<String> = fields.iter().filter(|f| !rows_of_field(&kind, f).iter().any(|r| labels.contains(r))).cloned().collect();
            if !missing.is_empty() {
                report.push((kind, missing, labels));
            }
        }
        report
    }

    #[test]
    fn rows_of_field_adds_the_explicit_labels() {
        assert_eq!(rows_of_field("Light", "range"), vec!["range"]);
        assert_eq!(rows_of_field("Rigid Body", "kind"), vec!["kind", "type"]);
        assert_eq!(rows_of_field("UI Text", "margins").len(), 5);
    }

    #[test]
    fn field_rows_name_real_components_and_fields() {
        let registry = ComponentRegistry::with_builtins();
        for (kind, field, labels) in FIELD_ROWS {
            let comp = registry.create(kind).unwrap_or_else(|| panic!("FIELD_ROWS: no component '{kind}'"));
            assert_eq!(comp.type_name(), *kind, "the registry name is the display name overrides use");
            assert!(field_names(&comp.save()).iter().any(|f| f == field), "FIELD_ROWS: '{kind}' has no field '{field}'");
            assert!(!labels.is_empty() && labels.iter().all(|l| *l == widgets::mixed_key(l)), "labels are normalized: {labels:?}");
        }
    }

    #[test]
    #[ignore = "audit: prints the component fields no inspector row is labelled after"]
    fn print_unmatched_fields() {
        for (kind, missing, labels) in unmatched() {
            println!("{kind}: fields {missing:?}\n    rows {labels:?}");
        }
    }
}
