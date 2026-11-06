//! Inspectors and gizmos for `Water` and `Buoyancy`.

use zestengine_core::{Buoyancy, GerstnerWave, ObjectId, RiverPoint, Water, WaterBody};
use egui::{DragValue, RichText, Ui};
use glam::{Vec2, Vec3};

use crate::components::{EditorComponents, Gizmo, GizmoEdit, GizmoSpec, Inspect, InspectCtx};
use crate::icons::Icon;
use crate::theme;
use crate::widgets;

pub(crate) fn register(e: &mut EditorComponents) {
    e.register::<Water>();
    e.register::<Buoyancy>();
    e.register::<zestengine_core::SkeletalWaterInteraction>();
}

impl Inspect for zestengine_core::SkeletalWaterInteraction {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut c = false;
        for (label, enabled, radius) in [
            ("Hands", &mut self.hands, &mut self.hand_radius),
            ("Feet", &mut self.feet, &mut self.foot_radius),
            ("Arms", &mut self.arms, &mut self.arm_radius),
            ("Legs", &mut self.legs, &mut self.leg_radius),
        ] {
            widgets::property_row(ui, label, |ui| {
                c |= ui.checkbox(enabled, "").changed();
                c |= ui.add(DragValue::new(radius).speed(0.005).range(0.005..=1.0).suffix(" m radius")).changed();
            });
        }
        drag(ui, "Momentum", &mut self.momentum, 0.01, 0.0..=4.0, &mut c);
        note(ui, "Put on the avatar's skinned model. Uses the final animated/IK bone pose for independent limb wakes and splashes. Disable Character's Capsule affects water to avoid a second large footprint. No limb rigid bodies are required.");
        c
    }
}
impl Gizmo for zestengine_core::SkeletalWaterInteraction {}

const WATER: egui::Color32 = egui::Color32::from_rgb(80, 170, 230);
const BANK: egui::Color32 = egui::Color32::from_rgb(55, 120, 170);
const DEPTH_HANDLE: egui::Color32 = egui::Color32::from_rgb(190, 130, 245);
const FLOW_HANDLE: egui::Color32 = egui::Color32::from_rgb(90, 220, 150);
/// One metre of handle length represents one metre/second of current.
const FLOW_HANDLE_SCALE: f32 = 1.0;
const WIDTH_HANDLE: egui::Color32 = egui::Color32::from_rgb(170, 230, 255);
/// Gizmo specs ahead of a river's handles: the centre line, then the banks.
const RIVER_LINE_SPECS: usize = 2;

/// How the Scene view adds points while a river path is being edited.
pub const RIVER_PATH_HINT: &str = "Click: add a point at the end. Shift+Click: insert it where it falls along the river. \
Click on the centre line: split that segment. Ctrl+Click: add before the first point. \
Drag a point to move it, cyan for width, purple for depth, green for flow speed. Delete: remove the selected point. Esc: done.";

/// The river path editor's state, shared by the Water inspector and the Scene
/// view through egui memory: which object it belongs to, whether the path is
/// being built, and the selected point.
#[derive(Clone, Copy, Default, PartialEq)]
pub struct RiverPathEdit {
    pub object: Option<ObjectId>,
    pub active: bool,
    pub selected: Option<usize>,
    /// The Scene view picked `selected`; the inspector scrolls it into view.
    pub reveal: bool,
}

impl RiverPathEdit {
    fn id() -> egui::Id {
        egui::Id::new("zestengine-river-path-edit")
    }

    pub fn get(ctx: &egui::Context) -> Self {
        ctx.data(|d| d.get_temp(Self::id())).unwrap_or_default()
    }

    pub fn set(self, ctx: &egui::Context) {
        ctx.data_mut(|d| d.insert_temp(Self::id(), self));
    }
}

/// What a river gizmo handle edits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RiverHandle {
    Point(usize),
    Width(usize),
    Depth(usize),
    Flow(usize),
}

/// The river handle behind gizmo `index` of a river `Water`'s `gizmos()`.
pub fn river_handle(water: &Water, index: usize) -> Option<RiverHandle> {
    let n = water.river.len();
    match index.checked_sub(RIVER_LINE_SPECS)? {
        i if i < n => Some(RiverHandle::Point(i)),
        i if i < 2 * n => Some(RiverHandle::Width(i - n)),
        i if i < 3 * n => Some(RiverHandle::Depth(i - 2 * n)),
        i if i < 4 * n => Some(RiverHandle::Flow(i - 3 * n)),
        _ => None,
    }
}

/// Unit vector across the river at each point, as the surface mesh has it.
fn river_sides(water: &Water) -> Vec<Vec3> {
    let (sections, at_point) = water.river_sections();
    (0..water.river.len()).map(|k| at_point.get(k).map_or(Vec3::Z, |&i| sections[i].side)).collect()
}

fn drag(ui: &mut Ui, label: &str, v: &mut f32, speed: f64, range: std::ops::RangeInclusive<f32>, changed: &mut bool) {
    widgets::property_row(ui, label, |ui| *changed |= ui.add(DragValue::new(v).speed(speed).range(range)).changed());
}

/// `drag` with a hover tooltip on the value.
fn drag_tip(ui: &mut Ui, label: &str, tip: &str, v: &mut f32, speed: f64, range: std::ops::RangeInclusive<f32>, changed: &mut bool) {
    widgets::property_row(ui, label, |ui| *changed |= ui.add(DragValue::new(v).speed(speed).range(range)).on_hover_text(tip).changed());
}

/// The interactive layer: the GPU shallow-water simulation (or CPU ripples).
fn simulation_ui(ui: &mut Ui, w: &mut Water, c: &mut bool) {
    widgets::eyebrow(ui, "Simulation");
    widgets::property_row(ui, "Interactive", |ui| {
        *c |= ui.checkbox(&mut w.interactive, "").on_hover_text("Bodies, characters and falling water push the water around").changed();
    });
    if !w.interactive {
        return;
    }
    widgets::property_row(ui, "GPU simulation", |ui| {
        *c |= ui
            .checkbox(&mut w.gpu_sim, "")
            .on_hover_text("Shallow-water simulation on the GPU: the water has mass and flow, piles up, sloshes and pushes floating things. Off: lighter CPU ripples")
            .changed();
    });
    drag_tip(
        ui,
        "Area (m)",
        "Widest stretch simulated. Water that fits is simulated whole (its shores reflect); bigger water gets a window this wide around the camera",
        &mut w.ripple_area,
        0.5,
        2.0..=1000.0,
        c,
    );
    widgets::property_row(ui, "Resolution", |ui| {
        *c |= ui
            .add(DragValue::new(&mut w.ripple_resolution).range(32..=512))
            .on_hover_text("Cells per side (rounded to a multiple of 8; 64, 128, 256 or 512 for dispersive waves). 256 costs about a tenth of a millisecond of GPU time, about two with dispersive waves")
            .changed();
    });
    drag_tip(ui, "Damping", "Share of the water's motion lost per second", &mut w.ripple_damping, 0.005, 0.0..=0.99, c);
    if w.gpu_sim {
        drag_tip(
            ui,
            "Wave depth (m)",
            "Waves travel as if the water were at most this deep, so deep water doesn't race ripples away; it still holds its full volume",
            &mut w.sim_wave_depth,
            0.05,
            0.1..=50.0,
            c,
        );
        drag_tip(ui, "Displacement", "Scales the water a body pushes aside as it enters and moves (bow waves, troughs, splashes)", &mut w.sim_displacement, 0.01, 0.0..=4.0, c);
        drag_tip(ui, "Momentum", "How strongly moving bodies drag the water under them along", &mut w.sim_momentum, 0.01, 0.0..=4.0, c);
        drag_tip(
            ui,
            "Dispersion",
            "Deep-water wave layer over the simulation: short waves run slower than long ones, so impacts leave spreading wave trains and moving bodies a V-shaped wake. \
             The simulation keeps the water's mass and flow; this adds the waves. Runs weaker over shallow water, where the simulation already moves its waves right. 0 = off",
            &mut w.sim_dispersion,
            0.01,
            0.0..=4.0,
            c,
        );
        widgets::property_row(ui, "Chop smoothing", |ui| {
            *c |= ui
                .add(egui::Slider::new(&mut w.sim_chop_smoothing, 0.0..=1.0))
                .on_hover_text(
                    "Drains the short choppy waves bodies and impacts leave, the way turbulence does in real water, while long swells and wakes carry on. \
                     0 = they bounce round until Damping wears them down",
                )
                .changed();
        });
        widgets::property_row(ui, "Ripples", |ui| {
            *c |= ui
                .add(egui::Slider::new(&mut w.sim_ripple_mix, 0.0..=1.0))
                .on_hover_text("How much of the fine dispersive ripples show: 0 turns them off (the water still sloshes and makes its wake), 1 shows all of them")
                .changed();
        });
        if w.sim_dispersion > 0.0 && w.sim_ripple_mix > 0.0 {
            drag_tip(
                ui,
                "Wake ripples",
                "How strongly bodies moving through start waves: their wake. Drops and splashes are not scaled by this. 0 = moving bodies leave no ripples, only the push of the water",
                &mut w.sim_wake_ripples,
                0.01,
                0.0..=2.0,
                c,
            );
            drag_tip(
                ui,
                "Ripple smoothing (s)",
                "Spreads a body's push over this long. A body moves once a frame; pushing the water in those steps rings fine ripples all round it, while a real body moving steadily only leaves its wake",
                &mut w.sim_ripple_smoothing,
                0.005,
                0.0..=0.5,
                c,
            );
            drag_tip(
                ui,
                "Fine ripple damping",
                "How fast the finest waves the grid can hold die out (1/s). Higher clears grid-sized ripples and checker patterns; lower keeps more fine detail",
                &mut w.sim_fine_damping,
                0.1,
                0.0..=40.0,
                c,
            );
        }
        let cell = w.ripple_area / zestengine_core::sim_resolution(w.ripple_resolution) as f32;
        note(ui, &format!("Cells up to {:.0} cm. Runs in Play and while editing (waterfalls churn it).", cell * 100.0));
        if w.sim_dispersion > 0.0 {
            let depth = if w.body == WaterBody::River { w.river.iter().map(|p| p.depth).fold(0.0, f32::max) } else { w.depth };
            if zestengine_core::ewave_supported(zestengine_core::sim_resolution(w.ripple_resolution)) {
                note(ui, &format!("Dispersive waves at {:.0}% for {:.1} m deep water.", zestengine_core::dispersive_share(depth) * 100.0, depth));
            } else {
                note(ui, "Dispersive waves need a Resolution of 64, 128, 256 or 512.");
            }
        }
    }
}

fn color(ui: &mut Ui, label: &str, c: &mut [f32; 3], changed: &mut bool) {
    widgets::property_row(ui, label, |ui| *changed |= ui.color_edit_button_rgb(c).changed());
}

fn note(ui: &mut Ui, s: &str) {
    ui.label(RichText::new(s).size(theme::SIZE_META).color(theme::TEXT_DIM));
}

pub(crate) fn wave_asset_ui(ui: &mut Ui, path: &std::path::Path) -> Option<bool> {
    let mut profile = match zestengine_assets::water_file::load_water_waves(path) {
        Ok(profile) => profile,
        Err(e) => { ui.label(format!("Cannot open water waves: {e:#}")); return None; }
    };
    ui.heading("Water Waves");
    let mut changed = false;
    let mut generated = profile.spectrum.is_some();
    if ui.checkbox(&mut generated, "Generate from spectrum").changed() {
        if generated { profile.spectrum = Some(Default::default()); }
        else { profile.waves = profile.resolved(); profile.spectrum = None; }
        changed = true;
    }
    if let Some(s) = &mut profile.spectrum {
        widgets::property_row(ui, "Seed", |ui| changed |= ui.add(DragValue::new(&mut s.seed)).changed());
        widgets::property_row(ui, "Wave count", |ui| changed |= ui.add(DragValue::new(&mut s.count).range(1..=8)).changed());
        drag(ui, "Direction (deg)", &mut s.direction, 1.0, -360.0..=360.0, &mut changed);
        drag(ui, "Spread (deg)", &mut s.spread, 1.0, 0.0..=360.0, &mut changed);
        drag(ui, "Shortest wave (m)", &mut s.min_wavelength, 0.1, 0.05..=1000.0, &mut changed);
        drag(ui, "Longest wave (m)", &mut s.max_wavelength, 0.1, 0.05..=1000.0, &mut changed);
        drag(ui, "Significant height (m)", &mut s.height, 0.01, 0.0..=100.0, &mut changed);
        drag(ui, "Steepness", &mut s.steepness, 0.01, 0.0..=1.0, &mut changed);
        note(ui, "Deterministic directional Gerstner waves. Wind and local depth further shape each water body's waves.");
    } else {
        let mut remove = None;
        for (i, wave) in profile.waves.iter_mut().enumerate() {
            ui.push_id(i, |ui| {
                ui.horizontal(|ui| {
                    ui.label(format!("Wave {}", i + 1));
                    if ui.small_button("Remove").clicked() { remove = Some(i); }
                });
                drag(ui, "Direction (deg)", &mut wave.direction, 1.0, -360.0..=360.0, &mut changed);
                drag(ui, "Wavelength (m)", &mut wave.wavelength, 0.1, 0.05..=1000.0, &mut changed);
                drag(ui, "Amplitude (m)", &mut wave.amplitude, 0.01, 0.0..=50.0, &mut changed);
                drag(ui, "Steepness", &mut wave.steepness, 0.01, 0.0..=1.0, &mut changed);
                drag(ui, "Speed", &mut wave.speed, 0.01, 0.0..=10.0, &mut changed);
            });
        }
        if let Some(i) = remove { profile.waves.remove(i); changed = true; }
        if profile.waves.len() < Water::MAX_WAVES && ui.button("Add wave").clicked() {
            profile.waves.push(GerstnerWave::default()); changed = true;
        }
    }
    if changed && let Err(e) = zestengine_assets::water_file::save_water_waves(path, &profile) {
        tracing::error!("saving water waves: {e:#}");
        ui.label(format!("Cannot save water waves: {e:#}"));
    }
    Some(changed)
}

impl Inspect for Water {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut c = false;
        widgets::eyebrow(ui, "Body");
        widgets::property_row(ui, "Type", |ui| {
            let opts: Vec<(WaterBody, &str)> = WaterBody::ALL.iter().map(|b| (*b, b.label())).collect();
            let before = self.body;
            if widgets::segmented(ui, &mut self.body, &opts).changed() {
                c = true;
                if self.body == WaterBody::Pool && before != WaterBody::Pool {
                    self.make_pool();
                }
            }
        });
        match self.body {
            WaterBody::Ocean => note(ui, "Endless; the surface is this object's height."),
            WaterBody::Lake | WaterBody::Pool => {
                widgets::property_row(ui, "Size (X, Z)", |ui| {
                    let before = self.size;
                    c |= ui.add(DragValue::new(&mut self.size[0]).speed(0.1).range(0.0..=10_000.0)).changed();
                    c |= ui.add(DragValue::new(&mut self.size[1]).speed(0.1).range(0.0..=10_000.0)).changed();
                    if c && self.size != before {
                        // Keep the interactive simulation over the whole resized
                        // body (up to the widest window the sim can hold) so the
                        // physics doesn't leave a section of the water dry.
                        let needed = self.size[0].max(self.size[1]) * 1.2;
                        if needed > self.ripple_area {
                            self.ripple_area = needed.min(512.0);
                        }
                    }
                });
                widgets::property_row(ui, "Shape", |ui| {
                    if ui.small_button("Rectangle").on_hover_text("Back to the Size rectangle").clicked() {
                        self.shape.clear();
                        c = true;
                    }
                    for preset in zestengine_core::OutlinePreset::ALL {
                        if ui.small_button(preset.label()).on_hover_text("Outline points fitted to Size; drag them in the viewport").clicked() {
                            self.shape = preset.points(self.size);
                            c = true;
                        }
                    }
                });
                widgets::property_row(ui, "", |ui| {
                    if ui
                        .button("Fit to mesh")
                        .on_hover_text(
                            "Trace the outline and depth from the pool, spring or shore around this object: place it inside the basin at the water level first. \
                             Uses static objects and colliders on layers the water meets",
                        )
                        .clicked()
                    {
                        self.fit_request = true;
                        c = true;
                    }
                });
                widgets::property_row(ui, "Smooth outline", |ui| {
                    c |= ui.checkbox(&mut self.smooth_shape, "").on_hover_text("A smooth curve through the points instead of straight edges").changed();
                });
                note(ui, "Or outline it: with 3 or more points the outline replaces the rectangle. Drag the points in the viewport to fit the pool or spring; Add point splits the longest edge.");
                points2(ui, &mut self.shape, &mut c);
            }
            WaterBody::River => {
                c |= river_points_ui(ui, &mut self.river);
            }
        }
        if self.body != WaterBody::River {
            drag(ui, "Depth", &mut self.depth, 0.05, 0.0..=1000.0, &mut c);
        }
        if self.body == WaterBody::Pool {
            drag(ui, "Steam", &mut self.steam, 0.01, 0.0..=4.0, &mut c);
            note(ui, "Steam rising off warm water (hot tubs, hot springs); 0 = none.");
        }
        widgets::property_row(ui, "Mesh resolution", |ui| c |= ui.add(DragValue::new(&mut self.resolution).range(2..=512)).changed());

        widgets::eyebrow(ui, "Waves");
        c |= _ctx.file_field(ui, "Wave profile", &mut self.wave_profile, "water/Ocean.waterwaves", &["waterwaves"]);
        if !self.wave_profile.is_empty() {
            note(ui, "The shared wave profile drives rendering and buoyancy. Clear it to use the inline waves below.");
        }
        ui.add_enabled_ui(self.wave_profile.is_empty(), |ui| {
        let mut remove = None;
        for (k, w) in self.waves.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("Wave {}", k + 1)).color(theme::TEXT_SECONDARY));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if widgets::icon_button(ui, Icon::Trash, "Remove", false).clicked() {
                        remove = Some(k);
                    }
                });
            });
            drag(ui, "Direction (deg)", &mut w.direction, 1.0, -360.0..=360.0, &mut c);
            drag(ui, "Wavelength (m)", &mut w.wavelength, 0.05, 0.1..=500.0, &mut c);
            drag(ui, "Amplitude (m)", &mut w.amplitude, 0.005, 0.0..=20.0, &mut c);
            drag(ui, "Steepness", &mut w.steepness, 0.005, 0.0..=1.0, &mut c);
            drag(ui, "Speed", &mut w.speed, 0.01, 0.0..=10.0, &mut c);
        }
        if let Some(k) = remove {
            self.waves.remove(k);
            c = true;
        }
        if self.waves.len() < 8 && widgets::dashed_button(ui, Some(Icon::Plus), "Add wave").clicked() {
            self.waves.push(GerstnerWave::default());
            c = true;
        }
        });
        drag(ui, "Wind response", &mut self.wind_response, 0.01, 0.0..=4.0, &mut c);
        note(
            ui,
            &format!(
                "Wind Zones raise a sea along the wind (now {:.1} m/s); depth slows waves and caps their height, the water's size limits their length.",
                self.runtime.wind_speed
            ),
        );

        if self.body != WaterBody::River {
            drag_tip(
                ui,
                "Shore calm depth (m)",
                "Waves calm where the water is shallower than this over the ground under it (baked from the static geometry, rebaked when it changes): gone at the waterline. 0 = full waves everywhere.",
                &mut self.shore_depth,
                0.05,
                0.0..=50.0,
                &mut c,
            );
        }

        widgets::eyebrow(ui, "Physics");
        widgets::property_row(ui, "Fluid", |ui| {
            ui.horizontal_wrapped(|ui| {
                for preset in zestengine_core::FLUID_PRESETS {
                    let hint = format!(
                        "Template: {} kg/m3, {} Pa s, with its colour, clarity, ripples and foam. Shape, waves and simulation are kept; tune anything after.",
                        preset.density, preset.viscosity
                    );
                    if ui.selectable_label(preset.matches(self), preset.name).on_hover_text(hint).clicked() {
                        preset.apply(self);
                        c = true;
                    }
                }
            });
        });
        drag(ui, "Density (kg/m3)", &mut self.density, 1.0, 1.0..=20_000.0, &mut c);
        widgets::property_row(ui, "Viscosity (Pa s)", |ui| {
            c |= ui
                .add(egui::Slider::new(&mut self.viscosity, 0.0001..=100_000.0).logarithmic(true).max_decimals(4))
                .on_hover_text(
                    "How thick the fluid is: water 0.001, oil ~0.1, honey ~10. Thicker fluid drags bodies harder (they sink or rise slowly to where their density \
                     settles them), damps waves (only long slow ones survive) and flows sluggishly. What floats is set by Density alone.",
                )
                .changed();
        });
        widgets::property_row(ui, "Fluid scale (1:N)", |ui| {
            c |= ui
                .add(egui::Slider::new(&mut self.fluid_scale, 1.0..=200.0).logarithmic(true).max_decimals(1))
                .on_hover_text(
                    "Stage the fluid as a scale model: it moves like the same fluid N times smaller. Honey's thickness shows at jar size; \
                     in a pool-sized scene it still makes slow waves, as real honey would. 1:50 or so makes a pool of honey move like a jar of it. \
                     (Froude similarity: viscosity x N^1.5.) 1 = real size.",
                )
                .changed();
        });
        if self.fluid_scale > 1.0 {
            note(ui, &format!("Acts with {:.3} Pa s.", self.effective_viscosity()));
        }
        drag(ui, "Drag", &mut self.drag, 0.01, 0.0..=50.0, &mut c);
        drag(ui, "Angular drag", &mut self.angular_drag, 0.01, 0.0..=50.0, &mut c);
        widgets::property_row(ui, "Current (m/s)", |ui| c |= widgets::vec3_field(ui, &mut self.current, 2, 0.01));

        simulation_ui(ui, self, &mut c);

        widgets::eyebrow(ui, "Reflection");
        widgets::property_row(ui, "Type", |ui| {
            let opts: Vec<(zestengine_core::SurfaceReflection, &str)> = [
                (zestengine_core::SurfaceReflection::Environment, "Environment"),
                (zestengine_core::SurfaceReflection::Planar, "Planar"),
                (zestengine_core::SurfaceReflection::Probes, "Probes"),
                (zestengine_core::SurfaceReflection::Off, "Off"),
            ]
            .into();
            c |= widgets::segmented(ui, &mut self.reflection, &opts).changed();
        });
        note(
            ui,
            "Environment = the scene's global reflection setting. Planar = a mirrored render of the room, warped by the ripples. \
             Probes = baked probes only, no SSR / ray-traced resolve. Off = matte, no reflections.",
        );

        widgets::eyebrow(ui, "Look");
        let l = &mut self.look;
        if l.optics.is_none() {
            note(ui, "This scene's colour/depth values are converted to physical water optics.");
            if widgets::secondary_button(ui, None, "Edit physical optics").clicked() {
                l.optics = Some(zestengine_core::WaterOptics::from_legacy(l));
                c = true;
            }
        }
        if let Some(optics) = &mut l.optics {
            widgets::property_row(ui, "Absorption RGB (1/m)", |ui| c |= widgets::vec3_field(ui, &mut optics.absorption, 3, 0.005));
            widgets::property_row(ui, "Scattering RGB (1/m)", |ui| c |= widgets::vec3_field(ui, &mut optics.scattering, 3, 0.005));
            drag(ui, "Index of refraction", &mut optics.ior, 0.001, 1.0..=2.5, &mut c);
            drag(ui, "Scattering direction", &mut optics.anisotropy, 0.005, -0.95..=0.95, &mut c);
            *optics = optics.sanitized();
            note(ui, "Absorption removes light; scattering colours the water volume. Values are per metre and also drive the underwater view. Positive scattering direction favours forward scattering.");
        }
        color(ui, "Shallow colour", &mut l.shallow_color, &mut c);
        color(ui, "Deep colour", &mut l.deep_color, &mut c);
        if l.optics.is_none() {
            drag(ui, "Absorption depth", &mut l.absorption_depth, 0.05, 0.05..=200.0, &mut c);
        } else { note(ui, "Shallow/deep colours below are the palette for splash and mist effects; the volume uses the optical coefficients above."); }
        drag(ui, "Clarity", &mut l.clarity, 0.005, 0.0..=1.0, &mut c);
        drag(ui, "Roughness", &mut l.roughness, 0.002, 0.0..=1.0, &mut c);
        drag(ui, "Surface detail", &mut l.ripple_strength, 0.005, 0.0..=2.0, &mut c);
        drag(ui, "Detail scale", &mut l.ripple_scale, 0.01, 0.05..=20.0, &mut c);
        drag(ui, "Shore foam (m)", &mut l.foam_width, 0.005, 0.0..=5.0, &mut c);
        drag(ui, "Crest foam", &mut l.crest_foam, 0.005, 0.0..=1.0, &mut c);
        widgets::property_row(ui, "Foam", |ui| {
            c |= ui.checkbox(&mut l.foam_on, "").on_hover_text("Shore, crest, plunge and whitewater foam on this water").changed();
        });
        if l.foam_on {
            drag(ui, "Foam amount", &mut l.foam, 0.01, 0.0..=2.0, &mut c);
        }
        drag(ui, "Refraction", &mut l.refraction, 0.01, 0.0..=2.0, &mut c);

        widgets::eyebrow(ui, "Underwater");
        let u = &mut self.underwater;
        widgets::property_row(ui, "Enabled", |ui| c |= ui.checkbox(&mut u.enabled, "").changed());
        c |= _ctx.file_field(ui, "Sound", &mut u.sound, "audio/underwater.ogg", &["wav", "ogg", "mp3", "flac"]);
        drag(ui, "Volume", &mut u.volume, 0.01, 0.0..=4.0, &mut c);
        drag(ui, "Muffle", &mut u.muffle, 0.005, 0.0..=1.0, &mut c);
        if self.look.optics.is_none() { widgets::property_row(ui, "Visibility (m)", |ui| {
            c |= ui
                .add(egui::Slider::new(&mut u.visibility, 0.5..=200.0).logarithmic(true).max_decimals(1))
                .on_hover_text("How far you can see under water: the view is nearly all fog at this distance. Murky: 2-5 m, a lake: ~10 m, a clear pool or tropical sea: 20-40 m")
                .changed();
        });
        }
        drag(ui, "Distortion", &mut u.distortion, 0.01, 0.0..=5.0, &mut c);
        drag(ui, "Blur", &mut u.blur, 0.05, 0.0..=12.0, &mut c);
        c |= _ctx.file_field(ui, "Post FX", &mut u.postfx, "none", &["postfx"]);
        drag(ui, "Caustics", &mut u.caustics, 0.01, 0.0..=4.0, &mut c);
        widgets::property_row(ui, "Project caustics", |ui| c |= ui.checkbox(&mut self.project_caustics, "").changed());
        if self.project_caustics {
            drag(ui, "Projection strength", &mut self.projection_strength, 0.01, 0.0..=8.0, &mut c);
        }
        note(ui, "Caustics: the moving light pattern on whatever lies under the surface. Projected: lights under the surface throw it out onto the walls and ceiling (1 = physical brightness).");
        note(ui, "Below the surface the view fogs toward the water's colours (Look), blurs, takes the Post FX profile over the camera's and the mix is muffled under the looping sound.");
        c
    }
}

impl Gizmo for Water {
    /// The outline (lake rectangle or polygon, river banks and centre line)
    /// with the wave crests at time 0, point handles for polygons and rivers,
    /// and a resizable box for a rectangle lake.
    fn gizmos(&self) -> Vec<GizmoSpec> {
        if self.body == WaterBody::River {
            return river_gizmos(self);
        }
        let mut lines = Vec::new();
        let mut handles = Vec::new();
        match self.body {
            WaterBody::Lake | WaterBody::Pool if self.shape.len() >= 2 => {
                let curve: Vec<Vec3> = self.outline().iter().map(|p| Vec3::new(p[0], 0.0, p[1])).collect();
                for k in 0..curve.len() {
                    lines.push((curve[k], curve[(k + 1) % curve.len()]));
                }
                handles.extend(self.shape.iter().map(|p| Vec3::new(p[0], 0.0, p[1])));
            }
            _ => {}
        }
        if self.body != WaterBody::River {
            let half = Vec2::new(
                if self.body.basin() && self.size[0] > 0.0 { self.size[0] * 0.5 } else { 10.0 },
                if self.body.basin() && self.size[1] > 0.0 { self.size[1] * 0.5 } else { 10.0 },
            );
            let n = 24;
            for row in 0..=6 {
                let z = -half.y + 2.0 * half.y * row as f32 / 6.0;
                let mut prev: Option<Vec3> = None;
                for k in 0..=n {
                    let x = -half.x + 2.0 * half.x * k as f32 / n as f32;
                    if !self.covers(Vec2::new(x, z)) {
                        prev = None;
                        continue;
                    }
                    let (h, _) = self.surface(Vec2::new(x, z), 0.0);
                    let p = Vec3::new(x, h, z);
                    if let Some(q) = prev {
                        lines.push((q, p));
                    }
                    prev = Some(p);
                }
            }
        }
        let mut g = vec![GizmoSpec::Lines { segments: lines, color: WATER, world: false }];
        g.extend(handles.into_iter().map(|p| GizmoSpec::Handle { position: p, anchor: p, color: WATER }));
        if self.body.basin() && self.shape.len() < 3 && self.size[0] > 0.0 && self.size[1] > 0.0 {
            let half = Vec3::new(self.size[0] * 0.5, self.depth.max(0.0) * 0.5, self.size[1] * 0.5);
            g.push(GizmoSpec::Box { center: Vec3::new(0.0, -half.y, 0.0), half_extents: half, object_anchored: false, blend: 0.0, color: WATER });
        }
        g
    }

    fn apply_gizmo_edit(&mut self, edit: GizmoEdit) {
        match edit {
            GizmoEdit::Handle { index, position } if self.body == WaterBody::River => match river_handle(self, index) {
                Some(RiverHandle::Point(k)) => self.river[k].position = position.to_array(),
                Some(RiverHandle::Width(k)) => {
                    let side = river_sides(self)[k];
                    let across = (position - Vec3::from(self.river[k].position)).dot(side).abs();
                    self.river[k].width = (across * 2.0).clamp(0.1, 500.0);
                }
                Some(RiverHandle::Depth(k)) => {
                    self.river[k].depth = (self.river[k].position[1] - position.y).clamp(0.0, 100.0);
                }
                Some(RiverHandle::Flow(k)) => {
                    let direction = river_flow_direction(self, k);
                    self.river[k].speed = ((position - Vec3::from(self.river[k].position)).dot(direction) / FLOW_HANDLE_SCALE).clamp(-50.0, 50.0);
                }
                None => {}
            },
            GizmoEdit::Handle { index, position } => {
                if let Some(p) = self.shape.get_mut(index.saturating_sub(1)) {
                    *p = [position.x, position.z];
                }
            }
            GizmoEdit::Box { half_extents, .. } => {
                self.size = [half_extents.x * 2.0, half_extents.z * 2.0];
                self.depth = half_extents.y * 2.0;
            }
            _ => {}
        }
    }
}

/// The river's centre line (with downstream chevrons), its banks and the
/// cross-section at each point, then a handle per point and a width handle on
/// each point's bank. Indices follow [`river_handle`].
fn river_gizmos(water: &Water) -> Vec<GizmoSpec> {
    let (sections, _) = water.river_sections();
    let sides = river_sides(water);
    let mut centre = Vec::new();
    let mut banks = Vec::new();
    for pair in sections.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        centre.push((a.center, b.center));
        for s in [-0.5, 0.5] {
            banks.push((a.center + a.side * (a.width * s), b.center + b.side * (b.width * s)));
        }
    }
    for pair in water.river.windows(2) {
        let (a, b) = (Vec3::from(pair[0].position), Vec3::from(pair[1].position));
        let len = (b - a).length();
        if len < 1e-3 {
            continue;
        }
        let dir = (b - a) / len;
        let across = Vec3::new(-dir.z, 0.0, dir.x).normalize_or(Vec3::Z);
        let size = (len * 0.12).min((pair[0].width + pair[1].width) * 0.15);
        let tip = (a + b) * 0.5 + dir * size * 0.5;
        for s in [-1.0, 1.0] {
            centre.push((tip - dir * size + across * (size * s), tip));
        }
    }
    for (p, side) in water.river.iter().zip(&sides) {
        let c = Vec3::from(p.position);
        let half = *side * (p.width * 0.5);
        let down = Vec3::Y * p.depth;
        banks.push((c - half, c + half));
        banks.push((c - half - down, c + half - down));
        banks.push((c - half, c - half - down));
        banks.push((c + half, c + half - down));
    }
    let mut g = vec![
        GizmoSpec::Lines { segments: centre, color: WATER, world: false },
        GizmoSpec::Lines { segments: banks, color: BANK, world: false },
    ];
    g.extend(water.river.iter().map(|p| {
        let c = Vec3::from(p.position);
        GizmoSpec::Handle { position: c, anchor: c, color: WATER }
    }));
    g.extend(water.river.iter().zip(&sides).map(|(p, side)| {
        let c = Vec3::from(p.position);
        GizmoSpec::Handle { position: c + *side * (p.width * 0.5), anchor: c, color: WIDTH_HANDLE }
    }));
    g.extend(water.river.iter().map(|p| {
        let c = Vec3::from(p.position);
        GizmoSpec::Handle { position: c - Vec3::Y * p.depth, anchor: c, color: DEPTH_HANDLE }
    }));
    g.extend(water.river.iter().enumerate().map(|(k, p)| {
        let c = Vec3::from(p.position);
        GizmoSpec::Handle { position: c + river_flow_direction(water, k) * (p.speed * FLOW_HANDLE_SCALE), anchor: c, color: FLOW_HANDLE }
    }));
    g
}

fn river_flow_direction(water: &Water, k: usize) -> Vec3 {
    let before = Vec3::from(water.river[k.saturating_sub(1)].position);
    let after = Vec3::from(water.river[(k + 1).min(water.river.len() - 1)].position);
    (after - before).normalize_or(Vec3::X)
}

/// The River section: the Edit Path toggle and the point list, which shares
/// its selected point with the Scene view.
fn river_points_ui(ui: &mut Ui, river: &mut Vec<RiverPoint>) -> bool {
    let mut c = false;
    let ctx = ui.ctx().clone();
    let mut edit = RiverPathEdit::get(&ctx);
    let before = edit;
    if edit.active {
        if widgets::primary_button(ui, Some(Icon::Check), "Finish Editing Path").clicked() {
            edit.active = false;
        }
        note(ui, RIVER_PATH_HINT);
    } else {
        if widgets::secondary_button(ui, Some(Icon::Target), "Edit Path")
            .on_hover_text("Lay the river's points by clicking in the Scene view")
            .clicked()
        {
            edit.active = true;
        }
        note(ui, "Points run upstream to downstream; the surface follows their heights. Edit Path to lay them in the Scene view.");
    }
    edit.selected = edit.selected.filter(|&k| k < river.len());
    let mut remove = None;
    for (k, p) in river.iter_mut().enumerate() {
        let selected = edit.selected == Some(k);
        let frame = egui::Frame::new()
            .fill(if selected { theme::accent_faint() } else { egui::Color32::TRANSPARENT })
            .corner_radius(4)
            .inner_margin(egui::Margin::symmetric(4, 2))
            .show(ui, |ui| {
                let mut row = false;
                ui.horizontal(|ui| {
                    let num = RichText::new(format!("{}", k + 1)).color(if selected { theme::accent() } else { theme::TEXT_SECONDARY });
                    if ui.add(egui::Label::new(num).sense(egui::Sense::click())).on_hover_text("Select this point").clicked() {
                        edit.selected = Some(k);
                    }
                    row |= widgets::vec3_field(ui, &mut p.position, 2, 0.05);
                    if widgets::icon_button(ui, Icon::Trash, "Remove", false).clicked() {
                        remove = Some(k);
                    }
                });
                ui.horizontal(|ui| {
                    row |= ui.add(DragValue::new(&mut p.width).speed(0.05).range(0.1..=500.0).prefix("width ")).changed();
                    row |= ui.add(DragValue::new(&mut p.depth).speed(0.05).range(0.0..=100.0).prefix("depth ")).changed();
                    row |= ui.add(DragValue::new(&mut p.speed).speed(0.05).range(-50.0..=50.0).prefix("flow ")).changed();
                });
                row
            });
        if frame.inner {
            c = true;
            edit.selected = Some(k);
        }
        if selected && edit.reveal {
            frame.response.scroll_to_me(Some(egui::Align::Center));
        }
    }
    edit.reveal = false;
    if let Some(k) = remove {
        river.remove(k);
        edit.selected = match edit.selected {
            Some(s) if s == k => None,
            Some(s) if s > k => Some(s - 1),
            s => s,
        };
        c = true;
    }
    if widgets::dashed_button(ui, Some(Icon::Plus), "Add point").clicked() {
        let next = match river.as_slice() {
            [.., a, b] => {
                let step = (Vec3::from(b.position) - Vec3::from(a.position)).normalize_or(Vec3::X) * 8.0;
                RiverPoint { position: (Vec3::from(b.position) + step).to_array(), ..*b }
            }
            [last] => RiverPoint { position: [last.position[0] + 8.0, last.position[1], last.position[2]], ..*last },
            [] => RiverPoint::default(),
        };
        river.push(next);
        edit.selected = Some(river.len() - 1);
        c = true;
    }
    if edit != before {
        edit.set(&ctx);
    }
    c
}

/// Editable list of 2D (X, Z) points.
fn points2(ui: &mut Ui, pts: &mut Vec<[f32; 2]>, changed: &mut bool) {
    let mut remove = None;
    for (k, p) in pts.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("{}", k + 1)).color(theme::TEXT_SECONDARY));
            *changed |= ui.add(DragValue::new(&mut p[0]).speed(0.05).prefix("x ")).changed();
            *changed |= ui.add(DragValue::new(&mut p[1]).speed(0.05).prefix("z ")).changed();
            if widgets::icon_button(ui, Icon::Trash, "Remove", false).clicked() {
                remove = Some(k);
            }
        });
    }
    if let Some(k) = remove {
        pts.remove(k);
        *changed = true;
    }
    if widgets::dashed_button(ui, Some(Icon::Plus), "Add point").clicked() {
        if pts.len() >= 3 {
            // Split the longest edge of the closed outline.
            let n = pts.len();
            let k = (0..n)
                .max_by(|&a, &b| {
                    let len = |i: usize| Vec2::from(pts[i]).distance(Vec2::from(pts[(i + 1) % n]));
                    len(a).total_cmp(&len(b))
                })
                .unwrap_or(0);
            let mid = (Vec2::from(pts[k]) + Vec2::from(pts[(k + 1) % n])) * 0.5;
            pts.insert(k + 1, mid.to_array());
        } else {
            let next = match pts.as_slice() {
                [] => [-5.0, -5.0],
                [.., a, b] => [b[0] + (b[0] - a[0]), b[1] + (b[1] - a[1])],
                [a] => [a[0] + 5.0, a[1]],
            };
            pts.push(next);
        }
        *changed = true;
    }
}

impl Inspect for Buoyancy {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut c = false;
        widgets::info_note(ui, "Optional. Every Rigid Body already floats by its density (mass over collider volume, or the physics material's density) and the material's buoyancy and water drag. Add this only to override the lift or place sample points by hand.");
        drag(ui, "Buoyancy", &mut self.buoyancy, 0.01, 0.0..=20.0, &mut c);
        note(ui, "Lift when fully under, as a multiple of the body's weight: 2 floats half under, 1 hovers, below 1 sinks.");
        drag(ui, "Point depth (m)", &mut self.point_depth, 0.005, 0.0..=20.0, &mut c);
        drag(ui, "Drag scale", &mut self.drag_scale, 0.01, 0.0..=20.0, &mut c);
        widgets::property_row(ui, "Follow current", |ui| c |= ui.checkbox(&mut self.follow_current, "").changed());
        widgets::eyebrow(ui, "Sample points");
        if self.points.is_empty() {
            note(ui, "None set: the corners and centre of each collider are used.");
        }
        let mut remove = None;
        for (k, p) in self.points.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                c |= widgets::vec3_field(ui, p, 2, 0.01);
                if widgets::icon_button(ui, Icon::Trash, "Remove", false).clicked() {
                    remove = Some(k);
                }
            });
        }
        if let Some(k) = remove {
            self.points.remove(k);
            c = true;
        }
        if widgets::dashed_button(ui, Some(Icon::Plus), "Add point").clicked() {
            self.points.push([0.0; 3]);
            c = true;
        }
        widgets::property_row(ui, "Submerged", |ui| ui.add(egui::ProgressBar::new(self.submerged).desired_width(120.0)));
        c
    }
}

impl Gizmo for Buoyancy {
    /// Authored sample points as draggable handles.
    fn gizmos(&self) -> Vec<GizmoSpec> {
        self.points.iter().map(|p| GizmoSpec::Handle { position: Vec3::from(*p), anchor: Vec3::ZERO, color: WATER }).collect()
    }

    fn apply_gizmo_edit(&mut self, edit: GizmoEdit) {
        if let GizmoEdit::Handle { index, position } = edit
            && let Some(p) = self.points.get_mut(index)
        {
            *p = position.to_array();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn river() -> Water {
        let p = |x: f32, width: f32| RiverPoint { position: [x, 0.0, 0.0], width, ..Default::default() };
        Water { body: WaterBody::River, river: vec![p(0.0, 4.0), p(10.0, 4.0), p(20.0, 6.0)], ..Default::default() }
    }

    /// The basin box's top is the rest surface (the object's own height,
    /// local Y 0), where the surface outline is drawn and where the renderer
    /// puts calm water (`gpu_tests::water` holds the simulated surface there):
    /// the volume hangs `depth` below it.
    #[test]
    fn the_basin_box_top_is_the_rest_surface() {
        let w = Water { body: WaterBody::Pool, size: [20.0, 20.0], depth: 5.0, waves: Vec::new(), ..Default::default() };
        let g = w.gizmos();
        let (center, half) = g
            .iter()
            .find_map(|s| match s {
                GizmoSpec::Box { center, half_extents, .. } => Some((*center, *half_extents)),
                _ => None,
            })
            .expect("a basin box");
        assert!((center.y + half.y).abs() < 1e-6, "top at the surface");
        assert!((center.y - half.y + 5.0).abs() < 1e-6, "bottom `depth` below");
        let lines = g.iter().find_map(|s| if let GizmoSpec::Lines { segments, .. } = s { Some(segments) } else { None }).unwrap();
        assert!(lines.iter().all(|(a, b)| a.y.abs() < 1e-6 && b.y.abs() < 1e-6), "calm water's outline lies on it");
        assert_eq!(w.surface(Vec2::new(3.0, -2.0), 1.0).0, 0.0, "and so does the physics surface");
    }

    #[test]
    fn river_handles_follow_the_gizmo_list() {
        let w = river();
        let g = w.gizmos();
        assert_eq!(g.len(), RIVER_LINE_SPECS + 12);
        for (i, spec) in g.iter().enumerate() {
            match (river_handle(&w, i), spec) {
                (None, GizmoSpec::Lines { .. }) => assert!(i < RIVER_LINE_SPECS),
                (Some(RiverHandle::Point(k)), GizmoSpec::Handle { position, .. }) => assert_eq!(position.to_array(), w.river[k].position),
                (Some(RiverHandle::Width(k)), GizmoSpec::Handle { position, anchor, .. }) => {
                    assert_eq!(anchor.to_array(), w.river[k].position);
                    assert!((position.distance(*anchor) - w.river[k].width * 0.5).abs() < 1e-5, "on the bank");
                }
                (Some(RiverHandle::Depth(k)), GizmoSpec::Handle { position, anchor, .. }) => {
                    assert_eq!(*position, *anchor - Vec3::Y * w.river[k].depth);
                }
                (Some(RiverHandle::Flow(k)), GizmoSpec::Handle { position, anchor, .. }) => {
                    assert!((position.distance(*anchor) - w.river[k].speed.abs() * FLOW_HANDLE_SCALE).abs() < 1e-5);
                }
                _ => panic!("gizmo {i} does not match its handle"),
            }
        }
        assert_eq!(river_handle(&w, RIVER_LINE_SPECS + 12), None);
    }

    #[test]
    fn depth_and_flow_handles_edit_cross_sections_without_moving_the_path() {
        let mut w = river();
        let before = w.river[1].position;
        let n = w.river.len();
        w.apply_gizmo_edit(GizmoEdit::Handle { index: RIVER_LINE_SPECS + 2 * n + 1, position: Vec3::from(before) - Vec3::Y * 3.0 });
        let direction = river_flow_direction(&w, 1);
        w.apply_gizmo_edit(GizmoEdit::Handle { index: RIVER_LINE_SPECS + 3 * n + 1, position: Vec3::from(before) - direction * 2.0 });
        assert_eq!(w.river[1].position, before);
        assert_eq!(w.river[1].depth, 3.0);
        assert!((w.river[1].speed + 2.0).abs() < 1e-5);
        let column = w.column(Vec2::new(before[0], before[2])).unwrap();
        assert!((column.depth - 3.0).abs() < 0.05);
        assert!(column.flow.dot(direction) < -1.9);
    }

    #[test]
    fn handle_drags_move_points_and_set_widths() {
        let mut w = river();
        w.apply_gizmo_edit(GizmoEdit::Handle { index: RIVER_LINE_SPECS + 1, position: Vec3::new(10.0, 2.0, 3.0) });
        assert_eq!(w.river[1].position, [10.0, 2.0, 3.0]);
        let mut w = river();
        let side = river_sides(&w)[2];
        w.apply_gizmo_edit(GizmoEdit::Handle { index: RIVER_LINE_SPECS + 3 + 2, position: Vec3::new(21.0, 5.0, 0.0) + side * 4.0 });
        assert!((w.river[2].width - 8.0).abs() < 1e-4, "twice the distance across: {}", w.river[2].width);
        assert_eq!(w.river[2].position, [20.0, 0.0, 0.0], "the point stays put");
    }
}
