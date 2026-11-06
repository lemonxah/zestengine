//! Shared building blocks for the project templates: a scene builder that
//! keeps stable object ids (so components can reference each other), the
//! material palette, input binding helpers, and common props / UI.

use zestengine_assets::{
    ComponentData, MaterialFile, MaterialRef, ObjectSource, PrimitiveShape, SceneEntry, SceneFile,
    WorldEnvironment,
};
use zestengine_core::{
    ActionBinding, ActionKind, BoxCollider, CameraComponent, Component, ControlScheme, InputSource,
    Key, LightComponent, LightKind, MouseAxis, MouseButton, ObjectId, ObjectRef, PadAxis,
    PadButton, ParticleEmitter, Terrain, TerrainCollider, UiAction, UiButton, UiPauseMenu, UiText,
};
use zestengine_render::{MaterialFeatures, MaterialParams};
use glam::{Quat, Vec3};

use super::TemplateContent;
use crate::ui_templates::{TemplateNode, UiTemplate};

pub const SKY: &str = "textures/sky_103_2k.png";
pub const MAIN_SCENE: &str = "scenes/main.scene";

/// Scene under construction. Indices returned by `add*` address objects;
/// `r(i)` gives a reference other components can store.
#[derive(Default)]
pub struct Scene {
    pub(super) entries: Vec<SceneEntry>,
    ids: Vec<ObjectId>,
    components: Vec<Vec<Box<dyn Component>>>,
    pub skybox: Option<String>,
    pub environment: WorldEnvironment,
    pub layers: zestengine_core::LayerSettings,
    /// The scene's `Directional Light` (travel direction, intensity); `None`
    /// takes the default key light. See [`sun`].
    sun: Option<([f32; 3], f32)>,
    /// A scene that intentionally has no directional light ([`no_sun`]).
    no_sun: bool,
}

/// Layer for characters and units: navmeshes bake without it, so a
/// character standing somewhere at bake time doesn't block that spot.
pub const CHARACTER_LAYER: u8 = 8;

impl Scene {
    pub fn new() -> Self {
        Self {
            skybox: Some(SKY.into()),
            ..Default::default()
        }
    }

    pub fn add(
        &mut self,
        name: &str,
        source: ObjectSource,
        parent: Option<usize>,
        t: [f32; 3],
    ) -> usize {
        let id = ObjectId::new();
        self.entries.push(SceneEntry {
            id: id.to_string(),
            name: name.into(),
            source,
            enabled: true,
            static_geometry: false,
            lightmap_scale: 1.0,
            cast_shadows: zestengine_core::ShadowCasting::On,
            layer: 0,
            material: MaterialRef::File(mat_path("default")),
            extra_materials: Vec::new(),
            renderer_split: false,
            parent,
            components: Vec::new(),
            translation: t,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
            model: None,
            model_part: None,
            prefab: None,
            prefab_parent: None,
            net_role: Default::default(),
            component_roles: Default::default(),
        });
        self.ids.push(id);
        self.components.push(Vec::new());
        self.entries.len() - 1
    }

    pub fn empty(&mut self, name: &str, parent: Option<usize>, t: [f32; 3]) -> usize {
        self.add(name, ObjectSource::Empty, parent, t)
    }

    /// A primitive with a palette material.
    pub fn prim(
        &mut self,
        name: &str,
        shape: PrimitiveShape,
        parent: Option<usize>,
        t: [f32; 3],
        s: [f32; 3],
        material: &str,
    ) -> usize {
        let i = self.add(name, ObjectSource::Primitive { shape }, parent, t);
        self.entries[i].scale = s;
        self.entries[i].material = MaterialRef::File(mat_path(material));
        i
    }

    /// A model file object (first mesh), keeping the model's own textures.
    pub fn model(
        &mut self,
        name: &str,
        path: &str,
        parent: Option<usize>,
        t: [f32; 3],
        scale: f32,
    ) -> usize {
        let i = self.add(
            name,
            ObjectSource::Model {
                path: path.into(),
                mesh: 0,
                extra_meshes: Vec::new(),
                model: None,
                mesh_id: None,
                extra_mesh_ids: Vec::new(),
            },
            parent,
            t,
        );
        self.entries[i].scale = [scale; 3];
        self.entries[i].material = MaterialRef::Inline {
            params: MaterialParams::default(),
            features: MaterialFeatures::default(),
            shader: "standard".into(),
            custom: Default::default(),
            render_queue: None,
            textures: Default::default(),
        };
        i
    }

    /// A solid box (cube primitive + matching collider), static level geometry.
    pub fn block(
        &mut self,
        name: &str,
        parent: Option<usize>,
        t: [f32; 3],
        s: [f32; 3],
        material: &str,
    ) -> usize {
        let i = self.prim(name, PrimitiveShape::Cube, parent, t, s, material);
        self.with(i, BoxCollider::default());
        self.entries[i].static_geometry = true;
        i
    }

    /// Ground plane `size` meters square with a thin box collider.
    pub fn ground(&mut self, size: f32, material: &str) -> usize {
        let half = size * 0.5;
        let i = self.prim(
            "Ground",
            PrimitiveShape::Plane,
            None,
            [0.0; 3],
            [half, 1.0, half],
            material,
        );
        self.with(
            i,
            BoxCollider {
                center: [0.0, -0.05, 0.0],
                size: [2.0, 0.1, 2.0],
                ..Default::default()
            },
        );
        self.entries[i].static_geometry = true;
        i
    }

    pub fn terrain(&mut self, size: f32, resolution: u32, height: f32, material: &str) -> usize {
        let n = (resolution + 1) as usize;
        let mut heights = Vec::with_capacity(n * n);
        for z in 0..n {
            for x in 0..n {
                let fx = x as f32 / resolution as f32;
                let fz = z as f32 / resolution as f32;
                let wave = (fx * std::f32::consts::TAU * 2.0).sin()
                    * (fz * std::f32::consts::TAU).cos()
                    * 0.5
                    + (fx * std::f32::consts::TAU * 4.0).cos() * 0.25;
                heights.push(wave * height);
            }
        }
        let i = self.empty("Terrain", None, [0.0; 3]);
        self.with(
            i,
            Terrain {
                size,
                resolution,
                heights,
                material: mat_path(material),
                texture_weights: Vec::new(),
                ..Default::default()
            },
        );
        self.with(i, TerrainCollider::default());
        self.entries[i].static_geometry = true;
        i
    }

    pub fn camera(&mut self, name: &str, parent: Option<usize>, t: [f32; 3], fov: f32) -> usize {
        let i = self.add(name, ObjectSource::Camera, parent, t);
        self.with(
            i,
            CameraComponent {
                id: 1,
                fov_deg: fov,
                near: 0.05,
                far: 300.0,
                ..Default::default()
            },
        );
        i
    }

    /// First-person cameras sit inside the avatar's own head, which would
    /// otherwise render the inside of that mesh. Everything on
    /// [`CHARACTER_LAYER`] (the avatar body) is excluded from this camera's
    /// view only; the held weapon isn't on that layer, so it stays visible.
    pub fn hide_own_body_from_camera(&mut self, camera: usize) {
        let component = self.components[camera]
            .iter_mut()
            .find_map(|c| c.as_any_mut().downcast_mut::<CameraComponent>());
        let component =
            component.expect("hide_own_body_from_camera: object has no CameraComponent");
        component.culling_mask &= !(1u32 << CHARACTER_LAYER as u32);
    }

    pub fn point_light(
        &mut self,
        name: &str,
        parent: Option<usize>,
        t: [f32; 3],
        color: [f32; 3],
        intensity: f32,
        range: f32,
    ) -> usize {
        let i = self.add(name, ObjectSource::Light, parent, t);
        self.with(
            i,
            LightComponent {
                kind: LightKind::Point,
                color,
                intensity,
                range,
                ..Default::default()
            },
        );
        i
    }

    /// Whether some object already carries a Directional `Light`.
    fn has_directional_light(&self) -> bool {
        self.components.iter().flatten().any(|c| c.as_any().downcast_ref::<LightComponent>().is_some_and(|l| l.kind == LightKind::Directional))
    }

    pub fn with(&mut self, i: usize, c: impl Component) -> &mut Self {
        self.components[i].push(Box::new(c));
        self
    }

    pub fn r(&self, i: usize) -> ObjectRef {
        ObjectRef::new(self.ids[i])
    }

    pub fn rot(&mut self, i: usize, q: Quat) -> &mut Self {
        self.entries[i].rotation = q.normalize().to_array();
        self
    }

    pub fn yaw(&mut self, i: usize, degrees: f32) -> &mut Self {
        self.rot(i, Quat::from_rotation_y(degrees.to_radians()))
    }

    pub fn character_layer(&mut self, i: usize) -> &mut Self {
        self.entries[i].layer = CHARACTER_LAYER;
        self.layers.names[CHARACTER_LAYER as usize] = "Characters".into();
        self
    }

    pub fn hidden(&mut self, i: usize) -> &mut Self {
        self.entries[i].enabled = false;
        self
    }

    pub fn dynamic(&mut self, i: usize) -> &mut Self {
        self.entries[i].static_geometry = false;
        self
    }

    /// Children of `parent` named `name`.
    pub fn child(&self, parent: usize, name: &str) -> Option<usize> {
        self.entries
            .iter()
            .position(|e| e.parent == Some(parent) && e.name == name)
    }

    /// First component of type `T` on object `i`.
    pub fn component_mut<T: Component>(&mut self, i: usize) -> Option<&mut T> {
        self.components[i]
            .iter_mut()
            .find_map(|c| c.as_any_mut().downcast_mut::<T>())
    }

    /// Append a UI layout (node 0 is the canvas). Returns the object index of
    /// each node.
    pub fn ui(&mut self, nodes: Vec<TemplateNode>) -> Vec<usize> {
        let mut map = Vec::with_capacity(nodes.len());
        for node in nodes {
            let parent = node.parent.map(|p| map[p]);
            let i = self.empty(&node.name, parent, [0.0; 3]);
            self.entries[i].enabled = node.active;
            self.components[i] = node.components;
            map.push(i);
        }
        map
    }

    pub fn finish(mut self) -> SceneFile {
        // Every scene carries a real `Directional Light` object (the old
        // built-in environment sun) unless it opted out with `no_sun`.
        if !self.no_sun && !self.has_directional_light() {
            let (direction, intensity) = self.sun.unwrap_or(([-0.4, -1.0, -0.3], 3.0));
            let i = self.add(zestengine_assets::DIRECTIONAL_LIGHT_NAME, ObjectSource::Light, None, [0.0, 10.0, 0.0]);
            let entry = zestengine_assets::directional_light_entry([1.0, 0.98, 0.92], intensity, direction, zestengine_core::LightMode::Mixed);
            self.entries[i].rotation = entry.rotation;
            self.components[i] = vec![Box::new(
                ron::from_str::<LightComponent>(&entry.components[0].data).expect("light component round-trips"),
            )];
        }
        for (entry, comps) in self.entries.iter_mut().zip(self.components) {
            entry.components = comps
                .iter()
                .map(|c| ComponentData::of(c.as_ref()))
                .collect();
        }
        SceneFile {
            entries: self.entries,
            skybox: self.skybox,
            environment: self.environment,
            layers: self.layers,
            editor_camera: None,
            collapsed: Vec::new(),
            scene_hidden: Vec::new(),
            scene_locked: Vec::new(),
            format_version: 0,
        }
    }

    /// The objects rooted at `root` as prefab entries (parents re-indexed).
    /// Ids are kept so references between prefab objects survive; every
    /// instance remaps them to fresh ones.
    pub fn prefab(mut self, root: usize) -> crate::prefab::Prefab {
        let mut keep = vec![root];
        for i in root + 1..self.entries.len() {
            if self.entries[i].parent.is_some_and(|p| keep.contains(&p)) {
                keep.push(i);
            }
        }
        self.skybox = None;
        let file = self.finish();
        let entries = keep
            .iter()
            .map(|&i| {
                let mut e = file.entries[i].clone();
                e.parent = e.parent.and_then(|p| keep.iter().position(|&k| k == p));
                if i == root {
                    e.parent = None;
                }
                e
            })
            .collect();
        crate::prefab::Prefab::new(entries)
    }
}

pub fn mat_path(name: &str) -> String {
    format!("materials/{name}.material")
}

fn material(
    name: &str,
    color: [f32; 3],
    roughness: f32,
    metallic: f32,
    emission: Option<([f32; 3], f32)>,
) -> MaterialFile {
    let mut params = MaterialParams {
        base_color: [color[0], color[1], color[2], 1.0],
        roughness,
        metallic,
        ..Default::default()
    };
    let mut features = MaterialFeatures::default();
    if let Some((c, i)) = emission {
        params.emission_color = c;
        params.emission_intensity = i;
        features.emission = true;
    }
    MaterialFile {
        name: name.into(),
        shader: "standard".into(),
        params,
        features,
        render_queue: None,
        textures: Default::default(),
        custom: Default::default(),
    }
}

/// The colour palette every template project gets.
pub fn palette() -> Vec<(String, MaterialFile)> {
    [
        material("default", [0.8, 0.8, 0.8], 0.6, 0.0, None),
        material("ground", [0.42, 0.45, 0.43], 0.9, 0.0, None),
        material("grass", [0.30, 0.42, 0.24], 0.95, 0.0, None),
        material("wall", [0.72, 0.70, 0.66], 0.8, 0.0, None),
        material("trim", [0.20, 0.22, 0.26], 0.6, 0.0, None),
        material("player", [0.20, 0.45, 0.85], 0.5, 0.0, None),
        material("ally", [0.25, 0.65, 0.95], 0.5, 0.0, None),
        material("enemy", [0.85, 0.25, 0.22], 0.5, 0.0, None),
        material("target", [0.92, 0.88, 0.80], 0.7, 0.0, None),
        material("bullseye", [0.85, 0.15, 0.12], 0.5, 0.0, None),
        material("metal", [0.55, 0.57, 0.60], 0.3, 1.0, None),
        material("wood", [0.50, 0.33, 0.20], 0.8, 0.0, None),
        material("mirror", [0.95, 0.95, 0.95], 0.02, 1.0, None),
        material(
            "marker",
            [0.2, 0.9, 0.6],
            0.5,
            0.0,
            Some(([0.2, 1.0, 0.6], 2.0)),
        ),
        material(
            "select_ring",
            [0.3, 1.0, 0.4],
            0.5,
            0.0,
            Some(([0.3, 1.0, 0.4], 2.5)),
        ),
        material(
            "glow",
            [1.0, 0.8, 0.3],
            0.5,
            0.0,
            Some(([1.0, 0.75, 0.3], 3.0)),
        ),
        material("skin", [0.85, 0.68, 0.55], 0.7, 0.0, None),
    ]
    .into_iter()
    .map(|m| (mat_path(&m.name), m))
    .collect()
}

// ------------------------------------------------------------------ input

pub fn keys(k: &[Key]) -> Vec<InputSource> {
    k.iter().map(|k| InputSource::Key(*k)).collect()
}

fn blank(name: &str, kind: ActionKind) -> ActionBinding {
    ActionBinding {
        deadzone: 0.0,
        ..ActionBinding::empty(name, kind)
    }
}

pub fn button(name: &str, inputs: Vec<InputSource>) -> ActionBinding {
    ActionBinding::button(name, inputs)
}

/// WASD + arrows 2D axis.
pub fn wasd(name: &str) -> ActionBinding {
    let mut a = blank(name, ActionKind::Axis2);
    a.pos_x = keys(&[Key::D, Key::Right]);
    a.neg_x = keys(&[Key::A, Key::Left]);
    a.pos_y = keys(&[Key::W, Key::Up]);
    a.neg_y = keys(&[Key::S, Key::Down]);
    a
}

pub fn mouse_look(name: &str) -> ActionBinding {
    let mut a = blank(name, ActionKind::Axis2);
    a.analog_x = Some(InputSource::MouseAxis(MouseAxis::X));
    a.analog_y = Some(InputSource::MouseAxis(MouseAxis::Y));
    a
}

pub fn wheel(name: &str) -> ActionBinding {
    let mut a = blank(name, ActionKind::Axis1);
    a.analog_x = Some(InputSource::MouseAxis(MouseAxis::Wheel));
    a
}

pub fn axis1(name: &str, pos: Vec<InputSource>, neg: Vec<InputSource>) -> ActionBinding {
    let mut a = blank(name, ActionKind::Axis1);
    a.pos_x = pos;
    a.neg_x = neg;
    a
}

pub fn stick(name: &str, x: PadAxis, y: PadAxis, scale: f32) -> ActionBinding {
    let mut a = blank(name, ActionKind::Axis2);
    a.analog_x = Some(InputSource::PadAxis(x));
    a.analog_y = Some(InputSource::PadAxis(y));
    a.deadzone = 0.15;
    a.scale = scale;
    a
}

pub fn mouse(b: MouseButton) -> InputSource {
    InputSource::Mouse(b)
}

pub fn pad(b: PadButton) -> InputSource {
    InputSource::PadButton(b)
}

pub fn pad_axis(a: PadAxis) -> InputSource {
    InputSource::PadAxis(a)
}

pub fn bindings(kb: Vec<ActionBinding>, gamepad: Vec<ActionBinding>) -> zestengine_core::Bindings {
    zestengine_core::Bindings::from_schemes(vec![
        ControlScheme {
            name: "KB+Mouse".into(),
            actions: kb,
        },
        ControlScheme {
            name: "Gamepad".into(),
            actions: gamepad,
        },
    ])
}

/// Shooter bindings (FPS / TPS): WASD, mouse look, jump, sprint, crouch,
/// fire, aim, reload, interact.
pub fn shooter_bindings() -> zestengine_core::Bindings {
    bindings(
        vec![
            wasd("Move"),
            mouse_look("Look"),
            button("Jump", keys(&[Key::Space])),
            button("Sprint", keys(&[Key::LShift])),
            button("Crouch", keys(&[Key::LCtrl, Key::C])),
            button("Fire", vec![mouse(MouseButton::Left)]),
            button("Aim", vec![mouse(MouseButton::Right)]),
            button("Reload", keys(&[Key::R])),
            button("Interact", keys(&[Key::E])),
        ],
        vec![
            stick("Move", PadAxis::LeftStickX, PadAxis::LeftStickY, 1.0),
            stick("Look", PadAxis::RightStickX, PadAxis::RightStickY, 8.0),
            button("Jump", vec![pad(PadButton::South)]),
            button("Sprint", vec![pad(PadButton::LeftThumb)]),
            button("Crouch", vec![pad(PadButton::East)]),
            button("Fire", vec![pad_axis(PadAxis::RightTrigger)]),
            button("Aim", vec![pad_axis(PadAxis::LeftTrigger)]),
            button("Reload", vec![pad(PadButton::West)]),
            button("Interact", vec![pad(PadButton::North)]),
        ],
    )
}

// ------------------------------------------------------------------ props

/// Particle emitter that only fires bursts on request (muzzle flash, impacts).
pub fn burst_emitter(
    color: [f32; 4],
    speed: [f32; 2],
    size: [f32; 2],
    lifetime: [f32; 2],
    cone: f32,
) -> ParticleEmitter {
    ParticleEmitter {
        rate: 0.0,
        burst: 0,
        lifetime,
        speed,
        direction: [0.0, 0.0, -1.0],
        cone_angle: cone,
        shape_radius: 0.02,
        gravity: 0.5,
        drag: 2.0,
        size,
        color_start: color,
        color_end: [color[0], color[1] * 0.5, color[2] * 0.3, 0.0],
        max_particles: 200,
        looping: true,
        play_on_start: true,
        world_space: true,
        ..Default::default()
    }
}

/// A root-level impact effect object (hitscan weapons move it to each hit).
pub fn impact_effect(s: &mut Scene) -> usize {
    let i = s.empty("Impact Effect", None, [0.0, -100.0, 0.0]);
    s.with(
        i,
        burst_emitter(
            [1.0, 0.85, 0.5, 1.0],
            [1.5, 4.0],
            [0.06, 0.01],
            [0.15, 0.4],
            60.0,
        ),
    );
    i
}

/// A shooting-range target: a post with a round plate on a pivot at its base.
pub fn target_dummy(s: &mut Scene, name: &str, t: [f32; 3], yaw: f32) -> usize {
    let root = s.empty(name, None, t);
    s.yaw(root, yaw);
    s.with(
        root,
        zestengine_core::Health {
            max: 50.0,
            current: 50.0,
            ..Default::default()
        },
    );
    s.with(root, zestengine_core::TargetDummy::default());
    s.with(
        root,
        BoxCollider {
            center: [0.0, 1.0, 0.0],
            size: [0.9, 2.0, 0.25],
            ..Default::default()
        },
    );
    s.with(
        root,
        zestengine_core::RigidBody {
            kind: zestengine_core::BodyKind::Kinematic,
            ..Default::default()
        },
    );
    s.prim(
        "Post",
        PrimitiveShape::Cube,
        Some(root),
        [0.0, 0.6, 0.0],
        [0.12, 1.2, 0.12],
        "wood",
    );
    s.prim(
        "Plate",
        PrimitiveShape::Cube,
        Some(root),
        [0.0, 1.45, 0.0],
        [0.85, 0.85, 0.06],
        "target",
    );
    s.prim(
        "Bullseye",
        PrimitiveShape::Cube,
        Some(root),
        [0.0, 1.45, 0.035],
        [0.3, 0.3, 0.02],
        "bullseye",
    );
    root
}

/// HUD canvas with crosshair, health bar, ammo and score bound to `player`.
pub fn shooter_hud(s: &mut Scene, player: ObjectRef, weapon: ObjectRef, objective: &str) {
    use zestengine_core::{HudBinding, HudValue};
    let nodes = s.ui(UiTemplate::Hud.build());
    let find = |s: &Scene, name: &str| nodes.iter().copied().find(|&i| s.entries[i].name == name);
    if let Some(i) = find(s, "Health Bar") {
        s.with(
            i,
            HudBinding {
                source: player,
                value: HudValue::Health,
                ..Default::default()
            },
        );
        if let Some(label) = s.child(i, "Label") {
            s.with(
                label,
                HudBinding {
                    source: player,
                    value: HudValue::Health,
                    format: "HP {value}".into(),
                    ..Default::default()
                },
            );
        }
    }
    if let Some(i) = find(s, "Stamina Bar") {
        s.hidden(i);
    }
    if let Some(i) = find(s, "Ammo") {
        s.with(
            i,
            HudBinding {
                source: weapon,
                value: HudValue::Ammo,
                format: "{value} / {max}".into(),
                ..Default::default()
            },
        );
    }
    if let Some(i) = find(s, "Score") {
        s.with(
            i,
            HudBinding {
                value: HudValue::Score,
                format: "Score: {value}".into(),
                ..Default::default()
            },
        );
    }
    if let Some(i) = find(s, "Objective")
        && let Some(t) = s.component_mut::<UiText>(i)
    {
        t.text = objective.into();
    }
}

/// Escape pause menu (Resume / Settings / Restart / Quit).
pub fn pause_menu(s: &mut Scene, scene_path: &str, relock_cursor: bool) {
    let nodes = s.ui(UiTemplate::PauseMenu.build());
    let root = nodes[0];
    if let Some(m) = s.component_mut::<UiPauseMenu>(root) {
        m.lock_cursor_in_game = relock_cursor;
    }
    if let Some(i) = nodes
        .iter()
        .copied()
        .find(|&i| s.entries[i].name == "Main Menu Button")
    {
        s.entries[i].name = "Restart Button".into();
        if let Some(b) = s.component_mut::<UiButton>(i) {
            b.on_click = vec![UiAction::LoadScene(scene_path.into())];
        }
        if let Some(label) = s.child(i, "Label")
            && let Some(t) = s.component_mut::<UiText>(label)
        {
            t.text = "Restart".into();
        }
    }
}

/// A key light so the scene reads well before any baking: sets the direction
/// and intensity of the scene's `Directional Light` object (added on `finish`).
pub fn sun(s: &mut Scene, direction: [f32; 3], intensity: f32) {
    s.sun = Some((Vec3::from(direction).normalize().to_array(), intensity));
    s.no_sun = false;
}

/// This scene intentionally has no directional light (lit by other means).
pub fn no_sun(s: &mut Scene) {
    s.no_sun = true;
}

// ------------------------------------------------------------------ empty

fn starter_scene() -> SceneFile {
    let mut s = Scene::new();
    s.skybox = None;
    s.camera("Main Camera", None, [0.0, 1.5, 5.0], 60.0);
    s.prim(
        "Ground",
        PrimitiveShape::Plane,
        None,
        [0.0; 3],
        [10.0, 1.0, 10.0],
        "default",
    );
    s.prim(
        "Cube",
        PrimitiveShape::Cube,
        None,
        [0.0, 0.5, 0.0],
        [1.0; 3],
        "default",
    );
    s.finish()
}

pub fn empty() -> TemplateContent {
    TemplateContent {
        scenes: vec![(MAIN_SCENE.into(), starter_scene())],
        materials: vec![(
            mat_path("default"),
            material("default", [0.8, 0.8, 0.8], 0.6, 0.0, None),
        )],
        prefabs: Vec::new(),
        bindings: zestengine_core::Bindings::default(),
        assets: Vec::new(),
        boot_scene: MAIN_SCENE.into(),
    }
}

/// Content wrapper for a single-scene template with the full palette.
pub fn content(
    scene: SceneFile,
    prefabs: Vec<(String, crate::prefab::Prefab)>,
    mut bindings: zestengine_core::Bindings,
    assets: Vec<&'static str>,
) -> TemplateContent {
    let mut assets = assets;
    for scheme in &mut bindings.schemes {
        scheme
            .actions
            .push(button("Switch Avatar", keys(&[Key::F6])));
    }
    assets.extend([super::avatar::MALE, super::avatar::FEMALE]);
    for path in [
        super::weapons::RIFLE,
        super::weapons::PISTOL,
        super::weapons::BLASTER,
    ] {
        if scene
            .entries
            .iter()
            .any(|e| matches!(&e.source, ObjectSource::Model { path: model, .. } if model == path))
        {
            assets.push(path);
        }
    }
    if scene.skybox.as_deref() == Some(SKY) && !assets.contains(&SKY) {
        assets.push(SKY);
    }
    TemplateContent {
        scenes: vec![(MAIN_SCENE.into(), scene)],
        materials: palette(),
        prefabs,
        bindings,
        assets,
        boot_scene: MAIN_SCENE.into(),
    }
}
