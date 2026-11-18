//! Project components, compiled and hot-loaded by the zestengine editor
//! (Tools → Build & Reload Components).
//!
//! A component's runtime data is `zestengine_core::TypedComponent` (no UI);
//! ECS systems query that data through `zestengine_core::ecs::SystemWorld`.
//! its *editor* inspector is `zestengine_editor::Inspect` and its viewport gizmo is
//! `zestengine_editor::Gizmo`. Register the runtime in `zestengine_register` and the
//! editor traits in `zestengine_register_editor`. A shipped game calls only the
//! former (no editor linkage).

use zestengine_core::{ActionCtx, ActionDesc, ActionType, ArgValue, ComponentRegistry, ObjectRef, ParamDesc, PropertyDesc, Transform, TypedComponent};
use zestengine_core::ecs::{Active, GlobalTransform, Owner, Parent, SystemWorld, Stage, SystemContext};
#[cfg(feature = "editor")]
use zestengine_editor::{EditorComponents, Gizmo, Inspect, InspectCtx};
use serde::{Deserialize, Serialize};

// ABI stamp: lets a host (the editor, a dedicated server loading this cdylib) refuse a build from a
// different rustc / engine version instead of crashing.
zestengine_core::export_plugin_abi!();

/// Register runtime behaviour (called by the editor and by a shipped game).
#[unsafe(no_mangle)]
pub fn zestengine_register(registry: &mut ComponentRegistry) {
    registry.register_ecs::<Orbit>();
    registry.register_system("project::orbit", Stage::Update, orbit_system);
    // zestengine: new components register here
}

/// Register editor-only traits (inspector + gizmo). Compiled only with the
/// `editor` feature (the editor builds plugins with it); absent in a game.
#[cfg(feature = "editor")]
#[unsafe(no_mangle)]
pub fn zestengine_register_editor(editor: &mut EditorComponents) {
    editor.register::<Orbit>();
    // zestengine: new components register their editor traits here
}

#[cfg(feature = "editor")]
mod import_hook_demo;

/// Import hook API this plugin was built against (editor builds only: hooks run
/// in the editor, never in a shipped game). The editor registers the hooks only
/// when it matches its own, so an old or newer plugin degrades to no hooks.
#[cfg(feature = "editor")]
#[unsafe(no_mangle)]
pub fn zestengine_import_hook_api_version() -> u32 {
    zestengine_assets::import_hooks::IMPORT_HOOK_API_VERSION
}

/// Register model import hooks. The demo hook is off unless
/// `ZESTENGINE_DEMO_IMPORT_HOOK=1`, so it never changes anyone's real imports.
#[cfg(feature = "editor")]
#[unsafe(no_mangle)]
pub fn zestengine_register_import_hooks(registry: &mut zestengine_assets::import_hooks::ImportHookRegistry) {
    if import_hook_demo::demo_hook_enabled() {
        registry.register_import_hook(Box::new(import_hook_demo::StripLodNodes));
    }
    // zestengine: new import hooks register here
}

/// Orbits the owning object around a target object (or its own start point if
/// no target is set).
///
/// The target is a saved object reference resolved to its native entity when
/// the system runs. Inspector/gizmo registration is independent of storage.
#[derive(Serialize, Deserialize)]
pub struct Orbit {
    pub degrees_per_second: f32,
    pub radius: f32,
    /// Object to orbit around (unset = orbit own start point). Set via the
    /// inspector's object picker; resolved by stable id each frame.
    pub target: ObjectRef,
    #[serde(skip)]
    applied: glam::Vec3,
}

impl Default for Orbit {
    fn default() -> Self {
        Self {
            degrees_per_second: 90.0,
            radius: 1.5,
            target: ObjectRef::NONE,
            applied: glam::Vec3::ZERO,
        }
    }
}

impl TypedComponent for Orbit {
    const NAME: &'static str = "Orbit";
    fn actions(&self) -> &'static [ActionDesc] { ORBIT_ACTIONS }
    fn properties(&self) -> &'static [PropertyDesc] { ORBIT_PROPERTIES }
    fn read_property(&self, property: &str) -> Option<ArgValue> {
        match property { "radius" => Some(ArgValue::Float(self.radius)), _ => None }
    }
    fn invoke_action(&mut self, _: &mut ActionCtx<'_>, action: &str, args: &[ArgValue]) -> Result<(), String> {
        match action {
            "SetRadius" => self.radius = args[0].as_f32().max(0.0),
            _ => return Err(zestengine_core::actions::unknown_action(Self::NAME, action)),
        }
        Ok(())
    }
}

static ORBIT_ACTIONS: &[ActionDesc] = &[ActionDesc::new("SetRadius", "Set Radius", &[ParamDesc::float("radius", 1.5)], true)];
static ORBIT_PROPERTIES: &[PropertyDesc] = &[PropertyDesc::new("radius", ActionType::Float)];

fn orbit_system(world: &SystemWorld<'_>, ctx: &mut SystemContext<'_>) {
    for (_, (owner, orbit)) in world.query::<(&Owner, &mut Orbit)>().iter() {
        if !world.get::<Active>(owner.0).is_some_and(|active| active.0) { continue; }
        let angle = ctx.time * orbit.degrees_per_second.to_radians();
        let offset = glam::Vec3::new(angle.cos(), 0.0, angle.sin()) * orbit.radius;
        // Orbit around the target object's world position if set and found;
        // otherwise orbit relative to the object's own start point.
        let center = orbit.target.id().and_then(|id| world.entity(id))
            .and_then(|target| world.get::<GlobalTransform>(target).map(|global| global.0.w_axis.truncate()));
        let mut transform = world.get_mut::<Transform>(owner.0).expect("orbit owner");
        match center {
            // Convert the desired world position through the parent's frame.
            Some(c) => {
                let parent = world.get::<Parent>(owner.0).and_then(|parent| parent.0)
                    .and_then(|parent| world.get::<GlobalTransform>(parent).map(|global| global.0));
                transform.translation = parent.map_or(c + offset, |parent| parent.inverse().transform_point3(c + offset));
            }
            None => {
                transform.translation += offset - orbit.applied;
                orbit.applied = offset;
            }
        }
    }
}

#[cfg(feature = "editor")]
impl Inspect for Orbit {
    fn inspector_ui(&mut self, ui: &mut egui::Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        changed |= ui
            .add(egui::Slider::new(&mut self.degrees_per_second, -360.0..=360.0).text("Speed (°/s)"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.radius, 0.0..=10.0).text("Radius"))
            .changed();
        // Object picker dropdown for the orbit target.
        changed |= ctx.object_ref(ui, "Target", &mut self.target);
        changed
    }
}

#[cfg(feature = "editor")]
impl Gizmo for Orbit {}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_core::ecs::{SceneWorld, ComponentSet};

    #[test]
    fn ecs_orbit_uses_saved_target_and_parent_space_and_keeps_serializable_fields() {
        let mut registry = ComponentRegistry::default();
        zestengine_register(&mut registry);
        let mut world = SceneWorld::new();
        let parent = world.spawn("Parent");
        let target = world.spawn("Target");
        let child = world.spawn("Orbiter");
        world.get_mut::<Transform>(parent).unwrap().translation = glam::Vec3::X * 10.0;
        world.get_mut::<Transform>(target).unwrap().translation = glam::Vec3::X * 20.0;
        world.set_parent(child, Some(parent), false).unwrap();
        let mut components = ComponentSet::from_vec(vec![Box::new(Orbit {
            target: ObjectRef::from(world.object_id(target).unwrap()), ..Default::default()
        })]);
        components.attach(&world, child);
        let input = Default::default();
        let mut commands = Vec::new();
        registry.run_systems(Stage::Update, &world, &mut SystemContext {
            dt: 0.25, unscaled_dt: 0.25, time: 0.0, water_time: 0.0,
            input: &input, commands: &mut commands, services: Default::default(),
        });
        assert!((world.get::<Transform>(child).unwrap().translation.x - 11.5).abs() < 1e-5);
        let restored: Orbit = ron_value(&components.get(0).unwrap().save());
        assert_eq!(restored.radius, 1.5);
        assert_eq!(restored.target.id(), world.object_id(target));
    }

    // Exercise the registry's scene-file loader, without adding a direct RON
    // dependency to a project plugin.
    fn ron_value(data: &str) -> Orbit {
        let mut registry = ComponentRegistry::default();
        registry.register_ecs::<Orbit>();
        *registry.load("Orbit", data).unwrap().into_any().downcast::<Orbit>().unwrap()
    }
}
