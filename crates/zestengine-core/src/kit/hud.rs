//! HUD bindings: show gameplay values in UI Text / UI Progress Bar elements
//! without code.

use serde::{Deserialize, Serialize};

use crate::ecs::{ObjectContext, Owner, Stage, SystemContext, SystemWorld};
use crate::MotionContext;
use crate::{ObjectRef, TypedComponent, UiProgressBar, UiText};

use super::{Health, HitscanWeapon, ResourceStore, RtsSelection, ScoreEvent};

/// Which gameplay value a HUD element shows.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub enum HudValue {
    /// `source`'s [`Health`]: value = current, max = max.
    Health,
    /// `source`'s [`HitscanWeapon`]: value = magazine, max = reserve.
    Ammo,
    /// Sum of every [`ScoreEvent`] since play started.
    Score,
    /// A named amount in `source`'s [`ResourceStore`].
    Resource(String),
    /// Units selected by `source`'s [`RtsSelection`].
    SelectedCount,
}

impl HudValue {
    /// One of each kind, for the inspector's picker.
    pub fn kinds() -> [HudValue; 5] {
        [Self::Health, Self::Ammo, Self::Score, Self::Resource("Gold".into()), Self::SelectedCount]
    }
    /// Inspector label of this value.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Health => "Health",
            Self::Ammo => "Ammo",
            Self::Score => "Score",
            Self::Resource(_) => "Resource",
            Self::SelectedCount => "Selected units",
        }
    }
}

/// `format` with `{value}` / `{max}` replaced (whole numbers).
pub fn format_hud(format: &str, value: f32, max: f32) -> String {
    format
        .replace("{value}", &format!("{}", value.round() as i64))
        .replace("{max}", &format!("{}", max.round() as i64))
}

/// Writes a gameplay value into this object's `UI Text` (through `format`)
/// and/or `UI Progress Bar` (as value / max) every frame.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct HudBinding {
    /// The object the value comes from.
    pub source: ObjectRef,
    /// Which value to show.
    pub value: HudValue,
    /// Text template: `{value}` and `{max}` are substituted.
    pub format: String,
    /// Text shown while `source`'s weapon reloads (Ammo only).
    pub reloading_text: String,
    /// Total score so far (runtime).
    #[serde(skip)]
    pub score: u32,
    /// Last text and value written to the UI (runtime).
    #[serde(skip)]
    pub shown: Option<(String, f32)>,
}

impl Default for HudBinding {
    fn default() -> Self {
        Self {
            source: ObjectRef::NONE,
            value: HudValue::Health,
            format: "{value}".into(),
            reloading_text: "Reloading".into(),
            score: 0,
            shown: None,
        }
    }
}

impl HudBinding {
    fn read(&self, ctx: &ObjectContext<'_, '_>) -> Option<(f32, f32, Option<String>)> {
        match &self.value {
            HudValue::Health => ctx.get::<Health>(self.source).map(|h| (h.current, h.max, None)),
            HudValue::Ammo => ctx.get::<HitscanWeapon>(self.source).map(|w| {
                let max = if w.infinite_reserve { w.magazine_size as f32 } else { w.reserve as f32 };
                let text = w.reloading().then(|| self.reloading_text.clone());
                (w.ammo as f32, max, text)
            }),
            HudValue::Score => Some((self.score as f32, 0.0, None)),
            HudValue::Resource(name) => ctx
                .get::<ResourceStore>(self.source)
                .and_then(|s| s.get(name).map(|r| (r.amount, r.capacity, None))),
            HudValue::SelectedCount => ctx.get::<RtsSelection>(self.source).map(|s| (s.selected.len() as f32, 0.0, None)),
        }
    }
}

impl TypedComponent for HudBinding {
    const NAME: &'static str = "HUD Binding";
}

impl HudBinding {
    fn update_native(&mut self, ctx: &mut ObjectContext<'_, '_>) {
        self.score += ctx.events::<ScoreEvent>().map(|e| e.points).sum::<u32>();
        let Some((value, max, over)) = self.read(ctx) else {
            return;
        };
        let text = over.unwrap_or_else(|| format_hud(&self.format, value, max));
        let fraction = if max > 0.0 { (value / max).clamp(0.0, 1.0) } else { 0.0 };
        let id = ctx.self_id();
        ctx.edit::<UiText>(id, |ui| { if ui.text != text { ui.text = text.clone(); } });
        ctx.edit::<UiProgressBar>(id, |bar| bar.value = fraction);
        self.shown = Some((text, fraction));
    }
}

pub(crate) fn register_systems(registry: &mut crate::ComponentRegistry) {
    registry.register_ecs::<HudBinding>();
    // Read after gameplay updates, including selections and actions while those
    // remaining components are converted. Score still consumes prior-frame events.
    registry.register_system_for::<HudBinding>("zestengine::hud", Stage::LateUpdate, update);
}

fn update(world: &SystemWorld<'_>, context: &mut SystemContext<'_>) {
    let mut query = world.query::<(&Owner, &mut HudBinding)>();
    let mut actors: Vec<_> = query.iter().filter(|(_, (owner, _))| context.is_active(world, owner.0)).collect();
    actors.sort_unstable_by_key(|(_, (owner, _))| context.order_key(world, owner.0));
    for (_, (owner, value)) in actors { value.update_native(&mut ObjectContext::new(world, owner.0, context)); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_hud_updates_new_ui_payloads_and_skips_disabled_instances() {
        use crate::{AnyBox, ComponentRegistry, InputState};
        use crate::ecs::SceneWorld;
        let registry = ComponentRegistry::with_builtins();
        let mut scene = SceneWorld::new();
        let health = scene.spawn("Health");
        scene.insert(health, Health { current: 40.0, ..Default::default() }).unwrap();
        let hud = scene.spawn("HUD");
        scene.insert(hud, HudBinding { source: scene.object_id(health).unwrap().into(), ..Default::default() }).unwrap();
        let input = InputState::default();
        let mut commands = Vec::new();
        registry.run_systems(Stage::LateUpdate, &scene, &mut SystemContext::new(&input, &mut commands));
        scene.insert(hud, UiText::default()).unwrap();
        scene.insert(hud, UiProgressBar::default()).unwrap();
        registry.run_systems(Stage::LateUpdate, &scene, &mut SystemContext::new(&input, &mut commands));
        assert_eq!(scene.get::<UiText>(hud).unwrap().text, "40");
        assert!((scene.get::<UiProgressBar>(hud).unwrap().value - 0.4).abs() < 1e-5);
        let value = scene.remove::<HudBinding>(hud).unwrap();
        scene.insert_disabled(hud, value).unwrap();
        let events = [AnyBox(Box::new(ScoreEvent { points: 5 }))];
        let mut context = SystemContext::new(&input, &mut commands);
        context.services.events = &events;
        registry.run_systems(Stage::LateUpdate, &scene, &mut context);
        assert_eq!(scene.get::<crate::ecs::DisabledComponent<HudBinding>>(hud).unwrap().score, 0);
        assert!(commands.is_empty(), "UI payloads are updated directly without erased modify closures");
    }

    #[test]
    fn format_substitutes_whole_numbers() {
        assert_eq!(format_hud("{value} / {max}", 29.6, 120.0), "30 / 120");
        assert_eq!(format_hud("HP {value}", 7.0, 0.0), "HP 7");
    }
}
