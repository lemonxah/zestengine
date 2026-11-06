//! HUD bindings: show gameplay values in UI Text / UI Progress Bar elements
//! without code.

use serde::{Deserialize, Serialize};

use crate::{ComponentCtx, ObjectRef, TypedComponent, UiProgressBar, UiText};

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
    fn read(&self, ctx: &ComponentCtx) -> Option<(f32, f32, Option<String>)> {
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
                .and_then(|s| s.get(name))
                .map(|r| (r.amount, r.capacity, None)),
            HudValue::SelectedCount => ctx.get::<RtsSelection>(self.source).map(|s| (s.selected.len() as f32, 0.0, None)),
        }
    }
}

impl TypedComponent for HudBinding {
    const NAME: &'static str = "HUD Binding";

    fn update(&mut self, ctx: &mut ComponentCtx) {
        self.score += ctx.events::<ScoreEvent>().map(|e| e.points).sum::<u32>();
        let Some((value, max, over)) = self.read(ctx) else {
            return;
        };
        let text = over.unwrap_or_else(|| format_hud(&self.format, value, max));
        let fraction = if max > 0.0 { (value / max).clamp(0.0, 1.0) } else { 0.0 };
        if self.shown.as_ref().is_some_and(|(t, f)| *t == text && (*f - fraction).abs() < 1e-4) {
            return;
        }
        let id = ctx.self_id();
        if ctx.get_self::<UiText>().is_some() {
            let t = text.clone();
            ctx.modify::<UiText>(id, move |ui| ui.text = t);
        }
        if ctx.get_self::<UiProgressBar>().is_some() {
            ctx.modify::<UiProgressBar>(id, move |bar| bar.value = fraction);
        }
        self.shown = Some((text, fraction));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_substitutes_whole_numbers() {
        assert_eq!(format_hud("{value} / {max}", 29.6, 120.0), "30 / 120");
        assert_eq!(format_hud("HP {value}", 7.0, 0.0), "HP 7");
    }
}
