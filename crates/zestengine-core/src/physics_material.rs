//! `.physmat` physics material assets: how a surface behaves in contact and
//! in water. Rigid bodies and colliders reference one by project-relative
//! path; a collider's material overrides its body's, and the project default
//! covers everything else.

use std::path::Path;

use serde::{Deserialize, Serialize};

/// File extension of `.physmat` assets.
pub const PHYSMAT_EXTENSION: &str = "physmat";

/// How two touching materials combine a coefficient. When they disagree the
/// stronger rule wins (Maximum > Multiply > Minimum > Average).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CombineMode {
    /// The mean of the two values.
    #[default]
    Average,
    /// The smaller of the two values.
    Minimum,
    /// The product of the two values.
    Multiply,
    /// The larger of the two values.
    Maximum,
}

impl CombineMode {
    /// Every mode, in declaration order.
    pub const ALL: [CombineMode; 4] = [Self::Average, Self::Minimum, Self::Multiply, Self::Maximum];

    /// Inspector label of this mode.
    pub fn label(self) -> &'static str {
        match self {
            Self::Average => "Average",
            Self::Minimum => "Minimum",
            Self::Multiply => "Multiply",
            Self::Maximum => "Maximum",
        }
    }
}

/// How a surface behaves in contact and in water.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PhysicsMaterial {
    /// Coulomb friction coefficient while sliding (0 = ice, ~0.5 wood, 1+
    /// rubber).
    pub friction: f32,
    /// Friction holding things at rest, usually a little above `friction`
    /// (it takes more to start something sliding than to keep it going).
    /// None = the same as `friction`.
    pub static_friction: Option<f32>,
    /// How this material's friction combines with the other surface's.
    pub friction_combine: CombineMode,
    /// Restitution 0..1 (1 = keeps all its speed on a bounce).
    pub bounciness: f32,
    /// How this material's bounciness combines with the other surface's.
    pub bounce_combine: CombineMode,
    /// kg/m^3, used for the mass of bodies whose Rigid Body mass is 0 (water
    /// is 1000, wood ~600, steel ~7850).
    pub density: f32,
    /// Scales the water's lift on the body (Archimedes, or a Buoyancy
    /// component's override).
    pub buoyancy: f32,
    /// Scales the water's drag on this body.
    pub water_drag: f32,
    /// Extra contact distance (m) so thin or fast shapes touch before they
    /// overlap.
    pub contact_offset: f32,
}

impl Default for PhysicsMaterial {
    fn default() -> Self {
        Self {
            friction: 0.5,
            static_friction: None,
            friction_combine: CombineMode::Average,
            bounciness: 0.0,
            bounce_combine: CombineMode::Average,
            density: 1000.0,
            buoyancy: 1.0,
            water_drag: 1.0,
            contact_offset: 0.0,
        }
    }
}

impl PhysicsMaterial {
    /// Static friction: `static_friction`, or `friction` when unset.
    pub fn static_mu(&self) -> f32 {
        self.static_friction.unwrap_or(self.friction).max(0.0)
    }

    /// A few starting points for the editor.
    pub const PRESETS: [(&'static str, PhysicsMaterial); 6] = [
        ("Wood", Self::preset(0.5, 0.6, 0.2, 600.0)),
        ("Metal", Self::preset(0.4, 0.55, 0.1, 7850.0)),
        ("Rubber", Self::preset(0.9, 1.1, 0.8, 1100.0)),
        ("Ice", Self::preset(0.02, 0.08, 0.05, 917.0)),
        ("Stone", Self::preset(0.6, 0.75, 0.05, 2600.0)),
        ("Foam", Self::preset(0.6, 0.8, 0.3, 50.0)),
    ];

    const fn preset(friction: f32, static_friction: f32, bounciness: f32, density: f32) -> Self {
        Self {
            friction,
            static_friction: Some(static_friction),
            friction_combine: CombineMode::Average,
            bounciness,
            bounce_combine: CombineMode::Average,
            density,
            buoyancy: 1.0,
            water_drag: 1.0,
            contact_offset: 0.0,
        }
    }
}

/// Read and parse a `.physmat`.
pub fn load_physmat(path: impl AsRef<Path>) -> Result<PhysicsMaterial, String> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    ron::from_str(&text).map_err(|e| format!("parsing {}: {e}", path.display()))
}

/// Serialize and write a `.physmat`.
pub fn save_physmat(path: impl AsRef<Path>, mat: &PhysicsMaterial) -> Result<(), String> {
    let path = path.as_ref();
    let text = ron::ser::to_string_pretty(mat, ron::ser::PrettyConfig::default()).map_err(|e| e.to_string())?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(path, text).map_err(|e| format!("writing {}: {e}", path.display()))
}

/// Project-wide physics (Project Settings > Physics). Contact stiffness is a
/// solver setting rather than a per-material one: every contact uses it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PhysicsSettings {
    /// Gravity (m/s^2, usually negative Y).
    pub gravity: [f32; 3],
    /// Material for bodies and colliders that don't name one ("" = built-in
    /// defaults).
    pub default_material: String,
    /// How hard contacts push overlapping shapes apart (Hz). Higher is
    /// stiffer; lower lets things sink in and settle softly.
    pub contact_stiffness: f32,
    /// Contact damping ratio: higher absorbs the push-back, lower springs.
    pub contact_damping: f32,
    /// Solver iterations per step: more is steadier stacks and joints.
    pub solver_iterations: u32,
}

impl Default for PhysicsSettings {
    fn default() -> Self {
        Self { gravity: [0.0, -9.81, 0.0], default_material: String::new(), contact_stiffness: 30.0, contact_damping: 5.0, solver_iterations: 4 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_fills_missing_fields() {
        let m = PhysicsMaterial { friction: 0.9, bounce_combine: CombineMode::Maximum, ..PhysicsMaterial::PRESETS[2].1 };
        let text = ron::ser::to_string(&m).unwrap();
        assert_eq!(ron::from_str::<PhysicsMaterial>(&text).unwrap(), m);
        let partial: PhysicsMaterial = ron::from_str("(bounciness: 0.7)").unwrap();
        assert_eq!(partial.bounciness, 0.7);
        assert_eq!(partial.density, 1000.0);
    }
}
