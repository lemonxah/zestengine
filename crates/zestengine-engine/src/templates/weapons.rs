//! Normalized BlenderKit weapon visuals. Gameplay and grip/muzzle roots remain
//! ordinary template objects, editable independently of the imported meshes.
use zestengine_assets::ObjectSource;
use super::builder::Scene;

pub const RIFLE: &str = "models/weapons/default/rifle.glb";
pub const PISTOL: &str = "models/weapons/default/pistol.glb";
pub const BLASTER: &str = "models/weapons/default/blaster.glb";

pub fn model(s: &mut Scene, name: &str, parent: usize, position: [f32; 3], path: &'static str) -> usize {
    let model = s.model(name, path, Some(parent), position, 1.0);
    if path == BLASTER
        && let ObjectSource::Model { extra_meshes, .. } = &mut s.entries[model].source {
            // The prepared blaster is one mesh with multiple PBR primitives.
            extra_meshes.extend(1..19);
        }
    model
}
