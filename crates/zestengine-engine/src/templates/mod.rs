//! Project templates for New Project: the starter games (empty, FPS, TPS,
//! top-down click-to-move, strategy, VR shooter, VR social) and the content
//! each writes into a fresh project: scenes, materials, prefabs, input
//! bindings and copied sample assets. Every template is built from the
//! built-in gameplay kit (`zestengine_core::kit`) and plays in the editor as is.
//! The catalogue (names, descriptions) lives in `zestengine_assets::project_templates`
//! so the Project Hub can list it without linking the engine.

mod builder;
mod avatar;
mod weapons;
mod cardgame;
mod fps;
mod novel;
mod strategy;
mod topdown;
mod tps;
mod vr_shooter;
mod vr_social;

use std::path::Path;

use anyhow::{Context as _, Result};
use zestengine_assets::{MaterialFile, SceneFile};
use zestengine_core::Bindings;

use crate::prefab::Prefab;

pub use zestengine_assets::project_templates::{
    DEFAULT_TEMPLATE, ProjectTemplate, TEMPLATES, TemplateCategory, TemplateIcon, find,
};

/// Everything a template writes into a project.
pub(crate) struct TemplateContent {
    /// Project-relative path -> scene.
    pub scenes: Vec<(String, SceneFile)>,
    pub materials: Vec<(String, MaterialFile)>,
    pub prefabs: Vec<(String, Prefab)>,
    pub bindings: Bindings,
    /// Repo-relative sample assets copied into the same relative path.
    pub assets: Vec<&'static str>,
    pub boot_scene: String,
}

pub(crate) fn build(id: &str) -> Result<TemplateContent> {
    let content = match id {
        "empty" => builder::empty(),
        "fps" => fps::build(),
        "tps" => tps::build(),
        "topdown" => topdown::build(),
        "strategy" => strategy::build(),
        "vr_shooter" => vr_shooter::build(),
        "vr_social" => vr_social::build(),
        "novel" => novel::build(),
        "cardgame" => cardgame::build(),
        other => anyhow::bail!(
            "unknown project template {other:?} (expected one of: {})",
            TEMPLATES.iter().map(|t| t.id).collect::<Vec<_>>().join(", ")
        ),
    };
    Ok(content)
}

/// Write a template's content into `root` (which already has its asset
/// folders). `zestengine` is the engine checkout to copy sample assets from; a
/// missing asset is skipped and any scene skybox using it is cleared.
/// Returns the boot scene and bindings for project.zestengine.
pub(crate) fn write_content(root: &Path, zestengine: Option<&Path>, id: &str) -> Result<(String, Bindings)> {
    let mut content = build(id)?;
    let mut missing = Vec::new();
    for rel in &content.assets {
        let copied = zestengine.map(|c| c.join(rel)).filter(|src| src.is_file()).map(|src| {
            let dst = root.join(rel);
            if let Some(dir) = dst.parent() {
                std::fs::create_dir_all(dir)?;
            }
            std::fs::copy(&src, &dst).with_context(|| format!("copying {}", src.display()))
        });
        match copied {
            Some(r) => {
                r?;
            }
            None => {
                tracing::warn!("template asset {rel} not found; skipped");
                missing.push(rel.to_string());
            }
        }
    }
    for (rel, scene) in &mut content.scenes {
        if scene.skybox.as_ref().is_some_and(|s| missing.contains(s)) {
            scene.skybox = None;
        }
        zestengine_assets::save_scene_file(root.join(&*rel), scene).with_context(|| format!("writing {rel}"))?;
    }
    for (rel, material) in &content.materials {
        zestengine_assets::save_material_file(root.join(rel), material).with_context(|| format!("writing {rel}"))?;
    }
    for (rel, prefab) in &content.prefabs {
        let path = root.join(rel);
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        prefab.save(&path).with_context(|| format!("writing {rel}"))?;
    }
    if content.assets.contains(&avatar::MALE) {
        avatar::write_controller(root)?;
    }
    Ok((content.boot_scene, content.bindings))
}

#[cfg(test)]
mod tests;
