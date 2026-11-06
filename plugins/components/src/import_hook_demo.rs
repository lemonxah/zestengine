//! Demo model import hook (IMP-6). Strips the coarser LOD nodes (`*_LOD1`,
//! `*_LOD2`, ...) from imported models and keeps `_LOD0`, so a model exported
//! with its own LOD chain imports as its full-detail mesh.
//!
//! It is OFF unless the editor is started with `ZESTENGINE_DEMO_IMPORT_HOOK=1`,
//! so it never alters a real import by default. Registered hooks show in the
//! model inspector's "Import Hooks" section; the name, version and order are
//! part of every model's cache key (bump `version()` when the behaviour changes).
//!
//! Hooks must be deterministic and depend only on their arguments (and files
//! read through `ctx.read_project_file`); they run in the editor only.

use zestengine_assets::import_hooks::{ImportContext, ImportedHierarchy, ModelImportHook};

/// Environment variable that turns the demo hook on.
pub const DEMO_HOOK_ENV: &str = "ZESTENGINE_DEMO_IMPORT_HOOK";

pub fn demo_hook_enabled() -> bool {
    std::env::var_os(DEMO_HOOK_ENV).is_some_and(|v| v != "0" && !v.is_empty())
}

/// Removes nodes named `<anything>_LOD<n>` for n >= 1.
pub struct StripLodNodes;

/// Whether `name` ends in `_LOD` followed by a number above zero.
pub fn is_coarse_lod(name: &str) -> bool {
    name.rsplit_once("_LOD").is_some_and(|(_, n)| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) && n.parse::<u32>().is_ok_and(|n| n >= 1))
}

impl ModelImportHook for StripLodNodes {
    fn name(&self) -> &str {
        "demo.strip_lod_nodes"
    }

    fn version(&self) -> u32 {
        1
    }

    fn postprocess_hierarchy(&mut self, ctx: &ImportContext, nodes: &mut ImportedHierarchy) {
        let mut removed = 0;
        for node in nodes.nodes.iter_mut().filter(|n| is_coarse_lod(&n.name)) {
            node.remove = true;
            removed += 1;
        }
        if removed > 0 {
            ctx.log_info(format!("removed {removed} LOD node(s)"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marks_the_coarse_lod_nodes_for_removal() {
        use zestengine_assets::import_hooks::ImportedNode;
        let node = |name: &str| ImportedNode { name: name.into(), transform: [0.0; 16], mesh_slots: 1, remove: false };
        let mut h = ImportedHierarchy { nodes: vec![node("Tree_LOD0"), node("Tree_LOD1"), node("Rock")] };
        let ctx = ImportContext::new(std::path::Path::new("/x/m.glb"), &Default::default());
        StripLodNodes.postprocess_hierarchy(&ctx, &mut h);
        assert_eq!(h.nodes.iter().map(|n| n.remove).collect::<Vec<_>>(), [false, true, false]);
    }

    #[test]
    fn matches_only_coarse_lods() {
        assert!(is_coarse_lod("Tree_LOD1"));
        assert!(is_coarse_lod("Tree_LOD12"));
        assert!(!is_coarse_lod("Tree_LOD0"));
        assert!(!is_coarse_lod("Tree_LOD"));
        assert!(!is_coarse_lod("Tree_LODx"));
        assert!(!is_coarse_lod("Tree"));
    }
}
