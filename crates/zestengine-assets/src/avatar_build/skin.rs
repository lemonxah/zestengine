//! Pass 9, second half: skin rebinding for `Merge Armature`.
//!
//! Zipping the object hierarchy moves bones, but a skinned mesh binds to the joints of its
//! model's skeleton, not to scene objects. For every merged outfit mesh this loads the outfit
//! model and the avatar's model, plans the rebind ([`crate::skin_rebind::plan_rebind`]) so
//! missing bones are known at build time, and tags the outfit object with a `Skin Rebound`
//! component. The runtime applies the same plan when the scene loads (`scene/skin_rebound.rs`
//! in the engine): the mesh is re-skinned onto the avatar's skeleton and the avatar's Animator
//! poses it, while the outfit object keeps its own active flag.

use std::collections::HashMap;

use zestengine_core::{MergeArmature, SkinRebound};

use super::*;
use crate::skin_rebind::{ExtraJoints, RebindOptions, plan_rebind};

/// What the rebind of one outfit model onto the avatar's skeleton did, for the build report.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct SkinRebindReport {
    /// Path of the outfit object below the avatar root.
    pub outfit: String,
    /// Project-relative model paths.
    pub outfit_model: String,
    pub avatar_model: String,
    pub joints: usize,
    pub matched: usize,
    /// Outfit bones with no avatar partner, then appended to the avatar's skeleton or folded
    /// into their nearest matched ancestor (see `folded`).
    pub missing: Vec<String>,
    pub added: usize,
    pub folded: usize,
    /// Joints whose bind pose differs from the avatar's (their vertices were corrected).
    pub corrected: usize,
}

/// A model's first skeleton, when one of its meshes is skinned to it.
fn skinned_skeleton(scene: &crate::Scene) -> Option<&crate::Skeleton> {
    let sk = scene.skeletons.first()?;
    scene.meshes.iter().any(|m| m.vertices.iter().any(|v| v.weights.iter().sum::<f32>() > 0.0)).then_some(sk)
}

impl Builder<'_> {
    fn load_model_cached(&self, cache: &mut HashMap<String, Option<crate::Scene>>, path: &str) {
        if !cache.contains_key(path) {
            let loaded = crate::load_model_with_meta(self.env.project_root.join(path)).ok();
            cache.insert(path.to_owned(), loaded);
        }
    }

    /// Rebind the skinned meshes below `m` (a `Merge Armature` object) onto the avatar.
    pub(super) fn rebind_skins(&mut self, m: usize, cfg: &MergeArmature, who: &str) {
        let model_of = |e: &SceneEntry| match &e.source {
            crate::ObjectSource::Model { path, .. } => Some(path.clone()),
            _ => None,
        };
        let outfit: Vec<usize> = self.sub.iter().copied().filter(|&i| self.is_within(i, m) && model_of(&self.scene.entries[i]).is_some()).collect();
        if outfit.is_empty() {
            return;
        }
        let mut cache: HashMap<String, Option<crate::Scene>> = HashMap::new();
        // The avatar's own skinned model: the first entry outside the outfit that has one.
        let avatar_candidates: Vec<usize> = self.sub.iter().copied().chain([self.root]).filter(|&i| !self.is_within(i, m) && model_of(&self.scene.entries[i]).is_some()).collect();
        let mut avatar: Option<(usize, String)> = None;
        for i in avatar_candidates {
            let path = model_of(&self.scene.entries[i]).unwrap();
            self.load_model_cached(&mut cache, &path);
            if cache.get(&path).and_then(|s| s.as_ref()).and_then(skinned_skeleton).is_some() {
                avatar = Some((i, path));
                break;
            }
        }
        let Some((avatar_entry, avatar_model)) = avatar else {
            self.warn(format!("{who}: the avatar has no skinned model, so the outfit keeps its own skeleton"));
            return;
        };
        if self.scene.entries[avatar_entry].id.is_empty() {
            self.warn(format!("{who}: the avatar's model object has no id; save the scene once so the outfit can be bound to it"));
            return;
        }
        let avatar_id = self.scene.entries[avatar_entry].id.clone();
        let opts = RebindOptions {
            prefix: cfg.prefix.clone(),
            suffix: cfg.suffix.clone(),
            mangle_names: cfg.mangle_names,
            auto_align: false,
            extras: ExtraJoints::Append,
            correct_bind_pose: true,
        };
        let mut reported: Vec<String> = Vec::new();
        for i in outfit {
            let path = model_of(&self.scene.entries[i]).unwrap();
            self.load_model_cached(&mut cache, &path);
            let Some(outfit_scene) = cache.get(&path).and_then(|s| s.as_ref()) else {
                self.warn(format!("{who}: model {path} could not be read; its skin is not rebound"));
                continue;
            };
            let Some(outfit_skel) = skinned_skeleton(outfit_scene) else { continue };
            let avatar_skel = cache.get(&avatar_model).and_then(|s| s.as_ref()).and_then(skinned_skeleton).unwrap();
            let plan = plan_rebind(avatar_skel, outfit_skel, &opts);
            let entry_path = self.paths[i].clone();
            if !reported.contains(&path) {
                reported.push(path.clone());
                let corrected = plan.corrected_joints();
                self.out.skin_rebinds.push(SkinRebindReport {
                    outfit: entry_path.clone(),
                    outfit_model: path.clone(),
                    avatar_model: avatar_model.clone(),
                    joints: outfit_skel.joints.len(),
                    matched: plan.matched.len(),
                    missing: plan.missing.clone(),
                    added: plan.added.len(),
                    folded: plan.folded.len(),
                    corrected,
                });
                if !plan.missing.is_empty() {
                    let shown: Vec<&str> = plan.missing.iter().take(6).map(String::as_str).collect();
                    let more = if plan.missing.len() > 6 { format!(" and {} more", plan.missing.len() - 6) } else { String::new() };
                    self.warn(format!(
                        "{who}: {} bone(s) of {path} have no partner on the avatar ({}{more}); they are added to the avatar's skeleton under their nearest matched bone",
                        plan.missing.len(),
                        shown.join(", ")
                    ));
                }
                if plan.matched.is_empty() {
                    self.warn(format!("{who}: no bone of {path} matches the avatar's; check the prefix and suffix"));
                }
            }
            let tag = SkinRebound {
                avatar: avatar_id.clone(),
                prefix: cfg.prefix.clone(),
                suffix: cfg.suffix.clone(),
                mangle_names: cfg.mangle_names,
                fold_extras: false,
                auto_align: false,
            };
            set_comp(&mut self.scene.entries[i], &tag);
        }
    }
}
