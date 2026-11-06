//! Clip lookups for the blend tree editor: length and root motion of a clip
//! by the name an Animator addresses it with (an authored `.anim` path or an
//! imported clip of the scene's models).

use super::*;
use crate::clip_motion::{ClipMotion, analyse_authored, analyse_imported};

impl LoadedScene {
    /// The skeleton an object's skinned mesh uses as `(joint name, parent)`; empty without one.
    pub fn skeleton_tree(&self, i: usize) -> Vec<(String, Option<usize>)> {
        let Some(o) = self.objects.get(i) else { return Vec::new() };
        o.render_slots()
            .find_map(|r| self.skinned_meshes.iter().find(|sm| sm.mesh_index == r.mesh))
            .and_then(|sm| self.skeletons.get(sm.skeleton))
            .map(|sk| sk.joints.iter().map(|j| (j.name.clone(), j.parent)).collect())
            .unwrap_or_default()
    }

    /// Length in seconds of the clip `name`, loading an authored clip if needed.
    pub fn clip_length(&mut self, name: &str) -> Option<f32> {
        if crate::anim_graph::is_clip_asset(name) {
            self.anim_assets.ensure_clip(name);
            return self.anim_assets.clips.get(name).and_then(|c| c.as_ref()).map(|c| c.length);
        }
        self.animations.iter().find(|c| c.name == name).map(|c| c.duration)
    }

    /// Average velocity and turn rate of the clip `name`'s root (see
    /// `clip_motion`). `None` for an unknown clip or one with no root motion
    /// channels.
    pub fn clip_motion(&mut self, name: &str) -> Option<ClipMotion> {
        if crate::anim_graph::is_clip_asset(name) {
            self.anim_assets.ensure_clip(name);
            return self.anim_assets.clips.get(name).and_then(|c| c.as_ref()).and_then(|c| analyse_authored(c));
        }
        let idx = self.animations.iter().position(|c| c.name == name)?;
        let clip = &self.animations[idx];
        let skel = self.skeletons.get(*self.clip_skeleton.get(idx)?)?;
        let root = skel.motion_root();
        let hips = skel
            .rig
            .as_ref()
            .and_then(|r| r.joint_name(zestengine_assets::human_rig::HumanBone::Hips))
            .and_then(|n| skel.joints.iter().position(|j| j.name == n));
        // The root when it carries motion, else the Hips (rigs that animate the pelvis).
        let moving = |m: &ClipMotion| m.velocity.length() > 1e-5 || m.angular_speed.abs() > 1e-5;
        let first = root.and_then(|r| analyse_imported(clip, r));
        match first {
            Some(m) if moving(&m) => Some(m),
            other => hips.and_then(|h| analyse_imported(clip, h)).filter(moving).or(other),
        }
    }
}
