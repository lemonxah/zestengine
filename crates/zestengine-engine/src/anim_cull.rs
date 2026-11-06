//! Animator visibility for `Animator.culling` (ANIM-4).
//!
//! The host hands the scene its camera's view-projection each frame; an
//! animated object counts as visible when its posed bounding sphere touches
//! the four side planes of that frustum (no near / far test: it works for any
//! depth convention and orthographic cameras). Shadow casters keep animating
//! while their shadow can reach the view: for a directional light the sphere
//! is swept [`SHADOW_REACH`] metres along the light direction, for a point or
//! spot light the object must sit inside the light's range sphere while that
//! sphere touches the frustum. When the renderer reports its shadow views
//! (cascades, spot / point faces) a caster must also sit inside one of them
//! (the near plane excluded, as `shadow_work` culls casters), otherwise no
//! shadow of it is drawn at all. Extra views (the XR eyes) count like the
//! host camera; while the renderer draws a secondary view the scene cannot
//! test (a planar mirror, a render-texture camera, the editor's camera
//! preview, a live reflection capture of dynamic objects) nothing is culled.
//!
//! [`pose_plan`] turns visibility and the Animator's culling mode into the
//! work an animated object does this frame, and [`SentPose`] keeps a pose
//! that did not change from being uploaded and re-skinned again.

use glam::{Mat4, Vec3, Vec4};
use zestengine_core::AnimatorCullingMode;

/// How far (m) a directional shadow is assumed to reach from its caster.
pub const SHADOW_REACH: f32 = 60.0;

/// A hidden animator whose bones something else reads (Bone Attach, Weapon
/// Hold) re-poses them every this many frames (root motion and IK every frame).
pub const HIDDEN_BONE_INTERVAL: u32 = 4;

/// The last frame's pose bounds may lag a hidden animation: the sphere tested
/// grows by this share plus [`CULL_MARGIN_M`] metres.
pub const CULL_MARGIN_SCALE: f32 = 1.25;
pub const CULL_MARGIN_M: f32 = 0.25;

/// What an animated object does this frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PosePlan {
    /// Seen: pose, drive the face, build and upload the palette, skin.
    Full,
    /// Not seen, but its bones are read (Always Animate, root motion, IK,
    /// Bone Attach / Weapon Hold targets): pose the skeleton; no blendshapes,
    /// palette, upload or skinning.
    Bones,
    /// Not seen: only the clock, the state machine and events advance.
    Clock,
    /// Not seen under Cull Completely: nothing runs.
    Frozen,
}

impl PosePlan {
    /// Whether the skeleton is posed this frame.
    pub fn poses(self) -> bool {
        matches!(self, Self::Full | Self::Bones)
    }
}

/// What reads an animated object's bones besides its own renderers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoseNeeds {
    /// A Bone Attach or Weapon Hold follows one of its bones.
    pub followed: bool,
    /// Its Animator moves the object by root motion.
    pub root_motion: bool,
    /// Humanoid IK solves on it (the local player's tracked body).
    pub ik: bool,
}

/// The work an animated object does this frame. `visible`: its renderers can
/// be seen in some view; `tick` staggers the low-rate bone updates.
pub fn pose_plan(mode: AnimatorCullingMode, visible: bool, needs: PoseNeeds, tick: u32) -> PosePlan {
    if visible {
        return PosePlan::Full;
    }
    match mode {
        // Unity's Always Animate: transforms keep updating offscreen; the
        // renderer's skinning is skipped all the same (nothing shows it).
        AnimatorCullingMode::AlwaysAnimate => PosePlan::Bones,
        AnimatorCullingMode::CullCompletely => PosePlan::Frozen,
        AnimatorCullingMode::CullUpdateTransforms => {
            if needs.root_motion || needs.ik || (needs.followed && tick.is_multiple_of(HIDDEN_BONE_INTERVAL)) {
                PosePlan::Bones
            } else {
                PosePlan::Clock
            }
        }
    }
}

/// The last pose sent for one skinned copy: a pose equal to it is not sent
/// again once it was sent twice in a row (the second send makes the skinner's
/// previous-frame pose catch up, so motion vectors read still).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SentPose {
    key: Option<u64>,
    repeats: u8,
}

impl SentPose {
    /// Whether the pose hashed as `key` must be sent this frame (and record it).
    pub fn need(&mut self, key: u64) -> bool {
        if self.key == Some(key) {
            if self.repeats >= 1 {
                return false;
            }
            self.repeats += 1;
            return true;
        }
        self.key = Some(key);
        self.repeats = 0;
        true
    }
}

/// Hash of a skinned copy's pose: its palette hash and blendshape weights.
pub fn pose_key(palette_hash: u64, weights: &[f32]) -> u64 {
    let mut h = palette_hash ^ 0xcbf29ce484222325;
    for w in weights {
        h ^= w.to_bits() as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h ^ (weights.len() as u64).rotate_left(17)
}

/// The side and far planes of a shadow view (near excluded: a caster between
/// the light and the volume still shadows it), as the renderer's `shadow_work`
/// culls casters.
#[derive(Clone, Debug)]
pub struct ShadowFrustum([Vec4; 5]);

impl ShadowFrustum {
    pub fn new(view_proj: &Mat4) -> Self {
        let r = |i: usize| view_proj.row(i);
        Self([r(3) + r(0), r(3) - r(0), r(3) + r(1), r(3) - r(1), r(3) - r(2)].map(norm_plane))
    }

    pub fn touches(&self, center: Vec3, radius: f32) -> bool {
        self.0.iter().all(|p| p.truncate().dot(center) + p.w >= -radius)
    }
}

fn norm_plane(p: Vec4) -> Vec4 {
    let l = p.truncate().length();
    if l > 1e-9 { p / l } else { p }
}

fn side_planes(view_proj: &Mat4) -> [Vec4; 4] {
    let r = |i: usize| view_proj.row(i);
    // Left, right, bottom, top (Gribb-Hartmann on the clip-space w row).
    [r(3) + r(0), r(3) - r(0), r(3) + r(1), r(3) - r(1)].map(norm_plane)
}

/// A light that can cast a visible shadow of an offscreen object.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ShadowSource {
    /// World travel direction, normalized.
    Directional(Vec3),
    Local { position: Vec3, range: f32 },
}

/// The side planes of the camera frusta that see the scene (normals point
/// inward), the realtime shadow-casting lights and the renderer's shadow views.
#[derive(Clone, Debug)]
pub struct CullView {
    /// The host camera first, then extra views (XR eyes).
    views: Vec<[Vec4; 4]>,
    pub shadows: Vec<ShadowSource>,
    /// The renderer's shadow views last frame; empty = unknown (not used).
    pub shadow_views: Vec<ShadowFrustum>,
}

impl CullView {
    pub fn new(view_proj: &Mat4, shadows: Vec<ShadowSource>) -> Self {
        Self { views: vec![side_planes(view_proj)], shadows, shadow_views: Vec::new() }
    }

    /// Another camera that sees the scene (an XR eye).
    pub fn add_view(&mut self, view_proj: &Mat4) {
        self.views.push(side_planes(view_proj));
    }

    /// Whether a sphere touches any camera's frustum.
    pub fn sphere_in_view(&self, center: Vec3, radius: f32) -> bool {
        self.views.iter().any(|planes| planes.iter().all(|p| p.truncate().dot(center) + p.w >= -radius))
    }

    /// Whether the object is seen or throws a shadow that can be seen.
    pub fn visible(&self, center: Vec3, radius: f32, casts_shadow: bool) -> bool {
        if self.sphere_in_view(center, radius) {
            return true;
        }
        if !casts_shadow {
            return false;
        }
        // Outside every shadow view the renderer draws no shadow of it at all.
        if !self.shadow_views.is_empty() && !self.shadow_views.iter().any(|f| f.touches(center, radius)) {
            return false;
        }
        self.shadows.iter().any(|s| match *s {
            ShadowSource::Directional(dir) => {
                // The caster's sphere swept along the light direction: sample it
                // densely enough that consecutive spheres overlap.
                let step = radius.max(0.25);
                let n = (SHADOW_REACH / step).ceil().clamp(1.0, 64.0) as usize;
                (1..=n).any(|k| self.sphere_in_view(center + dir * (SHADOW_REACH * k as f32 / n as f32), radius.max(SHADOW_REACH / n as f32)))
            }
            ShadowSource::Local { position, range } => {
                center.distance(position) <= range + radius && self.sphere_in_view(position, range)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view() -> CullView {
        // Camera at the origin looking down -Z, 90 degree FOV, square.
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0);
        CullView::new(&proj, Vec::new())
    }

    #[test]
    fn frustum_sides() {
        let v = view();
        assert!(v.sphere_in_view(Vec3::new(0.0, 0.0, -5.0), 0.5));
        assert!(!v.sphere_in_view(Vec3::new(0.0, 0.0, 5.0), 0.5), "behind the camera");
        assert!(!v.sphere_in_view(Vec3::new(20.0, 0.0, -5.0), 1.0), "far to the right");
        assert!(v.sphere_in_view(Vec3::new(5.5, 0.0, -5.0), 1.5), "edge touches");
        assert!(!v.sphere_in_view(Vec3::new(0.0, -20.0, -5.0), 1.0));
    }

    #[test]
    fn works_with_the_view_matrix_and_ortho() {
        let proj = Mat4::orthographic_rh(-5.0, 5.0, -5.0, 5.0, 0.1, 100.0);
        let view_m = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 10.0), Vec3::ZERO, Vec3::Y);
        let v = CullView::new(&(proj * view_m), Vec::new());
        assert!(v.sphere_in_view(Vec3::new(4.0, 0.0, 0.0), 0.5));
        assert!(!v.sphere_in_view(Vec3::new(8.0, 0.0, 0.0), 0.5));
    }

    #[test]
    fn offscreen_caster_keeps_animating_when_its_shadow_reaches_the_view() {
        let mut v = view();
        let outside = Vec3::new(30.0, 0.0, -10.0);
        assert!(!v.visible(outside, 1.0, true), "no lights: hidden");
        // Light travelling toward -X drops the shadow onto the visible area.
        v.shadows.push(ShadowSource::Directional(Vec3::NEG_X));
        assert!(v.visible(outside, 1.0, true));
        assert!(!v.visible(outside, 1.0, false), "a non-caster is still hidden");
        // Light travelling away from the view does not.
        v.shadows = vec![ShadowSource::Directional(Vec3::X)];
        assert!(!v.visible(outside, 1.0, true));
        // A point light near the view lights an object within its range.
        v.shadows = vec![ShadowSource::Local { position: Vec3::new(8.0, 0.0, -8.0), range: 6.0 }];
        assert!(v.visible(Vec3::new(12.0, 0.0, -8.0), 1.0, true));
        assert!(!v.visible(Vec3::new(40.0, 0.0, -8.0), 1.0, true));
    }

    #[test]
    fn a_caster_outside_every_shadow_view_casts_nothing_visible() {
        let mut v = view();
        v.shadows.push(ShadowSource::Directional(Vec3::NEG_X));
        let outside = Vec3::new(30.0, 0.0, -10.0);
        assert!(v.visible(outside, 1.0, true), "its shadow would sweep into the view");
        // A cascade around the visible area that does not reach the caster.
        let light = Mat4::orthographic_rh(-10.0, 10.0, -10.0, 10.0, 0.1, 100.0) * Mat4::look_at_rh(Vec3::new(0.0, 0.0, 50.0), Vec3::ZERO, Vec3::Y);
        v.shadow_views = vec![ShadowFrustum::new(&light)];
        assert!(!v.visible(outside, 1.0, true), "the renderer never draws it into a shadow map");
        // A wide one that does.
        let wide = Mat4::orthographic_rh(-40.0, 40.0, -40.0, 40.0, 0.1, 100.0) * Mat4::look_at_rh(Vec3::new(0.0, 0.0, 50.0), Vec3::ZERO, Vec3::Y);
        v.shadow_views.push(ShadowFrustum::new(&wide));
        assert!(v.visible(outside, 1.0, true));
        // Behind the light's near plane still casts (near is not tested).
        let f = ShadowFrustum::new(&light);
        assert!(f.touches(Vec3::new(0.0, 0.0, 80.0), 0.5));
        assert!(!f.touches(Vec3::new(0.0, 0.0, -80.0), 0.5), "past the far plane");
    }

    #[test]
    fn an_extra_view_counts_like_the_host_camera() {
        let mut v = view();
        let behind = Vec3::new(0.0, 0.0, 5.0);
        assert!(!v.visible(behind, 0.5, false));
        // An XR eye looking the other way.
        let eye = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0) * Mat4::look_at_rh(Vec3::ZERO, Vec3::Z, Vec3::Y);
        v.add_view(&eye);
        assert!(v.visible(behind, 0.5, false));
    }

    #[test]
    fn pose_plan_follows_visibility_and_the_culling_mode() {
        use AnimatorCullingMode::*;
        let none = PoseNeeds::default();
        for mode in AnimatorCullingMode::ALL {
            assert_eq!(pose_plan(mode, true, none, 0), PosePlan::Full, "{mode:?}: seen = full work");
        }
        assert_eq!(pose_plan(AlwaysAnimate, false, none, 0), PosePlan::Bones, "offscreen bones keep moving, no skinning");
        assert_eq!(pose_plan(CullUpdateTransforms, false, none, 0), PosePlan::Clock);
        assert_eq!(pose_plan(CullCompletely, false, none, 0), PosePlan::Frozen);
        // Root motion and IK are read every frame.
        let rm = PoseNeeds { root_motion: true, ..none };
        let ik = PoseNeeds { ik: true, ..none };
        for t in 0..HIDDEN_BONE_INTERVAL {
            assert_eq!(pose_plan(CullUpdateTransforms, false, rm, t), PosePlan::Bones);
            assert_eq!(pose_plan(CullUpdateTransforms, false, ik, t), PosePlan::Bones);
        }
        assert_eq!(pose_plan(CullCompletely, false, rm, 0), PosePlan::Frozen, "Cull Completely stops root motion too");
        // A Bone Attach target re-poses at a low rate.
        let followed = PoseNeeds { followed: true, ..none };
        let posed = (0..HIDDEN_BONE_INTERVAL * 3).filter(|&t| pose_plan(CullUpdateTransforms, false, followed, t) == PosePlan::Bones).count();
        assert_eq!(posed, 3);
        assert!(PosePlan::Bones.poses() && PosePlan::Full.poses() && !PosePlan::Clock.poses() && !PosePlan::Frozen.poses());
    }

    #[test]
    fn an_unchanged_pose_is_sent_twice_then_skipped() {
        let mut s = SentPose::default();
        let a = pose_key(1, &[0.0, 0.5]);
        assert!(s.need(a), "first pose");
        assert!(s.need(a), "repeated once so the previous-frame pose catches up");
        assert!(!s.need(a) && !s.need(a), "then skipped");
        let b = pose_key(2, &[0.0, 0.5]);
        assert_ne!(a, b);
        assert!(s.need(b), "a new palette is sent");
        assert!(s.need(b));
        assert!(!s.need(b));
        // Blendshape weights alone change the key.
        let c = pose_key(2, &[0.1, 0.5]);
        assert_ne!(b, c);
        assert!(s.need(c));
        assert_ne!(pose_key(2, &[]), pose_key(2, &[0.0]), "a weight list appearing is a change");
    }
}
