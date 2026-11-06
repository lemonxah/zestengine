//! Engine side of bone cloth: runs `Cloth Colliders` and `Bone Cloth` from the
//! skinning pass (between the animated pose and the skinning palette) and
//! performs the colliders' capsule auto-fit against the skinned vertices.

use std::collections::HashMap;

use zestengine_assets::Skeleton;
use zestengine_core::{
    BoneCloth, BoneClothSim, ClothCollider, ClothColliders, ClothFitPreset, ClothShape, ClothShapeWorld, Component,
    ObjectId, fit_capsule, principal_axis, rigid_frame,
};
use glam::{Mat4, Vec3};

use crate::humanoid::HumanoidRig;

pub(crate) fn has_cloth(components: &[Box<dyn Component>]) -> bool {
    components
        .iter()
        .any(|c| c.as_any().is::<BoneCloth>() || c.as_any().is::<ClothColliders>() || c.as_any().is::<zestengine_core::MeshCloth>())
}

pub(crate) fn has_mesh_cloth(components: &[Box<dyn Component>]) -> bool {
    components.iter().any(|c| c.as_any().is::<zestengine_core::MeshCloth>())
}

/// Whether a Mesh Cloth is bound to render slot `slot`.
pub(crate) fn has_mesh_cloth_slot(components: &[Box<dyn Component>], slot: usize) -> bool {
    components.iter().any(|c| c.as_any().downcast_ref::<zestengine_core::MeshCloth>().is_some_and(|m| m.slot == slot))
}

/// `ZESTENGINE_CLOTH_CPU=1`: skin objects with Mesh Cloth entirely on the CPU
/// (the pre-GPU-override path), as a fallback / A-B switch.
pub(crate) fn cpu_skinning_forced() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ZESTENGINE_CLOTH_CPU").is_some_and(|v| v != "0"))
}

/// Whether cloth on an object at `world` is past its cull distance from the
/// viewer.
fn culled(cull_distance: f32, world: Mat4, viewer: Option<Vec3>) -> bool {
    cull_distance > 0.0 && viewer.is_some_and(|eye| (world.w_axis.truncate() - eye).length() > cull_distance)
}

/// Pin weight per render vertex for a Mesh Cloth. Painted weights are first
/// fitted to the captured bind mesh (remapped if it changed since painting).
fn mesh_pins(mc: &mut zestengine_core::MeshCloth, skel: &Skeleton, verts: &[zestengine_render::Vertex]) -> Vec<f32> {
    use zestengine_core::{PaintFit, PinSource};
    match mc.pin {
        PinSource::Painted => {
            match mc.paint.fit(&mc.runtime.bind, &mc.runtime.indices) {
                PaintFit::Same => mc.runtime.paint_note.clear(),
                PaintFit::Fresh => mc.runtime.paint_note = "Nothing painted yet: every vertex is free.".into(),
                PaintFit::Remapped => {
                    tracing::warn!("Mesh Cloth: the mesh changed since its pins were painted; remapped them by nearest position");
                    mc.runtime.paint_note =
                        "The mesh changed since painting: the weights were carried over by nearest position. Check them, then save the scene.".into();
                }
            }
            mc.paint.pin_weights()
        }
        PinSource::VertexColor => {
            let ch = mc.pin_channel.min(3) as usize;
            verts.iter().map(|v| v.color[ch].clamp(0.0, 1.0)).collect()
        }
        PinSource::Bones => {
            let pinned: Vec<usize> =
                mc.pin_bones.iter().filter_map(|b| skel.joints.iter().position(|j| &j.name == b)).collect();
            verts
                .iter()
                .map(|v| {
                    let w: f32 = v.joints.iter().zip(v.weights).filter(|(j, _)| pinned.contains(&(**j as usize))).map(|(_, w)| w).sum();
                    if w >= 0.5 { 1.0 } else { 0.0 }
                })
                .collect()
        }
        PinSource::Top => {
            let (lo, hi) = verts.iter().fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(v.position[1]), hi.max(v.position[1])));
            let cut = hi - (hi - lo) * mc.pin_height.clamp(0.0, 1.0);
            verts.iter().map(|v| if v.position[1] >= cut { 1.0 } else { 0.0 }).collect()
        }
    }
}

/// Painted max distances in object units, when the cloth uses painted pins.
fn mesh_limits(mc: &zestengine_core::MeshCloth) -> Option<Vec<f32>> {
    (mc.pin == zestengine_core::PinSource::Painted).then(|| mc.paint.limits(mc.max_distance)).flatten()
}

/// Simulate the Mesh Cloth bound to render slot `slot` on these skinned
/// vertices (object space, in place). `bind` are the bind-pose vertices
/// (pins read their colours and weights). Past its cull distance from
/// `viewer`, or held by the pin painter, the cloth keeps the animated pose.
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_mesh(
    components: &mut [Box<dyn Component>],
    slot: usize,
    skel: &Skeleton,
    bind: &[zestengine_render::Vertex],
    verts: &mut [zestengine_render::Vertex],
    indices: &[u32],
    world: Mat4,
    others: &HashMap<ObjectId, Vec<ClothShapeWorld>>,
    winds: &[zestengine_core::WindSample],
    viewer: Option<Vec3>,
    time: f32,
    dt: f32,
) {
    let mut own: Vec<(String, ClothShapeWorld)> = Vec::new();
    for c in components.iter() {
        if let Some(cc) = c.as_any().downcast_ref::<ClothColliders>() {
            for col in cc.colliders.iter().filter(|c| c.enabled) {
                if let Some(frame) = cc.runtime.frame(&col.bone) {
                    own.push((col.name.clone(), col.world_shape(frame, world)));
                }
            }
        }
    }
    for c in components.iter_mut() {
        let Some(mc) = c.as_any_mut().downcast_mut::<zestengine_core::MeshCloth>() else { continue };
        if mc.slot != slot {
            continue;
        }
        if mc.runtime.bones.len() != skel.joints.len() {
            mc.runtime.bones = skel.joints.iter().map(|j| j.name.clone()).collect();
        }
        let collider_names: Vec<String> = own.iter().map(|(n, _)| n.clone()).collect();
        if mc.runtime.collider_names != collider_names {
            mc.runtime.collider_names = collider_names;
        }
        let shapes: Vec<ClothShapeWorld> = own
            .iter()
            .filter(|(n, _)| mc.colliders.is_empty() || mc.colliders.contains(n))
            .map(|(_, s)| *s)
            .chain(mc.collider_objects.iter().filter_map(|r| r.id()).flat_map(|id| others.get(&id).into_iter().flatten().copied()))
            .collect();
        let mut positions: Vec<Vec3> = verts.iter().map(|v| Vec3::from(v.position)).collect();
        let stale = mc.runtime.sim.as_ref().is_none_or(|s| s.vertex_count() != positions.len());
        if mc.runtime.reset_requested || stale {
            mc.runtime.reset_requested = false;
            mc.runtime.bind = bind.iter().map(|v| Vec3::from(v.position)).collect();
            mc.runtime.indices = indices.to_vec();
            mc.runtime.topology = zestengine_core::topology_hash(bind.len(), indices);
            let pins = mesh_pins(mc, skel, bind);
            mc.runtime.source_pins = if mc.pin == zestengine_core::PinSource::Painted { Vec::new() } else { pins.clone() };
            mc.runtime.sim = zestengine_core::MeshClothSim::build(&positions, indices, &pins, mc.particle_spacing, world);
            let limits = mesh_limits(mc);
            if let Some(sim) = mc.runtime.sim.as_mut() {
                sim.set_limits(limits.as_deref());
            }
            mc.runtime.pins_dirty = false;
            mc.runtime.retether = false;
        }
        let Some(mut sim) = mc.runtime.sim.take() else { continue };
        if mc.runtime.pins_dirty || mc.runtime.retether {
            if mc.pin == zestengine_core::PinSource::Painted && mc.paint.matches(sim.vertex_count(), mc.runtime.topology) {
                sim.set_pins(&mc.paint.pin_weights(), mc.runtime.retether);
                sim.set_limits(mesh_limits(mc).as_deref());
            }
            mc.runtime.pins_dirty = false;
            mc.runtime.retether = false;
        }
        mc.runtime.culled = culled(mc.cull_distance, world, viewer);
        if mc.runtime.hold || mc.runtime.culled {
            // Restart from the animated pose once it runs again.
            mc.runtime.idle = true;
        } else {
            if std::mem::take(&mut mc.runtime.idle) {
                sim.reset(&positions, world);
            }
            sim.step(mc, &positions, world, &shapes, winds, time, dt);
            let mut normals: Vec<Vec3> = verts.iter().map(|v| Vec3::from(v.normal)).collect();
            sim.apply(&mut positions, &mut normals, world);
            for ((v, p), n) in verts.iter_mut().zip(&positions).zip(&normals) {
                v.position = p.to_array();
                v.normal = n.to_array();
            }
        }
        if mc.runtime.paint_mode {
            mc.runtime.posed.clone_from(&positions);
        } else if !mc.runtime.posed.is_empty() {
            mc.runtime.posed = Vec::new();
        }
        mc.runtime.sim = Some(sim);
    }
}

/// World-space collider shapes per object, from each object's last resolved
/// bone frames, for cloth that lists other objects as collider sources.
pub(crate) fn shapes_by_object<'a>(
    objects: impl Iterator<Item = (ObjectId, &'a [Box<dyn Component>], Mat4)>,
) -> HashMap<ObjectId, Vec<ClothShapeWorld>> {
    let mut out = HashMap::new();
    for (id, components, world) in objects {
        for c in components {
            if let Some(cc) = c.as_any().downcast_ref::<ClothColliders>() {
                out.entry(id).or_insert_with(Vec::new).extend(cc.world_shapes(world, &[]));
            }
        }
    }
    out
}

/// Refresh the colliders' bone frames, run a pending auto-fit, then step every
/// Bone Cloth on the object and write its rotations into `locals`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn step(
    components: &mut [Box<dyn Component>],
    skel: &Skeleton,
    vertices: &[&[zestengine_render::Vertex]],
    locals: &mut [Mat4],
    world: Mat4,
    others: &HashMap<ObjectId, Vec<ClothShapeWorld>>,
    winds: &[zestengine_core::WindSample],
    viewer: Option<Vec3>,
    time: f32,
    dt: f32,
) {
    let n = skel.joints.len();
    if n == 0 || locals.len() < n {
        return;
    }
    let models = skel.world_matrices(locals);
    let parents: Vec<Option<usize>> = skel.joints.iter().map(|j| j.parent).collect();
    let names = || skel.joints.iter().map(|j| j.name.clone()).collect::<Vec<_>>();

    let mut own: Vec<(String, ClothShapeWorld)> = Vec::new();
    let mut collider_names: Vec<String> = Vec::new();
    for c in components.iter_mut() {
        let Some(cc) = c.as_any_mut().downcast_mut::<ClothColliders>() else { continue };
        if cc.runtime.bones.len() != n {
            cc.runtime.bones = names();
        }
        cc.runtime.frames.clear();
        cc.runtime.frames.extend(models.iter().map(|m| rigid_frame(*m)));
        if cc.runtime.fit_requested {
            cc.runtime.fit_requested = false;
            fit(cc, skel, vertices);
        }
        for col in &cc.colliders {
            collider_names.push(col.name.clone());
            if col.enabled
                && let Some(frame) = cc.runtime.frame(&col.bone)
            {
                own.push((col.name.clone(), col.world_shape(frame, world)));
            }
        }
    }

    for c in components.iter_mut() {
        let Some(cloth) = c.as_any_mut().downcast_mut::<BoneCloth>() else { continue };
        if cloth.runtime.bones.len() != n {
            cloth.runtime.bones = names();
        }
        if cloth.runtime.colliders != collider_names {
            cloth.runtime.colliders = collider_names.clone();
        }
        let shapes: Vec<ClothShapeWorld> = own
            .iter()
            .filter(|(name, _)| cloth.colliders.is_empty() || cloth.colliders.contains(name))
            .map(|(_, s)| *s)
            .chain(
                cloth
                    .collider_objects
                    .iter()
                    .filter_map(|r| r.id())
                    .flat_map(|id| others.get(&id).into_iter().flatten().copied()),
            )
            .collect();
        let stale = cloth.runtime.sim.as_ref().is_none_or(|s| !s.matches(cloth, n));
        if cloth.runtime.reset_requested || stale {
            cloth.runtime.reset_requested = false;
            cloth.runtime.sim = BoneClothSim::build(cloth, &parents, &cloth.runtime.bones, &models, world);
        }
        let Some(mut sim) = cloth.runtime.sim.take() else { continue };
        // Past the cull distance the bones keep their animated pose.
        if culled(cloth.cull_distance, world, viewer) {
            cloth.runtime.culled = true;
            cloth.runtime.sim = Some(sim);
            continue;
        }
        if std::mem::take(&mut cloth.runtime.culled) {
            sim.reset(&models, world);
        }
        sim.step_in_wind(cloth, &models, world, &shapes, winds, time, dt);
        sim.write_pose(&parents, &models, locals, world, cloth.blend);
        cloth.runtime.sim = Some(sim);
    }
}

/// The bones a fit preset covers.
fn preset_bones(cc: &ClothColliders, skel: &Skeleton) -> Vec<usize> {
    let rig = HumanoidRig::resolve(skel);
    let legs = [rig.l_upper_leg, rig.l_lower_leg, rig.r_upper_leg, rig.r_lower_leg];
    let arms = [rig.l_upper_arm, rig.l_lower_arm, rig.r_upper_arm, rig.r_lower_arm];
    let mut out: Vec<usize> = match cc.fit.preset {
        ClothFitPreset::Body => std::iter::once(rig.hips).chain(legs).chain(arms).flatten().collect(),
        ClothFitPreset::Legs => legs.into_iter().flatten().collect(),
        ClothFitPreset::Arms => arms.into_iter().flatten().collect(),
        ClothFitPreset::Selected => cc
            .fit
            .bones
            .iter()
            .filter_map(|b| skel.joints.iter().position(|j| &j.name == b))
            .collect(),
    };
    out.dedup();
    out
}

/// Fit a tapered capsule to the vertices each preset bone dominates, in the
/// bind pose, replacing earlier auto-fitted colliders.
fn fit(cc: &mut ClothColliders, skel: &Skeleton, vertices: &[&[zestengine_render::Vertex]]) {
    let bones = preset_bones(cc, skel);
    if bones.is_empty() {
        cc.runtime.fit_report = match cc.fit.preset {
            ClothFitPreset::Selected => "Pick bones to fit first".into(),
            _ => "No humanoid bones found; use Selected bones".into(),
        };
        return;
    }
    let bind: Vec<Mat4> = skel.joints.iter().map(|j| j.inverse_bind.inverse()).collect();
    let mut points: HashMap<usize, Vec<Vec3>> = HashMap::new();
    for verts in vertices {
        for v in verts.iter() {
            let (k, w) = v
                .weights
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(k, w)| (k, *w))
                .unwrap_or((0, 0.0));
            let j = v.joints[k] as usize;
            if w >= 0.4 && bones.contains(&j) {
                points.entry(j).or_default().push(Vec3::from(v.position));
            }
        }
    }
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); skel.joints.len()];
    for (j, joint) in skel.joints.iter().enumerate() {
        if let Some(p) = joint.parent {
            children[p].push(j);
        }
    }
    let coverage = cc.fit.coverage.clamp(0.5, 1.0);
    let mut fitted = Vec::new();
    let mut skipped = Vec::new();
    for &b in &bones {
        let name = skel.joints[b].name.clone();
        let Some(pts) = points.get(&b).filter(|p| p.len() >= 6) else {
            skipped.push(name);
            continue;
        };
        let origin = bind[b].w_axis.truncate();
        let centroid = pts.iter().copied().sum::<Vec3>() / pts.len() as f32;
        let toward = (centroid - origin).normalize_or_zero();
        let child = children[b]
            .iter()
            .map(|&c| (c, bind[c].w_axis.truncate()))
            .filter(|(_, p)| (*p - origin).length() > 1e-4)
            .max_by(|a, b| {
                let da = (a.1 - origin).normalize().dot(toward);
                let db = (b.1 - origin).normalize().dot(toward);
                da.total_cmp(&db)
            })
            .filter(|(_, p)| (*p - origin).normalize().dot(toward) > 0.3);
        let shape = match child {
            Some((_, end)) => fit_capsule(pts, origin, end, coverage),
            None => principal_axis(pts).and_then(|(mean, axis)| {
                let (lo, hi) = pts.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| {
                    let t = (*p - mean).dot(axis);
                    (lo.min(t), hi.max(t))
                });
                let (a, b) = (mean + axis * lo, mean + axis * hi);
                fit_capsule(pts, a, b, coverage).map(|(a, b, ra, rb)| {
                    let len = (b - a).length();
                    let dir = (b - a) / len.max(1e-6);
                    if ra + rb < len {
                        (a + dir * ra, b - dir * rb, ra, rb)
                    } else {
                        let mid = (a + b) * 0.5;
                        (mid, mid, ra, rb)
                    }
                })
            }),
        };
        let Some((start, end, ra, rb)) = shape else {
            skipped.push(name);
            continue;
        };
        let inv = rigid_frame(bind[b]).inverse();
        let ra = (ra + cc.fit.inset).max(0.005);
        let rb = (rb + cc.fit.inset).max(0.005);
        let sphere = (end - start).length() < 1e-4;
        fitted.push(ClothCollider {
            name: name.clone(),
            enabled: true,
            bone: name,
            shape: if sphere { ClothShape::Sphere } else { ClothShape::Capsule },
            center: inv.transform_point3(start).to_array(),
            end: inv.transform_point3(end).to_array(),
            radius: if sphere { ra.max(rb) } else { ra },
            end_radius: rb,
            normal: [0.0, 1.0, 0.0],
            auto: true,
        });
    }
    cc.colliders.retain(|c| !c.auto);
    let count = fitted.len();
    cc.colliders.extend(fitted);
    cc.runtime.selected = None;
    cc.runtime.fit_report = if skipped.is_empty() {
        format!("Fitted {count} colliders")
    } else {
        format!("Fitted {count} colliders; no skinned vertices for {}", skipped.join(", "))
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_assets::Joint;
    use zestengine_core::ClothFit;
    use glam::Quat;

    /// hips -> thigh -> shin -> foot down -Y, plus a skirt bone off the hips.
    fn leg() -> Skeleton {
        let joint = |name: &str, parent: Option<usize>, t: Vec3, world: Vec3| Joint {
            name: name.into(),
            parent,
            inverse_bind: Mat4::from_translation(-world),
            rest_translation: t,
            rest_rotation: Quat::IDENTITY,
            rest_scale: Vec3::ONE,
        };
        Skeleton {
            joints: vec![
                joint("Hips", None, Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 1.0, 0.0)),
                joint("LeftUpperLeg", Some(0), Vec3::new(0.1, 0.0, 0.0), Vec3::new(0.1, 1.0, 0.0)),
                joint("LeftLowerLeg", Some(1), Vec3::new(0.0, -0.45, 0.0), Vec3::new(0.1, 0.55, 0.0)),
                joint("LeftFoot", Some(2), Vec3::new(0.0, -0.45, 0.0), Vec3::new(0.1, 0.1, 0.0)),
                joint("Skirt", Some(0), Vec3::new(0.1, 0.0, 0.12), Vec3::new(0.1, 1.0, 0.12)),
                joint("SkirtEnd", Some(4), Vec3::new(0.0, -0.3, 0.0), Vec3::new(0.1, 0.7, 0.12)),
            ],
            rig: None,
            ..Default::default()
        }
    }

    /// A tapered cylinder of vertices around the thigh, all weighted to it.
    fn thigh_vertices() -> Vec<zestengine_render::Vertex> {
        let mut out = Vec::new();
        for i in 0..=12 {
            let t = i as f32 / 12.0;
            let r = 0.08 - 0.02 * t;
            for k in 0..12 {
                let a = k as f32 / 12.0 * std::f32::consts::TAU;
                out.push(zestengine_render::Vertex {
                    position: [0.1 + a.cos() * r, 1.0 - 0.45 * t, a.sin() * r],
                    joints: [1, 0, 0, 0],
                    weights: [1.0, 0.0, 0.0, 0.0],
                    ..Default::default()
                });
            }
        }
        out
    }

    /// A flat 6 x 6 quad sheet in the XZ plane skinned to one root bone, with
    /// its z = 0 edge painted pinned.
    fn painted_sheet() -> (Skeleton, Vec<zestengine_render::Vertex>, Vec<u32>, zestengine_core::MeshCloth) {
        let skel = Skeleton {
            joints: vec![Joint {
                name: "Root".into(),
                parent: None,
                inverse_bind: Mat4::IDENTITY,
                rest_translation: Vec3::ZERO,
                rest_rotation: Quat::IDENTITY,
                rest_scale: Vec3::ONE,
            }],
            rig: None,
            ..Default::default()
        };
        let n = 6u32;
        let mut verts = Vec::new();
        for z in 0..=n {
            for x in 0..=n {
                verts.push(zestengine_render::Vertex {
                    position: [x as f32 / n as f32, 0.0, z as f32 / n as f32],
                    joints: [0, 0, 0, 0],
                    weights: [1.0, 0.0, 0.0, 0.0],
                    ..Default::default()
                });
            }
        }
        let mut idx = Vec::new();
        for z in 0..n {
            for x in 0..n {
                let i = z * (n + 1) + x;
                idx.extend([i, i + n + 1, i + 1, i + 1, i + n + 1, i + n + 2]);
            }
        }
        let bind: Vec<Vec3> = verts.iter().map(|v| Vec3::from(v.position)).collect();
        let pins: Vec<f32> = bind.iter().map(|p| if p.z < 1e-3 { 1.0 } else { 0.0 }).collect();
        let mc = zestengine_core::MeshCloth {
            pin: zestengine_core::PinSource::Painted,
            paint: zestengine_core::ClothPaint::seed(&bind, &idx, &pins),
            shape_stiffness: 0.0,
            ..Default::default()
        };
        (skel, verts, idx, mc)
    }

    fn run_sheet(components: &mut [Box<dyn Component>], skel: &Skeleton, verts: &[zestengine_render::Vertex], idx: &[u32], viewer: Option<Vec3>, frames: usize) -> Vec<zestengine_render::Vertex> {
        let mut out = verts.to_vec();
        for _ in 0..frames {
            out = verts.to_vec();
            step_mesh(components, 0, skel, verts, &mut out, idx, Mat4::IDENTITY, &HashMap::new(), &[], viewer, 0.0, 1.0 / 60.0);
        }
        out
    }

    #[test]
    fn cloth_slot_selection_follows_the_bound_slot() {
        let (_, _, _, mut mc) = painted_sheet();
        mc.slot = 2;
        let comps: Vec<Box<dyn Component>> = vec![Box::new(mc)];
        assert!(has_mesh_cloth_slot(&comps, 2));
        assert!(!has_mesh_cloth_slot(&comps, 0));
        assert!(!has_mesh_cloth_slot(&[], 0));
    }

    #[test]
    fn simulated_sheet_patches_only_the_free_vertices() {
        let (skel, verts, idx, mc) = painted_sheet();
        let mut components: Vec<Box<dyn Component>> = vec![Box::new(mc)];
        let out = run_sheet(&mut components, &skel, &verts, &idx, None, 240);
        let mut patch = Vec::new();
        let n = zestengine_render::pack_cloth_patch(&verts, &out, 1e-5, &mut patch);
        assert!(n > 0, "the falling cloth produced overrides");
        let free = verts.iter().filter(|v| v.position[2] >= 1e-3).count();
        let patched_free = patch.chunks(zestengine_render::CLOTH_PATCH_FLOATS).filter(|e| verts[e[0].to_bits() as usize].position[2] >= 1e-3).count();
        assert_eq!(patched_free, free, "every free vertex is cloth-driven");
        for e in patch.chunks(zestengine_render::CLOTH_PATCH_FLOATS) {
            let a = &verts[e[0].to_bits() as usize];
            if a.position[2] < 1e-3 {
                // A pinned vertex only gets an entry for its re-lit normal; it never moves.
                assert!((Vec3::from(a.position) - Vec3::new(e[1], e[2], e[3])).length() < 1e-4);
            }
        }
    }

    #[test]
    fn painted_pins_hold_while_free_vertices_fall() {
        let (skel, verts, idx, mc) = painted_sheet();
        let mut components: Vec<Box<dyn Component>> = vec![Box::new(mc)];
        let out = run_sheet(&mut components, &skel, &verts, &idx, None, 240);
        for (a, b) in verts.iter().zip(&out) {
            if a.position[2] < 1e-3 {
                assert!((Vec3::from(a.position) - Vec3::from(b.position)).length() < 1e-4, "pinned vertex stays at its animated position");
            }
        }
        let low = out.iter().map(|v| v.position[1]).fold(f32::MAX, f32::min);
        assert!(low < -0.5, "free vertices fall: {low}");

        // Painting it all free (live, without a rebuild) drops the pinned edge too.
        let mc = components[0].as_any_mut().downcast_mut::<zestengine_core::MeshCloth>().unwrap();
        mc.paint.set_values(zestengine_core::PaintChannel::Pin, &vec![0.0; verts.len()]);
        mc.runtime.retether = true;
        let out = run_sheet(&mut components, &skel, &verts, &idx, None, 60);
        assert!(out.iter().all(|v| v.position[1] < -0.01), "nothing pinned any more");
    }

    #[test]
    fn distant_bone_cloth_keeps_the_animated_pose() {
        let skel = leg();
        let cloth = BoneCloth { roots: vec!["Skirt".into()], cull_distance: 5.0, ..Default::default() };
        let mut components: Vec<Box<dyn Component>> = vec![Box::new(cloth)];
        let others = HashMap::new();
        let run = |components: &mut [Box<dyn Component>], viewer: Vec3| {
            let mut locals = skel.rest_locals();
            for _ in 0..60 {
                locals = skel.rest_locals();
                step(components, &skel, &[], &mut locals, Mat4::from_rotation_x(1.2), &others, &[], Some(viewer), 0.0, 1.0 / 60.0);
            }
            locals
        };
        let far = run(&mut components, Vec3::new(0.0, 0.0, 40.0));
        assert_eq!(far, skel.rest_locals(), "culled: bones keep the animated pose");
        let near = run(&mut components, Vec3::ZERO);
        assert_ne!(near, skel.rest_locals(), "simulated up close");
    }

    #[test]
    fn held_or_distant_cloth_keeps_the_animated_pose() {
        let (skel, verts, idx, mut mc) = painted_sheet();
        mc.cull_distance = 10.0;
        let mut components: Vec<Box<dyn Component>> = vec![Box::new(mc)];
        let far = Some(Vec3::new(0.0, 0.0, 50.0));
        let out = run_sheet(&mut components, &skel, &verts, &idx, far, 30);
        assert!(verts.iter().zip(&out).all(|(a, b)| a.position == b.position), "culled cloth isn't simulated");
        let mc = components[0].as_any().downcast_ref::<zestengine_core::MeshCloth>().unwrap();
        assert!(mc.runtime.culled);
        let out = run_sheet(&mut components, &skel, &verts, &idx, Some(Vec3::ONE), 60);
        assert!(out.iter().any(|v| v.position[1] < -0.1), "simulates again up close");

        let mc = components[0].as_any_mut().downcast_mut::<zestengine_core::MeshCloth>().unwrap();
        mc.runtime.hold = true;
        let out = run_sheet(&mut components, &skel, &verts, &idx, None, 2);
        assert!(verts.iter().zip(&out).all(|(a, b)| a.position == b.position), "the painter's hold shows the animated pose");
    }

    #[test]
    fn auto_fit_wraps_the_thigh_and_cloth_slides_off_it() {
        let skel = leg();
        let verts = thigh_vertices();
        let colliders = ClothColliders {
            fit: ClothFit { preset: ClothFitPreset::Legs, ..Default::default() },
            ..Default::default()
        };
        let mut components: Vec<Box<dyn Component>> = vec![
            Box::new(colliders),
            Box::new(BoneCloth { roots: vec!["Skirt".into()], radius: [0.02, 0.02], ..Default::default() }),
        ];
        components[0].as_any_mut().downcast_mut::<ClothColliders>().unwrap().runtime.fit_requested = true;
        let mut locals = skel.rest_locals();
        let others = HashMap::new();
        for _ in 0..240 {
            locals = skel.rest_locals();
            step(&mut components, &skel, &[&verts], &mut locals, Mat4::IDENTITY, &others, &[], None, 0.0, 1.0 / 60.0);
        }
        let cc = components[0].as_any().downcast_ref::<ClothColliders>().unwrap();
        assert_eq!(cc.colliders.len(), 1, "{}", cc.runtime.fit_report);
        let thigh = &cc.colliders[0];
        assert_eq!(thigh.bone, "LeftUpperLeg");
        assert!(thigh.auto && thigh.shape == ClothShape::Capsule);
        assert!(thigh.radius > thigh.end_radius, "tapers toward the knee");
        assert!((thigh.radius - 0.08).abs() < 0.02, "{}", thigh.radius);

        let ClothShapeWorld::Capsule { a, b, ra, rb } = cc.world_shapes(Mat4::IDENTITY, &[])[0] else { panic!() };
        let cloth = components[1].as_any().downcast_ref::<BoneCloth>().unwrap();
        let sim = cloth.runtime.sim.as_ref().unwrap();
        for p in sim.particles.iter().filter(|p| !p.fixed) {
            let t = ((p.pos - a).dot(b - a) / (b - a).length_squared()).clamp(0.0, 1.0);
            let d = (p.pos - a.lerp(b, t)).length();
            assert!(d >= ra + (rb - ra) * t + 0.02 - 0.01, "particle inside the thigh capsule: {d}");
        }
        let skirt_end = skel.world_matrices(&locals)[5].w_axis.truncate();
        assert!(skirt_end.y < 0.75, "skirt hangs under gravity, got {skirt_end}");
    }
}
