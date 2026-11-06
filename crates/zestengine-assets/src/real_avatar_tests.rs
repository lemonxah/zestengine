//! Real-file audit of the user's avatar (skipped when the file is absent): every mesh part's world
//! bounds as the engine places them versus ufbx's own evaluated truth.

use crate::fbx_loader::matrix_to_mat4;
use glam::{Mat4, Vec3};

const AVATAR: &str = "/home/lemonxah/ZestEngineProjects/my-game-2/models/lemonjuice.fbx";

fn aabb(points: impl Iterator<Item = Vec3>) -> (Vec3, Vec3) {
    points.fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(lo, hi), p| (lo.min(p), hi.max(p)))
}

#[test]
fn audit_real_avatar() {
    let path = std::path::Path::new(AVATAR);
    if !path.exists() {
        eprintln!("AUDIT: {AVATAR} absent");
        return;
    }
    let settings = crate::load_asset_meta(path).ok().flatten().and_then(|m| m.importer.as_model().cloned()).unwrap_or_default();
    eprintln!("AUDIT settings: constraints {} skip_targets {} bake_axis {} scale {}", settings.import_constraints, settings.skip_blend_shape_targets, settings.bake_axis_conversion, settings.scale);
    let scene = crate::load_fbx_with(path, &settings).unwrap();
    let skel = scene.skeletons.first();
    eprintln!("AUDIT {} instances, {} skeletons, containers {:?}", scene.instances.len(), scene.skeletons.len(), skel.map(|s| s.containers.clone()));
    let opts = ufbx::LoadOpts {
        target_axes: ufbx::CoordinateAxes::right_handed_y_up(),
        target_unit_meters: 1.0,
        space_conversion: if settings.bake_axis_conversion { ufbx::SpaceConversion::ModifyGeometry } else { ufbx::SpaceConversion::TransformRoot },
        generate_missing_normals: true,
        load_external_files: false,
        ignore_missing_external_files: true,
        ..Default::default()
    };
    let fbx = ufbx::load_file(AVATAR, opts).unwrap();
    for i in &scene.instances {
        let (s, r, t) = i.transform.to_scale_rotation_translation();
        eprintln!("INST {:28} t {:.4?} r {:.4?} s {:.4?}", i.name, t, r, s);
    }
    let rest = skel.map(|s| s.palette(&s.rest_locals()));
    let mut k = 0usize;
    let mut body: Option<(Vec3, Vec3)> = None;
    let mut rows = Vec::new();
    let mut raw_offsets: Vec<(String, f32)> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    for node in &fbx.nodes {
        let Some(mesh) = &node.mesh else { continue };
        if mesh.num_triangles == 0 {
            continue;
        }
        let name = node.element.name.to_string();
        let skinned = mesh.skin_deformers.count > 0;
        // parent chain
        let mut chain = Vec::new();
        let mut up = node.parent.as_ref();
        while let Some(p) = up {
            if p.is_root {
                break;
            }
            chain.push(p.element.name.to_string());
            up = p.parent.as_ref();
        }
        if let (Some(skin), Some(sk)) = ((&mesh.skin_deformers).into_iter().next(), skel) {
            let bad: Vec<String> = skin
                .clusters
                .iter()
                .enumerate()
                .filter(|(c, cl)| cl.bone_node.as_ref().map(|b| b.element.name.to_string()) != sk.joints.get(*c).map(|j| j.name.clone()))
                .map(|(c, cl)| format!("{c}:{}!={}", cl.bone_node.as_ref().map_or(String::new(), |b| b.element.name.to_string()), sk.joints.get(c).map_or("-", |j| j.name.as_str())))
                .collect();
            eprintln!("CLUSTERS {name:28} {} clusters, {} mismatching skeleton joint order {:?}", skin.clusters.count, bad.len(), bad.iter().take(3).collect::<Vec<_>>());
        }
        // ufbx truth
        let g2w = matrix_to_mat4(&node.geometry_to_world);
        let truth: Vec<Vec3> = (0..mesh.num_vertices)
            .map(|i| {
                let p = mesh.vertices[i];
                let p = Vec3::new(p.x as f32, p.y as f32, p.z as f32);
                if let Some(skin) = (&mesh.skin_deformers).into_iter().next() {
                    let sv = skin.vertices[i];
                    let mut acc = Vec3::ZERO;
                    let mut ws = 0.0;
                    for w in 0..sv.num_weights as usize {
                        let sw = skin.weights[sv.weight_begin as usize + w];
                        let cl = &skin.clusters[sw.cluster_index as usize];
                        let Some(bn) = &cl.bone_node else { continue };
                        let m = matrix_to_mat4(&bn.node_to_world) * matrix_to_mat4(&cl.geometry_to_bone);
                        acc += m.transform_point3(p) * sw.weight as f32;
                        ws += sw.weight as f32;
                    }
                    if ws > 1e-6 { acc / ws } else { g2w.transform_point3(p) }
                } else {
                    g2w.transform_point3(p)
                }
            })
            .collect();
        let (tlo, thi) = aabb(truth.iter().copied());
        // engine
        let Some(inst) = scene.instances.get(k) else {
            skipped.push(name.clone());
            continue;
        };
        if inst.name != name {
            skipped.push(name.clone());
            continue;
        }
        k += 1;
        let mut pts = Vec::new();
        for slot in &inst.slots {
            for v in &scene.meshes[slot.mesh].vertices {
                let mut p = Vec3::from(v.position);
                if skinned && let Some(rest) = &rest {
                    p = zestengine_skin(p, v.joints, v.weights, rest);
                }
                pts.push(inst.transform.transform_point3(p));
            }
        }
        if pts.is_empty() {
            continue;
        }
        let raw: Vec<Vec3> = inst.slots.iter().flat_map(|sl| scene.meshes[sl.mesh].vertices.iter().map(|v| inst.transform.transform_point3(Vec3::from(v.position)))).collect();
        let (rlo, rhi) = aabb(raw.into_iter());
        eprintln!("RAW {name:28} raw-vs-truth centre offset {:.3?} (len {:.3})", ((rlo + rhi) - (tlo + thi)) * 0.5, (((rlo + rhi) - (tlo + thi)) * 0.5).length());
        let raw_off = (((rlo + rhi) - (tlo + thi)) * 0.5).length();
        let (elo, ehi) = aabb(pts.into_iter());
        let off = ((elo + ehi) - (tlo + thi)) * 0.5;
        raw_offsets.push((name.clone(), raw_off));
        rows.push((name.clone(), skinned, chain, off, (tlo, thi), (elo, ehi), mesh.skin_deformers.count, node.is_root));
        if name.to_lowercase().contains("body") && body.is_none() {
            body = Some((tlo, thi));
        }
    }
    for (name, skinned, chain, off, t, e, nd, _) in &rows {
        eprintln!(
            "AUDIT {name:30} skinned {skinned:5} deformers {nd} parents {:?} offset {:?} len {:.3} truth [{:.2?} {:.2?}] engine [{:.2?} {:.2?}]",
            chain.iter().take(3).collect::<Vec<_>>(), off, off.length(), t.0, t.1, e.0, e.1
        );
    }
    eprintln!("AUDIT body bounds {body:?}");
    eprintln!("AUDIT mesh nodes not imported as instances (skipped blendshape targets or empty): {skipped:?}");
    // Every part sits where ufbx's evaluated skin puts it, whether drawn raw (unposed) or skinned at rest,
    // and inside the body's bounds (grown by a hand's reach). The one tolerated outlier is a part with
    // more than four weights per vertex.
    assert!(!rows.is_empty());
    let (blo, bhi) = body.expect("a part named Body");
    for (name, _, _, off, (tlo, thi), (elo, ehi), ..) in &rows {
        assert!(off.length() < 0.06, "{name}: skinned bounds are {off:?} away from ufbx's");
        assert!(tlo.cmpge(blo - Vec3::splat(0.3)).all() && thi.cmple(bhi + Vec3::splat(0.3)).all(), "{name}: outside the body bounds");
        assert!(elo.cmpge(blo - Vec3::splat(0.3)).all() && ehi.cmple(bhi + Vec3::splat(0.3)).all(), "{name}: engine bounds outside the body bounds");
    }
    for (name, off) in &raw_offsets {
        assert!(*off < 0.06, "{name}: the unposed (raw) mesh is {off} m away from the skinned picture");
    }
}

fn zestengine_skin(p: Vec3, j: [u32; 4], w: [f32; 4], palette: &[Mat4]) -> Vec3 {
    crate::skeleton::skin_position(p, j, w, palette)
}
