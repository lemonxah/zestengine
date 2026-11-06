//! Collision modelled alongside a model's render meshes, named with the
//! Unreal convention: `UCX_<Mesh>` (convex hull), `UBX_<Mesh>` (box),
//! `USP_<Mesh>` (sphere) and `UCP_<Mesh>` (capsule), each optionally followed
//! by `_NN` when a mesh has several pieces. The pieces are taken out of the
//! model's instances (never drawn, baked or captured) and kept as collider
//! pieces of the mesh they are named after.

use glam::Vec3;

use crate::Scene;

/// The primitive an authored collision piece stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CollisionKind {
    Convex,
    Box,
    Sphere,
    Capsule,
}

impl CollisionKind {
    pub const ALL: [CollisionKind; 4] = [Self::Convex, Self::Box, Self::Sphere, Self::Capsule];

    /// Name prefix (with the underscore) that marks a piece of this kind.
    pub fn prefix(self) -> &'static str {
        match self {
            Self::Convex => "UCX_",
            Self::Box => "UBX_",
            Self::Sphere => "USP_",
            Self::Capsule => "UCP_",
        }
    }
}

/// One authored collision piece of a model.
#[derive(Clone, Debug)]
pub struct CollisionPiece {
    pub name: String,
    pub kind: CollisionKind,
    /// Index into [`Scene::instances`] of the render instance it belongs to;
    /// None = the model as a whole (no render mesh matched its name).
    pub target: Option<usize>,
    /// Vertex positions in the target instance's local space (model space
    /// when `target` is None).
    pub points: Vec<Vec3>,
}

/// The kind and the rest of the name (the target mesh name, maybe with a
/// `_NN` suffix) when `name` follows the collision naming convention. A
/// trailing `.NNN` duplicate suffix (Blender) is ignored.
pub fn parse_collision_name(name: &str) -> Option<(CollisionKind, &str)> {
    let kind = CollisionKind::ALL
        .into_iter()
        .find(|k| name.get(..4).is_some_and(|p| p.eq_ignore_ascii_case(k.prefix())))?;
    let rest = strip_duplicate_suffix(&name[4..]);
    (!rest.is_empty()).then_some((kind, rest))
}

/// `Cube.001` -> `Cube`.
fn strip_duplicate_suffix(name: &str) -> &str {
    match name.rsplit_once('.') {
        Some((base, n)) if !base.is_empty() && !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) => base,
        _ => name,
    }
}

/// `Cube_01` -> `Cube`.
fn strip_index_suffix(name: &str) -> Option<&str> {
    match name.rsplit_once('_') {
        Some((base, n)) if !base.is_empty() && !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) => Some(base),
        _ => None,
    }
}

/// Which of `names` (render instance names) a piece whose name parsed to
/// `rest` belongs to, case-insensitively: the whole rest first (a mesh may
/// itself end in `_01`), then the rest without its `_NN` piece suffix.
pub fn match_collision_target<'a>(rest: &str, names: impl Iterator<Item = &'a str> + Clone) -> Option<usize> {
    let candidates = std::iter::once(rest).chain(strip_index_suffix(rest));
    for want in candidates {
        let found = names
            .clone()
            .position(|n| n.eq_ignore_ascii_case(want))
            .or_else(|| names.clone().position(|n| strip_duplicate_suffix(n).eq_ignore_ascii_case(want)));
        if found.is_some() {
            return found;
        }
    }
    None
}

/// Move instances named as collision pieces out of `scene.instances` into
/// `scene.collision`, with their points in the space of the render instance
/// they belong to. Their meshes stay in `scene.meshes` (indices are stable)
/// but nothing references them for drawing.
pub fn extract_collision(scene: &mut Scene) {
    let kinds: Vec<Option<(CollisionKind, String)>> =
        scene.instances.iter().map(|i| parse_collision_name(&i.name).map(|(k, r)| (k, r.to_owned()))).collect();
    if kinds.iter().all(Option::is_none) {
        return;
    }
    let mut pieces = Vec::new();
    for (inst, kind) in std::mem::take(&mut scene.instances).into_iter().zip(kinds) {
        match kind {
            Some((kind, rest)) => pieces.push((kind, rest, inst)),
            None => scene.instances.push(inst),
        }
    }
    for (kind, rest, inst) in pieces {
        let target = match_collision_target(&rest, scene.instances.iter().map(|i| i.name.as_str()));
        let to_target = match target {
            Some(t) => scene.instances[t].transform.inverse() * inst.transform,
            None => inst.transform,
        };
        let points: Vec<Vec3> = inst
            .slots
            .iter()
            .filter_map(|s| scene.meshes.get(s.mesh))
            .flat_map(|m| m.vertices.iter().map(|v| to_target.transform_point3(Vec3::from(v.position))))
            .collect();
        if points.is_empty() {
            continue;
        }
        tracing::debug!(piece = %inst.name, ?kind, ?target, "authored collision piece");
        scene.collision.push(CollisionPiece { name: inst.name, kind, target, points });
    }
}

/// Meshes some instance draws (collision-only meshes are left out).
pub(crate) fn rendered_meshes(scene: &Scene) -> Vec<bool> {
    let mut used = vec![false; scene.meshes.len()];
    for s in scene.instances.iter().flat_map(|i| &i.slots) {
        if let Some(u) = used.get_mut(s.mesh) {
            *u = true;
        }
    }
    used
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Instance, MeshSlot};
    use zestengine_render::{MeshData, Vertex};
    use glam::Mat4;

    #[test]
    fn names_parse_by_prefix_case_insensitively() {
        assert_eq!(parse_collision_name("UCX_Wall_01"), Some((CollisionKind::Convex, "Wall_01")));
        assert_eq!(parse_collision_name("ubx_Crate"), Some((CollisionKind::Box, "Crate")));
        assert_eq!(parse_collision_name("USP_Ball.003"), Some((CollisionKind::Sphere, "Ball")));
        assert_eq!(parse_collision_name("Ucp_Pillar_2"), Some((CollisionKind::Capsule, "Pillar_2")));
        assert_eq!(parse_collision_name("UCX_"), None);
        assert_eq!(parse_collision_name("UCXWall"), None);
        assert_eq!(parse_collision_name("Wall"), None);
        assert_eq!(parse_collision_name("U"), None);
    }

    #[test]
    fn targets_match_whole_name_then_without_piece_suffix() {
        let names = ["Wall", "Door_01", "Crate.001"];
        let m = |rest| match_collision_target(rest, names.iter().copied());
        assert_eq!(m("wall"), Some(0));
        assert_eq!(m("Wall_00"), Some(0));
        assert_eq!(m("WALL_12"), Some(0));
        assert_eq!(m("Door_01"), Some(1));
        assert_eq!(m("Door_01_03"), Some(1));
        assert_eq!(m("Crate"), Some(2));
        assert_eq!(m("Roof"), None);
        assert_eq!(m("Wall_a"), None);
    }

    fn cube_mesh(half: f32) -> MeshData {
        let mut vertices = Vec::new();
        for c in 0..8 {
            let p = [if c & 1 == 0 { -half } else { half }, if c & 2 == 0 { -half } else { half }, if c & 4 == 0 { -half } else { half }];
            vertices.push(Vertex { position: p, ..Default::default() });
        }
        MeshData { vertices, indices: vec![0, 1, 2, 1, 3, 2], has_lightmap_uv: false }
    }

    #[test]
    fn collision_instances_leave_the_render_list_in_target_space() {
        let mut scene = Scene {
            meshes: vec![cube_mesh(1.0), cube_mesh(0.5), cube_mesh(0.25)],
            morphs: Vec::new(),
            springs: Default::default(),
            textures: Vec::new(),
            materials: Vec::new(),
            instances: vec![
                Instance::single("Crate", 0, 0, Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0))),
                Instance::single("UCX_Crate_00", 1, 0, Mat4::from_translation(Vec3::new(10.0, 2.0, 0.0))),
                Instance { name: "UBX_Nothing".into(), transform: Mat4::IDENTITY, slots: vec![MeshSlot { mesh: 2, material: 0 }] },
            ],
            skeletons: Vec::new(),
            animations: Vec::new(),
            collision: Vec::new(),
            identity: Default::default(),
        };
        extract_collision(&mut scene);
        assert_eq!(scene.instances.len(), 1);
        assert_eq!(scene.instances[0].name, "Crate");
        assert_eq!(scene.collision.len(), 2);
        let ucx = &scene.collision[0];
        assert_eq!((ucx.kind, ucx.target), (CollisionKind::Convex, Some(0)));
        // Placed 2 m above the crate, in the crate's own space.
        let lo = ucx.points.iter().fold(Vec3::splat(f32::MAX), |a, p| a.min(*p));
        assert!(lo.abs_diff_eq(Vec3::new(-0.5, 1.5, -0.5), 1e-5), "{lo}");
        let ubx = &scene.collision[1];
        assert_eq!((ubx.kind, ubx.target), (CollisionKind::Box, None));
        assert_eq!(rendered_meshes(&scene), vec![true, false, false]);
        assert_eq!(crate::ModelPart::of(&scene).len(), 1);
    }

    #[test]
    fn a_gltf_with_collision_nodes_loads_them_as_pieces_only() {
        let dir = std::env::temp_dir().join(format!("zestengine-ucx-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut bin: Vec<u8> = Vec::new();
        for c in 0..8 {
            for bit in [1, 2, 4] {
                bin.extend_from_slice(&(if c & bit == 0 { -0.5f32 } else { 0.5 }).to_le_bytes());
            }
        }
        let quads = [[0u32, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]];
        for q in quads {
            for i in [q[0], q[1], q[2], q[0], q[2], q[3]] {
                bin.extend_from_slice(&i.to_le_bytes());
            }
        }
        std::fs::write(dir.join("crate.bin"), &bin).unwrap();
        let gltf = r#"{
            "asset": {"version": "2.0"},
            "buffers": [{"uri": "crate.bin", "byteLength": 240}],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": 96},
                {"buffer": 0, "byteOffset": 96, "byteLength": 144}
            ],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3", "min": [-0.5, -0.5, -0.5], "max": [0.5, 0.5, 0.5]},
                {"bufferView": 1, "componentType": 5125, "count": 36, "type": "SCALAR"}
            ],
            "meshes": [{"name": "Cube", "primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
            "nodes": [
                {"name": "Crate", "mesh": 0, "translation": [5, 0, 0]},
                {"name": "UCX_Crate_00", "mesh": 0, "translation": [5, 1, 0]},
                {"name": "usp_Crate.001", "mesh": 0, "scale": [2, 2, 2]},
                {"name": "UBX_Lid", "mesh": 0}
            ],
            "scenes": [{"nodes": [0, 1, 2, 3]}],
            "scene": 0
        }"#;
        let path = dir.join("crate.gltf");
        std::fs::write(&path, gltf).unwrap();
        let scene = crate::load_model(&path).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(scene.instances.iter().map(|i| i.name.as_str()).collect::<Vec<_>>(), ["Crate"]);
        let kinds: Vec<_> = scene.collision.iter().map(|p| (p.kind, p.target)).collect();
        assert_eq!(kinds, [(CollisionKind::Convex, Some(0)), (CollisionKind::Sphere, Some(0)), (CollisionKind::Box, None)]);
        // The sphere piece sits at the model origin: 5 m behind the crate in its space.
        let (lo, hi) = scene.collision[1].points.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(a, b), p| (a.min(*p), b.max(*p)));
        assert!(((lo + hi) * 0.5).abs_diff_eq(Vec3::new(-5.0, 0.0, 0.0), 1e-4) && (hi - lo).abs_diff_eq(Vec3::splat(2.0), 1e-4));
    }
}
