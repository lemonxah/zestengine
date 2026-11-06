//! VRM spring bones: read the secondary motion a VRM avatar ships with
//! (`VRM.secondaryAnimation` in VRM 0.x, `VRMC_springBone` in VRM 1.0) so the
//! engine can turn it into Bone Cloth and Cloth Colliders.

use std::path::Path;

use zestengine_core::{ClothCollider, ClothShape};
use serde_json::Value;

/// One spring chain group as imported.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpringChain {
    pub name: String,
    pub roots: Vec<String>,
    /// Bone Cloth settings mapped from the VRM ones.
    pub stiffness: f32,
    pub damping: f32,
    pub gravity: [f32; 3],
    pub radius: f32,
    /// Names of the colliders this chain collides with.
    pub colliders: Vec<String>,
}

/// A VRM's spring bones and their colliders.
#[derive(Clone, Debug, Default)]
pub struct SpringImport {
    pub chains: Vec<SpringChain>,
    pub colliders: Vec<ClothCollider>,
}

impl SpringImport {
    pub fn is_empty(&self) -> bool {
        self.chains.is_empty()
    }

    /// Follow a uniform import scale `s` (the model's Scale Factor): lengths
    /// (collider offsets and radii, chain radius) and the gravity acceleration
    /// scale with the model; stiffness and damping are unitless. A negative
    /// scale mirrors the model, which changes no length.
    pub fn scale(&mut self, s: f32) {
        let k = s.abs();
        for chain in &mut self.chains {
            chain.radius *= k;
            chain.gravity = chain.gravity.map(|g| g * k);
        }
        for c in &mut self.colliders {
            c.center = c.center.map(|x| x * s);
            c.end = c.end.map(|x| x * s);
            c.radius *= k;
            c.end_radius *= k;
        }
    }
}

/// The glTF JSON of a `.gltf`, `.glb` or `.vrm` file.
pub(crate) fn gltf_json(path: &Path) -> Option<Value> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.starts_with(b"glTF") {
        let len = u32::from_le_bytes(bytes.get(12..16)?.try_into().ok()?) as usize;
        if bytes.get(16..20)? != b"JSON" {
            return None;
        }
        return serde_json::from_slice(bytes.get(20..20 + len)?).ok();
    }
    serde_json::from_slice(&bytes).ok()
}

fn f(v: &Value, key: &str, default: f32) -> f32 {
    v.get(key).and_then(Value::as_f64).map(|x| x as f32).unwrap_or(default)
}

fn vec3(v: Option<&Value>) -> Option<[f32; 3]> {
    let v = v?;
    if let Some(a) = v.as_array() {
        return Some([a.first()?.as_f64()? as f32, a.get(1)?.as_f64()? as f32, a.get(2)?.as_f64()? as f32]);
    }
    Some([f(v, "x", 0.0), f(v, "y", 0.0), f(v, "z", 0.0)])
}

/// Read a model's VRM spring bones (empty for anything else).
pub fn read_springs(path: &Path) -> SpringImport {
    gltf_json(path).map(|json| parse(&json)).unwrap_or_default()
}

fn parse(json: &Value) -> SpringImport {
    let node_name = |i: Option<u64>| -> Option<String> {
        let n = json.get("nodes")?.get(i? as usize)?;
        Some(n.get("name").and_then(Value::as_str).unwrap_or("node").to_owned())
    };
    let ext = json.get("extensions");
    if let Some(spring) = ext.and_then(|e| e.get("VRMC_springBone")) {
        return parse_vrm1(spring, &node_name);
    }
    if let Some(sec) = ext.and_then(|e| e.get("VRM")).and_then(|v| v.get("secondaryAnimation")) {
        return parse_vrm0(sec, &node_name);
    }
    SpringImport::default()
}

/// VRM stiffness (0..4, a pull toward the rest pose per frame) to Bone Cloth's
/// per-1/90 s pull, drag (velocity lost per frame) to damping, gravity power
/// to an acceleration.
#[allow(clippy::too_many_arguments)]
fn chain(name: String, roots: Vec<String>, stiffness: f32, drag: f32, power: f32, dir: [f32; 3], radius: f32, colliders: Vec<String>) -> SpringChain {
    SpringChain {
        name,
        roots,
        stiffness: (stiffness * 0.08).clamp(0.0, 1.0),
        damping: (drag * 0.5).clamp(0.0, 1.0),
        gravity: [dir[0] * power * 4.0, dir[1] * power * 4.0, dir[2] * power * 4.0],
        radius: radius.max(0.0),
        colliders,
    }
}

fn parse_vrm0(sec: &Value, node_name: &dyn Fn(Option<u64>) -> Option<String>) -> SpringImport {
    let mut out = SpringImport::default();
    let mut group_names: Vec<Vec<String>> = Vec::new();
    for (g, group) in sec.get("colliderGroups").and_then(Value::as_array).into_iter().flatten().enumerate() {
        let Some(bone) = node_name(group.get("node").and_then(Value::as_u64)) else {
            group_names.push(Vec::new());
            continue;
        };
        let mut names = Vec::new();
        for (k, c) in group.get("colliders").and_then(Value::as_array).into_iter().flatten().enumerate() {
            // VRM 0.x offsets are in Unity's left-handed space: flip X.
            let o = vec3(c.get("offset")).unwrap_or([0.0; 3]);
            let name = format!("{bone} {g}.{k}");
            names.push(name.clone());
            out.colliders.push(ClothCollider {
                name,
                bone: bone.clone(),
                shape: ClothShape::Sphere,
                center: [-o[0], o[1], o[2]],
                radius: f(c, "radius", 0.05),
                ..Default::default()
            });
        }
        group_names.push(names);
    }
    for (i, group) in sec.get("boneGroups").and_then(Value::as_array).into_iter().flatten().enumerate() {
        let roots: Vec<String> = group
            .get("bones")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|b| node_name(b.as_u64()))
            .collect();
        if roots.is_empty() {
            continue;
        }
        let colliders = group
            .get("colliderGroups")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|g| group_names.get(g.as_u64()? as usize))
            .flatten()
            .cloned()
            .collect();
        let dir = vec3(group.get("gravityDir")).unwrap_or([0.0, -1.0, 0.0]);
        let name = group.get("comment").and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_owned).unwrap_or(format!("Spring {}", i + 1));
        out.chains.push(chain(
            name,
            roots,
            f(group, "stiffiness", f(group, "stiffness", 1.0)),
            f(group, "dragForce", 0.4),
            f(group, "gravityPower", 0.0),
            [-dir[0], dir[1], dir[2]],
            f(group, "hitRadius", 0.02),
            colliders,
        ));
    }
    out
}

fn parse_vrm1(spring: &Value, node_name: &dyn Fn(Option<u64>) -> Option<String>) -> SpringImport {
    let mut out = SpringImport::default();
    let mut collider_names: Vec<Option<String>> = Vec::new();
    for (k, c) in spring.get("colliders").and_then(Value::as_array).into_iter().flatten().enumerate() {
        let Some(bone) = node_name(c.get("node").and_then(Value::as_u64)) else {
            collider_names.push(None);
            continue;
        };
        let shape = c.get("shape");
        let name = format!("{bone} {k}");
        let col = if let Some(s) = shape.and_then(|s| s.get("sphere")) {
            ClothCollider {
                name: name.clone(),
                bone,
                shape: ClothShape::Sphere,
                center: vec3(s.get("offset")).unwrap_or([0.0; 3]),
                radius: f(s, "radius", 0.05),
                ..Default::default()
            }
        } else if let Some(s) = shape.and_then(|s| s.get("capsule")) {
            let r = f(s, "radius", 0.05);
            ClothCollider {
                name: name.clone(),
                bone,
                shape: ClothShape::Capsule,
                center: vec3(s.get("offset")).unwrap_or([0.0; 3]),
                end: vec3(s.get("tail")).unwrap_or([0.0; 3]),
                radius: r,
                end_radius: r,
                ..Default::default()
            }
        } else {
            collider_names.push(None);
            continue;
        };
        collider_names.push(Some(name));
        out.colliders.push(col);
    }
    let groups: Vec<Vec<String>> = spring
        .get("colliderGroups")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|g| {
            g.get("colliders")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|c| collider_names.get(c.as_u64()? as usize)?.clone())
                .collect()
        })
        .collect();
    for (i, s) in spring.get("springs").and_then(Value::as_array).into_iter().flatten().enumerate() {
        let joints: Vec<&Value> = s.get("joints").and_then(Value::as_array).into_iter().flatten().collect();
        let Some(first) = joints.first() else { continue };
        let Some(root) = node_name(first.get("node").and_then(Value::as_u64)) else { continue };
        let colliders = s
            .get("colliderGroups")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|g| groups.get(g.as_u64()? as usize))
            .flatten()
            .cloned()
            .collect();
        let name = s.get("name").and_then(Value::as_str).filter(|n| !n.is_empty()).map(str::to_owned).unwrap_or(format!("Spring {}", i + 1));
        out.chains.push(chain(
            name,
            vec![root],
            f(first, "stiffness", 1.0),
            f(first, "dragForce", 0.4),
            f(first, "gravityPower", 0.0),
            vec3(first.get("gravityDir")).unwrap_or([0.0, -1.0, 0.0]),
            f(first, "hitRadius", 0.02),
            colliders,
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_import_scale_scales_spring_lengths_but_not_stiffness() {
        let mut s = SpringImport {
            chains: vec![SpringChain { name: "Hair".into(), stiffness: 1.0, damping: 0.4, gravity: [0.0, -9.8, 0.0], radius: 0.02, ..Default::default() }],
            colliders: vec![ClothCollider { center: [0.01, 0.1, 0.0], end: [0.0, 0.3, 0.0], radius: 0.09, end_radius: 0.08, ..Default::default() }],
        };
        s.scale(100.0);
        assert!((s.chains[0].radius - 2.0).abs() < 1e-5);
        assert!((s.chains[0].gravity[1] + 980.0).abs() < 1e-3, "gravity is a length per second squared");
        assert_eq!((s.chains[0].stiffness, s.chains[0].damping), (1.0, 0.4));
        let c = &s.colliders[0];
        assert!((c.center[1] - 10.0).abs() < 1e-4 && (c.end[1] - 30.0).abs() < 1e-4);
        assert!((c.radius - 9.0).abs() < 1e-4 && (c.end_radius - 8.0).abs() < 1e-4);
        // A negative scale mirrors positions and leaves lengths positive.
        s.scale(-0.01);
        assert!((s.colliders[0].center[1] + 0.1).abs() < 1e-4);
        assert!(s.colliders[0].radius > 0.0 && s.chains[0].radius > 0.0);
    }

    #[test]
    fn reads_vrm0_and_vrm1_springs() {
        let v0: Value = serde_json::from_str(
            r#"{"nodes":[{"name":"Hips"},{"name":"HairRoot"},{"name":"Head"}],
                "extensions":{"VRM":{"secondaryAnimation":{
                  "colliderGroups":[{"node":2,"colliders":[{"offset":{"x":0.01,"y":0.1,"z":0},"radius":0.09}]}],
                  "boneGroups":[{"comment":"Hair","stiffiness":1.0,"gravityPower":0.5,"gravityDir":{"x":0,"y":-1,"z":0},
                                 "dragForce":0.4,"hitRadius":0.02,"bones":[1],"colliderGroups":[0]}]}}}}"#,
        )
        .unwrap();
        let s = parse(&v0);
        assert_eq!(s.chains.len(), 1);
        assert_eq!(s.chains[0].roots, vec!["HairRoot".to_owned()]);
        assert_eq!(s.chains[0].name, "Hair");
        assert_eq!(s.colliders.len(), 1);
        assert_eq!(s.colliders[0].bone, "Head");
        assert_eq!(s.colliders[0].center[0], -0.01, "left-handed offset flipped");
        assert_eq!(s.chains[0].colliders, vec![s.colliders[0].name.clone()]);

        let v1: Value = serde_json::from_str(
            r#"{"nodes":[{"name":"Hips"},{"name":"Skirt_F"},{"name":"UpperLeg_L"}],
                "extensions":{"VRMC_springBone":{
                  "colliders":[{"node":2,"shape":{"capsule":{"offset":[0,0,0],"radius":0.07,"tail":[0,-0.4,0]}}}],
                  "colliderGroups":[{"name":"Legs","colliders":[0]}],
                  "springs":[{"name":"Skirt","joints":[{"node":1,"hitRadius":0.03,"stiffness":0.5,"gravityPower":1,"gravityDir":[0,-1,0],"dragForce":0.5}],"colliderGroups":[0]}]}}}"#,
        )
        .unwrap();
        let s = parse(&v1);
        assert_eq!(s.chains[0].roots, vec!["Skirt_F".to_owned()]);
        assert_eq!(s.colliders[0].shape, ClothShape::Capsule);
        assert_eq!(s.colliders[0].end, [0.0, -0.4, 0.0]);
        assert!(s.chains[0].gravity[1] < 0.0);
        assert_eq!(s.chains[0].colliders.len(), 1);
    }
}
