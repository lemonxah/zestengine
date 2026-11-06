//! Human bone mapping files.
//!
//! `.humanmap` is ZestEngine's own mapping file (RON): a name and the list of
//! `(human bone, joint name)` pairs, only the bones that are assigned.
//!
//! [`parse_unity_ht`] reads a Unity human template (`.ht`) so existing Unity
//! mappings can be brought over. A `.ht` is Unity YAML holding a text asset
//! with a human-bone-name to transform-name table under `m_BoneTemplate`.
//! The reader is deliberately tolerant, since the exact layout was not checked
//! against a real file: it accepts any of
//!
//! * `HumanName: TransformName` map lines (`LeftUpperArm: Arm.L`),
//! * `first: HumanName` followed by `second: TransformName`,
//! * `m_HumanName: HumanName` followed by `m_BoneName: TransformName`,
//!
//! and skips everything else (YAML header, `m_Name`, ...). Human names are
//! matched with [`HumanBone::from_name`], so `Left Upper Arm`, `leftUpperArm`
//! and `LeftUpperArm` all work. Unknown names and empty values are dropped.
//!
//! [`apply_template`] resolves a template against a skeleton: the first joint
//! whose name equals the template's name, equal after stripping a `ns:`
//! namespace on either side, or equal after stripping both. Bones the
//! template does not list stay unmapped.

use crate::human_rig::HumanBone;
use crate::Skeleton;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A saved bone mapping (`.humanmap`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HumanMap {
    pub name: String,
    pub bones: Vec<(HumanBone, String)>,
}

impl HumanMap {
    pub fn new(name: impl Into<String>, bones: &[(HumanBone, String)]) -> Self {
        Self { name: name.into(), bones: bones.to_vec() }
    }
}

/// Read a `.humanmap`.
pub fn load_humanmap(path: &Path) -> Result<HumanMap> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    ron::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

/// Write a `.humanmap`.
pub fn save_humanmap(path: &Path, map: &HumanMap) -> Result<()> {
    let text = ron::ser::to_string_pretty(map, ron::ser::PrettyConfig::default())?;
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))
}

/// Parse a Unity human template (`.ht`) into a mapping (see the module docs).
pub fn parse_unity_ht(text: &str) -> HumanMap {
    let mut map = HumanMap::default();
    let mut pending: Option<HumanBone> = None;
    let push = |map: &mut HumanMap, b: HumanBone, v: &str| {
        let v = v.trim().trim_matches(|c| c == '"' || c == '\'');
        if !v.is_empty() && !map.bones.iter().any(|(o, _)| *o == b) {
            map.bones.push((b, v.to_string()));
        }
    };
    for raw in text.lines() {
        let line = raw.trim().trim_start_matches("- ").trim();
        let Some((key, value)) = line.split_once(':') else { continue };
        let (key, value) = (key.trim(), value.trim());
        match key {
            "m_Name" => map.name = value.to_string(),
            "first" | "m_HumanName" => pending = HumanBone::from_name(value),
            "second" | "m_BoneName" => {
                if let Some(b) = pending.take() {
                    push(&mut map, b, value);
                }
            }
            _ => {
                if let Some(b) = HumanBone::from_name(key) {
                    push(&mut map, b, value);
                }
            }
        }
    }
    map
}

/// Read a `.ht` from disk.
pub fn load_unity_ht(path: &Path) -> Result<HumanMap> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut m = parse_unity_ht(&text);
    if m.name.is_empty() {
        m.name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    }
    Ok(m)
}

fn strip_ns(n: &str) -> &str {
    n.rsplit(':').next().unwrap_or(n)
}

/// Template and joint names match when equal, or equal once namespaces are stripped.
pub fn name_matches(template: &str, joint: &str) -> bool {
    template == joint || strip_ns(template) == joint || template == strip_ns(joint) || strip_ns(template) == strip_ns(joint)
}

/// Resolve a template against `skel`: each listed bone takes the first matching
/// joint (an exact match wins over a namespace-stripped one). Bones not listed
/// or not found stay unmapped; a joint is used for at most one bone.
pub fn apply_template(skel: &Skeleton, map: &HumanMap) -> Vec<(HumanBone, String)> {
    let mut used: Vec<usize> = Vec::new();
    let mut out = Vec::new();
    for (bone, name) in &map.bones {
        let exact = skel.joints.iter().position(|j| j.name == *name).filter(|j| !used.contains(j));
        let found = exact.or_else(|| skel.joints.iter().enumerate().position(|(i, j)| !used.contains(&i) && name_matches(name, &j.name)));
        if let Some(i) = found {
            used.push(i);
            out.push((*bone, skel.joints[i].name.clone()));
        }
    }
    out.sort_by_key(|(b, _)| *b as usize);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Joint;
    use glam::{Mat4, Quat, Vec3};

    fn skel(names: &[&str]) -> Skeleton {
        Skeleton {
            joints: names
                .iter()
                .enumerate()
                .map(|(i, n)| Joint {
                    name: (*n).into(),
                    parent: i.checked_sub(1),
                    inverse_bind: Mat4::IDENTITY,
                    rest_translation: Vec3::ZERO,
                    rest_rotation: Quat::IDENTITY,
                    rest_scale: Vec3::ONE,
                })
                .collect(),
            rig: None,
            ..Default::default()
        }
    }

    #[test]
    fn humanmap_round_trips() {
        let m = HumanMap::new("hero", &[(HumanBone::Hips, "pelvis".into()), (HumanBone::LeftHand, "hand.L".into())]);
        let dir = std::env::temp_dir().join(format!("zest_humanmap_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("hero.humanmap");
        save_humanmap(&p, &m).unwrap();
        assert_eq!(load_humanmap(&p).unwrap(), m);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn reads_the_ht_layouts_it_documents() {
        let a = "%YAML 1.1\n--- !u!1105 &1\nHumanTemplate:\n  m_Name: Hero\n  m_BoneTemplate:\n    Hips: mixamorig:Hips\n    Left Upper Arm: LeftArm\n    Bogus Bone: x\n    Head: ''\n";
        let m = parse_unity_ht(a);
        assert_eq!(m.name, "Hero");
        assert_eq!(m.bones, vec![(HumanBone::Hips, "mixamorig:Hips".into()), (HumanBone::LeftUpperArm, "LeftArm".into())]);
        let b = "HumanTemplate:\n  m_BoneTemplate:\n  - first: RightFoot\n    second: foot_r\n  - m_HumanName: Jaw\n    m_BoneName: jaw\n";
        let m = parse_unity_ht(b);
        assert_eq!(m.bones, vec![(HumanBone::RightFoot, "foot_r".into()), (HumanBone::Jaw, "jaw".into())]);
    }

    #[test]
    fn templates_match_across_namespaces() {
        let s = skel(&["root", "mixamorig:Hips", "mixamorig:Spine", "Head"]);
        let m = HumanMap::new(
            "t",
            &[(HumanBone::Hips, "Hips".into()), (HumanBone::Spine, "mixamorig:Spine".into()), (HumanBone::Head, "other:Head".into()), (HumanBone::Neck, "Neck".into())],
        );
        let r = apply_template(&s, &m);
        assert_eq!(
            r,
            vec![(HumanBone::Hips, "mixamorig:Hips".into()), (HumanBone::Spine, "mixamorig:Spine".into()), (HumanBone::Head, "Head".into())],
            "unlisted and missing bones stay unmapped"
        );
    }
}
