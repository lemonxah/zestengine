//! Avatar Mask: which part of a rig an animation layer affects.
//!
//! A mask holds 13 humanoid parts (the body groups plus the four IK goals) and
//! a list of weighted transform paths. A joint is affected when its human bone
//! belongs to an active part, or when it is listed in `elements` (with its
//! descendants when the element asks for them). The default mask has every part
//! on and no elements, which means "everything".
//!
//! Masks live inline on animator layers ([`crate::AnimLayer::mask`]) or as a
//! `.mask` asset ([`load_avatar_mask`], [`save_avatar_mask`]) that a layer
//! copies from. Old controllers stored a plain list of bone names per layer;
//! [`AvatarMask`] reads that list and turns it into elements that include
//! descendants (the old behaviour), with every humanoid part off.

use crate::human_rig::HumanBone;
use anyhow::{Context, Result};
use serde::{Deserialize, Deserializer, Serialize};
use std::path::Path;

/// The 13 humanoid mask parts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MaskPart {
    Root,
    Body,
    Head,
    LeftLeg,
    RightLeg,
    LeftArm,
    RightArm,
    LeftFingers,
    RightFingers,
    LeftFootIk,
    RightFootIk,
    LeftHandIk,
    RightHandIk,
}

/// All parts in display order.
pub const MASK_PARTS: [MaskPart; 13] = [
    MaskPart::Root,
    MaskPart::Body,
    MaskPart::Head,
    MaskPart::LeftLeg,
    MaskPart::RightLeg,
    MaskPart::LeftArm,
    MaskPart::RightArm,
    MaskPart::LeftFingers,
    MaskPart::RightFingers,
    MaskPart::LeftFootIk,
    MaskPart::RightFootIk,
    MaskPart::LeftHandIk,
    MaskPart::RightHandIk,
];

impl MaskPart {
    pub fn label(self) -> &'static str {
        match self {
            Self::Root => "Root",
            Self::Body => "Body",
            Self::Head => "Head",
            Self::LeftLeg => "Left Leg",
            Self::RightLeg => "Right Leg",
            Self::LeftArm => "Left Arm",
            Self::RightArm => "Right Arm",
            Self::LeftFingers => "Left Fingers",
            Self::RightFingers => "Right Fingers",
            Self::LeftFootIk => "Left Foot IK",
            Self::RightFootIk => "Right Foot IK",
            Self::LeftHandIk => "Left Hand IK",
            Self::RightHandIk => "Right Hand IK",
        }
    }

    /// Human bones the part drives. Root and the IK goals drive no bones of their own.
    pub fn bones(self) -> Vec<HumanBone> {
        use HumanBone as B;
        use crate::human_rig::ALL;
        let fingers = |left: bool| -> Vec<HumanBone> {
            let base = if left { 24 } else { 39 };
            ALL[base..base + 15].to_vec()
        };
        match self {
            Self::Body => vec![B::Hips, B::Spine, B::Chest, B::UpperChest],
            Self::Head => vec![B::Neck, B::Head, B::LeftEye, B::RightEye, B::Jaw],
            Self::LeftLeg => vec![B::LeftUpperLeg, B::LeftLowerLeg, B::LeftFoot, B::LeftToes],
            Self::RightLeg => vec![B::RightUpperLeg, B::RightLowerLeg, B::RightFoot, B::RightToes],
            Self::LeftArm => vec![B::LeftShoulder, B::LeftUpperArm, B::LeftLowerArm, B::LeftHand],
            Self::RightArm => vec![B::RightShoulder, B::RightUpperArm, B::RightLowerArm, B::RightHand],
            Self::LeftFingers => fingers(true),
            Self::RightFingers => fingers(false),
            Self::Root | Self::LeftFootIk | Self::RightFootIk | Self::LeftHandIk | Self::RightHandIk => Vec::new(),
        }
    }
}

/// One weighted transform of a mask.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MaskElement {
    /// Joint name or object path.
    pub path: String,
    /// 0..1; above 0.5 counts as active.
    pub weight: f32,
    /// The element covers everything below the transform too.
    pub descendants: bool,
}

impl Default for MaskElement {
    fn default() -> Self {
        Self { path: String::new(), weight: 1.0, descendants: false }
    }
}

/// See the module docs.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(default)]
pub struct AvatarMask {
    /// Active humanoid parts.
    pub parts: Vec<MaskPart>,
    pub elements: Vec<MaskElement>,
}

impl Default for AvatarMask {
    fn default() -> Self {
        Self { parts: MASK_PARTS.to_vec(), elements: Vec::new() }
    }
}

/// Reads the current struct or the old list of bone names (a visitor rather
/// than an untagged enum, which RON cannot buffer for struct variants).
impl<'de> Deserialize<'de> for AvatarMask {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = AvatarMask;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("an avatar mask or a list of bone names")
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<AvatarMask, A::Error> {
                let mut names = Vec::new();
                while let Some(n) = seq.next_element::<String>()? {
                    names.push(n);
                }
                Ok(AvatarMask::from_legacy(&names))
            }

            fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> std::result::Result<AvatarMask, A::Error> {
                let mut out = AvatarMask::default();
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "parts" => out.parts = map.next_value()?,
                        "elements" => out.elements = map.next_value()?,
                        _ => {
                            map.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                Ok(out)
            }
        }
        d.deserialize_any(V)
    }
}

impl AvatarMask {
    /// The old per-layer list: each name plus its descendants, nothing humanoid.
    /// An empty list was "everything".
    pub fn from_legacy(names: &[String]) -> Self {
        if names.is_empty() {
            return Self::default();
        }
        Self {
            parts: Vec::new(),
            elements: names.iter().map(|n| MaskElement { path: n.clone(), weight: 1.0, descendants: true }).collect(),
        }
    }

    /// Every part on and no listed transforms: the mask restricts nothing.
    pub fn is_everything(&self) -> bool {
        self.elements.is_empty() && MASK_PARTS.iter().all(|p| self.parts.contains(p))
    }

    pub fn has_part(&self, p: MaskPart) -> bool {
        self.parts.contains(&p)
    }

    pub fn set_part(&mut self, p: MaskPart, on: bool) {
        self.parts.retain(|x| *x != p);
        if on {
            self.parts.push(p);
            self.parts.sort_by_key(|x| MASK_PARTS.iter().position(|y| y == x));
        }
    }

    /// The element for `path`, if listed.
    pub fn element(&self, path: &str) -> Option<&MaskElement> {
        self.elements.iter().find(|e| e.path == path)
    }

    /// Set (adding when needed) a transform's weight.
    pub fn set_element(&mut self, path: &str, weight: f32, descendants: bool) {
        match self.elements.iter_mut().find(|e| e.path == path) {
            Some(e) => {
                e.weight = weight;
                e.descendants = descendants;
            }
            None => self.elements.push(MaskElement { path: path.into(), weight, descendants }),
        }
    }

    pub fn remove_element(&mut self, path: &str) {
        self.elements.retain(|e| e.path != path);
    }

    /// Whether `path` is listed with an active weight (above 0.5).
    pub fn is_active_element(&self, path: &str) -> bool {
        self.element(path).is_some_and(|e| e.weight > 0.5)
    }

    /// The transform tree toggle: switch node `node` on (weight 1, only that
    /// transform) or off. With `propagate` (Alt in the editor) everything
    /// below it follows. `names`/`parents` describe the skeleton the tree is
    /// drawn from, which need not be the rig the mask is used on.
    pub fn set_transform(&mut self, names: &[String], parents: &[Option<usize>], node: usize, on: bool, propagate: bool) {
        let mut apply = |i: usize| match (on, names.get(i)) {
            (true, Some(n)) => self.set_element(n, 1.0, false),
            (false, Some(n)) => self.remove_element(n),
            _ => {}
        };
        apply(node);
        if propagate {
            for i in 0..parents.len() {
                if i != node && is_under(parents, i, node) {
                    apply(i);
                }
            }
        }
    }

    /// Per-joint weight of the mask over a rig of `parents.len()` nodes.
    /// `lookup` finds a node by element path, `human` finds the node of a human
    /// bone. `None` = the mask restricts nothing.
    pub fn weights(
        &self,
        parents: &[Option<usize>],
        lookup: &dyn Fn(&str) -> Option<usize>,
        human: &dyn Fn(HumanBone) -> Option<usize>,
    ) -> Option<Vec<f32>> {
        if self.is_everything() {
            return None;
        }
        let mut out = vec![0.0f32; parents.len()];
        for part in &self.parts {
            for b in part.bones() {
                if let Some(j) = human(b) {
                    out[j] = 1.0;
                }
            }
        }
        for e in &self.elements {
            let Some(root) = lookup(&e.path) else { continue };
            for (i, slot) in out.iter_mut().enumerate() {
                let hit = i == root || (e.descendants && is_under(parents, i, root));
                if hit {
                    *slot = slot.max(e.weight.clamp(0.0, 1.0));
                }
            }
        }
        Some(out)
    }

    /// [`Self::weights`] thresholded at 0.5 (a node is affected or not).
    pub fn active(
        &self,
        parents: &[Option<usize>],
        lookup: &dyn Fn(&str) -> Option<usize>,
        human: &dyn Fn(HumanBone) -> Option<usize>,
    ) -> Option<Vec<bool>> {
        self.weights(parents, lookup, human).map(|w| w.into_iter().map(|x| x > 0.5).collect())
    }
}

/// Depth-first order of a parent table as `(node, depth)`: roots first,
/// children in index order. Nodes in a cycle are left out.
pub fn tree_order(parents: &[Option<usize>]) -> Vec<(usize, usize)> {
    let n = parents.len();
    let mut kids: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut roots = Vec::new();
    for (i, p) in parents.iter().enumerate() {
        match p.filter(|&p| p < n && p != i) {
            Some(p) => kids[p].push(i),
            None => roots.push(i),
        }
    }
    let mut out = Vec::with_capacity(n);
    let mut stack: Vec<(usize, usize)> = roots.into_iter().rev().map(|r| (r, 0)).collect();
    while let Some((i, d)) = stack.pop() {
        out.push((i, d));
        stack.extend(kids[i].iter().rev().map(|&k| (k, d + 1)));
    }
    out
}

fn is_under(parents: &[Option<usize>], mut i: usize, root: usize) -> bool {
    let mut guard = 0;
    while let Some(Some(p)) = parents.get(i) {
        if *p == root {
            return true;
        }
        i = *p;
        guard += 1;
        if guard > 256 {
            break;
        }
    }
    false
}

/// Read a `.mask` asset.
pub fn load_avatar_mask(path: &Path) -> Result<AvatarMask> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    ron::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

/// Write a `.mask` asset.
pub fn save_avatar_mask(path: &Path, mask: &AvatarMask) -> Result<()> {
    let text = ron::ser::to_string_pretty(mask, ron::ser::PrettyConfig::default())?;
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 0 root; 1 hips; 2 spine; 3 L upper arm; 4 L hand; 5 prop (child of hand); 6 R upper arm.
    fn rig() -> (Vec<Option<usize>>, Vec<&'static str>) {
        (vec![None, Some(0), Some(1), Some(2), Some(3), Some(4), Some(2)], vec!["root", "hips", "spine", "lua", "lh", "prop", "rua"])
    }

    fn resolve(m: &AvatarMask) -> Option<Vec<f32>> {
        let (parents, names) = rig();
        let lookup = |p: &str| names.iter().position(|n| *n == p);
        let human = |b: HumanBone| match b {
            HumanBone::Hips => Some(1),
            HumanBone::Spine => Some(2),
            HumanBone::LeftUpperArm => Some(3),
            HumanBone::LeftHand => Some(4),
            HumanBone::RightUpperArm => Some(6),
            _ => None,
        };
        m.weights(&parents, &lookup, &human)
    }

    #[test]
    fn default_mask_is_everything() {
        assert!(AvatarMask::default().is_everything());
        assert_eq!(resolve(&AvatarMask::default()), None);
        assert_eq!(MASK_PARTS.len(), 13);
    }

    #[test]
    fn parts_resolve_through_the_human_map() {
        let mut m = AvatarMask { parts: Vec::new(), elements: Vec::new() };
        m.set_part(MaskPart::LeftArm, true);
        assert_eq!(resolve(&m).unwrap(), vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
        m.set_part(MaskPart::Body, true);
        assert_eq!(resolve(&m).unwrap(), vec![0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]);
        m.set_part(MaskPart::Body, false);
        assert!(!m.has_part(MaskPart::Body));
    }

    #[test]
    fn elements_are_explicit_unless_they_ask_for_descendants() {
        let mut m = AvatarMask { parts: Vec::new(), elements: Vec::new() };
        m.set_element("lh", 1.0, false);
        assert_eq!(resolve(&m).unwrap()[5], 0.0, "the prop under the hand is not implied");
        m.set_element("lh", 0.75, true);
        let w = resolve(&m).unwrap();
        assert_eq!((w[4], w[5], w[3]), (0.75, 0.75, 0.0));
        let (parents, names) = rig();
        let active = m.active(&parents, &|p| names.iter().position(|n| *n == p), &|_| None).unwrap();
        assert_eq!(active, vec![false, false, false, false, true, true, false]);
        m.remove_element("lh");
        assert!(m.element("lh").is_none());
    }

    #[test]
    fn transform_tree_toggles_with_alt_propagation() {
        let (parents, names) = rig();
        let names: Vec<String> = names.into_iter().map(String::from).collect();
        let mut m = AvatarMask { parts: Vec::new(), elements: Vec::new() };
        // Plain click: that transform only.
        m.set_transform(&names, &parents, 3, true, false);
        assert!(m.is_active_element("lua") && !m.is_active_element("lh"));
        // Alt: the subtree (arm, hand, prop) follows; the sibling arm does not.
        m.set_transform(&names, &parents, 3, true, true);
        assert!(["lua", "lh", "prop"].iter().all(|n| m.is_active_element(n)));
        assert!(!m.is_active_element("rua") && !m.is_active_element("spine"));
        assert!(m.elements.iter().all(|e| !e.descendants), "each transform is listed on its own");
        let w = resolve(&m).unwrap();
        assert_eq!((w[3], w[4], w[5], w[6]), (1.0, 1.0, 1.0, 0.0));
        // Alt off removes the subtree again.
        m.set_transform(&names, &parents, 3, false, true);
        assert!(m.elements.is_empty());
    }

    #[test]
    fn tree_order_is_depth_first_with_depths() {
        let (parents, _) = rig();
        assert_eq!(tree_order(&parents), vec![(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 3)]);
        // Several roots and a cycle: cycle members are dropped.
        assert_eq!(tree_order(&[None, None, Some(3), Some(2)]), vec![(0, 0), (1, 0)]);
        assert!(tree_order(&[]).is_empty());
    }

    #[test]
    fn old_name_lists_load_as_descendant_elements() {
        let m: AvatarMask = ron::from_str("[\"lh\", \"rua\"]").unwrap();
        assert!(m.parts.is_empty());
        assert_eq!(m.elements.len(), 2);
        let w = resolve(&m).unwrap();
        assert_eq!((w[4], w[5], w[6], w[1]), (1.0, 1.0, 1.0, 0.0), "same nodes the old mask covered");
        let empty: AvatarMask = ron::from_str("[]").unwrap();
        assert!(empty.is_everything(), "an empty list was everything");
        let new: AvatarMask = ron::from_str("(parts: [Head], elements: [(path: \"x\")])").unwrap();
        assert_eq!(new.parts, vec![MaskPart::Head]);
        assert_eq!(new.elements[0].weight, 1.0);
        let round: AvatarMask = ron::from_str(&ron::to_string(&m).unwrap()).unwrap();
        assert_eq!(round, m);
    }

    #[test]
    fn mask_asset_round_trips() {
        let mut m = AvatarMask::default();
        m.set_part(MaskPart::LeftFingers, false);
        m.set_element("spine", 0.5, true);
        let dir = std::env::temp_dir().join(format!("zest_mask_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("upper.mask");
        save_avatar_mask(&p, &m).unwrap();
        assert_eq!(load_avatar_mask(&p).unwrap(), m);
        std::fs::remove_dir_all(&dir).ok();
    }
}
