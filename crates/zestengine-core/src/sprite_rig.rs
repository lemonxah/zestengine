//! 2D skeletal animation components (2D-7): authoring data only. The skinning,
//! IK and sprite-swap logic lives in `zestengine-assets` (`rig2d.rs`, `ik2d.rs`,
//! `sprite_library.rs`) and runs in the engine's sprite gather
//! (`engine/src/scene/rig2d.rs`); a scene with none of these components pays
//! nothing.
//!
//! * [`SpriteBone`] marks an object as a bone of a `.rig2d` skeleton. Bones are
//!   ordinary scene objects, so the existing animation tracks (position,
//!   rotation, scale by object path) animate them with no extra track kind.
//! * [`SpriteSkin`] sits next to a `SpriteRenderer`: it names the rig asset and
//!   the bone objects (index = skeleton bone index). The renderer's sprite id
//!   picks the skinned mesh inside the rig.
//! * [`SpriteLibraryRef`] / [`SpriteResolver`]: a sprite swap. A clip's
//!   `SpriteSwap` track writes `hash`; the engine looks the hash up in the
//!   library and changes what the sibling `SpriteRenderer` shows.
//! * [`IkSolver2D`]: a CCD / FABRIK / two-bone limb solver over the bone chain
//!   ending at `tip`, run after animation and before skinning.

use serde::{Deserialize, Serialize};

use crate::{ObjectRef, TypedComponent};

/// Mask of the sprite-swap hash: 24 bits, exactly representable as an `f32`
/// clip key.
pub const SPRITE_SWAP_HASH_MASK: u32 = 0x00FF_FFFF;

/// Hash of a library `category` + `label` (FNV-1a, 24 bits, never 0). The
/// assets crate's `sprite_library::entry_hash` is this function.
pub fn sprite_swap_hash(category: &str, label: &str) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for b in category.bytes().chain(std::iter::once(0)).chain(label.bytes()) {
        h = (h ^ b as u32).wrapping_mul(0x0100_0193);
    }
    let h = (h ^ (h >> 24)) & SPRITE_SWAP_HASH_MASK;
    if h == 0 { 1 } else { h }
}

/// A bone of a sprite rig.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SpriteBone {
    /// The skeleton bone's stable id (`Bone2D::id`); 0 = match by name.
    pub bone_id: u64,
    /// IK: clamp this bone's rotation relative to its parent.
    pub limit_rotation: bool,
    /// Degrees.
    pub min_angle: f32,
    /// Maximum rotation relative to the parent (degrees).
    pub max_angle: f32,
}

impl Default for SpriteBone {
    fn default() -> Self {
        Self { bone_id: 0, limit_rotation: false, min_angle: -90.0, max_angle: 90.0 }
    }
}

impl TypedComponent for SpriteBone {
    const NAME: &'static str = "Sprite Bone";
}

/// Skins the `SpriteRenderer` on the same object with a rig's weighted mesh.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SpriteSkin {
    /// Project-relative `.rig2d` path.
    pub rig: String,
    /// Bone objects, by skeleton bone index. Entries that are unset or missing
    /// are rebound automatically when `auto_rebind` is on.
    pub bones: Vec<ObjectRef>,
    /// The object whose descendants carry the [`SpriteBone`]s; unset = this object.
    pub root: ObjectRef,
    /// Rebind `bones` by bone id (then name) among the root's descendants
    /// whenever one is unresolved.
    pub auto_rebind: bool,
    /// Skin even when the sprite is not drawn (offscreen / culled).
    pub always_update: bool,
}

impl Default for SpriteSkin {
    fn default() -> Self {
        Self { rig: String::new(), bones: Vec::new(), root: ObjectRef::NONE, auto_rebind: true, always_update: false }
    }
}

impl TypedComponent for SpriteSkin {
    const NAME: &'static str = "Sprite Skin";
}

/// The sprite library an object (and the resolvers below it) swap from.
#[derive(Clone, PartialEq, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SpriteLibraryRef {
    /// Project-relative `.spritelib` path.
    pub library: String,
}

impl TypedComponent for SpriteLibraryRef {
    const NAME: &'static str = "Sprite Library";
}

/// Shows the library entry `hash` on the `SpriteRenderer` of this object. The
/// library is the nearest [`SpriteLibraryRef`] on this object or an ancestor.
/// `hash` is what `SpriteSwap` animation tracks write.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct SpriteResolver {
    /// Library category (e.g. "Head").
    pub category: String,
    /// Library entry label (e.g. "Happy").
    pub label: String,
    /// 24-bit swap hash; 0 = none yet. [`Self::set_entry`] keeps it in step
    /// with `category` / `label`; an animation overrides it directly.
    pub hash: u32,
}


impl SpriteResolver {
    /// Set the category and label, recomputing the 24-bit swap hash.
    pub fn set_entry(&mut self, category: &str, label: &str) {
        self.category = category.to_owned();
        self.label = label.to_owned();
        self.hash = sprite_swap_hash(category, label);
    }
}

impl TypedComponent for SpriteResolver {
    const NAME: &'static str = "Sprite Resolver";
}

/// Which IK algorithm solves the chain.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum IkSolverKind {
    /// Cyclic coordinate descent.
    #[default]
    Ccd,
    /// Forward and backward reaching.
    Fabrik,
    /// Closed-form two-bone limb.
    Limb,
}

impl IkSolverKind {
    /// Every solver, in declaration order.
    pub const ALL: [IkSolverKind; 3] = [Self::Ccd, Self::Fabrik, Self::Limb];
    /// Inspector label of this solver.
    pub fn label(self) -> &'static str {
        match self {
            Self::Ccd => "CCD",
            Self::Fabrik => "FABRIK",
            Self::Limb => "Limb (two bones)",
        }
    }
}

/// 2D inverse kinematics for the bone chain that ends at `tip` and runs up
/// `chain_length` bones through the parents.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct IkSolver2D {
    /// Which IK algorithm solves the chain.
    pub solver: IkSolverKind,
    /// The last bone of the chain (its end is the effector).
    pub tip: ObjectRef,
    /// Bones in the chain, counted up from `tip` (Limb uses 2).
    pub chain_length: u32,
    /// The target the tip reaches toward.
    pub target: ObjectRef,
    /// 0 = animation only, 1 = fully solved.
    pub weight: f32,
    /// Solver iterations.
    pub iterations: u32,
    /// Solve until the effector is within this distance of the target.
    pub tolerance: f32,
    /// Limb: bend the other way.
    pub flip: bool,
}

impl Default for IkSolver2D {
    fn default() -> Self {
        Self { solver: IkSolverKind::Ccd, tip: ObjectRef::NONE, chain_length: 2, target: ObjectRef::NONE, weight: 1.0, iterations: 10, tolerance: 0.001, flip: false }
    }
}

impl TypedComponent for IkSolver2D {
    const NAME: &'static str = "IK Solver 2D";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swap_hash_is_24_bit_stable_and_nonzero() {
        let h = sprite_swap_hash("Head", "Happy");
        assert!(h != 0 && h <= SPRITE_SWAP_HASH_MASK);
        assert_eq!(h as f32 as u32, h);
        assert_eq!(h, sprite_swap_hash("Head", "Happy"));
        assert_ne!(h, sprite_swap_hash("Head", "Sad"));
        let mut r = SpriteResolver::default();
        r.set_entry("Head", "Happy");
        assert_eq!(r.hash, h);
    }

    #[test]
    fn old_files_load_with_defaults() {
        let s: SpriteSkin = ron::from_str("()").unwrap();
        assert!(s.auto_rebind && s.bones.is_empty() && !s.always_update);
        let b: SpriteBone = ron::from_str("(bone_id: 5)").unwrap();
        assert_eq!((b.bone_id, b.limit_rotation, b.min_angle, b.max_angle), (5, false, -90.0, 90.0));
        let i: IkSolver2D = ron::from_str("(solver: Limb)").unwrap();
        assert_eq!((i.solver, i.chain_length, i.weight, i.iterations), (IkSolverKind::Limb, 2, 1.0, 10));
        let r: SpriteResolver = ron::from_str("(hash: 9)").unwrap();
        assert_eq!(r.hash, 9);
        let l: SpriteLibraryRef = ron::from_str("()").unwrap();
        assert!(l.library.is_empty());
    }

    #[test]
    fn components_round_trip() {
        let skin = SpriteSkin { rig: "rigs/hero.rig2d".into(), bones: vec![ObjectRef::NONE, ObjectRef::new(crate::ObjectId::new())], ..Default::default() };
        let text = ron::to_string(&skin).unwrap();
        assert_eq!(ron::from_str::<SpriteSkin>(&text).unwrap(), skin);
        let ik = IkSolver2D { solver: IkSolverKind::Fabrik, flip: true, ..Default::default() };
        assert_eq!(ron::from_str::<IkSolver2D>(&ron::to_string(&ik).unwrap()).unwrap(), ik);
    }
}
