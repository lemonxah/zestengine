//! Spawns a PSD character as scene objects (2D-7): the prefab builder behind
//! "Create Prefab / Instance from Layers". It reads the `.layers` tree the PSD importer wrote and,
//! when present, the `.rig2d` seed next to it, and builds the object list of a prefab:
//!
//! * a root object named after the file, at the canvas centre (the origin of the layer positions
//!   and of the rig, so the character stands where it was drawn);
//! * one object per layer-tree node under its parent: groups are empty transform nodes, sprite
//!   layers carry a `SpriteRenderer` showing the layer's sprite, whose `order_in_layer` is the PSD
//!   stack order (higher PSD layers draw on top), at the layer's position;
//! * with a rig: a `Skeleton` object holding one object per bone (parented like the skeleton, each
//!   with a `SpriteBone` carrying the bone's stable id, local position and rotation from the
//!   `.rig2d`), and a `SpriteSkin` on every sprite layer the rig has a mesh for, naming the rig,
//!   the bone objects by skeleton index and the character root (so one skeleton drives sprites that
//!   sit at different places).
//!
//! Pure data (no GPU, no engine): the engine writes the `Prefab` file and the editor links an
//! instance. Object ids are fresh UUIDs; references between the objects (the skins' bone lists)
//! are rewritten by the prefab instantiation like any other.

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use zestengine_core::{ObjectId, ObjectRef, SpriteBone, SpriteRenderer, SpriteSkin};

use crate::layered_import::LayerTree;
use crate::rig2d::{Rig2D, load_rig2d};
use crate::scene_file::{ComponentData, MaterialRef, ObjectSource, SceneEntry};
use crate::sprite::SpriteRef;

/// Name of the object that holds the bone objects.
pub const SKELETON_OBJECT: &str = "Skeleton";

/// What the builder made, besides the entries themselves.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LayersPrefabStats {
    pub sprites: usize,
    pub groups: usize,
    pub bones: usize,
    pub skinned: usize,
}

fn entry(name: &str, parent: Option<usize>, translation: [f32; 3], rotation_z: f32) -> SceneEntry {
    let (s, c) = (rotation_z * 0.5).sin_cos();
    SceneEntry {
        id: ObjectId::new().to_string(),
        name: name.to_owned(),
        source: ObjectSource::Empty,
        enabled: true,
        static_geometry: false,
        lightmap_scale: 1.0,
        cast_shadows: zestengine_core::ShadowCasting::On,
        layer: 0,
        material: MaterialRef::Model,
        extra_materials: Vec::new(),
        renderer_split: false,
        parent,
        components: Vec::new(),
        translation,
        rotation: [0.0, 0.0, s, c],
        scale: [1.0; 3],
        model: None,
        model_part: None,
        prefab: None,
        prefab_parent: None,
        net_role: Default::default(),
        component_roles: Default::default(),
    }
}

/// Build the prefab's entries (parent indices are local to the list) from a layer tree and an
/// optional rig. `project_root` / `rig_path` make the rig reference project-relative.
pub fn build_layers_prefab(name: &str, tree: &LayerTree, rig: Option<(&Rig2D, &str)>) -> (Vec<SceneEntry>, LayersPrefabStats) {
    let mut stats = LayersPrefabStats::default();
    let mut out: Vec<SceneEntry> = Vec::with_capacity(tree.nodes.len() + 8);
    out.push(entry(name, None, [0.0; 3], 0.0));
    let root_id = out[0].id.parse::<ObjectId>().ok();
    // Layer nodes: parents come before children in the importer's order only by luck, so map by index.
    let mut node_entry: Vec<Option<usize>> = vec![None; tree.nodes.len()];
    let mut pending: Vec<usize> = (0..tree.nodes.len()).collect();
    let mut guard = 0;
    while !pending.is_empty() && guard <= tree.nodes.len() {
        guard += 1;
        let mut later = Vec::new();
        for i in pending {
            let n = &tree.nodes[i];
            let parent = match n.parent {
                None => Some(0),
                Some(p) => match node_entry.get(p).copied().flatten() {
                    Some(e) => Some(e),
                    None if p >= tree.nodes.len() || p == i => Some(0), // a dangling or self parent: the root
                    None => {
                        later.push(i);
                        continue;
                    }
                },
            };
            let is_sprite = !n.sprite.is_empty();
            let mut e = entry(&n.name, parent, if is_sprite { [n.position[0], n.position[1], 0.0] } else { [0.0; 3] }, 0.0);
            if is_sprite {
                let sr = SpriteRenderer { sprite: n.sprite.clone(), order_in_layer: n.sorting_order.clamp(i16::MIN as i32, i16::MAX as i32) as i16, ..Default::default() };
                e.components.push(ComponentData::of(&sr));
                stats.sprites += 1;
            } else {
                stats.groups += 1;
            }
            node_entry[i] = Some(out.len());
            out.push(e);
        }
        pending = later;
    }
    // Nodes in a parent cycle never resolved: hang them from the root rather than lose them.
    for i in pending {
        let n = &tree.nodes[i];
        node_entry[i] = Some(out.len());
        out.push(entry(&n.name, Some(0), [n.position[0], n.position[1], 0.0], 0.0));
    }

    // The skeleton.
    if let Some((rig, rig_path)) = rig.filter(|(r, _)| !r.skeleton.bones.is_empty()) {
        let mut skel = rig.skeleton.clone();
        skel.ensure_ids();
        let holder = out.len();
        out.push(entry(SKELETON_OBJECT, Some(0), [0.0; 3], 0.0));
        let mut bone_entry: Vec<Option<usize>> = vec![None; skel.bones.len()];
        let mut pending: Vec<usize> = (0..skel.bones.len()).collect();
        let mut guard = 0;
        while !pending.is_empty() && guard <= skel.bones.len() {
            guard += 1;
            let mut later = Vec::new();
            for i in pending {
                let b = &skel.bones[i];
                let parent = if b.parent < 0 || b.parent as usize >= skel.bones.len() || b.parent as usize == i { Some(holder) } else { bone_entry[b.parent as usize] };
                let Some(parent) = parent else {
                    later.push(i);
                    continue;
                };
                let mut e = entry(&b.name, Some(parent), [b.position[0], b.position[1], 0.0], b.rotation);
                e.components.push(ComponentData::of(&SpriteBone { bone_id: b.id, ..Default::default() }));
                bone_entry[i] = Some(out.len());
                out.push(e);
                stats.bones += 1;
            }
            pending = later;
        }
        for i in pending {
            // A parent cycle: under the holder.
            bone_entry[i] = Some(out.len());
            let mut e = entry(&skel.bones[i].name, Some(holder), [0.0; 3], 0.0);
            e.components.push(ComponentData::of(&SpriteBone { bone_id: skel.bones[i].id, ..Default::default() }));
            out.push(e);
            stats.bones += 1;
        }
        let bones: Vec<ObjectRef> = bone_entry.iter().map(|e| e.and_then(|k| out[k].id.parse::<ObjectId>().ok()).map_or(ObjectRef::NONE, ObjectRef::new)).collect();
        let root_ref = root_id.map_or(ObjectRef::NONE, ObjectRef::new);
        // Skin every sprite layer the rig has a mesh for.
        for (i, n) in tree.nodes.iter().enumerate() {
            let Some(r) = SpriteRef::from_ref_string(&n.sprite) else { continue };
            if rig.part(r.sprite.0).is_none() {
                continue;
            }
            let Some(k) = node_entry[i] else { continue };
            let skin = SpriteSkin { rig: rig_path.to_owned(), bones: bones.clone(), root: root_ref, ..Default::default() };
            out[k].components.push(ComponentData::of(&skin));
            stats.skinned += 1;
        }
    }
    (out, stats)
}

/// `<stem>.layers` -> (the tree, the `.rig2d` next to it when there is one, project-relative).
pub fn load_layers_and_rig(project_root: &Path, layers: &Path) -> Result<(LayerTree, Option<(Rig2D, String)>)> {
    let text = std::fs::read_to_string(layers).with_context(|| format!("read {}", layers.display()))?;
    let tree: LayerTree = ron::from_str(&text).with_context(|| format!("parse {}", layers.display()))?;
    let rig_path = layers.with_extension("rig2d");
    let rig = if rig_path.is_file() {
        let rel = rig_path.strip_prefix(project_root).unwrap_or(&rig_path).to_string_lossy().replace('\\', "/");
        Some((load_rig2d(&rig_path)?, rel))
    } else {
        None
    };
    Ok((tree, rig))
}

/// Entries for the PSD whose layer tree is `layers` (a `.layers` path; the object is named after it).
pub fn layers_prefab_entries(project_root: &Path, layers: &Path) -> Result<(Vec<SceneEntry>, LayersPrefabStats)> {
    let (tree, rig) = load_layers_and_rig(project_root, layers)?;
    let name = layers.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "Character".into());
    Ok(build_layers_prefab(&name, &tree, rig.as_ref().map(|(r, p)| (r, p.as_str()))))
}

/// Where the prefab for `layers` goes: `<stem>.prefab` next to it.
pub fn layers_prefab_path(layers: &Path) -> PathBuf {
    layers.with_extension("prefab")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layered_import::{LayerTreeNode, import_psd_file};
    use crate::psd::{self, synth as ps};

    fn comp<T: 'static + Clone>(e: &SceneEntry, reg: &zestengine_core::ComponentRegistry) -> Option<T> {
        e.components.iter().find_map(|c| c.load(reg).and_then(|b| b.as_any().downcast_ref::<T>().cloned()))
    }

    fn registry() -> zestengine_core::ComponentRegistry {
        zestengine_core::ComponentRegistry::with_builtins()
    }

    #[test]
    fn a_layer_tree_becomes_objects_with_positions_and_stack_order() {
        let tree = LayerTree {
            canvas: [8, 8],
            ppu: 10.0,
            nodes: vec![
                LayerTreeNode { name: "bg".into(), parent: None, sprite: "c.png#0000000000000001".into(), position: [0.0, 0.0], sorting_order: 1 },
                LayerTreeNode { name: "Arm".into(), parent: None, sprite: String::new(), position: [0.0, 0.0], sorting_order: 3 },
                LayerTreeNode { name: "hand".into(), parent: Some(1), sprite: "c.png#0000000000000002".into(), position: [0.3, -0.1], sorting_order: 2 },
            ],
        };
        let (out, stats) = build_layers_prefab("char", &tree, None);
        assert_eq!((stats.sprites, stats.groups, stats.bones, stats.skinned), (2, 1, 0, 0));
        assert_eq!(out.len(), 4);
        assert_eq!((out[0].name.as_str(), out[0].parent), ("char", None));
        let by = |n: &str| out.iter().position(|e| e.name == n).unwrap();
        assert_eq!(out[by("Arm")].parent, Some(0));
        assert_eq!(out[by("hand")].parent, Some(by("Arm")), "the group hierarchy survives");
        assert_eq!(out[by("hand")].translation, [0.3, -0.1, 0.0]);
        let reg = registry();
        let (bg, hand) = (comp::<SpriteRenderer>(&out[by("bg")], &reg).unwrap(), comp::<SpriteRenderer>(&out[by("hand")], &reg).unwrap());
        assert_eq!(hand.sprite, "c.png#0000000000000002");
        assert!(bg.order_in_layer < hand.order_in_layer, "PSD stack order is the sorting order");
        assert!(comp::<SpriteRenderer>(&out[by("Arm")], &reg).is_none());
        // Every id is unique.
        let mut ids: Vec<&str> = out.iter().map(|e| e.id.as_str()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), out.len());
    }

    #[test]
    fn children_listed_before_their_parent_and_cycles_do_not_lose_objects() {
        let n = |name: &str, parent: Option<usize>| LayerTreeNode { name: name.into(), parent, sprite: String::new(), position: [0.0; 2], sorting_order: 0 };
        let tree = LayerTree { canvas: [1, 1], ppu: 1.0, nodes: vec![n("child", Some(1)), n("parent", None), n("a", Some(3)), n("b", Some(2)), n("self", Some(4)), n("oob", Some(99))] };
        let (out, _) = build_layers_prefab("c", &tree, None);
        assert_eq!(out.len(), 1 + 6, "every node is kept");
        let by = |n: &str| out.iter().position(|e| e.name == n).unwrap();
        assert_eq!(out[by("child")].parent, Some(by("parent")));
        assert_eq!(out[by("self")].parent, Some(0));
        assert_eq!(out[by("oob")].parent, Some(0));
        assert!(out[by("a")].parent.is_some() && out[by("b")].parent.is_some());
        for e in &out {
            assert!(e.parent.is_none_or(|p| p < out.len()));
        }
    }

    #[test]
    fn a_psd_with_a_rig_spawns_bones_skins_and_sprites() {
        let dir = std::env::temp_dir().join(format!("zest_layers_prefab_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let layers = vec![
            ps::layer("bg", 1, (0, 0, 8, 8), [9, 9, 9, 255], ps::Comp::Raw),
            ps::divider(),
            ps::layer("hand", 2, (4, 6, 6, 8), [255, 0, 0, 255], ps::Comp::Rle),
            ps::group("Arm", 3, true),
        ];
        let src = dir.join("char.psd");
        std::fs::write(&src, ps::file(8, 8, &layers, false)).unwrap();
        let files = import_psd_file(&dir, &src, &psd::ImportOptions { rig_seed: true, ppu: 10.0, ..Default::default() }).unwrap();
        let (out, stats) = layers_prefab_entries(&dir, files.layers.as_ref().unwrap()).unwrap();
        let rig = load_rig2d(files.rig.as_ref().unwrap()).unwrap();
        assert_eq!(stats.sprites, 2);
        assert_eq!(stats.bones, rig.skeleton.bones.len());
        assert_eq!(stats.skinned, rig.parts.len(), "every sprite with a rig mesh is skinned");
        let reg = registry();
        // Bones: under the Skeleton object, same hierarchy, ids and rest transforms as the rig.
        let skel = out.iter().position(|e| e.name == SKELETON_OBJECT).unwrap();
        for (i, b) in rig.skeleton.bones.iter().enumerate() {
            let e = out.iter().find(|e| e.name == b.name && comp::<SpriteBone>(e, &reg).is_some()).expect("bone object");
            let sb = comp::<SpriteBone>(e, &reg).unwrap();
            assert_ne!(sb.bone_id, 0);
            assert_eq!(e.translation[..2], b.position);
            let z = 2.0 * e.rotation[2].atan2(e.rotation[3]);
            assert!((z - b.rotation).abs() < 1e-5, "bone {i} rotation");
            let want_parent = if b.parent < 0 { skel } else { out.iter().position(|p| p.name == rig.skeleton.bones[b.parent as usize].name && comp::<SpriteBone>(p, &reg).is_some()).unwrap() };
            assert_eq!(e.parent, Some(want_parent));
        }
        // Skins: the rig path, one bone ref per skeleton bone, all pointing at real bone objects, the root set.
        let ids: Vec<ObjectId> = out.iter().filter_map(|e| e.id.parse::<ObjectId>().ok()).collect();
        let mut seen = 0;
        for e in &out {
            let Some(skin) = comp::<SpriteSkin>(e, &reg) else { continue };
            seen += 1;
            assert_eq!(skin.rig, "char.rig2d");
            assert_eq!(skin.bones.len(), rig.skeleton.bones.len());
            assert!(skin.bones.iter().all(|b| b.id().is_some_and(|id| ids.contains(&id))));
            assert_eq!(skin.root.id(), out[0].id.parse::<ObjectId>().ok());
            assert!(comp::<SpriteRenderer>(e, &reg).is_some(), "a skin sits next to its renderer");
        }
        assert_eq!(seen, stats.skinned);
        // Without the rig file there are no bones or skins.
        std::fs::remove_file(files.rig.as_ref().unwrap()).unwrap();
        let (plain, st) = layers_prefab_entries(&dir, files.layers.as_ref().unwrap()).unwrap();
        assert_eq!((st.bones, st.skinned), (0, 0));
        assert!(plain.iter().all(|e| comp::<SpriteBone>(e, &reg).is_none() && comp::<SpriteSkin>(e, &reg).is_none()));
        assert_eq!(layers_prefab_path(files.layers.as_ref().unwrap()), dir.join("char.prefab"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
