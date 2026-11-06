//! GameObject tiles at the scene level (2D-5): keeps one instance of a tile's prefab in every cell of
//! a tilemap that uses the tile. The data model and the pure matching are in
//! `zestengine_core::tile_object`; this is the part that touches the scene.
//!
//! [`LoadedScene::sync_tile_objects`] runs wherever the scene is live: the editor frame (so painting a
//! cell shows its object at once, as editor-time instantiation), Play, and the dedicated server's tick
//! (play-time spawn, with no renderer). Per tilemap it compares the cells' tiles with the
//! `TileObject` children it already has and spawns the missing instances (as linked prefab
//! instances, so edits to the prefab reach them), deletes the instances of erased or retiled cells,
//! and leaves the rest alone, so a hand-tweaked instance survives until its cell changes. It costs one
//! serial compare per tilemap per call while nothing changed; the scan that finds the maps is a
//! component-type check per object.
//!
//! Undo needs nothing of its own: a stroke's undo restores the tilemap (and, for structural
//! snapshots, the objects), and the next sync makes the instances match whatever came back.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use glam::{IVec2, Vec3};
use zestengine_assets::ComponentData;
use zestengine_core::{ComponentRegistry, HaveObject, ObjectId, TileObject, Tilemap, TilemapOrientation, WantObject, component_inner, plan_tile_objects};

use super::LoadedScene;
use crate::shaders::ShaderLibrary;
use crate::tile_db::{LoadedTile, TileDb, grid_for};
use zestengine_render::Renderer;

/// Sync state of a scene (default = nothing synced yet).
#[derive(Default)]
pub struct TileObjectSync {
    db: Option<TileDb>,
    /// `(tilemap serial, tile db revision)` each map was last synced at.
    synced: HashMap<ObjectId, (u64, u64)>,
    calls: u32,
    warned: HashSet<String>,
}

impl TileObjectSync {
    /// Forget everything (a scene reload, a project switch).
    #[allow(dead_code)] // reset path kept for scene reloads
    pub fn reset(&mut self) {
        self.db = None;
        self.synced.clear();
    }
}

/// The prefab of a loaded tile, if it is a plain tile that names one.
fn prefab_of(t: &LoadedTile) -> Option<&str> {
    match t {
        LoadedTile::Simple(a) if !a.game_object.is_empty() => Some(a.game_object.as_str()),
        _ => None,
    }
}

impl LoadedScene {
    /// Make the tile objects of every tilemap match its cells. Returns how many objects it spawned or
    /// removed. A tile that cannot be read, or a prefab that cannot be loaded, changes nothing for
    /// its cells (existing instances stay) and is reported once.
    pub fn sync_tile_objects(&mut self, mut renderer: Option<&mut Renderer>, shaders: &mut ShaderLibrary, project_root: &Path, registry: &ComponentRegistry) -> usize {
        let maps: Vec<ObjectId> = self.objects.iter().filter(|o| o.components.iter().any(|c| component_inner(c.as_ref()).as_any().is::<Tilemap>())).map(|o| o.id).collect();
        let mut st = std::mem::take(&mut self.tile_objects);
        if maps.is_empty() {
            st.synced.clear();
            self.tile_objects = st;
            return 0;
        }
        st.calls = st.calls.wrapping_add(1);
        let calls = st.calls;
        let (synced, warned) = (&mut st.synced, &mut st.warned);
        let db = st.db.get_or_insert_with(|| TileDb::new(project_root));
        if calls.is_multiple_of(30) {
            db.refresh();
        }
        let mut changed = 0;
        for id in &maps {
            let Some(t) = self.objects.iter().position(|o| o.id == *id) else { continue };
            let Some(map) = self.objects[t].components.iter().find_map(|c| component_inner(c.as_ref()).as_any().downcast_ref::<Tilemap>()) else { continue };
            let stamp = (map.serial(), db.revision());
            if synced.get(id) == Some(&stamp) {
                continue;
            }
            // What the map wants, and the cells whose tile could not be read (their instances stay).
            let mut want: Vec<WantObject> = Vec::new();
            let mut unknown: HashSet<IVec2> = HashSet::new();
            let mut cells: Vec<(IVec2, u16)> = Vec::new();
            map.for_each_cell(|c, i| cells.push((c, i)));
            let mut by_index: HashMap<u16, Option<String>> = HashMap::new();
            for (cell, index) in cells {
                let prefab = by_index
                    .entry(index)
                    .or_insert_with(|| {
                        let path = map.tile_path(index)?.to_owned();
                        db.get(&path).and_then(|t| prefab_of(&t).map(str::to_owned))
                    })
                    .clone();
                match (&prefab, map.tile_path(index).is_some_and(|p| db.peek(p).is_some())) {
                    (Some(p), _) => want.push(WantObject { cell, prefab: p.clone() }),
                    (None, false) => {
                        unknown.insert(cell);
                    }
                    (None, true) => {}
                }
            }
            let xz = map.orientation == TilemapOrientation::XZ;
            let anchor_grid = grid_for(self, t);
            // What exists.
            let mut have: Vec<HaveObject> = Vec::new();
            let mut have_ids: Vec<ObjectId> = Vec::new();
            for (i, o) in self.objects.iter().enumerate() {
                if o.parent != Some(t) {
                    continue;
                }
                if let Some(to) = o.components.iter().find_map(|c| component_inner(c.as_ref()).as_any().downcast_ref::<TileObject>()) {
                    have.push(HaveObject { object: i, cell: IVec2::new(to.cell[0], to.cell[1]), prefab: to.prefab.clone() });
                    have_ids.push(o.id);
                }
            }
            let plan = plan_tile_objects(&want, &have);
            let tile_names: HashMap<IVec2, String> = {
                let mut m = HashMap::new();
                if let Some(map) = self.objects[t].components.iter().find_map(|c| component_inner(c.as_ref()).as_any().downcast_ref::<Tilemap>()) {
                    for w in &want {
                        if let Some(p) = map.tile_at(w.cell) {
                            m.insert(w.cell, p.to_owned());
                        }
                    }
                }
                m
            };
            // Remove first (indices shift, so by id).
            for r in plan.remove {
                if unknown.contains(&have[r].cell) {
                    continue;
                }
                if let Some(i) = self.objects.iter().position(|o| o.id == have_ids[r]) {
                    self.remove_object(i);
                    changed += 1;
                }
            }
            for s in plan.spawn {
                let w = &want[s];
                let Some(t) = self.objects.iter().position(|o| o.id == *id) else { break };
                let local = anchor_grid.cell_center(w.cell.extend(0));
                let local = if xz { Vec3::new(local.x, 0.0, local.y) } else { local };
                let name = format!("{} ({}, {})", std::path::Path::new(&w.prefab).file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "Tile".into()), w.cell.x, w.cell.y);
                if !zestengine_core::asset_ref::resolve_path(project_root, &w.prefab).is_file() {
                    if warned.insert(w.prefab.clone()) {
                        tracing::warn!("tile object: prefab '{}' does not exist", w.prefab);
                    }
                    continue;
                }
                let mut entries = vec![crate::prefab::link_root(&crate::prefab::link_path(project_root, &zestengine_core::asset_ref::resolve_path(project_root, &w.prefab)), &name, local.to_array())];
                crate::prefab::expand_entries(&mut entries, project_root);
                if entries.len() == 1 && entries[0].prefab.as_ref().is_some_and(|l| !l.expanded) {
                    if warned.insert(w.prefab.clone()) {
                        tracing::warn!("tile object: prefab '{}' could not be read", w.prefab);
                    }
                    continue;
                }
                let marker = TileObject { cell: [w.cell.x, w.cell.y], tile: tile_names.get(&w.cell).cloned().unwrap_or_default(), prefab: w.prefab.clone() };
                entries[0].components.push(ComponentData::of(&marker));
                match self.append_entries_with(renderer.as_deref_mut(), shaders, project_root, registry, &entries) {
                    Ok(base) => {
                        self.objects[base].parent = Some(t);
                        changed += 1;
                    }
                    Err(e) => {
                        if warned.insert(w.prefab.clone()) {
                            tracing::warn!("tile object '{}': {e:#}", w.prefab);
                        }
                    }
                }
            }
            if let Some(map) = self.objects.iter().find(|o| o.id == *id).and_then(|o| o.components.iter().find_map(|c| component_inner(c.as_ref()).as_any().downcast_ref::<Tilemap>())) {
                synced.insert(*id, (map.serial(), db.revision()));
            }
        }
        synced.retain(|k, _| maps.contains(k));
        self.tile_objects = st;
        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_assets::{MaterialRef, ObjectSource, SceneEntry};
    use zestengine_core::{TileAsset, TileFile};

    fn entry(name: &str, parent: Option<usize>, components: Vec<ComponentData>) -> SceneEntry {
        SceneEntry {
            id: ObjectId::new().to_string(),
            name: name.into(),
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
            components,
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
            model: None,
            model_part: None,
            prefab: None,
            prefab_parent: None,
            net_role: Default::default(),
            component_roles: Default::default(),
        }
    }

    struct Rig {
        root: std::path::PathBuf,
        scene: LoadedScene,
        shaders: ShaderLibrary,
        registry: ComponentRegistry,
    }

    /// A project with `crate.prefab` (Crate > Lid), `crate.tile` naming it, `grass.tile` (no prefab) and a
    /// scene with one Tilemap object holding `cells`.
    fn rig(tag: &str, orientation: TilemapOrientation) -> Rig {
        let root = std::env::temp_dir().join(format!("zest-tileobj-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let prefab = crate::prefab::Prefab::new(vec![entry("Crate", None, vec![]), entry("Lid", Some(0), vec![])]);
        prefab.save(root.join("crate.prefab")).unwrap();
        TileFile::Tile(TileAsset { game_object: "crate.prefab".into(), ..Default::default() }).save(&root.join("crate.tile")).unwrap();
        TileFile::Tile(TileAsset::with_sprite("a.png#0000000000000001")).save(&root.join("grass.tile")).unwrap();
        let registry = ComponentRegistry::with_builtins();
        let mut shaders = ShaderLibrary::default();
        let mut scene = LoadedScene::empty();
        let mut map = Tilemap::default();
        map.orientation = orientation;
        let tm = entry("Map", None, vec![ComponentData::of(&map)]);
        scene.append_entries_with(None, &mut shaders, &root, &registry, &[tm]).unwrap();
        Rig { root, scene, shaders, registry }
    }

    impl Rig {
        fn map(&mut self) -> &mut Tilemap {
            self.scene.objects[0].components.iter_mut().find_map(|c| zestengine_core::component_inner_mut(c).as_any_mut().downcast_mut::<Tilemap>()).unwrap()
        }
        fn sync(&mut self) -> usize {
            self.scene.sync_tile_objects(None, &mut self.shaders, &self.root, &self.registry)
        }
        fn instances(&self) -> Vec<(IVec2, usize)> {
            let mut v: Vec<(IVec2, usize)> = self
                .scene
                .objects
                .iter()
                .enumerate()
                .filter_map(|(i, o)| o.components.iter().find_map(|c| component_inner(c.as_ref()).as_any().downcast_ref::<TileObject>()).map(|t| (IVec2::new(t.cell[0], t.cell[1]), i)))
                .collect();
            v.sort_by_key(|(c, _)| (c.x, c.y));
            v
        }
    }

    #[test]
    fn painting_makes_instances_erasing_removes_them_and_a_synced_map_costs_nothing() {
        let mut r = rig("paint", TilemapOrientation::XY);
        assert_eq!(r.sync(), 0, "an empty map has no instances");
        r.map().set_tile(IVec2::new(2, 3), Some("crate.tile"));
        r.map().set_tile(IVec2::new(-1, 0), Some("crate.tile"));
        r.map().set_tile(IVec2::new(0, 0), Some("grass.tile"));
        assert_eq!(r.sync(), 2, "one instance per crate cell, none for the plain tile");
        let inst = r.instances();
        assert_eq!(inst.iter().map(|(c, _)| *c).collect::<Vec<_>>(), vec![IVec2::new(-1, 0), IVec2::new(2, 3)]);
        // Each is a child of the map at its cell, with the prefab's own children under it.
        for (cell, i) in &inst {
            let o = &r.scene.objects[*i];
            assert_eq!(o.parent, Some(0));
            assert_eq!(o.translation, Vec3::new(cell.x as f32 + 0.5, cell.y as f32 + 0.5, 0.0), "at the cell centre, local to the map");
            assert!(o.name.starts_with("crate ("), "{}", o.name);
            assert!(r.scene.objects.iter().any(|c| c.name == "Lid" && c.parent == Some(*i)), "the whole prefab came with it");
            assert!(r.scene.prefab_link(*i).is_some(), "a linked instance: edits to the prefab reach it");
        }
        let n = r.scene.objects.len();
        assert_eq!(r.sync(), 0, "nothing changed: nothing to do");
        assert_eq!(r.scene.objects.len(), n);
        // A hand-made change to one instance survives an unrelated edit elsewhere in the map.
        let (_, keep) = r.instances()[1];
        r.scene.objects[keep].translation.z = 9.0;
        r.map().set_tile(IVec2::new(5, 5), Some("grass.tile"));
        r.sync();
        assert_eq!(r.scene.objects[r.instances()[1].1].translation.z, 9.0);
        // Erase one: its instance and the prefab's children go; the other stays.
        r.map().set_tile(IVec2::new(-1, 0), None);
        assert_eq!(r.sync(), 1);
        assert_eq!(r.instances().iter().map(|(c, _)| *c).collect::<Vec<_>>(), vec![IVec2::new(2, 3)]);
        assert_eq!(r.scene.objects.iter().filter(|o| o.name == "Lid").count(), 1);
        // Retile the remaining cell to a tile with no prefab: removed too.
        r.map().set_tile(IVec2::new(2, 3), Some("grass.tile"));
        r.sync();
        assert!(r.instances().is_empty());
        assert_eq!(r.scene.objects.iter().filter(|o| o.name == "Lid").count(), 0);
        let _ = std::fs::remove_dir_all(&r.root);
    }

    #[test]
    fn undo_of_the_map_edit_brings_the_instances_back_and_loading_does_not_double_them() {
        let mut r = rig("undo", TilemapOrientation::XY);
        r.map().set_tile(IVec2::new(1, 1), Some("crate.tile"));
        r.sync();
        let before = r.map().clone();
        r.map().set_tile(IVec2::new(1, 1), None);
        r.sync();
        assert!(r.instances().is_empty());
        // Undo restores the component; the next sync restores the instance.
        *r.map() = before.clone();
        assert_eq!(r.sync(), 1);
        assert_eq!(r.instances().len(), 1);
        // A save / load round trip of the scene objects (what Play and the editor do): the saved
        // instances are found again, not made a second time.
        let snap = r.scene.snapshot_objects();
        r.scene.restore_objects(&snap, &r.registry);
        r.scene.tile_objects.reset();
        assert_eq!(r.sync(), 0, "the loaded instance matches its cell");
        assert_eq!(r.instances().len(), 1);
        let _ = std::fs::remove_dir_all(&r.root);
    }

    #[test]
    fn xz_maps_place_instances_on_the_ground_and_unreadable_tiles_keep_what_exists() {
        let mut r = rig("xz", TilemapOrientation::XZ);
        r.map().set_tile(IVec2::new(2, 3), Some("crate.tile"));
        assert_eq!(r.sync(), 1);
        let (_, i) = r.instances()[0];
        assert_eq!(r.scene.objects[i].translation, Vec3::new(2.5, 0.0, 3.5), "cell (x, y) at local (x, 0, y)");
        // The tile file disappears: its cell's instance must not be deleted as if it were erased.
        std::fs::remove_file(r.root.join("crate.tile")).unwrap();
        r.scene.tile_objects.reset();
        r.map().set_tile(IVec2::new(9, 9), Some("grass.tile"));
        r.sync();
        assert_eq!(r.instances().len(), 1, "an unreadable tile changes nothing");
        // A missing prefab spawns nothing and does not fail.
        TileFile::Tile(TileAsset { game_object: "nope.prefab".into(), ..Default::default() }).save(&r.root.join("gone.tile")).unwrap();
        r.map().set_tile(IVec2::new(4, 4), Some("gone.tile"));
        r.sync();
        assert_eq!(r.instances().len(), 1);
        let _ = std::fs::remove_dir_all(&r.root);
    }
}
