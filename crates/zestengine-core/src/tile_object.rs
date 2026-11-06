//! GameObject tiles (2D-5): a `.tile` whose `game_object` names a prefab puts one instance of it in
//! every cell that uses the tile.
//!
//! The instances are ordinary scene objects, children of the tilemap's object, each carrying a
//! [`TileObject`] that records the cell and the prefab it was made for (that is the per-cell storage:
//! it is saved with the scene, so a loaded map finds its instances again and does not make them twice).
//! The engine keeps them in step with the map (`engine/src/scene/tile_objects.rs`), in the editor and
//! in play: painting a cell makes its instance, erasing the cell or changing its tile removes or
//! replaces it, and undo of a stroke restores the map, after which the same step restores the
//! instances. [`plan_tile_objects`] is that step's pure core.

use glam::IVec2;
use serde::{Deserialize, Serialize};

use crate::TypedComponent;

/// Marks a scene object as the instance of a tile's prefab in one cell of its parent tilemap.
#[derive(Clone, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TileObject {
    /// The cell (x, y) of the parent tilemap.
    pub cell: [i32; 2],
    /// The tile asset that asked for it (informational).
    pub tile: String,
    /// The prefab it is an instance of (project-relative `.prefab`).
    pub prefab: String,
}

impl TypedComponent for TileObject {
    const NAME: &'static str = "Tile Object";
}

/// An instance the map wants: this cell shows a tile whose prefab is `prefab`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct WantObject {
    /// The cell to fill.
    pub cell: IVec2,
    /// The prefab the cell's tile names.
    pub prefab: String,
}

/// An instance that exists: scene object `object` (an index the caller understands) for `cell`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct HaveObject {
    /// Scene object index the caller understands.
    pub object: usize,
    /// The cell the instance sits in.
    pub cell: IVec2,
    /// The prefab it was made for.
    pub prefab: String,
}

/// What to do to make the instances match the map.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct TileObjectPlan {
    /// Indices into `want` that have no instance yet.
    pub spawn: Vec<usize>,
    /// Indices into `have` to delete: instances of erased cells, of another prefab than the cell
    /// now asks for, and duplicates of one cell.
    pub remove: Vec<usize>,
}

/// Match `have` against `want` by (cell, prefab).
pub fn plan_tile_objects(want: &[WantObject], have: &[HaveObject]) -> TileObjectPlan {
    use std::collections::{HashMap, HashSet};
    let wanted: HashMap<(IVec2, &str), usize> = want.iter().enumerate().map(|(i, w)| ((w.cell, w.prefab.as_str()), i)).collect();
    let mut kept: HashSet<(IVec2, &str)> = HashSet::new();
    let mut plan = TileObjectPlan::default();
    for (i, h) in have.iter().enumerate() {
        let key = (h.cell, h.prefab.as_str());
        if wanted.contains_key(&key) && kept.insert(key) {
            continue;
        }
        plan.remove.push(i);
    }
    for (i, w) in want.iter().enumerate() {
        if !kept.contains(&(w.cell, w.prefab.as_str())) {
            plan.spawn.push(i);
        }
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(x: i32, y: i32, p: &str) -> WantObject {
        WantObject { cell: IVec2::new(x, y), prefab: p.into() }
    }
    fn h(o: usize, x: i32, y: i32, p: &str) -> HaveObject {
        HaveObject { object: o, cell: IVec2::new(x, y), prefab: p.into() }
    }

    #[test]
    fn painting_erasing_and_replacing_cells_plan_the_right_edits() {
        // Nothing yet: everything spawns.
        let p = plan_tile_objects(&[w(0, 0, "a"), w(1, 0, "a")], &[]);
        assert_eq!((p.spawn.clone(), p.remove.clone()), (vec![0, 1], vec![]));
        // In step: nothing to do.
        let p = plan_tile_objects(&[w(0, 0, "a"), w(1, 0, "a")], &[h(7, 0, 0, "a"), h(9, 1, 0, "a")]);
        assert_eq!(p, TileObjectPlan::default());
        // (1, 0) was erased and (2, 0) painted.
        let p = plan_tile_objects(&[w(0, 0, "a"), w(2, 0, "a")], &[h(7, 0, 0, "a"), h(9, 1, 0, "a")]);
        assert_eq!((p.spawn, p.remove), (vec![1], vec![1]));
        // The cell's tile changed to one with another prefab: the old instance goes, a new one is made.
        let p = plan_tile_objects(&[w(0, 0, "b")], &[h(7, 0, 0, "a")]);
        assert_eq!((p.spawn, p.remove), (vec![0], vec![0]));
        // Duplicates of one cell (a hand-merged scene) collapse to one; the rest are removed.
        let p = plan_tile_objects(&[w(0, 0, "a")], &[h(1, 0, 0, "a"), h(2, 0, 0, "a"), h(3, 0, 0, "a")]);
        assert_eq!((p.spawn.len(), p.remove), (0, vec![1, 2]));
        // A map with no game-object tiles removes every instance.
        let p = plan_tile_objects(&[], &[h(1, 0, 0, "a"), h(2, 4, 4, "a")]);
        assert_eq!((p.spawn.len(), p.remove), (0, vec![0, 1]));
    }

    #[test]
    fn the_component_round_trips_and_old_files_load() {
        let t = TileObject { cell: [-3, 7], tile: "t.tile".into(), prefab: "p.prefab".into() };
        assert_eq!(ron::from_str::<TileObject>(&ron::to_string(&t).unwrap()).unwrap(), t);
        assert_eq!(ron::from_str::<TileObject>("()").unwrap(), TileObject::default());
    }
}
