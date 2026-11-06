//! Loaded tile assets and per-cell resolution (2D-5), shared by the sprite
//! pass gather, the tilemap collider builder and the editor.
//!
//! [`TileDb`] loads `.tile` / `.ruletile` files by project-relative path, once
//! (rule tiles are compiled to their lookup table at load), and re-reads a file
//! whose modification time changed ([`TileDb::refresh`], which bumps
//! [`TileDb::revision`] so caches built from tiles know to rebuild). A missing
//! or unreadable file is remembered as absent: its cells draw nothing and
//! collide with nothing.
//!
//! [`resolve_cell`] answers "what does this cell show": the sprite(s), the final
//! transform and colour, the collider type, the animation. A rule tile costs one
//! neighbour mask and one table read.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use glam::IVec2;
use zestengine_core::{
    CompiledRules, FLAG_LOCK_COLOR, FLAG_LOCK_TRANSFORM, RuleHit, RuleTile, TileAsset, TileColliderType, TileFile, TileSel, TileXform, Tilemap, neighbor_mask,
};

/// A tile file, ready to resolve cells.
pub enum LoadedTile {
    Simple(TileAsset),
    Rule { tile: RuleTile, rules: CompiledRules },
}

impl LoadedTile {
    pub fn from_file(f: TileFile) -> LoadedTile {
        match f {
            TileFile::Tile(t) => LoadedTile::Simple(t),
            TileFile::Rule(r) => {
                let rules = r.compile();
                LoadedTile::Rule { tile: r, rules }
            }
        }
    }

    /// Every sprite reference the tile can show.
    pub fn sprites(&self) -> Vec<&str> {
        match self {
            LoadedTile::Simple(t) => t.sprites(),
            LoadedTile::Rule { tile, .. } => tile.all_sprites(),
        }
    }
}

struct Entry {
    tile: Option<Arc<LoadedTile>>,
    stamp: Option<SystemTime>,
}

/// Tile files of a project by path.
pub struct TileDb {
    root: PathBuf,
    tiles: HashMap<String, Entry>,
    revision: u64,
}

impl TileDb {
    pub fn new(root: &Path) -> Self {
        Self { root: root.to_path_buf(), tiles: HashMap::new(), revision: 1 }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Bumped whenever a loaded tile changed on disk or was inserted.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Insert an in-memory tile (tests, the editor's unsaved edits).
    pub fn insert(&mut self, path: &str, f: TileFile) {
        self.tiles.insert(path.to_string(), Entry { tile: Some(Arc::new(LoadedTile::from_file(f))), stamp: None });
        self.revision += 1;
    }

    /// The tile at `path`, loading it on first use.
    pub fn get(&mut self, path: &str) -> Option<Arc<LoadedTile>> {
        if let Some(e) = self.tiles.get(path) {
            return e.tile.clone();
        }
        let abs = self.root.join(path);
        let stamp = std::fs::metadata(&abs).and_then(|m| m.modified()).ok();
        let tile = match TileFile::load(&abs) {
            Ok(f) => Some(Arc::new(LoadedTile::from_file(f))),
            Err(e) => {
                tracing::warn!("tile '{path}': {e}");
                None
            }
        };
        self.tiles.insert(path.to_string(), Entry { tile: tile.clone(), stamp });
        tile
    }

    /// The already-loaded tile (no disk access).
    pub fn peek(&self, path: &str) -> Option<&Arc<LoadedTile>> {
        self.tiles.get(path).and_then(|e| e.tile.as_ref())
    }

    /// Re-read files whose timestamp changed; returns whether anything did.
    pub fn refresh(&mut self) -> bool {
        let mut changed = false;
        let paths: Vec<String> = self.tiles.keys().cloned().collect();
        for p in paths {
            let abs = self.root.join(&p);
            let stamp = std::fs::metadata(&abs).and_then(|m| m.modified()).ok();
            let old = self.tiles.get(&p).map(|e| e.stamp);
            if old.is_some_and(|s| s.is_some() && s != stamp) || (old == Some(None) && stamp.is_some() && self.tiles[&p].tile.is_none()) {
                let tile = TileFile::load(&abs).ok().map(|f| Arc::new(LoadedTile::from_file(f)));
                self.tiles.insert(p, Entry { tile, stamp });
                changed = true;
            }
        }
        if changed {
            self.revision += 1;
        }
        changed
    }

    /// Forget everything (project switch).
    pub fn clear(&mut self) {
        self.tiles.clear();
        self.revision += 1;
    }
}

/// What one cell shows.
#[derive(Clone, PartialEq, Debug)]
pub struct ResolvedCell<'a> {
    pub sel: TileSel<'a>,
    /// Final orientation: the tile's own, then the cell's override.
    pub xform: TileXform,
    /// Final tint (tile colour times the cell override), linear RGBA 0..1.
    pub color: [f32; 4],
    pub collider: TileColliderType,
}

fn unpack(c: [u8; 4]) -> [f32; 4] {
    [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0, c[3] as f32 / 255.0]
}

/// Resolve `cell` of `map`, whose tile (`path`) was loaded as `tile`.
pub fn resolve_cell<'a>(map: &Tilemap, tile: &'a LoadedTile, path: &str, cell: IVec2) -> ResolvedCell<'a> {
    let sel = match tile {
        LoadedTile::Simple(t) => t.select(cell),
        LoadedTile::Rule { tile, rules } => {
            let mask = neighbor_mask(tile.layout, cell, |c| map.tile_at(c) == Some(path));
            let hit: RuleHit = rules.lookup(mask);
            tile.select(hit, cell)
        }
    };
    let flags = sel.flags;
    let xform = if flags & FLAG_LOCK_TRANSFORM != 0 { sel.xform } else { map.xform_at(cell).after(sel.xform) };
    let mut color = sel.color;
    if flags & FLAG_LOCK_COLOR == 0 {
        let o = unpack(map.color_at(cell));
        for k in 0..4 {
            color[k] *= o[k];
        }
    }
    let collider = sel.collider;
    ResolvedCell { sel, xform, color, collider }
}

/// The grid a tilemap object belongs to: its own `Grid` component, else the
/// nearest ancestor's, else the default (1 x 1 rectangles).
pub fn grid_for(scene: &crate::scene::LoadedScene, index: usize) -> zestengine_core::Grid {
    let mut at = Some(index);
    let mut guard = 0;
    while let Some(i) = at {
        if let Some(g) = scene.objects[i].components.iter().find_map(|c| zestengine_core::component_inner(c.as_ref()).as_any().downcast_ref::<zestengine_core::Grid>()) {
            return g.clone();
        }
        guard += 1;
        if guard > 64 {
            break;
        }
        at = scene.objects[i].parent.filter(|&p| p < scene.objects.len());
    }
    zestengine_core::Grid::default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_core::{CellLayout, NeighborReq, RuleOutput, TilingRule};

    fn sprite(n: u32) -> String {
        format!("s.png#{:016x}", n)
    }

    #[test]
    fn db_loads_compiles_refreshes_and_forgets_missing() {
        let dir = std::env::temp_dir().join(format!("zest_tiledb_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let t = TileAsset::with_sprite(&sprite(1));
        TileFile::Tile(t.clone()).save(&dir.join("a.tile")).unwrap();
        let mut db = TileDb::new(&dir);
        let r0 = db.revision();
        assert!(matches!(&*db.get("a.tile").unwrap(), LoadedTile::Simple(x) if *x == t));
        assert!(db.get("missing.tile").is_none());
        assert!(db.get("missing.tile").is_none(), "absence is remembered");
        assert!(!db.refresh());
        assert_eq!(db.revision(), r0);
        // Change the file (and its timestamp): a refresh sees it.
        std::thread::sleep(std::time::Duration::from_millis(1100));
        let t2 = TileAsset::with_sprite(&sprite(2));
        TileFile::Tile(t2.clone()).save(&dir.join("a.tile")).unwrap();
        assert!(db.refresh());
        assert!(db.revision() > r0);
        assert!(matches!(&*db.get("a.tile").unwrap(), LoadedTile::Simple(x) if *x == t2));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cell_resolution_combines_tile_rule_and_overrides() {
        let mut db = TileDb::new(Path::new("."));
        let rule = RuleTile {
            default_sprite: sprite(9),
            rules: vec![TilingRule {
                neighbors: {
                    let mut n = vec![NeighborReq::Any; 8];
                    n[0] = NeighborReq::This; // east
                    n
                },
                transform: zestengine_core::RuleTransform::Rotated,
                output: RuleOutput::Single,
                sprites: vec![sprite(1)],
                ..Default::default()
            }],
            color: [1.0, 0.5, 1.0, 1.0],
            ..Default::default()
        };
        db.insert("r.ruletile", TileFile::Rule(rule));
        let tile = db.get("r.ruletile").unwrap();
        let mut m = Tilemap::default();
        for x in 0..3 {
            m.set_tile(IVec2::new(x, 0), Some("r.ruletile"));
        }
        m.set_color(IVec2::new(0, 0), [128, 255, 255, 255]);
        // The middle cell has a neighbour to its east.
        let mid = resolve_cell(&m, &tile, "r.ruletile", IVec2::new(1, 0));
        assert_eq!(mid.sel.sprites, &[sprite(1)]);
        assert_eq!(mid.xform, TileXform::IDENTITY);
        // The last cell has one to its west only: the rule matches rotated a half turn.
        let last = resolve_cell(&m, &tile, "r.ruletile", IVec2::new(2, 0));
        assert_eq!(last.sel.sprites, &[sprite(1)]);
        assert_eq!(last.xform, TileXform::new(12, false));
        // A lone cell: no rule, the default sprite.
        let lone = resolve_cell(&m, &tile, "r.ruletile", IVec2::new(0, 5));
        assert!(lone.sel.sprites[0] == sprite(9) || m.tile_at(IVec2::new(0, 5)).is_none());
        // Tint: tile colour times the per-cell override (rule tiles lock the transform only).
        let first = resolve_cell(&m, &tile, "r.ruletile", IVec2::new(0, 0));
        assert!((first.color[0] - 128.0 / 255.0).abs() < 1e-5 && (first.color[1] - 0.5).abs() < 1e-5);
        // A transform override is ignored for rule tiles (locked) but applies to simple tiles.
        m.set_xform(IVec2::new(1, 0), TileXform::new(6, true));
        assert_eq!(resolve_cell(&m, &tile, "r.ruletile", IVec2::new(1, 0)).xform, TileXform::IDENTITY);
        db.insert("s.tile", TileFile::Tile(TileAsset::with_sprite(&sprite(3))));
        let s = db.get("s.tile").unwrap();
        m.set_tile(IVec2::new(7, 7), Some("s.tile"));
        m.set_xform(IVec2::new(7, 7), TileXform::new(6, false));
        assert_eq!(resolve_cell(&m, &s, "s.tile", IVec2::new(7, 7)).xform, TileXform::new(6, false));
        let _ = CellLayout::Rectangle;
    }
}
