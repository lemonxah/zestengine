//! Writes the products of the Aseprite and PSD importers (2D-7) into a project:
//! the sheet PNG with a Sprite-kind `.meta` (Multiple mode, stable sprite ids),
//! a `.spritelib` of the frames or layers, `.anim` clips (Aseprite tags), and
//! for a PSD the layer tree (`.layers`) and the optional `.rig2d` seed.
//!
//! Re-importing keeps what can be kept: sprites match their previous entry by
//! name, so ids (and every scene reference) survive, and the `.meta` GUID is
//! reused. PSD sprites additionally keep pivots, borders and outlines
//! ([`crate::psd::Imported::match_previous`]).

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};

use crate::asset_meta::{AssetMeta, ImporterSettings, load_asset_meta, save_asset_meta};
use crate::sprite::{SpriteDef, SpriteMode, sprite_ref_string};
use crate::texture_import::{TextureImport, TextureKind};
use crate::{aseprite, psd};

/// Files an import wrote (absolute paths).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ImportedFiles {
    pub sheet: PathBuf,
    pub library: Option<PathBuf>,
    pub clips: Vec<PathBuf>,
    pub layers: Option<PathBuf>,
    pub rig: Option<PathBuf>,
}

/// Persisted PSD layer tree (`.layers`): what a prefab builder needs.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LayerTree {
    pub canvas: [u32; 2],
    pub ppu: f32,
    /// Sprite reference text per node (`"path#id"`; empty for groups).
    pub nodes: Vec<LayerTreeNode>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LayerTreeNode {
    pub name: String,
    pub parent: Option<usize>,
    pub sprite: String,
    pub position: [f32; 2],
    pub sorting_order: i32,
}

fn rel(project_root: &Path, p: &Path) -> String {
    p.strip_prefix(project_root).unwrap_or(p).to_string_lossy().replace('\\', "/")
}

fn safe(name: &str) -> String {
    let s: String = name.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
    if s.is_empty() { "clip".into() } else { s }
}

/// Write the sheet PNG and its Sprite `.meta`; returns the sprites with their final ids.
fn write_sheet(png: &Path, w: u32, h: u32, rgba: &[u8], mut sprites: Vec<SpriteDef>, ppu: f32) -> Result<Vec<SpriteDef>> {
    image::save_buffer(png, rgba, w, h, image::ColorType::Rgba8).with_context(|| format!("write {}", png.display()))?;
    let prev = load_asset_meta(png).ok().flatten();
    let (id, mut tex) = match prev {
        Some(m) => {
            let t = m.importer.as_texture().cloned().unwrap_or_default();
            (m.id, t)
        }
        None => (AssetMeta::default().id, TextureImport::default()),
    };
    tex.configured = true;
    tex.kind = TextureKind::Sprite;
    // Same-name sprites of the previous import keep their id.
    for s in &mut sprites {
        if s.id.is_none()
            && let Some(old) = tex.sprite.sprites.iter().find(|o| o.name == s.name)
        {
            s.id = old.id;
        }
    }
    tex.sprite.mode = SpriteMode::Multiple;
    tex.sprite.ppu = ppu;
    tex.sprite.sprites = sprites;
    let stem = png.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    tex.sprite.ensure_ids(&stem);
    let out = tex.sprite.sprites.clone();
    save_asset_meta(png, &AssetMeta { id, importer: ImporterSettings::Texture(tex) })?;
    Ok(out)
}

/// Import `src` (`.ase` / `.aseprite`) next to itself: `<stem>.png`, `<stem>.spritelib` and
/// one `<stem>_<tag>.anim` per tag.
pub fn import_aseprite_file(project_root: &Path, src: &Path, opt: &aseprite::ImportOptions) -> Result<ImportedFiles> {
    let bytes = std::fs::read(src).with_context(|| format!("read {}", src.display()))?;
    let file = aseprite::AseFile::parse(&bytes).with_context(|| format!("parse {}", src.display()))?;
    let mut opt = opt.clone();
    let stem = src.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "sprite".into());
    if opt.name == aseprite::ImportOptions::default().name {
        opt.name = stem.clone();
    }
    let mut imp = aseprite::import(&file, &opt)?;
    let dir = src.parent().unwrap_or(Path::new("."));
    let png = dir.join(format!("{stem}.png"));
    imp.sprites = write_sheet(&png, imp.sheet_w, imp.sheet_h, &imp.sheet, std::mem::take(&mut imp.sprites), opt.ppu)?;
    imp.bind_library(&rel(project_root, &png));
    let mut out = ImportedFiles { sheet: png, ..Default::default() };
    let lib = dir.join(format!("{stem}.spritelib"));
    crate::sprite_library::save_sprite_library(&lib, &imp.library)?;
    out.library = Some(lib);
    for (name, clip) in &imp.clips {
        let p = dir.join(format!("{stem}_{}.anim", safe(name)));
        crate::anim_clip::save_anim_clip(&p, clip)?;
        out.clips.push(p);
    }
    Ok(out)
}

/// Import `src` (`.psd` / `.psb`) next to itself: `<stem>.png`, `<stem>.layers`
/// and, with `rig_seed`, `<stem>.rig2d`.
pub fn import_psd_file(project_root: &Path, src: &Path, opt: &psd::ImportOptions) -> Result<ImportedFiles> {
    let bytes = std::fs::read(src).with_context(|| format!("read {}", src.display()))?;
    let file = psd::PsdFile::parse(&bytes).with_context(|| format!("parse {}", src.display()))?;
    let stem = src.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "character".into());
    let dir = src.parent().unwrap_or(Path::new("."));
    let png = dir.join(format!("{stem}.png"));
    let mut imp = psd::import(&file, opt)?;
    // Keep the edits of a previous import.
    if let Some(prev) = load_asset_meta(&png).ok().flatten().and_then(|m| m.importer.as_texture().map(|t| t.sprite.sprites.clone())) {
        imp.match_previous(&prev);
    }
    imp.sprites = write_sheet(&png, imp.sheet_w, imp.sheet_h, &imp.sheet, std::mem::take(&mut imp.sprites), opt.ppu)?;
    let sheet_rel = rel(project_root, &png);
    let mut out = ImportedFiles { sheet: png, ..Default::default() };
    let tree = LayerTree {
        canvas: [file.width, file.height],
        ppu: opt.ppu,
        nodes: imp
            .nodes
            .iter()
            .map(|n| LayerTreeNode {
                name: n.name.clone(),
                parent: n.parent,
                sprite: n.sprite.map(|s| sprite_ref_string(&sheet_rel, imp.sprites[s].id)).unwrap_or_default(),
                position: n.position,
                sorting_order: n.sorting_order,
            })
            .collect(),
    };
    let lp = dir.join(format!("{stem}.layers"));
    std::fs::write(&lp, ron::ser::to_string_pretty(&tree, ron::ser::PrettyConfig::default())?).with_context(|| format!("write {}", lp.display()))?;
    out.layers = Some(lp);
    if let Some(mut rig) = imp.rig.take() {
        for part in &mut rig.parts {
            if let Some(s) = imp.sprites.iter().find(|s| s.name == part.name) {
                part.sprite_id = s.id.0;
            }
        }
        let rp = dir.join(format!("{stem}.rig2d"));
        crate::rig2d::save_rig2d(&rp, &rig)?;
        out.rig = Some(rp);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aseprite::synth as ase;
    use crate::psd::synth as ps;

    fn temp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("zest_layered_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn aseprite_import_writes_sheet_library_and_clips_that_resolve() {
        let dir = temp("ase");
        let red: Vec<u8> = (0..4).flat_map(|_| [255, 0, 0, 255]).collect();
        let green: Vec<u8> = (0..4).flat_map(|_| [0, 255, 0, 255]).collect();
        let bytes = ase::file(
            4,
            4,
            32,
            1,
            &[
                (100, vec![ase::layer("l", 0, 0, 1, 0, 255, None), ase::cel(0, 0, 0, 255, 0, ase::Cel::Raw(2, 2, &red)), ase::tags(&[("run", 0, 1, 0, 0)])]),
                (80, vec![ase::cel(0, 1, 1, 255, 0, ase::Cel::Raw(2, 2, &green))]),
            ],
        );
        let src = dir.join("hero.aseprite");
        std::fs::write(&src, bytes).unwrap();
        let files = import_aseprite_file(&dir, &src, &aseprite::ImportOptions::default()).unwrap();
        assert!(files.sheet.exists() && files.library.as_ref().unwrap().exists());
        assert_eq!(files.clips.len(), 1);
        assert!(files.clips[0].ends_with("hero_run.anim"));
        // The sheet is a Sprite texture with two named, id'd sprites.
        let table = crate::sprite::load_sprite_table(&files.sheet).unwrap().expect("sprite table");
        assert_eq!(table.sprites.len(), 2);
        assert!(table.sprites.iter().all(|s| !s.id.is_none()));
        assert_eq!(table.sprites[0].name, "hero_0");
        // Every swap key of the clip resolves to a real sprite of the sheet.
        let lib = crate::sprite_library::load_resolved_library(&dir, files.library.as_ref().unwrap()).unwrap();
        let clip = crate::anim_clip::load_anim_clip(&files.clips[0]).unwrap();
        assert!((clip.length - 0.18).abs() < 1e-5);
        for k in &clip.tracks[0].keys {
            let text = lib.sprite(k.value[0] as u32).expect("hash resolves");
            let r = crate::sprite::SpriteRef::from_ref_string(text).expect("a sprite reference");
            assert_eq!(r.path, "hero.png");
            assert!(table.find(r.sprite).is_some());
        }
        // Re-import keeps the ids and the meta GUID.
        let ids: Vec<_> = table.sprites.iter().map(|s| s.id).collect();
        let guid = table.guid;
        import_aseprite_file(&dir, &src, &aseprite::ImportOptions::default()).unwrap();
        let again = crate::sprite::load_sprite_table(&files.sheet).unwrap().unwrap();
        assert_eq!(again.sprites.iter().map(|s| s.id).collect::<Vec<_>>(), ids);
        assert_eq!(again.guid, guid);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn psd_import_writes_sheet_tree_and_rig() {
        let dir = temp("psd");
        let layers = vec![
            ps::layer("bg", 1, (0, 0, 8, 8), [9, 9, 9, 255], ps::Comp::Raw),
            ps::divider(),
            ps::layer("hand", 2, (4, 6, 6, 8), [255, 0, 0, 255], ps::Comp::Rle),
            ps::group("Arm", 3, true),
        ];
        let src = dir.join("char.psd");
        std::fs::write(&src, ps::file(8, 8, &layers, false)).unwrap();
        let opt = psd::ImportOptions { rig_seed: true, ppu: 10.0, ..Default::default() };
        let files = import_psd_file(&dir, &src, &opt).unwrap();
        let table = crate::sprite::load_sprite_table(&files.sheet).unwrap().unwrap();
        assert_eq!(table.sprites.len(), 2);
        let tree: LayerTree = ron::from_str(&std::fs::read_to_string(files.layers.as_ref().unwrap()).unwrap()).unwrap();
        assert_eq!(tree.nodes.len(), 3);
        let hand = tree.nodes.iter().find(|n| n.name == "hand").unwrap();
        let r = crate::sprite::SpriteRef::from_ref_string(&hand.sprite).unwrap();
        assert_eq!(table.find(r.sprite).unwrap().name, "hand");
        let rig = crate::rig2d::load_rig2d(files.rig.as_ref().unwrap()).unwrap();
        assert_eq!(rig.parts.len(), 2);
        assert!(rig.parts.iter().all(|p| table.find(crate::sub_asset::SubAssetId(p.sprite_id)).is_some()), "rig parts name the imported sprite ids");
        // User edits survive a re-import.
        let mut meta = load_asset_meta(&files.sheet).unwrap().unwrap();
        if let ImporterSettings::Texture(t) = &mut meta.importer {
            t.sprite.sprites[0].pivot = [0.2, 0.8];
            t.sprite.sprites[0].pivot_preset = crate::sprite::PivotPreset::Custom;
        }
        let edited_name = meta.importer.as_texture().unwrap().sprite.sprites[0].name.clone();
        save_asset_meta(&files.sheet, &meta).unwrap();
        import_psd_file(&dir, &src, &opt).unwrap();
        let again = crate::sprite::load_sprite_table(&files.sheet).unwrap().unwrap();
        assert_eq!(again.sprites.iter().find(|s| s.name == edited_name).unwrap().pivot, [0.2, 0.8]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
