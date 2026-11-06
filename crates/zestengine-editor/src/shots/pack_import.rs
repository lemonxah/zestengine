//! Headless screenshots of the Megascans / Fab import dialog (see
//! `crate::shot`).

use super::*;
use zestengine_assets::pack_scan::{ModelLod, ModelVariant, SetTexture, TextureSet};
use crate::shot::Shot;

fn tex(name: &str, role: TexRole, conf: Confidence, res: u32) -> SetTexture {
    SetTexture { path: PathBuf::from(format!("/pack/{name}")), role, confidence: conf, resolution: Some(res), bytes: 4_200_000 }
}

fn sample_scan() -> PackScan {
    let rock = PackAsset {
        id: "tfjmddjfa".into(),
        name: "Mossy Granite Boulder".into(),
        kind: AssetKind::Model3d,
        megascans: true,
        dir: PathBuf::from("/pack/rock"),
        variants: vec![ModelVariant {
            name: "VarA".into(),
            lods: (0..4).map(|l| ModelLod { lod: l, path: PathBuf::from(format!("/pack/rock/rock_LOD{l}.fbx")), bytes: 9_000_000 >> (l * 2) }).collect(),
        }],
        sets: vec![TextureSet {
            name: "rock".into(),
            billboard: false,
            textures: vec![
                tex("rock_Albedo.jpg", TexRole::Albedo, Confidence::Known, 4096),
                tex("rock_Normal.jpg", TexRole::Normal, Confidence::Known, 4096),
                tex("rock_N.jpg", TexRole::NormalDx, Confidence::Weak, 2048),
                tex("rock_Roughness.jpg", TexRole::Roughness, Confidence::Known, 4096),
                tex("rock_AO.jpg", TexRole::Ao, Confidence::Known, 4096),
            ],
        }],
        size_m: None,
        warnings: vec!["Texture rock_N.jpg: role guessed from a short name".into()],
        json: None,
        include: true,
    };
    let surface = PackAsset { id: "wdbt1".into(), name: "Cobblestone Floor".into(), kind: AssetKind::Surface, size_m: Some([2.0, 2.0]), variants: vec![], warnings: vec![], include: true, ..rock.clone() };
    let plant = PackAsset { id: "ps2".into(), name: "Fern".into(), kind: AssetKind::Plant, megascans: false, include: false, ..rock.clone() };
    PackScan { root: PathBuf::from("/pack"), assets: vec![rock, surface, plant], warnings: vec!["3 files skipped: .blend".into()] }
}

#[test]
fn pack_import_dialog() {
    let mut shot = Shot::new(1000.0, 840.0, 1.25);
    let mut ui_state = PackImportUi { dialog: Some(Dialog { path_text: "/pack".into(), phase: Phase::Ready { scan: sample_scan(), temp: None, selected: 0 } }), ..Default::default() };
    let mut tasks = TaskManager::default();
    let img = shot.run(|ctx| ui_state.show(ctx, &mut tasks, Path::new("/proj")));
    shot.save_named(&img, "pack_import_ready");
    assert!(img.non_background() > 500);

    let mut shot = Shot::new(1000.0, 840.0, 1.25);
    let pkg = MhPackage {
        archive: PathBuf::from("/dl/asi20.mhpkg"),
        name: "asi20".into(),
        object_path: "/Game/MH/asi20.asi20".into(),
        asset_type: "Character".into(),
        engine_version: "5.8".into(),
        dependent_packages: vec!["/Game/MH/Common".into()],
        total_size: 1_900_000_000,
        files: vec![zestengine_assets::mhpkg::MhFile { path: "Content/MH/asi20.uasset".into(), bytes: 120_000_000 }],
        included_lods: 8,
        lod0_vertices: 38_000,
        has_lods: true,
        contains_grooms: true,
        unique_grooms: 6,
        strands_count: 240_000,
        contains_clothing: true,
        unique_clothing: 3,
        materials: 42,
        substrate_materials: 12,
        virtual_textures: 30,
        physics: true,
        editable_character: true,
        card_meshes: 4,
        platforms: vec!["Windows".into()],
        verification_warnings: 2,
        verification_errors: 0,
    };
    let mut ui_state = PackImportUi { dialog: Some(Dialog { path_text: "/dl/asi20.mhpkg".into(), phase: Phase::MetaHuman(Box::new(pkg)) }), ..Default::default() };
    let img = shot.run(|ctx| ui_state.show(ctx, &mut tasks, Path::new("/proj")));
    shot.save_named(&img, "pack_import_metahuman");
    assert!(img.non_background() > 500);
}
