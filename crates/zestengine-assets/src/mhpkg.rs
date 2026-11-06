//! Recognise MetaHuman packages (`.mhpkg`) and report what they hold.
//!
//! A `.mhpkg` is a zip (stored, no compression) with a `Manifest.json`
//! (asset name, type, engine version, dependent packages, the `details`
//! counters), a `FileList.json` and Unreal `.uasset` files. The `.uasset`
//! payload is Unreal's cooked-editor package format and is NOT read: this
//! module only opens the zip and reads the two JSON files, so the import
//! dialog can say what the package is and how to get the content out of
//! Unreal as FBX / glTF and textures, which the folder importer does read.

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail};
use serde_json::Value;

use crate::pack_zip::{ZipLimits, list_zip, read_zip_entry};

const MAX_JSON: u64 = 8 << 20;

/// One file in the package.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MhFile {
    pub path: String,
    pub bytes: u64,
}

/// What the manifest says.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MhPackage {
    pub archive: PathBuf,
    pub name: String,
    /// `/Game/MH/asi20.asi20`
    pub object_path: String,
    /// `Character`, `Outfit`, `Groom`, ...
    pub asset_type: String,
    /// `5.8` from `details.engineVersion`.
    pub engine_version: String,
    pub dependent_packages: Vec<String>,
    pub total_size: u64,
    pub files: Vec<MhFile>,
    pub included_lods: u32,
    pub lod0_vertices: u64,
    pub has_lods: bool,
    pub contains_grooms: bool,
    pub unique_grooms: u32,
    pub strands_count: u64,
    pub contains_clothing: bool,
    pub unique_clothing: u32,
    pub materials: u32,
    pub substrate_materials: u32,
    pub virtual_textures: u32,
    pub physics: bool,
    pub editable_character: bool,
    pub card_meshes: u32,
    pub platforms: Vec<String>,
    pub verification_warnings: usize,
    pub verification_errors: usize,
}

fn u64_of(v: &Value) -> u64 {
    match v {
        Value::Number(n) => n.as_u64().or_else(|| n.as_f64().map(|f| f.max(0.0) as u64)).unwrap_or(0),
        Value::String(s) => s.trim().parse().unwrap_or(0),
        _ => 0,
    }
}

fn str_of(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        _ => String::new(),
    }
}

fn bool_of(v: &Value) -> bool {
    match v {
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|x| x != 0.0),
        Value::String(s) => s.eq_ignore_ascii_case("true"),
        _ => false,
    }
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array().into_iter().flatten().map(str_of).filter(|s| !s.is_empty()).collect()
}

/// Parse `Manifest.json` text (unknown fields ignored, everything optional).
pub fn parse_manifest(json: &str, archive: &Path) -> Result<MhPackage> {
    let root: Value = serde_json::from_str(json).context("Manifest.json is not valid JSON")?;
    if !root.is_object() {
        bail!("Manifest.json is not an object");
    }
    let d = &root["details"];
    let ver = &root["verificationReport"];
    Ok(MhPackage {
        archive: archive.to_owned(),
        name: str_of(&root["name"]),
        object_path: str_of(&root["assetData"]["objectPath"]),
        asset_type: str_of(&root["assetType"]),
        engine_version: str_of(&d["engineVersion"]),
        dependent_packages: strings(&root["dependentPackages"]),
        total_size: u64_of(&root["totalSize"]),
        files: Vec::new(),
        included_lods: u64_of(&d["includedLods"]) as u32,
        lod0_vertices: u64_of(&d["lod0VertCount"]),
        has_lods: bool_of(&d["bHasLods"]),
        contains_grooms: bool_of(&d["bContainsGrooms"]),
        unique_grooms: u64_of(&d["numUniqueGrooms"]) as u32,
        strands_count: u64_of(&d["strandsCount"]),
        contains_clothing: bool_of(&d["bContainsClothing"]),
        unique_clothing: u64_of(&d["numUniqueClothingItems"]) as u32,
        materials: u64_of(&d["numMaterials"]) as u32,
        substrate_materials: u64_of(&d["numSubstrateMaterials"]) as u32,
        virtual_textures: u64_of(&d["numVirtualTextures"]) as u32,
        physics: bool_of(&d["bPhysics"]),
        editable_character: bool_of(&d["bIsEditableCharacter"]),
        card_meshes: u64_of(&d["cardMeshCount"]) as u32,
        platforms: strings(&d["platformsIncluded"]),
        verification_warnings: ver["warnings"].as_array().map_or(0, Vec::len),
        verification_errors: ver["errors"].as_array().map_or(0, Vec::len),
    })
}

/// Open a `.mhpkg` and read its manifest and file list.
pub fn read_mhpkg(path: &Path) -> Result<MhPackage> {
    let limits = ZipLimits::default();
    let list = list_zip(path, &limits)?;
    let manifest = list
        .entries
        .iter()
        .find(|e| !e.is_dir && e.name.eq_ignore_ascii_case("manifest.json"))
        .with_context(|| format!("{} has no Manifest.json: not a MetaHuman package", path.display()))?;
    let text = read_zip_entry(path, &manifest.name, MAX_JSON)?;
    let mut pkg = parse_manifest(&String::from_utf8_lossy(&text), path)?;
    pkg.files = list
        .entries
        .iter()
        .filter(|e| !e.is_dir && !e.name.eq_ignore_ascii_case("manifest.json") && !e.name.eq_ignore_ascii_case("filelist.json"))
        .map(|e| MhFile { path: e.name.clone(), bytes: e.size })
        .collect();
    if pkg.name.is_empty() {
        pkg.name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    }
    if pkg.total_size == 0 {
        pkg.total_size = pkg.files.iter().map(|f| f.bytes).sum();
    }
    Ok(pkg)
}

impl MhPackage {
    /// The `.uasset` count.
    pub fn uasset_count(&self) -> usize {
        self.files.iter().filter(|f| f.path.to_ascii_lowercase().ends_with(".uasset")).count()
    }

    /// What the manifest's counters say the package contains, one line each.
    pub fn contents(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.included_lods > 0 || self.lod0_vertices > 0 {
            out.push(format!("Meshes: {} LOD(s), {} vertices at LOD0", self.included_lods, self.lod0_vertices));
        } else {
            out.push("Meshes: none listed (the counters are 0; the textures may be the whole package)".into());
        }
        out.push(format!("Materials: {}{}", self.materials, if self.substrate_materials > 0 { format!(" ({} Substrate)", self.substrate_materials) } else { String::new() }));
        if self.contains_grooms || self.unique_grooms > 0 || self.strands_count > 0 {
            out.push(format!("Grooms: {} ({} strands)", self.unique_grooms, self.strands_count));
        } else {
            out.push("Grooms: none".into());
        }
        if self.card_meshes > 0 {
            out.push(format!("Hair cards: {} mesh(es)", self.card_meshes));
        }
        out.push(if self.contains_clothing { format!("Clothing: {} item(s)", self.unique_clothing) } else { "Clothing: none".into() });
        if self.virtual_textures > 0 {
            out.push(format!("Virtual textures: {}", self.virtual_textures));
        }
        out.push(format!("Physics asset: {}", if self.physics { "yes" } else { "no" }));
        out.push(format!("Textures and other .uasset files: {}", self.uasset_count()));
        out
    }
}

/// What to do instead, shown in the import dialog.
pub const EXPORT_HELP: &str = "A MetaHuman package holds Unreal Engine .uasset files, which ZestEngine does not read. \
To use this character: open it in Unreal Engine (the engine version shown above or newer) with the MetaHuman plugin, \
then in the Content Browser right-click the skeletal mesh and choose Asset Actions > Export to write an FBX (include all LODs and the skeleton), \
and export each texture (right-click > Asset Actions > Export, as PNG or TGA). Hair grooms only export as Alembic, which is not read: use the hair card meshes instead. \
Put the FBX and the textures in one folder (or a zip) and import that folder here: the textures are recognised by name (BaseColor, Normal, ORM and so on). \
Check the MetaHuman licence terms before using exported content.";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pack_zip::tests::{make_zip, scratch};

    const MANIFEST: &str = r#"{
        "name": "asi20", "assetData": {"objectPath": "/Game/MH/asi20.asi20"},
        "dependentPackages": ["/Game/MH/asi20", "/Game/MH/Head/4/Head_Color"],
        "assetType": "Character", "futureField": [1, 2],
        "details": {"includedLods": 8, "lod0VertCount": "24000", "bHasLods": true, "bContainsGrooms": true,
                    "numUniqueGrooms": 3, "strandsCount": 100000, "numMaterials": 12, "numSubstrateMaterials": 2,
                    "platformsIncluded": ["Windows"], "engineVersion": "5.6", "bPhysics": true},
        "totalSize": 46282420,
        "verificationReport": {"warnings": ["a", "b"], "errors": []}
    }"#;

    #[test]
    fn parses_a_synthesized_manifest() {
        let m = parse_manifest(MANIFEST, Path::new("x.mhpkg")).unwrap();
        assert_eq!((m.name.as_str(), m.asset_type.as_str(), m.engine_version.as_str()), ("asi20", "Character", "5.6"));
        assert_eq!((m.included_lods, m.lod0_vertices, m.materials, m.substrate_materials), (8, 24000, 12, 2));
        assert!(m.has_lods && m.contains_grooms && m.physics);
        assert_eq!((m.unique_grooms, m.strands_count), (3, 100000));
        assert_eq!(m.dependent_packages.len(), 2);
        assert_eq!((m.verification_warnings, m.verification_errors), (2, 0));
        assert_eq!(m.platforms, vec!["Windows"]);
        assert!(m.contents().iter().any(|l| l.starts_with("Grooms: 3")));
    }

    #[test]
    fn manifest_variants_and_garbage() {
        let m = parse_manifest(r#"{"name": "x"}"#, Path::new("x.mhpkg")).unwrap();
        assert_eq!(m.included_lods, 0);
        assert!(m.contents()[0].starts_with("Meshes: none"));
        assert!(parse_manifest("[]", Path::new("x")).is_err());
        assert!(parse_manifest("nope", Path::new("x")).is_err());
        // Wrong types do not panic.
        let m = parse_manifest(r#"{"details": {"includedLods": "x", "bContainsGrooms": "true"}, "dependentPackages": 5}"#, Path::new("x")).unwrap();
        assert!(m.contains_grooms);
        assert!(m.dependent_packages.is_empty());
    }

    #[test]
    fn reads_a_synthesized_stored_package() {
        let dir = scratch("mh");
        let z = dir.join("t.mhpkg");
        make_zip(&z, &[("t.uasset", &[0u8; 100], false), ("Head/Head_Color.uasset", &[0u8; 50], false), ("Manifest.json", MANIFEST.as_bytes(), false), ("FileList.json", b"{\"files\": []}", false)]);
        let m = read_mhpkg(&z).unwrap();
        assert_eq!(m.files.len(), 2);
        assert_eq!(m.uasset_count(), 2);
        assert_eq!(m.total_size, 46282420);
        // No manifest: refused with a clear message.
        let bad = dir.join("bad.mhpkg");
        make_zip(&bad, &[("a.uasset", b"x", false)]);
        assert!(format!("{:#}", read_mhpkg(&bad).unwrap_err()).contains("Manifest.json"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The real package from the author's Fab downloads, when present.
    #[test]
    fn reads_the_real_sample() {
        let p = Path::new("/home/lemonxah/Downloads/fab/asi20.mhpkg");
        if !p.is_file() {
            return;
        }
        let m = read_mhpkg(p).unwrap();
        assert_eq!((m.name.as_str(), m.asset_type.as_str(), m.engine_version.as_str()), ("asi20", "Character", "5.8"));
        assert_eq!(m.uasset_count(), 4);
        assert_eq!(m.dependent_packages.len(), 4);
        assert_eq!(m.total_size, 46282420);
        assert!(!m.contains_grooms);
        assert!(m.object_path.ends_with("asi20.asi20"));
    }
}
