//! Megascans / Quixel Bridge export metadata (`<id>.json`).
//!
//! The JSON is read leniently: every field is optional, unknown fields are
//! ignored, and numbers may arrive as strings. No export JSON was available
//! when this was written (Fab downloads of Megascans assets carry no JSON at
//! all, only the files), so the layout below follows the documented Bridge
//! export: `id`, `semanticTags` (`asset_type`, `name`, `tileable`, ...),
//! `meta` (`[{key, name, value}]` with `scanArea`, `height`, `tileable`),
//! `meshes` (`[{type, name, tris, uris: [{uri, lod, resolution, ...}]}]`) and
//! `maps` (`[{type, uri, resolution, physicalSize, ...}]`). Anything that does
//! not parse is skipped, and the import falls back to what the folder holds.

use std::path::Path;

use serde_json::Value;

/// What kind of Megascans asset this is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum AssetKind {
    /// A scanned 3D object (LOD0..n meshes, optional variations).
    #[default]
    Model3d,
    /// A plant: opacity + translucency, two-sided, billboard LOD.
    Plant,
    /// A tiling surface material.
    Surface,
    /// A texture atlas of cut-out elements.
    Atlas,
    Decal,
    /// A surface overlay (dirt, scratches, stains).
    Imperfection,
    /// A sculpting brush alpha.
    Brush,
    /// Textures with no mesh and no recognised type.
    Textures,
}

impl AssetKind {
    pub const ALL: [AssetKind; 8] = [AssetKind::Model3d, AssetKind::Plant, AssetKind::Surface, AssetKind::Atlas, AssetKind::Decal, AssetKind::Imperfection, AssetKind::Brush, AssetKind::Textures];

    pub fn label(self) -> &'static str {
        match self {
            AssetKind::Model3d => "3D asset",
            AssetKind::Plant => "3D plant",
            AssetKind::Surface => "Surface",
            AssetKind::Atlas => "Atlas",
            AssetKind::Decal => "Decal",
            AssetKind::Imperfection => "Imperfection",
            AssetKind::Brush => "Brush",
            AssetKind::Textures => "Textures",
        }
    }

    /// Read a Megascans type word (`3D asset`, `3d`, `3dplant`, `surface`, ...).
    pub fn from_word(word: &str) -> Option<AssetKind> {
        let w: String = word.to_ascii_lowercase().chars().filter(|c| c.is_ascii_alphanumeric()).collect();
        if w.is_empty() {
            return None;
        }
        Some(if w.contains("plant") || w.contains("foliage") {
            AssetKind::Plant
        } else if w.contains("surface") || w.contains("material") {
            AssetKind::Surface
        } else if w.contains("atlas") {
            AssetKind::Atlas
        } else if w.contains("decal") {
            AssetKind::Decal
        } else if w.contains("imperfection") {
            AssetKind::Imperfection
        } else if w.contains("brush") {
            AssetKind::Brush
        } else if w.starts_with("3d") || w.contains("asset") || w.contains("model") || w.contains("mesh") {
            AssetKind::Model3d
        } else {
            return None;
        })
    }
}

/// One mesh file the JSON lists.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct JsonMesh {
    pub uri: String,
    pub lod: Option<u32>,
    pub tris: Option<u64>,
    /// `original`, `high`, `lod`, ...
    pub kind: String,
}

/// One texture map the JSON lists.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct JsonMap {
    pub uri: String,
    /// `albedo`, `normal`, `roughness`, ...
    pub kind: String,
    /// Edge length in texels, when stated (`8192x8192`, `8K`, `8192`).
    pub resolution: Option<u32>,
}

/// What the export JSON said, reduced to what the importer uses.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MegascansMeta {
    pub id: String,
    pub name: String,
    pub kind: Option<AssetKind>,
    pub tags: Vec<String>,
    pub tileable: Option<bool>,
    /// Real-world size of the scanned area (surfaces, decals) in metres.
    pub size_m: Option<[f32; 2]>,
    /// Real-world object height in metres (3D assets), when stated.
    pub height_m: Option<f32>,
    pub meshes: Vec<JsonMesh>,
    pub maps: Vec<JsonMap>,
}

fn text(v: &Value) -> String {
    match v {
        Value::String(s) => s.trim().to_owned(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        _ => String::new(),
    }
}

fn number(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => numbers_in(s).first().copied(),
        _ => None,
    }
}

fn boolean(v: &Value) -> Option<bool> {
    match v {
        Value::Bool(b) => Some(*b),
        Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
            "true" | "yes" | "1" => Some(true),
            "false" | "no" | "0" => Some(false),
            _ => None,
        },
        Value::Number(n) => n.as_f64().map(|x| x != 0.0),
        _ => None,
    }
}

/// The decimal numbers in a string, in order (`2x2 m` -> 2, 2; a comma between
/// digits is a decimal point: `1,5 x 0,75` -> 1.5, 0.75).
pub fn numbers_in(s: &str) -> Vec<f64> {
    let chars: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_ascii_digit() || (chars[i] == '.' && chars.get(i + 1).is_some_and(|c| c.is_ascii_digit())) {
            let mut run = String::new();
            let mut dot = false;
            while i < chars.len() {
                let c = chars[i];
                if c.is_ascii_digit() {
                    run.push(c);
                } else if (c == '.' || c == ',') && !dot && chars.get(i + 1).is_some_and(|n| n.is_ascii_digit()) {
                    dot = true;
                    run.push('.');
                } else {
                    break;
                }
                i += 1;
            }
            if let Ok(x) = run.parse::<f64>() {
                out.push(x);
            }
        } else {
            i += 1;
        }
    }
    out
}

/// A real-world size like `2x2`, `2 x 2 m`, `1.5m x 1.5m`, `50 cm x 50 cm`, `3`
/// as `[w, h]` metres. One number means a square. Units: m (default), cm, mm,
/// ft, in; the unit word applies to the whole string.
pub fn parse_size_m(s: &str) -> Option<[f32; 2]> {
    let nums = numbers_in(s);
    let (w, h) = match nums.as_slice() {
        [] => return None,
        [a] => (*a, *a),
        [a, b, ..] => (*a, *b),
    };
    if !(w > 0.0 && h > 0.0) {
        return None;
    }
    let lower = s.to_ascii_lowercase();
    let unit = if lower.contains("cm") {
        0.01
    } else if lower.contains("mm") {
        0.001
    } else if lower.contains("ft") || lower.contains("feet") || lower.contains('\'') {
        0.3048
    } else if lower.contains(" in") || lower.contains("inch") || lower.contains('"') {
        0.0254
    } else {
        1.0
    };
    let (w, h) = ((w * unit) as f32, (h * unit) as f32);
    // Reject nonsense (a 10 km "tile").
    ((0.001..=1000.0).contains(&w) && (0.001..=1000.0).contains(&h)).then_some([w, h])
}

fn resolution_of(v: &Value) -> Option<u32> {
    let s = text(v);
    if s.is_empty() {
        return None;
    }
    // `8192x8192`, `8K`, `8192`.
    let first = s.split(['x', 'X', '*']).next().unwrap_or(&s).trim();
    crate::tex_roles::parse_resolution(first).or_else(|| first.parse::<u32>().ok().filter(|n| *n >= 16))
}

fn items(v: &Value) -> impl Iterator<Item = &Value> {
    v.as_array().into_iter().flatten()
}

/// Parse an export JSON. Returns `None` when the text is not JSON or does not
/// look like a Megascans export (none of `id`, `meshes`, `maps`,
/// `semanticTags`).
pub fn parse_megascans_json(json: &str) -> Option<MegascansMeta> {
    let root: Value = serde_json::from_str(json).ok()?;
    let obj = root.as_object()?;
    if !["meshes", "maps", "semanticTags", "id"].iter().any(|k| obj.contains_key(*k)) {
        return None;
    }
    let get = |k: &str| obj.get(k).unwrap_or(&Value::Null).clone();
    let mut m = MegascansMeta { id: text(&get("id")), ..Default::default() };
    let tags = get("semanticTags");
    m.name = text(&tags["name"]);
    if m.name.is_empty() {
        m.name = text(&get("name"));
    }
    m.kind = AssetKind::from_word(&text(&tags["asset_type"]))
        .or_else(|| AssetKind::from_word(&text(&get("type"))))
        // Bridge `categories` start with the type (`["surface", "ground"]`).
        .or_else(|| items(&get("categories")).next().and_then(|c| AssetKind::from_word(&text(c))));
    m.tileable = boolean(&tags["tileable"]);
    for t in items(&get("tags")) {
        let t = text(t);
        if !t.is_empty() {
            m.tags.push(t);
        }
    }
    for e in items(&get("meta")) {
        let key = text(&e["key"]).to_ascii_lowercase();
        let value = &e["value"];
        match key.as_str() {
            "scanarea" | "physicalsize" | "size" | "tilesize" => {
                if m.size_m.is_none() {
                    m.size_m = parse_size_m(&text(value));
                }
            }
            "height" => m.height_m = m.height_m.or_else(|| number(value).map(|h| h as f32)),
            "tileable" => m.tileable = m.tileable.or(boolean(value)),
            _ => {}
        }
    }
    for e in items(&get("meshes")) {
        let kind = text(&e["type"]);
        let tris = number(&e["tris"]).map(|t| t as u64);
        for u in items(&e["uris"]) {
            let uri = text(&u["uri"]);
            if uri.is_empty() {
                continue;
            }
            let lod = number(&u["lod"]).map(|l| l as u32).or_else(|| lod_of_name(&uri));
            m.meshes.push(JsonMesh { uri, lod, tris, kind: kind.clone() });
        }
    }
    for e in items(&get("maps")) {
        let uri = text(&e["uri"]);
        if uri.is_empty() {
            continue;
        }
        let kind = {
            let t = text(&e["type"]);
            if t.is_empty() { text(&e["name"]) } else { t }
        };
        if m.size_m.is_none() {
            m.size_m = parse_size_m(&text(&e["physicalSize"]));
        }
        m.maps.push(JsonMap { uri, kind, resolution: resolution_of(&e["resolution"]) });
    }
    if m.size_m.is_none() {
        m.size_m = parse_size_m(&text(&get("physicalSize")));
    }
    Some(m)
}

/// `LOD3` in a file name -> 3 (case-insensitive; `LOD10` is ten, not one).
pub fn lod_of_name(name: &str) -> Option<u32> {
    let lower = name.to_ascii_lowercase();
    let mut best = None;
    let mut from = 0;
    while let Some(p) = lower[from..].find("lod") {
        let at = from + p + 3;
        let digits: String = lower[at..].chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(n) = digits.parse::<u32>() {
            best = Some(n);
        }
        from = at;
    }
    best
}

/// Read and parse a JSON file; `None` for anything unreadable or not an export.
pub fn read_megascans_json(path: &Path) -> Option<MegascansMeta> {
    // Export JSONs are a few KB; refuse a huge file that is surely something else.
    if std::fs::metadata(path).ok()?.len() > 8 << 20 {
        return None;
    }
    parse_megascans_json(&std::fs::read_to_string(path).ok()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SURFACE: &str = r#"{
        "id": "udlladln", "version": 3, "unknownFuture": {"a": [1,2,3]},
        "semanticTags": {"asset_type": "surface", "name": "Slate Quarry Floor", "tileable": true, "colors": ["gray"]},
        "tags": ["slate", "quarry"],
        "meta": [
            {"key": "scanArea", "name": "Scan Area", "value": "2x2 m"},
            {"key": "tileable", "value": "true"}
        ],
        "maps": [
            {"type": "albedo", "uri": "udlladln_8K_Albedo.jpg", "resolution": "8192x8192", "physicalSize": "2x2"},
            {"type": "normal", "uri": "udlladln_8K_Normal.jpg", "resolution": 8192}
        ],
        "meshes": []
    }"#;

    #[test]
    fn parses_a_surface_export() {
        let m = parse_megascans_json(SURFACE).unwrap();
        assert_eq!((m.id.as_str(), m.name.as_str(), m.kind, m.tileable), ("udlladln", "Slate Quarry Floor", Some(AssetKind::Surface), Some(true)));
        assert_eq!(m.size_m, Some([2.0, 2.0]));
        assert_eq!(m.maps.len(), 2);
        assert_eq!(m.maps[0].resolution, Some(8192));
        assert_eq!(m.maps[1].resolution, Some(8192));
        assert_eq!(m.tags, vec!["slate", "quarry"]);
    }

    #[test]
    fn parses_a_3d_asset_with_lods_and_odd_types() {
        let json = r#"{
            "id": "xb5ebhf", "categories": ["3d asset", "rock"], "name": "Granite Boulder",
            "meshes": [
                {"type": "original", "tris": "120000", "uris": [{"uri": "xb5ebhf_LOD0.fbx", "lod": "0"}]},
                {"type": "lod", "uris": [{"uri": "xb5ebhf_LOD2.fbx"}, {"uri": "xb5ebhf_LOD10.fbx", "lod": 10}, {"uri": ""}]}
            ],
            "meta": [{"key": "height", "value": "1.25 m"}, {"key": "scanArea", "value": null}],
            "maps": "not an array"
        }"#;
        let m = parse_megascans_json(json).unwrap();
        assert_eq!(m.kind, Some(AssetKind::Model3d));
        assert_eq!(m.name, "Granite Boulder");
        assert_eq!(m.height_m, Some(1.25));
        let lods: Vec<_> = m.meshes.iter().map(|x| x.lod).collect();
        assert_eq!(lods, vec![Some(0), Some(2), Some(10)]);
        assert_eq!(m.meshes[0].tris, Some(120000));
        assert!(m.maps.is_empty());
    }

    #[test]
    fn rejects_non_exports_and_survives_garbage() {
        assert!(parse_megascans_json("not json").is_none());
        assert!(parse_megascans_json("[1,2]").is_none());
        assert!(parse_megascans_json(r#"{"foo": 1}"#).is_none());
        // Wrong types everywhere still parse to an (empty) export, never panic.
        let m = parse_megascans_json(r#"{"id": 5, "semanticTags": 3, "meta": {"a": 1}, "meshes": [5, {"uris": 4}], "maps": [null, {"uri": 7}]}"#).unwrap();
        assert_eq!(m.id, "5");
        assert_eq!(m.maps.len(), 1);
    }

    #[test]
    fn kind_words() {
        for (w, k) in [("3D asset", AssetKind::Model3d), ("3d", AssetKind::Model3d), ("3D plant", AssetKind::Plant), ("3dplant", AssetKind::Plant), ("Surface", AssetKind::Surface), ("atlas", AssetKind::Atlas), ("Decal", AssetKind::Decal), ("imperfection", AssetKind::Imperfection), ("Brush", AssetKind::Brush)] {
            assert_eq!(AssetKind::from_word(w), Some(k), "{w}");
        }
        assert_eq!(AssetKind::from_word("something"), None);
    }

    #[test]
    fn real_world_size_units() {
        assert_eq!(parse_size_m("2x2"), Some([2.0, 2.0]));
        assert_eq!(parse_size_m("2 x 2 m"), Some([2.0, 2.0]));
        assert_eq!(parse_size_m("1.5m x 0.75m"), Some([1.5, 0.75]));
        assert_eq!(parse_size_m("1,5 x 1,5"), Some([1.5, 1.5]));
        assert_eq!(parse_size_m("50 cm x 25 cm"), Some([0.5, 0.25]));
        assert_eq!(parse_size_m("3"), Some([3.0, 3.0]));
        assert_eq!(parse_size_m("none"), None);
        assert_eq!(parse_size_m("0 x 2"), None);
        assert_eq!(parse_size_m("5000000 x 1"), None);
    }

    #[test]
    fn lod_numbers_are_numeric() {
        assert_eq!(lod_of_name("x_LOD0.fbx"), Some(0));
        assert_eq!(lod_of_name("x_VarB_lod10.fbx"), Some(10));
        assert_eq!(lod_of_name("x.fbx"), None);
        assert_eq!(lod_of_name("gold_LODe.fbx"), None);
    }
}
