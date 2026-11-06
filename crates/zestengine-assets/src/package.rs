//! `.zestenginepak` content packages: user-generated worlds, avatars and props
//! that a running game loads without recompiling (see `docs/user-content.md`).
//!
//! A package is a zip archive:
//!
//! ```text
//! manifest.ron                  PackageManifest (kind, metadata, file table, hash)
//! files/<project-relative path> the entry .scene plus every asset it references
//! texcache/<path>.<srgb>.cbc    import-cache blobs for material textures (BC on desktop)
//! geocache/<model path>.zgeo    cluster DAGs of the models' big meshes (format 2; see `geo_cache`)
//! atlascache/<atlas path>.catlas  packed sprite atlases whose member textures are all in the package (see `sprite_atlas`)
//! thumbnail.png                 optional preview image
//! ```
//!
//! Only data is ever read from a package: no native code, no precompiled
//! shader binaries (shader sources are compiled by the engine at load). On
//! extraction every RON asset is rewritten so references to packaged files
//! become absolute paths into the extraction directory, absolute / `..` paths
//! pointing elsewhere are blanked, and unknown components are dropped. The
//! extraction directory is keyed by the verified content hash.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail};
use zestengine_core::ContentKind;
use serde::{Deserialize, Serialize};

use crate::{MaterialRef, MaterialTextures, ObjectSource, SceneFile};

pub const PACKAGE_EXTENSION: &str = "zestenginepak";
/// Archive layout version (bump on incompatible layout changes). 2 adds the
/// `geocache/` entries (cluster DAGs); version 1 packages still open and have
/// their DAGs built locally on first load.
pub const PACKAGE_FORMAT_VERSION: u32 = 2;
/// Asset format generation (scene / material / component data). A package
/// built by a newer engine generation is refused with a clear error.
pub const ENGINE_CONTENT_FORMAT: u32 = 1;

const MANIFEST_NAME: &str = "manifest.ron";
const FILES_PREFIX: &str = "files/";
const TEXCACHE_PREFIX: &str = "texcache/";
const GEOCACHE_PREFIX: &str = "geocache/";
const ATLASCACHE_PREFIX: &str = "atlascache/";
const THUMBNAIL_NAME: &str = "thumbnail.png";
const EXTRACTED_MARKER: &str = "extracted.ron";
const INDEX_NAME: &str = "sources.ron";
const AVATAR_DESCRIPTOR: &str = <zestengine_core::AvatarDescriptor as zestengine_core::TypedComponent>::NAME;

/// File types a package may carry. Anything else is skipped at build and
/// dropped at load (defence in depth: nothing in a package is ever executed).
pub const ALLOWED_EXTENSIONS: &[&str] = &[
    "scene", "material", "prefab", "anim", "animctrl", "postfx", "physmat", "audiofx", "audiosettings", "waterwaves", "frag", "ron", "loc", "meta", "lmuv", "gltf",
    "glb", "vrm", "bin", "fbx", "png", "jpg", "jpeg", "tga", "bmp", "exr", "hdr", "gif", "webp", "wav", "ogg", "flac",
    "mp3", "lightmap", "lightdata", "occlusion", "reflprobe", "audiobake", "wasm", "spriteatlas",
];
const SCENE_SIDECARS: &[&str] = &["lightmap", "lightdata", "occlusion", "reflprobe", "audiobake"];
const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "tga", "bmp", "exr", "hdr", "gif", "webp"];
/// Text files that are not RON (never path-rewritten).
const NON_RON_TEXT: &[&str] = &["gltf", "frag"];
/// Already-compressed formats stored without deflate.
const STORED_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp", "gif", "ogg", "mp3", "flac"];

// ------------------------------------------------------------------ manifest

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PackageStats {
    /// Rendered triangles of the entry scene (per placement).
    pub triangles: u64,
    /// GPU texture memory (compressed mips + model-embedded textures).
    pub texture_bytes: u64,
    /// Sum of the packaged file sizes (uncompressed).
    pub file_bytes: u64,
    pub objects: u32,
    pub shaders: u32,
    /// Material slots, skeleton joints and blendshape targets of the entry
    /// scene's models (feed the performance rank; absent in older packages).
    #[serde(default)]
    pub materials: u32,
    #[serde(default)]
    pub bones: u32,
    #[serde(default)]
    pub blendshapes: u32,
    /// Clusters and bytes of the packaged cluster DAGs (absent in older packages).
    #[serde(default)]
    pub clusters: u64,
    #[serde(default)]
    pub dag_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FileRecord {
    /// Archive path (`files/…`, `texcache/…` or `thumbnail.png`).
    pub path: String,
    pub size: u64,
    /// SHA-256 (hex) of the file bytes.
    pub hash: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PackageManifest {
    pub format_version: u32,
    pub engine_format: u32,
    pub kind: ContentKind,
    pub name: String,
    pub author: String,
    pub version: String,
    #[serde(default)]
    pub description: String,
    /// SHA-256 over the file table (paths, sizes, file hashes): the package id.
    pub content_hash: String,
    /// Package-relative path of the entry scene (under `files/`).
    pub entry: String,
    /// Component kinds the content uses (all must exist in the loading game).
    pub components: Vec<String>,
    #[serde(default)]
    pub thumbnail: Option<String>,
    pub stats: PackageStats,
    pub files: Vec<FileRecord>,
}

impl PackageManifest {
    /// Short id for logs / UI (first 12 hex chars of the content hash).
    pub fn short_id(&self) -> &str {
        &self.content_hash[..self.content_hash.len().min(12)]
    }
}

// -------------------------------------------------------------------- limits

/// Resource limits for untrusted content, enforced when building (as errors)
/// and again when loading (on the real data, not the manifest's claims).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContentLimits {
    pub max_package_bytes: u64,
    pub max_unpacked_bytes: u64,
    pub max_files: usize,
    pub max_file_bytes: u64,
    pub max_triangles: u64,
    pub max_texture_bytes: u64,
    pub max_texture_dim: u32,
    pub max_objects: usize,
    pub max_shaders: usize,
    /// Synced avatar-parameter bit budget (VRChat-like 256; only Avatars use it).
    #[serde(default = "default_param_bits")]
    pub max_param_bits: u32,
    /// Bytes of cluster DAG data a package may carry (geometry cooked by
    /// `geo_cache`; the DAG of a mesh is a little under its source size).
    #[serde(default = "default_dag_bytes")]
    pub max_dag_bytes: u64,
    /// Largest `.wasm` component module (the sandbox's untrusted size cap).
    #[serde(default = "default_wasm_bytes")]
    pub max_wasm_bytes: u64,
    /// Most `.wasm` modules a package may carry.
    #[serde(default = "default_wasm_modules")]
    pub max_wasm_modules: usize,
    /// Whether event bindings may call the actions of the package's modules
    /// (only the ones their manifest flags `ugc_safe`).
    #[serde(default = "default_true")]
    pub allow_wasm_actions: bool,
}

const MIB: u64 = 1024 * 1024;

fn default_param_bits() -> u32 {
    256
}

fn default_dag_bytes() -> u64 {
    256 * MIB
}

fn default_wasm_bytes() -> u64 {
    MIB
}

fn default_wasm_modules() -> usize {
    16
}

fn default_true() -> bool {
    true
}

impl ContentLimits {
    pub fn for_kind(kind: ContentKind) -> Self {
        match kind {
            ContentKind::World => Self {
                max_package_bytes: 512 * MIB,
                max_unpacked_bytes: 1024 * MIB,
                max_files: 4096,
                max_file_bytes: 256 * MIB,
                max_triangles: 5_000_000,
                max_texture_bytes: 1024 * MIB,
                max_texture_dim: 8192,
                max_objects: 50_000,
                max_shaders: 32,
                max_param_bits: 256,
                max_dag_bytes: 512 * MIB,
                max_wasm_bytes: MIB,
                max_wasm_modules: 16,
                allow_wasm_actions: true,
            },
            ContentKind::Avatar => Self {
                max_package_bytes: 64 * MIB,
                max_unpacked_bytes: 200 * MIB,
                max_files: 512,
                max_file_bytes: 64 * MIB,
                max_triangles: 150_000,
                max_texture_bytes: 150 * MIB,
                max_texture_dim: 4096,
                max_objects: 2_000,
                max_shaders: 8,
                max_param_bits: 256,
                max_dag_bytes: 48 * MIB,
                // Avatars carry no logic.
                max_wasm_bytes: MIB,
                max_wasm_modules: 0,
                allow_wasm_actions: false,
            },
            ContentKind::Prop => Self {
                max_package_bytes: 64 * MIB,
                max_unpacked_bytes: 200 * MIB,
                max_files: 512,
                max_file_bytes: 64 * MIB,
                max_triangles: 250_000,
                max_texture_bytes: 128 * MIB,
                max_texture_dim: 4096,
                max_objects: 2_000,
                max_shaders: 8,
                max_param_bits: 256,
                max_dag_bytes: 48 * MIB,
                max_wasm_bytes: 512 * 1024,
                max_wasm_modules: 4,
                allow_wasm_actions: true,
            },
        }
    }

    /// The loosest limit of each kind: the bound before the manifest is read.
    pub fn outer() -> Self {
        Self::for_kind(ContentKind::World)
    }

    /// Check measured stats; the error names every exceeded limit.
    pub fn check_stats(&self, s: &PackageStats) -> std::result::Result<(), String> {
        let mut over = Vec::new();
        if s.triangles > self.max_triangles {
            over.push(format!("{} triangles (limit {})", s.triangles, self.max_triangles));
        }
        if s.texture_bytes > self.max_texture_bytes {
            over.push(format!(
                "{:.1} MiB of textures (limit {:.1} MiB)",
                s.texture_bytes as f64 / MIB as f64,
                self.max_texture_bytes as f64 / MIB as f64
            ));
        }
        if s.file_bytes > self.max_unpacked_bytes {
            over.push(format!("{} bytes of files (limit {})", s.file_bytes, self.max_unpacked_bytes));
        }
        if s.objects as usize > self.max_objects {
            over.push(format!("{} objects (limit {})", s.objects, self.max_objects));
        }
        if s.shaders as usize > self.max_shaders {
            over.push(format!("{} custom shaders (limit {})", s.shaders, self.max_shaders));
        }
        if s.dag_bytes > self.max_dag_bytes {
            over.push(format!("{} bytes of cluster data (limit {})", s.dag_bytes, self.max_dag_bytes));
        }
        if over.is_empty() { Ok(()) } else { Err(format!("content exceeds limits: {}", over.join(", "))) }
    }
}

// ------------------------------------------------------------- RON strings

/// Rewrite every string literal in RON text through `f` (None keeps it).
/// Strings that themselves contain RON (serialized component data) are
/// visited recursively, so paths inside component fields are found too.
pub fn rewrite_ron_strings(text: &str, f: &mut dyn FnMut(&str) -> Option<String>) -> String {
    rewrite_inner(text, f, 0)
}

/// Call `f` on every string literal (recursively, see [`rewrite_ron_strings`]).
pub fn visit_ron_strings(text: &str, f: &mut dyn FnMut(&str)) {
    rewrite_inner(
        text,
        &mut |s| {
            f(s);
            None
        },
        0,
    );
}

fn rewrite_inner(text: &str, f: &mut dyn FnMut(&str) -> Option<String>, depth: u32) -> String {
    let b = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let (mut i, mut last) = (0usize, 0usize);
    while i < b.len() {
        let c = b[i];
        let raw = c == b'r'
            && matches!(b.get(i + 1), Some(b'"') | Some(b'#'))
            && (i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_'));
        if c == b'"' || raw {
            let parsed = if raw { parse_raw_string(text, i + 1) } else { parse_string(text, i + 1) };
            let Some((content, end)) = parsed else { break };
            let replacement = match f(&content) {
                Some(new) => Some(new),
                None if depth < 4 && content.contains('"') => {
                    let inner = rewrite_inner(&content, f, depth + 1);
                    (inner != content).then_some(inner)
                }
                None => None,
            };
            if let Some(r) = replacement {
                out.push_str(&text[last..i]);
                out.push_str(&escape_ron(&r));
                last = end;
            }
            i = end;
        } else if c == b'\'' {
            // Char literal ('x', '\n', '"').
            let mut j = i + 1;
            if b.get(j) == Some(&b'\\') {
                j += 2;
            }
            while j < b.len() && b[j] != b'\'' {
                j += 1;
            }
            i = j + 1;
        } else if c == b'/' && b.get(i + 1) == Some(&b'/') {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if c == b'/' && b.get(i + 1) == Some(&b'*') {
            i += 2;
            while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                i += 1;
            }
            i += 2;
        } else {
            i += 1;
        }
    }
    out.push_str(&text[last.min(text.len())..]);
    out
}

/// Parse a quoted string starting after the opening quote. Returns the
/// unescaped content and the byte index after the closing quote.
fn parse_string(text: &str, start: usize) -> Option<(String, usize)> {
    let mut out = String::new();
    let mut chars = text[start..].char_indices();
    while let Some((k, ch)) = chars.next() {
        match ch {
            '"' => return Some((out, start + k + 1)),
            '\\' => {
                let (_, e) = chars.next()?;
                match e {
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    '0' => out.push('\0'),
                    '\\' => out.push('\\'),
                    '"' => out.push('"'),
                    '\'' => out.push('\''),
                    'x' => {
                        let h: String = (0..2).filter_map(|_| chars.next().map(|(_, c)| c)).collect();
                        out.push(u8::from_str_radix(&h, 16).ok()? as char);
                    }
                    'u' => {
                        let (_, open) = chars.next()?;
                        if open != '{' {
                            return None;
                        }
                        let mut h = String::new();
                        loop {
                            let (_, c) = chars.next()?;
                            if c == '}' {
                                break;
                            }
                            h.push(c);
                        }
                        out.push(char::from_u32(u32::from_str_radix(&h, 16).ok()?)?);
                    }
                    '\n' => {
                        // Line continuation: skip leading whitespace on the next line.
                        let rest = &text[start + k + 2..];
                        let skip = rest.len() - rest.trim_start().len();
                        for _ in 0..rest[..skip].chars().count() {
                            chars.next();
                        }
                    }
                    _ => return None,
                }
            }
            c => out.push(c),
        }
    }
    None
}

/// Parse `r"…"` / `r#"…"#` starting after the `r`.
fn parse_raw_string(text: &str, start: usize) -> Option<(String, usize)> {
    let b = text.as_bytes();
    let mut i = start;
    let mut hashes = 0;
    while b.get(i) == Some(&b'#') {
        hashes += 1;
        i += 1;
    }
    if b.get(i) != Some(&b'"') {
        return None;
    }
    let body = i + 1;
    let close: String = std::iter::once('"').chain(std::iter::repeat_n('#', hashes)).collect();
    let end = text[body..].find(&close)? + body;
    Some((text[body..end].to_owned(), end + close.len()))
}

fn escape_ron(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{{{:x}}}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

// --------------------------------------------------------------------- paths

/// Normalize a candidate project-relative path (forward slashes, no `.`/`..`,
/// not absolute). None when the string can't be a safe relative path.
pub fn normalize_rel(s: &str) -> Option<String> {
    if s.is_empty() || s.len() > 512 || s.trim() != s || s.contains(['\0', '\n', '\r']) {
        return None;
    }
    let s = s.replace('\\', "/");
    let s = s.strip_prefix("./").unwrap_or(&s);
    if s.starts_with('/') || is_drive_path(s) || s.contains(':') {
        return None;
    }
    let mut parts = Vec::new();
    for part in s.split('/') {
        match part {
            "" | "." | ".." => return None,
            p => parts.push(p),
        }
    }
    Some(parts.join("/"))
}

fn is_drive_path(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':'
}

/// A string that addresses the file system outside the package: absolute, a
/// drive path, or climbing out with `..`.
fn escapes_package(s: &str) -> bool {
    if s.len() < 2 || s.contains(char::is_whitespace) {
        return false;
    }
    let norm = s.replace('\\', "/");
    norm.starts_with('/') || norm.starts_with("~/") || is_drive_path(&norm) || norm.split('/').any(|p| p == "..")
}

fn ext_of(path: &str) -> String {
    Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default()
}

fn ext_allowed(path: &str) -> bool {
    ALLOWED_EXTENSIONS.contains(&ext_of(path).as_str())
}

fn is_ron_text(path: &str) -> bool {
    let ext = ext_of(path);
    ext_allowed(path)
        && !NON_RON_TEXT.contains(&ext.as_str())
        && matches!(
            ext.as_str(),
            "scene" | "material" | "prefab" | "anim" | "animctrl" | "postfx" | "physmat" | "audiofx" | "audiosettings" | "waterwaves" | "ron" | "loc" | "meta"
        )
}

fn excluded_dir(rel: &str) -> bool {
    rel.split('/').any(|p| matches!(p, ".zestengine_texcache" | ".zestengine_modelcache" | ".zestengine_geocache" | "target" | ".git" | "build"))
}

// ------------------------------------------------------------- dependencies

/// Every project file the given RON text references, transitively (materials
/// -> textures, scenes -> models and sidecars, models -> meta sidecars and
/// external buffers / textures). Returns sorted project-relative paths and
/// warnings for references that were skipped.
pub fn collect_dependencies(root: &Path, entry_rel: &str, entry_text: &str) -> (BTreeSet<String>, Vec<String>) {
    let mut files = BTreeSet::new();
    let mut warnings = Vec::new();
    let mut queue = VecDeque::new();
    files.insert(entry_rel.to_owned());
    let consider = |s: &str, files: &mut BTreeSet<String>, queue: &mut VecDeque<String>, warnings: &mut Vec<String>| {
        // `@<id>` slots: the file the id names, plus the sidecar that maps it.
        let resolved;
        let s = if zestengine_core::asset_ref::is_ref(s) {
            match zestengine_core::asset_ref::resolve(root, s) {
                Some(r) => {
                    let meta = crate::meta_path(&r).to_string_lossy().replace('\\', "/");
                    if root.join(&meta).is_file() {
                        files.insert(meta);
                    }
                    resolved = r;
                    resolved.as_str()
                }
                None => {
                    warnings.push(format!("missing asset {s}"));
                    return;
                }
            }
        } else {
            s
        };
        let Some(rel) = normalize_rel(s) else { return };
        if excluded_dir(&rel) || files.contains(&rel) || !root.join(&rel).is_file() {
            return;
        }
        if !ext_allowed(&rel) {
            warnings.push(format!("skipping {rel}: file type not allowed in content packages"));
            files.insert(rel);
            return;
        }
        files.insert(rel.clone());
        queue.push_back(rel);
    };
    visit_ron_strings(entry_text, &mut |s| consider(s, &mut files, &mut queue, &mut warnings));
    add_scene_sidecars(root, entry_rel, &mut |s| consider(s, &mut files, &mut queue, &mut warnings));
    while let Some(rel) = queue.pop_front() {
        let abs = root.join(&rel);
        let ext = ext_of(&rel);
        if is_ron_text(&rel) {
            match std::fs::read_to_string(&abs) {
                Ok(text) => visit_ron_strings(&text, &mut |s| consider(s, &mut files, &mut queue, &mut warnings)),
                Err(e) => warnings.push(format!("reading {rel}: {e}")),
            }
        }
        if ext == "scene" {
            add_scene_sidecars(root, &rel, &mut |s| consider(s, &mut files, &mut queue, &mut warnings));
        }
        if matches!(ext.as_str(), "gltf" | "glb" | "vrm" | "fbx") {
            let meta = crate::meta_path(&rel).to_string_lossy().into_owned();
            consider(&meta, &mut files, &mut queue, &mut warnings);
            let lmuv = crate::lightmap_uv_marker_path(&rel).to_string_lossy().into_owned();
            consider(&lmuv, &mut files, &mut queue, &mut warnings);
            let dir = Path::new(&rel).parent().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
            let join = |r: &str| if dir.is_empty() { r.to_owned() } else { format!("{dir}/{r}") };
            let external = if ext == "fbx" { fbx_external_files(&abs) } else { gltf_uris(&abs) };
            for r in external {
                match normalize_rel(&r) {
                    Some(r) => consider(&join(&r), &mut files, &mut queue, &mut warnings),
                    None => warnings.push(format!("{rel}: external file {r:?} is outside the model folder; skipped")),
                }
            }
        }
    }
    files.retain(|f| ext_allowed(f));
    (files, warnings)
}

fn add_scene_sidecars(root: &Path, scene_rel: &str, add: &mut dyn FnMut(&str)) {
    let _ = root;
    for ext in SCENE_SIDECARS {
        let side = Path::new(scene_rel).with_extension(ext);
        add(&side.to_string_lossy());
    }
}

/// Non-embedded buffer / image URIs of a glTF (relative to its folder).
fn gltf_uris(path: &Path) -> Vec<String> {
    let Ok(bytes) = std::fs::read(path) else { return Vec::new() };
    gltf_uris_from_bytes(&bytes).unwrap_or_default()
}

fn gltf_uris_from_bytes(bytes: &[u8]) -> Option<Vec<String>> {
    let doc = gltf::Gltf::from_slice_without_validation(bytes).ok()?;
    let mut out = Vec::new();
    for b in doc.buffers() {
        if let gltf::buffer::Source::Uri(uri) = b.source() {
            out.push(uri.to_owned());
        }
    }
    for img in doc.images() {
        if let gltf::image::Source::Uri { uri, .. } = img.source() {
            out.push(uri.to_owned());
        }
    }
    out.retain(|u| !u.starts_with("data:"));
    Some(out.into_iter().map(|u| percent_decode(&u)).collect())
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%'
            && i + 2 < b.len()
            && let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16)
        {
            out.push(v);
            i += 3;
            continue;
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Texture files an FBX references by relative name (the loader resolves them
/// against the FBX's folder).
fn fbx_external_files(path: &Path) -> Vec<String> {
    let Some(p) = path.to_str() else { return Vec::new() };
    let opts = ufbx::LoadOpts {
        load_external_files: false,
        ignore_missing_external_files: true,
        ..Default::default()
    };
    let Ok(fbx) = ufbx::load_file(p, opts) else { return Vec::new() };
    let base = path.parent().unwrap_or(Path::new("."));
    let mut out = Vec::new();
    for tex in fbx.textures.iter() {
        if !tex.content.is_empty() {
            continue;
        }
        for name in [tex.relative_filename.as_ref(), tex.filename.as_ref()] {
            if !name.is_empty() && base.join(name).is_file() {
                out.push(name.replace('\\', "/"));
                break;
            }
        }
    }
    out
}

/// (texture, srgb) pairs a material's slots load; colour slots are sRGB.
pub fn material_texture_slots(t: &MaterialTextures) -> Vec<(PathBuf, bool)> {
    let slots: [(&Option<PathBuf>, bool); 21] = [
        (&t.albedo, true),
        (&t.normal, false),
        (&t.orm, false),
        (&t.emission, true),
        (&t.opacity, false),
        (&t.emission_mask, false),
        (&t.matcap[0], true),
        (&t.matcap_mask[0], false),
        (&t.matcap[1], true),
        (&t.matcap_mask[1], false),
        (&t.matcap[2], true),
        (&t.matcap_mask[2], false),
        (&t.ao, false),
        (&t.roughness, false),
        (&t.metallic, false),
        (&t.displacement, false),
        (&t.detail_albedo, true),
        (&t.detail_normal, false),
        (&t.detail_orm, false),
        (&t.detail_mask, false),
        (&t.translucency, false),
    ];
    slots.into_iter().filter_map(|(p, s)| p.clone().map(|p| (p, s))).collect()
}

/// (texture, srgb) pairs loaded through material slots by `scenes` (already
/// parsed, with their project-relative paths) and by every `.scene`,
/// `.prefab` and `.material` among `files`, limited to textures that are
/// themselves in `files`. Used by package builds (BC cache blobs) and the
/// Android cook (ASTC caches). Unparseable files add a warning.
pub fn material_texture_refs(
    root: &Path,
    files: &BTreeSet<String>,
    scenes: &[(&str, &SceneFile)],
    warnings: &mut Vec<String>,
) -> BTreeSet<(String, bool)> {
    #[derive(Deserialize)]
    struct PrefabEntries {
        entries: Vec<crate::SceneEntry>,
    }
    let mut refs = BTreeSet::new();
    let mut add_slots = |t: &MaterialTextures| {
        for (p, srgb) in material_texture_slots(t) {
            if let Some(rel) = normalize_rel(&p.to_string_lossy())
                && files.contains(&rel)
            {
                refs.insert((rel, srgb));
            }
        }
    };
    let mut entry_lists: Vec<Vec<crate::SceneEntry>> = Vec::new();
    for rel in files.iter().filter(|r| !scenes.iter().any(|(s, _)| s == r)) {
        match ext_of(rel).as_str() {
            "scene" => match crate::load_scene_file(root.join(rel)) {
                Ok(s) => entry_lists.push(s.entries),
                Err(e) => warnings.push(format!("{rel}: {e:#}")),
            },
            "prefab" => match std::fs::read_to_string(root.join(rel)).map_err(anyhow::Error::from).and_then(|t| {
                ron::from_str::<PrefabEntries>(&t).map_err(anyhow::Error::from)
            }) {
                Ok(p) => entry_lists.push(p.entries),
                Err(e) => warnings.push(format!("{rel}: {e:#}")),
            },
            "material" => match crate::load_material_file(root.join(rel)) {
                Ok(m) => add_slots(&m.textures),
                Err(e) => warnings.push(format!("{rel}: {e:#}")),
            },
            _ => {}
        }
    }
    let entries = scenes.iter().flat_map(|(_, s)| &s.entries).chain(entry_lists.iter().flatten());
    for e in entries {
        for m in crate::renderer_migration::material_refs(e) {
            if let MaterialRef::Inline { textures, .. } = &m {
                add_slots(textures);
            }
        }
    }
    refs
}

/// Asset-root folder an Android / Quest build ships its pre-encoded ASTC
/// caches in, as `<folder>/<texture path>.<srgb>.castc`. Not the hidden
/// per-file `.zestengine_texcache/` because the APK packager drops dot-prefixed
/// paths; [`seed_cooked_texture_caches`] moves them into place on device.
pub const COOKED_TEXCACHE_DIR: &str = "zestengine_texcache";

/// Shipped path of a texture's cooked ASTC cache (see [`COOKED_TEXCACHE_DIR`]).
pub fn cooked_texcache_rel(texture_rel: &str, srgb: bool) -> String {
    format!("{COOKED_TEXCACHE_DIR}/{texture_rel}.{}.{}", srgb as u8, crate::ASTC_CACHE_EXTENSION)
}

/// Inverse of [`cooked_texcache_rel`]: (texture path, srgb).
pub fn parse_cooked_texcache_rel(rel: &str) -> Option<(&str, bool)> {
    let rest = rel.strip_prefix(COOKED_TEXCACHE_DIR)?.strip_prefix('/')?;
    let rest = rest.strip_suffix(crate::ASTC_CACHE_EXTENSION)?.strip_suffix('.')?;
    let (tex, srgb) = rest.rsplit_once('.')?;
    let srgb = match srgb {
        "1" => true,
        "0" => false,
        _ => return None,
    };
    normalize_rel(tex).filter(|t| t == tex)?;
    Some((tex, srgb))
}

/// Install the cooked ASTC caches under `assets_root/`[`COOKED_TEXCACHE_DIR`]
/// into each texture's import cache (re-bound to the extracted file's mtime
/// and size, independent of the device), then delete the shipped copies.
/// Returns (caches installed, warnings).
pub fn seed_cooked_texture_caches(assets_root: &Path) -> (usize, Vec<String>) {
    let dir = assets_root.join(COOKED_TEXCACHE_DIR);
    let mut found = Vec::new();
    collect_files_rel(&dir, &dir, &mut found);
    let (mut seeded, mut warnings) = (0, Vec::new());
    for shipped in found {
        let rel = format!("{COOKED_TEXCACHE_DIR}/{shipped}");
        let Some((tex, srgb)) = parse_cooked_texcache_rel(&rel) else {
            warnings.push(format!("ignoring unknown cooked cache {rel}"));
            continue;
        };
        let result = std::fs::read(dir.join(&shipped))
            .map_err(anyhow::Error::from)
            .and_then(|blob| crate::seed_astc_cache(assets_root.join(tex), srgb, &blob, COOKED_MAX_TEXTURE_DIM));
        match result {
            Ok(_) => seeded += 1,
            Err(e) => warnings.push(format!("cooked texture cache for {tex}: {e:#}")),
        }
    }
    if dir.exists()
        && let Err(e) = std::fs::remove_dir_all(&dir)
    {
        warnings.push(format!("removing {}: {e}", dir.display()));
    }
    (seeded, warnings)
}

/// Largest texture a cooked cache may describe (Vulkan's common 2D limit).
const COOKED_MAX_TEXTURE_DIM: u32 = 16384;

/// `/`-separated paths of every file under `dir`, relative to `base`.
fn collect_files_rel(base: &Path, dir: &Path, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_files_rel(base, &path, out);
        } else if let Ok(rel) = path.strip_prefix(base) {
            out.push(rel.to_string_lossy().replace('\\', "/"));
        }
    }
}

/// Rendered triangles of a scene: each model placement counts its slots.
pub fn count_triangles(file: &SceneFile, models: &HashMap<String, crate::Scene>) -> u64 {
    let mut total = 0u64;
    for e in &file.entries {
        if let ObjectSource::Model { path, mesh, extra_meshes, .. } = &e.source
            && let Some(m) = models.get(path)
        {
            for &i in std::iter::once(mesh).chain(extra_meshes.iter()) {
                if let Some(mesh) = m.meshes.get(i) {
                    total += mesh.indices.len() as u64 / 3;
                }
            }
        }
    }
    total
}

/// Material slots, joints and blendshape targets across `models`.
pub fn model_complexity<'a>(models: impl IntoIterator<Item = &'a crate::Scene>) -> (u32, u32, u32) {
    let mut t = (0u32, 0u32, 0u32);
    for m in models {
        t.0 += m.materials.len() as u32;
        t.1 += m.skeletons.iter().map(|s| s.joints.len() as u32).sum::<u32>();
        t.2 += m.morphs.iter().map(|ts| ts.len() as u32).sum::<u32>();
    }
    t
}

/// Header-only estimate of a texture's GPU size: (width, height, bytes incl. mips).
pub fn estimate_texture_bytes(path: &Path) -> Option<(u32, u32, u64)> {
    let (w, h) = image::image_dimensions(path).ok()?;
    let ext = path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).unwrap_or_default();
    let hdr = matches!(ext.as_str(), "exr" | "hdr");
    let bc = zestengine_render::bc_supported() && w % 4 == 0 && h % 4 == 0 && w >= 4 && h >= 4;
    // ASTC at the finest footprint (4x4) is also 1 B/texel; HDR stays raw there.
    let astc = !zestengine_render::bc_supported() && zestengine_render::astc_supported() && !hdr;
    let per_texel = if bc || astc { 1 } else if hdr { 8 } else { 4 };
    Some((w, h, w as u64 * h as u64 * per_texel * 4 / 3))
}

/// Bytes a model's embedded textures occupy on the GPU (RGBA8 / RGBA16F + mips).
pub fn model_texture_bytes(model: &crate::Scene) -> u64 {
    model.textures.iter().map(|t| t.pixels.len() as u64 * 4 / 3).sum()
}

// ------------------------------------------------------------------- builder

/// How a component kind relates to the games that will load the content.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComponentStatus {
    /// Built into the engine: exists in every game.
    Builtin,
    /// Comes from a project plugin: only games that link it can run it.
    Plugin,
    /// Not registered at all.
    Unknown,
}

pub struct BuildOptions<'a> {
    pub kind: ContentKind,
    pub name: String,
    pub author: String,
    pub version: String,
    pub description: String,
    pub project_root: PathBuf,
    /// The entry scene (World: the scene; Avatar / Prop: the hierarchy as a scene).
    pub scene: SceneFile,
    /// Package-relative path for the entry scene, e.g. `scenes/forest.scene`.
    /// For a World it should be the scene's project path so baked sidecars
    /// next to it are packaged.
    pub entry: String,
    pub thumbnail: Option<Vec<u8>>,
    pub classify: &'a dyn Fn(&str) -> ComponentStatus,
    /// Drop plugin components instead of keeping them (they only work in
    /// games that link the plugin).
    pub strip_plugin_components: bool,
    pub include_texture_cache: bool,
    pub limits: ContentLimits,
}

/// Everything a build would write, plus what validation found.
pub struct PackagePlan {
    pub manifest: PackageManifest,
    /// Archive path -> bytes (sorted).
    pub files: BTreeMap<String, Vec<u8>>,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

impl PackagePlan {
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }
}

/// Gather, validate and measure a package without writing it.
pub fn plan_package(opts: &BuildOptions) -> Result<PackagePlan> {
    let mut warnings = Vec::new();
    let mut errors = Vec::new();
    let root = &opts.project_root;
    let Some(entry) = normalize_rel(&opts.entry).filter(|e| ext_of(e) == "scene") else {
        bail!("entry {:?} must be a relative .scene path", opts.entry);
    };
    if opts.name.trim().is_empty() {
        errors.push("the package needs a name".into());
    }
    let mut scene = opts.scene.clone();
    scene.editor_camera = None;
    scene.collapsed.clear();
    scene.strip_editor_view();
    if scene.entries.is_empty() {
        errors.push("nothing to package: the scene / selection has no objects".into());
    }

    let mut components = BTreeSet::new();
    for e in &mut scene.entries {
        let mut kept = Vec::new();
        for c in std::mem::take(&mut e.components) {
            match (opts.classify)(&c.kind) {
                ComponentStatus::Builtin => {
                    components.insert(c.kind.clone());
                    kept.push(c);
                }
                ComponentStatus::Plugin if !opts.strip_plugin_components => {
                    warnings.push(format!(
                        "{:?} on {:?} is a plugin component: it only works in games that link that plugin (other games drop it)",
                        c.kind, e.name
                    ));
                    components.insert(c.kind.clone());
                    kept.push(c);
                }
                ComponentStatus::Plugin => {
                    warnings.push(format!("dropped plugin component {:?} from {:?}", c.kind, e.name))
                }
                ComponentStatus::Unknown => {
                    warnings.push(format!("dropped unknown component {:?} from {:?}", c.kind, e.name))
                }
            }
        }
        e.components = kept;
    }

    if matches!(opts.kind, ContentKind::Avatar | ContentKind::Prop) {
        let roots = scene.entries.iter().filter(|e| e.parent.is_none()).count();
        if roots != 1 {
            errors.push(format!("{} content needs exactly one root object (found {roots})", opts.kind.label()));
        }
    }
    if opts.kind == ContentKind::Avatar
        && let Some(root_entry) = scene.entries.iter_mut().find(|e| e.parent.is_none())
        && !root_entry.components.iter().any(|c| c.kind == AVATAR_DESCRIPTOR)
    {
        warnings.push("avatar root had no Avatar Descriptor; added a default one (eye height 1.6 m)".into());
        root_entry.components.push(crate::ComponentData::new(AVATAR_DESCRIPTOR, ron::to_string(&zestengine_core::AvatarDescriptor::default())?));
        components.insert(AVATAR_DESCRIPTOR.into());
    }
    // Non-destructive avatar build: authoring components (toggles, merges,
    // menu items, scoped parameters) become the finished parameters, menu and
    // controller on this copy of the scene; generated assets go to
    // `.zest/generated/avatars/<id>/` and are packaged like any other file.
    if opts.kind == ContentKind::Avatar
        && let Some(root_idx) = scene.entries.iter().position(|e| e.parent.is_none())
        && crate::avatar_build::needs_build(&scene, root_idx)
    {
        let env = crate::avatar_build::BuildEnv { project_root: root, write: true, structural: true, budget_cap: opts.limits.max_param_bits };
        let built = crate::avatar_build::build_avatar(&mut scene, root_idx, &env);
        warnings.extend(built.warnings.iter().map(|w| format!("avatar build: {w}")));
        errors.extend(built.errors.iter().map(|e| format!("avatar build: {e}")));
        let present: BTreeSet<&str> = scene.entries.iter().flat_map(|e| e.components.iter().map(|c| c.kind.as_str())).collect();
        components.retain(|k| present.contains(k.as_str()));
        for k in [<zestengine_core::AvatarParameters as zestengine_core::TypedComponent>::NAME,
            <zestengine_core::AvatarMenuData as zestengine_core::TypedComponent>::NAME,
            <zestengine_core::Animator as zestengine_core::TypedComponent>::NAME,
        ] {
            if present.contains(k) {
                components.insert(k.into());
            }
        }
    }

    let scene_text = ron::ser::to_string_pretty(&scene, ron::ser::PrettyConfig::default())?;
    let (deps, dep_warnings) = collect_dependencies(root, &entry, &scene_text);
    warnings.extend(dep_warnings);

    let mut files: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    files.insert(format!("{FILES_PREFIX}{entry}"), scene_text.clone().into_bytes());
    let mut shaders = 0u32;
    let mut wasm_modules = 0usize;
    for rel in deps.iter().filter(|r| **r != entry) {
        let bytes = std::fs::read(root.join(rel)).with_context(|| format!("reading {rel}"))?;
        if bytes.len() as u64 > opts.limits.max_file_bytes {
            errors.push(format!("{rel} is {} bytes (per-file limit {})", bytes.len(), opts.limits.max_file_bytes));
        }
        if ext_of(rel) == "wasm" {
            wasm_modules += 1;
            if wasm_modules > opts.limits.max_wasm_modules {
                if wasm_modules == opts.limits.max_wasm_modules + 1 {
                    errors.push(format!("more than {} WASM modules (limit for this kind)", opts.limits.max_wasm_modules));
                }
            } else if let Err(e) = validate_wasm_module(&bytes, &opts.limits) {
                errors.push(format!("{rel}: {e}"));
            }
        }
        if ext_of(rel) == "frag" {
            shaders += 1;
            if let Err(e) = validate_shader_source(&String::from_utf8_lossy(&bytes)) {
                errors.push(format!("{rel}: {e}"));
            }
        }
        files.insert(format!("{FILES_PREFIX}{rel}"), bytes);
    }

    // Material textures (with the sRGB flag their slot loads them with) for
    // the BC cache and the texture-memory estimate.
    let tex_refs = material_texture_refs(root, &deps, &[(entry.as_str(), &scene)], &mut warnings);
    let mut texture_bytes = 0u64;
    for (rel, srgb) in &tex_refs {
        let abs = root.join(rel);
        let blob = if opts.include_texture_cache {
            match crate::texture_cache_blob(&abs, *srgb) {
                Ok(b) => Some(b),
                Err(e) => {
                    warnings.push(format!("texture cache for {rel}: {e:#}"));
                    None
                }
            }
        } else {
            None
        };
        match blob {
            Some(b) => {
                texture_bytes += b.len() as u64;
                files.insert(format!("{TEXCACHE_PREFIX}{rel}.{}.cbc", *srgb as u8), b);
            }
            None => {
                if let Ok((w, h)) = image::image_dimensions(&abs) {
                    texture_bytes += w as u64 * h as u64 * 4 * 4 / 3;
                    if w.max(h) > opts.limits.max_texture_dim {
                        errors.push(format!("{rel} is {w}x{h} (limit {})", opts.limits.max_texture_dim));
                    }
                }
            }
        }
    }

    // Triangles + embedded textures of the entry scene's models.
    let mut models = HashMap::new();
    let (mut clusters, mut dag_bytes) = (0u64, 0u64);
    for e in &scene.entries {
        if let ObjectSource::Model { path, .. } = &e.source
            && !models.contains_key(path)
        {
            match crate::load_model_with_meta(root.join(path)) {
                Ok(m) => {
                    texture_bytes += model_texture_bytes(&m);
                    // The geometry cook always runs: the cluster DAG of every big mesh ships
                    // with the model (built here, or read from the project's geometry cache).
                    if let Some(rel) = normalize_rel(path)
                        && let Some((geo, blob)) = crate::geo_cache::cook(&root.join(&rel), &m)
                    {
                        if blob.len() as u64 > opts.limits.max_file_bytes {
                            warnings.push(format!("cluster data for {rel} is {} bytes (per-file limit {}); it is rebuilt on load", blob.len(), opts.limits.max_file_bytes));
                        } else {
                            clusters += geo.clusters() as u64;
                            dag_bytes += geo.dag_bytes() as u64;
                            files.insert(format!("{GEOCACHE_PREFIX}{rel}.{}", crate::geo_cache::GEO_EXTENSION), blob);
                        }
                    }
                    models.insert(path.clone(), m);
                }
                Err(e) => errors.push(format!("model {path}: {e:#}")),
            }
        }
    }
    // Sprite atlases whose every member texture is in the package ride along with
    // their packed pages; an atlas with a member left out is not shipped (its
    // sprites then draw from their own textures).
    {
        let db = crate::sprite_atlas::AtlasDb::scan(root);
        let mut shipped_atlas = false;
        for e in db.atlases.iter().filter(|e| e.file.include_in_build && !e.members.is_empty()) {
            if e.members.iter().any(|m| !deps.contains(&m.rel)) || normalize_rel(&e.rel).as_deref() != Some(e.rel.as_str()) {
                continue;
            }
            if let Ok(bytes) = std::fs::read(&e.abs) {
                files.insert(format!("{FILES_PREFIX}{}", e.rel), bytes);
                shipped_atlas = true;
            }
            for m in &e.members {
                let meta = crate::meta_path(&m.rel).to_string_lossy().replace('\\', "/");
                if !files.contains_key(&format!("{FILES_PREFIX}{meta}"))
                    && let Ok(bytes) = std::fs::read(root.join(&meta))
                {
                    files.insert(format!("{FILES_PREFIX}{meta}"), bytes);
                }
            }
        }
        if shipped_atlas && opts.include_texture_cache {
            let (blobs, atlas_warnings) = db.cooked_blobs(crate::Platform::Desktop, &|rel| files.contains_key(&format!("{FILES_PREFIX}{rel}")));
            warnings.extend(atlas_warnings.into_iter().map(|w| format!("sprite atlas {w}")));
            for (rel, blob) in blobs {
                if blob.len() as u64 > opts.limits.max_file_bytes {
                    warnings.push(format!("packed atlas {rel} is {} bytes (per-file limit {}); it is packed on load", blob.len(), opts.limits.max_file_bytes));
                } else {
                    files.insert(format!("{ATLASCACHE_PREFIX}{rel}.{}", crate::sprite_atlas::COOKED_ATLAS_EXTENSION), blob);
                }
            }
        }
    }
    let triangles = count_triangles(&scene, &models);
    let (materials, bones, blendshapes) = model_complexity(models.values());

    let mut thumbnail = None;
    if let Some(png) = &opts.thumbnail {
        files.insert(THUMBNAIL_NAME.into(), png.clone());
        thumbnail = Some(THUMBNAIL_NAME.to_owned());
    }

    let file_bytes: u64 = files.values().map(|b| b.len() as u64).sum();
    let stats = PackageStats {
        triangles,
        texture_bytes,
        file_bytes,
        objects: scene.entries.len() as u32,
        shaders,
        materials,
        bones,
        blendshapes,
        clusters,
        dag_bytes,
    };
    if let Err(e) = opts.limits.check_stats(&stats) {
        errors.push(e);
    }
    if files.len() > opts.limits.max_files {
        errors.push(format!("{} files (limit {})", files.len(), opts.limits.max_files));
    }

    let records: Vec<FileRecord> = files
        .iter()
        .map(|(path, bytes)| FileRecord {
            path: path.clone(),
            size: bytes.len() as u64,
            hash: sha256::hex(&sha256::digest(bytes)),
        })
        .collect();
    let manifest = PackageManifest {
        format_version: PACKAGE_FORMAT_VERSION,
        engine_format: ENGINE_CONTENT_FORMAT,
        kind: opts.kind,
        name: opts.name.trim().to_owned(),
        author: opts.author.trim().to_owned(),
        version: if opts.version.trim().is_empty() { "1.0".into() } else { opts.version.trim().to_owned() },
        description: opts.description.clone(),
        content_hash: content_hash(&records),
        entry,
        components: components.into_iter().collect(),
        thumbnail,
        stats,
        files: records,
    };
    Ok(PackagePlan {
        manifest,
        files,
        warnings,
        errors,
    })
}

/// Write a validated plan as a `.zestenginepak` (temp file + rename).
pub fn write_package(plan: &PackagePlan, out: &Path) -> Result<()> {
    if !plan.errors.is_empty() {
        bail!("package has errors:\n{}", plan.errors.join("\n"));
    }
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = out.with_extension("zestenginepak.tmp");
    {
        let file = std::fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
        let mut zip = zip::ZipWriter::new(std::io::BufWriter::new(file));
        let stored = zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Stored);
        let deflated = zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Deflated);
        let manifest = ron::ser::to_string_pretty(&plan.manifest, ron::ser::PrettyConfig::default())?;
        zip.start_file(MANIFEST_NAME, deflated)?;
        zip.write_all(manifest.as_bytes())?;
        for (path, bytes) in &plan.files {
            let opts = if STORED_EXTENSIONS.contains(&ext_of(path).as_str()) { stored } else { deflated };
            zip.start_file(path.as_str(), opts)?;
            zip.write_all(bytes)?;
        }
        zip.finish()?.flush()?;
    }
    std::fs::rename(&tmp, out)?;
    Ok(())
}

/// [`plan_package`] + [`write_package`]. Returns the plan (manifest, warnings).
pub fn build_package(opts: &BuildOptions, out: &Path) -> Result<PackagePlan> {
    let plan = plan_package(opts)?;
    write_package(&plan, out)?;
    Ok(plan)
}

/// Content hash over the file table (sorted paths, sizes, file hashes).
fn content_hash(records: &[FileRecord]) -> String {
    let mut sorted: Vec<&FileRecord> = records.iter().collect();
    sorted.sort_by(|a, b| a.path.cmp(&b.path));
    let mut buf = Vec::new();
    for r in sorted {
        buf.extend_from_slice(r.path.as_bytes());
        buf.push(0);
        buf.extend_from_slice(&r.size.to_le_bytes());
        buf.extend_from_slice(r.hash.as_bytes());
        buf.push(b'\n');
    }
    sha256::hex(&sha256::digest(&buf))
}

/// Reject shader sources that try to read other files at compile time.
/// Check a packaged `.wasm` component module against the content limits: size,
/// the `host`-only import allowlist, the memory cap, a valid manifest and only
/// the capabilities untrusted content may use. Nothing is executed.
pub fn validate_wasm_module(bytes: &[u8], limits: &ContentLimits) -> std::result::Result<(), String> {
    let sandbox = zestengine_core::wasm::SandboxLimits {
        max_module_bytes: limits.max_wasm_bytes.min(usize::MAX as u64) as usize,
        ..zestengine_core::wasm::SandboxLimits::untrusted()
    };
    zestengine_core::wasm::inspect_module(bytes, &sandbox, zestengine_core::wasm::UNTRUSTED_CAPS).map(|_| ())
}

pub fn validate_shader_source(src: &str) -> std::result::Result<(), String> {
    if src.len() > 512 * 1024 {
        return Err("shader source larger than 512 KiB".into());
    }
    for bad in ["#include", "GL_GOOGLE_include_directive", "GL_ARB_shading_language_include"] {
        if src.contains(bad) {
            return Err(format!("shader uses {bad}; includes are not allowed in content packages"));
        }
    }
    Ok(())
}

// -------------------------------------------------------------------- reader

/// An opened, manifest-validated package (file contents are verified while
/// extracting).
pub struct PackageReader<R: Read + Seek> {
    pub manifest: PackageManifest,
    pub limits: ContentLimits,
    archive: zip::ZipArchive<R>,
}

impl PackageReader<std::io::BufReader<std::fs::File>> {
    /// Open a package file, checking its size against the loosest limits and
    /// the manifest against the limits of its kind (`limits_for`).
    pub fn open(path: &Path, limits_for: &dyn Fn(ContentKind) -> ContentLimits) -> Result<Self> {
        let size = std::fs::metadata(path).with_context(|| format!("opening {}", path.display()))?.len();
        let outer = ContentLimits::outer();
        if size > outer.max_package_bytes {
            bail!("package is {size} bytes (limit {})", outer.max_package_bytes);
        }
        let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let reader = Self::from_reader(std::io::BufReader::new(file), limits_for)?;
        if size > reader.limits.max_package_bytes {
            bail!(
                "{} package is {size} bytes (limit {})",
                reader.manifest.kind.label(),
                reader.limits.max_package_bytes
            );
        }
        Ok(reader)
    }
}

impl<R: Read + Seek> PackageReader<R> {
    pub fn from_reader(reader: R, limits_for: &dyn Fn(ContentKind) -> ContentLimits) -> Result<Self> {
        let mut archive = zip::ZipArchive::new(reader).context("not a .zestenginepak (zip) archive")?;
        let manifest_text = {
            let mut f = archive.by_name(MANIFEST_NAME).context("package has no manifest.ron")?;
            if f.size() > MIB {
                bail!("manifest.ron is larger than 1 MiB");
            }
            let mut s = String::new();
            f.by_ref().take(MIB + 1).read_to_string(&mut s).context("manifest.ron is not UTF-8")?;
            s
        };
        let manifest: PackageManifest = ron::from_str(&manifest_text).context("parsing manifest.ron")?;
        let limits = limits_for(manifest.kind);
        validate_manifest(&manifest, &limits)?;
        if archive.len() > limits.max_files + 2 {
            bail!("archive holds {} entries (limit {})", archive.len(), limits.max_files);
        }
        Ok(Self { manifest, limits, archive })
    }

    /// Read one listed file, checking its size and hash against the manifest.
    pub fn read_file(&mut self, archive_path: &str) -> Result<Vec<u8>> {
        let record = self
            .manifest
            .files
            .iter()
            .find(|r| r.path == archive_path)
            .with_context(|| format!("{archive_path} is not in the manifest"))?
            .clone();
        let mut f = self
            .archive
            .by_name(archive_path)
            .with_context(|| format!("{archive_path} listed but missing from the archive"))?;
        if f.size() != record.size {
            bail!("{archive_path}: size {} does not match the manifest ({})", f.size(), record.size);
        }
        let mut bytes = Vec::with_capacity(record.size as usize);
        f.by_ref().take(record.size + 1).read_to_end(&mut bytes)?;
        if bytes.len() as u64 != record.size {
            bail!("{archive_path}: truncated or oversized entry");
        }
        if sha256::hex(&sha256::digest(&bytes)) != record.hash {
            bail!("{archive_path}: content hash mismatch (corrupted or tampered package)");
        }
        Ok(bytes)
    }

    /// Read a packaged project file by its project-relative path.
    pub fn read_asset(&mut self, rel: &str) -> Result<Vec<u8>> {
        self.read_file(&format!("{FILES_PREFIX}{rel}"))
    }

    pub fn thumbnail(&mut self) -> Option<Vec<u8>> {
        let t = self.manifest.thumbnail.clone()?;
        self.read_file(&t).ok()
    }
}

/// Structural checks on a manifest (format versions, paths, sizes, claims).
pub fn validate_manifest(m: &PackageManifest, limits: &ContentLimits) -> Result<()> {
    if m.format_version > PACKAGE_FORMAT_VERSION {
        bail!(
            "package format {} is newer than this game supports ({PACKAGE_FORMAT_VERSION}); update the game",
            m.format_version
        );
    }
    if m.format_version == 0 {
        bail!("invalid package format version 0");
    }
    if m.engine_format > ENGINE_CONTENT_FORMAT {
        bail!(
            "content was built for engine format {} but this game supports up to {ENGINE_CONTENT_FORMAT}",
            m.engine_format
        );
    }
    if m.name.trim().is_empty() || m.name.len() > 128 {
        bail!("package name must be 1..=128 characters");
    }
    if m.author.len() > 128 || m.version.len() > 64 || m.description.len() > 4096 {
        bail!("manifest metadata too long");
    }
    if m.content_hash.len() != 64 || !m.content_hash.bytes().all(|b| b.is_ascii_hexdigit()) {
        bail!("manifest content hash is not a SHA-256 hex digest");
    }
    if normalize_rel(&m.entry).as_deref() != Some(m.entry.as_str()) || ext_of(&m.entry) != "scene" {
        bail!("manifest entry {:?} is not a safe relative .scene path", m.entry);
    }
    let entry_path = format!("{FILES_PREFIX}{}", m.entry);
    if !m.files.iter().any(|r| r.path == entry_path) {
        bail!("entry scene {} is not in the package", m.entry);
    }
    if m.files.len() > limits.max_files {
        bail!("{} files (limit {})", m.files.len(), limits.max_files);
    }
    let mut seen = BTreeSet::new();
    let mut total = 0u64;
    for r in &m.files {
        let ok = if let Some(rel) = r.path.strip_prefix(FILES_PREFIX).or_else(|| r.path.strip_prefix(TEXCACHE_PREFIX)).or_else(|| r.path.strip_prefix(GEOCACHE_PREFIX)).or_else(|| r.path.strip_prefix(ATLASCACHE_PREFIX)) {
            normalize_rel(rel).as_deref() == Some(rel)
        } else {
            r.path == THUMBNAIL_NAME
        };
        if !ok {
            bail!("unsafe or unknown path in package: {:?}", r.path);
        }
        if !seen.insert(r.path.to_ascii_lowercase()) {
            bail!("duplicate path in package: {:?}", r.path);
        }
        if r.size > limits.max_file_bytes {
            bail!("{} is {} bytes (per-file limit {})", r.path, r.size, limits.max_file_bytes);
        }
        if r.hash.len() != 64 {
            bail!("{}: bad file hash", r.path);
        }
        total += r.size;
    }
    if total > limits.max_unpacked_bytes {
        bail!("package unpacks to {total} bytes (limit {})", limits.max_unpacked_bytes);
    }
    if content_hash(&m.files) != m.content_hash {
        bail!("manifest content hash does not match its file table (corrupted or tampered package)");
    }
    limits.check_stats(&m.stats).map_err(|e| anyhow::anyhow!("manifest declares {e}"))?;
    Ok(())
}

// --------------------------------------------------------------------- cache

/// A package extracted into the content cache, ready for the scene loader.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExtractedPackage {
    pub manifest: PackageManifest,
    /// `<cache>/<content hash>`.
    pub dir: PathBuf,
    /// Absolute path of the extracted entry scene.
    pub entry: PathBuf,
    pub warnings: Vec<String>,
    /// Fingerprint of the component set used to strip unknown components.
    pub components_fingerprint: String,
}

pub struct ExtractOptions<'a> {
    /// Whether a component kind exists in this game (unknown ones are dropped).
    pub known_component: &'a dyn Fn(&str) -> bool,
    /// Stable fingerprint of the known set; a cached extraction made with a
    /// different set is redone.
    pub components_fingerprint: String,
    /// A dedicated server extracts: files it never reads (images, audio, shaders,
    /// materials ...) and the texture / geometry caches are skipped (NET-3).
    pub server_side: bool,
}

/// Content cache directory: extracted packages keyed by content hash, an
/// index of the sources (paths / URLs) they came from, and a downloads folder.
#[derive(Clone, Debug)]
pub struct ContentCache {
    pub root: PathBuf,
}

impl ContentCache {
    /// The root is made absolute: extracted assets reference each other by
    /// absolute path.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let root = std::path::absolute(&root).unwrap_or(root);
        // Whatever sits here came from outside the project: import hooks never touch it.
        crate::import_hooks::mark_untrusted_root(&root);
        Self { root }
    }

    pub fn dir_for(&self, hash: &str) -> PathBuf {
        self.root.join(hash)
    }

    pub fn downloads_dir(&self) -> PathBuf {
        self.root.join("downloads")
    }

    /// A completed extraction of `hash` made with the same component set.
    pub fn lookup(&self, hash: &str, components_fingerprint: &str) -> Option<ExtractedPackage> {
        if !is_hash(hash) {
            return None;
        }
        let dir = self.dir_for(hash);
        let text = std::fs::read_to_string(dir.join(EXTRACTED_MARKER)).ok()?;
        let ex: ExtractedPackage = ron::from_str(&text).ok()?;
        let valid = ex.dir == dir
            && ex.manifest.content_hash == hash
            && ex.components_fingerprint == components_fingerprint
            && ex.entry.is_file();
        if valid {
            // Touch for LRU pruning.
            let _ = std::fs::write(dir.join(".used"), b"");
        }
        valid.then_some(ex)
    }

    /// Content hash previously loaded from `source` (path or URL).
    pub fn lookup_source(&self, source: &str) -> Option<String> {
        self.read_index().get(source).cloned()
    }

    pub fn remember_source(&self, source: &str, hash: &str) {
        let mut index = self.read_index();
        index.insert(source.to_owned(), hash.to_owned());
        let _ = std::fs::create_dir_all(&self.root);
        if let Ok(text) = ron::ser::to_string_pretty(&index, ron::ser::PrettyConfig::default()) {
            let _ = std::fs::write(self.root.join(INDEX_NAME), text);
        }
    }

    fn read_index(&self) -> BTreeMap<String, String> {
        std::fs::read_to_string(self.root.join(INDEX_NAME))
            .ok()
            .and_then(|t| ron::from_str(&t).ok())
            .unwrap_or_default()
    }

    /// Extract (or reuse) a package. Files are streamed out with their sizes
    /// and hashes checked, then RON assets are rewritten / sanitized, glTF
    /// URIs and shader sources validated, unknown components dropped and BC
    /// cache blobs installed. The directory only appears once complete.
    pub fn extract<R: Read + Seek>(&self, reader: &mut PackageReader<R>, opts: &ExtractOptions) -> Result<ExtractedPackage> {
        let hash = reader.manifest.content_hash.clone();
        if let Some(ex) = self.lookup(&hash, &opts.components_fingerprint) {
            return Ok(ex);
        }
        std::fs::create_dir_all(&self.root).with_context(|| format!("creating {}", self.root.display()))?;
        // Re-mark now that the directory exists (canonical path), and keep hooks off while extracting.
        crate::import_hooks::mark_untrusted_root(&self.root);
        let _no_hooks = crate::import_hooks::suppress_hooks();
        let final_dir = self.dir_for(&hash);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let tmp = self.root.join(format!("{hash}.partial-{}-{nanos}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let result = self.extract_into(reader, opts, &tmp, &final_dir);
        match result {
            Ok(mut ex) => {
                let _ = std::fs::remove_dir_all(&final_dir);
                std::fs::rename(&tmp, &final_dir).context("moving extracted package into place")?;
                ex.dir = final_dir;
                Ok(ex)
            }
            Err(e) => {
                let _ = std::fs::remove_dir_all(&tmp);
                Err(e)
            }
        }
    }

    fn extract_into<R: Read + Seek>(
        &self,
        reader: &mut PackageReader<R>,
        opts: &ExtractOptions,
        tmp: &Path,
        final_dir: &Path,
    ) -> Result<ExtractedPackage> {
        let manifest = reader.manifest.clone();
        let limits = reader.limits.clone();
        let mut warnings = Vec::new();
        let files_dir = tmp.join("files");
        let final_files = final_dir.join("files");
        let mut assets: BTreeSet<String> = BTreeSet::new();
        let mut blobs: Vec<(String, bool, Vec<u8>)> = Vec::new();
        let mut geo_blobs: Vec<(String, Vec<u8>)> = Vec::new();
        let mut atlas_blobs: Vec<(String, Vec<u8>)> = Vec::new();
        // Client-only files a server leaves in the archive (still counted as "inside the package").
        let mut server_skipped: BTreeSet<String> = BTreeSet::new();
        for record in &manifest.files {
            if opts.server_side && (record.path.starts_with(TEXCACHE_PREFIX) || record.path.starts_with(GEOCACHE_PREFIX) || record.path.starts_with(ATLASCACHE_PREFIX) || record.path == THUMBNAIL_NAME) {
                continue;
            }
            if let Some(rel) = record.path.strip_prefix(FILES_PREFIX) {
                if opts.server_side && crate::net_strip::is_client_only_file(rel) {
                    server_skipped.insert(rel.to_owned());
                    continue;
                }
                if ext_of(rel) == "spv" || !ext_allowed(rel) {
                    warnings.push(format!("dropped {rel}: file type not allowed"));
                    continue;
                }
                let bytes = reader.read_file(&record.path)?;
                let out = files_dir.join(rel);
                if let Some(d) = out.parent() {
                    std::fs::create_dir_all(d)?;
                }
                std::fs::write(&out, &bytes).with_context(|| format!("writing {}", out.display()))?;
                assets.insert(rel.to_owned());
            } else if let Some(rest) = record.path.strip_prefix(TEXCACHE_PREFIX) {
                let parsed = rest
                    .strip_suffix(".1.cbc")
                    .map(|r| (r, true))
                    .or_else(|| rest.strip_suffix(".0.cbc").map(|r| (r, false)));
                match parsed {
                    Some((rel, srgb)) => blobs.push((rel.to_owned(), srgb, reader.read_file(&record.path)?)),
                    None => warnings.push(format!("ignoring unknown cache entry {rest}")),
                }
            } else if let Some(rest) = record.path.strip_prefix(GEOCACHE_PREFIX) {
                match rest.strip_suffix(crate::geo_cache::GEO_EXTENSION).and_then(|r| r.strip_suffix('.')) {
                    Some(rel) => geo_blobs.push((rel.to_owned(), reader.read_file(&record.path)?)),
                    None => warnings.push(format!("ignoring unknown cache entry {rest}")),
                }
            } else if let Some(rest) = record.path.strip_prefix(ATLASCACHE_PREFIX) {
                match rest.strip_suffix(crate::sprite_atlas::COOKED_ATLAS_EXTENSION).and_then(|r| r.strip_suffix('.')) {
                    Some(rel) => atlas_blobs.push((rel.to_owned(), reader.read_file(&record.path)?)),
                    None => warnings.push(format!("ignoring unknown cache entry {rest}")),
                }
            } else if record.path == THUMBNAIL_NAME {
                let bytes = reader.read_file(&record.path)?;
                std::fs::write(tmp.join(THUMBNAIL_NAME), bytes)?;
            }
        }
        if !assets.contains(&manifest.entry) {
            bail!("entry scene {} was not extracted", manifest.entry);
        }

        let abs_of = |rel: &str| final_files.join(rel).to_string_lossy().into_owned();
        let mut wasm_modules = 0usize;
        for rel in &assets {
            let path = files_dir.join(rel);
            let ext = ext_of(rel);
            if ext == "gltf" || ext == "glb" || ext == "vrm" {
                let bytes = std::fs::read(&path)?;
                let uris = gltf_uris_from_bytes(&bytes).unwrap_or_default();
                let dir = Path::new(rel).parent().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
                for uri in uris {
                    let target = normalize_rel(&uri).map(|u| if dir.is_empty() { u } else { format!("{dir}/{u}") });
                    if !target.as_ref().is_some_and(|t| assets.contains(t) || server_skipped.contains(t)) {
                        bail!("{rel} references {uri:?}, which is not inside the package");
                    }
                }
                continue;
            }
            if ext == "wasm" {
                wasm_modules += 1;
                if wasm_modules > limits.max_wasm_modules {
                    bail!("{rel}: more than {} WASM modules (limit {})", wasm_modules - 1, limits.max_wasm_modules);
                }
                let bytes = std::fs::read(&path)?;
                validate_wasm_module(&bytes, &limits).map_err(|e| anyhow::anyhow!("{rel}: {e}"))?;
                continue;
            }
            if ext == "frag" {
                let src = std::fs::read_to_string(&path).with_context(|| format!("{rel} is not UTF-8"))?;
                validate_shader_source(&src).map_err(|e| anyhow::anyhow!("{rel}: {e}"))?;
                continue;
            }
            if !is_ron_text(rel) {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                warnings.push(format!("{rel} is not UTF-8 text; dropped"));
                std::fs::remove_file(&path)?;
                continue;
            };
            let mut blanked = Vec::new();
            let rewritten = rewrite_ron_strings(&text, &mut |s| {
                if let Some(r) = normalize_rel(s)
                    && assets.contains(&r)
                {
                    return Some(abs_of(&r));
                }
                if escapes_package(s) {
                    blanked.push(s.to_owned());
                    return Some(String::new());
                }
                None
            });
            for b in blanked {
                warnings.push(format!("{rel}: removed reference to {b:?} outside the package"));
            }
            let rewritten = if ext == "scene" {
                match ron::from_str::<SceneFile>(&rewritten) {
                    Ok(mut scene) => {
                        let mut dropped = BTreeSet::new();
                        for e in &mut scene.entries {
                            e.components.retain(|c| {
                                let known = (opts.known_component)(&c.kind);
                                if !known {
                                    dropped.insert(c.kind.clone());
                                }
                                known
                            });
                        }
                        let objects = scene.entries.len();
                        if objects > limits.max_objects {
                            bail!("{rel} has {objects} objects (limit {})", limits.max_objects);
                        }
                        for d in dropped {
                            warnings.push(format!("{rel}: dropped unknown component {d:?}"));
                        }
                        ron::ser::to_string_pretty(&scene, ron::ser::PrettyConfig::default())?
                    }
                    Err(e) if *rel == manifest.entry => bail!("entry scene does not parse: {e}"),
                    Err(e) => {
                        warnings.push(format!("{rel} does not parse ({e}); kept as is"));
                        rewritten
                    }
                }
            } else {
                rewritten
            };
            std::fs::write(&path, rewritten)?;
        }

        for (rel, srgb, blob) in blobs {
            if !assets.contains(&rel) || !IMAGE_EXTENSIONS.contains(&ext_of(&rel).as_str()) {
                warnings.push(format!("texture cache for {rel} has no matching image; ignored"));
                continue;
            }
            if let Err(e) = crate::seed_texture_cache(files_dir.join(&rel), srgb, &blob, limits.max_texture_dim) {
                warnings.push(format!("texture cache for {rel} rejected: {e:#}"));
            }
        }

        // Cluster DAGs go next to their model. Each is validated in full (bad
        // traversal data can hang the GPU); one that fails, or that would exceed
        // the DAG budget, is dropped and the model's DAG is built locally instead.
        let mut dag_total = 0u64;
        for (rel, blob) in geo_blobs {
            if !assets.contains(&rel) || !matches!(ext_of(&rel).as_str(), "gltf" | "glb" | "vrm" | "fbx") {
                warnings.push(format!("cluster data for {rel} has no matching model; ignored"));
                continue;
            }
            match crate::geo_cache::decode(&blob) {
                Ok(geo) if dag_total + geo.dag_bytes() as u64 <= limits.max_dag_bytes => {
                    dag_total += geo.dag_bytes() as u64;
                    if let Err(e) = crate::geo_cache::install(&files_dir.join(&rel), &blob) {
                        warnings.push(format!("cluster data for {rel} not installed: {e}"));
                    }
                }
                Ok(_) => warnings.push(format!("cluster data for {rel} exceeds the {} byte budget; it is rebuilt on load", limits.max_dag_bytes)),
                Err(e) => warnings.push(format!("cluster data for {rel} rejected: {e}; it is rebuilt on load")),
            }
        }

        // Packed atlases go beside their `.spriteatlas`; a mismatch with the
        // extracted files only makes the first load pack the atlas itself.
        for (rel, blob) in atlas_blobs {
            if !assets.contains(&rel) || ext_of(&rel) != crate::sprite_atlas::ATLAS_EXTENSION {
                warnings.push(format!("packed atlas for {rel} has no matching atlas file; ignored"));
                continue;
            }
            if let Err(e) = crate::sprite_atlas::install_cooked_atlas(&files_dir.join(&rel), &blob) {
                warnings.push(format!("packed atlas for {rel} rejected: {e:#}"));
            }
        }

        let ex = ExtractedPackage {
            entry: final_files.join(&manifest.entry),
            manifest,
            dir: final_dir.to_owned(),
            warnings,
            components_fingerprint: opts.components_fingerprint.clone(),
        };
        std::fs::write(
            tmp.join(EXTRACTED_MARKER),
            ron::ser::to_string_pretty(&ex, ron::ser::PrettyConfig::default())?,
        )?;
        Ok(ex)
    }

    /// Delete one extracted package.
    pub fn evict(&self, hash: &str) -> Result<()> {
        if !is_hash(hash) {
            bail!("not a content hash: {hash:?}");
        }
        let dir = self.dir_for(hash);
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        Ok(())
    }

    /// Hashes of the extracted packages in the cache.
    pub fn entries(&self) -> Vec<String> {
        let Ok(rd) = std::fs::read_dir(&self.root) else { return Vec::new() };
        rd.filter_map(|e| e.ok())
            .filter_map(|e| e.file_name().to_str().map(str::to_owned))
            .filter(|n| is_hash(n))
            .collect()
    }

    /// Total bytes on disk under the cache root.
    pub fn size_bytes(&self) -> u64 {
        dir_size(&self.root)
    }

    /// Evict least-recently-used packages until the cache fits `max_bytes`.
    /// Returns how many were removed.
    pub fn prune(&self, max_bytes: u64, keep: &[&str]) -> usize {
        let mut entries: Vec<(std::time::SystemTime, String, u64)> = self
            .entries()
            .into_iter()
            .filter(|h| !keep.contains(&h.as_str()))
            .map(|h| {
                let dir = self.dir_for(&h);
                let used = std::fs::metadata(dir.join(".used"))
                    .or_else(|_| std::fs::metadata(dir.join(EXTRACTED_MARKER)))
                    .and_then(|m| m.modified())
                    .unwrap_or(std::time::UNIX_EPOCH);
                (used, h, dir_size(&dir))
            })
            .collect();
        entries.sort();
        let mut total = self.size_bytes();
        let mut removed = 0;
        for (_, h, size) in entries {
            if total <= max_bytes {
                break;
            }
            if self.evict(&h).is_ok() {
                total = total.saturating_sub(size);
                removed += 1;
            }
        }
        removed
    }
}

fn is_hash(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

fn dir_size(dir: &Path) -> u64 {
    let Ok(rd) = std::fs::read_dir(dir) else { return 0 };
    rd.filter_map(|e| e.ok())
        .map(|e| match e.file_type() {
            Ok(t) if t.is_dir() => dir_size(&e.path()),
            Ok(_) => e.metadata().map(|m| m.len()).unwrap_or(0),
            Err(_) => 0,
        })
        .sum()
}

/// Fingerprint of a component-name set (order independent).
pub fn components_fingerprint<'a>(names: impl IntoIterator<Item = &'a str>) -> String {
    let mut v: Vec<&str> = names.into_iter().collect();
    v.sort_unstable();
    v.dedup();
    sha256::hex(&sha256::digest(v.join("\n").as_bytes()))[..16].to_owned()
}

/// SHA-256 of a byte slice as lowercase hex.
pub fn sha256_hex(bytes: &[u8]) -> String {
    sha256::hex(&sha256::digest(bytes))
}

// -------------------------------------------------------------------- sha256

pub(crate) mod sha256 {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
        0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
        0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
        0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    pub fn digest(data: &[u8]) -> [u8; 32] {
        let mut h: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
        ];
        let bit_len = (data.len() as u64).wrapping_mul(8);
        let mut tail = data[data.len() - data.len() % 64..].to_vec();
        tail.push(0x80);
        while tail.len() % 64 != 56 {
            tail.push(0);
        }
        tail.extend_from_slice(&bit_len.to_be_bytes());
        let full = &data[..data.len() - data.len() % 64];
        for block in full.as_chunks::<64>().0.iter().chain(tail.as_chunks::<64>().0.iter()) {
            compress(&mut h, block);
        }
        let mut out = [0u8; 32];
        for (i, v) in h.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&v.to_be_bytes());
        }
        out
    }

    fn compress(h: &mut [u32; 8], block: &[u8]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = *h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (x, v) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *x = x.wrapping_add(v);
        }
    }

    pub fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ComponentData, MaterialFile, SceneEntry};

    fn temp_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let d = std::env::temp_dir().join(format!("zestengine-pkg-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn entry(name: &str, source: ObjectSource, material: MaterialRef, components: Vec<ComponentData>) -> SceneEntry {
        SceneEntry {
            id: zestengine_core::ObjectId::new().to_string(),
            name: name.into(),
            source,
            enabled: true,
            static_geometry: false,
            lightmap_scale: 1.0,
            cast_shadows: zestengine_core::ShadowCasting::On,
            layer: 0,
            material,
            extra_materials: vec![],
            renderer_split: false,
            parent: None,
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

    /// A small project: a primitive with a `.material` (repo texture + a
    /// generated 8x8 one), a component whose data references an audio file, a
    /// plugin component, an unknown one, and a baked sidecar.
    fn sample_project() -> (PathBuf, SceneFile) {
        let root = temp_dir("proj");
        let repo_tex = Path::new(env!("CARGO_MANIFEST_DIR")).join("../zestengine-editor/assets/ferris.png");
        std::fs::create_dir_all(root.join("textures")).unwrap();
        std::fs::copy(repo_tex, root.join("textures/ferris.png")).unwrap();
        image::RgbaImage::from_pixel(8, 8, image::Rgba([200, 100, 50, 255]))
            .save(root.join("textures/orm.png"))
            .unwrap();
        let mut mat: MaterialFile = ron::from_str(&format!(
            "(name: \"Ferris\", params: {}, features: {})",
            ron::to_string(&zestengine_render::MaterialParams::default()).unwrap(),
            ron::to_string(&zestengine_render::MaterialFeatures::default()).unwrap()
        ))
        .unwrap();
        mat.textures.albedo = Some("textures/ferris.png".into());
        mat.textures.orm = Some("textures/orm.png".into());
        crate::save_material_file(root.join("materials/ferris.material"), &mat).unwrap();
        std::fs::create_dir_all(root.join("audio")).unwrap();
        std::fs::write(root.join("audio/beep.wav"), b"RIFF0000WAVEfmt ").unwrap();
        std::fs::create_dir_all(root.join("scenes")).unwrap();
        std::fs::write(root.join("scenes/world.lightdata"), b"not really light data").unwrap();
        std::fs::write(root.join("secret.txt"), b"keep out").unwrap();
        let scene = SceneFile {
            entries: vec![
                entry(
                    "Cube",
                    ObjectSource::Primitive { shape: crate::PrimitiveShape::Cube },
                    MaterialRef::File("materials/ferris.material".into()),
                    vec![
                        ComponentData::new("Spin", "(axis:(0.0,1.0,0.0),degrees_per_second:45.0)"),
                        ComponentData::new("Audio Source", "(clip:\"audio/beep.wav\",volume:1.0)"),
                        ComponentData::new("My Plugin Thing", "()"),
                        ComponentData::new("Ghost", "()"),
                    ],
                ),
                entry("Light", ObjectSource::Light, MaterialRef::File(String::new()), vec![]),
            ],
            ..Default::default()
        };
        (root, scene)
    }

    fn classify(k: &str) -> ComponentStatus {
        match k {
            "Spin" | "Audio Source" | "Wasm Script" => ComponentStatus::Builtin,
            "My Plugin Thing" => ComponentStatus::Plugin,
            _ => ComponentStatus::Unknown,
        }
    }

    fn options(root: &Path, scene: SceneFile, kind: ContentKind) -> BuildOptions<'static> {
        BuildOptions {
            kind,
            name: "Test World".into(),
            author: "tester".into(),
            version: "1.2".into(),
            description: String::new(),
            project_root: root.to_owned(),
            scene,
            entry: "scenes/world.scene".into(),
            thumbnail: None,
            classify: &classify,
            strip_plugin_components: false,
            include_texture_cache: true,
            limits: ContentLimits::for_kind(kind),
        }
    }

    #[test]
    fn sha256_matches_known_vectors() {
        assert_eq!(sha256_hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        let long = vec![b'a'; 1000];
        assert_eq!(sha256_hex(&long), "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3");
    }

    #[test]
    fn ron_strings_are_found_and_rewritten_recursively() {
        let text = r##"(a: "x/y.png", c: 'q', d: "(clip:\"audio/a.wav\",n:1)", r: r#"raw.bin"#, e: "plain")"##;
        let mut seen = Vec::new();
        visit_ron_strings(text, &mut |s| seen.push(s.to_owned()));
        assert!(seen.contains(&"x/y.png".to_owned()));
        assert!(seen.contains(&"audio/a.wav".to_owned()), "nested component data is visited: {seen:?}");
        assert!(seen.contains(&"raw.bin".to_owned()));
        let out = rewrite_ron_strings(text, &mut |s| (s == "audio/a.wav").then(|| "/abs/a.wav".to_owned()));
        assert!(out.contains(r#"\"/abs/a.wav\""#), "{out}");
        assert!(out.contains(r#""x/y.png""#) && out.contains("'q'"));
    }

    #[test]
    fn relative_paths_are_normalized_and_escapes_rejected() {
        assert_eq!(normalize_rel("./a\\b.png").as_deref(), Some("a/b.png"));
        for bad in ["/etc/passwd", "../x", "a/../../x", "C:\\x", "a//b", "", " a", "http://x"] {
            assert!(normalize_rel(bad).is_none(), "{bad:?}");
        }
        assert!(escapes_package("/home/me/secret.png") && escapes_package("..\\up") && escapes_package("D:/x"));
        assert!(!escapes_package("Hello world /") && !escapes_package("textures/a.png"));
    }

    #[test]
    fn texture_refs_and_cooked_astc_caches() {
        let (root, scene) = sample_project();
        crate::save_scene_file(root.join("scenes/world.scene"), &scene).unwrap();
        for name in ["n.png", "e.png"] {
            image::RgbaImage::from_pixel(12, 6, image::Rgba([1, 2, 3, 255])).save(root.join("textures").join(name)).unwrap();
        }
        let inline = MaterialRef::Inline {
            params: Default::default(),
            features: Default::default(),
            shader: "standard".into(),
            custom: Default::default(),
            render_queue: None,
            textures: MaterialTextures {
                normal: Some("textures/n.png".into()),
                emission: Some("textures/e.png".into()),
                ao: Some("textures/not_cooked.png".into()),
                ..Default::default()
            },
        };
        #[derive(Serialize)]
        struct Prefab {
            entries: Vec<SceneEntry>,
        }
        let prefab = Prefab { entries: vec![entry("Enemy", ObjectSource::Light, inline, vec![])] };
        std::fs::create_dir_all(root.join("prefabs")).unwrap();
        std::fs::write(root.join("prefabs/enemy.prefab"), ron::to_string(&prefab).unwrap()).unwrap();
        let files: BTreeSet<String> = [
            "scenes/world.scene",
            "prefabs/enemy.prefab",
            "materials/ferris.material",
            "textures/ferris.png",
            "textures/orm.png",
            "textures/n.png",
            "textures/e.png",
        ]
        .map(String::from)
        .into();
        let mut warnings = Vec::new();
        let refs = material_texture_refs(&root, &files, &[], &mut warnings);
        assert!(warnings.is_empty(), "{warnings:?}");
        let want: BTreeSet<(String, bool)> = [
            ("textures/ferris.png", true),
            ("textures/orm.png", false),
            ("textures/n.png", false),
            ("textures/e.png", true),
        ]
        .map(|(r, s)| (r.to_owned(), s))
        .into();
        assert_eq!(refs, want);

        assert_eq!(cooked_texcache_rel("textures/n.png", false), "zestengine_texcache/textures/n.png.0.castc");
        assert_eq!(parse_cooked_texcache_rel("zestengine_texcache/a/b.png.1.castc"), Some(("a/b.png", true)));
        for bad in ["zestengine_texcache/b.png.2.castc", "texcache/b.png.1.castc", "zestengine_texcache/../x.png.1.castc", "zestengine_texcache/b.png.1.cbc"] {
            assert_eq!(parse_cooked_texcache_rel(bad), None, "{bad}");
        }

        // A cooked asset root (as the APK extracts it) gets its caches installed.
        let assets = temp_dir("cooked");
        for (rel, srgb) in &refs {
            let dst = assets.join(rel);
            std::fs::create_dir_all(dst.parent().unwrap()).unwrap();
            std::fs::copy(root.join(rel), &dst).unwrap();
            let blob = crate::astc_cache_blob(root.join(rel), *srgb, crate::AstcBlocks::default()).unwrap();
            let shipped = assets.join(cooked_texcache_rel(rel, *srgb));
            std::fs::create_dir_all(shipped.parent().unwrap()).unwrap();
            std::fs::write(shipped, blob).unwrap();
        }
        std::fs::write(assets.join("zestengine_texcache/stray.bin"), b"?").unwrap();
        let (seeded, warnings) = seed_cooked_texture_caches(&assets);
        assert_eq!(seeded, 4);
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(!assets.join(COOKED_TEXCACHE_DIR).exists(), "shipped copies removed");
        assert!(crate::import_cache_path(&assets.join("textures/ferris.png"), true, true).unwrap().is_file());
        assert!(crate::import_cache_path(&assets.join("textures/n.png"), false, true).unwrap().is_file());
        assert_eq!(seed_cooked_texture_caches(&assets), (0, vec![]), "nothing left to seed");
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&assets);
    }

    #[test]
    fn world_round_trip_build_extract_and_cache() {
        let (root, scene) = sample_project();
        let plan = plan_package(&options(&root, scene, ContentKind::World)).unwrap();
        assert!(plan.is_ok(), "{:?}", plan.errors);
        let m = &plan.manifest;
        assert_eq!(m.components, vec!["Audio Source".to_owned(), "My Plugin Thing".into(), "Spin".into()]);
        assert!(plan.warnings.iter().any(|w| w.contains("Ghost")), "unknown dropped at build");
        assert!(plan.warnings.iter().any(|w| w.contains("plugin component")), "plugin flagged");
        let paths: Vec<&str> = m.files.iter().map(|r| r.path.as_str()).collect();
        for p in [
            "files/scenes/world.scene",
            "files/materials/ferris.material",
            "files/textures/ferris.png",
            "files/textures/orm.png",
            "files/audio/beep.wav",
            "files/scenes/world.lightdata",
            "texcache/textures/ferris.png.1.cbc",
            "texcache/textures/orm.png.0.cbc",
        ] {
            assert!(paths.contains(&p), "{p} missing from {paths:?}");
        }
        assert!(!paths.iter().any(|p| p.contains("secret")));
        assert!(m.stats.texture_bytes > 0 && m.stats.objects == 2);

        let out = root.join("build/world.zestenginepak");
        write_package(&plan, &out).unwrap();
        let mut reader = PackageReader::open(&out, &ContentLimits::for_kind).unwrap();
        assert_eq!(reader.manifest, plan.manifest);
        assert_eq!(reader.read_asset("audio/beep.wav").unwrap(), b"RIFF0000WAVEfmt ");

        let cache = ContentCache::new(temp_dir("cache"));
        let known = |k: &str| k == "Spin" || k == "Audio Source" || zestengine_core::is_renderer_kind(k);
        let opts = ExtractOptions { known_component: &known, components_fingerprint: "fp1".into(), server_side: false };
        let ex = cache.extract(&mut reader, &opts).unwrap();
        assert_eq!(ex.dir, cache.dir_for(&m.content_hash));
        assert!(ex.warnings.iter().any(|w| w.contains("My Plugin Thing")), "{:?}", ex.warnings);

        let loaded = crate::load_scene_file(&ex.entry).unwrap();
        let cube = &loaded.entries[0];
        assert_eq!(cube.components.iter().filter(|c| !zestengine_core::is_renderer_kind(&c.kind)).count(), 2, "plugin component dropped by this game");
        let Some(MaterialRef::File(mat_path)) = crate::renderer_migration::material_refs(cube).into_iter().next() else { panic!() };
        let mat_path = &mat_path;
        assert!(Path::new(mat_path).is_absolute() && Path::new(mat_path).starts_with(&ex.dir));
        let audio = &cube.components.iter().find(|c| c.kind == "Audio Source").unwrap().data;
        let wav = ex.dir.join("files/audio/beep.wav");
        assert!(audio.contains(&*wav.to_string_lossy()), "component path rewritten: {audio}");
        let mat = crate::load_material_file(mat_path).unwrap();
        let albedo = mat.textures.albedo.unwrap();
        assert!(albedo.is_absolute() && albedo.is_file());
        assert!(ex.dir.join("files/scenes/world.lightdata").is_file());

        // A dedicated server extracts the same package without the files it never reads (NET-3).
        let mut reader = PackageReader::open(&out, &ContentLimits::for_kind).unwrap();
        let server_cache = ContentCache::new(temp_dir("server-cache"));
        let server_opts = ExtractOptions { known_component: &known, components_fingerprint: "fp1".into(), server_side: true };
        let sx = server_cache.extract(&mut reader, &server_opts).unwrap();
        assert!(sx.entry.is_file(), "the scene is extracted");
        for gone in ["files/textures/ferris.png", "files/textures/orm.png", "files/audio/beep.wav", "files/materials/ferris.material", "files/scenes/world.lightdata"] {
            assert!(!sx.dir.join(gone).exists(), "{gone} is client-only");
        }
        assert!(!sx.dir.join("texcache").exists() || std::fs::read_dir(sx.dir.join("texcache")).map(|d| d.count() == 0).unwrap_or(true));
        // Seeded import caches make load_texture_bc a cache hit.
        assert!(crate::import_cache_path(&ex.dir.join("files/textures/orm.png"), false, false).unwrap().is_file());
        let tex = crate::load_texture_bc(ex.dir.join("files/textures/orm.png"), false).unwrap();
        assert_eq!((tex.width, tex.height), (8, 8));

        // Same hash + component set: reused; different set: re-extracted.
        assert!(cache.lookup(&m.content_hash, "fp1").is_some());
        let again = cache.extract(&mut reader, &opts).unwrap();
        assert_eq!(again.entry, ex.entry);
        assert!(cache.lookup(&m.content_hash, "fp2").is_none());
        cache.remember_source("http://example/world.zestenginepak", &m.content_hash);
        assert_eq!(cache.lookup_source("http://example/world.zestenginepak").as_deref(), Some(m.content_hash.as_str()));
        assert_eq!(cache.entries(), vec![m.content_hash.clone()]);
        assert_eq!(cache.prune(0, &[]), 1);
        assert!(cache.lookup(&m.content_hash, "fp1").is_none());
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&cache.root);
    }

    #[test]
    fn sprite_atlases_travel_in_packages_and_extract_as_cache_hits() {
        let root = crate::sprite_atlas::tests::shipping_project("pak");
        std::fs::create_dir_all(root.join("scenes")).unwrap();
        let scene = SceneFile {
            entries: vec![entry("Sprite", ObjectSource::Light, MaterialRef::File(String::new()), vec![ComponentData::new("Audio Source", "(clip: \"Sprites/hero.png\")")])],
            ..Default::default()
        };
        let plan = plan_package(&options(&root, scene.clone(), ContentKind::World)).unwrap();
        assert!(plan.is_ok(), "{:?}", plan.errors);
        for f in ["files/ui.spriteatlas", "files/Sprites/hero.png", "atlascache/ui.spriteatlas.catlas"] {
            assert!(plan.files.contains_key(f), "{f} in {:?}", plan.files.keys().collect::<Vec<_>>());
        }
        let out = root.join("build/atlas.zestenginepak");
        write_package(&plan, &out).unwrap();
        let mut reader = PackageReader::open(&out, &ContentLimits::for_kind).unwrap();
        let known = |_: &str| true;
        let opts = ExtractOptions { known_component: &known, components_fingerprint: "fp".into(), server_side: false };
        let ex = ContentCache::new(temp_dir("atlascache")).extract(&mut reader, &opts).unwrap();
        assert!(!ex.warnings.iter().any(|w| w.contains("atlas")), "{:?}", ex.warnings);
        let files = ex.dir.join("files");
        assert!(files.join("ui.spriteatlas").is_file());
        // The packed pages were installed: the first cook in the extracted tree is a cache hit.
        let db = crate::sprite_atlas::AtlasDb::scan(&files);
        assert_eq!(db.atlases.len(), 1);
        assert_eq!(db.atlases[0].members.len(), 1);
        let (cooked, cached) = db.cook(0, crate::Platform::Desktop).unwrap();
        assert!(cached, "the shipped atlas matches the extracted files");
        assert_eq!(cooked.table.sprites.len(), 2);
        // A server never reads atlas pages.
        let server = ExtractOptions { server_side: true, ..opts };
        let mut reader = PackageReader::open(&out, &ContentLimits::for_kind).unwrap();
        let sx = ContentCache::new(temp_dir("atlascache-server")).extract(&mut reader, &server).unwrap();
        let (_, cached) = crate::sprite_atlas::AtlasDb::scan(&sx.dir.join("files")).cook(0, crate::Platform::Desktop).unwrap_or_else(|_| panic!("cook"));
        assert!(!cached);
        // A package without the cache (texture cache off) still carries the atlas file.
        let mut no_cache = options(&root, scene, ContentKind::World);
        no_cache.include_texture_cache = false;
        let plan = plan_package(&no_cache).unwrap();
        assert!(plan.files.contains_key("files/ui.spriteatlas") && !plan.files.contains_key("atlascache/ui.spriteatlas.catlas"));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A project holding the repo's DamagedHelmet (a mesh above the cluster threshold).
    fn model_project() -> Option<(PathBuf, SceneFile)> {
        let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/DamagedHelmet.glb");
        if !src.is_file() {
            return None;
        }
        let root = temp_dir("geo");
        std::fs::create_dir_all(root.join("models")).unwrap();
        std::fs::create_dir_all(root.join("scenes")).unwrap();
        std::fs::copy(src, root.join("models/DamagedHelmet.glb")).unwrap();
        let scene = SceneFile {
            entries: vec![entry(
                "Helmet",
                ObjectSource::Model { path: "models/DamagedHelmet.glb".into(), mesh: 0, extra_meshes: vec![], model: None, mesh_id: None, extra_mesh_ids: vec![] },
                MaterialRef::File(String::new()),
                vec![],
            )],
            ..Default::default()
        };
        Some((root, scene))
    }

    #[test]
    fn cluster_dags_are_cooked_validated_on_extraction_and_old_packages_still_open() {
        let Some((root, scene)) = model_project() else { return };
        let plan = plan_package(&options(&root, scene.clone(), ContentKind::World)).unwrap();
        assert!(plan.is_ok(), "{:?}", plan.errors);
        assert_eq!(plan.manifest.format_version, 2);
        let blob_path = "geocache/models/DamagedHelmet.glb.zgeo";
        let blob = plan.files.get(blob_path).expect("the cook adds the model's cluster data").clone();
        assert!(plan.manifest.stats.clusters > 100 && plan.manifest.stats.dag_bytes > 100_000, "{:?}", plan.manifest.stats);
        let geo = crate::geo_cache::decode(&blob).unwrap();
        assert_eq!(geo.clusters() as u64, plan.manifest.stats.clusters);
        assert_eq!(geo.dag_bytes() as u64, plan.manifest.stats.dag_bytes);

        // Round trip: the extracted model has its cluster file beside it and uses it.
        let out = root.join("build/helmet.zestenginepak");
        write_package(&plan, &out).unwrap();
        let mut reader = PackageReader::open(&out, &ContentLimits::for_kind).unwrap();
        let known = |_: &str| true;
        let opts = ExtractOptions { known_component: &known, components_fingerprint: "fp".into(), server_side: false };
        let cache = ContentCache::new(temp_dir("geocache"));
        let ex = cache.extract(&mut reader, &opts).unwrap();
        assert!(!ex.warnings.iter().any(|w| w.contains("cluster")), "{:?}", ex.warnings);
        let model = ex.dir.join("files/models/DamagedHelmet.glb");
        assert_eq!(std::fs::read(crate::geo_cache::geo_path(&model).unwrap()).unwrap(), blob);
        let scene_loaded = crate::load_model_with_meta(&model).unwrap();
        let again = crate::geo_cache::load(&model, crate::geo_cache::model_key(&scene_loaded, &crate::cluster_dag::DagOptions::default()));
        assert_eq!(again.as_ref(), Some(&geo), "the shipped DAG matches the model as loaded here");

        // A tampered DAG (valid file checksum, bad content) is dropped at extraction, not installed.
        let mut bad = geo.clone();
        let d = bad.meshes.iter_mut().flatten().next().unwrap();
        d.clusters[0].triangle_count = 200;
        let mut bad_plan = plan_package(&options(&root, scene.clone(), ContentKind::World)).unwrap();
        bad_plan.files.insert(blob_path.into(), crate::geo_cache::encode(&bad));
        repack(&mut bad_plan);
        let bad_out = root.join("build/bad.zestenginepak");
        write_package(&bad_plan, &bad_out).unwrap();
        let mut bad_reader = PackageReader::open(&bad_out, &ContentLimits::for_kind).unwrap();
        let ex = ContentCache::new(temp_dir("geocache2")).extract(&mut bad_reader, &opts).unwrap();
        assert!(ex.warnings.iter().any(|w| w.contains("rejected")), "{:?}", ex.warnings);
        assert!(!crate::geo_cache::geo_path(&ex.dir.join("files/models/DamagedHelmet.glb")).unwrap().exists());

        // The DAG budget is a content limit.
        let tight = ContentLimits { max_dag_bytes: 1000, ..ContentLimits::for_kind(ContentKind::World) };
        let mut o = options(&root, scene.clone(), ContentKind::World);
        o.limits = tight;
        let limited = plan_package(&o).unwrap();
        assert!(limited.errors.iter().any(|e| e.contains("cluster data")), "{:?}", limited.errors);

        // Format 1 packages (no cluster data, no new stats) still open.
        let mut old = plan_package(&options(&root, scene, ContentKind::World)).unwrap();
        old.files.remove(blob_path);
        old.manifest.format_version = 1;
        old.manifest.stats.clusters = 0;
        old.manifest.stats.dag_bytes = 0;
        repack(&mut old);
        let old_out = root.join("build/old.zestenginepak");
        write_package(&old, &old_out).unwrap();
        let mut old_reader = PackageReader::open(&old_out, &ContentLimits::for_kind).unwrap();
        let ex = ContentCache::new(temp_dir("geocache3")).extract(&mut old_reader, &opts).unwrap();
        assert!(!crate::geo_cache::geo_path(&ex.dir.join("files/models/DamagedHelmet.glb")).unwrap().exists());
        let stats: PackageStats = ron::from_str("(triangles: 1, texture_bytes: 2, file_bytes: 3, objects: 4, shaders: 5)").unwrap();
        assert_eq!((stats.clusters, stats.dag_bytes), (0, 0));
        // Limits written before the DAG budget existed read back with its default.
        let text = ron::to_string(&ContentLimits::for_kind(ContentKind::Avatar)).unwrap();
        let at = text.find(",max_dag_bytes:").expect("serialised");
        let digits = text[at + 15..].bytes().take_while(u8::is_ascii_digit).count();
        let without = format!("{}{}", &text[..at], &text[at + 15 + digits..]);
        let limits: ContentLimits = ron::from_str(&without).unwrap();
        assert_eq!(limits.max_dag_bytes, 256 * MIB);
        let _ = std::fs::remove_dir_all(&root);
    }

    fn repack(plan: &mut PackagePlan) {
        plan.manifest.files = plan
            .files
            .iter()
            .map(|(p, b)| FileRecord { path: p.clone(), size: b.len() as u64, hash: sha256_hex(b) })
            .collect();
        plan.manifest.content_hash = content_hash(&plan.manifest.files);
    }

    #[test]
    fn manifest_validation_catches_tampering_and_versions() {
        let (root, scene) = sample_project();
        let plan = plan_package(&options(&root, scene, ContentKind::World)).unwrap();
        let limits = ContentLimits::for_kind(ContentKind::World);
        validate_manifest(&plan.manifest, &limits).unwrap();

        let mut m = plan.manifest.clone();
        m.files[0].hash = "0".repeat(64);
        assert!(format!("{:#}", validate_manifest(&m, &limits).unwrap_err()).contains("hash"));

        let mut m = plan.manifest.clone();
        m.files[0].path = "files/../../evil.scene".into();
        m.content_hash = content_hash(&m.files);
        assert!(format!("{:#}", validate_manifest(&m, &limits).unwrap_err()).contains("unsafe"));

        let mut m = plan.manifest.clone();
        m.format_version = PACKAGE_FORMAT_VERSION + 1;
        assert!(format!("{:#}", validate_manifest(&m, &limits).unwrap_err()).contains("newer"));

        let mut m = plan.manifest.clone();
        m.engine_format = ENGINE_CONTENT_FORMAT + 1;
        assert!(validate_manifest(&m, &limits).is_err());

        // A file whose bytes differ from the manifest is caught on read.
        let mut bad = PackagePlan { manifest: plan.manifest.clone(), files: plan.files.clone(), warnings: vec![], errors: vec![] };
        bad.files.insert("files/audio/beep.wav".into(), b"RIFF0000WAVEfmX ".to_vec());
        let out = root.join("tampered.zestenginepak");
        write_package(&bad, &out).unwrap();
        let mut reader = PackageReader::open(&out, &ContentLimits::for_kind).unwrap();
        assert!(format!("{:#}", reader.read_asset("audio/beep.wav").unwrap_err()).contains("mismatch"));
        let cache = ContentCache::new(temp_dir("tamper-cache"));
        let known = |_: &str| true;
        let opts = ExtractOptions { known_component: &known, components_fingerprint: String::new(), server_side: false };
        assert!(cache.extract(&mut reader, &opts).is_err());
        assert!(cache.entries().is_empty(), "failed extraction leaves nothing behind");
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&cache.root);
    }

    #[test]
    fn limits_are_enforced_at_build_and_open() {
        let (root, scene) = sample_project();
        let mut opts = options(&root, scene.clone(), ContentKind::World);
        opts.limits.max_texture_bytes = 16;
        let plan = plan_package(&opts).unwrap();
        assert!(plan.errors.iter().any(|e| e.contains("textures")), "{:?}", plan.errors);
        assert!(write_package(&plan, &root.join("x.zestenginepak")).is_err());

        let plan = plan_package(&options(&root, scene, ContentKind::World)).unwrap();
        let out = root.join("ok.zestenginepak");
        write_package(&plan, &out).unwrap();
        let tiny = |kind: ContentKind| ContentLimits { max_objects: 1, ..ContentLimits::for_kind(kind) };
        let err = PackageReader::open(&out, &tiny).err().expect("over the object limit");
        assert!(format!("{err:#}").contains("objects"), "{err:#}");
        let small_file = |kind: ContentKind| ContentLimits { max_package_bytes: 10, ..ContentLimits::for_kind(kind) };
        assert!(PackageReader::open(&out, &small_file).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn avatar_needs_one_root_and_gets_a_descriptor() {
        let (root, mut scene) = sample_project();
        let plan = plan_package(&options(&root, scene.clone(), ContentKind::Avatar)).unwrap();
        assert!(plan.errors.iter().any(|e| e.contains("exactly one root")), "{:?}", plan.errors);
        scene.entries[1].parent = Some(0);
        let plan = plan_package(&options(&root, scene, ContentKind::Avatar)).unwrap();
        assert!(plan.is_ok(), "{:?}", plan.errors);
        assert!(plan.manifest.components.contains(&AVATAR_DESCRIPTOR.to_owned()));
        let _ = std::fs::remove_dir_all(&root);
    }


    /// A `.wasm` component module with a manifest section, from WAT.
    fn wasm_module(imports: &str) -> Vec<u8> {
        let manifest: String = r#"(name: "T", actions: [(name: "Go", ugc_safe: true)])"#.bytes().map(|b| format!("\\{b:02x}")).collect();
        wat::parse_str(format!(
            r#"(module (@custom "zest_manifest" "{manifest}") {imports}
              (memory (export "memory") 1)
              (func (export "zest_alloc") (param i32) (result i32) i32.const 64)
              (func (export "zest_invoke") (param i32 i32 i32) (result i32) i32.const 0))"#
        ))
        .unwrap()
    }

    #[test]
    fn wasm_modules_are_packaged_and_validated_at_build_and_extraction() {
        let (root, mut scene) = sample_project();
        std::fs::create_dir_all(root.join("logic")).unwrap();
        std::fs::write(root.join("logic/door.wasm"), wasm_module(r#"(import "host" "log" (func (param i32 i32)))"#)).unwrap();
        scene.entries[0].components.push(ComponentData::new("Wasm Script", "(module:\"logic/door.wasm\")"));
        let plan = plan_package(&options(&root, scene.clone(), ContentKind::World)).unwrap();
        assert!(plan.is_ok(), "{:?}", plan.errors);
        assert!(plan.manifest.files.iter().any(|f| f.path == "files/logic/door.wasm"));
        assert!(plan.manifest.components.contains(&"Wasm Script".to_owned()));

        // Extraction keeps the module and points the component at it.
        let out = root.join("wasm.zestenginepak");
        write_package(&plan, &out).unwrap();
        let cache = ContentCache::new(temp_dir("wasm-cache"));
        let known = |_: &str| true;
        let opts = ExtractOptions { known_component: &known, components_fingerprint: String::new(), server_side: false };
        let mut reader = PackageReader::open(&out, &ContentLimits::for_kind).unwrap();
        let ex = cache.extract(&mut reader, &opts).unwrap();
        let scene_text = std::fs::read_to_string(&ex.entry).unwrap();
        assert!(scene_text.contains(&ex.dir.join("files/logic/door.wasm").to_string_lossy().into_owned()));

        // Avatars carry no logic; a module in one is a build error.
        let mut avatar = scene.clone();
        avatar.entries.truncate(1);
        let plan_a = plan_package(&options(&root, avatar, ContentKind::Avatar)).unwrap();
        assert!(plan_a.errors.iter().any(|e| e.contains("WASM modules")), "{:?}", plan_a.errors);

        // Imports outside the `host` namespace fail the build...
        std::fs::write(root.join("logic/door.wasm"), wasm_module(r#"(import "wasi_snapshot_preview1" "fd_write" (func (param i32 i32 i32 i32) (result i32)))"#)).unwrap();
        let bad = plan_package(&options(&root, scene.clone(), ContentKind::World)).unwrap();
        assert!(bad.errors.iter().any(|e| e.contains("logic/door.wasm") && e.contains("host")), "{:?}", bad.errors);
        // ...and a package that sneaks one in anyway is refused on extraction.
        let mut forged = plan;
        forged.files.insert("files/logic/door.wasm".into(), std::fs::read(root.join("logic/door.wasm")).unwrap());
        repack(&mut forged);
        write_package(&forged, &out).unwrap();
        let mut reader = PackageReader::open(&out, &ContentLimits::for_kind).unwrap();
        let err = cache.extract(&mut reader, &opts).unwrap_err();
        assert!(format!("{err:#}").contains("door.wasm"), "{err:#}");
        // The size limit applies too.
        let tight = ContentLimits { max_wasm_bytes: 10, ..ContentLimits::for_kind(ContentKind::World) };
        assert!(validate_wasm_module(&wasm_module(""), &tight).is_err());
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&cache.root);
    }

    #[test]
    fn unsafe_package_contents_are_rejected_or_sanitized() {
        let (root, scene) = sample_project();
        let mut plan = plan_package(&options(&root, scene, ContentKind::World)).unwrap();
        // A material pointing outside the package gets the path blanked.
        let mat = String::from_utf8(plan.files["files/materials/ferris.material"].clone()).unwrap();
        let evil = mat.replace("textures/orm.png", "/home/someone/.ssh/id_rsa");
        plan.files.insert("files/materials/ferris.material".into(), evil.into_bytes());
        plan.files.insert("files/lib/evil.so".into(), b"\x7fELF".to_vec());
        repack(&mut plan);
        let out = root.join("sanitize.zestenginepak");
        write_package(&plan, &out).unwrap();
        let cache = ContentCache::new(temp_dir("sanitize-cache"));
        let known = |_: &str| true;
        let opts = ExtractOptions { known_component: &known, components_fingerprint: String::new(), server_side: false };
        let mut reader = PackageReader::open(&out, &ContentLimits::for_kind).unwrap();
        let ex = cache.extract(&mut reader, &opts).unwrap();
        assert!(ex.warnings.iter().any(|w| w.contains("id_rsa")), "{:?}", ex.warnings);
        assert!(ex.warnings.iter().any(|w| w.contains("evil.so")));
        assert!(!ex.dir.join("files/lib/evil.so").exists());
        let m = crate::load_material_file(ex.dir.join("files/materials/ferris.material")).unwrap();
        assert_eq!(m.textures.orm, Some(PathBuf::new()));

        // A glTF whose buffer lives outside the package is refused.
        let gltf = br#"{"asset":{"version":"2.0"},"buffers":[{"uri":"../../outside.bin","byteLength":4}]}"#;
        plan.files.insert("files/models/m.gltf".into(), gltf.to_vec());
        plan.files.remove("files/lib/evil.so");
        repack(&mut plan);
        write_package(&plan, &out).unwrap();
        let mut reader = PackageReader::open(&out, &ContentLimits::for_kind).unwrap();
        let err = cache.extract(&mut reader, &opts).unwrap_err();
        assert!(format!("{err:#}").contains("not inside the package"), "{err:#}");

        // Shader includes are refused at build and load.
        assert!(validate_shader_source("#include \"/etc/passwd\"\nvoid main(){}").is_err());
        assert!(validate_shader_source("void main(){}").is_ok());
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&cache.root);
    }
}
