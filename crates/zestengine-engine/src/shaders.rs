//! Runtime custom-shader library: loads `.frag` (GLSL) and `.hlsl` files from
//! anywhere in the project, compiles them via glslc, registers the SPIR-V with
//! the renderer, and exposes reflected property metadata for the inspector.
//! Polls file mtimes for hot reload.
//!
//! Keyword variants: a shader compiles once to a specialization-constant
//! module; each material's keyword set becomes its own lazily built, cached
//! pipeline (the driver folds the constants and drops disabled features).
//! Game builds additionally bake the exact keyword sets their materials use
//! into frozen-constant SPIR-V sidecars (`bake_project_variants`), which the
//! library prefers when present.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::Arc;
use std::time::SystemTime;

use zestengine_assets::{
    KeywordSet, PropStorage, ShaderCondOp, ShaderCondition, ShaderLang, ShaderPropKind,
    ShaderSource, TextureDefault,
};
use zestengine_core::{
    MaterialTexturePaths, ShaderCmpUi, ShaderCondUi, ShaderGroupUi, ShaderPresetUi,
    ShaderPropKindUi, ShaderPropUi, ShaderUiInfo,
};
use zestengine_render::{CustomTextureDefault, Renderer, ShaderId};

/// A shader parsed + compiled on a worker thread (or the error).
pub type PrecompiledShader = Result<(ShaderSource, Vec<u8>), String>;

/// Compile every custom shader a scene's materials reference (worker side).
pub fn precompile_scene_shaders(file: &zestengine_assets::SceneFile, project_root: &Path) -> Vec<(String, PrecompiledShader)> {
    let mut names: Vec<String> = Vec::new();
    let mut add = |name: &str| {
        if zestengine_assets::is_shader_path(Path::new(name)) && !names.iter().any(|n| n == name) {
            names.push(name.to_owned());
        }
    };
    for entry in &file.entries {
        for m in &zestengine_assets::material_refs(entry) {
            match m {
                zestengine_assets::MaterialRef::Inline { shader, .. } => add(shader),
                zestengine_assets::MaterialRef::File(path) => {
                    if let Ok(mf) = zestengine_assets::load_material_file(zestengine_core::asset_ref::resolve_path(project_root, path)) {
                        add(&mf.shader);
                    }
                }
                zestengine_assets::MaterialRef::Model => {}
            }
        }
    }
    names
        .into_iter()
        .map(|rel| {
            let compiled = zestengine_assets::load_shader_file(project_root.join(&rel)).map_err(|e| format!("{e:#}"));
            (rel, compiled)
        })
        .collect()
}

pub struct ShaderEntry {
    /// Parsed source (None when the file couldn't be read/parsed/compiled).
    pub source: Option<Arc<ShaderSource>>,
    /// The layout before the last hot reload (values are remapped by name).
    pub previous: Option<Arc<ShaderSource>>,
    /// Inspector view (display name, properties, error).
    pub ui: ShaderUiInfo,
    /// Renderer registration of the specialization-constant module; None while broken.
    pub id: Option<ShaderId>,
    mtime: Option<SystemTime>,
    /// Baked (frozen-keyword) variant modules; None = no sidecar for that set.
    variants: HashMap<KeywordSet, Option<ShaderId>>,
}

impl ShaderEntry {
    fn new(source: ShaderSource, id: ShaderId, mtime: Option<SystemTime>) -> Self {
        Self {
            ui: ui_from_source(&source),
            source: Some(Arc::new(source)),
            previous: None,
            id: Some(id),
            mtime,
            variants: HashMap::new(),
        }
    }

    fn broken(rel: &str, error: String, mtime: Option<SystemTime>) -> Self {
        Self {
            source: None,
            previous: None,
            ui: ShaderUiInfo {
                display_name: rel.to_owned(),
                language: language_label(rel),
                error: Some(error),
                ..Default::default()
            },
            id: None,
            mtime,
            variants: HashMap::new(),
        }
    }

    /// Defaults for `MaterialModel::custom_values` (zeros while broken, so
    /// values survive a broken intermediate save during hot editing).
    pub fn defaults(&self) -> Vec<f32> {
        self.source
            .as_ref()
            .map(|s| s.defaults())
            .unwrap_or_else(|| vec![0.0; zestengine_assets::SHADER_VALUE_BASE])
    }

    /// Make `values` match the current layout: carried over by name across a
    /// hot reload, otherwise reset to defaults when the length is off.
    pub fn fit_values(&self, values: &mut Vec<f32>) {
        let Some(source) = &self.source else {
            return;
        };
        if values.len() == source.value_len {
            return;
        }
        *values = match &self.previous {
            Some(old) if values.len() == old.value_len => source.remap_from(old, values),
            _ => source.defaults(),
        };
    }

    /// Carry values from the pre-reload layout by name (once per reload).
    pub fn remap_after_reload(&self, values: &mut Vec<f32>) {
        if let (Some(source), Some(old)) = (&self.source, &self.previous)
            && values.len() == old.value_len
        {
            *values = source.remap_from(old, values);
        }
    }

    /// Order `paths.shader` to this shader's texture properties (kept by name).
    pub fn sync_textures(&self, paths: &mut MaterialTexturePaths) {
        let Some(source) = &self.source else {
            return;
        };
        let want: Vec<&str> = source.textures().map(|p| p.name.as_str()).collect();
        if paths.shader.iter().map(|(n, _)| n.as_str()).eq(want.iter().copied()) {
            return;
        }
        let old = std::mem::take(&mut paths.shader);
        paths.shader = want
            .iter()
            .map(|&n| {
                let path = old.iter().find(|(o, _)| o == n).and_then(|(_, p)| p.clone());
                (n.to_owned(), path)
            })
            .collect();
    }
}

/// Project-relative shader path -> compiled entry.
#[derive(Default)]
pub struct ShaderLibrary {
    entries: HashMap<String, ShaderEntry>,
}

impl ShaderLibrary {
    /// Get a shader, loading and compiling it on first use.
    pub fn resolve(&mut self, renderer: &mut Renderer, project_root: &Path, rel: &str) -> &ShaderEntry {
        if !self.entries.contains_key(rel) {
            let entry = load_entry(renderer, project_root, rel);
            self.entries.insert(rel.to_owned(), entry);
        }
        &self.entries[rel]
    }

    /// Install a shader compiled off the main thread (scene loader worker), so
    /// `resolve` finds it without running `glslc` on the main thread.
    pub fn preload(&mut self, renderer: &mut Renderer, project_root: &Path, rel: &str, compiled: PrecompiledShader) {
        if self.entries.contains_key(rel) {
            return;
        }
        let mtime = std::fs::metadata(project_root.join(rel)).and_then(|m| m.modified()).ok();
        let result = compiled.and_then(|(source, spirv)| {
            register_source(renderer, &source, &spirv, None)
                .map(|id| (source, id))
                .map_err(|e| format!("{e:#}"))
        });
        let entry = match result {
            Ok((source, id)) => ShaderEntry::new(source, id, mtime),
            Err(e) => ShaderEntry::broken(rel, e, mtime),
        };
        self.entries.insert(rel.to_owned(), entry);
    }

    /// Already-loaded shader (no compile side effects); used on save paths.
    pub fn get(&self, rel: &str) -> Option<&ShaderEntry> {
        self.entries.get(rel)
    }

    /// Module for a material's keyword set: a baked frozen-keyword variant
    /// when the build shipped one, else the specialization-constant module.
    pub fn variant(&mut self, renderer: &mut Renderer, project_root: &Path, rel: &str, keywords: &KeywordSet) -> Option<ShaderId> {
        let entry = self.entries.get_mut(rel)?;
        let base = entry.id?;
        let baked = *entry.variants.entry(*keywords).or_insert_with(|| {
            let source = entry.source.as_ref();
            zestengine_assets::load_baked_variant(project_root.join(rel), keywords)
                .and_then(|spirv| match source {
                    Some(source) => register_source(renderer, source, &spirv, Some(keywords)).ok(),
                    None => renderer.register_shader(&spirv).ok(),
                })
        });
        Some(baked.unwrap_or(base))
    }

    /// Re-check mtimes; recompile changed files. Returns the shader paths
    /// that changed (their materials need re-applying).
    pub fn poll_reload(&mut self, renderer: &mut Renderer, project_root: &Path) -> Vec<String> {
        let stale: Vec<String> = self
            .entries
            .iter()
            .filter(|(rel, entry)| {
                let mtime = std::fs::metadata(project_root.join(rel)).and_then(|m| m.modified()).ok();
                mtime != entry.mtime
            })
            .map(|(rel, _)| rel.clone())
            .collect();
        for rel in &stale {
            zestengine_render::hitch::event("shader_reload", "", 0);
            let _h = zestengine_render::hitch::scope("shader_reload");
            tracing::info!("reloading shader {rel}");
            let mut entry = load_entry(renderer, project_root, rel);
            let old = self.entries.remove(rel);
            if let Some(old) = old {
                entry.previous = old.source.or(old.previous);
            }
            self.entries.insert(rel.clone(), entry);
        }
        stale
    }
}

fn language_label(rel: &str) -> &'static str {
    match zestengine_assets::shader_lang_for(Path::new(rel)) {
        ShaderLang::Glsl => "GLSL",
        ShaderLang::Hlsl => "HLSL",
    }
}

/// Register a compiled shader; one with a `// @vertex` section also gets its vertex hook
/// (`keywords` freezes the keyword constants of a baked variant).
fn register_source(renderer: &mut Renderer, source: &ShaderSource, spirv: &[u8], keywords: Option<&KeywordSet>) -> anyhow::Result<ShaderId> {
    match zestengine_assets::vertex_hook_glsl(source, keywords) {
        Some(hook) => renderer.register_shader_with_vertex(spirv, &hook),
        None => renderer.register_shader(spirv),
    }
}

fn load_entry(renderer: &mut Renderer, project_root: &Path, rel: &str) -> ShaderEntry {
    let abs = project_root.join(rel);
    let mtime = std::fs::metadata(&abs).and_then(|m| m.modified()).ok();
    match zestengine_assets::load_shader_file(&abs).and_then(|(source, spirv)| {
        let id = register_source(renderer, &source, &spirv, None)?;
        Ok((source, id))
    }) {
        Ok((source, id)) => ShaderEntry::new(source, id, mtime),
        Err(e) => {
            tracing::error!("shader {rel}: {e:#}");
            ShaderEntry::broken(rel, format!("{e:#}"), mtime)
        }
    }
}

/// Renderer fallback for a texture property's declared default.
pub fn texture_default(d: TextureDefault) -> CustomTextureDefault {
    match d {
        TextureDefault::White => CustomTextureDefault::White,
        TextureDefault::Black => CustomTextureDefault::Black,
        TextureDefault::Normal => CustomTextureDefault::Normal,
        TextureDefault::Gray => CustomTextureDefault::Gray,
    }
}

fn cond_ui(c: &ShaderCondition) -> ShaderCondUi {
    ShaderCondUi(
        c.0.iter()
            .map(|all| {
                all.iter()
                    .map(|t| {
                        let op = match t.op {
                            ShaderCondOp::Eq => ShaderCmpUi::Eq,
                            ShaderCondOp::Ne => ShaderCmpUi::Ne,
                            ShaderCondOp::Lt => ShaderCmpUi::Lt,
                            ShaderCondOp::Le => ShaderCmpUi::Le,
                            ShaderCondOp::Gt => ShaderCmpUi::Gt,
                            ShaderCondOp::Ge => ShaderCmpUi::Ge,
                        };
                        (t.offset, op, t.value)
                    })
                    .collect()
            })
            .collect(),
    )
}

fn ui_from_source(source: &ShaderSource) -> ShaderUiInfo {
    let mut tex_slot = zestengine_assets::SHADER_TEXTURE_SLOT_BASE;
    ShaderUiInfo {
        display_name: source.display_name.clone(),
        language: match source.lang {
            ShaderLang::Glsl => "GLSL",
            ShaderLang::Hlsl => "HLSL",
        },
        props: source
            .props
            .iter()
            .map(|p| ShaderPropUi {
                name: p.name.clone(),
                label: p.label.clone(),
                kind: match &p.kind {
                    ShaderPropKind::Float { min, max } => ShaderPropKindUi::Float { min: *min, max: *max },
                    ShaderPropKind::Drag => ShaderPropKindUi::Drag,
                    ShaderPropKind::Int { min, max } => ShaderPropKindUi::Int { min: *min, max: *max },
                    ShaderPropKind::Toggle => ShaderPropKindUi::Toggle,
                    ShaderPropKind::Enum(o) => ShaderPropKindUi::Enum(o.clone()),
                    ShaderPropKind::Color { hdr } => ShaderPropKindUi::Color { hdr: *hdr },
                    ShaderPropKind::Color3 { hdr } => ShaderPropKindUi::Color3 { hdr: *hdr },
                    ShaderPropKind::Vector(n) => ShaderPropKindUi::Vector(*n),
                    ShaderPropKind::Texture { default, .. } => {
                        let slot = tex_slot;
                        tex_slot += 1;
                        ShaderPropKindUi::Texture {
                            slot,
                            default: match default {
                                TextureDefault::White => "white",
                                TextureDefault::Black => "black",
                                TextureDefault::Normal => "normal",
                                TextureDefault::Gray => "gray",
                            },
                        }
                    }
                },
                offset: p.offset,
                keyword: matches!(p.storage, PropStorage::Keyword { .. }),
                group: p.group,
                header: p.header.clone(),
                show_if: cond_ui(&p.show_if),
                tooltip: p.tooltip.clone(),
            })
            .collect(),
        groups: source
            .groups
            .iter()
            .map(|g| ShaderGroupUi { label: g.label.clone(), toggle: g.toggle, show_if: cond_ui(&g.show_if) })
            .collect(),
        presets: source
            .presets
            .iter()
            .map(|p| ShaderPresetUi {
                name: p.name.clone(),
                values: p.values.iter().map(|(pi, v)| (source.props[*pi].offset, v.clone())).collect(),
            })
            .collect(),
        error: None,
    }
}

/// Bake every (shader, keyword set) the project's scenes and material files
/// use into frozen-keyword SPIR-V sidecars under `root` (a staged build or the
/// project). Returns the number of variants written.
pub fn bake_project_variants(root: &Path, log: &mut impl FnMut(String)) -> usize {
    let mut materials: Vec<(String, std::collections::BTreeMap<String, Vec<f32>>)> = Vec::new();
    let mut files = Vec::new();
    collect_files(root, 0, &mut files);
    for path in &files {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if ext == zestengine_assets::MATERIAL_EXTENSION {
            if let Ok(mf) = zestengine_assets::load_material_file(path) {
                materials.push((mf.shader, mf.custom));
            }
        } else if ext == zestengine_assets::SCENE_EXTENSION
            && let Ok(scene) = zestengine_assets::load_scene_file(path)
        {
            for entry in &scene.entries {
                for m in &zestengine_assets::material_refs(entry) {
                    if let zestengine_assets::MaterialRef::Inline { shader, custom, .. } = m {
                        materials.push((shader.clone(), custom.clone()));
                    }
                }
            }
        }
    }
    let mut sets: BTreeSet<(String, KeywordSet)> = BTreeSet::new();
    let mut sources: HashMap<String, Option<ShaderSource>> = HashMap::new();
    for (shader, custom) in materials {
        if !zestengine_assets::is_shader_path(Path::new(&shader)) {
            continue;
        }
        let source = sources.entry(shader.clone()).or_insert_with(|| {
            zestengine_assets::load_shader_file(root.join(&shader)).ok().map(|(s, _)| s)
        });
        if let Some(source) = source
            && source.keyword_count() > 0
        {
            sets.insert((shader, source.keywords(&source.pack(&custom))));
        }
    }
    let mut baked = 0;
    for (shader, keywords) in sets {
        match zestengine_assets::bake_shader_variant(root.join(&shader), &keywords) {
            Ok(_) => baked += 1,
            Err(e) => log(format!("warning: shader variant {shader} not baked: {e:#}")),
        }
    }
    baked
}

fn collect_files(dir: &Path, depth: usize, out: &mut Vec<std::path::PathBuf>) {
    if depth > 8 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') || name == "target" || name.ends_with(".variants") {
            continue;
        }
        if path.is_dir() {
            collect_files(&path, depth + 1, out);
        } else {
            out.push(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    use super::*;

    /// Every shipped shader compiles through the same path the engine uses
    /// (parse + preamble + glslc), as the specialization module and with all
    /// keywords both off and on.
    #[test]
    fn shipped_shaders_compile() {
        let root = repo_root();
        let mut files = Vec::new();
        collect_files(&root.join("shaders"), 0, &mut files);
        collect_files(&root.join("examples"), 0, &mut files);
        let shaders: Vec<_> = files.into_iter().filter(|p| zestengine_assets::is_shader_path(p)).collect();
        assert!(shaders.iter().any(|p| p.ends_with("zestengine_toon.frag")), "zestengine_toon.frag missing");
        for path in &shaders {
            let text = std::fs::read_to_string(path).unwrap();
            let source = zestengine_assets::parse_shader_source_lang(&text, "s", zestengine_assets::shader_lang_for(path))
                .unwrap_or_else(|e| panic!("{}: {e:#}", path.display()));
            let label = path.display().to_string();
            zestengine_assets::compile_shader(&source, &label).unwrap_or_else(|e| panic!("{e:#}"));
            for on in [0u8, 1] {
                let mut kw = [0u8; zestengine_assets::MAX_SHADER_KEYWORDS];
                for p in &source.props {
                    if let PropStorage::Keyword { id } = p.storage {
                        kw[id as usize] = on;
                    }
                }
                zestengine_assets::compile_shader_variant(&source, &label, Some(&kw)).unwrap_or_else(|e| panic!("{e:#}"));
            }
        }
    }

    /// The built-in standard shader compiles with the engine's glslc flags.
    #[cfg(feature = "client")]
    #[test]
    fn standard_shader_compiles() {
        let dir = Path::new(zestengine_render::ENGINE_SHADER_DIR);
        for name in ["standard.frag", "standard.vert"] {
            let out = std::process::Command::new("glslc")
                .args(["--target-env=vulkan1.3", "-O", "-o", "-"])
                .arg(dir.join(name))
                .output()
                .unwrap();
            assert!(out.status.success(), "{name}: {}", String::from_utf8_lossy(&out.stderr));
        }
    }

    #[test]
    fn fit_values_remaps_across_reload() {
        let old = zestengine_assets::parse_shader_source("// @property a Float = 1\n// @property b Float = 2\n", "o").unwrap();
        let new = zestengine_assets::parse_shader_source("// @property c Float = 3\n// @property b Float = 2\n// @property a Float = 1\n", "n").unwrap();
        let mut values = old.defaults();
        values[old.prop("b").unwrap().offset] = 7.0;
        let entry = ShaderEntry {
            ui: ShaderUiInfo::default(),
            source: Some(Arc::new(new.clone())),
            previous: Some(Arc::new(old)),
            id: None,
            mtime: None,
            variants: HashMap::new(),
        };
        entry.fit_values(&mut values);
        assert_eq!(values.len(), new.value_len);
        assert_eq!(values[new.prop("b").unwrap().offset], 7.0);
        assert_eq!(values[new.prop("c").unwrap().offset], 3.0);
    }
}
