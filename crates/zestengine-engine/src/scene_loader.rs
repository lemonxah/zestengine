//! Background scene loading shared by the editor and the standalone runtime.
//!
//! The worker half ([`load_on_worker`]) does every slow part off the main
//! thread: reading the `.scene`, importing models, uploading their meshes and
//! textures on the transfer queue, decoding + BC-encoding material textures,
//! decoding the skybox, reading baked lighting, and compiling custom shaders.
//! The main thread then [`install`]s the result (handles only, plus material
//! setup) and applies the skybox / baked lighting over the following frames
//! ([`PostLoad`]), so the UI never stalls for a whole load.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, channel};

use zestengine_core::ComponentRegistry;
use zestengine_render::{Renderer, TextureData};

use crate::scene::{BakedData, LoadedScene};
use crate::shaders::{PrecompiledShader, ShaderLibrary};

/// Skybox pixels decoded on the worker.
pub enum SkyboxData {
    Cube(Vec<TextureData>),
    Equirect(TextureData),
    None,
}

impl SkyboxData {
    /// Decode the skybox a scene uses (6-face cubemap wins when complete).
    pub fn read(skybox: Option<&str>, faces: Option<&[String; 6]>, root: &Path) -> Self {
        if let Some(faces) = faces.filter(|f| f.iter().all(|s| !s.is_empty())) {
            let loaded: Result<Vec<_>, _> = faces
                .iter()
                .map(|rel| zestengine_assets::load_texture_file(zestengine_core::asset_ref::resolve_path(root, rel), true))
                .collect();
            match loaded {
                Ok(data) => return Self::Cube(data),
                Err(e) => tracing::error!("loading cubemap skybox: {e:#}"),
            }
        }
        match skybox {
            Some(rel) => match zestengine_assets::load_texture_file(zestengine_core::asset_ref::resolve_path(root, rel), true) {
                Ok(data) => Self::Equirect(data),
                Err(e) => {
                    tracing::error!("loading skybox {rel}: {e:#}");
                    Self::None
                }
            },
            None => Self::None,
        }
    }

    pub fn install(&self, renderer: &mut Renderer) {
        let result = match self {
            Self::Cube(data) => {
                let refs: [&TextureData; 6] = std::array::from_fn(|i| &data[i]);
                renderer.set_skybox_cube(refs)
            }
            Self::Equirect(data) => renderer.set_skybox(Some(data)),
            Self::None => renderer.set_skybox(None),
        };
        if let Err(e) = result {
            tracing::error!("setting skybox: {e:#}");
            renderer.report_device_lost(&e);
        }
    }
}

/// Everything the worker produced for one scene.
pub struct LoadedSceneData {
    pub path: PathBuf,
    pub file: zestengine_assets::SceneFile,
    pub models: HashMap<String, zestengine_assets::Scene>,
    pub prepared: Option<HashMap<String, zestengine_render::PreparedScene>>,
    pub material_textures: Option<(Vec<(PathBuf, bool)>, zestengine_render::PreparedScene)>,
    pub skybox: SkyboxData,
    pub baked: Option<BakedData>,
    pub shaders: Vec<(String, PrecompiledShader)>,
    /// `<scene>.audiobake` sidecar (Steam Audio probe batch), if present.
    pub audio_bake: Option<Vec<u8>>,
    /// `<scene>.occlusion` sidecar (baked visibility), if present.
    pub occlusion: Option<crate::occlusion::OcclusionData>,
    /// `<scene>.reflprobe` sidecar (baked reflection cube faces), if present.
    pub reflection: crate::refl_slots::ReflProbeSet,
    /// The side the scene is built for (NET-3): content of the other side is
    /// stripped on install. `Both` keeps everything.
    pub net_side: zestengine_core::NetRole,
}

/// Load a scene's CPU / transfer-queue side. `extra_models` are additional
/// model paths to import (objects carried over from the previous scene).
pub fn load_on_worker(
    path: &Path,
    root: &Path,
    uploader: Option<&zestengine_render::Uploader>,
    extra_models: &[String],
    status: &dyn Fn(&str),
) -> Result<LoadedSceneData, String> {
    load_on_worker_checked(path, root, uploader, extra_models, status, &|_| Ok(()))
}

/// What [`load_on_worker_checked`] has decoded so far, before it is uploaded.
pub enum LoadStage<'a> {
    Models(&'a zestengine_assets::SceneFile, &'a HashMap<String, zestengine_assets::Scene>),
    Textures(&'a [zestengine_render::CompressedTexture]),
}

/// [`load_on_worker`] with a `check` run on the decoded data before each GPU
/// upload (user-content resource limits); an error aborts the load.
pub fn load_on_worker_checked(
    path: &Path,
    root: &Path,
    uploader: Option<&zestengine_render::Uploader>,
    extra_models: &[String],
    status: &dyn Fn(&str),
    check: &dyn Fn(LoadStage) -> Result<(), String>,
) -> Result<LoadedSceneData, String> {
    load_on_worker_impl(path, root, uploader, extra_models, status, check, true)
}

/// The GPU-free load of a dedicated server (and headless tests): everything
/// [`load_on_worker`] reads except what only a renderer uses. No upload, no
/// shader compile, no skybox decode and no baked lighting; the audio bake,
/// occlusion and reflection sidecars are kept for the caller to ignore or use.
/// Feed the result to [`install`] with `None` for the renderer.
/// `Server` strips client-only content when the scene is installed; `Both` keeps
/// everything (headless tests).
pub fn load_on_worker_cpu_for(
    path: &Path,
    root: &Path,
    extra_models: &[String],
    status: &dyn Fn(&str),
    side: zestengine_core::NetRole,
) -> Result<LoadedSceneData, String> {
    let mut data = load_on_worker_impl(path, root, None, extra_models, status, &|_| Ok(()), false)?;
    data.net_side = side;
    Ok(data)
}

fn load_on_worker_impl(
    path: &Path,
    root: &Path,
    uploader: Option<&zestengine_render::Uploader>,
    extra_models: &[String],
    status: &dyn Fn(&str),
    check: &dyn Fn(LoadStage) -> Result<(), String>,
    renderable: bool,
) -> Result<LoadedSceneData, String> {
    status("Loading 3D models…");
    let mut file = zestengine_assets::load_scene_file(path).map_err(|e| format!("{e:#}"))?;
    if let Some(n) = crate::bench_avatars::requested() {
        crate::bench_avatars::inject(&mut file, root, n);
    }
    // Saved scenes store linked prefab instances compacted (root + link);
    // build their objects here, before model handling, so every load path
    // (editor, player, content packages, server) sees plain entries.
    crate::prefab::expand_scene_file(&mut file, root, &Default::default());
    // Models that moved or were renamed are found again by their meta GUID.
    for (old, new) in file.relocate_models(root) {
        tracing::info!("model moved: {old} -> {new}");
    }
    let mut models = LoadedScene::parse_scene_models(&file, root).map_err(|e| format!("{e:#}"))?;
    file.expand_models(&|p| models.get(p).map(zestengine_assets::ModelPart::of));
    for rel in extra_models {
        if !models.contains_key(rel) {
            match zestengine_assets::load_model_with_meta(root.join(rel)) {
                Ok(m) => {
                    models.insert(rel.clone(), m);
                }
                Err(e) => tracing::warn!("carried-over model {rel}: {e:#}"),
            }
        }
    }
    check(LoadStage::Models(&file, &models))?;
    let mut prepared = None;
    let mut material_textures = None;
    if let Some(up) = uploader {
        status("Uploading 3D models…");
        let mut map = HashMap::new();
        for (p, scene) in &models {
            let model_path = root.join(p);
            let textures: Vec<_> = scene
                .textures
                .iter()
                .enumerate()
                .map(|(k, t)| zestengine_assets::compress_embedded_texture(Some(&model_path), k, t))
                .collect();
            let prep = up
                .prepare(&scene.meshes, &textures, !scene.skeletons.is_empty())
                .map_err(|e| format!("{e:#}"))?;
            map.insert(p.clone(), prep);
        }
        prepared = Some(map);

        let refs = LoadedScene::collect_material_texture_refs(&file, root);
        if !refs.is_empty() {
            status("Loading textures…");
            let compressed = parallel_map(&refs, |(abs, srgb)| {
                zestengine_assets::load_texture_bc(abs, *srgb).map_err(|e| format!("{e:#}"))
            })
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
            check(LoadStage::Textures(&compressed))?;
            status("Uploading textures…");
            let prep = up.prepare_compressed(&compressed).map_err(|e| format!("{e:#}"))?;
            material_textures = Some((refs, prep));
        }
    }
    // Material texture averages (GI emitter scale, bake albedo) are cached on
    // disk; computing them here keeps the decode off the main thread.
    let means = if renderable { LoadedScene::collect_material_mean_refs(&file, root) } else { Vec::new() };
    if !means.is_empty() {
        status("Analysing textures…");
        parallel_map(&means, |(abs, srgb)| zestengine_assets::texture_mean(abs, *srgb));
    }
    status("Loading skybox…");
    let skybox = if renderable { SkyboxData::read(file.skybox.as_deref(), file.environment.skybox_faces.as_ref(), root) } else { SkyboxData::None };
    status("Loading baked lighting…");
    let baked = if renderable { LoadedScene::read_bake_sidecars(&path.with_extension("")) } else { None };
    status("Compiling shaders…");
    let shaders = if renderable { crate::shaders::precompile_scene_shaders(&file, root) } else { Vec::new() };
    let audio_bake = std::fs::read(path.with_extension("audiobake")).ok();
    let occlusion = crate::occlusion::OcclusionData::read_sidecar(&path.with_extension(""));
    let reflection = crate::refl_slots::read_reflprobes(&path.with_extension("reflprobe"));
    Ok(LoadedSceneData {
        path: path.to_owned(),
        file,
        models,
        prepared,
        material_textures,
        skybox,
        baked,
        shaders,
        audio_bake,
        occlusion,
        reflection,
        net_side: zestengine_core::process_role(),
    })
}

/// Run `f` over `items` on a few low-priority threads, keeping cores free for
/// the UI and OS. Results come back in input order.
fn parallel_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let slots: Vec<std::sync::Mutex<Option<R>>> = items.iter().map(|_| std::sync::Mutex::new(None)).collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let threads = crate::sw_gi::bake_worker_count().min(items.len());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                crate::sw_gi::lower_compute_priority();
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= items.len() {
                        break;
                    }
                    *slots[i].lock().unwrap() = Some(f(&items[i]));
                }
            });
        }
    });
    slots.into_iter().map(|m| m.into_inner().unwrap().unwrap()).collect()
}

/// Skybox + baked lighting still to apply after [`install`].
pub struct PostLoad {
    pub skybox: Option<SkyboxData>,
    pub baked: Option<Option<BakedData>>,
    /// Baked reflection cubes read from the scene's sidecar (per probe).
    pub reflection: crate::refl_slots::ReflProbeSet,
}

/// Main-thread install: preloads the compiled shaders, then builds the scene
/// from the worker's uploads. Returns the scene and what's left to apply.
///
/// With no renderer (a dedicated server) the scene is built CPU-side only:
/// nothing is uploaded, compiled shaders are dropped, and skybox / baked
/// lighting have nothing to apply to (`PostLoad` is still returned so the
/// caller keeps one code path).
pub fn install(
    data: LoadedSceneData,
    mut renderer: Option<&mut Renderer>,
    registry: &ComponentRegistry,
    shaders: &mut ShaderLibrary,
    root: &Path,
    progress: impl FnMut(&str),
) -> anyhow::Result<(LoadedScene, PostLoad)> {
    if let Some(renderer) = renderer.as_deref_mut() {
        for (rel, compiled) in data.shaders {
            shaders.preload(renderer, root, &rel, compiled);
        }
    }
    // Models not placed by the scene file (carried-over objects) are installed
    // separately so they aren't uploaded twice.
    let placed: std::collections::HashSet<&str> = data
        .file
        .entries
        .iter()
        .filter_map(|e| match &e.source {
            zestengine_assets::ObjectSource::Model { path, .. } => Some(path.as_str()),
            _ => None,
        })
        .collect();
    let mut prepared = data.prepared;
    let mut extras = Vec::new();
    for (path, asset) in &data.models {
        if !placed.contains(path.as_str()) {
            extras.push((path.clone(), asset, prepared.as_mut().and_then(|p| p.remove(path))));
        }
    }
    let mut scene = LoadedScene::load_scene_file_with_models_for(
        data.net_side,
        renderer.as_deref_mut(),
        &data.file,
        root,
        registry,
        shaders,
        &data.models,
        prepared,
        data.material_textures,
        progress,
    )?;
    for (path, asset, prep) in extras {
        scene.install_model_template_with(renderer.as_deref_mut(), &path, asset, prep)?;
    }
    scene.audio_bake = data.audio_bake.map(std::sync::Arc::new);
    scene.occlusion = data.occlusion;
    Ok((
        scene,
        PostLoad {
            skybox: Some(data.skybox),
            baked: Some(data.baked),
            reflection: data.reflection,
        },
    ))
}

/// A load running on its own thread (runtime preloads / scene switches).
pub struct BackgroundLoad {
    pub path: String,
    rx: Receiver<Result<LoadedSceneData, String>>,
    result: Option<Result<LoadedSceneData, String>>,
}

impl BackgroundLoad {
    pub fn spawn(rel: &str, root: &Path, uploader: Option<zestengine_render::Uploader>, extra_models: Vec<String>) -> Self {
        let (tx, rx) = channel();
        let path = root.join(rel);
        let root = root.to_owned();
        std::thread::spawn(move || {
            let r = load_on_worker(&path, &root, uploader.as_ref(), &extra_models, &|_| {});
            let _ = tx.send(r);
        });
        Self {
            path: rel.to_owned(),
            rx,
            result: None,
        }
    }

    /// A load whose result is already known (user content decoded by the
    /// content manager's own worker), so it can go through `SceneFlow`.
    pub fn ready(path: impl Into<String>, result: Result<LoadedSceneData, String>) -> Self {
        let (_tx, rx) = channel();
        Self {
            path: path.into(),
            rx,
            result: Some(result),
        }
    }

    pub fn is_ready(&mut self) -> bool {
        if self.result.is_none()
            && let Ok(r) = self.rx.try_recv()
        {
            self.result = Some(r);
        }
        self.result.is_some()
    }

    /// Take the result, blocking if it isn't ready (callers check `is_ready`).
    pub fn wait(mut self) -> Result<LoadedSceneData, String> {
        match self.result.take() {
            Some(r) => r,
            None => self.rx.recv().map_err(|e| e.to_string())?,
        }
    }
}

/// Gameplay-driven scene flow shared by the runtime and editor Play mode:
/// background preloads, instant switches, objects carried across switches and
/// additive streaming.
#[derive(Default)]
pub struct SceneFlow {
    preloads: HashMap<String, BackgroundLoad>,
    additive: Vec<BackgroundLoad>,
    /// Requested switch, installed by `poll_switch` once its load finishes.
    switching: Option<BackgroundLoad>,
    /// Objects carried into the next scene on a switch.
    pub kept: std::collections::HashSet<zestengine_core::ObjectId>,
    /// Scenes currently streamed in additively.
    pub loaded_additive: Vec<String>,
}

impl SceneFlow {
    /// Kept objects that aren't under another kept object.
    fn kept_roots(&self, scene: &LoadedScene) -> Vec<usize> {
        let index = scene.id_index();
        let mut roots: Vec<usize> = self.kept.iter().filter_map(|id| index.get(id).copied()).collect();
        roots.retain(|&i| {
            let mut p = scene.objects[i].parent;
            while let Some(pi) = p {
                if self.kept.contains(&scene.objects[pi].id) {
                    return false;
                }
                p = scene.objects[pi].parent;
            }
            true
        });
        roots.sort_unstable();
        roots
    }

    fn kept_entries(&self, scene: &LoadedScene, root: &Path, shaders: &ShaderLibrary) -> Vec<zestengine_assets::SceneEntry> {
        let mut out: Vec<zestengine_assets::SceneEntry> = Vec::new();
        for i in self.kept_roots(scene) {
            if let Some(prefab) = scene.prefab_from_object(i, root, shaders) {
                let base = out.len();
                out.extend(prefab.entries.into_iter().map(|mut e| {
                    e.parent = e.parent.map(|p| p + base);
                    e
                }));
            }
        }
        out
    }

    fn kept_models(&self, scene: &LoadedScene, root: &Path, shaders: &ShaderLibrary) -> Vec<String> {
        let mut models = Vec::new();
        for e in self.kept_entries(scene, root, shaders) {
            if let zestengine_assets::ObjectSource::Model { path, .. } = e.source
                && !models.contains(&path)
            {
                models.push(path);
            }
        }
        models
    }

    /// Model paths of the objects that will be carried into the next scene
    /// (a custom loader imports them alongside the new scene).
    pub fn kept_model_paths(&self, scene: &LoadedScene, root: &Path, shaders: &ShaderLibrary) -> Vec<String> {
        self.kept_models(scene, root, shaders)
    }

    /// Switch to a scene that was already loaded elsewhere (a downloaded
    /// world package). Installed by [`Self::poll_switch`] like any switch.
    pub fn switch_to_loaded(&mut self, path: &str, data: LoadedSceneData) {
        self.switching = Some(BackgroundLoad::ready(path, Ok(data)));
    }

    /// Start loading `rel` in the background (no-op if already loading).
    pub fn preload(&mut self, rel: &str, root: &Path, renderer: Option<&Renderer>, scene: &LoadedScene, shaders: &ShaderLibrary) {
        if self.preloads.contains_key(rel) {
            return;
        }
        let extra = self.kept_models(scene, root, shaders);
        self.preloads
            .insert(rel.to_owned(), BackgroundLoad::spawn(rel, root, renderer.and_then(Renderer::uploader), extra));
    }

    /// Start replacing the scene with `rel`, reusing its preload when there is
    /// one. The current scene keeps running until [`Self::poll_switch`]
    /// installs it, so a transition (fall, corridor, fade) can cover the load.
    pub fn request_switch(&mut self, rel: &str, root: &Path, renderer: Option<&Renderer>, scene: &LoadedScene, shaders: &ShaderLibrary) {
        let load = match self.preloads.remove(rel) {
            Some(l) => l,
            None => BackgroundLoad::spawn(rel, root, renderer.and_then(Renderer::uploader), self.kept_models(scene, root, shaders)),
        };
        self.switching = Some(load);
    }

    /// Install a requested switch once its load finished, carrying kept
    /// objects over with their ids. Returns the new scene's path with it.
    pub fn poll_switch(
        &mut self,
        renderer: Option<&mut Renderer>,
        registry: &ComponentRegistry,
        shaders: &mut ShaderLibrary,
        root: &Path,
        old: &LoadedScene,
    ) -> Option<anyhow::Result<(String, LoadedScene, PostLoad)>> {
        if !self.switching.as_mut()?.is_ready() {
            return None;
        }
        let load = self.switching.take()?;
        let rel = load.path.clone();
        Some(self.install_switch(load, renderer, registry, shaders, root, old).map(|(s, p)| (rel, s, p)))
    }

    fn install_switch(
        &mut self,
        load: BackgroundLoad,
        mut renderer: Option<&mut Renderer>,
        registry: &ComponentRegistry,
        shaders: &mut ShaderLibrary,
        root: &Path,
        old: &LoadedScene,
    ) -> anyhow::Result<(LoadedScene, PostLoad)> {
        // Serialized now, not at request time, so kept objects arrive where
        // they were when the new scene appeared.
        let carried = self.kept_entries(old, root, shaders);
        let mut data = load.wait().map_err(|e| anyhow::anyhow!(e))?;
        // The new scene runs on the side the old one did (a dedicated server stays one).
        data.net_side = old.net_side;
        // Free the old scene's GPU resources (its objects were serialized above).
        if let Some(renderer) = renderer.as_deref_mut() {
            renderer.reset_scene()?;
        }
        let (mut scene, post) = install(data, renderer.as_deref_mut(), registry, shaders, root, |_| {})?;
        if !carried.is_empty() {
            scene.append_entries_with(renderer, shaders, root, registry, &carried)?;
        }
        self.additive.clear();
        self.loaded_additive.clear();
        Ok((scene, post))
    }

    /// Stream `rel` in next to the current scene (applied by `poll`).
    pub fn load_additive(&mut self, rel: &str, root: &Path, renderer: Option<&Renderer>) {
        if self.loaded_additive.iter().any(|r| r == rel) || self.additive.iter().any(|l| l.path == rel) {
            return;
        }
        let load = self
            .preloads
            .remove(rel)
            .unwrap_or_else(|| BackgroundLoad::spawn(rel, root, renderer.and_then(Renderer::uploader), Vec::new()));
        self.additive.push(load);
    }

    /// Install finished additive loads. Returns true when objects were added.
    pub fn poll(
        &mut self,
        mut renderer: Option<&mut Renderer>,
        registry: &ComponentRegistry,
        shaders: &mut ShaderLibrary,
        root: &Path,
        scene: &mut LoadedScene,
    ) -> bool {
        let mut changed = false;
        let mut i = 0;
        while i < self.additive.len() {
            if !self.additive[i].is_ready() {
                i += 1;
                continue;
            }
            let load = self.additive.remove(i);
            let rel = load.path.clone();
            match load.wait() {
                Ok(data) => match scene.append_loaded(renderer.as_deref_mut(), shaders, root, registry, data, &rel) {
                    Ok(_) => {
                        self.loaded_additive.push(rel);
                        changed = true;
                    }
                    Err(e) => tracing::error!("additive load {rel}: {e:#}"),
                },
                Err(e) => tracing::error!("additive load {rel}: {e}"),
            }
        }
        changed
    }

    /// Remove an additively loaded scene's objects. True when anything went.
    pub fn unload(&mut self, rel: &str, scene: &mut LoadedScene) -> bool {
        self.loaded_additive.retain(|r| r != rel);
        self.additive.retain(|l| l.path != rel);
        scene.unload_origin(rel) > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_loads_scene_without_gpu() {
        let dir = std::env::temp_dir().join(format!("zestengine-loader-test-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("scenes")).unwrap();
        let mut file = zestengine_assets::SceneFile::default();
        file.entries.push(zestengine_assets::SceneEntry {
            id: zestengine_core::ObjectId::new().to_string(),
            name: "Empty".into(),
            source: zestengine_assets::ObjectSource::Empty,
            enabled: true,
            static_geometry: false,
            lightmap_scale: 1.0,
            cast_shadows: zestengine_core::ShadowCasting::On,
            layer: 0,
            material: zestengine_assets::MaterialRef::File(String::new()),
            extra_materials: vec![],
            renderer_split: false,
            parent: None,
            components: vec![],
            translation: [1.0, 2.0, 3.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
            model: None,
            model_part: None,
            prefab: None,
            prefab_parent: None,
            net_role: Default::default(),
            component_roles: Default::default(),
        });
        let path = dir.join("scenes/a.scene");
        zestengine_assets::save_scene_file(&path, &file).unwrap();
        let statuses = std::sync::Mutex::new(Vec::new());
        let data = load_on_worker(&path, &dir, None, &[], &|s| statuses.lock().unwrap().push(s.to_owned())).unwrap();
        assert_eq!(data.file.entries.len(), 1);
        assert!(data.baked.is_none() && matches!(data.skybox, SkyboxData::None) && data.shaders.is_empty());
        assert!(statuses.lock().unwrap().len() >= 3, "reports progress");
        let mut bg = BackgroundLoad::spawn("scenes/a.scene", &dir, None, Vec::new());
        let start = std::time::Instant::now();
        while !bg.is_ready() && start.elapsed().as_secs() < 5 {
            std::thread::yield_now();
        }
        assert_eq!(bg.wait().unwrap().file.entries[0].translation, [1.0, 2.0, 3.0]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
