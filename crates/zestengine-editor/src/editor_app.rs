//! Editor application: the dockable egui shell (menu bar, Inspector, Scene
//! tree, Files, gizmos, picking) wrapped around the runtime. Compiled only
//! with the `editor` feature; a shipped game never links it.

use zestengine_engine::audio;
use zestengine_engine::bundle;
use crate::crash;
use crate::gizmo;
use crate::icon;
use zestengine_engine::log_capture;
use crate::lsp;
use zestengine_engine::plugins;
use zestengine_engine::scene;

#[path = "audio_bake.rs"]
mod audio_bake;
#[path = "audio_assets.rs"]
mod audio_assets;
#[path = "occlusion_bake.rs"]
mod occlusion_bake;
#[path = "prefab_editor.rs"]
mod prefab_editor;
#[path = "placement_glue.rs"]
mod placement_glue;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use egui_dock::{DockArea, DockState, NodeIndex};
use glam::{Mat4, Vec3};
use winit::application::ApplicationHandler;
use winit::event::{
    DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent,
};
use crate::egui_phosphor::regular as ph;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::{CursorGrabMode, Window, WindowId};

use crate::camera::FlyCamera;
use zestengine_core::{ComponentCommand, ComponentRegistry, ObjectId};
use crate::{
    CodeDiagnostic, CodeEditor, EditorComponents, FileBrowser, InspectorContent, InspectorPanel,
    MaterialModel, ObjectInfoModel, ScenePanel, ShaderUiInfo, TransformModel,
};
use zestengine_render::{CameraData, FrameInput, Renderer};
use crate::gizmo::{GizmoState, GizmoTool};
use zestengine_engine::scene::{LoadedScene, material_from_model, model_from_material, relative_to};
use zestengine_engine::shaders::ShaderLibrary;
use crate::undo::{ObjectState, SkyboxState, UndoEntry, UndoStack};

/// Startup steps the splash's progress bar counts: Vulkan, plugins, project,
/// scene parse, scene install, open.
const LOAD_STEPS: usize = 6;

/// egui-memory key holding the `LayerId` that hosts the viewport / camera view
/// (see `pointer_over_overlay`).
const VIEW_LAYER_KEY: &str = "zestengine-view-layer";

/// Frames rendered since start, used by the `ZESTENGINE_SHOT` dump to skip the
/// scene-loading frames.
static SHOT_FRAMES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Version, profile and the running binary's own link time ("0.1.0 · debug ·
/// built 2026-11-04 11:40:33 UTC"): shown in the About screen, the status bar,
/// the window title and the VR watch so the running build can be verified.
pub fn build_stamp() -> String {
    let profile = if cfg!(debug_assertions) { "debug" } else { "release" };
    let built = std::env::current_exe()
        .ok()
        .and_then(|p| p.metadata().ok())
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .and_then(|d| chrono::DateTime::from_timestamp(d.as_secs() as i64, 0))
        .map(|t| t.format("%Y-%m-%d %H:%M:%S UTC").to_string())
        .unwrap_or_else(|| "built at an unknown time".into());
    format!("{} · {profile} · built {built}", env!("CARGO_PKG_VERSION"))
}

pub struct AppConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
    /// Optional .scene / model file to open; falls back to the test scene.
    pub scene_path: Option<String>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            title: "zestengine editor".into(),
            width: 1600,
            height: 900,
            scene_path: None,
        }
    }
}

pub fn run(config: AppConfig) -> Result<()> {
    crash::install();
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let project_root = std::env::current_dir()?;
    let mut app = EngineApp {
        config,
        window: None,
        renderer: None,
        xr: None,
        xr_session: None,
        vr_targets: zestengine_core::TrackerTargets::default(),
        face_input: zestengine_engine::face_input::FaceInput::default(),
        avatar_preview: Default::default(),
        avatar_menu: Default::default(),
        avatar_menu_vr: Default::default(),
        avatar_osc: Default::default(),
        slimevr: None,
        body: zestengine_engine::body::BodyTracking::load(),
        vr_rig: zestengine_core::VrRig::default(),
        xr_stereo: None,
        vr_grab_anchor: None,
        vr_drag_hand: None,
        vr_panel_anchor: None,
        vr_prev_hand_dist: None,
        vr_prev_trigger: false,
        vr_prev_grip: false,
        vr_move_dist: 0.0,
        vr_menu_open: false,
        vr_wheel_open: false,
        vr_prev_menu: false,
        vr_ui_pointer: None,
        vr_ui_pressed: false,
        vr_ui_prev_pressed: false,
        vr_overlay_draw: None,
        vr_hand_joints: [None, None],
        scene: LoadedScene::empty(),
        egui_ctx: egui::Context::default(),
        egui_state: None,
        tasks: crate::tasks::TaskManager::default(),
        cluster_task: Default::default(),
        show_task_popup: false,
        bake_job: None,
        bake_pending_warmup: false,
        deferred_model_applies: Vec::new(),
        placement: Default::default(),
        splash: None,
        pending_init: false,
        splash_painted: false,
        renderer_rx: None,
        plugin_rx: None,
        plugin_status: Default::default(),
        ready_renderer: None,
        pending_window: None,
        load_status: std::sync::Arc::new(std::sync::Mutex::new(String::new())),
        profiler_window: None,
        profiler_egui_ctx: egui::Context::default(),
        profiler_egui_state: None,
        prof_history: ProfHistory::default(),
        last_render_stats: zestengine_render::RenderStats::default(),
        dock_state: default_layout(),
        inspector: InspectorPanel::new(),
        inspector_lock_target: None,
        scene_panel: ScenePanel::new(),
        prefab_editor: Default::default(),
        file_browser: FileBrowser::new(project_root.clone()),
        selection: Selection::None,
        multi_objects: Vec::new(),
        multi_edit: MultiEdit::default(),
        multi_files: Vec::new(),
        refl_bake_pending: None,
        file_material: None,
        file_meta: None,
        texture_ui: Default::default(),
        rig_ui: Default::default(),
        asset_preview: Default::default(),
        avatar_figure: Default::default(),
        sprite_atlas_ui: Default::default(),
        model_ui: Default::default(),
        slot_validator: Default::default(),
        import_prompt: None,
        open_editors: vec![],
        nvim_embed: crate::nvim_embed::NvimEmbed::new(project_root.clone()),
        lsp: None,
        lsp_failed: false,
        lsp_requests: HashMap::new(),
        file_diagnostics: HashMap::new(),
        gizmo: GizmoState::new(),
        widget_filter: WidgetFilter::default(),
        gizmo_drag: false,
        orbit_armed: false,
        camera_tab_visible: false,
        viewport_visible: true,
        vp_strip_width: 0.0,
        audio: audio::AudioEngine::new(),
        actions: Vec::new(),
        undo_stack: UndoStack::default(),
        suppress_undo_record: false,
        edit_session: None,
        edit_window: false,
        edit_form: EditForm::default(),
        edit_peers: Default::default(),
        edit_presence_next: None,
        edit_transform_next: None,
        edit_browser: None,
        edit_browser_next: None,
        edit_voice_seq: 0,
        components: ComponentRegistry::with_builtins(),
        editor_components: EditorComponents::with_builtins(),
        playing: false,
        play_paused: false,
        play_step: false,
        command_palette: Default::default(),
        viewport_stats: false,
        active_tabs: Default::default(),
        play_time: 0.0,
        play_scene_switched: false,
        play_origin_scene: None,
        play_snapshot: None,
        play_session: None,
        play_game_camera: false,
        focus_game_tab: false,
        focus_scene_tab: false,
        occlusion_ui: OcclusionUi::default(),
        camera_rect: egui::Rect::NOTHING,
        post_load: None,
        flow: zestengine_engine::scene_loader::SceneFlow::default(),
        content: zestengine_engine::content::ContentManager::new(project_root.join(".zestengine_content")),
        content_builder: Default::default(),
        game_ui: zestengine_engine::game_ui::GameUi::default(),
        ui_editor: Default::default(),
        anim_ctrl_tab: Default::default(),
        anim_clip_tab: Default::default(),
        sequencer_tab: Default::default(),
        shader_graph_tab: Default::default(),
        ui_rect_editor: Default::default(),
        game_ui_frame: zestengine_engine::game_ui::UiFrame::default(),
        world_ui: zestengine_engine::game_ui::WorldUi::default(),
        particles: zestengine_engine::particles::ParticleRenderer::default(),
        water: zestengine_engine::water_render::WaterRenderer::default(),
        water_fx: zestengine_engine::water_fx::WaterFx::default(),
        carry_fx: zestengine_engine::water_carry::CarryFx::default(),
        editor_water: crate::editor_water::EditorWater::default(),
        nav_preview: Vec::new(),
        world_ui_draws: Vec::new(),
        world_ui_prev: (false, false),
        game_cursor_wanted: false,
        game_cursor_locked: false,
        shaders: ShaderLibrary::default(),
        shader_files: Vec::new(),
        last_shader_scan: None,
        shader_scan_rx: None,
        engine_shader_mtime: None,
        last_asset_check: None,
        dirty_materials: HashSet::new(),
        last_material_edit: None,
        plugins: plugins::PluginHost::default(),
        plugin_build_error: None,
        status: None,
        reload_pending: false,
        pending_job: None,
        plugin_gate: Default::default(),
        plugin_fingerprint: None,
        plugin_build_started: None,
        pacing: zestengine_render::frame_limiter::FramePacing::new(),
        project: zestengine_assets::ProjectFile::default(),
        camera: FlyCamera::default(),
        orbit_pivot: None,
        looking: false,
        look_just_ended: false,
        panning: false,
        look_delta: (0.0, 0.0),
        keys: HashSet::new(),
        modifiers: winit::keyboard::ModifiersState::empty(),
        input: zestengine_engine::input_engine::InputManager::default(),
        net: None,
        voice: None,
        net_addr: "127.0.0.1:9000".to_string(),
        local_server: None,
        net_ui: crate::net_window::NetWindowState::default(),
        perf_ui: crate::perf_window::PerfWindowState::default(),
        show_performance: false,
        skinning: Default::default(),
        show_bindings: false,
        show_network: false,
        show_layers: false,
        show_pose_editor: false,
        show_mixer: false,
        mixer_changed: false,
        bindings_editor: crate::BindingsEditor::default(),
        last_cursor: None,
        viewport_rect: egui::Rect::EVERYTHING,
        project_root,
        current_scene_path: None,
        pending_lm_uv_models: None,
        pack_import: crate::pack_import_ui::PackImportUi::default(),
        scene_name_input: "scenes/world.scene".into(),
        show_stats: true,
        show_stats_overlay: false,
        show_help: false,
        about_open: false,
        project_settings_dirty: false,
        show_new_project: false,
        show_open_project: false,
        pending_project_switch: None,
        open_project_path: String::new(),
        lightmap_preview: false,
        scene_dirty: false,
        show_quit_dialog: false,
        quit_after_frame: false,
        relaunch_after_quit: false,
        editor_prefs: crate::editor_prefs::EditorPrefs::default_path()
            .map(|p| crate::editor_prefs::EditorPrefs::load(&p))
            .unwrap_or_default(),
        new_project_parent: String::new(),
        new_project_name: "my-game".into(),
        new_project_template: zestengine_engine::templates::DEFAULT_TEMPLATE.into(),
        new_project_filter: None,
        build_status: None,
        building_game: false,
        build_target: bundle::BuildTarget::Host,
        log_filter: LogFilter::default(),
        handle_drag: None,
        stats: Stats::default(),
        world: hecs::World::new(),
        start: Instant::now(),
        last_frame: Instant::now(),
        rt_gi: zestengine_engine::realtime_gi::RealtimeGiState::default(),
        refl_slots: zestengine_engine::refl_slots::ReflectionSlots::default(),
        refl_bake_queue: Vec::new(),
        refl_bake_current: None,
        refl_unsaved: Vec::new(),
        refl_applied_seq: 0,
        bake_serial: 0,
        bake_sizes: None,
        zestsurface_object: None,
        frame_timings: FrameTimings::default(),
        gi_debug: std::env::var("ZESTENGINE_VIEW_MODE").ok().and_then(|v| v.parse().ok()).unwrap_or(0),
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}

#[derive(Clone, PartialEq)]
enum Selection {
    None,
    Object(usize),
    File(PathBuf),
}

/// One action on the VR left-hand menu. The desktop editor keeps its full egui
/// UI; this is the VR-only quick menu pointed at with the right controller.
#[derive(Clone, Copy)]
enum VrAction {
    BodyMenu,
    CalibrateTpose,
    ClearCalibration,
    ToggleFootIk,
    ResetRig,
    DeleteSelected,
}

/// VR tool actions, surfaced as a "VR Tools" egui window so they're reachable
/// from the in-VR UI panel (and the desktop). The full editor UI is otherwise
/// shown as-is on the panel.
const VR_TOOLS: &[(&str, VrAction)] = &[
    ("Body Calibration", VrAction::BodyMenu),
    ("Calibrate T-pose", VrAction::CalibrateTpose),
    ("Clear Calibration", VrAction::ClearCalibration),
    ("Toggle Foot IK", VrAction::ToggleFootIk),
    ("Reset View/Scale", VrAction::ResetRig),
    ("Delete Selected", VrAction::DeleteSelected),
];

/// Pre-frame snapshot of the selection's editable state (undo diffing).
enum EditSnapshot {
    None,
    Object {
        index: usize,
        state: ObjectState,
        /// The materials of the open material sections, as they were before the frame.
        materials: Vec<(usize, Box<MaterialModel>)>,
    },
    File {
        path: PathBuf,
        model: Box<MaterialModel>,
    },
}

/// Plain-text file open in the rudimentary code editor.
struct OpenEditor {
    path: PathBuf,
    text: String,
    dirty: bool,
    language: String,
    diagnostics: Vec<CodeDiagnostic>,
    last_edit: Option<Instant>,
    /// Text changed since the last LSP `didChange` was sent.
    lsp_dirty: bool,
    /// Active completion popup + hover tooltip (LSP-fed).
    completion: Option<crate::CompletionState>,
    hover: Option<crate::HoverState>,
    /// Pending go-to-definition jump (0-based line, utf-8 column).
    goto: Option<(u32, u32)>,
    /// Reference list to show in the `gr` picker popup.
    references: Option<Vec<crate::ReferenceItem>>,
    /// Code actions (Ctrl+.) to show in the quick-fix popup.
    actions: Vec<crate::CodeActionItem>,
    /// The file's mtime when last written by us; a change on disk means an
    /// external editor (nvim) saved, and a clean editor reloads.
    disk_mtime: Option<std::time::SystemTime>,
}

/// An in-flight LSP request, keyed by request id, awaiting its response.
#[derive(Clone)]
enum LspRequestKind {
    Completion { path: PathBuf, anchor_char: usize, explicit: bool },
    Hover { path: PathBuf },
    Definition,
    References { path: PathBuf },
    CodeAction { path: PathBuf },
}

/// Extensions the code editor opens (scene/material files have dedicated
/// inspectors).
const CODE_EXTENSIONS: &[&str] = &[
    "rs", "frag", "hlsl", "vert", "glsl", "slang", "toml", "ron", "md", "txt", "json", "zestengine",
];

/// Scratch state for a `.material` file opened in the Inspector.
struct FileMaterial {
    path: PathBuf,
    file: zestengine_assets::MaterialFile,
    model: MaterialModel,
    dirty: bool,
}

/// A model file's `.meta` open in the Inspector for editing import options.
struct FileMeta {
    path: PathBuf,
    /// The edited copy the inspector writes to.
    meta: zestengine_assets::AssetMeta,
    /// The importer settings as last saved, serialised: "dirty" means the edited
    /// copy differs from this, so editing a value back makes the session clean.
    saved: String,
    dirty: bool,
}

/// The "Unapplied import settings" prompt: the Model Import inspector has edits
/// that were neither applied nor reverted, and something is about to leave it.
struct ImportPrompt {
    path: PathBuf,
    /// What the user was doing, redone once they pick Apply or Revert.
    resume: Option<EditorAction>,
    /// The selection moved away before the prompt came up: Cancel puts it back.
    restore_selection: bool,
    /// Closing the window: Apply / Revert go on to quit.
    quit: bool,
}

/// Co-edit session host/join form state.
#[derive(Default)]
struct EditForm {
    name: String,
    session_name: String,
    port: u16,
    secret: String,
    join_addr: String,
}

impl FileMeta {
    fn new(path: PathBuf, meta: zestengine_assets::AssetMeta) -> Self {
        let saved = Self::serialise(&meta);
        Self { path, meta, saved, dirty: false }
    }

    fn serialise(meta: &zestengine_assets::AssetMeta) -> String {
        ron::ser::to_string(&meta.importer).unwrap_or_default()
    }

    /// Recompute `dirty` against the saved settings (call after an edit).
    fn refresh_dirty(&mut self) {
        self.dirty = Self::serialise(&self.meta) != self.saved;
    }

    /// Write the edited settings and make them the saved baseline.
    fn apply(&mut self) -> anyhow::Result<()> {
        zestengine_assets::save_asset_meta(&self.path, &self.meta)?;
        self.saved = Self::serialise(&self.meta);
        self.dirty = false;
        Ok(())
    }

    /// Throw the edits away: reload the settings from disk.
    fn revert(&mut self) {
        match zestengine_assets::load_asset_meta(&self.path) {
            Ok(Some(meta)) => self.meta = meta,
            Ok(None) => {}
            Err(e) => tracing::error!("reloading .meta: {e:#}"),
        }
        self.saved = Self::serialise(&self.meta);
        self.dirty = false;
    }
}

/// A heavy, UI-blocking operation deferred one frame so a "busy" overlay paints
/// before it runs (otherwise the window looks frozen).
enum PendingJob {
    /// Plugin builds run on a worker now (`request_plugin_build`); kept for the overlay plumbing.
    #[allow(dead_code)]
    ReloadPlugins,
}

/// Work parked until the component-plugin build finishes (it needs the plugin's
/// components registered).
enum PluginWaiter {
    /// A worker-loaded scene, installed once the components exist.
    Scene(Box<zestengine_engine::scene_loader::LoadedSceneData>),
    /// Enter Play mode.
    Play,
}

/// Which pass of the chunked bake is running.
#[derive(PartialEq)]
enum BakeRunPhase {
    /// Main lightmaps + probes.
    Main,
    /// FluxVoxel voxel volumes (appended to the probe set).
    FluxVoxel,
}

/// Drives a chunked lighting bake across frames (one unit per frame). Holds the
/// GPU `BakeJob` plus the metadata needed to assemble `BakedData` when it
/// completes, and carries the partial result from the main pass into the FluxVoxel
/// pass. See the bake phase of the background-task plan.
struct BakeRunner {
    job: zestengine_render::BakeJob,
    task_id: crate::tasks::TaskId,
    cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    progress: std::sync::Arc<std::sync::Mutex<crate::tasks::TaskProgress>>,
    phase: BakeRunPhase,
    settings: zestengine_assets::BakeSettings,
    /// Main pass: object→lightmap-layer map and probe volumes for assembly.
    object_lightmap: std::collections::HashMap<usize, usize>,
    probe_volumes: Vec<scene::ProbeVolumeMeta>,
    /// Partial bake carried from the main pass into the FluxVoxel pass.
    baked: Option<scene::BakedData>,
}

#[derive(Default)]
struct Stats {
    frames: u64,
    fps: f32,
    frame_ms: f32,
    accum: f32,
    accum_frames: u32,
    /// Raw frame-to-frame times (worst frame, per-second histogram).
    log: zestengine_engine::runtime::FrameTimeLog,
}

impl Stats {
    fn tick(&mut self, dt: f32) {
        self.frames += 1;
        self.accum += dt;
        self.accum_frames += 1;
        if self.accum >= 0.25 {
            self.fps = self.accum_frames as f32 / self.accum;
            self.frame_ms = self.accum * 1000.0 / self.accum_frames as f32;
            self.accum = 0.0;
            self.accum_frames = 0;
        }
    }
}

/// Deferred editor operations (UI runs without renderer access; these are
/// processed afterwards).
enum EditorAction {
    /// Scene-view camera: orthographic (true) or perspective (false).
    SetOrtho(bool),
    SelectFile(PathBuf),
    /// Select a scene object by index (the Model section's child list).
    SelectObject(usize),
    /// The Environment tab changed the settings this frame (from `before`):
    /// record an undo step (`gesture`: a drag still in progress).
    EnvironmentEdited { before: Box<zestengine_assets::WorldEnvironment>, gesture: bool },
    /// Re-render the active Reflection Probe's cubemap from the current scene
    /// (realtime probe refresh — surroundings changed since the last capture).
    RecaptureReflections,
    /// Recapture then read the cube back and persist it next to the scene as a
    /// `.reflprobe` sidecar (baked reflection probe — loaded on scene open).
    BakeReflections,
    ImportModel(PathBuf),
    /// A file dropped on the Scene hierarchy (path, row it landed on): spawn at world origin.
    PlacementDrop(PathBuf, Option<usize>),
    EditInZestSurface(usize),
    CreateMaterial(PathBuf),
    CreateScene(PathBuf),
    CreateShader(PathBuf),
    CreatePostFx(PathBuf),
    CreateAudioAsset(PathBuf, bool),
    CreateWaterWaves(PathBuf),
    CreatePhysMat(PathBuf),
    CreateSpriteShape(PathBuf),
    CreateSpriteAtlas(PathBuf),
    CreateFolder(PathBuf),
    PickAt(egui::Pos2),
    AssignMaterialAt(egui::Pos2, PathBuf),
    /// (object index, material slot, `.material` path)
    AssignMaterialToObject(usize, usize, PathBuf),
    /// Give render slot `.1` of object `.0` a new value (picker, drop, reset to the model's).
    AssignSlot(usize, usize, zestengine_core::MaterialSlot),
    /// Copy the material of a slot to a new `.material` and assign the copy to that slot only.
    MakeUniqueMaterial(usize, usize),
    MaterialEdited(usize),
    ResetMaterial(usize),
    /// An image was dropped on a scene material's texture slot: (material index,
    /// slot 0..12, absolute image path). Converted to project-relative + applied.
    AssignTexture {
        material: usize,
        slot: usize,
        path: PathBuf,
    },
    /// An image was dropped on a texture slot of the open `.material` file
    /// editor: (slot 0..12, absolute image path).
    AssignFileTexture {
        slot: usize,
        path: PathBuf,
    },
    /// Extract an embedded (imported) material to a `.material` file so it can be
    /// edited + saved.
    ExtractMaterial(usize),
    SaveFileMaterial,
    /// Discard the open model's unapplied import edits (reload from disk).
    RevertFileMeta,
    /// Save the open texture's `.meta` (when it has unapplied edits) and rebuild
    /// its GPU copies and the materials that use it.
    ReimportTexture(PathBuf),
    /// Sampler fields of an open texture's import settings changed: rebind its materials now, no reimport.
    TextureSampler(PathBuf, zestengine_assets::SamplerOptions),
    /// A material texture slot's import-settings warning was acted on (Auto-fix,
    /// duplicate, ignore). `material` is the scene material index, `None` for the
    /// open `.material` file.
    FixTextureSlot {
        material: Option<usize>,
        slot: usize,
        fix: crate::SlotFix,
    },
    /// Save the model's `.meta` and reload the scene so it reimports with the new
    /// settings.
    ReimportModel(PathBuf),
    /// Extract Textures / Extract Materials / Search and Remap on a model's import
    /// session: files are written (or found) on a worker and the model's remap
    /// tables in the inspector's edited settings are filled (Apply saves them).
    ModelRemap(PathBuf, crate::model_import_editor::RemapJob),
    /// Rebuild a model's cluster geometry cache on a task worker (the model
    /// inspector's Geometry (clusters) section owns the job's progress).
    GeoRebuild(PathBuf, std::sync::Arc<crate::geometry_cache_ui::GeoJob>),
    /// A `.material` file's model changed in the Inspector: propagate to
    /// every scene material loaded from that file.
    FileMaterialEdited(PathBuf),
    /// Replace the scene with an empty one.
    NewScene,
    /// Open a code/text file in a dockable editor tab (focus if already open).
    OpenCodeFile(PathBuf),
    /// Save the open editor for this path.
    SaveOpenEditor(PathBuf),
    /// Close a code tab by path (vim `:q`/`:wq`).
    CloseCodeTab(PathBuf),
    /// Request LSP completion / hover / definition at a cursor char index.
    LspCompletion(PathBuf, usize, bool),
    LspHover(PathBuf, usize),
    LspGoto(PathBuf, usize),
    LspReferences(PathBuf, usize),
    LspCodeAction(PathBuf, usize),
    /// Open a file (if needed) and jump to a 0-based line/col (reference pick).
    OpenAndGoto(PathBuf, u32, u32),
    /// New Rust component module in the plugin crate.
    CreateComponent,
    LoadScene(PathBuf),
    SaveScene(Option<PathBuf>),
    /// Right mouse pressed over the viewport: enter mouse-look.
    StartLook,
    /// Scroll over the viewport while the Scene view shows the game camera: the
    /// wheel belongs to the game. (The fly camera's own dolly, like its orbit,
    /// is applied inside the egui pass by `viewport_ui`, not as an action, so the
    /// overlays drawn there share the camera with this frame's render.)
    Dolly(f32),
    /// F: frame the selected object.
    FocusSelected,
    Undo,
    Redo,
    /// Create an empty/camera/primitive in the scene.
    Spawn(zestengine_assets::ObjectSource),
    SpawnLight(zestengine_core::LightKind),
    SpawnAreaLight,
    SpawnProbeVolume,
    /// Post FX Volume; true = global, false = local box.
    SpawnPostFxVolume(bool),
    SpawnReflectionProbe,
    /// Spawn a plane in front of the view showing render-texture camera `usize`.
    CreateCameraScreen(usize),
    SpawnFluxVolume,
    SpawnAudioSource,
    SpawnAudioVolume,
    SpawnBoxCollider,
    SpawnSphereCollider,
    SpawnCapsuleCollider,
    SpawnWater,
    SpawnFloatingCube,
    SpawnBucket,
    SpawnWaterfall,
    SpawnWhirlpool,
    /// A global Wind Zone (waves, cloth, steam) ahead of the camera.
    SpawnWindZone,
    /// A flat heightfield landscape (Terrain component).
    SpawnTerrain,
    /// Add a Mesh Collider to every selected object that draws a mesh.
    AddMeshCollider,
    /// (child, new parent, before-sibling) reorder/move.
    MoveObject(usize, Option<usize>, Option<usize>),
    DeleteObject(usize),
    DuplicateObject(usize),
    /// Rename an object (F2 in the hierarchy); dispatched so co-edit can
    /// broadcast the rename alongside the local apply.
    RenameObject(usize, String),
    /// Create a ready-made UI object (under the selected / first canvas).
    SpawnUi(crate::UiPreset),
    /// A scene-view hide / lock change from the hierarchy or the `H` key.
    SceneView(crate::ViewRequest),
    /// Undo record for Active flags changed in the Inspector: (object, before, after).
    RecordEnabled(Vec<(zestengine_core::ObjectId, bool, bool)>),
    /// Bake Steam Audio reverb + pathing for the scene's Audio Probe Volumes.
    BakeAudio,
    /// Bake / clear the scene's occlusion-culling data.
    BakeOcclusion,
    ClearOcclusion,
    /// Point the editor camera along a direction, keeping its focus point.
    SnapView(Vec3),
    /// Build a ready-made UI layout (Main Menu, HUD, ...) as a new canvas.
    SpawnUiTemplate(zestengine_engine::ui_templates::UiTemplate),
    /// Switch a canvas between 2D (screen) and 3D (world panel).
    SetCanvasMode(usize, zestengine_core::UiRenderMode),
    /// Place a `.prefab` in the scene: at the point under the cursor (viewport
    /// drop) or in front of the camera.
    InstantiatePrefab(PathBuf, Option<egui::Pos2>),
    /// A prefab command: Hierarchy menu, Inspector / Overrides requests, Prefab Mode.
    Prefab(prefab_editor::PrefabAction),
    DuplicateFile(PathBuf),
    /// Files menu: make a variant of this `.prefab`.
    CreatePrefabVariant(PathBuf),
    /// Files menu: import an Aseprite / PSD file next to itself (2D-7).
    ImportLayered(PathBuf),
    ShowInFileManager(PathBuf),
    /// "Create Prefab / Instance from Layers" on a PSD's `.layers` file; true = also place an instance.
    LayersPrefab(PathBuf, bool),
    DeleteFile(PathBuf),
    SetSkybox(PathBuf),
    /// Convert Unity materials (Files panel context menu); see `unity_convert_ui`.
    UnityConvert(crate::unity_convert_ui::UnityJob),
    /// Open the Megascans / Fab import dialog (Files panel); `Some` scans that source at once.
    ImportPack(Option<PathBuf>),
    ClearSkybox,
    /// Assign one face of the 6-image cubemap skybox (index 0..6 = +X,-X,+Y,-Y,+Z,-Z).
    SetSkyboxFace(usize, PathBuf),
    /// Clear the cubemap faces (fall back to the equirect / procedural sky).
    ClearSkyboxFaces,
    /// Re-parent an object (None = unparent).
    SetParent(usize, Option<usize>),
    /// Hierarchy: `child` dropped on a bone row of `owner`'s model. Parents it to `owner`
    /// and adds a Bone Attach keeping its world transform.
    AttachToBone { child: usize, owner: usize, bone: String },
    /// Put these objects under a new Empty at their centre.
    GroupIntoEmpty(Vec<usize>),
    /// Hierarchy "Add Toggle": a menu toggle on this object, under its avatar.
    AddToggle(usize),
    /// Enter / leave Play mode (components run while playing).
    TogglePlay,
    /// Create the starter plugin crate under plugins/.
    CreatePluginCrate,
    SaveEditorPrefs,
    OpenInExternalEditor(PathBuf),
    /// Rebuild + reload all component plugins.
    ReloadPlugins,
    /// Run the GPU lighting bake (lightmaps + probes).
    BakeLighting,
    /// Discard the baked lighting result.
    ClearBake,
    /// Re-encode the scene's uncompressed bake sidecars as BC6H.
    RecompressBake,
    /// Re-run lightmap-UV generation for every model with a `.lmuv` marker (picks
    /// up the current unwrapper) by reloading the scene.
    RegenerateLightmapUvs,
    /// Generate lightmap UVs for ALL static models that lack a usable one (mark +
    /// reload), so the whole scene becomes bakeable in one click.
    GenerateAllLightmapUvs,
    /// Enable (true) or disable (false) auto-generated lightmap UVs for one model
    /// — writes/removes its `.lmuv` marker, then reloads. `true` generates UVs for
    /// the model's meshes that need them; `false` reverts to the model's own uv0.
    SetLightmapUvGen(String, bool),
    /// Scaffold a new project under `parent/<name>` and switch to it.
    NewProject { parent: PathBuf, name: String, template: String },
    /// Open (switch to) an existing project folder.
    OpenProject(PathBuf),
    /// Start the Project Hub executable.
    OpenProjectHub,
    /// Compile the current project into a standalone executable for a target.
    BuildGame(bundle::BuildTarget),
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
#[allow(dead_code)]
enum Tab {
    Viewport,
    /// Live view from the scene's main camera (smallest camera id).
    Camera,
    Scene,
    Inspector,
    /// World lighting / skybox setup.
    Environment,
    /// Lighting-bake settings + Bake / Clear ("FluxBaker").
    Baker,
    Files,
    /// Filterable console mirroring all tracing logs.
    Log,
    /// Visual editor for UI canvases (layout, templates).
    UiEditor,
    /// Animator Controller state-machine graph editor.
    AnimController,
    /// Animation clip (keyframe timeline) editor.
    AnimClip,
    /// Cinematics sequencer (timeline of scene-anchored transform/camera tracks).
    Sequencer,
    /// Visual shader graph (node-based surface shader) editor.
    ShaderGraph,
    /// Occlusion culling: bake settings, Bake / Clear, visualization.
    Occlusion,
    /// Project-wide settings (name, VR, boot scene, input, build target...).
    ProjectSettings,
    /// Independent code/shader editor tab (can be opened many times, dragged, closed).
    Code(PathBuf),
    /// Embedded Neovim (`nvim --embed`) terminal tab.
    Nvim,
}

fn default_layout() -> DockState<Tab> {
    // Camera shares the Viewport node (tab next to it); Environment shares the
    // Inspector node.
    // Project Settings docks into the main (central) area by default, next to
    // the Scene view, so its long form scrolls in a dock instead of overflowing
    // a floating window on small screens.
    let mut state = DockState::new(vec![Tab::Viewport, Tab::Camera, Tab::UiEditor, Tab::ProjectSettings]);
    let tree = state.main_surface_mut();
    // Right (Inspector) node a touch wider (~30px at the 1600px default)
    // so the inspector's rows fit without clipping.
    // At the 1600x960 reference: the inspector (324) runs full height on the
    // right; the Log and Files share the bottom dock under the work area, and
    // the Scene tree (264) sits left of the viewport above it.
    let [viewport, _right] = tree.split_right(
        NodeIndex::root(),
        1.0 - crate::theme::W_INSPECTOR / 1600.0,
        vec![Tab::Inspector, Tab::Environment, Tab::Baker, Tab::Occlusion],
    );
    let [viewport, _bottom] =
        tree.split_below(viewport, 1.0 - crate::theme::H_BOTTOM_DOCK / 894.0, vec![Tab::Log, Tab::Files]);
    let [_viewport, _scene] =
        tree.split_left(viewport, crate::theme::W_HIERARCHY / (1600.0 - crate::theme::W_INSPECTOR), vec![Tab::Scene]);
    state
}

// Log level colours for the console.
const LOG_ERROR: egui::Color32 = crate::theme::STATUS_ERROR;
const LOG_WARN: egui::Color32 = crate::theme::STATUS_WARN_FG;
const LOG_INFO: egui::Color32 = crate::theme::STATUS_INFO;
const LOG_DEBUG: egui::Color32 = crate::theme::TEXT_MUTED;
const LOG_TRACE: egui::Color32 = crate::theme::TEXT_DIM;

/// Occlusion tab state (bake settings + visualization toggles).
struct OcclusionUi {
    settings: zestengine_engine::occlusion::OcclusionSettings,
    /// Cull the Scene view with the baked data.
    cull_scene: bool,
    /// Cull from the game camera's position instead of the Scene camera, so
    /// flying around shows exactly what the game camera's cell draws.
    preview_game: bool,
    /// Draw the baked cell bounds and the culling camera's cell.
    show_cells: bool,
}

impl Default for OcclusionUi {
    fn default() -> Self {
        Self { settings: Default::default(), cull_scene: true, preview_game: false, show_cells: false }
    }
}

/// Log segmented filter (spec §7.2).
#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum LogMode {
    #[default]
    All,
    Warnings,
    Errors,
}

/// Active drag of a component gizmo handle (box face or range radius). One
/// unified state for all `GizmoSpec` kinds; the result is written back through
/// `EditorComponents::apply_gizmo_edit`, so built-ins and plugins resize the
/// same way and the drag suppresses the orbit camera for the whole gesture.
struct GizmoDrag {
    object: usize,
    /// Index into the object's `components`.
    component: usize,
    /// Index into that component's `gizmos()` list (the `GizmoEdit` index).
    gizmo: usize,
    start_cursor: egui::Pos2,
    kind: GizmoDragKind,
}

enum GizmoDragKind {
    BoxFace {
        axis: usize,
        sign: f32,
        start_size: Vec3,
        start_center: Vec3,
        /// Object world translation and transform at drag start.
        start_origin_world: Vec3,
        start_world: glam::Mat4,
        /// World scale along `axis` (component size → world meters).
        scale_a: f32,
        /// Screen pixels per 1 world-meter along `world_axis`.
        screen_axis: egui::Vec2,
    },
    Range {
        start_radius: f32,
        /// Screen pixels per 1 world-meter along the radial (camera-right) dir.
        screen_axis: egui::Vec2,
    },
    /// Point handle dragged in the camera-facing plane.
    Handle {
        start_local: Vec3,
        /// World camera right / up, and their screen pixels per world meter.
        right: Vec3,
        up: Vec3,
        screen_right: egui::Vec2,
        screen_up: egui::Vec2,
        /// Object world matrix inverse (world delta -> local delta).
        inv_world: glam::Mat4,
    },
}

/// Log tab view state: which levels to show, a substring filter, and whether
/// to keep the view pinned to the newest entry.
struct LogFilter {
    error: bool,
    warn: bool,
    info: bool,
    debug: bool,
    trace: bool,
    search: String,
    autoscroll: bool,
    /// Wrap long lines instead of extending (and horizontal-scrolling).
    wrap: bool,
    mode: LogMode,
    /// Filtered view, rebuilt only when the ring or the filter changes.
    cache: LogCache,
    /// Own text selection (anchor, head) by ring seq: egui's label selection
    /// is dropped when an endpoint scrolls out of the virtualised rows.
    sel: Option<(crate::log_select::LogPos, crate::log_select::LogPos)>,
    /// A primary-button selection drag that started inside the log is live.
    sel_drag: bool,
    /// Row geometry of the last frame (hit-testing), the scroll area's inner
    /// rect and its vertical offset.
    geos: Vec<LogRowGeo>,
    area: egui::Rect,
    scroll_y: f32,
}

/// Where one message galley was drawn last frame.
struct LogRowGeo {
    seq: u64,
    /// Physical line (non-wrap mode) or `u32::MAX` for a whole message.
    line: u32,
    galley: std::sync::Arc<egui::Galley>,
    pos: egui::Pos2,
    /// The whole row's rect (for vertical hit-testing).
    band: egui::Rect,
}

impl LogRowGeo {
    fn pos_at(&self, idx: usize) -> crate::log_select::LogPos {
        if self.line == u32::MAX {
            let (l, c) = crate::log_select::idx_to_lc(self.galley.text(), idx);
            (self.seq, l, c)
        } else {
            (self.seq, self.line, idx as u32)
        }
    }
}

/// Caret position under `p`; above/below every row clamps to the nearest row's
/// start/end.
fn log_hit(geos: &[LogRowGeo], p: egui::Pos2) -> Option<crate::log_select::LogPos> {
    let dist = |g: &LogRowGeo| (g.band.min.y - p.y).max(p.y - g.band.max.y).max(0.0);
    let g = geos.iter().min_by(|a, b| dist(a).total_cmp(&dist(b)))?;
    let idx = if p.y < g.band.min.y {
        0
    } else if p.y > g.band.max.y {
        g.galley.text().chars().count()
    } else {
        g.galley.cursor_from_pos(p - g.pos).index
    };
    Some(g.pos_at(idx))
}

/// The Log tab's filtered view (incrementally extended; see `log_capture`).
type LogCache = log_capture::LogView;

impl Default for LogFilter {
    fn default() -> Self {
        Self {
            error: true,
            warn: true,
            info: true,
            // Debug/trace are off the global filter by default; toggles still
            // work if the user raises RUST_LOG.
            debug: true,
            trace: true,
            search: String::new(),
            autoscroll: true,
            wrap: true,
            mode: LogMode::All,
            cache: LogCache::default(),
            sel: None,
            sel_drag: false,
            geos: Vec::new(),
            area: egui::Rect::NOTHING,
            scroll_y: 0.0,
        }
    }
}

impl LogFilter {
    fn shows(&self, level: tracing::Level) -> bool {
        match self.mode {
            LogMode::Warnings => return level == tracing::Level::WARN,
            LogMode::Errors => return level == tracing::Level::ERROR,
            LogMode::All => {}
        }
        match level {
            tracing::Level::ERROR => self.error,
            tracing::Level::WARN => self.warn,
            tracing::Level::INFO => self.info,
            tracing::Level::DEBUG => self.debug,
            tracing::Level::TRACE => self.trace,
        }
    }

    /// Shown levels as a `log_capture::level_bit` mask.
    fn level_mask(&self) -> u8 {
        use tracing::Level as L;
        [L::ERROR, L::WARN, L::INFO, L::DEBUG, L::TRACE]
            .into_iter()
            .filter(|&l| self.shows(l))
            .fold(0, |m, l| m | log_capture::level_bit(l))
    }
}

/// Per-kind visibility + size for a billboard widget.
#[derive(Clone, Copy)]
struct WidgetSetting {
    visible: bool,
    size: f32,
}

impl Default for WidgetSetting {
    fn default() -> Self {
        Self {
            visible: true,
            // Billboards read better a bit larger by default; the widget
            // filter still lets each kind be resized.
            size: 1.6,
        }
    }
}

/// What the viewport draws over the scene (the eye split button, top
/// right): the eye hides or shows every gizmo and billboard at once, its
/// dropdown picks which kinds show. A filtered-off billboard still draws when
/// its object is the current selection.
#[derive(Clone)]
struct WidgetFilter {
    /// Master toggle (the eye): when false, no gizmos or billboards are shown
    /// except the move / rotate / scale handles, which always show.
    enabled: bool,
    /// Component gizmos: collider shapes, volumes, water outlines, joints...
    components: bool,
    lights: WidgetSetting,
    cameras: WidgetSetting,
    probes: WidgetSetting,
    reflection_probes: WidgetSetting,
    audio: WidgetSetting,
    ui_canvases: WidgetSetting,
    post_volumes: WidgetSetting,
    /// Wind zones, navmesh surfaces, particle emitters, portals and terrain.
    placeables: WidgetSetting,
    /// The selected object's grey orientation cross (own toggle, separate from the
    /// transform handles). Hidden when the master toggle is off too.
    cross: bool,
    /// Preview the scene's screen-space UI over the Scene view outside Play
    /// (off by default: the Camera tab shows the game's HUD). Not a gizmo, so
    /// the master toggle doesn't hide it.
    game_ui: bool,
}

impl Default for WidgetFilter {
    fn default() -> Self {
        Self {
            enabled: true,
            components: true,
            lights: WidgetSetting::default(),
            cameras: WidgetSetting::default(),
            probes: WidgetSetting::default(),
            reflection_probes: WidgetSetting::default(),
            audio: WidgetSetting::default(),
            ui_canvases: WidgetSetting::default(),
            post_volumes: WidgetSetting::default(),
            placeables: WidgetSetting::default(),
            cross: true,
            game_ui: false,
        }
    }
}

struct EngineApp {
    config: AppConfig,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    /// OpenXR context (VR). `None` when no runtime/headset; editor runs flat.
    xr: Option<zestengine_xr::XrContext>,
    /// Live VR session (poses/lifecycle), once the device is created.
    xr_session: Option<zestengine_xr::XrSession>,
    /// Latest VR tracker poses for the full-body IK solver (consumed by the
    /// avatar-IK application step — humanoid bone mapping).
    #[allow(dead_code)]
    vr_targets: zestengine_core::TrackerTargets,
    /// Face tracking input (headset always; the OSC receiver only during Play).
    face_input: zestengine_engine::face_input::FaceInput,
    /// A Menu Toggle applied to the scene for a look (restored on stop / save / Play).
    avatar_preview: crate::avatar_edit::TogglePreview,
    /// The radial avatar menu (drawn over the Game view during Play).
    avatar_menu: zestengine_engine::avatar_menu::AvatarMenuHost,
    /// The same menu as a world panel for a headset in Play.
    avatar_menu_vr: zestengine_engine::avatar_menu::AvatarMenuVr,
    /// `/avatar/parameters` OSC bridge for the local avatar.
    avatar_osc: zestengine_engine::avatar_osc_host::AvatarOsc,
    /// SlimeVR server connection (full-body trackers), when enabled.
    slimevr: Option<zestengine_engine::slimevr::SlimeVr>,
    /// The player's measured body, tracker roles and calibration.
    body: zestengine_engine::body::BodyTracking,
    /// VR play-space rig (fly / scale / drag locomotion); identity until driven.
    vr_rig: zestengine_core::VrRig,
    /// The per-eye XR swapchains (`setup_stereo`), once created.
    xr_stereo: Option<zestengine_xr::StereoSetup>,
    /// Grab-drag anchor (world point under the grabbing controller at grab start).
    vr_grab_anchor: Option<glam::Vec3>,
    /// Which hand currently owns the grab-drag (`Some(true)` = left, `Some(false)`
    /// = right). Re-anchors when the active hand changes so either grip can drag.
    vr_drag_hand: Option<bool>,
    /// Hand-menu anchor captured when the menu opens: (head world position at open,
    /// flat forward direction). The panel sits along `flat` at a distance and size
    /// that BOTH scale with the player's VR scale, so it keeps the same apparent
    /// size/distance whether you've scaled yourself up or down. `None` = closed.
    vr_panel_anchor: Option<(glam::Vec3, glam::Vec3)>,
    /// Inter-hand distance last frame (for two-handed scale).
    vr_prev_hand_dist: Option<f32>,
    /// Right trigger state last frame (edge detection for pointer select).
    vr_prev_trigger: bool,
    vr_prev_grip: bool,
    /// Distance along the pointer ray to hold a grabbed object.
    vr_move_dist: f32,
    /// Whether the VR hand menu is open (toggled by the menu button).
    vr_menu_open: bool,
    /// The palette wheel (radial action buttons) is up; the menu button toggles
    /// it, and its actions reach the rest of the VR UI.
    vr_wheel_open: bool,
    /// Menu button state last frame (edge detection).
    vr_prev_menu: bool,
    /// VR UI cursor position in egui points (from the right pointer on the panel).
    vr_ui_pointer: Option<egui::Pos2>,
    /// VR UI click state (right trigger over the panel) + last-frame edge.
    vr_ui_pressed: bool,
    vr_ui_prev_pressed: bool,
    /// What to draw for the VR UI overlay this frame (panel transform + cursor).
    vr_overlay_draw: Option<zestengine_render::VrOverlayDraw>,
    /// Hand-tracking skeletons (stage space) read this frame; mapped to world
    /// and drawn by the XR overlay in edit mode.
    vr_hand_joints: [Option<zestengine_xr::HandJoints>; 2],
    /// Co-editing session (host or joiner); None when not in a session.
    edit_session: Option<zestengine_engine::edit_net::EditSession>,
    /// The Co-edit window is open.
    edit_window: bool,
    /// Host/join form state for the Co-edit window.
    edit_form: EditForm,
    /// Latest presence of every remote peer (for status and rig drawing).
    edit_peers: std::collections::HashMap<u64, zestengine_engine::edit_net::PeerPresence>,
    /// Next presence send (throttled to ~15 Hz).
    edit_presence_next: Option<std::time::Instant>,
    /// Next transform broadcast while VR-dragging (throttled to ~10 Hz).
    edit_transform_next: Option<std::time::Instant>,
    /// Co-edit window's LAN browser (sessions answering on the edit
    /// discovery port), refreshed while the window is open.
    edit_browser: Option<zestengine_engine::net::LanBrowser>,
    /// Next LAN browser refresh.
    edit_browser_next: Option<std::time::Instant>,
    /// Local voice sequence for co-edit mic frames.
    edit_voice_seq: u32,
    scene: LoadedScene,
    egui_ctx: egui::Context,
    egui_state: Option<egui_winit::State>,
    dock_state: DockState<Tab>,
    inspector: InspectorPanel,
    /// Selection the Inspector is pinned to while `inspector.locked` is on.
    inspector_lock_target: Option<Selection>,
    scene_panel: ScenePanel,
    /// Prefab instance markers, Overrides data and Prefab Mode state.
    prefab_editor: prefab_editor::PrefabEditor,
    file_browser: FileBrowser,
    selection: Selection,
    /// Full multi-selection of scene objects (anchor = `selection`). When it
    /// holds >1 and contains the anchor, the inspector shows shared components/
    /// values and edits apply to all; otherwise selection is single (`selection`).
    multi_objects: Vec<usize>,
    /// Values typed into the Inspector this frame for a multi-selection, set
    /// on every selected object by `propagate_multi_edit`.
    multi_edit: MultiEdit,
    /// Full multi-selection of files in the browser (ctrl/shift). The anchor is
    /// `Selection::File`. Used for batch operations and grid highlighting.
    multi_files: Vec<PathBuf>,
    /// Frames to wait after a reflection recapture before reading the cube back to
    /// disk (the capture lands in the next render frame). `Some(0)` = save now.
    refl_bake_pending: Option<u32>,
    file_material: Option<FileMaterial>,
    /// Model `.meta` currently shown in the Inspector (FBX/glTF import options).
    file_meta: Option<FileMeta>,
    /// Texture importer inspector state (preview job, tab, channel, mip).
    texture_ui: crate::texture_inspector::TextureUi,
    /// Rig section / Configure window state of the open model.
    rig_ui: crate::rig_editor::RigUi,
    /// Lazy 3D preview of the selected model / prefab file (inspector).
    asset_preview: crate::asset_preview::AssetPreview,
    /// The Configure window's figure when no scene object uses the model.
    avatar_figure: crate::asset_preview::AvatarFigure,
    sprite_atlas_ui: crate::sprite_atlas_ui::SpriteAtlasInspector,
    /// The model inspector's Scene / Meshes / Geometry / Materials sections.
    model_ui: crate::model_import_editor::ModelImportUi,
    /// Caches behind the material texture-slot warnings.
    slot_validator: crate::texture_inspector::SlotValidator,
    /// Unapplied Model Import edits are being left: Apply / Revert / Cancel is up.
    import_prompt: Option<ImportPrompt>,
    open_editors: Vec<OpenEditor>,
    nvim_embed: crate::nvim_embed::NvimEmbed,
    /// Language server (rust-analyzer) for `.rs` editing; spawned on demand.
    lsp: Option<lsp::LspClient>,
    /// Set if spawning the language server failed, so we don't retry.
    lsp_failed: bool,
    /// In-flight LSP completion/hover requests, keyed by request id.
    lsp_requests: HashMap<i64, LspRequestKind>,
    /// Per-file LSP problem tally (errors, warnings) for file-browser badges;
    /// keyed by absolute path, updated on every publishDiagnostics.
    file_diagnostics: HashMap<PathBuf, (u32, u32)>,
    gizmo: GizmoState,
    widget_filter: WidgetFilter,
    /// A left-drag that began on a gizmo handle: claims the whole gesture so
    /// orbit can't steal it even if the gizmo grabs a frame late.
    gizmo_drag: bool,
    /// The current left-drag began with the pointer over the viewport, so it
    /// may orbit. Drags that start on a panel never orbit.
    orbit_armed: bool,
    /// Set each frame iff the Camera tab was the visible/active dock tab; gates
    /// the camera-preview render so it (and its Flux trace) is skipped when the
    /// tab is hidden.
    camera_tab_visible: bool,
    /// Set each frame iff the Viewport tab was visible; gates the main scene draws
    /// and the viewport Flux trace so they're skipped when the viewport is hidden
    /// (e.g. only the Camera tab is shown).
    viewport_visible: bool,
    /// Width of the toolbar drawn over the right of the Scene view's tab strip
    /// (last frame), kept clear when fitting that leaf's tabs.
    vp_strip_width: f32,
    /// Audio playback (None if no output device). Drives AudioSource sounds
    /// in Play mode.
    audio: Option<audio::AudioEngine>,
    actions: Vec<EditorAction>,
    undo_stack: UndoStack,
    /// Set while applying undo/redo so the frame diff doesn't re-record it.
    suppress_undo_record: bool,
    /// Registered component types (built-ins now; plugins extend this).
    components: ComponentRegistry,
    /// Editor-side inspector/gizmo dispatch (built-ins + plugin editor traits).
    editor_components: EditorComponents,
    /// Play mode: components run every frame; edits aren't recorded to undo.
    playing: bool,
    /// Paused while playing: components/physics/audio freeze but the played
    /// state stays so you can inspect it. Cleared on Stop.
    play_paused: bool,
    /// Advance one fixed frame while paused (transport Step).
    play_step: bool,
    command_palette: crate::command_palette::CommandPalette,
    viewport_stats: bool,
    /// Tabs whose body was drawn last frame (the visible tab of each leaf),
    /// for the accent underline drawn on their tab buttons.
    active_tabs: std::collections::HashSet<Tab>,
    /// Play clock (seconds). Advances only while playing and not paused, so
    /// time-based components don't jump across a pause. Reset on Play start.
    play_time: f32,
    /// Object state captured at Play start, restored at Stop.
    play_snapshot: Option<PlaySnapshot>,
    /// The running game (components, physics, audio), shared with the
    /// standalone runtime. Some while playing.
    play_session: Option<zestengine_engine::play::PlaySession>,
    /// While playing, render the viewport from the scene's active camera (the
    /// game's view) instead of the editor fly camera. Toolbar toggle.
    play_game_camera: bool,
    /// Set when Play starts: bring the Camera tab (the game view) to the front
    /// once, after the dock is laid out this frame.
    focus_game_tab: bool,
    /// Set when Play stops: bring the Viewport tab (the Scene view) back to the
    /// front so the editor lands in the Scene view, not the Camera view.
    focus_scene_tab: bool,
    occlusion_ui: OcclusionUi,
    /// Screen rect of the Camera tab's image (the game view while playing).
    camera_rect: egui::Rect,
    /// Skybox / baked lighting still to apply after a background scene load.
    post_load: Option<zestengine_engine::scene_loader::PostLoad>,
    /// Play-mode scene streaming (preloads, carried objects, additive scenes).
    flow: zestengine_engine::scene_loader::SceneFlow,
    /// Play-mode user content (ctx.load_content / spawn_avatar / enter_world).
    content: zestengine_engine::content::ContentManager,
    /// Tools > Build Content Package window.
    content_builder: crate::content_build::BuildWindow,
    game_ui: zestengine_engine::game_ui::GameUi,
    ui_editor: crate::ui_editor::UiEditorTab,
    anim_ctrl_tab: crate::anim_controller_editor::AnimControllerTab,
    anim_clip_tab: crate::anim_clip_editor::AnimClipTab,
    sequencer_tab: crate::sequencer_editor::SequencerTab,
    shader_graph_tab: crate::shader_graph_editor::ShaderGraphTab,
    /// Move / resize handles for screen UI in the viewport overlay.
    ui_rect_editor: crate::ui_edit::UiRectEditor,
    /// UI clicks / hover since the last play tick.
    game_ui_frame: zestengine_engine::game_ui::UiFrame,
    world_ui: zestengine_engine::game_ui::WorldUi,
    /// Particle simulation + buffers (previews while editing).
    particles: zestengine_engine::particles::ParticleRenderer,
    water: zestengine_engine::water_render::WaterRenderer,
    water_fx: zestengine_engine::water_fx::WaterFx,
    carry_fx: zestengine_engine::water_carry::CarryFx,
    editor_water: crate::editor_water::EditorWater,
    /// Baked navmesh overlay (mesh, material, bounds center, radius).
    nav_preview: Vec<(zestengine_render::MeshHandle, zestengine_render::MaterialHandle, Vec3, f32)>,
    world_ui_draws: Vec<zestengine_render::WorldUiDraw>,
    /// Last frame's (mouse primary, VR trigger) down state for world-UI edges.
    world_ui_prev: (bool, bool),
    /// Gameplay asked for a locked cursor (`ctx.set_cursor_locked`).
    game_cursor_wanted: bool,
    /// The cursor is currently grabbed for the game (Escape releases it,
    /// clicking the viewport grabs it again).
    game_cursor_locked: bool,
    /// Set if a component switched scenes during play (via `ctx.load_scene`).
    /// Stop then reloads `play_origin_scene` instead of the snapshot restore.
    play_scene_switched: bool,
    /// Scene path active when Play started; reloaded on Stop after a runtime
    /// scene switch. None if the pre-play scene was never saved.
    play_origin_scene: Option<PathBuf>,
    /// Custom shader compile cache + hot reload.
    shaders: ShaderLibrary,
    /// Project-relative `.frag` paths for the shader picker.
    shader_files: Vec<String>,
    /// Last shader-file scan / hot-reload poll.
    last_shader_scan: Option<Instant>,
    /// In-flight project walk for the shader picker (runs off the UI thread).
    shader_scan_rx: Option<std::sync::mpsc::Receiver<Vec<String>>>,
    /// Last-seen mtime of the engine's standard.frag, for dev-time hot-reload
    /// when it is edited on disk outside the editor.
    engine_shader_mtime: Option<std::time::SystemTime>,
    /// Wall-clock of the last asset-change check (on window-focus regain), so
    /// externally-edited assets (e.g. a re-exported FBX) reimport automatically.
    last_asset_check: Option<std::time::SystemTime>,
    /// Scene materials edited since their last auto-save.
    dirty_materials: HashSet<usize>,
    /// Debounce for material auto-save (save when the gesture settles).
    last_material_edit: Option<Instant>,
    /// Rust component plugins (built + hot-loaded dylibs).
    plugins: plugins::PluginHost,
    /// Last failed plugin build's compiler output (shown in a window).
    plugin_build_error: Option<String>,
    /// Global status-bar message: (text, when set, show a spinner).
    status: Option<(String, Instant, bool)>,
    /// A plugin reload was requested; deferred one frame so the status bar can
    /// show "Compiling components…" before the (blocking) cargo build.
    reload_pending: bool,
    /// A heavy op to run after this frame paints its busy overlay (so the UI
    /// shows it's working instead of appearing frozen).
    pending_job: Option<PendingJob>,
    /// Background component-plugin build state + parked work (plugin_gate.rs).
    plugin_gate: crate::plugin_gate::BuildGate<PluginWaiter>,
    plugin_build_started: Option<Instant>,
    /// Plugin source fingerprint at the last successful build; Play rebuilds
    /// when the sources changed since (external editors included).
    plugin_fingerprint: Option<u64>,
    /// Frame-rate limiter for MAILBOX / IMMEDIATE (see `frame_limiter`).
    pacing: zestengine_render::frame_limiter::FramePacing,
    /// project.zestengine: name, last scene, per-project engine settings.
    project: zestengine_assets::ProjectFile,
    camera: FlyCamera,
    /// Orbit pivot, locked for the duration of one left-drag.
    orbit_pivot: Option<Vec3>,
    /// Right mouse held: mouse-look (cursor hidden + locked) + WASD fly.
    looking: bool,
    /// Set the frame mouse-look ends: egui's pointer was frozen during look
    /// (it never saw window events), so the next frame injects a PointerMoved
    /// to resync it, otherwise the first click is hit-tested at the stale
    /// position and the orbit-arm edge is missed (orbit dead until a 2nd click).
    look_just_ended: bool,
    /// Middle mouse held: pan.
    panning: bool,
    /// Raw mouse deltas accumulated while looking.
    look_delta: (f64, f64),
    keys: HashSet<KeyCode>,
    /// Ctrl / Shift / Alt as the window reports them.
    modifiers: winit::keyboard::ModifiersState,
    /// Input binding system (2C): drives Play-mode components from the active
    /// control scheme; editable in the Bindings window.
    input: zestengine_engine::input_engine::InputManager,
    /// Active networking session (2G) when hosting/joined from the Network panel.
    net: Option<zestengine_engine::net::NetSession>,
    /// Voice comms (task 8), created lazily once networking is active.
    voice: Option<zestengine_engine::voice::VoiceSystem>,
    /// Address entered in the Network panel for Join.
    net_addr: String,
    /// "Play as Server": the local dedicated server this editor joined.
    local_server: Option<zestengine_engine::local_server::LocalServer>,
    /// Network window extras (player name, LAN browser).
    net_ui: crate::net_window::NetWindowState,
    /// Performance window (GEO-2): ranks and avatar safety settings.
    perf_ui: crate::perf_window::PerfWindowState,
    show_performance: bool,
    /// Skinning Editor (2D-7), opened by double-clicking a `.rig2d`.
    skinning: crate::skinning_editor::SkinningEditorWindow,
    /// Bindings editor window open.
    show_bindings: bool,
    /// Network panel open.
    show_network: bool,
    /// Layers settings (names + collision matrix) window open.
    show_layers: bool,
    /// Weapon Pose editor window open.
    show_pose_editor: bool,
    /// Audio mixer window open.
    show_mixer: bool,
    /// The Audio Mixer panel edited the project's buses this frame.
    mixer_changed: bool,
    /// Audio bus mixer (volumes/mutes per bus). Drives playback gains.
    /// The project bindings editor (Project Settings > Input and the Input
    /// Bindings window); a pending rebind captures the next input.
    bindings_editor: crate::BindingsEditor,
    last_cursor: Option<(f64, f64)>,
    /// Viewport tab rect in egui points (updated每 frame by the tab).
    viewport_rect: egui::Rect,
    project_root: PathBuf,
    current_scene_path: Option<PathBuf>,
    /// Models (project-relative paths) that the in-progress bake found without a
    /// lightmap UV. `Some` shows a modal asking whether to generate one; cleared
    /// when the user picks an option.
    pending_lm_uv_models: Option<Vec<String>>,
    /// Megascans / Fab import dialog, Downloads watcher and offers (`pack_import_ui`).
    pack_import: crate::pack_import_ui::PackImportUi,
    scene_name_input: String,
    show_stats: bool,
    /// Desired state of the separate profiler window (persisted). Reconciled
    /// against `profiler_window` each frame to open/close the real OS window.
    show_stats_overlay: bool,
    /// The separate profiler OS window (movable to another monitor), its own
    /// egui context + winit input state, and a rolling history for the graphs.
    /// CPU-drawn splash window shown during startup; closed after the first
    /// editor frame. `pending_init` defers the heavy `init` until after the
    /// splash has been shown (so it's actually visible during the blocking load).
    /// Background task system (worker imports, chunked bake) + status-bar UI.
    tasks: crate::tasks::TaskManager,
    /// The task-list entry mirroring the cluster DAG build in flight.
    cluster_task: crate::tasks::ClusterBuildTask,
    /// Whether the status-bar background-tasks popup is expanded.
    show_task_popup: bool,
    /// Active chunked bake, stepped one unit per frame; None when idle.
    bake_job: Option<BakeRunner>,
    /// True for one frame after a bake starts, so the bake modal paints before
    /// the first UI-locking lightmap step.
    bake_pending_warmup: bool,
    /// Model imports that finished parsing while a bake is running; their GPU
    /// upload is deferred until the bake ends (it would realloc `meshes` and
    /// invalidate the bake's TLAS).
    deferred_model_applies: Vec<(PathBuf, zestengine_assets::Scene)>,
    /// Asset placement (drag a model / prefab / clip into the scene); see `placement_glue`.
    placement: placement_glue::Run,
    splash: Option<crate::splash::Splash>,
    pending_init: bool,
    /// Set once the splash has actually painted a frame, so the load only starts
    /// after it's visible (Wayland needs a configure roundtrip).
    splash_painted: bool,
    /// Flat-path renderer build in flight on a worker thread; `about_to_wait`
    /// polls this and calls `finish_init` when the renderer arrives.
    renderer_rx: Option<std::sync::mpsc::Receiver<Result<Renderer>>>,
    /// Startup plugin `cargo build`, running alongside the renderer build.
    plugin_rx: Option<std::sync::mpsc::Receiver<Result<Vec<String>>>>,
    /// The plugin build's one-line progress (elapsed seconds, up to date, first error).
    plugin_status: std::sync::Arc<std::sync::Mutex<String>>,
    /// Renderer (and VR context) waiting for the plugin build to finish.
    ready_renderer: Option<(Renderer, Option<zestengine_xr::XrContext>)>,
    /// The main window, held while the renderer + scene load before the editor
    /// opens (so it never appears half-loaded).
    pending_window: Option<Arc<Window>>,
    /// Current load phase ("Building plugins…", "Loading 3D models…", …), shown
    /// on the splash at startup and the modal for in-editor scene loads. Shared
    /// so the scene-parse worker can update it too.
    load_status: std::sync::Arc<std::sync::Mutex<String>>,
    profiler_window: Option<Arc<Window>>,
    profiler_egui_ctx: egui::Context,
    profiler_egui_state: Option<egui_winit::State>,
    prof_history: ProfHistory,
    /// Last frame's render stats, cached so the profiler window UI can read them
    /// outside the main egui pass.
    last_render_stats: zestengine_render::RenderStats,
    show_help: bool,
    about_open: bool,
    /// Set by the dockable Project Settings tab when it edits the project;
    /// saved once after the dock pass (see `redraw`).
    project_settings_dirty: bool,
    /// New Project dialog open + its inputs (parent dir, project name).
    show_new_project: bool,
    new_project_parent: String,
    new_project_name: String,
    /// Template id picked in the New Project dialog, and its category filter (None = all).
    new_project_template: String,
    new_project_filter: Option<zestengine_engine::templates::TemplateCategory>,
    /// Open Project dialog open + the folder path being entered.
    show_open_project: bool,
    /// Project chosen while the scene had unsaved edits: the Save / Discard /
    /// Cancel prompt is up, then the switch goes ahead.
    pending_project_switch: Option<PathBuf>,
    open_project_path: String,
    /// Baker's Man: render objects as a lightmap-UV checker (texel-density preview).
    lightmap_preview: bool,
    /// Last Build Game result message (shown in the settings/log).
    build_status: Option<String>,
    /// A game build task is running (one at a time).
    building_game: bool,
    /// Target picked in Project Settings.
    build_target: bundle::BuildTarget,
    /// Scene has unsaved edits since the last save/load (drives the exit prompt).
    scene_dirty: bool,
    /// Close requested with unsaved edits: the Save/Discard/Cancel dialog is up.
    show_quit_dialog: bool,
    /// Set by the quit dialog (or a clean close) to exit after this frame; the
    /// redraw loop performs the actual `event_loop.exit()`.
    quit_after_frame: bool,
    /// View > VR > "Restart editor": once this process has saved and is about to
    /// exit, start a fresh editor on the same project (see `relaunch_editor`).
    relaunch_after_quit: bool,
    /// User-level editor preferences (`editor.ron`): the editor's own VR switch.
    editor_prefs: crate::editor_prefs::EditorPrefs,
    /// Log tab filters (levels + search + autoscroll).
    log_filter: LogFilter,
    /// In-progress component-gizmo handle drag (box face / range radius).
    handle_drag: Option<GizmoDrag>,
    stats: Stats,
    #[allow(dead_code)] // entities arrive with the component-system milestone
    world: hecs::World,
    start: Instant,
    last_frame: Instant,
    /// Realtime-GI preview: temporally-accumulated probe SH from the last few
    /// updates (blended toward each new probe-only trace), the grid dimensions
    /// it was built for (a layout change forces a reset), whether RT-GI probes
    /// are currently uploaded (so we can clear them when it turns off), and a
    /// throttle timer.
    /// Realtime-GI driver (continuous probe re-trace from the realtime lights).
    rt_gi: zestengine_engine::realtime_gi::RealtimeGiState,
    /// The two reflection probes' cubes (renderer slots 0 / 1).
    refl_slots: zestengine_engine::refl_slots::ReflectionSlots,
    /// Reflection probes still to bake: (probe id, capture point, resolution).
    refl_bake_queue: Vec<(u128, glam::Vec3, u32)>,
    /// Probe whose static capture is being baked right now.
    refl_bake_current: Option<u128>,
    /// Captured probe faces waiting for the background encode + save.
    refl_unsaved: Vec<(u128, zestengine_engine::refl_slots::Faces)>,
    /// Newest reflection save applied (older results arriving late are dropped).
    refl_applied_seq: u64,
    /// Bumped per lighting bake; a lightmap save only applies to its own bake.
    bake_serial: u64,
    /// Cached sidecar size stats for the Baker's Man tab (None = re-measure).
    bake_sizes: Option<crate::bake_store::BakeSizes>,
    /// Which material slot of the selected object the inspector edits (for
    /// multi-material meshes). Clamped to the object's slot count.
    /// Object currently open in ZestPaint; its return manifest is polled.
    zestsurface_object: Option<zestengine_core::ObjectId>,
    /// EMA-smoothed per-frame CPU section timings (ms) for the stats overlay.
    frame_timings: FrameTimings,
    /// GI debug view: 0 = off, 1 = world normals, 2 = indirect/GI term.
    gi_debug: u32,
}

/// Per-frame main-thread CPU costs (ms), EMA-smoothed for a readable overlay.
#[derive(Default, Clone, Copy)]
struct FrameTimings {
    gi: f32,
    components: f32,
    physics: f32,
    audio: f32,
    draws: f32,
    render: f32,
    /// GPU-side timings (ms), read back from timestamp queries via RenderStats.
    gpu_frame: f32,
    gpu_gi: f32,
}

impl FrameTimings {
    /// Blend a fresh sample into the smoothed value.
    fn ema(slot: &mut f32, sample_ms: f32) {
        *slot += (sample_ms - *slot) * 0.2;
    }
}

/// A rolling series of time-bucketed samples (one value per ~50 ms bucket), so
/// the graph always spans the same wall-clock window regardless of frame rate.
struct Series {
    data: std::collections::VecDeque<f32>,
}

impl Default for Series {
    fn default() -> Self {
        Self {
            data: std::collections::VecDeque::with_capacity(Series::CAP),
        }
    }
}

impl Series {
    fn push(&mut self, v: f32) {
        if self.data.len() == Series::CAP {
            self.data.pop_front();
        }
        self.data.push_back(v);
    }

    fn last(&self) -> f32 {
        self.data.back().copied().unwrap_or(0.0)
    }

    fn max(&self) -> f32 {
        self.data.iter().copied().fold(0.0_f32, f32::max)
    }
}

/// One graphed metric: its rolling series, display style, and visibility.
struct Metric {
    label: &'static str,
    color: egui::Color32,
    unit: &'static str,
    visible: bool,
    series: Series,
    /// Max over the in-progress time bucket (flushed to `series` per bucket).
    pending: f32,
}

impl Metric {
    fn new(label: &'static str, color: egui::Color32, unit: &'static str, visible: bool) -> Self {
        Self {
            label,
            color,
            unit,
            visible,
            series: Series::default(),
            pending: 0.0,
        }
    }
}

/// Rolling history of every profiler metric, in a fixed order (see the index
/// constants), plus the current time-bucket accumulator.
struct ProfHistory {
    metrics: Vec<Metric>,
    bucket_t: f32,
}

impl Series {
    /// Graph window / resolution: 20 s of history at one sample per 50 ms.
    const WINDOW: f32 = 20.0;
    const BUCKET: f32 = 0.05;
    const CAP: usize = (Series::WINDOW / Series::BUCKET) as usize;
}

impl Default for ProfHistory {
    fn default() -> Self {
        use egui::Color32 as C;
        // Order MUST match `record_prof_sample`'s value array. Defaults: the
        // headline costs visible, the rest off to keep the graph readable.
        Self {
            metrics: vec![
                Metric::new("Frame time", C::from_rgb(120, 200, 255), "ms", true),
                Metric::new("FPS", C::from_rgb(140, 220, 140), "", false),
                Metric::new("GPU frame", C::from_rgb(255, 170, 90), "ms", true),
                Metric::new("GPU: Scene Flux GI", C::from_rgb(255, 120, 200), "ms", true),
                Metric::new("GPU: shadows", C::from_rgb(160, 210, 255), "ms", true),
                Metric::new("GPU: scene", C::from_rgb(255, 230, 120), "ms", true),
                Metric::new("GPU: reflections", C::from_rgb(120, 230, 200), "ms", true),
                Metric::new("GPU: post", C::from_rgb(210, 160, 255), "ms", true),
                Metric::new("GPU: egui", C::from_rgb(255, 180, 160), "ms", false),
                Metric::new("GPU: camera preview", C::from_rgb(180, 255, 160), "ms", false),
                Metric::new("GPU utilization", C::from_rgb(230, 200, 90), "%", false),
                Metric::new("CPU realtime GI", C::from_rgb(150, 180, 230), "ms", false),
                Metric::new("CPU components", C::from_rgb(120, 220, 220), "ms", false),
                Metric::new("CPU physics", C::from_rgb(180, 160, 240), "ms", false),
                Metric::new("CPU audio", C::from_rgb(200, 220, 130), "ms", false),
                Metric::new("CPU build draws", C::from_rgb(160, 200, 160), "ms", false),
                Metric::new("CPU render submit", C::from_rgb(220, 130, 130), "ms", false),
                Metric::new("Draw calls", C::from_rgb(190, 190, 190), "", false),
                Metric::new("GPU: Camera Flux GI", C::from_rgb(255, 90, 140), "ms", true),
                Metric::new("GPU: Camera GI gather", C::from_rgb(220, 100, 180), "ms", false),
                Metric::new("GPU: Camera GI denoise", C::from_rgb(190, 100, 210), "ms", false),
                Metric::new("GPU: Camera GI integrate", C::from_rgb(160, 100, 240), "ms", false),
                Metric::new("GPU: Camera GI cache", C::from_rgb(130, 160, 240), "ms", false),
                Metric::new("GPU: Scene GI cache", C::from_rgb(160, 210, 240), "ms", false),
            ],
            bucket_t: 0.0,
        }
    }
}

/// Round up to a "nice" axis maximum (1/2/5 × 10ⁿ) so the y grid lands on
/// readable values.
fn about_color_image() -> Option<egui::ColorImage> {
    use std::sync::OnceLock;
    static IMG: OnceLock<Option<egui::ColorImage>> = OnceLock::new();
    IMG.get_or_init(|| {
        let bytes = include_bytes!("../../zestengine-assets/zestengine.jpg");
        image::load_from_memory(bytes).ok().map(|img| {
            let rgba = img.to_rgba8();
            let (w, h) = rgba.dimensions();
            egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], rgba.as_raw())
        })
    })
    .clone()
}

fn nice_ceil(v: f32) -> f32 {
    if v <= 0.0 {
        return 1.0;
    }
    let exp = v.log10().floor();
    let base = 10f32.powf(exp);
    let n = v / base;
    let nice = if n <= 1.0 {
        1.0
    } else if n <= 2.0 {
        2.0
    } else if n <= 5.0 {
        5.0
    } else {
        10.0
    };
    nice * base
}

/// Draw all visible metrics into one combined graph with a shared linear axis
/// (value on the left and right edges, time along the bottom). Hovering picks the
/// nearest line: it's drawn bright while the others dim, and a tooltip shows that
/// metric's name, value, and how long ago the sample is.
fn draw_combined_graph(ui: &mut egui::Ui, h: &ProfHistory) {
    let (rect, resp) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), 260.0), egui::Sense::hover());
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 3.0, egui::Color32::from_gray(16));

    // Inset a plot area, leaving margins for the axis labels.
    let plot = egui::Rect::from_min_max(
        egui::pos2(rect.left() + 46.0, rect.top() + 8.0),
        egui::pos2(rect.right() - 46.0, rect.bottom() - 18.0),
    );

    // Shared axis: scale to the largest visible value, and use the common unit
    // when every visible series shares one (e.g. all "ms").
    let mut gmax = 0.0f32;
    let mut unit: Option<&str> = None;
    let mut unit_mismatch = false;
    for m in h.metrics.iter().filter(|m| m.visible) {
        gmax = gmax.max(m.series.max());
        match unit {
            None => unit = Some(m.unit),
            Some(u) if u != m.unit => unit_mismatch = true,
            _ => {}
        }
    }
    let unit = if unit_mismatch { "" } else { unit.unwrap_or("") };
    let gmax = nice_ceil(gmax.max(1e-3));
    let grid = egui::Color32::from_gray(34);
    let lbl_col = egui::Color32::from_gray(130);

    // Horizontal grid + value labels on both sides.
    for k in 0..=4 {
        let f = k as f32 / 4.0;
        let y = plot.bottom() - f * plot.height();
        p.line_segment(
            [egui::pos2(plot.left(), y), egui::pos2(plot.right(), y)],
            egui::Stroke::new(1.0_f32, grid),
        );
        let val = gmax * f;
        let txt = if unit.is_empty() {
            format!("{val:.1}")
        } else {
            format!("{val:.1}{unit}")
        };
        p.text(
            egui::pos2(plot.left() - 4.0, y),
            egui::Align2::RIGHT_CENTER,
            &txt,
            egui::FontId::monospace(10.0),
            lbl_col,
        );
        p.text(
            egui::pos2(plot.right() + 4.0, y),
            egui::Align2::LEFT_CENTER,
            &txt,
            egui::FontId::monospace(10.0),
            lbl_col,
        );
    }
    // Vertical grid + time labels (every 5 s; "now" on the right).
    let tdiv = (Series::WINDOW / 5.0) as i32;
    for k in 0..=tdiv {
        let f = k as f32 / tdiv as f32;
        let x = plot.right() - f * plot.width();
        p.line_segment(
            [egui::pos2(x, plot.top()), egui::pos2(x, plot.bottom())],
            egui::Stroke::new(1.0_f32, grid),
        );
        let label = if k == 0 {
            "now".to_string()
        } else {
            format!("-{}s", (f * Series::WINDOW) as i32)
        };
        p.text(
            egui::pos2(x, plot.bottom() + 3.0),
            egui::Align2::CENTER_TOP,
            label,
            egui::FontId::monospace(10.0),
            lbl_col,
        );
    }

    let n = h
        .metrics
        .iter()
        .filter(|m| m.visible)
        .map(|m| m.series.data.len())
        .max()
        .unwrap_or(0);
    if n < 2 {
        return;
    }
    let dx = plot.width() / (Series::CAP.saturating_sub(1)).max(1) as f32;
    let x0 = plot.right() - dx * (n - 1) as f32; // newest on the right
    let y_of = |v: f32| plot.bottom() - (v / gmax).clamp(0.0, 1.0) * plot.height();

    // Hover: nearest sample column, then the visible series whose point is
    // closest to the cursor.
    let hover = resp.hover_pos().filter(|pos| plot.contains(*pos));
    let mut hovered: Option<usize> = None;
    let mut hover_i = 0usize;
    if let Some(pos) = hover {
        hover_i = (((pos.x - x0) / dx).round() as i32).clamp(0, n as i32 - 1) as usize;
        let mut best = f32::INFINITY;
        for (mi, m) in h.metrics.iter().enumerate().filter(|(_, m)| m.visible) {
            if let Some(&v) = m.series.data.get(hover_i) {
                let d = (y_of(v) - pos.y).abs();
                if d < best {
                    best = d;
                    hovered = Some(mi);
                }
            }
        }
    }

    for (mi, m) in h.metrics.iter().enumerate().filter(|(_, m)| m.visible) {
        let dimmed = hovered.is_some() && hovered != Some(mi);
        let color = if dimmed {
            m.color.linear_multiply(0.22)
        } else {
            m.color
        };
        let width = if hovered == Some(mi) { 2.5_f32 } else { 1.3_f32 };
        let pts: Vec<egui::Pos2> = m
            .series
            .data
            .iter()
            .enumerate()
            .map(|(i, &v)| egui::pos2(x0 + dx * i as f32, y_of(v)))
            .collect();
        p.add(egui::Shape::line(pts, egui::Stroke::new(width, color)));
    }

    // Hover marker + tooltip for the picked series.
    if let (Some(pos), Some(mi)) = (hover, hovered) {
        let m = &h.metrics[mi];
        let v = m.series.data.get(hover_i).copied().unwrap_or(0.0);
        let x = x0 + dx * hover_i as f32;
        let y = y_of(v);
        p.line_segment(
            [egui::pos2(x, plot.top()), egui::pos2(x, plot.bottom())],
            egui::Stroke::new(1.0_f32, egui::Color32::from_gray(90)),
        );
        p.circle_filled(egui::pos2(x, y), 3.5, m.color);
        let ago = (n - 1 - hover_i) as f32 * Series::BUCKET;
        let unit = if m.unit.is_empty() {
            String::new()
        } else {
            format!(" {}", m.unit)
        };
        let txt = format!("{}\n{v:.2}{unit}   {ago:.1}s ago", m.label);
        let galley = p.layout_no_wrap(txt, egui::FontId::monospace(12.0), m.color);
        let sz = galley.size();
        let tip = egui::pos2(
            (pos.x + 12.0).min(plot.right() - sz.x - 6.0),
            (pos.y + 12.0).min(plot.bottom() - sz.y - 6.0),
        );
        p.rect_filled(
            egui::Rect::from_min_size(tip - egui::vec2(5.0, 3.0), sz + egui::vec2(10.0, 6.0)),
            3.0,
            egui::Color32::from_black_alpha(230),
        );
        p.galley(tip, galley, m.color);
    }
}

/// A clickable legend row (full-width hit area): colour swatch + label + current
/// value and rolling max. Returns the response so the caller can toggle the
/// metric's visibility; hidden metrics are dimmed.
fn legend_row(ui: &mut egui::Ui, m: &Metric) -> egui::Response {
    let cur = m.series.last();
    let maxv = m.series.max();
    let text_col = if m.visible {
        egui::Color32::WHITE
    } else {
        egui::Color32::from_gray(110)
    };
    let resp = ui
        .horizontal(|ui| {
            let (sw, _) =
                ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::hover());
            let sw_col = if m.visible { m.color } else { egui::Color32::from_gray(70) };
            ui.painter().rect_filled(sw, 2.0, sw_col);
            ui.label(egui::RichText::new(m.label).size(13.0).color(text_col));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let unit = if m.unit.is_empty() {
                    String::new()
                } else {
                    format!(" {}", m.unit)
                };
                ui.label(
                    egui::RichText::new(format!("{cur:.2}{unit}   (max {maxv:.1})"))
                        .monospace()
                        .size(12.0)
                        .color(text_col),
                );
            });
        })
        .response;
    // Make the whole row a click target (labels don't consume clicks).
    let resp = resp.interact(egui::Sense::click());
    if resp.hovered() {
        ui.painter()
            .rect_filled(resp.rect, 2.0, egui::Color32::from_white_alpha(8));
    }
    resp
}

/// Shared egui style for the editor (and the profiler window, which uses its own
/// context): bump tiny text to a readable floor, and use a thin floating
/// scrollbar.
///
/// The bar is a thin OVERLAY (`floating`, no allocated gutter): reserving width
/// for the bar only when it's shown reflows the content, which toggles the
/// scrollbar and bounces the layout every frame (seen in the inspector). Keeping
/// it a pure overlay means the bar never changes layout. (Avoiding overlap with
/// content is a per-ScrollArea concern — e.g. `auto_shrink`/inner margin — not a
/// global one.)
/// Set the editor accent (None = the default yellow) and restyle egui.
fn apply_accent(ctx: &egui::Context, accent: Option<[u8; 3]>) {
    let c = accent.map_or(crate::theme::ACCENT_DEFAULT, |[r, g, b]| egui::Color32::from_rgb(r, g, b));
    crate::theme::set_accent(c);
    crate::theme::apply(ctx);
}

/// The editor look (docs/design/editor-ui-spec.md): tokens, widget states,
/// spacing and type from `crate::theme`.
fn apply_editor_style(ctx: &egui::Context) {
    crate::theme::apply(ctx);
}

/// A plain label + right-aligned integer count row (no graph).
fn count_row(ui: &mut egui::Ui, label: &str, value: u32) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).size(13.0).color(egui::Color32::WHITE));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                egui::RichText::new(value.to_string())
                    .monospace()
                    .size(13.0)
                    .color(egui::Color32::WHITE),
            );
        });
    });
}

impl EngineApp {
    /// Update the splash window's status line (no-op once it's closed).
    /// Set the current load phase. Updates the shared string (read by the splash
    /// each tick and the in-editor modal) and presents the splash immediately so
    /// the phase shows even right before a blocking main-thread step.
    fn set_load_status(&mut self, status: &str) {
        *self.load_status.lock().unwrap() = status.to_string();
        if let Some(s) = self.splash.as_mut() {
            s.set_status(status);
        }
    }

    /// Enter startup step `step` of [`LOAD_STEPS`]: advances the splash's
    /// progress bar and shows `status`.
    fn set_load_step(&mut self, step: usize, status: &str) {
        if let Some(s) = self.splash.as_mut() {
            s.set_progress(step, LOAD_STEPS);
        }
        self.set_load_status(status);
    }

    /// Add the open project to the shared recent-projects list (the Project
    /// Hub shows it), so projects opened from the command line appear there.
    fn record_recent_project(&self) {
        if let Err(e) = zestengine_assets::recent_projects::record_opened(&self.project_root, &self.project.name, env!("CARGO_PKG_VERSION")) {
            tracing::warn!("recording recent project: {e:#}");
        }
    }

    /// Start the Project Hub (a separate executable next to the editor).
    fn launch_project_hub(&mut self) {
        use zestengine_assets::launcher;
        let started = launcher::find_executable(launcher::HUB_BIN, None, launcher::HUB_ENV).and_then(|hub| {
            let cmd = launcher::LaunchCommand::hub(&hub);
            cmd.command().spawn().map_err(|e| anyhow::anyhow!("starting {}: {e}", cmd.display()))
        });
        match started {
            Ok(_) => self.tasks.notify("Project Hub opened", crate::tasks::NotifyLevel::Info),
            Err(e) => {
                tracing::error!("Project Hub: {e:#}");
                self.tasks.notify(format!("Project Hub: {e}"), crate::tasks::NotifyLevel::Warn);
            }
        }
    }

    /// Whether the next editor start would try OpenXR: `vr_wanted_at_start`
    /// with this process's environment, the View > VR preference and the
    /// project's "Start in VR" as it is now.
    fn vr_wanted_next_start(&self) -> bool {
        crate::editor_prefs::vr_wanted_at_start(
            std::env::var("ZESTENGINE_VR").ok().as_deref(),
            zestengine_xr::vr_disabled_globally().is_some(),
            self.editor_prefs.vr,
            self.project.vr,
        )
    }

    /// View > VR: save the editor's VR preference (`editor.ron`, per user) and
    /// apply what can be applied now. Off while a session runs ends it live;
    /// on needs a restart, since the Vulkan device must be created against the
    /// OpenXR instance.
    fn set_vr_preference(&mut self, on: bool) {
        self.editor_prefs.vr = Some(on);
        self.save_editor_prefs();
        if !on && self.xr_session.is_some() {
            self.stop_vr();
            self.tasks.notify("VR stopped", crate::tasks::NotifyLevel::Info);
        } else if on && self.xr_session.is_none() {
            match zestengine_xr::vr_disabled_globally() {
                Some(why) => self.tasks.notify(format!("VR stays off on this machine: {why}"), crate::tasks::NotifyLevel::Warn),
                None => self.tasks.notify("VR starts at the next editor start (View > VR > Restart editor)", crate::tasks::NotifyLevel::Info),
            }
        }
    }

    fn save_editor_prefs(&mut self) {
        match crate::editor_prefs::EditorPrefs::default_path() {
            Some(path) => {
                if let Err(e) = self.editor_prefs.save(&path) {
                    tracing::error!("saving editor preferences: {e:#}");
                    self.tasks.notify(format!("Editor preferences not saved: {e}"), crate::tasks::NotifyLevel::Warn);
                } else {
                    tracing::info!("editor preferences saved to {} (vr: {:?})", path.display(), self.editor_prefs.vr);
                }
            }
            None => tracing::warn!("no user config folder: the editor preferences are not saved"),
        }
    }

    /// End the OpenXR session now and carry on flat. The eye renders, the IK
    /// targets and the hand menu are all gated on `xr_session`, so the desktop
    /// window keeps drawing as before; the GPU is idled first so no in-flight
    /// eye frame still writes a swapchain image the runtime is about to
    /// destroy. Starting VR again needs a restart.
    fn stop_vr(&mut self) {
        if self.xr_session.is_none() {
            return;
        }
        if let Some(renderer) = self.renderer.as_mut()
            && let Err(e) = renderer.flush_deferred_frees()
        {
            tracing::warn!("idling the GPU before ending the VR session: {e:#}");
        }
        // The session (and its swapchains) before the instance, like `Drop`.
        self.xr_session = None;
        self.xr_stereo = None;
        self.xr = None;
        self.vr_overlay_draw = None;
        self.vr_hand_joints = [None, None];
        self.vr_menu_open = false;
        self.vr_wheel_open = false;
        tracing::info!("VR stopped; the editor keeps rendering flat");
    }

    /// View > VR > "Restart editor": quit through the normal path (the
    /// unsaved-changes dialog, the project save) and then start a fresh editor
    /// on this project (`relaunch_editor`).
    fn request_relaunch(&mut self) {
        self.relaunch_after_quit = true;
        if self.scene_dirty {
            self.show_quit_dialog = true;
        } else {
            self.quit_after_frame = true;
        }
    }

    /// Spawn `<this executable> <project root>` (the hub's own command line:
    /// cwd = the project, the environment inherited) and let it outlive this
    /// process. The hub only keeps the child it started to notice its exit, so
    /// the replacement is simply unknown to it: its "already open" check and
    /// exit-status notice stop for this project until it is opened from the hub
    /// again.
    fn relaunch_editor(&mut self) {
        let exe = match std::env::current_exe() {
            Ok(exe) => exe,
            Err(e) => {
                tracing::error!("restarting the editor: no executable path: {e}");
                return;
            }
        };
        let cmd = zestengine_assets::launcher::LaunchCommand::open_project(&exe, &self.project_root);
        match cmd.command().spawn() {
            Ok(_) => tracing::info!("restarting the editor: {}", cmd.display()),
            Err(e) => tracing::error!("restarting the editor ({}): {e}", cmd.display()),
        }
    }

    /// View > VR: the editor's own OpenXR switch (a user preference, not the
    /// project's "Start in VR"), what it does now and at the next start, and
    /// "Restart editor" when those differ. Disabled, saying why, while the
    /// machine-wide off switch is set.
    fn vr_menu_ui(&mut self, ui: &mut egui::Ui) {
        let running = self.xr_session.is_some();
        let globally_off = zestengine_xr::vr_disabled_globally();
        let env = std::env::var("ZESTENGINE_VR").ok();
        let env_explicit = env.as_deref().is_some_and(|v| {
            matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "on" | "yes" | "0" | "false" | "off" | "no")
        });
        let next = self.vr_wanted_next_start();
        let mut on = self.editor_prefs.vr.unwrap_or(self.project.vr);
        let disabled_why = globally_off.as_deref().map_or(String::new(), |why| {
            format!("VR is switched off for every ZestEngine process on this machine: {why}. Unset ZESTENGINE_VR or delete the marker file, then restart.")
        });
        if ui
            .add_enabled(globally_off.is_none(), egui::Checkbox::new(&mut on, "VR (OpenXR)"))
            .on_hover_text(
                "Run the editor in the headset. Saved per user (editor.ron), not in the project: \
                 Project Settings > \"Start in VR\" is only the default here and still decides for \
                 built games. Off ends the session now; on needs a restart (the Vulkan device is \
                 created against the OpenXR instance).",
            )
            .on_disabled_hover_text(disabled_why)
            .changed()
        {
            self.set_vr_preference(on);
        }
        let mut status = if let Some(why) = &globally_off {
            format!("off: {why}")
        } else if running && next {
            "running".to_string()
        } else if running {
            "running; stops after restart".to_string()
        } else if next {
            "starts after restart".to_string()
        } else {
            "off".to_string()
        };
        if env_explicit && let Some(v) = &env {
            status.push_str(&format!(" (ZESTENGINE_VR={} overrides the setting)", v.trim()));
        }
        ui.label(egui::RichText::new(status).small().weak());
        if self.editor_prefs.vr.is_some()
            && ui
                .small_button("Follow the project's \"Start in VR\"")
                .on_hover_text("Forget the editor's own choice; the project setting decides again at the next start")
                .clicked()
        {
            self.editor_prefs.vr = None;
            self.save_editor_prefs();
        }
        if ui
            .add_enabled(running != next, egui::Button::new("Restart editor"))
            .on_hover_text("Save, quit and open this project again so the VR setting takes effect")
            .on_disabled_hover_text("A restart would not change VR")
            .clicked()
        {
            self.request_relaunch();
            ui.close();
        }
    }

    /// Switch to `root`, first asking to save when the scene has unsaved edits.
    fn request_switch_project(&mut self, root: PathBuf) {
        if self.scene_dirty {
            self.pending_project_switch = Some(root);
        } else {
            self.switch_project(root);
        }
    }

    /// Open the editor window (deferred until the renderer + scene finished
    /// loading, so it never appears half-loaded). The splash closes on the first
    /// editor frame.
    fn open_editor(&mut self) {
        if let Some(s) = self.splash.as_mut() {
            s.set_progress(LOAD_STEPS, LOAD_STEPS);
            s.set_status("Opening editor…");
        }
        *self.load_status.lock().unwrap() = String::new();
        self.window = self.pending_window.take();
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    /// Start the editor load: create the window, then build the Vulkan renderer
    /// on a worker thread so the splash keeps animating during the (heavy) build.
    /// VR is the exception — its Vulkan must be created against the OpenXR
    /// instance, so that path builds synchronously. `about_to_wait` runs
    /// `finish_init` once the renderer and the plugin build are both ready.
    fn start_load(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        // Plugins build on a worker alongside the renderer; `finish_init` waits
        // for both (scene files may reference plugin components).
        if plugins::PluginHost::any_plugins(&self.project_root) {
            let (tx, rx) = std::sync::mpsc::channel();
            let root = self.project_root.clone();
            let status = self.plugin_status.clone();
            std::thread::spawn(move || {
                let report = |t: &str| *status.lock().unwrap() = t.to_owned();
                let _ = tx.send(plugins::PluginHost::build_all_with(&root, &report));
            });
            self.plugin_rx = Some(rx);
        }
        self.set_load_step(1, "Starting Vulkan and building pipelines…");
        // Phosphor icon font (file-browser type icons, etc.).
        crate::install_icon_font(&self.egui_ctx);
        apply_editor_style(&self.egui_ctx);

        let (icon_w, icon_h, icon_rgba) = icon::rgba();
        let title = format!("{} — {}", self.config.title, build_stamp());
        tracing::info!("starting {} (zestengine {})", self.config.title, build_stamp());
        let attrs = Window::default_attributes()
            .with_title(title)
            .with_window_icon(winit::window::Icon::from_rgba(icon_rgba, icon_w, icon_h).ok())
            .with_inner_size(winit::dpi::LogicalSize::new(
                self.config.width,
                self.config.height,
            ));
        // Wayland resolves icons via app_id → .desktop file; X11 via
        // WM_CLASS. Same name on both, plus a best-effort desktop-entry
        // install so the compositor can actually find the icon.
        #[cfg(target_os = "linux")]
        let attrs = {
            use winit::platform::{wayland, x11};
            let attrs = wayland::WindowAttributesExtWayland::with_name(attrs, "zestengine", "zestengine");
            x11::WindowAttributesExtX11::with_name(attrs, "zestengine", "zestengine")
        };
        #[cfg(target_os = "linux")]
        icon::install_desktop_entry();
        let window = Arc::new(event_loop.create_window(attrs)?);

        // Best-effort center on the primary monitor. X11 honors this; Wayland
        // leaves window placement to the compositor (this becomes a no-op there).
        if let Some(mon) = window
            .primary_monitor()
            .or_else(|| window.available_monitors().next())
        {
            let ms = mon.size();
            let ws = window.outer_size();
            if ms.width > ws.width && ms.height > ws.height {
                let origin = mon.position();
                let x = origin.x + ((ms.width - ws.width) / 2) as i32;
                let y = origin.y + ((ms.height - ws.height) / 2) as i32;
                window.set_outer_position(winit::dpi::PhysicalPosition::new(x, y));
            }
        }

        // OpenXR (None when no headset). VR needs Vulkan created against the XR
        // instance, so build synchronously and finish immediately. Only tried
        // when asked for (`vr_wanted_at_start`: ZESTENGINE_VR, else the
        // machine-wide off switch, else View > VR, else the project's "Start in
        // VR"): the loader prints its own errors when no runtime is running,
        // which is noise otherwise. The renderer's device is fixed at startup,
        // so opening VR from a flat editor means View > VR and a restart
        // (stopping it works live).
        let project_vr = zestengine_assets::load_project_file(&self.project_root).is_ok_and(|p| p.vr);
        let globally_off = zestengine_xr::vr_disabled_globally();
        let wanted = crate::editor_prefs::vr_wanted_at_start(
            std::env::var("ZESTENGINE_VR").ok().as_deref(),
            globally_off.is_some(),
            self.editor_prefs.vr,
            project_vr,
        );
        tracing::info!(
            "VR at start: {} (editor preference {:?}, project {project_vr}, machine-wide off switch {:?})",
            if wanted { "requested" } else { "off" },
            self.editor_prefs.vr,
            globally_off,
        );
        let xr = if wanted {
            match zestengine_xr::XrContext::start("zestengine editor") {
                Ok(Some(x)) => Some(x),
                Ok(None) => {
                    tracing::warn!("VR was requested but no OpenXR runtime or headset is available; starting flat");
                    None
                }
                Err(e) => {
                    tracing::warn!("OpenXR start failed, starting flat: {e:#}");
                    None
                }
            }
        } else {
            None
        };
        if let Some(xr) = xr {
            tracing::info!("VR ready: {}", xr.system_name());
            let renderer =
                Renderer::new_with_xr(window.clone(), Some((&xr.instance, xr.system)))?;
            self.ready_renderer = Some((renderer, Some(xr)));
            self.pending_window = Some(window);
            return Ok(());
        }

        // Flat path: build the renderer off-thread. `about_to_wait` polls the
        // channel and calls `finish_init` when it lands, ticking the splash
        // animation in the meantime.
        let (tx, rx) = std::sync::mpsc::channel();
        let win = window.clone();
        // Handles are read here, on the main thread (Windows forbids reading
        // them from the worker).
        let handles = zestengine_render::WindowHandles::of(&window)?;
        std::thread::spawn(move || {
            let _ = tx.send(Renderer::new_with_handles(win, handles, None));
        });
        self.renderer_rx = Some(rx);
        self.pending_window = Some(window);
        Ok(())
    }

    /// Finish the load once the renderer exists: plugins, project, scene, and the
    /// rest of GPU setup. Stays on the main thread — the scene holds non-Send
    /// `dyn Component` instances, so it can't be built off-thread.
    fn finish_init(
        &mut self,
        window: Arc<Window>,
        mut renderer: Renderer,
        xr: Option<zestengine_xr::XrContext>,
        plugins_built: Option<Result<Vec<String>>>,
    ) -> Result<()> {
        self.set_load_step(3, "Loading project…");
        // Plugins first: scene files may reference plugin components.
        let loaded = plugins_built.map(|built| {
            built.and_then(|names| {
                self.plugins
                    .load_built(&self.project_root, &names, &mut self.components)?;
                // Register each plugin's editor inspector/gizmo (the symbols are
                // type-erased in the plugins crate; the cast restores the ABI).
                let registrars = self.plugins.editor_registrars().to_vec();
                for p in registrars {
                    let f: fn(&mut crate::components::EditorComponents) = unsafe { std::mem::transmute(p) };
                    f(&mut self.editor_components);
                }
                Ok(())
            })
        });
        if let Some(Err(e)) = loaded {
            tracing::error!("building component plugins: {e:#}");
            // The worker's status line holds the first compiler error; keep it on the splash.
            let line = self.plugin_status.lock().unwrap().clone();
            if !line.is_empty() {
                self.set_load_status(&line);
            }
            self.plugin_build_error = Some(format!("{e:#}"));
        }
        self.plugin_fingerprint = self.plugins_fingerprint();

        // project.zestengine: restores the last scene and per-project settings;
        // created with defaults on first run.
        match zestengine_assets::load_project_file(&self.project_root) {
            Ok(project) => {
                self.project = project;
                self.project.apply_role_overrides(&mut self.components);
            }
            Err(_) => {
                self.project = zestengine_assets::ProjectFile {
                    name: self
                        .project_root
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|| "zestengine project".into()),
                    ..Default::default()
                };
                if let Err(e) = zestengine_assets::save_project_file(&self.project_root, &self.project)
                {
                    tracing::warn!("creating project.zestengine: {e:#}");
                }
            }
        }
        self.migrate_audio_settings();
        window.set_title(&format!("{} — zestengine", self.project.name));
        self.record_recent_project();
        // Load the project's input bindings (2C) into the live input manager.
        self.sync_input_bindings();
        renderer.set_vsync(self.project.settings.vsync);
        self.show_stats = self.project.settings.show_stats;
        self.show_stats_overlay = self.project.settings.show_stats_overlay;
        self.gizmo.snap = self.project.settings.snap;
        self.gizmo.grid_size = self.project.settings.grid_size;
        self.gizmo.align_to_surface = self.project.settings.align_to_surface;
        apply_accent(&self.egui_ctx, self.project.settings.accent);

        self.set_load_step(4, "Loading scene…");
        // Decide the scene to load. A `.scene` is parsed on a worker thread (the
        // editor opens only once it's fully loaded, never half-loaded). Cheap/rare
        // cases (a single CLI model, the builtin test scene) load synchronously
        // here using the local `renderer`.
        let mut bg_scene: Option<PathBuf> = None;
        match self.config.scene_path.clone() {
            Some(path) if path.ends_with(".scene") => bg_scene = Some(PathBuf::from(path)),
            Some(path) => {
                let asset = zestengine_assets::load_model_with_meta(&path)?;
                self.scene
                    .add_asset_scene(&mut renderer, &asset, Some(Path::new(&path)))?;
            }
            None => {
                let last = self
                    .project
                    .last_scene
                    .clone()
                    .map(|rel| self.project_root.join(rel))
                    .filter(|abs| abs.exists());
                match last {
                    Some(abs) => bg_scene = Some(abs),
                    None => {
                        let asset = zestengine_assets::test_scene();
                        self.scene.add_asset_scene(&mut renderer, &asset, None)?;
                    }
                }
            }
        }
        self.egui_state = Some(egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            None,
            None,
            None,
        ));
        self.renderer = Some(renderer);
        // Hold the window closed (in `pending_window`) until the scene is loaded.
        self.pending_window = Some(window);
        // Now that the renderer's Vulkan device was created (via OpenXR when VR is
        // present), create the session that shares it.
        if let Some(xr) = xr {
            let (vi, pd, dev, qfi) = self.renderer.as_ref().unwrap().vulkan_raw_handles();
            match xr.create_session(vi, pd, dev, qfi) {
                Ok(session) => self.xr_session = Some(session),
                Err(e) => tracing::warn!("OpenXR session create failed: {e:#}"),
            }
            self.xr = Some(xr);
        }
        match bg_scene {
            // Parse on a worker; `apply_loaded_scene` uploads + opens the editor
            // when it lands (polled in `about_to_wait` while the splash shows).
            Some(path) => self.spawn_scene_load(path, false),
            // Synchronously-loaded model/test scene: finalize + open now.
            None => {
                self.after_scene_loaded();
                self.open_editor();
            }
        }
        Ok(())
    }

    /// Parse a `.scene` (file + its models) on a worker thread; the GPU upload
    /// happens in `apply_loaded_scene` on the main thread when it lands. `blocking`
    /// shows the in-editor modal (use it for runtime loads, not startup).
    fn spawn_scene_load(&mut self, path: PathBuf, blocking: bool) {
        let root = self.project_root.clone();
        let status = self.load_status.clone();
        // Loader thread uploads meshes/textures on the transfer queue (when
        // available), so the main thread only installs handles + builds — keeping
        // the UI/splash responsive. None → main-thread upload fallback.
        let uploader = self.renderer.as_ref().and_then(|r| r.uploader());
        let label = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "scene".into());
        self.tasks.spawn(
            format!("Loading {label}"),
            crate::tasks::TaskKind::LoadScene,
            blocking,
            move |cancel, _progress| {
                if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                    return Ok(crate::tasks::TaskPayload::None);
                }
                let set_status = |label: &str| *status.lock().unwrap() = label.to_string();
                let data = zestengine_engine::scene_loader::load_on_worker(&path, &root, uploader.as_ref(), &[], &set_status)?;
                // Editor only: fill each model's `.meta` id table (no re-import; the
                // models are already loaded) so scene references survive renames.
                for (rel, model) in &data.models {
                    if let Err(e) = zestengine_assets::record_model_ids(root.join(rel), model) {
                        tracing::warn!("recording ids for {rel}: {e:#}");
                    }
                }
                Ok(crate::tasks::TaskPayload::Scene(Box::new(data)))
            },
        );
    }

    /// Main-thread install of a worker-loaded scene. Only handle installs and
    /// material setup happen here; skybox + baked lighting follow over the next
    /// frames (`step_post_load`) so the editor keeps drawing. Opens the editor if
    /// this was the startup load.
    fn apply_loaded_scene(&mut self, data: zestengine_engine::scene_loader::LoadedSceneData) {
        zestengine_render::hitch::event("scene_load_finish", "", 0);
        let _h = zestengine_render::hitch::scope("apply_loaded_scene");
        if self.window.is_none() {
            self.set_load_step(5, "Installing scene…");
        } else {
            self.set_load_status("Installing scene…");
        }
        let path = data.path.clone();
        let editor_camera = data.file.editor_camera;
        let collapsed = data.file.collapsed.clone();
        let (view_hidden, view_locked) = (data.file.scene_hidden.clone(), data.file.scene_locked.clone());
        let mut splash = self.splash.take();
        let status = self.load_status.clone();
        let mut last = Instant::now();
        // Replacing an open scene: free its GPU resources first.
        if self.window.is_some() {
            if let Some(r) = self.renderer.as_mut()
                && let Err(e) = r.reset_scene()
            {
                tracing::error!("resetting scene: {e:#}");
            }
            self.reset_render_caches();
        }
        let installed = match self.renderer.as_mut() {
            Some(renderer) => zestengine_engine::scene_loader::install(
                data,
                Some(renderer),
                &self.components,
                &mut self.shaders,
                &self.project_root,
                |label: &str| {
                    *status.lock().unwrap() = label.to_string();
                    if let Some(s) = splash.as_mut()
                        && last.elapsed().as_millis() >= 16
                    {
                        let _ = s.set_status(label);
                        last = Instant::now();
                    }
                },
            ),
            None => {
                self.splash = splash;
                return;
            }
        };
        self.splash = splash;
        match installed {
            Ok((scene, post)) => {
                self.scene = scene;
                self.current_scene_path = Some(path);
                if let Some(c) = editor_camera {
                    self.camera.position = Vec3::from(c.position);
                    self.camera.yaw = c.yaw;
                    self.camera.pitch = c.pitch;
                }
                self.restore_collapsed(&collapsed);
                self.scene.set_view_ids(&view_hidden, &view_locked);
                self.selection = Selection::None;
                self.file_material = None;
                // A scene that was saved with the built-in sun now holds a
                // Directional Light object: unsaved until written back.
                self.scene_dirty = self.scene.legacy_sun_migrated;
                // Indices into the old scene are invalid.
                self.undo_stack.clear();
                // A loaded scene has no open prefab (it went with the old scene).
                self.prefab_editor.stage_closed_by_load();
                self.post_load = Some(post);
                if self.window.is_some() {
                    self.save_project();
                }
            }
            Err(e) => {
                tracing::error!("scene load: {e:#}");
                self.tasks
                    .notify(format!("Scene load failed: {e}"), crate::tasks::NotifyLevel::Warn);
            }
        }
        // Startup load: the scene is in, open the editor (post-load continues there).
        if self.window.is_none() {
            self.open_editor();
        }
    }

    /// Drop editor caches that hold renderer handles (after `reset_scene`).
    fn reset_render_caches(&mut self) {
        self.world_ui = zestengine_engine::game_ui::WorldUi::default();
        self.body.reset_render();
        self.body.scene_changed();
        self.particles = zestengine_engine::particles::ParticleRenderer::default();
        self.water = zestengine_engine::water_render::WaterRenderer::default();
        self.water_fx = zestengine_engine::water_fx::WaterFx::default();
        self.carry_fx = zestengine_engine::water_carry::CarryFx::default();
        self.nav_preview.clear();
    }

    /// One post-load step per frame: skybox, then baked lighting, then
    /// reflections.
    fn step_post_load(&mut self) {
        let _h = zestengine_render::hitch::scope("step_post_load");
        let Some(post) = self.post_load.as_mut() else {
            return;
        };
        if let Some(sky) = post.skybox.take() {
            if let Some(r) = self.renderer.as_mut() {
                sky.install(r);
            }
            return;
        }
        if let Some(baked) = post.baked.take() {
            self.scene.baked = baked;
            self.upload_baked_probes();
            return;
        }
        let faces = std::mem::take(&mut post.reflection);
        self.post_load = None;
        self.setup_reflections_with(Some(faces));
        self.tasks.notify("Scene loaded", crate::tasks::NotifyLevel::Info);
    }

    /// Post-load setup shared by sync and background scene loads: skybox, baked
    /// lighting upload, reflection env. Each `set_load_status` also redraws the
    /// splash, so the phase text reflects the step that is actually running (and
    /// is the truth, not a fixed label) rather than a single "Setting up lighting"
    /// covering everything.
    fn after_scene_loaded(&mut self) {
        self.set_load_status("Loading skybox…");
        let t0 = Instant::now();
        self.apply_skybox();
        tracing::info!("after_scene_loaded: skybox total {:?}", t0.elapsed());
        let t1 = Instant::now();

        // Baked lighting AFTER the renderer is stored, so `upload_baked_probes`
        // can push lightmaps/probes to the GPU (else a baked scene loads black).
        // Only announce the GPU upload when there is actually baked data on disk;
        // otherwise this is a no-op and a "uploading lighting" label would lie.
        let base = self.bake_base_path();
        self.scene.load_bake_sidecars(&base);
        if self
            .scene
            .baked
            .as_ref()
            .is_some_and(|b| !b.lightmaps.is_empty() || !b.probe_sh.is_empty())
        {
            self.set_load_status("Uploading baked lighting…");
        }
        self.upload_baked_probes();
        tracing::info!("after_scene_loaded: bake {:?}", t1.elapsed());
        self.setup_reflections();
    }

    /// Reflection env after a load: a Baked-mode probe loads its sidecar, a
    /// Realtime probe captures live.
    fn setup_reflections(&mut self) {
        self.setup_reflections_with(None);
    }

    /// [`setup_reflections`] with the baked cubes already read by the scene
    /// loader (None reads the sidecar here). Baked probes upload their cubes and
    /// realtime ones capture live, from `refl_slots.update` each frame.
    fn setup_reflections_with(&mut self, set: Option<zestengine_engine::refl_slots::ReflProbeSet>) {
        let set = set.unwrap_or_else(|| zestengine_engine::refl_slots::read_reflprobes(&self.bake_base_path().with_extension("reflprobe")));
        if self.scene.active_reflection_probe(glam::Vec3::ZERO).is_some() {
            self.set_load_status("Loading reflection probes…");
        }
        self.refl_slots.set_baked(set);
        self.bake_sizes = None;
    }

    fn set_looking(&mut self, looking: bool) {
        if self.looking == looking {
            return;
        }
        self.looking = looking;
        if let Some(window) = &self.window {
            window.set_cursor_visible(!looking);
            let grab = if looking {
                window
                    .set_cursor_grab(CursorGrabMode::Locked)
                    .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
            } else {
                window.set_cursor_grab(CursorGrabMode::None)
            };
            if let Err(e) = grab {
                tracing::debug!("cursor grab: {e}");
            }
        }
        if !looking {
            self.look_delta = (0.0, 0.0);
            // egui's pointer froze during look; resync it next frame.
            self.look_just_ended = true;
        }
    }

    /// The pointer is over an egui popup, menu, window or tooltip (a floating
    /// layer) that is not the one hosting the viewport / game view. Raw mouse
    /// presses over those belong to egui and must not reach the camera or the
    /// game. egui's own `consumed` flag is not usable here: it is also true
    /// over the viewport's own interact widget.
    fn pointer_over_overlay(&self) -> bool {
        let Some((x, y)) = self.last_cursor else {
            return false;
        };
        let ppp = self.egui_ctx.pixels_per_point();
        let Some(top) = self.egui_ctx.layer_id_at(egui::pos2(x as f32 / ppp, y as f32 / ppp)) else {
            return false;
        };
        let hosts_view: Option<egui::LayerId> = self.egui_ctx.data(|d| d.get_temp(egui::Id::new(VIEW_LAYER_KEY)));
        top.order != egui::Order::Background && Some(top) != hosts_view
    }

    fn cursor_in_viewport(&self) -> bool {
        let Some((x, y)) = self.last_cursor else {
            return false;
        };
        let ppp = self.egui_ctx.pixels_per_point();
        self.viewport_rect
            .contains(egui::pos2(x as f32 / ppp, y as f32 / ppp))
    }

    fn update_camera(&mut self, dt: f32) {
        let (dx, dy) = std::mem::take(&mut self.look_delta);
        if self.looking {
            self.camera.look(dx as f32, dy as f32);
            let key = |code: KeyCode| self.keys.contains(&code) as i32 as f32;
            let local = Vec3::new(
                key(KeyCode::KeyD) - key(KeyCode::KeyA),
                key(KeyCode::KeyE) - key(KeyCode::KeyQ),
                key(KeyCode::KeyW) - key(KeyCode::KeyS),
            );
            if local != Vec3::ZERO {
                let fast = self.keys.contains(&KeyCode::ShiftLeft)
                    || self.keys.contains(&KeyCode::ShiftRight);
                self.camera.fly(local, dt, fast);
            }
        }
    }

    /// Cursor position (egui points) → world ray. NDC is relative to the viewport
    /// tab rect, since the scene renders into a texture filling that rect (RTT).
    fn cursor_ray(&self, pos: egui::Pos2) -> Option<(Vec3, Vec3)> {
        cursor_ray_for(&self.camera, self.viewport_rect, pos)
    }

    /// The game camera's (view, proj, position) while playing with
    /// "Game camera" on; None means use the editor fly camera.
    fn game_view(&self, aspect: f32) -> Option<(Mat4, Mat4, Vec3)> {
        if !(self.playing && self.play_game_camera) {
            return None;
        }
        self.scene.main_camera_view_proj(aspect)
    }

    /// Position occlusion culling is evaluated from this frame: the game camera
    /// when previewing it (or when the Scene view follows it in play), else the
    /// Scene camera. None turns culling off.
    fn occlusion_camera(&self) -> Option<Vec3> {
        if !self.occlusion_ui.cull_scene || self.scene.occlusion.is_none() {
            return None;
        }
        let game = self.occlusion_ui.preview_game || (self.playing && self.play_game_camera);
        if game {
            self.scene.main_camera_view_proj(1.0).map(|(_, _, p)| p)
        } else {
            Some(self.camera.position)
        }
    }

    /// Where the game is shown while playing: the Camera tab when it's on
    /// screen, else the Scene view when it follows the game camera. None means
    /// the game isn't visible (the Scene view is free-flying).
    fn game_rect(&self) -> Option<egui::Rect> {
        if !self.playing {
            return None;
        }
        if self.camera_tab_visible && self.camera_rect.is_positive() {
            Some(self.camera_rect)
        } else if self.play_game_camera && self.viewport_visible {
            Some(self.viewport_rect)
        } else {
            None
        }
    }

    fn cursor_in_game(&self) -> bool {
        let (Some((x, y)), Some(rect)) = (self.last_cursor, self.game_rect()) else {
            return false;
        };
        let ppp = self.egui_ctx.pixels_per_point();
        rect.contains(egui::pos2(x as f32 / ppp, y as f32 / ppp))
    }

    /// Game view size / pointer / camera for gameplay (`ctx.screen_ray`, UI).
    fn game_screen_info(&mut self) -> zestengine_core::ScreenInfo {
        let ppp = self.egui_ctx.pixels_per_point();
        let in_camera_tab = self.playing && self.camera_tab_visible && self.camera_rect.is_positive();
        let rect = self.game_rect().unwrap_or(self.viewport_rect);
        if !rect.is_finite() || rect.width() < 1.0 || rect.height() < 1.0 {
            return zestengine_core::ScreenInfo { controls: self.input.controls_info(), ..Default::default() };
        }
        let size = glam::Vec2::new(rect.width() * ppp, rect.height() * ppp);
        let aspect = rect.width() / rect.height();
        let game_cam = if in_camera_tab { self.scene.main_camera_view_proj(aspect) } else { self.game_view(aspect) };
        let (view, proj) = match game_cam {
            Some((v, p, _)) => (v, p),
            None => (self.camera.view(), self.camera.proj(aspect)),
        };
        let pointer = self
            .last_cursor
            .map(|(x, y)| glam::Vec2::new(x as f32 - rect.left() * ppp, y as f32 - rect.top() * ppp))
            .filter(|p| p.x >= 0.0 && p.y >= 0.0 && p.x < size.x && p.y < size.y);
        zestengine_core::ScreenInfo {
            size,
            pointer,
            view,
            proj,
            pointer_over_ui: self.game_ui_frame.pointer_over_ui,
            ui_clicks: std::mem::take(&mut self.game_ui_frame.clicks),
            ui_submits: std::mem::take(&mut self.game_ui_frame.submits),
            ui_text_editing: self.game_ui_frame.text_editing,
            ui_cancel: std::mem::take(&mut self.game_ui_frame.cancel),
            vr: zestengine_engine::runtime::vr_frame(self.xr_session.as_ref(), &self.vr_targets),
            controls: self.input.controls_info(),
            body: self.body.info(),
        }
    }

    /// Create a UI preset. Non-canvas presets go under the selected UI object's
    /// canvas tree (or the selected UI object itself), else the first canvas,
    /// creating a canvas when the scene has none.
    fn spawn_ui(&mut self, preset: crate::UiPreset) -> anyhow::Result<()> {
        use zestengine_core::{
            Component, UiAlign, UiButton, UiCanvas, UiImage, UiPanel, UiProgressBar, UiRect, UiRenderMode, UiSlider,
            UiText, UiTextField, UiToggle,
        };
        use crate::UiPreset;
        let renderer = self.renderer.as_mut().ok_or_else(|| anyhow::anyhow!("no renderer"))?;
        let has = |scene: &scene::LoadedScene, i: usize, f: &dyn Fn(&dyn Component) -> bool| {
            scene.objects[i].components.iter().any(|c| f(c.as_ref()))
        };
        let is_canvas = |c: &dyn Component| c.as_any().is::<UiCanvas>();
        let is_ui = |c: &dyn Component| c.type_name().starts_with("UI ");
        let mut spawn = |scene: &mut scene::LoadedScene,
                         name: &str,
                         parent: Option<usize>,
                         comps: Vec<Box<dyn Component>>|
         -> anyhow::Result<usize> {
            let i = scene.spawn(renderer, zestengine_assets::ObjectSource::Empty, name.to_owned(), Vec3::ZERO)?;
            scene.objects[i].parent = parent;
            scene.objects[i].components = comps;
            Ok(i)
        };
        if preset == UiPreset::Canvas {
            let i = spawn(&mut self.scene, "Canvas", None, vec![Box::new(UiCanvas::default())])?;
            self.select_object(Some(i));
            return Ok(());
        }
        if preset == UiPreset::WorldCanvas {
            let pos = self.camera.position + self.camera.forward() * 2.0;
            let i = spawn(
                &mut self.scene,
                "World Canvas",
                None,
                vec![
                    Box::new(UiCanvas {
                        render_mode: UiRenderMode::World,
                        ..Default::default()
                    }),
                    Box::new(UiPanel::default()),
                ],
            )?;
            self.scene.objects[i].translation = pos;
            // Face the camera (+Z toward it).
            let to_cam = (self.camera.position - pos).with_y(0.0).normalize_or(Vec3::Z);
            self.scene.objects[i].rotation = glam::Quat::from_rotation_arc(Vec3::Z, to_cam);
            self.select_object(Some(i));
            return Ok(());
        }
        let selected_ui = match self.selection {
            Selection::Object(i) if i < self.scene.objects.len() && has(&self.scene, i, &is_ui) => Some(i),
            _ => None,
        };
        let parent = match selected_ui {
            Some(i) => i,
            None => match (0..self.scene.objects.len()).find(|&i| has(&self.scene, i, &is_canvas)) {
                Some(c) => c,
                None => spawn(&mut self.scene, "Canvas", None, vec![Box::new(UiCanvas::default())])?,
            },
        };
        let rect = |w: f32, h: f32| UiRect {
            size: [w, h],
            ..Default::default()
        };
        let fill = UiRect {
            anchor_min: [0.0, 0.0],
            anchor_max: [1.0, 1.0],
            size: [0.0, 0.0],
            ..Default::default()
        };
        let root = match preset {
            UiPreset::Canvas | UiPreset::WorldCanvas => unreachable!(),
            UiPreset::TextField => spawn(
                &mut self.scene,
                "Text Field",
                Some(parent),
                vec![Box::new(rect(360.0, 56.0)), Box::new(UiTextField::default())],
            )?,
            UiPreset::Panel => spawn(
                &mut self.scene,
                "Panel",
                Some(parent),
                vec![Box::new(rect(400.0, 300.0)), Box::new(UiPanel::default())],
            )?,
            UiPreset::Text => spawn(
                &mut self.scene,
                "Text",
                Some(parent),
                vec![Box::new(rect(300.0, 60.0)), Box::new(UiText::default())],
            )?,
            UiPreset::Image => spawn(
                &mut self.scene,
                "Image",
                Some(parent),
                vec![Box::new(rect(128.0, 128.0)), Box::new(UiImage::default())],
            )?,
            UiPreset::Slider => spawn(
                &mut self.scene,
                "Slider",
                Some(parent),
                vec![Box::new(rect(300.0, 32.0)), Box::new(UiSlider::default())],
            )?,
            UiPreset::ProgressBar => spawn(
                &mut self.scene,
                "Progress Bar",
                Some(parent),
                vec![Box::new(rect(360.0, 24.0)), Box::new(UiProgressBar::default())],
            )?,
            UiPreset::Button => {
                let b = spawn(
                    &mut self.scene,
                    "Button",
                    Some(parent),
                    vec![
                        Box::new(rect(240.0, 64.0)),
                        Box::new(UiPanel::default()),
                        Box::new(UiButton::default()),
                    ],
                )?;
                spawn(
                    &mut self.scene,
                    "Label",
                    Some(b),
                    vec![
                        Box::new(fill),
                        Box::new(UiText {
                            text: "Button".into(),
                            size: 28.0,
                            ..Default::default()
                        }),
                    ],
                )?;
                b
            }
            UiPreset::Toggle => {
                let t = spawn(
                    &mut self.scene,
                    "Toggle",
                    Some(parent),
                    vec![Box::new(rect(260.0, 40.0)), Box::new(UiToggle::default())],
                )?;
                spawn(
                    &mut self.scene,
                    "Label",
                    Some(t),
                    vec![
                        Box::new(UiRect {
                            position: [24.0, 0.0],
                            size: [-52.0, 0.0],
                            ..fill
                        }),
                        Box::new(UiText {
                            text: "Toggle".into(),
                            size: 26.0,
                            align_x: UiAlign::Start,
                            ..Default::default()
                        }),
                    ],
                )?;
                t
            }
        };
        self.select_object(Some(root));
        Ok(())
    }

    fn spawn_ui_template(&mut self, template: zestengine_engine::ui_templates::UiTemplate) -> anyhow::Result<()> {
        let renderer = self.renderer.as_mut().ok_or_else(|| anyhow::anyhow!("no renderer"))?;
        let mut indices: Vec<usize> = Vec::new();
        for node in template.build() {
            let i = self.scene.spawn(renderer, zestengine_assets::ObjectSource::Empty, node.name, Vec3::ZERO)?;
            let object = &mut self.scene.objects[i];
            object.parent = node.parent.map(|p| indices[p]);
            object.enabled = node.active;
            object.components = node.components;
            indices.push(i);
        }
        self.select_object(indices.first().copied());
        self.set_status(format!("Added {} UI", template.label()), false);
        Ok(())
    }

    /// 2D <-> 3D. A canvas going 3D is placed in front of the editor camera,
    /// facing it, with the reference aspect so the layout carries over.
    fn set_canvas_mode(&mut self, index: usize, mode: zestengine_core::UiRenderMode) {
        let Some(object) = self.scene.objects.get_mut(index) else {
            return;
        };
        let Some(canvas) = object
            .components
            .iter_mut()
            .find_map(|c| c.as_any_mut().downcast_mut::<zestengine_core::UiCanvas>())
        else {
            return;
        };
        canvas.render_mode = mode;
        if mode == zestengine_core::UiRenderMode::World {
            let [rw, rh] = canvas.reference_size;
            canvas.world_size = [1.6, 1.6 * rh.max(1.0) / rw.max(1.0)];
            let pos = self.camera.position + self.camera.forward() * 2.5;
            let to_cam = (self.camera.position - pos).with_y(0.0).normalize_or(Vec3::Z);
            object.translation = pos;
            object.rotation = glam::Quat::from_rotation_arc(Vec3::Z, to_cam);
            object.scale = Vec3::ONE;
        }
    }

    fn start_zestsurface(&mut self, index: usize) {
        match crate::zestsurface::export_object(&self.scene, &self.project_root, index) {
            Ok(Some(model)) => {
                let manifest =
                    crate::zestsurface::manifest_path(&self.project_root, &self.scene.objects[index].id);
                match crate::zestsurface::launch(&model, &manifest) {
                    Ok(_child) => {
                        self.zestsurface_object = Some(self.scene.objects[index].id);
                        tracing::info!("opened {} in ZestPaint", self.scene.objects[index].name);
                    }
                    Err(e) => tracing::error!("launching ZestPaint: {e:#}"),
                }
            }
            Ok(None) => tracing::warn!("object has no paintable mesh for ZestPaint"),
            Err(e) => tracing::error!("exporting to ZestPaint: {e:#}"),
        }
    }

    fn poll_zestsurface(&mut self) {
        let Some(id) = self.zestsurface_object else { return };
        let Some(ret) = crate::zestsurface::poll_return(&self.project_root, &id) else { return };
        self.zestsurface_object = None;
        let Self { renderer, scene, shaders, project_root, .. } = self;
        let Some(r) = renderer.as_mut() else { return };
        match crate::zestsurface::apply_return(scene, r, shaders, project_root.as_path(), &ret) {
            Ok(n) => tracing::info!("applied {n} textures from ZestPaint"),
            Err(e) => tracing::error!("applying ZestPaint textures: {e:#}"),
        }
    }

    fn update_particles(&mut self, dt: f32) {
        let rect = self.viewport_rect;
        let aspect = rect.width() / rect.height().max(1.0);
        let (view, proj) = match self.game_view(aspect) {
            Some((v, p, _)) => (v, p),
            None => (self.camera.view(), self.camera.proj(aspect)),
        };
        let dt = if self.playing && self.play_paused { 0.0 } else { dt };
        if let Some(renderer) = self.renderer.as_mut() {
            self.particles
                .update(renderer, &mut self.scene, &self.project_root, view.inverse(), dt);
            if let Some(s) = self.play_session.as_mut() {
                s.set_shared_clock(self.net.as_ref().and_then(|n| n.shared_clock()));
            }
            let play_time = self.play_session.as_ref().map(|s| s.water_time());
            fit_water_requests(&mut self.scene, &mut self.scene_dirty);
            // Not playing: bodies dragged in the editor still push the water.
            if !self.playing {
                self.editor_water.feed(&mut self.scene, dt);
            } else {
                self.editor_water.reset();
            }
            self.water.update(renderer, &mut self.scene, view.inverse().w_axis.truncate(), play_time, dt);
            let net = self.net.as_ref();
            let owns = |id: zestengine_core::ObjectId| net.is_none_or(|n| !n.is_connected() || n.has_authority_raw(id.raw()));
            self.scene.update_wave_riders(self.water.time(), dt, self.playing, &owns);
            self.water_fx.update(renderer, &mut self.scene, view.inverse(), play_time, dt);
            self.carry_fx.update(renderer, &mut self.scene, view.inverse(), play_time.is_some(), dt, &mut self.water_fx.sim.particles);
            self.scene.sync_terrain_meshes(renderer, &mut self.shaders, &self.project_root);
            self.scene.prepare_lods(renderer, &self.project_root);
            self.scene.update_auto_lods(renderer, &self.project_root);
            let mut cluster = renderer.cluster_settings();
            // The triangle budget aims for the display refresh / frame limiter cap.
            let target_ms = if cluster.target_from_env { cluster.target_frame_ms } else { self.pacing.target_hz().map_or(0.0, |h| (1000.0 / h) as f32) };
            let stream = zestengine_engine::runtime::apply_stream_setting(&mut cluster, (self.project.settings.cluster_stream, self.project.settings.cluster_stream_mb));
            if stream || cluster.mode != self.project.settings.cluster_rendering || (cluster.target_frame_ms - target_ms).abs() > 0.05 {
                cluster.mode = self.project.settings.cluster_rendering;
                cluster.target_frame_ms = target_ms;
                renderer.set_cluster_settings(cluster);
            }
            self.scene.update_cluster_dags(renderer, &self.project_root);
        }
        // The DAG build shows in the task list with a working Cancel (never waits on it).
        let building = self.scene.cluster_build_progress();
        self.cluster_task.sync(&mut self.tasks, building.as_ref().map(|(n, p)| (n.as_str(), p)));
        self.scene.set_lod_projection(&proj);
        self.scene.set_cull_view(&(proj * view));
        self.scene.lod_camera = Some(view.inverse().w_axis.truncate());
        self.update_nav_preview();
    }

    /// Editor navmesh overlay: baked from a temporary physics world when a
    /// NavMesh Surface asks for it (Inspector "Show navmesh" / "Rebake").
    fn update_nav_preview(&mut self) {
        let mut show = false;
        let mut rebake = false;
        for o in &mut self.scene.objects {
            if let Some(s) = o
                .components
                .iter_mut()
                .find_map(|c| c.as_any_mut().downcast_mut::<zestengine_core::NavMeshSurface>())
            {
                show |= s.show_preview;
                rebake |= std::mem::take(&mut s.rebake_preview);
            }
        }
        if !show {
            self.nav_preview.clear();
            return;
        }
        if self.nav_preview.is_empty() || rebake {
            self.nav_preview.clear();
            let mut pw = zestengine_engine::physics::PhysicsWorld::build(&self.scene);
            pw.step();
            let mut nav = zestengine_engine::navmesh::NavSystem::default();
            nav.bake(&self.scene, &pw);
            let Some(renderer) = self.renderer.as_mut() else {
                return;
            };
            for grid in nav.grids() {
                let data = zestengine_engine::navmesh::preview_mesh(grid, [0.2, 0.6, 1.0, 0.45]);
                if data.vertices.is_empty() {
                    continue;
                }
                let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
                for v in &data.vertices {
                    lo = lo.min(Vec3::from(v.position));
                    hi = hi.max(Vec3::from(v.position));
                }
                match (
                    renderer.upload_mesh(&data),
                    zestengine_engine::game_ui::unlit_material(renderer, None, "navmesh preview"),
                ) {
                    (Ok(m), Ok(mat)) => self.nav_preview.push((m, mat, (lo + hi) * 0.5, ((hi - lo) * 0.5).length())),
                    (Err(e), _) | (_, Err(e)) => tracing::error!("navmesh preview: {e:#}"),
                }
            }
        }
        for &(mesh, material, center, radius) in &self.nav_preview {
            self.scene.extra_draws.push(zestengine_render::DrawCmd {
                gi_source: None,
                mesh,
                material,
                transform: Mat4::IDENTITY,
                highlight: 0.0,
                mesh_center: center,
                bound_radius: radius,
                lightmap_layer: -1,
                lightmap_size: 0,
                static_geometry: false,
                planar_index: -1,
                occluded: false,
                shadow_casting: zestengine_render::ShadowCasting::On,
                instance_color: None,
                instance_param: None,
            });
        }
    }

    /// World-space canvases: drawn always (layout preview while editing),
    /// interactive while playing via the viewport mouse ray and the right VR
    /// controller.
    fn update_world_ui(&mut self) {
        self.scene.extra_draws.clear();
        let mut rays = Vec::new();
        let (mouse_down, vr_down) = (
            self.egui_ctx.input(|i| i.pointer.primary_down()),
            self.vr_prev_trigger,
        );
        let (prev_mouse, prev_vr) = self.world_ui_prev;
        self.world_ui_prev = (mouse_down, vr_down);
        if self.playing {
            if let Some((rp, rr)) = self.vr_targets.right_hand
                && self.xr_session.is_some()
            {
                let (wp, wr) = self.vr_rig.stage_pose_to_world((rp, rr));
                let r = zestengine_core::pointer_ray(wp, wr);
                rays.push(zestengine_engine::game_ui::UiRay {
                    origin: r.origin,
                    dir: r.dir,
                    down: vr_down,
                    pressed: vr_down && !prev_vr,
                    released: !vr_down && prev_vr,
                });
            }
            let screen = {
                let ppp = self.egui_ctx.pixels_per_point();
                let rect = self.viewport_rect;
                let aspect = rect.width() / rect.height().max(1.0);
                let (view, proj) = match self.game_view(aspect) {
                    Some((v, p, _)) => (v, p),
                    None => (self.camera.view(), self.camera.proj(aspect)),
                };
                let pointer = self
                    .last_cursor
                    .map(|(x, y)| glam::Vec2::new(x as f32 / ppp, y as f32 / ppp))
                    .filter(|p| rect.contains(egui::pos2(p.x, p.y)))
                    .map(|p| glam::Vec2::new((p.x - rect.left()) / rect.width(), (p.y - rect.top()) / rect.height()));
                (view, proj, pointer)
            };
            // Flat screen: the mouse ray, or the view centre while mouse-look has the cursor.
            let (view, proj, uv) = screen;
            let button = [mouse_down, mouse_down && !prev_mouse, !mouse_down && prev_mouse];
            rays.extend(zestengine_engine::game_ui::desktop_ui_ray(view, proj, uv, self.game_cursor_locked, button));
        }
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        let (draws, frame) = self.world_ui.update(renderer, &mut self.scene, &self.project_root, &rays, None);
        self.world_ui_draws = draws;
        self.game_ui_frame.pointer_over_ui |= frame.pointer_over_ui;
        self.game_ui_frame.clicks.extend(frame.clicks);
        self.game_ui_frame.submits.extend(frame.submits);
    }

    /// Where a dropped / placed object lands along a ray: the first surface
    /// hit, else the ground plane (y = 0), else 5 m ahead.
    fn drop_point(&self, origin: Vec3, dir: Vec3) -> Vec3 {
        if let Some(t) = self.scene.ray_hit(origin, dir) {
            return origin + dir * t;
        }
        if dir.y < -1e-3 {
            let t = -origin.y / dir.y;
            if t > 0.0 && t < 500.0 {
                return origin + dir * t;
            }
        }
        origin + dir * 5.0
    }

    /// Hierarchy "Add Toggle": a Menu Toggle on `index` that shows or hides it.
    fn add_avatar_toggle(&mut self, index: usize) {
        match crate::avatar_edit::add_toggle(&mut self.scene, index) {
            Ok(()) => {
                self.select_object(Some(index));
                self.scene_dirty = true;
                self.set_status("Added a menu toggle; it appears in the avatar's menu when the avatar is built", false);
            }
            Err(msg) => self.set_status(msg, false),
        }
    }

    fn select_object(&mut self, index: Option<usize>) {
        self.selection = match index {
            Some(i) => Selection::Object(i),
            None => Selection::None,
        };
        // Any single-source selection (viewport pick, spawn, hierarchy single
        // click) collapses the multi-selection to just this object.
        self.multi_objects = index.into_iter().collect();
    }

    /// The effective multi-selection of scene objects. Falls back to just the
    /// anchor when the stored set is stale (an external source changed the
    /// anchor without going through the scene panel).
    fn selected_object_indices(&self) -> Vec<usize> {
        match self.selection {
            Selection::Object(i) => {
                if self.multi_objects.len() > 1 && self.multi_objects.contains(&i) {
                    // The primary selection first (the outline draws it strongest).
                    let mut all = vec![i];
                    all.extend(self.multi_objects.iter().copied().filter(|&o| o != i));
                    all
                } else {
                    vec![i]
                }
            }
            _ => Vec::new(),
        }
    }

    /// Build the VR menu's floating panels: the editor UI split into the screen
    /// regions AROUND the viewport (left/right/top/bottom strips, viewport omitted),
    /// each as a 3D quad in a world-anchored arc that scales with the player. Returns
    /// the panels (parallel to the screen regions they sample, for pointer mapping).
/// Fixed screen rect of the VR palette wheel Area (bottom-right corner of the
/// UI texture). Shared by the egui pass that draws it and `update_vr` that
/// hit-tests / UV-maps it.
fn vr_wheel_rect(ctx: &egui::Context) -> egui::Rect {
    let screen = ctx.content_rect();
    let (w, h) = (300.0, 240.0);
    egui::Rect::from_min_size(
        egui::pos2(screen.max.x - w - 16.0, screen.max.y - h - 16.0),
        egui::vec2(w, h),
    )
}

/// Fixed screen rect of the VR perf watch Area (top-left corner of the UI
/// texture). Shared by the egui pass that draws it and `update_vr` that
/// UV-maps it onto the left wrist panel.
fn vr_watch_rect(ctx: &egui::Context) -> egui::Rect {
    let screen = ctx.content_rect();
    egui::Rect::from_min_size(
        egui::pos2(screen.min.x + 12.0, screen.min.y + 12.0),
        egui::vec2(210.0, 150.0),
    )
}

    fn vr_compute_panels(
        &self,
        head0: glam::Vec3,
        flat: glam::Vec3,
    ) -> ([Option<zestengine_render::VrPanel>; 4], Vec<egui::Rect>) {
        use glam::{Mat4, Quat, Vec3};
        let screen = self.egui_ctx.content_rect();
        let (sw, sh) = (screen.width().max(1.0), screen.height().max(1.0));
        let vp = self.viewport_rect;
        let scale = self.vr_rig.scale.max(1e-3);
        let mut regions: Vec<egui::Rect> = Vec::new();
        if vp.min.x - screen.min.x > 8.0 {
            regions.push(egui::Rect::from_min_max(
                screen.min,
                egui::pos2(vp.min.x, screen.max.y),
            ));
        }
        if screen.max.x - vp.max.x > 8.0 {
            regions.push(egui::Rect::from_min_max(
                egui::pos2(vp.max.x, screen.min.y),
                screen.max,
            ));
        }
        if vp.min.y - screen.min.y > 8.0 {
            regions.push(egui::Rect::from_min_max(
                egui::pos2(vp.min.x, screen.min.y),
                egui::pos2(vp.max.x, vp.min.y),
            ));
        }
        if screen.max.y - vp.max.y > 8.0 {
            regions.push(egui::Rect::from_min_max(
                egui::pos2(vp.min.x, vp.max.y),
                egui::pos2(vp.max.x, screen.max.y),
            ));
        }
        if regions.is_empty() {
            regions.push(screen); // fallback: no viewport rect yet -> one panel
        }
        regions.truncate(4);
        let n = regions.len();
        let base_yaw = flat.x.atan2(flat.z);
        let dist = 1.4 * scale;
        let ph = 0.9 * scale;
        let mut panels: [Option<zestengine_render::VrPanel>; 4] = [None; 4];
        for (i, r) in regions.iter().enumerate() {
            let frac = if n > 1 {
                i as f32 / (n as f32 - 1.0) - 0.5
            } else {
                0.0
            };
            let yaw = base_yaw + frac * 1.3; // ~±37° spread across the arc
            let dir = Vec3::new(yaw.sin(), 0.0, yaw.cos());
            let pos = head0 + dir * dist;
            let to_user = (head0 - pos).normalize_or(Vec3::Z);
            let face_yaw = to_user.x.atan2(to_user.z);
            let rot = Quat::from_rotation_y(face_yaw);
            let aspect = (r.width() / r.height().max(1.0)).clamp(0.1, 6.0);
            let pw = ph * aspect;
            let model = Mat4::from_scale_rotation_translation(Vec3::new(pw, ph, 1.0), rot, pos);
            let uv_rect = [
                (r.min.x - screen.min.x) / sw,
                (r.min.y - screen.min.y) / sh,
                (r.max.x - screen.min.x) / sw,
                (r.max.y - screen.min.y) / sh,
            ];
            panels[i] = Some(zestengine_render::VrPanel { model, uv_rect });
        }
        (panels, regions)
    }

    /// VR locomotion + pointer interaction, driven by controller input. Tracker/
    /// controller poses are stage-space (from `body_poses`); the `VrRig` maps them
    /// to world. Left stick = fly, right stick = turn + vertical, left grip =
    /// grab-drag the world, both grips = scale yourself, right trigger = pick/move,
    /// menu button = toggle the hand menu. All relative to the player rig.
    fn update_vr(&mut self, input: zestengine_xr::VrInput, dt: f32) {
        use glam::Vec3;
        let head = self.vr_targets.head;
        let lhand = self.vr_targets.left_hand;
        let rhand = self.vr_targets.right_hand;
        let speed = 3.0 * self.vr_rig.scale; // bigger player covers more ground

        // Fly relative to the head's facing (projected onto the ground plane).
        if let Some((_, hrot)) = head {
            let wr = self.vr_rig.stage_rot_to_world(hrot);
            let fwd = (wr * Vec3::NEG_Z * Vec3::new(1.0, 0.0, 1.0)).normalize_or_zero();
            let right = (wr * Vec3::X * Vec3::new(1.0, 0.0, 1.0)).normalize_or_zero();
            let (lx, ly) = input.left_stick;
            let delta = (right * lx + fwd * ly) * speed * dt;
            if delta.length_squared() > 0.0 {
                self.vr_rig.fly(delta);
            }
            // Right stick turns only (no vertical fly — use a grip-drag to move
            // yourself up/down instead).
            let (rx, _ry) = input.right_stick;
            // The avatar's VR menu uses the right stick while it is up.
            if rx.abs() > 0.05 && !self.avatar_menu_vr.is_open() {
                let pivot = head
                    .map(|(p, _)| self.vr_rig.stage_to_world(p))
                    .unwrap_or(self.vr_rig.origin);
                self.vr_rig.turn_about(pivot, -rx * 1.5 * dt);
            }
        }

        // Two-handed scale: change by the inter-hand distance ratio about the
        // midpoint (pull apart = world grows / you shrink).
        let both_grip = input.left_grip > 0.5 && input.right_grip > 0.5;
        if both_grip {
            if let (Some((lp, _)), Some((rp, _))) = (lhand, rhand) {
                let lw = self.vr_rig.stage_to_world(lp);
                let rw = self.vr_rig.stage_to_world(rp);
                let dist = (lw - rw).length();
                if let Some(prev) = self.vr_prev_hand_dist
                    && prev > 1e-3 && dist > 1e-3 {
                        self.vr_rig.scale_about((lw + rw) * 0.5, prev / dist);
                    }
                self.vr_prev_hand_dist = Some(dist);
            }
        } else {
            self.vr_prev_hand_dist = None;
        }

        // Grab-drag the world with EITHER grip (when not scaling). Up/down comes
        // from dragging vertically. The active hand re-anchors on change so you
        // can hand off the drag from one controller to the other.
        if both_grip {
            self.vr_grab_anchor = None;
            self.vr_drag_hand = None;
        } else {
            // Prefer whichever hand is gripping; if both single grips somehow,
            // left wins (both_grip already handled the two-grip case). While an
            // object is selected the right grip belongs to it (drag-move), so
            // only the left grip drags the world then.
            let right_owns_object = matches!(
                self.selection,
                Selection::Object(i) if i < self.scene.objects.len()
            );
            let drag = if input.left_grip > 0.5 {
                lhand.map(|(p, _)| (p, true))
            } else if input.right_grip > 0.5 && !right_owns_object {
                rhand.map(|(p, _)| (p, false))
            } else {
                None
            };
            match drag {
                Some((hp, is_left)) => {
                    if self.vr_drag_hand != Some(is_left) {
                        // New grab (or switched hands): re-anchor under this hand.
                        self.vr_grab_anchor = Some(self.vr_rig.grab_anchor(hp));
                        self.vr_drag_hand = Some(is_left);
                    }
                    if let Some(anchor) = self.vr_grab_anchor {
                        self.vr_rig.drag_to(anchor, hp);
                    }
                }
                None => {
                    self.vr_grab_anchor = None;
                    self.vr_drag_hand = None;
                }
            }
        }

        // Right-hand pointer.
        let trigger_edge = input.right_trigger && !self.vr_prev_trigger;
        self.vr_overlay_draw = None;
        self.vr_ui_pointer = None;
        self.vr_ui_pressed = false;
        if let Some((rp, rr)) = rhand {
            let ray = {
                let (wp, wr) = self.vr_rig.stage_pose_to_world((rp, rr));
                zestengine_core::pointer_ray(wp, wr)
            };
            if self.vr_menu_open {
                // VR-native multi-panel UI: split the editor UI into separate 3D
                // panels (the screen regions AROUND the viewport — left/right/top/
                // bottom strips), arranged in a world-anchored arc. The viewport
                // region is omitted (you're inside it). Point with the right
                // controller; trigger = click, mapped to the hit panel's region.
                if let Some((head0, flat)) = self.vr_panel_anchor {
                    let (panels, regions) = self.vr_compute_panels(head0, flat);
                    // Right-hand pointer: hit-test each panel, map to its screen
                    // region so egui sees a click at the right place.
                    for (vpanel, r) in panels.iter().flatten().zip(regions.iter()) {
                        let ppos = vpanel.model.w_axis.truncate();
                        let pn = vpanel.model.z_axis.truncate().normalize_or(glam::Vec3::Z);
                        if let Some(hit) = zestengine_core::ray_plane(ray, ppos, pn) {
                            let local = vpanel.model.inverse().transform_point3(hit);
                            if local.x.abs() <= 0.5 && local.y.abs() <= 0.5 {
                                let u = local.x + 0.5;
                                let v = local.y + 0.5;
                                let px = r.min.x + u * r.width();
                                let py = r.min.y + (1.0 - v) * r.height();
                                self.vr_ui_pointer = Some(egui::pos2(px, py));
                                self.vr_ui_pressed = input.right_trigger;
                                break;
                            }
                        }
                    }
                    self.vr_overlay_draw = Some(zestengine_render::VrOverlayDraw {
                        panels,
                        // Hands + laser endpoints are filled in at the render site
                        // (lasers show even when the hand menu is down).
                        left_hand: None,
                        right_hand: None,
                        left_hand_skeleton: None,
                        right_hand_skeleton: None,
                        left_laser_end: None,
                        right_laser_end: None,
                        remote_hands: Vec::new(),
                        remote_heads: Vec::new(),
                    });
                }
            } else {
                // Wheel + arm panels (built even while pointing at the scene, so
                // the laser can land on them this frame).
                let mut panels: [Option<zestengine_render::VrPanel>;
                    zestengine_render::VR_MAX_PANELS] = [None; zestengine_render::VR_MAX_PANELS];
                let mut panel_rects: [Option<egui::Rect>; zestengine_render::VR_MAX_PANELS] =
                    [None; zestengine_render::VR_MAX_PANELS];
                let screen = self.egui_ctx.content_rect();
                let (sw, sh) = (screen.width().max(1.0), screen.height().max(1.0));
                let scale = self.vr_rig.scale.max(1e-3);
                let uv = |r: egui::Rect| {
                    [
                        (r.min.x - screen.min.x) / sw,
                        (r.min.y - screen.min.y) / sh,
                        (r.max.x - screen.min.x) / sw,
                        (r.max.y - screen.min.y) / sh,
                    ]
                };
                // Palette wheel: world-anchored in front of the head, showing
                // the wheel's fixed screen corner of the UI texture.
                if self.vr_wheel_open
                    && let Some((head0, flat)) = self.vr_panel_anchor {
                        let base_yaw = flat.x.atan2(flat.z);
                        let dir = Vec3::new(base_yaw.sin(), 0.0, base_yaw.cos());
                        let pos = head0 + dir * (1.4 * scale);
                        let to_user = (head0 - pos).normalize_or(Vec3::Z);
                        let rot = glam::Quat::from_rotation_y(to_user.x.atan2(to_user.z));
                        let wr = Self::vr_wheel_rect(&self.egui_ctx);
                        let ph = 0.5 * scale;
                        let pw = ph * (wr.width() / wr.height().max(1.0));
                        let model = glam::Mat4::from_scale_rotation_translation(
                            Vec3::new(pw, ph, 1.0),
                            rot,
                            pos,
                        );
                        panels[0] = Some(zestengine_render::VrPanel { model, uv_rect: uv(wr) });
                        panel_rects[0] = Some(wr);
                    }
                // Left-arm inspector: the left screen strip (hierarchy +
                // inspector) attached to the left hand while an object is
                // selected, facing the head.
                if let Selection::Object(_) = self.selection
                    && let (Some((lp, _)), Some((hp, _))) = (lhand, head) {
                        let vp = self.viewport_rect;
                        if vp.min.x - screen.min.x > 8.0 {
                            let r = egui::Rect::from_min_max(
                                screen.min,
                                egui::pos2(vp.min.x, screen.max.y),
                            );
                            let panel_pos = self.vr_rig.stage_to_world(lp) + Vec3::Y * (0.38 * scale);
                            let head_pos = self.vr_rig.stage_to_world(hp);
                            let n = (head_pos - panel_pos).normalize_or(Vec3::NEG_Z);
                            let right = n.cross(Vec3::Y).normalize_or(Vec3::X);
                            let up = right.cross(n).normalize_or(Vec3::Y);
                            let rot = glam::Quat::from_mat3(&glam::Mat3::from_cols(right, up, n));
                            let ph = 0.42 * scale;
                            let pw = ph * (r.width() / r.height().max(1.0)).clamp(0.1, 4.0);
                            let model = glam::Mat4::from_scale_rotation_translation(
                                Vec3::new(pw, ph, 1.0),
                                rot,
                                panel_pos,
                            );
                            panels[1] =
                                Some(zestengine_render::VrPanel { model, uv_rect: uv(r) });
                            panel_rects[1] = Some(r);
                        }
                    }
                // VR perf watch: a small readout on the back of the left hand,
                // always on while the hand tracks (not clickable).
                if let (Some((lp, _)), Some((hp, _))) = (lhand, head) {
                    let wr = Self::vr_watch_rect(&self.egui_ctx);
                    let panel_pos = self.vr_rig.stage_to_world(lp) + Vec3::Y * (0.03 * scale);
                    let head_pos = self.vr_rig.stage_to_world(hp);
                    let n = (head_pos - panel_pos).normalize_or(Vec3::NEG_Z);
                    let right = n.cross(Vec3::Y).normalize_or(Vec3::X);
                    let up = right.cross(n).normalize_or(Vec3::Y);
                    let rot = glam::Quat::from_mat3(&glam::Mat3::from_cols(right, up, n));
                    let ph = 0.24 * scale;
                    let pw = ph * (wr.width() / wr.height().max(1.0));
                    let model = glam::Mat4::from_scale_rotation_translation(
                        Vec3::new(pw, ph, 1.0),
                        rot,
                        panel_pos,
                    );
                    panels[2] = Some(zestengine_render::VrPanel { model, uv_rect: uv(wr) });
                }
                // Right-hand pointer: hit-test the panels and map to their screen
                // rects so egui sees the click at the right place.
                for (vpanel, r) in panels.iter().zip(panel_rects.iter()) {
                    let (Some(vpanel), Some(r)) = (vpanel, r) else { continue };
                    let ppos = vpanel.model.w_axis.truncate();
                    let pn = vpanel.model.z_axis.truncate().normalize_or(glam::Vec3::Z);
                    if let Some(hit) = zestengine_core::ray_plane(ray, ppos, pn) {
                        let local = vpanel.model.inverse().transform_point3(hit);
                        if local.x.abs() <= 0.5 && local.y.abs() <= 0.5 {
                            let u = local.x + 0.5;
                            let v = local.y + 0.5;
                            let px = r.min.x + u * r.width();
                            let py = r.min.y + (1.0 - v) * r.height();
                            self.vr_ui_pointer = Some(egui::pos2(px, py));
                            self.vr_ui_pressed = input.right_trigger;
                            break;
                        }
                    }
                }
                if panels.iter().any(|p| p.is_some()) {
                    self.vr_overlay_draw = Some(zestengine_render::VrOverlayDraw {
                        panels,
                        // Hands + laser endpoints are filled in at the render site
                        // (lasers show even when the wheel is closed).
                        left_hand: None,
                        right_hand: None,
                        left_hand_skeleton: None,
                        right_hand_skeleton: None,
                        left_laser_end: None,
                        right_laser_end: None,
                        remote_hands: Vec::new(),
                        remote_heads: Vec::new(),
                    });
                }
                // Object mode: the trigger selects only (never while it is over a
                // panel). The right grip moves the selection along the laser at
                // the anchored distance (set on pick or on the grip's rising
                // edge, so a re-grab never teleports it).
                if trigger_edge
                    && self.vr_ui_pointer.is_none()
                    && !self.body.panel_visible()
                    && let Some(idx) = self.scene.pick(ray.origin, ray.dir) {
                        self.select_object(Some(idx));
                        let obj_pos = self.scene.world_transform(idx).w_axis.truncate();
                        self.vr_move_dist = (obj_pos - ray.origin).length();
                    }
                let grip_edge = input.right_grip > 0.5 && !self.vr_prev_grip;
                if grip_edge
                    && !self.body.panel_visible()
                    && let Selection::Object(idx) = self.selection
                        && idx < self.scene.objects.len() {
                            let obj_pos = self.scene.world_transform(idx).w_axis.truncate();
                            self.vr_move_dist = (obj_pos - ray.origin).length();
                        }
                if input.right_grip > 0.5
                    && !self.body.panel_visible()
                    && let Selection::Object(idx) = self.selection
                        && idx < self.scene.objects.len() {
                            let target = ray.origin + ray.dir * self.vr_move_dist;
                            let id = self.scene.objects[idx].id;
                            self.scene.set_local_transform(id, Some(target), None, None);
                            // Co-edit: broadcast the drag (~10 Hz; the grip's
                            // transform is local = world for parentless objects).
                            let now = std::time::Instant::now();
                            if self.edit_transform_next.is_none_or(|t| now >= t) {
                                self.edit_transform_next = Some(now + std::time::Duration::from_millis(100));
                                let (s, r, t) = self.scene.world_transform(idx).to_scale_rotation_translation();
                                let local = glam::Mat4::from_scale_rotation_translation(s, r, t);
                                self.edit_send(zestengine_engine::edit_net::EditOp::SetTransform { id, transform: local.to_cols_array() });
                            }
                        }
            }
        }
        self.vr_prev_trigger = input.right_trigger;
        self.vr_prev_grip = input.right_grip > 0.5;

        // Toggle the UI panel on the menu button's rising edge.
        if input.menu != self.vr_prev_menu {
            tracing::info!(
                "VR menu button = {} (controllers tracking: left={}, right={})",
                input.menu,
                lhand.is_some(),
                rhand.is_some(),
            );
        }
        // In Play the menu button belongs to the avatar's radial menu when the
        // scene has one; the editor's VR panels stay on their other entry points.
        let avatar_menu_owns = self.playing && (self.avatar_menu_vr.is_open() || zestengine_engine::avatar_menu::scene_has_menu(&self.scene));
        if input.menu && !self.vr_prev_menu && !avatar_menu_owns {
            if self.vr_menu_open {
                // Menu closes the world arc first when it is up.
                self.vr_menu_open = false;
                tracing::info!("VR hand menu closed");
            } else {
                self.vr_wheel_open = !self.vr_wheel_open;
                tracing::info!("VR palette wheel {}", if self.vr_wheel_open { "opened" } else { "closed" });
                if self.vr_wheel_open {
                    self.vr_anchor_ahead();
                } else {
                    self.vr_panel_anchor = None;
                }
            }
        }
        self.vr_prev_menu = input.menu;
    }

    /// World-anchor the VR panels ~1.2 m in front of the head at eye height,
    /// facing the user. The panel transforms are rebuilt each frame from this
    /// point + direction so distance/size track the player scale.
    fn vr_anchor_ahead(&mut self) {
        if let Some(hs) = self.vr_targets.head {
            let (hp, hr) = self.vr_rig.stage_pose_to_world(hs);
            let fwd = (hr * glam::Vec3::NEG_Z).normalize_or(glam::Vec3::NEG_Z);
            let flat = glam::Vec3::new(fwd.x, 0.0, fwd.z).normalize_or(fwd);
            self.vr_panel_anchor = Some((hp, flat));
        }
    }

    /// Co-edit: the wire label of an object source (both peers share the same
    /// project files, so a label round-trips the spawn).
    fn edit_source_label(source: &zestengine_assets::ObjectSource) -> String {
        use zestengine_assets::ObjectSource;
        match source {
            ObjectSource::Empty => "empty".into(),
            ObjectSource::Camera => "camera".into(),
            ObjectSource::Light => "light".into(),
            ObjectSource::Primitive { shape } => format!("prim:{}", shape.label()),
            ObjectSource::Model { path, mesh, .. } => format!("model:{path}:{mesh}"),
            ObjectSource::Builtin { mesh } => format!("builtin:{mesh}"),
        }
    }

    /// Co-edit: parse a wire label back into a spawn source.
    fn edit_source_from_label(label: &str) -> Option<zestengine_assets::ObjectSource> {
        use zestengine_assets::{ObjectSource, PrimitiveShape};
        match label {
            "empty" => Some(ObjectSource::Empty),
            "camera" => Some(ObjectSource::Camera),
            "light" => Some(ObjectSource::Light),
            s if s.starts_with("prim:") => {
                let shape = match &s[5..] {
                    "Cube" => PrimitiveShape::Cube,
                    "Sphere" => PrimitiveShape::Sphere,
                    "Capsule" => PrimitiveShape::Capsule,
                    "Plane" => PrimitiveShape::Plane,
                    "Bucket" => PrimitiveShape::Bucket,
                    _ => return None,
                };
                Some(ObjectSource::Primitive { shape })
            }
            s if s.starts_with("model:") => {
                let (path, mesh) = s[6..].rsplit_once(':')?;
                Some(ObjectSource::Model {
                    path: path.into(),
                    mesh: mesh.parse().ok()?,
                    extra_meshes: Vec::new(),
                    model: None,
                    mesh_id: None,
                    extra_mesh_ids: Vec::new(),
                })
            }
            _ => None,
        }
    }

    /// Broadcast a local structural edit to the session (no-op without one).
    fn edit_send(&mut self, op: zestengine_engine::edit_net::EditOp) {
        if let Some(s) = self.edit_session.as_mut() {
            s.send_op(op);
        }
    }

    /// Apply one remote edit op. Remote ops never echo (the session stamps the
    /// sender) and are not recorded to the local undo stack, so each peer's
    /// undo stays local while the scene converges.
    fn edit_apply(&mut self, op: zestengine_engine::edit_net::EditOp) {
        use zestengine_engine::edit_net::EditOp;
        self.suppress_undo_record = true;
        match op {
            EditOp::Spawn(sp) => {
                let Some(source) = Self::edit_source_from_label(&sp.source) else {
                    tracing::warn!("co-edit: unknown spawn source {}", sp.source);
                    self.suppress_undo_record = false;
                    return;
                };
                let name = match &source {
                    zestengine_assets::ObjectSource::Primitive { shape } => shape.label().to_owned(),
                    zestengine_assets::ObjectSource::Empty => "Empty".into(),
                    zestengine_assets::ObjectSource::Camera => "Camera".into(),
                    zestengine_assets::ObjectSource::Light => "Light".into(),
                    _ => "Object".to_owned(),
                };
                let position = glam::Vec3::from(sp.translation);
                let rotation = glam::Quat::from_array(sp.rotation);
                let scale = glam::Vec3::from(sp.scale);
                if let Some(r) = self.renderer.as_mut() {
                    match self.scene.spawn(r, source, name, position) {
                        Ok(index) => {
                            self.scene.objects[index].id = sp.id;
                            self.scene.objects[index].rotation = rotation;
                            self.scene.objects[index].scale = scale;
                        }
                        Err(e) => tracing::warn!("co-edit spawn failed: {e:#}"),
                    }
                }
            }
            EditOp::SetTransform { id, transform } => {
                let m = glam::Mat4::from_cols_array(&transform);
                let (scale, rotation, translation) = m.to_scale_rotation_translation();
                self.scene.set_local_transform(id, Some(translation), Some(rotation), Some(scale));
            }
            EditOp::Delete { id } => {
                if let Some(idx) = self.scene.objects.iter().position(|o| o.id == id) {
                    self.scene.remove_object(idx);
                }
            }
            EditOp::Duplicate { id, new_id } => {
                if let Some(idx) = self.scene.objects.iter().position(|o| o.id == id)
                    && let Some(new_root) =
                        self.scene.duplicate_object_linked(idx, &self.project_root, &self.shaders, &self.components)
                    {
                        self.scene.objects[new_root].id = new_id;
                        self.scene.copy_view_flags(idx, new_root);
                    }
            }
            EditOp::Rename { id, name } => {
                if let Some(idx) = self.scene.objects.iter().position(|o| o.id == id) {
                    self.scene.objects[idx].name = name;
                }
            }
        }
        self.suppress_undo_record = false;
        self.scene_dirty = true;
    }

    /// This editor's presence: head pose + hand joints in WORLD space (the
    /// shared scene space), with the player's true scale.
    fn edit_presence(&self) -> zestengine_engine::edit_net::PeerPresence {
        use zestengine_engine::edit_net::{NetHandJoints, PeerPresence};
        let head = self.vr_targets.head.map(|(p, r)| {
            let (wp, wr) = self.vr_rig.stage_pose_to_world((p, r));
            (wp.to_array(), wr.to_array())
        });
        let mut hands = [None, None];
        for (i, joints) in self.vr_hand_joints.iter().enumerate() {
            if let Some(j) = joints {
                let mut out = NetHandJoints::default();
                for (k, p) in j.positions.iter().enumerate() {
                    out.positions[k] = self.vr_rig.stage_to_world(*p).to_array();
                    out.valid[k] = j.valid[k];
                }
                hands[i] = Some(out);
            }
        }
        PeerPresence { peer: 0, head, hands, scale: self.vr_rig.scale }
    }

    /// Pump the co-edit session every frame while editing: deliver voice,
    /// apply remote ops, refresh peer presence and broadcast our own.
    fn pump_edit_session(&mut self) {
        let (events, voice_packets, local_peer) = {
            let Some(session) = self.edit_session.as_mut() else { return };
            (session.pump(), session.take_voice(), session.local_peer())
        };
        if !voice_packets.is_empty()
            && let Some(v) = self.voice.as_mut() {
                v.receive(voice_packets);
            }
        for ev in events {
            match ev {
                zestengine_engine::edit_net::EditEvent::Op { op, .. } => self.edit_apply(op),
                zestengine_engine::edit_net::EditEvent::PeerJoined { name, .. } => {
                    self.set_status(format!("{name} joined the edit session"), false);
                }
                zestengine_engine::edit_net::EditEvent::PeerLeft { id } => {
                    self.edit_peers.remove(&id);
                    self.set_status("A peer left the edit session", false);
                }
                zestengine_engine::edit_net::EditEvent::Presence(p) => {
                    self.edit_peers.insert(p.peer, *p);
                }
                zestengine_engine::edit_net::EditEvent::Connected { .. }
                | zestengine_engine::edit_net::EditEvent::Disconnected { .. } => {}
            }
        }
        let now = std::time::Instant::now();
        // Presence ~15 Hz; headset or not, a flat editor sends a head-less
        // presence so peers see the name in the roster.
        if self.edit_presence_next.is_none_or(|t| now >= t) {
            self.edit_presence_next = Some(now + std::time::Duration::from_millis(66));
            let mut presence = self.edit_presence();
            presence.peer = local_peer;
            if let Some(s) = self.edit_session.as_mut() {
                s.send_presence(presence);
            }
        }
        // Mic: voice-activity capture, same encoder as play mode.
        if let Some(v) = self.voice.as_mut() {
            let settings = self.scene.voice_settings().unwrap_or_default();
            for (seq, payload) in v.capture_encoded(&settings, false, false) {
                self.edit_voice_seq = seq;
                if let Some(s) = self.edit_session.as_mut() {
                    s.send_voice(seq, &payload);
                }
            }
        }
    }

    /// The Co-edit window: roster while connected, host/join forms otherwise.
    fn edit_window_ui(&mut self, ctx: &egui::Context) {
        if !self.edit_window {
            return;
        }
        let mut open = self.edit_window;
        egui::Window::new("Co-edit session")
            .open(&mut open)
            .default_width(400.0)
            .show(ctx, |ui| {
                if let Some(session) = self.edit_session.as_ref() {
                    let hosting = session.is_host();
                    ui.label(format!(
                        "In session ({})",
                        if hosting { "host" } else { "joined" }
                    ));
                    for (_, name) in session.peers() {
                        ui.label(format!("· {name}"));
                    }
                    ui.separator();
                    if ui.button("Leave").clicked() {
                        self.edit_session = None;
                        self.edit_peers.clear();
                    }
                    return;
                }
                ui.strong("Host");
                ui.horizontal(|ui| {
                    ui.label("Your name");
                    ui.text_edit_singleline(&mut self.edit_form.name);
                });
                ui.horizontal(|ui| {
                    ui.label("Session");
                    ui.text_edit_singleline(&mut self.edit_form.session_name);
                });
                ui.horizontal(|ui| {
                    ui.label("Port");
                    ui.add(egui::DragValue::new(&mut self.edit_form.port).range(0..=65535));
                });
                ui.horizontal(|ui| {
                    ui.label("Secret");
                    ui.text_edit_singleline(&mut self.edit_form.secret);
                });
                let host = ui.button("Host").clicked();
                ui.separator();
                ui.strong("Join");
                ui.horizontal(|ui| {
                    ui.label("Address");
                    ui.text_edit_singleline(&mut self.edit_form.join_addr);
                });
                // LAN discovery: edit-session hosts answer on the edit
                // discovery port (separate from the game browser's list).
                let now = std::time::Instant::now();
                if self.edit_browser_next.is_none_or(|t| now >= t) {
                    self.edit_browser_next = Some(now + std::time::Duration::from_secs(2));
                    if self.edit_browser.is_none() {
                        self.edit_browser = zestengine_engine::net::LanBrowser::with_port(zestengine_engine::edit_net::EDIT_DISCOVERY_PORT).ok();
                    }
                    if let Some(b) = self.edit_browser.as_mut() {
                        b.refresh();
                    }
                }
                if let Some(b) = self.edit_browser.as_mut() {
                    let sessions: Vec<(std::net::SocketAddr, String)> =
                        b.poll().iter().map(|s| (s.addr, s.name.clone())).collect();
                    if !sessions.is_empty() {
                        ui.label("On this LAN:");
                        for (addr, name) in sessions {
                            if ui.button(format!("{name} — {addr}")).clicked() {
                                self.edit_form.join_addr = addr.to_string();
                            }
                        }
                    }
                }
                let join = ui.button("Join").clicked();
                if host {
                    let (name, session_name, secret) = (
                        self.edit_form.name.clone(),
                        self.edit_form.session_name.clone(),
                        self.edit_form.secret.clone(),
                    );
                    let cfg = zestengine_engine::edit_net::EditConfig {
                        name,
                        session_name,
                        port: self.edit_form.port,
                        secret,
                        discovery: true,
                        discovery_port: zestengine_engine::edit_net::EDIT_DISCOVERY_PORT,
                    };
                    match zestengine_engine::net::UdpTransport::bind(cfg.port) {
                        Ok(t) => {
                            let addr = t.local_addr();
                            self.edit_session = Some(zestengine_engine::edit_net::EditSession::host_over(Box::new(t), cfg));
                            self.set_status(
                                format!("Hosting an edit session on {}", addr.map(|a| a.to_string()).unwrap_or_else(|| "an OS-assigned port".into())),
                                false,
                            );
                        }
                        Err(e) => self.set_status(format!("Could not host: {e}"), true),
                    }
                }
                if join {
                    let addr: std::net::SocketAddr = match self.edit_form.join_addr.parse() {
                        Ok(a) => a,
                        Err(_) => {
                            self.set_status("Join address must look like host:port", true);
                            return;
                        }
                    };
                    let (name, session_name, secret) = (
                        self.edit_form.name.clone(),
                        self.edit_form.session_name.clone(),
                        self.edit_form.secret.clone(),
                    );
                    let cfg = zestengine_engine::edit_net::EditConfig {
                        name,
                        session_name,
                        port: self.edit_form.port,
                        secret,
                        discovery: false,
                        discovery_port: zestengine_engine::edit_net::EDIT_DISCOVERY_PORT,
                    };
                    match zestengine_engine::net::UdpTransport::bind(0).and_then(|t| zestengine_engine::edit_net::EditSession::join_over(Box::new(t) as Box<dyn zestengine_engine::net::Transport>, addr, cfg).map_err(|e| std::io::Error::other(e.to_string()))) {
                        Ok(session) => {
                            self.edit_session = Some(session);
                            self.set_status(format!("Joining edit session at {addr}…"), false);
                        }
                        Err(e) => self.set_status(format!("Could not join: {e}"), true),
                    }
                }
            });
        self.edit_window = open;
    }

    /// Run a VR tool action (the in-VR egui "VR Tools" window calls these; they
    /// mirror operations that have no desktop button yet).
    fn dispatch_vr_action(&mut self, action: VrAction) {
        match action {
            VrAction::BodyMenu => self.body.menu_open = true,
            VrAction::CalibrateTpose => self.body.calibrate_now(),
            VrAction::ClearCalibration => self.body.clear_calibration(),
            VrAction::ToggleFootIk => {
                let on = self.scene.foot_ik_enabled();
                self.scene
                    .set_foot_ik((!on).then(zestengine_engine::humanoid::FootIkParams::default));
            }
            VrAction::ResetRig => self.vr_rig = zestengine_core::VrRig::default(),
            VrAction::DeleteSelected => {
                if let Selection::Object(idx) = self.selection
                    && idx < self.scene.objects.len() {
                        let before = self.scene.snapshot_objects();
                        self.scene.remove_object(idx);
                        self.record_structure(before);
                        self.selection = Selection::None;
                    }
            }
        }
    }

    /// The VR palette wheel: a fixed bottom-right egui Area with one-click
    /// actions (duplicate, delete, focus, undo, play, UI panels). `update_vr`
    /// hit-tests the same rect via [`vr_wheel_rect`], and the XR panel shows
    /// this corner of the UI texture when the wheel is open.
    fn vr_wheel_window(&mut self, ctx: &egui::Context) {
        if !self.vr_wheel_open || self.xr_session.is_none() {
            return;
        }
        let rect = Self::vr_wheel_rect(ctx);
        egui::Area::new("zestengine-vr-wheel".into())
            .fixed_pos(rect.min)
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.set_width(rect.width() - 24.0);
                    ui.set_height(rect.height() - 24.0);
                    ui.strong("VR");
                    ui.separator();
                    let obj = match self.selection {
                        Selection::Object(i) if i < self.scene.objects.len() => Some(i),
                        _ => None,
                    };
                    egui::Grid::new("vr-wheel-grid")
                        .num_columns(2)
                        .spacing([10.0, 8.0])
                        .show(ui, |ui| {
                            if ui
                                .add_enabled(obj.is_some(), egui::Button::new("Duplicate"))
                                .clicked()
                                && let Some(i) = obj {
                                    self.actions.push(EditorAction::DuplicateObject(i));
                                }
                            if ui
                                .add_enabled(obj.is_some(), egui::Button::new("Delete"))
                                .clicked()
                                && let Some(i) = obj {
                                    self.actions.push(EditorAction::DeleteObject(i));
                                }
                            ui.end_row();
                            if ui
                                .add_enabled(obj.is_some(), egui::Button::new("Focus"))
                                .clicked() {
                                    self.actions.push(EditorAction::FocusSelected);
                                }
                            if ui.button("Undo").clicked() {
                                self.actions.push(EditorAction::Undo);
                            }
                            ui.end_row();
                            if ui
                                .button(if self.playing { "Stop" } else { "Play" })
                                .clicked() {
                                    self.toggle_play();
                                }
                            if ui.button("UI panels").clicked() {
                                self.vr_menu_open = true;
                                self.vr_wheel_open = false;
                                self.vr_anchor_ahead();
                            }
                            ui.end_row();
                        });
                });
            });
    }

    /// The VR perf watch: a fixed top-left egui Area with live frame stats,
    /// UV-mapped onto the left wrist panel by `update_vr` while a VR session
    /// is active.
    fn vr_watch_window(&self, ctx: &egui::Context) {
        if self.xr_session.is_none() {
            return;
        }
        let rect = Self::vr_watch_rect(ctx);
        egui::Area::new("zestengine-vr-watch".into())
            .fixed_pos(rect.min)
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.set_width(rect.width() - 16.0);
                    let frame_ms = self.frame_timings.render;
                    let gpu_ms = self
                        .renderer
                        .as_ref()
                        .map(|r| r.stats().gpu_frame_ms)
                        .unwrap_or(0.0);
                    let fps = if frame_ms > 0.0 { 1000.0 / frame_ms } else { 0.0 };
                    ui.label(format!("Frame {frame_ms:5.1} ms"));
                    ui.label(format!("GPU   {gpu_ms:5.2} ms"));
                    ui.label(format!("FPS   {fps:5.0}"));
                    ui.label(format!("Scale {:.2}x", self.vr_rig.scale));
                    ui.label(egui::RichText::new(build_stamp()).small());
                });
            });
    }

    /// A small "VR Tools" egui window (only while a VR session is active) so the
    /// VR-specific actions are reachable from the in-headset UI panel as well as
    /// the desktop. The rest of the editor UI is shown on the panel unchanged.
    fn vr_tools_window(&mut self, ctx: &egui::Context) {
        if self.xr_session.is_none() {
            return;
        }
        let mut action = None;
        let mut gi_changed = false;
        egui::Window::new("VR Tools")
            .default_open(true)
            .show(ctx, |ui| {
                if self.vr_menu_open && ui.button("Hide UI panels").clicked() {
                    self.vr_menu_open = false;
                }
                ui.label(format!("Calibrated: {}", self.body.calibrated()));
                ui.label(format!("Foot IK: {}", self.scene.foot_ik_enabled()));
                ui.label(format!("Player scale: {:.2}x", self.vr_rig.scale));
                ui.separator();
                // Rendering / performance toggles. Realtime GI also lights the VR
                // view (via the world-probe volume); reflections + quality affect
                // cost. Any change re-triggers the GI trace.
                ui.label("Rendering");
                let gi = &mut self.scene.environment.realtime_gi;
                gi_changed |= ui.checkbox(&mut gi.enabled, "Realtime GI").changed();
                let refl = match gi.reflection_mode {
                    0 => "Reflections: Off",
                    1 => "Reflections: SSR",
                    _ => "Reflections: RT",
                };
                if ui.button(refl).clicked() {
                    gi.reflection_mode = (gi.reflection_mode + 1) % 3;
                    gi_changed = true;
                }
                if ui
                    .button(format!("GI Quality: {}", gi.quality.label()))
                    .clicked()
                {
                    use zestengine_assets::FluxQuality::*;
                    gi.quality = match gi.quality {
                        Performance => Balanced,
                        Balanced => High,
                        High => Ultra,
                        Ultra => Performance,
                    };
                    gi_changed = true;
                }
                if gi.mode.uses_bounces() {
                    ui.horizontal(|ui| {
                        ui.label("Diffuse bounces");
                        if ui.selectable_label(gi.bounces == 1, "Single").clicked() {
                            gi.bounces = 1; gi_changed = true;
                        }
                        if ui.selectable_label(gi.bounces > 1, "Infinite").clicked() {
                            gi.bounces = 4; gi_changed = true;
                        }
                    });
                }
                ui.separator();
                for (label, act) in VR_TOOLS {
                    if ui.button(*label).clicked() {
                        action = Some(*act);
                    }
                }
            });
        if gi_changed {
            self.rt_gi.invalidate();
        }
        if let Some(a) = action {
            self.dispatch_vr_action(a);
        }
    }

    fn process_actions(&mut self) {
        if self.renderer.is_none() {
            return;
        }
        let actions = std::mem::take(&mut self.actions);
        for action in actions {
            // Renderer exists (checked above); reborrow per action so other
            // &self helpers (cursor_ray, pick) stay usable.
            macro_rules! renderer {
                () => {
                    self.renderer.as_mut().unwrap()
                };
            }
            // Leaving a Model Import session with unapplied edits asks first:
            // selecting something else, entering play, switching project.
            if let Some(fm) = self.file_meta.as_ref().filter(|f| f.dirty)
                && self.import_prompt.is_none()
            {
                let leaving = match &action {
                    EditorAction::SelectFile(p) => *p != fm.path,
                    EditorAction::TogglePlay => !self.playing,
                    EditorAction::OpenProject(_) | EditorAction::OpenProjectHub => true,
                    _ => false,
                };
                if leaving {
                    self.import_prompt = Some(ImportPrompt { path: fm.path.clone(), resume: Some(action), restore_selection: false, quit: false });
                    continue;
                }
            }
            // Mark the scene unsaved for content mutations (transform/material
            // edits are caught separately in record_edits). Saving/loading
            // clears the flag.
            if matches!(
                &action,
                EditorAction::Spawn(_)
                    | EditorAction::SpawnLight(_)
                    | EditorAction::SpawnAreaLight
                    | EditorAction::SpawnProbeVolume
                    | EditorAction::SpawnPostFxVolume(_)
                    | EditorAction::SpawnAudioSource
                    | EditorAction::SpawnAudioVolume
                    | EditorAction::SpawnBoxCollider
                    | EditorAction::SpawnSphereCollider
                    | EditorAction::DeleteObject(_)
                    | EditorAction::DuplicateObject(_)
                    | EditorAction::InstantiatePrefab(..)
                    | EditorAction::SpawnUi(_)
                    | EditorAction::SpawnUiTemplate(_)
                    | EditorAction::SetCanvasMode(..)
                    | EditorAction::ImportModel(_)
                    | EditorAction::NewScene
                    | EditorAction::SetSkybox(_)
                    | EditorAction::ClearSkybox
                    | EditorAction::SetSkyboxFace(..)
                    | EditorAction::ClearSkyboxFaces
            ) {
                self.scene_dirty = true;
            }
            match action {
                EditorAction::SelectFile(path) => {
                    // Model file → load its .meta so import options show in the
                    // Inspector (creating the sidecar on first view).
                    // Clicking the file whose import edits are still unapplied keeps them.
                    let keep_meta = self.file_meta.as_ref().is_some_and(|f| f.dirty && f.path == path);
                    if !keep_meta {
                        self.file_meta = None;
                    }
                    let is_model = !keep_meta && path
                        .extension()
                        .and_then(|e| e.to_str())
                        .map(|e| matches!(e.to_ascii_lowercase().as_str(), "fbx" | "gltf" | "glb" | "obj") || is_texture_ext(e))
                        .unwrap_or(false);
                    if is_model {
                        match zestengine_assets::load_or_create_asset_meta(&path) {
                            Ok(meta) => {
                                self.file_meta = Some(FileMeta::new(path.clone(), meta))
                            }
                            Err(e) => tracing::error!("loading .meta: {e:#}"),
                        }
                    }
                    // A locked inspector keeps showing (and editing) the locked
                    // file: the selection moves, the file editor stays pinned.
                    if self.inspector.locked {
                        self.selection = Selection::File(path.clone());
                    } else {
                        self.open_material_inspector(&path);
                        self.selection = Selection::File(path);
                    }
                }
                EditorAction::EditInZestSurface(index) => self.start_zestsurface(index),
                EditorAction::PlacementDrop(path, row) => self.placement_hierarchy_drop(path, row),
                EditorAction::ImportModel(path) => {
                    // Parse the model on a worker thread; the GPU upload happens
                    // on the main thread in `apply_imported_model` when it lands.
                    let rel = self.project_root.join(zestengine_engine::scene::relative_to(&path, &self.project_root));
                    let source = PathBuf::from(zestengine_engine::scene::relative_to(&rel, &self.project_root));
                    let name = path
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|| "model".into());
                    self.tasks.spawn(
                        format!("Import {name}"),
                        crate::tasks::TaskKind::ImportModel,
                        false,
                        move |cancel, progress| {
                            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                                return Ok(crate::tasks::TaskPayload::None);
                            }
                            let phase = |done: f32, label: &str| {
                                if let Ok(mut p) = progress.lock() {
                                    *p = crate::tasks::TaskProgress::Fraction { done, label: label.into() };
                                }
                            };
                            phase(0.05, "Reading and converting the model");
                            let scene = zestengine_assets::load_model_with_meta(&path)
                                .map_err(|e| e.to_string())?;
                            phase(0.8, "Rig and ids");
                            // Keep the auto-generated rig in the model's .meta.
                            if let Err(e) = zestengine_engine::humanoid::persist_scene_rig(&path, &scene.skeletons, &scene.meshes) {
                                tracing::warn!("storing rig in .meta: {e:#}");
                            }
                            if let Err(e) = zestengine_assets::record_model_ids(&path, &scene) {
                                tracing::warn!("recording model ids: {e:#}");
                            }
                            Ok(crate::tasks::TaskPayload::Model {
                                source,
                                scene: Box::new(scene),
                            })
                        },
                    );
                }
                EditorAction::CreateMaterial(dir) => {
                    let path = unique_path(&dir, "new_material", "material");
                    let file = zestengine_assets::MaterialFile {
                        name: path
                            .file_stem()
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_else(|| "new material".into()),
                        shader: "standard".into(),
                        params: Default::default(),
                        features: Default::default(),
                        render_queue: None,
                        textures: Default::default(),
                        custom: Default::default(),
                    };
                    match zestengine_assets::save_material_file(&path, &file) {
                        Ok(()) => self.actions.push(EditorAction::SelectFile(path)),
                        Err(e) => tracing::error!("creating material: {e:#}"),
                    }
                }
                EditorAction::CreateScene(dir) => {
                    let path = unique_path(&dir, "new_scene", "scene");
                    if let Err(e) =
                        zestengine_assets::save_scene_file(&path, &zestengine_assets::SceneFile::new_default())
                    {
                        tracing::error!("creating scene: {e:#}");
                    }
                }
                EditorAction::CreateShader(dir) => {
                    let path = unique_path(&dir, "new_shader", "frag");
                    match std::fs::write(&path, zestengine_assets::SHADER_TEMPLATE) {
                        Ok(()) => self.actions.push(EditorAction::SelectFile(path)),
                        Err(e) => tracing::error!("creating shader: {e:#}"),
                    }
                }
                EditorAction::CreateWaterWaves(dir) => {
                    let path = unique_path(&dir, "Water Waves", "waterwaves");
                    match zestengine_assets::water_file::save_water_waves(&path, &Default::default()) {
                        Ok(()) => self.actions.push(EditorAction::SelectFile(path)),
                        Err(e) => tracing::error!("creating water waves: {e:#}"),
                    }
                }
                EditorAction::CreateAudioAsset(dir, settings) => {
                    let ext = if settings { zestengine_assets::AUDIO_SETTINGS_EXTENSION } else { zestengine_assets::AUDIOFX_EXTENSION };
                    let path = unique_path(&dir, if settings { "Audio Settings" } else { "Audio Environment" }, ext);
                    let result = if settings { zestengine_assets::save_audio_settings(&path, &Default::default()) }
                        else { zestengine_assets::save_audiofx(&path, &Default::default()) };
                    match result {
                        Ok(()) => self.actions.push(EditorAction::SelectFile(path)),
                        Err(e) => tracing::error!("creating audio asset: {e:#}"),
                    }
                }
                EditorAction::CreatePostFx(dir) => {
                    let path = unique_path(&dir, "new_postfx", "postfx");
                    match zestengine_assets::save_postfx(
                        &path,
                        &zestengine_assets::PostFxProfile::default(),
                    ) {
                        Ok(()) => self.actions.push(EditorAction::SelectFile(path)),
                        Err(e) => tracing::error!("creating postfx: {e:#}"),
                    }
                }
                EditorAction::CreatePhysMat(dir) => {
                    let path = unique_path(&dir, "new_physics_material", zestengine_core::PHYSMAT_EXTENSION);
                    match zestengine_core::save_physmat(&path, &zestengine_core::PhysicsMaterial::default()) {
                        Ok(()) => self.actions.push(EditorAction::SelectFile(path)),
                        Err(e) => tracing::error!("creating physics material: {e}"),
                    }
                }
                EditorAction::CreateSpriteAtlas(dir) => {
                    let path = unique_path(&dir, "new_sprite_atlas", zestengine_assets::sprite_atlas::ATLAS_EXTENSION);
                    match zestengine_assets::sprite_atlas::SpriteAtlasFile::default().save(&path) {
                        Ok(()) => self.actions.push(EditorAction::SelectFile(path)),
                        Err(e) => tracing::error!("creating sprite atlas: {e:#}"),
                    }
                }
                EditorAction::CreateSpriteShape(dir) => {
                    let path = unique_path(&dir, "new_sprite_shape", zestengine_core::SPRITE_SHAPE_EXTENSION);
                    let profile = zestengine_core::sprite_shape::preset_shape(zestengine_core::sprite_shape::ShapePreset::GroundWithFill).profile;
                    match zestengine_core::save_spriteshape(&path, &profile) {
                        Ok(()) => self.actions.push(EditorAction::SelectFile(path)),
                        Err(e) => tracing::error!("creating sprite shape profile: {e}"),
                    }
                }
                EditorAction::CreateFolder(dir) => {
                    let path = unique_path(&dir, "new_folder", "");
                    if let Err(e) = std::fs::create_dir_all(&path) {
                        tracing::error!("creating folder: {e:#}");
                    }
                }
                EditorAction::RecaptureReflections => {
                    // Reload / re-render both active probes' cubes from the
                    // current scene (same path as the on-load capture).
                    if self.scene.active_reflection_probe(glam::Vec3::ZERO).is_some() {
                        self.refl_slots.reset();
                        self.set_status("Recapturing reflection probes…", false);
                    } else {
                        self.set_status("No Reflection Probe in the scene — add one to an object first.", false);
                    }
                }
                EditorAction::BakeReflections => {
                    // Every probe in the scene gets its own baked cube.
                    let probes = self.scene.reflection_probe_captures();
                    if probes.is_empty() {
                        self.set_status("No Reflection Probe in the scene — add one to an object first.", false);
                    } else {
                        self.set_status(format!("Baking {} reflection probe(s)…", probes.len()), true);
                        self.refl_bake_queue = probes;
                    }
                }
                EditorAction::PickAt(pos) => {
                    // Clicking the 3D viewport drops any lingering code-editor
                    // text focus, so keyboard shortcuts (Ctrl+Z/Y) route to the
                    // editor again instead of egui's TextEdit consuming them.
                    self.egui_ctx.memory_mut(|m| m.stop_text_input());
                    // The click landed on in-game UI; it belongs to the game.
                    if self.playing && self.game_ui_frame.pointer_over_ui {
                        continue;
                    }
                    if let Some((origin, dir)) = self.cursor_ray(pos) {
                        let hit = self.scene.pick(origin, dir);
                        // Ctrl/Shift click adds/toggles into the multi-selection
                        // (like the file explorer); plain click is single-select.
                        let additive = self.egui_ctx.input(|i| {
                            i.modifiers.command || i.modifiers.shift
                        });
                        match (hit, additive) {
                            (Some(idx), true) => {
                                if let Some(p) =
                                    self.multi_objects.iter().position(|&x| x == idx)
                                {
                                    if self.multi_objects.len() > 1 {
                                        self.multi_objects.remove(p);
                                        if self.selection == Selection::Object(idx) {
                                            self.selection = Selection::Object(
                                                self.multi_objects[p.min(
                                                    self.multi_objects.len() - 1,
                                                )],
                                            );
                                        }
                                    }
                                } else {
                                    self.multi_objects.push(idx);
                                    self.selection = Selection::Object(idx);
                                }
                            }
                            (hit, _) => self.select_object(hit),
                        }
                    }
                }
                EditorAction::AssignMaterialAt(pos, path) => {
                    if let Some((origin, dir)) = self.cursor_ray(pos)
                        && let Some(hit) = self.scene.pick(origin, dir)
                    {
                        // Viewport drops target the primary slot (slot 0).
                        let Some(before) = self.scene.objects[hit].render.map(|r| r.material)
                        else {
                            continue;
                        };
                        let before_slot = self.scene.renderer_slot_value(hit, 0);
                        self.scene.assign_material(
                            renderer!(),
                            &mut self.shaders,
                            hit,
                            0,
                            &path,
                            &self.project_root,
                        );
                        let after = self.scene.objects[hit]
                            .render
                            .map(|r| r.material)
                            .unwrap_or(before);
                        if before != after {
                            self.undo_stack.record(
                                UndoEntry::Assign {
                                    object: hit,
                                    slot: 0,
                                    before,
                                    after,
                                    before_slot,
                                    after_slot: self.scene.renderer_slot_value(hit, 0),
                                },
                                false,
                            );
                        }
                        // A reassigned material may change emission/albedo.
                        self.rt_gi.invalidate();
                    }
                }
                EditorAction::AssignMaterialToObject(object, slot, path) => {
                    let Some(before) = self.scene.slot_material(object, slot) else {
                        continue;
                    };
                    let before_slot = self.scene.renderer_slot_value(object, slot);
                    self.scene.assign_material(
                        renderer!(),
                        &mut self.shaders,
                        object,
                        slot,
                        &path,
                        &self.project_root,
                    );
                    let after = self.scene.slot_material(object, slot).unwrap_or(before);
                    if before != after {
                        self.undo_stack.record(
                            UndoEntry::Assign {
                                object,
                                slot,
                                before,
                                after,
                                before_slot,
                                after_slot: self.scene.renderer_slot_value(object, slot),
                            },
                            false,
                        );
                    }
                    self.rt_gi.invalidate();
                }
                EditorAction::SelectObject(i) => {
                    if i < self.scene.objects.len() {
                        self.select_object(Some(i));
                    }
                }
                EditorAction::AssignSlot(object, slot, value) => {
                    let Some(before) = self.scene.slot_material(object, slot) else { continue };
                    let before_slot = self.scene.renderer_slot_value(object, slot);
                    self.scene.set_renderer_material(renderer!(), &mut self.shaders, &self.project_root, object, slot, value);
                    let after = self.scene.slot_material(object, slot).unwrap_or(before);
                    let after_slot = self.scene.renderer_slot_value(object, slot);
                    if before != after || before_slot != after_slot {
                        self.undo_stack.record(UndoEntry::Assign { object, slot, before, after, before_slot, after_slot }, false);
                    }
                    self.scene_dirty = true;
                    self.rt_gi.invalidate();
                }
                EditorAction::MakeUniqueMaterial(object, slot) => {
                    let Some(m) = self.scene.slot_material(object, slot) else { continue };
                    let model = self.scene.materials[m].model.clone();
                    let base = self.scene.materials[m]
                        .file
                        .as_ref()
                        .and_then(|f| f.file_stem())
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_else(|| model.name.chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' }).collect::<String>().trim_matches('_').to_owned());
                    let dir = self.project_root.join("materials");
                    let _ = std::fs::create_dir_all(&dir);
                    let path = unique_path(&dir, &format!("{}_copy", if base.is_empty() { "material" } else { &base }), "material");
                    let file = self.material_file_from_model(&model);
                    match zestengine_assets::save_material_file(&path, &file) {
                        Ok(()) => {
                            let before_slot = self.scene.renderer_slot_value(object, slot);
                            self.scene.assign_material(renderer!(), &mut self.shaders, object, slot, &path, &self.project_root);
                            let after = self.scene.slot_material(object, slot).unwrap_or(m);
                            let after_slot = self.scene.renderer_slot_value(object, slot);
                            self.undo_stack.record(UndoEntry::Assign { object, slot, before: m, after, before_slot, after_slot }, false);
                            self.scene_dirty = true;
                            self.set_status(format!("Made a unique copy: {}", relative_to(&path, &self.project_root)), false);
                        }
                        Err(e) => self.set_status(format!("Make Unique failed: {e:#}"), true),
                    }
                }
                EditorAction::MaterialEdited(index) => {
                    self.scene.apply_material(
                        renderer!(),
                        &mut self.shaders,
                        &self.project_root,
                        index,
                    );
                    // Emission/albedo may have changed; re-trace bounce light.
                    self.rt_gi.invalidate();
                }
                EditorAction::AssignTexture {
                    material,
                    slot,
                    path,
                } => {
                    if material < self.scene.materials.len() {
                        let rel = PathBuf::from(relative_to(&path, &self.project_root));
                        // A brand-new texture imports as its first slot expects.
                        let _ = zestengine_assets::ensure_first_import_for_slot(&path, slot);
                        if let Some(s) =
                            self.scene.materials[material].model.textures.slot_mut(slot)
                        {
                            *s = Some(rel);
                        }
                        self.scene.apply_material(
                            renderer!(),
                            &mut self.shaders,
                            &self.project_root,
                            material,
                        );
                        // A new texture (albedo/emission/etc.) can change bounce light.
                        self.rt_gi.invalidate();
                        // Persist to the backing `.material` file if there is one.
                        self.dirty_materials.insert(material);
                    }
                }
                EditorAction::AssignFileTexture { slot, path } => {
                    if let Some(fm) = self.file_material.as_mut() {
                        let rel = PathBuf::from(relative_to(&path, &self.project_root));
                        let _ = zestengine_assets::ensure_first_import_for_slot(&path, slot);
                        if let Some(s) = fm.model.textures.slot_mut(slot) {
                            *s = Some(rel);
                        }
                        fm.dirty = true;
                        self.actions
                            .push(EditorAction::FileMaterialEdited(fm.path.clone()));
                    }
                }
                EditorAction::ResetMaterial(index) => {
                    self.scene.materials[index].model = self.scene.materials[index].default.clone();
                    self.scene.apply_material(
                        renderer!(),
                        &mut self.shaders,
                        &self.project_root,
                        index,
                    );
                    self.rt_gi.invalidate();
                }
                EditorAction::ExtractMaterial(index) => {
                    self.extract_material(index);
                }
                EditorAction::FileMaterialEdited(path) => {
                    let Some(fm) = &self.file_material else {
                        continue;
                    };
                    let model = fm.model.clone();
                    for index in 0..self.scene.materials.len() {
                        if self.scene.materials[index].file.as_deref() != Some(&path) {
                            continue;
                        }
                        // Keep texture-derived state from the scene entry;
                        // the file editor doesn't change textures.
                        let has_normal = self.scene.materials[index].model.has_normal_texture;
                        let mut model = model.clone();
                        model.has_normal_texture = has_normal;
                        self.scene.materials[index].model = model;
                        self.scene.apply_material(
                            renderer!(),
                            &mut self.shaders,
                            &self.project_root,
                            index,
                        );
                    }
                    self.rt_gi.invalidate();
                }
                EditorAction::SaveFileMaterial => {
                    if let Some(fm) = &mut self.file_material {
                        let (params, features) = material_from_model(&fm.model);
                        fm.file.params = params;
                        fm.file.features = features;
                        fm.file.shader = fm.model.shader.clone();
                        fm.file.name = fm.model.name.clone();
                        fm.file.render_queue = Some(fm.model.render_queue);
                        let ignored = std::mem::take(&mut fm.file.textures.ignored_warnings);
                        fm.file.textures = zestengine_engine::scene::tex_file_from_paths(&fm.model.textures);
                        fm.file.textures.ignored_warnings = ignored;
                        if fm.model.shader != "standard" {
                            // Save custom values by property name (robust to
                            // property reordering in the shader source).
                            if let Some(source) = self
                                .shaders
                                .get(&fm.model.shader)
                                .and_then(|e| e.source.as_ref())
                            {
                                fm.file.custom = source.unpack(&fm.model.custom_values);
                            }
                        } else {
                            fm.file.custom.clear();
                        }
                        match zestengine_assets::save_material_file(&fm.path, &fm.file) {
                            Ok(()) => fm.dirty = false,
                            Err(e) => tracing::error!("saving material: {e:#}"),
                        }
                    }
                }
                EditorAction::RevertFileMeta => {
                    if let Some(fm) = &mut self.file_meta {
                        fm.revert();
                    }
                }
                EditorAction::ReimportTexture(path) => {
                    if let Some(fm) = &mut self.file_meta
                        && fm.path == path
                        && fm.dirty
                        && let Err(e) = fm.apply()
                    {
                        tracing::error!("saving .meta: {e:#}");
                    }
                    self.slot_validator.invalidate();
                    let n = self.scene.reimport_texture(renderer!(), &mut self.shaders, &self.project_root, &path);
                    self.rt_gi.invalidate();
                    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                    self.set_status(format!("Reimported {name} ({n} material{})", if n == 1 { "" } else { "s" }), false);
                }
                EditorAction::TextureSampler(path, sampler) => {
                    self.scene.apply_texture_sampler_live(renderer!(), &path, &sampler);
                }
                EditorAction::FixTextureSlot { material, slot, fix } => {
                    let textures = match material {
                        Some(m) => self.scene.materials.get(m).map(|e| e.model.textures.clone()),
                        None => self.file_material.as_ref().map(|f| f.model.textures.clone()),
                    };
                    if let Some(textures) = textures {
                        match crate::texture_inspector::apply_slot_fix(&self.project_root, &textures, slot, &fix) {
                            Ok(crate::texture_inspector::FixOutcome::Reimport(abs)) => {
                                self.actions.push(EditorAction::ReimportTexture(abs));
                            }
                            Ok(crate::texture_inspector::FixOutcome::Duplicated { slot, new_abs }) => {
                                self.actions.push(match material {
                                    Some(material) => EditorAction::AssignTexture { material, slot, path: new_abs },
                                    None => EditorAction::AssignFileTexture { slot, path: new_abs },
                                });
                            }
                            Ok(crate::texture_inspector::FixOutcome::Ignored(abs, slot)) => {
                                // Saved with the material when it has a file; otherwise this session only.
                                let file = match material {
                                    Some(m) => self.scene.materials.get(m).and_then(|e| e.file.clone()),
                                    None => self.file_material.as_ref().map(|f| f.path.clone()),
                                };
                                let rel = PathBuf::from(relative_to(&abs, &self.project_root));
                                match file {
                                    Some(f) => {
                                        if let Err(e) = crate::texture_inspector::persist_slot_ignore(&f, slot, &rel) {
                                            tracing::error!("saving ignored texture warning: {e:#}");
                                            self.slot_validator.ignore(abs, slot);
                                        } else if let Some(fm) = self.file_material.as_mut().filter(|fm| fm.path == f) {
                                            let entry = (slot, rel);
                                            if !fm.file.textures.ignored_warnings.contains(&entry) {
                                                fm.file.textures.ignored_warnings.push(entry);
                                            }
                                        }
                                    }
                                    None => self.slot_validator.ignore(abs, slot),
                                }
                            }
                            Err(e) => {
                                tracing::error!("texture fix: {e:#}");
                                self.set_status("Couldn't fix the texture (see log)", false);
                            }
                        }
                        self.slot_validator.invalidate();
                    }
                }
                EditorAction::ReimportModel(path) => {
                    if let Some(fm) = &mut self.file_meta
                        && fm.path == path
                        && let Err(e) = fm.apply() {
                            tracing::error!("saving .meta: {e:#}");
                        }
                    self.spawn_model_reimport(path);
                }
                EditorAction::ModelRemap(path, job) => {
                    self.start_model_remap(path, job);
                }
                EditorAction::GeoRebuild(path, job) => {
                    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                    self.tasks.spawn(format!("Rebuilding zestlets for {name}"), crate::tasks::TaskKind::ImportModel, false, move |cancel, progress| job.run(&path, &cancel, &progress));
                }
                EditorAction::OpenCodeFile(path) => {
                    self.open_code_file(path);
                }
                EditorAction::SaveOpenEditor(path) => {
                    if let Some(editor) = self.open_editors.iter_mut().find(|e| e.path == path) {
                        match std::fs::write(&editor.path, &editor.text) {
                            Ok(()) => {
                                editor.dirty = false;
                                editor.disk_mtime = std::fs::metadata(&editor.path)
                                    .ok()
                                    .and_then(|m| m.modified().ok());
                                tracing::info!("saved {}", editor.path.display());
                            }
                            Err(e) => tracing::error!("saving file: {e:#}"),
                        }
                    }
                    if let Some(lsp) = self.lsp.as_ref() {
                        lsp.save(&path);
                    }
                }
                EditorAction::CloseCodeTab(path) => {
                    if let Some(loc) = self.dock_state.find_tab(&Tab::Code(path.clone())) {
                        self.dock_state.remove_tab(loc);
                    }
                    self.open_editors.retain(|e| e.path != path);
                }
                EditorAction::LspCompletion(path, cursor, explicit) => {
                    if let Some(editor) = self.open_editors.iter().find(|e| e.path == path) {
                        let (line, character) = char_to_line_col(&editor.text, cursor);
                        let anchor_char = word_start(&editor.text, cursor);
                        if let Some(lsp) = self.lsp.as_mut() {
                            let id = lsp.completion(&path, line, character);
                            self.lsp_requests
                                .insert(id, LspRequestKind::Completion { path, anchor_char, explicit });
                        }
                    }
                }
                EditorAction::LspHover(path, cursor) => {
                    if let Some(editor) = self.open_editors.iter().find(|e| e.path == path) {
                        let (line, character) = char_to_line_col(&editor.text, cursor);
                        if let Some(lsp) = self.lsp.as_mut() {
                            let id = lsp.hover(&path, line, character);
                            self.lsp_requests.insert(id, LspRequestKind::Hover { path });
                        }
                    }
                }
                EditorAction::LspCodeAction(path, cursor) => {
                    if let Some(editor) = self.open_editors.iter().find(|e| e.path == path) {
                        let (line, character) = char_to_line_col(&editor.text, cursor);
                        if let Some(lsp) = self.lsp.as_mut() {
                            let id = lsp.code_action(&path, line, character);
                            self.lsp_requests.insert(id, LspRequestKind::CodeAction { path });
                        }
                    }
                }
                EditorAction::LspGoto(path, cursor) => {
                    if let Some(editor) = self.open_editors.iter().find(|e| e.path == path) {
                        let (line, character) = char_to_line_col(&editor.text, cursor);
                        if let Some(lsp) = self.lsp.as_mut() {
                            let id = lsp.definition(&path, line, character);
                            self.lsp_requests.insert(id, LspRequestKind::Definition);
                        }
                    }
                }
                EditorAction::LspReferences(path, cursor) => {
                    if let Some(editor) = self.open_editors.iter().find(|e| e.path == path) {
                        let (line, character) = char_to_line_col(&editor.text, cursor);
                        if let Some(lsp) = self.lsp.as_mut() {
                            let id = lsp.references(&path, line, character);
                            self.lsp_requests
                                .insert(id, LspRequestKind::References { path });
                        }
                    }
                }
                EditorAction::OpenAndGoto(path, line, col) => {
                    self.open_code_file(path.clone());
                    if let Some(editor) = self.open_editors.iter_mut().find(|e| e.path == path) {
                        editor.goto = Some((line, col));
                    }
                }
                EditorAction::CreateComponent => {
                    match plugins::create_component(&self.project_root) {
                        Ok(path) => self.actions.push(EditorAction::SelectFile(path)),
                        Err(e) => tracing::error!("creating component: {e:#}"),
                    }
                }
                EditorAction::NewScene => {
                    if let Err(e) = renderer!().reset_scene() {
                        tracing::error!("resetting scene: {e:#}");
                    }
                    self.reset_render_caches();
                    self.scene = LoadedScene::empty();
                    // A new scene starts with the default Directional Light.
                    self.scene.add_default_directional_light();
                    self.selection = Selection::None;
                    self.file_material = None;
                    self.current_scene_path = None;
                    self.undo_stack.clear();
                    self.apply_skybox();
                    self.save_project();
                }
                EditorAction::LoadScene(path) => {
                    // Background load: the editor keeps drawing (with a progress
                    // modal); the install + GPU reset happen when it lands.
                    self.spawn_scene_load(path, true);
                }
                EditorAction::Undo => self.apply_history(true),
                EditorAction::Redo => self.apply_history(false),
                EditorAction::Spawn(source) => {
                    let name = match &source {
                        zestengine_assets::ObjectSource::Empty => "Empty".to_owned(),
                        zestengine_assets::ObjectSource::Camera => "Camera".to_owned(),
                        zestengine_assets::ObjectSource::Primitive { shape } => {
                            shape.label().to_owned()
                        }
                        _ => "Object".to_owned(),
                    };
                    // Place in front of the camera, on the ground-ish plane.
                    let mut position = self.camera.position + self.camera.forward() * 3.0;
                    position.y = position.y.max(0.0);
                    match self.scene.spawn(renderer!(), source, name, position) {
                        Ok(index) => {
                            self.select_object(Some(index));
                            // Co-edit: broadcast the spawn with its id.
                            let o = &self.scene.objects[index];
                            self.edit_send(zestengine_engine::edit_net::EditOp::Spawn(Box::new(zestengine_engine::edit_net::SpawnOp {
                                id: o.id,
                                source: Self::edit_source_label(&o.source),
                                translation: o.translation.to_array(),
                                rotation: o.rotation.to_array(),
                                scale: o.scale.to_array(),
                            })));
                        }
                        Err(e) => tracing::error!("spawning object: {e:#}"),
                    }
                }
                EditorAction::SpawnTerrain => {
                    let mut position = self.camera.position + self.camera.forward() * 3.0;
                    position.y = 0.0;
                    match self.scene.spawn(
                        renderer!(),
                        zestengine_assets::ObjectSource::Empty,
                        "Terrain".to_owned(),
                        position,
                    ) {
                        Ok(index) => {
                            self.scene.objects[index]
                                .components
                                .push(Box::new(zestengine_core::Terrain::default()));
                            self.scene.objects[index]
                                .components
                                .push(Box::new(zestengine_core::TerrainCollider::default()));
                            self.select_object(Some(index));
                        }
                        Err(e) => tracing::error!("spawning terrain: {e:#}"),
                    }
                }
                EditorAction::SpawnLight(kind) => {
                    use zestengine_core::{LightComponent, LightKind};
                    let name = format!("{} Light", kind.label());
                    // Directional lights default to aiming down-ish (a sun);
                    // point/spot spawn in front of the camera like other
                    // objects.
                    let (position, rotation) = match kind {
                        LightKind::Directional => (
                            self.camera.position + self.camera.forward() * 3.0 + Vec3::Y * 3.0,
                            glam::Quat::from_rotation_arc(
                                Vec3::NEG_Z,
                                Vec3::new(-0.4, -1.0, -0.3).normalize(),
                            ),
                        ),
                        _ => {
                            let mut p = self.camera.position + self.camera.forward() * 3.0;
                            p.y = p.y.max(0.5);
                            (p, glam::Quat::IDENTITY)
                        }
                    };
                    match self.scene.spawn(
                        renderer!(),
                        zestengine_assets::ObjectSource::Light,
                        name,
                        position,
                    ) {
                        Ok(index) => {
                            let light = LightComponent {
                                kind,
                                ..LightComponent::default()
                            };
                            self.scene.objects[index].rotation = rotation;
                            self.scene.objects[index].components.push(Box::new(light));
                            self.select_object(Some(index));
                        }
                        Err(e) => tracing::error!("spawning light: {e:#}"),
                    }
                }
                EditorAction::SpawnAreaLight => {
                    use zestengine_core::AreaLight;
                    let mut p = self.camera.position + self.camera.forward() * 3.0;
                    p.y = p.y.max(0.5);
                    match self.scene.spawn(
                        renderer!(),
                        zestengine_assets::ObjectSource::Empty,
                        "Area Light".to_owned(),
                        p,
                    ) {
                        Ok(index) => {
                            // Face the camera: the emitter lights toward -Z.
                            self.scene.objects[index].rotation =
                                glam::Quat::from_rotation_arc(Vec3::NEG_Z, -self.camera.forward());
                            self.scene.objects[index]
                                .components
                                .push(Box::new(AreaLight::default()));
                            self.select_object(Some(index));
                        }
                        Err(e) => tracing::error!("spawning area light: {e:#}"),
                    }
                }
                EditorAction::SpawnProbeVolume => {
                    let mut p = self.camera.position + self.camera.forward() * 4.0;
                    p.y = p.y.max(1.5);
                    match self.scene.spawn(
                        renderer!(),
                        zestengine_assets::ObjectSource::Empty,
                        "Light Probe Volume".to_owned(),
                        p,
                    ) {
                        Ok(index) => {
                            self.scene.objects[index]
                                .components
                                .push(Box::new(zestengine_core::LightProbeVolume::default()));
                            self.select_object(Some(index));
                        }
                        Err(e) => tracing::error!("spawning probe volume: {e:#}"),
                    }
                }
                EditorAction::SpawnReflectionProbe => {
                    let mut p = self.camera.position + self.camera.forward() * 4.0;
                    p.y = p.y.max(1.5);
                    match self.scene.spawn(
                        renderer!(),
                        zestengine_assets::ObjectSource::Empty,
                        "Reflection Probe".to_owned(),
                        p,
                    ) {
                        Ok(index) => {
                            self.scene.objects[index]
                                .components
                                .push(Box::new(zestengine_core::ReflectionProbe::default()));
                            self.select_object(Some(index));
                        }
                        Err(e) => tracing::error!("spawning reflection probe: {e:#}"),
                    }
                }
                EditorAction::CreateCameraScreen(camera) => {
                    let Some(cam) = self.scene.objects.get(camera).and_then(|o| {
                        o.components.iter().find_map(|c| c.as_any().downcast_ref::<zestengine_core::CameraComponent>())
                    }) else {
                        continue;
                    };
                    let size = cam.texture_size;
                    let id = self.scene.objects[camera].id;
                    let name = self.scene.objects[camera].name.clone();
                    // An emissive material showing the feed: black base so scene
                    // lighting doesn't wash the picture out.
                    let path = unique_path(&self.project_root.join("materials"), &format!("{} screen", name.trim()), "material");
                    let mut file = zestengine_assets::MaterialFile {
                        name: path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
                        shader: "standard".into(),
                        params: Default::default(),
                        features: Default::default(),
                        render_queue: None,
                        textures: Default::default(),
                        custom: Default::default(),
                    };
                    file.params.base_color = [0.0, 0.0, 0.0, 1.0];
                    file.params.emission_color = [1.0, 1.0, 1.0];
                    file.params.emission_intensity = 1.0;
                    file.params.roughness = 0.35;
                    file.features.emission = true;
                    file.textures.emission = Some(PathBuf::from(format!("camera:{id}")));
                    if let Some(dir) = path.parent() {
                        let _ = std::fs::create_dir_all(dir);
                    }
                    if let Err(e) = zestengine_assets::save_material_file(&path, &file) {
                        tracing::error!("creating screen material: {e:#}");
                        continue;
                    }
                    // A plane facing the view, sized to the texture's aspect.
                    let forward = self.camera.forward();
                    let position = self.camera.position + forward * 3.0;
                    let normal = -forward;
                    let right = Vec3::Y.cross(normal).normalize_or(Vec3::X);
                    let down = normal.cross(right);
                    let rotation = glam::Quat::from_mat3(&glam::Mat3::from_cols(right, normal, down));
                    let aspect = size[0].max(1) as f32 / size[1].max(1) as f32;
                    match self.scene.spawn(
                        renderer!(),
                        zestengine_assets::ObjectSource::Primitive { shape: zestengine_assets::PrimitiveShape::Plane },
                        format!("{} Screen", name.trim()),
                        position,
                    ) {
                        Ok(index) => {
                            self.scene.objects[index].rotation = rotation;
                            self.scene.objects[index].scale = Vec3::new(0.8 * aspect, 1.0, 0.8);
                            let rel = PathBuf::from(zestengine_engine::scene::relative_to(&path, &self.project_root));
                            self.scene.assign_material(renderer!(), &mut self.shaders, index, 0, &rel, &self.project_root);
                            self.scene_dirty = true;
                            self.select_object(Some(index));
                        }
                        Err(e) => tracing::error!("spawning screen: {e:#}"),
                    }
                }
                EditorAction::SpawnFluxVolume => {
                    let mut p = self.camera.position + self.camera.forward() * 4.0;
                    p.y = p.y.max(1.5);
                    match self.scene.spawn(
                        renderer!(),
                        zestengine_assets::ObjectSource::Empty,
                        "Flux Volume".to_owned(),
                        p,
                    ) {
                        Ok(index) => {
                            self.scene.objects[index]
                                .components
                                .push(Box::new(zestengine_core::FluxVolume::default()));
                            self.select_object(Some(index));
                        }
                        Err(e) => tracing::error!("spawning flux volume: {e:#}"),
                    }
                }
                EditorAction::SpawnPostFxVolume(global) => {
                    let p = self.camera.position + self.camera.forward() * 4.0;
                    match self.scene.spawn(
                        renderer!(),
                        zestengine_assets::ObjectSource::Empty,
                        "Post FX Volume".to_owned(),
                        p,
                    ) {
                        Ok(index) => {
                            // A fresh profile so the volume works straight away
                            // (all effects off: it overrides nothing until enabled).
                            let dir = self.project_root.join("postfx");
                            let mut n = 1;
                            let path = loop {
                                let name = if n == 1 { "Post FX Volume.postfx".to_owned() } else { format!("Post FX Volume ({n}).postfx") };
                                let p = dir.join(name);
                                if !p.exists() {
                                    break p;
                                }
                                n += 1;
                            };
                            let mut volume = zestengine_core::VolumeComponent { global, ..Default::default() };
                            match zestengine_assets::save_postfx(&path, &zestengine_assets::PostFxProfile::default()) {
                                Ok(()) => {
                                    volume.profile = path
                                        .strip_prefix(&self.project_root)
                                        .unwrap_or(&path)
                                        .to_string_lossy()
                                        .replace('\\', "/");
                                }
                                Err(e) => tracing::error!("creating post fx profile: {e:#}"),
                            }
                            self.scene.objects[index].components.push(Box::new(volume));
                            self.select_object(Some(index));
                            self.scene_dirty = true;
                        }
                        Err(e) => tracing::error!("spawning post fx volume: {e:#}"),
                    }
                }
                EditorAction::SpawnAudioVolume => {
                    let p = self.camera.position + self.camera.forward() * 3.0;
                    match self.scene.spawn(renderer!(), zestengine_assets::ObjectSource::Empty, "Audio Volume".to_owned(), p) {
                        Ok(index) => {
                            let path = unique_path(&self.project_root.join("audio"), "Audio Volume", zestengine_assets::AUDIOFX_EXTENSION);
                            let mut volume = zestengine_core::AudioVolume::default();
                            match zestengine_assets::save_audiofx(&path, &Default::default()) {
                                Ok(()) => {
                                    let rel = path.strip_prefix(&self.project_root).unwrap_or(&path).to_string_lossy().replace('\\', "/");
                                    volume.profile = zestengine_core::asset_ref::reference(&self.project_root, &rel);
                                }
                                Err(e) => tracing::error!("creating audio environment profile: {e:#}"),
                            }
                            self.scene.objects[index].components.push(Box::new(volume));
                            self.select_object(Some(index));
                        }
                        Err(e) => tracing::error!("spawning audio volume: {e:#}"),
                    }
                }
                EditorAction::SpawnAudioSource => {
                    let p = self.camera.position + self.camera.forward() * 3.0;
                    match self.scene.spawn(
                        renderer!(),
                        zestengine_assets::ObjectSource::Empty,
                        "Audio Source".to_owned(),
                        p,
                    ) {
                        Ok(index) => {
                            self.scene.objects[index]
                                .components
                                .push(Box::new(zestengine_core::AudioSource::default()));
                            self.select_object(Some(index));
                        }
                        Err(e) => tracing::error!("spawning audio source: {e:#}"),
                    }
                }
                EditorAction::SpawnBoxCollider => {
                    let p = self.camera.position + self.camera.forward() * 3.0;
                    match self.scene.spawn(
                        renderer!(),
                        zestengine_assets::ObjectSource::Empty,
                        "Box Collider".to_owned(),
                        p,
                    ) {
                        Ok(index) => {
                            self.scene.objects[index]
                                .components
                                .push(Box::new(zestengine_core::BoxCollider::default()));
                            self.select_object(Some(index));
                        }
                        Err(e) => tracing::error!("spawning box collider: {e:#}"),
                    }
                }
                EditorAction::SpawnSphereCollider => {
                    let p = self.camera.position + self.camera.forward() * 3.0;
                    match self.scene.spawn(
                        renderer!(),
                        zestengine_assets::ObjectSource::Empty,
                        "Sphere Collider".to_owned(),
                        p,
                    ) {
                        Ok(index) => {
                            self.scene.objects[index]
                                .components
                                .push(Box::new(zestengine_core::SphereCollider::default()));
                            self.select_object(Some(index));
                        }
                        Err(e) => tracing::error!("spawning sphere collider: {e:#}"),
                    }
                }
                EditorAction::SpawnCapsuleCollider => {
                    let p = self.camera.position + self.camera.forward() * 3.0;
                    match self.scene.spawn(renderer!(), zestengine_assets::ObjectSource::Empty, "Capsule Collider".to_owned(), p) {
                        Ok(index) => {
                            self.scene.objects[index].components.push(Box::new(zestengine_core::CapsuleCollider::default()));
                            self.select_object(Some(index));
                        }
                        Err(e) => tracing::error!("spawning capsule collider: {e:#}"),
                    }
                }
                EditorAction::SpawnWater => {
                    // In front of the camera, surface at ground level.
                    let ahead = self.camera.position + self.camera.forward() * 8.0;
                    let p = Vec3::new(ahead.x, 0.0, ahead.z);
                    match self.scene.spawn(renderer!(), zestengine_assets::ObjectSource::Empty, "Water".to_owned(), p) {
                        Ok(index) => {
                            self.scene.objects[index].components.push(Box::new(zestengine_core::Water::default()));
                            self.select_object(Some(index));
                        }
                        Err(e) => tracing::error!("spawning water: {e:#}"),
                    }
                }
                EditorAction::SpawnFloatingCube => {
                    let p = self.camera.position + self.camera.forward() * 5.0 + Vec3::Y * 2.0;
                    let source = zestengine_assets::ObjectSource::Primitive { shape: zestengine_assets::PrimitiveShape::Cube };
                    match self.scene.spawn(renderer!(), source, "Floating Cube".to_owned(), p) {
                        Ok(index) => {
                            let c = &mut self.scene.objects[index].components;
                            c.push(Box::new(zestengine_core::BoxCollider::default()));
                            // Half the density of water: floats half under.
                            c.push(Box::new(zestengine_core::RigidBody { mass: 500.0, ..Default::default() }));
                            self.select_object(Some(index));
                        }
                        Err(e) => tracing::error!("spawning floating cube: {e:#}"),
                    }
                }
                EditorAction::SpawnBucket => {
                    let p = self.camera.position + self.camera.forward() * 2.0;
                    let source = zestengine_assets::ObjectSource::Primitive { shape: zestengine_assets::PrimitiveShape::Bucket };
                    match self.scene.spawn(renderer!(), source, "Bucket".to_owned(), p) {
                        Ok(index) => {
                            let o = &mut self.scene.objects[index];
                            // About 19 L inside; 5 kg, so it floats empty and
                            // goes under when brimful.
                            o.scale = Vec3::new(0.28, 0.3, 0.28);
                            let (w, f) = (zestengine_assets::BUCKET_WALL, zestengine_assets::BUCKET_FLOOR);
                            let wall = |center: [f32; 3], size: [f32; 3]| -> Box<dyn zestengine_core::Component> {
                                Box::new(zestengine_core::BoxCollider { center, size, ..Default::default() })
                            };
                            let (side, inner) = (0.5 - w * 0.5, 1.0 - 2.0 * w);
                            o.components.push(wall([side, 0.0, 0.0], [w, 1.0, 1.0]));
                            o.components.push(wall([-side, 0.0, 0.0], [w, 1.0, 1.0]));
                            o.components.push(wall([0.0, 0.0, side], [inner, 1.0, w]));
                            o.components.push(wall([0.0, 0.0, -side], [inner, 1.0, w]));
                            o.components.push(wall([0.0, -0.5 + f * 0.5, 0.0], [inner, f, inner]));
                            o.components.push(Box::new(zestengine_core::RigidBody { mass: 5.0, ..Default::default() }));
                            o.components.push(Box::new(zestengine_core::WaterContainer::default()));
                            o.components.push(Box::new(zestengine_core::Grabbable::default()));
                            self.select_object(Some(index));
                        }
                        Err(e) => tracing::error!("spawning bucket: {e:#}"),
                    }
                }
                EditorAction::SpawnWaterfall | EditorAction::SpawnWhirlpool => {
                    let fall = matches!(action, EditorAction::SpawnWaterfall);
                    let fwd = self.camera.forward();
                    let ahead = self.camera.position + fwd * 8.0;
                    let p = if fall { Vec3::new(ahead.x, 4.0, ahead.z) } else { Vec3::new(ahead.x, 0.0, ahead.z) };
                    let name = if fall { "Waterfall" } else { "Whirlpool" };
                    match self.scene.spawn(renderer!(), zestengine_assets::ObjectSource::Empty, name.to_owned(), p) {
                        Ok(index) => {
                            let o = &mut self.scene.objects[index];
                            if fall {
                                // Pour toward the camera.
                                let back = Vec3::new(-fwd.x, 0.0, -fwd.z).normalize_or(Vec3::Z);
                                o.rotation = glam::Quat::from_rotation_arc(Vec3::Z, back);
                                o.components.push(Box::new(zestengine_core::Waterfall::default()));
                            } else {
                                o.components.push(Box::new(zestengine_core::Whirlpool::default()));
                            }
                            self.select_object(Some(index));
                        }
                        Err(e) => tracing::error!("spawning {name}: {e:#}"),
                    }
                }
                EditorAction::SpawnWindZone => {
                    let ahead = self.camera.position + self.camera.forward() * 6.0;
                    match self.scene.spawn(renderer!(), zestengine_assets::ObjectSource::Empty, "Wind Zone".to_owned(), ahead) {
                        Ok(index) => {
                            self.scene.objects[index].components.push(Box::new(zestengine_core::ClothWind::default()));
                            self.select_object(Some(index));
                        }
                        Err(e) => tracing::error!("spawning Wind Zone: {e:#}"),
                    }
                }
                EditorAction::AddMeshCollider => {
                    let targets: Vec<usize> = match self.selection {
                        Selection::Object(i) if self.multi_objects.len() > 1 && self.multi_objects.contains(&i) => self.multi_objects.clone(),
                        Selection::Object(i) => vec![i],
                        _ => Vec::new(),
                    };
                    let mut added = 0;
                    let count = self.scene.objects.len();
                    for i in targets.into_iter().filter(|&i| i < count) {
                        let o = &mut self.scene.objects[i];
                        if o.render.is_some() && !o.components.iter().any(|c| c.as_any().is::<zestengine_core::MeshCollider>()) {
                            o.components.push(Box::new(zestengine_core::MeshCollider::default()));
                            added += 1;
                        }
                    }
                    if added == 0 {
                        self.set_status("Select one or more mesh objects to add a Mesh Collider to", false);
                    } else {
                        self.scene_dirty = true;
                    }
                }
                EditorAction::SetParent(child, parent) => {
                    let before = self.scene.snapshot_objects();
                    self.scene.set_parent(child, parent);
                    // Moved out of a bone: no longer follows it (the transform stays).
                    if child < self.scene.objects.len() && self.scene.objects[child].parent == parent {
                        self.scene.detach_bone_attach(child);
                    }
                    self.record_structure(before);
                }
                EditorAction::AttachToBone { child, owner, bone } => {
                    let before = self.scene.snapshot_objects();
                    match self.scene.attach_to_bone(child, owner, &bone) {
                        Ok(()) => {
                            self.record_structure(before);
                            self.scene_dirty = true;
                            self.set_status(format!("Attached to {bone}"), false);
                        }
                        Err(e) => self.set_status(format!("Cannot attach: {e}"), false),
                    }
                }
                EditorAction::GroupIntoEmpty(members) => {
                    // Structural, like delete and reorder: indices shift, so the
                    // new group becomes the selection and undo diffing skips it.
                    let before = self.scene.snapshot_objects();
                    if let Some(group) = self.scene.group_into_empty(&members) {
                        self.record_structure(before);
                        self.selection = Selection::Object(group);
                        self.multi_objects = vec![group];
                        self.inspector_lock_target = None;
                        self.suppress_undo_record = true;
                        self.scene_dirty = true;
                    }
                }
                EditorAction::AddToggle(index) => self.add_avatar_toggle(index),
                EditorAction::MoveObject(child, parent, before) => {
                    let snapshot = self.scene.snapshot_objects();
                    let map = self.scene.reorder_object(child, parent, before);
                    if !map.is_empty() {
                        // Dragged out of a bone: it no longer follows it.
                        self.scene.detach_bone_attach(map[child]);
                        self.record_structure(snapshot);
                        // Object indices shifted; remap selection + lock target,
                        // and keep this structural change out of undo diffing.
                        let remap = |sel: &Selection| match sel {
                            Selection::Object(i) if *i < map.len() => Selection::Object(map[*i]),
                            other => other.clone(),
                        };
                        #[allow(clippy::redundant_closure_call)]
                        let new_selection = remap(&self.selection);
                        self.selection = new_selection;
                        self.inspector_lock_target = self.inspector_lock_target.as_ref().map(remap);
                        self.suppress_undo_record = true;
                    }
                }
                EditorAction::DeleteObject(index) => {
                    let id = self.scene.objects.get(index).map(|o| o.id);
                    let before = self.scene.snapshot_objects();
                    self.scene.remove_object(index);
                    self.record_structure(before);
                    if let Some(id) = id {
                        self.edit_send(zestengine_engine::edit_net::EditOp::Delete { id });
                    }
                    // Indices shifted; drop selection/lock to stay safe.
                    self.selection = Selection::None;
                    self.inspector_lock_target = None;
                    self.inspector.locked = false;
                }
                EditorAction::RenameObject(index, name) => {
                    if index < self.scene.objects.len() {
                        self.scene.objects[index].name = name.clone();
                        self.edit_send(zestengine_engine::edit_net::EditOp::Rename { id: self.scene.objects[index].id, name });
                    }
                }
                EditorAction::SpawnUi(preset) => {
                    if let Err(e) = self.spawn_ui(preset) {
                        tracing::error!("spawning UI {}: {e:#}", preset.label());
                    }
                }
                EditorAction::SpawnUiTemplate(template) => {
                    if let Err(e) = self.spawn_ui_template(template) {
                        tracing::error!("spawning UI template {}: {e:#}", template.label());
                    }
                }
                EditorAction::SetCanvasMode(index, mode) => self.set_canvas_mode(index, mode),
                EditorAction::SceneView(req) => self.apply_view_request(req),
                EditorAction::RecordEnabled(items) => {
                    self.undo_stack.record(UndoEntry::Enabled(items), false);
                    self.suppress_undo_record = true;
                    self.scene_dirty = true;
                }
                EditorAction::EnvironmentEdited { before, gesture } => {
                    let after = Box::new(self.scene.environment.clone());
                    self.undo_stack.record(UndoEntry::Environment { before, after }, gesture);
                }
                EditorAction::BakeAudio => self.start_audio_bake(),
                EditorAction::BakeOcclusion => self.start_occlusion_bake(),
                EditorAction::ClearOcclusion => self.clear_occlusion(),
                EditorAction::SnapView(dir) => {
                    let focus = match self.selection {
                        Selection::Object(i) if i < self.scene.objects.len() => self.scene.world_transform(i).w_axis.truncate(),
                        _ => self.camera.position + self.camera.forward() * 10.0,
                    };
                    let dist = (focus - self.camera.position).length().max(2.0);
                    let d = dir.normalize_or(Vec3::NEG_Z);
                    self.camera.pitch = d.y.clamp(-0.999, 0.999).asin().clamp(-1.55, 1.55);
                    if d.x.abs() + d.z.abs() > 1e-4 {
                        self.camera.yaw = d.z.atan2(d.x);
                    }
                    self.camera.position = focus - self.camera.forward() * dist;
                }
                EditorAction::Prefab(action) => self.run_prefab_action(action),
                EditorAction::InstantiatePrefab(path, pos) => {
                    let ray = pos
                        .and_then(|p| self.cursor_ray(p))
                        .unwrap_or((self.camera.position, self.camera.forward()));
                    let at = self.drop_point(ray.0, ray.1);
                    // Structural like duplicate: one undo step; the snapshots
                    // carry object ids, so redo restores the same ids.
                    let before = self.scene.snapshot_objects();
                    // A linked instance: the root names the asset, expanding it
                    // builds the objects, and edits to the asset reach them.
                    let result = zestengine_engine::prefab::Prefab::load(&path).and_then(|prefab| {
                        let name = prefab.roots().first().map_or_else(|| "Prefab".to_owned(), |&r| prefab.entries[r].name.clone());
                        let mut entries = vec![zestengine_engine::prefab::link_root(&zestengine_engine::prefab::link_path(&self.project_root, &path), &name, at.to_array())];
                        let cache = &self.scene.prefabs;
                        zestengine_assets::expand_prefabs(&mut entries, &|p| cache.get(&self.project_root, p));
                        if entries.len() == 1 && entries[0].prefab.as_ref().is_some_and(|l| !l.expanded || cache.get(&self.project_root, &l.path).is_none()) {
                            anyhow::bail!("{} could not be read", path.display());
                        }
                        self.scene.append_entries(
                            renderer!(),
                            &mut self.shaders,
                            &self.project_root,
                            &self.components,
                            &entries,
                        )
                    });
                    match result {
                        Ok(root) => {
                            self.record_structure(before);
                            self.select_object(Some(root));
                        }
                        Err(e) => self.set_load_status(&format!("Prefab placement failed: {e:#}")),
                    }
                }
                EditorAction::DuplicateObject(index) => {
                    // Structural like delete: one undo step. Select the new copy.
                    let (id, before) = (
                        self.scene.objects.get(index).map(|o| o.id),
                        self.scene.snapshot_objects(),
                    );
                    if let Some(new_root) =
                        self.scene.duplicate_object_linked(index, &self.project_root, &self.shaders, &self.components)
                    {
                        self.scene.copy_view_flags(index, new_root);
                        self.record_structure(before);
                        if let Some(id) = id {
                            self.edit_send(zestengine_engine::edit_net::EditOp::Duplicate { id, new_id: self.scene.objects[new_root].id });
                        }
                        self.selection = Selection::Object(new_root);
                        self.prefab_editor.invalidate();
                    }
                }
                EditorAction::CreatePrefabVariant(base) => match zestengine_engine::prefab::create_variant(&self.project_root, &base) {
                    Ok(path) => {
                        tracing::info!("prefab variant {}", path.display());
                        self.set_status(format!("Created {}", path.file_name().map(|n| n.to_string_lossy()).unwrap_or_default()), false);
                        self.actions.push(EditorAction::SelectFile(path));
                    }
                    Err(e) => self.tasks.notify(format!("Creating the variant failed: {e:#}"), crate::tasks::NotifyLevel::Warn),
                },
                EditorAction::ShowInFileManager(path) => {
                    if let Err(e) = zestengine_assets::launcher::show_in_file_manager(&path) {
                        self.set_status(format!("Could not open the file manager: {e}"), false);
                    }
                }
                EditorAction::ImportLayered(src) => {
                    let ext = src.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).unwrap_or_default();
                    let r = if matches!(ext.as_str(), "psd" | "psb") {
                        zestengine_assets::layered_import::import_psd_file(&self.project_root, &src, &zestengine_assets::psd::ImportOptions { rig_seed: true, ..Default::default() })
                    } else {
                        zestengine_assets::layered_import::import_aseprite_file(&self.project_root, &src, &zestengine_assets::aseprite::ImportOptions::default())
                    };
                    match r {
                        Ok(files) => {
                            tracing::info!("imported {} -> {}", src.display(), files.sheet.display());
                            self.set_status(format!("Imported {} sprites and {} clips", src.file_name().map(|n| n.to_string_lossy()).unwrap_or_default(), files.clips.len()), false);
                            self.actions.push(EditorAction::SelectFile(files.sheet));
                        }
                        Err(e) => self.tasks.notify(format!("Importing {} failed: {e:#}", src.display()), crate::tasks::NotifyLevel::Warn),
                    }
                }
                EditorAction::LayersPrefab(layers, instance) => match zestengine_engine::prefab::create_from_layers(&self.project_root, &layers) {
                    Ok((path, st)) => {
                        let what = format!("{} sprites, {} bones, {} skinned", st.sprites, st.bones, st.skinned);
                        if instance {
                            self.set_status(format!("Created {} ({what}) and placed an instance", path.file_name().map(|n| n.to_string_lossy()).unwrap_or_default()), false);
                            self.actions.push(EditorAction::InstantiatePrefab(path, None));
                        } else {
                            self.set_status(format!("Created {} ({what})", path.file_name().map(|n| n.to_string_lossy()).unwrap_or_default()), false);
                            self.actions.push(EditorAction::SelectFile(path));
                        }
                    }
                    Err(e) => self.tasks.notify(format!("Creating the prefab from {} failed: {e:#}", layers.display()), crate::tasks::NotifyLevel::Warn),
                },
                EditorAction::DuplicateFile(path) => {
                    if let Some(dest) = duplicate_file_path(&path) {
                        match std::fs::copy(&path, &dest) {
                            Ok(_) => {
                                tracing::info!(
                                    "duplicated {} -> {}",
                                    path.display(),
                                    dest.display()
                                );
                                self.actions.push(EditorAction::SelectFile(dest));
                            }
                            Err(e) => tracing::error!("duplicating file: {e:#}"),
                        }
                    }
                }
                EditorAction::DeleteFile(path) => {
                    let result = if path.is_dir() {
                        std::fs::remove_dir_all(&path)
                    } else {
                        std::fs::remove_file(&path)
                    };
                    match result {
                        Ok(()) => {
                            zestengine_core::asset_ref::deleted(&path);
                            tracing::info!("deleted {}", path.display());
                            if self.selection == Selection::File(path.clone()) {
                                self.selection = Selection::None;
                            }
                            if self
                                .file_material
                                .as_ref()
                                .is_some_and(|fm| fm.path == path)
                            {
                                self.file_material = None;
                            }
                            self.open_editors.retain(|e| e.path != path);
                        }
                        Err(e) => tracing::error!("deleting {}: {e:#}", path.display()),
                    }
                }
                EditorAction::UnityConvert(job) => {
                    crate::unity_convert_ui::spawn(&mut self.tasks, job, &self.project_root);
                }
                EditorAction::ImportPack(source) => self.pack_import.open(source),
                EditorAction::SetSkybox(path) => {
                    let before = self.skybox_state();
                    let rel = path
                        .strip_prefix(&self.project_root)
                        .unwrap_or(&path)
                        .to_string_lossy()
                        .into_owned();
                    self.scene.skybox = Some(rel);
                    self.record_skybox(before);
                    self.apply_skybox();
                }
                EditorAction::ClearSkybox => {
                    let before = self.skybox_state();
                    self.scene.skybox = None;
                    self.record_skybox(before);
                    self.apply_skybox();
                }
                EditorAction::SetSkyboxFace(i, path) => {
                    let before = self.skybox_state();
                    let rel = path
                        .strip_prefix(&self.project_root)
                        .unwrap_or(&path)
                        .to_string_lossy()
                        .into_owned();
                    let faces = self
                        .scene
                        .environment
                        .skybox_faces
                        .get_or_insert_with(|| std::array::from_fn(|_| String::new()));
                    if i < 6 {
                        faces[i] = rel;
                    }
                    self.record_skybox(before);
                    self.apply_skybox();
                }
                EditorAction::ClearSkyboxFaces => {
                    let before = self.skybox_state();
                    self.scene.environment.skybox_faces = None;
                    self.record_skybox(before);
                    self.apply_skybox();
                }
                EditorAction::TogglePlay => self.toggle_play(),
                EditorAction::CreatePluginCrate => {
                    match plugins::create_template(&self.project_root) {
                        Ok(dir) => tracing::info!("component plugin at {}", dir.display()),
                        Err(e) => tracing::error!("creating plugin crate: {e:#}"),
                    }
                }
                EditorAction::SaveEditorPrefs => self.save_editor_prefs(),
                EditorAction::OpenInExternalEditor(path) => {
                    // ZESTENGINE_NVIM_CMD overrides the terminal + command; the
                    // default spawns the user's $TERMINAL with nvim. The file's
                    // CRATE root (the nearest Cargo.toml, e.g. the components
                    // directory) opens instead of the bare file, so nvim loads
                    // it as a project (its LSP + build context work).
                    let mut crate_dir = path.parent().map(std::path::Path::to_path_buf);
                    let mut probe = path.as_path();
                    while let Some(dir) = probe.parent() {
                        if dir.join("Cargo.toml").is_file() {
                            crate_dir = Some(dir.to_path_buf());
                            break;
                        }
                        probe = dir;
                    }
                    let target = crate_dir.unwrap_or_else(|| path.clone());
                    let cmd = std::env::var("ZESTENGINE_NVIM_CMD").unwrap_or_else(|_| {
                        let term = std::env::var("TERMINAL").unwrap_or_else(|_| "kitty".into());
                        format!("{term} -e nvim")
                    });
                    let mut parts = cmd.split_whitespace();
                    let Some(prog) = parts.next() else {
                        self.set_status("no external editor command", false);
                        return;
                    };
                    match std::process::Command::new(prog)
                        .args(parts)
                        .arg(&target)
                        .stdin(std::process::Stdio::null())
                        .stdout(std::process::Stdio::null())
                        .stderr(std::process::Stdio::null())
                        .spawn()
                    {
                        Ok(_) => self.set_status("opened in external editor", true),
                        Err(e) => {
                            self.set_status(format!("external editor failed: {e}"), false);
                            tracing::error!("spawning external editor: {e:#}");
                        }
                    }
                }
                EditorAction::ReloadPlugins => {
                    self.request_plugin_build();
                }
                EditorAction::BakeLighting => self.start_bake(),
                EditorAction::RecompressBake => self.recompress_bake(),
                EditorAction::ClearBake => {
                    self.scene.baked = None;
                    self.bake_sizes = None;
                    self.upload_baked_probes();
                    tracing::info!("cleared baked lighting");
                }
                EditorAction::RegenerateLightmapUvs => {
                    // Markers already exist; reloading re-imports each model and
                    // re-runs the (current) unwrapper.
                    if self.playing {
                        self.set_status("Stop play mode to regenerate UVs", false);
                    } else if self.scene_dirty {
                        self.set_status("Save the scene first to regenerate UVs", false);
                    } else if let Some(scene) = self.current_scene_path.clone() {
                        self.load_scene_runtime(&scene);
                        self.set_status("Regenerated lightmap UVs", false);
                    }
                }
                EditorAction::GenerateAllLightmapUvs => {
                    let models = self.scene.models_needing_lightmap_uv();
                    for m in &models {
                        if let Err(e) =
                            zestengine_assets::set_lightmap_uv_marker(self.project_root.join(m), true)
                        {
                            tracing::error!("writing lightmap-UV marker for {m}: {e:#}");
                        }
                    }
                    if self.playing {
                        self.set_status("Stop play mode, then reload", false);
                    } else if self.scene_dirty {
                        self.set_status("Save the scene first, then reload", false);
                    } else if let Some(scene) = self.current_scene_path.clone() {
                        self.load_scene_runtime(&scene);
                        self.set_status(
                            format!("Generated lightmap UVs for {} model(s)", models.len()),
                            false,
                        );
                    }
                }
                EditorAction::SetLightmapUvGen(path, on) => {
                    if let Err(e) =
                        zestengine_assets::set_lightmap_uv_marker(self.project_root.join(&path), on)
                    {
                        tracing::error!("setting lightmap-UV marker for {path}: {e:#}");
                    }
                    if self.playing {
                        self.set_status("Stop play mode, then reload", false);
                    } else if self.scene_dirty {
                        self.set_status("Save the scene first, then reload", false);
                    } else if let Some(scene) = self.current_scene_path.clone() {
                        self.load_scene_runtime(&scene);
                        self.set_status(
                            if on { "Generated lightmap UVs" } else { "Reverted to model UVs" },
                            false,
                        );
                    }
                }
                EditorAction::NewProject { parent, name, template } => {
                    self.set_status(format!("creating project {name}…"), true);
                    match bundle::scaffold_project(&parent, &name, &template) {
                        Ok(root) => {
                            tracing::info!("scaffolded project at {}", root.display());
                            self.request_switch_project(root);
                        }
                        Err(e) => {
                            tracing::error!("creating project: {e:#}");
                            self.set_status(format!("new project failed: {e}"), false);
                        }
                    }
                }
                EditorAction::OpenProject(root) => {
                    if root.join("project.zestengine").is_file() {
                        tracing::info!("opening project {}", root.display());
                        self.request_switch_project(root);
                    } else {
                        self.set_status(
                            format!("no project.zestengine in {}", root.display()),
                            false,
                        );
                    }
                }
                EditorAction::OpenProjectHub => self.launch_project_hub(),
                EditorAction::BuildGame(target) => {
                    if self.building_game {
                        self.tasks.notify("A game build is already running", crate::tasks::NotifyLevel::Info);
                        continue;
                    }
                    // Persist settings first so the build picks up boot_scene.
                    self.save_project();
                    self.building_game = true;
                    self.build_status = Some(format!("building for {}…", target.label()));
                    let project_root = self.project_root.clone();
                    let project = self.project.clone();
                    self.tasks.spawn(
                        format!("Build game ({})", target.label()),
                        crate::tasks::TaskKind::BuildGame,
                        false,
                        move |_cancel, progress| {
                            let result = bundle::build_game(&project_root, &project, target, |msg| {
                                tracing::info!("build: {msg}");
                                if let Ok(mut p) = progress.lock() {
                                    *p = crate::tasks::TaskProgress::Fraction { done: 0.0, label: msg };
                                }
                            });
                            Ok(crate::tasks::TaskPayload::GameBuilt(result.map_err(|e| format!("{e:#}"))))
                        },
                    );
                }
                // While the viewport shows the game, right mouse and the wheel
                // belong to the game (RTS orders, zoom), not the fly camera.
                EditorAction::StartLook if self.game_rect() == Some(self.viewport_rect) => {}
                EditorAction::StartLook => {
                    self.egui_ctx.memory_mut(|m| m.stop_text_input());
                    self.set_looking(true);
                }
                EditorAction::Dolly(amount) if self.game_rect() == Some(self.viewport_rect) => self.input.wheel(amount),
                // Only reached when the Scene view shows the game camera but the Camera
                // tab owns the game rect (the fly camera is not on screen then), or by the
                // probe's `action-dolly-stale` A/B mode, which reproduces the old
                // post-pass dolly on purpose. The real scroll dolly is in `viewport_ui`.
                EditorAction::Dolly(amount) => self.camera.dolly(amount),
                EditorAction::SetOrtho(on) => self.camera.ortho = on,
                EditorAction::FocusSelected => {
                    if let Selection::Object(i) = self.selection {
                        let (center, radius) = self.scene.object_bounds(i);
                        self.camera.focus(center, radius);
                    }
                }
                EditorAction::SaveScene(_) if self.prefab_editor.stage.active() => self.save_prefab_stage(),
                EditorAction::SaveScene(path) => {
                    // Play changes the scene (physics, scripts); saving would
                    // keep play-time positions. Stop first.
                    if self.playing {
                        self.set_status("Can't save while playing: stop Play first (changes made in Play aren't kept).", false);
                        continue;
                    }
                    let path = path
                        .or_else(|| self.current_scene_path.clone())
                        .unwrap_or_else(|| self.project_root.join(&self.scene_name_input));
                    // Materials are NOT rewritten on scene save; only manual
                    // edits persist (tracked in `dirty_materials`, flushed by
                    // `autosave_materials`, or saved via the material editor).
                    // A material with an existing `.material` file serializes as
                    // a file reference; one without stays inline in the scene.
                    self.avatar_preview.restore(&mut self.scene);
                    let mut file = self.scene.to_compact_scene_file(&self.project_root, &self.shaders);
                    // Persist the current editor viewpoint so the scene reopens
                    // framed the same way.
                    file.editor_camera = Some(zestengine_assets::EditorCamera {
                        position: self.camera.position.to_array(),
                        yaw: self.camera.yaw,
                        pitch: self.camera.pitch,
                    });
                    // Persist the scene-tree collapsed state (by stable object id)
                    // so the hierarchy reopens with the same rows expanded.
                    file.collapsed = self
                        .scene_panel
                        .collapsed_indices()
                        .filter_map(|i| self.scene.objects.get(i).map(|o| o.id.to_string()))
                        .collect();
                    (file.scene_hidden, file.scene_locked) = self.scene.view_ids();
                    match zestengine_assets::save_scene_file(&path, &file) {
                        Ok(()) => {
                            tracing::info!("scene saved to {}", path.display());
                            self.current_scene_path = Some(path);
                            self.scene_dirty = false;
                            self.save_project();
                        }
                        Err(e) => tracing::error!("saving scene: {e:#}"),
                    }
                }
            }
        }
    }

    /// Ctrl K: list the editor's commands and run the picked one.
    fn command_palette_ui(&mut self, ctx: &egui::Context) {
        use crate::command_palette::Command;
        use zestengine_assets::{ObjectSource, PrimitiveShape};
        if !self.command_palette.is_open() {
            return;
        }
        let mut cmds: Vec<Command<PaletteCmd>> = Vec::new();
        let mut add = |label: &str, group: &'static str, id: PaletteCmd| {
            cmds.push(Command { label: label.to_owned(), group, id });
        };
        add(if self.playing { "Stop" } else { "Play" }, "Play", PaletteCmd::TogglePlay);
        add("Save scene", "File", PaletteCmd::SaveScene);
        add("New scene", "File", PaletteCmd::NewScene);
        add("Build game", "File", PaletteCmd::BuildGame);
        add("Open Project Hub", "File", PaletteCmd::ProjectHub);
        add("New project", "File", PaletteCmd::NewProject);
        add("Open project", "File", PaletteCmd::OpenProject);
        if !cfg!(windows) {
            add("Build game for Windows", "File", PaletteCmd::BuildGameWindows);
        }
        add("Undo", "Edit", PaletteCmd::Undo);
        add("Redo", "Edit", PaletteCmd::Redo);
        add("Duplicate selection", "Edit", PaletteCmd::Duplicate);
        add("Focus selection", "View", PaletteCmd::Focus);
        add("Hide / show selection in scene view (H)", "View", PaletteCmd::HideSelection);
        add("Show all hidden objects (Shift+H)", "View", PaletteCmd::ShowAllHidden);
        add("Unlock all objects", "View", PaletteCmd::UnlockAll);
        add("Reset layout", "View", PaletteCmd::ResetLayout);
        add("Toggle game UI overlay", "View", PaletteCmd::ToggleGameUi);
        add("Show / hide gizmos and billboards", "View", PaletteCmd::ToggleGizmos);
        add("Show / hide the Scene view grid", "View", PaletteCmd::ToggleGrid);
        add("Toggle VR (OpenXR)", "View", PaletteCmd::ToggleVr);
        for (k, (_, label)) in PALETTE_TABS.iter().enumerate() {
            add(&format!("Open {label}"), "Windows", PaletteCmd::OpenTab(k));
        }
        add("Project settings", "Windows", PaletteCmd::ProjectSettings);
        add("Layers", "Windows", PaletteCmd::Layers);
        add("Input bindings", "Windows", PaletteCmd::InputBindings);
        add("Audio mixer", "Windows", PaletteCmd::AudioMixer);
        add("Bake lighting", "Lighting", PaletteCmd::BakeLighting);
        add("Bake audio", "Audio", PaletteCmd::BakeAudio);
        add("Reload component plugins", "Tools", PaletteCmd::ReloadPlugins);
        for shape in [PrimitiveShape::Cube, PrimitiveShape::Sphere, PrimitiveShape::Capsule, PrimitiveShape::Plane] {
            add(&format!("Create {}", shape.label()), "Object", PaletteCmd::Spawn(shape));
        }
        for (kind, label) in [
            (zestengine_core::LightKind::Directional, "Directional Light"),
            (zestengine_core::LightKind::Point, "Point Light"),
            (zestengine_core::LightKind::Spot, "Spot Light"),
        ] {
            add(&format!("Create {label}"), "Object", PaletteCmd::SpawnLight(kind));
        }
        add("Create Area Light", "Object", PaletteCmd::SpawnAreaLight);
        for p in crate::UiPreset::ALL {
            add(&format!("Create UI {}", p.label()), "UI", PaletteCmd::UiPreset(p));
        }
        for t in zestengine_engine::ui_templates::UiTemplate::ALL {
            add(&format!("Add UI template: {}", t.label()), "UI", PaletteCmd::UiTemplate(t));
        }
        let Some(cmd) = self.command_palette.ui(ctx, &cmds) else {
            return;
        };
        match cmd {
            PaletteCmd::TogglePlay => self.actions.push(EditorAction::TogglePlay),
            PaletteCmd::SaveScene => self.actions.push(EditorAction::SaveScene(None)),
            PaletteCmd::NewScene => self.actions.push(EditorAction::NewScene),
            PaletteCmd::BuildGame => self.actions.push(EditorAction::BuildGame(bundle::BuildTarget::Host)),
            PaletteCmd::ProjectHub => self.actions.push(EditorAction::OpenProjectHub),
            PaletteCmd::NewProject => {
                if self.new_project_parent.is_empty() {
                    self.new_project_parent = self.project_root.parent().map(|p| p.display().to_string()).unwrap_or_default();
                }
                self.show_new_project = true;
            }
            PaletteCmd::OpenProject => {
                if self.open_project_path.is_empty() {
                    self.open_project_path = self.project_root.display().to_string();
                }
                self.show_open_project = true;
            }
            PaletteCmd::BuildGameWindows => self.actions.push(EditorAction::BuildGame(bundle::BuildTarget::Windows)),
            PaletteCmd::Undo => self.actions.push(EditorAction::Undo),
            PaletteCmd::Redo => self.actions.push(EditorAction::Redo),
            PaletteCmd::Duplicate => {
                if let Selection::Object(i) = self.selection {
                    self.actions.push(EditorAction::DuplicateObject(i));
                }
            }
            PaletteCmd::Focus => self.actions.push(EditorAction::FocusSelected),
            PaletteCmd::HideSelection => self.toggle_hide_selection(),
            PaletteCmd::ShowAllHidden => self.actions.push(EditorAction::SceneView(crate::ViewRequest::ShowAll)),
            PaletteCmd::UnlockAll => self.actions.push(EditorAction::SceneView(crate::ViewRequest::UnlockAll)),
            PaletteCmd::ResetLayout => self.dock_state = default_layout(),
            PaletteCmd::ToggleGameUi => self.widget_filter.game_ui = !self.widget_filter.game_ui,
            PaletteCmd::ToggleGizmos => self.widget_filter.enabled = !self.widget_filter.enabled,
            PaletteCmd::ToggleGrid => self.project.settings.scene_grid = !self.project.settings.scene_grid,
            PaletteCmd::ToggleVr => {
                let on = !self.vr_wanted_next_start();
                self.set_vr_preference(on);
            }
            PaletteCmd::OpenTab(k) => {
                let tab = PALETTE_TABS[k].0.clone();
                match self.dock_state.find_tab(&tab) {
                    Some(loc) => self.dock_state.set_active_tab(loc),
                    None => self.dock_state.push_to_focused_leaf(tab),
                }
            }
            PaletteCmd::ProjectSettings => self.open_or_focus_tab(Tab::ProjectSettings),
            PaletteCmd::Layers => self.show_layers = true,
            PaletteCmd::InputBindings => self.show_bindings = true,
            PaletteCmd::AudioMixer => self.show_mixer = true,
            PaletteCmd::BakeLighting => self.actions.push(EditorAction::BakeLighting),
            PaletteCmd::BakeAudio => self.actions.push(EditorAction::BakeAudio),
            PaletteCmd::ReloadPlugins => self.actions.push(EditorAction::ReloadPlugins),
            PaletteCmd::Spawn(shape) => self.actions.push(EditorAction::Spawn(ObjectSource::Primitive { shape })),
            PaletteCmd::SpawnLight(kind) => self.actions.push(EditorAction::SpawnLight(kind)),
            PaletteCmd::SpawnAreaLight => self.actions.push(EditorAction::SpawnAreaLight),
            PaletteCmd::UiPreset(p) => self.actions.push(EditorAction::SpawnUi(p)),
            PaletteCmd::UiTemplate(t) => self.actions.push(EditorAction::SpawnUiTemplate(t)),
        }
    }

    fn menu_bar(&mut self, ctx: &egui::Context) {
        // Top-level menus, so hovering another one while a menu is open
        // switches to it (egui's MenuBar only opens on click).
        let mut bar: Vec<(egui::Id, egui::Rect)> = Vec::new();
        use crate::{icons::Icon, theme, widgets};
        egui::TopBottomPanel::top("zestengine-menubar")
            .exact_height(theme::H_MENU_BAR)
            .frame(egui::Frame::new().fill(theme::BG_CHROME).inner_margin(egui::Margin::symmetric(12, 0)))
            .show(ctx, |ui| {
            let full = ui.max_rect();
            // Spec §4: left and right zones flex, the transport stays centred.
            let center = egui::Rect::from_center_size(full.center(), egui::vec2(196.0, 30.0));
            let left = egui::Rect::from_min_max(full.min, egui::pos2(center.left() - 8.0, full.bottom()));
            let right = egui::Rect::from_min_max(egui::pos2(center.right() + 8.0, full.top()), full.max);
            let mut left_ui = ui.new_child(
                egui::UiBuilder::new().max_rect(left).layout(egui::Layout::left_to_right(egui::Align::Center)),
            );
            let (logo, _) = left_ui.allocate_exact_size(egui::vec2(20.0, 20.0), egui::Sense::hover());
            crate::icons::paint_logo(left_ui.painter(), logo, theme::accent(), theme::BG_CHROME);
            left_ui.label(egui::RichText::new("ZestEngine").font(theme::semibold(theme::SIZE_TITLE)).color(theme::TEXT_STRONG));
            left_ui.add_space(10.0);
            let menu_style = egui::style::StyleModifier::new(|s: &mut egui::Style| {
                s.spacing.button_padding = egui::vec2(9.0, 6.0);
                s.spacing.item_spacing.x = 0.0;
                let w = &mut s.visuals.widgets;
                for st in [&mut w.inactive, &mut w.hovered, &mut w.active, &mut w.open] {
                    st.bg_stroke = egui::Stroke::NONE;
                    st.corner_radius = egui::CornerRadius::same(theme::R_FIELD);
                }
                w.inactive.weak_bg_fill = egui::Color32::TRANSPARENT;
                w.inactive.fg_stroke.color = theme::TEXT_SECONDARY;
                w.hovered.weak_bg_fill = theme::BG_BUTTON;
                w.active.weak_bg_fill = theme::BG_BUTTON;
                w.open.weak_bg_fill = theme::BG_BUTTON;
            });
            egui::MenuBar::new().style(menu_style).ui(&mut left_ui, |ui| {
                top_menu(&mut bar, ui, "File", |ui| {
                    if ui
                        .button("Project Hub…")
                        .on_hover_text("Recent projects, templates and opening projects (a separate window)")
                        .clicked()
                    {
                        self.actions.push(EditorAction::OpenProjectHub);
                        ui.close();
                    }
                    if ui.button("New Project…").clicked() {
                        if self.new_project_parent.is_empty() {
                            // Default new projects beside the current project.
                            self.new_project_parent = self
                                .project_root
                                .parent()
                                .map(|p| p.display().to_string())
                                .unwrap_or_default();
                        }
                        self.show_new_project = true;
                        ui.close();
                    }
                    if ui.button("Open Project…").clicked() {
                        if self.open_project_path.is_empty() {
                            self.open_project_path = self.project_root.display().to_string();
                        }
                        self.show_open_project = true;
                        ui.close();
                    }
                    if ui.button("Project Settings…").clicked() {
                        self.open_or_focus_tab(Tab::ProjectSettings);
                        ui.close();
                    }
                    if ui
                        .button("Build Game")
                        .on_hover_text("cargo build --release + bundle assets into build/ (runs in the background)")
                        .clicked()
                    {
                        self.actions.push(EditorAction::BuildGame(bundle::BuildTarget::Host));
                        ui.close();
                    }
                    if !cfg!(windows)
                        && ui
                            .button("Build Game for Windows")
                            .on_hover_text("Cross-compile with cargo-xwin (MSVC) into build-windows/, Steam Audio DLL included")
                            .clicked()
                    {
                        self.actions.push(EditorAction::BuildGame(bundle::BuildTarget::Windows));
                        ui.close();
                    }
                    crate::widgets::menu_separator(ui);
                    if ui.button("New Scene").clicked() {
                        self.actions.push(EditorAction::NewScene);
                        ui.close();
                    }
                    ui.menu_button("Open Scene", |ui| {
                        crate::widgets::menu_frame(ui);
                        let scenes = scan_project_files(&self.project_root, "scene");
                        if scenes.is_empty() {
                            ui.label(
                                egui::RichText::new("no .scene files in the project")
                                    .small()
                                    .weak(),
                            );
                        }
                        for rel in scenes {
                            if ui.button(&rel).clicked() {
                                self.actions
                                    .push(EditorAction::LoadScene(self.project_root.join(&rel)));
                                ui.close();
                            }
                        }
                    });
                    crate::widgets::menu_separator(ui);
                    if ui.button("Save Scene        Ctrl+S").clicked() {
                        self.actions.push(EditorAction::SaveScene(None));
                        ui.close();
                    }
                    ui.menu_button("Save Scene As…", |ui| {
                        crate::widgets::menu_frame(ui);
                        ui.text_edit_singleline(&mut self.scene_name_input);
                        if ui.button("Save").clicked() {
                            let path = self.project_root.join(&self.scene_name_input);
                            self.actions.push(EditorAction::SaveScene(Some(path)));
                            ui.close();
                        }
                    });
                    crate::widgets::menu_separator(ui);
                    ui.label(
                        egui::RichText::new("Import: double-click a model in Files")
                            .small()
                            .weak(),
                    );
                    crate::widgets::menu_separator(ui);
                    if ui.button("Quit").clicked() {
                        if self.scene_dirty {
                            self.show_quit_dialog = true;
                        } else {
                            self.quit_after_frame = true;
                        }
                        ui.close();
                    }
                });
                top_menu(&mut bar, ui, "Edit", |ui| {
                    if ui
                        .add_enabled(
                            self.undo_stack.can_undo(),
                            egui::Button::new("Undo        Ctrl+Z"),
                        )
                        .clicked()
                    {
                        self.actions.push(EditorAction::Undo);
                        ui.close();
                    }
                    if ui
                        .add_enabled(
                            self.undo_stack.can_redo(),
                            egui::Button::new("Redo        Ctrl+Shift+Z"),
                        )
                        .clicked()
                    {
                        self.actions.push(EditorAction::Redo);
                        ui.close();
                    }
                    crate::widgets::menu_separator(ui);
                    if ui
                        .checkbox(&mut self.project.settings.vim_mode, "Vim mode")
                        .changed()
                    {
                        self.save_project();
                    }
                });
                top_menu(&mut bar, ui, "Session", |ui| {
                    if ui.button("Co-edit…").clicked() {
                        self.edit_window = true;
                        ui.close();
                    }
                    if self.edit_session.is_some()
                        && ui.button("Leave session").clicked()
                    {
                        self.edit_session = None;
                        self.edit_peers.clear();
                        ui.close();
                    }
                });
                top_menu(&mut bar, ui, "Object", |ui| {
                    use zestengine_assets::{ObjectSource, PrimitiveShape};
                    if ui.button("Create Empty").clicked() {
                        self.actions.push(EditorAction::Spawn(ObjectSource::Empty));
                        ui.close();
                    }
                    // Create a reusable .prefab from the selected object subtree
                    // (CHECKLIST T0 #7). Saved next to the scene; instantiate later
                    // via `Prefab::load(..).instantiate(pos)`.
                    if let Selection::Object(i) = self.selection
                        && ui.button("Create Prefab from Selection").clicked() {
                            if let Some(prefab) =
                                self.scene.prefab_from_object(i, &self.project_root, &self.shaders)
                            {
                                let name = self.scene.objects[i].name.replace(' ', "_");
                                let path = self.project_root.join(format!("{name}.prefab"));
                                match prefab.save(&path) {
                                    Ok(_) => {
                                        self.set_load_status(&format!("Saved prefab {}", path.display()));
                                        // The selection becomes an instance of what was just saved.
                                        self.connect_prefab(i, &path);
                                    }
                                    Err(e) => self.set_load_status(&format!("Prefab save failed: {e}")),
                                }
                            }
                            ui.close();
                        }
                    if ui.button("Create Camera").clicked() {
                        self.actions.push(EditorAction::Spawn(ObjectSource::Camera));
                        ui.close();
                    }
                    ui.menu_button("Create Light", |ui| {
                        crate::widgets::menu_frame(ui);
                        for kind in zestengine_core::LightKind::ALL {
                            if ui.button(kind.label()).clicked() {
                                self.actions.push(EditorAction::SpawnLight(kind));
                                ui.close();
                            }
                        }
                        if ui.button("Area Light").clicked() {
                            self.actions.push(EditorAction::SpawnAreaLight);
                            ui.close();
                        }
                    });
                    ui.menu_button("Create Volume", |ui| {
                        crate::widgets::menu_frame(ui);
                        for (label, action) in [
                            ("Post FX Volume (Box)", EditorAction::SpawnPostFxVolume(false)),
                            ("Post FX Volume (Global)", EditorAction::SpawnPostFxVolume(true)),
                            ("Light Probe Volume", EditorAction::SpawnProbeVolume),
                            ("Reflection Probe", EditorAction::SpawnReflectionProbe),
                            ("Flux Volume", EditorAction::SpawnFluxVolume),
                        ] {
                            if ui.button(label).clicked() {
                                self.actions.push(action);
                                ui.close();
                            }
                        }
                    });
                    if let Some(kind) = crate::scene_panel::audio_spawn_menu(ui) {
                        self.actions.push(match kind {
                            crate::SpawnKind::AudioSource => EditorAction::SpawnAudioSource,
                            crate::SpawnKind::AudioVolume => EditorAction::SpawnAudioVolume,
                            _ => unreachable!("audio submenu returned a non-audio object"),
                        });
                    }
                    ui.menu_button("Create Physics", |ui| {
                        crate::widgets::menu_frame(ui);
                        if ui.button("Wind Zone").on_hover_text("Wind for water waves, cloth and steam").clicked() {
                            self.actions.push(EditorAction::SpawnWindZone);
                            ui.close();
                        }
                    });
                    ui.menu_button("Create Collider", |ui| {
                        crate::widgets::menu_frame(ui);
                        if ui.button("Box Collider").clicked() {
                            self.actions.push(EditorAction::SpawnBoxCollider);
                            ui.close();
                        }
                        if ui.button("Sphere Collider").clicked() {
                            self.actions.push(EditorAction::SpawnSphereCollider);
                            ui.close();
                        }
                        if ui.button("Capsule Collider").clicked() {
                            self.actions.push(EditorAction::SpawnCapsuleCollider);
                            ui.close();
                        }
                        if ui.button("Mesh Collider").on_hover_text("Add a Mesh Collider to the selected mesh objects").clicked() {
                            self.actions.push(EditorAction::AddMeshCollider);
                            ui.close();
                        }
                    });
                    ui.menu_button("Create Water", |ui| {
                        crate::widgets::menu_frame(ui);
                        if ui.button("Water").clicked() {
                            self.actions.push(EditorAction::SpawnWater);
                            ui.close();
                        }
                        if ui.button("Waterfall").clicked() {
                            self.actions.push(EditorAction::SpawnWaterfall);
                            ui.close();
                        }
                        if ui.button("Whirlpool").clicked() {
                            self.actions.push(EditorAction::SpawnWhirlpool);
                            ui.close();
                        }
                        if ui.button("Floating Cube").clicked() {
                            self.actions.push(EditorAction::SpawnFloatingCube);
                            ui.close();
                        }
                        if ui.button("Bucket").on_hover_text("A container you can carry water in").clicked() {
                            self.actions.push(EditorAction::SpawnBucket);
                            ui.close();
                        }
                    });
                    crate::widgets::menu_separator(ui);
                    for shape in [
                        PrimitiveShape::Cube,
                        PrimitiveShape::Sphere,
                        PrimitiveShape::Capsule,
                        PrimitiveShape::Plane,
                    ] {
                        if ui.button(shape.label()).clicked() {
                            self.actions
                                .push(EditorAction::Spawn(ObjectSource::Primitive { shape }));
                            ui.close();
                        }
                    }
                });
                top_menu(&mut bar, ui, "Tools", |ui| {
                    for (tool, label) in [
                        (GizmoTool::Select, "Select      Q"),
                        (GizmoTool::Move, "Move        W"),
                        (GizmoTool::Rotate, "Rotate      E"),
                        (GizmoTool::Scale, "Scale       R"),
                        (GizmoTool::Rect, "Rect        T"),
                    ] {
                        if ui.radio(self.gizmo.tool == tool, label).clicked() {
                            self.gizmo.tool = tool;
                        }
                    }
                    crate::widgets::menu_separator(ui);
                    if ui.button("Neovim…").on_hover_text("Embedded Neovim (nvim --embed)").clicked() {
                        if let Some(loc) = self.dock_state.find_tab(&Tab::Nvim) {
                            self.dock_state.set_active_tab(loc);
                        } else {
                            if let Some((surface, node, _)) = self.dock_state.find_tab(&Tab::Viewport) {
                                self.dock_state.set_focused_node_and_surface((surface, node));
                            }
                            self.dock_state.push_to_focused_leaf(Tab::Nvim);
                        }
                        ui.close();
                    }
                    crate::widgets::menu_separator(ui);
                    if ui.button("Input Bindings…").clicked() {
                        self.show_bindings = true;
                        ui.close();
                    }
                    if ui.button("Network…").clicked() {
                        self.show_network = true;
                        ui.close();
                    }
                    if ui.button("Performance…").clicked() {
                        self.show_performance = true;
                        ui.close();
                    }
                    if ui.button("Layers…").clicked() {
                        self.show_layers = true;
                        ui.close();
                    }
                    if ui.button("Audio Mixer…").clicked() {
                        self.show_mixer = true;
                        ui.close();
                    }
                    if ui
                        .button("Build Content Package…")
                        .on_hover_text("Package the scene (World) or the selected hierarchy (Avatar / Prop) as a .zestenginepak games load at runtime")
                        .clicked()
                    {
                        let kind = if matches!(self.selection, Selection::Object(_)) {
                            zestengine_core::ContentKind::Prop
                        } else {
                            zestengine_core::ContentKind::World
                        };
                        self.content_builder.open_for(kind);
                        ui.close();
                    }
                    crate::widgets::menu_separator(ui);
                    let has_plugins = plugins::PluginHost::any_plugins(&self.project_root);
                    if !has_plugins && ui.button("Create Component Plugin").clicked() {
                        self.actions.push(EditorAction::CreatePluginCrate);
                        ui.close();
                    }
                    if has_plugins
                        && ui
                            .button("Build & Reload Components")
                            .on_hover_text(
                                "cargo-builds plugins/ in the background and hot-reloads them",
                            )
                            .clicked()
                    {
                        self.actions.push(EditorAction::ReloadPlugins);
                        ui.close();
                    }
                    crate::widgets::menu_separator(ui);
                    if ui.button("Bake Audio").on_hover_text("Steam Audio reverb + pathing for Audio Probe Volumes").clicked() {
                        self.actions.push(EditorAction::BakeAudio);
                        ui.close();
                    }
                    let can_bake = self
                        .renderer
                        .as_ref()
                        .is_some_and(|r| r.supports_baking());
                    let baked = self.scene.baked.is_some();
                    if ui
                        .add_enabled(can_bake, egui::Button::new("Bake Lighting"))
                        .on_hover_text(if can_bake {
                            "Ray-trace lightmaps + probes for Static objects and probe volumes (blocks the UI)"
                        } else {
                            "This GPU has no ray-query support; baking is unavailable"
                        })
                        .clicked()
                    {
                        self.actions.push(EditorAction::BakeLighting);
                        ui.close();
                    }
                    if ui
                        .add_enabled(baked, egui::Button::new("Clear Baked Lighting"))
                        .clicked()
                    {
                        self.actions.push(EditorAction::ClearBake);
                        ui.close();
                    }
                });
                // CloseOnClickOutside lets you toggle several checkboxes without
                // the menu snapping shut after each click.
                let (view, _) = egui::containers::menu::MenuButton::new("View")
                    .config(
                        egui::containers::menu::MenuConfig::new()
                            .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside),
                    )
                    .ui(ui, |ui| {
                        ui.checkbox(&mut self.show_stats, "FPS in menu bar");
                        ui.checkbox(&mut self.show_stats_overlay, "Profiler window")
                            .on_hover_text(
                                "Open the profiler in a separate window you can move to \
                                 another monitor",
                            );
                        ui.checkbox(&mut self.project.settings.selection_outline, "Selection outline")
                            .on_hover_text("Outline the selected objects in the scene view (colour and width: Project Settings > Appearance)");
                        if let Some(renderer) = self.renderer.as_mut() {
                            let mut vsync = renderer.vsync();
                            if ui
                                .checkbox(&mut vsync, "VSync")
                                .on_hover_text("Off = uncapped frame rate (real numbers)")
                                .changed()
                            {
                                renderer.set_vsync(vsync);
                            }
                            // FSR3 temporal upscaling: Native = off; the rest pick
                            // the quality preset (render scale).
                            let mut quality = renderer.upscale_quality();
                            egui::ComboBox::from_label("Upscaling")
                                .selected_text(quality.label())
                                .show_ui(ui, |ui| {
                                    for q in zestengine_render::UpscaleQuality::ALL {
                                        ui.selectable_value(&mut quality, q, q.label());
                                    }
                                });
                            if quality != renderer.upscale_quality() {
                                renderer.set_upscale_quality(quality);
                            }
                            // FSR3 frame generation (gated behind the upscaler).
                            let mut fg = renderer.frame_gen();
                            ui.add_enabled_ui(renderer.upscaler() == zestengine_render::Upscaler::Fsr3, |ui| {
                                if ui
                                    .checkbox(&mut fg, "Frame generation")
                                    .on_hover_text("Interpolates frames between rendered ones (FSR3)")
                                    .changed()
                                {
                                    renderer.set_frame_gen(fg);
                                }
                            });
                            let supported = renderer.gpu_cull_supported();
                            let mut cull = renderer.gpu_cull_settings();
                            let mut changed = ui
                                .add_enabled(supported, egui::Checkbox::new(&mut cull.enabled, "GPU culling"))
                                .on_hover_text(
                                    "Cull and draw opaque objects and shadow casters on the GPU \
                                     (indirect draws). Off = CPU frustum culling",
                                )
                                .changed();
                            changed |= ui
                                .add_enabled(
                                    supported && cull.enabled,
                                    egui::Checkbox::new(&mut cull.hiz_occlusion, "Hi-Z occlusion"),
                                )
                                .on_hover_text(
                                    "Also skip objects hidden behind the depth (two-phase: what the \
                                     previous frame's depth hides is re-tested against this frame's, \
                                     so objects coming into view appear the same frame)",
                                )
                                .changed();
                            if changed {
                                renderer.set_gpu_cull_settings(cull);
                            }
                        }
                        if ui
                            .add_enabled(
                                self.scene.skybox.is_some(),
                                egui::Button::new("Clear Skybox"),
                            )
                            .on_hover_text("Revert to the procedural gradient sky")
                            .clicked()
                        {
                            self.actions.push(EditorAction::ClearSkybox);
                            ui.close();
                        }
                        crate::widgets::menu_separator(ui);
                        ui.label(egui::RichText::new("GI Debug View").small().weak());
                        ui.radio_value(&mut self.gi_debug, 0, "Off (normal shading)");
                        ui.radio_value(&mut self.gi_debug, 1, "World normals")
                            .on_hover_text("Verify the surface normals feeding GI");
                        ui.radio_value(&mut self.gi_debug, 2, "Indirect GI only")
                            .on_hover_text("Isolate the Flux indirect bounce");
                        if ui.button("Reset Layout").clicked() {
                            self.dock_state = default_layout();
                            ui.close();
                        }
                        crate::widgets::menu_separator(ui);
                        self.vr_menu_ui(ui);
                    });
                bar.push((egui::Popup::default_response_id(&view), view.rect));
                top_menu(&mut bar, ui, "Windows", |ui| {
                    ui.label(egui::RichText::new("Open / focus a panel").small().weak());
                    for (tab, label) in [
                        (Tab::Viewport, "Scene"),
                        (Tab::Camera, "Camera"),
                        (Tab::UiEditor, "UI Editor"),
                        (Tab::AnimController, "Animator Controller"),
                        (Tab::AnimClip, "Animation"),
                        (Tab::Sequencer, "Sequencer"),
                        (Tab::ShaderGraph, "Shader Graph"),
                        (Tab::Scene, "Scene"),
                        (Tab::Inspector, "Inspector"),
                        (Tab::Environment, "Environment"),
                        (Tab::Baker, "FluxBaker"),
                        (Tab::Occlusion, "Occlusion"),
                        (Tab::ProjectSettings, "Project Settings"),
                        (Tab::Files, "Files"),
                        (Tab::Log, "Log"),
                    ] {
                        let open = self.dock_state.iter_all_tabs().any(|(_, t)| *t == tab);
                        if ui.selectable_label(open, label).clicked() {
                            if let Some(loc) = self.dock_state.find_tab(&tab) {
                                self.dock_state.set_active_tab(loc);
                            } else {
                                self.dock_state.push_to_focused_leaf(tab);
                            }
                            ui.close();
                        }
                    }
                    crate::widgets::menu_separator(ui);
                    if ui.selectable_label(self.body.window_open, "Body Tracking").on_hover_text("Height, tracker roles and full-body calibration").clicked() {
                        self.body.window_open = true;
                        ui.close();
                    }
                });
                top_menu(&mut bar, ui, "Help", |ui| {
                    ui.checkbox(&mut self.show_help, "Controls");
                    if ui.button("About").clicked() {
                        self.about_open = true;
                    }
                });

            });

            // Transport (centre): Play/Stop, Pause, Step, and the game camera
            // toggle while playing.
            let mut center_ui = ui.new_child(
                egui::UiBuilder::new().max_rect(center).layout(egui::Layout::left_to_right(egui::Align::Center)),
            );
            let trough_w = 4.0 + 82.0 + 2.0 * 30.0 + if self.playing { 30.0 } else { 0.0 } + 8.0;
            center_ui.add_space(((center.width() - trough_w) * 0.5).max(0.0));
            widgets::trough(&mut center_ui, |ui| {
                let (icon, label) = if self.playing { (Icon::Stop, "Stop") } else { (Icon::Play, "Play") };
                if widgets::primary_button_sized(ui, Some(icon), label, theme::H_FIELD)
                    .on_hover_text("Run components; Stop restores the scene")
                    .clicked()
                {
                    self.actions.push(EditorAction::TogglePlay);
                }
                let size = egui::vec2(28.0, theme::H_FIELD);
                ui.add_enabled_ui(self.playing, |ui| {
                    if widgets::icon_button_sized(ui, Icon::Pause, if self.play_paused { "Resume" } else { "Pause" }, self.play_paused, size)
                        .clicked()
                    {
                        self.play_paused = !self.play_paused;
                    }
                    if widgets::icon_button_sized(ui, Icon::Step, "Step one frame", false, size).clicked() {
                        self.play_paused = true;
                        self.play_step = true;
                    }
                });
                if self.playing
                    && widgets::icon_button_sized(
                        ui,
                        Icon::Camera,
                        "Scene follows the game camera (off = free camera; the game shows in the Camera tab)",
                        self.play_game_camera,
                        size,
                    )
                    .clicked()
                {
                    self.play_game_camera = !self.play_game_camera;
                }
            });

            // Right: breadcrumb, command search, preferences.
            let mut right_ui = ui.new_child(
                egui::UiBuilder::new().max_rect(right).layout(egui::Layout::right_to_left(egui::Align::Center)),
            );
            let settings_open = self.dock_state.iter_all_tabs().any(|(_, t)| *t == Tab::ProjectSettings);
            if widgets::icon_button_sized(&mut right_ui, Icon::Gear, "Project settings", settings_open, egui::vec2(28.0, 28.0))
                .clicked()
            {
                self.open_or_focus_tab(Tab::ProjectSettings);
            }
            right_ui.add_space(4.0);
            let search_w = 220.0_f32.min((right.width() - 40.0).max(0.0));
            if search_w > 120.0 {
                let (rect, resp) = right_ui.allocate_exact_size(egui::vec2(search_w, theme::H_TOOLBAR), egui::Sense::click());
                let p = right_ui.painter();
                p.rect(
                    rect,
                    egui::CornerRadius::same(theme::R_ROW),
                    if resp.hovered() { theme::BG_APP } else { theme::BG_FIELD },
                    egui::Stroke::new(1.0_f32, theme::LINE_FIELD),
                    egui::StrokeKind::Inside,
                );
                crate::icons::paint(
                    p,
                    egui::Rect::from_center_size(egui::pos2(rect.left() + 14.0, rect.center().y), egui::vec2(13.0, 13.0)),
                    Icon::Search,
                    theme::TEXT_DIM,
                );
                p.text(egui::pos2(rect.left() + 28.0, rect.center().y), egui::Align2::LEFT_CENTER, "Search commands", egui::FontId::proportional(theme::SIZE_BODY), theme::TEXT_DIM);
                p.text(egui::pos2(rect.right() - 10.0, rect.center().y), egui::Align2::RIGHT_CENTER, "Ctrl K", theme::mono(theme::SIZE_META), theme::TEXT_DIM);
                if resp.on_hover_text("Search commands (Ctrl K)").clicked() {
                    self.command_palette.open();
                }
                right_ui.add_space(10.0);
            }
            let scene = self
                .current_scene_path
                .as_ref()
                .and_then(|p| p.file_name())
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "untitled".into());
            let mut job = egui::text::LayoutJob::default();
            let font = egui::FontId::proportional(theme::SIZE_BODY);
            job.append(&self.project.name, 0.0, egui::TextFormat::simple(font.clone(), theme::TEXT_DIM));
            job.append(" / ", 0.0, egui::TextFormat::simple(font.clone(), theme::LINE_DASHED));
            job.append(&scene, 0.0, egui::TextFormat::simple(font, theme::TEXT_SECONDARY));
            job.wrap = egui::text::TextWrapping::truncate_at_width(right_ui.available_width().max(0.0));
            right_ui.label(job);
        });
        switch_hovered_menu(ctx, &bar);
        self.command_palette_ui(ctx);

        if let Some(error) = &self.plugin_build_error {
            let mut open = true;
            egui::Window::new("Component build failed")
                .open(&mut open)
                .default_width(560.0)
                .show(ctx, |ui| {
                    egui::ScrollArea::both().max_height(360.0).show(ui, |ui| {
                        ui.label(egui::RichText::new(error).small().monospace());
                    });
                });
            if !open {
                self.plugin_build_error = None;
            }
        }

        if self.show_help {
            egui::Window::new("Controls")
                .open(&mut self.show_help)
                .show(ctx, |ui| {
                    ui.label("Left click (no drag) — select object · Escape — deselect");
                    ui.label("Left drag — orbit selection or viewport center (Alt forces orbit over the gizmo)");
                    ui.label("F — focus the selected object");
                    ui.label("Right mouse (hold) — look · WASD fly · Q/E down/up · Shift fast");
                    ui.label("Middle drag — pan · Scroll — dolly");
                    ui.label("W / E / R — gizmo move / rotate / scale (buttons top-left)");
                    ui.label("Ctrl while dragging — snap to grid (controls top-center)");
                    ui.label("Ctrl+Z / Ctrl+Shift+Z — undo / redo");
                    ui.label("Drag a .material from Files onto a mesh or a material slot");
                    ui.label("Double-click a model file in Files to import it");
                });
        }
        if self.about_open {
            egui::Window::new("About ZestEngine")
                .open(&mut self.about_open)
                .show(ctx, |ui| {
                    if let Some(color) = about_color_image() {
                        let size = egui::vec2(color.size[0] as f32, color.size[1] as f32);
                        let max_w = 520.0_f32.min(size.x);
                        let scale = if size.x > max_w { max_w / size.x } else { 1.0 };
                        let tex = ctx.load_texture("zestengine-about", color, egui::TextureOptions::LINEAR);
                        ui.add(egui::Image::new(egui::load::SizedTexture::new(tex.id(), size * scale)));
                    }
                    ui.add_space(8.0);
                    ui.label("ZestEngine");
                    ui.label(build_stamp());
                    ui.label("A standalone 3D game engine in Rust + Vulkan.");
                });
        }
    }

    /// Open `tab` docked into the main (central) area next to the Scene view,
    /// or focus it if it's already open somewhere.
    fn open_or_focus_tab(&mut self, tab: Tab) {
        if let Some(loc) = self.dock_state.find_tab(&tab) {
            self.dock_state.set_active_tab(loc);
        } else {
            // Focus the Viewport's leaf so the tab lands in the main docking
            // area, not in a side (Inspector) or bottom (Log/Files) dock.
            if let Some((surface, node, _)) = self.dock_state.find_tab(&Tab::Viewport) {
                self.dock_state.set_focused_node_and_surface((surface, node));
            }
            self.dock_state.push_to_focused_leaf(tab);
        }
    }

    /// Open (or focus) a code/text file in a dockable editor tab.
    fn open_code_file(&mut self, path: PathBuf) {
        if !self.open_editors.iter().any(|e| e.path == path) {
            let language = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("txt")
                .to_owned();
            match std::fs::read_to_string(&path) {
                Ok(text) if text.len() <= 2 * 1024 * 1024 => {
                    // Rust files get language-server diagnostics.
                    if language == "rs"
                        && let Some(lsp) = self.ensure_lsp()
                    {
                        lsp.open(&path, "rust", &text);
                    }
                    self.open_editors.push(OpenEditor {
                        path: path.clone(),
                        text,
                        dirty: false,
                        language,
                        diagnostics: Vec::new(),
                        last_edit: None,
                        lsp_dirty: false,
                        completion: None,
                        hover: None,
                        goto: None,
                        references: None,
                        actions: Vec::new(),
                        disk_mtime: std::fs::metadata(&path).ok().and_then(|m| m.modified().ok()),
                    });
                }
                Ok(_) => {
                    tracing::warn!("file too large for the code editor");
                    return;
                }
                Err(e) => {
                    tracing::error!("opening file: {e:#}");
                    return;
                }
            }
        }
        let tab = Tab::Code(path);
        if let Some(loc) = self.dock_state.find_tab(&tab) {
            self.dock_state.set_active_tab(loc);
        } else {
            // Open alongside the Viewport so code sits in the main work area.
            if let Some((surface, node, _)) = self.dock_state.find_tab(&Tab::Viewport) {
                self.dock_state
                    .set_focused_node_and_surface((surface, node));
            }
            self.dock_state.push_to_focused_leaf(tab);
        }
    }

    /// True when the active/focused dock tab is a code editor. Used to keep
    /// Escape from doing the global deselect while editing (Escape is the vim
    /// Insert -> Normal key there).
    fn code_tab_focused(&mut self) -> bool {
        self.dock_state
            .find_active_focused()
            .map(|(_, tab)| matches!(tab, Tab::Code(_)))
            .unwrap_or(false)
    }

    /// Close the focused code tab (Ctrl+W), dropping its editor buffer.
    fn close_focused_code_tab(&mut self) {
        let path = self
            .dock_state
            .find_active_focused()
            .and_then(|(_, t)| match t {
                Tab::Code(p) => Some(p.clone()),
                _ => None,
            });
        if let Some(path) = path {
            if let Some(loc) = self.dock_state.find_tab(&Tab::Code(path.clone())) {
                self.dock_state.remove_tab(loc);
            }
            self.open_editors.retain(|e| e.path != path);
        }
    }

    /// Lazily spawn the Rust language server (rooted at the project), reusing
    /// it across files. Returns None if it isn't available.
    fn ensure_lsp(&mut self) -> Option<&mut lsp::LspClient> {
        if self.lsp.is_none() && !self.lsp_failed {
            match lsp::LspClient::spawn("rust-analyzer", &self.project_root) {
                Ok(client) => {
                    tracing::info!("started rust-analyzer");
                    self.lsp = Some(client);
                }
                Err(e) => {
                    tracing::warn!("rust-analyzer unavailable ({e}); no Rust diagnostics");
                    self.lsp_failed = true;
                }
            }
        }
        self.lsp.as_mut()
    }

    /// Poll the language server and push any new edits as `didChange`, then
    /// route diagnostics to the matching editor. Called once per frame.
    fn pump_lsp(&mut self) {
        // Send buffered changes.
        let changed: Vec<(PathBuf, String)> = self
            .open_editors
            .iter_mut()
            .filter(|e| e.lsp_dirty && e.language == "rs")
            .map(|e| {
                e.lsp_dirty = false;
                (e.path.clone(), e.text.clone())
            })
            .collect();
        // Definition jumps are collected here and applied after the LSP borrow
        // ends, since opening a target file needs `&mut self`.
        let mut goto_targets: Vec<(PathBuf, u32, u32)> = Vec::new();
        if let Some(lsp) = self.lsp.as_mut() {
            for (path, text) in changed {
                lsp.change(&path, "rust", &text);
            }
            // Drain server events → editors.
            for event in lsp.poll() {
                match event {
                    lsp::LspEvent::Diagnostics { path, diags } => {
                        // Project-wide tally so the file browser can badge files
                        // (and folders) that have problems, even unopened ones.
                        let errors = diags.iter().filter(|d| d.severity <= 1).count() as u32;
                        let warns = diags.len() as u32 - errors;
                        if errors == 0 && warns == 0 {
                            self.file_diagnostics.remove(&path);
                        } else {
                            self.file_diagnostics.insert(path.clone(), (errors, warns));
                        }
                        if let Some(editor) = self.open_editors.iter_mut().find(|e| e.path == path)
                        {
                            editor.diagnostics = diags
                                .into_iter()
                                .map(|d| CodeDiagnostic {
                                    level: if d.severity <= 1 { "error" } else { "warning" }
                                        .to_owned(),
                                    file: path
                                        .file_name()
                                        .map(|n| n.to_string_lossy().into_owned())
                                        .unwrap_or_default(),
                                    line: d.line + 1,
                                    message: d.message,
                                })
                                .collect();
                        }
                    }
                    lsp::LspEvent::Response { id, result } => {
                        if let Some(kind) = self.lsp_requests.remove(&id) {
                            if matches!(kind, LspRequestKind::Definition) {
                                if let Some(target) = parse_definition(&result) {
                                    goto_targets.push(target);
                                }
                            } else {
                                apply_lsp_response(&mut self.open_editors, kind, result);
                            }
                        }
                    }
                    lsp::LspEvent::Initialized => {}
                }
            }
        }
        // Open + jump to any definition targets (uses &mut self).
        for (path, line, col) in goto_targets {
            self.open_code_file(path.clone());
            if let Some(editor) = self.open_editors.iter_mut().find(|e| e.path == path) {
                editor.goto = Some((line, col));
            }
        }
    }

    /// Push the scene's baked probe SH + volume metadata to the renderer (set-0
    /// binding 2) so the standard shader samples it per fragment. Empty when no
    /// bake, reverting fragments to flat ambient.
    fn upload_baked_probes(&mut self) {
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        match &self.scene.baked {
            Some(b) => {
                // FluxVoxel voxel volumes are owned by `realtime_gi::update_flux_voxel`
                // (it seeds its static base from the baked SH, then adds dynamic
                // lights live), so skip them here — only plain DDGI volumes are
                // uploaded as static baked probes.
                let vols: Vec<_> = b
                    .probe_volumes
                    .iter()
                    .filter(|v| !v.flux_voxel && v.blend_distance <= 0.0)
                    .map(|v| {
                        (
                            v.world_to_local,
                            v.size,
                            [v.counts[0] as u32, v.counts[1] as u32, v.counts[2] as u32],
                            v.sh_base as u32,
                        )
                    })
                    .collect();
                renderer.set_baked_probes(&b.probe_sh, &vols);
                b.upload_light_volumes(renderer);
                b.upload_lightmaps(renderer);
            }
            None => {
                renderer.set_baked_probes(&[], &[]);
                renderer.set_light_volumes(&[]);
                renderer.set_baked_lightmaps(&[]);
            }
        }
    }

    /// Gather the scene's static geometry, baked lights, and probe volumes,
    /// run the GPU lighting bake, and store the result for runtime sampling.
    /// Blocks the UI while the GPU traces (off the hot path; explicit action).
    /// Start a chunked lighting bake (one unit/frame; see `step_bake`). No-op if
    /// one is already running or the GPU can't ray trace.
    fn start_bake(&mut self) {
        // Before baking, check for static models that have no lightmap UV (uv1 is
        // a uv0 fallback → would bake garbage, so the bake skips them). If any,
        // ask the user whether to generate a non-overlapping unwrap first.
        let needs = self.scene.models_needing_lightmap_uv();
        if !needs.is_empty() {
            self.pending_lm_uv_models = Some(needs);
            return; // the modal (lightmap_uv_window) drives what happens next
        }
        self.do_bake();
    }

    fn do_bake(&mut self) {
        use crate::tasks::{BakePhase, NotifyLevel, TaskKind, TaskProgress};
        if self.bake_job.is_some() {
            self.tasks.notify("Bake already running", NotifyLevel::Info);
            return;
        }
        let Some(renderer) = self.renderer.as_ref() else {
            return;
        };
        if !renderer.supports_baking() {
            self.tasks
                .notify("This GPU can't ray trace; bake unavailable", NotifyLevel::Warn);
            return;
        }
        let gather = self.scene.gather_bake();
        if gather.instances.is_empty() && gather.probes.is_empty() {
            self.tasks.notify(
                "Nothing to bake — mark objects Static and/or add a Light Probe Volume",
                NotifyLevel::Warn,
            );
            return;
        }
        let settings = self.scene.environment.bake;
        let input = zestengine_render::BakeInput {
            instances: &gather.instances,
            lights: &gather.lights,
            probes: &gather.probes,
            sky_color: gather.sky_color,
            bounces: settings.bounces,
            samples: settings.samples,
            probes_only: false,
            gpu_idle_frac: settings.gpu_throttle.clamp(0.0, 4.0),
        };
        let job = match renderer.bake_begin(&input) {
            Ok(j) => j,
            Err(e) => {
                self.tasks
                    .notify(format!("Bake failed: {e}"), NotifyLevel::Warn);
                return;
            }
        };
        // FluxVoxel volume count for the progress display (peek; recomputed when the
        // main pass completes).
        let flux_voxel_volumes = self
            .scene
            .gather_flux_voxel_bake()
            .map(|f| f.probe_volumes.len() as u32)
            .unwrap_or(0);
        let (task_id, cancel, progress) = self.tasks.register_stepped(
            "Bake lighting",
            TaskKind::Bake,
            TaskProgress::Bake {
                lights: gather.lights.len() as u32,
                bounces: settings.bounces,
                lightmap_done: 0,
                lightmap_total: gather.instances.len() as u32,
                probe_volumes: gather.probe_volumes.len() as u32,
                flux_voxel_volumes,
                phase: BakePhase::Lightmaps,
            },
        );
        let mut object_lightmap = std::collections::HashMap::new();
        for (layer, &obj) in gather.instance_objects.iter().enumerate() {
            object_lightmap.insert(obj, layer);
        }
        self.bake_job = Some(BakeRunner {
            job,
            task_id,
            cancel,
            progress,
            phase: BakeRunPhase::Main,
            settings,
            object_lightmap,
            probe_volumes: gather.probe_volumes,
            baked: None,
        });
        // Live preview: switch to baked mode with no lightmaps yet, so the
        // viewport darkens immediately and then lights up object-by-object as
        // each lightmap finishes (see `step_bake`).
        self.bake_serial += 1;
        self.scene.baked = Some(scene::BakedData::default());
        self.upload_baked_probes();
        // Paint the bake modal once before the first lightmap step locks the UI.
        self.bake_pending_warmup = true;
    }

    /// Advance the active bake by one unit per frame; handles cancel, progress,
    /// phase transition (main → FluxVoxel), and finalization.
    fn step_bake(&mut self) {
        use crate::tasks::{BakePhase, NotifyLevel, TaskProgress};
        if self.bake_job.is_none() {
            return;
        }
        // Cancel: tear down the GPU job, discard the partial result.
        if self
            .bake_job
            .as_ref()
            .unwrap()
            .cancel
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            let runner = self.bake_job.take().unwrap();
            if let Some(r) = self.renderer.as_ref() {
                let _ = r.bake_finish(runner.job);
            }
            self.tasks.complete_stepped(runner.task_id);
            self.tasks.notify("Bake cancelled", NotifyLevel::Info);
            return;
        }
        let step = {
            let runner = self.bake_job.as_mut().unwrap();
            match self.renderer.as_ref() {
                Some(r) => r.bake_step(&mut runner.job),
                None => return,
            }
        };
        match step {
            Err(e) => {
                tracing::error!("bake step: {e:#}");
                let runner = self.bake_job.take().unwrap();
                if let Some(r) = self.renderer.as_ref() {
                    let _ = r.bake_finish(runner.job);
                }
                self.tasks.complete_stepped(runner.task_id);
                self.tasks
                    .notify(format!("Bake failed: {e}"), NotifyLevel::Warn);
            }
            Ok(zestengine_render::BakeStep::Lightmap { done, .. }) => {
                // Snapshot the partial result, then drop the bake_job borrow so we
                // can apply it to the scene (live preview).
                let preview = {
                    let runner = self.bake_job.as_mut().unwrap();
                    if let Ok(mut p) = runner.progress.lock()
                        && let TaskProgress::Bake {
                            lightmap_done,
                            phase,
                            ..
                        } = &mut *p
                        {
                            *lightmap_done = done as u32;
                            *phase = BakePhase::Lightmaps;
                        }
                    // Only the main lightmap pass produces lightmaps to preview;
                    // the FluxVoxel pass is probes-only.
                    if runner.phase == BakeRunPhase::Main {
                        let lightmaps = runner.job.lightmaps_so_far().to_vec();
                        let n = lightmaps.len();
                        // Map only objects whose layer is already uploaded (others
                        // stay dark until their lightmap finishes).
                        let object_lightmap = runner
                            .object_lightmap
                            .iter()
                            .filter(|&(_, &layer)| layer < n)
                            .map(|(&o, &l)| (o, l))
                            .collect();
                        Some(scene::BakedData {
                            object_lightmap,
                            lightmaps,
                            lightmap_array: None,
                            probe_volumes: Vec::new(),
                            probe_sh: Vec::new(),
                        })
                    } else {
                        None
                    }
                };
                if let Some(baked) = preview {
                    self.scene.baked = Some(baked);
                    self.upload_baked_probes();
                }
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            Ok(zestengine_render::BakeStep::Probes) => {
                let runner = self.bake_job.as_mut().unwrap();
                if let Ok(mut p) = runner.progress.lock()
                    && let TaskProgress::Bake { phase, .. } = &mut *p {
                        *phase = match runner.phase {
                            BakeRunPhase::FluxVoxel => BakePhase::FluxVoxel,
                            BakeRunPhase::Main => BakePhase::Probes,
                        };
                    }
            }
            Ok(zestengine_render::BakeStep::Complete) => self.bake_phase_complete(),
        }
    }

    /// A bake pass finished: collect its output, advance main → FluxVoxel, or
    /// finalize when the FluxVoxel pass is done.
    fn bake_phase_complete(&mut self) {
        use crate::tasks::{BakePhase, TaskProgress};
        let mut runner = match self.bake_job.take() {
            Some(r) => r,
            None => return,
        };
        let output = match self.renderer.as_ref() {
            Some(r) => r.bake_finish(runner.job),
            None => return,
        };
        match runner.phase {
            BakeRunPhase::Main => {
                let baked = scene::BakedData {
                    object_lightmap: std::mem::take(&mut runner.object_lightmap),
                    lightmaps: output.lightmaps,
                    lightmap_array: None,
                    probe_volumes: std::mem::take(&mut runner.probe_volumes),
                    probe_sh: output.probes,
                };
                // Kick the FluxVoxel voxel pass if any FluxVolumes exist; else done.
                if let Some(fg) = self.scene.gather_flux_voxel_bake() {
                    let fin = zestengine_render::BakeInput {
                        instances: &fg.instances,
                        lights: &fg.lights,
                        probes: &fg.probes,
                        sky_color: fg.sky_color,
                        bounces: runner.settings.bounces,
                        samples: runner.settings.samples,
                        probes_only: true,
                        gpu_idle_frac: runner.settings.gpu_throttle.clamp(0.0, 4.0),
                    };
                    match self.renderer.as_ref().map(|r| r.bake_begin(&fin)) {
                        Some(Ok(job)) => {
                            if let Ok(mut p) = runner.progress.lock()
                                && let TaskProgress::Bake { phase, .. } = &mut *p {
                                    *phase = BakePhase::FluxVoxel;
                                }
                            self.bake_job = Some(BakeRunner {
                                job,
                                task_id: runner.task_id,
                                cancel: runner.cancel,
                                progress: runner.progress,
                                phase: BakeRunPhase::FluxVoxel,
                                settings: runner.settings,
                                object_lightmap: std::collections::HashMap::new(),
                                probe_volumes: fg.probe_volumes,
                                baked: Some(baked),
                            });
                            return;
                        }
                        _ => {
                            tracing::warn!("FluxVoxel bake begin failed; finalizing main bake only");
                        }
                    }
                }
                self.finalize_bake(baked, runner.task_id);
            }
            BakeRunPhase::FluxVoxel => {
                let mut baked = runner.baked.take().unwrap_or_else(|| scene::BakedData {
                    object_lightmap: std::collections::HashMap::new(),
                    lightmaps: Vec::new(),
                    lightmap_array: None,
                    probe_volumes: Vec::new(),
                    probe_sh: Vec::new(),
                });
                let offset = baked.probe_sh.len();
                for mut v in std::mem::take(&mut runner.probe_volumes) {
                    v.sh_base += offset;
                    baked.probe_volumes.push(v);
                }
                baked.probe_sh.extend(output.probes);
                self.finalize_bake(baked, runner.task_id);
            }
        }
    }

    /// Install a finished bake: store on the scene, reseed FluxVoxel, upload, save,
    /// clear the task, and flush any imports deferred during the bake.
    fn finalize_bake(&mut self, baked: scene::BakedData, task_id: crate::tasks::TaskId) {
        let n = baked.lightmaps.len();
        self.scene.baked = Some(baked);
        self.rt_gi.invalidate_flux_voxel();
        self.upload_baked_probes();
        self.save_bake();
        self.bake_sizes = None;
        self.tasks.complete_stepped(task_id);
        self.tasks
            .notify(format!("Bake complete: {n} lightmaps"), crate::tasks::NotifyLevel::Info);
        // Apply model imports that finished while the bake held the meshes.
        for (source, asset) in std::mem::take(&mut self.deferred_model_applies) {
            self.apply_imported_model(source, asset);
        }
    }

    /// Realtime-GI preview: while enabled (and the scene isn't baked), re-trace
    /// the auto probe grid from the realtime lights every ~0.2s, blend toward
    /// the previous result (temporal smoothing), and upload so un-baked surfaces
    /// show live indirect bounce. Reuses the bake path tracer (`probes_only`).
    fn update_realtime_gi(&mut self, dt: f32) {
        let vr_active = self.xr_session.is_some();
        if let Some(renderer) = self.renderer.as_mut() {
            self.rt_gi.update(renderer, &mut self.scene, dt, vr_active);
        }
    }

    /// Base path (no extension) for this scene's `.lightmap` / `.lightdata`
    /// sidecars: next to the scene file, or `baked/untitled` if unsaved.
    fn bake_base_path(&self) -> PathBuf {
        match &self.current_scene_path {
            Some(p) => p.with_extension(""),
            None => self.project_root.join("baked/untitled"),
        }
    }

    /// Read the captured reflection cube back from the GPU and persist its 6 faces
    /// to `<scene>.reflprobe` (a small RGBA8 sidecar: magic, size, 6×size²×4 bytes).
    /// A baked reflection probe — loaded on scene open so the cube survives reloads
    /// without re-rendering it. Failures are logged, never fatal.
    /// Handle the inspector's "Bake This Probe" button: find a Reflection Probe
    /// whose transient `bake_now` flag is set, capture from its centre at its
    /// resolution, and queue the readback/save (same path as Bake Reflections).
    fn poll_probe_bake_requests(&mut self) {
        if self.refl_bake_pending.is_some() {
            return; // a bake is already in flight
        }
        let mut req: Option<(usize, u32)> = None;
        for (i, obj) in self.scene.objects.iter_mut().enumerate() {
            for c in obj.components.iter_mut() {
                if let Some(p) = c.as_any_mut().downcast_mut::<zestengine_core::ReflectionProbe>()
                    && p.bake_now
                {
                    p.bake_now = false;
                    req = Some((i, p.resolution));
                    break;
                }
            }
            if req.is_some() {
                break;
            }
        }
        let Some((i, _)) = req else { return };
        let id = self.scene.objects[i].id.raw();
        if let Some(p) = self.scene.reflection_probe_captures().into_iter().find(|p| p.0 == id) {
            self.refl_bake_queue.push(p);
            self.set_status("Baking reflection probe…", true);
        }
    }

    /// The camera inspector's "Create Screen" button.
    fn poll_camera_screen_requests(&mut self) {
        for (i, obj) in self.scene.objects.iter_mut().enumerate() {
            for c in obj.components.iter_mut() {
                if let Some(cam) = c.as_any_mut().downcast_mut::<zestengine_core::CameraComponent>()
                    && cam.create_screen
                {
                    cam.create_screen = false;
                    self.actions.push(EditorAction::CreateCameraScreen(i));
                }
            }
        }
    }

    /// Start the next queued probe bake: a static capture into slot 0, read
    /// back and saved by `save_reflection_bake` once it lands.
    fn step_reflection_bake_queue(&mut self) {
        if self.refl_bake_pending.is_some() || self.refl_bake_current.is_some() || self.refl_bake_queue.is_empty() {
            return;
        }
        let Some(r) = self.renderer.as_mut() else {
            return;
        };
        if r.reflection_capture_in_flight() {
            return;
        }
        let (id, center, res) = self.refl_bake_queue.remove(0);
        r.request_reflection_capture_static(center, res);
        self.refl_bake_current = Some(id);
        self.refl_bake_pending = Some(2);
    }

    fn save_reflection_bake(&mut self) {
        let Some(id) = self.refl_bake_current.take() else {
            return;
        };
        let Some(faces) = self.renderer.as_ref().and_then(|r| r.read_reflection_faces()) else {
            self.set_status("Reflection bake failed (readback)", false);
            return;
        };
        let Ok(faces) = <[zestengine_render::TextureData; 6]>::try_from(faces) else {
            return;
        };
        self.refl_unsaved.push((id, faces));
        if self.refl_bake_queue.is_empty() {
            self.flush_reflection_saves();
        }
    }

    /// Encode (BC6H unless the scene opts out) and merge the captured cubes
    /// into the scene's `.reflprobe` on a worker; `apply_bake_saved` reloads
    /// the slots from the result.
    fn flush_reflection_saves(&mut self) {
        if self.refl_unsaved.is_empty() {
            return;
        }
        let new = std::mem::take(&mut self.refl_unsaved);
        let path = self.bake_base_path().with_extension("reflprobe");
        let compress = self.scene.environment.bake.compress_reflections;
        let label = if compress { "Compressing reflection probes" } else { "Saving reflection probes" };
        self.tasks.spawn(label, crate::tasks::TaskKind::Bake, false, move |_, _| {
            crate::bake_store::save_reflections_job(path, new, compress)
        });
        self.set_status("Saving reflection probes…", true);
    }

    /// Re-encode the scene's existing uncompressed sidecars as BC6H.
    fn recompress_bake(&mut self) {
        let base = self.bake_base_path();
        let serial = self.bake_serial;
        self.tasks.spawn("Compressing bake", crate::tasks::TaskKind::Bake, false, move |_, progress| {
            crate::bake_store::recompress_job(serial, base, progress)
        });
    }

    /// A background sidecar save finished: swap the encoded data in if it still
    /// belongs to the current scene / bake.
    fn apply_bake_saved(&mut self, saved: crate::bake_store::BakeSaved) {
        use crate::tasks::NotifyLevel;
        let base = self.bake_base_path();
        if let Some(lm) = saved.lightmaps {
            if lm.serial == self.bake_serial
                && lm.base == base
                && let Some(b) = self.scene.baked.as_mut()
            {
                b.object_lightmap = lm.object_lightmap;
                b.lightmap_array = Some(std::sync::Arc::new(lm.array));
                b.lightmaps = Vec::new();
                if let Some(r) = self.renderer.as_mut() {
                    b.upload_lightmaps(r);
                }
            }
            self.tasks.notify(lm.stats.describe("Lightmaps saved"), NotifyLevel::Info);
        }
        if let Some(rs) = saved.reflections {
            if rs.seq > self.refl_applied_seq && rs.path == base.with_extension("reflprobe") {
                self.refl_applied_seq = rs.seq;
                // The bake drew into slot 0: reload both slots from the new cubes.
                self.refl_slots.set_baked(rs.set);
            }
            self.set_status("Baked reflection probes", false);
            self.tasks.notify(rs.stats.describe("Reflection probes saved"), NotifyLevel::Info);
        }
        self.bake_sizes = None;
    }

    /// Write the probe data (`.lightdata`, small) now and hand the lightmaps to
    /// a background task that encodes (BC6H unless the scene opts out) and
    /// writes the `.lightmap`, then swaps the encoded array in.
    fn save_bake(&mut self) {
        let Some(baked) = &self.scene.baked else {
            return;
        };
        let base = self.bake_base_path();

        // .lightmap: static GI, one layer per lit object, in layer order.
        let mut layers: Vec<(usize, usize)> = baked
            .object_lightmap
            .iter()
            .filter(|&(_, &layer)| layer < baked.lightmaps.len())
            .map(|(&o, &l)| (o, l))
            .collect();
        layers.sort_by_key(|&(_, l)| l);
        let maps: Vec<(u32, zestengine_render::BakedLightmap)> =
            layers.iter().map(|&(o, l)| (o as u32, baked.lightmaps[l].clone())).collect();
        let (serial, compress) = (self.bake_serial, self.scene.environment.bake.compress_lightmaps);
        crate::bake_store::begin_lightmap_save(serial);
        if maps.is_empty() {
            // Probes-only bake: replace any older lightmaps with an empty file.
            crate::bake_store::write_empty_lightmaps(&base);
        } else {
            let lm_base = base.clone();
            let label = if compress { "Compressing lightmaps" } else { "Saving lightmaps" };
            self.tasks.spawn(label, crate::tasks::TaskKind::Bake, false, move |cancel, progress| {
                crate::bake_store::save_lightmaps_job(serial, lm_base, maps, compress, cancel, progress)
            });
        }
        let Some(baked) = &self.scene.baked else {
            return;
        };

        // .lightdata: probe volumes + SH for dynamic objects.
        let ld = zestengine_assets::LightDataFile {
            volumes: baked
                .probe_volumes
                .iter()
                .map(|v| zestengine_assets::ProbeVolumeData {
                    world_to_local: v.world_to_local.to_cols_array(),
                    size: v.size,
                    counts: [v.counts[0] as u32, v.counts[1] as u32, v.counts[2] as u32],
                    sh_base: v.sh_base as u32,
                    flux_voxel: v.flux_voxel,
                    blend_distance: v.blend_distance,
                    weight: v.weight,
                })
                .collect(),
            probes: baked
                .probe_sh
                .iter()
                .map(|sh| {
                    let c = &sh.coeffs;
                    [
                        c[0][0], c[0][1], c[0][2], c[1][0], c[1][1], c[1][2], c[2][0], c[2][1],
                        c[2][2], c[3][0], c[3][1], c[3][2],
                    ]
                })
                .collect(),
        };
        if let Err(e) = zestengine_assets::save_lightdata(base.with_extension("lightdata"), &ld) {
            tracing::error!("saving .lightdata: {e:#}");
        }
        tracing::info!("wrote {}", base.with_extension("lightdata").display());
    }

    fn toggle_play(&mut self) {
        // The open prefab's objects are in the scene: playing now would run them too.
        if !self.playing && self.prefab_editor.stage.active() {
            self.set_status("Leave Prefab Mode to play", false);
            return;
        }
        // A pending scene load would replace the scene mid-play.
        if !self.playing && self.tasks.blocking().is_some() {
            self.set_status("Wait for loading to finish", false);
            return;
        }
        // External edits to the plugin sources (nvim or another window) change
        // the fingerprint since the last build: rebuild before running them.
        if !self.playing
            && let Some(fp) = self.plugins_fingerprint()
            && self.plugin_fingerprint != Some(fp)
        {
            self.start_plugin_build();
        }
        // Components of a plugin that is still compiling would not exist yet.
        if !self.playing && self.plugin_gate.wait_or_run(PluginWaiter::Play).is_none() {
            self.set_status("Building components… Play starts when done", true);
            return;
        }
        self.play_paused = false;
        self.particles.reset();
        self.water_fx.reset();
        self.flow = zestengine_engine::scene_loader::SceneFlow::default();
        self.content.clear();
        if self.playing {
            self.playing = false;
            self.focus_scene_tab = true;
            self.face_input.set_osc(None);
            if let Some(session) = self.play_session.take() {
                self.audio = session.end();
            }
            if let Some(net) = self.net.as_mut() {
                net.reset_world();
            }
            self.game_cursor_wanted = false;
            self.set_game_cursor(false);
            // Pawn camera possession is play-only; revert to the default camera.
            self.scene.active_camera_override = None;
            self.sequencer_tab.end_play(&mut self.scene);
            if self.play_scene_switched {
                // A component switched scenes during play; the snapshot indices
                // no longer match the loaded scene. Return to the scene we
                // started from (reloaded from disk; unsaved pre-play edits are
                // lost, a known v1 limitation).
                self.play_scene_switched = false;
                self.play_snapshot = None;
                match self.play_origin_scene.take() {
                    Some(origin) => self.load_scene_runtime(&origin),
                    None => tracing::warn!(
                        "scene switched during play from an unsaved scene; staying put"
                    ),
                }
            } else if let Some(snapshot) = self.play_snapshot.take() {
                self.restore_play_snapshot(snapshot);
            }
        } else {
            self.avatar_preview.restore(&mut self.scene);
            self.play_snapshot = Some(PlaySnapshot {
                objects: self.scene.objects.iter().map(play_object_state).collect(),
            });
            self.play_scene_switched = false;
            self.play_origin_scene = None;
            // Avatars with toggles, merges or scoped parameters are built on
            // a copy first; Stop restores the authored components.
            let cap = zestengine_assets::package::ContentLimits::for_kind(zestengine_core::ContentKind::Avatar).max_param_bits;
            let reports = zestengine_engine::avatar_play::build_for_play(&mut self.scene, &self.project_root, &self.shaders, &self.components, cap);
            if let Some(r) = reports.iter().find(|r| !r.errors.is_empty()) {
                self.set_status(format!("Avatar '{}': {}", r.avatar, r.errors[0]), false);
            } else if let Some(r) = reports.first() {
                let warn = if r.warnings.is_empty() { String::new() } else { format!(", {} warning(s) in the log", r.warnings.len()) };
                self.set_status(
                    format!("Avatar '{}' built: {} layers, {} new parameters, {}/{} synced bits{warn}", r.avatar, r.layers, r.generated_params, r.synced_bits, r.max_bits),
                    false,
                );
            }
            // Sprite atlases pack in the background, cached by content hash (2D-4).
            crate::sprite_editor::warm_atlases(&self.project_root);
            self.playing = true;
            self.focus_game_tab = true;
            self.play_time = 0.0;
            if self.sequencer_tab.is_open() {
                self.sequencer_tab.begin_play(&self.scene);
            }
            // Components start on the first tick.
            let mut session = zestengine_engine::play::PlaySession::new(&self.scene, &self.project_root, self.audio.take());
            session.set_mixer(self.project.audio_mixer.clone());
            match zestengine_assets::project_audio_settings(&self.project, &self.project_root) {
                Ok(settings) => session.set_spatial_audio(&settings),
                Err(e) => tracing::error!("loading audio settings: {e:#}"),
            }
            session.set_physics(&self.project.physics, &self.scene);
            self.play_session = Some(session);
            let ft = self.project.tracking_input;
            self.face_input.set_osc(ft.osc_enabled.then_some(ft.osc_port));
        }
    }

    /// Stop-restore after play: objects spawned or destroyed during play are
    /// matched back to the pre-play state by id.
    fn restore_play_snapshot(&mut self, snapshot: PlaySnapshot) {
        let pre: std::collections::HashSet<ObjectId> =
            snapshot.objects.iter().map(|s| s.id).collect();
        // Drop objects spawned during play (highest index first keeps indices valid).
        for i in (0..self.scene.objects.len()).rev() {
            if i < self.scene.objects.len() && !pre.contains(&self.scene.objects[i].id) {
                self.scene.remove_object(i);
            }
        }
        let alive: std::collections::HashSet<ObjectId> =
            self.scene.objects.iter().map(|o| o.id).collect();
        if snapshot.objects.iter().any(|s| !alive.contains(&s.id)) {
            // Objects were destroyed during play; the in-memory scene can't
            // recreate them, so reload it from disk when possible.
            if let Some(path) = self.current_scene_path.clone()
                && !self.scene_dirty
            {
                self.load_scene_runtime(&path);
                return;
            }
            tracing::warn!("objects destroyed during play can't be restored (unsaved scene)");
        }
        let registry = &self.components;
        let index = self.scene.id_index();
        for state in snapshot.objects {
            if let Some(&i) = index.get(&state.id) {
                let object = &mut self.scene.objects[i];
                object.translation = state.translation;
                object.rotation = state.rotation;
                object.scale = state.scale;
                object.enabled = state.enabled;
                object.load_components(&state.components, registry);
            }
        }
    }

    /// Grab / release the OS cursor for gameplay mouse-look.
    fn set_game_cursor(&mut self, locked: bool) {
        if self.game_cursor_locked == locked {
            return;
        }
        self.game_cursor_locked = locked;
        if let Some(window) = &self.window {
            window.set_cursor_visible(!locked);
            let grab = if locked {
                window
                    .set_cursor_grab(CursorGrabMode::Locked)
                    .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
            } else {
                window.set_cursor_grab(CursorGrabMode::None)
            };
            if let Err(e) = grab {
                tracing::debug!("cursor grab: {e}");
            }
        }
    }

    /// Apply deferred component requests gathered during a lifecycle pass. The
    /// last `LoadScene` wins (loading once); a switch during play continues
    /// playing in the new scene.
    fn apply_component_commands(&mut self, commands: Vec<ComponentCommand>) {
        let mut load = None;
        let mut quit = false;
        for c in commands {
            let Some(c) = zestengine_engine::net_sync::route_command(&mut self.net, &self.scene, c) else {
                continue;
            };
            match c {
                ComponentCommand::LoadScene(rel) => load = Some(rel),
                ComponentCommand::PreloadScene(rel) => {
                    if let Some(r) = self.renderer.as_ref() {
                        self.flow.preload(&rel, &self.project_root, Some(r), &self.scene, &self.shaders);
                    }
                }
                ComponentCommand::LoadSceneAdditive(rel) => {
                    if let Some(r) = self.renderer.as_ref() {
                        self.flow.load_additive(&rel, &self.project_root, Some(r));
                    }
                }
                ComponentCommand::UnloadScene(rel) => {
                    if self.flow.unload(&rel, &mut self.scene)
                        && let Some(session) = self.play_session.as_mut()
                    {
                        session.structure_changed();
                    }
                }
                ComponentCommand::KeepOnSceneChange(id, keep) => {
                    if keep {
                        self.flow.kept.insert(id);
                    } else {
                        self.flow.kept.remove(&id);
                    }
                }
                ComponentCommand::SetResolution(w, h) => {
                    if let Some(window) = self.window.as_ref() {
                        let _ = window.request_inner_size(winit::dpi::PhysicalSize::new(w, h));
                    }
                    if let Some(r) = self.renderer.as_mut() {
                        r.resize();
                    }
                }
                ComponentCommand::SetVsync(on) => {
                    if let Some(r) = self.renderer.as_mut() {
                        r.set_vsync(on);
                    }
                }
                ComponentCommand::SetShadowResolution(res) => {
                    if let Some(r) = self.renderer.as_mut()
                        && let Err(e) = r.set_shadow_resolution(res.clamp(256, 8192))
                    {
                        tracing::warn!("set shadow resolution: {e:#}");
                    }
                }
                ComponentCommand::SetCursorLocked(locked) => {
                    self.game_cursor_wanted = locked;
                    self.set_game_cursor(locked);
                }
                // In the editor, quitting the game ends Play mode.
                ComponentCommand::Quit => quit = true,
                ComponentCommand::CalibrateTrackers => self.body.calibrate_now(),
                ComponentCommand::ClearTrackerCalibration => self.body.clear_calibration(),
                ComponentCommand::Body(c) => self.body.command(c),
                ComponentCommand::SetOscFaceTracking(port) => self.face_input.set_osc(port),
                ComponentCommand::Content(c) => self.content.queue(c),
                ComponentCommand::Input(c) => self.input.apply(c),
                other => tracing::warn!("unhandled component command {other:?}"),
            }
        }
        if quit {
            if self.playing {
                self.toggle_play();
            }
            return;
        }
        let Some(rel) = load else {
            return;
        };
        if self.playing && !self.play_scene_switched {
            self.play_origin_scene = self.current_scene_path.clone();
            self.play_scene_switched = true;
        }
        let path = self.project_root.join(&rel);
        if !self.playing {
            self.load_scene_runtime(&path);
            return;
        }
        // Play mode: preload-aware background switch that carries kept
        // objects; the current scene keeps playing until it's installed.
        if let Some(renderer) = self.renderer.as_ref() {
            self.flow
                .request_switch(&rel, &self.project_root, Some(renderer), &self.scene, &self.shaders);
        }
    }

    /// Install a play-mode scene switch once its background load finished.
    fn finish_play_switch(&mut self) {
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        let Some(result) =
            self.flow
                .poll_switch(Some(renderer), &self.components, &mut self.shaders, &self.project_root, &self.scene)
        else {
            return;
        };
        match result {
            Ok((rel, scene, post)) => {
                // Content worlds switch without a LoadScene command; remember
                // where Play started so Stop returns there.
                if self.playing && !self.play_scene_switched {
                    self.play_origin_scene = self.current_scene_path.clone();
                    self.play_scene_switched = true;
                }
                self.reset_render_caches();
                self.scene = scene;
                self.content.scene_replaced(&rel);
                self.scene.ensure_camera_components(&self.components);
                self.scene.ensure_light_components(&self.components);
                self.scene.ensure_camera_ids();
                self.selection = Selection::None;
                self.multi_objects.clear();
                self.current_scene_path = Some(self.project_root.join(rel));
                self.post_load = Some(post);
                if let Some(session) = self.play_session.as_mut() {
                    session.scene_changed(&self.scene);
                }
            }
            Err(e) => tracing::error!("switching scene: {e:#}"),
        }
    }

    /// Load a scene by absolute path, replacing the current one. Used by
    /// runtime (play-mode) scene switches and Stop-restore. Does not run start
    /// Input Bindings window (2C): pick the active control scheme, view/clear
    /// each action's bindings, and rebind by capturing the next key/mouse press.
    /// Layers settings (Unity-style): edit the 32 layer names, the symmetric
    /// Dedicated Weapon Pose editor: edit the selected object's Weapon Hold
    /// idle/aim poses with an immediate preview (selecting a slot loads it).
    fn pose_editor_window(&mut self, ctx: &egui::Context) {
        if ctx.data(|d| d.get_temp::<bool>(egui::Id::new(crate::OPEN_POSE_EDITOR_KEY)).unwrap_or(false)) {
            ctx.data_mut(|d| d.remove::<bool>(egui::Id::new(crate::OPEN_POSE_EDITOR_KEY)));
            self.show_pose_editor = true;
        }
        let index = match self.selection {
            Selection::Object(i) => Some(i),
            _ => None,
        };
        let mut open = self.show_pose_editor;
        let mut dirty = false;
        egui::Window::new("Weapon Pose")
            .open(&mut open)
            .resizable(true)
            .default_width(360.0)
            .show(ctx, |ui| {
                use zestengine_core::{WeaponFollow, WeaponHold, WeaponPoseRequest, WeaponPoseSlot};
                use crate::widgets;
                let Some(index) = index.filter(|&i| i < self.scene.objects.len()) else {
                    ui.label(egui::RichText::new("Select an object with a Weapon Hold to edit its pose.").weak());
                    return;
                };
                let Some(hold) = self.scene.objects[index]
                    .components
                    .iter_mut()
                    .find_map(|c| c.as_any_mut().downcast_mut::<WeaponHold>())
                else {
                    ui.label(egui::RichText::new("The selected object has no Weapon Hold.").weak());
                    return;
                };
                widgets::property_row(ui, "Pose", |ui| {
                    if widgets::segmented(ui, &mut hold.editing, &[(WeaponPoseSlot::Idle, "Idle"), (WeaponPoseSlot::Aim, "Aim")]).changed() {
                        hold.request = Some(WeaponPoseRequest::Load);
                        dirty = true;
                    }
                });
                widgets::property_row(ui, "", |ui| {
                    if widgets::secondary_button(ui, None, "Load pose").on_hover_text("Move the weapon to this pose").clicked() {
                        hold.request = Some(WeaponPoseRequest::Load);
                        dirty = true;
                    }
                    if widgets::secondary_button(ui, None, "Store pose").on_hover_text("Save where the weapon is now as this pose").clicked() {
                        hold.request = Some(WeaponPoseRequest::Store);
                        dirty = true;
                    }
                });
                let slot = hold.editing;
                let pose = hold.pose_mut(slot);
                widgets::property_row(ui, "Follows", |ui| {
                    egui::ComboBox::from_id_salt("pose-editor-follow")
                        .selected_text(match pose.follow {
                            WeaponFollow::Bone => "Anchor bone (sways)",
                            WeaponFollow::Avatar => "Avatar (no sway)",
                            WeaponFollow::World => "World (fixed)",
                        })
                        .show_ui(ui, |ui| {
                            dirty |= ui.selectable_value(&mut pose.follow, WeaponFollow::Bone, "Anchor bone (sways)").changed();
                            dirty |= ui.selectable_value(&mut pose.follow, WeaponFollow::Avatar, "Avatar (no sway)").changed();
                            dirty |= ui.selectable_value(&mut pose.follow, WeaponFollow::World, "World (fixed)").changed();
                        });
                });
                widgets::property_row(ui, "Position", |ui| dirty |= widgets::vec3_field(ui, &mut pose.position, 2, 0.02));
                widgets::property_row(ui, "Rotation", |ui| dirty |= widgets::vec3_field(ui, &mut pose.rotation, 2, 0.5));
                widgets::info_note(ui, "Move the weapon and its hand targets with the gizmo in the viewport, then Store pose. Selecting Idle/Aim previews it.");
            });
        self.show_pose_editor = open;
        if dirty {
            self.scene_dirty = true;
        }
    }

    /// collision matrix (which layers collide in physics), and toggle per-layer
    /// viewport visibility (the editor's render culling). Stored in the scene.
    fn layers_window(&mut self, ctx: &egui::Context) {
        use zestengine_core::NUM_LAYERS;
        let mut open = self.show_layers;
        egui::Window::new("Layers")
            .open(&mut open)
            .resizable(true)
            .default_width(520.0)
            .show(ctx, |ui| {
                ui.label(
                    egui::RichText::new(
                        "Layer names, the physics collision matrix, and viewport visibility. \
                         Assign an object's layer in the Inspector. Saved with the scene.",
                    )
                    .weak()
                    .small(),
                );
                ui.separator();
                // Highest layer index actually in use (+a couple spare rows), so
                // the matrix stays compact instead of always showing all 32.
                // Consistent with the object layer dropdown + camera culling mask.
                // The matrix can name beyond `shown` to reveal more layers (naming a
                // higher layer raises shown_count everywhere).
                let max_used = self
                    .scene
                    .objects
                    .iter()
                    .map(|o| o.layer as usize)
                    .max()
                    .unwrap_or(0);
                let shown = self.scene.layers.shown_count().max(max_used + 1);

                egui::ScrollArea::both().max_height(440.0).show(ui, |ui| {
                    ui.heading("Names & visibility");
                    egui::Grid::new("layer-names").striped(true).show(ui, |ui| {
                        ui.label("#");
                        ui.label("Name");
                        ui.label("Visible");
                        ui.end_row();
                        // The NAMES list shows ALL layers (scrollable) so you can name
                        // any of the 32; naming a higher one raises `shown_count`, which
                        // reveals it in the matrix + camera mask + object dropdown.
                        for l in 0..NUM_LAYERS {
                            ui.label(format!("{l}"));
                            let name = self
                                .scene
                                .layers
                                .names
                                .get_mut(l)
                                .map(|s| s as &mut String);
                            if let Some(name) = name {
                                // Layer 0 ("Default") keeps its name fixed.
                                ui.add_enabled(l != 0, egui::TextEdit::singleline(name));
                            } else {
                                ui.label("-");
                            }
                            let bit = 1u32 << (l as u32 & 31);
                            let mut vis = self.scene.visible_layers & bit != 0;
                            if ui.checkbox(&mut vis, "").changed() {
                                if vis {
                                    self.scene.visible_layers |= bit;
                                } else {
                                    self.scene.visible_layers &= !bit;
                                }
                            }
                            ui.end_row();
                        }
                    });

                    ui.separator();
                    ui.heading("Collision matrix");
                    ui.label(
                        egui::RichText::new("Checked = the two layers collide (physics).")
                            .weak()
                            .small(),
                    );
                    egui::Grid::new("layer-matrix").striped(true).show(ui, |ui| {
                        // Header row: blank corner + column labels.
                        ui.label("");
                        for c in 0..shown {
                            ui.label(egui::RichText::new(format!("{c}")).small());
                        }
                        ui.end_row();
                        for r in 0..shown {
                            ui.label(
                                egui::RichText::new(self.scene.layers.layer_name(r as u8)).small(),
                            );
                            for c in 0..shown {
                                // Lower triangle only (symmetric); blank the rest.
                                if c > r {
                                    ui.label("");
                                    continue;
                                }
                                let mut on = self.scene.layers.collide(r as u8, c as u8);
                                if ui.checkbox(&mut on, "").changed() {
                                    self.scene.layers.set_collide(r as u8, c as u8, on);
                                }
                            }
                            ui.end_row();
                        }
                    });
                    // Breathing room so the horizontal scrollbar (the 32-wide matrix
                    // needs one) sits below the grid instead of over its last row.
                    ui.add_space(18.0);
                });

                ui.separator();
                egui::CollapsingHeader::new("Sorting layers (2D)").default_open(false).show(ui, |ui| {
                    if crate::sorting_layers_ui(ui, &mut self.scene.layers.sorting) {
                        self.scene_dirty = true;
                    }
                });
                ui.separator();
                if ui.button("Reset to defaults").clicked() {
                    self.scene.layers = zestengine_core::LayerSettings::default();
                    self.scene.visible_layers = zestengine_core::all_layers_mask();
                }
            });
        self.show_layers = open;

        // Audio mixer window: a fader + mute per bus. `effective_gain(bus)` drives
        // playback volume (the audio backend multiplies each source by it).
        let mut mixer_open = self.show_mixer;
        egui::Window::new("Audio Mixer")
            .open(&mut mixer_open)
            .resizable(false)
            .show(ctx, |ui| {
                ui.label(
                    egui::RichText::new("Buses · linear gain, chained to Master")
                        .small()
                        .weak(),
                );
                // Grid so the mute icon / name / fader / gain columns line up.
                egui::Grid::new("audio-mixer-grid")
                    .num_columns(4)
                    .spacing([10.0, 6.0])
                    .show(ui, |ui| {
                        for name in self.project.audio_mixer.bus_names() {
                            // Mute toggle as a phosphor speaker icon (no checkbox box).
                            let muted = self.project.audio_mixer.is_muted(&name);
                            let icon = if muted { ph::SPEAKER_NONE } else { ph::SPEAKER_HIGH };
                            if ui
                                .add(egui::Button::new(icon).frame(false))
                                .on_hover_text(if muted { "Unmute" } else { "Mute" })
                                .clicked()
                            {
                                self.project.audio_mixer.set_muted(&name, !muted);
                                self.mixer_changed = true;
                            }
                            ui.label(egui::RichText::new(&name).monospace());
                            let mut v = self.project.audio_mixer.volume(&name);
                            ui.spacing_mut().slider_width = 130.0;
                            if crate::widgets::slider(
                                ui,
                                egui::Slider::new(&mut v, 0.0..=1.0).fixed_decimals(2))
                                .changed()
                            {
                                self.project.audio_mixer.set_volume(&name, v);
                                self.mixer_changed = true;
                            }
                            ui.label(
                                egui::RichText::new(format!(
                                    "{} {:.2}",
                                    ph::ARROW_RIGHT,
                                    self.project.audio_mixer.effective_gain(&name)
                                ))
                                .weak(),
                            );
                            ui.end_row();
                        }
                    });
            });
        self.show_mixer = mixer_open;
        if self.mixer_changed {
            self.mixer_changed = false;
            if let Some(session) = self.play_session.as_mut() {
                session.set_mixer(self.project.audio_mixer.clone());
            }
            self.save_project();
        }
    }

    /// Modal shown when a bake found static models without a lightmap UV. Offers
    /// to generate a non-overlapping unwrap (persisted via a `.lmuv` marker, then
    /// reproduced on reload), bake without them, or cancel.
    fn lightmap_uv_window(&mut self, ctx: &egui::Context) {
        let Some(models) = self.pending_lm_uv_models.clone() else {
            return;
        };
        let mut choice: Option<u8> = None;
        egui::Window::new("Generate Lightmap UVs?")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.label(
                    "These models have no lightmap UV (second UV set). Baking their \
                     overlapping texture UVs would produce a garbage lightmap, so the \
                     bake skips them:",
                );
                for m in &models {
                    ui.label(format!("  • {m}"));
                }
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new(
                        "Generate a non-overlapping lightmap UV for them? It's saved as a \
                         .lmuv marker next to the model and regenerated on load, so it \
                         persists. (Re-imports the scene.)",
                    )
                    .small()
                    .weak(),
                );
                crate::widgets::dialog_footer(ui, |ui| {
                    if crate::widgets::primary_button(ui, None, "Generate UVs & Reload").clicked() {
                        choice = Some(0);
                    }
                    if crate::widgets::secondary_button(ui, None, "Bake Without Them").clicked() {
                        choice = Some(1);
                    }
                    if crate::widgets::secondary_button(ui, None, "Cancel").clicked() {
                        choice = Some(2);
                    }
                });
            });
        match choice {
            Some(0) => {
                self.pending_lm_uv_models = None;
                for m in &models {
                    if let Err(e) =
                        zestengine_assets::set_lightmap_uv_marker(self.project_root.join(m), true)
                    {
                        tracing::error!("writing lightmap-UV marker for {m}: {e:#}");
                    }
                }
                if self.playing {
                    self.set_status("Stop play mode, then bake again", false);
                } else if self.scene_dirty {
                    self.set_status("Save the scene first, then bake again", false);
                } else if let Some(scene) = self.current_scene_path.clone() {
                    self.load_scene_runtime(&scene);
                    self.set_status("Generated lightmap UVs — click Bake again", false);
                } else {
                    self.set_status("Save the scene first, then bake again", false);
                }
            }
            Some(1) => {
                self.pending_lm_uv_models = None;
                self.do_bake();
            }
            Some(2) => self.pending_lm_uv_models = None,
            _ => {}
        }
    }

    /// The Input Bindings window: the same editor as Project Settings > Input.
    /// Saves to `project.zestengine` on every change.
    fn bindings_window(&mut self, ctx: &egui::Context) {
        let mut open = self.show_bindings;
        egui::Window::new("Input Bindings")
            .open(&mut open)
            .resizable(true)
            .default_width(460.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().max_height(560.0).show(ui, |ui| self.input_settings_ui(ui, "bindings-window"));
            });
        if !open {
            self.bindings_editor.cancel_rebind();
        }
        self.show_bindings = open;
    }

    /// Project Settings > Input: the bindings editor plus where players'
    /// own remaps live. Saves the project when the bindings change.
    fn input_settings_ui(&mut self, ui: &mut egui::Ui, id: &str) {
        if bindings_section_ui(ui, id, &mut self.project, &mut self.bindings_editor, &mut self.input) {
            self.bindings_changed();
        }
    }

    /// The project's bindings were edited: push them to the live input
    /// manager (Play mode picks them up at once) and save.
    fn bindings_changed(&mut self) {
        self.input.set_project_bindings(self.project.bindings.clone());
        self.save_project();
    }

    /// Load the project's bindings (upgrading old ones) and its Play-mode
    /// remaps into the input manager.
    fn sync_input_bindings(&mut self) {
        self.project.bindings.upgrade();
        self.input.set_project_bindings(self.project.bindings.clone());
        self.input.set_overrides_path(Some(zestengine_engine::input_engine::editor_bindings_path(&self.project_root)));
        // The editor's player identity: one per install, like a game's.
        if let Some(dir) = zestengine_engine::input_engine::player_bindings_path("ZestEngine Editor").parent() {
            zestengine_engine::net::set_identity_dir(dir.to_path_buf());
        }
        self.bindings_editor = crate::BindingsEditor::default();
    }

    /// Performance ranks and avatar safety settings (GEO-2).
    fn performance_window(&mut self, ctx: &egui::Context) {
        if !self.show_performance {
            return;
        }
        let selected = match self.selection {
            Selection::Object(i) if i < self.scene.objects.len() => Some(i),
            _ => None,
        };
        let mut open = self.show_performance;
        self.scene.set_perf_root(&self.project_root);
        crate::perf_window::performance_window(ctx, &mut open, &mut self.perf_ui, &mut self.scene, selected);
        self.show_performance = open;
    }

    /// Network panel (2G): host or join a session, see status, disconnect.
    fn network_window(&mut self, ctx: &egui::Context) {
        let mut open = self.show_network;
        let err = crate::net_window::network_window(ctx, &mut open, &mut self.net, &mut self.net_addr, &mut self.net_ui, &self.scene);
        self.show_network = open;
        if let Some(e) = err {
            self.set_status(e, false);
        }
        if std::mem::take(&mut self.net_ui.local_server_request) {
            self.play_as_server();
        }
        // The local server lives as long as the session that joined it.
        if self.net.is_none() {
            self.local_server = None;
        }
    }

    /// "Play as Server": run the saved scene as a local dedicated server and join it.
    fn play_as_server(&mut self) {
        if self.net.is_some() {
            self.set_status("Leave the current session first".to_string(), false);
            return;
        }
        let Some(path) = self.current_scene_path.clone() else {
            self.set_status("Save the scene first: the local server loads it from the project".to_string(), false);
            return;
        };
        if self.scene_dirty {
            self.set_status("The scene has unsaved changes: the local server runs the saved file (save first to include them)".to_string(), false);
        }
        let Ok(rel) = path.strip_prefix(&self.project_root).map(|p| p.to_string_lossy().replace('\\', "/")) else {
            self.set_status("The scene is outside the project folder".to_string(), false);
            return;
        };
        let res = {
            // The server thread builds its own registry: hand it the loaded plugins' registrations.
            let regs = self.plugins.registrations();
            zestengine_engine::local_server::LocalServer::start(self.project_root.clone(), Some(rel), move |r| {
                for f in &regs {
                    f(r);
                }
            })
        }; match res {
            Ok(server) => match crate::net_window::join_local(&self.net_ui, &self.scene, &server.addr().to_string()) {
                Ok(session) => {
                    self.set_status(format!("Local server on {}; joined as a player", server.addr()), false);
                    self.net = Some(session);
                    self.local_server = Some(server);
                }
                Err(e) => self.set_status(format!("Joining the local server failed: {e}"), false),
            },
            Err(e) => self.set_status(format!("{e:#}"), false),
        }
    }

    /// hooks (the caller decides) and does not persist to project.zestengine.
    /// New Project + Project Settings modal windows.
    fn project_windows(&mut self, ctx: &egui::Context) {
        // ---- New Project ----
        let mut open = self.show_new_project;
        let mut do_create = false;
        let mut cancel = false;
        egui::Window::new("New Project")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                do_create = crate::new_project_ui::new_project_ui(
                    ui,
                    crate::new_project_ui::NewProjectState {
                        parent: &mut self.new_project_parent,
                        name: &mut self.new_project_name,
                        template: &mut self.new_project_template,
                        filter: &mut self.new_project_filter,
                    },
                    &mut cancel,
                );
            });
        self.show_new_project = open && !do_create && !cancel;
        if do_create {
            self.actions.push(EditorAction::NewProject {
                parent: PathBuf::from(self.new_project_parent.clone()),
                name: self.new_project_name.trim().to_string(),
                template: self.new_project_template.clone(),
            });
        }

        // ---- Open Project ----
        let mut open = self.show_open_project;
        let mut do_open = false;
        egui::Window::new("Open Project")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                ui.label("Open an existing zestengine project folder (one containing project.zestengine).");
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    ui.label("Folder");
                    ui.text_edit_singleline(&mut self.open_project_path);
                });
                let dir = Path::new(&self.open_project_path);
                let valid = dir.join("project.zestengine").is_file();
                ui.add_space(4.0);
                if !valid && !self.open_project_path.trim().is_empty() {
                    ui.label(
                        egui::RichText::new("No project.zestengine in that folder.")
                            .small()
                            .color(egui::Color32::from_rgb(220, 160, 90)),
                    );
                }
                crate::widgets::dialog_footer(ui, |ui| {
                    ui.add_enabled_ui(valid, |ui| {
                        if crate::widgets::primary_button(ui, None, "Open").clicked() {
                            do_open = true;
                        }
                    });
                });
            });
        self.show_open_project = open && !do_open;
        if do_open {
            self.actions
                .push(EditorAction::OpenProject(PathBuf::from(self.open_project_path.clone())));
        }


        // ---- Unapplied import settings ----
        // The selection moved off a Model Import with edits (any route, not just
        // the file list): ask before they are forgotten. Cancel goes back.
        if self.import_prompt.is_none()
            && let Some(fm) = self.file_meta.as_ref().filter(|f| f.dirty)
            && self.selection != Selection::File(fm.path.clone())
        {
            self.import_prompt = Some(ImportPrompt { path: fm.path.clone(), resume: None, restore_selection: true, quit: false });
        }
        if let Some(prompt) = self.import_prompt.take() {
            let name = prompt.path.file_name().map_or_else(|| prompt.path.display().to_string(), |n| n.to_string_lossy().into_owned());
            let mut open = true;
            // 0 = Apply, 1 = Revert, 2 = Cancel.
            let mut choice: Option<u8> = None;
            egui::Window::new("Unapplied import settings")
                .open(&mut open)
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .show(ctx, |ui| {
                    ui.label(format!("Unapplied import settings for '{name}'"));
                    crate::widgets::dialog_footer(ui, |ui| {
                        if crate::widgets::primary_button(ui, None, "Apply").clicked() {
                            choice = Some(0);
                        }
                        if crate::widgets::secondary_button(ui, None, "Revert").clicked() {
                            choice = Some(1);
                        }
                        if crate::widgets::secondary_button(ui, None, "Cancel").clicked() {
                            choice = Some(2);
                        }
                    });
                });
            if !open {
                choice = Some(2); // the window's close button is Cancel
            }
            match choice {
                None => self.import_prompt = Some(prompt),
                Some(2) => {
                    if prompt.restore_selection {
                        self.selection = Selection::File(prompt.path.clone());
                    }
                }
                Some(c) => {
                    if c == 0 {
                        // Written now (not queued) so a quit can't outrun it.
                        if let Some(fm) = self.file_meta.as_mut()
                            && let Err(e) = fm.apply()
                        {
                            tracing::error!("saving .meta: {e:#}");
                        }
                        if !prompt.quit {
                            self.actions.push(if is_texture_path(&prompt.path) {
                                EditorAction::ReimportTexture(prompt.path.clone())
                            } else {
                                EditorAction::ReimportModel(prompt.path.clone())
                            });
                        }
                    } else if let Some(fm) = self.file_meta.as_mut() {
                        fm.revert();
                    }
                    if prompt.quit {
                        if self.scene_dirty {
                            self.show_quit_dialog = true;
                        } else {
                            self.save_project();
                            self.quit_after_frame = true;
                        }
                    }
                    if let Some(action) = prompt.resume {
                        self.actions.push(action);
                    }
                }
            }
        }

        // ---- Unsaved-changes (quit) dialog ----
        if self.show_quit_dialog {
            let mut open = true;
            egui::Window::new("Unsaved changes")
                .open(&mut open)
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .show(ctx, |ui| {
                    ui.label("This scene has unsaved changes. Save before quitting?");
                    crate::widgets::dialog_footer(ui, |ui| {
                        if crate::widgets::primary_button(ui, None, "Save & Quit").clicked() {
                            self.actions.push(EditorAction::SaveScene(None));
                            self.quit_after_frame = true;
                            self.show_quit_dialog = false;
                        }
                        if crate::widgets::danger_button(ui, None, "Discard & Quit").clicked() {
                            self.quit_after_frame = true;
                            self.show_quit_dialog = false;
                        }
                        if crate::widgets::secondary_button(ui, None, "Cancel").clicked() {
                            self.show_quit_dialog = false;
                            self.relaunch_after_quit = false;
                        }
                    });
                });
            // Window close (✕) acts as Cancel.
            if !open {
                self.show_quit_dialog = false;
                self.relaunch_after_quit = false;
            }
        }

        // ---- Unsaved changes before switching project ----
        if let Some(root) = self.pending_project_switch.clone() {
            let mut open = true;
            let mut decided = false;
            egui::Window::new("Switch project")
                .open(&mut open)
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                .show(ctx, |ui| {
                    let name = root.file_name().map_or_else(|| root.display().to_string(), |n| n.to_string_lossy().into_owned());
                    ui.label(format!("This scene has unsaved changes. Save before opening {name}?"));
                    crate::widgets::dialog_footer(ui, |ui| {
                        if crate::widgets::primary_button(ui, None, "Save & Switch").clicked() {
                            // Actions run in order: the save lands, then the switch.
                            self.actions.push(EditorAction::SaveScene(None));
                            self.actions.push(EditorAction::OpenProject(root.clone()));
                            decided = true;
                        }
                        if crate::widgets::secondary_button(ui, None, "Discard & Switch").clicked() {
                            self.scene_dirty = false;
                            self.actions.push(EditorAction::OpenProject(root.clone()));
                            decided = true;
                        }
                        if crate::widgets::secondary_button(ui, None, "Cancel").clicked() {
                            decided = true;
                        }
                    });
                });
            if decided || !open {
                self.pending_project_switch = None;
            }
        }
    }

    /// Point the editor at a different project root: reload its project file,
    /// file browser, plugins, and boot/last scene. Used after New Project.
    fn switch_project(&mut self, root: PathBuf) {
        // Keep the working folder on the project, as `zestengine <dir>` starts it.
        if let Err(e) = std::env::set_current_dir(&root) {
            tracing::warn!("entering {}: {e}", root.display());
        }
        self.project_root = root.clone();
        self.file_browser = FileBrowser::new(root.clone());
        self.current_scene_path = None;
        self.scene_dirty = false;
        self.selection = Selection::None;
        self.file_material = None;
        self.undo_stack.clear();
        self.shaders = ShaderLibrary::default();
        self.scene = LoadedScene::empty();

        // Fresh registry + plugins for the new project.
        self.components = zestengine_core::ComponentRegistry::with_builtins();
        self.editor_components = crate::EditorComponents::with_builtins();
        self.plugins = plugins::PluginHost::default();
        self.plugin_build_error = None;
        // Built on a worker; the boot scene below parks until it is done.
        if plugins::PluginHost::any_plugins(&root) {
            self.request_plugin_build();
        }

        self.project = zestengine_assets::load_project_file(&root).unwrap_or_default();
        self.migrate_audio_settings();
        self.project.apply_role_overrides(&mut self.components);
        self.sync_input_bindings();
        let boot = self
            .project
            .boot_scene
            .clone()
            .or_else(|| self.project.last_scene.clone());
        if let Some(rel) = boot {
            let abs = root.join(&rel);
            self.load_scene_runtime(&abs);
        }
        if let Some(window) = self.window.as_ref() {
            window.set_title(&format!("{} — zestengine", self.project.name));
        }
        self.record_recent_project();
        self.set_status(format!("opened project {}", self.project.name), false);
    }

    /// Reload or replace the scene from disk in the background (progress
    /// modal; the install happens when the worker finishes).
    fn load_scene_runtime(&mut self, path: &Path) {
        self.spawn_scene_load(path.to_path_buf(), true);
    }

    /// Restore the scene-tree collapsed rows from saved object ids (map ids back
    /// to current indices; unknown ids are dropped).
    fn restore_collapsed(&mut self, collapsed: &[String]) {
        use std::collections::HashMap;
        let by_id: HashMap<String, usize> = self
            .scene
            .objects
            .iter()
            .enumerate()
            .map(|(i, o)| (o.id.to_string(), i))
            .collect();
        let set: Vec<usize> = collapsed
            .iter()
            .filter_map(|id| by_id.get(id).copied())
            .collect();
        self.scene_panel.set_collapsed(set);
    }

    /// Snapshot the skybox image sources (equirect + cubemap faces).
    fn skybox_state(&self) -> SkyboxState {
        SkyboxState {
            equirect: self.scene.skybox.clone(),
            faces: self.scene.environment.skybox_faces.clone(),
        }
    }

    /// Record a skybox image change into the shared undo stack.
    fn record_skybox(&mut self, before: SkyboxState) {
        let after = self.skybox_state();
        self.undo_stack.record(
            UndoEntry::Skybox { before: Box::new(before), after: Box::new(after) },
            false,
        );
    }

    /// Push the scene's skybox (or the procedural sky) to the renderer.
    fn apply_skybox(&mut self) {
        let skybox = self.scene.skybox.clone();
        let faces = self.scene.environment.skybox_faces.clone();
        let project_root = self.project_root.clone();
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        // Cubemap (6-face) skybox takes precedence over the equirect one — but only
        // once ALL six faces are assigned (a partial set falls back to equirect).
        if let Some(faces) = faces.filter(|f| f.iter().all(|s| !s.is_empty())) {
            let loaded: Result<Vec<_>, _> = faces
                .iter()
                .map(|rel| zestengine_assets::load_texture_file(project_root.join(rel), true))
                .collect();
            match loaded {
                Ok(data) => {
                    let refs: [&zestengine_render::TextureData; 6] = std::array::from_fn(|i| &data[i]);
                    if let Err(e) = renderer.set_skybox_cube(refs) {
                        tracing::error!("setting cubemap skybox: {e:#}");
                    }
                    return;
                }
                Err(e) => tracing::error!("loading cubemap skybox: {e:#}"),
            }
        }
        match skybox {
            Some(rel) => {
                let abs = project_root.join(&rel);
                let t0 = Instant::now();
                match zestengine_assets::load_texture_file(&abs, true) {
                    Ok(data) => {
                        let decoded = t0.elapsed();
                        if let Err(e) = renderer.set_skybox(Some(&data)) {
                            tracing::error!("setting skybox: {e:#}");
                        }
                        tracing::info!(
                            "skybox {rel}: decode {:?} + set/env-cube {:?} ({}x{})",
                            decoded,
                            t0.elapsed() - decoded,
                            data.width,
                            data.height
                        );
                    }
                    Err(e) => {
                        tracing::error!("loading skybox {rel}: {e:#}");
                        let _ = renderer.set_skybox(None);
                    }
                }
            }
            None => {
                let _ = renderer.set_skybox(None);
            }
        }
    }

    /// Persist project.zestengine: current settings + the open scene.
    fn save_project(&mut self) {
        if let Some(renderer) = &self.renderer {
            self.project.settings.vsync = renderer.vsync();
        }
        self.project.settings.show_stats = self.show_stats;
        self.project.settings.show_stats_overlay = self.show_stats_overlay;
        self.project.settings.snap = self.gizmo.snap;
        self.project.settings.grid_size = self.gizmo.grid_size;
        self.project.settings.align_to_surface = self.gizmo.align_to_surface;
        self.project.last_scene = self
            .scene_path_for_project()
            .as_ref()
            .map(|p| relative_to(p, &self.project_root));
        if let Err(e) = zestengine_assets::save_project_file(&self.project_root, &self.project) {
            tracing::error!("saving project.zestengine: {e:#}");
        }
    }

    /// Rescan project `.frag` files for the shader picker and hot-reload any
    /// loaded shaders whose files changed. Called at most every 2 seconds.
    /// Set the global status-bar message (`spinner` shows a busy indicator).
    fn set_status(&mut self, msg: impl Into<String>, spinner: bool) {
        self.status = Some((msg.into(), Instant::now(), spinner));
    }

    /// While a heavy job is queued, dim the screen and show a centered
    /// "working…" card. It paints this frame; the blocking job runs right after
    /// render, so the user sees it's busy instead of a frozen window.
    fn busy_overlay(&self, ctx: &egui::Context) {
        // A pending blocking job (plugin compile) or a blocking background task
        // (focus re-import) dims the UI and blocks interaction until it's done.
        let label = match &self.pending_job {
            Some(PendingJob::ReloadPlugins) => "Compiling components…".to_string(),
            None => match self.tasks.blocking() {
                Some(h) => h.label.clone(),
                None => return,
            },
        };
        let label = label.as_str();
        // Dim the whole UI.
        let screen = ctx.content_rect();
        egui::Area::new("zestengine-busy-dim".into())
            .order(egui::Order::Foreground)
            .fixed_pos(screen.min)
            .show(ctx, |ui| {
                ui.painter()
                    .rect_filled(screen, 0.0, egui::Color32::from_black_alpha(140));
            });
        // Centered card with a spinner + label.
        // Live phase ("Loading 3D models…", "Setting up lighting…") shared with
        // the startup splash; shown as the modal's detail line.
        let phase = self.load_status.lock().unwrap().clone();
        egui::Area::new("zestengine-busy-card".into())
            .order(egui::Order::Tooltip)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.add(egui::Spinner::new());
                        ui.label(egui::RichText::new(label).strong());
                    });
                    let detail = if phase.is_empty() {
                        "The editor is busy — this can take a few seconds."
                    } else {
                        phase.as_str()
                    };
                    ui.label(egui::RichText::new(detail).small().weak());
                });
            });
        ctx.request_repaint();
    }

    /// While a bake runs, show a centered progress card (no full-screen dim, so
    /// the live lightmap build-up stays visible behind it). Each lightmap step is
    /// one long GPU submit that locks the frame; the card — painted before the
    /// first step and updated between steps — is the "something is happening".
    fn bake_modal(&self, ctx: &egui::Context) {
        let Some(runner) = self.bake_job.as_ref() else {
            return;
        };
        let (line, frac) = match runner.progress.lock().ok().as_deref() {
            Some(crate::tasks::TaskProgress::Bake {
                lightmap_done,
                lightmap_total,
                bounces,
                phase,
                ..
            }) => {
                let p = match phase {
                    crate::tasks::BakePhase::Lightmaps => "Tracing lightmaps",
                    crate::tasks::BakePhase::Probes => "Tracing probes",
                    crate::tasks::BakePhase::FluxVoxel => "Tracing FluxVoxel",
                };
                let total = (*lightmap_total).max(1);
                (
                    format!("{p} — {lightmap_done}/{total}  ·  {bounces} bounces"),
                    *lightmap_done as f32 / total as f32,
                )
            }
            _ => ("Building light data…".to_string(), 0.0),
        };
        egui::Area::new("zestengine-bake-card".into())
            .order(egui::Order::Tooltip)
            .anchor(egui::Align2::CENTER_TOP, egui::vec2(0.0, 24.0))
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.add(egui::Spinner::new());
                        ui.label(egui::RichText::new("Building light data").strong());
                    });
                    ui.add(
                        egui::ProgressBar::new(frac.clamp(0.0, 1.0))
                            .desired_width(260.0)
                            .text(line),
                    );
                    ui.label(
                        egui::RichText::new("The viewport may pause per lightmap.")
                            .small()
                            .weak(),
                    );
                });
            });
        ctx.request_repaint();
    }

    /// On window-focus regain: if any project asset changed in another app since
    /// the last check, reload the current scene (which reimports models/textures/
    /// materials/shaders fresh). Skips while playing or with unsaved scene edits
    /// (those would be clobbered), surfacing a hint instead.
    fn reload_changed_assets(&mut self) {
        use crate::tasks::{TaskKind, TaskPayload};
        zestengine_render::hitch::event("focus_gained", "", 0);
        let _h = zestengine_render::hitch::scope("reload_changed_assets");
        let now = std::time::SystemTime::now();
        // First focus just establishes a baseline (don't reload on startup).
        let Some(since) = self.last_asset_check.replace(now) else {
            return;
        };
        if self.playing || self.bake_job.is_some() {
            return;
        }
        if self.tasks.blocking().is_some() {
            return; // a re-import is already running
        }
        // The filesystem walk runs on a worker (it can be slow on a big project);
        // a blocking modal covers it, and the apply (GPU reload) runs on the main
        // thread when it returns.
        let root = self.project_root.clone();
        let scene_path = self.current_scene_path.clone();
        self.tasks.spawn(
            "Checking for changes…",
            TaskKind::FocusReimport,
            true,
            move |cancel, _progress| {
                if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                    return Ok(TaskPayload::None);
                }
                let mut changed = Vec::new();
                changed_assets(&root, since, &mut changed);
                if changed.is_empty() {
                    return Ok(TaskPayload::None);
                }
                let is_model = |p: &PathBuf| p.extension().and_then(|x| x.to_str()).is_some_and(|x| ["fbx", "gltf", "glb"].iter().any(|m| m.eq_ignore_ascii_case(x)));
                let label = changed[0].file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                let is_prefab = |p: &PathBuf| p.extension().and_then(|x| x.to_str()).is_some_and(|x| x.eq_ignore_ascii_case("prefab"));
                let other = changed.iter().any(|p| !is_model(p) && !is_prefab(p));
                let prefabs = changed.iter().filter(|p| is_prefab(p)).cloned().collect();
                let models = changed.into_iter().filter(is_model).collect();
                Ok(TaskPayload::FocusChanges { scene_path, label, models, prefabs, other })
            },
        );
    }

    /// Apply a captured input to the binding slot being rebound (2C), then
    /// persist bindings. Fed from the window-event handler (keys / mouse) and
    /// [`Self::poll_rebind_capture`] (gamepad / VR).
    fn apply_rebind(&mut self, result: zestengine_core::CaptureResult) {
        if self.bindings_editor.finish_rebind(&mut self.project.bindings, result) {
            self.bindings_changed();
        }
    }

    /// While a binding slot is armed, poll gamepads and VR controllers for it
    /// (keys and mouse buttons arrive as window events).
    fn poll_rebind_capture(&mut self) {
        if self.bindings_editor.rebinding().is_none() {
            return;
        }
        self.input.feed_vr(&zestengine_engine::runtime::vr_frame(self.xr_session.as_ref(), &self.vr_targets));
        let Some(capture) = self.bindings_editor.capture_mut() else {
            return;
        };
        let result = match self.input.poll_capture(capture) {
            Some(zestengine_core::CaptureResult::Captured(src)) if src.device() == zestengine_core::InputDevice::KeyboardMouse => None,
            r => r,
        };
        if let Some(r) = result {
            self.apply_rebind(r);
        }
    }

    /// Fingerprint of the plugin sources (mtime-based), or `None` when the
    /// project has no plugins. Compared at Play time: an external edit (nvim,
    /// another window) changes it, and Play rebuilds before running.
    fn plugins_fingerprint(&self) -> Option<u64> {
        let dirs = plugins::discover(&self.project_root);
        if dirs.is_empty() {
            return None;
        }
        let mut h: u64 = 0;
        for (dir, name) in dirs {
            h = h.wrapping_mul(0x100000001b3);
            h ^= plugins::build_fingerprint(&self.project_root, &dir);
            h ^= name.bytes().fold(0u64, |a, b| a.rotate_left(8) ^ b as u64);
        }
        Some(h)
    }

    /// Build + reload the component plugins. The cargo build runs on a worker
    /// (the UI stays alive, "Building components…" in the status bar, errors in
    /// the Log); [`Self::apply_plugins_built`] loads the result and re-instantiates
    /// scene components from their serialized state. A request while a build runs
    /// queues one rebuild behind it (never two cargo builds at once).
    fn request_plugin_build(&mut self) {
        if !self.plugin_gate.request() {
            self.set_status("Building components… (rebuild queued)", true);
            return;
        }
        self.start_plugin_build();
    }

    fn start_plugin_build(&mut self) {
        self.set_status("Building components…", true);
        self.plugin_build_started = Some(Instant::now());
        let root = self.project_root.clone();
        let status = self.plugin_status.clone();
        self.tasks.spawn("Building components…", crate::tasks::TaskKind::BuildComponents, false, move |_cancel, _progress| {
            let report = |t: &str| *status.lock().unwrap() = t.to_owned();
            let (built, diags) = plugins::PluginHost::build_all_with_diags(&root, &report);
            Ok(crate::tasks::TaskPayload::PluginsBuilt(built.map_err(|e| format!("{e:#}")), diags))
        });
    }

    /// Main-thread end of a plugin build: load + register, re-instantiate scene
    /// components, then run the work that waited (or start the queued rebuild).
    fn apply_plugins_built(&mut self, built: Result<Vec<String>, String>, diags: Vec<zestengine_plugins::CodeDiagnostic>) {
        // A queued rebuild supersedes this result: do not load the stale dylib.
        let result = if self.plugin_gate.rebuild_queued() {
            built
        } else {
            built.and_then(|names| {
                self.plugins
                    .load_built(&self.project_root, &names, &mut self.components)
                    .map_err(|e| format!("{e:#}"))?;
                let registrars = self.plugins.editor_registrars().to_vec();
                for p in registrars {
                    let f: fn(&mut crate::components::EditorComponents) = unsafe { std::mem::transmute(p) };
                    f(&mut self.editor_components);
                }
                Ok(names)
            })
        };
        match self.plugin_gate.finish(result.is_ok()) {
            crate::plugin_gate::Finished::Restart => {
                tracing::info!("component plugin sources changed during the build, rebuilding");
                self.start_plugin_build();
            }
            crate::plugin_gate::Finished::Done { ok: _, waiters } => {
                match result {
                    Ok(names) => {
                        self.plugin_build_error = None;
                        for object in &mut self.scene.objects {
                            let saved = object.save_components();
                            object.load_components(&saved, &self.components);
                        }
                        tracing::info!("reloaded plugins: {names:?} in {:?}", self.plugin_build_started.map(|t| t.elapsed()).unwrap_or_default());
                        self.set_status(format!("Components reloaded ({})", names.len()), false);
                        self.plugin_fingerprint = self.plugins_fingerprint();
                    }
                    Err(e) => {
                        tracing::error!("reloading plugins: {e}");
                        self.plugin_build_error = Some(e);
                        self.set_status("Component build failed", false);
                        // Show the build's compiler errors on the code
                        // inspectors and the file browser, like the LSP does.
                        for d in &diags {
                            let path = self.project_root.join(&d.file);
                            let errors = (d.level == "error") as u32;
                            let warns = (d.level != "error") as u32;
                            self.file_diagnostics
                                .entry(path.clone())
                                .and_modify(|(e, w)| {
                                    *e += errors;
                                    *w += warns;
                                })
                                .or_insert((errors, warns));
                            if let Some(editor) = self.open_editors.iter_mut().find(|e| e.path == path)
                            {
                                editor.diagnostics.push(crate::CodeDiagnostic {
                                    level: d.level.clone(),
                                    file: d.file.clone(),
                                    line: d.line,
                                    message: d.message.clone(),
                                });
                            }
                        }
                    }
                }
                self.plugin_build_started = None;
                for w in waiters {
                    match w {
                        PluginWaiter::Scene(data) => self.apply_loaded_scene(*data),
                        PluginWaiter::Play => {
                            if !self.playing {
                                self.toggle_play();
                            }
                        }
                    }
                }
            }
        }
    }

    /// Bottom status bar for the whole editor: project + object count on the
    /// left, current activity (rust-analyzer analyzing, compiling, last result)
    /// on the right.
    /// Apply a finished worker task on the main thread (GPU upload etc.).
    /// Import/texture/focus arms are filled in by their phases; cancelled and
    /// failed tasks just surface a notification.
    fn apply_task_result(&mut self, r: crate::tasks::TaskResult) {
        use crate::tasks::{NotifyLevel, TaskPayload};
        let _h = zestengine_render::hitch::scope("apply_task_result");
        if r.cancelled {
            self.tasks.notify("Task cancelled", NotifyLevel::Info);
            return;
        }
        let payload = match r.outcome {
            Ok(p) => p,
            Err(e) => {
                self.tasks.notify(format!("Task failed: {e}"), NotifyLevel::Warn);
                return;
            }
        };
        match payload {
            TaskPayload::None => {}
            TaskPayload::Notice(text) => self.tasks.notify(text, NotifyLevel::Info),
            TaskPayload::Model { source, scene } => {
                self.apply_imported_model(source, *scene);
            }
            TaskPayload::ModelReimport { source, scene } => self.apply_model_reimport(source, *scene),
            TaskPayload::Scene(data) => {
                // Components of a plugin that is still compiling would be dropped.
                if let Some(PluginWaiter::Scene(data)) = self.plugin_gate.wait_or_run(PluginWaiter::Scene(data)) {
                    self.apply_loaded_scene(*data);
                }
            }
            TaskPayload::PluginsBuilt(built, diags) => self.apply_plugins_built(built, diags),
            TaskPayload::AudioBake { scene_path, bytes } => self.apply_audio_bake(scene_path, bytes),
            TaskPayload::Occlusion { scene_path, data, secs } => self.apply_occlusion_bake(scene_path, *data, secs),
            TaskPayload::BakeSaved(saved) => self.apply_bake_saved(*saved),
            TaskPayload::GameBuilt(result) => {
                self.building_game = false;
                match result {
                    Ok(exe) => {
                        let msg = format!("built {}", exe.display());
                        tracing::info!("{msg}");
                        self.build_status = Some(msg.clone());
                        self.tasks.notify(msg, NotifyLevel::Info);
                    }
                    Err(e) => {
                        tracing::error!("build game failed: {e}");
                        self.build_status = Some(format!("build failed: {e}"));
                        self.tasks.notify("Game build failed (see Log)", NotifyLevel::Warn);
                    }
                }
            }
            TaskPayload::FocusChanges { scene_path, label, models, prefabs, other } => {
                use crate::tasks::NotifyLevel;
                // A changed model that is in the open scene reimports in place;
                // the rest of the scene (and any unsaved edit) stays as it is.
                let mut reimporting = 0;
                for abs in models {
                    let in_scene = abs
                        .strip_prefix(&self.project_root)
                        .is_ok_and(|rel| self.scene.has_model_template(&rel.to_string_lossy().replace('\\', "/")));
                    if in_scene {
                        self.spawn_model_reimport(abs);
                        reimporting += 1;
                    }
                }
                if reimporting > 0 {
                    self.tasks.notify(format!("Reimporting {reimporting} changed model(s)"), NotifyLevel::Info);
                }
                // An edited `.prefab` re-merges into the instances in place (their
                // overrides and the rest of the scene stay as they are).
                let rels: Vec<String> =
                    prefabs.iter().filter_map(|abs| abs.strip_prefix(&self.project_root).ok()).map(|rel| rel.to_string_lossy().replace('\\', "/")).collect();
                let merged = rels.iter().filter(|rel| self.remerge_prefab(rel)).count();
                if merged > 0 {
                    self.tasks.notify(format!("Updated instances of {merged} changed prefab(s)"), NotifyLevel::Info);
                }
                // Anything else that changed (textures, shaders, scenes, ...) still reloads the scene.
                if other {
                    if self.scene_dirty {
                        self.tasks.notify(format!("'{label}' changed on disk — reload to apply"), NotifyLevel::Warn);
                    } else if let Some(path) = scene_path {
                        zestengine_render::hitch::event("focus_reload_scene", "", 0);
                        self.load_scene_runtime(&path);
                        self.tasks.notify(format!("Reimported after '{label}' changed"), NotifyLevel::Info);
                    }
                }
            }
            TaskPayload::ModelRemap { path, material_remap, texture_remap, note, reasons } => {
                use crate::tasks::NotifyLevel;
                self.model_ui.set_match_reasons(&path, reasons);
                if let Some(fm) = self.file_meta.as_mut().filter(|f| f.path == path) {
                    if let Some(m) = fm.meta.importer.as_model_mut() {
                        m.material_remap = material_remap;
                        m.texture_remap = texture_remap;
                    }
                    fm.refresh_dirty();
                }
                self.tasks.notify(note, NotifyLevel::Info);
            }
        }
    }

    /// Swap a reimported model into the live scene in place (see
    /// `LoadedScene::reimport_model`). `source` is the project-relative path.
    fn apply_model_reimport(&mut self, source: PathBuf, asset: zestengine_assets::Scene) {
        use crate::tasks::NotifyLevel;
        zestengine_render::hitch::event("model_reimport_apply", "", 0);
        let _h = zestengine_render::hitch::scope("apply_model_reimport");
        // A bake holds mesh device addresses in its TLAS; replacing meshes would invalidate it.
        if self.bake_job.is_some() {
            self.tasks.notify("Reimport after the bake finishes", NotifyLevel::Warn);
            return;
        }
        let key = source.to_string_lossy().into_owned();
        let name = source.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let t0 = Instant::now();
        let result = self.scene.reimport_model(self.renderer.as_mut(), &mut self.shaders, &self.project_root, &key, &asset);
        match result {
            Ok(Some(r)) => {
                self.slot_validator.invalidate();
                self.rt_gi.invalidate();
                let mut note = if r.added_meshes + r.missing_meshes > 0 { format!(" ({} mesh(es) added, {} gone", r.added_meshes, r.missing_meshes) } else { String::new() };
                if r.orphaned_objects > 0 {
                    note += &format!("; {} object(s) still show a removed mesh", r.orphaned_objects);
                }
                if !note.is_empty() {
                    note += ")";
                }
                // The import recorded new ids and the material list in the `.meta`: an
                // open, unedited session reads them back instead of holding stale ones.
                let abs = self.project_root.join(&source);
                if let Some(fm) = self.file_meta.as_mut().filter(|f| f.path == abs && !f.dirty) {
                    fm.revert();
                }
                self.model_ui.forget();
                tracing::info!("reimported {key} in place: {r:?} in {:.1} ms", t0.elapsed().as_secs_f64() * 1000.0);
                self.set_status(format!("Reimported {name}: {} meshes, {} materials{note}", r.meshes, r.materials), false);
            }
            Ok(None) => self.set_status(format!("{name} is not in the open scene"), false),
            Err(e) => {
                tracing::error!("reimporting {key}: {e:#}");
                self.tasks.notify(format!("Reimport failed: {e}"), NotifyLevel::Warn);
            }
        }
    }

    /// Main-thread GPU apply for a worker-parsed model: upload meshes/textures
    /// into the scene. `source` is the project-relative path.
    fn apply_imported_model(&mut self, source: PathBuf, scene: zestengine_assets::Scene) {
        use crate::tasks::NotifyLevel;
        // A bake embeds mesh device addresses in its TLAS, so uploading new meshes
        // mid-bake would invalidate it — defer the upload until the bake ends.
        if self.bake_job.is_some() {
            self.tasks
                .notify("Import queued (baking)", NotifyLevel::Info);
            self.deferred_model_applies.push((source, scene));
            return;
        }
        let hook = self.placement_before_import(&source);
        if matches!(hook, placement_glue::ImportHook::Discard) {
            return;
        }
        let result = match self.renderer.as_mut() {
            Some(renderer) => self.scene.add_asset_scene(renderer, &scene, Some(source.as_path())),
            None => return,
        };
        let imported = result.is_ok();
        match result {
            Ok(()) => {
                self.scene_dirty = true;
                self.tasks
                    .notify(format!("Imported {}", source.display()), NotifyLevel::Info);
            }
            Err(e) => {
                tracing::error!("importing model: {e:#}");
                self.tasks
                    .notify(format!("Import failed: {e}"), NotifyLevel::Warn);
            }
        }
        self.placement_after_import(hook, imported);
    }

    /// The status-bar background-tasks popup: per-task progress bar + cancel.
    fn task_popup(&mut self, ctx: &egui::Context) {
        if !self.show_task_popup {
            return;
        }
        let mut cancel: Option<crate::tasks::TaskId> = None;
        egui::Window::new("Background tasks")
            .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-8.0, -32.0))
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                ui.set_min_width(320.0);
                let mut any = false;
                for h in self.tasks.visible() {
                    any = true;
                    let (frac, detail) = {
                        let p = h.progress.lock().unwrap();
                        (p.fraction(), p.detail())
                    };
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new(&h.label).strong());
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.small_button("Cancel").clicked() {
                                cancel = Some(h.id);
                            }
                        });
                    });
                    match frac {
                        Some(f) => {
                            ui.add(egui::ProgressBar::new(f).desired_height(8.0));
                        }
                        None => {
                            ui.add(egui::ProgressBar::new(0.0).desired_height(8.0).animate(true));
                        }
                    }
                    ui.label(egui::RichText::new(detail).size(12.0).weak());
                    ui.add_space(4.0);
                }
                if !any {
                    ui.label(egui::RichText::new("No background tasks").weak());
                }
            });
        if let Some(id) = cancel {
            self.tasks.cancel(id);
        }
    }

    fn status_bar(&mut self, ctx: &egui::Context) {
        let lsp_busy = !self.lsp_requests.is_empty();
        let recent = self
            .status
            .as_ref()
            .filter(|(_, t, _)| t.elapsed().as_secs_f32() < 6.0)
            .map(|(m, _, s)| (m.clone(), *s));
        let project = self.project.name.clone();
        let objects = self.scene.objects.len();
        // Full name + project-relative path of the selected file, so the user
        // can read the whole filename the grid tiles crop with an ellipsis.
        let selected_file = if let Selection::File(path) = &self.selection {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            let rel = path
                .strip_prefix(&self.project_root)
                .unwrap_or(path)
                .display()
                .to_string();
            Some((name, rel))
        } else {
            None
        };
        use crate::theme;
        let scene = self
            .current_scene_path
            .as_ref()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "untitled".into());
        let selected_object = match self.selection {
            Selection::Object(i) => self.scene.objects.get(i).map(|o| o.name.clone()),
            _ => None,
        };
        let meta = |t: &str| egui::RichText::new(t).size(theme::SIZE_META).color(theme::TEXT_MUTED);
        egui::TopBottomPanel::bottom("zestengine-status-bar")
            .exact_height(theme::H_STATUS_BAR)
            .frame(egui::Frame::new().fill(theme::BG_CHROME).inner_margin(egui::Margin::symmetric(12, 0)))
            .show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                ui.spacing_mut().item_spacing.x = 14.0;
                let (dot, state) = if self.playing {
                    (theme::accent(), if self.play_paused { "Paused" } else { "Playing" })
                } else {
                    (theme::STATUS_OK, "Ready")
                };
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 6.0;
                    let (r, _) = ui.allocate_exact_size(egui::vec2(7.0, 7.0), egui::Sense::hover());
                    ui.painter().circle_filled(r.center(), 3.5, dot);
                    ui.label(egui::RichText::new(state).size(theme::SIZE_META).color(if self.playing { dot } else { theme::TEXT_SECONDARY }));
                });
                ui.label(meta(&scene)).on_hover_text(&project);
                ui.label(meta(&format!("{objects} objects")));
                if let Some(name) = selected_object {
                    ui.label(meta(&format!("Selected: {name}")));
                } else if let Some((name, rel)) = selected_file {
                    ui.label(meta(&format!("Selected: {name}"))).on_hover_text(rel);
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        egui::RichText::new(format!("{:.0} fps · {:.2} ms", self.stats.fps, self.stats.frame_ms))
                            .font(theme::mono(theme::SIZE_META))
                            .color(theme::TEXT_MUTED),
                    );
                    ui.label(meta(&build_stamp())).on_hover_text("This build: version, profile and link time of the running binary");
                    if lsp_busy {
                        ui.add(egui::Spinner::new().size(12.0));
                        ui.label(meta("rust-analyzer"));
                    } else if let Some((msg, spinner)) = recent {
                        if spinner {
                            ui.add(egui::Spinner::new().size(12.0));
                        }
                        ui.label(egui::RichText::new(msg).size(theme::SIZE_META).color(theme::TEXT_SECONDARY));
                    }
                    // Background tasks + aggregate progress.
                    if let Some((n, frac)) = self.tasks.aggregate() {
                        if ui
                            .add(egui::Button::new(egui::RichText::new(format!("{} Tasks ({n})", ph::GEAR)).size(theme::SIZE_META)).frame(false))
                            .clicked()
                        {
                            self.show_task_popup = !self.show_task_popup;
                        }
                        ui.add(egui::ProgressBar::new(frac).desired_width(90.0).desired_height(6.0).fill(theme::accent()));
                    } else {
                        self.show_task_popup = false;
                    }
                    // Most recent task notification (auto-expires).
                    if let Some(note) = self.tasks.notifications.last() {
                        let col = match note.level {
                            crate::tasks::NotifyLevel::Warn => theme::STATUS_WARN_FG,
                            crate::tasks::NotifyLevel::Info => theme::TEXT_SECONDARY,
                        };
                        ui.label(egui::RichText::new(&note.text).size(theme::SIZE_META).color(col));
                    }
                });
            });
        });
    }

    fn refresh_shaders(&mut self) {
        let _h = zestengine_render::hitch::scope("refresh_shaders");
        // The project walk runs on a worker thread; pick up its result when it lands.
        if let Some(rx) = &self.shader_scan_rx {
            match rx.try_recv() {
                Ok(files) => {
                    self.shader_files = files;
                    self.shader_scan_rx = None;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => self.shader_scan_rx = None,
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
        if self
            .last_shader_scan
            .is_some_and(|t| t.elapsed().as_secs_f32() < 2.0)
        {
            return;
        }
        self.last_shader_scan = Some(Instant::now());
        if self.shader_scan_rx.is_none() {
            let root = self.project_root.clone();
            let (tx, rx) = std::sync::mpsc::channel();
            self.shader_scan_rx = Some(rx);
            std::thread::spawn(move || {
                let exts = ["frag", zestengine_assets::HLSL_SHADER_EXTENSION, zestengine_assets::SHADER_GRAPH_EXTENSION];
                let _ = tx.send(scan_project_exts(&root, &exts));
            });
        }
        if let Some(renderer) = self.renderer.as_mut() {
            let changed = self.shaders.poll_reload(renderer, &self.project_root);
            if !changed.is_empty() {
                self.scene.reapply_materials_using(
                    renderer,
                    &mut self.shaders,
                    &self.project_root,
                    &changed,
                );
                self.set_status(format!("Reloaded {} shader(s)", changed.len()), false);
            }
        }

        // Engine built-in shaders (e.g. standard.frag) are baked into the binary
        // at build time, so watch their source on disk and recompile + hot-swap
        // the pipeline when they change outside the editor.
        let frag = std::path::Path::new(zestengine_render::ENGINE_SHADER_DIR).join("standard.frag");
        let mtime = std::fs::metadata(&frag).and_then(|m| m.modified()).ok();
        if mtime != self.engine_shader_mtime {
            let first = self.engine_shader_mtime.is_none();
            self.engine_shader_mtime = mtime;
            if !first {
                // Drop the renderer borrow before set_status (which needs &mut self).
                if let Some(result) = self.renderer.as_mut().map(|r| r.reload_standard_shaders()) {
                    match result {
                        Ok(()) => {
                            self.set_status("Reloaded engine shader (standard.frag)", false)
                        }
                        Err(e) => self.set_status(format!("standard.frag reload failed: {e}"), true),
                    }
                }
            }
        }
    }

    /// Reflected shader info for the selection's material, resolved before
    /// the UI runs (compiles the shader on first use). Also initializes the
    /// model's custom values from the shader defaults.
    fn selected_shader_info(&mut self) -> Option<ShaderUiInfo> {
        let shader = match &self.selection {
            Selection::Object(i) => {
                let object = self.scene.objects.get(*i)?;
                let material = object.render.map(|r| r.material)?;
                self.scene.materials[material].model.shader.clone()
            }
            Selection::File(_) => self.file_material.as_ref()?.model.shader.clone(),
            Selection::None => return None,
        };
        if shader == "standard" {
            return None;
        }
        let renderer = self.renderer.as_mut()?;
        let entry = self.shaders.resolve(renderer, &self.project_root, &shader);
        let ui = entry.ui.clone();
        let has_source = entry.source.is_some();
        if has_source {
            let model = match &self.selection {
                Selection::Object(i) => {
                    let material = self.scene.objects[*i].render.map(|r| r.material)?;
                    &mut self.scene.materials[material].model
                }
                Selection::File(_) => &mut self.file_material.as_mut()?.model,
                Selection::None => return None,
            };
            entry.fit_values(&mut model.custom_values);
            entry.sync_textures(&mut model.textures);
        }
        Some(ui)
    }

    /// Reflected shader info for the open material sections of the selected object, resolved
    /// before the UI runs (compiles a custom shader on first use).
    fn object_shader_infos(&mut self) -> std::collections::HashMap<usize, ShaderUiInfo> {
        let mut out = std::collections::HashMap::new();
        let Selection::Object(i) = self.selection else { return out };
        let Some(o) = self.scene.objects.get(i) else { return out };
        let key = o.id.to_string();
        let slots: Vec<usize> = o.render_slots().map(|r| r.material).collect();
        for (m, group) in crate::group_slots(&slots) {
            if !crate::section_open(&self.egui_ctx, &key, group[0]) {
                continue;
            }
            let shader = self.scene.materials[m].model.shader.clone();
            if shader == "standard" {
                continue;
            }
            let Some(renderer) = self.renderer.as_mut() else { break };
            let entry = self.shaders.resolve(renderer, &self.project_root, &shader);
            let info = entry.ui.clone();
            if entry.source.is_some() {
                entry.fit_values(&mut self.scene.materials[m].model.custom_values);
                entry.sync_textures(&mut self.scene.materials[m].model.textures);
            }
            out.insert(m, info);
        }
        out
    }

    /// Load a .material file into the file inspector (the pinned inspector
    /// content for files). Clearing first lets non-material files show the
    /// empty state.
    fn open_material_inspector(&mut self, path: &std::path::Path) {
        load_material_file_into(path, &mut self.file_material);
    }

    /// Snapshot the selection's editable state before the UI runs, so edits
    /// can be diffed into undo entries afterwards.
    fn capture_edit_snapshot(&self) -> EditSnapshot {
        // Honour the inspector lock: when locked, the inspector edits the LOCKED
        // object, not `self.selection`. The snapshot must track the same target or
        // edits to a locked object diff against the wrong one and never record.
        let effective = if self.inspector.locked {
            self.inspector_lock_target
                .clone()
                .unwrap_or_else(|| self.selection.clone())
        } else {
            self.selection.clone()
        };
        match &effective {
            Selection::Object(i) if *i < self.scene.objects.len() => {
                let o = &self.scene.objects[*i];
                // Capture the materials of the open material sections: any of them may be
                // edited this frame.
                let key = o.id.to_string();
                let slots: Vec<usize> = o.render_slots().map(|r| r.material).collect();
                let materials = crate::group_slots(&slots)
                    .into_iter()
                    .filter(|(_, g)| crate::section_open(&self.egui_ctx, &key, g[0]))
                    .map(|(m, _)| (m, Box::new(self.scene.materials[m].model.clone())))
                    .collect();
                EditSnapshot::Object { index: *i, state: object_state(o), materials }
            }
            Selection::File(path) => match &self.file_material {
                Some(fm) if fm.path == *path => EditSnapshot::File {
                    path: path.clone(),
                    model: Box::new(fm.model.clone()),
                },
                _ => EditSnapshot::None,
            },
            _ => EditSnapshot::None,
        }
    }

    /// Diff the post-frame state against the snapshot and record undo
    /// entries. Continuous gestures coalesce inside the stack.
    /// Apply the anchor object's edit to the other multi-selected objects.
    /// Transforms propagate as a DELTA (translate/rotate/scale together, matching
    /// the gizmo); shared components are set to the anchor's new value wholesale
    /// (correct when the value was shared, which the shared-inspector enforces).
    fn propagate_multi_edit(
        &mut self,
        anchor: usize,
        before: &ObjectState,
        after: &ObjectState,
    ) -> Vec<UndoEntry> {
        let dt = after.translation - before.translation;
        let dr = after.rotation * before.rotation.inverse();
        let inv = |v: f32| if v.abs() > 1e-6 { v } else { 1.0 };
        let ds = glam::Vec3::new(
            after.scale.x / inv(before.scale.x),
            after.scale.y / inv(before.scale.y),
            after.scale.z / inv(before.scale.z),
        );
        // Components whose RON changed on the anchor this edit.
        let changed: Vec<(String, String)> = after
            .components
            .iter()
            .filter(|c| before.components.iter().find(|b| b.kind == c.kind).map(|b| &b.data) != Some(&c.data))
            .map(|c| (c.kind.clone(), c.data.clone()))
            .collect();
        // Components switched on or off on the anchor this edit.
        let toggled: Vec<(String, bool)> = after
            .components
            .iter()
            .filter(|c| before.components.iter().find(|b| b.kind == c.kind).is_some_and(|b| b.enabled != c.enabled))
            .map(|c| (c.kind.clone(), c.enabled))
            .collect();
        let targets: Vec<usize> = self
            .multi_objects
            .iter()
            .copied()
            .filter(|&j| j != anchor && j < self.scene.objects.len())
            .collect();
        let typed = std::mem::take(&mut self.multi_edit);
        let absolute = typed.any();
        let xform_changed = !absolute && (dt != glam::Vec3::ZERO || ds != glam::Vec3::ONE || before.rotation != after.rotation);
        // Per component, just the fields this edit changed.
        let field_edits: Vec<FieldEdit> = changed
            .iter()
            .map(|(name, ron)| {
                let before_ron = before.components.iter().find(|c| &c.kind == name).map(|c| c.data.as_str()).unwrap_or("");
                (name.clone(), crate::ron_fields::changed(before_ron, ron))
            })
            .collect();
        let mut entries = Vec::new();
        for j in targets {
            // Snapshot before mutating so the propagated edit is itself undoable
            // (otherwise undo would revert only the anchor, leaving the others).
            let t_before = object_state(&self.scene.objects[j]);
            if xform_changed {
                let o = &mut self.scene.objects[j];
                o.translation += dt;
                o.rotation = (dr * o.rotation).normalize();
                o.scale *= ds;
            }
            if absolute {
                typed.apply(&mut self.scene.objects[j]);
            }
            for (name, on) in &toggled {
                if let Some(slot) = self.scene.objects[j].components.iter_mut().find(|c| c.type_name() == name.as_str()) {
                    zestengine_core::set_component_enabled(slot, *on);
                }
            }
            if !changed.is_empty() {
                let mut comps = self.scene.objects[j].save_components();
                let mut touched = false;
                for ((name, ron), (_, fields)) in changed.iter().zip(&field_edits) {
                    if let Some(slot) = comps.iter_mut().find(|c| &c.kind == name) {
                        // Only the fields that changed, so the others keep their
                        // own values; whole-component copy when the RON can't be split.
                        slot.data = fields.as_ref().and_then(|f| crate::ron_fields::patch(&slot.data, f)).unwrap_or_else(|| ron.clone());
                        touched = true;
                    }
                }
                if touched {
                    self.scene.objects[j].load_components(&comps, &self.components);
                    self.refresh_renderer_materials(j);
                }
            }
            let t_after = object_state(&self.scene.objects[j]);
            if t_after != t_before {
                // Returned to the caller to bundle into ONE Group entry with the
                // anchor, so the whole multi-move is a single undo step (and one
                // coalescing unit across the drag, not N interleaved entries).
                entries.push(UndoEntry::Object {
                    index: j,
                    before: t_before,
                    after: t_after,
                });
            }
        }
        entries
    }

    /// Re-resolve an object's materials when its renderer component changed behind the render
    /// cache's back (undo, multi-select propagation, prefab revert). No-op without a renderer.
    fn refresh_renderer_materials(&mut self, index: usize) {
        if let Some(renderer) = self.renderer.as_mut()
            && self.scene.refresh_renderer_materials(renderer, &mut self.shaders, &self.project_root, index)
        {
            self.rt_gi.invalidate();
        }
    }

    fn record_edits(&mut self, pre: EditSnapshot, gesture_active: bool) {
        if std::mem::take(&mut self.suppress_undo_record) {
            return;
        }
        match pre {
            EditSnapshot::Object {
                index,
                state,
                materials,
            } if index < self.scene.objects.len() => {
                let o = &self.scene.objects[index];
                let now = object_state(o);
                if now != state {
                    self.scene_dirty = true;
                    let anchor = UndoEntry::Object {
                        index,
                        before: state.clone(),
                        after: now.clone(),
                    };
                    // Multi-select: mirror the anchor's edit to the other selected
                    // objects (transforms as a delta so they move/rotate/scale
                    // together; shared components set to the anchor's new value), and
                    // bundle anchor + others into ONE atomic Group so a multi-move is
                    // a single undo step that coalesces cleanly across the drag.
                    if self.multi_objects.len() > 1 && self.multi_objects.contains(&index) {
                        let mut group = vec![anchor];
                        group.extend(self.propagate_multi_edit(index, &state, &now));
                        let entry = if group.len() > 1 {
                            UndoEntry::Group(group)
                        } else {
                            group.pop().unwrap()
                        };
                        self.undo_stack.record(entry, gesture_active);
                    } else {
                        self.undo_stack.record(anchor, gesture_active);
                    }
                }
                for (material, model) in materials {
                    if material >= self.scene.materials.len() {
                        continue;
                    }
                    let current = &self.scene.materials[material].model;
                    if *current != *model {
                        self.scene_dirty = true;
                        self.dirty_materials.insert(material);
                        self.last_material_edit = Some(Instant::now());
                        self.undo_stack.record(
                            UndoEntry::Material { index: material, before: model, after: Box::new(current.clone()) },
                            gesture_active,
                        );
                    }
                }
            }
            EditSnapshot::File { path, model } => {
                if let Some(fm) = &self.file_material
                    && fm.path == path
                    && fm.model != *model
                {
                    self.last_material_edit = Some(Instant::now());
                    self.undo_stack.record(
                        UndoEntry::FileMaterial {
                            path,
                            before: model,
                            after: Box::new(fm.model.clone()),
                        },
                        gesture_active,
                    );
                }
            }
            _ => {}
        }
    }

    /// Auto-save edited materials once the edit gesture settles. Materials
    /// without a backing file get one created under `materials/`.
    fn autosave_materials(&mut self) {
        // Code editors: save each 1s after its last keystroke (saving .frag
        // files also triggers shader hot reload, for live shader editing).
        let to_save: Vec<PathBuf> = self
            .open_editors
            .iter()
            .filter(|e| {
                e.dirty
                    && e.last_edit
                        .is_some_and(|t| t.elapsed().as_secs_f32() >= 1.0)
            })
            .map(|e| e.path.clone())
            .collect();
        for path in to_save {
            self.actions.push(EditorAction::SaveOpenEditor(path));
        }
        // External edits (nvim): a clean editor whose file changed on disk
        // reloads the new content. A dirty editor keeps its buffer (the user's
        // unsaved work wins); the next save overwrites the external change.
        for e in self.open_editors.iter_mut() {
            let Ok(md) = std::fs::metadata(&e.path) else { continue };
            let Ok(mtime) = md.modified() else { continue };
            if e.disk_mtime != Some(mtime) {
                if e.dirty {
                    tracing::warn!("{} changed on disk with unsaved edits", e.path.display());
                } else {
                    if let Ok(text) = std::fs::read_to_string(&e.path) {
                        e.text = text;
                        e.diagnostics.clear(); // the LSP re-publishes
                    }
                    e.disk_mtime = Some(mtime);
                }
            }
        }
        if self
            .last_material_edit
            .is_none_or(|t| t.elapsed().as_secs_f32() < 0.8)
        {
            return;
        }
        self.last_material_edit = None;
        // A `.material` file open in the Inspector saves through the same
        // path as the manual 💾 button.
        if self.file_material.as_ref().is_some_and(|fm| fm.dirty) {
            self.actions.push(EditorAction::SaveFileMaterial);
        }
        for index in std::mem::take(&mut self.dirty_materials) {
            if index < self.scene.materials.len() {
                self.save_scene_material(index);
            }
        }
    }

    /// Persist one scene material to its `.material` file, **only if it already
    /// has one**. Embedded (imported) materials have no backing file and are
    /// read-only until the user extracts them ([`extract_material`]); we never
    /// auto-create a phantom `.material` for them.
    fn save_scene_material(&mut self, index: usize) {
        let Some(path) = self.scene.materials[index].file.clone() else {
            return;
        };
        let model = self.scene.materials[index].model.clone();
        let mut file = zestengine_assets::load_material_file(&path).unwrap_or_else(|_| {
            zestengine_assets::MaterialFile {
                name: model.name.clone(),
                shader: "standard".into(),
                params: Default::default(),
                features: Default::default(),
                render_queue: None,
                textures: Default::default(),
                custom: Default::default(),
            }
        });
        let (params, features) = material_from_model(&model);
        file.name = model.name.clone();
        file.shader = model.shader.clone();
        file.params = params;
        file.features = features;
        file.render_queue = Some(model.render_queue);
        let ignored = std::mem::take(&mut file.textures.ignored_warnings);
        file.textures = zestengine_engine::scene::tex_file_from_paths(&model.textures);
        file.textures.ignored_warnings = ignored;
        file.custom = if model.shader != "standard" {
            self.shaders
                .get(&model.shader)
                .and_then(|e| e.source.as_ref())
                .map(|s| s.unpack(&model.custom_values))
                .unwrap_or(file.custom)
        } else {
            Default::default()
        };
        match zestengine_assets::save_material_file(&path, &file) {
            Ok(()) => tracing::info!("auto-saved material to {}", path.display()),
            Err(e) => tracing::error!("auto-saving material: {e:#}"),
        }
    }

    /// A `.material` file describing `model` (textures as project-relative paths, custom shader
    /// values unpacked by name).
    fn material_file_from_model(&self, model: &MaterialModel) -> zestengine_assets::MaterialFile {
        let (params, features) = material_from_model(model);
        let mut file = zestengine_assets::MaterialFile {
            name: model.name.clone(),
            shader: model.shader.clone(),
            params,
            features,
            render_queue: Some(model.render_queue),
            textures: zestengine_engine::scene::tex_file_from_paths(&model.textures),
            custom: Default::default(),
        };
        if model.shader != "standard" {
            file.custom = self.shaders.get(&model.shader).and_then(|e| e.source.as_ref()).map(|s| s.unpack(&model.custom_values)).unwrap_or_default();
        }
        file
    }

    /// Extract an embedded (imported) material to a real `.material` file under
    /// `materials/`, assign it back to the material slot, and persist it. After
    /// that it's a normal editable asset. No-op if it already has a file.
    fn extract_material(&mut self, index: usize) {
        if index >= self.scene.materials.len() || self.scene.materials[index].file.is_some() {
            return;
        }
        let name = self.scene.materials[index].model.name.clone();
        let stem: String = name
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' })
            .collect();
        let stem = stem.trim_matches('_');
        let dir = self.project_root.join("materials");
        let path = unique_path(&dir, if stem.is_empty() { "material" } else { stem }, "material");
        self.scene.set_material_file(index, path.clone());
        self.save_scene_material(index);
        self.set_status(
            format!("Extracted material '{}' -> {}", name, relative_to(&path, &self.project_root)),
            false,
        );
    }

    /// Rebuild just one model on a worker (forced past the import cache), then
    /// swap its meshes and materials in place on the main thread: no scene reload,
    /// so unsaved edits and play mode are untouched.
    fn spawn_model_reimport(&mut self, path: PathBuf) {
        let Ok(rel) = path.strip_prefix(&self.project_root) else { return };
        let source = PathBuf::from(rel.to_string_lossy().replace('\\', "/"));
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        self.tasks.spawn(
            format!("Reimport {name}"),
            crate::tasks::TaskKind::ImportModel,
            false,
            move |cancel, _progress| {
                if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                    return Ok(crate::tasks::TaskPayload::None);
                }
                let (scene, _) = zestengine_assets::load_model_cached(&path, true).map_err(|e| format!("{e:#}"))?;
                if let Err(e) = zestengine_assets::record_model_ids(&path, &scene) {
                    tracing::warn!("recording model ids: {e:#}");
                }
                Ok(crate::tasks::TaskPayload::ModelReimport { source, scene: Box::new(scene) })
            },
        );
    }

    /// Extract Textures / Extract Materials / Search and Remap for the model open
    /// in the inspector. Runs on a worker against a copy of the edited settings;
    /// the filled tables come back as `TaskPayload::ModelRemap`.
    fn start_model_remap(&mut self, path: PathBuf, job: crate::model_import_editor::RemapJob) {
        use crate::model_import_editor::RemapJob;
        use zestengine_assets::material_remap as remap;
        let Some(mut model) = self.file_meta.as_ref().filter(|f| f.path == path).and_then(|f| f.meta.importer.as_model().cloned()) else { return };
        let root = self.project_root.clone();
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let label = match job {
            RemapJob::ExtractTextures => "Extracting textures from",
            RemapJob::ExtractMaterials => "Extracting materials from",
            RemapJob::Search => "Searching materials for",
        };
        self.tasks.spawn(format!("{label} {name}"), crate::tasks::TaskKind::ImportModel, false, move |cancel, _progress| {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                return Ok(crate::tasks::TaskPayload::None);
            }
            let fail = |e: anyhow::Error| format!("{e:#}");
            let rel = |p: &Path| remap::project_relative(&root, p);
            // The chosen folders (empty = beside the model), validated to stay in the project.
            let tex_dir = || remap::resolve_extract_dir(&root, &path, &model.extract_textures_dir, "Textures").map_err(|e| format!("{e:#}"));
            let mat_dir = || remap::resolve_extract_dir(&root, &path, &model.extract_materials_dir, "Materials").map_err(|e| format!("{e:#}"));
            let mut reasons = Vec::new();
            let note = match job {
                RemapJob::Search => {
                    // The texture-name rule needs each material's texture names, which only the parsed model has.
                    let sources: Vec<remap::SourceMaterial> = if model.material_naming == zestengine_assets::MaterialNaming::BasedOnTextureName || model.source_materials.is_empty() {
                        let (scene, _) = zestengine_assets::load_model_cached(&path, false).map_err(fail)?;
                        remap::source_materials_of(&scene)
                    } else {
                        model.source_materials.iter().map(|n| remap::SourceMaterial { name: n.clone(), textures: Vec::new() }).collect()
                    };
                    // Only rows that are still empty: a choice the user made stays.
                    let open: Vec<remap::SourceMaterial> = sources.into_iter().filter(|n| !model.material_remap.contains_key(&n.name)).collect();
                    let found = remap::search_matches(&root, &path, &open, model.material_naming, model.material_search);
                    for h in &found {
                        model.material_remap.insert(h.source.clone(), h.path.clone());
                        reasons.push((h.source.clone(), h.path.clone(), h.reason.clone()));
                    }
                    format!("Remapped {} of {} material(s) by {}", found.len(), open.len(), if model.material_naming == zestengine_assets::MaterialNaming::BasedOnTextureName { "texture name" } else { "name" })
                }
                RemapJob::ExtractTextures => {
                    let (scene, _) = zestengine_assets::load_model_cached(&path, false).map_err(fail)?;
                    let tex_dir = tex_dir()?;
                    let r = remap::extract_textures(&scene, &tex_dir, &root, &mut model).map_err(fail)?;
                    format!("Extracted {} texture(s) to {} ({} import setting file(s))", r.textures, rel(&tex_dir), r.metas)
                }
                RemapJob::ExtractMaterials => {
                    let (scene, _) = zestengine_assets::load_model_cached(&path, false).map_err(fail)?;
                    let (mat_dir, tex_dir) = (mat_dir()?, tex_dir()?);
                    // A VRM's MToon materials import as ZestEngine Toon (outline, rim, matcap, shade
                    // colour) before the plain extract, which then skips what is already mapped.
                    let opts = zestengine_assets::unity_convert::ConvertOptions { texture_dir: rel(&tex_dir), ..Default::default() };
                    let toon = zestengine_assets::mtoon::import_mtoon_materials(&path, &root, &mat_dir, &opts, &mut model).map_err(fail)?;
                    let r = remap::extract_materials(&scene, &mat_dir, &tex_dir, &root, &mut model).map_err(fail)?;
                    let mtoon_note = if toon.converted.is_empty() { String::new() } else { format!(" including {} MToon material(s) as ZestEngine Toon", toon.converted.len()) };
                    format!("Extracted {} material(s){mtoon_note} to {} ({} texture(s) to {})", r.materials + toon.converted.len(), rel(&mat_dir), r.textures, rel(&tex_dir))
                }
            };
            Ok(crate::tasks::TaskPayload::ModelRemap { path, material_remap: model.material_remap, texture_remap: model.texture_remap, note, reasons })
        });
    }

    /// Record a structural edit (the object list before it, and now) as one
    /// undo step, and keep its index shifts out of per-frame edit diffing.
    fn record_structure(&mut self, before: scene::ObjectsSnapshot) {
        let after = self.scene.snapshot_objects();
        self.undo_stack.record(UndoEntry::Structure { before: Box::new(before), after: Box::new(after) }, false);
        self.suppress_undo_record = true;
    }

    /// `H` / `Shift+H` / `Alt+H`: hide the selection in the scene view / show all.
    /// Only with the Scene view or Hierarchy focused, outside text editing and Play.
    fn scene_view_key(&mut self) {
        let focus_ok = self
            .dock_state
            .find_active_focused()
            .is_some_and(|(_, tab)| matches!(tab, Tab::Viewport | Tab::Scene));
        let text_editing = self.egui_ctx.wants_keyboard_input();
        let (shift, alt, ctrl) = (self.modifiers.shift_key(), self.modifiers.alt_key(), self.modifiers.control_key());
        match crate::view_key_action(true, shift, alt, ctrl, text_editing, self.playing, focus_ok) {
            Some(crate::ViewKeyAction::ShowAll) => self.actions.push(EditorAction::SceneView(crate::ViewRequest::ShowAll)),
            Some(crate::ViewKeyAction::ToggleHide) => self.toggle_hide_selection(),
            None => {}
        }
    }

    /// Hide the selected objects in the scene view; all hidden already: show them.
    fn toggle_hide_selection(&mut self) {
        let n = self.scene.objects.len();
        let targets: Vec<usize> = if self.multi_objects.len() > 1 {
            self.multi_objects.iter().copied().filter(|&i| i < n).collect()
        } else if let Selection::Object(i) = self.selection {
            vec![i].into_iter().filter(|&i| i < n).collect()
        } else {
            Vec::new()
        };
        if targets.is_empty() {
            return;
        }
        let all_hidden = targets.iter().all(|&i| self.scene.scene_view.hidden.contains(&self.scene.objects[i].id));
        self.actions.push(EditorAction::SceneView(crate::ViewRequest::Hide(targets, !all_hidden)));
    }

    /// Hierarchy eye / lock / `H`: change the scene-view hide / lock sets as
    /// one undo step. Never touches `enabled`.
    fn apply_view_request(&mut self, req: crate::ViewRequest) {
        use crate::ViewRequest as V;
        let before = self.scene.scene_view.clone();
        let n = self.scene.objects.len();
        match req {
            V::Hide(targets, on) => {
                for i in targets.into_iter().filter(|&i| i < n) {
                    self.scene.set_view_hidden(i, on);
                }
            }
            V::Lock(targets, on) => {
                for i in targets.into_iter().filter(|&i| i < n) {
                    self.scene.set_view_locked(i, on);
                }
            }
            V::Isolate(i) if i < n => {
                // Keep the row, its ancestors and its descendants; hide the rest.
                // A second Alt-click on the same isolated row shows all again.
                let keep = self.scene.isolate_keep(i);
                let already = (0..n).all(|j| self.scene.scene_view.hidden.contains(&self.scene.objects[j].id) == !keep[j]);
                self.scene.scene_view.hidden.clear();
                if !already {
                    for j in (0..n).filter(|&j| !keep[j]) {
                        self.scene.set_view_hidden(j, true);
                    }
                }
            }
            V::Isolate(_) => {}
            V::ShowAll => self.scene.scene_view.hidden.clear(),
            V::UnlockAll => self.scene.scene_view.locked.clear(),
        }
        let after = self.scene.scene_view.clone();
        if before != after {
            self.undo_stack.record(UndoEntry::View { before: Box::new(before), after: Box::new(after) }, false);
            self.suppress_undo_record = true;
            self.scene_dirty = true;
        }
    }

    /// Apply one history entry (true = undo, false = redo).
    fn apply_history(&mut self, undo: bool) {
        // The animation tabs keep their own document histories.
        match self.dock_state.find_active_focused().map(|(_, tab)| tab.clone()) {
            Some(Tab::AnimClip) => return self.anim_clip_tab.history_step(undo),
            Some(Tab::Sequencer) => return self.sequencer_tab.history_step(undo),
            Some(Tab::AnimController) => return self.anim_ctrl_tab.history_step(undo),
            Some(Tab::ShaderGraph) => return,
            _ => {}
        }
        let Some(entry) = (if undo {
            self.undo_stack.pop_undo()
        } else {
            self.undo_stack.pop_redo()
        }) else {
            return;
        };
        self.suppress_undo_record = true;
        self.apply_entry(entry, undo);
    }

    /// Apply a single history entry (recursing into `Group`). `apply_history` owns
    /// the pop + `suppress_undo_record`; this just mutates the scene.
    fn apply_entry(&mut self, entry: UndoEntry, undo: bool) {
        match entry {
            UndoEntry::Environment { before, after } => {
                let mut env = *if undo { before } else { after };
                // Skybox image sources are tracked by their own `Skybox` entries:
                // keep the ones currently on screen so an environment undo never
                // reverts them.
                env.skybox_faces = self.scene.environment.skybox_faces.clone();
                self.scene.environment = env;
                self.scene_dirty = true;
            }
            UndoEntry::Skybox { before, after } => {
                let state = if undo { before } else { after };
                self.scene.skybox = state.equirect.clone();
                self.scene.environment.skybox_faces = state.faces.clone();
                self.apply_skybox();
                self.scene_dirty = true;
            }
            UndoEntry::Structure { before, after } => {
                let snapshot = if undo { before } else { after };
                self.scene.restore_objects(&snapshot, &self.components);
                self.prefab_editor.invalidate();
                // Indices differ between the two lists: start from no selection.
                self.selection = Selection::None;
                self.multi_objects.clear();
                self.inspector_lock_target = None;
                self.inspector.locked = false;
                self.scene_dirty = true;
                self.scene.refresh_stage_hidden();
            }
            UndoEntry::Prefab { files, before, after } => {
                let snapshot = if undo { before } else { after };
                if let Err(e) = self.scene.restore_prefab_edit(&files, &snapshot, undo, &self.components) {
                    tracing::error!("prefab undo: {e:#}");
                }
                self.scene.refresh_stage_hidden();
                self.prefab_editor.invalidate();
                self.selection = Selection::None;
                self.multi_objects.clear();
                self.inspector_lock_target = None;
                self.inspector.locked = false;
                self.scene_dirty = true;
            }
            UndoEntry::View { before, after } => {
                let keep = self.scene.scene_view.edit;
                self.scene.scene_view = *if undo { before } else { after };
                self.scene.scene_view.edit = keep;
                self.scene_dirty = true;
            }
            UndoEntry::Enabled(items) => {
                for (id, was, now) in items {
                    if let Some(i) = self.scene.index_of(id) {
                        self.scene.objects[i].enabled = if undo { was } else { now };
                    }
                }
                self.scene_dirty = true;
            }
            UndoEntry::Tiles { object, edits } => {
                if let Some(i) = self.scene.index_of(object)
                    && let Some(m) = self.scene.objects[i].components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<zestengine_core::Tilemap>())
                {
                    zestengine_core::apply_edits(m, &edits, !undo);
                    self.scene_dirty = true;
                }
            }
            UndoEntry::Group(entries) => {
                // Undo reverts in reverse application order; redo replays forward.
                if undo {
                    for e in entries.into_iter().rev() {
                        self.apply_entry(e, undo);
                    }
                } else {
                    for e in entries {
                        self.apply_entry(e, undo);
                    }
                }
            }
            UndoEntry::Object {
                index,
                before,
                after,
            } => {
                let state = if undo { before } else { after };
                if let Some(o) = self.scene.objects.get_mut(index) {
                    o.name = state.name;
                    o.translation = state.translation;
                    o.rotation = state.rotation;
                    o.scale = state.scale;
                    o.net_role = state.net_role;
                    o.component_roles = state.component_roles;
                    o.load_components(&state.components, &self.components);
                }
                // The restored renderer may name other materials.
                self.refresh_renderer_materials(index);
            }
            UndoEntry::Material {
                index,
                before,
                after,
            } => {
                let model = if undo { before } else { after };
                if index < self.scene.materials.len() {
                    self.scene.materials[index].model = *model;
                    self.dirty_materials.insert(index);
                    self.last_material_edit = Some(Instant::now());
                    if let Some(renderer) = self.renderer.as_mut() {
                        self.scene.apply_material(
                            renderer,
                            &mut self.shaders,
                            &self.project_root,
                            index,
                        );
                    }
                    self.rt_gi.invalidate();
                }
            }
            UndoEntry::Assign {
                object,
                slot,
                before,
                after,
                before_slot,
                after_slot,
            } => {
                let (material, value) = if undo { (before, before_slot) } else { (after, after_slot) };
                if object < self.scene.objects.len() && material < self.scene.materials.len() {
                    self.scene.restore_renderer_slot(object, slot, material, value);
                }
                self.rt_gi.invalidate();
            }
            UndoEntry::FileMaterial {
                path,
                before,
                after,
            } => {
                let model = if undo { before } else { after };
                if let Some(fm) = &mut self.file_material
                    && fm.path == path
                {
                    fm.model = *model;
                    fm.dirty = true;
                }
            }
        }
    }

    /// Open the profiler as a separate OS window (movable to another monitor).
    /// Needs the event loop to create the winit window, so it's driven from
    /// `redraw` via `update_profiler_window`.
    fn open_profiler_window(&mut self, event_loop: &ActiveEventLoop) {
        if self.profiler_window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("ZestEngine Profiler")
            .with_inner_size(winit::dpi::LogicalSize::new(1120.0, 840.0));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                tracing::error!("creating profiler window: {e:#}");
                self.show_stats_overlay = false;
                return;
            }
        };
        let (Ok(disp), Ok(wh)) = (window.display_handle(), window.window_handle()) else {
            tracing::error!("profiler window exposed no raw handles");
            self.show_stats_overlay = false;
            return;
        };
        let (disp, wh) = (disp.as_raw(), wh.as_raw());
        let size = window.inner_size();
        match self.renderer.as_mut() {
            Some(r) => {
                if let Err(e) = r.open_profiler_window(disp, wh, size.width, size.height) {
                    tracing::error!("renderer profiler window: {e:#}");
                    self.show_stats_overlay = false;
                    return;
                }
            }
            None => return,
        }
        // Fresh egui context per open: the renderer's egui texture store was
        // destroyed on close, so a reused context (which thinks the font atlas
        // Managed(0) is already uploaded and won't resend it) would draw against
        // a now-empty store -> "Bad texture ID". A new context re-uploads all
        // textures on its first frame.
        self.profiler_egui_ctx = egui::Context::default();
        crate::install_icon_font(&self.profiler_egui_ctx);
        apply_editor_style(&self.profiler_egui_ctx);
        self.profiler_egui_state = Some(egui_winit::State::new(
            self.profiler_egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            None,
            None,
            None,
        ));
        window.request_redraw();
        self.profiler_window = Some(window);
    }

    /// Close the profiler window: the renderer tears down its surface/swapchain
    /// (full device idle) BEFORE the winit window drops, as required.
    fn close_profiler_window(&mut self) {
        if let Some(r) = self.renderer.as_mut() {
            r.close_profiler_window();
        }
        self.profiler_egui_state = None;
        self.profiler_window = None;
    }

    /// Fold this frame's metrics into the current time bucket; flush a sample to
    /// every series once `BUCKET` seconds have elapsed (keeps the 20 s window
    /// stable at any frame rate). Values array order MUST match `ProfHistory`.
    fn record_prof_sample(&mut self, dt: f32) {
        let s = self.last_render_stats;
        let t = &self.frame_timings;
        let util = if self.stats.frame_ms > 0.0 {
            (t.gpu_frame / self.stats.frame_ms * 100.0).clamp(0.0, 100.0)
        } else {
            0.0
        };
        let vals = [
            self.stats.frame_ms,
            self.stats.fps,
            t.gpu_frame,
            t.gpu_gi,
            s.gpu_shadows_ms,
            s.gpu_scene_ms,
            s.gpu_reflect_ms,
            s.gpu_post_ms,
            s.gpu_egui_ms,
            s.gpu_cam_preview_ms,
            util,
            t.gi,
            t.components,
            t.physics,
            t.audio,
            t.draws,
            t.render,
            s.draw_calls as f32,
            s.gpu_camera_gi_ms,
            s.gpu_camera_gi_gather_ms,
            s.gpu_camera_gi_denoise_ms,
            s.gpu_camera_gi_integrate_ms,
            s.gpu_camera_gi_surface_ms,
            s.gpu_gi_surface_ms,
        ];
        let h = &mut self.prof_history;
        // Spikes matter, so each bucket keeps the max seen across its frames.
        for (m, &v) in h.metrics.iter_mut().zip(vals.iter()) {
            m.pending = m.pending.max(v);
        }
        h.bucket_t += dt;
        if h.bucket_t >= Series::BUCKET {
            h.bucket_t = 0.0;
            for (m, &v) in h.metrics.iter_mut().zip(vals.iter()) {
                m.series.push(m.pending);
                m.pending = v; // next bucket starts from the current frame
            }
        }
    }

    /// Reconcile the desired profiler state with the actual window, then run +
    /// render the profiler egui UI into it. Called once per frame from `redraw`.
    fn update_profiler_window(&mut self, event_loop: &ActiveEventLoop) {
        if self.show_stats_overlay && self.profiler_window.is_none() {
            self.open_profiler_window(event_loop);
        } else if !self.show_stats_overlay && self.profiler_window.is_some() {
            self.close_profiler_window();
        }
        let Some(window) = self.profiler_window.clone() else {
            return;
        };
        // The profiler is a throttled secondary window: skip the whole egui
        // frame (not just the render) when it isn't due, so no texture delta is lost.
        if self.renderer.as_ref().is_some_and(|r| !r.profiler_due()) {
            return;
        }
        let raw_input = match self.profiler_egui_state.as_mut() {
            Some(state) => state.take_egui_input(&window),
            None => return,
        };
        let ctx = self.profiler_egui_ctx.clone();
        let output = ctx.run(raw_input, |ctx| self.profiler_ui(ctx));
        if let Some(state) = self.profiler_egui_state.as_mut() {
            state.handle_platform_output(&window, output.platform_output);
        }
        let primitives = ctx.tessellate(output.shapes, output.pixels_per_point);
        let draw = zestengine_render::EguiDraw {
            pixels_per_point: output.pixels_per_point,
            primitives,
            textures_delta: output.textures_delta,
        };
        if let Some(r) = self.renderer.as_mut()
            && let Err(e) = r.render_profiler(&draw) {
                tracing::error!("profiler render: {e:#}");
            }
        window.request_redraw();
    }

    /// The profiler window's egui contents: one combined realtime graph, a
    /// clickable legend (toggles each metric in the graph) with current values,
    /// and the non-time-series draw/pipeline counts.
    fn profiler_ui(&mut self, ctx: &egui::Context) {
        let s = self.last_render_stats;
        let frame_ms = self.stats.frame_ms;
        let fps = self.stats.fps;
        let mut toggle: Option<usize> = None;
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading(format!("{frame_ms:.1} ms   {fps:.0} fps"));
                let fl = &self.stats.log;
                ui.label(
                    egui::RichText::new(format!(
                        "worst {:.1} ms (last 1 s) · {:.1} ms (session)",
                        fl.last_worst_ms, fl.session_worst_ms
                    ))
                    .size(11.0)
                    .color(if fl.last_worst_ms > 16.7 {
                        egui::Color32::from_rgb(230, 170, 70)
                    } else {
                        egui::Color32::from_gray(150)
                    }),
                );
                ui.label(
                    egui::RichText::new(format!("work (no vblank wait):{}", fl.last_work.fields()))
                        .size(11.0)
                        .color(egui::Color32::from_gray(130)),
                )
                .on_hover_text("CPU time excluding the time blocked in acquire/present, and the last completed GPU frame; well under the frame budget with 16.7 ms frames means missed vblanks");
                match zestengine_render::hitch::last_hitch() {
                    Some(h) => {
                        let amber = egui::Color32::from_rgb(230, 170, 70);
                        let i = &h.interval;
                        ui.label(
                            egui::RichText::new(format!(
                                "last hitch {:.1} ms · cpu {:.1} · acquire {:.1} · present {:.1}{}",
                                h.total_ms,
                                i.cpu_ms,
                                i.acquire_ms,
                                i.present_ms,
                                i.gpu_ms.map_or(String::new(), |g| format!(" · gpu {g:.1}"))
                            ))
                            .size(11.0)
                            .color(amber),
                        )
                        .on_hover_text("Frames over ZESTENGINE_HITCH_MS (default 50; 10 catches missed vblanks at 120 Hz). Also logged under target frame_log::hitch");
                        for (name, ms) in &h.top {
                            ui.label(egui::RichText::new(format!("    {ms:>7.1} ms  {name}")).size(11.0).monospace().color(egui::Color32::from_gray(150)));
                        }
                        for (n, c) in &h.counters {
                            ui.label(egui::RichText::new(format!("    {n} x{c}")).size(11.0).color(egui::Color32::from_gray(130)));
                        }
                        for (ago, e) in &h.events {
                            ui.label(egui::RichText::new(format!("    -{ago:.2}s  {e}")).size(11.0).color(egui::Color32::from_gray(130)));
                        }
                    }
                    None => {
                        ui.label(egui::RichText::new("no hitch captured yet").size(11.0).color(egui::Color32::from_gray(110)));
                    }
                }
                let hist: String = zestengine_engine::runtime::FRAME_HIST_LABELS
                    .iter()
                    .zip(fl.last_hist)
                    .filter(|(_, n)| *n > 0)
                    .map(|(l, n)| format!("{l} ms: {n}   "))
                    .collect();
                ui.label(egui::RichText::new(hist).size(11.0).color(egui::Color32::from_gray(130)))
                    .on_hover_text("Frame-to-frame times over the last second, bucketed on 120 Hz vblank multiples");
                ui.label(
                    egui::RichText::new("20 s history · shared axis · hover a line for details")
                        .size(11.0)
                        .color(egui::Color32::from_gray(130)),
                );
                ui.add_space(2.0);

                draw_combined_graph(ui, &self.prof_history);
                ui.add_space(4.0);

                ui.label(egui::RichText::new("Click a row to show/hide it").strong());
                for (i, m) in self.prof_history.metrics.iter().enumerate() {
                    if legend_row(ui, m).clicked() {
                        toggle = Some(i);
                    }
                }

                ui.add_space(6.0);
                ui.label(egui::RichText::new("Draw / pipeline").strong());
                count_row(ui, "Draw calls", s.draw_calls);
                count_row(ui, "  opaque", s.opaque_draws);
                count_row(ui, "  transparent", s.transparent_draws);
                count_row(ui, "  outline", s.outline_draws);
                if s.error_draws > 0 {
                    count_row(ui, "  error", s.error_draws);
                }
                count_row(ui, "Materials drawn", s.materials_drawn);
                count_row(ui, "Pipeline binds", s.pipeline_binds);
                count_row(ui, "Shader variants", s.pipeline_variants);
                if s.gpu_skin_ms > 0.0 || s.gpu_morph_ms > 0.0 {
                    ui.label(format!("GPU skinning   {:.3} ms skin, {:.3} ms morph", s.gpu_skin_ms, s.gpu_morph_ms));
                }
                if s.gpu_culling {
                    ui.add_space(6.0);
                    ui.label(
                        egui::RichText::new(format!(
                            "GPU culling   {:.2} ms cull, {:.2} ms Hi-Z",
                            s.gpu_cull_ms, s.gpu_hiz_ms
                        ))
                        .strong(),
                    );
                    count_row(ui, "Indirect draws", s.indirect_draws);
                    count_row(ui, "Instances tested", s.gpu_cull_tested);
                    count_row(ui, "  visible", s.gpu_cull_visible);
                    if s.gpu_cull_recovered > 0 {
                        count_row(ui, "    recovered (late Hi-Z)", s.gpu_cull_recovered);
                    }
                    count_row(ui, "  frustum / size culled", s.gpu_cull_frustum);
                    count_row(ui, "  occluded (Hi-Z)", s.gpu_cull_occluded);
                    count_row(ui, "Shadow instances", s.gpu_cull_shadow_visible);
                    if s.gpu_cull_eye_visible > 0 {
                        count_row(ui, "VR eye instances", s.gpu_cull_eye_visible);
                    }
                }
                if s.cluster.enabled {
                    let c = s.cluster;
                    ui.add_space(6.0);
                    ui.label(
                        egui::RichText::new(format!(
                            "Zestlets{}   {:.2} ms cull, {:.2} ms draw",
                            if c.paused { " (paused: overflow)" } else { "" },
                            c.cull_ms,
                            c.draw_ms
                        ))
                        .strong(),
                    );
                    for line in c.coverage.panel_lines() {
                        ui.label(line);
                    }
                    count_row(ui, "Zestlet meshes", c.meshes);
                    count_row(ui, "Material batches", c.batches);
                    count_row(ui, "Instances", c.instances);
                    count_row(ui, "BVH nodes visited", c.nodes_visited);
                    count_row(ui, "Zestlets tested", c.clusters_tested);
                    count_row(ui, "  not in the LOD cut", c.culled_lod);
                    count_row(ui, "  frustum culled", c.culled_frustum);
                    count_row(ui, "  cone culled", c.culled_cone);
                    count_row(ui, "  Hi-Z culled", c.culled_hiz);
                    count_row(ui, "  visible", c.visible_clusters);
                    count_row(ui, "Triangles submitted", c.triangles);
                    count_row(ui, "Queue overflows", c.queue_overflows);
                    ui.label(format!(
                        "LOD error scale x{:.2} (target {}), estimated {} / {} triangles, {} instances over their share (up to x{:.1})",
                        c.error_scale,
                        if c.target_ms > 0.0 { format!("{:.1} ms", c.target_ms) } else { "off".to_owned() },
                        c.budget_estimated,
                        c.budget_limit,
                        c.fair_clamped,
                        c.max_instance_scale
                    ));
                    if c.stream_on {
                        ui.label(format!(
                            "Geometry streaming: {} pages resident ({:.1} of {:.0} MB), {} loading, {} waiting",
                            c.stream_pages, c.stream_resident_mb, c.stream_budget_mb, c.stream_loading, c.stream_pending
                        ));
                        count_row(ui, "  page requests", c.stream_requests.min(u32::MAX as u64) as u32);
                        count_row(ui, "  page loads", c.stream_loads.min(u32::MAX as u64) as u32);
                        count_row(ui, "  evictions", c.stream_evictions.min(u32::MAX as u64) as u32);
                        count_row(ui, "  stall frames (budget full)", c.stream_stall_frames.min(u32::MAX as u64) as u32);
                        count_row(ui, "  thrash reloads", c.stream_thrash.min(u32::MAX as u64) as u32);
                        count_row(ui, "  zestlets on missing pages", c.stream_skipped);
                        count_row(ui, "  coarse stand-ins", c.stream_standins);
                        ui.label(format!("  {:.1} MB streamed in total", c.stream_mb_streamed));
                    }
                    if c.skinned_clusters > 0 || c.skin_pool_overflows > 0 {
                        count_row(ui, "Skinned zestlets posed", c.skinned_clusters);
                        count_row(ui, "  skinned vertices", c.skin_vertices);
                        count_row(ui, "  skin pool overflows", c.skin_pool_overflows);
                    }
                }
            });
        });
        if let Some(i) = toggle
            && let Some(m) = self.prof_history.metrics.get_mut(i)
        {
            m.visible = !m.visible;
        }
    }

    /// Refresh the frame-limiter policy and suspend it while the window is
    /// minimised or a blocking job's busy overlay is showing.
    fn frame_limit_prepare(&mut self) {
        let (Some(w), Some(r)) = (&self.window, &self.renderer) else { return };
        self.pacing.update(w, r.main_present_mode());
        let hidden = w.is_minimized().unwrap_or(false);
        let busy = self.tasks.blocking().is_some() || self.pending_job.is_some() || self.reload_pending;
        self.pacing.set_suspended(hidden || busy);
    }

    fn redraw(&mut self, event_loop: &ActiveEventLoop) {
        let _hitch = zestengine_render::hitch::frame_begin();
        zestengine_render::hitch::phase("input_events");
        // Frame limiter (MAILBOX / IMMEDIATE only): normally `about_to_wait`
        // already slept in the event loop and this only spins the last ~1 ms.
        // Before the swapchain acquire and excluded from cpu_ms (own hitch scope).
        self.frame_limit_prepare();
        self.pacing.wait();
        let now = Instant::now();
        let raw_dt = (now - self.last_frame).as_secs_f32();
        let dt = raw_dt.min(0.1);
        self.last_frame = now;
        self.stats.tick(dt);
        if let Some(iv) = zestengine_render::hitch::last_interval() {
            self.stats.log.record_work(iv.cpu_ms, iv.gpu_ms);
        }
        zestengine_render::hitch::set_gpu_ms(Some(self.last_render_stats.gpu_frame_ms));
        self.stats.log.record(raw_dt * 1000.0);
        zestengine_render::hitch::phase("tasks_apply");
        self.step_post_load();
        // Background tasks: collect finished worker results, apply their GPU side
        // on this (main) thread, and expire old notifications.
        let results = self.tasks.poll_workers();
        for r in results {
            self.apply_task_result(r);
        }
        // Advance the chunked bake one unit this frame (GPU, main thread). The
        // first frame after starting is a warm-up: paint the bake modal once
        // before the first (UI-locking) lightmap step, so there's an indicator.
        if self.bake_pending_warmup {
            self.bake_pending_warmup = false;
            if let Some(w) = &self.window {
                w.request_redraw();
            }
        } else {
            self.step_bake();
        }
        self.tasks.prune_notifications();
        zestengine_render::hitch::phase("camera_shaders");
        self.update_camera(dt);
        self.refresh_shaders();
        zestengine_render::hitch::phase("realtime_gi");
        let gi_t = Instant::now();
        self.update_realtime_gi(dt);
        let gi_ms = gi_t.elapsed().as_secs_f32() * 1000.0;
        FrameTimings::ema(&mut self.frame_timings.gi, gi_ms);

        // If mouse-look just ended, egui's pointer state is stale: window
        // events are withheld while looking, so egui (a) never saw the right
        // button RELEASE, so it still thinks Secondary is held, which keeps
        // `press_origin` pinned and breaks drag detection for the next gesture
        // (a click still registers, but orbit/gizmo/widget drags don't), and
        // (b) has a frozen pointer position. Inject both a Secondary release
        // and a move at the true cursor so the next press starts a clean drag.
        let resync_pos = if std::mem::take(&mut self.look_just_ended) {
            let ppp = self.egui_ctx.pixels_per_point();
            self.last_cursor
                .map(|(x, y)| egui::pos2(x as f32 / ppp, y as f32 / ppp))
        } else {
            None
        };

        let Some(window) = self.window.clone() else {
            return;
        };
        let Some(egui_state) = self.egui_state.as_mut() else {
            return;
        };

        let mut raw_input = egui_state.take_egui_input(&window);
        if let Some(pos) = resync_pos {
            raw_input.events.push(egui::Event::PointerMoved(pos));
            raw_input.events.push(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Secondary,
                pressed: false,
                modifiers: egui::Modifiers::default(),
            });
        }
        // VR pointer: feed the right-controller-on-panel position into egui as the
        // mouse, so the whole desktop UI is clickable from inside the headset.
        if let Some(pos) = self.vr_ui_pointer {
            raw_input.events.push(egui::Event::PointerMoved(pos));
            if self.vr_ui_pressed != self.vr_ui_prev_pressed {
                raw_input.events.push(egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: self.vr_ui_pressed,
                    modifiers: egui::Modifiers::default(),
                });
            }
        } else if self.vr_ui_prev_pressed {
            // Pointer left the panel while pressed: release so egui doesn't stick.
            raw_input.events.push(egui::Event::PointerButton {
                pos: egui::Pos2::ZERO,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::default(),
            });
        }
        self.vr_ui_prev_pressed = self.vr_ui_pressed;
        let size = window.inner_size();
        // The editor viewport renders to an offscreen texture sized to the dock
        // rect (RTT), so the camera aspect comes from that rect, not the window.
        let ppp = self.egui_ctx.pixels_per_point();
        let vrect = self.viewport_rect;
        let (vw, vh) = (
            (vrect.width() * ppp).round().max(1.0) as u32,
            (vrect.height() * ppp).round().max(1.0) as u32,
        );
        let aspect = if vrect.is_finite() && vrect.height() > 1.0 {
            vrect.width() / vrect.height()
        } else {
            size.width.max(1) as f32 / size.height.max(1) as f32
        };
        if let Some(r) = self.renderer.as_mut() {
            let vpx = [vrect.min.x, vrect.min.y, vrect.max.x, vrect.max.y].map(|v| if v.is_finite() { (v * ppp).round().max(0.0) as u32 } else { 0 });
            // The probe's `action-*` modes queue scroll / orbit gestures that `viewport_ui`
            // applies inside the egui pass through the same helpers the wheel and a left
            // drag use. `action-dolly-stale` instead returns a wheel amount to push as an
            // action here, reproducing the old post-pass dolly for an A/B comparison.
            if let Some(amount) = crate::sky_probe::frame(&mut self.camera, r, vpx) {
                self.actions.push(EditorAction::Dolly(amount));
            }
            // `ZESTENGINE_SKY_PROBE_SELECT=<object name>`: select it so the transform
            // gizmo (and the selection overlays) are part of the composed frames.
            if let Some(name) = crate::sky_probe::take_selection_request()
                && let Some(i) = self.scene.objects.iter().position(|o| o.name == name)
            {
                self.selection = Selection::Object(i);
            }
        }
        let game_view = self.game_view(aspect);
        let (view, proj) = match game_view {
            Some((v, p, _)) => (v, p),
            None => (self.camera.view(), self.camera.proj(aspect)),
        };
        // The Scene view shows the game camera (play + "Game camera"): the fly
        // camera is not on screen, so `viewport_ui` must not refresh the overlay
        // matrices from it and the wheel goes to the game.
        let scene_shows_game = game_view.is_some();
        // A held pointer button may be a dock-border drag: the renderer keeps the
        // old viewport target until the size settles or the button is released.
        let pointer_held = self.egui_ctx.input(|i| i.pointer.any_down());
        if let Some(r) = self.renderer.as_mut() {
            r.set_viewport_resize_hold(pointer_held);
        }
        let viewport_extent = if self.viewport_visible && vrect.is_finite() {
            Some([vw, vh])
        } else {
            None
        };

        let render_stats = self
            .renderer
            .as_ref()
            .map(|r| r.stats())
            .unwrap_or_default();
        // Only fold profiling state when the profiler window is open, so a
        // closed profiler adds zero per-frame work (the renderer likewise skips
        // its GPU timestamp queries when no profiler window exists).
        // The stats copy is unconditional: the always-on FRAME timestamp pair
        // feeds the hitch / frame log's gpu_ms even with the profiler closed
        // (it used to be gated here, which left gpu_ms at `na`).
        self.last_render_stats = render_stats;
        zestengine_engine::bench_avatars::tick(&render_stats);
        if self.profiler_window.is_some() {
            FrameTimings::ema(&mut self.frame_timings.gpu_frame, render_stats.gpu_frame_ms);
            FrameTimings::ema(&mut self.frame_timings.gpu_gi, render_stats.gpu_gi_ms);
            self.record_prof_sample(dt);
        }

        let pre_edit = self.capture_edit_snapshot();
        let selected_object = match self.selection {
            Selection::Object(i) => Some(i),
            _ => None,
        };
        crate::river_edit::sync(&self.egui_ctx, &self.scene, selected_object, self.playing);
        crate::cloth_paint::sync(&self.egui_ctx, &mut self.scene, selected_object, self.playing);
        crate::terrain_paint::sync(&self.egui_ctx, &self.scene, selected_object, self.playing);
        crate::tile_edit::sync(&self.egui_ctx, &self.scene, selected_object, self.playing, &self.project_root);
        crate::shape_edit::sync(&self.egui_ctx, &self.scene, selected_object, self.playing);
        let shader_info = self.selected_shader_info();
        let section_shader_infos = self.object_shader_infos();
        let can_bake = self.renderer.as_ref().is_some_and(|r| r.supports_baking());
        // Winit-based viewport-hover test: stays correct right after a
        // cursor-grab (mouse-look) ends, when egui's own pointer state is
        // briefly stale until the next move. Used to re-arm orbit/dolly.
        let pointer_in_viewport = self.cursor_in_viewport();
        // Picker entries: standard + every project .frag.
        let mut shader_list: Vec<String> = Vec::with_capacity(self.shader_files.len() + 1);
        shader_list.push("standard".into());
        shader_list.extend(self.shader_files.iter().cloned());

        self.poll_rebind_capture();
        let egui_ctx = self.egui_ctx.clone();
        zestengine_render::hitch::phase("egui_run");
        let output = egui_ctx.run(raw_input, |ctx| {
            self.placement_capture_input(ctx);
            self.menu_bar(ctx);
            self.vr_tools_window(ctx);
            self.vr_wheel_window(ctx);
            self.vr_watch_window(ctx);
            let rig = self.vr_rig;
            self.body.window(ctx, &mut self.scene, &rig);
            self.project_windows(ctx);
            crate::tile_edit::window(ctx);
            self.bindings_window(ctx);
            self.network_window(ctx);
            self.edit_window_ui(ctx);
            self.performance_window(ctx);
            self.layers_window(ctx);
            // An ObjectRef chip in an inspector was clicked: select that object.
            let select_key = egui::Id::new(crate::SELECT_OBJECT_KEY);
            if let Some(index) = ctx.data(|d| d.get_temp::<usize>(select_key)) {
                ctx.data_mut(|d| d.remove::<usize>(select_key));
                if index < self.scene.objects.len() {
                    self.select_object(Some(index));
                }
            }
            self.pose_editor_window(ctx);
            if self.content_builder.open {
                let selected = match self.selection {
                    Selection::Object(i) => Some(i),
                    _ => None,
                };
                let src = crate::content_build::BuildSource {
                    scene: &self.scene,
                    shaders: &self.shaders,
                    root: &self.project_root,
                    selected,
                    scene_path: self.current_scene_path.as_deref(),
                    registry: &self.components,
                };
                self.content_builder.show(ctx, &src);
            }
            self.lightmap_uv_window(ctx);
            self.pack_import.show(ctx, &mut self.tasks, &self.project_root);
            self.busy_overlay(ctx);
            self.bake_modal(ctx);
            self.status_bar(ctx);
            self.task_popup(ctx);
            self.prefab_tick(ctx);
            self.prefab_stage_bar(ctx);
            {
                let selected = match &self.selection {
                    Selection::Object(i) => Some(*i),
                    _ => None,
                };
                // Edits may have happened when the pointer is released or a key is typed.
                let hint = ctx.input(|i| {
                    i.pointer.any_released() || i.events.iter().any(|e| matches!(e, egui::Event::Text(_) | egui::Event::Key { .. } | egui::Event::Paste(_)))
                });
                self.prefab_editor.refresh(&self.scene, selected, &self.project_root, &self.shaders, hint);
            }
            // The render stats now live in a separate OS window (the profiler),
            // reconciled + rendered after this egui pass — no in-viewport panel.

            let camera_preview = self
                .renderer
                .as_ref()
                .and_then(|r| r.camera_preview_texture());
            self.rig_ui.preview.set_texture(self.renderer.as_ref().and_then(|r| r.model_preview_texture()));
            self.asset_preview.set_live_texture(self.renderer.as_ref().and_then(|r| r.model_preview_texture()));
            let viewport_texture = self
                .renderer
                .as_ref()
                .and_then(|r| r.viewport_texture());
            let camera_overlay = matches!(
                self.selection,
                Selection::Object(i) if self.scene.camera_view_proj_for(i, 1.0).is_some()
            );
            let mut dock_state = std::mem::replace(&mut self.dock_state, DockState::new(vec![]));
            // The focused code tab drives which file's diagnostics the
            // Inspector shows (kept out of the editor to stop layout shift).
            let focused_code = dock_state
                .find_active_focused()
                .and_then(|(_, tab)| match tab {
                    Tab::Code(p) => Some(p.clone()),
                    _ => None,
                });
            // egui_dock only calls TabViewer::ui for the visible (active) tab of
            // each leaf, so camera_ui sets this flag iff the Camera tab is on
            // screen this frame. Reset before the dock, read after to gate the
            // (now Flux-traced) preview render.
            self.camera_tab_visible = false;
            self.viewport_visible = false;
            // Models whose lightmap UVs were auto-generated (.lmuv marker), for the
            // FluxBaker "Generated Lightmap UVs" list. Computed before the mutable
            // scene borrow below.
            let generated_uv_models =
                self.scene.models_with_generated_lightmap_uv(&self.project_root);
            if self.bake_sizes.is_none() {
                self.bake_sizes =
                    Some(crate::bake_store::measure(&self.bake_base_path(), self.scene.baked.as_ref(), self.refl_slots.baked()));
            }
            let ui_overlay_edit = !self.playing && self.widget_filter.game_ui;
            let active_prev = std::mem::take(&mut self.active_tabs);
            let fit_labels = fit_tab_labels(ctx, &dock_state, self.vp_strip_width);
            self.multi_edit = MultiEdit::default();
            self.model_ui.geo.cluster_on = self.project.settings.cluster_rendering == zestengine_render::ClusterMode::On;
            let mut tabs = EditorTabs {
                fit_labels: &fit_labels,
                camera_visible: &mut self.camera_tab_visible,
                camera_rect: &mut self.camera_rect,
                viewport_visible: &mut self.viewport_visible,
                vp_strip_width: &mut self.vp_strip_width,
                scene: &mut self.scene,
                selection: &mut self.selection,
                multi_objects: &mut self.multi_objects,
                multi_edit: &mut self.multi_edit,
                multi_files: &mut self.multi_files,
                inspector: &mut self.inspector,
                inspector_lock_target: &mut self.inspector_lock_target,
                scene_panel: &mut self.scene_panel,
                prefab: &mut self.prefab_editor,
                file_browser: &mut self.file_browser,
                file_diagnostics: &self.file_diagnostics,
                editor_components: &self.editor_components,
                file_material: &mut self.file_material,
                file_meta: &mut self.file_meta,
                texture_ui: &mut self.texture_ui,
                rig_ui: &mut self.rig_ui,
                asset_preview: &mut self.asset_preview,
                sprite_atlas_ui: &mut self.sprite_atlas_ui,
                model_ui: &mut self.model_ui,
                slot_validator: &mut self.slot_validator,
                open_editors: &mut self.open_editors,
                nvim_embed: &mut self.nvim_embed,
                code_zoom: &mut self.editor_prefs.code_zoom,
                focused_code,
                gizmo: &mut self.gizmo,
                widget_filter: &mut self.widget_filter,
                edit_peers: &self.edit_peers,
                ui_editor: &mut self.ui_editor,
                anim_ctrl_tab: &mut self.anim_ctrl_tab,
                anim_clip_tab: &mut self.anim_clip_tab,
                sequencer_tab: &mut self.sequencer_tab,
                shader_graph_tab: &mut self.shader_graph_tab,
                occlusion_ui: &mut self.occlusion_ui,
                ui_rect_editor: &mut self.ui_rect_editor,
                active_tabs_prev: &active_prev,
                playing: self.playing,
                ugc_context: self.content_builder.open,
                viewport_stats: &mut self.viewport_stats,
                zestlet_coverage: self.last_render_stats.cluster.coverage,
                stats: (
                    self.stats.fps,
                    self.stats.frame_ms,
                    self.renderer.as_ref().map_or(0, |r| r.stats().draw_calls),
                ),
                active_tabs: Default::default(),
                ui_overlay_edit,
                project_root: &self.project_root,
                gi_debug: &mut self.gi_debug,
                gizmo_drag: &mut self.gizmo_drag,
                orbit_armed: &mut self.orbit_armed,
                // Nothing else reads the camera until the dock pass is done
                // (`placement_frame` below runs after `drop(tabs)`).
                camera: &mut self.camera,
                orbit_pivot: &mut self.orbit_pivot,
                scene_shows_game,
                pointer_in_viewport,
                actions: &mut self.actions,
                viewport_rect: &mut self.viewport_rect,
                registry: &self.components,
                shader_list: &shader_list,
                shader_info: shader_info.as_ref(),
                section_shader_infos: &section_shader_infos,
                camera_preview,
                viewport_texture,
                camera_overlay,
                view,
                proj,
                looking: self.looking,
                vim_mode: self.project.settings.vim_mode,
                log_filter: &mut self.log_filter,
                handle_drag: &mut self.handle_drag,
                can_bake,
                lightmap_preview: &mut self.lightmap_preview,
                generated_uv_models: &generated_uv_models,
                bake_sizes: self.bake_sizes.unwrap_or_default(),
                project: &mut self.project,
                bindings_editor: &mut self.bindings_editor,
                input: &mut self.input,
                build_status: &self.build_status,
                build_target: &mut self.build_target,
                building_game: &self.building_game,
                slimevr: &self.slimevr,
                face_input: &self.face_input,
                avatar_preview: &mut self.avatar_preview,
                project_settings_dirty: &mut self.project_settings_dirty,
            };
            DockArea::new(&mut dock_state)
                .style(crate::theme::dock_style(ctx.style().as_ref()))
                // Per-tab close X (only on closeable tabs = Code); no group
                // close-all button.
                .show_close_buttons(true)
                .show_leaf_close_all_buttons(false)
                // No per-leaf collapse button: it only hid the leaf's contents
                // without resizing the neighbouring docks, which was confusing.
                .show_leaf_collapse_buttons(false)
                .show(ctx, &mut tabs);
            let active_now = std::mem::take(&mut tabs.active_tabs);
            drop(tabs);
            self.active_tabs = active_now;
            clamp_leaf_width(&mut dock_state, &Tab::Inspector, INSPECTOR_MIN_WIDTH);
            self.dock_state = dock_state;
            if self.project_settings_dirty {
                self.project.apply_role_overrides(&mut self.components);
                self.save_project();
                self.project_settings_dirty = false;
            }
            if std::mem::take(&mut self.focus_game_tab) && self.playing {
                match self.dock_state.find_tab(&Tab::Camera) {
                    Some(loc) => self.dock_state.set_active_tab(loc),
                    None => {
                        if let Some((surface, node, _)) = self.dock_state.find_tab(&Tab::Viewport) {
                            self.dock_state.set_focused_node_and_surface((surface, node));
                        }
                        self.dock_state.push_to_focused_leaf(Tab::Camera);
                    }
                }
            }
            if std::mem::take(&mut self.focus_scene_tab) && !self.playing
                && let Some(loc) = self.dock_state.find_tab(&Tab::Viewport)
            {
                self.dock_state.set_active_tab(loc);
            }
            for controller in crate::anim_controller_editor::take_open_requests(
                ctx,
                &mut self.anim_ctrl_tab,
                &mut self.anim_clip_tab,
                &self.project_root,
            ) {
                let tab = if controller { Tab::AnimController } else { Tab::AnimClip };
                if let Some(loc) = self.dock_state.find_tab(&tab) {
                    self.dock_state.set_active_tab(loc);
                } else {
                    // Open next to the Scene view (wide), not in the focused leaf.
                    if let Some((surface, node, _)) = self.dock_state.find_tab(&Tab::Viewport) {
                        self.dock_state.set_focused_node_and_surface((surface, node));
                    }
                    self.dock_state.push_to_focused_leaf(tab);
                }
            }
            self.skinning.show(ctx);
            if crate::shader_graph_editor::take_open_request(ctx, &mut self.shader_graph_tab, &self.project_root) {
                let tab = Tab::ShaderGraph;
                if let Some(loc) = self.dock_state.find_tab(&tab) {
                    self.dock_state.set_active_tab(loc);
                } else {
                    if let Some((surface, node, _)) = self.dock_state.find_tab(&Tab::Viewport) {
                        self.dock_state.set_focused_node_and_surface((surface, node));
                    }
                    self.dock_state.push_to_focused_leaf(tab);
                }
            }
            if std::mem::take(&mut self.sequencer_tab.open_request) {
                if let Some(loc) = self.dock_state.find_tab(&Tab::Sequencer) {
                    self.dock_state.set_active_tab(loc);
                } else {
                    if let Some((surface, node, _)) = self.dock_state.find_tab(&Tab::Viewport) {
                        self.dock_state.set_focused_node_and_surface((surface, node));
                    }
                    self.dock_state.push_to_focused_leaf(Tab::Sequencer);
                }
            }

            // Game UI: interactive while playing; while editing, a layout preview
            // over the Scene view (where it can be edited, Game UI filter on) or
            // else over the Camera tab, which shows what the game will.
            let game_rect = self.game_rect();
            let overlay_rect = if self.playing {
                game_rect
            } else if self.viewport_visible && self.widget_filter.game_ui {
                Some(self.viewport_rect)
            } else {
                (self.camera_tab_visible && self.camera_rect.is_positive()).then_some(self.camera_rect)
            };
            if let Some(rect) = overlay_rect {
                let nav = self.input.ui_nav();
                let input = (self.playing && !self.game_cursor_locked).then(|| {
                    ctx.input(|i| {
                        let mut keys = zestengine_engine::game_ui::UiKeys::default();
                        for e in &i.events {
                            match e {
                                egui::Event::Text(t) => keys.text.push_str(t),
                                egui::Event::Key { key, pressed: true, .. } => match key {
                                    egui::Key::Backspace => keys.backspaces += 1,
                                    egui::Key::Enter => keys.enter = true,
                                    egui::Key::Escape => keys.escape = true,
                                    _ => {}
                                },
                                _ => {}
                            }
                        }
                        zestengine_engine::game_ui::UiInput {
                            pointer: zestengine_engine::game_ui::UiPointer {
                                pos: i.pointer.latest_pos().filter(|p| rect.contains(*p)),
                                down: i.pointer.primary_down(),
                                pressed: i.pointer.primary_pressed(),
                                released: i.pointer.primary_released(),
                            },
                            nav,
                            keys,
                        }
                    })
                });
                let selected = match self.selection {
                    Selection::Object(i) if !self.playing => Some(i),
                    _ => None,
                };
                self.game_ui.reveal = selected;
                let frame = self.game_ui.run(
                    ctx,
                    rect,
                    &mut self.scene,
                    &self.project_root,
                    input.as_ref(),
                    None,
                    zestengine_engine::game_ui::UiScope::Screen,
                );
                let mut avatar_menu_open = false;
                if self.playing {
                    // Hold R for the avatar's radial menu; the pointer (or the arrow
                    // keys) picks a slice, click / Enter selects, Backspace goes back.
                    let menu_in = zestengine_engine::avatar_menu::egui_desktop_input(ctx, rect, self.game_cursor_locked);
                    avatar_menu_open = self.avatar_menu.drive(&mut self.scene, &menu_in, self.play_time as f64);
                    self.avatar_menu.show(ctx, rect);
                }
                if !self.playing {
                    let nodes = zestengine_engine::game_ui::layout_rects(
                        &self.scene,
                        rect,
                        ctx.pixels_per_point(),
                        zestengine_engine::game_ui::UiScope::Screen,
                        selected,
                    );
                    let painter = ctx
                        .layer_painter(egui::LayerId::new(egui::Order::Middle, egui::Id::new("zestengine-game-ui")))
                        .with_clip_rect(rect);
                    self.ui_rect_editor.paint(&painter, &self.scene, &nodes, selected);
                    // Spec §6.3: a selected screen-UI element shows its canvas frame.
                    if let Some(c) = selected.and_then(|s| screen_canvas_of(&self.scene, s)) {
                        use crate::theme;
                        let frame = rect.shrink(3.0);
                        let corners = [frame.left_top(), frame.right_top(), frame.right_bottom(), frame.left_bottom(), frame.left_top()];
                        painter.extend(egui::Shape::dashed_line(&corners, egui::Stroke::new(1.5_f32, theme::accent_soft()), 6.0, 4.0));
                        let text = format!("Canvas · {} × {} · Screen overlay", c[0] as u32, c[1] as u32);
                        let galley = painter.layout_no_wrap(text, egui::FontId::proportional(theme::SIZE_META), theme::accent());
                        let chip = egui::Rect::from_min_size(
                            egui::pos2(frame.left() + 12.0, frame.bottom() - 30.0),
                            galley.size() + egui::vec2(14.0, 8.0),
                        );
                        painter.rect_filled(chip, egui::CornerRadius::same(theme::R_FIELD), theme::with_alpha(theme::BG_APP, 0.9));
                        painter.rect_filled(chip, egui::CornerRadius::same(theme::R_FIELD), theme::accent_faint());
                        painter.galley(chip.min + egui::vec2(7.0, 4.0), galley, theme::accent());
                    }
                }
                self.game_ui_frame.pointer_over_ui = frame.pointer_over_ui || avatar_menu_open;
                self.game_ui_frame.text_editing = frame.text_editing;
                self.game_ui_frame.cancel |= frame.cancel;
                self.game_ui_frame.clicks.extend(frame.clicks);
                self.game_ui_frame.submits.extend(frame.submits);
            }

            // Dragging a model / prefab / clip over the viewport spawns it and follows the cursor.
            self.placement_frame(ctx);

            // Clear the published "dragged object" once the pointer is up, after
            // ObjectRef drop boxes have had this frame to consume it (so a later
            // plain release over a box can't re-apply a stale drag).
            if !ctx.input(|i| i.pointer.any_down()) {
                ctx.data_mut(|d| d.remove::<usize>(egui::Id::new(crate::DRAG_OBJECT_KEY)));
                ctx.data_mut(|d| d.remove::<String>(egui::Id::new(crate::DRAG_OBJECT_NAME_KEY)));
                ctx.data_mut(|d| {
                    d.remove::<String>(egui::Id::new(crate::DRAG_FILE_KEY))
                });
            }
        });

        if let Some(egui_state) = self.egui_state.as_mut() {
            egui_state.handle_platform_output(&window, output.platform_output);
        }
        let primitives = egui_ctx.tessellate(output.shapes, output.pixels_per_point);

        zestengine_render::hitch::phase("actions");
        self.process_actions();
        // Per-probe "Bake This Probe" requests from the inspector button.
        self.poll_probe_bake_requests();
        self.poll_camera_screen_requests();
        self.step_reflection_bake_queue();
        // Reflection-probe bake: the capture renders one cube face per frame, so
        // hold the readback until the renderer swaps the finished cube in (plus a
        // short settle countdown), or we'd persist the PRE-capture cube.
        if let Some(n) = self.refl_bake_pending {
            let capturing = self
                .renderer
                .as_ref()
                .is_some_and(|r| r.reflection_capture_in_flight());
            if capturing {
                // keep waiting; the countdown starts once the capture lands
            } else if n == 0 {
                self.refl_bake_pending = None;
                self.save_reflection_bake();
            } else {
                self.refl_bake_pending = Some(n - 1);
            }
        }
        // Exit requested by a clean close or the unsaved-changes dialog. Runs
        // after process_actions so a "Save & Quit" SaveScene lands first.
        if self.quit_after_frame {
            self.save_project();
            // Let a bake's background encode finish writing its sidecars.
            crate::bake_store::wait_for_saves(std::time::Duration::from_secs(30));
            // After the saves, so the replacement reads the project as left.
            if self.relaunch_after_quit {
                self.relaunch_editor();
            }
            event_loop.exit();
            return;
        }
        zestengine_render::hitch::phase("scene_update");
        self.pump_lsp();
        let t = self.start.elapsed().as_secs_f32();
        // Step: one fixed frame while paused.
        let step = std::mem::take(&mut self.play_step) && self.playing && self.play_paused;
        let dt = if step { 1.0 / 60.0 } else { dt };
        if self.playing && (!self.play_paused || step) {
            // Advance the play clock (pauses with the sim) so time-based
            // components don't jump across a pause.
            self.play_time += dt;
            // Resolve input + networking view for this frame, then drive
            // components. Component-driven motion must not land in undo history;
            // play edits are restored wholesale on Stop anyway.
            if let Some(net) = self.net.as_mut() {
                net.pump();
            }
            let net_view = self.net.as_mut().map(|n| n.view()).unwrap_or_default();
            self.input.feed_vr(&zestengine_engine::runtime::vr_frame(self.xr_session.as_ref(), &self.vr_targets));
            self.input.resolve_frame();
            let mut commands = Vec::new();
            // Gameplay can spawn/destroy objects (indices shift); keep the
            // selection on the same objects by id.
            let selected_id = match self.selection {
                Selection::Object(i) => self.scene.objects.get(i).map(|o| o.id),
                _ => None,
            };
            let multi_ids: Vec<ObjectId> = self
                .multi_objects
                .iter()
                .filter_map(|&i| self.scene.objects.get(i).map(|o| o.id))
                .collect();
            let count_before = self.scene.objects.len();
            let screen = self.game_screen_info();
            if let (Some(session), Some(renderer)) = (self.play_session.as_mut(), self.renderer.as_mut()) {
                commands = session.tick(
                    &mut zestengine_engine::play::World {
                        scene: &mut self.scene,
                        renderer: Some(renderer),
                        registry: &self.components,
                        shaders: &mut self.shaders,
                    },
                    &zestengine_engine::play::TickInput {
                        dt,
                        input: self.input.state(),
                        net: &net_view,
                        listener_fallback: self.camera.view().inverse(),
                        screen: &screen,
                    },
                );
                let t = session.timings;
                FrameTimings::ema(&mut self.frame_timings.components, t.components_ms);
                FrameTimings::ema(&mut self.frame_timings.physics, t.physics_ms);
                FrameTimings::ema(&mut self.frame_timings.audio, t.audio_ms);
                // Test buttons on action bindings: fire them in the running game.
                for req in crate::actions_ui::take_test_requests(&self.egui_ctx) {
                    let player = self
                        .scene
                        .objects
                        .iter()
                        .find(|o| {
                            o.components.iter().any(|c| {
                                let a = c.as_any();
                                a.is::<zestengine_core::CharacterController>()
                                    || a.is::<zestengine_core::FirstPersonController>()
                                    || a.is::<zestengine_core::ThirdPersonController>()
                                    || a.is::<zestengine_core::ClickToMove>()
                            })
                        })
                        .map(|o| (o.id, o.name.clone()));
                    let owner = self.scene.index_of(req.owner).map(|i| (req.owner, self.scene.objects[i].name.clone()));
                    let Some((instigator, who)) = player.or(owner) else {
                        crate::actions_ui::post_test_result(&self.egui_ctx, req.row, false, "The object is gone.".into());
                        continue;
                    };
                    let result = session.test_binding(
                        &mut zestengine_engine::play::World {
                            scene: &mut self.scene,
                            renderer: Some(renderer),
                            registry: &self.components,
                            shaders: &mut self.shaders,
                        },
                        &zestengine_engine::play::TickInput {
                            dt,
                            input: self.input.state(),
                            net: &net_view,
                            listener_fallback: self.camera.view().inverse(),
                            screen: &screen,
                        },
                        req.owner,
                        &req.binding,
                        Some(instigator),
                    );
                    let (ok, text) = match result {
                        Ok(host) => {
                            commands.extend(host);
                            (true, format!("Fired (instigator: {who})."))
                        }
                        Err(e) => (false, e),
                    };
                    crate::actions_ui::post_test_result(&self.egui_ctx, req.row, ok, text);
                }
            }
            if self.scene.objects.len() != count_before || selected_id.is_some_and(|id| {
                !matches!(self.selection, Selection::Object(i) if self.scene.objects.get(i).is_some_and(|o| o.id == id))
            }) {
                let index = self.scene.id_index();
                if let Some(id) = selected_id {
                    self.selection = index.get(&id).map_or(Selection::None, |&i| Selection::Object(i));
                }
                self.multi_objects = multi_ids.iter().filter_map(|id| index.get(id).copied()).collect();
            }
            // Apply deferred requests (e.g. ctx.load_scene) after the update
            // pass so the scene isn't swapped mid-iteration.
            self.apply_component_commands(commands);
            self.finish_play_switch();
            if let Some(renderer) = self.renderer.as_mut()
                && self.flow.poll(Some(renderer), &self.components, &mut self.shaders, &self.project_root, &mut self.scene)
                && let Some(session) = self.play_session.as_mut()
            {
                session.structure_changed();
            }
            if let Some(renderer) = self.renderer.as_mut() {
                let content = self.content.update(&mut zestengine_engine::content::ContentHost {
                    renderer: Some(renderer),
                    registry: &self.components,
                    shaders: &mut self.shaders,
                    root: &self.project_root,
                    scene: &mut self.scene,
                    flow: &mut self.flow,
                });
                if let Some(session) = self.play_session.as_mut() {
                    if content.structure_changed {
                        session.structure_changed();
                    }
                    for e in content.events {
                        session.post_event(e);
                    }
                }
            }
            // Replicate networked objects (2G): owner sends, others apply,
            // after predicting / simulating predicted pawns.
            if let Some(net) = self.net.as_mut() {
                if let Some(session) = self.play_session.as_mut() {
                    session.net_predict(&mut self.scene, net, self.input.state(), dt);
                }
                self.scene.network_sync(net, dt);
            }
            // Voice chat + lip sync (needs a Voice Chat component in the scene).
            let listener = self.scene.audio_listener().unwrap_or(self.camera.position);
            let right = self.camera.right();
            let input = self.input.state();
            zestengine_engine::voice::run_frame(&mut self.voice, self.net.as_mut(), &mut self.scene, |a| input.down(a), listener, right);
        } else {
            // Keep network connections alive (handshakes, pings) while editing.
            if let Some(net) = self.net.as_mut() {
                net.pump();
            }
            // Co-editing: apply remote ops, broadcast presence, deliver voice.
            self.pump_edit_session();
            // Coalesce undo entries only while a pointer drag is in progress (one
            // slider/handle gesture = one entry); discrete edits each get their own.
            let gesture_active = self.egui_ctx.input(|i| i.pointer.any_down());
            self.undo_stack.set_stroke(crate::cloth_paint::stroking(&self.egui_ctx) || crate::terrain_paint::stroking(&self.egui_ctx));
            // Tile strokes: per-cell diffs become their own undo entries, and the
            // generic component diff is skipped for frames where tiles changed.
            if crate::tile_edit::take_edited(&self.egui_ctx) {
                self.suppress_undo_record = true;
                self.scene_dirty = true;
            }
            for (object, edits) in crate::tile_edit::drain_strokes(&self.egui_ctx) {
                self.undo_stack.record(UndoEntry::Tiles { object, edits }, false);
            }
            self.record_edits(pre_edit, gesture_active);
            // Not simulating → decay the play-only timings toward zero.
            FrameTimings::ema(&mut self.frame_timings.components, 0.0);
            FrameTimings::ema(&mut self.frame_timings.physics, 0.0);
            FrameTimings::ema(&mut self.frame_timings.audio, 0.0);
        }
        self.autosave_materials();
        // Cameras always carry a Camera component (covers spawns, loaded
        // scenes, and scenes from before the component existed).
        self.scene.ensure_camera_components(&self.components);
        self.scene.ensure_light_components(&self.components);
        self.scene.ensure_camera_ids();
        self.sequencer_tab.tick(&mut self.scene, dt);
        // Highlight every multi-selected object (anchor + extras) in the viewport.
        zestengine_render::hitch::phase("draw_prep");
        let selected = self.selected_object_indices();
        let draws_t = Instant::now();
        // Apply the MAIN CAMERA's culling mask on top of the viewport's layer toggle,
        // so deselecting a layer in a camera's Culling Mask hides it in the editor too
        // (previously the editor only used the Layers-window toggle, so the per-camera
        // mask did nothing until you ran the game). Restored right after the draw
        // build so the Layers window still edits the toggle, not the intersection.
        self.update_world_ui();
        if let Some(renderer) = self.renderer.as_mut() {
            let rig = self.vr_rig;
            self.world_ui_draws.extend(self.body.panel_frame(renderer, &mut self.scene, &rig, dt));
            if self.playing {
                let vr = zestengine_engine::runtime::vr_frame(self.xr_session.as_ref(), &self.vr_targets);
                self.world_ui_draws.extend(self.avatar_menu_vr.frame(renderer, &mut self.scene, &rig, &vr, dt));
            } else if self.avatar_menu_vr.is_open() {
                // Play stopped with the menu up: close it (an inactive frame).
                self.avatar_menu_vr.frame(renderer, &mut self.scene, &rig, &zestengine_core::VrFrame::default(), dt);
            }
        }
        self.update_particles(dt);
        let view_toggle = self.scene.visible_layers;
        self.scene.visible_layers = view_toggle & self.scene.main_camera_culling_mask();
        self.scene.occlusion_camera = self.occlusion_camera();
        self.scene.lightmap_preview = self.lightmap_preview;
        // Scene-view hide / lock act only while editing; Play ignores them.
        self.scene.scene_view.edit = !self.playing;
        self.scene.sync_draws_multi(&selected, 1.0);
        if let Some(renderer) = self.renderer.as_mut() {
            let st = &self.project.settings;
            let ppp = self.egui_ctx.pixels_per_point();
            crate::theme::set_selection_outline_enabled(st.selection_outline);
            crate::theme::set_selection_outline(st.selection_outline_color.map(|[r, g, b]| egui::Color32::from_rgb(r, g, b)));
            let a = crate::theme::selection_outline();
            let c = [a.r(), a.g(), a.b()];
            renderer.set_ao_quality(st.ao_quality);
            // The scene camera's override only applies while playing; the editor camera uses the project mode.
            renderer.set_anti_aliasing(if self.playing { zestengine_engine::runtime::resolve_anti_aliasing(&self.scene, st.anti_aliasing) } else { st.anti_aliasing });
            renderer.set_taa_sharpen(st.taa_sharpen);
            renderer.set_selection_outline(zestengine_render::SelectionOutlineStyle {
                enabled: st.selection_outline,
                color: [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0],
                width_px: (st.selection_outline_width * ppp).max(1.0),
                occluded: 0.35,
            });
            // Scene-view ground grid, spaced by the move-snap step. Off while
            // the Scene view shows the game camera.
            let rgb = |c: egui::Color32| [c.r() as f32 / 255.0, c.g() as f32 / 255.0, c.b() as f32 / 255.0];
            use crate::theme;
            renderer.set_editor_grid(zestengine_render::EditorGridStyle {
                enabled: st.scene_grid && !(self.playing && self.play_game_camera),
                plane: zestengine_render::GridPlane::Auto,
                unit: self.gizmo.grid_size.max(0.01),
                opacity: st.scene_grid_opacity.clamp(0.0, 1.0),
                color: rgb(theme::TEXT_DIM),
                axis_colors: [rgb(theme::AXIS_X), rgb(theme::AXIS_Y), rgb(theme::AXIS_Z)],
            });
        }
        self.scene.visible_layers = view_toggle;
        FrameTimings::ema(
            &mut self.frame_timings.draws,
            draws_t.elapsed().as_secs_f32() * 1000.0,
        );
        // The sun is a Directional Light object in the scene: the light list is
        // just the scene's lights, and the key light (skybox sun disc, fog
        // in-scatter, custom-shader u_light_dir) is derived from its main
        // directional light (see `key_light_data`).
        let env = self.scene.environment.clone();
        let lights = self.scene.gather_lights();
        let world_light = zestengine_engine::scene::key_light_data(
            &lights,
            self.scene.baked_ambient().unwrap_or([
                env.ambient[0] * env.ambient_intensity,
                env.ambient[1] * env.ambient_intensity,
                env.ambient[2] * env.ambient_intensity,
            ]),
            // Skybox IBL (the env-cube specular reflection on metallic/smooth
            // surfaces) is gated by the Enable-skybox toggle: off -> metals stop
            // mirroring the sky, so with no lights the scene is black.
            if env.skybox_enabled { 1.0 } else { 0.0 },
        );

        // Render the camera preview only when the Camera tab is actually VISIBLE
        // (the active tab in its dock leaf this frame), or when a camera object is
        // selected (shown as a viewport overlay so its framing can be tweaked
        // live). The tab merely being present in the layout no longer forces the
        // (now Flux-traced) preview to render every frame. Selected camera takes
        // precedence over the main one.
        let selected_camera = match self.selection {
            Selection::Object(i) if self.scene.camera_view_proj_for(i, 1.0).is_some() => Some(i),
            _ => None,
        };
        let preview_view = selected_camera
            .and_then(|i| self.scene.camera_view_proj_for(i, 16.0 / 9.0))
            .or_else(|| {
                self.camera_tab_visible
                    .then(|| self.scene.main_camera_view_proj(16.0 / 9.0))
                    .flatten()
            });
        let camera_preview = preview_view.map(|(view, proj, position)| CameraData {
            view,
            proj,
            position,
        });

        let shadow_res = env.shadow_resolution.clamp(256, 8192);
        let shadow_pcf_texel = env.shadow_softness.max(0.0) / shadow_res as f32;

        // VR: pump the OpenXR session + read tracker poses for the IK solver and
        // controller inputs for locomotion/interaction. Done before binding the
        // renderer so `update_vr` (which mutates the scene/selection) doesn't clash
        // with the renderer borrow held below.
        let mut vr_input = zestengine_xr::VrInput::default();
        if let Some(session) = &mut self.xr_session {
            session.poll();
            let b = session.body_poses();
            let mut devices = zestengine_engine::body::xr_devices(&b);
            if let Some(slime) = &self.slimevr {
                slime.send_devices(b.head, b.left_hand, b.right_hand);
                devices.extend(slime.devices());
            }
            self.vr_targets = self.body.assemble(b.head, b.left_hand, b.right_hand, devices);
            vr_input = session.input();
            self.vr_hand_joints = session.hand_joints();
        }
        // In Play a VR Player owns locomotion: the headset rig follows it.
        let game_rig = self.playing.then(|| self.scene.vr_player_rig()).flatten();
        if let Some(rig) = game_rig {
            self.vr_rig = rig;
        } else if self.xr_session.is_some() {
            self.update_vr(vr_input, dt);
        }

        let game_view_now = self.game_view(aspect);
        // The Scene camera as it is NOW. `view` / `proj` above were taken before the egui
        // pass; scroll dolly and orbit move the camera inside it (`viewport_ui`) and any
        // action applied by `process_actions` (focus 'F', undo, spawn-and-frame, the
        // game-camera wheel edge case) after it, so they would be stale here while
        // `self.camera.position` below is current. Rendering that mix once moved the
        // camera's position without its view for one frame: the sky (which cancels the
        // camera translation with `camera_pos`), specular, LOD and culling all disagreed
        // for a frame and the viewport flickered on every scroll / orbit step. Right-drag
        // look never showed it: `update_camera` runs before the matrices are read.
        let (view, proj) = (self.camera.view(), self.camera.proj(aspect));
        self.poll_zestsurface();
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        if let Err(e) = renderer.set_shadow_resolution(shadow_res) {
            tracing::error!("setting shadow resolution: {e:#}");
        }
        // Feed VR tracker poses into the scene's humanoid IK targets (any source
        // can; IK isn't VR-specific). Remap through the captured T-pose calibration
        // when one exists so each tracker drives its bone naturally. Then advance +
        // CPU-skin (previews in-editor).
        zestengine_render::hitch::phase("skinning_pose");
        let face = self.face_input.poll(self.xr_session.as_ref());
        self.scene.set_local_face(face);
        let osc_msgs = self.face_input.take_avatar_osc();
        let ti = self.project.tracking_input;
        self.avatar_osc.configure(ti.osc_out_port, ti.osc_query.then_some(ti.osc_port));
        self.avatar_osc.frame(&mut self.scene, osc_msgs, self.playing && self.project.tracking_input.osc_enabled, self.play_time as f64);
        if self.project.tracking_input.slimevr != self.slimevr.is_some() {
            self.slimevr = self.project.tracking_input.slimevr.then(zestengine_engine::slimevr::SlimeVr::start);
        }
        let tracking = self.xr_session.as_ref().is_some_and(|s| s.running);
        let body_frame = zestengine_engine::body::BodyFrame { dt, tracking, apply: self.playing, triggers: [vr_input.left_trigger, vr_input.right_trigger] };
        let vr_targets = self.body.update(&mut self.scene, &self.vr_targets, body_frame);
        self.scene
            .set_ik_targets(self.xr_session.as_ref().map(|_| vr_targets));
        self.scene.update_water_wind(dt);
        self.scene.set_cloth_viewer(Some(game_view_now.map_or(self.camera.position, |g| g.2)));
        self.scene.weapon_poses_live = self.playing;
        // The Configure window's 3D figure poses its scene instance (reference pose, forced
        // T-pose or the muscle sliders) instead of the animation; nothing is saved or undone.
        let configured_object = self
            .rig_ui
            .configure_model()
            .and_then(|m| self.scene.object_using_model(m, &self.project_root, if let Selection::Object(i) = self.selection { Some(i) } else { None }))
            .filter(|_| !self.playing);
        let forced_pose = self.rig_ui.take_pose_override();
        self.scene.pose_override = match (configured_object, &forced_pose) {
            (Some(i), Some(locals)) => Some((i, locals.clone())),
            _ => None,
        };
        // Animator culling tests against the view this frame renders (the game camera in Play
        // when the viewport shows it, else the Scene camera). A headset's eyes are not
        // described here, so culling is off while one runs.
        if self.xr_session.is_some() {
            self.scene.clear_cull_view();
        } else {
            let (cv, cp) = game_view_now.map_or((view, proj), |g| (g.0, g.1));
            self.scene.set_cull_view(&(cp * cv));
        }
        // Own phases, so a log names the skinning itself apart from the tracking/IK/camera work around it.
        zestengine_render::hitch::phase("skin_update");
        self.scene.update_skinning(renderer, dt);
        zestengine_render::hitch::phase("view_gather");
        // Re-read the game camera after this frame's tick so it isn't a frame behind.
        let (view, proj, cam_pos) = game_view_now.unwrap_or((view, proj, self.camera.position));
        // Preview first, so the Environment tab's override list reflects the
        // Scene camera (the main call runs last).
        let water_t = self.water.time();
        let preview_postfx = camera_preview.as_ref().map(|c| zestengine_render::PostFx {
            underwater: zestengine_engine::underwater::view_fx(&self.scene, c.position, water_t),
            ..self.scene.effective_postfx(c.position, &self.project_root, water_t)
        });
        // The two reflection probes around the camera (blended per pixel);
        // realtime ones keep re-capturing so reflections track dynamic lighting,
        // paused while a bake owns the capture.
        let baking = self.refl_bake_pending.is_some() || self.refl_bake_current.is_some() || !self.refl_bake_queue.is_empty();
        let refl = self.refl_slots.update(renderer, self.scene.active_reflection_probes(cam_pos), !baking);
        let camera_views = self.scene.camera_target_views();
        for v in &camera_views {
            if let Err(e) = renderer.resize_camera_target(v.id, v.size[0], v.size[1]) {
                tracing::error!("resizing render-texture camera: {e:#}");
            }
        }
        let postfx = zestengine_render::PostFx {
            underwater: zestengine_engine::underwater::view_fx(&self.scene, cam_pos, water_t),
            ..self.scene.effective_postfx(cam_pos, &self.project_root, water_t)
        };
        let preview_postfx = preview_postfx.unwrap_or(postfx);
        let planar = self.scene.planar_surfaces();
        // A roof over the camera hides the skybox (the water's planar layer
        // then carries the reflections instead of the sky env).
        let under_roof = self.scene.camera_under_roof(cam_pos);
        tracing::debug!("under_roof: {under_roof} at {cam_pos:?}");
        let decals = self.scene.gather_decals(renderer, &self.project_root);
        let area_lights = self.scene.gather_area_lights(renderer, &self.project_root);
        renderer.set_area_lights(&area_lights);
        // GameObject tiles in edit mode (Play does this itself): painting a cell shows its object at once.
        if !self.playing {
            self.scene.sync_tile_objects(Some(&mut *renderer), &mut self.shaders, &self.project_root, &self.components);
        }
        // 2D sprites: the renderer draws what is queued here this frame.
        self.scene.gather_sprites(renderer, &self.project_root, view, proj, self.water.time());
        // Full light set (incl. baked) for the reflection-probe cube capture, so
        // a baked scene's reflections aren't captured in the dark. Only gathered
        // while a capture is queued or running (empty = use `lights`).
        let voxel_emitters = {
            let gi = &self.scene.environment.realtime_gi;
            if gi.enabled && gi.mode == zestengine_assets::GiMode::FluxVoxel {
                self.scene.flux_voxel_dynamic_emitters()
            } else {
                Vec::new()
            }
        };
        let capture_lights = if renderer.reflection_capture_in_flight() {
            self.scene.gather_lights_all()
        } else {
            Vec::new()
        };
        // The Configure window's model preview: only the configured instance's draws.
        let model_draws = configured_object.map(|i| self.scene.object_draw_indices(i)).unwrap_or_default();
        let model_request = if configured_object.is_some() { self.rig_ui.preview.take_request() } else { None };
        // The inspector's asset preview renders model-only; the rig window owns the target while it is open.
        // No scene object uses the model: the window's figure renders the model file on its own.
        let figure_model = self.rig_ui.configure_model().filter(|_| configured_object.is_none()).map(Path::to_path_buf);
        let figure_request = if figure_model.is_some() { self.rig_ui.preview.take_request() } else { None };
        self.avatar_figure.frame(&self.egui_ctx, renderer, figure_model.as_deref(), figure_request, forced_pose.as_deref());
        self.asset_preview.frame(renderer, configured_object.is_none() && figure_model.is_none());
        let figure_world = figure_model.is_some().then_some(glam::Mat4::IDENTITY).filter(|_| self.avatar_figure.usable());
        self.rig_ui.set_avatar(configured_object.map(|i| self.scene.world_transform(i)).or(figure_world));
        let mut frame = FrameInput {
            clear_color: [0.016, 0.016, 0.024, 1.0],
            camera: CameraData {
                view,
                proj,
                position: cam_pos,
            },
            light: world_light,
            lights: &lights,
            capture_lights: &capture_lights,
            voxel_emitters: &voxel_emitters,
            camera_preview,
            model_preview: model_request.map(|r| zestengine_render::ModelPreviewInput {
                camera: CameraData { view: r.view, proj: r.proj, position: r.position },
                size: r.size,
                draws: &model_draws,
            }),
            draw_skybox: env.skybox_enabled && !under_roof,
            shadow_pcf_texel,
            shadow_bias_scale: env.shadow_bias_scale,
            shadow_normal_offset: env.shadow_normal_offset,
            shadow_distance: env.shadow_distance,
            shadow_cascades: env.shadow_cascades,
            shadow_cascade_lambda: env.shadow_cascade_lambda,
            time: t,
            water_time: self.water.time(),
            water: self.water.bodies(),
            draws: &self.scene.draws,
            decals: &decals,
            lightmap_preview: self.lightmap_preview,
            gi_debug: self.gi_debug,
            // The editor renders the scene into the viewport texture (RTT), never
            // the swapchain — so the swapchain pass is clear + egui only.
            render_viewport: false,
            viewport_extent,
            postfx,
            preview_postfx,
            reflection_probe: refl[0],
            reflection_probes: refl,
            reflection_probe_b: refl[1],
            reflection_primary: 0,
            camera_targets: &camera_views,
            planar: &planar,
            fog: self.scene.fog_params(),
            voxel_specular: self.scene.environment.realtime_gi.mode
                == zestengine_assets::GiMode::FluxVoxel
                && self.scene.environment.realtime_gi.voxel_specular,
            egui: Some(zestengine_render::EguiDraw {
                pixels_per_point: output.pixels_per_point,
                primitives,
                textures_delta: output.textures_delta,
            }),
            world_ui: std::mem::take(&mut self.world_ui_draws),
        };
        let render_t = Instant::now();
        if let Err(e) = renderer.render(&frame) {
            tracing::error!("render failed: {e:#}");
            renderer.report_device_lost(&e);
            event_loop.exit();
        }
        // `ZESTENGINE_PLAY=1`: enter Play mode once the scene has loaded
        // (headless QA of the game-camera path; use with ZESTENGINE_SHOT).
        if std::env::var_os("ZESTENGINE_PLAY").is_some()
            && SHOT_FRAMES.load(std::sync::atomic::Ordering::Relaxed) == 120
            && !self.playing
        {
            // Queued: `toggle_play` itself is applied outside the render borrow.
            self.actions.push(EditorAction::TogglePlay);
        }
        // `ZESTENGINE_SHOT=<png>`: dump rendered scene-view frames for visual
        // QA - from frame 60 on, every 120 frames (the early frames load the
        // scene), written as `<png>.<n>.png`.
        if std::env::var_os("ZESTENGINE_SHOT").is_some() {
            let n = SHOT_FRAMES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if (60..2400).contains(&n)
                && n.is_multiple_of(120)
                && let Ok(Some((w, h, px))) = renderer.debug_read_viewport()
                && let Ok(p) = std::env::var("ZESTENGINE_SHOT")
            {
                let path = format!("{p}.{n:04}.png");
                if let Err(e) = image::save_buffer(&path, &px, w, h, image::ColorType::Rgba8) {
                    tracing::error!("ZESTENGINE_SHOT: {e:#}");
                }
            }
        }
        // `ZESTENGINE_FRAME_STATS=1`: per-pass GPU zone timings in the log (the
        // profiler window shows the same zones live).
        if zestengine_engine::runtime::frame_stats_on() {
            renderer.set_gpu_zone_timing(true);
            zestengine_engine::runtime::frame_stats(renderer);
        }

        // VR stereo: also render the scene into the headset's per-eye images. Gated
        // on an active XR session, so the desktop editor is unaffected when flat.
        // The eye view = the XR stage-space view composed with the play-space rig
        // (fly/scale/drag), so locomotion moves the player through the world.
        if self.xr_session.is_some() {
            if self.xr_stereo.is_none() {
                // The window's format when the runtime offers it (as before).
                let fmt = renderer.color_format_raw();
                if let Some(session) = self.xr_session.as_mut() {
                    match session.setup_stereo(&[fmt], false) {
                        Ok(setup) => {
                            self.xr_stereo = Some(setup);
                            tracing::info!("XR stereo swapchains ready ({}x{}, VkFormat {})", setup.width, setup.height, setup.format);
                        }
                        Err(e) => tracing::warn!("XR setup_stereo failed: {e:#}"),
                    }
                }
            }
            if let Some(stereo) = self.xr_stereo {
                let fmt = stereo.format;
                let stage_to_world = self.vr_rig.stage_to_world_mat();
                let world_to_stage = stage_to_world.inverse();
                // Controller markers always show (so you can see your tracked
                // hands), even when the hand menu is down. Map the stage-space
                // tracker poses into world space for the eye render.
                let hand_mat = |h: Option<(glam::Vec3, glam::Quat)>| {
                    h.map(|(p, q)| stage_to_world * glam::Mat4::from_rotation_translation(q, p))
                };
                let left_hand = hand_mat(self.vr_targets.left_hand);
                let right_hand = hand_mat(self.vr_targets.right_hand);
                // Per-hand laser endpoint: cast from the controller along its
                // forward and stop at the nearest of {any UI panel, a scene
                // object}, otherwise reach far (2000). Both hands stop on hits.
                let panels = self
                    .vr_overlay_draw
                    .as_ref()
                    .map(|o| o.panels)
                    .unwrap_or([None; zestengine_render::VR_MAX_PANELS]);
                let laser_end = |hand: Option<glam::Mat4>| -> Option<glam::Vec3> {
                    let m = hand?;
                    let origin = m.w_axis.truncate();
                    let dir = (-m.z_axis.truncate()).normalize_or(glam::Vec3::NEG_Z);
                    let mut t = 2000.0_f32;
                    for vpanel in panels.iter().flatten() {
                        let pm = vpanel.model;
                        let ppos = pm.w_axis.truncate();
                        let pn = pm.z_axis.truncate().normalize_or(glam::Vec3::Z);
                        if let Some(hit) =
                            zestengine_core::ray_plane(zestengine_core::Ray { origin, dir }, ppos, pn)
                        {
                            let local = pm.inverse().transform_point3(hit);
                            if local.x.abs() <= 0.5 && local.y.abs() <= 0.5 {
                                t = t.min((hit - origin).length());
                            }
                        }
                    }
                    if let Some(d) = self.scene.ray_hit(origin, dir) {
                        t = t.min(d);
                    }
                    Some(origin + dir * t)
                };
                let left_laser_end = laser_end(left_hand);
                let right_laser_end = laser_end(right_hand);
                // Hand-tracking skeletons in world space for the overlay.
                let hand_skel = |j: &Option<zestengine_xr::HandJoints>| {
                    j.map(|h| zestengine_render::HandSkeleton {
                        positions: h.positions.map(|p| stage_to_world.transform_point3(p)),
                        valid: h.valid,
                    })
                };
                let left_hand_skeleton = hand_skel(&self.vr_hand_joints[0]);
                let right_hand_skeleton = hand_skel(&self.vr_hand_joints[1]);
                // Remote co-edit peers: one color per peer id, true-scale
                // heads and tracked hands (the XR eyes show them next to the
                // local hands; presence is world-space already).
                let peer_rgb = |id: u64| {
                    const PALETTE: [[f32; 3]; 4] = [[0.4, 0.9, 0.4], [0.9, 0.4, 0.9], [1.0, 0.85, 0.2], [0.3, 0.9, 0.9]];
                    PALETTE[(id as usize) % PALETTE.len()]
                };
                let mut remote_hands = Vec::new();
                let mut remote_heads = Vec::new();
                for presence in self.edit_peers.values() {
                    let rgb = peer_rgb(presence.peer);
                    if let Some((pos, _)) = presence.head {
                        remote_heads.push((pos, 0.11 * presence.scale.max(0.1), rgb));
                    }
                    for hand in presence.hands.into_iter().flatten() {
                        remote_hands.push((
                            zestengine_render::HandSkeleton { positions: hand.positions.map(glam::Vec3::from), valid: hand.valid },
                            rgb,
                        ));
                    }
                }
                let overlay = match self.vr_overlay_draw.take() {
                    Some(mut ov) => {
                        ov.left_hand = left_hand;
                        ov.right_hand = right_hand;
                        ov.left_hand_skeleton = left_hand_skeleton;
                        ov.right_hand_skeleton = right_hand_skeleton;
                        ov.left_laser_end = left_laser_end;
                        ov.right_laser_end = right_laser_end;
                        ov.remote_hands = remote_hands;
                        ov.remote_heads = remote_heads;
                        Some(ov)
                    }
                    None => Some(zestengine_render::VrOverlayDraw {
                        panels: [None; zestengine_render::VR_MAX_PANELS],
                        left_hand,
                        right_hand,
                        left_hand_skeleton,
                        right_hand_skeleton,
                        left_laser_end,
                        right_laser_end,
                        remote_hands,
                        remote_heads,
                    }),
                };
                // Only re-render the UI texture when the menu (some panel) is up.
                if overlay.as_ref().is_some_and(|o| o.panels.iter().any(|p| p.is_some()))
                    && let Some(egui_draw) = frame.egui.as_ref()
                        && let Err(e) = renderer.render_vr_ui(egui_draw) {
                            tracing::error!("VR UI render: {e:#}");
                        }
                let begun = self
                    .xr_session
                    .as_mut()
                    .map(|s| s.begin_frame())
                    .unwrap_or(Ok(None));
                match begun {
                    Ok(Some(xr_frame)) => {
                        for (index, eye) in xr_frame.eyes.iter().enumerate() {
                            let world_view = eye.view * world_to_stage;
                            let eye_pos = world_view.inverse().w_axis.truncate();
                            frame.postfx.underwater = zestengine_engine::underwater::view_fx(&self.scene, eye_pos, water_t);
                            if let Err(e) = renderer.render_xr_eye_indexed(
                                index, eye.image, fmt, eye.width, eye.height, world_view, eye.proj,
                                &frame, overlay.clone(),
                            ) {
                                tracing::error!("XR eye render: {e:#}");
                            }
                        }
                        if let Some(session) = self.xr_session.as_mut()
                            && let Err(e) = session.end_frame(xr_frame) {
                                tracing::error!("XR end_frame: {e:#}");
                            }
                    }
                    Ok(None) => {}
                    Err(e) => tracing::error!("XR begin_frame: {e:#}"),
                }
            }
        }
        FrameTimings::ema(
            &mut self.frame_timings.render,
            render_t.elapsed().as_secs_f32() * 1000.0,
        );

        // The first real editor frame has now been presented, so close the
        // startup splash window (dropping it closes the borderless window).
        if self.splash.is_some() {
            self.splash = None;
        }

        // Profiler window: reconcile open/close (it's a separate OS window) and
        // render its egui UI. Cheap no-op when closed.
        self.update_profiler_window(event_loop);

        // Deferred heavy jobs: this frame already painted the busy overlay, so
        // run the blocking work now (the next frame shows the result + clears it).
        if self.reload_pending {
            self.reload_pending = false;
            self.request_plugin_build();
        }
        if let Some(job) = self.pending_job.take() {
            match job {
                PendingJob::ReloadPlugins => self.request_plugin_build(),
            }
        }
        if self.plugin_gate.is_building() {
            let line = self.plugin_status.lock().unwrap().clone();
            self.set_status(if line.is_empty() { "Building components…".to_string() } else { line }, true);
        }

        // With the limiter on, `about_to_wait` requests the next redraw once the
        // deadline is near; requesting here would defeat its event-loop sleep.
        if !self.pacing.active() {
            window.request_redraw();
        }
    }
}

/// Dock tab renderer; collects actions for post-UI processing.
/// Cursor position (egui points) → world ray from `camera` through the scene
/// viewport `rect` (the scene renders into a texture filling that rect, so NDC
/// is relative to it). Shared by the editor (`cursor_ray`) and the in-pass
/// orbit pivot lock in `viewport_ui`, which has only the camera and the rect.
fn cursor_ray_for(camera: &FlyCamera, rect: egui::Rect, pos: egui::Pos2) -> Option<(Vec3, Vec3)> {
    if !rect.is_finite() || rect.width() < 1.0 || rect.height() < 1.0 {
        return None;
    }
    let ndc_x = 2.0 * ((pos.x - rect.left()) / rect.width()) - 1.0;
    let ndc_y = 1.0 - 2.0 * ((pos.y - rect.top()) / rect.height());
    let aspect = rect.width() / rect.height();
    // Orthographic rays are parallel and start at the pointer on the near
    // plane; perspective rays fan out from the eye.
    let (origin, dir) = zestengine_core::ndc_ray(
        camera.view(),
        camera.proj(aspect),
        glam::Vec2::new(ndc_x, ndc_y),
        camera.position,
    )
    .unwrap_or((camera.position, camera.forward()));
    Some((origin, dir))
}

/// Build the file inspector's FileMaterial from a .material file on disk
/// (nothing for other extensions). Shared by the editor and the tab draw code.
fn load_material_file_into(path: &std::path::Path, out: &mut Option<FileMaterial>) {
    *out = None;
    if path.extension().is_some_and(|e| e == "material") {
        match zestengine_assets::load_material_file(path) {
            Ok(file) => {
                let mut model = model_from_material(
                    &file.name,
                    &file.params,
                    &file.features,
                    file.textures.normal.is_some(),
                );
                model.shader = file.shader.clone();
                if let Some(q) = file.render_queue {
                    model.render_queue = q;
                }
                model.textures = zestengine_engine::scene::tex_paths_from_file(&file.textures);
                *out = Some(FileMaterial {
                    path: path.to_path_buf(),
                    file,
                    model,
                    dirty: false,
                });
            }
            Err(e) => tracing::error!("opening material: {e:#}"),
        }
    }
}

struct EditorTabs<'a> {
    /// Shortened labels for tabs in a leaf too narrow for all of them.
    fit_labels: &'a HashMap<Tab, String>,
    /// Set true by `camera_ui` when the Camera tab is the visible/active tab.
    camera_visible: &'a mut bool,
    /// Set true by `viewport_ui` when the Viewport tab is the visible/active tab.
    viewport_visible: &'a mut bool,
    vp_strip_width: &'a mut f32,
    scene: &'a mut LoadedScene,
    selection: &'a mut Selection,
    multi_objects: &'a mut Vec<usize>,
    multi_edit: &'a mut MultiEdit,
    multi_files: &'a mut Vec<PathBuf>,
    inspector: &'a mut InspectorPanel,
    inspector_lock_target: &'a mut Option<Selection>,
    scene_panel: &'a mut ScenePanel,
    prefab: &'a mut prefab_editor::PrefabEditor,
    file_browser: &'a mut FileBrowser,
    file_diagnostics: &'a HashMap<PathBuf, (u32, u32)>,
    editor_components: &'a EditorComponents,
    vim_mode: bool,
    file_material: &'a mut Option<FileMaterial>,
    file_meta: &'a mut Option<FileMeta>,
    texture_ui: &'a mut crate::texture_inspector::TextureUi,
    rig_ui: &'a mut crate::rig_editor::RigUi,
    asset_preview: &'a mut crate::asset_preview::AssetPreview,
    sprite_atlas_ui: &'a mut crate::sprite_atlas_ui::SpriteAtlasInspector,
    model_ui: &'a mut crate::model_import_editor::ModelImportUi,
    slot_validator: &'a mut crate::texture_inspector::SlotValidator,
    open_editors: &'a mut Vec<OpenEditor>,
    nvim_embed: &'a mut crate::nvim_embed::NvimEmbed,
    /// Shared global editor zoom (Ctrl+scroll in a code editor).
    code_zoom: &'a mut f32,
    /// Path of the focused code tab (drives Inspector diagnostics).
    focused_code: Option<PathBuf>,
    gizmo: &'a mut GizmoState,
    widget_filter: &'a mut WidgetFilter,
    /// Remote co-edit peer presence (world space) for the viewport rig
    /// overlay.
    edit_peers: &'a std::collections::HashMap<u64, zestengine_engine::edit_net::PeerPresence>,
    ui_editor: &'a mut crate::ui_editor::UiEditorTab,
    anim_ctrl_tab: &'a mut crate::anim_controller_editor::AnimControllerTab,
    anim_clip_tab: &'a mut crate::anim_clip_editor::AnimClipTab,
    sequencer_tab: &'a mut crate::sequencer_editor::SequencerTab,
    shader_graph_tab: &'a mut crate::shader_graph_editor::ShaderGraphTab,
    occlusion_ui: &'a mut OcclusionUi,
    ui_rect_editor: &'a mut crate::ui_edit::UiRectEditor,
    playing: bool,
    /// The Content Builder is open: the scene is being authored as user content,
    /// so event-binding dropdowns offer only `ugc_safe` actions.
    ugc_context: bool,
    /// Expanded viewport stats pill (draw calls).
    viewport_stats: &'a mut bool,
    /// What share of the scene's triangles draws as zestlets and why the rest does not (the coverage view's legend).
    zestlet_coverage: zestengine_render::ClusterCoverage,
    /// (fps, frame ms, scene draw calls) for the stats pill.
    stats: (f32, f32, u32),
    /// Visible tabs last frame (accent underline) and this frame.
    active_tabs_prev: &'a std::collections::HashSet<Tab>,
    active_tabs: std::collections::HashSet<Tab>,
    /// Screen UI in the viewport overlay is editable (not playing, overlay on).
    ui_overlay_edit: bool,
    project_root: &'a Path,
    /// Viewport render mode (0 lit, 1 world normals, 2 indirect GI, 3 unlit).
    gi_debug: &'a mut u32,
    gizmo_drag: &'a mut bool,
    orbit_armed: &'a mut bool,
    /// The Scene (fly) camera. Scroll dolly and left-drag orbit move it HERE,
    /// inside the egui pass and before any viewport overlay is drawn, and then
    /// refresh `view` / `proj`: the overlays (gizmo, icons, handles, placement
    /// preview) and this frame's render, which reads the camera after the pass,
    /// then show the same pose. Applying them as actions after the pass left the
    /// overlays a frame behind the geometry on every wheel notch / orbit step.
    camera: &'a mut FlyCamera,
    /// The orbit pivot, locked on the first drag frame and cleared on release.
    orbit_pivot: &'a mut Option<Vec3>,
    /// The Scene view shows the game camera (play + "Game camera"): `view` /
    /// `proj` are the game's, the wheel goes to the game, and the fly camera is
    /// off screen so moving it must not touch the overlay matrices.
    scene_shows_game: bool,
    /// Winit-based "pointer is over the viewport" (robust after cursor-grab).
    pointer_in_viewport: bool,
    actions: &'a mut Vec<EditorAction>,
    viewport_rect: &'a mut egui::Rect,
    registry: &'a ComponentRegistry,
    shader_list: &'a [String],
    shader_info: Option<&'a ShaderUiInfo>,
    /// Reflected shader info of the open material sections of the selected object, by material.
    section_shader_infos: &'a std::collections::HashMap<usize, ShaderUiInfo>,
    /// Main-camera preview (egui texture + pixel size), if the renderer has
    /// rendered one yet.
    camera_preview: Option<(egui::TextureId, [f32; 2])>,
    camera_rect: &'a mut egui::Rect,
    /// The editor viewport's offscreen render (RTT), painted to fill the tab.
    viewport_texture: Option<(egui::TextureId, [f32; 2])>,
    /// A camera object is selected → draw its preview as a viewport overlay.
    camera_overlay: bool,
    view: glam::Mat4,
    proj: glam::Mat4,
    looking: bool,
    log_filter: &'a mut LogFilter,
    handle_drag: &'a mut Option<GizmoDrag>,
    /// GPU can ray-trace (Baker tab enables its Bake button).
    can_bake: bool,
    /// Baker tab: lightmap-UV checker preview toggle.
    lightmap_preview: &'a mut bool,
    /// Baker tab: model paths whose lightmap UVs were auto-generated (a `.lmuv`
    /// marker exists), so they can be regenerated/reverted. Computed per frame.
    generated_uv_models: &'a [String],
    /// Baker tab: on-disk / VRAM sizes of the scene's bake sidecars.
    bake_sizes: crate::bake_store::BakeSizes,
    /// Project Settings tab: the project file being edited in place.
    project: &'a mut zestengine_assets::ProjectFile,
    /// Project Settings > Input: the bindings editor state.
    bindings_editor: &'a mut crate::BindingsEditor,
    /// Live input manager (bindings push + Play-mode remaps).
    input: &'a mut zestengine_engine::input_engine::InputManager,
    /// Build status line shown at the bottom of Project Settings.
    build_status: &'a Option<String>,
    /// Build target picker in Project Settings.
    build_target: &'a mut bundle::BuildTarget,
    /// A build is running (disables the Build Game button).
    building_game: &'a bool,
    /// SlimeVR server status (Project Settings > External tracking).
    slimevr: &'a Option<zestengine_engine::slimevr::SlimeVr>,
    /// Face-tracking OSC status (Project Settings > External tracking).
    face_input: &'a zestengine_engine::face_input::FaceInput,
    avatar_preview: &'a mut crate::avatar_edit::TogglePreview,
    /// Set when Project Settings edits the project; saved after the dock pass.
    project_settings_dirty: &'a mut bool,
}

impl egui_dock::TabViewer for EditorTabs<'_> {
    type Tab = Tab;

    fn title(&mut self, tab: &mut Tab) -> egui::WidgetText {
        self.fit_labels.get(tab).cloned().unwrap_or_else(|| tab_label(tab)).into()
    }

    fn clear_background(&self, tab: &Tab) -> bool {
        !matches!(tab, Tab::Viewport)
    }

    fn on_close(&mut self, tab: &mut Tab) -> egui_dock::tab_viewer::OnCloseResponse {
        // Closing a code tab drops its buffer (auto-saved on edit settle).
        if let Tab::Code(path) = tab {
            self.open_editors.retain(|e| &e.path != path);
        }
        if matches!(tab, Tab::Nvim) {
            self.nvim_embed.shutdown();
        }
        egui_dock::tab_viewer::OnCloseResponse::Close
    }

    fn on_tab_button(&mut self, tab: &mut Tab, response: &egui::Response) {
        if self.fit_labels.contains_key(tab) {
            response.clone().on_hover_text(tab_label(tab));
        }
        if self.active_tabs_prev.contains(tab) {
            let r = response.rect;
            // In the tab's own layer, clipped to its visible part, so windows
            // cover it and scrolled-off tabs don't show it.
            response.ctx.layer_painter(response.layer_id).with_clip_rect(response.interact_rect).rect_filled(
                egui::Rect::from_min_max(egui::pos2(r.left(), r.bottom() - 2.0), r.right_bottom()),
                egui::CornerRadius::ZERO,
                crate::theme::accent(),
            );
        }
    }

    /// Tabs that scroll their own panels: a dock scroll bar on top would
    /// cover theirs (the Animator's side panels, the code editor).
    fn scroll_bars(&self, tab: &Tab) -> [bool; 2] {
        let own = matches!(tab, Tab::AnimController | Tab::AnimClip | Tab::Sequencer | Tab::ShaderGraph | Tab::Code(_) | Tab::Nvim);
        [!own, !own]
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Tab) {
        self.active_tabs.insert(tab.clone());
        // Panel padding (spec §2.7) for tabs that don't draw their own header
        // strip or fill edge to edge.
        let mut padded_ui;
        let ui: &mut egui::Ui = if matches!(tab, Tab::Viewport | Tab::Scene | Tab::UiEditor | Tab::AnimController | Tab::AnimClip | Tab::Sequencer | Tab::ShaderGraph | Tab::Files | Tab::Code(_)) {
            ui
        } else {
            let rect = ui.max_rect().shrink2(egui::vec2(crate::theme::PAD, 8.0));
            padded_ui = ui.new_child(egui::UiBuilder::new().max_rect(rect).layout(*ui.layout()));
            &mut padded_ui
        };
        match tab {
            Tab::Viewport => self.viewport_ui(ui),
            Tab::Camera => self.camera_ui(ui),
            Tab::UiEditor => {
                let selected = match self.selection {
                    Selection::Object(i) => Some(*i),
                    _ => None,
                };
                for request in self.ui_editor.ui(ui, self.scene, selected, self.project_root) {
                    use crate::ui_editor::UiEditorRequest;
                    match request {
                        UiEditorRequest::Select(i) => {
                            *self.selection = i.map_or(Selection::None, Selection::Object);
                            self.multi_objects.clear();
                        }
                        UiEditorRequest::Template(t) => self.actions.push(EditorAction::SpawnUiTemplate(t)),
                        UiEditorRequest::Add(p) => self.actions.push(EditorAction::SpawnUi(p)),
                        UiEditorRequest::SetMode(i, m) => self.actions.push(EditorAction::SetCanvasMode(i, m)),
                    }
                }
            }
            Tab::AnimController | Tab::AnimClip => {
                let selected = match self.selection {
                    Selection::Object(i) => Some(*i),
                    _ => None,
                };
                if matches!(tab, Tab::AnimController) {
                    self.anim_ctrl_tab.ui(ui, self.scene, selected, self.project_root, self.playing);
                } else {
                    self.anim_clip_tab.ui(ui, self.scene, selected, self.project_root, self.playing);
                }
            }
            Tab::ShaderGraph => {
                self.shader_graph_tab.ui(ui, self.project_root);
            }
            Tab::Sequencer => {
                let selected = match self.selection {
                    Selection::Object(i) => Some(*i),
                    _ => None,
                };
                self.sequencer_tab.ui(ui, self.scene, selected, self.project_root, self.playing);
            }
            Tab::Environment => {
                // Undo: compare the settings around the tab (RON: the type has
                // no PartialEq); one drag coalesces into one step.
                let before = self.scene.environment.clone();
                let before_text = ron::to_string(&before).unwrap_or_default();
                self.environment_ui(ui);
                if ron::to_string(&self.scene.environment).unwrap_or_default() != before_text {
                    let gesture = ui.input(|i| i.pointer.any_down());
                    self.actions.push(EditorAction::EnvironmentEdited { before: Box::new(before), gesture });
                }
            }
            Tab::Baker => self.baker_ui(ui),
            Tab::Occlusion => self.occlusion_tab(ui),
            Tab::ProjectSettings => {
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| self.project_settings_ui(ui));
            }
            Tab::Scene => {
                let model_roots = model_root_flags(&self.scene.objects);
                let rows: Vec<crate::SceneObjectRow> = self
                    .scene
                    .objects
                    .iter()
                    .enumerate()
                    .map(|(i, o)| crate::SceneObjectRow {
                        name: o.name.clone(),
                        parent: o.parent,
                        kind: if model_roots[i] { crate::theme::ObjectKind::Model } else { object_kind(o) },
                        enabled: o.enabled,
                        scene_hidden: self.scene.scene_view.hidden.contains(&o.id),
                        scene_locked: self.scene.scene_view.locked.contains(&o.id),
                        net_role: o.net_role,
                        net_inherited: if o.net_role.is_both() { inherited_net_role(&self.scene.objects, i) } else { zestengine_core::NetRole::Both },
                        bone: self.scene.attached_bone(i).map(str::to_owned),
                        prefab: {
                            let mut mark = self.prefab.marks.get(i).copied().unwrap_or_default();
                            if let Some(stage) = &self.scene.stage {
                                mark.locked = stage.locked(o.id);
                                mark.omit = mark.locked && stage.mode == zestengine_engine::scene::StageMode::Isolated;
                            }
                            mark
                        },
                    })
                    .collect();
                let mut selected = match self.selection {
                    Selection::Object(i) => Some(*i),
                    _ => None,
                };
                // Sync the panel's multi-selection from the engine's canonical set
                // so highlighting + ctrl/shift behaviour reflect the live state.
                self.scene_panel.set_multi(self.multi_objects.iter().copied());
                // A model root whose meshes are skinned lists the skeleton first (virtual rows).
                let mut armatures: Vec<(usize, (usize, usize))> = Vec::new();
                for (i, o) in self.scene.objects.iter().enumerate() {
                    if let Some(p) = o.parent
                        && model_roots.get(p).copied().unwrap_or(false)
                        && !armatures.iter().any(|(r, _)| *r == p)
                        && let Some(key) = self.scene.skeleton_index_of(i)
                    {
                        armatures.push((p, key));
                    }
                }
                let scene = &*self.scene;
                self.scene_panel.sync_armatures(&armatures, |(skeleton, _)| scene.skeleton_joint_tree(skeleton));
                let response = self.scene_panel.ui(ui, &rows, &mut selected);
                if response.selection_changed {
                    *self.multi_objects = self.scene_panel.multi();
                    *self.selection = match selected {
                        Some(i) => Selection::Object(i),
                        None => Selection::None,
                    };
                }
                for (child, parent) in response.reparent {
                    self.actions.push(EditorAction::SetParent(child, parent));
                }
                for (child, owner, bone) in response.attach_bone {
                    self.actions.push(EditorAction::AttachToBone { child, owner, bone });
                }
                for (child, parent, before) in response.moves {
                    self.actions
                        .push(EditorAction::MoveObject(child, parent, before));
                }
                if let Some(req) = response.view {
                    self.actions.push(EditorAction::SceneView(req));
                }
                if let Some(index) = response.delete {
                    self.actions.push(EditorAction::DeleteObject(index));
                }
                if let Some(members) = response.group {
                    self.actions.push(EditorAction::GroupIntoEmpty(members));
                }
                if let Some(index) = response.add_toggle
                    && index < self.scene.objects.len()
                {
                    self.actions.push(EditorAction::AddToggle(index));
                }
                if let Some((index, cmd)) = response.prefab {
                    self.actions.push(EditorAction::Prefab(prefab_editor::PrefabAction::Menu(index, cmd)));
                }
                if let Some(index) = response.edit_in_zestsurface
                    && index < self.scene.objects.len()
                {
                    self.actions.push(EditorAction::EditInZestSurface(index));
                }
                if let Some((index, name)) = response.rename
                    && index < self.scene.objects.len()
                {
                    // Caught by record_edits (dirty flag + undo) since F2 only
                    // renames the selected object, which is the edit snapshot.
                    self.scene.objects[index].name = name.clone();
                    self.actions.push(EditorAction::RenameObject(index, name));
                }
                if let Some((path, row)) = response.asset_drop {
                    self.actions.push(EditorAction::PlacementDrop(path, row));
                }
                if let Some(kind) = response.spawn {
                    use zestengine_assets::{ObjectSource, PrimitiveShape};
                    use crate::SpawnKind;
                    let source = match kind {
                        SpawnKind::Empty => ObjectSource::Empty,
                        SpawnKind::Camera => ObjectSource::Camera,
                        SpawnKind::Light(light_kind) => {
                            self.actions.push(EditorAction::SpawnLight(light_kind));
                            return;
                        }
                        SpawnKind::AreaLight => {
                            self.actions.push(EditorAction::SpawnAreaLight);
                            return;
                        }
                        SpawnKind::LightProbeVolume => {
                            self.actions.push(EditorAction::SpawnProbeVolume);
                            return;
                        }
                        SpawnKind::PostFxVolume => {
                            self.actions.push(EditorAction::SpawnPostFxVolume(false));
                            return;
                        }
                        SpawnKind::GlobalPostFxVolume => {
                            self.actions.push(EditorAction::SpawnPostFxVolume(true));
                            return;
                        }
                        SpawnKind::ReflectionProbe => {
                            self.actions.push(EditorAction::SpawnReflectionProbe);
                            return;
                        }
                        SpawnKind::FluxVolume => {
                            self.actions.push(EditorAction::SpawnFluxVolume);
                            return;
                        }
                        SpawnKind::AudioSource => {
                            self.actions.push(EditorAction::SpawnAudioSource);
                            return;
                        }
                        SpawnKind::AudioVolume => {
                            self.actions.push(EditorAction::SpawnAudioVolume);
                            return;
                        }
                        SpawnKind::BoxCollider => {
                            self.actions.push(EditorAction::SpawnBoxCollider);
                            return;
                        }
                        SpawnKind::SphereCollider => {
                            self.actions.push(EditorAction::SpawnSphereCollider);
                            return;
                        }
                        SpawnKind::CapsuleCollider => {
                            self.actions.push(EditorAction::SpawnCapsuleCollider);
                            return;
                        }
                        SpawnKind::MeshCollider => {
                            self.actions.push(EditorAction::AddMeshCollider);
                            return;
                        }
                        SpawnKind::Water => {
                            self.actions.push(EditorAction::SpawnWater);
                            return;
                        }
                        SpawnKind::FloatingCube => {
                            self.actions.push(EditorAction::SpawnFloatingCube);
                            return;
                        }
                        SpawnKind::Bucket => {
                            self.actions.push(EditorAction::SpawnBucket);
                            return;
                        }
                        SpawnKind::Waterfall => {
                            self.actions.push(EditorAction::SpawnWaterfall);
                            return;
                        }
                        SpawnKind::Whirlpool => {
                            self.actions.push(EditorAction::SpawnWhirlpool);
                            return;
                        }
                        SpawnKind::WindZone => {
                            self.actions.push(EditorAction::SpawnWindZone);
                            return;
                        }
                        SpawnKind::Cube => ObjectSource::Primitive {
                            shape: PrimitiveShape::Cube,
                        },
                        SpawnKind::Sphere => ObjectSource::Primitive {
                            shape: PrimitiveShape::Sphere,
                        },
                        SpawnKind::Capsule => ObjectSource::Primitive {
                            shape: PrimitiveShape::Capsule,
                        },
                        SpawnKind::Plane => ObjectSource::Primitive {
                            shape: PrimitiveShape::Plane,
                        },
                        SpawnKind::Terrain => {
                            self.actions.push(EditorAction::SpawnTerrain);
                            return;
                        }
                        SpawnKind::Ui(preset) => {
                            self.actions.push(EditorAction::SpawnUi(preset));
                            return;
                        }
                    };
                    self.actions.push(EditorAction::Spawn(source));
                }
            }
            Tab::Nvim => {
                self.nvim_embed.show(ui, self.project_root);
            }
            Tab::Code(path) => {
                if let Some(editor) = self.open_editors.iter_mut().find(|e| &e.path == path) {
                    let response = CodeEditor.ui(
                        ui,
                        path,
                        &mut editor.text,
                        &editor.language,
                        editor.dirty,
                        &editor.diagnostics,
                        false,
                        &mut editor.completion,
                        &mut editor.hover,
                        &mut editor.goto,
                        &mut editor.references,
                        &mut editor.actions,
                        self.code_zoom,
                        self.vim_mode,
                    );
                    if response.text_changed {
                        editor.dirty = true;
                        editor.lsp_dirty = true;
                        editor.last_edit = Some(Instant::now());
                    }
                    if response.save_requested {
                        self.actions
                            .push(EditorAction::SaveOpenEditor(path.clone()));
                    }
                    if response.zoom_changed {
                        self.actions.push(EditorAction::SaveEditorPrefs);
                    }
                    if response.open_external {
                        self.actions.push(EditorAction::OpenInExternalEditor(path.clone()));
                    }
                    if response.build_requested {
                        self.actions.push(EditorAction::ReloadPlugins);
                    }
                    if response.close_requested {
                        self.actions
                            .push(EditorAction::CloseCodeTab(path.clone()));
                    }
                    if let Some(cursor) = response.request_completion {
                        self.actions.push(EditorAction::LspCompletion(
                            path.clone(),
                            cursor,
                            response.completion_explicit,
                        ));
                    }
                    if let Some(cursor) = response.request_hover {
                        self.actions
                            .push(EditorAction::LspHover(path.clone(), cursor));
                    }
                    if let Some(cursor) = response.request_definition {
                        self.actions
                            .push(EditorAction::LspGoto(path.clone(), cursor));
                    }
                    if let Some(cursor) = response.request_references {
                        self.actions
                            .push(EditorAction::LspReferences(path.clone(), cursor));
                    }
                    if let Some(cursor) = response.request_code_action {
                        self.actions
                            .push(EditorAction::LspCodeAction(path.clone(), cursor));
                    }
                    if let Some((target, line, col)) = response.goto_location {
                        self.actions.push(EditorAction::OpenAndGoto(target, line, col));
                    }
                } else {
                    ui.label(format!("No editor buffer for {}", path.display()));
                }
            }
            Tab::Inspector => self.inspector_ui(ui),
            Tab::Files => {
                let selected = match self.selection {
                    Selection::File(path) => Some(path.clone()),
                    _ => None,
                };
                crate::set_sprite_probe(zestengine_assets::sprite::sprite_count);
                crate::set_atlas_probe(crate::sprite_editor::atlas_name_probe);
                crate::sprite_editor::set_atlas_root(self.project_root);
                crate::set_sprite_children_probe(|path, rel| {
                    zestengine_assets::sprite::sprite_children(path, rel)
                        .into_iter()
                        .map(|(id, name, reference)| crate::SpriteChild { id, name, reference })
                        .collect()
                });
                self.file_browser.set_multi(self.multi_files.iter().cloned());
                let response = self
                    .file_browser
                    .ui(ui, selected.as_deref(), self.file_diagnostics);
                if let Some(path) = response.clicked {
                    // Single click just selects (Inspector shows file info). The
                    // browser tracked the ctrl/shift multi-selection internally.
                    *self.multi_files = self.file_browser.multi();
                    self.actions.push(EditorAction::SelectFile(path));
                }
                if let Some(path) = response.activated {
                    // Double click opens by type: models import, code/text open
                    // an editor tab, scenes can be loaded from the Inspector.
                    let ext = path
                        .extension()
                        .and_then(|e| e.to_str())
                        .map(str::to_lowercase);
                    if matches!(ext.as_deref(), Some("gltf" | "glb" | "fbx")) {
                        self.actions.push(EditorAction::ImportModel(path));
                    } else if ext.as_deref().is_some_and(|e| CODE_EXTENSIONS.contains(&e)) {
                        self.actions.push(EditorAction::OpenCodeFile(path));
                    } else if ext.as_deref() == Some(zestengine_assets::rig2d::RIG_EXTENSION) {
                        crate::skinning_editor::request_open(path);
                    } else if ext.as_deref() == Some(zestengine_assets::SHADER_GRAPH_EXTENSION) {
                        let rel = crate::anim_controller_editor::rel_path(self.project_root, &path);
                        self.shader_graph_tab.open(self.project_root, &rel);
                        self.shader_graph_tab.open_request = true;
                    } else {
                        self.actions.push(EditorAction::SelectFile(path));
                    }
                }
                if response.clear_selection {
                    // Clicked empty space in the grid: drop the file selection so
                    // the Inspector clears (matches the scene-tree behaviour).
                    self.multi_files.clear();
                    if matches!(self.selection, Selection::File(_)) {
                        *self.selection = Selection::None;
                    }
                }
                if let Some(dir) = response.create_material_in {
                    self.actions.push(EditorAction::CreateMaterial(dir));
                }
                if let Some(dir) = response.create_scene_in {
                    self.actions.push(EditorAction::CreateScene(dir));
                }
                if let Some(dir) = response.create_shader_in {
                    self.actions.push(EditorAction::CreateShader(dir));
                }
                if let Some(dir) = response.create_anim_controller_in {
                    crate::anim_controller_editor::create_asset(&dir, true, self.project_root, self.anim_ctrl_tab, self.anim_clip_tab);
                }
                if let Some(dir) = response.create_anim_clip_in {
                    crate::anim_controller_editor::create_asset(&dir, false, self.project_root, self.anim_ctrl_tab, self.anim_clip_tab);
                }
                if let Some(dir) = response.create_behaviour_tree_in {
                    create_behaviour_tree(&dir, self.project_root);
                }
                if let Some(dir) = response.create_shader_graph_in {
                    crate::shader_graph_editor::create_asset(&dir, self.project_root, self.shader_graph_tab);
                }
                if let Some(dir) = response.create_waterwaves_in {
                    self.actions.push(EditorAction::CreateWaterWaves(dir));
                }
                if let Some((dir, settings)) = response.create_audio_in {
                    self.actions.push(EditorAction::CreateAudioAsset(dir, settings));
                }
                if let Some(dir) = response.create_postfx_in {
                    self.actions.push(EditorAction::CreatePostFx(dir));
                }
                if let Some(dir) = response.create_physmat_in {
                    self.actions.push(EditorAction::CreatePhysMat(dir));
                }
                if let Some(dir) = response.create_spriteatlas_in {
                    self.actions.push(EditorAction::CreateSpriteAtlas(dir));
                }
                if let Some(dir) = response.create_spriteshape_in {
                    self.actions.push(EditorAction::CreateSpriteShape(dir));
                }
                if let Some(dir) = response.create_folder_in {
                    self.actions.push(EditorAction::CreateFolder(dir));
                }
                if response.create_component {
                    self.actions.push(EditorAction::CreateComponent);
                }
                if let Some(path) = response.delete {
                    self.actions.push(EditorAction::DeleteFile(path));
                }
                if let Some(path) = response.set_skybox {
                    self.actions.push(EditorAction::SetSkybox(path));
                }
                if let Some(path) = response.convert_unity_material {
                    self.actions.push(EditorAction::UnityConvert(crate::unity_convert_ui::UnityJob::Material(path)));
                }
                if let Some(path) = response.convert_unity_folder {
                    self.actions.push(EditorAction::UnityConvert(crate::unity_convert_ui::UnityJob::Folder(path)));
                }
                if let Some(path) = response.import_unity_package {
                    self.actions.push(EditorAction::UnityConvert(crate::unity_convert_ui::UnityJob::Package(path)));
                }
                if let Some(path) = response.create_prefab_variant {
                    self.actions.push(EditorAction::CreatePrefabVariant(path));
                }
                if let Some(path) = response.import_layered {
                    self.actions.push(EditorAction::ImportLayered(path));
                }
                if let Some(path) = response.show_in_file_manager {
                    self.actions.push(EditorAction::ShowInFileManager(path));
                }
                if let Some((path, instance)) = response.layers_prefab {
                    self.actions.push(EditorAction::LayersPrefab(path, instance));
                }
                if let Some(source) = response.import_pack {
                    self.actions.push(EditorAction::ImportPack(source));
                }
                // Keep the selection and any open .material following
                // renamed/moved files.
                for (old, new) in response.moved {
                    // Sidecar ids follow the file, so slots keep it.
                    zestengine_core::asset_ref::moved(&old, &new);
                    for m in self.scene.materials.iter_mut() {
                        if let Some(f) = m.file.as_mut()
                            && let Ok(rest) = f.strip_prefix(&old)
                        {
                            *f = if rest.as_os_str().is_empty() { new.clone() } else { new.join(rest) };
                        }
                    }
                    if *self.selection == Selection::File(old.clone()) {
                        *self.selection = Selection::File(new.clone());
                    }
                    if let Some(fm) = self.file_material.as_mut()
                        && fm.path == old
                    {
                        fm.path = new.clone();
                    }
                    for editor in self.open_editors.iter_mut() {
                        if editor.path == old {
                            editor.path = new.clone();
                        }
                    }
                }
            }
            Tab::Log => self.log_ui(ui),
        }
    }
}

impl EditorTabs<'_> {
    /// Avatar Descriptor "Move to model rig": write its bone table into the
    /// avatar model's `.meta`, clear the table and reimport the model.
    fn move_avatar_overrides(&mut self, index: usize, table: Vec<(String, String)>) {
        let Some((rel, skel)) = crate::rig_editor::avatar_model(self.scene, index).map(|(r, s)| (r, s.clone())) else {
            return;
        };
        let abs = self.project_root.join(rel);
        if self.file_meta.as_ref().is_some_and(|f| f.path == abs && f.dirty) {
            tracing::warn!("{} has unapplied import settings; apply or revert them before moving the avatar overrides", abs.display());
            return;
        }
        match zestengine_engine::humanoid::move_overrides_to_model(&abs, &skel, &table) {
            Ok(n) => {
                tracing::info!("moved {n} bone overrides into {}", abs.display());
                if let Some(fm) = self.file_meta.as_mut().filter(|f| f.path == abs) {
                    fm.revert();
                }
                for c in &mut self.scene.objects[index].components {
                    if let Some(d) = zestengine_core::component_inner_mut(c).as_any_mut().downcast_mut::<zestengine_core::AvatarDescriptor>() {
                        // Entries that moved are gone; ones the model could not take stay.
                        d.bone_map.retain(|(role, joint)| {
                            zestengine_assets::human_rig::HumanBone::from_name(role).is_none() || skel.joints.iter().all(|sj| sj.name != *joint)
                        });
                    }
                }
                self.rig_ui.forget();
                self.actions.push(EditorAction::ReimportModel(abs));
            }
            Err(e) => tracing::error!("moving avatar overrides: {e:#}"),
        }
    }

    /// Project Settings dock tab: name, VR, boot scene, appearance, spatial
    /// audio, physics, external tracking, input and the build target. Edits the
    /// project in place; the caller saves once the dock pass ends.
    fn project_settings_ui(&mut self, ui: &mut egui::Ui) {
        let mut dirty = false;
        egui::Grid::new("project-settings-grid").num_columns(2).show(ui, |ui| {
            ui.label("Name");
            dirty |= ui.text_edit_singleline(&mut self.project.name).changed();
            ui.end_row();

            ui.label("Start in VR")
                .on_hover_text("A built game starts in VR when a headset and OpenXR runtime are present. Off: it always opens a flat window (Meta Quest builds always start in VR). The editor's own VR is switched from View > VR (OpenXR): that setting overrides this one for the editor; until it is set the editor follows this.");
            dirty |= ui.checkbox(&mut self.project.vr, "").changed();
            ui.end_row();

            ui.label("Ambient occlusion quality")
                .on_hover_text("Algorithm of screen-space ambient occlusion. Low = SAO (12 taps, spatial denoise), High = GTAO with a temporal filter (about 0.12 ms at 1440p on an RTX 4080). AO itself is switched on per scene (Environment > Post Processing) or per Volume profile, and only darkens indirect light. Off in VR.");
            egui::ComboBox::from_id_salt("ao-quality")
                .selected_text(self.project.settings.ao_quality.label())
                .show_ui(ui, |ui| {
                    for q in zestengine_render::AoQuality::ALL {
                        dirty |= ui.selectable_value(&mut self.project.settings.ao_quality, q, q.label()).changed();
                    }
                });
            ui.end_row();

            ui.label("Anti-aliasing")
                .on_hover_text("FXAA = cheap post-process (default). TAA = temporal, jittered, sharper and stable in motion (editor viewport and game). FSR 3 = temporal upscaling (game only; the editor viewport uses TAA). MSAA is not available (see Features.md). A camera can override this in its inspector.");
            egui::ComboBox::from_id_salt("anti-aliasing")
                .selected_text(self.project.settings.anti_aliasing.label())
                .show_ui(ui, |ui| {
                    for m in zestengine_render::AntiAliasing::ALL {
                        ui.add_enabled_ui(m.available(), |ui| {
                            dirty |= ui.selectable_value(&mut self.project.settings.anti_aliasing, m, m.label()).changed();
                        });
                    }
                });
            ui.end_row();

            ui.label("TAA / FSR sharpen")
                .on_hover_text("CAS-like sharpening of the TAA / FSR output (0 = off).");
            dirty |= ui.add(egui::Slider::new(&mut self.project.settings.taa_sharpen, 0.0..=1.0)).changed();
            ui.end_row();

            ui.label("Zestlets")
                .on_hover_text("Zestlets (formerly Nanite-lite), experimental (GEO-4). Static meshes of 2000+ triangles draw as GPU-culled zestlets (clusters of triangles) with a 1 px LOD cut. Off by default. ZESTENGINE_ZESTLETS=1/0 (or the older ZESTENGINE_CLUSTERS=1/0) overrides this. Not used in VR yet.");
            egui::ComboBox::from_id_salt("cluster-rendering")
                .selected_text(self.project.settings.cluster_rendering.label())
                .show_ui(ui, |ui| {
                    for m in zestengine_render::ClusterMode::ALL {
                        dirty |= ui.selectable_value(&mut self.project.settings.cluster_rendering, m, m.label()).changed();
                    }
                });
            ui.end_row();

            ui.label("Stream geometry")
                .on_hover_text("GEO-6. Big static Zestlet meshes keep only their coarse levels resident and load detail pages on demand, evicting the least recently used under the budget; a missing page is covered by its coarser parent, never a hole. Needs Zestlets. Applies to models loaded afterwards. Off by default. ZESTENGINE_CLUSTER_STREAM=1/0 overrides this.");
            ui.horizontal(|ui| {
                dirty |= ui.checkbox(&mut self.project.settings.cluster_stream, "").changed();
                ui.add_enabled_ui(self.project.settings.cluster_stream, |ui| {
                    dirty |= ui.add(egui::DragValue::new(&mut self.project.settings.cluster_stream_mb).range(16..=16384).suffix(" MB")).changed();
                });
            });
            ui.end_row();

            ui.label("Starting scene")
                .on_hover_text("The scene a built game loads first (boot scene).");
            // Walking the project every frame stalls the editor, so reuse the
            // last scan for a couple of seconds.
            let scenes_id = egui::Id::new("zestengine-boot-scene-files");
            let cached: Option<(Instant, Vec<String>)> = ui.data(|d| d.get_temp(scenes_id));
            let scenes = match cached {
                Some((at, files)) if at.elapsed().as_secs_f32() < 2.0 => files,
                _ => {
                    let files = scan_project_files(self.project_root, "scene");
                    ui.data_mut(|d| d.insert_temp(scenes_id, (Instant::now(), files.clone())));
                    files
                }
            };
            let current = self.project.boot_scene.clone().unwrap_or_else(|| "(none)".into());
            egui::ComboBox::from_id_salt("boot-scene")
                .selected_text(current)
                .show_ui(ui, |ui| {
                    for rel in &scenes {
                        if ui
                            .selectable_label(self.project.boot_scene.as_deref() == Some(rel.as_str()), rel)
                            .clicked()
                        {
                            self.project.boot_scene = Some(rel.clone());
                            dirty = true;
                        }
                    }
                    if scenes.is_empty() {
                        ui.label(egui::RichText::new("no .scene files").small().weak());
                    }
                });
            ui.end_row();
        });
        ui.add_space(8.0);
        crate::widgets::eyebrow(ui, "Appearance");
        crate::widgets::property_row(ui, "Accent", |ui| {
            let current = crate::theme::accent();
            for (name, c) in crate::theme::ACCENT_PRESETS {
                let (r, resp) = ui.allocate_exact_size(egui::vec2(22.0, 22.0), egui::Sense::click());
                ui.painter().rect_filled(r, egui::CornerRadius::same(crate::theme::R_FIELD), c);
                if c == current {
                    ui.painter().rect_stroke(r.expand(2.0), egui::CornerRadius::same(crate::theme::R_ROW), egui::Stroke::new(1.5_f32, crate::theme::TEXT_STRONG), egui::StrokeKind::Outside);
                }
                if resp.on_hover_text(name).clicked() {
                    self.project.settings.accent = Some([c.r(), c.g(), c.b()]);
                    dirty = true;
                }
            }
            let mut rgb = [current.r(), current.g(), current.b()];
            if ui.color_edit_button_srgb(&mut rgb).on_hover_text("Custom accent").changed() {
                self.project.settings.accent = Some(rgb);
                dirty = true;
            }
        });
        if dirty {
            apply_accent(ui.ctx(), self.project.settings.accent);
        }
        {
            let st = &mut self.project.settings;
            dirty |= crate::selection_outline_settings_ui(ui, &mut st.selection_outline, &mut st.selection_outline_color, &mut st.selection_outline_width);
        }
        ui.add_space(8.0);
        egui::CollapsingHeader::new("Spatial Audio").id_salt("zestengine-spatial-audio").show(ui, |ui| {
            dirty |= crate::components::asset_chip(ui, "project-audio-settings", &mut self.project.audio_settings,
                "Audio settings asset", &[zestengine_assets::AUDIO_SETTINGS_EXTENSION]);
            ui.label("Select the asset to edit output, simulation and bake settings.");
            ui.label("Use Audio Volume components with .audiofx profiles for room acoustics.");
        });
        egui::CollapsingHeader::new("Physics").id_salt("zestengine-physics-settings").show(ui, |ui| {
            let root = self.project_root.to_path_buf();
            dirty |= crate::physics_settings_ui(ui, &mut self.project.physics, || physmat_files(&root));
        });
        egui::CollapsingHeader::new("External tracking").id_salt("zestengine-external-tracking").show(ui, |ui| {
            let ft = &mut self.project.tracking_input;
            crate::widgets::eyebrow(ui, "SlimeVR");
            crate::widgets::property_row(ui, "Connect to server", |ui| dirty |= ui.checkbox(&mut ft.slimevr, "").changed());
            let slime = match self.slimevr {
                Some(s) if s.connected() => "Connected".to_owned(),
                Some(_) => format!("Waiting for the server at {}", zestengine_engine::slimevr::socket_path().display()),
                None => "Off".to_owned(),
            };
            ui.label(egui::RichText::new(slime).small().weak());
            crate::widgets::eyebrow(ui, "OSC (Open Sound Control)");
            crate::widgets::property_row(ui, "Receive", |ui| dirty |= ui.checkbox(&mut ft.osc_enabled, "").changed());
            crate::widgets::property_row(ui, "UDP port", |ui| {
                dirty |= ui.add(egui::DragValue::new(&mut ft.osc_port).range(1024..=65535)).changed();
            });
            crate::widgets::property_row(ui, "Send to UDP port", |ui| {
                dirty |= ui.add(egui::DragValue::new(&mut ft.osc_out_port).range(1024..=65535)).on_hover_text("Avatar parameter changes go to 127.0.0.1 on this port (VRChat tools listen on 9001)").changed();
            });
            crate::widgets::property_row(ui, "Advertise (OSCQuery)", |ui| {
                dirty |= ui.checkbox(&mut ft.osc_query, "").on_hover_text("Let tools find the ports on their own: a local HTTP tree plus an mDNS announcement").changed();
            });
            if ft.osc_query {
                ui.label(egui::RichText::new("OSCQuery runs during Play: a loopback HTTP tree and an mDNS announcement").small().weak());
            }
            let status = match self.face_input.osc_port() {
                Some(p) => format!("Listening on UDP {p}"),
                None => "Not listening (starts with Play when enabled)".to_owned(),
            };
            ui.label(egui::RichText::new(status).small().weak());
            crate::widgets::info_note(
                ui,
                "Only for tools that send over OSC (Open Sound Control): VRCFaceTracking, Babble, Steam Link. Headset face and eye tracking from WiVRn or Virtual Desktop is read directly through OpenXR and needs none of this.",
            );
        });
        egui::CollapsingHeader::new("Network roles").id_salt("zestengine-net-roles").show(ui, |ui| {
            crate::widgets::info_note(
                ui,
                "Which side a component type belongs to, for the whole project: Client-only types are removed from a dedicated server, Server-only types from the players. A role set on a single component in the Inspector wins over this list.",
            );
            let mut changed = false;
            let rows = zestengine_engine::net_roles::type_role_rows(self.registry, &self.project.net_role_overrides);
            let filter_id = egui::Id::new("zestengine-net-role-filter");
            let mut pick: String = ui.data_mut(|d| d.get_temp::<String>(filter_id)).unwrap_or_default();
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(&mut pick).hint_text("filter component types").desired_width(180.0));
            });
            let needle = pick.to_lowercase();
            egui::ScrollArea::vertical().max_height(220.0).id_salt("net-role-types").show(ui, |ui| {
                for (name, default, current) in rows.iter().filter(|(n, _, o)| o.is_some() || (!needle.is_empty() && n.to_lowercase().contains(&needle))) {
                    ui.horizontal(|ui| {
                        ui.label(name);
                        let mut role = current.unwrap_or(*default);
                        let before = role;
                        egui::ComboBox::from_id_salt(("net-role-type", name))
                            .selected_text(role.label())
                            .show_ui(ui, |ui| {
                                for r in zestengine_core::NetRole::ALL {
                                    ui.selectable_value(&mut role, r, r.label());
                                }
                            });
                        if role != before {
                            changed |= zestengine_engine::net_roles::set_type_override(&mut self.project.net_role_overrides, self.registry, name, Some(role));
                        }
                        if current.is_some() && ui.small_button("Reset").on_hover_text(format!("Back to the type's own role ({})", default.label())).clicked() {
                            changed |= zestengine_engine::net_roles::set_type_override(&mut self.project.net_role_overrides, self.registry, name, None);
                        }
                    });
                }
            });
            ui.data_mut(|d| d.insert_temp(filter_id, pick));
            if changed {
                dirty = true;
            }
        });
        egui::CollapsingHeader::new("Input")
            .id_salt("zestengine-input-settings")
            .show(ui, |ui| self.input_settings_ui(ui, "project-settings"));
        ui.separator();
        if let Some(status) = self.build_status {
            ui.label(egui::RichText::new(status).small());
        }
        ui.horizontal(|ui| {
            ui.label("Target");
            egui::ComboBox::from_id_salt("zestengine-build-target")
                .selected_text(self.build_target.label())
                .show_ui(ui, |ui| {
                    for t in bundle::BuildTarget::available() {
                        ui.selectable_value(&mut *self.build_target, t, t.label());
                    }
                });
        });
        if self.build_target.is_android() {
            crate::widgets::info_note(
                ui,
                "Builds a signed APK with tools/android/build_apk.sh. Needs the Android SDK, NDK and build-tools (ANDROID_HOME, ANDROID_NDK_HOME) and `rustup target add aarch64-linux-android`. Meta Quest builds also bundle the Khronos OpenXR loader (OPENXR_LOADER_SO, or fetched once). Release signing: ZESTENGINE_KEYSTORE / ZESTENGINE_KEYSTORE_PASS.",
            );
        }
        if *self.build_target == bundle::BuildTarget::Server {
            crate::widgets::info_note(
                ui,
                "Dedicated server: the zestengine-server binary (your own components linked in), the scenes with Client-only objects and components removed, collision-only models and no textures, audio or shaders; plus server.ron, run.sh and build-report.txt (sizes and warnings about kept content that points at stripped content). ZESTENGINE_SERVER_TARGET=<triple> cross-compiles.",
            );
        }
        if ui
            .add_enabled(!*self.building_game, egui::Button::new(if *self.build_target == bundle::BuildTarget::Server { "Build Server" } else { "Build Game" }))
            .on_hover_text(format!("Build and bundle into {}/", self.build_target.output_hint()))
            .clicked()
        {
            self.actions.push(EditorAction::BuildGame(*self.build_target));
        }
        if dirty {
            *self.project_settings_dirty = true;
        }
    }

    /// Project Settings > Input: the bindings editor plus the Player-remaps
    /// section (shared with the standalone Input Bindings window).
    fn input_settings_ui(&mut self, ui: &mut egui::Ui, id: &str) {
        if bindings_section_ui(ui, id, &mut *self.project, &mut *self.bindings_editor, &mut *self.input) {
            self.input.set_project_bindings(self.project.bindings.clone());
            *self.project_settings_dirty = true;
        }
    }

    /// Scroll dolly of the fly camera, applied inside the egui pass (see the
    /// `camera` field): moves the camera, then refreshes the overlay matrices.
    fn dolly_now(&mut self, amount: f32, rect: egui::Rect) {
        self.camera.dolly(amount);
        self.refresh_camera_matrices(rect);
    }

    /// One frame of left-drag orbit, applied inside the egui pass. The pivot is
    /// locked on the first frame of the drag (recomputing it mid-drag makes the
    /// rotation drift) and released by `orbit_end_now`.
    fn orbit_now(&mut self, dx: f32, dy: f32, rect: egui::Rect) {
        if self.orbit_pivot.is_none() {
            let target = match *self.selection {
                Selection::Object(i) => self.scene.world_transform(i).w_axis.truncate(),
                _ => {
                    // What's at the viewport center: the hit object, or a
                    // point ahead of the camera.
                    cursor_ray_for(self.camera, rect, rect.center())
                        .and_then(|(origin, dir)| self.scene.pick(origin, dir).map(|hit| self.scene.object_bounds(hit).0))
                        .unwrap_or_else(|| self.camera.position + self.camera.forward() * 5.0)
                }
            };
            // Orbit around the point on the CURRENT view ray at the target's
            // depth: the camera is already looking straight at it, so look-at
            // orbit engages with zero snap.
            let forward = self.camera.forward();
            let depth = (target - self.camera.position).dot(forward).max(0.5);
            *self.orbit_pivot = Some(self.camera.position + forward * depth);
        }
        if let Some(pivot) = *self.orbit_pivot {
            self.camera.orbit(pivot, dx, dy);
        }
        self.refresh_camera_matrices(rect);
    }

    /// Left drag ended: release the locked orbit pivot.
    fn orbit_end_now(&mut self) {
        *self.orbit_pivot = None;
    }

    /// An orbit is in progress (locked pivot): viewport widgets / gizmos are
    /// ignored and hidden so a rotate gesture can't grab a handle. Read live, so
    /// the overlays drawn after `orbit_now` / `orbit_end_now` this frame already
    /// reflect the drag's start or end (the frames are then a pure function of
    /// the camera pose and drag state, which the sky probe's mirror pairs check);
    /// the widgets that run before the orbit block see the previous frame's state.
    fn orbiting(&self) -> bool {
        self.orbit_pivot.is_some()
    }

    /// `view` / `proj` from the fly camera as it is now, for every overlay drawn
    /// after a dolly / orbit this frame (the render after the pass reads the same
    /// camera). Not while the Scene view shows the game camera: the matrices are
    /// the game's then and the fly camera is off screen.
    fn refresh_camera_matrices(&mut self, rect: egui::Rect) {
        if self.scene_shows_game || !rect.is_finite() || rect.height() < 1.0 {
            return;
        }
        let aspect = rect.width() / rect.height();
        self.view = self.camera.view();
        self.proj = self.camera.proj(aspect);
    }

    /// Screen position of a world point (full-window NDC mapping, matching the
    /// scene swapchain). `None` if behind the camera.
    fn world_to_screen(&self, world: Vec3, full_rect: egui::Rect) -> Option<egui::Pos2> {
        let clip = self.proj * self.view * world.extend(1.0);
        if clip.w <= 0.001 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        Some(egui::pos2(
            full_rect.left() + (ndc.x * 0.5 + 0.5) * full_rect.width(),
            full_rect.top() + (1.0 - (ndc.y * 0.5 + 0.5)) * full_rect.height(),
        ))
    }

    /// Radial direction (camera-right, world space) used to place audio range
    /// handles so they always face the viewer.
    fn camera_right(&self) -> Vec3 {
        self.view.inverse().x_axis.truncate().normalize_or(Vec3::X)
    }

    fn camera_forward(&self) -> Vec3 {
        (-self.view.inverse().z_axis.truncate()).normalize_or(Vec3::NEG_Z)
    }

    /// Unified component-gizmo interaction: grab a box-face or range-radius
    /// handle (from any component's `gizmos()`), suppress orbit for the gesture,
    /// and write the resize back via `apply_gizmo_edit`. Replaces the per-type
    /// probe/collider/audio handlers; FluxVolume/ReflectionProbe and plugins get
    /// working resize handles for free.
    fn gizmo_interaction(
        &mut self,
        response: &egui::Response,
        cursor: Option<egui::Pos2>,
        alt: bool,
    ) {
        let full_rect = *self.viewport_rect;

        // Continue / end an in-progress drag.
        if let Some(drag) = self.handle_drag.as_ref() {
            if !response.dragged_by(egui::PointerButton::Primary) {
                *self.handle_drag = None;
                return;
            }
            let Some(cur) = cursor else { return };
            let delta = cur - drag.start_cursor;
            let (object, component, gizmo) = (drag.object, drag.component, drag.gizmo);
            match &drag.kind {
                GizmoDragKind::BoxFace {
                    axis,
                    sign,
                    start_size,
                    start_center,
                    start_origin_world,
                    start_world,
                    scale_a,
                    screen_axis,
                    ..
                } => {
                    let (axis, sign) = (*axis, *sign);
                    let (start_size, start_center, start_origin_world, start_world, scale_a) =
                        (*start_size, *start_center, *start_origin_world, *start_world, *scale_a);
                    let len2 = screen_axis.length_sq();
                    let meters_world = if len2 > 1.0e-6 { delta.dot(*screen_axis) / len2 } else { 0.0 };
                    let local_delta = if scale_a.abs() > 1.0e-5 { meters_world / scale_a } else { 0.0 };
                    // Alt: both sides move about the centre (the grabbed face
                    // still follows the cursor). Otherwise only the grabbed
                    // face moves and the opposite one stays put.
                    let both = response.ctx.input(|i| i.modifiers.alt);
                    let mut axv = Vec3::ZERO;
                    axv[axis] = 1.0;
                    let mut new_size = start_size;
                    let new_center = if both {
                        new_size[axis] = (start_size[axis] + 2.0 * local_delta).max(0.1);
                        start_center
                    } else {
                        new_size[axis] = (start_size[axis] + local_delta).max(0.1);
                        start_center + axv * (sign * (new_size[axis] - start_size[axis]) * 0.5)
                    };
                    if let Some(c) = self.scene.objects[object].components.get_mut(component) {
                        self.editor_components.apply_gizmo_edit(
                            c.as_mut(),
                            crate::GizmoEdit::Box { index: gizmo, half_extents: new_size * 0.5, center: new_center },
                        );
                    }
                    // Where the component now puts its box (some keep it on the
                    // object's origin): move the object so the opposite face
                    // ends up where it started.
                    let mut shift = Vec3::ZERO;
                    if !both
                        && let Some(c) = self.scene.objects[object].components.get(component)
                        && let Some(crate::GizmoSpec::Box { center, half_extents, .. }) =
                            self.editor_components.gizmos(c.as_ref()).into_iter().nth(gizmo)
                    {
                        let opp_before = start_world.transform_point3(start_center - axv * (sign * start_size[axis] * 0.5));
                        let opp_after = start_world.transform_point3(center - axv * (sign * half_extents[axis]));
                        shift = opp_before - opp_after;
                    }
                    let parent_world =
                        self.scene.objects[object].parent.map_or(glam::Mat4::IDENTITY, |p| self.scene.world_transform(p));
                    self.scene.objects[object].translation = parent_world.inverse().transform_point3(start_origin_world + shift);
                }
                GizmoDragKind::Handle { start_local, right, up, screen_right, screen_up, inv_world } => {
                    // Solve delta = a * screen_right + b * screen_up for the
                    // world-meter amounts along camera right / up.
                    let det = screen_right.x * screen_up.y - screen_right.y * screen_up.x;
                    if det.abs() > 1.0e-6 {
                        let a = (delta.x * screen_up.y - delta.y * screen_up.x) / det;
                        let b = (screen_right.x * delta.y - screen_right.y * delta.x) / det;
                        let world_delta = *right * a + *up * b;
                        let position = *start_local + inv_world.transform_vector3(world_delta);
                        if let Some(c) = self.scene.objects[object].components.get_mut(component) {
                            self.editor_components.apply_gizmo_edit(
                                c.as_mut(),
                                crate::GizmoEdit::Handle { index: gizmo, position },
                            );
                        }
                    }
                }
                GizmoDragKind::Range {
                    start_radius,
                    screen_axis,
                } => {
                    let len2 = screen_axis.length_sq();
                    let meters_world = if len2 > 1.0e-6 {
                        delta.dot(*screen_axis) / len2
                    } else {
                        0.0
                    };
                    let new_radius = (start_radius + meters_world).max(0.0);
                    if let Some(c) = self.scene.objects[object].components.get_mut(component) {
                        self.editor_components.apply_gizmo_edit(
                            c.as_mut(),
                            crate::GizmoEdit::Range {
                                index: gizmo,
                                radius: new_radius,
                            },
                        );
                    }
                }
            }
            return;
        }

        // Maybe start a drag: a primary press on a handle (gizmo wins over orbit;
        // the transform gizmo still wins over us). Painting cloth pins owns drags.
        if !response.drag_started_by(egui::PointerButton::Primary) || alt || crate::cloth_paint::active(&response.ctx) || crate::terrain_paint::active(&response.ctx) || crate::tile_edit::active(&response.ctx) || crate::shape_edit::active(&response.ctx) {
            return;
        }
        let Selection::Object(i) = *self.selection else {
            return;
        };
        if !self.scene.is_active(i) {
            return;
        }
        let Some(press) = cursor else { return };
        if self.gizmo.pick_preview(press) && !crate::river_edit::building(&response.ctx) {
            return; // the move/rotate/scale gizmo owns this press
        }
        let full_rect2 = full_rect;
        let world = self.scene.world_transform(i);
        let (w_scale, w_rot, w_trans) = world.to_scale_rotation_translation();
        let axes = [Vec3::X, Vec3::Y, Vec3::Z];
        let mut best: Option<(f32, GizmoDrag)> = None;
        for (ci, comp) in self.scene.objects[i].components.iter().enumerate() {
            for (gi, spec) in self.editor_components.gizmos(comp.as_ref()).into_iter().enumerate() {
                match spec {
                    crate::GizmoSpec::Box { center, half_extents, .. } => {
                        for axis in 0..3 {
                            for sign in [-1.0f32, 1.0] {
                                let face_local = center + axes[axis] * (sign * half_extents[axis]);
                                let face_world = world.transform_point3(face_local);
                                let Some(fs) = self.world_to_screen(face_world, full_rect2) else {
                                    continue;
                                };
                                let world_axis = (w_rot * axes[axis]) * sign;
                                let Some(plus) =
                                    self.world_to_screen(face_world + world_axis, full_rect2)
                                else {
                                    continue;
                                };
                                let dist = handle_pick_dist(press, fs, plus - fs, 26.0);
                                if dist > 12.0 {
                                    continue;
                                }
                                if best.as_ref().is_none_or(|(d, _)| dist < *d) {
                                    best = Some((
                                        dist,
                                        GizmoDrag {
                                            object: i,
                                            component: ci,
                                            gizmo: gi,
                                            start_cursor: press,
                                            kind: GizmoDragKind::BoxFace {
                                                axis,
                                                sign,
                                                start_size: half_extents * 2.0,
                                                start_center: center,
                                                start_origin_world: w_trans,
                                                start_world: world,
                                                scale_a: w_scale[axis],
                                                screen_axis: plus - fs,
                                            },
                                        },
                                    ));
                                }
                            }
                        }
                    }
                    crate::GizmoSpec::Range { center, radius, .. } => {
                        let origin = world.transform_point3(center);
                        let right = self.camera_right();
                        let handle = origin + right * radius;
                        let Some(hs) = self.world_to_screen(handle, full_rect2) else {
                            continue;
                        };
                        let outward = self
                            .world_to_screen(handle + right, full_rect2)
                            .map(|p| p - hs)
                            .unwrap_or(egui::vec2(1.0, 0.0));
                        let dist = handle_pick_dist(press, hs, outward, 26.0);
                        if dist > 12.0 {
                            continue;
                        }
                        if best.as_ref().is_none_or(|(d, _)| dist < *d) {
                            best = Some((
                                dist,
                                GizmoDrag {
                                    object: i,
                                    component: ci,
                                    gizmo: gi,
                                    start_cursor: press,
                                    kind: GizmoDragKind::Range {
                                        start_radius: radius,
                                        screen_axis: outward,
                                    },
                                },
                            ));
                        }
                    }
                    crate::GizmoSpec::Handle { position, .. } => {
                        let at = world.transform_point3(position);
                        let Some(hs) = self.world_to_screen(at, full_rect2) else {
                            continue;
                        };
                        let dist = press.distance(hs);
                        if dist > 12.0 {
                            continue;
                        }
                        let right = self.camera_right();
                        let up = right.cross(self.camera_forward()).normalize_or(Vec3::Y);
                        let (Some(r), Some(u)) =
                            (self.world_to_screen(at + right, full_rect2), self.world_to_screen(at + up, full_rect2))
                        else {
                            continue;
                        };
                        // Handles win ties against box faces they sit on.
                        let dist = dist - 4.0;
                        if best.as_ref().is_none_or(|(d, _)| dist < *d) {
                            best = Some((
                                dist,
                                GizmoDrag {
                                    object: i,
                                    component: ci,
                                    gizmo: gi,
                                    start_cursor: press,
                                    kind: GizmoDragKind::Handle {
                                        start_local: position,
                                        right,
                                        up,
                                        screen_right: r - hs,
                                        screen_up: u - hs,
                                        inv_world: world.inverse(),
                                    },
                                },
                            ));
                        }
                    }
                    crate::GizmoSpec::Points { .. } | crate::GizmoSpec::Lines { .. } => {}
                }
            }
        }
        if let Some((_, drag)) = best {
            *self.handle_drag = Some(drag);
        }
    }

    /// The Log tab: a filterable console mirroring every tracing event (engine
    /// logs + routed compiler/shader errors).
    fn log_ui(&mut self, ui: &mut egui::Ui) {
        use crate::{icons::Icon, theme, widgets};
        let f = &mut *self.log_filter;
        let (total, warns, errors) = {
            let ring = log_capture::store().lock().unwrap();
            (ring.entries.len(), ring.warns, ring.errors)
        };
        let mut copy_shown = false;
        ui.horizontal(|ui| {
            ui.set_min_height(theme::H_TOOLBAR);
            let all = format!("All {total}");
            let w = format!("Warnings {warns}");
            let e = format!("Errors {errors}");
            widgets::segmented(ui, &mut f.mode, &[(LogMode::All, &all), (LogMode::Warnings, &w), (LogMode::Errors, &e)]);
            if f.mode == LogMode::All {
                ui.menu_button(egui::RichText::new("Levels").color(theme::TEXT_MUTED), |ui| {
                    crate::widgets::menu_frame(ui);
                    ui.checkbox(&mut f.error, egui::RichText::new("Error").color(LOG_ERROR));
                    ui.checkbox(&mut f.warn, egui::RichText::new("Warn").color(LOG_WARN));
                    ui.checkbox(&mut f.info, egui::RichText::new("Info").color(LOG_INFO));
                    ui.checkbox(&mut f.debug, egui::RichText::new("Debug").color(LOG_DEBUG));
                    ui.checkbox(&mut f.trace, egui::RichText::new("Trace").color(LOG_TRACE));
                });
            }
            widgets::search_field(ui, &mut f.search, "Filter log", 180.0);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if widgets::secondary_button_sized(ui, Some(Icon::Trash), "Clear", theme::H_BUTTON_SMALL).clicked() {
                    log_capture::store().lock().unwrap().clear();
                }
                copy_shown = widgets::secondary_button_sized(ui, None, "Copy", theme::H_BUTTON_SMALL)
                    .on_hover_text("Copy the lines shown (after the filters) to the clipboard")
                    .clicked();
                ui.checkbox(&mut f.wrap, "Wrap").on_hover_text("Wrap long lines instead of scrolling horizontally");
                ui.checkbox(&mut f.autoscroll, "Follow");
            });
        });
        ui.add_space(4.0);

        let wrap = f.wrap;
        // Sync the filtered view once per frame: a no-op when nothing changed,
        // only the new entries when the ring just grew (or wrapped), and a
        // full re-filter only when the filter itself changed.
        log_capture::poll();
        {
            let levels = f.level_mask();
            let ring = log_capture::store().lock().unwrap();
            f.cache.sync(&ring, levels, &f.search);
        }
        // Text of every shown entry (copy actions only; not built per frame).
        let shown_text = |cache: &LogCache| -> String {
            let ring = log_capture::store().lock().unwrap();
            let mut out = String::new();
            for it in &cache.items {
                if let Some(e) = ring.get(it.seq) {
                    let short = e.target.rsplit("::").next().unwrap_or(e.target);
                    let times = if e.repeat > 1 { format!(" (×{})", e.repeat) } else { String::new() };
                    out.push_str(&format!("{} {} {} {}{times}\n", e.time, e.level, short, e.message));
                }
            }
            out
        };
        if copy_shown {
            ui.ctx().copy_text(shown_text(&f.cache));
        }
        if f.cache.items.is_empty() {
            let msg = match f.mode {
                LogMode::Errors => "No errors since the scene loaded.",
                LogMode::Warnings => "No warnings since the scene loaded.",
                LogMode::All => "Nothing logged yet.",
            };
            ui.add_space(12.0);
            ui.label(egui::RichText::new(msg).color(theme::TEXT_DIM));
            return;
        }
        let level_style = |l: tracing::Level| match l {
            tracing::Level::ERROR => (LOG_ERROR, "ERROR"),
            tracing::Level::WARN => (LOG_WARN, "WARN"),
            tracing::Level::INFO => (LOG_INFO, "INFO"),
            tracing::Level::DEBUG => (LOG_DEBUG, "DEBUG"),
            tracing::Level::TRACE => (LOG_TRACE, "TRACE"),
        };
        let mono = theme::mono(theme::SIZE_META);
        // The line under the pointer this frame, for the right-click menu.
        let hovered_line = std::cell::RefCell::new(None::<String>);
        let geos = std::cell::RefCell::new(Vec::<LogRowGeo>::new());
        let sel_norm = f.sel.map(|(a, b)| crate::log_select::normalise(a, b));
        #[allow(clippy::too_many_arguments)]
        let row = |ui: &mut egui::Ui, level: tracing::Level, time: &str, source: &str, message: &str, repeat: u32, wrap: bool, seq: u64, line: u32| {
            let mut msg_geo = None::<(std::sync::Arc<egui::Galley>, egui::Pos2)>;
            let (color, tag) = level_style(level);
            let fill = if level == tracing::Level::WARN { theme::WARN_ROW_TINT } else { egui::Color32::TRANSPARENT };
            let framed = egui::Frame::new()
                .fill(fill)
                .corner_radius(egui::CornerRadius::same(theme::R_FIELD))
                .inner_margin(egui::Margin::symmetric(8, 3))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.horizontal_top(|ui| {
                        ui.spacing_mut().item_spacing.x = 10.0;
                        ui.add(egui::Label::new(egui::RichText::new(time).font(mono.clone()).color(theme::TEXT_DIM)).selectable(false));
                        ui.add_sized([40.0, 14.0], egui::Label::new(egui::RichText::new(tag).font(theme::mono_medium(theme::SIZE_META)).color(color)));
                        ui.add_sized(
                            [56.0, 14.0],
                            egui::Label::new(egui::RichText::new(source).font(mono.clone()).color(theme::TEXT_MUTED)).truncate(),
                        );
                        // Consecutive identical messages are one row with a count.
                        if repeat > 1 {
                            ui.add(
                                egui::Label::new(egui::RichText::new(format!("×{repeat}")).font(theme::mono_medium(theme::SIZE_META)).color(color))
                                    .selectable(false),
                            )
                            .on_hover_text(format!("Logged {repeat} times in a row"));
                        }
                        // Drawn by hand (hover-only) so the Log can run its own
                        // selection across virtualised rows and keep the scroll
                        // area scrollable during a drag.
                        let text = if message.contains('\r') { message.replace('\r', "") } else { message.to_owned() };
                        let wrap_w = if wrap { ui.available_width() } else { f32::INFINITY };
                        let galley = ui.painter().layout(text, mono.clone(), theme::TEXT_BODY, wrap_w);
                        let (rect, _) = ui.allocate_exact_size(galley.size(), egui::Sense::hover());
                        if let Some(sel) = sel_norm
                            && let Some((s, e)) = crate::log_select::row_range(sel, seq, if wrap { u32::MAX } else { line }, galley.text())
                        {
                            let hl = ui.visuals().selection.bg_fill;
                            let mut off = 0usize;
                            for pr in &galley.rows {
                                let rs = off;
                                let re = rs + pr.row.char_count_excluding_newline();
                                off += pr.row.char_count_including_newline();
                                let (a, b) = (s.max(rs), e.min(re));
                                let passes_break = e > re && s <= re && off > re;
                                if a < b || passes_break {
                                    let xa = galley.pos_from_cursor(egui::text::CCursor::new(a.min(b))).min.x;
                                    let xb = galley.pos_from_cursor(egui::text::CCursor::new(b.max(a))).min.x + if passes_break { 4.0 } else { 0.0 };
                                    let r = egui::Rect::from_min_max(
                                        rect.min + egui::vec2(xa, pr.min_y()),
                                        rect.min + egui::vec2(xb, pr.max_y()),
                                    );
                                    ui.painter().rect_filled(r, 0.0, hl);
                                }
                            }
                        }
                        ui.painter().galley(rect.min, galley.clone(), theme::TEXT_BODY);
                        msg_geo = Some((galley, rect.min));
                    });
                });
            if let Some((galley, pos)) = msg_geo {
                geos.borrow_mut().push(LogRowGeo { seq, line: if wrap { u32::MAX } else { line }, galley, pos, band: framed.response.rect });
            }
            if ui.rect_contains_pointer(framed.response.rect) {
                let times = if repeat > 1 { format!(" (×{repeat})") } else { String::new() };
                *hovered_line.borrow_mut() = Some(format!("{time} {level} {source} {message}{times}").trim().to_owned());
            }
        };
        let menu_id = egui::Id::new("zestengine-log-menu");
        // Selection drag (hit-tested against last frame's rows, which is what
        // the user saw): press inside starts it, the head follows the pointer,
        // and the pointer above/below the area scrolls it continuously. Plain
        // drag-to-scroll is off below so a drag never pans the content, and
        // the rows are hover-only so the wheel is never blocked by a drag.
        let ctx = ui.ctx().clone();
        let (down, pressed, ptr, dt, wheel_y, shift, copy_ev) = ui.input(|i| {
            (
                i.pointer.primary_down(),
                i.pointer.primary_pressed(),
                i.pointer.latest_pos(),
                i.stable_dt.min(0.1),
                i.smooth_scroll_delta.y,
                i.modifiers.shift,
                i.events.iter().any(|e| matches!(e, egui::Event::Copy)),
            )
        });
        let menu_open = ctx.data(|d| d.get_temp::<(egui::Pos2, Option<String>)>(menu_id)).is_some();
        let inside = ptr.is_some_and(|p| f.area.contains(p));
        if pressed && !menu_open && ptr.is_some() {
            f.sel_drag = false;
            if inside && let Some(h) = ptr.and_then(|p| log_hit(&f.geos, p)) {
                let anchor = if shift { f.sel.map(|s| s.0).unwrap_or(h) } else { h };
                f.sel = Some((anchor, h));
                f.sel_drag = true;
            } else if !inside {
                f.sel = None;
            }
        }
        let mut scroll_dy = 0.0f32;
        if f.sel_drag {
            if !down {
                f.sel_drag = false;
                if f.sel.is_some_and(|(a, b)| a == b) {
                    f.sel = None;
                }
            } else if let Some(p) = ptr {
                if let Some(h) = log_hit(&f.geos, p)
                    && let Some(s) = f.sel.as_mut()
                {
                    s.1 = h;
                }
                let v = crate::log_select::edge_scroll_speed(p.y, f.area.top(), f.area.bottom(), 20.0, 1200.0);
                if v != 0.0 {
                    scroll_dy += v * dt;
                    ctx.request_repaint();
                }
                if !inside {
                    // egui only scrolls under the pointer; forward the wheel.
                    let w = ctx.input_mut(|i| std::mem::take(&mut i.smooth_scroll_delta.y));
                    scroll_dy -= w;
                }
                // Selecting upward must not snap back to the newest entry.
                if (scroll_dy < 0.0 || wheel_y > 0.0) && f.autoscroll {
                    f.autoscroll = false;
                    ctx.request_repaint();
                }
            }
        }
        if inside || f.sel_drag {
            ctx.set_cursor_icon(egui::CursorIcon::Text);
        }
        let sel_text = |cache: &LogCache, sel: (crate::log_select::LogPos, crate::log_select::LogPos)| -> String {
            let ring = log_capture::store().lock().unwrap();
            let msgs = cache.items.iter().filter_map(|it| Some((it.seq, ring.get(it.seq)?.message.as_str())));
            crate::log_select::extract(sel.0, sel.1, msgs)
        };
        if copy_ev
            && !ctx.wants_keyboard_input()
            && let Some(sel) = f.sel.filter(|(a, b)| a != b)
        {
            ctx.copy_text(sel_text(&f.cache, sel));
        }
        let mut scroll = egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .scroll_source(egui::scroll_area::ScrollSource { drag: false, ..egui::scroll_area::ScrollSource::ALL })
            .stick_to_bottom(f.autoscroll);
        if scroll_dy != 0.0 {
            scroll = scroll.vertical_scroll_offset((f.scroll_y + scroll_dy).max(0.0));
        }
        // Clone just the visible entries out of the ring (short lock; layout
        // happens after it is released).
        type Fetched = (tracing::Level, String, String, String, u32, u64, u32);
        let fetch = |seqs: &mut dyn Iterator<Item = (u64, u32)>| -> Vec<Fetched> {
            let ring = log_capture::store().lock().unwrap();
            seqs.filter_map(|(seq, line)| {
                let e = ring.get(seq)?;
                let short = e.target.rsplit("::").next().unwrap_or(e.target).to_owned();
                // line == u32::MAX: the whole message (wrapped mode).
                if line == u32::MAX {
                    return Some((e.level, e.time.to_string(), short, e.message.clone(), e.repeat, seq, line));
                }
                let text = e.message.lines().nth(line as usize).unwrap_or("").to_owned();
                if line == 0 {
                    Some((e.level, e.time.to_string(), short, text, e.repeat, seq, line))
                } else {
                    Some((e.level, String::new(), String::new(), text, 1, seq, line))
                }
            })
            .collect()
        };
        let out = if wrap {
            // Virtualised with estimated heights: only entries intersecting the
            // viewport are laid out, so cost doesn't grow with the log length.
            let line_h = ui.text_style_height(&egui::TextStyle::Monospace);
            let char_w = ui.fonts_mut(|fo| fo.glyph_width(&mono, 'M')).max(1.0);
            let msg_w = (ui.available_width() - 8.0 * 2.0 - 40.0 - 56.0 - 10.0 * 3.0 - 70.0 - 14.0).max(char_w * 8.0);
            let cpl = (msg_w / char_w).floor().max(1.0) as u32;
            let spacing = 1.0;
            let heights: Vec<f32> = f
                .cache
                .items
                .iter()
                .map(|it| {
                    let n: u32 = it.line_lens.iter().map(|&l| l.div_ceil(cpl).max(1)).sum();
                    n as f32 * line_h + 6.0 + spacing
                })
                .collect();
            let total: f32 = heights.iter().sum();
            let items = &f.cache.items;
            scroll.show_viewport(ui, |ui, viewport| {
                ui.style_mut().interaction.selectable_labels = true;
                ui.spacing_mut().item_spacing.y = 0.0;
                let width = ui.available_width();
                let (rect, _) = ui.allocate_exact_size(egui::vec2(width, total), egui::Sense::hover());
                let mut y = 0.0;
                let mut visible: Vec<(usize, f32)> = Vec::new();
                for (i, h) in heights.iter().enumerate() {
                    if y + h >= viewport.min.y && y <= viewport.max.y {
                        visible.push((i, y));
                    } else if y > viewport.max.y {
                        break;
                    }
                    y += h;
                }
                let rows = fetch(&mut visible.iter().map(|&(i, _)| (items[i].seq, u32::MAX)));
                for (&(i, y), (level, time, source, message, repeat, seq, _)) in visible.iter().zip(&rows) {
                    let h = heights[i];
                    let r = egui::Rect::from_min_size(rect.min + egui::vec2(0.0, y), egui::vec2(width, h));
                    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(r));
                    row(&mut child, *level, time, source, message, *repeat, true, *seq, u32::MAX);
                }
            })
        } else {
            // One physical line per row so the list can be virtualized.
            let row_h = ui.text_style_height(&egui::TextStyle::Monospace) + 7.0;
            let cache = &f.cache;
            scroll.show_rows(ui, row_h, cache.lines.len(), |ui, range| {
                ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                ui.style_mut().interaction.selectable_labels = true;
                ui.spacing_mut().item_spacing.y = 0.0;
                let rows = fetch(&mut cache.lines[range].iter().map(|&(i, l)| (cache.items[i as usize].seq, l)));
                for (level, time, source, text, repeat, seq, line) in &rows {
                    row(ui, *level, time, source, text, *repeat, false, *seq, *line);
                }
            })
        };

        // Right-click menu. egui's own text selection clears on any press, so
        // this is drawn by hand: copy the line under the pointer, everything
        // shown, or clear the log.
        f.geos = geos.take();
        f.area = out.inner_rect;
        f.scroll_y = out.state.offset.y;
        let (right_clicked, pointer) =
            ui.input(|i| (i.pointer.button_clicked(egui::PointerButton::Secondary), i.pointer.interact_pos()));
        if right_clicked && let Some(p) = pointer.filter(|p| out.inner_rect.contains(*p)) {
            ctx.data_mut(|d| d.insert_temp(menu_id, (p, hovered_line.borrow().clone())));
        }
        if let Some((pos, line)) = ctx.data(|d| d.get_temp::<(egui::Pos2, Option<String>)>(menu_id)) {
            let mut close = false;
            let area = egui::Area::new(menu_id.with("area")).order(egui::Order::Foreground).fixed_pos(pos).show(&ctx, |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    widgets::menu_frame(ui);
                    if let Some(line) = &line
                        && ui.button("Copy line").clicked()
                    {
                        ctx.copy_text(line.clone());
                        close = true;
                    }
                    if let Some(sel) = f.sel.filter(|(a, b)| a != b)
                        && ui.button("Copy selection").clicked()
                    {
                        ctx.copy_text(sel_text(&f.cache, sel));
                        close = true;
                    }
                    if ui.button("Copy all shown").clicked() {
                        ctx.copy_text(shown_text(&f.cache));
                        close = true;
                    }
                    if ui.button("Clear log").clicked() {
                        log_capture::store().lock().unwrap().clear();
                        close = true;
                    }
                });
            });
            let pressed_outside = ctx.input(|i| i.pointer.any_pressed())
                && ctx.input(|i| i.pointer.interact_pos()).is_some_and(|p| !area.response.rect.contains(p));
            if close || pressed_outside || ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
                ctx.data_mut(|d| d.remove::<(egui::Pos2, Option<String>)>(menu_id));
            }
        }
    }

    /// The Camera tab: shows the main camera's offscreen preview, scaled to
    /// fit while preserving the target's aspect.
    fn camera_ui(&mut self, ui: &mut egui::Ui) {
        // The Camera tab is on screen this frame, so the preview is worth
        // rendering (the gate read after the dock pass in `redraw`).
        *self.camera_visible = true;
        ui.ctx().data_mut(|d| d.insert_temp(egui::Id::new(VIEW_LAYER_KEY), ui.layer_id()));
        let has_camera = self.scene.main_camera().is_some();
        match self.camera_preview {
            Some((texture, size)) if has_camera => {
                let avail = ui.available_size();
                let aspect = size[0] / size[1].max(1.0);
                let mut w = avail.x;
                let mut h = w / aspect;
                if h > avail.y {
                    h = avail.y;
                    w = h * aspect;
                }
                let image = ui.centered_and_justified(|ui| {
                    ui.add(
                        egui::Image::new(egui::load::SizedTexture::new(texture, egui::vec2(w, h)))
                            .maintain_aspect_ratio(true),
                    )
                });
                *self.camera_rect = egui::Rect::from_center_size(image.inner.rect.center(), egui::vec2(w, h));
            }
            _ => {
                ui.centered_and_justified(|ui| {
                    let msg = if has_camera {
                        "Rendering main camera…"
                    } else {
                        "No camera in the scene.\nAdd one from the Scene tree -> Camera."
                    };
                    ui.label(egui::RichText::new(msg).weak());
                });
            }
        }
    }

    /// Occlusion tab: bake settings, Bake / Clear, and the visualization
    /// toggles (cull the Scene view, preview from the game camera, cell grid).
    fn occlusion_tab(&mut self, ui: &mut egui::Ui) {
        use crate::{icons::Icon, theme, widgets};
        use egui::RichText;
        ui.label(
            RichText::new("Baked visibility per view cell: static objects hidden behind static geometry skip rendering")
                .size(theme::SIZE_META)
                .color(theme::TEXT_MUTED),
        );
        ui.add_space(8.0);
        let occ = &mut *self.occlusion_ui;
        widgets::eyebrow(ui, "Bake");
        widgets::property_row(ui, "Cell size", |ui| {
            ui.add(egui::DragValue::new(&mut occ.settings.cell_size).speed(0.1).range(0.5..=64.0).suffix(" m"))
                .on_hover_text("View-cell edge length. Smaller cells cull more precisely but take longer to bake and use more memory.");
        });
        widgets::property_row(ui, "Quality", |ui| {
            widgets::segmented(ui, &mut occ.settings.rays, &[(256, "Low"), (1024, "Medium"), (4096, "High")])
                .on_hover_text("Rays per sample point. Higher finds small or distant gaps more reliably.");
        });
        let statics = self.scene.objects.iter().filter(|o| o.static_geometry && o.render.is_some()).count();
        widgets::info_note(ui, &format!("{statics} static objects occlude and can be culled. Dynamic objects are never occlusion culled."));
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            if widgets::primary_button(ui, Some(Icon::Grid), "Bake occlusion").clicked() {
                self.actions.push(EditorAction::BakeOcclusion);
            }
            if ui
                .add_enabled_ui(self.scene.occlusion.is_some(), |ui| widgets::secondary_button(ui, Some(Icon::Trash), "Clear"))
                .inner
                .clicked()
            {
                self.actions.push(EditorAction::ClearOcclusion);
            }
        });
        widgets::section_gap(ui);

        widgets::eyebrow(ui, "Baked data");
        match &self.scene.occlusion {
            None => {
                ui.label(RichText::new("Not baked").color(theme::TEXT_DIM));
            }
            Some(o) => {
                let mono = theme::mono(theme::SIZE_META);
                widgets::property_row(ui, "Cells", |ui| {
                    ui.label(RichText::new(format!("{} ({} x {} x {})", o.cell_count(), o.dims[0], o.dims[1], o.dims[2])).font(mono.clone()));
                });
                widgets::property_row(ui, "Cell size", |ui| {
                    ui.label(RichText::new(format!("{:.2} m", o.cell_size)).font(mono.clone()));
                });
                widgets::property_row(ui, "Objects", |ui| {
                    ui.label(RichText::new(o.objects.len().to_string()).font(mono.clone()));
                });
                widgets::property_row(ui, "Size", |ui| {
                    ui.label(RichText::new(format!("{} KB", o.byte_size().div_ceil(1024))).font(mono.clone()));
                });
                let cam = self.scene.occlusion_camera;
                widgets::property_row(ui, "Camera cell", |ui| match cam.and_then(|c| o.cell_of(c)) {
                    Some(c) => {
                        let [x, y, z] = o.cell_coords(c);
                        ui.label(
                            RichText::new(format!("({x}, {y}, {z}) · {} / {} visible", o.visible_count(c), o.objects.len()))
                                .font(mono.clone())
                                .color(theme::accent()),
                        );
                    }
                    None => {
                        ui.label(RichText::new("outside baked bounds (no culling)").color(theme::TEXT_DIM));
                    }
                });
            }
        }
        widgets::section_gap(ui);

        widgets::eyebrow(ui, "Visualization");
        let occ = &mut *self.occlusion_ui;
        ui.checkbox(&mut occ.cull_scene, "Cull the Scene view");
        ui.add_enabled_ui(occ.cull_scene, |ui| {
            ui.checkbox(&mut occ.preview_game, "Preview from the game camera")
                .on_hover_text("Cull from the main camera's position: fly around in the Scene view to see what that camera draws.");
        });
        ui.checkbox(&mut occ.show_cells, "Show cells")
            .on_hover_text("Draw the baked bounds and the culling camera's cell in the Scene view.");
    }

    /// The Environment tab: world sun, ambient, and skybox setup.
    /// "FluxBaker": lighting-bake settings + Bake / Clear actions.
    fn baker_ui(&mut self, ui: &mut egui::Ui) {
        use egui::{DragValue, RichText};
        ui.label(
            RichText::new("Ray-traced lightmaps + light probes for static geometry")
                .size(crate::theme::SIZE_META)
                .color(crate::theme::TEXT_MUTED),
        );
        ui.add_space(8.0);
        // "In this scene": the GI volumes and probes the bake / Flux use.
        let volumes: Vec<(crate::theme::ObjectKind, String)> = self
            .scene
            .objects
            .iter()
            .map(|o| (object_kind(o), o.name.clone()))
            .filter(|(k, _)| {
                matches!(k, crate::theme::ObjectKind::FluxVolume | crate::theme::ObjectKind::ReflectionProbe)
            })
            .collect();
        egui::Frame::new()
            .fill(crate::theme::BG_APP)
            .stroke(egui::Stroke::new(1.0_f32, crate::theme::LINE_PANEL))
            .corner_radius(egui::CornerRadius::same(crate::theme::R_TOOLBAR))
            .inner_margin(egui::Margin::same(10))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(RichText::new("In this scene").font(crate::theme::semibold(crate::theme::SIZE_BODY)).color(crate::theme::TEXT_STRONG));
                if volumes.is_empty() {
                    ui.label(RichText::new("No Flux Volumes or Reflection Probes yet.").size(crate::theme::SIZE_META).color(crate::theme::TEXT_DIM));
                }
                for (kind, name) in &volumes {
                    ui.horizontal(|ui| {
                        crate::icons::icon(ui, kind.icon(), 14.0, kind.color());
                        ui.label(RichText::new(name).color(crate::theme::TEXT_SECONDARY));
                    });
                }
            });
        ui.add_space(8.0);

        // Scrollable settings; Bake/Clear pinned at the bottom.
        let sizes = self.bake_sizes;
        let baked = self.scene.baked.as_ref();
        let summary = baked.map(|b| (b.lightmap_count(), b.probe_sh.len(), b.lightmap_max_size()));
        let footer = 84.0;
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .max_height(ui.available_height() - footer)
            .show(ui, |ui| {
                let bake = &mut self.scene.environment.bake;
                crate::widgets::eyebrow(ui, "Resolution");
                ui.horizontal(|ui| {
                    crate::widgets::row_label(ui, "Texel Density")
                        .on_hover_text("Lightmap texels per world meter (Bakery-style)");
                    crate::widgets::slider(
                        ui,
                        egui::Slider::new(&mut bake.texel_density, 1.0..=1024.0)
                            .logarithmic(true)
                            .suffix(" /m"),
                    );
                });
                ui.horizontal(|ui| {
                    crate::widgets::row_label(ui, "Max Lightmap");
                    egui::ComboBox::from_id_salt("zestengine-bake-max")
                        .selected_text(format!("{}", bake.max_lightmap))
                        .show_ui(ui, |ui| {
                            for s in [128u32, 256, 512, 1024, 2048, 4096] {
                                ui.selectable_value(&mut bake.max_lightmap, s, format!("{s}"));
                            }
                        });
                });
                ui.horizontal(|ui| {
                    crate::widgets::row_label(ui, "GPU Throttle");
                    crate::widgets::slider(ui, egui::Slider::new(&mut bake.gpu_throttle, 0.0..=2.0).step_by(0.1))
                        .on_hover_text(
                            "Idle the GPU for this fraction of each bake dispatch so the desktop \
                             stays responsive while baking. 0 = fastest bake (hogs the GPU, can \
                             freeze the machine); 1 ≈ 50% GPU duty (≈2× slower, system usable).",
                        );
                });

                crate::widgets::section_gap(ui);
                crate::widgets::eyebrow(ui, "Quality");
                ui.horizontal(|ui| {
                    crate::widgets::row_label(ui, "Bounces")
                        .on_hover_text("Indirect light bounces (0 = direct + sky only)");
                    crate::widgets::slider(ui, egui::Slider::new(&mut bake.bounces, 0..=8));
                });
                ui.horizontal(|ui| {
                    crate::widgets::row_label(ui, "Samples")
                        .on_hover_text("Paths traced per texel / probe — higher = less noise");
                    ui.add(DragValue::new(&mut bake.samples).speed(8.0).range(1..=65536));
                });

                crate::widgets::section_gap(ui);
                crate::widgets::eyebrow(ui, "Storage");
                ui.checkbox(&mut bake.compress_lightmaps, "Compress lightmaps (BC6H)").on_hover_text(
                    "Store lightmaps as BC6H: ~16x smaller than raw RGBA32F on disk and in VRAM, \
                     encoded in the background after a bake. GPUs without BC decode them at load.",
                );
                ui.checkbox(&mut bake.compress_reflections, "Compress reflection probes (BC6H)").on_hover_text(
                    "Store baked probe cubes (every roughness mip) as BC6H instead of raw RGBA16F faces.",
                );
                for (stats, what) in [(sizes.lightmaps, "Lightmaps"), (sizes.reflections, "Reflection probes")] {
                    if let Some(st) = stats {
                        ui.label(RichText::new(st.describe(what)).small().weak());
                    }
                }
                if sizes.has_uncompressed()
                    && (bake.compress_lightmaps || bake.compress_reflections)
                    && ui
                        .button(format!("{} Compress existing bake", ph::ARROWS_CLOCKWISE))
                        .on_hover_text("Re-encode this scene's uncompressed .lightmap / .reflprobe as BC6H in the background.")
                        .clicked()
                {
                    self.actions.push(EditorAction::RecompressBake);
                }

                crate::widgets::section_gap(ui);
                crate::widgets::eyebrow(ui, "Lightmap UVs");
                let needing = self.scene.models_needing_lightmap_uv();
                if !needing.is_empty() {
                    ui.label(
                        RichText::new(format!(
                            "{} model(s) have no usable lightmap UV (the bake skips them):",
                            needing.len()
                        ))
                        .small()
                        .weak(),
                    );
                    for m in &needing {
                        ui.label(RichText::new(format!("  • {m}")).small());
                    }
                    if ui
                        .button(format!("{} Generate lightmap UVs for these", ph::GEAR))
                        .on_hover_text(
                            "Auto-unwrap a non-overlapping lightmap UV for each model that needs \
                             one (saved as a .lmuv marker; reloads the scene).",
                        )
                        .clicked()
                    {
                        self.actions.push(EditorAction::GenerateAllLightmapUvs);
                    }
                }
                if !self.generated_uv_models.is_empty() {
                    ui.label(RichText::new("Generated (auto-unwrapped):").small().strong());
                    for m in self.generated_uv_models {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(format!("• {m}")).small());
                            if ui
                                .small_button("Revert")
                                .on_hover_text("Remove the generated UVs; use the model's own uv0.")
                                .clicked()
                            {
                                self.actions
                                    .push(EditorAction::SetLightmapUvGen(m.clone(), false));
                            }
                        });
                    }
                    if ui
                        .button(format!("{} Regenerate all", ph::ARROWS_CLOCKWISE))
                        .on_hover_text(
                            "Re-run the unwrapper on every generated model (picks up unwrapper \
                             improvements). Reloads the scene.",
                        )
                        .clicked()
                    {
                        self.actions.push(EditorAction::RegenerateLightmapUvs);
                    }
                }
                if needing.is_empty() && self.generated_uv_models.is_empty() {
                    ui.label(
                        RichText::new("All models have usable lightmap UVs.").small().weak(),
                    );
                }

                crate::widgets::section_gap(ui);
                crate::widgets::eyebrow(ui, "Status");
                match summary {
                    Some((lm, probes, max)) => {
                        ui.label(format!("Baked: {lm} lightmap(s) (max {max}px), {probes} probe(s)"));
                        ui.label(
                            RichText::new(
                                ".lightmap = static GI · .lightdata = probe data for dynamic objects",
                            )
                            .small()
                            .weak(),
                        );
                    }
                    None => {
                        ui.label(RichText::new("No bake yet").weak());
                    }
                }
                if !self.can_bake {
                    ui.label(
                        RichText::new(format!("{} This GPU has no ray-query support; baking is unavailable", ph::WARNING))
                            .small()
                            .color(ui.visuals().warn_fg_color),
                    );
                }
                ui.label(
                    RichText::new("Tip: mark objects Static (Inspector) and add Light Probe Volumes for dynamic objects.")
                        .small()
                        .weak(),
                );
            });

        crate::widgets::section_gap(ui);
        ui.horizontal(|ui| {
            use crate::{icons::Icon, widgets};
            let clear_w = 72.0;
            let bake_w = (ui.available_width() - clear_w - 6.0).max(80.0);
            let bake = ui
                .add_enabled_ui(self.can_bake, |ui| {
                    ui.allocate_ui(egui::vec2(bake_w, crate::theme::H_PRIMARY), |ui| {
                        ui.set_min_width(bake_w);
                        widgets::primary_button(ui, Some(Icon::LightDirectional), "Bake lighting")
                    })
                    .inner
                })
                .inner;
            if bake.on_hover_text("Trace lightmaps + probes now (blocks the UI)").clicked() {
                self.actions.push(EditorAction::BakeLighting);
            }
            if ui
                .add_enabled_ui(summary.is_some(), |ui| widgets::secondary_button_sized(ui, None, "Clear", crate::theme::H_PRIMARY))
                .inner
                .on_hover_text("Discard all baked lighting data")
                .clicked()
            {
                self.actions.push(EditorAction::ClearBake);
            }
        });
        // Reflection probe baking: capture the active probe's cubemap and persist
        // it as a `.reflprobe` sidecar (loaded on scene open). Disabled until the
        // scene has a Reflection Probe (add one from the hierarchy → Light menu).
        let has_probe = self
            .scene
            .active_reflection_probe(glam::Vec3::ZERO)
            .is_some();
        ui.horizontal(|ui| {
            if ui
                .add_enabled_ui(has_probe, |ui| {
                    crate::widgets::secondary_button(ui, Some(crate::icons::Icon::ReflectionProbe), "Bake reflections")
                })
                .inner
                .on_hover_text(
                    "Capture the active Reflection Probe's cubemap and save it as a \
                     .reflprobe sidecar (loaded on scene open)",
                )
                .clicked()
            {
                self.actions.push(EditorAction::BakeReflections);
            }
            if ui
                .add_enabled_ui(has_probe, |ui| crate::widgets::secondary_button(ui, Some(crate::icons::Icon::Reset), "Recapture"))
                .inner
                .on_hover_text(
                    "Re-render the active probe's cubemap from the current scene \
                     (this session only, not saved)",
                )
                .clicked()
            {
                self.actions.push(EditorAction::RecaptureReflections);
            }
        });
        if !has_probe {
            ui.label(
                RichText::new("Add a Reflection Probe (hierarchy -> Light) to bake reflections.")
                    .small()
                    .weak(),
            );
        }
        crate::widgets::section_gap(ui);
        ui.checkbox(self.lightmap_preview, "UV checker preview")
            .on_hover_text(
                "Show objects as a lightmap-UV checkerboard — the cell size tracks each \
                 object's texel density (big squares = low resolution, stretched squares = \
                 UV distortion). Grey = not lightmapped (non-static).",
            );
    }

    fn environment_ui(&mut self, ui: &mut egui::Ui) {
        use egui::{DragValue, RichText};
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {

                {
                    let baked = self.scene.baked.is_some();
                    let env = &mut self.scene.environment;
                    // Key light: a "Directional Light" object in the Hierarchy, edited like any light.
                    crate::widgets::eyebrow(ui, "Ambient");
                    ui.horizontal(|ui| {
                        crate::widgets::row_label(ui, "Color");
                        ui.color_edit_button_rgb(&mut env.ambient);
                    });
                    ui.horizontal(|ui| {
                        crate::widgets::row_label(ui, "Intensity");
                        ui.add(
                            DragValue::new(&mut env.ambient_intensity)
                                .speed(0.02)
                                .range(0.0..=10.0),
                        );
                    });

                    crate::widgets::section_gap(ui);
                    crate::widgets::eyebrow(ui, "Flux GI (realtime)");
                    let gi = &mut env.realtime_gi;
                    // Settings stay EDITABLE even with a baked lightmap loaded: the
                    // FluxVoxel backend runs alongside a bake (the baked voxels are its
                    // static base), and for the other backends the values you set here
                    // take effect the moment the bake is cleared. (Previously the whole
                    // section was disabled when baked, so you couldn't tweak anything.)
                    ui.add_enabled_ui(true, |ui| {
                        ui.checkbox(&mut gi.enabled, "Enabled").on_hover_text(
                            "Flux: realtime global illumination. Screen-space probes \
                             march the scene distance field every frame for live indirect bounce \
                             — no bake, runs anywhere (no RT cores needed).",
                        );
                        ui.add_enabled_ui(gi.enabled, |ui| {
                            // Backend first: it decides which settings below apply.
                            // Each control is gated by `gi.mode.uses_*()` so the panel
                            // only shows what the chosen backend actually consumes
                            // (modular settings — FluxVoxel hides bounces/quality/etc).
                            ui.horizontal(|ui| {
                                crate::widgets::row_label(ui, "Backend");
                                egui::ComboBox::from_id_salt("zestengine-flux-mode")
                                    .selected_text(gi.mode.label())
                                    .show_ui(ui, |ui| {
                                        for m in zestengine_assets::GiMode::ALL {
                                            ui.selectable_value(&mut gi.mode, m, m.label());
                                        }
                                    });
                            })
                            .response
                            .on_hover_text(
                                "Trace backend (NOT auto): Flux marches the SDF (runs anywhere); \
                                 FluxRT ray-queries the scene BVH for exact geometry (needs a \
                                 ray-query GPU, falls back to Flux); FluxVoxel injects lights into \
                                 an analytic voxel volume — cheapest, no ray tracing, for VR / \
                                 low-end. Settings below adapt to the backend.",
                            );
                            ui.horizontal(|ui| {
                                crate::widgets::row_label(ui, "Ray Lighting Mode");
                                ray_lighting_mode_ui(ui, &mut gi.ray_lighting_mode, false);
                            });
                            let mode = gi.mode;
                            ui.horizontal(|ui| {
                                crate::widgets::row_label(ui, "Intensity");
                                crate::widgets::slider(ui, egui::Slider::new(&mut gi.intensity, 0.0..=4.0))
                                    .on_hover_text("Indirect bounce strength multiplier.");
                            });
                            if mode.uses_quality() {
                                ui.horizontal(|ui| {
                                    crate::widgets::row_label(ui, "Quality");
                                    egui::ComboBox::from_id_salt("zestengine-flux-quality")
                                        .selected_text(gi.quality.label())
                                        .show_ui(ui, |ui| {
                                            for q in zestengine_assets::FluxQuality::ALL {
                                                ui.selectable_value(&mut gi.quality, q, q.label());
                                            }
                                        });
                                })
                                .response
                                .on_hover_text(
                                    "Rays per screen probe each frame. Temporal accumulation \
                                     smooths the rest, so even Performance stays clean once settled.",
                                );
                            }
                            if mode.uses_bounces() {
                                ui.horizontal(|ui| {
                                    crate::widgets::row_label(ui, "Diffuse bounces");
                                    if ui.selectable_label(gi.bounces == 1, "Single").clicked() { gi.bounces = 1; }
                                    if ui.selectable_label(gi.bounces > 1, "Infinite").on_hover_text(
                                        "Diffuse light continues bouncing through successive updates. Quality controls sampling; new lighting takes time to settle.",
                                    ).clicked() { gi.bounces = 4; }
                                });
                            }
                            ui.horizontal(|ui| {
                                crate::widgets::row_label(ui, "Smoothing");
                                crate::widgets::slider(ui, egui::Slider::new(&mut gi.smoothing, 0.0..=1.0))
                                    .on_hover_text(
                                        "Temporal stability: higher = smoother but more motion lag; \
                                         lower = sharper/more responsive but noisier while moving.",
                                    );
                            });
                            // FluxVoxel-only controls: density of the auto scene grid +
                            // the expensive toggles (DDGI occlusion). All ON by default.
                            if mode.uses_voxel_density() {
                                ui.horizontal(|ui| {
                                    crate::widgets::row_label(ui, "Voxel density");
                                    crate::widgets::slider(
                                        ui,
                                        egui::Slider::new(&mut gi.voxel_density, 0.25..=4.0)
                                            .suffix(" /m"),
                                    )
                                    .on_hover_text(
                                        "Probes per world meter for the auto scene-covering grid \
                                         (used when there are no FluxVolumes). Higher = finer GI, \
                                         more cost. Per-FluxVolume density is set on each volume.",
                                    );
                                });
                                ui.checkbox(&mut gi.voxel_auto_grid, "Auto scene grid")
                                    .on_hover_text(
                                        "Build a grid covering the scene when no FluxVolumes are \
                                         placed, so everything gets voxel GI. Mutually exclusive \
                                         with placed FluxVolumes: placing any volume turns this off \
                                         and the author then controls coverage.",
                                    );
                                if gi.voxel_auto_grid {
                                    ui.horizontal(|ui| {
                                        crate::widgets::row_label(ui, "Grid mode");
                                        egui::ComboBox::from_id_salt("zestengine-voxel-gridmode")
                                            .selected_text(gi.voxel_grid_mode.label())
                                            .show_ui(ui, |ui| {
                                                for m in zestengine_assets::VoxelGridMode::ALL {
                                                    ui.selectable_value(
                                                        &mut gi.voxel_grid_mode,
                                                        m,
                                                        m.label(),
                                                    );
                                                }
                                            });
                                    })
                                    .response
                                    .on_hover_text(
                                        "Auto-grid layout:\n\
                                         • Whole scene: one box over everything (cost scales with level size).\n\
                                         • Camera clipmap: fixed cube following the camera — constant cost, best for big levels.\n\
                                         • Occupancy culled: tight bounds around geometry.\n\
                                         • Per object: tight grids hugging object clusters, gaps left uncovered.",
                                    );
                                    if gi.voxel_grid_mode
                                        == zestengine_assets::VoxelGridMode::CameraClipmap
                                    {
                                        ui.horizontal(|ui| {
                                            crate::widgets::row_label(ui, "Clipmap extent");
                                            crate::widgets::slider(
                                                ui,
                                                egui::Slider::new(
                                                    &mut gi.voxel_clipmap_extent,
                                                    4.0..=64.0,
                                                )
                                                .suffix(" m"),
                                            )
                                            .on_hover_text(
                                                "Half-size of the camera-following cube; the grid is \
                                                 2× this on a side, centred on the camera.",
                                            );
                                        });
                                    }
                                }
                                ui.checkbox(&mut gi.voxel_ddgi_occlusion, "DDGI occlusion")
                                    .on_hover_text(
                                        "Block voxel lights with geometry (cheap DDGI-style shadows \
                                         from a coarse scene occupancy grid). Turn off on low-end / \
                                         VR to save the per-probe marches.",
                                    );
                                ui.checkbox(&mut gi.voxel_propagation, "Light propagation")
                                    .on_hover_text(
                                        "LPV-style diffuse bounce: spread injected light through \
                                         the voxel grid so it fills shadowed pockets (≥1 bounce). \
                                         Off = direct-only (cheapest).",
                                    );
                                ui.checkbox(&mut gi.voxel_specular, "Specular GI")
                                    .on_hover_text(
                                        "Metallic / rough surfaces sample the voxel volume in the \
                                         reflection direction (VXGI-style glossy) so they pick up \
                                         emissive + voxel-light bounce, not just the reflection cube.",
                                    );
                            }
                            egui::CollapsingHeader::new("Advanced")
                                .id_salt("zestengine-flux-advanced")
                                .show(ui, |ui| {
                                    if mode.uses_gdf() {
                                        ui.horizontal(|ui| {
                                            crate::widgets::row_label(ui, "GDF resolution");
                                            egui::ComboBox::from_id_salt("zestengine-flux-gdfres")
                                                .selected_text(format!("{}³", gi.gdf_resolution))
                                                .show_ui(ui, |ui| {
                                                    for r in [64u32, 128, 256] {
                                                        ui.selectable_value(
                                                            &mut gi.gdf_resolution,
                                                            r,
                                                            format!("{r}³"),
                                                        );
                                                    }
                                                });
                                        });
                                    }
                                    if mode.uses_march_distance() {
                                        ui.horizontal(|ui| {
                                            crate::widgets::row_label(ui, "March distance");
                                            crate::widgets::slider(
                                                ui,
                                                egui::Slider::new(&mut gi.march_distance, 0.0..=500.0)
                                                    .suffix(" m"),
                                            )
                                            .on_hover_text(
                                                "Max trace distance (0 = auto from scene size).",
                                            );
                                        });
                                    }
                                    if mode.uses_samples() {
                                        ui.horizontal(|ui| {
                                            crate::widgets::row_label(ui, "Firefly clamp");
                                            crate::widgets::slider(ui, egui::Slider::new(
                                                &mut gi.firefly_clamp,
                                                0.5..=16.0,
                                            ))
                                            .on_hover_text(
                                                "Caps bright bounce outliers (lower = calmer).",
                                            );
                                        });
                                    }
                                    ui.horizontal(|ui| {
                                        crate::widgets::row_label(ui, "Emissive boost");
                                        crate::widgets::slider(ui, egui::Slider::new(&mut gi.emissive_boost, 0.0..=32.0).logarithmic(true))
                                            .on_hover_text(
                                                "How much light emissive objects throw into the GI (their own brightness \
                                                 is unchanged). 1 = physical; raise for stronger coloured pools.",
                                            );
                                    });
                                    ui.horizontal(|ui| {
                                        crate::widgets::row_label(ui, "Probe fallback");
                                        egui::ComboBox::from_id_salt("zestengine-flux-fallback")
                                            .selected_text(gi.probe_fallback.label())
                                            .show_ui(ui, |ui| {
                                                for f in zestengine_assets::ProbeFallback::ALL {
                                                    ui.selectable_value(
                                                        &mut gi.probe_fallback,
                                                        f,
                                                        f.label(),
                                                    );
                                                }
                                            });
                                    });
                                });
                        });
                    });
                    if baked {
                        let note = if gi.mode == zestengine_assets::GiMode::FluxVoxel {
                            "FluxVoxel runs alongside the bake (baked voxels are its static base) \
                             — these settings are live."
                        } else {
                            "A bake is loaded: Flux/FluxRT run in mixed mode, adding moving \
                             emitters and realtime lights on top of the lightmaps."
                        };
                        ui.label(RichText::new(note).small().weak());
                    }

                    // Reflections: a dedicated Environment section (NOT under GI),
                    // matching Unreal where reflection method is a global scene
                    // setting, not a per-material toggle. Type = how specular
                    // reflections are produced for the whole scene.
                    crate::widgets::section_gap(ui);
                    egui::CollapsingHeader::new("Reflections")
                        .id_salt("zestengine-env-reflections")
                        .default_open(true)
                        .show(ui, |ui| {
                            let gi = &mut env.realtime_gi;
                            ui.horizontal(|ui| {
                                crate::widgets::row_label(ui, "Type");
                                let label = match gi.reflection_mode {
                                    1 => "Screen-space (SSR)",
                                    2 => "Ray-traced",
                                    _ => "Reflection Probes",
                                };
                                egui::ComboBox::from_id_salt("zestengine-refl-type")
                                    .selected_text(label)
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(&mut gi.reflection_mode, 0, "Reflection Probes")
                                            .on_hover_text(
                                                "Box-projected cubemap probes, with the env / \
                                                 skylight cube as the fallback. Cheapest, works \
                                                 everywhere.",
                                            );
                                        ui.selectable_value(&mut gi.reflection_mode, 1, "Screen-space (SSR)")
                                            .on_hover_text(
                                                "Screen-space reflections layered over the probe / \
                                                 env base; off-screen rays fall back to the probes.",
                                            );
                                        ui.selectable_value(&mut gi.reflection_mode, 2, "Ray-traced")
                                            .on_hover_text(
                                                "Traces real geometry (needs a ray-query GPU; \
                                                 falls back to SSR if unavailable).",
                                            );
                                    });
                            });
                            // Intensity / distance / roughness tune SSR + Ray-traced.
                            gi.ssr_enabled = gi.reflection_mode == 1;
                            ui.add_enabled_ui(gi.reflection_mode != 0, |ui| {
                                ui.horizontal(|ui| {
                                    crate::widgets::row_label(ui, "Intensity");
                                    crate::widgets::slider(ui, egui::Slider::new(&mut gi.ssr_intensity, 0.0..=2.0))
                                        .on_hover_text("Reflection strength multiplier.");
                                });
                                ui.horizontal(|ui| {
                                    crate::widgets::row_label(ui, "Max distance");
                                    crate::widgets::slider(
                                        ui,
                                        egui::Slider::new(&mut gi.ssr_max_distance, 1.0..=200.0)
                                            .suffix(" m"),
                                    )
                                    .on_hover_text("Max view-space ray distance.");
                                });
                                ui.horizontal(|ui| {
                                    crate::widgets::row_label(ui, "Roughness cutoff");
                                    crate::widgets::slider(ui, egui::Slider::new(&mut gi.ssr_roughness_cutoff, 0.0..=1.0))
                                        .on_hover_text(
                                            "Surfaces rougher than this skip SSR / ray tracing (the probe / env \
                                             cube carries them instead). Raise it to ray trace glossier surfaces.",
                                        );
                                });
                                if gi.reflection_mode == 2 {
                                    ui.horizontal(|ui| {
                                        crate::widgets::row_label(ui, "Rays per pixel");
                                        crate::widgets::segmented(
                                            ui,
                                            &mut gi.rt_samples,
                                            &[(1, "1"), (2, "2"), (4, "4"), (8, "8"), (16, "16")],
                                        )
                                        .on_hover_text(
                                            "Rough surfaces spread these over their glossy lobe: more rays = \
                                             smoother glossy reflections, at a matching GPU cost.",
                                        );
                                    });
                                    ui.horizontal(|ui| {
                                        crate::widgets::row_label(ui, "Bounces");
                                        crate::widgets::segmented(ui, &mut gi.rt_bounces, &[(1, "1"), (2, "2"), (3, "3"), (4, "4"), (8, "8")])
                                            .on_hover_text(
                                                "Reflections of reflections: 2+ shows mirrors and metals inside \
                                                 reflections instead of their environment fallback.",
                                            );
                                    });
                                }
                            });
                            crate::widgets::section_gap(ui);
                            ui.label(RichText::new("Reflection Probes").small().strong());
                            ui.label(
                                RichText::new(
                                    "Add a Reflection Probe component to an object for a \
                                     box-projected cubemap. The highest-priority probe covering a \
                                     surface is used (higher Importance wins, then smaller box), \
                                     fading to the skybox at the box edge. Recapture after editing \
                                     the scene.",
                                )
                                .small()
                                .weak(),
                            );
                            ui.horizontal(|ui| {
                                if ui
                                    .button("Recapture")
                                    .on_hover_text(
                                        "Re-render the active probe's cubemap from the current \
                                         scene (this session only).",
                                    )
                                    .clicked()
                                {
                                    self.actions.push(EditorAction::RecaptureReflections);
                                }
                                if ui
                                    .button("Bake to disk")
                                    .on_hover_text(
                                        "Recapture, then persist the cube as a .reflprobe sidecar \
                                         (loaded on scene open as a baked reflection probe).",
                                    )
                                    .clicked()
                                {
                                    self.actions.push(EditorAction::BakeReflections);
                                }
                            });
                        });

                    crate::widgets::section_gap(ui);
                    crate::widgets::eyebrow(ui, "Shadows");
                    ui.horizontal(|ui| {
                        crate::widgets::row_label(ui, "Resolution");
                        egui::ComboBox::from_id_salt("zestengine-shadow-res")
                            .selected_text(format!("{}", env.shadow_resolution))
                            .show_ui(ui, |ui| {
                                for res in [512u32, 1024, 2048, 4096] {
                                    ui.selectable_value(
                                        &mut env.shadow_resolution,
                                        res,
                                        format!("{res} × {res}"),
                                    );
                                }
                            });
                    });
                    ui.horizontal(|ui| {
                        crate::widgets::row_label(ui, "Distance");
                        crate::widgets::slider(
                            ui,
                            egui::Slider::new(&mut env.shadow_distance, 5.0..=150.0).suffix(" m"),
                        )
                        .on_hover_text(
                            "Directional shadow coverage. Smaller = sharper, less reach.",
                        );
                    });
                    ui.horizontal(|ui| {
                        crate::widgets::row_label(ui, "Softness");
                        crate::widgets::slider(ui, egui::Slider::new(&mut env.shadow_softness, 0.0..=4.0));
                    });
                    ui.horizontal(|ui| {
                        crate::widgets::row_label(ui, "Bias scale");
                        crate::widgets::slider(
                            ui,
                            egui::Slider::new(&mut env.shadow_bias_scale, 100.0..=6000.0).logarithmic(true),
                        )
                        .on_hover_text(
                            "Receiver depth-bias strength. Point/spot lights use a sub-texel default; 1500 is 1x. Directional lights retain slope scaling. Raise for acne; lower if contact shadows detach.",
                        );
                    });
                    ui.horizontal(|ui| {
                        crate::widgets::row_label(ui, "Normal offset");
                        crate::widgets::slider(
                            ui,
                            egui::Slider::new(&mut env.shadow_normal_offset, 0.1..=8.0).logarithmic(true),
                        )
                        .on_hover_text(
                            "Directional shadow normal offset in shadow texels, scaled by the angle to the light. Point and spot lights use receiver-plane filtering without this offset. Default 2.",
                        );
                    });
                    ui.horizontal(|ui| {
                        crate::widgets::row_label(ui, "Cascades");
                        crate::widgets::slider(ui, egui::Slider::new(&mut env.shadow_cascades, 1..=4))
                            .on_hover_text("Directional-light shadow cascades. Fewer = a sharper spread over a short range and fewer shadow passes; mobile caps it at 2.");
                    });
                    ui.horizontal(|ui| {
                        crate::widgets::row_label(ui, "Cascade split");
                        crate::widgets::slider(ui, egui::Slider::new(&mut env.shadow_cascade_lambda, 0.05..=1.0))
                            .on_hover_text("0 = even cascade lengths, 1 = logarithmic (finest near the camera). Default 0.7.");
                    });
                    ui.label(
                        RichText::new(format!(
                            "≈ {:.0} texels/m, PCF spacing ≈ {:.5} uv",
                            env.shadow_resolution as f32 / env.shadow_distance.max(1.0),
                            env.shadow_softness.max(0.0) / env.shadow_resolution as f32
                        ))
                        .small()
                        .weak(),
                    );

                    crate::widgets::section_gap(ui);
                    egui::CollapsingHeader::new("Fog")
                        .id_salt("zestengine-env-fog")
                        .show(ui, |ui| {
                            ui.checkbox(&mut env.fog_enabled, "Enabled")
                                .on_hover_text("Exponential distance + height fog (atmospheric depth).");
                            ui.add_enabled_ui(env.fog_enabled, |ui| {
                                ui.horizontal(|ui| {
                                    crate::widgets::row_label(ui, "Color");
                                    ui.color_edit_button_rgb(&mut env.fog_color);
                                });
                                ui.horizontal(|ui| {
                                    crate::widgets::row_label(ui, "Density");
                                    crate::widgets::slider(ui, egui::Slider::new(&mut env.fog_density, 0.0..=0.5).logarithmic(true));
                                });
                                ui.horizontal(|ui| {
                                    crate::widgets::row_label(ui, "Start distance");
                                    crate::widgets::slider(ui, egui::Slider::new(&mut env.fog_start_distance, 0.0..=100.0).suffix(" m"));
                                });
                                ui.horizontal(|ui| {
                                    crate::widgets::row_label(ui, "Height falloff");
                                    crate::widgets::slider(ui, egui::Slider::new(&mut env.fog_height_falloff, 0.0..=1.0))
                                        .on_hover_text("Fog thins above the height reference (0 = uniform).");
                                });
                                ui.horizontal(|ui| {
                                    crate::widgets::row_label(ui, "Height ref");
                                    crate::widgets::slider(ui, egui::Slider::new(&mut env.fog_height_ref, -50.0..=50.0).suffix(" m"));
                                });
                            });
                        });

                    crate::widgets::section_gap(ui);
                    egui::CollapsingHeader::new("Post-processing (global)")
                        .id_salt("zestengine-env-postfx")
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.label(
                                RichText::new(
                                    "The scene's base post FX. Post FX Volumes blend on top by priority and \
                                     weight, and each effect a volume enables overrides the same effect here.",
                                )
                                .small()
                                .weak(),
                            );
                            let overrides = &self.scene.postfx_overrides;
                            if overrides.iter().any(|(_, _, fx)| !fx.is_empty()) {
                                for (name, weight, fx) in overrides.iter().filter(|(_, _, fx)| !fx.is_empty()) {
                                    crate::widgets::info_note(
                                        ui,
                                        &format!("Overridden by \"{name}\" ({:.0}%): {}", weight * 100.0, fx.join(", ")),
                                    );
                                }
                            }
                            let env = &mut self.scene.environment;
                            postfx_editor_ui(ui, &mut env.postfx, false);
                        });
                }

                // ---- Skybox section: enable toggle + the texture sources (a single
                // 360° equirect image, OR a 6-face cubemap that overrides it). The
                // toggle also gates whether the skybox lights the scene (IBL): off →
                // metals stop reflecting it, so with no lights the scene is black.
                crate::widgets::section_gap(ui);
                crate::widgets::eyebrow(ui, "Skybox");
                ui.checkbox(&mut self.scene.environment.skybox_enabled, "Enable skybox")
                    .on_hover_text(
                        "Draw the skybox AND let it light the scene (image-based lighting): \
                         metallic/smooth surfaces reflect it. Off = no skybox reflection, so \
                         with no lights the scene is black.",
                    );
                let is_image = |p: &std::path::Path| {
                    p.extension()
                        .and_then(|e| e.to_str())
                        .map(|e| e.to_ascii_lowercase())
                        .is_some_and(|e| {
                            matches!(e.as_str(), "png" | "jpg" | "jpeg" | "bmp" | "tga" | "hdr" | "exr")
                        })
                };

                // 360° equirectangular image slot (used when no full cubemap is set).
                let slot = egui::Frame::group(ui.style())
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.label("360° image (equirectangular)");
                        match &self.scene.skybox {
                            Some(path) => {
                                ui.label(RichText::new(path).small().weak());
                            }
                            None => {
                                ui.label(
                                    RichText::new("Procedural gradient — drop an image here")
                                        .small()
                                        .weak(),
                                );
                            }
                        }
                    })
                    .response;
                if slot
                    .dnd_hover_payload::<std::path::PathBuf>()
                    .is_some_and(|p| is_image(&p))
                {
                    ui.painter().rect_stroke(
                        slot.rect,
                        4.0,
                        egui::Stroke::new(2.0_f32, ui.visuals().selection.stroke.color),
                        egui::StrokeKind::Outside,
                    );
                }
                if let Some(p) = slot.dnd_release_payload::<std::path::PathBuf>()
                    && is_image(&p)
                {
                    self.actions.push(EditorAction::SetSkybox((*p).clone()));
                }
                ui.horizontal(|ui| {
                    const IMAGES: [&str; 7] = ["png", "jpg", "jpeg", "bmp", "tga", "hdr", "exr"];
                    let current = self.scene.skybox.clone().unwrap_or_default();
                    match crate::asset_browse_button(ui, "skybox-360", &IMAGES, &current) {
                        Some(p) if p.is_empty() => self.actions.push(EditorAction::ClearSkybox),
                        Some(p) => self.actions.push(EditorAction::SetSkybox(self.project_root.join(p))),
                        None => {}
                    }
                    ui.label(RichText::new("Browse images").small().weak());
                });
                if self.scene.skybox.is_some() && ui.button("Clear 360° image").clicked() {
                    self.actions.push(EditorAction::ClearSkybox);
                }

                // 6-face cubemap. When all six faces are assigned they OVERRIDE the
                // 360° image; a partial set falls back to the equirect/procedural sky.
                egui::CollapsingHeader::new("Cubemap (6 faces)")
                    .id_salt("zestengine-skybox-cube")
                    .show(ui, |ui| {
                        ui.label(
                            RichText::new(
                                "Drop an image on each face. All six together override the \
                                 360° image above.",
                            )
                            .small()
                            .weak(),
                        );
                        const FACE_LABELS: [&str; 6] =
                            ["+X right", "-X left", "+Y top", "-Y bottom", "+Z front", "-Z back"];
                        for i in 0..6 {
                            let assigned = self
                                .scene
                                .environment
                                .skybox_faces
                                .as_ref()
                                .map(|f| f[i].clone())
                                .filter(|s| !s.is_empty());
                            let face = egui::Frame::group(ui.style())
                                .show(ui, |ui| {
                                    ui.set_width(ui.available_width());
                                    ui.horizontal(|ui| {
                                        const IMAGES: [&str; 7] = ["png", "jpg", "jpeg", "bmp", "tga", "hdr", "exr"];
                                        let current = assigned.clone().unwrap_or_default();
                                        if let Some(p) = crate::asset_browse_button(ui, FACE_LABELS[i], &IMAGES, &current)
                                            && !p.is_empty()
                                        {
                                            self.actions.push(EditorAction::SetSkyboxFace(i, self.project_root.join(p)));
                                        }
                                        ui.label(FACE_LABELS[i]);
                                        match &assigned {
                                            Some(p) => {
                                                ui.label(RichText::new(p).small().weak());
                                            }
                                            None => {
                                                ui.label(
                                                    RichText::new("drop image")
                                                        .small()
                                                        .weak(),
                                                );
                                            }
                                        }
                                    });
                                })
                                .response;
                            if face
                                .dnd_hover_payload::<std::path::PathBuf>()
                                .is_some_and(|p| is_image(&p))
                            {
                                ui.painter().rect_stroke(
                                    face.rect,
                                    4.0,
                                    egui::Stroke::new(
                                        2.0_f32,
                                        ui.visuals().selection.stroke.color,
                                    ),
                                    egui::StrokeKind::Outside,
                                );
                            }
                            if let Some(p) = face.dnd_release_payload::<std::path::PathBuf>()
                                && is_image(&p)
                            {
                                self.actions.push(EditorAction::SetSkyboxFace(i, (*p).clone()));
                            }
                        }
                        if self.scene.environment.skybox_faces.is_some()
                            && ui.button("Clear faces").clicked()
                        {
                            self.actions.push(EditorAction::ClearSkyboxFaces);
                        }
                    });

                ui.add_space(8.0);
                ui.label(
                    RichText::new(
                        "Disable the sun, zero the ambient, and turn off the skybox \
                         for a fully black world.",
                    )
                    .small()
                    .weak(),
                );
            });
    }

    fn viewport_ui(&mut self, ui: &mut egui::Ui) {
        // The Viewport tab is on screen this frame, so the main scene render is
        // worth doing (gate read after the dock pass in `redraw`).
        *self.viewport_visible = true;
        let rect = ui.max_rect();
        *self.viewport_rect = rect;
        ui.ctx().data_mut(|d| d.insert_temp(egui::Id::new(VIEW_LAYER_KEY), ui.layer_id()));
        // Paint the editor viewport's offscreen render (RTT) filling the tab; the
        // gizmo + billboards draw on top. The texture is THIS frame's render: the
        // renderer records the viewport pass before the egui swapchain pass in the
        // same command buffer, with the camera as it is after this pass. So every
        // overlay below must use the camera as it is after this frame's scroll /
        // orbit, which is why those gestures are applied here (`dolly_now`,
        // `orbit_now`) before the first overlay, not as actions after the pass.
        if let Some((tex, _)) = self.viewport_texture {
            ui.painter().image(
                tex,
                rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        }
        let response = ui.interact(
            rect,
            ui.id().with("viewport-interact"),
            egui::Sense::click_and_drag(),
        );

        // Right mouse over the viewport starts mouse-look. Detected through
        // this widget (not raw winit) so egui's hit-testing decides; clicks
        // on panels, tab bars, or resize handles never reach us.
        if response.hovered()
            && ui.input(|i| i.pointer.button_pressed(egui::PointerButton::Secondary))
        {
            self.actions.push(EditorAction::StartLook);
        }

        // Scroll dollies the camera only while the pointer is over the viewport
        // AND the viewport widget actually has the pointer (not occluded by a
        // floating window). The winit rect test alone let scroll bleed through
        // windows sitting over the viewport; `contains_pointer()` is
        // occlusion-aware (same reason clicks use `hovered()` above), so a window
        // over the cursor now swallows the wheel like it swallows clicks.
        // The sky probe's `action-*` modes inject the same gestures (empty otherwise).
        let probe_gestures = crate::sky_probe::take_gestures();
        if self.pointer_in_viewport && response.contains_pointer() {
            let scroll = ui.input(|i| i.raw_scroll_delta.y);
            // Ctrl+Scroll sizes the cloth pin brush instead.
            if scroll != 0.0 && !((crate::cloth_paint::active(ui.ctx()) || crate::terrain_paint::active(ui.ctx())) && ui.input(|i| i.modifiers.command)) {
                if self.scene_shows_game {
                    // The wheel belongs to the game (zoom): `process_actions`
                    // routes it to the game's input.
                    self.actions.push(EditorAction::Dolly(scroll / 50.0));
                } else {
                    self.dolly_now(scroll / 50.0, rect);
                }
            }
        }
        for g in &probe_gestures {
            if let crate::sky_probe::ProbeAction::Dolly(amount) = *g {
                self.dolly_now(amount, rect);
            }
        }

        // Left drag orbits (around the selection, or whatever sits at the
        // viewport center) unless the gizmo grabbed the drag. Alt forces
        // the orbit even when starting on a gizmo handle.
        let alt = ui.input(|i| i.modifiers.alt);
        // `hover_pos()` is None while a drag is in progress, which would make
        // the probe/gizmo handle hit-tests bail the instant a drag starts (so
        // orbit stole the gesture). Fall back to the interaction position,
        // which stays valid through the whole press+drag.
        let cursor = response
            .hover_pos()
            .or_else(|| response.interact_pointer_pos());
        let mut gizmo_changed = false;
        // Building a river path hides the transform gizmo (clicks lay points).
        let river_building = !self.playing
            && matches!(*self.selection, Selection::Object(i) if self.scene.is_active(i))
            && crate::river_edit::building(ui.ctx());
        // Painting cloth pins: left drags paint (no orbit, no transform gizmo).
        let cloth_painting = !self.playing && crate::cloth_paint::active(ui.ctx());
        let terrain_painting = !self.playing && (crate::terrain_paint::active(ui.ctx()) || crate::tile_edit::active(ui.ctx()) || crate::shape_edit::active(ui.ctx()));
        let river_building = river_building || cloth_painting || terrain_painting;

        // Claim a drag for the gizmo the instant it starts on a handle (using
        // the gizmo's own enlarged pick band), so orbit is suppressed for the
        // whole gesture even before/if the gizmo's grab registers. Alt forces
        // orbit. Cleared when the drag ends.
        if response.drag_started_by(egui::PointerButton::Primary) {
            let press = response.interact_pointer_pos();
            // Orbit may only begin when the press is over the viewport (egui
            // can otherwise route a panel drag to this big interact); it then
            // continues through the gesture even if the pointer leaves. The
            // winit-based check keeps it working right after mouse-look ends.
            *self.orbit_armed =
                !cloth_painting && !terrain_painting && self.pointer_in_viewport && press.is_some_and(|p| rect.contains(p));
            *self.gizmo_drag = !alt
                && !river_building
                && matches!(self.selection, Selection::Object(_))
                && press.is_some_and(|p| self.gizmo.pick_preview(p));
        }
        if response.drag_stopped_by(egui::PointerButton::Primary) {
            *self.gizmo_drag = false;
            *self.orbit_armed = false;
        }

        // Screen UI in the game-UI overlay: click selects an element, the
        // selected one moves / resizes with 2D handles (its 3D transform does
        // nothing, so the 3D gizmo is hidden for it). Handles paint after the
        // overlay, in `redraw`.
        let mut ui_clicked = false;
        let mut ui_selected_screen = false;
        if self.ui_overlay_edit && !self.looking && !self.orbiting() {
            let selected = match *self.selection {
                Selection::Object(i) => Some(i),
                _ => None,
            };
            let nodes = zestengine_engine::game_ui::layout_rects(
                self.scene,
                rect,
                ui.ctx().pixels_per_point(),
                zestengine_engine::game_ui::UiScope::Screen,
                selected,
            );
            ui_selected_screen = selected.is_some_and(|s| nodes.iter().any(|n| n.0 == s));
            let out = self.ui_rect_editor.interact(ui, &response, self.scene, &nodes, selected, rect, false);
            if let Some(Some(i)) = out.clicked {
                *self.selection = Selection::Object(i);
                self.multi_objects.clear();
                ui_clicked = true;
            }
            if out.dragging {
                *self.orbit_armed = false;
                *self.gizmo_drag = false;
            }
            if let Some(s) = selected.filter(|_| ui_selected_screen && response.contains_pointer()) {
                crate::ui_edit::UiRectEditor::nudge(ui, self.scene, s);
            }
        }

        // While the camera is actively orbiting, ignore every viewport widget so
        // a rotate gesture can't grab a probe/collider/audio handle or the gizmo
        // as the cursor sweeps across them.
        if !self.orbiting() && self.widget_filter.enabled && self.widget_filter.components {
            // Light Probe Volume box-resize: dragging a face handle changes the
            // volume's size along that axis, keeping the opposite face fixed. Run
            // before the transform gizmo so a grabbed handle wins the drag; the
            // gizmo still moves/rotates the object when no handle is grabbed.
            self.gizmo_interaction(&response, cursor, alt);
        }
        let probe_active = self.handle_drag.is_some();
        let new_handle_drag = response
            .drag_started_by(egui::PointerButton::Primary)
            .then(|| self.handle_drag.as_ref().map(|d| (d.object, d.component, d.gizmo)))
            .flatten();

        // Plain left drag (not claimed by the gizmo or a component handle): orbit
        // the camera around a pivot locked at drag start. Applied HERE, before the
        // first overlay, so the gizmo, icons and handles below are drawn with the
        // orbited camera, the one this frame's viewport texture is rendered with.
        // Everything it depends on is already decided: `orbit_armed` / `gizmo_drag`
        // at drag start (via `pick_preview`, above), `handle_drag` by
        // `gizmo_interaction`, and a screen-UI handle drag (`out.dragging`). The
        // one thing not yet updated is `gizmo.is_focused()`, which is last frame's
        // state: it only matters if the gizmo grabs a drag that `pick_preview`
        // missed, and then only for that first frame.
        if response.dragged_by(egui::PointerButton::Primary)
            && *self.orbit_armed
            && !self.gizmo.is_focused()
            && !*self.gizmo_drag
            && !probe_active
        {
            let delta = response.drag_delta();
            self.orbit_now(delta.x, delta.y, rect);
        }
        if response.drag_stopped_by(egui::PointerButton::Primary) {
            self.orbit_end_now();
        }
        for g in &probe_gestures {
            match *g {
                crate::sky_probe::ProbeAction::Orbit(dx, dy) => self.orbit_now(dx, dy, rect),
                crate::sky_probe::ProbeAction::OrbitEnd => self.orbit_end_now(),
                crate::sky_probe::ProbeAction::Dolly(_) => {}
            }
        }

        // The move / rotate / scale handles always show (the eye and its
        // menu only hide the other gizmos and the billboards).
        if !self.orbiting()
            && !probe_active
            && !river_building
            && !ui_selected_screen
            && !matches!(self.gizmo.tool, GizmoTool::Select | GizmoTool::Rect)
            && let Selection::Object(i) = *self.selection
        {
            {
                let pivot_local = match (self.gizmo.pivot, self.scene.objects[i].render) {
                    (gizmo::PivotMode::Center, Some(render)) => {
                        self.scene.mesh_center_local(render.mesh)
                    }
                    _ => Vec3::ZERO,
                };
                let drag_started = response.drag_started_by(egui::PointerButton::Primary) && !alt;
                let dragging = response.dragged_by(egui::PointerButton::Primary) && !alt;
                // Gizmo operates in world space; parented objects convert
                // back to parent-local afterwards.
                let world = self.scene.world_transform(i);
                let (mut w_scale, mut w_rot, mut w_trans) =
                    zestengine_engine::transform_edit::decompose_trs(world).unwrap_or_else(|| world.to_scale_rotation_translation());
                // The scene now renders into the viewport texture that fills this
                // tab (RTT), so the gizmo's NDC mapping uses the tab rect.
                gizmo_changed = self.gizmo.interact(
                    ui,
                    rect,
                    self.view,
                    self.proj,
                    (&mut w_trans, &mut w_rot, &mut w_scale),
                    pivot_local,
                    cursor,
                    drag_started,
                    dragging,
                );
                if gizmo_changed {
                    let parent_world = self.scene.objects[i]
                        .parent
                        .map_or(glam::Mat4::IDENTITY, |p| self.scene.world_transform(p));
                    // Delta applied to the true world matrix, then taken into the
                    // parent's space: local values stay local (see transform_edit).
                    if let Some((scale, rotation, translation)) =
                        zestengine_engine::transform_edit::gizmo_writeback(parent_world, world, (&w_scale, &w_rot, &w_trans))
                    {
                        let object = &mut self.scene.objects[i];
                        object.translation = translation;
                        object.rotation = rotation;
                        object.scale = scale;
                    }
                }
            }
        }

        // Camera frustum widget: only the selected camera shows its
        // orientation, FOV and framing (near/far rects + up-marker).
        if self.widget_filter.enabled {
            let selected = match self.selection {
                Selection::Object(i) => Some(*i),
                _ => None,
            };
            let view_proj = self.proj * self.view;
            let full_rect = *self.viewport_rect;
            let painter = ui.painter();
            let to_screen = |clip: glam::Vec4| -> egui::Pos2 {
                let ndc = clip.truncate() / clip.w;
                egui::pos2(
                    full_rect.left() + (ndc.x * 0.5 + 0.5) * full_rect.width(),
                    full_rect.top() + (1.0 - (ndc.y * 0.5 + 0.5)) * full_rect.height(),
                )
            };
            for (i, object) in self.scene.objects.iter().enumerate() {
                if !matches!(object.source, zestengine_assets::ObjectSource::Camera) {
                    continue;
                }
                if selected != Some(i) || !self.scene.is_active(i) {
                    continue;
                }
                let camera = object
                    .components
                    .iter()
                    .find_map(|c| c.as_any().downcast_ref::<zestengine_core::CameraComponent>());
                let (fov, near, far) =
                    camera.map_or((60.0, 0.1, 100.0), |c| (c.fov_deg, c.near, c.far));
                // Orthographic cameras draw a box (constant half-height).
                let ortho_half = camera
                    .filter(|c| c.projection == zestengine_core::CameraProjection::Orthographic)
                    .map(|c| c.ortho_size.max(1e-4));
                let world = self.scene.world_transform(i);
                let aspect = (full_rect.width() / full_rect.height().max(1.0)).max(0.1);
                let near = near.max(0.01);
                // Full frustum gets unwieldy with big far planes; cap the
                // widget's reach while the FOV/aspect stay exact.
                let far = far.clamp(near + 0.01, 25.0);
                let corners = |d: f32| {
                    let h = ortho_half.unwrap_or_else(|| (fov.to_radians() * 0.5).tan() * d);
                    let w = h * aspect;
                    // Cameras look down -Z (glTF convention).
                    [
                        world.transform_point3(Vec3::new(-w, h, -d)),
                        world.transform_point3(Vec3::new(w, h, -d)),
                        world.transform_point3(Vec3::new(w, -h, -d)),
                        world.transform_point3(Vec3::new(-w, -h, -d)),
                    ]
                };
                let near_c = corners(near);
                let far_c = corners(far);
                let origin = world.transform_point3(Vec3::ZERO);
                let stroke = egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(110, 255, 140));
                // Lines are clipped against the editor camera's near plane
                // in clip space (partially-behind segments draw instead of
                // vanishing), then against the screen rect; near-plane
                // clipping can yield million-pixel endpoints that the
                // tessellator won't rasterize reliably.
                let line = |a: Vec3, b: Vec3| {
                    const W_EPS: f32 = 0.001;
                    let mut ca = view_proj * a.extend(1.0);
                    let mut cb = view_proj * b.extend(1.0);
                    if ca.w <= W_EPS && cb.w <= W_EPS {
                        return; // fully behind the editor camera
                    }
                    if ca.w < W_EPS {
                        let t = (W_EPS - ca.w) / (cb.w - ca.w);
                        ca += (cb - ca) * t;
                    } else if cb.w < W_EPS {
                        let t = (W_EPS - cb.w) / (ca.w - cb.w);
                        cb += (ca - cb) * t;
                    }
                    if let Some(segment) =
                        clip_segment_to_rect(to_screen(ca), to_screen(cb), full_rect.expand(40.0))
                    {
                        painter.line_segment(segment, stroke);
                    }
                };
                for k in 0..4 {
                    line(near_c[k], near_c[(k + 1) % 4]);
                    line(far_c[k], far_c[(k + 1) % 4]);
                    line(near_c[k], far_c[k]);
                    line(origin, near_c[k]);
                }
                // Up marker: triangle on the near rect's top edge.
                let up_h = ortho_half.unwrap_or_else(|| (fov.to_radians() * 0.5).tan() * near);
                let up_tip = world.transform_point3(Vec3::new(0.0, up_h * 1.8, -near));
                line(near_c[0], up_tip);
                line(near_c[1], up_tip);
            }
        }

        // Rig overlay: the selected character's skeleton coloured by mapping
        // state while the Rig section's "Show skeleton" is on or Configure is open.
        self.scene.rig_overlay_for = match *self.selection {
            Selection::Object(i) if self.rig_ui.overlay || self.rig_ui.configure => Some(i),
            _ => None,
        };
        if (self.rig_ui.overlay || self.rig_ui.configure)
            && let Selection::Object(i) = *self.selection
            && let Some(skel) = self.scene.object_skeleton(i)
        {
            let edited = self
                .rig_ui
                .configure
                .then(|| self.file_meta.as_ref().and_then(|f| f.meta.importer.as_model()).map(|m| &m.rig))
                .flatten()
                .filter(|r| r.human.iter().any(|(_, n)| skel.joints.iter().any(|j| j.name == *n)));
            if let Some(rig) = edited.or(skel.rig.as_ref().filter(|r| r.is_configured())) {
                let selected = self.rig_ui.selected_bone();
                let preview = self.rig_ui.muscle_preview(skel, rig);
                // The Muscles tab's sliders outrank the animation (a model with clips would
                // otherwise always play over them in the editor).
                // The mesh shows the Configure window's pose (reference / sliders) while it is open.
                let preview = preview.or_else(|| self.scene.pose_override.as_ref().filter(|(o, _)| *o == i).map(|(_, l)| l.clone()));
                let live = self.scene.rig_overlay_pose.as_deref().filter(|_| preview.is_none());
                let pose = crate::rig_editor::OverlayPose::select(live, preview.as_deref());
                let configure = self.rig_ui.configure;
                let (report, bad) = self.rig_ui.report(rig, skel);
                crate::rig_editor::draw_overlay(
                    ui.painter(),
                    self.proj * self.view,
                    *self.viewport_rect,
                    self.scene.world_transform(i),
                    skel,
                    rig,
                    report,
                    bad,
                    selected,
                    pose,
                    configure,
                );
            }
        }

        // Co-edit remote rigs: each peer's tracked hands as bone lines and a
        // head circle, one color per peer id (world space, true scale), painted
        // over the viewport like the rig overlay.
        if !self.edit_peers.is_empty() && self.widget_filter.enabled {
            let view_proj = self.proj * self.view;
            let full_rect = *self.viewport_rect;
            let painter = ui.painter();
            let to_screen = |clip: glam::Vec4| -> egui::Pos2 {
                let ndc = clip.truncate() / clip.w;
                egui::pos2(
                    full_rect.left() + (ndc.x * 0.5 + 0.5) * full_rect.width(),
                    full_rect.top() + (1.0 - (ndc.y * 0.5 + 0.5)) * full_rect.height(),
                )
            };
            const PEER_PALETTE: [[u8; 3]; 4] = [[76, 220, 90], [220, 90, 220], [255, 217, 51], [77, 220, 220]];
            for presence in self.edit_peers.values() {
                let rgb = PEER_PALETTE[(presence.peer as usize) % PEER_PALETTE.len()];
                let color = egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
                let stroke = egui::Stroke::new(2.0_f32, color);
                let line = |a: glam::Vec3, b: glam::Vec3| {
                    const W_EPS: f32 = 0.001;
                    let mut ca = view_proj * a.extend(1.0);
                    let mut cb = view_proj * b.extend(1.0);
                    if ca.w <= W_EPS && cb.w <= W_EPS {
                        return;
                    }
                    if ca.w < W_EPS {
                        let t = (W_EPS - ca.w) / (cb.w - ca.w);
                        ca += (cb - ca) * t;
                    } else if cb.w < W_EPS {
                        let t = (W_EPS - cb.w) / (ca.w - cb.w);
                        cb += (ca - cb) * t;
                    }
                    if let Some(segment) =
                        clip_segment_to_rect(to_screen(ca), to_screen(cb), full_rect.expand(40.0))
                    {
                        painter.line_segment(segment, stroke);
                    }
                };
                for hand in presence.hands.iter().flatten() {
                    for &(ja, jb) in zestengine_render::HAND_BONES.iter() {
                        if hand.valid[ja] && hand.valid[jb] {
                            line(glam::Vec3::from(hand.positions[ja]), glam::Vec3::from(hand.positions[jb]));
                        }
                    }
                }
                if let Some((pos, _)) = presence.head {
                    let clip = view_proj * glam::Vec4::new(pos[0], pos[1], pos[2], 1.0);
                    if clip.w > 0.001 {
                        let c = to_screen(clip);
                        // World-space head radius -> screen radius through the
                        // projection's vertical scale at this depth.
                        let px_per_world = (self.proj.y_axis.y * full_rect.height() * 0.5 / clip.w).abs();
                        let r_px = (0.11 * presence.scale.max(0.1)) * px_per_world;
                        painter.circle_stroke(c, r_px, egui::Stroke::new(2.5_f32, color));
                    }
                }
            }
        }

        // Component gizmos for the selected object: resizable boxes (with face
        // handles + optional blend shell), interior point clouds, and range
        // spheres — each component declares its set via `gizmos()` and they are
        // ALL drawn here uniformly (built-ins and plugins alike).
        // Every selected object's gizmos, not just the active one's.
        let gizmo_sels: Vec<usize> = match self.selection {
            _ if !(self.widget_filter.enabled && self.widget_filter.components) => Vec::new(),
            Selection::Object(i) if self.multi_objects.len() > 1 && self.multi_objects.contains(i) => self.multi_objects.clone(),
            Selection::Object(i) => vec![*i],
            _ => Vec::new(),
        };
        for sel in gizmo_sels.into_iter().filter(|&s| s < self.scene.objects.len() && self.scene.is_active(s)) {
            let view_proj = self.proj * self.view;
            let full_rect = *self.viewport_rect;
            let painter = ui.painter();
            const W_EPS: f32 = 0.001;
            let to_screen = |clip: glam::Vec4| -> egui::Pos2 {
                let ndc = clip.truncate() / clip.w;
                egui::pos2(
                    full_rect.left() + (ndc.x * 0.5 + 0.5) * full_rect.width(),
                    full_rect.top() + (1.0 - (ndc.y * 0.5 + 0.5)) * full_rect.height(),
                )
            };
            let project = |p: Vec3| -> Option<egui::Pos2> {
                let clip = view_proj * p.extend(1.0);
                (clip.w > W_EPS).then(|| to_screen(clip))
            };
            let world = self.scene.world_transform(sel);
            let cam_pos = self.view.inverse().w_axis.truncate();
            let line = |a: Vec3, b: Vec3, stroke: egui::Stroke| {
                let mut ca = view_proj * a.extend(1.0);
                let mut cb = view_proj * b.extend(1.0);
                if ca.w <= W_EPS && cb.w <= W_EPS {
                    return;
                }
                if ca.w < W_EPS {
                    let t = (W_EPS - ca.w) / (cb.w - ca.w);
                    ca += (cb - ca) * t;
                } else if cb.w < W_EPS {
                    let t = (W_EPS - cb.w) / (ca.w - cb.w);
                    cb += (ca - cb) * t;
                }
                if let Some(s) =
                    clip_segment_to_rect(to_screen(ca), to_screen(cb), full_rect.expand(40.0))
                {
                    painter.line_segment(s, stroke);
                }
            };
            // Depth-aware line: subdivide the edge and dim the parts whose
            // midpoint sits behind scene geometry (ray-AABB occlusion against the
            // other objects), so a box reads as "behind something" where hidden.
            let occ_line = |a: Vec3, b: Vec3, stroke: egui::Stroke| {
                let dim = egui::Stroke::new(
                    stroke.width,
                    egui::Color32::from_rgb(
                        stroke.color.r() / 3,
                        stroke.color.g() / 3,
                        stroke.color.b() / 3,
                    ),
                );
                const SUB: usize = 10;
                for k in 0..SUB {
                    let t0 = k as f32 / SUB as f32;
                    let t1 = (k + 1) as f32 / SUB as f32;
                    let mid = a.lerp(b, (t0 + t1) * 0.5);
                    let d = mid - cam_pos;
                    let dist = d.length();
                    let occluded = dist > 1.0e-3
                        && self
                            .scene
                            .ray_hit(cam_pos, d / dist)
                            .is_some_and(|h| h < dist - 0.05);
                    line(a.lerp(b, t0), a.lerp(b, t1), if occluded { dim } else { stroke });
                }
            };
            let box_edges = |center: Vec3, half: Vec3, stroke: egui::Stroke| {
                let corner = |sx: f32, sy: f32, sz: f32| {
                    world.transform_point3(
                        center + Vec3::new(half.x * sx, half.y * sy, half.z * sz),
                    )
                };
                let c = [
                    corner(-1.0, -1.0, -1.0),
                    corner(1.0, -1.0, -1.0),
                    corner(1.0, 1.0, -1.0),
                    corner(-1.0, 1.0, -1.0),
                    corner(-1.0, -1.0, 1.0),
                    corner(1.0, -1.0, 1.0),
                    corner(1.0, 1.0, 1.0),
                    corner(-1.0, 1.0, 1.0),
                ];
                for k in 0..4 {
                    occ_line(c[k], c[(k + 1) % 4], stroke);
                    occ_line(c[k + 4], c[(k + 1) % 4 + 4], stroke);
                    occ_line(c[k], c[k + 4], stroke);
                }
            };
            let axes = [Vec3::X, Vec3::Y, Vec3::Z];
            // Shown with the eye on and Component gizmos ticked.
            let show_components = self.widget_filter.enabled && self.widget_filter.components;
            // Mesh Collider: the shape it actually builds (triangles, a hull,
            // or the convex pieces a moving body gets).
            if let Some(mc) = self.scene.objects[sel].components.iter().filter(|_| show_components).find_map(|c| c.as_any().downcast_ref::<zestengine_core::MeshCollider>()) {
                let moving = self.scene.objects[sel]
                    .components
                    .iter()
                    .find_map(|c| c.as_any().downcast_ref::<zestengine_core::RigidBody>())
                    .is_some_and(|rb| rb.kind != zestengine_core::BodyKind::Fixed);
                if let Some(wire) = zestengine_engine::physics::mesh_collider_wire_for(self.scene, sel, zestengine_engine::physics::mesh_shape_mode(self.scene, sel, mc, moving)) {
                    let stroke = egui::Stroke::new(1.0_f32, egui::Color32::from_rgba_unmultiplied(120, 235, 140, 150));
                    for (a, b) in wire.iter() {
                        if let (Some(pa), Some(pb)) = (project(world.transform_point3(*a)), project(world.transform_point3(*b))) {
                            painter.line_segment([pa, pb], stroke);
                        }
                    }
                }
            }
            // Joints: a dashed line from the anchor to the connected body.
            for comp in self.scene.objects[sel].components.iter().filter(|_| show_components) {
                let Some(j) = comp.as_any().downcast_ref::<zestengine_core::PhysicsJoint>() else { continue };
                let Some(id) = j.connected.id() else { continue };
                let Some(other) = self.scene.objects.iter().position(|o| o.id == id) else { continue };
                let a = world.transform_point3(Vec3::from(j.anchor));
                let b = self.scene.world_transform(other).w_axis.truncate();
                if let (Some(pa), Some(pb)) = (project(a), project(b)) {
                    painter.extend(egui::Shape::dashed_line(
                        &[pa, pb],
                        egui::Stroke::new(1.5_f32, egui::Color32::from_rgb(255, 200, 90)),
                        6.0,
                        4.0,
                    ));
                    painter.circle_stroke(pb, 5.0, egui::Stroke::new(1.5_f32, egui::Color32::from_rgb(255, 200, 90)));
                }
            }
            for comp in self.scene.objects[sel].components.iter().filter(|_| show_components) {
                for spec in self.editor_components.gizmos(comp.as_ref()) {
                    match spec {
                        crate::GizmoSpec::Box {
                            center,
                            half_extents,
                            blend,
                            color,
                            ..
                        } => {
                            box_edges(center, half_extents, egui::Stroke::new(1.5_f32, color));
                            if blend.abs() > 0.001 {
                                // Blend is in world units; the box is drawn in object space.
                                let scale = Vec3::new(
                                    world.x_axis.truncate().length(),
                                    world.y_axis.truncate().length(),
                                    world.z_axis.truncate().length(),
                                )
                                .max(Vec3::splat(1e-4));
                                let inner = (half_extents - Vec3::splat(blend) / scale).max(Vec3::ZERO);
                                let dim = egui::Color32::from_rgb(
                                    color.r() / 2,
                                    color.g() / 2,
                                    color.b() / 2,
                                );
                                box_edges(center, inner, egui::Stroke::new(1.0_f32, dim));
                            }
                            let center_screen = project(world.transform_point3(center));
                            for axis in 0..3 {
                                for sign in [-1.0f32, 1.0] {
                                    let fc = world.transform_point3(
                                        center + axes[axis] * (sign * half_extents[axis]),
                                    );
                                    let Some(p) = project(fc) else { continue };
                                    let outward = center_screen
                                        .map(|cs| p - cs)
                                        .filter(|v| v.length() > 1.0)
                                        .or_else(|| {
                                            project(fc + world.transform_vector3(axes[axis]) * sign)
                                                .map(|a| a - p)
                                        })
                                        .unwrap_or(egui::vec2(0.0, -1.0));
                                    let hovered = cursor.is_some_and(|cu| {
                                        handle_pick_dist(cu, p, outward, 26.0) <= 12.0
                                    });
                                    let (scale, col) = handle_style(hovered, color);
                                    draw_arrow_handle(painter, p, outward, scale, col);
                                }
                            }
                        }
                        crate::GizmoSpec::Handle { position, anchor, color } => {
                            let (Some(p), a) = (project(world.transform_point3(position)), project(world.transform_point3(anchor)))
                            else {
                                continue;
                            };
                            if let Some(a) = a {
                                painter.extend(egui::Shape::dashed_line(&[a, p], egui::Stroke::new(1.0_f32, color.gamma_multiply(0.6)), 4.0, 3.0));
                            }
                            let hovered = cursor.is_some_and(|cu| cu.distance(p) <= 12.0);
                            let r = if hovered { 8.0 } else { 6.5 };
                            painter.circle(p, r, color.gamma_multiply(0.35), egui::Stroke::new(2.0_f32, color));
                            painter.circle_filled(p, 2.0, egui::Color32::WHITE);
                        }
                        crate::GizmoSpec::Lines { segments, color, world: in_world } => {
                            let stroke = egui::Stroke::new(1.5_f32, color);
                            for (a, b) in segments {
                                if in_world {
                                    line(a, b, stroke);
                                } else {
                                    line(world.transform_point3(a), world.transform_point3(b), stroke);
                                }
                            }
                        }
                        crate::GizmoSpec::Points { positions, color } => {
                            for local in positions {
                                if let Some(p) = project(world.transform_point3(local))
                                    && full_rect.contains(p)
                                {
                                    painter.circle_filled(p, 1.6, color);
                                }
                            }
                        }
                        crate::GizmoSpec::Range {
                            center,
                            radius,
                            color,
                        } => {
                            let origin = world.transform_point3(center);
                            let stroke = egui::Stroke::new(1.5_f32, color);
                            let circle = |a: Vec3, b: Vec3| {
                                const SEG: usize = 32;
                                let mut prev = origin + a * radius;
                                for k in 1..=SEG {
                                    let ang = k as f32 / SEG as f32 * std::f32::consts::TAU;
                                    let q = origin + (a * ang.cos() + b * ang.sin()) * radius;
                                    line(prev, q, stroke);
                                    prev = q;
                                }
                            };
                            circle(Vec3::X, Vec3::Y);
                            circle(Vec3::Y, Vec3::Z);
                            circle(Vec3::X, Vec3::Z);
                            let right = self.camera_right();
                            let hp = origin + right * radius;
                            if let Some(p) = project(hp) {
                                let outward = project(hp + right)
                                    .map(|a| a - p)
                                    .unwrap_or(egui::vec2(1.0, 0.0));
                                let hovered = cursor.is_some_and(|cu| {
                                    handle_pick_dist(cu, p, outward, 26.0) <= 12.0
                                });
                                let (scale, col) = handle_style(hovered, color);
                                draw_arrow_handle(painter, p, outward, scale, col);
                            }
                        }
                    }
                }
            }
        }

        // Billboards: a drawn bulb at every light and a drawn camera at every
        // camera so they read as real icons (egui's font has no 💡/🎥 glyphs).
        // `billboards` records the (screen pos, object index) click targets.
        let mut billboards: Vec<(egui::Pos2, usize)> = Vec::new();
        let mut icon_drawn: Vec<usize> = Vec::new();
        {
            let view_proj = self.proj * self.view;
            let full_rect = *self.viewport_rect;
            let painter = ui.painter();
            for (i, object) in self.scene.objects.iter().enumerate() {
                if !self.scene.is_active(i) || self.scene.view_hidden(i) {
                    continue; // no widgets for disabled or scene-hidden objects
                }
                let is_light = object.components.iter().any(|c| {
                    c.as_any()
                        .downcast_ref::<zestengine_core::LightComponent>()
                        .is_some()
                });
                let is_camera = matches!(object.source, zestengine_assets::ObjectSource::Camera);
                let is_probe = object.components.iter().any(|c| {
                    c.as_any()
                        .downcast_ref::<zestengine_core::LightProbeVolume>()
                        .is_some()
                });
                let is_refl_probe = object.components.iter().any(|c| {
                    c.as_any()
                        .downcast_ref::<zestengine_core::ReflectionProbe>()
                        .is_some()
                });
                let is_audio = object.components.iter().any(|c| {
                    c.as_any()
                        .downcast_ref::<zestengine_core::AudioSource>()
                        .is_some()
                });
                let is_flux_volume = object.components.iter().any(|c| {
                    c.as_any()
                        .downcast_ref::<zestengine_core::FluxVolume>()
                        .is_some()
                });
                let is_canvas = object
                    .components
                    .iter()
                    .any(|c| c.as_any().downcast_ref::<zestengine_core::UiCanvas>().is_some());
                let is_post = object
                    .components
                    .iter()
                    .any(|c| c.as_any().is::<zestengine_core::VolumeComponent>());
                let is_wind = object.components.iter().any(|c| c.as_any().downcast_ref::<zestengine_core::ClothWind>().is_some());
                let is_nav = object.components.iter().any(|c| c.as_any().downcast_ref::<zestengine_core::NavMeshSurface>().is_some());
                let is_particle = object.components.iter().any(|c| c.as_any().downcast_ref::<zestengine_core::ParticleEmitter>().is_some());
                let is_portal = object.components.iter().any(|c| c.as_any().downcast_ref::<zestengine_core::ScenePortal>().is_some());
                let is_terrain = object.components.iter().any(|c| c.as_any().downcast_ref::<zestengine_core::Terrain>().is_some());
                if !is_post
                    && !is_light
                    && !is_camera
                    && !is_probe
                    && !is_refl_probe
                    && !is_audio
                    && !is_flux_volume
                    && !is_canvas
                    && !is_wind
                    && !is_nav
                    && !is_particle
                    && !is_portal
                    && !is_terrain
                {
                    continue;
                }
                let pos = self.scene.world_transform(i).w_axis.truncate();
                let clip = view_proj * pos.extend(1.0);
                if clip.w <= 0.001 {
                    continue; // behind the camera
                }
                let ndc = clip.truncate() / clip.w;
                let screen = egui::pos2(
                    full_rect.left() + (ndc.x * 0.5 + 0.5) * full_rect.width(),
                    full_rect.top() + (1.0 - (ndc.y * 0.5 + 0.5)) * full_rect.height(),
                );
                if !full_rect.contains(screen) {
                    continue;
                }
                let selected = matches!(*self.selection, Selection::Object(s) if s == i)
                    || (self.multi_objects.len() > 1 && self.multi_objects.contains(&i));
                // Icon priority: light, probe volume (incl. FluxVolume), reflection
                // probe, audio, UI canvas, camera.
                let setting = if is_light {
                    self.widget_filter.lights
                } else if is_probe || is_flux_volume {
                    self.widget_filter.probes
                } else if is_refl_probe {
                    self.widget_filter.reflection_probes
                } else if is_audio {
                    self.widget_filter.audio
                } else if is_canvas {
                    self.widget_filter.ui_canvases
                } else if is_post {
                    self.widget_filter.post_volumes
                } else if is_wind || is_nav || is_particle || is_portal || is_terrain {
                    self.widget_filter.placeables
                } else {
                    self.widget_filter.cameras
                };
                // Master eye off hides everything; otherwise filtered-off
                // billboards still show for the selected object.
                if !self.widget_filter.enabled || (!setting.visible && !selected) {
                    continue;
                }
                if is_light {
                    draw_light_icon(painter, screen, selected, setting.size);
                } else if is_probe || is_flux_volume {
                    draw_probe_icon(painter, screen, selected, setting.size);
                } else if is_refl_probe {
                    draw_reflection_probe_icon(painter, screen, selected, setting.size);
                } else if is_audio {
                    draw_audio_icon(painter, screen, selected, setting.size);
                } else if is_canvas {
                    draw_ui_canvas_icon(painter, screen, selected, setting.size);
                } else if is_post {
                    draw_post_volume_icon(painter, screen, selected, setting.size);
                } else if is_wind {
                    draw_wind_icon(painter, screen, selected, setting.size);
                } else if is_nav {
                    draw_nav_icon(painter, screen, selected, setting.size);
                } else if is_particle {
                    draw_particle_icon(painter, screen, selected, setting.size);
                } else if is_portal {
                    draw_portal_icon(painter, screen, selected, setting.size);
                } else if is_terrain {
                    draw_terrain_icon(painter, screen, selected, setting.size);
                } else {
                    draw_camera_icon(painter, screen, selected, setting.size);
                }
                if selected && crate::theme::selection_outline_enabled() {
                    draw_selection_ring(painter, screen, setting.size);
                }
                icon_drawn.push(i);
                // Locked objects keep their icon but a click passes through.
                if self.scene.view_pickable(i) {
                    billboards.push((screen, i));
                }
            }
        }

        // Selected objects that draw nothing and have no billboard (empties,
        // groups, collider-only objects, bones...): a marker at the origin so the
        // selection is always visible. Mesh objects get the renderer's outline.
        if self.widget_filter.enabled && crate::theme::selection_outline_enabled() {
            let view_proj = self.proj * self.view;
            let full_rect = *self.viewport_rect;
            let selected: Vec<usize> = match *self.selection {
                Selection::Object(i) if self.multi_objects.len() > 1 && self.multi_objects.contains(&i) => self.multi_objects.clone(),
                Selection::Object(i) => vec![i],
                _ => Vec::new(),
            };
            for i in selected {
                if i >= self.scene.objects.len() || icon_drawn.contains(&i) || !self.scene.is_active(i) {
                    continue;
                }
                if self.scene.objects[i].render.is_some() || self.scene.sprite_outline(i).is_some() {
                    continue; // outlined by the renderer / drawn as a sprite rect
                }
                let pos = self.scene.world_transform(i).w_axis.truncate();
                let clip = view_proj * pos.extend(1.0);
                if clip.w <= 0.001 {
                    continue;
                }
                let ndc = clip.truncate() / clip.w;
                let screen = egui::pos2(
                    full_rect.left() + (ndc.x * 0.5 + 0.5) * full_rect.width(),
                    full_rect.top() + (1.0 - (ndc.y * 0.5 + 0.5)) * full_rect.height(),
                );
                if full_rect.expand(20.0).contains(screen) {
                    draw_selection_marker(ui.painter(), screen);
                }
            }
        }

        // Light gizmo for the selected object whenever it carries a Light
        // component (not only dedicated Light objects). Directional/spot show
        // their aim; point/spot show their range; spot shows its cone.
        let light_sels: Vec<usize> = match *self.selection {
            _ if !(self.widget_filter.enabled && self.widget_filter.components) => Vec::new(),
            Selection::Object(i) if self.multi_objects.len() > 1 && self.multi_objects.contains(&i) => self.multi_objects.clone(),
            Selection::Object(i) => vec![i],
            _ => Vec::new(),
        };
        for i in light_sels {
            if i >= self.scene.objects.len() || !self.scene.is_active(i) {
                continue;
            }
            let Some(light) = self.scene.objects[i].components.iter().find_map(|c| c.as_any().downcast_ref::<zestengine_core::LightComponent>()) else {
                continue;
            };
            use zestengine_core::LightKind;
            let view_proj = self.proj * self.view;
            let full_rect = *self.viewport_rect;
            let painter = ui.painter();
            let stroke = egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(255, 226, 120));
            let to_screen = |clip: glam::Vec4| -> egui::Pos2 {
                let ndc = clip.truncate() / clip.w;
                egui::pos2(
                    full_rect.left() + (ndc.x * 0.5 + 0.5) * full_rect.width(),
                    full_rect.top() + (1.0 - (ndc.y * 0.5 + 0.5)) * full_rect.height(),
                )
            };
            let line = |a: Vec3, b: Vec3| {
                const W_EPS: f32 = 0.001;
                let mut ca = view_proj * a.extend(1.0);
                let mut cb = view_proj * b.extend(1.0);
                if ca.w <= W_EPS && cb.w <= W_EPS {
                    return;
                }
                if ca.w < W_EPS {
                    let t = (W_EPS - ca.w) / (cb.w - ca.w);
                    ca += (cb - ca) * t;
                } else if cb.w < W_EPS {
                    let t = (W_EPS - cb.w) / (ca.w - cb.w);
                    cb += (ca - cb) * t;
                }
                if let Some(segment) =
                    clip_segment_to_rect(to_screen(ca), to_screen(cb), full_rect.expand(40.0))
                {
                    painter.line_segment(segment, stroke);
                }
            };

            let world = self.scene.world_transform(i);
            let origin = world.transform_point3(Vec3::ZERO);
            let (_, rotation, _) = world.to_scale_rotation_translation();
            let fwd = rotation * Vec3::NEG_Z;
            let right = rotation * Vec3::X;
            let up = rotation * Vec3::Y;
            // Wire circle of `radius` centered at `c`, in the plane
            // spanned by axes `a`/`b`.
            let circle = |c: Vec3, a: Vec3, b: Vec3, radius: f32| {
                const SEG: usize = 24;
                let mut prev = c + a * radius;
                for k in 1..=SEG {
                    let ang = k as f32 / SEG as f32 * std::f32::consts::TAU;
                    let p = c + (a * ang.cos() + b * ang.sin()) * radius;
                    line(prev, p);
                    prev = p;
                }
            };

            match light.kind {
                LightKind::Directional => {
                    // A bundle of parallel rays through a small disc.
                    let len = 2.0;
                    let r = 0.35;
                    circle(origin, right, up, r);
                    for k in 0..8 {
                        let ang = k as f32 / 8.0 * std::f32::consts::TAU;
                        let off = (right * ang.cos() + up * ang.sin()) * r;
                        line(origin + off, origin + off + fwd * len);
                    }
                    line(origin, origin + fwd * (len + 0.3));
                }
                LightKind::Point => {
                    let radius = light.range.min(25.0);
                    circle(origin, right, up, radius);
                    circle(origin, right, fwd, radius);
                    circle(origin, up, fwd, radius);
                }
                LightKind::Spot => {
                    let dist = light.range.min(25.0);
                    let half = (light.spot_angle.to_radians() * 0.5).tan();
                    let end = origin + fwd * dist;
                    let rad = half * dist;
                    circle(end, right, up, rad);
                    for k in 0..4 {
                        let ang = k as f32 / 4.0 * std::f32::consts::TAU;
                        let rim = end + (right * ang.cos() + up * ang.sin()) * rad;
                        line(origin, rim);
                    }
                }
            }
        }

        // (AudioSource ranges + collider box/sphere are now drawn by the generic
        // component-gizmo pass above via `gizmos()`.)

        // 2D collider outlines of bodies with Debug Draw on (filled by Play).
        if !self.scene.debug_lines2d.is_empty() {
            let view_proj = self.proj * self.view;
            let full_rect = *self.viewport_rect;
            let painter = ui.painter();
            let stroke = egui::Stroke::new(1.5_f32, egui::Color32::from_rgb(120, 255, 160));
            let to_screen = |clip: glam::Vec4| -> egui::Pos2 {
                let ndc = clip.truncate() / clip.w;
                egui::pos2(
                    full_rect.left() + (ndc.x * 0.5 + 0.5) * full_rect.width(),
                    full_rect.top() + (1.0 - (ndc.y * 0.5 + 0.5)) * full_rect.height(),
                )
            };
            for &(a, b) in &self.scene.debug_lines2d {
                const W_EPS: f32 = 0.001;
                let mut ca = view_proj * a.extend(1.0);
                let mut cb = view_proj * b.extend(1.0);
                if ca.w <= W_EPS && cb.w <= W_EPS {
                    continue;
                }
                if ca.w < W_EPS {
                    let t = (W_EPS - ca.w) / (cb.w - ca.w);
                    ca += (cb - ca) * t;
                } else if cb.w < W_EPS {
                    let t = (W_EPS - cb.w) / (ca.w - cb.w);
                    cb += (ca - cb) * t;
                }
                if let Some(seg) = clip_segment_to_rect(to_screen(ca), to_screen(cb), full_rect) {
                    painter.line_segment(seg, stroke);
                }
            }
        }

        // Selected sprites, sprite shapes and tilemaps: their rect outline (the 2D
        // selection outline, drawn in 2D and 3D views) and a pivot cross, in the
        // selection colour (the primary full strength, other selected ones softer).
        if crate::theme::selection_outline_enabled() {
            let view_proj = self.proj * self.view;
            let full_rect = *self.viewport_rect;
            let painter = ui.painter();
            let to_screen = |p: Vec3| -> Option<egui::Pos2> {
                let clip = view_proj * p.extend(1.0);
                if clip.w <= 0.001 {
                    return None;
                }
                let ndc = clip.truncate() / clip.w;
                Some(egui::pos2(
                    full_rect.left() + (ndc.x * 0.5 + 0.5) * full_rect.width(),
                    full_rect.top() + (1.0 - (ndc.y * 0.5 + 0.5)) * full_rect.height(),
                ))
            };
            let all: Vec<usize> = match *self.selection {
                Selection::Object(i) if self.multi_objects.len() > 1 && self.multi_objects.contains(&i) => {
                    let mut v = vec![i];
                    v.extend(self.multi_objects.iter().copied().filter(|&o| o != i));
                    v
                }
                Selection::Object(i) => vec![i],
                _ => Vec::new(),
            };
            for (n, i) in all.into_iter().enumerate() {
                if i >= self.scene.objects.len() || !self.scene.is_active(i) {
                    continue;
                }
                let Some(rect) = self.scene.sprite_outline(i) else { continue };
                let col = crate::theme::selection_outline();
                let col = if n == 0 { col } else { crate::theme::with_alpha(col, 0.6) };
                let stroke = egui::Stroke::new(2.0_f32, col);
                for k in 0..4 {
                    if let (Some(a), Some(b)) = (to_screen(rect[k]), to_screen(rect[(k + 1) % 4])) {
                        painter.line_segment([a, b], stroke);
                    }
                }
                let origin = self.scene.world_transform(i).w_axis.truncate();
                if n == 0
                    && let Some(c) = to_screen(origin)
                {
                    painter.line_segment([c - egui::vec2(5.0, 0.0), c + egui::vec2(5.0, 0.0)], stroke);
                    painter.line_segment([c - egui::vec2(0.0, 5.0), c + egui::vec2(0.0, 5.0)], stroke);
                }
            }
        }

        // Mesh-collider AABB (yellow): not editable (follows the mesh), so it's
        // not a `gizmos()` resize box — kept as a dedicated read-only draw.
        if let Selection::Object(i) = *self.selection
            && self.scene.is_active(i)
            && self.widget_filter.enabled
            && self.widget_filter.components
        {
            const YELLOW: egui::Color32 = egui::Color32::from_rgb(240, 220, 70);
            let world = self.scene.world_transform(i);
            let view_proj = self.proj * self.view;
            let full_rect = *self.viewport_rect;
            let painter = ui.painter();
            let to_screen = |clip: glam::Vec4| -> egui::Pos2 {
                let ndc = clip.truncate() / clip.w;
                egui::pos2(
                    full_rect.left() + (ndc.x * 0.5 + 0.5) * full_rect.width(),
                    full_rect.top() + (1.0 - (ndc.y * 0.5 + 0.5)) * full_rect.height(),
                )
            };
            let line = |a: Vec3, b: Vec3, stroke: egui::Stroke| {
                const W_EPS: f32 = 0.001;
                let mut ca = view_proj * a.extend(1.0);
                let mut cb = view_proj * b.extend(1.0);
                if ca.w <= W_EPS && cb.w <= W_EPS {
                    return;
                }
                if ca.w < W_EPS {
                    let t = (W_EPS - ca.w) / (cb.w - ca.w);
                    ca += (cb - ca) * t;
                } else if cb.w < W_EPS {
                    let t = (W_EPS - cb.w) / (ca.w - cb.w);
                    cb += (ca - cb) * t;
                }
                if let Some(seg) = clip_segment_to_rect(to_screen(ca), to_screen(cb), full_rect) {
                    painter.line_segment(seg, stroke);
                }
            };
            let stroke = egui::Stroke::new(1.4_f32, YELLOW);
            let obj = &self.scene.objects[i];

            // Mesh collider: the mesh's AABB (follows the mesh, not editable).
            if obj
                .components
                .iter()
                .any(|c| c.as_any().downcast_ref::<zestengine_core::MeshCollider>().is_some())
                && let Some((min, max)) = self.scene.render_mesh_bounds(i)
            {
                let corner = |sx: f32, sy: f32, sz: f32| {
                    world.transform_point3(Vec3::new(
                        if sx < 0.0 { min.x } else { max.x },
                        if sy < 0.0 { min.y } else { max.y },
                        if sz < 0.0 { min.z } else { max.z },
                    ))
                };
                let c = [
                    corner(-1.0, -1.0, -1.0),
                    corner(1.0, -1.0, -1.0),
                    corner(1.0, 1.0, -1.0),
                    corner(-1.0, 1.0, -1.0),
                    corner(-1.0, -1.0, 1.0),
                    corner(1.0, -1.0, 1.0),
                    corner(1.0, 1.0, 1.0),
                    corner(-1.0, 1.0, 1.0),
                ];
                for k in 0..4 {
                    line(c[k], c[(k + 1) % 4], stroke);
                    line(c[k + 4], c[(k + 1) % 4 + 4], stroke);
                    line(c[k], c[k + 4], stroke);
                }
            }
        }

        // Orientation cross: three short gray lines along the selected
        // object's LOCAL axes, crossing at the pivot, so its orientation is
        // readable under the move/scale gizmo. Editor-camera overlay only.
        // Hidden when gizmos are globally off (eye button) or its own toggle is off.
        if self.widget_filter.enabled
            && self.widget_filter.cross
            && let Selection::Object(i) = *self.selection
            && matches!(self.gizmo.tool, GizmoTool::Move | GizmoTool::Scale)
            && self.scene.is_active(i)
        {
            let world = self.scene.world_transform(i);
            let origin = world.w_axis.truncate();
            // Follow the gizmo's orientation mode: object rotation in Local,
            // world axes in Global.
            let rot = if self.gizmo.local_orientation {
                world.to_scale_rotation_translation().1
            } else {
                glam::Quat::IDENTITY
            };
            let cam = self.view.inverse().w_axis.truncate();
            // Screen-constant length (~world units that project to a small,
            // steady on-screen size).
            let len = if zestengine_core::is_orthographic(&self.proj) {
                // Orthographic: screen size depends on zoom, not distance.
                (1.0 / self.proj.y_axis.y.abs().max(1e-6)) * 0.25
            } else {
                (origin - cam).length() * 0.12
            }
            .clamp(0.05, 50.0);
            let view_proj = self.proj * self.view;
            let full_rect = *self.viewport_rect;
            let painter = ui.painter();
            let to_screen = |clip: glam::Vec4| -> egui::Pos2 {
                let ndc = clip.truncate() / clip.w;
                egui::pos2(
                    full_rect.left() + (ndc.x * 0.5 + 0.5) * full_rect.width(),
                    full_rect.top() + (1.0 - (ndc.y * 0.5 + 0.5)) * full_rect.height(),
                )
            };
            let stroke = egui::Stroke::new(1.0_f32, egui::Color32::from_gray(150));
            let line = |a: Vec3, b: Vec3| {
                const W_EPS: f32 = 0.001;
                let mut ca = view_proj * a.extend(1.0);
                let mut cb = view_proj * b.extend(1.0);
                if ca.w <= W_EPS && cb.w <= W_EPS {
                    return;
                }
                if ca.w < W_EPS {
                    let t = (W_EPS - ca.w) / (cb.w - ca.w);
                    ca += (cb - ca) * t;
                } else if cb.w < W_EPS {
                    let t = (W_EPS - cb.w) / (ca.w - cb.w);
                    cb += (ca - cb) * t;
                }
                if let Some(seg) = clip_segment_to_rect(to_screen(ca), to_screen(cb), full_rect) {
                    painter.line_segment(seg, stroke);
                }
            };
            for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
                let d = rot * (axis * len);
                line(origin - d, origin + d);
            }
        }

        // The selected river's path editing: handle clicks pick its points and,
        // while building, clicks lay new ones instead of picking objects.
        let mut river_click = false;
        if let Selection::Object(i) = *self.selection
            && !self.playing
            && self.widget_filter.enabled
            && self.scene.is_active(i)
        {
            let view = crate::river_edit::View::new(self.view, self.proj, *self.viewport_rect);
            let new_drag = new_handle_drag.filter(|d| d.0 == i).map(|d| (d.1, d.2));
            let click = response
                .clicked()
                .then(|| response.interact_pointer_pos())
                .flatten()
                .filter(|_| !ui_clicked && !self.looking && !self.orbiting());
            let dragging = response.dragged() || self.handle_drag.is_some();
            river_click = crate::river_edit::viewport(ui, self.scene, i, &view, cursor, new_drag, click, dragging);
            let view = crate::cloth_paint::View::new(self.view, self.proj, *self.viewport_rect);
            river_click |= crate::cloth_paint::viewport(ui, self.scene, i, &view, &response);
            river_click |= crate::terrain_paint::viewport(ui, self.scene, i, &view, &response);
            river_click |= crate::shape_edit::viewport(ui, self.scene, i, &view, &response);
        }
        {
            let tview = crate::cloth_paint::View::new(self.view, self.proj, *self.viewport_rect);
            river_click |= crate::tile_edit::viewport(ui, self.scene, &tview, &response, self.playing);
        }

        // Click-to-pick on release-without-drag, unless the gizmo owns the
        // cursor or we're flying.
        let gizmo_busy =
            self.gizmo.is_focused() || cursor.is_some_and(|p| self.gizmo.pick_preview(p));
        if response.clicked()
            && !river_click
            && !ui_clicked
            && !gizmo_busy
            && !gizmo_changed
            && !self.looking
            && let Some(pos) = response.interact_pointer_pos()
        {
            // A click near a camera/light billboard selects it directly;
            // these objects have no mesh AABB to ray-pick against.
            let hit = billboards
                .iter()
                .map(|(p, i)| (p.distance(pos), *i))
                .filter(|(d, _)| *d <= 16.0)
                .min_by(|a, b| a.0.total_cmp(&b.0))
                .map(|(_, i)| i);
            match hit {
                Some(i) => *self.selection = Selection::Object(i),
                None => self.actions.push(EditorAction::PickAt(pos)),
            }
        }

        // Viewport overlays: gizmo tool buttons (top-left), pivot /
        // orientation / snap controls (top-center).
        self.viewport_overlays(ui, rect);

        // Selected-camera preview (bottom-right): live view through the camera
        // being edited, so its framing can be tweaked from the viewport.
        if self.camera_overlay
            && let Some((texture, size)) = self.camera_preview
        {
            let w = 260.0;
            let h = w / (size[0] / size[1].max(1.0));
            egui::Area::new(ui.id().with("vp-cam-overlay"))
                .order(egui::Order::Middle)
                .pivot(egui::Align2::RIGHT_BOTTOM)
                .fixed_pos(rect.right_bottom() + egui::vec2(-8.0, -8.0))
                .show(ui.ctx(), |ui| {
                    egui::Frame::popup(ui.style()).show(ui, |ui| {
                        ui.label(egui::RichText::new(format!("{} Camera", ph::CAMERA)).small().weak());
                        ui.add(
                            egui::Image::new(egui::load::SizedTexture::new(
                                texture,
                                egui::vec2(w, h),
                            ))
                            .maintain_aspect_ratio(true),
                        );
                    });
                });
        }

        // Drop from the Files panel onto the viewport: a `.material` assigns to
        // the hit mesh; a model file (FBX/glTF) imports into the scene.
        if let Some(payload) = response.dnd_release_payload::<PathBuf>() {
            let ext = payload
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_ascii_lowercase());
            if let Some("material") = ext.as_deref()
                && let Some(pos) = ui.input(|i| i.pointer.latest_pos())
            {
                self.actions
                    .push(EditorAction::AssignMaterialAt(pos, (*payload).clone()));
            }
        }
    }

    /// Viewport overlays (spec §6): tab-strip controls, tool and transform
    /// troughs, playing chip, orientation widget, stats pill, play outline.
    fn viewport_overlays(&mut self, ui: &mut egui::Ui, rect: egui::Rect) {
        use crate::{icons::Icon, theme, widgets};
        let ctx = ui.ctx().clone();

        // Occlusion visualization: baked bounds + the culling camera's cell.
        if self.occlusion_ui.show_cells
            && let Some(o) = &self.scene.occlusion
        {
            let painter = ui.painter_at(rect);
            let wire = |mn: Vec3, mx: Vec3, stroke: egui::Stroke| {
                let c = |i: usize| Vec3::new(
                    if i & 1 == 0 { mn.x } else { mx.x },
                    if i & 2 == 0 { mn.y } else { mx.y },
                    if i & 4 == 0 { mn.z } else { mx.z },
                );
                for (a, b) in [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)] {
                    if let (Some(p), Some(q)) = (self.world_to_screen(c(a), rect), self.world_to_screen(c(b), rect)) {
                        painter.line_segment([p, q], stroke);
                    }
                }
            };
            let (bmn, bmx) = o.bounds();
            wire(bmn, bmx, egui::Stroke::new(1.0_f32, theme::with_alpha(theme::TEXT_MUTED, 0.6)));
            if let Some(cell) = self.scene.occlusion_camera.and_then(|c| o.cell_of(c)) {
                let (mn, mx) = o.cell_bounds(cell);
                wire(mn, mx, egui::Stroke::new(1.5_f32, theme::accent()));
            }
        }

        // Right side of the viewport tab strip: shading, gizmos, stats, view.
        let strip = egui::Area::new(ui.id().with("vp-strip"))
            .order(egui::Order::Middle)
            .pivot(egui::Align2::RIGHT_CENTER)
            .fixed_pos(egui::pos2(rect.right() - 8.0, rect.top() - theme::H_TAB_STRIP * 0.5))
            .show(&ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 6.0;
                    const SHADING: [(u32, &str); 8] = [
                        (0, "Lit"),
                        (3, "Unlit"),
                        (2, "Lighting only"),
                        (1, "Normals"),
                        (4, "Reflection cube"),
                        (5, "Reflection vector"),
                        (7, "Shadow layers"),
                        (zestengine_render::VIEW_AO, "Ambient occlusion"),
                    ];
                    let mode = &mut *self.gi_debug;
                    let current = SHADING.iter().chain(zestengine_render::ZESTLET_VIEW_MODES.iter()).find(|(m, _)| *m == *mode).map_or("Lit", |(_, l)| l);
                    egui::ComboBox::from_id_salt("vp-shading").width(120.0).selected_text(current).show_ui(ui, |ui| {
                        for (m, label) in SHADING {
                            ui.selectable_value(mode, m, label);
                        }
                        ui.separator();
                        for (m, label) in zestengine_render::ZESTLET_VIEW_MODES {
                            ui.selectable_value(mode, m, label);
                        }
                    });
                    widgets::trough(ui, |ui| {
                        // Split button: the eye shows / hides every gizmo and
                        // billboard, the arrow picks which kinds show.
                        ui.spacing_mut().item_spacing.x = 0.0;
                        let shown = self.widget_filter.enabled;
                        let (icon, tip) = if shown { (Icon::Eye, "Hide gizmos and billboards") } else { (Icon::EyeOff, "Show gizmos and billboards") };
                        if widgets::icon_button(ui, icon, tip, shown).clicked() {
                            self.widget_filter.enabled = !shown;
                        }
                        // Highlighted while its dropdown is open (as of last frame).
                        let open_id = egui::Id::new("vp-widget-filter-open");
                        let was_open = ui.data(|d| d.get_temp::<bool>(open_id).unwrap_or(false));
                        let arrow = widgets::dropdown_arrow_button(ui, "Choose what shows in the viewport", was_open);
                        egui::Popup::menu(&arrow)
                            .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
                            .show(|ui| billboard_menu(ui, self.widget_filter, &mut self.project.settings));
                        let open = egui::Popup::is_id_open(ui.ctx(), egui::Popup::default_response_id(&arrow));
                        ui.data_mut(|d| d.insert_temp(open_id, open));
                        ui.spacing_mut().item_spacing.x = 4.0;
                        if widgets::icon_button(ui, Icon::Stats, "Stats", *self.viewport_stats).clicked() {
                            *self.viewport_stats = !*self.viewport_stats;
                        }
                    });
                    egui::ComboBox::from_id_salt("vp-view").width(100.0).selected_text("Perspective").show_ui(ui, |ui| {
                        for (label, dir) in [
                            ("Top", glam::Vec3::NEG_Y),
                            ("Front", glam::Vec3::NEG_Z),
                            ("Right", glam::Vec3::NEG_X),
                        ] {
                            if ui.button(label).clicked() {
                                self.actions.push(EditorAction::SnapView(dir));
                                ui.close();
                            }
                        }
                    });
                });
            });
        *self.vp_strip_width = strip.response.rect.width() + 16.0;

        // Top-left: tools, transform options, playing chip (wraps, keeping
        // 96 px clear for the orientation widget).
        let left_w = (rect.width() - 96.0 - 24.0).max(120.0);
        egui::Area::new(ui.id().with("vp-tools"))
            .order(egui::Order::Middle)
            .fixed_pos(rect.left_top() + egui::vec2(12.0, 12.0))
            .show(&ctx, |ui| {
                ui.set_max_width(left_w);
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(8.0, 8.0);
                    widgets::trough(ui, |ui| {
                        for (tool, icon, hint) in [
                            (GizmoTool::Select, Icon::ToolSelect, "Select (Q)"),
                            (GizmoTool::Move, Icon::ToolMove, "Move (W)"),
                            (GizmoTool::Rotate, Icon::ToolRotate, "Rotate (E)"),
                            (GizmoTool::Scale, Icon::ToolScale, "Scale (R)"),
                            (GizmoTool::Rect, Icon::ToolRect, "Rect (T)"),
                        ] {
                            if widgets::accent_icon_button(ui, icon, hint, self.gizmo.tool == tool, egui::vec2(30.0, 28.0)).clicked() {
                                self.gizmo.tool = tool;
                            }
                        }
                    });
                    widgets::trough(ui, |ui| {
                        if widgets::segment(ui, self.gizmo.pivot == gizmo::PivotMode::Origin, "Pivot")
                            .on_hover_text("Transform around the object's origin")
                            .clicked()
                        {
                            self.gizmo.pivot = gizmo::PivotMode::Origin;
                        }
                        if widgets::segment(ui, self.gizmo.pivot == gizmo::PivotMode::Center, "Center")
                            .on_hover_text("Transform around the mesh bounds center")
                            .clicked()
                        {
                            self.gizmo.pivot = gizmo::PivotMode::Center;
                        }
                        trough_divider(ui);
                        if widgets::segment(ui, !self.gizmo.local_orientation, "Global").clicked() {
                            self.gizmo.local_orientation = false;
                        }
                        if widgets::segment(ui, self.gizmo.local_orientation, "Local").clicked() {
                            self.gizmo.local_orientation = true;
                        }
                        trough_divider(ui);
                        let snap = self.gizmo.snap;
                        let (r, resp) = ui.allocate_exact_size(egui::vec2(64.0, theme::H_BUTTON_SMALL), egui::Sense::click());
                        let fill = if snap { theme::BG_SEGMENT_ACTIVE } else if resp.hovered() { theme::BG_BUTTON } else { egui::Color32::TRANSPARENT };
                        ui.painter().rect_filled(r, theme::R_ROW, fill);
                        let col = if snap { theme::accent() } else { theme::TEXT_MUTED };
                        crate::icons::paint(
                            ui.painter(),
                            egui::Rect::from_center_size(egui::pos2(r.left() + 14.0, r.center().y), egui::vec2(13.0, 13.0)),
                            Icon::Snap,
                            col,
                        );
                        ui.painter().text(egui::pos2(r.left() + 26.0, r.center().y), egui::Align2::LEFT_CENTER, "Snap", egui::FontId::proportional(theme::SIZE_BODY), col);
                        if resp.on_hover_text("Snap to grid (or hold Ctrl)").clicked() {
                            self.gizmo.snap = !snap;
                        }
                        ui.style_mut().override_font_id = Some(theme::mono(theme::SIZE_META));
                        ui.add_sized(
                            [56.0, theme::H_BUTTON_SMALL],
                            egui::DragValue::new(&mut self.gizmo.grid_size).speed(0.05).range(0.01..=10.0).max_decimals(2).min_decimals(2).suffix(" m"),
                        )
                        .on_hover_text("Snap increment");
                    });
                    widgets::trough(ui, |ui| {
                        let on = self.gizmo.align_to_surface;
                        if widgets::segment(ui, on, "Align")
                            .on_hover_text("Placing a dragged-in model or prefab: tilt it so its up follows the surface (hold Alt to flip while placing)")
                            .clicked()
                        {
                            self.gizmo.align_to_surface = !on;
                        }
                    });
                    widgets::trough(ui, |ui| {
                        let ortho = zestengine_core::is_orthographic(&self.proj);
                        if widgets::segment(ui, !ortho, "3D").on_hover_text("Perspective Scene view").clicked() && ortho {
                            self.actions.push(EditorAction::SetOrtho(false));
                        }
                        if widgets::segment(ui, ortho, "2D")
                            .on_hover_text("Orthographic Scene view (no foreshortening). Scroll zooms.")
                            .clicked()
                            && !ortho
                        {
                            self.actions.push(EditorAction::SetOrtho(true));
                        }
                    });
                    if self.playing {
                        widgets::chip(ui, "Playing — edits revert when you stop");
                    }
                });
            });

        // Top-right: orientation widget; click an axis to look along it.
        egui::Area::new(ui.id().with("vp-axes"))
            .order(egui::Order::Middle)
            .pivot(egui::Align2::RIGHT_TOP)
            .fixed_pos(rect.right_top() + egui::vec2(-12.0, 12.0))
            .show(&ctx, |ui| {
                ui.vertical_centered(|ui| {
                    let (r, _) = ui.allocate_exact_size(egui::vec2(64.0, 64.0), egui::Sense::hover());
                    let p = ui.painter();
                    let c = r.center();
                    p.circle(c, 32.0, egui::Color32::from_rgba_unmultiplied(17, 18, 20, 191), egui::Stroke::new(1.0_f32, theme::LINE_FIELD));
                    let view = self.view;
                    let mut axes: Vec<(f32, egui::Pos2, usize, bool)> = Vec::new();
                    for (a, dir) in [glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z].into_iter().enumerate() {
                        for neg in [false, true] {
                            let d = if neg { -dir } else { dir };
                            let v = view.transform_vector3(d);
                            axes.push((v.z, c + egui::vec2(v.x, -v.y) * 22.0, a, neg));
                        }
                    }
                    // Far axes first so near ones draw on top.
                    axes.sort_by(|x, y| x.0.total_cmp(&y.0));
                    for (_, pos, a, neg) in axes {
                        let col = theme::axis(a);
                        if neg {
                            p.circle(pos, 5.0, egui::Color32::from_rgba_unmultiplied(17, 18, 20, 220), egui::Stroke::new(1.2_f32, col));
                        } else {
                            p.line_segment([c, pos], egui::Stroke::new(1.5_f32, col));
                            p.circle_filled(pos, 7.0, col);
                            p.text(pos, egui::Align2::CENTER_CENTER, ["X", "Y", "Z"][a], theme::mono_medium(theme::SIZE_TINY), theme::TEXT_ON_ACCENT);
                        }
                        let hit = egui::Rect::from_center_size(pos, egui::vec2(14.0, 14.0));
                        let resp = ui.interact(hit, ui.id().with(("vp-axis", a, neg)), egui::Sense::click());
                        let label = format!("Look along {}{}", if neg { "+" } else { "-" }, ["X", "Y", "Z"][a]);
                        if resp.on_hover_text(label).clicked() {
                            let dir = [glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z][a];
                            self.actions.push(EditorAction::SnapView(if neg { dir } else { -dir }));
                        }
                    }
                    ui.label(egui::RichText::new("Persp").size(theme::SIZE_META).color(theme::TEXT_MUTED));
                });
            });

        // Bottom-left stats pill.
        egui::Area::new(ui.id().with("vp-stats"))
            .order(egui::Order::Middle)
            .pivot(egui::Align2::LEFT_BOTTOM)
            .fixed_pos(rect.left_bottom() + egui::vec2(12.0, -12.0))
            .show(&ctx, |ui| {
                egui::Frame::new()
                    .fill(theme::OVERLAY)
                    .stroke(egui::Stroke::new(1.0_f32, theme::LINE_FIELD))
                    .corner_radius(egui::CornerRadius::same(theme::R_ROW))
                    .inner_margin(egui::Margin::symmetric(8, 4))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            let mono = theme::mono(theme::SIZE_META);
                            ui.label(egui::RichText::new(format!("{:.0}", self.stats.0)).font(mono.clone()).color(theme::accent()));
                            ui.label(egui::RichText::new("fps").font(mono.clone()).color(theme::TEXT_MUTED));
                            ui.label(egui::RichText::new(format!(" {:.2} ms", self.stats.1)).font(mono.clone()).color(theme::TEXT_DIM));
                            if *self.viewport_stats {
                                ui.label(egui::RichText::new(format!(" · {} draws", self.stats.2)).font(mono).color(theme::TEXT_DIM));
                            }
                        });
                    });
            });

        // Zestlet coverage view: a legend of the colours with each reason's triangle share. Only while that
        // mode is active (nothing is built otherwise).
        if *self.gi_debug == zestengine_render::VIEW_ZESTLET_COVERAGE {
            egui::Area::new(ui.id().with("vp-zestlet-legend"))
                .order(egui::Order::Middle)
                .pivot(egui::Align2::LEFT_BOTTOM)
                .fixed_pos(rect.left_bottom() + egui::vec2(12.0, -12.0))
                .interactable(false)
                .show(&ctx, |ui| {
                    egui::Frame::new()
                        .fill(theme::OVERLAY)
                        .stroke(egui::Stroke::new(1.0_f32, theme::LINE_FIELD))
                        .corner_radius(egui::CornerRadius::same(theme::R_ROW))
                        .inner_margin(egui::Margin::symmetric(8, 6))
                        .show(ui, |ui| {
                            let mono = theme::mono(theme::SIZE_META);
                            ui.label(egui::RichText::new("Zestlet coverage").font(mono.clone()).color(theme::TEXT_DIM));
                            for row in self.zestlet_coverage.legend() {
                                ui.horizontal(|ui| {
                                    ui.spacing_mut().item_spacing.x = 6.0;
                                    let (swatch, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                                    let c = row.colour.map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
                                    ui.painter().rect_filled(swatch, egui::CornerRadius::same(2), egui::Color32::from_rgb(c[0], c[1], c[2]));
                                    ui.label(egui::RichText::new(&row.label).font(mono.clone()).color(theme::TEXT_MUTED));
                                    if let Some(share) = row.share {
                                        ui.label(egui::RichText::new(format!("{:.0}%", share * 100.0)).font(mono.clone()).color(theme::TEXT_DIM));
                                    }
                                });
                            }
                        });
                });
        }

        // Play mode: 2 px inset accent outline.
        if self.playing {
            ui.painter().rect_stroke(rect.shrink(1.0), egui::CornerRadius::ZERO, egui::Stroke::new(2.0_f32, theme::accent()), egui::StrokeKind::Inside);
        }
    }

    fn inspector_ui(&mut self, ui: &mut egui::Ui) {
        // Keep the inspector laid out at a usable width so its rows don't
        // collapse if the dock is dragged narrow (egui_dock has no per-leaf
        // minimum, so we enforce it from the content side).
        ui.set_min_width(330.0);
        let shader_refs: Vec<&str> = self.shader_list.iter().map(String::as_str).collect();

        // Build & Reload, right where the user edits components (also in the
        // code editor header).
        if ui
            .button("Build & Reload Components")
            .on_hover_text("cargo-builds plugins/ in the background and hot-reloads them")
            .clicked()
        {
            self.actions.push(EditorAction::ReloadPlugins);
        }

        // When a code tab is focused, show its problems here (kept out of the
        // editor so the code area doesn't shift while typing).
        if let Some(path) = self.focused_code.clone()
            && let Some(editor) = self.open_editors.iter().find(|e| e.path == path)
        {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            ui.horizontal(|ui| {
                ui.heading("Problems");
                ui.label(egui::RichText::new(name).small().weak());
            });
            if editor.diagnostics.is_empty() {
                ui.label(egui::RichText::new(format!("{} No problems", ph::CHECK)).size(14.0).weak());
            } else {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for d in &editor.diagnostics {
                            let color = if d.level == "error" {
                                ui.visuals().error_fg_color
                            } else {
                                ui.visuals().warn_fg_color
                            };
                            ui.label(
                                egui::RichText::new(format!(
                                    "{} {} — {}",
                                    if d.level == "error" { ph::X_CIRCLE } else { ph::WARNING },
                                    d.line,
                                    d.message
                                ))
                                .size(14.0)
                                .color(color),
                            );
                        }
                    });
            }
            return;
        }

        // Effective selection (honours the lock).
        let effective = if self.inspector.locked {
            self.inspector_lock_target
                .clone()
                .unwrap_or_else(|| self.selection.clone())
        } else {
            self.selection.clone()
        };
        // Pinned header, stays put while the body scrolls. For an object it's the
        // single flex row [enabled][name (grows)][static][lock]; otherwise just
        // the lock toggle.
        if self.inspector.lock_bar(ui) {
            if self.inspector.locked {
                *self.inspector_lock_target = Some(self.selection.clone());
            } else {
                *self.inspector_lock_target = None;
                // Just unlocked: the file editor follows the live selection
                // again (it was pinned to the locked file while locked).
                if let Selection::File(p) = &self.selection
                    && self.file_material.as_ref().is_none_or(|fm| fm.path != *p)
                {
                    load_material_file_into(p, self.file_material);
                }
            }
        }
        match &effective {
            Selection::Object(i) if *i < self.scene.objects.len() => {
                let index = *i;
                let shown = self.scene.layers.shown_count();
                let layer_names: Vec<String> = (0..shown as u8).map(|l| self.scene.layers.layer_name(l)).collect();
                let model_root = model_root_flags(&self.scene.objects).get(index).copied().unwrap_or(false);
                let kind = if model_root { crate::theme::ObjectKind::Model } else { object_kind(&self.scene.objects[index]) };
                let type_line = match self.scene.objects[index].parent.and_then(|p| self.scene.objects.get(p)) {
                    Some(p) => format!("{} · child of {}", kind.label(), p.name),
                    None => kind.label().to_owned(),
                };
                let group: Vec<usize> = if self.multi_objects.len() > 1 && self.multi_objects.contains(&index) {
                    self.multi_objects.iter().copied().filter(|&j| j < self.scene.objects.len()).collect()
                } else {
                    vec![index]
                };
                let names_mixed = group.iter().any(|&j| self.scene.objects[j].name != self.scene.objects[index].name);
                let obj = &mut self.scene.objects[index];
                let mut name = if names_mixed { String::new() } else { obj.name.clone() };
                let mut enabled = obj.enabled;
                let mut static_geometry = obj.static_geometry;
                let mut layer = obj.layer;
                let mut net_role = obj.net_role;
                let before = (name.clone(), enabled, static_geometry, layer);
                let net_before = net_role;
                let net_warnings = zestengine_engine::net_roles::server_view_warnings(&self.scene.objects, self.registry, index);
                let hr = self.inspector.object_header(
                    ui,
                    kind,
                    &type_line,
                    &mut name,
                    &mut enabled,
                    &mut static_geometry,
                    &mut layer,
                    &layer_names,
                    &mut net_role,
                    &net_warnings,
                );
                if hr.object_changed {
                    // Every selected object takes what changed; a blank name over
                    // differing names leaves them as they are.
                    let mut enabled_edits = Vec::new();
                    for &j in &group {
                        let o = &mut self.scene.objects[j];
                        if name != before.0 && !name.is_empty() {
                            o.name = name.clone();
                        }
                        if enabled != before.1 {
                            if o.enabled != enabled {
                                enabled_edits.push((o.id, o.enabled, enabled));
                            }
                            o.enabled = enabled;
                        }
                        if static_geometry != before.2 {
                            o.static_geometry = static_geometry;
                        }
                        if layer != before.3 {
                            o.layer = layer;
                        }
                        if net_role != net_before {
                            o.net_role = net_role;
                        }
                    }
                    if !enabled_edits.is_empty() {
                        // The Inspector is the only place that enables / disables.
                        self.actions.push(EditorAction::RecordEnabled(enabled_edits));
                    }
                }
                // Lightmap UVs (model objects): auto-generate toggle (off by
                // default) + regenerate. Generated UVs persist as a `.lmuv` marker.
                let model_path = match &self.scene.objects[index].source {
                    zestengine_assets::ObjectSource::Model { path, .. } => Some(path.clone()),
                    _ => None,
                };
                if let Some(mpath) = model_path {
                    let has_marker = self.generated_uv_models.iter().any(|m| m == &mpath);
                    ui.horizontal(|ui| {
                        let mut on = has_marker;
                        if ui
                            .checkbox(&mut on, "Auto-gen lightmap UVs")
                            .on_hover_text(
                                "When this model has no usable lightmap UV (no 2nd UV set AND an \
                                 overlapping uv0), auto-unwrap a non-overlapping one for baking. \
                                 Off (default) = use the model's own UVs. Reloads the scene.",
                            )
                            .changed()
                        {
                            self.actions.push(EditorAction::SetLightmapUvGen(mpath.clone(), on));
                        }
                        if has_marker
                            && ui
                                .small_button("Regenerate")
                                .on_hover_text("Re-run the unwrapper (picks up improvements).")
                                .clicked()
                        {
                            self.actions.push(EditorAction::RegenerateLightmapUvs);
                        }
                    });
                }
            }
            _ => {}
        }
        ui.separator();
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                // Reserve a constant right gutter so the floating scrollbar
                // overlays empty padding instead of the widgets (slider pips on
                // the right edge). A fixed margin never reflows, so no bounce.
                egui::Frame::default()
                    .inner_margin(egui::Margin { right: 12, ..egui::Margin::ZERO })
                    .show(ui, |ui| match &effective {
                Selection::None => {
                    self.inspector.ui(ui, InspectorContent::Empty, &shader_refs);
                }
                Selection::Object(i) => {
                    let index = *i;
                    if index >= self.scene.objects.len() {
                        // Locked onto an object that was since deleted.
                        self.inspector.ui(ui, InspectorContent::Empty, &shader_refs);
                        return;
                    }
                    // Multi-select banner: the inspector shows the anchor object,
                    // but edits to shared transform/component values apply to ALL
                    // selected objects (see `propagate_multi_edit`). Components not
                    // present on every selected object are listed so it's clear
                    // which ones won't propagate.
                    let multi = self.multi_objects.len();
                    if multi > 1 && self.multi_objects.contains(&index) {
                        let sel: Vec<usize> = self
                            .multi_objects
                            .iter()
                            .copied()
                            .filter(|&j| j < self.scene.objects.len())
                            .collect();
                        // Component names shared by ALL selected objects.
                        let shared: std::collections::BTreeSet<String> = {
                            let mut sets = sel.iter().map(|&j| {
                                self.scene.objects[j]
                                    .save_components()
                                    .into_iter()
                                    .map(|c| c.kind)
                                    .collect::<std::collections::BTreeSet<String>>()
                            });
                            match sets.next() {
                                Some(first) => sets.fold(first, |a, b| {
                                    a.intersection(&b).cloned().collect()
                                }),
                                None => Default::default(),
                            }
                        };
                        egui::Frame::new()
                            .fill(egui::Color32::from_rgb(48, 40, 70))
                            .inner_margin(egui::Margin::symmetric(8, 5))
                            .show(ui, |ui| {
                                ui.label(
                                    egui::RichText::new(format!("{multi} objects selected"))
                                        .strong()
                                        .color(egui::Color32::from_rgb(200, 180, 255)),
                                );
                                ui.label(
                                    egui::RichText::new(
                                        "Editing applies to all selected. Showing the active object.",
                                    )
                                    .weak()
                                    .small(),
                                );
                                if shared.is_empty() {
                                    ui.label(
                                        egui::RichText::new("No components shared by all.")
                                            .small()
                                            .color(egui::Color32::from_rgb(220, 160, 120)),
                                    );
                                } else {
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "Shared: {}",
                                            shared.into_iter().collect::<Vec<_>>().join(", ")
                                        ))
                                        .small()
                                        .weak(),
                                    );
                                }
                            });
                        ui.separator();
                    }
                    let object = &self.scene.objects[index];
                    let render = object.render;
                    // Rotation is shown as the Euler triple the Inspector last wrote
                    // (stable while dragging one axis), else derived from the quaternion.
                    let euler_id = egui::Id::new(("zestengine-euler-cache", index));
                    let euler_cache = ui.data(|d| d.get_temp::<zestengine_engine::transform_edit::EulerCache>(euler_id));
                    let shown_euler = zestengine_engine::transform_edit::display_euler(euler_cache, object.rotation);
                    let group: Vec<usize> = if self.multi_objects.len() > 1 && self.multi_objects.contains(&index) {
                        self.multi_objects.iter().copied().filter(|&j| j < self.scene.objects.len()).collect()
                    } else {
                        Vec::new()
                    };
                    let mixed = {
                        let objs: Vec<&scene::SceneObject> = group.iter().map(|&j| &self.scene.objects[j]).collect();
                        let axis = |f: &dyn Fn(&scene::SceneObject) -> [f32; 3]| -> [bool; 3] {
                            let a = f(object);
                            std::array::from_fn(|k| objs.iter().any(|o| (f(o)[k] - a[k]).abs() > 1e-4))
                        };
                        let euler = |o: &scene::SceneObject| zestengine_engine::transform_edit::euler_deg(o.rotation);
                        crate::MixedInfo {
                            name: objs.iter().any(|o| o.name != object.name),
                            enabled: objs.iter().any(|o| o.enabled != object.enabled),
                            static_geometry: objs.iter().any(|o| o.static_geometry != object.static_geometry),
                            cast_shadows: objs.iter().any(|o| o.cast_shadows() != object.cast_shadows()),
                            translation: axis(&|o| o.translation.to_array()),
                            rotation: axis(&euler),
                            scale: axis(&|o| o.scale.to_array()),
                        }
                    };
                    // Per component, the fields that differ across the selection
                    // (shown as an em dash by the component inspectors).
                    let mixed_components: Vec<(String, Vec<String>)> = if group.is_empty() {
                        Vec::new()
                    } else {
                        object
                            .components
                            .iter()
                            .map(|c| {
                                let kind = c.type_name();
                                let rons: Vec<String> = group
                                    .iter()
                                    .filter_map(|&j| self.scene.objects[j].components.iter().find(|x| x.type_name() == kind).map(|x| x.save()))
                                    .collect();
                                let fields = crate::ron_fields::mixed(&rons).iter().map(|f| crate::widgets::mixed_key(f)).collect();
                                (kind.to_owned(), fields)
                            })
                            .collect()
                    };
                    ui.data_mut(|d| d.insert_temp(egui::Id::new(crate::MIXED_COMPONENTS_KEY), mixed_components));
                    let role_views = zestengine_engine::net_roles::component_role_views(&self.scene.objects[index], self.registry);
                    ui.data_mut(|d| d.insert_temp(egui::Id::new(crate::COMPONENT_ROLES_KEY), role_views));
                    // A slot was double-clicked last frame: show that file.
                    if let Some(abs) = ui.data_mut(|d| d.remove_temp::<String>(egui::Id::new(crate::OPEN_FILE_KEY))) {
                        self.actions.push(EditorAction::SelectFile(std::path::PathBuf::from(abs)));
                    }
                    let root = self.project_root.to_string_lossy().into_owned();
                    ui.data_mut(|d| {
                        d.insert_temp(egui::Id::new(crate::PROJECT_ROOT_KEY), root);
                        d.insert_temp(egui::Id::new(crate::PLAY_MODE_KEY), self.playing);
                        let authored = zestengine_engine::authored_collision::authored_pieces(self.scene, index)
                            .map(|p| (p.len(), zestengine_engine::authored_collision::pieces_summary(&p)));
                        d.insert_temp(egui::Id::new(crate::AUTHORED_COLLISION_KEY), authored);
                        d.insert_temp(egui::Id::new(crate::ASSET_TEMPLATES_KEY), asset_templates().to_vec());
                    });
                    let key = object.id.to_string();
                    let slot_materials: Vec<usize> = object.render_slots().map(|r| r.material).collect();
                    let defaults = self.scene.slot_default_materials(&object.source);
                    let name_of = |m: usize| self.scene.materials.get(m).map(|e| e.model.name.clone()).unwrap_or_default();
                    let skinned = self.scene.is_skinned_object(index);
                    let has_renderer = object.renderer_any().is_some();
                    let model_root = {
                        let flags = model_root_flags(&self.scene.objects);
                        flags.get(index).copied().unwrap_or(false).then(|| {
                            let children: Vec<usize> = (0..self.scene.objects.len()).filter(|&c| self.scene.objects[c].parent == Some(index)).collect();
                            let path = children
                                .iter()
                                .find_map(|&c| match &self.scene.objects[c].source {
                                    zestengine_assets::ObjectSource::Model { path, .. } => Some(path.clone()),
                                    _ => None,
                                })
                                .unwrap_or_default();
                            let (clips, _skeletons, humanoid, materials) = self.scene.model_summary(&path).unwrap_or_default();
                            let skinned = children.iter().any(|&c| self.scene.is_skinned_object(c));
                            crate::ModelRootInfo {
                                meshes: children.iter().filter(|&&c| self.scene.objects[c].render.is_some()).count(),
                                materials,
                                clips,
                                rig: if humanoid { "Humanoid" } else if skinned { "Generic" } else { "None" }.to_owned(),
                                children: children
                                    .iter()
                                    .map(|&c| crate::ModelChild {
                                        object: c,
                                        name: self.scene.objects[c].name.clone(),
                                        materials: self.scene.objects[c].render_slots().map(|r| name_of(r.material)).collect(),
                                    })
                                    .collect(),
                                path,
                            }
                        })
                    };
                    let mut info = ObjectInfoModel {
                        mixed,
                        model_root,
                        name: object.name.clone(),
                        enabled: object.enabled,
                        static_geometry: object.static_geometry,
                        key: key.clone(),
                        kind: object.kind_label(),
                        transform: TransformModel {
                            translation: object.translation.to_array(),
                            rotation_deg: shown_euler,
                            scale: object.scale.to_array(),
                            parent: object.parent.and_then(|p| self.scene.objects.get(p)).map(|p| p.name.clone()),
                            world: zestengine_engine::transform_edit::decompose_trs(self.scene.world_transform(index)).map(|(s, r, t)| {
                                (t.to_array(), zestengine_engine::transform_edit::euler_deg(r), s.to_array())
                            }),
                        },
                        mesh: render.map(|r| {
                            let mi = self.scene.mesh_info(r.mesh);
                            (mi.vertices, mi.triangles)
                        }),
                        filter: object.has_mesh_filter().then(|| {
                            let (vertices, triangles) = render.map(|r| self.scene.mesh_info(r.mesh)).map_or((0, 0), |m| (m.vertices, m.triangles));
                            let (label, model_path) = match &object.source {
                                zestengine_assets::ObjectSource::Primitive { shape } => (shape.label().to_owned(), None),
                                zestengine_assets::ObjectSource::Builtin { mesh } => (format!("Builtin mesh {mesh}"), None),
                                zestengine_assets::ObjectSource::Model { path, mesh, .. } => (format!("Mesh {mesh}"), Some(path.clone())),
                                _ => (String::new(), None),
                            };
                            crate::MeshFilterInfo { label, model_path, vertices, triangles }
                        }),
                        renderer: has_renderer.then(|| crate::RendererInfo {
                            slots: slot_materials
                                .iter()
                                .enumerate()
                                .map(|(k, &m)| crate::SlotInfo {
                                    submesh: format!("Submesh {k}"),
                                    model_material: defaults.get(k).map(|&d| name_of(d)).unwrap_or_default(),
                                    material: name_of(m),
                                })
                                .collect(),
                            skinned,
                            joints: if skinned { self.scene.skinned_joint_names(index) } else { Vec::new() },
                        }),
                        blendshapes: self.scene.morph_names_shared(index),
                    };
                    // One material section per distinct material (slots sharing one share the
                    // section). Only open sections lend their model, so a closed one costs nothing.
                    let groups = crate::group_slots(&slot_materials);
                    let user_counts = {
                        let id = egui::Id::new(("zestengine-material-users", index));
                        let stamp = (self.scene.objects.len(), false);
                        type CachedView = (Instant, (usize, bool), std::collections::HashMap<usize, u32>);
            let cached: Option<CachedView> = ui.data(|d| d.get_temp(id));
                        match cached {
                            Some((at, st, map)) if st == stamp && at.elapsed().as_secs_f32() < 1.0 => map,
                            _ => {
                                let map = self.scene.material_user_counts();
                                ui.data_mut(|d| d.insert_temp(id, (Instant::now(), stamp, map.clone())));
                                map
                            }
                        }
                    };
                    struct SectionPlan {
                        slots: Vec<usize>,
                        material: usize,
                        open: bool,
                        warnings: Vec<crate::SlotWarning>,
                        title: String,
                        shader: String,
                        shared_by: u32,
                        source: crate::MaterialSource,
                        embedded: bool,
                    }
                    let mut plans: Vec<SectionPlan> = Vec::new();
                    if has_renderer {
                        for (m, slots) in &groups {
                            let open = crate::section_open(ui.ctx(), &key, slots[0]);
                            let embedded = self.scene.materials[*m].file.is_none();
                            let warnings = if open && !embedded {
                                let entry = &self.scene.materials[*m];
                                let owner = entry.file.as_ref().map(|f| relative_to(f, self.project_root)).unwrap_or_else(|| entry.model.name.clone());
                                let textures = entry.model.textures.clone();
                                let mats = &self.scene.materials;
                                let root = self.project_root;
                                let file = entry.file.as_deref();
                                self.slot_validator.warnings(root, &textures, &owner, file, || scene_material_textures(mats, root))
                            } else {
                                Vec::new()
                            };
                            let entry = &self.scene.materials[*m];
                            let model_name = match &object.source {
                                zestengine_assets::ObjectSource::Model { path, .. } => path.rsplit('/').next().unwrap_or(path).to_owned(),
                                _ => "default".to_owned(),
                            };
                            let source = match (&entry.file, defaults.contains(m)) {
                                (Some(f), _) => crate::MaterialSource::File(relative_to(f, self.project_root)),
                                (None, true) => crate::MaterialSource::Model { model: model_name },
                                (None, false) => crate::MaterialSource::Edited,
                            };
                            plans.push(SectionPlan {
                                slots: slots.clone(),
                                material: *m,
                                open,
                                warnings,
                                title: crate::section_title(&entry.model.name, slots, slot_materials.len()),
                                shader: if entry.model.shader == "standard" { "Standard".to_owned() } else { entry.model.shader.clone() },
                                shared_by: user_counts.get(m).copied().unwrap_or(1),
                                source,
                                embedded,
                            });
                        }
                    }
                    if self.slot_validator.scanning() {
                        ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
                    }
                    // Object list (id + name) for ObjectRef pickers, built as an
                    // owned Vec before the mutable component borrow so it doesn't
                    // alias scene.objects.
                    let objects: Vec<(ObjectId, String)> = self
                        .scene
                        .objects
                        .iter()
                        .map(|o| (o.id, o.name.clone()))
                        .collect();
                    // What every object's components expose to event bindings
                    // (the action list editor's dropdowns).
                    crate::actions_ui::publish_catalog(
                        ui,
                        crate::actions_ui::ActionCatalog::build(
                            self.scene.objects[index].id,
                            self.scene.objects.iter().map(|o| (o.id, o.components.as_slice())),
                        )
                        .with_playing(self.playing),
                    );
                    // User content (an object from a loaded package, or anything
                    // while the Content Builder is open) may only bind ugc_safe actions.
                    crate::actions_ui::set_ugc_only(
                        ui,
                        self.ugc_context || self.scene.object_origin.contains_key(&self.scene.objects[index].id),
                    );
                    // Layer names for the camera culling-mask picker (cloned before
                    // the mutable scene borrow below).
                    let layer_names: Vec<String> = self.scene.layers.names.clone();
                    // The sprite inspectors' sorting-layer popups read this copy.
                    crate::publish_sorting_layers(ui.ctx(), &self.scene.layers.sorting);
                    // Split borrows: components live on the object, the
                    // material model in the scene's material list.
                    // Assets assigned to component slots, edited in place.
                    let root = self.project_root.to_path_buf();
                    let postfx_edited = std::cell::Cell::new(false);
                    let asset_editor = |ui: &mut egui::Ui, rel: &std::path::Path| -> Option<bool> {
                        let path = root.join(rel);
                        match path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).as_deref() {
                            Some("waterwaves") => crate::water_ui::wave_asset_ui(ui, &path),
                            Some("audiofx" | "audiosettings") => crate::audio_ui::audio_asset_ui(ui, &path),
                            Some(zestengine_core::PHYSMAT_EXTENSION) => {
                                let mut m = zestengine_core::load_physmat(&path).ok()?;
                                let changed = crate::physmat_editor_ui(ui, &mut m);
                                if changed && let Err(e) = zestengine_core::save_physmat(&path, &m) {
                                    tracing::error!("saving physics material: {e}");
                                }
                                Some(changed)
                            }
                            Some(zestengine_core::SPRITE_SHAPE_EXTENSION) => {
                                let mut p = zestengine_core::load_spriteshape(&path).ok()?;
                                let changed = crate::sprite_shape_profile_ui(ui, &mut p);
                                if changed && let Err(e) = zestengine_core::save_spriteshape(&path, &p) {
                                    tracing::error!("saving sprite shape profile: {e}");
                                }
                                Some(changed)
                            }
                            Some("postfx") => {
                                let mut p = zestengine_assets::load_postfx(&path).ok()?;
                                let changed = postfx_editor_ui(ui, &mut p, true);
                                if changed {
                                    if let Err(e) = zestengine_assets::save_postfx(&path, &p) {
                                        tracing::error!("saving postfx: {e:#}");
                                    }
                                    postfx_edited.set(true);
                                }
                                Some(changed)
                            }
                            _ => None,
                        }
                    };
                    // An Avatar Descriptor shows its model's rig status.
                    let rig_key = egui::Id::new(crate::avatar_rig::AVATAR_RIG_KEY);
                    let avatar_info = self.scene.objects[index]
                        .components
                        .iter()
                        .any(|c| c.as_any().is::<zestengine_core::AvatarDescriptor>())
                        .then(|| crate::rig_editor::avatar_rig_info(self.scene, index))
                        .flatten();
                    // Menu Toggle: the preview request from last frame, then the
                    // synced-bit totals for the avatar around the selection.
                    let preview_key = egui::Id::new(crate::avatar_ui::AVATAR_PREVIEW_KEY);
                    if let Some(want) = ui.data_mut(|d| d.remove_temp::<bool>(preview_key)) {
                        if want {
                            self.avatar_preview.start(self.scene, index);
                        } else {
                            self.avatar_preview.restore(self.scene);
                        }
                    }
                    let info_key = egui::Id::new(crate::avatar_ui::AVATAR_INFO_KEY);
                    let wants_info = self.scene.objects[index].components.iter().any(|c| c.as_any().is::<zestengine_core::MenuToggle>() || c.as_any().is::<zestengine_core::AvatarParameters>());
                    let totals = if wants_info { crate::avatar_edit::bit_totals(self.scene, index) } else { None };
                    let picks = if wants_info || self.scene.objects[index].components.iter().any(|c| c.as_any().is::<zestengine_core::ObjectToggle>()) {
                        crate::avatar_ui::pick_list(self.scene, index)
                    } else {
                        Vec::new()
                    };
                    // Avatar Descriptor: "Build preview" in the Generated output foldout.
                    let build_key = egui::Id::new(crate::avatar_ui::AVATAR_BUILD_REQUEST_KEY);
                    let gen_key = egui::Id::new(crate::avatar_ui::AVATAR_GENERATED_KEY);
                    if ui.data_mut(|d| d.remove_temp::<bool>(build_key)).unwrap_or(false)
                        && let Some(root) = crate::avatar_edit::avatar_root_of(self.scene, index)
                    {
                        let out = crate::avatar_ui::generated_output(self.scene, root, self.project_root, &zestengine_engine::shaders::ShaderLibrary::default(), zestengine_assets::package::ContentLimits::for_kind(zestengine_core::ContentKind::Avatar).max_param_bits);
                        ui.data_mut(|d| d.insert_temp(gen_key, out));
                    }
                    let preview_on = self.avatar_preview.is_on(self.scene.objects[index].id);
                    if !preview_on && self.avatar_preview.active() {
                        // A different object is selected now: put the scene back.
                        self.avatar_preview.restore(self.scene);
                    }
                    ui.data_mut(|d| match totals {
                        Some((used, max, toggles)) => d.insert_temp(info_key, crate::avatar_ui::AvatarInfo { used_bits: used, max_bits: max, toggles, preview_active: preview_on, picks }),
                        None => {
                            d.remove_temp::<crate::avatar_ui::AvatarInfo>(info_key);
                        }
                    });
                    ui.data_mut(|d| match avatar_info {
                        Some(i) => d.insert_temp(rig_key, i),
                        None => {
                            d.remove_temp::<crate::avatar_rig::AvatarRigInfo>(rig_key);
                        }
                    });
                    let scene = &mut *self.scene;
                    let components = &mut scene.objects[index].components;
                    let open_models: Vec<usize> = plans.iter().filter(|p| p.open && !p.embedded).map(|p| p.material).collect();
                    let mut lent: std::collections::HashMap<usize, &mut MaterialModel> =
                        scene.materials.iter_mut().enumerate().filter(|(i, _)| open_models.contains(i)).map(|(i, e)| (i, &mut e.model)).collect();
                    let sections: Vec<crate::MaterialSectionModel> = plans
                        .into_iter()
                        .map(|p| crate::MaterialSectionModel {
                            slots: p.slots,
                            material_index: p.material,
                            title: p.title,
                            shader: p.shader,
                            shared_by: p.shared_by,
                            source: p.source,
                            embedded: p.embedded,
                            model: lent.remove(&p.material),
                            shader_info: self.section_shader_infos.get(&p.material),
                            slot_warnings: p.warnings,
                        })
                        .collect();
                    // Prefab instance: the Prefab row (Overrides dropdown) and the
                    // override marks on the rows drawn below.
                    if let Some(panel) = self.prefab.panel.clone() {
                        crate::prefab_ui::prefab_header(ui, &panel, &mut self.prefab.overrides);
                    }
                    if let Some(p) = self.prefab.inspect.clone() {
                        crate::prefab_ui::publish_inspect(ui, p);
                    }
                    let response = self.inspector.ui(
                        ui,
                        InspectorContent::Object {
                            info: &mut info,
                            sections,
                            components,
                            registry: self.registry,
                            editor_components: self.editor_components,
                            objects: &objects,
                            layer_names: &layer_names,
                            asset_editor: Some(&asset_editor),
                        },
                        &shader_refs,
                    );
                    crate::prefab_ui::removed_components_ui(ui);
                    crate::prefab_ui::clear_inspect(ui);
                    if let Some(rel) = ui.data_mut(|d| d.remove_temp::<String>(egui::Id::new(crate::avatar_rig::OPEN_RIG_KEY))) {
                        self.actions.push(EditorAction::SelectFile(self.project_root.join(rel)));
                        self.rig_ui.configure = true;
                    }
                    if let Some(table) = ui.data_mut(|d| d.remove_temp::<Vec<(String, String)>>(egui::Id::new(crate::avatar_rig::MOVE_TO_MODEL_KEY))) {
                        self.move_avatar_overrides(index, table);
                    }
                    if postfx_edited.get() {
                        // Volumes pick up the edit next frame.
                        self.scene.invalidate_postfx_cache();
                    }
                    if response.object_changed && self.multi_objects.len() > 1 && self.multi_objects.contains(&index) {
                        // Typed transform values apply to the whole selection as
                        // absolute values (propagate_multi_edit sets them).
                        let o = &self.scene.objects[index];
                        let old = (
                            o.translation.to_array(),
                            zestengine_engine::transform_edit::display_euler(euler_cache, o.rotation),
                            o.scale.to_array(),
                        );
                        let t = &info.transform;
                        let pick = |new: [f32; 3], was: [f32; 3]| -> [Option<f32>; 3] {
                            std::array::from_fn(|k| ((new[k] - was[k]).abs() > 1e-6).then_some(new[k]))
                        };
                        self.multi_edit.translation = pick(t.translation, old.0);
                        self.multi_edit.rotation_deg = pick(t.rotation_deg, old.1);
                        self.multi_edit.scale = pick(t.scale, old.2);
                    }
                    if response.object_changed {
                        let object = &mut self.scene.objects[index];
                        object.name = info.name;
                        object.enabled = info.enabled;
                        object.static_geometry = info.static_geometry;
                        object.translation = Vec3::from(info.transform.translation);
                        // Local rotation, rewritten only when the Inspector actually
                        // changed it (a rename must not round-trip it through Euler).
                        if info.transform.rotation_deg != shown_euler {
                            let cache = zestengine_engine::transform_edit::EulerCache::new(info.transform.rotation_deg);
                            object.rotation = cache.quat;
                            ui.data_mut(|d| d.insert_temp(euler_id, cache));
                        }
                        object.scale = Vec3::from(info.transform.scale);
                    }
                    for event in response.renderer_events {
                        use crate::RendererEvent as E;
                        match event {
                            E::Assign { slot, value } => self.actions.push(EditorAction::AssignSlot(index, slot, value)),
                            E::MakeUnique { slot } => self.actions.push(EditorAction::MakeUniqueMaterial(index, slot)),
                            E::Edited(m) => self.actions.push(EditorAction::MaterialEdited(m)),
                            E::Reset(m) => self.actions.push(EditorAction::ResetMaterial(m)),
                            E::Extract(m) => self.actions.push(EditorAction::ExtractMaterial(m)),
                            E::Texture { material, slot, path } => self.actions.push(EditorAction::AssignTexture { material, slot, path }),
                            E::Fix { material, slot, fix } => self.actions.push(EditorAction::FixTextureSlot { material: Some(material), slot, fix }),
                            E::SelectObject(i) => self.actions.push(EditorAction::SelectObject(i)),
                            E::OpenImportSettings(rel) => self.actions.push(EditorAction::SelectFile(self.project_root.join(rel))),
                            E::Reveal(rel) => {
                                let abs = self.project_root.join(rel).to_string_lossy().into_owned();
                                ui.data_mut(|d| d.insert_temp(egui::Id::new(crate::PING_FILE_KEY), abs));
                            }
                        }
                    }
                    // Component add/remove takes effect immediately so the
                    // frame's undo diff captures it as one Object entry.
                    if let Some(name) = response.add_component
                        && let Some(component) = self.registry.create(name)
                    {
                        let rigid_body = component.as_any().is::<zestengine_core::RigidBody>();
                        let mut component = component;
                        if let Some((lo, hi)) = object_bounds(self.scene, index) {
                            fit_collider(component.as_any_mut(), lo, hi);
                        }
                        self.scene.objects[index].components.push(component);
                        if rigid_body {
                            let bounds = object_bounds(self.scene, index);
                            let ball = zestengine_engine::physics::mesh_is_ball(self.scene, index);
                            let authored = zestengine_engine::authored_collision::authored_set(self.scene, index).is_some();
                            ensure_collider(&mut self.scene.objects[index], ball, authored);
                            if let (Some((lo, hi)), Some(last)) = (bounds, self.scene.objects[index].components.last_mut()) {
                                fit_collider(last.as_any_mut(), lo, hi);
                            }
                        }
                    }
                    if let Some(remove) = response.remove_component
                        && remove < self.scene.objects[index].components.len()
                    {
                        zestengine_engine::net_roles::remove_component(&mut self.scene.objects[index], remove);
                    }
                    if let Some((component, role)) = response.set_component_role {
                        zestengine_engine::net_roles::set_component_role(&mut self.scene.objects[index], component, role);
                    }
                }
                Selection::File(path) => {
                    let path_display = path.display().to_string();
                    let ext = path
                        .extension()
                        .and_then(|e| e.to_str())
                        .map(str::to_lowercase);
                    match ext.as_deref() {
                        Some("material") => {
                            if let Some(fm) = self.file_material.as_mut() {
                                let owner = relative_to(&fm.path, self.project_root);
                                let textures = fm.model.textures.clone();
                                let mats = &self.scene.materials;
                                let root = self.project_root;
                                let slot_warnings = self.slot_validator.warnings(root, &textures, &owner, Some(&fm.path), || scene_material_textures(mats, root));
                                if self.slot_validator.scanning() {
                                    ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
                                }
                                ui.data_mut(|d| d.insert_temp(egui::Id::new(crate::SLOT_WARNINGS_KEY), slot_warnings));
                                let response = self.inspector.ui(
                                    ui,
                                    InspectorContent::MaterialFile {
                                        path: path_display,
                                        material: &mut fm.model,
                                        shader_info: self.shader_info,
                                        dirty: fm.dirty,
                                    },
                                    &shader_refs,
                                );
                                if response.material_changed {
                                    fm.dirty = true;
                                    // Objects using this .material update
                                    // live, not only on save.
                                    self.actions
                                        .push(EditorAction::FileMaterialEdited(fm.path.clone()));
                                }
                                if response.save_material {
                                    self.actions.push(EditorAction::SaveFileMaterial);
                                }
                                if let Some((slot, path)) = response.texture_dropped {
                                    self.actions
                                        .push(EditorAction::AssignFileTexture { slot, path });
                                }
                                if let Some((slot, fix)) = ui.data_mut(|d| d.remove_temp::<(usize, crate::SlotFix)>(egui::Id::new(crate::SLOT_FIX_KEY))) {
                                    self.actions.push(EditorAction::FixTextureSlot { material: None, slot, fix });
                                }
                            } else {
                                ui.label("Failed to open material (see log).");
                            }
                        }
                        Some("scene") => {
                            let response = self.inspector.ui(
                                ui,
                                InspectorContent::SceneFile { path: path_display },
                                &shader_refs,
                            );
                            if response.load_scene {
                                self.actions.push(EditorAction::LoadScene(path.clone()));
                            }
                        }
                        Some("animctrl" | "anim") => {
                            crate::anim_controller_editor::file_inspector(ui, path, self.project_root, self.anim_ctrl_tab, self.anim_clip_tab);
                        }
                        Some("behaviour") => {
                            behaviour_tree_editor_ui(ui, path);
                        }
                        Some("shadergraph") => {
                            crate::shader_graph_editor::file_inspector(ui, path, self.project_root, self.shader_graph_tab);
                        }
                        Some("seq") => {
                            crate::sequencer_editor::file_inspector(ui, path, self.project_root, self.sequencer_tab);
                        }
                        Some("postfx") => {
                            ui.heading("Post FX Profile");
                            ui.label(egui::RichText::new(&path_display).small().weak());
                            ui.separator();
                            let mut profile = zestengine_assets::load_postfx(path).unwrap_or_default();
                            if postfx_editor_ui(ui, &mut profile, true) {
                                if let Err(e) = zestengine_assets::save_postfx(path, &profile) {
                                    tracing::error!("saving postfx: {e:#}");
                                }
                                // Volumes pick up the edit live next frame.
                                self.scene.invalidate_postfx_cache();
                            }
                        }
                        Some("tile") | Some("ruletile") => {
                            ui.label(egui::RichText::new(&path_display).small().weak());
                            ui.separator();
                            crate::tile_file_ui(ui, path);
                        }
                        Some(zestengine_core::SPRITE_SHAPE_EXTENSION) => {
                            ui.heading("Sprite Shape Profile");
                            ui.label(egui::RichText::new(&path_display).small().weak());
                            ui.separator();
                            match zestengine_core::load_spriteshape(path) {
                                Ok(mut profile) => {
                                    if crate::sprite_shape_profile_ui(ui, &mut profile)
                                        && let Err(e) = zestengine_core::save_spriteshape(path, &profile)
                                    {
                                        tracing::error!("saving sprite shape profile: {e}");
                                    }
                                }
                                Err(e) => {
                                    ui.label(format!("Failed to open the profile: {e}"));
                                }
                            }
                        }
                        Some(zestengine_assets::sprite_atlas::ATLAS_EXTENSION) => {
                            let root = self.project_root.to_path_buf();
                            self.sprite_atlas_ui.ui(ui, &root, path);
                        }
                        Some("waterwaves") => { crate::water_ui::wave_asset_ui(ui, path); }
                        Some("audiofx" | "audiosettings") => {
                            crate::audio_ui::audio_asset_ui(ui, path);
                        }
                        Some("physmat") => {
                            ui.heading("Physics Material");
                            ui.label(egui::RichText::new(&path_display).small().weak());
                            ui.separator();
                            let mut mat = zestengine_core::load_physmat(path).unwrap_or_default();
                            if crate::physmat_editor_ui(ui, &mut mat)
                                && let Err(e) = zestengine_core::save_physmat(path, &mat)
                            {
                                tracing::error!("saving physics material: {e}");
                            }
                        }
                        Some("prefab") => {
                            ui.heading(
                                std::path::Path::new(&path_display)
                                    .file_name()
                                    .map(|n| n.to_string_lossy().into_owned())
                                    .unwrap_or_else(|| path_display.clone()),
                            );
                            ui.label(egui::RichText::new(&path_display).small().weak());
                            self.asset_preview.ui(
                                ui,
                                &crate::asset_preview::PreviewSource::Prefab { path: path.clone(), root: self.project_root.to_path_buf() },
                            );
                            ui.add_space(6.0);
                            ui.label("Drag onto the viewport to place it, or:");
                            if ui.button("Place in scene").clicked() {
                                self.actions
                                    .push(EditorAction::InstantiatePrefab(path.clone(), None));
                            }
                            if ui.button("Edit Prefab").on_hover_text("Open the prefab in Prefab Mode").clicked() {
                                self.actions.push(EditorAction::Prefab(prefab_editor::PrefabAction::Stage(prefab_editor::StageGo::Open(
                                    zestengine_engine::prefab::link_path(self.project_root, path),
                                    zestengine_engine::scene::StageMode::Isolated,
                                    None,
                                ))));
                            }
                            ui.label(
                                egui::RichText::new(
                                    "Gameplay: ctx.spawn_prefab(\"<project-relative path>\", pos, rot)",
                                )
                                .small()
                                .weak(),
                            );
                        }
                        Some(ext) if CODE_EXTENSIONS.contains(&ext) => {
                            // Code/text files open in their own dockable editor
                            // tab; the inspector just offers a button to open it.
                            ui.heading(
                                std::path::Path::new(&path_display)
                                    .file_name()
                                    .map(|n| n.to_string_lossy().into_owned())
                                    .unwrap_or_else(|| path_display.clone()),
                            );
                            ui.label(egui::RichText::new(&path_display).small().weak());
                            if ui.button(format!("{} Open in editor", ph::PENCIL_SIMPLE)).clicked() {
                                self.actions.push(EditorAction::OpenCodeFile(path.clone()));
                            }
                        }
                        Some("fbx" | "gltf" | "glb" | "obj") => {
                            let unapplied = self.file_meta.as_ref().is_some_and(|f| f.path == *path && f.dirty);
                            ui.heading(if unapplied { "Model Import *" } else { "Model Import" });
                            ui.label(egui::RichText::new(&path_display).small().weak());
                            ui.separator();
                            self.asset_preview.ui(ui, &crate::asset_preview::PreviewSource::Model(path.clone()));
                            ui.add_space(6.0);
                            let mut do_revert = false;
                            let mut do_reimport = false;
                            let mut remap_job: Option<crate::model_import_editor::RemapJob> = None;
                            let mut rig_apply = false;
                            let mut rig_revert = false;
                            if let Some(fm) = self.file_meta.as_mut().filter(|f| f.path == *path) {
                                let dirty = fm.dirty;
                                match fm.meta.importer.as_model_mut() {
                                    Some(model) => {
                                        let model_out = self.model_ui.show(ui, path, model);
                                        if model_out.reimport {
                                            self.actions.push(EditorAction::ReimportModel(path.clone()));
                                        }
                                        if let Some(job) = model_out.geo_rebuild.clone() {
                                            self.actions.push(EditorAction::GeoRebuild(path.clone(), job));
                                        }
                                        let mut edited = model_out.changed;
                                        remap_job = if model_out.extract_textures {
                                            Some(crate::model_import_editor::RemapJob::ExtractTextures)
                                        } else if model_out.extract_materials {
                                            Some(crate::model_import_editor::RemapJob::ExtractMaterials)
                                        } else if model_out.search_remap {
                                            Some(crate::model_import_editor::RemapJob::Search)
                                        } else {
                                            None
                                        };
                                        // Adopt the material list a parse found without making the session dirty.
                                        let adopted = model_out.fill_sources.is_some();
                                        if let Some(names) = model_out.fill_sources {
                                            model.source_materials = names;
                                        }
                                        let rig_out = self.rig_ui.section(ui, path, &mut model.rig, dirty);
                                        edited |= rig_out.changed;
                                        rig_apply |= rig_out.apply;
                                        rig_revert |= rig_out.revert;
                                        if adopted {
                                            fm.saved = FileMeta::serialise(&fm.meta);
                                        }
                                        if edited {
                                            fm.refresh_dirty();
                                        }
                                    }
                                    None => {
                                        ui.label("No import settings for this asset type.");
                                    }
                                }
                                ui.separator();
                                ui.horizontal(|ui| {
                                    if ui
                                        .add_enabled(fm.dirty, egui::Button::new(format!("{} Revert", ph::ARROW_COUNTER_CLOCKWISE)))
                                        .on_hover_text("Discard the edits and go back to the saved settings")
                                        .clicked()
                                    {
                                        do_revert = true;
                                    }
                                    if ui
                                        .add_enabled(fm.dirty, egui::Button::new(format!("{} Apply", ph::CHECK)))
                                        .on_hover_text("Save the settings and reimport this model in place (the scene is not reloaded)")
                                        .clicked()
                                    {
                                        do_reimport = true;
                                    }
                                    if !fm.dirty
                                        && ui
                                            .button(format!("{} Reimport", ph::ARROWS_CLOCKWISE))
                                            .on_hover_text("Rebuild this model from the file, in place")
                                            .clicked()
                                    {
                                        do_reimport = true;
                                    }
                                    if fm.dirty {
                                        ui.label(egui::RichText::new("unapplied").small().weak());
                                    }
                                });
                            } else {
                                ui.label("Loading import settings…");
                            }
                            do_revert |= rig_revert;
                            do_reimport |= rig_apply;
                            if rig_apply {
                                // The reimported model carries the new rig; read it again.
                                self.rig_ui.forget();
                            }
                            if do_reimport {
                                self.model_ui.forget();
                            }
                            if do_revert {
                                self.actions.push(EditorAction::RevertFileMeta);
                            }
                            if do_reimport {
                                self.actions.push(EditorAction::ReimportModel(path.clone()));
                            }
                            if let Some(job) = remap_job {
                                self.actions.push(EditorAction::ModelRemap(path.clone(), job));
                            }
                        }
                        Some(e) if is_texture_ext(e) => {
                            if let Some(fm) = self.file_meta.as_mut().filter(|f| f.path == *path) {
                                ensure_texture_importer(fm);
                                let dirty = fm.dirty;
                                let mut sampler_edit = None;
                                let out = match fm.meta.importer.as_texture_mut() {
                                    Some(t) => {
                                        let before = t.sampler;
                                        let out = self.texture_ui.show(ui, path, t, dirty);
                                        if t.sampler != before {
                                            sampler_edit = Some(t.sampler);
                                        }
                                        out
                                    }
                                    None => Default::default(),
                                };
                                if out.changed {
                                    fm.refresh_dirty();
                                }
                                if let Some(sampler) = sampler_edit {
                                    self.actions.push(EditorAction::TextureSampler(path.clone(), sampler));
                                }
                                if out.apply {
                                    self.actions.push(EditorAction::ReimportTexture(path.clone()));
                                }
                                if out.revert {
                                    self.actions.push(EditorAction::RevertFileMeta);
                                }
                            } else {
                                ui.label("Loading import settings…");
                            }
                        }
                        _ => {
                            let size = std::fs::metadata(path).map(|m| m.len()).ok();
                            self.inspector.ui(
                                ui,
                                InspectorContent::File {
                                    path: path_display,
                                    size,
                                },
                                &shader_refs,
                            );
                        }
                    }
                }
                    });
            });
    }
}

/// Image file extensions the texture importer inspector handles.
fn is_texture_ext(ext: &str) -> bool {
    matches!(ext.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg" | "tga" | "bmp" | "exr" | "hdr" | "webp" | "gif")
}

fn is_texture_path(path: &std::path::Path) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(is_texture_ext)
}

/// Make sure the open image's `.meta` holds a texture importer (older metas say
/// `Generic`), re-baselining the saved snapshot so the conversion isn't an edit.
fn ensure_texture_importer(fm: &mut FileMeta) {
    if fm.meta.importer.as_texture().is_none() {
        fm.meta.importer = zestengine_assets::ImporterSettings::Texture(zestengine_assets::TextureImport::default());
        fm.saved = FileMeta::serialise(&fm.meta);
        fm.dirty = false;
    }
}

/// (owner, textures) of every material in the open scene, for slot usage.
fn scene_material_textures(mats: &[zestengine_engine::scene::MaterialEntry], root: &std::path::Path) -> Vec<(String, zestengine_core::MaterialTexturePaths)> {
    mats.iter()
        .map(|m| (m.file.as_ref().map(|f| relative_to(f, root)).unwrap_or_else(|| m.model.name.clone()), m.model.textures.clone()))
        .collect()
}

/// Sliders for a `.postfx` profile (the post-processing asset editor). Returns
/// true when any value changed (caller saves + invalidates the cache).
/// Post FX settings editor. `volume`: editing a Volume profile, where each
/// effect only overrides the global post settings while it's enabled.
/// One component's changed fields (None = replace it whole).
type FieldEdit = (String, Option<Vec<(String, String)>>);

/// A body needs a shape: give an object without a collider a Mesh Collider
/// (its own mesh, or its authored collision pieces when it has some), or a
/// Box Collider when it has no mesh.
fn ensure_collider(object: &mut zestengine_engine::scene::SceneObject, ball: bool, authored: bool) {
    let has = object.components.iter().any(|c| {
        let a = c.as_any();
        a.is::<zestengine_core::BoxCollider>()
            || a.is::<zestengine_core::SphereCollider>()
            || a.is::<zestengine_core::CapsuleCollider>()
            || a.is::<zestengine_core::MeshCollider>()
            || a.is::<zestengine_core::CharacterController>()
    });
    if has {
        return;
    }
    // Primitives get their exact shape (a sphere stays perfectly round);
    // other meshes use their mesh; empties a box.
    use zestengine_assets::{ObjectSource, PrimitiveShape};
    let collider: Box<dyn zestengine_core::Component> = match &object.source {
        _ if authored => Box::new(zestengine_core::MeshCollider::default()),
        ObjectSource::Primitive { shape: PrimitiveShape::Sphere } => Box::new(zestengine_core::SphereCollider::default()),
        _ if ball => Box::new(zestengine_core::SphereCollider::default()),
        ObjectSource::Primitive { shape: PrimitiveShape::Capsule } => Box::new(zestengine_core::CapsuleCollider::default()),
        ObjectSource::Primitive { shape: PrimitiveShape::Cube | PrimitiveShape::Plane } => Box::new(zestengine_core::BoxCollider::default()),
        _ if object.render.is_some() => Box::new(zestengine_core::MeshCollider::default()),
        _ => Box::new(zestengine_core::BoxCollider::default()),
    };
    object.components.push(collider);
}

/// Bounds of an object's meshes and its children's, in its local space.
fn object_bounds(scene: &zestengine_engine::scene::LoadedScene, index: usize) -> Option<(Vec3, Vec3)> {
    let world = scene.world_transforms();
    let to_local = world[index].inverse();
    let mut lo = Vec3::splat(f32::MAX);
    let mut hi = Vec3::splat(f32::MIN);
    for (j, w) in world.iter().enumerate() {
        let mut p = Some(j);
        let mut under = false;
        while let Some(k) = p {
            if k == index {
                under = true;
                break;
            }
            p = scene.objects[k].parent;
        }
        if !under {
            continue;
        }
        let m = to_local * *w;
        for r in scene.objects[j].render_slots() {
            let (a, b) = scene.mesh_aabb(r.mesh);
            for c in 0..8 {
                let corner = Vec3::new(if c & 1 == 0 { a.x } else { b.x }, if c & 2 == 0 { a.y } else { b.y }, if c & 4 == 0 { a.z } else { b.z });
                let q = m.transform_point3(corner);
                lo = lo.min(q);
                hi = hi.max(q);
            }
        }
    }
    (lo.x <= hi.x).then_some((lo, hi))
}

/// Size a freshly added Box / Sphere / Capsule Collider to the bounds `lo`..`hi`
/// (local). A flat mesh (a plane) gets a thin slab under its surface.
fn fit_collider(component: &mut dyn std::any::Any, lo: Vec3, hi: Vec3) {
    const MIN: f32 = 0.05;
    let mut size = hi - lo;
    let mut center = (lo + hi) * 0.5;
    for k in 0..3 {
        if size[k] < MIN {
            center[k] -= (MIN - size[k]) * 0.5;
            size[k] = MIN;
        }
    }
    if let Some(b) = component.downcast_mut::<zestengine_core::BoxCollider>() {
        b.center = center.to_array();
        b.size = size.to_array();
    } else if let Some(s) = component.downcast_mut::<zestengine_core::SphereCollider>() {
        s.center = center.to_array();
        s.radius = size.max_element() * 0.5;
    } else if let Some(c) = component.downcast_mut::<zestengine_core::CapsuleCollider>() {
        let axis = if size.x >= size.y && size.x >= size.z { 0 } else if size.y >= size.z { 1 } else { 2 };
        c.direction = [zestengine_core::ColliderAxis::X, zestengine_core::ColliderAxis::Y, zestengine_core::ColliderAxis::Z][axis];
        let across = (0..3).filter(|&k| k != axis).map(|k| size[k]).fold(0.0f32, f32::max) * 0.5;
        c.center = center.to_array();
        c.height = size[axis];
        c.radius = across.min(size[axis] * 0.5);
    }
}

/// New-asset recipes for the inspector's asset fields ([+] next to a slot).
fn asset_templates() -> &'static [crate::AssetTemplate] {
    static TEMPLATES: std::sync::OnceLock<Vec<crate::AssetTemplate>> = std::sync::OnceLock::new();
    TEMPLATES.get_or_init(|| {
        fn ron_of<T: serde::Serialize>(v: &T) -> String {
            ron::ser::to_string_pretty(v, ron::ser::PrettyConfig::default()).unwrap_or_default()
        }
        let t = |ext: &str, name: &str, folder: &str, contents: String| crate::AssetTemplate {
            ext: ext.into(),
            name: name.into(),
            folder: folder.into(),
            contents,
        };
        let mut controller = zestengine_assets::AnimatorController::default();
        controller.layers[0].machine.states.push(zestengine_assets::AnimState { name: "Idle".into(), ..Default::default() });
        controller.layers[0].machine.default_state = "Idle".into();
        let material = zestengine_assets::MaterialFile {
            name: "New Material".into(),
            shader: "standard".into(),
            params: Default::default(),
            features: Default::default(),
            render_queue: None,
            textures: Default::default(),
            custom: Default::default(),
        };
        vec![
            t("waterwaves", "Water Waves", "water", ron_of(&zestengine_core::WaterWaveProfile::default())),
            t(zestengine_assets::AUDIOFX_EXTENSION, "Audio Environment", "audio", ron_of(&zestengine_core::AudioEnvironmentProfile::default())),
            t(zestengine_assets::AUDIO_SETTINGS_EXTENSION, "Audio Settings", "audio", ron_of(&zestengine_core::SpatialAudioSettings::default())),
            t("material", "Material", "materials", ron_of(&material)),
            t(zestengine_core::PHYSMAT_EXTENSION, "Physics Material", "physics", ron_of(&zestengine_core::PhysicsMaterial::default())),
            t(zestengine_core::SPRITE_SHAPE_EXTENSION, "Sprite Shape Profile", "shapes", ron_of(&zestengine_core::SpriteShapeProfile::default())),
            t(zestengine_assets::POSTFX_EXTENSION, "Post FX Profile", "postfx", ron_of(&zestengine_assets::PostFxProfile::default())),
            t(zestengine_assets::ANIM_CONTROLLER_EXTENSION, "Controller", "animation", ron_of(&controller)),
            t(zestengine_assets::ANIM_CLIP_EXTENSION, "Clip", "animation", ron_of(&zestengine_assets::AnimClip::default())),
        ]
    })
}

/// Project-relative paths of every `.physmat` under `root`.
fn physmat_files(root: &Path) -> Vec<String> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<String>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let path = entry.path();
            if name.starts_with('.') || name == "target" {
                continue;
            }
            if path.is_dir() {
                walk(root, &path, out);
            } else if path.extension().is_some_and(|e| e == zestengine_core::PHYSMAT_EXTENSION)
                && let Ok(rel) = path.strip_prefix(root)
            {
                out.push(rel.to_string_lossy().replace('\\', "/"));
            }
        }
    }
    let mut out = Vec::new();
    walk(root, root, &mut out);
    out.sort();
    out
}

// Each property has its own override, so a volume can adjust reflections while
// inheriting GI quality from the environment or a lower-priority volume.
fn flux_override_ui(ui: &mut egui::Ui, label: &str, value: &mut Option<f32>, default: f32,
                    range: std::ops::RangeInclusive<f32>, integer: bool) -> bool {
    let mut changed = false;
    ui.push_id(label, |ui| {
        ui.horizontal(|ui| {
            let mut enabled = value.is_some();
            if ui.checkbox(&mut enabled, label).changed() {
                *value = enabled.then_some(default);
                changed = true;
            }
            if let Some(v) = value {
                let slider = egui::Slider::new(v, range);
                changed |= crate::widgets::slider(ui, if integer { slider.integer() } else { slider }).changed();
            }
        });
    });
    changed
}

fn ray_lighting_mode_ui(ui: &mut egui::Ui, mode: &mut zestengine_render::RayLightingMode, inherit: bool) -> bool {
    let mut changed = false;
    egui::ComboBox::from_id_salt("flux-ray-lighting-mode").selected_text(mode.label()).show_ui(ui, |ui| {
        for choice in zestengine_render::RayLightingMode::ALL {
            if !inherit && choice == zestengine_render::RayLightingMode::Default { continue; }
            changed |= ui.selectable_value(mode, choice, choice.label()).changed();
        }
    }).response.on_hover_text("Hardware tracing only. Hit lighting evaluates material and direct shadows at ray hits; secondary diffuse bounces still use the Surface Cache. Hit Lighting for GI and Reflections has the highest GPU cost.");
    changed
}

fn postfx_editor_ui(ui: &mut egui::Ui, p: &mut zestengine_assets::PostFxProfile, volume: bool) -> bool {
    use zestengine_assets::TonemapMode;
    use egui::{DragValue, RichText};
    let mut changed = false;

    ui.collapsing("Flux Global Illumination", |ui| {
        ui.horizontal(|ui| {
            let mut enabled = p.flux.ray_lighting_mode.is_some();
            if ui.checkbox(&mut enabled, "Ray Lighting Mode").changed() {
                p.flux.ray_lighting_mode = enabled.then_some(zestengine_render::RayLightingMode::Default);
                changed = true;
            }
            if let Some(mode) = &mut p.flux.ray_lighting_mode { changed |= ray_lighting_mode_ui(ui, mode, true); }
        });
        changed |= flux_override_ui(ui, "Scene Lighting Update Speed", &mut p.flux.scene_lighting_update_speed, 1.0, 0.5..=16.0, false);
        changed |= flux_override_ui(ui, "Final Gather Quality", &mut p.flux.final_gather_quality, 1.0, 0.25..=4.0, false);
        changed |= flux_override_ui(ui, "Final Gather Lighting Update Speed", &mut p.flux.final_gather_update_speed, 1.0, 0.5..=8.0, false);
        changed |= flux_override_ui(ui, "Max Trace Distance (m)", &mut p.flux.max_trace_distance, 200.0, 0.1..=2000.0, false);
    });
    ui.collapsing("Flux Reflections", |ui| {
        changed |= flux_override_ui(ui, "Quality", &mut p.flux.reflection_quality, 1.0, 0.25..=4.0, false);
        changed |= flux_override_ui(ui, "Max Roughness To Trace", &mut p.flux.max_roughness_to_trace, 0.6, 0.0..=1.0, false);
        changed |= flux_override_ui(ui, "Max Reflection Bounces", &mut p.flux.max_reflection_bounces, 1.0, 1.0..=8.0, true);
    });
    ui.separator();

    if volume {
        crate::widgets::info_note(
            ui,
            "Only enabled effects override the scene's global post settings (Environment > Post).",
        );
        changed |= ui.checkbox(&mut p.tonemap.enabled, "Tonemap").changed();
    } else {
        ui.label(RichText::new("Tonemap").strong());
    }
    ui.add_enabled_ui(!volume || p.tonemap.enabled, |ui| {
        ui.horizontal(|ui| {
            ui.label("Mode");
            egui::ComboBox::from_id_salt("postfx-tonemap")
                .selected_text(p.tonemap.mode.label())
                .show_ui(ui, |ui| {
                    for m in TonemapMode::ALL {
                        changed |= ui.selectable_value(&mut p.tonemap.mode, m, m.label()).changed();
                    }
                });
        });
        ui.horizontal(|ui| {
            ui.label(if p.tonemap.auto_exposure.enabled {"Exposure compensation (EV)"} else {"Exposure (EV)"});
            changed |= crate::widgets::slider(ui, egui::Slider::new(&mut p.tonemap.exposure, -8.0..=8.0)).changed();
        });
        changed |= ui.checkbox(&mut p.tonemap.auto_exposure.enabled,"Auto exposure (histogram)").changed();
        if p.tonemap.auto_exposure.enabled {
            let a=&mut p.tonemap.auto_exposure;
            for (label,value,range) in [
                ("Minimum EV100",&mut a.min_ev100,-20.0..=24.0),
                ("Maximum EV100",&mut a.max_ev100,-20.0..=24.0),
                ("Speed up (stops/s)",&mut a.speed_up,0.0..=20.0),
                ("Speed down (stops/s)",&mut a.speed_down,0.0..=20.0),
            ] {ui.horizontal(|ui| {ui.label(label);changed|=crate::widgets::slider(ui,egui::Slider::new(value,range)).changed();});}
            ui.collapsing("Exposure metering",|ui| {
                for (label,value,range) in [
                    ("Low percentile",&mut a.low_percent,0.0..=99.0),
                    ("High percentile",&mut a.high_percent,1.0..=100.0),
                    ("Histogram minimum EV100",&mut a.histogram_min_ev100,-20.0..=23.0),
                    ("Histogram maximum EV100",&mut a.histogram_max_ev100,-19.0..=24.0),
                ] {ui.horizontal(|ui| {ui.label(label);changed|=crate::widgets::slider(ui,egui::Slider::new(value,range)).changed();});}
            });
        }
    });

    ui.separator();
    changed |= ui.checkbox(&mut p.color_grading.enabled, "Color Grading").changed();
    ui.add_enabled_ui(p.color_grading.enabled, |ui| {
        let g = &mut p.color_grading;
        ui.horizontal(|ui| {
            ui.label("Post Exposure");
            changed |= crate::widgets::slider(ui, egui::Slider::new(&mut g.exposure, -4.0..=4.0)).changed();
        });
        ui.horizontal(|ui| {
            ui.label("Contrast");
            changed |= crate::widgets::slider(ui, egui::Slider::new(&mut g.contrast, 0.0..=2.0)).changed();
        });
        ui.horizontal(|ui| {
            ui.label("Saturation");
            changed |= crate::widgets::slider(ui, egui::Slider::new(&mut g.saturation, 0.0..=2.0)).changed();
        });
        ui.horizontal(|ui| {
            ui.label("Temperature");
            changed |= crate::widgets::slider(ui, egui::Slider::new(&mut g.temperature, -1.0..=1.0)).changed();
        });
        ui.horizontal(|ui| {
            ui.label("Tint");
            changed |= crate::widgets::slider(ui, egui::Slider::new(&mut g.tint, -1.0..=1.0)).changed();
        });
    });

    ui.separator();
    changed |= ui.checkbox(&mut p.vignette.enabled, "Vignette").changed();
    ui.add_enabled_ui(p.vignette.enabled, |ui| {
        let v = &mut p.vignette;
        ui.horizontal(|ui| {
            ui.label("Intensity");
            changed |= crate::widgets::slider(ui, egui::Slider::new(&mut v.intensity, 0.0..=1.0)).changed();
        });
        ui.horizontal(|ui| {
            ui.label("Smoothness");
            changed |= crate::widgets::slider(ui, egui::Slider::new(&mut v.smoothness, 0.0..=1.0)).changed();
        });
        ui.horizontal(|ui| {
            ui.label("Color");
            changed |= ui.color_edit_button_rgb(&mut v.color).changed();
        });
    });

    ui.separator();
    changed |= ui.checkbox(&mut p.bloom.enabled, "Bloom").changed();
    ui.add_enabled_ui(p.bloom.enabled, |ui| {
        let b = &mut p.bloom;
        ui.horizontal(|ui| {
            ui.label("Threshold");
            changed |= ui.add(DragValue::new(&mut b.threshold).speed(0.02).range(0.0..=10.0)).changed();
        });
        ui.horizontal(|ui| {
            ui.label("Intensity");
            changed |= crate::widgets::slider(ui, egui::Slider::new(&mut b.intensity, 0.0..=3.0)).changed();
        });
        ui.horizontal(|ui| {
            ui.label("Radius");
            changed |= crate::widgets::slider(ui, egui::Slider::new(&mut b.radius, 0.0..=1.0)).changed();
        });
    });

    ui.separator();
    changed |= ui
        .checkbox(&mut p.chromatic_aberration.enabled, "Chromatic Aberration (needs HDR pass)")
        .changed();
    ui.add_enabled_ui(p.chromatic_aberration.enabled, |ui| {
        ui.horizontal(|ui| {
            ui.label("Intensity");
            changed |= crate::widgets::slider(
                ui,
                egui::Slider::new(&mut p.chromatic_aberration.intensity, 0.0..=2.0))
                .changed();
        });
    });

    ui.separator();
    changed |= ui
        .checkbox(&mut p.ambient_occlusion.enabled, "Ambient Occlusion (indirect light only)")
        .on_hover_text("Screen-space AO; the algorithm (SAO / GTAO) is a Project Setting. Off in VR.")
        .changed();
    ui.add_enabled_ui(p.ambient_occlusion.enabled, |ui| {
        let ao = &mut p.ambient_occlusion;
        ui.horizontal(|ui| {
            ui.label("Intensity");
            changed |= crate::widgets::slider(ui, egui::Slider::new(&mut ao.intensity, 0.0..=4.0)).changed();
        });
        ui.horizontal(|ui| {
            ui.label("Radius (m)");
            changed |= crate::widgets::slider(ui, egui::Slider::new(&mut ao.radius, 0.05..=5.0).logarithmic(true)).changed();
        });
        ui.horizontal(|ui| {
            ui.label("Falloff");
            changed |= crate::widgets::slider(ui, egui::Slider::new(&mut ao.falloff, 0.05..=1.0)).changed();
        });
        changed |= ui.checkbox(&mut ao.multi_bounce, "Multi-bounce").changed();
    });

    changed
}

/// Liang–Barsky: clip a 2D segment to a rect. None = fully outside.
fn clip_segment_to_rect(a: egui::Pos2, b: egui::Pos2, rect: egui::Rect) -> Option<[egui::Pos2; 2]> {
    let d = b - a;
    let mut t0 = 0.0f32;
    let mut t1 = 1.0f32;
    for (p, q) in [
        (-d.x, a.x - rect.left()),
        (d.x, rect.right() - a.x),
        (-d.y, a.y - rect.top()),
        (d.y, rect.bottom() - a.y),
    ] {
        if p == 0.0 {
            if q < 0.0 {
                return None; // parallel and outside
            }
        } else {
            let r = q / p;
            if p < 0.0 {
                t0 = t0.max(r);
            } else {
                t1 = t1.min(r);
            }
            if t0 > t1 {
                return None;
            }
        }
    }
    Some([a + d * t0, a + d * t1])
}

/// All project-relative paths with the given extension, sorted.
fn scan_project_files(root: &Path, ext: &str) -> Vec<String> {
    scan_project_exts(root, &[ext])
}

/// All project-relative paths with any of the given extensions, sorted, in a
/// single walk.
fn scan_project_exts(root: &Path, exts: &[&str]) -> Vec<String> {
    fn walk(dir: &Path, root: &Path, exts: &[&str], depth: usize, out: &mut Vec<String>) {
        if depth > 8 {
            return;
        }
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') || name == "target" {
                continue;
            }
            if path.is_dir() {
                walk(&path, root, exts, depth + 1, out);
            } else if path.extension().is_some_and(|e| exts.iter().any(|x| e == *x)) {
                out.push(relative_to(&path, root));
            }
        }
    }
    let mut out = Vec::new();
    walk(root, root, exts, 0, &mut out);
    out.sort();
    out
}

/// Pre-play object state, restored on Stop (matched by id).
struct PlayObjectState {
    id: ObjectId,
    translation: glam::Vec3,
    rotation: glam::Quat,
    scale: glam::Vec3,
    enabled: bool,
    components: Vec<zestengine_assets::ComponentData>,
}

struct PlaySnapshot {
    objects: Vec<PlayObjectState>,
}

fn play_object_state(o: &scene::SceneObject) -> PlayObjectState {
    PlayObjectState {
        id: o.id,
        translation: o.translation,
        rotation: o.rotation,
        scale: o.scale,
        enabled: o.enabled,
        components: o.save_components(),
    }
}

/// Values typed into the Inspector for a multi-selection: each `Some` is set
/// on every selected object (transform axes as absolute values).
#[derive(Clone, Copy, Debug, Default)]
struct MultiEdit {
    translation: [Option<f32>; 3],
    rotation_deg: [Option<f32>; 3],
    scale: [Option<f32>; 3],
}

impl MultiEdit {
    fn any(&self) -> bool {
        self.translation.iter().chain(&self.rotation_deg).chain(&self.scale).any(Option::is_some)
    }

    fn apply(&self, o: &mut scene::SceneObject) {
        let mut t = o.translation.to_array();
        let mut s = o.scale.to_array();
        let mut r = zestengine_engine::transform_edit::euler_deg(o.rotation);
        for k in 0..3 {
            if let Some(v) = self.translation[k] {
                t[k] = v;
            }
            if let Some(v) = self.scale[k] {
                s[k] = v;
            }
            if let Some(v) = self.rotation_deg[k] {
                r[k] = v;
            }
        }
        o.translation = glam::Vec3::from(t);
        o.scale = glam::Vec3::from(s);
        if self.rotation_deg.iter().any(Option::is_some) {
            o.rotation = zestengine_engine::transform_edit::euler_to_quat(r);
        }
    }
}

fn object_state(o: &scene::SceneObject) -> ObjectState {
    ObjectState {
        name: o.name.clone(),
        translation: o.translation,
        rotation: o.rotation,
        scale: o.scale,
        net_role: o.net_role,
        component_roles: o.component_roles.clone(),
        components: o.save_components(),
    }
}

/// Draw one hand-drawn light bulb: glass disc at `glass`, screw base on the
/// far side of `up` (the direction the bulb points). egui's font has no 💡
/// glyph, so lights and probe volumes share this primitive.
fn draw_bulb(painter: &egui::Painter, glass: egui::Pos2, up: egui::Vec2, scale: f32, selected: bool) {
    let glass_col = if selected {
        egui::Color32::from_rgb(255, 242, 150)
    } else {
        egui::Color32::from_rgb(235, 215, 120)
    };
    let dark = egui::Color32::from_rgb(70, 62, 35);
    let up = up.normalized();
    let right = egui::vec2(-up.y, up.x);
    let r = 7.0 * scale;
    if selected {
        // Emitted-light rays.
        for k in 0..8 {
            let a = k as f32 / 8.0 * std::f32::consts::TAU;
            let d = egui::vec2(a.cos(), a.sin());
            painter.line_segment(
                [glass + d * (r + 2.0 * scale), glass + d * (r + 5.0 * scale)],
                egui::Stroke::new(1.5_f32, glass_col),
            );
        }
    }
    painter.circle_filled(glass, r, glass_col);
    painter.circle_stroke(glass, r, egui::Stroke::new(1.0_f32, dark));
    // Screw base: a small block on the far side of `up`, with a thread line.
    let bc = glass - up * (r + 1.5 * scale);
    let hw = r * 0.55;
    let hh = 2.5 * scale;
    let base = vec![
        bc + right * hw + up * hh,
        bc - right * hw + up * hh,
        bc - right * hw - up * hh,
        bc + right * hw - up * hh,
    ];
    painter.add(egui::Shape::convex_polygon(base, dark, egui::Stroke::NONE));
    painter.line_segment(
        [bc - right * (hw - 1.0), bc + right * (hw - 1.0)],
        egui::Stroke::new(1.0_f32, glass_col),
    );
}

/// Hover/active emphasis for a draggable handle (probe volumes, and the
/// box/sphere colliders that reuse this): slightly bigger + a bit brighter
/// when emphasized, the base color otherwise. The pointer stays normal.
fn handle_style(emphasized: bool, base: egui::Color32) -> (f32, egui::Color32) {
    if emphasized {
        // Lighten toward white without going fully white.
        let lift = |c: u8| c.saturating_add((255 - c) / 2);
        (
            1.25,
            egui::Color32::from_rgb(lift(base.r()), lift(base.g()), lift(base.b())),
        )
    } else {
        (1.0, base)
    }
}

/// Draw a resize-handle arrow at `at` pointing along screen-space `dir`
/// (away from the box). A short shaft + filled triangular head.
/// Distance from `press` to the drawn arrow handle: the segment from `at` (face
/// center) outward along `dir` for `len` pixels (covering shaft + arrowhead).
/// Lets the whole arrow be a click target, not just the base.
fn handle_pick_dist(press: egui::Pos2, at: egui::Pos2, dir: egui::Vec2, len: f32) -> f32 {
    let d = if dir.length() > 1e-3 {
        dir.normalized()
    } else {
        return press.distance(at);
    };
    let t = (press - at).dot(d).clamp(0.0, len);
    press.distance(at + d * t)
}

fn draw_arrow_handle(
    painter: &egui::Painter,
    at: egui::Pos2,
    dir: egui::Vec2,
    scale: f32,
    color: egui::Color32,
) {
    let dir = if dir.length() > 1e-3 {
        dir.normalized()
    } else {
        egui::vec2(0.0, -1.0)
    };
    let perp = egui::vec2(-dir.y, dir.x);
    let head = 9.0 * scale;
    let half_w = 5.5 * scale;
    let shaft = 6.0 * scale;
    // Shaft from just outside the face toward the tip.
    let base = at + dir * shaft;
    painter.line_segment([at, base], egui::Stroke::new(2.5 * scale, color));
    // Arrowhead triangle.
    let tip = base + dir * head;
    let l = base + perp * half_w;
    let r = base - perp * half_w;
    painter.add(egui::Shape::convex_polygon(
        vec![tip, l, r],
        color,
        egui::Stroke::NONE,
    ));
}

/// Wind Zone billboard: three horizontal gusts, each with an arrowhead.
fn draw_wind_icon(painter: &egui::Painter, center: egui::Pos2, selected: bool, scale: f32) {
    let color = if selected { egui::Color32::from_rgb(190, 235, 255) } else { egui::Color32::from_rgb(120, 185, 215) };
    for k in 0..3 {
        let y = center.y + (k as f32 - 1.0) * 4.0 * scale;
        let x = center.x + (k as f32 - 1.0) * 2.5 * scale;
        let tip = egui::pos2(x + 7.0 * scale, y);
        painter.line_segment([egui::pos2(x - 7.0 * scale, y), tip], egui::Stroke::new(1.2_f32, color));
        painter.add(egui::Shape::convex_polygon(
            vec![tip, tip - egui::vec2(3.0 * scale, 1.6 * scale), tip - egui::vec2(3.0 * scale, -1.6 * scale)],
            color,
            egui::Stroke::NONE,
        ));
    }
}

/// NavMeshSurface billboard: a tilted quad with a grid.
fn draw_nav_icon(painter: &egui::Painter, center: egui::Pos2, selected: bool, scale: f32) {
    let color = if selected { egui::Color32::from_rgb(190, 250, 190) } else { egui::Color32::from_rgb(110, 195, 135) };
    let dark = egui::Color32::from_rgb(50, 95, 60);
    let s = 6.0 * scale;
    let (tl, tr, br, bl) = (
        center + egui::vec2(-s, -s * 0.6),
        center + egui::vec2(s, -s * 0.6),
        center + egui::vec2(s, s * 0.6),
        center + egui::vec2(-s, s * 0.6),
    );
    painter.add(egui::Shape::convex_polygon(vec![tl, tr, br, bl], egui::Color32::from_rgb(55, 90, 65), egui::Stroke::new(1.0_f32, color)));
    painter.line_segment([center + egui::vec2(0.0, -s * 0.6), center + egui::vec2(0.0, s * 0.6)], egui::Stroke::new(1.0_f32, dark));
    painter.line_segment([center + egui::vec2(-s, 0.0), center + egui::vec2(s, 0.0)], egui::Stroke::new(1.0_f32, dark));
}

/// ParticleEmitter billboard: a centre dot with a radiating burst.
fn draw_particle_icon(painter: &egui::Painter, center: egui::Pos2, selected: bool, scale: f32) {
    let color = if selected { egui::Color32::from_rgb(255, 225, 150) } else { egui::Color32::from_rgb(225, 165, 80) };
    painter.circle_filled(center, 2.0 * scale, color);
    for k in 0..6 {
        let a = (k as f32 / 6.0) * std::f32::consts::TAU;
        painter.circle_filled(center + egui::vec2(a.cos(), a.sin()) * 6.0 * scale, 1.4 * scale, color);
    }
}

/// ScenePortal billboard: a doorframe with a dark opening.
fn draw_portal_icon(painter: &egui::Painter, center: egui::Pos2, selected: bool, scale: f32) {
    let color = if selected { egui::Color32::from_rgb(215, 185, 255) } else { egui::Color32::from_rgb(150, 115, 205) };
    let frame = egui::Rect::from_center_size(center, egui::vec2(13.0 * scale, 15.0 * scale));
    painter.rect_filled(frame, 1.5 * scale, egui::Color32::from_rgb(35, 28, 55));
    painter.rect_stroke(frame, 1.5 * scale, egui::Stroke::new(1.4_f32, color), egui::StrokeKind::Inside);
    let open = egui::Rect::from_center_size(center + egui::vec2(0.0, 1.0 * scale), egui::vec2(8.0 * scale, 10.0 * scale));
    painter.rect_stroke(open, 1.0 * scale, egui::Stroke::new(1.0_f32, color), egui::StrokeKind::Inside);
}

/// Terrain billboard: two mountain peaks.
fn draw_terrain_icon(painter: &egui::Painter, center: egui::Pos2, selected: bool, scale: f32) {
    let color = if selected { egui::Color32::from_rgb(190, 250, 190) } else { egui::Color32::from_rgb(110, 190, 115) };
    let base_y = center.y + 6.0 * scale;
    let left = egui::pos2(center.x - 8.0 * scale, base_y);
    let peak = egui::pos2(center.x, center.y - 6.0 * scale);
    let right = egui::pos2(center.x + 8.0 * scale, base_y);
    painter.add(egui::Shape::convex_polygon(vec![left, peak, right], color, egui::Stroke::NONE));
    let p2 = egui::pos2(center.x + 6.0 * scale, center.y - 1.5 * scale);
    painter.add(egui::Shape::convex_polygon(vec![egui::pos2(center.x + 2.0 * scale, base_y), p2, egui::pos2(center.x + 8.5 * scale, base_y)], color, egui::Stroke::NONE));
}

/// Draw a light-bulb billboard (glass + screw base) centered at `center`.
fn draw_light_icon(painter: &egui::Painter, center: egui::Pos2, selected: bool, scale: f32) {
    draw_bulb(painter, center - egui::vec2(0.0, 2.0 * scale), egui::vec2(0.0, -1.0), scale, selected);
}

/// Draw the light-probe-volume emblem: three bulbs (the same image as the
/// light widget) fanned from a shared base at -42° / 0° / +42°, like Unity's
/// Light Probe Group icon.
fn draw_probe_icon(painter: &egui::Painter, center: egui::Pos2, selected: bool, scale: f32) {
    let base = center + egui::vec2(0.0, 8.0 * scale);
    let bulb_scale = scale * 0.62;
    let stem = 11.0 * scale;
    for deg in [-42.0_f32, 0.0, 42.0] {
        let a = deg.to_radians();
        // 0° points straight up; +deg tilts toward +X (screen-right).
        let dir = egui::vec2(a.sin(), -a.cos());
        draw_bulb(painter, base + dir * stem, dir, bulb_scale, selected);
    }
}

/// Draw a reflection-probe icon: a chrome sphere with a specular highlight (so
/// it reads as a mirror ball, distinct from the light-probe bulbs).
fn draw_reflection_probe_icon(
    painter: &egui::Painter,
    center: egui::Pos2,
    selected: bool,
    scale: f32,
) {
    let r = 8.0 * scale;
    let rim = if selected {
        egui::Color32::from_rgb(150, 220, 255)
    } else {
        egui::Color32::from_rgb(110, 170, 210)
    };
    // Chrome body: a vertical light→dark gradient faked with two stacked circles.
    painter.circle_filled(center, r, egui::Color32::from_rgb(70, 90, 110));
    painter.circle_filled(
        center - egui::vec2(0.0, r * 0.28),
        r * 0.72,
        egui::Color32::from_rgb(120, 150, 175),
    );
    // Specular highlight dot, upper-left.
    painter.circle_filled(
        center + egui::vec2(-r * 0.34, -r * 0.40),
        r * 0.24,
        egui::Color32::from_rgb(235, 245, 255),
    );
    painter.circle_stroke(center, r, egui::Stroke::new(1.5_f32, rim));
}

/// Draw a speaker icon (cabinet + cone + sound waves) centered at `center`.
/// Post FX volume billboard: the volume icon on a dark disc.
fn draw_post_volume_icon(painter: &egui::Painter, center: egui::Pos2, selected: bool, scale: f32) {
    use crate::theme;
    let r = 8.5 * scale;
    painter.circle_filled(center, r, theme::with_alpha(theme::BG_APP, 0.85));
    let color = if selected { egui::Color32::WHITE } else { theme::TYPE_POSTFX };
    painter.circle_stroke(center, r, egui::Stroke::new(1.5_f32, theme::TYPE_POSTFX));
    crate::icons::paint(
        painter,
        egui::Rect::from_center_size(center, egui::vec2(r * 1.25, r * 1.25)),
        crate::icons::Icon::PostFxVolume,
        color,
    );
}

/// UI canvas billboard: a small window with a title bar and two buttons.
fn draw_ui_canvas_icon(painter: &egui::Painter, center: egui::Pos2, selected: bool, scale: f32) {
    let body = if selected {
        egui::Color32::from_rgb(210, 250, 240)
    } else {
        egui::Color32::from_rgb(160, 215, 200)
    };
    let dark = egui::Color32::from_rgb(40, 65, 60);
    let frame = egui::Rect::from_center_size(center, egui::vec2(16.0 * scale, 12.0 * scale));
    painter.rect_filled(frame, 1.5 * scale, body);
    painter.rect_stroke(frame, 1.5 * scale, egui::Stroke::new(1.0_f32, dark), egui::StrokeKind::Inside);
    let bar_y = frame.top() + 3.0 * scale;
    painter.line_segment(
        [egui::pos2(frame.left(), bar_y), egui::pos2(frame.right(), bar_y)],
        egui::Stroke::new(1.0_f32, dark),
    );
    for k in 0..2 {
        let y = bar_y + (3.0 + k as f32 * 3.5) * scale;
        let button = egui::Rect::from_min_max(
            egui::pos2(frame.left() + 3.0 * scale, y - 1.0 * scale),
            egui::pos2(frame.right() - 3.0 * scale, y + 1.0 * scale),
        );
        painter.rect_filled(button, 0.5 * scale, dark);
    }
}

fn draw_audio_icon(painter: &egui::Painter, center: egui::Pos2, selected: bool, scale: f32) {
    let body = if selected {
        egui::Color32::from_rgb(235, 225, 255)
    } else {
        egui::Color32::from_rgb(200, 190, 220)
    };
    let dark = egui::Color32::from_rgb(55, 50, 70);
    // Square cabinet on the left, trapezoid cone opening to the right.
    let cab = egui::Rect::from_center_size(
        center + egui::vec2(-5.0 * scale, 0.0),
        egui::vec2(5.0 * scale, 7.0 * scale),
    );
    painter.rect_filled(cab, 1.0 * scale, body);
    let cone = vec![
        egui::pos2(cab.right(), center.y - 3.0 * scale),
        egui::pos2(center.x + 3.0 * scale, center.y - 7.0 * scale),
        egui::pos2(center.x + 3.0 * scale, center.y + 7.0 * scale),
        egui::pos2(cab.right(), center.y + 3.0 * scale),
    ];
    painter.add(egui::Shape::convex_polygon(
        cone,
        body,
        egui::Stroke::new(1.0_f32, dark),
    ));
    painter.rect_stroke(cab, 1.0 * scale, egui::Stroke::new(1.0_f32, dark), egui::StrokeKind::Inside);
    // Sound waves: two arcs to the right (brighter when selected).
    let wave = if selected {
        egui::Color32::from_rgb(150, 220, 255)
    } else {
        egui::Color32::from_rgb(120, 170, 200)
    };
    for (k, r) in [(0u8, 4.0_f32), (1, 7.0)] {
        let cx = center.x + 5.0 * scale;
        let steps = 7;
        let mut prev: Option<egui::Pos2> = None;
        for s in 0..=steps {
            // -50°..50° arc opening right.
            let a = (-0.9 + 1.8 * s as f32 / steps as f32) * std::f32::consts::FRAC_PI_2;
            let p = egui::pos2(cx + a.cos() * r * scale, center.y + a.sin() * r * scale);
            if let Some(pp) = prev {
                painter.line_segment([pp, p], egui::Stroke::new(1.0 + k as f32 * 0.0, wave));
            }
            prev = Some(p);
        }
    }
}

/// Draw a little video-camera icon (body + lens) centered at `center`.
/// Selection ring around a billboard icon (lights, cameras, probes, ...) in the
/// selection outline colour, so a selected gizmo is unmistakable.
fn draw_selection_ring(painter: &egui::Painter, center: egui::Pos2, scale: f32) {
    let c = crate::theme::selection_outline();
    let r = 14.0 * scale.max(0.6);
    painter.circle_filled(center, r, crate::theme::with_alpha(c, 0.14));
    painter.circle_stroke(center, r, egui::Stroke::new(2.0_f32, c));
}

/// Marker for a selected object that has neither a mesh nor a billboard (an empty,
/// a group, a collider-only object): a ring with a crosshair at its origin.
fn draw_selection_marker(painter: &egui::Painter, center: egui::Pos2) {
    let c = crate::theme::selection_outline();
    let stroke = egui::Stroke::new(2.0_f32, c);
    painter.circle_filled(center, 9.0, crate::theme::with_alpha(c, 0.14));
    painter.circle_stroke(center, 9.0, stroke);
    for d in [egui::vec2(1.0, 0.0), egui::vec2(0.0, 1.0)] {
        painter.line_segment([center - d * 15.0, center + d * 15.0], stroke);
    }
    painter.circle_filled(center, 2.0, c);
}

fn draw_camera_icon(painter: &egui::Painter, center: egui::Pos2, selected: bool, scale: f32) {
    let body_col = if selected {
        egui::Color32::from_rgb(225, 238, 255)
    } else {
        egui::Color32::from_rgb(190, 210, 235)
    };
    let dark = egui::Color32::from_rgb(40, 55, 75);
    let body = egui::Rect::from_min_size(
        center + egui::vec2(-9.0 * scale, -5.0 * scale),
        egui::vec2(12.0 * scale, 10.0 * scale),
    );
    painter.rect_filled(body, 2.0, body_col);
    painter.rect_stroke(
        body,
        2.0,
        egui::Stroke::new(1.0_f32, dark),
        egui::StrokeKind::Inside,
    );
    // Lens: a trapezoid jutting to the right (classic video-camera silhouette).
    let lx = body.right();
    let cy = center.y;
    let lens = vec![
        egui::pos2(lx, cy - 2.5 * scale),
        egui::pos2(lx + 6.0 * scale, cy - 4.5 * scale),
        egui::pos2(lx + 6.0 * scale, cy + 4.5 * scale),
        egui::pos2(lx, cy + 2.5 * scale),
    ];
    painter.add(egui::Shape::convex_polygon(
        lens,
        body_col,
        egui::Stroke::new(1.0_f32, dark),
    ));
    // Lens dot on the body front.
    painter.circle_filled(egui::pos2(body.left() + 4.0 * scale, cy), 1.6 * scale, dark);
}

/// Char index → (line, utf-8 byte column) for an LSP position (we negotiate
/// utf-8 position encoding).
fn char_to_line_col(text: &str, char_index: usize) -> (u32, u32) {
    let mut line = 0u32;
    let mut col = 0u32;
    for (i, ch) in text.chars().enumerate() {
        if i == char_index {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 0;
        } else {
            col += ch.len_utf8() as u32;
        }
    }
    (line, col)
}

/// Char index of the start of the identifier word ending at `cursor`.
fn word_start(text: &str, cursor: usize) -> usize {
    let chars: Vec<char> = text.chars().collect();
    let mut start = cursor.min(chars.len());
    while start > 0 {
        let c = chars[start - 1];
        if c.is_alphanumeric() || c == '_' {
            start -= 1;
        } else {
            break;
        }
    }
    start
}

/// Route a completion/hover response into the matching open editor.
fn apply_lsp_response(editors: &mut [OpenEditor], kind: LspRequestKind, result: serde_json::Value) {
    match kind {
        LspRequestKind::Completion { path, anchor_char, explicit } => {
            let arr = result
                .get("items")
                .and_then(|i| i.as_array())
                .or_else(|| result.as_array());
            let items: Vec<crate::CompletionItem> = arr
                .map(|arr| {
                    arr.iter()
                        .map(|it| {
                            let label = it
                                .get("label")
                                .and_then(|l| l.as_str())
                                .unwrap_or("")
                                .to_owned();
                            let mut insert_text = it
                                .pointer("/textEdit/newText")
                                .and_then(|t| t.as_str())
                                .or_else(|| it.get("insertText").and_then(|t| t.as_str()))
                                .unwrap_or(&label)
                                .to_owned();
                            // Snippet completions (function calls): expand the
                            // `${1:…}` placeholders into the visible text and
                            // keep their tab stops.
                            let (mut stops, mut exit) = (Vec::new(), None);
                            if it.get("insertTextFormat").and_then(|v| v.as_u64()) == Some(2) {
                                let (expanded, s, e) = crate::code_editor::expand_snippet(&insert_text);
                                insert_text = expanded;
                                stops = s;
                                exit = e;
                            }
                            let detail = it
                                .get("detail")
                                .and_then(|d| d.as_str())
                                .unwrap_or("")
                                .to_owned();
                            crate::CompletionItem {
                                label,
                                insert_text,
                                detail,
                                stops,
                                exit,
                            }
                        })
                        .collect()
                })
                .unwrap_or_default();
            if !items.is_empty()
                && let Some(editor) = editors.iter_mut().find(|e| e.path == path)
            {
                // A refreshed popup keeps its explicit + navigated flags (they
                // describe how the USER opened and used it, not the request).
                let navigated = editor.completion.as_ref().is_some_and(|s| s.navigated);
                editor.completion = Some(crate::CompletionState {
                    items,
                    selected: 0,
                    anchor_char,
                    explicit: explicit || editor.completion.as_ref().is_some_and(|s| s.explicit),
                    navigated,
                });
            }
        }
        LspRequestKind::Hover { path } => {
            let text = hover_text(&result);
            if let Some(editor) = editors.iter_mut().find(|e| e.path == path) {
                editor.hover = if text.is_empty() {
                    None
                } else {
                    Some(crate::HoverState { text })
                };
            }
        }
        // Definition jumps are handled in `pump_lsp` (they need to open files).
        LspRequestKind::Definition => {}
        LspRequestKind::References { path } => {
            let refs = parse_references(&result);
            if let Some(editor) = editors.iter_mut().find(|e| e.path == path) {
                editor.references = if refs.is_empty() { None } else { Some(refs) };
            }
        }
        LspRequestKind::CodeAction { path } => {
            let acts = parse_code_actions(&result, &path);
            if let Some(editor) = editors.iter_mut().find(|e| e.path == path) {
                editor.actions = acts;
            }
        }
    }
}

/// Parse an LSP `textDocument/codeAction` result into pickable quick fixes,
/// keeping only the edits that touch `path` (the cursor's file).
fn parse_code_actions(result: &serde_json::Value, path: &Path) -> Vec<crate::CodeActionItem> {
    let Some(arr) = result.as_array() else {
        return Vec::new();
    };
    let uri = format!("file://{}", path.display());
    arr.iter()
        .filter_map(|action| {
            let title = action.get("title")?.as_str()?.to_owned();
            let changes = action.pointer("/edit/changes")?.as_object()?;
            let edits = changes.get(&uri)?.as_array()?;
            let mut out = Vec::new();
            for e in edits {
                let start = e.pointer("/range/start")?;
                let end = e.pointer("/range/end")?;
                let new_text = e.get("newText")?.as_str()?.to_owned();
                out.push((
                    start.get("line")?.as_u64()? as u32,
                    start.get("character")?.as_u64()? as u32,
                    end.get("line")?.as_u64()? as u32,
                    end.get("character")?.as_u64()? as u32,
                    new_text,
                ));
            }
            Some(crate::CodeActionItem { title, edits: out })
        })
        .filter(|a| !a.edits.is_empty())
        .collect()
}

/// Parse an LSP `textDocument/references` result (array of `Location`) into a
/// pickable list.
fn parse_references(result: &serde_json::Value) -> Vec<crate::ReferenceItem> {
    let Some(arr) = result.as_array() else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|loc| {
            let uri = loc.get("uri").and_then(|u| u.as_str())?;
            let path = uri.strip_prefix("file://").map(PathBuf::from)?;
            let range = loc.get("range")?;
            let line = range.pointer("/start/line").and_then(|l| l.as_u64())? as u32;
            let col = range
                .pointer("/start/character")
                .and_then(|c| c.as_u64())
                .unwrap_or(0) as u32;
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            Some(crate::ReferenceItem {
                label: format!("{name}:{}", line + 1),
                path,
                line,
                col,
            })
        })
        .collect()
}

/// Parse an LSP `textDocument/definition` result into `(path, line, col)`.
///
/// Accepts a single `Location`, an array of `Location`, or an array of
/// `LocationLink` (the `target*` shape), taking the first entry.
fn parse_definition(result: &serde_json::Value) -> Option<(PathBuf, u32, u32)> {
    let loc = if result.is_array() {
        result.get(0)?
    } else {
        result
    };
    // LocationLink uses `targetUri`/`targetSelectionRange`; Location uses
    // `uri`/`range`.
    let uri = loc
        .get("uri")
        .or_else(|| loc.get("targetUri"))
        .and_then(|u| u.as_str())?;
    let path = uri.strip_prefix("file://").map(PathBuf::from)?;
    let range = loc
        .get("range")
        .or_else(|| loc.get("targetSelectionRange"))
        .or_else(|| loc.get("targetRange"))?;
    let line = range.pointer("/start/line").and_then(|l| l.as_u64())? as u32;
    let col = range
        .pointer("/start/character")
        .and_then(|c| c.as_u64())
        .unwrap_or(0) as u32;
    Some((path, line, col))
}

/// Extract plain text from an LSP `hover.contents` value.
fn hover_text(result: &serde_json::Value) -> String {
    let Some(contents) = result.get("contents") else {
        return String::new();
    };
    if let Some(s) = contents.as_str() {
        return s.to_owned();
    }
    if let Some(v) = contents.get("value").and_then(|v| v.as_str()) {
        return v.to_owned();
    }
    if let Some(arr) = contents.as_array() {
        return arr
            .iter()
            .map(|c| {
                c.as_str()
                    .or_else(|| c.get("value").and_then(|v| v.as_str()))
                    .unwrap_or("")
            })
            .collect::<Vec<_>>()
            .join("\n");
    }
    String::new()
}

/// A unique sibling path for duplicating `src`: `<stem>_copy.<ext>` (numbered
/// if taken). Works for files and directories.
/// First project asset file (recursively) modified after `since`, or None.
/// Skips `target/` and dot-directories. Used by the focus-regain reimport.
fn changed_assets(root: &Path, since: std::time::SystemTime, out: &mut Vec<PathBuf>) {
    const EXTS: &[&str] = &[
        "fbx", "gltf", "glb", "obj", "png", "jpg", "jpeg", "tga", "hdr", "material", "frag",
        "hlsl", "scene", "postfx", "wav", "flac", "mp3",
    ];
    let Ok(entries) = std::fs::read_dir(root) else { return };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') || name == "target" {
            continue;
        }
        let path = entry.path();
        let Ok(ft) = entry.file_type() else { continue };
        if ft.is_dir() {
            changed_assets(&path, since, out);
        } else if let Some(ext) = path.extension().and_then(|x| x.to_str())
            && EXTS.iter().any(|e| e.eq_ignore_ascii_case(ext))
            && let Ok(modified) = entry.metadata().and_then(|m| m.modified())
            && modified > since
        {
            out.push(path);
        }
    }
}

fn duplicate_file_path(src: &Path) -> Option<PathBuf> {
    let dir = src.parent()?;
    let stem = src.file_stem()?.to_string_lossy();
    let ext = src
        .extension()
        .map(|e| e.to_string_lossy().into_owned())
        .unwrap_or_default();
    Some(unique_path(dir, &format!("{stem}_copy"), &ext))
}

fn unique_path(dir: &Path, stem: &str, ext: &str) -> PathBuf {
    let make = |n: u32| {
        let name = if n == 0 {
            stem.to_owned()
        } else {
            format!("{stem}_{n}")
        };
        if ext.is_empty() {
            dir.join(name)
        } else {
            dir.join(format!("{name}.{ext}"))
        }
    };
    (0..1000)
        .map(make)
        .find(|p| !p.exists())
        .unwrap_or_else(|| make(0))
}

fn create_behaviour_tree(dir: &Path, _root: &Path) {
    let path = unique_path(dir, "New Behaviour", zestengine_assets::BEHAVIOUR_TREE_EXTENSION);
    let tree = zestengine_assets::BehaviourTree {
        root: zestengine_core::ai::BtNode::Sequence { children: Vec::new() },
        ..Default::default()
    };
    if let Err(e) = zestengine_assets::save_behaviour_tree(&path, &tree) {
        tracing::error!("creating behaviour tree: {e:#}");
    }
}

fn behaviour_tree_editor_ui(ui: &mut egui::Ui, path: &Path) {
    let mut tree = zestengine_assets::load_behaviour_tree(path).unwrap_or_default();
    ui.heading("Behaviour Tree");
    ui.label(egui::RichText::new(path.display().to_string()).small().weak());
    ui.separator();
    let mut changed = false;
    changed |= ui.text_edit_singleline(&mut tree.name).changed();
    changed |= crate::edit_bt_node(ui, &mut tree.root, 0);
    if changed && let Err(e) = zestengine_assets::save_behaviour_tree(path, &tree) {
        tracing::error!("saving behaviour tree: {e:#}");
    }
}

impl Drop for EngineApp {
    fn drop(&mut self) {
        // The XR swapchains live on the renderer's device (fields drop in
        // declaration order, renderer first): end the session before the device.
        // (View > VR off does the same early, through `stop_vr`.)
        self.xr_session = None;
        // Unwinding out of the event loop (a panic) skips `exiting`, so the
        // egui-winit states would drop after the display connection is gone and
        // their Wayland clipboard worker would SIGSEGV in `wl_proxy_destroy`,
        // hiding the real panic. Leak them; the process is dying anyway.
        if std::thread::panicking() {
            std::mem::forget(self.egui_state.take());
            std::mem::forget(self.profiler_egui_state.take());
        }
    }
}

impl ApplicationHandler for EngineApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() || self.pending_init {
            return;
        }
        // Show the splash window FIRST, then defer the heavy `init` to
        // `about_to_wait`. The event loop configures the splash window in
        // between, so it's actually on screen while the (blocking) init runs.
        // Drop an animated `zestengine.webp` / `splash.gif` in the project root to
        // override the embedded static image.
        match crate::splash::Splash::new(event_loop, &self.project_root) {
            Ok(mut s) => {
                let project = zestengine_assets::load_project_file(&self.project_root).ok();
                let name = project.as_ref().map(|p| p.name.clone()).unwrap_or_else(|| {
                    self.project_root.file_name().map_or_else(|| "zestengine project".into(), |n| n.to_string_lossy().into_owned())
                });
                s.set_project(&name, project.and_then(|p| p.settings.accent));
                s.set_status("Starting…");
                self.splash = Some(s);
            }
            Err(e) => tracing::warn!("splash window: {e:#}"),
        }
        self.pending_init = true;
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            // Feed the binding system (2C) so Play-mode mouselook works; the
            // editor fly-cam still uses look_delta gated on `looking`.
            // Flying the Scene camera during play doesn't steer the game.
            if self.playing && !self.looking {
                self.input.mouse_motion(delta.0, delta.1);
            }
            if self.looking {
                self.look_delta.0 += delta.0;
                self.look_delta.1 += delta.1;
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        // Splash is its own borderless window: just repaint it on request and
        // swallow its events. It's closed after the first editor frame.
        if let Some(s) = self.splash.as_mut()
            && window_id == s.window_id() {
                if matches!(event, WindowEvent::RedrawRequested) {
                    s.present();
                }
                return;
            }

        // Profiler is a separate OS window: route its events here and return so
        // the main-window editor logic below never sees them.
        if let Some(pw) = self.profiler_window.clone()
            && window_id == pw.id() {
                if let Some(state) = self.profiler_egui_state.as_mut() {
                    let r = state.on_window_event(&pw, &event);
                    if r.repaint {
                        pw.request_redraw();
                    }
                }
                match event {
                    WindowEvent::CloseRequested => {
                        // Closing the window just turns the profiler off.
                        self.show_stats_overlay = false;
                        self.close_profiler_window();
                    }
                    WindowEvent::Resized(size) => {
                        if let Some(r) = self.renderer.as_mut() {
                            r.resize_profiler_window(size.width, size.height);
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        // Rendered from the main redraw loop; nothing to do here.
                    }
                    _ => {}
                }
                return;
            }

        let egui_wants =
            if let (Some(state), Some(window)) = (self.egui_state.as_mut(), self.window.as_ref()) {
                // While looking, the cursor is locked: keep egui out of it. The
                // same while the game holds the cursor: egui's pointer is still
                // where Play was clicked, so a Fire click (or Space on the
                // focused button) would press Play again and stop the game.
                let game_input = self.game_cursor_locked
                    && matches!(
                        event,
                        WindowEvent::MouseInput { .. }
                            | WindowEvent::MouseWheel { .. }
                            | WindowEvent::CursorMoved { .. }
                            | WindowEvent::KeyboardInput { .. }
                    );
                if self.looking || game_input {
                    false
                } else {
                    let response = state.on_window_event(window, &event);
                    if response.repaint {
                        window.request_redraw();
                    }
                    response.consumed
                }
            } else {
                false
            };
        // Rebind capture (2C): while a binding slot is armed, the next key
        // press is captured into it (Esc cancels). Highest priority so it
        // pre-empts editor shortcuts. Mouse buttons are captured by the
        // listening button itself, gamepad / VR in `poll_rebind_capture`.
        if self.bindings_editor.rebinding().is_some()
            && let WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } = &event
        {
            let result = if *code == KeyCode::Escape {
                zestengine_core::CaptureResult::Cancelled
            } else {
                zestengine_core::CaptureResult::Captured(zestengine_core::InputSource::Key(zestengine_engine::input_engine::map_key(*code)))
            };
            self.apply_rebind(result);
            return;
        }

        match event {
            WindowEvent::CloseRequested => {
                if let Some(fm) = self.file_meta.as_ref().filter(|f| f.dirty) {
                    // Unapplied import settings: ask before they are lost.
                    self.import_prompt = Some(ImportPrompt { path: fm.path.clone(), resume: None, restore_selection: false, quit: true });
                    if let Some(w) = self.window.as_ref() {
                        w.request_redraw();
                    }
                } else if self.scene_dirty {
                    // Hold the close; the Save/Discard/Cancel dialog decides.
                    self.show_quit_dialog = true;
                    if let Some(w) = self.window.as_ref() {
                        w.request_redraw();
                    }
                } else {
                    self.save_project();
                    event_loop.exit();
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } if self.game_cursor_locked => self.set_game_cursor(false),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } if !egui_wants
                && !self.egui_ctx.wants_keyboard_input()
                && !self.code_tab_focused() =>
            {
                // ...but not while a code editor / text field has focus (there
                // Escape is the vim Insert -> Normal key, not a deselect). The
                // first Escape only ends river path editing.
                if !crate::river_edit::exit(&self.egui_ctx) && !crate::cloth_paint::exit(&self.egui_ctx) && !crate::terrain_paint::exit(&self.egui_ctx) && !crate::tile_edit::exit(&self.egui_ctx) && !crate::shape_edit::exit(&self.egui_ctx) {
                    self.selection = Selection::None;
                    self.file_material = None;
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state,
                        ..
                    },
                ..
            } => match state {
                ElementState::Released => {
                    self.keys.remove(&code);
                    self.input.key(code, false);
                }
                // A game UI text field has the keyboard (it reads egui text events).
                ElementState::Pressed if self.playing && self.game_ui_frame.text_editing => {}
                ElementState::Pressed if !egui_wants => {
                    self.keys.insert(code);
                    if !self.looking {
                        self.input.key(code, true);
                    }
                    // The window's modifier state, not our held-key set: a Ctrl
                    // released while the window was unfocused never sends its
                    // release, which left Ctrl "held" (plain D duplicated,
                    // W / E / R / T stopped switching tools).
                    let ctrl = self.modifiers.control_key();
                    if !self.looking {
                        match code {
                            // Unity-style: W move, E scale, R rotate (these
                            // only fire when not mouse-looking, so they don't
                            // clash with WASDQE fly).
                            KeyCode::KeyW if !ctrl => self.gizmo.tool = GizmoTool::Move,
                            KeyCode::KeyE if !ctrl => self.gizmo.tool = GizmoTool::Rotate,
                            KeyCode::KeyR if !ctrl && !self.placement.session_captures_input() => self.gizmo.tool = GizmoTool::Scale,
                            KeyCode::KeyQ if !ctrl => self.gizmo.tool = GizmoTool::Select,
                            KeyCode::KeyT if !ctrl => self.gizmo.tool = GizmoTool::Rect,
                            KeyCode::KeyK if ctrl => self.command_palette.open(),
                            KeyCode::KeyH => self.scene_view_key(),
                            // Scene view grid (editing only: G stays free for the game in Play).
                            KeyCode::KeyG if !ctrl && !self.playing => {
                                self.project.settings.scene_grid = !self.project.settings.scene_grid;
                            }
                            KeyCode::KeyS if ctrl => {
                                self.actions.push(EditorAction::SaveScene(None));
                            }
                            KeyCode::KeyF => {
                                self.actions.push(EditorAction::FocusSelected);
                            }
                            KeyCode::KeyZ if ctrl => {
                                let shift = self.modifiers.shift_key();
                                self.actions.push(if shift {
                                    EditorAction::Redo
                                } else {
                                    EditorAction::Undo
                                });
                            }
                            KeyCode::KeyY if ctrl => {
                                self.actions.push(EditorAction::Redo);
                            }
                            KeyCode::KeyW if ctrl => {
                                self.close_focused_code_tab();
                            }
                            KeyCode::KeyD if ctrl => {
                                // Duplicate the selected object (scene) or file.
                                match &self.selection {
                                    Selection::Object(i) => self
                                        .actions
                                        .push(EditorAction::DuplicateObject(*i)),
                                    Selection::File(p) => self
                                        .actions
                                        .push(EditorAction::DuplicateFile(p.clone())),
                                    Selection::None => {}
                                }
                            }
                            // Removes the selected river point (in the Scene view).
                            KeyCode::Delete if crate::river_edit::building(&self.egui_ctx) || crate::shape_edit::active(&self.egui_ctx) => {}
                            KeyCode::Delete => {
                                // Acts on the current selection: an object in the
                                // scene, or a file/folder in the browser (same as
                                // their right-click Delete).
                                match &self.selection {
                                    Selection::Object(i) => self
                                        .actions
                                        .push(EditorAction::DeleteObject(*i)),
                                    Selection::File(p) => self
                                        .actions
                                        .push(EditorAction::DeleteFile(p.clone())),
                                    Selection::None => {}
                                }
                            }
                            _ => {}
                        }
                    }
                }
                ElementState::Pressed => {
                    // egui consumed the press (e.g. a focused text field), but
                    // still track modifier keys so `self.keys` doesn't desync,
                    // otherwise a Ctrl press swallowed here leaves `ctrl` false
                    // for a later viewport shortcut (Ctrl+Z undo, etc.).
                    if matches!(
                        code,
                        KeyCode::ControlLeft
                            | KeyCode::ControlRight
                            | KeyCode::ShiftLeft
                            | KeyCode::ShiftRight
                            | KeyCode::AltLeft
                            | KeyCode::AltRight
                    ) {
                        self.keys.insert(code);
                    }
                }
            },
            WindowEvent::Resized(_) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize();
                }
            }
            WindowEvent::Occluded(o) => self.pacing.set_occluded(o),
            // Moved to another monitor / rescaled: re-read the refresh rate.
            WindowEvent::Moved(_) | WindowEvent::ScaleFactorChanged { .. } => {
                if let Some(w) = &self.window {
                    self.pacing.refresh_monitor(w);
                }
            }
            // Regained focus → reimport assets edited in other apps (e.g. an FBX
            // re-exported from Blender) without a manual reimport.
            WindowEvent::Focused(true) => self.reload_changed_assets(),
            // Keys released while another window has focus never send their
            // release: forget everything held so nothing stays stuck down.
            WindowEvent::Focused(false) => {
                zestengine_render::hitch::event("focus_lost", "", 0);
                for code in std::mem::take(&mut self.keys) {
                    self.input.key(code, false);
                }
                self.modifiers = winit::keyboard::ModifiersState::empty();
            }
            WindowEvent::ModifiersChanged(m) => self.modifiers = m.state(),
            WindowEvent::MouseInput { button, state, .. } => {
                let pressed = state == ElementState::Pressed;
                // A press that lands on an egui popup / menu / window belongs
                // to egui. Releases always pass so nothing stays held.
                let on_overlay = pressed && self.pointer_over_overlay();
                if self.playing && !on_overlay {
                    self.input.mouse_button(button, pressed);
                    if pressed
                        && self.game_cursor_wanted
                        && !self.game_cursor_locked
                        && button == MouseButton::Left
                        && self.cursor_in_game()
                        && !self.looking
                        && !self.egui_ctx.is_using_pointer()
                    {
                        self.set_game_cursor(true);
                    }
                }
                match button {
                    MouseButton::Right
                        // Look starts from the viewport tab widget (see
                        // viewport_ui); only the release is handled here so
                        // it can't get stuck while the cursor is locked.
                        if !pressed => {
                            self.set_looking(false);
                        }
                    MouseButton::Middle => {
                        self.panning = pressed
                            && !on_overlay
                            && self.cursor_in_viewport()
                            && !self.egui_ctx.is_using_pointer();
                    }
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some((lx, ly)) = self.last_cursor
                    && self.panning
                {
                    self.camera
                        .pan((position.x - lx) as f32, (position.y - ly) as f32);
                }
                self.last_cursor = Some((position.x, position.y));
            }
            // While looking the cursor is locked and egui is bypassed, so
            // handle scroll here; otherwise the viewport tab handles it.
            WindowEvent::MouseWheel { delta, .. } if self.looking => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 60.0,
                };
                self.camera.dolly(scroll);
            }
            WindowEvent::RedrawRequested => self.redraw(event_loop),
            _ => {}
        }
    }

    /// The event loop is leaving but the display connection is still open: drop
    /// the egui-winit states now. Their clipboard (smithay-clipboard on Wayland)
    /// joins a worker thread that destroys Wayland objects, and doing that
    /// after the connection closes crashes with SIGSEGV in `wl_proxy_destroy`.
    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // Close every audio stream (output, voice mic/out, Steam Audio) now, in a
        // defined order while the system is quiet, rather than whenever the app
        // struct drops late in teardown (after the GPU/display are gone). A
        // stream still live while the process dies is what leaves the PipeWire
        // client node dangling.
        if let Some(a) = self.audio.as_mut() {
            a.stop_all();
        }
        self.voice = None;
        if let Some(session) = self.play_session.take() {
            // `end` stops the session's voices and hands the engine back.
            drop(session.end());
        }
        self.audio = None;
        self.egui_state = None;
        self.profiler_egui_state = None;
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Startup: drive the splash animation and the deferred/threaded load.
        // Runs until the main window exists (i.e. finish_init has completed).
        if self.window.is_none() {
            // Splash: show the current load phase + animate every tick.
            if let Some(s) = self.splash.as_mut() {
                let status = self.load_status.lock().unwrap().clone();
                let painted = if status.is_empty() {
                    s.present()
                } else {
                    s.set_status(&status)
                };
                self.splash_painted = painted || self.splash_painted;
            }
            // Kick off the load once the splash has shown a frame (Wayland needs
            // the configure roundtrip first); proceed immediately if no splash.
            if self.pending_init && (self.splash_painted || self.splash.is_none()) {
                self.pending_init = false;
                if let Err(e) = self.start_load(event_loop) {
                    tracing::error!("starting load failed: {e:#}");
                    event_loop.exit();
                }
            }
            if let Some(rx) = self.renderer_rx.as_ref() {
                match rx.try_recv() {
                    Ok(Ok(renderer)) => {
                        self.renderer_rx = None;
                        tracing::info!(elapsed = ?self.start.elapsed(), "renderer ready");
                        self.ready_renderer = Some((renderer, None));
                    }
                    Ok(Err(e)) => {
                        tracing::error!("renderer build failed: {e:#}");
                        event_loop.exit();
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => {}
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        tracing::error!("renderer build thread died");
                        event_loop.exit();
                    }
                }
            }
            // Renderer + plugin build ready → finish_init (project, then the
            // scene parse; the editor opens only once the scene is applied).
            if self.ready_renderer.is_some() {
                let plugins_built = match self.plugin_rx.as_ref().map(|rx| rx.try_recv()) {
                    None => Some(None),
                    Some(Ok(built)) => Some(Some(built)),
                    Some(Err(std::sync::mpsc::TryRecvError::Empty)) => {
                        if let Some(s) = self.splash.as_mut() {
                            s.set_progress(2, LOAD_STEPS);
                        }
                        let line = self.plugin_status.lock().unwrap().clone();
                        *self.load_status.lock().unwrap() = if line.is_empty() { "Building component plugins…".into() } else { line };
                        None
                    }
                    Some(Err(std::sync::mpsc::TryRecvError::Disconnected)) => {
                        Some(Some(Err(anyhow::anyhow!("plugin build thread died"))))
                    }
                };
                if let Some(plugins_built) = plugins_built {
                    self.plugin_rx = None;
                    let (renderer, xr) = self.ready_renderer.take().unwrap();
                    let window = self.pending_window.take().expect("pending window");
                    if let Err(e) = self.finish_init(window, renderer, xr, plugins_built) {
                        tracing::error!("initialization failed: {e:#}");
                        event_loop.exit();
                    }
                }
            }
            // Scene parse finished → upload on the main thread + open the editor.
            let results = self.tasks.poll_workers();
            for r in results {
                self.apply_task_result(r);
            }
            return;
        }
        // Frame limiter: while the next frame is not due, sleep in the event
        // loop (`WaitUntil`) instead of in `redraw`, so input / OS events are
        // handled the moment they arrive. Wake ~1 ms early; `redraw` spins the rest.
        self.frame_limit_prepare();
        if let Some(d) = self.pacing.sleep_before_redraw() {
            zestengine_render::hitch::limiter_idle_begin();
            event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + d));
            return;
        }
        event_loop.set_control_flow(ControlFlow::Poll);
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

/// Reference size of the 2D canvas `i` belongs to (itself or an ancestor).
fn screen_canvas_of(scene: &LoadedScene, mut i: usize) -> Option<[f32; 2]> {
    for _ in 0..64 {
        let o = scene.objects.get(i)?;
        if let Some(c) = o.components.iter().find_map(|c| c.as_any().downcast_ref::<zestengine_core::UiCanvas>()) {
            return (c.render_mode == zestengine_core::UiRenderMode::Screen).then_some(c.reference_size);
        }
        i = o.parent?;
    }
    None
}

/// Type family of an object for icons and colours (editor spec §2.4).
/// Which objects are the root of an imported model: an empty grouping object, named after the
/// model file, whose children are that model's meshes (what `add_asset_scene` creates). They
/// show the blue model icon instead of the grey group one.
fn model_root_flags(objects: &[scene::SceneObject]) -> Vec<bool> {
    let mut flags = vec![false; objects.len()];
    for o in objects {
        let (Some(parent), zestengine_assets::ObjectSource::Model { path, .. }) = (o.parent, &o.source) else { continue };
        let Some(root) = objects.get(parent) else { continue };
        if flags[parent] || !matches!(root.source, zestengine_assets::ObjectSource::Empty) {
            continue;
        }
        let stem = std::path::Path::new(path).file_stem().map(|s| s.to_string_lossy().into_owned());
        if stem.as_deref() == Some(root.name.as_str()) {
            flags[parent] = true;
        }
    }
    flags
}

fn object_kind(o: &scene::SceneObject) -> crate::theme::ObjectKind {
    use crate::theme::ObjectKind as K;
    let has = |f: &dyn Fn(&dyn std::any::Any) -> bool| o.components.iter().any(|c| f(c.as_any()));
    if let Some(l) = o.components.iter().find_map(|c| c.as_any().downcast_ref::<zestengine_core::LightComponent>()) {
        return match l.kind {
            zestengine_core::LightKind::Directional => K::DirectionalLight,
            zestengine_core::LightKind::Spot => K::SpotLight,
            zestengine_core::LightKind::Point => K::PointLight,
        };
    }
    if matches!(o.source, zestengine_assets::ObjectSource::Camera) || has(&|c| c.is::<zestengine_core::CameraComponent>()) {
        return K::Camera;
    }
    if has(&|c| c.is::<zestengine_core::UiCanvas>()) {
        return K::UiCanvas;
    }
    if has(&|c| c.is::<zestengine_core::UiToggle>()) {
        return K::UiToggle;
    }
    if has(&|c| c.is::<zestengine_core::UiText>()) {
        return K::UiText;
    }
    if o.components.iter().any(|c| c.type_name().starts_with("UI ")) {
        return K::UiOther;
    }
    if has(&|c| c.is::<zestengine_core::ReflectionProbe>() || c.is::<zestengine_core::PlanarReflection>()) {
        return K::ReflectionProbe;
    }
    if has(&|c| c.is::<zestengine_core::FluxVolume>() || c.is::<zestengine_core::LightProbeVolume>()) {
        return K::FluxVolume;
    }
    if has(&|c| c.is::<zestengine_core::AudioSource>() || c.is::<zestengine_core::AudioVolume>()) {
        return K::Audio;
    }
    if has(&|c| c.is::<zestengine_core::VolumeComponent>()) {
        return K::PostFxVolume;
    }
    if has(&|c| c.is::<zestengine_core::Water>() || c.is::<zestengine_core::Waterfall>() || c.is::<zestengine_core::Whirlpool>()) {
        return K::Water;
    }
    match o.source {
        zestengine_assets::ObjectSource::Model { .. } if o.render.is_none() => K::Model,
        zestengine_assets::ObjectSource::Model { .. } | zestengine_assets::ObjectSource::Primitive { .. } => K::Mesh,
        _ if o.render.is_some() => K::Mesh,
        _ => K::Group,
    }
}

/// Narrowest the inspector dock may get before rows start clipping.
const INSPECTOR_MIN_WIDTH: f32 = 352.0;

/// egui_dock has no per-leaf minimum: after each frame, widen the leaf holding
/// `tab` back to `min` by moving its nearest horizontal split (dragging the
/// divider past it just stops there).
/// Render the bindings editor plus the "Player remaps" section, shared by the
/// standalone Input Bindings window and the dockable Project Settings tab.
/// Returns true when the project's bindings changed (the caller pushes them to
/// the live input manager and persists the project).
fn bindings_section_ui(
    ui: &mut egui::Ui,
    id: &str,
    project: &mut zestengine_assets::ProjectFile,
    bindings_editor: &mut crate::BindingsEditor,
    input: &mut zestengine_engine::input_engine::InputManager,
) -> bool {
    let mut changed = false;
    if crate::bindings_editor_ui(ui, id, &mut project.bindings, bindings_editor) {
        changed = true;
    }
    ui.add_space(6.0);
    crate::widgets::eyebrow(ui, "Player remaps");
    let overrides = input.controls().overrides();
    let note = if overrides.is_empty() {
        "No Play-mode remaps. Games save their players' remaps (ctx.rebind, the Controls menu) to input.ron in the user's config folder, on top of these defaults.".to_owned()
    } else {
        format!(
            "Play mode has {} remapped action(s) on top of these defaults (saved in .zestengine/input_overrides.ron). Shipped games keep their players' remaps in input.ron in the user's config folder.",
            overrides.actions.len()
        )
    };
    crate::widgets::info_note(ui, &note);
    if !overrides.is_empty() && ui.button("Clear Play-mode remaps").clicked() {
        input.apply(zestengine_core::InputCommand::ResetAll);
    }
    changed
}

fn tab_label(tab: &Tab) -> String {
    match tab {
        Tab::Viewport => format!("{}  Scene", ph::CUBE),
        Tab::Camera => format!("{}  Camera", ph::VIDEO_CAMERA),
        Tab::UiEditor => format!("{}  UI Editor", ph::LAYOUT),
        Tab::AnimController => format!("{}  Animator Controller", ph::FLOW_ARROW),
        Tab::AnimClip => format!("{}  Animation", ph::FILM_STRIP),
        Tab::Sequencer => format!("{}  Sequencer", ph::FILM_STRIP),
        Tab::ShaderGraph => format!("{}  Shader Graph", ph::FLOW_ARROW),
        Tab::Scene => "Scene".to_owned(),
        Tab::Inspector => "Inspector".to_owned(),
        Tab::Environment => "Environment".to_owned(),
        Tab::Baker => "FluxBaker".to_owned(),
        Tab::Occlusion => "Occlusion".to_owned(),
        Tab::ProjectSettings => "Project Settings".to_owned(),
        Tab::Files => "Files".to_owned(),
        Tab::Log => {
            let warns = log_capture::store().lock().unwrap().warns;
            if warns > 0 { format!("Log  {warns}") } else { "Log".to_owned() }
        }
        Tab::Code(path) => path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "Untitled".to_owned()),
        Tab::Nvim => "Neovim".to_owned(),
    }
}

/// Labels for leaves whose tabs overflow their bar (last frame's width): the
/// widest labels are cut with an ellipsis to an equal share, so every tab
/// stays visible instead of egui_dock scrolling them under the edge.
/// `vp_strip` is the toolbar width over the Scene view's strip, reserved in the
/// leaf showing that tab.
fn fit_tab_labels(ctx: &egui::Context, dock: &DockState<Tab>, vp_strip: f32) -> HashMap<Tab, String> {
    // egui_dock's tab padding (2 x 8 px) and close button (24 px).
    const PAD: f32 = 16.0;
    const CLOSE: f32 = 24.0;
    let font = egui::TextStyle::Button.resolve(&ctx.style());
    let painter = ctx.layer_painter(egui::LayerId::background());
    let measure = |s: &str| painter.layout_no_wrap(s.to_owned(), font.clone(), egui::Color32::WHITE).size().x;
    let mut out = HashMap::new();
    for (_, node) in dock.iter_all_nodes() {
        let egui_dock::Node::Leaf(leaf) = node else { continue };
        let strip = leaf.tabs.get(leaf.active.0).is_some_and(|t| *t == Tab::Viewport);
        if leaf.rect.width() < 1.0 || leaf.tabs.len() < 2 {
            continue;
        }
        let avail = (leaf.rect.width() - if strip { vp_strip } else { 0.0 }).max(1.0);
        let labels: Vec<String> = leaf.tabs.iter().map(tab_label).collect();
        let widths: Vec<f32> = labels.iter().map(|l| measure(l)).collect();
        // Every tab has a close X (closed panels reopen from the Windows menu).
        let chrome = (PAD + CLOSE) * leaf.tabs.len() as f32;
        if widths.iter().sum::<f32>() + chrome <= avail {
            continue;
        }
        // Water-fill: labels narrower than an equal share keep their width and
        // hand the rest to the wider ones, which share the remaining cap.
        let mut budget = (avail - chrome - 1.0).max(0.0);
        let mut sorted = widths.clone();
        sorted.sort_by(f32::total_cmp);
        let mut left = sorted.len();
        let mut cap = budget / left as f32;
        for w in sorted {
            if w > cap {
                break;
            }
            budget -= w;
            left -= 1;
            cap = if left > 0 { budget / left as f32 } else { f32::INFINITY };
        }
        let ellipsis = measure("\u{2026}");
        for ((tab, label), w) in leaf.tabs.iter().zip(&labels).zip(&widths) {
            if *w <= cap {
                continue;
            }
            let mut cut = String::new();
            for ch in label.chars() {
                cut.push(ch);
                // Always keep the first character (the icon, where there is one).
                if measure(&cut) + ellipsis > cap && cut.chars().count() > 1 {
                    cut.pop();
                    break;
                }
            }
            // Too narrow even for one character and the ellipsis: just the icon.
            let short = if cut.chars().count() == 1 && measure(&cut) + ellipsis > cap {
                cut
            } else {
                format!("{}\u{2026}", cut.trim_end())
            };
            out.insert(tab.clone(), short);
        }
    }
    out
}

fn clamp_leaf_width(dock: &mut DockState<Tab>, tab: &Tab, min: f32) {
    let Some((surface, node, _)) = dock.find_tab(tab) else {
        return;
    };
    let tree = &mut dock[surface];
    let mut child = node;
    while let Some(parent) = child.parent() {
        if let egui_dock::Node::Horizontal(split) = &mut tree[parent] {
            let w = split.rect.width();
            if w > min + 120.0 {
                let right = child == parent.right();
                let leaf_w = if right { w * (1.0 - split.fraction) } else { w * split.fraction };
                if leaf_w < min {
                    split.fraction = if right { 1.0 - min / w } else { min / w };
                }
            }
            return;
        }
        child = parent;
    }
}

/// Command palette entries (Ctrl K).
#[derive(Clone, Copy)]
enum PaletteCmd {
    TogglePlay,
    SaveScene,
    NewScene,
    ProjectHub,
    NewProject,
    OpenProject,
    BuildGame,
    BuildGameWindows,
    Undo,
    Redo,
    Duplicate,
    Focus,
    ResetLayout,
    ToggleGameUi,
    ToggleGizmos,
    ToggleGrid,
    ToggleVr,
    HideSelection,
    ShowAllHidden,
    UnlockAll,
    OpenTab(usize),
    ProjectSettings,
    Layers,
    InputBindings,
    AudioMixer,
    BakeLighting,
    BakeAudio,
    ReloadPlugins,
    Spawn(zestengine_assets::PrimitiveShape),
    SpawnLight(zestengine_core::LightKind),
    SpawnAreaLight,
    UiPreset(crate::UiPreset),
    UiTemplate(zestengine_engine::ui_templates::UiTemplate),
}

const PALETTE_TABS: [(Tab, &str); 14] = [
    (Tab::Viewport, "Scene"),
    (Tab::Camera, "Camera"),
    (Tab::UiEditor, "UI Editor"),
    (Tab::AnimController, "Animator Controller"),
    (Tab::AnimClip, "Animation"),
    (Tab::Sequencer, "Sequencer"),
    (Tab::ShaderGraph, "Shader Graph"),
    (Tab::Scene, "Scene"),
    (Tab::Inspector, "Inspector"),
    (Tab::Environment, "Environment"),
    (Tab::Baker, "FluxBaker"),
    (Tab::Occlusion, "Occlusion"),
    (Tab::Files, "Files"),
    (Tab::Log, "Log"),
];

/// Per-kind billboard visibility + size, orientation cross, game UI overlay.
fn billboard_menu(ui: &mut egui::Ui, filter: &mut WidgetFilter, settings: &mut zestengine_assets::ProjectSettings) {
    ui.checkbox(&mut filter.enabled, "Show gizmos and billboards")
        .on_hover_text("Same as clicking the eye: off hides everything below (the move / rotate / scale handles always show)");
    ui.separator();
    ui.add_enabled_ui(filter.enabled, |ui| {
        ui.label(egui::RichText::new("Gizmos").size(crate::theme::SIZE_META).color(crate::theme::TEXT_MUTED));
        ui.checkbox(&mut filter.components, "Component gizmos")
            .on_hover_text("Collider shapes, probe and post volumes, water outlines, joints and other component handles");
        ui.checkbox(&mut filter.cross, "Orientation cross").on_hover_text(
            "The grey axis cross at the selected object's pivot (Move/Scale tools). Independent of the transform handles.",
        );
        ui.add_space(4.0);
        billboard_settings(ui, filter);
    });
    ui.separator();
    ui.checkbox(&mut filter.game_ui, "Game UI overlay")
        .on_hover_text("Preview the scene's UI canvases over the viewport (always shown in Play)");
    ui.separator();
    // The ground grid is not a gizmo: the master toggle above leaves it alone.
    ui.horizontal(|ui| {
        ui.checkbox(&mut settings.scene_grid, "Grid").on_hover_text(
            "Ground grid at y = 0 (the XY / YZ plane in 2D side views), spaced by the Snap step with a major line every 10. Shortcut: G",
        );
        ui.add_enabled(settings.scene_grid, egui::Slider::new(&mut settings.scene_grid_opacity, 0.05..=1.0).show_value(false))
            .on_hover_text("Grid opacity");
    });
}

/// The billboard kinds with their show toggles and sizes.
fn billboard_settings(ui: &mut egui::Ui, filter: &mut WidgetFilter) {
    ui.label(egui::RichText::new("Billboards").size(crate::theme::SIZE_META).color(crate::theme::TEXT_MUTED));
    for (label, setting) in [
        ("Lights", &mut filter.lights),
        ("Cameras", &mut filter.cameras),
        ("Probe Volumes", &mut filter.probes),
        ("Reflection Probes", &mut filter.reflection_probes),
        ("Audio Sources", &mut filter.audio),
        ("UI Canvases", &mut filter.ui_canvases),
        ("Post FX Volumes", &mut filter.post_volumes),
        ("Placeables", &mut filter.placeables),
    ] {
        ui.horizontal(|ui| {
            ui.checkbox(&mut setting.visible, label);
            ui.add_enabled(setting.visible, egui::Slider::new(&mut setting.size, 0.3..=4.0).show_value(false))
                .on_hover_text("Widget size");
        });
    }
}

/// Thin vertical divider between segments in a trough.
fn trough_divider(ui: &mut egui::Ui) {
    let (r, _) = ui.allocate_exact_size(egui::vec2(9.0, crate::theme::H_BUTTON_SMALL), egui::Sense::hover());
    ui.painter().vline(r.center().x, r.shrink2(egui::vec2(0.0, 5.0)).y_range(), egui::Stroke::new(1.0_f32, crate::theme::LINE_FIELD));
}

/// A top-level menu-bar menu, recorded for [`switch_hovered_menu`].
fn top_menu<R>(
    bar: &mut Vec<(egui::Id, egui::Rect)>,
    ui: &mut egui::Ui,
    label: &str,
    add_contents: impl FnOnce(&mut egui::Ui) -> R,
) {
    let r = ui.menu_button(label, |ui| {
        crate::widgets::menu_frame(ui);
        add_contents(ui)
    });
    bar.push((egui::Popup::default_response_id(&r.response), r.response.rect));
}

/// While one top-level menu is open, hovering another opens that one instead.
fn switch_hovered_menu(ctx: &egui::Context, bar: &[(egui::Id, egui::Rect)]) {
    let Some(open) = bar.iter().position(|(id, _)| egui::Popup::is_id_open(ctx, *id)) else {
        return;
    };
    let Some(pos) = ctx.pointer_hover_pos() else {
        return;
    };
    if let Some(i) = bar.iter().position(|(_, r)| r.contains(pos))
        && i != open
    {
        egui::Popup::open_id(ctx, bar[i].0);
        ctx.request_repaint();
    }
}

/// Fit each Water whose inspector asked for it to the basin around it
/// (outline, size and depth).
fn fit_water_requests(scene: &mut LoadedScene, dirty: &mut bool) {
    let asked: Vec<usize> = (0..scene.objects.len())
        .filter(|&i| scene.objects[i].components.iter().any(|c| c.as_any().downcast_ref::<zestengine_core::Water>().is_some_and(|w| w.fit_request)))
        .collect();
    if asked.is_empty() {
        return;
    }
    let world = scene.world_transforms();
    for wi in asked {
        let fit = zestengine_engine::water_walls::fit_basin(scene, wi, &world);
        let name = scene.objects[wi].name.clone();
        let Some(water) = scene.objects[wi].components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<zestengine_core::Water>()) else {
            continue;
        };
        water.fit_request = false;
        match fit {
            Some((outline, depth)) => {
                // The Size box covers the outline around the origin.
                let reach = outline.iter().fold(glam::Vec2::ZERO, |m, p| m.max(glam::Vec2::from(*p).abs()));
                water.size = (reach * 2.0).to_array();
                water.shape = outline;
                water.smooth_shape = false;
                if let Some(d) = depth {
                    water.depth = d;
                }
                if !water.body.basin() {
                    water.body = zestengine_core::WaterBody::Lake;
                }
                tracing::info!("{name}: fitted to its basin ({} points)", water.shape.len());
            }
            None => tracing::warn!("{name}: no walls or shore around it to fit to; place it inside the basin at the water level"),
        }
        *dirty = true;
    }
}


/// The role of the nearest ancestor of object `i` whose own role is not `Both`
/// (what the hierarchy badge shows dimmed on its descendants).
fn inherited_net_role(objects: &[zestengine_engine::scene::SceneObject], i: usize) -> zestengine_core::NetRole {
    let (mut cur, mut guard) = (objects[i].parent, 0);
    while let Some(p) = cur.filter(|&p| p < objects.len() && guard < 128) {
        if !objects[p].net_role.is_both() {
            return objects[p].net_role;
        }
        cur = objects[p].parent;
        guard += 1;
    }
    zestengine_core::NetRole::Both
}

#[cfg(test)]
mod collider_fit_tests {
    use super::*;

    #[test]
    fn colliders_fit_the_bounds_and_planes_get_a_slab() {
        let mut b = zestengine_core::BoxCollider::default();
        fit_collider(&mut b, Vec3::new(-1.0, 0.0, -2.0), Vec3::new(3.0, 1.0, 2.0));
        assert_eq!((b.center, b.size), ([1.0, 0.5, 0.0], [4.0, 1.0, 4.0]));
        let mut plane = zestengine_core::BoxCollider::default();
        fit_collider(&mut plane, Vec3::new(-5.0, 0.0, -5.0), Vec3::new(5.0, 0.0, 5.0));
        assert!(plane.size[1] > 0.0 && plane.center[1] + plane.size[1] * 0.5 <= 1e-6, "slab under the surface: {:?} {:?}", plane.center, plane.size);
        let mut c = zestengine_core::CapsuleCollider::default();
        fit_collider(&mut c, Vec3::new(-0.3, 0.0, -0.3), Vec3::new(0.3, 2.0, 0.3));
        assert!(c.direction == zestengine_core::ColliderAxis::Y && c.height == 2.0 && (c.radius - 0.3).abs() < 1e-6);
    }
}
