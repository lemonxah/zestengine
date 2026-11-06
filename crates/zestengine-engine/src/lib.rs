//! zestengine-engine: application loop, scene management, editor shell.
//!
//! The editor is a dockable layout (egui_dock) around a transparent
//! Viewport tab: Scene list, unified Inspector, project Files browser,
//! menu bar, transform gizmos, picking, and drag & drop assets.

// Client-driven code (the frame loop, GPU passes) has no caller in a server build; its
// helpers stay compiled so the shared scene code is the same code.
#![cfg_attr(not(feature = "client"), allow(dead_code))]

// Without the `client` feature the render-free stand-in answers to the same crate
// name, so scene code keeps its `zestengine_render::` paths: plain data types are
// the real ones, and `Renderer` is inert (never constructed; APIs take `Option`).
#[cfg(not(feature = "client"))]
extern crate zestengine_render_lite as zestengine_render;

// Always compiled (runtime + bundling path; no egui / editor).
pub mod anim_graph;
pub mod anim_props;
pub mod anim_muscle;
pub mod anim_mirror;
pub mod anim_cull;
pub mod clip_motion;
pub mod clip_source;
pub mod avatar_runtime;
#[cfg(feature = "client")]
pub mod avatar_menu;
pub mod avatar_net;
#[cfg(feature = "client")]
pub mod avatar_osc_host;
#[cfg(feature = "client")]
pub mod avatar_play;
#[cfg(feature = "client")]
pub mod avatar_safety;
pub mod bench_avatars;
pub mod audio;
pub mod audio_environment;
pub mod cloth;
pub mod face_drive;
pub mod skin_cpu;
#[cfg(feature = "client")]
pub mod face_input;
#[cfg(feature = "client")]
pub mod slimevr;
#[cfg(feature = "client")]
pub mod viseme;
#[cfg(feature = "client")]
pub mod voice_codec;
pub mod bundle;
pub mod local_server;
pub mod server_browser;
pub mod server_build;
pub mod server_deploy;
pub mod net_roles;
pub mod render_util;
pub mod humanoid;
pub mod resource_tracker;
#[cfg(feature = "client")]
pub mod body;
pub mod audio_analysis;
pub mod audio_tap;
pub mod audio_mixer;
pub mod navmesh;
pub mod particles;
pub mod water_render;
mod water_interaction;
mod water_assets;
pub mod water_fx;
pub mod water_carry;
pub mod water_walls;
pub mod underwater;
pub mod prefab;
pub mod placement;
pub mod transform_edit;
pub mod log_capture;
pub mod physics;
pub mod physics2d;
pub mod tile_db;
pub mod tile_collider;
pub mod authored_collision;
pub use zestengine_plugins as plugins;
pub use plugins::RuntimePlugins;
#[cfg(feature = "client")]
pub mod game_ui;
#[cfg(feature = "client")]
pub mod text_setup;
pub mod text3d;
pub mod play;
#[cfg(feature = "client")]
pub mod realtime_gi;
pub mod refl_slots;
#[cfg(feature = "client")]
pub mod runtime;
pub mod scene;
pub mod scene_loader;
// User-generated content packages (.zestenginepak) loaded at runtime.
pub mod content;
pub mod occlusion;
// Public so the headless `gi_preview` example can drive the real probe march.
pub mod sw_gi;
#[cfg(feature = "client")]
pub mod input_engine;
// Public so tools (dedicated servers, bots, test harnesses) can drive sessions directly.
pub mod net;
// The GPU-free game host a dedicated server builds on (works with or without `client`).
pub mod server;
pub mod net_identity;
pub mod net_validate;
pub mod net_predict;
pub mod net_relay;
pub mod net_secure;
pub mod net_sync;
pub mod edit_net;
pub use edit_net::{EditConfig, EditEvent, EditOp, EditSession, NetHandJoints, PeerPresence};
#[cfg(feature = "client")]
pub mod voice;
pub mod shaders;

// Editor-only modules (egui, gizmos, LSP, undo, free-fly camera, window icon).
pub mod water_probe;

#[cfg(feature = "client")]
pub use runtime::{GameConfig, XrMode, run_game};
#[cfg(all(feature = "client", target_os = "android"))]
pub use runtime::{android_assets_root, run_game_android};
#[cfg(all(feature = "client", target_os = "android"))]
pub use winit::platform::android::activity::AndroidApp;
pub use bundle::{BuildTarget, build_project_dir, scaffold_project};
pub use server_build::build_server_dir;
pub mod templates;
pub use templates::{ProjectTemplate, TemplateCategory};


/// Install the tracing subscriber (stdout, plus the editor's Log tab capture).
pub fn init_logging() {
    log_capture::init();
}

pub mod ui_templates;

mod avatar_hands;
