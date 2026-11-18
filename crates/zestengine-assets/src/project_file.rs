//! `project.zestengine` — per-project state (RON): project name, the scene to
//! reopen on startup, and engine settings specific to this project.
//! Created on first run; extended with `#[serde(default)]` fields as the
//! engine grows so old files keep loading.

use std::path::Path;

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};

pub const PROJECT_FILE_NAME: &str = "project.zestengine";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ProjectFile {
    pub mods: zestengine_core::mods::ModSettings,
    pub name: String,
    /// Project-relative path of the last opened scene; reloaded on startup
    /// so the editor reopens the scene you were last in.
    pub last_scene: Option<String>,
    /// Project-relative path of the scene a built game loads first. Distinct
    /// from `last_scene`: the editor restores `last_scene`, while a shipped game
    /// boots `boot_scene`. Set in Project Settings.
    #[serde(default)]
    pub boot_scene: Option<String>,
    pub settings: ProjectSettings,
    /// Input control schemes (2C): rebindable in the editor and at runtime,
    /// shared by the editor Play mode and a shipped game.
    #[serde(default)]
    pub bindings: zestengine_core::Bindings,
    /// Audio buses (volumes / mutes), applied in the editor and a built game.
    #[serde(default)]
    pub audio_mixer: zestengine_core::AudioMixer,
    /// Legacy inline audio settings, migrated to audio_settings by the editor.
    #[serde(default, skip_serializing_if = "default_spatial_audio")]
    pub spatial_audio: zestengine_core::SpatialAudioSettings,
    /// `.audiosettings` asset; empty preserves legacy inline spatial_audio.
    pub audio_settings: String,
    /// OSC face-tracking receiver (off unless enabled).
    #[serde(default)]
    pub tracking_input: zestengine_core::TrackingInputSettings,
    /// Gravity, contact stiffness and the default physics material.
    #[serde(default)]
    pub physics: zestengine_core::PhysicsSettings,
    /// A built game starts in VR when a headset and OpenXR runtime are
    /// present. Off: it always opens a flat window (a Meta Quest build
    /// starts in VR regardless).
    #[serde(default)]
    pub vr: bool,
    #[serde(default = "one", skip_serializing_if = "is_one")]
    pub vr_render_scale: f32,
    /// Project-wide network role per component type name (NET-3), winning over the type's own
    /// default. Applied to the registry by [`ProjectFile::apply_role_overrides`]; an explicit
    /// per-component role in a scene still wins over this.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub net_role_overrides: std::collections::BTreeMap<String, zestengine_core::NetRole>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layers: Option<zestengine_core::LayerSettings>,
}

fn one() -> f32 {
    1.0
}

fn is_one(value: &f32) -> bool {
    *value == 1.0
}

pub const VR_RENDER_SCALE_RANGE: std::ops::RangeInclusive<f32> = 0.5..=1.5;

fn default_spatial_audio(settings: &zestengine_core::SpatialAudioSettings) -> bool {
    *settings == zestengine_core::SpatialAudioSettings::default()
}

impl ProjectFile {
    /// Push [`ProjectFile::net_role_overrides`] into `registry`.
    pub fn apply_role_overrides(&self, registry: &mut zestengine_core::ComponentRegistry) {
        registry.set_role_overrides(self.net_role_overrides.iter().map(|(k, v)| (k.clone(), *v)));
    }

    pub fn eye_render_scale(&self) -> f32 {
        if self.vr_render_scale.is_finite() {
            self.vr_render_scale.clamp(*VR_RENDER_SCALE_RANGE.start(), *VR_RENDER_SCALE_RANGE.end())
        } else {
            1.0
        }
    }

    pub fn apply_layers(&self) {
        zestengine_core::set_project_layers(self.layers.clone());
    }

    pub fn adopt_layers(&mut self, scene: &zestengine_core::LayerSettings) -> bool {
        let scene = scene.clone().normalized();
        if self.layers.as_ref() == Some(&scene) {
            return false;
        }
        self.layers = Some(scene);
        self.apply_layers();
        true
    }
}

impl Default for ProjectFile {
    fn default() -> Self {
        Self {
            mods: Default::default(),
            name: "zestengine project".into(),
            last_scene: None,
            boot_scene: None,
            settings: ProjectSettings::default(),
            bindings: zestengine_core::Bindings::default(),
            audio_mixer: zestengine_core::AudioMixer::default(),
            spatial_audio: zestengine_core::SpatialAudioSettings::default(),
            audio_settings: String::new(),
            tracking_input: zestengine_core::TrackingInputSettings::default(),
            physics: zestengine_core::PhysicsSettings::default(),
            vr: false,
            vr_render_scale: 1.0,
            net_role_overrides: Default::default(),
            layers: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ProjectSettings {
    pub vsync: bool,
    pub show_stats: bool,
    pub show_stats_overlay: bool,
    /// Gizmo grid snapping.
    pub snap: bool,
    pub grid_size: f32,
    /// Asset placement: tilt dragged-in assets so their up follows the surface normal.
    pub align_to_surface: bool,
    /// Vim modal editing in the code editor.
    pub vim_mode: bool,
    /// Editor accent colour (sRGB), the only yellow in the UI; None = default.
    pub accent: Option<[u8; 3]>,
    /// Draw meshes that have a cluster DAG from GPU-culled clusters (GEO-4,
    /// experimental). `ZESTENGINE_CLUSTERS=1/0` overrides it.
    pub cluster_rendering: zestengine_render::ClusterMode,
    /// Stream the pages of big cluster meshes in and out under a memory budget
    /// (GEO-6; needs cluster rendering; off by default).
    /// `ZESTENGINE_CLUSTER_STREAM=1/0` overrides it.
    pub cluster_stream: bool,
    /// Budget of streamed geometry pages, in MB (`ZESTENGINE_CLUSTER_STREAM_MB`).
    pub cluster_stream_mb: u32,
    /// Combined geometry buffer budget in MiB; required ordinary geometry is reserved first.
    pub geometry_budget_mb: u32,
    /// Outline the selected objects in the scene view.
    pub selection_outline: bool,
    /// Outline colour (sRGB); None follows the editor accent.
    pub selection_outline_color: Option<[u8; 3]>,
    /// Outline width in logical pixels (scaled by the display's pixels per point).
    pub selection_outline_width: f32,
    /// Draw the editor ground grid in the Scene view (line spacing = `grid_size`).
    pub scene_grid: bool,
    /// Opacity of the Scene view grid, 0..1.
    pub scene_grid_opacity: f32,
    /// Anti-aliasing mode (a camera may override it). Default FXAA.
    pub anti_aliasing: zestengine_render::AntiAliasing,
    /// TAA / FSR output sharpening (CAS-like), 0..1.
    pub taa_sharpen: f32,
    /// Screen-space ambient occlusion algorithm (Low = SAO, High = GTAO). AO is
    /// enabled per scene / Volume in the post-processing profile.
    pub ao_quality: zestengine_render::AoQuality,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub frame_generation: bool,
}

impl Default for ProjectSettings {
    fn default() -> Self {
        Self {
            vsync: true,
            show_stats: true,
            show_stats_overlay: false,
            snap: false,
            grid_size: 0.5,
            align_to_surface: true,
            vim_mode: false,
            accent: None,
            cluster_rendering: zestengine_render::ClusterMode::Off,
            cluster_stream: false,
            cluster_stream_mb: 256,
            geometry_budget_mb: zestengine_render::DEFAULT_GEOMETRY_MB,
            selection_outline: true,
            selection_outline_color: None,
            selection_outline_width: 2.0,
            scene_grid: true,
            scene_grid_opacity: 1.0,
            anti_aliasing: zestengine_render::AntiAliasing::Fxaa,
            taa_sharpen: 0.25,
            ao_quality: zestengine_render::AoQuality::High,
            frame_generation: false,
        }
    }
}

pub fn load_project_file(root: impl AsRef<Path>) -> Result<ProjectFile> {
    let path = root.as_ref().join(PROJECT_FILE_NAME);
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    ron::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

pub fn save_project_file(root: impl AsRef<Path>, project: &ProjectFile) -> Result<()> {
    let path = root.as_ref().join(PROJECT_FILE_NAME);
    let text = ron::ser::to_string_pretty(project, ron::ser::PrettyConfig::default())?;
    std::fs::write(&path, text).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod vr_tests {
    use super::ProjectFile;

    #[test]
    fn the_eye_render_scale_defaults_to_native_and_is_clamped() {
        let old: ProjectFile = ron::from_str("(vr: true)").unwrap();
        assert_eq!(old.eye_render_scale(), 1.0);
        assert!(!ron::to_string(&old).unwrap().contains("vr_render_scale"));
        let set: ProjectFile = ron::from_str("(vr: true, vr_render_scale: 0.8)").unwrap();
        assert_eq!(set.eye_render_scale(), 0.8);
        assert!(ron::to_string(&set).unwrap().contains("vr_render_scale"));
        assert_eq!(ProjectFile { vr_render_scale: 9.0, ..Default::default() }.eye_render_scale(), 1.5);
        assert_eq!(ProjectFile { vr_render_scale: 0.0, ..Default::default() }.eye_render_scale(), 0.5);
        assert_eq!(ProjectFile { vr_render_scale: f32::NAN, ..Default::default() }.eye_render_scale(), 1.0);
    }
}

#[cfg(test)]
mod layer_tests {
    use super::ProjectFile;
    use zestengine_core::LayerSettings;

    #[test]
    fn project_layers_win_over_scene_layers_and_old_projects_keep_scene_layers() {
        let mut scene = LayerSettings::default();
        scene.names[3] = "Enemies".into();
        scene.set_collide(3, 3, false);
        let old: ProjectFile = ron::from_str("(name: \"old\")").unwrap();
        assert!(old.layers.is_none());
        assert!(!ron::to_string(&old).unwrap().contains("layers"), "a project without layers does not gain the field");
        assert_eq!(LayerSettings::resolve(old.layers.as_ref(), &scene), scene);

        let mut project = old.clone();
        assert!(project.adopt_layers(&scene), "the first scene's layers become the project's");
        assert!(!project.adopt_layers(&scene));
        assert_eq!(zestengine_core::project_layers(), Some(scene.clone()));
        let text = ron::ser::to_string_pretty(&project, Default::default()).unwrap();
        let back: ProjectFile = ron::from_str(&text).unwrap();
        assert_eq!(back.layers, Some(scene.clone()));

        let mut other = LayerSettings::default();
        other.names[3] = "Props".into();
        let resolved = LayerSettings::resolve(back.layers.as_ref(), &other);
        assert_eq!(resolved.names[3], "Enemies");
        assert!(!resolved.collide(3, 3));
        assert_eq!(LayerSettings::for_scene(&other), resolved);

        let short: ProjectFile = ron::from_str("(layers: Some((names: [\"Default\", \"Water\"], collision: [1])))").unwrap();
        let resolved = LayerSettings::resolve(short.layers.as_ref(), &other);
        assert_eq!(resolved.names.len(), zestengine_core::NUM_LAYERS);
        assert_eq!(resolved.collision_mask(0), 1);
        assert_eq!(resolved.collision_mask(9), u32::MAX);
        zestengine_core::set_project_layers(None);
        assert_eq!(LayerSettings::for_scene(&other), other);
    }
}

#[cfg(test)]
mod scene_grid_tests {
    use super::ProjectSettings;

    #[test]
    fn projects_saved_before_the_grid_setting_show_the_grid() {
        let s: ProjectSettings = ron::from_str("(vsync: false, grid_size: 0.25)").unwrap();
        assert!(s.scene_grid);
        assert_eq!(s.scene_grid_opacity, 1.0);
        assert_eq!(s.grid_size, 0.25);
    }
}
