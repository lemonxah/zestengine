//! Audio profile persistence; runtime types are shared through core.
use std::path::Path;
use anyhow::Result;
use crate::profile_file::{load, save};
use zestengine_core::{AudioEnvironmentProfile, SpatialAudioSettings};

pub const AUDIOFX_EXTENSION: &str = "audiofx";
pub const AUDIO_SETTINGS_EXTENSION: &str = "audiosettings";

pub fn load_audiofx(path: impl AsRef<Path>) -> Result<AudioEnvironmentProfile> {
    Ok(load::<AudioEnvironmentProfile>(path.as_ref())?.sanitized())
}
pub fn save_audiofx(path: impl AsRef<Path>, value: &AudioEnvironmentProfile) -> Result<()> {
    save(path.as_ref(), &value.sanitized())
}
pub fn load_audio_settings(path: impl AsRef<Path>) -> Result<SpatialAudioSettings> {
    Ok(load::<SpatialAudioSettings>(path.as_ref())?.sanitized())
}
pub fn save_audio_settings(path: impl AsRef<Path>, value: &SpatialAudioSettings) -> Result<()> {
    save(path.as_ref(), &value.sanitized())
}

/// Resolve an asset, preserving legacy inline project settings when unassigned.
pub fn project_audio_settings(project: &crate::ProjectFile, root: &Path) -> Result<SpatialAudioSettings> {
    if project.audio_settings.trim().is_empty() {
        Ok(project.spatial_audio.sanitized())
    } else {
        load_audio_settings(zestengine_core::asset_ref::resolve_path(root, &project.audio_settings))
    }
}

/// Editor migration: preserve inline values in a new asset without overwriting
/// an existing file. Read-only runtime loading never performs this migration.
pub fn migrate_project_audio_settings(project: &mut crate::ProjectFile, root: &Path) -> Result<bool> {
    use std::io::Write;
    if !project.audio_settings.trim().is_empty() { return Ok(false); }
    std::fs::create_dir_all(root.join("audio"))?;
    let text = ron::ser::to_string_pretty(&project.spatial_audio, ron::ser::PrettyConfig::default())?;
    for n in 0..10_000 {
        let rel = if n == 0 { "audio/Project.audiosettings".to_owned() }
            else { format!("audio/Project ({n}).audiosettings") };
        match std::fs::OpenOptions::new().write(true).create_new(true).open(root.join(&rel)) {
            Ok(mut file) => {
                file.write_all(text.as_bytes())?;
                project.audio_settings = rel;
                project.spatial_audio = SpatialAudioSettings::default();
                return Ok(true);
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.into()),
        }
    }
    anyhow::bail!("could not find an unused audio settings asset name")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn legacy_projects_and_profile_defaults() {
        let project: crate::ProjectFile = ron::from_str("(spatial_audio: (rays: 512))").unwrap();
        assert_eq!(project_audio_settings(&project, Path::new(".")).unwrap().rays, 512);
        let p: AudioEnvironmentProfile = ron::from_str("(exterior_gain: 0.3)").unwrap();
        assert_eq!(p.exterior_gain, 0.3);
        assert_eq!(p.interior_gain, 1.0);
    }

    #[test]
    fn migration_preserves_settings_and_never_overwrites_assets() {
        let root = std::env::temp_dir().join(format!("zest-audio-assets-{}", zestengine_core::ObjectId::new()));
        std::fs::create_dir_all(root.join("audio")).unwrap();
        let occupied = root.join("audio/Project.audiosettings");
        std::fs::write(&occupied, "existing user file").unwrap();
        let mut project = crate::ProjectFile::default();
        project.spatial_audio.rays = 512;
        project.spatial_audio.output = zestengine_core::OutputLayout::Stereo;
        let original = project.spatial_audio.clone();
        assert!(migrate_project_audio_settings(&mut project, &root).unwrap());
        assert_eq!(std::fs::read_to_string(occupied).unwrap(), "existing user file");
        assert_eq!(project_audio_settings(&project, &root).unwrap(), original);
        assert!(!migrate_project_audio_settings(&mut project, &root).unwrap());
        crate::save_project_file(&root, &project).unwrap();
        assert!(!std::fs::read_to_string(root.join(crate::PROJECT_FILE_NAME)).unwrap().contains("spatial_audio:"),
            "migrated projects must not save a second settings copy");
        let loaded = crate::load_project_file(&root).unwrap();
        assert_eq!(loaded.audio_settings, project.audio_settings);
        assert_eq!(project_audio_settings(&loaded, &root).unwrap().rays, 512);
        let path = root.join("Cabin.audiofx");
        let profile = AudioEnvironmentProfile { exterior_gain: 0.2, exterior_lowpass_hz: 900.0, ..Default::default() };
        save_audiofx(&path, &profile).unwrap();
        assert_eq!(load_audiofx(&path).unwrap(), profile);
        std::fs::write(&path, "malformed").unwrap();
        assert!(load_audiofx(&path).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
