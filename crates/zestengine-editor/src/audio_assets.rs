//! Editor-only migration of legacy inline audio configuration.
use super::EngineApp;

impl EngineApp {
    pub(super) fn migrate_audio_settings(&mut self) {
        match zestengine_assets::audio_file::migrate_project_audio_settings(&mut self.project, &self.project_root) {
            Ok(true) => {
                if let Err(e) = zestengine_assets::save_project_file(&self.project_root, &self.project) {
                    tracing::warn!("saving audio settings reference: {e:#}");
                }
            }
            Ok(false) => {}
            Err(e) => tracing::warn!("keeping legacy audio settings: {e:#}"),
        }
    }
}
