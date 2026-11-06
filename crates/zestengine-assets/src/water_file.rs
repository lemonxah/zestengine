//! Reusable water-wave asset persistence. Generation lives in core.
use std::path::Path;
use zestengine_core::WaterWaveProfile;
pub const WATER_WAVES_EXTENSION: &str = "waterwaves";
pub fn load_water_waves(path: impl AsRef<Path>) -> anyhow::Result<WaterWaveProfile> {
    crate::profile_file::load(path.as_ref())
}
pub fn save_water_waves(path: impl AsRef<Path>, profile: &WaterWaveProfile) -> anyhow::Result<()> {
    crate::profile_file::save(path.as_ref(), profile)
}
