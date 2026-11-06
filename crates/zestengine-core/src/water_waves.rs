//! Portable wave-asset contract and deterministic Gerstner wave generation.
use serde::{Deserialize, Serialize};
use crate::{GerstnerWave, Water};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WaterWaveSpectrum {
    pub seed: u32,
    pub count: u32,
    pub direction: f32,
    pub spread: f32,
    pub min_wavelength: f32,
    pub max_wavelength: f32,
    /// Significant wave height (approximately four times RMS surface height).
    pub height: f32,
    pub steepness: f32,
}
impl Default for WaterWaveSpectrum {
    fn default() -> Self {
        Self { seed: 17, count: 6, direction: 20.0, spread: 65.0, min_wavelength: 0.8,
            max_wavelength: 16.0, height: 0.65, steepness: 0.55 }
    }
}
impl WaterWaveSpectrum {
    pub fn generate(&self) -> Vec<GerstnerWave> {
        let finite = |v: f32, d: f32| if v.is_finite() { v } else { d };
        let n = self.count.clamp(1, Water::MAX_WAVES as u32) as usize;
        let lo = finite(self.min_wavelength, 0.8).clamp(0.05, 1000.0);
        let hi = finite(self.max_wavelength, 16.0).clamp(lo, 1000.0);
        let height = finite(self.height, 0.65).clamp(0.0, 100.0);
        let mut state = self.seed;
        let mut random = || { state = state.wrapping_mul(1664525).wrapping_add(1013904223); (state >> 8) as f32 / 16777216.0 };
        let mut waves: Vec<_> = (0..n).map(|i| {
            let t = (i as f32 + random()) / n as f32;
            let wavelength = lo * (hi / lo).powf(t);
            GerstnerWave {
                direction: finite(self.direction, 0.0) + (random() - 0.5) * finite(self.spread, 65.0).clamp(0.0, 360.0),
                wavelength,
                amplitude: (wavelength / hi).sqrt(),
                steepness: finite(self.steepness, 0.55).clamp(0.0, 1.0), speed: 1.0,
            }
        }).collect();
        let sum = waves.iter().map(|w| w.amplitude * w.amplitude).sum::<f32>();
        let scale = height / (4.0 * (sum * 0.5).sqrt()).max(1e-6);
        for wave in &mut waves { wave.amplitude *= scale; }
        waves
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WaterWaveProfile {
    /// None uses the authored list verbatim (subject to runtime safety limits).
    pub spectrum: Option<WaterWaveSpectrum>,
    pub waves: Vec<GerstnerWave>,
}
impl Default for WaterWaveProfile {
    fn default() -> Self { Self { spectrum: Some(WaterWaveSpectrum::default()), waves: Vec::new() } }
}
impl WaterWaveProfile {
    pub fn resolved(&self) -> Vec<GerstnerWave> {
        if let Some(spectrum) = &self.spectrum { return spectrum.generate(); }
        self.waves.iter().take(Water::MAX_WAVES).filter(|w| [w.direction, w.wavelength, w.amplitude, w.steepness, w.speed].iter().all(|v| v.is_finite()))
            .map(|w| GerstnerWave { direction: w.direction.rem_euclid(360.0), wavelength: w.wavelength.clamp(0.05, 1000.0),
                amplitude: w.amplitude.clamp(0.0, 50.0), steepness: w.steepness.clamp(0.0, 1.0), speed: w.speed.clamp(0.0, 10.0) }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn generated_sea_is_repeatable_and_has_the_authored_energy() {
        let spectrum = WaterWaveSpectrum::default();
        let a = spectrum.generate();
        assert_eq!(a, spectrum.generate());
        assert_ne!(a, WaterWaveSpectrum { seed: 18, ..spectrum.clone() }.generate());
        let hs = 4.0 * (a.iter().map(|w| w.amplitude.powi(2)).sum::<f32>() * 0.5).sqrt();
        assert!((hs - spectrum.height).abs() < 1e-5);
        assert!(a.iter().all(|w| w.wavelength >= spectrum.min_wavelength && w.wavelength <= spectrum.max_wavelength));
    }
}
