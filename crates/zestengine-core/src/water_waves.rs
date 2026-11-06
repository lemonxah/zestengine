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
    /// Jitter within each wavelength stratum, 0 = evenly spaced, 1 = random.
    pub randomness: f32,
    /// Exponent of the logarithmic wavelength distribution; >1 favours short waves.
    pub wavelength_falloff: f32,
    /// Relative amplitude scales as (wavelength / longest)^falloff before
    /// normalisation to significant height. Zero gives equal wave energy.
    pub amplitude_falloff: f32,
    /// Multipliers of steepness at the shortest/longest wavelength.
    pub small_wave_steepness: f32,
    pub large_wave_steepness: f32,
}
impl Default for WaterWaveSpectrum {
    fn default() -> Self {
        Self { seed: 17, count: 6, direction: 20.0, spread: 65.0, min_wavelength: 0.8,
            max_wavelength: 16.0, height: 0.65, steepness: 0.55,
            randomness: 1.0, wavelength_falloff: 1.0, amplitude_falloff: 0.5,
            small_wave_steepness: 1.0, large_wave_steepness: 1.0 }
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
            let jitter = 0.5 + (random() - 0.5) * finite(self.randomness, 1.0).clamp(0.0, 1.0);
            let t = ((i as f32 + jitter) / n as f32).powf(finite(self.wavelength_falloff, 1.0).clamp(0.05, 10.0));
            let wavelength = lo * (hi / lo).powf(t);
            GerstnerWave {
                direction: finite(self.direction, 0.0) + (random() - 0.5) * finite(self.spread, 65.0).clamp(0.0, 360.0),
                wavelength,
                amplitude: (wavelength / hi).powf(finite(self.amplitude_falloff, 0.5).clamp(0.0, 10.0)),
                steepness: (finite(self.steepness, 0.55) * (
                    finite(self.small_wave_steepness, 1.0).clamp(0.0, 2.0) * (1.0 - t)
                    + finite(self.large_wave_steepness, 1.0).clamp(0.0, 2.0) * t)).clamp(0.0, 1.0), speed: 1.0,
            }
        }).collect();
        // Avoid energy underflow for steep falloff across wide spectra.
        let sum = waves.iter().map(|w| (w.amplitude as f64).powi(2)).sum::<f64>();
        let scale = height as f64 / (4.0 * (sum * 0.5).sqrt()).max(f64::MIN_POSITIVE);
        for wave in &mut waves { wave.amplitude = (wave.amplitude as f64 * scale) as f32; }
        waves
    }
}

/// Engine extensions generate the same portable waves used by rendering and
/// headless physics. Implementations must be deterministic for the same settings.
/// Register on LoadedScene before resolving assets; this API has no renderer or
/// editor dependencies. Output is validated and bounded by WaterWaveProfile.
pub trait WaterWaveGenerator: Send + Sync {
    fn generate(&self, settings: &std::collections::BTreeMap<String, f32>) -> Result<Vec<GerstnerWave>, String>;
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WaterWaveGeneratorConfig {
    pub id: String,
    pub settings: std::collections::BTreeMap<String, f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WaterWaveProfile {
    /// Registered engine generator. None uses the built-in spectrum/list below.
    pub generator: Option<WaterWaveGeneratorConfig>,
    /// None uses the authored list verbatim (subject to runtime safety limits).
    pub spectrum: Option<WaterWaveSpectrum>,
    pub waves: Vec<GerstnerWave>,
}
impl Default for WaterWaveProfile {
    fn default() -> Self { Self { generator: None, spectrum: Some(WaterWaveSpectrum::default()), waves: Vec::new() } }
}
impl WaterWaveProfile {
    pub fn resolved(&self) -> Vec<GerstnerWave> {
        if let Some(spectrum) = &self.spectrum { return spectrum.generate(); }
        Self::validated(&self.waves)
    }
    pub fn generated(&self, generator: &dyn WaterWaveGenerator) -> Result<Vec<GerstnerWave>, String> {
        let config = self.generator.as_ref().ok_or("profile has no custom generator")?;
        let waves = generator.generate(&config.settings)?;
        if waves.iter().any(|w| [w.direction, w.wavelength, w.amplitude, w.steepness, w.speed].iter().any(|v| !v.is_finite())) {
            return Err(format!("wave generator '{}' produced non-finite values", config.id));
        }
        Ok(Self::validated(&waves))
    }
    fn validated(waves: &[GerstnerWave]) -> Vec<GerstnerWave> {
        waves.iter().filter(|w| [w.direction, w.wavelength, w.amplitude, w.steepness, w.speed].iter().all(|v| v.is_finite())).take(Water::MAX_WAVES)
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

    #[test]
    fn legacy_profiles_keep_their_distribution_and_new_controls_keep_energy() {
        let legacy: WaterWaveProfile = ron::from_str("(spectrum:Some((seed:17,count:6,direction:20.0,spread:65.0,min_wavelength:0.8,max_wavelength:16.0,height:0.65,steepness:0.55)),waves:[])").unwrap();
        assert_eq!(legacy, WaterWaveProfile::default());
        let base = WaterWaveSpectrum { count: 32, spread: 0.0, randomness: 0.0, ..Default::default() };
        let regular = base.generate();
        assert_eq!(regular.len(), 32);
        let altered = WaterWaveSpectrum { wavelength_falloff: 3.0, amplitude_falloff: 2.0,
            small_wave_steepness: 0.0, large_wave_steepness: 1.5, ..base.clone() }.generate();
        assert!(altered.iter().zip(&regular).all(|(a,b)| a.wavelength < b.wavelength));
        assert!(altered.first().unwrap().steepness < altered.last().unwrap().steepness);
        let hs = 4.0 * (altered.iter().map(|w| w.amplitude.powi(2)).sum::<f32>() * 0.5).sqrt();
        assert!((hs - base.height).abs() < 1e-5);
        assert_eq!(regular, WaterWaveSpectrum { seed: 42, ..base }.generate());
    }

    #[test]
    fn hostile_spectrum_inputs_stay_finite_and_bounded() {
        let spectrum = WaterWaveSpectrum { count: u32::MAX, height: f32::INFINITY,
            min_wavelength: f32::NAN, max_wavelength: -1.0, randomness: f32::NAN,
            amplitude_falloff: f32::INFINITY, wavelength_falloff: f32::NAN,
            small_wave_steepness: f32::INFINITY, large_wave_steepness: -1.0, ..Default::default() };
        let waves = spectrum.generate();
        assert_eq!(waves.len(), Water::MAX_WAVES);
        assert!(waves.iter().all(|w| w.amplitude.is_finite() && w.wavelength >= 0.05 && (0.0..=1.0).contains(&w.steepness)));
    }

    #[test]
    fn generator_output_is_bounded_and_nonfinite_output_is_rejected() {
        struct Generator(bool);
        impl WaterWaveGenerator for Generator {
            fn generate(&self, _: &std::collections::BTreeMap<String, f32>) -> Result<Vec<GerstnerWave>, String> {
                Ok(vec![GerstnerWave { amplitude: if self.0 { f32::NAN } else { 100.0 }, ..Default::default() }; 100])
            }
        }
        let profile = WaterWaveProfile { generator: Some(Default::default()), ..Default::default() };
        let waves = profile.generated(&Generator(false)).unwrap();
        assert_eq!(waves.len(), Water::MAX_WAVES);
        assert!(waves.iter().all(|w| w.amplitude == 50.0));
        assert!(profile.generated(&Generator(true)).is_err());
    }
}
