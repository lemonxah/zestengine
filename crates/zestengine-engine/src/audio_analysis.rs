//! Audio analysis for reactive materials (the engine side of ZestEngine Toon's
//! `AUDIO` feature, our counterpart of an AudioLink-style input): four band
//! energies (bass, low mid, high mid, treble), overall volume, and a short
//! history of each, from a mono sample stream.
//!
//! Pure Rust and allocation-free per block: feed it the mixed output with
//! [`AudioAnalyzer::push`], read [`AudioBands`] once per frame, and hand them to
//! the materials with `LoadedScene::set_audio_bands`. Band edges (Hz):
//! bass 20-250, low mid 250-2000, high mid 2000-6000, treble 6000-16000.

/// FFT window length (samples). Must be a power of two.
pub const WINDOW: usize = 1024;
/// Frames of band history kept (about 1 s at 60 fps).
pub const HISTORY: usize = 64;

/// Band energies, each 0..1 after the analyser's gain and smoothing.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AudioBands {
    pub bass: f32,
    pub low_mid: f32,
    pub high_mid: f32,
    pub treble: f32,
    /// Overall level (RMS of the window).
    pub volume: f32,
}

impl AudioBands {
    /// The four band values as the Toon `audio_bands` vector.
    pub fn vec4(&self) -> [f32; 4] {
        [self.bass, self.low_mid, self.high_mid, self.treble]
    }
}

/// Short-time band analysis of a mono stream.
pub struct AudioAnalyzer {
    sample_rate: f32,
    ring: Vec<f32>,
    write: usize,
    filled: usize,
    window: Vec<f32>,
    re: Vec<f32>,
    im: Vec<f32>,
    bands: AudioBands,
    history: Vec<AudioBands>,
    /// Multiplies the raw energies (a full-scale sine in a band reads about 1).
    pub gain: f32,
    /// Exponential smoothing time constant in seconds (rise is immediate; fall follows this).
    pub release: f32,
}

impl AudioAnalyzer {
    pub fn new(sample_rate: u32) -> Self {
        let window: Vec<f32> = (0..WINDOW).map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / WINDOW as f32).cos()).collect();
        Self {
            sample_rate: sample_rate.max(1) as f32,
            ring: vec![0.0; WINDOW],
            write: 0,
            filled: 0,
            window,
            re: vec![0.0; WINDOW],
            im: vec![0.0; WINDOW],
            bands: AudioBands::default(),
            history: Vec::with_capacity(HISTORY),
            gain: 1.0,
            release: 0.12,
        }
    }

    /// Append mono samples.
    pub fn push(&mut self, samples: &[f32]) {
        for &s in samples {
            self.ring[self.write] = s;
            self.write = (self.write + 1) % WINDOW;
        }
        self.filled = (self.filled + samples.len()).min(WINDOW);
    }

    /// Analyse the latest window; `dt` is the seconds since the last call (for the release smoothing).
    pub fn update(&mut self, dt: f32) -> AudioBands {
        if self.filled < WINDOW / 4 {
            return self.bands;
        }
        let mut sum_sq = 0.0;
        for i in 0..WINDOW {
            let s = self.ring[(self.write + i) % WINDOW];
            sum_sq += s * s;
            self.re[i] = s * self.window[i];
            self.im[i] = 0.0;
        }
        fft(&mut self.re, &mut self.im);
        let bin_hz = self.sample_rate / WINDOW as f32;
        // A full-scale sine under a Hann window peaks at WINDOW / 4 in its bin.
        let norm = 4.0 / WINDOW as f32;
        let band = |lo: f32, hi: f32, re: &[f32], im: &[f32]| -> f32 {
            let a = ((lo / bin_hz) as usize).max(1);
            let b = ((hi / bin_hz) as usize).min(WINDOW / 2 - 1).max(a);
            let mut peak = 0.0f32;
            for k in a..=b {
                peak = peak.max((re[k] * re[k] + im[k] * im[k]).sqrt() * norm);
            }
            peak
        };
        let raw = AudioBands {
            bass: band(20.0, 250.0, &self.re, &self.im),
            low_mid: band(250.0, 2000.0, &self.re, &self.im),
            high_mid: band(2000.0, 6000.0, &self.re, &self.im),
            treble: band(6000.0, 16000.0, &self.re, &self.im),
            volume: (sum_sq / WINDOW as f32).sqrt() * std::f32::consts::SQRT_2,
        };
        let fall = if self.release > 1e-4 { (-dt.max(0.0) / self.release).exp() } else { 0.0 };
        let follow = |old: f32, new: f32| -> f32 { (if new >= old { new } else { new + (old - new) * fall }) * 1.0 };
        let g = self.gain;
        self.bands = AudioBands {
            bass: follow(self.bands.bass, (raw.bass * g).clamp(0.0, 1.0)),
            low_mid: follow(self.bands.low_mid, (raw.low_mid * g).clamp(0.0, 1.0)),
            high_mid: follow(self.bands.high_mid, (raw.high_mid * g).clamp(0.0, 1.0)),
            treble: follow(self.bands.treble, (raw.treble * g).clamp(0.0, 1.0)),
            volume: follow(self.bands.volume, (raw.volume * g).clamp(0.0, 1.0)),
        };
        if self.history.len() == HISTORY {
            self.history.remove(0);
        }
        self.history.push(self.bands);
        self.bands
    }

    /// The most recent bands.
    pub fn bands(&self) -> AudioBands {
        self.bands
    }

    /// Oldest to newest, up to [`HISTORY`] frames.
    pub fn history(&self) -> &[AudioBands] {
        &self.history
    }
}

/// In-place radix-2 FFT.
fn fft(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -std::f32::consts::TAU / len as f32;
        let (wr, wi) = (ang.cos(), ang.sin());
        for start in (0..n).step_by(len) {
            let (mut cr, mut ci) = (1.0f32, 0.0f32);
            for k in 0..len / 2 {
                let (a, b) = (start + k, start + k + len / 2);
                let (tr, ti) = (re[b] * cr - im[b] * ci, re[b] * ci + im[b] * cr);
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                let nr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = nr;
            }
        }
        len <<= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(hz: f32, amp: f32, n: usize, rate: u32) -> Vec<f32> {
        (0..n).map(|i| amp * (std::f32::consts::TAU * hz * i as f32 / rate as f32).sin()).collect()
    }

    fn analyse(hz: f32, amp: f32) -> AudioBands {
        let mut a = AudioAnalyzer::new(48_000);
        a.push(&sine(hz, amp, 4096, 48_000));
        a.update(0.016)
    }

    #[test]
    fn each_band_hears_its_own_frequency() {
        let b = analyse(100.0, 0.8);
        assert!(b.bass > 0.6 && b.low_mid < 0.1 && b.high_mid < 0.05 && b.treble < 0.05, "{b:?}");
        let b = analyse(1000.0, 0.8);
        assert!(b.low_mid > 0.6 && b.bass < 0.1 && b.high_mid < 0.1 && b.treble < 0.05, "{b:?}");
        let b = analyse(4000.0, 0.8);
        assert!(b.high_mid > 0.6 && b.low_mid < 0.1 && b.treble < 0.1, "{b:?}");
        let b = analyse(10_000.0, 0.8);
        assert!(b.treble > 0.6 && b.high_mid < 0.1 && b.bass < 0.05, "{b:?}");
    }

    #[test]
    fn volume_tracks_amplitude_and_silence_is_zero() {
        let loud = analyse(440.0, 0.9).volume;
        let quiet = analyse(440.0, 0.1).volume;
        assert!((loud - 0.9).abs() < 0.08, "RMS of a sine scaled to its amplitude: {loud}");
        assert!(quiet < loud * 0.2);
        let mut a = AudioAnalyzer::new(44_100);
        a.push(&vec![0.0; 2048]);
        assert_eq!(a.update(0.016), AudioBands::default());
    }

    #[test]
    fn release_smooths_the_fall_and_history_is_bounded() {
        let mut a = AudioAnalyzer::new(48_000);
        a.push(&sine(100.0, 0.9, 2048, 48_000));
        let hit = a.update(0.016).bass;
        a.push(&vec![0.0; 2048]);
        let after = a.update(0.016).bass;
        assert!(after < hit && after > hit * 0.5, "decays instead of cutting: {hit} -> {after}");
        for _ in 0..200 {
            a.update(0.016);
        }
        assert_eq!(a.history().len(), HISTORY);
        assert!(a.bands().bass < 0.01, "and reaches silence");
    }

    #[test]
    fn too_little_audio_keeps_the_last_bands() {
        let mut a = AudioAnalyzer::new(48_000);
        a.push(&[0.5; 16]);
        assert_eq!(a.update(0.016), AudioBands::default());
    }
}
