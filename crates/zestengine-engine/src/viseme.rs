//! Speech to visemes, per 20 ms frame of 16 kHz audio: loudness gates
//! silence, the first two formants (spectral centroids of the F1 and F2 bands)
//! pick the vowel, high-band energy picks fricatives and a sharp onset after a
//! pause reads as a plosive. Heuristic, not a trained model, but cheap enough
//! to run on every voice and good enough to sell the mouth shapes.

use zestengine_core::{VISEMES, VisemeFrame};

const N: usize = 512;
const RATE: f32 = 16_000.0;
const GATE_DB: f32 = -48.0;

/// Vowel formant prototypes (F1, F2) in Hz.
const VOWELS: [(&str, f32, f32); 5] = [("aa", 750.0, 1250.0), ("E", 530.0, 1850.0), ("ih", 350.0, 2150.0), ("oh", 500.0, 900.0), ("ou", 330.0, 800.0)];

fn index(name: &str) -> usize {
    VISEMES.iter().position(|v| *v == name).unwrap_or(0)
}

#[derive(Default)]
pub struct VisemeAnalyzer {
    prev_level: f32,
}

impl VisemeAnalyzer {
    pub fn analyze(&mut self, pcm: &[i16]) -> VisemeFrame {
        let mut out = VisemeFrame::default();
        let rms = (pcm.iter().map(|s| (*s as f32).powi(2)).sum::<f32>() / pcm.len().max(1) as f32).sqrt();
        let db = 20.0 * (rms / 32768.0).max(1e-6).log10();
        let level = ((db - GATE_DB) / 30.0).clamp(0.0, 1.0);
        let onset = self.prev_level < 0.15 && level > 0.4;
        self.prev_level = level;
        if level <= 0.0 {
            out.0[0] = 1.0;
            return out;
        }

        let power = spectrum(pcm);
        let hz = RATE / N as f32;
        let band = |lo: f32, hi: f32| -> f32 { power[(lo / hz) as usize..(hi / hz) as usize].iter().sum() };
        // Formants as the strongest peaks of a lightly smoothed spectrum: F1 in
        // its band, F2 above it (back vowels put F2 inside the F1 band).
        let smooth: Vec<f32> = (0..power.len())
            .map(|k| (power[k.saturating_sub(1)] + 2.0 * power[k] + power[(k + 1).min(power.len() - 1)]) * 0.25)
            .collect();
        let peak = |lo: f32, hi: f32| -> f32 {
            let (a, b) = ((lo / hz) as usize, (hi / hz) as usize);
            let k = (a..b.max(a + 1)).max_by(|x, y| smooth[*x].total_cmp(&smooth[*y])).unwrap_or(a);
            k as f32 * hz
        };
        let total = band(80.0, 7900.0).max(1e-9);
        let high = band(3500.0, 7900.0) / total;
        let f1 = peak(250.0, 1000.0);
        let f2 = peak(f1 + 300.0, 2800.0);

        let fricative = smoothstep(0.3, 0.6, high);
        let voiced = (1.0 - fricative) * level;
        let mut vowel = [0.0f32; 5];
        for (k, (_, p1, p2)) in VOWELS.iter().enumerate() {
            let d = ((f1 - p1) / 160.0).powi(2) + ((f2 - p2) / 380.0).powi(2);
            vowel[k] = (-0.5 * d).exp();
        }
        let vsum: f32 = vowel.iter().sum::<f32>().max(1e-6);
        for (k, (name, _, _)) in VOWELS.iter().enumerate() {
            out.0[index(name)] = voiced * vowel[k] / vsum;
        }
        let fric = fricative * level;
        out.0[index("SS")] = fric * smoothstep(0.55, 0.75, high);
        out.0[index("CH")] = fric * (1.0 - smoothstep(0.55, 0.75, high)) * 0.7;
        out.0[index("FF")] = fric * (1.0 - level) * 0.6;
        out.0[index("TH")] = fric * (1.0 - level) * 0.3;
        if onset {
            out.0[index("PP")] = 0.6;
            out.0[index("DD")] = 0.3;
        }
        out.0[index("nn")] = voiced * (1.0 - smoothstep(0.02, 0.08, band(1000.0, 3000.0) / total)) * 0.5;
        out.0[0] = 1.0 - level;
        out
    }
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Power spectrum (N / 2 bins) of a Hann-windowed, zero-padded frame.
fn spectrum(pcm: &[i16]) -> Vec<f32> {
    let n = pcm.len().min(N);
    let mut re = vec![0.0f32; N];
    let mut im = vec![0.0f32; N];
    for (i, s) in pcm.iter().take(n).enumerate() {
        let w = 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / n as f32).cos();
        re[i] = *s as f32 / 32768.0 * w;
    }
    fft(&mut re, &mut im);
    (0..N / 2).map(|k| re[k] * re[k] + im[k] * im[k]).collect()
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
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -std::f32::consts::TAU / len as f32;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (s, c) = (ang * k as f32).sin_cos();
                let (a, b) = (start + k, start + k + len / 2);
                let (tr, ti) = (re[b] * c - im[b] * s, re[b] * s + im[b] * c);
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        len <<= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A voiced frame: 140 Hz harmonics shaped by two formant peaks.
    fn vowel(f1: f32, f2: f32) -> Vec<i16> {
        (0..320)
            .map(|i| {
                let t = i as f32 / RATE;
                let mut v = 0.0;
                for h in 1..50 {
                    let f = 140.0 * h as f32;
                    let amp = (-((f - f1) / 120.0).powi(2)).exp() + 0.7 * (-((f - f2) / 180.0).powi(2)).exp();
                    v += amp * (std::f32::consts::TAU * f * t).sin();
                }
                (v * 4000.0) as i16
            })
            .collect()
    }

    fn top(f: &VisemeFrame) -> &'static str {
        let (k, _) = f.0.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
        VISEMES[k]
    }

    #[test]
    fn vowels_silence_and_hiss_classify() {
        let mut a = VisemeAnalyzer::default();
        assert_eq!(top(&a.analyze(&[0; 320])), "sil");
        assert_eq!(top(&a.analyze(&vowel(750.0, 1250.0))), "aa");
        assert_eq!(top(&a.analyze(&vowel(330.0, 800.0))), "ou");
        assert_eq!(top(&a.analyze(&vowel(350.0, 2150.0))), "ih");
        let mut seed = 12345u32;
        let hiss: Vec<i16> = (0..320)
            .map(|i| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let n = ((seed >> 16) as i16) as f32;
                // First difference: a crude high-pass, so the energy sits up high.
                (n * if i % 2 == 0 { 0.4 } else { -0.4 }) as i16
            })
            .collect();
        let f = a.analyze(&hiss);
        assert!(f.get(index("SS")) + f.get(index("CH")) > 0.3, "{:?}", f.0);
    }
}
