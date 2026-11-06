//! A tap on the mixed audio output, feeding [`crate::audio_analysis`] (ZestEngine
//! Toon's `AUDIO` feature).
//!
//! rodio mixes every `Sink` inside the output stream, where nothing can read
//! the sum, so each voice carries a [`Tapped`] wrapper at the end of its source
//! chain. On the audio thread the wrapper downmixes its frames to mono and
//! stores them in a per-voice ring of atomics (no lock, no allocation; it does
//! nothing at all while the tap is disabled). Once a frame the main thread
//! sums the newest samples of every live voice ([`AudioTap::mix`]), scaled by
//! the sink's volume (bus gain and distance fall-off) and resampled to
//! [`ANALYSIS_RATE`], then runs the analyser on that.
//!
//! The sum is aligned on each voice's newest sample, so it is the mix to within
//! the output buffer's latency. It is taken per voice before rodio's own mix, so
//! it is the pre-limiter sum: loud overlapping voices read above 1 and clamp in
//! the analyser. Per-bus taps are not kept (voices do not know their bus here).

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

use crate::audio_analysis::{AudioAnalyzer, AudioBands, WINDOW};

/// Samples kept per voice (a power of two well above the longest read).
pub const RING: usize = 8192;
/// The rate the mix is resampled to for analysis.
pub const ANALYSIS_RATE: u32 = 48_000;

/// One voice's recent mono samples.
pub struct VoiceTap {
    ring: Box<[AtomicU32]>,
    /// Total samples ever written.
    head: AtomicUsize,
    rate: u32,
    gain: AtomicU32,
    speed: AtomicU32,
    enabled: Arc<AtomicBool>,
}

impl VoiceTap {
    fn new(rate: u32, enabled: Arc<AtomicBool>) -> Arc<Self> {
        Arc::new(Self {
            ring: (0..RING).map(|_| AtomicU32::new(0)).collect(),
            head: AtomicUsize::new(0),
            rate: rate.max(1),
            gain: AtomicU32::new(1.0f32.to_bits()),
            speed: AtomicU32::new(1.0f32.to_bits()),
            enabled,
        })
    }

    /// Audio thread: store one mono sample.
    #[inline]
    pub fn push(&self, mono: f32) {
        let h = self.head.load(Ordering::Relaxed);
        self.ring[h % RING].store(mono.to_bits(), Ordering::Relaxed);
        self.head.store(h.wrapping_add(1), Ordering::Release);
    }

    #[inline]
    fn on(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// The sink's volume (the output stage scales every sample by it).
    pub fn set_gain(&self, gain: f32) {
        self.gain.store(gain.max(0.0).to_bits(), Ordering::Relaxed);
    }

    /// The sink's playback speed (pitch shifts the spectrum).
    pub fn set_speed(&self, speed: f32) {
        self.speed.store(speed.clamp(0.1, 4.0).to_bits(), Ordering::Relaxed);
    }

    /// Add this voice's newest `out.len()` samples (at [`ANALYSIS_RATE`], the
    /// last element the newest) into `out`.
    fn add_into(&self, out: &mut [f32]) {
        let head = self.head.load(Ordering::Acquire);
        if head == 0 || out.is_empty() {
            return;
        }
        let gain = f32::from_bits(self.gain.load(Ordering::Relaxed));
        let speed = f32::from_bits(self.speed.load(Ordering::Relaxed));
        let step = (self.rate as f32 * speed / ANALYSIS_RATE as f32).clamp(0.05, 4.0);
        let n = out.len();
        let newest = (head - 1) as f32;
        let at = |i: usize| f32::from_bits(self.ring[i % RING].load(Ordering::Relaxed));
        for (j, o) in out.iter_mut().enumerate() {
            let pos = newest - (n - 1 - j) as f32 * step;
            if pos < 0.0 {
                continue;
            }
            let i0 = pos.floor() as usize;
            let frac = pos - i0 as f32;
            let a = at(i0);
            let b = if i0 + 1 < head { at(i0 + 1) } else { a };
            *o += (a + (b - a) * frac) * gain;
        }
    }
}

/// The set of voice taps. Cheap to share; only the main thread locks it.
pub struct AudioTap {
    enabled: Arc<AtomicBool>,
    voices: Mutex<Vec<Weak<VoiceTap>>>,
}

impl Default for AudioTap {
    fn default() -> Self {
        Self { enabled: Arc::new(AtomicBool::new(false)), voices: Mutex::new(Vec::new()) }
    }
}

impl AudioTap {
    /// Whether voices record samples. Off: the audio thread only reads this flag.
    pub fn set_enabled(&self, on: bool) {
        self.enabled.store(on, Ordering::Relaxed);
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// A tap for a new voice whose source runs at `rate` Hz.
    pub fn register(&self, rate: u32) -> Arc<VoiceTap> {
        let tap = VoiceTap::new(rate, self.enabled.clone());
        let mut voices = self.voices.lock().unwrap();
        voices.retain(|w| w.strong_count() > 0);
        voices.push(Arc::downgrade(&tap));
        tap
    }

    /// Voices still alive.
    pub fn voice_count(&self) -> usize {
        self.voices.lock().unwrap().iter().filter(|w| w.strong_count() > 0).count()
    }

    /// Overwrite `out` with the summed mono mix of every live voice (newest last).
    pub fn mix(&self, out: &mut [f32]) {
        out.fill(0.0);
        let mut voices = self.voices.lock().unwrap();
        voices.retain(|w| match w.upgrade() {
            Some(v) => {
                v.add_into(out);
                true
            }
            None => false,
        });
    }
}

/// Mix tap -> analyser, one `frame` per game frame.
pub struct AudioReactive {
    analyzer: AudioAnalyzer,
    scratch: Vec<f32>,
    /// Frames since the last "does any material use AUDIO" probe.
    since_probe: u32,
    /// Some material consumed the bands last probe/frame.
    pub active: bool,
}

/// Frames between probes for AUDIO materials while the tap is off.
const PROBE_EVERY: u32 = 30;

impl Default for AudioReactive {
    fn default() -> Self {
        Self { analyzer: AudioAnalyzer::new(ANALYSIS_RATE), scratch: vec![0.0; WINDOW], since_probe: PROBE_EVERY, active: false }
    }
}

impl AudioReactive {
    /// Analyse the tap's current mix.
    pub fn analyze(&mut self, tap: &AudioTap, dt: f32) -> AudioBands {
        self.analyze_with(|out| tap.mix(out), dt)
    }

    /// Analyse whatever `read` fills (the newest [`WINDOW`] mono samples at [`ANALYSIS_RATE`]).
    pub fn analyze_with(&mut self, read: impl FnOnce(&mut [f32]), dt: f32) -> AudioBands {
        read(&mut self.scratch);
        self.analyzer.push(&self.scratch);
        self.analyzer.update(dt)
    }

    /// Whether this frame should look for AUDIO materials while idle.
    pub fn should_probe(&mut self) -> bool {
        self.since_probe += 1;
        if self.since_probe >= PROBE_EVERY {
            self.since_probe = 0;
            true
        } else {
            false
        }
    }

    pub fn analyzer(&self) -> &AudioAnalyzer {
        &self.analyzer
    }
}

/// A rodio source that copies its frames, downmixed to mono, into a [`VoiceTap`].
#[cfg(feature = "client")]
pub struct Tapped<S> {
    inner: S,
    tap: Arc<VoiceTap>,
    acc: f32,
    idx: u16,
}

#[cfg(feature = "client")]
impl<S: rodio::Source<Item = f32>> Tapped<S> {
    pub fn new(inner: S, tap: Arc<VoiceTap>) -> Self {
        Self { inner, tap, acc: 0.0, idx: 0 }
    }
}

#[cfg(feature = "client")]
impl<S: rodio::Source<Item = f32>> Iterator for Tapped<S> {
    type Item = f32;

    #[inline]
    fn next(&mut self) -> Option<f32> {
        let x = self.inner.next()?;
        if self.tap.on() {
            self.acc += x;
            self.idx += 1;
            let ch = self.inner.channels().max(1);
            if self.idx >= ch {
                self.tap.push(self.acc / ch as f32);
                self.acc = 0.0;
                self.idx = 0;
            }
        }
        Some(x)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[cfg(feature = "client")]
impl<S: rodio::Source<Item = f32>> rodio::Source for Tapped<S> {
    fn current_frame_len(&self) -> Option<usize> {
        self.inner.current_frame_len()
    }
    fn channels(&self) -> u16 {
        self.inner.channels()
    }
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate()
    }
    fn total_duration(&self) -> Option<std::time::Duration> {
        self.inner.total_duration()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(hz: f32, amp: f32, n: usize, rate: u32) -> Vec<f32> {
        (0..n).map(|i| amp * (std::f32::consts::TAU * hz * i as f32 / rate as f32).sin()).collect()
    }

    fn feed(v: &VoiceTap, s: &[f32]) {
        for &x in s {
            v.push(x);
        }
    }

    #[test]
    fn synthetic_mixer_output_reaches_the_bands() {
        let tap = AudioTap::default();
        tap.set_enabled(true);
        let bass = tap.register(48_000);
        let treble = tap.register(48_000);
        feed(&bass, &sine(100.0, 0.5, 4096, 48_000));
        feed(&treble, &sine(10_000.0, 0.4, 4096, 48_000));
        let mut r = AudioReactive::default();
        let b = r.analyze(&tap, 0.016);
        assert!(b.bass > 0.35 && b.treble > 0.25, "{b:?}");
        assert!(b.low_mid < 0.1 && b.high_mid < 0.1, "{b:?}");
        assert!(b.volume > 0.3, "{b:?}");
    }

    #[test]
    fn silence_and_dead_voices_read_zero() {
        let tap = AudioTap::default();
        let mut r = AudioReactive::default();
        assert_eq!(r.analyze(&tap, 0.016), AudioBands::default());
        let v = tap.register(44_100);
        feed(&v, &sine(440.0, 0.9, 4096, 44_100));
        assert_eq!(tap.voice_count(), 1);
        drop(v);
        assert_eq!(tap.voice_count(), 0);
        assert_eq!(r.analyze(&tap, 0.016), AudioBands::default(), "a dropped voice leaves no sound");
    }

    #[test]
    fn gain_scales_and_other_rates_are_resampled() {
        let tap = AudioTap::default();
        let v = tap.register(44_100);
        feed(&v, &sine(1000.0, 0.8, 4096, 44_100));
        let mut out = vec![0.0; WINDOW];
        tap.mix(&mut out);
        let loud = AudioReactive::default().analyze(&tap, 0.016);
        assert!(loud.low_mid > 0.6, "44.1 kHz sine lands in the right band after resampling: {loud:?}");
        v.set_gain(0.25);
        let quiet = AudioReactive::default().analyze(&tap, 0.016);
        assert!((quiet.low_mid - loud.low_mid * 0.25).abs() < 0.05, "{loud:?} {quiet:?}");
        // Speed 3 triples the pitch: 1 kHz now reads as 3 kHz (high mid).
        v.set_gain(1.0);
        v.set_speed(3.0);
        let fast = AudioReactive::default().analyze(&tap, 0.016);
        assert!(fast.high_mid > 0.4 && fast.low_mid < 0.2, "{fast:?}");
    }

    #[test]
    fn disabled_tap_costs_one_flag_read() {
        let tap = AudioTap::default();
        assert!(!tap.is_enabled());
        let v = tap.register(48_000);
        assert!(!v.on());
    }

    #[test]
    fn probing_is_throttled() {
        let mut r = AudioReactive::default();
        assert!(r.should_probe());
        let hits = (0..PROBE_EVERY * 3).filter(|_| r.should_probe()).count();
        assert_eq!(hits, 3);
    }

    #[cfg(feature = "client")]
    #[test]
    fn sine_through_a_rodio_mixer_reads_nonzero_bands() {
        use rodio::Source;
        // A device-free rodio graph: a Sink feeding a DynamicMixer, pulled by hand.
        let (controller, mut mixer) = rodio::dynamic_mixer::mixer::<f32>(2, 48_000);
        let tap = AudioTap::default();
        tap.set_enabled(true);
        let (sink, queue) = rodio::Sink::new_idle();
        controller.add(queue);
        let vt = tap.register(48_000);
        sink.set_volume(0.5);
        sink.append(Tapped::new(rodio::source::SineWave::new(100.0).amplify(1.0), vt.clone()).convert_samples::<f32>().take_duration(std::time::Duration::from_secs(1)));
        // The engine copies the sink's volume into the tap before each read.
        vt.set_gain(sink.volume());
        let pulled: Vec<f32> = mixer.by_ref().take(48_000 * 2 / 10).collect();
        assert!(pulled.iter().any(|s| s.abs() > 0.1), "the mixer produced sound");
        let b = AudioReactive::default().analyze(&tap, 0.016);
        assert!(b.bass > 0.2 && b.volume > 0.15, "{b:?}");
        assert!(b.treble < 0.05 && b.high_mid < 0.05, "{b:?}");
    }
}
