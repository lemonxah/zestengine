//! Voice chat: mic capture, voice activity / push-to-talk, Opus (or PCM)
//! frames over the net transport, and per-peer playback through an adaptive
//! jitter buffer into positioned (stereo) sinks with proximity falloff.
//!
//! Lag and jitter: each peer's buffer measures arrival jitter (RFC 3550 style,
//! against the 20 ms send cadence) and holds just enough frames to ride it out
//! (40 to 300 ms). A missing frame is rebuilt from the next packet's forward
//! error correction, else concealed by the codec; a long gap re-buffers. When
//! the buffer runs long it drops frames during quiet stretches to win the
//! latency back.
//!
//! Every played frame (and the local mic while transmitting) also runs
//! through the viseme analyser, so lip sync matches what is heard.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use zestengine_core::{VisemeFrame, VoiceChat, VoiceTransmit};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use glam::Vec3;
use rodio::{OutputStream, OutputStreamHandle, Source, SpatialSink};

use crate::net::{NetSession, VoicePacket};
use crate::viseme::VisemeAnalyzer;
use crate::voice_codec::{Decoder, Encoder, FRAME, RATE};

const FRAME_SECS: f64 = FRAME as f64 / RATE as f64;
/// Bound on queued mic audio (~600 ms) so a stalled sender can't build lag.
const MAX_CAPTURE: usize = RATE as usize * 6 / 10;
/// Voice activity stays open this long after the level drops (s).
const VAD_HOLD: f32 = 0.35;
/// Frames concealed in a row before the buffer re-arms.
const MAX_CONCEAL: u32 = 5;

/// One peer's receive side: reorders packets and plays them out on a steady
/// 20 ms cadence. Shared between the frame loop (push) and the audio thread
/// (pull).
pub(crate) struct JitterBuffer {
    packets: BTreeMap<u32, Vec<u8>>,
    next: Option<u32>,
    playing: bool,
    decoder: Decoder,
    /// Smoothed arrival jitter (s) and the last transit sample.
    jitter: f64,
    last_transit: Option<f64>,
    concealed: u32,
    analyzer: VisemeAnalyzer,
    pub visemes: VisemeFrame,
    /// Time base for arrival stamps.
    clock: Instant,
}

impl Default for JitterBuffer {
    fn default() -> Self {
        Self {
            packets: BTreeMap::new(),
            next: None,
            playing: false,
            decoder: Decoder::default(),
            jitter: 0.0,
            last_transit: None,
            concealed: 0,
            analyzer: VisemeAnalyzer::default(),
            visemes: VisemeFrame::default(),
            clock: Instant::now(),
        }
    }
}

impl JitterBuffer {
    /// Frames to hold before playing, from the measured jitter.
    pub fn target_frames(&self) -> usize {
        (((FRAME_SECS + 3.0 * self.jitter) / FRAME_SECS).ceil() as usize).clamp(2, 15)
    }

    pub fn push(&mut self, seq: u32, payload: Vec<u8>) {
        self.push_at(seq, payload, self.clock.elapsed().as_secs_f64());
    }

    fn push_at(&mut self, seq: u32, payload: Vec<u8>, now: f64) {
        if let Some(next) = self.next
            && seq.wrapping_sub(next) > u32::MAX / 2
        {
            return; // late: its slot already played
        }
        let transit = now - seq as f64 * FRAME_SECS;
        if let Some(last) = self.last_transit {
            let d = (transit - last).abs();
            self.jitter += (d - self.jitter) / 16.0;
        }
        self.last_transit = Some(transit);
        self.packets.insert(seq, payload);
        // A sender that restarted (seq jumped far ahead) resets the playout.
        if let Some(next) = self.next
            && seq.wrapping_sub(next) > 500
            && seq.wrapping_sub(next) < u32::MAX / 2
        {
            self.packets.retain(|s, _| *s == seq);
            self.next = None;
            self.playing = false;
        }
    }

    /// The next 20 ms of audio to play.
    pub fn pull(&mut self) -> Vec<i16> {
        if !self.playing {
            if self.packets.len() < self.target_frames() {
                self.visemes = VisemeFrame::default();
                return vec![0; FRAME];
            }
            self.playing = true;
            self.next = self.packets.keys().next().copied();
        }
        let Some(next) = self.next else {
            self.playing = false;
            return vec![0; FRAME];
        };
        let pcm = match self.packets.remove(&next) {
            Some(p) => {
                self.concealed = 0;
                self.decoder.decode(&p).unwrap_or_else(|| vec![0; FRAME])
            }
            None if self.packets.is_empty() => {
                // Underrun: stop and buffer up again.
                self.playing = false;
                self.next = None;
                self.visemes = VisemeFrame::default();
                return self.decoder.conceal();
            }
            None => {
                self.concealed += 1;
                let next_payload = self.packets.get(&next.wrapping_add(1)).cloned();
                let pcm = next_payload.and_then(|p| self.decoder.recover(&p)).unwrap_or_else(|| self.decoder.conceal());
                if self.concealed > MAX_CONCEAL {
                    // A long gap: resume from the oldest packet we have.
                    self.next = self.packets.keys().next().map(|s| s.wrapping_sub(1));
                    self.concealed = 0;
                }
                pcm
            }
        };
        self.next = self.next.map(|n| n.wrapping_add(1));
        // Too much queued: skip a quiet frame to win the latency back.
        if self.packets.len() > self.target_frames() + 3 && rms_db(&pcm) < -45.0
            && let Some(n) = self.next {
                self.packets.remove(&n);
                self.next = Some(n.wrapping_add(1));
            }
        while self.packets.len() > 50 {
            let oldest = *self.packets.keys().next().unwrap();
            self.packets.remove(&oldest);
        }
        self.visemes = self.analyzer.analyze(&pcm);
        pcm
    }
}

fn rms_db(pcm: &[i16]) -> f32 {
    let rms = (pcm.iter().map(|s| (*s as f32).powi(2)).sum::<f32>() / pcm.len().max(1) as f32).sqrt();
    20.0 * (rms / 32768.0).max(1e-6).log10()
}

/// A never-ending rodio source pulling 20 ms frames from a jitter buffer.
struct JitterSource {
    buf: Arc<Mutex<JitterBuffer>>,
    frame: Vec<i16>,
    at: usize,
}

impl Iterator for JitterSource {
    type Item = f32;
    fn next(&mut self) -> Option<f32> {
        if self.at >= self.frame.len() {
            self.frame = self.buf.lock().map(|mut b| b.pull()).unwrap_or_else(|_| vec![0; FRAME]);
            self.at = 0;
        }
        let s = self.frame[self.at];
        self.at += 1;
        Some(s as f32 / 32768.0)
    }
}

impl Source for JitterSource {
    fn current_frame_len(&self) -> Option<usize> {
        None
    }
    fn channels(&self) -> u16 {
        1
    }
    fn sample_rate(&self) -> u32 {
        RATE as u32
    }
    fn total_duration(&self) -> Option<std::time::Duration> {
        None
    }
}

struct PeerVoice {
    sink: SpatialSink,
    buf: Arc<Mutex<JitterBuffer>>,
}

/// Mic capture + every peer's playback. Created when a scene with a Voice
/// Chat component goes online.
pub struct VoiceSystem {
    _in_stream: Option<cpal::Stream>,
    capture: Arc<Mutex<VecDeque<i16>>>,
    _out_stream: OutputStream,
    out_handle: OutputStreamHandle,
    encoder: Encoder,
    peers: HashMap<u64, PeerVoice>,
    seq: u32,
    vad_open: f32,
    analyzer: VisemeAnalyzer,
    local_visemes: VisemeFrame,
    /// The local player is being heard.
    pub transmitting: bool,
}

impl VoiceSystem {
    /// Audio output + mic capture. None without an output device; no mic just
    /// means listen-only.
    pub fn new(settings: &VoiceChat) -> Option<Self> {
        let (stream, handle) = OutputStream::try_default().map_err(|e| tracing::warn!("voice: no audio output: {e}")).ok()?;
        let capture = Arc::new(Mutex::new(VecDeque::new()));
        let in_stream = start_capture(capture.clone());
        Some(Self {
            _in_stream: in_stream,
            capture,
            _out_stream: stream,
            out_handle: handle,
            encoder: Encoder::new(settings.bitrate),
            peers: HashMap::new(),
            seq: 0,
            vad_open: 0.0,
            analyzer: VisemeAnalyzer::default(),
            local_visemes: VisemeFrame::default(),
            transmitting: false,
        })
    }

    /// Encode and send the captured mic audio according to the transmit mode.
    /// `ptt` is whether the push-to-talk action is held.
    pub fn capture_and_send(&mut self, net: &mut NetSession, settings: &VoiceChat, ptt: bool, muted: bool) {
        for (seq, payload) in self.capture_encoded(settings, ptt, muted) {
            net.send_voice(seq, &payload);
        }
    }

    /// Drain the mic into encoded `(seq, payload)` frames that pass the
    /// transmit gate (shared by play-mode and co-edit capture). The sequence
    /// advances with time either way, so receivers see silence as a gap.
    pub fn capture_encoded(&mut self, settings: &VoiceChat, ptt: bool, muted: bool) -> Vec<(u32, Vec<u8>)> {
        let frames: Vec<Vec<i16>> = {
            let mut cap = self.capture.lock().unwrap();
            let mut v = Vec::new();
            while cap.len() >= FRAME {
                v.push(cap.drain(..FRAME).collect());
            }
            v
        };
        let mut out = Vec::new();
        for mut frame in frames {
            if settings.mic_gain != 1.0 {
                for s in &mut frame {
                    *s = (*s as f32 * settings.mic_gain).clamp(-32768.0, 32767.0) as i16;
                }
            }
            let level = rms_db(&frame);
            if level > settings.activation_db {
                self.vad_open = VAD_HOLD;
            } else {
                self.vad_open -= FRAME_SECS as f32;
            }
            let send = !muted
                && settings.enabled
                && match settings.transmit {
                    VoiceTransmit::PushToTalk => ptt,
                    VoiceTransmit::AlwaysOn => true,
                    VoiceTransmit::VoiceActivity => self.vad_open > 0.0,
                };
            if send {
                let payload = self.encoder.encode(&frame);
                self.local_visemes = self.analyzer.analyze(&frame);
                out.push((self.seq, payload));
            }
            // The sequence advances with time either way, so receivers see a
            // gap (silence) rather than a burst when speech resumes.
            self.seq = self.seq.wrapping_add(1);
        }
        self.transmitting = !out.is_empty() || (self.transmitting && self.vad_open > 0.0 && settings.transmit == VoiceTransmit::VoiceActivity);
        if !self.transmitting {
            self.local_visemes = VisemeFrame::default();
        }
        out
    }

    /// Queue received packets into their peers' jitter buffers.
    pub fn receive(&mut self, packets: Vec<VoicePacket>) {
        for p in packets {
            let handle = &self.out_handle;
            let peer = self.peers.entry(p.from).or_insert_with(|| {
                let buf = Arc::new(Mutex::new(JitterBuffer::default()));
                let sink = SpatialSink::try_new(handle, [0.0; 3], [-0.1, 0.0, 0.0], [0.1, 0.0, 0.0])
                    .unwrap_or_else(|_| SpatialSink::try_new(handle, [0.0; 3], [-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]).expect("spatial sink"));
                sink.append(JitterSource { buf: buf.clone(), frame: Vec::new(), at: 0 });
                PeerVoice { sink, buf }
            });
            if let Ok(mut b) = peer.buf.lock() {
                b.push(p.seq, p.payload);
            }
        }
    }

    /// Position every peer's voice (their avatar's head) relative to the
    /// listener and apply the proximity falloff.
    pub fn update(&mut self, settings: &VoiceChat, listener: Vec3, listener_right: Vec3, speakers: &HashMap<u64, Vec3>) {
        let ear = listener_right.normalize_or(Vec3::X) * 0.1;
        for (peer, v) in &self.peers {
            let at = speakers.get(peer).copied();
            let gain = match at {
                Some(p) => settings.gain(p.distance(listener)),
                None => 1.0,
            };
            let emitter = at.unwrap_or(listener);
            v.sink.set_emitter_position(emitter.to_array());
            v.sink.set_left_ear_position((listener - ear).to_array());
            v.sink.set_right_ear_position((listener + ear).to_array());
            v.sink.set_volume(if settings.enabled { gain } else { 0.0 });
        }
        self.peers.retain(|p, _| speakers.contains_key(p) || speakers.is_empty());
    }

    /// What each remote peer's mouth is doing right now (as heard).
    pub fn peer_visemes(&self) -> HashMap<u64, VisemeFrame> {
        self.peers.iter().filter_map(|(p, v)| v.buf.lock().ok().map(|b| (*p, b.visemes))).collect()
    }

    pub fn local_visemes(&self) -> VisemeFrame {
        self.local_visemes
    }
}

/// One frame of voice for a networked scene: starts with the scene's Voice
/// Chat (and stops without one), sends the mic, plays and positions every
/// peer, and hands lip sync its visemes. `ptt` reports an input action as held.
pub fn run_frame(
    voice: &mut Option<VoiceSystem>,
    net: Option<&mut NetSession>,
    scene: &mut crate::scene::LoadedScene,
    ptt: impl Fn(&str) -> bool,
    listener: Vec3,
    listener_right: Vec3,
) {
    let (Some(net), Some(settings)) = (net, scene.voice_settings().filter(|s| s.enabled)) else {
        *voice = None;
        scene.set_visemes(None, &HashMap::new(), &[]);
        return;
    };
    if voice.is_none() {
        *voice = VoiceSystem::new(&settings);
    }
    let Some(v) = voice.as_mut() else { return };
    v.capture_and_send(net, &settings, ptt(&settings.push_to_talk_action), false);
    let owners = net.owners_snapshot();
    v.receive(net.take_voice());
    let speakers = scene.peer_voice_positions(&owners);
    v.update(&settings, listener, listener_right, &speakers);
    let local = v.transmitting.then(|| v.local_visemes());
    scene.set_visemes(local, &v.peer_visemes(), &owners);
}

/// Open the default mic and stream mono 16 kHz samples into `capture`.
fn start_capture(capture: Arc<Mutex<VecDeque<i16>>>) -> Option<cpal::Stream> {
    let host = cpal::default_host();
    let device = host.default_input_device()?;
    let config = device.default_input_config().ok()?;
    let src_rate = config.sample_rate().0;
    let channels = config.channels() as usize;
    let err = |e| tracing::warn!("voice capture error: {e}");
    let cap = capture.clone();
    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device
            .build_input_stream(&config.into(), move |data: &[f32], _: &_| push_mono(&cap, data, channels, src_rate), err, None)
            .ok()?,
        cpal::SampleFormat::I16 => device
            .build_input_stream(
                &config.into(),
                move |data: &[i16], _: &_| {
                    let f: Vec<f32> = data.iter().map(|s| *s as f32 / 32768.0).collect();
                    push_mono(&cap, &f, channels, src_rate);
                },
                err,
                None,
            )
            .ok()?,
        _ => return None,
    };
    if let Err(e) = stream.play() {
        tracing::warn!("voice: mic start failed: {e}");
        return None;
    }
    tracing::info!("voice: mic capture started ({src_rate} Hz, {channels} ch)");
    Some(stream)
}

/// Downmix to mono and resample `src_rate` -> 16 kHz (linear), pushing i16.
fn push_mono(capture: &Arc<Mutex<VecDeque<i16>>>, data: &[f32], channels: usize, src_rate: u32) {
    if channels == 0 {
        return;
    }
    let step = src_rate as f32 / RATE as f32;
    let frames = data.len() / channels;
    let mut buf = capture.lock().unwrap();
    let mut pos = 0.0f32;
    while (pos as usize) < frames {
        let i = pos as usize;
        let i1 = (i + 1).min(frames - 1);
        let t = pos - i as f32;
        let mut sum = 0.0;
        for c in 0..channels {
            sum += data[i * channels + c] * (1.0 - t) + data[i1 * channels + c] * t;
        }
        let mono = (sum / channels as f32).clamp(-1.0, 1.0);
        buf.push_back((mono * 32767.0) as i16);
        pos += step;
    }
    while buf.len() > MAX_CAPTURE {
        buf.pop_front();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn speech(seq: u32) -> Vec<u8> {
        let mut enc = Encoder::new(24_000);
        let pcm: Vec<i16> = (0..FRAME).map(|i| (((seq as usize * FRAME + i) as f32 * 0.15).sin() * 6000.0) as i16).collect();
        enc.encode(&pcm)
    }

    #[test]
    fn steady_packets_play_after_a_short_buffer() {
        let mut jb = JitterBuffer::default();
        for s in 0..4u32 {
            jb.push_at(s, speech(s), s as f64 * FRAME_SECS);
        }
        assert_eq!(jb.target_frames(), 2, "no jitter: the minimum hold");
        let played: Vec<Vec<i16>> = (0..4).map(|_| jb.pull()).collect();
        assert!(played.iter().all(|f| rms_db(f) > -30.0), "all four frames play");
        assert_eq!(jb.pull().len(), FRAME, "an underrun plays concealment, not a click");
        assert!(!jb.playing, "and re-buffers");
    }

    #[test]
    fn jitter_grows_the_buffer_and_reordering_is_absorbed() {
        let mut jb = JitterBuffer::default();
        let mut t = 0.0;
        for s in 0..40u32 {
            t += FRAME_SECS + if s % 2 == 0 { 0.03 } else { -0.03 };
            jb.push_at(s, speech(s), t);
        }
        assert!(jb.target_frames() >= 4, "jittery link holds more: {}", jb.target_frames());
        let mut jb = JitterBuffer::default();
        for s in [1u32, 0, 3, 2, 5, 4] {
            jb.push_at(s, speech(s), s as f64 * FRAME_SECS);
        }
        jb.pull();
        assert_eq!(jb.next, Some(1), "plays in sequence order");
    }

    #[test]
    fn a_lost_packet_is_concealed_not_skipped() {
        let mut jb = JitterBuffer::default();
        for s in [0u32, 1, 3, 4] {
            jb.push_at(s, speech(s), s as f64 * FRAME_SECS);
        }
        for _ in 0..3 {
            jb.pull();
        }
        assert_eq!(jb.next, Some(3), "frame 2 was filled in");
        assert!(jb.packets.contains_key(&3));
    }

    #[test]
    fn late_packets_are_dropped() {
        let mut jb = JitterBuffer::default();
        for s in 0..3u32 {
            jb.push_at(s, speech(s), s as f64 * FRAME_SECS);
        }
        jb.pull();
        jb.pull();
        jb.push_at(0, speech(0), 0.1);
        assert!(!jb.packets.contains_key(&0));
    }
}
