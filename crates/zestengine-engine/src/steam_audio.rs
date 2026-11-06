//! Steam Audio spatialization (Valve's spatial audio SDK, Apache-2.0) through
//! the `audionimbus` bindings to Valve's `libphonon`.
//!
//! Each spatial or multichannel sound is a [`SpatialVoice`]: a rodio `Source`
//! that renders 1024-sample frames at 48 kHz into the project's output layout.
//! Mono sources go through a direct effect (air
//! absorption, directivity, occlusion / transmission) then HRTF binaural,
//! speaker panning, or ambisonic encode + decode. Reflections (own or the
//! listener-centric reverb) and pathing render to world-space ambisonics and
//! are decoded with the listener's orientation. Ambisonic clips are rotated by
//! the listener and decoded binaurally (or panned to speakers); 5.1 / 7.1 beds
//! play through virtual surround on headphones. The authored distance rolloff
//! applies to the complete positioned voice, including reflections and paths.
//!
//! The game thread computes each voice's [`VoiceState`] per frame and hands it
//! over through a shared slot. [`Simulation`] is a worker thread that owns the
//! Steam Audio scene (static meshes with acoustic materials, instanced moving
//! occluders), the probe batch of the scene's audio bake, and the simulator:
//! direct occlusion runs ~30 times a second, reflections and pathing at the
//! project's update rate; their outputs land in per-voice slots.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use audionimbus::{
    AirAbsorptionModel, AmbisonicsBinauralEffect, AmbisonicsBinauralEffectParams, AmbisonicsBinauralEffectSettings,
    AmbisonicsDecodeEffect, AmbisonicsDecodeEffectParams, AmbisonicsDecodeEffectSettings, AmbisonicsEncodeEffect,
    AmbisonicsEncodeEffectParams, AmbisonicsEncodeEffectSettings, AmbisonicsPanningEffect,
    AmbisonicsPanningEffectParams, AmbisonicsPanningEffectSettings, AmbisonicsRotationEffect,
    AmbisonicsRotationEffectParams, AmbisonicsRotationEffectSettings, AudioBufferMut, AudioBufferRef, AudioSettings,
    BakedDataIdentifier, BakedDataVariation, BinauralEffect, BinauralEffectParams, BinauralEffectSettings, Context,
    ContextSettings, Convolution, CoordinateSystem, DirectEffect, DirectEffectParams, DirectEffectSettings,
    Directivity, Equalizer, Hrtf, HrtfInterpolation, HrtfSettings, Hybrid, PanningEffect, PanningEffectParams,
    PanningEffectSettings, Parametric, PathEffect, PathEffectParams, PathEffectSettings, ReflectionEffect,
    ReflectionEffectParams, ReflectionEffectSettings, Rendering, Sofa, SpeakerLayout, Transmission, Vector3,
    VirtualSurroundEffect, VirtualSurroundEffectParams, VirtualSurroundEffectSettings, num_ambisonics_channels,
};
use zestengine_core::{ObjectId, OutputLayout, RayTracerKind, ReflectionMode, SpatialAudioSettings};

/// Simulated voices and which of their direct / reflection / pathing results
/// are in use.
type SimSources<R> = HashMap<VoiceKey, (SimSource<R>, (bool, bool, bool))>;
use glam::{Mat4, Vec3};
use rodio::cpal::FromSample;
use rodio::source::UniformSourceIterator;
use rodio::{Sample, Source};

pub const SAMPLE_RATE: u32 = 48_000;
pub const FRAME: usize = 1024;

/// Everything the audio thread needs to render one frame of a voice.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoiceState {
    /// Listener-space unit vector towards the source (-Z ahead, +X right).
    pub direction: [f32; 3],
    /// Distance attenuation, 0..1.
    pub attenuation: f32,
    /// Live Audio Volume reflection level; does not affect direct sound or pathing.
    pub reverb_gain: f32,
    /// Per-band (low / mid / high) air absorption gains.
    pub air: [f32; 3],
    /// Directivity gain, 0..1.
    pub directivity: f32,
    /// 1 = unobstructed, 0 = fully blocked.
    pub occlusion: f32,
    /// Per-band transmission through blocking geometry.
    pub transmission: [f32; 3],
    /// Binaural (true) or plain panning.
    pub hrtf: bool,
    /// Listener right / up / ahead / position (decodes world-space ambisonics).
    pub listener: [[f32; 3]; 4],
}

impl Default for VoiceState {
    fn default() -> Self {
        Self {
            direction: [0.0, 0.0, -1.0],
            attenuation: 1.0,
            reverb_gain: 1.0,
            air: [1.0; 3],
            directivity: 1.0,
            occlusion: 1.0,
            transmission: [1.0; 3],
            hrtf: true,
            listener: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [0.0; 3]],
        }
    }
}

/// Shared parameter slot: written by the game thread, read per audio frame.
pub type VoiceSlot = Arc<Mutex<VoiceState>>;

/// Latest reflection / reverb parameters from the simulation (taken by the
/// audio thread).
pub type ReflSlot = Arc<Mutex<Option<ReflParams>>>;
/// Latest pathing parameters from the simulation.
pub type PathSlot = Arc<Mutex<Option<PathEffectParams>>>;

/// Reflection effect parameters of the project's reflection type.
pub enum ReflParams {
    Convolution(ReflectionEffectParams<Convolution>),
    Parametric(ReflectionEffectParams<Parametric>),
    Hybrid(ReflectionEffectParams<Hybrid>),
}

enum ReflEffect {
    Convolution(ReflectionEffect<Convolution>),
    Parametric(ReflectionEffect<Parametric>),
    Hybrid(ReflectionEffect<Hybrid>),
}

impl ReflEffect {
    fn new(mode: ReflectionMode, context: &Context, audio: &AudioSettings, settings: &ReflectionEffectSettings) -> Option<Self> {
        Some(match mode {
            ReflectionMode::Off => return None,
            ReflectionMode::Convolution => Self::Convolution(ReflectionEffect::try_new(context, audio, settings).ok()?),
            ReflectionMode::Parametric => Self::Parametric(ReflectionEffect::try_new(context, audio, settings).ok()?),
            ReflectionMode::Hybrid => Self::Hybrid(ReflectionEffect::try_new(context, audio, settings).ok()?),
        })
    }

    fn apply(&mut self, params: &ReflParams, input: &AudioBufferRef, out: &mut AudioBufferMut) -> bool {
        match (self, params) {
            (Self::Convolution(e), ReflParams::Convolution(p)) => e.apply(p, input, out).is_ok(),
            (Self::Parametric(e), ReflParams::Parametric(p)) => e.apply(p, input, out).is_ok(),
            (Self::Hybrid(e), ReflParams::Hybrid(p)) => e.apply(p, input, out).is_ok(),
            _ => false,
        }
    }
}

/// What the game thread knows about an emitter this frame.
#[derive(Clone, Copy, Debug)]
pub struct Emitter {
    pub transform: Mat4,
    /// Distance attenuation from the source's rolloff settings.
    pub attenuation: f32,
    pub air_absorption: bool,
    pub directivity: f32,
    pub directivity_sharpness: f32,
    pub hrtf: bool,
}

/// Per-voice rendering choices.
#[derive(Clone, Default)]
pub struct VoiceOptions {
    /// A positioned (spatial) mono source.
    pub spatial: bool,
    /// Ambisonic clip order (0 = not ambisonic).
    pub ambisonic_order: u32,
    /// Encode to ambisonics and decode instead of direct HRTF / panning.
    pub via_ambisonics: bool,
    /// Reflections / reverb parameters for this voice.
    pub reflections: Option<ReflSlot>,
    /// Pathing parameters for this voice.
    pub pathing: Option<PathSlot>,
}

/// The Steam Audio context, HRTF and project settings, shared by every voice.
pub struct SteamAudio {
    context: Context,
    hrtf: Hrtf,
    settings: AudioSettings,
    config: SpatialAudioSettings,
}

fn v3(v: Vec3) -> Vector3 {
    Vector3::new(v.x, v.y, v.z)
}

fn speaker_layout(layout: OutputLayout) -> SpeakerLayout {
    match layout {
        OutputLayout::Headphones | OutputLayout::Stereo => SpeakerLayout::Stereo,
        OutputLayout::Quad => SpeakerLayout::Quadraphonic,
        OutputLayout::Surround51 => SpeakerLayout::Surround5_1,
        OutputLayout::Surround71 => SpeakerLayout::Surround7_1,
    }
}

/// Speaker layout of a multichannel bed, by channel count.
fn bed_layout(channels: u16) -> Option<SpeakerLayout> {
    match channels {
        4 => Some(SpeakerLayout::Quadraphonic),
        6 => Some(SpeakerLayout::Surround5_1),
        8 => Some(SpeakerLayout::Surround7_1),
        _ => None,
    }
}

impl SteamAudio {
    /// Default settings, default HRTF.
    pub fn new() -> Result<Self, String> {
        Self::with_settings(&SpatialAudioSettings::default(), Path::new("."))
    }

    /// Context + HRTF for `config`; a `.sofa` HRTF that fails to load falls
    /// back to Steam Audio's default with a warning.
    pub fn with_settings(config: &SpatialAudioSettings, project_root: &Path) -> Result<Self, String> {
        let config = config.sanitized();
        let context = Context::try_new(&ContextSettings::default()).map_err(|e| format!("context: {e}"))?;
        let settings = AudioSettings { sampling_rate: SAMPLE_RATE, frame_size: FRAME as u32 };
        let default_hrtf = || Hrtf::try_new(&context, &settings, &HrtfSettings::default()).map_err(|e| format!("HRTF: {e}"));
        let hrtf = if config.hrtf_sofa.trim().is_empty() {
            default_hrtf()?
        } else {
            let path = project_root.join(config.hrtf_sofa.trim());
            let sofa = HrtfSettings {
                sofa_information: Some(Sofa::Filename(path.to_string_lossy().into_owned())),
                ..HrtfSettings::default()
            };
            match Hrtf::try_new(&context, &settings, &sofa) {
                Ok(h) => h,
                Err(e) => {
                    tracing::warn!("SOFA HRTF {}: {e}; using the default HRTF", path.display());
                    default_hrtf()?
                }
            }
        };
        Ok(Self { context, hrtf, settings, config })
    }

    pub fn config(&self) -> &SpatialAudioSettings {
        &self.config
    }

    /// Whether a clip with `channels` needs rendering here (beyond plain rodio).
    pub fn wants(&self, opts: &VoiceOptions, channels: u16) -> bool {
        opts.spatial
            || opts.ambisonic_order > 0
            || (self.config.output == OutputLayout::Headphones && bed_layout(channels).is_some())
    }

    /// Wrap a decoded clip in a voice driven by `slot`.
    pub fn voice<S>(&self, source: S, slot: VoiceSlot, opts: VoiceOptions) -> Result<SpatialVoice<S>, String>
    where
        S: Source,
        S::Item: Sample,
        f32: FromSample<S::Item>,
    {
        let ctx = &self.context;
        let audio = &self.settings;
        let out_layout = self.config.output;
        let out_channels = out_layout.channels() as usize;
        let channels = source.channels().max(1);
        let order = self.config.order;
        let amb_channels = num_ambisonics_channels(order) as usize;
        let decode_settings = |max_order: u32| AmbisonicsDecodeEffectSettings {
            speaker_layout: speaker_layout(out_layout),
            hrtf: self.hrtf.clone(),
            max_order,
            rendering: if out_layout == OutputLayout::Headphones { Rendering::Binaural } else { Rendering::Panning },
        };

        let mode = if opts.ambisonic_order > 0 {
            let clip_order = (opts.ambisonic_order.min(3) as usize).min(((channels as f32).sqrt() as usize).saturating_sub(1)).max(1) as u32;
            let rotation = AmbisonicsRotationEffect::try_new(ctx, audio, &AmbisonicsRotationEffectSettings { max_order: clip_order })
                .map_err(|e| format!("ambisonics rotation: {e}"))?;
            let render = if out_layout == OutputLayout::Headphones {
                AmbiRender::Binaural(
                    AmbisonicsBinauralEffect::try_new(
                        ctx,
                        audio,
                        &AmbisonicsBinauralEffectSettings { hrtf: self.hrtf.clone(), max_order: clip_order },
                    )
                    .map_err(|e| format!("ambisonics binaural: {e}"))?,
                )
            } else {
                AmbiRender::Panning(
                    AmbisonicsPanningEffect::try_new(
                        ctx,
                        audio,
                        &AmbisonicsPanningEffectSettings { speaker_layout: speaker_layout(out_layout), max_order: clip_order },
                    )
                    .map_err(|e| format!("ambisonics panning: {e}"))?,
                )
            };
            let n = num_ambisonics_channels(clip_order) as usize;
            Mode::Ambisonic { order: clip_order, rotation, render, input: vec![0.0; FRAME * n], rotated: vec![0.0; FRAME * n] }
        } else if !opts.spatial && let Some(layout) = bed_layout(channels) {
            let surround = if out_layout == OutputLayout::Headphones {
                Some(
                    VirtualSurroundEffect::try_new(ctx, audio, &VirtualSurroundEffectSettings { speaker_layout: layout, hrtf: self.hrtf.clone() })
                        .map_err(|e| format!("virtual surround: {e}"))?,
                )
            } else {
                None
            };
            Mode::Bed { surround, input: vec![0.0; FRAME * channels as usize] }
        } else {
            let direct = DirectEffect::try_new(ctx, audio, &DirectEffectSettings { num_channels: 1 })
                .map_err(|e| format!("direct effect: {e}"))?;
            let binaural = BinauralEffect::try_new(ctx, audio, &BinauralEffectSettings { hrtf: self.hrtf.clone() })
                .map_err(|e| format!("binaural effect: {e}"))?;
            let panning = PanningEffect::try_new(ctx, audio, &PanningEffectSettings { speaker_layout: speaker_layout(out_layout) })
                .map_err(|e| format!("panning effect: {e}"))?;
            let encode = if opts.via_ambisonics {
                let enc = AmbisonicsEncodeEffect::try_new(ctx, audio, &AmbisonicsEncodeEffectSettings { max_order: order.max(1) })
                    .map_err(|e| format!("ambisonics encode: {e}"))?;
                let dec = AmbisonicsDecodeEffect::try_new(ctx, audio, &decode_settings(order.max(1)))
                    .map_err(|e| format!("ambisonics decode: {e}"))?;
                Some((enc, dec, vec![0.0; FRAME * num_ambisonics_channels(order.max(1)) as usize]))
            } else {
                None
            };
            Mode::Mono { direct, binaural, panning, encode, direct_out: vec![0.0; FRAME] }
        };

        let mono_input = matches!(mode, Mode::Mono { .. });
        let reflections = if mono_input && self.config.reflections != ReflectionMode::Off {
            opts.reflections.and_then(|slot| {
                let settings = ReflectionEffectSettings {
                    impulse_response_size: (self.config.duration * SAMPLE_RATE as f32) as u32,
                    num_channels: amb_channels as u32,
                };
                Some(VoiceReflections { effect: ReflEffect::new(self.config.reflections, ctx, audio, &settings)?, slot, current: None })
            })
        } else {
            None
        };
        let pathing = if mono_input {
            opts.pathing.and_then(|slot| {
                let effect = PathEffect::try_new(ctx, audio, &PathEffectSettings { max_order: order, spatialization: None }).ok()?;
                Some(VoicePathing { effect, slot, current: None })
            })
        } else {
            None
        };
        let world_decode = if reflections.is_some() || pathing.is_some() {
            Some(AmbisonicsDecodeEffect::try_new(ctx, audio, &decode_settings(order)).map_err(|e| format!("ambisonics decode: {e}"))?)
        } else {
            None
        };

        let state = *slot.lock().unwrap();
        Ok(SpatialVoice {
            input: UniformSourceIterator::new(source, channels, SAMPLE_RATE),
            channels,
            out_channels,
            order,
            slot,
            state,
            distance_gain: state.attenuation,
            mode,
            reflections,
            pathing,
            world_decode,
            hrtf: self.hrtf.clone(),
            headphones: out_layout == OutputLayout::Headphones,
            mono: vec![0.0; FRAME],
            amb: vec![0.0; FRAME * amb_channels],
            amb_tmp: vec![0.0; FRAME * amb_channels],
            spk: vec![0.0; FRAME * out_channels],
            spk_tmp: vec![0.0; FRAME * out_channels],
            out: Vec::with_capacity(FRAME * out_channels),
            pos: 0,
            done: false,
        })
    }

    /// The voice state for an emitter heard by a listener (world transforms;
    /// both look down their local -Z). Occlusion / transmission are left at
    /// "unobstructed" for the simulation to fill in.
    pub fn state_for(&self, emitter: &Emitter, listener: Mat4) -> VoiceState {
        let src = emitter.transform.w_axis.truncate();
        let ear = listener.w_axis.truncate();
        let ahead = -listener.z_axis.truncate().normalize_or(Vec3::Z);
        let up = listener.y_axis.truncate().normalize_or(Vec3::Y);
        let right = listener.x_axis.truncate().normalize_or(Vec3::X);
        let direction = if src.distance_squared(ear) < 1e-8 {
            [0.0, 0.0, -1.0]
        } else {
            let d = audionimbus::relative_direction(&self.context, v3(src), v3(ear), v3(ahead), v3(up));
            [d.x, d.y, d.z]
        };
        let air = if emitter.air_absorption {
            // SAFETY: only the callback air-absorption model is unsound.
            let eq = unsafe { audionimbus::air_absorption(&self.context, v3(src), v3(ear), &AirAbsorptionModel::Default) };
            eq.0
        } else {
            [1.0; 3]
        };
        let directivity = if emitter.directivity > 0.0 {
            let pattern = Directivity::WeightedDipole {
                weight: emitter.directivity.clamp(0.0, 1.0),
                power: emitter.directivity_sharpness.max(0.0),
            };
            audionimbus::directivity_attenuation(&self.context, coords(emitter.transform), v3(ear), &pattern)
        } else {
            1.0
        };
        VoiceState {
            direction,
            attenuation: emitter.attenuation.clamp(0.0, 1.0),
            reverb_gain: 1.0,
            air,
            directivity,
            occlusion: 1.0,
            transmission: [1.0; 3],
            hrtf: emitter.hrtf,
            listener: [right.to_array(), up.to_array(), ahead.to_array(), ear.to_array()],
        }
    }
}

enum AmbiRender {
    Binaural(AmbisonicsBinauralEffect),
    Panning(AmbisonicsPanningEffect),
}

enum Mode {
    /// Positioned mono source.
    Mono {
        direct: DirectEffect,
        binaural: BinauralEffect,
        panning: PanningEffect,
        encode: Option<(AmbisonicsEncodeEffect, AmbisonicsDecodeEffect, Vec<f32>)>,
        direct_out: Vec<f32>,
    },
    /// Ambisonic clip: rotate by the listener, then decode.
    Ambisonic {
        order: u32,
        rotation: AmbisonicsRotationEffect,
        render: AmbiRender,
        input: Vec<f32>,
        rotated: Vec<f32>,
    },
    /// Speaker-layout bed (virtual surround on headphones).
    Bed { surround: Option<VirtualSurroundEffect>, input: Vec<f32> },
}

struct VoiceReflections {
    effect: ReflEffect,
    slot: ReflSlot,
    current: Option<ReflParams>,
}

struct VoicePathing {
    effect: PathEffect,
    slot: PathSlot,
    current: Option<PathEffectParams>,
}

/// A clip rendered through Steam Audio as interleaved audio at 48 kHz in the
/// project's output layout.
pub struct SpatialVoice<S>
where
    S: Source,
    S::Item: Sample,
    f32: FromSample<S::Item>,
{
    input: UniformSourceIterator<S, f32>,
    channels: u16,
    out_channels: usize,
    order: u32,
    slot: VoiceSlot,
    state: VoiceState,
    /// Previous output gain, ramped over a frame when the listener moves.
    distance_gain: f32,
    mode: Mode,
    reflections: Option<VoiceReflections>,
    pathing: Option<VoicePathing>,
    world_decode: Option<AmbisonicsDecodeEffect>,
    hrtf: Hrtf,
    headphones: bool,
    mono: Vec<f32>,
    /// World-space ambisonics (reflections + paths), planar.
    amb: Vec<f32>,
    amb_tmp: Vec<f32>,
    /// Planar output in the speaker layout.
    spk: Vec<f32>,
    spk_tmp: Vec<f32>,
    /// Interleaved output of the current frame.
    out: Vec<f32>,
    pos: usize,
    done: bool,
}

fn listener_coords(l: [[f32; 3]; 4]) -> CoordinateSystem {
    CoordinateSystem {
        right: Vector3::new(l[0][0], l[0][1], l[0][2]),
        up: Vector3::new(l[1][0], l[1][1], l[1][2]),
        ahead: Vector3::new(l[2][0], l[2][1], l[2][2]),
        origin: Vector3::new(l[3][0], l[3][1], l[3][2]),
    }
}

fn add_into(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d += *s;
    }
}

impl<S> SpatialVoice<S>
where
    S: Source,
    S::Item: Sample,
    f32: FromSample<S::Item>,
{
    /// Read up to one frame of interleaved input as planar channels into
    /// `planar` (or a mono downmix when `planar` is None). Returns frames read.
    fn read_frame(&mut self, planar: Option<&mut Vec<f32>>) -> usize {
        let ch = self.channels as usize;
        let mut n = 0;
        match planar {
            Some(buf) => {
                let lanes = buf.len() / FRAME;
                while n < FRAME {
                    let mut got = 0;
                    for c in 0..ch {
                        match self.input.next() {
                            Some(s) => {
                                if c < lanes {
                                    buf[c * FRAME + n] = s;
                                }
                                got += 1;
                            }
                            None => break,
                        }
                    }
                    if got == 0 {
                        break;
                    }
                    n += 1;
                }
                for c in 0..lanes {
                    buf[c * FRAME + n..(c + 1) * FRAME].fill(0.0);
                }
            }
            None => {
                while n < FRAME {
                    let mut sum = 0.0;
                    let mut got = 0;
                    for _ in 0..ch {
                        match self.input.next() {
                            Some(s) => {
                                sum += s;
                                got += 1;
                            }
                            None => break,
                        }
                    }
                    if got == 0 {
                        break;
                    }
                    self.mono[n] = sum / got as f32;
                    n += 1;
                }
                self.mono[n..].fill(0.0);
            }
        }
        n
    }

    /// Pull and render the next frame into `out`. False when the clip ended.
    fn render_frame(&mut self) -> bool {
        let mut mode = std::mem::replace(&mut self.mode, Mode::Bed { surround: None, input: Vec::new() });
        let n = match &mut mode {
            Mode::Mono { .. } => self.read_frame(None),
            Mode::Ambisonic { input, .. } | Mode::Bed { input, .. } => self.read_frame(Some(input)),
        };
        if n == 0 {
            self.mode = mode;
            return false;
        }
        if n < FRAME {
            self.done = true;
        }
        if let Ok(s) = self.slot.try_lock() {
            self.state = *s;
        }
        let st = self.state;
        let oc = self.out_channels;
        self.spk.fill(0.0);
        let listener = listener_coords(st.listener);

        let rendered = (|| -> Option<()> {
            match &mut mode {
                Mode::Mono { direct, binaural, panning, encode, direct_out } => {
                    let params = DirectEffectParams {
                        // Apply the authored rolloff once, after mixing indirect
                        // sound; otherwise reverb/pathing bypass max distance.
                        distance_attenuation: None,
                        air_absorption: Some(Equalizer(st.air)),
                        directivity: Some(st.directivity),
                        occlusion: Some(st.occlusion),
                        transmission: Some(Transmission::FrequencyDependent(Equalizer(st.transmission))),
                    };
                    let input = AudioBufferRef::try_from(&self.mono[..]).ok()?;
                    let mut out = AudioBufferMut::try_from(&mut direct_out[..]).ok()?;
                    direct.apply(&params, &input, &mut out).ok()?;
                    let dry = AudioBufferRef::try_from(&direct_out[..]).ok()?;
                    let direction = Vector3::new(st.direction[0], st.direction[1], st.direction[2]);
                    let mut spk = AudioBufferMut::try_new(&mut self.spk[..], oc).ok()?;
                    if let Some((enc, dec, amb)) = encode {
                        let order = self.order.max(1);
                        let mut amb_out = AudioBufferMut::try_new(&mut amb[..], num_ambisonics_channels(order) as usize).ok()?;
                        enc.apply(&AmbisonicsEncodeEffectParams { direction, order }, &dry, &mut amb_out).ok()?;
                        let amb_in = AudioBufferRef::try_new(&amb[..], num_ambisonics_channels(order) as usize).ok()?;
                        // Encoded in listener space: decode with an identity orientation
                        // (audionimbus' default frame looks down +Z, Steam Audio's -Z).
                        let identity = listener_coords(VoiceState::default().listener);
                        let params = AmbisonicsDecodeEffectParams { order, hrtf: self.hrtf.clone(), orientation: identity };
                        dec.apply(&params, &amb_in, &mut spk).ok()?;
                    } else if self.headphones && st.hrtf {
                        let params = BinauralEffectParams {
                            direction,
                            interpolation: HrtfInterpolation::Bilinear,
                            spatial_blend: 1.0,
                            hrtf: self.hrtf.clone(),
                            peak_delays: None,
                        };
                        binaural.apply(&params, &dry, &mut spk).ok()?;
                    } else {
                        panning.apply(&PanningEffectParams { direction }, &dry, &mut spk).ok()?;
                    }
                }
                Mode::Ambisonic { order, rotation, render, input, rotated } => {
                    let lanes = num_ambisonics_channels(*order) as usize;
                    let amb_in = AudioBufferRef::try_new(&input[..], lanes).ok()?;
                    let mut amb_out = AudioBufferMut::try_new(&mut rotated[..], lanes).ok()?;
                    rotation
                        .apply(&AmbisonicsRotationEffectParams { orientation: listener, order: *order }, &amb_in, &mut amb_out)
                        .ok()?;
                    let rot = AudioBufferRef::try_new(&rotated[..], lanes).ok()?;
                    let mut spk = AudioBufferMut::try_new(&mut self.spk[..], oc).ok()?;
                    match render {
                        AmbiRender::Binaural(e) => {
                            e.apply(&AmbisonicsBinauralEffectParams { hrtf: self.hrtf.clone(), order: *order }, &rot, &mut spk).ok()?;
                        }
                        AmbiRender::Panning(e) => {
                            e.apply(&AmbisonicsPanningEffectParams { order: *order }, &rot, &mut spk).ok()?;
                        }
                    }
                }
                Mode::Bed { surround, input } => {
                    let lanes = self.channels as usize;
                    match surround {
                        Some(vs) => {
                            let bed = AudioBufferRef::try_new(&input[..], lanes).ok()?;
                            let mut spk = AudioBufferMut::try_new(&mut self.spk[..], oc).ok()?;
                            vs.apply(&VirtualSurroundEffectParams { hrtf: self.hrtf.clone() }, &bed, &mut spk).ok()?;
                        }
                        None => {
                            for c in 0..lanes.min(oc) {
                                self.spk[c * FRAME..(c + 1) * FRAME].copy_from_slice(&input[c * FRAME..(c + 1) * FRAME]);
                            }
                        }
                    }
                }
            }
            Some(())
        })();
        if rendered.is_none() {
            self.spk.fill(0.0);
        }

        // Reflections / reverb and paths (world-space ambisonics), decoded
        // with the listener's orientation and mixed in.
        if matches!(mode, Mode::Mono { .. }) && self.world_decode.is_some() {
            let lanes = num_ambisonics_channels(self.order) as usize;
            self.amb.fill(0.0);
            let mut any = false;
            if let Some(r) = self.reflections.as_mut() {
                if let Ok(mut s) = r.slot.try_lock()
                    && let Some(p) = s.take()
                {
                    r.current = Some(p);
                }
                if let Some(p) = r.current.as_ref()
                    && let (Ok(input), Ok(mut out)) =
                        (AudioBufferRef::try_from(&self.mono[..]), AudioBufferMut::try_new(&mut self.amb_tmp[..], lanes))
                    && r.effect.apply(p, &input, &mut out)
                {
                    for (dst, wet) in self.amb.iter_mut().zip(&self.amb_tmp) {
                        *dst += wet * st.reverb_gain;
                    }
                    any = true;
                }
            }
            if let Some(path) = self.pathing.as_mut() {
                if let Ok(mut s) = path.slot.try_lock()
                    && let Some(p) = s.take()
                {
                    path.current = Some(p);
                }
                if let Some(p) = path.current.as_ref()
                    && let (Ok(input), Ok(mut out)) =
                        (AudioBufferRef::try_from(&self.mono[..]), AudioBufferMut::try_new(&mut self.amb_tmp[..], lanes))
                    && path.effect.apply(p, &input, &mut out).is_ok()
                {
                    add_into(&mut self.amb, &self.amb_tmp);
                    any = true;
                }
            }
            if any
                && let Some(dec) = self.world_decode.as_mut()
                && let (Ok(amb), Ok(mut tmp)) =
                    (AudioBufferRef::try_new(&self.amb[..], lanes), AudioBufferMut::try_new(&mut self.spk_tmp[..], oc))
            {
                let params = AmbisonicsDecodeEffectParams { order: self.order, hrtf: self.hrtf.clone(), orientation: listener };
                if dec.apply(&params, &amb, &mut tmp).is_ok() {
                    add_into(&mut self.spk, &self.spk_tmp);
                }
            }
        }
        let positioned = matches!(mode, Mode::Mono { .. });
        self.mode = mode;

        self.out.clear();
        for k in 0..FRAME {
            // Gate even an existing reverb tail outside the authored range.
            // Ambisonic recordings and surround beds retain their own level.
            let gain = if positioned {
                self.distance_gain + (st.attenuation - self.distance_gain) * ((k + 1) as f32 / FRAME as f32)
            } else {
                1.0
            };
            for c in 0..oc {
                self.out.push(self.spk[c * FRAME + k] * gain);
            }
        }
        self.distance_gain = st.attenuation;
        self.pos = 0;
        true
    }
}

impl<S> Iterator for SpatialVoice<S>
where
    S: Source,
    S::Item: Sample,
    f32: FromSample<S::Item>,
{
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        if self.pos >= self.out.len() && (self.done || !self.render_frame()) {
            return None;
        }
        let s = self.out[self.pos];
        self.pos += 1;
        Some(s)
    }
}

impl<S> Source for SpatialVoice<S>
where
    S: Source,
    S::Item: Sample,
    f32: FromSample<S::Item>,
{
    fn current_frame_len(&self) -> Option<usize> {
        None
    }

    fn channels(&self) -> u16 {
        self.out_channels as u16
    }

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    fn total_duration(&self) -> Option<Duration> {
        None
    }
}

// ------------------------------------------------------------ simulation

/// A voice as the simulation knows it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum VoiceKey {
    Object(ObjectId),
    Shot(u64),
}

/// Acoustic material values: absorption, scattering, transmission.
pub type MaterialValues = ([f32; 3], f32, [f32; 3]);

/// World geometry for the simulation: one merged static mesh (per-triangle
/// materials) plus moving occluders.
#[derive(Clone, Default, Debug)]
pub struct Geometry {
    pub vertices: Vec<[f32; 3]>,
    pub triangles: Vec<[i32; 3]>,
    /// Index into `materials` per triangle (empty = all material 0).
    pub material_indices: Vec<usize>,
    /// Empty = Steam Audio's Generic material.
    pub materials: Vec<MaterialValues>,
    pub dynamic: Vec<DynamicMesh>,
}

/// A moving occluder: a local-space mesh placed by its object's transform.
#[derive(Clone, Debug)]
pub struct DynamicMesh {
    pub id: ObjectId,
    pub vertices: Vec<[f32; 3]>,
    pub triangles: Vec<[i32; 3]>,
    pub material: MaterialValues,
}

/// One simulated voice this tick.
#[derive(Clone)]
pub struct SimVoice {
    pub key: VoiceKey,
    pub position: Vec3,
    pub occlusion: bool,
    /// Reflection slot and whether the voice simulates its own reflections
    /// (true) or takes the listener-centric reverb (false).
    pub reflections: Option<(ReflSlot, bool)>,
    pub pathing: Option<PathSlot>,
}

/// Occlusion (0 = blocked) and per-band transmission of one voice.
pub type OcclusionResult = (f32, [f32; 3]);

#[derive(Default)]
struct SimInput {
    listener: Mat4,
    voices: Vec<SimVoice>,
    geometry: Option<Geometry>,
    dynamic_transforms: Vec<(ObjectId, Mat4)>,
    /// New audio bake (serialized probe batch); `Some(None)` clears it.
    bake: Option<Option<Arc<Vec<u8>>>>,
    /// The listener is inside a baked probe volume.
    listener_in_bake: bool,
}

/// Direct occlusion runs this often.
const DIRECT_PERIOD: Duration = Duration::from_millis(33);

/// Background Steam Audio simulation: direct (occlusion + transmission),
/// reflections (per source and listener reverb, real-time or baked) and
/// pathing, against the latest geometry and audio bake.
pub struct Simulation {
    input: Arc<(Mutex<SimInput>, Condvar)>,
    results: Arc<Mutex<HashMap<VoiceKey, OcclusionResult>>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Simulation {
    pub fn new(steam: &SteamAudio) -> Self {
        let input = Arc::new((Mutex::new(SimInput::default()), Condvar::new()));
        let results = Arc::new(Mutex::new(HashMap::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let thread = {
            let (input, results, stop) = (input.clone(), results.clone(), stop.clone());
            let context = steam.context.clone();
            let settings = steam.settings;
            let config = steam.config.clone();
            std::thread::Builder::new()
                .name("steam-audio-simulation".into())
                .spawn(move || {
                    let shared = WorkerShared { input: &input, results: &results, stop: &stop };
                    if let Err(e) = run_worker(&context, settings, &config, &shared) {
                        tracing::warn!("Steam Audio simulation stopped: {e}");
                    }
                })
                .map_err(|e| tracing::warn!("Steam Audio simulation thread: {e}"))
                .ok()
        };
        Self { input, results, stop, thread }
    }

    /// Replace the world geometry (applied on the next run).
    pub fn set_geometry(&self, geometry: Geometry) {
        let (lock, wake) = &*self.input;
        lock.lock().unwrap().geometry = Some(geometry);
        wake.notify_one();
    }

    /// Replace (or clear) the audio bake.
    pub fn set_bake(&self, bake: Option<Arc<Vec<u8>>>) {
        let (lock, wake) = &*self.input;
        lock.lock().unwrap().bake = Some(bake);
        wake.notify_one();
    }

    /// The listener pose, voices and moving-occluder transforms for the next runs.
    pub fn submit(&self, listener: Mat4, voices: Vec<SimVoice>, dynamic_transforms: Vec<(ObjectId, Mat4)>, listener_in_bake: bool) {
        let mut input = self.input.0.lock().unwrap();
        input.listener = listener;
        input.voices = voices;
        input.dynamic_transforms = dynamic_transforms;
        input.listener_in_bake = listener_in_bake;
    }

    /// Latest direct results (voices without one are unobstructed).
    pub fn results(&self) -> HashMap<VoiceKey, OcclusionResult> {
        self.results.lock().unwrap().clone()
    }
}

impl Drop for Simulation {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.input.1.notify_one();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn coords(m: Mat4) -> CoordinateSystem {
    CoordinateSystem {
        right: v3(m.x_axis.truncate().normalize_or(Vec3::X)),
        up: v3(m.y_axis.truncate().normalize_or(Vec3::Y)),
        ahead: v3(-m.z_axis.truncate().normalize_or(Vec3::Z)),
        origin: v3(m.w_axis.truncate()),
    }
}

/// Steam Audio's row-major 4x4 (translation in the last column), from glam.
fn matrix(m: Mat4) -> audionimbus::Matrix<f32, 4, 4> {
    audionimbus::Matrix::new(m.transpose().to_cols_array_2d())
}

fn material(v: MaterialValues) -> audionimbus::Material {
    audionimbus::Material { absorption: v.0, scattering: v.1, transmission: v.2 }
}

struct WorkerShared<'a> {
    input: &'a (Mutex<SimInput>, Condvar),
    results: &'a Mutex<HashMap<VoiceKey, OcclusionResult>>,
    stop: &'a AtomicBool,
}

/// A ray tracer back end: how to make its scenes and simulation settings.
trait Tracer: audionimbus::RayTracer + Sized + 'static {
    type Backend: Clone;
    fn scene(context: &Context, device: &Self::Backend) -> Result<audionimbus::Scene<Self>, String>;
    fn settings<D, R, P, RE>(
        s: audionimbus::SimulationSettings<audionimbus::DefaultRayTracer, D, R, P, RE>,
        device: &Self::Backend,
    ) -> audionimbus::SimulationSettings<Self, D, R, P, RE>;
}

impl Tracer for audionimbus::DefaultRayTracer {
    type Backend = ();
    fn scene(context: &Context, _: &()) -> Result<audionimbus::Scene<Self>, String> {
        audionimbus::Scene::try_new(context).map_err(|e| format!("scene: {e}"))
    }
    fn settings<D, R, P, RE>(
        s: audionimbus::SimulationSettings<audionimbus::DefaultRayTracer, D, R, P, RE>,
        _: &(),
    ) -> audionimbus::SimulationSettings<Self, D, R, P, RE> {
        s
    }
}

impl Tracer for audionimbus::Embree {
    type Backend = audionimbus::EmbreeDevice;
    fn scene(context: &Context, device: &Self::Backend) -> Result<audionimbus::Scene<Self>, String> {
        audionimbus::Scene::try_with_embree(context, device.clone()).map_err(|e| format!("Embree scene: {e}"))
    }
    fn settings<D, R, P, RE>(
        s: audionimbus::SimulationSettings<audionimbus::DefaultRayTracer, D, R, P, RE>,
        _: &Self::Backend,
    ) -> audionimbus::SimulationSettings<Self, D, R, P, RE> {
        s.with_embree()
    }
}

impl Tracer for audionimbus::RadeonRays {
    type Backend = (audionimbus::OpenClDevice, audionimbus::RadeonRaysDevice);
    fn scene(context: &Context, device: &Self::Backend) -> Result<audionimbus::Scene<Self>, String> {
        audionimbus::Scene::try_with_radeon_rays(context, device.1.clone()).map_err(|e| format!("Radeon Rays scene: {e}"))
    }
    fn settings<D, R, P, RE>(
        s: audionimbus::SimulationSettings<audionimbus::DefaultRayTracer, D, R, P, RE>,
        device: &Self::Backend,
    ) -> audionimbus::SimulationSettings<Self, D, R, P, RE> {
        s.with_radeon_rays(device.0.clone(), device.1.clone())
    }
}

/// A reflection effect type: its simulation settings and parameters.
trait ReflKind:
    audionimbus::ReflectionEffectType
    + audionimbus::ReflectionEffectCompatible<audionimbus::Reflections, Self>
    + Send
    + Sync
    + Sized
    + 'static
{
    type Algo: audionimbus::ReflectionsAlgorithm<EffectType = Self>;
    fn algorithm(c: &SpatialAudioSettings) -> Self::Algo;
    fn parameters(baked: Option<BakedDataIdentifier>) -> Self::SimulationParameters;
    fn wrap(p: ReflectionEffectParams<Self>) -> ReflParams;
}

macro_rules! refl_kind {
    ($kind:ty, $algo:ident, $variant:ident, |$baked:ident| $params:expr) => {
        impl ReflKind for $kind {
            type Algo = audionimbus::$algo;
            fn algorithm(c: &SpatialAudioSettings) -> Self::Algo {
                audionimbus::$algo {
                    max_num_rays: c.rays,
                    num_diffuse_samples: c.diffuse_samples,
                    max_duration: c.duration,
                    max_num_sources: c.max_sources + 1,
                    num_threads: crate::sw_gi::bake_worker_count().clamp(1, 4) as u32,
                    max_order: c.order,
                }
            }
            fn parameters($baked: Option<BakedDataIdentifier>) -> Self::SimulationParameters {
                $params
            }
            fn wrap(p: ReflectionEffectParams<Self>) -> ReflParams {
                ReflParams::$variant(p)
            }
        }
    };
}

refl_kind!(Convolution, ConvolutionSettings, Convolution, |baked| audionimbus::ConvolutionParameters {
    baked_data_identifier: baked
});
refl_kind!(Parametric, ParametricSettings, Parametric, |baked| audionimbus::ParametricParameters {
    baked_data_identifier: baked,
    ..Default::default()
});
refl_kind!(Hybrid, HybridSettings, Hybrid, |baked| audionimbus::HybridParameters {
    baked_data_identifier: baked,
    ..Default::default()
});

const REVERB_ID: BakedDataIdentifier = BakedDataIdentifier::Reflections { variation: BakedDataVariation::Reverb };
const PATHING_ID: BakedDataIdentifier = BakedDataIdentifier::Pathing { variation: BakedDataVariation::Dynamic };

/// Pick the ray tracer (falling back to Default when its device can't be
/// created) and reflection type, then run the typed worker.
fn run_worker(
    context: &Context,
    settings: AudioSettings,
    config: &SpatialAudioSettings,
    shared: &WorkerShared,
) -> Result<(), String> {
    fn with_refl<T: Tracer>(
        context: &Context,
        settings: AudioSettings,
        config: &SpatialAudioSettings,
        shared: &WorkerShared,
        device: T::Backend,
    ) -> Result<(), String> {
        match config.reflections {
            ReflectionMode::Convolution => worker::<T, Convolution>(context, settings, config, shared, device),
            ReflectionMode::Hybrid => worker::<T, Hybrid>(context, settings, config, shared, device),
            ReflectionMode::Parametric | ReflectionMode::Off => worker::<T, Parametric>(context, settings, config, shared, device),
        }
    }
    let chosen = match config.ray_tracer {
        RayTracerKind::Embree => match audionimbus::EmbreeDevice::try_new(context) {
            Ok(device) => {
                tracing::info!("Steam Audio simulation: Embree ray tracer");
                return with_refl::<audionimbus::Embree>(context, settings, config, shared, device);
            }
            Err(e) => format!("Embree unavailable ({e})"),
        },
        RayTracerKind::RadeonRays => {
            let device = audionimbus::OpenClDeviceList::try_new(
                context,
                &audionimbus::OpenClDeviceSettings {
                    device_type: audionimbus::OpenClDeviceType::Gpu,
                    num_compute_units_to_reserve: 0,
                    fraction_of_compute_units_for_impulse_response_update: 0.5,
                    requires_true_audio_next: false,
                },
            )
            .map_err(|e| e.to_string())
            .and_then(|list| {
                if list.num_devices() == 0 {
                    return Err("no OpenCL GPU".into());
                }
                let cl = audionimbus::OpenClDevice::try_new(context, &list, 0).map_err(|e| e.to_string())?;
                let rr = audionimbus::RadeonRaysDevice::try_new(&cl).map_err(|e| e.to_string())?;
                Ok((cl, rr))
            });
            match device {
                Ok(device) => {
                    tracing::info!("Steam Audio simulation: Radeon Rays ray tracer");
                    return with_refl::<audionimbus::RadeonRays>(context, settings, config, shared, device);
                }
                Err(e) => format!("Radeon Rays unavailable ({e})"),
            }
        }
        RayTracerKind::Default => String::new(),
    };
    if !chosen.is_empty() {
        tracing::warn!("Steam Audio: {chosen}; using the default ray tracer");
    }
    with_refl::<audionimbus::DefaultRayTracer>(context, settings, config, shared, ())
}

type Sim<T, RE> = audionimbus::Simulator<T, audionimbus::Direct, audionimbus::Reflections, audionimbus::Pathing, RE>;
type SimSource<RE> = audionimbus::Source<audionimbus::Direct, audionimbus::Reflections, audionimbus::Pathing, RE>;

/// The scene being simulated and its instanced moving occluders.
struct WorldScene<T: Tracer> {
    scene: audionimbus::Scene<T>,
    instances: HashMap<ObjectId, audionimbus::InstancedMeshHandle>,
    has_geometry: bool,
}

fn build_scene<T: Tracer>(context: &Context, device: &T::Backend, geo: &Geometry) -> Result<WorldScene<T>, String> {
    use audionimbus::{InstancedMesh, InstancedMeshSettings, StaticMesh, StaticMeshSettings, Triangle};
    let mut scene = T::scene(context, device)?;
    let tri = |t: &[i32; 3]| Triangle::new(t[0], t[1], t[2]);
    let vec3 = |v: &[f32; 3]| Vector3::new(v[0], v[1], v[2]);
    if !geo.triangles.is_empty() {
        let vertices: Vec<Vector3> = geo.vertices.iter().map(vec3).collect();
        let triangles: Vec<Triangle> = geo.triangles.iter().map(tri).collect();
        let materials: Vec<audionimbus::Material> = if geo.materials.is_empty() {
            vec![audionimbus::Material::GENERIC]
        } else {
            geo.materials.iter().copied().map(material).collect()
        };
        let material_indices: Vec<usize> = if geo.material_indices.len() == triangles.len() {
            geo.material_indices.iter().map(|&m| m.min(materials.len() - 1)).collect()
        } else {
            vec![0; triangles.len()]
        };
        let mesh = StaticMesh::try_new(
            &scene,
            &StaticMeshSettings { vertices: &vertices, triangles: &triangles, material_indices: &material_indices, materials: &materials },
        )
        .map_err(|e| format!("static mesh: {e:?}"))?;
        scene.add_static_mesh(mesh);
    }
    let mut instances = HashMap::new();
    for dynamic in geo.dynamic.iter().filter(|d| !d.triangles.is_empty()) {
        let mut sub = T::scene(context, device)?;
        let vertices: Vec<Vector3> = dynamic.vertices.iter().map(vec3).collect();
        let triangles: Vec<Triangle> = dynamic.triangles.iter().map(tri).collect();
        let materials = [material(dynamic.material)];
        let indices = vec![0usize; triangles.len()];
        let mesh = StaticMesh::try_new(
            &sub,
            &StaticMeshSettings { vertices: &vertices, triangles: &triangles, material_indices: &indices, materials: &materials },
        )
        .map_err(|e| format!("moving occluder mesh: {e:?}"))?;
        sub.add_static_mesh(mesh);
        sub.commit();
        let instance = InstancedMesh::try_new(&scene, &InstancedMeshSettings { sub_scene: sub, transform: matrix(Mat4::IDENTITY) })
            .map_err(|e| format!("instanced mesh: {e}"))?;
        instances.insert(dynamic.id, scene.add_instanced_mesh(instance));
    }
    scene.commit();
    let has_geometry = !geo.triangles.is_empty() || !instances.is_empty();
    Ok(WorldScene { scene, instances, has_geometry })
}

fn worker<T: Tracer, RE: ReflKind>(
    context: &Context,
    settings: AudioSettings,
    config: &SpatialAudioSettings,
    shared: &WorkerShared,
    device: T::Backend,
) -> Result<(), String>
where
    (): audionimbus::DirectCompatible<audionimbus::Direct>
        + audionimbus::ReflectionsCompatible<audionimbus::Reflections>
        + audionimbus::PathingCompatible<audionimbus::Pathing>,
{
    use audionimbus::{
        DirectSimulationParameters, DirectSimulationSettings, Occlusion as OcclusionModel, OcclusionAlgorithm,
        PathingSimulationParameters, PathingSimulationSettings, ProbeBatch, ReflectionsSharedInputs, SerializedObject,
        SimulationInputs, SimulationParameters, SimulationSettings, SimulationSharedInputs, TransmissionParameters,
    };
    let sim_settings = T::settings(
        SimulationSettings::new(&settings)
            .with_direct(DirectSimulationSettings { max_num_occlusion_samples: 4 })
            .with_reflections(RE::algorithm(config))
            .with_pathing(PathingSimulationSettings { num_visibility_samples: 4 }),
        &device,
    );
    let mut simulator: Sim<T, RE> = audionimbus::Simulator::try_new(context, &sim_settings).map_err(|e| format!("simulator: {e}"))?;
    let mut world: WorldScene<T> = build_scene(context, &device, &Geometry::default())?;
    simulator.set_scene(&world.scene);
    simulator.commit();
    let mut probes: Option<(ProbeBatch, bool, bool)> = None;
    let mut sources: SimSources<RE> = HashMap::new();
    let reverb = SimSource::<RE>::try_new(&simulator).map_err(|e| format!("reverb source: {e}"))?;
    simulator.add_source(&reverb);
    simulator.commit();
    let reflections_on = config.reflections != ReflectionMode::Off;
    let indirect_period = Duration::from_secs_f32(1.0 / config.update_rate.max(1.0));
    let mut last_indirect = Instant::now() - indirect_period;
    let occlusion = DirectSimulationParameters::new().with_occlusion(
        OcclusionModel::new(OcclusionAlgorithm::Raycast).with_transmission(TransmissionParameters { num_transmission_rays: 1 }),
    );

    let (lock, wake) = shared.input;
    while !shared.stop.load(Ordering::Relaxed) {
        let (listener, voices, geometry, transforms, bake, in_bake) = {
            let guard = lock.lock().unwrap();
            let (mut g, _) = wake.wait_timeout(guard, DIRECT_PERIOD).unwrap();
            (g.listener, g.voices.clone(), g.geometry.take(), g.dynamic_transforms.clone(), g.bake.take(), g.listener_in_bake)
        };
        if shared.stop.load(Ordering::Relaxed) {
            break;
        }
        let mut commit = false;
        if let Some(geo) = geometry {
            world = build_scene(context, &device, &geo)?;
            simulator.set_scene(&world.scene);
            commit = true;
        }
        if !transforms.is_empty() && !world.instances.is_empty() {
            for (id, m) in &transforms {
                if let Some(h) = world.instances.get(id) {
                    world.scene.update_instanced_mesh_transform(*h, matrix(*m));
                }
            }
            world.scene.commit();
            commit = true;
        }
        if let Some(bake) = bake {
            if let Some((old, _, _)) = probes.take() {
                simulator.remove_probe_batch(&old);
            }
            if let Some(bytes) = bake {
                let loaded = SerializedObject::try_with_buffer(context, (*bytes).clone())
                    .map_err(|e| e.to_string())
                    .and_then(|mut so| ProbeBatch::load(context, &mut so).map_err(|e| e.to_string()));
                match loaded {
                    Ok(batch) => {
                        let has_reverb = batch.data_size(REVERB_ID) > 0;
                        let has_pathing = batch.data_size(PATHING_ID) > 0;
                        tracing::info!(
                            "Steam Audio bake: {} probes (reverb {has_reverb}, pathing {has_pathing})",
                            batch.num_probes()
                        );
                        simulator.add_probe_batch(&batch);
                        probes = Some((batch, has_reverb, has_pathing));
                    }
                    Err(e) => tracing::warn!("Steam Audio bake load: {e}"),
                }
            }
            commit = true;
        }
        let pathing_ready = probes.as_ref().is_some_and(|p| p.2);
        // Sources: (occlusion, own reflections, pathing) decide what's simulated.
        let wanted: HashMap<VoiceKey, (bool, bool, bool)> = voices
            .iter()
            .map(|v| {
                let own = v.reflections.as_ref().is_some_and(|r| r.1) && reflections_on;
                (v.key, (v.occlusion, own, v.pathing.is_some() && pathing_ready))
            })
            .collect();
        sources.retain(|key, (src, flags)| {
            let keep = wanted.get(key) == Some(flags);
            if !keep {
                simulator.remove_source(src);
                commit = true;
            }
            keep
        });
        for (key, flags) in &wanted {
            if !sources.contains_key(key) {
                let src = SimSource::<RE>::try_new(&simulator).map_err(|e| format!("source: {e}"))?;
                simulator.add_source(&src);
                sources.insert(*key, (src, *flags));
                commit = true;
            }
        }
        if commit {
            simulator.commit();
        }
        if !world.has_geometry {
            shared.results.lock().unwrap().clear();
            continue;
        }

        let reflection_shared = ReflectionsSharedInputs {
            num_rays: config.rays,
            num_bounces: config.bounces,
            duration: config.duration,
            order: config.order,
            irradiance_min_distance: 1.0,
        };
        let shared_inputs = SimulationSharedInputs::new(coords(listener)).with_direct().with_reflections(reflection_shared).with_pathing();
        let _ = simulator.set_shared_inputs(&shared_inputs);

        for v in &voices {
            let Some((src, (occ, own, path))) = sources.get(&v.key) else {
                continue;
            };
            let direct = if *occ { occlusion.clone() } else { DirectSimulationParameters::new() };
            let source = coords(Mat4::from_translation(v.position));
            let pathing = probes.as_ref().filter(|_| *path).map(|(batch, _, _)| PathingSimulationParameters {
                pathing_probes: batch.clone(),
                visibility_radius: 1.0,
                visibility_threshold: 0.1,
                visibility_range: 100.0,
                pathing_order: config.order,
                enable_validation: true,
                find_alternate_paths: true,
                deviation: audionimbus::DeviationModel::Default,
            });
            // One call per simulation type; pathing last (each call resets
            // the pathing-probe reference the source keeps).
            let _ = src.set_direct_inputs(&SimulationInputs { source, parameters: SimulationParameters::new().with_direct(direct) });
            if *own {
                let _ = src.set_reflections_inputs(&SimulationInputs {
                    source,
                    parameters: SimulationParameters::new().with_reflections(RE::parameters(None)),
                });
            }
            if let Some(p) = pathing {
                let _ = src.set_pathing_inputs(&SimulationInputs { source, parameters: SimulationParameters::new().with_pathing(p) });
            }
        }
        simulator.run_direct();
        let mut out = HashMap::with_capacity(sources.len());
        for (key, (src, _)) in &sources {
            if let Ok(direct) = src.get_direct_outputs() {
                let transmission = match direct.transmission {
                    Some(Transmission::FrequencyDependent(eq) | Transmission::FrequencyIndependent(eq)) => eq.0,
                    None => [1.0; 3],
                };
                out.insert(*key, (direct.occlusion.unwrap_or(1.0), transmission));
            }
        }
        *shared.results.lock().unwrap() = out;

        if last_indirect.elapsed() < indirect_period {
            continue;
        }
        last_indirect = Instant::now();
        let wants_reverb = config.listener_reverb && reflections_on && voices.iter().any(|v| v.reflections.as_ref().is_some_and(|r| !r.1));
        if wants_reverb {
            let baked = (in_bake && probes.as_ref().is_some_and(|p| p.1)).then_some(REVERB_ID);
            let _ = reverb.set_reflections_inputs(&SimulationInputs {
                source: coords(listener),
                parameters: SimulationParameters::new().with_reflections(RE::parameters(baked)),
            });
        }
        if reflections_on && (wants_reverb || sources.values().any(|(_, f)| f.1)) && simulator.run_reflections().is_ok() {
            for v in &voices {
                let Some((slot, own)) = &v.reflections else {
                    continue;
                };
                let params = if *own {
                    sources.get(&v.key).filter(|(_, f)| f.1).and_then(|(s, _)| s.get_reflections_outputs().ok())
                } else if wants_reverb {
                    reverb.get_reflections_outputs().ok()
                } else {
                    None
                };
                if let Some(p) = params {
                    *slot.lock().unwrap() = Some(RE::wrap(p));
                }
            }
        }
        if pathing_ready && sources.values().any(|(_, f)| f.2) && simulator.run_pathing().is_ok() {
            for v in &voices {
                if let (Some(slot), Some((src, flags))) = (&v.pathing, sources.get(&v.key))
                    && flags.2
                    && let Ok(p) = src.get_pathing_outputs()
                {
                    *slot.lock().unwrap() = Some(p);
                }
            }
        }
    }
    Ok(())
}

// ------------------------------------------------------------ baking

/// A probe volume to bake: the world box (unit cube -0.5..0.5 mapped by
/// `transform`) and its floor-grid placement.
#[derive(Clone, Debug)]
pub struct BakeVolume {
    pub transform: Mat4,
    pub spacing: f32,
    pub height: f32,
    pub reverb: bool,
    pub pathing: bool,
}

/// Bake listener-centric reverb and pathing for the probes placed in
/// `volumes` against `geometry`'s static meshes. Returns the serialized probe
/// batch (the `.audiobake` sidecar). `progress` gets 0..1 and a phase label.
pub fn bake(
    geometry: &Geometry,
    volumes: &[BakeVolume],
    config: &SpatialAudioSettings,
    progress: impl Fn(f32, &str) + Send + Sync + 'static,
    cancel: &AtomicBool,
) -> Result<Vec<u8>, String> {
    use audionimbus::{
        PathBakeParams, PathBaker, ProbeArray, ProbeBatch, ProbeGenerationParams, ProgressCallback, ReflectionsBakeFlags,
        ReflectionsBakeParams, ReflectionsBaker,
    };
    let config = config.sanitized();
    if volumes.is_empty() {
        return Err("no Audio Probe Volume in the scene".into());
    }
    let context = Context::try_new(&ContextSettings::default()).map_err(|e| format!("context: {e}"))?;
    let static_only = Geometry { dynamic: Vec::new(), ..geometry.clone() };
    if static_only.triangles.is_empty() {
        return Err("no static geometry to bake against (mark meshes Static)".into());
    }
    let world: WorldScene<audionimbus::DefaultRayTracer> = build_scene(&context, &(), &static_only)?;
    let mut batch = ProbeBatch::try_new(&context).map_err(|e| format!("probe batch: {e}"))?;
    for v in volumes {
        let mut array = ProbeArray::try_new(&context).map_err(|e| format!("probe array: {e}"))?;
        array.generate_probes(
            &world.scene,
            &ProbeGenerationParams::UniformFloor { spacing: v.spacing.max(0.1), height: v.height.max(0.0), transform: matrix(v.transform) },
        );
        batch.add_probe_array(&array);
    }
    batch.commit();
    if batch.num_probes() == 0 {
        return Err("no probes placed: each volume needs floor geometry inside it".into());
    }
    let threads = crate::sw_gi::bake_worker_count().max(1) as u32;
    let progress = Arc::new(progress);
    let reverb = volumes.iter().any(|v| v.reverb);
    let pathing = volumes.iter().any(|v| v.pathing);
    let phases = (reverb as u32 + pathing as u32).max(1) as f32;
    let mut phase = 0.0;
    if reverb {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        let p = progress.clone();
        let base = phase;
        let cb = ProgressCallback::new(move |f| p((base + f) / phases, "Baking reverb"));
        ReflectionsBaker::<audionimbus::DefaultRayTracer>::new()
            .bake_with_progress_callback(
                &context,
                &mut batch,
                &world.scene,
                ReflectionsBakeParams {
                    identifier: REVERB_ID,
                    bake_flags: ReflectionsBakeFlags::BAKE_CONVOLUTION | ReflectionsBakeFlags::BAKE_PARAMETRIC,
                    num_rays: config.bake_rays,
                    num_diffuse_samples: 1024.min(config.bake_rays),
                    num_bounces: config.bake_bounces,
                    simulated_duration: config.bake_duration,
                    saved_duration: config.bake_duration,
                    order: config.order,
                    num_threads: threads,
                    irradiance_min_distance: 1.0,
                    bake_batch_size: 1,
                },
                cb,
            )
            .map_err(|e| format!("reverb bake: {e:?}"))?;
        phase += 1.0;
    }
    if pathing {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        let p = progress.clone();
        let base = phase;
        let cb = ProgressCallback::new(move |f| p((base + f) / phases, "Baking pathing"));
        PathBaker::<audionimbus::DefaultRayTracer>::new()
            .bake_with_progress_callback(
                &context,
                &mut batch,
                &world.scene,
                PathBakeParams {
                    identifier: PATHING_ID,
                    num_samples: 16,
                    radius: 0.5,
                    threshold: 0.1,
                    visibility_range: 50.0,
                    path_range: 100.0,
                    num_threads: threads,
                },
                cb,
            )
            .map_err(|e| format!("pathing bake: {e:?}"))?;
    }
    progress(1.0, "Saving");
    Ok(batch.try_save(&context).map_err(|e| format!("saving bake: {e}"))?.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rodio::buffer::SamplesBuffer;

    fn tone(seconds: f32, rate: u32, channels: u16) -> SamplesBuffer<f32> {
        let n = (seconds * rate as f32) as usize;
        let data: Vec<f32> = (0..n)
            .flat_map(|i| {
                let v = (i as f32 * 440.0 * std::f32::consts::TAU / rate as f32).sin() * 0.5;
                std::iter::repeat_n(v, channels as usize)
            })
            .collect();
        SamplesBuffer::new(channels, rate, data)
    }

    fn energy(samples: &[f32], channel: usize, channels: usize) -> f32 {
        samples.iter().skip(channel).step_by(channels).map(|s| s * s).sum()
    }

    fn spatial() -> VoiceOptions {
        VoiceOptions { spatial: true, ..Default::default() }
    }

    /// A closed 10 m box room (inward-facing walls), as one static mesh.
    fn room(materials: Vec<MaterialValues>) -> Geometry {
        let s = 5.0;
        let v = [[-s, 0.0, -s], [s, 0.0, -s], [s, 0.0, s], [-s, 0.0, s], [-s, 4.0, -s], [s, 4.0, -s], [s, 4.0, s], [-s, 4.0, s]];
        let quads = [[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]];
        let mut triangles = Vec::new();
        for q in quads {
            for t in [[q[0], q[1], q[2]], [q[0], q[2], q[3]], [q[0], q[2], q[1]], [q[0], q[3], q[2]]] {
                triangles.push(t);
            }
        }
        let n = triangles.len();
        Geometry { vertices: v.to_vec(), triangles, material_indices: vec![0; n], materials, dynamic: Vec::new() }
    }

    #[test]
    fn voice_renders_whole_frames_of_stereo() {
        let sa = SteamAudio::new().expect("steam audio");
        let slot: VoiceSlot = Arc::new(Mutex::new(VoiceState::default()));
        let voice = sa.voice(tone(0.5, 44_100, 2), slot, spatial()).unwrap();
        assert_eq!(voice.channels(), 2);
        let out: Vec<f32> = voice.collect();
        assert_eq!(out.len() % (FRAME * 2), 0);
        assert_eq!(out.len() / (FRAME * 2), (24_000usize).div_ceil(FRAME));
        assert!(out.iter().all(|s| s.is_finite()));
        assert!(energy(&out, 0, 2) > 0.0);
    }

    #[test]
    fn direction_and_slot_updates_steer_the_output() {
        let sa = SteamAudio::new().expect("steam audio");
        let right = Emitter {
            transform: Mat4::from_translation(Vec3::new(3.0, 0.0, 0.0)),
            attenuation: 1.0,
            air_absorption: false,
            directivity: 0.0,
            directivity_sharpness: 1.0,
            hrtf: true,
        };
        let state = sa.state_for(&right, Mat4::IDENTITY);
        assert!(state.direction[0] > 0.9, "source on the right: {:?}", state.direction);
        let slot: VoiceSlot = Arc::new(Mutex::new(state));
        let out: Vec<f32> = sa.voice(tone(0.3, 48_000, 1), slot.clone(), spatial()).unwrap().collect();
        assert!(energy(&out, 1, 2) > energy(&out, 0, 2) * 1.5, "right ear louder");
        // Ambisonic encode + decode keeps the image on the right too.
        let opts = VoiceOptions { via_ambisonics: true, ..spatial() };
        let out: Vec<f32> = sa.voice(tone(0.3, 48_000, 1), slot.clone(), opts).unwrap().collect();
        assert!(energy(&out, 1, 2) > energy(&out, 0, 2), "encoded source still on the right");
        *slot.lock().unwrap() = VoiceState { attenuation: 0.0, ..state };
        let out: Vec<f32> = sa.voice(tone(0.3, 48_000, 1), slot, spatial()).unwrap().collect();
        assert!(energy(&out, 0, 2) + energy(&out, 1, 2) < 1e-3);
    }

    #[test]
    fn distance_limits_include_parametric_reverb_and_its_tail() {
        let sa = SteamAudio::new().unwrap();
        let slot = Arc::new(Mutex::new(VoiceState::default()));
        let reflections = Arc::new(Mutex::new(Some(ReflParams::Parametric(
            ReflectionEffectParams::<Parametric>::new([0.5; 3], 4, SAMPLE_RATE),
        ))));
        let opts = VoiceOptions { reflections: Some(reflections), ..spatial() };
        let mut voice = sa.voice(tone(2.0, SAMPLE_RATE, 2), slot.clone(), opts).unwrap();
        let audible: Vec<_> = voice.by_ref().take(FRAME * 2 * 12).collect();
        assert!(energy(&audible, 0, 2) + energy(&audible, 1, 2) > 1.0);

        // Move outside max distance while a reverb tail is already ringing.
        slot.lock().unwrap().attenuation = 0.0;
        let _transition: Vec<_> = voice.by_ref().take(FRAME * 2).collect();
        let distant: Vec<_> = voice.by_ref().take(FRAME * 2 * 12).collect();
        assert!(distant.iter().all(|s| s.abs() < 1e-6), "reverb must obey the source's max distance");

        // Moving back into range must preserve playback and spatial direction.
        *slot.lock().unwrap() = VoiceState { direction: [1.0, 0.0, 0.0], ..Default::default() };
        let returned: Vec<_> = voice.by_ref().take(FRAME * 2 * 12).collect();
        assert!(energy(&returned, 1, 2) > energy(&returned, 0, 2), "right-hand source must remain locatable with reverb");
    }

    #[test]
    fn distance_rolloff_is_applied_once_to_direct_sound() {
        let sa = SteamAudio::new().unwrap();
        let render_energy = |attenuation| {
            let slot = Arc::new(Mutex::new(VoiceState { attenuation, ..Default::default() }));
            let out: Vec<_> = sa.voice(tone(0.3, SAMPLE_RATE, 2), slot, spatial()).unwrap().collect();
            energy(&out, 0, 2) + energy(&out, 1, 2)
        };
        let full = render_energy(1.0);
        assert!(full > 1.0);
        for gain in [0.5, 0.1, 0.0] {
            let relative_energy = render_energy(gain) / full;
            assert!((relative_energy - gain * gain).abs() < 1e-5,
                "distance gain {gain} must scale amplitude once: relative energy {relative_energy}");
        }
    }

    #[test]
    fn audio_volume_reverb_level_preserves_direct_sound() {
        let sa = SteamAudio::new().unwrap();
        let render = |reflections| {
            let slot = Arc::new(Mutex::new(VoiceState { reverb_gain: 0.0, ..Default::default() }));
            sa.voice(tone(0.3, SAMPLE_RATE, 1), slot, VoiceOptions { reflections, ..spatial() })
                .unwrap().collect::<Vec<_>>()
        };
        let dry = render(None);
        let wet_slot = Arc::new(Mutex::new(Some(ReflParams::Parametric(
            ReflectionEffectParams::<Parametric>::new([0.5; 3], 4, SAMPLE_RATE),
        ))));
        let muted_reverb = render(Some(wet_slot));
        assert_eq!(dry, muted_reverb, "a zero reverb level must leave the positioned direct sound intact");
        assert!(energy(&dry, 0, 2) > 1.0);
    }

    #[test]
    fn speaker_layouts_ambisonic_clips_and_surround_beds() {
        let surround = SpatialAudioSettings { output: OutputLayout::Surround51, ..Default::default() };
        let sa = SteamAudio::with_settings(&surround, Path::new(".")).unwrap();
        let slot: VoiceSlot = Arc::new(Mutex::new(VoiceState::default()));
        let v = sa.voice(tone(0.1, 48_000, 1), slot.clone(), spatial()).unwrap();
        assert_eq!(v.channels(), 6);
        assert_eq!(v.count() % (FRAME * 6), 0);

        let sa = SteamAudio::new().unwrap();
        // First-order ambisonic clip (4 channels) decoded to binaural stereo.
        let opts = VoiceOptions { ambisonic_order: 1, ..Default::default() };
        let out: Vec<f32> = sa.voice(tone(0.2, 48_000, 4), slot.clone(), opts).unwrap().collect();
        assert!(out.iter().all(|s| s.is_finite()) && energy(&out, 0, 2) > 0.0);
        // A 5.1 bed plays through virtual surround on headphones.
        assert!(sa.wants(&VoiceOptions::default(), 6));
        let out: Vec<f32> = sa.voice(tone(0.2, 48_000, 6), slot, VoiceOptions::default()).unwrap().collect();
        assert!(energy(&out, 0, 2) > 0.0 && energy(&out, 1, 2) > 0.0);
    }

    #[test]
    fn missing_sofa_falls_back_to_the_default_hrtf() {
        let s = SpatialAudioSettings { hrtf_sofa: "does/not/exist.sofa".into(), ..Default::default() };
        assert!(SteamAudio::with_settings(&s, Path::new(".")).is_ok());
    }

    fn wait<T>(deadline: Duration, mut f: impl FnMut() -> Option<T>) -> Option<T> {
        let end = Instant::now() + deadline;
        loop {
            if let Some(v) = f() {
                return Some(v);
            }
            if Instant::now() > end {
                return None;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    #[test]
    fn occlusion_blocks_sound_behind_a_wall() {
        let sa = SteamAudio::new().expect("steam audio");
        let sim = Simulation::new(&sa);
        sim.set_geometry(Geometry {
            vertices: vec![[-10.0, -10.0, -5.0], [10.0, -10.0, -5.0], [10.0, 10.0, -5.0], [-10.0, 10.0, -5.0]],
            triangles: vec![[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]],
            ..Default::default()
        });
        let voice = |key, z| SimVoice { key, position: Vec3::new(0.0, 0.0, z), occlusion: true, reflections: None, pathing: None };
        let (behind, open) = (VoiceKey::Shot(1), VoiceKey::Shot(2));
        sim.submit(Mat4::IDENTITY, vec![voice(behind, -10.0), voice(open, 5.0)], Vec::new(), false);
        let results = wait(Duration::from_secs(5), || Some(sim.results()).filter(|r| r.len() == 2)).expect("results");
        let (occ_behind, transmission) = results[&behind];
        assert!(occ_behind < 0.5, "wall occludes: {occ_behind}");
        assert!(transmission.iter().all(|t| *t < 1.0), "some sound passes through: {transmission:?}");
        assert!(results[&open].0 > 0.9, "clear line of sight");
    }

    #[test]
    fn moving_occluder_follows_its_transform() {
        let sa = SteamAudio::new().expect("steam audio");
        let sim = Simulation::new(&sa);
        let id = ObjectId::new();
        // A 4 m panel at its local origin, facing Z.
        let panel = DynamicMesh {
            id,
            vertices: vec![[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [2.0, 2.0, 0.0], [-2.0, 2.0, 0.0]],
            triangles: vec![[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]],
            material: zestengine_core::AcousticPreset::Concrete.values().unwrap(),
        };
        sim.set_geometry(Geometry { dynamic: vec![panel], ..Default::default() });
        let key = VoiceKey::Shot(7);
        let voice = SimVoice { key, position: Vec3::new(0.0, 0.0, -10.0), occlusion: true, reflections: None, pathing: None };
        let between = Mat4::from_translation(Vec3::new(0.0, 0.0, -5.0));
        sim.submit(Mat4::IDENTITY, vec![voice.clone()], vec![(id, between)], false);
        let blocked = wait(Duration::from_secs(5), || sim.results().get(&key).map(|r| r.0).filter(|o| *o < 0.5));
        assert!(blocked.is_some(), "panel between listener and source occludes");
        let aside = Mat4::from_translation(Vec3::new(50.0, 0.0, -5.0));
        sim.submit(Mat4::IDENTITY, vec![voice], vec![(id, aside)], false);
        let clear = wait(Duration::from_secs(5), || sim.results().get(&key).map(|r| r.0).filter(|o| *o > 0.9));
        assert!(clear.is_some(), "moved aside, line of sight clears");
    }

    #[test]
    fn reflections_reach_the_voice_slot_for_each_type() {
        for mode in [ReflectionMode::Parametric, ReflectionMode::Convolution, ReflectionMode::Hybrid] {
            let cfg = SpatialAudioSettings { reflections: mode, rays: 512, update_rate: 30.0, ..Default::default() };
            let sa = SteamAudio::with_settings(&cfg, Path::new(".")).unwrap();
            let sim = Simulation::new(&sa);
            sim.set_geometry(room(vec![zestengine_core::AcousticPreset::Concrete.values().unwrap()]));
            let own: ReflSlot = Arc::new(Mutex::new(None));
            let reverb: ReflSlot = Arc::new(Mutex::new(None));
            let listener = Mat4::from_translation(Vec3::new(0.0, 1.5, 0.0));
            let voices = vec![
                SimVoice { key: VoiceKey::Shot(1), position: Vec3::new(2.0, 1.5, 0.0), occlusion: true, reflections: Some((own.clone(), true)), pathing: None },
                SimVoice { key: VoiceKey::Shot(2), position: Vec3::new(-2.0, 1.5, 0.0), occlusion: true, reflections: Some((reverb.clone(), false)), pathing: None },
            ];
            sim.submit(listener, voices, Vec::new(), false);
            let got = wait(Duration::from_secs(10), || {
                (own.lock().unwrap().is_some() && reverb.lock().unwrap().is_some()).then_some(())
            });
            assert!(got.is_some(), "{mode:?}: reflection and reverb parameters arrive");
            // The parameters drive a voice's reflection effect without errors.
            let slot: VoiceSlot = Arc::new(Mutex::new(sa.state_for(
                &Emitter {
                    transform: Mat4::from_translation(Vec3::new(2.0, 1.5, 0.0)),
                    attenuation: 1.0,
                    air_absorption: false,
                    directivity: 0.0,
                    directivity_sharpness: 1.0,
                    hrtf: true,
                },
                listener,
            )));
            let opts = VoiceOptions { reflections: Some(own.clone()), ..spatial() };
            let mut voice = sa.voice(tone(0.5, 48_000, 1), slot.clone(), opts).unwrap();
            let out: Vec<f32> = voice.by_ref().take(FRAME * 2 * 8).collect();
            assert!(out.iter().all(|s| s.is_finite()), "{mode:?}: finite output");
            assert!(energy(&out, 0, 2) + energy(&out, 1, 2) > 1.0, "{mode:?}: audible inside range");
            slot.lock().unwrap().attenuation = 0.0;
            let _transition: Vec<_> = voice.by_ref().take(FRAME * 2).collect();
            let distant: Vec<_> = voice.collect();
            assert!(!distant.is_empty());
            assert!(distant.iter().all(|s| s.abs() < 1e-6), "{mode:?}: reflections respect max distance");
        }
    }

    #[test]
    fn bake_places_probes_and_round_trips() {
        let cfg = SpatialAudioSettings { bake_rays: 512, bake_bounces: 2, bake_duration: 0.5, ..Default::default() };
        let volume = BakeVolume {
            transform: Mat4::from_scale_rotation_translation(Vec3::new(9.0, 3.5, 9.0), glam::Quat::IDENTITY, Vec3::new(0.0, 2.0, 0.0)),
            spacing: 3.0,
            height: 1.5,
            reverb: true,
            pathing: true,
        };
        let stop = AtomicBool::new(false);
        let bytes = bake(&room(Vec::new()), &[volume], &cfg, |_, _| {}, &stop).expect("bake");
        assert!(!bytes.is_empty());
        let context = Context::try_new(&ContextSettings::default()).unwrap();
        let mut so = audionimbus::SerializedObject::try_with_buffer(&context, bytes).unwrap();
        let batch = audionimbus::ProbeBatch::load(&context, &mut so).unwrap();
        assert!(batch.num_probes() > 0);
        assert!(batch.data_size(REVERB_ID) > 0, "reverb baked");
        assert!(batch.data_size(PATHING_ID) > 0, "pathing baked");
        // No volumes / no geometry are reported, not panicked on.
        assert!(bake(&room(Vec::new()), &[], &cfg, |_, _| {}, &stop).is_err());
    }

    #[test]
    fn air_absorption_and_directivity_attenuate() {
        let sa = SteamAudio::new().expect("steam audio");
        let far = Emitter {
            transform: Mat4::from_translation(Vec3::new(0.0, 0.0, -200.0)),
            attenuation: 1.0,
            air_absorption: true,
            directivity: 1.0,
            directivity_sharpness: 2.0,
            hrtf: true,
        };
        let s = sa.state_for(&far, Mat4::IDENTITY);
        assert!(s.air[2] < s.air[0], "highs absorbed more: {:?}", s.air);
        let side = Mat4::from_translation(Vec3::new(5.0, 0.0, -200.0));
        let s = sa.state_for(&far, side);
        assert!(s.directivity < 0.2, "listener at 90 degrees: {}", s.directivity);
    }
}
