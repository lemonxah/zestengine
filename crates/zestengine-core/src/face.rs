//! Face tracking: a frame of Unified Expressions weights (plus eye gaze) from
//! any source, OpenXR face tracking on the headset or an OSC sender, and the
//! rules that map it onto an avatar's blendshapes by name.
//!
//! Blendshape names resolve case- and separator-insensitively against the
//! base shapes, the blended shapes avatars commonly carry (`MouthSmileLeft`,
//! `BrowUpRight`, `LipFunnel`, ...), ARKit-style names (`eyeBlinkLeft`,
//! `mouthRollUpper`, ...) and the VRM blink presets.

use glam::Vec2;
use serde::{Deserialize, Serialize};

use crate::TypedComponent;

/// The base Unified Expressions shapes, in wire order.
pub const EXPRESSIONS: [&str; 102] = [
    "EyeLookOutRight", "EyeLookInRight", "EyeLookUpRight", "EyeLookDownRight",
    "EyeLookOutLeft", "EyeLookInLeft", "EyeLookUpLeft", "EyeLookDownLeft",
    "EyeClosedRight", "EyeClosedLeft", "EyeSquintRight", "EyeSquintLeft",
    "EyeWideRight", "EyeWideLeft", "EyeDilationRight", "EyeDilationLeft",
    "EyeConstrictRight", "EyeConstrictLeft",
    "BrowPinchRight", "BrowPinchLeft", "BrowLowererRight", "BrowLowererLeft",
    "BrowInnerUpRight", "BrowInnerUpLeft", "BrowOuterUpRight", "BrowOuterUpLeft",
    "NoseSneerRight", "NoseSneerLeft", "NasalDilationRight", "NasalDilationLeft",
    "NasalConstrictRight", "NasalConstrictLeft",
    "CheekSquintRight", "CheekSquintLeft", "CheekPuffRight", "CheekPuffLeft",
    "CheekSuckRight", "CheekSuckLeft",
    "JawOpen", "JawRight", "JawLeft", "JawForward", "JawBackward", "JawClench", "JawMandibleRaise",
    "MouthClosed",
    "LipSuckUpperRight", "LipSuckUpperLeft", "LipSuckLowerRight", "LipSuckLowerLeft",
    "LipSuckCornerRight", "LipSuckCornerLeft",
    "LipFunnelUpperRight", "LipFunnelUpperLeft", "LipFunnelLowerRight", "LipFunnelLowerLeft",
    "LipPuckerUpperRight", "LipPuckerUpperLeft", "LipPuckerLowerRight", "LipPuckerLowerLeft",
    "MouthUpperUpRight", "MouthUpperUpLeft", "MouthLowerDownRight", "MouthLowerDownLeft",
    "MouthUpperDeepenRight", "MouthUpperDeepenLeft",
    "MouthUpperRight", "MouthUpperLeft", "MouthLowerRight", "MouthLowerLeft",
    "MouthCornerPullRight", "MouthCornerPullLeft", "MouthCornerSlantRight", "MouthCornerSlantLeft",
    "MouthFrownRight", "MouthFrownLeft", "MouthStretchRight", "MouthStretchLeft",
    "MouthDimpleRight", "MouthDimpleLeft", "MouthRaiserUpper", "MouthRaiserLower",
    "MouthPressRight", "MouthPressLeft", "MouthTightenerRight", "MouthTightenerLeft",
    "TongueOut", "TongueUp", "TongueDown", "TongueRight", "TongueLeft", "TongueRoll",
    "TongueBendDown", "TongueCurlUp", "TongueSquish", "TongueFlat", "TongueTwistRight", "TongueTwistLeft",
    "SoftPalateClose", "ThroatSwallow", "NeckFlexRight", "NeckFlexLeft",
];

/// Lowercase with `_`, `-`, spaces and dots removed: the form names compare in.
pub fn normalize(name: &str) -> String {
    name.chars().filter(|c| !matches!(c, '_' | '-' | ' ' | '.')).flat_map(char::to_lowercase).collect()
}

/// Index of a base shape by (normalized-insensitive) name.
pub fn expression_index(name: &str) -> Option<usize> {
    let n = normalize(name);
    EXPRESSIONS.iter().position(|e| normalize(e) == n)
}

fn idx(name: &str) -> usize {
    expression_index(name).unwrap_or_else(|| panic!("unknown expression {name}"))
}

/// Shapes avatars carry that blend several base shapes (their weight is the
/// mean), keyed by normalized name.
fn blended(n: &str) -> Option<&'static [&'static str]> {
    Some(match n {
        "mouthsmileright" => &["MouthCornerPullRight", "MouthCornerSlantRight"],
        "mouthsmileleft" => &["MouthCornerPullLeft", "MouthCornerSlantLeft"],
        "mouthsmile" => &["MouthCornerPullRight", "MouthCornerSlantRight", "MouthCornerPullLeft", "MouthCornerSlantLeft"],
        "mouthsadright" => &["MouthFrownRight", "MouthStretchRight"],
        "mouthsadleft" => &["MouthFrownLeft", "MouthStretchLeft"],
        "browupright" => &["BrowInnerUpRight", "BrowOuterUpRight"],
        "browupleft" => &["BrowInnerUpLeft", "BrowOuterUpLeft"],
        "browdownright" => &["BrowLowererRight", "BrowPinchRight"],
        "browdownleft" => &["BrowLowererLeft", "BrowPinchLeft"],
        "browinnerup" => &["BrowInnerUpRight", "BrowInnerUpLeft"],
        "browouterup" => &["BrowOuterUpRight", "BrowOuterUpLeft"],
        "cheekpuff" => &["CheekPuffRight", "CheekPuffLeft"],
        "cheeksuck" => &["CheekSuckRight", "CheekSuckLeft"],
        "cheeksquint" => &["CheekSquintRight", "CheekSquintLeft"],
        "nosesneer" => &["NoseSneerRight", "NoseSneerLeft"],
        "eyesquint" => &["EyeSquintRight", "EyeSquintLeft"],
        "eyewide" => &["EyeWideRight", "EyeWideLeft"],
        "eyeclosed" | "blink" => &["EyeClosedRight", "EyeClosedLeft"],
        "eyeblinkleft" | "blinkl" | "blinkleft" => &["EyeClosedLeft"],
        "eyeblinkright" | "blinkr" | "blinkright" => &["EyeClosedRight"],
        "lipsuckupper" | "mouthrollupper" => &["LipSuckUpperRight", "LipSuckUpperLeft"],
        "lipsucklower" | "mouthrolllower" => &["LipSuckLowerRight", "LipSuckLowerLeft"],
        "lipfunnelupper" => &["LipFunnelUpperRight", "LipFunnelUpperLeft"],
        "lipfunnellower" => &["LipFunnelLowerRight", "LipFunnelLowerLeft"],
        "lipfunnel" | "mouthfunnel" => {
            &["LipFunnelUpperRight", "LipFunnelUpperLeft", "LipFunnelLowerRight", "LipFunnelLowerLeft"]
        }
        "lippuckerupper" => &["LipPuckerUpperRight", "LipPuckerUpperLeft"],
        "lippuckerlower" => &["LipPuckerLowerRight", "LipPuckerLowerLeft"],
        "lippuckerright" => &["LipPuckerUpperRight", "LipPuckerLowerRight"],
        "lippuckerleft" => &["LipPuckerUpperLeft", "LipPuckerLowerLeft"],
        "lippucker" | "mouthpucker" => {
            &["LipPuckerUpperRight", "LipPuckerUpperLeft", "LipPuckerLowerRight", "LipPuckerLowerLeft"]
        }
        "mouthupperup" => &["MouthUpperUpRight", "MouthUpperUpLeft"],
        "mouthlowerdown" => &["MouthLowerDownRight", "MouthLowerDownLeft"],
        "mouthpress" => &["MouthPressRight", "MouthPressLeft"],
        "mouthstretch" => &["MouthStretchRight", "MouthStretchLeft"],
        "mouthtightener" => &["MouthTightenerRight", "MouthTightenerLeft"],
        "mouthdimple" => &["MouthDimpleRight", "MouthDimpleLeft"],
        "mouthfrown" => &["MouthFrownRight", "MouthFrownLeft"],
        "mouthright" => &["MouthUpperRight", "MouthLowerRight"],
        "mouthleft" => &["MouthUpperLeft", "MouthLowerLeft"],
        "mouthshrugupper" => &["MouthRaiserUpper"],
        "mouthshruglower" => &["MouthRaiserLower"],
        "mouthclose" => &["MouthClosed"],
        _ => return None,
    })
}

/// One frame of face tracking: base shape weights (0..1) and eye gaze
/// (x right, y up, each -1..1 for roughly +-45 degrees).
#[derive(Clone, Debug, PartialEq)]
pub struct FaceFrame {
    /// Base shape weights (0..1), in `EXPRESSIONS` order.
    pub weights: [f32; EXPRESSIONS.len()],
    /// Left-eye gaze (x right, y up, each -1..1 for roughly +-45 degrees).
    pub gaze_left: Vec2,
    /// Right-eye gaze (x right, y up, each -1..1).
    pub gaze_right: Vec2,
    /// Gaze data is present (the eye-look shapes follow it).
    pub has_gaze: bool,
}

impl Default for FaceFrame {
    fn default() -> Self {
        Self { weights: [0.0; EXPRESSIONS.len()], gaze_left: Vec2::ZERO, gaze_right: Vec2::ZERO, has_gaze: false }
    }
}

/// How a blendshape name reads a [`FaceFrame`].
#[derive(Clone, Debug, PartialEq)]
pub enum ShapeSource {
    /// One base expression, by index.
    Base(usize),
    /// The mean of several base expressions.
    Mean(Vec<usize>),
}

impl ShapeSource {
    /// What a blendshape (or expression) name reads, after stripping any path
    /// prefix such as `FT/v2/` or a mesh name.
    pub fn resolve(name: &str) -> Option<Self> {
        let leaf = name.rsplit(['/', '.']).next().unwrap_or(name);
        let n = normalize(leaf);
        if let Some(i) = EXPRESSIONS.iter().position(|e| normalize(e) == n) {
            return Some(Self::Base(i));
        }
        blended(&n).map(|names| Self::Mean(names.iter().map(|s| idx(s)).collect()))
    }

    /// What this source reads from a face frame (0..1).
    pub fn read(&self, f: &FaceFrame) -> f32 {
        match self {
            Self::Base(i) => f.weight(*i),
            Self::Mean(v) => v.iter().map(|&i| f.weight(i)).sum::<f32>() / v.len().max(1) as f32,
        }
    }
}

impl FaceFrame {
    /// Base weight `i`; the eye-look shapes come from the gaze when present.
    pub fn weight(&self, i: usize) -> f32 {
        if self.has_gaze && i < 8 {
            let (g, right) = if i < 4 { (self.gaze_right, true) } else { (self.gaze_left, false) };
            // Out/In follow the eye's side: the right eye looks out to the right.
            let out = if right { g.x } else { -g.x };
            return match i % 4 {
                0 => out.max(0.0),
                1 => (-out).max(0.0),
                2 => g.y.max(0.0),
                _ => (-g.y).max(0.0),
            }
            .min(1.0);
        }
        self.weights[i]
    }

    fn set(&mut self, name: &str, v: f32) {
        self.weights[idx(name)] = v.clamp(0.0, 1.0);
    }

    /// Split a signed value between a positive and a negative shape.
    fn signed(&mut self, pos: &[&str], neg: &[&str], v: f32) {
        for p in pos {
            self.set(p, v.max(0.0));
        }
        for n in neg {
            self.set(n, (-v).max(0.0));
        }
    }

    fn lid(&mut self, closed: &str, wide: &str, v: f32) {
        // Eye openness: 0 closed, ~0.75 relaxed, 1 wide.
        self.set(closed, 1.0 - (v / 0.75).clamp(0.0, 1.0));
        self.set(wide, ((v - 0.75) / 0.25).clamp(0.0, 1.0));
    }

    /// Apply one named parameter as OSC senders (VRCFaceTracking v2, Babble,
    /// Steam Link) publish them, with or without the `/avatar/parameters/`,
    /// `FT/` and `v2/` prefixes. False for names it doesn't know.
    pub fn set_param(&mut self, address: &str, v: f32) -> bool {
        let mut name = address.strip_prefix("/avatar/parameters/").unwrap_or(address);
        name = name.strip_prefix("FT/").unwrap_or(name);
        name = name.strip_prefix("v2/").unwrap_or(name);
        let n = normalize(name);
        let gaze = |f: &mut Self| f.has_gaze = true;
        match n.as_str() {
            "eyeleftx" => { self.gaze_left.x = v; gaze(self) }
            "eyelefty" => { self.gaze_left.y = v; gaze(self) }
            "eyerightx" => { self.gaze_right.x = v; gaze(self) }
            "eyerighty" => { self.gaze_right.y = v; gaze(self) }
            "eyex" => { self.gaze_left.x = v; self.gaze_right.x = v; gaze(self) }
            "eyey" => { self.gaze_left.y = v; self.gaze_right.y = v; gaze(self) }
            "eyelidleft" => self.lid("EyeClosedLeft", "EyeWideLeft", v),
            "eyelidright" => self.lid("EyeClosedRight", "EyeWideRight", v),
            "eyelid" => {
                self.lid("EyeClosedLeft", "EyeWideLeft", v);
                self.lid("EyeClosedRight", "EyeWideRight", v);
            }
            "jawx" => self.signed(&["JawRight"], &["JawLeft"], v),
            "jawz" => self.signed(&["JawForward"], &["JawBackward"], v),
            "mouthx" => self.signed(&["MouthUpperRight", "MouthLowerRight"], &["MouthUpperLeft", "MouthLowerLeft"], v),
            "mouthupperx" => self.signed(&["MouthUpperRight"], &["MouthUpperLeft"], v),
            "mouthlowerx" => self.signed(&["MouthLowerRight"], &["MouthLowerLeft"], v),
            "tonguex" => self.signed(&["TongueRight"], &["TongueLeft"], v),
            "tonguey" => self.signed(&["TongueUp"], &["TongueDown"], v),
            "cheekpuffsuckleft" => self.signed(&["CheekPuffLeft"], &["CheekSuckLeft"], v),
            "cheekpuffsuckright" => self.signed(&["CheekPuffRight"], &["CheekSuckRight"], v),
            "cheekpuffsuck" => self.signed(&["CheekPuffLeft", "CheekPuffRight"], &["CheekSuckLeft", "CheekSuckRight"], v),
            "smilefrownleft" => self.signed(&["MouthCornerPullLeft", "MouthCornerSlantLeft"], &["MouthFrownLeft"], v),
            "smilefrownright" => self.signed(&["MouthCornerPullRight", "MouthCornerSlantRight"], &["MouthFrownRight"], v),
            "smilefrown" => self.signed(
                &["MouthCornerPullLeft", "MouthCornerSlantLeft", "MouthCornerPullRight", "MouthCornerSlantRight"],
                &["MouthFrownLeft", "MouthFrownRight"],
                v,
            ),
            "smilesadleft" => {
                self.signed(&["MouthCornerPullLeft", "MouthCornerSlantLeft"], &["MouthFrownLeft", "MouthStretchLeft"], v)
            }
            "smilesadright" => {
                self.signed(&["MouthCornerPullRight", "MouthCornerSlantRight"], &["MouthFrownRight", "MouthStretchRight"], v)
            }
            "browexpressionleft" => {
                self.signed(&["BrowInnerUpLeft", "BrowOuterUpLeft"], &["BrowLowererLeft", "BrowPinchLeft"], v)
            }
            "browexpressionright" => {
                self.signed(&["BrowInnerUpRight", "BrowOuterUpRight"], &["BrowLowererRight", "BrowPinchRight"], v)
            }
            _ => match ShapeSource::resolve(name) {
                Some(ShapeSource::Base(i)) => self.weights[i] = v.clamp(0.0, 1.0),
                Some(ShapeSource::Mean(list)) => {
                    for i in list {
                        self.weights[i] = v.clamp(0.0, 1.0);
                    }
                }
                None => return false,
            },
        }
        true
    }

    /// Map `XR_FB_face_tracking2` weights (70, in the extension's order) onto
    /// Unified Expressions.
    pub fn from_fb_face2(w: &[f32]) -> Self {
        let mut f = Self::default();
        if w.len() < 70 {
            return f;
        }
        let pairs: [(&str, usize); 58] = [
            ("BrowLowererLeft", 0), ("BrowLowererRight", 1), ("BrowPinchLeft", 0), ("BrowPinchRight", 1),
            ("CheekPuffLeft", 2), ("CheekPuffRight", 3), ("CheekSquintLeft", 4), ("CheekSquintRight", 5),
            ("CheekSuckLeft", 6), ("CheekSuckRight", 7), ("MouthRaiserLower", 8), ("MouthRaiserUpper", 9),
            ("MouthDimpleLeft", 10), ("MouthDimpleRight", 11), ("EyeClosedLeft", 12), ("EyeClosedRight", 13),
            ("BrowInnerUpLeft", 22), ("BrowInnerUpRight", 23), ("JawOpen", 24), ("JawLeft", 25), ("JawRight", 26),
            ("JawForward", 27), ("EyeSquintLeft", 28), ("EyeSquintRight", 29), ("MouthFrownLeft", 30),
            ("MouthFrownRight", 31), ("MouthCornerPullLeft", 32), ("MouthCornerPullRight", 33),
            ("MouthCornerSlantLeft", 32), ("MouthCornerSlantRight", 33), ("LipFunnelLowerLeft", 34),
            ("LipFunnelUpperLeft", 35), ("LipFunnelLowerRight", 36), ("LipFunnelUpperRight", 37),
            ("MouthPressLeft", 38), ("MouthPressRight", 39), ("LipPuckerUpperLeft", 40), ("LipPuckerLowerLeft", 40),
            ("LipPuckerUpperRight", 41), ("LipPuckerLowerRight", 41), ("MouthStretchLeft", 42),
            ("MouthStretchRight", 43), ("LipSuckLowerLeft", 44), ("LipSuckUpperLeft", 45), ("LipSuckLowerRight", 46),
            ("LipSuckUpperRight", 47), ("MouthTightenerLeft", 48), ("MouthTightenerRight", 49),
            ("MouthClosed", 50), ("MouthLowerDownLeft", 51), ("MouthLowerDownRight", 52),
            ("NoseSneerLeft", 55), ("NoseSneerRight", 56), ("BrowOuterUpLeft", 57), ("BrowOuterUpRight", 58),
            ("EyeWideLeft", 59), ("EyeWideRight", 60), ("TongueOut", 68),
        ];
        for (name, k) in pairs {
            f.set(name, w[k]);
        }
        f.set("MouthUpperUpLeft", w[61]);
        f.set("MouthUpperUpRight", w[62]);
        f.set("MouthUpperLeft", w[53]);
        f.set("MouthLowerLeft", w[53]);
        f.set("MouthUpperRight", w[54]);
        f.set("MouthLowerRight", w[54]);
        // Gaze from the eye-look weights (look left / right / up / down per eye).
        f.gaze_left = Vec2::new(w[18] - w[16], w[20] - w[14]);
        f.gaze_right = Vec2::new(w[19] - w[17], w[21] - w[15]);
        f.has_gaze = true;
        f
    }

    /// Ease toward `target`: `k` is the share of the gap closed this frame.
    pub fn approach(&mut self, target: &FaceFrame, k: f32) {
        let k = k.clamp(0.0, 1.0);
        for (a, b) in self.weights.iter_mut().zip(&target.weights) {
            *a += (b - *a) * k;
        }
        self.gaze_left += (target.gaze_left - self.gaze_left) * k;
        self.gaze_right += (target.gaze_right - self.gaze_right) * k;
        self.has_gaze = target.has_gaze;
    }

    /// Compact wire form: (index, weight * 255) for every non-zero shape,
    /// then gaze as four signed bytes.
    pub fn encode(&self, out: &mut Vec<u8>) {
        let live: Vec<(u8, u8)> = self
            .weights
            .iter()
            .enumerate()
            .filter_map(|(i, w)| {
                let q = (w.clamp(0.0, 1.0) * 255.0).round() as u8;
                (q > 0).then_some((i as u8, q))
            })
            .collect();
        out.push(live.len() as u8);
        for (i, q) in live {
            out.push(i);
            out.push(q);
        }
        out.push(self.has_gaze as u8);
        for v in [self.gaze_left.x, self.gaze_left.y, self.gaze_right.x, self.gaze_right.y] {
            out.push(((v.clamp(-1.0, 1.0) * 127.0).round() as i8) as u8);
        }
    }

    /// Inverse of [`Self::encode`]; returns the frame and bytes read.
    pub fn decode(data: &[u8]) -> Option<(Self, usize)> {
        let mut f = Self::default();
        let n = *data.first()? as usize;
        let mut at = 1;
        for _ in 0..n {
            let (i, q) = (*data.get(at)? as usize, *data.get(at + 1)?);
            *f.weights.get_mut(i)? = q as f32 / 255.0;
            at += 2;
        }
        f.has_gaze = *data.get(at)? != 0;
        let g: Vec<f32> = data.get(at + 1..at + 5)?.iter().map(|b| *b as i8 as f32 / 127.0).collect();
        f.gaze_left = Vec2::new(g[0], g[1]);
        f.gaze_right = Vec2::new(g[2], g[3]);
        Some((f, at + 5))
    }
}

/// Project-level tracking input beyond the OpenXR runtime. Headset face and
/// eye tracking (OpenXR) is read whenever the runtime offers it; these only
/// run when enabled here (or from code).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct TrackingInputSettings {
    /// Connect to a running SlimeVR server for full-body trackers.
    pub slimevr: bool,
    /// OSC receiver for face tracking from external tools.
    pub osc_enabled: bool,
    /// UDP port the OSC receiver listens on.
    pub osc_port: u16,
    /// UDP port the avatar `/avatar/parameters` bridge sends to (tools listen
    /// here; VRChat's convention is 9001).
    pub osc_out_port: u16,
    /// Advertise the OSC ports with OSCQuery (HTTP tree plus mDNS) so tools
    /// find them without configuration.
    pub osc_query: bool,
}

impl Default for TrackingInputSettings {
    fn default() -> Self {
        Self { slimevr: false, osc_enabled: false, osc_port: 9000, osc_out_port: crate::avatar_osc::DEFAULT_OUT_PORT, osc_query: false }
    }
}

/// Where a [`FaceTracking`] reads its expressions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum FaceSource {
    /// A remote player's replicated face on their avatar, this machine's face
    /// tracking otherwise.
    #[default]
    Auto,
    /// This machine: the headset's face tracking, else the OSC receiver.
    Local,
    /// The face the Network Avatar above this object receives.
    Network,
}

impl FaceSource {
    /// Every source, in declaration order.
    pub const ALL: [FaceSource; 3] = [FaceSource::Auto, FaceSource::Local, FaceSource::Network];
    /// Inspector label of this source.
    pub fn label(self) -> &'static str {
        match self {
            FaceSource::Auto => "Auto",
            FaceSource::Local => "Local",
            FaceSource::Network => "Network",
        }
    }
}

/// Engine-filled state of a [`FaceTracking`], not saved.
#[derive(Clone, Debug, Default)]
pub struct FaceTrackingRuntime {
    /// The smoothed frame driving the blendshapes.
    pub frame: FaceFrame,
    /// Tracking data arrived recently.
    pub active: bool,
    /// (blendshape, what drives it) for every matched blendshape.
    pub matched: Vec<(String, String)>,
    /// Every blendshape on the model.
    pub shapes: Vec<String>,
}

/// Drives the model's blendshapes (and eye bones) from face tracking using
/// Unified Expressions names. Blendshapes match by name automatically;
/// `mappings` override or add pairs.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FaceTracking {
    /// Where the expressions come from.
    pub source: FaceSource,
    /// 0 = raw, toward 1 = smoother (and laggier).
    pub smoothing: f32,
    /// Turn the eye bones with the gaze.
    pub eye_bones: bool,
    /// Gaze of 1 turns the eyes this far: (yaw, pitch) in degrees.
    pub eye_angle: [f32; 2],
    /// (blendshape, expression) pairs that override the automatic match.
    pub mappings: Vec<(String, String)>,
    /// Engine-filled state (not saved).
    #[serde(skip)]
    pub runtime: FaceTrackingRuntime,
}

impl Default for FaceTracking {
    fn default() -> Self {
        Self {
            source: FaceSource::Auto,
            smoothing: 0.4,
            eye_bones: true,
            eye_angle: [30.0, 25.0],
            mappings: Vec::new(),
            runtime: FaceTrackingRuntime::default(),
        }
    }
}

impl TypedComponent for FaceTracking {
    const NAME: &'static str = "Face Tracking";
}

impl FaceTracking {
    /// What drives `shape`: a mapping, else its own name.
    pub fn source_for(&self, shape: &str) -> Option<(ShapeSource, String)> {
        if let Some((_, expr)) = self.mappings.iter().find(|(s, _)| s == shape) {
            return ShapeSource::resolve(expr).map(|s| (s, expr.clone()));
        }
        ShapeSource::resolve(shape).map(|s| (s, shape.rsplit(['/', '.']).next().unwrap_or(shape).to_owned()))
    }
}

/// Engine-filled state of [`BlendShapes`], not saved.
#[derive(Clone, Debug, Default)]
pub struct BlendShapesRuntime {
    /// Every blendshape on the model.
    pub names: Vec<String>,
}

/// Fixed blendshape weights on a model, set in the inspector or from code
/// (`ctx.get_mut::<BlendShapes>(id)`); Face Tracking overrides the shapes it
/// drives while it tracks.
#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct BlendShapes {
    /// (name, weight) pairs, in insertion order.
    pub weights: Vec<(String, f32)>,
    /// Engine-filled state (not saved).
    #[serde(skip)]
    pub runtime: BlendShapesRuntime,
}

impl TypedComponent for BlendShapes {
    const NAME: &'static str = "Blend Shapes";
}

impl BlendShapes {
    /// Set (or create) a shape's weight.
    pub fn set(&mut self, name: &str, weight: f32) {
        match self.weights.iter_mut().find(|(n, _)| n == name) {
            Some((_, w)) => *w = weight,
            None => self.weights.push((name.to_owned(), weight)),
        }
    }

    /// A shape's weight (0 when absent).
    pub fn get(&self, name: &str) -> f32 {
        self.weights.iter().find(|(n, _)| n == name).map(|(_, w)| *w).unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_resolve_across_conventions() {
        assert_eq!(ShapeSource::resolve("JawOpen"), Some(ShapeSource::Base(idx("JawOpen"))));
        assert_eq!(ShapeSource::resolve("jaw_open"), Some(ShapeSource::Base(idx("JawOpen"))));
        assert_eq!(ShapeSource::resolve("Body.FT/v2/JawOpen"), Some(ShapeSource::Base(idx("JawOpen"))));
        assert_eq!(ShapeSource::resolve("eyeBlinkLeft"), Some(ShapeSource::Mean(vec![idx("EyeClosedLeft")])));
        assert_eq!(ShapeSource::resolve("Blink_L"), Some(ShapeSource::Mean(vec![idx("EyeClosedLeft")])));
        assert!(matches!(ShapeSource::resolve("MouthSmileLeft"), Some(ShapeSource::Mean(v)) if v.len() == 2));
        assert!(ShapeSource::resolve("Hair_Long").is_none());
    }

    #[test]
    fn osc_parameters_fill_the_frame() {
        let mut f = FaceFrame::default();
        assert!(f.set_param("/avatar/parameters/FT/v2/JawOpen", 0.6));
        assert!(f.set_param("/avatar/parameters/v2/SmileFrownLeft", -0.5));
        assert!(f.set_param("/avatar/parameters/FT/v2/EyeLidLeft", 0.0));
        assert!(f.set_param("/avatar/parameters/FT/v2/EyeLeftX", 0.5));
        assert!(f.set_param("MouthSmileRight", 0.8));
        assert!(!f.set_param("/avatar/parameters/VelocityX", 1.0));
        assert!((f.weights[idx("JawOpen")] - 0.6).abs() < 1e-6);
        assert!((f.weights[idx("MouthFrownLeft")] - 0.5).abs() < 1e-6);
        assert_eq!(f.weights[idx("MouthCornerPullLeft")], 0.0);
        assert_eq!(f.weights[idx("EyeClosedLeft")], 1.0);
        assert!((f.weights[idx("MouthCornerSlantRight")] - 0.8).abs() < 1e-6);
        // Left eye looking right = looking in, toward the nose.
        assert!((f.weight(idx("EyeLookInLeft")) - 0.5).abs() < 1e-6);
        assert_eq!(f.weight(idx("EyeLookOutLeft")), 0.0);
    }

    #[test]
    fn fb_face_tracking_maps_onto_unified_expressions() {
        let mut w = [0.0f32; 70];
        w[24] = 0.7;
        w[32] = 0.4;
        w[12] = 1.0;
        w[19] = 0.5;
        let f = FaceFrame::from_fb_face2(&w);
        assert!((f.weights[idx("JawOpen")] - 0.7).abs() < 1e-6);
        assert!((ShapeSource::resolve("MouthSmileLeft").unwrap().read(&f) - 0.4).abs() < 1e-6);
        assert_eq!(f.weights[idx("EyeClosedLeft")], 1.0);
        assert!((f.weight(idx("EyeLookOutRight")) - 0.5).abs() < 1e-6, "right eye looking right is out");
    }

    #[test]
    fn wire_form_round_trips() {
        let mut f = FaceFrame::default();
        f.set_param("JawOpen", 0.5);
        f.set_param("EyeRightY", -0.25);
        let mut b = Vec::new();
        f.encode(&mut b);
        let (g, n) = FaceFrame::decode(&b).unwrap();
        assert_eq!(n, b.len());
        assert!((g.weights[idx("JawOpen")] - 0.5).abs() < 0.01);
        assert!((g.gaze_right.y + 0.25).abs() < 0.01 && g.has_gaze);
    }
}
