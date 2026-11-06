//! Being under water: which water body a camera or the audio listener is in
//! (and how deep), the renderer's underwater look for a camera, and the
//! smoothed muffle / ambience state the audio mix follows.

use zestengine_core::{UnderwaterSettings, Water, WaterLook};
use zestengine_render::{GpuWater, UnderwaterFx};
use glam::{Mat4, Vec2, Vec3};

use crate::scene::LoadedScene;

/// A camera this close above the surface still gets the underwater pass, so a
/// near plane straddling the surface shows its waterline.
const VIEW_MARGIN: f32 = 0.25;

/// Seconds the muffle and ambience take to fade in or out.
pub const FADE_SECS: f32 = 0.2;

/// The water a point is in.
#[derive(Clone, Debug, PartialEq)]
pub struct Submersion {
    /// Depth below the surface (negative = above it, within the column).
    pub depth: f32,
    /// World surface height over the point.
    pub surface: f32,
    pub look: WaterLook,
    pub settings: UnderwaterSettings,
    /// The body's transform, packed waves, rest surface (local y) at the
    /// point and the water clock, for a waterline that follows the waves.
    pub world: Mat4,
    pub body: GpuWater,
    pub rest_surface: f32,
    pub clock: f32,
}

/// The deepest active water body (with its underwater look enabled) whose
/// column holds `p` and whose surface is no more than `margin` below it.
/// `t` is the water clock (`WaterRenderer::time`, the play clock in Play).
pub fn probe(scene: &LoadedScene, p: Vec3, t: f32, margin: f32) -> Option<Submersion> {
    let mut best: Option<Submersion> = None;
    for (i, o) in scene.objects.iter().enumerate() {
        let Some(water) = o.components.iter().find_map(|c| c.as_any().downcast_ref::<Water>()) else {
            continue;
        };
        if !water.underwater.enabled || !scene.frame_active(i) {
            continue;
        }
        let world = scene.frame_world(i);
        let Some((depth, surface)) = water.depth_below(world, p, t) else {
            continue;
        };
        if depth < -margin || best.as_ref().is_some_and(|b| b.depth >= depth) {
            continue;
        }
        let local = world.inverse().transform_point3(p);
        let rest_surface = water.column(Vec2::new(local.x, local.z)).map_or(0.0, |c| c.surface);
        best = Some(Submersion {
            depth,
            surface,
            look: water.look,
            settings: water.underwater.clone(),
            world,
            body: crate::water_render::pack_water(water),
            rest_surface,
            clock: t,
        });
    }
    best
}

/// The underwater look for a camera at `pos`, or None when it is clear of
/// every water body.
pub fn view_fx(scene: &LoadedScene, pos: Vec3, t: f32) -> Option<UnderwaterFx> {
    probe(scene, pos, t, VIEW_MARGIN).map(|s| fx_for(&s))
}

/// Turn a water body's look + underwater settings into the renderer's terms.
pub fn fx_for(s: &Submersion) -> UnderwaterFx {
    let look = &s.look;
    let absorb = look.absorption_depth.max(0.25);
    let mut body = s.body;
    // Preserve old authored underwater visibility until optical coefficients
    // are explicitly adopted; new water uses one medium above and below.
    if look.optics.is_none() { body.set_optics([0.0; 4], [0.0; 4]); }
    UnderwaterFx {
        surface: s.surface,
        shallow_color: look.shallow_color,
        deep_color: look.deep_color,
        density: visibility_density(s.settings.visibility),
        light_falloff: 0.35 / absorb,
        distortion: s.settings.distortion.max(0.0),
        blur: s.settings.blur.max(0.0),
        caustics: s.settings.caustics.max(0.0),
        water_to_world: s.world,
        body,
        rest_surface: s.rest_surface,
        clock: s.clock,
    }
}

/// Extinction per metre (green; red fades sooner, blue later) that leaves 5%
/// of the view at `visibility` metres: ln 20 / visibility.
pub fn visibility_density(visibility: f32) -> f32 {
    20f32.ln() / visibility.max(0.1)
}

/// Move `current` toward `target` by at most `step`.
pub fn approach(current: f32, target: f32, step: f32) -> f32 {
    if current < target { (current + step).min(target) } else { (current - step).max(target) }
}

/// The listener's smoothed time under water: drives the muffle on the whole
/// mix and the water's ambience loop, fading over [`FADE_SECS`]. Keeps the
/// last water's settings while fading out of it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UnderwaterMix {
    pub amount: f32,
    pub settings: Option<UnderwaterSettings>,
}

impl UnderwaterMix {
    /// Advance by `dt` (unscaled seconds); `under` is the water the listener
    /// is in this frame, if any.
    pub fn step(&mut self, under: Option<&UnderwaterSettings>, dt: f32) {
        if let Some(s) = under
            && self.settings.as_ref() != Some(s)
        {
            self.settings = Some(s.clone());
        }
        let target = if under.is_some() { 1.0 } else { 0.0 };
        self.amount = approach(self.amount, target, dt.max(0.0) / FADE_SECS);
        if self.amount <= 0.0 && under.is_none() {
            self.settings = None;
        }
    }

    /// How muffled the rest of the mix should be (0..1).
    pub fn muffle(&self) -> f32 {
        self.settings.as_ref().map_or(0.0, |s| self.amount * s.muffle.clamp(0.0, 1.0))
    }

    /// The ambience clip and its volume, while there is one to hear.
    pub fn ambience(&self) -> Option<(&str, f32)> {
        let s = self.settings.as_ref()?;
        (self.amount > 0.0 && !s.sound.trim().is_empty()).then(|| (s.sound.as_str(), self.amount * s.volume.max(0.0)))
    }
}

/// Where the listener is under water: the water it's in, if its ear is below
/// the surface.
pub fn listener(scene: &LoadedScene, ear: Vec3, t: f32) -> Option<UnderwaterSettings> {
    probe(scene, ear, t, 0.0).filter(|s| s.depth > 0.0).map(|s| s.settings)
}

/// Low-pass + duck for one interleaved stream, following a muffle target
/// (0 = untouched, 1 = fully muffled). Two one-pole stages (12 dB / octave)
/// whose cutoff slides from open to ~500 Hz; the coefficient is updated every
/// block, gliding to the target so there's no zipper noise.
pub struct MuffleFilter {
    channels: usize,
    sample_rate: f32,
    state: Vec<[f32; 2]>,
    amount: f32,
    alpha: f32,
    gain: f32,
    frame: u32,
}

const MUFFLE_BLOCK: u32 = 64;
const OPEN_HZ: f32 = 18_000.0;
const MUFFLED_HZ: f32 = 500.0;

impl MuffleFilter {
    pub fn new(channels: u16, sample_rate: u32) -> Self {
        let channels = channels.max(1) as usize;
        Self { channels, sample_rate: sample_rate.max(1) as f32, state: vec![[0.0; 2]; channels], amount: 0.0, alpha: 1.0, gain: 1.0, frame: 0 }
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Filter one sample of channel `ch` (samples arrive interleaved, channel
    /// 0 first in each frame).
    pub fn process(&mut self, x: f32, ch: usize, target: f32) -> f32 {
        if ch == 0 {
            if self.frame == 0 {
                let glide = MUFFLE_BLOCK as f32 / (0.05 * self.sample_rate);
                self.amount = approach(self.amount, target.clamp(0.0, 1.0), glide);
                let cutoff = OPEN_HZ * (MUFFLED_HZ / OPEN_HZ).powf(self.amount);
                self.alpha = 1.0 - (-std::f32::consts::TAU * cutoff / self.sample_rate).exp();
                self.gain = 1.0 - 0.3 * self.amount;
            }
            self.frame = (self.frame + 1) % MUFFLE_BLOCK;
        }
        let Some(s) = self.state.get_mut(ch) else {
            return x;
        };
        if self.amount <= 0.0 {
            // Bypassed, but tracking the input so engaging is seamless.
            *s = [x, x];
            return x;
        }
        s[0] += self.alpha * (x - s[0]);
        s[1] += self.alpha * (s[0] - s[1]);
        s[1] * self.gain
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::SceneObject;
    use zestengine_assets::ObjectSource;
    use zestengine_core::{Component, ObjectId};
    use glam::Quat;

    fn scene_with(water: Water, at: Vec3) -> LoadedScene {
        let mut scene = LoadedScene::empty();
        let components: Vec<Box<dyn Component>> = vec![Box::new(water)];
        scene.objects.push(SceneObject {
            net_role: Default::default(),
            component_roles: Default::default(),
            id: ObjectId::new(),
            name: String::new(),
            render: None,
            extra_render: Vec::new(),
            source: ObjectSource::Empty,
            enabled: true,
            static_geometry: false,
            layer: 0,
            parent: None,
            translation: at,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            components,
        });
        scene
    }

    fn calm() -> Water {
        Water { waves: Vec::new(), ..Default::default() }
    }

    #[test]
    fn detects_a_camera_under_the_surface() {
        let scene = scene_with(calm(), Vec3::new(0.0, 1.0, 0.0));
        let s = probe(&scene, Vec3::new(2.0, -1.0, 2.0), 0.0, 0.0).unwrap();
        assert!((s.depth - 2.0).abs() < 1e-5 && (s.surface - 1.0).abs() < 1e-5);
        assert!(probe(&scene, Vec3::new(2.0, 3.0, 2.0), 0.0, 0.0).is_none(), "above, outside the margin");
        assert!(probe(&scene, Vec3::new(40.0, -1.0, 0.0), 0.0, 0.0).is_none(), "beside the lake");
        let fx = view_fx(&scene, Vec3::new(0.0, 1.1, 0.0), 0.0).expect("just above: the waterline may cross the near plane");
        assert!((fx.surface - 1.0).abs() < 1e-5 && fx.density > 0.0);
        assert!(listener(&scene, Vec3::new(0.0, 1.1, 0.0), 0.0).is_none(), "the ear is still dry");
        assert!(listener(&scene, Vec3::new(0.0, 0.5, 0.0), 0.0).is_some());
    }

    #[test]
    fn disabled_or_inactive_water_is_ignored() {
        let mut off = calm();
        off.underwater.enabled = false;
        assert!(probe(&scene_with(off, Vec3::ZERO), Vec3::new(0.0, -1.0, 0.0), 0.0, 0.0).is_none());
        let mut scene = scene_with(calm(), Vec3::ZERO);
        scene.objects[0].enabled = false;
        assert!(probe(&scene, Vec3::new(0.0, -1.0, 0.0), 0.0, 0.0).is_none());
    }

    #[test]
    fn the_view_fades_out_at_the_visibility_distance() {
        let scene = scene_with(calm(), Vec3::ZERO);
        let mut s = probe(&scene, Vec3::new(0.0, -1.0, 0.0), 0.0, 0.0).unwrap();
        s.settings.visibility = 8.0;
        let left = (-fx_for(&s).density * 8.0).exp();
        assert!((left - 0.05).abs() < 1e-3, "5% of the view left at 8 m: {left}");
        s.settings.visibility = 30.0;
        assert!(fx_for(&s).density < visibility_density(8.0), "clearer water sees further");
    }

    #[test]
    fn mix_fades_in_and_out_over_the_fade_time() {
        let water = UnderwaterSettings { sound: "audio/under.ogg".into(), volume: 0.5, muffle: 0.8, ..Default::default() };
        let mut mix = UnderwaterMix::default();
        assert_eq!(mix.muffle(), 0.0);
        assert!(mix.ambience().is_none());
        mix.step(Some(&water), FADE_SECS * 0.5);
        assert!((mix.amount - 0.5).abs() < 1e-5);
        assert!((mix.muffle() - 0.4).abs() < 1e-5);
        assert_eq!(mix.ambience(), Some(("audio/under.ogg", 0.25)));
        mix.step(Some(&water), 1.0);
        assert_eq!(mix.amount, 1.0);
        // Surfacing keeps the water's settings until the fade ends.
        mix.step(None, FADE_SECS * 0.25);
        assert!((mix.amount - 0.75).abs() < 1e-5);
        assert!(mix.ambience().is_some());
        mix.step(None, 1.0);
        assert_eq!(mix, UnderwaterMix::default());
    }

    #[test]
    fn approach_clamps_at_the_target() {
        assert_eq!(approach(0.0, 1.0, 0.3), 0.3);
        assert_eq!(approach(0.9, 1.0, 0.3), 1.0);
        assert_eq!(approach(0.2, 0.0, 0.3), 0.0);
    }

    fn rms_through(filter: &mut MuffleFilter, hz: f32, target: f32) -> f32 {
        let sr = 48_000.0;
        let mut sum = 0.0;
        let n = 48_000;
        for i in 0..n {
            let x = (std::f32::consts::TAU * hz * i as f32 / sr).sin();
            let y = filter.process(x, 0, target);
            if i >= n / 2 {
                sum += y * y;
            }
        }
        (sum / (n / 2) as f32).sqrt()
    }

    #[test]
    fn muffle_cuts_highs_and_bypasses_when_dry() {
        let dry = rms_through(&mut MuffleFilter::new(1, 48_000), 4_000.0, 0.0);
        assert!((dry - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-3, "untouched when dry: {dry}");
        let highs = rms_through(&mut MuffleFilter::new(1, 48_000), 4_000.0, 1.0);
        let lows = rms_through(&mut MuffleFilter::new(1, 48_000), 100.0, 1.0);
        assert!(highs < 0.05, "4 kHz is muffled away: {highs}");
        assert!(lows > 0.4, "100 Hz still comes through: {lows}");
    }
}
