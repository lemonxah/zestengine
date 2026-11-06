//! Shadow diagnostics (`ZESTENGINE_SHADOW_DEBUG`) and the depth-bias scale.
//!
//! `ZESTENGINE_SHADOW_DEBUG=1` logs, once per second, each shadow layer's caster
//! count (tested) and how many pass the CPU frustum test, plus last frame's GPU
//! cull total for shadow views. `=2` also lists every draw dropped from the
//! caster list (and the reason), and every caster a layer's frustum rejects.
//! The shader side is the "Shadow layer" viewport render mode (gi_debug 7).

use std::sync::OnceLock;
use std::time::{Duration, Instant};

use glam::Mat4;

/// Local-light receiver depth-bias multiplier: 1500 is 1x, with the default light bias
/// contributing half a shadow texel (see `shadow_factor` in standard.frag).
/// The scene's `shadow_bias_scale` replaces it at runtime;
/// 0 or less means "use this default".
pub const DEFAULT_BIAS_SCALE: f32 = 1500.0;

/// Effective bias scale for a scene value (non-positive/NaN = the default).
pub fn effective_bias_scale(v: f32) -> f32 {
    if v.is_finite() && v > 0.0 { v } else { DEFAULT_BIAS_SCALE }
}

/// Directional-light normal-offset, in shadow texels at a receiver facing
/// the light edge-on (scaled by sin of the angle to the light; see
/// `shadow_factor` in standard.frag). The scene's `shadow_normal_offset`
/// replaces it at runtime; 0 or less means "use this default".
/// Point/spot receiver-plane filtering does not displace the receiver normal.
pub const DEFAULT_NORMAL_OFFSET: f32 = 2.0;

/// Effective normal-offset scale for a scene value (non-positive/NaN = the default).
pub fn effective_normal_offset(v: f32) -> f32 {
    if v.is_finite() && v > 0.0 { v } else { DEFAULT_NORMAL_OFFSET }
}

/// 0 = off, 1 = per-layer counts, 2 = plus per-draw reasons.
pub fn level() -> u8 {
    static L: OnceLock<u8> = OnceLock::new();
    *L.get_or_init(|| match std::env::var("ZESTENGINE_SHADOW_DEBUG") {
        Ok(v) => match v.trim() {
            "" | "0" | "off" | "false" => 0,
            "2" | "all" | "verbose" => 2,
            _ => 1,
        },
        Err(_) => 0,
    })
}

/// Throttle: true at most once a second, only when the debug level is on.
pub fn due() -> bool {
    if level() == 0 {
        return false;
    }
    static LAST: OnceLock<std::sync::Mutex<Instant>> = OnceLock::new();
    let m = LAST.get_or_init(|| std::sync::Mutex::new(Instant::now() - Duration::from_secs(2)));
    let Ok(mut last) = m.lock() else { return false };
    if last.elapsed() >= Duration::from_secs(1) {
        *last = Instant::now();
        true
    } else {
        false
    }
}

/// Log the shadow caster picture. `drop_reason(i)` says why draw `i` is not a
/// caster; `in_frustum(i, vp)` is the CPU frustum test of caster `i` in a view.
pub fn report(
    views: &[(usize, Mat4)],
    draw_count: usize,
    casters: &[usize],
    gpu_shadow_visible: u32,
    drop_reason: &dyn Fn(usize) -> Option<&'static str>,
    in_frustum: &dyn Fn(usize, &Mat4) -> bool,
) {
    tracing::info!(
        target: "shadow_debug",
        "{} draws, {} shadow casters; last frame's GPU cull kept {} shadow instances over {} views",
        draw_count, casters.len(), gpu_shadow_visible, views.len()
    );
    for (layer, vp) in views {
        let visible = casters.iter().filter(|&&i| in_frustum(i, vp)).count();
        tracing::info!(
            target: "shadow_debug",
            "layer {layer}: {} casters tested, {visible} inside the light frustum",
            casters.len()
        );
        if level() >= 2 {
            for &i in casters {
                if !in_frustum(i, vp) {
                    tracing::info!(target: "shadow_debug", "layer {layer}: draw {i} dropped: outside the light frustum");
                }
            }
        }
    }
    if level() >= 2 {
        for i in 0..draw_count {
            if let Some(why) = drop_reason(i) {
                tracing::info!(target: "shadow_debug", "draw {i} is not a caster: {why}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bias_scale_falls_back_to_default() {
        assert_eq!(effective_bias_scale(0.0), DEFAULT_BIAS_SCALE);
        assert_eq!(effective_bias_scale(-3.0), DEFAULT_BIAS_SCALE);
        assert_eq!(effective_bias_scale(f32::NAN), DEFAULT_BIAS_SCALE);
        assert_eq!(effective_bias_scale(800.0), 800.0);
    }
}
