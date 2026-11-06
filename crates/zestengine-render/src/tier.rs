//! Render quality presets. A tier caps what the renderer does regardless of
//! the scene's settings, so one project runs on a desktop GPU and on a
//! standalone headset's tiled mobile GPU.

/// Render quality preset; see [`RenderTier::for_platform`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RenderTier {
    /// Everything the scene asks for.
    #[default]
    Desktop,
    /// Standalone headsets / phones: smaller shadows with fewer cascades, no
    /// ray tracing, no screen-space or voxel GI (baked lighting only), no SSR,
    /// a lower-resolution bloom chain, MSAA and multiview for the XR eyes.
    Mobile,
}

impl RenderTier {
    /// Mobile on Android, Desktop elsewhere. `ZESTENGINE_RENDER_TIER=mobile` /
    /// `desktop` overrides it (e.g. to preview the mobile tier on a PC).
    pub fn for_platform() -> Self {
        match std::env::var("ZESTENGINE_RENDER_TIER").ok().as_deref().map(str::to_ascii_lowercase).as_deref() {
            Some("mobile") => Self::Mobile,
            Some("desktop") => Self::Desktop,
            _ if cfg!(target_os = "android") => Self::Mobile,
            _ => Self::Desktop,
        }
    }

    pub fn is_mobile(self) -> bool {
        self == Self::Mobile
    }

    /// Largest shadow-atlas layer size (texels per side).
    pub fn max_shadow_resolution(self) -> u32 {
        match self {
            Self::Desktop => 8192,
            Self::Mobile => 1024,
        }
    }

    /// Cascades for the main directional light (at most 4).
    pub fn shadow_cascades(self) -> usize {
        match self {
            Self::Desktop => 4,
            Self::Mobile => 2,
        }
    }

    /// Realtime GI (screen-probe gather, voxel GI) and ray-traced passes.
    /// Off on mobile: baked lightmaps / probes carry the indirect light.
    pub fn realtime_gi(self) -> bool {
        self == Self::Desktop
    }

    /// Screen-space and ray-traced reflections (the probes always work).
    pub fn reflections(self) -> bool {
        self == Self::Desktop
    }

    /// Divisor applied to a view's extent before its bloom chain is built
    /// (the chain itself starts at half of that).
    pub fn bloom_divisor(self) -> u32 {
        match self {
            Self::Desktop => 1,
            Self::Mobile => 2,
        }
    }

    /// MSAA sample count for the XR eye scene pass (resolved in the pass).
    pub fn xr_msaa(self) -> u8 {
        match self {
            Self::Desktop => 1,
            Self::Mobile => 4,
        }
    }

    /// Render both XR eyes in one multiview pass when the device can.
    pub fn xr_multiview(self) -> bool {
        self == Self::Mobile
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mobile_is_cheaper_than_desktop() {
        let (d, m) = (RenderTier::Desktop, RenderTier::Mobile);
        assert!(m.max_shadow_resolution() < d.max_shadow_resolution());
        assert!(m.shadow_cascades() < d.shadow_cascades() && d.shadow_cascades() <= 4);
        assert!(!m.realtime_gi() && !m.reflections() && d.realtime_gi() && d.reflections());
        assert!(m.bloom_divisor() > d.bloom_divisor());
        assert_eq!((d.xr_msaa(), m.xr_msaa()), (1, 4));
        assert!(m.xr_multiview() && !d.xr_multiview());
    }
}
