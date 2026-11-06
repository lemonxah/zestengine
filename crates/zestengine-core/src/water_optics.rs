//! Homogeneous water optics, in inverse metres. Shared by surface rendering,
//! submerged views and tools; no backend resources or light selection here.
use glam::Vec3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WaterOptics {
    /// RGB absorption coefficients (1/m). Red usually attenuates first.
    pub absorption: [f32; 3],
    /// RGB scattering coefficients (1/m).
    pub scattering: [f32; 3],
    /// Refractive index relative to air.
    pub ior: f32,
    /// Henyey–Greenstein anisotropy: positive scatters light forward.
    pub anisotropy: f32,
}

impl Default for WaterOptics {
    fn default() -> Self {
        Self { absorption: [0.32, 0.065, 0.025], scattering: [0.015, 0.035, 0.045], ior: 1.333, anisotropy: 0.65 }
    }
}

impl WaterOptics {
    pub fn sanitized(self) -> Self {
        let finite = |v: f32, fallback: f32, lo: f32, hi: f32| if v.is_finite() { v.clamp(lo, hi) } else { fallback };
        let d = Self::default();
        Self {
            absorption: std::array::from_fn(|i| finite(self.absorption[i], d.absorption[i], 0.0, 100.0)),
            scattering: std::array::from_fn(|i| finite(self.scattering[i], d.scattering[i], 0.0, 100.0)),
            ior: finite(self.ior, d.ior, 1.0, 2.5),
            anisotropy: finite(self.anisotropy, d.anisotropy, -0.95, 0.95),
        }
    }

    pub fn extinction(self) -> Vec3 {
        let v = self.sanitized();
        Vec3::from(v.absorption) + Vec3::from(v.scattering)
    }

    /// Beer–Lambert transmittance along the ray, not vertical water depth.
    pub fn transmittance(self, distance: f32) -> Vec3 {
        let distance = if distance.is_nan() { 0.0 } else { distance.clamp(0.0, 1.0e6) };
        (-self.extinction() * distance).exp()
    }

    /// Integral of scattering along a segment under unit incident radiance.
    /// Multiplying by a phase function and incident light gives radiance.
    pub fn integrated_scattering(self, distance: f32) -> Vec3 {
        Vec3::from(self.sanitized().scattering) / self.extinction().max(Vec3::splat(1e-6))
            * (Vec3::ONE - self.transmittance(distance))
    }

    pub fn normal_reflectance(self) -> f32 {
        let n = self.sanitized().ior;
        ((n - 1.0) / (n + 1.0)).powi(2)
    }

    /// Unpolarized dielectric Fresnel, including the underwater critical
    /// angle. `cosine` is the absolute view/normal cosine; `exiting` means
    /// travelling from water into air. The result partitions reflected and
    /// transmitted energy at the surface.
    pub fn surface_reflectance(self, cosine: f32, exiting: bool) -> f32 {
        let ior = self.sanitized().ior;
        if ior == 1.0 { return 0.0; }
        let c = if cosine.is_finite() { cosine.abs().clamp(0.0, 1.0) } else { 1.0 };
        let eta = if exiting { ior } else { 1.0 / ior };
        let sin2 = eta * eta * (1.0 - c * c);
        if sin2 >= 1.0 { return 1.0; }
        let ct = (1.0 - sin2).sqrt();
        let rs = (eta * c - ct) / (eta * c + ct).max(1e-6);
        let rp = (eta * ct - c) / (eta * ct + c).max(1e-6);
        (0.5 * (rs * rs + rp * rp)).clamp(0.0, 1.0)
    }

    /// Normalized Henyey–Greenstein phase, integral over the sphere = 1.
    pub fn phase(self, cosine: f32) -> f32 {
        let g = self.sanitized().anisotropy;
        let c = if cosine.is_finite() { cosine.clamp(-1.0, 1.0) } else { 0.0 };
        (1.0 - g * g) / (4.0 * std::f32::consts::PI * (1.0 + g * g - 2.0 * g * c).max(1e-4).powf(1.5))
    }

    /// Convert older colour/depth controls once at the rendering boundary.
    /// All scenes then use the same optical model, including non-water fluids.
    pub fn from_legacy(look: &crate::WaterLook) -> Self {
        let depth = if look.absorption_depth.is_finite() { look.absorption_depth.max(0.01) } else { 4.0 };
        let rate = 3.0 / depth;
        let colour = Vec3::from(look.deep_color).clamp(Vec3::ZERO, Vec3::splat(0.98));
        let extinction = Vec3::new(1.55, 1.0, 0.8) * rate;
        Self {
            absorption: (extinction * (Vec3::ONE - colour)).to_array(),
            scattering: (extinction * colour).to_array(),
            ior: 1.333,
            anisotropy: 0.65,
        }.sanitized()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn surface_fresnel_has_a_clear_snell_window_and_no_reflection_at_matched_ior() {
        let water = WaterOptics::default();
        for exiting in [false, true] {
            assert!((water.surface_reflectance(1.0, exiting) - water.normal_reflectance()).abs() < 1e-6);
        }
        assert_eq!(water.surface_reflectance(0.5, true), 1.0);
        assert!(water.surface_reflectance(0.9, true) < 0.04);
        let clear = WaterOptics { ior: 1.0, ..water };
        for i in 0..=100 {
            let c=i as f32/100.0;
            assert_eq!(clear.surface_reflectance(c, false), 0.0);
            for exiting in [false,true] { assert!((0.0..=1.0).contains(&water.surface_reflectance(c, exiting))); }
        }
    }
    #[test]
    fn old_water_looks_keep_legacy_controls_and_new_looks_roundtrip() {
        let old: crate::WaterLook = ron::from_str("(deep_color: (0.1, 0.2, 0.3), absorption_depth: 7.0)").unwrap();
        assert!(old.optics.is_none());
        assert!(WaterOptics::from_legacy(&old).transmittance(10.0).is_finite());
        let new = crate::WaterLook::default();
        let saved = ron::to_string(&new).unwrap();
        assert_eq!(ron::from_str::<crate::WaterLook>(&saved).unwrap(), new);
        assert!(new.optics.is_some());
    }
    #[test]
    fn optical_segments_compose_and_conserve_energy() {
        let o = WaterOptics::default();
        assert_eq!(o.transmittance(0.0), Vec3::ONE);
        assert_eq!(o.integrated_scattering(0.0), Vec3::ZERO);
        assert!((o.transmittance(7.0) - o.transmittance(3.0) * o.transmittance(4.0)).length() < 1e-6);
        let t = o.transmittance(10.0);
        assert!(t.x < t.y && t.y < t.z);
        assert!((t + o.integrated_scattering(10.0)).cmple(Vec3::ONE).all());
        assert!((o.normal_reflectance() - 0.02037).abs() < 1e-4);
        let clear = WaterOptics { absorption: [0.0; 3], scattering: [0.0; 3], ..o };
        assert_eq!(clear.transmittance(100.0), Vec3::ONE);
        assert_eq!(clear.integrated_scattering(100.0), Vec3::ZERO);
    }

    #[test]
    fn phase_is_normalized_and_finite_for_authored_extremes() {
        for g in [-0.95, 0.0, 0.65, 0.95] {
            let o = WaterOptics { anisotropy: g, ..Default::default() };
            let n = 100_000;
            let integral: f32 = (0..n).map(|i| o.phase(-1.0 + (i as f32 + 0.5) * 2.0 / n as f32)).sum::<f32>()
                * 4.0 * std::f32::consts::PI / n as f32;
            assert!((integral - 1.0).abs() < 0.002, "{g}: {integral}");
        }
        let bad = WaterOptics { absorption: [f32::NAN, -1.0, f32::INFINITY], scattering: [-1.0; 3], ior: f32::NAN, anisotropy: 9.0 };
        assert!(bad.transmittance(10.0).is_finite());
        assert!(bad.phase(f32::NAN).is_finite());
    }
}
