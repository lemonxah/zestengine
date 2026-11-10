//! Coverage-dependent shadow cache signatures. Opaque static casters retain batching/caching.
use super::*;
impl Renderer {
    pub(super) fn shadow_material_signature(&self, d: &DrawCmd, time: f32) -> u64 {
        use ash::vk::Handle;
        let m = &self.materials[d.material.0];
        if m.error { return 0; }
        let mut h = 0xcbf29ce484222325u64;
        let mut mix = |v: u64| { h ^= v; h = h.wrapping_mul(0x100000001b3); };
        // Displaced/hooked surfaces depend on camera/time. Coverage-only materials do not.
        if (m.params.tess.enabled && self.pipeline_cache.tessellation())
            || m.custom_shader.is_some_and(|s| self.pipeline_cache.has_hook(s.0)) {
            mix(self.frame_serial);
        }
        if m.features.alpha_mode == AlphaMode::Cutout {
            mix(1);
            // Descriptor replacements also cover wrap/filter changes without new texture handles.
            mix(m.set.as_raw());
            let p = &m.params;
            let alpha = if m.features.per_instance { d.instance_color.unwrap_or(p.base_color)[3] } else { p.base_color[3] };
            for value in [alpha, p.alpha_cutoff].into_iter().chain(p.albedo_tiling).chain(p.albedo_offset).chain(p.base_scroll) {
                mix(value.to_bits() as u64);
            }
            if p.base_scroll.iter().any(|&v| v != 0.0) { mix(time.to_bits() as u64); }
            for slot in [0, 4] {
                if let Some(t) = m.slots[slot] {
                    mix(t.0 as u64 + 1);
                    mix(self.textures[t.0].view.as_raw());
                    mix(self.texture_revisions.get(&t.0).copied().unwrap_or(0));
                } else { mix(0); }
            }
        }
        h
    }
}
