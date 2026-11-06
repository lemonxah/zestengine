// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Independent homogeneous-medium equations, mirrored by core::WaterOptics.
#ifndef ZEST_WATER_OPTICS
#define ZEST_WATER_OPTICS
// Reuse the unused wave.w lanes: the shared frame remains within Vulkan's
// guaranteed 16 KiB uniform limit. These lanes are independent of wave count.
#ifdef ZEST_WATER_FRAME
vec4 water_absorption(GpuWater w) {
    if (w.wave_b[3].w < 1.0) {
        vec3 e = vec3(1.55, 1.0, 0.8) * 3.0 / max(w.shallow.w, 0.01);
        return vec4(e * (1.0 - clamp(w.deep.rgb, 0.0, 0.98)), 1.333);
    }
    return vec4(w.wave_b[0].w, w.wave_b[1].w, w.wave_b[2].w, w.wave_b[3].w);
}
vec4 water_scattering_coeff(GpuWater w) {
    if (w.wave_b[3].w < 1.0) {
        vec3 e = vec3(1.55, 1.0, 0.8) * 3.0 / max(w.shallow.w, 0.01);
        return vec4(e * clamp(w.deep.rgb, 0.0, 0.98), 0.65);
    }
    return vec4(w.wave_b[4].w, w.wave_b[5].w, w.wave_b[6].w, w.wave_b[7].w);
}
#endif
vec3 water_transmittance(vec3 absorption, vec3 scattering, float distance_m) {
    return exp(-(max(absorption, vec3(0.0)) + max(scattering, vec3(0.0))) * clamp(distance_m, 0.0, 1e6));
}
vec3 water_scatter_integral(vec3 absorption, vec3 scattering, vec3 transmittance) {
    return max(scattering, vec3(0.0)) / max(absorption + scattering, vec3(1e-6)) * (1.0 - transmittance);
}
float water_normal_reflectance(float ior) {
    float r = (max(ior, 1.0) - 1.0) / (max(ior, 1.0) + 1.0);
    return r * r;
}
// Exact dielectric Fresnel: unlike Schlick, this also describes Snell's
// window seen from below and approaches total internal reflection smoothly.
float water_surface_reflectance(float cosine, float eta) {
    if (abs(eta - 1.0) < 1e-6) return 0.0;
    float c = clamp(abs(cosine), 0.0, 1.0);
    float sin2 = eta * eta * (1.0 - c * c);
    if (sin2 >= 1.0) return 1.0;
    float ct = sqrt(max(1.0 - sin2, 0.0));
    vec2 ratios = vec2(eta * c - ct, eta * ct - c) / max(vec2(eta * c + ct, eta * ct + c), vec2(1e-6));
    return clamp(0.5 * dot(ratios, ratios), 0.0, 1.0);
}
float water_phase(float cosine, float anisotropy) {
    float g = clamp(anisotropy, -0.95, 0.95);
    float d = max(1.0 + g * g - 2.0 * g * clamp(cosine, -1.0, 1.0), 1e-4);
    return (1.0 - g * g) / (12.566370614359 * d * sqrt(d));
}
#endif
