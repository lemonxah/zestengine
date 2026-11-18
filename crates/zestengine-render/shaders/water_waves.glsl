// Shared water wave evaluation for raster, displaced RT geometry and ray hits.
#ifndef ZEST_WATER_WAVES
#define ZEST_WATER_WAVES
#ifdef WATER_OCEAN_DATA
#include "water_ocean.glsl"
#endif
// How much of a wave of wave number `k` (rad/m) a footprint of `foot`
// metres can show: all of it from four footprints per wavelength, none
// below two.
float water_band(float k, float foot) {
    return k > 0.0 ? smoothstep(2.0, 4.0, 6.2831853 / (k * max(foot, 1e-5))) : 1.0;
}

// Mip of a body's ripple field (`size` texels across the `width`-metre
// window) that holds no detail finer than `foot` metres.
float water_ripple_lod(float foot, float width, float size) {
    return max(log2(max(foot, 1e-5) * size / max(width, 1e-5)) + 0.75, 0.0);
}

// The ripple field fades out toward the window border.
float water_ripple_fade(vec2 uv) {
    vec2 e = min(uv, 1.0 - uv);
    return smoothstep(0.0, 0.08, min(e.x, e.y));
}

// Gerstner sum at the local rest point `xz`, the same function as
// zestengine_core::Water::displacement, band-limited to a sampling footprint of
// `foot` metres (vertex spacing, pixel size): waves shorter than four
// footprints fade out, gone at two, so a coarse grid or a distant pixel
// never aliases them. `lost` is the slope variance the faded waves would
// have added (rougher highlights stand in for them). `crest` is the summed
// Q*k*sin term (compression at the crests), normalised to -1..1.
void water_evaluate_waves(GpuWater body, vec2 xz, float t, float foot, out vec3 disp, out vec3 nrm, out float crest, out float lost) {
#define WATER_WAVE_SOURCE body
#include "water_waves_body.glsl"
#undef WATER_WAVE_SOURCE
}

#endif
