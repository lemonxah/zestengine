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
    disp = vec3(0.0);
    nrm = vec3(0.0, 1.0, 0.0);
    crest = 0.0;
    lost = 0.0;
    float crest_max = 0.0;
    int n = min(int(body.look1.z + 0.5), MAX_WATER_WAVES);
    for (int i = 0; i < n; ++i) {
        vec4 a = body.wave_a[i];
        vec4 b = body.wave_b[i];
        float w = water_band(a.z, foot);
        float phase = a.z * dot(a.xy, xz) - a.w * t;
        float s = sin(phase);
        float c = cos(phase);
        disp.x += w * b.x * a.x * c;
        disp.z += w * b.x * a.y * c;
        disp.y += w * b.y * s;
        float wa = w * a.z * b.y;
        nrm.x -= a.x * wa * c;
        nrm.z -= a.y * wa * c;
        nrm.y -= w * b.z * s;
        crest += w * b.z * s;
        crest_max += w * abs(b.z);
        float slope = a.z * b.y;
        lost += (1.0 - w * w) * 0.5 * slope * slope;
    }
    #ifdef WATER_OCEAN_DATA
    if(body.ocean.y>0u){
        vec4 ocean=water_ocean_field(body,xz,foot,0u);
        vec4 gradient=water_ocean_field(body,xz,foot,1u);
        disp+=ocean.xyz;
        nrm.xz-=gradient.xy;
        lost+=0.5*max(gradient.w-dot(gradient.xy,gradient.xy),0.0);
    }
    if(body.recording.y>0u){
        disp+=water_recorded_field(body,xz,foot,0u).xyz;
        vec4 gradient=water_recorded_field(body,xz,foot,1u);
        nrm.xz-=gradient.xy;lost+=0.5*max(gradient.w-dot(gradient.xy,gradient.xy),0.0);
    }
    #endif
    nrm = normalize(nrm);
    crest = crest_max > 1e-5 ? crest / crest_max : 0.0;
}

#endif
