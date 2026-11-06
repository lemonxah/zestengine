// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Shared by standard.vert and standard.frag: the frame UBO, the material FX
// block and the water wave functions.

const int MAX_LIGHTS = 128;

struct Light {
    vec4 pos_kind;   // xyz world position, w = kind (0 dir, 1 point, 2 spot)
    vec4 dir_range;  // xyz travel direction (normalized), w = range
    vec4 color;      // rgb color*intensity, w = cos(outer half-angle)
    vec4 spot;       // x = cos(inner half-angle)
};

const int MAX_SHADOW_VIEWS = 64;
const int MAX_PROBE_VOLUMES = 8;

// One baked light-probe volume: world->local plus the grid layout (matches
// GpuProbeVolume on the CPU).
struct ProbeVolume {
    mat4 world_to_local; // local box spans -size/2 .. +size/2
    vec4 size_base;      // xyz = local box size, w = first probe index (sh_base)
    vec4 counts;         // xyz = probe counts per axis
    vec4 params;         // Light Volume mode: x = edge fade (m, 0 = classic cascade volume), y = weight
};

const int MAX_WATER_BODIES = 4;
const int MAX_WATER_WAVES = 8;
const int MAX_WATER_FOAM = 4;
// How far the graded mesh of an endless water body reaches from the camera;
// the surface fades out just before this so it blends into the sky at the
// horizon. Keep in sync with ENDLESS_EXTENT in
// crates/zestengine-engine/src/water_render.rs.
const float WATER_HORIZON_EXTENT = 240.0;

// One water surface's waves + look (matches GpuWater on the CPU, packed from
// zestengine_core::Water::resolved_waves).
struct GpuWater {
    vec4 wave_a[MAX_WATER_WAVES]; // xy direction (local XZ), z wave number, w angular speed
    vec4 wave_b[MAX_WATER_WAVES]; // xyz wave; w lanes: absorption RGB/IOR, scattering RGB/anisotropy
    vec4 shallow; // rgb shallow colour, w absorption depth
    vec4 deep;    // rgb deep colour, w clarity
    vec4 look0;   // x roughness, y ripple strength, z ripple scale, w foam width
    vec4 look1;   // x crest foam, y refraction, z wave count, w summed amplitude
    vec4 ripple;  // xy ripple window centre (local XZ), z width (0 = none)
    mat4 to_local; // world to the body's local space
    vec4 bounds;   // xy half extents in local XZ (0 = endless), z rest surface, w caustic strength
    vec4 foam[MAX_WATER_FOAM]; // churned spots: xy centre (local XZ), z radius, w amount
    vec4 extra;    // x lights under the surface throw caustics out of it (pools), y shore depth (0 = off), zw shore texture half extents (local XZ)
    vec4 reflect;  // x reflection mode (0 env, 1 planar, 2 probes, 3 off), yz local current (m/s)
};

layout(set = 0, binding = 0) uniform FrameData {
    mat4 view;
    mat4 proj;
    mat4 view_proj;
    vec4 camera_pos;
    vec4 light_dir;
    vec4 light_color;
    vec4 ambient;
    vec4 misc; // x = time, y = light count, z = shadow spacing, w = probe-volume count
    vec4 postfx0; // x tonemap mode, y exposure EV, z grade exposure, w contrast
    vec4 postfx1; // x saturation, y temperature, z tint, w grading enabled
    vec4 postfx2; // x vignette enabled, y intensity, z smoothness, w screen width
    vec4 postfx3; // xyz vignette color, w screen height
    vec4 cascade_splits; // far view-space distance of each directional cascade
    Light lights[MAX_LIGHTS];
    mat4 shadow_vp[MAX_SHADOW_VIEWS];
    ProbeVolume probe_volumes[MAX_PROBE_VOLUMES];
    vec4 debug; // x = lightmap-UV checker preview, y = gi debug, z = screen-GI active
    mat4 inv_view;
    mat4 inv_proj;
    vec4 ssr; // x = enabled, y = intensity, z = max view-space dist, w = roughness cutoff
    vec4 refl_center;  // xyz = probe CAPTURE position (re-vector anchor), w = intensity (0 = no zone)
    vec4 refl_extents; // xyz = probe box half-extents, w = box-projection enabled
    vec4 fog_color;    // xyz = fog colour, w = density (0 = no fog)
    vec4 fog_params;   // x = height falloff, y = height ref, z = start distance
    vec4 refl_box;     // xyz = probe box CENTER (capture + box offset), w unused
    vec4 refl2_center;  // second probe (binding 9): same layout, w = intensity (0 = none)
    vec4 refl2_extents;
    vec4 refl2_box;
    vec4 refl_blend;    // x = slot 0 blend distance, y = slot 1 blend distance, z = slot on top where boxes overlap
    vec4 refl3_center;  // third probe (binding 11)
    vec4 refl3_extents;
    vec4 refl3_box;
    vec4 refl4_center;  // fourth probe (binding 12)
    vec4 refl4_extents;
    vec4 refl4_box;
    vec4 refl_blend2;   // x = slot 2 blend distance, y = slot 3 blend distance
    vec4 water;         // x = water clock, y = opaque copy valid (bindings 5/6), z = body count
    GpuWater water_bodies[MAX_WATER_BODIES];
    // Previous frame's jittered view-projection: reprojects this frame's
    // world position into last frame's clip space for the motion vectors.
    mat4 prev_view_proj;
#ifdef MULTIVIEW
    // Multiview pass (both XR eyes at once): per-eye cameras by gl_ViewIndex.
    // The camera fields above hold a view enclosing both eyes.
    mat4 eye_view[2];
    mat4 eye_view_proj[2];
    vec4 eye_camera_pos[2];
#endif
} frame;

// The camera this invocation renders for (an eye under multiview).
#ifdef MULTIVIEW
#define CAMERA_POS frame.eye_camera_pos[gl_ViewIndex]
#define VIEW_MATRIX frame.eye_view[gl_ViewIndex]
#define VIEW_PROJ frame.eye_view_proj[gl_ViewIndex]
#else
#define CAMERA_POS frame.camera_pos
#define VIEW_MATRIX frame.view
#define VIEW_PROJ frame.view_proj
#endif

// Extended per-material params (beyond the 128-byte push block).
layout(set = 1, binding = 4) uniform MaterialFx {
    vec4 rim;    // rgb rim colour, w rim power
    vec4 toon;   // x rim strength, y ramp smoothness, z emission pulse, w _
    vec4 scroll; // xy base UV scroll, zw emission UV scroll
    vec4 matcap; // xyz matcap layer strengths, w _
    vec4 matcap_blend; // xyz blend modes (0 add / 1 mul / 2 replace), w _
    vec4 albedo_st;   // xy tiling (scale), zw offset
    vec4 normal_st;
    vec4 orm_st;
    vec4 emission_st;
    vec4 orm_invert;  // x ao, y roughness, z metallic: 1 = invert sampled value
    vec4 parallax;    // x = displacement scale (0 = off)
    vec4 water;       // x = water body slot (FEAT_WATER)
    vec4 detail;      // x albedo strength, y normal strength, z ORM strength, w blend (0 x2, 1 overlay, 2 add, 3 replace)
    vec4 detail_st;   // xy tiling, zw offset
    vec4 trans0;      // x translucency, y power, z distortion, w map holds thickness
    vec4 trans1;      // rgb translucency colour, w see-through transmission
    vec4 trans2;      // rgb transmission tint, w refraction strength
    vec4 trans3;      // x rough-transmission blur, y thickness, z map G holds transmission
    vec4 tess0;       // tessellation: x displacement scale (m), y mid level, z target edge length (px), w max factor
    vec4 tess1;       // x phong smoothing (0..1), y > 0.5 when the material is tessellated
} fx;

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

// UV of local point `xz` in a body's ripple window (outside 0..1 when it has none).
vec2 water_ripple_uv(int slot, vec2 xz) {
    vec4 r = frame.water_bodies[slot].ripple;
    return r.z > 0.0 ? (xz - r.xy) / r.z + 0.5 : vec2(-1.0);
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
void water_waves(int slot, vec2 xz, float t, float foot, out vec3 disp, out vec3 nrm, out float crest, out float lost) {
    disp = vec3(0.0);
    nrm = vec3(0.0, 1.0, 0.0);
    crest = 0.0;
    lost = 0.0;
    float crest_max = 0.0;
    int n = min(int(frame.water_bodies[slot].look1.z + 0.5), MAX_WATER_WAVES);
    for (int i = 0; i < n; ++i) {
        vec4 a = frame.water_bodies[slot].wave_a[i];
        vec4 b = frame.water_bodies[slot].wave_b[i];
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
    nrm = normalize(nrm);
    crest = crest_max > 1e-5 ? crest / crest_max : 0.0;
}

// Zestlet viewport modes (frame.debug.y 8..11): colours from an integer id and from a DAG
// level. Pure arithmetic, no tables; only the debug branch of standard.frag / the clustered
// standard.vert calls them. Mirrored by `zestlet_view.rs` (tested there).
vec3 zestlet_hash_colour(uint id) {
    uint h = id * 2654435761u;
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    h *= 3266489917u;
    h ^= h >> 16;
    return 0.5 + 0.5 * cos(6.2831853 * (float(h & 1023u) / 1024.0 + vec3(0.0, 0.33, 0.67)));
}

vec3 zestlet_level_colour(uint level) {
    return 0.5 + 0.5 * cos(6.2831853 * (fract(float(level) * 0.137 + 0.05) + vec3(0.0, 0.33, 0.67)));
}
