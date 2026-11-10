// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// Underwater view: drawn fullscreen over the lit HDR scene (after the forward
// pass, before reflections / post) when the camera is in a water body. Two
// blended draws share this shader: mode 0 multiplies the scene by the
// transmittance along the underwater part of each view ray (plus the light
// lost on its way down; the caustics are already on the lit surfaces, see
// standard.frag's water_caustic_gain), mode 1 adds the water's
// in-scattered colour. The water's waves are evaluated per pixel, so a near
// plane straddling the surface gets a waterline that rides the waves, and the
// fog's surface cut and the depth darkening follow the moving surface.

layout(set = 0, binding = 0) uniform sampler2D u_depth;

#include "water_types.glsl"
layout(set = 0, binding = 1) uniform Waves {
    mat4 to_world;
    vec4 info; // x rest surface fallback, y wave count, z water clock
    GpuWater body;
    vec4 absorption; // IOR zero retains legacy authored visibility
    vec4 scattering;
    vec4 illumination;
} wv;
layout(std430, set = 0, binding = 2) readonly buffer Coverage {
    GpuWater bodies[MAX_WATER_BODIES];
    vec4 data[];
} scene;
#define WATER_COVERAGE_DATA scene.data
#include "water_coverage.glsl"
#include "water_exclusions.glsl"
#include "water_path.glsl"
#undef WATER_COVERAGE_DATA
#define WATER_OCEAN_DATA scene.data
#include "water_waves.glsl"
#define WATER_BODY_DATA scene.bodies
#define WATER_COVERAGE_DATA scene.data
#include "water_connections.glsl"
#include "water_ownership.glsl"
#include "water_media.glsl"

layout(push_constant) uniform Underwater {
    mat4 inv_view_proj;
    vec4 camera;  // xyz camera position, w time
    vec4 water;   // x surface height, y extinction / m, z light falloff / m, w unused
    vec4 deep;    // rgb lit deep colour, w mode (0 multiply, 1 add)
    vec4 shallow; // rgb lit shallow colour
} pc;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

// Red goes first under water, blue last.
const vec3 SPECTRAL = vec3(1.55, 1.0, 0.8);

// Gerstner displacement of the rest point `xz` (local), as in water_waves().
vec3 wave_disp(vec2 xz) {
    vec3 d,n;float crest,lost;
    water_connected_waves(wv.body,xz,wv.info.z,0.0,d,n,crest,lost);
    return d*water_connected_shore(wv.body,xz,water_shore_field(wv.body,xz));
}

// World height of the moving surface over world point `p` (solving for the
// rest point the waves carry there, like Water::surface).
float surface_y(vec3 p) {
    vec3 l = (wv.body.to_local * vec4(p, 1.0)).xyz;
    vec2 rest = l.xz;
    for (int k = 0; k < 3; ++k) {
        rest = l.xz - wave_disp(rest).xz;
    }
    vec4 column; float edge;
    bool covered = water_column(wv.body, l.xz, column, edge);
    edge=water_connected_edge(wv.body,l.xz,edge);
    float h = (covered ? column.x : wv.info.x) + wave_disp(rest).y * (1.0 - edge * edge);
    return (wv.to_world * vec4(l.x, h, l.z, 1.0)).y;
}

vec3 unproject(vec2 uv, float d) {
    // The scene renders with a Y-flipped viewport (see ssr_resolve.frag).
    vec2 ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    vec4 p = pc.inv_view_proj * vec4(ndc, d, 1.0);
    return p.xyz / p.w;
}

void main() {
    bool add = pc.deep.w > 0.5;
    vec3 ro = pc.camera.xyz;
    float h = surface_y(ro);

    float d = texelFetch(u_depth, ivec2(gl_FragCoord.xy), 0).r;
    bool sky = d >= 0.9999;
    vec3 far_p = unproject(v_uv, sky ? 1.0 : d);

    vec3 near_p = unproject(v_uv, 0.0);
    float under = 1.0 - smoothstep(-0.0015, 0.0015, near_p.y - surface_y(near_p));
    if (under <= 0.0) {
        o_color = add ? vec4(0.0) : vec4(1.0);
        return;
    }

    vec3 ray = far_p - ro;
    float dist = sky ? 1e4 : length(ray);
    vec3 rd = ray / max(length(ray), 1e-6);
    if(water_multiple_media()){
        vec3 trans,fog;float wet;water_media_trace(ro,rd,vec2(0.0,dist),true,trans,fog,wet);
        if(add){o_color=vec4(fog*wv.illumination.rgb*under,0.0);}
        else{if(!sky)trans*=water_media_receiver(far_p);o_color=vec4(mix(vec3(1.0),trans,under),1.0);}
        return;
    }
    // Only the part of the ray below the surface (looking up through it, the
    // air beyond isn't water).
    float t0 = 0.0, t1 = dist;
    if (abs(rd.y) > 1e-5) {
        // Where the ray meets the surface: start from the level over the
        // camera, then re-read the wave height where it lands.
        float ts = (h - ro.y) / rd.y;
        for (int k = 0; k < 2; ++k) {
            float hs = surface_y(ro + rd * max(ts, 0.0));
            ts = (hs - ro.y) / rd.y;
        }
        if (rd.y > 0.0) t1 = min(t1, ts);
        else t0 = max(t0, ts);
    } else if (ro.y > h) {
        t1 = t0;
    }
    // A river can rise, fall, leave and re-enter the view ray. Its own
    // channel intersections replace the single surface plane interval.
    if (wv.body.coverage.x == 3u || wv.body.coverage.x == 4u) { t0 = 0.0; t1 = dist; }
    t0 = max(t0, 0.0); t1 = max(t1, t0);
    vec3 local_origin = (wv.body.to_local * vec4(ro, 1.0)).xyz;
    vec3 local_dir = mat3(wv.body.to_local) * rd;
    float seg = water_medium_distance(wv.body, local_origin, local_dir, vec2(t0, t1));
    vec3 extinction = wv.absorption.w >= 1.0 ? max(wv.absorption.rgb + wv.scattering.rgb, vec3(0.0)) : pc.water.y * SPECTRAL;
    vec3 trans = exp(-extinction * seg);

    if (add) {
        float cam_depth = max(h - ro.y, 0.0);
        // Brighter toward the surface, darker down into the deep.
        vec3 base = mix(pc.shallow.rgb, pc.deep.rgb, smoothstep(-0.5, 0.6, -rd.y));
        vec3 light_extinction = wv.absorption.w >= 1.0 ? extinction : pc.water.z * SPECTRAL;
        vec3 fog = base * exp(-light_extinction * cam_depth);
        o_color = vec4(fog * (1.0 - trans) * under, 0.0);
        return;
    }

    vec3 m = trans;
    if (!sky) {
        vec3 local_end = (wv.body.to_local * vec4(far_p, 1.0)).xyz;
        vec4 end_column; float end_edge;
        bool in_body = water_column(wv.body, local_end.xz, end_column, end_edge);
        float depth_p = in_body && local_end.y >= end_column.x-end_column.y ? max(surface_y(far_p) - far_p.y, 0.0) : 0.0;
        if (depth_p > 0.0 && !water_excluded(wv.body, (wv.body.to_local * vec4(far_p, 1.0)).xyz)) {
            m *= exp(-pc.water.z * SPECTRAL * depth_p);
        }
    }
    o_color = vec4(mix(vec3(1.0), m, under), 1.0);
}
