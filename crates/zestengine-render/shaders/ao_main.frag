// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// Screen-space ambient occlusion, half resolution, from the AO depth pyramid
// (ao_prep.frag). Two quality modes (specialization constant QUALITY):
//
//   0 Low  - Scalable Ambient Obscurance (McGuire et al. 2012): a spiral of
//            taps over the depth mips, obscurance by the classic falloff
//            f(v) = max(r^2 - |v|^2, 0)^3 * max(v.n - bias, 0) / (|v|^2 + eps).
//   1 High - Ground-Truth Ambient Occlusion (Jimenez et al. 2016, following
//            Intel's XeGTAO): per pixel, a few slices through the view vector;
//            each finds the two maximum horizon angles by marching the depth,
//            then integrates the cosine-weighted visible arc analytically
//            against the projected normal. Distance falloff fades far
//            occluders out so thin geometry does not black out the scene.
//
// Normals are reconstructed from the depth (the G-buffer is written by the
// forward pass, after this runs). Output: visibility in [0, 1] (1 = open).

#include "ao_common.glsl"

layout(constant_id = 0) const int QUALITY = 1;

// Linear depth pyramid (mip 0 = half resolution), point-sampled.
layout(set = 0, binding = 0) uniform sampler2D u_lin;

layout(push_constant) uniform Push {
    vec4 view;  // P00, P11, offx, offy (see ao_view_pos)
    vec4 p0;    // radius (world), intensity, falloff (fraction of radius), pixels per unit at depth 1
    vec4 p1;    // half-res size xy, frame index, flags (1 = ortho, 2 = temporal noise)
    vec4 p2;    // max screen radius (px), mip count, _, _
} pc;

layout(location = 0) out float o_ao;

const int GTAO_SLICES = 4;
const int GTAO_STEPS = 6;
const float GTAO_MIP_BIAS = 3.3;
const float GTAO_HORIZON_EPS = 0.03;
const int SAO_TAPS = 12;
// Calibrated so a 90 degree corner lands near GTAO's (~0.4-0.5) at defaults.
const float SAO_STRENGTH = 2.0;
const float SAO_TURNS = 7.0;

bool g_ortho;

float fast_acos(float x) {
    float ax = abs(x);
    float r = (-0.156583 * ax + AO_HALF_PI) * sqrt(max(1.0 - ax, 0.0));
    return x >= 0.0 ? r : AO_PI - r;
}

vec2 g_size;
vec2 g_texel;

vec3 pos_at(vec2 uv, float mip) {
    float z = textureLod(u_lin, uv, mip).r;
    return ao_view_pos(uv, z, pc.view, g_ortho);
}

float gtao(vec2 uv, vec3 P, vec3 N, float screen_r, float noise_slice, float noise_step) {
    vec3 V = g_ortho ? vec3(0.0, 0.0, -1.0) : normalize(-P);
    float radius = pc.p0.x;
    float falloff_range = max(pc.p0.z, 0.05) * radius;
    float falloff_from = radius - falloff_range;
    float falloff_mul = -1.0 / falloff_range;
    float falloff_add = falloff_from / falloff_range + 1.0;
    // At least ~1.3 px from the centre so the first tap never hits itself.
    float min_s = 1.3 / screen_r;
    float vis = 0.0;
    for (int s = 0; s < GTAO_SLICES; ++s) {
        float phi = (float(s) + noise_slice) * AO_PI / float(GTAO_SLICES);
        float cphi = cos(phi);
        float sphi = sin(phi);
        vec2 omega = vec2(cphi, -sphi) * screen_r; // pixels, +y down
        vec3 dir = vec3(cphi, sphi, 0.0);          // view space, +y up
        vec3 ortho_dir = dir - dot(dir, V) * V;
        vec3 axis = normalize(cross(ortho_dir, V));
        vec3 proj_n = N - axis * dot(N, axis);
        float proj_len = length(proj_n);
        float sgn_n = sign(dot(ortho_dir, proj_n));
        float cos_n = clamp(dot(proj_n, V) / max(proj_len, 1e-4), 0.0, 1.0);
        float n = sgn_n * fast_acos(cos_n);
        float low0 = cos(n + AO_HALF_PI);
        float low1 = cos(n - AO_HALF_PI);
        float hc0 = low0;
        float hc1 = low1;
        for (int j = 0; j < GTAO_STEPS; ++j) {
            float t = (float(j) + noise_step) / float(GTAO_STEPS);
            t = t * t + min_s;
            vec2 off = round(omega * t);
            float mip = clamp(log2(length(off)) - GTAO_MIP_BIAS, 0.0, pc.p2.y - 1.0);
            vec2 d_uv = off * g_texel;
            vec3 S0 = pos_at(uv + d_uv, mip);
            vec3 S1 = pos_at(uv - d_uv, mip);
            vec3 d0 = S0 - P;
            vec3 d1 = S1 - P;
            float l0 = length(d0);
            float l1 = length(d1);
            // Small epsilon: depth-reconstructed normals are noisy, so a tap
            // lying within a couple of degrees of the tangent plane is the
            // surface itself, not an occluder (grazing planes self-occlude).
            // Guard, don't subtract: occluders above the epsilon count at
            // their full cosine, so corners stay dark.
            float shc0 = dot(d0, V) / max(l0, 1e-6);
            float shc1 = dot(d1, V) / max(l1, 1e-6);
            float w0 = clamp(l0 * falloff_mul + falloff_add, 0.0, 1.0);
            float w1 = clamp(l1 * falloff_mul + falloff_add, 0.0, 1.0);
            // Off-screen or sky taps carry no occluder.
            if (S0.z >= AO_SKY * 0.5 || any(lessThan(uv + d_uv, vec2(0.0))) || any(greaterThan(uv + d_uv, vec2(1.0)))) w0 = 0.0;
            if (S1.z >= AO_SKY * 0.5 || any(lessThan(uv - d_uv, vec2(0.0))) || any(greaterThan(uv - d_uv, vec2(1.0)))) w1 = 0.0;
            if (shc0 > GTAO_HORIZON_EPS) {
                hc0 = max(hc0, mix(low0, shc0, w0));
            }
            if (shc1 > GTAO_HORIZON_EPS) {
                hc1 = max(hc1, mix(low1, shc1, w1));
            }
        }
        float h0 = -fast_acos(clamp(hc1, -1.0, 1.0));
        float h1 = fast_acos(clamp(hc0, -1.0, 1.0));
        h0 = n + clamp(h0 - n, -AO_HALF_PI, AO_HALF_PI);
        h1 = n + clamp(h1 - n, -AO_HALF_PI, AO_HALF_PI);
        float sin_n = sin(n);
        float iarc0 = (cos_n + 2.0 * h0 * sin_n - cos(2.0 * h0 - n)) * 0.25;
        float iarc1 = (cos_n + 2.0 * h1 * sin_n - cos(2.0 * h1 - n)) * 0.25;
        vis += proj_len * (iarc0 + iarc1);
    }
    vis = clamp(vis / float(GTAO_SLICES), 0.0, 1.0);
    return pow(vis, max(pc.p0.y, 0.0));
}

float sao(vec2 uv, vec3 P, vec3 N, float screen_r, float spin) {
    float radius = pc.p0.x;
    float r2 = radius * radius;
    // Bias grows with distance (depth precision), epsilon keeps 1/|v|^2 bounded.
    float bias = 0.002 * P.z + 0.01 * radius;
    const float eps = 0.01;
    float sum = 0.0;
    for (int i = 0; i < SAO_TAPS; ++i) {
        float alpha = (float(i) + 0.5) / float(SAO_TAPS);
        float angle = alpha * SAO_TURNS * 2.0 * AO_PI + spin;
        float h = max(screen_r * alpha, 1.3);
        vec2 off = round(vec2(cos(angle), -sin(angle)) * h);
        float mip = clamp(floor(log2(h)) - 3.0, 0.0, pc.p2.y - 1.0);
        vec2 suv = uv + off * g_texel;
        if (any(lessThan(suv, vec2(0.0))) || any(greaterThan(suv, vec2(1.0)))) continue;
        vec3 Q = pos_at(suv, mip);
        if (Q.z >= AO_SKY * 0.5) continue;
        vec3 v = Q - P;
        float vv = dot(v, v);
        float vn = dot(v, N);
        float f = max(r2 - vv, 0.0);
        sum += f * f * f * max((vn - bias) / (eps + vv), 0.0);
    }
    float intensity_div_r6 = max(pc.p0.y, 0.0) / (r2 * r2 * r2);
    return clamp(1.0 - sum * intensity_div_r6 * (SAO_STRENGTH / float(SAO_TAPS)), 0.0, 1.0);
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    g_size = pc.p1.xy;
    g_texel = 1.0 / g_size;
    uint flags = uint(pc.p1.w + 0.5);
    g_ortho = (flags & 1u) != 0u;
    ivec2 hi = ivec2(g_size) - 1;
    vec2 uv = (vec2(p) + 0.5) * g_texel;

    float zc = texelFetch(u_lin, p, 0).r;
    if (zc >= AO_SKY * 0.5) {
        o_ao = 1.0;
        return;
    }
    vec3 P = ao_view_pos(uv, zc, pc.view, g_ortho);

    // Normal from the depth: per axis, the neighbour on the same surface (the
    // smaller depth step) so an edge does not bend the normal.
    float zl = texelFetch(u_lin, clamp(p + ivec2(-1, 0), ivec2(0), hi), 0).r;
    float zr = texelFetch(u_lin, clamp(p + ivec2(1, 0), ivec2(0), hi), 0).r;
    float zt = texelFetch(u_lin, clamp(p + ivec2(0, -1), ivec2(0), hi), 0).r;
    float zb = texelFetch(u_lin, clamp(p + ivec2(0, 1), ivec2(0), hi), 0).r;
    vec3 dx = abs(zr - zc) < abs(zc - zl)
        ? ao_view_pos(uv + vec2(g_texel.x, 0.0), zr, pc.view, g_ortho) - P
        : P - ao_view_pos(uv - vec2(g_texel.x, 0.0), zl, pc.view, g_ortho);
    vec3 dy = abs(zb - zc) < abs(zc - zt)
        ? ao_view_pos(uv + vec2(0.0, g_texel.y), zb, pc.view, g_ortho) - P
        : P - ao_view_pos(uv - vec2(0.0, g_texel.y), zt, pc.view, g_ortho);
    vec3 N = normalize(cross(dx, dy));
    vec3 to_eye = g_ortho ? vec3(0.0, 0.0, -1.0) : -P;
    if (dot(N, to_eye) < 0.0) {
        N = -N;
    }

    // Screen-space radius (half-res pixels) of the world radius at this depth.
    float screen_r = pc.p0.x * pc.p0.w / (g_ortho ? 1.0 : zc);
    if (screen_r < 1.0) {
        o_ao = 1.0;
        return;
    }
    screen_r = min(screen_r, pc.p2.x);

    // Noise: a 4x4 ordered pattern (the 4x4 denoise averages it out), rotated
    // per frame when the temporal filter accumulates.
    float frame = (flags & 2u) != 0u ? pc.p1.z : 0.0;
    float noise_slice = fract(ao_bayer4(p) + frame * 0.618034);
    float noise_step = fract(ao_ign(vec2(p)) + frame * 0.754878);

    if (QUALITY == 0) {
        o_ao = sao(uv, P, N, screen_r, noise_slice * 2.0 * AO_PI);
    } else {
        o_ao = gtao(uv, P, N, screen_r, noise_slice, noise_step);
    }
}
