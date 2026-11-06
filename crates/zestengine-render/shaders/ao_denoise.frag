// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// AO denoise, half resolution: a 4x4 depth-aware box (exactly one period of
// the 4x4 noise pattern ao_main.frag rotates by), then, when the temporal
// filter is on, an exponential blend with last frame's result reprojected by
// the camera motion. History is rejected where its stored depth disagrees
// with the reprojected depth (disocclusion) and clamped to the current
// neighbourhood's range (moving objects, which the camera reprojection does
// not follow). Output: rg = (visibility, linear depth), the next history.

#include "ao_common.glsl"

layout(set = 0, binding = 0) uniform sampler2D u_raw;  // ao_main output (point)
layout(set = 0, binding = 1) uniform sampler2D u_lin;  // linear depth, half res (mip 0, point)
layout(set = 0, binding = 2) uniform sampler2D u_hist; // previous output (bilinear)

layout(push_constant) uniform Push {
    // Current AO view space (x, y up, +z forward) -> previous frame's clip space.
    mat4 reproj;
    vec4 view;  // P00, P11, offx, offy
    vec4 p0;    // half size xy, flags (1 ortho, 2 temporal, 4 history valid), blend weight of the new frame
} pc;

layout(location = 0) out vec2 o_ao;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec2 size = pc.p0.xy;
    ivec2 hi = ivec2(size) - 1;
    uint flags = uint(pc.p0.z + 0.5);
    bool ortho = (flags & 1u) != 0u;
    float zc = texelFetch(u_lin, p, 0).r;
    if (zc >= AO_SKY * 0.5) {
        o_ao = vec2(1.0, zc);
        return;
    }
    // Depth tolerance: relative in perspective, absolute-ish in ortho.
    float tol = ortho ? 0.05 : 0.04 * zc;
    // Plane-aware: compare against the depth extrapolated along the local
    // slope (the smaller one-sided step per axis, so an edge does not count),
    // else a grazing plane, whose depth changes by >4% per pixel, rejects its
    // own neighbours and the noise pattern is not averaged out.
    float zl = texelFetch(u_lin, clamp(p + ivec2(-1, 0), ivec2(0), hi), 0).r;
    float zr = texelFetch(u_lin, clamp(p + ivec2(1, 0), ivec2(0), hi), 0).r;
    float zt = texelFetch(u_lin, clamp(p + ivec2(0, -1), ivec2(0), hi), 0).r;
    float zb = texelFetch(u_lin, clamp(p + ivec2(0, 1), ivec2(0), hi), 0).r;
    float gx = abs(zr - zc) < abs(zc - zl) ? zr - zc : zc - zl;
    float gy = abs(zb - zc) < abs(zc - zt) ? zb - zc : zc - zt;
    if (zl >= AO_SKY * 0.5 || zr >= AO_SKY * 0.5) gx = 0.0;
    if (zt >= AO_SKY * 0.5 || zb >= AO_SKY * 0.5) gy = 0.0;
    float sum = 0.0;
    float wsum = 0.0;
    float mn = 1.0;
    float mx = 0.0;
    for (int j = -2; j <= 1; ++j) {
        for (int i = -2; i <= 1; ++i) {
            ivec2 q = clamp(p + ivec2(i, j), ivec2(0), hi);
            float z = texelFetch(u_lin, q, 0).r;
            float a = texelFetch(u_raw, q, 0).r;
            float w = clamp(1.0 - abs(z - (zc + gx * float(i) + gy * float(j))) / tol, 0.0, 1.0);
            sum += a * w;
            wsum += w;
            if (w > 0.5) {
                mn = min(mn, a);
                mx = max(mx, a);
            }
        }
    }
    float ao = wsum > 1e-4 ? sum / wsum : texelFetch(u_raw, p, 0).r;
    mn = min(mn, ao);
    mx = max(mx, ao);

    if ((flags & 6u) == 6u) {
        vec2 uv = (vec2(p) + 0.5) / size;
        vec3 P = ao_view_pos(uv, zc, pc.view, ortho);
        vec4 prev = pc.reproj * vec4(P.x, P.y, -P.z, 1.0);
        vec2 puv = (prev.xy / prev.w) * vec2(0.5, -0.5) + 0.5;
        if (all(greaterThanEqual(puv, vec2(0.0))) && all(lessThanEqual(puv, vec2(1.0)))) {
            vec2 h = texture(u_hist, puv).rg;
            // Previous linear depth of this surface: clip w in perspective.
            float expect = ortho ? zc : prev.w;
            float ptol = ortho ? 0.05 : 0.05 * expect;
            if (abs(h.y - expect) < ptol) {
                float hist = clamp(h.x, mn - 0.02, mx + 0.02);
                ao = mix(hist, ao, pc.p0.w);
            }
        }
    }
    o_ao = vec2(ao, zc);
}
