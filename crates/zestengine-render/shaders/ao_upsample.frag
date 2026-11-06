// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// AO upsample: half-resolution denoised AO -> full resolution, depth-aware
// (joint bilateral). Each full-res pixel blends its 4 nearest half-res texels
// by the bilinear weight times how close each texel's depth is to its own, so
// occlusion never bleeds across a silhouette. The forward pass samples the
// result per pixel (standard.frag, set 0 binding 13).

#include "ao_common.glsl"

layout(set = 0, binding = 0) uniform sampler2D u_ao;    // rg = (visibility, linear depth), half res
layout(set = 0, binding = 1) uniform sampler2D u_depth; // full-res depth buffer

layout(push_constant) uniform Push {
    vec4 lin;  // P22, P32, ortho, _
    vec4 size; // half size xy, full size zw
} pc;

layout(location = 0) out float o_ao;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float d = texelFetch(u_depth, p, 0).r;
    if (d >= 1.0) {
        o_ao = 1.0;
        return;
    }
    float z = ao_linearize(d, pc.lin);
    vec2 hp = (vec2(p) + 0.5) * 0.5 - 0.5;
    ivec2 base = ivec2(floor(hp));
    vec2 f = hp - vec2(base);
    ivec2 hi = ivec2(pc.size.xy) - 1;
    float tol = pc.lin.z > 0.5 ? 0.02 : 0.02 * z;
    float sum = 0.0;
    float wsum = 0.0;
    float best = 1.0;
    float best_dz = 1e30;
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
            vec2 t = texelFetch(u_ao, clamp(base + ivec2(i, j), ivec2(0), hi), 0).rg;
            float bw = (i == 0 ? 1.0 - f.x : f.x) * (j == 0 ? 1.0 - f.y : f.y);
            float dz = abs(t.y - z);
            float w = (bw + 1e-3) * exp2(-dz / tol);
            sum += t.x * w;
            wsum += w;
            if (dz < best_dz) {
                best_dz = dz;
                best = t.x;
            }
        }
    }
    o_ao = wsum > 1e-5 ? sum / wsum : best;
}
