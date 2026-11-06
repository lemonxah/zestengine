// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// AO depth prep: full-resolution depth buffer -> half-resolution linear depth
// (mip 0 of the AO depth pyramid), or one level of the pyramid -> the next.
//
// Level 0 keeps the CLOSEST of each 2x2 full-res block (a real surface sample,
// biased to the foreground so silhouettes keep their occluder). Deeper levels
// take one sample of each 2x2 on a rotated grid (Scalable Ambient Obscurance's
// subsampling): a real surface point, never an average floating in mid-air.

#include "ao_common.glsl"

layout(set = 0, binding = 0) uniform sampler2D u_src;

layout(push_constant) uniform Push {
    vec4 lin;   // P22, P32, ortho, mode (0 = from the depth buffer, 1 = pyramid level)
    ivec4 size; // source size xy (texels of the bound source level), level index z
} pc;

layout(location = 0) out float o_depth;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec2 hi = pc.size.xy - 1;
    if (pc.lin.w < 0.5) {
        float d = 1.0;
        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < 2; ++i) {
                d = min(d, texelFetch(u_src, min(p * 2 + ivec2(i, j), hi), 0).r);
            }
        }
        o_depth = ao_linearize(d, pc.lin);
    } else {
        ivec2 q = p * 2 + ivec2((p.y & 1) ^ 1, (p.x & 1) ^ 1);
        o_depth = texelFetch(u_src, min(q, hi), 0).r;
    }
}
