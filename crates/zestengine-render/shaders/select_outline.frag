// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// Selection outline composite (fullscreen, over the tonemapped viewport): a
// constant-pixel-width ring OUTSIDE the selection mask. The mask's interior is
// never touched. Parts of the selection hidden behind other geometry get a
// dimmer ring (full strength where the selected surface is visible).

layout(set = 0, binding = 0) uniform sampler2D u_mask;

layout(push_constant) uniform Push {
    vec4 primary;   // rgb outline colour (target space), a opacity
    vec4 secondary; // same for the non-primary selected objects
    vec4 params;    // x width px, y occluded strength
} pc;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec2 sz = textureSize(u_mask, 0);
    if (texelFetch(u_mask, p, 0).r > 0.0) {
        discard; // inside the selection
    }
    float w = max(pc.params.x, 0.5);
    int radius = int(ceil(w));
    int dirs = radius > 2 ? 16 : 8;
    float best = 1e9;
    vec2 hit = vec2(0.0);
    for (int r = 1; r <= radius; r++) {
        for (int i = 0; i < dirs; i++) {
            float a = 6.2831853 * float(i) / float(dirs);
            ivec2 o = ivec2(round(vec2(cos(a), sin(a)) * float(r)));
            ivec2 q = p + o;
            if (q.x < 0 || q.y < 0 || q.x >= sz.x || q.y >= sz.y) continue;
            float d = length(vec2(o));
            if (d >= best) continue;
            vec2 m = texelFetch(u_mask, q, 0).rg;
            if (m.r > 0.0) {
                best = d;
                hit = m;
            }
        }
    }
    float cover = clamp(w - best + 1.0, 0.0, 1.0);
    if (cover <= 0.0) discard;
    vec4 col = hit.r > 0.75 ? pc.primary : pc.secondary;
    float vis = hit.g > 0.5 ? 1.0 : pc.params.y;
    o_color = vec4(col.rgb, col.a * cover * vis);
}
