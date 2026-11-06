// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// Deferred projected-decal pass. One fullscreen draw per decal: it
// reconstructs the opaque surface's world position from the scene depth,
// projects that point into the decal's local box, and alpha-blends the decal
// texture onto the already-lit HDR colour. Opaque surfaces only: transparent
// draws never write depth, and the water surface clears its G-buffer normal.

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

layout(set = 0, binding = 0) uniform sampler2D u_depth; // scene depth (opaque)
layout(set = 0, binding = 1) uniform sampler2D u_gbuf;  // octahedral view-space normal
layout(set = 0, binding = 2) uniform DecalFrame {
    mat4 inv_view_proj; // world = inv_view_proj * clip
    mat4 inv_view;      // view normal -> world normal
} frame;

layout(set = 1, binding = 0) uniform sampler2D u_decal;

layout(push_constant) uniform DecalPush {
    mat4 world_to_local; // inverse of the projector's world transform
    vec4 size_depth;     // x width, y height, z depth
    vec4 fade_tiling;    // x fade, y tiling.x, z tiling.y
    vec4 tint;           // rgba
    vec4 forward;        // xyz world-space projector +Z (toward the surface)
} pc;

const bool FLIP_Y = true;

// Octahedral decode of the view-space normal the forward pass stored (rg).
vec3 oct_decode(vec2 e) {
    e = e * 2.0 - 1.0;
    vec3 n = vec3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    float t = max(-n.z, 0.0);
    n.x += n.x >= 0.0 ? -t : t;
    n.y += n.y >= 0.0 ? -t : t;
    return normalize(n);
}

vec3 world_pos_at(vec2 uv) {
    float d = texture(u_depth, uv).r;
    vec2 ndc = uv * 2.0 - 1.0;
    if (FLIP_Y) ndc.y = -ndc.y;
    vec4 w = frame.inv_view_proj * vec4(ndc, d, 1.0);
    return w.xyz / w.w;
}

vec3 world_normal_at(vec2 uv) {
    vec3 nv = oct_decode(texture(u_gbuf, uv).rg);
    return normalize((frame.inv_view * vec4(nv, 0.0)).xyz);
}

void main() {
    float d = texture(u_depth, v_uv).r;
    if (d >= 1.0) discard; // far plane: sky / no opaque surface

    vec3 world = world_pos_at(v_uv);
    vec3 local = (pc.world_to_local * vec4(world, 1.0)).xyz;
    float width = max(pc.size_depth.x, 1e-4);
    float height = max(pc.size_depth.y, 1e-4);
    float depth = max(pc.size_depth.z, 1e-4);
    if (local.x < -width * 0.5 || local.x > width * 0.5
        || local.y < -height * 0.5 || local.y > height * 0.5
        || local.z < -depth || local.z > 0.0) {
        discard;
    }

    vec2 uv = vec2(local.x / width + 0.5, local.y / height + 0.5);
    uv *= max(pc.fade_tiling.yz, vec2(1e-4));
    vec4 tex = texture(u_decal, uv);

    // Fade toward the far end of the box (along the projector's -Z).
    float fade_start = depth * (1.0 - clamp(pc.fade_tiling.x, 0.0, 1.0));
    float depth_fade = 1.0 - smoothstep(fade_start, depth, -local.z);

    // Fade out on surfaces facing away from the projector.
    vec3 n = world_normal_at(v_uv);
    float angle = smoothstep(0.0, 0.4, clamp(dot(n, pc.forward.xyz), 0.0, 1.0));

    float alpha = tex.a * depth_fade * angle;
    if (alpha <= 0.0) discard;
    o_color = vec4(tex.rgb * pc.tint.rgb, alpha * clamp(pc.tint.a, 0.0, 1.0));
}
