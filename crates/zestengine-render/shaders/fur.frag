// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// Fur shells: each shell keeps only the pixels inside a strand. The strand
// grid is the mesh UV scaled by the density; a strand's length is a hash of its
// cell and its width tapers from the root to the tip. Opaque and alpha-tested,
// so the shells need no sorting. The shading is a wrapped key-light term over
// a root-to-tip colour ramp (not the full PBR light loop: no shadows, no GI).

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec3 v_normal;

layout(location = 0) out vec4 o_color;
layout(location = 1) out vec4 o_gbuf;

layout(set = 0, binding = 0) uniform FrameData {
    mat4 view;
    mat4 proj;
    mat4 view_proj;
    vec4 camera_pos;
    vec4 light_dir;
    vec4 light_color;
    vec4 ambient;
    vec4 misc;
} frame;

layout(push_constant) uniform Push {
    mat4 model;
    vec4 base_color;
    vec4 emission;
    vec4 params0;
    vec4 params1;
} pc;

float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

void main() {
    float t = pc.params0.x;
    vec2 g = v_uv * pc.params0.z;
    float h = hash(floor(g));
    float strand_len = mix(0.4, 1.0, h);
    float radius = 0.5 * (1.0 - t / strand_len);
    if (t > strand_len || length(fract(g) - 0.5) > radius) {
        discard;
    }
    vec3 n = normalize(v_normal);
    float ndl = clamp(dot(n, -normalize(frame.light_dir.xyz)), 0.0, 1.0);
    vec3 col = mix(pc.emission.rgb, pc.base_color.rgb, t) * mix(0.35, 1.0, ndl);
    o_color = vec4(col, 1.0);
    o_gbuf = vec4(0.0);
}
