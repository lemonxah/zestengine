// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// Error shader: unmistakable animated magenta/purple swirl for objects whose
// material is missing, failed to load, or references an unknown shader.
// Deliberately ugly and animated so it can't be confused with a pink material.

layout(set = 0, binding = 0) uniform FrameData {
    mat4 view;
    mat4 proj;
    mat4 view_proj;
    vec4 camera_pos;
    vec4 light_dir;
    vec4 light_color;
    vec4 ambient;
    vec4 misc; // x = time in seconds
} frame;

layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec2 v_uv;
layout(location = 3) in vec4 v_color;
layout(location = 4) in vec4 v_tangent;

layout(location = 0) out vec4 o_color;
// Deferred-SSR G-buffer (gbuf pipeline variants only). Error surfaces are not
// reflectors; depth-only variants mask all colour writes so this is ignored.
layout(location = 1) out vec4 o_gbuf;

void main() {
    o_gbuf = vec4(0.0);
    float t = frame.misc.x;
    vec3 wp = v_world_pos * 4.0;
    float r = length(wp.xz) + wp.y * 0.7;
    float a = atan(wp.z, wp.x);
    float swirl = sin(r * 3.0 + a * 2.0 - t * 4.0);
    float pulse = 0.85 + 0.15 * sin(t * 6.0);
    vec3 magenta = vec3(1.0, 0.0, 0.85);
    vec3 purple = vec3(0.35, 0.0, 0.55);
    o_color = vec4(mix(magenta, purple, 0.5 + 0.5 * swirl) * pulse, 1.0);
}
