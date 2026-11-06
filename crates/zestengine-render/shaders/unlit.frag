// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// Built-in unlit shader (world-space UI, particles): albedo * vertex color *
// tint, no lighting. Same interface as custom shaders.
layout(set = 1, binding = 0) uniform sampler2D t_albedo;
layout(push_constant) uniform Push {
    mat4 model;
    vec4 d0;
    vec4 d1;
    vec4 d2;
    vec4 d3;
} pc;
layout(location = 2) in vec2 v_uv;
layout(location = 3) in vec4 v_color;
layout(location = 0) out vec4 o_color;

void main() {
    o_color = texture(t_albedo, v_uv) * v_color;
    if (o_color.a <= 0.002) discard;
}
