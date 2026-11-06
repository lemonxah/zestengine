// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// Fur shells: the mesh re-drawn once per shell, each pushed out along its
// normals by length * t. Same vertex inputs and per-instance model matrix as
// outline.vert; the shell index arrives in the push constants (params0.x).

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;
layout(location = 3) in vec4 a_color;
layout(location = 4) in vec4 a_tangent;

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

layout(location = 8) in vec4 a_model0;
layout(location = 9) in vec4 a_model1;
layout(location = 10) in vec4 a_model2;
layout(location = 11) in vec4 a_model3;

layout(push_constant) uniform Push {
    mat4 model;
    vec4 base_color; // rgb tip colour
    vec4 emission;   // rgb root colour
    vec4 params0;    // x shell t (0..1), y length (world units), z density (strands per uv unit)
    vec4 params1;
} pc;

layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec3 v_normal;

void main() {
    mat4 model = mat4(vec4(a_model0.xyz, 0.0), a_model1, a_model2, a_model3);
    vec3 n = normalize(mat3(model) * a_normal);
    vec4 world = model * vec4(a_position, 1.0);
    world.xyz += n * (pc.params0.y * pc.params0.x);
    gl_Position = frame.view_proj * world;
    v_uv = a_uv;
    v_normal = n;
}
