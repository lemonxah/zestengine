// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// Selection outline: inverted hull. The mesh is re-drawn inflated along its
// normals with front faces culled, leaving a silhouette border.

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

// Per-instance model matrix (instance-rate vertex stream), so draws of one
// mesh + material batch into a single instanced draw (pc.model is unused).
layout(location = 8) in vec4 a_model0;
layout(location = 9) in vec4 a_model1;
layout(location = 10) in vec4 a_model2;
layout(location = 11) in vec4 a_model3;

layout(push_constant) uniform Push {
    mat4 model;
    vec4 base_color;
    vec4 emission;
    vec4 params0; // xyz = mesh AABB center (object space)
    vec4 params1; // y = width from vertex colour, w = highlight strength (scales outline width)
} pc;

void main() {
    mat4 model = mat4(vec4(a_model0.xyz, 0.0), a_model1, a_model2, a_model3);
    // Inflate radially from the mesh center: the direction depends only on
    // position, so vertices duplicated across hard edges (per-face normals)
    // move together and the hull stays watertight, with no gaps at cube corners.
    // Concave regions, where the radial direction leaves the surface, fall
    // back to the vertex normal.
    vec3 radial = a_position - pc.params0.xyz;
    vec3 dir = (length(radial) > 1e-5 && dot(normalize(radial), a_normal) > 0.05)
        ? normalize(radial)
        : a_normal;
    vec3 world_dir = normalize(mat3(model) * dir);
    vec4 world = model * vec4(a_position, 1.0);
    // Roughly constant screen-space width: scale with view distance.
    float dist = length(frame.camera_pos.xyz - world.xyz);
    // params1.y > 0.5: per-vertex width from the red vertex colour.
    float vertex_width = pc.params1.y > 0.5 ? a_color.r : 1.0;
    world.xyz += world_dir * 0.005 * dist * pc.params1.w * vertex_width;
    gl_Position = frame.view_proj * world;
}
