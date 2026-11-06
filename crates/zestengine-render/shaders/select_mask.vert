// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// Selection mask: the selected draws' geometry as flat coverage (no lighting).
// Position + the per-instance model matrix only, so every mesh kind (static,
// GPU-skinned, morphed, cloth-patched, instanced) draws the same way: the
// vertex buffer a mesh handle owns is already the deformed one.

layout(location = 0) in vec3 a_position;
layout(location = 8) in vec4 a_model0;
layout(location = 9) in vec4 a_model1;
layout(location = 10) in vec4 a_model2;
layout(location = 11) in vec4 a_model3;

layout(push_constant) uniform Push {
    mat4 view_proj;
    vec4 params; // x = selection strength written to the mask (1 primary, 0.5 secondary)
} pc;

void main() {
    // a_model0.w carries per-draw light bits for the lit path: not geometry.
    mat4 model = mat4(vec4(a_model0.xyz, 0.0), a_model1, a_model2, a_model3);
    gl_Position = pc.view_proj * model * vec4(a_position, 1.0);
}
