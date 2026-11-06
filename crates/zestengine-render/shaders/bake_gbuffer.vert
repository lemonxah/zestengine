// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// Lightmap bake, gbuffer pass. Rasterizes a mesh into its lightmap (uv1)
// space: the second UV set becomes the clip-space position, so each lightmap
// texel is filled with the world position + normal of the surface that maps
// there. The compute pass then traces lighting per texel.

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;
layout(location = 3) in vec4 a_color;
layout(location = 4) in vec4 a_tangent;
layout(location = 5) in vec2 a_uv1;

layout(push_constant) uniform Push {
    mat4 model;
} pc;

layout(location = 0) out vec3 v_world_pos;
layout(location = 1) out vec3 v_world_normal;

void main() {
    vec4 world = pc.model * vec4(a_position, 1.0);
    v_world_pos = world.xyz;
    v_world_normal = normalize(mat3(pc.model) * a_normal);
    // uv1 in [0,1] → NDC [-1,1]. The bake viewport has no Y flip, so the
    // texel grid matches the readback row order directly.
    vec2 ndc = a_uv1 * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
