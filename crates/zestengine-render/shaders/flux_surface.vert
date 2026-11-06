// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// Camera-independent material captures. Card projections live in world space.
layout(location=0) in vec3 a_position;
layout(location=1) in vec3 a_normal;
layout(location=2) in vec2 a_uv;
layout(location=3) in vec4 a_color;
layout(location=4) in vec4 a_tangent;
struct Card { mat4 view_proj; vec4 normal_texel; uvec4 rect; mat4 capture_to_world; };
layout(set=0, binding=0, std430) readonly buffer Cards { Card cards[]; };
layout(push_constant) uniform Push {
    mat4 model;
    vec4 base;
    vec4 emission;
    vec4 material; // metallic, roughness, cutoff, normal strength
    uvec4 info;    // card, features, time bits, reserved
} pc;
layout(location=0) out vec3 world;
layout(location=1) out vec3 normal;
layout(location=2) out vec2 uv;
layout(location=3) out vec4 color;
layout(location=4) out vec4 tangent;
void main() {
    world = (pc.model * vec4(a_position, 1.0)).xyz;
    normal = normalize(transpose(inverse(mat3(pc.model))) * a_normal);
    tangent = vec4(mat3(pc.model) * a_tangent.xyz, a_tangent.w * sign(determinant(mat3(pc.model))));
    uv = a_uv;
    color = a_color;
    gl_Position = cards[pc.info.x].view_proj * vec4(world, 1.0);
}
