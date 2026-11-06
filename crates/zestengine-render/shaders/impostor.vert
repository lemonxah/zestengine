// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// Octahedral impostor billboards (GEO-5): one camera-facing quad per instance,
// read from a storage buffer (no vertex buffers). The quad frames the model's
// bounding sphere exactly like the capture's views do; its basis is built in
// the model's local space from the direction to the camera (the capture's
// `view_basis`), then rotated into the world. Variants: default = scene pass,
// MULTIVIEW = both XR eyes in one pass.

#ifdef MULTIVIEW
#extension GL_EXT_multiview : require
#include "standard_common.glsl"
#else
layout(set = 0, binding = 0) uniform FrameData {
    mat4 view;
    mat4 proj;
    mat4 view_proj;
    vec4 camera_pos;
    vec4 light_dir;
    vec4 light_color;
    vec4 ambient;
} frame;
#define VIEW_PROJ frame.view_proj
#define CAMERA_POS frame.camera_pos
#endif

struct Inst {
    mat4 model;          // model space (the capture's frame) -> world
    vec4 centre_radius;  // bounding sphere of the captured views, model space
    vec4 tint;           // rgb tint, a = mix factor towards it
};

layout(std430, set = 1, binding = 2) readonly buffer Instances { Inst inst[]; };

layout(push_constant) uniform PC {
    uint first;   // first instance of this atlas's run
    uint grid;    // views per side
    uint pad0;
    uint pad1;
} pc;

layout(location = 0) out vec2 v_uv;
layout(location = 1) flat out vec3 v_dir;    // model -> camera, model space
layout(location = 2) flat out vec4 v_tint;
layout(location = 3) flat out vec3 v_rot0;   // model -> world rotation (columns)
layout(location = 4) flat out vec3 v_rot1;
layout(location = 5) flat out vec3 v_rot2;

const vec2 CORNERS[6] = vec2[6](vec2(-1, 1), vec2(1, 1), vec2(-1, -1), vec2(-1, -1), vec2(1, 1), vec2(1, -1));

void main() {
    Inst g = inst[pc.first + uint(gl_InstanceIndex)];
    mat3 m3 = mat3(g.model);
    vec3 sc = vec3(length(m3[0]), length(m3[1]), length(m3[2]));
    float s = max(max(sc.x, sc.y), max(sc.z, 1e-6));
    mat3 rot = mat3(m3[0] / max(sc.x, 1e-6), m3[1] / max(sc.y, 1e-6), m3[2] / max(sc.z, 1e-6));
    vec3 centre = (g.model * vec4(g.centre_radius.xyz, 1.0)).xyz;
    vec3 to_cam = CAMERA_POS.xyz - centre;
    vec3 dir_w = to_cam / max(length(to_cam), 1e-6);
    vec3 dir = normalize(transpose(rot) * dir_w);
    vec3 hint = abs(dir.y) > 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
    vec3 right = normalize(cross(hint, dir));
    vec3 up = cross(dir, right);
    float radius = g.centre_radius.w * s;
    vec2 c = CORNERS[gl_VertexIndex % 6];
    vec3 world = centre + (rot * right) * (c.x * radius) + (rot * up) * (c.y * radius);
    gl_Position = VIEW_PROJ * vec4(world, 1.0);
    v_uv = vec2(c.x * 0.5 + 0.5, 0.5 - c.y * 0.5);
    v_dir = dir;
    v_tint = g.tint;
    v_rot0 = rot[0];
    v_rot1 = rot[1];
    v_rot2 = rot[2];
}
