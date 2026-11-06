// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// Tessellation control stage of a displaced (tessellated) material: sets the
// subdivision of each triangle patch from the screen-space length of its edges
// (docs/shaders.md, "Tessellation and displacement"). One source for the camera
// passes (main, depth prepass, XR, planar) and, with -DSHADOW, for the shadow
// pass, which receives the same world-space corners from shadow.vert -DZE_TESS.
//
// Each edge's level depends only on that edge's two world positions and the
// camera, so two patches that share an edge pick the same level and the
// tessellated surface has no cracks. The shadow pass reads the SAME camera
// (the frame UBO's main view), so a caster is subdivided exactly as the camera
// sees it and its shadow matches the displaced surface.
#include "standard_common.glsl"

layout(set = 1, binding = 16) uniform sampler2D t_displacement;

layout(vertices = 3) out;

#ifdef SHADOW
layout(location = 0) in vec3 v_world_pos[];
layout(location = 1) in vec3 v_normal[];
layout(location = 2) in vec2 v_uv[];
layout(location = 3) flat in vec4 v_lvp0[];
layout(location = 4) flat in vec4 v_lvp1[];
layout(location = 5) flat in vec4 v_lvp2[];
layout(location = 6) flat in vec4 v_lvp3[];

layout(location = 0) out vec3 c_world_pos[];
layout(location = 1) out vec3 c_normal[];
layout(location = 2) out vec2 c_uv[];
layout(location = 3) flat out vec4 c_lvp0[];
layout(location = 4) flat out vec4 c_lvp1[];
layout(location = 5) flat out vec4 c_lvp2[];
layout(location = 6) flat out vec4 c_lvp3[];
#else
layout(location = 0) in vec3 v_world_pos[];
layout(location = 1) in vec3 v_normal[];
layout(location = 2) in vec2 v_uv[];
layout(location = 3) in vec4 v_color[];
layout(location = 4) in vec4 v_tangent[];
layout(location = 5) in vec2 v_uv1[];
layout(location = 6) flat in float v_lightmap[];
layout(location = 7) in vec4 v_cur_clip[];
layout(location = 8) in vec4 v_prev_clip[];
layout(location = 9) in vec4 v_instance_color[];
layout(location = 10) in vec4 v_instance_param[];

layout(location = 0) out vec3 c_world_pos[];
layout(location = 1) out vec3 c_normal[];
layout(location = 2) out vec2 c_uv[];
layout(location = 3) out vec4 c_color[];
layout(location = 4) out vec4 c_tangent[];
layout(location = 5) out vec2 c_uv1[];
layout(location = 6) flat out float c_lightmap[];
layout(location = 7) out vec4 c_cur_clip[];
layout(location = 8) out vec4 c_prev_clip[];
layout(location = 9) out vec4 c_instance_color[];
layout(location = 10) out vec4 c_instance_param[];
#endif
// Per patch: x = mip of the height map to sample, y = world metres per height-map UV unit.
layout(location = 12) patch out vec2 p_info;

// Tessellation level of the edge a-b: its projected length over the target edge length.
float edge_level(vec3 a, vec3 b) {
    float len = distance(a, b);
    float dist = max(distance(frame.camera_pos.xyz, 0.5 * (a + b)), 0.05);
    float height = frame.postfx3.w > 1.0 ? frame.postfx3.w : 1080.0;
    float px = len * (0.5 * height * abs(frame.proj[1][1])) / dist;
    return clamp(px / max(fx.tess0.z, 1.0), 1.0, max(fx.tess0.w, 1.0));
}

void main() {
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    c_world_pos[gl_InvocationID] = v_world_pos[gl_InvocationID];
    c_normal[gl_InvocationID] = v_normal[gl_InvocationID];
    c_uv[gl_InvocationID] = v_uv[gl_InvocationID];
#ifdef SHADOW
    c_lvp0[gl_InvocationID] = v_lvp0[gl_InvocationID];
    c_lvp1[gl_InvocationID] = v_lvp1[gl_InvocationID];
    c_lvp2[gl_InvocationID] = v_lvp2[gl_InvocationID];
    c_lvp3[gl_InvocationID] = v_lvp3[gl_InvocationID];
#else
    c_color[gl_InvocationID] = v_color[gl_InvocationID];
    c_tangent[gl_InvocationID] = v_tangent[gl_InvocationID];
    c_uv1[gl_InvocationID] = v_uv1[gl_InvocationID];
    c_lightmap[gl_InvocationID] = v_lightmap[gl_InvocationID];
    c_cur_clip[gl_InvocationID] = v_cur_clip[gl_InvocationID];
    c_prev_clip[gl_InvocationID] = v_prev_clip[gl_InvocationID];
    c_instance_color[gl_InvocationID] = v_instance_color[gl_InvocationID];
    c_instance_param[gl_InvocationID] = v_instance_param[gl_InvocationID];
#endif
    if (gl_InvocationID == 0) {
        float e0 = edge_level(v_world_pos[1], v_world_pos[2]);
        float e1 = edge_level(v_world_pos[2], v_world_pos[0]);
        float e2 = edge_level(v_world_pos[0], v_world_pos[1]);
        float inner = max(e0, max(e1, e2));
        gl_TessLevelOuter[0] = e0;
        gl_TessLevelOuter[1] = e1;
        gl_TessLevelOuter[2] = e2;
        gl_TessLevelInner[0] = inner;
        // Height-map footprint of one output edge, to pick the mip: texels spanned by the
        // longest UV edge, divided by the subdivision.
        vec2 st = fx.albedo_st.xy;
        float uvl = max(distance(v_uv[0] * st, v_uv[1] * st), max(distance(v_uv[1] * st, v_uv[2] * st), distance(v_uv[2] * st, v_uv[0] * st)));
        float wl = distance(v_world_pos[0], v_world_pos[1]) + distance(v_world_pos[1], v_world_pos[2]) + distance(v_world_pos[2], v_world_pos[0]);
        float uvs = distance(v_uv[0] * st, v_uv[1] * st) + distance(v_uv[1] * st, v_uv[2] * st) + distance(v_uv[2] * st, v_uv[0] * st);
        float texels = uvl * float(textureSize(t_displacement, 0).x) / inner;
        p_info = vec2(log2(max(texels, 1.0)), wl / max(uvs, 1e-6));
    }
}
