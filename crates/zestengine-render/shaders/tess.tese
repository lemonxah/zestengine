// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
#ifdef MULTIVIEW
#extension GL_EXT_multiview : require
#endif
// Tessellation evaluation stage of a displaced material (see tess.tesc): places
// each generated vertex on the patch, smooths it with Phong tessellation
// (every corner's tangent plane pulls the point toward the curved surface),
// then displaces it along the smoothed normal by the height map. The main,
// depth-prepass, XR and planar passes run this exact code with an `invariant`
// position, so the depth prepass and the EQUAL-tested main pass agree
// bit for bit. -DSHADOW runs the same displacement and projects it with the
// light's matrix instead (tessellation stages can't read push constants, so
// shadow.vert -DZE_TESS passes the matrix along as flat varyings).
#include "standard_common.glsl"

layout(set = 1, binding = 16) uniform sampler2D t_displacement;

// `cw`: Vulkan's tessellation domain has its origin at the upper left, which turns the generated
// triangles around relative to GL; cw keeps them counter-clockwise like the patch they came from.
layout(triangles, equal_spacing, cw) in;

#ifdef SHADOW
layout(location = 0) in vec3 c_world_pos[];
layout(location = 1) in vec3 c_normal[];
layout(location = 2) in vec2 c_uv[];
layout(location = 3) flat in vec4 c_lvp0[];
layout(location = 4) flat in vec4 c_lvp1[];
layout(location = 5) flat in vec4 c_lvp2[];
layout(location = 6) flat in vec4 c_lvp3[];
#else
layout(location = 0) in vec3 c_world_pos[];
layout(location = 1) in vec3 c_normal[];
layout(location = 2) in vec2 c_uv[];
layout(location = 3) in vec4 c_color[];
layout(location = 4) in vec4 c_tangent[];
layout(location = 5) in vec2 c_uv1[];
layout(location = 6) flat in float c_lightmap[];
layout(location = 7) in vec4 c_cur_clip[];
layout(location = 8) in vec4 c_prev_clip[];
layout(location = 9) in vec4 c_instance_color[];
layout(location = 10) in vec4 c_instance_param[];

layout(location = 0) out vec3 v_world_pos;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec2 v_uv;
layout(location = 3) out vec4 v_color;
layout(location = 4) out vec4 v_tangent;
layout(location = 5) out vec2 v_uv1;
layout(location = 6) flat out float v_lightmap;
layout(location = 7) out vec4 v_cur_clip;
layout(location = 8) out vec4 v_prev_clip;
layout(location = 9) out vec4 v_instance_color;
layout(location = 10) out vec4 v_instance_param;
#endif
layout(location = 12) patch in vec2 p_info;

invariant gl_Position;

vec3 bary3(vec3 b, vec3 a0, vec3 a1, vec3 a2) { return b.x * a0 + b.y * a1 + b.z * a2; }

// Height-map sample at the (tiled) UV: white = peak.
float height_at(vec2 uv, float lod) {
    return textureLod(t_displacement, uv * fx.albedo_st.xy + fx.albedo_st.zw, lod).r;
}

void main() {
    vec3 b = gl_TessCoord;
    vec3 p0 = c_world_pos[0], p1 = c_world_pos[1], p2 = c_world_pos[2];
    vec3 n0 = normalize(c_normal[0]), n1 = normalize(c_normal[1]), n2 = normalize(c_normal[2]);
    vec3 lin = bary3(b, p0, p1, p2);
    vec3 n = normalize(bary3(b, n0, n1, n2));
    // Phong tessellation: project the point on each corner's tangent plane and blend.
    vec3 q0 = lin - dot(lin - p0, n0) * n0;
    vec3 q1 = lin - dot(lin - p1, n1) * n1;
    vec3 q2 = lin - dot(lin - p2, n2) * n2;
    vec3 smoothed = mix(lin, bary3(b, q0, q1, q2), clamp(fx.tess1.x, 0.0, 1.0));
    vec2 uv = bary3(vec3(b), vec3(c_uv[0], 0.0), vec3(c_uv[1], 0.0), vec3(c_uv[2], 0.0)).xy;
    float h = height_at(uv, p_info.x);
    vec3 pos = smoothed + n * ((h - fx.tess0.y) * fx.tess0.x);
#ifdef SHADOW
    gl_Position = mat4(c_lvp0[0], c_lvp1[0], c_lvp2[0], c_lvp3[0]) * vec4(pos, 1.0);
#else
    // Normal from the height field's slope along the interpolated tangent frame.
    vec4 tan4 = b.x * c_tangent[0] + b.y * c_tangent[1] + b.z * c_tangent[2];
    vec3 T = tan4.xyz - n * dot(tan4.xyz, n);
    T = dot(T, T) > 1e-12 ? normalize(T) : normalize(cross(n, abs(n.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
    vec3 B = cross(n, T) * (tan4.w < 0.0 ? -1.0 : 1.0);
    vec2 e = 1.0 / vec2(textureSize(t_displacement, 0));
    e *= exp2(p_info.x);
    vec2 ruv = uv * fx.albedo_st.xy + fx.albedo_st.zw;
    float du = textureLod(t_displacement, ruv + vec2(e.x, 0.0), p_info.x).r - textureLod(t_displacement, ruv - vec2(e.x, 0.0), p_info.x).r;
    float dv = textureLod(t_displacement, ruv + vec2(0.0, e.y), p_info.x).r - textureLod(t_displacement, ruv - vec2(0.0, e.y), p_info.x).r;
    float k = fx.tess0.x / max(p_info.y, 1e-6) / (2.0 * e.x);
    float kv = fx.tess0.x / max(p_info.y, 1e-6) / (2.0 * e.y);
    vec3 dn = normalize(n - du * k * T - dv * kv * B);
    v_world_pos = pos;
    v_normal = dn;
    v_uv = uv;
    v_color = b.x * c_color[0] + b.y * c_color[1] + b.z * c_color[2];
    v_tangent = vec4(normalize(T - dn * dot(T, dn)), tan4.w < 0.0 ? -1.0 : 1.0);
    v_uv1 = b.x * c_uv1[0] + b.y * c_uv1[1] + b.z * c_uv1[2];
    v_lightmap = c_lightmap[0];
    v_instance_color = b.x * c_instance_color[0] + b.y * c_instance_color[1] + b.z * c_instance_color[2];
    v_instance_param = b.x * c_instance_param[0] + b.y * c_instance_param[1] + b.z * c_instance_param[2];
    gl_Position = VIEW_PROJ * vec4(pos, 1.0);
    v_cur_clip = gl_Position;
    // The displacement is taken as static between frames (like a vertex hook's): carry it into
    // the previous clip position so only the object's own motion shows as velocity.
    vec4 prev_lin = b.x * c_prev_clip[0] + b.y * c_prev_clip[1] + b.z * c_prev_clip[2];
    v_prev_clip = prev_lin + frame.prev_view_proj * vec4(pos - lin, 0.0);
#endif
}
