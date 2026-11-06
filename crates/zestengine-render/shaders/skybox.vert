// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// Sky pass, vertex stage: one fullscreen triangle, no vertex buffer.
//
// It does NOT compute world directions per vertex. It passes the clip-space
// position (exact under interpolation: NDC is affine in screen space) and a
// constant 3x3 "ray matrix" K (flat) that the fragment stage applies per pixel:
//     world_dir(pixel) = K * vec3(ndc.xy, 1)
// K = R^-1 * M^-1, where R is the camera rotation (the 3x3 of the view matrix,
// no translation, so large world coordinates cannot leak in) and M maps a
// view-space direction to (clip.x, clip.y, clip.w) - the projection without its
// depth row, so near / far / reversed or infinite far planes do not matter.
// TAA jitter and asymmetric (XR) frusta are in M's third column and carried over.
// Orthographic cameras have no perspective divide (all rays are parallel); they
// get a fixed 60-degree pseudo-perspective around the view axis so the sky
// does not swim or go flat as the ortho size (zoom) changes.

#ifdef MULTIVIEW
// Multiview: the full frame block, per-eye cameras at its end (gl_ViewIndex).
#extension GL_EXT_multiview : require
#include "standard_common.glsl"
#else
layout(set = 0, binding = 0) uniform FrameData {
    mat4 view;
    mat4 proj;
    mat4 view_proj;
    vec4 camera_pos;
} frame;
#define VIEW_MATRIX frame.view
#define VIEW_PROJ frame.view_proj
#endif

layout(location = 0) noperspective out vec2 v_ndc;
layout(location = 1) flat out vec3 v_k0;
layout(location = 2) flat out vec3 v_k1;
layout(location = 3) flat out vec3 v_k2;

void main() {
    // (0,0), (2,0), (0,2) -> NDC (-1,-1), (3,-1), (-1,3): covers the viewport.
    vec2 p = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    vec2 ndc = p * 2.0 - 1.0;

    mat3 rot = mat3(VIEW_MATRIX);
    mat3 rot_inv = inverse(rot);
#ifdef MULTIVIEW
    // No per-eye projection in the block: P's first three columns are
    // VIEW_PROJ * (R^-1 column, 0) exactly (the view's translation drops out).
    vec4 p0 = VIEW_PROJ * vec4(rot_inv[0], 0.0);
    vec4 p1 = VIEW_PROJ * vec4(rot_inv[1], 0.0);
    vec4 p2 = VIEW_PROJ * vec4(rot_inv[2], 0.0);
#else
    vec4 p0 = frame.proj[0];
    vec4 p1 = frame.proj[1];
    vec4 p2 = frame.proj[2];
#endif
    mat3 m;
    if (abs(p2.w) > 1e-6) {
        // Perspective: (clip.x, clip.y, clip.w) = M * view_dir.
        m = mat3(vec3(p0.x, p0.y, p0.w), vec3(p1.x, p1.y, p1.w), vec3(p2.x, p2.y, p2.w));
    } else {
        // Orthographic: 60-degree vertical pseudo-perspective, same aspect and
        // axis signs (y flip) as the ortho projection.
        const float F = 1.7320508; // 1 / tan(30 deg)
        float sx = abs(p0.x) / max(abs(p1.y), 1e-12);
        m = mat3(vec3(F * sx * sign(p0.x), 0.0, 0.0), vec3(0.0, F * sign(p1.y), 0.0), vec3(0.0, 0.0, -1.0));
    }
    mat3 k = rot_inv * inverse(m);
    v_k0 = k[0];
    v_k1 = k[1];
    v_k2 = k[2];
    v_ndc = ndc;

    // z = w: depth exactly 1.0 (the far plane); LESS_OR_EQUAL passes against
    // the cleared depth, every surface drawn later lands in front.
    gl_Position = vec4(ndc, 1.0, 1.0);
}
