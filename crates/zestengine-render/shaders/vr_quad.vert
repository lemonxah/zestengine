// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// VR overlay quad: a unit quad in the XY plane (local [-0.5,0.5]) positioned in
// the world by the push-constant model-view-projection. Used to draw the
// left-hand UI panel (textured with the editor's egui output) and solid markers
// (pointer ray + cursor) in the headset's eye images. No vertex buffer.

layout(push_constant) uniform Push {
    mat4 mvp;
    vec4 params;  // x = mode (0 textured / 1 solid), yzw = solid colour
    vec4 uv_rect; // xy = uv min, zw = uv max (screen-region this quad samples)
} pc;

layout(location = 0) out vec2 v_uv;

// Two triangles covering the quad; UV in [0,1].
const vec2 P[6] = vec2[6](
    vec2(-0.5, -0.5), vec2(0.5, -0.5), vec2(0.5, 0.5),
    vec2(-0.5, -0.5), vec2(0.5, 0.5), vec2(-0.5, 0.5)
);

void main() {
    vec2 p = P[gl_VertexIndex];
    v_uv = mix(pc.uv_rect.xy, pc.uv_rect.zw, p + 0.5);
    gl_Position = pc.mvp * vec4(p, 0.0, 1.0);
}
