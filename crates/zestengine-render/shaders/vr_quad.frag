// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// VR overlay quad fragment: sample the egui UI texture for the panel (mode 0,
// flipping Y because egui's origin is top-left), or output a solid colour for
// pointer/cursor markers (mode 1).

layout(set = 0, binding = 0) uniform sampler2D u_tex;

layout(push_constant) uniform Push {
    mat4 mvp;
    vec4 params; // x = mode, yzw = solid colour
} pc;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

void main() {
    if (pc.params.x < 0.5) {
        vec4 c = texture(u_tex, vec2(v_uv.x, 1.0 - v_uv.y));
        if (c.a < 0.01) discard; // let the scene show through transparent UI areas
        o_color = c;
    } else {
        o_color = vec4(pc.params.yzw, 1.0);
    }
}
