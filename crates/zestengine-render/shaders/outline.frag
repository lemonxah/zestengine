// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// Selection outline: flat bright purple. Never touches the object's own
// surface, so the material being edited stays fully visible.

layout(location = 0) out vec4 o_color;
// Deferred-SSR G-buffer (gbuf pipeline variants only): the outline is not a
// reflector. Zero reflectance keeps the resolve pass from reflecting it.
layout(location = 1) out vec4 o_gbuf;

layout(push_constant) uniform Push {
    mat4 model;
    vec4 base_color;
    vec4 emission;
    vec4 params0;
    vec4 params1; // x > 0.5: per-material toon outline coloured by base_color
} pc;

void main() {
    o_color = pc.params1.x > 0.5 ? pc.base_color : vec4(0.72, 0.25, 1.0, 1.0);
    o_gbuf = vec4(0.0);
}
