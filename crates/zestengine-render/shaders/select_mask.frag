// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// Selection mask: R = selection strength, G = 1 where the draw passed the
// scene depth test (the pipeline's write mask picks which channel lands).

layout(push_constant) uniform Push {
    mat4 view_proj;
    vec4 params;
} pc;

layout(location = 0) out vec4 o_mask;

void main() {
    o_mask = vec4(pc.params.x, 1.0, 0.0, 0.0);
}
