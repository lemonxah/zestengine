// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// Lightmap gbuffer outputs: world position (a = 1 marks a covered texel) and
// world normal. Empty texels keep their cleared a = 0 and are skipped by the
// trace.

layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_world_normal;

layout(location = 0) out vec4 o_pos;
layout(location = 1) out vec4 o_normal;

void main() {
    o_pos = vec4(v_world_pos, 1.0);
    o_normal = vec4(normalize(v_world_normal), 0.0);
}
