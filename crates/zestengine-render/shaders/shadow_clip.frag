// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
#version 450
#include "shadow_clip.glsl"
layout(push_constant) uniform Push {
    mat4 light_vp;
    mat4 region;
} pc;
layout(location = 0) in vec3 v_shadow_world;
void main() {
    if (shadow_region_contains(pc.region, v_shadow_world)) discard;
}
