// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
#version 450
#include "standard_common.glsl"
layout(set = 1, binding = 0) uniform sampler2D t_albedo;
layout(set = 1, binding = 5) uniform sampler2D t_opacity;
layout(push_constant) uniform Push {
    mat4 light_vp;
    mat4 coverage; // [0][0] = material alpha cutoff; model comes from instances.
} pc;
layout(location = 2) in vec2 v_uv;
layout(location = 7) in vec4 v_shadow_color;
void main() {
    vec2 uv = v_uv * fx.albedo_st.xy + fx.albedo_st.zw + fx.scroll.xy * frame.misc.x;
    float alpha = texture(t_albedo, uv).a * texture(t_opacity, uv).r * v_shadow_color.a;
    if (alpha < pc.coverage[0][0]) discard;
}
