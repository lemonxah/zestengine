// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// 2D sprite pass, fragment stage (2D-3), unlit. One source, selected by defines:
//   (none)       plain 2D texture at set 1 binding 0
//   ATLAS_ARRAY  sampler2DArray; the layer is per-instance data
//   MASK         mask-shape draw: no colour output, discards texels whose alpha
//                is below the instance's cutoff (stencil write happens in the
//                pipeline's stencil state)
// Push constants (batch-wide): x blend mode (0 = straight alpha, 1 = the texture
// is premultiplied and the output stays premultiplied, 2 = additive: the output is
// premultiplied by alpha for a ONE / ONE blend, 3 = multiply: the output is the tint
// faded to white by alpha for a DST_COLOR / ZERO blend), y global alpha cutoff
// (0 = off), zw unused.

#ifdef ATLAS_ARRAY
layout(set = 1, binding = 0) uniform sampler2DArray t_sprite;
#else
layout(set = 1, binding = 0) uniform sampler2D t_sprite;
#endif

layout(push_constant) uniform PC {
    vec4 params;
} pc;

layout(location = 0) in vec2 v_uv;
layout(location = 1) flat in vec4 v_color;
layout(location = 2) flat in uint v_layer;
layout(location = 3) flat in float v_cutoff;

layout(location = 0) out vec4 o_color;

void main() {
#ifdef ATLAS_ARRAY
    vec4 t = texture(t_sprite, vec3(v_uv, float(v_layer)));
#else
    vec4 t = texture(t_sprite, v_uv);
#endif
    vec4 c = t * v_color;
#ifdef MASK
    // Mask shapes use the texture alpha alone (tint-independent).
    if (t.a < v_cutoff || t.a <= 0.0) discard;
    o_color = vec4(0.0);
#else
    if (pc.params.y > 0.0 && c.a < pc.params.y) discard;
    if (c.a <= 0.0) discard;
    if (pc.params.x > 2.5) {
        o_color = vec4(mix(vec3(1.0), c.rgb, c.a), 1.0);
    } else if (pc.params.x > 1.5) {
        o_color = vec4(c.rgb * c.a, c.a);
    } else if (pc.params.x > 0.5) {
        // Premultiplied texture: tint the colour by alpha too, so fading works.
        o_color = vec4(t.rgb * v_color.rgb * v_color.a, c.a);
    } else {
        o_color = c;
    }
#endif
}
