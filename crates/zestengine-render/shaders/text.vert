// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// Instanced SDF/MTSDF text: vertex pulling, no vertex or index buffers. One
// draw covers any number of glyph quads (6 vertices each) read from a storage
// buffer. Build variants: default = screen space (model maps layout px to
// clip), WORLD = scene pass (model maps layout px to world), WORLD+MULTIVIEW
// = both XR eyes in one pass.

#ifdef WORLD
#ifdef MULTIVIEW
#extension GL_EXT_multiview : require
#include "standard_common.glsl"
#else
layout(set = 0, binding = 0) uniform FrameData {
    mat4 view;
    mat4 proj;
    mat4 view_proj;
    vec4 camera_pos;
} frame;
#define VIEW_PROJ frame.view_proj
#define VIEW_MATRIX frame.view
#endif
#define TSET 1
#else
#define TSET 0
#endif

struct Inst {
    vec2 pos;      // quad top-left, layout px
    vec2 size;     // quad size, layout px
    uvec2 uv;      // u0|v0<<16, u1|v1<<16 (normalized to the page)
    uint color;    // RGBA8 sRGB
    uint meta;     // layer | kind<<8 | flags<<10 | spread*8<<16
    uint fx;       // dilate i8 | outline u8 | softness u8
    uint outline;  // RGBA8 outline colour
};

layout(std430, set = TSET, binding = 2) readonly buffer Instances { Inst inst[]; };

layout(push_constant) uniform PC {
    mat4 model;
    vec4 tint;   // linear RGBA multiplied into every glyph
    vec4 clip;   // layout-px clip rect x, y, w, h
    vec4 params; // x billboard (0 none, 1 Y, 2 full), y flags (1 supersample, 2 clip), zw layout-px offset
} pc;

layout(location = 0) out vec2 v_uv;
layout(location = 1) flat out vec4 v_color;
layout(location = 2) flat out vec4 v_outline;
layout(location = 3) flat out uvec2 v_meta; // meta, fx
layout(location = 4) out vec2 v_lpx;        // layout px, for clipping

vec4 srgb_to_linear(vec4 c) {
    vec3 lo = c.rgb / 12.92;
    vec3 hi = pow((c.rgb + 0.055) / 1.055, vec3(2.4));
    return vec4(mix(hi, lo, lessThanEqual(c.rgb, vec3(0.04045))), c.a);
}

const vec2 CORNERS[6] = vec2[6](vec2(0, 0), vec2(1, 0), vec2(0, 1), vec2(0, 1), vec2(1, 0), vec2(1, 1));
const float ITALIC_SHEAR = 0.21;

void main() {
    Inst g = inst[gl_InstanceIndex];
    vec2 c = CORNERS[gl_VertexIndex % 6];
    vec2 p = g.pos + c * g.size;
    if ((((g.meta >> 10) & 0x3fu) & 1u) != 0u) {
        // Italic: shear the top of the quad right.
        p.x += (1.0 - c.y) * g.size.y * ITALIC_SHEAR;
    }
    v_lpx = p;
    p += pc.params.zw;

    uint u0 = g.uv.x & 0xffffu, v0 = g.uv.x >> 16, u1 = g.uv.y & 0xffffu, v1 = g.uv.y >> 16;
    v_uv = mix(vec2(u0, v0), vec2(u1, v1), c) / 65535.0;

    vec4 col = srgb_to_linear(unpackUnorm4x8(g.color));
    vec4 oc = srgb_to_linear(unpackUnorm4x8(g.outline));
    v_color = vec4(col.rgb * pc.tint.rgb, col.a * pc.tint.a);
    v_outline = vec4(oc.rgb * pc.tint.rgb, oc.a * pc.tint.a);
    v_meta = uvec2(g.meta, g.fx);

#ifdef WORLD
    int bb = int(pc.params.x + 0.5);
    vec3 world;
    if (bb == 0) {
        world = (pc.model * vec4(p, 0.0, 1.0)).xyz;
    } else {
        // Face the camera: keep the model's translation and scale, take the
        // axes from the view. Layout y is down, world up is +y.
        vec3 origin = pc.model[3].xyz;
        float sx = length(pc.model[0].xyz);
        float sy = length(pc.model[1].xyz);
        vec3 right = vec3(VIEW_MATRIX[0][0], VIEW_MATRIX[1][0], VIEW_MATRIX[2][0]);
        vec3 up = vec3(VIEW_MATRIX[0][1], VIEW_MATRIX[1][1], VIEW_MATRIX[2][1]);
        if (bb == 1) {
            right = normalize(vec3(right.x, 0.0, right.z));
            up = vec3(0.0, 1.0, 0.0);
        }
        world = origin + right * (p.x * sx) - up * (p.y * sy);
    }
    gl_Position = VIEW_PROJ * vec4(world, 1.0);
#else
    gl_Position = pc.model * vec4(p, 0.0, 1.0);
#endif
}
