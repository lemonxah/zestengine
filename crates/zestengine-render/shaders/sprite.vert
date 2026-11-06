// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// 2D sprite pass, vertex stage (2D-3). One instanced draw covers any run of
// sprites; per-instance data comes from vertex binding 1 (SpriteInstance, 112
// bytes), the geometry from the pooled sprite mesh buffer (binding 0):
//   MODE_MESH  (0): the sprite's own pooled mesh (tight or full rect), pivot
//                   relative, y up.
//   MODE_SLICE (1): the 16-vertex 9-slice template; the vertex carries its grid
//                   coordinate (i, j) in 0..3 and the position/UV are computed
//                   here from rect/border/uv_border (no CPU mesh expansion).
//   MODE_QUAD  (2): the unit quad; one tile (Tiled mode) or any plain rect.
//   FLAG_ROTATED: the sprite's content is stored turned 90 degrees clockwise in
//                 its atlas page; QUAD draws turn the UVs back (`quad_uv` in
//                 sprite2d.rs, formula for formula).
//   FLAG_XZ:      the quad lies in the world XZ plane (a ground tilemap): the 2D
//                 position / affine give world (x, z) and i_pos_z.z is the height (world y).
//   FLAG_ANIM:    tile animation. i_anim indexes the frame table (set 2): a
//                 header (count, fps, flags, rotated mask) followed by the frames'
//                 UV rects; the frame comes from the push-constant clock plus the
//                 instance phase (i_cutoff), so animated tiles cost no CPU per
//                 frame. `anim_frame` in sprite2d.rs is the tested twin.
// Flips mirror the local position about the pivot (never a negative scale, so
// flipping does not break a batch). Build variants: default, MULTIVIEW (both
// XR eyes in one pass). The 9-slice maths mirrors `nine_slice_stops` /
// `slice_vertex` in sprite2d.rs formula for formula (those are unit-tested).

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
#endif

const uint FLAG_FLIP_X = 1u;
const uint FLAG_FLIP_Y = 2u;
const uint FLAG_ROTATED = 128u;
const uint FLAG_ANIM = 256u;
const uint FLAG_XZ = 512u;
const uint ANIM_LOOP_ONCE = 1u;
const uint MODE_SLICE = 1u;
const uint MODE_QUAD = 2u;

layout(set = 2, binding = 0, std430) readonly buffer AnimTable {
    vec4 anim[];
} anim_table;

layout(push_constant) uniform PC {
    vec4 params;   // x blend, y cutoff (fragment); z animation clock (vertex)
} pc;

layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_uv;

layout(location = 2) in vec4 i_affine;     // a, b, c, d
layout(location = 3) in vec3 i_pos_z;      // world x, y, z
layout(location = 4) in uint i_flags;
layout(location = 5) in vec4 i_uv_rect;    // u0, v0 (top), u1, v1 (bottom)
layout(location = 6) in vec4 i_rect;       // local min x, min y, size x, size y
layout(location = 7) in vec4 i_border;     // world L, B, R, T
layout(location = 8) in vec4 i_uv_border;  // uv L, B, R, T
layout(location = 9) in uint i_color;      // linear RGBA8
layout(location = 10) in uint i_layer;     // array layer
layout(location = 11) in float i_cutoff;   // alpha cutoff (mask draws); phase (animated tiles)
layout(location = 12) in uint i_anim;      // frame table header index (FLAG_ANIM)

layout(location = 0) out vec2 v_uv;
layout(location = 1) flat out vec4 v_color;
layout(location = 2) flat out uint v_layer;
layout(location = 3) flat out float v_cutoff;

// Four grid lines along one axis of a 9-slice drawn at `size` with border
// widths `lo`/`hi`: [0, lo', size - hi', size]. When the target is smaller than
// the border sum both borders shrink by the same factor and meet.
vec4 nine_slice_stops(float size, float lo, float hi) {
    size = max(size, 0.0);
    lo = max(lo, 0.0);
    hi = max(hi, 0.0);
    float sum = lo + hi;
    float s = (sum > size && sum > 0.0) ? size / sum : 1.0;
    return vec4(0.0, lo * s, size - hi * s, size);
}

// The frame shown t seconds into an animation (mirrors `anim_frame`).
uint anim_frame(uint count, float fps, uint flags, float t) {
    if (count <= 1u || !(fps > 0.0) || !(t > 0.0)) return 0u;
    float n = floor(t * fps);
    if ((flags & ANIM_LOOP_ONCE) != 0u) return uint(min(n, float(count - 1u)));
    return uint(mod(n, float(count)));
}

void main() {
    vec2 local;
    vec4 uv_rect = i_uv_rect;
    vec4 rect = i_rect;
    bool rotated = (i_flags & FLAG_ROTATED) != 0u;
    if ((i_flags & FLAG_ANIM) != 0u) {
        vec4 h = anim_table.anim[i_anim];
        uint count = floatBitsToUint(h.x);
        uint k = anim_frame(count, h.y, floatBitsToUint(h.z), pc.params.z + i_cutoff);
        uv_rect = anim_table.anim[i_anim + 1u + 2u * k];
        rect = anim_table.anim[i_anim + 2u + 2u * k];
        rotated = ((floatBitsToUint(h.w) >> k) & 1u) != 0u;
    }
    vec2 uv;
    uint mode = (i_flags >> 2) & 3u;
    if (mode == MODE_SLICE) {
        int i = int(a_pos.x + 0.5);
        int j = int(a_pos.y + 0.5);
        vec4 xs = nine_slice_stops(i_rect.z, i_border.x, i_border.z);
        vec4 ys = nine_slice_stops(i_rect.w, i_border.y, i_border.w);
        vec4 us = vec4(uv_rect.x, uv_rect.x + i_uv_border.x, uv_rect.z - i_uv_border.z, uv_rect.z);
        // v runs top-down in the image, y bottom-up in the sprite.
        vec4 vs = vec4(uv_rect.w, uv_rect.w - i_uv_border.y, uv_rect.y + i_uv_border.w, uv_rect.y);
        local = i_rect.xy + vec2(xs[i], ys[j]);
        uv = vec2(us[i], vs[j]);
    } else if (mode == MODE_QUAD) {
        local = rect.xy + a_pos * rect.zw;
        if (rotated) {
            uv = vec2(mix(uv_rect.x, uv_rect.z, a_pos.y), mix(uv_rect.y, uv_rect.w, a_pos.x));
        } else {
            uv = vec2(mix(uv_rect.x, uv_rect.z, a_pos.x), mix(uv_rect.w, uv_rect.y, a_pos.y));
        }
    } else {
        local = a_pos;
        uv = a_uv;
    }
    if ((i_flags & FLAG_FLIP_X) != 0u) local.x = -local.x;
    if ((i_flags & FLAG_FLIP_Y) != 0u) local.y = -local.y;

    vec2 world = vec2(i_affine.x * local.x + i_affine.z * local.y,
                      i_affine.y * local.x + i_affine.w * local.y) + i_pos_z.xy;
    vec3 wp = ((i_flags & FLAG_XZ) != 0u) ? vec3(world.x, i_pos_z.z, world.y) : vec3(world, i_pos_z.z);
    gl_Position = VIEW_PROJ * vec4(wp, 1.0);
    v_uv = uv;
    v_color = unpackUnorm4x8(i_color);
    v_layer = i_layer;
    v_cutoff = ((i_flags & FLAG_ANIM) != 0u) ? 0.0 : i_cutoff;
}
