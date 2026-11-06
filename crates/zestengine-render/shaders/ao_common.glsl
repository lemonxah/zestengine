// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Shared helpers of the screen-space ambient occlusion passes (ao.rs).
//
// Conventions: pixel coordinates have +y DOWN (the scene renders with a
// negative-height viewport, so framebuffer row 0 is the top of the image).
// "Linear depth" is the positive view-space distance along the camera's
// forward axis (-z of the right-handed view space). AO view space is x right,
// y up, z forward (positive), so a view position is (x, y, linear depth).

const float AO_PI = 3.14159265;
const float AO_HALF_PI = 1.57079633;
// Linear depth written for background (sky) pixels: never occludes, never
// receives occlusion.
const float AO_SKY = 1.0e6;

// Linearize a [0, 1] depth-buffer value. `lin` = (P22, P32, ortho, _) of the
// (jittered) projection matrix, column-major glam layout (P22 = z_axis.z,
// P32 = w_axis.z).
float ao_linearize(float d, vec4 lin) {
    if (d >= 1.0) {
        return AO_SKY;
    }
    if (lin.z > 0.5) {
        return (lin.y - d) / lin.x;
    }
    return lin.y / (d + lin.x);
}

// View position of the pixel at normalized screen `uv` (y down) with linear
// depth `z`. `view` = (P00, P11, offx, offy): for a perspective projection
// x = z * (ndc.x + P20) / P00; for an orthographic one x = (ndc.x - P30) / P00
// (the caller passes off = (P20, P21) or (-P30, -P31)).
vec3 ao_view_pos(vec2 uv, float z, vec4 view, bool ortho) {
    vec2 ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    vec2 xy = (ndc + view.zw) / view.xy;
    return vec3(ortho ? xy : xy * z, z);
}

// Interleaved gradient noise (Jimenez 2014): a cheap per-pixel value in [0, 1).
float ao_ign(vec2 p) {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

// 4x4 ordered (Bayer) pattern in [0, 1): a 4x4 box filter averages all 16
// rotations exactly, so the spatial denoise removes the pattern completely.
float ao_bayer4(ivec2 p) {
    const float m[16] = float[16](0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0, 3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0);
    return m[(p.y & 3) * 4 + (p.x & 3)] / 16.0;
}
