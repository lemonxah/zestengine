// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// Temporal anti-aliasing resolve (aa.rs). Runs after the reflection / fog
// resolve and before bloom + post, on linear HDR at the render extent.
//
//  * velocity: the scene pass writes cur_ndc - prev_ndc (jitter delta included)
//    for the geometry it draws; pixels no pipeline wrote (sky, sprites, text)
//    keep the clear sentinel and are reprojected from depth with the camera
//    matrices (prev_vp * inv(vp)). The jitter delta is subtracted so history
//    follows the UNjittered image.
//  * history: RGBA16F, rgb = resolved colour, a = linear depth of that pixel.
//    Catmull-Rom fetch, 3x3 YCoCg variance clip (on a luma-compressed colour so
//    HDR highlights do not blow the box), Karis luminance weighting in the blend.
//  * rejection: history off-screen, reset, or its stored depth differing from the
//    reprojected depth by more than 5 percent (disocclusion) -> current frame only;
//    the reactive mask (transparents) raises the blend.

layout(set = 0, binding = 0) uniform sampler2D u_color;
layout(set = 0, binding = 1) uniform sampler2D u_history;  // linear, clamp
layout(set = 0, binding = 2) uniform sampler2D u_depth;    // nearest
layout(set = 0, binding = 3) uniform sampler2D u_velocity; // nearest
layout(set = 0, binding = 4) uniform sampler2D u_reactive; // nearest

layout(push_constant) uniform Taa {
    mat4 reproj; // prev_vp * inv(vp), both as rendered (jittered)
    vec4 jit;    // xy: this frame's jitter minus the previous frame's (NDC), z: base blend, w: 1 = no history
    vec4 lin;    // x P22, y P32, z 1 = orthographic, w variance-clip gamma
} pc;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

const float SENTINEL = 16384.0; // velocity written by no pipeline (cleared to 32768)

float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

float lin_depth(float d) {
    return pc.lin.z > 0.5 ? (pc.lin.y - d) / pc.lin.x : pc.lin.y / (d + pc.lin.x);
}

// Luma compression: the clip and the edge weights see a bounded range.
vec3 tm(vec3 c) { return c / (1.0 + luma(c)); }
vec3 itm(vec3 t) { return t / (1.0 - min(luma(t), 0.9995)); }

vec3 to_ycocg(vec3 c) {
    return vec3(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * c.r - 0.5 * c.b, -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}
vec3 from_ycocg(vec3 c) {
    return vec3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

// Clip q toward the centre of the box [lo, hi] (Playdead / Salvi).
vec3 clip_box(vec3 lo, vec3 hi, vec3 q) {
    vec3 centre = 0.5 * (hi + lo);
    vec3 ext = 0.5 * (hi - lo) + 1e-5;
    vec3 v = q - centre;
    vec3 a = abs(v / ext);
    float m = max(a.x, max(a.y, a.z));
    return m > 1.0 ? centre + v / m : q;
}

// Catmull-Rom history fetch in 5 bilinear taps.
vec3 history_catmull(vec2 uv, vec2 size) {
    vec2 pos = uv * size;
    vec2 centre = floor(pos - 0.5) + 0.5;
    vec2 f = pos - centre;
    vec2 f2 = f * f, f3 = f2 * f;
    vec2 w0 = -0.5 * f3 + f2 - 0.5 * f;
    vec2 w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
    vec2 w2 = -1.5 * f3 + 2.0 * f2 + 0.5 * f;
    vec2 w3 = 0.5 * f3 - 0.5 * f2;
    vec2 w12 = w1 + w2;
    vec2 t0 = (centre - 1.0) / size;
    vec2 t3 = (centre + 2.0) / size;
    vec2 t12 = (centre + w2 / w12) / size;
    vec3 r = textureLod(u_history, vec2(t12.x, t0.y), 0.0).rgb * (w12.x * w0.y)
           + textureLod(u_history, vec2(t0.x, t12.y), 0.0).rgb * (w0.x * w12.y)
           + textureLod(u_history, t12, 0.0).rgb * (w12.x * w12.y)
           + textureLod(u_history, vec2(t3.x, t12.y), 0.0).rgb * (w3.x * w12.y)
           + textureLod(u_history, vec2(t12.x, t3.y), 0.0).rgb * (w12.x * w3.y);
    float wsum = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
    return max(r / wsum, vec3(0.0));
}

void main() {
    ivec2 isize = textureSize(u_color, 0);
    vec2 size = vec2(isize);
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec2 uv = (vec2(p) + 0.5) / size;
    vec3 cur = clamp(texelFetch(u_color, p, 0).rgb, vec3(0.0), vec3(64000.0));
    float d_own = texelFetch(u_depth, p, 0).r;
    if (pc.jit.w > 0.5) {
        o_color = vec4(cur, lin_depth(d_own));
        return;
    }

    // 3x3 neighbourhood: colour statistics and the closest depth (its velocity
    // keeps moving silhouettes crisp).
    vec3 m1 = vec3(0.0), m2 = vec3(0.0), nmin = vec3(1e9), nmax = vec3(-1e9);
    float d_near = d_own;
    ivec2 p_near = p;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            ivec2 q = clamp(p + ivec2(x, y), ivec2(0), isize - 1);
            vec3 c = to_ycocg(tm(clamp(texelFetch(u_color, q, 0).rgb, vec3(0.0), vec3(64000.0))));
            m1 += c;
            m2 += c * c;
            nmin = min(nmin, c);
            nmax = max(nmax, c);
            float d = texelFetch(u_depth, q, 0).r;
            if (d < d_near) {
                d_near = d;
                p_near = q;
            }
        }
    }
    vec3 mean = m1 / 9.0;
    vec3 sigma = sqrt(max(m2 / 9.0 - mean * mean, vec3(0.0)));
    vec3 lo = max(mean - pc.lin.w * sigma, nmin);
    vec3 hi = min(mean + pc.lin.w * sigma, nmax);

    // Reprojection.
    vec2 ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    vec2 vel = texelFetch(u_velocity, p_near, 0).rg;
    if (vel.x > SENTINEL || vel.y > SENTINEL) {
        vec4 pp = pc.reproj * vec4(ndc, d_near, 1.0);
        vel = ndc - pp.xy / pp.w;
    }
    vel -= pc.jit.xy;
    vec2 prev_ndc = ndc - vel;
    vec2 prev_uv = vec2(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);

    // Depth this pixel had in the previous frame (camera motion only).
    vec4 pe = pc.reproj * vec4(ndc, d_own, 1.0);
    float expect = lin_depth(pe.z / pe.w);

    float blend = pc.jit.z;
    float speed_px = length(vel * 0.5 * size);
    blend = clamp(blend + speed_px * 0.02, blend, max(blend, 0.3));
    bool valid = all(greaterThanEqual(prev_uv, vec2(0.0))) && all(lessThanEqual(prev_uv, vec2(1.0)));
    if (valid) {
        // Disocclusion: none of the history texels around the reprojected
        // position holds a surface at the expected depth.
        vec2 hp = prev_uv * size - 0.5;
        ivec2 h0 = ivec2(floor(hp));
        float best = 1e9;
        for (int i = 0; i < 4; ++i) {
            ivec2 q = clamp(h0 + ivec2(i & 1, i >> 1), ivec2(0), isize - 1);
            float hd = texelFetch(u_history, q, 0).a;
            best = min(best, abs(hd - expect) / max(expect, 1e-4));
        }
        if (best > 0.05) valid = false;
    }
    float reactive = texelFetch(u_reactive, p_near, 0).r;
    blend = max(blend, reactive * 0.9);
    if (!valid) blend = 1.0;

    vec3 out_c = cur;
    if (blend < 1.0) {
        vec3 hist = history_catmull(prev_uv, size);
        vec3 hist_t = from_ycocg(clip_box(lo, hi, to_ycocg(tm(hist))));
        hist = itm(max(hist_t, vec3(0.0)));
        // Karis luminance weights: a bright outlier cannot dominate the blend.
        float wc = blend / (1.0 + luma(cur));
        float wh = (1.0 - blend) / (1.0 + luma(hist));
        out_c = (cur * wc + hist * wh) / max(wc + wh, 1e-6);
    }
    o_color = vec4(out_c, lin_depth(d_own));
}
