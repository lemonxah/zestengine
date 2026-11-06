// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// SDF/MTSDF text coverage with derivative-based anti-aliasing: the edge ramp
// is always one screen pixel wide, at any scale, under any transform, per eye.
// Outline and softness come from the true-distance channel (MTSDF alpha, or
// the R8 SDF); an optional 4-tap rotated-grid supersample calms minified
// (distant VR) text.

#ifdef WORLD
#define TSET 1
#else
#define TSET 0
#endif

layout(set = TSET, binding = 0) uniform sampler2DArray atlas_rgba; // MTSDF / colour
layout(set = TSET, binding = 1) uniform sampler2DArray atlas_r8;   // single-channel SDF

layout(push_constant) uniform PC {
    mat4 model;
    vec4 tint;
    vec4 clip;
    vec4 params;
} pc;

layout(location = 0) in vec2 v_uv;
layout(location = 1) flat in vec4 v_color;
layout(location = 2) flat in vec4 v_outline;
layout(location = 3) flat in uvec2 v_meta;
layout(location = 4) in vec2 v_lpx;

layout(location = 0) out vec4 out_color;

float median3(vec3 v) {
    return max(min(v.r, v.g), min(max(v.r, v.g), v.b));
}

void main() {
    // Derivatives first, in uniform control flow; sampling uses them explicitly.
    vec2 dx = dFdx(v_uv);
    vec2 dy = dFdy(v_uv);

    uint flags = uint(pc.params.y + 0.5);
    if ((flags & 2u) != 0u) {
        vec2 lo = pc.clip.xy;
        vec2 hi = pc.clip.xy + pc.clip.zw;
        if (v_lpx.x < lo.x || v_lpx.y < lo.y || v_lpx.x > hi.x || v_lpx.y > hi.y) discard;
    }

    uint kind = (v_meta.x >> 8) & 3u;
    if (kind == 3u) {
        // Flat rect (underline, strikethrough, highlight).
        out_color = vec4(v_color.rgb * v_color.a, v_color.a);
        return;
    }
    float layer = float(v_meta.x & 0xffu);
    if (kind == 2u) {
        vec4 t = textureGrad(atlas_rgba, vec3(v_uv, layer), dx, dy);
        out_color = vec4(t.rgb * v_color.rgb, t.a) * v_color.a;
        if (out_color.a <= 0.003) discard;
        return;
    }

    float spread = max(float((v_meta.x >> 16) & 0xffu) / 8.0, 1.0);
    // Sign-extend the dilation byte; outline and softness are unsigned.
    float dilate = float(int(v_meta.y << 24) >> 24) / 127.0 * 0.25;
    float outline_w = float((v_meta.y >> 8) & 0xffu) / 255.0 * 0.5;
    float soft = float((v_meta.y >> 16) & 0xffu) / 255.0 * 0.5;

    bool mtsdf = kind == 0u;
    vec2 atlas_size = mtsdf ? vec2(textureSize(atlas_rgba, 0).xy) : vec2(textureSize(atlas_r8, 0).xy);
    // Texels per screen pixel, then screen pixels per unit of normalized
    // distance (the field spans 2 * spread texels).
    float tpp = max(0.5 * (length(dx * atlas_size) + length(dy * atlas_size)), 1e-4);
    float px_range = max(2.0 * spread / tpp, 1.0);
    float ramp = px_range / (1.0 + soft * px_range);

    // Optional supersample where the glyph is minified below a texel per pixel.
    bool ss = (flags & 1u) != 0u && tpp > 1.0;
    const vec2 OFFS[4] = vec2[4](vec2(-0.375, -0.125), vec2(0.125, -0.375), vec2(0.375, 0.125), vec2(-0.125, 0.375));
    int taps = ss ? 4 : 1;

    float face = 0.0;
    float rim = 0.0;
    for (int i = 0; i < taps; i++) {
        vec2 uv = v_uv + (ss ? dx * OFFS[i].x + dy * OFFS[i].y : vec2(0.0));
        float sd_face;
        float sd_true;
        if (mtsdf) {
            vec4 t = textureGrad(atlas_rgba, vec3(uv, layer), dx, dy);
            sd_face = median3(t.rgb);
            sd_true = t.a;
        } else {
            sd_face = textureGrad(atlas_r8, vec3(uv, layer), dx, dy).r;
            sd_true = sd_face;
        }
        face += clamp((sd_face - 0.5 + dilate) * ramp + 0.5, 0.0, 1.0);
        if (outline_w > 0.0) {
            rim += clamp((sd_true - 0.5 + dilate + outline_w) * ramp + 0.5, 0.0, 1.0);
        }
    }
    face /= float(taps);
    rim /= float(taps);

    // Premultiplied: face over outline.
    float a_face = face * v_color.a;
    float a_out = outline_w > 0.0 ? max(rim - face, 0.0) * v_outline.a : 0.0;
    float a = a_face + a_out;
    if (a <= 0.003) discard;
    out_color = vec4(v_color.rgb * a_face + v_outline.rgb * a_out, a);
}
