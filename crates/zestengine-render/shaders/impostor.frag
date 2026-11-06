// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// Octahedral impostor, fragment stage: the four views nearest to the camera
// direction (the capture's `lookup`: bilinear over the octahedral grid) are
// blended, weighted by their coverage, so the picture turns smoothly between
// views instead of popping. Coverage cuts the silhouette (alpha test). Layers:
// 0 .. g*g-1 albedo (rgb, coverage), g*g .. 2*g*g-1 aux (rg octahedral normal
// in model space, b depth, a coverage); one layer per view, so filtering never
// reads a neighbouring view.

layout(set = 0, binding = 0) uniform FrameData {
    mat4 view;
    mat4 proj;
    mat4 view_proj;
    vec4 camera_pos;
    vec4 light_dir;
    vec4 light_color;
    vec4 ambient;
} frame;

layout(set = 2, binding = 0) uniform sampler2DArray atlas;

layout(push_constant) uniform PC {
    uint first;
    uint grid;
    uint pad0;
    uint pad1;
} pc;

layout(location = 0) in vec2 v_uv;
layout(location = 1) flat in vec3 v_dir;
layout(location = 2) flat in vec4 v_tint;
layout(location = 3) flat in vec3 v_rot0;
layout(location = 4) flat in vec3 v_rot1;
layout(location = 5) flat in vec3 v_rot2;

layout(location = 0) out vec4 out_color;

// Octahedral encoding of a unit direction into [-1, 1]^2 (assets impostor::dir_to_oct).
vec2 dir_to_oct(vec3 d) {
    float l1 = abs(d.x) + abs(d.y) + abs(d.z);
    vec2 e = d.xy / l1;
    if (d.z < 0.0) {
        e = (1.0 - abs(e.yx)) * vec2(e.x >= 0.0 ? 1.0 : -1.0, e.y >= 0.0 ? 1.0 : -1.0);
    }
    return e;
}

vec3 oct_to_dir(vec2 e) {
    vec3 n = vec3(e, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) {
        n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    }
    return normalize(n);
}

// The captured albedo is unlit; the scene's PBR diffuse of a rough dielectric is about this fraction of
// albedo * (ambient + light * N.L) (measured against the same mesh in the real renderer).
const float DIFFUSE_SCALE = 0.72;

void main() {
    float g = float(pc.grid);
    vec2 o = dir_to_oct(v_dir);
    vec2 f = (o * 0.5 + 0.5) * g - 0.5;
    vec2 f0 = floor(f);
    vec2 t = f - f0;
    ivec2 lo = clamp(ivec2(f0), ivec2(0), ivec2(pc.grid - 1u));
    ivec2 hi = clamp(ivec2(f0) + 1, ivec2(0), ivec2(pc.grid - 1u));
    ivec2 cell[4] = ivec2[4](ivec2(lo.x, lo.y), ivec2(hi.x, lo.y), ivec2(lo.x, hi.y), ivec2(hi.x, hi.y));
    float w[4] = float[4]((1.0 - t.x) * (1.0 - t.y), t.x * (1.0 - t.y), (1.0 - t.x) * t.y, t.x * t.y);
    float blended = 0.0;
    float wsum = 0.0;
    vec3 colour = vec3(0.0);
    vec3 normal = vec3(0.0);
    for (int k = 0; k < 4; ++k) {
        float layer = float(cell[k].y * int(pc.grid) + cell[k].x);
        vec4 a = texture(atlas, vec3(v_uv, layer));
        vec4 x = texture(atlas, vec3(v_uv, layer + g * g));
        blended += w[k] * a.a;
        colour += w[k] * a.a * a.rgb;
        wsum += w[k] * a.a;
        normal += w[k] * a.a * oct_to_dir(x.rg * 2.0 - 1.0);
    }
    // Silhouette: the views' blended coverage (a thin limb, e.g. a T-pose arm
    // 2 to 3 texels wide, can fall under the cut between views: the picture
    // turns smoothly and does not pop, at the price of breaking thin parts up).
    if (blended < 0.5) {
        discard;
    }
    colour /= max(wsum, 1e-4);
    vec3 n_model = normalize(normal + vec3(1e-6));
    vec3 n = normalize(mat3(v_rot0, v_rot1, v_rot2) * n_model);
    float ndl = max(dot(n, -normalize(frame.light_dir.xyz)), 0.0);
    vec3 lit = DIFFUSE_SCALE * colour * (frame.ambient.rgb + frame.light_color.rgb * ndl);
    out_color = vec4(mix(lit, v_tint.rgb * (frame.ambient.rgb + frame.light_color.rgb * 0.5), v_tint.a), 1.0);
}
