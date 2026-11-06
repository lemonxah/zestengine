// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Shared by gpu_cull.comp and cluster_cull.comp: the culled view and the Hi-Z
// occlusion test. The includer declares `uniform sampler2D hiz` before this.
// The test is mirrored on the CPU in gpu_cull.rs (hiz tests).

#include "cluster_view.glsl"

// Nearest NDC depth of the sphere's bounding box vs the farthest depth the
// pyramid holds over its screen rectangle. Anything touching the previous
// camera's near plane or leaving its screen counts as visible.
bool occluded_tex(sampler2D hiz, View v, vec3 c, float r) {
    vec3 lo = c - vec3(r);
    vec3 hi = c + vec3(r);
    vec2 mn = vec2(1e30);
    vec2 mx = vec2(-1e30);
    float zmin = 1e30;
    for (int k = 0; k < 8; ++k) {
        vec3 p = vec3((k & 1) != 0 ? hi.x : lo.x, (k & 2) != 0 ? hi.y : lo.y, (k & 4) != 0 ? hi.z : lo.z);
        vec4 clip = v.hiz_vp * vec4(p, 1.0);
        if (clip.w <= 1e-5) {
            return false;
        }
        vec3 ndc = clip.xyz / clip.w;
        mn = min(mn, ndc.xy);
        mx = max(mx, ndc.xy);
        zmin = min(zmin, ndc.z);
    }
    if (zmin <= 0.0 || mn.x < -1.0 || mn.y < -1.0 || mx.x > 1.0 || mx.y > 1.0) {
        return false;
    }
    // Negative-height viewport: NDC +y is the top image row.
    vec2 uv0 = vec2(mn.x * 0.5 + 0.5, 0.5 - mx.y * 0.5);
    vec2 uv1 = vec2(mx.x * 0.5 + 0.5, 0.5 - mn.y * 0.5);
    vec2 px = (uv1 - uv0) * v.hiz.xy;
    int max_lod = int(v.hiz.z) - 1;
    int lod = clamp(int(ceil(log2(max(max(px.x, px.y), 1.0)))), 0, max_lod);
    ivec2 a;
    ivec2 b;
    for (;;) {
        ivec2 dim = textureSize(hiz, lod);
        a = clamp(ivec2(floor(uv0 * vec2(dim))), ivec2(0), dim - 1);
        b = clamp(ivec2(floor(uv1 * vec2(dim))), ivec2(0), dim - 1);
        if (all(lessThanEqual(b - a, ivec2(1))) || lod >= max_lod) {
            break;
        }
        lod += 1;
    }
    float zmax = max(
        max(texelFetch(hiz, a, lod).r, texelFetch(hiz, ivec2(b.x, a.y), lod).r),
        max(texelFetch(hiz, ivec2(a.x, b.y), lod).r, texelFetch(hiz, b, lod).r));
    return zmin > zmax;
}

// The pyramid the includer declared (`hiz`).
bool occluded(View v, vec3 c, float r) {
    return occluded_tex(hiz, v, c, r);
}
