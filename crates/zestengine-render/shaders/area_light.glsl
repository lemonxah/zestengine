// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Area lights (rectangle, disc) by linearly transformed cosines, included by
// standard.frag. Our own implementation of the technique in Heitz et al.,
// "Real-Time Polygonal-Light Shading with Linearly Transformed Cosines" (2016):
//   * diffuse  = the clamped-cosine integral over the light's polygon (exact:
//                horizon clip + Lambert's edge sum);
//   * specular = the same integral over the polygon pushed through the inverse
//                LTC matrix of the GGX lobe (a table of 4 numbers per view angle
//                and roughness, ltc_lut.glsl, fitted by ltc.rs), times the
//                split-sum reflectance the rest of the shader already uses.
// A light is two consecutive entries of frame.lights: kind 4 (rectangle) or 5
// (disc) carrying centre, right axis x half width, colour x intensity and flags,
// then kind 6 carrying the up axis x half height. cross(right, up) is the emitting
// side. No shadows (v1); the lobe integral is exact for a uniformly emitting
// polygon, a texture is sampled once per lobe at a mip matching the lobe's width.

#include "ltc_lut.glsl"

// Emission atlas: 4 x 2 tiles of 256 x 256, full mip chain, filled by the renderer
// from each textured area light's texture every frame.
layout(set = 1, binding = 22) uniform sampler2D t_area_atlas;

// The LTC matrix table (ltc_lut.glsl holds the same numbers, the renderer uploads them as a
// LTC_N x LTC_N RGBA32F texture). It MUST stay a texture: a dynamically indexed constant array
// is copied into per-invocation local memory at shader entry, whether or not an area light is
// ever shaded, which made every pixel of every material ~70x slower.
layout(set = 1, binding = 23) uniform sampler2D t_ltc_lut;

const float AREA_TILE_PX = 256.0;
const int AREA_POLY_MAX = 10;

vec4 ltc_matrix(float ndotv, float roughness) {
    float u = min(sqrt(max(1.0 - ndotv, 0.0)), LTC_U_MAX);
    float fx = clamp(u / LTC_U_MAX, 0.0, 1.0) * float(LTC_N - 1);
    float fy = clamp((roughness - LTC_MIN_ROUGHNESS) / (1.0 - LTC_MIN_ROUGHNESS), 0.0, 1.0) * float(LTC_N - 1);
    int x0 = int(floor(fx));
    int y0 = int(floor(fy));
    int x1 = min(x0 + 1, LTC_N - 1);
    int y1 = min(y0 + 1, LTC_N - 1);
    float tx = fx - float(x0);
    float ty = fy - float(y0);
    return mix(mix(texelFetch(t_ltc_lut, ivec2(x0, y0), 0), texelFetch(t_ltc_lut, ivec2(x1, y0), 0), tx),
               mix(texelFetch(t_ltc_lut, ivec2(x0, y1), 0), texelFetch(t_ltc_lut, ivec2(x1, y1), 0), tx), ty);
}

// Integral of the clamped-cosine density over the spherical polygon with corner
// directions p[0..n) (not normalised, any winding): clip to the z >= 0 horizon,
// then (1 / 2 pi) * sum of edge angle x edge-plane normal z. `fvec` receives the
// sum of angle x edge-plane normal (the vector form factor, pointing at the
// light's cosine-weighted centre up to sign).
float ltc_cosine_integral(vec3 p[AREA_POLY_MAX], int n, out vec3 fvec) {
    vec3 q[AREA_POLY_MAX];
    int m = 0;
    for (int i = 0; i < n; ++i) {
        p[i] = normalize(p[i]);
    }
    for (int i = 0; i < n; ++i) {
        vec3 a = p[i];
        vec3 b = p[i + 1 == n ? 0 : i + 1];
        bool ain = a.z >= 0.0;
        bool bin = b.z >= 0.0;
        if (ain) {
            q[m++] = a;
        }
        if (ain != bin) {
            q[m++] = normalize(mix(a, b, a.z / (a.z - b.z)));
        }
    }
    fvec = vec3(0.0);
    if (m < 3) {
        return 0.0;
    }
    float sum = 0.0;
    for (int i = 0; i < m; ++i) {
        vec3 a = q[i];
        vec3 b = q[i + 1 == m ? 0 : i + 1];
        vec3 c = cross(a, b);
        float s = length(c);
        if (s < 1e-7) {
            continue;
        }
        float x = clamp(dot(a, b), -1.0, 1.0);
        // theta = acos(x); near x = 1 (a tiny edge) the cross product already carries it.
        float theta = acos(x);
        sum += theta * c.z / s;
        fvec += theta * c / s;
    }
    return abs(sum) * (0.5 / 3.14159265);
}

// Emission colour of the light's texture along `dir` (from the shaded point), as
// seen through a lobe `spread` metres wide at the light's plane.
vec3 area_texture(int tile, vec3 P, vec3 dir, vec3 center, vec3 r, vec3 u, vec3 en, float spread) {
    float denom = dot(dir, en);
    if (abs(denom) < 1e-4) {
        denom = denom < 0.0 ? -1e-4 : 1e-4;
    }
    float t = dot(center - P, en) / denom;
    vec3 hit = P + dir * max(t, 0.0) - center;
    vec2 uv = vec2(dot(hit, r) / dot(r, r), dot(hit, u) / dot(u, u)) * 0.5 + 0.5;
    float size = max(length(r), length(u)) * 2.0;
    float lod = clamp(log2(max(spread / max(size, 1e-3) * AREA_TILE_PX, 1.0)), 0.0, 8.0);
    float half_texel = 0.5 * exp2(lod) / AREA_TILE_PX;
    uv = clamp(uv, vec2(half_texel), vec2(1.0 - half_texel));
    vec2 tile_uv = vec2(0.25, 0.5);
    vec2 origin = vec2(float(tile & 3), float(tile >> 2)) * tile_uv;
    return textureLod(t_area_atlas, origin + uv * tile_uv, lod).rgb;
}

// One area light for the shaded point. `diffuse` gets the clamped-cosine
// factor already multiplied by the light radiance (the caller applies albedo x kd),
// `spec` the specular radiance (the caller adds it as is).
void area_light_eval(int i, vec3 P, vec3 N, vec3 V, float NdotV, float roughness, vec3 f0,
                     out vec3 diffuse, out vec3 spec, out float diffuse_factor) {
    diffuse = vec3(0.0);
    spec = vec3(0.0);
    diffuse_factor = 0.0;
    Light L = frame.lights[i];
    Light U = frame.lights[i + 1];
    vec3 center = L.pos_kind.xyz;
    vec3 r = L.dir_range.xyz;
    vec3 u = U.dir_range.xyz;
    float range = max(L.dir_range.w, 1e-3);
    bool disc = L.pos_kind.w > 4.5;
    bool two_sided = L.color.w > 0.5;
    int tile = int(L.spot.x + 0.5) - 1;
    vec3 en = normalize(cross(r, u));
    float side = dot(en, P - center);
    if (side <= 0.0 && !two_sided) {
        return;
    }
    float dist = length(center - P);
    float fade = clamp(1.0 - dist / range, 0.0, 1.0);
    fade *= fade;
    if (fade <= 0.0) {
        return;
    }
    // Corners relative to the shaded point.
    vec3 corner[AREA_POLY_MAX];
    int n;
    if (disc) {
        n = 8;
        for (int k = 0; k < 8; ++k) {
            float a = (float(k) + 0.5) * (6.2831853 / 8.0);
            // An inscribed octagon covers 90% of the disc: widen it to the disc's area.
            corner[k] = center + (r * cos(a) + u * sin(a)) * 1.054 - P;
        }
    } else {
        n = 4;
        corner[0] = center - r - u - P;
        corner[1] = center + r - u - P;
        corner[2] = center + r + u - P;
        corner[3] = center - r + u - P;
    }
    vec3 T1 = V - N * dot(V, N);
    float t1len = length(T1);
    T1 = t1len > 1e-4 ? T1 / t1len : (abs(N.x) < 0.9 ? normalize(cross(N, vec3(1.0, 0.0, 0.0))) : normalize(cross(N, vec3(0.0, 1.0, 0.0))));
    vec3 T2 = cross(N, T1);
    vec3 local[AREA_POLY_MAX];
    for (int k = 0; k < n; ++k) {
        local[k] = vec3(dot(corner[k], T1), dot(corner[k], T2), dot(corner[k], N));
    }
    // Diffuse: the cosine lobe itself.
    vec3 fv_d;
    float fd = ltc_cosine_integral(local, n, fv_d);
    // Specular: the polygon through the GGX lobe's inverse matrix [[a 0 b] [0 c 0] [d 0 1]].
    vec4 m = ltc_matrix(NdotV, roughness);
    vec3 warped[AREA_POLY_MAX];
    for (int k = 0; k < n; ++k) {
        vec3 l = local[k];
        warped[k] = vec3(m.x * l.x + m.y * l.z, m.z * l.y, m.w * l.x + l.z);
    }
    vec3 fv_s;
    float fs = ltc_cosine_integral(warped, n, fv_s);
    vec3 radiance = L.color.rgb * fade;
    vec3 tint_d = vec3(1.0);
    vec3 tint_s = vec3(1.0);
    if (tile >= 0) {
        vec3 to_c = center - P;
        // Diffuse: the cosine-weighted direction of the light, blurred by the distance.
        vec3 dir_d = fv_d.x * T1 + fv_d.y * T2 + fv_d.z * N;
        dir_d = dot(dir_d, to_c) >= 0.0 ? dir_d : -dir_d;
        dir_d = dot(dir_d, dir_d) > 1e-8 ? normalize(dir_d) : normalize(to_c);
        float plane_dist = max(abs(dot(to_c, en)), 1e-3);
        tint_d = area_texture(tile, P, dir_d, center, r, u, en, plane_dist);
        // Specular: where the mirror ray lands, sliding to the diffuse spot as the lobe widens.
        vec3 R = reflect(-V, N);
        vec3 dir_s = normalize(mix(R, dir_d, roughness * roughness));
        float spread = plane_dist * mix(0.03, 1.0, roughness * roughness);
        tint_s = area_texture(tile, P, dir_s, center, r, u, en, spread);
    }
    vec2 ab = env_brdf_approx(roughness, NdotV);
    vec3 refl = f0 * ab.x + vec3(ab.y);
    diffuse = radiance * tint_d * fd;
    diffuse_factor = fd * fade;
    spec = radiance * tint_s * refl * fs;
}
