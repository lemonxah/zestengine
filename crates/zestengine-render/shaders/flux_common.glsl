// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.
//
// Flux sampling and first-order spherical-harmonic math.
// Equations and buffer conventions: docs/shaders.md, "Flux math".
#ifndef FLUX_COMMON
#define FLUX_COMMON

const float FLUX_PI = 3.14159265359;

// Orthonormal real SH through degree one. Keep the screen-probe buffer's
// established ordering (constant, -y, +z, -x); signs are a basis convention.
// Normalization follows integral(Y_i * Y_j, sphere) = delta_ij.
vec4 flux_sh_l1_basis(vec3 direction) {
    const float constant_band = sqrt(1.0 / (4.0 * FLUX_PI));
    const float linear_band = sqrt(3.0 / (4.0 * FLUX_PI));
    return vec4(constant_band, linear_band * vec3(-direction.y, direction.z, -direction.x));
}

// Integrate L(w) * max(dot(normal,w),0) over the hemisphere for L1 radiance.
// The constant and linear bands integrate with factors pi and 2*pi/3.
// Ramamoorthi & Hanrahan (2001), equations 3, 7 and 9:
// https://graphics.stanford.edu/papers/envmap/envmap.pdf
vec3 flux_sh_l1_irradiance(vec3 normal, vec3 constant_rgb,
                          vec3 minus_y_rgb, vec3 z_rgb, vec3 minus_x_rgb) {
    vec4 basis = flux_sh_l1_basis(normal);
    vec3 linear_rgb = minus_y_rgb * basis.y + z_rgb * basis.z + minus_x_rgb * basis.w;
    return max(vec3(0.0), FLUX_PI * (constant_rgb * basis.x + (2.0 / 3.0) * linear_rgb));
}

// Malley's disk projection: radius^2 is uniform, height=sqrt(1-radius^2).
// This gives density dot(normal,direction)/pi per unit solid angle.
// https://pbr-book.org/4ed/Sampling_Algorithms/Sampling_Multidimensional_Functions#Cosine-WeightedHemisphereSampling
vec3 cosine_hemisphere_dir(vec3 normal, vec2 sample_uv) {
    float radius = sqrt(sample_uv.x);
    float azimuth = 2.0 * FLUX_PI * sample_uv.y;
    // Rotate the signed Z axis to normal with a unit quaternion. Choosing
    // the south chart below -0.5 keeps the denominator away from zero and
    // preserves the directional-bin convention used by history/denoising.
    float pole = normal.z >= -0.5 ? 1.0 : -1.0;
    vec3 local = vec3(radius * cos(azimuth), pole * radius * sin(azimuth),
                      pole * sqrt(max(0.0, 1.0 - sample_uv.x)));
    vec4 rotation = normalize(vec4(cross(vec3(0.0, 0.0, pole), normal), 1.0 + pole * normal.z));
    vec3 rotated = local + 2.0 * cross(rotation.xyz, cross(rotation.xyz, local) + rotation.w * local);
    return normalize(rotated);
}

// --------------------------------------------------------------- denoise utils
// Sample a light's finite source for visibility. This matches the baker's
// disc approximation, with temporal/probe samples resolving the penumbra.
// Testing only the centre incorrectly blocks lights mounted inside fixtures.
void flux_light_shadow_ray(vec3 l, float dist, float radius, bool directional, vec2 u,
                           out vec3 shadow_dir, out float shadow_dist) {
    shadow_dir = l;
    shadow_dist = dist;
    if (radius <= 1e-5) return;
    vec3 t = normalize(cross(abs(l.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0), l));
    vec3 b = cross(l, t);
    float a = 2.0 * FLUX_PI * u.x;
    vec2 disk = vec2(cos(a), sin(a)) * (sqrt(u.y) * radius);
    vec3 v = l * (directional ? 1.0 : dist) + t * disk.x + b * disk.y;
    shadow_dir = normalize(v);
    if (!directional) shadow_dist = length(v);
}

// Cap the brightest channel with one common scale to preserve RGB ratios.
vec3 firefly_clamp(vec3 radiance, float max_intensity) {
    float ceiling = max(max_intensity, 0.0);
    float peak = max(radiance.r, max(radiance.g, radiance.b));
    return radiance * (ceiling / max(max(peak, ceiling), 1e-20));
}

// Surface matching must not loosen when the camera moves farther away.
float plane_depth_weight(vec3 frag_pos, vec3 frag_normal, vec3 sample_pos) {
    vec3 delta = sample_pos - frag_pos;
    float plane_dist = abs(dot(delta, frag_normal));
    float tolerance = 0.015 + 0.02 * length(delta);
    return exp(-plane_dist * plane_dist / (tolerance * tolerance));
}

// --------------------------------------------------------- probe placement
// GDF samples are stored at grid nodes spanning [min,max]. Normalized texture
// coordinates address texel centres, not those nodes: account for the half texel.
vec3 flux_gdf_uv(vec3 p, vec3 lo, vec3 hi, ivec3 size) {
    vec3 t = clamp((p - lo) / max(hi - lo, vec3(1e-6)), 0.0, 1.0);
    return (t * vec3(size - 1) + 0.5) / vec3(size);
}

float flux_gdf_distance(sampler3D field, vec3 p, vec3 lo, vec3 hi) {
    vec3 q = clamp(p, lo, hi);
    float outside = length(p - q);
    float d = textureLod(field, flux_gdf_uv(q, lo, hi, textureSize(field, 0)), 0.0).r;
    return outside > 0.0 ? length(vec2(outside, max(d, 0.0))) : d;
}

// World position from a full-res depth texel (negative-height viewport: flip).
vec3 flux_world_from_texel(mat4 inv_vp, ivec2 full, ivec2 t, float depth, bool flip) {
    vec2 uv = (vec2(t) + 0.5) / vec2(full);
    vec2 ndc = uv * 2.0 - 1.0;
    if (flip) ndc.y = -ndc.y;
    vec4 w = inv_vp * vec4(ndc, depth, 1.0);
    return w.xyz / w.w;
}

// Place screen probe `px` (of `probe_dims`) on a real surface pixel: the
// closest-to-camera full-res texel of its tile (a bilinear depth at the tile
// corner averaged across creases and sat off the surface), with the normal from
// +-1 full-res neighbours, picking the closer-in-depth side per axis so the
// tangent stays on this surface. Returns false for sky.
bool flux_probe_surface(sampler2D depth_tex, uvec2 px, uvec2 probe_dims, mat4 inv_vp, bool flip,
                        out vec3 p, out vec3 n) {
    ivec2 full = textureSize(depth_tex, 0);
    // Partition the WHOLE depth image. A fixed floor(full / probe_dims)
    // stride shrinks 4-pixel tiles to 3 whenever the target isn't divisible
    // by four, leaving its last quarter untraced and misregistering the rest.
    ivec2 t0 = ivec2(px) * full / ivec2(probe_dims);
    ivec2 t1 = ivec2(px + 1u) * full / ivec2(probe_dims);
    ivec2 tsz = max(t1 - t0, ivec2(1));
    float depth = 1.0;
    ivec2 best = clamp(t0, ivec2(0), full - 1);
    for (int y = 0; y < tsz.y; ++y) {
        for (int x = 0; x < tsz.x; ++x) {
            ivec2 t = clamp(t0 + ivec2(x, y), ivec2(0), full - 1);
            float d = texelFetch(depth_tex, t, 0).r;
            if (d < depth) { depth = d; best = t; }
        }
    }
    if (depth >= 1.0) return false;
    p = flux_world_from_texel(inv_vp, full, best, depth, flip);
    ivec2 lo = full - 1;
    float dl = texelFetch(depth_tex, clamp(best - ivec2(1, 0), ivec2(0), lo), 0).r;
    float dr = texelFetch(depth_tex, clamp(best + ivec2(1, 0), ivec2(0), lo), 0).r;
    float dd = texelFetch(depth_tex, clamp(best - ivec2(0, 1), ivec2(0), lo), 0).r;
    float du = texelFetch(depth_tex, clamp(best + ivec2(0, 1), ivec2(0), lo), 0).r;
    // At image edges use the inward neighbour, never reconstruct an
    // out-of-image position using the clamped edge depth.
    bool left = best.x > 0 && (best.x == lo.x || abs(dl - depth) < abs(dr - depth));
    bool down = best.y > 0 && (best.y == lo.y || abs(dd - depth) < abs(du - depth));
    vec3 dpx = left
        ? p - flux_world_from_texel(inv_vp, full, best - ivec2(1, 0), dl, flip)
        : flux_world_from_texel(inv_vp, full, best + ivec2(1, 0), dr, flip) - p;
    vec3 dpy = down
        ? p - flux_world_from_texel(inv_vp, full, best - ivec2(0, 1), dd, flip)
        : flux_world_from_texel(inv_vp, full, best + ivec2(0, 1), du, flip) - p;
    vec3 c = cross(dpx, dpy);
    if (dot(c, c) < 1e-20) return false;
    n = normalize(c);
    return true;
}

// Numerical bias in world space. Scaling this with camera distance moved the
// traced surface as the view moved, changing occlusion near thin fixtures.
float flux_probe_offset(vec3 p) {
    return max(0.002, max(abs(p.x), max(abs(p.y), abs(p.z))) * 2e-6);
}

#endif // FLUX_COMMON
