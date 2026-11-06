// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Water-only focusing, consuming the shared scene-light inputs.
// Slope (local XZ gradient) and, when `want_hess`, Hessian (xx, zz, xz) of
// body `s`'s surface over its local point `xz`, band-limited to `foot`
// metres: the Gerstner sum, the simulated field and the fine scrolling
// ripples (world `wxz`), the same surface the water shader draws.
void water_surface_slope(int s, vec2 xz, vec2 wxz, float foot, bool want_hess, out vec2 g, out vec3 h) {
    g = vec2(0.0);
    h = vec3(0.0);
    float t = frame.water.x;
    int n = min(int(water_scene.bodies[s].look1.z + 0.5), MAX_WATER_WAVES);
    for (int i = 0; i < n; ++i) {
        vec4 a = water_scene.bodies[s].wave_a[i];
        float amp = water_scene.bodies[s].wave_b[i].y * water_band(a.z, foot);
        float phase = a.z * dot(a.xy, xz) - a.w * t;
        g += (amp * a.z * cos(phase)) * a.xy;
        h -= (amp * a.z * a.z * sin(phase)) * vec3(a.x * a.x, a.y * a.y, a.x * a.y);
    }
    vec4 r = water_scene.bodies[s].ripple;
    vec2 uv = water_ripple_uv(s, xz);
    float fade = water_ripple_fade(uv);
    if (fade > 0.0) {
        float size = float(textureSize(u_water_ripples[s], 0).x);
        float lod = water_ripple_lod(foot, r.z, size);
        g += textureLod(u_water_ripples[s], uv, lod).gb * fade;
        if (want_hess) {
            // Central differences of the gradient a texel of that mip apart.
            float e = exp2(floor(lod)) / size;
            vec2 px = textureLod(u_water_ripples[s], uv + vec2(e, 0.0), lod).gb;
            vec2 nx = textureLod(u_water_ripples[s], uv - vec2(e, 0.0), lod).gb;
            vec2 pz = textureLod(u_water_ripples[s], uv + vec2(0.0, e), lod).gb;
            vec2 nz = textureLod(u_water_ripples[s], uv - vec2(0.0, e), lod).gb;
            h += fade * vec3(px.x - nx.x, pz.y - nz.y, 0.5 * (px.y - nx.y + pz.x - nz.x)) / (2.0 * e * r.z);
        }
    }
    vec4 look0 = water_scene.bodies[s].look0;
    vec2 flow = water_scene.bodies[s].reflect.yz; // the body's current (local m/s)
    float scale = max(look0.z, 0.01);
    float k = look0.y * 0.55;
    if (k > 0.0) {
        // The same flow-carrying ripple mix the surface itself draws
        // (water_ripples): two half-period phases cross-fading, advected by
        // the current, so the caustics drift with it instead of flickering
        // in place.
        const float PERIOD = 4.0;
        float ph0 = fract(t / PERIOD);
        float ph1 = fract(t / PERIOD + 0.5);
        float w0 = 1.0 - abs(1.0 - 2.0 * ph0);
        vec2 g0 = water_ripple_slope(wxz - flow * (ph0 * PERIOD), t, scale, foot);
        vec2 g1 = water_ripple_slope(wxz - flow * (ph1 * PERIOD) + vec2(0.37, 0.71), t, scale, foot);
        g += mix(g1, g0, w0) * k;
        if (want_hess) {
            float e = 0.05 / scale;
            vec2 gx = mix(water_ripple_slope(wxz + vec2(e, 0.0) - flow * (ph1 * PERIOD) + vec2(0.37, 0.71), t, scale, foot), water_ripple_slope(wxz + vec2(e, 0.0) - flow * (ph0 * PERIOD), t, scale, foot), w0);
            vec2 gz = mix(water_ripple_slope(wxz + vec2(0.0, e) - flow * (ph1 * PERIOD) + vec2(0.37, 0.71), t, scale, foot), water_ripple_slope(wxz + vec2(0.0, e) - flow * (ph0 * PERIOD), t, scale, foot), w0);
            vec2 gmid = mix(g1, g0, w0);
            h += (k / e) * vec3(gx.x - gmid.x, gz.y - gmid.y, 0.5 * (gx.y - gmid.y + gz.x - gmid.x));
        }
    }
}

// Caustic focusing through body `s`'s surface: light crossing it at local
// `entry` bends by the slope there and lands `kappa` x slope further on, so a
// receiver gets 1/det(I + kappa H) of it, read at the surface point whose
// light lands here. Averaged over the receiver that keeps its brightness.
float water_focus(int s, vec2 entry, vec2 to_world, float kappa, float foot) {
    vec2 g;
    vec3 h;
    water_surface_slope(s, entry, entry + to_world, foot, false, g, h);
    vec2 src = entry - kappa * g;
    water_surface_slope(s, src, src + to_world, foot, true, g, h);
    float det = (1.0 + kappa * h.x) * (1.0 + kappa * h.y) - kappa * kappa * h.z * h.z;
    return min(inversesqrt(det * det + 0.01), 8.0);
}

// Light gain from caustics at world `p` for one light (1 = none). `travel`
// is a directional light's direction of travel; a point/spot light is at
// `light_pos` (`point`).
//   Under a surface, lit from above: light refracted into the water focuses
// where the surface curves, bent by (1 - 1/1.33) of the slope over the
// slanted depth. Caustic strength (bounds.w) scales the focusing.
//   Above a pool's surface (extra.x > 0), lit from a light under it: the
// light leaves the water bent the other way, by (1.33 - 1) of the slope over
// its path through the air, and throws the moving pattern on walls and
// ceiling; extra.x scales that light (1 physical), and the Fresnel loss at
// the surface takes the grazing rays.
float water_caustic_gain(vec3 p, vec3 travel, vec3 light_pos, bool point) {
    int n = min(int(frame.water.z + 0.5), MAX_WATER_BODIES);
    float pix = length(fwidth(p));
    for (int s = 0; s < n; ++s) {
        vec4 b = water_scene.bodies[s].bounds;
        float ior = max(water_absorption(water_scene.bodies[s]).w, 1.0);
        mat4 to_local = water_scene.bodies[s].to_local;
        vec3 l = (to_local * vec4(p, 1.0)).xyz;
        vec3 lp = (to_local * vec4(light_pos, 1.0)).xyz;
        vec2 to_world = p.xz - l.xz;
        float depth = b.z - l.y;
        bool inside = b.x <= 0.0 || (abs(l.x) <= b.x && abs(l.z) <= b.y);
        if (depth > 0.005 && inside) {
            if (b.w <= 0.0 || (point && lp.y < b.z)) {
                return 1.0;
            }
            vec3 dir = point ? normalize(l - lp) : normalize(mat3(to_local) * travel);
            vec3 tr = refract(dir, vec3(0.0, 1.0, 0.0), 1.0 / ior);
            if (dir.y >= -0.001 || tr.y >= -0.001) continue;
            float slant = depth / -tr.y;
            float kappa = b.w * slant * (1.0 - 1.0 / ior);
            vec2 entry = l.xz - tr.xz * slant;
            // Deeper floors see the pattern softer (lights have a size).
            float foot = max(pix, 0.02 + 0.03 * depth);
            vec3 transmission = water_transmittance(water_absorption(water_scene.bodies[s]).rgb, water_scattering_coeff(water_scene.bodies[s]).rgb, slant);
            float contrast = dot(transmission, vec3(0.2126, 0.7152, 0.0722));
            return mix(1.0, water_focus(s, entry, to_world, kappa, foot), contrast);
        }
        float strength = water_scene.bodies[s].extra.x;
        float under = b.z - lp.y;
        if (!point || strength <= 0.0 || depth > 0.0 || under <= 0.005) {
            continue;
        }
        if (b.x > 0.0 && (abs(lp.x) > b.x || abs(lp.z) > b.y)) {
            continue;
        }
        // Where the bent ray leaves the water: 1.33 sin(water angle) = sin(air
        // angle) along the line from the light to here.
        float rise = -depth;
        vec2 across = l.xz - lp.xz;
        float reach = length(across);
        float lo = 0.0;
        float hi = reach;
        for (int k = 0; k < 10; ++k) {
            float r = 0.5 * (lo + hi);
            float sw = r / sqrt(r * r + under * under);
            float sa = (reach - r) / sqrt((reach - r) * (reach - r) + rise * rise);
            if (ior * sw > sa) {
                hi = r;
            } else {
                lo = r;
            }
        }
        float r = 0.5 * (lo + hi);
        vec2 entry = lp.xz + (reach > 1e-5 ? across / reach : vec2(0.0)) * r;
        if (b.x > 0.0 && (abs(entry.x) > b.x || abs(entry.y) > b.y)) {
            return 0.0;
        }
        float air = sqrt((reach - r) * (reach - r) + rise * rise);
        float cos_air = rise / max(air, 1e-4);
        float f0 = water_normal_reflectance(ior);
        float fresnel = f0 + (1.0 - f0) * pow(1.0 - clamp(cos_air, 0.0, 1.0), 5.0);
        // A point source's spreading beam dilutes the focusing far away.
        float kappa = (ior - 1.0) * air * (ior * under) / (ior * under + air);
        float foot = max(pix * 0.5, 0.01 + 0.01 * air);
        return strength * (1.0 - fresnel) * water_focus(s, entry, to_world, kappa, foot);
    }
    return 1.0;
}
