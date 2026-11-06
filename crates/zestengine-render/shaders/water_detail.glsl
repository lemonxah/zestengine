// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Water-only flow detail and foam. Included after the shared surface inputs.
// Gradient noise with analytic derivatives: (value, d/dx, d/dy).
vec2 water_hash(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return fract(sin(p) * 43758.5453) * 2.0 - 1.0;
}

vec3 water_noised(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    vec2 du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
    vec2 ga = water_hash(i);
    vec2 gb = water_hash(i + vec2(1.0, 0.0));
    vec2 gc = water_hash(i + vec2(0.0, 1.0));
    vec2 gd = water_hash(i + vec2(1.0, 1.0));
    float va = dot(ga, f);
    float vb = dot(gb, f - vec2(1.0, 0.0));
    float vc = dot(gc, f - vec2(0.0, 1.0));
    float vd = dot(gd, f - vec2(1.0, 1.0));
    float k = va - vb - vc + vd;
    float v = va + u.x * (vb - va) + u.y * (vc - va) + u.x * u.y * k;
    vec2 d = ga + u.x * (gb - ga) + u.y * (gc - ga) + u.x * u.y * (ga - gb - gc + gd)
           + du * (u.yx * k + vec2(vb, vc) - va);
    return vec3(v, d);
}

// Two drifting ripple layers, slopes only.
vec2 water_ripple_slope(vec2 p, float t, float scale) {
    vec2 p1 = p * scale * 0.9 + vec2(t * 0.05, t * 0.031);
    vec2 p2 = p * scale * 2.3 + vec2(-t * 0.043, t * 0.067);
    return water_noised(p1).yz * 0.6 + water_noised(p2).yz * 0.4;
}

// The same two layers' height (their value field; the slope above is its
// gradient). Used where the surface height itself matters: the planar
// reflection's warp.
float water_ripple_height(vec2 p, float t, float scale) {
    vec2 p1 = p * scale * 0.9 + vec2(t * 0.05, t * 0.031);
    vec2 p2 = p * scale * 2.3 + vec2(-t * 0.043, t * 0.067);
    return water_noised(p1).x * 0.6 + water_noised(p2).x * 0.4;
}

// Ripples tilting the wave normal, carried along the world `flow` (m/s) by
// two half-period-offset phases that cross-fade (a flow map), faded out
// where a pixel covers more than a ripple (no sparkle in the distance).
vec3 water_ripples(vec3 N, vec2 p, float t, float strength, float scale, vec2 flow) {
    const float PERIOD = 2.0;
    float ph0 = fract(t / PERIOD);
    float ph1 = fract(t / PERIOD + 0.5);
    float w0 = 1.0 - abs(1.0 - 2.0 * ph0);
    vec2 g0 = water_ripple_slope(p - flow * (ph0 * PERIOD), t, scale);
    vec2 g1 = water_ripple_slope(p - flow * (ph1 * PERIOD) + vec2(0.37, 0.71), t, scale);
    vec2 g = mix(g1, g0, w0);
    float fade = 1.0 - smoothstep(0.35, 1.2, length(fwidth(p * scale * 2.3)));
    return normalize(N - vec3(g.x, 0.0, g.y) * (strength * 0.25 * fade));
}

// Foam cells: the edges of a jittered cell grid (second-nearest minus nearest
// feature distance), so foam reads as a lacy network of bubbles rather than
// smooth noise. 1 on the bubble walls.
float water_bubbles(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float f1 = 8.0;
    float f2 = 8.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec2 g = vec2(x, y);
            vec2 o = water_hash(i + g) * 0.5 + 0.5;
            float d = length(g + o - f);
            if (d < f1) {
                f2 = f1;
                f1 = d;
            } else if (d < f2) {
                f2 = d;
            }
        }
    }
    return 1.0 - smoothstep(0.0, 0.18, f2 - f1);
}

// Foam texture: two bubble scales carried along `flow` (m/s) by a flow map's
// two cross-fading phases; `density` fills the cells in so thick foam reads
// solid and thin foam stays lacy.
float water_foam_pattern(vec2 p, float t, vec2 flow, float density) {
    const float PERIOD = 3.0;
    float ph0 = fract(t / PERIOD);
    float ph1 = fract(t / PERIOD + 0.5);
    float w0 = 1.0 - abs(1.0 - 2.0 * ph0);
    vec2 a = p - flow * (ph0 * PERIOD);
    vec2 b = p - flow * (ph1 * PERIOD) + vec2(3.7, 1.3);
    float f0 = water_bubbles(a * 3.0) * 0.6 + water_bubbles(a * 7.0 + 1.7) * 0.4;
    float f1 = water_bubbles(b * 3.0) * 0.6 + water_bubbles(b * 7.0 + 1.7) * 0.4;
    return clamp(mix(mix(f1, f0, w0), 1.0, density * density), 0.0, 1.0);
}

