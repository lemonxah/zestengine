// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Include after `rnd`.
// Where on a light a shadow ray aims. Independent draws by default. A caller
// that accumulates one point over many updates (surface-cache direct light)
// sets a progressive low-discrepancy point instead: a penumbra then converges
// far faster than with independent draws. Each light shifts the same point.
#ifndef FLUX_LIGHT_SAMPLE
#define FLUX_LIGHT_SAMPLE
vec2 flux_light_point = vec2(-1.0);
vec2 flux_light_sample(uint light, inout uint rng) {
    if (flux_light_point.x < 0.0) return vec2(rnd(rng), rnd(rng));
    return fract(flux_light_point + float(light) * vec2(0.6180339887, 0.3247179572));
}
#endif
