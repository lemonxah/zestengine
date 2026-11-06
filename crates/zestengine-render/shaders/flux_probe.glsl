// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#ifndef FLUX_PROBE
#define FLUX_PROBE

// Equal-probability cells in a cosine-weighted hemisphere. Keep directional
// radiance until AFTER filtering; averaging paths into RGB first loses the
// visibility information needed to reject unrelated light transport.
const uint FLUX_DIRECTION_COUNT = 16u;
struct FluxRay {
    vec3 radiance;
    // Half precision distance and valid history length (not radiance).
    uint state;
};
// std430: 448 bytes, mirrored by SCREEN_PROBE_BYTES in gpu_gi.rs.
struct ScreenProbe {
    // c0.w = camera distance; c1..3.w = normal; c4.yzw = world position.
    vec4 c[5];
    FluxRay rays[16];
    // Before adaptive software tracing, setup/placement use these lanes for
    // lighting-guide/count scratch. After resolve, only the spatial filter
    // writes them. Neighbours read rays/metadata, never these lanes, so the
    // filter needs no in-place neighbour read/write hazard.
    vec4 filtered_sh[4];
    // Software detail tracing shares receiver setup across directional jobs.
    uvec4 history_indices;
    vec4 history_weights;
    vec4 trace_origin;
};

vec2 flux_direction_uv(uint bin, vec2 jitter) {
    return (vec2(bin & 3u, bin >> 2u) + jitter) * 0.25;
}

uint flux_ray_state(float distance, float frames) {
    return packHalf2x16(vec2(clamp(distance, 0.0, 65504.0), frames));
}

float flux_distance_weight(float a, float b, float receiver_distance) {
    // Angular cells have finite extent. Reject clearly different occluders
    // without requiring independently jittered rays to hit the identical point.
    float tolerance = 0.05 + 0.25 * min(a, b) + receiver_distance;
    float d = abs(a - b) / tolerance;
    return exp(-d * d);
}

#endif
