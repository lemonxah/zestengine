// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Gather includes this after its descriptor declarations. Reprojection validates
// each receiver before accepting directional history. A fresh or
// disoccluded direction starts at one sample instead of inheriting a fixed long
// history. Lighting can decrease as well as increase; this is a running mean.
struct FluxHistory { uvec4 indices; vec4 weights; };

vec3 flux_history_barycentric(vec2 p, vec2 a, vec2 b, vec2 c) {
    vec2 u = b-a, v = c-a, d = p-a;
    float det = u.x*v.y-u.y*v.x;
    if (abs(det) < 1e-6) return vec3(-1.0);
    float y = (d.x*v.y-d.y*v.x)/det;
    float z = (u.x*d.y-u.y*d.x)/det;
    return vec3(1.0-y-z, y, z);
}

FluxHistory flux_history(vec3 p, vec3 n, uvec2 size) {
    FluxHistory history;
    history.indices = uvec4(0u);
    history.weights = vec4(0.0);
    float min_alpha = clamp(pc.temporal.x, 0.02, 1.0);
    if (min_alpha >= 0.999 || pc.march.z >= 0.999) return history;
    vec4 clip = pc.prev_view_proj * vec4(p, 1.0);
    if (clip.w <= 1e-4) return history;
    vec2 uv = clip.xy / clip.w;
    if (pc.misc.w != 0u) uv.y = -uv.y;
    uv = uv * 0.5 + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return history;
    vec2 g = uv * vec2(size) - 0.5;
    // A probe samples the nearest depth pixel in its tile, not its centre.
    // Interpolating by nominal tile centres repeatedly blurs history sideways
    // even for a fixed camera. Reproject the actual previous receivers, then
    // interpolate on their screen-space triangles. A 3x3 neighbourhood covers
    // the possible within-tile displacement in either direction.
    ivec2 centre = ivec2(floor(g + 0.5));
    vec2 positions[9];
    uint indices[9];
    float valid[9];
    float nearest = 1e10;
    uint fallback = 0u;
    for (int k = 0; k < 9; ++k) {
        ivec2 q = centre + ivec2(k % 3 - 1, k / 3 - 1);
        positions[k] = vec2(q);
        indices[k] = 0u;
        valid[k] = 0.0;
        if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, ivec2(size)))) continue;
        uint index = uint(q.y) * size.x + uint(q.x);
        indices[k] = index;
        if (sh_prev[index].c[0].w <= 0.0) continue;
        vec4 previous_clip = pc.prev_view_proj * vec4(sh_prev[index].c[4].yzw, 1.0);
        if (previous_clip.w <= 1e-4) continue;
        vec2 previous_uv = previous_clip.xy / previous_clip.w;
        if (pc.misc.w != 0u) previous_uv.y = -previous_uv.y;
        positions[k] = (previous_uv * 0.5 + 0.5) * vec2(size) - 0.5;
        vec3 pn = vec3(sh_prev[index].c[1].w, sh_prev[index].c[2].w, sh_prev[index].c[3].w);
        vec3 delta = p - sh_prev[index].c[4].yzw;
        float tolerance = 0.015 + 0.02 * length(delta);
        // The hemisphere basis changes handedness across n.z=-0.5. Its cell
        // indices cannot be reused there without a directional remap.
        if (dot(n, pn) < 0.98
            || (n.z >= -0.5) != (pn.z >= -0.5)
            || abs(dot(delta, n)) > tolerance || abs(dot(delta, pn)) > tolerance) continue;
        valid[k] = 1.0;
        float distance2 = dot(positions[k]-g, positions[k]-g);
        if (distance2 < nearest) { nearest = distance2; fallback = index; }
    }
    for (int quad = 0; quad < 4; ++quad) {
        int a = (quad / 2) * 3 + quad % 2;
        for (int tri = 0; tri < 2; ++tri) {
            ivec3 at = tri == 0 ? ivec3(a, a+1, a+4) : ivec3(a, a+4, a+3);
            vec3 w = flux_history_barycentric(g, positions[at.x], positions[at.y], positions[at.z]);
            if (any(lessThan(w, vec3(-1e-4)))) continue;
            w = max(w, vec3(0.0));
            w /= max(w.x+w.y+w.z, 1e-6);
            w *= vec3(valid[at.x], valid[at.y], valid[at.z]);
            history.indices = uvec4(indices[at.x], indices[at.y], indices[at.z], 0u);
            history.weights = vec4(w, 0.0);
            return history;
        }
    }
    // At a viewport edge there may be no complete triangle. Reuse only a
    // nearby validated surface, with confidence falling off beyond one tile.
    if (nearest < 2.25) {
        history.indices.x = fallback;
        history.weights.x = 1.0-smoothstep(1.0, 2.25, nearest);
    }
    return history;
}

FluxRay flux_accumulate_direction(FluxHistory history, uint bin, FluxRay ray) {
    float min_alpha = clamp(pc.temporal.x, 0.02, 1.0);
    float distance = unpackHalf2x16(ray.state).x;
    vec3 radiance = vec3(0.0);
    float weight = 0.0, frames = 0.0;
    for (int k = 0; k < 4; ++k) {
        if (history.weights[k] <= 0.0) continue;
        FluxRay prev = sh_prev[history.indices[k]].rays[bin];
        vec2 state = unpackHalf2x16(prev.state);
        if (state.y < 1.0 || any(isnan(prev.radiance)) || any(isinf(prev.radiance))) continue;
        // Different jittered hits within a cell are not disocclusions. Receiver
        // validity controls history; hit-distance rejection is spatial only.
        float w = history.weights[k];
        radiance += prev.radiance * w;
        frames += state.y * w;
        weight += w;
    }
    if (weight < 0.05) return ray;
    float count = min(1.0 + frames, 1.0 / min_alpha);
    float alpha = max(1.0 / count, 1.0 - weight);
    ray.radiance = mix(radiance / weight, ray.radiance, alpha);
    ray.state = flux_ray_state(distance, 1.0 / alpha);
    return ray;
}

#include "flux_importance.glsl"
// Only geometrically validated history may guide a new receiver. A miss or
// invalid sample contributes no preference, giving a uniform startup budget.
void flux_history_lighting(FluxHistory history,out vec4 lighting[4]) {
    for(uint group=0u;group<4u;++group) lighting[group]=vec4(0.0);
    if(pc.transport.y>0.5) return; // diagnostic uniform-budget reference
    for(uint k=0u;k<4u;++k) {
        if(history.weights[k]<=0.0) continue;
        for(uint bin=0u;bin<16u;++bin) {
            FluxRay previous=sh_prev[history.indices[k]].rays[bin];
            if(unpackHalf2x16(previous.state).y<1.0 || any(isnan(previous.radiance)) || any(isinf(previous.radiance))) continue;
            lighting[bin/4u][bin%4u]+=dot(max(previous.radiance,vec3(0.0)),vec3(0.2126,0.7152,0.0722))*history.weights[k];
        }
    }
}
