// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#ifndef FLUX_RAY_SAMPLE_GLSL
#define FLUX_RAY_SAMPLE_GLSL
// Each ray owns its RNG stream: shading or queue order cannot change the next
// ray's direction. Both inline and split schedules use the same identities.
uint flux_trace_seed(uvec2 px,uint cell,uint sample_index) {
    return hash(px.x*1973u+px.y*9277u+pc.misc.x*26699u
        +cell*0x9e3779b9u+sample_index*0x85ebca6bu);
}
vec3 flux_trace_direction(vec3 n,uint cell,uint sample_index,uint count,inout uint rng) {
    vec2 jitter=vec2((float(sample_index)+rnd(rng))/float(count),rnd(rng));
    return cosine_hemisphere_dir(n,flux_direction_uv(cell,jitter));
}
#endif
