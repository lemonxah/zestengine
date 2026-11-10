// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Screen work writes deterministic sample slots and compacts only unresolved
// rays. World tracing fills those slots; resolve uses the original addition
// order, independent of atomic append order or workgroup scheduling.
struct FluxTraceSample { vec4 result; uvec4 identity; };
layout(constant_id=3) const bool FLUX_QUEUE_VERIFY=false;
#ifndef FLUX_QUEUE_BINDING
#define FLUX_QUEUE_BINDING 27
#define FLUX_QUEUE_WIDTH 32u
#endif
layout(set=0,binding=FLUX_QUEUE_BINDING) buffer FluxTraceQueue {
    uint queue_x; uint queue_y; uint queue_z; uint queue_count;
    uint queue_allocated; uint queue_reserved0; uint queue_reserved1; uint queue_reserved2;
    uint queue_slots[];
};
layout(set=0,binding=FLUX_QUEUE_BINDING+1) buffer FluxTraceSamples { FluxTraceSample queue_samples[]; };
#include "flux_ray_sample.glsl"

#ifdef FLUX_QUEUE_WORLD
void main() {
    uint job=(gl_WorkGroupID.x+gl_WorkGroupID.y*65535u)*FLUX_QUEUE_WIDTH+gl_LocalInvocationIndex;
    if(job>=queue_count) return;
    uint slot=queue_slots[job];
    uvec4 id=queue_samples[slot].identity;
    uvec2 px=uvec2(id.x%pc.misc.y,id.x/pc.misc.y);
    vec3 n=vec3(sh_out[id.x].c[1].w,sh_out[id.x].c[2].w,sh_out[id.x].c[3].w);
    vec3 p=sh_out[id.x].c[4].yzw;
    uint rng=flux_trace_seed(px,id.y,id.z);
    vec3 direction=flux_trace_direction(n,id.y,id.z,id.w,rng);
    float distance;
#ifdef FLUX_GATHER_RT
    vec3 radiance=rq_incoming(p+n*flux_probe_offset(p),direction,pc.march.y,rng,distance);
#else
    vec3 radiance=incoming(sh_out[id.x].trace_origin.xyz,direction,pc.march.x,pc.march.y,pc.counts.y,rng,distance);
#endif
    queue_samples[slot].result=vec4(flux_limit_gather(radiance,flux_gather_max_intensity()),min(distance,65504.0));
}
#endif
