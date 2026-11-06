// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// One group traces a requested probe. Four octahedral texels per lane keep
// workgroups portable and the per-frame ray budget explicit (256 per probe).
void main() {
    uint queue_index=gl_WorkGroupID.x;
    if(queue_index>=min(rc_work[0],rc_config.z)) return;
    uint index=rc_work[RC_QUEUE_START+queue_index];ivec4 cell=rc_probes[index].cell;
    vec3 origin=rc_position(cell.xyz,cell.w);float start=rc_spacing[cell.w]*rc_spacing.w;
    for(uint lane=gl_LocalInvocationIndex;lane<256u;lane+=64u) {
        vec3 direction=rc_decode_direction((vec2(lane%16u,lane/16u)+0.5)/16.0);
        uint rng=hash(index*747796405u+lane*2891336453u);float distance;
        flux_world_source_valid=true;
#ifdef FLUX_RADIANCE_RT
        vec3 value=rq_incoming(origin+direction*start,direction,max(pc.march.y-start,0.01),rng,distance);
#else
        vec3 value=incoming(origin+direction*start,direction,pc.march.x,max(pc.march.y-start,0.01),pc.counts.y,rng,distance);
#endif
        value=clamp(value,vec3(0),vec3(65000));distance=flux_world_source_valid ? min(distance+start,65000.0) : -1.0;
        rc_probes[index].rays[lane]=uvec2(packHalf2x16(value.rg),packHalf2x16(vec2(value.b,distance)));
    }
    if(gl_LocalInvocationIndex==0u) {
        rc_probes[index].state.x=rc_config.y;rc_probes[index].state.y=rc_config.x;rc_probes[index].state.w=1u;
    }
}
