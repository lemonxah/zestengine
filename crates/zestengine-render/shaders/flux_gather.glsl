// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Two screen probes per 32-thread workgroup. Each probe's guided cells own
// contiguous ray intervals; lanes trace fixed-size batches of those intervals.
// This balances ray counts while retaining cell-local averages and history.
shared vec4 gather_position[2]; // w = receiver distance, zero means sky/padding
shared vec3 gather_normal[2];
shared FluxHistory gather_history[2];
shared vec3 gather_radiance[32];
shared uint gather_samples[32];
shared uint gather_starts[32];
shared vec4 gather_batch[32]; // incoming radiance, first hit distance

vec4 gather_trace_sample(vec3 origin,vec3 n,uvec2 px,uint cell,uint sample_index,uint count) {
    // Ray identity is independent of its lane and other paths' RNG usage.
    uint rng=hash(px.x*1973u+px.y*9277u+pc.misc.x*26699u
        +cell*0x9e3779b9u+sample_index*0x85ebca6bu);
    vec2 jitter=vec2((float(sample_index)+rnd(rng))/float(count),rnd(rng));
    vec3 direction=cosine_hemisphere_dir(n,flux_direction_uv(cell,jitter));
    float distance;
#ifdef FLUX_GATHER_RT
    vec3 incoming_radiance=rq_incoming(origin,direction,pc.march.y,rng,distance);
#else
    vec3 incoming_radiance=incoming(origin,direction,pc.march.x,pc.march.y,pc.counts.y,rng,distance);
#endif
    return vec4(firefly_clamp(incoming_radiance,max(pc.march.w,0.5)),min(distance,65504.0));
}

void main() {
    uint lane = gl_LocalInvocationIndex;
    uint probe = lane / FLUX_DIRECTION_COUNT;
    uint bin = lane % FLUX_DIRECTION_COUNT;
    uvec2 size = pc.misc.yz;
    uvec2 px = uvec2(gl_WorkGroupID.x * 2u + probe, gl_WorkGroupID.y);
    bool in_bounds = all(lessThan(px, size));
    if (bin == 0u) {
        vec3 p, n;
        bool valid = false;
        if (in_bounds) valid = flux_probe_surface(u_depth, px, size,
            pc.inv_view_proj, pc.misc.w != 0u, p, n);
        gather_position[probe] = vec4(0.0);
        if (valid) {
            if (dot(n, pc.cam.xyz - p) < 0.0) n = -n;
            gather_position[probe] = vec4(p, length(p - pc.cam.xyz));
            gather_normal[probe] = n;
            gather_history[probe] = flux_history(p, n, size);
            vec4 lighting[4];uvec4 counts[4];
            flux_history_lighting(gather_history[probe],lighting);
            uint seed=hash(px.x*1973u+px.y*9277u+pc.misc.x*26699u);
            flux_allocate_rays(lighting,clamp((pc.counts.x+15u)/16u,1u,16u),rnd(seed),counts);
            uint prefix=0u;
            for(uint cell=0u;cell<16u;++cell) {
                uint count=counts[cell/4u][cell%4u];
                gather_starts[probe*16u+cell]=prefix;
                gather_samples[probe*16u+cell]=count;prefix+=count;
            }
        }
    }
    // Padding and sky lanes must participate in every workgroup barrier.
    barrier();
    gather_radiance[lane] = vec3(0.0);
    uint per_cell = clamp((pc.counts.x + 15u) / 16u, 1u, 16u);
    uint pidx = px.y * size.x + px.x;
    bool valid=gather_position[probe].w>0.0;
    vec3 origin=vec3(0.0),n=vec3(0.0);
    uint count=1u,start=0u;
    if(valid) {
        vec3 p=gather_position[probe].xyz;n=gather_normal[probe];
        float offset=flux_probe_offset(p);
#ifndef FLUX_GATHER_RT
        offset=max(offset,gdf_surface_offset(pc.march.x));
#endif
        origin=p+n*offset;
        count=gather_samples[lane];start=gather_starts[lane];
    }
    vec4 sum=vec4(0.0);
    if(pc.transport.z>0.5) {
        // Diagnostic reference: same rays and addition order, serial per cell.
        if(valid) for(uint i=0u;i<count;++i) {
            sum+=gather_trace_sample(origin,n,px,bin,i,count);
        }
    } else {
        for(uint batch=0u;batch<per_cell;++batch) {
            uint first=batch*16u,work=first+bin;
            if(valid) {
                uint lo=0u,hi=15u;
                for(uint step=0u;step<4u;++step) {
                    uint mid=(lo+hi+1u)/2u;
                    if(work>=gather_starts[probe*16u+mid]) lo=mid;else hi=mid-1u;
                }
                uint owner=probe*16u+lo;
                gather_batch[lane]=gather_trace_sample(origin,n,px,lo,
                    work-gather_starts[owner],gather_samples[owner]);
            }
            // Sky/padding lanes take both barriers too. Only one batch is kept
            // in shared memory, independent of quality and directional weights.
            barrier();
            if(valid) for(uint work=max(start,first);work<min(start+count,first+16u);++work) {
                uint at=probe*16u+work-first;
                sum+=gather_batch[at];
            }
            barrier();
        }
    }
    if(valid) {
        FluxRay ray;
        ray.radiance = sum.rgb * (pc.sky.w / float(count));
        ray.state = flux_ray_state(sum.w / float(count), 1.0);
        ray = flux_accumulate_direction(gather_history[probe], bin, ray);
        sh_out[pidx].rays[bin] = ray;
        gather_radiance[lane] = ray.radiance;
    }
    barrier();
    if (bin == 0u && in_bounds) {
        vec3 irr = vec3(0.0);
        for (uint i = 0u; i < FLUX_DIRECTION_COUNT; ++i) {
            irr += gather_radiance[lane + i] / float(FLUX_DIRECTION_COUNT);
        }
        float distance = gather_position[probe].w;
        // Receiver validity comes from the visible depth surface. A ray hitting
        // an opaque back face blocks that direction, not the other directions
        // of a valid receiver. A random back-face percentage used to discard
        // entire floor probes beside single-sided walls and reset their history.
        vec3 n = distance > 0.0 ? gather_normal[probe] : vec3(0.0);
        sh_out[pidx].c[0] = vec4(0.0, 0.0, 0.0, distance);
        sh_out[pidx].c[1] = vec4(0.0, 0.0, 0.0, n.x);
        sh_out[pidx].c[2] = vec4(0.0, 0.0, 0.0, n.y);
        sh_out[pidx].c[3] = vec4(0.0, 0.0, 0.0, n.z);
        sh_out[pidx].c[4] = vec4(0.0, gather_position[probe].xyz);
        imageStore(u_out, ivec2(px), vec4(irr, distance));
    }
}
