// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Raw current-frame samples must not alias accumulated history: neighbouring
// receivers interpolate these samples while writing their own history records.
layout(set=0,binding=25) buffer RawRays {
    uint job_count;
    uint dispatch_x; uint dispatch_y; uint dispatch_z;
    FluxRay raw_rays[]; // dense samples, then a compact list of trace owners
};

vec3 receiver_normal(uint i) {
    return vec3(sh_out[i].c[1].w,sh_out[i].c[2].w,sh_out[i].c[3].w);
}
bool compatible_receivers(uint a,uint b) {
    if(sh_out[a].c[0].w<=0.0 || sh_out[b].c[0].w<=0.0) return false;
    vec3 na=receiver_normal(a),nb=receiver_normal(b);
    vec3 delta=sh_out[a].c[4].yzw-sh_out[b].c[4].yzw;
    float tolerance=0.005+0.001*min(sh_out[a].c[0].w,sh_out[b].c[0].w);
    return (na.z>=-0.5)==(nb.z>=-0.5) && dot(na,nb)>0.995 && max(abs(dot(delta,na)),abs(dot(delta,nb)))<tolerance
        && dot(delta,delta)<0.25;
}

// A 16px tile may share a probe only if EVERY dense receiver belongs to the
// same local plane. Otherwise subdivide to 8px and finally 4px. Empty pixels,
// thin silhouettes, corners and curved surfaces therefore retain detail.
bool planar_tile(uvec2 base,uint span,uvec2 size,out uint owner) {
    uvec2 center=min(base+uvec2((span-1u)/2u),size-1u);
    owner=center.y*size.x+center.x;
    if(sh_out[owner].c[0].w<=0.0) return false;
    for(uint y=0u;y<span;++y) for(uint x=0u;x<span;++x) {
        uvec2 q=base+uvec2(x,y);
        if(any(greaterThanEqual(q,size))) continue;
        if(!compatible_receivers(owner,q.y*size.x+q.x)) return false;
    }
    return true;
}
void assign_tile(uvec2 base,uint span,uvec2 size,uint owner) {
    if(sh_out[owner].c[0].w>0.0) {
        // Receiver setup stores validated lighting in filtered_sh as scratch.
        // Placement turns it into counts; the spatial filter overwrites this
        // storage only after tracing and resolving every receiver.
        vec4 lighting[4];uvec4 counts[4];
        for(uint group=0u;group<4u;++group) lighting[group]=sh_out[owner].filtered_sh[group];
        uint seed=hash((owner%size.x)*1973u+(owner/size.x)*9277u+pc.misc.x*26699u);
        flux_allocate_rays(lighting,clamp(((pc.counts.x+15u)/16u)*span,1u,16u),rnd(seed),counts);
        for(uint group=0u;group<4u;++group) sh_out[owner].filtered_sh[group]=vec4(counts[group]);
        uint job=atomicAdd(job_count,1u);
        raw_rays[size.x*size.y*FLUX_DIRECTION_COUNT+job].state=owner;
        atomicMax(dispatch_x,(job+64u)/64u);
    }
    for(uint y=0u;y<span;++y) for(uint x=0u;x<span;++x) {
        uvec2 q=base+uvec2(x,y);
        if(any(greaterThanEqual(q,size))) continue;
        uint index=q.y*size.x+q.x;
        sh_out[index].trace_origin.w=uintBitsToFloat(owner);
        sh_out[index].c[0].x=float(span);
    }
}

// Separate receiver work from direction tracing. Each ray workgroup contains
// neighbouring receivers in one directional cell, preserving SDF locality.
void main() {
    uvec2 px=gl_GlobalInvocationID.xy, size=pc.misc.yz;
#ifdef FLUX_DIRECTIONAL_TRACE
    uint job=gl_WorkGroupID.x*64u+gl_LocalInvocationIndex;
    if(job>=job_count) return;
    uint receiver=raw_rays[size.x*size.y*FLUX_DIRECTION_COUNT+job].state;
    px=uvec2(receiver%size.x,receiver/size.x);
#endif
#ifdef FLUX_ADAPTIVE_PLACEMENT
    uvec2 base=px*4u;
    if(any(greaterThanEqual(base,size))) return;
    // Diagnostic dense reference keeps the identical trace/resolve path.
    if(pc.gdf_min.w>0.5) {
        for(uint y=0u;y<4u;++y) for(uint x=0u;x<4u;++x) {
            uvec2 q=base+uvec2(x,y);
            if(all(lessThan(q,size))) assign_tile(q,1u,size,q.y*size.x+q.x);
        }
        return;
    }
    uint owner;
    if(pc.counts.x<64u && planar_tile(base,4u,size,owner)) { assign_tile(base,4u,size,owner); return; }
    for(uint y=0u;y<4u;y+=2u) for(uint x=0u;x<4u;x+=2u) {
        uvec2 sub=base+uvec2(x,y);
        if(any(greaterThanEqual(sub,size))) continue;
        if(planar_tile(sub,2u,size,owner)) assign_tile(sub,2u,size,owner);
        else for(uint j=0u;j<2u;++j) for(uint i=0u;i<2u;++i) {
            uvec2 q=sub+uvec2(i,j);
            if(all(lessThan(q,size))) assign_tile(q,1u,size,q.y*size.x+q.x);
        }
    }
#else
    if(any(greaterThanEqual(px,size))) return;
    uint index=px.y*size.x+px.x;
#ifdef FLUX_RECEIVER_SETUP
    vec3 p,n;
    if(!flux_probe_surface(u_depth,px,size,pc.inv_view_proj,pc.misc.w!=0u,p,n)) {
        for(uint c=0u;c<5u;++c) sh_out[index].c[c]=vec4(0.0);
        return;
    }
    if(dot(n,pc.cam.xyz-p)<0.0) n=-n;
    sh_out[index].c[0]=vec4(0,0,0,length(p-pc.cam.xyz));
    sh_out[index].c[1]=vec4(0,0,0,n.x);
    sh_out[index].c[2]=vec4(0,0,0,n.y);
    sh_out[index].c[3]=vec4(0,0,0,n.z);
    sh_out[index].c[4]=vec4(0,p);
    FluxHistory history=flux_history(p,n,size);
    sh_out[index].history_indices=history.indices;
    sh_out[index].history_weights=history.weights;
    vec4 lighting[4];flux_history_lighting(history,lighting);
    for(uint group=0u;group<4u;++group) sh_out[index].filtered_sh[group]=lighting[group];
    sh_out[index].trace_origin=vec4(p+n*max(software_surface_offset(p,pc.march.x),flux_probe_offset(p)),0);
#elif defined(FLUX_ADAPTIVE_RESOLVE)
    float cam_distance=sh_out[index].c[0].w;
    vec3 irradiance=vec3(0);
    if(cam_distance>0.0) {
        uint own=floatBitsToUint(sh_out[index].trace_origin.w);
        uint span=uint(sh_out[index].c[0].x);
        // Interpolate neighbouring trace owners with compact screen-space tent
        // weights, rejecting disconnected planes before any radiance is read.
        // Fine receivers keep their own sample. Coarse planar regions blend
        // across tile boundaries instead of displaying constant lighting blocks.
        uint indices[9]; float weights[9]; uint used=0u; float total=0.0;
        for(int y=-1;y<=1;++y) for(int x=-1;x<=1;++x) {
            ivec2 q=ivec2(px)+ivec2(x,y)*int(span);
            if(any(lessThan(q,ivec2(0))) || any(greaterThanEqual(q,ivec2(size)))) continue;
            uint candidate=floatBitsToUint(sh_out[uint(q.y)*size.x+uint(q.x)].trace_origin.w);
            if(!compatible_receivers(index,candidate)) continue;
            vec2 delta=abs(vec2(px)-vec2(candidate%size.x,candidate/size.x));
            vec2 tent=max(vec2(0),vec2(1)-delta/float(span));
            float w=tent.x*tent.y;
            if(w<=0.0) continue;
            indices[used]=candidate; weights[used]=w; total+=w; ++used;
        }
        if(used==0u) {indices[0]=own; weights[0]=1.0; total=1.0; used=1u;}
        FluxHistory history;
        history.indices=sh_out[index].history_indices;
        history.weights=sh_out[index].history_weights;
        for(uint bin=0u;bin<FLUX_DIRECTION_COUNT;++bin) {
            vec3 radiance=vec3(0); float distance=0.0;
            for(uint j=0u;j<used;++j) {
                FluxRay sample_ray=raw_rays[indices[j]*FLUX_DIRECTION_COUNT+bin];
                float w=weights[j]/total;
                radiance+=w*sample_ray.radiance;
                distance+=w*unpackHalf2x16(sample_ray.state).x;
            }
            FluxRay ray;
            ray.radiance=radiance; ray.state=flux_ray_state(distance,1.0);
            ray=flux_accumulate_direction(history,bin,ray);
            sh_out[index].rays[bin]=ray;
            irradiance+=ray.radiance/float(FLUX_DIRECTION_COUNT);
        }
    }
    imageStore(u_out,ivec2(px),vec4(irradiance,cam_distance));
#else
    if(sh_out[index].c[0].w<=0.0 || floatBitsToUint(sh_out[index].trace_origin.w)!=index) return;
    uint bin=gl_WorkGroupID.z;
    vec3 n=vec3(sh_out[index].c[1].w,sh_out[index].c[2].w,sh_out[index].c[3].w);
    vec3 origin=sh_out[index].trace_origin.xyz;
    // Coarse probes cover more receivers. Spend part of the saved work on
    // directional variance: sample count grows with linear tile size, while
    // receiver coverage grows quadratically. Ultra also refines flat regions
    // to 8px. Dense 4px edge receivers keep their full requested ray budget.
    uint count=uint(sh_out[index].filtered_sh[bin/4u][bin%4u]);
    uint rng=hash(px.x*1973u+px.y*9277u+pc.misc.x*26699u+bin*0x9e3779b9u);
    vec3 radiance=vec3(0.0); float distance=0.0;
    for(uint i=0u;i<count;++i) {
        vec2 jitter=vec2((float(i)+rnd(rng))/float(count),rnd(rng));
        vec3 d=cosine_hemisphere_dir(n,flux_direction_uv(bin,jitter));
        float hit_distance;
        radiance+=firefly_clamp(incoming(origin,d,pc.march.x,pc.march.y,pc.counts.y,rng,hit_distance),max(pc.march.w,0.5));
        distance+=min(hit_distance,65504.0);
    }
    FluxRay ray;
    ray.radiance=radiance*(pc.sky.w/float(count));
    // Raw storage keeps the actual trace count for diagnostics; resolve
    // creates its own temporal history length.
    ray.state=flux_ray_state(distance/float(count),float(count));
    raw_rays[index*FLUX_DIRECTION_COUNT+bin]=ray;
#endif
#endif
}
