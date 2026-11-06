// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Automatic, world-aligned directional radiance cache. The first local segment
// is always traced; cached probes represent lighting outside a cell-sized sphere.
// Specialize gather pipelines so disabled/reference code does not burden
// normal ray tracing with extra branches and register lifetimes.
layout(constant_id=0) const bool RC_FULL_INTERPOLATION=false;
layout(constant_id=1) const bool RC_ENABLED=true;
const int RC_GRID=16;
const int RC_LEVELS=3;
const uint RC_PROBES=12288u;
const uint RC_QUEUE_START=48u;
const uint RC_MAX_UPDATES=512u; // 128 full probes or 512 quarter-cost probes
const uint RC_REQUEST_START=RC_QUEUE_START+RC_MAX_UPDATES;
struct RadianceProbe {
    ivec4 cell;
    uvec4 state; // epoch, last traced frame, last requested frame, stored angular resolution (0 = invalid)
    uvec2 rays[256]; // half RGB and distance; negative distance = unsupported hit
};
layout(set=0,binding=RC_BINDING,std430) buffer RadianceData {RadianceProbe rc_probes[];};
layout(set=0,binding=RC_BINDING+1,std430) buffer RadianceWork {uint rc_work[];};
layout(set=0,binding=RC_BINDING+2,std140) uniform RadianceParams {
    ivec4 rc_origin[3];
    vec4 rc_spacing; // xyz cell sizes, w trace sphere radius / cell size
    uvec4 rc_config; // serial, lighting/scene epoch, update budget, flags: enabled1 / diagnostic2 / adaptive resolution4 / clipmap blending8
    vec4 rc_camera; // xyz world position, w distant-probe threshold in metres
};
bool flux_world_source_valid=true;
uint rc_requested_levels=0u;
uint rc_index(ivec3 cell,int level) {
    uvec3 p=uvec3(cell)&uvec3(15u);
    return uint(level)*4096u+p.x+16u*p.y+256u*p.z;
}
vec3 rc_position(ivec3 cell,int level) {return (vec3(cell)+0.5)*rc_spacing[level];}
// Keep resolution stable near the distance boundary. Existing radiance remains
// readable until a budgeted update replaces all texels and publishes its size.
uint rc_trace_resolution(uint index) {
    if((rc_config.w&4u)==0u) return 16u;
    ivec4 cell=rc_probes[index].cell;
    float threshold=rc_camera.w;
    uint previous=rc_probes[index].state.w;
    if(previous==8u) threshold*=0.9;
    else if(previous==16u) threshold*=1.1;
    vec3 delta=rc_position(cell.xyz,cell.w)-rc_camera.xyz;
    return dot(delta,delta)>threshold*threshold ? 8u : 16u;
}
uint rc_trace_cost(uint index) {
    uint resolution=rc_trace_resolution(index);
    return resolution*resolution/64u;
}
// Choose one level per ray before tracing its local segment. Stochastic
// selection blends the overlap without doubling directional cache lookups.
// The selected level must also determine occlusion distance and requests:
// sampling a coarse level after a fine local trace would skip nearby blockers.
int rc_level(vec3 p,float random_sample) {
    if(!RC_ENABLED) return -1;
    float target=clamp(random_sample,0.0,0.99999994);
    for(int level=0;level<RC_LEVELS;++level) {
        vec3 local=p/rc_spacing[level]-0.5-vec3(rc_origin[level].xyz);
        if(any(lessThan(local,vec3(0))) || any(greaterThanEqual(local,vec3(15)))) continue;
        if((rc_config.w&8u)!=0u) {
            vec3 edge=min(local,vec3(15)-local);
            float weight=clamp(min(edge.x,min(edge.y,edge.z))*0.5,0.0,1.0);
            if(target>=weight) continue;
        }
        return level;
    }
    // The outermost grid fades into full tracing as well.
    return -1;
}
float rc_local_distance(int level,float maximum) {
    return level<0 ? maximum : min(maximum,rc_spacing[level]*(rc_spacing.w+sqrt(3.0)));
}
void rc_request(vec3 p,int level) {
    if(!RC_ENABLED || level<0) return;
    uint bit=1u<<uint(level);
    if((rc_requested_levels&bit)!=0u) return;
    rc_requested_levels|=bit;
    ivec3 cell=ivec3(floor(p/rc_spacing[level]-0.5));
    // Rays from one receiver can choose different overlapping levels. Publish
    // each level once per invocation, then coalesce nearby receivers per cell.
    if(atomicExchange(rc_work[RC_REQUEST_START+rc_index(cell,level)],1u)!=0u) return;
    for(int c=0;c<8;++c) {
        ivec3 corner=cell+ivec3(c&1,(c>>1)&1,(c>>2)&1);
        atomicMax(rc_probes[rc_index(corner,level)].state.z,rc_config.x);
    }
    if((rc_config.w&2u)!=0u) atomicAdd(rc_work[5],1u);
}
vec3 rc_decode_direction(vec2 uv) {
    vec2 e=uv*2.0-1.0;vec3 n=vec3(e,1.0-abs(e.x)-abs(e.y));
    if(n.z<0.0) n.xy=(1.0-abs(n.yx))*mix(vec2(-1),vec2(1),greaterThanEqual(n.xy,vec2(0)));
    return normalize(n);
}
vec2 rc_encode_direction(vec3 n) {
    n/=abs(n.x)+abs(n.y)+abs(n.z);
    if(n.z<0.0) n.xy=(1.0-abs(n.yx))*mix(vec2(-1),vec2(1),greaterThanEqual(n.xy,vec2(0)));
    return n.xy*0.5+0.5;
}
ivec2 rc_wrap(ivec2 p,int resolution) {
    if(p.x<0) {p.x=-p.x-1;p.y=resolution-1-p.y;}
    if(p.x>=resolution) {p.x=2*resolution-1-p.x;p.y=resolution-1-p.y;}
    if(p.y<0) {p.y=-p.y-1;p.x=resolution-1-p.x;}
    if(p.y>=resolution) {p.y=2*resolution-1-p.y;p.x=resolution-1-p.x;}
    return clamp(p,ivec2(0),ivec2(resolution-1));
}
bool rc_direction(uint index,vec3 direction,float maximum_intensity,out vec4 value) {
    int resolution=int(rc_probes[index].state.w);
    vec2 p=rc_encode_direction(direction)*float(resolution)-0.5;ivec2 lo=ivec2(floor(p));vec2 f=fract(p);
    value=vec4(0);
    for(int y=0;y<2;++y) for(int x=0;x<2;++x) {
        ivec2 texel=rc_wrap(lo+ivec2(x,y),resolution);uvec2 packed=rc_probes[index].rays[texel.x+texel.y*resolution];
        vec4 v=vec4(unpackHalf2x16(packed.x),unpackHalf2x16(packed.y));
        if(v.w<0.0) return false;
        // Match the gather limiter before interpolation, as for individual
        // traced rays. Storage stays physical; limiting the blended result
        // instead would brighten strong sources as cache coverage warms up.
        v.rgb*=min(1.0,maximum_intensity/max(max(v.r,max(v.g,v.b)),1e-6));
        value+=v*(x==0 ? 1.0-f.x : f.x)*(y==0 ? 1.0-f.y : f.y);
    }
    return true;
}
bool rc_probe(ivec3 cell,int level,vec3 origin,vec3 direction,float maximum,float maximum_intensity,out vec4 value) {
    uint index=rc_index(cell,level);
    float radius=rc_spacing[level]*rc_spacing.w*1.5;
    vec3 delta=origin-rc_position(cell,level);float b=dot(delta,direction);
    float t=-b+sqrt(max(b*b-dot(delta,delta)+radius*radius,0.0));
    vec3 projected=normalize(delta+direction*t);vec4 sample_value;
    if(!rc_direction(index,projected,maximum_intensity,sample_value)) return false;
    float correction=sample_value.w>=maximum*0.99 ? 1.0 : clamp(t*t/max(radius*radius*dot(projected,direction),1e-4),0.25,4.0);
    value=vec4(sample_value.rgb*correction,clamp(sample_value.w-b,0.0,maximum));
    return true;
}
bool rc_sample(int level,vec3 origin,vec3 direction,float maximum,float maximum_intensity,float random_sample,out vec3 radiance,out float distance) {
    radiance=vec3(0);distance=0.0;
    if(!RC_ENABLED || level<0) return false;
    ivec3 cell=ivec3(floor(origin/rc_spacing[level]-0.5));
    vec3 f=clamp(origin/rc_spacing[level]-0.5-vec3(cell),0.0,1.0);
    // Require complete residency before choosing samples. A missing or dirty
    // corner always falls back to tracing; it must not behave like a dark probe.
    for(int c=0;c<8;++c) {
        ivec3 corner=cell+ivec3(c&1,(c>>1)&1,(c>>2)&1);uint index=rc_index(corner,level);
        if(any(notEqual(rc_probes[index].cell,ivec4(corner,level))) || rc_probes[index].state.x!=rc_config.y || rc_probes[index].state.w==0u) return false;
    }
    const bool full=RC_FULL_INTERPOLATION;
    // Lumen's stochastic trilinear interpolation: choose corners according to
    // their trilinear weight, then average equally. Two stratified samples
    // reduce variance while requiring only8 directional texel reads, vs32 for
    // the full reference. Do not consume the path RNG: cache misses must retain
    // the exact same subsequent world-tracing sample stream.
    for(int sample_index=0;sample_index<(full ? 8 : 2);++sample_index) {
        int corner_index=sample_index;float weight=0.5;
        if(!full) {
            // Clamp after addition too: adding1 can round the largest float
            // below1 up to2. Never select a zero-weight corner at that endpoint.
            float target=min((clamp(random_sample,0.0,0.99999994)+float(sample_index))*0.5,0.99999994);
            float cumulative=0.0;corner_index=0;
            for(int c=0;c<8;++c) {
                ivec3 off=ivec3(c&1,(c>>1)&1,(c>>2)&1);
                vec3 weights=mix(vec3(1)-f,f,vec3(off));float weight=weights.x*weights.y*weights.z;
                if(weight<=0.0) continue;
                corner_index=c;cumulative+=weight;
                if(target<cumulative) break;
            }
        }
        ivec3 off=ivec3(corner_index&1,(corner_index>>1)&1,(corner_index>>2)&1);
        if(full) {vec3 weights=mix(vec3(1)-f,f,vec3(off));weight=weights.x*weights.y*weights.z;}
        vec4 value;if(!rc_probe(cell+off,level,origin,direction,maximum,maximum_intensity,value)) return false;
        radiance+=value.rgb*weight;distance+=value.w*weight;
    }
    if((rc_config.w&2u)!=0u) atomicAdd(rc_work[3],1u);
    return true;
}
