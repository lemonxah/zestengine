// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Perspective-correct traversal of a current-view min/max depth hierarchy.
// A miss (including an exhausted budget) belongs to the world tracer/probe.
// No colour/history is read here: diffuse and reflection consumers validate
// their own hit materials and lighting before accepting screen-space radiance.
#include "flux_depth_plane.glsl"
layout(set=0,binding=FLUX_DEPTH_BINDING,std430) readonly buffer FluxDepth {
    uvec4 levels[32]; vec2 depths[];
} flux_hz;
struct FluxScreenHit { vec2 uv; float distance; uint iterations; bool valid; };

bool flux_clip_plane(float a,float b,inout float lo,inout float hi) {
    if(abs(b)<1e-10) return a>=0.0;
    float t=-a/b;
    if(b>0.0) lo=max(lo,t); else hi=min(hi,t);
    return lo<=hi;
}
FluxScreenHit flux_trace_screen(sampler2D source_depth,vec3 origin,vec3 direction,float max_distance,
    mat4 projection,mat4 inverse_projection,bool flip_y,uint budget) {
    FluxScreenHit result=FluxScreenHit(vec2(0),max_distance,0u,false);
    vec4 a=projection*vec4(origin,1),b=projection*vec4(direction,0);
    float lo=0.0,hi=max_distance;
    // Clip in homogeneous coordinates, before division. Handles near-plane
    // crossings, rays towards the camera, orthographic and asymmetric views.
    if(!flux_clip_plane(a.w-1e-5,b.w,lo,hi)
        || !flux_clip_plane(a.z,b.z,lo,hi)
        || !flux_clip_plane(a.w-a.z,b.w-b.z,lo,hi)
        || !flux_clip_plane(a.w+a.x,b.w+b.x,lo,hi)
        || !flux_clip_plane(a.w-a.x,b.w-b.x,lo,hi)
        || !flux_clip_plane(a.w+a.y,b.w+b.y,lo,hi)
        || !flux_clip_plane(a.w-a.y,b.w-b.y,lo,hi) || hi<=lo) return result;
    vec4 c0=a+b*lo,c1=a+b*hi;
    vec3 start=c0.xyz/c0.w,end=c1.xyz/c1.w;
    start.xy=start.xy*.5+.5;end.xy=end.xy*.5+.5;
    if(flip_y) {start.y=1.0-start.y;end.y=1.0-end.y;}
    vec3 delta=end-start;
    vec2 full=vec2(flux_hz.levels[0].xy);
    float pixels=max(abs(delta.x)*full.x,abs(delta.y)*full.y);
    // Subpixel rays have no useful screen-space coverage. Trace the world.
    if(pixels<1.0) return result;
    float progress=0.0;
    float epsilon=min(1e-5,0.001/pixels);
    int max_level=int(flux_hz.levels[0].w)-1;
    int level=min(4,max_level);
    for(uint iteration=0u;iteration<budget && progress<1.0;++iteration) {
        result.iterations=iteration+1u;
        vec3 p=start+delta*min(progress+epsilon,1.0);
        uvec4 shape=flux_hz.levels[level];
        float cell_size=exp2(float(level));
        ivec2 cell=clamp(ivec2(floor(p.xy*full/cell_size)),ivec2(0),ivec2(shape.xy)-1);
        // Odd edge cells keep their actual pixel boundaries, not normalized
        // mip texel widths: a 323px image must not stretch its 162px level.
        vec2 edge=(vec2(cell)+step(vec2(0),delta.xy))*cell_size/full;
        vec2 crossing=vec2(2.0);
        if(abs(delta.x)>1e-10) crossing.x=(edge.x-start.x)/delta.x;
        if(abs(delta.y)>1e-10) crossing.y=(edge.y-start.y)/delta.y;
        float exit_t=min(1.0,min(crossing.x,crossing.y));
        exit_t=max(exit_t,progress+epsilon);
        float z0=start.z+delta.z*progress,z1=start.z+delta.z*exit_t;
        vec3 plane=vec3(0);
        vec2 range;
        if(level==0) {
            plane=flux_depth_plane(source_depth,cell,vec4(inverse_projection[2][2],inverse_projection[3][2],inverse_projection[2][3],inverse_projection[3][3]));
            float padding=0.5*(abs(plane.y)+abs(plane.z));
            range=clamp(vec2(plane.x-padding,plane.x+padding),0.0,1.0);
        } else range=flux_hz.depths[shape.z+uint(cell.y)*shape.x+uint(cell.x)];
        bool overlap=max(z0,z1)>=range.x && min(z0,z1)<=range.y && range.x<1.0;
        if(overlap && level>0) {
            // The ray cannot intersect before it reaches this cell's depth
            // interval. Skip that empty segment before descending; otherwise
            // grazing/sloped rays waste the budget descending at every XY edge.
            if(delta.z>0.0 && z0<range.x) progress=max(progress,(range.x-start.z)/delta.z);
            if(delta.z<0.0 && z0>range.y) progress=max(progress,(range.y-start.z)/delta.z);
            --level;continue;
        }
        if(overlap) {
            float scene_start=plane.x+dot(plane.yz,start.xy*full-(vec2(cell)+0.5));
            float denominator=delta.z-dot(plane.yz,delta.xy*full);
            float t=abs(denominator)>1e-10 ? (scene_start-start.z)/denominator : -1.0;
            vec2 uv=start.xy+delta.xy*t;
            vec2 ndc=uv*2.0-1.0;if(flip_y) ndc.y=-ndc.y;
            vec4 world=inverse_projection*vec4(ndc,start.z+delta.z*t,1);
            float distance=dot(world.xyz/world.w-origin,direction);
            // Reject the originating footprint rather than inventing a thick
            // slab that joins foreground silhouettes to background surfaces.
            if(t>=progress-epsilon && t<=exit_t+epsilon && plane.x<1.0
                && distance>0.01 && t*pixels>1.5 && distance<=max_distance) {
                result.uv=uv;result.distance=distance;result.valid=true;return result;
            }
        }
        progress=exit_t+epsilon;
        level=min(level+1,max_level);
    }
    return result;
}
