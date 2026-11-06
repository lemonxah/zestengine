// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Primary world-representation gaps only: the persistent surface-cache radiosity pass must never
// consume camera history. Misses/uncertain hits continue into the world backend.
#ifndef FLUX_SURFACE_LIGHTING
#include "flux_screen_trace.glsl"
layout(set=0,binding=FLUX_HISTORY_BINDING) uniform sampler2D flux_screen_history;
vec3 flux_history_position(ivec2 pixel,float depth) {
    vec2 uv=(vec2(pixel)+0.5)/vec2(textureSize(flux_screen_history,0));
    vec2 ndc=uv*2.0-1.0;if(pc.screen.options.z>0.5) ndc.y=-ndc.y;
    vec4 p=pc.screen.previous_inverse_view_projection*vec4(ndc,depth,1);
    return p.xyz/p.w;
}
bool flux_screen_incoming(vec3 origin,vec3 direction,float max_distance,inout uint rng,
    out float distance,out vec3 radiance) {
    // Combined HDR history cannot separate emitted from reflected radiance.
    // With a non-unit emissive boost, use world material transport instead.
    if(pc.screen.options.x<0.5 || pc.probe_info.y>0.5 || pc.probe_info.w>0.5 || abs(pc.transport.x-1.0)>1e-6) return false;
    vec3 view_origin=(pc.screen.view*vec4(origin,1)).xyz;
    vec3 view_direction=mat3(pc.screen.view)*direction;
    FluxScreenHit hit=flux_trace_screen(u_depth,view_origin,view_direction,max_distance,
        pc.screen.projection,pc.screen.inverse_projection,pc.screen.options.y>0.5,96u);
    if(!hit.valid) return false;
    vec3 p=origin+direction*hit.distance;
    // Analytic emitters already have explicit sampling. Their world hit path
    // separates emitted and reflected light; a combined HDR pixel cannot.
    if(on_emitter(p)) return false;
    ivec2 size=textureSize(u_depth,0);
    uvec2 pixel=uvec2(clamp(ivec2(hit.uv*vec2(size)),ivec2(0),size-1));
    vec3 surface,normal;
    if(!flux_probe_surface(u_depth,pixel,uvec2(size),pc.inv_view_proj,pc.misc.w!=0u,surface,normal)) return false;
    if(dot(normal,pc.cam.xyz-surface)<0.0) normal=-normal;
    if(dot(normal,-direction)<0.05) return false;
    vec4 clip=pc.screen.previous_view_projection*vec4(p,1);
    if(clip.w<=0.0) return false;
    vec3 projected=clip.xyz/clip.w;
    if(projected.z<0.0 || projected.z>=1.0) return false;
    vec2 uv=projected.xy*0.5+0.5;if(pc.screen.options.z>0.5) uv.y=1.0-uv.y;
    ivec2 previous_size=textureSize(flux_screen_history,0);
    ivec2 previous_pixel=ivec2(floor(uv*vec2(previous_size)));
    if(any(lessThan(previous_pixel,ivec2(1))) || any(greaterThanEqual(previous_pixel,previous_size-1))) return false;
    vec4 previous=texelFetch(flux_screen_history,previous_pixel,0);
    if(previous.a>=1.0) return false;
    vec3 previous_p=flux_history_position(previous_pixel,previous.a);
    float tolerance=max(0.01,length(p-pc.screen.previous_camera.xyz)*pc.screen.options.w);
    // Point-to-plane depth rejection avoids accepting another surface behind
    // this one, while allowing ordinary subpixel reprojection on the same plane.
    if(abs(dot(previous_p-p,normal))>tolerance) return false;
    vec3 tangent[2];
    for(int axis=0;axis<2;++axis) {
        ivec2 offset=axis==0 ? ivec2(1,0) : ivec2(0,1);
        vec4 left=texelFetch(flux_screen_history,previous_pixel-offset,0);
        vec4 right=texelFetch(flux_screen_history,previous_pixel+offset,0);
        bool use_left=left.a<1.0 && (right.a>=1.0 || abs(left.a-previous.a)<abs(right.a-previous.a));
        float d=use_left ? left.a : right.a;
        if(d>=1.0) return false;
        vec3 neighbour=flux_history_position(previous_pixel+(use_left ? -offset : offset),d);
        if(abs(dot(neighbour-previous_p,normal))>tolerance) return false;
        tangent[axis]=(neighbour-previous_p)*(use_left ? -1.0 : 1.0);
    }
    vec3 previous_n=cross(tangent[0],tangent[1]);
    if(dot(previous_n,previous_n)<1e-15) return false;
    previous_n=normalize(previous_n);
    if(dot(previous_n,pc.screen.previous_camera.xyz-previous_p)<0.0) previous_n=-previous_n;
    if(dot(previous_n,normal)<0.9) return false;
    vec2 edge=min(min(hit.uv,vec2(1)-hit.uv),min(uv,vec2(1)-uv));
    float confidence=smoothstep(0.01,0.06,min(edge.x,edge.y));
    if(rnd(rng)>confidence) return false;
    // A camera-visible source is not necessarily visible along this GI ray.
    // Single-sided panes and hidden foreground geometry must still occlude it.
    if(flux_screen_blocked(origin,direction,hit.distance)) return false;
    // Also reject colour exposed only by camera back-face culling. Otherwise
    // that invalid screen source can light neighbours behind the pane and bleed
    // through the screen-probe filter onto receivers on its opposite side.
    vec3 camera_ray=p-pc.cam.xyz;float camera_distance=length(camera_ray);
    if(camera_distance>0.01 && flux_screen_blocked(pc.cam.xyz,camera_ray/camera_distance,camera_distance)) return false;
    radiance=previous.rgb;distance=hit.distance;
    return true;
}
#endif
