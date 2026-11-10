// Validate the surface and virtual-hit hypotheses independently. A lerp of
// their positions describes neither surface and can discard useful history.
struct ReflectionHistory { vec3 color; vec2 moments; float frames; bool valid; };
ReflectionHistory reflection_history_at(vec3 reprojected,vec3 receiver,vec3 n,
    vec2 rr,float hit_distance,float view_distance,ivec2 size) {
    ReflectionHistory history;history.color=vec3(0);history.moments=vec2(0);history.frames=0.0;history.valid=false;
    vec4 clip=pc.previous_view_projection*vec4(reprojected,1);
    if(clip.w<=0.0 || any(isnan(clip)) || any(isinf(clip))) return history;
    vec3 projected=clip.xyz/clip.w;
    vec2 uv=vec2(projected.x*0.5+0.5,0.5-projected.y*0.5);
    if(projected.z<0.0 || projected.z>=1.0 || any(lessThan(uv,vec2(0))) || any(greaterThanEqual(uv,vec2(1)))) return history;
    vec2 pixel=uv*vec2(size)-0.5,fraction=fract(pixel);
    ivec2 first=ivec2(floor(pixel));float total=0.0;
    for(int corner=0;corner<4;++corner) {
        ivec2 offset=ivec2(corner&1,corner>>1),q=first+offset;
        if(any(lessThan(q,ivec2(0))) || any(greaterThanEqual(q,size))) continue;
        vec4 meta=texelFetch(previous_geometry,q,0);
        if(any(isnan(meta)) || any(isinf(meta))) continue;
        uint packed=uint(round(meta.w)),count=packed>>12;
        if(count==0u || meta.x>=0.9999) continue;
        vec2 old_rr=vec2(float((packed>>6)&63u),float(packed&63u))/63.0;
        if(abs(old_rr.x-rr.x)>0.04 || abs(old_rr.y-rr.y)>0.04 || dot(n,oct_decode(meta.yz))<0.97) continue;
        vec3 old_world=position((vec2(q)+0.5)/vec2(size),meta.x,pc.previous_inverse_view_projection);
        if(abs(dot(old_world-receiver,n))>max(0.005,view_distance*pc.options.w)) continue;
        vec4 color=texelFetch(previous_color,q,0);
        if(any(isnan(color)) || any(isinf(color))) continue;
        if(abs(color.a-hit_distance)>max(0.1,0.3*max(color.a,hit_distance))) continue;
        vec2 weights=mix(vec2(1)-fraction,fraction,vec2(offset));float w=weights.x*weights.y;
        vec2 moments=texelFetch(previous_moments,q,0).rg;
        if(any(isnan(moments)) || any(isinf(moments))) continue;
        history.moments+=moments*w;
        history.color+=color.rgb*w;history.frames+=float(count)*w;total+=w;
    }
    history.valid=total>0.25;
    if(history.valid) {history.color/=total;history.moments/=total;history.frames/=total;}
    return history;
}
vec3 reflection_temporal(vec3 current,vec3 mean,vec3 second,vec3 lower,vec3 upper,
    vec3 world,vec3 previous_world,vec3 n,vec2 rr,float hit_distance,ivec2 size,out float frames,inout vec2 moments) {
    frames=1.0;
    float distance=length(world-pc.camera.xyz);
    ReflectionHistory surface=reflection_history_at(previous_world,previous_world,n,rr,hit_distance,distance,size);
    ReflectionHistory hit;hit.color=vec3(0);hit.moments=vec2(0);hit.frames=0.0;hit.valid=false;
    // Dominant GGX direction approximation, also used by Lumen to limit the
    // virtual-hit hypothesis to sufficiently directional reflection lobes.
    float smoothness=clamp(1.0-rr.x,0.0,1.0);
    float dominant=smoothness*(sqrt(smoothness)+rr.x);
    if(dominant>0.5 && distance>1e-6) {
        vec3 virtual_position=previous_world-(pc.camera.xyz-world)/distance*hit_distance;
        hit=reflection_history_at(virtual_position,previous_world,n,rr,hit_distance,distance,size);
    }
    if(!surface.valid && !hit.valid) return current;
    vec3 surface_color=to_ycocg(surface.color),hit_color=to_ycocg(hit.color);
    float surface_weight=hit.valid ? 0.0 : 1.0;
    if(surface.valid && hit.valid && length(mean-hit_color)>length(mean-surface_color)+0.01) surface_weight=0.8;
    vec3 extent=2.0*sqrt(max(second-mean*mean,vec3(0)));
    vec3 lo=max(lower,mean-extent),hi=min(upper,mean+extent);
    vec3 old=mix(hit_color,surface_color,surface_weight);
    vec3 clipped=mix(clamp(hit_color,lo,hi),clamp(surface_color,lo,hi),surface_weight);
    // Let changed lighting replace unreliable history quickly, retaining a
    // small amount of blending to avoid an abrupt on/off transition.
    float mismatch=length(abs(clipped-old)/max(extent,vec3(0.1)));
    float confidence=0.25+0.75*clamp(1.0-mismatch,0.0,1.0);
    float old_frames=surface_weight>0.5 ? surface.frames : hit.frames;
    float max_frames=mix(2.0,pc.options.z,smoothstep(0.08,0.5,rr.x));
    frames=min(old_frames*confidence+1.0,max_frames);
    vec2 old_moments=mix(hit.moments,surface.moments,surface_weight);
    // Recenter history moments when lighting is clipped. Carry only the
    // confidence-weighted variance so stale light cannot keep the filter wide.
    float old_variance=max(old_moments.y-old_moments.x*old_moments.x,0.0)*confidence;
    moments=mix(vec2(clipped.x,clipped.x*clipped.x+old_variance),moments,1.0/frames);
    return mix(to_rgb(clipped),current,1.0/frames);
}
