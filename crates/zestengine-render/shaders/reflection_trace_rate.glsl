// Adaptive half-resolution tracing keeps full-resolution receivers at edges
// and on sharp reflections. All lanes independently agree on a 2x2 owner.
vec3 reflection_rate_normal(vec2 e) {
    e=e*2.0-1.0;vec3 n=vec3(e,1.0-abs(e.x)-abs(e.y));
    float t=max(-n.z,0.0);n.xy+=mix(vec2(t),vec2(-t),greaterThanEqual(n.xy,vec2(0)));
    return normalize(n);
}
vec3 reflection_rate_position(ivec2 p,ivec2 size,float depth,mat4 inverse_projection,bool flip_y) {
    vec2 ndc=(vec2(p)+0.5)/vec2(size)*2.0-1.0;
    if(flip_y) ndc.y=-ndc.y;
    vec4 v=inverse_projection*vec4(ndc,depth,1);return v.xyz/v.w;
}
int reflection_trace_owner(ivec2 pixel,ivec2 size,sampler2D depth_image,sampler2D material_image,
    mat4 inverse_projection,bool flip_y,float cutoff,uint factor,uint frame) {
    if(factor<2u) return -1;
    ivec2 first=pixel&~ivec2(1);
    if(any(greaterThanEqual(first+ivec2(1),size))) return -1;
    vec4 base=texelFetch(material_image,first,0);
    uvec2 material=uvec2(round(base.ba*vec2(1023.0,3.0)));
    float roughness=float(material.x>>4)/63.0;
    if(roughness<0.25 || roughness>cutoff || ((material.x&15u)|(material.y<<4))==0u) return -1;
    float depth=texelFetch(depth_image,first,0).r;
    if(depth>=0.9999) return -1;
    vec3 point=reflection_rate_position(first,size,depth,inverse_projection,flip_y);
    vec3 normal=reflection_rate_normal(base.rg);
    for(int i=1;i<4;++i) {
        ivec2 p=first+ivec2(i&1,i>>1);
        vec4 g=texelFetch(material_image,p,0);
        if(any(notEqual(uvec2(round(g.ba*vec2(1023.0,3.0))),material))) return -1;
        if(dot(normal,reflection_rate_normal(g.rg))<0.9995) return -1;
        float d=texelFetch(depth_image,p,0).r;
        if(d>=0.9999) return -1;
        vec3 other=reflection_rate_position(p,size,d,inverse_projection,flip_y);
        if(abs(dot(other-point,normal))>max(0.0005,length(point)*0.0002)) return -1;
    }
    return int(frame&3u);
}

// Negative hit distance -1 means no reflection. -2..-5 identify an untraced
// receiver and its current-frame owner; no previous frame or stale value is used.
vec4 reflection_trace_sample(sampler2D samples,ivec2 pixel) {
    vec4 value=texelFetch(samples,pixel,0);
    if(value.a<=-2.0 && value.a>=-5.0) {
        int owner=int(-value.a)-2;
        ivec2 p=(pixel&~ivec2(1))+ivec2(owner&1,owner>>1);
        if(any(greaterThanEqual(p,textureSize(samples,0)))) return vec4(0,0,0,-1);
        value=texelFetch(samples,p,0);
        if(value.a<0.0) return vec4(0,0,0,-1);
    }
    return value;
}
