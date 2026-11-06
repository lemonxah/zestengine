// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Uniform area sampling. Returns irradiance / sample PDF before visibility;
// the caller traces the segment to the actual sampled emitter position.
layout(constant_id=2) const bool FLUX_HAS_AREA_LIGHTS=true;
layout(set=0,binding=FLUX_AREA_BINDING) uniform sampler2D flux_area_atlas;
vec3 flux_area_sample(vec4 position_kind,vec4 right_range,vec4 color_sides,vec4 texture_slot,
                     vec3 up,vec3 p,vec3 n,vec2 xi,out vec3 direction,out float distance) {
    direction=n; distance=0.0;
    vec3 right=right_range.xyz;
    vec3 cross_axes=cross(right,up);
    float jacobian=length(cross_axes);
    if(jacobian<1e-8) return vec3(0.0);
    bool disc=position_kind.w>4.5;
    vec2 q=xi*2.0-1.0;
    if(disc) {float angle=6.28318530718*xi.y;q=sqrt(xi.x)*vec2(cos(angle),sin(angle));}
    vec3 delta=position_kind.xyz+right*q.x+up*q.y-p;
    distance=length(delta);
    if(distance<1e-4) return vec3(0.0);
    direction=delta/distance;
    float source_cos=dot(cross_axes/jacobian,-direction);
    source_cos=color_sides.w>0.5 ? abs(source_cos) : max(source_cos,0.0);
    float fade=max(1.0-length(position_kind.xyz-p)/max(right_range.w,1e-3),0.0);
    float weight=max(dot(n,direction),0.0)*source_cos*(disc ? 3.14159265359 : 4.0)*jacobian/(distance*distance)*fade*fade;
    if(weight<=0.0) return vec3(0.0);
    vec3 tint=vec3(1.0);
    int tile=int(texture_slot.x+0.5)-1;
    if(tile>=0) {
        vec2 uv=clamp(q*0.5+0.5,vec2(0.5/256.0),vec2(1.0-0.5/256.0));
        tint=textureLod(flux_area_atlas,(vec2(tile&3,tile>>2)+uv)*vec2(0.25,0.5),0.0).rgb;
    }
    return color_sides.rgb*tint*weight;
}
