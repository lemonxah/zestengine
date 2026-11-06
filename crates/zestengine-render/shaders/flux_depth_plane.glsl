// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Local device-depth plane from the closest valid neighbour on each axis.
// Never interpolate across a sky edge or a large foreground/background jump.
vec3 flux_depth_plane(sampler2D source,ivec2 p,vec4 projection_depth) {
    ivec2 size=textureSize(source,0);
    float d=texelFetch(source,p,0).r;
    float view_z=(projection_depth.x*d+projection_depth.y)/(projection_depth.z*d+projection_depth.w);
    vec2 slope=vec2(0);
    for(int axis=0;axis<2;++axis) {
        ivec2 step=axis==0 ? ivec2(1,0) : ivec2(0,1);
        float best=2.0;
        for(int sign=-1;sign<=1;sign+=2) {
            ivec2 q=p+step*sign;
            if(any(lessThan(q,ivec2(0))) || any(greaterThanEqual(q,size))) continue;
            float neighbour=texelFetch(source,q,0).r;
            if(neighbour>=1.0) continue;
            float neighbour_z=(projection_depth.x*neighbour+projection_depth.y)/(projection_depth.z*neighbour+projection_depth.w);
            if(abs(neighbour_z-view_z)>max(abs(view_z)*0.1,0.01)) continue;
            float delta=(neighbour-d)*float(sign);
            if(abs(delta)<abs(best)) best=delta;
        }
        if(abs(best)<1.0) slope[axis]=best;
    }
    return vec3(d,slope);
}
