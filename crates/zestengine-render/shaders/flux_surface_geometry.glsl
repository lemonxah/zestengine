// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#ifndef FLUX_SURFACE_GEOMETRY
#define FLUX_SURFACE_GEOMETRY
// Pack the geometric normal into the unused mantissa of position.w. Values
// remain finite in [1.5,2), retaining the existing w>0.5 coverage convention.
// Shading/normal-map normals stay in the normal atlas and drive lighting.
float surface_encode_geometry(vec3 n) {
    n/=abs(n.x)+abs(n.y)+abs(n.z);
    vec2 e=n.xy;
    if(n.z<0.0) e=(1.0-abs(e.yx))*mix(vec2(-1),vec2(1),greaterThanEqual(e,vec2(0)));
    uvec2 q=uvec2(round(clamp(e*0.5+0.5,0.0,1.0)*2046.0));
    return uintBitsToFloat(0x3fc00000u|q.x|(q.y<<11u));
}
vec3 surface_geometry_normal(float value,vec3 fallback) {
    if(value<1.5) return fallback; // old/synthetic captures with validity only
    uint bits=floatBitsToUint(value);
    vec2 e=vec2(bits&2047u,(bits>>11u)&2047u)*(2.0/2046.0)-1.0;
    vec3 n=vec3(e,1.0-abs(e.x)-abs(e.y));
    float t=max(-n.z,0.0);
    n.xy+=mix(vec2(t),vec2(-t),greaterThanEqual(n.xy,vec2(0)));
    return normalize(n);
}
#endif
