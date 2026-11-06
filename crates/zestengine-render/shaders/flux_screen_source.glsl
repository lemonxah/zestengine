// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Physical diffuse and emission are independent of display exposure, GI gain
// and the camera's specular response. Six half floats plus full precision depth
// fit in one 128-bit integer attachment. The depth sign tags legacy/custom HDR.
#ifndef FLUX_SCREEN_SOURCE
#define FLUX_SCREEN_SOURCE
uvec4 flux_pack_source(vec3 diffuse,vec3 emission,float depth,bool physical) {
    if(any(isnan(diffuse)) || any(isinf(diffuse)) || any(isnan(emission)) || any(isinf(emission)))
        return uvec4(0,0,0,floatBitsToUint(1.0));
    diffuse=clamp(diffuse,vec3(0),vec3(65000));
    emission=clamp(emission,vec3(0),vec3(65000));
    return uvec4(packHalf2x16(diffuse.rg),packHalf2x16(vec2(diffuse.b,emission.r)),
        packHalf2x16(emission.gb),floatBitsToUint(depth)|(physical ? 0u : 0x80000000u));
}
float flux_source_depth(uvec4 value) {return uintBitsToFloat(value.w&0x7fffffffu);}
bool flux_source_physical(uvec4 value) {return (value.w&0x80000000u)==0u;}
vec3 flux_source_diffuse(uvec4 value) {return vec3(unpackHalf2x16(value.x),unpackHalf2x16(value.y).x);}
vec3 flux_source_emission(uvec4 value) {return vec3(unpackHalf2x16(value.y).y,unpackHalf2x16(value.z));}
#endif
