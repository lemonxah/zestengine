// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Standard opaque material evaluation at a traced triangle. Included after
// the bindless u_tex declaration by both FluxRT and reflection hit shaders.
#include "material_response.glsl"
vec4 rt_material_tex(int index,vec2 uv,vec4 st,float cone,float density,vec4 fallback) {
    if(index<0) return fallback;
    vec2 size=vec2(textureSize(u_tex[nonuniformEXT(index)],0));
    // Include texture tiling in the ray footprint. A zero cone selects mip 0
    // for GI; reflection cones retain their distance/roughness footprint.
    float footprint=cone*density*max(abs(st.x),abs(st.y))*sqrt(size.x*size.y);
    float lod=max(log2(max(footprint,1e-6)),0.0);
    return textureLod(u_tex[nonuniformEXT(index)],uv*st.xy+st.zw,lod);
}
struct RtSurfaceMaterial {
    vec3 albedo; vec3 emission; vec3 normal;
    float metallic; float roughness;
};
RtSurfaceMaterial rt_surface_material(Instance inst,vec2 uv,vec3 vertex_color,vec3 N,vec4 tangent,float cone,float density) {
    RtSurfaceMaterial result;
    result.albedo=(inst.textures.x>=0 ? inst.base.rgb*rt_material_tex(inst.textures.x,uv,inst.albedo_st,cone,density,vec4(1)).rgb : inst.albedo.rgb)*vertex_color;
    float mask=0.0;
    if(any(greaterThan(inst.detail.xyz,vec3(0))))
        mask=rt_material_tex(inst.detail_textures.w,uv,inst.albedo_st,cone,density,vec4(1)).r;
    if(inst.detail.x>0.0) {
        vec3 d=rt_material_tex(inst.detail_textures.x,uv,inst.detail_st,cone,density,vec4(1)).rgb;
        result.albedo=material_detail_albedo(result.albedo,d,inst.detail.x*mask,inst.detail.w);
    }
    vec3 orm=rt_material_tex(inst.textures.z,uv,inst.orm_st,cone,density,vec4(1)).rgb;
    if(inst.detail.z>0.0) orm*=mix(vec3(1),rt_material_tex(inst.detail_textures.z,uv,inst.detail_st,cone,density,vec4(1)).rgb,clamp(inst.detail.z*mask,0.0,1.0));
    float rough=rt_material_tex(inst.extra_textures.x,uv,inst.orm_st,cone,density,vec4(1)).r;
    float metal=rt_material_tex(inst.extra_textures.y,uv,inst.orm_st,cone,density,vec4(1)).r;
    rough=mix(rough,1.0-rough,inst.orm_invert.y);
    metal=mix(metal,1.0-metal,inst.orm_invert.z);
    result.roughness=clamp(inst.params.y*orm.g*rough,0.045,1.0);
    result.metallic=clamp(inst.params.x*orm.b*metal,0.0,1.0);
    result.normal=N;
    if(inst.textures.y>=0 || inst.detail.y>0.0) {
        vec3 T=tangent.xyz-N*dot(N,tangent.xyz);
        if(dot(T,T)>1e-8) {
            T=normalize(T);vec3 tn=vec3(0,0,1);
            if(inst.textures.y>=0) {
                tn=material_decode_normal(rt_material_tex(inst.textures.y,uv,inst.normal_st,cone,density,vec4(0.5,0.5,1,1)).xyz);
                tn.xy*=inst.params.z;
            }
            if(inst.detail.y>0.0) {
                vec3 dn=material_decode_normal(rt_material_tex(inst.detail_textures.y,uv,inst.detail_st,cone,density,vec4(128.0/255.0,128.0/255.0,1,1)).xyz);
                tn.xy+=dn.xy*inst.detail.y*mask;
            }
            result.normal=normalize(mat3(T,cross(N,T)*tangent.w,N)*normalize(tn));
        }
    }
    result.albedo=clamp(result.albedo,0.0,1.0);
    // Raw and fallback emission already contain the enabled flag and the
    // frame's pulse. Mask coordinates deliberately exclude emission scrolling.
    result.emission=inst.textures.w>=0 ? inst.emission_raw.rgb*rt_material_tex(inst.textures.w,uv,inst.emission_st,cone,density,vec4(1)).rgb : inst.emission.rgb;
    result.emission*=rt_material_tex(inst.extra_textures.z,uv,inst.emission_mask_st,cone,density,vec4(1)).r;
    return result;
}
