// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
#include "flux_surface_geometry.glsl"
#include "material_response.glsl"
layout(location=0) in vec3 world;
layout(location=1) in vec3 normal;
layout(location=2) in vec2 uv;
layout(location=3) in vec4 color;
layout(location=4) in vec4 tangent;
layout(location=0) out vec4 o_position;
layout(location=1) out vec4 o_normal;
layout(location=2) out vec4 o_albedo;
layout(location=3) out vec4 o_emission;
layout(push_constant) uniform Push {
    mat4 model; vec4 base; vec4 emission; vec4 material; uvec4 info;
} pc;
struct Card { mat4 view_proj; vec4 normal_texel; uvec4 rect; mat4 capture_to_world; };
layout(set=0, binding=0, std430) readonly buffer Cards { Card cards[]; };
layout(set=1, binding=0) uniform sampler2D t_albedo;
layout(set=1, binding=1) uniform sampler2D t_normal;
layout(set=1, binding=2) uniform sampler2D t_orm;
layout(set=1, binding=3) uniform sampler2D t_emission;
layout(set=1, binding=5) uniform sampler2D t_opacity;
layout(set=1, binding=6) uniform sampler2D t_emission_mask;
layout(set=1, binding=14) uniform sampler2D t_roughness;
layout(set=1, binding=15) uniform sampler2D t_metallic;
layout(set=1, binding=17) uniform sampler2D t_detail_albedo;
layout(set=1, binding=18) uniform sampler2D t_detail_normal;
layout(set=1, binding=19) uniform sampler2D t_detail_orm;
layout(set=1, binding=20) uniform sampler2D t_detail_mask;
// Same prefix as MaterialFx. No frame/camera lighting enters these captures.
layout(set=1, binding=4) uniform MaterialFx {
    vec4 rim; vec4 toon; vec4 scroll; vec4 matcap; vec4 matcap_blend;
    vec4 albedo_st; vec4 normal_st; vec4 orm_st; vec4 emission_st;
    vec4 orm_invert; vec4 parallax; vec4 water; vec4 detail; vec4 detail_st;
} fx;
bool feature(uint mask) { return (pc.info.y & mask) != 0u; }
void main() {
    vec3 N = normalize(normal);
    vec3 geometric=cross(dFdx(world),dFdy(world));
    geometric=dot(geometric,geometric)>1e-16 ? normalize(geometric) : N;
    vec3 face = cards[pc.info.x].normal_texel.xyz;
    // Capture the surface facing this card, excluding edge-on faces. A two-
    // sided material has a separate capture for each side of a thin sheet.
    float facing = dot(N, face);
    if (feature(8u) && facing < 0.0) N = -N;
    else if (feature(16u) && !feature(8u)) N = -N;
    if (dot(N, face) < 0.15) discard;
    if(dot(geometric,N)<0.0) geometric=-geometric;
    float time = uintBitsToFloat(pc.info.z);
    vec2 auv = uv * fx.albedo_st.xy + fx.albedo_st.zw + fx.scroll.xy * time;
    vec2 nuv = uv * fx.normal_st.xy + fx.normal_st.zw;
    vec2 ouv = uv * fx.orm_st.xy + fx.orm_st.zw;
    vec2 euv = uv * fx.emission_st.xy + fx.emission_st.zw;
    vec2 duv = uv * fx.detail_st.xy + fx.detail_st.zw;
    vec4 albedo = texture(t_albedo, auv) * pc.base * color;
    if (feature(4u) && albedo.a * texture(t_opacity, auv).r < pc.material.z) discard;
    float mask = texture(t_detail_mask, auv).r;
    if (fx.detail.x > 0.0) {
        albedo.rgb=material_detail_albedo(albedo.rgb,texture(t_detail_albedo,duv).rgb,fx.detail.x*mask,fx.detail.w);
    }
    if (feature(1u) || fx.detail.y > 0.0) {
        vec3 T = tangent.xyz - N * dot(tangent.xyz, N);
        if (dot(T,T) > 1e-8) {
            T = normalize(T);
            vec3 tn = vec3(0,0,1);
            if (feature(1u)) { tn = material_decode_normal(texture(t_normal,nuv).xyz); tn.xy *= pc.material.w; }
            if (fx.detail.y > 0.0) tn.xy += material_decode_normal(texture(t_detail_normal,duv).xyz).xy * fx.detail.y * mask;
            N = normalize(mat3(T,cross(N,T)*tangent.w,N)*normalize(tn));
        }
    }
    vec3 orm = texture(t_orm,ouv).rgb;
    orm *= mix(vec3(1.0),texture(t_detail_orm,duv).rgb,clamp(fx.detail.z*mask,0.0,1.0));
    float metal = texture(t_metallic,ouv).r;
    metal = mix(metal,1.0-metal,fx.orm_invert.z);
    float rough = texture(t_roughness,ouv).r;
    rough = mix(rough,1.0-rough,fx.orm_invert.y);
    vec3 emission = vec3(0.0);
    if (feature(2u)) {
        float pulse = material_emission_pulse(time,fx.toon.z);
        emission = texture(t_emission,euv+fx.scroll.zw*time).rgb * pc.emission.rgb * pulse * texture(t_emission_mask,euv).r;
    }
    o_position = vec4(world,surface_encode_geometry(geometric));
    o_normal = vec4(N,clamp(pc.material.y*orm.g*rough,0.045,1.0));
    o_albedo = vec4(clamp(albedo.rgb,0.0,1.0),clamp(pc.material.x*orm.b*metal,0.0,1.0));
    o_emission = vec4(max(emission,vec3(0.0)),1.0+float(pc.info.w));
}
