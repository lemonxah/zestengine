// Shared reflection reconstruction interfaces and geometry transforms.
layout(local_size_x=8,local_size_y=8) in;
layout(set=0,binding=0) uniform sampler2D raw_reflection;
layout(set=0,binding=1) uniform sampler2D scene_color;
layout(set=0,binding=2) uniform sampler2D scene_depth;
layout(set=0,binding=3) uniform sampler2D scene_gbuffer;
layout(set=0,binding=4) uniform sampler2D previous_color;
layout(set=0,binding=5) uniform sampler2D previous_geometry;
layout(set=0,binding=6,rgba16f) uniform writeonly image2D result;
layout(set=0,binding=7,rgba16f) uniform image2D history_color;
layout(set=0,binding=8,rgba32f) uniform image2D history_geometry;
layout(set=0,binding=11) uniform sampler2D previous_moments;
layout(set=0,binding=12,rgba32f) uniform image2D history_moments;
layout(set=0,binding=10) uniform sampler2D receiver_motion;
layout(set=0,binding=9) uniform Params {
    mat4 inverse_view_projection;
    mat4 inverse_view;
    mat4 previous_view_projection;
    mat4 previous_inverse_view_projection;
    vec4 camera;
    vec4 fog_color, fog_params, fog_light, fog_sun;
    vec4 options; // history valid, enabled, max history frames, relative plane tolerance
    vec4 motion; // current view has object correspondence
} pc;
vec3 oct_decode(vec2 e) {
    e=e*2.0-1.0;vec3 n=vec3(e,1.0-abs(e.x)-abs(e.y));
    float t=max(-n.z,0.0);n.xy+=mix(vec2(t),vec2(-t),greaterThanEqual(n.xy,vec2(0)));
    return normalize(n);
}
vec2 oct_encode(vec3 n) {
    n/=abs(n.x)+abs(n.y)+abs(n.z);
    vec2 e=n.z>=0.0 ? n.xy : (1.0-abs(n.yx))*mix(vec2(-1),vec2(1),greaterThanEqual(n.xy,vec2(0)));
    return e*0.5+0.5;
}
vec2 material(vec4 g) {
    uint b=uint(round(g.b*1023.0)),a=uint(round(g.a*3.0));
    return vec2(float(b>>4),float((b&15u)|(a<<4)))/63.0;
}
vec3 normal(ivec2 p) {return normalize(mat3(pc.inverse_view)*oct_decode(texelFetch(scene_gbuffer,p,0).rg));}
vec3 position(vec2 uv,float d,mat4 inverse_vp) {
    vec4 p=inverse_vp*vec4(uv.x*2.0-1.0,1.0-uv.y*2.0,d,1);return p.xyz/p.w;
}
vec3 to_ycocg(vec3 c) {return vec3(dot(c,vec3(0.25,0.5,0.25)),(c.r-c.b)*0.5,(-c.r+2.0*c.g-c.b)*0.25);}
vec3 to_rgb(vec3 c) {return vec3(c.x+c.y-c.z,c.x+c.z,c.x-c.y-c.z);}
#include "reflection_fog.glsl"
vec3 composite(vec3 color,vec2 uv,float depth) {
    if(pc.fog_color.w<=0.0) return color;
    vec3 end_world=position(uv,depth,pc.inverse_view_projection);
    if(depth>=0.9999) {
        vec3 far_world=position(uv,1.0,pc.inverse_view_projection);
        end_world=pc.camera.xyz+normalize(far_world-pc.camera.xyz)*200.0;
    }
    return reflection_fog_apply_fog(color,end_world,pc.camera.xyz,pc.fog_color,pc.fog_params,pc.fog_light,pc.fog_sun);
}
