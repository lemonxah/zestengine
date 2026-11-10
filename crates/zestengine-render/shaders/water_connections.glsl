// One shared endpoint traversal for surface vertices, pixels, ray hits and
// underwater. Links are resolved acyclically by the engine; the bounded loop
// also makes malformed external data harmless. Body geometry stays local.
#ifndef ZEST_WATER_CONNECTIONS
#define ZEST_WATER_CONNECTIONS
#include "water_coverage.glsl"
int water_connection(GpuWater body,vec2 xz,out float weight,out vec3 target_point) {
    weight=0.0;target_point=vec3(0.0);
    if(body.connections.x==0u && body.connections.y==0u)return -1;
    vec4 column;float edge,along;
    if(!water_column(body,xz,column,edge,along))return -1;
    for(int end=0;end<2;++end){
        uint target=body.connections[end];
        if(target==0u || target>uint(MAX_WATER_BODIES))continue;
        float distance_m=end==0 ? along : body.connection_lengths.z-along;
        float band=max(min(body.connection_lengths[end],body.connection_lengths.z*0.5),0.01);
        float w=1.0-smoothstep(0.0,band,distance_m);
        if(w<=0.0)continue;
        target_point=(WATER_BODY_DATA[target-1u].to_local*inverse(body.to_local)*vec4(xz.x,column.x,xz.y,1.0)).xyz;
        weight=w;return int(target)-1;
    }
    return -1;
}
float water_shore_field(GpuWater body,vec2 xz){
    uint n=body.shore_field.y;if(n==0u)return 1.0;
    float half_extent=uintBitsToFloat(body.shore_field.z),depth=uintBitsToFloat(body.shore_field.w);
    if(half_extent<=0.0||depth<=0.0)return 1.0;
    if(any(greaterThan(abs(xz),vec2(half_extent))))return 1.0;vec2 uv=xz/(2.0*half_extent)+0.5;
    vec2 p=clamp(uv*float(n)-0.5,vec2(0.0),vec2(float(n-1u))),f=fract(p);uvec2 lo=uvec2(floor(p)),hi=min(lo+1u,uvec2(n-1u));
    uvec4 indices=uvec4(lo.y*n+lo.x,lo.y*n+hi.x,hi.y*n+lo.x,hi.y*n+hi.x);vec4 values;
    for(int i=0;i<4;++i)values[i]=WATER_COVERAGE_DATA[body.shore_field.x+indices[i]/4u][indices[i]%4u];
    return smoothstep(0.0,depth,mix(mix(values.x,values.y,f.x),mix(values.z,values.w,f.x),f.y));
}
float water_connected_shore(GpuWater root,vec2 xz,float root_gain){
    float result=root_gain,previous=root_gain,remaining=1.0,weight;vec3 p;int target=water_connection(root,xz,weight,p);
    for(int step=0;step<MAX_WATER_BODIES&&target>=0;++step){float gain=water_shore_field(WATER_BODY_DATA[target],p.xz);
        remaining*=weight;result+=remaining*(gain-previous);previous=gain;target=water_connection(WATER_BODY_DATA[target],p.xz,weight,p);}
    return clamp(result,0.0,1.0);
}
void water_connected_waves(GpuWater root,vec2 xz,float t,float foot,out vec3 d,out vec3 normal,out float crest,out float lost){
    // Common unconnected bodies keep their original single evaluation.
    if(root.connections.x==0u && root.connections.y==0u){water_evaluate_waves(root,xz,t,foot,d,normal,crest,lost);return;}
    water_evaluate_waves(root,xz,t,foot,d,normal,crest,lost);
    vec3 previous_d=d,previous_n=normal;float previous_c=crest,previous_l=lost;
    mat4 root_local=root.to_local;float weight;vec3 p;int target=water_connection(root,xz,weight,p);float remaining=1.0;
    // Keep only vectors live across iterations. Copying whole 32-wave bodies
    // into loop-carried locals creates excessive per-invocation scratch space.
    for(int step=0;step<MAX_WATER_BODIES && target>=0;++step){
        remaining*=weight;vec3 sd,sn;float sc,sl;
        water_evaluate_waves(WATER_BODY_DATA[target],p.xz,t,foot,sd,sn,sc,sl);
        mat3 to_root=mat3(root_local*inverse(WATER_BODY_DATA[target].to_local));
        sd=to_root*sd;sn=normalize(transpose(inverse(to_root))*sn);
        d+=remaining*(sd-previous_d);normal+=remaining*(sn-previous_n);crest+=remaining*(sc-previous_c);lost+=remaining*(sl-previous_l);
        previous_d=sd;previous_n=sn;previous_c=sc;previous_l=sl;
        target=water_connection(WATER_BODY_DATA[target],p.xz,weight,p);
    }
    normal=normalize(normal);
}
GpuWater water_connected_material(GpuWater root,vec2 xz){
    if(root.connections.x==0u && root.connections.y==0u)return root;
    GpuWater result=root;float remaining=1.0;
    vec4 shallow=root.shallow,deep=root.deep,look0=root.look0;vec2 look1=root.look1.xy;float foam=root.ripple.w;
    vec4 absorption=vec4(root.wave_b[0].w,root.wave_b[1].w,root.wave_b[2].w,root.wave_b[3].w);
    vec4 scattering=vec4(root.wave_b[4].w,root.wave_b[5].w,root.wave_b[6].w,root.wave_b[7].w);
    float weight;vec3 p;int target=water_connection(root,xz,weight,p);
    for(int step=0;step<MAX_WATER_BODIES && target>=0;++step){
        remaining*=weight;
        vec4 next_shallow=WATER_BODY_DATA[target].shallow,next_deep=WATER_BODY_DATA[target].deep,next_look0=WATER_BODY_DATA[target].look0;
        vec2 next_look1=WATER_BODY_DATA[target].look1.xy;float next_foam=WATER_BODY_DATA[target].ripple.w;
        result.shallow+=remaining*(next_shallow-shallow);result.deep+=remaining*(next_deep-deep);
        result.look0+=remaining*(next_look0-look0);result.look1.xy+=remaining*(next_look1-look1);result.ripple.w+=remaining*(next_foam-foam);
        for(int i=0;i<4;++i){float a=WATER_BODY_DATA[target].wave_b[i].w,s=WATER_BODY_DATA[target].wave_b[i+4].w;
            result.wave_b[i].w+=remaining*(a-absorption[i]);result.wave_b[i+4].w+=remaining*(s-scattering[i]);absorption[i]=a;scattering[i]=s;}
        shallow=next_shallow;deep=next_deep;look0=next_look0;look1=next_look1;foam=next_foam;
        target=water_connection(WATER_BODY_DATA[target],p.xz,weight,p);
    }
    return result;
}
float water_connected_edge(GpuWater body,vec2 xz,float edge){
    if(body.connections.x==0u && body.connections.y==0u)return edge;
    float result=edge,remaining=1.0;float weight;vec3 p;int target=water_connection(body,xz,weight,p);
    for(int step=0;step<MAX_WATER_BODIES && target>=0;++step){
        vec4 column;float next_edge;if(!water_column(WATER_BODY_DATA[target],p.xz,column,next_edge))break;
        remaining*=weight;result+=remaining*(next_edge-edge);edge=next_edge;
        target=water_connection(WATER_BODY_DATA[target],p.xz,weight,p);
    }
    return clamp(result,0.0,1.0);
}
#endif
