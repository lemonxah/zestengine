// Common periodic ocean and finite recorded-water mip sampling.
vec4 water_grid_mip(uint base,uint side,vec2 uv,uint level,uint stride,uint lane,bool periodic) {
    for(uint i=0u;i<level && side>1u;++i){base+=side*side*stride;side/=2u;}
    if(side==0u || base+side*side*stride>uint(WATER_OCEAN_DATA.length()))return vec4(0.0);
    vec2 q=periodic ? fract(uv)*float(side) : clamp(uv*float(side)-0.5,vec2(0.0),vec2(float(side-1u)));
    uvec2 a=uvec2(floor(q)),b=periodic ? (a+1u)%side : min(a+1u,side-1u);vec2 f=fract(q);
    return mix(mix(WATER_OCEAN_DATA[base+(a.y*side+a.x)*stride+lane],WATER_OCEAN_DATA[base+(a.y*side+b.x)*stride+lane],f.x),
        mix(WATER_OCEAN_DATA[base+(b.y*side+a.x)*stride+lane],WATER_OCEAN_DATA[base+(b.y*side+b.x)*stride+lane],f.x),f.y);
}
vec4 water_grid_field(uint base,uint side,vec2 uv,float footprint,float cell,uint stride,uint lane,bool periodic){
    if(side==0u)return vec4(0.0);
    float lod=clamp(log2(max(footprint,1e-6)/cell),0.0,log2(float(side)));
    uint level=uint(floor(lod));
    return mix(water_grid_mip(base,side,uv,level,stride,lane,periodic),water_grid_mip(base,side,uv,level+1u,stride,lane,periodic),fract(lod));
}
vec4 water_ocean_field(GpuWater body,vec2 xz,float footprint,uint lane){
    if(body.ocean.y==0u)return vec4(0.0);
    float size=uintBitsToFloat(body.ocean.z);
    return water_grid_field(body.ocean.x,body.ocean.y,xz/size,footprint,size/float(body.ocean.y),2u,lane,true);
}
vec4 water_recorded_field(GpuWater body,vec2 xz,float footprint,uint lane){
    if(body.recording.y==0u || body.recording_window.z<=0.0)return vec4(0.0);
    vec2 q=(xz-body.recording_window.xy)/body.recording_window.z;
    vec2 edge=min(q,vec2(float(body.recording.y-1u))-q);
    float fade=smoothstep(0.0,2.0,min(edge.x,edge.y));
    if(fade<=0.0)return vec4(0.0);
    vec2 uv=(q+0.5)/float(body.recording.y);
    return water_grid_field(body.recording.x,body.recording.y,uv,footprint,body.recording_window.z,3u,lane,false)*fade;
}
