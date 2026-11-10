// The mesh projects vertices just outside a narrow river to its nearest bank.
// Match Water::shore_sample there, without changing gameplay containment.
void water_vertex_column(GpuWater body,vec2 p,out vec4 column,out float edge){
    if(water_column(body,p,column,edge) || body.coverage.x!=3u)return;
    float best=-1e30;uint base=body.coverage.y,count=body.coverage.z;
    for(uint i=0u;i+1u<count;){
        uvec4 meta=floatBitsToUint(WATER_COVERAGE_DATA[base+i+1u]);
        if(meta.x<=i||meta.x>count)break;
        if(meta.y==0u){i+=2u;continue;}
        if(i+7u<count){
            vec4 a=WATER_COVERAGE_DATA[base+i+2u],b=WATER_COVERAGE_DATA[base+i+5u];
            vec2 delta=b.xz-a.xz;float t=dot(delta,delta)>1e-8?clamp(dot(p-a.xz,delta)/dot(delta,delta),0.0,1.0):0.0;
            float half_width=max(mix(a.w,b.w,t),0.01)*0.5,d=length(p-a.xz-delta*t),bank=half_width-d;
            if(bank>best){best=bank;column.x=mix(a.y,b.y,t);edge=clamp(d/half_width,0.0,1.0);}
        }i=meta.x;
    }
}
