#ifndef ZEST_WATER_MEDIA
#define ZEST_WATER_MEDIA
// Traverse all authored media in ray order. Each interval has one owner;
// overlaps never multiply extinction twice, and dry hulls expose lower water.
// No fixed ray step or fixed array of shoreline intersections is used.
float water_medium_next(GpuWater body,vec3 origin,vec3 direction,vec2 range){
    float next=range.y;uint kind=body.coverage.x;
    if(kind==3u)next=water_river_next(body,origin,direction,range);
    else if(kind==4u)next=water_mesh_next(body,origin,direction,range);
    else{
        if(kind==2u||kind==5u)next=water_next_shore(body,origin.xz,direction.xz,range);
        else for(int axis=0;axis<2;++axis){float extent=body.bounds[axis];int k=axis*2;
            if(extent>0.0&&abs(direction[k])>1e-12){water_ray_root((-extent-origin[k])/direction[k],range,next);water_ray_root((extent-origin[k])/direction[k],range,next);}}
        if(abs(direction.y)>1e-12){water_ray_root((body.bounds.z-origin.y)/direction.y,range,next);
            if(kind!=0u)water_ray_root((body.bounds.z-uintBitsToFloat(body.coverage.w)-origin.y)/direction.y,range,next);}
    }
    uint count=water_exclusion_count(body);
    for(uint i=0u;i<count;++i){vec2 hit;if(water_box_interval(water_exclusion_matrix(body,i),origin,direction,range,hit)){
        water_ray_root(hit.x,range,next);water_ray_root(hit.y,range,next);}}
    return next;
}
bool water_multiple_media(){
    int count=0;for(int i=0;i<MAX_WATER_BODIES;++i)if(WATER_BODY_DATA[i].ownership.y>0.5){if(++count>1)return true;}return false;
}
void water_media_trace(vec3 origin,vec3 direction,vec2 range,bool underwater,out vec3 trans,out vec3 scatter,out float wet){
    trans=vec3(1.0);scatter=vec3(0.0);wet=0.0;float cursor=range.x;
    while(cursor<range.y){
        float next=range.y;
        for(int i=0;i<MAX_WATER_BODIES;++i){if(WATER_BODY_DATA[i].ownership.y<0.5)continue;
            vec3 p=(WATER_BODY_DATA[i].to_local*vec4(origin,1.0)).xyz,d=mat3(WATER_BODY_DATA[i].to_local)*direction;
            next=min(next,water_medium_next(WATER_BODY_DATA[i],p,d,vec2(cursor,next)));
        }
        if(next<=cursor)break;
        vec3 point=origin+direction*(cursor+(next-cursor)*0.5);
        int owner=-1;float rank=-1.0;vec3 local=vec3(0.0);vec4 column=vec4(0.0);
        for(int i=0;i<MAX_WATER_BODIES;++i){if(WATER_BODY_DATA[i].ownership.y<0.5||WATER_BODY_DATA[i].ownership.x<=rank)continue;
            vec3 p=(WATER_BODY_DATA[i].to_local*vec4(point,1.0)).xyz;vec4 c;float edge;
            if(water_column(WATER_BODY_DATA[i],p.xz,c,edge)&&p.y<=c.x&&p.y>=c.x-c.y&&!water_excluded(WATER_BODY_DATA[i],p)){
                owner=i;rank=WATER_BODY_DATA[i].ownership.x;local=p;column=c;
            }
        }
        if(owner>=0){
            GpuWater material=water_connected_material(WATER_BODY_DATA[owner],local.xz);
            vec3 sigma_s=max(vec3(material.wave_b[4].w,material.wave_b[5].w,material.wave_b[6].w),vec3(0.0));
            vec3 extinction=max(vec3(material.wave_b[0].w,material.wave_b[1].w,material.wave_b[2].w),vec3(0.0))+sigma_s;
            vec3 source=sigma_s/max(extinction,vec3(1e-8));
            if(underwater){
                if(material.underwater.w>0.5)extinction=material.underwater.x*vec3(1.55,1.0,0.8);
                if(material.underwater.z<0.5)extinction=vec3(0.0);
                vec3 up=normalize(transpose(mat3(material.to_local))*vec3(0.0,1.0,0.0));
                vec3 start=(material.to_local*vec4(origin,1.0)).xyz;
                float depth=max(column.x-start.y,0.0)/max(length(transpose(mat3(material.to_local))*vec3(0.0,1.0,0.0)),1e-6);
                vec3 light_extinction=extinction;
                if(material.underwater.w>0.5){source=mix(material.shallow.rgb,material.deep.rgb,smoothstep(-0.5,0.6,-dot(direction,up)));light_extinction=material.underwater.y*vec3(1.55,1.0,0.8);}
                source*=exp(-light_extinction*depth);
            }
            vec3 segment=exp(-extinction*(next-cursor));scatter+=trans*(vec3(1.0)-segment)*source;trans*=segment;wet+=next-cursor;
        }
        cursor=next;
    }
}
vec3 water_media_receiver(vec3 point){
    int owner=-1;float rank=-1.0,depth=0.0;
    for(int i=0;i<MAX_WATER_BODIES;++i){if(WATER_BODY_DATA[i].ownership.y<0.5||WATER_BODY_DATA[i].ownership.x<=rank)continue;
        vec3 p=(WATER_BODY_DATA[i].to_local*vec4(point,1.0)).xyz;vec4 c;float edge;
        if(water_column(WATER_BODY_DATA[i],p.xz,c,edge)&&p.y<=c.x&&p.y>=c.x-c.y&&!water_excluded(WATER_BODY_DATA[i],p)){
            owner=i;rank=WATER_BODY_DATA[i].ownership.x;depth=(c.x-p.y)/max(length(transpose(mat3(WATER_BODY_DATA[i].to_local))*vec3(0.0,1.0,0.0)),1e-6);
        }
    }
    return owner<0?vec3(1.0):exp(-WATER_BODY_DATA[owner].underwater.y*vec3(1.55,1.0,0.8)*depth);
}
#endif
