// Exact ray events of custom triangle columns. Events include footprint
// edges, each sloping surface/floor, and ordering changes in stacked faces.
float water_mesh_next(GpuWater body,vec3 origin,vec3 direction,vec2 range){
    uint base=body.coverage.y,count=body.coverage.z,size=uint(WATER_COVERAGE_DATA.length());
    if(base>size||count>size-base)return range.y;
    float cursor=range.x,next=range.y;uint i=0u;
        while(i+1u<count){vec4 box=WATER_COVERAGE_DATA[base+i];uvec4 meta=floatBitsToUint(WATER_COVERAGE_DATA[base+i+1u]);
            if(meta.x<=i||meta.x>count)break;
            vec2 hit=vec2(cursor,next);if(!water_path_slab(origin.x,direction.x,box.x,box.z,hit)||!water_path_slab(origin.z,direction.z,box.y,box.w,hit)){i=meta.x;continue;}
            if(meta.y==0u){i+=2u;continue;}
            if(i+4u<count){vec3 a=WATER_COVERAGE_DATA[base+i+2u].xyz,b=WATER_COVERAGE_DATA[base+i+3u].xyz,c=WATER_COVERAGE_DATA[base+i+4u].xyz;
                vec2 ab=b.xz-a.xz,ac=c.xz-a.xz;float determinant=water_path_cross(ab,ac);
                if(abs(determinant)>1e-10){
                    vec2 u=vec2(water_path_cross(origin.xz-a.xz,ac),water_path_cross(direction.xz,ac))/determinant;
                    vec2 v=vec2(water_path_cross(ab,origin.xz-a.xz),water_path_cross(ab,direction.xz))/determinant;
                    vec2 height=vec2(a.y,0.0)+(b.y-a.y)*u+(c.y-a.y)*v;
                    if(abs(u.y)>1e-12)water_ray_root(-u.x/u.y,vec2(cursor,next),next);
                    if(abs(v.y)>1e-12)water_ray_root(-v.x/v.y,vec2(cursor,next),next);
                    if(abs(u.y+v.y)>1e-12)water_ray_root((1.0-u.x-v.x)/(u.y+v.y),vec2(cursor,next),next);
                    float dy=direction.y-height.y;if(abs(dy)>1e-12){water_ray_root((height.x-origin.y)/dy,vec2(cursor,next),next);water_ray_root((height.x-uintBitsToFloat(body.coverage.w)-origin.y)/dy,vec2(cursor,next),next);}
                }
            }i=meta.x;
        }
    return next;
}
float water_mesh_distance(GpuWater body,vec3 origin,vec3 direction,vec2 range){
    float cursor=range.x,wet=0.0;
    while(cursor<range.y){float next=water_mesh_next(body,origin,direction,vec2(cursor,range.y));
        if(next<=cursor)break;
        vec3 p=origin+direction*(cursor+(next-cursor)*0.5);vec4 column;float edge;
        if(water_column(body,p.xz,column,edge)&&p.y<=column.x&&p.y>=column.x-column.y)wet+=max(next-cursor-water_dry_distance(body,origin,direction,vec2(cursor,next)),0.0);
        cursor=next;
    }return wet;
}
