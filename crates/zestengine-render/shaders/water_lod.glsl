// Shared zone geomorph. The parent cell is a four-triangle centre fan, just
// like water_zone.rs. Fine triangles subdivide that fan, so interpolating its
// displaced vertices converges to the coarse surface before the tile changes.
// Caller provides water_vertex_sample(slot, xz, spacing, rest_y, edge).
vec3 water_lod_position(int slot,vec2 xz,float spacing,float zone_step,vec3 fine) {
    if(zone_step<=0.0 || WATER_BODY_DATA[slot].zone_lod.w==0.0)return fine;
    mat4 mapping=WATER_BODY_DATA[slot].zone_to_water;
    mat2 basis=mat2(mapping[0].xz,mapping[2].xz);
    if(abs(determinant(basis))<1e-10)return fine;
    vec2 p=inverse(basis)*(xz-mapping[3].xz);
    vec4 policy=WATER_BODY_DATA[slot].zone_lod;
    vec3 camera=WATER_BODY_DATA[slot].zone_camera.xyz;
    float parent=2.0*zone_step,tile=parent*policy.x;
    // At tile edges choose the farthest incident tile. Both sides then use
    // the same weight, including vertices retained by 2:1 edge stitching.
    vec2 tile_at=floor((p+tile*1e-5)/tile)*tile;
    float distance_m=0.0;
    for(int z=0;z<2;++z)for(int x=0;x<2;++x){
        vec2 lo=tile_at-vec2(x,z)*tile;
        if(any(lessThan(p,lo-vec2(tile*2e-5)))||any(greaterThan(p,lo+tile+vec2(tile*2e-5))))continue;
        vec2 delta=camera.xz-clamp(camera.xz,lo,lo+tile);
        distance_m=max(distance_m,length(vec3(delta,camera.y)));
    }
    // Topology uses a quantized camera; finish early enough to cover its
    // maximum rounding error. The render weight follows the unsnapped camera.
    float threshold=tile*policy.y;
    float guard=policy.z*policy.x*0.22;
    float weight=smoothstep(threshold*0.9,threshold-guard,distance_m);
    if(weight<=0.0)return fine;
    vec2 lo=floor(p/parent)*parent,center=lo+parent*0.5;
    vec2 q=(p-center)/(parent*0.5);
    vec2 a,b;float wa,wb;
    if(abs(q.x)>abs(q.y)){
        float side=q.x<0.0?-1.0:1.0;
        a=center+vec2(side,-1.0)*(parent*0.5);b=center+vec2(side,1.0)*(parent*0.5);
        wa=(abs(q.x)-q.y)*0.5;wb=(abs(q.x)+q.y)*0.5;
    }else{
        float side=q.y<0.0?-1.0:1.0;
        a=center+vec2(-1.0,side)*(parent*0.5);b=center+vec2(1.0,side)*(parent*0.5);
        wa=(abs(q.y)-q.x)*0.5;wb=(abs(q.y)+q.x)*0.5;
    }
    vec3 coarse=vec3(0.0);
    vec2 points[3]=vec2[3](center,a,b);float weights[3]=float[3](1.0-wa-wb,wa,wb);
    for(int i=0;i<3;++i){if(weights[i]<=1e-6)continue;
        vec2 local=basis*points[i]+mapping[3].xz;
        vec4 column;float edge;water_vertex_column(WATER_BODY_DATA[slot],local,column,edge);
        coarse+=water_vertex_sample(slot,local,spacing*2.0,column.x,edge)*weights[i];
    }
    return mix(fine,coarse,weight);
}
