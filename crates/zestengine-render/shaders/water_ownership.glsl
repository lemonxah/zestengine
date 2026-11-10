#ifndef ZEST_WATER_OWNERSHIP
#define ZEST_WATER_OWNERSHIP
// Compare full transformed volumes, not XZ masks: a raised aqueduct must
// preserve unrelated water below it. Explicit dry volumes reveal that water.
bool water_owned(GpuWater body,vec3 world_point){
    if(body.ownership.y<0.5)return true;
    for(int i=0;i<MAX_WATER_BODIES;++i){GpuWater other=WATER_BODY_DATA[i];
        if(other.ownership.y<0.5 || other.ownership.x<=body.ownership.x)continue;
        vec3 p=(other.to_local*vec4(world_point,1.0)).xyz;
        vec4 column;float edge;if(!water_column(other,p.xz,column,edge)||water_excluded(other,p))continue;
        if(p.y>=column.x-column.y && p.y<=column.x+other.ownership.z+0.05)return false;
    }
    return true;
}
#endif
