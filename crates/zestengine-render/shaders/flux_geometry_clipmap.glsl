// Geometry fields are acceleration regions, not world boundaries. Sampler
// array indices below are static, so divergent level selection needs no
// non-uniform descriptor indexing feature.
#ifdef FLUX_GDF_CLIPMAP_BINDING
layout(set=0,binding=FLUX_GDF_CLIPMAP_BINDING,std140) uniform FluxGeometryClipmaps {
    vec4 lo[4]; vec4 hi[4]; uvec4 meta;
} flux_geometry;
bool flux_field_detail_only() {return flux_geometry.meta.y!=0u;}
bool flux_field_active(int level) {return (flux_geometry.meta.x & (1u<<uint(level)))!=0u;}
vec3 flux_field_min(int level) {return flux_geometry.lo[level].xyz;}
vec3 flux_field_max(int level) {return flux_geometry.hi[level].xyz;}
ivec3 flux_field_dims(int level) {
    if(level==1) return textureSize(u_dist[1],0);
    if(level==2) return textureSize(u_dist[2],0);
    if(level==3) return textureSize(u_dist[3],0);
    return textureSize(u_dist[0],0);
}
float flux_field_raw(int level,vec3 p) {
    vec3 uv=flux_gdf_uv(p,flux_field_min(level),flux_field_max(level),flux_field_dims(level));
    if(level==1) return textureLod(u_dist[1],uv,0.0).r;
    if(level==2) return textureLod(u_dist[2],uv,0.0).r;
    if(level==3) return textureLod(u_dist[3],uv,0.0).r;
    return textureLod(u_dist[0],uv,0.0).r;
}
uint flux_field_index(int level,vec3 p) {
    vec3 uv=flux_gdf_uv(p,flux_field_min(level),flux_field_max(level),flux_field_dims(level));
    if(level==1) return textureLod(u_index[1],uv,0.0).r;
    if(level==2) return textureLod(u_index[2],uv,0.0).r;
    if(level==3) return textureLod(u_index[3],uv,0.0).r;
    return textureLod(u_index[0],uv,0.0).r;
}
#else
bool flux_field_detail_only() {return false;}
bool flux_field_active(int level) {return level==0;}
vec3 flux_field_min(int level) {return pc.gdf_min.xyz;}
vec3 flux_field_max(int level) {return pc.gdf_max.xyz;}
ivec3 flux_field_dims(int level) {return textureSize(u_dist,0);}
float flux_field_raw(int level,vec3 p) {return textureLod(u_dist,flux_gdf_uv(p,pc.gdf_min.xyz,pc.gdf_max.xyz,textureSize(u_dist,0)),0.0).r;}
uint flux_field_index(int level,vec3 p) {return textureLod(u_index,flux_gdf_uv(p,pc.gdf_min.xyz,pc.gdf_max.xyz,textureSize(u_index,0)),0.0).r;}
#endif
vec3 flux_field_cell(int level) {
    return (flux_field_max(level)-flux_field_min(level))/max(vec3(flux_field_dims(level)-1),vec3(1.0));
}
int flux_field_level(vec3 p) {
    int chosen=0;float best_outside=1e30,best_cell=1e30;
    for(int i=0;i<4;++i) if(flux_field_active(i)) {
        float outside=length(p-clamp(p,flux_field_min(i),flux_field_max(i)));
        float cell=length(flux_field_cell(i));
        if(outside<best_outside || (outside==best_outside && cell<best_cell)) {
            chosen=i;best_outside=outside;best_cell=cell;
        }
    }
    return chosen;
}
float flux_field_distance(int level,vec3 p) {
    vec3 q=clamp(p,flux_field_min(level),flux_field_max(level));
    float outside=length(p-q);
    // In uncovered gaps, step only to a field boundary. Static mesh tracing
    // handles actual geometry there; a field's clamped value must not jump
    // across a newly resident finer grid.
    return outside>0.0 ? outside : flux_field_raw(level,q);
}
void flux_field_bounds(out vec3 lo,out vec3 hi) {
    lo=flux_field_min(0);hi=flux_field_max(0);
    for(int i=1;i<4;++i) if(flux_field_active(i)) {lo=min(lo,flux_field_min(i));hi=max(hi,flux_field_max(i));}
}
float flux_field_step(vec3 ro,vec3 rd,float t,float distance,float eps) {
#ifdef FLUX_GDF_CLIPMAP_BINDING
    for(int i=0;i<4;++i) if(flux_field_active(i)) {
        float enter,leave;
        if(mesh_sdf_box(ro,rd,flux_field_min(i),flux_field_max(i),t+distance+eps,enter,leave)) {
            if(enter>=t) distance=min(distance,max(enter-t,eps*0.25));
            if(leave>=t) distance=min(distance,max(leave-t,eps*0.25));
        }
    }
#endif
    return distance;
}
// Trace uncovered ray intervals too, including gaps between independently
// scrolled grids after fast travel/teleport. This also protects a second view
// far from the camera that currently controls geometry residency.
bool flux_field_uncovered(vec3 ro,vec3 rd,float start,float limit,out float hit,out uint who) {
    vec2 spans[4];
    for(int i=0;i<4;++i) {
        float a,b;spans[i]=vec2(limit);
        if(flux_field_active(i) && mesh_sdf_box(ro,rd,flux_field_min(i),flux_field_max(i),limit,a,b)) spans[i]=vec2(max(a,start),b);
    }
    for(int i=1;i<4;++i) for(int j=i;j>0 && spans[j].x<spans[j-1].x;--j) {
        vec2 tmp=spans[j];spans[j]=spans[j-1];spans[j-1]=tmp;
    }
    float cursor=start;
    for(int i=0;i<4;++i) {
        if(spans[i].y<cursor) continue;
        if(spans[i].x>cursor && mesh_sdf_trace_interval(ro,rd,cursor,min(spans[i].x,limit),true,hit,who)) return true;
        cursor=max(cursor,spans[i].y);
        if(cursor>=limit) return false;
    }
    return cursor<limit && mesh_sdf_trace_interval(ro,rd,cursor,limit,true,hit,who);
}
