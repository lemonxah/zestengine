// Shared software intersection implementation used by GI, reflections and geometry regression queries.
#ifndef FLUX_GDF_SOURCE_REFINABLE
#define FLUX_GDF_SOURCE_REFINABLE(index) false
#endif
#include "flux_geometry_clipmap.glsl"
float scene_distance(vec3 p) {
    return flux_field_distance(flux_field_level(p),p);
}
uint scene_index(vec3 p) {
    return flux_field_index(flux_field_level(p),p);
}
bool march_range(vec3 ro, vec3 rd, float eps, float min_dist, float max_dist, out vec3 hit, out uint who) {
    hit=vec3(0.0);who=0u;
    if(max_dist<min_dist || max_dist<=0.0) return false;
    float mesh_t;uint mesh_who;
    bool mesh_hit=mesh_sdf_trace_range(ro,rd,min_dist,max_dist,mesh_t,mesh_who);
    float limit=mesh_hit ? mesh_t : max_dist;
    if(flux_field_detail_only()) {
        // The first pass already tested dynamic/emissive geometry at full range
        // and static geometry near the receiver. Complete the static interval
        // without consulting the pending/stale coarse acceleration snapshot.
        float t;uint index;
        if(mesh_sdf_trace_interval(ro,rd,min_dist,limit,true,t,index)) {mesh_hit=true;mesh_t=t;mesh_who=index;}
        if(mesh_hit) {hit=ro+rd*mesh_t;who=mesh_who;return true;}
        return false;
    }
    float enter,leave;
    vec3 field_lo,field_hi;flux_field_bounds(field_lo,field_hi);
    bool crosses=mesh_sdf_box(ro,rd,field_lo,field_hi,limit,enter,leave);
    float far_start=max(min_dist,MESH_DETAIL_DISTANCE-eps);
    // A field domain is an acceleration region, not a promise that the world
    // ends there. Trace static detail in the uncovered ray intervals too.
    // Dynamic/emissive fields already cover the full range in the first pass.
    if(MESH_SDF_NODE_COUNT>0.0) {
        float t;uint index;
        if(far_start<=limit && flux_field_uncovered(ro,rd,far_start,limit,t,index)) {
            mesh_hit=true;mesh_t=t;mesh_who=index;limit=t;
        }
    }
    bool refine=FLUX_GDF_REFINABLE && MESH_SDF_NODE_COUNT>0.0;
    bool partial=!refine && MESH_SDF_NODE_COUNT>0.0;
    bool partial_detail_tested=false;
    // Only complete identified snapshots may replace the coarse near field.
    // Otherwise an unrelated detail mesh would hide every unmapped near wall.
    float t=max(min_dist,refine ? MESH_DETAIL_DISTANCE-eps : eps*2.0);
    t=max(t,enter);
    limit=min(limit,leave);
    // Global reconstruction can erase a sub-voxel wall even though its local
    // field still represents it. Expand only the broad-phase candidate band:
    // detailed intersection below determines the actual hit, so openings stay
    // open and shading never uses a dilated coarse surface. A cell diagonal
    // allows for reconstruction of both mesh and global samples.
    for(uint i=0u;crosses && i<MAX_STEPS && t<=limit;++i) {
        vec3 p=ro+rd*t;
        int level=flux_field_level(p);
        vec3 cell=flux_field_cell(level);
        bool inside=all(greaterThanEqual(p,flux_field_min(level))) && all(lessThanEqual(p,flux_field_max(level)));
        uint coarse_index=partial && inside ? flux_field_index(level,p) : 0u;
        bool represented=partial && inside && FLUX_GDF_SOURCE_REFINABLE(coarse_index);
        float candidate_band=inside && (represented || refine) ? length(cell) : 0.0;
        float d=flux_field_distance(level,p)-candidate_band;
        // Outside all resident regions the mesh pass above owns the interval.
        // Do not turn the artificial field boundary into a geometry hit.
        if(!inside) {t+=flux_field_step(ro,rd,t,max(d,eps),eps);continue;}
        vec3 cell_t=cell/max(abs(rd),vec3(1e-8));
        float covered_step=max(1e-4,0.5*min(cell_t.x,min(cell_t.y,cell_t.z)));
        if(d<eps) {
            if(refine) {
                // A negative open field is not a surface. Search the mesh past
                // the detail band only when the near pass missed: a nearer mesh
                // hit is in front, and replacing it with the far search paints
                // that farther surface's light back onto the receiver.
                if(!mesh_hit) {
                    float refined_t;uint refined_who;
                    if(mesh_sdf_trace_interval(ro,rd,max(far_start,enter),limit,true,refined_t,refined_who)) {
                        hit=ro+rd*refined_t;who=refined_who;return true;
                    }
                }
                break;
            }
            if(represented) {
                // Near detail was already tested. Resolve all remaining
                // represented static geometry once, then keep marching the
                // coarse field for any closer *unmapped* occluder. A missing
                // unrelated mesh must not refill this source's alpha holes.
                if(!partial_detail_tested) {
                    float refined_t;uint refined_who;
                    if(mesh_sdf_trace_interval(ro,rd,max(far_start,enter),limit,true,refined_t,refined_who)
                        && (!mesh_hit || refined_t<mesh_t)) {
                        mesh_hit=true;mesh_t=refined_t;mesh_who=refined_who;limit=min(limit,refined_t);
                    }
                    partial_detail_tested=true;
                }
                t+=flux_field_step(ro,rd,t,covered_step,eps);continue;
            }
            hit=p;who=scene_index(p);return true;
        }
        t+=flux_field_step(ro,rd,t,max(d,eps),eps);
    }
    if(mesh_hit) {hit=ro+rd*mesh_t;who=mesh_who;return true;}
    return false;
}
bool march(vec3 ro,vec3 rd,float eps,float max_dist,out vec3 hit,out uint who) {
    return march_range(ro,rd,eps,0.0,max_dist,hit,who);
}
bool occluded(vec3 ro, vec3 rd, float dist, float eps) {
    vec3 hit; uint who;
#ifdef MESH_ALPHA_FRAME
    bool previous=mesh_shadow_ray;mesh_shadow_ray=true;
#endif
    bool blocked=march(ro,rd,eps,max(dist-eps,0.0),hit,who);
#ifdef MESH_ALPHA_FRAME
    mesh_shadow_ray=previous;
#endif
    return blocked;
}
// Tetrahedral gradient: 4 distance taps instead of 6 central differences.
// Select a single field for all gradient taps. Switching levels across a
// boundary would differentiate a resolution change instead of the surface.
vec3 scene_normal(vec3 p, float eps) {
    int level=flux_field_level(p);
    const vec2 k = vec2(1.0, -1.0);
    vec3 g = k.xyy * flux_field_raw(level,p + k.xyy * eps)
           + k.yyx * flux_field_raw(level,p + k.yyx * eps)
           + k.yxy * flux_field_raw(level,p + k.yxy * eps)
           + k.xxx * flux_field_raw(level,p + k.xxx * eps);
    return length(g) > 1e-8 ? normalize(g) : vec3(0.0, 1.0, 0.0);
}
vec3 hit_normal(vec3 p,uint who,float eps) {
    return (who&MESH_SDF_HIT)!=0u ? mesh_sdf_normal(who,p) : scene_normal(p,eps);
}
