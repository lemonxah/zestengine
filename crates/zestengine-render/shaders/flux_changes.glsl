// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Where the scene recently changed (flux_changes.rs). Every lighting history
// keeps its accumulated light except near these regions, so one moving object
// does not cost the rest of the scene its variance reduction.
// Include after the pass's Params block `pc`, which ends with
//   vec4 change_info;       x = region count
//   vec4 change_bounds[32]; [2i]   = min.xyz, reach in metres: positive where
//                                    geometry moved, negative for a light or
//                                    material change
//                           [2i+1] = max.xyz, frames since the region changed,
//                                    plus 4096 where a light changed, plus
//                                    8192 for animation in place
#ifndef FLUX_CHANGES
#define FLUX_CHANGES
const uint FLUX_CHANGE_REGIONS = 16u;
// What a consumer reacts to: everything (bounce light), moved geometry, or a
// changed light (other surfaces' direct light).
const int FLUX_CHANGE_ANY = 0, FLUX_CHANGE_GEOMETRY = 1, FLUX_CHANGE_DIRECT = 2;
// Screen histories beside moving geometry follow about the last ten frames
// for this long after it moved, as the whole view used to.
const float FLUX_MOTION_ALPHA = 0.10;
const float FLUX_MOTION_FRAMES = 8.0;

// Test-only switch (exposure.w bit 1): histories ignore every change.
bool flux_changes_ignored() {return (uint(pc.exposure.w+0.5)&2u)!=0u;}

// How much of the lighting history at `p` a recent change invalidates: one
// inside a changed region and up to its reach, then falling with the solid
// angle the region subtends. `newer_than` = frames since the history at `p`
// was last refreshed; an older change is already part of it.
float flux_change_weight(vec3 p,float newer_than,int kinds) {
    if(flux_changes_ignored()) return 0.0;
    float weight=0.0;
    uint count=min(uint(pc.change_info.x),FLUX_CHANGE_REGIONS);
    for(uint i=0u;i<count;++i) {
        vec4 lo=pc.change_bounds[i*2u],hi=pc.change_bounds[i*2u+1u];
        bool direct=mod(hi.w,8192.0)>=4096.0;
        // A source that keeps changing in place is animation: bounce light
        // averages it like any other light. Only its direct light follows.
        bool sustained=hi.w>=8192.0;
        if(mod(hi.w,4096.0)>newer_than || (kinds==FLUX_CHANGE_GEOMETRY && lo.w<=0.0) || (kinds==FLUX_CHANGE_DIRECT && !direct)
            || (kinds==FLUX_CHANGE_ANY && sustained)) continue;
        vec3 outside=max(max(lo.xyz-p,p-hi.xyz),vec3(0.0));
        float reach2=lo.w*lo.w,distance2=dot(outside,outside);
        weight=max(weight,distance2<=reach2 ? 1.0 : reach2/distance2);
    }
    return weight;
}

// How much a moved body (bounding sphere `centre`, `radius`) changes the light
// a source sends towards `p`. `to_source` is the unit direction from `p` to
// the source, `span` its distance (above 1e15: a directional light, whose
// `source_radius` is then the tangent of its angular radius).
// Direct light (`bounce` false): one where the body lies on the paths from `p`
// to the source, fading over half its radius.
// Bounce light (`bounce` true): the shadow lands on surfaces along that path,
// however far behind the body, and what they reflect changes around them. One
// inside the shadow, then falling with the solid angle the shadowed surfaces
// cover from `p`. Zero where `p` is not behind the body: its own reach covers
// that side (flux_change_weight).
float flux_shadow_weight(vec3 p,vec3 to_source,float span,float source_radius,vec3 centre,float radius,bool bounce) {
    float along=dot(centre-p,to_source);
    if(bounce && (along<=0.0 || along>=span)) return 0.0;
    along=clamp(along,0.0,span);
    vec3 nearest=p+to_source*along-centre;
    // The paths to an extended source fan out from `p` towards it.
    float reach=radius+source_radius*(span<1e15 ? along/span : along);
    float distance2=dot(nearest,nearest);
    if(!bounce) return 1.0-smoothstep(reach*reach,2.25*reach*reach,distance2);
    // Measured at the body, where the shadow is as wide as the body: the
    // shadow and the distance to it grow by the same factor behind it.
    return distance2<=reach*reach ? 1.0 : reach*reach/distance2;
}

// Minimum blend of a screen history at `p`: the settled value, or faster
// beside geometry that moved within the last few frames.
float flux_motion_alpha(vec3 p,float settled) {
    return max(settled,FLUX_MOTION_ALPHA*flux_change_weight(p,FLUX_MOTION_FRAMES,FLUX_CHANGE_GEOMETRY));
}
#endif
