// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Diffuse radiosity: one transport step from the previous surface-lighting
// atlas. Repeated budgeted updates propagate multiple bounces without tracing
// a growing ray tree at every screen pixel. GI display intensity is excluded.
bool surface_is_emitter(vec3 p) {
#ifdef FLUX_SURFACE_RT
    return on_emitter(p);
#else
    for (uint i=0u;i<pc.counts.w;++i) {
        float r=max(emitters[i].center_radius.w,1e-3);
        if (length(p-emitters[i].center_radius.xyz)<r*1.5+0.05) return true;
    }
    return false;
#endif
}
vec3 surface_direct_light(vec3 p,vec3 n,inout uint rng) {
#ifdef FLUX_SURFACE_RT
    return direct_light(p,n,rng)+emitter_light(p,n);
#else
    return direct_light(p,n,pc.march.x,rng)+emitter_light(p,n,pc.march.x);
#endif
}
vec3 surface_incoming(vec3 origin,vec3 direction,inout uint rng) {
    vec3 hp,n,albedo,emission;
#ifdef FLUX_SURFACE_RT
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq,tlas,gl_RayFlagsNoneEXT,RT_MASK_SOLID,origin,1e-3,direction,pc.march.y);
    while(rayQueryProceedEXT(rq)) {
        if(rayQueryGetIntersectionTypeEXT(rq,false)==gl_RayQueryCandidateIntersectionTriangleEXT && candidate_covered(rq)) rayQueryConfirmIntersectionEXT(rq);
    }
    if (rayQueryGetIntersectionTypeEXT(rq,true)==gl_RayQueryCommittedIntersectionNoneEXT) return pc.sky.rgb;
    uint ci=rayQueryGetIntersectionInstanceCustomIndexEXT(rq,true);
    uint primitive=rayQueryGetIntersectionPrimitiveIndexEXT(rq,true);
    Instance instance=insts.data[ci];
    Indices idx=Indices(instance.idx_addr); Vertices vertices=Vertices(instance.vtx_addr);
    Vertex a=vertices.v[idx.i[primitive*3u]],b=vertices.v[idx.i[primitive*3u+1u]],c=vertices.v[idx.i[primitive*3u+2u]];
    mat3 model=mat3(rayQueryGetIntersectionObjectToWorldEXT(rq,true));
    n=cross(model*(b.position-a.position),model*(c.position-a.position));
    if (dot(n,n)<1e-12) return vec3(0.0);
    n=normalize(n);
    if (instance.opacity.w<0.0) n=-n;
    if (dot(n,direction)>0.0) {
        if (instance.opacity.w<0.5) return vec3(0.0);
        n=-n;
    }
    hp=origin+direction*rayQueryGetIntersectionTEXT(rq,true);
    albedo=instance.albedo.rgb*(1.0-clamp(instance.albedo.w,0.0,1.0));
    emission=instance.emission.rgb;
    // Match hardware coverage at the actual triangle hit. Software capture
    // texels must not decorrelate its opacity mask and emission texture.
    if(instance.opacity.y==-2.0) {
        vec2 bc=rayQueryGetIntersectionBarycentricsEXT(rq,true);float w=1.0-bc.x-bc.y;
        vec4 tangent=a.tangent*w+b.tangent*bc.x+c.tangent*bc.y;
        tangent=vec4(model*tangent.xyz,tangent.w*sign(determinant(model)));
        RtSurfaceMaterial material=rt_surface_material(instance,a.uv*w+b.uv*bc.x+c.uv*bc.y,
            a.color.rgb*w+b.color.rgb*bc.x+c.color.rgb*bc.y,n,tangent,0.0,0.0);
        albedo=material.albedo*(1.0-material.metallic);emission=material.emission;
    }
    float error=0.01;
#else
    uint who;
    if (!march(origin,direction,pc.march.x,pc.march.y,hp,who)) return pc.sky.rgb;
    // A radiosity ray that is still inside the fat coarse shell hits the card
    // being updated and stores that card's own light. Keep a near hit when the
    // field does not contain the origin.
    float shell=gdf_surface_offset(origin,pc.march.x);
    if (scene_distance(origin)<shell && length(hp-origin)<shell
        && !march_range(origin,direction,pc.march.x,shell,max(pc.march.y,shell),hp,who)) return pc.sky.rgb;
    bool backface;
    if (!software_material_normal(hp,who,pc.march.x,direction,n,backface)) return vec3(0.0);
    vec4 material=hit_albedo(who);
    albedo=material.rgb*(1.0-clamp(material.w,0.0,1.0));
    emission=hit_emission(who);
    float error=(who&MESH_SDF_HIT)!=0u ? 0.01 : gdf_surface_offset(hp,pc.march.x);
#endif
#ifdef FLUX_SURFACE_RT
    SurfaceSample hit=surface_lookup_instance(hp,n,error,uint(instance.emission_raw.w));
    if(instance.opacity.y==-2.0) hit.valid=false;
#else
    SurfaceSample hit=(who&MESH_SDF_HIT)!=0u
        ? surface_lookup_mesh(hp,n,error,who&~MESH_SDF_HIT) : surface_lookup_gdf(hp,n,error,who);
    if (backface && !hit.valid) return vec3(0.0);
#endif
    bool cached;
    vec3 radiance=surface_radiance(hit,cached);
    if (cached) return radiance + (surface_is_emitter(hp) ? vec3(0.0) : hit.emission*pc.transport.x);
    if (hit.valid) { albedo=hit.albedo*(1.0-hit.metallic); emission=hit.emission; n=hit.normal; }
    // Missing cards / initial cache warmup retain a physical one-hit shade.
    // Analytic emitter sampling already accounts for the emitters themselves.
    if (surface_is_emitter(hp)) emission=vec3(0.0);
    return emission*pc.transport.x+(albedo/PI)*surface_direct_light(hp+n*max(pc.march.x,0.005),n,rng);
}
// Settled radiosity blend: one sixteenth of each update, reached after a
// warm-up weighted towards the newest samples (2/(n+1)), so early estimates
// that lack later bounces do not linger. Lumen keeps four updates; that left
// each texel's sixteen-ray noise visible in every gather that read it.
const float SURFACE_RADIOSITY_HISTORY=31.0;
// Bounce light needs several updates to follow a change: each one reads the
// previous atlas. Keep following for this many frames after the last change
// nearby, as the former scene-wide reactive window did.
const float SURFACE_FOLLOW_FRAMES=32.0;
// Receivers share their traced radiance with neighbours on their plane within
// this distance. Where a page is fine enough for a mirror to show its texels,
// that is two receivers each way; a coarse page shares with fewer.
const float SURFACE_SHARE_REACH=0.5;
// Test-only switch (exposure.w bit 2): the surface cache as it was before its
// extra smoothing (independent light samples, nearest neighbours only).
bool surface_smoothing_off() {return (uint(pc.exposure.w+0.5)&4u)!=0u;}
#ifdef FLUX_SURFACE_RT
#define SURFACE_LIGHT(i) lights.data[i]
#define SURFACE_EMITTER(i) emitters.data[i]
#else
#define SURFACE_LIGHT(i) lights[i]
#define SURFACE_EMITTER(i) emitters[i]
#endif
// Moved geometry changes direct light wherever its old or new place lies
// between a texel and a light or analytic emitter, however long that shadow
// is: one there. With `bounce`, how much of the light reflected towards `p`
// changed with those shadows instead (flux_shadow_weight).
float surface_shadow_change(vec3 p,float newer_than,bool bounce) {
    if(flux_changes_ignored()) return 0.0;
    float weight=0.0;
    uint count=min(uint(pc.change_info.x),FLUX_CHANGE_REGIONS);
    for(uint r=0u;r<count;++r) {
        vec4 lo=pc.change_bounds[r*2u],hi=pc.change_bounds[r*2u+1u];
        if(mod(hi.w,4096.0)>newer_than || lo.w<=0.0) continue;
        vec3 centre=0.5*(lo.xyz+hi.xyz);
        float radius=0.5*length(hi.xyz-lo.xyz);
        for(uint i=0u;i<pc.counts.z;++i) {
            vec4 pos_kind=SURFACE_LIGHT(i).pos_kind,dir_range=SURFACE_LIGHT(i).dir_range;
            uint kind=uint(pos_kind.w+0.5);
            if(kind==6u) continue;
            // Disc of a point or sun light; an area light spans right + up.
            float source_radius=max(SURFACE_LIGHT(i).spot.z,0.0);
            if((kind==4u || kind==5u) && i+1u<pc.counts.z) source_radius=length(dir_range.xyz)+length(SURFACE_LIGHT(i+1u).dir_range.xyz);
            vec3 to_light=-dir_range.xyz;float span=1e16;
            if(kind!=0u) {
                to_light=pos_kind.xyz-p;span=length(to_light);
                // Bounce reaches a texel just outside the range from the lit surfaces inside it.
                if(span<1e-4 || (dir_range.w>0.0 && span>dir_range.w+source_radius+(bounce ? 3.0*radius : 0.0))) continue;
            }
            weight=max(weight,flux_shadow_weight(p,normalize(to_light),span,source_radius,centre,radius,bounce));
        }
        for(uint i=0u;i<pc.counts.w;++i) {
            vec4 emitter=SURFACE_EMITTER(i).center_radius;
            vec3 to_light=emitter.xyz-p;float span=length(to_light);
            if(span<1e-4) continue;
            weight=max(weight,flux_shadow_weight(p,to_light/span,span,max(emitter.w,0.0),centre,radius,bounce));
        }
    }
    return weight;
}

// One workgroup is one 8x8 lighting tile of a single card. Every receiver
// publishes this update's traced radiance so its neighbours can share it
// before temporal accumulation, as Lumen filters neighbouring radiosity probes.
// Four accumulated updates alone leave each texel with its own 16 ray noise.
shared vec4 tile_gathered[64]; // rgb mean incoming radiance, w = 1 when traced
shared vec3 tile_position[64];
shared vec3 tile_normal[64];

void main() {
    uint lane=gl_LocalInvocationIndex;
    tile_gathered[lane]=vec4(0.0);tile_position[lane]=vec3(0.0);tile_normal[lane]=vec3(0.0);
    // No invocation may leave before the barrier below.
    bool job=gl_WorkGroupID.z<surface_tree_info.w;
    SurfaceNode update=surface_nodes[surface_tree_info.z+(job ? gl_WorkGroupID.z : 0u)];
    uint card_id=update.left;
    bool update_direct=job && (update.right&1u)!=0u,update_indirect=job && (update.right&2u)!=0u;
    SurfaceCard card=surface_cards[card_id];
    int spacing=surface_lighting_spacing();
    // Jobs retain the page update/sample indices and supply a local tile
    // origin. The physical texel and ray sequence are identical to the former
    // rectangular dispatch, including on mixed-resolution pages.
    ivec2 local=ivec2(gl_GlobalInvocationID.xy)+ivec2(update.bmax.xy);
    bool inside=job && all(lessThan(local,ivec2(card.rect.z/uint(spacing))));
    ivec2 texel=ivec2(card.rect.xy)/spacing+local;
    ivec2 material_texel=texel*spacing+ivec2(spacing/2);
    vec4 position=inside ? texelFetch(surface_position,material_texel,0) : vec4(0.0);
    bool receiver=inside && position.w>=0.5;
    if (inside && !receiver) {
        if (update_direct) imageStore(surface_direct_out,texel,vec4(0.0));
        if (update_indirect) imageStore(surface_indirect_out,texel,vec4(0.0));
    }
    vec3 p=vec3(0.0),n=vec3(0.0,0.0,1.0),albedo=vec3(0.0),gathered=vec3(0.0);
    bool traced=false;
    if (receiver) {
        p=surface_world_position(card,position.xyz);
        n=normalize(surface_world_normal(card,texelFetch(surface_normal,material_texel,0).xyz));
        vec4 material=texelFetch(surface_albedo,material_texel,0);
        albedo=material.rgb*(1.0-material.a);
        // Physical atlas growth/relocation must not reshuffle a resident card's
        // ray sequence. Its logical identity and local probe coordinate are stable.
        uint logical_card=card.virtual_page.x>0u ? card.virtual_page.x-1u : card_id;
        // Each128-texel tile has120 useful material texels; the receiver count
        // follows the quality-dependent spacing.
        // Overlapping sibling gutters therefore share the same logical probe
        // coordinate and ray sequence, independently of physical allocation.
        uvec2 logical_probe=uvec2(local)+uvec2(card.virtual_page.z&65535u,card.virtual_page.z>>16u)*(120u/uint(spacing));
        uint spatial_seed=hash(logical_card*0x9e3779b9u+logical_probe.x*1973u+logical_probe.y*9277u
            +card.virtual_page.y*0x85ebca6bu);
        // Each lighting component owns its update sequence and accumulation. A
        // faster direct schedule must not alter radiosity's directional samples.
        if (update_direct) {
            uint rng=hash(spatial_seed+uint(update.bmin.x)*747796405u);
            vec4 old_direct=(update.right&4u)!=0u ? texelFetch(surface_direct,texel,0) : vec4(0.0);
            // A changed light, or moved geometry between this texel and a light,
            // makes the accumulated samples describe another scene. Elsewhere
            // they stay valid: a prop moving across the room keeps this texel's
            // soft-shadow history. Ages are frames since this page was updated.
            float age=float((update.right>>8u)&4095u);
            float change=(update.right&32u)!=0u ? 0.0
                : max(flux_change_weight(p,age,FLUX_CHANGE_DIRECT),surface_shadow_change(p,age,false));
            float count=min(old_direct.a+1.0,32.0);
            float alpha=max(1.0/count,change);
            // This texel's shadow rays visit each light along a progressive
            // low-discrepancy sequence, shifted per texel: the accumulated
            // penumbra of a light with a size settles without visible grain.
            vec2 shift=vec2(float(hash(spatial_seed^0x68bc21ebu)),float(hash(spatial_seed^0x02e5be93u)))*(1.0/4294967296.0);
            if(!surface_smoothing_off()) flux_light_point=fract(shift+vec2(0.7548776662,0.5698402910)*float(uint(update.bmin.x)+1u));
            vec3 direct=(albedo/PI)*surface_direct_light(p+n*0.005,n,rng);
            flux_light_point=vec2(-1.0);
            direct=mix(old_direct.rgb,direct,alpha);
            // An undisturbed history keeps its exact count; a restart is as long as its blend.
            imageStore(surface_direct_out,texel,vec4(clamp(direct,0.0,65000.0),change*count>1.0 ? 1.0/alpha : count));
        }
        if (update_indirect && pc.counts.y>1u && max(albedo.r,max(albedo.g,albedo.b))>0.001) {
            uint sample_index=uint(update.bmin.y);
            uint rng=hash(spatial_seed+sample_index*747796405u);
#ifdef FLUX_SURFACE_RT
            float offset=max(0.005,card.normal_texel.w*0.05);
#else
            float offset=software_surface_offset(p,pc.march.x);
#endif
            // Scene Lighting Quality scales angular resolution independently of
            // the page update budget. Default4x4 retains the existing sequence.
            // A progressive low-discrepancy shift explores each cell across page
            // updates; a fixed per-probe rotation decorrelates adjacent probes.
            // No ray tree: every direction performs one transport step only.
            vec2 rotation=vec2(float(spatial_seed),float(hash(spatial_seed)))*(1.0/4294967296.0);
            vec2 jitter=fract(rotation+vec2(0.754877666,0.569840296)*float(sample_index+1u));
            uint resolution=update.bmax.z>0.0 ? uint(clamp(update.bmax.z,2.0,8.0)) : 4u;
            uint rays=resolution*resolution;
            // Test-only paired reference: the unlimited, unfiltered estimate.
            bool reference=pc.exposure.z>0.5;
            float limit=reference ? 1.0e30 : flux_radiosity_max_intensity();
            for (uint i=0u;i<rays;++i) {
                vec2 u=flux_radiosity_uv(i,sample_index,spatial_seed,jitter,resolution);
                vec3 direction=cosine_hemisphere_dir(n,u);
                gathered+=firefly_clamp(surface_incoming(p+n*offset,direction,rng),limit);
            }
            gathered/=float(rays);
            traced=true;
        }
        // Neighbour gating uses the geometric normal that every cache lookup validates
        // against; shading normals (normal maps, smooth vertex normals) would reject
        // neighbours on perfectly planar surfaces.
        vec3 geometric_normal=normalize(surface_world_normal(card,position.w>=1.5
            ? surface_geometry_normal(position.w,vec3(0,0,1)) : texelFetch(surface_normal,material_texel,0).xyz));
        tile_gathered[lane]=vec4(gathered,float(traced));tile_position[lane]=p;tile_normal[lane]=geometric_normal;
    }
    barrier();
    if (!receiver || !update_indirect) return;
    vec4 old_indirect=(update.right&8u)!=0u ? texelFetch(surface_indirect,texel,0) : vec4(0.0);
    float indirect_count=min(old_indirect.a+1.0,SURFACE_RADIOSITY_HISTORY);
    vec3 indirect=vec3(0.0);
    if (traced && pc.exposure.z>0.5) indirect=albedo*gathered;
    else if (traced) {
        // Share incoming radiance, not reflected light: each receiver keeps its
        // own albedo. Neighbours must lie on this receiver's plane, by the same
        // test the cache lookup applies, and within SURFACE_SHARE_REACH of it.
        // A tile edge simply has fewer of them.
        float tolerance=max(0.005,card.normal_texel.w*0.1);
        vec3 own_normal=tile_normal[lane];
        vec3 sum=gathered;float total=1.0;
        ivec2 cell=ivec2(gl_LocalInvocationID.xy);
        int reach=surface_smoothing_off() ? 1 : 2;
        const float falloff[3]=float[3](1.0,0.5,0.25);
        for (int y=-reach;y<=reach;++y) for (int x=-reach;x<=reach;++x) {
            ivec2 q=cell+ivec2(x,y);
            if ((x==0 && y==0) || any(lessThan(q,ivec2(0))) || any(greaterThan(q,ivec2(7)))) continue;
            uint other=uint(q.y*8+q.x);
            if (tile_gathered[other].w<0.5) continue;
            vec3 delta=tile_position[other]-p;
            if (dot(tile_normal[other],own_normal)<0.9
                || max(abs(dot(delta,own_normal)),abs(dot(delta,tile_normal[other])))>tolerance) continue;
            // The nearest ring always shares, as before; the second only nearby.
            if (max(abs(x),abs(y))>1 && dot(delta,delta)>SURFACE_SHARE_REACH*SURFACE_SHARE_REACH) continue;
            float weight=falloff[abs(x)]*falloff[abs(y)];
            sum+=weight*tile_gathered[other].rgb;total+=weight;
        }
        indirect=albedo*sum/total;
    }
    // A larger card costs more texels and is consequently scheduled less
    // often. Where the scene changed recently, or at any time since this
    // page was last updated, preserve the established short response in
    // frame time instead of multiplying stale-light lifetime by its area.
    // The weight is one beside the change and falls with the solid angle it
    // covers from this texel; settled texels keep their variance reduction
    // however often they are scheduled, and whatever moves elsewhere. A moved
    // body also changes the bounce around the shadows it casts or lifts,
    // which can lie far from the body itself.
    float age=float(update.right>>20u),newer_than=max(age,SURFACE_FOLLOW_FRAMES);
    float change=(update.right&16u)!=0u ? 0.0
        : max(flux_change_weight(p,newer_than,FLUX_CHANGE_ANY),surface_shadow_change(p,newer_than,true));
    float settled=2.0/(indirect_count+1.0),follow=change*(1.0-pow(0.75,max(age,1.0)));
    float alpha=max(settled,follow);
    indirect=mix(old_indirect.rgb,indirect,alpha);
    // Preserve finite HDR radiance; never max-blend brightness or feed exposed
    // screen history back into the transport equation. An undisturbed history
    // keeps its exact count; after following a change it is as long as its blend.
    imageStore(surface_indirect_out,texel,vec4(clamp(indirect,0.0,65000.0),follow>settled ? 2.0/alpha-1.0 : indirect_count));
}
