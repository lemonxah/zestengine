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
            float count=min(old_direct.a+1.0,32.0);
            vec3 direct=(albedo/PI)*surface_direct_light(p+n*0.005,n,rng);
            direct=mix(old_direct.rgb,direct,1.0/count);
            imageStore(surface_direct_out,texel,vec4(clamp(direct,0.0,65000.0),count));
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
    float indirect_count=min(old_indirect.a+1.0,4.0);
    vec3 indirect=vec3(0.0);
    if (traced && pc.exposure.z>0.5) indirect=albedo*gathered;
    else if (traced) {
        // Share incoming radiance, not reflected light: each receiver keeps its
        // own albedo. Neighbours must lie on this receiver's plane, by the same
        // test the cache lookup applies. A tile edge simply has fewer of them.
        float tolerance=max(0.005,card.normal_texel.w*0.1);
        vec3 own_normal=tile_normal[lane];
        vec3 sum=gathered;float total=1.0;
        ivec2 cell=ivec2(gl_LocalInvocationID.xy);
        for (int y=-1;y<=1;++y) for (int x=-1;x<=1;++x) {
            ivec2 q=cell+ivec2(x,y);
            if ((x==0 && y==0) || any(lessThan(q,ivec2(0))) || any(greaterThan(q,ivec2(7)))) continue;
            uint other=uint(q.y*8+q.x);
            if (tile_gathered[other].w<0.5) continue;
            vec3 delta=tile_position[other]-p;
            if (dot(tile_normal[other],own_normal)<0.9
                || max(abs(dot(delta,own_normal)),abs(dot(delta,tile_normal[other])))>tolerance) continue;
            float weight=(x==0 || y==0) ? 0.5 : 0.25;
            sum+=weight*tile_gathered[other].rgb;total+=weight;
        }
        indirect=albedo*sum/total;
    }
    // A larger card costs more texels and is consequently scheduled less
    // often. Preserve the established short response in frame time instead
    // of multiplying stale-light lifetime by its area. The CPU supplies age
    // only during recent lighting changes; settled pages retain four-update
    // variance reduction regardless of their scheduling frequency.
    float age_alpha=1.0-pow(0.75,max(update.bmin.z,1.0));
    indirect=mix(old_indirect.rgb,indirect,max(1.0/indirect_count,age_alpha));
    // Preserve finite HDR radiance; never max-blend brightness or feed exposed
    // screen history back into the transport equation.
    imageStore(surface_indirect_out,texel,vec4(clamp(indirect,0.0,65000.0),indirect_count));
}
