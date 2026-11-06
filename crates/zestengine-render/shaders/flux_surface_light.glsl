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
    rayQueryInitializeEXT(rq,tlas,gl_RayFlagsNoneEXT,RT_MASK_SOLID,origin,1e-3,direction,length(pc.gdf_max.xyz-pc.gdf_min.xyz));
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
    if (dot(n,direction)>0.0) return vec3(0.0);
    hp=origin+direction*rayQueryGetIntersectionTEXT(rq,true);
    albedo=instance.albedo.rgb*(1.0-clamp(instance.albedo.w,0.0,1.0));
    emission=instance.emission.rgb;
    float error=0.01;
#else
    uint who;
    if (!march(origin,direction,pc.march.x,length(pc.gdf_max.xyz-pc.gdf_min.xyz),hp,who)) return pc.sky.rgb;
    n=hit_normal(hp,who,pc.march.x);
    if (dot(n,direction)>0.0) n=-n;
    vec4 material=hit_albedo(who);
    albedo=material.rgb*(1.0-clamp(material.w,0.0,1.0));
    emission=hit_emission(who);
    float error=(who&MESH_SDF_HIT)!=0u ? 0.01 : gdf_surface_offset(pc.march.x);
#endif
#ifdef FLUX_SURFACE_RT
    SurfaceSample hit=surface_lookup_instance(hp,n,error,uint(instance.emission_raw.w));
#else
    SurfaceSample hit=(who&MESH_SDF_HIT)!=0u
        ? surface_lookup_mesh(hp,n,error,who&~MESH_SDF_HIT) : surface_lookup_gdf(hp,n,error,who);
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
void main() {
    if (gl_WorkGroupID.z>=surface_tree_info.w) return;
    SurfaceNode update=surface_nodes[surface_tree_info.z+gl_WorkGroupID.z];
    uint card_id=update.left;
    bool update_direct=(update.right&1u)!=0u,update_indirect=(update.right&2u)!=0u;
    SurfaceCard card=surface_cards[card_id];
    const int spacing=4;
    ivec2 local=ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(local,ivec2(card.rect.z/uint(spacing))))) return;
    ivec2 texel=ivec2(card.rect.xy)/spacing+local;
    ivec2 material_texel=texel*spacing+ivec2(spacing/2);
    ivec2 extent=imageSize(surface_direct_out);
    vec4 position=texelFetch(surface_position,material_texel,0);
    if (position.w<0.5) {
        if (update_direct) imageStore(surface_direct_out,texel,vec4(0.0));
        if (update_indirect) imageStore(surface_indirect_out,texel,vec4(0.0));
        return;
    }
    vec3 p=surface_world_position(card,position.xyz);
    vec3 n=normalize(surface_world_normal(card,texelFetch(surface_normal,material_texel,0).xyz));
    vec4 material=texelFetch(surface_albedo,material_texel,0);
    vec3 albedo=material.rgb*(1.0-material.a);
    uint spatial_seed=hash(uint(texel.x)+uint(texel.y)*uint(extent.x));
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
    if (!update_indirect) return;
    uint sample_index=uint(update.bmin.y);
    uint rng=hash(spatial_seed+sample_index*747796405u);
    vec4 old_indirect=(update.right&8u)!=0u ? texelFetch(surface_indirect,texel,0) : vec4(0.0);
    float indirect_count=min(old_indirect.a+1.0,4.0);
    vec3 indirect=vec3(0.0);
    if (pc.counts.y>1u && max(albedo.r,max(albedo.g,albedo.b))>0.001) {
#ifdef FLUX_SURFACE_RT
        float offset=max(0.005,card.normal_texel.w*0.05);
#else
        float offset=software_surface_offset(p,pc.march.x);
#endif
        // A 4x4 directional grid, like Lumen's default radiosity probes.
        // A progressive low-discrepancy shift explores each cell across page
        // updates; a fixed per-probe rotation decorrelates adjacent probes.
        // No ray tree: every direction performs one transport step only.
        vec2 rotation=vec2(float(spatial_seed),float(hash(spatial_seed)))*(1.0/4294967296.0);
        vec2 jitter=fract(rotation+vec2(0.754877666,0.569840296)*float(sample_index+1u));
        for (uint i=0u;i<16u;++i) {
            vec2 u=(vec2(i&3u,i>>2u)+jitter)*0.25;
            vec3 direction=cosine_hemisphere_dir(n,u);
            indirect+=surface_incoming(p+n*offset,direction,rng);
        }
        indirect*=albedo*(1.0/16.0);
    }
    indirect=mix(old_indirect.rgb,indirect,1.0/indirect_count);
    // Preserve finite HDR radiance; never max-blend brightness or feed exposed
    // screen history back into the transport equation.
    imageStore(surface_indirect_out,texel,vec4(clamp(indirect,0.0,65000.0),indirect_count));
}
