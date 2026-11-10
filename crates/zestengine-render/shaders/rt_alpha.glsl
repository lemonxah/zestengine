// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Masked and alpha-blended TLAS instances produce triangle candidates. Ordinary
// opaque surfaces remain hardware-committed, including their back faces.
// Evaluate binary coverage at mip 0 so small holes do not disappear when a
// distant reflection's colour cone selects a coarser averaged colour mip.
#ifdef RT_ALPHA_FRAME
bool rt_alpha_partial_crossing=false;
#endif
bool candidate_covered(rayQueryEXT rq) {
    Instance inst=RT_ALPHA_INSTANCES.data[rayQueryGetIntersectionInstanceCustomIndexEXT(rq,false)];
    bool blended=inst.opacity.y==-2.0;
    if(inst.opacity.y<0.0 && !blended) return true;
#ifndef RT_ALPHA_FRAME
    // Opaque receiver/legacy resolve queries do not see raster blend surfaces.
    if(blended) return false;
#endif
    uint prim=uint(rayQueryGetIntersectionPrimitiveIndexEXT(rq,false));
    vec2 bc=rayQueryGetIntersectionBarycentricsEXT(rq,false);
    Indices idx=Indices(inst.idx_addr);Vertices vtx=Vertices(inst.vtx_addr);
    Vertex a=vtx.v[idx.i[prim*3u]],b=vtx.v[idx.i[prim*3u+1u]],c=vtx.v[idx.i[prim*3u+2u]];
    if(blended && inst.opacity.w<0.5) {
        // Match transport's world-space winding, including mirrored transforms.
        mat3 model=mat3(rayQueryGetIntersectionObjectToWorldEXT(rq,false));
        vec3 normal=cross(model*(b.position-a.position),model*(c.position-a.position));
        if(inst.opacity.w<0.0) normal=-normal;
        if(dot(normal,rayQueryGetWorldRayDirectionEXT(rq))>=0.0) return false;
    }
    float w=1.0-bc.x-bc.y;
    vec2 uv=(a.uv*w+b.uv*bc.x+c.uv*bc.y)*inst.albedo_st.xy+inst.albedo_st.zw;
    // Preserve constant attributes exactly. Summing three weighted ones can
    // round below one and randomly open a mask whose alpha equals its cutoff.
    float vertex_alpha=a.color.a+(b.color.a-a.color.a)*bc.x+(c.color.a-a.color.a)*bc.y;
    float alpha=inst.opacity.x*vertex_alpha;
    if(inst.textures.x>=0) alpha*=textureLod(u_tex[nonuniformEXT(inst.textures.x)],uv,0.0).a;
    int opacity_texture=int(inst.opacity.z);
    if(opacity_texture>=0) alpha*=textureLod(u_tex[nonuniformEXT(opacity_texture)],uv,0.0).r;
    if(!blended) return alpha>=inst.opacity.y;
#ifdef RT_ALPHA_FRAME
    // Stochastic coverage estimates alpha*surface + (1-alpha)*behind in one
    // traversal. Independent frame/ray samples accumulate through the existing
    // GI/reflection histories; alpha 0/1 remains exactly open/closed. Do not
    // multiply accepted emission by alpha again.
    alpha=clamp(alpha,0.0,1.0);
    if(alpha<=0.0 || alpha>=1.0) return alpha>=1.0;
    rt_alpha_partial_crossing=true;
    uvec3 o=floatBitsToUint(rayQueryGetWorldRayOriginEXT(rq));
    uvec3 d=floatBitsToUint(rayQueryGetWorldRayDirectionEXT(rq));
    uint seed=uint(RT_ALPHA_FRAME)*747796405u
        ^ uint(rayQueryGetIntersectionInstanceCustomIndexEXT(rq,false))*2891336453u
        ^ prim*277803737u ^ o.x ^ (o.y*1973u) ^ (o.z*9277u)
        ^ (d.x*26699u) ^ (d.y*31847u) ^ (d.z*48371u);
    seed^=seed>>16;seed*=0x7feb352du;seed^=seed>>15;seed*=0x846ca68bu;seed^=seed>>16;
    return float(seed)*(1.0/4294967296.0)<alpha;
#else
    return false;
#endif
}

