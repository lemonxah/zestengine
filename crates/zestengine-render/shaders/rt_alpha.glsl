// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Only alpha-tested TLAS instances produce triangle candidates. Ordinary
// opaque surfaces remain hardware-committed, including their back faces.
// Evaluate binary coverage at mip 0 so small holes do not disappear when a
// distant reflection's colour cone selects a coarser averaged colour mip.
bool candidate_covered(rayQueryEXT rq) {
    Instance inst=RT_ALPHA_INSTANCES.data[rayQueryGetIntersectionInstanceCustomIndexEXT(rq,false)];
    if(inst.opacity.y<0.0) return true;
    uint prim=uint(rayQueryGetIntersectionPrimitiveIndexEXT(rq,false));
    vec2 bc=rayQueryGetIntersectionBarycentricsEXT(rq,false);
    Indices idx=Indices(inst.idx_addr);Vertices vtx=Vertices(inst.vtx_addr);
    Vertex a=vtx.v[idx.i[prim*3u]],b=vtx.v[idx.i[prim*3u+1u]],c=vtx.v[idx.i[prim*3u+2u]];
    float w=1.0-bc.x-bc.y;
    vec2 uv=(a.uv*w+b.uv*bc.x+c.uv*bc.y)*inst.albedo_st.xy+inst.albedo_st.zw;
    float alpha=inst.opacity.x*(a.color.a*w+b.color.a*bc.x+c.color.a*bc.y);
    if(inst.textures.x>=0) alpha*=textureLod(u_tex[nonuniformEXT(inst.textures.x)],uv,0.0).a;
    int opacity_texture=int(inst.opacity.z);
    if(opacity_texture>=0) alpha*=textureLod(u_tex[nonuniformEXT(opacity_texture)],uv,0.0).r;
    return alpha>=inst.opacity.y;
}

