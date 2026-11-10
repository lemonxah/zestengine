// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Coverage-only lookup for software traversal. Do not pull material shading,
// lighting filtering, or feedback writes into the nested distance-field march.
float surface_mesh_opacity(vec3 p,vec3 n,uint index) {
    bool represented=false;
    SurfaceNode mapping=surface_nodes[surface_tree_info.x+surface_tree_info.y];
    SurfaceNode source=surface_nodes[mapping.left+index];
    // Binary masks occlude from either side, even when imported shading
    // normals disagree with triangle winding (the SDF follows winding).
    // Blend retains its directional coverage/emission policy.
    bool masked=(uint(source.bmin.x+0.5)&16u)!=0u;
    if(source.left==0u) {
        // Until capture residency arrives, retain the known scalar policy.
        // Masked surfaces use a cutoff; only Blend uses fractional coverage.
        if((uint(source.bmin.x+0.5)&16u)!=0u) return source.bmin.y>=source.bmin.z ? 1.0 : 0.0;
        return surface_source_opacity(index);
    }
    uint node=source.left-1u,end=surface_nodes[node].right;
    float best=1e30;uint level=0u,chart=0u;ivec2 chosen=ivec2(0);
    while(node<end) {
        SurfaceNode branch=surface_nodes[node];
        if(any(lessThan(p,branch.bmin-0.01)) || any(greaterThan(p,branch.bmax+0.01))) {node=branch.right;continue;}
        if((branch.left&0x80000000u)==0u) {++node;continue;}
        SurfaceCard card=surface_cards[branch.left&0x7fffffffu];node=branch.right;
        float facing=dot(card.normal_texel.xyz,n);
        if(card.rect.w==0u || (masked ? abs(facing) : facing)<0.25) continue;
        vec3 q=(card.view_proj*vec4(p,1.0)).xyz;
        float depth_error=0.01+0.01*length(vec3(card.view_proj[0][2],card.view_proj[1][2],card.view_proj[2][2]));
        if(any(greaterThan(abs(q.xy),vec2(card.virtual_page.y>0u ? 0.937501 : 1.0))) || q.z < -depth_error || q.z>1.0+depth_error) continue;
        ivec2 base=ivec2(floor((q.xy*0.5+0.5)*float(card.rect.z)-0.5));
        for(int corner=0;corner<4;++corner) {
            ivec2 local=base+ivec2(corner&1,corner>>1);
            if(any(lessThan(local,ivec2(0))) || any(greaterThanEqual(local,ivec2(card.rect.z)))) continue;
            ivec2 at=ivec2(card.rect.xy)+local;
            vec4 position=texelFetch(surface_position,at,0);if(position.w<0.5) continue;
            vec3 delta=surface_world_position(card,position.xyz)-p;
            float distance=dot(delta,delta),tolerance=card.normal_texel.w*2.0+0.01;
            if(distance>tolerance*tolerance) continue;
            bool same=represented && chart!=0u && chart==card.virtual_page.x;
            if(same && card.virtual_page.y<level) continue;
            if(distance>=best && (!same || card.virtual_page.y==level)) continue;
            vec3 normal=surface_geometry_normal(position.w,texelFetch(surface_normal,at,0).xyz);
            float alignment=dot(surface_world_normal(card,normal),n);
            if((masked ? abs(alignment) : alignment)<0.5) continue;
            best=distance;chosen=at;represented=true;level=card.virtual_page.y;chart=card.virtual_page.x;
        }
    }
    // A captured source with no corresponding surface must not turn a signed
    // distance discontinuity between open layers into an emitting surface.
    if(!represented) return 0.0;
    float metadata=texelFetch(surface_emission,chosen,0).a;
    return metadata<0.0 ? clamp(-metadata-(metadata<=-3.0 ? 3.0 : 1.0),0.0,1.0) : 1.0;
}
