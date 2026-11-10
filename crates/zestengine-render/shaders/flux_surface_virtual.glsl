// Virtual surface topology. Links belong to the same fence-owned card snapshot
// as the material/lighting descriptors;0 means the page is not captured.
uvec2 surface_tile(SurfaceCard card) {
    return uvec2(card.virtual_page.z&65535u,card.virtual_page.z>>16u);
}
bool surface_light_ready(SurfaceCard card,bool hit_indirect) {
#ifdef FLUX_SURFACE_LIGHTING
    const uint direct_ready=2u,indirect_ready=16u;
#else
    const uint direct_ready=4u,indirect_ready=32u;
#endif
    uint required=hit_indirect ? indirect_ready : direct_ready;
    // Fine radiance needs complete secondary transport. A coarse direct-only
    // page retains the established hit-lighting/warmup behaviour.
    if(!hit_indirect && SURFACE_MULTIBOUNCE && card.virtual_page.y>0u) required|=indirect_ready;
    return (card.rect.w&required)==required;
}
bool surface_light_value_valid(vec4 value) {
    return value.a>0.5 && !any(isnan(value)) && !any(isinf(value));
}
// Resolve an adjacent core probe, including diagonals. An unavailable
// intermediate page does not hide a diagonal reachable through the other edge.
bool surface_neighbour_probe(uint original,ivec2 local,bool hit_indirect,out uint resolved,out ivec2 at) {
    SurfaceCard card=surface_cards[original];
    int spacing=surface_lighting_spacing(),gutter=4/spacing;
    resolved=original;at=ivec2(card.rect.xy)/spacing+local;
    if(card.virtual_page.y==0u) return true;
    uvec2 tile=surface_tile(card);int size=int(card.rect.z)/spacing;
    int core=size-2*gutter;
    int edge_x=local.x<gutter && tile.x>0u ? 0 : local.x>=size-gutter && tile.x+1u<(1u<<card.virtual_page.y) ? 1 : -1;
    int edge_y=local.y<gutter && tile.y>0u ? 2 : local.y>=size-gutter && tile.y+1u<(1u<<card.virtual_page.y) ? 3 : -1;
    if(edge_x<0 && edge_y<0) return true;
    uint link=0u;
    if(edge_x>=0) {
        link=card.neighbours[edge_x];
        if(link!=0u && edge_y>=0) link=surface_cards[link-1u].neighbours[edge_y];
    } else link=card.neighbours[edge_y];
    if(link==0u && edge_x>=0 && edge_y>=0) {
        link=card.neighbours[edge_y];
        if(link!=0u) link=surface_cards[link-1u].neighbours[edge_x];
    }
    if(link==0u) return false;
    SurfaceCard neighbour=surface_cards[link-1u];
    if(neighbour.virtual_page.x!=card.virtual_page.x || neighbour.virtual_page.y!=card.virtual_page.y
        || !surface_light_ready(neighbour,hit_indirect)) return false;
    if(edge_x>=0) local.x+=edge_x==0 ? core : -core;
    if(edge_y>=0) local.y+=edge_y==2 ? core : -core;
    resolved=link-1u;at=ivec2(neighbour.rect.xy)/spacing+local;
    return true;
}
float surface_detail_weight(SurfaceCard card,vec2 pixel,bool hit_indirect) {
    if(card.virtual_page.y==0u) return 1.0;
    uvec2 tile=surface_tile(card);uint count=1u<<card.virtual_page.y;
    vec2 uv=(pixel-vec2(card.rect.xy)+0.5)/float(card.rect.z);
    // Logical edges sit four material texels inside the128-pixel capture. Fade
    // to the ancestor over one probe only at edges lacking a usable neighbour.
    vec4 distance=vec4(uv-vec2(0.03125),vec2(0.96875)-uv)*float(card.rect.z)/float(surface_lighting_spacing());
    float weight=1.0;
    for(int edge=0;edge<4;++edge) {
        bool interior=edge==0 ? tile.x>0u : edge==1 ? tile.x+1u<count : edge==2 ? tile.y>0u : tile.y+1u<count;
        if(!interior) continue;
        uint link=card.neighbours[edge];
        if(link!=0u && surface_light_ready(surface_cards[link-1u],hit_indirect)) continue;
        float d=edge==0 ? distance.x : edge==1 ? distance.z : edge==2 ? distance.y : distance.w;
        weight=min(weight,clamp(d,0.0,1.0));
    }
    return weight;
}
vec2 surface_ancestor_pixel(SurfaceCard child,SurfaceCard ancestor,vec2 pixel) {
    uint delta=child.virtual_page.y-ancestor.virtual_page.y;
    float scale=1.0/float(1u<<delta);
    uvec2 relative=surface_tile(child)&uvec2((1u<<delta)-1u);
    // Same padded-chart mapping as CPU history migration; no world-position
    // quantization when a continuous hit crosses a tile or mip boundary.
    vec2 uv=(pixel-vec2(child.rect.xy)+0.5)/float(child.rect.z);
    uv=uv*scale+vec2(0.5)+((vec2(relative)+0.5)*scale-0.5)*0.9375-0.5*scale;
    return vec2(ancestor.rect.xy)+uv*float(ancestor.rect.z)-0.5;
}
