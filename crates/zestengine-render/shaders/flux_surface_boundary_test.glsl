#define SURFACE_BINDING 1
#define SURFACE_NODE_COUNT 0u
#define SURFACE_MULTIBOUNCE true
#include "flux_surface.glsl"
layout(local_size_x=1) in;
layout(set=0,binding=0) writeonly buffer Results { vec4 results[]; };
layout(set=0,binding=12) readonly buffer Queries { vec4 queries[]; };
void main() {
    uint i=gl_GlobalInvocationID.x;vec4 query=queries[i];uint info=uint(query.w);
    SurfaceSample hit;hit.valid=true;hit.card=info&65535u;
    SurfaceCard card=surface_cards[hit.card];
    hit.position=query.xyz;hit.geometry_normal=card.normal_texel.xyz;hit.geometry_encoded=surface_encode_geometry(hit.geometry_normal);
    vec2 cp=(card.view_proj*vec4(query.xyz,1)).xy;
    hit.lighting_pixel=vec2(card.rect.xy)+(cp*0.5+0.5)*float(card.rect.z)-0.5;
    bool valid;vec3 value=info<65536u ? surface_radiance(hit,valid) : surface_hit_indirect(hit,vec3(0.25),valid);
    results[i]=vec4(value,valid ? 1.0 : 0.0);
}
