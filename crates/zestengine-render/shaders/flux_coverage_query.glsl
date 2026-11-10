layout(local_size_x=1) in;
#include "flux_common.glsl"
const uint MAX_STEPS=96u;
#ifdef FLUX_GDF_CLIPMAP_BINDING
layout(set=0,binding=0) uniform sampler3D u_dist[4];
layout(set=0,binding=1) uniform usampler3D u_index[4];
#else
layout(set=0,binding=0) uniform sampler3D u_dist;
layout(set=0,binding=1) uniform usampler3D u_index;
#endif
layout(set=0,binding=7) uniform Params {vec4 gdf_min;vec4 gdf_max;vec4 options;} pc;
#define MESH_SDF_BINDING 2
#define MESH_SDF_NODE_COUNT pc.gdf_max.w
#include "flux_mesh_sdf.glsl"
#define FLUX_GDF_REFINABLE (pc.options.x>0.5)
#include "flux_software_geometry.glsl"
struct Ray {vec4 origin;vec4 direction;};
layout(set=0,binding=5) readonly buffer Rays {Ray rays[];};
layout(set=0,binding=6) writeonly buffer Results {vec4 results[];};
void main() {
    uint i=gl_GlobalInvocationID.x;Ray r=rays[i];vec3 p;uint who;
    if(pc.options.y>0.5) {results[i]=vec4(scene_normal(r.origin.xyz,0.02),0.0);return;}
    if(pc.options.z>0.5) {
        int level=flux_field_level(r.origin.xyz);
        results[i]=vec4(scene_distance(r.origin.xyz),float(scene_index(r.origin.xyz)),float(level),0.0);return;
    }
    bool found=march_range(r.origin.xyz,r.direction.xyz,0.02,r.origin.w,r.direction.w,p,who);
    results[i]=vec4(found ? length(p-r.origin.xyz) : -1.0,float(who & ~MESH_SDF_HIT),found && (who&MESH_SDF_HIT)!=0u ? 1.0 : 0.0,
        occluded(r.origin.xyz,r.direction.xyz,r.direction.w,0.02) ? 1.0 : 0.0);
}
