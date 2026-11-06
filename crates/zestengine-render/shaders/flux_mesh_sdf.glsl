// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Mesh-local detail fields and a stackless world BVH. Ray t stays in world
// units under nonuniform scale/shear; the inverse-matrix norm bounds steps.
struct MeshSdfInstance {
    mat4 inverse; vec4 lo_scale; vec4 hi_dynamic; uvec4 dims_offset;
    vec4 albedo; vec4 emission; vec4 geometry_lo; vec4 geometry_hi;
};
struct MeshSdfNode { vec3 lo; uint instance; vec3 hi; uint escape; };
layout(set=0,binding=MESH_SDF_BINDING) readonly buffer MeshSdfInstances { MeshSdfInstance mesh_sdfs[]; };
layout(set=0,binding=MESH_SDF_BINDING+1) readonly buffer MeshSdfNodes { MeshSdfNode mesh_sdf_nodes[]; };
layout(set=0,binding=MESH_SDF_BINDING+2) uniform sampler3D mesh_sdf_voxels;
const uint MESH_SDF_HIT = 0x80000000u;
const float MESH_DETAIL_DISTANCE = 2.0;
float mesh_sdf_sample(MeshSdfInstance m, vec3 p) {
    vec3 q=clamp(p,m.lo_scale.xyz,m.hi_dynamic.xyz);
    vec3 g=(q-m.lo_scale.xyz)/(m.hi_dynamic.xyz-m.lo_scale.xyz)*vec3(m.dims_offset.xyz-1u);
    uvec3 size=uvec3(textureSize(mesh_sdf_voxels,0)); uint offset=m.dims_offset.w;
    uvec3 base=uvec3(offset&1023u,(offset>>10u)&1023u,offset>>20u);
    float d=textureLod(mesh_sdf_voxels,(vec3(base)+g+0.5)/vec3(size),0.0).r;
    float outside=length(p-q);
    return outside>0.0 ? length(vec2(outside,max(d,0.0))) : d;
}
bool mesh_sdf_box(vec3 ro,vec3 rd,vec3 lo,vec3 hi,float limit,out float enter,out float leave) {
    enter=0.0; leave=limit;
    for(int axis=0;axis<3;++axis) {
        if(abs(rd[axis])<1e-9) { if(ro[axis]<lo[axis] || ro[axis]>hi[axis]) return false; }
        else {
            float a=(lo[axis]-ro[axis])/rd[axis], b=(hi[axis]-ro[axis])/rd[axis];
            enter=max(enter,min(a,b)); leave=min(leave,max(a,b));
        }
    }
    return enter<=leave;
}
float mesh_sdf_bias(vec3 p) {
    float bias=0.005, best=1e20;
    uint ni=0u;
    while(ni<uint(MESH_SDF_NODE_COUNT)) {
        MeshSdfNode node=mesh_sdf_nodes[ni];
        if(any(lessThan(p,node.lo)) || any(greaterThan(p,node.hi))) { ni=node.escape; continue; }
        if(node.instance==0xffffffffu) {++ni;continue;}
        MeshSdfInstance m=mesh_sdfs[node.instance]; ni=node.escape;
        float d=abs(mesh_sdf_sample(m,(m.inverse*vec4(p,1.0)).xyz)*m.lo_scale.w);
        if(d<best) {
            vec3 cell=(m.hi_dynamic.xyz-m.lo_scale.xyz)/vec3(m.dims_offset.xyz-1u);
            bias=max(0.005,min(cell.x,min(cell.y,cell.z))*m.lo_scale.w*0.5);
            best=d;
        }
    }
    return bias;
}
vec3 mesh_sdf_normal(uint who,vec3 p) {
    MeshSdfInstance m=mesh_sdfs[who & ~MESH_SDF_HIT];
    vec3 q=(m.inverse*vec4(p,1.0)).xyz;
    vec3 cell=(m.hi_dynamic.xyz-m.lo_scale.xyz)/vec3(m.dims_offset.xyz-1u);
    float e=max(1e-5,min(cell.x,min(cell.y,cell.z))*0.5);
    vec3 g=vec3(mesh_sdf_sample(m,q+vec3(e,0,0))-mesh_sdf_sample(m,q-vec3(e,0,0)),
                mesh_sdf_sample(m,q+vec3(0,e,0))-mesh_sdf_sample(m,q-vec3(0,e,0)),
                mesh_sdf_sample(m,q+vec3(0,0,e))-mesh_sdf_sample(m,q-vec3(0,0,e)));
    g=transpose(mat3(m.inverse))*g;
    return dot(g,g)>1e-16 ? normalize(g) : vec3(0,1,0);
}
bool mesh_sdf_trace_interval(vec3 ro,vec3 rd,float min_dist,float max_dist,bool far_field,out float hit_t,out uint who) {
    hit_t=max_dist; who=0u; bool hit=false;
    uint ni=0u;
    while(ni<uint(MESH_SDF_NODE_COUNT)) {
        MeshSdfNode node=mesh_sdf_nodes[ni]; float enter,leave;
        if(!mesh_sdf_box(ro,rd,node.lo,node.hi,hit_t,enter,leave) || leave<min_dist) { ni=node.escape; continue; }
        if(node.instance==0xffffffffu) {++ni;continue;}
        uint index=node.instance; MeshSdfInstance m=mesh_sdfs[index]; ni=node.escape;
        // Full-range dynamic/emissive meshes were already tested by the near
        // pass. A coarse-field candidate only needs static-mesh refinement.
        if(far_field && m.hi_dynamic.w>0.5) continue;
        vec3 local_ro=(m.inverse*vec4(ro,1.0)).xyz, local_rd=mat3(m.inverse)*rd;
        float limit=far_field || m.hi_dynamic.w>0.5 ? hit_t : min(hit_t,MESH_DETAIL_DISTANCE);
        vec3 padding=vec3(0.002/max(m.lo_scale.w,1e-8));
        if(!mesh_sdf_box(local_ro,local_rd,m.geometry_lo.xyz-padding,m.geometry_hi.xyz+padding,limit,enter,leave)) continue;
        float t=max(max(enter,0.002),min_dist);
        if(t>leave) continue;
        // Along this fixed ray, local direction length is a tighter Lipschitz
        // bound than the transform's worst-case scale (especially long walls).
        float ray_scale=1.0/max(length(local_rd),1e-8);
        // Transform the ray into atlas coordinates once per candidate mesh.
        // Repeated samples only advance/clamp this affine ray, rather than
        // rebuilding the volume-to-atlas transform at every sphere step.
        vec3 atlas_size=vec3(textureSize(mesh_sdf_voxels,0));
        uint packed=m.dims_offset.w;
        vec3 base=vec3(packed&1023u,(packed>>10u)&1023u,packed>>20u);
        vec3 uv_lo=(base+0.5)/atlas_size;
        vec3 uv_hi=(base+vec3(m.dims_offset.xyz)-0.5)/atlas_size;
        vec3 uv_scale=vec3(m.dims_offset.xyz-1u)/((m.hi_dynamic.xyz-m.lo_scale.xyz)*atlas_size);
        vec3 uv_origin=uv_lo+(local_ro-m.lo_scale.xyz)*uv_scale;
        vec3 uv_direction=local_rd*uv_scale;
        for(uint step=0u;step<80u && t<=leave;++step) {
            // Open meshes can have negative distance throughout empty space
            // on their back side. Intersect the surface from either side,
            // never the box where we first enter that negative half-space.
            float signed_distance=textureLod(mesh_sdf_voxels,clamp(uv_origin+uv_direction*t,uv_lo,uv_hi),0.0).r;
            vec3 p=local_ro+local_rd*t;
            float outside=length(max(max(m.lo_scale.xyz-p,p-m.hi_dynamic.xyz),vec3(0.0)));
            float d=(outside>0.0 ? length(vec2(outside,max(signed_distance,0.0))) : abs(signed_distance))*ray_scale;
            if(d<0.002) {hit=true;hit_t=t;who=index|MESH_SDF_HIT;break;}
            t+=max(d*0.9,0.002);
        }
    }
    return hit;
}
bool mesh_sdf_trace_range(vec3 ro,vec3 rd,float min_dist,float max_dist,out float hit_t,out uint who) {
    return mesh_sdf_trace_interval(ro,rd,min_dist,max_dist,false,hit_t,who);
}

// Preserve existing callers while allowing a cache miss to resume the same
// ray without moving its mesh-detail interval farther into the scene.
bool mesh_sdf_trace(vec3 ro,vec3 rd,float max_dist,out float hit_t,out uint who) {
    return mesh_sdf_trace_range(ro,rd,0.0,max_dist,hit_t,who);
}
