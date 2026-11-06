// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Shared scalar RT material ABI: 288 bytes, matching rt_reflect::GpuInstance.
struct Vertex {
    vec3 position; vec3 normal; vec2 uv; vec4 color; vec4 tangent; vec2 uv1;
    // Skinning lanes: unused here, but part of the 104-byte engine Vertex, so the
    // array stride matches (without them every fetch past vertex 0 was wrong).
    uvec4 joints; vec4 weights;
};
layout(buffer_reference, scalar) readonly buffer Vertices { Vertex v[]; };
layout(buffer_reference, scalar) readonly buffer Indices { uint i[]; };

// Mirrors GpuInstance in rt_reflect.rs (288 bytes).
struct Instance {
    uint64_t vtx_addr; uint64_t idx_addr;
    vec4 albedo;       // base x mean albedo (untextured fallback)
    vec4 emission;     // emission x emission-map mean
    vec4 base;         // base colour factor (x sampled albedo), w water slot + 1
    vec4 emission_raw; // emission colour x intensity (x sampled emission)
    ivec4 textures;    // albedo, normal, orm, emission (-1 = none)
    vec4 albedo_st; vec4 normal_st; vec4 orm_st; vec4 emission_st;
    vec4 params;       // x metallic, y roughness, z normal strength
    vec4 opacity;      // x base alpha, y cutoff (-1 opaque), z opacity texture index
    ivec4 extra_textures; // split roughness, metallic, emission mask, reserved
    ivec4 detail_textures; // albedo, normal, ORM, mask
    vec4 detail; vec4 detail_st; vec4 orm_invert; vec4 emission_mask_st;
};
