// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Cluster geometry pool, shared by cluster_cull.comp and the clustered
// variant of standard.vert (vertex pulling). The includer defines
// CLUSTER_SET (the descriptor set holding the pool). Layouts mirror the
// repr(C) structs in cluster.rs (checked there); the cooked source is
// zestengine-assets cluster_dag.rs.
//
// Bounds discipline: every index into a pool buffer is checked against the
// mesh's counts and the buffer's own length before it is used, so a bad cooked
// file or a stale instance can only drop geometry, never read or write out of
// range.

#include "cluster_view.glsl"

const uint CLUSTER_MAX_TRIS = 128u;
// Vertices drawn per cluster instance (128 triangles * 3).
const uint CLUSTER_DRAW_VERTICES = 384u;
const uint BVH_EMPTY = 0xFFFFFFFFu;
const uint BVH_LEAF = 0x80000000u;
// Clusters per BVH leaf run, at most (the cooked builder's cap; a corrupt count cannot make a thread loop long).
const uint BVH_RUN_MAX = 8u;

struct Cluster {
    vec4 bounds;       // culling sphere (object space)
    vec4 lod_sphere;
    vec4 parent_sphere;
    float lod_error;
    float parent_error; // FLT_MAX for a root
    uint cone;          // axis snorm8 x3, cutoff unorm8 (255 = never cull)
    uint vertex_offset;
    uint triangle_offset; // bytes, relative to the mesh's triangle bytes
    uint counts;          // vertex_count | triangle_count << 16
    uint bone_offset;
    uint bone_flags;
    uint morph_offset;
    uint morph_count;
    uint section;
    uint level;
};

struct BvhChild {
    vec4 bounds;
    vec4 lod_sphere;
    float max_parent_error;
    uint child; // BVH_EMPTY, BVH_LEAF | first cluster, or a node index
    uint count; // clusters in a leaf run
    uint pad;
};

struct BvhNode {
    BvhChild c[4];
};

// One DAG in the pool: where its parts start and how many there are.
struct MeshRec {
    uvec4 a;   // node_base, node_count, cluster_base, cluster_count
    uvec4 b;   // vertex_base, vertex_count, tri_base (bytes), tri_len (bytes)
    vec4 grid; // quantisation: mesh_min xyz, step
    uvec4 c;   // skin_base, bone_base, morph_base (pool entries), w 1 = skin data present
};

// A drawn mesh instance.
struct Inst {
    mat4 model;
    vec4 color;
    vec4 param;
    vec4 sphere; // object-space bounding sphere
    uvec4 ref;   // x mesh record, y material batch, z skin instance + 1 (0 = static), w pose margin (float bits)
    vec4 eye;    // xyz camera in object space, w 1 = the cone test is valid
    vec4 extra;  // x LOD error scale (budget / fair share; <= 0 or 1 = none)
};

layout(std430, set = CLUSTER_SET, binding = 0) readonly buffer ClusterPool { Cluster clusters[]; };
layout(std430, set = CLUSTER_SET, binding = 1) readonly buffer VertexPool { uint verts[]; };
layout(std430, set = CLUSTER_SET, binding = 2) readonly buffer TrianglePool { uint tris[]; };
layout(std430, set = CLUSTER_SET, binding = 4) readonly buffer MeshTable { MeshRec meshes[]; };
layout(std430, set = CLUSTER_SET, binding = 5) readonly buffer Instances { Inst insts[]; };
// Page table of streamed meshes (GEO-6), one entry per page: x vertex base (pool
// vertex index) or PAGE_NONE when the page is not resident, y triangle byte base
// (in the triangle pool), z vertices and w triangle bytes of the page.
layout(std430, set = CLUSTER_SET, binding = 19) readonly buffer PageTable { uvec4 pages[]; };

const uint PAGE_NONE = 0xFFFFFFFFu;

// A streamed mesh keeps its clusters and BVH resident; its vertices and
// triangles live in pages, found through the page table. For these meshes
// `MeshRec.c` = (table base, page count, -, flags) with flag bit 1 (value 2) set
// (bit 0 is "skinned"), a cluster's `morph_offset` is its page and `morph_count`
// the page of its finer children (PAGE_NONE: none), and its vertex / triangle
// offsets are relative to the page.
//
// A streamed SKINNED mesh (flags 3) keeps the skin data of a cluster in its page
// too. The page's triangle block (the page entry's y byte base, w byte length) is
//   [skin records, 8 bytes per page vertex: uvec2 (bone slots, unorm8 weights)]
//   [triangle bytes: `triangle_offset` is relative to the end of the skin records]
//   [aux per cluster, at word `bone_offset` of the block: morph count, ceil(bones/2)
//    words of u16 joint ids, then the sparse 16-byte blendshape entries]
// (assets cluster_pages.rs is the writer; no separate skin / bone / morph pool is
// read for these meshes).
bool mesh_streamed(MeshRec m) {
    return (m.c.w & 2u) != 0u;
}

bool mesh_skin_paged(MeshRec m) {
    return (m.c.w & 3u) == 3u;
}

// Bytes of the skin records at the start of a skinned page's triangle block.
uint page_skin_bytes(MeshRec m, uvec4 e) {
    return mesh_skin_paged(m) ? e.z * 8u : 0u;
}

// Page `page` of streamed mesh `m`; false when it is out of range or not resident.
bool page_entry(MeshRec m, uint page, out uvec4 e) {
    e = uvec4(0u);
    if (page >= m.c.y || m.c.x + page >= uint(pages.length())) {
        return false;
    }
    e = pages[m.c.x + page];
    return e.x != PAGE_NONE;
}

float inst_lod_scale(Inst it) {
    return it.extra.x > 1.0 ? it.extra.x : 1.0;
}

// Object-space distance the pose can move a bind-pose vertex (skinned instances;
// 0 for static ones). The cull grows every bound by it, so culling on bind-pose
// bounds stays conservative for any pose.
float inst_margin(Inst it) {
    return it.ref.z != 0u ? max(uintBitsToFloat(it.ref.w), 0.0) : 0.0;
}

uint tri_byte(uint i) {
    return (tris[i >> 2] >> ((i & 3u) * 8u)) & 255u;
}

float snorm16(uint w, uint hi) {
    return clamp(float(bitfieldExtract(int(w), int(hi * 16u), 16)) / 32767.0, -1.0, 1.0);
}

// Inverse of the octahedral encoding (assets cluster_dag::oct_decode).
vec3 oct_decode(vec2 e) {
    vec3 n = vec3(e, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) {
        vec2 s = vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
        n.xy = (vec2(1.0) - abs(n.yx)) * s;
    }
    return n / max(length(n), 1e-20);
}

// Position of a quantised vertex: mesh_min + q * step (step is a power of two,
// so the product is exact and a fused multiply-add gives the same bits).
vec3 decode_position(MeshRec m, uint w0, uint w1) {
    vec3 q = vec3(float(w0 & 0xFFFFu), float(w0 >> 16), float(w1 & 0xFFFFu));
    return m.grid.xyz + q * m.grid.w;
}

// Debug colour of a cluster: hue from its index, brightness from its DAG level.
vec3 cluster_debug_color(uvec2 tag) {
    uint h = tag.x * 2654435761u;
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    vec3 hue = 0.5 + 0.5 * cos(6.2831853 * (float(h & 1023u) / 1024.0 + vec3(0.0, 0.33, 0.67)));
    return hue * (0.45 + 0.55 / (1.0 + 0.35 * float(tag.y)));
}

// Cluster-local vertex index of draw vertex `vid` (the triangle byte cluster_pull
// reads); 0xFFFFFFFF = degenerate. Same checks as cluster_pull.
uint cluster_local_vertex(uint inst, uint cluster, uint vid) {
    if (inst >= uint(insts.length())) {
        return 0xFFFFFFFFu;
    }
    uint mi = insts[inst].ref.x;
    if (mi >= uint(meshes.length())) {
        return 0xFFFFFFFFu;
    }
    MeshRec m = meshes[mi];
    if (cluster >= m.a.w || m.a.z + cluster >= uint(clusters.length())) {
        return 0xFFFFFFFFu;
    }
    Cluster c = clusters[m.a.z + cluster];
    uint tbase = m.b.z;
    uint tlimit = m.b.w;
    if (mesh_streamed(m)) {
        uvec4 pe;
        if (!page_entry(m, c.morph_offset, pe)) {
            return 0xFFFFFFFFu;
        }
        uint skin = page_skin_bytes(m, pe);
        tbase = pe.y + skin;
        tlimit = pe.w > skin ? pe.w - skin : 0u;
    }
    uint tri = vid / 3u;
    if (tri >= (c.counts >> 16) || tri >= CLUSTER_MAX_TRIS) {
        return 0xFFFFFFFFu;
    }
    uint rel = c.triangle_offset + tri * 3u + (vid - tri * 3u);
    if (rel >= tlimit || (tbase + rel) >> 2 >= uint(tris.length())) {
        return 0xFFFFFFFFu;
    }
    uint local = tri_byte(tbase + rel);
    return local >= (c.counts & 0xFFFFu) ? 0xFFFFFFFFu : local;
}

// Fetch one vertex of one cluster of one instance: `vid` is the 0..383 vertex
// index of the draw (triangle = vid / 3). False = degenerate (past the
// cluster's triangle count, or an index out of range): the caller emits a
// collapsed vertex.
// `col` is the vertex colour (RGBA8, white for a mesh without colours): the 7th word of the
// 28-byte pool vertex, so the stages that ignore it (depth prepass, shadows) never read it.
bool cluster_pull(uint inst, uint cluster, uint vid, out vec3 pos, out vec3 nrm, out vec4 tan, out vec2 uv, out vec4 col, out uvec2 tag) {
    pos = vec3(0.0);
    col = vec4(1.0);
    nrm = vec3(0.0, 1.0, 0.0);
    tan = vec4(1.0, 0.0, 0.0, 1.0);
    uv = vec2(0.0);
    tag = uvec2(0u);
    if (inst >= uint(insts.length())) {
        return false;
    }
    uint mi = insts[inst].ref.x;
    if (mi >= uint(meshes.length())) {
        return false;
    }
    MeshRec m = meshes[mi];
    if (cluster >= m.a.w || m.a.z + cluster >= uint(clusters.length())) {
        return false;
    }
    Cluster c = clusters[m.a.z + cluster];
    tag = uvec2(cluster, c.level);
    uint vbase = m.b.x;
    uint vlimit = m.b.y;
    uint tbase = m.b.z;
    uint tlimit = m.b.w;
    if (mesh_streamed(m)) {
        uvec4 pe;
        if (!page_entry(m, c.morph_offset, pe)) {
            return false;
        }
        uint skin = page_skin_bytes(m, pe);
        vbase = pe.x;
        tbase = pe.y + skin;
        vlimit = pe.z;
        tlimit = pe.w > skin ? pe.w - skin : 0u;
    }
    uint tri = vid / 3u;
    if (tri >= (c.counts >> 16) || tri >= CLUSTER_MAX_TRIS) {
        return false;
    }
    uint rel = c.triangle_offset + tri * 3u + (vid - tri * 3u);
    if (rel >= tlimit || (tbase + rel) >> 2 >= uint(tris.length())) {
        return false;
    }
    uint local = tri_byte(tbase + rel);
    if (local >= (c.counts & 0xFFFFu) || c.vertex_offset + local >= vlimit) {
        return false;
    }
    uint vi = (vbase + c.vertex_offset + local) * 7u;
    if (vi + 6u >= uint(verts.length())) {
        return false;
    }
    uint w0 = verts[vi];
    uint w1 = verts[vi + 1u];
    uint w2 = verts[vi + 2u];
    uint w3 = verts[vi + 3u];
    pos = decode_position(m, w0, w1);
    nrm = oct_decode(vec2(snorm16(w2, 0u), snorm16(w2, 1u)));
    tan = vec4(oct_decode(vec2(snorm16(w3, 0u), snorm16(w3, 1u))), (w1 >> 16 & 1u) != 0u ? -1.0 : 1.0);
    uv = vec2(uintBitsToFloat(verts[vi + 4u]), uintBitsToFloat(verts[vi + 5u]));
    col = unpackUnorm4x8(verts[vi + 6u]);
    return true;
}

float max_scale(mat4 m) {
    return max(max(length(m[0].xyz), length(m[1].xyz)), length(m[2].xyz));
}

// "Accurate enough": the object-space error `err` of a sphere projects to at
// most the target number of pixels. err * s * ppu / (distance - radius) <= px,
// cross-multiplied so a sphere around the eye is only accurate at zero error.
// Monotone in (err, sphere), as ClusterDag::select_* requires.
// `k` scales the pixel target per instance (the budget's coarsening; >= 1).
// An orthographic shadow view (v.lod.x > 0) has no distance: its texel size is
// constant, so `lod.x` stands in for the distance (1 unit; `lod.y` = texels per
// world unit).
bool lod_ok(View v, mat4 model, float s, float err, vec4 sph, float k) {
    if (v.lod.x > 0.0) {
        return err * s * v.lod.y <= v.lod.w * k * v.lod.x;
    }
    vec3 c = (model * vec4(sph.xyz, 1.0)).xyz;
    float d = max(distance(c, v.cam.xyz) - sph.w * s, 0.0);
    return err * s * v.lod.y <= v.lod.w * k * d;
}

