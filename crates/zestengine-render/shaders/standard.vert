// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
#ifdef MULTIVIEW
#extension GL_EXT_multiview : require
#endif
// zestengine standard shader, vertex stage (phase 1)

#ifdef CLUSTER
// Clustered variant (GEO-4): no vertex buffers. The vertex is pulled from the
// cluster pool (cluster_common.glsl) by (instance, cluster) from the visible
// list and the draw's vertex index; the same decode feeds the depth prepass and
// the main pass, so their positions are bit-identical.
vec3 a_position;
vec3 a_normal;
vec2 a_uv;
vec4 a_color;
vec4 a_tangent;
vec2 a_uv1;
vec3 a_prev_position;
#else
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;
layout(location = 3) in vec4 a_color;
layout(location = 4) in vec4 a_tangent; // xyz tangent, w handedness
layout(location = 5) in vec2 a_uv1;     // lightmap UVs
// Previous-frame skinned position (binding 2): static meshes bind the same
// buffer as `a_position`, so `a_prev_position == a_position` there.
layout(location = 12) in vec3 a_prev_position;

#endif
// Water surfaces displace their grid with the Gerstner waves here.
// (Hooked builds, ZE_HOOK, move the two ids out of the custom keyword range 0..31.)
#ifndef ZE_ID_WATER
#define ZE_ID_WATER 4
#endif
#ifndef ZE_ID_PERINST
#define ZE_ID_PERINST 6
#endif
layout(constant_id = ZE_ID_WATER) const bool FEAT_WATER = false;
// Per-instance material properties (base colour + one parameter vec4).
layout(constant_id = ZE_ID_PERINST) const bool FEAT_PER_INSTANCE = false;

#include "standard_common.glsl"

// Water: the interactive ripple field (r height, gb gradient), material slot
// 15, with mips for reading it band-limited.
layout(set = 1, binding = 16) uniform sampler2D t_water_ripple;
// Water: the depth of water over the ground (r, metres), baked over the
// body's local XZ -extra.zw..extra.zw; material slot 14.
layout(set = 1, binding = 15) uniform sampler2D t_water_shore;

// Share of the waves left at local `xz`: they calm over water shallower
// than the body's shore depth (zestengine_core::Water::shore_damping).
float water_shore(int slot, vec2 xz) {
    vec4 e = frame.water_bodies[slot].extra;
    if (e.y <= 0.0 || e.z <= 0.0) {
        return 1.0;
    }
    vec2 uv = xz / (2.0 * e.zw) + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) {
        return 1.0;
    }
    return smoothstep(0.0, e.y, textureLod(t_water_shore, uv, 0.0).r);
}

// Per-instance model matrix (instance-rate vertex stream), so draws of one
// mesh + material batch into a single instanced draw (pc.model is unused).
#ifdef CLUSTER
vec4 a_model0;
vec4 a_model1;
vec4 a_model2;
vec4 a_model3;
vec4 a_instance_color;
vec4 a_instance_param;
#else
layout(location = 8) in vec4 a_model0;
layout(location = 9) in vec4 a_model1;
layout(location = 10) in vec4 a_model2;
layout(location = 11) in vec4 a_model3;
layout(location = 13) in vec4 a_instance_color;
layout(location = 14) in vec4 a_instance_param;

#endif

layout(push_constant) uniform Push {
    mat4 model;
    vec4 base_color;
    vec4 emission; // rgb * intensity
    vec4 params0;  // metallic, roughness, toon_steps, pbr_toon_blend
    vec4 params1;  // alpha_cutoff, normal_strength, occlusion_strength, _
} pc;

#ifdef CLUSTER
#define CLUSTER_SET 2
#include "cluster_common.glsl"
// (instance, cluster) pairs of this batch's region (cluster_cull.comp).
layout(std430, set = 2, binding = 11) readonly buffer Visible { uvec2 visible[]; };
// Skinned instances (GEO-5): the posed vertices of the visible clusters, written
// once per frame by cluster_skin.comp and shared by every pass, and each
// visible entry's base in that pool (0xFFFFFFFF = dropped).
struct SkinVert {
    vec4 pos;
    vec4 nrm;
    vec4 tan;
    vec4 prev;
};
layout(std430, set = 2, binding = 17) readonly buffer SkinPool { SkinVert skinned[]; };
layout(std430, set = 2, binding = 18) readonly buffer SkinSlots { uint slots[]; };
// The culled views (the camera's is [0]): only the Cluster LOD view reads it, to
// tint the clusters that stand in for finer detail whose page is not resident.
layout(std430, set = 2, binding = 7) readonly buffer Views { View views[]; };

// A drawn cluster of a streamed mesh that is too coarse for the camera and whose
// finer children's page is missing (the cull's `lacking` case: it asks for that page).
bool cluster_standin(uint inst, uint cluster) {
    if (inst >= uint(insts.length()) || views.length() == 0) {
        return false;
    }
    Inst it = insts[inst];
    if (it.ref.x >= uint(meshes.length())) {
        return false;
    }
    MeshRec m = meshes[it.ref.x];
    if (!mesh_streamed(m) || cluster >= m.a.w || m.a.z + cluster >= uint(clusters.length())) {
        return false;
    }
    Cluster c = clusters[m.a.z + cluster];
    uvec4 ce;
    if (c.morph_count == PAGE_NONE || page_entry(m, c.morph_count, ce)) {
        return false;
    }
    return !lod_ok(views[0], it.model, max_scale(it.model), c.lod_error, c.lod_sphere, inst_lod_scale(it));
}
#endif

layout(location = 0) out vec3 v_world_pos;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec2 v_uv;
layout(location = 3) out vec4 v_color;
layout(location = 4) out vec4 v_tangent;
layout(location = 5) out vec2 v_uv1;
// Baked-lightmap layer (-1 = none): the instance's own when it carries one
// (GPU-culled batches pack layer + 1 into the model matrix's col0.w, so
// lightmapped statics share a batch), else the draw's push lane.
layout(location = 6) flat out float v_lightmap;
// Current/previous-frame clip positions (the motion-vector source; the
// fragment divides by w and writes the screen-space difference).
layout(location = 7) out vec4 v_cur_clip;
layout(location = 8) out vec4 v_prev_clip;
// Per-instance material properties (read by the fragment only when the
// material opts in via FEAT_PER_INSTANCE).
layout(location = 9) out vec4 v_instance_color;
layout(location = 10) out vec4 v_instance_param;
#ifdef ZE_HOOK
// A custom shader's `zestengine_vertex(inout VertexData v)` (docs/shaders.md,
// "Vertex hook"): spliced in by the render crate when it builds the hooked
// vertex module for that shader. The depth prepass, the main pass and the XR
// pass all use this module, so their positions stay bit-identical.
#include "zestengine_hook.glsl"
#endif

// Bit-identical positions in the depth prepass and the main pass (the main
// pass depth-tests EQUAL against the prepass).
invariant gl_Position;

void main() {
#ifdef CLUSTER
    // pc.model[1][0] carries the batch's first entry in the visible list
    // (bits); gl_InstanceIndex counts within the batch (firstInstance = 0).
    uint entry = floatBitsToUint(pc.model[1][0]) + uint(gl_InstanceIndex);
    uvec2 pick = entry < uint(visible.length()) ? visible[entry] : uvec2(0xFFFFFFFFu);
    uvec2 tag = uvec2(0u);
    bool live = pick.x < uint(insts.length()) &&
        cluster_pull(pick.x, pick.y, uint(gl_VertexIndex), a_position, a_normal, a_tangent, a_uv, a_color, tag);
    a_prev_position = a_position;
    if (live && insts[pick.x].ref.z != 0u) {
        // Skinned instance: the pose comes from the per-frame pool, by the
        // vertex's cluster-local index (the triangle byte cluster_pull used).
        uint base = entry < uint(slots.length()) ? slots[entry] : 0xFFFFFFFFu;
        uint local = cluster_local_vertex(pick.x, pick.y, uint(gl_VertexIndex));
        if (base == 0xFFFFFFFFu || local == 0xFFFFFFFFu || base + local >= uint(skinned.length())) {
            live = false;
        } else {
            SkinVert sv = skinned[base + local];
            a_position = sv.pos.xyz;
            a_normal = sv.nrm.xyz;
            a_tangent = vec4(sv.tan.xyz, a_tangent.w);
            a_prev_position = sv.prev.xyz;
        }
    }
    if (!live) {
        // Collapsed vertex: the triangle has zero area and is never rasterised.
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        v_world_pos = vec3(0.0);
        v_normal = vec3(0.0, 1.0, 0.0);
        v_uv = vec2(0.0);
        v_color = vec4(1.0);
        v_tangent = vec4(1.0, 0.0, 0.0, 1.0);
        v_uv1 = vec2(0.0);
        v_lightmap = -1.0;
        v_cur_clip = gl_Position;
        v_prev_clip = gl_Position;
        v_instance_color = vec4(1.0);
        v_instance_param = vec4(0.0);
        return;
    }
    // Zestlet viewport modes (frame.debug.y 8 LOD, 9 zestlets, 10 coverage; 11 colours per triangle
    // in the fragment): the tint rides in v_color, and v_uv1 = (1e6, ..) marks the draw as a zestlet
    // for the fragment (ordinary draws carry real UVs there). Normal modes skip all of it.
    a_uv1 = vec2(0.0);
    int zmode = int(frame.debug.y + 0.5);
    if (zmode >= 8) {
        vec3 tint = vec3(0.1, 0.8, 0.25);
        if (zmode == 8) {
            tint = zestlet_level_colour(tag.y);
            // Streaming (GEO-6): zestlets standing in for pages that have not arrived glow red-orange.
            if (cluster_standin(pick.x, pick.y)) {
                tint = mix(tint, vec3(1.0, 0.18, 0.05), 0.75);
            }
        } else if (zmode == 9 || zmode == 11) {
            // (11 keys the per-triangle colour by this zestlet too: gl_PrimitiveID restarts per drawn cluster.)
            tint = zestlet_hash_colour(tag.x * 2246822519u ^ pick.x * 2654435761u);
        }
        a_color = vec4(tint, 1.0);
        a_uv1 = vec2(1.0e6);
    }
    Inst pulled = insts[pick.x];
    a_model0 = pulled.model[0];
    a_model1 = pulled.model[1];
    a_model2 = pulled.model[2];
    a_model3 = pulled.model[3];
    a_instance_color = pulled.color;
    a_instance_param = pulled.param;
#endif
    mat4 model = mat4(vec4(a_model0.xyz, 0.0), a_model1, a_model2, a_model3);
    v_lightmap = a_model0.w > 0.5 ? a_model0.w - 1.0 : pc.params1.w;
    vec3 position = a_position;
    vec3 normal = a_normal;
    vec4 tangent = a_tangent;
    vec2 uv1 = a_uv1;
    vec4 color = a_color;
    vec2 uv0 = a_uv;
    vec3 hook_world_offset = vec3(0.0);
    vec3 hook_obj_offset = vec3(0.0);
    vec2 water_rest = vec2(0.0);
    if (FEAT_WATER) {
        // Rest point in the water's local space (an ocean grid is drawn
        // shifted to follow the camera, so it comes through the model
        // matrix), displaced by the waves; they calm toward a river's banks
        // (a_color.b = edge). a_color.rg carries the local flow (m/s / 10 +
        // 0.5), a_uv1.x the grid's spacing here: the displacement holds no
        // detail finer than it (the pixel shader adds that to the normal).
        int slot = clamp(int(fx.water.x + 0.5), 0, MAX_WATER_BODIES - 1);
        water_rest = (frame.water_bodies[slot].to_local * (model * vec4(a_position, 1.0))).xz;
        float spacing = max(a_uv1.x, 1e-3);
        vec3 d;
        float crest;
        float lost;
        water_waves(slot, water_rest, frame.water.x, spacing, d, normal, crest, lost);
        // Waves calm over the shallows near a shore (a_color.a out).
        float shoal = water_shore(slot, water_rest);
        d *= shoal;
        normal = normalize(mix(vec3(0.0, 1.0, 0.0), normal, shoal));
        vec2 ruv = water_ripple_uv(slot, water_rest);
        float rfade = water_ripple_fade(ruv);
        if (rfade > 0.0) {
            float lod = water_ripple_lod(spacing, frame.water_bodies[slot].ripple.z, float(textureSize(t_water_ripple, 0).x));
            d.y += textureLod(t_water_ripple, ruv, lod).r * rfade;
        }
        float calm = 1.0 - a_color.b * a_color.b;
        d.y *= calm;
        normal = normalize(mix(vec3(0.0, 1.0, 0.0), normal, calm));
        position = a_position + d;
        tangent = vec4(normalize(vec3(normal.y, -normal.x, 0.0)), 1.0);
        uv1 = vec2(d.y, spacing);
        vec3 flow = mat3(model) * vec3(a_color.r - 0.5, 0.0, a_color.g - 0.5) * 10.0;
        color = vec4(flow.xz, a_color.b, shoal);
    }
#ifdef ZE_HOOK
    {
        VertexData zv;
        zv.position = position;
        zv.normal = normal;
        zv.tangent = tangent;
        zv.uv = uv0;
        zv.uv1 = uv1;
        zv.color = color;
        zv.model = model;
        zv.world_position = (model * vec4(position, 1.0)).xyz;
        zv.world_offset = vec3(0.0);
        zv.time = frame.misc.x;
        zv.vertex_index = gl_VertexIndex;
        zestengine_vertex(zv);
        hook_obj_offset = zv.position - position;
        hook_world_offset = zv.world_offset;
        position = zv.position;
        normal = zv.normal;
        tangent = zv.tangent;
        uv0 = zv.uv;
        uv1 = zv.uv1;
        color = zv.color;
    }
#endif
    vec4 world = model * vec4(position, 1.0);
    world.xyz += hook_world_offset;
    v_world_pos = world.xyz;
    // Normals need the inverse-transpose of the model's 3x3 so they stay correct
    // under non-uniform scale (plain mat3(model) skews them); it equals
    // mat3(model) for uniform scale/rotation. Tangents transform like directions,
    // so they use the plain basis.
    mat3 model3 = mat3(model);
    mat3 normal_mat = transpose(inverse(model3));
    v_normal = normalize(normal_mat * normal);
    v_tangent = vec4(normalize(model3 * tangent.xyz), tangent.w);
    // Water: the rest point (water-local XZ) the pixel shader evaluates its
    // normal, ripples and foam at.
    v_uv = FEAT_WATER ? water_rest : uv0;
    v_uv1 = uv1;
    v_color = color;
    v_instance_color = a_instance_color;
    v_instance_param = a_instance_param;
    gl_Position = VIEW_PROJ * world;
    v_cur_clip = gl_Position;
    // A vertex hook's displacement is assumed unchanged since the last frame
    // (the hook sees no previous time), so the velocity only carries the
    // object's own motion.
    v_prev_clip = frame.prev_view_proj * (model * vec4(a_prev_position + hook_obj_offset, 1.0) + vec4(hook_world_offset, 0.0));
}
