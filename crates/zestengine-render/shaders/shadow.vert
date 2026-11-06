// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// Shadow depth pass: render scene geometry from a light's point of view into
// a depth map. Vertex-only (no fragment stage); the light's view-projection
// arrives as push constants, the model matrix from the instance stream (so
// casters of one mesh draw as one instanced or GPU-culled indirect draw).
//
// -DCLUSTER (cluster shadows, GEO-5): no vertex buffers. The caster's vertices
// are pulled from the cluster pool by (instance, cluster) from the visible
// list the shadow view's cut wrote (cluster_cull.comp, same instance list
// format as the camera's), skinned instances read the posed vertices that
// cluster_skin.comp wrote for those entries: skinned casters never run the
// ordinary GPU skin pass for shadows.

// -DZE_HOOK: a custom shader's vertex hook (zestengine_vertex, spliced in by the
// render crate as zestengine_hook.glsl) runs here exactly as in standard.vert,
// so a swaying caster throws the shadow of the swaying surface.
// -DZE_TESS: the caster's material is tessellated: this stage only hands the
// world-space patch corners (and the light matrix, since tessellation stages
// see no push constants) to shadow.tesc / shadow.tese, which displace them
// exactly as tess.tese does for the camera.
#if defined(CLUSTER) && (defined(ZE_HOOK) || defined(ZE_TESS))
#error "vertex hooks and tessellation are not available on the cluster path"
#endif
#ifdef ZE_HOOK
#include "standard_common.glsl"
#endif

layout(push_constant) uniform Push {
    mat4 light_vp;
    mat4 model; // CLUSTER: [1][0] = first entry of this batch in the visible list (bits); else unused (instance stream)
} pc;

#ifdef CLUSTER
#define CLUSTER_SET 2
#include "cluster_common.glsl"
layout(std430, set = 2, binding = 11) readonly buffer Visible { uvec2 visible[]; };
struct SkinVert {
    vec4 pos;
    vec4 nrm;
    vec4 tan;
    vec4 prev;
};
layout(std430, set = 2, binding = 17) readonly buffer SkinPool { SkinVert skinned[]; };
layout(std430, set = 2, binding = 18) readonly buffer SkinSlots { uint slots[]; };
#else
layout(location = 0) in vec3 a_position;
#if defined(ZE_HOOK) || defined(ZE_TESS)
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;
layout(location = 3) in vec4 a_color;
layout(location = 4) in vec4 a_tangent;
layout(location = 5) in vec2 a_uv1;
#endif

layout(location = 8) in vec4 a_model0;
layout(location = 9) in vec4 a_model1;
layout(location = 10) in vec4 a_model2;
layout(location = 11) in vec4 a_model3;
#endif

#ifdef ZE_HOOK
#include "zestengine_hook.glsl"
#endif

#ifdef ZE_TESS
layout(location = 0) out vec3 v_world_pos;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec2 v_uv;
layout(location = 3) flat out vec4 v_lvp0;
layout(location = 4) flat out vec4 v_lvp1;
layout(location = 5) flat out vec4 v_lvp2;
layout(location = 6) flat out vec4 v_lvp3;
#endif

void main() {
#ifdef CLUSTER
    uint entry = floatBitsToUint(pc.model[1][0]) + uint(gl_InstanceIndex);
    uvec2 pick = entry < uint(visible.length()) ? visible[entry] : uvec2(0xFFFFFFFFu);
    vec3 a_position = vec3(0.0);
    vec3 n;
    vec4 t;
    vec2 uv;
    vec4 col;
    uvec2 tag;
    bool live = pick.x < uint(insts.length()) &&
        cluster_pull(pick.x, pick.y, uint(gl_VertexIndex), a_position, n, t, uv, col, tag);
    if (live && insts[pick.x].ref.z != 0u) {
        uint base = entry < uint(slots.length()) ? slots[entry] : 0xFFFFFFFFu;
        uint local = cluster_local_vertex(pick.x, pick.y, uint(gl_VertexIndex));
        if (base == 0xFFFFFFFFu || local == 0xFFFFFFFFu || base + local >= uint(skinned.length())) {
            live = false;
        } else {
            a_position = skinned[base + local].pos.xyz;
        }
    }
    if (!live) {
        // Collapsed vertex: zero area, never rasterised.
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }
    Inst pulled = insts[pick.x];
    mat4 model = mat4(vec4(pulled.model[0].xyz, 0.0), pulled.model[1], pulled.model[2], pulled.model[3]);
#else
    mat4 model = mat4(vec4(a_model0.xyz, 0.0), a_model1, a_model2, a_model3);
#endif
#if defined(ZE_HOOK) || defined(ZE_TESS)
    vec3 position = a_position;
    vec3 normal = a_normal;
    vec2 uv0 = a_uv;
    vec3 hook_world_offset = vec3(0.0);
#ifdef ZE_HOOK
    {
        VertexData zv;
        zv.position = position;
        zv.normal = normal;
        zv.tangent = a_tangent;
        zv.uv = uv0;
        zv.uv1 = a_uv1;
        zv.color = a_color;
        zv.model = model;
        zv.world_position = (model * vec4(position, 1.0)).xyz;
        zv.world_offset = vec3(0.0);
        zv.time = frame.misc.x;
        zv.vertex_index = gl_VertexIndex;
        zestengine_vertex(zv);
        hook_world_offset = zv.world_offset;
        position = zv.position;
        normal = zv.normal;
        uv0 = zv.uv;
    }
#endif
    vec4 world = model * vec4(position, 1.0);
    world.xyz += hook_world_offset;
#ifdef ZE_TESS
    v_world_pos = world.xyz;
    v_normal = normalize(transpose(inverse(mat3(model))) * normal);
    v_uv = uv0;
    v_lvp0 = pc.light_vp[0];
    v_lvp1 = pc.light_vp[1];
    v_lvp2 = pc.light_vp[2];
    v_lvp3 = pc.light_vp[3];
#endif
    gl_Position = pc.light_vp * world;
#else
    gl_Position = pc.light_vp * model * vec4(a_position, 1.0);
#endif
}
