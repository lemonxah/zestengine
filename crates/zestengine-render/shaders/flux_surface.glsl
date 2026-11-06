// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Shared point lookup for software and hardware ray hits. SURFACE_BINDING is
// the first free binding in the calling pipeline. Gather defaults can be
// overridden by reflection shaders with their own uniform layout.
#include "flux_surface_geometry.glsl"
#ifndef SURFACE_NODE_COUNT
#define SURFACE_NODE_COUNT uint(pc.probe_info.z)
#endif
#ifndef SURFACE_MULTIBOUNCE
#define SURFACE_MULTIBOUNCE (pc.counts.y > 1u)
#endif
struct SurfaceCard { mat4 view_proj; vec4 normal_texel; uvec4 rect; mat4 capture_to_world; };
vec3 surface_world_position(SurfaceCard card,vec3 p) {
    return (card.rect.w&64u)!=0u ? (card.capture_to_world*vec4(p,1.0)).xyz : p;
}
vec3 surface_world_normal(SurfaceCard card,vec3 n) {
    return (card.rect.w&64u)!=0u ? normalize(mat3(card.capture_to_world)*n) : n;
}
struct SurfaceNode { vec3 bmin; uint left; vec3 bmax; uint right; };
layout(set=0,binding=SURFACE_BINDING) readonly buffer SurfaceCards { SurfaceCard surface_cards[]; };
#ifdef SURFACE_FEEDBACK_KIND
layout(set=0,binding=SURFACE_BINDING+1) buffer SurfaceNodes { uvec4 surface_tree_info; SurfaceNode surface_nodes[]; };
#else
layout(set=0,binding=SURFACE_BINDING+1) readonly buffer SurfaceNodes { uvec4 surface_tree_info; SurfaceNode surface_nodes[]; };
#endif
layout(set=0,binding=SURFACE_BINDING+2) uniform sampler2D surface_position;
layout(set=0,binding=SURFACE_BINDING+3) uniform sampler2D surface_normal;
layout(set=0,binding=SURFACE_BINDING+4) uniform sampler2D surface_albedo;
layout(set=0,binding=SURFACE_BINDING+5) uniform sampler2D surface_emission;
layout(set=0,binding=SURFACE_BINDING+6) uniform sampler2D surface_direct;
layout(set=0,binding=SURFACE_BINDING+7) uniform sampler2D surface_indirect;
#ifdef FLUX_SURFACE_LIGHTING
layout(set=0,binding=SURFACE_BINDING+8,rgba16f) uniform writeonly image2D surface_direct_out;
layout(set=0,binding=SURFACE_BINDING+9,rgba16f) uniform writeonly image2D surface_indirect_out;
#endif
struct SurfaceSample { vec3 position; vec3 geometry_normal; float geometry_encoded; vec3 albedo; vec3 normal; vec3 emission; float metallic; float roughness; ivec2 texel; vec2 lighting_pixel; uint card; bool static_geometry; bool valid; };
SurfaceSample surface_lookup_range(vec3 p, vec3 n, float error, uint start, uint count) {
    SurfaceSample result; result.valid = false;
    if (start >= count) return result;
    uint ni = start;
    vec2 best_pixel = vec2(0.0);
    ivec2 best_at = ivec2(0); vec4 best_normal = vec4(0.0); vec4 best_position=vec4(0.0); uint best_card = 0u;
    float best = 1e30;
    while (ni < count) {
        SurfaceNode node = surface_nodes[ni];
        if (any(lessThan(p,node.bmin-vec3(error))) || any(greaterThan(p,node.bmax+vec3(error)))) { ni = node.right; continue; }
        if ((node.left & 0x80000000u) == 0u) {
            ++ni; // first child; its escape points to the next sibling
            continue;
        }
        uint first = node.left & 0x7fffffffu;
        ni = node.right;
        // Leaves identify individual exterior/interior cards. Their clipped
        // bounds and normal reject unrelated layers before texture access.
        {
            SurfaceCard card = surface_cards[first];
            if (card.rect.w == 0u || dot(card.normal_texel.xyz,n) < 0.25) continue;
            vec3 cp = (card.view_proj*vec4(p,1.0)).xyz;
            // SDF hit uncertainty is in world units; thin/interior capture
            // volumes must not turn it into an arbitrary normalized-depth cut.
            float depth_error = 0.01 + error*length(vec3(card.view_proj[0][2],card.view_proj[1][2],card.view_proj[2][2]));
            if (abs(cp.x)>1.0 || abs(cp.y)>1.0 || cp.z < -depth_error || cp.z > 1.0+depth_error) continue;
            vec2 q = (cp.xy*0.5+0.5)*float(card.rect.z)-0.5;
            ivec2 base = ivec2(floor(q));
            // Check neighbouring captured positions, never interpolate across
            // card borders or disjoint surfaces in a concave mesh.
            for (int corner=0; corner<4; ++corner) {
                ivec2 local = base+ivec2(corner&1,corner>>1);
                if (any(lessThan(local,ivec2(0))) || any(greaterThanEqual(local,ivec2(card.rect.z)))) continue;
                ivec2 at = ivec2(card.rect.xy)+local;
                vec4 pos = texelFetch(surface_position,at,0);
                if (pos.w < 0.5) continue;
                pos.xyz=surface_world_position(card,pos.xyz);
                // Reject distant/noncompetitive samples before fetching and
                // decoding normals. Squared distances preserve the same
                // nearest-surface and world-space tolerance checks.
                vec3 delta=pos.xyz-p;
                float d=dot(delta,delta);
                float tolerance=card.normal_texel.w*2.0+error;
                if (d>tolerance*tolerance || d>=best) continue;
                vec4 normal = texelFetch(surface_normal,at,0);
                if (dot(surface_world_normal(card,surface_geometry_normal(pos.w,normal.xyz)),n) < 0.5) continue;
                best = d;
                best_at = at; best_pixel = vec2(card.rect.xy)+q;
                best_normal = normal; best_position=pos; best_card = first; result.valid = true;
            }
        }
    }
    if (result.valid) {
        vec4 albedo = texelFetch(surface_albedo,best_at,0);
        result.position = best_position.xyz;
        result.geometry_encoded = best_position.w;
        SurfaceCard card=surface_cards[best_card];
        result.geometry_normal = surface_world_normal(card,surface_geometry_normal(best_position.w,best_normal.xyz));
        result.albedo = albedo.rgb; result.metallic = albedo.a;
        result.normal = surface_world_normal(card,best_normal.xyz); result.roughness = best_normal.a;
        vec4 emitted=texelFetch(surface_emission,best_at,0);
        result.emission=emitted.rgb;result.static_geometry=emitted.a>1.5;
        result.texel = best_at; result.lighting_pixel = best_pixel; result.card = best_card;
#ifdef SURFACE_FEEDBACK_KIND
        // Thin demand feedback to 1/16 invocations, rotating the sample each
        // frame. Only view traces write it; radiosity must not prioritize its
        // own sources as if a camera had seen them. Queue capacity matches CPU.
        uint feedback_sample = gl_GlobalInvocationID.x*1973u + gl_GlobalInvocationID.y*9277u
            + gl_GlobalInvocationID.z*26699u + surface_nodes[surface_tree_info.z+1024u].right;
        if ((feedback_sample & 15u)==0u)
            atomicOr(surface_nodes[surface_tree_info.z+1024u+best_card].left,SURFACE_FEEDBACK_KIND);
#endif
    }
    return result;
}

SurfaceSample surface_lookup(vec3 p, vec3 n, float error) {
    return surface_lookup_range(p,n,error,0u,SURFACE_NODE_COUNT);
}
SurfaceSample surface_lookup_instance(vec3 p,vec3 n,float error,uint source_draw_plus_one) {
    SurfaceSample missing; missing.valid=false;
    if (SURFACE_NODE_COUNT==0u || source_draw_plus_one==0u || source_draw_plus_one>surface_tree_info.y) return missing;
    uint root=surface_nodes[surface_tree_info.x+source_draw_plus_one-1u].left;
    // A known object with missing cards must use its own hit shading. Never
    // substitute a nearby object's material or radiance via the global tree.
    if (root==0u) return missing;
    --root;
    return surface_lookup_range(p,n,error,root,surface_nodes[root].right);
}

// Both software fields preserve their own instance-table ordering. Known
// uncaptured objects use their own hit material instead of a neighbour's card.
SurfaceSample surface_lookup_source(vec3 p,vec3 n,float error,SurfaceNode mapping,uint index) {
    SurfaceSample missing; missing.valid=false;
    if (index>=mapping.right) return missing;
    SurfaceNode source=surface_nodes[mapping.left+index];
    if (source.right==0u) return surface_lookup(p,n,error);
    if (source.left==0u) return missing;
    uint root=source.left-1u;
    return surface_lookup_range(p,n,error,root,surface_nodes[root].right);
}
SurfaceSample surface_lookup_mesh(vec3 p,vec3 n,float error,uint index) {
    SurfaceSample missing; missing.valid=false;
    if (SURFACE_NODE_COUNT==0u) return missing;
    SurfaceNode mapping=surface_nodes[surface_tree_info.x+surface_tree_info.y];
    return surface_lookup_source(p,n,error,mapping,index);
}
SurfaceSample surface_lookup_gdf(vec3 p,vec3 n,float error,uint index) {
    SurfaceSample missing; missing.valid=false;
    if (SURFACE_NODE_COUNT==0u) return missing;
    SurfaceNode mesh_mapping=surface_nodes[surface_tree_info.x+surface_tree_info.y];
    SurfaceNode mapping=surface_nodes[mesh_mapping.left+mesh_mapping.right];
    return surface_lookup_source(p,n,error,mapping,index);
}

// Reflected diffuse radiance, excluding material emission (the caller applies
// its own emission/analytic-source policy). The material capture has four times
// the resolution of its diffuse-lighting
// probes. Clamp to the selected card so filtering never reaches another object.
vec3 surface_filtered_lighting(SurfaceSample surface, bool hit_indirect, vec3 hit_diffuse, out bool valid) {
    valid = false;
    if (!surface.valid) return vec3(0.0);
    SurfaceCard card = surface_cards[surface.card];
#ifdef FLUX_SURFACE_LIGHTING
    const uint ready = 2u, indirect_ready = 16u; // previous frame only: no in-dispatch feedback
#else
    const uint ready = 4u, indirect_ready = 32u;
#endif
    if (hit_indirect && !SURFACE_MULTIBOUNCE) { valid=true; return vec3(0.0); }
    if ((card.rect.w & (hit_indirect ? indirect_ready : ready)) == 0u) return vec3(0.0);
    // Sample lighting at the continuous hit position, not at the nearest
    // material texel: snapping makes radiance jump as a ray slides on a wall.
    // Lighting probes capture material texel 4*i+2 (centre 4*i+2.5), hence the
    // half-material-texel correction already included in lighting_pixel.
    vec2 grid=surface.lighting_pixel*0.25-0.5;
    ivec2 base=ivec2(floor(grid)); vec2 f=fract(grid);
    ivec2 lo=ivec2(card.rect.xy)/4;
    ivec2 hi=lo+ivec2(card.rect.z/4u)-1;
    vec3 sum=vec3(0.0); float total=0.0;
    for(int corner=0;corner<4;++corner) {
        ivec2 offset=ivec2(corner&1,corner>>1);
        ivec2 q=clamp(base+offset,lo,hi);
        vec2 bilinear=mix(vec2(1.0)-f,f,vec2(offset));
        float weight=bilinear.x*bilinear.y;
        if(weight<=0.0) continue;
        // Lighting texel q was shaded from material texel 4*q+2. Validate that
        // exact receiver, not the high-resolution texel nearest the ray hit.
        vec4 position=textureQueryLevels(surface_position)>2 ? texelFetch(surface_position,q,2)
            : texelFetch(surface_position,q*4+2,0);
        if(position.w<0.5) continue;
        position.xyz=surface_world_position(card,position.xyz);
        vec3 normal=surface.geometry_normal;
        if(position.w!=surface.geometry_encoded) normal=surface_world_normal(card,position.w>=1.5
            ? surface_geometry_normal(position.w,vec3(0,0,1)) : texelFetch(surface_normal,q*4+2,0).xyz);
        vec3 delta=position.xyz-surface.position;
        float tolerance=max(0.005,card.normal_texel.w*0.1);
        if(dot(normal,surface.geometry_normal)<0.9
            || max(abs(dot(delta,normal)),abs(dot(delta,surface.geometry_normal)))>tolerance) continue;
        vec3 lighting;
        if(hit_indirect) {
            vec4 indirect=texelFetch(surface_indirect,q,0);
            if(indirect.a<=0.5) continue;
            // Remove the response of the actual cached receiver before applying
            // the exact ray-hit material. Never divide by the nearest hit texel:
            // its material can differ from the low-resolution lighting receiver.
            vec4 material=texelFetch(surface_albedo,q*4+2,0);
            vec3 cached_diffuse=material.rgb*(1.0-material.a);
            if(any(bvec3(hit_diffuse.r>1e-4 && cached_diffuse.r<1e-4,
                         hit_diffuse.g>1e-4 && cached_diffuse.g<1e-4,
                         hit_diffuse.b>1e-4 && cached_diffuse.b<1e-4))) continue;
            lighting=indirect.rgb*hit_diffuse/max(cached_diffuse,vec3(1e-4));
        } else {
            vec4 direct=texelFetch(surface_direct,q,0);
            if(direct.a<=0.5) continue;
            vec3 indirect=SURFACE_MULTIBOUNCE && (card.rect.w&indirect_ready)!=0u ? texelFetch(surface_indirect,q,0).rgb : vec3(0.0);
            lighting=direct.rgb+indirect;
        }
        sum+=weight*lighting; total+=weight;
    }
    // Missing coverage falls back to hit lighting. Never treat an invalid
    // cache texel as a black surface, and never max-blend valid dark samples.
    valid=total>1e-4;
    return valid ? max(sum/total,vec3(0.0)) : vec3(0.0);
}

vec3 surface_radiance(SurfaceSample surface, out bool valid) {
    return surface_filtered_lighting(surface,false,vec3(0),valid);
}
// Cached secondary diffuse transport only; exact direct lighting is added by
// the ray-hit caller. Emission is deliberately absent from both entry points.
vec3 surface_hit_indirect(SurfaceSample surface, vec3 hit_diffuse, out bool valid) {
    if(!SURFACE_MULTIBOUNCE) { valid=true;return vec3(0); }
    return surface_filtered_lighting(surface,true,hit_diffuse,valid);
}
