// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Shared software world geometry, normals, materials and shadowed direct lighting.
#define FLUX_GDF_REFINABLE (SURFACE_NODE_COUNT>0u && surface_nodes[surface_tree_info.x+surface_tree_info.y].bmin.x>0.5)
#define FLUX_GDF_SOURCE_REFINABLE(index) ((surface_source_sides(index,false)&128u)!=0u)
#include "flux_software_geometry.glsl"
bool software_material_normal(vec3 p,uint who,float eps,vec3 direction,out vec3 n,out bool uncaptured_back) {
    uint sides=surface_source_sides(who&~MESH_SDF_HIT,(who&MESH_SDF_HIT)!=0u);
    n=hit_normal(p,who,eps);
    // A transformed SDF gradient lacks the determinant sign carried by a
    // world-space triangle cross product. Match RT/raster facing for every
    // material, including opaque and masked emissive surfaces.
    bool mirrored=(who&MESH_SDF_HIT)!=0u
        ? determinant(mat3(mesh_sdfs[who&~MESH_SDF_HIT].inverse))<0.0
        : (sides&256u)!=0u;
    if(mirrored) n=-n;
    bool two_sided=(sides&2u)!=0u;
    if(!two_sided && (sides&4u)!=0u) n=-n;
    bool back=dot(n,direction)>0.0;
    uncaptured_back=back && (sides&1u)==0u;
    if(back && (sides&1u)!=0u && !two_sided) return false;
    if(back) n=-n;
    return true;
}
vec4 hit_albedo(uint who) {
    return (who&MESH_SDF_HIT)!=0u ? mesh_sdfs[who&~MESH_SDF_HIT].albedo : insts[who].albedo;
}
vec3 hit_emission(uint who) {
    return (who&MESH_SDF_HIT)!=0u ? mesh_sdfs[who&~MESH_SDF_HIT].emission.rgb : insts[who].emission.rgb;
}
float light_cos(float deg) { return cos(radians(deg) * 0.5); }
float sphere_solid_angle(float radius, float dist) {
    if (dist <= radius) return 2.0 * PI;
    float s = min(radius / dist, 0.999);
    return 2.0 * PI * (1.0 - sqrt(1.0 - s * s));
}
// Analytic emitter NEE. Samples each emissive sphere directly (zero variance,
// so a small bright sphere lights the floor cleanly instead of needing rays to
// randomly hit it). No occlusion test: GDF-based occlusion blotches against the
// coarse field, and screen-space occlusion false-shadows whenever the emitter is
// on-screen (the ray grazes the receiver floor). The hemisphere bounce path still
// occludes via the GDF march, so real geometry shadows the indirect bounce.
vec3 emitter_light(vec3 p, vec3 n, float eps) {
    vec3 sum = vec3(0.0);
    for (uint i = 0u; i < pc.counts.w; ++i) {
        vec3 c = emitters[i].center_radius.xyz;
        float r = max(emitters[i].center_radius.w, 1e-3);
        vec3 d = c - p; float dist = length(d);
        vec3 l = dist > 1e-4 ? d / dist : n;
        // Soft terminator: light fades out smoothly as the emitter sinks below the
        // surface's horizon, instead of a hard cosine stop that made the pool edge
        // abrupt. A surface facing clearly away (more than ~17 deg past the
        // terminator) still gets nothing, so emission doesn't bleed onto back-faces.
        float ndl = clamp(dot(n, l) + 0.3, 0.0, 1.0) / 1.3;
        if (ndl <= 0.0) continue;
        // Occlusion: march the GDF to just before the sphere; skip if a solid
        // occluder (e.g. the couch) blocks it, so the emitter can't leak through.
        // Stop clearly short of the sphere: its radius is an estimate, and ending
        // exactly at the surface grazed the emitter itself on some frames.
        if (occluded(p + n * (eps * 2.0), l, max(dist - r * 1.25 - 0.02, eps * 2.0), eps)) continue;
        float sa = sphere_solid_angle(r, max(dist, r * 1.001));
        sum += emitters[i].emission.rgb * (ndl * sa);
    }
    return sum;
}
// Passes without a view exposure keep the fixed physical threshold.
#ifndef FLUX_LIGHT_CUTOFF
#define FLUX_LIGHT_CUTOFF 1e-3
#endif
vec3 direct_light(vec3 p, vec3 n, float eps, inout uint rng) {
    vec3 sum = vec3(0.0);
    float cutoff = FLUX_LIGHT_CUTOFF; // one exposure read per call, not per light
    for (uint i = 0u; i < pc.counts.z; ++i) {
        Light l = lights[i];
        int kind = int(l.pos_kind.w + 0.5);
        if(kind==6) continue;
        if(FLUX_HAS_AREA_LIGHTS && (kind==4 || kind==5)) {
            if(i+1u>=pc.counts.z) continue;
            vec3 direction; float distance;
            vec3 irradiance=flux_area_sample(l.pos_kind,l.dir_range,l.color,l.spot,lights[i+1u].dir_range.xyz,
                p,n,vec2(rnd(rng),rnd(rng)),direction,distance);
            if(max(irradiance.r,max(irradiance.g,irradiance.b))>0.0 &&
                !occluded(p,direction,max(distance-0.002,0.001),eps)) sum+=irradiance;
            ++i; continue;
        }
        vec3 to_light; float dist; float atten;
        if (kind == 0) { to_light = normalize(-l.dir_range.xyz); dist = 1e16; atten = 1.0; }
        else {
            vec3 d = l.pos_kind.xyz - p; dist = length(d);
            if (dist < 1e-4) continue;
            vec3 tl = d / dist;
            atten = realtime_distance_falloff(dist,l.dir_range.w);
            if (kind == 2) {
                float cd = dot(normalize(l.dir_range.xyz), -tl);
                float ci = light_cos(l.spot.x), co = light_cos(l.spot.y);
                atten *= realtime_spot_falloff(cd,ci,co);
            }
            to_light = tl;
        }
        float ndl = max(dot(n, to_light), 0.0);
        vec3 c = l.color.rgb * (ndl * atten);
        // No shadow march for a light that can't visibly light this hit.
        if (max(c.r, max(c.g, c.b)) < cutoff) continue;
        vec3 shadow_dir; float shadow_dist;
        flux_light_shadow_ray(to_light, dist, max(l.spot.z, 0.0), kind == 0,
                              vec2(rnd(rng), rnd(rng)), shadow_dir, shadow_dist);
        if (occluded(p + n * (eps * 2.0), shadow_dir, shadow_dist, eps)) continue;
        sum += c;
    }
    return sum;
}
// Surface offset that pushes a ray start out of the coarse GDF voxel its origin
// sits in. The global SDF over-estimates distance (its surface is "fat" by ~a
// voxel), so a ray starting only `eps` off the surface immediately re-reports a
// hit on its OWN mesh and reads that mesh's lit cache — self-illumination that
// lit the back/non-normal side of static meshes (the only geometry in the GDF).
// Offsetting by ~a voxel along the normal skips that self-hit.
float gdf_surface_offset(vec3 p,float eps) {
    if(flux_field_detail_only()) return max(eps*2.0,mesh_sdf_bias(p));
    vec3 cell = flux_field_cell(flux_field_level(p));
    float vox = max(cell.x, max(cell.y, cell.z));
    return max(eps * 2.0, vox * 1.5);
}

float software_surface_offset(vec3 p,float eps) {
    return pc.gdf_max.w>0.0 ? mesh_sdf_bias(p) : gdf_surface_offset(p,eps);
}

