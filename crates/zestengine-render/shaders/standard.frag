// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
#ifdef MULTIVIEW
#extension GL_EXT_multiview : require
#endif
// zestengine standard shader, fragment stage (phase 1: PBR/toon hybrid core).
// Feature toggles are specialization constants: each material's enabled
// feature set selects a pipeline variant; disabled features compile out.

layout(constant_id = 0) const bool FEAT_TOON = false;
layout(constant_id = 1) const bool FEAT_NORMAL_MAP = false;
layout(constant_id = 2) const bool FEAT_EMISSION = false;
layout(constant_id = 3) const uint ALPHA_MODE = 0u; // 0 opaque, 1 cutout, 2 blend, 3 transclipping
layout(constant_id = 4) const bool FEAT_WATER = false;
// Motion-vector pass (velocity pipeline variants): write the screen-space
// reprojection difference into the RG16F velocity attachment.
layout(constant_id = 5) const bool WRITE_VELOCITY = false;
// Per-instance material properties (base colour + one parameter vec4).
layout(constant_id = 6) const bool FEAT_PER_INSTANCE = false;

#include "standard_common.glsl"
#include "realtime_light.glsl"
#include "material_response.glsl"

// Water: copies of the opaque scene (depth, HDR colour) taken before the
// transparent draws; valid for this pass when frame.water.y > 0.5.
layout(set = 0, binding = 5) uniform sampler2D u_opaque_depth;
layout(set = 0, binding = 6) uniform sampler2D u_opaque_color;
// Reflection probe: prefiltered environment cubemap (mirror at mip 0, rougher
// toward higher mips). Sampled by reflection direction + roughness.
layout(set = 0, binding = 7) uniform samplerCube u_env;
// Second reflection probe, blended per pixel with u_env by box influence.
layout(set = 0, binding = 9) uniform samplerCube u_env_b;
// Third/fourth reflection probes.
layout(set = 0, binding = 11) uniform samplerCube u_env_c;
layout(set = 0, binding = 12) uniform samplerCube u_env_d;
// Planar reflections: per-plane mirrored scene renders, sampled at screen UV
// by surfaces whose push constant carries a planar layer (pc.emission.w >= 1).
layout(set = 0, binding = 8) uniform sampler2DArray u_planar;
// Each water body's ripple field by body slot (r height, gb local XZ
// gradient, with mips): the caustics on what lies under it.
layout(set = 0, binding = 10) uniform sampler2D u_water_ripples[MAX_WATER_BODIES];
// Max roughness mip = (mip count - 1) of the CURRENTLY bound cube, queried at
// runtime so it's correct whether u_env is the 64px skybox cube (7 mips) or a
// higher-res captured reflection-probe cube (e.g. 256px = 9 mips). A hardcoded
// constant under-blurred probe reflections (sampled mip 6 of 9 -> too sharp on
// rough surfaces, the "defined blob on a rough floor" artifact). See
// REFLECTION_PROBES_MATH.md 6.5.
#define ENV_MAX_LOD (float(textureQueryLevels(u_env) - 1))

layout(set = 0, binding = 1) uniform sampler2DArrayShadow u_shadow;

// Baked probe SH-L1: 4 coefficients (RGB in .xyz) per probe.
struct Probe { vec4 c[6]; }; // c[0..3]=radiance+dist SH, c[4]=dist² SH (Chebyshev)
layout(set = 0, binding = 2) readonly buffer Probes { Probe probes[]; };

// Baked lightmaps (static-object GI): one array layer per object, sampled by
// uv1. Each texel is incoming irradiance E; the caller applies albedo/PI.
layout(set = 0, binding = 3) uniform sampler2DArray u_lightmap;

// Screen-space GI gather (sparse screen probes). One texel per
// SGI_DIV×SGI_DIV screen block; rgb = indirect irradiance, a = probe camera
// distance (for depth-aware upsampling). Used in place of the coarse world-probe
// grid when active (frame.debug.z > 0.5). Sampled by screen UV.
layout(set = 0, binding = 4) uniform sampler2D u_screen_gi;
#ifndef MULTIVIEW
// Screen-space ambient occlusion (ao.rs): full-resolution visibility of this
// pixel, computed from the depth prepass before this pass. Read only while
// frame.refl_blend2.w > 0.5 (1 = on, 2 = on with multi-bounce); never in the
// multiview (VR) variant.
layout(set = 0, binding = 13) uniform sampler2D u_ssao;
#endif

// Multi-bounce AO (Jimenez et al. 2016, "Practical Realtime Strategies for
// Accurate Indirect Occlusion"): the occlusion seen on a surface of albedo `a`
// after interreflections, fitted as a cubic in the visibility.
vec3 ao_multi_bounce(float v, vec3 a) {
    vec3 x = 2.0404 * a - 0.3324;
    vec3 y = -4.7951 * a + 0.6417;
    vec3 z = 2.7552 * a + 0.6903;
    return max(vec3(v), ((v * x + y) * v + z) * v);
}
// Flux integrate already resolves the denoised probes at full resolution,
// including per-pixel normals and emitter shadows. Filtering it again here
// blurred contact shadows over 17 pixels and repeated 25 texture reads for
// every shaded fragment. Sample the resolved irradiance at this pixel.
vec3 screen_gi_sample(vec2 suv) {
    return texture(u_screen_gi, suv).rgb;
}

// Diffuse irradiance / PI from the probe's SH-L1 radiance coefficients, in
// direction n. The bake stores radiance projected onto SH (Y0 = 0.282095,
// Y1 = 0.488603; sh1~y, sh2~z, sh3~x). Converting radiance SH to Lambertian
// irradiance applies the cosine-lobe band factors A0 = PI, A1 = 2PI/3; dividing
// by PI (the diffuse BRDF) leaves the L1 band scaled by 2/3
// (0.488603 * 2/3 = 0.325735). The constant term then matches the flat ambient.
vec3 sh_eval(uint i, vec3 n) {
    return probes[i].c[0].rgb * 0.282095
         + 0.325735 * (probes[i].c[1].rgb * n.y
                     + probes[i].c[2].rgb * n.z
                     + probes[i].c[3].rgb * n.x);
}

// RADIANCE (not irradiance) from the probe SH-L1 in direction `dir`: the raw L1
// band reconstruction (no cosine-lobe factor), used for the FluxVoxel specular-
// from-volume term. Very low-frequency, so only a rough-glossy approximation.
vec3 sh_radiance(uint i, vec3 dir) {
    return probes[i].c[0].rgb * 0.282095
         + 0.488603 * (probes[i].c[1].rgb * dir.y
                     + probes[i].c[2].rgb * dir.z
                     + probes[i].c[3].rgb * dir.x);
}

// Expected distance to geometry from probe `i` in world direction `dir`,
// reconstructed from the SH-L1 stored in the coeffs' .w lanes (raw band factors,
// no cosine convolution). Zero (the bake path) means "no visibility data".
float dist_at(uint i, vec3 dir) {
    return probes[i].c[0].w * 0.282095
         + 0.488603 * (probes[i].c[1].w * dir.y
                     + probes[i].c[2].w * dir.z
                     + probes[i].c[3].w * dir.x);
}
// Mean SQUARED distance (second moment), from the dist² SH in c[4]. Pairs with
// dist_at for the two-moment Chebyshev visibility (matches the gather).
float dist2_at(uint i, vec3 dir) {
    return probes[i].c[4].x * 0.282095
         + 0.488603 * (probes[i].c[4].y * dir.y
                     + probes[i].c[4].z * dir.z
                     + probes[i].c[4].w * dir.x);
}

// Normalized box coords (0..1 across the volume) for a world point.
vec3 volume_coords(ProbeVolume v, vec3 world_pos) {
    vec3 local = (v.world_to_local * vec4(world_pos, 1.0)).xyz;
    vec3 size = max(v.size_base.xyz, vec3(1e-4));
    return (local + size * 0.5) / size;
}

// DDGI-style 8-corner blend of probe irradiance within one volume. Each corner's
// trilinear weight is modulated by:
//   - visibility: the probe's stored directional distance vs. the actual probe→
//     fragment distance (a soft Chebyshev-lite test). A probe occluded from the
//     fragment (e.g. on the far side of a wall) is down-weighted, killing leaks,
//   - front-facing: probes roughly behind the surface contribute less.
// Falls back to plain trilinear where no visibility data exists (bake path).
vec3 sample_volume(ProbeVolume v, vec3 world_pos, vec3 n) {
    vec3 t = volume_coords(v, world_pos);
    ivec3 cnt = ivec3(v.counts.xyz + 0.5);
    vec3 size = max(v.size_base.xyz, vec3(1e-4));
    vec3 center = -(transpose(mat3(v.world_to_local)) * v.world_to_local[3].xyz); // volume origin (rotation-aware)
    vec3 cell = size / max(vec3(cnt - 1), vec3(1.0));
    vec3 g = clamp(t, vec3(0.0), vec3(1.0)) * vec3(cnt - 1);
    ivec3 g0 = ivec3(floor(g));
    vec3 f = g - vec3(g0);
    // Smootherstep (quintic) the interpolation factor so the trilinear blend is
    // C2-continuous across cell boundaries. Plain linear trilinear kinks the
    // gradient at every boundary, and even smoothstep (cubic, C1) leaves a jump in
    // the SECOND derivative there — which the eye still reads as faint grid lines
    // (Mach banding) between voxels. The quintic is C2, so the cell edges blend
    // away entirely without needing a finer grid.
    f = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    vec3 sum = vec3(0.0);
    float wsum = 0.0;
    for (int i = 0; i < 8; ++i) {
        ivec3 off = ivec3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        ivec3 gc = clamp(g0 + off, ivec3(0), cnt - 1);
        vec3 tw = mix(vec3(1.0) - f, f, vec3(off));
        float w = tw.x * tw.y * tw.z;
        uint idx = uint(v.size_base.w) + uint(gc.x + gc.y * cnt.x + gc.z * cnt.x * cnt.y);
        // Probe world position from its grid coords.
        vec3 gn = vec3(gc) / max(vec3(cnt - 1), vec3(1.0));
        vec3 ppos = center + transpose(mat3(v.world_to_local)) * ((gn - 0.5) * size) + probes[idx].c[5].xyz;
        vec3 to_frag = world_pos - ppos;
        float pd = length(to_frag);
        vec3 dir = pd > 1e-5 ? to_frag / pd : n;
        float md = max(dist_at(idx, dir), 0.0);
        float md2 = max(dist2_at(idx, dir), 0.0);
        // Visibility (DDGI two-moment Chebyshev, matching the gather): smooth
        // occlusion from the distance variance, so it darkens the bounce behind
        // an occluder as a soft gradient. md==0 → no data; md2==0 → baked sidecar
        // (single-moment smoothstep fallback). Keeps a small floor so shadowed
        // faces still gather soft fill instead of crushing to black.
        float band = 3.0 * max(cell.x, max(cell.y, cell.z));
        float vis;
        if (md <= 1e-4 || pd <= md) {
            vis = 1.0;
        } else if (md2 > 1e-4) {
            float variance = max(md2 - md * md, 1e-4);
            float dd = pd - md;
            // Higher floor (0.5) = gentler occlusion, so any residual per-probe
            // moment variation reads as a soft contact-shadow, not hard grid dots.
            vis = max(variance / (variance + dd * dd), 0.5);
        } else {
            vis = clamp(1.0 - (pd - md) / max(band, 1e-3), 0.5, 1.0);
        }
        // Front-facing weight (probe should be on the surface's lit side). Kept
        // gentle + a generous floor so shadowed/back faces still gather soft fill
        // from surrounding probes instead of crushing to black. Leak prevention
        // is the visibility term's job, not this one.
        float nw = clamp(dot(-dir, n) * 0.5 + 0.5, 0.0, 1.0);
        nw = nw * 0.65 + 0.35;
        w *= vis * nw;
        sum += w * sh_eval(idx, n);
        wsum += w;
    }
    return wsum > 1e-5 ? max(sum / wsum, vec3(0.0)) : vec3(0.0);
}

// Trilinearly-interpolated baked irradiance at world_pos for normal n, plus a
// coverage weight in [0,1] for the caller to blend against ambient. Volumes are
// ordered finest-first (cascades): the first that contains the point wins, but
//   - near an INNER cascade's boundary we cross-fade into the next (coarser)
//     cascade so the resolution change isn't a visible seam (coverage stays 1),
//   - the OUTERMOST cascade keeps full coverage everywhere inside its box and
//     fades to ambient over one probe cell OUTSIDE it (DDGI). Fading inside the
//     box starved the surfaces volumes are built for: a floor resting on the
//     bottom face got ~2% of its GI, leaving only the probe dots visible.

// How far a point (volume coords `t`) lies outside volume `v`, in probe cells.
float volume_outside_cells(ProbeVolume v, vec3 t) {
    vec3 cells = max(v.counts.xyz - 1.0, vec3(1.0));
    vec3 out_g = max(max(-t, t - 1.0), vec3(0.0)) * cells;
    return max(out_g.x, max(out_g.y, out_g.z));
}

// `world_pos` moved onto the nearest point of volume `v`'s box.
vec3 volume_clamp(ProbeVolume v, vec3 world_pos) {
    vec3 half_size = max(v.size_base.xyz, vec3(1e-4)) * 0.5;
    vec3 local = clamp((v.world_to_local * vec4(world_pos, 1.0)).xyz, -half_size, half_size);
    mat3 local_to_world = transpose(mat3(v.world_to_local));
    return local_to_world * (local - v.world_to_local[3].xyz);
}

// Volumes in the frame UBO that are baked cascades: they come first, the Light Volumes
// (params.x > 0) after them.
int baked_volumes() {
    int total = int(frame.misc.w + 0.5);
    int n = 0;
    for (int i = 0; i < MAX_PROBE_VOLUMES && i < total; ++i) {
        if (frame.probe_volumes[i].params.x > 0.0) {
            break;
        }
        n++;
    }
    return n;
}

// Light Volumes: baked SH light for objects without a lightmap. Every Light Volume covering
// `world_pos` contributes its plain trilinear probe irradiance weighted by its weight x an edge
// fade (smoothstep over `blend distance` metres inward from the faces); the result is their
// weighted mean and `coverage` (0..1) says how much of the final light it should replace.
vec3 light_volume_irradiance(vec3 world_pos, vec3 n, out float coverage) {
    int total = int(frame.misc.w + 0.5);
    vec3 acc = vec3(0.0);
    float wsum = 0.0;
    for (int vi = 0; vi < MAX_PROBE_VOLUMES && vi < total; ++vi) {
        ProbeVolume v = frame.probe_volumes[vi];
        if (v.params.x <= 0.0) {
            continue;
        }
        vec3 local = (v.world_to_local * vec4(world_pos, 1.0)).xyz;
        vec3 size = max(v.size_base.xyz, vec3(1e-4));
        vec3 inside = size * 0.5 - abs(local);
        float d = min(inside.x, min(inside.y, inside.z));
        if (d <= 0.0) {
            continue;
        }
        float w = v.params.y * smoothstep(0.0, v.params.x, d);
        if (w <= 0.0) {
            continue;
        }
        ivec3 cnt = ivec3(v.counts.xyz + 0.5);
        vec3 g = clamp((local + size * 0.5) / size, vec3(0.0), vec3(1.0)) * vec3(cnt - 1);
        ivec3 g0 = ivec3(floor(g));
        vec3 f = g - vec3(g0);
        vec3 e = vec3(0.0);
        for (int i = 0; i < 8; ++i) {
            ivec3 off = ivec3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
            ivec3 gc = clamp(g0 + off, ivec3(0), cnt - 1);
            vec3 tw = mix(vec3(1.0) - f, f, vec3(off));
            uint idx = uint(v.size_base.w) + uint(gc.x + gc.y * cnt.x + gc.z * cnt.x * cnt.y);
            e += tw.x * tw.y * tw.z * sh_eval(idx, n);
        }
        acc += w * max(e, vec3(0.0));
        wsum += w;
    }
    coverage = min(wsum, 1.0);
    return wsum > 1e-5 ? acc / wsum : vec3(0.0);
}

float sample_probes(vec3 world_pos, vec3 n, out vec3 irradiance) {
    int vcount = baked_volumes();
    for (int vi = 0; vi < vcount && vi < MAX_PROBE_VOLUMES; ++vi) {
        ProbeVolume v = frame.probe_volumes[vi];
        vec3 t = volume_coords(v, world_pos);
        bool has_next = (vi + 1 < vcount && vi + 1 < MAX_PROBE_VOLUMES);
        float outside = volume_outside_cells(v, t);
        if (outside > 0.0 && (has_next || outside >= 1.0)) {
            continue;
        }
        if (outside > 0.0) {
            irradiance = sample_volume(v, volume_clamp(v, world_pos), n);
            return 1.0 - smoothstep(0.0, 1.0, outside);
        }
        vec3 fine = sample_volume(v, world_pos, n);
        // Distance to the nearest box face in normalized units (0 at the face,
        // 0.5 at the center).
        vec3 edge = min(t, vec3(1.0) - t);
        float e = min(edge.x, min(edge.y, edge.z));
        if (has_next) {
            // Wide, smooth cross-fade into the coarser cascade hides the band.
            float inner = smoothstep(0.0, 0.22, e);
            if (inner < 1.0) {
                ProbeVolume v2 = frame.probe_volumes[vi + 1];
                vec3 coarse = sample_volume(v2, world_pos, n);
                fine = mix(coarse, fine, inner);
            }
            irradiance = fine;
            return 1.0;
        }
        irradiance = fine;
        return 1.0;
    }
    irradiance = vec3(0.0);
    return 0.0;
}

// FluxVoxel specular-from-volume (VXGI-style glossy approximation): trilinearly
// sample the probe volume's RADIANCE in the reflection direction `dir`, so a
// metallic/rough surface receives the emissive + voxel-light bounce held in the
// volume — not just the reflection cube. Coarse (SH-L1 is very low frequency), so
// it's only used for rough-glossy surfaces. Returns false when `world_pos` is
// outside every volume. No cascade cross-fade / visibility — a glossy fill term.
bool volume_radiance(vec3 world_pos, vec3 dir, out vec3 rad) {
    int vcount = baked_volumes();
    for (int vi = 0; vi < vcount && vi < MAX_PROBE_VOLUMES; ++vi) {
        ProbeVolume v = frame.probe_volumes[vi];
        vec3 t = volume_coords(v, world_pos);
        if (any(lessThan(t, vec3(0.0))) || any(greaterThan(t, vec3(1.0)))) {
            continue;
        }
        ivec3 cnt = ivec3(v.counts.xyz + 0.5);
        vec3 g = clamp(t, vec3(0.0), vec3(1.0)) * vec3(cnt - 1);
        ivec3 g0 = ivec3(floor(g));
        vec3 f = g - vec3(g0);
        f = f * f * f * (f * (f * 6.0 - 15.0) + 10.0); // smootherstep (C2): blend voxel edges
        vec3 sum = vec3(0.0);
        float wsum = 0.0;
        for (int i = 0; i < 8; ++i) {
            ivec3 off = ivec3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
            ivec3 gc = clamp(g0 + off, ivec3(0), cnt - 1);
            vec3 tw = mix(vec3(1.0) - f, f, vec3(off));
            float w = tw.x * tw.y * tw.z;
            uint idx = uint(v.size_base.w) + uint(gc.x + gc.y * cnt.x + gc.z * cnt.x * cnt.y);
            sum += w * max(sh_radiance(idx, dir), vec3(0.0));
            wsum += w;
        }
        rad = wsum > 1e-5 ? sum / wsum : vec3(0.0);
        return true;
    }
    return false;
}

// Smooth probe irradiance in normal `n` — pure trilinear `sh_eval`, NO Chebyshev /
// front-facing weighting. Used to add realtime voxel GI on top of a LIGHTMAP: the
// bake already carries the static occlusion, so the per-probe Chebyshev variation
// (which draws the grid as fixed dots on bright baked surfaces) isn't wanted here.
vec3 sample_irradiance_smooth(vec3 world_pos, vec3 n) {
    int vcount = baked_volumes();
    for (int vi = 0; vi < vcount && vi < MAX_PROBE_VOLUMES; ++vi) {
        ProbeVolume v = frame.probe_volumes[vi];
        vec3 t = volume_coords(v, world_pos);
        // Outermost volume: a one-cell margin outside the box (as sample_probes).
        bool has_next = (vi + 1 < vcount && vi + 1 < MAX_PROBE_VOLUMES);
        float outside = volume_outside_cells(v, t);
        if (outside > 0.0 && (has_next || outside >= 1.0)) {
            continue;
        }
        float fade = 1.0 - smoothstep(0.0, 1.0, outside);
        ivec3 cnt = ivec3(v.counts.xyz + 0.5);
        vec3 g = clamp(t, vec3(0.0), vec3(1.0)) * vec3(cnt - 1);
        ivec3 g0 = ivec3(floor(g));
        vec3 f = g - vec3(g0);
        // Plain trilinear: smoothstep-style weights flatten near each probe, so
        // every probe owns a plateau and the field reads as a grid.
        mat3 local_to_world = transpose(mat3(v.world_to_local));
        vec3 center = -(local_to_world * v.world_to_local[3].xyz);
        vec3 size = max(v.size_base.xyz, vec3(1e-4));
        vec3 sum = vec3(0.0);
        float wsum = 0.0;
        for (int i = 0; i < 8; ++i) {
            ivec3 off = ivec3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
            ivec3 gc = clamp(g0 + off, ivec3(0), cnt - 1);
            vec3 tw = mix(vec3(1.0) - f, f, vec3(off));
            float w = tw.x * tw.y * tw.z;
            // Backface weight (DDGI): a probe behind this surface (under a floor
            // that sits just above the volume's bottom layer) sees only the
            // back of it, so it must not darken the surface it can't see.
            vec3 gn = vec3(gc) / max(vec3(cnt - 1), vec3(1.0));
            vec3 to_probe = center + local_to_world * ((gn - 0.5) * size) - world_pos;
            // A step across the surface plane, not an angle falloff: weighting
            // front probes by angle changes non-linearly between probes and
            // shows each one as a bump.
            float facing = dot(to_probe, to_probe) > 1e-8 ? smoothstep(-0.15, 0.05, dot(normalize(to_probe), n)) : 1.0;
            w *= facing + 1e-3;
            uint idx = uint(v.size_base.w) + uint(gc.x + gc.y * cnt.x + gc.z * cnt.x * cnt.y);
            sum += w * sh_eval(idx, n);
            wsum += w;
        }
        return wsum > 1e-5 ? max(sum / wsum, vec3(0.0)) * fade : vec3(0.0);
    }
    return vec3(0.0);
}

// Returns 1.0 fully lit, 0.0 fully shadowed. `light.spot` packs
// (cos_inner, shadow_base_layer, bias, view_count); base < 0 = no shadow.
// `nrm` is the surface normal and `ldir` the direction to the light (both
// unit), used to slope-scale the depth bias so grazing receivers don't alias.
// Viewport mode 7 (Shadow layers): the atlas layer / cascade that shadowed the
// first shadow-casting light at this pixel, and its result. -1 = none.
int g_dbg_layer = -1;
float g_dbg_lit = -1.0;
// Geometric (vertex) normal facing the viewer, set in main(); shadow_factor's
// normal-offset uses it so a normal map doesn't jitter the shadow lookup.
vec3 g_geo_n = vec3(0.0, 1.0, 0.0);
vec3 g_shadow_plane_n = vec3(0.0, 1.0, 0.0);

// Bilinear shadow visibility with a separate receiver depth at each texel.
// Sampling exactly at centres makes the comparison sampler read one depth.
float shadow_plane_bilinear(vec2 uv,int layer,float reference,vec2 receiver_uv,vec2 gradient,vec2 size) {
    vec2 p=uv*size-0.5,base=floor(p),f=fract(p);float sum=0.0;
    for(int y=0;y<2;++y) for(int x=0;x<2;++x) {
        vec2 offset=vec2(x,y);
        vec2 weight=mix(vec2(1)-f,f,offset);
        vec2 at=(clamp(base+offset,vec2(0),size-1.0)+0.5)/size;
        sum+=weight.x*weight.y*texture(u_shadow,vec4(at,float(layer),reference+dot(gradient,at-receiver_uv)));
    }
    return sum;
}

float shadow_factor(Light light, vec3 world_pos, vec3 nrm, vec3 ldir) {
    int base = int(light.spot.y);
    if (base < 0) {
        return 1.0;
    }
    float bias = light.spot.z;
    // spot.w packs the shadow view count; its sign is the filter mode
    // (positive = soft/PCF, negative = hard/single tap).
    bool soft = light.spot.w >= 0.0;
    int vcount = int(abs(light.spot.w) + 0.5);
    int kind = int(light.pos_kind.w + 0.5);
    int sub = 0;
    if (kind == 0 && vcount > 1) {
        // Cascaded directional. View-space depth gives a starting cascade, but
        // the sphere-fit cascades don't align exactly with the depth slices, so
        // a boundary fragment can project just outside the chosen cascade's map
        // and read the lit border -> a bright seam. Walk up to the first cascade
        // whose projected uv sits inside [margin, 1-margin] (margin = PCF kernel
        // radius, so taps never reach the border). Coarser cascades enclose the
        // finer ones, so this always converges.
        float depth = -(VIEW_MATRIX * vec4(world_pos, 1.0)).z;
        int start = vcount - 1;
        for (int c = 0; c < vcount && c < 4; ++c) {
            if (depth < frame.cascade_splits[c]) { start = c; break; }
        }
        float margin = frame.misc.z * float(2 + 1); // PCF radius R=2, +1 slack
        sub = vcount - 1;
        for (int c = start; c < vcount && c < 4; ++c) {
            vec4 cc = frame.shadow_vp[base + c] * vec4(world_pos, 1.0);
            if (cc.w <= 0.0) { sub = c; continue; }
            vec2 cuv = (cc.xy / cc.w) * 0.5 + 0.5;
            sub = c;
            if (cuv.x >= margin && cuv.x <= 1.0 - margin
                && cuv.y >= margin && cuv.y <= 1.0 - margin) {
                break;
            }
        }
    } else if (vcount == 6) {
        // Point light: pick the cube face from the dominant axis.
        vec3 fl = world_pos - light.pos_kind.xyz;
        vec3 a = abs(fl);
        if (a.x >= a.y && a.x >= a.z) sub = fl.x > 0.0 ? 0 : 1;
        else if (a.y >= a.z) sub = fl.y > 0.0 ? 2 : 3;
        else sub = fl.z > 0.0 ? 4 : 5;
    }
    int layer = base + sub;
    if (g_dbg_layer < 0) {
        g_dbg_layer = layer;
    }
    mat4 svp = frame.shadow_vp[layer];
    vec4 lc0 = svp * vec4(world_pos, 1.0);
    if (lc0.w <= 0.0) {
        return 1.0;
    }
    vec2 ndc0 = lc0.xy / lc0.w;
    // Size offsets from actual shadow texels, independently of PCF softness.
    // Normal offset handles grazing receiver self-shadowing. Multiplying a
    // second, large depth offset by slope moved the test through nearby walls.
    float bias_texel = 1.0 / float(textureSize(u_shadow,0).x);
    vec3 side = normalize(cross(ldir, abs(ldir.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
    vec4 lc_side = svp * vec4(world_pos + side * 0.05, 1.0);
    float uv_per_m = length(lc_side.xy / max(lc_side.w, 1e-4) - ndc0) * 0.5 / 0.05;
    float texel_m = bias_texel / max(uv_per_m, 1e-6);
    // frame.water.w = depth-bias scale from the scene (Environment
    // shadow_bias_scale), frame.refl_blend2.z = normal-offset strength
    // (Environment shadow_normal_offset); 0 on passes that don't set them ->
    // the 1500 / 2.0 defaults.
    float bias_scale = frame.water.w > 0.0 ? frame.water.w : 1500.0;
    float noff_scale = frame.refl_blend2.z > 0.0 ? frame.refl_blend2.z : 2.0;
    float ngl = dot(g_geo_n, ldir);
    // Point/spot taps compare at their actual texel centres below, so no
    // receiver normal displacement is needed. Keep directional calibration.
    float noff_m = kind==0 ? texel_m*noff_scale*sqrt(max(1.0-ngl*ngl,0.0)) : 0.0;
    vec3 spos = world_pos + (ngl >= 0.0 ? g_geo_n : -g_geo_n) * noff_m;
    vec4 lc = svp * vec4(spos, 1.0);
    if (lc.w <= 0.0) {
        return 1.0;
    }
    vec3 proj = lc.xyz / lc.w;
    vec2 uv = proj.xy * 0.5 + 0.5;
    // Default light bias .002 = half a texel plus a small precision margin.
    // Preserve the environment multiplier (1500 = 1x), but do not reinterpret
    // the default as several texels and then amplify it again at grazing angles.
    float bias_m = texel_m * (0.125 + max(bias,0.0) / 0.004) * (bias_scale / 1500.0);
    if(kind==0) {
        float ndotl=clamp(dot(nrm,ldir),0.0,1.0);
        float slope=sqrt(max(1.0-ndotl*ndotl,0.0))/max(ndotl,0.2);
        bias_m=texel_m*(bias+0.0008)*bias_scale*(1.0+slope);
    }
    vec4 lc_bias = svp * vec4(spos + ldir * bias_m, 1.0);
    float ref = lc_bias.w > 1e-4 ? lc_bias.z / lc_bias.w : proj.z;
    // Beyond the far plane there is no occluder. For directional/spot, a uv
    // outside the map means outside the light's frustum -> unshadowed. Point
    // lights tile 6 overlapping faces, so a fragment is always covered by its
    // selected face; per-tap clamping below keeps PCF off the atlas border
    // (which reads as lit and would draw a seam between faces).
    bool is_point = (vcount == 6);
    if (ref > 1.0) {
        return 1.0;
    }
    if (!is_point && (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)) {
        return 1.0;
    }
    if(kind!=0) {
        // Compare each physical texel against its own receiver-plane depth.
        // Hardware bilinear comparison uses ONE reference for all four texels;
        // compensating only the outer PCF taps still needs a slope bias and
        // causes either acne or detached contact shadows.
        vec3 tangent=normalize(cross(g_shadow_plane_n,abs(g_shadow_plane_n.y)<0.99 ? vec3(0,1,0) : vec3(1,0,0)));
        vec3 bitangent=cross(g_shadow_plane_n,tangent);
        vec4 ta=svp*vec4(tangent,0),tb=svp*vec4(bitangent,0);
        vec3 da=(ta.xyz-proj*ta.w)/lc.w,db=(tb.xyz-proj*tb.w)/lc.w;
        float det=da.x*db.y-da.y*db.x;
        vec2 gradient=abs(det)>1e-12
            ? 2.0*vec2(da.z*db.y-db.z*da.y,da.x*db.z-db.x*da.z)/det : vec2(0);
        vec2 size=vec2(textureSize(u_shadow,0).xy);
        if(!soft || frame.misc.z<=0.0) return shadow_plane_bilinear(uv,layer,ref,uv,gradient,size);
        if(abs(frame.misc.z*size.x-1.0)<0.001) {
            // A 5x5 bilinear box shares a 6x6 set of physical texels. Evaluate
            // those 36 samples once, retaining continuous edge weights as the
            // receiver moves; equivalent to 100 individual comparisons.
            vec2 p=uv*size-0.5,base=floor(p),f=fract(p);
            float sum=0.0;
            for(int y=-2;y<=3;++y) for(int x=-2;x<=3;++x) {
                float wx=x==-2 ? 1.0-f.x : x==3 ? f.x : 1.0;
                float wy=y==-2 ? 1.0-f.y : y==3 ? f.y : 1.0;
                vec2 at=(clamp(base+vec2(x,y),vec2(0),size-1.0)+0.5)/size;
                sum+=wx*wy*texture(u_shadow,vec4(at,float(layer),ref+dot(gradient,at-uv)));
            }
            return sum/25.0;
        }
        float sum=0.0;
        for(int y=-2;y<=2;++y) for(int x=-2;x<=2;++x) {
            vec2 at=clamp(uv+vec2(x,y)*frame.misc.z,vec2(0),vec2(1));
            sum+=shadow_plane_bilinear(at,layer,ref,uv,gradient,size);
        }
        return sum/25.0;
    }
    // Directional lights retain their existing normal-offset / PCF policy.
    if (!soft) return texture(u_shadow,vec4(clamp(uv,vec2(0),vec2(1)),float(layer),ref));
    float sum=0.0;
    for(int y=-2;y<=2;++y) for(int x=-2;x<=2;++x) {
        vec2 at=clamp(uv+vec2(x,y)*frame.misc.z,vec2(0),vec2(1));
        sum+=texture(u_shadow,vec4(at,float(layer),ref));
    }
    return sum/25.0;
}

layout(set = 1, binding = 0) uniform sampler2D t_albedo;
layout(set = 1, binding = 1) uniform sampler2D t_normal;
layout(set = 1, binding = 2) uniform sampler2D t_orm; // R occlusion, G roughness, B metallic
layout(set = 1, binding = 3) uniform sampler2D t_emission;


// Split AO / Roughness / Metallic maps. Each multiplies the matching packed-ORM
// channel; the 1×1 default is white, so an unassigned slot is a no-op.
layout(set = 1, binding = 13) uniform sampler2D t_ao;
layout(set = 1, binding = 14) uniform sampler2D t_roughness;
layout(set = 1, binding = 15) uniform sampler2D t_metallic;
// Height / displacement map for parallax occlusion mapping (white = flat).
layout(set = 1, binding = 16) uniform sampler2D t_displacement;
// Detail layer (white / flat-normal 1x1 defaults) and the translucency map.
layout(set = 1, binding = 17) uniform sampler2D t_detail_albedo;
layout(set = 1, binding = 18) uniform sampler2D t_detail_normal;
layout(set = 1, binding = 19) uniform sampler2D t_detail_orm;
layout(set = 1, binding = 20) uniform sampler2D t_detail_mask;
layout(set = 1, binding = 21) uniform sampler2D t_translucency;

// Extended texture slots (bindings 5-12).
layout(set = 1, binding = 5) uniform sampler2D t_opacity;
layout(set = 1, binding = 6) uniform sampler2D t_emission_mask;
layout(set = 1, binding = 7) uniform sampler2D t_matcap0;
layout(set = 1, binding = 8) uniform sampler2D t_matcap0_mask;
layout(set = 1, binding = 9) uniform sampler2D t_matcap1;
layout(set = 1, binding = 10) uniform sampler2D t_matcap1_mask;
layout(set = 1, binding = 11) uniform sampler2D t_matcap2;
layout(set = 1, binding = 12) uniform sampler2D t_matcap2_mask;

layout(push_constant) uniform Push {
    // The model matrix comes from the instance stream; column 0 here holds the
    // 128-bit mask of lights that can reach this draw (uint bits).
    mat4 model;
    vec4 base_color;
    vec4 emission;
    vec4 params0; // metallic, roughness, toon_steps, pbr_toon_blend
    vec4 params1; // alpha_cutoff, normal_strength, occlusion_strength, _
} pc;

layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec2 v_uv;
layout(location = 3) in vec4 v_color;
layout(location = 4) in vec4 v_tangent;
layout(location = 5) in vec2 v_uv1;
// Baked-lightmap layer (or the lightmap-UV preview's resolution), see standard.vert.
layout(location = 6) flat in float v_lightmap;
// Current/previous-frame clip positions (from the vertex stage).
layout(location = 7) in vec4 v_cur_clip;
layout(location = 8) in vec4 v_prev_clip;
// Per-instance material properties (read only when FEAT_PER_INSTANCE).
layout(location = 9) in vec4 v_instance_color;
layout(location = 10) in vec4 v_instance_param;

layout(location = 0) out vec4 o_color;
// Deferred-SSR G-buffer (gbuf pipeline variants only): the specular reflectance
// weight (Fresnel*ao*spec, the factor multiplying environment radiance) in rgb,
// surface roughness in alpha. The fullscreen SSR resolve pass reads this to do
// current-frame screen-space reflections without the forward pass's 1-frame lag.
layout(location = 1) out vec4 o_gbuf;
// Motion vectors (velocity variants): clip-space reprojection difference.
layout(location = 2) out vec2 o_velocity;
// Reactive mask (velocity variants): 1 = transparent/reactive pixel (the
// upscaler resets temporal history here), 0 = opaque.
layout(location = 3) out float o_reactive;

const float PI = 3.14159265359;

// Narkowicz ACES filmic tonemap: rolls HDR highlights off to [0,1] so bright
// surfaces (a close point light, stacked ambient + baked bounce) show detail
// instead of clipping to flat white. Operates in linear; the sRGB swapchain
// applies gamma on write.
vec3 tonemap_aces(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// Combine one matcap layer with the colour beneath it. `t` is the layer's
// mask*strength weight; `mode` selects 0 add / 1 multiply / 2 replace.
vec3 blend_matcap(vec3 base, vec3 mc, float t, float mode) {
    if (mode > 1.5) return mix(base, mc, t);          // replace
    if (mode > 0.5) return mix(base, base * mc, t);   // multiply
    return base + mc * t;                             // add
}

// Per-pixel post from the blended Volume profile: exposure → grading → tonemap →
// vignette. (Chromatic aberration + bloom need a fullscreen pass; follow-up.)
vec3 apply_postfx(vec3 color, vec2 fragcoord) {
    // HDR output (debug.w): the fullscreen post pass does exposure/grade/tonemap/
    // bloom/vignette, so surfaces output linear radiance unchanged.
    if (frame.debug.w > 0.5) {
        return color;
    }
    color *= exp2(frame.postfx0.y);
    if (frame.postfx1.w > 0.5) {
        color *= exp2(frame.postfx0.z);
        float temp = frame.postfx1.y, tint = frame.postfx1.z;
        color.r *= 1.0 + temp * 0.2 + tint * 0.1;
        color.g *= 1.0 - tint * 0.2;
        color.b *= 1.0 - temp * 0.2 + tint * 0.1;
        color = (color - 0.18) * frame.postfx0.w + 0.18;
        float l = dot(color, vec3(0.2126, 0.7152, 0.0722));
        color = mix(vec3(l), color, frame.postfx1.x);
        color = max(color, vec3(0.0));
    }
    int mode = int(frame.postfx0.x + 0.5);
    if (mode == 1) {
        color = color / (color + vec3(1.0));
    } else if (mode == 2) {
        color = tonemap_aces(color);
    }
    if (frame.postfx2.x > 0.5) {
        vec2 uv = fragcoord / vec2(max(frame.postfx2.w, 1.0), max(frame.postfx3.w, 1.0));
        float dist = length(uv - 0.5) * 1.41421356;
        float sm = max(frame.postfx2.z, 1e-3);
        float mask = clamp((dist - (1.0 - sm)) / sm, 0.0, 1.0) * frame.postfx2.y;
        color = mix(color, frame.postfx3.xyz, mask);
    }
    return color;
}

// Radius (m) every punctual light is treated as for specular highlights.
const float LIGHT_SOURCE_RADIUS = 0.05;

float d_ggx(float NdotH, float a) {
    float a2 = a * a;
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-6);
}

float g_smith(float NdotV, float NdotL, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    float gv = NdotV / (NdotV * (1.0 - k) + k);
    float gl = NdotL / (NdotL * (1.0 - k) + k);
    return gv * gl;
}

vec3 f_schlick(float VdotH, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
}

// Karis' analytic split-sum environment BRDF (the "EnvBRDFApprox"): the
// scale+bias for the prefiltered environment specular, so reflections are
// roughness- and view-correct and DON'T spike to a harsh mirror at grazing the
// way a raw Fresnel term does. Returns (scale, bias): spec = f0*scale + bias.
vec2 env_brdf_approx(float roughness, float NoV) {
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
    return vec2(-1.04, 1.04) * a004 + r.zw;
}

// Area lights (LTC rectangle / disc, optional emission texture).
#include "area_light.glsl"

// Octahedral encode of a unit vector into [0,1]^2 (lossless enough at 16f). The
// SSR G-buffer stores the view-space shading normal this way so the resolve pass
// gets the EXACT normal (no depth reconstruction = no bent reflection rays).
vec2 oct_encode(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    vec2 e = n.z >= 0.0
        ? n.xy
        : (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    return e * 0.5 + 0.5;
}

// Roughness-aware Fresnel for the ambient/indirect term (rougher surfaces
// reflect less at grazing angles, so more energy stays in the diffuse).
vec3 f_schlick_rough(float cosT, vec3 f0, float rough) {
    vec3 fmax = max(vec3(1.0 - rough), f0);
    return f0 + (fmax - f0) * pow(clamp(1.0 - cosT, 0.0, 1.0), 5.0);
}

// Parallax occlusion mapping: march the tangent-space view ray through the
// height field and return the UV of the first intersection, giving a flat
// surface apparent depth without extra geometry. `uv` is the (tiled) start
// coordinate, `vt` the tangent-space view direction (surface->eye), `scale` the
// max UV shift at full height. Height map convention: white = peak (depth 0).
vec2 parallax_uv(vec2 uv, vec3 vt, float scale) {
    const int MAX_L = 32;
    // More layers at grazing angles (where the parallax shift is largest).
    float n = mix(float(MAX_L), 8.0, clamp(abs(vt.z), 0.0, 1.0));
    float layer_d = 1.0 / n;
    vec2 step_uv = (vt.xy / max(abs(vt.z), 0.1)) * scale / n;
    vec2 cur = uv;
    float cur_d = 0.0;
    float h = 1.0 - texture(t_displacement, cur).r;
    for (int i = 0; i < MAX_L; ++i) {
        if (cur_d >= h) break;
        cur -= step_uv;
        h = 1.0 - texture(t_displacement, cur).r;
        cur_d += layer_d;
    }
    // Soft occlusion: interpolate between the last two steps across the surface.
    vec2 prev = cur + step_uv;
    float after = h - cur_d;
    float before = (1.0 - texture(t_displacement, prev).r) - (cur_d - layer_d);
    float w = after / (after - before);
    return mix(cur, prev, clamp(w, 0.0, 1.0));
}

// Box-projected parallax (Unity-style) for one reflection probe: reproject the
// reflection ray onto the probe's box so flat surfaces line up with the
// captured environment instead of treating it as infinitely distant.
// Only surfaces INSIDE the box are projected: for a surface outside it the
// ray-box hit lands on the wrong face, so the raw direction is used instead
// (REFLECTION_PROBES_MATH.md 6.4). The box center may differ from the capture
// point (Lagarde); the re-vector anchors on the capture position. A small
// tolerance keeps surfaces flush with a box face (a mirror on the wall, the
// floor) counted as inside.
vec3 probe_dir(vec3 R, vec3 capture, vec3 center, vec3 extents, bool project) {
    if (!project) return R;
    vec3 bmin = center - extents;
    vec3 bmax = center + extents;
    vec3 eps = extents * 0.02 + 0.02;
    bool inside = all(greaterThanEqual(v_world_pos, bmin - eps))
               && all(lessThanEqual(v_world_pos, bmax + eps));
    if (!inside) return R;
    vec3 invR = 1.0 / R;
    vec3 t1 = (bmax - v_world_pos) * invR;
    vec3 t2 = (bmin - v_world_pos) * invR;
    vec3 tmax = max(t1, t2);
    // A fragment just outside the box (eps tolerance) can produce a negative
    // exit distance; clamp so the hit never lands behind it.
    float dist = max(min(min(tmax.x, tmax.y), tmax.z), 0.0);
    return v_world_pos + R * dist - capture;
}

// Probe influence: 0 outside the box, ramping to 1 across the `blend` shell
// just inside its faces (the Reflection Probe's blend distance).
float probe_influence(vec3 p, vec3 center, vec3 extents, float blend) {
    vec3 d = extents - abs(p - center);
    float inner = min(d.x, min(d.y, d.z));
    if (blend <= 1e-4) return inner >= 0.0 ? 1.0 : 0.0;
    return clamp(inner / blend, 0.0, 1.0);
}

#define ZEST_WATER_FRAME
#include "water_optics.glsl"
#include "water_detail.glsl"
#include "water_volume.glsl"

// View-space distance to the opaque scene behind framebuffer pixel `frag`.
float water_scene_dist(vec2 frag) {
    ivec2 size = textureSize(u_opaque_depth, 0);
    float d = texelFetch(u_opaque_depth, clamp(ivec2(frag), ivec2(0), size - 1), 0).r;
    if (d >= 1.0) {
        return 1e6;
    }
    vec2 suv = frag / vec2(max(frame.postfx2.w, 1.0), max(frame.postfx3.w, 1.0));
    vec4 p = frame.inv_proj * vec4(suv.x * 2.0 - 1.0, 1.0 - suv.y * 2.0, d, 1.0);
    return abs(p.w) > 1e-8 ? length(p.xyz / p.w) : 1e6;
}

#include "water_caustics.glsl"


void main() {
    // Default the SSR G-buffer to "not a reflector"; the reflection block below
    // overwrites it for lit surfaces. Keeps early debug returns valid.
    o_gbuf = vec4(0.0);
    // Lightmap-UV checker preview: visualise per-object texel density. params1.w
    // carries this object's lightmap resolution (texels/side) in preview mode; a
    // fixed 8-texel cell means low-res objects show big squares, high-res small.
    if (frame.debug.x > 0.5) {
        float res = v_lightmap;
        if (res < 1.0) {
            o_color = vec4(0.12, 0.12, 0.14, 1.0); // not lightmapped (non-static)
        } else {
            vec2 c = floor(v_uv1 * res / 8.0);
            float check = mod(c.x + c.y, 2.0);
            o_color = vec4(mix(vec3(0.12, 0.12, 0.15), vec3(0.95, 0.55, 0.2), check), 1.0);
        }
        return;
    }

    vec3 N = normalize(v_normal);
    if (!gl_FrontFacing) {
        N = -N;
    }
    g_geo_n = N;
    // The shadow receiver is the rasterized triangle plane, not its smoothed
    // or normal-mapped shading normal. Evaluate derivatives before light loops.
    vec3 plane_n=cross(dFdx(v_world_pos),dFdy(v_world_pos));
    g_shadow_plane_n=dot(plane_n,plane_n)>1e-20 ? normalize(plane_n) : N;

    // Parallax occlusion mapping displaces the mesh UV along the tangent-space
    // view ray through the height field, so every map below samples the same
    // apparent surface point. Off (scale 0) leaves the UV untouched. The shift
    // is computed in albedo-tiled space then mapped back to mesh UV so the
    // per-map tiling/offset below stays coherent.
    vec2 v_uv_d = v_uv;
    if (fx.parallax.x > 0.0) {
        vec3 Tg = normalize(v_tangent.xyz - N * dot(v_tangent.xyz, N));
        vec3 Bg = cross(N, Tg) * v_tangent.w;
        vec3 to_eye = CAMERA_POS.xyz - v_world_pos;
        vec3 vt = normalize(vec3(dot(to_eye, Tg), dot(to_eye, Bg), dot(to_eye, N)));
        vec2 a0 = v_uv * fx.albedo_st.xy + fx.albedo_st.zw;
        vec2 a1 = parallax_uv(a0, vt, fx.parallax.x);
        v_uv_d = v_uv + (a1 - a0) / max(fx.albedo_st.xy, vec2(1e-4));
    }

    // Per-texture UV transform (tiling*uv + offset) plus the animated scroll, on
    // the parallax-displaced mesh UV. Albedo's UV is the shared `uv` used by the
    // opacity + mask samplers below; normal/orm/emission get their own tiling.
    vec2 uv = v_uv_d * fx.albedo_st.xy + fx.albedo_st.zw + fx.scroll.xy * frame.misc.x;
    vec2 uv_normal = v_uv_d * fx.normal_st.xy + fx.normal_st.zw;
    vec2 uv_orm = v_uv_d * fx.orm_st.xy + fx.orm_st.zw;
    vec2 uv_emission = v_uv_d * fx.emission_st.xy + fx.emission_st.zw;

    vec4 base_color = FEAT_PER_INSTANCE ? v_instance_color : pc.base_color;
    vec4 params0 = FEAT_PER_INSTANCE ? v_instance_param : pc.params0;
    vec4 albedo = texture(t_albedo, uv) * base_color * v_color;
    albedo.a *= texture(t_opacity, uv).r; // dedicated opacity map (default white)
    if ((ALPHA_MODE == 1u || ALPHA_MODE == 3u) && albedo.a < pc.params1.x) {
        discard;
    }

    // Detail layer: a second albedo / normal / ORM set at its own tiling, masked
    // by t_detail_mask (sampled with the base UV). All strengths 0 = off.
    bool detail_on = fx.detail.x > 0.0 || fx.detail.y > 0.0 || fx.detail.z > 0.0;
    vec2 uv_detail = v_uv_d * fx.detail_st.xy + fx.detail_st.zw;
    float detail_mask = detail_on ? texture(t_detail_mask, uv).r : 0.0;
    if (fx.detail.x > 0.0) {
        vec3 d = texture(t_detail_albedo, uv_detail).rgb;
        albedo.rgb=material_detail_albedo(albedo.rgb,d,fx.detail.x*detail_mask,fx.detail.w);
    }

    if (FEAT_NORMAL_MAP || fx.detail.y > 0.0) {
        vec3 T = normalize(v_tangent.xyz - N * dot(v_tangent.xyz, N));
        vec3 B = cross(N, T) * v_tangent.w;
        vec3 tn = vec3(0.0, 0.0, 1.0);
        if (FEAT_NORMAL_MAP) {
            tn = material_decode_normal(texture(t_normal, uv_normal).xyz);
            tn.xy *= pc.params1.y;
        }
        if (fx.detail.y > 0.0) {
            vec3 dn = material_decode_normal(texture(t_detail_normal, uv_detail).xyz);
            // Whiteout-style blend: add the slopes, keep the base's z.
            tn = vec3(tn.xy + dn.xy * (fx.detail.y * detail_mask), tn.z);
        }
        N = normalize(mat3(T, B, N) * normalize(tn));
    }

    vec3 orm = texture(t_orm, uv_orm).rgb;
    if (fx.detail.z > 0.0) {
        // The detail ORM multiplies the base channels (white = no change).
        orm *= mix(vec3(1.0), texture(t_detail_orm, uv_detail).rgb, clamp(fx.detail.z * detail_mask, 0.0, 1.0));
    }
    // Split AO/Roughness/Metallic maps multiply their packed-ORM channel (white
    // default = no-op), each with an optional invert (e.g. smoothness->roughness).
    float ao_s = texture(t_ao, uv_orm).r;
    float rough_s = texture(t_roughness, uv_orm).r;
    float metal_s = texture(t_metallic, uv_orm).r;
    ao_s = mix(ao_s, 1.0 - ao_s, fx.orm_invert.x);
    rough_s = mix(rough_s, 1.0 - rough_s, fx.orm_invert.y);
    metal_s = mix(metal_s, 1.0 - metal_s, fx.orm_invert.z);
    float ao = mix(1.0, orm.r * ao_s, pc.params1.z);
    // Toon rim strength now comes from the FX block (so occlusion stays usable).
    float rim_strength = FEAT_TOON ? fx.toon.x : 0.0;
    float roughness = clamp(orm.g * rough_s * params0.y, 0.045, 1.0);
    float metallic = clamp(orm.b * metal_s * params0.x, 0.0, 1.0);

    #include "water_surface.glsl"

    vec3 V = normalize(CAMERA_POS.xyz - v_world_pos);
    float NdotV = max(dot(N, V), 1e-4);
    vec3 f0 = mix(vec3(0.04), albedo.rgb, metallic);
    vec3 diffuse_color = albedo.rgb * (1.0 - metallic);
    if (FEAT_WATER) {
        // The see-through share comes from the refraction instead.
        f0 = vec3(water_f0);
    }
    // Transmission: the share of the diffuse that lets the scene behind show
    // (needs the opaque copy, i.e. the material draws with Scene Colour Read).
    float transmit_amt = (!FEAT_WATER && fx.trans1.w > 0.0 && frame.water.y > 0.5) ? clamp(fx.trans1.w, 0.0, 1.0) : 0.0;
    // The translucency / thickness map (binding 21) also feeds the transmission:
    // green = transmission factor, red = volume thickness (when flagged as such).
    float transmit_thick = 1.0;
    if (transmit_amt > 0.0 && (fx.trans0.w > 0.5 || fx.trans3.z > 0.5)) {
        vec4 tm = texture(t_translucency, uv);
        if (fx.trans3.z > 0.5) {
            transmit_amt *= tm.g;
        }
        if (fx.trans0.w > 0.5) {
            transmit_thick = tm.r;
        }
    }
    diffuse_color *= 1.0 - transmit_amt;
    // Translucency: light arriving from behind a thin surface (leaf, paper, cloth).
    bool translucent = fx.trans0.x > 0.0;
    float trans_map = 1.0;
    if (translucent) {
        trans_map = texture(t_translucency, uv).r;
        trans_map = mix(trans_map, 1.0 - trans_map, fx.trans0.w);
    }
    vec3 trans_acc = vec3(0.0);

    // Accumulate every active scene light. The CPU guarantees at least one
    // (a directional fallback when the scene has no light objects).
    int light_count = int(frame.misc.y + 0.5);
    // Surfaces under water catch caustic light (seen from above too), and
    // lights under a pool throw it out onto what's above.
    bool caustics = !FEAT_WATER && frame.water.z > 0.5;
    vec3 color = vec3(0.0);
    bool has_lightmap = v_lightmap >= 0.0;
    uvec4 light_mask = floatBitsToUint(pc.model[0]);
    vec3 voxel_emit = vec3(0.0); // FluxVoxel moving emitters (indirect, kind 3)
    for (int w = 0; w < 4; ++w) {
        uint bits = light_mask[w];
        while (bits != 0u) {
            int i = w * 32 + findLSB(bits);
            bits &= bits - 1u;
            if (i >= light_count || i >= MAX_LIGHTS) {
                bits = 0u;
                continue;
            }
            Light light = frame.lights[i];
            int kind = int(light.pos_kind.w + 0.5);
            if (kind == 3) {
                // The voxel grid's injection kernel per pixel: a probe grid
                // shows its cells under an orb's sharp falloff.
                vec3 d = light.pos_kind.xyz - v_world_pos;
                float dist = length(d);
                float f = clamp(1.0 - dist / max(light.dir_range.w, 1e-3), 0.0, 1.0);
                vec3 Le = dist > 1e-4 ? d / dist : N;
                voxel_emit += light.color.rgb * (f * f * max(0.11141 + 0.25465 * dot(N, Le), 0.0));
                continue;
            }
            if (kind >= 4) {
                // Area light (rectangle 4, disc 5): two slots; 6 is the continuation. Shadow-less,
                // applies to lightmapped and dynamic surfaces alike (a bake has no area lights).
                if (kind <= 5 && i + 1 < MAX_LIGHTS) {
                    vec3 a_diff, a_spec;
                    float a_fd;
                    area_light_eval(i, v_world_pos, N, V, NdotV, roughness, f0, a_diff, a_spec, a_fd);
                    vec3 a_kd = max(vec3(1.0) - (f0 * env_brdf_approx(roughness, NdotV).x + vec3(env_brdf_approx(roughness, NdotV).y)), vec3(0.0));
                    vec3 a_lit = a_kd * diffuse_color * a_diff;
                    if (FEAT_TOON) {
                        // Cel ramp on the form factor, like N.L for a point light.
                        float steps = max(params0.z, 2.0);
                        float scaled = clamp(a_fd * 2.0, 0.0, 1.0) * steps;
                        float lower = floor(scaled);
                        float sm = clamp(fx.toon.y, 0.01, 0.49);
                        float banded = (lower + smoothstep(0.5 - sm, 0.5 + sm, scaled - lower)) / steps;
                        float smooth_f = clamp(a_fd * 2.0, 0.0, 1.0);
                        vec3 toon_lit = a_kd * diffuse_color * (a_diff / max(a_fd, 1e-5)) * (banded * 0.5);
                        a_lit = mix(a_lit, toon_lit, clamp(params0.w, 0.0, 1.0));
                    }
                    color += a_lit + a_spec;
                }
                continue;
            }
            // Baked/Mixed light (flagged spot.y<0 & spot.w>0): its energy is already
            // in the lightmap, so skip it for lightmapped (static) surfaces, but DO
            // apply it to non-lightmapped objects (dynamic / added-after-bake) so they
            // aren't black in a baked scene. Realtime lights are never skipped.
            bool light_baked = light.spot.y < 0.0 && light.spot.w > 0.5;
            bool baked_here = light_baked && has_lightmap;
            if (baked_here && !caustics) {
                continue;
            }

            // Direction to the light + distance-based attenuation.
            vec3 L;
            float attenuation = 1.0;
            // Lights are small spheres, not points: their angular size widens
            // the specular lobe (Karis 2013 sphere-light roughness), which keeps
            // a mirror-smooth highlight finite instead of overflowing half floats.
            float a_spec = roughness * roughness;
            float a_light = a_spec + 0.00465;
            if (kind == 0) {
                L = normalize(-light.dir_range.xyz);
            } else {
                vec3 to_light = light.pos_kind.xyz - v_world_pos;
                float dist = length(to_light);
                a_light = min(a_spec + LIGHT_SOURCE_RADIUS / (2.0 * max(dist, 1e-3)), 1.0);
                L = (dist > 1e-5) ? to_light / dist : N;
                // Smooth inverse-square-ish falloff clamped to the range.
                attenuation = realtime_distance_falloff(dist,light.dir_range.w);
                if (kind == 2) {
                    // Spot cone: full inside the inner angle, smooth to the outer.
                    float cos_dir = dot(normalize(light.dir_range.xyz), -L);
                    float cos_outer = light.color.w;
                    float cos_inner = light.spot.x;
                    attenuation *= realtime_spot_falloff(cos_dir,cos_inner,cos_outer);
                }
            }

            vec3 radiance = light.color.rgb * attenuation;
            if (radiance == vec3(0.0)) {
                continue;
            }
            float caustic_gain = caustics ? water_caustic_gain(v_world_pos, light.dir_range.xyz, light.pos_kind.xyz, kind != 0) : 1.0;
            // A baked light is already in the lightmap: only the caustics'
            // change to it is added here.
            if (baked_here) {
                if (abs(caustic_gain - 1.0) < 1e-3) {
                    continue;
                }
                caustic_gain -= 1.0;
            }

            // Facing away: every term below is scaled by N·L (the toon ramp too),
            // so skip the BRDF and the shadow-map lookups.
            float NdotL = dot(N, L);
            if (translucent && !baked_here && NdotL < 0.2) {
                // Backlit: the light bent by the normal, seen along the view ray.
                vec3 Lt = normalize(L + N * fx.trans0.z);
                float tv = pow(clamp(dot(V, -Lt), 0.0, 1.0), max(fx.trans0.y, 0.1));
                float back = smoothstep(0.2, -0.3, NdotL);
                if (tv * back > 1e-3) {
                    // The shadow lookup is offset to the light's side of the surface.
                    trans_acc += radiance * shadow_factor(light, v_world_pos, -N, L) * (tv * back);
                }
            }
            if (NdotL <= 0.0) {
                continue;
            }
            vec3 H = normalize(V + L);
            float NdotH = max(dot(N, H), 0.0);
            float VdotH = max(dot(V, H), 0.0);

            // Cook-Torrance specular + energy-conserving Lambertian diffuse: the
            // Fresnel term F is the specular reflectance kS, so the diffuse keeps
            // only the remaining energy kD = (1 - F). (Metallic is already folded
            // into diffuse_color, which is 0 for pure metals.)
            vec3 F = f_schlick(VdotH, f0);
            float lobe = a_spec / a_light;
            vec3 spec = d_ggx(NdotH, a_light) * (lobe * lobe)
                * g_smith(NdotV, NdotL, roughness)
                * F
                / max(4.0 * NdotV * NdotL, 1e-4);
            vec3 kd = vec3(1.0) - F;
            vec3 lit = (kd * diffuse_color / PI + spec) * NdotL;

            if (FEAT_TOON) {
                // Cel ramp: quantize N·L into `steps` bands with a soft edge at each
                // boundary (a clean toon terminator without hard aliasing). `steps`
                // controls band count, params0.w blends toon<->smooth-PBR.
                float steps = max(params0.z, 2.0);
                float scaled = clamp(NdotL, 0.0, 1.0) * steps;
                float lower = floor(scaled);
                float sm = clamp(fx.toon.y, 0.01, 0.49);
                float edge = smoothstep(0.5 - sm, 0.5 + sm, scaled - lower);
                float banded = (lower + edge) / steps;
                // Specular as a crisp toon highlight, only on lit bands.
                vec3 lit_toon = (kd * diffuse_color / PI) * banded
                    + spec * smoothstep(0.0, 0.02, banded);
                lit = mix(lit, lit_toon, clamp(params0.w, 0.0, 1.0));
            }

            float sf = shadow_factor(light, v_world_pos, N, L);
            if (g_dbg_layer >= 0 && g_dbg_lit < 0.0) {
                g_dbg_lit = sf;
            }
            color += lit * radiance * sf * caustic_gain;
        }
    }

    if (translucent) {
        color += trans_acc * albedo.rgb * fx.trans1.rgb * (fx.trans0.x * trans_map * (1.0 - metallic));
    }

    // Indirect diffuse (as irradiance/PI), in priority order:
    //   1. baked lightmap for static objects (params1.w = array layer, >= 0),
    //   2. baked probe SH where a volume covers this fragment,
    //   3. flat scene ambient.
    vec3 indirect;
    vec3 flux_gi = vec3(0.0); // realtime screen GI part (view mode 6)
    if (has_lightmap) {
        int layer = int(v_lightmap + 0.5);
        indirect = texture(u_lightmap, vec3(v_uv1, float(layer))).rgb / PI;
        // MIXED LIGHTING: a lightmapped (static-baked) surface ALSO receives the
        // realtime FluxVoxel probe GI, added on top of the baked bounce. The bake
        // can't contain dynamic/movable emitters, so without this an emissive orb
        // lights ONLY metals (via reflections) and never the baked floor/couch.
        // Use the SMOOTH (non-occluded) probe sample — the lightmap already carries
        // the static occlusion, and the DDGI Chebyshev's per-probe variation showed
        // as fixed dots on the bright baked surface. Returns 0 with no realtime GI.
        // debug.z < -0.5 = static bake capture: realtime GI (dynamic emitters'
        // bounce) must not bake into the reflection cube.
        if (frame.debug.z > -0.5) {
            indirect += sample_irradiance_smooth(v_world_pos, N);
        }
        // Mixed lighting (debug.z > 1.5): Flux gathers only what the bake can't
        // hold (moving emitters, realtime lights) and it adds on top.
        if (frame.debug.z > 1.5) {
            vec2 suv = gl_FragCoord.xy / vec2(frame.postfx2.w, frame.postfx3.w);
            flux_gi = screen_gi_sample(suv);
            indirect += flux_gi;
        }
    } else if (frame.debug.z > 0.5) {
        // Screen-space GI: depth-aware upsample of the sparse screen probes.
        // Emissive surfaces are in the depth prepass again, so they sample their
        // OWN GI here (their emission still dominates the look).
        vec2 suv = gl_FragCoord.xy / vec2(frame.postfx2.w, frame.postfx3.w);
        flux_gi = screen_gi_sample(suv);
        indirect = flux_gi;
        // Mixed lighting: the gather is dynamic-only, so a non-lightmapped object
        // keeps the baked probes underneath it.
        if (frame.debug.z > 1.5) {
            vec3 occ_irr;
            float cov = sample_probes(v_world_pos, N, occ_irr);
            indirect += mix(frame.ambient.rgb, sample_irradiance_smooth(v_world_pos, N), cov);
        }
    } else {
        // Probe GI where a cascade covers the fragment, fading to flat ambient
        // at the outermost cascade's edge (coverage < 1) and beyond it (0).
        // Use the SAME smooth, full-strength voxel sample as the lightmap path (no
        // per-probe Chebyshev dots/dimming) so a non-baked floor lights up just like
        // a baked one. `sample_probes` is still called for its coverage value (the
        // cascade fade-to-ambient at the volume edge); FluxVoxel's analytic radiance
        // has no occlusion to lose, and the SDF/RT march already bakes occlusion into
        // the probe values, so dropping the Chebyshev here only removes interpolation
        // leak-guarding — a fair trade for the smooth, even result.
        if (frame.debug.z < -0.5) {
            // Static bake capture: no realtime probe GI, flat ambient only.
            indirect = frame.ambient.rgb;
        } else {
            vec3 occ_irr;
            float cov = sample_probes(v_world_pos, N, occ_irr);
            indirect = mix(frame.ambient.rgb, sample_irradiance_smooth(v_world_pos, N), cov);
        }
    }
    // Energy-conserving ambient: split indirect into diffuse (non-reflected) and
    // a cheap specular share so metals/smooth surfaces pick up environment colour
    // instead of reading flat. Uses the probe/lightmap irradiance as a stand-in
    // for environment radiance (no reflection probes yet). Approximate, but it
    // keeps metals lively. Smoother surfaces concentrate the spec response.
    // Light Volumes (baked SH for objects without a lightmap): replace the probe / ambient
    // light by the weight-blended volume light where volumes cover this point.
    if (!has_lightmap && frame.debug.z > -0.5) {
        float lv_cov;
        vec3 lv_e = light_volume_irradiance(v_world_pos, N, lv_cov);
        indirect = mix(indirect, lv_e, lv_cov);
    }
    indirect += voxel_emit;
    // Screen-space AO: indirect light only (direct lights are untouched).
    // Transparent / water surfaces are not in the depth prepass, so the AO
    // there belongs to whatever is behind them: skipped.
    float ssao = 1.0;
#ifndef MULTIVIEW
    if (frame.refl_blend2.w > 0.5 && ALPHA_MODE != 2u && !FEAT_WATER) {
        ssao = texelFetch(u_ssao, ivec2(gl_FragCoord.xy), 0).r;
    }
#endif
    vec3 ssao_diffuse = frame.refl_blend2.w > 1.5 ? ao_multi_bounce(ssao, diffuse_color) : vec3(ssao);
    vec3 F_amb = f_schlick_rough(NdotV, f0, roughness);
    vec3 kd_amb = vec3(1.0) - F_amb;
    color += kd_amb * indirect * diffuse_color * ao * ssao_diffuse;
    // Baked lights' caustics take light away in the dark bands.
    color = max(color, vec3(0.0));
    float spec_amb = mix(1.0, 0.25, roughness); // rough surfaces scatter it away
    // Specular environment radiance, in priority order:
    //   1. SSR hit (on-screen, smooth/metallic), then
    //   2. the prefiltered reflection-probe cube (roughness -> mip), then
    //   3. the indirect-irradiance stand-in (no skybox -> neutral black cube).
    vec3 R = reflect(-V, N);
    // Footprint mip: a smooth curved mirror (a chrome sphere) spans many cube
    // texels per pixel, and sampling mip 0 there aliases into sparkle (fine
    // floor detail, small bright lights). Measured on the raw reflection
    // vector, before box projection, whose derivatives jump at box edges.
    float env_aa_lod = textureQueryLod(u_env, R).x;
    // Box-projected parallax (Unity-style) when a reflection-probe zone covers
    // this fragment: reproject the reflection ray onto the probe's box so flat
    // surfaces line up with the captured environment instead of treating it as
    // infinitely distant. `refl_center.w` = probe intensity (0 = no zone).
    float probe_intensity = frame.refl_center.w;
    vec3 R_raw = R;
    if (probe_intensity > 0.0) {
        R = probe_dir(R, frame.refl_center.xyz, frame.refl_box.xyz, frame.refl_extents.xyz, frame.refl_extents.w > 0.5);
    }
    // No reflection-probe zone → the fallback is the skybox cube, scaled by the
    // skylight/env intensity (frame.ambient.w). So turning ambient/skylight off
    // stops metals mirroring the sky and the scene goes black with no lights;
    // default (intensity 1) is unchanged. A probe zone uses its own intensity.
    float env_scale = probe_intensity > 0.0 ? probe_intensity : frame.ambient.w;
    // The env cube ALWAYS carries the environment (skybox texture/cube, or the
    // procedural sky when none is set), so this is the reflection's base at any
    // angle — never the dark diffuse-GI stand-in. The fullscreen SSR resolve pass
    // (deferred, current-frame) overlays screen-space reflections on top of this.
    vec3 spec_env = textureLod(u_env, R, max(roughness * ENV_MAX_LOD, env_aa_lod)).rgb * env_scale;
    // Multi-probe blend: slot 0 is the fallback, additional slots blend in by
    // their box influence. This supports 3+ overlapping reflection probes.
    float pw0 = probe_influence(v_world_pos, frame.refl_box.xyz, frame.refl_extents.xyz, frame.refl_blend.x);
    float pw1 = probe_influence(v_world_pos, frame.refl2_box.xyz, frame.refl2_extents.xyz, frame.refl_blend.y);
    float pw2 = probe_influence(v_world_pos, frame.refl3_box.xyz, frame.refl3_extents.xyz, frame.refl_blend2.x);
    float pw3 = probe_influence(v_world_pos, frame.refl4_box.xyz, frame.refl4_extents.xyz, frame.refl_blend2.y);
    float wsum = pw0;
    vec3 mix_spec = vec3(0.0);
    if (probe_intensity > 0.0) {
        mix_spec += spec_env * pw0;
    }
    float probe2_intensity = frame.refl2_center.w;
    if (probe2_intensity > 0.0 && pw1 > 0.0) {
        vec3 R2 = probe_dir(R_raw, frame.refl2_center.xyz, frame.refl2_box.xyz, frame.refl2_extents.xyz, frame.refl2_extents.w > 0.5);
        float lod2 = max(roughness * float(textureQueryLevels(u_env_b) - 1), env_aa_lod);
        mix_spec += textureLod(u_env_b, R2, lod2).rgb * probe2_intensity * pw1;
        wsum += pw1;
    }
    float probe3_intensity = frame.refl3_center.w;
    if (probe3_intensity > 0.0 && pw2 > 0.0) {
        vec3 R3 = probe_dir(R_raw, frame.refl3_center.xyz, frame.refl3_box.xyz, frame.refl3_extents.xyz, frame.refl3_extents.w > 0.5);
        float lod3 = max(roughness * float(textureQueryLevels(u_env_c) - 1), env_aa_lod);
        mix_spec += textureLod(u_env_c, R3, lod3).rgb * probe3_intensity * pw2;
        wsum += pw2;
    }
    float probe4_intensity = frame.refl4_center.w;
    if (probe4_intensity > 0.0 && pw3 > 0.0) {
        vec3 R4 = probe_dir(R_raw, frame.refl4_center.xyz, frame.refl4_box.xyz, frame.refl4_extents.xyz, frame.refl4_extents.w > 0.5);
        float lod4 = max(roughness * float(textureQueryLevels(u_env_d) - 1), env_aa_lod);
        mix_spec += textureLod(u_env_d, R4, lod4).rgb * probe4_intensity * pw3;
        wsum += pw3;
    }
    if (wsum > 1e-4) {
        spec_env = mix_spec / wsum;
    }
    if (probe_intensity < 0.0) {
        // Reflection-cube capture: no reflections of reflections (no probe
        // feedback); diffuse GI stands in for the environment.
        spec_env = indirect;
    }
    // Planar reflection: this surface has its own mirrored scene render
    // (pc.emission.w = layer+1 in the integer part, roughness fade cutoff in
    // the fraction). Rendered with the same camera/viewport convention as this
    // pass, so the reflection for THIS pixel sits at the same screen UV — a
    // true mirror with per-pixel parallax, replacing the probe/env on smooth
    // surfaces and fading back to it as roughness rises. frame.refl_box.w = 0
    // in the planar/capture/preview passes themselves (no recursion).
    bool planar_mirror = pc.emission.w >= 1.0 && frame.refl_box.w > 0.5;
    // Share of the specular environment the planar render supplies; the deferred
    // SSR/RT resolve must only replace the remainder (see the G-buffer below).
    float planar_w = 0.0;
    if (planar_mirror) {
        float planar_layer = floor(pc.emission.w) - 1.0;
        float cutoff = max(fract(pc.emission.w), 0.05);
        vec2 suv = gl_FragCoord.xy / vec2(frame.postfx2.w, frame.postfx3.w);
        if (FEAT_WATER) {
            // The water's ripples warp the mirror: the reflected ray tilts
            // with the surface slope, so the sampled image shifts by
            // ~2 * height * slope in screen space. `to_local` gives the
            // rest plane's world height; the vertex displacement (waves +
            // simulated ripples) is in `v_world_pos.y`, and the two
            // normal-only drifting layers add their analytic height.
            int ws = clamp(int(fx.water.x + 0.5), 0, MAX_WATER_BODIES - 1);
            mat4 wl = frame.water_bodies[ws].to_local;
            float plane_y = -dot(vec3(wl[0].y, wl[1].y, wl[2].y), wl[3].xyz);
            float h = v_world_pos.y - plane_y;
            vec4 wlook = frame.water_bodies[ws].look0;
            h += water_ripple_height(v_world_pos.xz, frame.water.x, max(wlook.z, 0.01)) * (wlook.y * 0.25 / max(wlook.z, 0.01));
            // The shift is 2 h slope in WORLD units; convert to UV: divide
            // by the world-per-pixel scale (dFdx of world X/Z) and the
            // viewport size. (The previous form skipped that conversion, so
            // the warp was orders of magnitude too small to see.)
            float vw = max(frame.postfx2.w, 1.0);
            float vh = max(frame.postfx3.w, 1.0);
            float dxw = max(abs(dFdx(v_world_pos.x)), 1e-5);
            float dzw = max(abs(dFdy(v_world_pos.z)), 1e-5);
            suv -= vec2(2.0 * h * dFdx(v_world_pos.y) / dxw / vw, 2.0 * h * dFdy(v_world_pos.y) / dzw / vh);
            suv = clamp(suv, 0.001, 0.999);
        }
        vec3 planar_col = texture(u_planar, vec3(suv, planar_layer)).rgb;
        float w = 1.0 - smoothstep(cutoff * 0.5, cutoff, roughness);
        planar_w = w;
        spec_env = mix(spec_env, planar_col, w);
    }
    // Per-surface reflection mode: the water body's own (GpuWater.reflect.x,
    // 0 env / 1 planar / 2 probes / 3 off) and the material's (fx.parallax.w,
    // same numbers). Off drops the environment term entirely; probes-only
    // keeps the probe reflection but skips the deferred resolve below.
    if (FEAT_WATER && frame.water_bodies[clamp(int(fx.water.x + 0.5), 0, MAX_WATER_BODIES - 1)].reflect.x >= 3.0) {
        spec_env = vec3(0.0);
    }
    if (fx.parallax.w >= 3.0) {
        spec_env = vec3(0.0);
    }
    // Reflectance weight that multiplies environment radiance. The resolve pass
    // re-derives env radiance from this same cube and swaps in the SSR hit:
    //   final += reflectance * conf * (ssr_radiance - env_radiance)
    // so SSR exactly replaces the env reflection where it has a confident hit and
    // leaves the env reflection untouched (this colour, fogged below) elsewhere.
    // Split-sum environment BRDF instead of a raw Fresnel*spec term: bounded,
    // roughness-correct specular that doesn't blow out to a harsh mirror band at
    // grazing angles on rough/dielectric surfaces.
    vec2 env_ab = env_brdf_approx(roughness, NdotV);
    // Reflection strength (fx.parallax.y, default 1) scales the env reflection
    // here AND the gbuffer reflectivity below so the deferred SSR/RT resolve
    // tracks it. Reflection *technique* (Probes / SSR / Ray-traced) is a GLOBAL
    // scene setting (Environment → Reflections), NOT a per-material toggle — UE
    // does the same (reflection method is a post-process/project setting, not a
    // material checkbox), so the old per-material SSR/RT gate was removed.
    float refl_strength = fx.parallax.y;
    // Specular occlusion from the screen-space AO (Lagarde & de Rousiers 2014).
    float spec_occ = ssao < 1.0 ? clamp(pow(NdotV + ssao, exp2(-16.0 * roughness - 1.0)) - 1.0 + ssao, 0.0, 1.0) : 1.0;
    vec3 reflectance = (f0 * env_ab.x + vec3(env_ab.y)) * ao * spec_occ * refl_strength;
    if (FEAT_WATER) {
        reflectance = f_schlick(NdotV, vec3(water_f0)) * refl_strength * (1.0 - 0.8 * water_foam);
    }
    color += reflectance * spec_env;
    // FluxVoxel specular-from-volume (frame.fog_params.w = voxel_specular flag):
    // sample the volume in the reflection direction so SMOOTH METALS pick up the
    // local emissive/voxel bounce the (coarse) reflection cube misses. This term is
    // VIEW-DEPENDENT (R = reflect(-V, N)), so it must NOT drive a rough/dielectric
    // floor — there the emissive arrives as VIEW-INDEPENDENT diffuse GI (above), so
    // the floor glow looks the same from every angle instead of riding the camera.
    // Gate by metallic·(1-roughness): ~0 for dielectric OR rough, only smooth metals
    // get it. (Earlier `reflectance·vrad·roughness` lit rough floors at grazing
    // Fresnel → the "GI only on the camera side" artifact.)
    // It is a FALLBACK reflection source, not an extra layer: where a probe
    // zone or a planar mirror already reflects the scene, adding it showed
    // every emissive twice (sharp from the probe, a smeared voxel streak).
    if (frame.fog_params.w > 0.5 && probe_intensity == 0.0 && !planar_mirror) {
        float spec_w = metallic * (1.0 - roughness);
        if (spec_w > 0.001) {
            vec3 vrad;
            if (volume_radiance(v_world_pos, R, vrad)) {
                color += f0 * vrad * spec_w;
            }
        }
    }
    // SSR G-buffer: octahedral view-space normal (rg) + roughness (b) + a scalar
    // reflectivity (a, the env-reflection weight's luminance). The resolve pass
    // uses the stored normal directly for an exact reflection ray. The global
    // reflection mode decides whether the resolve runs at all.
    vec3 Nv = normalize(mat3(VIEW_MATRIX) * N);
    // The resolve adds reflectivity * conf * (ssr - probe_env): correct only for
    // the probe part of spec_env. Where the planar render replaced it, that
    // delta swapped a real mirror image for a screen-space hit minus the WRONG
    // environment (blown complementary colours, noise at oblique angles), so the
    // planar share is withheld from the deferred pass.
    float reflectivity = dot(reflectance, vec3(0.2126, 0.7152, 0.0722)) * (1.0 - planar_w);
    if (FEAT_WATER) {
        // The water writes its own depth (pipeline.rs), so the resolve
        // starts its ray at the surface with this rippled normal. It is
        // blended over the scene by `water_alpha`: its reflection shares the
        // pixel by that much (thin water at a contact edge barely reflects).
        reflectivity *= water_alpha;
        // Per-water reflection mode: probes-only (2) and off (3) skip the
        // deferred resolve.
        if (frame.water_bodies[clamp(int(fx.water.x + 0.5), 0, MAX_WATER_BODIES - 1)].reflect.x >= 2.0) {
            reflectivity = 0.0;
        }
    }
    if (fx.parallax.w >= 2.0) {
        reflectivity = 0.0;
    }
    // Packed for the 10:10:10:2 target: roughness (6 bits) in b's high bits,
    // reflectivity (6 bits) split over b's low 4 and a's 2. Unpacked with an
    // exact texel fetch in the resolves (never filtered).
    uint r6 = uint(round(clamp(roughness, 0.0, 1.0) * 63.0));
    uint f6 = uint(round(clamp(reflectivity, 0.0, 1.0) * 63.0));
    o_gbuf = vec4(oct_encode(Nv), float((r6 << 4) | (f6 & 15u)) / 1023.0, float(f6 >> 4) / 3.0);

    if (FEAT_EMISSION) {
        // Emission uses its own tiling/offset, can scroll, pulse over time, and
        // be masked (the mask shares the emission UV transform).
        vec2 euv = uv_emission + fx.scroll.zw * frame.misc.x;
        float pulse = material_emission_pulse(frame.misc.x,fx.toon.z);
        float emask = texture(t_emission_mask, uv_emission).r;
        color += texture(t_emission, euv).rgb * pc.emission.rgb * pulse * emask;
    }

    // Matcaps (Poiyomi-style): sample 3 view-space sphere-mapped layers, each
    // scaled by its mask + strength and combined by its per-layer blend mode.
    // Default matcaps are black with strength 0 so unused layers cost nothing.
    if (fx.matcap.x > 0.0 || fx.matcap.y > 0.0 || fx.matcap.z > 0.0) {
        vec3 vn = normalize(mat3(VIEW_MATRIX) * N);
        vec2 muv = vn.xy * 0.5 + 0.5;
        color = blend_matcap(color, texture(t_matcap0, muv).rgb,
                             texture(t_matcap0_mask, uv).r * fx.matcap.x, fx.matcap_blend.x);
        color = blend_matcap(color, texture(t_matcap1, muv).rgb,
                             texture(t_matcap1_mask, uv).r * fx.matcap.y, fx.matcap_blend.y);
        color = blend_matcap(color, texture(t_matcap2, muv).rgb,
                             texture(t_matcap2_mask, uv).r * fx.matcap.z, fx.matcap_blend.z);
    }

    // Toon rim light (Poiyomi-style): a Fresnel edge glow in the rim colour,
    // with tunable power + strength from the FX block.
    if (FEAT_TOON && rim_strength > 0.0) {
        float rim = pow(1.0 - NdotV, max(fx.rim.w, 0.1));
        color += rim * rim_strength * fx.rim.rgb;
    }

    // Viewport render modes (frame.debug.y): 1 = world normals, 2 = indirect/GI
    // term only, 3 = unlit (raw albedo, no lighting/post), 4 = reflection cube
    // (every surface a perfect mirror of u_env at R — reveals the captured cube's
    // orientation directly, no BRDF/roughness), 5 = reflection direction R as RGB,
    // 6 = the realtime Flux screen GI alone (no bake underneath).
    int gi_dbg = int(frame.debug.y + 0.5);
    if (gi_dbg == 1) {
        o_color = vec4(N * 0.5 + 0.5, 1.0);
        return;
    } else if (gi_dbg == 2) {
        o_color = vec4(indirect, 1.0);
        return;
    } else if (gi_dbg == 3) {
        o_color = vec4(albedo.rgb, 1.0);
        return;
    } else if (gi_dbg == 4) {
        // Perfect-mirror sample of the reflection cube (sharp mip 0), no shading.
        o_color = vec4(textureLod(u_env, R, 0.0).rgb * env_scale, 1.0);
        return;
    } else if (gi_dbg == 5) {
        // Reflection direction visualized: +X red, +Y green, +Z blue.
        o_color = vec4(normalize(R) * 0.5 + 0.5, 1.0);
        return;
    } else if (gi_dbg == 6) {
        o_color = vec4(flux_gi, 1.0);
        return;
    } else if (gi_dbg == 12) {
        // Screen-space AO alone (ao.rs VIEW_AO).
        o_color = vec4(vec3(ssao), 1.0);
        return;
    } else if (gi_dbg >= 8) {
        // Zestlet modes: 8 LOD level, 9 every zestlet its own colour, 10 coverage (green = zestlet, the
        // ordinary path in its refusal colour, passed in the instance colour), 11 colour per triangle.
        // The clustered vertex shader tints v_color and marks v_uv1.x = 1e6; the rest stays grey.
        bool zestlet = v_uv1.x > 1.0e5;
        vec3 zc = vec3(0.35);
        if (gi_dbg == 11) {
            // gl_PrimitiveID restarts for every drawn cluster, so a zestlet also keys by its own colour.
            uvec3 key = zestlet ? uvec3(v_color.rgb * 255.0 + 0.5) : uvec3(0u);
            zc = zestlet_hash_colour(uint(gl_PrimitiveID) + ((key.x << 16) | (key.y << 8) | key.z) * 40503u);
        } else if (zestlet) {
            zc = v_color.rgb;
        } else if (gi_dbg == 10) {
            zc = v_instance_color.rgb;
        }
        o_color = vec4(zc, 1.0);
        return;
    } else if (gi_dbg == 7) {
        // Shadow layer id as a colour (0 red, 1 green, 2 blue, ...), darkened
        // where that layer's shadow factor says occluded; grey = unshadowed.
        if (g_dbg_layer < 0) {
            o_color = vec4(vec3(0.15), 1.0);
        } else {
            float h = float(g_dbg_layer);
            vec3 c = 0.5 + 0.5 * cos(6.2831853 * (h * 0.173 + vec3(0.0, 0.33, 0.67)));
            o_color = vec4(c * (0.25 + 0.75 * g_dbg_lit), 1.0);
        }
        return;
    }

    // Fog is now a raymarched volumetric medium in the deferred resolve pass
    // (ssr_resolve.frag) — a real participating medium that hangs + drifts in the
    // air, not a per-surface colour tint — so it's no longer applied here.

    if (FEAT_WATER) {
        int water_slot = clamp(int(fx.water.x + 0.5), 0, MAX_WATER_BODIES - 1);
        vec3 volume = water_volume_lighting(water_slot, water_scattering, water_path, water_up, V);
        color += (water_behind + volume) * (1.0 - f_schlick(NdotV, vec3(water_f0)).x);
        // Fog: the deferred resolve fogs the opaque scene, but the water is
        // drawn transparent afterwards and would keep full colour right up to
        // the horizon's hard mesh edge. Apply the same medium parameters
        // (density, height falloff, start distance) so far water blends into
        // the sky like the rest of the scene.
        if (frame.fog_color.w > 0.0) {
            float fog_d = max(length(CAMERA_POS.xyz - v_world_pos) - frame.fog_params.z, 0.0);
            float fog_h = exp(-frame.fog_params.x * max(0.0, v_world_pos.y - frame.fog_params.y));
            color = mix(color, frame.fog_color.rgb, 1.0 - exp(-frame.fog_color.w * fog_h * fog_d));
        }
    }
    if (transmit_amt > 0.0) {
        // The opaque scene behind, shifted by the shading normal (zero offset for
        // a flat pane), tinted by the volume attenuation, minus what reflects.
        vec2 sz = vec2(textureSize(u_opaque_color, 0));
        vec3 Nv = mat3(VIEW_MATRIX) * N;
        vec2 suv = clamp(gl_FragCoord.xy / sz + vec2(Nv.x, -Nv.y) * fx.trans2.w, vec2(0.002), vec2(0.998));
        // Rough transmission: roughness (stretched by the volume's thickness) picks
        // a level of the opaque copy's mip chain; a 5-tap tent over that level
        // hides the box filter's blockiness. Level 0 (smooth, or blur off) is the
        // single sharp tap as before.
        float blur = clamp(fx.trans3.x, 0.0, 1.0) * roughness * (0.35 + 0.65 * clamp(fx.trans3.y * transmit_thick, 0.0, 1.0));
        float max_lod = float(textureQueryLevels(u_opaque_color) - 1);
        float lod = clamp(blur * 6.0, 0.0, max_lod);
        vec3 behind;
        if (lod < 0.05) {
            behind = textureLod(u_opaque_color, suv, 0.0).rgb;
        } else {
            vec2 o = vec2(0.75 * exp2(lod)) / sz;
            behind = textureLod(u_opaque_color, suv, lod).rgb * 0.4
                + (textureLod(u_opaque_color, clamp(suv + vec2(o.x, o.y), vec2(0.002), vec2(0.998)), lod).rgb
                 + textureLod(u_opaque_color, clamp(suv + vec2(-o.x, o.y), vec2(0.002), vec2(0.998)), lod).rgb
                 + textureLod(u_opaque_color, clamp(suv + vec2(o.x, -o.y), vec2(0.002), vec2(0.998)), lod).rgb
                 + textureLod(u_opaque_color, clamp(suv + vec2(-o.x, -o.y), vec2(0.002), vec2(0.998)), lod).rgb) * 0.15;
        }
        color += behind * fx.trans2.rgb * (transmit_amt * (1.0 - f_schlick(NdotV, vec3(0.04)).x));
    }

    color = apply_postfx(color, gl_FragCoord.xy);
    float alpha = (ALPHA_MODE == 2u || ALPHA_MODE == 3u) ? albedo.a : 1.0;
    if (FEAT_WATER) {
        alpha = water_alpha;
    }
    // Never hand NaN / Inf to the HDR target: bloom and TAA would spread it.
    if (any(isnan(color)) || any(isinf(color))) {
        color = vec3(0.0);
    }
    if (WRITE_VELOCITY) {
        vec2 cur = v_cur_clip.xy / max(abs(v_cur_clip.w), 1e-6);
        vec2 prev = v_prev_clip.xy / max(abs(v_prev_clip.w), 1e-6);
        o_velocity = cur - prev;
        o_reactive = 1.0 - alpha;
    }
    o_color = vec4(min(color, vec3(64000.0)), alpha);
}
