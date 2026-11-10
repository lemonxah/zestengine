// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// Deferred screen-space reflections, resolve pass.
//
// Runs fullscreen AFTER the forward pass, so it marches against the CURRENT
// frame's lit colour (no 1-frame lag) using the depth prepass + the forward
// pass's reflectance/roughness G-buffer. The forward pass already added the
// environment-cube reflection (reflectance * env), correctly fogged; this pass
// re-derives that same env radiance and swaps in the screen-space hit:
//   out = scene + reflectance * conf * (ssr_radiance - env_radiance)
// so a confident SSR hit exactly replaces the env reflection and a miss leaves
// the forward result untouched. Normals come from the forward G-buffer. The hierarchical tracer
// reconstructs local depth planes and rejects foreground/background bridges.

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

layout(set = 0, binding = 0) uniform sampler2D u_color; // current lit HDR colour
layout(set = 0, binding = 1) uniform sampler2D u_gbuf;  // reflectance.rgb, roughness.a
layout(set = 0, binding = 2) uniform sampler2D u_depth; // scene depth prepass
layout(set = 0, binding = 3) uniform samplerCube u_env; // prefiltered environment
layout(set = 0, binding = 5) uniform samplerCube u_env_b; // second reflection probe
layout(set = 0, binding = 11) uniform samplerCube u_env_c; // third reflection probe
layout(set = 0, binding = 12) uniform samplerCube u_env_d; // fourth reflection probe

#define REFLECTION_UNIFORM_BINDING 4
#include "reflection_uniforms.glsl"

// Max roughness mip of the CURRENTLY bound cube, queried at runtime (correct for
// the 64px skybox cube or a higher-res captured probe cube). Matches standard.frag.
// Declared before probe_blend.glsl so its `probe_env` can use it.
#define ENV_MAX_LOD (float(textureQueryLevels(u_env) - 1))

#include "probe_blend.glsl"

// The scene renders with a Y-flipped (negative-height) viewport, so a sampling
// UV maps to NDC with a flipped Y. This constant matches the forward pass.
const bool FLIP_Y = true;

// Full view-space position of the scene surface at a sampling UV + sampled depth.
vec3 view_pos(vec2 uv, float d) {
    vec2 ndc = uv * 2.0 - 1.0;
    if (FLIP_Y) ndc.y = -ndc.y;
    vec4 vp = u.inv_proj * vec4(ndc, d, 1.0);
    return vp.xyz / vp.w;
}

// Octahedral decode of the view-space normal the forward pass stored (rg).
vec3 oct_decode(vec2 e) {
    e = e * 2.0 - 1.0;
    vec3 n = vec3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    float t = max(-n.z, 0.0);
    n.x += n.x >= 0.0 ? -t : t;
    n.y += n.y >= 0.0 ? -t : t;
    return normalize(n);
}

#ifndef REFLECTION_RAW
#include "reflection_fog.glsl"
#endif

#define FLUX_DEPTH_BINDING 13
#include "flux_reflection_screen.glsl"

vec4 trace(vec3 P,vec3 N,out float distance) {
    distance=0.0;
    vec3 dir=reflect(normalize(P),N);
    float toward_camera=1.0-smoothstep(0.1,0.6,dir.z);
    if(toward_camera<=0.0) return vec4(0);
    vec4 hit=flux_reflection_screen(u_depth,u_gbuf,u_color,P+N*0.01,dir,
        max(u.ssr.z,1.0),u.proj,u.inv_proj,FLIP_Y,toward_camera,-1.0,distance);
    return hit;
}

// Roughness + reflectivity from the packed G-buffer texel (see standard.frag).
vec2 gbuf_rough_refl(vec4 g) {
    uint b = uint(round(g.b * 1023.0));
    uint a = uint(round(g.a * 3.0));
    return vec2(float(b >> 4) / 63.0, float((b & 15u) | (a << 4)) / 63.0);
}

void main() {
    vec3 base = texture(u_color, v_uv).rgb;
    float depth = texture(u_depth, v_uv).r;
    vec4 g = texelFetch(u_gbuf, ivec2(gl_FragCoord.xy), 0);
    vec2 rr = gbuf_rough_refl(g);
    float roughness = rr.x;
    float reflectivity = rr.y; // env-reflection weight (0 = not a reflector)
    bool is_sky = depth >= 0.9999;

    vec3 outc = base;
    vec3 replacement = vec3(0);
    float hit_distance = -1.0;
    vec3 worldPos;

    if (is_sky) {
        // Reconstruct the world-space view ray to a far point so fog still
        // accumulates through empty air over the sky.
        vec2 ndc = v_uv * 2.0 - 1.0;
        if (FLIP_Y) ndc.y = -ndc.y;
        vec4 vp = u.inv_proj * vec4(ndc, 1.0, 1.0);
        vec3 vdir = normalize(vp.xyz / vp.w);
        vec3 wdir = normalize(mat3(u.inv_view) * vdir);
        worldPos = u.camera_pos.xyz + wdir * 200.0;
    } else {
        vec3 P = view_pos(v_uv, depth);
        worldPos = (u.inv_view * vec4(P, 1.0)).xyz;
        // SSR for reflective, smooth-enough surfaces (gated hard; rough/matte
        // keeps the clean env-cube reflection only).
        if (u.ssr.x > 0.5 && u.ssr.y > 0.0 && roughness <= u.ssr.w && reflectivity >= 1e-4) {
            vec3 N = oct_decode(g.rg); // exact view-space shading normal
            vec3 Vv = normalize(-P);
            float NdotV = max(dot(N, Vv), 1e-3);
            vec3 Nw = normalize(mat3(u.inv_view) * N);
            vec3 Vw = normalize(u.camera_pos.xyz - worldPos);
            vec3 Rw = reflect(-Vw, Nw);
            vec3 Rw_raw = Rw;
            float probe_intensity = u.refl_center.w;
            if (probe_intensity > 0.0 && u.refl_extents.w > 0.5) {
                vec3 bmin = u.refl_box.xyz - u.refl_extents.xyz;
                vec3 bmax = u.refl_box.xyz + u.refl_extents.xyz;
                // Only box-project surfaces inside the probe box (matches
                // standard.frag; outside, the box proxy mis-places the reflection).
                // Flush-surface tolerance + backward-hit clamp match standard.frag.
                vec3 eps = u.refl_extents.xyz * 0.02 + 0.02;
                bool inside = all(greaterThanEqual(worldPos, bmin - eps))
                           && all(lessThanEqual(worldPos, bmax + eps));
                if (inside) {
                    vec3 invR = 1.0 / Rw;
                    vec3 t1 = (bmax - worldPos) * invR;
                    vec3 t2 = (bmin - worldPos) * invR;
                    vec3 tmax = max(t1, t2);
                    float dist = max(min(min(tmax.x, tmax.y), tmax.z), 0.0);
                    Rw = (worldPos + Rw * dist) - u.refl_center.xyz;
                }
            }
            float env_scale = probe_intensity > 0.0 ? probe_intensity : u.screen.z;
            vec3 env = textureLod(u_env, Rw, roughness * ENV_MAX_LOD).rgb * env_scale;
            // Same two-probe blend the forward pass applied, so the SSR
            // replacement subtracts exactly the environment it added.
            if (u.refl2_center.w > 0.0 && probe_intensity >= 0.0) {
                vec3 R2 = probe_box_dir(worldPos, Rw_raw, u.refl2_center.xyz, u.refl2_box.xyz, u.refl2_extents.xyz, u.refl2_extents.w > 0.5);
                vec3 env_b = textureLod(u_env_b, R2, roughness * float(textureQueryLevels(u_env_b) - 1)).rgb * u.refl2_center.w;
                float wb = probe_b_weight(worldPos, u.refl_box.xyz, u.refl_extents.xyz, u.refl_blend.x,
                                          u.refl2_box.xyz, u.refl2_extents.xyz, u.refl_blend.y, u.refl_blend.z > 0.5);
                env = mix(env, env_b, wb);
            }
            // 3+ probes: the forward pass normalises all active slots, so do the
            // same here (the two-probe path above stays byte-for-byte unchanged).
            if (u.refl3_center.w > 0.0) {
                env = probe_env(worldPos, Rw_raw, roughness, true);
            }
            hit_distance = 0.0;
            float distance;
            vec4 ssr = trace(P, N, distance);
            if (ssr.a > 0.0) {
                float rough_fade = 1.0 - smoothstep(u.ssr.w * 0.5, u.ssr.w, roughness);
                // Mirror-smooth surfaces (calm water, polished floors) keep
                // their reflection much nearer grazing, where it shows most:
                // a flat mirror's ray there runs away from the camera across
                // the screen, which the march follows well.
                float mirror = 1.0 - smoothstep(0.05, 0.25, roughness);
                float graze_fade = smoothstep(mix(0.15, 0.04, mirror), mix(0.45, 0.14, mirror), NdotV);
                float w = clamp(ssr.a * rough_fade * graze_fade * u.ssr.y, 0.0, 1.0);
                vec3 refl = base + reflectivity * w * (ssr.rgb - env);
                // A non-finite sample (e.g. from a freshly recreated target)
                // must not reach the frame.
                if (!any(isnan(refl)) && !any(isinf(refl))) {
                    outc = refl;
                    replacement = reflectivity * w * (ssr.rgb - env);
                    hit_distance = max(distance, 0.0);
                }
            }
        }
    }

#ifdef REFLECTION_RAW
    // Signed radiance replacement + ray distance, exactly the RT reconstruction contract.
    o_color = vec4(replacement, hit_distance);
#else
    // Volumetric fog applied to everything (surfaces + sky) so it hangs + drifts.
    if (u.fog_color.w > 0.0) {
        outc = reflection_fog_apply_fog(outc, worldPos, u.camera_pos.xyz, u.fog_color, u.fog_params, u.fog_light, u.fog_sun);
    }
    o_color = vec4(outc, 1.0);
#endif
}
