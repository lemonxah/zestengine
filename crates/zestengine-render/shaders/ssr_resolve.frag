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

layout(set = 0, binding = 4) uniform SsrData {
    mat4 proj;
    mat4 inv_proj;
    mat4 view;
    mat4 inv_view;
    vec4 camera_pos;   // xyz world camera
    vec4 ssr;          // x enabled, y intensity, z max view dist, w roughness cutoff
    vec4 refl_center;  // xyz probe CAPTURE position (re-vector anchor), w probe intensity (0 = none)
    vec4 refl_extents; // xyz half-extents, w box-projection enabled
    vec4 fog_color;    // xyz colour, w density (0 = no fog)
    vec4 fog_params;   // x height falloff, y height ref, z start distance, w _
    vec4 fog_light;    // xyz sun colour*intensity, w anisotropy g
    vec4 fog_sun;      // xyz dir to sun, w time
    vec4 screen;       // x width, y height, z _, w _
    vec4 refl_box;     // xyz probe box CENTER (capture + box offset), w _
    vec4 refl2_center; // second probe (binding 5), same layout; w = intensity (0 = none)
    vec4 refl2_extents;
    vec4 refl2_box;
    vec4 refl_blend;   // x slot 0 blend, y slot 1 blend, z slot 1 on top
    vec4 refl3_center; // third probe (binding 11)
    vec4 refl3_extents;
    vec4 refl3_box;
    vec4 refl4_center; // fourth probe (binding 12)
    vec4 refl4_extents;
    vec4 refl4_box;
    vec4 refl_blend2;  // x slot 2 blend, y slot 3 blend
} u;

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

// --- Volumetric fog: a real raymarched participating medium (animated, patchy,
// with sun in-scattering), so fog hangs + drifts in the air instead of just
// tinting surface colour. ---
const float PI = 3.14159265359;

float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

float vnoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash13(i + vec3(0, 0, 0)), hash13(i + vec3(1, 0, 0)), f.x),
            mix(hash13(i + vec3(0, 1, 0)), hash13(i + vec3(1, 1, 0)), f.x), f.y),
        mix(mix(hash13(i + vec3(0, 0, 1)), hash13(i + vec3(1, 0, 1)), f.x),
            mix(hash13(i + vec3(0, 1, 1)), hash13(i + vec3(1, 1, 1)), f.x), f.y),
        f.z);
}

float fbm(vec3 p) {
    float a = 0.5, s = 0.0;
    for (int i = 0; i < 3; ++i) {
        s += a * vnoise(p);
        p *= 2.02;
        a *= 0.5;
    }
    return s;
}

// Henyey-Greenstein phase (forward-scatter sun glow when looking toward the sun).
float hg_phase(float cosT, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * cosT, 1e-4), 1.5));
}

// March from the camera to `endWorld`, accumulating animated height-fog density +
// sun in-scattering. Returns the fogged colour.
vec3 apply_fog(vec3 color, vec3 endWorld) {
    vec3 ro = u.camera_pos.xyz;
    vec3 seg = endWorld - ro;
    float dist = length(seg);
    if (dist < 1e-3) return color;
    vec3 rd = seg / dist;
    const int STEPS = 48;
    float stepLen = dist / float(STEPS);
    vec3 wind = vec3(0.06, 0.0, 0.04) * u.fog_sun.w; // drift over time
    float phase = hg_phase(dot(rd, normalize(u.fog_sun.xyz)), u.fog_light.w);
    float start = u.fog_params.z;
    float transmittance = 1.0;
    vec3 inscatter = vec3(0.0);
    for (int i = 0; i < STEPS; ++i) {
        float d = stepLen * (float(i) + 0.5);
        if (d < start) continue;
        vec3 p = ro + rd * d;
        float height = exp(-u.fog_params.x * max(0.0, p.y - u.fog_params.y));
        float n = fbm(p * 0.08 + wind); // patchy, drifting density
        float dens = u.fog_color.w * height * clamp(0.25 + 1.5 * n, 0.0, 2.0) * stepLen;
        if (dens <= 0.0) continue;
        vec3 lit = u.fog_color.rgb + u.fog_light.rgb * phase;
        float a = 1.0 - exp(-dens);
        inscatter += transmittance * a * lit;
        transmittance *= exp(-dens);
        if (transmittance < 0.01) break;
    }
    return color * transmittance + inscatter;
}

#define FLUX_DEPTH_BINDING 13
#include "flux_screen_trace.glsl"

// Geometry-only hierarchical trace, then validate the visible hit's normal
// before sampling current physical HDR. A miss retains the environment probe.
vec4 trace(vec3 P,vec3 N) {
    vec3 dir=reflect(normalize(P),N);
    float toward_camera=1.0-smoothstep(0.1,0.6,dir.z);
    if(toward_camera<=0.0) return vec4(0);
    FluxScreenHit hit=flux_trace_screen(u_depth,P+N*0.01,dir,max(u.ssr.z,1.0),u.proj,u.inv_proj,FLIP_Y,96u);
    if(!hit.valid) return vec4(0);
    ivec2 texel=clamp(ivec2(hit.uv*u.screen.xy),ivec2(0),textureSize(u_gbuf,0)-1);
    if(dot(oct_decode(texelFetch(u_gbuf,texel,0).rg),dir)>=0.05) return vec4(0);
    vec2 edge=smoothstep(vec2(0),vec2(.06),hit.uv)*(1.0-smoothstep(vec2(.94),vec2(1),hit.uv));
    float confidence=smoothstep(.45,.8,edge.x*edge.y*toward_camera);
    return vec4(texture(u_color,hit.uv).rgb,confidence);
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
        if (u.ssr.x > 0.5 && roughness <= u.ssr.w && reflectivity >= 1e-4) {
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
            float env_scale = probe_intensity > 0.0 ? probe_intensity : 1.0;
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
            vec4 ssr = trace(P, N);
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
                if (!any(isnan(refl)) && !any(isinf(refl))) outc = refl;
            }
        }
    }

    // Volumetric fog applied to everything (surfaces + sky) so it hangs + drifts.
    if (u.fog_color.w > 0.0) {
        outc = apply_fog(outc, worldPos);
    }
    o_color = vec4(outc, 1.0);
}
