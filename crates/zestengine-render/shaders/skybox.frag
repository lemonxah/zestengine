// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// Sky pass, fragment stage. Every output is a pure function of this pixel's
// view ray and the frame's camera / sun: no time, frame index or temporal noise
// (only the ZESTENGINE_SKY_DEBUG parity tint changes between frames).
//
// Source (push constants):
//   params0.z > 0.5 -> cube: the shared environment cube (binding 7), mip 0
//   params0.x > 0.5 -> equirect texture in set 1 slot 0 (mipmapped, repeat U / clamp V)
//   else            -> procedural single-scattering sky lit by the frame's
//                      directional light (light_dir / light_color), with a sun disc
// params0.w > 0.5 -> HDR output (the post pass tonemaps); else tonemap / grade
//                    inline and add a fixed per-pixel dither against banding.
// emission.w > 0.5 -> debug tint (emission.rgb).

layout(set = 0, binding = 0) uniform FrameData {
    mat4 view;
    mat4 proj;
    mat4 view_proj;
    vec4 camera_pos;
    vec4 light_dir;   // xyz: direction the main directional light travels
    vec4 light_color; // rgb: colour * intensity
    vec4 ambient;
    vec4 misc;
    vec4 postfx0; // x tonemap mode, y exposure EV, z grade exposure, w contrast
    vec4 postfx1; // x saturation, y temperature, z tint, w grading enabled
    vec4 postfx2; // x vignette enabled, y intensity, z smoothness, w screen width
    vec4 postfx3; // xyz vignette color, w screen height
} frame;

layout(set = 1, binding = 0) uniform sampler2D t_sky;
layout(set = 0, binding = 7) uniform samplerCube u_env;

layout(push_constant) uniform Push {
    mat4 model;
    vec4 base_color;
    vec4 emission;
    vec4 params0;
    vec4 params1;
} pc;

layout(location = 0) noperspective in vec2 v_ndc;
layout(location = 1) flat in vec3 v_k0;
layout(location = 2) flat in vec3 v_k1;
layout(location = 3) flat in vec3 v_k2;

layout(location = 0) out vec4 o_color;
// Deferred-SSR G-buffer (gbuf pipeline variants only): the sky reflects nothing.
layout(location = 1) out vec4 o_gbuf;

const float PI = 3.14159265359;
const float TAU = 6.28318530718;

// ---------------------------------------------------------------------------
// Procedural sky: analytic single scattering in an exponential Rayleigh + Mie
// atmosphere over a spherical Earth. Optical depths use the closed-form air
// mass of a ray through a thin exponential shell (no ray marching), so the cost
// is a handful of exp / sqrt per pixel. Keep in sync with sky.rs (CPU copy for
// the default reflection environment).
// ---------------------------------------------------------------------------
const vec3 BETA_R = vec3(5.802e-3, 13.558e-3, 33.1e-3); // Rayleigh scattering, 1/km
const float BETA_M = 3.996e-3;                          // Mie scattering, 1/km
const float BETA_M_EXT = 4.44e-3;                       // Mie extinction, 1/km
const float H_R = 8.0;                                  // scale heights, km
const float H_M = 1.2;
const float R_EARTH = 6360.0;
const float MIE_G = 0.76;
// Sun irradiance scale: a clear noon zenith of about 0.3 (the old gradient's level).
const float SUN_E = 20.0;
const float SUN_RADIUS = 0.0093; // ~0.53 deg radius disc (twice the real sun, readable)
const float SUN_DISC = 3.0;      // disc radiance / SUN_E
const vec3 NIGHT = vec3(0.0025, 0.0035, 0.007);
const vec3 GROUND_ALBEDO = vec3(0.09, 0.085, 0.08);

// Path length (in scale heights) along elevation cosine `mu` through a shell of scale height h.
float air_mass(float mu, float h) {
    float r = R_EARTH / h;
    return sqrt(r * r * mu * mu + 2.0 * r + 1.0) - r * mu;
}

float phase_rayleigh(float nu) {
    return 3.0 / (16.0 * PI) * (1.0 + nu * nu);
}

float phase_mie(float nu) {
    // Cornette-Shanks.
    float g2 = MIE_G * MIE_G;
    return 3.0 / (8.0 * PI) * ((1.0 - g2) * (1.0 + nu * nu)) / ((2.0 + g2) * pow(max(1.0 + g2 - 2.0 * MIE_G * nu, 1e-4), 1.5));
}

vec3 transmittance(float mu) {
    return exp(-(BETA_R * H_R * air_mass(mu, H_R) + BETA_M_EXT * H_M * air_mass(mu, H_M)));
}

// Radiance of the sky above the horizon (view elevation cosine >= 0), without the sun disc.
vec3 sky_scatter(vec3 v, vec3 s, vec3 tint) {
    float mu = max(v.y, 0.0);
    float nu = dot(v, s);
    float mr = air_mass(mu, H_R);
    float mm = air_mass(mu, H_M);
    vec3 tau_r = BETA_R * H_R * mr;
    float tau_m = BETA_M * H_M * mm;
    vec3 ext = tau_r + BETA_M_EXT * H_M * mm;
    // Homogeneous-path in-scatter: sum(scatter) * (1 - e^-ext) / ext.
    vec3 frac = (1.0 - exp(-ext)) / max(ext, vec3(1e-6));
    vec3 sun_t = transmittance(s.y);
    vec3 single = (tau_r * phase_rayleigh(nu) + tau_m * phase_mie(nu)) * frac * sun_t;
    // Cheap multiple-scattering term: an isotropic share of the sunlight the
    // column scatters, so twilight fades to blue instead of cutting to black.
    float day = smoothstep(-0.18, 0.08, s.y);
    vec3 multi = (1.0 - exp(-ext)) * (BETA_R / BETA_R.b) * 0.006 * day;
    return SUN_E * tint * (single + multi) + NIGHT;
}

vec3 procedural_sky(vec3 v, vec3 s, vec3 tint, float nu_width) {
    float hl = length(v.xz);
    vec3 horizon = sky_scatter(hl > 1e-5 ? vec3(v.x / hl, 0.0, v.z / hl) : vec3(1.0, 0.0, 0.0), s, tint);
    vec3 color;
    if (v.y >= 0.0) {
        color = sky_scatter(v, s, tint);
        // Sun disc (limb darkened), antialiased over this pixel's footprint
        // (nu_width = fwidth of cos(angle to the sun): fixed per pixel, no shimmer).
        float nu = dot(v, s);
        float cos_r = cos(SUN_RADIUS);
        float aa = max(nu_width, 1e-6);
        float disc = smoothstep(cos_r - aa, cos_r + aa, nu);
        if (disc > 0.0) {
            float t = clamp(acos(clamp(nu, -1.0, 1.0)) / SUN_RADIUS, 0.0, 1.0);
            float limb = 1.0 - 0.6 * (1.0 - sqrt(max(1.0 - t * t, 0.0)));
            // The disc sinks behind the horizon over its own radius.
            float above = smoothstep(-SUN_RADIUS, SUN_RADIUS, v.y);
            color += SUN_E * SUN_DISC * tint * transmittance(max(v.y, 0.0)) * disc * limb * above;
        }
    } else {
        // Virtual ground: diffuse albedo lit by the transmitted sun and the
        // sky, fading into the horizon haze (aerial perspective).
        float sun_up = max(s.y, 0.0);
        vec3 zenith = sky_scatter(vec3(0.0, 1.0, 0.0), s, tint);
        vec3 lit = GROUND_ALBEDO / PI * (SUN_E * tint * transmittance(s.y) * sun_up * 0.35 + zenith * PI);
        float haze = exp(-(-v.y) * 9.0);
        color = mix(lit, horizon, haze);
    }
    return color;
}

// ---------------------------------------------------------------------------
// Equirect: u = atan(z, x) / 2pi + 0.5, v = acos(y) / pi (the CPU convention of
// ReflectionEnv::from_equirect / cube_chain_to_equirect). Gradients come
// analytically from the ray's screen derivatives, so the u wrap at the -X
// meridian has no derivative jump and the mip / anisotropy footprint is the
// pixel's true one, also toward the poles.
// ---------------------------------------------------------------------------
vec3 sample_equirect(vec3 n, vec3 dndx, vec3 dndy) {
    float u = atan(n.z, n.x) / TAU + 0.5;
    float v = acos(clamp(n.y, -1.0, 1.0)) / PI;
    float rxz2 = max(n.x * n.x + n.z * n.z, 1e-12);
    float rxz = sqrt(rxz2);
    // du is scaled by sin(theta) (= rxz): an equirect row holds only
    // 2pi sin(theta) worth of angle, so toward the poles it is oversampled in u
    // and its content there is already band-limited. Measuring the u footprint
    // in true angle keeps the mip choice set by the pixel's real angular size:
    // finite at the pole (no 1/sin blow-up into coarse mips that average rows
    // far from the pole, which shows as a spot) and seam-free (no atan jump).
    vec2 gx = vec2((n.x * dndx.z - n.z * dndx.x) / (TAU * rxz), -dndx.y / (PI * rxz));
    vec2 gy = vec2((n.x * dndy.z - n.z * dndy.x) / (TAU * rxz), -dndy.y / (PI * rxz));
    return textureGrad(t_sky, vec2(u, v), gx, gy).rgb;
}

// ---------------------------------------------------------------------------
// Inline post (LDR targets that skip the post pass).
// ---------------------------------------------------------------------------
vec3 tonemap_aces(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 apply_postfx(vec3 color, vec2 fragcoord) {
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

// Interleaved gradient noise (Jimenez 2014) of the pixel position only: a
// fixed pattern, identical every frame.
float ign(vec2 p) {
    return fract(52.9829189 * fract(dot(floor(p), vec2(0.06711056, 0.00583715))));
}

void main() {
    mat3 k = mat3(v_k0, v_k1, v_k2);
    vec3 d = k * vec3(v_ndc, 1.0);
    float len = length(d);
    vec3 n = d / len;
    // Screen derivatives of the ray, analytically (d is linear in ndc): exact
    // and identical for every pixel of a 2x2 quad edge.
    vec3 ddx = k * vec3(dFdx(v_ndc), 0.0);
    vec3 ddy = k * vec3(dFdy(v_ndc), 0.0);
    vec3 dndx = (ddx - n * dot(n, ddx)) / len;
    vec3 dndy = (ddy - n * dot(n, ddy)) / len;

    vec3 color;
    if (pc.params0.z > 0.5) {
        color = textureLod(u_env, n, 0.0).rgb;
    } else if (pc.params0.x > 0.5) {
        color = sample_equirect(n, dndx, dndy);
    } else {
        // Sun: against the main directional light; a missing / zero light falls
        // back to the default sun so the sky never goes NaN.
        vec3 ld = frame.light_dir.xyz;
        float ll = dot(ld, ld);
        vec3 s = (ll > 1e-8 && ll < 1e8) ? -ld * inversesqrt(ll) : normalize(vec3(0.4, 1.0, 0.3));
        // Light colour sets the sun's tint only; the sky's level is fixed
        // (scene exposure does not follow the light intensity).
        vec3 lc = max(frame.light_color.rgb, vec3(0.0));
        float lm = max(lc.r, max(lc.g, lc.b));
        vec3 tint = lm > 1e-6 ? mix(vec3(1.0), lc / lm, 0.6) : vec3(1.0);
        float nu_width = abs(dot(dndx, s)) + abs(dot(dndy, s));
        color = procedural_sky(n, s, tint, nu_width);
    }
    color = max(color, vec3(0.0));

    // ZESTENGINE_SKY_DEBUG: emission.w > 0.5 tints the sky with emission.rgb (frame parity).
    if (pc.emission.w > 0.5) color = mix(color, pc.emission.rgb, 0.5);
    o_gbuf = vec4(0.0);
    if (pc.params0.w > 0.5) {
        o_color = vec4(color, 1.0);
    } else {
        vec3 c = apply_postfx(color, gl_FragCoord.xy);
        // +-0.5 LSB of 8-bit sRGB, fixed per pixel.
        vec3 e = pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2)) + (ign(gl_FragCoord.xy) - 0.5) / 255.0;
        o_color = vec4(pow(clamp(e, 0.0, 1.0), vec3(2.2)), 1.0);
    }
}
