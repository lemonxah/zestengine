// Shared final HDR fog composition. Fog is never accumulated in reflection history.
// --- Volumetric fog: a real raymarched participating medium (animated, patchy,
// with sun in-scattering), so fog hangs + drifts in the air instead of just
// tinting surface colour. ---


float reflection_fog_hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

float reflection_fog_vnoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(reflection_fog_hash13(i + vec3(0, 0, 0)), reflection_fog_hash13(i + vec3(1, 0, 0)), f.x),
            mix(reflection_fog_hash13(i + vec3(0, 1, 0)), reflection_fog_hash13(i + vec3(1, 1, 0)), f.x), f.y),
        mix(mix(reflection_fog_hash13(i + vec3(0, 0, 1)), reflection_fog_hash13(i + vec3(1, 0, 1)), f.x),
            mix(reflection_fog_hash13(i + vec3(0, 1, 1)), reflection_fog_hash13(i + vec3(1, 1, 1)), f.x), f.y),
        f.z);
}

float reflection_fog_fbm(vec3 p) {
    float a = 0.5, s = 0.0;
    for (int i = 0; i < 3; ++i) {
        s += a * reflection_fog_vnoise(p);
        p *= 2.02;
        a *= 0.5;
    }
    return s;
}

// Henyey-Greenstein phase (forward-scatter sun glow when looking toward the sun).
float reflection_fog_hg_phase(float cosT, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * 3.14159265359 * pow(max(1.0 + g2 - 2.0 * g * cosT, 1e-4), 1.5));
}

// March from the camera to `endWorld`, accumulating animated height-fog density +
// sun in-scattering. Returns the fogged colour.
vec3 reflection_fog_apply_fog(vec3 color, vec3 endWorld, vec3 camera, vec4 fog_color, vec4 fog_params, vec4 fog_light, vec4 fog_sun) {
    vec3 ro = camera;
    vec3 seg = endWorld - ro;
    float dist = length(seg);
    if (dist < 1e-3) return color;
    vec3 rd = seg / dist;
    const int STEPS = 48;
    float stepLen = dist / float(STEPS);
    vec3 wind = vec3(0.06, 0.0, 0.04) * fog_sun.w; // drift over time
    float phase = reflection_fog_hg_phase(dot(rd, normalize(fog_sun.xyz)), fog_light.w);
    float start = fog_params.z;
    float transmittance = 1.0;
    vec3 inscatter = vec3(0.0);
    for (int i = 0; i < STEPS; ++i) {
        float d = stepLen * (float(i) + 0.5);
        if (d < start) continue;
        vec3 p = ro + rd * d;
        float height = exp(-fog_params.x * max(0.0, p.y - fog_params.y));
        float n = reflection_fog_fbm(p * 0.08 + wind); // patchy, drifting density
        float dens = fog_color.w * height * clamp(0.25 + 1.5 * n, 0.0, 2.0) * stepLen;
        if (dens <= 0.0) continue;
        vec3 lit = fog_color.rgb + fog_light.rgb * phase;
        float a = 1.0 - exp(-dens);
        inscatter += transmittance * a * lit;
        transmittance *= exp(-dens);
        if (transmittance < 0.01) break;
    }
    return color * transmittance + inscatter;
}

