// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

//! shader "Toon (Poiyomi-lite)"
//! prop tint color default(1, 1, 1, 1)
//! prop rim_color color3 default(1, 1, 1)
//! prop rim_strength float range(0, 3) default(0.6)
//! prop rim_power float range(0.5, 8) default(4)
//! prop ramp_steps float range(2, 8) default(3)
//! prop ramp_smooth float range(0, 0.5) default(0.12)
//! prop spec_strength float range(0, 2) default(0.35)
//! prop spec_size float range(0, 1) default(0.1)
//! prop emission_strength float range(0, 8) default(0)
//
// A lightweight Poiyomi-style toon shader on the zestengine standard pipeline, so it
// shares the scene's lights, shadows, probe GI (lightdata) and baked lightmaps.
// Cel ramp + rim + toon specular over a PBR-ish base, all tunable in the
// Inspector via the props above.

// Quantize x into `steps` cel bands with a soft edge of width `smooth`.
float toon_ramp(float x, float steps, float smooth_w) {
    float s = clamp(x, 0.0, 1.0) * steps;
    float lower = floor(s);
    float w = max(smooth_w * steps, 1e-3);
    float edge = smoothstep(0.5 - w, 0.5 + w, fract(s));
    return (lower + edge) / steps;
}

void main() {
    vec4 albedo = texture(t_albedo, v_uv) * tint * v_color;

    vec3 N = normalize(v_normal);
    vec3 V = normalize(u_camera_pos - v_world_pos);
    float ndv = max(dot(N, V), 1e-4);

    // Combine the key directional + all point/spot lights, then cel-band the
    // total intensity while preserving hue (classic toon look).
    vec3 lights = u_light_color * max(dot(N, -u_light_dir), 0.0)
                + zestengine_direct_diffuse(v_world_pos, N);
    float lum = max(max(lights.r, lights.g), lights.b);
    float ramp = toon_ramp(lum, ramp_steps, ramp_smooth);
    vec3 direct = lights * (ramp / max(lum, 1e-4));

    // Toon specular: a crisp Blinn highlight, hard-edged into the lit band.
    vec3 H = normalize(-u_light_dir + V);
    float spec = pow(max(dot(N, H), 0.0), mix(128.0, 4.0, spec_size));
    spec = smoothstep(0.5, 0.52, spec) * spec_strength;

    // Baked / probe GI (matches the standard shader's indirect).
    vec3 indirect = zestengine_baked_gi(v_world_pos, N);

    vec3 color = albedo.rgb * (direct + indirect) + spec * u_light_color;

    // Fresnel rim light, tinted and added as an edge glow.
    float rim = pow(1.0 - ndv, rim_power) * rim_strength;
    color += rim * rim_color;

    color += texture(t_emission, v_uv).rgb * emission_strength;

    color = zestengine_postfx(color, gl_FragCoord.xy);
    o_color = vec4(color, albedo.a);
}
