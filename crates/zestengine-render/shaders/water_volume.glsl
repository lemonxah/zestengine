// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Single scattering along the refracted water column. The existing light
// attenuation, area-light evaluation, layer masks and shadow maps are inputs;
// this module does not change the renderer's lighting or GI policy.
vec3 water_volume_lighting(int slot, vec3 integral, float path, vec3 up, vec3 view) {
    if (max(integral.r, max(integral.g, integral.b)) < 1e-6) return vec3(0.0);
    GpuWater water = frame.water_bodies[slot];
    vec3 incident = max(frame.ambient.rgb, vec3(0.0));
    uvec4 mask = floatBitsToUint(pc.model[0]);
    int count = min(int(frame.misc.y + 0.5), MAX_LIGHTS);
    // Surface shadowing and midpoint incident attenuation approximate the
    // light integral; the view-ray optical integral itself is analytic.
    float mid_depth = max(path * 0.5 * abs(dot(view, up)), 0.0);
    for (int word = 0; word < 4; ++word) {
        uint bits = mask[word];
        while (bits != 0u) {
            int i = word * 32 + findLSB(bits);
            bits &= bits - 1u;
            if (i >= count) break;
            Light light = frame.lights[i];
            int kind = int(light.pos_kind.w + 0.5);
            if (kind == 3 || kind == 6) continue;
            vec3 delta = light.pos_kind.xyz - v_world_pos;
            float distance_m = length(delta);
            vec3 L = kind == 0 ? normalize(-light.dir_range.xyz) : delta / max(distance_m, 1e-5);
            vec3 radiance;
            if (kind == 4 || kind == 5) {
                if (i + 1 >= count) continue;
                vec3 diffuse, specular;
                float form_factor;
                area_light_eval(i, v_world_pos, up, view, max(dot(up, view), 1e-4), 1.0, vec3(0.0), diffuse, specular, form_factor);
                radiance = diffuse * 3.14159265359;
            } else {
                float attenuation = kind == 0 ? 1.0 : realtime_distance_falloff(distance_m, light.dir_range.w);
                if (kind == 2) attenuation *= realtime_spot_falloff(dot(normalize(light.dir_range.xyz), -L), light.spot.x, light.color.w);
                radiance = light.color.rgb * attenuation * shadow_factor(light, v_world_pos, up, L);
            }
            float light_path = mid_depth / max(abs(dot(up, L)), 0.1);
            if (kind != 0) light_path = min(light_path, distance_m);
            vec3 attenuation = water_transmittance(water_absorption(water).rgb, water_scattering_coeff(water).rgb, light_path);
            incident += max(radiance, vec3(0.0)) * attenuation * water_phase(dot(-L, view), water_scattering_coeff(water).w);
        }
    }
    return integral * incident;
}
