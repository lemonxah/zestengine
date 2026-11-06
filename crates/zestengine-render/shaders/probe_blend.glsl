// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Reflection-probe helpers for the passes that composite over the forward
// pass's environment reflection (SSR resolve, RT reflections). They must match
// standard.frag's probe_dir / probe_influence, or those passes subtract a
// different environment than the forward pass added.

// Box-projected reflection direction for a probe; the raw direction for a
// surface outside its box (flush surfaces count as inside).
vec3 probe_box_dir(vec3 P, vec3 R, vec3 capture, vec3 center, vec3 extents, bool project) {
    if (!project) return R;
    vec3 bmin = center - extents;
    vec3 bmax = center + extents;
    vec3 eps = extents * 0.02 + 0.02;
    bool inside = all(greaterThanEqual(P, bmin - eps)) && all(lessThanEqual(P, bmax + eps));
    if (!inside) return R;
    vec3 invR = 1.0 / R;
    vec3 tmax = max((bmax - P) * invR, (bmin - P) * invR);
    float dist = max(min(min(tmax.x, tmax.y), tmax.z), 0.0);
    return P + R * dist - capture;
}

// 0 outside the box, ramping to 1 across the blend shell inside its faces.
float probe_influence(vec3 p, vec3 center, vec3 extents, float blend) {
    vec3 d = extents - abs(p - center);
    float inner = min(d.x, min(d.y, d.z));
    if (blend <= 1e-4) return inner >= 0.0 ? 1.0 : 0.0;
    return clamp(inner / blend, 0.0, 1.0);
}

// Weight of the second probe (slot 1) at p; `b_on_top` = slot 1 wins where
// both boxes overlap. Outside both, slot 0 is used.
float probe_b_weight(vec3 p, vec3 box0, vec3 ext0, float blend0, vec3 box1, vec3 ext1, float blend1, bool b_on_top) {
    float f0 = probe_influence(p, box0, ext0, blend0);
    float f1 = probe_influence(p, box1, ext1, blend1);
    return b_on_top ? f1 : f1 * (1.0 - f0);
}

// Forward-pass multi-probe blend (3+ probes): slot 0 is the fallback and every
// active slot contributes `sample * intensity * influence`, normalised by the
// total influence. `extra` must be true only when at least one of slots 2/3 is
// active, so the two-probe path above is left untouched.
vec3 probe_env(vec3 worldPos, vec3 R_raw, float roughness, bool extra) {
    if (!extra) return vec3(0.0);
    float wsum = 0.0;
    vec3 acc = vec3(0.0);
    float intensity = u.refl_center.w;
    if (intensity > 0.0) {
        vec3 R = probe_box_dir(worldPos, R_raw, u.refl_center.xyz, u.refl_box.xyz, u.refl_extents.xyz, u.refl_extents.w > 0.5);
        float w = probe_influence(worldPos, u.refl_box.xyz, u.refl_extents.xyz, u.refl_blend.x);
        acc += textureLod(u_env, R, roughness * ENV_MAX_LOD).rgb * intensity * w;
        wsum += w;
    }
    float i2 = u.refl2_center.w;
    if (i2 > 0.0) {
        vec3 R2 = probe_box_dir(worldPos, R_raw, u.refl2_center.xyz, u.refl2_box.xyz, u.refl2_extents.xyz, u.refl2_extents.w > 0.5);
        float w = probe_influence(worldPos, u.refl2_box.xyz, u.refl2_extents.xyz, u.refl_blend.y);
        acc += textureLod(u_env_b, R2, roughness * float(textureQueryLevels(u_env_b) - 1)).rgb * i2 * w;
        wsum += w;
    }
    float i3 = u.refl3_center.w;
    if (i3 > 0.0) {
        vec3 R3 = probe_box_dir(worldPos, R_raw, u.refl3_center.xyz, u.refl3_box.xyz, u.refl3_extents.xyz, u.refl3_extents.w > 0.5);
        float w = probe_influence(worldPos, u.refl3_box.xyz, u.refl3_extents.xyz, u.refl_blend2.x);
        acc += textureLod(u_env_c, R3, roughness * float(textureQueryLevels(u_env_c) - 1)).rgb * i3 * w;
        wsum += w;
    }
    float i4 = u.refl4_center.w;
    if (i4 > 0.0) {
        vec3 R4 = probe_box_dir(worldPos, R_raw, u.refl4_center.xyz, u.refl4_box.xyz, u.refl4_extents.xyz, u.refl4_extents.w > 0.5);
        float w = probe_influence(worldPos, u.refl4_box.xyz, u.refl4_extents.xyz, u.refl_blend2.y);
        acc += textureLod(u_env_d, R4, roughness * float(textureQueryLevels(u_env_d) - 1)).rgb * i4 * w;
        wsum += w;
    }
    return wsum > 1e-4 ? acc / wsum : vec3(0.0);
}
