// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// @shader "Terrain Splat"
// @property splat_r "Layer R (Base)" Texture2D = white @srgb
// @property splat_g "Layer G" Texture2D = white @srgb
// @property splat_b "Layer B" Texture2D = white @srgb
// @property splat_a "Layer A" Texture2D = white @srgb
// @property splat_tiling "Splat Tiling" Vector2 = (16, 16)

// Heightfield terrain splat: four albedo layers blended by the per-vertex
// weights painted with the Terrain texture brush (R/G/B/A -> the four
// layers). Vertex color is filled by the engine's heightfield mesh; unpainted
// vertices are (1,0,0,0) so the base layer shows.

void main() {
    vec4 w = v_color;
    vec2 uv = v_uv * splat_tiling;
    vec4 albedo = texture(splat_r, splat_r_uv(uv)) * w.r
                + texture(splat_g, splat_g_uv(uv)) * w.g
                + texture(splat_b, splat_b_uv(uv)) * w.b
                + texture(splat_a, splat_a_uv(uv)) * w.a;
    vec3 n = normalize(gl_FrontFacing ? v_normal : -v_normal);
    vec3 color = albedo.rgb * (zestengine_direct_lambert(v_world_pos, n) + zestengine_baked_gi(v_world_pos, n));
    o_color = vec4(zestengine_postfx(color, gl_FragCoord.xy), albedo.a);
}
