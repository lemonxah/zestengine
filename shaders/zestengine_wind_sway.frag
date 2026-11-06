// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// @shader "ZestEngine Wind Sway"
//
// The template for a VERTEX HOOK (docs/shaders.md, "Vertex hook"): grass, leaves,
// banners and cloth-like props that sway in the wind. The `// @vertex` section below
// defines `zestengine_vertex(inout VertexData v)`, which the engine runs in every
// pass that draws the material: the main pass, the depth prepass, the XR pass, planar
// mirrors and the shadow pass, with the same maths, so the shadow follows the swaying
// surface and the EQUAL depth test of the prepass never disagrees with the main pass.
//
// Setup: the sway is rooted at object-space y = 0 and grows with height, so model
// the plant standing on y = 0. Give the draw a bound with some margin (the sway moves
// vertices outside the mesh's own bounds) and keep Cast Shadows on.
//
// @property tint "Tint" Color = (0.35, 0.8, 0.25, 1)
// @property albedo "Albedo" Texture2D = white
// @property sway "Sway (m)" Range(0, 1) = 0.15 @tooltip("Sideways travel at the top of the plant, in world units")
// @property sway_speed "Speed" Range(0, 10) = 2 @tooltip("Radians per second of the main wave")
// @property sway_height "Plant Height (m)" Float = 1 @tooltip("Object-space height where the sway is at full strength")
// @property sway_gust "Gust" Range(0, 1) = 0.5 @tooltip("Strength of the second, faster wave")
// @property sway_dir "Wind Direction" Range(0, 6.2831853) = 0.6 @tooltip("Radians around +Y")
//
// @vertex
void zestengine_vertex(inout VertexData v) {
    // 0 at the roots, 1 at the top; the squared falloff keeps the base still.
    float h = clamp(v.position.y / max(sway_height, 1e-3), 0.0, 1.0);
    h *= h;
    // The waves travel across the field: a phase from the vertex's world position.
    float phase = dot(v.world_position.xz, vec2(0.7, 0.45));
    float wave = sin(v.time * sway_speed + phase) + sway_gust * sin(v.time * sway_speed * 2.3 + phase * 1.7);
    vec2 wind = vec2(cos(sway_dir), sin(sway_dir));
    v.world_offset.xz += wind * (wave * sway * h);
    // Dip a little as it leans, so the blade keeps its length.
    v.world_offset.y -= abs(wave) * sway * sway * h * 0.5;
}
// @endvertex

void main() {
    vec4 base = texture(albedo, albedo_uv(v_uv)) * tint * v_color;
    vec3 n = normalize(v_normal);
    if (!gl_FrontFacing) n = -n;
    // Sun, shadowed and lit from both faces (thin foliage), plus the ambient.
    vec3 lit = zestengine_direct_lambert(v_world_pos, n) + zestengine_direct_lambert(v_world_pos, -n) * 0.4 + frame.ambient.rgb;
    o_color = vec4(zestengine_postfx(base.rgb * lit, gl_FragCoord.xy), 1.0);
}
