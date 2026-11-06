// Opaque scene sampled by water/transmission. Multiview copies have one layer
// per eye; neither coordinates nor depth reconstruction use the centre camera.
#ifdef MULTIVIEW
layout(set = 0, binding = 5) uniform sampler2DArray u_opaque_depth;
layout(set = 0, binding = 6) uniform sampler2DArray u_opaque_color;
vec4 opaque_color_at(vec2 uv, float lod) {
    return textureLod(u_opaque_color, vec3(uv, gl_ViewIndex), lod);
}
float opaque_depth_at(ivec2 pixel) {
    return texelFetch(u_opaque_depth, ivec3(pixel, gl_ViewIndex), 0).r;
}
#else
layout(set = 0, binding = 5) uniform sampler2D u_opaque_depth;
layout(set = 0, binding = 6) uniform sampler2D u_opaque_color;
vec4 opaque_color_at(vec2 uv, float lod) { return textureLod(u_opaque_color, uv, lod); }
float opaque_depth_at(ivec2 pixel) { return texelFetch(u_opaque_depth, pixel, 0).r; }
#endif

float water_scene_dist(vec2 frag) {
    ivec2 size = textureSize(u_opaque_depth, 0).xy;
    float depth = opaque_depth_at(clamp(ivec2(frag), ivec2(0), size - 1));
    if (depth >= 1.0) return 1e6;
    vec2 uv = frag / vec2(size);
#ifdef MULTIVIEW
    mat4 inv_proj = frame.water_inv_proj[gl_ViewIndex];
#else
    mat4 inv_proj = frame.inv_proj;
#endif
    vec4 p = inv_proj * vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    return abs(p.w) > 1e-8 ? length(p.xyz / p.w) : 1e6;
}
