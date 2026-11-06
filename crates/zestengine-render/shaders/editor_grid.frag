// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// Editor ground grid (Scene view only): a fullscreen pass over the tonemapped
// viewport. Each pixel's view ray is intersected with the grid plane (the
// world plane through the origin whose normal is axis `params.z`), the hit is
// depth tested by hand against the scene depth (opaque geometry in front
// hides it), and anti-aliased lines are blended over the colour.
//
// Level of detail per pixel: the minor spacing is `unit * 10^n`, the smallest
// power of ten whose cells stay at least MIN_PX pixels apart, and the lines of
// a level fade out as they approach that density, so moving the camera blends
// smoothly between levels. `grid_lod` in grid.rs mirrors this formula.

layout(set = 0, binding = 0) uniform sampler2D u_depth; // scene depth (depth aspect)

layout(push_constant) uniform Push {
    mat4 inv_view_proj; // world = inv_view_proj * clip (the scene pass's matrices)
    vec4 params;        // x unit (m), y opacity, z plane normal axis (0 X, 1 Y, 2 Z), w 1 = orthographic
    vec4 line_color;    // rgb line colour (target space), a minor-line alpha
    vec4 axis_u;        // rgb colour of the line along the plane's first axis (v = 0), a major-line alpha
    vec4 axis_v;        // rgb colour of the line along the plane's second axis (u = 0), a axis alpha
} pc;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

const float MIN_PX = 3.0;

vec3 unproject(vec2 ndc, float z) {
    vec4 w = pc.inv_view_proj * vec4(ndc, z, 1.0);
    return w.xyz / w.w;
}

// Coverage (0..1) of lines through the integers of `c` (coords in cells),
// `dc` cells per pixel, `width` in pixels. Families denser than ~4 px fade out
// on their own (the anisotropic direction at grazing angles).
float lines(vec2 c, vec2 dc, float width) {
    dc = max(dc, vec2(1e-6));
    vec2 dist_px = abs(fract(c + 0.5) - 0.5) / dc;
    vec2 a = clamp(width * 0.5 + 0.5 - dist_px, 0.0, 1.0);
    a *= 1.0 - smoothstep(0.15, 0.3, dc);
    return max(a.x, a.y);
}

float line1(float c, float dc, float width) {
    dc = max(dc, 1e-6);
    float a = clamp(width * 0.5 + 0.5 - abs(c) / dc, 0.0, 1.0);
    return a * (1.0 - smoothstep(0.15, 0.3, dc));
}

void main() {
    ivec2 px = ivec2(gl_FragCoord.xy);
    vec2 size = vec2(textureSize(u_depth, 0));
    vec2 uv = (vec2(px) + 0.5) / size;
    // The scene pass draws with a flipped viewport: the top row is NDC y = +1.
    vec2 ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    vec3 near_p = unproject(ndc, 0.0);
    vec3 far_p = unproject(ndc, 1.0);
    vec3 seg = far_p - near_p;
    int axis = int(pc.params.z + 0.5);
    float denom = seg[axis];
    float t = abs(denom) > 1e-9 ? -near_p[axis] / denom : -1.0;
    vec3 hit = near_p + seg * clamp(t, 0.0, 1.0);

    // Plane coordinates (u, v) in grid units; derivatives before any discard.
    vec2 coord = axis == 0 ? hit.yz : (axis == 1 ? hit.xz : hit.xy);
    float unit = max(pc.params.x, 1e-4);
    vec2 cw = coord / unit;
    vec2 dudv = vec2(length(vec2(dFdx(cw.x), dFdy(cw.x))), length(vec2(dFdx(cw.y), dFdy(cw.y))));

    float lod = max(0.0, log2(length(dudv) * MIN_PX) * 0.30103 + 1.0);
    float level = floor(lod);
    float fade = lod - level;
    float s0 = pow(10.0, level);
    float s1 = s0 * 10.0;
    float s2 = s1 * 10.0;
    float l0 = lines(cw / s0, dudv / s0, 1.0);
    float l1 = lines(cw / s1, dudv / s1, 1.0);
    float l2 = lines(cw / s2, dudv / s2, 1.25);
    float minor_a = pc.line_color.a;
    float major_a = pc.axis_u.a;
    float a = max(l2 * major_a, max(l1 * mix(major_a, minor_a, fade), l0 * minor_a * (1.0 - fade)));

    // The two axis lines through the origin.
    float au = line1(cw.y, dudv.y, 2.0); // v = 0: runs along u
    float av = line1(cw.x, dudv.x, 2.0); // u = 0: runs along v
    float axis_a = pc.axis_v.a;
    vec3 col = pc.line_color.rgb;
    float ax = max(au, av);
    if (ax > 0.0) {
        vec3 acol = au >= av ? pc.axis_u.rgb : pc.axis_v.rgb;
        col = mix(col, acol, ax);
        a = max(a, ax * axis_a);
    }

    // Fades: grazing angles (thins the far field toward the horizon) and,
    // in perspective, the far end of the view.
    vec3 dir = normalize(seg);
    float grazing = smoothstep(0.0, 0.06, abs(dir[axis]));
    float far_fade = pc.params.w > 0.5 ? 1.0 : 1.0 - smoothstep(0.4, 0.95, t);
    a *= grazing * far_fade * clamp(pc.params.y, 0.0, 1.0);

    // Hidden behind opaque geometry (transparent draws write no depth). A
    // small relative tolerance keeps the grid on a coplanar floor.
    float d = texelFetch(u_depth, px, 0).r;
    if (d < 1.0) {
        vec3 scene = unproject(ndc, d);
        float ds = length(scene - near_p);
        float dg = t * length(seg);
        if (dg > ds + max(ds * 2e-3, 1e-3)) a = 0.0;
    }
    if (t <= 0.0 || t >= 1.0 || a < 1.0 / 512.0) discard;
    o_color = vec4(col, a);
}
