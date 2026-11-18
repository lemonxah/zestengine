// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.

// ShadowPush packs a sphere intersected with a light-local box. Test the
// interpolated world position, not interpolated distances at triangle corners:
// the latter cannot cut a hole through a coarse triangle.
bool shadow_region_contains(mat4 region, vec3 world) {
    float radius = region[0].w;
    if (radius <= 0.0) return false;
    vec3 d = world - region[1].yzw;
    vec4 q = region[3];
    vec3 p = d + 2.0 * cross(q.xyz, cross(q.xyz, d) + q.w * d);
    vec3 positive = vec3(region[2].x, region[0].y, region[2].z);
    vec3 negative = vec3(region[2].y, region[0].z, region[2].w);
    return dot(d, d) < radius * radius
        && all(lessThan(p, positive)) && all(lessThan(-p, negative));
}
