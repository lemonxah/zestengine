// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.

#ifndef FLUX_SURFACE_NORMAL
#define FLUX_SURFACE_NORMAL
// Pixel tangents shrink with camera distance and render resolution. Their
// cross product has units of area, so an absolute length cutoff rejects valid
// close-up surfaces. Scale each tangent first: degeneracy then depends only on
// their angle, while zero-length/collinear tangents still fail safely.
bool flux_surface_normal(vec3 dx, vec3 dy, out vec3 normal) {
    normal = vec3(0.0);
    float sx = max(abs(dx.x), max(abs(dx.y), abs(dx.z)));
    float sy = max(abs(dy.x), max(abs(dy.y), abs(dy.z)));
    if (!(sx > 0.0) || !(sy > 0.0)) return false;
    vec3 c = cross(dx / sx, dy / sy);
    float squared = dot(c, c);
    if (!(squared > 1e-12) || isinf(squared)) return false;
    normal = c * inversesqrt(squared);
    return true;
}
#endif
