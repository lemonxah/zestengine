// Exact transformed dry-box queries. WATER_COVERAGE_DATA names the frame's
// vec4 storage array; every matrix transforms water-local to [-1,1]^3.
uint water_exclusion_count(GpuWater body) {
    uint size = uint(WATER_COVERAGE_DATA.length());
    return body.exclusions.x <= size ? min(body.exclusions.y, (size - body.exclusions.x) / 4u) : 0u;
}
mat4 water_exclusion_matrix(GpuWater body, uint i) {
    uint o = body.exclusions.x + i * 4u;
    return mat4(WATER_COVERAGE_DATA[o], WATER_COVERAGE_DATA[o+1u], WATER_COVERAGE_DATA[o+2u], WATER_COVERAGE_DATA[o+3u]);
}
bool water_excluded(GpuWater body, vec3 local) {
    if (body.exclusions.y == 0u) return false;
    uint count = water_exclusion_count(body);
    for (uint i = 0u; i < count; ++i) {
        vec3 p = (water_exclusion_matrix(body, i) * vec4(local, 1.0)).xyz;
        if (all(lessThanEqual(abs(p), vec3(1.0)))) return true;
    }
    return false;
}
// Direction retains its original parameterization through the transform;
// world-unit rays consequently return world metres with arbitrary box scale.
bool water_box_interval(mat4 to_unit, vec3 origin, vec3 direction, vec2 range, out vec2 hit) {
    vec3 p = (to_unit * vec4(origin, 1.0)).xyz;
    vec3 d = (to_unit * vec4(direction, 0.0)).xyz;
    hit = range;
    for (int axis = 0; axis < 3; ++axis) {
        if (abs(d[axis]) < 1e-12) {
            if (abs(p[axis]) > 1.0) return false;
        } else {
            vec2 t = (vec2(-1.0, 1.0) - p[axis]) / d[axis];
            hit.x = max(hit.x, min(t.x, t.y));
            hit.y = min(hit.y, max(t.x, t.y));
        }
    }
    return hit.y > hit.x;
}
// Union length, including overlapping/nested boxes, without a fixed volume
// limit or ray steps which could skip a thin hull. At most two events per box.
float water_dry_distance(GpuWater body, vec3 origin, vec3 direction, vec2 range) {
    if (body.exclusions.y == 0u || range.y <= range.x) return 0.0;
    uint count = water_exclusion_count(body);
    float cursor = range.x, dry = 0.0;
    for (uint step = 0u; step <= count * 2u && cursor < range.y; ++step) {
        float next = range.y, end = cursor;
        for (uint i = 0u; i < count; ++i) {
            vec2 hit;
            if (!water_box_interval(water_exclusion_matrix(body, i), origin, direction, range, hit) || hit.y <= cursor) continue;
            if (hit.x <= cursor) end = max(end, hit.y);
            else next = min(next, hit.x);
        }
        if (end > cursor) { dry += end - cursor; cursor = end; }
        else cursor = next;
    }
    return min(dry, max(range.y-range.x, 0.0));
}
