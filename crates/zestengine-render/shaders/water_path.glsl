// Optical length inside a basin's authored footprint and floor, minus dry
// volumes. Callers supply the interval already clipped to the moving surface.
// Local directions retain world-unit ray parameterization (do not normalize).
// River channels additionally intersect their varying rest surface and floor.
bool water_path_slab(float p, float d, float lo, float hi, inout vec2 span) {
    if (abs(d) < 1e-12) return p >= lo && p <= hi;
    vec2 t = (vec2(lo, hi) - p) / d;
    span = vec2(max(span.x, min(t.x, t.y)), min(span.y, max(t.x, t.y)));
    return span.y > span.x;
}
float water_path_cross(vec2 a, vec2 b) { return a.x*b.y-a.y*b.x; }

// Find the next shore crossing in ray order. Stackless bounds pruning avoids
// inspecting every edge on every interval. No ray march or fixed event array:
// narrow channels and repeated exits/re-entries in concave shores survive.
float water_next_shore(GpuWater body, vec2 origin, vec2 direction, vec2 span) {
    uint base = body.coverage.y, count = body.coverage.z;
    float next = span.y;
    uint i = 0u;
    while (i + 1u < count) {
        vec4 box = WATER_COVERAGE_DATA[base+i];
        uvec4 meta = floatBitsToUint(WATER_COVERAGE_DATA[base+i+1u]);
        if (meta.x <= i || meta.x > count) break;
        // Include zero-width edge bounds: a shore intersection is a point.
        vec2 hit = vec2(span.x, next);
        bool candidate = true;
        for (int axis = 0; axis < 2; ++axis) {
            if (abs(direction[axis]) < 1e-12) {
                candidate = candidate && origin[axis] >= box[axis] && origin[axis] <= box[axis+2];
            } else {
                vec2 t = (vec2(box[axis], box[axis+2])-origin[axis])/direction[axis];
                hit = vec2(max(hit.x, min(t.x,t.y)), min(hit.y, max(t.x,t.y)));
            }
        }
        if (!candidate || hit.y+1e-6*max(1.0,abs(hit.y)) < hit.x) { i = meta.x; continue; }
        if (meta.y == 0u) { i += 2u; continue; }
        if (i + 2u < count) {
            vec4 ab = WATER_COVERAGE_DATA[base+i+2u];
            vec2 edge=ab.zw-ab.xy;float determinant=water_path_cross(direction,edge);
            // Include shared vertices. An extra tangent partition is harmless
            // because interval membership is queried afterward; dropping both
            // sides of a rounded corner crossing can lose an entire island.
            if(abs(determinant)>1e-12){
                float along=water_path_cross(ab.xy-origin,direction)/determinant;
                float t=water_path_cross(ab.xy-origin,edge)/determinant;
                if(along>=-1e-6&&along<=1.000001&&t>span.x)next=min(next,t);
            }
        }
        i = meta.x;
    }
    return next;
}

#include "water_river_path.glsl"
#include "water_mesh_path.glsl"

float water_medium_distance(GpuWater body, vec3 origin, vec3 direction, vec2 span) {
    if (span.y <= span.x) return 0.0;
    uint kind = body.coverage.x;
    if (kind == 3u) return water_river_distance(body,origin,direction,span);
    if (kind == 4u) return water_mesh_distance(body,origin,direction,span);
    if (kind != 0u && kind != 3u) {
        float floor_y = body.bounds.z - uintBitsToFloat(body.coverage.w);
        if (abs(direction.y) < 1e-12) {
            if (origin.y < floor_y) return 0.0;
        } else {
            float t = (floor_y-origin.y)/direction.y;
            if (direction.y > 0.0) span.x = max(span.x,t);
            else span.y = min(span.y,t);
        }
    }
    if (kind <= 1u) {
        if (body.bounds.x > 0.0 && !water_path_slab(origin.x,direction.x,-body.bounds.x,body.bounds.x,span)) return 0.0;
        if (body.bounds.y > 0.0 && !water_path_slab(origin.z,direction.z,-body.bounds.y,body.bounds.y,span)) return 0.0;
    }
    if (span.y <= span.x) return 0.0;
    if (kind != 2u && kind != 5u) return max(span.y-span.x-water_dry_distance(body,origin,direction,span),0.0);
    uint base = body.coverage.y, count = body.coverage.z, size = uint(WATER_COVERAGE_DATA.length());
    if (base > size || count > size-base) return 0.0;
    float cursor = span.x, wet = 0.0;
    // Every polygon edge contributes at most one crossing. BVH record count
    // is a conservative bound, not a cap on authored shoreline complexity.
    for (uint event = 0u; event <= count && cursor < span.y; ++event) {
        float next = water_next_shore(body,origin.xz,direction.xz,vec2(cursor,span.y));
        vec4 column; float edge;
        if (water_column(body,origin.xz+direction.xz*(cursor+(next-cursor)*0.5),column,edge))
            wet += max(next-cursor-water_dry_distance(body,origin,direction,vec2(cursor,next)),0.0);
        cursor = next;
    }
    return wet;
}
