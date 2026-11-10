#ifndef ZEST_WATER_COVERAGE
#define ZEST_WATER_COVERAGE
// Water-only spatial queries. The tree is packed from the same outline and
// river samples used by Water::column; no rectangular/average-height proxy.
// Define WATER_COVERAGE_DATA to the vec4 runtime storage array before inclusion.
// Column: surface height, floor depth, local XZ flow. Edge: 0 centre, 1 bank.
bool water_column(GpuWater body, vec2 xz, out vec4 column, out float edge, out float along, out vec2 gradient) {
    column = vec4(body.bounds.z, uintBitsToFloat(body.coverage.w), body.reflect.yz);
    edge = 0.0; along = 0.0; gradient=vec2(0.0);
    uint kind = body.coverage.x;
    if (kind <= 1u) {
        // Legacy standalone callers have not supplied an authored floor.
        if (kind == 0u) column.y = 1e20;
        return (body.bounds.x <= 0.0 || abs(xz.x) <= body.bounds.x)
            && (body.bounds.y <= 0.0 || abs(xz.y) <= body.bounds.y);
    }
    uint base = body.coverage.y;
    uint count = body.coverage.z;
    if (base > uint(WATER_COVERAGE_DATA.length()) || count > uint(WATER_COVERAGE_DATA.length()) - base) return false;
    bool covered = false;
    float best = 1e30;
    uint best_index = 0xffffffffu;
    uint i = 0u;
    while (i + 1u < count) {
        vec4 box = WATER_COVERAGE_DATA[base + i];
        uvec4 meta = floatBitsToUint(WATER_COVERAGE_DATA[base + i + 1u]);
        // Escape links are relative to the body's root, not the frame array.
        if (meta.x <= i || meta.x > count) return false;
        bool candidate = (kind == 2u || kind == 5u)
            ? xz.y >= box.y && xz.y < box.w && xz.x < box.z
            : all(greaterThanEqual(xz, box.xy)) && all(lessThanEqual(xz, box.zw));
        if (!candidate) { i = meta.x; continue; }
        if (meta.y == 0u) { i += 2u; continue; }
        if ((kind == 2u || kind == 5u) && i + 2u < count) {
            vec4 ab = WATER_COVERAGE_DATA[base + i + 2u];
            if ((ab.y > xz.y) != (ab.w > xz.y)
                && xz.x < (ab.z - ab.x) * (xz.y - ab.y) / (ab.w - ab.y) + ab.x) covered = !covered;
        } else if(kind==4u && i+4u<count){
            vec3 a=WATER_COVERAGE_DATA[base+i+2u].xyz,b=WATER_COVERAGE_DATA[base+i+3u].xyz,c=WATER_COVERAGE_DATA[base+i+4u].xyz;
            vec2 ab=b.xz-a.xz,ac=c.xz-a.xz,p=xz-a.xz;
            float det=ab.x*ac.y-ab.y*ac.x;
            if(abs(det)>1e-10){float u=(p.x*ac.y-p.y*ac.x)/det,v=(ab.x*p.y-ab.y*p.x)/det;
                float height=a.y+(b.y-a.y)*u+(c.y-a.y)*v;
                if(u>=-1e-6&&v>=-1e-6&&u+v<=1.000001&&(!covered||height>column.x)){covered=true;column.x=height;gradient=vec2((b.y-a.y)*ac.y-(c.y-a.y)*ab.y,(c.y-a.y)*ab.x-(b.y-a.y)*ac.x)/det;}
            }
        } else if (kind == 3u && i + 7u < count) {
            vec4 a = WATER_COVERAGE_DATA[base + i + 2u];
            vec4 af = WATER_COVERAGE_DATA[base + i + 3u];
            vec4 ad = WATER_COVERAGE_DATA[base + i + 4u];
            vec4 b = WATER_COVERAGE_DATA[base + i + 5u];
            vec4 bf = WATER_COVERAGE_DATA[base + i + 6u];
            vec4 bd = WATER_COVERAGE_DATA[base + i + 7u];
            precise vec2 segment = b.xz - a.xz;
            precise float len2 = segment.x * segment.x + segment.y * segment.y;
            precise vec2 relative = xz - a.xz;
            precise float projected = relative.x * segment.x + relative.y * segment.y;
            precise float t = len2 > 1e-8 ? clamp(projected / len2, 0.0, 1.0) : 0.0;
            precise vec2 delta = a.xz + segment * t - xz;
            precise float distance_m = sqrt(delta.x * delta.x + delta.y * delta.y);
            precise float half_width = max(a.w + (b.w - a.w) * t, 0.01) * 0.5;
            // Shared gameplay ranking: near-coincident channel runs retain
            // authoring order, independent of BVH traversal or arithmetic ULPs.
            precise float rank = floor(distance_m * 65536.0 + 0.5);
            if (distance_m <= half_width && (rank < best || (rank == best && meta.z < best_index))) {
                best = rank;
                best_index = meta.z;
                vec3 dir = mix(af.xyz, bf.xyz, t);
                dir = dot(dir, dir) > 1e-20 ? normalize(dir) : vec3(1.0, 0.0, 0.0);
                column = vec4(mix(a.y, b.y, t), max(mix(ad.x, bd.x, t), 0.0),
                    dir.xz * mix(af.w, bf.w, t) + body.reflect.yz + mix(ad.yz,bd.yz,t));
                edge = distance_m / half_width;
                along = mix(ad.w,bd.w,t);
                gradient=(t>0.0&&t<1.0&&len2>1e-8)?(b.y-a.y)*segment/len2:vec2(0.0);
                covered = true;
            }
        }
        i = meta.x;
    }
    return kind==5u ? !covered : covered;
}

bool water_column(GpuWater body,vec2 xz,out vec4 column,out float edge,out float along){vec2 gradient;return water_column(body,xz,column,edge,along,gradient);}
vec2 water_rest_gradient(GpuWater body,vec2 xz){
    if(body.coverage.x!=3u && body.coverage.x!=4u)return vec2(0.0);
    vec4 column;float edge,along;vec2 gradient;water_column(body,xz,column,edge,along,gradient);return gradient;
}
bool water_column(GpuWater body, vec2 xz, out vec4 column, out float edge) {
    float along;return water_column(body,xz,column,edge,along);
}
#endif
