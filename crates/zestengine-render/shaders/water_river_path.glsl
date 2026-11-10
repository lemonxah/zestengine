// Ray events through the sampled river used by water_column. Between events,
// projected segment parameters, width membership and vertical membership are
// fixed. Overlapping segments are split again where nearest ownership changes.
// There is no world-space step size that can skip a narrow channel.
struct WaterRiverRay {
    vec3 distance2; // t^2, t, constant
    vec2 radius;   // constant, t
    vec2 top;
    vec2 depth;
};
float water_ray_eval(vec3 p, float t) { return (p.x*t+p.y)*t+p.z; }
void water_ray_root(float t, vec2 range, inout float next) {
    if (t > range.x && t < next) next = t;
}
void water_ray_roots(vec3 p, vec2 range, inout float next) {
    if (abs(p.x) < 1e-12) {
        if (abs(p.y) > 1e-12) water_ray_root(-p.z/p.y,range,next);
        return;
    }
    float discriminant = p.y*p.y-4.0*p.x*p.z;
    if (discriminant < 0.0) return;
    float root = sqrt(discriminant);
    float q = -0.5*(p.y+(p.y < 0.0 ? -root : root));
    if (q == 0.0) water_ray_root(-p.y/(2.0*p.x),range,next);
    else {
        water_ray_root(q/p.x,range,next);
        water_ray_root(p.z/q,range,next);
    }
}
bool water_river_leaf(GpuWater body, vec3 origin, vec3 direction, vec2 range, inout uint cursor, out uint leaf) {
    uint base=body.coverage.y, count=body.coverage.z;
    while (cursor+1u<count) {
        uint i=cursor;
        vec4 box=WATER_COVERAGE_DATA[base+i];
        uvec4 meta=floatBitsToUint(WATER_COVERAGE_DATA[base+i+1u]);
        if (meta.x<=i || meta.x>count) return false;
        vec2 hit=range;
        bool candidate=water_path_slab(origin.x,direction.x,box.x,box.z,hit)
            && water_path_slab(origin.z,direction.z,box.y,box.w,hit);
        if (!candidate) {cursor=meta.x;continue;}
        if (meta.y==0u) {cursor+=2u;continue;}
        cursor=meta.x;
        if (i+7u<count) {leaf=base+i;return true;}
    }
    return false;
}
vec2 water_river_projection(uint leaf, vec3 origin, vec3 direction) {
    vec4 a=WATER_COVERAGE_DATA[leaf+2u], b=WATER_COVERAGE_DATA[leaf+5u];
    vec2 ab=b.xz-a.xz;
    float len2=dot(ab,ab);
    return len2>1e-8 ? vec2(dot(origin.xz-a.xz,ab),dot(direction.xz,ab))/len2 : vec2(0.0);
}
WaterRiverRay water_river_polynomial(uint leaf, vec3 origin, vec3 direction, float at) {
    vec4 a=WATER_COVERAGE_DATA[leaf+2u], b=WATER_COVERAGE_DATA[leaf+5u];
    float ad=WATER_COVERAGE_DATA[leaf+4u].x, bd=WATER_COVERAGE_DATA[leaf+7u].x;
    vec2 u=water_river_projection(leaf,origin,direction);
    float along=u.x+u.y*at;
    if (along<=0.0) u=vec2(0.0);
    else if (along>=1.0) u=vec2(1.0,0.0);
    vec2 p=origin.xz-a.xz-(b.xz-a.xz)*u.x, d=direction.xz-(b.xz-a.xz)*u.y;
    WaterRiverRay ray;
    ray.distance2=vec3(dot(d,d),2.0*dot(p,d),dot(p,p));
    ray.radius=(vec2(a.w,0.0)+(b.w-a.w)*u)*0.5;
    ray.top=vec2(a.y,0.0)+(b.y-a.y)*u;
    ray.depth=vec2(ad,0.0)+(bd-ad)*u;
    return ray;
}
float water_river_next(GpuWater body, vec3 origin, vec3 direction, vec2 range) {
    uint base=body.coverage.y, count=body.coverage.z, size=uint(WATER_COVERAGE_DATA.length());
    if (base>size || count>size-base || count==0u) return range.y;
    float cursor=range.x;
        float next=range.y;
        uint scan=0u, leaf;
        // First partition at clamped segment projection changes.
        while (water_river_leaf(body,origin,direction,vec2(cursor,next),scan,leaf)) {
            vec2 u=water_river_projection(leaf,origin,direction);
            if (abs(u.y)>1e-12) {
                water_ray_root(-u.x/u.y,vec2(cursor,next),next);
                water_ray_root((1.0-u.x)/u.y,vec2(cursor,next),next);
            }
        }
        float at=cursor+(next-cursor)*0.5;
        scan=0u;
        // Then banks, changing floor depth and the sloping rest surface.
        while (water_river_leaf(body,origin,direction,vec2(cursor,next),scan,leaf)) {
            WaterRiverRay ray=water_river_polynomial(leaf,origin,direction,at);
            vec3 bank=ray.distance2-vec3(ray.radius.y*ray.radius.y,2.0*ray.radius.x*ray.radius.y,ray.radius.x*ray.radius.x);
            water_ray_roots(bank,vec2(cursor,next),next);
            water_ray_roots(vec3(0.0,direction.y-ray.top.y,origin.y-ray.top.x),vec2(cursor,next),next);
            water_ray_roots(vec3(0.0,direction.y-ray.top.y+ray.depth.y,origin.y-ray.top.x+ray.depth.x),vec2(cursor,next),next);
            water_ray_roots(vec3(0.0,ray.depth.y,ray.depth.x),vec2(cursor,next),next);
        }
        // In a self-crossing river, only the nearest covered channel owns
        // the column. Narrow the interval until its winner cannot change.
        while (next>cursor) {
            at=cursor+(next-cursor)*0.5;
            float rank=1e30; uint owner=0xffffffffu;
            vec3 owner_distance=vec3(0.0);
            scan=0u;
            while (water_river_leaf(body,origin,direction,vec2(cursor,next),scan,leaf)) {
                WaterRiverRay ray=water_river_polynomial(leaf,origin,direction,at);
                float distance_m=sqrt(max(water_ray_eval(ray.distance2,at),0.0));
                float radius=ray.radius.x+ray.radius.y*at;
                uint index=floatBitsToUint(WATER_COVERAGE_DATA[leaf+1u].z);
                float candidate=floor(distance_m*65536.0+0.5);
                if (distance_m<=radius && (candidate<rank || (candidate==rank && index<owner))) {
                    rank=candidate;owner=index;owner_distance=ray.distance2;
                }
            }
            if (owner==0xffffffffu) break;
            float split=next;
            scan=0u;
            while (water_river_leaf(body,origin,direction,vec2(cursor,next),scan,leaf)) {
                WaterRiverRay ray=water_river_polynomial(leaf,origin,direction,at);
                float radius=ray.radius.x+ray.radius.y*at;
                if (water_ray_eval(ray.distance2,at)<=radius*radius)
                    water_ray_roots(ray.distance2-owner_distance,vec2(cursor,split),split);
            }
            if (split>=next) break;
            next=split;
        }
    return next;
}
float water_river_distance(GpuWater body, vec3 origin, vec3 direction, vec2 range) {
    float cursor=range.x,wet=0.0;
    while(cursor<range.y){
        float next=water_river_next(body,origin,direction,vec2(cursor,range.y));
        if (next<=cursor) break;
        vec3 p=origin+direction*(cursor+(next-cursor)*0.5);
        vec4 column;float edge;
        if (water_column(body,p.xz,column,edge) && p.y<=column.x && p.y>=column.x-column.y)
            wet+=max(next-cursor-water_dry_distance(body,origin,direction,vec2(cursor,next)),0.0);
        cursor=next;
    }
    return wet;
}
