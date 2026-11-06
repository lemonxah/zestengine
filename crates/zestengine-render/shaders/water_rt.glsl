// Water transmission at a reflected ray hit. Uses the same optical medium as
// the raster water and the renderer's existing opaque hit-lighting functions.
// Only opaque geometry is queried here. Nested water segments and exact exit
// clipping are not yet integrated along this transmitted ray.
vec3 water_rt_receiver(vec3 origin, vec3 direction, float cone, out float distance_m) {
    rayQueryEXT rq;
    distance_m = max(u.quality.z, 1.0);
    rayQueryInitializeEXT(rq, tlas, gl_RayFlagsNoneEXT, RT_MASK_SOLID, origin, 0.005, direction, distance_m);
    while (rayQueryProceedEXT(rq)) {
        if (rayQueryGetIntersectionTypeEXT(rq, false) == gl_RayQueryCandidateIntersectionTriangleEXT && candidate_covered(rq))
            rayQueryConfirmIntersectionEXT(rq);
    }
    if (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionNoneEXT)
        return sample_env(origin, direction, 0.0);
    distance_m = rayQueryGetIntersectionTEXT(rq, true);
    Instance inst = instances.data[rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true)];
    int prim = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
    Indices idx = Indices(inst.idx_addr);
    Vertices vtx = Vertices(inst.vtx_addr);
    Vertex a = vtx.v[idx.i[prim * 3]], b = vtx.v[idx.i[prim * 3 + 1]], c = vtx.v[idx.i[prim * 3 + 2]];
    vec2 bc = rayQueryGetIntersectionBarycentricsEXT(rq, true);
    float w = 1.0 - bc.x - bc.y;
    mat3 m = mat3(rayQueryGetIntersectionObjectToWorldEXT(rq, true));
    vec3 n = normalize(transpose(mat3(rayQueryGetIntersectionWorldToObjectEXT(rq, true))) * (a.normal*w + b.normal*bc.x + c.normal*bc.y));
    if (dot(n, direction) > 0.0) n = -n;
    vec3 surface_n = n;
    vec2 uv = a.uv*w + b.uv*bc.x + c.uv*bc.y;
    vec2 uv1 = a.uv1*w + b.uv1*bc.x + c.uv1*bc.y;
    vec4 ot = a.tangent*w + b.tangent*bc.x + c.tangent*bc.y;
    float area = length(cross(m*(b.position-a.position), m*(c.position-a.position)));
    vec2 e = b.uv-a.uv, f = c.uv-a.uv;
    float density = sqrt(abs(e.x*f.y-e.y*f.x) / max(area, 1e-12));
    RtSurfaceMaterial material = rt_surface_material(inst, uv, a.color.rgb*w + b.color.rgb*bc.x + c.color.rgb*bc.y,
        n, vec4(m*ot.xyz, ot.w*sign(determinant(m))), cone, density);
    vec3 p = origin + direction * distance_m;
    bool baked = inst.params.w >= 0.0;
    bool cached = false;
    bool hit = u.lighting_mode.x >= 2u;
    vec3 result = vec3(0.0);
    if (!baked) {
        SurfaceSample surface = surface_lookup_instance(p, surface_n, 0.01, uint(inst.emission_raw.w));
        result = hit ? surface_hit_indirect(surface, material.albedo*(1.0-material.metallic), cached) : surface_radiance(surface, cached);
    }
    if (!cached) {
        vec3 indirect = baked ? textureLod(u_lightmap, vec3(uv1, inst.params.w), 0.0).rgb / PI : sample_env(p, material.normal, 1.0)*0.3;
        result = material.albedo * (1.0-material.metallic) * indirect;
    }
    if (!cached || hit) {
        result += hit ? direct_lighting(p, material.normal, baked, true, -direction, material.albedo, material.metallic, material.roughness)
            : material.albedo*(1.0-material.metallic)/PI*direct_light(p, material.normal, baked);
    }
    return result + material.emission;
}

vec3 water_rt_transmission(int slot, vec3 p, vec3 incoming, vec3 normal, bool back, float cone, out float reflectance) {
    GpuWater body = u.water_bodies[slot];
    vec4 absorption = water_absorption(body), scattering = water_scattering_coeff(body);
    vec3 transmitted = refract(incoming, normal, back ? absorption.w : 1.0 / absorption.w);
    reflectance = water_surface_reflectance(dot(-incoming, normal), back ? absorption.w : 1.0 / absorption.w);
    if (dot(transmitted, transmitted) < 0.5) { reflectance = 1.0; return vec3(0.0); }
    transmitted = normalize(mix(incoming, transmitted, clamp(body.look1.y, 0.0, 1.0)));
    float path;
    vec3 receiver = water_rt_receiver(p + transmitted*0.005, transmitted, cone, path);
    // The transmitted segment is air when looking out from below.
    if (back) return receiver * (1.0-reflectance);
    vec3 transmittance = water_transmittance(absorption.rgb, scattering.rgb, path);
    vec3 integral = water_scatter_integral(absorption.rgb, scattering.rgb, transmittance);
    vec3 up = normalize(transpose(mat3(body.to_local))*vec3(0.0, 1.0, 0.0));
    vec3 incident = sample_env(p, up, 1.0)*0.3;
    for (uint i = 0u; i < uint(u.params.z); ++i) {
        Light light = lights.data[i];
        uint kind = uint(light.position.w);
        if (kind != 0u && kind != 1u && kind != 2u) continue;
        vec3 delta = light.position.xyz-p;
        float distance_m = length(delta);
        vec3 L = kind == 0u ? normalize(-light.direction.xyz) : delta/max(distance_m, 1e-5);
        float attenuation = kind == 0u ? 1.0 : realtime_distance_falloff(distance_m, light.direction.w);
        if (kind == 2u) attenuation *= realtime_spot_falloff(dot(normalize(light.direction.xyz), -L), light.spot.x, light.spot.y);
        if (attenuation <= 0.0 || occluded(p+up*0.005, L, kind == 0u ? 1e16 : distance_m)) continue;
        float light_path = path*0.5*abs(dot(-transmitted, up))/max(abs(dot(up,L)),0.1);
        if (kind != 0u) light_path = min(light_path,distance_m);
        incident += light.color.rgb*attenuation*water_transmittance(absorption.rgb,scattering.rgb,light_path)*water_phase(dot(L,transmitted),scattering.w);
    }
    return (receiver*transmittance*clamp(body.deep.w,0.0,1.0)+integral*incident)*(1.0-reflectance);
}
