// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Water-only flow detail and foam. Included after the shared surface inputs.
// Gradient noise with analytic derivatives: (value, d/dx, d/dy).
vec2 water_hash(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return fract(sin(p) * 43758.5453) * 2.0 - 1.0;
}

vec3 water_noised(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    vec2 du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
    vec2 ga = water_hash(i);
    vec2 gb = water_hash(i + vec2(1.0, 0.0));
    vec2 gc = water_hash(i + vec2(0.0, 1.0));
    vec2 gd = water_hash(i + vec2(1.0, 1.0));
    float va = dot(ga, f);
    float vb = dot(gb, f - vec2(1.0, 0.0));
    float vc = dot(gc, f - vec2(0.0, 1.0));
    float vd = dot(gd, f - vec2(1.0, 1.0));
    float k = va - vb - vc + vd;
    float v = va + u.x * (vb - va) + u.y * (vc - va) + u.x * u.y * k;
    vec2 d = ga + u.x * (gb - ga) + u.y * (gc - ga) + u.x * u.y * (ga - gb - gc + gd)
           + du * (u.yx * k + vec2(vb, vc) - va);
    return vec3(v, d);
}

// Three length scales with a shared, slowly warped domain. The derivative
// includes the warp Jacobian, so surface normals and caustic focusing agree.
// Frequencies fade independently; unresolved slopes become roughness instead
// of either shimmering or turning the distant water into a perfect mirror.
float water_detail_band(float footprint, float frequency) {
    return 1.0 - smoothstep(0.25, 0.85, footprint * frequency);
}
vec3 water_detail_field_quality(vec2 p, float t, float scale, float foot,float quality) {
    vec2 q = p * scale;
    vec3 broad = water_noised(q * 0.63 + vec2(-0.07, 0.045) * t);
    vec2 warp = vec2(0.42, 0.27);
    mat2 jacobian = mat2(1.0) + outerProduct(warp, broad.yz * 0.63);
    q += warp * broad.x;
    const mat2 chop = mat2(0.86, 0.51, -0.31, 0.52);
    const mat2 fine = mat2(0.64, -0.77, 0.46, 0.38);
    vec3 bands = vec3(water_detail_band(foot * scale, 0.63),
        quality>=2.0?water_detail_band(foot * scale, 3.7):0.0, quality>=3.0?water_detail_band(foot * scale, 12.5):0.0);
    vec3 middle=vec3(0.0),small=vec3(0.0);
    if(bands.y>0.0)middle=water_noised(chop*q*3.7+vec2(0.11,-0.08)*t);
    if(bands.z>0.0)small=water_noised(fine*q*12.5+vec2(-0.19,-0.13)*t);
    vec3 weights = vec3(0.48, 0.36, 0.24) * bands;
    float h = broad.x * weights.x / 0.63 + middle.x * weights.y / 3.7 + small.x * weights.z / 12.5;
    vec2 g = broad.yz * weights.x + transpose(jacobian) *
        (transpose(chop) * middle.yz * weights.y + transpose(fine) * small.yz * weights.z);
    return vec3(h, g);
}
vec3 water_detail_field(vec2 p,float t,float scale,float foot){return water_detail_field_quality(p,t,scale,foot,3.0);}
vec2 water_ripple_slope(vec2 p, float t, float scale, float foot) {
    return water_detail_field(p, t, scale, foot).yz;
}

// The same advected detail feeds the water normal and caustics. A zero detail
// strength remains exactly flat, including old foam-free indoor pools.
vec3 water_ripples_at_quality(vec3 N, vec2 p, float t, float strength, float scale, vec2 flow, float foot,float quality, out float variance) {
    if(strength<=0.0){variance=0.0;return N;}
    const float PERIOD = 4.0;
    float ph0 = fract(t / PERIOD);
    float ph1 = fract(t / PERIOD + 0.5);
    float w0 = 1.0 - abs(1.0 - 2.0 * ph0);
    vec2 g0 = water_detail_field_quality(p - flow * (ph0 * PERIOD), t, scale, foot,quality).yz;
    vec2 g1 = water_detail_field_quality(p - flow * (ph1 * PERIOD) + vec2(0.37, 0.71), t, scale, foot,quality).yz;
    vec2 g = mix(g1, g0, w0);
    float gain = max(strength, 0.0) * 0.55;
    vec3 bands = vec3(water_detail_band(foot * scale, 0.63),
        water_detail_band(foot * scale, 3.7), water_detail_band(foot * scale, 12.5));
    if(quality<3.0)bands.z=0.0;if(quality<2.0)bands.y=0.0;
    variance = gain * gain * dot(vec3(0.055, 0.022, 0.01), vec3(1.0) - bands * bands);
    vec3 slope = vec3(g.x, 0.0, g.y);
    // Perturb in the normal's tangent plane: do not flatten a wave crest.
    return normalize(N - (slope - N * dot(slope, N)) * gain);
}

vec3 water_ripples_at(vec3 N,vec2 p,float t,float strength,float scale,vec2 flow,float foot,out float variance){
    return water_ripples_at_quality(N,p,t,strength,scale,flow,foot,3.0,variance);
}
#ifndef WATER_COMPUTE
vec3 water_ripples(vec3 N, vec2 p, float t, float strength, float scale, vec2 flow, out float variance) {
    return water_ripples_at(N,p,t,strength,scale,flow,max(length(dFdx(p)),length(dFdy(p))),variance);
}

// Convert water-column depth into distance along the surface. A foam width in
// metres should stay narrow on a beach and at a steep pool wall alike.
float water_contact_distance(float depth, vec2 rest) {
    vec2 dx = dFdx(rest), dy = dFdy(rest);
    float det = dx.x * dy.y - dx.y * dy.x;
    vec2 gradient = abs(det) > 1e-10 ? vec2(
        dFdx(depth) * dy.y - dFdy(depth) * dx.y,
        dFdy(depth) * dx.x - dFdx(depth) * dy.x) / det : vec2(1.0);
    return depth / max(length(gradient), 0.15);
}

// Project the difference between a flat and rippled reflected ray. Fine normal
// detail now bends planar reflections even when the mesh has no big waves.
vec2 water_planar_uv(vec2 uv, vec3 normal, vec3 view, vec3 up) {
    vec3 flat_ray = reflect(-view, up);
    vec3 bent_ray = reflect(-view, normal);
    float reach = clamp(length(CAMERA_POS.xyz - v_world_pos) * 0.35, 0.5, 4.0);
    vec4 flat_clip = VIEW_PROJ * vec4(v_world_pos + flat_ray * reach, 1.0);
    vec4 bent_clip = VIEW_PROJ * vec4(v_world_pos + bent_ray * reach, 1.0);
    vec2 delta = vec2(0.0);
    if (flat_clip.w > 0.05 && bent_clip.w > 0.05) {
        delta = (bent_clip.xy / bent_clip.w - flat_clip.xy / flat_clip.w) * vec2(0.5, -0.5);
        delta /= max(1.0, length(delta) / 0.045);
    }
    // Keep the reflected scene aligned at screen edges; no stretched border.
    float edge = min(min(uv.x, uv.y), min(1.0-uv.x, 1.0-uv.y));
    return clamp(uv + delta * smoothstep(0.0, 0.06, edge), 0.001, 0.999);
}

#endif

// Foam cells: the edges of a jittered cell grid (second-nearest minus nearest
// feature distance), so foam reads as a lacy network of bubbles rather than
// smooth noise. 1 on the bubble walls.
float water_bubbles(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float f1 = 8.0;
    float f2 = 8.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec2 g = vec2(x, y);
            vec2 o = water_hash(i + g) * 0.5 + 0.5;
            float d = length(g + o - f);
            if (d < f1) {
                f2 = f1;
                f1 = d;
            } else if (d < f2) {
                f2 = d;
            }
        }
    }
    return 1.0 - smoothstep(0.0, 0.18, f2 - f1);
}

// Foam texture: two bubble scales carried along `flow` (m/s) by a flow map's
// two cross-fading phases; `density` fills the cells in so thick foam reads
// solid and thin foam stays lacy.
float water_foam_pattern(vec2 p, float t, vec2 flow, float density, float foot,float quality) {
    const float PERIOD = 4.0;
    float ph0 = fract(t / PERIOD);
    float ph1 = fract(t / PERIOD + 0.5);
    float w0 = 1.0 - abs(1.0 - 2.0 * ph0);
    vec2 a = p - flow * (ph0 * PERIOD);
    vec2 b = p - flow * (ph1 * PERIOD) + vec2(3.7, 1.3);
    // Large irregular rafts, smaller holes, then bubble walls. Avoid an even
    // white outline or the old oversized, uniformly spaced cell pattern.
    float patch0 = water_noised(a * 1.7 + vec2(t * 0.025, 0.0)).x;
    float patch1 = water_noised(b * 1.7 + vec2(t * 0.025, 0.0)).x;
    float raft = mix(patch1, patch0, w0);
    float coverage = smoothstep(0.22 - density * 0.65, 0.34 - density * 0.45, raft);
    float fine = water_detail_band(foot, 18.0);
    float f0=0.45,f1=0.45;
    if(fine>0.0 && quality>=3.0){f0=mix(0.45,water_bubbles(a*18.0),fine);f1=mix(0.45,water_bubbles(b*18.0),fine);}
    float cells = mix(f1, f0, w0);
    return coverage * mix(cells, 1.0, density * density * 0.75);
}

// One foam response for the main surface and reflected ray hits. Contacts
// come from scene depth in raster and opaque ray queries in reflections.
float water_foam_coverage(GpuWater body, vec2 rest, vec2 world_xz, float wt, vec2 flow,
    float surface_h, float crest_n, float wcalm, vec4 rip, float contact_depth, float shore_distance, float foam_foot) {
    vec4 look0=body.look0, look1=body.look1;
        float height = clamp(surface_h / max(look1.w, 1e-4), -1.0, 1.0);
        float crest = smoothstep(0.35, 0.95, crest_n * wcalm) * smoothstep(0.1, 0.7, height) * look1.x;
        // Contact foam in two layers: dense against whatever the water
        // touches, patchy lace further out, the band lapping in and out.
        // Caller supplies a footprint-aware contact distance.
        float lap = 0.8 + 0.2 * sin(wt * 1.3 + dot(world_xz, vec2(0.7, 0.4)));
        float fw = max(look0.w, 1e-3) * lap;
        float dense = 1.0 - smoothstep(0.0, fw * 0.22, shore_distance);
        float lace = 1.0 - smoothstep(fw * 0.22, fw, shore_distance);
        // Zero width explicitly disables shore foam without affecting wakes.
        if (look0.w <= 0.0) { dense = 0.0; lace = 0.0; }
        // Whitewater where the simulated water runs fast for its depth
        // (Froude number u / sqrt(g h) past ~0.8: plunge pools, hard wakes),
        // not ordinary sloshing in shallow water.
        float wave_speed = sqrt(9.81 * clamp(contact_depth, 0.1, 5.0));
        float white = smoothstep(0.8, 1.6, rip.a / wave_speed);
        // Churned water (waterfall plunges, whirlpools): thick in the middle,
        // breaking into lace toward the edge, carried outward.
        float churn = 0.0;
        vec2 churn_flow = vec2(0.0);
        for (int k = 0; k < MAX_WATER_FOAM; ++k) {
            vec4 spot = body.foam[k];
            if (spot.z <= 0.0) {
                continue;
            }
            vec2 d = rest - spot.xy;
            float r = length(d) / spot.z;
            float c = spot.w * (1.0 - smoothstep(0.25, 1.0, r));
            if (c > churn) {
                churn = c;
                vec2 out_dir = r > 1e-3 ? d / (r * spot.z) : vec2(0.0);
                churn_flow = (inverse(mat3(body.to_local)) * vec3(out_dir.x, 0.0, out_dir.y)).xz * (0.5 + 0.5 * r);
            }
        }
        churn = clamp(churn, 0.0, 1.0);
        flow += churn_flow * churn;
        float density = max(max(dense, churn * 0.9), max(white * 0.8, crest * 0.75));
        float cover = max(max(max(dense, lace * 0.8), max(crest, white)), churn);
        // Avoid the costly cellular pattern on open water and foam-free pools.
        // Caller supplies the pixel or ray-cone footprint.
        float bubbles = 0.0;
        if (cover > 0.001 && body.ripple.w > 0.0) {
            bubbles = water_foam_pattern(world_xz, wt, flow, max(density, cover * 0.65), foam_foot,water_quality(body.quality.y));
        }
        float water_foam = clamp(cover * bubbles * 1.6, 0.0, 1.0);
        // The body's Foam amount (ripple.w): 0 turns it all off.
        water_foam *= clamp(body.ripple.w, 0.0, 2.0);
        water_foam = min(water_foam, 1.0);
    return water_foam;
}
