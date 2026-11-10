// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Water-only focusing, consuming the shared scene-light inputs.
#define WATER_COVERAGE_DATA water_scene.coverage
#include "water_coverage.glsl"
#include "water_exclusions.glsl"
#include "water_ownership.glsl"
#include "water_path.glsl"
#include "water_media.glsl"
#undef WATER_COVERAGE_DATA
// Analytic wave derivatives once per contributing body. Avoid recursively
// reevaluating complete connection chains for each finite-difference sample.
void water_wave_derivatives(int slot,vec2 xz,float t,float foot,bool want_hess,out vec2 g,out vec3 h){
    g=vec2(0.0);h=vec3(0.0);
    int n = min(int(water_scene.bodies[slot].look1.z + 0.5), MAX_WATER_WAVES);
    for (int i = 0; i < n; ++i) {
        vec4 a = water_scene.bodies[slot].wave_a[i];
        float amp = water_scene.bodies[slot].wave_b[i].y * water_band(a.z, foot);
        float phase = a.z * dot(a.xy, xz) - a.w * t;
        g += (amp * a.z * cos(phase)) * a.xy;
        h -= (amp * a.z * a.z * sin(phase)) * vec3(a.x * a.x, a.y * a.y, a.x * a.y);
    }
    if(water_scene.bodies[slot].ocean.y>0u || water_scene.bodies[slot].recording.y>0u) {
        g+=water_ocean_field(water_scene.bodies[slot],xz,foot,1u).xy+water_recorded_field(water_scene.bodies[slot],xz,foot,1u).xy;
        if(want_hess) {
            float ocean_cell=water_scene.bodies[slot].ocean.y>0u ? uintBitsToFloat(water_scene.bodies[slot].ocean.z)/float(water_scene.bodies[slot].ocean.y) : 1e10;
            float cached_cell=water_scene.bodies[slot].recording.y>0u ? water_scene.bodies[slot].recording_window.z : 1e10;
            float e=max(foot,min(ocean_cell,cached_cell));
            vec2 px=water_ocean_field(water_scene.bodies[slot],xz+vec2(e,0.0),foot,1u).xy+water_recorded_field(water_scene.bodies[slot],xz+vec2(e,0.0),foot,1u).xy;
            vec2 nx=water_ocean_field(water_scene.bodies[slot],xz-vec2(e,0.0),foot,1u).xy+water_recorded_field(water_scene.bodies[slot],xz-vec2(e,0.0),foot,1u).xy;
            vec2 pz=water_ocean_field(water_scene.bodies[slot],xz+vec2(0.0,e),foot,1u).xy+water_recorded_field(water_scene.bodies[slot],xz+vec2(0.0,e),foot,1u).xy;
            vec2 nz=water_ocean_field(water_scene.bodies[slot],xz-vec2(0.0,e),foot,1u).xy+water_recorded_field(water_scene.bodies[slot],xz-vec2(0.0,e),foot,1u).xy;
            h+=vec3(px.x-nx.x,pz.y-nz.y,0.5*(px.y-nx.y+pz.x-nz.x))/(2.0*e);
        }
    }

}
// Slope (local XZ gradient) and, when `want_hess`, Hessian (xx, zz, xz) of
// body `s`'s surface over its local point `xz`, band-limited to `foot`
// metres: the Gerstner sum, the simulated field and the fine scrolling
// ripples (world `wxz`), the same surface the water shader draws.
void water_surface_slope(int s, vec2 xz, float foot, bool want_hess, out vec2 g, out vec3 h) {
    g = vec2(0.0);
    h = vec3(0.0);
    vec4 column; float edge;
    if (!water_column(water_scene.bodies[s], xz, column, edge)) return;
    mat4 to_world = inverse(water_scene.bodies[s].to_local);
    vec2 wxz = (to_world * vec4(xz.x, column.x, xz.y, 1.0)).xz;
    float t = frame.water.x;
    water_wave_derivatives(s,xz,t,foot,want_hess,g,h);
    vec2 previous_g=g;vec3 previous_h=h;
    float weight;vec3 point;int target=water_connection(water_scene.bodies[s],xz,weight,point);float remaining=1.0;
    for(int step=0;step<MAX_WATER_BODIES && target>=0;++step){
        vec2 tg;vec3 th;water_wave_derivatives(target,point.xz,t,foot,want_hess,tg,th);
        mat4 to_target=water_scene.bodies[target].to_local*to_world;
        mat2 jacobian=mat2(to_target[0].xz,to_target[2].xz);
        float vertical=(inverse(to_target)*vec4(0.0,1.0,0.0,0.0)).y;
        tg=transpose(jacobian)*tg*vertical;
        mat2 hessian=transpose(jacobian)*mat2(th.x,th.z,th.z,th.y)*jacobian*vertical;th=vec3(hessian[0][0],hessian[1][1],hessian[0][1]);
        remaining*=weight;g+=remaining*(tg-previous_g);h+=remaining*(th-previous_h);previous_g=tg;previous_h=th;
        target=water_connection(water_scene.bodies[target],point.xz,weight,point);
    }
    float shore=water_connected_shore(water_scene.bodies[s],xz,water_shore_field(water_scene.bodies[s],xz));
    float bank=water_connected_edge(water_scene.bodies[s],xz,edge);float calm=1.0-bank*bank;
    g*=shore*calm;h*=shore*calm;
    vec4 r = water_scene.bodies[s].ripple;
    vec2 uv = water_ripple_uv(s, xz);
    float fade = water_ripple_fade(uv)*calm;
    if (fade > 0.0) {
        float size = float(textureSize(u_water_ripples[s], 0).x);
        float lod = water_ripple_lod(foot, r.z, size);
        g += textureLod(u_water_ripples[s], uv, lod).gb * fade;
        if (want_hess) {
            // Central differences of the gradient a texel of that mip apart.
            float e = exp2(floor(lod)) / size;
            vec2 px = textureLod(u_water_ripples[s], uv + vec2(e, 0.0), lod).gb;
            vec2 nx = textureLod(u_water_ripples[s], uv - vec2(e, 0.0), lod).gb;
            vec2 pz = textureLod(u_water_ripples[s], uv + vec2(0.0, e), lod).gb;
            vec2 nz = textureLod(u_water_ripples[s], uv - vec2(0.0, e), lod).gb;
            h += fade * vec3(px.x - nx.x, pz.y - nz.y, 0.5 * (px.y - nx.y + pz.x - nz.x)) / (2.0 * e * r.z);
        }
    }
    vec4 look0 = water_scene.bodies[s].look0;
    vec4 previous_look=look0;remaining=1.0;target=water_connection(water_scene.bodies[s],xz,weight,point);
    for(int step=0;step<MAX_WATER_BODIES && target>=0;++step){
        vec4 next_look=water_scene.bodies[target].look0;remaining*=weight;look0+=remaining*(next_look-previous_look);previous_look=next_look;
        target=water_connection(water_scene.bodies[target],point.xz,weight,point);
    }
    vec2 flow = (mat3(to_world) * vec3(column.z+water_recorded_field(water_scene.bodies[s],xz,foot,2u).x, 0.0, column.w+water_recorded_field(water_scene.bodies[s],xz,foot,2u).y)).xz;
    float scale = max(look0.z, 0.01);
    float k = look0.y * 0.55;
    if (k > 0.0 && water_quality(water_scene.bodies[s].quality.w)>1.0) {
        // The same flow-carrying ripple mix the surface itself draws
        // (water_ripples): two half-period phases cross-fading, advected by
        // the current, so the caustics drift with it instead of flickering
        // in place.
        const float PERIOD = 4.0;
        float ph0 = fract(t / PERIOD);
        float ph1 = fract(t / PERIOD + 0.5);
        float w0 = 1.0 - abs(1.0 - 2.0 * ph0);
        vec2 g0 = water_detail_field_quality(wxz - flow * (ph0 * PERIOD), t, scale, foot,water_quality(water_scene.bodies[s].quality.w)).yz;
        vec2 g1 = water_detail_field_quality(wxz - flow * (ph1 * PERIOD) + vec2(0.37, 0.71), t, scale, foot,water_quality(water_scene.bodies[s].quality.w)).yz;
        mat2 jacobian = mat2(to_world[0].xz, to_world[2].xz);
        g += transpose(jacobian) * mix(g1, g0, w0) * k;
        if (want_hess) {
            float e = 0.05 / scale;
            vec2 gx = mix(water_detail_field_quality(wxz + vec2(e, 0.0) - flow * (ph1 * PERIOD) + vec2(0.37, 0.71), t, scale, foot,water_quality(water_scene.bodies[s].quality.w)).yz, water_detail_field_quality(wxz + vec2(e, 0.0) - flow * (ph0 * PERIOD), t, scale, foot,water_quality(water_scene.bodies[s].quality.w)).yz, w0);
            vec2 gz = mix(water_detail_field_quality(wxz + vec2(0.0, e) - flow * (ph1 * PERIOD) + vec2(0.37, 0.71), t, scale, foot,water_quality(water_scene.bodies[s].quality.w)).yz, water_detail_field_quality(wxz + vec2(0.0, e) - flow * (ph0 * PERIOD), t, scale, foot,water_quality(water_scene.bodies[s].quality.w)).yz, w0);
            vec2 gmid = mix(g1, g0, w0);
            vec3 wh = (k / e) * vec3(gx.x - gmid.x, gz.y - gmid.y, 0.5 * (gx.y - gmid.y + gz.x - gmid.x));
            mat2 lh = transpose(jacobian) * mat2(wh.x, wh.z, wh.z, wh.y) * jacobian;
            h += vec3(lh[0][0], lh[1][1], lh[0][1]);
        }
    }
}

// Caustic focusing through body `s`'s surface: light crossing it at local
// `entry` bends by the slope there and lands `kappa` x slope further on, so a
// receiver gets 1/det(I + kappa H) of it, read at the surface point whose
// light lands here. Averaged over the receiver that keeps its brightness.
float water_focus(int s, vec2 entry, float kappa, float foot) {
    vec2 g;
    vec3 h;
    water_surface_slope(s, entry, foot, false, g, h);
    vec2 src = entry - kappa * g;
    water_surface_slope(s, src, foot, true, g, h);
    float det = (1.0 + kappa * h.x) * (1.0 + kappa * h.y) - kappa * kappa * h.z * h.z;
    return min(inversesqrt(det * det + 0.01), 8.0);
}

// Light gain from caustics at world `p` for one light (1 = none). `travel`
// is a directional light's direction of travel; a point/spot light is at
// `light_pos` (`point`).
//   Under a surface, lit from above: light refracted into the water focuses
// where the surface curves, bent by (1 - 1/1.33) of the slope over the
// slanted depth. Caustic strength (bounds.w) scales the focusing.
//   Above a pool's surface (extra.x > 0), lit from a light under it: the
// light leaves the water bent the other way, by (1.33 - 1) of the slope over
// its path through the air, and throws the moving pattern on walls and
// ceiling; extra.x scales that light (1 physical), and the Fresnel loss at
// the surface takes the grazing rays.
float water_caustic_gain(vec3 p, vec3 travel, vec3 light_pos, bool point) {
    int n = min(int(frame.water.z + 0.5), MAX_WATER_BODIES);
    float pix = length(fwidth(p));
    for (int s = 0; s < n; ++s) {
        vec4 b = water_scene.bodies[s].bounds;
        float ior = max(water_absorption(water_scene.bodies[s]).w, 1.0);
        mat4 to_local = water_scene.bodies[s].to_local;
        vec3 l = (to_local * vec4(p, 1.0)).xyz;
        if (water_excluded(water_scene.bodies[s], l) || !water_owned(water_scene.bodies[s],(inverse(water_scene.bodies[s].to_local)*vec4(l,1.0)).xyz)) continue;
        vec3 lp = (to_local * vec4(light_pos, 1.0)).xyz;
        vec4 column; float edge;
        bool inside = water_column(water_scene.bodies[s], l.xz, column, edge);
        float depth = column.x - l.y;
        if (inside && depth > 0.005 && depth <= column.y) {
            if (b.w <= 0.0) continue;
            vec3 dir = point ? normalize(l - lp) : normalize(mat3(to_local) * travel);
            vec3 tr = refract(dir, vec3(0.0, 1.0, 0.0), 1.0 / ior);
            if (dir.y >= -0.001 || tr.y >= -0.001) continue;
            float slant = depth / -tr.y;
            vec2 entry = l.xz - tr.xz * slant;
            bool wet = true;
            // A sloped river's entry is not at the receiver's column height.
            // Re-evaluate along the refracted ray, then reject bank crossings.
            for (int k = 0; k < 4; ++k) {
                vec4 at_entry; float entry_edge;
                if (!water_column(water_scene.bodies[s], entry, at_entry, entry_edge)) { wet = false; break; }
                slant = (at_entry.x - l.y) / -tr.y;
                entry = l.xz - tr.xz * slant;
            }
            vec4 at_entry; float entry_edge;
            if (!wet || slant <= 0.0 || !water_column(water_scene.bodies[s], entry, at_entry, entry_edge)
                || (point && lp.y <= at_entry.x)) continue;
            if (water_excluded(water_scene.bodies[s], vec3(entry.x, at_entry.x, entry.y))) continue;
            float kappa = b.w * slant * (1.0 - 1.0 / ior);
            float foot = max(pix, 0.02 + 0.03 * depth);
            vec3 transmission = water_transmittance(water_absorption(water_scene.bodies[s]).rgb, water_scattering_coeff(water_scene.bodies[s]).rgb, slant);
            float contrast = dot(transmission, vec3(0.2126, 0.7152, 0.0722));
            return mix(1.0, water_focus(s, entry, kappa, foot), contrast);
        }
        float strength = water_scene.bodies[s].extra.x;
        vec4 light_column; float light_edge;
        if (!point || strength <= 0.0 || !water_column(water_scene.bodies[s], lp.xz, light_column, light_edge)) continue;
        if (water_excluded(water_scene.bodies[s], lp)) continue;
        float under = light_column.x - lp.y;
        depth = light_column.x - l.y;
        if (depth > 0.0 || under <= 0.005 || under > light_column.y) continue;
        // Where the bent ray leaves the water: 1.33 sin(water angle) = sin(air
        // angle) along the line from the light to here.
        float rise = -depth;
        vec2 across = l.xz - lp.xz;
        float reach = length(across);
        float lo = 0.0;
        float hi = reach;
        for (int k = 0; k < 10; ++k) {
            float r = 0.5 * (lo + hi);
            float sw = r / sqrt(r * r + under * under);
            float sa = (reach - r) / sqrt((reach - r) * (reach - r) + rise * rise);
            if (ior * sw > sa) {
                hi = r;
            } else {
                lo = r;
            }
        }
        float r = 0.5 * (lo + hi);
        vec2 entry = lp.xz + (reach > 1e-5 ? across / reach : vec2(0.0)) * r;
        vec4 exit_column; float exit_edge;
        if (!water_column(water_scene.bodies[s], entry, exit_column, exit_edge)) continue;
        // Projection currently assumes a locally flat exit; reject a river
        // height discontinuity rather than focusing through an unrelated plane.
        if (abs(exit_column.x - light_column.x) > 0.05) continue;
        if (water_excluded(water_scene.bodies[s], vec3(entry.x, exit_column.x, entry.y))) continue;
        float air = sqrt((reach - r) * (reach - r) + rise * rise);
        float cos_air = rise / max(air, 1e-4);
        float f0 = water_normal_reflectance(ior);
        float fresnel = f0 + (1.0 - f0) * pow(1.0 - clamp(cos_air, 0.0, 1.0), 5.0);
        // A point source's spreading beam dilutes the focusing far away.
        float kappa = (ior - 1.0) * air * (ior * under) / (ior * under + air);
        float foot = max(pix * 0.5, 0.01 + 0.01 * air);
        return strength * (1.0 - fresnel) * water_focus(s, entry, kappa, foot);
    }
    return 1.0;
}
