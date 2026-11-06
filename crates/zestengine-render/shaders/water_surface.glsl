// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Surface evaluation inside the fragment entry point. No scene-light policy here.
    // Water: wave normal plus ripples, a body coloured by depth absorption,
    // the opaque scene refracted through it, and shore + crest foam.
    vec3 water_behind = vec3(0.0);
    vec3 water_scattering = vec3(0.0);
    float water_f0 = 0.02;
    float water_path = 0.0;
    vec3 water_up = vec3(0.0, 1.0, 0.0);
    float water_alpha = 1.0;
    float water_foam = 0.0;
    if (FEAT_WATER) {
        int ws = clamp(int(fx.water.x + 0.5), 0, MAX_WATER_BODIES - 1);
        vec4 shallow = frame.water_bodies[ws].shallow;
        vec4 deep = frame.water_bodies[ws].deep;
        vec4 look0 = frame.water_bodies[ws].look0;
        vec4 look1 = frame.water_bodies[ws].look1;
        vec3 absorption = max(water_absorption(frame.water_bodies[ws]).rgb, vec3(0.0));
        vec3 scattering = max(water_scattering_coeff(frame.water_bodies[ws]).rgb, vec3(0.0));
        float ior = max(water_absorption(frame.water_bodies[ws]).w, 1.0);
        water_f0 = water_normal_reflectance(ior);
        float wt = frame.water.x;
        // Directions from the water's local space to world (normals by the
        // inverse transpose of local-to-world), and its rest "up".
        mat3 to_world_n = transpose(mat3(frame.water_bodies[ws].to_local));
        vec3 up = normalize(to_world_n * vec3(0.0, 1.0, 0.0));
        water_up = up;
        // The full surface per pixel at its rest point (v_uv): the mesh only
        // carries what its spacing can hold. Everything is band-limited to
        // the pixel's footprint, and what is too fine to show roughens the
        // highlight instead of sparkling.
        vec2 rest = v_uv;
        float foot = max(max(length(dFdx(rest)), length(dFdy(rest))), 1e-4);
        float calm = 1.0 - v_color.b * v_color.b;
        // The waves (not the simulated layer) also calm over the shallows
        // near a shore (the vertex shader's water_shore).
        float wcalm = calm * clamp(v_color.a, 0.0, 1.0);
        vec3 wd;
        vec3 wn;
        float crest_n;
        float lost;
        water_waves(ws, rest, wt, foot, wd, wn, crest_n, lost);
        wn = normalize(mix(vec3(0.0, 1.0, 0.0), wn, wcalm));
        // The simulated layer at the footprint's mip (r height, gb slope).
        vec2 ruv = water_ripple_uv(ws, rest);
        float rfade = water_ripple_fade(ruv) * calm;
        vec4 rip = texture(t_displacement, ruv) * rfade;
        vec3 nl = normalize(vec3(wn.x / max(wn.y, 0.05) - rip.g, 1.0, wn.z / max(wn.y, 0.05) - rip.b));
        N = normalize(to_world_n * nl);
        if (!gl_FrontFacing) {
            N = -N;
        }
        // World-space flow (m/s), decoded by the vertex shader.
        vec2 flow = v_color.rg;
        N = water_ripples(N, v_world_pos.xz, wt, look0.y, max(look0.z, 0.01), flow);
        float surface_h = wd.y * wcalm + rip.r;
        roughness = clamp(look0.x, 0.02, 1.0);
        metallic = 0.0;
        ao = 1.0;
        vec3 Vw = normalize(CAMERA_POS.xyz - v_world_pos);
        float clarity = clamp(deep.w, 0.0, 1.0);
        float shore = 0.0;
        float contact_depth = 1e4;
        water_path = max(shallow.w, 0.01);
        vec3 transmit = water_transmittance(absorption, scattering, water_path);
        water_scattering = water_scatter_integral(absorption, scattering, transmit);
        if (frame.water.y > 0.5) {
            vec3 cam = CAMERA_POS.xyz;
            vec3 I = -Vw;
            vec2 screen = vec2(max(frame.postfx2.w, 1.0), max(frame.postfx3.w, 1.0));
            float surf_dist = length((VIEW_MATRIX * vec4(v_world_pos, 1.0)).xyz);
            float flat_path = max(water_scene_dist(gl_FragCoord.xy) - surf_dist, 0.0);
            // Behind the surface: down from above, up from below.
            vec3 into = gl_FrontFacing ? -up : up;
            float thick = flat_path * max(dot(I, into), 0.0);
            // Refraction: the view ray bent by Snell's law through the
            // surface normal (look.refraction scales the bend, 1 physical),
            // followed to the bottom: the thickness here first, then the
            // depth re-read where the bent ray lands. A lookup that lands on
            // something out of the water, off screen, or a ray reflected
            // whole falls back to the straight view.
            vec2 rp = gl_FragCoord.xy;
            float path = flat_path;
            vec3 T = refract(I, N, gl_FrontFacing ? 1.0 / ior : ior);
            bool total_internal_reflection = dot(T, T) < 0.5;
            if (total_internal_reflection) water_f0 = 1.0;
            float bend = max(look1.y, 0.0);
            // Fade the bend near what the surface meets: the band around an
            // object shows it straight - no displaced copy beside it, and no
            // hard seam where the bent ray switches to the object (the bend
            // would offset the floor by up to depth * tan(bend) there).
            bend *= smoothstep(0.15, 0.6, thick);
            if (bend > 0.0 && dot(T, T) > 0.5 && thick > 1e-4) {
                T = normalize(mix(I, T, bend));
                float down = max(dot(T, into), 0.05);
                float len = thick / down;
                vec2 hit = rp;
                float keep = 0.0;
                for (int k = 0; k < 2; ++k) {
                    vec4 c = VIEW_PROJ * vec4(v_world_pos + T * len, 1.0);
                    if (c.w <= 1e-4) {
                        keep = 0.0;
                        break;
                    }
                    vec2 suv = vec2(c.x / c.w * 0.5 + 0.5, 0.5 - c.y / c.w * 0.5);
                    vec2 edge = min(suv, 1.0 - suv);
                    keep = smoothstep(0.0, 0.04, min(edge.x, edge.y));
                    if (keep <= 0.0) {
                        break;
                    }
                    hit = suv * screen;
                    float dq = water_scene_dist(hit);
                    // A bent ray landing on something inside the water column,
                    // closer than the straight path's bottom, is an object
                    // between the camera and the bottom (a submerged body's
                    // side): show it where it really is (the straight view)
                    // instead of as a displaced ghost copy beside it.
                    if (dq < surf_dist + flat_path - 0.5) {
                        keep = 0.0;
                        break;
                    }
                    vec3 q = v_world_pos + T * len;
                    vec3 behind_p = cam + normalize(q - cam) * min(dq, 1e4);
                    // Never refract anything above the water's rest surface:
                    // the part of a body sticking out of the water must not
                    // ghost on the surface beside it.
                    if (gl_FrontFacing) {
                        mat4 wl = frame.water_bodies[ws].to_local;
                        float receiver_y = (wl * vec4(behind_p, 1.0)).y;
                        float surface_y = (wl * vec4(v_world_pos, 1.0)).y;
                        if (receiver_y > surface_y) {
                            keep = 0.0;
                            break;
                        }
                    }
                    float below = dot(behind_p - v_world_pos, into);
                    if (below <= 0.0) {
                        keep = 0.0;
                        break;
                    }
                    len = below / down;
                }
                rp = mix(rp, hit, keep);
                path = mix(flat_path, len, keep);
            }
            path = min(path, 1e4);
            water_path = path;
            transmit = water_transmittance(absorption, scattering, path);
            water_scattering = water_scatter_integral(absorption, scattering, transmit);
            // Looking out from below: this segment is air. The underwater
            // pass integrates camera-to-surface water exactly once.
            if (!gl_FrontFacing) {
                transmit = vec3(1.0);
                water_scattering = vec3(0.0);
                clarity = 1.0;
                water_path = 0.0;
            }
            vec3 behind = textureLod(u_opaque_color, rp / vec2(textureSize(u_opaque_color, 0)), 0.0).rgb;
            water_behind = total_internal_reflection ? vec3(0.0) : behind * transmit * clarity;
            if (total_internal_reflection) water_scattering = vec3(0.0);
            // Soft where the surface meets something: over a few pixels or
            // the mesh's spacing, whichever is wider, so the waterline on a
            // body never shows the triangles it cuts.
            float contact = clamp(max(2.0 * foot, 0.1 * v_uv1.y), 0.03, 0.3);
            water_alpha = smoothstep(0.0, contact, thick);
            shore = 1.0 - smoothstep(0.0, max(look0.w, 1e-3), thick);
            contact_depth = thick;
        } else {
            water_alpha = 1.0 - clarity * 0.4;
        }
        // Endless bodies (bounds.xy == 0) fade out near the graded mesh's far
        // edge so the surface blends into the sky at the horizon instead of
        // ending in a hard displaced edge. Only at grazing view angles (the
        // horizon), never from above. WATER_HORIZON_EXTENT must match
        // ENDLESS_EXTENT in crates/zestengine-engine/src/water_render.rs.
        if (frame.water_bodies[ws].bounds.x <= 0.0 && frame.water_bodies[ws].bounds.y <= 0.0) {
            float dist = length(v_world_pos.xz - CAMERA_POS.xz);
            float graze = 1.0 - abs(dot(normalize(N), normalize(CAMERA_POS.xyz - v_world_pos)));
            water_alpha *= 1.0 - smoothstep(WATER_HORIZON_EXTENT * 0.55, WATER_HORIZON_EXTENT * 0.95, dist) * graze;
        }
        float height = clamp(surface_h / max(look1.w, 1e-4), -1.0, 1.0);
        float crest = smoothstep(0.35, 0.95, crest_n * wcalm) * smoothstep(0.1, 0.7, height) * look1.x;
        // Contact foam in two layers: dense against whatever the water
        // touches, patchy lace further out, the band lapping in and out.
        float lap = 0.85 + 0.15 * sin(wt * 1.3 + dot(v_world_pos.xz, vec2(0.7, 0.4)));
        float fw = max(look0.w, 1e-3) * lap;
        float dense = 1.0 - smoothstep(0.0, fw * 0.35, contact_depth);
        float lace = 1.0 - smoothstep(fw * 0.35, fw, contact_depth);
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
            vec4 spot = frame.water_bodies[ws].foam[k];
            if (spot.z <= 0.0) {
                continue;
            }
            vec2 d = rest - spot.xy;
            float r = length(d) / spot.z;
            float c = spot.w * (1.0 - smoothstep(0.25, 1.0, r));
            if (c > churn) {
                churn = c;
                vec2 out_dir = r > 1e-3 ? d / (r * spot.z) : vec2(0.0);
                churn_flow = (transpose(mat3(frame.water_bodies[ws].to_local)) * vec3(out_dir.x, 0.0, out_dir.y)).xz * (0.5 + 0.5 * r);
            }
        }
        churn = clamp(churn, 0.0, 1.0);
        flow += churn_flow * churn;
        float density = max(max(dense, churn * 0.9), max(white * 0.8, crest * 0.6));
        float cover = max(max(max(dense, lace * 0.7), max(crest, white)), churn);
        float bubbles = water_foam_pattern(v_world_pos.xz, wt, flow, density);
        water_foam = clamp(cover * smoothstep(0.25, 0.75, bubbles * (0.6 + 0.4 * cover) + cover * 0.25), 0.0, 1.0);
        // The body's Foam amount (ripple.w): 0 turns it all off.
        water_foam *= clamp(frame.water_bodies[ws].ripple.w, 0.0, 2.0);
        water_foam = min(water_foam, 1.0);
        // Water's body is a volume, not a diffuse painted surface. Only foam
        // contributes a surface diffuse lobe; the medium is integrated below.
        albedo = vec4(vec3(0.92) * water_foam, 1.0);
        water_behind *= 1.0 - water_foam;
        water_scattering *= 1.0 - water_foam;
        roughness = mix(roughness, 0.6, water_foam);
        // Specular anti-aliasing: the normal's spread over the pixel and the
        // slope of the waves too fine to draw widen the lobe (GGX alpha^2 +
        // 2 sigma^2), so small bright highlights don't flicker into specks.
        vec3 dndx = dFdx(N);
        vec3 dndy = dFdy(N);
        float spread = 0.25 * (dot(dndx, dndx) + dot(dndy, dndy));
        float a = roughness * roughness;
        float a2 = clamp(a * a + min(2.0 * spread, 0.18) + 2.0 * lost * wcalm, 0.0, 1.0);
        roughness = clamp(sqrt(sqrt(a2)), 0.02, 1.0);
    }

