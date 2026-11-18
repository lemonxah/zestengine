    for (int w = 0; w < 4; ++w) {
        uint bits = light_mask[w];
        while (bits != 0u) {
            int i = w * 32 + findLSB(bits);
            bits &= bits - 1u;
            if (i >= light_count || i >= MAX_LIGHTS) {
                bits = 0u;
                continue;
            }
            Light light = frame.lights[i];
            int kind = int(light.pos_kind.w + 0.5);
            if (kind == 3) {
                // The voxel grid's injection kernel per pixel: a probe grid
                // shows its cells under an orb's sharp falloff.
                vec3 d = light.pos_kind.xyz - v_world_pos;
                float dist = length(d);
                float f = clamp(1.0 - dist / max(light.dir_range.w, 1e-3), 0.0, 1.0);
                vec3 Le = dist > 1e-4 ? d / dist : N;
                voxel_emit += light.color.rgb * (f * f * max(0.11141 + 0.25465 * dot(N, Le), 0.0));
                continue;
            }
            if (kind >= 4) {
                // Area light (rectangle 4, disc 5): two slots; 6 is the continuation. Shadow-less,
                // applies to lightmapped and dynamic surfaces alike (a bake has no area lights).
                if (kind <= 5 && i + 1 < MAX_LIGHTS) {
                    vec3 a_diff, a_spec;
                    float a_fd;
                    area_light_eval(i, v_world_pos, N, V, NdotV, roughness, f0, a_diff, a_spec, a_fd);
                    vec3 a_kd = max(vec3(1.0) - (f0 * env_brdf_approx(roughness, NdotV).x + vec3(env_brdf_approx(roughness, NdotV).y)), vec3(0.0));
                    vec3 a_lit = a_kd * diffuse_color * a_diff;
                    if (FEAT_TOON) {
                        // Cel ramp on the form factor, like N.L for a point light.
                        float steps = max(params0.z, 2.0);
                        float scaled = clamp(a_fd * 2.0, 0.0, 1.0) * steps;
                        float lower = floor(scaled);
                        float sm = clamp(fx.toon.y, 0.01, 0.49);
                        float banded = (lower + smoothstep(0.5 - sm, 0.5 + sm, scaled - lower)) / steps;
                        float smooth_f = clamp(a_fd * 2.0, 0.0, 1.0);
                        vec3 toon_lit = a_kd * diffuse_color * (a_diff / max(a_fd, 1e-5)) * (banded * 0.5);
                        a_lit = mix(a_lit, toon_lit, clamp(params0.w, 0.0, 1.0));
                    }
                    color += a_lit + a_spec;
                    if (WRITE_FLUX_SOURCE) transport_diffuse+=diffuse_color*a_diff;
                }
                continue;
            }
            // Baked/Mixed light (flagged spot.y<0 & spot.w>0): its energy is already
            // in the lightmap, so skip it for lightmapped (static) surfaces, but DO
            // apply it to non-lightmapped objects (dynamic / added-after-bake) so they
            // aren't black in a baked scene. Realtime lights are never skipped.
            bool light_baked = light.spot.y < 0.0 && light.spot.w > 0.5;
            bool baked_here = light_baked && has_lightmap;
            if (baked_here && !caustics) {
                continue;
            }

            // Direction to the light + distance-based attenuation.
            vec3 L;
            float attenuation = 1.0;
            // Lights are small spheres, not points: their angular size widens
            // the specular lobe (Karis 2013 sphere-light roughness), which keeps
            // a mirror-smooth highlight finite instead of overflowing half floats.
            float a_spec = roughness * roughness;
            float a_light = a_spec + 0.00465;
            if (kind == 0) {
                L = normalize(-light.dir_range.xyz);
            } else {
                vec3 to_light = light.pos_kind.xyz - v_world_pos;
                float dist = length(to_light);
                a_light = min(a_spec + LIGHT_SOURCE_RADIUS / (2.0 * max(dist, 1e-3)), 1.0);
                L = (dist > 1e-5) ? to_light / dist : N;
                // Smooth inverse-square-ish falloff clamped to the range.
                attenuation = realtime_distance_falloff(dist,light.dir_range.w);
                if (kind == 2) {
                    // Spot cone: full inside the inner angle, smooth to the outer.
                    float cos_dir = dot(normalize(light.dir_range.xyz), -L);
                    float cos_outer = light.color.w;
                    float cos_inner = light.spot.x;
                    attenuation *= realtime_spot_falloff(cos_dir,cos_inner,cos_outer);
                }
            }

            vec3 radiance = light.color.rgb * attenuation;
            if (radiance == vec3(0.0)) {
                continue;
            }
            // Facing away: every term below is scaled by N·L (the toon ramp too),
            // so skip the BRDF and the shadow-map lookups.
            float NdotL = dot(N, L);
            if (translucent && !baked_here && NdotL < 0.2) {
                // Backlit: the light bent by the normal, seen along the view ray.
                vec3 Lt = normalize(L + N * fx.trans0.z);
                float tv = pow(clamp(dot(V, -Lt), 0.0, 1.0), max(fx.trans0.y, 0.1));
                float back = smoothstep(0.2, -0.3, NdotL);
                if (tv * back > 1e-3) {
                    // The shadow lookup is offset to the light's side of the surface.
                    trans_acc += radiance * shadow_factor(light, v_world_pos, -N, L) * (tv * back);
                }
            }
            if (NdotL <= 0.0) {
                continue;
            }
            float sf = 1.0;
            if (!(FEAT_WATER && (water_off & 64u) != 0u) || baked_here) {
                sf = shadow_factor(light, v_world_pos, N, L);
            }
            if (baked_here && sf <= 0.0) {
                continue;
            }
            float caustic_gain = 1.0;
#ifdef LIGHT_LOOP_CAUSTICS
            if (sf > 0.0) {
                caustic_gain = water_caustic_gain(v_world_pos, light.dir_range.xyz, light.pos_kind.xyz, kind != 0, i);
            }
#endif
            // A baked light is already in the lightmap: only the caustics'
            // change to it is added here.
            if (baked_here) {
                if (abs(caustic_gain - 1.0) < 1e-3) {
                    continue;
                }
                caustic_gain -= 1.0;
            }
            vec3 H = normalize(V + L);
            float NdotH = max(dot(N, H), 0.0);
            float VdotH = max(dot(V, H), 0.0);

            // Cook-Torrance specular + energy-conserving Lambertian diffuse: the
            // Fresnel term F is the specular reflectance kS, so the diffuse keeps
            // only the remaining energy kD = (1 - F). (Metallic is already folded
            // into diffuse_color, which is 0 for pure metals.)
            vec3 F = f_schlick(VdotH, f0);
            float lobe = a_spec / a_light;
            vec3 spec = d_ggx(NdotH, a_light) * (lobe * lobe)
                * g_smith(NdotV, NdotL, roughness)
                * F
                / max(4.0 * NdotV * NdotL, 1e-4);
            vec3 kd = vec3(1.0) - F;
            vec3 lit = (kd * diffuse_color / PI + spec) * NdotL;

            if (FEAT_TOON) {
                // Cel ramp: quantize N·L into `steps` bands with a soft edge at each
                // boundary (a clean toon terminator without hard aliasing). `steps`
                // controls band count, params0.w blends toon<->smooth-PBR.
                float steps = max(params0.z, 2.0);
                float scaled = clamp(NdotL, 0.0, 1.0) * steps;
                float lower = floor(scaled);
                float sm = clamp(fx.toon.y, 0.01, 0.49);
                float edge = smoothstep(0.5 - sm, 0.5 + sm, scaled - lower);
                float banded = (lower + edge) / steps;
                // Specular as a crisp toon highlight, only on lit bands.
                vec3 lit_toon = (kd * diffuse_color / PI) * banded
                    + spec * smoothstep(0.0, 0.02, banded);
                lit = mix(lit, lit_toon, clamp(params0.w, 0.0, 1.0));
            }

            if (g_dbg_layer >= 0 && g_dbg_lit < 0.0) {
                g_dbg_lit = sf;
            }
            color += lit * radiance * sf * caustic_gain;
            if (WRITE_FLUX_SOURCE) transport_diffuse+=diffuse_color*(NdotL/PI)*radiance*sf;
        }
    }
