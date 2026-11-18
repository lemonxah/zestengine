    disp = vec3(0.0);
    nrm = vec3(0.0, 1.0, 0.0);
    crest = 0.0;
    lost = 0.0;
    float crest_max = 0.0;
    int n = min(int(WATER_WAVE_SOURCE.look1.z + 0.5), MAX_WATER_WAVES);
    for (int i = 0; i < n; ++i) {
        vec4 a = WATER_WAVE_SOURCE.wave_a[i];
        vec4 b = WATER_WAVE_SOURCE.wave_b[i];
        float w = water_band(a.z, foot);
        float phase = a.z * dot(a.xy, xz) - a.w * t;
        float s = sin(phase);
        float c = cos(phase);
        disp.x += w * b.x * a.x * c;
        disp.z += w * b.x * a.y * c;
        disp.y += w * b.y * s;
        float wa = w * a.z * b.y;
        nrm.x -= a.x * wa * c;
        nrm.z -= a.y * wa * c;
        nrm.y -= w * b.z * s;
        crest += w * b.z * s;
        crest_max += w * abs(b.z);
        float slope = a.z * b.y;
        lost += (1.0 - w * w) * 0.5 * slope * slope;
    }
    #ifdef WATER_OCEAN_DATA
    if(WATER_WAVE_SOURCE.ocean.y>0u){
        vec4 ocean=water_ocean_field(WATER_WAVE_SOURCE,xz,foot,0u);
        vec4 gradient=water_ocean_field(WATER_WAVE_SOURCE,xz,foot,1u);
        disp+=ocean.xyz;
        nrm.xz-=gradient.xy;
        lost+=0.5*max(gradient.w-dot(gradient.xy,gradient.xy),0.0);
    }
    if(WATER_WAVE_SOURCE.recording.y>0u){
        disp+=water_recorded_field(WATER_WAVE_SOURCE,xz,foot,0u).xyz;
        vec4 gradient=water_recorded_field(WATER_WAVE_SOURCE,xz,foot,1u);
        nrm.xz-=gradient.xy;lost+=0.5*max(gradient.w-dot(gradient.xy,gradient.xy),0.0);
    }
    #endif
    nrm = normalize(nrm);
    crest = crest_max > 1e-5 ? crest / crest_max : 0.0;
