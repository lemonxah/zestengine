// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Material math shared by the visible surface, Flux capture and ray hits.
#ifndef ZEST_MATERIAL_RESPONSE
#define ZEST_MATERIAL_RESPONSE
vec3 material_decode_normal(vec3 sample_value) {
    vec3 n=sample_value*2.0-1.0;
    // BC5 has no blue channel. Reconstruct its positive tangent-space Z.
    if(n.z<=-0.999) n.z=sqrt(max(1.0-dot(n.xy,n.xy),0.0));
    return n;
}
vec3 material_detail_albedo(vec3 a,vec3 d,float strength,float mode_value) {
    int mode=int(mode_value+0.5);
    vec3 r=mode==0 ? a*d*4.594794 : mode==1 ?
        mix(2.0*a*d,1.0-2.0*(1.0-a)*(1.0-d),step(vec3(0.5),a)) :
        mode==2 ? a+d-0.2140 : d;
    return max(mix(a,r,clamp(strength,0.0,1.0)),vec3(0));
}
float material_emission_pulse(float time,float rate) {
    return rate>0.0 ? 0.5+0.5*sin(time*rate) : 1.0;
}
#endif
