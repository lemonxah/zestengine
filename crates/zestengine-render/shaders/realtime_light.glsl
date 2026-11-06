// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// Shared realtime light units. Preserve the visible standard-material falloff
// while GI, surface radiosity and reflected hits use the same source energy.
// Offline FluxBaker retains its existing bake equations.
#ifndef ZEST_REALTIME_LIGHT
#define ZEST_REALTIME_LIGHT
float realtime_distance_falloff(float distance_to_light,float light_range) {
    float fade=clamp(1.0-distance_to_light/max(light_range,1e-3),0.0,1.0);
    return fade*fade/(1.0+distance_to_light*distance_to_light);
}
float realtime_spot_falloff(float cosine,float inner_cosine,float outer_cosine) {
    // Zero blend is a hard cone; smoothstep with identical edges is undefined.
    if(inner_cosine-outer_cosine<=1e-6) return cosine>=outer_cosine ? 1.0 : 0.0;
    float t=clamp((cosine-outer_cosine)/(inner_cosine-outer_cosine),0.0,1.0);
    return t*t*(3.0-2.0*t);
}
#endif
