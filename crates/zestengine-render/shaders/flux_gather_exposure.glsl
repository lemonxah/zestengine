// Lumen's gather limiter operates on pre-exposed ray radiance. Flux stores
// physical radiance, so transform only the limit into physical units instead.
// Shared surface/radiance caches must never depend on a camera's exposure.
#if !defined(FLUX_SURFACE_LIGHTING) && !defined(FLUX_RADIANCE_UPDATE)
layout(set=0,binding=FLUX_EXPOSURE_BINDING,std430) readonly buffer FluxExposure {
    uint histogram[64]; vec4 state;
} flux_exposure;
// Physical radiance per unit of displayed radiance, for the view recording this
// pass: two to the power of minus its manual plus adapted stops.
float flux_exposure_scale() {
    float stops=pc.exposure.x;
    // A new meter has no samples yet. Ignore its uninitialized adaptation
    // until the first HDR frame has been metered. Manual exposure still works.
    if(pc.exposure.y>0.5 && flux_exposure.state.w>0.0
        && !isnan(flux_exposure.state.x) && !isinf(flux_exposure.state.x)) stops+=flux_exposure.state.x;
    return exp2(-clamp(stops,-32.0,32.0));
}
#endif
float flux_gather_max_intensity() {
    float limit=max(pc.march.w,0.5);
#if !defined(FLUX_SURFACE_LIGHTING) && !defined(FLUX_RADIANCE_UPDATE)
    limit*=flux_exposure_scale();
#endif
    return limit;
}
// Radiosity rays are limited ten times higher than gather rays (Lumen: 40 against
// 10 pre-exposed). The limit is PHYSICAL: the surface cache is shared by every
// view and stores physical radiance, so it must not follow any view's exposure.
// Each gather ray that reads the cache is still limited by its own view's exposure.
float flux_radiosity_max_intensity() {
    return 10.0*max(pc.march.w,0.5);
}
// A light is skipped where it cannot visibly light a hit. Per-view gather shading
// follows that view's exposure, so a dim scene that adaptation brightens keeps its
// dim lights. Shared caches use one fixed, much lower floor: they never depend on a
// view, yet still store the light of very dim sources.
#if !defined(FLUX_SURFACE_LIGHTING) && !defined(FLUX_RADIANCE_UPDATE)
#define FLUX_LIGHT_CUTOFF (1e-3*min(flux_exposure_scale(),1.0))
#else
#define FLUX_LIGHT_CUTOFF 1e-5
#endif
