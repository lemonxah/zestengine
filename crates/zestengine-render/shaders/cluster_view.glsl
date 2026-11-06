// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// The culled view (one per camera / shadow layer), shared by the culling shaders
// and the clustered vertex shader's debug tint. Mirrors GpuView in cluster.rs / gpu_cull.rs.

#ifndef CLUSTER_VIEW_GLSL
#define CLUSTER_VIEW_GLSL
struct View {
    vec4 planes[6];  // left, right, bottom, top, near, far (normalized)
    mat4 hiz_vp;     // view-projection the depth pyramid was rendered with
    vec4 hiz;        // xy pyramid mip-0 size, z mip count, w 1 = Hi-Z test on
    vec4 cam;        // xyz camera position, w max distance (0 = off)
    vec4 lod;        // x min projected radius px (0 = off), y px per unit at distance 1, z near-plane test, w cluster LOD target (px)
};
#endif
