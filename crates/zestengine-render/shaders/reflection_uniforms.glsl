layout(set = 0, binding = REFLECTION_UNIFORM_BINDING) uniform SsrData {
    mat4 proj;
    mat4 inv_proj;
    mat4 view;
    mat4 inv_view;
    vec4 camera_pos;   // xyz world camera
    vec4 ssr;          // x enabled, y intensity, z max view dist, w roughness cutoff
    vec4 refl_center;  // xyz probe CAPTURE position (re-vector anchor), w probe intensity (0 = none)
    vec4 refl_extents; // xyz half-extents, w box-projection enabled
    vec4 fog_color;    // xyz colour, w density (0 = no fog)
    vec4 fog_params;   // x height falloff, y height ref, z start distance, w _
    vec4 fog_light;    // xyz sun colour*intensity, w anisotropy g
    vec4 fog_sun;      // xyz dir to sun, w time
    vec4 screen;       // x width, y height, z environment IBL intensity, w _
    vec4 refl_box;     // xyz probe box CENTER (capture + box offset), w _
    vec4 refl2_center; // second probe (binding 5), same layout; w = intensity (0 = none)
    vec4 refl2_extents;
    vec4 refl2_box;
    vec4 refl_blend;   // x slot 0 blend, y slot 1 blend, z slot 1 on top
    vec4 refl3_center; // third probe (binding 11)
    vec4 refl3_extents;
    vec4 refl3_box;
    vec4 refl4_center; // fourth probe (binding 12)
    vec4 refl4_extents;
    vec4 refl4_box;
    vec4 refl_blend2;  // x slot 2 blend, y slot 3 blend
} u;
