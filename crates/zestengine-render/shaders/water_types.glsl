// Shared water storage ABI (GpuWater in types.rs). No lighting-frame fields.
#ifndef ZEST_WATER_TYPES
#define ZEST_WATER_TYPES
const int MAX_WATER_BODIES = 16;
const int MAX_WATER_WAVES = 32;
const int MAX_WATER_FOAM = 4;
struct GpuWater {
    vec4 wave_a[MAX_WATER_WAVES]; // xy direction (local XZ), z wave number, w angular speed
    vec4 wave_b[MAX_WATER_WAVES]; // xyz wave; w lanes: absorption RGB/IOR, scattering RGB/anisotropy
    vec4 shallow; // rgb shallow colour, w absorption depth
    vec4 deep;    // rgb deep colour, w clarity
    vec4 look0;   // x roughness, y ripple strength, z ripple scale, w foam width
    vec4 look1;   // x crest foam, y refraction, z wave count, w summed amplitude
    vec4 ripple;  // xy ripple window centre (local XZ), z width (0 = none)
    mat4 to_local; // world to the body's local space
    vec4 bounds;   // xy half extents in local XZ (0 = endless), z rest surface, w caustic strength
    vec4 foam[MAX_WATER_FOAM]; // churned spots: xy centre (local XZ), z radius, w amount
    vec4 extra;    // x lights under the surface throw caustics out of it (pools), y shore depth (0 = off), zw shore texture half extents (local XZ)
    vec4 reflect;  // x reflection mode (0 env, 1 planar, 2 probes, 3 off), yz local current (m/s)
    uvec4 coverage; // kind, vec4 offset/count, depth bits
    uvec4 exclusions; // dry-box matrix offset/count in coverage storage
    uvec4 ocean; // FFT field offset, resolution, patch-size bits, reserved
    uvec4 recording; // recorded field offset/resolution
    vec4 recording_window; // xy first cell centre, z cell size
    uvec4 connections; // receiving slots + 1; zero disables
    vec4 connection_lengths; // upstream/downstream bands, total river arc length
    vec4 ownership; // rank, enabled, conservative wave height, reserved
    mat4 zone_to_water;
    vec4 zone_camera;
    vec4 zone_lod; // tile quads, LOD scale, base cell, enabled
    vec4 underwater; // density, light falloff, enabled, legacy visibility
    vec4 quality;
    uvec4 shore_field;
};
float water_quality(float value){return value>0.0?value:3.0;}
#endif
