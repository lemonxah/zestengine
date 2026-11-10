// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see LICENSE and NOTICE.
// Shared screen hit contract for SSR and screen-first hardware reflections.
// Geometry traversal never reads lighting. Only a front-facing, finite hit is
// accepted here. Confidence at screen edges leaves the unresolved contribution
// to the consumer's world tracer/environment. Inputs are current physical HDR,
// before this reflection resolve or display exposure (no recursive feedback).
#include "flux_screen_trace.glsl"

vec4 flux_reflection_screen(sampler2D depth, sampler2D gbuf, sampler2D color,
    vec3 origin, vec3 direction, float max_distance, mat4 proj, mat4 inv_proj,
    bool flip_y, float confidence_scale, float recursive_roughness, out float distance) {
    distance = 0.0;
    FluxScreenHit hit = flux_trace_screen(depth, origin, direction, max_distance,
        proj, inv_proj, flip_y, 96u);
    if (!hit.valid) return vec4(0.0);
    ivec2 texel = clamp(ivec2(hit.uv * vec2(textureSize(gbuf, 0))),
        ivec2(0), textureSize(gbuf, 0) - 1);
    vec4 material = texelFetch(gbuf, texel, 0);
    // Current pre-resolve HDR only contains environment specular. A visible
    // mirror cannot terminate a multi-bounce path with that incomplete source.
    // Send reflective hits to the world continuation until resolved reflection
    // history is available as a separate, validated source.
    uint packed_b = uint(round(material.b * 1023.0));
    uint packed_a = uint(round(material.a * 3.0));
    float roughness = float(packed_b >> 4) / 63.0;
    float reflectivity = float((packed_b & 15u) | (packed_a << 4)) / 63.0;
    if (reflectivity > 1e-4 && roughness <= recursive_roughness) return vec4(0.0);
    if (dot(oct_decode(material.rg), direction) >= 0.05)
        return vec4(0.0);
    vec3 radiance = textureLod(color, hit.uv, 0.0).rgb;
    if (any(isnan(radiance)) || any(isinf(radiance))) return vec4(0.0);
    vec2 edge = smoothstep(vec2(0.0), vec2(0.06), hit.uv)
        * (1.0 - smoothstep(vec2(0.94), vec2(1.0), hit.uv));
    float confidence = smoothstep(0.45, 0.8, edge.x * edge.y * confidence_scale);
    distance = hit.distance;
    return vec4(clamp(radiance, vec3(0.0), vec3(65000.0)), confidence);
}
