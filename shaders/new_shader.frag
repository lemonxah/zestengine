// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

//! shader "Albedo Sampler"
//! prop tint color default(1, 1, 1, 1)

// Custom engine shader (fragment stage)
// Samples the albedo texture with optional tint.
// Uses the engine-provided inputs:
//   t_albedo   - albedo texture
//   v_uv       - texture coordinates
//   v_color    - vertex color
// Output: o_color

void main() {
    // Sample albedo texture and multiply by vertex color + tint
    vec4 albedo = texture(t_albedo, v_uv) * v_color;
    
    // Apply inspector tint
    vec3 finalColor = albedo.rgb * tint.rgb;
    
    // Output final color (preserve alpha from albedo if desired)
    o_color = vec4(finalColor, albedo.a);
}
