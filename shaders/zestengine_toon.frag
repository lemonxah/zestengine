// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// @shader "ZestEngine Toon"
//
// Poiyomi-style toon uber-shader on the custom shader system. Every feature
// group is a keyword (specialization constant): a material's pipeline only
// contains the features it enables, so a Simple material costs about as much
// as a plain toon shader. Presets set keyword combinations; individual
// features can still be toggled afterwards.
//
// @preset "Simple" lighting_mode=Multilayer RECEIVE_SHADOWS=1 NORMAL_MAP=0 ALPHA=0 RIM=0 MATCAP=0 MATCAP2=0 SPECULAR=0 REFLECTIONS=0 EMISSION=0 EMISSION2=0 DETAIL=0 DECAL=0 DISSOLVE=0 GLITTER=0 UV_DISCARD=0 PARALLAX=0 VERTEX_COLOR=0 OUTLINE=0 COLOR_MASK=0 DISTANCE_FADE=0 PANOSPHERE=0 FLIPBOOK=0 GRAB=0 AUDIO=0 FUR=0
// @preset "Medium" lighting_mode=Multilayer RECEIVE_SHADOWS=1 NORMAL_MAP=1 ALPHA=0 RIM=1 MATCAP=1 MATCAP2=0 SPECULAR=1 specular_mode=Toon REFLECTIONS=0 EMISSION=1 EMISSION2=0 DETAIL=0 DECAL=0 DISSOLVE=0 GLITTER=0 UV_DISCARD=0 PARALLAX=0 VERTEX_COLOR=0 OUTLINE=0 COLOR_MASK=0 DISTANCE_FADE=0 PANOSPHERE=0 FLIPBOOK=0 GRAB=0 AUDIO=0 FUR=0
// @preset "Heavy" lighting_mode=Multilayer RECEIVE_SHADOWS=1 NORMAL_MAP=1 ALPHA=1 RIM=1 MATCAP=1 MATCAP2=1 SPECULAR=1 specular_mode=Anisotropic REFLECTIONS=1 EMISSION=1 EMISSION2=1 DETAIL=1 DECAL=1 DISSOLVE=1 GLITTER=1 UV_DISCARD=1 PARALLAX=1 VERTEX_COLOR=1 OUTLINE=1 COLOR_MASK=0 DISTANCE_FADE=0 PANOSPHERE=0 FLIPBOOK=0 GRAB=0 AUDIO=0 FUR=0
//
// @property main_color "Color" Color = (1, 1, 1, 1)
// @property main_st "Main Tiling / Offset" Vector = (1, 1, 0, 0) @tooltip("Albedo, normal and ORM slots: xy tiling, zw offset")
// @property saturation "Saturation" Range(0, 2) = 1
// @property hue_shift "Hue Shift" Range(0, 1) = 0 @tooltip("Turns around the colour wheel")
// @property hue_speed "Hue Shift Speed" Range(0, 2) = 0 @tooltip("Turns per second added to the hue shift")
// @property brightness "Brightness" Range(0, 2) = 1
// @property alpha_cutoff "Alpha Cutoff" Range(0, 1) = 0.5 @tooltip("Used when the material's Transparency mode is Cutout")
// @property lighting_mode "Lighting" Enum(Ramp, Multilayer, Realistic, Flat, Wrapped) = Multilayer @keyword
// @property RECEIVE_SHADOWS "Receive Shadows" Toggle = 1 @keyword
//
// @foldout "Shading"
// @property shadow_color "Shadow Color" Color3 = (0.62, 0.55, 0.7)
// @property shadow_strength "Shadow Strength" Range(0, 1) = 1
// @property shadow_mask "Shadow Mask" Texture2D = white @linear
// @property ramp_tex "Ramp" Texture2D = white @show_if(lighting_mode == Ramp)
// @property ramp_offset "Ramp Offset" Range(-1, 1) = 0 @show_if(lighting_mode == Ramp)
// @header "Layer 1"
// @property layer1_border "Border" Range(0, 1) = 0.5 @show_if(lighting_mode == Multilayer)
// @property layer1_blur "Blur" Range(0, 0.5) = 0.04 @show_if(lighting_mode == Multilayer)
// @header "Layer 2"
// @property layer2_color "Color" Color3 = (0.35, 0.3, 0.45) @show_if(lighting_mode == Multilayer)
// @property layer2_strength "Strength" Range(0, 1) = 0 @show_if(lighting_mode == Multilayer)
// @property layer2_border "Border" Range(0, 1) = 0.25 @show_if(lighting_mode == Multilayer)
// @property layer2_blur "Blur" Range(0, 0.5) = 0.04 @show_if(lighting_mode == Multilayer)
// @header "Layer 3"
// @property layer3_color "Color" Color3 = (0.25, 0.22, 0.35) @show_if(lighting_mode == Multilayer)
// @property layer3_strength "Strength" Range(0, 1) = 0 @show_if(lighting_mode == Multilayer)
// @property layer3_border "Border" Range(0, 1) = 0.1 @show_if(lighting_mode == Multilayer)
// @property layer3_blur "Blur" Range(0, 0.5) = 0.04 @show_if(lighting_mode == Multilayer)
// @header "Wrapped"
// @property wrap_amount "Wrap" Range(0, 1) = 0.5 @show_if(lighting_mode == Wrapped) @tooltip("How far light wraps past the terminator; 1 is a half-lambert")
// @header "Realistic"
// @property toon_blend "Toon Blend" Range(0, 1) = 0 @show_if(lighting_mode == Realistic)
// @property metallic "Metallic" Range(0, 1) = 0 @show_if(lighting_mode == Realistic)
// @property roughness "Roughness" Range(0, 1) = 0.6 @show_if(lighting_mode == Realistic)
// @header "Light"
// @property light_min "Min Brightness" Range(0, 1) = 0.02
// @property light_max "Max Brightness" Range(0, 10) = 2
// @property monochrome_light "Monochrome Lighting" Range(0, 1) = 0
// @property indirect_strength "Indirect Strength" Range(0, 2) = 1
// @property occlusion_strength "Occlusion (ORM.r)" Range(0, 1) = 1
// @end
//
// @feature COLOR_MASK "RGBA Colour Masking"
// @property cmask_tex "Mask (RGBA)" Texture2D = black @linear @tooltip("Each channel selects a region that is tinted / hue shifted separately")
// @property cmask_r "Red Tint" Color3 = (1, 1, 1)
// @property cmask_g "Green Tint" Color3 = (1, 1, 1)
// @property cmask_b "Blue Tint" Color3 = (1, 1, 1)
// @property cmask_a "Alpha Tint" Color3 = (1, 1, 1)
// @property cmask_hue "Hue Shift (R G B A)" Vector = (0, 0, 0, 0) @tooltip("Turns per channel")
// @end
//
// @feature NORMAL_MAP "Normal Map"
// @property normal_strength "Strength" Range(0, 2) = 1
// @end
//
// @feature ALPHA "Alpha Options"
// @property alpha_mask "Alpha Mask" Texture2D = white @linear
// @property alpha_mod "Alpha Offset" Range(-1, 1) = 0
// @property alpha_dither "Dithered Alpha" Toggle = 0 @tooltip("Screen-door transparency that works in opaque / cutout mode")
// @property alpha_fresnel "Fresnel Fade" Range(0, 1) = 0
// @property alpha_to_emission "Alpha Boost" Range(0, 1) = 0 @tooltip("Keep emission visible where alpha fades")
// @end
//
// @feature RIM "Rim Lighting"
// @property rim_color "Color" Color3(HDR) = (1, 1, 1)
// @property rim_strength "Strength" Range(0, 4) = 1
// @property rim_width "Width" Range(0, 1) = 0.35
// @property rim_sharpness "Sharpness" Range(0, 1) = 0.25
// @property rim_light_based "Light Based" Range(0, 1) = 0.5
// @property rim_blend "Blend" Enum(Add, Replace, Multiply) = Add
// @property rim_mask "Mask" Texture2D = white @linear
// @end
//
// @feature MATCAP "Matcap 1"
// @property matcap_tex "Matcap" Texture2D = black
// @property matcap_mask "Mask" Texture2D = white @linear
// @property matcap_strength "Strength" Range(0, 2) = 1
// @property matcap_blend "Blend" Enum(Add, Multiply, Replace, Screen) = Add
// @property matcap_lit "Light Based" Range(0, 1) = 0
// @end
//
// @feature MATCAP2 "Matcap 2"
// @property matcap2_tex "Matcap" Texture2D = black
// @property matcap2_mask "Mask" Texture2D = white @linear
// @property matcap2_strength "Strength" Range(0, 2) = 1
// @property matcap2_blend "Blend" Enum(Add, Multiply, Replace, Screen) = Multiply
// @property matcap2_lit "Light Based" Range(0, 1) = 0
// @end
//
// @feature SPECULAR "Specular"
// @property specular_mode "Mode" Enum(Toon, Anisotropic, Realistic) = Toon @keyword
// @property spec_color "Color" Color3(HDR) = (1, 1, 1)
// @property spec_strength "Strength" Range(0, 4) = 0.5
// @property spec_size "Size" Range(0, 1) = 0.2 @show_if(specular_mode != Anisotropic)
// @property spec_smooth "Smoothness" Range(0, 0.5) = 0.02 @show_if(specular_mode == Toon)
// @property spec_mask "Mask" Texture2D = white @linear
// @header "Anisotropic (hair)"
// @property aniso_shift "Shift Noise" Texture2D = gray @linear @show_if(specular_mode == Anisotropic)
// @property aniso_shift_amount "Shift" Range(-1, 1) = 0 @show_if(specular_mode == Anisotropic)
// @property aniso_noise "Noise Strength" Range(0, 1) = 0.3 @show_if(specular_mode == Anisotropic)
// @property aniso_exponent "Exponent" Range(1, 512) = 96 @show_if(specular_mode == Anisotropic)
// @property aniso_direction "Direction" Enum(Tangent, Bitangent) = Bitangent @show_if(specular_mode == Anisotropic)
// @property aniso2_color "Secondary Color" Color3 = (0.8, 0.6, 0.5) @show_if(specular_mode == Anisotropic)
// @property aniso2_shift "Secondary Shift" Range(-1, 1) = -0.2 @show_if(specular_mode == Anisotropic)
// @property aniso2_exponent "Secondary Exponent" Range(1, 512) = 24 @show_if(specular_mode == Anisotropic)
// @end
//
// @feature REFLECTIONS "Reflections"
// @property reflection_strength "Strength" Range(0, 2) = 1
// @property reflection_roughness "Roughness" Range(0, 1) = 0.3
// @property reflection_fresnel "Fresnel" Range(0, 1) = 0.5
// @end
//
// @feature EMISSION "Emission"
// @property emission_color "Color" Color3(HDR) = (1, 1, 1)
// @property emission_strength "Strength" Range(0, 20) = 1 @tooltip("Uses the material's Emission texture slot")
// @property emission_mask "Mask" Texture2D = white @linear
// @property emission_scroll "Scroll Speed" Vector2 = (0, 0)
// @property emission_pulse_speed "Pulse Speed" Range(0, 20) = 0
// @property emission_pulse_amount "Pulse Amount" Range(0, 1) = 1
// @property emission_hue_speed "Hue Shift Speed" Range(0, 2) = 0 @tooltip("Colour over time")
// @end
//
// @feature EMISSION2 "Emission 2"
// @property emission2_tex "Texture" Texture2D = white
// @property emission2_color "Color" Color3(HDR) = (1, 0.4, 0.8)
// @property emission2_strength "Strength" Range(0, 20) = 1
// @property emission2_scroll "Scroll Speed" Vector2 = (0, 0)
// @property emission2_pulse_speed "Pulse Speed" Range(0, 20) = 2
// @property emission2_pulse_amount "Pulse Amount" Range(0, 1) = 0
// @property emission2_hue_speed "Hue Shift Speed" Range(0, 2) = 0
// @end
//
// @feature DETAIL "Detail Textures"
// @property detail_albedo "Detail Albedo (x2)" Texture2D = gray @linear
// @property detail_normal "Detail Normal" Texture2D = normal
// @property detail_mask "Mask" Texture2D = white @linear
// @property detail_albedo_strength "Albedo Strength" Range(0, 1) = 1
// @property detail_normal_strength "Normal Strength" Range(0, 2) = 1
// @end
//
// @feature DECAL "Decal"
// @property decal_tex "Texture" Texture2D = white
// @property decal_color "Tint" Color = (1, 1, 1, 1)
// @property decal_position "Position (UV)" Vector2 = (0.5, 0.5)
// @property decal_scale "Scale" Vector2 = (0.25, 0.25)
// @property decal_rotation "Rotation" Range(0, 360) = 0
// @property decal_blend "Blend" Enum(Replace, Multiply, Add) = Replace
// @property decal_emission "Emission" Range(0, 10) = 0
// @end
//
// @feature DISSOLVE "Dissolve"
// @property dissolve_noise "Noise" Texture2D = gray @linear
// @property dissolve_amount "Amount" Range(0, 1) = 0
// @property dissolve_edge_width "Edge Width" Range(0, 0.25) = 0.04
// @property dissolve_edge_color "Edge Color" Color3(HDR) = (4, 1.5, 0.3)
// @property dissolve_invert "Invert" Toggle = 0
// @end
//
// @feature GLITTER "Glitter"
// @property glitter_color "Color" Color3(HDR) = (2, 2, 2)
// @property glitter_mask "Mask" Texture2D = white @linear
// @property glitter_density "Density" Range(10, 600) = 180
// @property glitter_size "Size" Range(0.05, 0.5) = 0.25
// @property glitter_speed "Speed" Range(0, 10) = 2
// @property glitter_view "View Dependence" Range(0, 1) = 0.6
// @property glitter_rainbow "Rainbow" Range(0, 1) = 0
// @property glitter_lit "Light Based" Range(0, 1) = 0.3
// @end
//
// @feature UV_DISCARD "UV Tile Discard"
// @property discard_uv "UV Channel" Enum(UV0, UV1) = UV0
// @header "1 = discard tile (u 0-3 in xyzw)"
// @property discard_row0 "Row v 0" Vector = (0, 0, 0, 0)
// @property discard_row1 "Row v 1" Vector = (0, 0, 0, 0)
// @property discard_row2 "Row v 2" Vector = (0, 0, 0, 0)
// @property discard_row3 "Row v 3" Vector = (0, 0, 0, 0)
// @end
//
// @feature PARALLAX "Parallax"
// @property height_tex "Height" Texture2D = white @linear
// @property height_scale "Scale" Range(0, 0.15) = 0.03
// @property parallax_steps "Steps" IntRange(4, 48) = 16
// @end
//
// @feature VERTEX_COLOR "Vertex Color"
// @property vc_mode "Mode" Enum(Multiply, Tint, AlphaOnly, EmissionMask) = Multiply
// @property vc_strength "Strength" Range(0, 1) = 1
// @property vc_linearize "Linearize (sRGB)" Toggle = 0
// @end
//
// @feature OUTLINE "Outline"
// @property outline_color "Color" Color = (0.05, 0.04, 0.06, 1) @tooltip("Drawn by the engine as an inverted hull after the object")
// @property outline_width "Width" Range(0, 10) = 1
// @property outline_vertex_width "Width From Vertex Colour" Toggle = 0 @tooltip("Scales the width by the red vertex colour channel (paint thin areas black)")
// @end
//
// @feature FUR "Fur Shells"
// @property fur_tip "Tip Colour" Color3 = (0.85, 0.7, 0.5) @tooltip("Drawn by the engine as shells after the object (opaque, strand-masked); the shells are shaded by a simple key-light term, not this shader")
// @property fur_root "Root Colour" Color3 = (0.25, 0.16, 0.1)
// @property fur_length "Length (m)" Range(0, 0.5) = 0.03
// @property fur_layers "Shells" Range(2, 48) = 16 @tooltip("One extra draw of the mesh per shell")
// @property fur_density "Density" Range(10, 600) = 150 @tooltip("Strands per UV unit")
// @end
//
// @feature DISTANCE_FADE "Distance Fade"
// @property fade_min "Start Distance" Range(0, 100) = 1 @tooltip("Fully faded at or below this distance (metres)")
// @property fade_max "End Distance" Range(0, 100) = 3 @tooltip("Fully visible at or beyond this distance")
// @property fade_invert "Invert" Toggle = 0 @tooltip("Fade out with distance instead (far fade)")
// @end
//
// @feature PANOSPHERE "Panosphere"
// @property pano_tex "Texture (equirectangular)" Texture2D = black
// @property pano_color "Tint" Color3(HDR) = (1, 1, 1)
// @property pano_amount "Amount" Range(0, 1) = 1
// @property pano_blend "Blend" Enum(Replace, Add, Multiply) = Replace
// @property pano_spin "Spin Speed" Range(-1, 1) = 0 @tooltip("Turns per second around the up axis")
// @property pano_rotation "Rotation" Range(0, 1) = 0
// @property pano_unlit "Unlit" Toggle = 1 @tooltip("Skip lighting where the panosphere shows; pair with Cull Front Faces on a sphere mesh")
// @end
//
// @feature FLIPBOOK "Flipbook"
// @property flip_tex "Sprite Sheet" Texture2D = white
// @property flip_color "Tint" Color = (1, 1, 1, 1)
// @property flip_cols "Columns" IntRange(1, 16) = 4
// @property flip_rows "Rows" IntRange(1, 16) = 4
// @property flip_frames "Frames" IntRange(1, 256) = 16 @tooltip("Frames used, counted row by row from the top left")
// @property flip_fps "Frames Per Second" Range(0, 60) = 12
// @property flip_manual "Manual Frame" Float = -1 @tooltip("0 or more picks a frame (drive it from an animation); negative plays by time")
// @property flip_crossfade "Crossfade" Toggle = 0
// @property flip_blend "Blend" Enum(Replace, Add, Multiply) = Replace
// @property flip_amount "Amount" Range(0, 1) = 1
// @property flip_emission "Emission" Range(0, 10) = 0
// @end
//
// @feature GRAB "Grab Pass (Refraction)"
// @property grab_amount "Amount" Range(0, 1) = 1
// @property grab_tint "Tint" Color3 = (1, 1, 1)
// @property grab_distortion "Distortion" Range(0, 1) = 0.1 @tooltip("Shifts the background by the surface normal, in 100 pixel units")
// @property grab_blur "Blur Radius (px)" Range(0, 32) = 0
// @property grab_chromatic "Chromatic Split" Range(0, 1) = 0 @tooltip("Different distortion per colour channel")
// @end
//
// @feature AUDIO "Audio Reactive"
// @property audio_bands "Bands (bass, low mid, high mid, treble)" Vector = (0, 0, 0, 0) @tooltip("Written by the engine's audio analysis, or animate it by hand")
// @property audio_volume "Volume" Range(0, 1) = 0
// @property audio_emission "Emission Gain" Range(0, 20) = 0
// @property audio_emission_band "Emission Band" Enum(Bass, LowMid, HighMid, Treble, Volume) = Bass
// @property audio_alpha "Alpha Gain" Range(-1, 1) = 0
// @property audio_alpha_band "Alpha Band" Enum(Bass, LowMid, HighMid, Treble, Volume) = Volume
// @property audio_rim "Rim Gain" Range(0, 4) = 0
// @property audio_rim_band "Rim Band" Enum(Bass, LowMid, HighMid, Treble, Volume) = Treble
// @property audio_hue "Hue Shift Gain" Range(0, 1) = 0
// @property audio_hue_band "Hue Band" Enum(Bass, LowMid, HighMid, Treble, Volume) = LowMid
// @end

vec3 ct_hue_shift(vec3 c, float turns) {
    float a = turns * 6.28318530718;
    const vec3 k = vec3(0.57735);
    float ca = cos(a);
    return c * ca + cross(k, c) * sin(a) + k * dot(k, c) * (1.0 - ca);
}

float ct_hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

vec3 ct_blend(vec3 base, vec3 layer, float t, int mode) {
    if (mode == 0) return base + layer * t;
    if (mode == 1) return mix(base, base * layer, t);
    if (mode == 2) return mix(base, layer, t);
    return mix(base, 1.0 - (1.0 - base) * (1.0 - layer), t);
}

// Lit fraction (rgb, shadow-tinted) for a half-lambert term in [0, 1].
vec3 ct_toon(float x, float shadow_amt) {
    vec3 shadow_tint = mix(vec3(1.0), shadow_color, shadow_amt);
    if (lighting_mode == lighting_mode_Ramp) {
        vec3 ramp = texture(ramp_tex, vec2(clamp(x + ramp_offset, 0.01, 0.99), 0.5)).rgb;
        return mix(vec3(1.0), ramp, shadow_amt);
    }
    if (lighting_mode == lighting_mode_Multilayer) {
        float t1 = smoothstep(layer1_border - layer1_blur, layer1_border + layer1_blur + 1e-4, x);
        vec3 c = mix(shadow_tint, vec3(1.0), t1);
        float t2 = smoothstep(layer2_border - layer2_blur, layer2_border + layer2_blur + 1e-4, x);
        c = mix(c, c * mix(vec3(1.0), layer2_color, shadow_amt), layer2_strength * (1.0 - t2));
        float t3 = smoothstep(layer3_border - layer3_blur, layer3_border + layer3_blur + 1e-4, x);
        return mix(c, c * mix(vec3(1.0), layer3_color, shadow_amt), layer3_strength * (1.0 - t3));
    }
    return vec3(1.0);
}

float ct_audio(int band) {
    return band == 4 ? audio_volume : audio_bands[clamp(band, 0, 3)];
}

vec2 ct_matcap_uv(vec3 n) {
    vec3 vn = normalize(mat3(frame.view) * n);
    return vn.xy * 0.5 + 0.5;
}

void main() {
    vec3 flip_glow = vec3(0.0);
    vec3 pano_col = vec3(0.0);
    vec2 uv0 = v_uv;
    vec2 uv = uv0 * main_st.xy + main_st.zw;
    vec3 Ng = normalize(gl_FrontFacing ? v_normal : -v_normal);
    vec3 V = normalize(u_camera_pos - v_world_pos);
    vec3 T = normalize(v_tangent.xyz - Ng * dot(Ng, v_tangent.xyz));
    vec3 B = cross(Ng, T) * (v_tangent.w < 0.0 ? -1.0 : 1.0);

    if (UV_DISCARD) {
        vec2 duv = discard_uv == discard_uv_UV1 ? v_uv1 : uv0;
        ivec2 tile = ivec2(floor(duv));
        if (tile.x >= 0 && tile.x < 4 && tile.y >= 0 && tile.y < 4) {
            vec4 row = tile.y == 0 ? discard_row0 : tile.y == 1 ? discard_row1 : tile.y == 2 ? discard_row2 : discard_row3;
            if (row[tile.x] > 0.5) discard;
        }
    }

    if (PARALLAX) {
        vec3 vt = normalize(vec3(dot(V, T), dot(V, B), dot(V, Ng)));
        float steps = float(parallax_steps);
        vec2 delta = vt.xy / max(vt.z, 0.15) * height_scale / steps;
        float layer = 1.0;
        float h = texture(height_tex, height_tex_uv(uv)).r;
        vec2 puv = uv;
        for (int i = 0; i < 48 && i < parallax_steps && h < layer; ++i) {
            layer -= 1.0 / steps;
            puv -= delta;
            h = texture(height_tex, height_tex_uv(puv)).r;
        }
        uv = puv;
    }

    vec4 albedo = texture(t_albedo, uv) * main_color;
    if (VERTEX_COLOR) {
        vec4 vc = v_color;
        if (vc_linearize) vc.rgb = pow(vc.rgb, vec3(2.2));
        if (vc_mode == vc_mode_Multiply) albedo *= mix(vec4(1.0), vc, vc_strength);
        else if (vc_mode == vc_mode_Tint) albedo.rgb = mix(albedo.rgb, albedo.rgb * vc.rgb * 2.0, vc_strength);
        else if (vc_mode == vc_mode_AlphaOnly) albedo.a *= mix(1.0, vc.a, vc_strength);
    }
    if (DETAIL) {
        float dm = texture(detail_mask, detail_mask_uv(uv0)).r;
        vec3 d = texture(detail_albedo, detail_albedo_uv(uv0)).rgb * 2.0;
        albedo.rgb *= mix(vec3(1.0), d, dm * detail_albedo_strength);
    }
    if (COLOR_MASK) {
        vec4 cm = texture(cmask_tex, cmask_tex_uv(uv0));
        vec3 a = albedo.rgb;
        a = mix(a, ct_hue_shift(a, cmask_hue.x) * cmask_r, cm.r);
        a = mix(a, ct_hue_shift(a, cmask_hue.y) * cmask_g, cm.g);
        a = mix(a, ct_hue_shift(a, cmask_hue.z) * cmask_b, cm.b);
        a = mix(a, ct_hue_shift(a, cmask_hue.w) * cmask_a, cm.a);
        albedo.rgb = a;
    }
    if (FLIPBOOK) {
        vec2 fuv = flip_tex_uv(uv0);
        vec2 grid = vec2(max(flip_cols, 1), max(flip_rows, 1));
        vec2 dx = dFdx(fuv) / grid, dy = dFdy(fuv) / grid;
        float f = flip_manual >= 0.0 ? flip_manual : u_time * flip_fps;
        int frames = max(min(flip_frames, int(grid.x * grid.y)), 1);
        float f0 = floor(f);
        int i0 = int(mod(f0, float(frames)));
        int i1 = int(mod(f0 + 1.0, float(frames)));
        // Stay half a texel inside the tile so bilinear filtering never reads the neighbour frame.
        vec2 inset = 0.5 * grid / vec2(textureSize(flip_tex, 0));
        vec2 cell = clamp(fuv, inset, vec2(1.0) - inset);
        vec2 t0 = (vec2(float(i0) - floor(float(i0) / grid.x) * grid.x, floor(float(i0) / grid.x)) + cell) / grid;
        vec2 t1 = (vec2(float(i1) - floor(float(i1) / grid.x) * grid.x, floor(float(i1) / grid.x)) + cell) / grid;
        vec4 fc = textureGrad(flip_tex, t0, dx, dy);
        if (flip_crossfade) fc = mix(fc, textureGrad(flip_tex, t1, dx, dy), fract(f));
        fc *= flip_color;
        if (any(lessThan(fuv, vec2(0.0))) || any(greaterThan(fuv, vec2(1.0)))) fc = vec4(0.0);
        float fa = fc.a * flip_amount;
        if (flip_blend == flip_blend_Replace) albedo.rgb = mix(albedo.rgb, fc.rgb, fa);
        else if (flip_blend == flip_blend_Add) albedo.rgb += fc.rgb * fa;
        else albedo.rgb = mix(albedo.rgb, albedo.rgb * fc.rgb, fa);
        flip_glow = fc.rgb * fa * flip_emission;
    }
    if (PANOSPHERE) {
        vec3 d = normalize(v_world_pos - u_camera_pos);
        float spin = (pano_rotation + u_time * pano_spin) * 6.28318530718;
        float cs = cos(spin), sn = sin(spin);
        d = vec3(cs * d.x + sn * d.z, d.y, -sn * d.x + cs * d.z);
        vec2 puv = vec2(atan(d.z, d.x) * 0.15915494 + 0.5, acos(clamp(d.y, -1.0, 1.0)) * 0.31830989);
        // The atan seam would pick the lowest mip; take the gradient from the wrapped coordinate.
        vec2 gx = dFdx(puv), gy = dFdy(puv);
        gx.x -= round(gx.x);
        gy.x -= round(gy.x);
        pano_col = textureGrad(pano_tex, puv, gx, gy).rgb * pano_color;
        if (pano_blend == pano_blend_Replace) albedo.rgb = mix(albedo.rgb, pano_col, pano_amount);
        else if (pano_blend == pano_blend_Add) albedo.rgb += pano_col * pano_amount;
        else albedo.rgb = mix(albedo.rgb, albedo.rgb * pano_col, pano_amount);
    }
    if (DECAL) {
        float r = radians(decal_rotation);
        vec2 p = uv0 - decal_position;
        p = mat2(cos(r), -sin(r), sin(r), cos(r)) * p;
        vec2 duv = p / max(decal_scale, vec2(1e-4)) + 0.5;
        if (all(greaterThanEqual(duv, vec2(0.0))) && all(lessThanEqual(duv, vec2(1.0)))) {
            vec4 dc = texture(decal_tex, duv) * decal_color;
            if (decal_blend == decal_blend_Replace) albedo.rgb = mix(albedo.rgb, dc.rgb, dc.a);
            else if (decal_blend == decal_blend_Multiply) albedo.rgb = mix(albedo.rgb, albedo.rgb * dc.rgb, dc.a);
            else albedo.rgb += dc.rgb * dc.a;
        }
    }
    float luma = dot(albedo.rgb, vec3(0.2126, 0.7152, 0.0722));
    albedo.rgb = max(mix(vec3(luma), albedo.rgb, saturation), vec3(0.0));
    float hue_turns = hue_shift + u_time * hue_speed;
    if (AUDIO) hue_turns += audio_hue * ct_audio(audio_hue_band);
    if (hue_turns != 0.0) albedo.rgb = ct_hue_shift(albedo.rgb, hue_turns);
    albedo.rgb *= brightness;

    float alpha = albedo.a;
    if (ALPHA) {
        alpha = clamp(alpha * texture(alpha_mask, alpha_mask_uv(uv0)).r + alpha_mod, 0.0, 1.0);
        alpha *= mix(1.0, max(dot(Ng, V), 0.0), alpha_fresnel);
        if (alpha_dither && alpha < zestengine_dither(gl_FragCoord.xy)) discard;
    }
    if (AUDIO) alpha = clamp(alpha + audio_alpha * ct_audio(audio_alpha_band), 0.0, 1.0);
    if (DISTANCE_FADE) {
        float fd = distance(u_camera_pos, v_world_pos);
        float fa = smoothstep(fade_min, max(fade_max, fade_min + 1e-3), fd);
        if (fade_invert) fa = 1.0 - fa;
        alpha *= fa;
        // Opaque / cutout materials fade by screen-door dither.
        if (zestengine_alpha_mode != 2 && fa < zestengine_dither(gl_FragCoord.xy)) discard;
    }
    if (zestengine_alpha_mode == 1 && alpha < alpha_cutoff) discard;

    if (DISSOLVE) {
        float n = texture(dissolve_noise, dissolve_noise_uv(uv0)).r;
        if (dissolve_invert) n = 1.0 - n;
        if (n < dissolve_amount) discard;
    }

    vec3 N = Ng;
    if (NORMAL_MAP) {
        vec3 tn = texture(t_normal, uv).xyz * 2.0 - 1.0;
        tn.xy *= normal_strength;
        if (DETAIL) {
            vec3 dn = texture(detail_normal, detail_normal_uv(uv0)).xyz * 2.0 - 1.0;
            float dm = texture(detail_mask, detail_mask_uv(uv0)).r;
            tn.xy += dn.xy * detail_normal_strength * dm;
        }
        N = normalize(mat3(T, B, Ng) * normalize(tn));
    } else if (DETAIL) {
        vec3 dn = texture(detail_normal, detail_normal_uv(uv0)).xyz * 2.0 - 1.0;
        float dm = texture(detail_mask, detail_mask_uv(uv0)).r;
        dn.xy *= detail_normal_strength * dm;
        N = normalize(mat3(T, B, Ng) * normalize(dn));
    }
    float ndv = max(dot(N, V), 1e-4);

    vec3 orm = texture(t_orm, uv).rgb;
    float shadow_amt = shadow_strength * texture(shadow_mask, shadow_mask_uv(uv0)).r;
    float rough = clamp(roughness * orm.g, 0.03, 1.0);
    float metal = clamp(metallic * orm.b, 0.0, 1.0);
    vec3 f0 = mix(vec3(0.04), albedo.rgb, metal);
    float spec_m = SPECULAR ? texture(spec_mask, spec_mask_uv(uv0)).r : 0.0;

    vec3 direct = vec3(0.0);
    vec3 spec = vec3(0.0);
    float lit_sum = 0.0;
    for (int i = 0; i < u_light_count && i < MAX_LIGHTS; ++i) {
        vec3 L, rad;
        if (!zestengine_light(i, v_world_pos, L, rad)) continue;
        float ndl = dot(N, L);
        float sh = 1.0;
        if (RECEIVE_SHADOWS && ndl > -0.3) sh = zestengine_shadow(i, v_world_pos, Ng, L);
        float lambert = max(ndl, 0.0) * sh;
        float x = (ndl * 0.5 + 0.5) * mix(1.0, sh, 0.85);
        vec3 band;
        if (lighting_mode == lighting_mode_Flat) {
            band = vec3(mix(1.0, sh, shadow_amt * 0.5));
        } else if (lighting_mode == lighting_mode_Wrapped) {
            float xw = clamp((ndl + wrap_amount) / (1.0 + wrap_amount), 0.0, 1.0) * mix(1.0, sh, 0.85);
            band = mix(mix(vec3(1.0), shadow_color, shadow_amt), vec3(1.0), xw * xw * (3.0 - 2.0 * xw));
        } else if (lighting_mode == lighting_mode_Realistic) {
            vec3 toon = ct_toon(x, shadow_amt);
            band = mix(vec3(lambert) * (1.0 - metal), toon, toon_blend);
        } else {
            band = ct_toon(x, shadow_amt);
        }
        direct += rad * band;
        lit_sum += dot(rad, vec3(0.333)) * lambert;

        if (SPECULAR) {
            vec3 H = normalize(L + V);
            float ndh = max(dot(N, H), 0.0);
            vec3 s;
            if (specular_mode == specular_mode_Toon) {
                float p = pow(ndh, mix(256.0, 8.0, spec_size));
                s = vec3(smoothstep(0.5 - spec_smooth, 0.5 + spec_smooth + 1e-4, p));
            } else if (specular_mode == specular_mode_Anisotropic) {
                vec3 dir = aniso_direction == aniso_direction_Tangent ? T : B;
                float noise = (texture(aniso_shift, aniso_shift_uv(uv0)).r - 0.5) * aniso_noise;
                vec3 t1 = normalize(dir + N * (aniso_shift_amount + noise));
                vec3 t2 = normalize(dir + N * (aniso2_shift + noise));
                float th1 = dot(t1, H), th2 = dot(t2, H);
                float k1 = pow(sqrt(max(1.0 - th1 * th1, 0.0)), aniso_exponent);
                float k2 = pow(sqrt(max(1.0 - th2 * th2, 0.0)), aniso2_exponent);
                float dir_atten = smoothstep(-1.0, 0.0, th1);
                s = (vec3(k1) + aniso2_color * k2 * 0.5) * dir_atten;
            } else {
                float a = rough * rough;
                float a2 = a * a;
                float d = ndh * ndh * (a2 - 1.0) + 1.0;
                float D = a2 / (ZESTENGINE_PI * d * d);
                float vis = 0.5 / max(mix(2.0 * ndl * ndv, ndl + ndv, a), 1e-4);
                vec3 F = f0 + (1.0 - f0) * pow(1.0 - max(dot(V, H), 0.0), 5.0);
                s = D * vis * F * ZESTENGINE_PI;
            }
            spec += s * rad * max(ndl, 0.0) * sh;
        }
    }
    direct /= ZESTENGINE_PI;
    spec *= spec_color * spec_strength * spec_m / ZESTENGINE_PI;
    direct = min(direct, vec3(light_max));
    direct = mix(direct, vec3(dot(direct, vec3(0.2126, 0.7152, 0.0722))), monochrome_light);

    float ao = mix(1.0, orm.r, occlusion_strength);
    vec3 indirect = zestengine_baked_gi(v_world_pos, N) * indirect_strength * ao;
    indirect = mix(indirect, vec3(dot(indirect, vec3(0.2126, 0.7152, 0.0722))), monochrome_light);

    vec3 diffuse_albedo = lighting_mode == lighting_mode_Realistic ? albedo.rgb * (1.0 - metal * (1.0 - toon_blend)) : albedo.rgb;
    vec3 color = diffuse_albedo * (direct + indirect);
    color = max(color, albedo.rgb * light_min);
    color += spec;
    float light_lum = clamp(dot(direct + indirect, vec3(0.333)), 0.0, 1.0);

    if (REFLECTIONS) {
        vec3 R = reflect(-V, N);
        float r = lighting_mode == lighting_mode_Realistic ? rough : reflection_roughness;
        vec3 F = f0 + (1.0 - f0) * pow(1.0 - ndv, 5.0) * reflection_fresnel;
        color += zestengine_env(v_world_pos, R, r) * F * reflection_strength * ao;
    }

    if (RIM) {
        float edge = 1.0 - ndv;
        float w = max(rim_width, 1e-3);
        float soft = max(1.0 - rim_sharpness, 0.002) * w;
        float r = smoothstep(1.0 - w, 1.0 - w + soft, edge);
        r *= mix(1.0, clamp(lit_sum, 0.0, 1.0), rim_light_based);
        r *= (rim_strength + (AUDIO ? audio_rim * ct_audio(audio_rim_band) : 0.0)) * texture(rim_mask, rim_mask_uv(uv0)).r;
        color = ct_blend(color, rim_color, min(r, rim_blend == rim_blend_Add ? 1e4 : 1.0), rim_blend);
    }

    if (MATCAP) {
        vec3 mc = texture(matcap_tex, ct_matcap_uv(N)).rgb * mix(1.0, light_lum, matcap_lit);
        float t = matcap_strength * texture(matcap_mask, matcap_mask_uv(uv0)).r;
        color = ct_blend(color, mc, t, matcap_blend);
    }
    if (MATCAP2) {
        vec3 mc = texture(matcap2_tex, ct_matcap_uv(N)).rgb * mix(1.0, light_lum, matcap2_lit);
        float t = matcap2_strength * texture(matcap2_mask, matcap2_mask_uv(uv0)).r;
        color = ct_blend(color, mc, t, matcap2_blend);
    }

    vec3 emission = vec3(0.0);
    if (EMISSION) {
        vec2 euv = uv0 + emission_scroll * u_time;
        float pulse = mix(1.0, 0.5 + 0.5 * sin(u_time * emission_pulse_speed), emission_pulse_amount * step(1e-4, emission_pulse_speed));
        vec3 ec = ct_hue_shift(emission_color, u_time * emission_hue_speed);
        float mask = texture(emission_mask, emission_mask_uv(uv0)).r;
        if (VERTEX_COLOR && vc_mode == vc_mode_EmissionMask) mask *= mix(1.0, dot(v_color.rgb, vec3(0.333)), vc_strength);
        emission += texture(t_emission, euv).rgb * ec * emission_strength * pulse * mask;
    }
    if (EMISSION2) {
        vec2 euv = emission2_tex_uv(uv0) + emission2_scroll * u_time;
        float pulse = mix(1.0, 0.5 + 0.5 * sin(u_time * emission2_pulse_speed), emission2_pulse_amount);
        vec3 ec = ct_hue_shift(emission2_color, u_time * emission2_hue_speed);
        vec4 e = texture(emission2_tex, euv);
        emission += e.rgb * e.a * ec * emission2_strength * pulse;
    }
    if (DECAL && decal_emission > 0.0) {
        float r = radians(decal_rotation);
        vec2 p = mat2(cos(r), -sin(r), sin(r), cos(r)) * (uv0 - decal_position);
        vec2 duv = p / max(decal_scale, vec2(1e-4)) + 0.5;
        if (all(greaterThanEqual(duv, vec2(0.0))) && all(lessThanEqual(duv, vec2(1.0)))) {
            vec4 dc = texture(decal_tex, duv) * decal_color;
            emission += dc.rgb * dc.a * decal_emission;
        }
    }

    if (GLITTER) {
        vec2 g = uv0 * glitter_density;
        vec2 cell = floor(g);
        float h = ct_hash(cell);
        vec2 jitter = vec2(ct_hash(cell + 17.0), ct_hash(cell + 41.0)) - 0.5;
        float d = length(fract(g) - 0.5 - jitter * 0.5);
        float shape = 1.0 - smoothstep(glitter_size * 0.35, glitter_size * 0.5, d);
        vec3 rnd = normalize(vec3(ct_hash(cell + 3.0), ct_hash(cell + 7.0), ct_hash(cell + 11.0)) - 0.5);
        float facing = dot(reflect(-V, N), rnd) * 0.5 + 0.5;
        float phase = u_time * glitter_speed + h * 6.28318 + facing * glitter_view * 12.0;
        float sparkle = pow(0.5 + 0.5 * sin(phase), 12.0) * shape;
        vec3 gc = mix(glitter_color, glitter_color * (0.5 + 0.5 * cos(6.28318 * (h + vec3(0.0, 0.33, 0.67)))), glitter_rainbow);
        float mask = texture(glitter_mask, glitter_mask_uv(uv0)).r;
        emission += gc * sparkle * mask * mix(1.0, light_lum, glitter_lit);
    }

    if (DISSOLVE && dissolve_amount > 0.0) {
        float n = texture(dissolve_noise, dissolve_noise_uv(uv0)).r;
        if (dissolve_invert) n = 1.0 - n;
        float edge = 1.0 - smoothstep(dissolve_amount, dissolve_amount + dissolve_edge_width + 1e-4, n);
        emission += dissolve_edge_color * edge;
    }

    emission += flip_glow;
    if (AUDIO && audio_emission > 0.0) {
        float g = audio_emission * ct_audio(audio_emission_band);
        emission += (EMISSION ? emission_color : albedo.rgb) * g;
    }
    if (PANOSPHERE && pano_unlit) {
        // Unlit panosphere: the sky replaces the lighting where it shows.
        color = mix(color, pano_col, pano_amount);
    }

    color += emission;
    if (ALPHA && alpha_to_emission > 0.0) {
        alpha = max(alpha, clamp(dot(emission, vec3(0.333)), 0.0, 1.0) * alpha_to_emission);
    }

    color = zestengine_postfx(color, gl_FragCoord.xy);
    if (GRAB) {
        // Background through the surface: normal-driven offset in pixels,
        // optional blur and per-channel split. y flips (frag y is downward).
        vec3 vn = normalize(mat3(frame.view) * N);
        vec2 off = vec2(vn.x, -vn.y) * grab_distortion * 100.0;
        vec3 g;
        if (grab_chromatic > 0.0) {
            vec2 c = off * grab_chromatic * 0.5;
            g = vec3(zestengine_grab_blur(gl_FragCoord.xy, off + c, grab_blur, color).r,
                     zestengine_grab_blur(gl_FragCoord.xy, off, grab_blur, color).g,
                     zestengine_grab_blur(gl_FragCoord.xy, off - c, grab_blur, color).b);
        } else {
            g = zestengine_grab_blur(gl_FragCoord.xy, off, grab_blur, color);
        }
        // The grabbed colour already went through the scene's post chain: it is
        // mixed in after postfx, and emission keeps showing over it.
        color = mix(color, g * grab_tint, grab_amount * (1.0 - clamp(dot(emission, vec3(0.333)), 0.0, 1.0)));
    }

    // Multiply blending (dst * src) has no alpha term: fade towards white.
    if (zestengine_alpha_mode == 2 && zestengine_blend_mode == 2) color = mix(vec3(1.0), color, alpha);
    o_color = vec4(color, zestengine_alpha_mode == 2 ? alpha : 1.0);
}
