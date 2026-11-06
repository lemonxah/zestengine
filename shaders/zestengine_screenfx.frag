// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

// @shader "ZestEngine Screen FX"
//
// A screen-space effect volume: draw a cube or sphere mesh with this material
// and every pixel it covers is re-sampled from the opaque scene behind it with
// filters, shake, distortion, blur, pixelation, chromatic split and noise, only
// where the scene point lies inside the volume's shape. Material setup:
// Transparency Transparent (the soft volume edge is the alpha), Cull Front
// Faces on (so the camera can stand inside the volume), and a render queue of
// 3000 or more. It reads the opaque scene colour, so the renderer copies it
// before the draw (Scene Colour Read follows the GRAB toggle below).
//
// @preset "Colour grade" FILTER=1 SHAKE=0 DISTORT=0 BLUR=0 PIXELATE=0 CHROMATIC=0 NOISE=0
// @preset "Underwater" FILTER=1 SHAKE=0 DISTORT=1 BLUR=1 PIXELATE=0 CHROMATIC=0 NOISE=0
// @preset "Glitch" FILTER=0 SHAKE=1 DISTORT=0 BLUR=0 PIXELATE=1 CHROMATIC=1 NOISE=1
//
// @property GRAB "Scene Colour Read" Toggle = 1 @keyword @tooltip("Leave on: the volume samples the opaque scene")
// @property volume_shape "Volume" Enum(Everywhere, Box, Sphere) = Box @keyword @tooltip("Box and Sphere fit the mesh's unit cube / unit sphere (diameter 1 in object space), scaled by its transform")
// @property volume_falloff "Edge Falloff" Range(0, 1) = 0.15 @show_if(volume_shape != Everywhere)
// @property fx_amount "Amount" Range(0, 1) = 1
//
// @feature FILTER "Filtering"
// @property filt_hue "Hue Shift" Range(0, 1) = 0
// @property filt_saturation "Saturation" Range(0, 2) = 1
// @property filt_contrast "Contrast" Range(0, 2) = 1
// @property filt_brightness "Brightness" Range(0, 2) = 1
// @property filt_tint "Tint" Color3 = (1, 1, 1)
// @property filt_invert "Invert" Range(0, 1) = 0
// @end
//
// @feature SHAKE "Shake"
// @property shake_amount "Amount (px)" Range(0, 64) = 8
// @property shake_speed "Speed" Range(0, 60) = 25
// @end
//
// @feature DISTORT "Distortion"
// @property distort_amount "Amount (px)" Range(0, 100) = 14
// @property distort_scale "Scale" Range(0.5, 40) = 6
// @property distort_speed "Speed" Range(0, 10) = 1
// @end
//
// @feature BLUR "Blur"
// @property blur_radius "Radius (px)" Range(0, 32) = 6
// @end
//
// @feature PIXELATE "Pixelate"
// @property pixel_size "Block Size (px)" Range(1, 64) = 8
// @end
//
// @feature CHROMATIC "Chromatic Split"
// @property ca_amount "Amount (px)" Range(0, 32) = 4
// @end
//
// @feature NOISE "Noise"
// @property noise_amount "Amount" Range(0, 1) = 0.12
// @property noise_speed "Speed" Range(0, 60) = 24
// @property noise_colour "Coloured" Toggle = 0
// @end

float sfx_hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float sfx_noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(sfx_hash(i), sfx_hash(i + vec2(1, 0)), f.x), mix(sfx_hash(i + vec2(0, 1)), sfx_hash(i + vec2(1, 1)), f.x), f.y);
}

vec3 sfx_hue(vec3 c, float turns) {
    float a = turns * 6.28318530718;
    const vec3 k = vec3(0.57735);
    float ca = cos(a);
    return c * ca + cross(k, c) * sin(a) + k * dot(k, c) * (1.0 - ca);
}

void main() {
    vec2 frag = gl_FragCoord.xy;

    // Volume mask: where the scene point behind this pixel lies in the shape.
    float mask = 1.0;
    if (volume_shape != volume_shape_Everywhere) {
        vec3 world;
        if (zestengine_grab_world_pos(frag, world)) {
            mat4 model = mat4(vec4(pc.model[0].xyz, 0.0), pc.model[1], pc.model[2], pc.model[3]);
            vec3 local = (inverse(model) * vec4(world, 1.0)).xyz;
            float d = volume_shape == volume_shape_Box ? max(abs(local.x), max(abs(local.y), abs(local.z))) : length(local);
            float edge = max(volume_falloff * 0.5, 1e-3);
            mask = 1.0 - smoothstep(0.5 - edge, 0.5, d);
        }
        // Sky behind the volume counts as inside.
    }
    if (mask <= 0.0) discard;

    vec2 off = vec2(0.0);
    if (SHAKE) {
        float t = floor(u_time * shake_speed);
        off += (vec2(sfx_hash(vec2(t, 1.3)), sfx_hash(vec2(t, 7.9))) - 0.5) * 2.0 * shake_amount;
    }
    if (DISTORT) {
        vec2 p = frag / max(frame.postfx3.w, 1.0) * distort_scale + u_time * distort_speed;
        off += (vec2(sfx_noise(p), sfx_noise(p + 17.3)) - 0.5) * 2.0 * distort_amount;
    }
    vec2 pos = frag;
    if (PIXELATE) pos = (floor(frag / pixel_size) + 0.5) * pixel_size;

    vec3 base = vec3(0.0);
    float blur = BLUR ? blur_radius : 0.0;
    vec3 c;
    if (CHROMATIC) {
        vec2 ca = vec2(ca_amount, 0.0);
        c = vec3(zestengine_grab_blur(pos, off + ca, blur, base).r, zestengine_grab_blur(pos, off, blur, base).g, zestengine_grab_blur(pos, off - ca, blur, base).b);
    } else {
        c = zestengine_grab_blur(pos, off, blur, base);
    }

    if (FILTER) {
        c = sfx_hue(c, filt_hue);
        float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
        c = mix(vec3(l), c, filt_saturation);
        c = (c - 0.18) * filt_contrast + 0.18;
        c *= filt_brightness * filt_tint;
        c = mix(c, vec3(1.0) - c, filt_invert);
    }
    if (NOISE) {
        float t = floor(u_time * noise_speed);
        vec3 n = noise_colour ? vec3(sfx_hash(frag + t), sfx_hash(frag + t + 11.0), sfx_hash(frag + t + 23.0)) : vec3(sfx_hash(frag + t));
        c += (n - 0.5) * noise_amount;
    }
    // ZTest Always: the volume's own faces may lie behind scene geometry that is
    // inside it; the mask above already handles objects in front of the volume.
    gl_FragDepth = 0.0;
    o_color = vec4(max(c, vec3(0.0)), mask * fx_amount);
}
