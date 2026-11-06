// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

#version 450
// Post-processing fullscreen pass: samples the linear-HDR scene-color target and
// applies the camera's blended Volume profile. Chromatic aberration and bloom
// (which need to read the rendered frame), then exposure, color grading,
// tonemap, and vignette. Writes display-space color to the swapchain.

layout(set = 0, binding = 0) uniform sampler2D u_hdr;
layout(set = 1, binding = 0) uniform sampler2D u_bloom;
layout(set = 2, binding = 0, std430) readonly buffer Exposure { uint histogram[64]; vec4 state; } exposure;

layout(push_constant) uniform Post {
    vec4 p0; // x tonemap mode, y exposure EV, z grade exposure, w contrast
    vec4 p1; // x saturation, y temperature, z tint, w grading enabled
    vec4 p2; // x vignette enabled, y intensity, z smoothness, w (unused)
    vec4 p3; // xyz vignette color, w (unused)
    vec4 p4; // x bloom enabled, y threshold, z intensity, w radius
    vec4 p5; // x CA enabled, y CA intensity, z underwater wobble (0 = off), w time
    vec4 p6; // xyz waterline: near-plane height above the surface = x*u + y*v + z, w underwater blur (px at 1080p)
    vec4 p7; // x FXAA enabled, y sharpen amount (CAS-like, 0 = off), z auto exposure enabled
} pc;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

float exposure_scale() {
    return exp2(pc.p0.y+(pc.p7.z>0.5 ? exposure.state.x : 0.0));
}

vec3 aces(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}


// ---- FXAA 3.11 (quality), run on a cheap-tonemapped luma of the HDR source ----
// Returns the sub-pixel UV offset position the rest of the stack samples at
// (the source is linear HDR: exposure + Reinhard l/(1+l) bring it to [0, 1],
// sqrt approximates the perceptual curve FXAA's thresholds were tuned for).
float fxaa_luma(vec2 p, float ev) {
    vec3 c = texture(u_hdr, p).rgb * ev;
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    return sqrt(l / (1.0 + l));
}

vec2 fxaa_uv(vec2 uv) {
    const float EDGE_THRESHOLD = 0.125, EDGE_THRESHOLD_MIN = 0.0312, SUBPIX = 0.75;
    const int STEPS = 12;
    const float STEP[12] = float[](1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 4.0, 8.0);
    vec2 rcp = 1.0 / vec2(textureSize(u_hdr, 0));
    float ev = exposure_scale();
    float lM = fxaa_luma(uv, ev);
    float lN = fxaa_luma(uv + vec2(0.0, -rcp.y), ev);
    float lS = fxaa_luma(uv + vec2(0.0, rcp.y), ev);
    float lW = fxaa_luma(uv + vec2(-rcp.x, 0.0), ev);
    float lE = fxaa_luma(uv + vec2(rcp.x, 0.0), ev);
    float rangeMax = max(max(lN, lS), max(lE, max(lW, lM)));
    float rangeMin = min(min(lN, lS), min(lE, min(lW, lM)));
    float range = rangeMax - rangeMin;
    if (range < max(EDGE_THRESHOLD_MIN, rangeMax * EDGE_THRESHOLD)) {
        return uv;
    }
    float lNW = fxaa_luma(uv + vec2(-rcp.x, -rcp.y), ev);
    float lNE = fxaa_luma(uv + vec2(rcp.x, -rcp.y), ev);
    float lSW = fxaa_luma(uv + vec2(-rcp.x, rcp.y), ev);
    float lSE = fxaa_luma(uv + vec2(rcp.x, rcp.y), ev);
    float lNS = lN + lS, lWE = lW + lE;
    float edgeH = abs(-2.0 * lW + (lNW + lSW)) + abs(-2.0 * lM + lNS) * 2.0 + abs(-2.0 * lE + (lNE + lSE));
    float edgeV = abs(-2.0 * lN + (lNW + lNE)) + abs(-2.0 * lM + lWE) * 2.0 + abs(-2.0 * lS + (lSW + lSE));
    bool horz = edgeH >= edgeV;
    float lengthSign = horz ? rcp.y : rcp.x;
    float lumaN = horz ? lN : lW;
    float lumaS = horz ? lS : lE;
    float gradN = lumaN - lM, gradS = lumaS - lM;
    float lumaNN = lumaN + lM, lumaSS = lumaS + lM;
    bool pairN = abs(gradN) >= abs(gradS);
    float gradient = max(abs(gradN), abs(gradS));
    if (pairN) lengthSign = -lengthSign;
    // Sub-pixel (single pixel feature) blend amount.
    float subA = (lNS + lWE) * 2.0 + (lNW + lSW + lNE + lSE);
    float subC = clamp(abs(subA * (1.0 / 12.0) - lM) / range, 0.0, 1.0);
    float subF = (-2.0 * subC + 3.0) * subC * subC;
    float subH = subF * subF * SUBPIX;
    // Walk along the edge in both directions to find its ends.
    vec2 posB = uv;
    vec2 offNP = horz ? vec2(rcp.x, 0.0) : vec2(0.0, rcp.y);
    if (horz) posB.y += lengthSign * 0.5; else posB.x += lengthSign * 0.5;
    vec2 posN = posB - offNP, posP = posB + offNP;
    float endN = fxaa_luma(posN, ev), endP = fxaa_luma(posP, ev);
    if (!pairN) lumaNN = lumaSS;
    float gradScaled = gradient * 0.25;
    float lumaMM = lM - lumaNN * 0.5;
    bool mLTzero = lumaMM < 0.0;
    endN -= lumaNN * 0.5;
    endP -= lumaNN * 0.5;
    bool doneN = abs(endN) >= gradScaled, doneP = abs(endP) >= gradScaled;
    if (!doneN) posN -= offNP * STEP[0];
    if (!doneP) posP += offNP * STEP[0];
    for (int i = 1; i < STEPS && !(doneN && doneP); ++i) {
        if (!doneN) endN = fxaa_luma(posN, ev) - lumaNN * 0.5;
        if (!doneP) endP = fxaa_luma(posP, ev) - lumaNN * 0.5;
        doneN = abs(endN) >= gradScaled;
        doneP = abs(endP) >= gradScaled;
        if (!doneN) posN -= offNP * STEP[i];
        if (!doneP) posP += offNP * STEP[i];
    }
    float dstN = horz ? uv.x - posN.x : uv.y - posN.y;
    float dstP = horz ? posP.x - uv.x : posP.y - uv.y;
    bool goodN = (endN < 0.0) != mLTzero;
    bool goodP = (endP < 0.0) != mLTzero;
    float spanLength = dstP + dstN;
    bool nearN = dstN < dstP;
    float dst = min(dstN, dstP);
    bool goodSpan = nearN ? goodN : goodP;
    float pixelOffset = dst * (-1.0 / spanLength) + 0.5;
    float pixelOffsetSub = max(goodSpan ? pixelOffset : 0.0, subH);
    if (horz) uv.y += pixelOffsetSub * lengthSign; else uv.x += pixelOffsetSub * lengthSign;
    return uv;
}

// CAS-like sharpen: unsharp mask on the tone-mapped colour, limited to the
// neighbourhood min/max so it never rings or overshoots.
vec3 sharpen(vec3 color, vec2 uv, float amount) {
    vec2 px = 1.0 / vec2(textureSize(u_hdr, 0));
    vec3 n = texture(u_hdr, uv + vec2(0.0, -px.y)).rgb;
    vec3 s = texture(u_hdr, uv + vec2(0.0, px.y)).rgb;
    vec3 w = texture(u_hdr, uv + vec2(-px.x, 0.0)).rgb;
    vec3 e = texture(u_hdr, uv + vec2(px.x, 0.0)).rgb;
    vec3 tc = color / (1.0 + color), tn = n / (1.0 + n), ts = s / (1.0 + s), tw = w / (1.0 + w), te = e / (1.0 + e);
    vec3 mn = min(min(tn, ts), min(min(tw, te), tc));
    vec3 mx = max(max(tn, ts), max(max(tw, te), tc));
    vec3 blur = (tn + ts + tw + te) * 0.25;
    vec3 t = clamp(tc + (tc - blur) * (amount * 1.5), mn, mx);
    t = min(t, vec3(0.999));
    return t / (1.0 - t);
}

void main() {
    vec2 uv = v_uv;

    // Under water: a slow wobble and a soft blur, only below the waterline.
    float under = 0.0;
    if (pc.p5.z > 0.0 || pc.p6.w > 0.0) {
        float above = pc.p6.x * uv.x + pc.p6.y * uv.y + pc.p6.z;
        under = 1.0 - smoothstep(-0.0015, 0.0015, above);
        float t = pc.p5.w;
        vec2 w = vec2(sin(uv.y * 21.0 + t * 1.7) + 0.5 * sin(uv.y * 47.0 - t * 2.3 + uv.x * 9.0),
                      cos(uv.x * 17.0 + t * 1.3) + 0.5 * cos(uv.x * 39.0 + t * 2.1 - uv.y * 7.0));
        uv = clamp(uv + w * (0.0016 * pc.p5.z * under), vec2(0.0), vec2(1.0));
    }

    // FXAA: everything below samples the scene at the anti-aliased position.
    if (pc.p7.x > 0.5) {
        uv = clamp(fxaa_uv(uv), vec2(0.0), vec2(1.0));
    }

    // Chromatic aberration: split the channels along the screen radius.
    vec3 color;
    if (pc.p5.x > 0.5) {
        vec2 dir = (uv - 0.5) * (pc.p5.y * 0.02);
        color.r = texture(u_hdr, uv + dir).r;
        color.g = texture(u_hdr, uv).g;
        color.b = texture(u_hdr, uv - dir).b;
    } else {
        color = texture(u_hdr, uv).rgb;
    }
    if (pc.p7.y > 0.0) {
        color = sharpen(color, uv, pc.p7.y);
    }
    if (under > 0.0 && pc.p6.w > 0.0) {
        // Golden-angle disk of taps, radius scaled to the target height.
        vec2 px = pc.p6.w * under / vec2(textureSize(u_hdr, 0).y) * vec2(float(textureSize(u_hdr, 0).y) / float(textureSize(u_hdr, 0).x), 1.0) * (float(textureSize(u_hdr, 0).y) / 1080.0);
        vec3 sum = color;
        for (int k = 1; k < 12; ++k) {
            float r = sqrt(float(k) / 11.0);
            float a = float(k) * 2.39996;
            sum += texture(u_hdr, clamp(uv + vec2(cos(a), sin(a)) * r * px, vec2(0.0), vec2(1.0))).rgb;
        }
        color = sum / 12.0;
    }

    // Exposure (EV), before bloom (UE order: exposure -> bloom -> tonemap); the
    // bloom prefilter applies the same exposure before its threshold.
    float ev = exposure_scale();
    color *= ev;

    // Bloom: level 0 of the mip-chain bloom (bloom.comp), already exposed and
    // thresholded; the chain spreads it wide with no ring or tap artifacts.
    if (pc.p4.x > 0.5) {
        color += texture(u_bloom, uv).rgb * pc.p4.z;
    }

    // Color grading (linear).
    if (pc.p1.w > 0.5) {
        color *= exp2(pc.p0.z);                       // post exposure
        float temp = pc.p1.y, tint = pc.p1.z;         // white balance
        color.r *= 1.0 + temp * 0.2 + tint * 0.1;
        color.g *= 1.0 - tint * 0.2;
        color.b *= 1.0 - temp * 0.2 + tint * 0.1;
        color = (color - 0.18) * pc.p0.w + 0.18;      // contrast @ mid-grey
        float l = dot(color, vec3(0.2126, 0.7152, 0.0722));
        color = mix(vec3(l), color, pc.p1.x);         // saturation
        color = max(color, vec3(0.0));
    }

    // Tonemap.
    int mode = int(pc.p0.x + 0.5);
    if (mode == 1) {
        color = color / (color + vec3(1.0));   // Reinhard
    } else if (mode == 2) {
        color = aces(color);                   // ACES
    }                                          // 0 = none

    // Vignette (display space).
    if (pc.p2.x > 0.5) {
        float dist = length(uv - 0.5) * 1.41421356;
        float sm = max(pc.p2.z, 1e-3);
        float mask = clamp((dist - (1.0 - sm)) / sm, 0.0, 1.0) * pc.p2.y;
        color = mix(color, pc.p3.xyz, mask);
    }

    o_color = vec4(color, 1.0);
}
