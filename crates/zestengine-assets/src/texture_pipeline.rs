//! Pixel operations of the texture import pipeline, on tightly packed RGBA8.
//! Order used by `tex_cache`: alpha source, alpha dilation, resize, height to
//! normal, flip green, channel select, then the mip chain (linear-space
//! filtering, per-mip normal renormalisation, coverage preservation).
//! Everything here is deterministic and CPU-only so it can be unit tested.

use image::{ImageBuffer, Rgba, imageops};

use crate::texture_import::{Channel, MipFilter, NormalFilter, ResizeFilter};

pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

pub fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

fn srgb_lut() -> [f32; 256] {
    std::array::from_fn(|v| srgb_to_linear(v as f32 / 255.0))
}

/// Rec. 709 luminance of stored (non-linearised) channels.
pub fn luma(r: u8, g: u8, b: u8) -> u8 {
    (0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32 + 0.5) as u8
}

/// True when any texel is not fully opaque.
pub fn has_alpha(px: &[u8]) -> bool {
    px.as_chunks::<4>().0.iter().any(|p| p[3] != 255)
}

/// Alpha = luminance of the colour.
pub fn gray_to_alpha(px: &mut [u8]) {
    for p in px.as_chunks_mut::<4>().0 {
        p[3] = luma(p[0], p[1], p[2]);
    }
}

/// Make the image opaque.
pub fn drop_alpha(px: &mut [u8]) {
    for p in px.as_chunks_mut::<4>().0 {
        p[3] = 255;
    }
}

/// Keep one source channel as a grey, opaque image.
pub fn select_channel(px: &mut [u8], ch: Channel) {
    for p in px.as_chunks_mut::<4>().0 {
        let v = p[ch.index()];
        *p = [v, v, v, 255];
    }
}

/// Copy the colour of the nearest visible texel (alpha > 0) into every fully
/// transparent texel, leaving alpha at 0, so filtering and mips don't pull
/// dark fringes in at edges. Jump flooding: O(n log n) however far the
/// nearest visible texel is.
pub fn alpha_dilate(px: &mut [u8], w: u32, h: u32) {
    let (wi, hi) = (w as usize, h as usize);
    let n = wi * hi;
    if n == 0 || px.len() != n * 4 {
        return;
    }
    const NONE: u32 = u32::MAX;
    let mut seed: Vec<u32> = (0..n).map(|i| if px[i * 4 + 3] > 0 { i as u32 } else { NONE }).collect();
    let seeds = seed.iter().filter(|&&s| s != NONE).count();
    if seeds == 0 || seeds == n {
        return;
    }
    let dist2 = |a: usize, b: u32| -> u64 {
        let (ax, ay) = ((a % wi) as i64, (a / wi) as i64);
        let (bx, by) = ((b as usize % wi) as i64, (b as usize / wi) as i64);
        ((ax - bx).pow(2) + (ay - by).pow(2)) as u64
    };
    let mut step = (wi.max(hi).next_power_of_two() / 2).max(1) as i64;
    loop {
        let mut next = seed.clone();
        for y in 0..hi as i64 {
            for x in 0..wi as i64 {
                let i = (y as usize) * wi + x as usize;
                let mut best = seed[i];
                let mut best_d = if best == NONE { u64::MAX } else { dist2(i, best) };
                for dy in [-step, 0, step] {
                    for dx in [-step, 0, step] {
                        let (qx, qy) = (x + dx, y + dy);
                        if qx < 0 || qy < 0 || qx >= wi as i64 || qy >= hi as i64 {
                            continue;
                        }
                        let s = seed[(qy as usize) * wi + qx as usize];
                        if s == NONE {
                            continue;
                        }
                        let d = dist2(i, s);
                        if d < best_d {
                            best_d = d;
                            best = s;
                        }
                    }
                }
                next[i] = best;
            }
        }
        seed = next;
        if step == 1 {
            break;
        }
        step /= 2;
    }
    for i in 0..n {
        if px[i * 4 + 3] == 0 && seed[i] != NONE {
            let s = seed[i] as usize * 4;
            let (r, g, b) = (px[s], px[s + 1], px[s + 2]);
            px[i * 4..i * 4 + 3].copy_from_slice(&[r, g, b]);
        }
    }
}

/// Resize in linear light: sRGB colour is decoded, filtered as floats and
/// re-encoded (alpha and linear data filter as they are). Bilinear uses a
/// triangle kernel, Mitchell a Catmull-Rom cubic.
pub fn resize(px: &[u8], w: u32, h: u32, nw: u32, nh: u32, srgb: bool, filter: ResizeFilter) -> Vec<u8> {
    if (w, h) == (nw, nh) || w == 0 || h == 0 {
        return px.to_vec();
    }
    let lut = srgb_lut();
    let data: Vec<f32> = px
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| {
            [
                if srgb { lut[p[0] as usize] } else { p[0] as f32 / 255.0 },
                if srgb { lut[p[1] as usize] } else { p[1] as f32 / 255.0 },
                if srgb { lut[p[2] as usize] } else { p[2] as f32 / 255.0 },
                p[3] as f32 / 255.0,
            ]
        })
        .collect();
    let Some(img) = ImageBuffer::<Rgba<f32>, Vec<f32>>::from_raw(w, h, data) else { return px.to_vec() };
    let ft = match filter {
        ResizeFilter::Mitchell => imageops::FilterType::CatmullRom,
        ResizeFilter::Bilinear => imageops::FilterType::Triangle,
    };
    let out = imageops::resize(&img, nw, nh, ft);
    let enc = |v: f32, color: bool| -> u8 {
        let v = v.clamp(0.0, 1.0);
        ((if color && srgb { linear_to_srgb(v) } else { v }) * 255.0 + 0.5) as u8
    };
    out.as_raw().as_chunks::<4>().0.iter().flat_map(|p| [enc(p[0], true), enc(p[1], true), enc(p[2], true), enc(p[3], false)]).collect()
}

/// Derive a tangent-space normal map from a height map (luminance).
/// `slope = height difference per texel * bumpiness * 8`; Smooth uses a Sobel
/// kernel, Sharp central differences. Green points up (OpenGL).
pub fn height_to_normal(px: &mut [u8], w: u32, h: u32, bumpiness: f32, filter: NormalFilter) {
    let (wi, hi) = (w as i64, h as i64);
    if wi == 0 || hi == 0 || px.len() != (w * h * 4) as usize {
        return;
    }
    let height: Vec<f32> = px.as_chunks::<4>().0.iter().map(|p| luma(p[0], p[1], p[2]) as f32 / 255.0).collect();
    let at = |x: i64, y: i64| height[(y.clamp(0, hi - 1) * wi + x.clamp(0, wi - 1)) as usize];
    let k = bumpiness * 8.0;
    for y in 0..hi {
        for x in 0..wi {
            let (dx, dy) = match filter {
                NormalFilter::Sharp => ((at(x + 1, y) - at(x - 1, y)) * 0.5, (at(x, y + 1) - at(x, y - 1)) * 0.5),
                NormalFilter::Smooth => (
                    ((at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2.0 * at(x - 1, y) + at(x - 1, y + 1))) / 8.0,
                    ((at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2.0 * at(x, y - 1) + at(x + 1, y - 1))) / 8.0,
                ),
            };
            // Row index grows downward, so +Y (up) is the negative row slope.
            let n = glam_normalize([-dx * k, dy * k, 1.0]);
            let o = ((y * wi + x) * 4) as usize;
            for c in 0..3 {
                px[o + c] = ((n[c] * 0.5 + 0.5) * 255.0 + 0.5) as u8;
            }
            px[o + 3] = 255;
        }
    }
}

fn glam_normalize(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-8);
    [v[0] / l, v[1] / l, v[2] / l]
}

/// DirectX to OpenGL normal convention: invert green.
pub fn flip_green(px: &mut [u8]) {
    for p in px.as_chunks_mut::<4>().0 {
        p[1] = 255 - p[1];
    }
}

/// Renormalise every texel's (r, g, b) as a unit vector. A texel with no
/// blue (two-channel sources) gets Z rebuilt from X and Y first.
pub fn renormalize(px: &mut [u8]) {
    for p in px.as_chunks_mut::<4>().0 {
        let x = p[0] as f32 / 255.0 * 2.0 - 1.0;
        let y = p[1] as f32 / 255.0 * 2.0 - 1.0;
        let z = if p[2] == 0 { (1.0 - x * x - y * y).max(0.0).sqrt() } else { p[2] as f32 / 255.0 * 2.0 - 1.0 };
        let n = glam_normalize([x, y, z]);
        for c in 0..3 {
            p[c] = ((n[c] * 0.5 + 0.5) * 255.0 + 0.5) as u8;
        }
    }
}

/// How to build the mip chain.
#[derive(Clone, Copy, Debug)]
pub struct MipParams {
    /// Colour channels are sRGB (filter in linear light).
    pub srgb: bool,
    pub filter: MipFilter,
    /// Renormalise each level as a normal map.
    pub normal: bool,
    /// Preserve the fraction of texels with alpha at or above this cutoff (0..1).
    pub coverage_cutoff: Option<f32>,
    /// False = just the base level.
    pub enabled: bool,
}

impl Default for MipParams {
    fn default() -> Self {
        Self { srgb: false, filter: MipFilter::Box, normal: false, coverage_cutoff: None, enabled: true }
    }
}

/// Mip chain (largest first) down to 1x1: each level `max(dim / 2, 1)`.
pub fn mip_chain(pixels: &[u8], width: u32, height: u32, p: &MipParams) -> Vec<(Vec<u8>, u32, u32)> {
    let mut out = vec![(pixels.to_vec(), width, height)];
    if !p.enabled {
        return out;
    }
    let lut = srgb_lut();
    let base_cov = p.coverage_cutoff.map(|c| coverage(pixels, c));
    let (mut w, mut h) = (width.max(1), height.max(1));
    while w > 1 || h > 1 {
        let prev = &out.last().unwrap().0;
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        let mut down = match p.filter {
            MipFilter::Box => box_down(prev, w, h, nw, nh, p.srgb, &lut),
            MipFilter::Kaiser => kaiser_down(prev, w, h, nw, nh, p.srgb, &lut),
        };
        if p.normal {
            renormalize(&mut down);
        }
        if let (Some(cutoff), Some(target)) = (p.coverage_cutoff, base_cov) {
            rescale_alpha_to_coverage(&mut down, cutoff, target);
        }
        (w, h) = (nw, nh);
        out.push((down, w, h));
    }
    out
}

fn box_down(prev: &[u8], w: u32, h: u32, nw: u32, nh: u32, srgb: bool, lut: &[f32; 256]) -> Vec<u8> {
    let mut down = vec![0u8; (nw * nh * 4) as usize];
    for y in 0..nh {
        for x in 0..nw {
            for c in 0..4 {
                let tap = |dx: u32, dy: u32| {
                    let (sx, sy) = ((x * 2 + dx).min(w - 1), (y * 2 + dy).min(h - 1));
                    prev[((sy * w + sx) * 4 + c) as usize]
                };
                let taps = [tap(0, 0), tap(1, 0), tap(0, 1), tap(1, 1)];
                down[((y * nw + x) * 4 + c) as usize] = if srgb && c < 3 {
                    let lin = taps.iter().map(|&t| lut[t as usize]).sum::<f32>() / 4.0;
                    (linear_to_srgb(lin) * 255.0 + 0.5) as u8
                } else {
                    ((taps.iter().map(|&t| t as u32).sum::<u32>() + 2) / 4) as u8
                };
            }
        }
    }
    down
}

fn bessel_i0(x: f32) -> f32 {
    let (mut sum, mut term) = (1.0f32, 1.0f32);
    for k in 1..24 {
        term *= (x / (2.0 * k as f32)).powi(2);
        sum += term;
    }
    sum
}

/// Eight Kaiser-windowed sinc weights for a 2:1 reduction; taps sit at
/// -3.5 .. +3.5 source texels from the destination texel's centre.
fn kaiser8() -> [f32; 8] {
    const BETA: f32 = 4.0;
    let mut w = [0.0f32; 8];
    for (i, w) in w.iter_mut().enumerate() {
        let t = i as f32 - 3.5;
        let sinc = |x: f32| if x.abs() < 1e-6 { 1.0 } else { (std::f32::consts::PI * x).sin() / (std::f32::consts::PI * x) };
        let u = t / 4.0;
        *w = sinc(t / 2.0) * bessel_i0(BETA * (1.0 - u * u).max(0.0).sqrt()) / bessel_i0(BETA);
    }
    let sum: f32 = w.iter().sum();
    w.map(|v| v / sum)
}

fn kaiser_down(prev: &[u8], w: u32, h: u32, nw: u32, nh: u32, srgb: bool, lut: &[f32; 256]) -> Vec<u8> {
    let k = kaiser8();
    let (wi, hi) = (w as i64, h as i64);
    // To linear floats.
    let src: Vec<f32> = prev
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| {
            [
                if srgb { lut[p[0] as usize] } else { p[0] as f32 / 255.0 },
                if srgb { lut[p[1] as usize] } else { p[1] as f32 / 255.0 },
                if srgb { lut[p[2] as usize] } else { p[2] as f32 / 255.0 },
                p[3] as f32 / 255.0,
            ]
        })
        .collect();
    // Horizontal: (w x h) -> (nw x h).
    let mut tmp = vec![0.0f32; (nw * h * 4) as usize];
    for y in 0..hi {
        for x in 0..nw as i64 {
            for c in 0..4 {
                let mut acc = 0.0;
                for (i, wt) in k.iter().enumerate() {
                    let sx = (2 * x - 3 + i as i64).clamp(0, wi - 1);
                    acc += wt * src[((y * wi + sx) * 4) as usize + c];
                }
                tmp[((y * nw as i64 + x) * 4) as usize + c] = acc;
            }
        }
    }
    // Vertical: (nw x h) -> (nw x nh).
    let mut out = vec![0u8; (nw * nh * 4) as usize];
    for y in 0..nh as i64 {
        for x in 0..nw as i64 {
            for c in 0..4 {
                let mut acc = 0.0;
                for (i, wt) in k.iter().enumerate() {
                    let sy = (2 * y - 3 + i as i64).clamp(0, hi - 1);
                    acc += wt * tmp[((sy * nw as i64 + x) * 4) as usize + c];
                }
                let v = acc.clamp(0.0, 1.0);
                out[((y * nw as i64 + x) * 4) as usize + c] = ((if srgb && c < 3 { linear_to_srgb(v) } else { v }) * 255.0 + 0.5) as u8;
            }
        }
    }
    out
}

/// Fraction of texels whose alpha is at or above `cutoff` (0..1).
pub fn coverage(px: &[u8], cutoff: f32) -> f32 {
    let thr = (cutoff.clamp(0.0, 1.0) * 255.0).round() as u8;
    let n = px.len() / 4;
    if n == 0 {
        return 0.0;
    }
    px.as_chunks::<4>().0.iter().filter(|p| p[3] >= thr).count() as f32 / n as f32
}

/// Scale alpha so the fraction of texels at or above `cutoff` matches
/// `target` (binary search on the scale), keeping alpha-tested foliage from
/// thinning out in the distance.
pub fn rescale_alpha_to_coverage(px: &mut [u8], cutoff: f32, target: f32) {
    if target <= 0.0 || px.len() < 4 {
        return;
    }
    let thr = (cutoff.clamp(0.0, 1.0) * 255.0).round().max(1.0);
    let mut hist = [0u32; 256];
    for p in px.as_chunks::<4>().0 {
        hist[p[3] as usize] += 1;
    }
    let n = (px.len() / 4) as f32;
    let cov = |s: f32| -> f32 { (0..256).filter(|&a| a as f32 * s >= thr).map(|a| hist[a]).sum::<u32>() as f32 / n };
    if (cov(1.0) - target).abs() < 0.5 / n {
        return;
    }
    let (mut lo, mut hi) = (0.0f32, 16.0f32);
    for _ in 0..24 {
        let mid = 0.5 * (lo + hi);
        if cov(mid) < target { lo = mid } else { hi = mid }
    }
    let s = hi;
    for p in px.as_chunks_mut::<4>().0 {
        p[3] = (p[3] as f32 * s + 0.5).min(255.0) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, c: [u8; 4]) -> Vec<u8> {
        (0..w * h).flat_map(|_| c).collect()
    }

    #[test]
    fn alpha_source_helpers() {
        let mut px = vec![255, 0, 0, 10, 0, 0, 255, 200];
        gray_to_alpha(&mut px);
        assert_eq!(px[3], luma(255, 0, 0));
        assert!(has_alpha(&px));
        drop_alpha(&mut px);
        assert!(!has_alpha(&px));
        let mut px = vec![10, 20, 30, 40];
        select_channel(&mut px, Channel::G);
        assert_eq!(px, [20, 20, 20, 255]);
    }

    #[test]
    fn dilation_spreads_colour_into_transparent_texels_only() {
        // 5x1: [red solid][clear][clear][clear][blue solid]
        let mut px = vec![0u8; 5 * 4];
        px[..4].copy_from_slice(&[255, 0, 0, 255]);
        px[16..20].copy_from_slice(&[0, 0, 255, 255]);
        alpha_dilate(&mut px, 5, 1);
        assert_eq!(&px[4..8], &[255, 0, 0, 0], "nearer to red");
        assert_eq!(&px[12..16], &[0, 0, 255, 0], "nearer to blue");
        assert!(px.chunks(4).all(|p| p[3] == 0 || p[3] == 255), "alpha untouched");
        assert_eq!(&px[..4], &[255, 0, 0, 255]);
    }

    #[test]
    fn dilation_reaches_far_texels_and_skips_degenerate_images() {
        let mut px = vec![0u8; 16 * 16 * 4];
        px[3] = 255;
        px[0] = 9;
        alpha_dilate(&mut px, 16, 16);
        assert_eq!(&px[(15 * 16 + 15) * 4..(15 * 16 + 15) * 4 + 3], &[9, 0, 0]);
        let mut clear = vec![0u8; 4 * 4 * 4];
        alpha_dilate(&mut clear, 4, 4);
        assert!(clear.iter().all(|&b| b == 0));
    }

    #[test]
    fn resize_in_linear_space_keeps_mid_grey_brightness() {
        // Black/white checker averaged in linear light is sRGB ~188, not 128.
        let mut px = Vec::new();
        for y in 0..8u32 {
            for x in 0..8u32 {
                let v = if (x + y) % 2 == 0 { 255 } else { 0 };
                px.extend_from_slice(&[v, v, v, 255]);
            }
        }
        let out = resize(&px, 8, 8, 1, 1, true, ResizeFilter::Bilinear);
        assert!((170..=205).contains(&out[0]), "{}", out[0]);
        assert_eq!(out[3], 255);
        let lin = resize(&px, 8, 8, 1, 1, false, ResizeFilter::Bilinear);
        assert!((115..=140).contains(&lin[0]), "{}", lin[0]);
        assert_eq!(resize(&px, 8, 8, 8, 8, true, ResizeFilter::Mitchell), px);
    }

    #[test]
    fn height_ramp_tilts_normals_and_flat_is_up() {
        // Height increases with x: normal leans toward -X.
        let mut px = Vec::new();
        for _y in 0..4 {
            for x in 0..4u32 {
                let v = (x * 60) as u8;
                px.extend_from_slice(&[v, v, v, 255]);
            }
        }
        height_to_normal(&mut px, 4, 4, 0.25, NormalFilter::Sharp);
        let mid = &px[(4 + 1) * 4..(4 + 1) * 4 + 3];
        assert!(mid[0] < 120, "x leans negative: {mid:?}");
        assert!((mid[1] as i32 - 128).abs() <= 2, "no y slope: {mid:?}");
        assert!(mid[2] > 128);
        let mut flat = solid(4, 4, [100, 100, 100, 255]);
        height_to_normal(&mut flat, 4, 4, 0.25, NormalFilter::Smooth);
        assert!(flat.chunks(4).all(|p| (p[0] as i32 - 128).abs() <= 1 && (p[1] as i32 - 128).abs() <= 1 && p[2] >= 254));
    }

    #[test]
    fn flip_and_renormalise() {
        let mut px = vec![128, 200, 255, 255];
        flip_green(&mut px);
        assert_eq!(px[1], 55);
        // A shortened normal (as averaging produces) is brought back to unit length.
        let mut short = vec![190, 128, 160, 255];
        renormalize(&mut short);
        let v = |b: u8| b as f32 / 255.0 * 2.0 - 1.0;
        let len = (v(short[0]).powi(2) + v(short[1]).powi(2) + v(short[2]).powi(2)).sqrt();
        assert!((len - 1.0).abs() < 0.02, "{len}");
        // Two-channel texel: Z rebuilt.
        let mut rg = vec![128, 128, 0, 255];
        renormalize(&mut rg);
        assert!(rg[2] > 250);
    }

    #[test]
    fn mip_chain_box_and_kaiser_reach_one_by_one() {
        let px = solid(7, 5, [200, 100, 50, 255]);
        for filter in [MipFilter::Box, MipFilter::Kaiser] {
            let chain = mip_chain(&px, 7, 5, &MipParams { filter, ..MipParams::default() });
            let last = chain.last().unwrap();
            assert_eq!((last.1, last.2), (1, 1));
            assert_eq!(chain.len(), 3, "7x5 -> 3x2 -> 1x1 {filter:?}");
            // A flat image stays flat through either filter.
            for (p, ..) in &chain {
                assert!(p.chunks(4).all(|t| (t[0] as i32 - 200).abs() <= 1 && (t[1] as i32 - 100).abs() <= 1), "{filter:?}");
            }
        }
        assert_eq!(mip_chain(&px, 7, 5, &MipParams { enabled: false, ..MipParams::default() }).len(), 1);
    }

    #[test]
    fn kaiser_weights_sum_to_one_and_are_symmetric() {
        let k = kaiser8();
        assert!((k.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        for i in 0..4 {
            assert!((k[i] - k[7 - i]).abs() < 1e-6);
        }
        assert!(k[3] > k[0]);
    }

    #[test]
    fn srgb_mips_do_not_darken_and_normal_mips_stay_unit() {
        let mut px = Vec::new();
        for i in 0..16u32 {
            let v = if i % 2 == 0 { 255 } else { 0 };
            px.extend_from_slice(&[v, v, v, 255]);
        }
        let chain = mip_chain(&px, 4, 4, &MipParams { srgb: true, ..MipParams::default() });
        assert!(chain[1].0[0] > 170, "{}", chain[1].0[0]);
        // Normal map: opposing tilts average toward a short vector that is renormalised.
        let mut n = Vec::new();
        for i in 0..16u32 {
            n.extend_from_slice(if i % 2 == 0 { &[230, 128, 200, 255] } else { &[26, 128, 200, 255] });
        }
        let chain = mip_chain(&n, 4, 4, &MipParams { normal: true, ..MipParams::default() });
        let t = &chain[1].0[..3];
        let v = |b: u8| b as f32 / 255.0 * 2.0 - 1.0;
        let len = (v(t[0]).powi(2) + v(t[1]).powi(2) + v(t[2]).powi(2)).sqrt();
        assert!((len - 1.0).abs() < 0.02, "{len}");
    }

    #[test]
    fn preserve_coverage_keeps_alpha_tested_density() {
        // Checkerboard of alpha 255 / 100: 50% of texels clear the 0.5 cutoff.
        let mut px = Vec::new();
        for y in 0..16u32 {
            for x in 0..16u32 {
                px.extend_from_slice(&[0, 255, 0, if (x + y) % 2 == 0 { 255 } else { 100 }]);
            }
        }
        // A 2x2 box average gives alpha ~178 everywhere: 100% coverage; with
        // an 0.8 cutoff (204) the plain chain loses everything.
        let plain = mip_chain(&px, 16, 16, &MipParams { coverage_cutoff: None, ..MipParams::default() });
        assert_eq!(coverage(&plain[1].0, 0.8), 0.0);
        let kept = mip_chain(&px, 16, 16, &MipParams { coverage_cutoff: Some(0.8), ..MipParams::default() });
        let base = coverage(&px, 0.8);
        assert!((base - 0.5).abs() < 1e-6);
        // Uniform alpha can only snap to all or nothing; the rescale picks "all".
        assert!(coverage(&kept[1].0, 0.8) >= 0.99);
        // A graded alpha ramp is matched closely.
        let mut ramp = Vec::new();
        for y in 0..32u32 {
            for x in 0..32u32 {
                ramp.extend_from_slice(&[0, 0, 0, ((x * 8 + y) % 256) as u8]);
            }
        }
        let target = coverage(&ramp, 0.5);
        let m = mip_chain(&ramp, 32, 32, &MipParams { coverage_cutoff: Some(0.5), ..MipParams::default() });
        for (lvl, ..) in m.iter().skip(1).take(3) {
            assert!((coverage(lvl, 0.5) - target).abs() < 0.08, "{} vs {target}", coverage(lvl, 0.5));
        }
    }
}
