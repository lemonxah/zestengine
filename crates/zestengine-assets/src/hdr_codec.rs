//! CPU codec for baked HDR data: BC6H (UF16) encode via `intel_tex_2`, BC6H
//! decode via `bcdec_rs` (the fallback for GPUs without BC), and transcoding
//! of whole lightmap arrays / reflection cubes between [`HdrTextureFormat`]s.
//!
//! Everything here is pure CPU work meant for worker threads (bake save task,
//! scene loader); nothing touches the GPU.

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context as _, Result, bail};
use zestengine_render::{
    BakedLightmap, HdrCube, HdrTextureArray, HdrTextureFormat, convert_uncompressed, lightmap_array_size,
    resample_rgba32f, rgba32f_to_rgba16f,
};

/// BC6H preset for bakes: ISPC "basic" (endpoint refinement on). Measured on
/// a 2048² HDR card: ~0.5 s on one thread, ~0.1 s on 8 (very_fast is ~20x
/// quicker but its worst block error is ~40% higher; slow is ~5x slower for
/// no measurable gain). Bakes are encoded once, off the main thread.
fn bc6h_settings() -> intel_tex_2::bc6h::EncodeSettings {
    intel_tex_2::bc6h::basic_settings()
}

const F16_ONE: [u8; 2] = half::f16::ONE.to_bits().to_le_bytes();

/// Clamp an f16 texel channel into BC6H UF16's range (no negatives / NaN).
fn uf16_bits(b: [u8; 2]) -> [u8; 2] {
    let h = half::f16::from_le_bytes(b);
    if h.is_nan() || h.is_sign_negative() {
        [0, 0]
    } else if h.is_infinite() {
        half::f16::MAX.to_bits().to_le_bytes()
    } else {
        b
    }
}

/// Copy RGBA16F `src` (`w`×`h`) into a buffer padded to the 4x4 block grid by
/// replicating the last row / column, so partial edge blocks encode the image's
/// own colours. Returns (pixels, padded width, padded height).
pub fn pad_to_blocks(src: &[u8], w: u32, h: u32) -> (Vec<u8>, u32, u32) {
    let (w, h) = (w.max(1), h.max(1));
    let (pw, ph) = (w.div_ceil(4) * 4, h.div_ceil(4) * 4);
    let mut out = vec![0u8; (pw * ph * 8) as usize];
    for y in 0..ph {
        let sy = y.min(h - 1);
        for x in 0..pw {
            let sx = x.min(w - 1);
            let si = ((sy * w + sx) * 8) as usize;
            let di = ((y * pw + x) * 8) as usize;
            let Some(px) = src.get(si..si + 8) else { continue };
            for c in 0..3 {
                let v = uf16_bits([px[c * 2], px[c * 2 + 1]]);
                out[di + c * 2..di + c * 2 + 2].copy_from_slice(&v);
            }
            out[di + 6..di + 8].copy_from_slice(&F16_ONE);
        }
    }
    (out, pw, ph)
}

/// BC6H-encode one RGBA16F level of any size (padded to whole blocks).
/// Output is `ceil(w/4)*ceil(h/4)` blocks, row-major, as Vulkan expects.
/// Splits block rows into strips over `threads` scoped workers (which inherit
/// the caller's thread priority).
pub fn encode_bc6h(src: &[u8], w: u32, h: u32, threads: usize) -> Vec<u8> {
    let (padded, pw, ph) = pad_to_blocks(src, w, h);
    let settings = bc6h_settings();
    let block_rows = (ph / 4) as usize;
    let row_bytes = (pw * 8 * 4) as usize;
    let out_row = (pw / 4) as usize * 16;
    let per = block_rows.div_ceil(threads.clamp(1, block_rows.max(1)));
    let mut out = vec![0u8; block_rows * out_row];
    std::thread::scope(|s| {
        for (strip, dst) in padded.chunks(per * row_bytes).zip(out.chunks_mut(per * out_row)) {
            s.spawn(move || {
                let surface = intel_tex_2::RgbaSurface {
                    data: strip,
                    width: pw,
                    height: (strip.len() / row_bytes) as u32 * 4,
                    stride: pw * 8,
                };
                intel_tex_2::bc6h::compress_blocks_into(&settings, &surface, dst);
            });
        }
    });
    out
}

/// Decode a BC6H (UF16) level to RGBA16F bytes (alpha = 1).
pub fn decode_bc6h(blocks: &[u8], w: u32, h: u32) -> Result<Vec<u8>> {
    let (w, h) = (w.max(1) as usize, h.max(1) as usize);
    let (bw, bh) = (w.div_ceil(4), h.div_ceil(4));
    if blocks.len() < bw * bh * 16 {
        bail!("BC6H level too short: {} bytes for {w}x{h}", blocks.len());
    }
    let mut out = vec![0u8; w * h * 8];
    let mut texels = [0u16; 48];
    for by in 0..bh {
        for bx in 0..bw {
            let i = (by * bw + bx) * 16;
            bcdec_rs::bc6h_half(&blocks[i..i + 16], &mut texels, 12, false);
            for ty in 0..4 {
                let y = by * 4 + ty;
                if y >= h {
                    break;
                }
                for tx in 0..4 {
                    let x = bx * 4 + tx;
                    if x >= w {
                        break;
                    }
                    let o = (y * w + x) * 8;
                    let t = (ty * 4 + tx) * 3;
                    for c in 0..3 {
                        out[o + c * 2..o + c * 2 + 2].copy_from_slice(&texels[t + c].to_le_bytes());
                    }
                    out[o + 6..o + 8].copy_from_slice(&F16_ONE);
                }
            }
        }
    }
    Ok(out)
}

/// Convert one `w`×`h` level from `from` to `to`.
pub fn transcode_level(src: &[u8], from: HdrTextureFormat, to: HdrTextureFormat, w: u32, h: u32, threads: usize) -> Result<Vec<u8>> {
    if from == to {
        return Ok(src.to_vec());
    }
    let half = match from {
        HdrTextureFormat::Bc6h => decode_bc6h(src, w, h)?,
        HdrTextureFormat::AstcHdr4x4 => crate::tex_cache::decode_astc_hdr_level(src, w, h).context("decoding ASTC HDR")?,
        f => convert_uncompressed(src, f, HdrTextureFormat::Rgba16f).unwrap_or_default(),
    };
    Ok(match to {
        HdrTextureFormat::Bc6h => encode_bc6h(&half, w, h, threads),
        HdrTextureFormat::AstcHdr4x4 => crate::tex_cache::encode_astc_hdr_level(&half, w, h).context("encoding ASTC HDR")?,
        t => convert_uncompressed(&half, HdrTextureFormat::Rgba16f, t).unwrap_or_default(),
    })
}

/// Transcode every layer of a lightmap array (no-op when already `to`).
pub fn transcode_array(array: HdrTextureArray, to: HdrTextureFormat, threads: usize) -> Result<HdrTextureArray> {
    if array.format == to {
        return Ok(array);
    }
    let s = array.size;
    let layers = array
        .layers
        .iter()
        .map(|l| transcode_level(l, array.format, to, s, s, threads))
        .collect::<Result<_>>()?;
    Ok(HdrTextureArray { format: to, size: s, layers })
}

/// Transcode every face and mip of a reflection cube (no-op when already `to`).
pub fn transcode_cube(cube: HdrCube, to: HdrTextureFormat, threads: usize) -> Result<HdrCube> {
    if cube.format == to {
        return Ok(cube);
    }
    let mut faces: [Vec<Vec<u8>>; 6] = Default::default();
    for (dst, src) in faces.iter_mut().zip(&cube.faces) {
        for (m, level) in src.iter().enumerate() {
            let s = (cube.size >> m).max(1);
            dst.push(transcode_level(level, cube.format, to, s, s, threads)?);
        }
    }
    Ok(HdrCube { format: to, size: cube.size, faces })
}

/// Build the stored lightmap array for a bake: every layer resampled to the
/// common array size (exactly what the renderer's raw path uploads), then
/// stored as `format`. `progress` gets 0..1; returns None when `cancel` is set.
pub fn encode_lightmaps(
    maps: &[&BakedLightmap],
    format: HdrTextureFormat,
    threads: usize,
    cancel: &AtomicBool,
    progress: &dyn Fn(f32),
) -> Option<HdrTextureArray> {
    let size = lightmap_array_size(maps.iter().map(|l| l.size));
    let mut layers = Vec::with_capacity(maps.len());
    for (i, l) in maps.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return None;
        }
        let px = resample_rgba32f(&l.pixels, l.size, size);
        layers.push(match format {
            HdrTextureFormat::Rgba32f => bytemuck::cast_slice(&px).to_vec(),
            f => transcode_level(&rgba32f_to_rgba16f(&px), HdrTextureFormat::Rgba16f, f, size, size, threads).ok()?,
        });
        progress((i + 1) as f32 / maps.len().max(1) as f32);
    }
    Some(HdrTextureArray { format, size, layers })
}

/// BC6H-encode a cube's whole mip chain (any source format).
pub fn encode_cube(cube: HdrCube, threads: usize) -> Result<HdrCube> {
    transcode_cube(cube, HdrTextureFormat::Bc6h, threads)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_render::rgba16f_to_rgba32f;

    /// Synthetic HDR test card: a smooth horizontal gradient 0..4 with a
    /// soft bright spot (peak ~200) in the middle.
    fn hdr_card(w: u32, h: u32) -> Vec<f32> {
        let mut px = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let u = x as f32 / (w - 1).max(1) as f32;
                let v = y as f32 / (h - 1).max(1) as f32;
                let d2 = (u - 0.5).powi(2) + (v - 0.5).powi(2);
                let spot = 200.0 * (-d2 * 60.0).exp();
                px.extend_from_slice(&[4.0 * u + spot, 2.0 * v + spot * 0.8, 0.5 + spot * 0.6, 1.0]);
            }
        }
        px
    }

    /// Error relative to the local magnitude (BC6H is a float format, so its
    /// error scales with the value); `floor` keeps near-zero texels sane (BC6H
    /// interpolates in f16 bit space, so blocks spanning ~0 lose absolute
    /// precision at their dark end).
    fn max_rel_error(a: &[f32], b: &[f32], floor: f32) -> f32 {
        a.chunks(4)
            .zip(b.chunks(4))
            .flat_map(|(p, q)| (0..3).map(move |c| (p[c] - q[c]).abs() / p[c].abs().max(floor)))
            .fold(0.0, f32::max)
    }

    fn mean_rel_error(a: &[f32], b: &[f32], floor: f32) -> f32 {
        let n = a.len() / 4 * 3;
        a.chunks(4)
            .zip(b.chunks(4))
            .flat_map(|(p, q)| (0..3).map(move |c| (p[c] - q[c]).abs() / p[c].abs().max(floor)))
            .sum::<f32>()
            / n as f32
    }

    #[test]
    fn bc6h_round_trip_error_is_bounded() {
        let (w, h) = (64, 64);
        let src = hdr_card(w, h);
        let half = rgba32f_to_rgba16f(&src);
        let blocks = encode_bc6h(&half, w, h, 3);
        assert_eq!(blocks.len(), HdrTextureFormat::Bc6h.level_bytes(w, h));
        let back = rgba16f_to_rgba32f(&decode_bc6h(&blocks, w, h).unwrap());
        // Measured with the basic preset: mean 0.012, max 0.104.
        let mean = mean_rel_error(&src, &back, 1.0);
        let max = max_rel_error(&src, &back, 1.0);
        assert!(mean < 0.02, "mean relative error {mean}");
        assert!(max < 0.15, "max relative error {max}");
        // The bright spot keeps its peak (no clipping / exponent loss).
        let peak = |p: &[f32]| p.chunks(4).map(|t| t[0]).fold(0.0, f32::max);
        assert!((peak(&back) - peak(&src)).abs() / peak(&src) < 0.05);
        // Alpha decodes as 1 (BC6H has none).
        assert!(back.chunks(4).all(|t| t[3] == 1.0));
    }

    #[test]
    fn non_multiple_of_four_sizes_pad_and_crop() {
        for (w, h) in [(1, 1), (2, 2), (3, 5), (13, 7), (30, 30)] {
            // Smooth and lit everywhere (ambient floor), as lightmaps are; the
            // bright spot would be a sub-block outlier at these tiny sizes.
            let src: Vec<f32> = (0..w * h)
                .flat_map(|i| {
                    let t = (i % w + i / w) as f32;
                    [0.5 + t * 0.05, 1.0 + t * 0.03, 2.0, 1.0]
                })
                .collect();
            let half = rgba32f_to_rgba16f(&src);
            let (padded, pw, ph) = pad_to_blocks(&half, w, h);
            assert_eq!((pw % 4, ph % 4), (0, 0));
            assert_eq!(padded.len(), (pw * ph * 8) as usize);
            // Padding replicates the last column / row.
            let at = |x: u32, y: u32| &padded[((y * pw + x) * 8) as usize..((y * pw + x) * 8 + 6) as usize];
            assert_eq!(at(pw - 1, 0), at(w - 1, 0));
            assert_eq!(at(0, ph - 1), at(0, h - 1));
            let blocks = encode_bc6h(&half, w, h, 2);
            assert_eq!(blocks.len(), (w.div_ceil(4) * h.div_ceil(4) * 16) as usize);
            let back = rgba16f_to_rgba32f(&decode_bc6h(&blocks, w, h).unwrap());
            assert_eq!(back.len(), src.len());
            // A layout / crop mistake shows as gross error.
            let mean = mean_rel_error(&src, &back, 1.0);
            assert!(mean < 0.03, "{w}x{h}: mean relative error {mean}");
        }
    }

    #[test]
    fn negative_and_nan_inputs_encode_as_black() {
        let px = [-3.0f32, f32::NAN, -0.5, 1.0].repeat(16);
        let half: Vec<u8> = px.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect();
        let back = rgba16f_to_rgba32f(&decode_bc6h(&encode_bc6h(&half, 4, 4, 1), 4, 4).unwrap());
        assert!(back.chunks(4).all(|t| t[..3].iter().all(|&v| v == 0.0)), "{back:?}");
    }

    #[test]
    fn threaded_and_single_encode_match() {
        let (w, h) = (40, 36);
        let half = rgba32f_to_rgba16f(&hdr_card(w, h));
        assert_eq!(encode_bc6h(&half, w, h, 1), encode_bc6h(&half, w, h, 4));
        assert_eq!(encode_bc6h(&half, w, h, 1), encode_bc6h(&half, w, h, 64));
    }

    #[test]
    fn lightmaps_encode_at_the_common_array_size() {
        let a = BakedLightmap { size: 32, pixels: hdr_card(32, 32) };
        let b = BakedLightmap { size: 8, pixels: vec![1.0; 8 * 8 * 4] };
        let cancel = AtomicBool::new(false);
        let arr = encode_lightmaps(&[&a, &b], HdrTextureFormat::Bc6h, 2, &cancel, &|_| {}).unwrap();
        assert_eq!(arr.size, 32);
        assert!(arr.layers.iter().all(|l| l.len() == HdrTextureFormat::Bc6h.level_bytes(32, 32)));
        let flat = rgba16f_to_rgba32f(&decode_bc6h(&arr.layers[1], 32, 32).unwrap());
        assert!(flat.chunks(4).all(|t| (t[0] - 1.0).abs() < 0.01));
        // Fallback for a device without BC: decode to E5B9G9R9.
        let e5 = transcode_array(arr, HdrTextureFormat::Rgb9e5, 1).unwrap();
        assert!(e5.layers.iter().all(|l| l.len() == 32 * 32 * 4));
        cancel.store(true, Ordering::Relaxed);
        assert!(encode_lightmaps(&[&a], HdrTextureFormat::Bc6h, 1, &cancel, &|_| {}).is_none());
    }

    #[cfg(feature = "client")]
    #[test]
    fn cube_chain_compresses_every_mip() {
        let faces: [zestengine_render::TextureData; 6] = std::array::from_fn(|_| zestengine_render::TextureData {
            width: 64,
            height: 64,
            pixels: rgba32f_to_rgba16f(&hdr_card(64, 64)),
            srgb: false,
            hdr: true,
        });
        let chain = zestengine_render::reflection_cube_chain(&faces);
        assert_eq!(chain.mips(), 7);
        let bc = encode_cube(chain.clone(), 2).unwrap();
        assert!(bc.is_well_formed());
        assert_eq!(bc.faces[0][5].len(), 16, "2x2 mip is one padded block");
        assert_eq!(bc.faces[0][6].len(), 16, "1x1 mip is one padded block");
        let back = transcode_cube(bc, HdrTextureFormat::Rgba16f, 1).unwrap();
        assert!(back.is_well_formed());
        // Mips at 4x4 and below fold the bright spot into one block (a
        // sub-block outlier); they are the roughest lobes and aren't bounded.
        for mip in [0, 2] {
            let a = rgba16f_to_rgba32f(&chain.faces[3][mip]);
            let b = rgba16f_to_rgba32f(&back.faces[3][mip]);
            let mean = mean_rel_error(&a, &b, 1.0);
            assert!(mean < 0.05, "mip {mip}: mean relative error {mean}");
        }
    }

    #[test]
    fn astc_hdr_transcodes_round_trip() {
        let astc = HdrTextureFormat::AstcHdr4x4;
        let (w, h) = (12u32, 8u32);
        let half: Vec<u8> = (0..w * h)
            .flat_map(|i| [0.5 + (i % w) as f32 * 0.4, 2.0 + (i / w) as f32 * 0.25, 0.1, 1.0])
            .flat_map(|v| half::f16::from_f32(v).to_bits().to_le_bytes())
            .collect();
        let blocks = transcode_level(&half, HdrTextureFormat::Rgba16f, astc, w, h, 1).unwrap();
        assert_eq!(blocks.len(), astc.level_bytes(w, h));
        assert_eq!(transcode_level(&blocks, astc, astc, w, h, 1).unwrap(), blocks, "same format is a pass-through");
        let back = transcode_level(&blocks, astc, HdrTextureFormat::Rgba16f, w, h, 1).unwrap();
        let floats = |b: &[u8]| -> Vec<f32> { b.as_chunks::<2>().0.iter().map(|c| half::f16::from_bits(u16::from_le_bytes(*c)).to_f32()).collect() };
        let mean = mean_rel_error(&floats(&half), &floats(&back), 1.0);
        assert!(mean < 0.08, "mean relative error {mean}");
        let array = HdrTextureArray { format: astc, size: 8, layers: vec![transcode_level(&half[..8 * 8 * 8], HdrTextureFormat::Rgba16f, astc, 8, 8, 1).unwrap()] };
        assert_eq!(transcode_array(array, HdrTextureFormat::Rgb9e5, 1).unwrap().format, HdrTextureFormat::Rgb9e5);
    }

    /// Throughput of the bake preset on a 2048² layer (run with --ignored).
    #[test]
    #[ignore]
    fn bench_bc6h_2048() {
        let (w, h) = (2048, 2048);
        let half = rgba32f_to_rgba16f(&hdr_card(w, h));
        for threads in [1, 4, 8] {
            let t = std::time::Instant::now();
            let b = encode_bc6h(&half, w, h, threads);
            eprintln!("BC6H basic 2048² x{threads}: {:?} ({} bytes)", t.elapsed(), b.len());
        }
        let (p, pw, ph) = pad_to_blocks(&half, w, 256);
        for (name, s) in [("very_fast", intel_tex_2::bc6h::very_fast_settings()), ("very", intel_tex_2::bc6h::very_settings()), ("basic", intel_tex_2::bc6h::basic_settings()), ("slow", intel_tex_2::bc6h::slow_settings())] {
            let t = std::time::Instant::now();
            let _ = intel_tex_2::bc6h::compress_blocks(&s, &intel_tex_2::RgbaSurface { data: &p, width: pw, height: ph, stride: pw * 8 });
            eprintln!("BC6H {name} 2048x256 x1: {:?}", t.elapsed());
        }
        let b = encode_bc6h(&half, w, h, 8);
        let t = std::time::Instant::now();
        let _ = decode_bc6h(&b, w, h).unwrap();
        eprintln!("decode 2048²: {:?}", t.elapsed());
    }
}
