//! Hi-Z in the XR stereo pass (GEO-5 leftover): two per-eye pyramids, the early cull tests the
//! previous XR frame's, the pass is split after its early draws, both pyramids are rebuilt from
//! this frame's depth layers and the late cull re-tests what was hidden. A cluster is culled only
//! when hidden in BOTH eyes. The clustered image must be pixel-identical to the same frame
//! without Hi-Z (an OR test would drop the clusters only one eye sees), single-sample and 4x MSAA.

use anyhow::Result;

use super::xr_cluster::{FORMAT, H, Scene, Target, W, diff, eyes, frame_input, per_frame_counts, save_sheet, set_mode, stereo_frame};
use crate::Renderer;

/// Frames with Hi-Z on, then (image, early rejects, late tested, late recovered, early tested).
fn hiz_frames(r: &mut Renderer, scene: &Scene, target: &Target, n: usize) -> Result<(Vec<u8>, crate::cluster::ClusterCounts)> {
    let mut img = Vec::new();
    for _ in 0..n {
        img = stereo_frame(r, scene, target, true, true, true)?;
    }
    let c = per_frame_counts(r).ok_or_else(|| anyhow::anyhow!("no XR cluster counters"))?;
    Ok((img, c))
}

fn one(r: &mut Renderer, scene: &Scene, target: &Target, label: &str) -> Result<(String, Vec<u8>, Vec<u8>)> {
    // The scene without the wall, no Hi-Z: what the disocclusion below must reproduce.
    let mut open_ref = Vec::new();
    for _ in 0..3 {
        open_ref = stereo_frame(r, scene, target, false, true, false)?;
    }
    let mut reference = Vec::new();
    for _ in 0..3 {
        reference = stereo_frame(r, scene, target, true, true, false)?;
    }
    let base = per_frame_counts(r).ok_or_else(|| anyhow::anyhow!("no XR cluster counters"))?;
    anyhow::ensure!(base.hiz == 0 && base.tested > 0, "{label}: the no-Hi-Z frame tested {} clusters, rejected {} by Hi-Z", base.tested, base.hiz);
    let (img, c) = hiz_frames(r, scene, target, 6)?;
    let (max, n) = diff(&reference, &img);
    let mut out = format!(
        "\n[xr-hiz] {label}: clusters tested {} (no Hi-Z {}), early Hi-Z rejects {}, recovered by the late pass {}, still hidden {}, visible {} (no Hi-Z {}), triangles {} (no Hi-Z {}); image vs no-Hi-Z: {n} px differ, max {max}/255",
        c.tested, base.tested, c.hiz, c.late_recovered, c.hiz - c.late_recovered.min(c.hiz), c.visible, base.visible, c.triangles, base.triangles
    );
    anyhow::ensure!(max == 0, "{label}: the Hi-Z image differs from the no-Hi-Z one ({n} px, max {max})");
    anyhow::ensure!(c.hiz > 0, "{label}: Hi-Z rejected nothing (counters {c:?})");
    anyhow::ensure!(c.visible < base.visible, "{label}: Hi-Z did not reduce the drawn clusters ({} vs {})", c.visible, base.visible);
    // The late pass keeps what is really hidden hidden (the wall does not move).
    anyhow::ensure!(c.hiz > c.late_recovered, "{label}: nothing stayed hidden after the late pass");
    // The wall vanishes: the pyramid of the last frame still hides the clusters behind it, the early
    // cull rejects them, the late pass (this frame's pyramids) must bring them all back, the same frame.
    let gone = stereo_frame(r, scene, target, false, true, true)?;
    let c2 = per_frame_counts(r).ok_or_else(|| anyhow::anyhow!("no XR cluster counters"))?;
    let (max2, n2) = diff(&open_ref, &gone);
    out += &format!("\n[xr-hiz] {label}: the wall is removed: early rejects {} from the stale pyramids, recovered by the late pass {}, drawn {}; image vs no-Hi-Z: {n2} px differ, max {max2}/255", c2.hiz, c2.late_recovered, c2.visible);
    anyhow::ensure!(c2.hiz > 0 && c2.late_recovered == c2.hiz, "{label}: the late pass did not recover every stale rejection ({} of {})", c2.late_recovered, c2.hiz);
    anyhow::ensure!(max2 == 0, "{label}: disoccluded frame differs from the no-Hi-Z one ({n2} px, max {max2})");
    out += &format!("; drawn clusters {} -> {} ({} culled)", base.visible, c.visible, base.visible - c.visible);
    Ok((out, reference, img))
}

pub(super) fn run(r: &mut Renderer, scene: &Scene, stereo: &Target) -> Result<String> {
    let mut report = String::from("\n[xr-hiz] real Renderer, multiview stereo pass, a wall 25 cm from the eyes, clusters + per-eye Hi-Z (hidden in both eyes only):");
    let (line, reference, img) = one(r, scene, stereo, "1x")?;
    report += &line;
    // A fresh view of the same scene with the camera moved a little: the previous pyramid is stale,
    // the late pass must recover what the move uncovers and the image still equals the no-Hi-Z one.
    let layer = (super::xr_cluster::W * super::xr_cluster::H * 4) as usize;
    let d: Vec<u8> = reference.iter().zip(&img).map(|(a, b)| a.abs_diff(*b).saturating_mul(8)).collect();
    save_sheet("geo5_xr_hiz.png", &[&reference[..layer], &img[..layer], &d[..layer], &reference[layer..], &img[layer..], &d[layer..]]);
    report += "\n[xr-hiz]   picture: docs/design/geo5_xr_hiz.png (left eye: no Hi-Z, Hi-Z, difference x8; then the right eye)";
    // 4x MSAA (the mobile tier's): the pass resolves its multisampled colour and depth after the late half.
    if r.xr_msaa == 1 {
        r.xr_msaa = 4;
        let t = Target::new(r, 2)?;
        match one(r, scene, &t, "4x MSAA") {
            Ok((line, _, _)) => report += &line,
            Err(e) => {
                t.destroy(r);
                r.xr_msaa = 1;
                return Err(e);
            }
        }
        t.destroy(r);
        r.xr_msaa = 1;
    }
    Ok(report)
}

/// One per-eye frame (both eyes, one `render_xr_eye_indexed` call each); the counters of each eye's cull.
fn per_eye_hiz_frame(r: &mut Renderer, scene: &Scene, targets: &[Target; 2], wall: bool, hiz: bool) -> Result<([Vec<u8>; 2], [crate::cluster::ClusterCounts; 2])> {
    let (eyes, cull) = eyes();
    set_mode(r, true, hiz);
    let draws = scene.draws(wall);
    let input = frame_input(&draws, cull.1);
    r.render_xr_frame(&input)?;
    let mut counts = [crate::cluster::ClusterCounts::default(), crate::cluster::ClusterCounts::default()];
    for e in 0..2 {
        r.render_xr_eye_indexed(e, targets[e].raw(), FORMAT.as_raw() as u32, W, H, eyes[e].view, eyes[e].proj, &input, None)?;
        // The counters of this eye's cull are copied out once its commands ran.
        let img = targets[e].read(r)?;
        counts[e] = per_frame_counts(r).ok_or_else(|| anyhow::anyhow!("no per-eye cluster counters"))?;
        let _ = img;
    }
    Ok(([targets[0].read(r)?, targets[1].read(r)?], counts))
}

/// Hi-Z in the per-eye (non-multiview) cluster path: each eye tests its own pyramid of the last frame,
/// the pass splits after its early draws, the pyramid is rebuilt from this frame's depth and the late
/// cull re-tests what was hidden. The image must equal the same frame without Hi-Z exactly.
pub(super) fn run_per_eye(r: &mut Renderer, scene: &Scene, targets: &[Target; 2]) -> Result<String> {
    let mut open_ref = [Vec::new(), Vec::new()];
    for _ in 0..3 {
        open_ref = per_eye_hiz_frame(r, scene, targets, false, false)?.0;
    }
    let (mut reference, mut base) = (Default::default(), Default::default());
    for _ in 0..3 {
        let (i, c) = per_eye_hiz_frame(r, scene, targets, true, false)?;
        reference = i;
        base = c;
    }
    anyhow::ensure!(base[0].hiz == 0 && base[0].tested > 0, "per-eye no-Hi-Z frame: tested {} rejected {}", base[0].tested, base[0].hiz);
    let (mut img, mut c) = (Default::default(), Default::default());
    for _ in 0..6 {
        let (i, k) = per_eye_hiz_frame(r, scene, targets, true, true)?;
        img = i;
        c = k;
    }
    let mut out = String::from("\n[xr-hiz] per-eye (non-multiview) pass, a wall 25 cm from the eyes, one pyramid per eye:");
    for e in 0..2 {
        let (max, n) = diff(&reference[e], &img[e]);
        out += &format!(
            "\n[xr-hiz]   eye {e}: clusters tested {} (no Hi-Z {}), early Hi-Z rejects {}, recovered by the late pass {}, visible {} (no Hi-Z {}), triangles {} ({}); image vs no-Hi-Z: {n} px differ, max {max}/255",
            c[e].tested, base[e].tested, c[e].hiz, c[e].late_recovered, c[e].visible, base[e].visible, c[e].triangles, base[e].triangles
        );
        anyhow::ensure!(max == 0, "per-eye Hi-Z image of eye {e} differs from the no-Hi-Z one ({n} px, max {max})");
        anyhow::ensure!(c[e].hiz > 0, "eye {e}: per-eye Hi-Z rejected nothing ({:?})", c[e]);
        anyhow::ensure!(c[e].visible < base[e].visible, "eye {e}: Hi-Z did not reduce the drawn clusters ({} vs {})", c[e].visible, base[e].visible);
        anyhow::ensure!(c[e].hiz > c[e].late_recovered, "eye {e}: nothing stayed hidden after the late pass");
    }
    // The wall vanishes: stale pyramids reject what is behind it, the late pass must recover all of it.
    let (gone, c2) = per_eye_hiz_frame(r, scene, targets, false, true)?;
    for e in 0..2 {
        let (max, n) = diff(&open_ref[e], &gone[e]);
        out += &format!("\n[xr-hiz]   eye {e}, wall removed: early rejects {} from the stale pyramid, recovered {}, drawn {}; image vs no-Hi-Z {n} px differ, max {max}", c2[e].hiz, c2[e].late_recovered, c2[e].visible);
        anyhow::ensure!(c2[e].hiz > 0 && c2[e].late_recovered == c2[e].hiz, "eye {e}: the late pass did not recover every stale rejection ({} of {})", c2[e].late_recovered, c2[e].hiz);
        anyhow::ensure!(max == 0, "eye {e}: disoccluded per-eye frame differs from the no-Hi-Z one ({n} px)");
    }
    Ok(out)
}
