//! skin.comp / morph.comp / cloth_override.comp against their CPU references.

use glam::{Mat4, Quat, Vec3};

use super::gpu;
use crate::morph::{MorphTables, morph_reference};
use crate::skin_headless::{SkinFrame, max_vertex_error, skin_reference};
use crate::{Vertex, pack_cloth_patch};

/// A tube along +Y (`rings` x `segs` vertices) bound to three bones at y = 0,
/// 1.5 and 3 with smooth blend weights, plus the awkward vertices real
/// avatars have: four influences, weights that do not sum to one, an
/// out-of-range joint index, a zero-weight vertex and a non-unit tangent.
fn tube(rings: usize, segs: usize) -> Vec<Vertex> {
    let mut v = Vec::new();
    for r in 0..rings {
        let y = 3.0 * r as f32 / (rings - 1) as f32;
        for s in 0..segs {
            let a = std::f32::consts::TAU * s as f32 / segs as f32;
            let (sa, ca) = a.sin_cos();
            // Blend between the two nearest bones.
            let f = y / 1.5;
            let (j0, j1, t) = if f < 1.0 { (0u32, 1u32, f) } else { (1, 2, f - 1.0) };
            v.push(Vertex {
                position: [0.3 * ca, y, 0.3 * sa],
                normal: [ca, 0.0, sa],
                uv: [s as f32 / segs as f32, y / 3.0],
                tangent: [-sa, 0.0, ca, if s % 2 == 0 { 1.0 } else { -1.0 }],
                joints: [j0, j1, 0, 0],
                weights: [1.0 - t, t, 0.0, 0.0],
                ..Default::default()
            });
        }
    }
    let n = v.len();
    // Four influences with a 0.5-sum (the shader renormalises).
    v[1].joints = [0, 1, 2, 1];
    v[1].weights = [0.1, 0.2, 0.15, 0.05];
    // Out-of-range joint: ignored, the rest renormalised.
    v[2].joints = [7, 1, 0, 0];
    v[2].weights = [0.5, 0.5, 0.0, 0.0];
    // Weightless vertex: left in the bind pose.
    v[3].weights = [0.0; 4];
    // Non-unit tangent.
    v[4].tangent = [0.0, 3.0, 0.0, 1.0];
    // Only a zero-length rotated tangent (degenerate) must not produce NaN.
    v[5].tangent = [0.0, 0.0, 0.0, 1.0];
    // Only out-of-range joints with weight: blended matrix is zero, sum > 0.
    v[n - 1].joints = [9, 9, 9, 9];
    v[n - 1].weights = [0.25; 4];
    v
}

/// Bone poses: root, middle, tip. Absolute palette = pose * inverse bind.
fn palette(poses: [(Quat, Vec3); 3], scale: f32) -> Vec<Mat4> {
    let pivots = [0.0f32, 1.5, 3.0];
    (0..3)
        .map(|i| {
            let p = Vec3::new(0.0, pivots[i], 0.0);
            let (q, t) = poses[i];
            Mat4::from_translation(p + t) * Mat4::from_scale_rotation_translation(Vec3::splat(scale), q, Vec3::ZERO) * Mat4::from_translation(-p)
        })
        .collect()
}

fn extreme_a() -> Vec<Mat4> {
    palette(
        [
            (Quat::from_axis_angle(Vec3::X, std::f32::consts::PI), Vec3::new(1.0, 2.0, -3.0)),
            (Quat::from_axis_angle(Vec3::new(1.0, 1.0, 0.0).normalize(), 2.9), Vec3::ZERO),
            (Quat::from_axis_angle(Vec3::Z, -std::f32::consts::FRAC_PI_2) * Quat::from_axis_angle(Vec3::Y, 3.0), Vec3::new(0.0, 0.5, 0.0)),
        ],
        1.0,
    )
}

fn extreme_b() -> Vec<Mat4> {
    palette(
        [
            (Quat::from_axis_angle(Vec3::Y, 2.0), Vec3::ZERO),
            (Quat::from_axis_angle(Vec3::Z, std::f32::consts::PI), Vec3::new(0.0, -1.0, 0.0)),
            (Quat::from_rotation_arc(Vec3::Y, Vec3::new(1.0, -1.0, 0.5).normalize()), Vec3::new(5.0, 0.0, 0.0)),
        ],
        1.7,
    )
}

/// The synthetic tube with an index buffer, for the full-Renderer test.
pub(crate) fn skinned_tube_for_renderer() -> (Vec<Vertex>, Vec<u32>) {
    let (rings, segs) = (12usize, 8usize);
    let v = tube(rings, segs);
    let mut idx = Vec::new();
    for r in 0..rings - 1 {
        for s in 0..segs {
            let (a, b) = ((r * segs + s) as u32, (r * segs + (s + 1) % segs) as u32);
            let (c, d) = (a + segs as u32, b + segs as u32);
            idx.extend([a, b, c, b, d, c]);
        }
    }
    (v, idx)
}

fn finite(v: &[Vertex]) -> bool {
    v.iter().all(|v| {
        v.position.iter().chain(&v.normal).chain(&v.tangent).all(|x| x.is_finite())
    })
}

#[test]
fn skin_matches_cpu_with_extreme_bones_and_prev_stream() {
    let Some(g) = gpu() else { return };
    let base = tube(24, 16);
    let poses = [extreme_a(), extreme_b(), vec![Mat4::IDENTITY; 3], extreme_a(), extreme_b()];
    let frames: Vec<SkinFrame> = poses.iter().map(|p| SkinFrame { palette: p.clone(), ..Default::default() }).collect();
    let out = g.skin_frames(&base, 3, &[], &frames).unwrap();
    let mut prev_cpu: Vec<Vertex> = base.clone();
    let mut worst = 0.0f32;
    for (i, (f, o)) in frames.iter().zip(&out).enumerate() {
        assert!(finite(&o.vertices), "frame {i}: NaN/inf in the GPU result");
        let want = skin_reference(&base, &f.palette);
        let err = max_vertex_error(&o.vertices, &want);
        worst = worst.max(err);
        assert!(err < 2e-4, "frame {i}: GPU vs CPU skinning max error {err}");
        // The prev stream is last frame's skinned position (the bind pose before frame 0).
        for (k, (p, w)) in o.prev.iter().zip(&prev_cpu).enumerate() {
            for c in 0..3 {
                assert!((p[c] - w.position[c]).abs() < 2e-4, "frame {i} vertex {k}: prev {p:?} vs {:?}", w.position);
            }
        }
        prev_cpu = want;
        assert_eq!(o.validation_errors, 0);
    }
    eprintln!("skin_matches_cpu: max error over 5 frames x {} vertices = {worst:e}", base.len());
}

#[test]
fn skin_vertex_count_edges() {
    let Some(g) = gpu() else { return };
    // 1, 63, 64, 65, 129 vertices: workgroup boundaries (local_size_x = 64).
    for n in [1usize, 63, 64, 65, 129] {
        let base: Vec<Vertex> = tube(48, 16).into_iter().take(n).collect();
        let frames = [SkinFrame { palette: extreme_a(), ..Default::default() }];
        let out = g.skin_frames(&base, 3, &[], &frames).unwrap();
        let want = skin_reference(&base, &frames[0].palette);
        let err = max_vertex_error(&out[0].vertices, &want);
        assert!(err < 2e-4, "{n} vertices: error {err}");
    }
}

fn weightless_grid(n: usize) -> Vec<Vertex> {
    (0..n)
        .map(|i| Vertex {
            position: [i as f32 * 0.1, (i % 7) as f32 * 0.2, -(i as f32)],
            normal: Vec3::new(1.0, (i % 3) as f32, 0.5).normalize().to_array(),
            ..Default::default()
        })
        .collect()
}

/// Sparse shapes touching scattered vertices, one vertex in several shapes.
/// One shape's delta list: (vertex, delta position, delta normal).
type ShapeDelta = (u32, [f32; 3], [f32; 3]);

fn shapes_for(n: usize) -> Vec<Vec<ShapeDelta>> {
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut c = Vec::new();
    for i in (0..n).step_by(3) {
        a.push((i as u32, [0.1 * i as f32, 0.2, -0.3], [0.5, 0.0, 0.1]));
    }
    for i in (0..n).step_by(5) {
        b.push((i as u32, [0.0, -1.0, 0.25 * (i % 4) as f32], [0.0, 0.7, 0.0]));
    }
    for i in (0..n).step_by(11) {
        c.push((i as u32, [3.0, 0.0, 0.0], [0.0, 0.0, 0.0])); // position only
    }
    vec![a, b, c]
}

#[test]
fn morph_matches_reference_idempotent_and_reversible() {
    let Some(g) = gpu() else { return };
    // 1 and 65 vertices straddle the dispatch rounding; 500 spans many groups.
    for n in [1usize, 2, 65, 500] {
        let base = weightless_grid(n);
        let shapes = shapes_for(n);
        let refs: Vec<&[ShapeDelta]> = shapes.iter().map(|s| s.as_slice()).collect();
        let tables = MorphTables::build(&base, &refs);
        if tables.is_empty() {
            continue;
        }
        let sets: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.5, 0.25, 1.0],
            vec![1.0, 0.0, 0.0], // back to the first: nothing may accumulate
            vec![1.0, 0.0, 0.0], // unchanged: the pass is skipped
            vec![-0.7, 1.5, 0.3],
            vec![5e-5, 0.0, 0.0], // dead zone: bind pose restored
            vec![0.0, 0.0, 0.0],
            vec![0.4], // short weight list: missing shapes read as 0
            vec![0.2, 0.3, 0.4, 9.0, 9.0], // extra weights ignored
        ];
        let frames: Vec<SkinFrame> = sets
            .iter()
            .map(|w| SkinFrame { palette: vec![Mat4::IDENTITY], morph: Some(w.clone()), ..Default::default() })
            .collect();
        let out = g.skin_frames(&base, 1, &refs, &frames).unwrap();
        for (i, (w, o)) in sets.iter().zip(&out).enumerate() {
            let want = morph_reference(&base, &tables, w);
            let err = max_vertex_error(&o.vertices, &want);
            assert!(err < 1e-5, "{n} vertices, weight set {i} {w:?}: error {err}");
            assert_eq!(o.validation_errors, 0);
        }
        eprintln!("morph n={n}: {} weight sets match morph_reference", sets.len());
    }
}

#[test]
fn morph_then_skin_equals_cpu_morph_then_skin() {
    let Some(g) = gpu() else { return };
    let base = tube(20, 12);
    let n = base.len();
    let shapes = shapes_for(n);
    let refs: Vec<&[ShapeDelta]> = shapes.iter().map(|s| s.as_slice()).collect();
    let tables = MorphTables::build(&base, &refs);
    let sets = [vec![0.3, 0.9, 0.0], vec![0.0, 0.0, 0.0], vec![1.0, 1.0, 0.5]];
    let pals = [extreme_a(), extreme_b(), extreme_a()];
    let frames: Vec<SkinFrame> = sets
        .iter()
        .zip(&pals)
        .map(|(w, p)| SkinFrame { palette: p.clone(), morph: Some(w.clone()), ..Default::default() })
        .collect();
    let out = g.skin_frames(&base, 3, &refs, &frames).unwrap();
    for (i, o) in out.iter().enumerate() {
        let want = skin_reference(&morph_reference(&base, &tables, &sets[i]), &pals[i]);
        let err = max_vertex_error(&o.vertices, &want);
        assert!(err < 3e-4, "frame {i}: morph+skin error {err}");
    }
}

#[test]
fn cloth_override_replaces_only_cloth_vertices() {
    let Some(g) = gpu() else { return };
    let base = tube(24, 16);
    let n = base.len();
    let pal = extreme_a();
    let animated = skin_reference(&base, &pal);
    // Cloth moves every 4th vertex (position), every 9th (normal only).
    let mut sim = animated.clone();
    for i in (0..n).step_by(4) {
        sim[i].position[1] -= 0.5 + i as f32 * 0.001;
    }
    for i in (0..n).step_by(9) {
        sim[i].normal = [0.0, 0.0, 1.0];
    }
    let mut packed = Vec::new();
    let count = pack_cloth_patch(&animated, &sim, 1e-5, &mut packed);
    assert!(count > 0 && (count as usize) < n);
    let frames = [
        SkinFrame { palette: pal.clone(), cloth: Some((packed.clone(), count)), ..Default::default() },
        // Cloth off again (count 0): plain skinning must come back, including the vertices cloth moved.
        SkinFrame { palette: extreme_b(), cloth: Some((Vec::new(), 0)), ..Default::default() },
        // Cloth with a patch referencing a vertex past the end: ignored by the shader.
        SkinFrame {
            palette: pal.clone(),
            cloth: Some(([f32::from_bits(n as u32 + 10), 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 0.0].to_vec(), 1)),
            ..Default::default()
        },
    ];
    let out = g.skin_frames(&base, 3, &[], &frames).unwrap();
    // Frame 0: cloth vertices = simulated pos/normal (tangent stays skinned), others skinned.
    let moved: std::collections::HashSet<u32> = packed.chunks(8).map(|c| c[0].to_bits()).collect();
    let mut want = animated.clone();
    for &i in &moved {
        want[i as usize].position = sim[i as usize].position;
        want[i as usize].normal = sim[i as usize].normal;
    }
    let err = max_vertex_error(&out[0].vertices, &want);
    assert!(err < 2e-4, "cloth frame error {err}");
    // Frame 1: pure skinning with pose B.
    let err = max_vertex_error(&out[1].vertices, &skin_reference(&base, &extreme_b()));
    assert!(err < 2e-4, "cloth-off frame error {err}");
    // Frame 2: the out-of-range entry changed nothing.
    let err = max_vertex_error(&out[2].vertices, &animated);
    assert!(err < 2e-4, "out-of-range patch entry changed the mesh: {err}");
    // The prev stream of frame 1 holds frame 0's *cloth* position (what was on screen).
    for (k, p) in out[1].prev.iter().enumerate() {
        for (c, v) in p.iter().enumerate() {
            assert!((v - want[k].position[c]).abs() < 2e-4, "prev vertex {k}");
        }
    }
}

#[test]
fn a_mesh_skipped_by_the_cluster_path_is_skinned_from_its_last_pose_when_read_again() {
    let Some(g) = gpu() else { return };
    let base = tube(24, 16);
    let frames = [
        // The pose is queued in a frame the cluster path draws the mesh: the dispatch is dropped.
        SkinFrame { palette: extreme_a(), cluster_skip: true, ..Default::default() },
        // The pose did not change (the engine no longer re-sends it) and the ordinary buffer is read again.
        SkinFrame { skip_skin: true, ..Default::default() },
    ];
    let out = g.skin_frames(&base, 3, &[], &frames).unwrap();
    let err = max_vertex_error(&out[0].vertices, &base);
    assert!(err < 2e-4, "the skipped frame left the bind pose in the buffer: {err}");
    let err = max_vertex_error(&out[1].vertices, &skin_reference(&base, &extreme_a()));
    assert!(err < 2e-4, "the stale buffer was not re-skinned from its last pose: {err}");
}
