//! GPU skinning + blendshapes of the project's real avatars (models/avatars)
//! against the CPU references, on a real Vulkan device (skips cleanly when the
//! machine has none: `ZESTENGINE_TEST_GPU=0` forces that).

use glam::{Mat4, Quat, Vec3};
use zestengine_assets::{Scene, load_gltf};
use zestengine_render::cluster_headless::HeadlessGpu;
use zestengine_render::skin_headless::{SkinFrame, max_vertex_error, skin_reference};
use zestengine_render::{MorphTables, Vertex, morph_reference};

fn root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Deterministic pseudo-random in [0, 1).
fn rnd(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    (*state >> 8) as f32 / (1u32 << 24) as f32
}

fn avatar_run(file: &str) {
    let Some(g) = HeadlessGpu::new().unwrap() else { return };
    let path = root().join("models/avatars/default").join(file);
    let scene: Scene = load_gltf(&path).unwrap_or_else(|e| panic!("{}: {e:#}", path.display()));
    let skel = scene.skeletons.first().expect("avatar has a skeleton");
    let rest = skel.rest_locals();
    let mut seed = 12345u32;
    let mut worst = 0.0f32;
    let mut skinned_meshes = 0;
    for (mi, mesh) in scene.meshes.iter().enumerate() {
        if !mesh.vertices.iter().any(|v| v.weights.iter().sum::<f32>() > 0.5) || mesh.vertices.is_empty() {
            continue;
        }
        skinned_meshes += 1;
        // Poses: rest, then two with every joint rotated by up to ~160 degrees.
        let mut palettes = vec![skel.palette(&rest)];
        for _ in 0..2 {
            let locals: Vec<Mat4> = rest
                .iter()
                .map(|m| {
                    let axis = Vec3::new(rnd(&mut seed) - 0.5, rnd(&mut seed) - 0.5, rnd(&mut seed) - 0.5).normalize_or(Vec3::Y);
                    *m * Mat4::from_quat(Quat::from_axis_angle(axis, (rnd(&mut seed) - 0.5) * 2.8))
                })
                .collect();
            palettes.push(skel.palette(&locals));
        }
        let morphs: &[zestengine_assets::MorphTarget] = scene.morphs.get(mi).map(|v| v.as_slice()).unwrap_or(&[]);
        type MorphDelta = (u32, [f32; 3], [f32; 3]);
        let shapes: Vec<&[MorphDelta]> = morphs.iter().map(|m| m.deltas.as_slice()).collect();
        let tables = MorphTables::build(&mesh.vertices, &shapes);
        let weight_sets: Vec<Vec<f32>> = (0..palettes.len())
            .map(|k| (0..shapes.len()).map(|_| if k == 0 { 0.0 } else { rnd(&mut seed) * 1.2 }).collect())
            .collect();
        let frames: Vec<SkinFrame> = palettes
            .iter()
            .zip(&weight_sets)
            .map(|(p, w)| SkinFrame { palette: p.clone(), morph: (!shapes.is_empty()).then(|| w.clone()), ..Default::default() })
            .collect();
        let out = g.skin_frames(&mesh.vertices, skel.joints.len(), &shapes, &frames).unwrap();
        let mut prev: Vec<Vertex> = mesh.vertices.clone();
        for (i, o) in out.iter().enumerate() {
            let morphed = if shapes.is_empty() { mesh.vertices.clone() } else { morph_reference(&mesh.vertices, &tables, &weight_sets[i]) };
            let want = skin_reference(&morphed, &palettes[i]);
            assert!(
                o.vertices.iter().all(|v| v.position.iter().chain(&v.normal).chain(&v.tangent).all(|x| x.is_finite())),
                "{file} mesh {mi} frame {i}: non-finite output"
            );
            // Skin positions are model-sized (~1-2 m); allow a few ulps of f32 accumulation.
            let err = max_vertex_error(&o.vertices, &want);
            worst = worst.max(err);
            assert!(err < 1e-3, "{file} mesh {mi} frame {i}: max error {err}");
            // prev stream = last frame's result (bind pose for frame 0). Morphing already
            // happened in the bind buffer, but the vertex buffer holds the *previous output*.
            for (k, (p, w)) in o.prev.iter().zip(&prev).enumerate() {
                for (c, pv) in p.iter().enumerate() {
                    assert!((pv - w.position[c]).abs() < 1e-3, "{file} mesh {mi} frame {i} vertex {k}: prev stream");
                }
            }
            prev = want;
        }
        eprintln!(
            "{file} mesh {mi}: {} verts, {} joints, {} shapes: ok",
            mesh.vertices.len(),
            skel.joints.len(),
            shapes.len()
        );
    }
    assert!(skinned_meshes > 0, "{file}: no skinned mesh found");
    eprintln!("{file}: {skinned_meshes} skinned meshes, worst error {worst:e}");
}

#[test]
fn male_avatar_skins_and_morphs_on_the_gpu() {
    avatar_run("male.glb");
}

#[test]
fn female_avatar_skins_and_morphs_on_the_gpu() {
    avatar_run("female.glb");
}
