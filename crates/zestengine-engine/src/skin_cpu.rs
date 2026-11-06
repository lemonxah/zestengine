//! CPU skin + morph for meshes the editor/runtime skins on the CPU: a fused,
//! allocation-free pass over a reused scratch buffer, plus the "did anything
//! change?" key that lets a still pose skip the whole re-skin and re-upload.

use std::collections::HashMap;

use glam::{Mat4, Vec3};
use zestengine_assets::MorphTarget;
use zestengine_render::Vertex;

/// Weights at or below this magnitude do not move a vertex.
const WEIGHT_EPS: f32 = 1e-4;

#[derive(Default)]
pub(crate) struct CpuSkin {
    /// Last uploaded `(weights, palette)` hash per (object raw id or `u64::MAX`, source mesh).
    keys: HashMap<(u64, usize), u64>,
    scratch: Vec<Vertex>,
    touched: Vec<bool>,
    touched_list: Vec<u32>,
}

#[inline]
fn mix(h: u64, w: u64) -> u64 {
    (h ^ w).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29)
}

/// Hash of a joint palette (bit exact: any changed joint changes it).
pub(crate) fn palette_hash(palette: &[Mat4]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ palette.len() as u64;
    for m in palette {
        for c in [m.x_axis, m.y_axis, m.z_axis, m.w_axis] {
            h = mix(h, (c.x.to_bits() as u64) << 32 | c.y.to_bits() as u64);
            h = mix(h, (c.z.to_bits() as u64) << 32 | c.w.to_bits() as u64);
        }
    }
    h
}

/// Hash of the morph weights (bit exact).
pub(crate) fn weights_hash(weights: &[f32]) -> u64 {
    weights.iter().fold(0x1234_5678_9abc_def0u64 ^ weights.len() as u64, |h, w| mix(h, w.to_bits() as u64))
}

impl CpuSkin {
    /// True when this mesh was last uploaded with the same weights and palette;
    /// otherwise records them as the new state (the caller uploads).
    pub(crate) fn unchanged(&mut self, owner: u64, mesh: usize, weights: &[f32], palette_hash: u64) -> bool {
        let key = mix(weights_hash(weights), palette_hash);
        self.keys.insert((owner, mesh), key) == Some(key)
    }

    /// Forget an object's recorded uploads (its GPU copies were re-created).
    pub(crate) fn forget_object(&mut self, owner: u64) {
        self.keys.retain(|(o, _), _| *o != owner);
    }

    /// Keep only the recorded uploads of objects still posed this pass.
    pub(crate) fn retain_objects(&mut self, mut keep: impl FnMut(u64) -> bool) {
        self.keys.retain(|(o, _), _| *o == u64::MAX || keep(*o));
    }

    /// `base` morphed by `weights` and linear-blend skinned by `palette`
    /// (what `face_drive::morphed` then `skin_vertices` produce), into the
    /// reused scratch buffer. Only active shapes cost anything.
    pub(crate) fn morph_skin(&mut self, base: &[Vertex], morphs: &[MorphTarget], weights: &[f32], palette: &[Mat4]) -> &[Vertex] {
        self.scratch.clear();
        self.scratch.extend_from_slice(base);
        if weights.iter().any(|w| w.abs() > WEIGHT_EPS) {
            self.touched.resize(base.len(), false);
            self.touched_list.clear();
            for (m, &w) in morphs.iter().zip(weights) {
                if w.abs() <= WEIGHT_EPS {
                    continue;
                }
                for &(i, dp, dn) in &m.deltas {
                    let Some(v) = self.scratch.get_mut(i as usize) else { continue };
                    v.position = (Vec3::from(v.position) + Vec3::from(dp) * w).to_array();
                    v.normal = (Vec3::from(v.normal) + Vec3::from(dn) * w).to_array();
                    if !self.touched[i as usize] {
                        self.touched[i as usize] = true;
                        self.touched_list.push(i);
                    }
                }
            }
            for &i in &self.touched_list {
                let v = &mut self.scratch[i as usize];
                v.normal = Vec3::from(v.normal).normalize_or(Vec3::Y).to_array();
                self.touched[i as usize] = false;
            }
        }
        for v in &mut self.scratch {
            v.position = zestengine_assets::skin_position(Vec3::from(v.position), v.joints, v.weights, palette).to_array();
            v.normal = zestengine_assets::skin_direction(Vec3::from(v.normal), v.joints, v.weights, palette).to_array();
        }
        &self.scratch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mesh() -> (Vec<Vertex>, Vec<MorphTarget>) {
        let base: Vec<Vertex> = (0..64)
            .map(|i| Vertex {
                position: [i as f32 * 0.1, (i % 7) as f32, 1.0 - i as f32 * 0.02],
                normal: Vec3::new(0.2, 1.0, 0.1 * i as f32).normalize().to_array(),
                joints: [i % 3, (i + 1) % 3, 0, 0],
                weights: [0.6, 0.4, 0.0, 0.0],
                ..Default::default()
            })
            .collect();
        let morphs = (0..12u32)
            .map(|s| MorphTarget {
                name: format!("shape{s}"),
                deltas: (0..64u32).filter(|i| (i + s) % 3 == 0).map(|i| (i, [0.01 * s as f32, 0.2, -0.1], [0.05, 0.0, 0.1 * s as f32])).collect(),
            })
            .collect();
        (base, morphs)
    }

    fn reference(base: &[Vertex], morphs: &[MorphTarget], w: &[f32], palette: &[Mat4]) -> Vec<Vertex> {
        let v = crate::face_drive::morphed(base, morphs, w);
        crate::scene::skin_vertices(v.as_deref().unwrap_or(base), palette)
    }

    #[test]
    fn fused_morph_skin_is_identical_to_the_two_step_path() {
        let (base, morphs) = mesh();
        let palettes = [
            vec![Mat4::IDENTITY; 3],
            vec![Mat4::from_rotation_y(0.4), Mat4::from_translation(Vec3::new(0.1, 0.2, 0.3)), Mat4::from_scale(Vec3::splat(1.1))],
        ];
        let mut cpu = CpuSkin::default();
        for palette in &palettes {
            for scale in [0.0f32, 0.5, 0.0, 1.0, 0.0] {
                let w: Vec<f32> = (0..12).map(|i| if i % 4 == 1 { scale } else { 0.0 }).collect();
                let got = cpu.morph_skin(&base, &morphs, &w, palette).to_vec();
                let want = reference(&base, &morphs, &w, palette);
                for (a, b) in got.iter().zip(&want) {
                    assert_eq!((a.position, a.normal), (b.position, b.normal), "scale {scale}");
                }
            }
        }
    }

    #[test]
    fn a_still_pose_is_unchanged_and_any_change_is_seen() {
        let mut cpu = CpuSkin::default();
        let p = vec![Mat4::IDENTITY; 4];
        let ph = palette_hash(&p);
        assert!(!cpu.unchanged(7, 0, &[0.0, 0.5], ph), "first upload");
        assert!(cpu.unchanged(7, 0, &[0.0, 0.5], ph));
        assert!(!cpu.unchanged(7, 0, &[0.0, 0.6], ph), "weight moved");
        let mut q = p.clone();
        q[2] = Mat4::from_rotation_x(0.01);
        assert!(!cpu.unchanged(7, 0, &[0.0, 0.6], palette_hash(&q)), "joint moved");
        assert!(!cpu.unchanged(8, 0, &[0.0, 0.6], palette_hash(&q)), "other object");
        cpu.forget_object(7);
        assert!(!cpu.unchanged(7, 0, &[0.0, 0.6], palette_hash(&q)), "re-created copies re-upload");
    }
}
