//! Built-in test scene: generated meshes + materials exercising the
//! standard shader's phase-1 feature set. Used when no glTF path is given.

use zestengine_render::{AlphaMode, MaterialFeatures, MaterialParams, MeshData, TextureData, Vertex};
use glam::{Mat4, Quat, Vec3};

use crate::scene_file::PrimitiveShape;
use crate::{Instance, Scene, SceneMaterial};

/// Generate the mesh for a creatable primitive shape.
pub fn primitive_mesh(shape: PrimitiveShape) -> MeshData {
    match shape {
        PrimitiveShape::Cube => cube(1.0),
        PrimitiveShape::Sphere => uv_sphere(0.5, 48, 24),
        PrimitiveShape::Capsule => capsule(0.25, 1.0, 32, 8),
        PrimitiveShape::Plane => plane(2.0, 1.0),
        PrimitiveShape::Bucket => bucket(),
    }
}

/// Wall and floor thickness of the Bucket primitive (unit size).
pub const BUCKET_WALL: f32 = 0.04;
pub const BUCKET_FLOOR: f32 = 0.05;

/// Unit open-top box: outer faces, inner faces and the rim, each quad its
/// own cell in a 4x4 lightmap atlas.
fn bucket() -> MeshData {
    let h = 0.5;
    let i = h - BUCKET_WALL;
    let f = -h + BUCKET_FLOOR;
    let p = |x: f32, y: f32, z: f32| Vec3::new(x, y, z);
    let quads: [([Vec3; 4], Vec3); 14] = [
        ([p(-h, -h, h), p(h, -h, h), p(h, h, h), p(-h, h, h)], Vec3::Z),
        ([p(h, -h, -h), p(-h, -h, -h), p(-h, h, -h), p(h, h, -h)], Vec3::NEG_Z),
        ([p(h, -h, h), p(h, -h, -h), p(h, h, -h), p(h, h, h)], Vec3::X),
        ([p(-h, -h, -h), p(-h, -h, h), p(-h, h, h), p(-h, h, -h)], Vec3::NEG_X),
        ([p(-h, -h, -h), p(h, -h, -h), p(h, -h, h), p(-h, -h, h)], Vec3::NEG_Y),
        ([p(i, f, i), p(-i, f, i), p(-i, h, i), p(i, h, i)], Vec3::NEG_Z),
        ([p(-i, f, -i), p(i, f, -i), p(i, h, -i), p(-i, h, -i)], Vec3::Z),
        ([p(i, f, -i), p(i, f, i), p(i, h, i), p(i, h, -i)], Vec3::NEG_X),
        ([p(-i, f, i), p(-i, f, -i), p(-i, h, -i), p(-i, h, i)], Vec3::X),
        ([p(-i, f, i), p(i, f, i), p(i, f, -i), p(-i, f, -i)], Vec3::Y),
        ([p(-h, h, h), p(h, h, h), p(i, h, i), p(-i, h, i)], Vec3::Y),
        ([p(h, h, -h), p(-h, h, -h), p(-i, h, -i), p(i, h, -i)], Vec3::Y),
        ([p(h, h, h), p(h, h, -h), p(i, h, -i), p(i, h, i)], Vec3::Y),
        ([p(-h, h, -h), p(-h, h, h), p(-i, h, i), p(-i, h, -i)], Vec3::Y),
    ];
    const M: f32 = 0.06;
    let mut vertices = Vec::with_capacity(56);
    let mut indices = Vec::with_capacity(84);
    for (q, (mut c, n)) in quads.into_iter().enumerate() {
        if (c[1] - c[0]).cross(c[2] - c[0]).dot(n) < 0.0 {
            c.swap(1, 3);
        }
        let base = vertices.len() as u32;
        let (col, row) = ((q % 4) as f32, (q / 4) as f32);
        let t = (c[1] - c[0]).normalize_or(Vec3::X);
        for (k, pos) in c.into_iter().enumerate() {
            let uv = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]][k];
            let uv1 = [(col + M + uv[0] * (1.0 - 2.0 * M)) / 4.0, (row + M + uv[1] * (1.0 - 2.0 * M)) / 4.0];
            vertices.push(Vertex { position: pos.to_array(), normal: n.to_array(), uv, uv1, tangent: [t.x, t.y, t.z, 1.0], ..Default::default() });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    MeshData { vertices, indices, has_lightmap_uv: true }
}

/// Build a `MeshData`, using the primary (clean, non-overlapping) UVs as the
/// lightmap UV set too. Primitives need no separate unwrap.
fn mesh_with_lightmap(mut vertices: Vec<Vertex>, indices: Vec<u32>) -> MeshData {
    for v in &mut vertices {
        v.uv1 = v.uv;
    }
    MeshData { vertices, indices, has_lightmap_uv: true }
}

/// Capsule: cylinder of height `height` between two hemispheres of `radius`,
/// total height = height + 2*radius, centered at the origin.
fn capsule(radius: f32, height: f32, segments: u32, rings: u32) -> MeshData {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let half = height * 0.5;
    let total_rings = rings * 2 + 1; // top hemi, equator band, bottom hemi

    for ring in 0..=total_rings {
        // Map ring index to a latitude angle plus a cylinder offset.
        let (theta, y_offset) = if ring <= rings {
            // Top hemisphere: theta 0..PI/2
            (
                ring as f32 / rings as f32 * std::f32::consts::FRAC_PI_2,
                half,
            )
        } else {
            // Bottom hemisphere: theta PI/2..PI
            (
                std::f32::consts::FRAC_PI_2
                    + (ring - rings - 1) as f32 / rings as f32 * std::f32::consts::FRAC_PI_2,
                -half,
            )
        };
        let (sin_t, cos_t) = theta.sin_cos();
        for segment in 0..=segments {
            let u = segment as f32 / segments as f32;
            let phi = u * std::f32::consts::TAU;
            let (sin_p, cos_p) = phi.sin_cos();
            let normal = [sin_t * cos_p, cos_t, sin_t * sin_p];
            vertices.push(Vertex {
                position: [
                    normal[0] * radius,
                    normal[1] * radius + y_offset,
                    normal[2] * radius,
                ],
                normal,
                uv: [u, ring as f32 / total_rings as f32],
                tangent: [-sin_p, 0.0, cos_p, 1.0],
                ..Default::default()
            });
        }
    }
    let stride = segments + 1;
    for ring in 0..total_rings {
        for segment in 0..segments {
            let a = ring * stride + segment;
            let b = a + stride;
            indices.extend_from_slice(&[a, a + 1, b, b, a + 1, b + 1]);
        }
    }
    mesh_with_lightmap(vertices, indices)
}

pub fn test_scene() -> Scene {
    let meshes = vec![uv_sphere(0.5, 48, 24), cube(1.0), plane(10.0, 4.0)];
    const SPHERE: usize = 0;
    const CUBE: usize = 1;
    const PLANE: usize = 2;

    let textures = vec![checker_texture(256, 32)];
    const CHECKER: usize = 0;

    let materials = vec![
        SceneMaterial {
            name: "Toon Sphere".into(),
            params: MaterialParams {
                base_color: [0.45, 0.6, 1.0, 1.0],
                roughness: 0.6,
                toon_steps: 3.0,
                pbr_toon_blend: 1.0,
                ..Default::default()
            },
            features: MaterialFeatures {
                toon: true,
                ..Default::default()
            },
            albedo: None,
            normal: None,
            orm: None,
            emission: None,
        },
        SceneMaterial {
            name: "Brushed Metal".into(),
            params: MaterialParams {
                base_color: [0.95, 0.93, 0.88, 1.0],
                metallic: 1.0,
                roughness: 0.3,
                ..Default::default()
            },
            features: MaterialFeatures::default(),
            albedo: None,
            normal: None,
            orm: None,
            emission: None,
        },
        SceneMaterial {
            name: "Emissive Core".into(),
            params: MaterialParams {
                base_color: [0.05, 0.05, 0.08, 1.0],
                roughness: 0.9,
                emission_color: [0.1, 0.9, 0.8],
                emission_intensity: 4.0,
                ..Default::default()
            },
            features: MaterialFeatures {
                emission: true,
                ..Default::default()
            },
            albedo: None,
            normal: None,
            orm: None,
            emission: None,
        },
        SceneMaterial {
            name: "Checker Floor".into(),
            params: MaterialParams {
                roughness: 0.85,
                ..Default::default()
            },
            features: MaterialFeatures::default(),
            albedo: Some(CHECKER),
            normal: None,
            orm: None,
            emission: None,
        },
        SceneMaterial {
            name: "Glass Panel".into(),
            params: MaterialParams {
                base_color: [0.5, 0.75, 0.9, 0.35],
                roughness: 0.1,
                ..Default::default()
            },
            features: MaterialFeatures {
                alpha_mode: AlphaMode::Blend,
                double_sided: true,
                ..Default::default()
            },
            albedo: None,
            normal: None,
            orm: None,
            emission: None,
        },
    ];

    let instances = vec![
        Instance::single("Floor", PLANE, 3, Mat4::IDENTITY),
        Instance::single(
            "Toon Sphere",
            SPHERE,
            0,
            Mat4::from_translation(Vec3::new(-1.3, 0.5, 0.0)),
        ),
        Instance::single(
            "Metal Cube",
            CUBE,
            1,
            Mat4::from_rotation_translation(Quat::from_rotation_y(0.6), Vec3::new(1.3, 0.5, 0.0)),
        ),
        Instance::single(
            "Emissive Core",
            SPHERE,
            2,
            Mat4::from_scale_rotation_translation(
                Vec3::splat(0.6),
                Quat::IDENTITY,
                Vec3::new(0.0, 0.3, 1.4),
            ),
        ),
        Instance::single(
            "Glass Panel",
            CUBE,
            4,
            Mat4::from_scale_rotation_translation(
                Vec3::new(1.6, 1.0, 0.05),
                Quat::from_rotation_y(-0.4),
                Vec3::new(0.0, 0.6, -1.6),
            ),
        ),
    ];

    Scene {
        meshes,
        textures,
        materials,
        instances,
        skeletons: Vec::new(),
        morphs: Vec::new(),
        springs: Default::default(),
        animations: Vec::new(),
        collision: Vec::new(),
        identity: Default::default(),
    }
}

fn checker_texture(size: u32, cell: u32) -> TextureData {
    let mut pixels = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let even = ((x / cell) + (y / cell)).is_multiple_of(2);
            let v = if even { 200 } else { 90 };
            pixels.extend_from_slice(&[v, v, v, 255]);
        }
    }
    TextureData {
        width: size,
        height: size,
        pixels,
        srgb: true,
        hdr: false,
    }
}

fn plane(size: f32, uv_tiles: f32) -> MeshData {
    let h = size * 0.5;
    let vertices = vec![
        Vertex {
            position: [-h, 0.0, -h],
            uv: [0.0, 0.0],
            ..Default::default()
        },
        Vertex {
            position: [h, 0.0, -h],
            uv: [uv_tiles, 0.0],
            ..Default::default()
        },
        Vertex {
            position: [h, 0.0, h],
            uv: [uv_tiles, uv_tiles],
            ..Default::default()
        },
        Vertex {
            position: [-h, 0.0, h],
            uv: [0.0, uv_tiles],
            ..Default::default()
        },
    ];
    mesh_with_lightmap(vertices, vec![0, 2, 1, 0, 3, 2])
}

/// Decode a heightmap image and resample it into a `(grid + 1)²` row-major
/// height grid (Z outer, X inner). Heights are the source's luminance in
/// metres: a pure 0..1 greyscale image maps to `[0, 1]` metres, ready to be
/// scaled/offset by the caller. Any colour space or HDR source is reduced to
/// luminance first, so `.png`/`.jpg`/`.exr`/`.hdr` all work.
pub fn import_heightmap(path: &std::path::Path, grid: u32) -> anyhow::Result<Vec<f32>> {
    use anyhow::Context as _;
    let img = crate::material_file::open_image_compat(path)
        .with_context(|| format!("loading heightmap {}", path.display()))?;
    let n = grid as usize + 1;
    let resized = image::imageops::resize(&img.to_luma32f(), n as u32, n as u32, image::imageops::FilterType::Triangle);
    let mut heights = Vec::with_capacity(n * n);
    for z in 0..n {
        for x in 0..n {
            heights.push(resized.get_pixel(x as u32, z as u32).0[0]);
        }
    }
    Ok(heights)
}

/// Tessellate a heightfield into a single square mesh. `heights` is a
/// `(resolution + 1)²` row-major grid over Z (outer) then X (inner); `size_xz`
/// is the world-space extent of the square in meters. Vertices sit on the XZ
/// plane centred at the origin with Y from the grid, upward normals from
/// central differences, and a whole-map 0..1 UV set that also serves as the
/// lightmap (non-overlapping). Front faces are wound to follow the normals.
pub fn heightfield_mesh(size_xz: f32, resolution: u32, heights: &[f32]) -> MeshData {
    heightfield_mesh_with_weights(size_xz, resolution, heights, None)
}

/// [`heightfield_mesh`] with per-vertex splat weights written into `color`
/// (RGBA = the four splat layers). `weights`, when given, is row-major over
/// Z then X and must match `(resolution + 1)²` entries; otherwise every
/// vertex is `(1, 0, 0, 0)` so a splat material shows only layer 0.
pub fn heightfield_mesh_with_weights(
    size_xz: f32,
    resolution: u32,
    heights: &[f32],
    weights: Option<&[[f32; 4]]>,
) -> MeshData {
    assert!(resolution >= 1, "heightfield resolution must be >= 1");
    let n = (resolution + 1) as usize;
    assert_eq!(heights.len(), n * n, "heights must be (resolution + 1)^2");
    assert!(weights.is_none_or(|w| w.len() == n * n), "weights must be (resolution + 1)^2");
    let weight_of = |i: usize| weights.map_or([1.0, 0.0, 0.0, 0.0], |w| w[i]);

    let step = size_xz / resolution as f32;
    let half = size_xz * 0.5;
    // Clamped central-difference sample so border vertices keep valid normals.
    let sample = |x: i64, z: i64| {
        let x = x.clamp(0, resolution as i64) as usize;
        let z = z.clamp(0, resolution as i64) as usize;
        heights[z * n + x]
    };

    let mut vertices = Vec::with_capacity(n * n);
    for z in 0..n {
        for x in 0..n {
            let px = x as f32 * step - half;
            let pz = z as f32 * step - half;
            let py = heights[z * n + x];
            let (xi, zi) = (x as i64, z as i64);
            // Gradient of the implicit surface y - h(x, z) = 0.
            let dh_dx = sample(xi + 1, zi) - sample(xi - 1, zi);
            let dh_dz = sample(xi, zi + 1) - sample(xi, zi - 1);
            let normal = Vec3::new(-dh_dx, 2.0 * step, -dh_dz).normalize_or(Vec3::Y);
            // +X tangent, orthogonalized against the normal (heightfields never
            // point exactly along X, so this is always non-zero).
            let tangent = (Vec3::X - normal * normal.dot(Vec3::X)).normalize_or(Vec3::Z);
            vertices.push(Vertex {
                position: [px, py, pz],
                normal: normal.to_array(),
                uv: [x as f32 / resolution as f32, z as f32 / resolution as f32],
                color: weight_of(z * n + x),
                tangent: [tangent.x, tangent.y, tangent.z, 1.0],
                ..Default::default()
            });
        }
    }

    let mut indices = Vec::with_capacity(resolution as usize * resolution as usize * 6);
    for z in 0..resolution as usize {
        for x in 0..resolution as usize {
            let a = (z * n + x) as u32;
            let b = a + 1;
            let c = a + n as u32;
            let d = c + 1;
            // Wound so a flat grid's triangles face +Y (matches the normals).
            indices.extend_from_slice(&[a, c, b, c, d, b]);
        }
    }
    for v in &mut vertices {
        v.uv1 = v.uv;
    }
    MeshData { vertices, indices, has_lightmap_uv: true }
}

fn cube(size: f32) -> MeshData {
    let h = size * 0.5;
    // (normal, tangent, four corners)
    type CubeFace = ([f32; 3], [f32; 4], [[f32; 3]; 4]);
    let faces: [CubeFace; 6] = [
        (
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [[-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h]],
        ),
        (
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0, 1.0],
            [[h, -h, -h], [-h, -h, -h], [-h, h, -h], [h, h, -h]],
        ),
        (
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 1.0],
            [[h, -h, h], [h, -h, -h], [h, h, -h], [h, h, h]],
        ),
        (
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [[-h, -h, -h], [-h, -h, h], [-h, h, h], [-h, h, -h]],
        ),
        (
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [[-h, h, h], [h, h, h], [h, h, -h], [-h, h, -h]],
        ),
        (
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [[-h, -h, -h], [h, -h, -h], [h, -h, h], [-h, -h, h]],
        ),
    ];
    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    // uv0 is per-face 0..1 (texture mapping, tiles/overlaps across faces); uv1
    // is a non-overlapping lightmap atlas: the 6 faces packed into a 3x2 grid,
    // each face inset by a gutter so the bake/bilinear filter don't bleed
    // between faces. (uv0 alone can't be a lightmap: all faces would share the
    // same 0..1 region and fight over the same texels.)
    const M: f32 = 0.06; // per-cell inset (gutter): bigger = wider inter-face gap
                         // so bilinear filtering doesn't bleed between faces.
    for (f, (normal, tangent, corners)) in faces.into_iter().enumerate() {
        let base = vertices.len() as u32;
        let col = (f % 3) as f32;
        let row = (f / 3) as f32;
        for (i, position) in corners.into_iter().enumerate() {
            let uv = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]][i];
            let uv1 = [
                (col + M + uv[0] * (1.0 - 2.0 * M)) / 3.0,
                (row + M + uv[1] * (1.0 - 2.0 * M)) / 2.0,
            ];
            vertices.push(Vertex {
                position,
                normal,
                uv,
                uv1,
                tangent,
                ..Default::default()
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    // The cube packs its 6 faces into a non-overlapping 3×2 lightmap atlas.
    MeshData { vertices, indices, has_lightmap_uv: true }
}

fn uv_sphere(radius: f32, segments: u32, rings: u32) -> MeshData {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for ring in 0..=rings {
        let v = ring as f32 / rings as f32;
        let theta = v * std::f32::consts::PI;
        let (sin_t, cos_t) = theta.sin_cos();
        for segment in 0..=segments {
            let u = segment as f32 / segments as f32;
            let phi = u * std::f32::consts::TAU;
            let (sin_p, cos_p) = phi.sin_cos();
            let normal = [sin_t * cos_p, cos_t, sin_t * sin_p];
            vertices.push(Vertex {
                position: [normal[0] * radius, normal[1] * radius, normal[2] * radius],
                normal,
                uv: [u, v],
                // d(position)/du: tangent along longitude.
                tangent: [-sin_p, 0.0, cos_p, 1.0],
                ..Default::default()
            });
        }
    }
    let stride = segments + 1;
    for ring in 0..rings {
        for segment in 0..segments {
            let a = ring * stride + segment;
            let b = a + stride;
            indices.extend_from_slice(&[a, a + 1, b, b, a + 1, b + 1]);
        }
    }
    mesh_with_lightmap(vertices, indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bucket_is_open_on_top_and_wound_outward() {
        let m = primitive_mesh(PrimitiveShape::Bucket);
        assert_eq!(m.indices.len(), 14 * 6);
        for t in m.indices.chunks(3) {
            let [a, b, c] = [t[0], t[1], t[2]].map(|i| Vec3::from(m.vertices[i as usize].position));
            let n = Vec3::from(m.vertices[t[0] as usize].normal);
            assert!((b - a).cross(c - a).dot(n) > 0.0, "front faces follow the normals");
        }
        let inner = 0.5 - BUCKET_WALL;
        let covers_mouth = m.vertices.iter().any(|v| v.position[1] > 0.49 && v.position[0].abs() < inner - 1e-3 && v.position[2].abs() < inner - 1e-3);
        assert!(!covers_mouth, "nothing across the opening");
    }

    #[test]
    fn heightfield_counts_and_flat_geometry() {
        let res = 4;
        let m = heightfield_mesh(100.0, res, &[0.0; 25]);
        assert_eq!(m.vertices.len(), 25);
        assert_eq!(m.indices.len(), res as usize * res as usize * 6);
        for v in &m.vertices {
            assert_eq!(v.position[1], 0.0);
            assert!((v.normal[1] - 1.0).abs() < 1e-5, "flat heightfield points up");
            assert!(v.position[0].abs() <= 50.0 + 1e-4);
            assert!(v.position[2].abs() <= 50.0 + 1e-4);
        }
        let min = m.vertices.iter().fold([f32::INFINITY; 2], |a, v| [a[0].min(v.position[0]), a[1].min(v.position[2])]);
        let max = m.vertices.iter().fold([f32::NEG_INFINITY; 2], |a, v| [a[0].max(v.position[0]), a[1].max(v.position[2])]);
        assert!((min[0] + 50.0).abs() < 1e-4 && (max[0] - 50.0).abs() < 1e-4, "x extent is size_xz");
        assert!((min[1] + 50.0).abs() < 1e-4 && (max[1] - 50.0).abs() < 1e-4, "z extent is size_xz");
    }

    #[test]
    fn heightfield_winding_follows_normals() {
        let res = 8;
        let n = (res + 1) as usize;
        let heights: Vec<f32> = (0..n * n)
            .map(|i| {
                let (x, z) = (i % n, i / n);
                ((x as f32 * 0.3).sin() * (z as f32 * 0.25).cos()) * 2.0
            })
            .collect();
        let m = heightfield_mesh(64.0, res, &heights);
        for t in m.indices.chunks(3) {
            let [a, b, c] = [t[0], t[1], t[2]].map(|i| Vec3::from(m.vertices[i as usize].position));
            let nrm = Vec3::from(m.vertices[t[0] as usize].normal);
            assert!((b - a).cross(c - a).dot(nrm) > 0.0, "front faces follow the normals");
        }
        for v in &m.vertices {
            let nrm = Vec3::from(v.normal);
            assert!((nrm.length() - 1.0).abs() < 1e-4, "unit normals");
            assert!(v.normal[1] > 0.0, "heightfield normals point up");
        }
    }

    #[test]
    fn heightfield_applies_heights_row_major() {
        let m = heightfield_mesh(10.0, 1, &[0.0, 1.0, 2.0, 3.0]);
        let y_at = |x: f32, z: f32| {
            m.vertices
                .iter()
                .find(|v| (v.position[0] - x).abs() < 1e-4 && (v.position[2] - z).abs() < 1e-4)
                .expect("grid corner")
                .position[1]
        };
        assert_eq!(y_at(-5.0, -5.0), 0.0);
        assert_eq!(y_at(5.0, -5.0), 1.0);
        assert_eq!(y_at(-5.0, 5.0), 2.0);
        assert_eq!(y_at(5.0, 5.0), 3.0);
    }

    #[test]
    fn heightmap_import_resamples_to_the_grid_and_reads_luminance() {
        let dir = std::env::temp_dir().join(format!("zestengine-hm-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // 2x2 red/green/blue/white, resampled to a 2x2 grid (3x3 vertices).
        let img = image::RgbaImage::from_raw(2, 2, vec![
            255, 0, 0, 255, 0, 255, 0, 255,
            0, 0, 255, 255, 255, 255, 255, 255,
        ]).unwrap();
        let path = dir.join("h.png");
        img.save(&path).unwrap();
        let heights = import_heightmap(&path, 2).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(heights.len(), 9);
        // Red is darker than blue, which is darker than white.
        assert!(heights[0] < heights[8]);
        assert!(heights.iter().all(|h| (0.0..=1.0).contains(h)));
    }

    #[test]
    fn heightfield_weights_fill_vertex_color() {
        let m = heightfield_mesh_with_weights(1.0, 1, &[0.0, 0.0, 0.0, 0.0], Some(&[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]));
        assert_eq!(m.vertices[0].color, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(m.vertices[3].color, [0.0, 0.0, 0.0, 1.0]);
    }
}
