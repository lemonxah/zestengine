//! Octahedral impostors (GEO-5): a small atlas of views of a model, captured
//! once from its bind pose, drawn as a camera-facing quad that picks the views
//! nearest to the camera direction. The step between a full mesh and the
//! rank-tinted capsule for content the avatar safety policy hides or the
//! triangle budget cannot afford.
//!
//! The atlas has `grid` x `grid` views of `tile` x `tile` texels spread over
//! the whole sphere of directions by the octahedral mapping. Each view is an
//! orthographic projection of the model's bounding sphere, so every tile frames
//! the same `radius`. Two RGBA8 layers: albedo (rgb colour, a coverage) and aux
//! (rg octahedral normal, b depth within the sphere, a coverage).
//!
//! Two captures fill the same atlas ([`CaptureKind`]). The TEXTURED one is an
//! offscreen render of the model's bind pose through the engine's own
//! pipelines (`zestengine_render::Renderer::start_impostor_capture`: materials
//! and textures as imported, albedo through the shader's unlit mode, normals
//! and depth beside it); the engine runs it between frames and builds the
//! [`Impostor`] with [`Impostor::from_render`]. The fallback, used when no
//! renderer is available (tools, tests, a failed GPU capture), is a
//! deterministic CPU rasteriser (fixed iteration order, no threads) that reads
//! vertex positions, normals, vertex colour and the material's base colour
//! factor but NOT the textures. Cached beside the model like the other caches
//! (`.zestengine_geocache/<file>.zimp`), keyed by a hash of the mesh, the
//! settings, the capture kind (plus a hash of the materials for the textured
//! one) and [`IMPOSTOR_VERSION`].
//!
//! Limits, by design: the views are of the BIND pose, so an impostor of an
//! animated model looks wrong in motion (a T-pose statue turning to face the
//! camera); it is for distance and safety fallbacks only. Near the two poles
//! of the mapping neighbouring views are rolled against each other (the up
//! vector flips), which shows as a twist when the camera looks straight down
//! or up at the model.

use std::path::{Path, PathBuf};

use zestengine_render::MeshData;

use crate::geo_cache::GEOCACHE_DIR;
use crate::import_cache::Hasher128;

/// Bump when the capture changes its output for the same input (part of the key).
/// 2: textured renderer capture added (files carry their [`CaptureKind`]); version-1 files
/// (CPU captures, no kind field) still read, but their key no longer matches so they are recaptured.
pub const IMPOSTOR_VERSION: u32 = 2;
pub const IMPOSTOR_EXTENSION: &str = "zimp";
const FILE_FORMAT: u32 = 2;
/// The first file format (no `kind` field).
const FILE_FORMAT_V1: u32 = 1;
const MAGIC: &[u8; 4] = b"ZIMP";
/// Largest grid / tile accepted (a corrupt file cannot make a huge allocation).
pub const MAX_GRID: u32 = 32;
pub const MAX_TILE: u32 = 512;

/// Capture settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImpostorSettings {
    /// Views per side (`grid * grid` in all).
    pub grid: u32,
    /// Texels per view side.
    pub tile: u32,
}

impl Default for ImpostorSettings {
    /// 8 x 8 views of 64 x 64: 512 x 512 texels, 2 MB for both layers.
    fn default() -> Self {
        Self { grid: 8, tile: 64 }
    }
}

/// How an impostor's atlas was captured.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CaptureKind {
    /// The CPU rasteriser: flat vertex colour x base colour, no textures.
    #[default]
    Cpu = 0,
    /// The engine's renderer: materials and textures as imported.
    Textured = 1,
}

impl CaptureKind {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Cpu),
            1 => Some(Self::Textured),
            _ => None,
        }
    }
}

/// A captured impostor.
#[derive(Clone, Debug, PartialEq)]
pub struct Impostor {
    pub key: u128,
    pub kind: CaptureKind,
    pub grid: u32,
    pub tile: u32,
    /// Centre and radius of the bounding sphere every view frames.
    pub centre: [f32; 3],
    pub radius: f32,
    /// `(grid * tile)^2` RGBA8 texels, views laid out row by row.
    pub albedo: Vec<u8>,
    pub aux: Vec<u8>,
}

// ---------------------------------------------------------------- mapping

/// Octahedral encoding of a unit direction into `[-1, 1]^2`.
pub fn dir_to_oct(d: [f32; 3]) -> [f32; 2] {
    let l1 = d[0].abs() + d[1].abs() + d[2].abs();
    let (mut x, mut y) = (d[0] / l1, d[1] / l1);
    if d[2] < 0.0 {
        let (ox, oy) = (x, y);
        x = (1.0 - oy.abs()) * if ox >= 0.0 { 1.0 } else { -1.0 };
        y = (1.0 - ox.abs()) * if oy >= 0.0 { 1.0 } else { -1.0 };
    }
    [x, y]
}

/// Inverse of [`dir_to_oct`].
pub fn oct_to_dir(e: [f32; 2]) -> [f32; 3] {
    let mut n = [e[0], e[1], 1.0 - e[0].abs() - e[1].abs()];
    if n[2] < 0.0 {
        let (ox, oy) = (n[0], n[1]);
        n[0] = (1.0 - oy.abs()) * if ox >= 0.0 { 1.0 } else { -1.0 };
        n[1] = (1.0 - ox.abs()) * if oy >= 0.0 { 1.0 } else { -1.0 };
    }
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-20);
    [n[0] / l, n[1] / l, n[2] / l]
}

/// The direction (from the model towards the camera) view cell `(i, j)` was captured from.
pub fn cell_dir(grid: u32, i: u32, j: u32) -> [f32; 3] {
    oct_to_dir([(i as f32 + 0.5) / grid as f32 * 2.0 - 1.0, (j as f32 + 0.5) / grid as f32 * 2.0 - 1.0])
}

/// The four views nearest to the camera direction `dir` and their bilinear
/// weights (summing to 1): `(cell index = j * grid + i, weight)`.
pub fn lookup(grid: u32, dir: [f32; 3]) -> [(u32, f32); 4] {
    let o = dir_to_oct(dir);
    let g = grid as f32;
    let (fx, fy) = ((o[0] * 0.5 + 0.5) * g - 0.5, (o[1] * 0.5 + 0.5) * g - 0.5);
    let (x0, y0) = (fx.floor(), fy.floor());
    let (tx, ty) = (fx - x0, fy - y0);
    let cl = |v: f32| (v.max(0.0) as u32).min(grid - 1);
    let (xa, xb, ya, yb) = (cl(x0), cl(x0 + 1.0), cl(y0), cl(y0 + 1.0));
    [
        (ya * grid + xa, (1.0 - tx) * (1.0 - ty)),
        (ya * grid + xb, tx * (1.0 - ty)),
        (yb * grid + xa, (1.0 - tx) * ty),
        (yb * grid + xb, tx * ty),
    ]
}

/// Right and up of the view looking at the model from direction `dir`
/// (towards the camera): the frame both capture and the billboard use.
pub fn view_basis(dir: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let d = glam::Vec3::from(dir).normalize_or_zero();
    let hint = if d.y.abs() > 0.999 { glam::Vec3::Z } else { glam::Vec3::Y };
    let right = hint.cross(d).normalize_or_zero();
    let up = d.cross(right);
    (right.to_array(), up.to_array())
}

/// Camera-facing quad of an impostor: the four corners (world space) for a
/// camera at `camera` and the direction from the model to the camera.
pub fn billboard(centre: [f32; 3], radius: f32, camera: [f32; 3]) -> ([[f32; 3]; 4], [f32; 3]) {
    let c = glam::Vec3::from(centre);
    let dir = (glam::Vec3::from(camera) - c).normalize_or_zero();
    let dir = if dir == glam::Vec3::ZERO { glam::Vec3::Z } else { dir };
    let (r, u) = view_basis(dir.to_array());
    let (r, u) = (glam::Vec3::from(r) * radius, glam::Vec3::from(u) * radius);
    ([(c - r + u).to_array(), (c + r + u).to_array(), (c - r - u).to_array(), (c + r - u).to_array()], dir.to_array())
}

impl Impostor {
    pub fn atlas_size(&self) -> u32 {
        self.grid * self.tile
    }

    fn texel(&self, layer: &[u8], cell: u32, px: u32, py: u32) -> [u8; 4] {
        let (ci, cj) = (cell % self.grid, cell / self.grid);
        let at = (((cj * self.tile + py) * self.atlas_size() + ci * self.tile + px) * 4) as usize;
        [layer[at], layer[at + 1], layer[at + 2], layer[at + 3]]
    }

    /// Coverage (0..1) the billboard shows at `uv` (0..1 across the quad,
    /// v down) for a camera in direction `dir`: the weighted sum of the four
    /// nearest views' alpha, nearest texel in each.
    pub fn coverage(&self, dir: [f32; 3], uv: [f32; 2]) -> f32 {
        let px = ((uv[0] * self.tile as f32) as u32).min(self.tile - 1);
        let py = ((uv[1] * self.tile as f32) as u32).min(self.tile - 1);
        lookup(self.grid, dir).iter().map(|&(cell, w)| w * self.texel(&self.albedo, cell, px, py)[3] as f32 / 255.0).sum()
    }

    /// Coverage from the single nearest view (crisper silhouettes than the
    /// blend, which smears two different silhouettes together): the alpha the
    /// draw uses for its cut-out, while colour is blended.
    pub fn coverage_nearest(&self, dir: [f32; 3], uv: [f32; 2]) -> f32 {
        let px = ((uv[0] * self.tile as f32) as u32).min(self.tile - 1);
        let py = ((uv[1] * self.tile as f32) as u32).min(self.tile - 1);
        let best = lookup(self.grid, dir).into_iter().fold((0, -1.0f32), |b, (c, w)| if w > b.1 { (c, w) } else { b });
        self.texel(&self.albedo, best.0, px, py)[3] as f32 / 255.0
    }

    /// Bytes of both layers.
    pub fn byte_len(&self) -> usize {
        self.albedo.len() + self.aux.len()
    }

    /// Capture `mesh` (bind pose) with `base_color` multiplying its vertex colours.
    pub fn capture(mesh: &MeshData, base_color: [f32; 4], settings: &ImpostorSettings) -> Result<Self, String> {
        let (grid, tile) = (settings.grid, settings.tile);
        check_settings(settings)?;
        let (centre, radius) = bounds(mesh)?;
        let size = (grid * tile) as usize;
        let mut me = Impostor {
            key: key(mesh, base_color, settings),
            kind: CaptureKind::Cpu,
            grid,
            tile,
            centre,
            radius,
            albedo: vec![0; size * size * 4],
            aux: vec![0; size * size * 4],
        };
        for j in 0..grid {
            for i in 0..grid {
                me.capture_view(mesh, base_color, i, j);
            }
        }
        Ok(me)
    }

    /// An impostor from a renderer capture (`zestengine_render::CapturedImpostor` layers):
    /// validated against the settings, keyed by the caller's [`key_textured`].
    pub fn from_render(key: u128, settings: &ImpostorSettings, centre: [f32; 3], radius: f32, albedo: Vec<u8>, aux: Vec<u8>) -> Result<Self, String> {
        check_settings(settings)?;
        let layer = ((settings.grid * settings.tile) as usize).pow(2) * 4;
        if albedo.len() != layer || aux.len() != layer {
            return Err(format!("impostor layers are {} / {} bytes, expected {layer}", albedo.len(), aux.len()));
        }
        if !centre.iter().all(|v| v.is_finite()) || !radius.is_finite() || radius <= 0.0 {
            return Err("impostor bounds not finite".into());
        }
        Ok(Self { key, kind: CaptureKind::Textured, grid: settings.grid, tile: settings.tile, centre, radius, albedo, aux })
    }

    fn capture_view(&mut self, mesh: &MeshData, base: [f32; 4], ci: u32, cj: u32) {
        let t = self.tile as usize;
        let dir = glam::Vec3::from(cell_dir(self.grid, ci, cj));
        let (right, up) = view_basis(dir.to_array());
        let (right, up) = (glam::Vec3::from(right), glam::Vec3::from(up));
        let centre = glam::Vec3::from(self.centre);
        let r = self.radius;
        // Vertices in view space: x right, y up (pixels), z towards the camera (-1..1 of the radius).
        let to_view = |p: [f32; 3]| {
            let q = glam::Vec3::from(p) - centre;
            glam::Vec3::new(
                (q.dot(right) / r * 0.5 + 0.5) * t as f32,
                (0.5 - q.dot(up) / r * 0.5) * t as f32,
                q.dot(dir) / r,
            )
        };
        let mut zbuf = vec![f32::MIN; t * t];
        let size = self.atlas_size() as usize;
        let origin = ((cj as usize * t) * size + ci as usize * t) * 4;
        for tri in mesh.indices.as_chunks::<3>().0 {
            let (Some(va), Some(vb), Some(vc)) = (
                mesh.vertices.get(tri[0] as usize),
                mesh.vertices.get(tri[1] as usize),
                mesh.vertices.get(tri[2] as usize),
            ) else {
                continue;
            };
            let (a, b, c) = (to_view(va.position), to_view(vb.position), to_view(vc.position));
            let area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
            if area.abs() < 1e-9 {
                continue;
            }
            let (x0, x1) = (a.x.min(b.x).min(c.x).floor().max(0.0) as usize, (a.x.max(b.x).max(c.x).ceil() as usize).min(t));
            let (y0, y1) = (a.y.min(b.y).min(c.y).floor().max(0.0) as usize, (a.y.max(b.y).max(c.y).ceil() as usize).min(t));
            for y in y0..y1 {
                for x in x0..x1 {
                    let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                    let w0 = ((b.x - px) * (c.y - py) - (b.y - py) * (c.x - px)) / area;
                    let w1 = ((c.x - px) * (a.y - py) - (c.y - py) * (a.x - px)) / area;
                    let w2 = 1.0 - w0 - w1;
                    if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                        continue;
                    }
                    let z = w0 * a.z + w1 * b.z + w2 * c.z;
                    if z <= zbuf[y * t + x] {
                        continue;
                    }
                    zbuf[y * t + x] = z;
                    let mix = |f: &dyn Fn(&zestengine_render::Vertex) -> [f32; 4]| -> [f32; 4] {
                        let (p, q, s) = (f(va), f(vb), f(vc));
                        std::array::from_fn(|k| w0 * p[k] + w1 * q[k] + w2 * s[k])
                    };
                    let col = mix(&|v| v.color);
                    let n = mix(&|v| [v.normal[0], v.normal[1], v.normal[2], 0.0]);
                    let mut n = glam::Vec3::new(n[0], n[1], n[2]).normalize_or_zero();
                    if n == glam::Vec3::ZERO {
                        n = dir;
                    }
                    if n.dot(dir) < 0.0 && va.normal == [0.0; 3] {
                        n = -n;
                    }
                    let o = dir_to_oct(n.to_array());
                    let q8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                    let at = origin + (y * size + x) * 4;
                    self.albedo[at] = q8(col[0] * base[0]);
                    self.albedo[at + 1] = q8(col[1] * base[1]);
                    self.albedo[at + 2] = q8(col[2] * base[2]);
                    self.albedo[at + 3] = 255;
                    self.aux[at] = q8(o[0] * 0.5 + 0.5);
                    self.aux[at + 1] = q8(o[1] * 0.5 + 0.5);
                    self.aux[at + 2] = q8(z * 0.5 + 0.5);
                    self.aux[at + 3] = 255;
                }
            }
        }
    }

    // ------------------------------------------------------------- file

    /// Serialise: `ZIMP`, format, version, key, grid, tile, kind, centre, radius, both layers, checksum.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(64 + self.byte_len());
        b.extend_from_slice(MAGIC);
        b.extend_from_slice(&FILE_FORMAT.to_le_bytes());
        b.extend_from_slice(&IMPOSTOR_VERSION.to_le_bytes());
        b.extend_from_slice(&self.key.to_le_bytes());
        b.extend_from_slice(&self.grid.to_le_bytes());
        b.extend_from_slice(&self.tile.to_le_bytes());
        b.extend_from_slice(&(self.kind as u32).to_le_bytes());
        for v in self.centre.iter().chain(std::iter::once(&self.radius)) {
            b.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        b.extend_from_slice(&self.albedo);
        b.extend_from_slice(&self.aux);
        let mut h = Hasher128::new();
        h.update(&b);
        b.extend_from_slice(&h.finish().to_le_bytes());
        b
    }

    /// Parse and validate; `Err` for a bad checksum, another format / version, or sizes out of range.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        const HEAD_V1: usize = 4 + 4 + 4 + 16 + 4 + 4 + 16;
        const HEAD: usize = HEAD_V1 + 4;
        if bytes.len() < HEAD_V1 + 16 {
            return Err("impostor file too short".into());
        }
        let (body, sum) = bytes.split_at(bytes.len() - 16);
        let mut h = Hasher128::new();
        h.update(body);
        if h.finish().to_le_bytes() != sum {
            return Err("impostor checksum differs".into());
        }
        if &body[..4] != MAGIC {
            return Err("not an impostor file".into());
        }
        let u32_at = |at: usize| u32::from_le_bytes(body[at..at + 4].try_into().unwrap());
        // Format 1 (version 1) has no `kind` field and is always a CPU capture.
        let format = u32_at(4);
        let v1 = format == FILE_FORMAT_V1 && u32_at(8) == 1;
        if !v1 && (format != FILE_FORMAT || u32_at(8) != IMPOSTOR_VERSION) {
            return Err("impostor file from another version".into());
        }
        if !v1 && body.len() < HEAD {
            return Err("impostor file too short".into());
        }
        let head = if v1 { HEAD_V1 } else { HEAD };
        let key = u128::from_le_bytes(body[12..28].try_into().unwrap());
        let (grid, tile) = (u32_at(28), u32_at(32));
        if !(1..=MAX_GRID).contains(&grid) || !(4..=MAX_TILE).contains(&tile) {
            return Err("impostor sizes out of range".into());
        }
        let kind = if v1 { CaptureKind::Cpu } else { CaptureKind::from_u32(u32_at(36)).ok_or("impostor capture kind unknown")? };
        let f = |at: usize| f32::from_bits(u32_at(at + head - HEAD_V1));
        let (centre, radius) = ([f(36), f(40), f(44)], f(48));
        if !centre.iter().all(|v| v.is_finite()) || !radius.is_finite() || radius <= 0.0 {
            return Err("impostor bounds not finite".into());
        }
        let layer = ((grid * tile) as usize).pow(2) * 4;
        if body.len() != head + 2 * layer {
            return Err("impostor file has the wrong size".into());
        }
        Ok(Self { key, kind, grid, tile, centre, radius, albedo: body[head..head + layer].to_vec(), aux: body[head + layer..head + 2 * layer].to_vec() })
    }
}

fn check_settings(settings: &ImpostorSettings) -> Result<(), String> {
    let (grid, tile) = (settings.grid, settings.tile);
    if !(1..=MAX_GRID).contains(&grid) || !(4..=MAX_TILE).contains(&tile) {
        return Err(format!("impostor grid {grid} / tile {tile} out of range"));
    }
    Ok(())
}

/// Centre and radius of the sphere every view frames (the bounding sphere about the box centre, 2% slack).
pub fn bounds(mesh: &MeshData) -> Result<([f32; 3], f32), String> {
    if mesh.vertices.is_empty() || mesh.indices.len() < 3 {
        return Err("empty mesh".into());
    }
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for v in &mesh.vertices {
        for a in 0..3 {
            if !v.position[a].is_finite() {
                return Err("non-finite vertex".into());
            }
            lo[a] = lo[a].min(v.position[a]);
            hi[a] = hi[a].max(v.position[a]);
        }
    }
    let centre = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.5];
    let radius =
        mesh.vertices.iter().map(|v| glam::Vec3::from(v.position).distance(glam::Vec3::from(centre))).fold(0.0f32, f32::max).max(1e-4) * 1.02;
    Ok((centre, radius))
}

/// Cache key of the CPU capture: the mesh data, the colour factor, the settings and the version.
pub fn key(mesh: &MeshData, base_color: [f32; 4], settings: &ImpostorSettings) -> u128 {
    key_for(mesh, base_color, settings, CaptureKind::Cpu, 0)
}

/// Cache key of the textured renderer capture: the mesh data, the settings, the version and
/// `materials`, a hash the caller makes of everything the capture's look depends on besides
/// the geometry (material parameters and texture contents).
pub fn key_textured(mesh: &MeshData, settings: &ImpostorSettings, materials: u128) -> u128 {
    key_for(mesh, [1.0; 4], settings, CaptureKind::Textured, materials)
}

fn key_for(mesh: &MeshData, base_color: [f32; 4], settings: &ImpostorSettings, kind: CaptureKind, materials: u128) -> u128 {
    let mut h = Hasher128::new();
    h.field(&IMPOSTOR_VERSION.to_le_bytes());
    h.field(&FILE_FORMAT.to_le_bytes());
    h.field(&(kind as u32).to_le_bytes());
    h.field(&materials.to_le_bytes());
    h.field(&settings.grid.to_le_bytes());
    h.field(&settings.tile.to_le_bytes());
    for c in base_color {
        h.update(&c.to_bits().to_le_bytes());
    }
    h.field(bytemuck::cast_slice(&mesh.vertices));
    h.field(bytemuck::cast_slice(&mesh.indices));
    h.finish()
}

/// Where a model's impostor is stored: `<dir>/.zestengine_geocache/<file>.zimp`.
pub fn impostor_path(source: &Path) -> Option<PathBuf> {
    let name = source.file_name()?.to_string_lossy();
    Some(source.parent()?.join(GEOCACHE_DIR).join(format!("{name}.{IMPOSTOR_EXTENSION}")))
}

/// The stored impostor for `source` when its key is `want` (the textured path: look here before
/// capturing on the renderer, and [`store`] the result).
pub fn load_cached(source: &Path, want: u128) -> Option<Impostor> {
    if std::env::var_os("ZESTENGINE_NO_GEO_CACHE").is_some() {
        return None;
    }
    let bytes = std::fs::read(impostor_path(source)?).ok()?;
    Impostor::from_bytes(&bytes).ok().filter(|i| i.key == want)
}

/// Write `imp` as `source`'s cached impostor (best effort).
pub fn store(source: &Path, imp: &Impostor) {
    if std::env::var_os("ZESTENGINE_NO_GEO_CACHE").is_some() {
        return;
    }
    if let Some(p) = impostor_path(source)
        && let Err(e) = crate::write_atomic(&p, &imp.to_bytes())
    {
        tracing::debug!("impostor cache not written: {e:#}");
    }
}

/// The stored impostor when its key matches, else captured and stored.
/// Capture is CPU work (tens of ms for an avatar): call it from a worker.
pub fn load_or_capture(source: &Path, mesh: &MeshData, base_color: [f32; 4], settings: &ImpostorSettings) -> Result<Impostor, String> {
    let want = key(mesh, base_color, settings);
    let path = impostor_path(source);
    if std::env::var_os("ZESTENGINE_NO_GEO_CACHE").is_none()
        && let Some(p) = &path
        && let Ok(bytes) = std::fs::read(p)
        && let Ok(imp) = Impostor::from_bytes(&bytes)
        && imp.key == want
    {
        return Ok(imp);
    }
    let imp = Impostor::capture(mesh, base_color, settings)?;
    if std::env::var_os("ZESTENGINE_NO_GEO_CACHE").is_none()
        && let Some(p) = &path
        && let Err(e) = crate::write_atomic(p, &imp.to_bytes())
    {
        tracing::debug!("impostor cache not written: {e:#}");
    }
    Ok(imp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_render::Vertex;

    fn vtx(p: [f32; 3], n: [f32; 3]) -> Vertex {
        Vertex { position: p, normal: n, color: [0.8, 0.5, 0.2, 1.0], ..Default::default() }
    }

    fn sphere(lat: usize, lon: usize, r: f32) -> MeshData {
        let mut v = Vec::new();
        for i in 0..=lat {
            let th = std::f32::consts::PI * i as f32 / lat as f32;
            for j in 0..=lon {
                let ph = std::f32::consts::TAU * j as f32 / lon as f32;
                let n = [th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin()];
                v.push(vtx([n[0] * r, n[1] * r, n[2] * r], n));
            }
        }
        let mut idx = Vec::new();
        for i in 0..lat {
            for j in 0..lon {
                let a = (i * (lon + 1) + j) as u32;
                let b = a + lon as u32 + 1;
                idx.extend([a, b, a + 1, a + 1, b, b + 1]);
            }
        }
        MeshData { vertices: v, indices: idx, has_lightmap_uv: false }
    }

    fn torus(major: usize, minor: usize, big: f32, small: f32) -> MeshData {
        let mut v = Vec::new();
        for i in 0..=major {
            let a = std::f32::consts::TAU * i as f32 / major as f32;
            for j in 0..=minor {
                let b = std::f32::consts::TAU * j as f32 / minor as f32;
                let n = [b.cos() * a.cos(), b.sin(), b.cos() * a.sin()];
                v.push(vtx([(big + small * b.cos()) * a.cos(), small * b.sin(), (big + small * b.cos()) * a.sin()], n));
            }
        }
        let mut idx = Vec::new();
        for i in 0..major {
            for j in 0..minor {
                let a = (i * (minor + 1) + j) as u32;
                let b = a + minor as u32 + 1;
                idx.extend([a, b, a + 1, a + 1, b, b + 1]);
            }
        }
        MeshData { vertices: v, indices: idx, has_lightmap_uv: false }
    }

    /// Silhouette of `mesh` from `dir` at `tile` x `tile` with the same framing as the impostor, rasterised afresh.
    fn silhouette(imp: &Impostor, mesh: &MeshData, dir: [f32; 3], tile: u32) -> Vec<bool> {
        let settings = ImpostorSettings { grid: 1, tile };
        // A one-view capture along `dir`: reuse the rasteriser by making a 1x1 grid whose single
        // view is taken from the requested direction.
        let mut one = Impostor {
            key: 0,
            kind: CaptureKind::Cpu,
            grid: 1,
            tile,
            centre: imp.centre,
            radius: imp.radius,
            albedo: vec![0; (tile * tile * 4) as usize],
            aux: vec![0; (tile * tile * 4) as usize],
        };
        let _ = settings;
        one.capture_dir(mesh, [1.0; 4], dir);
        (0..(tile * tile) as usize).map(|i| one.albedo[i * 4 + 3] > 127).collect()
    }

    impl Impostor {
        /// Test helper: capture one view along an arbitrary direction into cell (0, 0).
        fn capture_dir(&mut self, mesh: &MeshData, base: [f32; 4], dir: [f32; 3]) {
            // `capture_view` takes its direction from the cell; a 1x1 grid has the +z cell, so
            // temporarily rotate the mesh into that frame instead of duplicating the rasteriser.
            let d = glam::Vec3::from(dir).normalize();
            let (right, up) = view_basis(dir);
            let (right, up) = (glam::Vec3::from(right), glam::Vec3::from(up));
            let (r1, u1) = view_basis([0.0, 0.0, 1.0]);
            let (r1, u1) = (glam::Vec3::from(r1), glam::Vec3::from(u1));
            let to_z = glam::Mat3::from_cols(r1, u1, glam::Vec3::Z) * glam::Mat3::from_cols(right, up, d).transpose();
            let c = glam::Vec3::from(self.centre);
            let mut m = MeshData { vertices: mesh.vertices.clone(), indices: mesh.indices.clone(), has_lightmap_uv: false };
            for v in &mut m.vertices {
                v.position = (c + to_z * (glam::Vec3::from(v.position) - c)).to_array();
                v.normal = (to_z * glam::Vec3::from(v.normal)).to_array();
            }
            self.capture_view(&m, base, 0, 0);
        }
    }

    #[test]
    fn octahedral_mapping_round_trips_and_lookup_weights_sum_to_one() {
        let mut seed = 3u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        };
        for _ in 0..2000 {
            let d = glam::Vec3::new(rnd(), rnd(), rnd()).normalize_or_zero();
            if d == glam::Vec3::ZERO {
                continue;
            }
            let back = glam::Vec3::from(oct_to_dir(dir_to_oct(d.to_array())));
            assert!(back.distance(d) < 1e-4, "{d:?} -> {back:?}");
            let w = lookup(8, d.to_array());
            let sum: f32 = w.iter().map(|x| x.1).sum();
            assert!((sum - 1.0).abs() < 1e-5 && w.iter().all(|x| x.1 >= -1e-6 && x.0 < 64));
        }
        // At a cell's own direction the lookup is that cell alone.
        for (i, j) in [(0, 0), (3, 5), (7, 7), (4, 1)] {
            let w = lookup(8, cell_dir(8, i, j));
            let hit = w.iter().find(|x| x.0 == j * 8 + i).unwrap();
            assert!(hit.1 > 0.999, "cell ({i},{j}) weight {}", hit.1);
        }
        // Basis is orthonormal and the billboard faces the camera.
        for d in [[0.0, 0.0, 1.0], [0.3, 0.8, -0.2], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]] {
            let dn = glam::Vec3::from(d).normalize();
            let (r, u) = view_basis(dn.to_array());
            let (r, u) = (glam::Vec3::from(r), glam::Vec3::from(u));
            assert!((r.length() - 1.0).abs() < 1e-5 && (u.length() - 1.0).abs() < 1e-5);
            assert!(r.dot(u).abs() < 1e-5 && r.dot(dn).abs() < 1e-5 && u.dot(dn).abs() < 1e-5);
        }
        let (quad, dir) = billboard([1.0, 2.0, 3.0], 2.0, [1.0, 2.0, 13.0]);
        assert_eq!(dir, [0.0, 0.0, 1.0]);
        assert!(quad.iter().all(|p| (p[2] - 3.0).abs() < 1e-5));
    }

    #[test]
    fn capture_is_deterministic_and_the_file_round_trips() {
        let m = sphere(24, 48, 1.0);
        let s = ImpostorSettings { grid: 4, tile: 32 };
        let a = Impostor::capture(&m, [1.0; 4], &s).unwrap();
        let b = Impostor::capture(&m, [1.0; 4], &s).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.to_bytes(), b.to_bytes());
        assert!(a.albedo.iter().skip(3).step_by(4).any(|&x| x == 255));
        let back = Impostor::from_bytes(&a.to_bytes()).unwrap();
        assert_eq!(back, a);
        // Damage anywhere is rejected; a different mesh / colour / setting is a different key.
        let mut bytes = a.to_bytes();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 1;
        assert!(Impostor::from_bytes(&bytes).is_err());
        assert!(Impostor::from_bytes(&a.to_bytes()[..100]).is_err());
        let mut m2 = sphere(24, 48, 1.0);
        m2.vertices[10].position[0] += 0.01;
        assert_ne!(key(&m, [1.0; 4], &s), key(&m2, [1.0; 4], &s));
        assert_ne!(key(&m, [1.0; 4], &s), key(&m, [0.5; 4], &s));
        assert_ne!(key(&m, [1.0; 4], &s), key(&m, [1.0; 4], &ImpostorSettings { grid: 8, tile: 32 }));
        assert!(Impostor::capture(&m, [1.0; 4], &ImpostorSettings { grid: 0, tile: 32 }).is_err());
        assert!(Impostor::capture(&MeshData { vertices: vec![], indices: vec![], has_lightmap_uv: false }, [1.0; 4], &s).is_err());
    }

    #[test]
    fn cache_is_keyed_and_replaced_when_the_mesh_changes() {
        let dir = std::env::temp_dir().join(format!("zest-imp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("m.glb");
        std::fs::write(&src, b"x").unwrap();
        let m = sphere(16, 32, 1.0);
        let s = ImpostorSettings { grid: 4, tile: 16 };
        let first = load_or_capture(&src, &m, [1.0; 4], &s).unwrap();
        assert!(impostor_path(&src).unwrap().is_file());
        assert_eq!(load_or_capture(&src, &m, [1.0; 4], &s).unwrap(), first);
        let mut m2 = sphere(16, 32, 1.0);
        m2.vertices[3].position[1] += 0.2;
        let second = load_or_capture(&src, &m2, [1.0; 4], &s).unwrap();
        assert_ne!(second.key, first.key);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Intersection over union of the impostor's billboard coverage against the real mesh's silhouette.
    fn iou(imp: &Impostor, mesh: &MeshData, dir: [f32; 3]) -> f32 {
        let t = imp.tile;
        let truth = silhouette(imp, mesh, dir, t);
        let (mut inter, mut uni) = (0usize, 0usize);
        for y in 0..t {
            for x in 0..t {
                let uv = [(x as f32 + 0.5) / t as f32, (y as f32 + 0.5) / t as f32];
                let a = imp.coverage_nearest(dir, uv) > 0.5;
                let b = truth[(y * t + x) as usize];
                inter += (a && b) as usize;
                uni += (a || b) as usize;
            }
        }
        inter as f32 / uni.max(1) as f32
    }

    fn check_iou(name: &str, mesh: &MeshData, min_cell: f32, min_between: f32) {
        let imp = Impostor::capture(mesh, [1.0; 4], &ImpostorSettings::default()).unwrap();
        // Exactly on a captured view the billboard IS that view.
        let on = iou(&imp, mesh, cell_dir(imp.grid, 2, 5));
        // Between views (the worst case: cell corners) and at oblique angles.
        let dirs = [[0.35f32, 0.4, 0.85], [-0.6, 0.3, 0.74], [0.8, -0.3, -0.5], [-0.2, 0.5, -0.84], [0.7, 0.1, 0.7]];
        let mut worst = 1.0f32;
        let mut line = String::new();
        for d in dirs {
            let v = iou(&imp, mesh, d);
            worst = worst.min(v);
            line += &format!(" {v:.3}");
        }
        eprintln!("[impostor] {name}: IoU on a captured view {on:.3}; between views:{line} (atlas {} KB)", imp.byte_len() / 1024);
        assert!(on > min_cell, "{name}: on-view IoU {on}");
        assert!(worst > min_between, "{name}: worst between-view IoU {worst}");
    }

    #[test]
    fn silhouette_matches_the_mesh_sphere() {
        check_iou("sphere", &sphere(48, 96, 1.0), 0.98, 0.95);
    }

    #[test]
    fn silhouette_matches_the_mesh_torus() {
        check_iou("torus", &torus(64, 32, 1.0, 0.35), 0.97, 0.65);
    }

    #[test]
    fn silhouette_matches_the_built_in_avatar() {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/avatars/default/male.glb");
        if !p.exists() {
            eprintln!("[impostor] male.glb not in the repo: skipped");
            return;
        }
        let Ok(scene) = crate::load_gltf(&p) else { return };
        let Some(m) = scene.meshes.iter().max_by_key(|m| m.indices.len()) else { return };
        check_iou("male avatar (bind pose)", m, 0.95, 0.60);
    }

    /// A file in the version-1 layout (no capture kind), as the previous release wrote it.
    fn v1_bytes(imp: &Impostor) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(MAGIC);
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&imp.key.to_le_bytes());
        b.extend_from_slice(&imp.grid.to_le_bytes());
        b.extend_from_slice(&imp.tile.to_le_bytes());
        for v in imp.centre.iter().chain(std::iter::once(&imp.radius)) {
            b.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        b.extend_from_slice(&imp.albedo);
        b.extend_from_slice(&imp.aux);
        let mut h = Hasher128::new();
        h.update(&b);
        b.extend_from_slice(&h.finish().to_le_bytes());
        b
    }

    #[test]
    fn version_one_files_still_read_and_recapture_under_the_new_key() {
        let m = sphere(12, 24, 1.0);
        let s = ImpostorSettings { grid: 4, tile: 16 };
        let imp = Impostor::capture(&m, [1.0; 4], &s).unwrap();
        let old = Impostor::from_bytes(&v1_bytes(&imp)).unwrap();
        assert_eq!((old.kind, old.grid, old.tile, old.centre, old.radius), (CaptureKind::Cpu, 4, 16, imp.centre, imp.radius));
        assert_eq!(old.albedo, imp.albedo);
        assert_eq!(old.aux, imp.aux);
        // And through the cache: a stored v1 file is replaced by a fresh capture.
        let dir = std::env::temp_dir().join(format!("zest-imp-v1-{}", std::process::id()));
        std::fs::create_dir_all(dir.join(GEOCACHE_DIR)).unwrap();
        let src = dir.join("m.glb");
        std::fs::write(&src, b"x").unwrap();
        std::fs::write(impostor_path(&src).unwrap(), v1_bytes(&Impostor { key: 7, ..imp.clone() })).unwrap();
        let fresh = load_or_capture(&src, &m, [1.0; 4], &s).unwrap();
        assert_eq!(fresh.key, key(&m, [1.0; 4], &s));
        assert_eq!(Impostor::from_bytes(&std::fs::read(impostor_path(&src).unwrap()).unwrap()).unwrap(), fresh);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn textured_impostors_carry_their_kind_and_key() {
        let m = sphere(12, 24, 1.0);
        let s = ImpostorSettings { grid: 4, tile: 16 };
        let (centre, radius) = bounds(&m).unwrap();
        let layer = (4 * 16usize).pow(2) * 4;
        let k = key_textured(&m, &s, 11);
        let imp = Impostor::from_render(k, &s, centre, radius, vec![9; layer], vec![3; layer]).unwrap();
        assert_eq!(imp.kind, CaptureKind::Textured);
        let back = Impostor::from_bytes(&imp.to_bytes()).unwrap();
        assert_eq!(back, imp);
        assert_eq!(back.kind, CaptureKind::Textured);
        // The key separates capture kinds and material sets from each other and from the CPU capture.
        assert_ne!(k, key_textured(&m, &s, 12));
        assert_ne!(k, key(&m, [1.0; 4], &s));
        assert_ne!(k, key_textured(&m, &ImpostorSettings { grid: 8, tile: 16 }, 11));
        // Bad layers / settings / bounds are rejected.
        assert!(Impostor::from_render(k, &s, centre, radius, vec![0; layer - 4], vec![0; layer]).is_err());
        assert!(Impostor::from_render(k, &ImpostorSettings { grid: 0, tile: 16 }, centre, radius, vec![0; layer], vec![0; layer]).is_err());
        assert!(Impostor::from_render(k, &s, centre, f32::NAN, vec![0; layer], vec![0; layer]).is_err());
        // Atlas layout validation against the GPU side: exactly what the upload accepts.
        let up = zestengine_render::ImpostorUpload { grid: imp.grid, tile: imp.tile, centre: imp.centre, radius: imp.radius, albedo: &imp.albedo, aux: &imp.aux };
        up.validate().unwrap();
        // The cache round trip keeps the textured kind.
        let dir = std::env::temp_dir().join(format!("zest-imp-tex-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("m.glb");
        std::fs::write(&src, b"x").unwrap();
        assert!(load_cached(&src, k).is_none());
        store(&src, &imp);
        assert_eq!(load_cached(&src, k).unwrap(), imp);
        assert!(load_cached(&src, k ^ 1).is_none(), "a different key misses");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_render_crates_view_maths_equals_this_one() {
        use zestengine_render::offscreen as r;
        for j in 0..8 {
            for i in 0..8 {
                assert_eq!(r::cell_dir(8, i, j), cell_dir(8, i, j));
            }
        }
        for d in [[0.0, 0.0, 1.0], [0.3, 0.8, -0.2], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [-0.7, 0.1, -0.7]] {
            assert_eq!(r::view_basis(d), view_basis(d));
            assert_eq!(r::dir_to_oct(d), dir_to_oct(d));
            assert_eq!(r::oct_to_dir([d[0], d[1]]), oct_to_dir([d[0], d[1]]));
        }
    }
}
