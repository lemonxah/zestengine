//! Avatar skinning benchmark (`ZESTENGINE_BENCH_AVATARS=<n>`): spawns `n` copies
//! of the project's avatar models in a grid when a scene loads, and logs the
//! per-second CPU frame time and GPU frame / skinning / morph times under the
//! `frame_log` target so before/after numbers for GPU skinning work can be
//! recorded. Nothing here measures anything by itself: the numbers come from
//! the GPU timestamp zones (`SKIN`, `MORPH`, `FRAME`) of the running renderer.

use std::path::Path;
use std::time::Instant;

use zestengine_assets::{MaterialRef, ObjectSource, SceneEntry, SceneFile};
use zestengine_render::{MaterialFeatures, MaterialParams};
use zestengine_core::{Animator, ShadowCasting};

/// Layer used by the template avatars (characters).
const CHARACTER_LAYER: u8 = 8;
/// Distance between neighbouring avatars (m).
const SPACING: f32 = 1.5;
/// Models tried when the project has no avatar directory content.
const DEFAULT_MODELS: [&str; 2] = ["models/avatars/default/male.glb", "models/avatars/default/female.glb"];
const CONTROLLER: &str = "animations/default_locomotion.animctrl";

/// Avatar count requested by `ZESTENGINE_BENCH_AVATARS`, if set and non-zero.
pub fn requested() -> Option<usize> {
    static N: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("ZESTENGINE_BENCH_AVATARS").ok()?.trim().parse().ok().filter(|n| *n > 0))
}

/// Per-frame hook for the host loops: no-op unless the benchmark is on.
pub fn tick(stats: &zestengine_render::RenderStats) {
    static LOG: std::sync::Mutex<Option<(Instant, BenchLog)>> = std::sync::Mutex::new(None);
    let Some(n) = requested() else { return };
    let Ok(mut g) = LOG.lock() else { return };
    let now = Instant::now();
    match g.as_mut() {
        Some((prev, log)) => {
            let ms = now.duration_since(*prev).as_secs_f32() * 1000.0;
            *prev = now;
            log.record(n, ms, stats);
        }
        None => *g = Some((now, BenchLog::default())),
    }
}

/// Grid slots for `n` avatars, centred on the origin, `SPACING` apart.
pub fn grid_positions(n: usize) -> Vec<[f32; 3]> {
    if n == 0 {
        return Vec::new();
    }
    let cols = (n as f32).sqrt().ceil().max(1.0) as usize;
    let rows = n.div_ceil(cols);
    let (cx, cz) = ((cols - 1) as f32 * 0.5, (rows - 1) as f32 * 0.5);
    (0..n)
        .map(|i| [((i % cols) as f32 - cx) * SPACING, 0.0, ((i / cols) as f32 - cz) * SPACING])
        .collect()
}

/// Avatar model files (project-relative) to cycle through: `.glb`/`.vrm` files
/// under `models/avatars`, else the built-in template avatars when present.
pub fn avatar_models(root: &Path) -> Vec<String> {
    let mut found = Vec::new();
    let mut stack = vec![root.join("models/avatars")];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "glb" || x == "vrm")
                && let Ok(rel) = p.strip_prefix(root)
            {
                found.push(rel.to_string_lossy().replace('\\', "/"));
            }
        }
    }
    found.sort();
    if found.is_empty() {
        found = DEFAULT_MODELS.iter().filter(|m| root.join(m).is_file()).map(|m| m.to_string()).collect();
    }
    found
}

fn entry(name: String, source: ObjectSource, t: [f32; 3]) -> SceneEntry {
    SceneEntry {
        id: zestengine_core::ObjectId::new().to_string(),
        name,
        source,
        enabled: true,
        static_geometry: false,
        lightmap_scale: 1.0,
        cast_shadows: ShadowCasting::On,
        layer: CHARACTER_LAYER,
        material: MaterialRef::Inline {
            params: MaterialParams::default(),
            features: MaterialFeatures::default(),
            shader: "standard".into(),
            custom: Default::default(),
            render_queue: None,
            textures: Default::default(),
        },
        extra_materials: Vec::new(),
        renderer_split: false,
        parent: None,
        components: Vec::new(),
        translation: t,
        // Facing +Z like the template avatars (yaw by pi).
        rotation: [0.0, 1.0, 0.0, 0.0],
        scale: [1.0; 3],
        model: None,
        model_part: None,
        prefab: None,
        prefab_parent: None,
        net_role: Default::default(),
        component_roles: Default::default(),
    }
}

/// Append `n` avatar copies to `file` (see module docs). Returns how many were
/// added (0 when the project has no avatar model).
pub fn inject(file: &mut SceneFile, root: &Path, n: usize) -> usize {
    let models = avatar_models(root);
    if models.is_empty() {
        tracing::warn!(target: "frame_log", "ZESTENGINE_BENCH_AVATARS={n}: no avatar models under models/avatars");
        return 0;
    }
    let animated = root.join(CONTROLLER).is_file();
    for (i, pos) in grid_positions(n).into_iter().enumerate() {
        let path = &models[i % models.len()];
        // The built-in avatars carry a second primitive (joint-detail material).
        let extra = if path.contains("avatars/default/") { vec![1] } else { Vec::new() };
        let source = ObjectSource::Model {
            path: path.clone(),
            mesh: 0,
            extra_meshes: extra,
            model: None,
            mesh_id: None,
            extra_mesh_ids: Vec::new(),
        };
        let mut e = entry(format!("Bench Avatar {i}"), source, pos);
        if animated {
            let anim = Animator { controller: CONTROLLER.into(), ..Default::default() };
            e.components.push(zestengine_assets::ComponentData::of(&anim));
        }
        file.entries.push(e);
    }
    tracing::info!(
        target: "frame_log",
        "ZESTENGINE_BENCH_AVATARS: spawned {n} avatars ({} model file(s), animated={animated})",
        models.len()
    );
    n
}

/// One-second accumulator for the benchmark log line.
pub struct BenchLog {
    started: Instant,
    frames: u32,
    cpu_ms: f64,
    gpu_ms: f64,
    skin_ms: f64,
    morph_ms: f64,
    skin_max: f32,
}

impl Default for BenchLog {
    fn default() -> Self {
        Self { started: Instant::now(), frames: 0, cpu_ms: 0.0, gpu_ms: 0.0, skin_ms: 0.0, morph_ms: 0.0, skin_max: 0.0 }
    }
}

impl BenchLog {
    /// Fold one frame (`cpu_ms` = frame interval) and emit the per-second line
    /// to `frame_log` when a second has passed. Returns the line (tests).
    pub fn record(&mut self, n: usize, frame_ms: f32, s: &zestengine_render::RenderStats) -> Option<String> {
        self.frames += 1;
        self.cpu_ms += frame_ms as f64;
        self.gpu_ms += s.gpu_frame_ms as f64;
        self.skin_ms += s.gpu_skin_ms as f64;
        self.morph_ms += s.gpu_morph_ms as f64;
        self.skin_max = self.skin_max.max(s.gpu_skin_ms);
        if self.started.elapsed().as_secs_f32() < 1.0 {
            return None;
        }
        let line = self.line(n);
        tracing::info!(target: "frame_log", "{line}");
        *self = Self::default();
        Some(line)
    }

    fn line(&self, n: usize) -> String {
        let f = self.frames.max(1) as f64;
        format!(
            "bench avatars={n} frames={} frame_mean={:.2}ms gpu_frame_mean={:.3}ms gpu_skin_mean={:.3}ms (max {:.3}) gpu_morph_mean={:.3}ms",
            self.frames, self.cpu_ms / f, self.gpu_ms / f, self.skin_ms / f, self.skin_max, self.morph_ms / f
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_is_centred_and_complete() {
        let g = grid_positions(7);
        assert_eq!(g.len(), 7);
        // 3 columns x 3 rows: first row centred around x = 0.
        assert_eq!(g[0], [-SPACING, 0.0, -SPACING]);
        assert_eq!(g[2], [SPACING, 0.0, -SPACING]);
        assert!(grid_positions(0).is_empty());
        assert_eq!(grid_positions(1), vec![[0.0, 0.0, 0.0]]);
        let mut seen = g.clone();
        seen.dedup();
        assert_eq!(seen.len(), 7, "no two avatars share a slot");
    }

    #[test]
    fn inject_adds_n_models_cycling_files() {
        let dir = std::env::temp_dir().join(format!("zest_bench_av_{}", std::process::id()));
        let av = dir.join("models/avatars/a");
        std::fs::create_dir_all(&av).unwrap();
        std::fs::write(av.join("one.glb"), b"x").unwrap();
        std::fs::write(av.join("two.vrm"), b"x").unwrap();
        std::fs::write(av.join("note.txt"), b"x").unwrap();
        let mut file = SceneFile::default();
        assert_eq!(inject(&mut file, &dir, 5), 5);
        assert_eq!(file.entries.len(), 5);
        let paths: Vec<_> = file
            .entries
            .iter()
            .map(|e| match &e.source {
                ObjectSource::Model { path, .. } => path.clone(),
                _ => panic!("model expected"),
            })
            .collect();
        assert_eq!(paths[0], "models/avatars/a/one.glb");
        assert_eq!(paths[1], "models/avatars/a/two.vrm");
        assert_eq!(paths[2], paths[0]);
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(inject(&mut SceneFile::default(), &dir, 3), 0, "no models, nothing added");
    }

    #[test]
    fn log_line_averages_per_frame() {
        let mut b = BenchLog::default();
        let s = zestengine_render::RenderStats { gpu_frame_ms: 4.0, gpu_skin_ms: 1.0, ..Default::default() };
        b.frames = 0;
        for _ in 0..2 {
            b.frames += 1;
            b.cpu_ms += 10.0;
            b.gpu_ms += s.gpu_frame_ms as f64;
            b.skin_ms += s.gpu_skin_ms as f64;
        }
        let l = b.line(64);
        assert!(l.contains("avatars=64") && l.contains("frame_mean=10.00ms") && l.contains("gpu_skin_mean=1.000ms"), "{l}");
    }
}
