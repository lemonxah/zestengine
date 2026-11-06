//! What share of a scene's triangles the cluster path actually draws, and why
//! the rest stays on the ordinary path (the "refusal report").
//!
//! Pure data and formatting, no GPU: the engine classifies every mesh of a model
//! when its cluster DAGs are installed ([`ClusterCoverage::add`]), logs
//! [`ClusterCoverage::summary`], and hands the scene total to the renderer so it
//! shows in the render statistics. Compiled by `zestengine-render-lite` too.

/// Why a mesh is kept off the cluster path. Order = display order of ties.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ClusterRefusal {
    /// Under the DAG threshold (`GEO_MIN_TRIANGLES`): not worth a DAG.
    TooSmall,
    /// The draw carries a baked lightmap, which the Zestlet path does not sample, so that object draws ordinary
    /// (a lightmap UV set on its own no longer refuses a mesh; the install step never reports this reason, the
    /// coverage view finds it per draw).
    LightmapUvs,
    /// The builder gave up (bone / blendshape limits, degenerate or invalid geometry).
    BuildFailed,
    /// The cooked DAG does not describe the mesh that was installed (stale cache).
    StaleDag,
    /// The renderer refused the install (pool limits, device).
    InstallRefused,
    /// The mesh's material uses a custom shader (includes vertex hooks).
    CustomShader,
    /// Alpha blended / transparent queue / scene colour read.
    Transparent,
    /// Tessellated displacement.
    Tessellation,
    /// Stencil writer or reader.
    Stencil,
    /// Big enough for a DAG, but none is installed yet and no refusal is recorded: the geometry cache is
    /// still loading / building (or the model's worker has not reached this mesh). Transient.
    Pending,
    /// A skinned DAG held back to the ordinary skinned draw because the skinned-vertex pool is full
    /// (`SkinAdmission`: the farthest skinned instances give way first).
    SkinPoolFull,
    /// A skinned DAG with no pose uploaded yet (first frames after placement); transient.
    NoPose,
    /// Cloth simulation is rewriting this mesh's vertices this frame.
    ClothSim,
    /// A per-object feature keeps the draw off the batches (outline, fur, water, scene-colour read, shadow
    /// casting off, ...) although the mesh has a DAG.
    DrawFeature,
}

/// Number of [`ClusterRefusal`] values.
pub const REFUSAL_REASONS: usize = 14;

impl ClusterRefusal {
    pub const ALL: [ClusterRefusal; REFUSAL_REASONS] = [
        Self::TooSmall,
        Self::LightmapUvs,
        Self::BuildFailed,
        Self::StaleDag,
        Self::InstallRefused,
        Self::CustomShader,
        Self::Transparent,
        Self::Tessellation,
        Self::Stencil,
        Self::Pending,
        Self::SkinPoolFull,
        Self::NoPose,
        Self::ClothSim,
        Self::DrawFeature,
    ];

    pub fn index(self) -> usize {
        self as usize
    }

    /// Short noun phrase for the log and the stats panel.
    pub fn label(self) -> &'static str {
        match self {
            Self::TooSmall => "too small",
            Self::LightmapUvs => "baked lightmap",
            Self::BuildFailed => "DAG build failed",
            Self::StaleDag => "stale DAG",
            Self::InstallRefused => "install refused",
            Self::CustomShader => "custom shader",
            Self::Transparent => "transparent",
            Self::Tessellation => "tessellation",
            Self::Stencil => "stencil",
            Self::Pending => "DAG pending",
            Self::SkinPoolFull => "skin pool full",
            Self::NoPose => "waiting for pose",
            Self::ClothSim => "cloth simulated",
            Self::DrawFeature => "draw feature",
        }
    }

    /// What would put such a mesh on the cluster path.
    pub fn hint(self) -> &'static str {
        match self {
            Self::TooSmall => "meshes under 2000 triangles never get a DAG",
            Self::LightmapUvs => "an object with a baked lightmap draws on the ordinary path (the Zestlet path pulls no second UV set); a lightmap UV set alone no longer keeps a mesh off Zestlets",
            Self::BuildFailed => "see the 'Zestlet DAG for mesh' warnings",
            Self::StaleDag => "the stored geometry cache is out of date: rebuild it from the model inspector",
            Self::InstallRefused => "the Zestlet pools are full or the device refused (see the debug log)",
            Self::CustomShader => "custom shaders always draw on the ordinary path",
            Self::Transparent => "blended and transparent materials draw on the ordinary path",
            Self::Tessellation => "tessellated materials draw on the ordinary path",
            Self::Stencil => "stencil materials draw on the ordinary path",
            Self::Pending => "the geometry cache for this model is still loading or building; it switches to zestlets when the DAG installs",
            Self::SkinPoolFull => "too many skinned zestlet instances for the skinned-vertex pool: the farthest draw ordinary (raise the pool or cull the avatars)",
            Self::NoPose => "the skinned mesh has not received its first pose yet; transient",
            Self::ClothSim => "the mesh is cloth-simulated this frame, which rewrites its vertices on the CPU",
            Self::DrawFeature => "an outline, fur, water, grab-pass or shadow-off setting keeps this object off the zestlet batches",
        }
    }
}

// ---- Zestlet viewport modes (the shading dropdown's `gi_debug` numbers) -------------------------------------

/// Viewport shading: every zestlet's coarse level (the older "Cluster LOD" mode, number kept).
pub const VIEW_ZESTLET_LOD: u32 = 8;
/// Viewport shading: every zestlet its own stable colour, ordinary meshes flat grey.
pub const VIEW_ZESTLETS: u32 = 9;
/// Viewport shading: zestlet meshes green, ordinary meshes in their refusal colour.
pub const VIEW_ZESTLET_COVERAGE: u32 = 10;
/// Viewport shading: a colour per triangle id.
pub const VIEW_ZESTLET_TRIANGLES: u32 = 11;

/// The zestlet entries of the viewport shading dropdown: (mode number, label).
pub const ZESTLET_VIEW_MODES: [(u32, &str); 4] = [
    (VIEW_ZESTLETS, "Zestlets"),
    (VIEW_ZESTLET_LOD, "Zestlet LOD"),
    (VIEW_ZESTLET_COVERAGE, "Zestlet coverage"),
    (VIEW_ZESTLET_TRIANGLES, "Zestlet triangles"),
];

/// Whether viewport mode `mode` is one of the zestlet views (the renderer then feeds the ordinary
/// draws their refusal colour; every other mode costs nothing extra).
pub fn is_zestlet_view(mode: u32) -> bool {
    (VIEW_ZESTLET_LOD..=VIEW_ZESTLET_TRIANGLES).contains(&mode)
}

/// Meshes below this many triangles never get a DAG (`GEO_MIN_TRIANGLES` in the assets crate, which
/// has a test that the two agree); the coverage view calls an unclassified mesh under it "too small".
pub const ZESTLET_MIN_TRIANGLES: u32 = 2000;

/// Flat colour of the ordinary path in the Zestlets / LOD views.
pub const ORDINARY_GREY: [f32; 3] = [0.35, 0.35, 0.35];
/// Coverage view: a mesh drawn as zestlets.
pub const COVERAGE_ZESTLET: [f32; 3] = [0.1, 0.8, 0.25];
/// Coverage view: on the ordinary path for a reason nothing recorded. Every known path records one
/// ([`ClusterRefusal::Pending`] for "not built yet"), so seeing this colour means a new, unclassified path.
pub const COVERAGE_OTHER: [f32; 3] = [0.5, 0.5, 0.5];

/// The id hash of `standard_common.glsl::zestlet_hash_colour`.
pub fn zestlet_hash(id: u32) -> u32 {
    let mut h = id.wrapping_mul(2654435761);
    h ^= h >> 15;
    h = h.wrapping_mul(2246822519);
    h ^= h >> 13;
    h = h.wrapping_mul(3266489917);
    h ^ (h >> 16)
}

fn palette(t: f32) -> [f32; 3] {
    [0.0f32, 0.33, 0.67].map(|o| 0.5 + 0.5 * (std::f32::consts::TAU * (t + o)).cos())
}

/// `zestlet_hash_colour`: the stable colour of an id (any `u32`).
pub fn zestlet_hash_colour(id: u32) -> [f32; 3] {
    palette((zestlet_hash(id) & 1023) as f32 / 1024.0)
}

/// The id the Zestlets view hashes for cluster `cluster` of instance `instance` (clustered `standard.vert`).
pub fn zestlet_id(instance: u32, cluster: u32) -> u32 {
    cluster.wrapping_mul(2246822519) ^ instance.wrapping_mul(2654435761)
}

/// `zestlet_level_colour`: the colour of DAG level `level` in the Zestlet LOD view.
pub fn zestlet_level_colour(level: u32) -> [f32; 3] {
    palette((level as f32 * 0.137 + 0.05).fract())
}

impl ClusterRefusal {
    /// The coverage view's colour of a mesh on the ordinary path for this reason (also the legend swatch).
    pub fn view_colour(self) -> [f32; 3] {
        match self {
            Self::TooSmall => [0.88, 0.88, 0.92],
            Self::LightmapUvs => [0.95, 0.85, 0.15],
            Self::BuildFailed => [0.95, 0.15, 0.15],
            Self::StaleDag => [0.95, 0.55, 0.1],
            Self::InstallRefused => [0.6, 0.15, 0.8],
            Self::CustomShader => [0.2, 0.75, 0.95],
            Self::Transparent => [0.3, 0.45, 0.95],
            Self::Tessellation => [0.1, 0.8, 0.75],
            Self::Stencil => [0.7, 0.9, 0.2],
            Self::Pending => [0.6, 0.4, 0.2],
            Self::SkinPoolFull => [0.25, 0.05, 0.5],
            Self::NoPose => [0.1, 0.2, 0.55],
            Self::ClothSim => [0.95, 0.5, 0.8],
            Self::DrawFeature => [0.4, 0.1, 0.15],
        }
    }
}

/// One row of the coverage view's legend.
#[derive(Clone, Debug, PartialEq)]
pub struct LegendRow {
    pub label: String,
    pub colour: [f32; 3],
    /// Share of the scene's classified triangles, `None` for the catch-all row.
    pub share: Option<f32>,
}

/// Meshes and triangles on the cluster path and, per reason, off it. `Copy` so
/// it rides in `RenderStats`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClusterCoverage {
    pub meshes_on: u32,
    pub tris_on: u64,
    pub meshes_off: [u32; REFUSAL_REASONS],
    pub tris_off: [u64; REFUSAL_REASONS],
}

impl ClusterCoverage {
    /// Count one mesh of `triangles` triangles: drawn as Zestlets (`None`) or refused for a reason.
    pub fn add(&mut self, triangles: u64, refusal: Option<ClusterRefusal>) {
        match refusal {
            None => {
                self.meshes_on += 1;
                self.tris_on += triangles;
            }
            Some(r) => {
                self.meshes_off[r.index()] += 1;
                self.tris_off[r.index()] += triangles;
            }
        }
    }

    /// The legend of the Zestlet coverage view: Zestlets first, then every reason that holds triangles
    /// (largest first), then the "other" catch-all for meshes nothing classified.
    pub fn legend(&self) -> Vec<LegendRow> {
        let mut rows = vec![LegendRow { label: "Zestlets".into(), colour: COVERAGE_ZESTLET, share: Some(self.on_share() as f32) }];
        for r in self.reasons_by_triangles() {
            if self.tris_off[r.index()] > 0 {
                rows.push(LegendRow { label: r.label().to_owned(), colour: r.view_colour(), share: Some(share(self.tris_off[r.index()], self.tris_total()) as f32) });
            }
        }
        rows.push(LegendRow { label: "other".into(), colour: COVERAGE_OTHER, share: None });
        rows
    }

    /// Add another model's numbers.
    pub fn merge(&mut self, other: &ClusterCoverage) {
        self.meshes_on += other.meshes_on;
        self.tris_on += other.tris_on;
        for i in 0..REFUSAL_REASONS {
            self.meshes_off[i] += other.meshes_off[i];
            self.tris_off[i] += other.tris_off[i];
        }
    }

    pub fn meshes_off_total(&self) -> u32 {
        self.meshes_off.iter().sum()
    }

    pub fn tris_off_total(&self) -> u64 {
        self.tris_off.iter().sum()
    }

    pub fn tris_total(&self) -> u64 {
        self.tris_on + self.tris_off_total()
    }

    pub fn is_empty(&self) -> bool {
        self.meshes_on == 0 && self.meshes_off_total() == 0
    }

    /// Share (0..=1) of all triangles drawn from clusters.
    pub fn on_share(&self) -> f64 {
        share(self.tris_on, self.tris_total())
    }

    /// Share (0..=1) of all triangles on the ordinary path.
    pub fn off_share(&self) -> f64 {
        if self.tris_total() == 0 { 0.0 } else { 1.0 - self.on_share() }
    }

    /// Reasons present, largest triangle count first (ties: declaration order).
    pub fn reasons_by_triangles(&self) -> Vec<ClusterRefusal> {
        let mut v: Vec<ClusterRefusal> = ClusterRefusal::ALL.iter().copied().filter(|r| self.meshes_off[r.index()] > 0).collect();
        v.sort_by(|a, b| self.tris_off[b.index()].cmp(&self.tris_off[a.index()]).then(a.index().cmp(&b.index())));
        v
    }

    /// The one-line log summary of a model, e.g. `64 meshes drawn as Zestlets (31% of
    /// 20.1M triangles); 87 meshes on the ordinary path: 61 DAG build failed (52%
    /// of tris), 20 too small (4%)`.
    pub fn summary(&self) -> String {
        let total = self.tris_total();
        let mut s = format!("{} drawn as Zestlets ({:.0}% of {} triangles)", meshes(self.meshes_on), self.on_share() * 100.0, count(total));
        let off = self.meshes_off_total();
        if off == 0 {
            return s;
        }
        s += &format!("; {} on the ordinary path: ", meshes(off));
        let parts: Vec<String> = self
            .reasons_by_triangles()
            .into_iter()
            .enumerate()
            .map(|(k, r)| {
                let pct = share(self.tris_off[r.index()], total) * 100.0;
                let n = self.meshes_off[r.index()];
                if k == 0 { format!("{n} {} ({pct:.0}% of tris)", r.label()) } else { format!("{n} {} ({pct:.0}%)", r.label()) }
            })
            .collect();
        s += &parts.join(", ");
        s
    }

    /// Whether Zestlets will do little for this scene: more than half of the
    /// triangles stay on the ordinary path.
    pub fn mostly_ordinary(&self) -> bool {
        self.off_share() > 0.5
    }

    /// The warning for [`mostly_ordinary`](Self::mostly_ordinary), naming the main reason.
    pub fn warning(&self) -> Option<String> {
        if !self.mostly_ordinary() {
            return None;
        }
        let top = self.reasons_by_triangles().into_iter().next()?;
        Some(format!(
            "{:.0}% of the triangles stay on the ordinary path, so Zestlets will not help this scene; mostly {} ({:.0}% of triangles): {}",
            self.off_share() * 100.0,
            top.label(),
            share(self.tris_off[top.index()], self.tris_total()) * 100.0,
            top.hint()
        ))
    }

    /// Lines for the render-stats panel: nothing before any model is classified.
    pub fn panel_lines(&self) -> Vec<String> {
        if self.is_empty() {
            return Vec::new();
        }
        let mut lines = vec![format!("Coverage: {} drawn as Zestlets ({:.0}% of {} triangles)", meshes(self.meshes_on), self.on_share() * 100.0, count(self.tris_total()))];
        for r in self.reasons_by_triangles() {
            lines.push(format!(
                "  {} {}: {} triangles ({:.0}%)",
                self.meshes_off[r.index()],
                r.label(),
                count(self.tris_off[r.index()]),
                share(self.tris_off[r.index()], self.tris_total()) * 100.0
            ));
        }
        lines
    }
}

fn share(part: u64, whole: u64) -> f64 {
    if whole == 0 { 0.0 } else { part as f64 / whole as f64 }
}

fn meshes(n: u32) -> String {
    format!("{n} mesh{}", if n == 1 { "" } else { "es" })
}

/// 20 100 000 -> "20.1M", 4 200 -> "4.2k".
pub fn count(n: u64) -> String {
    match n {
        0..=9_999 => n.to_string(),
        10_000..=999_999 => format!("{:.1}k", n as f64 / 1e3),
        _ => format!("{:.1}M", n as f64 / 1e6),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ClusterRefusal::*;

    fn sample() -> ClusterCoverage {
        let mut c = ClusterCoverage::default();
        // 31% drawn as Zestlets: 64 meshes x 97 656 = ~6.25M of 20.1M.
        for _ in 0..64 {
            c.add(97_656, None);
        }
        for _ in 0..61 {
            c.add(171_000, Some(BuildFailed));
        }
        for _ in 0..20 {
            c.add(40_000, Some(TooSmall));
        }
        for _ in 0..6 {
            c.add(400_000, Some(CustomShader));
        }
        c
    }

    #[test]
    fn counts_and_shares_add_up() {
        let c = sample();
        assert_eq!(c.meshes_on, 64);
        assert_eq!(c.meshes_off_total(), 87);
        assert_eq!(c.tris_total(), c.tris_on + c.tris_off_total());
        assert!((c.on_share() + c.off_share() - 1.0).abs() < 1e-12);
        assert_eq!(c.reasons_by_triangles(), vec![BuildFailed, CustomShader, TooSmall]);
    }

    #[test]
    fn the_summary_names_each_reason_with_its_triangle_share() {
        let s = sample().summary();
        assert!(s.starts_with("64 meshes drawn as Zestlets (31% of 19.9M triangles); 87 meshes on the ordinary path: 61 DAG build failed (52% of tris), 6 custom shader (12%), 20 too small (4%)"), "{s}");
    }

    #[test]
    fn a_fully_clustered_model_has_no_ordinary_part_and_no_warning() {
        let mut c = ClusterCoverage::default();
        c.add(5000, None);
        assert_eq!(c.summary(), "1 mesh drawn as Zestlets (100% of 5000 triangles)");
        assert!(c.warning().is_none() && !c.mostly_ordinary());
    }

    #[test]
    fn the_warning_fires_above_half_the_triangles_only() {
        let c = sample();
        assert!(c.mostly_ordinary(), "{}%", c.off_share() * 100.0);
        let w = c.warning().unwrap();
        assert!(w.contains("will not help") && w.contains("DAG build failed") && w.contains("69%"), "{w}");
        let mut half = ClusterCoverage::default();
        half.add(100, None);
        half.add(100, Some(Stencil));
        assert!(!half.mostly_ordinary(), "exactly half is not more than half");
        half.add(1, Some(Stencil));
        assert!(half.mostly_ordinary());
        assert!(ClusterCoverage::default().warning().is_none());
    }

    #[test]
    fn merging_models_sums_every_counter() {
        let (a, b) = (sample(), sample());
        let mut m = a;
        m.merge(&b);
        assert_eq!(m.meshes_on, 128);
        assert_eq!(m.tris_off[BuildFailed.index()], 2 * a.tris_off[BuildFailed.index()]);
        assert_eq!(m.summary().split(';').next().unwrap(), "128 meshes drawn as Zestlets (31% of 39.8M triangles)");
    }

    #[test]
    fn panel_lines_are_empty_until_something_is_classified() {
        assert!(ClusterCoverage::default().panel_lines().is_empty());
        let lines = sample().panel_lines();
        assert_eq!(lines.len(), 4);
        assert!(lines[0].starts_with("Coverage: 64 meshes drawn as Zestlets"));
        assert!(lines[1].contains("61 DAG build failed"));
    }

    #[test]
    fn every_reason_has_a_distinct_slot_and_text() {
        let mut seen = std::collections::HashSet::new();
        for (i, r) in ClusterRefusal::ALL.iter().enumerate() {
            assert_eq!(r.index(), i);
            assert!(seen.insert(r.label()) && !r.hint().is_empty());
        }
        assert_eq!(count(20_100_000), "20.1M");
        assert_eq!(count(4_200), "4200");
        assert_eq!(count(42_000), "42.0k");
    }

    #[test]
    fn view_mode_numbers_are_stable_and_listed() {
        assert_eq!((VIEW_ZESTLET_LOD, VIEW_ZESTLETS, VIEW_ZESTLET_COVERAGE, VIEW_ZESTLET_TRIANGLES), (8, 9, 10, 11));
        for (m, _) in ZESTLET_VIEW_MODES {
            assert!(is_zestlet_view(m));
        }
        assert!(!is_zestlet_view(0) && !is_zestlet_view(7) && !is_zestlet_view(12));
    }

    #[test]
    fn hash_colours_are_stable_distinct_and_in_range() {
        // Fixed values: a colour that drifts would silently change every saved screenshot of the view.
        assert_eq!(zestlet_hash(0), 0);
        assert_eq!(zestlet_hash(1), zestlet_hash(1));
        let mut seen = std::collections::HashSet::new();
        for id in 0..400u32 {
            let c = zestlet_hash_colour(id);
            assert!(c.iter().all(|v| (0.0..=1.0).contains(v)), "{c:?}");
            seen.insert(zestlet_hash(id) & 1023);
        }
        assert!(seen.len() > 300, "400 ids landed on only {} hues", seen.len());
        assert_ne!(zestlet_id(0, 1), zestlet_id(1, 0));
        assert_ne!(zestlet_id(3, 5), zestlet_id(3, 6));
    }

    #[test]
    fn level_colours_differ_per_level() {
        let cols: Vec<_> = (0..12).map(zestlet_level_colour).collect();
        for a in 0..cols.len() {
            for b in a + 1..cols.len() {
                let d: f32 = (0..3).map(|k| (cols[a][k] - cols[b][k]).abs()).sum();
                assert!(d > 0.12, "levels {a} and {b} are too alike: {:?} {:?}", cols[a], cols[b]);
            }
        }
    }

    #[test]
    fn every_refusal_has_its_own_colour_apart_from_zestlet_green_and_other() {
        let mut all: Vec<[f32; 3]> = ClusterRefusal::ALL.iter().map(|r| r.view_colour()).collect();
        all.push(COVERAGE_ZESTLET);
        all.push(COVERAGE_OTHER);
        for a in 0..all.len() {
            for b in a + 1..all.len() {
                let d: f32 = (0..3).map(|k| (all[a][k] - all[b][k]).abs()).sum();
                assert!(d > 0.25, "palette entries {a} and {b} are too alike: {:?} {:?}", all[a], all[b]);
            }
        }
    }

    #[test]
    fn every_refusal_is_labelled_and_explained_and_not_a_silent_grey() {
        let mut labels = std::collections::HashSet::new();
        for r in ClusterRefusal::ALL {
            assert!(labels.insert(r.label()), "duplicate label {}", r.label());
            assert!(!r.hint().is_empty());
            assert_ne!(r.view_colour(), COVERAGE_OTHER);
            assert_eq!(ClusterRefusal::ALL[r.index()], r, "ALL is in declaration order");
        }
        assert_eq!(ClusterRefusal::ALL.len(), REFUSAL_REASONS);
        // "Not built yet" has its own reason, so an unbuilt mesh is never the catch-all grey.
        assert_eq!(ClusterRefusal::Pending.label(), "DAG pending");
    }

    #[test]
    fn the_legend_lists_zestlets_the_present_reasons_by_size_and_other() {
        let mut c = ClusterCoverage::default();
        c.add(600, None);
        c.add(100, Some(TooSmall));
        c.add(300, Some(CustomShader));
        c.add(0, Some(Stencil));
        let rows = c.legend();
        let labels: Vec<&str> = rows.iter().map(|r| r.label.as_str()).collect();
        assert_eq!(labels, ["Zestlets", "custom shader", "too small", "other"]);
        assert!((rows[0].share.unwrap() - 0.6).abs() < 1e-6);
        assert!((rows[1].share.unwrap() - 0.3).abs() < 1e-6);
        assert_eq!(rows[1].colour, CustomShader.view_colour());
        assert_eq!(rows[3].share, None);
        // An empty scene still lists the two fixed rows, with a zero share.
        let empty = ClusterCoverage::default().legend();
        assert_eq!(empty.len(), 2);
        assert_eq!(empty[0].share, Some(0.0));
    }
}
