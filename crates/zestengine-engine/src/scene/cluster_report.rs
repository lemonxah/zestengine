//! The refusal report of the cluster path: which meshes of a model are kept off
//! it and why. Pure functions over plain facts (no renderer), so the rules are
//! unit-tested; `cluster_geo` gathers the facts when a model's DAGs are installed
//! and logs `ClusterCoverage::summary`.

use zestengine_core::{AlphaModeModel, MaterialModel, StencilModel};
use zestengine_render::{ClusterCoverage, ClusterRefusal};

/// What the install step learnt about one mesh of a model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshFacts {
    pub triangles: u64,
    /// The geometry cache holds a DAG for the mesh.
    pub has_dag: bool,
    /// The mesh has its own lightmap UV set (informational: it no longer refuses a mesh).
    pub lightmap_uv: bool,
    /// The DAG describes the installed mesh (same triangle count).
    pub dag_matches: bool,
    /// The renderer took the DAG.
    pub installed: bool,
}

/// Why the mesh stays on the ordinary path, `None` when it draws from clusters
/// (as far as the install can tell; `material` is the draw-time check).
pub fn classify(f: &MeshFacts, material: Option<ClusterRefusal>, min_triangles: usize) -> Option<ClusterRefusal> {
    if !f.has_dag {
        return Some(if (f.triangles as usize) < min_triangles { ClusterRefusal::TooSmall } else { ClusterRefusal::BuildFailed });
    }
    if !f.dag_matches {
        return Some(ClusterRefusal::StaleDag);
    }
    if !f.installed {
        return Some(ClusterRefusal::InstallRefused);
    }
    material
}

/// The draw-time material rules of the cluster path (`gpu_eligible` in the
/// renderer) that can be read off a material's settings. Per-object features
/// (fur, outline, water, baked lightmaps) are decided per draw and not seen here.
pub fn material_refusal(m: &MaterialModel) -> Option<ClusterRefusal> {
    if m.shader != "standard" && !m.shader.is_empty() {
        return Some(ClusterRefusal::CustomShader);
    }
    if matches!(m.alpha_mode, AlphaModeModel::Blend | AlphaModeModel::Transclipping) || m.render_queue >= 3000 || m.grab {
        return Some(ClusterRefusal::Transparent);
    }
    if m.tess.enabled {
        return Some(ClusterRefusal::Tessellation);
    }
    if m.stencil != StencilModel::default() {
        return Some(ClusterRefusal::Stencil);
    }
    None
}

/// Classify every mesh of a model into one coverage record.
pub fn coverage_of(meshes: impl IntoIterator<Item = (MeshFacts, Option<ClusterRefusal>)>, min_triangles: usize) -> ClusterCoverage {
    let mut c = ClusterCoverage::default();
    for (facts, material) in meshes {
        c.add(facts.triangles, classify(&facts, material, min_triangles));
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use ClusterRefusal::*;

    const MIN: usize = 2000;

    fn ok(triangles: u64) -> MeshFacts {
        MeshFacts { triangles, has_dag: true, lightmap_uv: false, dag_matches: true, installed: true }
    }

    #[test]
    fn a_plain_installed_mesh_is_on_clusters() {
        assert_eq!(classify(&ok(5000), None, MIN), None);
    }

    #[test]
    fn meshes_without_a_dag_are_too_small_or_failed_builds() {
        let no_dag = |t| MeshFacts { has_dag: false, ..ok(t) };
        assert_eq!(classify(&no_dag(1999), None, MIN), Some(TooSmall));
        assert_eq!(classify(&no_dag(2000), None, MIN), Some(BuildFailed));
    }

    #[test]
    fn vertex_colours_and_lightmap_uv_sets_do_not_refuse() {
        // Vertex colours travel in the cluster vertex (builder version 3), and a lightmap UV set was never a reason.
        assert_eq!(classify(&MeshFacts { lightmap_uv: true, ..ok(9000) }, None, MIN), None);
    }

    #[test]
    fn stale_dags_and_refused_installs_come_before_the_material_rules() {
        assert_eq!(classify(&MeshFacts { dag_matches: false, ..ok(9000) }, Some(CustomShader), MIN), Some(StaleDag));
        assert_eq!(classify(&MeshFacts { installed: false, ..ok(9000) }, Some(CustomShader), MIN), Some(InstallRefused));
        assert_eq!(classify(&ok(9000), Some(CustomShader), MIN), Some(CustomShader), "an installed mesh with a custom shader still draws ordinary");
    }

    #[test]
    fn material_settings_map_to_the_renderers_eligibility_rules() {
        let base = super::super::model_from_material("m", &zestengine_render::MaterialParams::default(), &zestengine_render::MaterialFeatures::default(), false);
        assert_eq!(material_refusal(&base), None);
        assert_eq!(material_refusal(&MaterialModel { shader: "shaders/water.frag".into(), ..base.clone() }), Some(CustomShader));
        assert_eq!(material_refusal(&MaterialModel { alpha_mode: AlphaModeModel::Blend, ..base.clone() }), Some(Transparent));
        assert_eq!(material_refusal(&MaterialModel { alpha_mode: AlphaModeModel::Cutout, ..base.clone() }), None, "alpha test is fine");
        assert_eq!(material_refusal(&MaterialModel { render_queue: 3000, ..base.clone() }), Some(Transparent));
        assert_eq!(material_refusal(&MaterialModel { grab: true, ..base.clone() }), Some(Transparent));
        let mut tess = base.clone();
        tess.tess.enabled = true;
        assert_eq!(material_refusal(&tess), Some(Tessellation));
        let mut st = base.clone();
        st.stencil.reference = 3;
        assert_eq!(material_refusal(&st), Some(Stencil));
    }

    #[test]
    fn a_model_is_summed_by_triangles() {
        let colour = MeshFacts { installed: false, ..ok(60_000) };
        let c = coverage_of(
            [(ok(30_000), None), (colour, None), (colour, None), (MeshFacts { has_dag: false, ..ok(100) }, None), (ok(10_000), Some(CustomShader))],
            MIN,
        );
        assert_eq!((c.meshes_on, c.tris_on), (1, 30_000));
        assert_eq!((c.meshes_off[InstallRefused.index()], c.tris_off[InstallRefused.index()]), (2, 120_000));
        assert_eq!(c.meshes_off[TooSmall.index()], 1);
        assert_eq!(c.tris_total(), 160_100);
        assert!(c.mostly_ordinary());
        assert!(c.summary().contains("2 install refused (75% of tris)"), "{}", c.summary());
    }
}
