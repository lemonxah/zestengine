//! Stable, mesh-level Zestlet admission. Shared with the headless renderer.

use serde::{Deserialize, Serialize};

/// Project / engine setting: draw meshes that have a cluster DAG through the
/// GPU cluster path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClusterMode {
    #[default]
    Off,
    /// Automatic mesh admission; keeps the existing serialized project value `On`.
    On,
    /// Ignore the mesh benefit filter (diagnostics and A/B measurements).
    Forced,
}

impl ClusterMode {
    pub const ALL: [ClusterMode; 3] = [ClusterMode::Off, ClusterMode::On, ClusterMode::Forced];

    pub fn label(self) -> &'static str {
        match self {
            ClusterMode::Off => "Off",
            ClusterMode::On => "Automatic",
            ClusterMode::Forced => "Forced",
        }
    }
}

/// The coarsest cut is a measured property of the cooked DAG, including
/// disconnected components which never simplified. Count padded triangle slots:
/// every cluster invokes the vertex shader for 128 triangles, even when sparse.
/// This is a conservative admission heuristic, not a frame-time prediction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MeshBenefit {
    pub leaf_triangles: u64,
    pub root_clusters: u64,
    pub skinned: bool,
}

impl MeshBenefit {
    pub fn worthwhile(self) -> bool {
        // Small meshes rarely repay traversal and indirect-draw setup. Skinning
        // also duplicates work when other passes still require the ordinary mesh.
        let minimum = if self.skinned { 16_384 } else { 8_192 };
        self.leaf_triangles >= minimum && self.root_clusters > 0
            && self.root_clusters.saturating_mul(128 * 4) <= self.leaf_triangles
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cluster_selection_requires_size_and_actual_reduction() {
        let mesh = |leaf, roots, skinned| MeshBenefit { leaf_triangles: leaf, root_clusters: roots, skinned }.worthwhile();
        assert!(!mesh(8_191, 1, false));
        assert!(mesh(8_192, 16, false));
        assert!(!mesh(8_192, 17, false));
        assert!(!mesh(16_383, 1, true));
        assert!(mesh(16_384, 32, true));
        assert!(!mesh(1_000_000, 8_000, false), "unsimplifiable hard surface");
        assert!(mesh(1_000_000, 20, false));
        assert!(!mesh(100_000, 0, false), "empty root cut");
        assert!(!mesh(u32::MAX as u64, u64::MAX, false), "no overflow admission");
    }
}
