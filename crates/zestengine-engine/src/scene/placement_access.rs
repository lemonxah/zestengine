//! Read access to the CPU mesh data (positions, indices, bounds) for the asset
//! placement snap rays (`crate::placement`). A child of `scene` so the private
//! mesh arrays stay private.

use glam::Vec3;

use super::LoadedScene;

/// CPU mesh geometry: (positions, triangle indices, object-space bounds).
type MeshCpu<'a> = (&'a [Vec3], &'a [u32], (Vec3, Vec3));

impl LoadedScene {
    /// CPU geometry of scene mesh `mesh`: (positions, triangle indices, object-space bounds).
    pub(crate) fn mesh_cpu(&self, mesh: usize) -> Option<MeshCpu<'_>> {
        let (pos, idx) = self.mesh_geometry.get(mesh)?;
        let bounds = *self.mesh_bounds.get(mesh)?;
        Some((pos.as_slice(), idx.as_slice(), bounds))
    }
}
