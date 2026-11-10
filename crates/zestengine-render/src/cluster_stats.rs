/// Statistics of the cluster path (read back from the frame slot's previous
/// submission, no CPU wait).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ClusterStats {
    /// The path is requested and available.
    pub enabled: bool,
    /// Off for a few frames after an overflow.
    pub paused: bool,
    pub meshes: u32,
    /// Installed mesh handles selected/rejected by the benefit policy (includes aliases).
    pub automatic: bool,
    pub selected_meshes: u32,
    pub ordinary_meshes: u32,
    pub memory: GeometryMemory,
    pub geometry_budget_bytes: u64,
    pub memory_denials: u64,
    pub pool_trims: u32,
    pub memory_paused: bool,
    pub batches: u32,
    /// Instances entering the pass / rejected by the instance frustum test.
    pub instances: u32,
    pub instances_culled: u32,
    pub nodes_visited: u32,
    pub clusters_tested: u32,
    /// Tested clusters that are not in the LOD cut.
    pub culled_lod: u32,
    pub culled_frustum: u32,
    pub culled_cone: u32,
    pub culled_hiz: u32,
    /// Hi-Z rejects of the early pass that the late pass drew after all.
    pub recovered: u32,
    pub visible_clusters: u32,
    /// Triangles submitted to the vertex stage (before the hardware culls).
    pub triangles: u32,
    /// Frames in which a queue or list overflowed (and the path paused).
    pub queue_overflows: u32,
    /// Fullest of the four pool arrays (percent).
    pub pool_percent: u32,
    /// Cluster shadows: caster instances culled against the shadow views, and
    /// the clusters / triangles their (coarser) cuts drew into the shadow maps.
    /// Skinned meshes whose ordinary GPU skin pass was skipped this frame (cluster shadows + camera).
    pub skin_skipped: u32,
    /// Skinned instances that drew the ordinary skinned mesh last frame because the skinned-vertex pool was under pressure, and how many are held back.
    pub skin_fallbacks: u32,
    pub skin_denied: u32,
    pub shadow_instances: u32,
    pub shadow_clusters: u32,
    pub shadow_triangles: u32,
    /// Cluster cull plus the skin pass (one GPU zone).
    pub cull_ms: f32,
    /// The main pass's cluster draws (early + late batches).
    pub draw_ms: f32,
    /// The depth prepass's cluster draws.
    pub prepass_ms: f32,
    /// Posing the visible skinned clusters (early + late halves; inside `cull_ms`).
    pub skin_ms: f32,
    /// Skinned clusters posed last frame, and posed-pool vertices reserved.
    pub skinned_clusters: u32,
    pub skin_vertices: u32,
    /// Frames in which the skinned-vertex pool was full.
    pub skin_pool_overflows: u32,
    /// Global LOD error scale the budget controller applies (1 = none), and
    /// the frame time it aims for (ms; 0 = controller off).
    pub error_scale: f32,
    pub target_ms: f32,
    /// Estimated triangles of the instances at that scale, the limit they are
    /// shared out of (0 = none) and how many instances were coarsened further
    /// by the fair share, and the largest scale any instance got.
    pub budget_estimated: u32,
    pub budget_limit: u32,
    pub fair_clamped: u32,
    pub max_instance_scale: f32,
    /// Geometry streaming (GEO-6): on, resident / loading pages (resident bytes
    /// count the pages outside the root pages), the budget, the page requests of
    /// the policy, loads, evictions, bytes streamed, frames a wanted page could
    /// not be admitted, requests waiting, and, from the last cull, clusters on
    /// missing pages and clusters standing in for missing children.
    pub stream_on: bool,
    pub stream_pages: u32,
    pub stream_loading: u32,
    pub stream_resident_mb: f32,
    pub stream_budget_mb: f32,
    pub stream_requests: u64,
    /// Of `stream_requests`, the ones only shadow views asked for.
    pub stream_shadow_requests: u64,
    pub stream_loads: u64,
    pub stream_evictions: u64,
    pub stream_mb_streamed: f32,
    pub stream_stall_frames: u64,
    pub stream_pending: u32,
    pub stream_skipped: u32,
    pub stream_standins: u32,
    /// Pages loaded again soon after an eviction.
    pub stream_thrash: u64,
    /// Which meshes draw from clusters and why the rest do not (set by the engine).
    pub coverage: crate::ClusterCoverage,
}

/// Renderer-owned geometry buffer allocations (GPU and host-visible), in bytes.
/// Excludes images, pipelines, descriptor pools, acceleration structures, allocator
/// block slack, CPU copies and uploads still owned by background workers. This is
/// buffer accounting, not a device-wide VRAM meter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GeometryMemory {
    /// All installed ordinary vertex/index buffers, including retained fallbacks.
    pub ordinary_meshes: u64,
    /// Ordinary bind poses, palettes, previous positions, morphs and cloth patches.
    pub ordinary_skin: u64,
    /// Ordinary culling/indirect/instance buffers across desktop and XR slots.
    pub ordinary_cull: u64,
    /// Ordinary meshes and skin buffers waiting for in-flight frames to retire.
    pub ordinary_retired: u64,
    /// All seven Zestlet pools, including unused capacity and streamed payload blocks.
    pub pools: u64,
    /// Reserved pool ranges, a subset of `pools`; streamed blocks include spare space.
    pub pool_claimed: u64,
    /// Cull queues, instances, commands and readbacks across every frame/eye slot.
    pub frames: u64,
    /// Posed vertex pools, slots, palettes and skin-dispatch buffers for all slots.
    pub skin: u64,
    /// Stream page table and request/touch buffers (payload is already in `pools`).
    pub stream_control: u64,
    /// Zestlet staging and replaced pools awaiting retirement; includes host-visible memory.
    pub transient: u64,
    /// Informational payload subsets, NOT added to allocation totals below.
    pub stream_roots: u64,
    pub stream_resident: u64,
    pub stream_loading: u64,
}

impl GeometryMemory {
    pub fn zestlet_allocated(self) -> u64 {
        self.pools + self.frames + self.skin + self.stream_control + self.transient
    }

    pub fn total_allocated(self) -> u64 {
        self.zestlet_allocated() + self.ordinary_meshes + self.ordinary_skin + self.ordinary_cull + self.ordinary_retired
    }

    pub fn panel_lines(self) -> [String; 4] {
        let mb = |n: u64| n as f64 / 1_048_576.0;
        [
            format!("Geometry buffers: {:.1} MiB allocated (Zestlets {:.1}, ordinary {:.1})", mb(self.total_allocated()), mb(self.zestlet_allocated()), mb(self.total_allocated() - self.zestlet_allocated())),
            format!("Zestlet pools: {:.1} MiB allocated, {:.1} claimed; frame buffers {:.1}, skin {:.1}", mb(self.pools), mb(self.pool_claimed), mb(self.frames), mb(self.skin)),
            format!("Stream control {:.1} MiB; Zestlet staging/retired {:.1}; ordinary meshes {:.1}, skin {:.1}, cull {:.1}, retired {:.1}", mb(self.stream_control), mb(self.transient), mb(self.ordinary_meshes), mb(self.ordinary_skin), mb(self.ordinary_cull), mb(self.ordinary_retired)),
            format!("Stream payload (already in pools): roots {:.1} MiB, resident detail {:.1}; loading demand {:.1}", mb(self.stream_roots), mb(self.stream_resident), mb(self.stream_loading)),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cluster_memory_totals_count_allocations_once() {
        let mut m = GeometryMemory { ordinary_meshes: 1, ordinary_skin: 2, ordinary_cull: 3, ordinary_retired: 4,
            pools: 10, frames: 20, skin: 30, stream_control: 40, transient: 50, ..Default::default() };
        assert_eq!(m.zestlet_allocated(), 150);
        assert_eq!(m.total_allocated(), 160);
        m.pool_claimed = 9;
        m.stream_roots = 4;
        m.stream_resident = 3;
        m.stream_loading = 100;
        assert_eq!(m.total_allocated(), 160, "payload subsets and pending demand are not extra allocations");
    }
}

impl ClusterStats {
    pub fn memory_budget_line(self) -> String {
        format!("Geometry budget: {:.0} MiB; {:.1} MiB over; {} growth/upload allocation refusals, {} pool trims{}",
            self.geometry_budget_bytes as f64 / 1048576.0,
            self.memory.total_allocated().saturating_sub(self.geometry_budget_bytes) as f64 / 1048576.0,
            self.memory_denials, self.pool_trims, if self.memory_paused { "; ordinary fallback while memory is tight" } else { "" })
    }
}
