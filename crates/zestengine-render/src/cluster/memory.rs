//! Zestlet allocation accounting and fence-safe reclamation. No shading changes.
use super::*;
use crate::cluster_memory::{allocation_charge, admits, trim_target, TRIM_DELAY};

impl ClusterStore {
    pub(super) fn allocated_bytes(&self) -> u64 {
        let pools = [&self.clusters, &self.vertices, &self.triangles, &self.nodes, &self.skin, &self.bones, &self.morphs];
        let (control, staged) = self.stream.allocated_buffer_bytes();
        pools.iter().map(|p| p.buffer.allocated_bytes()).sum::<u64>()
            + self.dummy.allocated_bytes() + control + staged
            + self.retired.iter().chain(&self.spent).map(|(_, b)| b.allocated_bytes()).sum::<u64>()
            + self.uploads.iter().map(|u| u.staging.allocated_bytes()).sum::<u64>()
    }

    /// Leave a small copy/reclamation reserve; old allocations remain charged until retirement.
    pub(super) fn admit(&mut self, extra: u64) -> bool {
        self.admit_reserving_loads(extra, 0)
    }

    pub(super) fn admit_reserving_loads(&mut self, extra: u64, arriving_page: u64) -> bool {
        let reserve = if self.budget_limit == u64::MAX { 0 } else { (self.budget_limit / 16).min(16 << 20) };
        let loading = self.stream.pending_staging_bytes().saturating_sub(arriving_page)
            .saturating_sub(if self.incoming_loading { self.incoming_bytes } else { 0 });
        let reserved = self.external_bytes.saturating_add(self.incoming_bytes).saturating_add(loading);
        let ok = admits(self.budget_limit.saturating_sub(reserve), self.allocated_bytes(), reserved, extra);
        if !ok { self.budget_denials += 1; }
        ok
    }

    pub(super) fn record_pool_copies(&mut self, device: &ash::Device, cb: vk::CommandBuffer, serial: u64) {
        let copies = std::mem::take(&mut self.pool_copies);
        if copies.is_empty() { return; }
        let dependency = || unsafe {
            let b = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE);
            device.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(), &[b], &[], &[]);
        };
        for (src, dst, bytes) in copies {
            dependency();
            unsafe { device.cmd_copy_buffer(cb, src, dst, &[vk::BufferCopy::default().size(bytes)]); }
            if let Some((at, _)) = self.retired.iter_mut().find(|(_, b)| b.handle == src) { *at = serial; }
        }
        // Later page/DAG writes may overlap the copied prefix.
        dependency();
    }

    /// Trim only free tails, never move a live mesh/page. One copy at most per desktop frame.
    pub(super) fn trim_pools(&mut self, device: &ash::Device, allocator: &mut Allocator, _command_pool: vk::CommandPool, _queue: vk::Queue, serial: u64) {
        if self.last_trim_serial == serial { return; }
        self.last_trim_serial = serial;
        for k in 0..7 {
            let p = self.pool_mut(k);
            let Some(target) = trim_target(p.ranges.capacity() as u64, p.ranges.high_water() as u64, 16) else { p.quiet = 0; continue };
            p.quiet = p.quiet.saturating_add(1);
            if p.quiet < TRIM_DELAY { continue; }
            let bytes = target * POOL_ELEM[k] as u64;
            if !admits(self.budget_limit, self.allocated_bytes(), self.external_bytes.saturating_add(self.stream.pending_staging_bytes()), allocation_charge(bytes)) { continue; }
            let Ok(new) = Buffer::new(device, allocator, bytes,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly, "cluster-pool-trimmed") else { continue };
            let p = self.pool_mut(k);
            let old = p.buffer.handle;
            let copy_bytes = p.ranges.high_water() as u64 * POOL_ELEM[k] as u64;
            if copy_bytes > 0 { self.pool_copies.push((old, new.handle, copy_bytes)); }
            let p = self.pool_mut(k);
            let old = std::mem::replace(&mut p.buffer, new);
            p.ranges.shrink(target as u32);
            p.quiet = 0;
            self.retired.push((serial, old));
            self.trimmed += 1;
            break;
        }
    }
}

impl ClusterFrame {
    pub(super) fn growth_charge(&self, plan: &CullPlan, instances: usize, batches: usize, views: usize, meshes: usize, skin_vertices: u32) -> u64 {
        [(&self.instances, instances * size_of::<InstRec>()), (&self.batches, batches * size_of::<BatchRec>()),
         (&self.views, views * size_of::<GpuView>()), (&self.mesh_table, meshes * size_of::<MeshRec>()),
         (&self.queue_a, plan.node_cap as usize * 8), (&self.queue_b, plan.node_cap as usize * 8),
         (&self.visible, plan.visible_half as usize * 16), (&self.occluded, plan.occ_cap as usize * 8),
         (&self.cmds, plan.batches as usize * 32), (&self.batch_counts, plan.batches as usize * 8),
         (&self.skin_pool, skin_vertices.max(1) as usize * crate::cluster_skin::SKIN_VERT_BYTES),
         (&self.skin_slots, if skin_vertices > 0 { plan.visible_half as usize * 8 } else { 8 })]
            .iter().map(|(b, need)| b.growth_charge(*need)).sum()
    }

    pub(super) fn trim_retired(&mut self, device: &ash::Device, allocator: &mut Allocator, used: bool) {
        for b in [&mut self.instances, &mut self.batches, &mut self.views, &mut self.mesh_table, &mut self.queue_a,
            &mut self.queue_b, &mut self.visible, &mut self.occluded, &mut self.cmds, &mut self.batch_counts,
            &mut self.skin_pool, &mut self.skin_slots] {
            b.trim_retired(device, allocator, used);
        }
    }
}

impl ClusterSystem {
    pub(super) fn frame_bytes(&self) -> u64 {
        self.frames.iter().map(|f| { let (a, b) = f.allocated_buffer_bytes(); a + b }).sum::<u64>() + self.skin.allocated_buffer_bytes()
    }
}

impl Renderer {
    pub(super) fn ordinary_geometry_bytes(&self) -> u64 {
        self.meshes.iter().map(|m| m.vertex_buffer.allocated_bytes() + m.index_buffer.allocated_bytes()).sum::<u64>()
            + self.skinner.as_ref().map_or(0, |s| s.allocated_buffer_bytes())
            + self.geometry_retired_bytes.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// `cull` includes the instance culler while it is temporarily taken out of Renderer.
    pub(crate) fn cluster_refresh_memory_with_cull(&mut self, cull: u64) {
        let ordinary = self.ordinary_geometry_bytes() + cull;
        if let Some(c) = self.cluster.as_mut() {
            c.store.budget_limit = (self.cluster_settings.geometry_budget_mb.max(1) as u64) << 20;
            c.store.external_bytes = ordinary + c.frame_bytes();
        }
    }

    pub(super) fn cluster_refresh_memory(&mut self) {
        let cull = self.gpu_cull.as_ref().map_or(0, |c| c.allocated_buffer_bytes());
        self.cluster_refresh_memory_with_cull(cull);
    }

    pub(crate) fn cluster_memory_maintenance(&mut self, slot: usize) {
        self.cluster_refresh_memory();
        let Some(c) = self.cluster.as_mut() else { return };
        let Some(a) = self.allocator.as_ref() else { return };
        let mut allocator = a.lock().unwrap();
        // Slot reuse is fenced by the caller. `used` is tracked separately from readback,
        // since XR deliberately does not ingest these counters.
        let f = &mut c.frames[slot];
        let used = std::mem::take(&mut f.memory_used);
        let skin_used = std::mem::take(&mut f.skin_memory_used);
        f.trim_retired(&self.ctx.device, &mut allocator, used);
        c.skin.trim_retired(&self.ctx.device, &mut allocator, slot, skin_used);
        // Only the desktop boundary advances shared-pool maintenance; an XR eye may
        // run after desktop commands using those pools were already recorded.
        if slot < FRAMES_IN_FLIGHT {
            c.store.reclaim(self.frame_serial);
            c.store.reclaim_buffers(&self.ctx.device, &mut allocator);
            c.store.release_empty_stream_blocks();
            c.store.trim_pools(&self.ctx.device, &mut allocator, self.command_pool, self.ctx.queue, self.frame_serial);
        }
        drop(allocator);
        self.cluster_refresh_memory();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster_headless::HeadlessGpu;

    #[test]
    fn range_tail_reclamation_keeps_live_offsets_and_holes() {
        let mut r = RangeAlloc::new(1024);
        assert_eq!(r.alloc(100), Some(0));
        assert_eq!(r.alloc(100), Some(100));
        r.free(0, 100);
        assert_eq!(r.high_water(), 200);
        r.shrink(256);
        assert_eq!(r.used(), 100);
        assert_eq!(r.alloc(100), Some(0));
        assert_eq!(r.alloc(56), Some(200));
        assert_eq!(r.alloc(1), None);
    }

    #[test]
    fn cluster_memory_gpu_growth_trim_and_admission_preserve_geometry() {
        let Some(gpu) = HeadlessGpu::new().unwrap() else { return };
        let device = &gpu.device;
        let mut allocator = gpu.allocator().unwrap();
        let mut store = ClusterStore::new(device, &mut allocator, PoolSizes::default()).unwrap();
        let initial = store.allocated_bytes();
        assert!(initial < 2 << 20, "initial pools must not reserve the old 165 MiB");
        let mesh = crate::cluster::tests::quad_grid(20);
        store.budget_limit = initial + (1 << 20);
        store.external_bytes = 1 << 20;
        assert!(store.install(device, &mut allocator, gpu.pool, gpu.queue, 0, &mesh).is_err());
        assert_eq!(store.mesh_count(), 0);
        assert!(store.budget_denials > 0);
        store.budget_limit = 32 << 20;
        store.external_bytes = 0;
        store.install(device, &mut allocator, gpu.pool, gpu.queue, 0, &mesh).unwrap();
        assert!(store.alias(1, 0));
        assert!(store.grow(1, 100_000, device, &mut allocator, gpu.pool, gpu.queue));
        assert!(store.grow(1, 200_000, device, &mut allocator, gpu.pool, gpu.queue));
        let pending = store.retired_buffers();
        store.reclaim(10);
        store.reclaim_buffers(device, &mut allocator);
        assert_eq!(store.retired_buffers(), pending, "unrecorded chained copies pin every old source");
        let grown = store.allocated_bytes();
        assert!(grown > initial);
        assert!(store.retired_buffers() > 0, "old storage remains charged through retirement");
        for serial in 11..=TRIM_DELAY as u64 + 30 {
            store.reclaim(serial);
            store.reclaim_buffers(device, &mut allocator);
            store.trim_pools(device, &mut allocator, gpu.pool, gpu.queue, serial);
            if !store.pool_copies.is_empty() {
                alloc::one_time_submit(device, gpu.pool, gpu.queue, |cb| store.record_uploads(device, cb, serial)).unwrap();
            }
        }
        assert!(store.trimmed > 0);
        assert!(store.allocated_bytes() < initial);
        let bytes = bytemuck::cast_slice::<_, u8>(&mesh.vertices);
        let mut readback = Buffer::new(device, &mut allocator, bytes.len() as u64,
            vk::BufferUsageFlags::TRANSFER_DST, MemoryLocation::GpuToCpu, "cluster-memory-test-readback").unwrap();
        alloc::one_time_submit(device, gpu.pool, gpu.queue, |cb| unsafe {
            device.cmd_copy_buffer(cb, store.vertices.buffer.handle, readback.handle, &[vk::BufferCopy::default().size(bytes.len() as u64)]);
        }).unwrap();
        assert_eq!(&readback.read()[..bytes.len()], bytes, "live quantized vertices survive grow and trim byte-for-byte");
        readback.destroy(device, &mut allocator);
        store.remove(0, 1000);
        store.reclaim(1001);
        assert!(store.record_of(1).is_some(), "alias pins shared storage");
        store.remove(1, 1001);
        store.reclaim(1002);
        assert!(store.vertices.ranges.used() > 0, "in-flight draws still pin the removed DAG");
        store.reclaim(1010);
        assert_eq!(store.vertices.ranges.used(), 0);
        store.destroy(device, &mut allocator);
        eprintln!("Zestlet memory GPU probe: initial {initial} bytes, growth peak {grown}; admission, shared ownership, retirement and byte-exact trim passed");
    }

    #[test]
    fn cluster_memory_gpu_slot_buffers_release_only_after_quiet_retired_uses() {
        let Some(gpu) = HeadlessGpu::new().unwrap() else { return };
        let mut allocator = gpu.allocator().unwrap();
        let mut b = GrowBuffer::new(vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, "cluster-memory-slot-test");
        b.ensure(&gpu.device, &mut allocator, 1 << 20).unwrap();
        let before = b.allocated_bytes();
        b.ensure(&gpu.device, &mut allocator, 64).unwrap();
        for _ in 0..TRIM_DELAY - 1 { assert_eq!(b.trim_retired(&gpu.device, &mut allocator, true), 0); }
        assert_eq!(b.allocated_bytes(), before);
        assert!(b.trim_retired(&gpu.device, &mut allocator, true) > 0);
        assert!(b.allocated_bytes() <= 4096);
        b.ensure(&gpu.device, &mut allocator, 1 << 20).unwrap();
        assert!(b.allocated_bytes() >= before, "active demand can grow again");
        b.destroy(&gpu.device, &mut allocator);
    }
}
