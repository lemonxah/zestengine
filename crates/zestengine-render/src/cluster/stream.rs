//! Geometry streaming in the cluster store (GEO-6): the page table and I/O
//! buffers, the page blocks inside the vertex / triangle pools, page installs
//! and evictions recorded into the frame without a queue wait, and the readback
//! of the cull's page requests. The policy lives in `cluster_stream.rs`.
//!
//! A streamed mesh keeps its clusters and BVH in the ordinary pool ranges and
//! the root page (page 0) in its ordinary vertex / triangle ranges. Every other
//! page lives in a *block*: one range reserved from each pool when streaming is
//! configured and sub-allocated per page, so streaming cannot take pool space
//! from other meshes and never grows a pool while rendering. A page install
//! reserves its ranges now, the next frame records the copy and the page table
//! write; an eviction writes the table entry as not resident at once and
//! returns the ranges after `FRAMES_IN_FLIGHT` frames, when no frame that may
//! still read the page is in flight (the same deferred free as a removed DAG).

use std::collections::BTreeMap;

use super::*;
use crate::cluster_stream::{PageManager, PageSpec, Plan, StreamPolicy, StreamStats};

/// Shadow views scale the priority of their page requests by this (2^-40) in
/// `cluster_cull.comp`, which puts every shadow value below the smallest main
/// value (1e-6): the class travels in the one word the GPU dedups with an atomic max.
pub const SHADOW_PRIO_SCALE: f32 = 9.094_947e-13;

/// Split a request word's float into (projected-error ratio, from a shadow view).
pub fn decode_request_priority(value: f32) -> (f32, bool) {
    if value > 0.0 && value < 1.0e-6 {
        (value / SHADOW_PRIO_SCALE, true)
    } else {
        (value, false)
    }
}

/// Reserve `pages` consecutive page table entries; the error says what to raise.
pub(super) fn alloc_table_entries(entries: &mut RangeAlloc, pages: usize, max_pages: usize) -> Result<u32, String> {
    entries.alloc(pages as u32).ok_or_else(|| {
        format!(
            "streaming page table full: {pages} more pages needed, {} of {max_pages} entries in use; raise ZESTENGINE_CLUSTER_STREAM_MAX_PAGES (up to {STREAM_HARD_MAX_PAGES}, 16 bytes of GPU memory per entry)",
            entries.used()
        )
    })
}

/// A block of one pool handed to page allocations.
pub(super) struct Block {
    /// First element of the block in the pool.
    base: u32,
    alloc: RangeAlloc,
}

impl Block {
    fn alloc(&mut self, n: u32) -> Option<u32> {
        self.alloc.alloc(n)
    }
}

/// Where one page of a streamed DAG lives.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Slot {
    pub resident: bool,
    /// (start, length) of the page's vertices (inside the vertex block) and triangle block
    /// words (as an absolute pool word index: the triangle blocks are several).
    pub v: (u32, u32),
    pub t: (u32, u32),
}

/// Streaming state of one DAG record.
#[derive(Clone, Debug)]
pub(super) struct Streamed {
    pub table_base: u32,
    /// Renderer mesh the record was installed for (load orders name it).
    pub mesh: usize,
    pub pages: Vec<PageLayout>,
    pub slots: Vec<Slot>,
}

/// A page staged in a host-visible buffer by the loader (`PageStager`), waiting for `Renderer::cluster_stream_deliver`.
pub struct StagedPage {
    staging: Buffer,
    vertices: u32,
    tri_bytes: u32,
}

impl StagedPage {
    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        self.staging.destroy(device, allocator);
    }
}

/// Stages page payloads from any thread (device and allocator only: no queue).
#[derive(Clone)]
pub struct PageStager {
    pub(crate) device: ash::Device,
    pub(crate) allocator: std::sync::Arc<std::sync::Mutex<Allocator>>,
}

impl PageStager {
    /// Copy a page's payload (`vertices * 28` bytes of vertices, then `tri_bytes` of triangles) into a staging buffer.
    pub fn stage(&self, payload: &[u8], vertices: u32, tri_bytes: u32) -> Result<StagedPage> {
        let want = vertices as usize * size_of::<ClusterVertexRec>() + tri_bytes as usize;
        if payload.len() != want || !tri_bytes.is_multiple_of(4) || want == 0 {
            bail!("page payload has {} bytes, expected {want}", payload.len());
        }
        let mut staging = Buffer::new(&self.device, &mut self.allocator.lock().unwrap(), want as u64, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu, "cluster-page-staging")?;
        staging.write(0, payload);
        Ok(StagedPage { staging, vertices, tri_bytes })
    }
}

/// A page the renderer wants read and staged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageLoad {
    pub mesh: MeshHandle,
    pub page: u32,
    pub vertices: u32,
    pub tri_bytes: u32,
}

/// Page copy waiting for the next frame's command buffer.
pub(super) struct QueuedPage {
    id: u32,
    page: u32,
    staging: Buffer,
    copies: [(usize, vk::BufferCopy); 2],
}

/// A freed record: (serial of the eviction, vertex range, triangle range).
type FreedRecord = (u64, (u32, u32), (u32, u32));

struct Held {
    id: u32,
    page: u32,
    staged: StagedPage,
}

/// The GPU side: page table, I/O buffer, table entry allocator.
pub(super) struct StreamGpu {
    pub table: Buffer,
    pub io: Buffer,
    entries: RangeAlloc,
    /// One past the highest table entry in use (the touched flags to clear and copy back).
    pub high: u32,
}

pub(super) struct StreamState {
    pub gpu: Option<StreamGpu>,
    vblock: Option<Block>,
    /// Triangle-pool blocks pages are put into: the first is sized with the vertex block, more
    /// are added when a skinned streamed mesh needs a larger share (skin records and blendshape
    /// entries travel in the triangle block of its pages).
    tblocks: Vec<Block>,
    /// Share of the budget the triangle blocks hold (bytes); at least `MIN_TRI_FRACTION`.
    tri_fraction: f64,
    pub pager: PageManager,
    pub policy: StreamPolicy,
    /// The budget last asked for, before the pools clamped it: `policy.budget_bytes` can be lower,
    /// so comparing a new request against the clamped value would never match.
    requested_budget: u64,
    configured: bool,
    table_writes: BTreeMap<u32, [u32; 4]>,
    queued: Vec<QueuedPage>,
    /// (serial of the eviction, vertex range, triangle range): freed `FRAMES_IN_FLIGHT` later.
    free: Vec<FreedRecord>,
    held: Vec<Held>,
    orders: Vec<PageLoad>,
    /// Table base -> (record id, entries), to map a global page back to its record.
    owners: BTreeMap<u32, (u32, u32)>,
    clear_io: bool,
    /// Pages whose install found no room in the block (fragmentation).
    pub no_room: u64,
}

impl Default for StreamState {
    fn default() -> Self {
        Self {
            gpu: None,
            vblock: None,
            tblocks: Vec::new(),
            tri_fraction: MIN_TRI_FRACTION,
            pager: PageManager::new(StreamPolicy::default()),
            policy: StreamPolicy::default(),
            requested_budget: 0,
            configured: false,
            table_writes: BTreeMap::new(),
            queued: Vec::new(),
            free: Vec::new(),
            held: Vec::new(),
            orders: Vec::new(),
            owners: BTreeMap::new(),
            clear_io: false,
            no_room: 0,
        }
    }
}

/// Share of a page budget the triangle blocks hold at least (static pages are mostly vertices).
const MIN_TRI_FRACTION: f64 = 0.30;

/// Elements of the vertex block and words of the triangle blocks that hold a budget of `bytes`
/// (pages are mostly vertices; the margin absorbs fragmentation and a different mix). A skinned
/// mesh's pages are mostly triangle block (skin, joint ids and blendshape entries), so the
/// triangle share follows `tri_fraction`.
fn block_sizes(bytes: u64, tri_fraction: f64) -> (u32, u32) {
    let v = (bytes as f64 * 0.95 / size_of::<ClusterVertexRec>() as f64) as u64 + 4096;
    let t = (bytes as f64 * tri_fraction.clamp(MIN_TRI_FRACTION, 1.0) / 4.0) as u64 + 4096;
    (v.min(u32::MAX as u64 / 2) as u32, t.min(u32::MAX as u64 / 2) as u32)
}

impl ClusterStore {
    /// Whether any streamed DAG is installed (the streaming passes record only then).
    pub(crate) fn stream_active(&self) -> bool {
        self.stream.gpu.as_ref().is_some_and(|g| g.entries.used() > 0)
    }

    pub(crate) fn stream_high(&self) -> u32 {
        self.stream.gpu.as_ref().map_or(0, |g| g.high)
    }

    /// (page table, I/O) buffers for the descriptor set; small dummies while streaming was never used.
    pub(crate) fn stream_buffers(&self) -> (vk::Buffer, vk::Buffer) {
        match &self.stream.gpu {
            Some(g) => (g.table.handle, g.io.handle),
            None => (self.dummy.handle, self.dummy.handle),
        }
    }

    pub(crate) fn stream_stats(&self) -> StreamStats {
        self.stream.pager.stats()
    }

    pub(crate) fn stream_policy(&self) -> StreamPolicy {
        self.stream.policy
    }

    pub(crate) fn stream_configured(&self) -> bool {
        self.stream.configured
    }

    /// The budget the last `stream_configure` was asked for (before the pool limits lowered it).
    pub(crate) fn stream_requested_budget(&self) -> u64 {
        self.stream.requested_budget
    }

    fn ensure_stream_gpu(&mut self, device: &ash::Device, allocator: &mut Allocator) -> Result<()> {
        if self.stream.gpu.is_some() {
            return Ok(());
        }
        let max_pages = stream_max_pages();
        let table = Buffer::new(
            device,
            allocator,
            (max_pages * 16) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            "cluster-page-table",
        )?;
        let io = Buffer::new(
            device,
            allocator,
            stream_io_bytes(max_pages),
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::GpuOnly,
            "cluster-stream-io",
        )?;
        self.stream.gpu = Some(StreamGpu { table, io, entries: RangeAlloc::new(max_pages as u32), high: 0 });
        self.stream.clear_io = true;
        Ok(())
    }

    pub(super) fn destroy_stream(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        if let Some(mut g) = self.stream.gpu.take() {
            g.table.destroy(device, allocator);
            g.io.destroy(device, allocator);
        }
        for mut q in self.stream.queued.drain(..) {
            q.staging.destroy(device, allocator);
        }
        for mut h in self.stream.held.drain(..) {
            h.staged.destroy(device, allocator);
        }
    }

    /// Make the triangle blocks big enough for a streamed DAG whose pages are mostly triangle
    /// block (skinned meshes: skin records, joint ids, blendshape entries). Raises the share of
    /// the budget the blocks hold; with blocks already created an extra block is added (pages
    /// resident in the first stay where they are). Call before the DAG's table and pool ranges.
    pub(super) fn ensure_tri_share(&mut self, device: &ash::Device, allocator: &mut Allocator, command_pool: vk::CommandPool, queue: vk::Queue, layout: &StreamLayout) -> Result<()> {
        let share = (layout.tri_share() + 0.1).min(1.0);
        let old = self.stream.tri_fraction;
        if share <= old {
            return Ok(());
        }
        if self.stream.configured && !self.stream.tblocks.is_empty() {
            let words = (((share - old) * self.stream.policy.budget_bytes as f64) / 4.0).min(u32::MAX as f64 / 4.0) as u32;
            let at = self.triangles.ranges.alloc(words).or_else(|| self.grow(2, words, device, allocator, command_pool, queue).then(|| self.triangles.ranges.alloc(words)).flatten());
            let Some(at) = at else {
                bail!("the cluster triangle pool cannot hold {} MB more for streamed skin and blendshape data", (words as u64 * 4) >> 20);
            };
            self.stream.tblocks.push(Block { base: at, alloc: RangeAlloc::new(words) });
        }
        self.stream.tri_fraction = share;
        Ok(())
    }

    /// Reserve the page table entries of a streamed DAG before its pool ranges.
    pub(super) fn alloc_table(&mut self, device: &ash::Device, allocator: &mut Allocator, pages: usize) -> Result<u32> {
        self.ensure_stream_gpu(device, allocator)?;
        let g = self.stream.gpu.as_mut().expect("just created");
        let base = alloc_table_entries(&mut g.entries, pages, stream_max_pages()).map_err(anyhow::Error::msg)?;
        g.high = g.high.max(base + pages as u32);
        Ok(base)
    }

    pub(super) fn free_table(&mut self, base: u32, pages: usize) {
        if let Some(g) = self.stream.gpu.as_mut() {
            g.entries.free(base, pages as u32);
        }
    }

    /// Configure streaming: the budget and limits, and the page blocks that hold
    /// them. Returns the policy in effect (the budget is lowered to what the pools
    /// can hold). Blocks are only re-created while no streamed page is resident,
    /// so call this before meshes stream or accept the current blocks.
    pub(crate) fn stream_configure(&mut self, device: &ash::Device, allocator: &mut Allocator, command_pool: vk::CommandPool, queue: vk::Queue, mut policy: StreamPolicy) -> Result<StreamPolicy> {
        self.ensure_stream_gpu(device, allocator)?;
        let requested = policy.budget_bytes;
        let covers = |s: &Self, bytes: u64| {
            let (v, t) = block_sizes(bytes, s.stream.tri_fraction);
            s.stream.vblock.as_ref().is_some_and(|b| b.alloc.capacity() >= v) && s.stream.tblocks.iter().map(|b| u64::from(b.alloc.capacity())).sum::<u64>() >= u64::from(t)
        };
        if !covers(self, policy.budget_bytes) {
            let in_use = self.stream.queued.len() + self.stream.held.len() + self.stream.free.len() > 0 || self.stream.pager.stats().resident_pages + self.stream.pager.stats().loading_pages > 0;
            if in_use {
                // Cannot move the blocks under live pages: keep what is there.
                let cap_v = self.stream.vblock.as_ref().map_or(0, |b| b.alloc.capacity()) as f64 * size_of::<ClusterVertexRec>() as f64 / 0.95;
                let cap_t = self.stream.tblocks.iter().map(|b| f64::from(b.alloc.capacity())).sum::<f64>() * 4.0 / self.stream.tri_fraction;
                policy.budget_bytes = policy.budget_bytes.min(cap_v.min(cap_t) as u64);
            } else {
                if let Some(b) = self.stream.vblock.take() {
                    self.vertices.ranges.free(b.base, b.alloc.capacity());
                }
                for b in std::mem::take(&mut self.stream.tblocks) {
                    self.triangles.ranges.free(b.base, b.alloc.capacity());
                }
                // Halve the budget until both blocks fit (the pools grow within their limits first).
                let mut budget = policy.budget_bytes;
                loop {
                    let (v, t) = block_sizes(budget, self.stream.tri_fraction);
                    let vb = self.vertices.ranges.alloc(v).or_else(|| {
                        self.grow(1, v, device, allocator, command_pool, queue).then(|| self.vertices.ranges.alloc(v)).flatten()
                    });
                    if let Some(vb) = vb {
                        let tb = self.triangles.ranges.alloc(t).or_else(|| {
                            self.grow(2, t, device, allocator, command_pool, queue).then(|| self.triangles.ranges.alloc(t)).flatten()
                        });
                        if let Some(tb) = tb {
                            self.stream.vblock = Some(Block { base: vb, alloc: RangeAlloc::new(v) });
                            self.stream.tblocks.push(Block { base: tb, alloc: RangeAlloc::new(t) });
                            break;
                        }
                        self.vertices.ranges.free(vb, v);
                    }
                    budget /= 2;
                    if budget < (4 << 20) {
                        bail!("the cluster pools cannot hold a streaming budget (device storage buffer limit)");
                    }
                }
                if budget < policy.budget_bytes {
                    tracing::warn!("Zestlet streaming: budget lowered from {} MB to {} MB (pool limits)", policy.budget_bytes >> 20, budget >> 20);
                    policy.budget_bytes = budget;
                }
            }
        }
        self.stream.policy = policy;
        self.stream.requested_budget = requested;
        self.stream.pager.set_policy(policy);
        self.stream.configured = true;
        Ok(policy)
    }

    /// Register a freshly installed streamed record: manager state and the table entries.
    pub(super) fn register_stream(&mut self, id: u32, mesh: usize, table_base: u32, layout: StreamLayout, root: (u32, u32)) {
        let np = layout.pages.len();
        for (i, p) in layout.pages.iter().enumerate() {
            let entry = if i == 0 { [root.0, root.1 * 4, p.vertices, p.tri_bytes] } else { [PAGE_NONE, 0, p.vertices, p.tri_bytes] };
            self.stream.table_writes.insert(table_base + i as u32, entry);
        }
        let specs: Vec<PageSpec> = layout.pages.iter().map(|p| PageSpec { bytes: p.bytes(), deps: p.deps.clone() }).collect();
        self.stream.pager.add_mesh(id, &specs);
        self.stream.owners.insert(table_base, (id, np as u32));
        let mut slots = vec![Slot::default(); np];
        slots[0].resident = true;
        if let Some(Some(r)) = self.records.get_mut(id as usize) {
            r.rec.c = [table_base, np as u32, 0, r.rec.c[3] | 2];
            r.stream = Some(Streamed { table_base, mesh, pages: layout.pages, slots });
        }
    }

    /// The record and local page of global table entry `g`.
    fn owner_of(&self, g: u32) -> Option<(u32, u32)> {
        let (&base, &(id, n)) = self.stream.owners.range(..=g).next_back()?;
        (g - base < n).then_some((id, g - base))
    }

    fn streamed(&self, id: u32) -> Option<&Streamed> {
        self.records.get(id as usize)?.as_ref()?.stream.as_ref()
    }

    fn streamed_mut(&mut self, id: u32) -> Option<&mut Streamed> {
        self.records.get_mut(id as usize)?.as_mut()?.stream.as_mut()
    }

    /// Whether `page` of record `id` is resident for the renderer (copied or queued).
    #[allow(dead_code)] // residency probe used by the stream tests
    pub(crate) fn page_resident(&self, id: u32, page: u32) -> bool {
        self.streamed(id).and_then(|s| s.slots.get(page as usize)).is_some_and(|s| s.resident)
    }

    /// Feed one retired frame's readback to the policy: `requests` are (global
    /// page, priority bits), `touched` the table entries the cull drew from.
    pub(crate) fn stream_ingest(&mut self, requests: &[(u32, u32)], touched: impl Iterator<Item = u32>, serial: u64) {
        self.stream.pager.begin_frame(serial);
        for g in touched {
            if let Some((id, page)) = self.owner_of(g) {
                self.stream.pager.touch(id, page);
            }
        }
        for &(g, bits) in requests {
            let ratio = f32::from_bits(bits);
            if !ratio.is_finite() {
                continue;
            }
            // Shadow views write their ratio scaled by SHADOW_PRIO_SCALE (below any main ratio).
            let (ratio, shadow) = decode_request_priority(ratio);
            if let Some((id, page)) = self.owner_of(g) {
                self.stream.pager.request_as(id, page, ratio, shadow);
            }
        }
    }

    /// Decide this frame's loads and evictions; the loads become orders for the loader.
    pub(crate) fn stream_plan(&mut self, serial: u64) {
        if !self.stream.configured {
            return;
        }
        self.stream.pager.begin_frame(serial);
        let plan = self.stream.pager.plan();
        self.apply_plan(plan, serial);
    }

    fn apply_plan(&mut self, plan: Plan, serial: u64) {
        for (id, page) in plan.evicts {
            self.evict_page(id, page, serial);
        }
        for (id, page) in plan.loads {
            if let Some(s) = self.streamed(id) {
                let p = &s.pages[page as usize];
                self.stream.orders.push(PageLoad { mesh: MeshHandle(s.mesh), page, vertices: p.vertices, tri_bytes: p.tri_bytes });
            }
        }
    }

    /// Orders the loader should serve now (drains the list).
    pub(crate) fn take_orders(&mut self) -> Vec<PageLoad> {
        std::mem::take(&mut self.stream.orders)
    }

    fn evict_page(&mut self, id: u32, page: u32, serial: u64) {
        let queued = self.stream.queued.iter().position(|q| q.id == id && q.page == page);
        let Some(s) = self.streamed_mut(id) else { return };
        let slot = std::mem::take(&mut s.slots[page as usize]);
        if !slot.resident {
            return;
        }
        let entry = [PAGE_NONE, 0, s.pages[page as usize].vertices, s.pages[page as usize].tri_bytes];
        let base = s.table_base;
        self.stream.table_writes.insert(base + page, entry);
        match queued {
            // Installed but never recorded: nothing on the GPU can name the ranges.
            Some(at) => {
                let q = self.stream.queued.swap_remove(at);
                self.spent.push((serial, q.staging));
                self.free_page_now(slot);
            }
            None => self.stream.free.push((serial, slot.v, slot.t)),
        }
    }

    fn free_page_now(&mut self, slot: Slot) {
        if let Some(b) = self.stream.vblock.as_mut() {
            b.alloc.free(slot.v.0, slot.v.1);
        }
        if let Some(b) = self.stream.tblocks.iter_mut().find(|b| slot.t.0 >= b.base && slot.t.0 - b.base < b.alloc.capacity()) {
            let base = b.base;
            b.alloc.free(slot.t.0 - base, slot.t.1);
        }
    }

    /// A page the loader read: reserve its ranges, queue the copy and table write
    /// for the next frame. Held until the pages it depends on are resident.
    pub(crate) fn stream_deliver(&mut self, device: &ash::Device, allocator: &mut Allocator, mesh: usize, page: u32, staged: StagedPage) {
        let Some(&id) = self.by_mesh.get(&mesh) else {
            let mut st = staged;
            st.destroy(device, allocator);
            return;
        };
        self.stream.held.push(Held { id, page, staged });
        self.stream_retry_held(device, allocator);
    }

    /// Install held pages whose dependencies are resident now.
    pub(crate) fn stream_retry_held(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        loop {
            let ready = self.stream.held.iter().position(|h| {
                self.streamed(h.id).is_some_and(|s| s.pages[h.page as usize].deps.iter().all(|&d| s.slots[d as usize].resident))
            });
            let Some(at) = ready else { break };
            let h = self.stream.held.remove(at);
            self.install_page(device, allocator, h);
        }
    }

    fn install_page(&mut self, device: &ash::Device, allocator: &mut Allocator, h: Held) {
        let Held { id, page, mut staged } = h;
        let layout = self.streamed(id).map(|s| (s.pages[page as usize].vertices, s.pages[page as usize].tri_bytes, s.table_base, s.slots[page as usize].resident));
        // Only a page the policy is waiting for is installed (it may have failed with a parent, or its mesh gone, while the bytes were read).
        if layout.is_none() || !self.stream.pager.is_loading(id, page) {
            staged.destroy(device, allocator);
            return;
        }
        let (v, t, base, resident) = layout.expect("checked");
        if resident || v != staged.vertices || t != staged.tri_bytes {
            staged.destroy(device, allocator);
            self.fail_loading(device, allocator, id, page);
            return;
        }
        let words = t / 4;
        let vs = self.stream.vblock.as_mut().and_then(|b| b.alloc(v));
        // Absolute pool word of the page's triangle block, from whichever block has room.
        let ts = if vs.is_some() { self.stream.tblocks.iter_mut().find_map(|b| b.alloc(words).map(|at| b.base + at)) } else { None };
        let (Some(vs), Some(ts)) = (vs, ts) else {
            if let (Some(vs), Some(b)) = (vs, self.stream.vblock.as_mut()) {
                b.alloc.free(vs, v);
            }
            self.stream.no_room += 1;
            staged.destroy(device, allocator);
            self.fail_loading(device, allocator, id, page);
            return;
        };
        let vbase = self.stream.vblock.as_ref().expect("block").base;
        let vbytes = v as u64 * size_of::<ClusterVertexRec>() as u64;
        let copies = [
            (1usize, vk::BufferCopy::default().src_offset(0).dst_offset((vbase + vs) as u64 * size_of::<ClusterVertexRec>() as u64).size(vbytes)),
            (2usize, vk::BufferCopy::default().src_offset(vbytes).dst_offset(ts as u64 * 4).size(t as u64)),
        ];
        self.stream.table_writes.insert(base + page, [vbase + vs, ts * 4, v, t]);
        let s = self.streamed_mut(id).expect("checked");
        s.slots[page as usize] = Slot { resident: true, v: (vs, v), t: (ts, words) };
        self.stream.queued.push(QueuedPage { id, page, staging: staged.staging, copies });
        self.stream.pager.loaded(id, page);
    }

    /// A load that cannot complete: back to absent in the policy (with its loading dependents).
    fn fail_loading(&mut self, device: &ash::Device, allocator: &mut Allocator, id: u32, page: u32) {
        let failed = self.stream.pager.failed(id, page);
        for p in failed {
            while let Some(at) = self.stream.held.iter().position(|h| h.id == id && h.page == p) {
                let mut h = self.stream.held.remove(at);
                h.staged.destroy(device, allocator);
            }
        }
    }

    /// The loader could not read a page.
    pub(crate) fn stream_fail(&mut self, device: &ash::Device, allocator: &mut Allocator, mesh: usize, page: u32) {
        if let Some(&id) = self.by_mesh.get(&mesh) {
            self.fail_loading(device, allocator, id, page);
        }
    }

    /// Whether the next `record_uploads` has streaming work (page copies, table writes, the first clear).
    pub(super) fn record_stream_pending(&self) -> bool {
        self.stream.gpu.is_some() && (!self.stream.queued.is_empty() || !self.stream.table_writes.is_empty() || self.stream.clear_io)
    }

    /// Streaming part of `record_uploads`: page copies, table writes, the first clear of the I/O buffer.
    pub(super) fn record_stream(&mut self, device: &ash::Device, cb: vk::CommandBuffer, serial: u64) {
        let Some(g) = self.stream.gpu.as_ref() else { return };
        let (table, io) = (g.table.handle, g.io.handle);
        // The table entries and the I/O buffer are read by shaders of frames still in
        // flight: order those reads before these writes.
        let b = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::VERTEX_SHADER | vk::PipelineStageFlags2::TRANSFER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::TRANSFER_READ)
            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)];
        unsafe { device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&b)) };
        if std::mem::take(&mut self.stream.clear_io) {
            unsafe { device.cmd_fill_buffer(cb, io, 0, vk::WHOLE_SIZE, 0) };
        }
        let handles = [
            self.clusters.buffer.handle,
            self.vertices.buffer.handle,
            self.triangles.buffer.handle,
        ];
        for q in std::mem::take(&mut self.stream.queued) {
            for (k, region) in &q.copies {
                unsafe { device.cmd_copy_buffer(cb, q.staging.handle, handles[*k], std::slice::from_ref(region)) };
            }
            self.spent.push((serial, q.staging));
        }
        // Table entries, consecutive runs in one update (at most 4096 bytes each).
        let writes: Vec<(u32, [u32; 4])> = std::mem::take(&mut self.stream.table_writes).into_iter().collect();
        let mut i = 0;
        while i < writes.len() {
            let mut j = i + 1;
            while j < writes.len() && writes[j].0 == writes[j - 1].0 + 1 && j - i < 256 {
                j += 1;
            }
            let data: Vec<[u32; 4]> = writes[i..j].iter().map(|w| w.1).collect();
            unsafe { device.cmd_update_buffer(cb, table, writes[i].0 as u64 * 16, bytemuck::cast_slice(&data)) };
            i = j;
        }
    }

    /// Return the ranges of evicted pages that no frame in flight can read.
    pub(super) fn reclaim_stream(&mut self, serial: u64) {
        let mut i = 0;
        while i < self.stream.free.len() {
            let (at, v, t) = self.stream.free[i];
            if at + FRAMES_IN_FLIGHT as u64 <= serial {
                self.stream.free.swap_remove(i);
                self.free_page_now(Slot { resident: false, v, t });
            } else {
                i += 1;
            }
        }
    }

    /// A record is being freed (its removal retired): return every page it held.
    pub(super) fn release_stream(&mut self, s: Streamed) {
        for (i, slot) in s.slots.iter().enumerate() {
            if slot.resident && i != 0 {
                self.free_page_now(*slot);
            }
        }
        self.free_table(s.table_base, s.pages.len());
        self.stream.owners.remove(&s.table_base);
    }

    /// A record loses its last user: the policy forgets it and its queued / held pages are dropped.
    pub(super) fn drop_stream(&mut self, id: u32, serial: u64) {
        self.stream.pager.remove_mesh(id);
        let mut i = 0;
        while i < self.stream.queued.len() {
            if self.stream.queued[i].id == id {
                let q = self.stream.queued.swap_remove(i);
                self.spent.push((serial, q.staging));
            } else {
                i += 1;
            }
        }
        let mut i = 0;
        while i < self.stream.held.len() {
            if self.stream.held[i].id == id {
                let h = self.stream.held.swap_remove(i);
                self.spent.push((serial, h.staged.staging));
            } else {
                i += 1;
            }
        }
    }

    /// Evict every streamed page now (tests; a reconfigure).
    pub(crate) fn stream_evict_all(&mut self, serial: u64) {
        for (id, page) in self.stream.pager.evict_all() {
            self.evict_page(id, page, serial);
        }
    }

    /// Pages the staged / held queues still carry (tests).
    #[allow(dead_code)] // queue probe used by the stream tests
    pub(crate) fn stream_backlog(&self) -> (usize, usize) {
        (self.stream.queued.len(), self.stream.held.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangle_blocks_follow_the_skinned_share_of_the_pages() {
        let mb = 64u64 << 20;
        let (v0, t0) = block_sizes(mb, MIN_TRI_FRACTION);
        let (v1, t1) = block_sizes(mb, 0.9);
        assert_eq!(v0, v1, "the vertex block does not depend on the triangle share");
        assert!(u64::from(t1) * 4 >= (mb as f64 * 0.9) as u64, "a mostly-triangle-block page mix must fit the budget");
        assert!(t1 > 2 * t0);
        // A share below the static minimum or above 1 is clamped.
        assert_eq!(block_sizes(mb, 0.0), (v0, t0));
        assert_eq!(block_sizes(mb, 7.0).1, block_sizes(mb, 1.0).1);
        let layout = StreamLayout { pages: vec![PageLayout { vertices: 10, tri_bytes: 40, deps: vec![] }, PageLayout { vertices: 100, tri_bytes: 2400, deps: vec![0] }] };
        assert!((layout.tri_share() - 2400.0 / (100.0 * size_of::<ClusterVertexRec>() as f64 + 2400.0)).abs() < 1e-9, "the root page is not part of the streamed share");
    }

    #[test]
    fn the_page_table_is_a_bound_with_a_clear_error() {
        let mut t = RangeAlloc::new(100);
        assert_eq!(alloc_table_entries(&mut t, 60, 100), Ok(0));
        assert_eq!(alloc_table_entries(&mut t, 40, 100), Ok(60));
        let e = alloc_table_entries(&mut t, 1, 100).unwrap_err();
        assert!(e.contains("ZESTENGINE_CLUSTER_STREAM_MAX_PAGES") && e.contains("100 of 100"), "{e}");
        t.free(0, 60);
        assert!(alloc_table_entries(&mut t, 61, 100).is_err(), "a hole too small is refused as well");
        assert_eq!(alloc_table_entries(&mut t, 60, 100), Ok(0), "freed entries are reused");
    }

    #[test]
    fn the_default_page_table_is_beyond_the_old_65536_cap() {
        assert!(stream_max_pages() >= 65_536 || std::env::var_os("ZESTENGINE_CLUSTER_STREAM_MAX_PAGES").is_some());
        assert!(stream_touch_base(stream_max_pages()) + stream_max_pages() < u32::MAX as usize);
        // The shader finds the touched flags at PRIO_BASE + pages and the readback covers every flag.
        assert_eq!(stream_io_bytes(1000), ((STREAM_PRIO_BASE + 2000) * 4) as u64);
        assert_eq!(stream_readback_bytes(1000), (STREAM_PRIO_BASE as u64 + 1000) * 4);
    }

    #[test]
    fn shadow_priorities_round_trip_and_stay_below_every_main_value() {
        let (r, shadow) = decode_request_priority(2.5);
        assert!((r - 2.5).abs() < 1e-6 && !shadow);
        for ratio in [1.0e-6f32, 0.01, 1.0, 37.0, 1.0e6] {
            // What cluster_cull.comp writes: max(ratio, 1e-6) * 2^-40.
            let word = ratio.max(1.0e-6) * SHADOW_PRIO_SCALE;
            assert!(word < 1.0e-6, "shadow word {word} reaches the main range");
            let (back, shadow) = decode_request_priority(word);
            assert!(shadow && (back / ratio.max(1.0e-6) - 1.0).abs() < 1e-5, "{ratio} -> {word} -> {back}");
        }
    }
}
