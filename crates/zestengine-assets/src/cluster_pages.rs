//! Cluster pages (GEO-6): a mesh's cluster DAG split into fixed-size pages the
//! renderer can stream in and out, the page graph that keeps every cut crack
//! free while pages are missing, and the paged file format.
//!
//! # What is paged
//!
//! The cluster records, the BVH and the group links are small (96 bytes per
//! cluster against 2 to 3 KB of vertices and triangles) and stay resident. Only
//! the heavy payload, the quantised vertices and the triangle bytes, lives in
//! pages. A cluster belongs to exactly one page and its `vertex_offset` /
//! `triangle_offset` become relative to that page.
//!
//! # Residency rules (the dependency closure)
//!
//! A cluster is consumed by one group (`ClusterLink::parent_group`), which
//! simplifies all its inputs into the group's outputs. A page holds whole
//! *input sets*: all inputs of a group (all of one level) share a page. With
//! that, two rules keep the cut closed under missing pages:
//!
//! * **R (siblings)**: if a cluster is resident, so are all inputs of its
//!   consuming group (they are on the same page, by construction);
//! * **R2 (parents)**: if a cluster is resident, so are the outputs of its
//!   consuming group; those are inputs of other groups, so this is a
//!   dependency of the page on [`PageInfo::deps`], transitively up to the root.
//!
//! The *root page* (page 0) holds every cluster of the coarse top levels
//! (up to [`PageOptions::resident_bytes`]) and is always resident, so a cut
//! always exists. A resident page set closed under `deps` is "valid".
//!
//! # Skinned and blendshape meshes
//!
//! A skinned or blendshape DAG pages exactly like a static one; the extra data
//! of a cluster travels with the cluster, in the same page, so one page table
//! lookup finds everything the skin stage needs. The page's *triangle block*
//! (what the GPU holds in the triangle pool, `PageInfo::tri_bytes`) is
//!
//! ```text
//! [ skin records, 8 bytes per page vertex ]        parallel to the page's vertices
//! [ triangle bytes of every cluster, each run padded to 4 ]
//! [ per cluster: 1 word morph count, ceil(bones / 2) words of u16 joint ids,
//!   16-byte sparse blendshape entries ]            "aux", word aligned
//! ```
//!
//! so the shaders find a cluster's skin at `8 * (vertex_offset + i)` into the
//! block, its triangles at `8 * page_vertices + triangle_offset`, and its aux
//! record at `bone_offset` (a word offset into the block; the page-relative
//! rewrite of `bone_offset`). Static meshes have no skin region and no aux. A
//! morph-only mesh (no skin weights) gets zero skin records. Measured on the
//! project avatars (see Features.md): bone lists are tiny, the skin stream is a
//! third of the vertex bytes and blendshape entries can be several times the
//! vertices, so all of it pages instead of staying resident.
//!
//! # The cut with missing pages
//!
//! `child_page[c]` is the page holding the inputs of the group that produced
//! `c` (the cluster's finer children). The cull draws a resident cluster when
//!
//! ```text
//! (lod_ok(c) || child page not resident)  &&  !lod_ok(parent)
//! ```
//!
//! i.e. the ordinary rule, except that a cluster that is too coarse for the view
//! still stands in for its children while their page is missing, and asks for
//! that page. Clusters on a non-resident page are skipped (their parents stand
//! in). [`PagedLayout::select`] is the CPU reference; the tests prove it covers
//! every finest cluster exactly once for any valid resident set.
//!
//! # File format (`ZPG1`, a mesh slot of the geometry cache)
//!
//! `magic, format, skeleton length, plan length, directory length`, then the
//! skeleton (a [`ClusterDag`] without vertices / triangles / sources), the plan
//! (`cluster -> page`), the page directory (offsets, lengths, checksums) and the
//! page blobs. A page's *hot* blob (vertices then triangles, what the GPU needs)
//! and its *cold* blob (source vertex ids, only for rebuilding a full DAG on
//! the CPU) are separate, so streaming never reads the cold data. Each blob has
//! a checksum; the whole is validated before anything reaches the GPU.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::cluster_dag::{ClusterDag, ClusterVertex, ClusterVertexSkin, DagHeader, FLAG_SKINNED, GpuCluster, MorphEntry, NO_LINK};
use crate::import_cache::Hasher128;

/// Format written now (2: directory entries carry the raw lengths and a codec, so blobs can be zstd compressed).
pub const PAGE_FORMAT: u32 = 2;
/// Oldest format [`PagedFile`] reads (1: raw blobs, 64-byte directory entries).
pub const PAGE_FORMAT_MIN: u32 = 1;
const PAGE_MAGIC: &[u8; 4] = b"ZPG1";

/// Bytes of one quantised vertex.
pub const VERTEX_BYTES: usize = size_of::<ClusterVertex>();
/// Default payload bytes of a page (the splitter packs whole input sets up to this).
pub const DEFAULT_PAGE_BYTES: usize = 96 * 1024;
/// Default budget of the always-resident coarse levels.
pub const DEFAULT_RESIDENT_BYTES: usize = 512 * 1024;
/// Meshes with less payload than this stay one resident page.
pub const DEFAULT_MIN_STREAM_BYTES: usize = 2 * 1024 * 1024;
/// Most pages a mesh may have (the renderer's page table is bounded).
pub const MAX_MESH_PAGES: usize = 1 << 20;
/// Largest payload of one page the loader accepts (an input set of 32 clusters
/// at the format's maxima is below 0.5 MB for static meshes; blendshape entries
/// of a dense avatar can make one input set a few MB).
pub const MAX_PAGE_BYTES: usize = 32 * 1024 * 1024;
/// Bytes of the skin record of one vertex in a skinned page block.
pub const SKIN_BYTES: usize = size_of::<ClusterVertexSkin>();

/// How to split a DAG.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PageOptions {
    /// Payload bytes a page is packed to (a single input set may exceed it).
    pub page_bytes: usize,
    /// Coarse top levels up to this many payload bytes form the resident root page.
    pub resident_bytes: usize,
    /// Meshes under this payload size are not split at all.
    pub min_stream_bytes: usize,
}

impl Default for PageOptions {
    fn default() -> Self {
        Self { page_bytes: DEFAULT_PAGE_BYTES, resident_bytes: DEFAULT_RESIDENT_BYTES, min_stream_bytes: DEFAULT_MIN_STREAM_BYTES }
    }
}

/// Payload bytes of one cluster: (vertices, triangle bytes padded to 4).
pub fn cluster_payload(c: &GpuCluster) -> (u32, u32) {
    (u32::from(c.vertex_count), (u32::from(c.triangle_count) * 3 + 3) & !3)
}

/// Whether the DAG carries skin weights or blendshapes (their data pages with the clusters).
pub fn is_skinned(h: &DagHeader) -> bool {
    h.flags & FLAG_SKINNED != 0 || h.morph_targets > 0
}

/// Words of a skinned cluster's aux record: the morph count, the joint ids (u16 pairs) and the morph entries.
pub fn cluster_aux_words(c: &GpuCluster) -> u64 {
    1 + u64::from(c.bone_count).div_ceil(2) + u64::from(c.morph_count) * (size_of::<MorphEntry>() as u64 / 4)
}

/// What a cluster adds to its page: vertices and triangle-block bytes (see the module docs).
fn cluster_block(skinned: bool, c: &GpuCluster) -> (u32, u64) {
    let (v, t) = cluster_payload(c);
    let extra = if skinned { u64::from(v) * SKIN_BYTES as u64 + 4 * cluster_aux_words(c) } else { 0 };
    (v, u64::from(t) + extra)
}

/// Where each cluster's data starts inside its page (cluster order of the page).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClusterSpan {
    /// First vertex, relative to the page.
    pub vertex: u32,
    /// Triangle bytes, relative to the start of the triangle region.
    pub tri: u32,
    /// Aux record, in words from the start of the triangle block (skinned pages).
    pub aux_word: u32,
}

/// Layout of a page's triangle block.
pub struct PageSpans {
    /// Bytes of the skin region at the start of the block (0 for static meshes).
    pub skin_bytes: u32,
    pub spans: Vec<ClusterSpan>,
}

/// The block layout of `info` under `skel`'s cluster records (the sizes are in `info`, from the same records).
pub fn page_spans(skel: &ClusterDag, info: &PageInfo) -> PageSpans {
    let skinned = is_skinned(&skel.header);
    let skin_bytes = if skinned { info.vertices * SKIN_BYTES as u32 } else { 0 };
    let tri_total: u32 = info.clusters.iter().map(|&c| cluster_payload(&skel.clusters[c as usize]).1).sum();
    let (mut v, mut t, mut aux) = (0u32, 0u32, (skin_bytes + tri_total) / 4);
    let mut spans = Vec::with_capacity(info.clusters.len());
    for &c in &info.clusters {
        let cl = &skel.clusters[c as usize];
        spans.push(ClusterSpan { vertex: v, tri: t, aux_word: aux });
        let (cv, ct) = cluster_payload(cl);
        v += cv;
        t += ct;
        if skinned {
            aux = aux.wrapping_add(cluster_aux_words(cl) as u32);
        }
    }
    PageSpans { skin_bytes, spans }
}

/// Largest displacement of each blendshape target (the pose margin of the renderer;
/// `ClusterMeshUpload::morph_reach` computes the same from an upload).
pub fn morph_reach_of(dag: &ClusterDag) -> Vec<f32> {
    let mut reach = vec![0.0f32; dag.header.morph_targets as usize];
    for e in &dag.morph_entries {
        if let Some(r) = reach.get_mut(e.target as usize) {
            let d = e.d_pos.map(|b| half::f16::from_bits(b).to_f32());
            *r = r.max((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt());
        }
    }
    reach
}

/// One page.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PageInfo {
    /// Finest DAG level in the page (the root page: the lowest level it holds).
    pub level: u32,
    /// Clusters, ascending.
    pub clusters: Vec<u32>,
    pub vertices: u32,
    pub tri_bytes: u32,
    /// Pages that must be resident when this one is (rule R2), ascending.
    pub deps: Vec<u32>,
}

impl PageInfo {
    pub fn bytes(&self) -> usize {
        self.vertices as usize * VERTEX_BYTES + self.tri_bytes as usize
    }
}

/// Page of every cluster and the graph between pages.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PagePlan {
    /// Page 0 is the always-resident root page.
    pub pages: Vec<PageInfo>,
    pub cluster_page: Vec<u32>,
    /// Page holding the inputs of the group that produced each cluster ([`NO_LINK`] for the finest level).
    pub child_page: Vec<u32>,
    /// Largest displacement of each blendshape target. The morph entries live in the
    /// pages, but the renderer's pose margin needs this before any page is read, so the
    /// paged file stores it beside the plan (empty without blendshapes).
    pub morph_reach: Vec<f32>,
}

/// Sizes of a plan, for reports and budgets.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PlanStats {
    pub pages: usize,
    pub root_bytes: usize,
    pub streamed_bytes: usize,
    pub largest_page: usize,
    pub mean_page: usize,
    pub max_deps: usize,
}

impl PagePlan {
    /// A plan with everything on the root page (small meshes, skinned meshes, old caches).
    pub fn single(dag: &ClusterDag) -> Self {
        Self::from_assignment(dag, vec![0; dag.clusters.len()]).expect("one page is always a valid plan")
    }

    pub fn is_single(&self) -> bool {
        self.pages.len() <= 1
    }

    pub fn stats(&self) -> PlanStats {
        let root_bytes = self.pages.first().map_or(0, PageInfo::bytes);
        let rest: Vec<usize> = self.pages.iter().skip(1).map(PageInfo::bytes).collect();
        PlanStats {
            pages: self.pages.len(),
            root_bytes,
            streamed_bytes: rest.iter().sum(),
            largest_page: rest.iter().copied().max().unwrap_or(0),
            mean_page: if rest.is_empty() { 0 } else { rest.iter().sum::<usize>() / rest.len() },
            max_deps: self.pages.iter().map(|p| p.deps.len()).max().unwrap_or(0),
        }
    }

    /// Rebuild everything derived (page contents, sizes, `child_page`, `deps`)
    /// from the cluster -> page assignment and check the invariants the
    /// renderer depends on. `Err` names the first violation.
    pub fn from_assignment(dag: &ClusterDag, cluster_page: Vec<u32>) -> Result<Self, String> {
        macro_rules! check {
            ($c:expr, $($m:tt)*) => {
                if !($c) {
                    return Err(format!($($m)*));
                }
            };
        }
        let nc = dag.clusters.len();
        check!(cluster_page.len() == nc && dag.links.len() == nc, "page assignment length differs from the cluster count");
        let np = cluster_page.iter().copied().max().map_or(0, |m| m as usize + 1);
        check!((1..=MAX_MESH_PAGES).contains(&np), "page count {np} out of range");
        let skinned = is_skinned(&dag.header);
        let mut pages = vec![PageInfo { level: u32::MAX, ..Default::default() }; np];
        for (c, &p) in cluster_page.iter().enumerate() {
            let pg = &mut pages[p as usize];
            pg.clusters.push(c as u32);
            let (v, t) = cluster_block(skinned, &dag.clusters[c]);
            pg.vertices = pg.vertices.checked_add(v).ok_or("page vertex count overflows")?;
            pg.tri_bytes = u32::try_from(u64::from(pg.tri_bytes) + t).map_err(|_| "page triangle bytes overflow")?;
            pg.level = pg.level.min(dag.links[c].level);
        }
        for (i, pg) in pages.iter().enumerate() {
            check!(!pg.clusters.is_empty(), "page {i} is empty");
            // A one-page plan is the whole mesh resident (it has no budget to respect).
            check!(np == 1 || pg.bytes() <= MAX_PAGE_BYTES, "page {i} holds {} bytes", pg.bytes());
        }
        // Rule R: the inputs of a group share a page. child_page: where they are.
        let mut child_page = vec![NO_LINK; nc];
        for (c, l) in dag.links.iter().enumerate() {
            if l.group == NO_LINK {
                continue;
            }
            let g = dag.groups.get(l.group as usize).ok_or("producer group out of range")?;
            let (f, n) = (g.first_child as usize, g.child_count as usize);
            let kids = dag.group_children.get(f..f.checked_add(n).ok_or("group children overflow")?).ok_or("group children out of range")?;
            let first = *kids.first().ok_or("a group without children")?;
            let page = *cluster_page.get(first as usize).ok_or("group child out of range")?;
            check!(kids.iter().all(|&k| cluster_page.get(k as usize) == Some(&page)), "inputs of group {} are split over pages", l.group);
            child_page[c] = page;
        }
        // Rule R2: the outputs of a cluster's consuming group are pages it needs.
        for (p, pg) in pages.iter_mut().enumerate() {
            let mut deps: Vec<u32> = Vec::new();
            for &c in &pg.clusters {
                let pgrp = dag.links[c as usize].parent_group;
                if pgrp == NO_LINK {
                    continue;
                }
                let g = &dag.groups[pgrp as usize];
                let (f, n) = (g.first_output as usize, g.output_count as usize);
                check!(f.checked_add(n).is_some_and(|e| e <= nc), "group outputs out of range");
                for q in &cluster_page[f..f + n] {
                    let q = *q;
                    if q as usize != p {
                        deps.push(q);
                    }
                }
            }
            deps.sort_unstable();
            deps.dedup();
            pg.deps = deps;
        }
        // The root page has no dependencies and holds every root cluster; all
        // other dependencies point strictly up the DAG, so loading can never cycle.
        check!(pages[0].deps.is_empty(), "the root page depends on another page");
        for (c, l) in dag.links.iter().enumerate() {
            check!(l.parent_group != NO_LINK || cluster_page[c] == 0, "root cluster {c} is not on the root page");
        }
        for (i, pg) in pages.iter().enumerate().skip(1) {
            for &d in &pg.deps {
                check!(d == 0 || pages[d as usize].level > pg.level, "page {i} depends on page {d}, which is not above it");
            }
        }
        let morph_reach = if dag.morph_entries.is_empty() { vec![0.0; dag.header.morph_targets as usize] } else { morph_reach_of(dag) };
        Ok(Self { pages, cluster_page, child_page, morph_reach })
    }

    /// Split `dag` into pages. Deterministic (a pure function of the DAG and
    /// options). A small DAG stays one
    /// page. Skinned and blendshape DAGs split like static ones (their skin data
    /// travels in the page, see the module docs).
    pub fn split(dag: &ClusterDag, opts: &PageOptions) -> Self {
        let nc = dag.clusters.len();
        let streamable = !dag.groups.is_empty() && nc > 0;
        let skinned = is_skinned(&dag.header);
        let bytes_of = |c: usize| {
            let (v, t) = cluster_block(skinned, &dag.clusters[c]);
            v as usize * VERTEX_BYTES + t as usize
        };
        let total: usize = (0..nc).map(bytes_of).sum();
        if !streamable || total < opts.min_stream_bytes.max(1) || dag.links.len() != nc {
            return Self::single(dag);
        }
        let top = dag.links.iter().map(|l| l.level).max().unwrap_or(0);
        let mut level_bytes = vec![0usize; top as usize + 1];
        for c in 0..nc {
            level_bytes[dag.links[c].level as usize] += bytes_of(c);
        }
        // The lowest level L such that the levels >= L fit the resident budget (at least the top level).
        let mut root_from = top;
        let mut acc = level_bytes[top as usize];
        while root_from > 0 && acc + level_bytes[root_from as usize - 1] <= opts.resident_bytes {
            root_from -= 1;
            acc += level_bytes[root_from as usize];
        }
        if root_from == 0 {
            return Self::single(dag);
        }
        let mut cluster_page = vec![0u32; nc];
        // Groups by the level of their inputs, in group order (spatially coherent: the builder
        // forms groups from neighbouring clusters).
        let mut by_level: Vec<Vec<u32>> = vec![Vec::new(); root_from as usize];
        for (gi, g) in dag.groups.iter().enumerate() {
            if g.level >= 1 && g.level - 1 < root_from {
                by_level[g.level as usize - 1].push(gi as u32);
            }
        }
        let mut next = 1u32;
        for groups in &by_level {
            let mut used = 0usize;
            let mut open = false;
            for &gi in groups {
                let g = &dag.groups[gi as usize];
                let kids = &dag.group_children[g.first_child as usize..][..g.child_count as usize];
                let set: usize = kids.iter().map(|&k| bytes_of(k as usize)).sum();
                if open && used + set > opts.page_bytes {
                    next += 1;
                    used = 0;
                }
                open = true;
                used += set;
                for &k in kids {
                    cluster_page[k as usize] = next;
                }
            }
            if open {
                next += 1;
            }
        }
        match Self::from_assignment(dag, cluster_page) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!("cluster pages: the split failed its own checks ({e}); keeping one page");
                Self::single(dag)
            }
        }
    }

    /// Rewrite the clusters for the GPU: `vertex_offset` / `triangle_offset`
    /// relative to the page (the triangle offset to the triangle region, after the
    /// skin region of a skinned page), `morph_offset` = the page, `morph_count` = the
    /// page of the children ([`NO_LINK`] for the finest level), and for a skinned
    /// DAG `bone_offset` = the word offset of the cluster's aux record in the page's
    /// triangle block (its morph count, joint ids and morph entries).
    pub fn gpu_clusters(&self, dag: &ClusterDag) -> Vec<GpuCluster> {
        let mut out = dag.clusters.clone();
        let skinned = is_skinned(&dag.header);
        for pg in &self.pages {
            let spans = page_spans(dag, pg);
            for (&c, sp) in pg.clusters.iter().zip(&spans.spans) {
                let cl = &mut out[c as usize];
                cl.vertex_offset = sp.vertex;
                cl.triangle_offset = sp.tri;
                cl.morph_offset = self.cluster_page[c as usize];
                cl.morph_count = self.child_page[c as usize];
                if skinned {
                    cl.bone_offset = sp.aux_word;
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------- renderer upload

#[cfg(feature = "client")]
impl PagePlan {
    /// The renderer's upload for a streamed mesh: the skeleton's clusters
    /// relocated page-relative, the BVH, the root page's payload and the page
    /// graph. `skeleton` needs no vertices (see [`skeleton_of`]).
    pub fn to_gpu_upload(&self, skeleton: &ClusterDag, root: &PagePayload) -> zestengine_render::ClusterMeshUpload {
        zestengine_render::ClusterMeshUpload {
            clusters: bytemuck::cast_slice(&self.gpu_clusters(skeleton)).to_vec(),
            vertices: bytemuck::cast_slice(&root.vertices).to_vec(),
            triangles: root.triangles.clone(),
            nodes: bytemuck::cast_slice(&skeleton.nodes).to_vec(),
            mesh_min: skeleton.header.mesh_min,
            step: skeleton.header.quant_step,
            stream: Some(zestengine_render::StreamLayout {
                pages: self.pages.iter().map(|p| zestengine_render::PageLayout { vertices: p.vertices, tri_bytes: p.tri_bytes, deps: p.deps.clone() }).collect(),
            }),
            skinned: is_skinned(&skeleton.header),
            morph_targets: skeleton.header.morph_targets,
            morph_reach: self.morph_reach.clone(),
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------- reference cut

/// A DAG with its page plan; the reference implementation of the streaming
/// cut (what `cluster_cull.comp` computes).
pub struct PagedLayout<'a> {
    pub dag: &'a ClusterDag,
    pub plan: &'a PagePlan,
}

/// Result of [`PagedLayout::select`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StreamCut {
    /// Clusters to draw, ascending.
    pub clusters: Vec<u32>,
    /// Pages the view wants but are missing, with the largest projected-error
    /// ratio (>= 1: the cluster is too coarse for the view) that asked for them.
    pub requests: Vec<(u32, f32)>,
}

impl PagedLayout<'_> {
    /// The cut under `ok(error, sphere)` ("accurate enough") with `resident`
    /// per page. `ratio(error, sphere)` is the projected error over the target
    /// (the request priority). Page 0 must be resident.
    pub fn select(&self, ok: &dyn Fn(f32, [f32; 4]) -> bool, ratio: &dyn Fn(f32, [f32; 4]) -> f32, resident: &[bool]) -> StreamCut {
        let mut cut = StreamCut::default();
        let mut want: std::collections::BTreeMap<u32, f32> = Default::default();
        for (i, c) in self.dag.clusters.iter().enumerate() {
            if !resident[self.plan.cluster_page[i] as usize] {
                continue;
            }
            let child = self.plan.child_page[i];
            let fine_ok = ok(c.lod_error, c.lod_sphere);
            let child_missing = child != NO_LINK && !resident[child as usize];
            if (fine_ok || child_missing) && !ok(c.parent_lod_error, c.parent_lod_sphere) {
                cut.clusters.push(i as u32);
                if !fine_ok && child_missing {
                    let r = ratio(c.lod_error, c.lod_sphere);
                    let e = want.entry(child).or_insert(0.0);
                    *e = e.max(r);
                }
            }
        }
        cut.requests = want.into_iter().collect();
        cut
    }

    /// Pages a page needs resident, transitively (parents first, then `page`).
    pub fn closure(&self, page: u32) -> Vec<u32> {
        let mut out = Vec::new();
        let mut seen = vec![false; self.plan.pages.len()];
        fn visit(plan: &PagePlan, p: u32, seen: &mut [bool], out: &mut Vec<u32>) {
            if std::mem::replace(&mut seen[p as usize], true) {
                return;
            }
            for &d in &plan.pages[p as usize].deps {
                visit(plan, d, seen, out);
            }
            out.push(p);
        }
        visit(self.plan, page, &mut seen, &mut out);
        out
    }
}

// ---------------------------------------------------------------- payloads

/// The heavy data of one page, in the page's cluster order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PagePayload {
    pub vertices: Vec<ClusterVertex>,
    pub triangles: Vec<u8>,
}

impl PagePayload {
    pub fn bytes(&self) -> usize {
        self.vertices.len() * VERTEX_BYTES + self.triangles.len()
    }

    /// Vertices then triangles, the layout the GPU copy takes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(self.bytes());
        b.extend_from_slice(bytemuck::cast_slice(&self.vertices));
        b.extend_from_slice(&self.triangles);
        b
    }

    /// The inverse of [`to_bytes`](Self::to_bytes) for page `page` of a plan.
    pub fn from_bytes(info: &PageInfo, bytes: &[u8]) -> Result<Self, String> {
        let vb = info.vertices as usize * VERTEX_BYTES;
        if bytes.len() != vb + info.tri_bytes as usize {
            return Err(format!("page payload has {} bytes, the plan says {}", bytes.len(), vb + info.tri_bytes as usize));
        }
        Ok(Self { vertices: bytemuck::pod_collect_to_vec(&bytes[..vb]), triangles: bytes[vb..].to_vec() })
    }
}

/// Check a page's payload against the skeleton: counts, triangle indices inside
/// their cluster's vertices, vertices inside the cluster's bounds, and for a skinned
/// DAG the skin records (joint slots inside the cluster's bone list) and the aux
/// records (counts, blendshape entries inside the cluster). A page that fails is
/// never uploaded.
pub fn validate_payload(skeleton: &ClusterDag, plan: &PagePlan, page: u32, p: &PagePayload) -> Result<(), String> {
    let info = plan.pages.get(page as usize).ok_or("page out of range")?;
    if p.vertices.len() != info.vertices as usize || p.triangles.len() != info.tri_bytes as usize {
        return Err(format!("page {page}: payload sizes differ from the plan"));
    }
    let slack = skeleton.header.quant_step * 4.0;
    let skinned = is_skinned(&skeleton.header);
    let spans = page_spans(skeleton, info);
    let tri_base = spans.skin_bytes as usize;
    for (&c, sp) in info.clusters.iter().zip(&spans.spans) {
        let cl = &skeleton.clusters[c as usize];
        let (v, t) = (sp.vertex as usize, sp.tri as usize);
        let verts = &p.vertices[v..v + cl.vertex_count as usize];
        let tris = &p.triangles[tri_base + t..][..cl.triangle_count as usize * 3];
        if !tris.iter().all(|&i| (i as u16) < cl.vertex_count) {
            return Err(format!("page {page}: cluster {c} has a triangle index past its vertices"));
        }
        for vert in verts {
            let q = skeleton.header.decode_position(vert.pos);
            let d = ((q[0] - cl.bounds[0]).powi(2) + (q[1] - cl.bounds[1]).powi(2) + (q[2] - cl.bounds[2]).powi(2)).sqrt();
            if !d.partial_cmp(&(cl.bounds[3] * (1.0 + 1e-4) + slack)).is_some_and(std::cmp::Ordering::is_le) {
                return Err(format!("page {page}: cluster {c} has a vertex outside its bounds"));
            }
        }
        if skinned {
            let skin = &p.triangles[v * SKIN_BYTES..][..cl.vertex_count as usize * SKIN_BYTES];
            for rec in skin.as_chunks::<SKIN_BYTES>().0 {
                if (0..4).any(|k| rec[4 + k] != 0 && u16::from(rec[k]) >= cl.bone_count) {
                    return Err(format!("page {page}: cluster {c} has a skin index past its bone list"));
                }
            }
            let aux = block_aux(p, cl, sp).map_err(|e| format!("page {page}: cluster {c}: {e}"))?;
            if aux.morphs.as_chunks::<{ size_of::<MorphEntry>() }>().0.iter().any(|m| {
                let (target, vertex) = (u16::from_le_bytes([m[0], m[1]]), u16::from_le_bytes([m[2], m[3]]));
                u32::from(target) >= skeleton.header.morph_targets || vertex >= cl.vertex_count
            }) {
                return Err(format!("page {page}: cluster {c} has a blendshape entry outside the cluster"));
            }
        }
    }
    Ok(())
}

/// A skinned cluster's aux record inside a page block.
struct Aux<'a> {
    /// Joint ids, one per bone slot.
    bones: Vec<u16>,
    /// Sparse blendshape entries (16 bytes each).
    morphs: &'a [u8],
}

/// Parse the aux record of `cl` (cluster record from the skeleton, span in its page).
fn block_aux<'a>(p: &'a PagePayload, cl: &GpuCluster, sp: &ClusterSpan) -> Result<Aux<'a>, String> {
    let at = sp.aux_word as usize * 4;
    let words = cluster_aux_words(cl) as usize;
    let block = p.triangles.get(at..at.checked_add(words * 4).ok_or("aux overflows")?).ok_or("aux record outside the page")?;
    if u32::from_le_bytes(block[..4].try_into().unwrap()) != cl.morph_count {
        return Err("aux morph count differs from the cluster's".into());
    }
    let nb = cl.bone_count as usize;
    let nbw = nb.div_ceil(2);
    let bones = (0..nb)
        .map(|i| {
            let w = u32::from_le_bytes(block[4 + (i / 2) * 4..8 + (i / 2) * 4].try_into().unwrap());
            if i % 2 == 0 { w as u16 } else { (w >> 16) as u16 }
        })
        .collect();
    Ok(Aux { bones, morphs: &block[4 + nbw * 4..] })
}

/// Where page payloads come from (the streamer's worker reads through this).
pub trait PageSource: Send + Sync {
    /// The hot payload of `page` (validated).
    fn read_page(&self, page: u32) -> Result<PagePayload, String>;
}

/// Pages cut out of a DAG in memory.
pub struct MemPageSource {
    pub dag: Arc<ClusterDag>,
    pub plan: Arc<PagePlan>,
}

impl PageSource for MemPageSource {
    fn read_page(&self, page: u32) -> Result<PagePayload, String> {
        let info = self.plan.pages.get(page as usize).ok_or("page out of range")?;
        Ok(gather_payload(&self.dag, info))
    }
}

/// Cut a page's payload out of a full DAG.
fn gather_payload(dag: &ClusterDag, info: &PageInfo) -> PagePayload {
    let skinned = is_skinned(&dag.header);
    let mut out = PagePayload { vertices: Vec::with_capacity(info.vertices as usize), triangles: Vec::new() };
    let mut skin: Vec<u8> = Vec::new();
    let mut tris: Vec<u8> = Vec::new();
    let mut aux: Vec<u8> = Vec::new();
    for &c in &info.clusters {
        let cl = &dag.clusters[c as usize];
        let (_, ct) = cluster_payload(cl);
        let vo = cl.vertex_offset as usize;
        out.vertices.extend_from_slice(&dag.vertices[vo..][..cl.vertex_count as usize]);
        tris.extend_from_slice(&dag.triangles[cl.triangle_offset as usize..][..ct as usize]);
        if skinned {
            if dag.vertex_skin.is_empty() {
                skin.resize(skin.len() + cl.vertex_count as usize * SKIN_BYTES, 0);
            } else {
                skin.extend_from_slice(bytemuck::cast_slice(&dag.vertex_skin[vo..][..cl.vertex_count as usize]));
            }
            aux.extend_from_slice(&cl.morph_count.to_le_bytes());
            let bones = &dag.bone_ids[cl.bone_offset as usize..][..cl.bone_count as usize];
            for pair in bones.chunks(2) {
                let w = u32::from(pair[0]) | pair.get(1).map_or(0, |&b| u32::from(b) << 16);
                aux.extend_from_slice(&w.to_le_bytes());
            }
            aux.extend_from_slice(bytemuck::cast_slice(&dag.morph_entries[cl.morph_offset as usize..][..cl.morph_count as usize]));
        }
    }
    out.triangles = skin;
    out.triangles.extend_from_slice(&tris);
    out.triangles.extend_from_slice(&aux);
    out
}

// ---------------------------------------------------------------- file format

/// A directory entry of format 1 (raw blobs).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
struct DirEntryV1 {
    hot_off: u64,
    hot_len: u64,
    cold_off: u64,
    cold_len: u64,
    hot_sum: [u64; 2],
    cold_sum: [u64; 2],
}

/// A directory entry of format 2. `*_len` are the stored bytes (what the sums
/// cover), `*_raw` the decoded sizes; `codec` bit 0 = hot blob zstd, bit 1 = cold blob zstd.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
struct DirEntry {
    hot_off: u64,
    hot_len: u64,
    cold_off: u64,
    cold_len: u64,
    hot_sum: [u64; 2],
    cold_sum: [u64; 2],
    hot_raw: u32,
    cold_raw: u32,
    codec: u32,
    pad: u32,
}

const CODEC_HOT_ZSTD: u32 = 1;
const CODEC_COLD_ZSTD: u32 = 2;

impl From<DirEntryV1> for DirEntry {
    fn from(e: DirEntryV1) -> Self {
        Self {
            hot_off: e.hot_off,
            hot_len: e.hot_len,
            cold_off: e.cold_off,
            cold_len: e.cold_len,
            hot_sum: e.hot_sum,
            cold_sum: e.cold_sum,
            hot_raw: e.hot_len as u32,
            cold_raw: e.cold_len as u32,
            codec: 0,
            pad: 0,
        }
    }
}

/// How the blobs of a paged file are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageCodec {
    Raw,
    /// zstd at this level (a blob that does not shrink is stored raw).
    Zstd(i32),
}

impl Default for PageCodec {
    /// zstd level 3 (about 1 GB/s to decode), unless `ZESTENGINE_GEO_PAGE_CODEC=raw`.
    fn default() -> Self {
        match std::env::var("ZESTENGINE_GEO_PAGE_CODEC").as_deref() {
            Ok("raw") => PageCodec::Raw,
            _ => PageCodec::Zstd(3),
        }
    }
}

/// Store `raw` under `codec`: (bytes, whether they are zstd).
fn pack_blob(raw: &[u8], codec: PageCodec) -> (Vec<u8>, bool) {
    if let PageCodec::Zstd(level) = codec
        && let Ok(z) = zstd::bulk::compress(raw, level)
        && z.len() < raw.len()
    {
        return (z, true);
    }
    (raw.to_vec(), false)
}

/// Decode a stored blob to exactly `raw_len` bytes.
fn unpack_blob(stored: Vec<u8>, zstd_coded: bool, raw_len: u32) -> Result<Vec<u8>, String> {
    if !zstd_coded {
        return if stored.len() == raw_len as usize { Ok(stored) } else { Err("stored blob has the wrong length".into()) };
    }
    let out = zstd::bulk::decompress(&stored, raw_len as usize).map_err(|e| format!("zstd: {e}"))?;
    if out.len() == raw_len as usize { Ok(out) } else { Err("decoded blob has the wrong length".into()) }
}

const HEADER_BYTES: usize = 4 + 4 + 8 + 8 + 8 + 16 + 8 + 8;

fn sum128(bytes: &[u8]) -> [u64; 2] {
    let mut h = Hasher128::new();
    h.update(bytes);
    let v = h.finish();
    [v as u64, (v >> 64) as u64]
}

/// The paged file for `dag` under `plan` (the full DAG, vertices included).
pub fn encode_paged(dag: &ClusterDag, plan: &PagePlan) -> Vec<u8> {
    encode_paged_with(dag, plan, PageCodec::default())
}

/// [`encode_paged`] with an explicit blob codec.
pub fn encode_paged_with(dag: &ClusterDag, plan: &PagePlan, codec: PageCodec) -> Vec<u8> {
    encode_paged_format(dag, plan, codec, PAGE_FORMAT)
}

/// The writer for `format` 1 (raw only; tests keep old files readable) or 2.
fn encode_paged_format(dag: &ClusterDag, plan: &PagePlan, codec: PageCodec, format: u32) -> Vec<u8> {
    assert!(format == PAGE_FORMAT || (format == 1 && codec == PageCodec::Raw), "format 1 has no compression");
    let entry_bytes = if format == 1 { size_of::<DirEntryV1>() } else { size_of::<DirEntry>() };
    let skeleton = skeleton_of(dag);
    let skel = skeleton.to_bytes();
    let mut plan_b = Vec::new();
    plan_b.extend_from_slice(&(plan.pages.len() as u32).to_le_bytes());
    plan_b.extend_from_slice(&(plan.cluster_page.len() as u32).to_le_bytes());
    plan_b.extend_from_slice(&(dag.vertices.len() as u64).to_le_bytes());
    plan_b.extend_from_slice(&(dag.triangles.len() as u64).to_le_bytes());
    plan_b.extend_from_slice(bytemuck::cast_slice(&plan.cluster_page));
    // Blendshape reach (one f32 per target; nothing for a static mesh, so format-2 static files are unchanged).
    plan_b.extend_from_slice(bytemuck::cast_slice(&plan.morph_reach));
    let mut blobs: Vec<u8> = Vec::new();
    let mut dir = Vec::with_capacity(plan.pages.len());
    let dir_len = plan.pages.len() * entry_bytes;
    let base = (HEADER_BYTES + skel.len() + plan_b.len() + dir_len) as u64;
    for info in &plan.pages {
        let hot_raw = gather_payload(dag, info).to_bytes();
        let cold_raw: Vec<u8> = info
            .clusters
            .iter()
            .flat_map(|&c| {
                let cl = &dag.clusters[c as usize];
                dag.vertex_source[cl.vertex_offset as usize..][..cl.vertex_count as usize].iter().flat_map(|s| s.to_le_bytes())
            })
            .collect();
        let (hot, hot_z) = pack_blob(&hot_raw, codec);
        let (cold, cold_z) = pack_blob(&cold_raw, codec);
        let e = DirEntry {
            hot_off: base + blobs.len() as u64,
            hot_len: hot.len() as u64,
            cold_off: base + (blobs.len() + hot.len()) as u64,
            cold_len: cold.len() as u64,
            hot_sum: sum128(&hot),
            cold_sum: sum128(&cold),
            hot_raw: hot_raw.len() as u32,
            cold_raw: cold_raw.len() as u32,
            codec: (u32::from(hot_z) * CODEC_HOT_ZSTD) | (u32::from(cold_z) * CODEC_COLD_ZSTD),
            pad: 0,
        };
        blobs.extend_from_slice(&hot);
        blobs.extend_from_slice(&cold);
        dir.push(e);
    }
    let mut out = Vec::with_capacity(base as usize + blobs.len());
    out.extend_from_slice(PAGE_MAGIC);
    out.extend_from_slice(&format.to_le_bytes());
    out.extend_from_slice(&(skel.len() as u64).to_le_bytes());
    out.extend_from_slice(&(plan_b.len() as u64).to_le_bytes());
    out.extend_from_slice(&(dir_len as u64).to_le_bytes());
    let dir_b: Vec<u8> = if format == 1 {
        let v1: Vec<DirEntryV1> = dir
            .iter()
            .map(|e| DirEntryV1 { hot_off: e.hot_off, hot_len: e.hot_len, cold_off: e.cold_off, cold_len: e.cold_len, hot_sum: e.hot_sum, cold_sum: e.cold_sum })
            .collect();
        bytemuck::cast_slice(&v1).to_vec()
    } else {
        bytemuck::cast_slice(&dir).to_vec()
    };
    let mut sums = skel.clone();
    sums.extend_from_slice(&plan_b);
    sums.extend_from_slice(&dir_b);
    let s = sum128(&sums);
    out.extend_from_slice(&s[0].to_le_bytes());
    out.extend_from_slice(&s[1].to_le_bytes());
    out.extend_from_slice(&(blobs.len() as u64).to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&skel);
    out.extend_from_slice(&plan_b);
    out.extend_from_slice(&dir_b);
    out.extend_from_slice(&blobs);
    out
}

/// The cluster records, links, groups and BVH without any payload.
pub fn skeleton_of(dag: &ClusterDag) -> ClusterDag {
    ClusterDag {
        header: dag.header.clone(),
        clusters: dag.clusters.clone(),
        links: dag.links.clone(),
        groups: dag.groups.clone(),
        group_children: dag.group_children.clone(),
        nodes: dag.nodes.clone(),
        ..Default::default()
    }
}

/// Where the bytes of a paged mesh are read from.
#[derive(Clone)]
enum Backing {
    Mem(Arc<[u8]>),
    File { path: std::path::PathBuf, base: u64 },
}

impl Backing {
    fn read(&self, off: u64, len: u64) -> Result<Vec<u8>, String> {
        match self {
            Backing::Mem(b) => {
                let end = off.checked_add(len).ok_or("read overflows")?;
                b.get(off as usize..end as usize).map(<[u8]>::to_vec).ok_or_else(|| "read past the end of the paged mesh".into())
            }
            Backing::File { path, base } => {
                use std::io::{Read, Seek, SeekFrom};
                let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
                f.seek(SeekFrom::Start(base.checked_add(off).ok_or("read overflows")?)).map_err(|e| e.to_string())?;
                let mut buf = vec![0u8; len as usize];
                f.read_exact(&mut buf).map_err(|e| format!("{}: {e}", path.display()))?;
                Ok(buf)
            }
        }
    }
}

/// A paged mesh opened from bytes or a file: the skeleton and plan are in
/// memory, the page blobs are read (and checksummed) on demand.
pub struct PagedFile {
    pub skeleton: ClusterDag,
    pub plan: PagePlan,
    dir: Vec<DirEntry>,
    total_vertices: u64,
    total_tri_bytes: u64,
    backing: Backing,
}

impl PagedFile {
    /// Parse a paged mesh held in memory (all of `bytes`).
    pub fn from_bytes(bytes: Arc<[u8]>) -> Result<Self, String> {
        Self::open(Backing::Mem(bytes), None)
    }

    /// Open the paged mesh stored at `base .. base + len` of `path`; only the
    /// header, skeleton, plan and directory are read now.
    pub fn open_file(path: &std::path::Path, base: u64, len: u64) -> Result<Self, String> {
        Self::open(Backing::File { path: path.to_path_buf(), base }, Some(len))
    }

    fn open(backing: Backing, total: Option<u64>) -> Result<Self, String> {
        let head = backing.read(0, HEADER_BYTES as u64)?;
        if &head[..4] != PAGE_MAGIC {
            return Err("not a paged cluster mesh".into());
        }
        let u32_at = |at: usize| u32::from_le_bytes(head[at..at + 4].try_into().unwrap());
        let u64_at = |at: usize| u64::from_le_bytes(head[at..at + 8].try_into().unwrap());
        let format = u32_at(4);
        if !(PAGE_FORMAT_MIN..=PAGE_FORMAT).contains(&format) {
            return Err("paged cluster mesh format differs".into());
        }
        let entry_bytes = if format == 1 { size_of::<DirEntryV1>() } else { size_of::<DirEntry>() };
        let (skel_len, plan_len, dir_len) = (u64_at(8), u64_at(16), u64_at(24));
        let sum = [u64_at(32), u64_at(40)];
        let blob_len = u64_at(48);
        let meta_len = skel_len.checked_add(plan_len).and_then(|v| v.checked_add(dir_len)).ok_or("lengths overflow")?;
        let whole = (HEADER_BYTES as u64).checked_add(meta_len).and_then(|v| v.checked_add(blob_len)).ok_or("lengths overflow")?;
        if meta_len > MAX_META_BYTES || total.is_some_and(|t| t != whole) {
            return Err("paged cluster mesh has the wrong size".into());
        }
        if let Backing::Mem(b) = &backing
            && b.len() as u64 != whole
        {
            return Err("paged cluster mesh has the wrong size".into());
        }
        let meta = backing.read(HEADER_BYTES as u64, meta_len)?;
        if sum128(&meta) != sum {
            return Err("paged cluster mesh header checksum differs".into());
        }
        let (skel_b, rest) = meta.split_at(skel_len as usize);
        let (plan_b, dir_b) = rest.split_at(plan_len as usize);
        let skeleton = ClusterDag::from_bytes(skel_b)?;
        skeleton.validate_skeleton()?;
        if plan_b.len() < 24 || dir_b.len() % entry_bytes != 0 {
            return Err("paged cluster mesh tables are malformed".into());
        }
        let np = u32::from_le_bytes(plan_b[0..4].try_into().unwrap()) as usize;
        let nc = u32::from_le_bytes(plan_b[4..8].try_into().unwrap()) as usize;
        let total_vertices = u64::from_le_bytes(plan_b[8..16].try_into().unwrap());
        let total_tri_bytes = u64::from_le_bytes(plan_b[16..24].try_into().unwrap());
        let targets = skeleton.header.morph_targets as usize;
        if nc != skeleton.clusters.len() || targets > (1 << 16) || plan_b.len() != 24 + nc * 4 + targets * 4 || dir_b.len() / entry_bytes != np {
            return Err("paged cluster mesh tables disagree".into());
        }
        let cluster_page: Vec<u32> = bytemuck::pod_collect_to_vec(&plan_b[24..24 + nc * 4]);
        let mut plan = PagePlan::from_assignment(&skeleton, cluster_page)?;
        if plan.pages.len() != np {
            return Err("paged cluster mesh page count differs".into());
        }
        plan.morph_reach = bytemuck::pod_collect_to_vec(&plan_b[24 + nc * 4..]);
        if !plan.morph_reach.iter().all(|r| r.is_finite() && *r >= 0.0) {
            return Err("paged cluster mesh blendshape reach is not finite".into());
        }
        let dir: Vec<DirEntry> = if format == 1 {
            bytemuck::pod_collect_to_vec::<u8, DirEntryV1>(dir_b).into_iter().map(DirEntry::from).collect()
        } else {
            bytemuck::pod_collect_to_vec(dir_b)
        };
        let blob_start = HEADER_BYTES as u64 + meta_len;
        for (i, (e, info)) in dir.iter().zip(&plan.pages).enumerate() {
            let in_blobs = |off: u64, len: u64| off >= blob_start && off.checked_add(len).is_some_and(|end| end <= whole);
            let (hot_z, cold_z) = (e.codec & CODEC_HOT_ZSTD != 0, e.codec & CODEC_COLD_ZSTD != 0);
            // Raw sizes follow the plan; a stored blob is its raw size, or (zstd) any size up to the raw size.
            let sizes_ok = e.hot_raw as u64 == info.bytes() as u64
                && e.cold_raw as u64 == info.vertices as u64 * 4
                && e.codec & !(CODEC_HOT_ZSTD | CODEC_COLD_ZSTD) == 0
                && if hot_z { e.hot_len <= e.hot_raw as u64 } else { e.hot_len == e.hot_raw as u64 }
                && if cold_z { e.cold_len <= e.cold_raw as u64 } else { e.cold_len == e.cold_raw as u64 };
            if !sizes_ok || !in_blobs(e.hot_off, e.hot_len) || !in_blobs(e.cold_off, e.cold_len) {
                return Err(format!("page {i}: directory entry is out of range"));
            }
        }
        Ok(Self { skeleton, plan, dir, total_vertices, total_tri_bytes, backing })
    }

    pub fn read_cold(&self, page: u32) -> Result<Vec<u32>, String> {
        let e = self.dir.get(page as usize).ok_or("page out of range")?;
        let b = self.backing.read(e.cold_off, e.cold_len)?;
        if sum128(&b) != e.cold_sum {
            return Err(format!("page {page}: source ids checksum differs"));
        }
        let b = unpack_blob(b, e.codec & CODEC_COLD_ZSTD != 0, e.cold_raw).map_err(|w| format!("page {page}: source ids: {w}"))?;
        Ok(bytemuck::pod_collect_to_vec(&b))
    }

    /// Read every page back into a full DAG (validated by the caller with [`ClusterDag::validate`]).
    pub fn assemble(&self) -> Result<ClusterDag, String> {
        let nv = usize::try_from(self.total_vertices).map_err(|_| "vertex count")?;
        let nt = usize::try_from(self.total_tri_bytes).map_err(|_| "triangle bytes")?;
        let sk = &self.skeleton;
        let want_v: u64 = sk.clusters.iter().map(|c| u64::from(c.vertex_count)).sum();
        let want_t: u64 = sk.clusters.iter().map(|c| u64::from(cluster_payload(c).1)).sum();
        if want_v != self.total_vertices || want_t != self.total_tri_bytes || nv > MAX_ASSEMBLED || nt > MAX_ASSEMBLED * 4 {
            return Err("paged cluster mesh totals disagree with its clusters".into());
        }
        let skinned = is_skinned(&sk.header);
        let mut dag = sk.clone();
        dag.vertices = vec![ClusterVertex::default(); nv];
        dag.triangles = vec![0u8; nt];
        dag.vertex_source = vec![0u32; nv];
        // Skin streams, joint ids and blendshape entries are laid out by the clusters' own offsets;
        // the totals follow from them and the fill checks below prove they tile the arrays.
        let (mut nb, mut nm) = (0usize, 0usize);
        if skinned {
            for cl in &sk.clusters {
                nb = nb.max(cl.bone_offset as usize + cl.bone_count as usize);
                nm = nm.max(cl.morph_offset as usize + cl.morph_count as usize);
            }
            if nb > MAX_ASSEMBLED || nm > MAX_ASSEMBLED {
                return Err("paged cluster mesh skin arrays are absurd".into());
            }
            if sk.header.flags & FLAG_SKINNED != 0 {
                dag.vertex_skin = vec![ClusterVertexSkin::default(); nv];
            }
            dag.bone_ids = vec![0u16; nb];
            dag.morph_entries = vec![MorphEntry::default(); nm];
        }
        let mut filled_v = vec![false; nv];
        let mut filled_t = vec![false; nt / 4];
        let mut filled_b = vec![false; nb];
        let mut filled_m = vec![false; nm];
        for (p, info) in self.plan.pages.iter().enumerate() {
            let payload = self.read_page(p as u32)?;
            let cold = self.read_cold(p as u32)?;
            let spans = page_spans(sk, info);
            for (&c, sp) in info.clusters.iter().zip(&spans.spans) {
                let cl = &sk.clusters[c as usize];
                let (cv, ct) = (cl.vertex_count as usize, cluster_payload(cl).1 as usize);
                let (vo, to) = (cl.vertex_offset as usize, cl.triangle_offset as usize);
                if vo + cv > nv || to + ct > nt || to % 4 != 0 || filled_v[vo..vo + cv].iter().any(|&f| f) || filled_t[to / 4..(to + ct) / 4].iter().any(|&f| f) {
                    return Err(format!("cluster {c}: its data overlaps another cluster's or lies outside the arrays"));
                }
                filled_v[vo..vo + cv].fill(true);
                filled_t[to / 4..(to + ct) / 4].fill(true);
                let v = sp.vertex as usize;
                dag.vertices[vo..vo + cv].copy_from_slice(&payload.vertices[v..v + cv]);
                dag.vertex_source[vo..vo + cv].copy_from_slice(&cold[v..v + cv]);
                let t = spans.skin_bytes as usize + sp.tri as usize;
                dag.triangles[to..to + ct].copy_from_slice(&payload.triangles[t..t + ct]);
                if skinned {
                    if !dag.vertex_skin.is_empty() {
                        let src: &[ClusterVertexSkin] = bytemuck::cast_slice(&payload.triangles[v * SKIN_BYTES..(v + cv) * SKIN_BYTES]);
                        dag.vertex_skin[vo..vo + cv].copy_from_slice(src);
                    }
                    let aux = block_aux(&payload, cl, sp).map_err(|e| format!("cluster {c}: {e}"))?;
                    let (bo, bc) = (cl.bone_offset as usize, cl.bone_count as usize);
                    let (mo, mc) = (cl.morph_offset as usize, cl.morph_count as usize);
                    if filled_b[bo..bo + bc].iter().any(|&f| f) || filled_m[mo..mo + mc].iter().any(|&f| f) {
                        return Err(format!("cluster {c}: its bone list or blendshape entries overlap another cluster's"));
                    }
                    filled_b[bo..bo + bc].fill(true);
                    filled_m[mo..mo + mc].fill(true);
                    dag.bone_ids[bo..bo + bc].copy_from_slice(&aux.bones);
                    // Pages are byte arrays: copy the entries out rather than casting at an unknown alignment.
                    for (k, e) in aux.morphs.as_chunks::<{ size_of::<MorphEntry>() }>().0.iter().enumerate() {
                        dag.morph_entries[mo + k] = bytemuck::pod_read_unaligned(e);
                    }
                }
            }
        }
        if !filled_v.iter().all(|&f| f) || !filled_t.iter().all(|&f| f) || !filled_b.iter().all(|&f| f) || !filled_m.iter().all(|&f| f) {
            return Err("clusters do not cover the vertex, triangle, bone and blendshape arrays".into());
        }
        Ok(dag)
    }
}

impl PageSource for PagedFile {
    fn read_page(&self, page: u32) -> Result<PagePayload, String> {
        let e = self.dir.get(page as usize).ok_or("page out of range")?;
        let b = self.backing.read(e.hot_off, e.hot_len)?;
        if sum128(&b) != e.hot_sum {
            return Err(format!("page {page}: checksum differs"));
        }
        let b = unpack_blob(b, e.codec & CODEC_HOT_ZSTD != 0, e.hot_raw).map_err(|w| format!("page {page}: {w}"))?;
        let payload = PagePayload::from_bytes(&self.plan.pages[page as usize], &b)?;
        validate_payload(&self.skeleton, &self.plan, page, &payload)?;
        Ok(payload)
    }
}

/// Most vertices a paged mesh may reassemble to (a bound on allocation from untrusted sizes).
const MAX_ASSEMBLED: usize = super::cluster_dag::MAX_CLUSTERS * super::cluster_dag::CLUSTER_MAX_VERTS;
/// Largest skeleton + plan + directory.
const MAX_META_BYTES: u64 = 1 << 30;

/// Parse a paged mesh into a full, structurally checked DAG plus its plan.
pub fn decode_paged(bytes: &[u8]) -> Result<(ClusterDag, PagePlan), String> {
    let file = PagedFile::from_bytes(bytes.into())?;
    let dag = file.assemble()?;
    Ok((dag, file.plan))
}

/// Whether `bytes` (a geometry cache slot) hold a paged mesh.
pub fn is_paged(bytes: &[u8]) -> bool {
    bytes.starts_with(PAGE_MAGIC)
}

#[cfg(test)]
mod tests;
