//! Cluster DAGs for the GPU cluster path (GEO-4): once a model is placed and
//! cluster rendering is on, its geometry (`load_model_with_geometry`: the
//! cooked cache, built on a miss) is loaded on a worker thread, and the DAGs of
//! its plain static meshes are installed into the renderer's cluster store.
//! Nothing here runs when the setting is off.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, TryRecvError};

use zestengine_assets::geo_cache::BuildProgress;
use zestengine_render::{ClusterCoverage, ClusterRefusal};

use super::*;

type Geometry = Option<zestengine_assets::geo_cache::ModelGeometry>;

/// DAGs staged on the worker into host-visible buffers (per mesh, `None` = not staged).
#[cfg(feature = "client")]
type Staged = Vec<Option<zestengine_render::StagedClusterDag>>;
#[cfg(not(feature = "client"))]
type Staged = ();

/// What the worker hands back: the geometry and, when the device has a transfer
/// queue, its DAGs already converted and staged (nothing left for the main
/// thread but reserving pool ranges).
struct Loaded {
    geo: Geometry,
    staged: Staged,
    /// Per mesh: the page source of a streamed DAG (GEO-6) and, without a
    /// transfer queue, its upload (the worker could not stage it).
    streams: Vec<Option<StreamPrep>>,
}

/// A mesh the worker prepared for streaming.
struct StreamPrep {
    #[cfg(feature = "client")]
    source: std::sync::Arc<dyn zestengine_assets::cluster_pages::PageSource>,
    #[cfg(feature = "client")]
    upload: Option<zestengine_render::ClusterMeshUpload>,
    /// Pages and payload MB, for the log.
    pages: usize,
    mb: f32,
}

#[derive(Default)]
pub(crate) struct ClusterRuntime {
    /// Models already installed or given up on.
    done: HashSet<String>,
    queue: VecDeque<String>,
    in_flight: Option<(String, Receiver<Loaded>, Arc<BuildProgress>)>,
    /// Per installed model: which meshes draw from clusters and why the rest do not.
    coverage: HashMap<String, ClusterCoverage>,
    /// `coverage` changed since the renderer was told.
    coverage_dirty: bool,
    /// (object count, template count) at the last scan for new models.
    scan: (usize, usize),
    /// The big meshes of models installed while cluster rendering was on, so
    /// the worker need not decode the model a second time.
    sources: HashMap<String, zestengine_assets::geo_cache::GeoSource>,
    /// Serves the page orders of streamed meshes (GEO-6), created with the first one.
    #[cfg(feature = "client")]
    streamer: Option<zestengine_assets::cluster_stream::ClusterStreamer>,
    /// Scene mesh indices whose DAG is installed in the renderer (GEO-4): these
    /// meshes draw from clusters, whose own LOD cut replaces the discrete
    /// auto-LOD swap. Only meaningful while `active`.
    meshes: HashSet<usize>,
    /// The cluster path was on at the last update.
    active: bool,
}

impl ClusterRuntime {
    /// Whether scene mesh `mesh` draws from clusters, so the discrete auto-LOD
    /// swap must leave it alone (two LOD systems on one mesh would fight: the
    /// swap would hand the renderer a level mesh that has no DAG).
    /// The model whose DAGs are being built or loaded now and its progress handle
    /// (the editor lists it as a task and can cancel it). Never blocks.
    pub(crate) fn build_progress(&self) -> Option<(&str, &Arc<BuildProgress>)> {
        self.in_flight.as_ref().map(|(path, _, progress)| (path.as_str(), progress))
    }

    /// Coverage of every classified model, summed.
    pub(crate) fn total_coverage(&self) -> ClusterCoverage {
        let mut total = ClusterCoverage::default();
        for c in self.coverage.values() {
            total.merge(c);
        }
        total
    }

    pub(crate) fn covers_mesh(&self, mesh: usize) -> bool {
        self.active && self.meshes.contains(&mesh)
    }

    /// Keep what the DAG build reads of a model that was just installed.
    pub(crate) fn keep_source(&mut self, path: &str, asset: &zestengine_assets::Scene) {
        if !self.done.contains(path) && !self.sources.contains_key(path) {
            let source = zestengine_assets::geo_cache::GeoSource::of(asset);
            self.sources.insert(path.to_owned(), source);
        }
    }
}

impl LoadedScene {
    /// The model whose cluster DAGs are being built or loaded (name, handle), for the task list.
    pub fn cluster_build_progress(&self) -> Option<(String, Arc<BuildProgress>)> {
        self.cluster_geo.build_progress().map(|(n, p)| (n.to_owned(), p.clone()))
    }

    /// Install finished DAGs and start loading the next model's. Cheap when the
    /// cluster path is off or nothing is pending; call once per frame.
    pub fn update_cluster_dags(&mut self, renderer: &mut Renderer, project_root: &Path) {
        #[cfg(feature = "client")]
        if let Some(s) = self.cluster_geo.streamer.as_mut() {
            s.pump(renderer);
        }
        self.cluster_geo.active = renderer.cluster_rendering_enabled();
        if !self.cluster_geo.active {
            return;
        }
        if let Some((path, rx, progress)) = self.cluster_geo.in_flight.take() {
            match rx.try_recv() {
                Ok(loaded) => self.install_cluster_dags(renderer, &path, loaded, progress.is_cancelled()),
                Err(TryRecvError::Empty) => self.cluster_geo.in_flight = Some((path, rx, progress)),
                Err(TryRecvError::Disconnected) => {
                    self.cluster_geo.done.insert(path);
                }
            }
        }
        self.share_skinned_dags(renderer);
        if std::mem::take(&mut self.cluster_geo.coverage_dirty) {
            renderer.set_cluster_coverage(self.cluster_geo.total_coverage());
        }
        let key = (self.objects.len(), self.model_templates.len());
        if key != self.cluster_geo.scan {
            self.cluster_geo.scan = key;
            for o in &self.objects {
                let ObjectSource::Model { path, .. } = &o.source else { continue };
                if o.render.is_none()
                    || self.cluster_geo.done.contains(path)
                    || self.cluster_geo.queue.contains(path)
                    || self.cluster_geo.in_flight.as_ref().is_some_and(|f| f.0 == *path)
                    || !self.model_templates.contains_key(path)
                {
                    continue;
                }
                self.cluster_geo.queue.push_back(path.clone());
            }
        }
        if self.cluster_geo.in_flight.is_none()
            && let Some(path) = self.cluster_geo.queue.pop_front()
        {
            let full = project_root.join(&path);
            let kept = self.cluster_geo.sources.remove(&path);
            #[cfg(feature = "client")]
            let uploader = renderer.uploader();
            #[cfg(feature = "client")]
            let stream_on = renderer.cluster_stream_enabled();
            let (tx, rx) = std::sync::mpsc::channel();
            let progress = BuildProgress::new();
            let worker_progress = progress.clone();
            let spawned = std::thread::Builder::new().name("cluster-dag".into()).spawn(move || {
                // With streaming on, a stored paged mesh stays on disk (its skeleton only is
                // decoded: peak memory is the clusters, not the whole payload).
                #[cfg(feature = "client")]
                let lazy = stream_on;
                #[cfg(not(feature = "client"))]
                let lazy = false;
                let loaded = match kept {
                    // The meshes the load already decoded.
                    Some(source) if lazy => Some(source.load_or_build_lazy_tracked(&full, &worker_progress)),
                    Some(source) => Some(zestengine_assets::geo_cache::LazyGeometry::eager(source.load_or_build_tracked(&full, &worker_progress))),
                    None => match zestengine_assets::load_model_with_geometry_tracked(&full, &worker_progress) {
                        Ok((_scene, geo)) => Some(zestengine_assets::geo_cache::LazyGeometry::eager(geo)),
                        Err(e) => {
                            tracing::warn!("Zestlet geometry of {}: {e:#}", full.display());
                            None
                        }
                    },
                };
                // Convert and stage the DAGs here (the conversion and the copy into
                // the staging buffer are the heavy part of an install).
                #[cfg(feature = "client")]
                let (geo, staged, streams): (Geometry, Staged, Vec<Option<StreamPrep>>) = match loaded {
                    Some(mut lazy) => {
                        let n = lazy.geo.meshes.len();
                        let key = lazy.geo.key;
                        let mut staged = Vec::with_capacity(n);
                        let mut streams = Vec::with_capacity(n);
                        for i in 0..n {
                            if lazy.geo.meshes[i].is_none() {
                                staged.push(None);
                                streams.push(None);
                                continue;
                            }
                            // A paged mesh read from its file: the root page and the skeleton are all the upload needs.
                            if stream_on && let Some(file) = lazy.paged[i].take() {
                                match prepare_stream_file(file) {
                                    Ok((up_dag, prep)) => {
                                        staged.push(uploader.as_ref().and_then(|up| up.stage_cluster_dag(&up_dag).ok()));
                                        streams.push(Some(StreamPrep { upload: if staged.last().is_some_and(Option::is_some) { None } else { Some(up_dag) }, ..prep }));
                                        continue;
                                    }
                                    Err(file) => lazy.paged[i] = Some(file),
                                }
                            }
                            // Everything else needs the full DAG (a paged slot that cannot stream is assembled here).
                            let Some(dag) = lazy.make_full(i) else {
                                staged.push(None);
                                streams.push(None);
                                continue;
                            };
                            // Big static meshes are split into pages when streaming is on (GEO-6).
                            if stream_on && let Some((up_dag, prep)) = prepare_stream(dag, &full, key) {
                                staged.push(uploader.as_ref().and_then(|up| up.stage_cluster_dag(&up_dag).ok()));
                                streams.push(Some(StreamPrep { upload: if staged.last().is_some_and(Option::is_some) { None } else { Some(up_dag) }, ..prep }));
                                continue;
                            }
                            streams.push(None);
                            let up_dag = dag.to_gpu_upload();
                            staged.push(match &uploader {
                                Some(up) => up.stage_cluster_dag(&up_dag).ok(),
                                _ => None,
                            });
                        }
                        (Some(lazy.geo), staged, streams)
                    }
                    None => (None, Vec::new(), Vec::new()),
                };
                #[cfg(not(feature = "client"))]
                let (geo, staged, streams): (Geometry, Staged, Vec<Option<StreamPrep>>) = (loaded.map(|l| l.geo), (), Vec::new());
                let _ = tx.send(Loaded { geo, staged, streams });
            });
            match spawned {
                Ok(_) => self.cluster_geo.in_flight = Some((path, rx, progress)),
                Err(e) => {
                    tracing::warn!("Zestlet DAG thread: {e}");
                    self.cluster_geo.done.insert(path);
                }
            }
        }
    }

    /// GPU-skinned copies of a model (one per animated instance) draw from the
    /// DAG installed for the model's template mesh; each keeps its own pose
    /// (`Renderer::alias_cluster_dag`). Retried every frame until the template's
    /// DAG has been installed; a lookup per slot, nothing when no slot needs it.
    fn share_skinned_dags(&mut self, renderer: &mut Renderer) {
        for (&id, inst) in &self.anim_instances {
            if !inst.gpu {
                continue;
            }
            for (k, &(m, handle)) in inst.slots.iter().enumerate() {
                let Some(&template) = self.mesh_handles.get(m) else { continue };
                if template == MeshHandle::NONE || handle == MeshHandle::NONE {
                    continue;
                }
                let mut alias_failed = false;
                if renderer.has_cluster_dag(template) && !renderer.has_cluster_dag(handle) {
                    // A refusal (pool state) leaves the copy on the ordinary path.
                    if renderer.alias_cluster_dag(handle, template).is_ok() {
                        let class = self.avatar_share_class(id);
                        renderer.set_cluster_share_class(handle, class);
                    } else {
                        alias_failed = true;
                    }
                }
                // The coverage view colours the draw's own mesh handle: the posed copies (and the LOD-level
                // copies in `active`) carry the template's refusal, so none of them is an unexplained grey.
                let reason = if alias_failed { Some(zestengine_render::ClusterRefusal::InstallRefused) } else { renderer.mesh_refusal(template) };
                for copy in std::iter::once(handle).chain(inst.active.get(k).copied().filter(|a| *a != MeshHandle::NONE && *a != handle)) {
                    if renderer.mesh_refusal(copy) != reason {
                        renderer.set_mesh_refusal(copy, reason);
                    }
                }
            }
        }
    }

    /// Weight of an animated object in the triangle budget's fair share: the
    /// local player's avatar outweighs other players' avatars (untrusted
    /// content); anything else is an ordinary share.
    fn avatar_share_class(&self, id: ObjectId) -> zestengine_render::cluster_budget::ShareClass {
        use zestengine_render::cluster_budget::ShareClass;
        let mut at = self.index_of(id);
        for _ in 0..=self.objects.len() {
            let Some(k) = at else { break };
            if let Some(a) = self.objects[k].components.iter().find_map(|c| c.as_any().downcast_ref::<zestengine_core::NetworkAvatar>()) {
                return if a.remote { ShareClass::Untrusted } else { ShareClass::Local };
            }
            at = self.objects[k].parent;
        }
        ShareClass::Trusted
    }

    /// Without the `client` feature there is no GPU cluster store: the DAG is dropped.
    #[cfg(not(feature = "client"))]
    fn install_cluster_dags(&mut self, _renderer: &mut Renderer, path: &str, _loaded: Loaded, _cancelled: bool) {
        self.cluster_geo.done.insert(path.to_owned());
        self.cluster_geo.sources.remove(path);
    }

    #[cfg(feature = "client")]
    fn install_cluster_dags(&mut self, renderer: &mut Renderer, path: &str, loaded: Loaded, cancelled: bool) {
        self.cluster_geo.done.insert(path.to_owned());
        self.cluster_geo.sources.remove(path);
        let Loaded { geo, mut staged, mut streams } = loaded;
        if cancelled {
            tracing::info!("Zestlets: {path}: zestlet build cancelled; the ordinary path draws it");
        }
        let (Some(geo), Some((origin, template_materials))) = (geo.filter(|_| !cancelled), self.model_templates.get(path).cloned()) else {
            for st in staged.iter_mut().filter_map(Option::take) {
                renderer.discard_staged_cluster_dag(st);
            }
            // Nothing will ever install for this model (cancelled, unreadable, or its geometry failed):
            // say so rather than leaving its meshes unexplained.
            if let Some((origin, mats)) = self.model_templates.get(path).cloned() {
                for i in 0..mats.len() {
                    let (Some(&handle), Some(info)) = (self.mesh_handles.get(origin + i), self.mesh_infos.get(origin + i)) else { break };
                    if renderer.has_cluster_dag(handle) {
                        continue;
                    }
                    let reason = if (info.triangles as usize) < zestengine_assets::cluster_dag::GEO_MIN_TRIANGLES { ClusterRefusal::TooSmall } else { ClusterRefusal::BuildFailed };
                    renderer.set_mesh_refusal(handle, Some(reason));
                }
            }
            return;
        };
        let (mut clusters, mut refused) = (0usize, 0usize);
        // Per mesh: (the DAG describes the installed mesh, the renderer took it).
        let mut outcome = vec![(true, false); geo.meshes.len()];
        for (i, dag) in geo.meshes.iter().enumerate() {
            let Some(dag) = dag else { continue };
            let Some(&handle) = self.mesh_handles.get(origin + i) else { continue };
            // The DAG must describe the mesh that was installed (same import).
            let triangles = dag.header.source_triangles + dag.header.dropped_triangles;
            if self.mesh_infos.get(origin + i).is_none_or(|m| m.triangles != triangles) {
                tracing::warn!("Zestlet DAG of {path} mesh {i} does not match the installed mesh (triangle count); ignored");
                if let Some(st) = staged.get_mut(i).and_then(Option::take) {
                    renderer.discard_staged_cluster_dag(st);
                }
                refused += 1;
                outcome[i].0 = false;
                continue;
            }
            // Staged on the worker: reserve ranges and let the next frame record
            // the copy (no queue submit, no wait). Without a transfer queue the
            // conversion and staging happen here, still without the wait.
            let prep = streams.get_mut(i).and_then(Option::take);
            let result = match staged.get_mut(i).and_then(Option::take) {
                Some(st) => renderer.install_staged_cluster_dag(handle, st),
                None => match prep.as_ref().and_then(|p| p.upload.as_ref()) {
                    Some(up) => renderer.install_cluster_dag_deferred(handle, up),
                    None => renderer.install_cluster_dag_deferred(handle, &dag.to_gpu_upload()),
                },
            };
            match result {
                Ok(()) => {
                    outcome[i].1 = true;
                    self.cluster_geo.meshes.insert(origin + i);
                    clusters += dag.clusters.len();
                    if let Some(p) = prep {
                        let streamer = self.cluster_geo.streamer.get_or_insert_with(|| zestengine_assets::cluster_stream::ClusterStreamer::new(renderer));
                        streamer.register(handle, p.source);
                        tracing::info!("Zestlet streaming: {path} mesh {i}: {} pages, {:.1} MB streamed on demand", p.pages, p.mb);
                    }
                }
                Err(e) => {
                    refused += 1;
                    tracing::debug!("Zestlet DAG of {path} mesh {i}: {e:#}");
                }
            }
        }
        for st in staged.iter_mut().filter_map(Option::take) {
            renderer.discard_staged_cluster_dag(st);
        }
        // The refusal report: every mesh of the model, on clusters or not and why.
        let facts: Vec<_> = geo.meshes.iter().enumerate().map(|(i, dag)| {
            let mesh = origin + i;
            let facts = cluster_report::MeshFacts {
                triangles: self.mesh_infos.get(mesh).map_or(0, |m| u64::from(m.triangles)),
                has_dag: dag.is_some(),
                lightmap_uv: self.mesh_has_lightmap_uv.get(mesh).copied().unwrap_or(false),
                dag_matches: outcome[i].0,
                installed: outcome[i].1,
            };
            let material = template_materials.get(i).and_then(|&m| self.materials.get(m)).and_then(|m| cluster_report::material_refusal(&m.model));
            (facts, material)
        }).collect();
        // The Zestlet coverage viewport mode colours each ordinary mesh by its reason.
        for (i, (f, material)) in facts.iter().enumerate() {
            if let Some(&handle) = self.mesh_handles.get(origin + i) {
                renderer.set_mesh_refusal(handle, cluster_report::classify(f, *material, zestengine_assets::cluster_dag::GEO_MIN_TRIANGLES));
            }
        }
        {
            // Per-mesh detail (mesh index within the model: "idx:triangles"), grouped by reason.
            let mut by: std::collections::BTreeMap<&'static str, Vec<String>> = Default::default();
            for (i, (f, material)) in facts.iter().enumerate() {
                if let Some(r) = cluster_report::classify(f, *material, zestengine_assets::cluster_dag::GEO_MIN_TRIANGLES)
                    && r != ClusterRefusal::TooSmall {
                        by.entry(r.label()).or_default().push(format!("{i}:{}", f.triangles));
                    }
            }
            for (label, v) in by {
                tracing::info!("Zestlets: {path}: refused {label} x{} [mesh:tris] {}", v.len(), v.join(" "));
            }
        }
        let coverage = cluster_report::coverage_of(facts, zestengine_assets::cluster_dag::GEO_MIN_TRIANGLES);
        tracing::info!("Zestlets: {path}: {} ({clusters} zestlets installed, {refused} DAG(s) refused)", coverage.summary());
        if let Some(w) = coverage.warning() {
            tracing::warn!("Zestlets: {path}: {w}");
        }
        self.cluster_geo.coverage.insert(path.to_owned(), coverage);
        self.cluster_geo.coverage_dirty = true;
    }
}

/// Prepare the streamed upload of a paged mesh that is still on disk: its root page
/// and skeleton make the upload, the file itself is the page source. Gives the file
/// back when it is not worth streaming or its root page does not read (the caller
/// assembles it instead).
#[cfg(feature = "client")]
#[allow(clippy::result_large_err)] // the Err is the report struct, not an error
fn prepare_stream_file(file: zestengine_assets::cluster_pages::PagedFile) -> Result<(zestengine_render::ClusterMeshUpload, StreamPrep), zestengine_assets::cluster_pages::PagedFile> {
    use zestengine_assets::cluster_pages::PageSource;
    if file.plan.is_single() {
        return Err(file);
    }
    let Ok(root) = file.read_page(0) else { return Err(file) };
    let up = file.plan.to_gpu_upload(&file.skeleton, &root);
    let st = file.plan.stats();
    Ok((up, StreamPrep { source: std::sync::Arc::new(file), upload: None, pages: st.pages, mb: st.streamed_bytes as f32 / 1048576.0 }))
}

/// Split a DAG into pages and prepare the streamed upload and its page source:
/// the page file next to the model when its geometry cache holds one for this
/// mesh (pages are then read from disk), else the DAG in memory. `None` when
/// the mesh is not worth streaming (small, skinned, blendshapes).
#[cfg(feature = "client")]
fn prepare_stream(dag: &zestengine_assets::cluster_dag::ClusterDag, model: &Path, key: u128) -> Option<(zestengine_render::ClusterMeshUpload, StreamPrep)> {
    use std::sync::Arc;
    use zestengine_assets::cluster_pages::{MemPageSource, PageOptions, PagePlan, PageSource, skeleton_of};
    let plan = PagePlan::split(dag, &PageOptions::default());
    if plan.is_single() {
        return None;
    }
    let skeleton = skeleton_of(dag);
    let from_file = zestengine_assets::geo_cache::open_paged(model, key)
        .and_then(|files| files.into_iter().flatten().find(|f| f.plan == plan && f.skeleton == skeleton));
    let (source, root): (Arc<dyn PageSource>, _) = match from_file {
        Some(f) => {
            let root = f.read_page(0).ok()?;
            (Arc::new(f), root)
        }
        None => {
            let mem = MemPageSource { dag: Arc::new(dag.clone()), plan: Arc::new(plan.clone()) };
            let root = mem.read_page(0).ok()?;
            (Arc::new(mem), root)
        }
    };
    let up = plan.to_gpu_upload(&skeleton, &root);
    let st = plan.stats();
    Some((up, StreamPrep { source, upload: None, pages: st.pages, mb: st.streamed_bytes as f32 / 1048576.0 }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clustered_meshes_skip_the_discrete_lod_swap_only_while_the_path_is_on() {
        let mut rt = ClusterRuntime::default();
        rt.meshes.insert(4);
        assert!(!rt.covers_mesh(4), "inactive path: the auto-LOD swap keeps working");
        rt.active = true;
        assert!(rt.covers_mesh(4));
        assert!(!rt.covers_mesh(5), "a mesh without a DAG keeps the discrete swap");
        rt.active = false;
        assert!(!rt.covers_mesh(4), "turning the path off returns the mesh to the swap");
    }

    #[cfg(feature = "client")]
    #[test]
    fn a_paged_file_prepares_its_streamed_upload_without_the_payload() {
        use zestengine_assets::cluster_dag::{DagOptions, build_cluster_dag};
        use zestengine_assets::cluster_pages::{PageOptions, PagePlan, PagedFile, encode_paged};
        // A 90 x 90 grid sphere-ish patch (the cluster builder needs a real surface).
        let n = 90u32;
        let mut vertices = Vec::new();
        for i in 0..=n {
            for j in 0..=n {
                let (u, v) = (std::f32::consts::PI * i as f32 / n as f32, std::f32::consts::TAU * j as f32 / n as f32);
                let p = [u.sin() * v.cos(), u.cos(), u.sin() * v.sin()];
                vertices.push(zestengine_render::Vertex { position: p, normal: p, ..Default::default() });
            }
        }
        let mut indices = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let a = i * (n + 1) + j;
                indices.extend([a, a + n + 1, a + 1, a + 1, a + n + 1, a + n + 2]);
            }
        }
        let mesh = zestengine_render::MeshData { vertices, indices, has_lightmap_uv: false };
        let dag = build_cluster_dag(&mesh, &[], &DagOptions { threads: 2, ..Default::default() }).expect("DAG");
        // Whole (one page): not worth streaming, the file comes back for the caller to assemble.
        let whole = PagePlan::split(&dag, &PageOptions::default());
        assert!(whole.is_single());
        let file = PagedFile::from_bytes(encode_paged(&dag, &whole).into()).unwrap();
        assert!(prepare_stream_file(file).is_err());
        // Split small: the upload carries the skeleton and the root page only.
        let plan = PagePlan::split(&dag, &PageOptions { page_bytes: 2048, resident_bytes: 4096, min_stream_bytes: 1 });
        if plan.is_single() {
            return;
        }
        let file = PagedFile::from_bytes(encode_paged(&dag, &plan).into()).unwrap();
        let Ok((up, prep)) = prepare_stream_file(file) else { panic!("a split file did not prepare") };
        assert_eq!(prep.pages, plan.pages.len());
        assert!(up.vertices.len() < dag.vertices.len(), "the upload holds the root page, not every vertex");
    }
}
