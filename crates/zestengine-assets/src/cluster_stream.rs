//! The geometry streamer (GEO-6): serves the page orders of the renderer from
//! a worker thread. The renderer decides what to load and evict
//! (`Renderer::cluster_stream_orders`, budget and hysteresis in
//! `zestengine_render::cluster_stream`); this reads each ordered page from its
//! [`PageSource`] (a geometry cache file or memory), validates it, copies it
//! into a host-visible staging buffer, and hands it back
//! (`Renderer::cluster_stream_deliver`); the next frame records the copy into
//! its own command buffer. Nothing here waits for the GPU and nothing idles the
//! device.
//!
//! One worker thread reads in order: orders arrive parents first, and the
//! renderer holds a delivered page until its parents are installed anyway.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;

use zestengine_render::{MeshHandle, PageLoad, PageStager, Renderer, StagedPage};

use crate::cluster_pages::PageSource;

struct Job {
    load: PageLoad,
    source: Arc<dyn PageSource>,
}

enum Done {
    Page { load: PageLoad, staged: StagedPage, bytes: u64, secs: f64 },
    Failed { load: PageLoad, why: String },
}

/// What the streamer did.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StreamerStats {
    pub submitted: u64,
    pub delivered: u64,
    pub failed: u64,
    pub bytes_read: u64,
    /// Seconds the worker spent reading, validating and staging.
    pub read_secs: f64,
    pub in_flight: u32,
    /// Pages orders named a mesh nobody registered for.
    pub unknown: u64,
}

pub struct ClusterStreamer {
    jobs: Option<Sender<Job>>,
    done: Receiver<Done>,
    thread: Option<JoinHandle<()>>,
    sources: Vec<(MeshHandle, Arc<dyn PageSource>, u64)>,
    pumps: u64,
    stats: StreamerStats,
}

impl ClusterStreamer {
    pub fn new(renderer: &Renderer) -> Self {
        let stager: PageStager = renderer.cluster_page_stager();
        let (jobs, rx) = channel::<Job>();
        let (done_tx, done) = channel::<Done>();
        let thread = std::thread::Builder::new()
            .name("cluster-stream".into())
            .spawn(move || {
                while let Ok(Job { load, source }) = rx.recv() {
                    let t = std::time::Instant::now();
                    let result = source.read_page(load.page).and_then(|p| {
                        if p.vertices.len() as u32 != load.vertices || p.triangles.len() as u32 != load.tri_bytes {
                            return Err("page payload differs from the order".to_string());
                        }
                        let bytes = p.to_bytes();
                        stager.stage(&bytes, load.vertices, load.tri_bytes).map(|s| (s, bytes.len() as u64)).map_err(|e| format!("{e:#}"))
                    });
                    let msg = match result {
                        Ok((staged, bytes)) => Done::Page { load, staged, bytes, secs: t.elapsed().as_secs_f64() },
                        Err(why) => Done::Failed { load, why },
                    };
                    if done_tx.send(msg).is_err() {
                        break;
                    }
                }
            })
            .ok();
        Self { jobs: Some(jobs), done, thread, sources: Vec::new(), pumps: 0, stats: StreamerStats::default() }
    }

    /// Serve the pages of `mesh` from `source`.
    pub fn register(&mut self, mesh: MeshHandle, source: Arc<dyn PageSource>) {
        self.sources.retain(|s| s.0 != mesh);
        self.sources.push((mesh, source, self.pumps));
    }

    pub fn unregister(&mut self, mesh: MeshHandle) {
        self.sources.retain(|s| s.0 != mesh);
    }

    pub fn registered(&self) -> usize {
        self.sources.len()
    }

    pub fn stats(&self) -> StreamerStats {
        self.stats
    }

    /// Once per frame: queue the renderer's new orders, hand finished pages back.
    pub fn pump(&mut self, renderer: &mut Renderer) {
        self.pumps += 1;
        for load in renderer.cluster_stream_orders() {
            let Some((_, source, _)) = self.sources.iter().find(|s| s.0 == load.mesh) else {
                self.stats.unknown += 1;
                renderer.cluster_stream_fail(load.mesh, load.page);
                continue;
            };
            let sent = self.jobs.as_ref().is_some_and(|j| j.send(Job { load, source: source.clone() }).is_ok());
            if sent {
                self.stats.submitted += 1;
            } else {
                renderer.cluster_stream_fail(load.mesh, load.page);
            }
        }
        while let Ok(d) = self.done.try_recv() {
            match d {
                Done::Page { load, staged, bytes, secs } => {
                    self.stats.delivered += 1;
                    self.stats.bytes_read += bytes;
                    self.stats.read_secs += secs;
                    renderer.cluster_stream_deliver(load.mesh, load.page, staged);
                }
                Done::Failed { load, why } => {
                    self.stats.failed += 1;
                    tracing::warn!("cluster streaming: page {} of mesh {} could not be read: {why}", load.page, load.mesh.index());
                    renderer.cluster_stream_fail(load.mesh, load.page);
                }
            }
        }
        self.stats.in_flight = (self.stats.submitted - self.stats.delivered - self.stats.failed) as u32;
        // Forget meshes the renderer no longer draws from clusters (after the install had time to land).
        if self.pumps.is_multiple_of(300) {
            let pumps = self.pumps;
            self.sources.retain(|(m, _, at)| pumps - *at < 600 || renderer.has_cluster_dag(*m));
        }
    }
}

impl ClusterStreamer {
    /// Stop the worker and release the pages it staged but nobody installed.
    pub fn shutdown(&mut self, renderer: &mut Renderer) {
        self.jobs = None;
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        while let Ok(d) = self.done.try_recv() {
            if let Done::Page { staged, .. } = d {
                // No mesh owns it: the renderer frees the staging buffer.
                renderer.cluster_stream_deliver(MeshHandle::NONE, 0, staged);
            }
        }
    }
}

impl Drop for ClusterStreamer {
    fn drop(&mut self) {
        self.jobs = None;
        // Pages still in flight hold staging buffers; they are freed with the allocator.
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}
