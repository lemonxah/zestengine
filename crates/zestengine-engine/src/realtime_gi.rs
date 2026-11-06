//! Realtime GI scene inputs and optional world-probe cache. Flux and FluxRT
//! gather visible lighting every frame in the renderer; this driver maintains
//! their distance field, emitters and asynchronous world-probe fallback.
//! Shared by the editor and runtime. No offline baker runs in this frame loop.
//!
//! Scene input rebuilds and world-probe updates settle when inputs stop
//! changing. Screen gathering continues to track lights and moving geometry.

use std::hash::{Hash, Hasher};

use zestengine_render::Renderer;

use crate::scene::{BakeGather, LoadedScene};

/// How many blended re-traces to run after an input change (temporal settle).
/// Paired with the low default temporal blend (~0.12) this gives a gentle
/// ~2-3s ease-in to the converged GI, then idles.
const SETTLE_UPDATES: u32 = 96;

/// Probe-grid layout (`world_to_local`, size, counts, sh_base) for an upload.
type VolUpload = (glam::Mat4, [f32; 3], [u32; 3], u32);

/// Hash of the FluxVoxel static inputs (volumes + static-light positions/colors/
/// ranges) so the baked base is recomputed only when they actually change.
fn flux_voxel_static_hash(
    volumes: &[(glam::Vec3, glam::Vec3, [u32; 3])],
    lights: &[zestengine_render::BakeLight],
) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for (c, s, n) in volumes {
        for v in [c.x, c.y, c.z, s.x, s.y, s.z] {
            v.to_bits().hash(&mut h);
        }
        n.hash(&mut h);
    }
    for l in lights {
        // Everything that changes a light's contribution: rotating a spot or
        // retyping a light must rebuild the base too.
        (l.kind as u8).hash(&mut h);
        for v in [
            l.position.x,
            l.position.y,
            l.position.z,
            l.direction.x,
            l.direction.y,
            l.direction.z,
            l.color[0],
            l.color[1],
            l.color[2],
            l.range,
            l.spot_inner_deg,
            l.spot_outer_deg,
            l.radius,
        ] {
            v.to_bits().hash(&mut h);
        }
    }
    h.finish()
}

/// Everything a FluxVoxel static-base rebuild needs, owned so it can run on a
/// worker thread.
struct FluxVoxelInputs {
    /// Build-time baked base (accumulator, positions, volume metas) when it
    /// matches the live volume layout.
    baked: Option<BakedBase>,
    volumes: Vec<(glam::Vec3, glam::Vec3, [u32; 3])>,
    static_lights: Vec<zestengine_render::BakeLight>,
    occ: Option<crate::sw_gi::SceneOccupancy>,
    propagate: bool,
}

/// A rebuilt FluxVoxel static base (all volumes concatenated).
#[derive(Default)]
struct FluxVoxelBase {
    static_: Vec<[glam::Vec3; 4]>,
    positions: Vec<glam::Vec3>,
    meta: Vec<VolUpload>,
    dist: Vec<[f32; 4]>,
    dist2: Vec<[f32; 4]>,
}

/// Rebuild the FluxVoxel static base: seed from the bake when present, else
/// relocate + inject the static lights, propagate, blur, and compute the DDGI
/// distance moments per volume.
fn build_flux_voxel_base(inputs: FluxVoxelInputs) -> FluxVoxelBase {
    const PROP_ITERS: u32 = 3;
    const PROP_GAIN: f32 = 0.12;
    let start = std::time::Instant::now();
    let mut out = FluxVoxelBase::default();
    if let Some((acc, positions, meta)) = inputs.baked {
        // Moments for the disk-baked base too, so its voxel light occludes.
        if let Some(o) = &inputs.occ {
            let (d, d2) = crate::sw_gi::flux_distance_moments(&positions, o);
            out.dist = d;
            out.dist2 = d2;
        }
        out.static_ = acc;
        out.positions = positions;
        out.meta = meta;
    } else {
        let mut base = 0u32;
        for (center, size, counts) in &inputs.volumes {
            let mut positions =
                crate::sw_gi::flux_volume_positions(*center, *size, *counts);
            // Probe relocation: nudge probes out of solid geometry so they
            // inject from open space. We do NOT zero trapped probes — that
            // punched dark holes into the field (interior probes DO carry
            // light, and the DDGI moments already prevent leaks at sample
            // time), which showed as darker blobs inside the couch.
            if let Some(o) = &inputs.occ {
                let _ = crate::sw_gi::relocate_probes(&mut positions, o);
            }
            let mut acc = vec![[glam::Vec3::ZERO; 4]; positions.len()];
            crate::sw_gi::flux_inject(&mut acc, &positions, &inputs.static_lights);
            // Propagate the STATIC base now (cached) instead of per frame:
            // static lights + emissive (the dominant lighting) carry the
            // bounce; dynamic movers stay direct-only for speed.
            if inputs.propagate {
                crate::sw_gi::flux_propagate(&mut acc, *counts, PROP_ITERS, PROP_GAIN);
            }
            // Smooth the cached base so a static-only scene (no movers, the
            // per-frame early-out) is also a soft gradient, not blobby.
            crate::sw_gi::blur_acc(&mut acc, *counts, 2);
            // DDGI distance moments for this volume's probes (cached).
            let (mut d, mut d2) = match &inputs.occ {
                Some(o) => crate::sw_gi::flux_distance_moments(&positions, o),
                None => (
                    vec![[0.0f32; 4]; positions.len()],
                    vec![[0.0f32; 4]; positions.len()],
                ),
            };
            // Smooth the moments across the grid too — otherwise the per-probe
            // distance variation makes the Chebyshev draw the grid as fixed
            // dots on surfaces. Softens the shadow edge slightly; removes dots.
            crate::sw_gi::blur_moments(&mut d, *counts, 2);
            crate::sw_gi::blur_moments(&mut d2, *counts, 2);
            let n = positions.len() as u32;
            out.static_.extend(acc);
            out.positions.extend(positions);
            out.dist.append(&mut d);
            out.dist2.append(&mut d2);
            out.meta.push((
                glam::Mat4::from_translation(-*center),
                size.to_array(),
                *counts,
                base,
            ));
            base += n;
        }
    }
    tracing::debug!(
        target: "frame_stats",
        "FluxVoxel static base rebuilt: {} probes in {:.2} ms",
        out.positions.len(),
        start.elapsed().as_secs_f64() * 1000.0
    );
    out
}

/// A build-time baked base: accumulator, probe positions, volume metas.
type BakedBase = (Vec<[glam::Vec3; 4]>, Vec<glam::Vec3>, Vec<VolUpload>);
/// A CPU march in flight: its thread, volume layout, counts and cascade.
type MarchJob = (std::thread::JoinHandle<Vec<zestengine_render::ProbeSh>>, Vec<VolUpload>, [usize; 3], usize);
/// A finished march: probes, volume layout, counts and cascade.
type MarchResult = (Vec<zestengine_render::ProbeSh>, Vec<VolUpload>, [usize; 3], usize);

#[derive(Default)]
pub struct RealtimeGiState {
    /// Temporally-accumulated probe SH (blended toward each new trace).
    accum: Vec<zestengine_render::ProbeSh>,
    /// Grid dimensions the accumulation was built for (a change forces a reset).
    counts: [usize; 3],
    /// Whether our probes are currently uploaded (so we can clear on disable).
    active: bool,
    /// Seconds until the next allowed trace.
    timer: f32,
    /// Hash of the last-traced inputs; unchanged + settled = skip the trace.
    hash: u64,
    /// Remaining blended re-traces after a change; 0 = converged, skip work.
    settle: u32,
    /// Consecutive converged polls with no input change. Once past a threshold the
    /// scene is "settled" and the per-frame gather+hash (CPU) throttles to ~12 Hz —
    /// a static scene doesn't need its whole GI input rebuilt every frame just to
    /// notice nothing changed. Reset to 0 the instant a change is detected, so a
    /// moving light still tracks within ~one throttle tick.
    idle_polls: u32,
    /// Set externally (e.g. a material edit) to force the next trace even when
    /// the input hash is unchanged. Cleared the moment it's consumed.
    force: bool,
    /// True while the inputs are actively changing (a light/emitter is moving).
    /// Drives the snap-to-latest response so bounce light tracks in realtime;
    /// false lets it settle smoothly toward the converged value.
    moving: bool,
    /// Monotonic per-trace seed so software-GI samples jitter each update and
    /// temporal accumulation averages out the noise.
    seed: u32,
    /// Hash of the geometry the cached GPU GDF was built from. The GDF (3D
    /// distance/index textures) is re-uploaded only when this changes, so a static
    /// scene keeps a high-resolution GDF for free instead of rebuilding per trace.
    gdf_hash: Option<u64>,
    /// Persistent, un-blurred temporal accumulation of per-cascade traces. Each
    /// frame only one cascade is re-traced (round-robin) and blended in here;
    /// `target` is its spatially-filtered view. Cached-GI policy: never re-trace the
    /// whole field at once; amortize across frames and accumulate temporally.
    raw: Vec<zestengine_render::ProbeSh>,
    /// Which cascade to trace next (round-robin over the probe volumes).
    trace_cursor: usize,
    /// Latest finished trace (intensity applied), the per-frame ease target.
    target: Vec<zestengine_render::ProbeSh>,
    /// Volume layout to upload alongside `target` / `accum`.
    target_vols: Vec<VolUpload>,
    /// One-shot activation diagnostic guard.
    logged: bool,
    /// In-flight software march on a background thread (so moving objects in
    /// Play mode don't hitch the frame). Carries the volume layout to upload
    /// with the result.
    /// In-flight CPU march (cascade `usize`) on a background thread.
    job: Option<MarchJob>,
    /// Volume layout, counts, and cascade index for an in-flight async GPU march
    /// (collected via `gi_march_poll` next frame, so it never blocks the frame).
    gpu_pending: Option<(Vec<VolUpload>, [usize; 3], usize)>,
    /// FluxVoxel BAKED static base: SH-L1 accumulator per probe (all volumes
    /// concatenated), rebuilt only when volumes / static FluxVoxel Lights change.
    flux_voxel_static: Vec<[glam::Vec3; 4]>,
    /// World position per FluxVoxel probe (parallel to `flux_voxel_static`), so dynamic
    /// FluxVoxel Lights can be mixed into the baked base each frame.
    flux_voxel_positions: Vec<glam::Vec3>,
    /// Per-probe offset of `flux_voxel_positions` from the grid (relocation),
    /// uploaded so the shaders' visibility test uses the moved probe.
    flux_voxel_offsets: Vec<glam::Vec3>,
    /// FluxVoxel volume metas to upload with the probes (world_to_local, size,
    /// counts, base offset).
    flux_voxel_meta: Vec<VolUpload>,
    /// Hash of (volumes + static lights); the static base is re-baked only when
    /// this changes.
    flux_voxel_hash: u64,
    /// Hash of the dynamic (moving) lights last uploaded. When unchanged AND the
    /// static base didn't rebuild, the per-frame clone+inject+upload is skipped
    /// entirely — a steady scene (nothing moving) costs ~zero per frame.
    flux_voxel_dyn_hash: u64,
    /// Cached per-probe DDGI distance moments (SH-L1 of distance-to-geometry +
    /// its square), parallel to `flux_voxel_static`. Computed once from the scene
    /// occupancy when the static base rebuilds; the shader's Chebyshev uses them to
    /// occlude voxel light leak-free, free per frame. Empty / zero = occlusion off.
    flux_voxel_dist: Vec<[f32; 4]>,
    flux_voxel_dist2: Vec<[f32; 4]>,
    /// Background static-base rebuild in flight: its target hash + worker.
    flux_voxel_job: Option<(u64, std::thread::JoinHandle<FluxVoxelBase>)>,
}

impl RealtimeGiState {
    fn apply_flux_voxel_base(&mut self, base: FluxVoxelBase) {
        self.flux_voxel_static = base.static_;
        self.flux_voxel_positions = base.positions;
        self.flux_voxel_meta = base.meta;
        self.flux_voxel_dist = base.dist;
        self.flux_voxel_dist2 = base.dist2;
        self.flux_voxel_offsets.clear();
    }

    /// Force the next trace even if the input hash is unchanged. Use after an
    /// edit that affects bounce light but may not be captured by the hash
    /// (e.g. a material reassignment or a non-hashed property change).
        pub fn invalidate(&mut self) {
        self.force = true;
    }

    /// Drop the cached FluxVoxel static base so the next `update_flux_voxel` reseeds it
    /// (from a freshly completed build-time bake, or from the analytic fallback).
        pub fn invalidate_flux_voxel(&mut self) {
        self.flux_voxel_hash = 0;
    }

    /// FluxVoxel backend. Injects the scene's FluxVoxel Lights into author-placed
    /// FluxVolume voxel grids (or one auto whole-scene volume). STATIC sources are
    /// BAKED into a cached base (rebuilt only when volumes/static lights change);
    /// DYNAMIC sources mix into a clone of that base every frame (the cheap "fake
    /// GI on moving lights"). The standard shader samples the result via
    /// `sample_probes`. No ray tracing.
    fn update_flux_voxel(&mut self, renderer: &mut Renderer, scene: &mut LoadedScene) {
        // Camera position (last frame) lets the CameraClipmap grid follow the view.
        let view_pos = renderer.last_view_pos();
        let volumes = scene.flux_voxel_volumes_view(view_pos);
        if volumes.is_empty() {
            // Once (not every frame: `set_baked_probes` idles the device): fall
            // back to the bake's probes, or none. u64::MAX marks "cleared".
            if self.flux_voxel_hash != u64::MAX {
                crate::runtime::upload_probes(renderer, scene.baked.as_ref());
                self.flux_voxel_hash = u64::MAX;
            }
            return;
        }
        let lights = scene.flux_voxel_lights();
        let static_lights: Vec<zestengine_render::BakeLight> =
            lights.iter().filter(|(_, s)| *s).map(|(l, _)| *l).collect();
        let dynamic_lights: Vec<zestengine_render::BakeLight> =
            lights.iter().filter(|(_, s)| !*s).map(|(l, _)| *l).collect();
        // Expensive GI work (occupancy build, occluded inject, propagation,
        // relocation) all happens in the CACHED static-base rebuild below — NOT per
        // frame. Per frame is only the cheap dynamic-light direct inject. (Doing the
        // occupancy rebuild + whole-grid propagation every frame was the FluxVoxel
        // perf regression — it's the cheap VR backend, the per-frame cost must stay
        // tiny.) These toggles (default on) gate the cached work and fold into the
        // hash so flipping one reseeds the base.
        let occlusion_on = scene.environment.realtime_gi.voxel_ddgi_occlusion;
        let propagate = scene.environment.realtime_gi.voxel_propagation;

        // Prefer the BUILD-TIME baked static base (multi-bounce voxels persisted to
        // .lightdata): seed the static SH straight from disk so runtime does zero
        // per-voxel range/falloff work — just the additive dynamic-light pass below.
        // Falls back to the analytic direct-only injection when no bake exists.
        //
        // BUT the bake is at a FIXED resolution. If the live volume density (or the
        // auto-grid density) now differs, drop the bake and re-grid analytically so
        // the density slider actually takes effect live (otherwise the grid stays
        // locked to the bake until a re-bake). Same volume count + same per-volume
        // counts => the bake still matches => keep it.
        let baked = scene.flux_voxel_baked().filter(|(_, _, meta)| {
            meta.len() == volumes.len()
                && meta
                    .iter()
                    .zip(&volumes)
                    .all(|((_, _, baked_counts, _), (_, _, live_counts))| baked_counts == live_counts)
        });
        let h = match &baked {
            // Content-derived hash so a different bake (or a freshly reloaded scene)
            // reseeds the static base; identical baked voxels keep the cache warm.
            // `invalidate_flux_voxel` (called after a bake) zeroes the hash to be safe.
            Some((acc, _, _)) => {
                let mut hh = 0x9E37_79B9_u64 ^ acc.len() as u64;
                if let Some(f) = acc.first() {
                    hh ^= (f[0].x.to_bits() as u64) << 1 ^ f[1].y.to_bits() as u64;
                }
                if let Some(l) = acc.last() {
                    hh = hh.rotate_left(13) ^ l[0].z.to_bits() as u64;
                }
                hh | 1 // never collide with a "disabled" 0
            }
            None => flux_voxel_static_hash(&volumes, &static_lights),
        };
        // Fold the toggles in so flipping DDGI occlusion / propagation reseeds the
        // cached static base (which now bakes in the occlusion + the propagated bounce).
        let mut h = h;
        if occlusion_on {
            h = h.rotate_left(7) ^ 0xDD61;
        }
        if propagate {
            h = h.rotate_left(3) ^ 0x9F17;
        }
        // A finished background rebuild replaces the base.
        let mut rebuilt = false;
        if self.flux_voxel_job.as_ref().is_some_and(|(_, j)| j.is_finished()) {
            let (job_hash, job) = self.flux_voxel_job.take().unwrap();
            match job.join() {
                Ok(base) => {
                    self.apply_flux_voxel_base(base);
                    self.flux_voxel_hash = job_hash;
                    rebuilt = true;
                }
                Err(_) => tracing::error!("FluxVoxel static rebuild panicked"),
            }
        }
        if h != self.flux_voxel_hash && self.flux_voxel_job.is_none() {
            // Build the occupancy grid ONCE here (cached path); used for relocation +
            // the DDGI distance moments. Injection itself is always the cheap analytic
            // (non-occluded) path — the shader's Chebyshev does the occlusion from the
            // moments, so DDGI costs ~zero per frame (no per-probe DDA anywhere).
            let inputs = FluxVoxelInputs {
                baked,
                volumes: volumes.clone(),
                static_lights: static_lights.clone(),
                occ: if occlusion_on { scene.flux_occupancy() } else { None },
                propagate,
            };
            if self.flux_voxel_static.is_empty() {
                // Nothing to show yet: build now so the first frame is lit.
                self.apply_flux_voxel_base(build_flux_voxel_base(inputs));
                self.flux_voxel_hash = h;
                rebuilt = true;
            } else {
                // Clipmap steps and static-light edits rebuild off the main
                // thread; the current base keeps rendering until the new one lands.
                let job = std::thread::Builder::new()
                    .name("fluxvoxel-rebuild".into())
                    .spawn(move || build_flux_voxel_base(inputs));
                match job {
                    Ok(j) => self.flux_voxel_job = Some((h, j)),
                    Err(e) => tracing::error!("FluxVoxel rebuild thread: {e}"),
                }
            }
        }

        // PER FRAME (cheap): skip everything when nothing moved since the last
        // upload — a steady scene re-uploads zero probes. Only a static rebuild or a
        // changed dynamic-light set re-does the work.
        let dyn_hash = flux_voxel_static_hash(&[], &dynamic_lights);
        if !rebuilt && dyn_hash == self.flux_voxel_dyn_hash {
            return;
        }
        self.flux_voxel_dyn_hash = dyn_hash;
        // Mix the dynamic lights' DIRECT (analytic, NON-occluded) term into a clone
        // of the cached base. No occlusion DDA, no propagation here — those are baked
        // into the cached static base, so the per-frame cost (and stability) does NOT
        // depend on DDGI being on. Moving lights stay smooth + cheap.
        let mut acc = self.flux_voxel_static.clone();
        crate::sw_gi::flux_inject(&mut acc, &self.flux_voxel_positions, &dynamic_lights);
        // Smooth the field so it reads as a soft gradient, not blotchy per-probe
        // blobs (each emitter lights nearby probes with a sharp falloff). Per-volume
        // box blur; cheap O(probes) and keeps the soft Lumen-style look.
        for (_, _, counts, base) in &self.flux_voxel_meta {
            let n = (counts[0].max(2) * counts[1].max(2) * counts[2].max(2)) as usize;
            let b = *base as usize;
            if b + n <= acc.len() {
                // 1 iteration here — the cached static base is already pre-blurred;
                // this just smooths the per-frame dynamic-light addition.
                crate::sw_gi::blur_acc(&mut acc[b..b + n], *counts, 1);
            }
        }
        // Build probes, attaching the cached DDGI distance moments (geometry-based,
        // so independent of the dynamic lights) when present. The shader's Chebyshev
        // uses them to occlude voxel light leak-free — free per frame.
        let have_moments = self.flux_voxel_dist.len() == acc.len();
        if self.flux_voxel_offsets.len() != self.flux_voxel_positions.len() {
            // Grid positions per volume (as the shaders compute them) vs. where
            // relocation put each probe.
            self.flux_voxel_offsets.clear();
            for (w2l, size, counts, base) in &self.flux_voxel_meta {
                let center = -w2l.w_axis.truncate();
                let grid = crate::sw_gi::flux_volume_positions(center, glam::Vec3::from(*size), *counts);
                for (k, g) in grid.iter().enumerate() {
                    let p = self.flux_voxel_positions.get(*base as usize + k).copied().unwrap_or(*g);
                    self.flux_voxel_offsets.push(p - *g);
                }
            }
        }
        let probes: Vec<zestengine_render::ProbeSh> = acc
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let mut p = if have_moments {
                    crate::sw_gi::acc_to_probe_moments(s, self.flux_voxel_dist[i], self.flux_voxel_dist2[i])
                } else {
                    crate::sw_gi::acc_to_probe(s)
                };
                if let Some(o) = self.flux_voxel_offsets.get(i) {
                    p.offset = o.to_array();
                }
                p
            })
            .collect();
        // Per frame, only the SH payload changes (the volume layout is fixed
        // until a rebuild). `set_baked_probes` does a device_wait_idle + buffer
        // realloc + descriptor rewrite on EVERY call — a full GPU stall — which at
        // this per-frame cadence was the FluxVoxel bottleneck (it ran SLOWER than
        // Flux/FluxRT, which already use the cheap in-place path below, despite
        // doing less GPU work). Use the stall-free in-place SSBO rewrite; only fall
        // back to the full resize path on a rebuild or an actual probe-count change.
        if rebuilt || !renderer.update_probe_sh(&probes) {
            renderer.set_baked_probes(&probes, &self.flux_voxel_meta);
        }
    }

    pub fn update(
        &mut self,
        renderer: &mut Renderer,
        scene: &mut LoadedScene,
        dt: f32,
        vr_active: bool,
    ) {
        // The realtime GI backend the scene selected: Flux (software SDF march),
        // FluxRT (hardware ray-query), or FluxVoxel (analytic voxel volume).
        let mut gi = scene.environment.realtime_gi;
        // The mobile tier runs baked lighting only (lightmaps + probes).
        gi.enabled &= renderer.tier().realtime_gi();
        // Select the effective backend before either the screen input upload
        // or world-probe scheduling. Returning early here used to leave FluxRT
        // with no GDF on non-RT devices, so its advertised fallback never ran.
        if gi.mode == zestengine_assets::GiMode::FluxRT && !renderer.supports_baking() {
            gi.mode = zestengine_assets::GiMode::Flux;
        }
        let backend = gi.mode;
        // The GDF + emitter feed Flux samples must refresh every frame so moving
        // emitters track without lag (the field is no longer user-facing).
        gi.update_interval = 0.0;
        // Flux can run with a bake loaded: mixed lighting keeps the baked
        // static contribution and transports the live dynamic contribution.
        let on = gi.enabled;
        // Always drain a finished async GPU march (even when off, so toggling GI
        // off mid-flight can't wedge a re-enable; the in-flight march is freed).
        let gpu_fresh = if renderer.gi_marching() {
            renderer.gi_march_poll()
        } else {
            None
        };
        // Push the Flux trace params AND the reflection settings (Environment tab)
        // every frame, before any early return: the reflection mode / SSR live in
        // the same struct, and only pushing it on the Flux/FluxRT GI-on path left
        // the renderer on its default (SSR on) for FluxVoxel or GI off, whatever
        // the Reflections setting said. Cheap; a changed setting re-traces.
        renderer.set_flux_settings(zestengine_render::FluxSettings {
            samples: gi.quality.samples(),
            bounces: gi.bounces.clamp(1, 4),
            march_distance: gi.march_distance.max(0.0),
            firefly_clamp: gi.firefly_clamp.max(0.5),
            smoothing: gi.smoothing.clamp(0.0, 1.0),
            intensity: gi.intensity.max(0.0),
            emissive_boost: gi.emissive_boost.max(0.0),
            ssr_enabled: gi.reflection_mode == 1 && gi.ssr_enabled,
            ssr_intensity: gi.ssr_intensity.max(0.0),
            ssr_max_distance: gi.ssr_max_distance.max(0.0),
            ssr_roughness_cutoff: gi.ssr_roughness_cutoff.clamp(0.0, 1.0),
            reflection_mode: gi.reflection_mode.min(2) as u32,
            rt_samples: gi.rt_samples.clamp(1, 16),
            mixed_lighting: scene.baked.is_some(),
            rt_bounces: gi.rt_bounces.clamp(1, 8),
            ray_lighting_mode: gi.ray_lighting_mode,
            // The existing "Ray Query" GI mode selects the hardware trace backend
            // (screen_gi_rt); falls back to software when the device lacks it.
            rt_trace: gi.mode == zestengine_assets::GiMode::FluxRT,
        });
        // Main Flux gathers transport the actual captured material emission.
        // Legacy sphere proxies suppress real ray-hit emission and freeze video,
        // masks and pulsing at a CPU average. Keep them only for the optional
        // world-probe fallback below, not the screen/surface transport pipeline.
        renderer.gi_set_emitters(&[]);
        // FluxVoxel backend: fill the probe grid from the FluxVolume voxels and skip
        // the Flux march entirely. No GDF is built, so the screen-space gather stays
        // off and the forward shader samples THESE probes directly (desktop preview
        // and VR eye render). This runs even when a bake exists (unlike Flux/FluxRT)
        // BECAUSE the build-time baked voxels ARE its static base — `update_flux_voxel`
        // seeds from them, then adds dynamic FluxVoxel Lights live each frame.
        if backend == zestengine_assets::GiMode::FluxVoxel {
            // FluxVoxel fills the probe grid directly; the screen-space GDF trace
            // never runs, so keep its gate off.
            renderer.set_gi_enabled(false);
            self.job = None;
            self.gpu_pending = None;
            if gi.enabled {
                self.update_flux_voxel(renderer, scene);
                self.active = true;
            } else if self.active {
                // Back to the bake's probes (or none).
                crate::runtime::upload_probes(renderer, scene.baked.as_ref());
                self.flux_voxel_hash = 0;
                self.active = false;
            }
            return;
        }
        if !on {
            // GI disabled (or a bake is active): stop the screen-space trace.
            // The GDF stays cached so re-enabling is instant, but the gate below
            // is what actually halts the per-frame gather — otherwise disabling
            // GI left the camera-gated trace running (the FPS cliff on move).
            renderer.set_gi_enabled(false);
            if self.active {
                // Back to the bake's probes (or none).
                crate::runtime::upload_probes(renderer, scene.baked.as_ref());
                self.accum.clear();
                self.target.clear();
                self.counts = [0; 3];
                self.active = false;
                self.settle = 0;
                self.logged = false;
            }
            self.job = None; // detach any in-flight CPU march; its result is dropped
            self.gpu_pending = None; // drop the polled GPU result above
            return;
        }
        // GI is on: allow the screen-space gather (gated also on a built GDF).
        renderer.set_gi_enabled(true);

        // 1) Collect a finished async march, if any (CPU thread or GPU). The
        // tuple carries which cascade was traced (round-robin amortization).
        let mut fresh: Option<MarchResult> = None;
        if let Some((handle, _, _, _)) = &self.job
            && handle.is_finished()
        {
            let (handle, vols, counts, k) = self.job.take().unwrap();
            if let Ok(probes) = handle.join() {
                fresh = Some((probes, vols, counts, k));
            }
        }
        if let Some(probes) = gpu_fresh
            && let Some((vols, counts, k)) = self.gpu_pending.take()
        {
            fresh = Some((probes, vols, counts, k));
        }

        // 2) Kick off a new trace on the cadence, when inputs changed / settling.
        // Never start one while a march (CPU thread or GPU) is still in flight.
        self.timer -= dt;
        if self.timer <= 0.0 && self.job.is_none() && !renderer.gi_marching() {
            // Screen gathering runs every visible frame. The optional world
            // cache is amortized across cascades and submitted asynchronously;
            // neither backend invokes the offline lightmap/probe baker here.
            self.timer = gi.update_interval;
            if let Some(gather) = scene.gather_realtime_gi() {
                let hash = hash_inputs(&gather, &gi);
                let changed = hash != self.hash || !self.active || self.force;
                self.force = false;
                if changed {
                    self.hash = hash;
                    self.settle = SETTLE_UPDATES;
                    self.idle_polls = 0;
                } else if self.settle == 0 {
                    self.idle_polls = self.idle_polls.saturating_add(1);
                }
                // Settled (converged + idle for a while): throttle the per-frame
                // gather+hash to ~12 Hz. A change is then noticed within ~one tick
                // (≤80 ms), which is imperceptible, but a static scene stops paying
                // the full O(n) gather every frame.
                if self.idle_polls > 24 {
                    self.timer = self.timer.max(0.08);
                }
                // Track motion so the ingest below snaps to the latest trace
                // while things move, then settles smoothly once they stop.
                self.moving = changed;
                if self.settle > 0 {
                    self.settle -= 1;
                    self.seed = self.seed.wrapping_add(0x9E37_79B9);
                    let vols = vol_uploads(&gather);
                    let counts = gather.probe_volumes[0].counts;
                    // Amortize: trace only one cascade this frame (round-robin),
                    // so per-frame GI work is bounded regardless of total probe
                    // count. The ingest blends just this cascade into the
                    // persistent field and re-filters it.
                    let cascades = gather.probe_volumes.len().max(1);
                    let k = self.trace_cursor % cascades;
                    self.trace_cursor = (k + 1) % cascades;
                    let cbase = gather.probe_volumes[k].sh_base;
                    let cn = {
                        let c = gather.probe_volumes[k].counts;
                        c[0] * c[1] * c[2]
                    };
                    let cend = (cbase + cn).min(gather.probes.len());
                    let cascade_probes = gather.probes[cbase..cend].to_vec();
                    // Both screen backends share an optional, coarse software
                    // world cache. FluxRT still traces visible GI and emitter
                    // shadows against the TLAS; a fallback cache update must not
                    // synchronously rebuild the offline baker's AS/pipelines.
                    if gi.mode != zestengine_assets::GiMode::FluxVoxel {
                        let (insts, scene_size) = scene.software_gi_inputs(&gather);
                        // GDF bounds/dims over the coarsest cascade (full-scene box).
                        let coarsest = gather.probe_volumes.last().unwrap();
                        let center = -coarsest.world_to_local.w_axis.truncate();
                        let size = glam::Vec3::from(coarsest.size);
                        let (gmin, gmax) = (center - size * 0.5, center + size * 0.5);
                        // 128³ field (was 64³): finer distance + nearest-instance
                        // index → smoother sphere occlusion and emitter footprints
                        // (less GDF blockiness). Cached (rebuilt only when static
                        // geometry changes), so the larger CPU build is one-time.
                        let gres = gi.gdf_resolution.clamp(16, 256);
                        let voxel = (size.max_element() / gres as f32).max(0.03);
                        let dims = [
                            ((size.x / voxel).round() as u32 + 1).clamp(8, gres),
                            ((size.y / voxel).round() as u32 + 1).clamp(8, gres),
                            ((size.z / voxel).round() as u32 + 1).clamp(8, gres),
                        ];
                        let eps = (scene_size * 0.004).max(1e-3);
                        let max_dist = scene_size * 1.5 + 1.0;
                        // `vols`/`counts` go to whichever path produces the trace,
                        // exactly once: GPU march (fresh) or CPU thread (job).
                        let mut pending = Some((vols, counts));
                        if renderer.gi_gpu_available() {
                            // The GDF is built from STATIC geometry only, so a
                            // moving dynamic object (a player, physics body) never
                            // triggers the costly field rebuild. Emissive geometry
                            // stays out of this coarse field to avoid padded-box
                            // self occlusion. The per-mesh fields uploaded below
                            // include it and trace its actual surface captures.
                            let gdf_insts: Vec<_> = insts
                                .iter()
                                .filter(|i| i.static_geometry && i.emission.iter().all(|&v| v == 0.0))
                                .cloned()
                                .collect();
                            let gdf_insts = if gdf_insts.is_empty() {
                                insts
                                    .iter()
                                    .filter(|i| i.emission.iter().all(|&v| v == 0.0))
                                    .cloned()
                                    .collect()
                            } else {
                                gdf_insts
                            };
                            // Rebuild + re-upload the GDF only when the static
                            // geometry/materials/bounds hash changes, not every
                            // trace, so a static scene keeps a sharp cached field
                            // for free while lights/emitters/dynamics move.
                            let gh = hash_gdf_inputs(&gdf_insts, gmin, gmax, dims);
                            if self.gdf_hash != Some(gh) {
                                let gdf = crate::sw_gi::build_gdf(&gdf_insts, gmin, gmax, dims);
                                let materials: Vec<_> = gdf_insts
                                    .iter()
                                    .map(|i| zestengine_render::GpuGiMaterial {
                                        albedo: i.albedo,
                                        emission: i.emission,
                                        metallic: i.metallic,
                                        roughness: i.roughness,
                                    })
                                    .collect();
                                let sources: Vec<_>=gdf_insts.iter().map(|i|i.gi_source).collect();
                                let uploaded=renderer.gi_set_gdf_identified(
                                    &gdf.dist,
                                    &gdf.index,
                                    gdf.dims,
                                    gdf.min.to_array(),
                                    gdf.max.to_array(),
                                    &materials, &sources,
                                );
                                match uploaded {
                                    Ok(())=>self.gdf_hash=Some(gh),
                                    Err(e)=>tracing::warn!("Flux global distance field: {e:#}"),
                                }
                            }
                            // Mesh fields preserve fine geometry and include objects omitted
                            // from the static GDF. Moving instances update only transforms/BVH;
                            // renderer snapshots share immutable voxel storage across frames.
                            let detail: Vec<_> = insts.iter().map(|i| zestengine_render::GiMeshSdf { gi_source: i.gi_source,
                                world_to_local: i.inv, volume: i.sdf.clone(),
                                material: zestengine_render::GpuGiMaterial { albedo:i.albedo, emission:i.emission,
                                    metallic:i.metallic, roughness:i.roughness },
                                in_global_field: gdf_insts.iter().any(|g| std::sync::Arc::ptr_eq(&g.sdf,&i.sdf) && g.inv==i.inv),
                            }).collect();
                            if let Err(e)=renderer.gi_set_mesh_sdfs(&detail) { tracing::warn!("Flux mesh distance fields: {e:#}"); }
                            // Emissive instances → sphere area-lights for NEE, so
                            // their GI fill is sampled directly (smooth) instead of
                            // via blotchy random ray hits.
                            // Mixed lighting: static emitters are in the bake,
                            // so only moving ones light through Flux.
                            let mixed = scene.baked.is_some();
                            let live: Vec<_> = insts.iter().filter(|i| !(mixed && i.static_geometry)).cloned().collect();
                            let emitters: Vec<_> = crate::sw_gi::emitter_spheres(&live)
                                .into_iter()
                                .map(|(center, radius, emission)| zestengine_render::GpuGiEmitter {
                                    center,
                                    radius,
                                    emission: emission.map(|v| v * gi.emissive_boost.max(0.0)),
                                })
                                .collect();
                            // The optional legacy world-probe march below still
                            // accepts average sphere emitters. Main Flux/FluxRT
                            // use captured emission and never suppress real hits.
                            // Mark active so the change-detect can settle: a static
                            // scene stops re-feeding, a moving emitter (hash change)
                            // resumes it. Without this the Flux-only path (no probe
                            // ingest to set it) would re-feed every frame forever.
                            self.active = true;
                            // Flux drives the main view from scene geometry and
                            // captured materials. The legacy world-probe march is for
                            // the in-game camera / off-screen fallback, so skip it
                            // entirely when the fallback is Off. That's the "only
                            // Flux runs" path, and it kills the redundant march cost.
                            // VR renders per-eye in world space and can't use the
                            // (camera-space) Flux screen gather, so it samples the
                            // world-probe DDGI volume instead — force the march that
                            // populates it whenever a headset is active.
                            let run_probes =
                                gi.probe_fallback != zestengine_assets::ProbeFallback::Off || vr_active;
                            let march = zestengine_render::GpuGiMarch {
                                lights: &gather.lights,
                                emitters: &emitters,
                                probes: &cascade_probes,
                                samples: gi.samples,
                                bounces: gi.bounces,
                                sky: gather.sky_color,
                                eps,
                                max_dist,
                                seed: self.seed,
                            };
                            // Submit asynchronously; the result is collected by
                            // the poll in step 1 next frame, so the (potentially
                            // heavy) march never blocks the frame.
                            if run_probes && renderer.gi_march_begin(&march) {
                                let (vols, counts) = pending.take().unwrap();
                                self.gpu_pending = Some((vols, counts, k));
                            }
                        }
                        // No GPU compute (or the march failed): CPU march thread.
                        // Also gated: Off = Flux-only, no world-probe fallback.
                        let run_probes =
                            gi.probe_fallback != zestengine_assets::ProbeFallback::Off || vr_active;
                        if run_probes
                            && let Some((vols, counts)) = pending.take()
                        {
                            let lights = gather.lights.clone();
                            let probes = cascade_probes.clone();
                            let (sky, samples, seed) = (gather.sky_color, gi.samples, self.seed);
                            let bounces = gi.bounces;
                            let handle = std::thread::spawn(move || {
                                crate::sw_gi::march_probes(
                                    &insts, &lights, &probes, sky, samples, scene_size, bounces,
                                    seed,
                                )
                            });
                            self.job = Some((handle, vols, counts, k));
                        }
                    }
                }
            }
        }

        // 3) Ingest the freshly-traced cascade: temporally blend it into the
        // persistent `raw` field, then spatially filter that cascade into
        // `target`. Only one cascade is touched per frame (amortized).
        if let Some((mut probes, vols, counts, k)) = fresh {
            // Total probes across all cascades (the persistent field size).
            let total: usize = vols
                .iter()
                .map(|(_, _, c, _)| (c[0] * c[1] * c[2]) as usize)
                .sum();
            // Resize (or first result): allocate persistent buffers and re-point
            // the GPU descriptors via the full upload, then fill this cascade.
            let resized = self.raw.len() != total || self.counts != counts;
            if resized {
                self.counts = counts;
                self.raw = vec![zestengine_render::ProbeSh::default(); total];
                self.target = vec![zestengine_render::ProbeSh::default(); total];
                self.accum = vec![zestengine_render::ProbeSh::default(); total];
                renderer.set_baked_probes(&self.accum, &vols);
                self.active = true;
            }
            // Apply intensity to the fresh cascade.
            let intensity = gi.intensity.max(0.0);
            if (intensity - 1.0).abs() > 1e-3 {
                for p in &mut probes {
                    for b in 0..4 {
                        for c in 0..3 {
                            p.coeffs[b][c] *= intensity;
                        }
                    }
                }
            }
            // Locate this cascade's slice in the flat field.
            let (cnt, base) = {
                let (_, _, c, b) = &vols[k.min(vols.len().saturating_sub(1))];
                ([c[0] as usize, c[1] as usize, c[2] as usize], *b as usize)
            };
            let n = cnt[0] * cnt[1] * cnt[2];
            if base + n <= self.raw.len() && probes.len() == n {
                // Temporal blend into raw. Snap on resize / while moving; average
                // gently when static so residual variance converges smoothly.
                let alpha = if resized {
                    1.0
                } else if self.moving {
                    0.2 + 0.8 * gi.temporal_blend.clamp(0.0, 1.0)
                } else {
                    0.03
                };
                for (t, f) in self.raw[base..base + n].iter_mut().zip(&probes) {
                    for b in 0..4 {
                        for c in 0..3 {
                            t.coeffs[b][c] += (f.coeffs[b][c] - t.coeffs[b][c]) * alpha;
                        }
                        t.dist[b] += (f.dist[b] - t.dist[b]) * alpha;
                    }
                }
                // Spatially filter this cascade (from raw into target). Recomputed
                // from raw each time so repeated frames don't over-blur. Software
                // grids are coarser, so blur harder (also softens trilinear facets).
                let iters = if gi.mode == zestengine_assets::GiMode::Flux { 6 } else { 2 };
                let mut filtered = self.raw[base..base + n].to_vec();
                crate::sw_gi::blur_probe_grid(&mut filtered, cnt, iters);
                self.target[base..base + n].copy_from_slice(&filtered);
            }
            self.target_vols = vols;
            if !self.logged {
                self.logged = true;
                let n = self.target.len().max(1) as f32;
                let avg = self
                    .target
                    .iter()
                    .map(|p| {
                        0.2126 * p.coeffs[0][0] + 0.7152 * p.coeffs[0][1] + 0.0722 * p.coeffs[0][2]
                    })
                    .sum::<f32>()
                    / n;
                tracing::info!(
                    "realtime GI active ({:?}): {} probes, avg L0 luminance {:.3}",
                    gi.mode,
                    self.target.len(),
                    avg
                );
            }
        }

        // 4) Per-frame ease of the uploaded probes toward the (already EMA-
        // smoothed) target, so the result glides between trace updates instead
        // of stepping. Fixed short time-constant; variance smoothing lives in
        // the EMA above, this is purely visual continuity. Cheap in-place upload.
        if self.target.is_empty() || self.accum.len() != self.target.len() {
            return;
        }
        // Glide near-instantly while moving (track the latest trace) and gently
        // when settling (smooth out residual variance).
        let tau = if self.moving { 0.015 } else { 0.08 };
        let f = 1.0 - (-dt / tau).exp();
        let mut max_delta = 0.0f32;
        for (acc, tgt) in self.accum.iter_mut().zip(&self.target) {
            for b in 0..4 {
                for c in 0..3 {
                    let d = tgt.coeffs[b][c] - acc.coeffs[b][c];
                    acc.coeffs[b][c] += d * f;
                    max_delta = max_delta.max(d.abs());
                }
                // Ease the visibility moments too (not counted in max_delta; the
                // radiance settling already keeps the upload alive while it eases).
                acc.dist[b] += (tgt.dist[b] - acc.dist[b]) * f;
            }
        }
        // Skip the upload once fully converged + static (nothing to push).
        if max_delta > 1e-5
            && !renderer.update_probe_sh(&self.accum) {
                renderer.set_baked_probes(&self.accum, &self.target_vols);
            }
        self.active = true;
    }
}

/// Probe-volume layout tuples for `set_baked_probes`.
fn vol_uploads(gather: &BakeGather) -> Vec<VolUpload> {
    gather
        .probe_volumes
        .iter()
        .map(|v| {
            (
                v.world_to_local,
                v.size,
                [v.counts[0] as u32, v.counts[1] as u32, v.counts[2] as u32],
                v.sh_base as u32,
            )
        })
        .collect()
}

/// Hash the realtime-GI inputs (light positions/colors, object transforms, GI
/// settings) so an unchanged scene can skip the probe re-trace. f32s fold by bit
/// pattern; exact, which is fine since we only ask "did anything change".
fn hash_inputs(gather: &BakeGather, gi: &zestengine_assets::RealtimeGi) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    let f = |x: f32, h: &mut std::collections::hash_map::DefaultHasher| x.to_bits().hash(h);
    gather.instance_sources.hash(&mut h);
    for l in &gather.lights {
        (l.kind as u8).hash(&mut h);
        for v in [
            l.position.x,
            l.position.y,
            l.position.z,
            l.direction.x,
            l.direction.y,
            l.direction.z,
            l.color[0],
            l.color[1],
            l.color[2],
            l.range,
            l.spot_inner_deg,
            l.spot_outer_deg,
            l.radius,
        ] {
            f(v, &mut h);
        }
    }
    for (i, inst) in gather.instances.iter().enumerate() {
        let is_static = gather.instance_static.get(i).copied().unwrap_or(true);
        // Mesh distance fields include dynamic occluders in software too.
        // They must refresh even when static geometry already supplies a GDF;
        // hash_gdf_inputs separately keeps that static voxel field resident.
        inst.mesh.index().hash(&mut h);
        is_static.hash(&mut h);
        // Quantize to ~1mm so a resting body's solver jitter doesn't re-trace.
        for v in inst.transform.to_cols_array() {
            f((v * 1024.0).round() / 1024.0, &mut h);
        }
        // Material emission + albedo affect the bounce, so editing an emitter's
        // strength/colour (or a surface's albedo) must re-trace the GI, not just
        // moving the object. Without these the GI only updated on a transform change.
        for v in inst.emission.iter().chain(inst.albedo.iter()) {
            f(*v, &mut h);
        }
        f(inst.metallic, &mut h);
        f(inst.roughness, &mut h);
        // Cast Shadows changes which rays the instance blocks.
        inst.shadow_casting.hash(&mut h);
    }
    for v in [
        gather.sky_color[0],
        gather.sky_color[1],
        gather.sky_color[2],
        gi.intensity,
        gi.probe_spacing,
        gi.temporal_blend,
        gi.emissive_boost,
    ] {
        f(v, &mut h);
    }
    gi.bounces.hash(&mut h);
    gi.samples.hash(&mut h);
    gi.gdf_resolution.hash(&mut h);
    gi.probe_fallback.hash(&mut h);
    // Backend changes can alter the gather layout and geometry included above.
    gi.mode.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_assets::{GiMode, RealtimeGi};

    fn gather() -> BakeGather {
        BakeGather {
            instance_sources: Vec::new(),
            instances: vec![zestengine_render::BakeInstance {
                mesh: zestengine_render::MeshHandle::NONE,
                transform: glam::Mat4::IDENTITY,
                lightmap_size: 8,
                albedo: [0.7; 3], emission: [0.0; 3], metallic: 0.0, roughness: 0.5,
                shadow_casting: zestengine_render::ShadowCasting::On,
            }],
            instance_objects: vec![0], instance_static: vec![false],
            lights: Vec::new(), probes: Vec::new(), probe_volumes: Vec::new(), sky_color: [0.0; 3],
        }
    }

    #[test]
    fn realtime_inputs_refresh_after_settings_and_material_edits() {
        let mut g = gather();
        let gi = RealtimeGi::default();
        let initial = hash_inputs(&g, &gi);
        assert_ne!(initial, hash_inputs(&g, &RealtimeGi { gdf_resolution: 64, ..gi }));
        assert_ne!(initial, hash_inputs(&g, &RealtimeGi { emissive_boost: 2.0, ..gi }));
        g.instance_sources.push(zestengine_render::GiSource {object:1,slot:0});
        let identified=hash_inputs(&g,&gi);
        g.instance_sources[0].slot=1;
        assert_ne!(identified,hash_inputs(&g,&gi),"identity-only changes must publish a new snapshot");
        let before_material=hash_inputs(&g,&gi);
        g.instances[0].metallic = 1.0;
        assert_ne!(before_material, hash_inputs(&g, &gi));
    }

    #[test]
    fn moving_geometry_refreshes_software_and_hardware_with_static_geometry() {
        for mode in [GiMode::Flux, GiMode::FluxRT] {
            for has_static in [false,true] {
                let gi = RealtimeGi { mode, ..Default::default() };
                let mut g = gather();
                if has_static {
                    g.instances.push(g.instances[0].clone());
                    g.instance_static.push(true);
                }
                let before = hash_inputs(&g, &gi);
                g.instances[0].transform = glam::Mat4::from_translation(glam::Vec3::X);
                assert_ne!(before, hash_inputs(&g, &gi));
            }
        }
    }
}

/// Hash only the inputs the cached GDF is built from: per-instance geometry
/// (world→local transform, world scale, mesh SDF identity) and materials
/// (albedo/emission), plus the field bounds and resolution. Lights/probes/sky
/// are NOT included: they change per trace but don't affect the distance field,
/// so the GDF survives a moving light and is re-uploaded only when geometry does.
fn hash_gdf_inputs(
    insts: &[crate::sw_gi::SdfInstance],
    min: glam::Vec3,
    max: glam::Vec3,
    dims: [u32; 3],
) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    let f = |x: f32, h: &mut std::collections::hash_map::DefaultHasher| x.to_bits().hash(h);
    for inst in insts {
        inst.gi_source.hash(&mut h);
        for v in inst.inv.to_cols_array() {
            f(v, &mut h);
        }
        f(inst.scale, &mut h);
        f(inst.metallic, &mut h);
        f(inst.roughness, &mut h);
        for v in inst.albedo.iter().chain(inst.emission.iter()) {
            f(*v, &mut h);
        }
        // Mesh SDF identity: same Arc → same field, so pointer is enough.
        (std::sync::Arc::as_ptr(&inst.sdf) as usize).hash(&mut h);
    }
    for v in min.to_array().iter().chain(max.to_array().iter()) {
        f(*v, &mut h);
    }
    dims.hash(&mut h);
    h.finish()
}
