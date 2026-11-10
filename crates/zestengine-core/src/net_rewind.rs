//! Lag compensation: the host's short history of networked hitboxes, so a
//! shot can be tested against where targets were when the shooter saw them.
//!
//! The host records every networked object that has a hit shape (Character
//! Controller capsule, Sphere / Capsule / Box Collider) each frame, stamped
//! with the session clock. A client firing sends `ctx.net_view_time()` (or
//! the host derives it from the shooter's latency with
//! `ctx.ray_cast_rewound`), and the host tests the ray against the poses at
//! that time with `ctx.ray_cast_at`. Rewinds are capped at [`MAX_REWIND`].
//! Only the recorded hitboxes are rewound, not world geometry: compare the
//! hit distance with a normal `ctx.raycast` for walls in the way.

use std::collections::{HashMap, VecDeque};

use glam::{Quat, Vec3};

use crate::{MotionContext, ObjectId};

/// How far back a shot may be rewound (seconds).
pub const MAX_REWIND: f64 = 0.5;
/// Seconds of history kept per object.
const KEEP: f64 = 1.0;

/// A hit volume in the object's local space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HitShape {
    /// A sphere.
    Sphere {
        /// Centre in object-local space.
        center: Vec3,
        /// Radius.
        radius: f32,
    },
    /// Segment `a`..`b` swept by `radius`.
    Capsule {
        /// Segment start in object-local space.
        a: Vec3,
        /// Segment end in object-local space.
        b: Vec3,
        /// Radius of the sweep.
        radius: f32,
    },
    /// An axis-aligned box.
    Box {
        /// Centre in object-local space.
        center: Vec3,
        /// Half extents.
        half: Vec3,
    },
}

/// An object's world pose at one moment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HitPose {
    /// Session clock of the record.
    pub time: f64,
    /// World position.
    pub pos: Vec3,
    /// World rotation.
    pub rot: Quat,
    /// World scale.
    pub scale: Vec3,
}

/// A rewound hit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RewindHit {
    /// The object that was hit.
    pub object: ObjectId,
    /// Hit position in world space.
    pub point: Vec3,
    /// Distance from the ray origin.
    pub distance: f32,
    /// The time the target was tested at.
    pub time: f64,
}

#[derive(Clone, Debug, Default)]
struct Track {
    shapes: Vec<HitShape>,
    poses: VecDeque<HitPose>,
}

/// Recent poses of every hittable networked object (host side).
#[derive(Clone, Debug)]
pub struct HitHistory {
    tracks: HashMap<ObjectId, Track>,
    /// Time of the newest record.
    pub now: f64,
    /// How far back a rewind may go (never more than [`HARD_MAX_REWIND`](crate::HARD_MAX_REWIND)).
    max_rewind: f64,
    /// Seconds a timestamp may be ahead of the clock.
    future_tolerance: f64,
    /// Reject out-of-window timestamps instead of clamping them (dedicated servers).
    strict: bool,
    /// Rewind queries rejected for a timestamp from the future, older than
    /// the cap, or not a number (strict mode).
    rejected: std::cell::Cell<u64>,
}

impl Default for HitHistory {
    fn default() -> Self {
        Self { tracks: HashMap::new(), now: 0.0, max_rewind: MAX_REWIND, future_tolerance: 0.1, strict: false, rejected: Default::default() }
    }
}

/// Why a rewind timestamp was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RewindReject {
    /// The timestamp was not a finite number.
    NotANumber,
    /// The timestamp is ahead of the server clock.
    Future,
    /// The timestamp is older than the rewind window.
    TooOld,
}

/// Check a client-supplied rewind time against the server clock `now`: it
/// must be a finite number, not more than `future_tolerance` ahead of `now`
/// and not older than `max_rewind` (itself capped below one second).
pub fn check_rewind_time(now: f64, time: f64, max_rewind: f64, future_tolerance: f64) -> Result<f64, RewindReject> {
    let cap = max_rewind.clamp(0.0, crate::HARD_MAX_REWIND as f64);
    if !time.is_finite() {
        Err(RewindReject::NotANumber)
    } else if time > now + future_tolerance {
        Err(RewindReject::Future)
    } else if time < now - cap {
        Err(RewindReject::TooOld)
    } else {
        Ok(time.min(now))
    }
}

impl HitHistory {
    /// Empty the history and restart the clock at `now` (session clock).
    /// A host taking over after a migration has none of the old host's
    /// recorded hitboxes, so its lag-compensation history starts fresh at
    /// the handover moment; any poses from an earlier stint as host are
    /// dropped so rewound ray casts never serve stale hitboxes.
    pub fn restart(&mut self, now: f64) {
        self.tracks.clear();
        self.now = now;
    }

    /// Record an object's pose (call once per frame per object).
    pub fn record(&mut self, id: ObjectId, time: f64, pos: Vec3, rot: Quat, scale: Vec3, shapes: &[HitShape]) {
        self.now = self.now.max(time);
        let t = self.tracks.entry(id).or_default();
        if t.shapes != shapes {
            t.shapes = shapes.to_vec();
        }
        if t.poses.back().is_some_and(|p| p.time >= time) {
            t.poses.pop_back();
        }
        t.poses.push_back(HitPose { time, pos, rot, scale });
        while t.poses.front().is_some_and(|p| p.time < time - KEEP) {
            t.poses.pop_front();
        }
    }

    /// Forget objects not recorded for a while (despawned, or no longer hittable).
    pub fn prune(&mut self, now: f64) {
        self.tracks.retain(|_, t| t.poses.back().is_some_and(|p| now - p.time < KEEP));
    }

    /// Set the rewind window (clamped to `0..=HARD_MAX_REWIND`) and whether
    /// timestamps outside it are rejected (`strict`) or clamped into it.
    pub fn configure(&mut self, max_rewind: f32, future_tolerance: f32, strict: bool) {
        let m = if max_rewind.is_finite() { max_rewind } else { MAX_REWIND as f32 };
        self.max_rewind = (m.clamp(0.0, crate::HARD_MAX_REWIND)) as f64;
        self.future_tolerance = if future_tolerance.is_finite() { future_tolerance.clamp(0.0, 1.0) as f64 } else { 0.1 };
        self.strict = strict;
    }

    /// The rewind window in seconds.
    pub fn max_rewind(&self) -> f64 {
        self.max_rewind
    }

    /// Rewind queries refused so far (strict mode).
    pub fn rejected(&self) -> u64 {
        self.rejected.get()
    }

    /// Objects being tracked.
    pub fn len(&self) -> usize {
        self.tracks.len()
    }

    /// True when nothing is tracked.
    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }

    /// An object's pose at `time` (interpolated; clamped to the recorded span).
    pub fn pose_at(&self, id: ObjectId, time: f64) -> Option<HitPose> {
        sample(&self.tracks.get(&id)?.poses, time)
    }

    /// First recorded hitbox the ray hits, with every object posed as it was at `time`.
    /// A `time` outside the rewind window is clamped into it, or (strict mode:
    /// dedicated servers) refused with no hit, so a client cannot ask for a
    /// timestamp from the future or from before the cap.
    pub fn ray_cast_at(&self, time: f64, origin: Vec3, dir: Vec3, max_dist: f32, exclude: &[ObjectId]) -> Option<RewindHit> {
        let time = if self.strict {
            match check_rewind_time(self.now, time, self.max_rewind, self.future_tolerance) {
                Ok(t) => t,
                Err(_) => {
                    self.rejected.set(self.rejected.get() + 1);
                    return None;
                }
            }
        } else if time.is_finite() {
            time.clamp(self.now - self.max_rewind, self.now)
        } else {
            return None;
        };
        self.cast(time, origin, dir, max_dist, exclude)
    }

    /// Like [`ray_cast_at`](Self::ray_cast_at) but always clamps the time into
    /// the window: for times the server derived itself (round trip and view delay).
    pub fn ray_cast_clamped(&self, time: f64, origin: Vec3, dir: Vec3, max_dist: f32, exclude: &[ObjectId]) -> Option<RewindHit> {
        if !time.is_finite() {
            return None;
        }
        self.cast(time.clamp(self.now - self.max_rewind, self.now), origin, dir, max_dist, exclude)
    }

    fn cast(&self, time: f64, origin: Vec3, dir: Vec3, max_dist: f32, exclude: &[ObjectId]) -> Option<RewindHit> {
        let dir = dir.normalize_or_zero();
        if dir == Vec3::ZERO {
            return None;
        }
        let mut best: Option<RewindHit> = None;
        for (id, track) in &self.tracks {
            if exclude.contains(id) {
                continue;
            }
            let Some(pose) = sample(&track.poses, time) else {
                continue;
            };
            for shape in &track.shapes {
                if let Some(d) = ray_shape(origin, dir, shape, &pose)
                    && d <= max_dist
                    && best.is_none_or(|b| d < b.distance)
                {
                    best = Some(RewindHit { object: *id, point: origin + dir * d, distance: d, time });
                }
            }
        }
        best
    }
}

fn sample(poses: &VecDeque<HitPose>, time: f64) -> Option<HitPose> {
    let newest = *poses.back()?;
    if time >= newest.time {
        return Some(newest);
    }
    let i = poses.iter().position(|p| p.time > time)?;
    if i == 0 {
        return Some(poses[0]);
    }
    let (a, b) = (poses[i - 1], poses[i]);
    let t = ((time - a.time) / (b.time - a.time).max(1e-9)) as f32;
    Some(HitPose { time, pos: a.pos.lerp(b.pos, t), rot: a.rot.slerp(b.rot, t), scale: a.scale.lerp(b.scale, t) })
}

fn ray_shape(o: Vec3, d: Vec3, shape: &HitShape, pose: &HitPose) -> Option<f32> {
    let world = |p: Vec3| pose.pos + pose.rot * (p * pose.scale);
    let s = pose.scale.abs().max_element();
    match *shape {
        HitShape::Sphere { center, radius } => ray_sphere(o, d, world(center), radius * s),
        HitShape::Capsule { a, b, radius } => ray_capsule(o, d, world(a), world(b), radius * s),
        HitShape::Box { center, half } => {
            let inv = pose.rot.inverse();
            ray_box(inv * (o - world(center)), inv * d, half * pose.scale.abs())
        }
    }
}

fn ray_sphere(o: Vec3, d: Vec3, c: Vec3, r: f32) -> Option<f32> {
    let oc = o - c;
    let b = oc.dot(d);
    let cc = oc.dot(oc) - r * r;
    if cc <= 0.0 {
        return Some(0.0);
    }
    let h = b * b - cc;
    if h < 0.0 {
        return None;
    }
    let t = -b - h.sqrt();
    (t >= 0.0).then_some(t)
}

fn ray_capsule(o: Vec3, d: Vec3, pa: Vec3, pb: Vec3, r: f32) -> Option<f32> {
    let ba = pb - pa;
    let oa = o - pa;
    let baba = ba.dot(ba);
    if baba < 1e-10 {
        return ray_sphere(o, d, pa, r);
    }
    let bard = ba.dot(d);
    let baoa = ba.dot(oa);
    let a = baba - bard * bard;
    let mut best = None;
    if a.abs() > 1e-8 {
        let b = baba * d.dot(oa) - baoa * bard;
        let c = baba * oa.dot(oa) - baoa * baoa - r * r * baba;
        let h = b * b - a * c;
        if h >= 0.0 {
            let t = (-b - h.sqrt()) / a;
            let y = baoa + t * bard;
            if y > 0.0 && y < baba && t >= 0.0 {
                best = Some(t);
            }
        }
    }
    // End caps (also covers rays parallel to the axis and origins inside).
    for cap in [ray_sphere(o, d, pa, r), ray_sphere(o, d, pb, r)].into_iter().flatten() {
        if best.is_none_or(|b| cap < b) {
            best = Some(cap);
        }
    }
    best
}

fn ray_box(o: Vec3, d: Vec3, half: Vec3) -> Option<f32> {
    let mut tmin = 0.0f32;
    let mut tmax = f32::INFINITY;
    for k in 0..3 {
        if d[k].abs() < 1e-9 {
            if o[k].abs() > half[k] {
                return None;
            }
            continue;
        }
        let inv = 1.0 / d[k];
        let (t1, t2) = ((-half[k] - o[k]) * inv, (half[k] - o[k]) * inv);
        tmin = tmin.max(t1.min(t2));
        tmax = tmax.min(t1.max(t2));
        if tmin > tmax {
            return None;
        }
    }
    Some(tmin)
}

impl crate::ecs::ObjectContext<'_, '_> {
    /// The session clock in seconds: the host's time as this peer estimates it.
    ///
    /// **Immediate.** Every peer reads nearly the same value at the same moment, so it can time-stamp events across the
    /// network. It never jumps or runs backwards, not even across a host migration. It is `0.0` offline and on a client
    /// that has not heard from the host yet.
    pub fn net_time(&self) -> f64 {
        self.net_view().clock
    }

    /// The session-clock time of the world state this peer is showing for remote objects right now. Send it with a shot
    /// so the host can test the hit against what the shooter actually saw.
    ///
    /// Remote objects are drawn behind real time by the one-way transit delay plus the interpolation delay, so this is
    /// `net_time() - rtt / 2 - view_delay`, in seconds.
    ///
    /// **Immediate.** A read of the network view. Outside a session it is `0.0`.
    ///
    /// [`NetValue`](crate::NetValue) has no `f64`. Sending this as a `Float` (`f32`) loses precision on long sessions (about 1 ms after
    /// three hours, 8 ms after a day), so send the eight bytes instead, as in the example. The host reads it back with
    /// [`ray_cast_at`](Self::ray_cast_at), or ignores it and uses [`ray_cast_rewound`](Self::ray_cast_rewound).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Client: say where and when the shot was fired from.
    /// let seen_at = NetValue::Bytes(ctx.net_view_time().to_le_bytes().to_vec());
    /// ctx.send_rpc(None, "Fire", vec![origin.into(), dir.into(), seen_at], RpcTarget::Server, true);
    /// ```
    pub fn net_view_time(&self) -> f64 {
        self.net_view().clock - self.net_view().rtt_ms as f64 / 2000.0 - self.net_view().view_delay as f64
    }

    /// Host only: tests a ray against the networked hitboxes posed as they were at session-clock `time` (lag
    /// compensation), ignoring this object.
    ///
    /// **Immediate.** A read of the host's recorded hitbox history. It returns the nearest [`RewindHit`] (object, point,
    /// distance, the time tested) within `max_dist` metres of `origin` along `dir`. `dir` need not be normalized, but a
    /// zero vector hits nothing.
    ///
    /// # Returns
    ///
    /// `None` when nothing is hit, when called offline or on a client (only the host records a history), when `dir` is
    /// zero, or when `time` is not a finite number. On a dedicated server, or with strict validation on, `None` also
    /// covers a `time` outside the rewind window (from the future, or too old): it is refused and counted instead of
    /// being clamped. Elsewhere such a `time` is clamped into the window. The window is the scene's
    /// [`NetValidation::max_rewind`](crate::NetValidation::max_rewind) (0.25 s by default, never more than
    /// [`HARD_MAX_REWIND`](crate::HARD_MAX_REWIND)); [`MAX_REWIND`] is only the default of a bare [`HitHistory`].
    ///
    /// # What is rewound
    ///
    /// Only hitboxes: networked objects ([`NetworkObject`](crate::NetworkObject) with `sync_transform` on) that have a Character Controller
    /// capsule or a non-trigger Sphere, Capsule or Box Collider. The host records them once per frame, after the frame's
    /// systems have run, with the pose the other peers are shown. World geometry and physics bodies are not rewound:
    /// compare `distance` with [`raycast`](Self::raycast) to see whether a wall was in the way.
    ///
    /// # Gotchas
    ///
    /// Only this object is ignored, not the shooter's avatar. A ray that starts inside a hitbox hits it at (or very
    /// near) distance 0, so exclude the shooter's own objects with
    /// [`ray_cast_at_filtered`](Self::ray_cast_at_filtered), or use [`ray_cast_rewound`](Self::ray_cast_rewound), which
    /// works out the time on the server and excludes everything the shooter owns. `time`, `origin` and `dir` normally
    /// come from a client: check them against what the shooter's avatar could have done.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Host: the client sent "Fire" with origin, direction and its `net_view_time` (8 bytes).
    /// let shots: Vec<(Vec3, Vec3, f64)> = ctx.rpcs_named("Fire").filter_map(|rpc| {
    ///     let (origin, dir) = (rpc.arg(0)?.as_vec3()?, rpc.arg(1)?.as_vec3()?);
    ///     let seen_at = f64::from_le_bytes(rpc.arg(2)?.as_bytes()?.try_into().ok()?);
    ///     Some((origin, dir, seen_at))
    /// }).collect();
    /// for (origin, dir, seen_at) in shots {
    ///     let hit = ctx.ray_cast_at(seen_at, origin, dir, 100.0);
    ///     let wall = ctx.raycast(origin, dir, 100.0);   // geometry is not rewound
    ///     if let Some(hit) = hit.filter(|h| wall.is_none_or(|w| w.distance > h.distance)) {
    ///         self.apply_damage(hit.object, 10);
    ///     }
    /// }
    /// ```
    pub fn ray_cast_at(&self, time: f64, origin: Vec3, dir: Vec3, max_dist: f32) -> Option<RewindHit> {
        self.ray_cast_at_filtered(time, origin, dir, max_dist, &[self.self_id()])
    }

    /// Host only: like [`ray_cast_at`](Self::ray_cast_at), but ignoring exactly the objects in `exclude` instead of just
    /// this object.
    ///
    /// **Immediate.** Use it to leave out the shooter's own hitboxes (its avatar and whatever it carries). `exclude` is
    /// the whole list: this object is not added to it. The `time` handling (clamped into the rewind window, or refused
    /// with `None` on a dedicated server), what is rewound, and the `None` cases (offline, not the host, nothing hit) are
    /// the same as for [`ray_cast_at`](Self::ray_cast_at); the query itself is [`HitHistory::ray_cast_at`].
    pub fn ray_cast_at_filtered(&self, time: f64, origin: Vec3, dir: Vec3, max_dist: f32, exclude: &[ObjectId]) -> Option<RewindHit> {
        self.net_view().rewind.as_ref()?.lock().ok()?.ray_cast_at(time, origin, dir, max_dist, exclude)
    }

    /// Host only: tests a ray against the hitboxes as `shooter` saw the world when it fired, with the time worked out by
    /// the server.
    ///
    /// The time is the session clock now, minus the shooter's measured round trip ([`PeerInfo::rtt_ms`](crate::PeerInfo::rtt_ms)), minus the view
    /// delay it reported ([`PeerInfo::view_delay`](crate::PeerInfo::view_delay)). It is always clamped into the rewind window and never refused, so a
    /// client cannot ask for a time of its own. This object and everything the shooter owns (its player and the props it
    /// holds) are ignored, so a shot cannot hit its own shooter.
    ///
    /// **Immediate.** A read of the host's recorded hitbox history. It returns the nearest [`RewindHit`] within `max_dist`
    /// metres of `origin` along `dir`.
    ///
    /// # Returns
    ///
    /// `None` when nothing is hit, when called offline or on a client (only the host records a history), when `dir` is
    /// zero, and when `shooter` is not a peer in the session (its latency is unknown). What is rewound, and what is not,
    /// is described on [`ray_cast_at`](Self::ray_cast_at): hitboxes only, so compare the distance with
    /// [`raycast`](Self::raycast) for walls. `origin` and `dir` normally come from the client; on a dedicated server
    /// check them against the shooter's avatar before trusting them.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Host: a client asked to fire. `rpc.from` is the connection the RPC arrived on, so it cannot be forged.
    /// let shots: Vec<(u64, Vec3, Vec3)> = ctx.rpcs_named("Fire")
    ///     .filter_map(|rpc| Some((rpc.from, rpc.arg(0)?.as_vec3()?, rpc.arg(1)?.as_vec3()?)))
    ///     .collect();
    /// for (shooter, origin, dir) in shots {
    ///     let hit = ctx.ray_cast_rewound(shooter, origin, dir, 100.0);
    ///     let wall = ctx.raycast(origin, dir, 100.0);   // geometry is not rewound
    ///     if let Some(hit) = hit.filter(|h| wall.is_none_or(|w| w.distance > h.distance)) {
    ///         self.apply_damage(hit.object, 10);
    ///     }
    /// }
    /// ```
    pub fn ray_cast_rewound(&self, shooter: u64, origin: Vec3, dir: Vec3, max_dist: f32) -> Option<RewindHit> {
        let peer = self.net_view().peer(shooter)?;
        let time = self.net_view().clock - peer.rtt_ms as f64 / 1000.0 - peer.view_delay as f64;
        let mut exclude: Vec<ObjectId> = self.net_view().owners.iter().filter(|(_, o)| **o == shooter).map(|(id, _)| *id).collect();
        exclude.push(self.self_id());
        // The time is the server's own estimate: clamp it into the window instead of refusing.
        self.net_view().rewind.as_ref()?.lock().ok()?.ray_cast_clamped(time, origin, dir, max_dist, &exclude)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewind_times_are_checked_against_the_window() {
        use crate::RewindReject::*;
        assert_eq!(check_rewind_time(10.0, 9.9, 0.25, 0.1), Ok(9.9));
        assert_eq!(check_rewind_time(10.0, 10.05, 0.25, 0.1), Ok(10.0), "a little ahead is clamped to now");
        assert_eq!(check_rewind_time(10.0, 10.5, 0.25, 0.1), Err(Future));
        assert_eq!(check_rewind_time(10.0, 9.7, 0.25, 0.1), Err(TooOld));
        assert_eq!(check_rewind_time(10.0, f64::NAN, 0.25, 0.1), Err(NotANumber));
        // A cap above the hard limit is cut to it: 0.95 s ago is too old even for a "5 s" window.
        assert_eq!(check_rewind_time(10.0, 9.05, 5.0, 0.1), Err(TooOld));
        assert!(check_rewind_time(10.0, 9.2, 5.0, 0.1).is_ok());
        let mut h = HitHistory::default();
        h.configure(5.0, 0.1, true);
        assert!(h.max_rewind() < 1.0, "the history window is capped below a second");
        h.configure(f32::NAN, 0.1, false);
        assert_eq!(h.max_rewind(), MAX_REWIND, "a nonsense setting keeps the default");
    }

    #[test]
    fn rewound_ray_hits_where_the_target_was() {
        let mut h = HitHistory::default();
        let id = ObjectId::new();
        let capsule = [HitShape::Capsule { a: Vec3::new(0.0, 0.35, 0.0), b: Vec3::new(0.0, 1.45, 0.0), radius: 0.35 }];
        // The target runs along +X at 5 m/s, recorded at 60 Hz for a second.
        for k in 0..=60 {
            let t = k as f64 / 60.0;
            h.record(id, t, Vec3::new(5.0 * t as f32, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE, &capsule);
        }
        // A shooter at z = 10 aiming at x = 3 (where the target was at t = 0.6).
        let origin = Vec3::new(3.0, 1.0, 10.0);
        let hit = h.ray_cast_at(0.6, origin, Vec3::NEG_Z, 100.0, &[]).expect("hit in the past");
        assert_eq!(hit.object, id);
        assert!((hit.distance - (10.0 - 0.35)).abs() < 1e-3, "{hit:?}");
        assert!(h.ray_cast_at(1.0, origin, Vec3::NEG_Z, 100.0, &[]).is_none(), "the target has moved on by now");
        assert!(h.ray_cast_at(0.6, origin, Vec3::NEG_Z, 100.0, &[id]).is_none(), "excluded");
        // Rewinds are clamped to MAX_REWIND: t = 0.1 is tested at 0.5.
        let at = h.ray_cast_at(0.1, Vec3::new(2.5, 1.0, 10.0), Vec3::NEG_Z, 100.0, &[]).expect("clamped time");
        assert!((at.time - 0.5).abs() < 1e-9);
        assert!(h.pose_at(id, 0.25).is_some_and(|p| (p.pos.x - 1.25).abs() < 1e-4), "interpolated pose");
        h.prune(5.0);
        assert!(h.is_empty());
    }

    #[test]
    fn shapes() {
        let pose = HitPose { time: 0.0, pos: Vec3::new(0.0, 0.0, -5.0), rot: Quat::from_rotation_y(0.7), scale: Vec3::ONE };
        let b = HitShape::Box { center: Vec3::ZERO, half: Vec3::splat(0.5) };
        assert!(ray_shape(Vec3::ZERO, Vec3::NEG_Z, &b, &pose).is_some_and(|d| d > 4.2 && d < 4.6));
        assert!(ray_shape(Vec3::new(2.0, 0.0, 0.0), Vec3::NEG_Z, &b, &pose).is_none());
        let s = HitShape::Sphere { center: Vec3::ZERO, radius: 1.0 };
        assert!(ray_shape(Vec3::ZERO, Vec3::NEG_Z, &s, &pose).is_some_and(|d| (d - 4.0).abs() < 1e-4));
        // A capsule seen end-on (ray along its axis) is hit at its cap.
        let c = HitShape::Capsule { a: Vec3::new(0.0, 0.0, -1.0), b: Vec3::new(0.0, 0.0, 1.0), radius: 0.5 };
        let p0 = HitPose { rot: Quat::IDENTITY, ..pose };
        assert!(ray_shape(Vec3::ZERO, Vec3::NEG_Z, &c, &p0).is_some_and(|d| (d - 3.5).abs() < 1e-4));
    }
}
