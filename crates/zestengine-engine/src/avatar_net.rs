//! Avatar parameter replication (`P_AVPARAM`).
//!
//! The worn avatar's synced parameters travel as one bit-packed packet keyed
//! by a schema hash (`zestengine_core::avatar_sync`), not as name + f32 pairs
//! like `NetworkAnimator`: at most 32 payload bytes for the default 256-bit
//! budget, so it never nears the MTU. The owner sends when a synced value
//! changed (10 Hz, 30 Hz while a puppet is open) and repeats the whole state
//! once a second; receivers drop packets whose schema hash differs, apply
//! Bools and Ints at once and ease Floats toward the new value over the
//! send interval.
//!
//! Per avatar and frame this is a loop over the synced entries (at most a few
//! hundred, usually a dozen) with no allocation after the first frame.

use zestengine_core::{AvParamSchema, AvatarParameters, NetworkAvatar, ObjectId, SyncClock, smooth_step};

use crate::net::NetSession;
use crate::scene::LoadedScene;

/// A remote float within this of its target counts as arrived.
const ARRIVED: f32 = 1.0 / 512.0;
/// Shortest and longest smoothing time, seconds.
const MIN_INTERVAL: f32 = 0.03;
const MAX_INTERVAL: f32 = 1.0;

/// Per-avatar send / receive memory, kept in the session.
#[derive(Default)]
pub struct AvatarNetState {
    schema: Option<AvParamSchema>,
    store_len: usize,
    clock: SyncClock,
    last_sent: Vec<f32>,
    scratch: Vec<u8>,
    // Receiving.
    seq_seen: u64,
    targets: Vec<f32>,
    last_recv: f64,
    interval: f32,
    /// Packets dropped because their schema hash differed.
    pub rejected: u32,
    warned: bool,
    // The avatar root below the network object, cached.
    root: Option<(usize, ObjectId)>,
    scanned: Option<f64>,
}

impl AvatarNetState {
    pub fn schema_hash(&self) -> Option<u32> {
        self.schema.as_ref().map(AvParamSchema::hash)
    }
}

/// The avatar (an object with `AvatarParameters`) at or below `net_obj`.
fn find_root(scene: &LoadedScene, net_obj: usize) -> Option<usize> {
    (0..scene.objects.len()).find(|&k| {
        scene.objects[k].components.iter().any(|c| c.as_any().is::<AvatarParameters>())
            && std::iter::successors(Some(k), |&p| scene.objects[p].parent).take(64).any(|p| p == net_obj)
    })
}

/// Replicate the avatar parameters of the network object `i` (id `raw`).
/// `authority`: this peer owns it, so it sends; otherwise it receives.
pub fn sync(scene: &mut LoadedScene, s: &mut NetSession, i: usize, raw: u128, authority: bool, now: f64, dt: f32) {
    if !scene.objects[i].components.iter().any(|c| c.as_any().is::<NetworkAvatar>()) {
        return;
    }
    let mut st = s.avparam_state.remove(&raw).unwrap_or_default();
    // Where the avatar's parameters live: cached, rescanned at most once a second.
    let cached = st.root.and_then(|(k, id)| scene.objects.get(k).filter(|o| o.id == id).map(|_| k));
    let root = cached.or_else(|| {
        if st.scanned.is_some_and(|t| now - t < 1.0) {
            return None;
        }
        st.scanned = Some(now);
        let k = find_root(scene, i)?;
        st.root = Some((k, scene.objects[k].id));
        Some(k)
    });
    if let Some(k) = root {
        step(scene, s, &mut st, k, raw, authority, now, dt);
    }
    s.avparam_state.insert(raw, st);
}

#[allow(clippy::too_many_arguments)]
fn step(scene: &mut LoadedScene, s: &mut NetSession, st: &mut AvatarNetState, k: usize, raw: u128, authority: bool, now: f64, dt: f32) {
    let Some(ap) = scene.objects[k].components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<AvatarParameters>()) else {
        return;
    };
    if !ap.ready {
        return;
    }
    let fast = ap.fast_sync;
    let store = &mut ap.store;
    if st.schema.is_none() || st.store_len != store.len() {
        let schema = AvParamSchema::from_store(store);
        st.last_sent = vec![f32::NAN; schema.len()];
        st.targets.clear();
        st.schema = Some(schema);
        st.store_len = store.len();
    }
    let Some(schema) = st.schema.as_ref() else { return };
    if schema.is_empty() {
        return;
    }
    if authority {
        let mut changed = false;
        for (e, last) in schema.entries().iter().zip(&st.last_sent) {
            if store.get(e.slot).to_bits() != last.to_bits() {
                changed = true;
                break;
            }
        }
        if !st.clock.poll(now, changed, fast) {
            return;
        }
        st.scratch.clear();
        schema.encode(store, &mut st.scratch);
        s.send_avparam(raw, &st.scratch);
        for (e, last) in schema.entries().iter().zip(st.last_sent.iter_mut()) {
            *last = store.get(e.slot);
        }
        return;
    }

    // Receiving: take the newest packet once, then ease toward it each frame.
    if let Some((seq, bytes)) = s.avparam_in.get(&raw)
        && *seq != st.seq_seen
    {
        st.seq_seen = *seq;
        match schema.decode_values(bytes, &mut st.targets) {
            Ok(()) => {
                let gap = (now - st.last_recv) as f32;
                st.interval = if st.last_recv > 0.0 { gap.clamp(MIN_INTERVAL, MAX_INTERVAL) } else { MIN_INTERVAL };
                st.last_recv = now;
            }
            Err(e) => {
                st.rejected += 1;
                if !st.warned {
                    st.warned = true;
                    tracing::warn!("avatar parameters from a peer ignored: {e}");
                }
            }
        }
    }
    if st.targets.len() != schema.len() {
        return;
    }
    for (e, &target) in schema.entries().iter().zip(&st.targets) {
        let cur = store.get(e.slot);
        if cur == target {
            continue;
        }
        let next = if e.kind == zestengine_core::AvatarParamKind::Float {
            let n = smooth_step(cur, target, dt, st.interval);
            if (n - target).abs() < ARRIVED { target } else { n }
        } else {
            target
        };
        store.set(e.slot, next);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_core::{AvatarParam, AvatarParamKind, AvatarParamStore};

    #[test]
    fn floats_ease_and_ints_snap_toward_a_decoded_packet() {
        // The per-entry rule `step` applies, on a bare store.
        let params = [AvatarParam::new("A", AvatarParamKind::Float), AvatarParam::new("N", AvatarParamKind::Int)];
        let mut store = AvatarParamStore::new(&params);
        let schema = AvParamSchema::from_store(&store);
        let mut tx = store.clone();
        tx.set(tx.index_of("A").unwrap(), 1.0);
        tx.set(tx.index_of("N").unwrap(), 5.0);
        let mut bytes = Vec::new();
        schema.encode(&tx, &mut bytes);
        let mut targets = Vec::new();
        schema.decode_values(&bytes, &mut targets).unwrap();
        for (e, &t) in schema.entries().iter().zip(&targets) {
            let cur = store.get(e.slot);
            let next = if e.kind == AvatarParamKind::Float { smooth_step(cur, t, 0.05, 0.1) } else { t };
            store.set(e.slot, next);
        }
        assert!((store.get(store.index_of("A").unwrap()) - 0.5).abs() < 1e-6, "halfway after half an interval");
        assert_eq!(store.get(store.index_of("N").unwrap()), 5.0);
    }
}
