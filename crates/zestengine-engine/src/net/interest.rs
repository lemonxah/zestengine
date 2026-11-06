//! Host-side interest management (NET-5): who is told about which object, how
//! often, and within what bandwidth. A child module of `net`, so it works on
//! the session's private state; the data structures (spatial hash, visible
//! set with hysteresis, tiers, budget) are in `zestengine_core::net_interest`.
//!
//! Off unless `NetConfig::interest.enabled` (and only on the host). Then:
//!
//! - every refresh (`refresh_hz`) the grid is rebuilt from the replicated
//!   transforms and each client's visible set is recomputed around its player
//!   object; an object that enters is sent as a spawn plus its state, one that
//!   leaves is despawned on that client (runtime spawns) or just stops being
//!   updated (scene objects);
//! - reliable object messages (owner, variables, active, parent ...) only go
//!   to clients that see the object;
//! - snapshot entries wait in a per-client, per-object slot (a newer one
//!   merges into an unsent older one, so no field is lost) until the object's
//!   distance tier allows another update; due entries leave the slot highest
//!   waiting-time-to-interval ratio first while the client's byte budget
//!   lasts, so nothing waits forever;
//! - what leaves is batched: entries of many senders share MTU-sized
//!   `P_SNAPB` packets, with 32-bit ids once the client knows the mapping.

use std::sync::Arc;

use super::*;
use zestengine_core::{ClientInterest, InterestChange, InterestConfig, SpatialGrid};
#[cfg(test)]
use zestengine_core::NetValidation;

/// A snapshot entry waiting for its turn to go to one client.
#[derive(Clone, Copy)]
struct Pend {
    entry: SnapEntry,
    origin: u64,
    time: f64,
}

#[derive(Default)]
struct ObjSlot {
    pending: Option<Pend>,
    last_sent: f64,
    last_pose: f64,
    /// What this client was last sent of the object, field by field, and when the last full
    /// entry went (per-client delta baseline).
    baseline: Option<SnapEntry>,
    baseline_full: f64,
}

/// `e` against what the client already has: the fields that changed, or `None` when nothing
/// did. `base` is updated to what the client will have after this entry. A `refresh` due entry
/// goes out whole. Precision follows the entry's own quantisation.
pub(super) fn delta_against(base: &mut Option<SnapEntry>, e: &SnapEntry, now: f64, last_full: &mut f64, refresh: f32) -> Option<SnapEntry> {
    let full = base.is_none() || now - *last_full >= refresh as f64;
    if full {
        *base = Some(*e);
        *last_full = now;
        return Some(*e);
    }
    let b = base.as_mut().expect("checked above");
    let (pe, re) = if e.quantized { (0.001, 0.1f32) } else { (0.0001, 0.01) };
    let mut out = SnapEntry { id: e.id, pos: None, rot: None, scale: None, vel: None, quantized: e.quantized };
    let mut any = false;
    if let Some(p) = e.pos
        && b.pos.is_none_or(|q| q.distance(p) > pe)
    {
        out.pos = Some(p);
        b.pos = Some(p);
        any = true;
    }
    if let Some(r) = e.rot
        && b.rot.is_none_or(|q| q.angle_between(r).to_degrees() > re)
    {
        out.rot = Some(r);
        b.rot = Some(r);
        any = true;
    }
    if let Some(s) = e.scale
        && b.scale.is_none_or(|q| q.distance(s) > 1e-4)
    {
        out.scale = Some(s);
        b.scale = Some(s);
        any = true;
    }
    if let Some(v) = e.vel
        && b.vel.is_none_or(|q| q.distance(v) > 0.01)
    {
        out.vel = Some(v);
        b.vel = Some(v);
        any = true;
    }
    any.then_some(out)
}

/// One client's interest state.
pub(super) struct ClientNet {
    ci: ClientInterest,
    objs: HashMap<u128, ObjSlot>,
    /// Objects with a pending entry.
    dirty: Vec<u128>,
    /// The ids this client learned a 32-bit alias for.
    short_known: HashSet<u128>,
    /// Packets built / entries pending now (stats).
    pending: u32,
}

/// What a class of unreliable per-object traffic is subject to.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Stream {
    /// Animator parameters, avatar parameters: relevance only (they are sent
    /// on change, so a skipped one would never come back).
    State,
    /// Avatar part poses: relevance, tier rate, budget.
    Pose,
    /// Face tracking: relevance, detail range, tier rate, budget.
    Face,
}

#[derive(Default)]
pub struct Interest {
    pub(super) cfg: Arc<InterestConfig>,
    grid: SpatialGrid,
    /// Position of every object the grid knows.
    pos: HashMap<u128, Vec3>,
    pub(super) clients: HashMap<u64, ClientNet>,
    players: Vec<u128>,
    changes: Vec<InterestChange>,
    events: Vec<(u64, InterestChange)>,
    peers: Vec<u64>,
    scored: Vec<(f32, u128)>,
    keep: Vec<u128>,
    batch: Vec<(u64, f64, SnapEntry)>,
    buf: Vec<u8>,
    next_refresh: f64,
    last_tick: f64,
    /// Host: 32-bit aliases handed out so far.
    short: HashMap<u128, u32>,
    next_short: u32,
    /// Client: the aliases the host taught us.
    pub(super) short_in: HashMap<u32, u128>,
    /// Client: the same aliases the other way round (ids we send upstream as 32 bits).
    pub(super) short_out: HashMap<u128, u32>,
    /// Host: alias -> id, to read the aliases clients send upstream.
    short_rev: HashMap<u32, u128>,
}

/// One client's numbers for the stats UI and the admin console.
#[derive(Clone, Copy, Debug, Default)]
pub struct InterestClientStats {
    pub peer: u64,
    /// Objects the client is told about.
    pub relevant: u32,
    /// Bytes per second sent to it over the last second.
    pub rate_out: f32,
    pub bytes_total: u64,
    /// Updates still waiting (budget or tier).
    pub pending: u32,
    /// Times an update was held back by the budget.
    pub dropped_budget: u64,
    /// Updates merged into a newer one before they were sent (tier rate).
    pub throttled: u64,
    /// Entries left out because the client already had every field (delta baselines).
    pub suppressed: u64,
    pub entered: u64,
    pub left: u64,
}

fn merge(old: SnapEntry, new: SnapEntry) -> SnapEntry {
    SnapEntry {
        id: new.id,
        pos: new.pos.or(old.pos),
        rot: new.rot.or(old.rot),
        scale: new.scale.or(old.scale),
        vel: new.vel.or(old.vel),
        quantized: new.quantized,
    }
}

impl Interest {
    pub(super) fn with_config(cfg: InterestConfig) -> Self {
        Self { cfg: Arc::new(cfg), ..Default::default() }
    }

    /// A new world (scene switch): forget what clients saw and the aliases.
    pub(super) fn reset(&mut self) {
        for c in self.clients.values_mut() {
            c.ci.visible.clear();
            c.ci.full.clear();
            c.objs.clear();
            c.dirty.clear();
            c.short_known.clear();
        }
        self.short.clear();
        self.short_rev.clear();
        self.short_in.clear();
        self.short_out.clear();
        self.pos.clear();
    }
}

impl NetSession {
    /// Interest management is on (hosts only).
    pub fn interest_on(&self) -> bool {
        self.is_host && self.interest.cfg.enabled
    }

    /// Per-client interest numbers (empty while interest management is off).
    pub fn interest_stats(&self) -> Vec<InterestClientStats> {
        let mut v: Vec<InterestClientStats> = self
            .interest
            .clients
            .iter()
            .map(|(p, c)| InterestClientStats {
                peer: *p,
                relevant: c.ci.visible.len() as u32,
                rate_out: c.ci.rate_out,
                bytes_total: c.ci.bytes_total,
                pending: c.pending,
                dropped_budget: c.ci.dropped_budget,
                throttled: c.ci.throttled,
                suppressed: c.ci.suppressed,
                entered: c.ci.entered,
                left: c.ci.left,
            })
            .collect();
        v.sort_by_key(|s| s.peer);
        v
    }

    /// The interest settings in force.
    pub fn interest_config(&self) -> &InterestConfig {
        &self.interest.cfg
    }

    /// Where an object is: its replicated transform, else where it spawned.
    fn object_pos(&self, id: u128) -> Option<Vec3> {
        self.world.xforms.get(&id).map(|x| x.pos).or_else(|| self.world.spawns.iter().find(|s| s.id == id).map(|s| s.translation))
    }

    /// Host: can `peer` see object `id`? (Its own objects, anything while
    /// interest management is off, otherwise its visible set.)
    fn sees(&self, peer: u64, id: u128) -> bool {
        if !self.interest_on() || self.owner_of_raw(id) == peer {
            return true;
        }
        self.interest.clients.get(&peer).is_some_and(|c| c.ci.visible.contains(&id))
    }

    /// A client's accounting hook for everything sent to `to` (see `send_raw`).
    pub(super) fn interest_account(&mut self, to: SocketAddr, bytes: usize) {
        if !self.interest.cfg.enabled || !self.is_host {
            return;
        }
        let Some(peer) = self.addr_peer.get(&to).copied() else {
            return;
        };
        let now = self.now();
        let cfg = self.interest.cfg.clone();
        if let Some(c) = self.interest.clients.get_mut(&peer) {
            c.ci.spend(&cfg, bytes, now);
        }
    }

    /// Host: once per pump. Budget refill, relevance refresh when due, flush.
    pub(super) fn interest_tick(&mut self, now: f64) {
        if !self.interest_on() {
            return;
        }
        let cfg = self.interest.cfg.clone();
        let dt = (now - self.interest.last_tick).clamp(0.0, 1.0) as f32;
        self.interest.last_tick = now;
        for c in self.interest.clients.values_mut() {
            c.ci.refill(&cfg, dt);
        }
        if now >= self.interest.next_refresh {
            self.interest.next_refresh = now + 1.0 / cfg.refresh_hz as f64;
            self.interest_refresh(&cfg);
            self.interest_apply_changes();
        }
        self.interest_flush(now);
    }

    fn interest_refresh(&mut self, cfg: &Arc<InterestConfig>) {
        let it = &mut self.interest;
        it.pos.clear();
        it.grid.clear(cfg.cell_size);
        for (id, x) in &self.world.xforms {
            it.pos.insert(*id, x.pos);
            it.grid.insert(*id, x.pos);
        }
        for s in &self.world.spawns {
            if let std::collections::hash_map::Entry::Vacant(e) = it.pos.entry(s.id) {
                e.insert(s.translation);
                it.grid.insert(s.id, s.translation);
            }
        }
        it.players.clear();
        it.players.extend(self.world.player_obj.values().copied());
        it.peers.clear();
        it.peers.extend(self.conns.iter().filter(|(p, c)| c.confirmed && self.peers.contains_key(p)).map(|(p, _)| *p));
        let n = it.peers.len();
        let (mut relevant, mut dropped, mut throttled) = (0u64, 0u64, 0u64);
        for i in 0..n {
            let peer = it.peers[i];
            let cn = it.clients.entry(peer).or_insert_with(|| ClientNet {
                ci: ClientInterest::new(cfg),
                objs: HashMap::new(),
                dirty: Vec::new(),
                short_known: HashSet::new(),
                pending: 0,
            });
            let pos = &it.pos;
            cn.ci.center = self.world.player_obj.get(&peer).and_then(|o| pos.get(o)).copied();
            it.changes.clear();
            cn.ci.refresh(cfg, &it.grid, &self.world.always_relevant, &it.players, |id| pos.get(&id).copied(), &mut it.changes);
            for ch in &it.changes {
                it.events.push((peer, *ch));
            }
            relevant += cn.ci.visible.len() as u64;
            dropped += cn.ci.dropped_budget;
            throttled += cn.ci.throttled;
        }
        // Clients that left.
        it.clients.retain(|p, _| self.conns.contains_key(p));
        self.stats.interest_clients = n as u32;
        self.stats.interest_objects = it.grid.len() as u32;
        self.stats.interest_relevant_avg = if n > 0 { relevant as f32 / n as f32 } else { 0.0 };
        self.stats.interest_dropped = dropped;
        self.stats.interest_throttled = throttled;
    }

    /// Send the spawns / states of what entered and the despawns of what left.
    fn interest_apply_changes(&mut self) {
        let mut events = std::mem::take(&mut self.interest.events);
        for (peer, ch) in events.drain(..) {
            match ch {
                InterestChange::Enter(id) => self.interest_enter(peer, id),
                InterestChange::Leave(id) => self.interest_leave(peer, id),
            }
        }
        self.interest.events = events;
    }

    fn short_of(&mut self, id: u128) -> u32 {
        if let Some(s) = self.interest.short.get(&id) {
            return *s;
        }
        self.interest.next_short += 1;
        let s = self.interest.next_short;
        self.interest.short.insert(id, s);
        self.interest.short_rev.insert(s, id);
        s
    }

    /// An object became relevant to `peer`: spawn it there and bring it up to date.
    fn interest_enter(&mut self, peer: u64, id: u128) {
        if self.owner_of_raw(id) == peer {
            // It already has the object; it gets the short id so it can send it upstream.
            self.teach_alias(peer, id);
            return;
        }
        if let Some(rec) = self.world.spawns.iter().find(|s| s.id == id) {
            let m = enc_spawn(rec);
            self.send_reliable(peer, &m);
        }
        let (owner, seq) = self.world.owners.get(&id).copied().unwrap_or((0, 0));
        if self.world.owners.contains_key(&id) {
            let mut m = vec![M_OWNER];
            put_u128(&mut m, id);
            put_u64(&mut m, owner);
            put_u64(&mut m, seq);
            self.send_reliable(peer, &m);
        }
        if let Some(parent) = self.world.parents.get(&id).copied() {
            let mut m = vec![M_PARENT];
            put_u128(&mut m, id);
            put_u128(&mut m, parent);
            self.send_reliable(peer, &m);
        }
        if let Some(on) = self.world.actives.get(&id).copied() {
            let mut m = vec![M_ACTIVE];
            put_u128(&mut m, id);
            m.push(on as u8);
            self.send_reliable(peer, &m);
        }
        if let Some(vars) = self.world.vars.get(&id) {
            let msgs: Vec<Vec<u8>> = vars.iter().map(|(k, v)| enc_var(id, k, v, self.host_peer, 0)).collect();
            for m in msgs {
                self.send_reliable(peer, &m);
            }
        }
        if let Some(x) = self.world.xforms.get(&id).copied() {
            let mut m = vec![M_XFORM];
            enc_entry(&mut m, &SnapEntry { id, pos: Some(x.pos), rot: Some(x.rot), scale: Some(x.scale), vel: None, quantized: false });
            self.send_reliable(peer, &m);
        }
        if let Some(params) = self.world.anims.get(&id) {
            let mut m = vec![M_ANIM];
            put_u128(&mut m, id);
            put_u16(&mut m, params.len() as u16);
            for (k, v) in params {
                put_str(&mut m, k);
                put_f32(&mut m, *v);
            }
            self.send_reliable(peer, &m);
        }
        if let Some(packet) = self.world.avparams.get(&id) {
            let n = packet.len().min(u16::MAX as usize);
            let mut m = vec![M_AVPARAM];
            put_u128(&mut m, id);
            put_u16(&mut m, n as u16);
            m.extend_from_slice(&packet[..n]);
            self.send_reliable(peer, &m);
        }
        self.teach_alias(peer, id);
    }

    /// Tell `peer` the 32-bit alias of `id` (both ways: snapshots come down with it and the
    /// client may send its own objects' snapshots up with it).
    fn teach_alias(&mut self, peer: u64, id: u128) {
        if !self.interest.cfg.short_ids {
            return;
        }
        let s = self.short_of(id);
        let mut m = vec![M_NETID];
        put_u128(&mut m, id);
        put_u32(&mut m, s);
        self.send_reliable(peer, &m);
        if let Some(c) = self.interest.clients.get_mut(&peer) {
            c.short_known.insert(id);
        }
    }

    /// An object left `peer`'s interest: a runtime spawn is despawned there.
    fn interest_leave(&mut self, peer: u64, id: u128) {
        if let Some(c) = self.interest.clients.get_mut(&peer) {
            c.objs.remove(&id);
            c.dirty.retain(|d| *d != id);
            c.short_known.remove(&id);
        }
        if self.owner_of_raw(id) != peer && self.world.spawns.iter().any(|s| s.id == id) {
            let mut d = vec![M_DESPAWN];
            put_u128(&mut d, id);
            self.send_reliable(peer, &d);
        }
    }

    /// Host: forget a client (it left).
    pub(super) fn interest_drop_peer(&mut self, peer: u64) {
        self.interest.clients.remove(&peer);
    }

    /// Host: forget an object (it was despawned).
    pub(super) fn interest_forget(&mut self, id: u128) {
        for c in self.interest.clients.values_mut() {
            c.ci.visible.remove(&id);
            c.ci.full.remove(&id);
            c.objs.remove(&id);
            c.short_known.remove(&id);
        }
        if let Some(s) = self.interest.short.remove(&id) {
            self.interest.short_rev.remove(&s);
        }
    }

    /// Host: a runtime spawn goes to the clients it is relevant to (the
    /// owner always); they then see it until it leaves.
    pub(super) fn broadcast_spawn(&mut self, rec: &SpawnRecord, except: Option<u64>) {
        let msg = enc_spawn(rec);
        if !self.interest_on() {
            self.broadcast_reliable(&msg, except);
            return;
        }
        let radius = self.interest.cfg.radius;
        let always = self.world.always_relevant.contains(&rec.id);
        let peers = self.world_members();
        for p in peers {
            if Some(p) == except {
                continue;
            }
            let center = self.world.player_obj.get(&p).and_then(|o| self.object_pos(*o));
            let near = always || rec.owner == p || center.is_some_and(|c| c.distance(rec.translation) <= radius);
            if !near {
                continue;
            }
            if let Some(c) = self.interest.clients.get_mut(&p) {
                c.ci.visible.insert(rec.id);
            }
            self.send_reliable(p, &msg);
            // Spawned straight into the visible set, so no Enter event will teach the short id.
            self.teach_alias(p, rec.id);
        }
    }

    /// Host: put snapshot entries into the clients' slots. `except` is the
    /// sender (it has them); `skip_predicted` leaves out predicted objects
    /// to their own owner (it reconciles from the host's state instead).
    pub(super) fn interest_route(&mut self, origin: u64, time: f64, entries: &[SnapEntry], except: Option<u64>, skip_predicted: bool) {
        let it = &mut self.interest;
        for (peer, cn) in it.clients.iter_mut() {
            if Some(*peer) == except {
                continue;
            }
            for e in entries {
                let owned = self.world.owners.get(&e.id).is_some_and(|(o, _)| o == peer);
                if skip_predicted && owned && self.world.predicted.contains(&e.id) {
                    continue;
                }
                if !(owned || cn.ci.visible.contains(&e.id)) {
                    continue;
                }
                // The object's own range, when it set one, still applies.
                if let Some(range) = self.world.relevancy.get(&e.id).copied().filter(|r| *r > 0.0)
                    && let (Some(c), Some(p)) = (cn.ci.center, it.pos.get(&e.id))
                    && c.distance(*p) > range
                {
                    continue;
                }
                let slot = cn.objs.entry(e.id).or_default();
                match &mut slot.pending {
                    Some(p) => {
                        p.entry = merge(p.entry, *e);
                        p.origin = origin;
                        p.time = time;
                        cn.ci.throttled += 1;
                    }
                    None => {
                        slot.pending = Some(Pend { entry: *e, origin, time });
                        cn.dirty.push(e.id);
                    }
                }
            }
        }
    }

    /// Host: send what is due and affordable to each client, batched.
    pub(super) fn interest_flush(&mut self, now: f64) {
        let cfg = self.interest.cfg.clone();
        let mut peers = std::mem::take(&mut self.interest.peers);
        peers.clear();
        peers.extend(self.interest.clients.iter().filter(|(_, c)| !c.dirty.is_empty()).map(|(p, _)| *p));
        let mut scored = std::mem::take(&mut self.interest.scored);
        let mut keep = std::mem::take(&mut self.interest.keep);
        let mut batch = std::mem::take(&mut self.interest.batch);
        let mut buf = std::mem::take(&mut self.interest.buf);
        for &peer in &peers {
            let Some(mut cn) = self.interest.clients.remove(&peer) else {
                continue;
            };
            scored.clear();
            keep.clear();
            for id in cn.dirty.drain(..) {
                let Some(slot) = cn.objs.get(&id) else {
                    continue;
                };
                if slot.pending.is_none() {
                    continue;
                }
                let dist = match (cn.ci.center, self.interest.pos.get(&id)) {
                    (Some(c), Some(p)) => c.distance(*p),
                    _ => 0.0,
                };
                let interval = cn.ci.interval(&cfg, id, dist);
                let age = (now - slot.last_sent) as f32;
                if age >= interval {
                    scored.push((age / interval.max(0.02), id));
                } else {
                    keep.push(id);
                }
            }
            scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            batch.clear();
            let mut allow = if cfg.budget == 0 { f32::INFINITY } else { cn.ci.credit };
            for &(_, id) in &scored {
                let slot = cn.objs.get_mut(&id).unwrap();
                let p = slot.pending.unwrap();
                let mut e = p.entry;
                let size = entry_size_b(&e) as f32 + 3.0;
                if size > allow {
                    cn.ci.dropped_budget += 1;
                    keep.push(id);
                    continue;
                }
                slot.pending = None;
                slot.last_sent = now;
                if cfg.delta {
                    match delta_against(&mut slot.baseline, &e, now, &mut slot.baseline_full, cfg.delta_refresh) {
                        Some(d) => e = d,
                        None => {
                            // Nothing this client does not already have: costs no bytes.
                            cn.ci.suppressed += 1;
                            continue;
                        }
                    }
                }
                // Charged at its real (delta) size; the check above used the full one.
                allow -= entry_size_b(&e) as f32 + 3.0;
                if cfg.short_ids && cn.short_known.contains(&id)
                    && let Some(s) = self.interest.short.get(&id)
                {
                    e.id = *s as u128;
                }
                batch.push((p.origin, p.time, e));
            }
            cn.dirty.extend(keep.iter().copied());
            cn.pending = cn.dirty.len() as u32;
            // One run per (origin, time); stable, so each run keeps its priority order.
            batch.sort_by_key(|a| (a.0, a.1.to_bits()));
            let mut sent = 0usize;
            let mut i = 0;
            while i < batch.len() {
                buf.clear();
                buf.push(P_SNAPB);
                buf.push(0);
                let mut groups = 0u8;
                while i < batch.len() && groups < 255 {
                    let (origin, time) = (batch[i].0, batch[i].1);
                    let head = buf.len();
                    if head + 17 + entry_size_b(&batch[i].2) > MTU {
                        break;
                    }
                    put_u64(&mut buf, origin);
                    put_f64(&mut buf, time);
                    buf.push(0);
                    let mut n = 0u8;
                    while i < batch.len() && batch[i].0 == origin && batch[i].1 == time && n < 255 && buf.len() + entry_size_b(&batch[i].2) <= MTU {
                        enc_entry_b(&mut buf, &batch[i].2);
                        n += 1;
                        i += 1;
                    }
                    buf[head + 16] = n;
                    groups += 1;
                    if n == 0 {
                        break;
                    }
                }
                if groups == 0 {
                    i += 1; // an entry that can never fit: skip rather than loop
                    continue;
                }
                buf[1] = groups;
                sent += buf.len();
                self.send_to_peer(peer, &buf);
            }
            if sent > 0 {
                // The sealed size is a little larger; `interest_account` is
                // skipped for this client (it is out of the map), so book it here.
                cn.ci.spend(&cfg, sent + 28, now);
            }
            self.interest.clients.insert(peer, cn);
        }
        self.interest.peers = peers;
        self.interest.scored = scored;
        self.interest.keep = keep;
        self.interest.batch = batch;
        self.interest.buf = buf;
    }

    /// Host: send an unreliable per-object packet to the clients that see it.
    pub(super) fn fanout_obj(&mut self, data: &[u8], except: Option<u64>, id: u128, stream: Stream) {
        if !self.interest_on() {
            self.fanout(data, except);
            return;
        }
        let now = self.now();
        let cfg = self.interest.cfg.clone();
        let mut peers = std::mem::take(&mut self.interest.peers);
        peers.clear();
        peers.extend(self.conns.keys().copied().filter(|p| Some(*p) != except));
        let obj_pos = self.interest.pos.get(&id).copied().or_else(|| self.world.xforms.get(&id).map(|x| x.pos));
        for &p in &peers {
            let owner = self.owner_of_raw(id) == p;
            let Some(cn) = self.interest.clients.get_mut(&p) else {
                continue;
            };
            if !owner && !cn.ci.visible.contains(&id) {
                continue;
            }
            if stream != Stream::State {
                let dist = match (cn.ci.center, obj_pos) {
                    (Some(c), Some(o)) => c.distance(o),
                    _ => 0.0,
                };
                if stream == Stream::Face && dist > cfg.avatar_detail_range.min(cfg.radius) {
                    continue;
                }
                let slot = cn.objs.entry(id).or_default();
                let interval = if cn.ci.full.contains(&id) { cfg.full_interval() } else { cfg.interval(dist) };
                // 15 % slack: the host runs on a tick, so a sender at exactly the tier's rate would
                // otherwise lose every other packet to a hair of rounding.
                if now - slot.last_pose < interval as f64 * 0.85 {
                    cn.ci.throttled += 1;
                    continue;
                }
                if !cn.ci.affords(&cfg, data.len() + 28) {
                    cn.ci.dropped_budget += 1;
                    continue;
                }
                slot.last_pose = now;
            }
            self.send_to_peer(p, data);
        }
        self.interest.peers = peers;
    }

    /// Host: relay voice to the listeners within earshot of the speaker.
    pub(super) fn fanout_voice(&mut self, data: &[u8], except: Option<u64>, from: u64) {
        if !self.interest_on() {
            self.fanout(data, except);
            return;
        }
        let range = self.interest.cfg.voice_range;
        let speaker = self.world.player_obj.get(&from).and_then(|o| self.object_pos(*o));
        let mut peers = std::mem::take(&mut self.interest.peers);
        peers.clear();
        peers.extend(self.conns.keys().copied().filter(|p| Some(*p) != except));
        for &p in &peers {
            // Unplaced speaker or listener: hear it (nothing to measure).
            let center = self.interest.clients.get(&p).and_then(|c| c.ci.center);
            if let (Some(a), Some(b)) = (speaker, center)
                && a.distance(b) > range
            {
                continue;
            }
            self.send_to_peer(p, data);
        }
        self.interest.peers = peers;
    }

    /// Host: should a reliable object message reach `peer`? (`msg` starts with
    /// the tag and the 128-bit object id.)
    pub(super) fn interest_wants(&self, peer: u64, msg: &[u8]) -> bool {
        if msg.len() < 17 || !matches!(msg[0], M_OWNER | M_DESPAWN | M_VAR | M_ACTIVE | M_PARENT | M_ANIM | M_AVPARAM | M_XFORM) {
            return true;
        }
        let mut id = [0u8; 16];
        id.copy_from_slice(&msg[1..17]);
        self.sees(peer, u128::from_le_bytes(id))
    }

    /// Client: a 32-bit alias from the host.
    pub(super) fn on_netid(&mut self, c: &mut Cursor) {
        let id = c.u128();
        let short = c.u32();
        if !c.bad && short != 0 {
            self.interest.short_in.insert(short, id);
            self.interest.short_out.insert(id, short);
        }
    }

    /// Host: turn the aliases in a client's upstream snapshot back into ids. Only an alias the
    /// host taught that very client counts; anything else is dropped (returns how many).
    pub(super) fn resolve_upstream(&self, peer: u64, entries: &mut Vec<SnapEntry>) -> usize {
        let before = entries.len();
        let known = self.interest.clients.get(&peer).map(|c| &c.short_known);
        entries.retain_mut(|e| {
            if e.id > u32::MAX as u128 {
                return true;
            }
            match self.interest.short_rev.get(&(e.id as u32)) {
                Some(id) if known.is_some_and(|k| k.contains(id)) => {
                    e.id = *id;
                    true
                }
                _ => false,
            }
        });
        before - entries.len()
    }
}

/// Turn the aliases in `entries` back into network ids; drop unknown ones.
pub(super) fn resolve_ids(short_in: &HashMap<u32, u128>, entries: &mut Vec<SnapEntry>) {
    entries.retain_mut(|e| {
        if e.id > u32::MAX as u128 {
            return true;
        }
        match short_in.get(&(e.id as u32)) {
            Some(id) => {
                e.id = *id;
                true
            }
            None => false,
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    const TICK: f64 = 1.0 / 30.0;

    fn base_cfg(name: &str, interest: InterestConfig) -> NetConfig {
        let v = NetValidation { enabled: false, ..Default::default() };
        NetConfig {
            player_name: name.to_string(),
            discovery: false,
            secret: String::new(),
            relay: String::new(),
            account_name: String::new(),
            account_password: String::new(),
            accounts: Vec::new(),
            identity: Identity::generate(),
            validation: v,
            interest,
            max_players: 400,
            timeout: 30.0,
            ..Default::default()
        }
    }

    fn icfg(on: bool) -> InterestConfig {
        InterestConfig { enabled: on, radius: 100.0, hysteresis: 10.0, cell_size: 40.0, refresh_hz: 10.0, ..Default::default() }
    }

    struct World {
        hub: MemHub,
        host: NetSession,
        clients: Vec<NetSession>,
        /// Each client's player object and position.
        players: Vec<(u128, Vec3)>,
        tick: u64,
    }

    impl World {
        fn new(n: usize, interest: InterestConfig, at: impl Fn(usize) -> Vec3) -> Self {
            let hub = MemHub::new();
            let mut host = NetSession::host_on(Box::new(hub.endpoint()), base_cfg("Host", interest));
            host.use_manual_clock();
            let addr = host.local_addr().unwrap();
            let mut clients: Vec<NetSession> = (0..n)
                .map(|i| {
                    let mut c = NetSession::join_on(Box::new(hub.endpoint()), addr, base_cfg(&format!("P{i}"), InterestConfig::default()));
                    c.use_manual_clock();
                    c
                })
                .collect();
            let mut w = Self { hub, host, clients: Vec::new(), players: Vec::new(), tick: 0 };
            for _ in 0..(60 + n * 2) {
                w.hub.advance(TICK);
                w.host.advance(TICK);
                w.host.pump();
                for c in clients.iter_mut() {
                    c.advance(TICK);
                    c.pump();
                }
            }
            assert!(clients.iter().all(|c| c.is_connected()), "all clients joined");
            for (i, c) in clients.iter().enumerate() {
                let pos = at(i);
                let id = w.host.spawn_everywhere(NetSpawnSource::Prefab("prefabs/player.prefab".into()), pos, Quat::IDENTITY, c.local_peer());
                w.host.world.player_obj.insert(c.local_peer(), id.raw());
                w.players.push((id.raw(), pos));
            }
            w.clients = clients;
            w
        }

        /// One tick: everyone pumps, every client reports its player's position.
        fn step(&mut self, mut host_extra: impl FnMut(&mut NetSession, f64)) {
            self.tick += 1;
            self.hub.advance(TICK);
            self.host.advance(TICK);
            self.host.pump();
            let now = self.host.now();
            host_extra(&mut self.host, now);
            for (i, c) in self.clients.iter_mut().enumerate() {
                c.advance(TICK);
                c.pump();
                if self.tick.is_multiple_of(2) {
                    let (id, pos) = self.players[i];
                    c.send_snapshot(&[SnapEntry { id, pos: Some(pos), rot: Some(Quat::IDENTITY), quantized: true, ..Default::default() }]);
                }
            }
        }

        fn run(&mut self, secs: f64) {
            for _ in 0..(secs / TICK) as usize {
                self.step(|_, _| {});
            }
        }
    }

    fn line(i: usize) -> Vec3 {
        // 0 and 1 are 10 m apart; 2 is far away.
        [Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0), Vec3::new(900.0, 0.0, 0.0)][i % 3]
    }

    #[test]
    fn only_the_near_players_are_replicated_and_spawned() {
        let mut w = World::new(3, icfg(true), line);
        w.run(2.0);
        let (a, b, c) = (w.players[0].0, w.players[1].0, w.players[2].0);
        let ids = |s: &mut NetSession| -> Vec<u128> {
            s.view().world_ops.iter().filter_map(|op| if let NetWorldOp::Spawn { root_id, .. } = op { Some(root_id.raw()) } else { None }).collect()
        };
        let a_spawns = ids(&mut w.clients[0]);
        assert!(a_spawns.contains(&b) && !a_spawns.contains(&c), "A is told about B, not C");
        assert!(w.clients[0].remote_transform(ObjectId::from_raw(b)).is_some(), "B's movement reaches A");
        assert!(w.clients[0].remote_transform(ObjectId::from_raw(c)).is_none());
        assert!(w.clients[2].remote_transform(ObjectId::from_raw(a)).is_none(), "C hears nothing about A");
        assert!(w.clients[2].world.spawns.iter().all(|s| s.id == c), "C only has its own spawn");
        // Metrics: A has itself and B, C only itself.
        let st = w.host.interest_stats();
        assert_eq!(st.len(), 3);
        assert!(st[0].relevant >= 2 && st[2].relevant <= 1, "{st:?}");
        assert!(w.host.stats.interest_relevant_avg < 3.0);

        // C walks to A: it enters (spawn + state), and leaves again past the hysteresis (despawn).
        w.players[2].1 = Vec3::new(40.0, 0.0, 0.0);
        w.run(1.5);
        assert!(w.clients[0].world.spawns.iter().any(|s| s.id == c), "C spawned on A once near");
        assert!(w.clients[0].remote_transform(ObjectId::from_raw(c)).is_some());
        w.players[2].1 = Vec3::new(105.0, 0.0, 0.0);
        w.run(1.0);
        assert!(w.clients[0].world.spawns.iter().any(|s| s.id == c), "inside the hysteresis band it stays");
        w.players[2].1 = Vec3::new(400.0, 0.0, 0.0);
        w.run(1.0);
        assert!(w.clients[0].world.spawns.iter().all(|s| s.id != c), "far again: despawned on A");
        assert!(w.clients[0].view().world_ops.iter().any(|op| matches!(op, NetWorldOp::Despawn(id) if id.raw() == c)));
    }

    #[test]
    fn interest_off_keeps_the_old_behaviour() {
        let mut w = World::new(3, icfg(false), line);
        w.run(2.0);
        let c = w.players[2].0;
        assert!(w.clients[0].remote_transform(ObjectId::from_raw(c)).is_some(), "everyone gets everything");
        assert_eq!(w.clients[0].world.spawns.len(), 3);
        assert!(w.host.interest_stats().is_empty());
    }

    #[test]
    fn voice_and_avatar_data_follow_range_rules() {
        let mut w = World::new(3, icfg(true), line);
        w.run(1.0);
        let a = w.players[0].0;
        let poses = {
            let mut p = [None; TRACKER_SLOTS];
            p[0] = Some((Vec3::new(0.0, 1.6, 0.0), Quat::IDENTITY));
            p
        };
        for _ in 0..10 {
            w.clients[0].send_avatar(a, &poses);
            w.clients[0].send_voice(1, &[1, 2, 3]);
            w.run(0.1);
        }
        assert!(w.clients[1].avatar_in.contains_key(&a), "B sees A's pose");
        assert!(!w.clients[2].avatar_in.contains_key(&a), "C does not");
        assert!(!w.clients[1].take_voice().is_empty(), "B hears A");
        assert!(w.clients[2].take_voice().is_empty(), "C is out of earshot");
        // Close to A but outside the voice range (50 m): sees the avatar, does not hear it.
        w.players[2].1 = Vec3::new(80.0, 0.0, 0.0);
        w.run(1.0);
        for _ in 0..5 {
            w.clients[0].send_avatar(a, &poses);
            w.clients[0].send_voice(2, &[4]);
            w.run(0.1);
        }
        assert!(w.clients[2].avatar_in.contains_key(&a), "C now sees A's pose");
        assert!(w.clients[2].take_voice().is_empty(), "80 m is beyond the 50 m voice range");
    }

    #[test]
    fn budget_is_respected_and_nothing_starves() {
        // 150 host-simulated objects all around one client, a budget far under what they need.
        let budget = 6_000;
        let cfg = InterestConfig { budget, tiers: vec![], nearest_full: 0, ..icfg(true) };
        let mut w = World::new(1, cfg, |_| Vec3::ZERO);
        let ids: Vec<ObjectId> = (0..150).map(|_| ObjectId::new()).collect();
        for (k, id) in ids.iter().enumerate() {
            w.host.spawn(NetSpawnSource::Prefab("prefabs/npc.prefab".into()), *id, Vec3::new((k % 15) as f32 * 5.0, 0.0, (k / 15) as f32 * 5.0), Quat::IDENTITY, 0);
        }
        let mut t = 0.0f32;
        
        let mut seen_at: HashMap<u128, u32> = HashMap::new();
        // Settle (spawns, baselines), then measure.
        for _ in 0..90 {
            w.step(|h, _| {
                t += 0.033;
                let entries: Vec<SnapEntry> = ids
                    .iter()
                    .enumerate()
                    .map(|(k, id)| SnapEntry { id: id.raw(), pos: Some(Vec3::new((k % 15) as f32 * 5.0 + t, 0.0, (k / 15) as f32 * 5.0)), quantized: true, ..Default::default() })
                    .collect();
                h.send_snapshot(&entries);
            });
        }
        let bytes0 = w.host.interest_stats()[0].bytes_total;
        let secs = 10.0;
        for _ in 0..(secs / TICK) as usize {
            w.step(|h, _| {
                t += 0.033;
                let entries: Vec<SnapEntry> = ids
                    .iter()
                    .enumerate()
                    .map(|(k, id)| SnapEntry { id: id.raw(), pos: Some(Vec3::new((k % 15) as f32 * 5.0 + t, 0.0, (k / 15) as f32 * 5.0)), quantized: true, ..Default::default() })
                    .collect();
                h.send_snapshot(&entries);
            });
            for id in &ids {
                if let Some(x) = w.clients[0].remote_transform(*id) {
                    seen_at.insert(id.raw(), (x.translation.x * 100.0) as u32);
                }
            }
        }
        let st = w.host.interest_stats()[0];
        let rate = (st.bytes_total - bytes0) as f32 / secs as f32;
        assert!(st.dropped_budget > 0, "the budget bit: {st:?}");
        assert!(rate <= budget as f32 * 1.35, "rate {rate} B/s over budget {budget}");
        // Starvation-free: every object kept being updated (its x advanced past the start).
        let moved = ids.iter().filter(|id| w.clients[0].remote_transform(**id).is_some_and(|x| x.translation.x > (ids.iter().position(|i| i == *id).unwrap() % 15) as f32 * 5.0 + t - 8.0)).count();
        assert_eq!(moved, ids.len(), "every object has a recent update ({} of {})", moved, ids.len());
    }

    /// NPC entries sent to one client with delta baselines on and off: the same updates arrive
    /// (a mover keeps moving), but a still object costs (nearly) nothing with baselines.
    #[test]
    fn delta_baselines_skip_what_the_client_already_has() {
        let run = |delta: bool| {
            let cfg = InterestConfig { delta, tiers: vec![], nearest_full: 0, ..icfg(true) };
            let mut w = World::new(1, cfg, |_| Vec3::ZERO);
            let still = ObjectId::new();
            let mover = ObjectId::new();
            for id in [still, mover] {
                w.host.spawn(NetSpawnSource::Prefab("prefabs/npc.prefab".into()), id, Vec3::new(5.0, 0.0, 0.0), Quat::IDENTITY, 0);
            }
            let mut t = 0.0f32;
            let mut feed = |w: &mut World, secs: f64| {
                for _ in 0..(secs / TICK) as usize {
                    w.step(|h, _| {
                        t += 0.033;
                        h.send_snapshot(&[
                            SnapEntry { id: still.raw(), pos: Some(Vec3::new(5.0, 0.0, 0.0)), rot: Some(Quat::IDENTITY), quantized: true, ..Default::default() },
                            SnapEntry { id: mover.raw(), pos: Some(Vec3::new(5.0 + t, 0.0, 0.0)), quantized: true, ..Default::default() },
                        ]);
                    });
                }
            };
            feed(&mut w, 1.0);
            let before = w.host.interest_stats()[0];
            feed(&mut w, 4.0);
            let after = w.host.interest_stats()[0];
            let moved = w.clients[0].remote_transform(mover).map(|x| x.translation.x).unwrap_or(0.0);
            (after.bytes_total - before.bytes_total, after.suppressed - before.suppressed, moved, w.clients[0].remote_transform(still).map(|x| x.translation))
        };
        let (bytes_off, suppressed_off, moved_off, _) = run(false);
        let (bytes_on, suppressed_on, moved_on, still_on) = run(true);
        assert_eq!(suppressed_off, 0);
        assert!(suppressed_on > 50, "the still object's repeats were skipped: {suppressed_on}");
        assert!(bytes_on < bytes_off * 9 / 10, "{bytes_on} B with baselines vs {bytes_off} B without");
        assert!(moved_on > 5.0 + 3.0 && (moved_on - moved_off).abs() < 0.5, "the mover keeps arriving: {moved_on} vs {moved_off}");
        assert!(still_on.is_some_and(|p| (p - Vec3::new(5.0, 0.0, 0.0)).length() < 0.01), "the still object stays where it is");
    }

    #[test]
    fn delta_against_sends_only_changed_fields_and_refreshes() {
        let e = |x: f32, rot: bool| SnapEntry { id: 9, pos: Some(Vec3::new(x, 0.0, 0.0)), rot: rot.then_some(Quat::IDENTITY), quantized: true, ..Default::default() };
        let mut base = None;
        let mut last_full = 0.0;
        let d = |base: &mut Option<SnapEntry>, last: &mut f64, e: &SnapEntry, now: f64| delta_against(base, e, now, last, 1.0);
        let first = d(&mut base, &mut last_full, &e(1.0, true), 0.0).unwrap();
        assert!(first.pos.is_some() && first.rot.is_some(), "the first entry is whole");
        assert!(d(&mut base, &mut last_full, &e(1.0, true), 0.1).is_none(), "nothing changed");
        let moved = d(&mut base, &mut last_full, &e(2.0, true), 0.2).unwrap();
        assert!(moved.pos.is_some() && moved.rot.is_none(), "only the position changed: {moved:?}");
        assert!(d(&mut base, &mut last_full, &e(2.0, false), 0.3).is_none());
        let refresh = d(&mut base, &mut last_full, &e(2.0, true), 1.2).unwrap();
        assert!(refresh.pos.is_some() && refresh.rot.is_some(), "a full entry after the refresh time");
        // Sub-millimetre jitter on a quantised entry is not a change.
        assert!(d(&mut base, &mut last_full, &e(2.0004, true), 1.3).is_none());
    }

    #[test]
    fn upstream_ids_are_aliases_only_for_objects_the_client_was_taught() {
        let mut w = World::new(3, icfg(true), line);
        w.run(2.0);
        let (a, c) = (w.players[0].0, w.players[2].0);
        // A learned its own alias and sends 32-bit ids; it is smaller on the wire.
        eprintln!("a={a} owner={} peers {:?} c1 in {:?} known0 {:?} cfg {:?}", w.host.owner_of_raw(a), w.clients.iter().map(|c| c.local_peer()).collect::<Vec<_>>(), w.clients[1].interest.short_in, w.host.interest.clients.get(&w.clients[0].local_peer()).map(|c| c.short_known.clone()), w.host.interest.cfg.short_ids);
        eprintln!("host short {:?}; client0 in {:?} visible {:?}", w.host.interest.short, w.clients[0].interest.short_in, w.host.interest.clients.get(&w.clients[0].local_peer()).map(|c| c.ci.visible.clone()));
        let alias = *w.clients[0].interest.short_out.get(&a).expect("A was taught an alias for its own object");
        let entry = SnapEntry { id: a, pos: Some(Vec3::ZERO), rot: Some(Quat::IDENTITY), quantized: true, ..Default::default() };
        let full = build_snap_packets(1, 0.0, &[entry]);
        let short = build_upstream_packets(1, 0.0, &[SnapEntry { id: alias as u128, ..entry }]);
        assert!(short[0].len() + 8 <= full[0].len(), "{} vs {} bytes", short[0].len(), full[0].len());
        // The host resolves the alias for A, but not for a client that was never taught it.
        let peer_a = w.clients[0].local_peer();
        let alias_c = *w.host.interest.short.get(&c).expect("C has an alias");
        let ask = |w: &World, peer: u64, id: u32| {
            let mut v = vec![SnapEntry { id: id as u128, pos: Some(Vec3::ZERO), ..Default::default() }];
            w.host.resolve_upstream(peer, &mut v);
            v.first().map(|e| e.id)
        };
        assert_eq!(ask(&w, peer_a, alias), Some(a));
        assert_eq!(ask(&w, peer_a, alias_c), None, "A was never taught C's alias");
        assert_eq!(ask(&w, w.clients[2].local_peer(), alias_c), Some(c));
        assert_eq!(ask(&w, peer_a, 4_000_000), None, "unknown alias");
        // Aliases die with the world: after a reset nothing resolves any more.
        w.host.interest.reset();
        w.clients[0].interest.reset();
        assert_eq!(ask(&w, peer_a, alias), None);
        assert!(w.clients[0].interest.short_out.is_empty() && w.clients[0].interest.short_in.is_empty());
    }

    /// 200 clients, a moving crowd (and server-simulated NPCs), interest on and
    /// off: server CPU per tick and bandwidth per client. Numbers go to stdout.
    #[test]
    #[ignore = "scale test (about a minute); run with --ignored --nocapture"]
    fn scale_200_clients() {
        let n = 200;
        let wide = 1200.0f32;
        let mut rng = 0x2545F4914F6CDD1Du64;
        let mut rnd = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 40) as f32 / (1u64 << 24) as f32
        };
        let start: Vec<Vec3> = (0..n).map(|_| Vec3::new(rnd(), 0.0, rnd())).collect();
        let dirs: Vec<f32> = (0..n).map(|_| rnd() * std::f32::consts::TAU).collect();
        println!("scale test: {n} clients, 100 server NPCs, {} Hz", (1.0 / TICK).round());
        for (label, cfg, area) in [
            ("1200 m: interest off", icfg(false), wide),
            ("1200 m: interest on, 100 m", icfg(true), wide),
            ("300 m (dense): interest off", icfg(false), 300.0),
            ("300 m (dense): interest on", icfg(true), 300.0),
            ("300 m (dense): on + 12 kB/s", InterestConfig { budget: 12_000, ..icfg(true) }, 300.0),
        ] {
            let mut w = World::new(n, cfg, |i| start[i] * area);
            let mut dir = dirs.clone();
            let npc: Vec<ObjectId> = (0..100).map(|_| ObjectId::new()).collect();
            let npc_pos: Vec<Vec3> = (0..100).map(|_| Vec3::new(rnd() * area, 0.0, rnd() * area)).collect();
            for (id, p) in npc.iter().zip(&npc_pos) {
                w.host.spawn(NetSpawnSource::Prefab("prefabs/npc.prefab".into()), *id, *p, Quat::IDENTITY, 0);
            }
            // Only the server's part of each tick is timed.
            let secs = 20.0;
            let warm = (6.0 / TICK) as usize;
            let ticks = (secs / TICK) as usize;
            let mut cpu = Vec::with_capacity(ticks);
            let (mut b0, mut p0) = (0, 0);
            for k in 0..warm + ticks {
                if k == warm {
                    (b0, p0) = (w.host.stats.bytes_out, w.host.stats.packets_out);
                }
                for (i, d) in dir.iter_mut().enumerate() {
                    if rnd() < 0.05 {
                        *d += (rnd() - 0.5) * 2.0;
                    }
                    let p = &mut w.players[i].1;
                    *p += Vec3::new(d.cos(), 0.0, d.sin()) * (4.0 * TICK as f32);
                    p.x = p.x.clamp(0.0, area);
                    p.z = p.z.clamp(0.0, area);
                }
                w.tick += 1;
                w.hub.advance(TICK);
                let t0 = Instant::now();
                w.host.advance(TICK);
                w.host.pump();
                let tick = w.tick;
                let entries: Vec<SnapEntry> = npc
                    .iter()
                    .zip(&npc_pos)
                    .enumerate()
                    .map(|(k, (id, p))| SnapEntry {
                        id: id.raw(),
                        pos: Some(*p + Vec3::new((tick as f32 * 0.1 + k as f32).sin() * 5.0, 0.0, 0.0)),
                        quantized: true,
                        ..Default::default()
                    })
                    .collect();
                w.host.send_snapshot(&entries);
                if k >= warm {
                    cpu.push(t0.elapsed().as_secs_f64() * 1000.0);
                }
                for (i, c) in w.clients.iter_mut().enumerate() {
                    c.advance(TICK);
                    c.pump();
                    if w.tick.is_multiple_of(2) {
                        let (id, pos) = w.players[i];
                        c.send_snapshot(&[SnapEntry { id, pos: Some(pos), rot: Some(Quat::IDENTITY), quantized: true, ..Default::default() }]);
                    }
                }
            }
            let bytes = w.host.stats.bytes_out - b0;
            let packets = w.host.stats.packets_out - p0;
            cpu.sort_by(|a, b| a.total_cmp(b));
            let mean = cpu.iter().sum::<f64>() / cpu.len() as f64;
            let p99 = cpu[(cpu.len() as f64 * 0.99) as usize];
            let rel = w.host.stats.interest_relevant_avg;
            let st = w.host.interest_stats();
            let held: u64 = st.iter().map(|c| c.dropped_budget).sum();
            println!(
                "{label:<30} server CPU/tick mean {mean:.2} ms p99 {p99:.2} ms | out {:.1} kB/s per client ({:.1} MB/s total, {:.0} packets/s per client) | relevant/client {}{}",
                bytes as f64 / secs / n as f64 / 1024.0,
                bytes as f64 / secs / 1e6,
                packets as f64 / secs / n as f64,
                if rel > 0.0 { format!("{rel:.0}") } else { "all".into() },
                if held > 0 { format!(" | held by budget {held}") } else { String::new() },
            );
        }
    }
}
