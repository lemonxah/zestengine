//! Server-side validation of client-originated data (NET-4).
//!
//! The session (`net.rs`) calls these checks where it ingests what a client
//! sent; this module owns the rules and the per-peer memory they need, and is
//! free of sockets, so every rule is unit-tested here.
//!
//! - **Rate limits**: one token bucket per peer and traffic kind (`Limit`),
//!   plus per-RPC overrides. The all-packets bucket is taken first, so a
//!   flood costs one bucket update per packet and nothing else.
//! - **Movement**: finite numbers, world bounds, a speed budget with a
//!   teleport tolerance and an acceleration cap per object kind; a rejected
//!   move yields the last accepted position for a correction message.
//! - **VR poses**: head near the avatar root, hands within arm reach of the
//!   head, controller speeds, all relative to the root (so a teleport of the
//!   whole rig is not a violation).
//! - **Ownership**: only objects flagged Takeover / Grabbable, free (or whose
//!   owner went silent), within reach of the requester's avatar, with a
//!   per-object cooldown; never server-owned, protected or unknown objects.
//! - **RPCs, spawns, sizes**: permissions per RPC name, the spawn allow-list,
//!   path sanity, string and argument limits.
//! - **Strikes**: every rejection adds points to the offender's player id.
//!   Score thresholds warn, drop, kick; repeated kicks ban by identity key.
//!   Bans persist in `bans.ron`.
//!
//! The gameplay rules (everything but sizes, finite checks and rate limits)
//! apply on `Topology::Dedicated`, or on a listen host that sets
//! `strict_on_listen_host`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};
use zestengine_core::{NetValidation, OwnershipMode, RpcPermission, RpcRule, TRACKER_SLOTS, Topology};

/// Traffic kinds with their own per-peer token bucket.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Limit {
    Packets,
    Messages,
    Rpc,
    Spawn,
    Ownership,
    Chat,
    Voice,
    AvParam,
    Avatar,
    Snapshot,
    Anim,
    Face,
    Var,
    Input,
}

const LIMITS: usize = 14;

impl Limit {
    pub fn label(self) -> &'static str {
        match self {
            Self::Packets => "packet",
            Self::Messages => "message",
            Self::Rpc => "rpc",
            Self::Spawn => "spawn",
            Self::Ownership => "ownership request",
            Self::Chat => "chat",
            Self::Voice => "voice",
            Self::AvParam => "avatar parameter",
            Self::Avatar => "avatar pose",
            Self::Snapshot => "snapshot",
            Self::Anim => "animator",
            Self::Face => "face",
            Self::Var => "variable write",
            Self::Input => "input",
        }
    }
}

/// Why a client's data was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Reason {
    /// A token bucket ran dry.
    RateLimited,
    /// NaN or infinity where a number was expected.
    NotFinite,
    /// A position outside the world bounds, or an absurd scale.
    OutOfBounds,
    /// A move beyond the speed budget (teleport).
    Teleport,
    /// Speed or acceleration over the cap.
    Speed,
    /// A head / hand pose a body cannot make.
    ImpossiblePose,
    /// An ownership request the rules refuse.
    OwnershipDenied,
    /// An RPC the sender may not call, or whose arguments are wrong.
    RpcDenied,
    /// A spawn the sender may not make.
    SpawnDenied,
    /// A field, message or queue over its limit.
    TooLarge,
    /// A packet that does not parse, or a message of the wrong kind.
    Malformed,
    /// A sender id or authority that is not the sender's own.
    Forged,
    /// A timestamp from the future, or older than the rewind cap.
    BadTime,
    /// The anti-cheat provider rejected the client (SEC-2).
    AntiCheat,
}

pub const REASONS: usize = 14;

impl Reason {
    pub const ALL: [Reason; REASONS] = [
        Self::RateLimited,
        Self::NotFinite,
        Self::OutOfBounds,
        Self::Teleport,
        Self::Speed,
        Self::ImpossiblePose,
        Self::OwnershipDenied,
        Self::RpcDenied,
        Self::SpawnDenied,
        Self::TooLarge,
        Self::Malformed,
        Self::Forged,
        Self::BadTime,
        Self::AntiCheat,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::RateLimited => "rate limit exceeded",
            Self::NotFinite => "NaN or infinite value",
            Self::OutOfBounds => "outside the world bounds",
            Self::Teleport => "teleport",
            Self::Speed => "speed or acceleration over the cap",
            Self::ImpossiblePose => "impossible pose",
            Self::OwnershipDenied => "ownership request refused",
            Self::RpcDenied => "RPC not allowed",
            Self::SpawnDenied => "spawn not allowed",
            Self::TooLarge => "over a size limit",
            Self::Malformed => "malformed data",
            Self::Forged => "forged sender or authority",
            Self::BadTime => "bad timestamp",
            Self::AntiCheat => "anti-cheat rejected the client",
        }
    }

    /// Strike points one rejection adds.
    pub fn weight(self) -> f32 {
        match self {
            Self::RateLimited => 0.5,
            Self::NotFinite => 10.0,
            Self::OutOfBounds => 5.0,
            Self::Teleport => 5.0,
            Self::Speed => 3.0,
            Self::ImpossiblePose => 4.0,
            Self::OwnershipDenied => 2.0,
            Self::RpcDenied => 5.0,
            Self::SpawnDenied => 5.0,
            Self::TooLarge => 5.0,
            Self::Malformed => 3.0,
            Self::Forged => 20.0,
            Self::BadTime => 3.0,
            Self::AntiCheat => 100.0,
        }
    }
}

/// What the strike score asks the session to do after a rejection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Escalation {
    /// Nothing beyond dropping the data.
    None,
    /// Tell the peer, once, that it is being watched.
    Warn,
    /// Remove the peer with a reason.
    Kick,
    /// Remove the peer and refuse its player id.
    Ban,
}

/// A token bucket.
#[derive(Clone, Copy, Debug)]
struct TokenBucket {
    tokens: f32,
    last: f64,
    rate: f32,
    burst: f32,
}

impl TokenBucket {
    fn new(rate: f32, burst: f32) -> Self {
        Self { tokens: burst, last: f64::NEG_INFINITY, rate, burst }
    }

    fn take(&mut self, now: f64) -> bool {
        if self.last.is_finite() {
            self.tokens = (self.tokens + (now - self.last).max(0.0) as f32 * self.rate).min(self.burst);
        }
        self.last = now;
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

/// What kind of owned object a move belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Prop,
    Avatar,
}

/// Why a snapshot entry was refused, and where the owner should be put back.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rejection {
    pub reason: Reason,
    /// The last accepted position, when a correction should be sent.
    pub correct_to: Option<Vec3>,
}

/// One owned object's movement memory.
#[derive(Clone, Copy, Debug)]
struct Track {
    owner: u64,
    pos: Vec3,
    vel: Vec3,
    t: f64,
    /// Metres the object may be ahead of its speed budget.
    slack: f32,
    last_correct: f64,
}

/// One avatar's pose memory (relative to its root).
#[derive(Clone, Copy, Debug)]
struct PoseTrack {
    owner: u64,
    rel: [Option<Vec3>; TRACKER_SLOTS],
    slack: [f32; TRACKER_SLOTS],
    t: f64,
    /// The root when the poses were taken, and until when a root jump (a
    /// teleport the server has not caught up with) excuses the head-radius
    /// and speed checks.
    root: Vec3,
    grace_until: f64,
}

/// What a snapshot / pose timestamp check says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stamp {
    Fresh,
    /// Older than or equal to one already applied (UDP reordering or a replay):
    /// dropped without a strike.
    Stale,
    /// Not a number, or absurdly far ahead.
    Bad,
}

/// The ownership decision for one request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Claim {
    Grant,
    /// Request mode with an owner present: forward the request to it.
    Ask,
    Deny(DenyWhy),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenyWhy {
    /// Within the per-object cooldown.
    Cooldown,
    /// Host-only object, a protected object (player, predicted) or an unknown one.
    NotClaimable,
    /// Owned by the server.
    ServerOwned,
    /// Owned by another peer that is still there.
    OwnedByOther,
    OutOfRange,
    /// The requester has no avatar to measure from.
    NoAvatar,
}

/// The facts the claim rules need (gathered by the session).
#[derive(Clone, Copy, Debug)]
pub struct ClaimCtx {
    pub id: u128,
    pub by: u64,
    pub cur: u64,
    pub mode: OwnershipMode,
    pub host_peer: u64,
    /// The object is known to the host (a Network Object in the scene, or a spawn).
    pub known: bool,
    /// A player object or a predicted controller.
    pub protected: bool,
    pub owner_silent: bool,
    pub requester: Option<Vec3>,
    pub object: Option<Vec3>,
}

/// An RPC verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RpcVerdict {
    Allow,
    Deny(Reason),
}

/// Everything a spawn check needs.
#[derive(Clone, Copy, Debug)]
pub struct SpawnReq<'a> {
    /// The project-relative prefab path, or `None` for a clone of a scene object.
    pub prefab: Option<&'a str>,
    pub pos: Vec3,
    pub rot: Quat,
    /// Objects this peer has spawned and still has.
    pub live: usize,
    /// Of those, how many came from this same prefab path (0 for a clone).
    pub live_prefab: usize,
    /// Live client spawns of this prefab across every peer.
    pub live_prefab_total: usize,
    pub requester: Option<Vec3>,
}

struct PeerState {
    key: [u8; 32],
    name: String,
    buckets: [TokenBucket; LIMITS],
    /// Buckets of RPCs with their own limits, by rule index.
    rpc_buckets: Vec<(usize, TokenBucket)>,
    counts: [u32; REASONS],
    last_log: [f64; REASONS],
    suppressed: [u32; REASONS],
    warned: bool,
    /// Per-object time of the last ownership request.
    claim_last: HashMap<u128, f64>,
    /// Variable keys this peer has written and the bytes of each (its variable budget).
    vars: HashMap<String, usize>,
}

/// A player id's record across connections.
#[derive(Clone, Copy, Debug, Default)]
struct Offender {
    score: f32,
    last: f64,
    kicks: u32,
    last_kick: f64,
}

/// One refused player id.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct BanEntry {
    /// The player id fingerprint (see `net_identity::fingerprint`).
    pub fingerprint: String,
    /// Unix seconds the ban ends; 0 = permanent.
    pub until: u64,
    /// Unix seconds it was made.
    pub at: u64,
    pub reason: String,
}


/// Banned player ids, optionally mirrored to a RON file.
#[derive(Debug, Default)]
pub struct BanList {
    entries: Vec<BanEntry>,
    path: Option<PathBuf>,
}

#[derive(Serialize, Deserialize, Default)]
#[serde(default)]
struct BanFile {
    bans: Vec<BanEntry>,
}

pub fn unix_now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

impl BanList {
    /// Load `path` (missing = empty) and keep writing changes to it.
    pub fn open(path: &Path) -> Self {
        let entries = match std::fs::read_to_string(path) {
            Ok(text) => match ron::from_str::<BanFile>(&text) {
                Ok(f) => f.bans,
                Err(e) => {
                    tracing::warn!("net: ban list {} unreadable ({e}); starting empty", path.display());
                    Vec::new()
                }
            },
            Err(_) => Vec::new(),
        };
        Self { entries, path: Some(path.to_path_buf()) }
    }

    pub fn entries(&self) -> &[BanEntry] {
        &self.entries
    }

    pub fn is_banned(&self, fingerprint: &str, now_unix: u64) -> Option<&BanEntry> {
        self.entries
            .iter()
            .find(|b| (b.until == 0 || b.until > now_unix) && crate::net_identity::fingerprint_matches(&b.fingerprint, fingerprint))
    }

    /// Ban `fingerprint` for `secs` (0 = permanent), replacing an older entry.
    pub fn add(&mut self, fingerprint: &str, secs: u64, reason: &str, now_unix: u64) {
        self.entries.retain(|b| !crate::net_identity::fingerprint_matches(&b.fingerprint, fingerprint));
        let until = if secs == 0 { 0 } else { now_unix + secs };
        self.entries.push(BanEntry { fingerprint: fingerprint.to_owned(), until, at: now_unix, reason: reason.to_owned() });
        self.save();
    }

    pub fn remove(&mut self, fingerprint: &str) -> bool {
        let before = self.entries.len();
        self.entries.retain(|b| !crate::net_identity::fingerprint_matches(&b.fingerprint, fingerprint));
        let changed = self.entries.len() != before;
        if changed {
            self.save();
        }
        changed
    }

    /// Forget bans that ended.
    pub fn prune(&mut self, now_unix: u64) {
        let before = self.entries.len();
        self.entries.retain(|b| b.until == 0 || b.until > now_unix);
        if self.entries.len() != before {
            self.save();
        }
    }

    fn save(&self) {
        let Some(path) = &self.path else {
            return;
        };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let text = match ron::ser::to_string_pretty(&BanFile { bans: self.entries.clone() }, ron::ser::PrettyConfig::default()) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!("net: could not encode the ban list: {e}");
                return;
            }
        };
        // Write beside the file and move it over, so a crash never leaves half a list.
        let tmp = path.with_extension("ron.tmp");
        if let Err(e) = std::fs::write(&tmp, text).and_then(|_| std::fs::rename(&tmp, path)) {
            tracing::warn!("net: could not save the ban list {}: {e}", path.display());
        }
    }
}

/// The validation state of one session.
pub struct Validator {
    cfg: NetValidation,
    topology: Topology,
    peers: HashMap<u64, PeerState>,
    offenders: HashMap<[u8; 32], Offender>,
    tracks: HashMap<u128, Track>,
    poses: HashMap<u128, PoseTrack>,
    /// Newest accepted snapshot stamp per sending peer.
    stamps: HashMap<u64, f64>,
    rules: Vec<RpcRule>,
    pub bans: BanList,
    /// Violations counted since the session started.
    pub total: u64,
}

/// Offender records kept at most.
const MAX_OFFENDERS: usize = 4096;
/// Seconds between two log lines for one peer and reason.
const LOG_EVERY: f64 = 5.0;
/// Corrections for one object are at least this far apart (s).
const CORRECT_EVERY: f64 = 0.15;
/// Shortest time step the acceleration estimate uses (s).
const MIN_DT: f64 = 0.02;
/// A timestamp this far ahead of the newest one is nonsense (s).
const MAX_STAMP_JUMP: f64 = 30.0;
/// Seconds after the rig root jumps during which pose limits relax.
const ROOT_GRACE: f64 = 0.6;
/// Longest attachment chain a client may build (parent of parent of ...).
pub const MAX_PARENT_DEPTH: usize = 32;
/// Entries kept in a peer's claim-cooldown map.
const MAX_CLAIM_MEMORY: usize = 64;

fn finite3(v: Vec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

fn finite_quat(q: Quat) -> bool {
    q.x.is_finite() && q.y.is_finite() && q.z.is_finite() && q.w.is_finite()
}

impl Validator {
    pub fn new(cfg: NetValidation, topology: Topology) -> Self {
        let cfg = cfg.sanitized();
        let mut rules: Vec<RpcRule> = BUILTIN_RPCS
            .iter()
            .map(|n| RpcRule { name: (*n).to_owned(), permission: RpcPermission::Anyone, rate: 10.0, burst: 20.0, max_args: 2, interaction: true })
            .collect();
        // Player reports: a slow bucket, so a report cannot be used to flood the admin queue.
        rules.push(RpcRule { name: REPORT_RPC.to_owned(), permission: RpcPermission::Anyone, rate: 0.2, burst: 3.0, max_args: 2, interaction: false });
        for r in &cfg.rpc_rules {
            match rules.iter_mut().find(|x| x.name == r.name) {
                Some(x) => *x = r.clone(),
                None => rules.push(r.clone()),
            }
        }
        Self {
            cfg,
            topology,
            peers: HashMap::new(),
            offenders: HashMap::new(),
            tracks: HashMap::new(),
            poses: HashMap::new(),
            stamps: HashMap::new(),
            rules,
            bans: BanList::default(),
            total: 0,
        }
    }

    pub fn settings(&self) -> &NetValidation {
        &self.cfg
    }

    pub fn topology(&self) -> Topology {
        self.topology
    }

    /// The gameplay rules (movement, poses, ownership, spawns, RPC permissions) apply.
    pub fn strict(&self) -> bool {
        self.cfg.enabled && (self.topology == Topology::Dedicated || self.cfg.strict_on_listen_host)
    }

    /// Register (or replace) an RPC's permission and limits.
    pub fn register_rpc(&mut self, rule: RpcRule) {
        match self.rules.iter_mut().find(|x| x.name == rule.name) {
            Some(x) => *x = rule,
            None => self.rules.push(rule),
        }
        // Peers' override buckets point at rule indices: rebuild lazily.
        for p in self.peers.values_mut() {
            p.rpc_buckets.clear();
        }
    }

    pub fn rpc_rule(&self, name: &str) -> Option<&RpcRule> {
        self.rules.iter().find(|r| r.name == name)
    }

    // ------------------------------------------------------------ peers and strikes

    pub fn peer_joined(&mut self, peer: u64, key: [u8; 32], name: &str) {
        let r = &self.cfg.rates;
        let b = |s: zestengine_core::RateSetting| TokenBucket::new(s.rate, s.burst);
        let buckets = [
            b(r.packets),
            b(r.messages),
            b(r.rpc),
            b(r.spawn),
            b(r.ownership),
            b(r.chat),
            b(r.voice),
            b(r.avparam),
            b(r.avatar),
            b(r.snapshot),
            b(r.anim),
            b(r.face),
            b(r.var),
            b(r.input),
        ];
        self.peers.insert(
            peer,
            PeerState {
                key,
                name: name.to_owned(),
                buckets,
                rpc_buckets: Vec::new(),
                counts: [0; REASONS],
                last_log: [f64::NEG_INFINITY; REASONS],
                suppressed: [0; REASONS],
                warned: false,
                claim_last: HashMap::new(),
                vars: HashMap::new(),
            },
        );
    }

    pub fn peer_left(&mut self, peer: u64) {
        self.peers.remove(&peer);
        self.tracks.retain(|_, t| t.owner != peer);
        self.poses.retain(|_, t| t.owner != peer);
        self.stamps.remove(&peer);
    }

    /// An object's ownership changed or it despawned: forget its movement memory.
    pub fn forget_object(&mut self, id: u128) {
        self.tracks.remove(&id);
        self.poses.remove(&id);
    }

    /// Violation counts of a peer, by reason.
    pub fn counts(&self, peer: u64) -> [u32; REASONS] {
        self.peers.get(&peer).map_or([0; REASONS], |p| p.counts)
    }

    pub fn count(&self, peer: u64, reason: Reason) -> u32 {
        self.peers.get(&peer).map_or(0, |p| p.counts[reason as usize])
    }

    /// The decayed strike score of a peer's player id.
    pub fn score(&self, peer: u64, now: f64) -> f32 {
        let Some(p) = self.peers.get(&peer) else {
            return 0.0;
        };
        self.offenders.get(&p.key).map_or(0.0, |o| decayed(o, now, self.cfg.strikes.decay_per_sec))
    }

    /// The peer is over the `drop_at` score: its gameplay data is ignored
    /// until the score decays.
    pub fn dropping(&self, peer: u64, now: f64) -> bool {
        self.cfg.enabled && self.score(peer, now) >= self.cfg.strikes.drop_at
    }

    /// Count a rejection and add its strike points. O(1). `detail` goes to
    /// the (rate-limited) log line.
    pub fn report(&mut self, peer: u64, reason: Reason, detail: &str, now: f64) -> Escalation {
        self.total += 1;
        let Some(p) = self.peers.get_mut(&peer) else {
            return Escalation::None;
        };
        let i = reason as usize;
        p.counts[i] = p.counts[i].saturating_add(1);
        if now - p.last_log[i] >= LOG_EVERY {
            let skipped = std::mem::take(&mut p.suppressed[i]);
            tracing::warn!(
                "net: peer {peer} ({}, {}): {}{}{}",
                p.name,
                crate::net_identity::short_fingerprint(&crate::net_identity::fingerprint(&p.key)),
                reason.label(),
                if detail.is_empty() { String::new() } else { format!(" ({detail})") },
                if skipped > 0 { format!(" [+{skipped} more]") } else { String::new() },
            );
            p.last_log[i] = now;
        } else {
            p.suppressed[i] = p.suppressed[i].saturating_add(1);
        }
        if !self.cfg.enabled {
            return Escalation::None;
        }
        let key = p.key;
        let policy = self.cfg.strikes.clone();
        if self.offenders.len() >= MAX_OFFENDERS && !self.offenders.contains_key(&key) {
            self.offenders.retain(|_, o| decayed(o, now, policy.decay_per_sec) > 0.5 || o.kicks > 0);
            if self.offenders.len() >= MAX_OFFENDERS {
                return Escalation::None;
            }
        }
        let o = self.offenders.entry(key).or_default();
        o.score = decayed(o, now, policy.decay_per_sec) + reason.weight();
        o.last = now;
        if o.kicks > 0 && now - o.last_kick > policy.kick_memory as f64 {
            o.kicks = 0;
        }
        if o.score >= policy.kick_at {
            o.kicks += 1;
            o.last_kick = now;
            o.score = 0.0;
            return if o.kicks >= policy.ban_after_kicks { Escalation::Ban } else { Escalation::Kick };
        }
        if o.score >= policy.warn_at {
            if !p.warned {
                p.warned = true;
                return Escalation::Warn;
            }
        } else if o.score < policy.warn_at * 0.5 {
            p.warned = false;
        }
        Escalation::None
    }

    // ------------------------------------------------------------ rate limits

    /// Take a token for `kind`; `false` = over the limit (not recorded as a
    /// strike: the caller reports `Reason::RateLimited` so it can name the traffic).
    pub fn allow(&mut self, peer: u64, kind: Limit, now: f64) -> bool {
        if !self.cfg.enabled {
            return true;
        }
        match self.peers.get_mut(&peer) {
            Some(p) => p.buckets[kind as usize].take(now),
            None => true,
        }
    }

    // ------------------------------------------------------------ structure

    pub fn string_ok(&self, s: &str) -> bool {
        s.len() <= self.cfg.max_string as usize
    }

    pub fn chat_ok(&self, s: &str) -> bool {
        s.len() <= self.cfg.max_chat as usize
    }

    pub fn var_ok(&self, key: &str, bytes: usize) -> bool {
        self.string_ok(key) && bytes <= self.cfg.max_var_bytes as usize
    }

    /// Charge a variable write to the writer's budget (distinct keys and total bytes it has
    /// written). `false` = over budget, nothing is recorded. A rewrite of a key replaces its
    /// old size, so updating one value never fills the budget.
    pub fn var_budget(&mut self, peer: u64, key: &str, bytes: usize) -> bool {
        let (max_total, max_keys) = (self.cfg.max_var_total as usize, self.cfg.max_var_keys as usize);
        let Some(p) = self.peers.get_mut(&peer) else { return true };
        let others: usize = p.vars.iter().filter(|(k, _)| k.as_str() != key).map(|(_, b)| *b).sum();
        let new_key = !p.vars.contains_key(key);
        if others + bytes > max_total || (new_key && p.vars.len() >= max_keys) {
            return false;
        }
        p.vars.insert(key.to_owned(), bytes);
        true
    }

    /// Bytes of variables a peer has written (inspector / tests).
    pub fn var_bytes(&self, peer: u64) -> usize {
        self.peers.get(&peer).map_or(0, |p| p.vars.values().sum())
    }

    /// A voice packet of this payload size is within the limit.
    pub fn voice_ok(&self, bytes: usize) -> bool {
        bytes <= self.cfg.max_voice_packet as usize
    }

    /// Whether `peer` may attach `id` under `parent` (0 = detach). `owner_of` is the ownership
    /// table (0 = unowned), `parents` the current attachments. Refuses itself, a cycle, a chain
    /// deeper than [`MAX_PARENT_DEPTH`], and (when the gameplay rules apply) a parent owned by
    /// another client: attaching your object to someone else's drags it with theirs.
    pub fn parent_ok(&self, peer: u64, host_peer: u64, id: u128, parent: u128, owner_of: impl Fn(u128) -> u64, parents: &HashMap<u128, u128>) -> Result<(), Reason> {
        if parent == 0 {
            return Ok(());
        }
        if parent == id {
            return Err(Reason::Malformed);
        }
        let mut cur = parent;
        for _ in 0..MAX_PARENT_DEPTH {
            match parents.get(&cur).copied().filter(|p| *p != 0) {
                None => {
                    if self.strict() {
                        let o = owner_of(parent);
                        if o != 0 && o != peer && o != host_peer {
                            return Err(Reason::OwnershipDenied);
                        }
                    }
                    return Ok(());
                }
                Some(next) if next == id => return Err(Reason::Malformed),
                Some(next) => cur = next,
            }
        }
        Err(Reason::TooLarge)
    }

    /// Entries one snapshot packet may carry.
    pub fn max_entries(&self) -> usize {
        self.cfg.max_entries as usize
    }

    pub fn max_message(&self) -> usize {
        self.cfg.max_message as usize
    }

    pub fn max_reorder(&self) -> usize {
        self.cfg.max_reorder as usize
    }

    pub fn world_ok(&self, p: Vec3) -> bool {
        let (lo, hi) = (self.cfg.world_min, self.cfg.world_max);
        finite3(p) && (0..3).all(|k| p[k] >= lo[k] && p[k] <= hi[k])
    }

    /// Whether the extra floats of a predicted pawn's input are sane.
    pub fn input_extras_ok(&self, extra: &[f32]) -> bool {
        extra.iter().all(|v| v.is_finite() && v.abs() <= self.cfg.max_input_extra)
    }

    /// A packet timestamp: monotonic per sender (older ones are dropped
    /// quietly), finite, not absurdly far ahead.
    pub fn stamp(&mut self, peer: u64, time: f64) -> Stamp {
        if !time.is_finite() {
            return Stamp::Bad;
        }
        match self.stamps.get(&peer) {
            Some(&last) if time < last - 1.0 => Stamp::Stale,
            Some(&last) if time > last + MAX_STAMP_JUMP => Stamp::Bad,
            Some(&last) => {
                if time > last {
                    self.stamps.insert(peer, time);
                }
                Stamp::Fresh
            }
            None => {
                self.stamps.insert(peer, time);
                Stamp::Fresh
            }
        }
    }

    // ------------------------------------------------------------ movement

    fn limits(&self, kind: Kind) -> (f32, f32) {
        match kind {
            Kind::Prop => (self.cfg.prop_speed, self.cfg.prop_accel),
            Kind::Avatar => (self.cfg.avatar_speed, self.cfg.avatar_accel),
        }
    }

    /// Check one snapshot entry an owner sent for `id`. `baseline` is where
    /// the server last had the object (the start of a new track).
    #[allow(clippy::too_many_arguments)]
    pub fn entry(
        &mut self,
        owner: u64,
        id: u128,
        kind: Kind,
        pos: Option<Vec3>,
        rot: Option<Quat>,
        scale: Option<Vec3>,
        vel: Option<Vec3>,
        baseline: Option<Vec3>,
        now: f64,
    ) -> Result<(), Rejection> {
        let bad = |reason| Err(Rejection { reason, correct_to: None });
        if pos.is_some_and(|v| !finite3(v)) || scale.is_some_and(|v| !finite3(v)) || vel.is_some_and(|v| !finite3(v)) || rot.is_some_and(|q| !finite_quat(q)) {
            return bad(Reason::NotFinite);
        }
        if let Some(s) = scale
            && (s.min_element() <= 0.0 || s.max_element() > self.cfg.max_scale)
        {
            return bad(Reason::OutOfBounds);
        }
        if !self.cfg.enabled {
            return Ok(());
        }
        if let Some(p) = pos
            && !self.world_ok(p)
        {
            return bad(Reason::OutOfBounds);
        }
        if !self.strict() {
            return Ok(());
        }
        let (speed, accel) = self.limits(kind);
        if let Some(v) = vel
            && v.length() > speed * 2.0
        {
            return bad(Reason::Speed);
        }
        let Some(p) = pos else {
            return Ok(());
        };
        let tol = self.cfg.teleport_tolerance;
        let t = self.tracks.entry(id).or_insert_with(|| Track {
            owner,
            pos: baseline.unwrap_or(p),
            vel: Vec3::ZERO,
            t: now,
            slack: tol,
            last_correct: f64::NEG_INFINITY,
        });
        if t.owner != owner {
            *t = Track { owner, pos: baseline.unwrap_or(p), vel: Vec3::ZERO, t: now, slack: tol, last_correct: f64::NEG_INFINITY };
        }
        let dt = (now - t.t).clamp(0.0, 1.0);
        let avail = t.slack + speed * dt as f32;
        let d = p.distance(t.pos);
        if d > avail {
            let correct = (now - t.last_correct >= CORRECT_EVERY).then(|| {
                t.last_correct = now;
                t.pos
            });
            return Err(Rejection { reason: Reason::Teleport, correct_to: correct });
        }
        let step = dt.max(MIN_DT);
        let v = (p - t.pos) / step as f32;
        if accel > 0.0 && v.length() > speed * 0.5 && (v - t.vel).length() / step as f32 > accel {
            let correct = (now - t.last_correct >= CORRECT_EVERY).then(|| {
                t.last_correct = now;
                t.pos
            });
            return Err(Rejection { reason: Reason::Speed, correct_to: correct });
        }
        t.slack = (avail - d).min(tol);
        t.pos = p;
        t.vel = v;
        t.t = now;
        Ok(())
    }

    // ------------------------------------------------------------ VR poses

    /// Check an avatar pose packet from the owner of `id`. `root` is the
    /// avatar root's position as the server has it; `scale` the avatar scale.
    /// On success the poses become the new baseline.
    pub fn avatar_pose(
        &mut self,
        owner: u64,
        id: u128,
        root: Option<Vec3>,
        scale: f32,
        poses: &[Option<(Vec3, Quat)>; TRACKER_SLOTS],
        now: f64,
    ) -> Result<(), Reason> {
        for (p, r) in poses.iter().flatten() {
            if !finite3(*p) || !finite_quat(*r) {
                return Err(Reason::NotFinite);
            }
        }
        if !self.cfg.enabled {
            return Ok(());
        }
        if poses.iter().flatten().any(|(p, _)| !self.world_ok(*p)) {
            return Err(Reason::OutOfBounds);
        }
        if !self.strict() {
            return Ok(());
        }
        let Some(root) = root else {
            return Ok(());
        };
        let scale = if scale.is_finite() { scale.clamp(0.1, 10.0) } else { 1.0 };
        let c = &self.cfg;
        let slack = c.reach_slack;
        let prev = self.poses.get(&id).filter(|t| t.owner == owner).copied();
        let mut rel = [None; TRACKER_SLOTS];
        for (k, p) in poses.iter().enumerate() {
            rel[k] = p.map(|(p, _)| p - root);
        }
        // The rig root jumped (a teleport or respawn the server's copy of the
        // root has not caught up with, or already has): poses are judged against
        // the body only for a moment.
        let mut grace_until = prev.map_or(0.0, |t| t.grace_until);
        if let Some(t) = prev
            && t.root.distance(root) > 0.3
        {
            grace_until = now + ROOT_GRACE;
        }
        let in_grace = now < grace_until;
        // The head: near the root, at a human height.
        let head = rel[0].or_else(|| prev.and_then(|t| t.rel[0]));
        if let Some(h) = rel[0].filter(|_| !in_grace) {
            let horizontal = (h.x * h.x + h.z * h.z).sqrt();
            if horizontal > (c.head_radius + slack) * scale || h.y < c.head_min_height * scale - slack || h.y > c.head_max_height * scale + slack {
                return Err(Reason::ImpossiblePose);
            }
        }
        if let Some(h) = head {
            for (k, r) in rel.iter().enumerate().skip(1) {
                let Some(r) = r else {
                    continue;
                };
                // Hands (slots 1, 2) reach as far as an arm; the other trackers as far as a body.
                let reach = if k <= 2 { c.arm_reach } else { c.body_reach };
                if r.distance(h) > (reach + slack) * scale {
                    return Err(Reason::ImpossiblePose);
                }
            }
        }
        // Speeds relative to the root, with the same budget as props.
        let mut slacks = prev.map_or([slack; TRACKER_SLOTS], |t| t.slack);
        if let Some(t) = prev.filter(|_| !in_grace) {
            let dt = (now - t.t).clamp(0.0, 1.0) as f32;
            for k in 0..TRACKER_SLOTS {
                let (Some(a), Some(b)) = (rel[k], t.rel[k]) else {
                    continue;
                };
                let speed = if k == 0 { c.head_speed } else { c.hand_speed } * scale;
                let avail = slacks[k] + speed * dt;
                let d = a.distance(b);
                if d > avail {
                    return Err(Reason::Speed);
                }
                slacks[k] = (avail - d).min(slack);
            }
        }
        let mut keep = prev.map_or([None; TRACKER_SLOTS], |t| t.rel);
        for (k, r) in rel.iter().enumerate() {
            if r.is_some() {
                keep[k] = *r;
            }
        }
        self.poses.insert(id, PoseTrack { owner, rel: keep, slack: slacks, t: now, root, grace_until });
        Ok(())
    }

    // ------------------------------------------------------------ ownership

    /// Decide an ownership request from a client.
    pub fn claim(&mut self, c: &ClaimCtx, now: f64) -> Claim {
        if c.cur == c.by {
            return Claim::Grant;
        }
        let strict = self.strict();
        let grabbable = c.mode == OwnershipMode::Grabbable;
        if !(strict || grabbable) {
            // The old rules: Takeover is any peer any time, Request only when free, Fixed never.
            return match c.mode {
                OwnershipMode::Takeover | OwnershipMode::Grabbable => Claim::Grant,
                OwnershipMode::Request if c.cur == 0 => Claim::Grant,
                OwnershipMode::Request => Claim::Ask,
                OwnershipMode::Fixed => Claim::Deny(DenyWhy::NotClaimable),
            };
        }
        // The per-object cooldown.
        if let Some(p) = self.peers.get_mut(&c.by) {
            if p.claim_last.len() >= MAX_CLAIM_MEMORY {
                let cd = self.cfg.claim_cooldown as f64;
                p.claim_last.retain(|_, t| now - *t < cd);
            }
            if let Some(&last) = p.claim_last.get(&c.id)
                && now - last < self.cfg.claim_cooldown as f64
            {
                return Claim::Deny(DenyWhy::Cooldown);
            }
            if p.claim_last.len() < MAX_CLAIM_MEMORY {
                p.claim_last.insert(c.id, now);
            }
        }
        if c.mode == OwnershipMode::Fixed || c.protected || !c.known {
            return Claim::Deny(DenyWhy::NotClaimable);
        }
        if c.cur == c.host_peer && c.cur != 0 {
            return Claim::Deny(DenyWhy::ServerOwned);
        }
        let free = c.cur == 0 || c.owner_silent;
        if !free {
            return if c.mode == OwnershipMode::Request { Claim::Ask } else { Claim::Deny(DenyWhy::OwnedByOther) };
        }
        match (c.requester, c.object) {
            (Some(a), Some(b)) if finite3(a) && finite3(b) => {
                if a.distance(b) > self.cfg.grab_range {
                    return Claim::Deny(DenyWhy::OutOfRange);
                }
            }
            (None, _) => return Claim::Deny(DenyWhy::NoAvatar),
            // The object's position is unknown: it cannot be shown to be in reach.
            _ => return Claim::Deny(DenyWhy::OutOfRange),
        }
        Claim::Grant
    }

    // ------------------------------------------------------------ RPCs

    /// Check an RPC a client sent. `obj_owner` is the owner of the object it
    /// was sent on (0 = unowned / none), `has_obj` whether it names one.
    #[allow(clippy::too_many_arguments)]
    pub fn rpc(&mut self, peer: u64, host_peer: u64, name: &str, nargs: usize, bytes: usize, has_obj: bool, obj_owner: u64, now: f64) -> RpcVerdict {
        if name.len() > self.cfg.max_string as usize || nargs > self.cfg.max_rpc_args as usize || bytes > self.cfg.max_rpc_bytes as usize {
            return RpcVerdict::Deny(Reason::TooLarge);
        }
        let rule_idx = self.rules.iter().position(|r| r.name == name);
        let (general, burst) = (self.cfg.rates.rpc.rate, self.cfg.rates.rpc.burst);
        let allowed = if !self.cfg.enabled {
            true
        } else if let Some(i) = rule_idx
            && (self.rules[i].rate > 0.0 || self.rules[i].burst > 0.0)
        {
            let (r, b) = (self.rules[i].rate, self.rules[i].burst);
            let rate = if r > 0.0 { r } else { general };
            let bst = if b >= 1.0 { b } else { burst };
            match self.peers.get_mut(&peer) {
                Some(p) => {
                    let slot = match p.rpc_buckets.iter().position(|(k, _)| *k == i) {
                        Some(s) => s,
                        None => {
                            p.rpc_buckets.push((i, TokenBucket::new(rate, bst)));
                            p.rpc_buckets.len() - 1
                        }
                    };
                    p.rpc_buckets[slot].1.take(now) && p.buckets[Limit::Rpc as usize].take(now)
                }
                None => false,
            }
        } else {
            self.allow(peer, Limit::Rpc, now)
        };
        if !allowed {
            return RpcVerdict::Deny(Reason::RateLimited);
        }
        if let Some(i) = rule_idx
            && self.rules[i].max_args > 0
            && nargs > self.rules[i].max_args as usize
        {
            return RpcVerdict::Deny(Reason::TooLarge);
        }
        if !self.cfg.enabled || peer == host_peer {
            return RpcVerdict::Allow;
        }
        let permission = match rule_idx {
            Some(i) => self.rules[i].permission,
            None if self.topology == Topology::Dedicated && self.cfg.strict_rpc_permissions => RpcPermission::Owner,
            None => RpcPermission::Anyone,
        };
        match permission {
            RpcPermission::Anyone => RpcVerdict::Allow,
            RpcPermission::Server => RpcVerdict::Deny(Reason::RpcDenied),
            RpcPermission::Owner if has_obj && obj_owner == peer => RpcVerdict::Allow,
            RpcPermission::Owner => RpcVerdict::Deny(Reason::RpcDenied),
        }
    }

    /// Whether an RPC needs the requester within interaction range of its object.
    pub fn needs_interaction(&self, name: &str) -> bool {
        self.rules.iter().any(|r| r.name == name && r.interaction)
    }

    /// The requester's avatar is within interaction range of the object.
    pub fn interaction_ok(&self, requester: Option<Vec3>, object: Option<Vec3>) -> bool {
        match (requester, object) {
            (Some(a), Some(b)) => finite3(a) && finite3(b) && a.distance(b) <= self.cfg.interact_range,
            _ => false,
        }
    }

    // ------------------------------------------------------------ spawns

    /// Check a spawn a client asks for.
    pub fn spawn(&mut self, peer: u64, req: &SpawnReq, now: f64) -> Result<(), Reason> {
        if !self.allow(peer, Limit::Spawn, now) {
            return Err(Reason::RateLimited);
        }
        if !finite3(req.pos) || !finite_quat(req.rot) {
            return Err(Reason::NotFinite);
        }
        if let Some(path) = req.prefab
            && (path.len() > self.cfg.max_string as usize || !safe_path(path))
        {
            return Err(Reason::SpawnDenied);
        }
        if !self.cfg.enabled {
            return Ok(());
        }
        if !self.world_ok(req.pos) {
            return Err(Reason::OutOfBounds);
        }
        if req.live >= self.cfg.max_client_spawns as usize {
            return Err(Reason::SpawnDenied);
        }
        if let Some(path) = req.prefab
            && let Some(q) = self.cfg.spawn_quotas.iter().find(|q| q.matches(path))
            && ((q.per_peer > 0 && req.live_prefab >= q.per_peer as usize) || (q.total > 0 && req.live_prefab_total >= q.total as usize))
        {
            return Err(Reason::SpawnDenied);
        }
        if self.strict() {
            if !self.prefab_allowed(req.prefab) {
                return Err(Reason::SpawnDenied);
            }
            match req.requester {
                Some(r) if r.distance(req.pos) <= self.cfg.spawn_range => {}
                _ => return Err(Reason::SpawnDenied),
            }
        } else if !self.cfg.client_spawn_prefabs.is_empty() && !self.prefab_allowed(req.prefab) {
            return Err(Reason::SpawnDenied);
        }
        Ok(())
    }

    fn prefab_allowed(&self, prefab: Option<&str>) -> bool {
        let list = &self.cfg.client_spawn_prefabs;
        match prefab {
            Some(path) => list.iter().any(|e| match e.strip_suffix('*') {
                Some(prefix) => path.starts_with(prefix),
                None => e == path,
            }),
            None => list.iter().any(|e| e == "clone:*"),
        }
    }
}

/// The RPC a client calls (target `Server`) to report another player: `[Int(peer), Str(reason)]`.
pub const REPORT_RPC: &str = "zestengine.report";

/// RPC names the engine itself sends from clients.
const BUILTIN_RPCS: [&str; 2] = ["zestengine.interact", "zestengine.interact_release"];

/// A project-relative path with no way out of the project.
pub fn safe_path(p: &str) -> bool {
    !p.is_empty()
        && !p.starts_with('/')
        && !p.starts_with('\\')
        && !p.contains(':')
        && !p.contains('\0')
        && !p.split(['/', '\\']).any(|part| part == "..")
}

fn decayed(o: &Offender, now: f64, per_sec: f32) -> f32 {
    (o.score - (now - o.last).max(0.0) as f32 * per_sec).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(strict: bool) -> Validator {
        let mut v = Validator::new(NetValidation::default(), if strict { Topology::Dedicated } else { Topology::ListenHost });
        v.peer_joined(2, [7; 32], "Mallory");
        v.peer_joined(3, [8; 32], "Honest");
        v
    }

    const OBJ: u128 = 0xABCD;

    fn mv(v: &mut Validator, pos: Vec3, base: Option<Vec3>, now: f64) -> Result<(), Rejection> {
        v.entry(2, OBJ, Kind::Prop, Some(pos), None, None, None, base, now)
    }

    #[test]
    fn buckets_burst_then_refill() {
        let mut v = v(true);
        let burst = v.settings().rates.rpc.burst as usize;
        assert!((0..burst).all(|_| v.allow(2, Limit::Rpc, 1.0)));
        assert!(!v.allow(2, Limit::Rpc, 1.0), "the burst is spent");
        assert!(v.allow(3, Limit::Rpc, 1.0), "another peer has its own bucket");
        assert!(v.allow(2, Limit::Rpc, 1.0 + 1.0), "it refills with time");
        assert!(v.allow(99, Limit::Rpc, 1.0), "a peer the validator does not know is not throttled (fail open)");
    }

    #[test]
    fn non_finite_and_out_of_bounds_are_refused() {
        let mut v = v(false);
        for bad in [Vec3::new(f32::NAN, 0.0, 0.0), Vec3::new(0.0, f32::INFINITY, 0.0), Vec3::new(0.0, 0.0, f32::NEG_INFINITY)] {
            assert_eq!(mv(&mut v, bad, None, 0.0).unwrap_err().reason, Reason::NotFinite);
        }
        assert_eq!(mv(&mut v, Vec3::new(1.0e7, 0.0, 0.0), None, 0.0).unwrap_err().reason, Reason::OutOfBounds);
        let e = v.entry(2, OBJ, Kind::Prop, None, Some(Quat::from_xyzw(f32::NAN, 0.0, 0.0, 1.0)), None, None, None, 0.0);
        assert_eq!(e.unwrap_err().reason, Reason::NotFinite);
        let e = v.entry(2, OBJ, Kind::Prop, None, None, Some(Vec3::splat(1.0e9)), None, None, 0.0);
        assert_eq!(e.unwrap_err().reason, Reason::OutOfBounds);
        assert!(mv(&mut v, Vec3::new(10.0, 1.0, 10.0), None, 0.0).is_ok());
    }

    #[test]
    fn teleports_are_caught_with_a_tolerance_and_corrected() {
        let mut v = v(true);
        let start = Vec3::new(0.0, 1.0, 0.0);
        assert!(mv(&mut v, start, Some(start), 0.0).is_ok());
        // Walking at 5 m/s, 60 Hz: fine for a long time.
        let mut t = 0.0;
        let mut p = start;
        for _ in 0..120 {
            t += 1.0 / 60.0;
            p.x += 5.0 / 60.0;
            assert!(mv(&mut v, p, None, t).is_ok());
        }
        // A jitter burst (two frames' worth at once) is inside the tolerance.
        t += 1.0 / 60.0;
        p.x += 0.4;
        assert!(mv(&mut v, p, None, t).is_ok());
        // A 40 m jump in one frame is a teleport, and the owner is told where it was.
        t += 1.0 / 60.0;
        let e = mv(&mut v, p + Vec3::new(40.0, 0.0, 0.0), None, t).unwrap_err();
        assert_eq!(e.reason, Reason::Teleport);
        assert!(e.correct_to.is_some_and(|c| c.distance(p) < 1e-4), "corrected to the last accepted position");
        // Corrections are rate limited, rejections are not.
        let e = mv(&mut v, p + Vec3::new(40.0, 0.0, 0.0), None, t + 0.001).unwrap_err();
        assert_eq!((e.reason, e.correct_to), (Reason::Teleport, None));
        // Back where it was: accepted again.
        assert!(mv(&mut v, p, None, t + 0.02).is_ok());
    }

    #[test]
    fn speed_hacks_are_caught_by_the_budget() {
        let mut v = v(true);
        let mut p = Vec3::ZERO;
        assert!(mv(&mut v, p, Some(p), 0.0).is_ok());
        let mut t = 0.0;
        let mut refused = 0;
        // 200 m/s for a second: only the tolerance plus 60 m/s gets through.
        for _ in 0..60 {
            t += 1.0 / 60.0;
            let next = p + Vec3::new(200.0 / 60.0, 0.0, 0.0);
            match mv(&mut v, next, None, t) {
                Ok(()) => p = next,
                Err(_) => refused += 1,
            }
        }
        assert!(refused > 30, "refused {refused}");
        assert!(p.x < 80.0, "moved {} m in a second, cap is 60 m/s plus tolerance", p.x);
    }

    #[test]
    fn listen_hosts_only_get_the_sanity_checks_unless_strict() {
        let mut lax = v(false);
        assert!(mv(&mut lax, Vec3::ZERO, Some(Vec3::ZERO), 0.0).is_ok());
        assert!(mv(&mut lax, Vec3::new(4000.0, 0.0, 0.0), None, 0.01).is_ok(), "no movement budget on a listen host");
        let mut strict = Validator::new(NetValidation { strict_on_listen_host: true, ..Default::default() }, Topology::ListenHost);
        strict.peer_joined(2, [7; 32], "M");
        assert!(strict.strict());
        assert!(mv(&mut strict, Vec3::ZERO, Some(Vec3::ZERO), 0.0).is_ok());
        assert_eq!(mv(&mut strict, Vec3::new(4000.0, 0.0, 0.0), None, 0.01).unwrap_err().reason, Reason::Teleport);
    }

    fn pose(head: Vec3, l: Vec3, r: Vec3) -> [Option<(Vec3, Quat)>; TRACKER_SLOTS] {
        let mut p = [None; TRACKER_SLOTS];
        p[0] = Some((head, Quat::IDENTITY));
        p[1] = Some((l, Quat::IDENTITY));
        p[2] = Some((r, Quat::IDENTITY));
        p
    }

    #[test]
    fn vr_poses_must_be_a_body() {
        let mut v = v(true);
        let root = Vec3::new(10.0, 0.0, 10.0);
        let head = root + Vec3::new(0.1, 1.7, 0.0);
        let good = pose(head, head + Vec3::new(-0.3, -0.4, 0.3), head + Vec3::new(0.3, -0.4, 0.3));
        assert!(v.avatar_pose(2, 1, Some(root), 1.0, &good, 0.0).is_ok());
        // Hands 5 m from the head.
        let long_arms = pose(head, head + Vec3::new(5.0, 0.0, 0.0), head);
        assert_eq!(v.avatar_pose(2, 1, Some(root), 1.0, &long_arms, 0.01), Err(Reason::ImpossiblePose));
        // The head 30 m from the root.
        let far = pose(root + Vec3::new(30.0, 1.7, 0.0), root + Vec3::new(30.0, 1.5, 0.0), root + Vec3::new(30.0, 1.5, 0.0));
        assert_eq!(v.avatar_pose(2, 2, Some(root), 1.0, &far, 0.01), Err(Reason::ImpossiblePose));
        // NaN.
        let nan = pose(Vec3::NAN, head, head);
        assert_eq!(v.avatar_pose(2, 1, Some(root), 1.0, &nan, 0.02), Err(Reason::NotFinite));
        // A hand that jumps 1.1 m in one frame (66 m/s) is a speed hack; a normal swing is not.
        let swing = pose(head, head + Vec3::new(-0.3, -0.4, 0.7), head + Vec3::new(0.3, -0.4, 0.3));
        assert!(v.avatar_pose(2, 1, Some(root), 1.0, &swing, 0.1).is_ok());
        let snap = pose(head, head + Vec3::new(-0.3, -0.4, 0.7), head + Vec3::new(0.3, 0.6, 0.3));
        assert_eq!(v.avatar_pose(2, 1, Some(root), 1.0, &snap, 0.1 + 1.0 / 90.0), Err(Reason::Speed));
        // A giant avatar reaches further.
        let reach = pose(head, head + Vec3::new(1.9, 0.0, 0.0), head);
        assert_eq!(v.avatar_pose(2, 5, Some(root), 1.0, &reach, 0.5), Err(Reason::ImpossiblePose));
        assert!(v.avatar_pose(2, 5, Some(root), 2.0, &reach, 0.5).is_ok());
        // Teleporting the whole rig is not a speed violation: poses are root-relative.
        let moved = Vec3::new(500.0, 0.0, 500.0);
        let h2 = moved + Vec3::new(0.1, 1.7, 0.0);
        let teleported = pose(h2, h2 + Vec3::new(-0.3, -0.4, 0.7), h2 + Vec3::new(0.3, -0.4, 0.3));
        assert!(v.avatar_pose(2, 1, Some(moved), 1.0, &teleported, 0.2).is_ok());
    }

    #[test]
    fn timestamps_must_be_finite_and_move_forward() {
        let mut v = v(true);
        assert_eq!(v.stamp(2, 10.0), Stamp::Fresh);
        assert_eq!(v.stamp(2, 10.5), Stamp::Fresh);
        assert_eq!(v.stamp(2, 10.2), Stamp::Fresh, "mild reordering is fine");
        assert_eq!(v.stamp(2, 3.0), Stamp::Stale, "an old packet is dropped quietly");
        assert_eq!(v.stamp(2, f64::NAN), Stamp::Bad);
        assert_eq!(v.stamp(2, 1.0e9), Stamp::Bad);
    }

    fn claim_ctx(mode: OwnershipMode) -> ClaimCtx {
        ClaimCtx {
            id: 5,
            by: 2,
            cur: 0,
            mode,
            host_peer: 1,
            known: true,
            protected: false,
            owner_silent: false,
            requester: Some(Vec3::ZERO),
            object: Some(Vec3::new(1.0, 0.0, 0.0)),
        }
    }

    #[test]
    fn ownership_rules() {
        let mut v = v(true);
        let mut t = 0.0;
        let mut ask = |v: &mut Validator, c: ClaimCtx| {
            t += 1.0; // past the cooldown
            v.claim(&c, t)
        };
        // Grabbable and Takeover: free, in reach.
        assert_eq!(ask(&mut v, claim_ctx(OwnershipMode::Grabbable)), Claim::Grant);
        assert_eq!(ask(&mut v, claim_ctx(OwnershipMode::Takeover)), Claim::Grant);
        // Someone else's grabbed object.
        assert_eq!(ask(&mut v, ClaimCtx { cur: 3, ..claim_ctx(OwnershipMode::Takeover) }), Claim::Deny(DenyWhy::OwnedByOther));
        // ... unless its owner went silent.
        assert_eq!(ask(&mut v, ClaimCtx { cur: 3, owner_silent: true, ..claim_ctx(OwnershipMode::Takeover) }), Claim::Grant);
        // Server-owned, host-only, protected and unknown objects.
        assert_eq!(ask(&mut v, ClaimCtx { cur: 1, ..claim_ctx(OwnershipMode::Takeover) }), Claim::Deny(DenyWhy::ServerOwned));
        assert_eq!(ask(&mut v, claim_ctx(OwnershipMode::Fixed)), Claim::Deny(DenyWhy::NotClaimable));
        assert_eq!(ask(&mut v, ClaimCtx { protected: true, ..claim_ctx(OwnershipMode::Takeover) }), Claim::Deny(DenyWhy::NotClaimable));
        assert_eq!(ask(&mut v, ClaimCtx { known: false, ..claim_ctx(OwnershipMode::Takeover) }), Claim::Deny(DenyWhy::NotClaimable));
        // Out of reach, no avatar, no object position.
        assert_eq!(ask(&mut v, ClaimCtx { object: Some(Vec3::new(50.0, 0.0, 0.0)), ..claim_ctx(OwnershipMode::Grabbable) }), Claim::Deny(DenyWhy::OutOfRange));
        assert_eq!(ask(&mut v, ClaimCtx { requester: None, ..claim_ctx(OwnershipMode::Grabbable) }), Claim::Deny(DenyWhy::NoAvatar));
        assert_eq!(ask(&mut v, ClaimCtx { object: None, ..claim_ctx(OwnershipMode::Grabbable) }), Claim::Deny(DenyWhy::OutOfRange));
        // Request mode asks the owner.
        assert_eq!(ask(&mut v, ClaimCtx { cur: 3, ..claim_ctx(OwnershipMode::Request) }), Claim::Ask);
        // The owner re-claiming its own object is always fine.
        assert_eq!(v.claim(&ClaimCtx { cur: 2, ..claim_ctx(OwnershipMode::Fixed) }, 100.0), Claim::Grant);
        // The cooldown: two requests for one object in quick succession.
        assert_eq!(v.claim(&claim_ctx(OwnershipMode::Grabbable), 200.0), Claim::Grant);
        assert_eq!(v.claim(&claim_ctx(OwnershipMode::Grabbable), 200.1), Claim::Deny(DenyWhy::Cooldown));
        assert_eq!(v.claim(&claim_ctx(OwnershipMode::Grabbable), 200.6), Claim::Grant);
    }

    #[test]
    fn listen_hosts_keep_the_old_takeover_but_grabbable_is_strict() {
        let mut v = v(false);
        let far = ClaimCtx { cur: 3, object: Some(Vec3::new(500.0, 0.0, 0.0)), ..claim_ctx(OwnershipMode::Takeover) };
        assert_eq!(v.claim(&far, 1.0), Claim::Grant, "Takeover: any peer, any time");
        assert_eq!(v.claim(&ClaimCtx { mode: OwnershipMode::Grabbable, ..far }, 2.0), Claim::Deny(DenyWhy::OwnedByOther));
        assert_eq!(v.claim(&ClaimCtx { cur: 0, mode: OwnershipMode::Grabbable, ..far }, 3.0), Claim::Deny(DenyWhy::OutOfRange));
        assert_eq!(v.claim(&ClaimCtx { cur: 3, mode: OwnershipMode::Fixed, ..far }, 4.0), Claim::Deny(DenyWhy::NotClaimable));
    }

    #[test]
    fn rpc_permissions_and_limits() {
        let mut cfg = NetValidation { strict_rpc_permissions: true, ..Default::default() };
        cfg.rpc_rules.push(RpcRule { name: "admin.kick".into(), permission: RpcPermission::Server, ..Default::default() });
        cfg.rpc_rules.push(RpcRule { name: "chat.emote".into(), permission: RpcPermission::Anyone, rate: 1.0, burst: 2.0, ..Default::default() });
        cfg.rpc_rules.push(RpcRule { name: "gun.fire".into(), permission: RpcPermission::Owner, ..Default::default() });
        let mut v = Validator::new(cfg, Topology::Dedicated);
        v.peer_joined(2, [7; 32], "M");
        let call = |v: &mut Validator, name: &str, has_obj: bool, owner: u64, t: f64| v.rpc(2, 1, name, 1, 10, has_obj, owner, t);
        assert_eq!(call(&mut v, "admin.kick", true, 2, 0.0), RpcVerdict::Deny(Reason::RpcDenied), "server-only");
        assert_eq!(call(&mut v, "gun.fire", true, 2, 0.0), RpcVerdict::Allow, "owner of the object");
        assert_eq!(call(&mut v, "gun.fire", true, 3, 0.0), RpcVerdict::Deny(Reason::RpcDenied), "not the owner");
        assert_eq!(call(&mut v, "gun.fire", false, 0, 0.0), RpcVerdict::Deny(Reason::RpcDenied), "no object");
        assert_eq!(call(&mut v, "chat.emote", false, 0, 0.0), RpcVerdict::Allow, "anyone");
        assert_eq!(call(&mut v, "unlisted", true, 3, 0.0), RpcVerdict::Deny(Reason::RpcDenied), "strict: unflagged RPCs are owner-only");
        assert_eq!(call(&mut v, "unlisted", true, 2, 0.0), RpcVerdict::Allow);
        assert_eq!(call(&mut v, "zestengine.interact", true, 3, 0.0), RpcVerdict::Allow, "the engine's own RPC is open");
        // Per-RPC override: 2 in a burst, then dry.
        assert_eq!(call(&mut v, "chat.emote", false, 0, 0.0), RpcVerdict::Allow);
        assert_eq!(call(&mut v, "chat.emote", false, 0, 0.0), RpcVerdict::Deny(Reason::RateLimited));
        assert_eq!(call(&mut v, "chat.emote", false, 0, 1.1), RpcVerdict::Allow, "refilled");
        // Sizes.
        assert_eq!(v.rpc(2, 1, &"x".repeat(1000), 0, 0, true, 2, 5.0), RpcVerdict::Deny(Reason::TooLarge));
        assert_eq!(v.rpc(2, 1, "unlisted", 99, 0, true, 2, 5.0), RpcVerdict::Deny(Reason::TooLarge));
        assert_eq!(v.rpc(2, 1, "unlisted", 1, 1_000_000, true, 2, 5.0), RpcVerdict::Deny(Reason::TooLarge));
        // The server may call anything.
        v.peer_joined(1, [1; 32], "Server");
        assert_eq!(v.rpc(1, 1, "admin.kick", 0, 0, false, 0, 5.0), RpcVerdict::Allow);
        // Without the strict flag, unflagged RPCs stay open (today's behaviour).
        let mut open = Validator::new(NetValidation::default(), Topology::Dedicated);
        open.peer_joined(2, [7; 32], "M");
        assert_eq!(open.rpc(2, 1, "anything", 0, 0, false, 0, 0.0), RpcVerdict::Allow);
    }

    #[test]
    fn spawns_need_an_allowlist_on_a_dedicated_server() {
        let near = Vec3::new(1.0, 0.0, 0.0);
        let req = |prefab: Option<&'static str>, live: usize, requester: Option<Vec3>, pos: Vec3| SpawnReq { prefab, pos, rot: Quat::IDENTITY, live, live_prefab: 0, live_prefab_total: 0, requester };
        let mut d = Validator::new(NetValidation::default(), Topology::Dedicated);
        d.peer_joined(2, [7; 32], "M");
        assert_eq!(d.spawn(2, &req(Some("prefabs/ball.prefab"), 0, Some(Vec3::ZERO), near), 0.0), Err(Reason::SpawnDenied), "default: none");
        let cfg = NetValidation { client_spawn_prefabs: vec!["prefabs/ball.prefab".into(), "prefabs/fx/*".into()], ..Default::default() };
        let mut d = Validator::new(cfg, Topology::Dedicated);
        d.peer_joined(2, [7; 32], "M");
        let me = Some(Vec3::ZERO);
        assert_eq!(d.spawn(2, &req(Some("prefabs/ball.prefab"), 0, me, near), 0.0), Ok(()));
        assert_eq!(d.spawn(2, &req(Some("prefabs/fx/boom.prefab"), 0, me, near), 1.0), Ok(()));
        assert_eq!(d.spawn(2, &req(Some("prefabs/rocket.prefab"), 0, me, near), 2.0), Err(Reason::SpawnDenied));
        assert_eq!(d.spawn(2, &req(None, 0, me, near), 3.0), Err(Reason::SpawnDenied), "clones need clone:*");
        assert_eq!(d.spawn(2, &req(Some("../../etc/passwd"), 0, me, near), 4.0), Err(Reason::SpawnDenied));
        assert_eq!(d.spawn(2, &req(Some("prefabs/ball.prefab"), 0, me, Vec3::new(100.0, 0.0, 0.0)), 5.0), Err(Reason::SpawnDenied), "out of the avatar's range");
        assert_eq!(d.spawn(2, &req(Some("prefabs/ball.prefab"), 0, None, near), 6.0), Err(Reason::SpawnDenied), "no avatar");
        assert_eq!(d.spawn(2, &req(Some("prefabs/ball.prefab"), 32, me, near), 7.0), Err(Reason::SpawnDenied), "quota");
        assert_eq!(d.spawn(2, &req(Some("prefabs/ball.prefab"), 0, me, Vec3::new(f32::NAN, 0.0, 0.0)), 8.0), Err(Reason::NotFinite));
        // The spawn bucket.
        let mut n = 0;
        for k in 0..60 {
            if d.spawn(2, &req(Some("prefabs/ball.prefab"), 0, me, near), 9.0 + k as f64 * 0.001).is_ok() {
                n += 1;
            }
        }
        assert!(n <= 21, "{n} spawns got through the burst");
        // A listen host with no list allows anything (but still rejects traversal).
        let mut l = Validator::new(NetValidation::default(), Topology::ListenHost);
        l.peer_joined(2, [7; 32], "M");
        assert_eq!(l.spawn(2, &req(Some("anything.prefab"), 0, None, near), 0.0), Ok(()));
        assert_eq!(l.spawn(2, &req(Some("/etc/passwd"), 0, None, near), 1.0), Err(Reason::SpawnDenied));
    }

    #[test]
    fn per_prefab_quotas_cap_each_prefab_per_peer_and_in_total() {
        use zestengine_core::SpawnQuota;
        let me = Some(Vec3::ZERO);
        let near = Vec3::new(1.0, 0.0, 0.0);
        let cfg = NetValidation {
            client_spawn_prefabs: vec!["prefabs/*".into()],
            spawn_quotas: vec![
                SpawnQuota { prefab: "prefabs/rocket.prefab".into(), per_peer: 2, total: 0 },
                SpawnQuota { prefab: "prefabs/fx/*".into(), per_peer: 0, total: 5 },
            ],
            ..Default::default()
        };
        let mut d = Validator::new(cfg, Topology::Dedicated);
        d.peer_joined(2, [7; 32], "M");
        let req = |prefab: &'static str, live_prefab: usize, total: usize| SpawnReq { prefab: Some(prefab), pos: near, rot: Quat::IDENTITY, live: live_prefab, live_prefab, live_prefab_total: total, requester: me };
        assert_eq!(d.spawn(2, &req("prefabs/rocket.prefab", 1, 1), 0.0), Ok(()));
        assert_eq!(d.spawn(2, &req("prefabs/rocket.prefab", 2, 2), 1.0), Err(Reason::SpawnDenied), "third rocket of one peer");
        assert_eq!(d.spawn(2, &req("prefabs/ball.prefab", 9, 9), 2.0), Ok(()), "unquoted prefabs only face the general cap");
        assert_eq!(d.spawn(2, &req("prefabs/fx/boom.prefab", 0, 4), 3.0), Ok(()));
        assert_eq!(d.spawn(2, &req("prefabs/fx/boom.prefab", 0, 5), 4.0), Err(Reason::SpawnDenied), "the server-wide total is reached");
    }

    #[test]
    fn variable_budget_is_per_peer_and_rewrites_replace() {
        let cfg = NetValidation { max_var_total: 1000, max_var_keys: 3, ..Default::default() };
        let mut v = Validator::new(cfg, Topology::Dedicated);
        v.peer_joined(2, [7; 32], "M");
        v.peer_joined(3, [8; 32], "N");
        assert!(v.var_budget(2, "a", 600));
        assert!(!v.var_budget(2, "b", 500), "1100 bytes over the 1000 budget");
        assert!(v.var_budget(2, "a", 900), "rewriting a key replaces its size");
        assert!(!v.var_budget(2, "b", 200));
        assert!(v.var_budget(2, "b", 100));
        assert!(v.var_budget(2, "c", 0));
        assert!(!v.var_budget(2, "d", 0), "key limit 3");
        assert_eq!(v.var_bytes(2), 1000);
        assert!(v.var_budget(3, "a", 1000), "another peer has its own budget");
        v.peer_left(2);
        assert_eq!(v.var_bytes(2), 0);
    }

    #[test]
    fn voice_packets_are_size_limited() {
        let v = Validator::new(NetValidation { max_voice_packet: 400, ..Default::default() }, Topology::Dedicated);
        assert!(v.voice_ok(120) && v.voice_ok(400) && !v.voice_ok(401));
        assert!(Validator::new(NetValidation { max_voice_packet: 0, ..Default::default() }, Topology::Dedicated).voice_ok(1500), "0 falls back to the default");
    }

    #[test]
    fn parent_ids_must_not_loop_or_reach_other_players_objects() {
        let mut parents: HashMap<u128, u128> = HashMap::new();
        let mut owners: HashMap<u128, u64> = HashMap::new();
        owners.insert(10, 2); // mine
        owners.insert(11, 3); // someone else's
        owners.insert(12, 1); // the host's / scene
        let owner_of = |id: u128| owners.get(&id).copied().unwrap_or(0);
        let d = Validator::new(NetValidation::default(), Topology::Dedicated);
        assert_eq!(d.parent_ok(2, 1, 10, 0, owner_of, &parents), Ok(()), "detach");
        assert_eq!(d.parent_ok(2, 1, 10, 10, owner_of, &parents), Err(Reason::Malformed), "itself");
        assert_eq!(d.parent_ok(2, 1, 10, 12, owner_of, &parents), Ok(()), "a host-owned object");
        assert_eq!(d.parent_ok(2, 1, 10, 11, owner_of, &parents), Err(Reason::OwnershipDenied), "another client's object");
        assert_eq!(d.parent_ok(2, 1, 10, 99, owner_of, &parents), Ok(()), "unowned scene object");
        // A cycle: 12 under 10 already, now 10 under 12.
        parents.insert(12, 10);
        assert_eq!(d.parent_ok(2, 1, 10, 12, owner_of, &parents), Err(Reason::Malformed), "cycle");
        // A chain deeper than the cap.
        let mut deep: HashMap<u128, u128> = HashMap::new();
        for k in 100..140u128 {
            deep.insert(k, k + 1);
        }
        assert_eq!(d.parent_ok(2, 1, 5, 100, |_| 0, &deep), Err(Reason::TooLarge));
        // A listen host without the strict flag only refuses structural nonsense.
        let l = Validator::new(NetValidation::default(), Topology::ListenHost);
        parents.clear();
        assert_eq!(l.parent_ok(2, 1, 10, 11, owner_of, &parents), Ok(()));
        assert_eq!(l.parent_ok(2, 1, 10, 10, owner_of, &parents), Err(Reason::Malformed));
    }

    #[test]
    fn strikes_warn_then_kick_then_ban_by_key() {
        let mut v = v(true);
        let mut now = 0.0;
        let mut seen = Vec::new();
        // NotFinite is worth 10 points: 20 warns, 100 kicks.
        for _ in 0..11 {
            now += 0.01;
            let e = v.report(2, Reason::NotFinite, "test", now);
            if e != Escalation::None {
                seen.push(e);
            }
        }
        assert_eq!(seen, vec![Escalation::Warn, Escalation::Kick]);
        assert_eq!(v.count(2, Reason::NotFinite), 11);
        // The peer rejoins (same key, new peer id) and offends again: banned.
        v.peer_left(2);
        v.peer_joined(4, [7; 32], "Mallory");
        let mut last = Escalation::None;
        for _ in 0..11 {
            now += 0.01;
            last = v.report(4, Reason::NotFinite, "test", now);
        }
        assert_eq!(last, Escalation::Ban);
        // Another key is unaffected.
        assert_eq!(v.report(3, Reason::NotFinite, "test", now), Escalation::None);
        assert_eq!(v.score(3, now), 10.0);
    }

    #[test]
    fn scores_decay_and_drop_level_ignores_the_peer() {
        let mut v = v(true);
        for k in 0..6 {
            v.report(2, Reason::NotFinite, "", k as f64 * 0.01);
        }
        assert!(v.score(2, 0.06) >= 59.0);
        assert!(v.dropping(2, 0.06));
        assert!(!v.dropping(2, 30.0), "decayed below the drop level");
        assert!(v.score(2, 100.0) == 0.0);
        // Honest peers never go there.
        assert!(!v.dropping(3, 0.0));
    }

    #[test]
    fn disabled_validation_still_keeps_the_structural_checks() {
        let mut v = Validator::new(NetValidation { enabled: false, ..Default::default() }, Topology::Dedicated);
        v.peer_joined(2, [7; 32], "M");
        assert!(!v.strict());
        assert!((0..10_000).all(|_| v.allow(2, Limit::Rpc, 0.0)), "no rate limit");
        assert_eq!(mv(&mut v, Vec3::NAN, None, 0.0).unwrap_err().reason, Reason::NotFinite, "NaN is always refused");
        assert_eq!(v.report(2, Reason::Forged, "", 0.0), Escalation::None);
        assert!(!v.string_ok(&"x".repeat(1000)));
    }

    #[test]
    fn bans_persist_expire_and_match_any_fingerprint_spelling() {
        let dir = std::env::temp_dir().join(format!("zestengine-bans-{}-{}", std::process::id(), unix_now()));
        let path = dir.join(".server/bans.ron");
        let fp = crate::net_identity::fingerprint(&[9; 32]);
        {
            let mut b = BanList::open(&path);
            assert!(b.is_banned(&fp, 100).is_none());
            b.add(&fp, 0, "flooding", 100);
            b.add("aaaa-bbbb", 60, "temp", 100);
        }
        let b = BanList::open(&path);
        assert_eq!(b.entries().len(), 2);
        assert!(b.is_banned(&fp.to_uppercase().replace('-', " "), 1_000_000).is_some(), "permanent, any spelling");
        assert!(b.is_banned("aaaabbbb", 120).is_some());
        assert!(b.is_banned("aaaabbbb", 161).is_none(), "the temp ban ended");
        let mut b = b;
        b.prune(200);
        assert_eq!(BanList::open(&path).entries().len(), 1);
        assert!(b.remove(&fp));
        assert!(BanList::open(&path).entries().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_flood_of_reports_costs_constant_memory() {
        let mut v = v(true);
        for k in 0..100_000 {
            v.report(2, Reason::RateLimited, "", k as f64 * 1e-6);
        }
        assert_eq!(v.count(2, Reason::RateLimited), 100_000);
        assert_eq!(v.offenders.len(), 1);
        // And the log is rate limited: suppression counted, not allocated.
        assert!(v.peers[&2].suppressed[Reason::RateLimited as usize] > 99_000);
    }

    #[test]
    fn safe_paths() {
        assert!(safe_path("prefabs/a.prefab"));
        for bad in ["", "/abs", "\\abs", "a/../b", "..", "C:/x", "a\0b"] {
            assert!(!safe_path(bad), "{bad:?}");
        }
    }
}
