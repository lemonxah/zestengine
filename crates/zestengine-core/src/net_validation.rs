//! Session topology and the server-side validation settings (NET-4).
//!
//! Plain data, readable from the scene's `Network Manager` and overridable in
//! the dedicated server's `server.ron`. The rules themselves live in the
//! engine (`zestengine_engine::net_validate`); everything here has serde
//! defaults, so old scenes and configs load unchanged.

use serde::{Deserialize, Serialize};

/// Longest lag-compensation window the engine ever allows, seconds. The
/// `NetValidation::max_rewind` setting is clamped to it.
pub const HARD_MAX_REWIND: f32 = 0.9;

/// Who runs a session.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Topology {
    /// A player hosts: peers trust the host, the host may leave (host
    /// migration) and plays too.
    #[default]
    ListenHost,
    /// A dedicated server: the server is the authority for everything, there
    /// is no host migration, its identity key is pinned by clients and the
    /// strict validation rules apply to every client.
    Dedicated,
}

impl Topology {
    /// Every topology, in declaration order.
    pub const ALL: [Topology; 2] = [Self::ListenHost, Self::Dedicated];
    /// Inspector label of this topology.
    pub fn label(self) -> &'static str {
        match self {
            Self::ListenHost => "Listen host",
            Self::Dedicated => "Dedicated server",
        }
    }
    /// Stable wire encoding.
    pub fn to_u8(self) -> u8 {
        self as u8
    }
    /// Decode a wire value (1 = Dedicated, anything else = ListenHost).
    pub fn from_u8(v: u8) -> Self {
        if v == 1 { Self::Dedicated } else { Self::ListenHost }
    }
}

/// Who may send an RPC.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RpcPermission {
    /// Any peer (the behaviour of every RPC before NET-4).
    #[default]
    Anyone,
    /// Only the owner of the object the RPC is sent on (and the server).
    Owner,
    /// Only the server; a client sending it is a violation.
    Server,
}

impl RpcPermission {
    /// Every permission, in declaration order.
    pub const ALL: [RpcPermission; 3] = [Self::Anyone, Self::Owner, Self::Server];
    /// Inspector label of this permission.
    pub fn label(self) -> &'static str {
        match self {
            Self::Anyone => "Anyone",
            Self::Owner => "Owner only",
            Self::Server => "Server only",
        }
    }
}

/// One RPC's permission and limits (registered by name).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RpcRule {
    /// RPC name this rule applies to.
    pub name: String,
    /// Who may send it.
    pub permission: RpcPermission,
    /// Calls per second per peer; 0 = the general RPC limit.
    pub rate: f32,
    /// Burst size; 0 = the general RPC burst.
    pub burst: f32,
    /// Most arguments; 0 = the general limit.
    pub max_args: u32,
    /// The requester's avatar must be near the object (interaction range)
    /// and the user id in the first argument must be one of its own objects.
    pub interaction: bool,
}

impl Default for RpcRule {
    fn default() -> Self {
        Self { name: String::new(), permission: RpcPermission::Anyone, rate: 0.0, burst: 0.0, max_args: 0, interaction: false }
    }
}

/// A token bucket's refill rate (per second) and burst size.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RateSetting {
    /// Refills per second.
    pub rate: f32,
    /// Bucket size (the most that may be spent at once).
    pub burst: f32,
}

impl RateSetting {
    /// Build a token-bucket setting.
    pub const fn new(rate: f32, burst: f32) -> Self {
        Self { rate, burst }
    }
}

impl Default for RateSetting {
    fn default() -> Self {
        Self { rate: 10.0, burst: 20.0 }
    }
}

/// Per-peer token-bucket limits by traffic kind.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RateLimits {
    /// Every authenticated packet (checked first: a flood costs one bucket take).
    pub packets: RateSetting,
    /// Reliable messages (all kinds together).
    pub messages: RateSetting,
    /// RPC messages.
    pub rpc: RateSetting,
    /// Networked spawns.
    pub spawn: RateSetting,
    /// Ownership requests.
    pub ownership: RateSetting,
    /// Chat lines.
    pub chat: RateSetting,
    /// Voice packets.
    pub voice: RateSetting,
    /// `P_AVPARAM` avatar parameter packets.
    pub avparam: RateSetting,
    /// VR / avatar pose packets.
    pub avatar: RateSetting,
    /// State snapshots.
    pub snapshot: RateSetting,
    /// Animator updates.
    pub anim: RateSetting,
    /// Face-tracking packets.
    pub face: RateSetting,
    /// Variable writes.
    pub var: RateSetting,
    /// Predicted-pawn input packets.
    pub input: RateSetting,
}

impl Default for RateLimits {
    fn default() -> Self {
        Self {
            packets: RateSetting::new(1500.0, 600.0),
            messages: RateSetting::new(300.0, 300.0),
            rpc: RateSetting::new(30.0, 60.0),
            spawn: RateSetting::new(10.0, 20.0),
            ownership: RateSetting::new(4.0, 8.0),
            chat: RateSetting::new(5.0, 64.0),
            voice: RateSetting::new(100.0, 120.0),
            avparam: RateSetting::new(40.0, 40.0),
            avatar: RateSetting::new(200.0, 120.0),
            snapshot: RateSetting::new(300.0, 200.0),
            anim: RateSetting::new(60.0, 60.0),
            face: RateSetting::new(120.0, 120.0),
            var: RateSetting::new(60.0, 120.0),
            input: RateSetting::new(300.0, 200.0),
        }
    }
}

/// Escalation of repeated violations. Every rejection adds strike points to
/// the offender's player id (not its address); the score decays over time.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StrikePolicy {
    /// Points lost per second.
    pub decay_per_sec: f32,
    /// Score at which the peer is warned (a private message, once per level).
    pub warn_at: f32,
    /// Score at which everything the peer sends for gameplay is dropped
    /// until the score decays again, with a correction for what it moved.
    pub drop_at: f32,
    /// Score at which the peer is kicked (with a reason message).
    pub kick_at: f32,
    /// Kicks (within `kick_memory`) after which the next kick is a ban.
    pub ban_after_kicks: u32,
    /// Seconds a kick is remembered for the ban count.
    pub kick_memory: f32,
    /// Ban length in seconds; 0 = permanent.
    pub ban_secs: u64,
}

impl Default for StrikePolicy {
    fn default() -> Self {
        Self { decay_per_sec: 1.0, warn_at: 20.0, drop_at: 50.0, kick_at: 100.0, ban_after_kicks: 2, kick_memory: 1800.0, ban_secs: 3600 }
    }
}

/// A cap on how many live objects of one prefab a client may have spawned (NET-4).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct SpawnQuota {
    /// Prefab path; a trailing `*` matches a prefix. The first matching entry applies.
    pub prefab: String,
    /// Live spawns of it one client may have (0 = no per-client cap).
    pub per_peer: u32,
    /// Live spawns of it all clients together may have (0 = no total cap).
    pub total: u32,
}


impl SpawnQuota {
    /// True when the quota's prefab matches `path` (a trailing `*` matches
    /// the prefix).
    pub fn matches(&self, path: &str) -> bool {
        match self.prefab.strip_suffix('*') {
            Some(prefix) => path.starts_with(prefix),
            None => self.prefab == path,
        }
    }
}

/// Everything the server-side validation layer is configured with. The
/// defaults are conservative for a social VR game; every limit can be raised
/// per project (Network Manager) or per server (`server.ron`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NetValidation {
    /// Master switch for everything below except the structural limits
    /// (malformed packets, string and size caps), which always apply.
    pub enabled: bool,
    /// Apply the gameplay rules (movement budgets, VR pose limits, ownership
    /// range, spawn allow-list, owner-only RPCs) on a listen host too.
    /// Dedicated servers always apply them.
    pub strict_on_listen_host: bool,
    /// RPCs without a rule are owner-only on dedicated servers.
    pub strict_rpc_permissions: bool,
    /// Per-RPC permissions and limits.
    pub rpc_rules: Vec<RpcRule>,
    /// Prefab paths clients may spawn (trailing `*` = prefix; `clone:*`
    /// allows cloning scene templates). Empty = none on a dedicated server,
    /// anything on a listen host that allows client spawns.
    pub client_spawn_prefabs: Vec<String>,
    /// Live objects one client may have spawned at once.
    pub max_client_spawns: u32,
    /// A client spawn must be this close (m) to the client's avatar (dedicated).
    pub spawn_range: f32,
    /// Per-prefab caps on live client spawns (on top of `max_client_spawns`).
    pub spawn_quotas: Vec<SpawnQuota>,

    // World bounds and movement.
    /// Minimum world position (m) an owned object may be at.
    pub world_min: [f32; 3],
    /// Maximum world position (m) an owned object may be at.
    pub world_max: [f32; 3],
    /// Speed cap (m/s) of an owned prop (thrown objects included).
    pub prop_speed: f32,
    /// Speed cap (m/s) of an owned avatar / player root that is not predicted.
    pub avatar_speed: f32,
    /// Acceleration cap (m/s^2) of owned props; 0 = off.
    pub prop_accel: f32,
    /// Acceleration cap (m/s^2) of an owned avatar; 0 = off.
    pub avatar_accel: f32,
    /// Metres an object may be ahead of its speed budget at once (network
    /// jitter, a burst of queued packets). Anything beyond is a teleport.
    pub teleport_tolerance: f32,
    /// Largest scale component an owned object may announce.
    pub max_scale: f32,
    /// Largest value (absolute) in a predicted pawn's extra input floats.
    pub max_input_extra: f32,

    // VR poses (all relative to the avatar root, scaled by the avatar scale).
    /// Head distance from the root on the ground plane (room scale).
    pub head_radius: f32,
    /// Lowest head height (m) relative to the avatar root.
    pub head_min_height: f32,
    /// Highest head height (m) relative to the avatar root.
    pub head_max_height: f32,
    /// Hand reach from the head (arm span).
    pub arm_reach: f32,
    /// Extra metres allowed on every reach limit (tracking noise).
    pub reach_slack: f32,
    /// Other trackers (hips, feet, elbows ...) from the head.
    pub body_reach: f32,
    /// Controller speed (m/s) relative to the root.
    pub hand_speed: f32,
    /// Head speed cap (m/s) relative to the root.
    pub head_speed: f32,

    // Ownership.
    /// Reach (m) from the requester's avatar to an object it wants to own.
    pub grab_range: f32,
    /// Seconds between two ownership requests for one object by one peer.
    pub claim_cooldown: f32,
    /// An owner silent this long (s) loses a contested object.
    pub owner_silent: f32,

    // Lag compensation.
    /// Largest rewind (s), clamped to `HARD_MAX_REWIND`.
    pub max_rewind: f32,
    /// How far a timestamp may be ahead of the server clock (s).
    pub future_tolerance: f32,

    // Size limits (always enforced).
    /// Longest string (bytes) a client may send.
    pub max_string: u32,
    /// Longest chat message (bytes).
    pub max_chat: u32,
    /// Most RPC arguments.
    pub max_rpc_args: u32,
    /// Largest RPC payload (bytes).
    pub max_rpc_bytes: u32,
    /// Largest variable value (bytes).
    pub max_var_bytes: u32,
    /// All the variable bytes one client may have written at once (its budget; 0 = default).
    pub max_var_total: u32,
    /// Distinct variable keys one client may have written at once (0 = default).
    pub max_var_keys: u32,
    /// Largest voice packet (bytes) a client may send. Voice is opaque Opus to the server, so the
    /// packet size and the packet rate (`rates.voice`) bound the volume of voice traffic; loudness
    /// is the listener's mixer, not a server rule.
    pub max_voice_packet: u32,
    /// Most tracked entries a client may have.
    pub max_entries: u32,
    /// Reliable message size from a client (bytes).
    pub max_message: u32,
    /// Reliable packets a client may have received out of order at once.
    pub max_reorder: u32,
    /// Interaction range (m) for RPCs flagged `interaction`.
    pub interact_range: f32,

    /// Per-peer token-bucket limits.
    pub rates: RateLimits,
    /// Escalation of repeated violations.
    pub strikes: StrikePolicy,
}

impl Default for NetValidation {
    fn default() -> Self {
        Self {
            enabled: true,
            strict_on_listen_host: false,
            strict_rpc_permissions: false,
            rpc_rules: Vec::new(),
            client_spawn_prefabs: Vec::new(),
            max_client_spawns: 32,
            spawn_range: 15.0,
            spawn_quotas: Vec::new(),
            world_min: [-5000.0, -2000.0, -5000.0],
            world_max: [5000.0, 5000.0, 5000.0],
            prop_speed: 60.0,
            avatar_speed: 30.0,
            prop_accel: 4000.0,
            avatar_accel: 600.0,
            teleport_tolerance: 1.5,
            max_scale: 100.0,
            max_input_extra: 10_000.0,
            head_radius: 3.0,
            head_min_height: -0.5,
            head_max_height: 3.0,
            arm_reach: 1.2,
            reach_slack: 0.4,
            body_reach: 2.0,
            hand_speed: 30.0,
            head_speed: 15.0,
            grab_range: 4.0,
            claim_cooldown: 0.4,
            owner_silent: 3.0,
            max_rewind: 0.25,
            future_tolerance: 0.1,
            max_string: 256,
            max_chat: 512,
            max_rpc_args: 16,
            max_rpc_bytes: 1024,
            max_var_bytes: 64 * 1024,
            max_var_total: 1024 * 1024,
            max_var_keys: 256,
            max_voice_packet: 1500,
            max_entries: 64,
            max_message: 256 * 1024,
            max_reorder: 2048,
            interact_range: 5.0,
            rates: RateLimits::default(),
            strikes: StrikePolicy::default(),
        }
    }
}

fn pos_or(v: f32, fallback: f32) -> f32 {
    if v.is_finite() && v >= 0.0 { v } else { fallback }
}

impl NetValidation {
    /// Replace nonsense (NaN, negatives, a rewind over the hard cap) with the
    /// defaults so a hand-edited file can never switch a check off by accident.
    pub fn sanitized(mut self) -> Self {
        let d = Self::default();
        self.max_rewind = pos_or(self.max_rewind, d.max_rewind).min(HARD_MAX_REWIND);
        self.future_tolerance = pos_or(self.future_tolerance, d.future_tolerance).min(1.0);
        for (v, f) in [
            (&mut self.spawn_range, d.spawn_range),
            (&mut self.prop_speed, d.prop_speed),
            (&mut self.avatar_speed, d.avatar_speed),
            (&mut self.prop_accel, d.prop_accel),
            (&mut self.avatar_accel, d.avatar_accel),
            (&mut self.teleport_tolerance, d.teleport_tolerance),
            (&mut self.max_scale, d.max_scale),
            (&mut self.max_input_extra, d.max_input_extra),
            (&mut self.head_radius, d.head_radius),
            (&mut self.head_max_height, d.head_max_height),
            (&mut self.arm_reach, d.arm_reach),
            (&mut self.reach_slack, d.reach_slack),
            (&mut self.body_reach, d.body_reach),
            (&mut self.hand_speed, d.hand_speed),
            (&mut self.head_speed, d.head_speed),
            (&mut self.grab_range, d.grab_range),
            (&mut self.claim_cooldown, d.claim_cooldown),
            (&mut self.owner_silent, d.owner_silent),
            (&mut self.interact_range, d.interact_range),
        ] {
            *v = pos_or(*v, f);
        }
        if !self.head_min_height.is_finite() {
            self.head_min_height = d.head_min_height;
        }
        for k in 0..3 {
            if !self.world_min[k].is_finite() || !self.world_max[k].is_finite() || self.world_min[k] >= self.world_max[k] {
                self.world_min[k] = d.world_min[k];
                self.world_max[k] = d.world_max[k];
            }
        }
        // Limits of zero would reject everything: fall back to the defaults.
        for (v, f) in [
            (&mut self.max_string, d.max_string),
            (&mut self.max_chat, d.max_chat),
            (&mut self.max_rpc_args, d.max_rpc_args),
            (&mut self.max_rpc_bytes, d.max_rpc_bytes),
            (&mut self.max_var_bytes, d.max_var_bytes),
            (&mut self.max_var_total, d.max_var_total),
            (&mut self.max_var_keys, d.max_var_keys),
            (&mut self.max_voice_packet, d.max_voice_packet),
            (&mut self.max_entries, d.max_entries),
            (&mut self.max_message, d.max_message),
            (&mut self.max_reorder, d.max_reorder),
        ] {
            if *v == 0 {
                *v = f;
            }
        }
        self.max_entries = self.max_entries.min(255);
        let dr = RateLimits::default();
        let r = &mut self.rates;
        for (s, f) in [
            (&mut r.packets, dr.packets),
            (&mut r.messages, dr.messages),
            (&mut r.rpc, dr.rpc),
            (&mut r.spawn, dr.spawn),
            (&mut r.ownership, dr.ownership),
            (&mut r.chat, dr.chat),
            (&mut r.voice, dr.voice),
            (&mut r.avparam, dr.avparam),
            (&mut r.avatar, dr.avatar),
            (&mut r.snapshot, dr.snapshot),
            (&mut r.anim, dr.anim),
            (&mut r.face, dr.face),
            (&mut r.var, dr.var),
            (&mut r.input, dr.input),
        ] {
            if !s.rate.is_finite() || s.rate < 0.0 || !s.burst.is_finite() || s.burst < 1.0 {
                *s = f;
            }
        }
        let ds = StrikePolicy::default();
        let k = &mut self.strikes;
        k.decay_per_sec = pos_or(k.decay_per_sec, ds.decay_per_sec);
        k.warn_at = pos_or(k.warn_at, ds.warn_at);
        k.drop_at = pos_or(k.drop_at, ds.drop_at);
        k.kick_at = pos_or(k.kick_at, ds.kick_at);
        k.kick_memory = pos_or(k.kick_memory, ds.kick_memory);
        if k.kick_at <= 0.0 {
            k.kick_at = ds.kick_at;
        }
        k.ban_after_kicks = k.ban_after_kicks.max(1);
        self
    }

    /// The rule registered for an RPC name.
    pub fn rule(&self, name: &str) -> Option<&RpcRule> {
        self.rpc_rules.iter().find(|r| r.name == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_round_trip_and_old_files_load() {
        let v = NetValidation::default();
        let text = ron::to_string(&v).unwrap();
        let back: NetValidation = ron::from_str(&text).unwrap();
        assert_eq!(v, back);
        // A file naming one field keeps every other default.
        let partial: NetValidation = ron::from_str("(max_rewind: 0.5)").unwrap();
        assert_eq!(partial.max_rewind, 0.5);
        assert_eq!(partial.rates, RateLimits::default());
        let t: Topology = ron::from_str("Dedicated").unwrap();
        assert_eq!(t, Topology::Dedicated);
    }

    #[test]
    fn old_network_manager_files_still_load() {
        use crate::NetworkManager;
        let old: NetworkManager = ron::from_str("(dedicated: true, max_players: 8)").unwrap();
        assert!(old.is_dedicated(), "the old flag reads as a dedicated topology");
        assert_eq!(old.topology, Topology::ListenHost);
        assert_eq!(old.validation, NetValidation::default());
        let new: NetworkManager = ron::from_str("(topology: Dedicated, validation: (max_rewind: 0.1))").unwrap();
        assert!(new.is_dedicated() && new.validation.max_rewind == 0.1);
        assert!(!ron::from_str::<NetworkManager>("()").unwrap().is_dedicated());
        let text = ron::to_string(&NetworkManager::default()).unwrap();
        assert!(!text.contains("dedicated:"), "the legacy flag is no longer written: {text}");
        let back: NetworkManager = ron::from_str(&text).unwrap();
        assert_eq!(back.validation, NetValidation::default());
    }

    #[test]
    fn sanitizing_clamps_the_rewind_and_repairs_nonsense() {
        let mut v = NetValidation { max_rewind: 5.0, prop_speed: f32::NAN, grab_range: -1.0, max_entries: 9999, ..Default::default() };
        v.rates.rpc = RateSetting::new(f32::INFINITY, 0.0);
        v.world_min = [10.0, 0.0, 0.0];
        v.world_max = [0.0, 1.0, 1.0];
        let s = v.sanitized();
        let d = NetValidation::default();
        assert_eq!(s.max_rewind, HARD_MAX_REWIND);
        const { assert!(HARD_MAX_REWIND < 1.0) };
        assert_eq!(s.prop_speed, d.prop_speed);
        assert_eq!(s.grab_range, d.grab_range);
        assert_eq!(s.max_entries, 255);
        assert_eq!(s.rates.rpc, d.rates.rpc);
        assert_eq!((s.world_min[0], s.world_max[0]), (d.world_min[0], d.world_max[0]), "a reversed axis falls back; valid axes stay");
        assert_eq!((s.world_min[1], s.world_max[1]), (0.0, 1.0));
        assert_eq!(NetValidation { max_rewind: -1.0, ..Default::default() }.sanitized().max_rewind, d.max_rewind);
    }
}
