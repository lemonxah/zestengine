//! Co-editing sessions (see `docs/tasks/coediting.md`): two or more editors
//! sharing one world, with presence and voice.
//!
//! Transport and channels are [`NetSession`]'s: sealed UDP (or an in-memory
//! hub in tests), host-authoritative star, a reliable ordered channel for
//! scene ops and an unreliable one for presence. Voice rides the existing
//! voice channel. The editor world is NOT the play world, so this module
//! only borrows the session plumbing; scene ops travel as RPCs:
//!
//! - `edit.op`    (reliable)   one scene op, applied by everyone except the
//!   sender (who applied it locally before sending).
//! - `edit.log`   (reliable)   the host's op-log chunk, replayed to a peer
//!   that just joined (late-join catch-up; both editors load the same scene
//!   files, so the net only carries the delta log).
//! - `edit.pres`  (unreliable) head + hand joints + scale.
//! - voice                      Opus frames through [`NetSession::send_voice`].
//!
//! All editors are authors: the host relays every op in arrival order, which
//! is the session-wide apply order (deterministic convergence).

use std::net::SocketAddr;

use anyhow::Result;
use zestengine_core::{NetEvent, NetValue, ObjectId, RpcTarget};

use crate::net::{NetConfig, NetSession, Transport, VoicePacket};

const OP_RPC: &str = "edit.op";
const LOG_RPC: &str = "edit.log";
const PRES_RPC: &str = "edit.pres";
/// Ops per catch-up chunk (the reliable channel reassembles each message).
const LOG_CHUNK: usize = 64;
const HAND_JOINTS: usize = 26;

/// A peer's tracked hand: joint positions (world space) + validity mask.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NetHandJoints {
    pub positions: [[f32; 3]; HAND_JOINTS],
    pub valid: [bool; HAND_JOINTS],
}

/// A peer's VR presence for one frame (world space, at true scale).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PeerPresence {
    pub peer: u64,
    pub head: Option<([f32; 3], [f32; 4])>,
    pub hands: [Option<NetHandJoints>; 2],
    pub scale: f32,
}

/// One authoritative edit, applied by every editor in the same order.
/// The spawn payload is boxed: ops ride in per-frame vectors, and the common
/// variants (SetTransform/Delete) should stay small.
#[derive(Clone, Debug, PartialEq)]
pub enum EditOp {
    Spawn(Box<SpawnOp>),
    SetTransform { id: ObjectId, transform: [f32; 16] },
    Delete { id: ObjectId },
    Duplicate { id: ObjectId, new_id: ObjectId },
    Rename { id: ObjectId, name: String },
}

/// The payload of [`EditOp::Spawn`].
#[derive(Clone, Debug, PartialEq)]
pub struct SpawnOp {
    pub id: ObjectId,
    /// Editor spawn source (`zestengine_assets::ObjectSource` label).
    pub source: String,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

/// Session events drained by [`EditSession::pump`].
#[derive(Clone, Debug, PartialEq)]
pub enum EditEvent {
    Connected { peer: u64 },
    Disconnected { reason: String },
    PeerJoined { id: u64, name: String },
    PeerLeft { id: u64 },
    /// An op to apply (never the sender's own echo).
    Op { from: u64, op: EditOp },
    /// Boxed: presence carries two 26-joint hands (~700 bytes); the enum
    /// otherwise stays at a machine word.
    Presence(Box<PeerPresence>),
}

/// Session settings; the net layer's defaults cover the rest (LAN / direct
/// only in v1; the relay can come later).
#[derive(Clone, Debug)]
pub struct EditConfig {
    pub name: String,
    pub session_name: String,
    /// Host: the UDP port to listen on (0 = OS-assigned). Ignored by joiners.
    pub port: u16,
    /// Join secret (empty = open session).
    pub secret: String,
    /// Reply to LAN discovery queries on `discovery_port` (edit sessions use
    /// [`EDIT_DISCOVERY_PORT`], separate from the game browser's port).
    pub discovery: bool,
    pub discovery_port: u16,
}

/// Edit sessions answer discovery on their own port so the game browser and
/// the co-edit browser stay separate lists.
pub const EDIT_DISCOVERY_PORT: u16 = crate::net::DISCOVERY_PORT + 1;

impl Default for EditConfig {
    fn default() -> Self {
        Self {
            name: "Editor".into(),
            session_name: "Edit session".into(),
            port: 0,
            secret: String::new(),
            discovery: true,
            discovery_port: EDIT_DISCOVERY_PORT,
        }
    }
}

/// One co-editing session. Host one side, join with the other; call
/// [`EditSession::pump`] every frame on both.
pub struct EditSession {
    net: NetSession,
    /// Host: every op applied so far, replayed to late joiners.
    log: Vec<EditOp>,
    peers: Vec<(u64, String)>,
}

impl EditSession {
    /// Host over `transport` (UDP, or a [`crate::net::MemHub`] endpoint in
    /// tests). Transport binding is the caller's job. With `cfg.discovery`
    /// the session also answers LAN discovery queries on `cfg.discovery_port`.
    pub fn host_over(transport: Box<dyn Transport>, cfg: EditConfig) -> Self {
        let mut net = NetSession::host_on(transport, net_config(&cfg));
        if cfg.discovery {
            net.enable_discovery(cfg.discovery_port);
        }
        let name = cfg.name;
        let local_peer = net.local_peer();
        Self {
            net,
            log: Vec::new(),
            peers: vec![(local_peer, name)],
        }
    }

    /// Join the host at `addr` over `transport`.
    pub fn join_over(
        transport: Box<dyn Transport>,
        addr: SocketAddr,
        cfg: EditConfig,
    ) -> Result<Self> {
        let net = NetSession::join_on(transport, addr, net_config(&cfg));
        Ok(Self {
            net,
            log: Vec::new(),
            peers: Vec::new(),
        })
    }

    /// This peer's id (0 until the handshake completes on a joiner).
    pub fn local_peer(&self) -> u64 {
        self.net.local_peer()
    }

    pub fn is_host(&self) -> bool {
        self.net.is_host()
    }

    /// Everyone in the session, the local peer included.
    pub fn peers(&self) -> &[(u64, String)] {
        &self.peers
    }

    /// Apply `op` locally and broadcast it (reliable, ordered by the host).
    pub fn send_op(&mut self, op: EditOp) {
        if !self.net.is_connected() {
            return;
        }
        let bytes = encode_op(&op);
        self.net
            .send_rpc(None, OP_RPC, &[NetValue::Bytes(bytes)], RpcTarget::All, true);
        // The host's log is the session history: append before the local rpc
        // is pushed by `send_rpc`, so a peer joining this instant gets it too.
        if self.net.is_host() {
            self.log.push(op);
        }
    }

    /// Send this frame's presence (unreliable; a lost frame is just stale).
    pub fn send_presence(&mut self, presence: PeerPresence) {
        if !self.net.is_connected() {
            return;
        }
        let bytes = encode_presence(&presence);
        self.net
            .send_rpc(None, PRES_RPC, &[NetValue::Bytes(bytes)], RpcTarget::All, false);
    }

    /// Send one encoded voice frame to everyone (relayed by the host).
    pub fn send_voice(&mut self, seq: u32, payload: &[u8]) {
        self.net.send_voice(seq, payload);
    }

    /// Drain voice packets received since the last call.
    pub fn take_voice(&mut self) -> Vec<VoicePacket> {
        self.net.take_voice()
    }

    /// Run the session and drain events: peer changes, ops to apply (in the
    /// session's apply order) and the latest presence per peer.
    pub fn pump(&mut self) -> Vec<EditEvent> {
        self.net.pump();
        let view = self.net.view();
        let mut out = Vec::new();
        for e in view.events {
            match e {
                NetEvent::Connected { peer } => out.push(EditEvent::Connected { peer }),
                NetEvent::Disconnected { reason } => {
                    self.peers.clear();
                    out.push(EditEvent::Disconnected { reason });
                }
                NetEvent::PeerJoined { id, name } => {
                    if self.net.is_host() {
                        self.replay_log(id);
                    }
                    if !self.peers.iter().any(|(p, _)| *p == id) {
                        self.peers.push((id, name.clone()));
                    }
                    out.push(EditEvent::PeerJoined { id, name });
                }
                NetEvent::PeerLeft { id, .. } => {
                    self.peers.retain(|(p, _)| *p != id);
                    out.push(EditEvent::PeerLeft { id });
                }
                _ => {}
            }
        }
        for rpc in view.rpcs {
            if rpc.from == self.net.local_peer() {
                continue; // own echo (already applied locally)
            }
            match rpc.name.as_str() {
                OP_RPC => {
                    let Some(op) = rpc.args.first().and_then(|a| a.as_bytes()).and_then(decode_op) else {
                        continue;
                    };
                    if self.net.is_host() {
                        self.log.push(op.clone());
                    }
                    out.push(EditEvent::Op { from: rpc.from, op });
                }
                LOG_RPC => {
                    // A catch-up chunk; each op is idempotent, and chunks
                    // arrive in order on the reliable channel.
                    let Some(bytes) = rpc.args.first().and_then(|a| a.as_bytes()) else {
                        continue;
                    };
                    for op in decode_log(bytes) {
                        out.push(EditEvent::Op { from: rpc.from, op });
                    }
                }
                PRES_RPC => {
                    let Some(bytes) = rpc.args.first().and_then(|a| a.as_bytes()) else {
                        continue;
                    };
                    let mut presence = decode_presence(bytes);
                    presence.peer = rpc.from;
                    out.push(EditEvent::Presence(Box::new(presence)));
                }
                _ => {}
            }
        }
        out
    }

    /// Host: send the op log to a newly joined peer in ordered chunks.
    fn replay_log(&mut self, peer: u64) {
        for (part, chunk) in self.log.chunks(LOG_CHUNK).enumerate() {
            let total = self.log.len().div_ceil(LOG_CHUNK) as i64;
            let bytes = encode_log(chunk);
            self.net.send_rpc(
                None,
                LOG_RPC,
                &[NetValue::Bytes(bytes), NetValue::Int(part as i64), NetValue::Int(total)],
                RpcTarget::Peer(peer),
                true,
            );
        }
    }
}

fn net_config(cfg: &EditConfig) -> NetConfig {
    NetConfig {
        player_name: cfg.name.clone(),
        session_name: cfg.session_name.clone(),
        max_players: 8,
        discovery: cfg.discovery,
        discovery_port: cfg.discovery_port,
        secret: cfg.secret.clone(),
        ..NetConfig::default()
    }
}

// ------------------------------------------------------------- encoding

fn put_u8(b: &mut Vec<u8>, v: u8) {
    b.push(v);
}

fn put_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}

fn put_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}

fn put_u128(b: &mut Vec<u8>, v: u128) {
    b.extend_from_slice(&v.to_le_bytes());
}

fn put_f32(b: &mut Vec<u8>, v: f32) {
    b.extend_from_slice(&v.to_le_bytes());
}

fn put_str(b: &mut Vec<u8>, s: &str) {
    put_u32(b, s.len() as u32);
    b.extend_from_slice(s.as_bytes());
}

fn put_id(b: &mut Vec<u8>, id: ObjectId) {
    put_u128(b, id.raw());
}

/// One chunk of the host's op log (a list of ops, prefixed by a count).
fn encode_log(ops: &[EditOp]) -> Vec<u8> {
    let mut b = Vec::new();
    put_u32(&mut b, ops.len() as u32);
    for op in ops {
        b.extend_from_slice(&encode_op(op));
    }
    b
}

fn decode_log(bytes: &[u8]) -> Vec<EditOp> {
    let mut r = Reader { b: bytes, p: 0 };
    let n = r.u32().unwrap_or(0) as usize;
    let mut out = Vec::new();
    for _ in 0..n.min(4096) {
        let Some(op) = decode_op(&bytes[r.p..]) else {
            break;
        };
        r.p += op_len(&bytes[r.p..]);
        out.push(op);
    }
    out
}

fn encode_op(op: &EditOp) -> Vec<u8> {
    let mut b = Vec::new();
    match op {
        EditOp::Spawn(sp) => {
            put_u8(&mut b, 0);
            put_id(&mut b, sp.id);
            put_str(&mut b, &sp.source);
            for v in sp.translation {
                put_f32(&mut b, v);
            }
            for v in sp.rotation {
                put_f32(&mut b, v);
            }
            for v in sp.scale {
                put_f32(&mut b, v);
            }
        }
        EditOp::SetTransform { id, transform } => {
            put_u8(&mut b, 1);
            put_id(&mut b, *id);
            for v in transform {
                put_f32(&mut b, *v);
            }
        }
        EditOp::Delete { id } => {
            put_u8(&mut b, 2);
            put_id(&mut b, *id);
        }
        EditOp::Duplicate { id, new_id } => {
            put_u8(&mut b, 3);
            put_id(&mut b, *id);
            put_id(&mut b, *new_id);
        }
        EditOp::Rename { id, name } => {
            put_u8(&mut b, 4);
            put_id(&mut b, *id);
            put_str(&mut b, name);
        }
    }
    b
}

/// Byte length of the op at `bytes[0..]` (used to advance through a log).
fn op_len(bytes: &[u8]) -> usize {
    let Some(&tag) = bytes.first() else {
        return 0;
    };
    let mut p = 1;
    match tag {
        0 => {
            p += 16;
            let Some(n) = bytes.get(p..p + 4).map(|v| u32::from_le_bytes(v.try_into().unwrap())) else {
                return 0;
            };
            p += 4 + n as usize + 3 * 4 + 4 * 4 + 3 * 4;
        }
        1 => p += 16 + 16 * 4,
        2 => p += 16,
        3 => p += 16 + 16,
        4 => {
            p += 16;
            let Some(n) = bytes.get(p..p + 4).map(|v| u32::from_le_bytes(v.try_into().unwrap())) else {
                return 0;
            };
            p += 4 + n as usize;
        }
        _ => return 0,
    }
    p
}

fn decode_op(bytes: &[u8]) -> Option<EditOp> {
    let mut r = Reader { b: bytes, p: 0 };
    let tag = r.u8()?;
    let op = match tag {
        0 => {
            let id = r.id()?;
            let source = r.string()?;
            let mut translation = [0.0f32; 3];
            let mut rotation = [0.0f32; 4];
            let mut scale = [0.0f32; 3];
            for v in translation.iter_mut() {
                *v = r.f32()?;
            }
            for v in rotation.iter_mut() {
                *v = r.f32()?;
            }
            for v in scale.iter_mut() {
                *v = r.f32()?;
            }
            EditOp::Spawn(Box::new(SpawnOp { id, source, translation, rotation, scale }))
        }
        1 => {
            let id = r.id()?;
            let mut transform = [0.0f32; 16];
            for v in transform.iter_mut() {
                *v = r.f32()?;
            }
            EditOp::SetTransform { id, transform }
        }
        2 => EditOp::Delete { id: r.id()? },
        3 => EditOp::Duplicate { id: r.id()?, new_id: r.id()? },
        4 => EditOp::Rename { id: r.id()?, name: r.string()? },
        _ => return None,
    };
    (r.p <= r.b.len()).then_some(op)
}

fn encode_presence(p: &PeerPresence) -> Vec<u8> {
    let mut b = Vec::new();
    put_u64(&mut b, p.peer);
    match p.head {
        Some((pos, rot)) => {
            put_u8(&mut b, 1);
            for v in pos {
                put_f32(&mut b, v);
            }
            for v in rot {
                put_f32(&mut b, v);
            }
        }
        None => put_u8(&mut b, 0),
    }
    put_f32(&mut b, p.scale);
    for hand in p.hands {
        match hand {
            Some(h) => {
                put_u8(&mut b, 1);
                for pos in h.positions {
                    for v in pos {
                        put_f32(&mut b, v);
                    }
                }
                let mut mask = 0u32;
                for (i, ok) in h.valid.iter().enumerate() {
                    if *ok {
                        mask |= 1 << i;
                    }
                }
                put_u32(&mut b, mask);
            }
            None => put_u8(&mut b, 0),
        }
    }
    b
}

fn decode_presence(bytes: &[u8]) -> PeerPresence {
    let mut r = Reader { b: bytes, p: 0 };
    let peer = r.u64().unwrap_or(0);
    let head = if r.u8() == Some(1) {
        let mut pos = [0.0f32; 3];
        let mut rot = [0.0f32; 4];
        for v in pos.iter_mut() {
            *v = r.f32().unwrap_or(0.0);
        }
        for v in rot.iter_mut() {
            *v = r.f32().unwrap_or(0.0);
        }
        Some((pos, rot))
    } else {
        None
    };
    let scale = r.f32().unwrap_or(1.0);
    let mut hands = [None, None];
    for hand in hands.iter_mut() {
        if r.u8() == Some(1) {
            let mut h = NetHandJoints::default();
            for pos in h.positions.iter_mut() {
                for v in pos.iter_mut() {
                    *v = r.f32().unwrap_or(0.0);
                }
            }
            let mask = r.u32().unwrap_or(0);
            for (i, ok) in h.valid.iter_mut().enumerate() {
                *ok = mask & (1 << i) != 0;
            }
            *hand = Some(h);
        }
    }
    PeerPresence { peer, head, hands, scale }
}

/// Minimal cursor over a byte slice (mirrors the net module's reader).
struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn u8(&mut self) -> Option<u8> {
        let v = *self.b.get(self.p)?;
        self.p += 1;
        Some(v)
    }

    fn u32(&mut self) -> Option<u32> {
        let v = u32::from_le_bytes(self.b.get(self.p..self.p + 4)?.try_into().ok()?);
        self.p += 4;
        Some(v)
    }

    fn u64(&mut self) -> Option<u64> {
        let v = u64::from_le_bytes(self.b.get(self.p..self.p + 8)?.try_into().ok()?);
        self.p += 8;
        Some(v)
    }

    fn id(&mut self) -> Option<ObjectId> {
        let v = u128::from_le_bytes(self.b.get(self.p..self.p + 16)?.try_into().ok()?);
        self.p += 16;
        Some(ObjectId::from_raw(v))
    }

    fn f32(&mut self) -> Option<f32> {
        let v = f32::from_le_bytes(self.b.get(self.p..self.p + 4)?.try_into().ok()?);
        self.p += 4;
        Some(v)
    }

    fn string(&mut self) -> Option<String> {
        let n = self.u32()? as usize;
        if n > 1 << 20 {
            return None;
        }
        let s = String::from_utf8(self.b.get(self.p..self.p + n)?.to_vec()).ok()?;
        self.p += n;
        Some(s)
    }
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::MemHub;

    fn cfg(name: &str) -> EditConfig {
        EditConfig { name: name.into(), session_name: "test".into(), secret: "s3cret".into(), ..Default::default() }
    }

    fn host(hub: &MemHub) -> EditSession {
        let mut h = EditSession::host_over(Box::new(hub.endpoint()), cfg("Host"));
        h.net.use_manual_clock();
        h
    }

    fn client(hub: &MemHub, host: &EditSession, name: &str) -> EditSession {
        let addr = host.net.local_addr().unwrap();
        let mut c = EditSession::join_over(Box::new(hub.endpoint()), addr, cfg(name)).unwrap();
        c.net.use_manual_clock();
        c
    }

    /// Step every session for `frames` frames, accumulating each session's
    /// drained events (events are one-shot; `pump` delivers them exactly once).
    fn run(all: &mut [&mut EditSession], frames: usize) -> Vec<Vec<EditEvent>> {
        let mut acc: Vec<Vec<EditEvent>> = all.iter().map(|_| Vec::new()).collect();
        for _ in 0..frames {
            for (i, s) in all.iter_mut().enumerate() {
                s.net.advance(1.0 / 60.0);
                acc[i].extend(s.pump());
            }
        }
        acc
    }

    fn ops_of(events: &[EditEvent]) -> Vec<(u64, EditOp)> {
        events
            .iter()
            .filter_map(|e| match e {
                EditEvent::Op { from, op } => Some((*from, op.clone())),
                _ => None,
            })
            .collect()
    }

    fn id(n: u128) -> ObjectId {
        ObjectId::from_raw(n)
    }

    #[test]
    fn host_and_join_reach_the_same_membership() {
        let hub = MemHub::new();
        let mut h = host(&hub);
        let mut a = client(&hub, &h, "Ann");
        let mut b = client(&hub, &h, "Bob");
        let events = run(&mut [&mut h, &mut a, &mut b], 120);
        let names: Vec<&str> = h.peers().iter().map(|(_, n)| n.as_str()).collect();
        assert_eq!(names, vec!["Host", "Ann", "Bob"]);
        assert!(a.peers().iter().any(|(_, n)| n == "Bob"));
        assert!(events[2].iter().any(|e| matches!(e, EditEvent::PeerJoined { name, .. } if name == "Ann")));
        assert!(events[0].iter().any(|e| matches!(e, EditEvent::PeerJoined { name, .. } if name == "Bob")));
    }

    #[test]
    fn ops_apply_everywhere_in_order_and_never_echo_to_the_sender() {
        let hub = MemHub::new();
        let mut h = host(&hub);
        let mut a = client(&hub, &h, "Ann");
        let mut b = client(&hub, &h, "Bob");
        run(&mut [&mut h, &mut a, &mut b], 120);
        let ops = [
            EditOp::SetTransform { id: id(1), transform: [1.0f32; 16] },
            EditOp::Delete { id: id(2) },
            EditOp::Rename { id: id(3), name: "pot".into() },
        ];
        a.send_op(ops[0].clone());
        a.send_op(ops[1].clone());
        b.send_op(ops[2].clone());
        let ev = run(&mut [&mut h, &mut a, &mut b], 40);
        // Every peer sees all three ops in the host's relay order, and the
        // senders never see their own echo.
        let h_ops = ops_of(&ev[0]);
        assert!(h_ops.iter().any(|(_, o)| *o == ops[0])
            && h_ops.iter().any(|(_, o)| *o == ops[1])
            && h_ops.iter().any(|(_, o)| *o == ops[2]));
        let a_ops = ops_of(&ev[1]);
        eprintln!("[ops test] Ann (peer {}) received {:?}", a.local_peer(), a_ops);
        assert!(!a_ops.iter().any(|(_, o)| *o == ops[0]), "sender must not get its own op back");
        assert!(a_ops.iter().any(|(_, o)| *o == ops[2]), "a sees Bob's op");
        let b_ops = ops_of(&ev[2]);
        assert!(b_ops.iter().any(|(_, o)| *o == ops[0]) && b_ops.iter().any(|(_, o)| *o == ops[1]));
        assert!(!b_ops.iter().any(|(_, o)| *o == ops[2]), "sender must not get its own op back");
    }

    #[test]
    fn late_joiner_receives_the_hosts_op_log() {
        let hub = MemHub::new();
        let mut h = host(&hub);
        let mut a = client(&hub, &h, "Ann");
        run(&mut [&mut h, &mut a], 120);
        a.send_op(EditOp::Spawn(Box::new(SpawnOp {
            id: id(9),
            source: "Primitive".into(),
            translation: [1.0, 2.0, 3.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
        })));
        a.send_op(EditOp::SetTransform { id: id(9), transform: [2.0f32; 16] });
        run(&mut [&mut h, &mut a], 40);
        // Bob joins after the ops happened; the host replays the log.
        let mut b = client(&hub, &h, "Bob");
        let ev = run(&mut [&mut h, &mut a, &mut b], 120);
        let b_ops = ops_of(&ev[2]);
        assert!(
            b_ops.iter().any(|(_, o)| matches!(o, EditOp::Spawn(sp) if sp.id.raw() == 9)),
            "late joiner must get the spawn from the log: {b_ops:?}"
        );
        assert!(
            b_ops.iter().any(|(_, o)| matches!(o, EditOp::SetTransform { id, .. } if id.raw() == 9)),
            "late joiner must get the transform from the log: {b_ops:?}"
        );
    }

    #[test]
    fn presence_and_voice_flow_between_peers() {
        let hub = MemHub::new();
        let mut h = host(&hub);
        let mut a = client(&hub, &h, "Ann");
        let mut b = client(&hub, &h, "Bob");
        run(&mut [&mut h, &mut a, &mut b], 120);
        let presence = PeerPresence { peer: 0, head: Some(([0.1, 1.5, 0.2], [0.0; 4])), hands: [None; 2], scale: 0.9 };
        a.send_presence(presence);
        a.send_voice(7, &[1, 2, 3, 4]);
        let ev = run(&mut [&mut h, &mut a, &mut b], 40);
        for (i, peer) in [(0usize, "host"), (2, "Bob")] {
            assert!(
                ev[i].iter().any(|e| matches!(e, EditEvent::Presence(p) if p.peer == a.local_peer() && p.scale == 0.9)),
                "{peer} must receive Ann's presence"
            );
        }
        let voice = b.take_voice();
        assert!(!voice.is_empty(), "Bob must receive Ann's voice packet");
        let a_echo = a.take_voice();
        assert!(a_echo.is_empty(), "the sender must not hear her own voice");
    }

    #[test]
    fn op_codec_round_trips() {
        let ops = [
            EditOp::Spawn(Box::new(SpawnOp {
                id: id(7),
                source: "Water".into(),
                translation: [1.5, -2.0, 0.25],
                rotation: [0.0, 1.0, 0.0, 0.0],
                scale: [2.0; 3],
            })),
            EditOp::SetTransform { id: id(7), transform: [3.5f32; 16] },
            EditOp::Delete { id: id(8) },
            EditOp::Duplicate { id: id(8), new_id: id(9) },
            EditOp::Rename { id: id(9), name: "pond".into() },
        ];
        for op in &ops {
            assert_eq!(decode_op(&encode_op(op)), Some(op.clone()), "round trip {op:?}");
        }
        let log = encode_log(&ops);
        assert_eq!(decode_log(&log), ops.to_vec());
    }

    #[test]
    fn presence_codec_round_trips() {
        let mut hand = NetHandJoints::default();
        hand.positions[0] = [1.0, 2.0, 3.0];
        hand.valid[0] = true;
        hand.valid[25] = true;
        let presence = PeerPresence {
            peer: 42,
            head: Some(([0.0, 1.6, 0.0], [0.0, 0.0, 0.0, 1.0])),
            hands: [Some(hand), None],
            scale: 1.3,
        };
        assert_eq!(decode_presence(&encode_presence(&presence)), presence);
    }
}
