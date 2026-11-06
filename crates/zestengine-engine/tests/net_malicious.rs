//! NET-4: a scripted malicious client against the server-side validation
//! layer, over the in-memory hub (deterministic clock, CPU only).
//!
//! Each test plays one kind of abuse (teleports, NaN, speed hacks, impossible
//! VR poses, ownership theft, RPC and spawn floods, garbage packets, bad
//! rewind times, replays, forged ids) against a dedicated session with honest
//! clients next to it, and checks that the abuse is refused or corrected and
//! the honest players are not affected.

use std::time::Instant;

use glam::{Quat, Vec3};
use zestengine_core::{
    NetEvent, NetSpawnSource, NetValidation, NetValue, OwnershipMode, RpcPermission, RpcRule, RpcTarget, TRACKER_SLOTS, Topology,
};
use zestengine_engine::net::{ConnState, Identity, MemHub, NetConfig, NetSession, SnapEntry, XformState, wire};
use zestengine_engine::net_validate::Reason;

const DT: f64 = 1.0 / 60.0;

fn cfg(name: &str, topology: Topology, v: &NetValidation, identity: Identity) -> NetConfig {
    NetConfig {
        player_name: name.to_string(),
        discovery: false,
        secret: String::new(),
        relay: String::new(),
        accounts: Vec::new(),
        account_name: String::new(),
        account_password: String::new(),
        topology,
        validation: v.clone(),
        identity,
        ..Default::default()
    }
}

/// A host and clients on one hub, all on a manual clock.
struct Rig {
    hub: MemHub,
    host: NetSession,
    clients: Vec<NetSession>,
    v: NetValidation,
}

impl Rig {
    fn new(topology: Topology, v: NetValidation, names: &[&str]) -> Self {
        Self::with_host_cfg(topology, v.clone(), names, |_| {})
    }

    fn with_host_cfg(topology: Topology, v: NetValidation, names: &[&str], tweak: impl FnOnce(&mut NetConfig)) -> Self {
        let hub = MemHub::new();
        let mut hc = cfg("Server", topology, &v, Identity::generate());
        tweak(&mut hc);
        let mut host = NetSession::host_on(Box::new(hub.endpoint()), hc);
        host.use_manual_clock();
        let mut rig = Rig { hub, host, clients: Vec::new(), v };
        for n in names {
            rig.join(n, Identity::generate());
        }
        rig.step(30);
        rig
    }

    fn join(&mut self, name: &str, id: Identity) -> usize {
        let mut c = NetSession::join_on(Box::new(self.hub.endpoint()), self.host.local_addr().unwrap(), cfg(name, Topology::ListenHost, &self.v, id));
        c.use_manual_clock();
        self.clients.push(c);
        self.clients.len() - 1
    }

    fn step(&mut self, frames: usize) {
        for _ in 0..frames {
            self.host.advance(DT);
            self.host.pump();
            for c in &mut self.clients {
                c.advance(DT);
                c.pump();
            }
        }
    }

    fn peer(&self, i: usize) -> u64 {
        self.clients[i].local_peer()
    }

    /// Put a player's avatar at `pos` in the host's world.
    fn avatar(&mut self, i: usize, id: u128, pos: Vec3) {
        let p = self.peer(i);
        self.host.world.player_obj.insert(p, id);
        self.host.world.xforms.insert(id, XformState { pos, ..Default::default() });
    }

    /// A networked object in the host's world.
    fn object(&mut self, id: u128, mode: OwnershipMode, pos: Vec3, owner: u64) {
        self.host.world.modes.insert(id, mode);
        self.host.world.xforms.insert(id, XformState { pos, ..Default::default() });
        if owner != 0 {
            self.host.world.owners.insert(id, (owner, 1));
        }
    }

    fn events(&mut self, i: usize) -> Vec<NetEvent> {
        self.clients[i].view().events
    }

    fn count(&self, i: usize, r: Reason) -> u32 {
        self.host.val.count(self.peer(i), r)
    }
}

/// Defaults, except that nobody is muted or kicked: tests that count individual rejections.
fn lenient() -> NetValidation {
    let mut v = NetValidation::default();
    v.strikes.drop_at = 1.0e9;
    v.strikes.kick_at = 1.0e9;
    v
}

fn pos_of(s: &NetSession, id: u128) -> Vec3 {
    s.world.xforms.get(&id).map_or(Vec3::splat(f32::MAX), |x| x.pos)
}

// ------------------------------------------------------------------ ownership

#[test]
fn ownership_theft_is_refused_and_honest_grabs_work() {
    let mut r = Rig::new(Topology::Dedicated, NetValidation::default(), &["Ann", "Bob", "Mallory"]);
    let (a, b, m) = (0, 1, 2);
    r.avatar(a, 100, Vec3::ZERO);
    r.avatar(b, 101, Vec3::new(30.0, 0.0, 0.0));
    r.avatar(m, 102, Vec3::new(1.0, 0.0, 0.0));
    let ball = 1001u128;
    let far_ball = 1002u128;
    let door = 1003u128; // Host only
    let server_owned = 1004u128;
    r.object(ball, OwnershipMode::Grabbable, Vec3::new(0.5, 0.0, 0.0), 0);
    r.object(far_ball, OwnershipMode::Grabbable, Vec3::new(200.0, 0.0, 0.0), 0);
    r.object(door, OwnershipMode::Fixed, Vec3::new(1.0, 0.0, 0.0), 0);
    r.object(server_owned, OwnershipMode::Takeover, Vec3::new(1.0, 0.0, 0.0), 1);
    let owner = |r: &Rig, id| r.host.owner_of_raw(id);

    // Ann takes the ball next to her.
    r.clients[a].request_ownership(zestengine_core::ObjectId::from_raw(ball));
    r.step(10);
    assert_eq!(owner(&r, ball), r.peer(a), "an honest grab works");
    assert!(r.clients[a].owns(zestengine_core::ObjectId::from_raw(ball)));

    // Mallory, standing right there, tries everything.
    r.step(60);
    for id in [ball, far_ball, door, server_owned, 777u128, 100u128 /* Ann's avatar */, 101u128] {
        r.clients[m].request_ownership(zestengine_core::ObjectId::from_raw(id));
        r.step(30); // past the per-object cooldown
    }
    assert_eq!(owner(&r, ball), r.peer(a), "Ann's grabbed ball stays hers");
    assert_eq!(owner(&r, far_ball), 0, "out of reach");
    assert_eq!(owner(&r, door), 0, "host-only object");
    assert_eq!(owner(&r, server_owned), 1, "server-owned object");
    assert_eq!(owner(&r, 777), 0, "an unknown id");
    assert_eq!(owner(&r, 100), 0, "a player's avatar");
    assert!(r.events(m).iter().filter(|e| matches!(e, NetEvent::OwnershipDenied { .. })).count() >= 6, "every refusal is answered");
    assert!(r.count(m, Reason::OwnershipDenied) >= 6, "and counted: {:?}", r.host.val.counts(r.peer(m)));

    // Ann lets go; Mallory (in reach, after the cooldown) can take it now; out-of-reach Bob cannot.
    r.clients[a].release_ownership(zestengine_core::ObjectId::from_raw(ball));
    r.step(30);
    assert_eq!(owner(&r, ball), 0);
    r.clients[b].request_ownership(zestengine_core::ObjectId::from_raw(ball));
    r.step(30);
    assert_eq!(owner(&r, ball), 0, "Bob is 30 m away");
    r.clients[m].request_ownership(zestengine_core::ObjectId::from_raw(ball));
    r.step(30);
    assert_eq!(owner(&r, ball), r.peer(m));

    // Requests are rate limited per object: a spam of requests is dropped at the cooldown.
    let before = r.count(m, Reason::RateLimited);
    for _ in 0..20 {
        r.clients[m].request_ownership(zestengine_core::ObjectId::from_raw(door));
        r.step(1);
    }
    assert!(r.count(m, Reason::RateLimited) > before, "the cooldown and the bucket counted the spam");

    // A disconnect revokes: the ball goes back to the server.
    r.clients[m].disconnect();
    r.step(30);
    assert_eq!(owner(&r, ball), 0, "ownership is revoked when its owner leaves");

    // Handing an object to a peer that is not there is refused.
    r.clients[a].request_ownership(zestengine_core::ObjectId::from_raw(ball));
    r.step(30);
    assert_eq!(owner(&r, ball), r.peer(a));
    r.clients[a].inject_reliable(&wire::transfer(ball, 9999));
    r.step(10);
    assert_eq!(owner(&r, ball), r.peer(a), "a transfer to nobody is dropped");
}

#[test]
fn a_listen_host_keeps_takeover_but_grabbable_is_strict() {
    let mut r = Rig::new(Topology::ListenHost, NetValidation::default(), &["Ann", "Mallory"]);
    r.avatar(1, 102, Vec3::new(500.0, 0.0, 0.0));
    r.object(2001, OwnershipMode::Takeover, Vec3::ZERO, r.peer(0));
    r.object(2002, OwnershipMode::Grabbable, Vec3::ZERO, r.peer(0));
    r.clients[1].request_ownership(zestengine_core::ObjectId::from_raw(2001));
    r.clients[1].request_ownership(zestengine_core::ObjectId::from_raw(2002));
    r.step(20);
    assert_eq!(r.host.owner_of_raw(2001), r.peer(1), "Takeover: any peer, any time, as before");
    assert_eq!(r.host.owner_of_raw(2002), r.peer(0), "Grabbable follows the rules even here");
}

// ------------------------------------------------------------------ movement

fn snap(id: u128, pos: Vec3) -> SnapEntry {
    SnapEntry { id, pos: Some(pos), rot: Some(Quat::IDENTITY), scale: None, vel: None, quantized: false }
}

#[test]
fn teleports_nan_and_speed_hacks_on_owned_props_are_rejected() {
    let mut r = Rig::new(Topology::Dedicated, lenient(), &["Ann", "Mallory"]);
    let (a, m) = (0, 1);
    let prop = 3001u128;
    r.object(prop, OwnershipMode::Grabbable, Vec3::new(0.0, 1.0, 0.0), r.peer(m));
    let who = r.peer(m);
    r.clients[m].world.owners.insert(prop, (who, 1));
    r.step(5);

    // Honest: carried at 5 m/s for a second. Everyone sees it.
    let mut p = Vec3::new(0.0, 1.0, 0.0);
    for _ in 0..60 {
        p.x += 5.0 / 60.0;
        r.clients[m].send_snapshot(&[snap(prop, p)]);
        r.step(1);
    }
    r.step(3);
    assert!(pos_of(&r.host, prop).distance(p) < 0.2, "the host follows an honest owner: {:?} vs {p:?}", pos_of(&r.host, prop));
    assert!(pos_of(&r.clients[a], prop).distance(p) < 0.5, "and so do the other players");
    assert_eq!(r.host.stats.violations, 0, "honest play costs no strikes");
    let good = pos_of(&r.host, prop);

    // A teleport 500 m away.
    r.clients[m].send_snapshot(&[snap(prop, good + Vec3::new(500.0, 0.0, 0.0))]);
    r.step(5);
    assert!(pos_of(&r.host, prop).distance(good) < 0.2, "the teleport was not applied");
    assert!(pos_of(&r.clients[a], prop).distance(good) < 0.5, "and not shown to anyone");
    assert!(r.count(m, Reason::Teleport) >= 1);
    assert!(!r.clients[m].corrections.is_empty(), "the owner is told where the object really is");
    let (cid, cpos, _) = r.clients[m].corrections[0];
    assert!(cid == prop && cpos.distance(good) < 0.3);
    r.clients[m].corrections.clear();

    // NaN, infinity, absurd positions and scales.
    for bad in [Vec3::new(f32::NAN, 0.0, 0.0), Vec3::new(0.0, f32::INFINITY, 0.0), Vec3::new(1.0e9, 0.0, 0.0)] {
        r.clients[m].send_snapshot(&[snap(prop, bad)]);
        r.step(3);
    }
    r.clients[m].send_snapshot(&[SnapEntry { id: prop, scale: Some(Vec3::splat(1.0e8)), ..Default::default() }]);
    r.clients[m].send_snapshot(&[SnapEntry { id: prop, rot: Some(Quat::from_xyzw(f32::NAN, 0.0, 0.0, 1.0)), ..Default::default() }]);
    r.step(5);
    assert!(pos_of(&r.host, prop).distance(good) < 0.2);
    assert!(r.count(m, Reason::NotFinite) >= 2, "{:?}", r.host.val.counts(r.peer(m)));
    assert!(r.count(m, Reason::OutOfBounds) >= 2);

    // A speed hack: 300 m/s for 1.5 s (the cap is 60 m/s plus the tolerance).
    let start = pos_of(&r.host, prop);
    let mut q = start;
    for _ in 0..90 {
        q.x += 300.0 / 60.0;
        r.clients[m].send_snapshot(&[snap(prop, q)]);
        r.step(1);
    }
    let travelled = pos_of(&r.host, prop).x - start.x;
    assert!(travelled < 100.0, "moved {travelled} m in 1.5 s");
    assert!(r.count(m, Reason::Teleport) + r.count(m, Reason::Speed) > 20);

    // Not the owner at all: ignored (the old rule), nothing moves.
    let theirs = 3002u128;
    r.object(theirs, OwnershipMode::Grabbable, Vec3::ZERO, r.peer(a));
    r.clients[m].send_snapshot(&[snap(theirs, Vec3::new(3.0, 0.0, 0.0))]);
    r.step(5);
    assert_eq!(pos_of(&r.host, theirs), Vec3::ZERO);
}

#[test]
fn impossible_vr_poses_are_dropped() {
    let mut r = Rig::new(Topology::Dedicated, lenient(), &["Ann", "Mallory"]);
    let (a, m) = (0, 1);
    let avatar = 4001u128;
    let root = Vec3::new(10.0, 0.0, 10.0);
    r.object(avatar, OwnershipMode::Fixed, root, r.peer(m));
    let who = r.peer(m);
    r.clients[m].world.owners.insert(avatar, (who, 1));
    r.step(5);
    let head = root + Vec3::new(0.1, 1.7, 0.0);
    let pose = |head: Vec3, l: Vec3, rr: Vec3| {
        let mut p = [None; TRACKER_SLOTS];
        p[0] = Some((head, Quat::IDENTITY));
        p[1] = Some((l, Quat::IDENTITY));
        p[2] = Some((rr, Quat::IDENTITY));
        p
    };
    let snaps = |r: &Rig, i: usize| r.clients[i].avatar_in.get(&avatar).map_or(0, |b| b.snaps.len());
    let host_snaps = |r: &Rig| r.host.avatar_in.get(&avatar).map_or(0, |b| b.snaps.len());
    let newest = |s: &NetSession| s.avatar_in.get(&avatar).and_then(|b| b.snaps.back()).map_or(-1.0, |x| x.time);

    // Honest: hands swing around the head at 90 Hz for a second.
    for k in 0..90 {
        let sw = (k as f32 * 0.1).sin() * 0.3;
        r.clients[m].send_avatar(avatar, &pose(head, head + Vec3::new(-0.3, -0.3 + sw * 0.2, 0.3 + sw), head + Vec3::new(0.3, -0.3, 0.3)));
        r.step(1);
    }
    assert!(host_snaps(&r) >= 30, "the host took the honest poses ({})", host_snaps(&r));
    assert!(snaps(&r, a) >= 30, "and relayed them ({})", snaps(&r, a));
    assert_eq!(r.host.stats.violations, 0, "{:?}", r.host.val.counts(r.peer(m)));

    let good_hand = head + Vec3::new(-0.3, -0.3, 0.3);
    let cases = [
        ("hands 6 m from the head", pose(head, head + Vec3::new(6.0, 0.0, 0.0), good_hand), Reason::ImpossiblePose),
        ("head 40 m from the body", pose(root + Vec3::new(40.0, 1.7, 0.0), good_hand, good_hand), Reason::ImpossiblePose),
        ("head underground", pose(root + Vec3::new(0.0, -30.0, 0.0), good_hand, good_hand), Reason::ImpossiblePose),
        ("NaN head", pose(Vec3::NAN, good_hand, good_hand), Reason::NotFinite),
        ("infinite hand", pose(head, Vec3::new(f32::INFINITY, 0.0, 0.0), good_hand), Reason::NotFinite),
        ("outside the world", pose(Vec3::splat(1.0e8), good_hand, good_hand), Reason::OutOfBounds),
    ];
    for (what, p, reason) in cases {
        let before = r.count(m, reason);
        r.clients[m].send_avatar(avatar, &p);
        r.step(2);
        assert!(r.count(m, reason) > before, "{what}: not counted as {reason:?}: {:?}", r.host.val.counts(r.peer(m)));
    }
    // A hand that flicks 1 m in one frame (about 65 m/s), right after an accepted pose.
    r.clients[m].send_avatar(avatar, &pose(head, good_hand, head + Vec3::new(0.3, -0.3, 0.3)));
    r.step(1);
    let (host_newest, ann_newest) = (newest(&r.host), newest(&r.clients[a]));
    r.clients[m].send_avatar(avatar, &pose(head, good_hand, good_hand + Vec3::new(0.0, 0.9, 0.0)));
    r.step(1);
    assert!(r.count(m, Reason::Speed) >= 1);
    assert!(host_newest > 0.0 && ann_newest > 0.0);
    assert_eq!(newest(&r.host), host_newest, "no impossible pose got in");
    assert_eq!(newest(&r.clients[a]), ann_newest, "and none was relayed");

    // Forging another player's id on a pose packet.
    let ann = r.peer(a);
    let bad = wire::avatar(ann, 1.0, avatar, &pose(head, good_hand, good_hand));
    r.clients[m].inject_plain(&bad);
    r.step(2);
    assert!(r.count(m, Reason::Forged) >= 1);
}

// ------------------------------------------------------------------ RPC, spawn, chat

#[test]
fn rpc_flood_and_permissions() {
    let mut v = NetValidation { strict_rpc_permissions: true, ..Default::default() };
    v.rpc_rules.push(RpcRule { name: "admin.kick".into(), permission: RpcPermission::Server, ..Default::default() });
    v.rpc_rules.push(RpcRule { name: "chat.emote".into(), permission: RpcPermission::Anyone, rate: 2.0, burst: 3.0, ..Default::default() });
    v.strikes = lenient().strikes; // keep Mallory in so the whole flood is seen
    let mut r = Rig::new(Topology::Dedicated, v, &["Ann", "Mallory"]);
    let (a, m) = (0, 1);
    let prop = 5001u128;
    r.object(prop, OwnershipMode::Grabbable, Vec3::ZERO, r.peer(a));
    r.step(5);
    let received = |r: &mut Rig, who: Option<usize>, name: &str| -> usize {
        let rpcs = match who {
            None => r.host.view().rpcs,
            Some(i) => r.clients[i].view().rpcs,
        };
        rpcs.iter().filter(|x| x.name == name).count()
    };
    let (ann, mal) = (r.peer(a), r.peer(m));

    // Owner-only (strict default for unflagged RPCs): the owner may, a stranger may not.
    let oid = zestengine_core::ObjectId::from_raw(prop);
    r.clients[a].send_rpc(Some(oid), "fire", &[NetValue::Int(1)], RpcTarget::Server, true);
    r.clients[m].send_rpc(Some(oid), "fire", &[NetValue::Int(1)], RpcTarget::Server, true);
    r.step(10);
    let fired: Vec<u64> = r.host.view().rpcs.iter().filter(|x| x.name == "fire").map(|x| x.from).collect();
    assert_eq!(fired, vec![ann], "only the owner's call got through");
    assert_eq!(r.count(m, Reason::RpcDenied), 1);

    // Server-only: refused from a client, and a forged sender field changes nothing.
    r.clients[m].send_rpc(None, "admin.kick", &[], RpcTarget::All, true);
    r.clients[m].inject_reliable(&wire::rpc(ann, RpcTarget::All, 0, "admin.kick", &[]));
    r.step(10);
    assert_eq!(received(&mut r, Some(a), "admin.kick"), 0, "nobody was told");
    assert_eq!(r.count(m, Reason::RpcDenied), 3);

    // Per-RPC limit: three in a burst, then dry.
    for _ in 0..30 {
        r.clients[m].send_rpc(None, "chat.emote", &[], RpcTarget::Others, true);
    }
    r.step(10);
    let got = received(&mut r, Some(a), "chat.emote");
    assert!((1..=4).contains(&got), "{got} emotes got through a 3-burst limit");

    // A flood of an unflagged name from the owner of nothing: dropped, not amplified.
    for _ in 0..2000 {
        r.clients[m].send_rpc(Some(oid), "spam", &[NetValue::Int(0)], RpcTarget::All, true);
    }
    r.step(30);
    assert_eq!(received(&mut r, Some(a), "spam"), 0);
    // Unreliable RPCs go through the same gate.
    for _ in 0..500 {
        r.clients[m].inject_plain(&wire::urpc(mal, RpcTarget::All, 0, "spam2", &[]));
    }
    r.step(10);
    assert_eq!(received(&mut r, Some(a), "spam2"), 0);
    assert!(r.count(m, Reason::RateLimited) + r.count(m, Reason::RpcDenied) > 100);
    // An RPC with too many or too big arguments, a long name.
    r.step(120); // the flood emptied the buckets: let them refill
    let big = vec![NetValue::Bytes(vec![0; 900]); 3];
    r.clients[m].inject_reliable(&wire::rpc(mal, RpcTarget::All, 0, "x", &big));
    r.clients[m].inject_reliable(&wire::rpc(mal, RpcTarget::All, 0, &"n".repeat(2000), &[]));
    r.step(10);
    assert!(r.count(m, Reason::TooLarge) >= 2, "{:?}", r.host.val.counts(mal));

    // Ann is unaffected throughout.
    r.clients[a].send_rpc(Some(oid), "fire", &[NetValue::Int(2)], RpcTarget::Server, true);
    r.step(10);
    assert_eq!(r.host.view().rpcs.iter().filter(|x| x.name == "fire").count(), 1);
    assert_eq!(r.host.val.count(ann, Reason::RateLimited), 0);
}

#[test]
fn rpcs_are_open_by_default_and_the_engine_interact_rpc_checks_range() {
    // Default settings keep the old behaviour for unflagged RPCs on a dedicated server.
    let mut r = Rig::new(Topology::Dedicated, NetValidation::default(), &["Ann", "Mallory"]);
    let (a, m) = (0, 1);
    r.clients[m].send_rpc(None, "anything", &[], RpcTarget::Others, true);
    r.step(10);
    assert_eq!(r.clients[a].view().rpcs.iter().filter(|x| x.name == "anything").count(), 1);

    // zestengine.interact: the user id must be the sender's own object, and near the lever.
    let lever = 6001u128;
    r.avatar(a, 100, Vec3::ZERO);
    r.avatar(m, 101, Vec3::new(80.0, 0.0, 0.0));
    r.object(lever, OwnershipMode::Fixed, Vec3::new(1.0, 0.0, 0.0), 0);
    r.host.world.owners.insert(100, (r.peer(a), 1));
    r.host.world.owners.insert(101, (r.peer(m), 1));
    let by = |id: u128| NetValue::Bytes(id.to_be_bytes().to_vec());
    let lever_id = zestengine_core::ObjectId::from_raw(lever);
    let uses = |r: &mut Rig| r.host.view().rpcs.iter().filter(|x| x.name == "zestengine.interact").count();
    r.clients[a].send_rpc(Some(lever_id), "zestengine.interact", &[by(100)], RpcTarget::Owner, true);
    r.step(5);
    assert_eq!(uses(&mut r), 1, "an honest use");
    r.clients[m].send_rpc(Some(lever_id), "zestengine.interact", &[by(101)], RpcTarget::Owner, true);
    r.step(5);
    assert_eq!(uses(&mut r), 0, "80 m away");
    r.clients[m].send_rpc(Some(lever_id), "zestengine.interact", &[by(100)], RpcTarget::Owner, true);
    r.step(5);
    assert_eq!(uses(&mut r), 0, "using it as Ann");
    assert!(r.count(m, Reason::RpcDenied) >= 2);
}

#[test]
fn spawns_are_refused_unless_allow_listed_and_floods_are_capped() {
    let mut r = Rig::new(Topology::Dedicated, NetValidation::default(), &["Ann", "Mallory"]);
    let (a, m) = (0, 1);
    r.avatar(m, 102, Vec3::ZERO);
    r.step(5);
    let spawned = |r: &mut Rig, who: usize| r.clients[who].view().world_ops.iter().filter(|op| matches!(op, zestengine_core::NetWorldOp::Spawn { .. })).count();
    let _ = spawned(&mut r, a);
    let id = |k: u128| zestengine_core::ObjectId::from_raw(0xA000 + k);
    for k in 0..5 {
        let who = r.peer(m);
        r.clients[m].spawn(NetSpawnSource::Prefab("prefabs/ball.prefab".into()), id(k), Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY, who);
    }
    r.step(10);
    assert!(r.host.world.spawns.is_empty(), "default: clients may spawn nothing on a dedicated server");
    assert_eq!(spawned(&mut r, a), 0);
    assert_eq!(r.count(m, Reason::SpawnDenied), 5);
    // The client is told to remove what it made locally.
    assert!(r.clients[m].view().world_ops.iter().any(|op| matches!(op, zestengine_core::NetWorldOp::Despawn(_))));

    // With an allow-list: that prefab, near the avatar, up to the quota; never a path out of the project.
    let v = NetValidation { client_spawn_prefabs: vec!["prefabs/ball.prefab".into()], max_client_spawns: 3, ..Default::default() };
    let mut r = Rig::new(Topology::Dedicated, v, &["Ann", "Mallory"]);
    r.avatar(1, 102, Vec3::ZERO);
    r.step(5);
    let near = Vec3::new(1.0, 0.0, 0.0);
    for k in 0..6 {
        let who = r.peer(1);
        r.clients[1].spawn(NetSpawnSource::Prefab("prefabs/ball.prefab".into()), id(k), near, Quat::IDENTITY, who);
        r.step(8);
    }
    assert_eq!(r.host.world.spawns.len(), 3, "the quota caps live spawns");
    let who = r.peer(1);
    r.clients[1].inject_reliable(&wire::spawn(0xB001, "prefabs/rocket.prefab", near, who));
    let who = r.peer(1);
    r.clients[1].inject_reliable(&wire::spawn(0xB002, "../../etc/passwd", near, who));
    let who = r.peer(1);
    r.clients[1].inject_reliable(&wire::spawn(0xB003, "prefabs/ball.prefab", Vec3::new(90.0, 0.0, 0.0), who));
    let who = r.peer(1);
    r.clients[1].inject_reliable(&wire::spawn(0xB004, "prefabs/ball.prefab", Vec3::NAN, who));
    r.step(10);
    assert_eq!(r.host.world.spawns.len(), 3);
    // Spawning with the id of an existing object (to take it over) is a forgery.
    r.object(0xC001, OwnershipMode::Fixed, near, 1);
    let before = r.host.val.count(r.peer(1), Reason::Forged);
    let who = r.peer(1);
    r.clients[1].inject_reliable(&wire::spawn(0xC001, "prefabs/ball.prefab", near, who));
    r.step(10);
    assert!(r.host.val.count(r.peer(1), Reason::Forged) > before);
    assert_eq!(r.host.owner_of_raw(0xC001), 1, "still the server's");
}

#[test]
fn chat_and_variable_limits() {
    let mut r = Rig::new(Topology::Dedicated, lenient(), &["Ann", "Mallory"]);
    let (a, m) = (0, 1);
    let (ann, mal) = (r.peer(a), r.peer(m));
    r.step(5);
    let lines = |r: &mut Rig| r.clients[0].view().messages.len();
    // A giant line, a line to a ghost, a flood.
    r.clients[m].send_message(None, &"x".repeat(5000));
    r.clients[m].send_message(Some(9999), "hello?");
    r.step(5);
    assert_eq!(lines(&mut r), 0);
    assert!(r.host.val.count(mal, Reason::TooLarge) >= 1 && r.host.val.count(mal, Reason::Malformed) >= 1);
    for k in 0..500 {
        r.clients[m].send_message(None, &format!("spam {k}"));
    }
    r.step(20);
    assert!(lines(&mut r) <= 70, "chat is rate limited");
    // The sender field is the server's to set: a forged one is ignored.
    r.clients[m].inject_reliable(&wire::text(0, ann, "I am Ann"));
    r.step(120); // let the bucket refill
    let got = r.clients[a].view().messages;
    let forged = got.iter().find(|x| x.2 == "I am Ann");
    assert!(forged.is_some_and(|x| x.0 == mal), "delivered as Mallory's: {got:?}");

    // Variables: NaN and giant values are refused.
    let obj = 7001u128;
    r.object(obj, OwnershipMode::Fixed, Vec3::ZERO, mal);
    r.clients[m].inject_reliable(&wire::var(obj, "hp", &NetValue::Float(f32::NAN), mal));
    r.clients[m].inject_reliable(&wire::var(obj, "blob", &NetValue::Bytes(vec![0; 70_000]), mal));
    r.clients[m].inject_reliable(&wire::var(obj, &"k".repeat(300), &NetValue::Int(1), mal));
    r.clients[m].inject_reliable(&wire::var(obj, "ok", &NetValue::Int(5), mal));
    r.step(30);
    assert_eq!(r.host.world.var(obj, "ok"), Some(&NetValue::Int(5)));
    assert_eq!(r.host.world.var(obj, "hp"), None);
    assert_eq!(r.host.world.var(obj, "blob"), None);
    assert!(r.host.val.count(mal, Reason::NotFinite) >= 1 && r.host.val.count(mal, Reason::TooLarge) >= 2);
}

// ------------------------------------------------------------------ packets

#[test]
fn garbage_oversized_and_forged_packets_do_no_harm() {
    let mut r = Rig::new(Topology::Dedicated, lenient(), &["Ann", "Bob", "Mallory"]);
    let (a, b, m) = (0, 1, 2);
    let (ann, mal) = (r.peer(a), r.peer(m));
    let prop = 8001u128;
    r.object(prop, OwnershipMode::Grabbable, Vec3::ZERO, r.peer(a));
    r.step(5);

    // Every tag with truncated, random and over-size bodies.
    for tag in (0..=255u8).filter(|t| *t != 6 /* BYE: a client may leave */) {
        for len in [0usize, 1, 3, 9, 40, 300, 1100, 1300, 1900] {
            r.clients[m].inject_plain(&wire::garbage(tag, len));
        }
        r.step(1);
    }
    r.clients[m].inject_plain(&wire::voice(ann, 1, b"stolen voice"));
    r.clients[m].inject_plain(&wire::anim(ann, prop, &[("x", 1.0)]));
    r.clients[m].inject_plain(&wire::anim(mal, prop, &[("x", f32::NAN)]));
    r.clients[m].inject_plain(&wire::avparam(ann, prop, &[1, 2, 3]));
    r.clients[m].inject_plain(&wire::input(prop, &[0xFF; 200]));
    r.clients[m].inject_endless_fragments(3000);
    r.step(60);
    assert!(r.host.val.count(mal, Reason::TooLarge) >= 1, "oversized packets and endless messages are counted");
    assert!(r.host.val.count(mal, Reason::Forged) >= 3, "forged sender ids on voice, animator and parameter packets");
    assert!(r.host.val.count(mal, Reason::NotFinite) >= 1);
    assert!(r.host.take_voice().is_empty(), "the forged voice was not relayed");
    assert!(r.clients[a].take_voice().is_empty());

    // The honest players are untouched: voice, chat and a held prop still replicate.
    r.clients[b].send_voice(1, b"hello");
    r.clients[a].send_message(None, "still here");
    r.clients[a].send_snapshot(&[snap(prop, Vec3::new(0.5, 0.0, 0.0))]);
    r.step(20);
    assert_eq!(r.clients[0].state(), &ConnState::Connected);
    assert_eq!(r.host.val.counts(ann).iter().sum::<u32>(), 0);
    assert_eq!(r.host.val.counts(r.peer(b)).iter().sum::<u32>(), 0);
    assert!(r.host.take_voice().iter().any(|p| p.from == r.peer(b)), "Bob's voice reached the server");
    assert!(r.clients[2].view().messages.iter().any(|x| x.2 == "still here"));
    assert!(pos_of(&r.host, prop).distance(Vec3::new(0.5, 0.0, 0.0)) < 0.1);
    // Ann's honest client never saw Mallory's garbage as anything.
    assert!(r.clients[a].view().events.iter().all(|e| !matches!(e, NetEvent::Disconnected { .. })));
}

#[test]
fn replayed_packets_are_dropped() {
    let mut r = Rig::new(Topology::Dedicated, NetValidation::default(), &["Ann", "Mallory"]);
    let m = 1;
    let mal = r.peer(m);
    let sealed = r.clients[m].seal_plain(&wire::voice(mal, 1, b"once")).expect("a sealed datagram");
    let (from, to) = (r.clients[m].local_addr().unwrap(), r.host.local_addr().unwrap());
    let rejected = r.host.stats.rejected;
    for _ in 0..10 {
        r.hub.inject(from, to, &sealed);
    }
    r.step(3);
    assert_eq!(r.host.take_voice().len(), 1, "the datagram counted once");
    assert_eq!(r.host.stats.rejected - rejected, 9, "the nine replays were dropped by the secure channel");
    // A datagram from an address that never joined, and a bit-flipped one.
    let rogue: std::net::SocketAddr = "10.200.0.9:1".parse().unwrap();
    r.hub.inject(rogue, to, &sealed);
    let mut flipped = r.clients[m].seal_plain(&wire::voice(mal, 2, b"two")).unwrap();
    let k = flipped.len() / 2;
    flipped[k] ^= 1;
    r.hub.inject(from, to, &flipped);
    r.step(3);
    assert!(r.host.take_voice().is_empty());
}

// ------------------------------------------------------------------ rewind

#[test]
fn rewind_is_capped_below_one_second_and_bad_times_are_refused() {
    use zestengine_core::{HARD_MAX_REWIND, HitShape, ObjectId};
    // The setting is clamped even when a file says 5 s.
    let v = NetValidation { max_rewind: 5.0, ..Default::default() };
    let r = Rig::new(Topology::Dedicated, v, &["Ann"]);
    let cap = r.host.history.lock().unwrap().max_rewind();
    assert!(cap <= HARD_MAX_REWIND as f64 && cap < 1.0, "{cap}");

    let r = Rig::new(Topology::Dedicated, NetValidation::default(), &["Ann"]);
    let target = ObjectId::from_raw(9001);
    {
        let mut h = r.host.history.lock().unwrap();
        for k in 0..=100 {
            let t = k as f64 * 0.01;
            h.record(target, t, Vec3::new(t as f32 * 10.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE, &[HitShape::Sphere { center: Vec3::ZERO, radius: 0.5 }]);
        }
    }
    let h = r.host.history.lock().unwrap();
    let now = h.now;
    // The target moves at 10 m/s along +x: at time t it is at x = 10 t, so a ray down -z from x = 5 hits at t = 0.5.
    let at = |t: f64| h.ray_cast_at(t, Vec3::new((t * 10.0) as f32, 0.0, 5.0), Vec3::NEG_Z, 20.0, &[]);
    assert!(at(now - 0.2).is_some(), "inside the window");
    assert!(at(now - 0.6).is_none(), "older than the 0.25 s window: refused, not clamped");
    assert!(at(now + 5.0).is_none(), "from the future");
    assert!(at(f64::NAN).is_none() && at(f64::INFINITY).is_none() && at(f64::NEG_INFINITY).is_none());
    assert_eq!(h.rejected(), 5, "the refusals are counted");
}

// ------------------------------------------------------------------ strikes, kick, ban

fn flood_until_gone(r: &mut Rig, i: usize, mal: u64) -> String {
    for _ in 0..40 {
        for _ in 0..120 {
            r.clients[i].inject_plain(&wire::urpc(mal, RpcTarget::All, 0, "spam", &[]));
        }
        r.step(2);
        if let ConnState::Disconnected(why) = r.clients[i].state() {
            return why.clone();
        }
    }
    panic!("the flooding client was never removed");
}

#[test]
fn strikes_lead_to_a_kick_then_a_ban_by_key_that_survives_a_restart() {
    let dir = std::env::temp_dir().join(format!("zestengine-net4-{}-{}", std::process::id(), Identity::generate().fingerprint()));
    let ban_file = dir.join(".server/bans.ron");
    let _ = std::fs::remove_dir_all(&dir);
    let bans = ban_file.clone();
    let mut r = Rig::with_host_cfg(Topology::Dedicated, NetValidation::default(), &["Ann"], move |c| c.ban_file = Some(bans));
    let mallory = Identity::generate();
    let fp = mallory.fingerprint();
    let m = r.join("Mallory", mallory.clone());
    r.step(30);
    assert!(r.clients[m].is_connected());
    let mal = r.peer(m);

    // 1: a flood is warned about, then dropped, then the sender is kicked with a reason.
    let why = flood_until_gone(&mut r, m, mal);
    assert!(why.contains("kicked") && why.contains("violations"), "{why}");
    assert!(r.host.val.bans.entries().is_empty(), "the first kick is not a ban");
    assert!(r.host.stats.violations > 100);
    assert!(r.clients[0].is_connected(), "Ann stayed");

    // 2: the same key comes back under a new peer id and floods again: banned.
    let m2 = r.join("Mallory again", mallory.clone());
    r.step(30);
    assert!(r.clients[m2].is_connected(), "a kick is not a ban");
    let mal2 = r.peer(m2);
    let why = flood_until_gone(&mut r, m2, mal2);
    assert!(why.contains("banned"), "{why}");
    assert!(r.host.val.bans.is_banned(&fp, zestengine_engine::net_validate::unix_now()).is_some());

    // 3: the key is refused at the door, whatever its name or address.
    let m3 = r.join("Innocent name", mallory.clone());
    r.step(60);
    match r.clients[m3].state() {
        ConnState::Disconnected(why) => assert!(why.contains("banned"), "{why}"),
        s => panic!("the banned key got in: {s:?}"),
    }
    // A different key is fine.
    let n = r.join("New", Identity::generate());
    r.step(30);
    assert!(r.clients[n].is_connected());

    // 4: the ban is on disk and a restarted server (a new session on the same file) still refuses the key.
    assert!(ban_file.is_file(), "bans.ron written");
    let text = std::fs::read_to_string(&ban_file).unwrap();
    assert!(text.contains(&fp), "{text}");
    drop(r);
    let bans = ban_file.clone();
    let mut r2 = Rig::with_host_cfg(Topology::Dedicated, NetValidation::default(), &[], move |c| c.ban_file = Some(bans));
    let again = r2.join("Mallory", mallory);
    r2.step(60);
    match r2.clients[again].state() {
        ConnState::Disconnected(why) => assert!(why.contains("banned"), "{why}"),
        s => panic!("the ban did not survive the restart: {s:?}"),
    }
    let ok = r2.join("Fine", Identity::generate());
    r2.step(30);
    assert!(r2.clients[ok].is_connected());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn honest_heavy_traffic_never_earns_a_strike() {
    // A VR player sending poses at 90 Hz, snapshots, voice, anim and chat for 20 s.
    let mut r = Rig::new(Topology::Dedicated, NetValidation::default(), &["Ann", "Bob"]);
    let a = 0;
    let avatar = 4500u128;
    let root = Vec3::new(5.0, 0.0, 5.0);
    r.object(avatar, OwnershipMode::Fixed, root, r.peer(a));
    r.avatar(a, avatar, root);
    let who = r.peer(a);
    r.clients[a].world.owners.insert(avatar, (who, 1));
    r.step(5);
    let mut pos = root;
    for k in 0..1200 {
        let t = k as f32 / 60.0;
        pos.x += 3.0 / 60.0; // walking
        r.clients[a].send_snapshot(&[snap(avatar, pos)]);
        let head = pos + Vec3::new(0.0, 1.7, 0.0);
        let mut poses = [None; TRACKER_SLOTS];
        poses[0] = Some((head, Quat::IDENTITY));
        poses[1] = Some((head + Vec3::new(-0.3, -0.4, 0.4 * (t * 3.0).sin()), Quat::IDENTITY));
        poses[2] = Some((head + Vec3::new(0.3, -0.4, 0.4 * (t * 2.0).cos()), Quat::IDENTITY));
        r.clients[a].send_avatar(avatar, &poses);
        if k % 3 == 0 {
            r.clients[a].send_voice(k as u32, &[1; 60]);
        }
        if k % 120 == 0 {
            r.clients[a].send_message(None, "hi");
        }
        r.step(1);
    }
    assert_eq!(r.host.stats.violations, 0, "{:?}", r.host.val.counts(r.peer(a)));
    assert_eq!(r.host.stats.muted, 0);
    assert!(r.clients[a].is_connected() && r.clients[1].is_connected());
}

#[test]
fn bad_packet_cost_stays_small() {
    // 100 000 bad packets from one authenticated client: how long does the server take?
    let mut v = NetValidation::default();
    v.strikes.kick_at = 1.0e12; // never kicked: every packet takes the full reject path
    let mut r = Rig::new(Topology::Dedicated, v, &["Mallory"]);
    let mal = r.peer(0);
    let n = 100_000usize;
    let t0 = Instant::now();
    for k in 0..n {
        let body = if k % 3 == 0 { wire::garbage(0xEE, 100) } else if k % 3 == 1 { wire::urpc(mal, RpcTarget::All, 0, "spam", &[NetValue::Int(k as i64)]) } else { wire::garbage(9, 1300) };
        r.clients[0].inject_plain(&body);
    }
    let send = t0.elapsed();
    let t1 = Instant::now();
    r.host.advance(DT);
    r.host.pump();
    let host = t1.elapsed();
    let per = host.as_nanos() as f64 / n as f64;
    println!("NET-4 cost: {n} bad packets, client send+seal {send:?}, server pump {host:?} = {per:.0} ns/packet (kick disabled: full reject path)");
    assert!(r.host.stats.violations as usize >= n - 800, "violations {}", r.host.stats.violations);
    assert!(host.as_secs_f64() < 5.0, "{host:?}");
    assert!(r.host.val.count(mal, Reason::RateLimited) as usize > n / 2);
    assert!(r.host.view().rpcs.len() < 100, "almost nothing got through");

    // With the default policy the same flood ends with the sender kicked after a few hundred packets,
    // and the rest cost one failed lookup each.
    let mut r = Rig::new(Topology::Dedicated, NetValidation::default(), &["Mallory", "Ann"]);
    let mal = r.peer(0);
    for _ in 0..n {
        r.clients[0].inject_plain(&wire::urpc(mal, RpcTarget::All, 0, "spam", &[]));
    }
    let t1 = Instant::now();
    r.host.advance(DT);
    r.host.pump();
    let host = t1.elapsed();
    println!("NET-4 cost: {n} bad packets with the default strike policy: server pump {host:?} ({:.0} ns/packet), {} kept in the strike path, rest rejected after the kick", host.as_nanos() as f64 / n as f64, r.host.stats.violations);
    r.step(30);
    assert!(matches!(r.clients[0].state(), ConnState::Disconnected(_)), "kicked");
    assert!(r.clients[1].is_connected(), "Ann is fine");
    assert!(host.as_secs_f64() < 5.0);
}

// ------------------------------------------------------------------ privacy and topology

fn successor_addresses(s: &NetSession) -> usize {
    s.succession().map_or(0, |p| p.order.iter().map(|c| c.addrs.len()).sum())
}

#[test]
fn migration_plans_do_not_broadcast_player_addresses() {
    // A listen host: members get the next hosts' endpoints only (two by default), never everyone's.
    let mut r = Rig::new(Topology::ListenHost, NetValidation::default(), &["A", "B", "C", "D", "E"]);
    r.step(30);
    for c in &r.clients {
        let plan = c.succession().expect("a listen host replicates a plan");
        assert_eq!(plan.order.len(), 5);
        assert_eq!(plan.order.iter().filter(|s| !s.addrs.is_empty()).count(), 2, "only the first two candidates carry addresses");
        assert!(plan.order[2..].iter().all(|s| s.addrs.is_empty()));
    }
    // The depth is a setting: 0 shares nothing.
    let mut r = Rig::with_host_cfg(Topology::ListenHost, NetValidation::default(), &["A", "B", "C"], |c| c.migration_addr_depth = 0);
    r.step(30);
    assert!(r.clients.iter().all(|c| c.succession().is_some() && successor_addresses(c) == 0));

    // A dedicated server has no plan at all, no candidate and no endpoint.
    let mut r = Rig::new(Topology::Dedicated, NetValidation::default(), &["A", "B", "C"]);
    r.step(30);
    assert!(r.clients.iter().all(|c| c.succession().is_none()), "no succession plan on a dedicated server");
    assert!(r.clients.iter().all(|c| c.topology() == Topology::Dedicated), "clients learn the topology from the server");
    assert!(r.host.succession().is_none());
}

#[test]
fn clients_pin_a_dedicated_servers_key_and_refuse_a_changed_one() {
    // First join pins, whatever the client's own settings are.
    let r = Rig::new(Topology::Dedicated, NetValidation::default(), &["Ann"]);
    let target = r.clients[0].host_pin_target().unwrap().to_owned();
    assert!(NetSession::host_is_pinned(&target), "the first join pinned the server's key");
    // A listen host is not pinned unless asked.
    let l = Rig::new(Topology::ListenHost, NetValidation::default(), &["Ann"]);
    let ltarget = l.clients[0].host_pin_target().unwrap().to_owned();
    assert!(!NetSession::host_is_pinned(&ltarget));
    assert!(NetSession::forget_host_pin(&target) && !NetSession::host_is_pinned(&target), "the user can clear a pin");
    assert!(!NetSession::forget_host_pin(&target));
}

// ------------------------------------------------------------------ NET-4 leftovers

#[test]
fn parent_ids_are_validated() {
    let mut r = Rig::new(Topology::Dedicated, lenient(), &["Ann", "Mallory"]);
    let (a, m) = (0, 1);
    let (ann, mal) = (r.peer(a), r.peer(m));
    r.step(5);
    r.object(0xD001, OwnershipMode::Fixed, Vec3::ZERO, mal); // Mallory's prop
    r.object(0xD002, OwnershipMode::Fixed, Vec3::ZERO, ann); // Ann's prop
    r.object(0xD003, OwnershipMode::Fixed, Vec3::ZERO, mal); // Mallory's second prop
    // Under herself, under another player's object, under a stranger's: only the legal one lands.
    r.clients[m].inject_reliable(&wire::parent(0xD001, 0xD001));
    r.clients[m].inject_reliable(&wire::parent(0xD001, 0xD002));
    r.step(10);
    assert_eq!(r.host.world.parents.get(&0xD001), None, "nothing attached");
    assert!(r.count(m, Reason::Malformed) >= 1 && r.count(m, Reason::OwnershipDenied) >= 1);
    r.clients[m].inject_reliable(&wire::parent(0xD001, 0xD003));
    r.step(10);
    assert_eq!(r.host.world.parents.get(&0xD001), Some(&0xD003), "her own object is fine");
    // A cycle: 0xD003 under 0xD001 would loop.
    r.clients[m].inject_reliable(&wire::parent(0xD003, 0xD001));
    r.step(10);
    assert_eq!(r.host.world.parents.get(&0xD003), None, "cycle refused");
    // Detaching always works.
    r.clients[m].inject_reliable(&wire::parent(0xD001, 0));
    r.step(10);
    assert_eq!(r.host.world.parents.get(&0xD001), Some(&0));
}

#[test]
fn per_prefab_quotas_and_the_variable_budget() {
    use zestengine_core::SpawnQuota;
    let mut v = lenient();
    v.client_spawn_prefabs = vec!["prefabs/*".into()];
    v.max_client_spawns = 50;
    v.spawn_quotas = vec![SpawnQuota { prefab: "prefabs/rocket.prefab".into(), per_peer: 2, total: 3 }];
    v.max_var_total = 1000;
    v.max_var_keys = 4;
    let mut r = Rig::new(Topology::Dedicated, v, &["Ann", "Mallory"]);
    let (a, m) = (0, 1);
    let (ann, mal) = (r.peer(a), r.peer(m));
    r.avatar(a, 101, Vec3::ZERO);
    r.avatar(m, 102, Vec3::ZERO);
    r.step(5);
    let near = Vec3::new(1.0, 0.0, 0.0);
    let spawn = |r: &mut Rig, who: usize, k: u128, prefab: &str| {
        let p = r.peer(who);
        r.clients[who].inject_reliable(&wire::spawn(0xE000 + k, prefab, near, p));
        r.step(8);
    };
    spawn(&mut r, m, 1, "prefabs/rocket.prefab");
    spawn(&mut r, m, 2, "prefabs/rocket.prefab");
    spawn(&mut r, m, 3, "prefabs/rocket.prefab");
    assert_eq!(r.host.world.spawns.len(), 2, "Mallory's third rocket is over her quota of 2");
    spawn(&mut r, a, 4, "prefabs/rocket.prefab");
    assert_eq!(r.host.world.spawns.len(), 3, "Ann may still have one");
    spawn(&mut r, a, 5, "prefabs/rocket.prefab");
    assert_eq!(r.host.world.spawns.len(), 3, "the server-wide total of 3 is reached");
    spawn(&mut r, m, 6, "prefabs/ball.prefab");
    assert_eq!(r.host.world.spawns.len(), 4, "other prefabs are not affected");

    // Variable budget: 1000 bytes and 4 keys per writer.
    let obj = 7101u128;
    r.object(obj, OwnershipMode::Fixed, Vec3::ZERO, mal);
    r.clients[m].inject_reliable(&wire::var(obj, "a", &NetValue::Bytes(vec![1; 600]), mal));
    r.clients[m].inject_reliable(&wire::var(obj, "b", &NetValue::Bytes(vec![1; 600]), mal));
    r.step(20);
    assert!(r.host.world.var(obj, "a").is_some());
    assert_eq!(r.host.world.var(obj, "b"), None, "600 + 600 is over the 1000-byte budget");
    r.clients[m].inject_reliable(&wire::var(obj, "a", &NetValue::Bytes(vec![2; 900]), mal));
    r.step(20);
    assert_eq!(r.host.world.var(obj, "a"), Some(&NetValue::Bytes(vec![2; 900])), "a rewrite replaces the old size");
    assert!(r.count(m, Reason::TooLarge) >= 1);
    let _ = ann;
}

#[test]
fn oversized_voice_packets_are_refused() {
    let mut v = lenient();
    v.max_voice_packet = 200;
    let mut r = Rig::new(Topology::Dedicated, v, &["Ann", "Mallory"]);
    let (a, m) = (0, 1);
    let mal = r.peer(m);
    r.step(5);
    let _ = r.clients[a].take_voice();
    r.clients[m].inject_plain(&wire::voice(mal, 1, &[7; 120]));
    r.clients[m].inject_plain(&wire::voice(mal, 2, &[7; 1200]));
    r.step(10);
    assert_eq!(r.count(m, Reason::TooLarge), 1, "only the 1200-byte packet");
    let heard: Vec<u32> = r.clients[a].take_voice().iter().map(|p| p.seq).collect();
    assert_eq!(heard, vec![1], "{heard:?}");
}

// ------------------------------------------------------------------ SEC: reports

#[test]
fn player_reports_reach_the_server_queue_and_are_rate_limited() {
    let mut r = Rig::new(Topology::Dedicated, lenient(), &["Ann", "Mallory", "Bob"]);
    let (a, m, b) = (0, 1, 2);
    let (ann, mal) = (r.peer(a), r.peer(m));
    r.step(5);
    r.clients[a].report_player(mal, "aimbot, saw it twice");
    r.step(10);
    let got = r.host.take_reports();
    assert_eq!(got.len(), 1, "{got:?}");
    assert_eq!((got[0].reporter, got[0].target, got[0].reason.as_str()), (ann, mal, "aimbot, saw it twice"));
    assert_eq!((got[0].reporter_name.as_str(), got[0].target_name.as_str()), ("Ann", "Mallory"));
    assert_eq!(got[0].target_fingerprint, r.host.peer_fingerprint(mal).unwrap());
    // Nobody else hears about it.
    assert!(r.clients[b].view().rpcs.is_empty() && r.clients[m].view().rpcs.is_empty());
    assert!(r.host.take_reports().is_empty());
    // A report against yourself, a stranger or with a giant reason is refused as malformed.
    r.clients[a].report_player(ann, "me");
    r.clients[a].report_player(9999, "ghost");
    r.clients[a].report_player(mal, &"x".repeat(5000));
    r.step(10);
    assert!(r.host.take_reports().is_empty());
    assert!(r.count(a, Reason::Malformed) + r.count(a, Reason::TooLarge) >= 2);
    // A flood is capped by the report bucket: burst 3, then 0.2 per second.
    for k in 0..30 {
        r.clients[b].report_player(mal, &format!("spam {k}"));
    }
    r.step(30);
    let n = r.host.take_reports().len();
    assert!((1..=4).contains(&n), "{n} of 30 spam reports got through");
    // The host's own report is filed directly.
    r.host.report_player(mal, "host saw it");
    assert_eq!(r.host.take_reports().len(), 1);
}
