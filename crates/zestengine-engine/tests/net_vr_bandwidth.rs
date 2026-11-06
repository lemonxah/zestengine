//! NET-5: the 50-VR-player bandwidth measurement with a realistic avatar traffic model.
//!
//! Every simulated player is a headset user and sends what the engine's own avatar stack sends:
//! avatar pose packets (`send_avatar`: head, both hands and a body tracker) at 30 to 90 Hz,
//! the root transform as a snapshot at 15 Hz, avatar parameter changes (`send_avparam`, an
//! OSC-like burst of a few per second), voice frames (`send_voice`, Opus-sized 20 ms frames at
//! 24 to 32 kbps while speaking, about a quarter of the players at any time) and face tracking
//! (`send_face`, about 60 live blendshapes at 30 Hz) for a subset of the players.
//!
//! The server is a Dedicated session with the default validation on. Two transports: the
//! in-memory hub on a manual clock (deterministic, fast) and real UDP on loopback (a real clock,
//! real sockets, real encryption); the same model drives both.
//!
//! The fast checks run in CI. The full tables are `#[ignore]`d:
//!
//! ```text
//! cargo test -p zestengine-engine --release --test net_vr_bandwidth -- --ignored --nocapture --test-threads=1
//! ```

use std::time::{Duration, Instant};

use glam::{Quat, Vec3};
use zestengine_core::{FaceFrame, InterestConfig, NetSpawnSource, NetValidation, TRACKER_SLOTS, Topology};
use zestengine_engine::net::{Identity, MemHub, NetConfig, NetSession, SnapEntry};

/// Simulation step: the fastest pose rate.
const DT: f64 = 1.0 / 90.0;
/// The server's tick: one `advance` + `pump` every third step.
const SERVER_EVERY: u64 = 3;

#[derive(Clone, Copy, Debug)]
enum Layout {
    /// Everyone in one plaza of this radius (a social-VR hub instance).
    Plaza(f32),
    /// A square world with players in groups of eight (events spread over a venue).
    Spread { side: f32 },
}

#[derive(Clone, Debug)]
struct Model {
    players: usize,
    layout: Layout,
    /// Pose rates handed out round-robin (Hz): flat-screen-like 30, Quest 72, PCVR 90.
    pose_hz: Vec<f32>,
    /// Avatar parameter packets per second per player (bursty).
    param_per_s: f32,
    /// Fraction of players speaking at any instant (a two-state chain, 4 s talking, 12 s quiet).
    speaking: f32,
    /// Opus payload of a 20 ms frame, bytes (24 kbps = 60, 32 kbps = 80).
    voice_bytes: (usize, usize),
    /// Every k-th player sends face tracking (0 = nobody).
    face_every: usize,
    face_hz: f32,
    face_live: usize,
}

impl Model {
    fn vr(players: usize, layout: Layout) -> Self {
        Self { players, layout, pose_hz: vec![30.0, 72.0, 72.0, 90.0], param_per_s: 3.0, speaking: 0.25, voice_bytes: (60, 80), face_every: 4, face_hz: 30.0, face_live: 60 }
    }
    fn poses_only(mut self) -> Self {
        self.param_per_s = 0.0;
        self.speaking = 0.0;
        self.face_every = 0;
        self
    }
    fn with_params(mut self) -> Self {
        self.param_per_s = 3.0;
        self
    }
    fn with_face(mut self) -> Self {
        self.face_every = 4;
        self
    }
}

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

struct Player {
    id: u128,
    pos: Vec3,
    yaw: f32,
    dir: f32,
    phase: f32,
    pose_every: f64,
    next_pose: f64,
    next_root: f64,
    next_face: f64,
    next_voice: f64,
    seq: u32,
    speaking: bool,
    face: bool,
    voice_len: usize,
}

#[derive(Default, Clone, Debug)]
struct Report {
    label: String,
    #[allow(dead_code)] // wall-clock columns kept for manual table reading
    secs: f64,
    /// Server output, bytes per second (all clients) and packets per second.
    out_bps: f64,
    out_pps: f64,
    /// Server input, bytes per second.
    #[allow(dead_code)] // manual table reading
    in_bps: f64,
    /// Mean downstream bytes per second per client, and the busiest client's.
    down_mean: f64,
    down_max: f64,
    /// Mean upstream bytes per second per client.
    up_mean: f64,
    cpu_mean_ms: f64,
    cpu_p99_ms: f64,
    violations: u64,
    relevant_avg: f32,
}

impl Report {
    fn row(&self) -> String {
        format!(
            "| {} | {:.1} kB/s | {:.1} kB/s | {:.2} MB/s | {:.0} | {:.2} / {:.2} ms | {:.1} kB/s | {} |",
            self.label,
            self.down_mean / 1024.0,
            self.down_max / 1024.0,
            self.out_bps / 1e6,
            self.out_pps,
            self.cpu_mean_ms,
            self.cpu_p99_ms,
            self.up_mean / 1024.0,
            if self.relevant_avg > 0.0 { format!("{:.0}", self.relevant_avg) } else { "all".into() },
        )
    }
}

fn net_cfg(name: &str, interest: InterestConfig, topology: Topology) -> NetConfig {
    NetConfig {
        player_name: name.to_string(),
        discovery: false,
        secret: String::new(),
        relay: String::new(),
        account_name: String::new(),
        account_password: String::new(),
        accounts: Vec::new(),
        identity: Identity::generate(),
        validation: NetValidation::default(),
        interest,
        topology,
        max_players: 128,
        timeout: 30.0,
        ..Default::default()
    }
}

/// Where player `i` starts.
fn start_pos(m: &Model, i: usize, rng: &mut Rng) -> Vec3 {
    match m.layout {
        Layout::Plaza(r) => {
            let a = rng.f() * std::f32::consts::TAU;
            let d = r * rng.f().sqrt();
            Vec3::new(a.cos() * d, 0.0, a.sin() * d)
        }
        Layout::Spread { side } => {
            // Groups of eight around one of a few random centres.
            let groups = m.players.div_ceil(8);
            let mut g = Rng(0x9E37_79B9_7F4A_7C15 ^ ((i / 8) as u64 + 1).wrapping_mul(0x2545_F491_4F6C_DD1D));
            let _ = groups;
            let c = Vec3::new(g.f() * side, 0.0, g.f() * side);
            c + Vec3::new((rng.f() - 0.5) * 12.0, 0.0, (rng.f() - 0.5) * 12.0)
        }
    }
}

fn pose(p: &Player, t: f64) -> [Option<(Vec3, Quat)>; TRACKER_SLOTS] {
    let rot = Quat::from_rotation_y(p.yaw);
    let t = t as f32 + p.phase;
    let head = p.pos + Vec3::new(0.0, 1.65 + 0.02 * (t * 1.3).sin(), 0.0);
    let hand = |side: f32, w: f32| head + rot * Vec3::new(0.28 * side, -0.45 + 0.12 * (t * w).sin(), 0.30 + 0.15 * (t * w * 0.7).cos());
    let mut s = [None; TRACKER_SLOTS];
    s[0] = Some((head, rot));
    s[1] = Some((hand(-1.0, 2.1), rot));
    s[2] = Some((hand(1.0, 1.7), rot));
    // A hip tracker, the "body root" of a full-body user.
    s[3] = Some((p.pos + Vec3::new(0.0, 0.95, 0.0), rot));
    s
}

fn face_frame(live: usize, t: f32) -> FaceFrame {
    let mut f = FaceFrame::default();
    for (i, w) in f.weights.iter_mut().enumerate().take(live) {
        *w = 0.05 + 0.9 * (0.5 + 0.5 * (t * (1.0 + i as f32 * 0.13) + i as f32).sin());
    }
    f.has_gaze = true;
    f.gaze_left = glam::Vec2::new((t * 0.7).sin() * 0.3, 0.1);
    f.gaze_right = f.gaze_left;
    f
}

/// Run the model against already connected sessions. `step_clock` advances a manual clock (None
/// on real sockets, where the loop paces itself on the wall clock).
fn drive(label: &str, m: &Model, host: &mut NetSession, clients: &mut [NetSession], hub: Option<&MemHub>, secs: f64, warm: f64) -> Report {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let n = clients.len();
    let mut players: Vec<Player> = Vec::with_capacity(n);
    for (i, c) in clients.iter().enumerate() {
        let pos = start_pos(m, i, &mut rng);
        let id = host.spawn_everywhere(NetSpawnSource::Prefab("prefabs/player.prefab".into()), pos, Quat::IDENTITY, c.local_peer());
        host.world.player_obj.insert(c.local_peer(), id.raw());
        let hz = m.pose_hz[i % m.pose_hz.len()] as f64;
        let (lo, hi) = m.voice_bytes;
        players.push(Player {
            id: id.raw(),
            pos,
            yaw: rng.f() * std::f32::consts::TAU,
            dir: rng.f() * std::f32::consts::TAU,
            phase: rng.f() * 10.0,
            pose_every: 1.0 / hz,
            next_pose: rng.f() as f64 / hz,
            next_root: rng.f() as f64 / 15.0,
            next_face: rng.f() as f64 / m.face_hz as f64,
            next_voice: rng.f() as f64 * 0.02,
            seq: 0,
            speaking: rng.f() < m.speaking,
            face: m.face_every > 0 && i % m.face_every == 0,
            voice_len: lo + ((rng.f() * (hi - lo + 1) as f32) as usize).min(hi - lo),
        });
    }
    let real = hub.is_none();
    let total_steps = ((secs + warm) / DT) as u64;
    let warm_steps = (warm / DT) as u64;
    let mut cpu: Vec<f64> = Vec::new();
    let mut mark = None;
    let wall = Instant::now();
    let mut t = 0.0f64;
    for k in 0..total_steps {
        if k == warm_steps {
            mark = Some((host.stats.bytes_out, host.stats.packets_out, host.stats.bytes_in, clients.iter().map(|c| (c.stats.bytes_in, c.stats.bytes_out)).collect::<Vec<_>>(), host.stats.violations));
        }
        t += DT;
        if let Some(h) = hub {
            h.advance(DT);
        } else {
            // Real sockets: keep to the wall clock.
            let due = Duration::from_secs_f64(t);
            if let Some(wait) = due.checked_sub(wall.elapsed()) {
                std::thread::sleep(wait);
            }
        }
        if k % SERVER_EVERY == 0 {
            let t0 = Instant::now();
            host.advance(DT * SERVER_EVERY as f64);
            host.pump();
            if k >= warm_steps {
                cpu.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
        }
        for (i, c) in clients.iter_mut().enumerate() {
            c.advance(DT);
            c.pump();
            let p = &mut players[i];
            // Walk about at 1.4 m/s, turning now and then.
            if rng.f() < 0.02 {
                p.dir += (rng.f() - 0.5) * 2.0;
            }
            // Heads turn at a human rate: at most 2 rad/s, so the hands (0.3 m off) stay within the speed budget.
            p.yaw += (p.dir - p.yaw).clamp(-2.0 * DT as f32, 2.0 * DT as f32);
            p.pos += Vec3::new(p.yaw.cos(), 0.0, p.yaw.sin()) * (1.4 * DT as f32);
            let (lo, hi) = match m.layout {
                Layout::Plaza(r) => (-r, r),
                Layout::Spread { side } => (0.0, side),
            };
            p.pos.x = p.pos.x.clamp(lo, hi);
            p.pos.z = p.pos.z.clamp(lo, hi);
            if t >= p.next_root {
                p.next_root += 1.0 / 15.0;
                c.send_snapshot(&[SnapEntry { id: p.id, pos: Some(p.pos), rot: Some(Quat::from_rotation_y(p.yaw)), quantized: true, ..Default::default() }]);
            }
            if t >= p.next_pose {
                p.next_pose += p.pose_every;
                c.send_avatar(p.id, &pose(p, t));
            }
            if m.param_per_s > 0.0 && rng.f() < m.param_per_s * DT as f32 {
                // Schema hash (4 bytes) + a few synced parameter bits, as `AvParamSchema::encode` writes.
                let mut b = vec![0xA5, 0x5A, 0x11, 0x22];
                b.extend((0..24).map(|_| (rng.f() * 255.0) as u8));
                c.send_avparam(p.id, &b);
            }
            if m.speaking > 0.0 {
                while t >= p.next_voice {
                    p.next_voice += 0.02;
                    // Two-state chain per 20 ms frame: talk bursts of ~4 s, silences of ~12 s.
                    let stop = 1.0 / (4.0 * 50.0);
                    let start = stop * m.speaking / (1.0 - m.speaking);
                    p.speaking = if p.speaking { rng.f() >= stop } else { rng.f() < start };
                    if p.speaking {
                        p.seq += 1;
                        let mut payload = vec![1u8]; // codec byte: Opus
                        payload.extend((0..p.voice_len).map(|j| (j as u32).wrapping_mul(p.seq) as u8));
                        c.send_voice(p.seq, &payload);
                    }
                }
            }
            if p.face && t >= p.next_face {
                p.next_face += 1.0 / m.face_hz as f64;
                c.send_face(p.id, &face_frame(m.face_live, t as f32));
            }
        }
        let _ = real;
    }
    if host.stats.violations > 0 {
        for c in clients.iter() {
            let counts = host.val.counts(c.local_peer());
            if !format!("{counts:?}").contains("[]") {
                println!("violations of peer {}: {counts:?}", c.local_peer());
            }
        }
    }
    let (b0, p0, i0, c0, v0) = mark.expect("warm-up shorter than the run");
    let mut down: Vec<f64> = clients.iter().zip(&c0).map(|(c, b)| (c.stats.bytes_in - b.0) as f64 / secs).collect();
    let up: f64 = clients.iter().zip(&c0).map(|(c, b)| (c.stats.bytes_out - b.1) as f64 / secs).sum::<f64>() / n as f64;
    down.sort_by(|a, b| a.total_cmp(b));
    cpu.sort_by(|a, b| a.total_cmp(b));
    Report {
        label: label.to_string(),
        secs,
        out_bps: (host.stats.bytes_out - b0) as f64 / secs,
        out_pps: (host.stats.packets_out - p0) as f64 / secs,
        in_bps: (host.stats.bytes_in - i0) as f64 / secs,
        down_mean: down.iter().sum::<f64>() / n as f64,
        down_max: *down.last().unwrap_or(&0.0),
        up_mean: up,
        cpu_mean_ms: cpu.iter().sum::<f64>() / cpu.len().max(1) as f64,
        cpu_p99_ms: cpu.get((cpu.len() as f64 * 0.99) as usize).copied().unwrap_or(0.0),
        violations: host.stats.violations - v0,
        relevant_avg: host.stats.interest_relevant_avg,
    }
}

/// One run over the in-memory hub.
fn run_mem(label: &str, m: &Model, interest: InterestConfig, secs: f64) -> Report {
    let hub = MemHub::new();
    let mut host = NetSession::host_on(Box::new(hub.endpoint()), net_cfg("Server", interest, Topology::Dedicated));
    host.use_manual_clock();
    let addr = host.local_addr().unwrap();
    let mut clients: Vec<NetSession> = (0..m.players)
        .map(|i| {
            let mut c = NetSession::join_on(Box::new(hub.endpoint()), addr, net_cfg(&format!("P{i}"), InterestConfig::default(), Topology::ListenHost));
            c.use_manual_clock();
            c
        })
        .collect();
    for _ in 0..(90 + m.players * 3) {
        hub.advance(DT);
        host.advance(DT);
        host.pump();
        for c in clients.iter_mut() {
            c.advance(DT);
            c.pump();
        }
    }
    assert!(clients.iter().all(|c| c.is_connected()), "all {} clients joined", m.players);
    drive(label, m, &mut host, &mut clients, Some(&hub), secs, 4.0)
}

/// One run over real UDP sockets on loopback.
fn run_udp(label: &str, m: &Model, interest: InterestConfig, secs: f64) -> Report {
    let mut host = NetSession::host_with(0, net_cfg("Server", interest, Topology::Dedicated)).expect("bind");
    let port = host.local_addr().unwrap().port();
    let addr = format!("127.0.0.1:{port}");
    let mut clients: Vec<NetSession> = (0..m.players).map(|i| NetSession::join_with(&addr, net_cfg(&format!("P{i}"), InterestConfig::default(), Topology::ListenHost)).expect("join")).collect();
    let start = Instant::now();
    while !clients.iter().all(|c| c.is_connected()) && start.elapsed() < Duration::from_secs(30) {
        host.pump();
        for c in clients.iter_mut() {
            c.pump();
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(clients.iter().all(|c| c.is_connected()), "all {} clients joined over UDP", m.players);
    drive(label, m, &mut host, &mut clients, None, secs, 4.0)
}

fn on() -> InterestConfig {
    InterestConfig { enabled: true, ..Default::default() }
}

const HEADER: &str = "| case | down / client (mean) | down / client (busiest) | server out total | server packets/s | server CPU per tick mean / p99 | up / client | relevant / client |\n|---|---|---|---|---|---|---|---|";

// ------------------------------------------------------------------ CI-sized checks

#[test]
fn the_traffic_model_reaches_everyone_and_validates() {
    let m = Model::vr(8, Layout::Plaza(20.0));
    let r = run_mem("8 players", &m, on(), 3.0);
    println!("{HEADER}\n{}", r.row());
    assert_eq!(r.violations, 0, "honest VR traffic (poses, params, voice, face) passes the strict validation");
    assert!(r.down_mean > 1000.0, "players receive each other's avatars: {:.0} B/s", r.down_mean);
    assert!(r.out_bps > 0.0 && r.up_mean > 1000.0);
}

/// A 90 Hz headset is relayed at the near rate (30 Hz) to a neighbour with interest on, and at
/// the full 90 Hz with it off.
#[test]
fn a_90_hz_pose_stream_is_relayed_at_the_near_rate() {
    let rate = |interest: InterestConfig| -> f64 {
        let mut m = Model::vr(2, Layout::Plaza(3.0)).poses_only();
        m.pose_hz = vec![90.0];
        let hub = MemHub::new();
        let mut host = NetSession::host_on(Box::new(hub.endpoint()), net_cfg("Server", interest, Topology::Dedicated));
        host.use_manual_clock();
        let addr = host.local_addr().unwrap();
        let mut clients: Vec<NetSession> = (0..2).map(|i| {
            let mut c = NetSession::join_on(Box::new(hub.endpoint()), addr, net_cfg(&format!("P{i}"), InterestConfig::default(), Topology::ListenHost));
            c.use_manual_clock();
            c
        }).collect();
        for _ in 0..200 {
            hub.advance(DT);
            host.advance(DT);
            host.pump();
            for c in clients.iter_mut() {
                c.advance(DT);
                c.pump();
            }
        }
        let r = drive("x", &m, &mut host, &mut clients, Some(&hub), 4.0, 2.0);
        r.down_mean
    };
    let off = rate(InterestConfig::default());
    let near = rate(on());
    println!("one neighbour at 90 Hz: off {off:.0} B/s, on {near:.0} B/s");
    assert!(near < off * 0.6, "the near rate caps the relay: {near:.0} vs {off:.0}");
}

#[test]
fn interest_cuts_the_traffic_of_a_spread_out_crowd() {
    let m = Model::vr(24, Layout::Spread { side: 900.0 });
    let off = run_mem("off", &m, InterestConfig::default(), 3.0);
    let on_ = run_mem("on", &m, on(), 3.0);
    println!("{HEADER}\n{}\n{}", off.row(), on_.row());
    assert!(on_.out_bps < off.out_bps * 0.5, "interest at least halves a spread-out crowd: {:.0} vs {:.0} B/s", on_.out_bps, off.out_bps);
}

// ------------------------------------------------------------------ the full tables

/// 50 players on the in-memory hub: layouts x interest, and what each kind of traffic costs.
#[test]
#[ignore = "measurement (minutes); run with --release --ignored --nocapture"]
fn vr_50_players_memhub_table() {
    let plaza = Model::vr(50, Layout::Plaza(30.0));
    let spread = Model::vr(50, Layout::Spread { side: 400.0 });
    let rows = vec![
        run_mem("plaza 30 m, interest off", &plaza, InterestConfig::default(), 10.0),
        run_mem("plaza 30 m, interest on", &plaza, on(), 10.0),
        run_mem("spread 400 m, interest off", &spread, InterestConfig::default(), 10.0),
        run_mem("spread 400 m, interest on", &spread, on(), 10.0),
    ];
    println!("{HEADER}");
    for r in &rows {
        println!("{}", r.row());
    }
    println!("\nplaza 30 m, interest on, by traffic kind (cumulative):\n{HEADER}");
    let p = plaza.clone().poses_only();
    for (l, m) in [("poses + root", p.clone()), ("+ avatar params", p.clone().with_params()), ("+ face (every 4th)", p.clone().with_params().with_face()), ("+ voice (25 % speaking)", plaza.clone())] {
        println!("{}", run_mem(l, &m, on(), 10.0).row());
    }
}

/// 50 players over real UDP on loopback.
#[test]
#[ignore = "measurement (minutes, real time); run with --release --ignored --nocapture"]
fn vr_50_players_udp_table() {
    let plaza = Model::vr(50, Layout::Plaza(30.0));
    let spread = Model::vr(50, Layout::Spread { side: 400.0 });
    println!("{HEADER}");
    for (l, m, i) in [
        ("UDP plaza 30 m, interest off", &plaza, InterestConfig::default()),
        ("UDP plaza 30 m, interest on", &plaza, on()),
        ("UDP spread 400 m, interest off", &spread, InterestConfig::default()),
        ("UDP spread 400 m, interest on", &spread, on()),
    ] {
        println!("{}", run_udp(l, m, i, 10.0).row());
    }
}
