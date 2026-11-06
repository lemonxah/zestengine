//! Direct SlimeVR support: ZestEngine connects to the SlimeVR server's driver
//! bridge the way its SteamVR driver does, so SlimeVR full-body tracking works
//! under any OpenXR runtime (WiVRn, Monado, SteamVR) without SteamVR.
//!
//! The bridge is a local socket (`SlimeVRDriver` in `$SLIMEVR_SOCKET_DIR`, else
//! `$XDG_DATA_HOME` or `~/.local/share` + `/dev.slimevr.SlimeVR` on Linux; the
//! `\\.\pipe\SlimeVRDriver` named pipe on Windows) carrying length-prefixed
//! protobuf messages (`messages.ProtobufMessage`, protocol 2). We announce the
//! headset and controllers, stream their poses (the server's head and hand
//! reference), and read back the trackers it solves, by role, in the same
//! tracking space.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use glam::{Quat, Vec3};

const PROTOCOL_VERSION: u64 = 2;
/// Server tracker roles (`dev.slimevr.tracking.trackers.TrackerRole`).
const ROLE_WAIST: i64 = 1;
const ROLE_LEFT_FOOT: i64 = 2;
const ROLE_RIGHT_FOOT: i64 = 3;
const ROLE_CHEST: i64 = 4;
const ROLE_LEFT_KNEE: i64 = 5;
const ROLE_RIGHT_KNEE: i64 = 6;
const ROLE_LEFT_ELBOW: i64 = 7;
const ROLE_RIGHT_ELBOW: i64 = 8;
const ROLE_LEFT_CONTROLLER: i64 = 13;
const ROLE_RIGHT_CONTROLLER: i64 = 14;
const ROLE_HMD: i64 = 19;
/// Our devices' tracker ids on the bridge.
const ID_HMD: i64 = 0;
const ID_LEFT: i64 = 1;
const ID_RIGHT: i64 = 2;
/// Tracker data older than this is dropped.
const STALE: Duration = Duration::from_millis(500);

/// A tracker the server shares with us.
#[derive(Clone, Debug)]
struct Remote {
    role: i64,
    serial: String,
    name: String,
    pose: Option<(Vec3, Quat)>,
    at: Instant,
}

#[derive(Default)]
struct Shared {
    trackers: HashMap<i64, Remote>,
    connected: bool,
    writer: Option<Box<dyn Write + Send>>,
    announced: bool,
}

/// Client of the SlimeVR server's driver bridge. A background thread keeps
/// (re)connecting and reading; poses are sent from the frame loop.
pub struct SlimeVr {
    shared: Arc<Mutex<Shared>>,
    stop: Arc<std::sync::atomic::AtomicBool>,
}

impl SlimeVr {
    pub fn start() -> Self {
        let shared = Arc::new(Mutex::new(Shared::default()));
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (s, flag) = (shared.clone(), stop.clone());
        let _ = std::thread::Builder::new().name("slimevr".into()).spawn(move || run(s, flag));
        tracing::info!("SlimeVR: looking for the server at {}", socket_path().display());
        Self { shared, stop }
    }

        pub fn connected(&self) -> bool {
        self.shared.lock().is_ok_and(|s| s.connected)
    }

    /// Send our headset and controller poses (tracking space) to the server.
    pub fn send_devices(&self, head: Option<(Vec3, Quat)>, left: Option<(Vec3, Quat)>, right: Option<(Vec3, Quat)>) {
        let Ok(mut s) = self.shared.lock() else { return };
        if !s.connected {
            return;
        }
        let mut out = Vec::new();
        if !s.announced {
            out.extend(frame(&wrap(6, &field_varint(1, PROTOCOL_VERSION))));
            for (id, role, name) in [(ID_HMD, ROLE_HMD, "HMD"), (ID_LEFT, ROLE_LEFT_CONTROLLER, "Left Controller"), (ID_RIGHT, ROLE_RIGHT_CONTROLLER, "Right Controller")] {
                out.extend(frame(&wrap(3, &tracker_added(id, role, name))));
            }
            s.announced = true;
        }
        for (id, pose) in [(ID_HMD, head), (ID_LEFT, left), (ID_RIGHT, right)] {
            if let Some((p, r)) = pose {
                out.extend(frame(&wrap(1, &position(id, p, r))));
            }
        }
        if let Some(w) = s.writer.as_mut()
            && w.write_all(&out).and_then(|_| w.flush()).is_err()
        {
            s.writer = None;
            s.connected = false;
        }
    }

    /// The body trackers the server solved (tracking space), each with the
    /// role the server gave it.
    pub fn devices(&self) -> Vec<crate::body::TrackerDevice> {
        let Ok(s) = self.shared.lock() else { return Vec::new() };
        let mut out: Vec<(i64, crate::body::TrackerDevice)> = s
            .trackers
            .iter()
            .filter(|(_, r)| r.at.elapsed() < STALE)
            .filter_map(|(id, r)| {
                let slot = role_slot(r.role)?;
                let key = if r.serial.is_empty() { id.to_string() } else { r.serial.clone() };
                let label = if r.name.is_empty() { format!("SlimeVR {}", zestengine_core::role_name(slot)) } else { format!("SlimeVR {}", r.name) };
                Some((*id, crate::body::TrackerDevice { id: format!("slimevr:{key}"), label, native: Some(slot), pose: r.pose }))
            })
            .collect();
        out.sort_by_key(|(id, _)| *id);
        out.into_iter().map(|(_, d)| d).collect()
    }

    /// The trackers by their server role.
    #[cfg(test)]
    fn targets(&self) -> zestengine_core::TrackerTargets {
        let mut slots = [None; zestengine_core::TRACKER_SLOTS];
        for d in self.devices() {
            if let Some(slot) = d.native {
                slots[slot] = d.pose;
            }
        }
        zestengine_core::TrackerTargets::from_slots(slots)
    }
}

/// Tracker slot of a server role.
fn role_slot(role: i64) -> Option<usize> {
    Some(match role {
        ROLE_WAIST => 3,
        ROLE_LEFT_FOOT => 4,
        ROLE_RIGHT_FOOT => 5,
        ROLE_CHEST => 6,
        ROLE_LEFT_ELBOW => 7,
        ROLE_RIGHT_ELBOW => 8,
        ROLE_LEFT_KNEE => 9,
        ROLE_RIGHT_KNEE => 10,
        _ => return None,
    })
}

impl Drop for SlimeVr {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Ok(mut s) = self.shared.lock() {
            s.writer = None;
        }
    }
}

/// Where the server listens for drivers.
pub fn socket_path() -> PathBuf {
    if cfg!(windows) {
        return PathBuf::from(r"\\.\pipe\SlimeVRDriver");
    }
    let dir = std::env::var_os("SLIMEVR_SOCKET_DIR").map(PathBuf::from).unwrap_or_else(|| {
        let data = std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
            .unwrap_or_default();
        data.join("dev.slimevr.SlimeVR")
    });
    dir.join("SlimeVRDriver")
}

type Stream = (Box<dyn Read + Send>, Box<dyn Write + Send>);

fn connect() -> std::io::Result<Stream> {
    #[cfg(unix)]
    {
        let s = std::os::unix::net::UnixStream::connect(socket_path())?;
        let w = s.try_clone()?;
        Ok((Box::new(s), Box::new(w)))
    }
    #[cfg(not(unix))]
    {
        let f = std::fs::OpenOptions::new().read(true).write(true).open(socket_path())?;
        let w = f.try_clone()?;
        Ok((Box::new(f), Box::new(w)))
    }
}

fn run(shared: Arc<Mutex<Shared>>, stop: Arc<std::sync::atomic::AtomicBool>) {
    use std::sync::atomic::Ordering;
    while !stop.load(Ordering::Relaxed) {
        let Ok((mut reader, writer)) = connect() else {
            std::thread::sleep(Duration::from_secs(2));
            continue;
        };
        tracing::info!("SlimeVR: connected to the server");
        if let Ok(mut s) = shared.lock() {
            s.writer = Some(writer);
            s.connected = true;
            s.announced = false;
        }
        let mut buf: Vec<u8> = Vec::new();
        let mut chunk = [0u8; 4096];
        loop {
            match reader.read(&mut chunk) {
                Ok(0) | Err(_) => break,
                Ok(n) => buf.extend_from_slice(&chunk[..n]),
            }
            while buf.len() >= 4 {
                let len = u32::from_le_bytes(buf[..4].try_into().unwrap()) as usize;
                if !(4..=65536).contains(&len) {
                    buf.clear();
                    break;
                }
                if buf.len() < len {
                    break;
                }
                if let Ok(mut s) = shared.lock() {
                    handle(&mut s, &buf[4..len]);
                }
                buf.drain(..len);
            }
            if stop.load(Ordering::Relaxed) {
                return;
            }
        }
        tracing::info!("SlimeVR: disconnected");
        if let Ok(mut s) = shared.lock() {
            s.writer = None;
            s.connected = false;
            s.trackers.clear();
        }
    }
}

/// Apply one `ProtobufMessage` from the server.
fn handle(s: &mut Shared, msg: &[u8]) {
    for (field, value) in fields(msg) {
        let Value::Bytes(body) = value else { continue };
        match field {
            1 => {
                let mut id = 0;
                let (mut p, mut q) = ([None; 3], [0.0f32, 0.0, 0.0, 1.0]);
                for (f, v) in fields(body) {
                    match (f, v) {
                        (1, Value::Varint(x)) => id = x as i64,
                        (2..=4, Value::Fixed32(x)) => p[f as usize - 2] = Some(f32::from_bits(x)),
                        (5..=8, Value::Fixed32(x)) => q[f as usize - 5] = f32::from_bits(x),
                        _ => {}
                    }
                }
                if let Some(r) = s.trackers.get_mut(&id) {
                    let rot = Quat::from_xyzw(q[0], q[1], q[2], q[3]).normalize();
                    let pos = match p {
                        [Some(x), Some(y), Some(z)] => Vec3::new(x, y, z),
                        _ => r.pose.map(|(p, _)| p).unwrap_or(Vec3::ZERO),
                    };
                    r.pose = Some((pos, rot));
                    r.at = Instant::now();
                }
            }
            3 => {
                let (mut id, mut role, mut serial, mut name) = (0, 0, String::new(), String::new());
                for (f, v) in fields(body) {
                    match (f, v) {
                        (1, Value::Varint(x)) => id = x as i64,
                        (2, Value::Bytes(b)) => serial = String::from_utf8_lossy(b).into_owned(),
                        (3, Value::Bytes(b)) => name = String::from_utf8_lossy(b).into_owned(),
                        (4, Value::Varint(x)) => role = x as i64,
                        _ => {}
                    }
                }
                s.trackers.insert(id, Remote { role, serial, name, pose: None, at: Instant::now() });
            }
            _ => {}
        }
    }
}

// ------------------------------------------------------------------ protobuf

enum Value<'a> {
    Varint(u64),
    Fixed32(u32),
    Fixed64,
    Bytes(&'a [u8]),
}

fn varint(data: &[u8], at: &mut usize) -> Option<u64> {
    let mut v = 0u64;
    for shift in (0..64).step_by(7) {
        let b = *data.get(*at)?;
        *at += 1;
        v |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            return Some(v);
        }
    }
    None
}

/// Top-level fields of a protobuf message (stops at the first malformed one).
fn fields(data: &[u8]) -> Vec<(u32, Value<'_>)> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < data.len() {
        let Some(tag) = varint(data, &mut at) else { break };
        let (field, wire) = ((tag >> 3) as u32, tag & 7);
        let value = match wire {
            0 => match varint(data, &mut at) {
                Some(v) => Value::Varint(v),
                None => break,
            },
            5 => match data.get(at..at + 4) {
                Some(b) => {
                    at += 4;
                    Value::Fixed32(u32::from_le_bytes(b.try_into().unwrap()))
                }
                None => break,
            },
            1 => {
                at += 8;
                Value::Fixed64
            }
            2 => {
                let Some(len) = varint(data, &mut at) else { break };
                let Some(b) = data.get(at..at + len as usize) else { break };
                at += len as usize;
                Value::Bytes(b)
            }
            _ => break,
        };
        out.push((field, value));
    }
    out
}

fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

fn field_varint(field: u32, v: u64) -> Vec<u8> {
    let mut out = Vec::new();
    put_varint(&mut out, (field as u64) << 3);
    put_varint(&mut out, v);
    out
}

fn field_float(out: &mut Vec<u8>, field: u32, v: f32) {
    put_varint(out, ((field as u64) << 3) | 5);
    out.extend(v.to_le_bytes());
}

fn field_bytes(out: &mut Vec<u8>, field: u32, b: &[u8]) {
    put_varint(out, ((field as u64) << 3) | 2);
    put_varint(out, b.len() as u64);
    out.extend_from_slice(b);
}

/// A `ProtobufMessage` holding `body` in its oneof `field`.
fn wrap(field: u32, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    field_bytes(&mut out, field, body);
    out
}

/// Length-prefixed bridge frame (the length counts its own 4 bytes).
fn frame(msg: &[u8]) -> Vec<u8> {
    let mut out = ((msg.len() + 4) as u32).to_le_bytes().to_vec();
    out.extend_from_slice(msg);
    out
}

fn tracker_added(id: i64, role: i64, name: &str) -> Vec<u8> {
    let mut out = field_varint(1, id as u64);
    field_bytes(&mut out, 2, format!("zestengine-{name}").as_bytes());
    field_bytes(&mut out, 3, name.as_bytes());
    out.extend(field_varint(4, role as u64));
    field_bytes(&mut out, 5, b"ZestEngine");
    out
}

fn position(id: i64, p: Vec3, r: Quat) -> Vec<u8> {
    let mut out = field_varint(1, id as u64);
    for (f, v) in [(2, p.x), (3, p.y), (4, p.z), (5, r.x), (6, r.y), (7, r.z), (8, r.w)] {
        field_float(&mut out, f, v);
    }
    out.extend(field_varint(9, 3));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_messages_become_role_targets() {
        let mut s = Shared::default();
        let added = wrap(3, &tracker_added(7, ROLE_WAIST, "Waist"));
        handle(&mut s, &added);
        let pos = wrap(1, &position(7, Vec3::new(0.1, 0.95, -0.2), Quat::from_rotation_y(0.5)));
        handle(&mut s, &pos);
        let shared = Arc::new(Mutex::new(s));
        let svr = SlimeVr { shared, stop: Arc::new(std::sync::atomic::AtomicBool::new(true)) };
        let t = svr.targets();
        let (p, r) = t.hips.expect("waist becomes hips");
        assert!((p - Vec3::new(0.1, 0.95, -0.2)).length() < 1e-6);
        assert!(r.angle_between(Quat::from_rotation_y(0.5)) < 1e-5);
        assert!(t.left_foot.is_none());
        let d = svr.devices();
        assert_eq!((d.len(), d[0].id.as_str(), d[0].native), (1, "slimevr:zestengine-Waist", Some(3)), "keyed by serial");
    }

    #[test]
    fn frames_carry_their_own_length() {
        let f = frame(&wrap(6, &field_varint(1, 2)));
        assert_eq!(u32::from_le_bytes(f[..4].try_into().unwrap()) as usize, f.len());
        let fs = fields(&f[4..]);
        assert!(matches!(fs.as_slice(), [(6, Value::Bytes(_))]));
    }

    #[cfg(unix)]
    #[test]
    fn talks_to_a_server_over_the_socket() {
        let dir = std::env::temp_dir().join(format!("zestengine-slimevr-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("SlimeVRDriver");
        let _ = std::fs::remove_file(&path);
        let listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
        // SAFETY: tests in this module are the only readers of this variable.
        unsafe { std::env::set_var("SLIMEVR_SOCKET_DIR", &dir) };
        let svr = SlimeVr::start();
        let (mut server, _) = listener.accept().unwrap();
        let mut msgs = frame(&wrap(3, &tracker_added(4, ROLE_LEFT_FOOT, "Left Foot")));
        msgs.extend(frame(&wrap(1, &position(4, Vec3::new(0.15, 0.1, 0.05), Quat::IDENTITY))));
        server.write_all(&msgs).unwrap();
        let mut got = None;
        for _ in 0..200 {
            if let Some(p) = svr.targets().left_foot {
                got = Some(p);
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!((got.expect("left foot from the server").0 - Vec3::new(0.15, 0.1, 0.05)).length() < 1e-6);
        // We announce the headset and stream its pose.
        svr.send_devices(Some((Vec3::new(0.0, 1.7, 0.0), Quat::IDENTITY)), None, None);
        server.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        let mut head = [0u8; 4];
        server.read_exact(&mut head).unwrap();
        let len = u32::from_le_bytes(head) as usize;
        let mut body = vec![0u8; len - 4];
        server.read_exact(&mut body).unwrap();
        assert!(matches!(fields(&body).as_slice(), [(6, Value::Bytes(_))]), "version first");
        drop(svr);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
