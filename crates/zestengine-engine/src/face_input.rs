//! This machine's face tracking: the headset's OpenXR face / eye tracking when
//! the runtime offers it (WiVRn, Virtual Desktop, SteamVR), else an opt-in OSC
//! receiver for external tools. Both produce a `FaceFrame`.

use std::net::UdpSocket;
use std::time::{Duration, Instant};

use zestengine_core::FaceFrame;
use glam::{Quat, Vec2, Vec3};

/// OSC data older than this counts as no tracking.
const OSC_STALE: Duration = Duration::from_secs(1);

#[derive(Default)]
pub struct FaceInput {
    osc: Option<OscReceiver>,
}

struct OscReceiver {
    socket: UdpSocket,
    port: u16,
    frame: FaceFrame,
    last: Option<Instant>,
    /// `/avatar/parameters/...` messages the face frame did not take, for the
    /// avatar OSC bridge (bounded; the bridge filters to declared parameters).
    avatar_msgs: Vec<(String, f32)>,
}

/// Most avatar messages held between two frames.
const MAX_AVATAR_MSGS: usize = 512;

impl FaceInput {
    /// Start (`Some(port)`) or stop the OSC receiver.
    pub fn set_osc(&mut self, port: Option<u16>) {
        match port {
            Some(port) if self.osc.as_ref().is_some_and(|o| o.port == port) => {}
            Some(port) => {
                self.osc = match UdpSocket::bind(("0.0.0.0", port)).and_then(|s| s.set_nonblocking(true).map(|_| s)) {
                    Ok(socket) => {
                        tracing::info!("face tracking: OSC receiver on UDP {port}");
                        Some(OscReceiver { socket, port, frame: FaceFrame::default(), last: None, avatar_msgs: Vec::new() })
                    }
                    Err(e) => {
                        tracing::warn!("face tracking: can't listen for OSC on UDP {port}: {e}");
                        None
                    }
                }
            }
            None => {
                if self.osc.take().is_some() {
                    tracing::info!("face tracking: OSC receiver stopped");
                }
            }
        }
    }

    /// Avatar parameter messages received since the last call.
    pub fn take_avatar_osc(&mut self) -> Vec<(String, f32)> {
        self.osc.as_mut().map(|o| std::mem::take(&mut o.avatar_msgs)).unwrap_or_default()
    }

        pub fn osc_port(&self) -> Option<u16> {
        self.osc.as_ref().map(|o| o.port)
    }

    /// This frame's face: headset first, then fresh OSC data.
    pub fn poll(&mut self, xr: Option<&zestengine_xr::XrSession>) -> Option<FaceFrame> {
        let osc = self.osc.as_mut().and_then(|o| o.poll());
        let mut headset = xr.and_then(|s| s.face_weights()).map(|w| FaceFrame::from_fb_face2(&w));
        if let Some(gaze) = xr.and_then(|s| s.eye_gaze()) {
            let f = headset.get_or_insert_with(|| osc.clone().unwrap_or_default());
            let g = gaze_xy(gaze);
            f.gaze_left = g;
            f.gaze_right = g;
            f.has_gaze = true;
        }
        headset.or(osc)
    }
}

/// A gaze rotation (forward -Z) as -1..1 right / up, 1 being 45 degrees.
fn gaze_xy(q: Quat) -> Vec2 {
    let d = q * Vec3::NEG_Z;
    let scale = 1.0 / std::f32::consts::FRAC_PI_4;
    Vec2::new(d.x.atan2(-d.z) * scale, d.y.atan2(Vec2::new(d.x, d.z).length()) * scale).clamp(Vec2::splat(-1.0), Vec2::ONE)
}

impl OscReceiver {
    fn poll(&mut self) -> Option<FaceFrame> {
        let mut buf = [0u8; 8192];
        while let Ok((n, _)) = self.socket.recv_from(&mut buf) {
            let mut any = false;
            let (frame, msgs) = (&mut self.frame, &mut self.avatar_msgs);
            for_each_message(&buf[..n], &mut |address, value| {
                let used = frame.set_param(address, value);
                any |= used;
                if !used && address.starts_with(zestengine_core::avatar_osc::PARAM_PREFIX) && msgs.len() < MAX_AVATAR_MSGS {
                    msgs.push((address.to_owned(), value));
                }
            });
            if any {
                self.last = Some(Instant::now());
            }
        }
        self.last.filter(|t| t.elapsed() < OSC_STALE).map(|_| self.frame.clone())
    }
}

/// Visit every (address, first numeric argument) in an OSC packet, bundles
/// included. Floats, ints, doubles and true / false arguments are read.
pub fn for_each_message(data: &[u8], f: &mut dyn FnMut(&str, f32)) {
    if data.starts_with(b"#bundle\0") {
        let mut at = 16;
        while at + 4 <= data.len() {
            let size = i32::from_be_bytes(data[at..at + 4].try_into().unwrap()).max(0) as usize;
            at += 4;
            if at + size > data.len() {
                return;
            }
            for_each_message(&data[at..at + size], f);
            at += size;
        }
        return;
    }
    let Some((address, at)) = osc_string(data, 0) else { return };
    let Some((tags, at)) = osc_string(data, at) else { return };
    let Some(tags) = tags.strip_prefix(',') else { return };
    let value = match tags.chars().next() {
        Some('f') => data.get(at..at + 4).map(|b| f32::from_be_bytes(b.try_into().unwrap())),
        Some('i') => data.get(at..at + 4).map(|b| i32::from_be_bytes(b.try_into().unwrap()) as f32),
        Some('d') => data.get(at..at + 8).map(|b| f64::from_be_bytes(b.try_into().unwrap()) as f32),
        Some('T') => Some(1.0),
        Some('F') => Some(0.0),
        _ => None,
    };
    if let Some(v) = value {
        f(address, v);
    }
}

/// A null-terminated string padded to 4 bytes, and the offset after it.
fn osc_string(data: &[u8], at: usize) -> Option<(&str, usize)> {
    let rest = data.get(at..)?;
    let end = rest.iter().position(|&b| b == 0)?;
    let s = std::str::from_utf8(&rest[..end]).ok()?;
    Some((s, at + (end + 4) / 4 * 4))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pad(s: &str) -> Vec<u8> {
        let mut b = s.as_bytes().to_vec();
        b.push(0);
        while !b.len().is_multiple_of(4) {
            b.push(0);
        }
        b
    }

    fn message(address: &str, v: f32) -> Vec<u8> {
        let mut b = pad(address);
        b.extend(pad(",f"));
        b.extend(v.to_be_bytes());
        b
    }

    #[test]
    fn parses_messages_and_bundles() {
        let mut bundle = pad("#bundle");
        bundle.extend([0u8; 8]);
        for m in [message("/avatar/parameters/FT/v2/JawOpen", 0.5), message("/avatar/parameters/FT/v2/EyeLeftX", -0.25)] {
            bundle.extend((m.len() as i32).to_be_bytes());
            bundle.extend(m);
        }
        let mut seen = Vec::new();
        for_each_message(&bundle, &mut |a, v| seen.push((a.to_owned(), v)));
        assert_eq!(seen, vec![("/avatar/parameters/FT/v2/JawOpen".to_owned(), 0.5), ("/avatar/parameters/FT/v2/EyeLeftX".to_owned(), -0.25)]);
    }

    #[test]
    fn receiver_turns_packets_into_a_frame() {
        let mut input = FaceInput::default();
        let port = {
            let probe = UdpSocket::bind("127.0.0.1:0").unwrap();
            probe.local_addr().unwrap().port()
        };
        input.set_osc(Some(port));
        assert_eq!(input.osc_port(), Some(port));
        let sender = UdpSocket::bind("127.0.0.1:0").unwrap();
        sender.send_to(&message("/avatar/parameters/FT/v2/JawOpen", 0.75), ("127.0.0.1", port)).unwrap();
        let mut frame = None;
        for _ in 0..50 {
            frame = input.poll(None);
            if frame.is_some() {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let f = frame.expect("frame from OSC");
        assert!((f.weights[zestengine_core::expression_index("JawOpen").unwrap()] - 0.75).abs() < 1e-6);
        input.set_osc(None);
        assert!(input.poll(None).is_none());
    }

    #[test]
    fn avatar_parameter_messages_are_kept_for_the_bridge_and_face_ones_are_not() {
        let mut input = FaceInput::default();
        let port = {
            let probe = UdpSocket::bind("127.0.0.1:0").unwrap();
            probe.local_addr().unwrap().port()
        };
        input.set_osc(Some(port));
        let sender = UdpSocket::bind("127.0.0.1:0").unwrap();
        sender.send_to(&message("/avatar/parameters/Hat", 1.0), ("127.0.0.1", port)).unwrap();
        sender.send_to(&message("/avatar/parameters/FT/v2/JawOpen", 0.5), ("127.0.0.1", port)).unwrap();
        sender.send_to(&message("/input/Jump", 1.0), ("127.0.0.1", port)).unwrap();
        let mut got = Vec::new();
        for _ in 0..50 {
            input.poll(None);
            got.extend(input.take_avatar_osc());
            if !got.is_empty() {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(got, vec![("/avatar/parameters/Hat".to_owned(), 1.0)], "the face parameter went to the face frame, other addresses are ignored");
        assert!(input.take_avatar_osc().is_empty(), "taken once");
    }

    #[test]
    fn gaze_converts_to_unit_range() {
        let right = Quat::from_rotation_y(-std::f32::consts::FRAC_PI_4);
        let g = gaze_xy(right);
        assert!((g.x - 1.0).abs() < 1e-4 && g.y.abs() < 1e-4, "{g}");
    }
}
