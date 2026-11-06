//! The `/avatar/parameters` OSC bridge for the local avatar: messages from
//! external tools (received by the face-tracking OSC socket, UDP 9000 by
//! default) write the avatar's *declared* parameters, and changes of declared
//! parameters go out to `127.0.0.1:9001`, with `/avatar/change <id>` when the
//! avatar changes. The mapping and filter live in `zestengine_core::avatar_osc`;
//! this file owns the socket and the avatar lookup. It runs only while the
//! OSC setting is on (Tracking Input settings), like face tracking OSC.
//!
//! The output port is a project setting (`osc_out_port`, default 9001) and
//! OSCQuery can be switched on beside it: a loopback HTTP server answering
//! the parameter tree and an mDNS announcement of `_oscjson._tcp` and
//! `_osc._udp` (`zestengine_core::avatar_oscquery`). The announcement is sent
//! every few seconds; it was checked against its own decoder, not against a
//! real browsing tool.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, UdpSocket};

use zestengine_core::avatar_osc::{self, OscBridge};
use zestengine_core::avatar_oscquery;
use zestengine_core::{AvatarParameters, BuiltinParam, ObjectId};

use crate::scene::LoadedScene;

/// Seconds between mDNS announcements and tree refreshes.
const QUERY_INTERVAL: f64 = 3.0;
/// A connection that has not sent a full request head by then is dropped.
const QUERY_TIMEOUT: f64 = 2.0;

#[derive(Default)]
pub struct AvatarOsc {
    /// Where messages go (0 = the default, 9001).
    out_port: u16,
    /// OSCQuery on, with the port OSC is received on.
    query_in_port: Option<u16>,
    http: Option<TcpListener>,
    http_port: u16,
    mdns: Option<UdpSocket>,
    conns: Vec<(TcpStream, Vec<u8>, f64)>,
    tree: String,
    last_query: Option<f64>,
    out: Option<UdpSocket>,
    bridge: Option<OscBridge>,
    /// The local avatar the bridge was built for, and its store size.
    root: Option<(usize, ObjectId, usize)>,
    last_scan: Option<f64>,
    held: Vec<(usize, f32)>,
    messages: Vec<Vec<u8>>,
}

impl AvatarOsc {
    /// Apply the project's OSC output port and OSCQuery setting (`Some(port)`
    /// = advertise, with `port` as the receive port). Cheap to call every
    /// frame; sockets are rebuilt only when something changed.
    pub fn configure(&mut self, out_port: u16, query_in_port: Option<u16>) {
        let out_port = if out_port == 0 { avatar_osc::DEFAULT_OUT_PORT } else { out_port };
        if out_port != self.out_port {
            self.out_port = out_port;
            self.out = None;
            self.bridge = None;
        }
        if query_in_port != self.query_in_port {
            self.query_in_port = query_in_port;
            if query_in_port.is_none() {
                self.stop_query();
            }
        }
    }

    fn stop_query(&mut self) {
        self.http = None;
        self.mdns = None;
        self.conns.clear();
        self.last_query = None;
    }

    /// The OSCQuery HTTP port while it is running.
    #[allow(dead_code)] // introspected by external OSC tooling
    pub fn query_port(&self) -> Option<u16> {
        self.http.as_ref().map(|_| self.http_port)
    }

    /// Serve pending HTTP requests and announce on mDNS. `tree` is refreshed
    /// by the caller once the avatar's parameters are known.
    fn poll_query(&mut self, now: f64) {
        let Some(in_port) = self.query_in_port else { return };
        if self.http.is_none() {
            self.http = TcpListener::bind(("127.0.0.1", 0)).and_then(|l| l.set_nonblocking(true).map(|_| l)).ok();
            self.http_port = self.http.as_ref().and_then(|l| l.local_addr().ok()).map_or(0, |a| a.port());
        }
        let Some(listener) = &self.http else { return };
        while let Ok((c, _)) = listener.accept() {
            if c.set_nonblocking(true).is_ok() && self.conns.len() < 8 {
                self.conns.push((c, Vec::new(), now));
            }
        }
        let host = avatar_oscquery::host_info_json("Zest Engine", in_port);
        let tree = &self.tree;
        let mut buf = [0u8; 1024];
        self.conns.retain_mut(|(c, req, since)| {
            if now - *since > QUERY_TIMEOUT {
                return false;
            }
            match c.read(&mut buf) {
                Ok(0) => return false,
                Ok(n) if req.len() + n <= 4096 => req.extend_from_slice(&buf[..n]),
                Ok(_) => return false,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(_) => return false,
            }
            match avatar_oscquery::http_response(req, tree, &host) {
                Some(resp) => {
                    // The reply is small; a failed write just drops the client.
                    let _ = c.set_nonblocking(false);
                    let _ = c.set_write_timeout(Some(std::time::Duration::from_millis(50)));
                    let _ = c.write_all(&resp);
                    false
                }
                None => true,
            }
        });
        if self.last_query.is_none_or(|t| now - t >= QUERY_INTERVAL) {
            self.last_query = Some(now);
            if self.mdns.is_none() {
                self.mdns = UdpSocket::bind(("0.0.0.0", 0)).and_then(|s| s.set_nonblocking(true).map(|_| s)).ok();
            }
            if let Some(m) = &self.mdns {
                let pkt = avatar_oscquery::mdns_announcement("Zest Engine", "zest-engine", [127, 0, 0, 1], self.http_port, in_port);
                let _ = m.send_to(&pkt, (std::net::Ipv4Addr::from(avatar_oscquery::MDNS_GROUP), avatar_oscquery::MDNS_PORT));
            }
        }
    }

    /// One frame. `incoming` are the `/avatar/parameters/...` messages the
    /// OSC receiver took; `enabled` is the OSC setting.
    pub fn frame(&mut self, scene: &mut LoadedScene, incoming: Vec<(String, f32)>, enabled: bool, now: f64) {
        if !enabled {
            if self.out.take().is_some() || self.bridge.take().is_some() {
                self.root = None;
            }
            self.stop_query();
            return;
        }
        if self.out.is_none() {
            let port = if self.out_port == 0 { avatar_osc::DEFAULT_OUT_PORT } else { self.out_port };
            self.out = UdpSocket::bind(("127.0.0.1", 0)).and_then(|s| s.connect(("127.0.0.1", port)).map(|_| s)).and_then(|s| s.set_nonblocking(true).map(|_| s)).ok();
        }
        self.poll_query(now);
        // The local avatar: cached; rescanned at most once a second.
        let valid = self.root.filter(|&(i, id, len)| {
            scene.objects.get(i).is_some_and(|o| {
                o.id == id && o.components.iter().any(|c| c.as_any().downcast_ref::<AvatarParameters>().is_some_and(|a| a.ready && a.store.len() == len))
            })
        });
        if valid.is_none() {
            if self.last_scan.is_some_and(|t| now - t < 1.0) && incoming.is_empty() {
                return;
            }
            self.last_scan = Some(now);
            self.root = (0..scene.objects.len()).find_map(|i| {
                let a = scene.objects[i].components.iter().find_map(|c| c.as_any().downcast_ref::<AvatarParameters>())?;
                (a.ready && a.store.get_builtin(BuiltinParam::IsLocal) > 0.5 && scene.is_active(i)).then(|| (i, scene.objects[i].id, a.store.len()))
            });
            self.bridge = None;
        }
        let Some((i, id, _)) = self.root else { return };
        let Some(ap) = scene.objects[i].components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<AvatarParameters>()) else {
            return;
        };
        let store = &mut ap.store;
        // The OSCQuery tree follows the declared parameters.
        if self.query_in_port.is_some() && (self.bridge.is_none() || self.last_query == Some(now)) {
            let params: Vec<(String, zestengine_core::AvatarParamKind, f32)> =
                (0..store.len()).filter_map(|s| store.declared(s).map(|p| (store.name(s).to_owned(), p.kind, store.get(s)))).collect();
            self.tree = avatar_oscquery::tree_json(&params);
        }
        if self.bridge.is_none() {
            self.bridge = Some(OscBridge::new(store));
            // A new listener learns the avatar and every value.
            if let Some(out) = &self.out {
                let _ = out.send(&avatar_osc::encode_avatar_change(&id.to_string()));
            }
        }
        let Some(bridge) = self.bridge.as_mut() else { return };
        for (address, value) in incoming {
            if let Some((slot, v)) = bridge.accept(store, &address, value, now) {
                store.set(slot, v);
            }
        }
        self.held.clear();
        bridge.poll_pending(store, now, &mut self.held);
        for &(slot, v) in &self.held {
            store.set(slot, v);
        }
        self.messages.clear();
        bridge.changed_messages(store, &mut self.messages);
        if let Some(out) = &self.out {
            for m in &self.messages {
                let _ = out.send(m);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};
    use zestengine_assets::ObjectSource;
    use zestengine_core::{AvatarParam, AvatarParamKind};

    use crate::scene::SceneObject;

    fn scene_with_avatar() -> LoadedScene {
        let mut s = LoadedScene::empty();
        let mut ap = AvatarParameters { params: vec![AvatarParam::new("Hat", AvatarParamKind::Bool), AvatarParam::new("Smile", AvatarParamKind::Float)], ..Default::default() };
        ap.rebuild_store();
        s.objects.push(SceneObject {
            net_role: Default::default(),
            component_roles: Default::default(),
            id: ObjectId::new(),
            name: "Avatar".into(),
            render: None,
            extra_render: Vec::new(),
            source: ObjectSource::Empty,
            enabled: true,
            static_geometry: false,
            layer: 0,
            parent: None,
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            components: vec![Box::new(ap)],
        });
        s
    }

    fn value(s: &LoadedScene, name: &str) -> f32 {
        let ap = s.objects[0].components.iter().find_map(|c| c.as_any().downcast_ref::<AvatarParameters>()).unwrap();
        ap.store.get(ap.store.index_of(name).unwrap())
    }

    #[test]
    fn incoming_messages_write_declared_parameters_only() {
        let mut s = scene_with_avatar();
        let mut osc = AvatarOsc::default();
        let incoming = vec![
            ("/avatar/parameters/Hat".to_owned(), 1.0),
            ("/avatar/parameters/Smile".to_owned(), 0.5),
            // Built-ins and unknown names are ignored.
            ("/avatar/parameters/IsLocal".to_owned(), 0.0),
            ("/avatar/parameters/Grounded".to_owned(), 1.0),
            ("/avatar/parameters/Nope".to_owned(), 1.0),
        ];
        osc.frame(&mut s, incoming, true, 1.0);
        assert_eq!((value(&s, "Hat"), value(&s, "Smile")), (1.0, 0.5));
        assert_eq!(value(&s, "IsLocal"), 1.0);
        assert_eq!(value(&s, "Grounded"), 0.0);
        // Off: nothing is applied.
        osc.frame(&mut s, vec![("/avatar/parameters/Hat".to_owned(), 0.0)], false, 3.0);
        assert_eq!(value(&s, "Hat"), 1.0);
    }

    #[test]
    fn oscquery_serves_the_declared_parameters_over_loopback() {
        let mut s = scene_with_avatar();
        let mut osc = AvatarOsc::default();
        osc.configure(9055, Some(9000));
        osc.frame(&mut s, Vec::new(), true, 0.0);
        let port = osc.query_port().expect("http server started");
        let mut c = TcpStream::connect(("127.0.0.1", port)).unwrap();
        c.write_all(b"GET /avatar/parameters/Hat HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
        // The server answers on its next frames.
        for t in 1..20 {
            osc.frame(&mut s, Vec::new(), true, t as f64 * 0.01);
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        c.set_read_timeout(Some(std::time::Duration::from_secs(2))).unwrap();
        let mut body = String::new();
        c.read_to_string(&mut body).unwrap();
        assert!(body.starts_with("HTTP/1.1 200 OK"), "{body}");
        assert!(body.contains("/avatar/parameters/Hat") && body.contains("\"TYPE\":\"T\""), "{body}");
        // Turning it off closes the server; the output port follows the setting.
        osc.configure(9055, None);
        assert!(osc.query_port().is_none());
        assert_eq!(osc.out_port, 9055);
        osc.configure(0, None);
        assert_eq!(osc.out_port, avatar_osc::DEFAULT_OUT_PORT);
    }
}
