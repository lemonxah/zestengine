//! OSCQuery for the avatar OSC bridge: the JSON tree that tells tools which
//! `/avatar/parameters/...` addresses exist, a tiny HTTP responder for it, and
//! the mDNS announcement (`_oscjson._tcp` and `_osc._udp`) that lets tools
//! such as VRCFaceTracking find the ports without typing them in. No sockets
//! here (the engine's `avatar_osc_host` owns them); everything is bytes in and
//! bytes out so it can be tested.
//!
//! The tree follows the public OSCQuery proposal (`FULL_PATH`, `ACCESS` 0-3,
//! `TYPE` tags `i` `f` `T`, `VALUE`, `HOST_INFO`), the same shape VRChat's
//! avatar tree has, so existing tools read it unchanged.

use crate::avatar_params::AvatarParamKind;

/// Service type tools browse for to read the tree.
pub const SERVICE_JSON: &str = "_oscjson._tcp.local";
/// Service type announcing where OSC is received.
pub const SERVICE_OSC: &str = "_osc._udp.local";
/// mDNS multicast group and port.
pub const MDNS_GROUP: [u8; 4] = [224, 0, 0, 251];
pub const MDNS_PORT: u16 = 5353;

fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o
}

fn type_tag(k: AvatarParamKind) -> &'static str {
    match k {
        AvatarParamKind::Bool => "T",
        AvatarParamKind::Int => "i",
        AvatarParamKind::Float => "f",
    }
}

fn value_json(k: AvatarParamKind, v: f32) -> String {
    match k {
        AvatarParamKind::Bool => (v != 0.0).to_string(),
        AvatarParamKind::Int => format!("{}", v.round() as i64),
        AvatarParamKind::Float => format!("{}", if v.is_finite() { v } else { 0.0 }),
    }
}

/// The whole tree for an avatar with `params` (name, kind, current value).
pub fn tree_json(params: &[(String, AvatarParamKind, f32)]) -> String {
    let mut s = String::from(
        "{\"DESCRIPTION\":\"root node\",\"FULL_PATH\":\"/\",\"ACCESS\":0,\"CONTENTS\":{\"avatar\":{\"FULL_PATH\":\"/avatar\",\"ACCESS\":2,\"CONTENTS\":{\"change\":{\"FULL_PATH\":\"/avatar/change\",\"ACCESS\":3,\"TYPE\":\"s\",\"VALUE\":[\"\"]},\"parameters\":{\"FULL_PATH\":\"/avatar/parameters\",\"ACCESS\":2,\"CONTENTS\":{",
    );
    for (i, (name, kind, v)) in params.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        let n = esc(name);
        s.push_str(&format!(
            "\"{n}\":{{\"FULL_PATH\":\"/avatar/parameters/{n}\",\"ACCESS\":3,\"TYPE\":\"{}\",\"VALUE\":[{}]}}",
            type_tag(*kind),
            value_json(*kind, *v)
        ));
    }
    s.push_str("}}}}}");
    s
}

/// `?HOST_INFO` answer.
pub fn host_info_json(name: &str, osc_port: u16) -> String {
    format!(
        "{{\"NAME\":\"{}\",\"OSC_PORT\":{osc_port},\"OSC_TRANSPORT\":\"UDP\",\"EXTENSIONS\":{{\"ACCESS\":true,\"VALUE\":true,\"TYPE\":true,\"DESCRIPTION\":true}}}}",
        esc(name)
    )
}

/// The sub-tree at `path` of a [`tree_json`] document: the node whose
/// `FULL_PATH` is `path`, found by walking the object text (the format is our
/// own, so a small scanner is enough). `None` for an unknown path.
pub fn node_at(tree: &str, path: &str) -> Option<String> {
    let want = format!("\"FULL_PATH\":\"{}\"", path);
    let at = tree.find(&want)?;
    // The node's object opens at the `{` before this key.
    let start = tree[..at].rfind('{')?;
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc_next = false;
    for (i, c) in tree[start..].char_indices() {
        if in_str {
            if esc_next {
                esc_next = false;
            } else if c == '\\' {
                esc_next = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(tree[start..start + i + 1].to_owned());
                }
            }
            _ => {}
        }
    }
    None
}

/// An HTTP/1.1 response to `request` (the bytes read so far). `None` while the
/// request head is incomplete.
pub fn http_response(request: &[u8], tree: &str, host_info: &str) -> Option<Vec<u8>> {
    let text = std::str::from_utf8(request).ok()?;
    if !text.contains("\r\n\r\n") && !text.contains("\n\n") {
        return None;
    }
    let line = text.lines().next().unwrap_or("");
    let mut parts = line.split_whitespace();
    let (method, target) = (parts.next().unwrap_or(""), parts.next().unwrap_or("/"));
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let (status, body) = if method != "GET" {
        ("405 Method Not Allowed", String::new())
    } else if query.split('&').any(|q| q == "HOST_INFO") {
        ("200 OK", host_info.to_owned())
    } else {
        let path = if path.len() > 1 { path.trim_end_matches('/') } else { path };
        match node_at(tree, path) {
            Some(n) => ("200 OK", n),
            None => ("404 Not Found", String::new()),
        }
    };
    Some(
        format!("HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n{body}", body.len())
            .into_bytes(),
    )
}

// --------------------------------------------------------------------- mDNS

fn name(out: &mut Vec<u8>, dotted: &str) {
    for label in dotted.split('.').filter(|l| !l.is_empty()) {
        out.push(label.len().min(63) as u8);
        out.extend_from_slice(&label.as_bytes()[..label.len().min(63)]);
    }
    out.push(0);
}

fn record(out: &mut Vec<u8>, owner: &str, rtype: u16, cache_flush: bool, data: &[u8]) {
    name(out, owner);
    out.extend_from_slice(&rtype.to_be_bytes());
    out.extend_from_slice(&(if cache_flush { 0x8001u16 } else { 1 }).to_be_bytes());
    out.extend_from_slice(&120u32.to_be_bytes());
    out.extend_from_slice(&(data.len() as u16).to_be_bytes());
    out.extend_from_slice(data);
}

/// An unsolicited mDNS response announcing two services under `instance`
/// (`_oscjson._tcp` on `http_port`, `_osc._udp` on `osc_port`) at `ip`.
/// Records: PTR, SRV, TXT per service and one A. Sent on start and every few
/// seconds, as the goodbye-less minimum a tool browsing later still hears.
pub fn mdns_announcement(instance: &str, host: &str, ip: [u8; 4], http_port: u16, osc_port: u16) -> Vec<u8> {
    let mut o = Vec::with_capacity(400);
    // ID 0, flags: response + authoritative, 0 questions, 7 answers.
    o.extend_from_slice(&[0, 0, 0x84, 0, 0, 0, 0, 7, 0, 0, 0, 0]);
    let host_fqdn = format!("{host}.local");
    for (service, port) in [(SERVICE_JSON, http_port), (SERVICE_OSC, osc_port)] {
        let inst = format!("{instance}.{service}");
        let mut ptr = Vec::new();
        name(&mut ptr, &inst);
        record(&mut o, service, 12, false, &ptr);
        let mut srv = vec![0, 0, 0, 0];
        srv.extend_from_slice(&port.to_be_bytes());
        name(&mut srv, &host_fqdn);
        record(&mut o, &inst, 33, true, &srv);
        record(&mut o, &inst, 16, true, &[0]);
    }
    record(&mut o, &host_fqdn, 1, true, &ip);
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> Vec<(String, AvatarParamKind, f32)> {
        vec![("Hat".into(), AvatarParamKind::Bool, 1.0), ("Outfit".into(), AvatarParamKind::Int, 2.0), ("Smile".into(), AvatarParamKind::Float, 0.5)]
    }

    #[test]
    fn the_tree_lists_declared_parameters_with_types_and_values() {
        let t = tree_json(&params());
        assert!(t.contains("\"FULL_PATH\":\"/avatar/parameters/Hat\",\"ACCESS\":3,\"TYPE\":\"T\",\"VALUE\":[true]"));
        assert!(t.contains("\"/avatar/parameters/Outfit\",\"ACCESS\":3,\"TYPE\":\"i\",\"VALUE\":[2]"));
        assert!(t.contains("\"TYPE\":\"f\",\"VALUE\":[0.5]"));
        assert!(t.contains("/avatar/change"));
        // Names with quotes are escaped, so the document stays valid.
        let weird = tree_json(&[("A\"B".into(), AvatarParamKind::Bool, 0.0)]);
        assert!(weird.contains("A\\\"B"));
    }

    #[test]
    fn a_node_can_be_fetched_by_path() {
        let t = tree_json(&params());
        let n = node_at(&t, "/avatar/parameters/Outfit").unwrap();
        assert!(n.starts_with('{') && n.ends_with('}') && n.contains("\"TYPE\":\"i\"") && !n.contains("Hat"));
        let all = node_at(&t, "/avatar/parameters").unwrap();
        assert!(all.contains("Hat") && all.contains("Smile"));
        assert!(node_at(&t, "/nope").is_none());
    }

    #[test]
    fn http_answers_get_requests_and_waits_for_a_full_head() {
        let t = tree_json(&params());
        let h = host_info_json("Zest", 9000);
        assert!(http_response(b"GET / HTTP/1.1\r\nHost: x\r\n", &t, &h).is_none(), "head incomplete");
        let r = String::from_utf8(http_response(b"GET /?HOST_INFO HTTP/1.1\r\n\r\n", &t, &h).unwrap()).unwrap();
        assert!(r.starts_with("HTTP/1.1 200 OK") && r.contains("\"OSC_PORT\":9000"));
        let r = String::from_utf8(http_response(b"GET /avatar/parameters/Hat HTTP/1.1\r\n\r\n", &t, &h).unwrap()).unwrap();
        assert!(r.contains("200 OK") && r.contains("\"TYPE\":\"T\""));
        let r = String::from_utf8(http_response(b"GET /missing HTTP/1.1\r\n\r\n", &t, &h).unwrap()).unwrap();
        assert!(r.starts_with("HTTP/1.1 404"));
        let r = String::from_utf8(http_response(b"POST / HTTP/1.1\r\n\r\n", &t, &h).unwrap()).unwrap();
        assert!(r.starts_with("HTTP/1.1 405"));
        // Content-Length matches the body.
        let r = String::from_utf8(http_response(b"GET / HTTP/1.1\r\n\r\n", &t, &h).unwrap()).unwrap();
        let (head, body) = r.split_once("\r\n\r\n").unwrap();
        assert!(head.contains(&format!("Content-Length: {}", body.len())));
    }

    #[test]
    fn the_mdns_announcement_is_a_well_formed_response() {
        let p = mdns_announcement("Zest", "zest-host", [127, 0, 0, 1], 38000, 9000);
        assert_eq!(&p[..4], &[0, 0, 0x84, 0]);
        assert_eq!(u16::from_be_bytes([p[6], p[7]]), 7, "answer count");
        // Walk the records: every name and rdata length stays inside the packet.
        let mut at = 12;
        let mut kinds = Vec::new();
        for _ in 0..7 {
            while p[at] != 0 {
                at += 1 + p[at] as usize;
            }
            at += 1;
            let rtype = u16::from_be_bytes([p[at], p[at + 1]]);
            let len = u16::from_be_bytes([p[at + 8], p[at + 9]]) as usize;
            kinds.push(rtype);
            at += 10 + len;
            assert!(at <= p.len());
        }
        assert_eq!(at, p.len());
        assert_eq!(kinds, vec![12, 33, 16, 12, 33, 16, 1]);
        let text = String::from_utf8_lossy(&p).into_owned();
        assert!(text.contains("_oscjson") && text.contains("_osc") && text.contains("zest-host"));
        // The SRV for the HTTP service carries its port.
        assert!(p.windows(2).any(|w| w == 38000u16.to_be_bytes()));
    }
}
