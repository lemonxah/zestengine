//! The editor's Network window: host / join (direct, from the LAN list, or
//! online by room code through a relay), then the session status, room
//! code, host migration (who takes over, a migration in progress), the
//! player list with link quality and verified player ids (kick / ban on the
//! host), bans, traffic and owned objects.

use crate::{icons::Icon, theme, widgets};
use egui::RichText;

use zestengine_engine::net::{ConnState, LanBrowser, NetSession, RelayPhase, short_fingerprint};
use zestengine_engine::server_browser::{BrowserStatus, ServerBrowser, SortKey};
use zestengine_engine::scene::LoadedScene;

/// Window state kept between frames.
pub struct NetWindowState {
    pub player_name: String,
    /// Relay server for online play (empty = the scene's Network Manager setting).
    pub relay: String,
    /// Join secret (empty = the scene's Network Manager setting).
    pub secret: String,
    pub account_name: String,
    pub account_password: String,
    show_secret: bool,
    show_account_password: bool,
    browser: Option<LanBrowser>,
    last_refresh: f64,
    /// The online server list (a directory query).
    servers: ServerBrowser,
    /// Set by the "Play as Server" button; the editor starts the local server and joins it.
    pub local_server_request: bool,
}

/// Join the local server at `addr` as the editor's player (the scene's Network Manager settings
/// with this window's name).
pub fn join_local(st: &NetWindowState, scene: &LoadedScene, addr: &str) -> std::io::Result<NetSession> {
    let (mut cfg, _) = zestengine_engine::net_sync::config_for(scene);
    cfg.player_name = st.player_name.clone();
    cfg.secret = String::new();
    cfg.relay = String::new();
    NetSession::join_with(addr, cfg)
}

impl Default for NetWindowState {
    fn default() -> Self {
        Self {
            player_name: std::env::var("ZESTENGINE_NAME").unwrap_or_else(|_| "Editor".to_string()),
            relay: std::env::var("ZESTENGINE_RELAY").unwrap_or_default(),
            secret: std::env::var("ZESTENGINE_SECRET").unwrap_or_default(),
            account_name: std::env::var("ZESTENGINE_ACCOUNT").unwrap_or_default(),
            account_password: std::env::var("ZESTENGINE_PASSWORD").unwrap_or_default(),
            show_secret: false,
            show_account_password: false,
            browser: None,
            last_refresh: -10.0,
            servers: ServerBrowser::default(),
            local_server_request: false,
        }
    }
}

enum Act {
    Host(u16),
    Join(String),
    Disconnect,
    EndSession,
    Kick(u64),
    Ban(u64),
    Unban(String),
}

/// Draw the window. Returns an error to show in the status bar, if any.
pub fn network_window(
    ctx: &egui::Context,
    open: &mut bool,
    net: &mut Option<NetSession>,
    addr: &mut String,
    st: &mut NetWindowState,
    scene: &LoadedScene,
) -> Option<String> {
    let mut act = None;
    let now = ctx.input(|i| i.time);
    egui::Window::new("Network")
        .open(open)
        .resizable(false)
        .default_width(320.0)
        .show(ctx, |ui| match net.as_ref() {
            Some(s) => connected_ui(ui, s, scene, &mut act),
            None => {
                let (cfg, port) = zestengine_engine::net_sync::config_for(scene);
                widgets::eyebrow(ui, "Session");
                widgets::property_row(ui, "Name", |ui| ui.text_edit_singleline(&mut st.player_name));
                widgets::property_row(ui, "Address", |ui| {
                    ui.add(egui::TextEdit::singleline(addr).hint_text("host:port or room code"))
                        .on_hover_text("Host: the port to listen on. Join: host:port on the LAN / a forwarded port, or a room code (ABC123, or ABC123@relay:port) to join online")
                });
                widgets::property_row(ui, "Relay", |ui| {
                    let hint = if cfg.relay.is_empty() { "none (LAN / direct only)".to_string() } else { cfg.relay.clone() };
                    ui.add(egui::TextEdit::singleline(&mut st.relay).hint_text(hint))
                        .on_hover_text("Relay server (host:port, default port 47800) for online play without port forwarding. Run one with `zestengine-relay`.")
                });
                widgets::property_row(ui, "Secret", |ui| {
                    ui.horizontal(|ui| {
                        let hint = if cfg.secret.is_empty() { "none (open session)" } else { "from the Network Manager" };
                        ui.add(egui::TextEdit::singleline(&mut st.secret).password(!st.show_secret).hint_text(hint).desired_width(150.0))
                            .on_hover_text("Players must enter the same secret to join; it also keys the session encryption");
                        let eye = if st.show_secret { Icon::EyeOff } else { Icon::Eye };
                        if widgets::icon_button(ui, eye, "Show / hide", st.show_secret).clicked() {
                            st.show_secret = !st.show_secret;
                        }
                        if widgets::icon_button(ui, Icon::Reset, "Generate a random secret (80 bits: no key stretching needed)", false).clicked() {
                            st.secret = zestengine_engine::net::generate_secret();
                            st.show_secret = true;
                        }
                    })
                });
                widgets::property_row(ui, "Account", |ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut st.account_name)
                            .hint_text("optional username"),
                    )
                    .on_hover_text("Hosts with account auth enabled require this");
                });
                widgets::property_row(ui, "Password", |ui| {
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::TextEdit::singleline(&mut st.account_password)
                                .password(!st.show_account_password)
                                .hint_text("optional password")
                                .desired_width(150.0),
                        );
                        let eye = if st.show_account_password { Icon::EyeOff } else { Icon::Eye };
                        if widgets::icon_button(ui, eye, "Show / hide", st.show_account_password).clicked() {
                            st.show_account_password = !st.show_account_password;
                        }
                    });
                });
                let me = zestengine_engine::net::Identity::install().fingerprint();
                widgets::property_row(ui, "Player ID", |ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(short_fingerprint(&me)).font(theme::mono(11.0))).on_hover_text(&me);
                        if widgets::icon_button(ui, Icon::Duplicate, "Copy your full player id (hosts put it on allow / ban lists)", false).clicked() {
                            ui.ctx().copy_text(me.clone());
                        }
                    })
                });
                ui.horizontal(|ui| {
                    if widgets::primary_button(ui, Some(Icon::Play), "Host").clicked() {
                        let p = addr.rsplit(':').next().and_then(|p| p.parse::<u16>().ok()).unwrap_or(port);
                        act = Some(Act::Host(p));
                    }
                    if widgets::secondary_button(ui, None, "Join").clicked() {
                        act = Some(Act::Join(addr.clone()));
                    }
                    if widgets::secondary_button(ui, Some(Icon::Play), "Play as Server")
                        .on_hover_text("Run the saved scene as a local dedicated server (Server-only content kept, Client-only removed) and join it as a player")
                        .clicked()
                    {
                        st.local_server_request = true;
                    }
                });
                let online = !st.relay.trim().is_empty() || !cfg.relay.is_empty();
                widgets::info_note(
                    ui,
                    &format!(
                        "{} · up to {} players. Host binds the port in Address{}. Join takes host:port, or a room code{}. Networked objects replicate while playing.",
                        cfg.session_name,
                        cfg.max_players,
                        if online { " and opens a room on the relay" } else { "" },
                        if online { " (through the relay)" } else { " (set a relay first)" },
                    ),
                );
                widgets::section_gap(ui);
                ui.horizontal(|ui| {
                    widgets::eyebrow(ui, "LAN games");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if widgets::icon_button(ui, Icon::Reset, "Search the local network", false).clicked() {
                            st.last_refresh = -10.0;
                            if st.browser.is_none() {
                                st.browser = LanBrowser::new().ok();
                            }
                        }
                    });
                });
                if let Some(b) = st.browser.as_mut() {
                    if now - st.last_refresh > 2.0 {
                        st.last_refresh = now;
                        b.refresh();
                    }
                    let sessions = b.poll().to_vec();
                    if sessions.is_empty() {
                        ui.label(RichText::new("Searching...").weak());
                    }
                    for s in sessions {
                        ui.horizontal(|ui| {
                            ui.label(&s.name);
                            ui.label(RichText::new(format!("{}/{}", s.players, s.max_players)).font(theme::mono(11.0)));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if widgets::secondary_button(ui, None, "Join").clicked() {
                                    *addr = s.addr.to_string();
                                    act = Some(Act::Join(s.addr.to_string()));
                                }
                                ui.label(RichText::new(s.addr.to_string()).font(theme::mono(11.0)).weak());
                            });
                        });
                    }
                    ctx.request_repaint_after(std::time::Duration::from_millis(500));
                }
                widgets::section_gap(ui);
                servers_ui(ui, ctx, st, &cfg.relay, addr, &mut act);
            }
        });
    let (mut cfg, _) = zestengine_engine::net_sync::config_for(scene);
    cfg.player_name = st.player_name.clone();
    if !st.relay.trim().is_empty() {
        cfg.relay = st.relay.trim().to_string();
    }
    if !st.secret.is_empty() {
        cfg.secret = st.secret.clone();
    }
    cfg.account_name = st.account_name.trim().to_string();
    cfg.account_password = st.account_password.clone();
    match act? {
        Act::Disconnect => {
            if let Some(mut s) = net.take() {
                s.disconnect();
            }
            None
        }
        Act::EndSession => {
            if let Some(mut s) = net.take() {
                s.close_session();
            }
            None
        }
        Act::Kick(peer) => {
            if let Some(s) = net.as_mut() {
                s.kick(peer, "removed by the host");
            }
            None
        }
        Act::Ban(peer) => {
            if let Some(s) = net.as_mut() {
                s.ban(peer);
            }
            None
        }
        Act::Unban(fp) => {
            if let Some(s) = net.as_mut() {
                s.unban(&fp);
            }
            None
        }
        Act::Host(port) => match NetSession::host_with(port, cfg) {
            Ok(s) => {
                *net = Some(s);
                None
            }
            Err(e) => Some(format!("Host failed: {e}")),
        },
        Act::Join(a) => match NetSession::join_with(&a, cfg) {
            Ok(s) => {
                *net = Some(s);
                None
            }
            Err(e) => Some(format!("Join failed: {e}")),
        },
    }
}

/// The online server list: a directory (a relay run with `--directory`), filters, one row per
/// server with ping and a Join button. The logic is in `server_browser`.
fn servers_ui(ui: &mut egui::Ui, ctx: &egui::Context, st: &mut NetWindowState, scene_relay: &str, addr: &mut String, act: &mut Option<Act>) {
    let relay_hint = if !st.relay.trim().is_empty() { st.relay.trim().to_owned() } else { scene_relay.to_owned() };
    let sb = &mut st.servers;
    ui.horizontal(|ui| {
        widgets::eyebrow(ui, "Online servers");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if widgets::icon_button(ui, Icon::Reset, "Ask the directory for the server list", sb.busy()).clicked() && !sb.busy() {
                if sb.directory.trim().is_empty() {
                    sb.directory = relay_hint.clone();
                }
                sb.refresh();
            }
        });
    });
    widgets::property_row(ui, "Directory", |ui| {
        ui.add(egui::TextEdit::singleline(&mut sb.directory).hint_text(if relay_hint.is_empty() { "relay host:port".to_string() } else { relay_hint.clone() }))
            .on_hover_text("A relay started with --directory. Empty uses the Relay above.")
    });
    widgets::property_row(ui, "Filter", |ui| {
        ui.horizontal(|ui| {
            widgets::search_field(ui, &mut sb.filter.name_contains, "name", 110.0);
            ui.add(egui::TextEdit::singleline(&mut sb.filter.region).hint_text("region").desired_width(56.0));
            ui.add(egui::TextEdit::singleline(&mut sb.filter.game).hint_text("game").desired_width(70.0));
        })
        .response
    });
    ui.horizontal_wrapped(|ui| {
        ui.checkbox(&mut sb.filter.hide_full, "Hide full");
        ui.checkbox(&mut sb.filter.hide_empty, "Hide empty");
        ui.checkbox(&mut sb.filter.hide_password, "Hide passworded");
        ui.checkbox(&mut sb.filter.hide_incompatible, "Hide other versions");
    });
    if sb.poll() || sb.busy() {
        ctx.request_repaint_after(std::time::Duration::from_millis(100));
    }
    match sb.status().clone() {
        BrowserStatus::Idle => {
            ui.label(RichText::new("Press refresh to list servers.").weak());
        }
        BrowserStatus::Querying => {
            ui.label(RichText::new("Asking the directory...").weak());
        }
        BrowserStatus::Failed(e) => {
            widgets::warn_badge(ui, &e);
        }
        BrowserStatus::Pinging | BrowserStatus::Ready => {}
    }
    if sb.total() == 0 {
        if *sb.status() == BrowserStatus::Ready {
            ui.label(RichText::new("No servers match.").weak());
        }
        return;
    }
    ui.horizontal(|ui| {
        for (label, key) in [("Name", SortKey::Name), ("Players", SortKey::Players), ("Region", SortKey::Region), ("Ping", SortKey::Ping)] {
            let arrow = if sb.sort == key { if sb.ascending { " ▲" } else { " ▼" } } else { "" };
            if widgets::ghost_button(ui, &format!("{label}{arrow}"), 20.0).clicked() {
                sb.sort_by(key);
            }
        }
    });
    let mut join = None;
    for row in sb.rows() {
        ui.horizontal(|ui| {
            let name = if row.listing.info.password { format!("🔒 {}", row.name()) } else { row.name().to_owned() };
            ui.label(name).on_hover_text(format!("{} · {} · v{} · {}", row.listing.info.game, row.listing.info.map, row.listing.info.version, row.listing.addr));
            ui.label(RichText::new(row.players_text()).font(theme::mono(11.0)));
            if !row.listing.info.region.is_empty() {
                widgets::chip(ui, &row.listing.info.region);
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let button = ui.add_enabled_ui(row.joinable(), |ui| widgets::secondary_button(ui, None, "Join")).inner;
                let button = match row.why_not() {
                    Some(why) => button.on_disabled_hover_text(why),
                    None => button,
                };
                if button.clicked() {
                    join = sb.join_target(row);
                }
                let ping = match (row.ping_ms, row.unreachable) {
                    (Some(ms), _) => format!("{ms:.0} ms"),
                    (None, true) => "no answer".to_owned(),
                    (None, false) => "...".to_owned(),
                };
                ui.label(RichText::new(ping).font(theme::mono(11.0)).weak());
            });
        });
    }
    if let Some(target) = join {
        *addr = target.clone();
        *act = Some(Act::Join(target));
    }
}

fn connected_ui(ui: &mut egui::Ui, s: &NetSession, scene: &LoadedScene, act: &mut Option<Act>) {
    let role = if s.is_host() { "Hosting" } else { "Client" };
    ui.horizontal(|ui| {
        match s.state() {
            ConnState::Connected => {
                let epoch = if s.epoch() > 0 { format!(" · host #{}", s.epoch() + 1) } else { String::new() };
                widgets::chip(ui, &format!("{role} · peer {}{epoch}", s.local_peer()));
            }
            ConnState::Connecting => {
                widgets::badge(ui, "Connecting...");
            }
            ConnState::Disconnected(_) => {
                widgets::warn_badge(ui, "Disconnected");
            }
        }
        if let Some(a) = s.local_addr() {
            ui.label(RichText::new(a.to_string()).font(theme::mono(11.0)).weak());
        }
    });
    if let ConnState::Disconnected(reason) = s.state() {
        widgets::info_note(ui, reason);
    }
    if let Some(m) = s.migration_status() {
        ui.horizontal(|ui| {
            widgets::warn_badge(ui, "Host lost");
            let path = if m.relayed { "via the relay" } else { "directly" };
            let who = if m.target == s.local_peer() { "taking over".to_string() } else { format!("joining peer {} {path}", m.target) };
            ui.label(RichText::new(format!("{who} · {:.1} s", m.seconds)).font(theme::mono(11.0)));
        });
        widgets::info_note(ui, "The host left; the next player in line takes over and everyone reconnects to it. Objects the host simulated pause until then.");
    }
    let back = s.reconnecting();
    if s.is_host() && !back.is_empty() {
        let list = back.iter().map(|p| format!("#{p}")).collect::<Vec<_>>().join(", ");
        widgets::info_note(ui, &format!("Took over as host: waiting for {list} to reconnect."));
    }
    if let Some(r) = s.relay_status() {
        widgets::section_gap(ui);
        widgets::eyebrow(ui, "Online");
        widgets::property_row(ui, "Relay", |ui| ui.label(RichText::new(r.relay.to_string()).font(theme::mono(11.0))));
        match &r.phase {
            RelayPhase::Registering => {
                widgets::property_row(ui, "Room", |ui| widgets::badge(ui, "Registering..."));
            }
            RelayPhase::Failed(why) => {
                widgets::property_row(ui, "Room", |ui| widgets::warn_badge(ui, "Relay error"));
                widgets::info_note(ui, why);
            }
            RelayPhase::Registered => {
                widgets::property_row(ui, "Room", |ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(&r.code).font(theme::mono(15.0)).color(theme::accent()));
                        if widgets::icon_button(ui, Icon::Duplicate, "Copy the room code", false).clicked() {
                            ui.ctx().copy_text(r.code.clone());
                        }
                    })
                });
                if let Some(p) = r.public {
                    widgets::property_row(ui, "Public", |ui| ui.label(RichText::new(p.to_string()).font(theme::mono(11.0)).weak()));
                }
                if s.is_host() {
                    widgets::info_note(ui, "Players join with this room code (Join > Address), no port forwarding needed.");
                }
            }
        }
    }
    widgets::section_gap(ui);
    widgets::eyebrow(ui, &format!("Players ({}/{})", s.peer_count(), s.cfg.max_players));
    let next = s.next_host();
    for p in s.peer_infos() {
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("#{}", p.id)).font(theme::mono(11.0)).weak());
            ui.label(&p.name);
            if p.is_host {
                widgets::badge(ui, "host");
            }
            if p.id == s.local_peer() {
                widgets::badge(ui, "you");
            }
            if Some(p.id) == next {
                widgets::badge(ui, "next host").on_hover_text("Takes over if the host leaves (host migration)");
            }
            if !p.fingerprint.is_empty() {
                let fp = ui
                    .add(egui::Label::new(RichText::new(short_fingerprint(&p.fingerprint)).font(theme::mono(10.0)).weak()).sense(egui::Sense::click()))
                    .on_hover_text(format!("Verified player id {}\nClick to copy", p.fingerprint));
                if fp.clicked() {
                    ui.ctx().copy_text(p.fingerprint.clone());
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if s.is_host() && p.id != s.local_peer() && !s.reconnecting().contains(&p.id) {
                    if widgets::icon_button(ui, Icon::Lock, "Ban: kick and refuse this player id from now on", false).clicked() {
                        *act = Some(Act::Ban(p.id));
                    }
                    if widgets::icon_button(ui, Icon::Close, "Kick from the session", false).clicked() {
                        *act = Some(Act::Kick(p.id));
                    }
                }
                if p.id != s.local_peer() && p.rtt_ms > 0.0 {
                    if let Some(route) = s.peer_route(p.id) {
                        widgets::badge(ui, route);
                    }
                    if p.loss > 0.005 {
                        ui.label(RichText::new(format!("{:.1}% loss", p.loss * 100.0)).font(theme::mono(11.0)).weak());
                    }
                    ui.label(RichText::new(format!("{:.0} ms", p.rtt_ms)).font(theme::mono(11.0)));
                }
            });
        });
    }
    if s.is_host() && !s.cfg.ban.is_empty() {
        egui::CollapsingHeader::new(format!("Banned ({})", s.cfg.ban.len())).show(ui, |ui| {
            for fp in &s.cfg.ban {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(short_fingerprint(fp)).font(theme::mono(11.0))).on_hover_text(fp);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if widgets::icon_button(ui, Icon::LockOpen, "Unban", false).clicked() {
                            *act = Some(Act::Unban(fp.clone()));
                        }
                    });
                });
            }
            widgets::info_note(ui, "For this session only; add ids to the Network Manager's Ban List to keep them out of every session.");
        });
    }
    widgets::section_gap(ui);
    widgets::eyebrow(ui, "Traffic");
    let stats = s.stats;
    let mono = |ui: &mut egui::Ui, label: &str, v: String| {
        widgets::property_row(ui, label, |ui| ui.label(RichText::new(v).font(theme::mono(11.0))));
    };
    mono(ui, "In / Out", format!("{:.1} / {:.1} kB/s", stats.rate_in / 1024.0, stats.rate_out / 1024.0));
    mono(ui, "Packets", format!("{} in · {} out · {} resent", stats.packets_in, stats.packets_out, stats.resends));
    if s.is_host() && s.interest_config().enabled {
        mono(
            ui,
            "Interest",
            format!(
                "{} tracked · {:.0} relevant/player · {} held by budget · {} merged by tiers",
                stats.interest_objects, stats.interest_relevant_avg, stats.interest_dropped, stats.interest_throttled
            ),
        );
        for c in s.interest_stats().iter().take(8) {
            mono(ui, &format!("  peer {}", c.peer), format!("{} relevant · {:.1} kB/s · {} waiting", c.relevant, c.rate_out / 1024.0, c.pending));
        }
    }
    if stats.rejected > 0 {
        mono(ui, "Rejected", format!("{} (failed authentication)", stats.rejected));
    }
    if stats.half_open_refused > 0 {
        mono(ui, "Refused joins", format!("{} (too many handshakes at once)", stats.half_open_refused));
    }
    if s.migrations() > 0 {
        mono(ui, "Migrations", format!("{} · host is peer {}", s.migrations(), s.host_peer()));
    }
    if !s.is_host() {
        let q = s.quality();
        mono(ui, "Ping", format!("{:.0} ms · jitter {:.0} ms", q.rtt_ms, q.jitter_ms));
        mono(ui, "Loss", format!("{:.1}% in · {:.1}% out", q.loss_in * 100.0, q.loss_out * 100.0));
        mono(ui, "Interp delay", format!("{:.0} ms", s.view_delay * 1000.0));
    }
    let owned = s.world.owners.values().filter(|(o, _)| *o == s.local_peer() && *o != 0).count();
    mono(ui, "Objects", format!("{} owned · {} spawned", owned, s.world.spawns.len()));
    // Networked objects in this scene and who simulates them.
    let rows: Vec<(String, u64, bool)> = scene
        .objects
        .iter()
        .filter_map(|o| {
            let n = o.components.iter().find_map(|c| c.as_any().downcast_ref::<zestengine_core::NetworkObject>())?;
            n.state.connected.then(|| (o.name.clone(), n.state.owner, n.state.authority))
        })
        .collect();
    if !rows.is_empty() {
        egui::CollapsingHeader::new(format!("Networked objects ({})", rows.len())).show(ui, |ui| {
            egui::ScrollArea::vertical().max_height(160.0).show(ui, |ui| {
                for (name, owner, authority) in rows {
                    ui.horizontal(|ui| {
                        ui.label(if name.is_empty() { "(unnamed)".to_string() } else { name });
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let who = if owner == 0 { "host".to_string() } else { format!("peer {owner}") };
                            let text = RichText::new(who).font(theme::mono(11.0));
                            ui.label(if authority { text.color(theme::accent()) } else { text.weak() });
                        });
                    });
                }
            });
        });
    }
    widgets::section_gap(ui);
    ui.horizontal(|ui| {
        let hand_off = s.is_host() && s.cfg.migration && s.next_host().is_some();
        let (label, tip) = if hand_off {
            ("Leave", "Leave and hand the session to the next host (everyone else stays)")
        } else {
            ("Disconnect", "Leave the session")
        };
        if widgets::secondary_button(ui, Some(Icon::Stop), label).on_hover_text(tip).clicked() {
            *act = Some(Act::Disconnect);
        }
        if s.is_host() && s.peer_count() > 1 && widgets::secondary_button(ui, Some(Icon::Close), "End session").on_hover_text("Close the session for everyone (no host migration)").clicked() {
            *act = Some(Act::EndSession);
        }
    });
}
