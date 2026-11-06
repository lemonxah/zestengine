//! Inspectors and gizmos for the networking components (Network Object,
//! Network Variables, Network Animator, Network Avatar, Network Manager).
//! A child module of `components` so it shares the themed row helpers.

use zestengine_core::{
    InterpolationMode, NetAutoStart, NetValue, NetVar, NetworkAnimator, NetworkAvatar, NetworkManager, NetworkObject,
    NetworkVariables, ObjectRef, OwnershipMode, HARD_MAX_REWIND, RpcPermission, RpcRule, Topology,
};
use egui::{DragValue, RichText, Ui};

use super::{Gizmo, GizmoEdit, GizmoSpec, Inspect, InspectCtx, enum_row, property_row};
use crate::{icons::Icon, theme, widgets};

impl Inspect for NetworkObject {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        if self.state.connected {
            ui.horizontal(|ui| {
                let owner = match self.state.owner {
                    0 => "Unowned (host)".to_string(),
                    p => format!("Owner: peer {p}"),
                };
                ui.label(RichText::new(owner).font(theme::mono(11.0)));
                if self.state.authority {
                    widgets::chip(ui, "You simulate this");
                } else {
                    widgets::badge(ui, "remote");
                }
            });
            ui.add_space(4.0);
        }
        enum_row(ui, "Ownership", "zestengine-net-own", &mut self.ownership, &OwnershipMode::ALL, OwnershipMode::label, &mut changed);
        if self.ownership == OwnershipMode::Grabbable {
            widgets::info_note(ui, "Grabbable: a player can take it when it is free and their avatar is within reach. Use Host only for anything the server simulates.");
        }
        widgets::eyebrow(ui, "Replicate");
        property_row(ui, "Transform", &mut changed, |ui| ui.checkbox(&mut self.sync_transform, ""));
        property_row(ui, "Scale", &mut changed, |ui| ui.checkbox(&mut self.sync_scale, ""));
        property_row(ui, "Rigid Body", &mut changed, |ui| {
            ui.checkbox(&mut self.sync_rigidbody, "")
                .on_hover_text("The owner simulates; other peers make the body kinematic and follow")
        });
        property_row(ui, "Parent", &mut changed, |ui| ui.checkbox(&mut self.sync_parent, ""));
        property_row(ui, "Active", &mut changed, |ui| ui.checkbox(&mut self.sync_active, ""));
        property_row(ui, "Predicted", &mut changed, |ui| {
            ui.checkbox(&mut self.predicted, "").on_hover_text(
                "For a Pawn, First / Third Person Character or VR Character: the host simulates it from its owner's input (server authority); the owner moves at once and reconciles with the host's state",
            )
        });
        widgets::eyebrow(ui, "Smoothing");
        enum_row(
            ui,
            "Mode",
            "zestengine-net-interp",
            &mut self.interpolation,
            &InterpolationMode::ALL,
            InterpolationMode::label,
            &mut changed,
        );
        property_row(ui, "Delay", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.interp_delay).speed(0.005).range(0.0..=1.0).suffix(" s"))
                .on_hover_text("How far behind the newest snapshot remote peers render (absorbs jitter); the minimum when Adaptive")
        });
        property_row(ui, "Adaptive", &mut changed, |ui| {
            ui.checkbox(&mut self.adaptive_delay, "")
                .on_hover_text("Raise the delay when the measured jitter and packet loss need it, and lower it back when the link calms down")
        });
        widgets::eyebrow(ui, "Bandwidth");
        property_row(ui, "Send Rate", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.send_rate).speed(0.5).range(1.0..=120.0).suffix(" Hz"))
        });
        property_row(ui, "Move Threshold", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.position_threshold).speed(0.0005).range(0.0..=1.0).suffix(" m"))
        });
        property_row(ui, "Turn Threshold", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.rotation_threshold).speed(0.05).range(0.0..=45.0).suffix(" deg"))
        });
        property_row(ui, "Quantize", &mut changed, |ui| {
            ui.checkbox(&mut self.quantize, "").on_hover_text("1 mm positions, 32-bit rotations")
        });
        property_row(ui, "Relevancy", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.relevancy).speed(0.5).range(0.0..=100_000.0).suffix(" m"))
                .on_hover_text("Only send to players within this distance (0 = everyone)")
        });
        property_row(ui, "Always Relevant", &mut changed, |ui| {
            ui.checkbox(&mut self.always_relevant, "")
                .on_hover_text("Interest management (Network Manager): tell every player about this object whatever the distance")
        });
        property_row(ui, "Despawn On Leave", &mut changed, |ui| {
            ui.checkbox(&mut self.despawn_on_owner_leave, "")
                .on_hover_text("Remove the object when its owner disconnects (otherwise it becomes unowned)")
        });
        changed
    }
}

impl Gizmo for NetworkObject {
    fn gizmos(&self) -> Vec<GizmoSpec> {
        if self.relevancy <= 0.0 {
            return Vec::new();
        }
        vec![GizmoSpec::Range { center: glam::Vec3::ZERO, radius: self.relevancy, color: egui::Color32::from_rgb(90, 200, 250) }]
    }
    fn apply_gizmo_edit(&mut self, edit: GizmoEdit) {
        if let GizmoEdit::Range { radius, .. } = edit {
            self.relevancy = radius.max(0.0);
        }
    }
}

const VALUE_KINDS: [&str; 5] = ["Bool", "Int", "Float", "Vec3", "String"];

fn convert(v: &NetValue, kind: &str) -> NetValue {
    match kind {
        "Bool" => NetValue::Bool(v.as_bool().unwrap_or(false)),
        "Int" => NetValue::Int(v.as_int().unwrap_or(0)),
        "Float" => NetValue::Float(v.as_float().unwrap_or(0.0)),
        "Vec3" => NetValue::Vec3(v.as_vec3().unwrap_or_default().to_array()),
        _ => NetValue::Str(v.as_str().unwrap_or_default().to_owned()),
    }
}

fn value_editor(ui: &mut Ui, v: &mut NetValue) -> bool {
    match v {
        NetValue::Bool(b) => ui.checkbox(b, "").changed(),
        NetValue::Int(i) => ui.add(DragValue::new(i)).changed(),
        NetValue::Float(f) => ui.add(DragValue::new(f).speed(0.01)).changed(),
        NetValue::Vec3(a) => widgets::vec3_field(ui, a, 2, 0.02),
        NetValue::Str(s) => ui.text_edit_singleline(s).changed(),
        NetValue::Quat(_) | NetValue::Bytes(_) => {
            ui.label(RichText::new("(set at runtime)").weak());
            false
        }
    }
}

impl Inspect for NetworkVariables {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        let mut remove = None;
        for (k, var) in self.vars.iter_mut().enumerate() {
            ui.push_id(("zestengine-netvar", k), |ui| {
                ui.horizontal(|ui| {
                    changed |= ui
                        .add(egui::TextEdit::singleline(&mut var.name).hint_text("name").desired_width(96.0))
                        .changed();
                    let mut kind = var.value.kind_label();
                    egui::ComboBox::from_id_salt("kind").width(64.0).selected_text(kind).show_ui(ui, |ui| {
                        for k in VALUE_KINDS {
                            ui.selectable_value(&mut kind, k, k);
                        }
                    });
                    if kind != var.value.kind_label() {
                        var.value = convert(&var.value, kind);
                        changed = true;
                    }
                    if widgets::icon_button(ui, Icon::Trash, "Remove variable", false).clicked() {
                        remove = Some(k);
                    }
                });
                widgets::property_row(ui, "Default", |ui| changed |= value_editor(ui, &mut var.value));
            });
        }
        if let Some(k) = remove {
            self.vars.remove(k);
            changed = true;
        }
        if widgets::dashed_button(ui, Some(Icon::Plus), "Add variable").clicked() {
            let name = format!("var{}", self.vars.len());
            self.vars.push(NetVar { name, value: NetValue::Int(0) });
            changed = true;
        }
        property_row(ui, "Anyone Can Write", &mut changed, |ui| {
            ui.checkbox(&mut self.anyone_can_write, "")
                .on_hover_text("Let any peer write these, not only the object's authority")
        });
        widgets::info_note(ui, "Read with ctx.net_var, write with ctx.set_net_var. Late joiners get current values.");
        changed
    }
}
impl Gizmo for NetworkVariables {}

impl Inspect for NetworkAnimator {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        let mut list = self.parameters.join(", ");
        property_row(ui, "Parameters", &mut changed, |ui| {
            ui.add(egui::TextEdit::singleline(&mut list).hint_text("all").desired_width(f32::INFINITY))
                .on_hover_text("Comma-separated Animator parameters to sync; empty syncs every parameter")
        });
        if changed {
            self.parameters = list.split(',').map(|s| s.trim().to_owned()).filter(|s| !s.is_empty()).collect();
        }
        property_row(ui, "Send Rate", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.send_rate).speed(0.5).range(1.0..=60.0).suffix(" Hz"))
        });
        widgets::info_note(ui, "Needs an Animator on this object. Triggers sync as values; use a bool or an RPC for must-arrive one-shots.");
        changed
    }
}
impl Gizmo for NetworkAnimator {}

impl Inspect for NetworkAvatar {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        changed |= ctx.object_ref(ui, "Head", &mut self.head);
        changed |= ctx.object_ref(ui, "Left Hand", &mut self.left_hand);
        changed |= ctx.object_ref(ui, "Right Hand", &mut self.right_hand);
        changed |= ctx.object_ref(ui, "Hips", &mut self.hips);
        changed |= ctx.object_ref(ui, "Left Foot", &mut self.left_foot);
        changed |= ctx.object_ref(ui, "Right Foot", &mut self.right_foot);
        property_row(ui, "Send Rate", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.send_rate).speed(0.5).range(1.0..=90.0).suffix(" Hz"))
        });
        property_row(ui, "Drive From VR", &mut changed, |ui| {
            ui.checkbox(&mut self.drive_from_vr, "")
                .on_hover_text("The owner's head and hands follow the local headset and controllers")
        });
        property_row(ui, "Follow Local Player", &mut changed, |ui| {
            ui.checkbox(&mut self.follow_local_player, "")
                .on_hover_text("Player prefab mode: the owner's copy mirrors the scene's VR Character rig (head and hands, VR or desktop)")
        });
        if self.follow_local_player {
            ui.label("Hidden For Owner");
            let mut remove = None;
            for (k, r) in self.owner_hidden.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    changed |= ctx.object_ref(ui, &format!("Object {}", k + 1), r);
                    if ui.small_button("x").on_hover_text("Remove").clicked() {
                        remove = Some(k);
                    }
                });
            }
            if let Some(k) = remove {
                self.owner_hidden.remove(k);
                changed = true;
            }
            if ui.small_button("+ Add").clicked() {
                self.owner_hidden.push(ObjectRef::NONE);
                changed = true;
            }
        }
        widgets::info_note(ui, "Put this on the player rig root; parts are its children. Remote peers get the poses written back and as IK targets. For a Network Manager player prefab, turn on Follow Local Player and list the avatar's own visuals under Hidden For Owner.");
        changed
    }
}
impl Gizmo for NetworkAvatar {}

impl Inspect for NetworkManager {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        property_row(ui, "Session Name", &mut changed, |ui| ui.text_edit_singleline(&mut self.session_name));
        property_row(ui, "Port", &mut changed, |ui| ui.add(DragValue::new(&mut self.port).range(1..=65535)));
        property_row(ui, "Max Players", &mut changed, |ui| ui.add(DragValue::new(&mut self.max_players).range(1..=256)));
        property_row(ui, "Timeout", &mut changed, |ui| {
            ui.add(DragValue::new(&mut self.timeout).speed(0.1).range(1.0..=120.0).suffix(" s"))
        });
        property_row(ui, "LAN Discovery", &mut changed, |ui| ui.checkbox(&mut self.discovery, ""));
        widgets::eyebrow(ui, "Online");
        property_row(ui, "Relay", &mut changed, |ui| {
            ui.add(egui::TextEdit::singleline(&mut self.relay).hint_text("none (LAN / direct only)"))
                .on_hover_text("Relay server (host:port, default port 47800): hosting also opens a room there, and players join by room code without port forwarding")
        });
        property_row(ui, "Room Code", &mut changed, |ui| {
            ui.add(egui::TextEdit::singleline(&mut self.room_code).hint_text("assigned by the relay"))
                .on_hover_text("A fixed room code for hosts (empty = the relay picks one)")
        });
        property_row(ui, "Secret", &mut changed, |ui| {
            ui.add(egui::TextEdit::singleline(&mut self.secret).password(true).hint_text("none (open session)"))
                .on_hover_text("Peers must know it to join; it also keys the session encryption. Use a long random value.")
        });
        widgets::eyebrow(ui, "Security");
        property_row(ui, "Host Migration", &mut changed, |ui| {
            ui.checkbox(&mut self.host_migration, "")
                .on_hover_text("When the host leaves, the next player in line takes over and the session goes on")
        });
        property_row(ui, "Pin Host Identity", &mut changed, |ui| {
            ui.checkbox(&mut self.pin_host, "")
                .on_hover_text("Clients remember the host's player id for this address / room code and refuse a different one later")
        });
        changed |= fingerprint_list(ui, "Allow List", &mut self.allow_list, "anyone with the secret");
        changed |= fingerprint_list(ui, "Ban List", &mut self.ban_list, "nobody");
        changed |= account_list(ui, "Accounts", &mut self.account_list);
        widgets::info_note(ui, "Lists hold player ids (fingerprints like 1a2b-3c4d-...), one per line: copy them from the Network window.");
        widgets::eyebrow(ui, "Players");
        changed |= ctx.file_field(ui, "Player Prefab", &mut self.player_prefab, "prefabs/player.prefab", &["prefab"]);
        property_row(ui, "Spawn Tag", &mut changed, |ui| ui.text_edit_singleline(&mut self.spawn_tag));
        if self.dedicated {
            // An old file: the checkbox became the topology.
            self.dedicated = false;
            self.topology = Topology::Dedicated;
            changed = true;
        }
        enum_row(ui, "Topology", "zestengine-net-topology", &mut self.topology, &Topology::ALL, Topology::label, &mut changed);
        if self.topology == Topology::Dedicated {
            widgets::info_note(
                ui,
                "Dedicated: the server gets no player, there is no host migration, clients pin the server's key and the strict validation rules below apply to every client.",
            );
        }
        property_row(ui, "Clients Can Spawn", &mut changed, |ui| ui.checkbox(&mut self.clients_can_spawn, ""));
        validation_ui(ui, self, &mut changed);
        interest_ui(ui, self, &mut changed);
        anticheat_ui(ui, self, &mut changed);
        widgets::eyebrow(ui, "Built game");
        enum_row(ui, "Auto Start", "zestengine-net-auto", &mut self.auto_start, &NetAutoStart::ALL, NetAutoStart::label, &mut changed);
        if self.auto_start == NetAutoStart::Join {
            property_row(ui, "Address", &mut changed, |ui| {
                ui.add(egui::TextEdit::singleline(&mut self.address).hint_text("host:port or room code"))
            });
        }
        widgets::info_note(ui, "The editor's Network window (Tools) hosts and joins with these settings.");
        changed
    }
}
impl Gizmo for NetworkManager {}

/// The server-side validation settings (NET-4): the ones a project usually
/// changes. Everything else is in `server.ron` (`validation: (..)`).
fn validation_ui(ui: &mut Ui, m: &mut NetworkManager, changed: &mut bool) {
    widgets::eyebrow(ui, "Validation");
    let v = &mut m.validation;
    property_row(ui, "Enabled", changed, |ui| {
        ui.checkbox(&mut v.enabled, "").on_hover_text("Rate limits, movement budgets, VR pose limits, ownership rules and strikes. Size and NaN checks always apply.")
    });
    if !v.enabled {
        return;
    }
    property_row(ui, "Strict Listen", changed, |ui| {
        ui.checkbox(&mut v.strict_on_listen_host, "").on_hover_text("Apply the dedicated-server rules (movement, ownership range, spawn allow-list) to a player-hosted session too")
    });
    property_row(ui, "Strict RPCs", changed, |ui| {
        ui.checkbox(&mut v.strict_rpc_permissions, "").on_hover_text("On a dedicated server, an RPC without a rule below can only be sent by the owner of its object")
    });
    property_row(ui, "Max Rewind", changed, |ui| {
        ui.add(DragValue::new(&mut v.max_rewind).speed(0.005).range(0.0..=HARD_MAX_REWIND).suffix(" s"))
            .on_hover_text("Lag compensation window; timestamps older or from the future are refused. Never more than 0.9 s.")
    });
    property_row(ui, "Grab Range", changed, |ui| {
        ui.add(DragValue::new(&mut v.grab_range).speed(0.05).range(0.5..=50.0).suffix(" m"))
            .on_hover_text("How close a player's avatar must be to take ownership of a Takeover / Grabbable object")
    });
    property_row(ui, "Prop Speed Cap", changed, |ui| {
        ui.add(DragValue::new(&mut v.prop_speed).speed(0.5).range(1.0..=500.0).suffix(" m/s"))
            .on_hover_text("Fastest an object a client owns may move before it counts as a speed hack")
    });
    property_row(ui, "Teleport Tolerance", changed, |ui| {
        ui.add(DragValue::new(&mut v.teleport_tolerance).speed(0.05).range(0.0..=50.0).suffix(" m"))
            .on_hover_text("Metres an owned object may be ahead of its speed budget at once (network jitter); more is a teleport")
    });
    property_row(ui, "Arm Reach", changed, |ui| {
        ui.add(DragValue::new(&mut v.arm_reach).speed(0.01).range(0.3..=3.0).suffix(" m"))
            .on_hover_text("How far a hand may be from the head (scaled by the avatar's size)")
    });
    property_row(ui, "Hand Speed Cap", changed, |ui| {
        ui.add(DragValue::new(&mut v.hand_speed).speed(0.5).range(1.0..=200.0).suffix(" m/s"))
    });
    property_row(ui, "Kick At", changed, |ui| {
        ui.add(DragValue::new(&mut v.strikes.kick_at).speed(1.0).range(5.0..=10_000.0).suffix(" strikes"))
            .on_hover_text("Strike points (they decay by 1 per second) at which a client is kicked; a second kick is a ban by player id")
    });
    *changed |= string_lines(ui, "Client Spawns", &mut v.client_spawn_prefabs, "none (dedicated)", "Prefab paths clients may spawn, one per line; a trailing * is a prefix; clone:* allows cloning scene templates");
    let mut rules: Vec<String> = v.rpc_rules.iter().map(|r| format!("{}:{}", r.name, rpc_permission_key(r.permission))).collect();
    if string_lines(ui, "RPC Rules", &mut rules, "name:anyone|owner|server", "One per line. Unlisted RPCs follow Strict RPC Permissions.") {
        v.rpc_rules = rules
            .iter()
            .filter_map(|l| {
                let (name, perm) = l.rsplit_once(':')?;
                let permission = match perm.trim().to_ascii_lowercase().as_str() {
                    "owner" => RpcPermission::Owner,
                    "server" => RpcPermission::Server,
                    _ => RpcPermission::Anyone,
                };
                Some(RpcRule { name: name.trim().to_owned(), permission, ..Default::default() })
            })
            .collect();
        *changed = true;
    }
    widgets::info_note(ui, "Rate limits, VR pose limits and the rest of the strike policy are in server.ron: validation: (..).");
}

/// Anti-cheat provider (SEC-2). Provider payloads and verdicts are the engine's; the
/// product itself (e.g. Easy Anti-Cheat) is a separate cargo feature.
fn anticheat_ui(ui: &mut Ui, m: &mut NetworkManager, changed: &mut bool) {
    widgets::eyebrow(ui, "Anti-cheat");
    let a = &mut m.anticheat;
    property_row(ui, "Provider", changed, |ui| {
        ui.text_edit_singleline(&mut a.provider).on_hover_text("none, eos (Easy Anti-Cheat; needs the eos-anticheat feature, not built yet) or a provider your game installs")
    });
    if !a.enabled() {
        return;
    }
    property_row(ui, "Require", changed, |ui| {
        ui.checkbox(&mut a.require, "").on_hover_text("Server: hold a client out of gameplay until the provider trusts it; remove it when rejected. Off = advisory only.")
    });
    if a.require {
        property_row(ui, "Pending Timeout", changed, |ui| {
            ui.add(DragValue::new(&mut a.pending_timeout).speed(0.5).range(1.0..=600.0).suffix(" s"))
                .on_hover_text("Remove a client the provider has not decided on after this long")
        });
    }
    widgets::info_note(ui, "Provider options and per-server overrides are in server.ron: anticheat: (provider: .., require: .., options: {..}).");
}

/// Interest management (NET-5): who is told about what, and how often.
fn interest_ui(ui: &mut Ui, m: &mut NetworkManager, changed: &mut bool) {
    widgets::eyebrow(ui, "Interest management");
    let i = &mut m.interest;
    property_row(ui, "Enabled", changed, |ui| {
        ui.checkbox(&mut i.enabled, "").on_hover_text("Host only: tell each player about the objects near it, at a rate that falls with distance, within a byte budget. Off = everyone gets everything.")
    });
    if !i.enabled {
        return;
    }
    property_row(ui, "Radius", changed, |ui| {
        ui.add(DragValue::new(&mut i.radius).speed(1.0).range(1.0..=100_000.0).suffix(" m"))
            .on_hover_text("An object becomes relevant to a player inside this distance")
    });
    property_row(ui, "Hysteresis", changed, |ui| {
        ui.add(DragValue::new(&mut i.hysteresis).speed(0.5).range(0.0..=10_000.0).suffix(" m"))
            .on_hover_text("... and stays relevant until radius + hysteresis, so objects on the border do not flicker")
    });
    property_row(ui, "Cell Size", changed, |ui| {
        ui.add(DragValue::new(&mut i.cell_size).speed(1.0).range(1.0..=10_000.0).suffix(" m"))
            .on_hover_text("Spatial hash cell; about a third of the radius")
    });
    property_row(ui, "Refresh", changed, |ui| ui.add(DragValue::new(&mut i.refresh_hz).speed(0.1).range(0.5..=120.0).suffix(" Hz")));
    property_row(ui, "Nearest Full Rate", changed, |ui| {
        ui.add(DragValue::new(&mut i.nearest_full).speed(0.2).range(0..=256).suffix(" players"))
            .on_hover_text("This many of the nearest players always update at full rate")
    });
    property_row(ui, "Budget", changed, |ui| {
        ui.add(DragValue::new(&mut i.budget).speed(500.0).range(0..=100_000_000).suffix(" B/s"))
            .on_hover_text("Outgoing bytes per second and player (0 = unlimited). The busiest, longest waiting updates go first.")
    });
    property_row(ui, "Voice Range", changed, |ui| {
        ui.add(DragValue::new(&mut i.voice_range).speed(0.5).range(0.0..=100_000.0).suffix(" m"))
            .on_hover_text("Voice reaches listeners within this distance of the speaker")
    });
    property_row(ui, "Face / Param Range", changed, |ui| {
        ui.add(DragValue::new(&mut i.avatar_detail_range).speed(0.5).range(0.0..=100_000.0).suffix(" m"))
            .on_hover_text("Face tracking only reaches players this close (avatar parameters follow the radius)")
    });
    property_row(ui, "Short Ids", changed, |ui| {
        ui.checkbox(&mut i.short_ids, "").on_hover_text("32-bit network ids in snapshots instead of 128-bit")
    });
    // Update-rate tiers: "distance:rate" per line.
    let mut lines: Vec<String> = i.tiers.iter().map(|t| format!("{}:{}", t.distance, t.rate)).collect();
    if string_lines(ui, "Rate Tiers", &mut lines, "0:0", "distance:updates-per-second, one per line (rate 0 = every update)") {
        i.tiers = lines
            .iter()
            .filter_map(|l| {
                let (d, r) = l.split_once(':')?;
                Some(zestengine_core::RateTier { distance: d.trim().parse().ok()?, rate: r.trim().parse().ok()? })
            })
            .collect();
        *changed = true;
    }
    widgets::info_note(ui, "Needs clients of this build. A client without a player object only receives Always Relevant objects. The rest is in server.ron: interest: (..).");
}

fn rpc_permission_key(p: RpcPermission) -> &'static str {
    match p {
        RpcPermission::Anyone => "anyone",
        RpcPermission::Owner => "owner",
        RpcPermission::Server => "server",
    }
}

/// One string per line.
fn string_lines(ui: &mut Ui, label: &str, list: &mut Vec<String>, empty: &str, hover: &str) -> bool {
    let mut changed = false;
    let mut text = list.join("\n");
    property_row(ui, label, &mut changed, |ui| {
        ui.add(egui::TextEdit::multiline(&mut text).hint_text(empty).desired_rows(2).desired_width(f32::INFINITY).font(egui::TextStyle::Monospace))
            .on_hover_text(hover)
    });
    if changed {
        *list = text.lines().map(|l| l.trim().to_owned()).filter(|l| !l.is_empty()).collect();
    }
    changed
}

/// One fingerprint per line.
fn fingerprint_list(ui: &mut Ui, label: &str, list: &mut Vec<String>, empty: &str) -> bool {
    let mut changed = false;
    let mut text = list.join("\n");
    property_row(ui, label, &mut changed, |ui| {
        ui.add(egui::TextEdit::multiline(&mut text).hint_text(empty).desired_rows(2).desired_width(f32::INFINITY).font(egui::TextStyle::Monospace))
    });
    if changed {
        *list = text.lines().map(|l| l.trim().to_owned()).filter(|l| !l.is_empty()).collect();
    }
    changed
}

/// One account per line: `name:password`.
fn account_list(ui: &mut Ui, label: &str, list: &mut Vec<String>) -> bool {
    let mut changed = false;
    let mut text = list.join("\n");
    property_row(ui, label, &mut changed, |ui| {
        ui.add(
            egui::TextEdit::multiline(&mut text)
                .hint_text("name:password")
                .desired_rows(2)
                .desired_width(f32::INFINITY)
                .font(egui::TextStyle::Monospace),
        )
        .on_hover_text("When non-empty, joining requires one of these credentials")
    });
    if changed {
        *list = text
            .lines()
            .map(|l| l.trim().to_owned())
            .filter(|l| !l.is_empty())
            .collect();
    }
    changed
}
