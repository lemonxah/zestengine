//! The admin command language: typed on the stdin console, or sent as signed
//! UDP packets by `zestengine-admin` (see [`crate::remote`]); both run the same
//! commands through `Server::exec`.

use serde::{Deserialize, Serialize};

/// What an admin key may do. Ordered: each role includes the ones before it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AdminRole {
    /// Look only: status, players, stats, bans.
    #[default]
    Observer,
    /// Also keep order: kick, ban, unban, say.
    Moderator,
    /// Everything: scene changes, save, config reload, restart, shutdown.
    Admin,
}

/// One admin key listed in `server.ron`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AdminKey {
    /// Who it belongs to (shown in the log).
    pub name: String,
    /// The Ed25519 public key, 64 hex digits (`zestengine-admin keygen`).
    pub key: String,
    pub role: AdminRole,
}

impl Default for AdminKey {
    fn default() -> Self {
        Self { name: String::new(), key: String::new(), role: AdminRole::Observer }
    }
}

/// Seconds for `30s`, `15m`, `2h`, `7d`, `2w`; `perm` / `permanent` / `forever` = 0.
pub fn parse_duration(s: &str) -> Option<u64> {
    if matches!(s.to_ascii_lowercase().as_str(), "perm" | "permanent" | "forever") {
        return Some(0);
    }
    let unit = s.chars().last()?;
    let n: u64 = s[..s.len() - unit.len_utf8()].parse().ok()?;
    let mult = match unit.to_ascii_lowercase() {
        's' => 1,
        'm' => 60,
        'h' => 3600,
        'd' => 86_400,
        'w' => 604_800,
        _ => return None,
    };
    n.checked_mul(mult).filter(|v| *v > 0)
}

/// One console command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdminCommand {
    Status,
    Players,
    /// Interest management numbers per player.
    Interest,
    /// Tick, traffic, violation and persistence counters.
    Stats,
    Kick { id: u64, reason: String },
    /// Ban a player id: `target` is a connected player's id or a fingerprint; `secs` 0 = permanent.
    Ban { target: String, secs: u64, reason: String },
    Unban(String),
    Bans,
    /// Player reports waiting for a moderator (newest last).
    Reports,
    /// Empty the report queue.
    ReportsClear,
    /// The admin keys now in force.
    Keys,
    /// Let a new key sign commands, without a restart or a config edit.
    KeyAdd { name: String, key: String, role: AdminRole },
    /// Retire a key by name or by 8+ digits of it.
    KeyRemove(String),
    Say(String),
    Scene(String),
    /// Write the persistent state now.
    Save,
    /// Re-read `server.ron` and apply what can change while running.
    Reload,
    /// Stop with a "restarting" notice and exit code 75 so a service manager starts it again.
    /// A non-empty `redirect` (`host:port` or a room code) is offered to the players.
    Restart { redirect: String },
    /// Like `quit`, offering players somewhere else to go.
    Shutdown { redirect: String },
    Quit,
    Help,
}

impl AdminCommand {
    /// The weakest role that may run this command.
    pub fn required_role(&self) -> AdminRole {
        match self {
            Self::Status | Self::Players | Self::Interest | Self::Stats | Self::Bans | Self::Help => AdminRole::Observer,
            Self::Kick { .. } | Self::Ban { .. } | Self::Unban(_) | Self::Say(_) | Self::Reports | Self::ReportsClear => AdminRole::Moderator,
            Self::Scene(_) | Self::Save | Self::Reload | Self::Restart { .. } | Self::Shutdown { .. } | Self::Quit | Self::Keys | Self::KeyAdd { .. } | Self::KeyRemove(_) => AdminRole::Admin,
        }
    }
}

pub const HELP: &str = "\
commands:
  status               uptime, scene, tick rate, players, traffic
  players              connected players (id, name, ping)
  interest             per player: relevant objects, bytes/s, updates held by the budget
  stats                ticks, traffic, violations, saves, remote admin counters
  kick <id> [reason]   remove a player (shown the reason)
  ban <id|fingerprint> [30m|2h|7d|perm] [reason]
                       refuse a player id (and kick it if connected); default permanent
  unban <fingerprint>  lift a ban (a unique prefix of 8+ digits is enough)
  bans                 the ban list
  reports              player reports waiting for a moderator (`reports clear` empties the queue)
  keys                 the admin keys in force (admin)
  key add <name> <64-hex-key> <observer|moderator|admin>
                       let a new key sign commands now; kept across restarts (admin)
  key remove <name|8+ digits of the key>
                       retire a key (the last Admin key cannot be removed) (admin)
  say <text>           message every player
  scene <path>         switch everyone to a project scene
  save                 write the persistent state now
  reload               re-read server.ron (motd, stats interval, admin keys, persistence ...)
  restart [addr]       stop with exit code 75 (a supervisor restarts it); players may be sent to addr
  shutdown [addr]      stop gracefully; players may be sent to addr (host:port or room code)
  quit                 shut down gracefully (also: stop, exit)
  help                 this list";

fn one_arg(rest: &str) -> Result<String, String> {
    if rest.contains(char::is_whitespace) {
        Err("takes at most one argument (a host:port or room code)".into())
    } else {
        Ok(rest.to_string())
    }
}

impl AdminCommand {
    /// Parse one line. `Ok(None)` for a blank line.
    pub fn parse(line: &str) -> Result<Option<Self>, String> {
        let line = line.trim();
        if line.is_empty() {
            return Ok(None);
        }
        let (cmd, rest) = match line.split_once(char::is_whitespace) {
            Some((c, r)) => (c, r.trim()),
            None => (line, ""),
        };
        let no_args = |c: Self| if rest.is_empty() { Ok(Some(c)) } else { Err(format!("{cmd} takes no arguments")) };
        match cmd.to_ascii_lowercase().as_str() {
            "status" => no_args(Self::Status),
            "stats" => no_args(Self::Stats),
            "bans" => no_args(Self::Bans),
            "save" => no_args(Self::Save),
            "reload" => no_args(Self::Reload),
            "restart" => Ok(Some(Self::Restart { redirect: one_arg(rest)? })),
            "shutdown" => Ok(Some(Self::Shutdown { redirect: one_arg(rest)? })),
            "unban" => {
                if rest.is_empty() || rest.contains(char::is_whitespace) {
                    Err("usage: unban <fingerprint>".into())
                } else {
                    Ok(Some(Self::Unban(rest.to_string())))
                }
            }
            "ban" => {
                let mut it = rest.split_whitespace();
                let target = it.next().ok_or("usage: ban <id|fingerprint> [30m|2h|7d|perm] [reason]")?.to_string();
                let tail: Vec<&str> = it.collect();
                let (secs, reason) = match tail.first().and_then(|d| parse_duration(d)) {
                    Some(s) => (s, tail[1..].join(" ")),
                    None => (0, tail.join(" ")),
                };
                Ok(Some(Self::Ban { target, secs, reason }))
            }
            "players" | "list" => no_args(Self::Players),
            "keys" => no_args(Self::Keys),
            "reports" => match rest {
                "" => Ok(Some(Self::Reports)),
                "clear" => Ok(Some(Self::ReportsClear)),
                _ => Err("usage: reports [clear]".into()),
            },
            "key" => {
                let mut it = rest.split_whitespace();
                match it.next().map(str::to_ascii_lowercase).as_deref() {
                    Some("add") => {
                        let (Some(name), Some(key), Some(role), None) = (it.next(), it.next(), it.next(), it.next()) else {
                            return Err("usage: key add <name> <64-hex-key> <observer|moderator|admin>".into());
                        };
                        let role = crate::admin_keys::parse_role(role).ok_or("the role is observer, moderator or admin")?;
                        Ok(Some(Self::KeyAdd { name: name.to_owned(), key: key.to_owned(), role }))
                    }
                    Some("remove" | "rm") => match (it.next(), it.next()) {
                        (Some(who), None) => Ok(Some(Self::KeyRemove(who.to_owned()))),
                        _ => Err("usage: key remove <name|8+ digits of the key>".into()),
                    },
                    _ => Err("usage: key add <name> <hex> <role> | key remove <name>".into()),
                }
            }
            "interest" => no_args(Self::Interest),
            "quit" | "stop" | "exit" => no_args(Self::Quit),
            "help" | "?" => no_args(Self::Help),
            "kick" => {
                let (id, reason) = match rest.split_once(char::is_whitespace) {
                    Some((i, r)) => (i, r.trim()),
                    None => (rest, ""),
                };
                if id.is_empty() {
                    return Err("usage: kick <id> [reason]".into());
                }
                let id = id.parse().map_err(|_| format!("kick: {id:?} is not a player id (see `players`)"))?;
                Ok(Some(Self::Kick { id, reason: reason.to_string() }))
            }
            "say" => {
                if rest.is_empty() {
                    Err("usage: say <text>".into())
                } else {
                    Ok(Some(Self::Say(rest.to_string())))
                }
            }
            "scene" => {
                if rest.is_empty() {
                    Err("usage: scene <project-relative path>".into())
                } else {
                    Ok(Some(Self::Scene(rest.to_string())))
                }
            }
            other => Err(format!("unknown command {other:?} (try `help`)")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commands_parse() {
        let p = |s| AdminCommand::parse(s);
        assert_eq!(p("status"), Ok(Some(AdminCommand::Status)));
        assert_eq!(p("  PLAYERS  "), Ok(Some(AdminCommand::Players)));
        assert_eq!(p("reports"), Ok(Some(AdminCommand::Reports)));
        assert_eq!(p("reports clear"), Ok(Some(AdminCommand::ReportsClear)));
        assert!(AdminCommand::parse("reports now").unwrap_err().contains("usage"));
        assert_eq!(p("keys"), Ok(Some(AdminCommand::Keys)));
        assert_eq!(p("key add bob ab Moderator"), Ok(Some(AdminCommand::KeyAdd { name: "bob".into(), key: "ab".into(), role: AdminRole::Moderator })));
        assert_eq!(p("key remove bob"), Ok(Some(AdminCommand::KeyRemove("bob".into()))));
        assert!(AdminCommand::parse("key add bob").unwrap_err().contains("usage"));
        assert!(AdminCommand::parse("key add bob ab root").unwrap_err().contains("role"));
        assert!(AdminCommand::parse("key remove").unwrap_err().contains("usage"));
        assert!(AdminCommand::parse("key").unwrap_err().contains("usage"));
        assert_eq!(p("interest"), Ok(Some(AdminCommand::Interest)));
        assert_eq!(p("stop"), Ok(Some(AdminCommand::Quit)));
        assert_eq!(p("kick 3"), Ok(Some(AdminCommand::Kick { id: 3, reason: String::new() })));
        assert_eq!(p("kick 3 being rude  "), Ok(Some(AdminCommand::Kick { id: 3, reason: "being rude".into() })));
        assert_eq!(p("say hello  everyone"), Ok(Some(AdminCommand::Say("hello  everyone".into()))));
        assert_eq!(p("scene scenes/arena.scene"), Ok(Some(AdminCommand::Scene("scenes/arena.scene".into()))));
        assert_eq!(p(""), Ok(None));
        assert_eq!(p("   "), Ok(None));
    }

    #[test]
    fn bad_input_is_explained() {
        let p = |s| AdminCommand::parse(s).unwrap_err();
        assert!(p("kick").contains("usage"));
        assert!(p("kick bob").contains("not a player id"));
        assert!(p("say").contains("usage"));
        assert!(p("scene").contains("usage"));
        assert!(p("status now").contains("no arguments"));
        assert!(p("dance").contains("unknown command"));
    }

    #[test]
    fn ban_commands_and_durations_parse() {
        let p = |s| AdminCommand::parse(s);
        let ban = |t: &str, secs, r: &str| Ok(Some(AdminCommand::Ban { target: t.into(), secs, reason: r.into() }));
        assert_eq!(p("ban 3"), ban("3", 0, ""));
        assert_eq!(p("ban 3 2h cheating with aimbot"), ban("3", 7200, "cheating with aimbot"));
        assert_eq!(p("ban 3 perm"), ban("3", 0, ""));
        assert_eq!(p("ban 3 spamming links"), ban("3", 0, "spamming links"), "not a duration: part of the reason");
        assert_eq!(p("ban 0a1b-2c3d 30m"), ban("0a1b-2c3d", 1800, ""));
        assert_eq!(p("unban 0a1b2c3d"), Ok(Some(AdminCommand::Unban("0a1b2c3d".into()))));
        assert_eq!(p("bans"), Ok(Some(AdminCommand::Bans)));
        assert_eq!(p("reload"), Ok(Some(AdminCommand::Reload)));
        assert_eq!(p("save"), Ok(Some(AdminCommand::Save)));
        assert_eq!(p("restart"), Ok(Some(AdminCommand::Restart { redirect: String::new() })));
        assert_eq!(p("shutdown 10.0.0.2:9000"), Ok(Some(AdminCommand::Shutdown { redirect: "10.0.0.2:9000".into() })));
        assert_eq!(p("stats"), Ok(Some(AdminCommand::Stats)));
        assert!(AdminCommand::parse("ban").unwrap_err().contains("usage"));
        assert!(AdminCommand::parse("unban").unwrap_err().contains("usage"));
        assert!(AdminCommand::parse("shutdown a b").unwrap_err().contains("at most one"));
        for (d, secs) in [("30s", Some(30)), ("15m", Some(900)), ("2H", Some(7200)), ("7d", Some(604_800)), ("1w", Some(604_800)), ("forever", Some(0)), ("0m", None), ("m", None), ("5x", None), ("12", None)] {
            assert_eq!(parse_duration(d), secs, "{d}");
        }
    }

    #[test]
    fn roles_are_ordered_and_gate_commands() {
        use AdminRole::*;
        assert!(Observer < Moderator && Moderator < Admin);
        let role = |s: &str| AdminCommand::parse(s).unwrap().unwrap().required_role();
        for (cmd, want) in [("status", Observer), ("players", Observer), ("bans", Observer), ("stats", Observer), ("reports", Moderator), ("reports clear", Moderator), ("keys", Admin), ("key remove bob", Admin), ("key add a b admin", Admin), ("kick 1", Moderator), ("ban 1", Moderator), ("unban abcdef12", Moderator), ("say hi", Moderator), ("scene a.scene", Admin), ("reload", Admin), ("save", Admin), ("restart", Admin), ("shutdown", Admin), ("quit", Admin)] {
            assert_eq!(role(cmd), want, "{cmd}");
        }
    }
}
