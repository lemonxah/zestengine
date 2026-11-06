//! The `deploy/` folder of a dedicated-server build (NET-7): a hardened systemd unit (single and
//! `@` template), a production `server.ron`, an env file, a Dockerfile and compose file, firewall
//! and logrotate examples and an install script, generated from the templates in
//! `server_deploy_templates/` with the project's name and ports filled in.
//!
//! None of this is run by the build: it is text. See docs/deployment/dedicated-server.md.

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};

/// Name of the folder inside `build-server/`.
pub const DEPLOY_DIR: &str = "deploy";
/// Default game UDP port.
pub const DEFAULT_GAME_PORT: u16 = 9000;

/// What the templates are filled with.
#[derive(Clone, Debug, PartialEq)]
pub struct DeployParams {
    /// Display name (the project's).
    pub name: String,
    /// Unit, user, folder and image name: `[a-z0-9-]`.
    pub slug: String,
    pub game_port: u16,
    /// Signed-admin UDP port; the rule is game + 1.
    pub admin_port: u16,
    /// Project-relative scene to host; `None` takes the project's boot scene.
    pub scene: Option<String>,
}

impl DeployParams {
    pub fn new(name: &str, scene: Option<&str>, game_port: u16) -> Self {
        Self {
            name: name.replace(['"', '\\', '\n', '\r'], ""),
            slug: slugify(name),
            game_port,
            admin_port: game_port.saturating_add(1),
            scene: scene.map(str::to_owned),
        }
    }

    pub fn install_dir(&self) -> String {
        format!("/opt/{}", self.slug)
    }

    /// The state folder written into `server.ron`: outside the build folder.
    pub fn state_dir(&self) -> String {
        format!("/var/lib/{}", self.slug)
    }

    pub fn log_dir(&self) -> String {
        format!("/var/log/{}", self.slug)
    }
}

/// The UDP port the scene's Network Manager names (the first one found), if the scene has one.
/// The dedicated server takes this port when `server.ron` has none, so the deploy files must
/// open the same one.
pub fn scene_game_port(project_root: &Path, scene: Option<&str>) -> Option<u16> {
    let rel = scene?;
    let file = zestengine_assets::load_scene_file(project_root.join(rel)).ok()?;
    file.entries.iter().flat_map(|e| e.components.iter()).find(|c| c.kind == <zestengine_core::NetworkManager as zestengine_core::TypedComponent>::NAME).and_then(|c| ron::from_str::<zestengine_core::NetworkManager>(&c.data).ok()).map(|m| m.port).filter(|p| *p != 0)
}

/// The port to deploy: the scene's Network Manager port, else [`DEFAULT_GAME_PORT`].
pub fn deploy_game_port(project_root: &Path, scene: Option<&str>) -> u16 {
    scene_game_port(project_root, scene).unwrap_or(DEFAULT_GAME_PORT)
}

/// Port of instance `n` (0-based) of a multi-instance host: game, admin. Game ports step by 10.
pub fn instance_ports(base_game_port: u16, n: u16) -> (u16, u16) {
    let game = base_game_port.saturating_add(n.saturating_mul(10));
    (game, game.saturating_add(1))
}

/// A lowercase `[a-z0-9-]` name safe for a unit, a user and a path; never empty.
pub fn slugify(name: &str) -> String {
    let mut s = String::new();
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            s.push(c.to_ascii_lowercase());
        } else if !s.ends_with('-') && !s.is_empty() {
            s.push('-');
        }
    }
    let s = s.trim_matches('-');
    // Linux user names stop at 32 characters, and the unit adds "@instance".
    let s: String = s.chars().take(28).collect();
    let s = s.trim_matches('-').to_owned();
    if s.is_empty() { "zestgame".into() } else { s }
}

fn fill(template: &str, p: &DeployParams) -> String {
    let (game1, admin1) = instance_ports(p.game_port, 1);
    let scene_line = p.scene.as_ref().map(|s| format!("    scene: \"{}\",\n", s.replace(['"', '\\'], ""))).unwrap_or_default();
    let mut out = template.to_owned();
    for (key, value) in [
        ("name", p.name.clone()),
        ("slug", p.slug.clone()),
        ("install_dir", p.install_dir()),
        ("state_dir", p.state_dir()),
        ("log_dir", p.log_dir()),
        ("game_port", p.game_port.to_string()),
        ("admin_port", p.admin_port.to_string()),
        ("game_port_1", game1.to_string()),
        ("admin_port_1", admin1.to_string()),
        ("scene_line", scene_line),
    ] {
        out = out.replace(&format!("{{{{{key}}}}}"), &value);
    }
    out
}

/// One generated file: name inside `deploy/`, text, executable.
pub struct DeployFile {
    pub name: String,
    pub text: String,
    pub executable: bool,
}

/// Every file of the `deploy/` folder.
pub fn deploy_files(p: &DeployParams) -> Vec<DeployFile> {
    let slug = &p.slug;
    let list: [(String, &str, bool); 13] = [
        (format!("{slug}.service"), include_str!("server_deploy_templates/service.tmpl"), false),
        (format!("{slug}@.service"), include_str!("server_deploy_templates/service_template.tmpl"), false),
        ("server.ron".into(), include_str!("server_deploy_templates/server.ron.tmpl"), false),
        ("instance.ron.example".into(), include_str!("server_deploy_templates/instance.ron.tmpl"), false),
        (format!("{slug}.env.example"), include_str!("server_deploy_templates/env.tmpl"), false),
        ("Dockerfile".into(), include_str!("server_deploy_templates/Dockerfile.tmpl"), false),
        ("docker-compose.yml".into(), include_str!("server_deploy_templates/compose.tmpl"), false),
        ("nftables.nft".into(), include_str!("server_deploy_templates/nftables.tmpl"), false),
        ("ufw.sh".into(), include_str!("server_deploy_templates/ufw.tmpl"), false),
        ("logrotate.conf".into(), include_str!("server_deploy_templates/logrotate.tmpl"), false),
        ("kubernetes.yaml".into(), include_str!("server_deploy_templates/kubernetes.tmpl"), false),
        ("install.sh".into(), include_str!("server_deploy_templates/install.tmpl"), true),
        ("README.txt".into(), include_str!("server_deploy_templates/readme.tmpl"), false),
    ];
    list.into_iter().map(|(name, tmpl, executable)| DeployFile { name, text: fill(tmpl, p), executable }).collect()
}

/// Write the `deploy/` folder into `build_dir`; returns its path.
pub fn write_deploy(build_dir: &Path, p: &DeployParams) -> Result<PathBuf> {
    let dir = build_dir.join(DEPLOY_DIR);
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    for f in deploy_files(p) {
        let path = dir.join(&f.name);
        std::fs::write(&path, &f.text).with_context(|| format!("writing {}", path.display()))?;
        #[cfg(unix)]
        if f.executable {
            use std::os::unix::fs::PermissionsExt as _;
            let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755));
        }
    }
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> DeployParams {
        DeployParams::new("Arena \"X\" Online!", Some("scenes/main.scene"), 9000)
    }

    fn get(files: &[DeployFile], name: &str) -> String {
        files.iter().find(|f| f.name == name).unwrap_or_else(|| panic!("no {name}")).text.clone()
    }

    #[test]
    fn slugs_are_safe() {
        assert_eq!(slugify("Arena \"X\" Online!"), "arena-x-online");
        assert_eq!(slugify("  ../../etc "), "etc");
        assert_eq!(slugify("***"), "zestgame");
        assert!(slugify(&"a".repeat(80)).len() <= 28);
        assert_eq!(instance_ports(9000, 0), (9000, 9001));
        assert_eq!(instance_ports(9000, 3), (9030, 9031));
    }

    #[test]
    fn no_placeholder_is_left_and_names_are_filled() {
        let p = params();
        let files = deploy_files(&p);
        assert_eq!(files.len(), 13);
        for f in &files {
            assert!(!f.text.contains("{{"), "{} has an unfilled placeholder:\n{}", f.name, f.text);
            assert!(f.text.contains("arena-x-online") || f.name.ends_with(".nft") && f.text.contains("arena-x-online"), "{} lacks the slug", f.name);
        }
    }

    #[test]
    fn unit_is_hardened_and_restarts_on_75() {
        let files = deploy_files(&params());
        let unit = get(&files, "arena-x-online.service");
        for needle in [
            "User=arena-x-online",
            "StateDirectory=arena-x-online",
            "ProtectSystem=strict",
            "NoNewPrivileges=yes",
            "PrivateTmp=yes",
            "RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX",
            "Restart=on-failure",
            "RestartForceExitStatus=75",
            "SuccessExitStatus=0",
            "TimeoutStopSec=10",
            "LimitNOFILE=",
            "#CPUAffinity=",
            "#Nice=",
            "DynamicUser=yes",
            "--no-console",
            "ExecStart=/opt/arena-x-online/zestengine-server --config /etc/arena-x-online/server.ron",
            "UDP 9000",
            "UDP 9001",
        ] {
            assert!(unit.contains(needle), "unit lacks {needle:?}:\n{unit}");
        }
        let tmpl = get(&files, "arena-x-online@.service");
        for needle in ["--config /etc/arena-x-online/%i.ron", "StateDirectory=arena-x-online/%i", "RestartForceExitStatus=75", "ProtectSystem=strict", "9010/9011", "Port allocation rule"] {
            assert!(tmpl.contains(needle), "template unit lacks {needle:?}:\n{tmpl}");
        }
    }

    #[test]
    fn production_ron_keeps_state_outside_the_build_folder() {
        let files = deploy_files(&params());
        let ron = get(&files, "server.ron");
        assert!(ron.contains("state_dir: \"/var/lib/arena-x-online\""), "{ron}");
        assert!(ron.contains("project: \"/opt/arena-x-online/assets\"") && ron.contains("scene: \"scenes/main.scene\""), "{ron}");
        assert!(ron.contains("port: 9000,") && ron.contains("admin_port: 9001") && ron.contains("server_name: \"Arena X Online!\""), "{ron}");
        for needle in ["PUT_THE_64_HEX_DIGIT_PUBLIC_KEY_HERE", "zestengine-admin keygen", "metrics_addr", "health_interval: 5.0", "lan_discovery: false", "ZESTENGINE_SERVER_PASSWORD"] {
            assert!(ron.contains(needle), "ron lacks {needle:?}");
        }
        assert!(!ron.contains("password: \""), "no secret in the file");
        assert!(!ron.contains("build-server"), "state must not point into the build folder");
        let _: ron::Value = ron::from_str(&ron).expect("valid RON");
        let inst = get(&files, "instance.ron.example");
        assert!(inst.contains("state_dir: \"/var/lib/arena-x-online/<instance>\"") && inst.contains("9100 + N"), "{inst}");
        let env = get(&files, "arena-x-online.env.example");
        assert!(env.contains("ZESTENGINE_SERVER_PASSWORD=") && env.contains("ZESTENGINE_RELAY_TOKEN="));
    }

    #[test]
    fn container_files_have_the_volume_ports_and_healthcheck() {
        let files = deploy_files(&params());
        let df = get(&files, "Dockerfile");
        for needle in ["VOLUME /var/lib/arena-x-online", "EXPOSE 9000/udp 9001/udp", "USER 10001:10001", "HEALTHCHECK", "--healthcheck", "--no-console", "STOPSIGNAL SIGTERM", "ARG BASE=debian:stable-slim"] {
            assert!(df.contains(needle), "Dockerfile lacks {needle:?}:\n{df}");
        }
        let dc = get(&files, "docker-compose.yml");
        for needle in ["9000:9000/udp", "state:/var/lib/arena-x-online", "restart: unless-stopped", "stop_grace_period: 10s", "read_only: true", "cap_drop: [ALL]", "env_file: arena-x-online.env", "127.0.0.1:9001:9001/udp"] {
            assert!(dc.contains(needle), "compose lacks {needle:?}:\n{dc}");
        }
    }

    #[test]
    fn firewall_logrotate_and_install_files() {
        let files = deploy_files(&params());
        let nft = get(&files, "nftables.nft");
        assert!(nft.contains("udp dport 9000 accept") && nft.contains("udp dport 9001 ip saddr $OPS_NET accept") && nft.contains("47777"), "{nft}");
        let ufw = get(&files, "ufw.sh");
        assert!(ufw.contains("ufw allow 9000/udp") && ufw.contains("port 9001 proto udp"), "{ufw}");
        let lr = get(&files, "logrotate.conf");
        assert!(lr.contains("/var/log/arena-x-online/*.log") && lr.contains("copytruncate") && lr.contains("rotate 14"), "{lr}");
        let inst = files.iter().find(|f| f.name == "install.sh").unwrap();
        assert!(inst.executable && inst.text.starts_with("#!/bin/sh") && inst.text.contains("useradd") && inst.text.contains("systemctl daemon-reload"));
        assert!(!inst.text.contains("systemctl enable --now arena-x-online\n"), "the script must not start anything itself except as advice");
    }

    #[test]
    fn kubernetes_manifest_has_pvc_deployment_and_udp_service() {
        let files = deploy_files(&DeployParams::new("Arena", Some("scenes/main.scene"), 7777));
        let k = get(&files, "kubernetes.yaml");
        assert!(!k.contains("{{"));
        for needle in ["kind: PersistentVolumeClaim", "kind: Deployment", "kind: Service", "protocol: UDP", "containerPort: 7777", "port: 7777", "type: Recreate", "replicas: 1", "mountPath: /var/lib/arena", "claimName: arena-state", "--healthcheck", "readOnlyRootFilesystem: true", "name: arena-env"] {
            assert!(k.contains(needle), "manifest lacks {needle:?}:\n{k}");
        }
        assert!(!k.contains("TCP"), "the game is UDP only");
        // Three documents, each valid YAML-ish: separators count.
        assert_eq!(k.matches("\n---\n").count(), 2);
        assert!(!k.contains("PASSWORD=secret"), "no secret in the manifest");
        // Ports follow the params everywhere.
        let ron = get(&files, "server.ron");
        assert!(ron.contains("port: 7777,"));
    }

    #[test]
    fn deploy_port_comes_from_the_scene_network_manager() {
        let parent = std::env::temp_dir().join(format!("zestengine-deploy-port-{}", std::process::id()));
        std::fs::create_dir_all(&parent).unwrap();
        let dir = crate::bundle::scaffold_project(&parent, "vr_social", "vr_social").unwrap();
        let project = zestengine_assets::load_project_file(&dir).unwrap();
        let boot = project.boot_scene.clone().unwrap();
        let mut file = zestengine_assets::load_scene_file(dir.join(&boot)).unwrap();
        file.entries.iter_mut().for_each(|e| e.components.retain(|c| c.kind != "Network Manager"));
        zestengine_assets::save_scene_file(dir.join("scenes/none.scene"), &file).unwrap();
        let nm = zestengine_core::NetworkManager { port: 7123, ..Default::default() };
        file.entries[0].components.push(zestengine_assets::ComponentData::new(<zestengine_core::NetworkManager as zestengine_core::TypedComponent>::NAME, ron::to_string(&nm).unwrap()));
        zestengine_assets::save_scene_file(dir.join("scenes/net.scene"), &file).unwrap();
        assert_eq!(scene_game_port(&dir, Some("scenes/net.scene")), Some(7123));
        assert_eq!(deploy_game_port(&dir, Some("scenes/net.scene")), 7123);
        assert_eq!(deploy_game_port(&dir, Some("scenes/none.scene")), DEFAULT_GAME_PORT);
        assert_eq!(deploy_game_port(&dir, Some("scenes/missing.scene")), DEFAULT_GAME_PORT);
        assert_eq!(deploy_game_port(&dir, None), DEFAULT_GAME_PORT);
        let _ = std::fs::remove_dir_all(&parent);
    }

    #[test]
    fn write_deploy_creates_the_folder() {
        let dir = std::env::temp_dir().join(format!("zestengine-deploy-{}", std::process::id()));
        let out = write_deploy(&dir, &params()).unwrap();
        assert!(out.join("arena-x-online.service").is_file() && out.join("Dockerfile").is_file() && out.join("README.txt").is_file());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            assert_eq!(std::fs::metadata(out.join("install.sh")).unwrap().permissions().mode() & 0o111, 0o111);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
