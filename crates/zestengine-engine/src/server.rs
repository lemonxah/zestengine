//! Headless game host: load a scene file and run gameplay, physics and
//! networking with no window, no GPU and no audio device (NET-1).
//!
//! This is the loop a dedicated server builds on. It is the same code the
//! player runs: [`PlaySession`] for components and physics, the scene loader
//! for the file, `net_sync` for replication. The only things missing are the
//! ones that need a device: meshes are not uploaded (collision geometry,
//! skeletons and clips are kept), material files and shaders are not resolved
//! (a material is the model's template material or a placeholder), and the
//! skybox and baked lighting have nothing to apply to.
//!
//! Available with or without the `client` feature, so the same host runs in
//! tests and tools of a full build; with `client` off it is the only way to
//! run a scene.

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use glam::Mat4;

use zestengine_core::{ComponentCommand, ComponentRegistry, InputState};

pub use crate::net::NetSession;
pub use crate::play::PlaySession;
pub use crate::scene::LoadedScene;
use crate::play::{TickInput, World};
use crate::scene_loader::SceneFlow;
use crate::shaders::ShaderLibrary;

/// What to run.
#[derive(Clone, Debug, Default)]
pub struct HeadlessConfig {
    /// Project folder (holds `project.zestengine`, `scenes/`, `models/` ...).
    pub assets_root: PathBuf,
    /// Project-relative scene to start with; `None` takes the project's boot scene.
    pub scene: Option<String>,
    /// Physics settings; `None` takes the project file's (or the defaults).
    pub physics: Option<zestengine_core::PhysicsSettings>,
    /// Which content the host builds (NET-3): `Server` strips client-only
    /// objects and components from the scene and from every spawn; the default
    /// `Both` keeps everything.
    pub net_side: zestengine_core::NetRole,
}

/// One scene running without a renderer.
pub struct HeadlessHost {
    root: PathBuf,
    scene: LoadedScene,
    scene_path: String,
    play: PlaySession,
    registry: ComponentRegistry,
    shaders: ShaderLibrary,
    flow: SceneFlow,
    net: Option<NetSession>,
    input: InputState,
    /// Host commands the session left for the next tick (scene switch requests ...).
    deferred: Vec<ComponentCommand>,
    quit: bool,
    ticks: u64,
}

/// One networking + gameplay step, shared with the player's idle loop
/// (`GameApp::headless_tick`), after the caller pumped the session: tick components and physics,
/// route the networking commands, replicate, then pose skeletons (Bone Attach,
/// Weapon Hold and IK feed hitboxes and traces even when nothing is skinned for
/// display). Returns the commands left for the host.
#[allow(clippy::too_many_arguments)]
pub(crate) fn step(
    play: &mut PlaySession,
    scene: &mut LoadedScene,
    registry: &ComponentRegistry,
    shaders: &mut ShaderLibrary,
    net: &mut Option<NetSession>,
    input: &InputState,
    listener_fallback: Mat4,
    dt: f32,
) -> Vec<ComponentCommand> {
    let net_view = net.as_mut().map(|n| n.view()).unwrap_or_default();
    let screen = zestengine_core::ScreenInfo { size: glam::Vec2::ONE, ..Default::default() };
    let host = play.tick(
        &mut World { scene, renderer: None, registry, shaders },
        &TickInput { dt, input, net: &net_view, listener_fallback, screen: &screen },
    );
    let mut left = Vec::new();
    for c in host {
        if let Some(c) = crate::net_sync::route_command(net, scene, c) {
            left.push(c);
        }
    }
    if let Some(net) = net.as_mut() {
        play.net_predict(scene, net, input, dt);
        scene.network_sync(net, dt);
    }
    scene.update_water_wind(dt);
    scene.pose_skeletons(dt);
    left
}

impl HeadlessHost {
    /// Load the scene and start a play session on it. `register` adds the game's
    /// own component types to the built-ins, as `run_game` does.
    pub fn open(config: HeadlessConfig, register: impl FnOnce(&mut ComponentRegistry)) -> Result<Self> {
        let mut registry = ComponentRegistry::with_builtins();
        register(&mut registry);
        let project = zestengine_assets::load_project_file(&config.assets_root).ok();
        if let Some(p) = &project {
            p.apply_role_overrides(&mut registry);
        }
        let scene_path = match config.scene.clone() {
            Some(s) => s,
            None => project
                .as_ref()
                .and_then(|p| p.boot_scene.clone().or_else(|| p.last_scene.clone()))
                .context("no scene given and the project has no boot_scene")?,
        };
        let physics = config.physics.or_else(|| project.map(|p| p.physics)).unwrap_or_default();
        let mut shaders = ShaderLibrary::default();
        let scene = load_scene_for(config.net_side, &config.assets_root, &scene_path, &registry, &mut shaders)?;
        let mut play = PlaySession::new(&scene, &config.assets_root, None);
        play.set_physics(&physics, &scene);
        Ok(Self {
            root: config.assets_root,
            scene,
            scene_path,
            play,
            registry,
            shaders,
            flow: SceneFlow::default(),
            net: None,
            input: InputState::default(),
            deferred: Vec::new(),
            quit: false,
            ticks: 0,
        })
    }

    pub fn scene(&self) -> &LoadedScene {
        &self.scene
    }

    pub fn scene_mut(&mut self) -> &mut LoadedScene {
        &mut self.scene
    }

    /// Tell the play session the scene's objects were edited from outside
    /// (a restored transform): physics bodies and caches are rebuilt from it.
    pub fn scene_edited(&mut self) {
        self.play.scene_changed(&self.scene);
    }

    pub fn play(&self) -> &PlaySession {
        &self.play
    }

    pub fn play_mut(&mut self) -> &mut PlaySession {
        &mut self.play
    }

    /// Project-relative path of the running scene.
    pub fn scene_path(&self) -> &str {
        &self.scene_path
    }

    pub fn assets_root(&self) -> &Path {
        &self.root
    }

    pub fn registry(&self) -> &ComponentRegistry {
        &self.registry
    }

    /// Attach a networking session (host or client); it is pumped every tick.
    pub fn set_net(&mut self, net: NetSession) {
        self.net = Some(net);
    }

    pub fn net(&self) -> Option<&NetSession> {
        self.net.as_ref()
    }

    pub fn net_mut(&mut self) -> Option<&mut NetSession> {
        self.net.as_mut()
    }

    /// Start hosting / joining if the scene's Network Manager asks for it
    /// (`auto_start`), like a built game does on its first frame. Once per process.
    pub fn auto_start_net(&mut self) {
        crate::net_sync::auto_start(&mut self.net, &self.scene);
    }

    /// Input fed to components (a server has none; tests may press keys).
    pub fn input_mut(&mut self) -> &mut InputState {
        &mut self.input
    }

    /// A component asked to quit.
    pub fn quit_requested(&self) -> bool {
        self.quit
    }

    /// Forget a handled quit request (a server that restarts its scene instead of exiting).
    pub fn clear_quit(&mut self) {
        self.quit = false;
    }

    pub fn ticks(&self) -> u64 {
        self.ticks
    }

    /// Advance `dt` seconds: networking, components, physics (fixed sub-steps),
    /// replication, skeleton posing, and a requested scene switch once its
    /// background load is in.
    pub fn tick(&mut self, dt: f32) {
        self.ticks += 1;
        if let Some(net) = self.net.as_mut() {
            net.pump();
        }
        let mut host = std::mem::take(&mut self.deferred);
        host.extend(step(
            &mut self.play,
            &mut self.scene,
            &self.registry,
            &mut self.shaders,
            &mut self.net,
            &self.input,
            Mat4::IDENTITY,
            dt,
        ));
        self.apply_host_commands(host);
        self.poll_scene_switch();
    }

    fn apply_host_commands(&mut self, commands: Vec<ComponentCommand>) {
        for c in commands {
            match c {
                ComponentCommand::LoadScene(rel) => {
                    self.flow.request_switch(&rel, &self.root, None, &self.scene, &self.shaders);
                }
                ComponentCommand::PreloadScene(rel) => {
                    self.flow.preload(&rel, &self.root, None, &self.scene, &self.shaders);
                }
                ComponentCommand::LoadSceneAdditive(rel) => self.flow.load_additive(&rel, &self.root, None),
                ComponentCommand::UnloadScene(rel) => {
                    if self.flow.unload(&rel, &mut self.scene) {
                        self.play.structure_changed();
                    }
                }
                ComponentCommand::KeepOnSceneChange(id, keep) => {
                    if keep {
                        self.flow.kept.insert(id);
                    } else {
                        self.flow.kept.remove(&id);
                    }
                }
                ComponentCommand::Quit => self.quit = true,
                // Window, renderer, cursor, body-tracking and UI requests mean nothing without a client.
                other => tracing::debug!("headless host ignores {other:?}"),
            }
        }
    }

    fn poll_scene_switch(&mut self) {
        if self.flow.poll(None, &self.registry, &mut self.shaders, &self.root, &mut self.scene) {
            self.play.structure_changed();
        }
        let Some(result) = self.flow.poll_switch(None, &self.registry, &mut self.shaders, &self.root, &self.scene) else {
            return;
        };
        match result {
            Ok((rel, scene, _post)) => {
                self.scene = scene;
                if self.scene.net_side != zestengine_core::NetRole::Server {
                    self.scene.ensure_camera_components(&self.registry);
                    self.scene.ensure_light_components(&self.registry);
                }
                self.scene.ensure_camera_ids();
                self.scene_path = rel;
                self.play.scene_changed(&self.scene);
            }
            Err(e) => tracing::error!("loading scene: {e:#}"),
        }
    }
}

/// Load `scene_rel` (project-relative) CPU-side: file, prefab instances, models'
/// collision / skeleton / clip data, objects and components.
pub fn load_scene(root: &Path, scene_rel: &str, registry: &ComponentRegistry, shaders: &mut ShaderLibrary) -> Result<LoadedScene> {
    load_scene_for(zestengine_core::NetRole::Both, root, scene_rel, registry, shaders)
}

/// [`load_scene`] for one side of a networked game (see [`HeadlessConfig::net_side`]).
pub fn load_scene_for(
    side: zestengine_core::NetRole,
    root: &Path,
    scene_rel: &str,
    registry: &ComponentRegistry,
    shaders: &mut ShaderLibrary,
) -> Result<LoadedScene> {
    let data = crate::scene_loader::load_on_worker_cpu_for(&root.join(scene_rel), root, &[], &|_| {}, side)
        .map_err(|e| anyhow::anyhow!(e))
        .with_context(|| format!("loading {scene_rel}"))?;
    let (mut scene, _post) = crate::scene_loader::install(data, None, registry, shaders, root, |_| {})?;
    // Same safety net the player applies to hand-written scenes. A server has no
    // use for cameras or lights, and adding them back would undo the strip.
    if side != zestengine_core::NetRole::Server {
        scene.ensure_camera_components(registry);
        scene.ensure_light_components(registry);
    }
    scene.ensure_camera_ids();
    Ok(scene)
}

/// The content role a player process should filter scenes with, given its
/// session: a pure client of a dedicated server drops `Server` content
/// ([`zestengine_core::NetRole::Client`]); offline play, a listen host and a
/// client of a listen host keep everything (`Both`: a listen client may be
/// promoted to host by a migration, and the host runs the server logic).
pub fn session_role(net: Option<&crate::net::NetSession>) -> zestengine_core::NetRole {
    use zestengine_core::{NetRole, Topology};
    match net {
        Some(n) if !n.is_host() && n.topology() == Topology::Dedicated => NetRole::Client,
        _ => NetRole::Both,
    }
}

/// A shutdown notice that offers the players somewhere else to go: `notice` plus a
/// machine-readable `[redirect: <host:port or room code>]` that [`redirect_target`] finds in
/// the disconnect reason a player's session ends with.
pub fn redirect_notice(notice: &str, target: &str) -> String {
    format!("{notice} [redirect: {}]", target.trim())
}

/// The join target offered by a server's shutdown notice, if the text carries a valid one
/// (an address or a room code; anything else is ignored).
pub fn redirect_target(disconnect_reason: &str) -> Option<String> {
    let rest = &disconnect_reason[disconnect_reason.rfind("[redirect: ")? + "[redirect: ".len()..];
    let target = rest[..rest.find(']')?].trim();
    if target.is_empty() || target.len() > 128 || target.contains(char::is_whitespace) {
        return None;
    }
    crate::net::JoinTarget::parse(target, "").ok().map(|_| target.to_owned())
}

/// A player whose dedicated server went away offering a place to continue (a restart, a
/// move to another machine): join it with the same session settings. Returns the target joined.
/// Only a client of a dedicated server follows; a listen host's notice is not trusted this way.
pub fn follow_redirect(net: &mut Option<crate::net::NetSession>, scene: &LoadedScene) -> Option<String> {
    use crate::net::ConnState;
    let session = net.as_ref()?;
    if session.is_host() || session.topology() != zestengine_core::Topology::Dedicated {
        return None;
    }
    let ConnState::Disconnected(reason) = session.state() else { return None };
    let target = redirect_target(reason)?;
    let (cfg, _) = crate::net_sync::config_for(scene);
    match crate::net::NetSession::join_with(&target, cfg) {
        Ok(s) => {
            tracing::info!("net: the server asked players to continue at {target}");
            *net = Some(s);
            Some(target)
        }
        Err(e) => {
            tracing::warn!("net: cannot follow the server's redirect to {target}: {e}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::{MemHub, NetConfig};
    use crate::prefab::Prefab;
    use crate::scene::SceneObject;
    use glam::{Quat, Vec3};
    use serde::{Deserialize, Serialize};
    use zestengine_assets::{MaterialRef, ObjectSource, SceneEntry};
    use zestengine_core::{
        BoxCollider, Component, ComponentCtx, MeshCollider, NetValue, NetworkObject, NetworkVariables, ObjectId, RigidBody, SphereCollider,
        TypedComponent,
    };

    const DT: f32 = 1.0 / 60.0;

    /// Counts trigger entries.
    #[derive(Serialize, Deserialize, Default)]
    struct Counter {
        triggers: u32,
    }
    impl TypedComponent for Counter {
        const NAME: &'static str = "Server Test Counter";
        fn on_trigger_enter(&mut self, _ctx: &mut ComponentCtx, _other: ObjectId) {
            self.triggers += 1;
        }
    }

    /// Host-side gameplay: counts updates into a replicated variable.
    #[derive(Serialize, Deserialize, Default)]
    struct Scorer {
        #[serde(skip)]
        n: i32,
    }
    impl TypedComponent for Scorer {
        const NAME: &'static str = "Server Test Scorer";
        fn update(&mut self, ctx: &mut ComponentCtx) {
            if ctx.is_server() {
                self.n += 1;
                ctx.set_net_var_self("score", self.n);
            }
        }
    }

    fn register(r: &mut ComponentRegistry) {
        r.register::<Counter>();
        r.register::<Scorer>();
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("zestengine-server-test-{tag}-{}", ObjectId::new()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn object(name: &str, pos: Vec3, source: ObjectSource, components: Vec<Box<dyn Component>>) -> SceneObject {
        SceneObject {
            net_role: Default::default(),
            component_roles: Default::default(),
            id: ObjectId::new(),
            name: name.into(),
            render: None,
            extra_render: Vec::new(),
            source,
            enabled: true,
            static_geometry: false,
            layer: 0,
            parent: None,
            translation: pos,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            components,
        }
    }

    fn model(path: &str) -> ObjectSource {
        ObjectSource::Model { path: path.into(), mesh: 0, extra_meshes: vec![], model: None, mesh_id: None, extra_mesh_ids: vec![] }
    }

    /// A project with a floor, a trigger zone, a falling ball, a replicated
    /// counter, and a linked prefab instance of a real glb model.
    fn project() -> (PathBuf, ObjectId, ObjectId, ObjectId) {
        let root = temp_dir("project");
        std::fs::create_dir_all(root.join("scenes")).unwrap();
        std::fs::create_dir_all(root.join("prefabs")).unwrap();
        std::fs::create_dir_all(root.join("models")).unwrap();
        let glb = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/props/default/metal_crate.glb");
        std::fs::copy(&glb, root.join("models/crate.glb")).expect("repo models are present");

        let shaders = ShaderLibrary::default();
        let mut s = LoadedScene::empty();
        s.objects.push(object("Floor", Vec3::ZERO, ObjectSource::Empty, vec![Box::new(BoxCollider { size: [20.0, 1.0, 20.0], ..Default::default() })]));
        let zone = object(
            "Zone",
            Vec3::new(0.0, 2.0, 0.0),
            ObjectSource::Empty,
            vec![Box::new(BoxCollider { size: [2.0, 1.0, 2.0], is_trigger: true, ..Default::default() }), Box::new(Counter::default())],
        );
        let zone_id = zone.id;
        s.objects.push(zone);
        let ball = object("Ball", Vec3::new(0.0, 4.0, 0.0), ObjectSource::Empty, vec![Box::new(SphereCollider::default()), Box::new(RigidBody::default())]);
        let ball_id = ball.id;
        s.objects.push(ball);
        let scorer = object("Scorer", Vec3::ZERO, ObjectSource::Empty, vec![Box::new(NetworkObject::default()), Box::new(NetworkVariables::default()), Box::new(Scorer::default())]);
        let scorer_id = scorer.id;
        s.objects.push(scorer);
        let mut file = s.to_scene_file(&root, &shaders);

        // The prefab: one model object with a mesh collider, placed in the scene as a link.
        let mut p = LoadedScene::empty();
        p.objects.push(object("Crate", Vec3::ZERO, model("models/crate.glb"), vec![Box::new(MeshCollider::default())]));
        let mut entries = p.to_scene_file(&root, &shaders).entries;
        for e in &mut entries {
            e.material = MaterialRef::File(String::new());
        }
        Prefab::new(entries).save(root.join("prefabs/crate.prefab")).unwrap();
        file.entries.push(crate::prefab::link_root("prefabs/crate.prefab", "Crate Instance", [5.0, 0.5, 0.0]));
        zestengine_assets::save_scene_file(root.join("scenes/test.scene"), &file).unwrap();
        let _: Option<SceneEntry> = None;
        (root, zone_id, ball_id, scorer_id)
    }

    fn counter(host: &HeadlessHost, id: ObjectId) -> u32 {
        let i = host.scene().index_of(id).unwrap();
        host.scene().objects[i].components.iter().find_map(|c| c.as_any().downcast_ref::<Counter>()).unwrap().triggers
    }

    #[test]
    fn scene_file_loads_without_a_renderer_and_physics_runs() {
        let (root, zone, ball, _) = project();
        let config = HeadlessConfig { assets_root: root.clone(), scene: Some("scenes/test.scene".into()), physics: None, net_side: Default::default() };
        let mut host = HeadlessHost::open(config, register).unwrap();
        // Prefab instance expanded CPU-side, with its model's collision geometry and no GPU handles.
        let crate_i = host.scene().objects.iter().position(|o| o.name.starts_with("Crate")).expect("prefab instance expanded");
        let slot = host.scene().objects[crate_i].render.expect("model object has a render slot");
        assert!(host.scene().mesh_geometry(slot.mesh).is_some_and(|(v, i)| !v.is_empty() && !i.is_empty()), "collision geometry kept");
        assert_eq!(host.scene().mesh_handle(slot.mesh), Some(zestengine_render::MeshHandle::NONE), "nothing uploaded");
        for _ in 0..180 {
            host.tick(DT);
        }
        let b = host.scene().world_transform(host.scene().index_of(ball).unwrap()).w_axis.truncate();
        assert!((b.y - 1.0).abs() < 0.1, "ball rests on the floor: {b:?}");
        assert_eq!(counter(&host, zone), 1, "ball passed through the zone once");
        assert_eq!(host.ticks(), 180);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn two_headless_hosts_replicate_over_a_loopback_hub() {
        let (root, _, _, scorer) = project();
        let open = || HeadlessHost::open(HeadlessConfig { assets_root: root.clone(), scene: Some("scenes/test.scene".into()), physics: None, net_side: Default::default() }, register);
        // Same file, so the same object ids on both peers.
        let (mut server, mut client) = (open().unwrap(), open().unwrap());
        let hub = MemHub::new();
        let cfg = |name: &str| NetConfig { player_name: name.into(), discovery: false, secret: String::new(), relay: String::new(), ..Default::default() };
        let mut host_net = NetSession::host_on(Box::new(hub.endpoint()), cfg("Server"));
        host_net.use_manual_clock();
        let mut join_net = NetSession::join_on(Box::new(hub.endpoint()), host_net.local_addr().unwrap(), cfg("Ann"));
        join_net.use_manual_clock();
        server.set_net(host_net);
        client.set_net(join_net);
        for _ in 0..120 {
            for h in [&mut server, &mut client] {
                h.net_mut().unwrap().advance(DT as f64);
                h.tick(DT);
            }
        }
        assert!(client.net().unwrap().is_connected(), "client joined the headless server");
        let var = |h: &HeadlessHost| {
            let i = h.scene().index_of(scorer).unwrap();
            h.scene().objects[i].components.iter().find_map(|c| c.as_any().downcast_ref::<NetworkVariables>()).and_then(|v| v.get("score").cloned())
        };
        let (Some(NetValue::Int(s)), Some(NetValue::Int(c))) = (var(&server), var(&client)) else { panic!("score replicated: {:?} / {:?}", var(&server), var(&client)) };
        assert!(s > 60, "server counted updates: {s}");
        assert!(c > 0 && c <= s, "client sees the server's value: {c} (server {s})");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A player that joins a dedicated server filters scenes as a `Client`;
    /// offline play, a listen host and a listen host's client stay `Both`.
    #[test]
    fn joining_a_dedicated_server_makes_the_process_a_client() {
        use zestengine_core::{NetRole, Topology};
        let cfg = |name: &str, topology| NetConfig { player_name: name.into(), discovery: false, secret: String::new(), relay: String::new(), topology, ..Default::default() };
        let pair = |topology| {
            let hub = MemHub::new();
            let mut host = NetSession::host_on(Box::new(hub.endpoint()), cfg("Host", topology));
            host.use_manual_clock();
            let mut client = NetSession::join_on(Box::new(hub.endpoint()), host.local_addr().unwrap(), cfg("Ann", topology));
            client.use_manual_clock();
            for _ in 0..120 {
                host.advance(0.016);
                host.pump();
                client.advance(0.016);
                client.pump();
            }
            assert!(client.is_connected(), "{topology:?}: client joined");
            (host, client)
        };
        assert_eq!(session_role(None), NetRole::Both, "offline play keeps everything");
        let (host, client) = pair(Topology::Dedicated);
        assert_eq!(session_role(Some(&client)), NetRole::Client);
        assert_eq!(session_role(Some(&host)), NetRole::Both, "the hosting session itself is not a client");
        let (host, client) = pair(Topology::ListenHost);
        assert_eq!(session_role(Some(&client)), NetRole::Both, "a listen host's client may be promoted by a migration");
        assert_eq!(session_role(Some(&host)), NetRole::Both);
    }

    #[test]
    fn a_template_project_loads_and_ticks_without_a_renderer() {
        let parent = temp_dir("template");
        for id in ["topdown", "vr_social"] {
            let root = crate::bundle::scaffold_project(&parent, id, id).unwrap_or_else(|e| panic!("{id}: {e:#}"));
            let mut host = HeadlessHost::open(HeadlessConfig { assets_root: root, ..Default::default() }, |_| {}).unwrap_or_else(|e| panic!("{id}: {e:#}"));
            assert!(!host.scene().objects.is_empty(), "{id}: objects");
            for _ in 0..60 {
                host.tick(DT);
            }
            assert!(host.scene().objects.iter().all(|o| o.translation.is_finite()), "{id}: transforms stay finite");
        }
        let _ = std::fs::remove_dir_all(&parent);
    }

    /// The test project's scene with extra role-tagged content: a client-only
    /// effect with a `Both` child, a server-only score keeper, an audio source
    /// and a light on the floor, and a trigger counter pointing at the effect.
    fn role_project() -> (PathBuf, String) {
        let (root, ..) = project();
        let path = root.join("scenes/test.scene");
        let mut file = zestengine_assets::load_scene_file(&path).unwrap();
        let reg = {
            let mut r = ComponentRegistry::with_builtins();
            register(&mut r);
            r
        };
        let template = file.entries[0].clone();
        let add = |name: &str, parent: Option<usize>, role: zestengine_core::NetRole, file: &mut zestengine_assets::SceneFile| {
            let mut e = template.clone();
            e.id = ObjectId::new().to_string();
            e.name = name.into();
            e.parent = parent;
            e.net_role = role;
            e.components.clear();
            file.entries.push(e);
            file.entries.len() - 1
        };
        use zestengine_core::NetRole;
        let smoke = add("Smoke", None, NetRole::Client, &mut file);
        add("Smoke Child", Some(smoke), NetRole::Both, &mut file);
        add("Score Keeper", None, NetRole::Server, &mut file);
        let smoke_id = file.entries[smoke].id.clone();
        let floor = &mut file.entries[0];
        floor.components.push(zestengine_assets::ComponentData::of(reg.create("Audio Source").unwrap().as_ref()));
        floor.components.push(zestengine_assets::ComponentData::of(reg.create("Light").unwrap().as_ref()));
        // The Light stays on the server by explicit override; the Audio Source follows its type default.
        floor.component_roles.insert("Light".into(), NetRole::Both);
        let zone = file.entries.iter_mut().find(|e| e.name == "Zone").unwrap();
        for c in &mut zone.components {
            if c.kind == "Server Test Counter" {
                c.data = format!("(triggers: 0, effect: \"{smoke_id}\")");
            }
        }
        zestengine_assets::save_scene_file(&path, &file).unwrap();
        (root, smoke_id)
    }

    fn open_side(root: &Path, side: zestengine_core::NetRole) -> HeadlessHost {
        HeadlessHost::open(HeadlessConfig { assets_root: root.to_owned(), scene: Some("scenes/test.scene".into()), physics: None, net_side: side }, register).unwrap()
    }

    fn has(host: &HeadlessHost, name: &str) -> bool {
        host.scene().objects.iter().any(|o| o.name == name)
    }

    #[test]
    fn roles_filter_the_scene_on_server_client_and_listen_hosts() {
        use zestengine_core::NetRole;
        let (root, _) = role_project();
        let mut server = open_side(&root, NetRole::Server);
        assert!(!has(&server, "Smoke") && !has(&server, "Smoke Child"), "client-only object and its Both child are gone");
        assert!(has(&server, "Score Keeper") && has(&server, "Floor") && has(&server, "Zone"));
        let floor = server.scene().objects.iter().find(|o| o.name == "Floor").unwrap();
        let kinds: Vec<&str> = floor.components.iter().map(|c| c.type_name()).collect();
        assert!(kinds.contains(&"Box Collider") && kinds.contains(&"Light"), "collider and the overridden light stay: {kinds:?}");
        assert!(!kinds.contains(&"Audio Source"), "audio follows its type default: {kinds:?}");
        for _ in 0..10 {
            server.tick(DT);
        }

        let client = open_side(&root, NetRole::Client);
        assert!(has(&client, "Smoke") && has(&client, "Smoke Child") && !has(&client, "Score Keeper"));
        let floor = client.scene().objects.iter().find(|o| o.name == "Floor").unwrap();
        assert!(floor.components.iter().any(|c| c.type_name() == "Audio Source"));

        let both = open_side(&root, NetRole::Both);
        assert!(has(&both, "Smoke") && has(&both, "Smoke Child") && has(&both, "Score Keeper"), "a listen host keeps everything");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn spawns_drop_the_other_sides_content() {
        use zestengine_core::NetRole;
        let (root, _) = role_project();
        let reg = {
            let mut r = ComponentRegistry::with_builtins();
            register(&mut r);
            r
        };
        let mk = |net_role| {
            let mut entries = zestengine_assets::load_scene_file(root.join("scenes/test.scene")).unwrap().entries;
            entries.truncate(1);
            let mut root_e = entries[0].clone();
            root_e.name = "Pickup".into();
            root_e.id = ObjectId::new().to_string();
            root_e.components.clear();
            root_e.parent = None;
            let mut fx = root_e.clone();
            fx.name = "Pickup Glow".into();
            fx.id = ObjectId::new().to_string();
            fx.parent = Some(0);
            fx.net_role = net_role;
            vec![root_e, fx]
        };
        for (side, glow) in [(NetRole::Server, false), (NetRole::Client, true), (NetRole::Both, true)] {
            let mut host = open_side(&root, side);
            let mut shaders = ShaderLibrary::default();
            let first = host.scene_mut().append_entries_with(None, &mut shaders, &root, &reg, &mk(NetRole::Client)).unwrap();
            assert_eq!(host.scene().objects[first].name, "Pickup");
            assert_eq!(has(&host, "Pickup Glow"), glow, "{side:?}: client-only child of a spawn");
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_stripped_target_resolves_to_missing_without_a_crash() {
        let (root, smoke_id) = role_project();
        let mut host = open_side(&root, zestengine_core::NetRole::Server);
        let id: ObjectId = smoke_id.parse().unwrap();
        assert!(host.scene().index_of(id).is_none(), "the lookup says missing");
        for _ in 0..30 {
            host.tick(DT);
        }
        let _ = std::fs::remove_dir_all(&root);
    }
}
