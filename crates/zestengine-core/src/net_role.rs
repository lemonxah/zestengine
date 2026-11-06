//! Content roles for dedicated servers (NET-3): which side of a networked game
//! an object or component belongs to.
//!
//! A scene object and each of its components carries a [`NetRole`]. `Both` (the
//! default, and what every old scene means) keeps the thing everywhere; `Client`
//! content is stripped from a dedicated server (cameras, audio, UI, particles ...);
//! `Server` content is stripped from a client (spawners, score keepers, anything
//! the players must not receive). A component without an explicit role takes its
//! type's default ([`TypedComponent::NET_ROLE`](crate::TypedComponent::NET_ROLE),
//! with a table for the built-ins in [`builtin_default_role`]).
//!
//! The decision rules live in [`NetRole::kept_on`]; the strip itself runs on scene
//! entries in `zestengine_assets::net_strip` (load time and server cook).

use std::sync::atomic::{AtomicU8, Ordering};

use serde::{Deserialize, Serialize};

/// Where an object or component exists.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NetRole {
    /// Only on players' machines.
    Client,
    /// Only on the dedicated server.
    Server,
    /// Everywhere (listen hosts, offline play, editor included).
    #[default]
    Both,
}

impl NetRole {
    /// Every role, in declaration order.
    pub const ALL: [NetRole; 3] = [NetRole::Both, NetRole::Client, NetRole::Server];

    /// True for `Both` (the role kept everywhere).
    pub fn is_both(&self) -> bool {
        *self == NetRole::Both
    }

    /// Whether content with this role exists in a process playing `side`.
    /// `side` is the process role: `Server` = dedicated server, `Client` = a
    /// joined player, `Both` = listen host / offline / editor (keeps everything).
    pub fn kept_on(self, side: NetRole) -> bool {
        self == NetRole::Both || side == NetRole::Both || self == side
    }

    /// Short name for lists and logs.
    pub fn label(self) -> &'static str {
        match self {
            NetRole::Both => "Both",
            NetRole::Client => "Client only",
            NetRole::Server => "Server only",
        }
    }

    /// One-letter hierarchy badge; `None` for `Both`.
    pub fn badge(self) -> Option<char> {
        match self {
            NetRole::Both => None,
            NetRole::Client => Some('C'),
            NetRole::Server => Some('S'),
        }
    }
}

/// Built-in component types that exist only for display, sound, input from a
/// player's device or other client-side presentation. Everything not listed is
/// `Both`: gameplay, physics, trigger, network, animation (poses feed hitboxes)
/// and avatar data components.
const CLIENT_ONLY: &[&str] = &[
    // Rendering and lighting.
    "Camera",
    "Light",
    "Light Probe Volume",
    "FluxVoxel Light",
    "Flux Volume",
    "Reflection Probe",
    "Planar Reflection",
    "Per Instance",
    "Mesh Renderer",
    "Skinned Mesh Renderer",
    "Volume (Post FX)",
    "Decal Projector",
    "Terrain Foliage",
    "Particle Emitter",
    "LOD Group",
    "Mesh Cloth",
    "Cloth Colliders",
    "Bone Cloth",
    "Wind Zone",
    "Text 3D",
    "Sprite Renderer",
    "Sorting Group",
    "Sprite Mask",
    // Audio.
    "Audio Source",
    "Audio Listener",
    "Acoustic Material",
    "Audio Probe Volume",
    // UI.
    "UI Canvas",
    "UI Rect",
    "UI Panel",
    "UI Image",
    "UI Text",
    "UI Button",
    "UI Toggle",
    "UI Slider",
    "UI Progress Bar",
    "UI Pause Menu",
    "UI Text Field",
    "HUD Binding",
    "Controls Menu",
    // Player-side cameras and avatar presentation.
    "Follow Camera",
    "RTS Camera",
    "Lip Sync",
    "Face Tracking",
    "Blend Shapes",
    "Avatar Menu Item",
    "Menu Toggle",
    "Menu Installer",
    "Merge Animator",
    "Merge Armature",
    "Avatar Menu",
];

/// The default role of a built-in component type by registry name.
pub fn builtin_default_role(name: &str) -> NetRole {
    if CLIENT_ONLY.contains(&name) { NetRole::Client } else { NetRole::Both }
}

/// Every built-in type with a non-`Both` default (documentation and tests).
pub fn builtin_client_only() -> &'static [&'static str] {
    CLIENT_ONLY
}

static PROCESS_ROLE: AtomicU8 = AtomicU8::new(2);

/// The role this process plays for content filtering at load time: `Both`
/// (keeps everything) unless the player runs as a pure joined client. A
/// dedicated server passes `Server` explicitly to its host instead of using this.
pub fn process_role() -> NetRole {
    match PROCESS_ROLE.load(Ordering::Relaxed) {
        0 => NetRole::Client,
        1 => NetRole::Server,
        _ => NetRole::Both,
    }
}

/// Set [`process_role`] (a player that only ever joins servers sets `Client`).
pub fn set_process_role(role: NetRole) {
    PROCESS_ROLE.store(
        match role {
            NetRole::Client => 0,
            NetRole::Server => 1,
            NetRole::Both => 2,
        },
        Ordering::Relaxed,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keep_rules() {
        use NetRole::*;
        for r in [Client, Server, Both] {
            assert!(r.kept_on(Both), "{r:?} stays in a listen host");
        }
        assert!(Both.kept_on(Server) && Both.kept_on(Client));
        assert!(Server.kept_on(Server) && !Server.kept_on(Client));
        assert!(Client.kept_on(Client) && !Client.kept_on(Server));
    }

    #[test]
    fn default_is_both_and_serde_round_trips() {
        assert_eq!(NetRole::default(), NetRole::Both);
        for r in NetRole::ALL {
            let text = ron::to_string(&r).unwrap();
            assert_eq!(ron::from_str::<NetRole>(&text).unwrap(), r);
        }
    }

    #[derive(serde::Serialize, serde::Deserialize, Default)]
    struct ServerOnlyThing;
    impl crate::TypedComponent for ServerOnlyThing {
        const NAME: &'static str = "Test Server Only Thing";
        const NET_ROLE: NetRole = NetRole::Server;
    }

    #[test]
    fn a_game_component_declares_its_own_default() {
        let mut reg = crate::ComponentRegistry::with_builtins();
        reg.register::<ServerOnlyThing>();
        assert_eq!(reg.default_role("Test Server Only Thing"), NetRole::Server);
        assert_eq!(reg.default_role("No Such Plugin Component"), NetRole::Both, "unknown types stay on both sides");
    }

    #[test]
    fn builtin_table_names_registered_types() {
        let reg = crate::ComponentRegistry::with_builtins();
        let names: Vec<_> = reg.names().collect();
        for n in builtin_client_only() {
            assert!(names.contains(n), "table names unknown component {n:?}");
            assert_eq!(reg.default_role(n), NetRole::Client);
        }
        for n in ["Box Collider", "Mesh Collider", "Rigid Body", "Network Object", "Network Manager", "Animator", "Spawn Point", "Health", "Voice Chat"] {
            assert_eq!(reg.default_role(n), NetRole::Both, "{n}");
        }
    }
}
