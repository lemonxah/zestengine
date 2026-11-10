//! User-generated content (UGC) API: the gameplay half of `.zestenginepak`
//! packages. Components request loads / spawns / world switches through
//! [`crate::ecs::ObjectContext`]; the engine's `ContentManager` does the work off the main
//! thread and reports back with [`ContentEvent`]s (read via `ctx.events`).
//! [`AvatarDescriptor`] marks the root of an avatar package so it can be
//! attached to a player and the VR rig.

use std::sync::atomic::{AtomicU64, Ordering};

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::{ComponentCommand, ObjectId, ObjectRef, TypedComponent};

/// What a content package holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContentKind {
    /// A whole scene; entering it replaces the current world.
    #[default]
    World,
    /// A character hierarchy rooted at an [`AvatarDescriptor`].
    Avatar,
    /// An object hierarchy spawned into the current world.
    Prop,
}

impl ContentKind {
    /// Every kind, in declaration order.
    pub const ALL: [ContentKind; 3] = [Self::World, Self::Avatar, Self::Prop];

    /// Inspector label of this kind.
    pub fn label(self) -> &'static str {
        match self {
            Self::World => "World",
            Self::Avatar => "Avatar",
            Self::Prop => "Prop",
        }
    }
}

/// Handle for one content request; matches the request to its events.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ContentRequest(pub u64);

static NEXT_REQUEST: AtomicU64 = AtomicU64::new(1);

impl ContentRequest {
    /// A fresh request id.
    pub fn next() -> Self {
        Self(NEXT_REQUEST.fetch_add(1, Ordering::Relaxed))
    }
}

/// Summary of a loaded package.
#[derive(Clone, Debug, Default)]
pub struct ContentInfo {
    /// Content hash (hex); stable id for spawning / unloading.
    pub id: String,
    /// What the package holds.
    pub kind: ContentKind,
    /// Package name.
    pub name: String,
    /// Package author.
    pub author: String,
    /// Package version.
    pub version: String,
    /// Non-fatal issues found while loading (dropped components, sanitized paths).
    pub warnings: Vec<String>,
}

/// Progress and results of content requests, broadcast to every component.
#[derive(Clone, Debug)]
pub enum ContentEvent {
    /// A load step finished (what and how far).
    Progress {
        /// The request making progress.
        request: ContentRequest,
        /// The step's name.
        stage: String,
    },
    /// The package is parsed, validated and uploaded; ready to spawn.
    Loaded {
        /// The request that finished.
        request: ContentRequest,
        /// Summary of the loaded package.
        info: ContentInfo,
    },
    /// A prop / avatar instance was added. `root` is the id the request returned.
    Spawned {
        /// The request that spawned it.
        request: ContentRequest,
        /// Summary of the spawned package.
        info: ContentInfo,
        /// The new instance's root object.
        root: ObjectId,
    },
    /// A world package replaced the current scene.
    WorldEntered {
        /// The request that entered the world.
        request: ContentRequest,
        /// Summary of the world package.
        info: ContentInfo,
    },
    /// The request failed.
    Failed {
        /// The failed request.
        request: ContentRequest,
        /// The source it was asked to load.
        source: String,
        /// Why it failed.
        error: String,
    },
    /// A package was removed.
    Unloaded {
        /// Content id of the removed package.
        id: String,
    },
}

impl ContentEvent {
    /// The request of the event, when it has one.
    pub fn request(&self) -> Option<ContentRequest> {
        match self {
            Self::Progress { request, .. }
            | Self::Loaded { request, .. }
            | Self::Spawned { request, .. }
            | Self::WorldEntered { request, .. }
            | Self::Failed { request, .. } => Some(*request),
            Self::Unloaded { .. } => None,
        }
    }
}

/// How a spawned package instance is placed.
#[derive(Clone, Debug)]
pub enum ContentPlacement {
    /// A new root object at a world transform.
    World {
        /// World position to place the root at (m).
        translation: Vec3,
        /// World rotation to place the root at.
        rotation: Quat,
    },
    /// The avatar of a player object: parented under it at its origin, any
    /// previous avatar of that player removed, and the player's camera moved
    /// to the descriptor's viewpoint.
    AvatarOf(ObjectId),
}

/// Content requests (see the `ctx` methods below).
#[derive(Debug)]
pub enum ContentCommand {
    /// Fetch + validate + upload a package (path, `http(s)://` URL or content id).
    Load {
        /// The request id events report under.
        request: ContentRequest,
        /// Path, URL or content id to load.
        source: String,
    },
    /// Load (if needed) then spawn a Prop / Avatar package.
    Spawn {
        /// The request id events report under.
        request: ContentRequest,
        /// Path, URL or content id to load.
        source: String,
        /// The new instance's root object id.
        root_id: ObjectId,
        /// How the instance is placed.
        placement: ContentPlacement,
    },
    /// Load (if needed) then switch to a World package (seamless scene switch).
    EnterWorld {
        /// The request id events report under.
        request: ContentRequest,
        /// Path, URL or content id to load.
        source: String,
    },
    /// Remove every instance of a package and forget it.
    Unload(String),
}

impl crate::ecs::ObjectContext<'_, '_> {
    /// Loads a content package (a `.zestenginepak` world, avatar or prop) without placing it anywhere.
    ///
    /// `source` is a file path (absolute, or relative to the working directory or the project's asset root), an `http://`
    /// URL, or a content id: the 64-digit hash of a package loaded earlier ([`ContentInfo::id`]). An `https://` URL fails
    /// with a [`ContentEvent::Failed`] unless the game installed a fetcher that speaks TLS; the built-in one is plain HTTP.
    ///
    /// **Queued (host).** Pushes [`ComponentCommand::Content`]. The play session does not handle it, so the host
    /// application hands it to the engine's content manager after the frame's last stage. Downloading, validating,
    /// unpacking and uploading to the GPU then run on a background thread over the following frames; nothing blocks the
    /// game. The returned [`ContentRequest`] is the handle every resulting event carries, so match it in
    /// [`content_events`](Self::content_events): zero or more [`ContentEvent::Progress`], then exactly one of
    /// [`ContentEvent::Loaded`] or [`ContentEvent::Failed`] (with the reason). A package that is already loaded answers
    /// with `Loaded` straight away without fetching again, and two requests for the same source share one load but each
    /// get their own events.
    ///
    /// Loading only keeps the package ready. [`spawn_content`](Self::spawn_content), [`spawn_avatar`](Self::spawn_avatar)
    /// and [`enter_world`](Self::enter_world) load it first when needed, so call this to preload or to inspect the
    /// [`ContentInfo`] (name, author, kind, warnings) before committing. Free it with
    /// [`unload_content`](Self::unload_content). Only hosts that run a content manager handle these requests (the
    /// player and the editor's Play mode); a headless dedicated server ignores them and no event ever arrives.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Start the load and remember the request.
    /// self.pending = Some(ctx.load_content("worlds/arena.zestenginepak"));
    ///
    /// // On later frames, wait for its result.
    /// for event in ctx.content_events() {
    ///     match event {
    ///         ContentEvent::Loaded { request, info } if Some(*request) == self.pending => self.show_title(&info.name),
    ///         ContentEvent::Failed { request, error, .. } if Some(*request) == self.pending => self.show_error(error),
    ///         _ => {}
    ///     }
    /// }
    /// ```
    pub fn load_content(&mut self, source: impl Into<String>) -> ContentRequest {
        let request = ContentRequest::next();
        self.commands.push(ComponentCommand::Content(ContentCommand::Load {
            request,
            source: source.into(),
        }));
        request
    }

    /// Loads a prop or avatar package (when it is not loaded yet) and adds one instance of it to the world at a world
    /// position and rotation.
    ///
    /// Returns at once with the request handle and the id the instance's root object **will** have. The id is yours to
    /// store, but the object only exists when [`ContentEvent::Spawned`] arrives (its `root` is the same id), some frames
    /// later; until then lookups of that id find nothing. Failures arrive as [`ContentEvent::Failed`]: the package is a
    /// World (use [`enter_world`](Self::enter_world)), is not a valid package, exceeds the content limits, or could not be
    /// downloaded.
    ///
    /// **Queued (host).** Pushes [`ComponentCommand::Content`]. The play session does not handle it, so the host
    /// application hands it to the engine's content manager after the frame's last stage. The events are
    /// [`ContentEvent::Progress`] (any number), [`ContentEvent::Loaded`], then [`ContentEvent::Spawned`]. Every call adds
    /// one more instance, all sharing the one loaded package. The instance exists on this peer only: the call does not
    /// replicate it, so use [`net_spawn`](Self::net_spawn) for objects every player must see. Remove all instances of the
    /// package with [`unload_content`](Self::unload_content). A headless dedicated server ignores the request.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Place a downloaded lantern 3 m in front of the origin.
    /// let (request, root) = ctx.spawn_content("props/lantern.zestenginepak", Vec3::new(0.0, 1.0, -3.0), Quat::IDENTITY);
    /// self.lantern = Some(root);   // a valid id right away; the object itself appears with ContentEvent::Spawned
    /// ```
    pub fn spawn_content(
        &mut self,
        source: impl Into<String>,
        translation: Vec3,
        rotation: Quat,
    ) -> (ContentRequest, ObjectId) {
        let request = ContentRequest::next();
        let root_id = ObjectId::new();
        self.commands.push(ComponentCommand::Content(ContentCommand::Spawn {
            request,
            source: source.into(),
            root_id,
            placement: ContentPlacement::World { translation, rotation },
        }));
        (request, root_id)
    }

    /// Makes an avatar package the visual of `player`, replacing the avatar it wore before. Returns `None` (and queues
    /// nothing) when `player` is an unset reference.
    ///
    /// Otherwise it works like [`spawn_content`](Self::spawn_content): it returns the request handle and the root id at
    /// once and completes later with [`ContentEvent::Spawned`]. It fails with [`ContentEvent::Failed`] when the package
    /// is not an Avatar, or when `player` is no longer in the scene by the time the package is ready. When the avatar is
    /// ready its root is parented under `player` at its origin, the player's previous avatar is removed and freed, the
    /// player's camera moves to the eye point of the avatar's [`AvatarDescriptor`], the player's Character Controller
    /// capsule takes the descriptor's body size (when the descriptor gives one), and skinned meshes get full-body IK
    /// (meshes with blend shapes also get face tracking and lip sync).
    ///
    /// **Queued (host).** Pushes [`ComponentCommand::Content`]. The play session does not handle it, so the host
    /// application hands it to the engine's content manager after the frame's last stage.
    ///
    /// In a session, when `player` carries a [`NetworkAvatar`](crate::NetworkAvatar) and is your own player (not another
    /// peer's copy), the avatar you wear is announced through a synced variable and every other peer loads and wears it on
    /// its copy of your player. They fetch it from the `http(s)://` URL it was loaded from, or by content id when it is
    /// already in their cache, so an avatar loaded from a local file path only shows for peers that already have it.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Wear a downloaded avatar; `player` is the object with the Character Controller (and Network Avatar).
    /// if let Some((request, _root)) = ctx.spawn_avatar(player, "http://example.com/avatars/robot.zestenginepak") {
    ///     self.wearing = Some(request);
    /// }
    /// ```
    pub fn spawn_avatar(&mut self, player: impl Into<ObjectRef>, source: impl Into<String>) -> Option<(ContentRequest, ObjectId)> {
        let player = player.into().id()?;
        let request = ContentRequest::next();
        let root_id = ObjectId::new();
        self.commands.push(ComponentCommand::Content(ContentCommand::Spawn {
            request,
            source: source.into(),
            root_id,
            placement: ContentPlacement::AvatarOf(player),
        }));
        Some((request, root_id))
    }

    /// Switches to a World package: the current scene keeps running while the package loads, then the whole world is
    /// replaced in one seamless scene switch.
    ///
    /// Loads the package first when needed and returns the request handle at once. The events are
    /// [`ContentEvent::Progress`] (any number), [`ContentEvent::Loaded`], then [`ContentEvent::WorldEntered`] once the new
    /// scene is in, or [`ContentEvent::Failed`] when the package is not a World or cannot be loaded. Objects marked with
    /// [`keep_on_scene_change`](Self::keep_on_scene_change) come along; everything else, including content instances you
    /// spawned earlier, belongs to the old scene and is gone.
    ///
    /// **Queued (host).** Pushes [`ComponentCommand::Content`]. The play session does not handle it, so the host
    /// application hands it to the engine's content manager after the frame's last stage. The switch is local to this
    /// peer: it does not tell other peers to follow (for a project scene the host can use
    /// [`net_load_scene`](Self::net_load_scene)). A headless dedicated server ignores the request.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Leave the lobby for a downloaded world.
    /// let request = ctx.enter_world("worlds/castle.zestenginepak");
    /// self.entering = Some(request);   // ContentEvent::WorldEntered { request, .. } arrives once it is in
    /// ```
    pub fn enter_world(&mut self, source: impl Into<String>) -> ContentRequest {
        let request = ContentRequest::next();
        self.commands.push(ComponentCommand::Content(ContentCommand::EnterWorld {
            request,
            source: source.into(),
        }));
        request
    }

    /// Removes every instance of a content package from the world and drops the package from memory.
    ///
    /// `id` is the package's content id ([`ContentInfo::id`]) or the source string it was loaded from.
    ///
    /// **Queued (host).** Pushes [`ComponentCommand::Content`]. The play session does not handle it, so the host
    /// application hands it to the engine's content manager after the frame's last stage, which removes the instances
    /// (avatars players were wearing included) and frees what only they used. A [`ContentEvent::Unloaded`] with the
    /// content id arrives on a later frame. An `id` that matches no loaded package does nothing, with no event. Loading
    /// or spawning the same source afterwards loads it again.
    pub fn unload_content(&mut self, id: impl Into<String>) {
        self.commands.push(ComponentCommand::Content(ContentCommand::Unload(id.into())));
    }

    /// The [`ContentEvent`]s that arrived this frame, for every content request: the ones you made, and the ones the
    /// engine makes itself (for example to show another player's avatar).
    ///
    /// **Immediate.** The engine delivers each event once, on the frame after it happened, to every system. Nothing is
    /// kept, so look for the events you need in the frame they arrive and match them against the [`ContentRequest`] you
    /// stored ([`ContentEvent::request`] is `None` only for [`ContentEvent::Unloaded`]). Empty when no content manager is
    /// running, as on a headless dedicated server.
    pub fn content_events(&self) -> impl Iterator<Item = &ContentEvent> {
        self.events::<ContentEvent>()
    }
}

/// Root marker of an avatar package: where the eyes are, optional humanoid
/// bone overrides and the body capsule. The VR rig places the player's camera
/// at `view_position` and drives the humanoid skeleton from the trackers.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AvatarDescriptor {
    /// Eye point relative to the avatar root (y = eye height), meters.
    pub view_position: [f32; 3],
    /// Humanoid role -> skeleton joint name, for rigs whose bone names the
    /// automatic mapper doesn't recognise (roles: hips, head, left_upper_arm,
    /// left_lower_arm, left_hand, right_*, left_upper_leg, left_lower_leg,
    /// left_foot, right_*).
    pub bone_map: Vec<(String, String)>,
    /// Body capsule applied to the player's CharacterController (0 = keep).
    pub collider_radius: f32,
    /// Body capsule height applied to the player's CharacterController (0 = keep).
    pub collider_height: f32,
}

impl Default for AvatarDescriptor {
    fn default() -> Self {
        Self {
            view_position: [0.0, 1.6, 0.08],
            bone_map: Vec::new(),
            collider_radius: 0.3,
            collider_height: 1.75,
        }
    }
}

impl AvatarDescriptor {
    /// The descriptor's eye height above the avatar root (m).
    pub fn eye_height(&self) -> f32 {
        self.view_position[1]
    }
}

impl TypedComponent for AvatarDescriptor {
    const NAME: &'static str = "Avatar Descriptor";
}
