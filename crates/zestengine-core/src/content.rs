//! User-generated content (UGC) API: the gameplay half of `.zestenginepak`
//! packages. Components request loads / spawns / world switches through
//! [`ComponentCtx`]; the engine's `ContentManager` does the work off the main
//! thread and reports back with [`ContentEvent`]s (read via `ctx.events`).
//! [`AvatarDescriptor`] marks the root of an avatar package so it can be
//! attached to a player and the VR rig.

use std::sync::atomic::{AtomicU64, Ordering};

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::{ComponentCommand, ComponentCtx, ObjectId, ObjectRef, TypedComponent};

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

impl ComponentCtx<'_> {
    /// Load a content package without placing it. `source` is a file path, an
    /// `http(s)://` URL or the content id of a package loaded earlier.
    /// Completion arrives as [`ContentEvent::Loaded`] / [`ContentEvent::Failed`].
    pub fn load_content(&mut self, source: impl Into<String>) -> ContentRequest {
        let request = ContentRequest::next();
        self.commands.push(ComponentCommand::Content(ContentCommand::Load {
            request,
            source: source.into(),
        }));
        request
    }

    /// Spawn a Prop (or Avatar) package at a world transform. Returns the root
    /// object's id immediately; it exists once [`ContentEvent::Spawned`] fires.
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

    /// Make an Avatar package the visual of `player` (replacing its previous one).
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

    /// Switch to a World package; the current scene keeps running until it is
    /// ready (objects marked `keep_on_scene_change` come along).
    pub fn enter_world(&mut self, source: impl Into<String>) -> ContentRequest {
        let request = ContentRequest::next();
        self.commands.push(ComponentCommand::Content(ContentCommand::EnterWorld {
            request,
            source: source.into(),
        }));
        request
    }

    /// Remove a package's instances and drop it from memory (`id` = content id
    /// or the source it was loaded from).
    pub fn unload_content(&mut self, id: impl Into<String>) {
        self.commands.push(ComponentCommand::Content(ContentCommand::Unload(id.into())));
    }

    /// Content events from last frame.
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
