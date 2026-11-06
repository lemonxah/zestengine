//! zestengine-core: the engine-runtime API shared by the engine, editor, and
//! plugins. Egui-free by design so a shipped game links it (through the engine)
//! without pulling in any editor code.
//!
//! Holds the component/behaviour system: the [`Component`]/[`TypedComponent`]
//! traits (lifecycle hooks only, no UI), [`ComponentCtx`] (the in-game API
//! surface), [`ComponentRegistry`], [`Transform`], the deferred
//! [`ComponentCommand`]s, and the built-in component data. Editor-only concerns
//! (inspector UI, viewport gizmos) are separate traits in `zestengine-editor`.

use std::collections::HashMap;

use glam::{Mat4, Quat, Vec3};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

mod audio_mixer;
pub mod plugin_abi;
pub use plugin_abi::{PLUGIN_ABI_VERSION, PluginAbi};
pub mod avatar_params;
pub use avatar_params::{
    AvatarMenuItem, AvatarParam, AvatarParamKind, AvatarParamScope, AvatarParamStore, AvatarParamSync, AvatarParameters, BudgetLevel, BuiltinParam, MenuItemKind,
    ParamBudget, SavedParams, quantise_float, synced_bits,
};
pub mod avatar_authoring;
pub use avatar_authoring::{
    AVATAR_MENU_EXTENSION, AvatarLayer, AvatarMenu, AvatarMenuData, AvatarMenuNode, MenuInstaller, MenuToggle, MergeAnimator, MergeArmature, MergeMode, MergePathMode,
    ObjectToggle, SkinRebound, ObjectToggleEntry, ToggleAction, ToggleMode, normalise_label, split_menu_path,
};
pub mod avatar_sync;
pub use avatar_sync::{AvParamSchema, SyncClock, SyncError, smooth_step};
pub mod avatar_osc;
pub mod avatar_oscquery;
pub use avatar_osc::OscBridge;
pub mod param_driver;
pub use param_driver::{DriverEntry, DriverOp, PARAMETER_DRIVER, ParamDriverSettings, ParameterDriver, Remap};
pub mod ai;
pub use ai::{
    BehaviourTreeAgent, Blackboard, BtAction, BtNode, BtStatus, BtValue, Perception, PerceptionComponent,
    PerceptionState, tick,
};
pub mod destruction;
pub mod sequencer;
pub mod vehicle;
pub use vehicle::{Vehicle, Wheel};
mod cloth;
mod cloth_mesh;
pub use cloth_mesh::{MeshCloth, MeshClothRuntime, MeshClothSim, MeshParticle, PinSource};
mod cloth_pins;
pub use cloth_pins::{
    BrushFalloff, BrushMode, ClothPaint, Dab, PackedBytes, PaintChannel, PaintFit, WeldTopology, mirror_map, mirror_x,
    ray_triangles, select_by_height, topology_hash,
};
mod face;
pub mod constraint;
pub use constraint::{AimUp, Constraint, ConstraintKind, ConstraintSource, ConstraintSpec, SolveParams, SourceSpec, SourceWorld, TrsOffset, aim_rotation, solve_world};
pub mod custom_props;
pub use custom_props::{CustomProperties, CustomValue};
mod voice;
mod water;
mod water_optics;
pub use water_optics::WaterOptics;
mod water_waves;
pub use water_waves::{WaterWaveProfile, WaterWaveSpectrum};
mod water_interaction;
pub use water_interaction::SkeletalWaterInteraction;
mod water_sim;
mod water_pipe;
mod water_ewave;
mod physics_material;
mod physics2d;
mod sprite2d;
mod sprite_rig;
pub mod sprite_shape;
mod tilemap;
mod tile_asset;
mod tile_paint;
pub mod asset_ref;
pub mod lod;
mod water_fx;
mod water_container;
pub use water_container::{
    LITRES_PER_M3, WaterContainer, WaterContainerRuntime, held_volume, inside_box, level_for_volume, rim_corners, rim_level, spill_point, surface_polygon,
    volume_below, wet_lip,
};
mod wave_rider;
pub use wave_rider::{RideMode, RideState, WaveRider};
pub use water_fx::{GRAVITY, Vortex, WaterSplash, Waterfall, Whirlpool, fall_time, vortex_flow};
pub use physics2d::{
    BodyType2D, BoxCollider2D, CapsuleCollider2D, CapsuleDirection2D, CircleCollider2D, Collider2D, ColliderProps, CompositeCollider2D,
    CompositeGeometry2D, ContactPoint2D, EdgeCollider2D, Effector2D, EffectorKind2D, Interpolation2D, Joint2D, Joint2DKind,
    PointForceMode2D, PolygonCollider2D, RayHit2D, Rigidbody2D, SleepMode2D,
};
pub use tilemap::{
    CHUNK_SIZE, CellLayout, CellState, CellSwizzle, ChunkData, CullingBounds, FLAG_LOCK_COLOR, FLAG_LOCK_TRANSFORM, Grid, HEX_VERTICAL, ROT_STEPS, TileChunk,
    TileRef, TileSortOrder, TileXform, Tilemap, TilemapCollider2D, TilemapData, TilemapMode, TilemapOrientation, TilemapRenderer, cell_at, cell_index, chunk_of,
    neighbor_offset,
};
pub mod tile_object;
pub use tile_object::{HaveObject, TileObject, TileObjectPlan, WantObject, plan_tile_objects};
pub use tile_asset::{
    AnimSel, CompiledRules, NeighborReq, PALETTE_EXTENSION, RULETILE_EXTENSION, RingSpec, RuleHit, RuleOutput, RuleTile, RuleTransform, TILE_EXTENSION, TileAnimation,
    TileAsset, TileColliderType, TileFile, TilePalette, TileSel, TilingRule, cell_hash, load_palette, neighbor_mask, refresh_cells, ring_spec, save_palette, value_noise,
};
pub use tile_paint::{
    Brush, CellEdit, Selection as TileSelection, StrokeRecord, TileClipboard, TileTool, apply_edits, box_cells, box_fill, erase_cells, flood_fill, flood_fill_cells,
    line_brush, line_cells, move_cells, paint_brush, paint_brush_mode, pick_brush, random_pick, rotate_brush_cw, tool_for_key, transform_cells, box_fill_random,
};
pub use sprite2d::{
    DEFAULT_SORTING_LAYER, MaskInteraction2D, SortingGroup, SortingLayer, SortingLayerId, SortingLayers, SpriteDrawMode, SpriteMask,
    SpriteMaskRange, SpriteRenderer, SpriteSortPoint, SpriteTileMode, split_sprite_ref,
};
pub use sprite_shape::{
    AngleRange, EdgeFit, SPRITE_SHAPE_EXTENSION, ShapePoint, ShapeTangentMode, SpriteShapeProfile, SpriteShapeRenderer, Tessellation, load_spriteshape, save_spriteshape,
    select_range,
};
pub use sprite_rig::{
    IkSolver2D, IkSolverKind, SPRITE_SWAP_HASH_MASK, SpriteBone, SpriteLibraryRef, SpriteResolver, SpriteSkin, sprite_swap_hash,
};
pub use physics_material::{CombineMode, PHYSMAT_EXTENSION, PhysicsMaterial, PhysicsSettings, load_physmat, save_physmat};
pub use water_sim::{RippleSim, WaterPush};
pub use water_ewave::{EwaveLayer, EwaveParams, SURFACE_TENSION, dispersive_share, ewave_omega, ewave_supported};
pub use water_pipe::{
    CELL_OBSTACLE, CELL_SHORE, CELL_WATER, MAX_SIM_SUBSTEPS, RUNUP_HEIGHT, PipeParams, PipeSim, SimField, SimWindow, WaterDisplacer, WaterSimFeed, choose_window,
    max_sim_dt, pipe_substeps, sim_resolution,
};
pub use water::{
    Buoyancy, FLUID_PRESETS, FluidPreset, GerstnerWave, OutlinePreset, ResolvedWave, RiverPoint, RiverSection, SHORE_NO_GROUND, ShoreDepth, SurfaceReflection, UnderwaterSettings, Water,
    WaterBody, WaterColumn, WaterLook, WaterRuntime, shore_damping,
};
pub use voice::{
    LipSync, LipSyncMode, LipSyncRuntime, VISEMES, VisemeFrame, VoiceChat, VoiceRange, VoiceRolloff, VoiceTransmit,
    viseme_index,
};
pub use mesh_renderer::{MaterialSlot, MeshRenderer, RendererData, RendererRuntime, SkinnedMeshRenderer, MESH_RENDERER, SKINNED_MESH_RENDERER, is_renderer_kind, renderer_of, renderer_of_mut};
pub use face::{
    BlendShapes, BlendShapesRuntime, EXPRESSIONS, normalize, FaceFrame, FaceSource, TrackingInputSettings, FaceTracking, FaceTrackingRuntime, ShapeSource,
    expression_index,
};
pub use cloth::{
    BoneCloth, BoneClothRuntime, BoneClothSim, ClothWind, ClothWindParams, WindSample, WindShape, ClothCollider, ClothColliderRuntime, ClothColliders, ClothConnection,
    ClothFit, ClothFitPreset, ClothParticle, ClothShape, ClothShapeWorld, ClothTeleport, fit_capsule, max_scale,
    principal_axis, rigid_frame,
};
mod spatial_audio;
mod audio_volume;
pub mod volume;
pub mod transition;
pub use audio_volume::{AudioEnvironmentProfile, AudioVolume};
mod gameplay;
pub mod actions;
pub use actions::{
    ActionCtx, ActionDesc, ActionEvent, ActionType, ArgSource, ArgValue, EventActions, EventBinding, FireSource, InstanceSel,
    OBJECT_ACTIONS, OBJECT_COMPONENT, OBJECT_PROPERTIES, ParamDefault, ParamDesc, PropertyDesc, PropertyReader, fire_all, invoke_on, invoke_on_with,
    object_property, pick_instance,
};
mod ik;
mod input;
mod input_remap;
mod localization;
mod models;
mod savegame;
mod ui;
mod text3d;
pub use text3d::{Text3D, Text3DBillboard};
mod vr;
mod body;
pub use body::{
    AvatarScaling, BodyCommand, BodyFit, BodyInfo, BodyProfile, DEFAULT_EYE_HEIGHT, EYE_TO_CROWN, FIRST_BODY_SLOT,
    HEIGHT_CAPTURE_SECONDS, HeightSampler, StoredPose, TrackerRole, arm_reach, arm_span, assign_roles, fit_avatar,
    robust_mean, role_name,
};
pub use ik::{
    HumanoidIk, HumanoidIkRuntime, IkSource, IkTargets, TrackingLocomotion, TRACKER_ROLES, TRACKER_SLOTS, TrackerCalibration, TrackerTargets, TwoBoneSolution, solve_fabrik,
    solve_two_bone,
};
pub use audio_mixer::AudioMixer;
pub use spatial_audio::{
    AcousticMaterial, AcousticPreset, AudioProbeVolume, OutputLayout, RayTracerKind, ReflectionMode, SpatialAudioSettings,
};
pub use localization::Localization;
pub use savegame::{SaveGame, SaveValue};
pub mod material;
pub use material::*;
pub use gameplay::{AnyBox, ComponentLookup, Deferred, MaterialEdit, MaterialQuery, NavQuery, PhysicsQuery, RayFilter, RayHit, ScreenInfo};
pub use ui::{
    UiAction, UiAlign, UiButton, UiButtonState, UiCanvas, UiFillDirection, UiImage, UiPanel, UiPauseMenu,
    UiProgressBar, UiRect, UiRenderMode, UiScaleMode, UiSlider, UiText, UiTextField, UiTextOverflow, UiTextRenderer,
    UiToggle,
};
pub use vr::{Ray, VrFrame, VrRig, pointer_ray, ray_plane};
mod net;
pub use net::{
    HOST_PEER, InterpolationMode, NetAutoStart, NetEvent, NetObjectState, NetQuality, NetRpc, NetSpawnSource, NetValue,
    NetVar, NetView, NetWorldOp, NetworkAnimator, NetworkAvatar, NetworkManager, NetworkObject, NetworkVariables,
    OwnershipMode, PeerInfo, RpcTarget, WORN_AVATAR_VAR, WornAvatar, AVATAR_SCALE_VAR, PLAYER_EYE_HEIGHT_VAR,
};
mod anticheat;
pub use anticheat::{
    AntiCheatConfig, AntiCheatProvider, ClientStatus, MAX_PAYLOAD as ANTICHEAT_MAX_PAYLOAD, NullProvider, PROVIDER_EOS, PROVIDER_NONE,
    TEST_CHALLENGE, TEST_RESPONSE, TestPlan, TestProvider, TestState, UnavailableProvider, Verdict, create_provider,
};
mod net_interest;
mod net_rewind;
mod net_validation;
pub mod net_role;
pub mod mesh_renderer;
mod persist;
pub use persist::PersistentState;
pub use net_role::{NetRole, builtin_default_role, process_role, set_process_role};
pub use net_validation::{HARD_MAX_REWIND, NetValidation, RateLimits, RateSetting, RpcPermission, RpcRule, SpawnQuota, StrikePolicy, Topology};
pub use net_interest::{Change as InterestChange, ClientInterest, InterestConfig, RateTier, SpatialGrid};
pub use net_rewind::{HitHistory, HitPose, HitShape, MAX_REWIND, RewindHit, RewindReject, check_rewind_time};
mod kit;
pub use kit::*;
mod motion;
pub use motion::{MotionState, MoveHost, Predictable, as_predictable, driven_remotely, is_predictable};
mod content;
pub use content::{
    AvatarDescriptor, ContentCommand, ContentEvent, ContentInfo, ContentKind, ContentPlacement, ContentRequest,
};
pub mod script;
pub use script::Script;
pub mod wasm;
pub use wasm::WasmScript;
pub use input::{
    ActionBinding, ActionKind, ActionValue, Bindings, ControlScheme, InputSource, InputState, Key,
    MouseAxis, MouseButton, PadAxis, PadButton, RawInput, VrAxis, VrButton, resolve,
};
pub use input_remap::{
    ActionOverride, ActionTuning, BindingConflict, BindingOverrides, BindingSlot, CaptureKind, CaptureRequest,
    CaptureResult, ConflictPolicy, ControlsInfo, ControlsRuntime, InputCapture, InputCommand, InputDevice, RebindError,
    RemapEntry, active_sources, all_sources, device_activity, stick_name,
};
pub use models::{
    AlphaModeModel, BlendModeModel, DETAIL_SLOT_BASE, DetailModel, MatcapBlend, MaterialModel, MaterialTexturePaths, TessellationModel, TransmissionModel, ShaderCmpUi, ShaderCondUi,
    ShaderGroupUi, ShaderPresetUi, ShaderPropKindUi, ShaderPropUi, ShaderUiInfo, StencilCompareModel, StencilFaceModel,
    StencilModel, StencilOpModel, TEXTURE_SLOT_LABELS,
};

// ------------------------------------------------------------------ object id

/// A stable, globally-unique object identity (UUID v4). Assigned when an object
/// is created (editor spawn / import / scene load) and serialized in `.scene`,
/// so cross-references survive reloads, reordering, and networking. Names stay
/// purely cosmetic. Serialized as the canonical hyphenated string.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(into = "String", try_from = "String")]
pub struct ObjectId(u128);

impl ObjectId {
    /// The "unassigned" id (all-zero). Used as a placeholder until a real id is
    /// generated (e.g. loading a legacy scene without ids).
    pub const NIL: ObjectId = ObjectId(0);

    /// Generate a fresh random v4 UUID.
    pub fn new() -> Self {
        let mut b = [0u8; 16];
        getrandom::fill(&mut b).expect("getrandom for object id");
        b[6] = (b[6] & 0x0f) | 0x40; // version 4
        b[8] = (b[8] & 0x3f) | 0x80; // RFC 4122 variant
        ObjectId(u128::from_be_bytes(b))
    }

    /// Whether this is the unassigned [`ObjectId::NIL`] placeholder (no real
    /// id has been generated or resolved yet).
    pub fn is_nil(&self) -> bool {
        self.0 == 0
    }

    /// Raw 128-bit value for wire encoding (networking).
    pub fn raw(&self) -> u128 {
        self.0
    }

    /// Reconstruct from a raw 128-bit value (networking).
    pub fn from_raw(v: u128) -> Self {
        ObjectId(v)
    }
}

impl Default for ObjectId {
    fn default() -> Self {
        ObjectId::NIL
    }
}

impl std::fmt::Display for ObjectId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let b = self.0.to_be_bytes();
        write!(
            f,
            "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13],
            b[14], b[15]
        )
    }
}

impl std::fmt::Debug for ObjectId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ObjectId({self})")
    }
}

impl std::str::FromStr for ObjectId {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let hex: String = s.chars().filter(|c| *c != '-').collect();
        if hex.len() != 32 {
            return Err(format!("invalid object id {s:?}"));
        }
        u128::from_str_radix(&hex, 16)
            .map(ObjectId)
            .map_err(|e| format!("invalid object id {s:?}: {e}"))
    }
}

impl From<ObjectId> for String {
    fn from(id: ObjectId) -> String {
        id.to_string()
    }
}

impl TryFrom<String> for ObjectId {
    type Error = String;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        s.parse()
    }
}

/// A reference to another object, settable in the editor and resolved at
/// runtime through [`ComponentCtx`]. `None` = unset. Components store this
/// instead of an index or a name.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug, Serialize, Deserialize)]
pub struct ObjectRef(pub Option<ObjectId>);

impl ObjectRef {
    /// An unset reference (no target object).
    pub const NONE: ObjectRef = ObjectRef(None);
    /// A reference to the object with this id.
    pub fn new(id: ObjectId) -> Self {
        ObjectRef(Some(id))
    }
    /// The referenced object's id, or `None` when unset.
    pub fn id(&self) -> Option<ObjectId> {
        self.0
    }
    /// Whether a target object is set.
    pub fn is_set(&self) -> bool {
        self.0.is_some()
    }
}

impl From<ObjectId> for ObjectRef {
    fn from(id: ObjectId) -> Self {
        ObjectRef(Some(id))
    }
}

// ------------------------------------------------------------------ transform

/// A decomposed transform (translation / rotation / scale). The gameplay-side
/// view of any object. The in-game API resolves object references into this.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    /// Position in meters (world space when read through [`ComponentCtx::object_transform`]).
    pub translation: Vec3,
    /// Orientation as a unit quaternion (world space when read through the
    /// context; local space when held by a component).
    pub rotation: Quat,
    /// Per-axis scale multiplier (1 = unscaled).
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl Transform {
    /// Decompose a matrix into translation / rotation / scale (TRS order:
    /// scale, then rotation, then translation).
    pub fn from_matrix(m: Mat4) -> Self {
        let (scale, rotation, translation) = m.to_scale_rotation_translation();
        Self {
            translation,
            rotation,
            scale,
        }
    }
    /// Recompose the transform into a `Mat4` (TRS order).
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
    /// Local -Z in world space (the "looking" direction).
    pub fn forward(&self) -> Vec3 {
        self.rotation * Vec3::NEG_Z
    }
    /// Local +X rotated into the transform's frame.
    pub fn right(&self) -> Vec3 {
        self.rotation * Vec3::X
    }
    /// Local +Y rotated into the transform's frame.
    pub fn up(&self) -> Vec3 {
        self.rotation * Vec3::Y
    }
}

// -------------------------------------------------------------- in-game ctx

/// Deferred engine request a component can make from a lifecycle hook. The
/// engine applies these after the update pass finishes (so the scene isn't
/// Deferred world mutations from components (applied after the update pass;
/// the pass owns every component mutably mid-iteration).
#[derive(Debug)]
pub enum ComponentCommand {
    /// Switch to another scene (level change, menu -> game). Path is
    /// project-relative, e.g. "scenes/level2.scene". Replaces the current scene.
    LoadScene(String),
    /// Start loading a scene in the background; a later `LoadScene` /
    /// `LoadSceneAdditive` of it switches instantly once ready.
    PreloadScene(String),
    /// Add another scene's objects to the current one (streaming rooms).
    LoadSceneAdditive(String),
    /// Remove the objects an additive load brought in.
    UnloadScene(String),
    /// Carry an object (and its children) into the next scene on `LoadScene`.
    KeepOnSceneChange(ObjectId, bool),
    /// Make this camera object the active render camera (2A).
    SetActiveCamera(ObjectId),
    /// Write another object's LOCAL transform (each field optional). Lets a pawn
    /// drive a child camera (pitch / spring-arm) without owning its component.
    SetLocalTransform {
        /// Target object.
        id: ObjectId,
        /// New local position in meters (`None` = keep).
        translation: Option<Vec3>,
        /// New local rotation (`None` = keep).
        rotation: Option<Quat>,
        /// New local scale (`None` = keep).
        scale: Option<Vec3>,
    },
    /// Teleport: write another object's WORLD position/rotation (each field
    /// optional). The engine converts to local using the object's parent, so
    /// this is correct whether the target is a root or nested. Local-space
    /// writes use `SetLocalTransform`.
    SetWorldTransform {
        /// Target object.
        id: ObjectId,
        /// New world position in meters (`None` = keep).
        translation: Option<Vec3>,
        /// New world rotation (`None` = keep).
        rotation: Option<Quat>,
    },
    /// Deferred edit of one material slot's params: the engine applies the
    /// closure to the live params, writes the result back into the scene
    /// material, and refreshes the renderer's copy after the pass.
    ModifyMaterialParams {
        /// Object whose material is edited.
        id: ObjectId,
        /// Render slot index (0 = the main material).
        slot: usize,
        /// Closure applied to the live [`MaterialParams`].
        edit: MaterialEdit,
    },
    /// Assign a `.material` file (project-relative) to one of an object's
    /// render slots (slot 0 = the main material).
    AssignMaterial {
        /// Object whose render slot is reassigned.
        id: ObjectId,
        /// Render slot index (0 = the main material).
        slot: usize,
        /// Project-relative `.material` file.
        path: String,
    },
    /// Local peer requests authority over a networked object (2G). Granted +
    /// broadcast by the engine's network system; the owner replicates its
    /// transform to everyone else.
    RequestOwnership(ObjectId),
    /// Local peer relinquishes authority over a networked object, so another peer
    /// can claim it.
    ReleaseOwnership(ObjectId),
    /// Change the window/render resolution at runtime (graphics settings).
    SetResolution(u32, u32),
    /// Toggle vsync at runtime.
    SetVsync(bool),
    /// Set shadow-map resolution (256..=8192) at runtime.
    SetShadowResolution(u32),
    /// Send a networked text message (None target = broadcast / public).
    NetMessage {
        /// Recipient peer id (`None` = broadcast / public).
        to: Option<u64>,
        /// Message text.
        text: String,
    },
    /// Hand a networked object to another peer (owner or host only).
    TransferOwnership(ObjectId, u64),
    /// Spawn on every peer: applied locally right away, then replicated.
    NetSpawn {
        /// Where the spawn came from (local request, replication, scene streaming).
        source: NetSpawnSource,
        /// Id the spawned root gets, chosen by the caller so it can be
        /// referenced immediately.
        root_id: ObjectId,
        /// World position to spawn at, in meters.
        translation: Vec3,
        /// World rotation to spawn at.
        rotation: Quat,
        /// Owner peer (0 = host authority).
        owner: u64,
    },
    /// Despawn a networked object on every peer.
    NetDespawn(ObjectId),
    /// Write a synced variable (applied locally, then replicated).
    NetSetVar {
        /// Networked object whose variable is written.
        id: ObjectId,
        /// Variable name.
        key: String,
        /// New value.
        value: NetValue,
    },
    /// Send an RPC / network event.
    NetRpc {
        /// Target object (`None` = session-wide RPC).
        object: Option<ObjectId>,
        /// RPC / event name.
        name: String,
        /// Serialized arguments.
        args: Vec<NetValue>,
        /// Who receives it (see [`RpcTarget`]).
        target: RpcTarget,
        /// Guaranteed delivery (retransmitted); `false` = fire-and-forget.
        reliable: bool,
    },
    /// Start hosting / join / leave a session.
    NetHost {
        /// Port to listen on.
        port: u16,
    },
    /// Join the session hosted at `address` (ip or hostname, optionally `:port`).
    NetJoin {
        /// Host address (ip or hostname, optionally `:port`).
        address: String,
    },
    /// Leave the current session.
    NetDisconnect,
    /// Host: remove a player from the session (optionally banning its id).
    NetKick {
        /// Peer to remove.
        peer: u64,
        /// Reason shown to the removed player.
        reason: String,
        /// Also ban the peer so it cannot rejoin.
        ban: bool,
    },
    /// Host-driven scene switch for every peer.
    NetLoadScene(String),
    /// Instantiate a `.prefab` (project-relative path). The first root gets
    /// `root_id` so the caller can reference it immediately.
    SpawnPrefab {
        /// Project-relative `.prefab` path.
        path: String,
        /// Id the spawned root gets, so the caller can reference it immediately.
        root_id: ObjectId,
        /// World position to spawn at, in meters.
        translation: Vec3,
        /// World rotation to spawn at.
        rotation: Quat,
    },
    /// Clone an existing object (and its children) as a new root object.
    Instantiate {
        /// Object to clone.
        original: ObjectId,
        /// Id the clone's root gets.
        new_id: ObjectId,
        /// World position for the clone, in meters.
        translation: Vec3,
        /// World rotation for the clone.
        rotation: Quat,
    },
    /// Remove an object and its children from the scene.
    Destroy(ObjectId),
    /// Enable / disable an object (disabled objects don't update, render or collide).
    SetActive(ObjectId, bool),
    /// Run `f` against the first component of type `type_id` on object `id`.
    Modify {
        /// Object whose component is modified.
        id: ObjectId,
        /// The component type to find.
        type_id: std::any::TypeId,
        /// Closure applied to the component (downcast inside to your type).
        f: Deferred,
    },
    /// Broadcast a gameplay event, readable by every component next frame.
    Emit(AnyBox),
    /// Call a component's exposed action (see [`actions`]). `slot` is the
    /// component's cached position on the object; the name is re-checked.
    InvokeAction {
        /// Object carrying the component.
        id: ObjectId,
        /// Component's cached slot on the object.
        slot: u32,
        /// Component type name.
        component: &'static str,
        /// Action name (re-checked against the component).
        action: &'static str,
        /// Coerced argument values.
        args: Vec<ArgValue>,
        /// Object that triggered the action, when any.
        instigator: Option<ObjectId>,
    },
    /// Push a rigid body's center of mass with a continuous force
    /// (Newtons, world space).
    AddForce {
        /// Body object.
        id: ObjectId,
        /// Force vector in Newtons.
        force: Vec3,
    },
    /// Instantly change a rigid body's momentum (impulse = force x time,
    /// world space).
    AddImpulse {
        /// Body object.
        id: ObjectId,
        /// Impulse vector in Newton-seconds.
        impulse: Vec3,
    },
    /// Apply a rotational force around a rigid body's center of mass
    /// (world space).
    AddTorque {
        /// Body object.
        id: ObjectId,
        /// Torque vector.
        torque: Vec3,
    },
    /// Set a rigid body's linear velocity directly (world space).
    SetVelocity {
        /// Body object.
        id: ObjectId,
        /// Linear velocity in meters per second.
        linear: Vec3,
    },
    /// Re-bake every NavMesh Surface (after the level geometry changed).
    RebakeNavMesh,
    /// Change a joint's motor at runtime (deg/s or m/s; max force).
    SetJointMotor {
        /// Joint object.
        id: ObjectId,
        /// Target motor speed (degrees/second for hinges, meters/second for sliders).
        speed: f32,
        /// Maximum force the motor may apply.
        max_force: f32,
    },
    /// Move a [`CharacterController`] by `delta`, sliding along obstacles.
    MoveCharacter {
        /// Object carrying the [`CharacterController`].
        id: ObjectId,
        /// Movement delta in world space, meters.
        delta: Vec3,
    },
    /// Fire-and-forget sound. `position: None` plays it non-spatially.
    PlaySound {
        /// Project-relative audio clip.
        clip: String,
        /// World position the sound plays from (`None` = non-spatial).
        position: Option<Vec3>,
        /// Playback volume 0..1.
        volume: f32,
    },
    /// (Re)start the [`AudioSource`] on an object.
    PlayAudio(ObjectId),
    /// Stop the [`AudioSource`] on an object.
    StopAudio(ObjectId),
    /// Set a persistent save value (see [`SaveGame`]).
    SaveSet {
        /// Save key.
        key: String,
        /// Value to store.
        value: SaveValue,
    },
    /// Remove a key from the save data.
    SaveRemove(String),
    /// Write the save data to a slot file.
    SaveWrite(String),
    /// Replace the save data with a slot file's contents (next pass).
    SaveLoad(String),
    /// Switch the localization language.
    SetLanguage(String),
    /// Set an audio mixer bus's volume.
    SetBusVolume {
        /// Bus name (see the project Audio Mixer).
        bus: String,
        /// Volume 0..1.
        volume: f32,
    },
    /// Mute or unmute an audio mixer bus.
    SetBusMuted {
        /// Bus name.
        bus: String,
        /// `true` = mute.
        muted: bool,
    },
    /// Lock + hide the OS cursor (mouse-look) or release it.
    SetCursorLocked(bool),
    /// Scale game time (0 = paused, 1 = normal). Affects `dt` and physics.
    SetTimeScale(f32),
    /// Exit the game (stops Play mode in the editor).
    Quit,
    /// Capture full-body tracker calibration now (the player holds a T-pose).
    CalibrateTrackers,
    /// Drop the tracker calibration (back to the default offsets).
    ClearTrackerCalibration,
    /// Player height measurement, body calibration and the VR body menu.
    Body(BodyCommand),
    /// Start (`Some(port)`) or stop (`None`) the OSC face-tracking receiver.
    SetOscFaceTracking(Option<u16>),
    /// User-generated content request (load / spawn / enter world / unload).
    Content(ContentCommand),
    /// Rebind / reset / capture request for the live input bindings.
    Input(InputCommand),
}

/// Per-frame update context handed to components in Play mode. Transform
/// fields are the owning object's local TRS.
pub struct ComponentCtx<'a> {
    /// Frame delta in seconds, scaled by the time scale (fixed step in `fixed_update`).
    pub dt: f32,
    /// Frame delta ignoring the time scale (UI, pause menus).
    pub unscaled_dt: f32,
    /// Seconds since engine start.
    pub time: f32,
    /// The water clock: shared by every peer in a networked game, so waves
    /// line up on all clients.
    pub water_time: f32,
    /// The owning object's local position in meters. Write it directly for
    /// local-space movement; for a world-space value use
    /// [`ComponentCtx::set_world_position`], which converts through the
    /// object's parent.
    pub translation: &'a mut Vec3,
    /// The owning object's local rotation (unit quaternion). Writing here
    /// rotates the object in its parent's frame.
    pub rotation: &'a mut Quat,
    /// The owning object's local scale (1 = unscaled).
    pub scale: &'a mut Vec3,
    /// Deferred engine requests; drained and applied after the update pass.
    pub commands: &'a mut Vec<ComponentCommand>,
    /// Read-only world view: every object's world transform this frame
    /// (snapshot at the start of the update pass) for resolving object
    /// references. Parallel to `object_names`.
    pub world_transforms: &'a [Mat4],
    /// Object names, parallel to `world_transforms`.
    pub object_names: &'a [String],
    /// Stable object ids, parallel to `world_transforms`. The preferred way to
    /// reference objects (names are cosmetic and may collide or be empty).
    pub object_ids: &'a [ObjectId],
    /// Index of the object that owns the component being updated.
    pub self_index: usize,
    /// World matrix of the owning object's parent (None if it's a root). Used to
    /// convert a desired world position/transform back into the local TRS that
    /// the `translation`/`rotation`/`scale` fields hold.
    pub parent_world: Option<Mat4>,
    /// Resolved input action snapshot for this frame (2C). Read with
    /// `ctx.input.axis2("Move")`, `ctx.input.pressed("Jump")`, etc.
    pub input: &'a InputState,
    /// Networking view (2G): ownership + local peer. Empty when offline.
    pub net: &'a NetView,
    /// Spawn points this frame as `(object index, tag)`, from objects carrying a
    /// [`SpawnPoint`] component. Resolve a transform via `spawn_point(tag)`.
    pub spawn_points: &'a [(usize, String)],
    /// Id -> index map for `object_ids` (fast reference resolution).
    pub id_index: &'a HashMap<ObjectId, usize>,
    /// Whether each object is active (enabled with all parents enabled).
    pub active: &'a [bool],
    /// Other components in the scene, for [`ComponentCtx::get`].
    pub lookup: &'a dyn ComponentLookup,
    /// Physics queries (raycasts, velocities). `None` outside Play mode.
    pub physics: Option<&'a dyn PhysicsQuery>,
    /// Navmesh path queries. `None` outside Play mode or before the bake.
    pub nav: Option<&'a dyn NavQuery>,
    /// Material params of any object's render slots (slot 0 = the main
    /// material), for shader control: `ctx.material_params(...)` reads,
    /// `ctx.modify_material(...)` edits. `None` outside Play mode.
    pub materials: Option<&'a dyn MaterialQuery>,
    /// Events emitted last frame via [`ComponentCtx::emit`].
    pub events: &'a [AnyBox],
    /// Current time scale.
    pub time_scale: f32,
    /// Game view size, pointer, camera and UI clicks.
    pub screen: &'a ScreenInfo,
    /// Persistent save data (key-value), see `ctx.save_value`.
    pub save: &'a SaveGame,
    /// Localized string tables (`localization/*.loc`).
    pub strings: &'a Localization,
}

impl ComponentCtx<'_> {
    /// Queue a scene switch (level change, menu -> game). Project-relative
    /// path. Applied after the current update pass.
    pub fn load_scene(&mut self, path: impl Into<String>) {
        self.commands.push(ComponentCommand::LoadScene(path.into()));
    }

    /// Find an object by name; returns its index for the lookups below. Names
    /// aren't guaranteed unique, so the first match wins.
    pub fn find_object(&self, name: &str) -> Option<usize> {
        self.object_names.iter().position(|n| n == name)
    }

    /// World [`Transform`] (translation/rotation/scale) of an object: the
    /// gameplay view of any object reference (snapshot at frame start).
    pub fn object_transform(&self, index: usize) -> Option<Transform> {
        self.world_transforms
            .get(index)
            .map(|m| Transform::from_matrix(*m))
    }

    /// Raw world matrix of an object (when you need the matrix, not the
    /// decomposed transform).
    pub fn object_matrix(&self, index: usize) -> Option<Mat4> {
        self.world_transforms.get(index).copied()
    }

    /// World position of an object.
    pub fn object_position(&self, index: usize) -> Option<Vec3> {
        self.world_transforms.get(index).map(|m| m.w_axis.truncate())
    }

    /// Find an object by name and return its world [`Transform`] in one step.
    pub fn object_transform_named(&self, name: &str) -> Option<Transform> {
        self.find_object(name).and_then(|i| self.object_transform(i))
    }

    /// World [`Transform`] of the owning object.
    pub fn self_transform(&self) -> Transform {
        self.object_transform(self.self_index).unwrap_or_default()
    }

    /// Place the owning object at a world-space position, accounting for its
    /// parent. Writes the local translation (the field the engine reads), so
    /// this is correct whether the object is a root or nested under parents.
    /// Prefer this over assigning `*ctx.translation` directly when the value you
    /// have is a world coordinate (e.g. another object's position).
    pub fn set_world_position(&mut self, world: Vec3) {
        *self.translation = match self.parent_world {
            Some(p) => p.inverse().transform_point3(world),
            None => world,
        };
    }

    /// The owning object's stable id.
    pub fn self_id(&self) -> ObjectId {
        self.object_ids
            .get(self.self_index)
            .copied()
            .unwrap_or_default()
    }

    /// Resolve a stable id to its current index.
    pub fn index_of(&self, id: ObjectId) -> Option<usize> {
        self.id_index.get(&id).copied()
    }

    /// Resolve an [`ObjectRef`] to its index (None if unset or gone).
    pub fn resolve(&self, target: ObjectRef) -> Option<usize> {
        self.index_of(target.id()?)
    }

    /// World [`Transform`] of a referenced object. The preferred way to read
    /// another object you hold a reference to.
    pub fn transform_of(&self, target: ObjectRef) -> Option<Transform> {
        self.object_transform(self.resolve(target)?)
    }

    /// World position of a referenced object.
    pub fn position_of(&self, target: ObjectRef) -> Option<Vec3> {
        self.object_position(self.resolve(target)?)
    }

    // ---- camera possession (2A) ----

    /// Make a referenced camera object the active render camera (deferred).
    pub fn set_active_camera(&mut self, target: ObjectRef) {
        if let Some(id) = target.id() {
            self.commands.push(ComponentCommand::SetActiveCamera(id));
        }
    }

    /// Write another object's local transform (any subset of fields).
    pub fn set_local_transform(
        &mut self,
        target: ObjectRef,
        translation: Option<Vec3>,
        rotation: Option<Quat>,
        scale: Option<Vec3>,
    ) {
        if let Some(id) = target.id() {
            self.commands.push(ComponentCommand::SetLocalTransform {
                id,
                translation,
                rotation,
                scale,
            });
        }
    }

    /// Teleport a referenced object to a world-space position (respawns,
    /// portals, moving the player). The engine converts to local using the
    /// object's parent, so it is correct whether the target is a root or
    /// nested under parents.
    pub fn set_world_position_of(&mut self, target: ObjectRef, position: Vec3) {
        self.set_world_transform_of(target, Some(position), None);
    }

    /// Write a referenced object's world-space position and/or rotation (each
    /// field optional). The engine converts to the object's local TRS using
    /// its parent; use `set_local_transform` to write local values directly.
    pub fn set_world_transform_of(
        &mut self,
        target: ObjectRef,
        translation: Option<Vec3>,
        rotation: Option<Quat>,
    ) {
        if let Some(id) = target.id() {
            self.commands.push(ComponentCommand::SetWorldTransform {
                id,
                translation,
                rotation,
            });
        }
    }

    // ---- material / shader control ----

    /// The live editable params of a referenced object's material slot (slot
    /// 0 = the main material). `None` when the object or slot has no
    /// material. Edit them (deferred) with [`Self::modify_material`].
    pub fn material_params(&self, target: ObjectRef, slot: usize) -> Option<MaterialParams> {
        let i = self.resolve(target)?;
        self.materials.and_then(|m| m.material_params(i, slot))
    }

    /// Queue a deferred edit of a material slot's params. Applied after the
    /// update pass: the engine edits the live params, writes the result into
    /// the scene material and refreshes the renderer's copy. The shader
    /// control hook: flicker a light's emissive surface
    /// (`m.emission_intensity`), scroll UVs, pulse emission, tint a body,
    /// dial roughness - every field of [`MaterialParams`].
    pub fn modify_material(
        &mut self,
        target: ObjectRef,
        slot: usize,
        f: impl FnOnce(&mut MaterialParams) + 'static,
    ) {
        if let Some(id) = target.id() {
            self.commands.push(ComponentCommand::ModifyMaterialParams {
                id,
                slot,
                edit: MaterialEdit(Box::new(f)),
            });
        }
    }

    /// Assign a `.material` file (project-relative) to a render slot of a
    /// referenced object (slot 0 = the main material). Lets a plugin
    /// component carry a material path and apply it in play.
    pub fn assign_material(&mut self, target: ObjectRef, slot: usize, path: impl Into<String>) {
        if let Some(id) = target.id() {
            self.commands.push(ComponentCommand::AssignMaterial {
                id,
                slot,
                path: path.into(),
            });
        }
    }

    // ---- networking (2G) ----

    /// Whether the local peer currently has authority over the owning object.
    pub fn owns_self(&self) -> bool {
        self.net.owns(self.self_id())
    }

    /// Whether the local peer has authority over a referenced object.
    pub fn owns(&self, target: ObjectRef) -> bool {
        target.id().map(|id| self.net.owns(id)).unwrap_or(false)
    }

    /// Request authority over the owning object (e.g. on grab). The engine grants
    /// + broadcasts it; once owned, the object's transform replicates to peers.
    pub fn request_ownership(&mut self) {
        self.commands
            .push(ComponentCommand::RequestOwnership(self.self_id()));
    }

    /// Release authority over the owning object so another peer can claim it.
    pub fn release_ownership(&mut self) {
        self.commands
            .push(ComponentCommand::ReleaseOwnership(self.self_id()));
    }

    /// Request authority over a referenced networked object.
    pub fn request_ownership_of(&mut self, target: ObjectRef) {
        if let Some(id) = target.id() {
            self.commands.push(ComponentCommand::RequestOwnership(id));
        }
    }

    // ---- graphics settings (runtime) ----

    /// Change the game resolution at runtime.
    pub fn set_resolution(&mut self, width: u32, height: u32) {
        self.commands
            .push(ComponentCommand::SetResolution(width, height));
    }
    /// Toggle vsync at runtime.
    pub fn set_vsync(&mut self, on: bool) {
        self.commands.push(ComponentCommand::SetVsync(on));
    }
    /// Set shadow-map resolution at runtime (256..=8192).
    pub fn set_shadow_resolution(&mut self, res: u32) {
        self.commands
            .push(ComponentCommand::SetShadowResolution(res));
    }

    // ---- networked messaging (2G) ----

    /// Broadcast a public text message to all peers.
    pub fn broadcast(&mut self, text: impl Into<String>) {
        self.commands.push(ComponentCommand::NetMessage {
            to: None,
            text: text.into(),
        });
    }
    /// Send a private text message to one peer.
    pub fn send_to(&mut self, peer: u64, text: impl Into<String>) {
        self.commands.push(ComponentCommand::NetMessage {
            to: Some(peer),
            text: text.into(),
        });
    }
    /// Text messages received this frame: `(from_peer, is_private, text)`.
    pub fn messages(&self) -> &[(u64, bool, String)] {
        &self.net.messages
    }

    /// World [`Transform`] of the first spawn point with this tag (2, spawn
    /// points). `None` if no matching [`SpawnPoint`] exists.
    pub fn spawn_point(&self, tag: &str) -> Option<Transform> {
        let idx = self
            .spawn_points
            .iter()
            .find(|(_, t)| t == tag)
            .map(|(i, _)| *i)?;
        self.object_transform(idx)
    }
}

// ----------------------------------------------------------- component trait

/// Object-safe component, as stored on scene objects. Implement
/// [`TypedComponent`] instead of this directly. Runtime-only: editor inspector
/// UI and viewport gizmos are separate traits in `zestengine-editor`.
pub trait Component: 'static {
    /// The registered component type name (the `NAME` const of its
    /// [`TypedComponent`]), used for serialization and by-name lookups.
    fn type_name(&self) -> &'static str;
    /// Called once, the first frame the object is active in Play mode.
    fn start(&mut self, ctx: &mut ComponentCtx);
    /// Per-frame behaviour (Play mode only).
    fn update(&mut self, ctx: &mut ComponentCtx);
    /// Runs after every component finished `update` this frame.
    fn late_update(&mut self, ctx: &mut ComponentCtx);
    /// Runs once per fixed physics step, before the step.
    fn fixed_update(&mut self, ctx: &mut ComponentCtx);
    /// A solid collider on this object started touching `other` (3D physics).
    fn on_collision_enter(&mut self, ctx: &mut ComponentCtx, other: ObjectId);
    /// A solid collider on this object stopped touching `other` (3D physics).
    fn on_collision_exit(&mut self, ctx: &mut ComponentCtx, other: ObjectId);
    /// `other` entered a trigger on this object, or this object entered
    /// `other`'s trigger (3D physics).
    fn on_trigger_enter(&mut self, ctx: &mut ComponentCtx, other: ObjectId);
    /// `other` left a trigger on this object, or this object left `other`'s
    /// trigger (3D physics).
    fn on_trigger_exit(&mut self, ctx: &mut ComponentCtx, other: ObjectId);
    /// An `Animator` on this object passed an animation event marker; `name`
    /// is the marker.
    fn on_animation_event(&mut self, ctx: &mut ComponentCtx, name: &str);
    /// One call per event emitted last frame, delivered only for the types
    /// this component subscribed to (see [`Component::subscribed_events`]),
    /// before `update`. Downcast the [`std::any::Any`] to your event type.
    fn on_event(&mut self, ctx: &mut ComponentCtx, event: &dyn std::any::Any);
    /// Declare this component's event subscriptions (pub/sub routing): push
    /// `std::any::TypeId::of::<E>()` into `sink` for each event type `E` to
    /// receive in [`Component::on_event`]. The engine collects this once per
    /// frame; unsubscribed events never reach the component.
    fn subscribed_events(&self, sink: &mut dyn FnMut(std::any::TypeId));
    /// 2D physics contacts. A 2D solid collider on this object started touching
    /// `other`. The defaults forward enter / exit to the plain callbacks (so a
    /// component written for 3D reacts to 2D too); override these to tell them
    /// apart. Stay fires once per frame while the contact lasts.
    fn on_collision_enter_2d(&mut self, ctx: &mut ComponentCtx, other: ObjectId) {
        self.on_collision_enter(ctx, other)
    }
    /// 2D physics: fires once per frame while a 2D contact lasts.
    fn on_collision_stay_2d(&mut self, _ctx: &mut ComponentCtx, _other: ObjectId) {}
    /// 2D physics: a 2D solid collider stopped touching `other`; defaults to
    /// [`Component::on_collision_exit`].
    fn on_collision_exit_2d(&mut self, ctx: &mut ComponentCtx, other: ObjectId) {
        self.on_collision_exit(ctx, other)
    }
    /// 2D physics: like [`Component::on_trigger_enter`] for the 2D world.
    fn on_trigger_enter_2d(&mut self, ctx: &mut ComponentCtx, other: ObjectId) {
        self.on_trigger_enter(ctx, other)
    }
    /// 2D physics: fires once per frame while a 2D trigger contact lasts.
    fn on_trigger_stay_2d(&mut self, _ctx: &mut ComponentCtx, _other: ObjectId) {}
    /// 2D physics: like [`Component::on_trigger_exit`] for the 2D world.
    fn on_trigger_exit_2d(&mut self, ctx: &mut ComponentCtx, other: ObjectId) {
        self.on_trigger_exit(ctx, other)
    }
    /// Serialize to RON.
    fn save(&self) -> String;
    /// Downcasting for engine + editor systems that read typed component data.
    fn as_any(&self) -> &dyn std::any::Any;
    /// Mutable downcasting (assigning ids, editor inspector/gizmo dispatch).
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
    /// The component as a movement controller networked prediction can
    /// rewind and replay, if it is one.
    fn predictable(&self) -> Option<&dyn Predictable> {
        None
    }
    /// Mutable version of [`Component::predictable`].
    fn predictable_mut(&mut self) -> Option<&mut dyn Predictable> {
        None
    }
    /// Functions this component exposes to event bindings (see [`actions`]).
    fn actions(&self) -> &'static [ActionDesc] {
        &[]
    }
    /// Run one of [`Component::actions`]; `args` are already coerced to the
    /// declared parameter types and as long as the parameter list.
    fn invoke_action(&mut self, _ctx: &mut ActionCtx, name: &str, _args: &[ArgValue]) -> Result<(), String> {
        Err(actions::unknown_action(self.type_name(), name))
    }
    /// Values other bindings may read from this component.
    fn properties(&self) -> &'static [PropertyDesc] {
        &[]
    }
    /// Read one of the values declared in [`Component::properties`] by name.
    fn read_property(&self, _name: &str) -> Option<ArgValue> {
        None
    }
    /// Short text telling several components of one type on an object apart
    /// (a Wasm Script's module name); bindings store it next to the index.
    fn instance_label(&self) -> Option<String> {
        None
    }
}

/// A component switched off with the inspector's checkbox (edit mode): it
/// keeps its place in the list, its settings and its saved data, but none of
/// its callbacks run and systems that look components up by type don't see
/// it. Turning it back on unwraps it.
pub struct Disabled(pub Box<dyn Component>);

impl Component for Disabled {
    fn type_name(&self) -> &'static str {
        self.0.type_name()
    }
    fn start(&mut self, _ctx: &mut ComponentCtx) {}
    fn update(&mut self, _ctx: &mut ComponentCtx) {}
    fn late_update(&mut self, _ctx: &mut ComponentCtx) {}
    fn fixed_update(&mut self, _ctx: &mut ComponentCtx) {}
    fn on_collision_enter(&mut self, _ctx: &mut ComponentCtx, _other: ObjectId) {}
    fn on_collision_exit(&mut self, _ctx: &mut ComponentCtx, _other: ObjectId) {}
    fn on_trigger_enter(&mut self, _ctx: &mut ComponentCtx, _other: ObjectId) {}
    fn on_trigger_exit(&mut self, _ctx: &mut ComponentCtx, _other: ObjectId) {}
    fn on_animation_event(&mut self, _ctx: &mut ComponentCtx, _name: &str) {}
    fn on_event(&mut self, _ctx: &mut ComponentCtx, _event: &dyn std::any::Any) {}
    fn subscribed_events(&self, _sink: &mut dyn FnMut(std::any::TypeId)) {}
    fn save(&self) -> String {
        self.0.save()
    }
    fn actions(&self) -> &'static [ActionDesc] {
        self.0.actions()
    }
    fn invoke_action(&mut self, _ctx: &mut ActionCtx, _name: &str, _args: &[ArgValue]) -> Result<(), String> {
        Err(format!("{} is disabled", self.0.type_name()))
    }
    fn properties(&self) -> &'static [PropertyDesc] {
        self.0.properties()
    }
    fn read_property(&self, name: &str) -> Option<ArgValue> {
        self.0.read_property(name)
    }
    fn instance_label(&self) -> Option<String> {
        self.0.instance_label()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Whether `c` is switched on (not wrapped in [`Disabled`]).
pub fn component_enabled(c: &dyn Component) -> bool {
    !c.as_any().is::<Disabled>()
}

/// The component itself, whether or not it is switched off.
pub fn component_inner(c: &dyn Component) -> &dyn Component {
    match c.as_any().downcast_ref::<Disabled>() {
        Some(d) => d.0.as_ref(),
        None => c,
    }
}

/// The component itself, mutably, whether or not it is switched off.
pub fn component_inner_mut(c: &mut Box<dyn Component>) -> &mut dyn Component {
    if c.as_any().is::<Disabled>() {
        let d = c.as_any_mut().downcast_mut::<Disabled>().expect("checked above");
        return d.0.as_mut();
    }
    c.as_mut()
}

/// Switch the component in `slot` on or off in place; true when it changed.
pub fn set_component_enabled(slot: &mut Box<dyn Component>, on: bool) -> bool {
    if component_enabled(slot.as_ref()) == on {
        return false;
    }
    if on {
        let d = slot.as_any_mut().downcast_mut::<Disabled>().expect("checked above");
        let inner = std::mem::replace(&mut d.0, Box::new(Placeholder));
        *slot = inner;
    } else {
        let taken = std::mem::replace(slot, Box::new(Placeholder));
        *slot = Box::new(Disabled(taken));
    }
    true
}

/// Stands in for a slot's component for the instant it is being swapped.
#[derive(Serialize, Deserialize, Default)]
struct Placeholder;

impl TypedComponent for Placeholder {
    const NAME: &'static str = "Placeholder";
}

/// Typed component interface; blanket-implements [`Component`]. All lifecycle
/// hooks default to no-ops — override the ones you need. Inspector UI lives in
/// `zestengine-editor`'s `Inspect` trait, not here.
pub trait TypedComponent: Serialize + DeserializeOwned + Default + 'static {
    const NAME: &'static str;
    /// Previous serialized names, accepted on load but hidden in Add menus.
    const ALIASES: &'static [&'static str] = &[];
    /// Where this type exists by default (NET-3): `Client` types are stripped
    /// from a dedicated server, `Server` types from players. A scene can
    /// override it per component. The built-ins are classified in
    /// [`net_role::builtin_default_role`] (used while this stays `Both`).
    const NET_ROLE: NetRole = NetRole::Both;
    /// Called once, the first frame the object is active in Play mode
    /// (Play start, or later for spawned / re-enabled objects).
    fn start(&mut self, _ctx: &mut ComponentCtx) {}
    /// Called every frame while playing.
    fn update(&mut self, _ctx: &mut ComponentCtx) {}
    /// Called every frame after all components ran `update`.
    fn late_update(&mut self, _ctx: &mut ComponentCtx) {}
    /// Called once per fixed physics step (60 Hz), before the step. Apply
    /// forces here; `ctx.dt` is the fixed step.
    fn fixed_update(&mut self, _ctx: &mut ComponentCtx) {}
    /// A solid collider on this object started touching `other`.
    fn on_collision_enter(&mut self, _ctx: &mut ComponentCtx, _other: ObjectId) {}
    /// A solid collider on this object stopped touching `other`.
    fn on_collision_exit(&mut self, _ctx: &mut ComponentCtx, _other: ObjectId) {}
    /// `other` entered a trigger on this object (or this object entered `other`'s trigger).
    fn on_trigger_enter(&mut self, _ctx: &mut ComponentCtx, _other: ObjectId) {}
    fn on_trigger_exit(&mut self, _ctx: &mut ComponentCtx, _other: ObjectId) {}
    /// An `Animator` on this object passed an animation event marker.
    fn on_animation_event(&mut self, _ctx: &mut ComponentCtx, _name: &str) {}
    /// Declare the event types this component subscribes to (pub/sub routing):
    /// push `std::any::TypeId::of::<MyEvent>()` into `sink` for each. The
    /// engine collects this once per frame and delivers ONLY those event
    /// types to [`TypedComponent::on_event`] - everything else never reaches
    /// the component. The default subscribes to nothing, which keeps
    /// `on_event` out of the dispatch entirely.
    fn subscribed_events(&self, _sink: &mut dyn FnMut(std::any::TypeId)) {}
    /// One call per subscribed event emitted last frame (see
    /// [`ComponentCtx::emit`]), before `update`. Downcast to your event type:
    /// `let Some(e) = event.downcast_ref::<Hit>() else { return };`.
    fn on_event(&mut self, _ctx: &mut ComponentCtx, _event: &dyn std::any::Any) {}
    /// 2D physics: like the plain callbacks but only for contacts in the 2D
    /// world. The enter / exit defaults call the plain callback, so a component
    /// that only implements those reacts to 2D too.
    fn on_collision_enter_2d(&mut self, ctx: &mut ComponentCtx, other: ObjectId) {
        TypedComponent::on_collision_enter(self, ctx, other)
    }
    /// Once per frame while a 2D contact lasts.
    fn on_collision_stay_2d(&mut self, _ctx: &mut ComponentCtx, _other: ObjectId) {}
    fn on_collision_exit_2d(&mut self, ctx: &mut ComponentCtx, other: ObjectId) {
        TypedComponent::on_collision_exit(self, ctx, other)
    }
    fn on_trigger_enter_2d(&mut self, ctx: &mut ComponentCtx, other: ObjectId) {
        TypedComponent::on_trigger_enter(self, ctx, other)
    }
    fn on_trigger_stay_2d(&mut self, _ctx: &mut ComponentCtx, _other: ObjectId) {}
    fn on_trigger_exit_2d(&mut self, ctx: &mut ComponentCtx, other: ObjectId) {
        TypedComponent::on_trigger_exit(self, ctx, other)
    }
    /// A movement component that implements [`Predictable`] returns
    /// `Some(self)` from this and [`Self::predictable_mut`]: an owner-predicted
    /// `NetworkObject` carrying it then moves at once on its owner and is
    /// reconciled against the host, like the built-in controllers.
    fn predictable(&self) -> Option<&dyn Predictable> {
        None
    }
    fn predictable_mut(&mut self) -> Option<&mut dyn Predictable> {
        None
    }
    /// Functions exposed to event bindings; declare them as a `static` slice.
    fn actions(&self) -> &'static [ActionDesc] {
        &[]
    }
    /// Run an action. `args` match the declared parameters in count and type.
    fn invoke_action(&mut self, _ctx: &mut ActionCtx, name: &str, _args: &[ArgValue]) -> Result<(), String> {
        Err(actions::unknown_action(Self::NAME, name))
    }
    /// Values other bindings may read from this component.
    fn properties(&self) -> &'static [PropertyDesc] {
        &[]
    }
    fn read_property(&self, _name: &str) -> Option<ArgValue> {
        None
    }
    /// See [`Component::instance_label`].
    fn instance_label(&self) -> Option<String> {
        None
    }
}

impl<T: TypedComponent> Component for T {
    fn type_name(&self) -> &'static str {
        T::NAME
    }
    fn start(&mut self, ctx: &mut ComponentCtx) {
        TypedComponent::start(self, ctx)
    }
    fn update(&mut self, ctx: &mut ComponentCtx) {
        TypedComponent::update(self, ctx)
    }
    fn late_update(&mut self, ctx: &mut ComponentCtx) {
        TypedComponent::late_update(self, ctx)
    }
    fn fixed_update(&mut self, ctx: &mut ComponentCtx) {
        TypedComponent::fixed_update(self, ctx)
    }
    fn on_collision_enter(&mut self, ctx: &mut ComponentCtx, other: ObjectId) {
        TypedComponent::on_collision_enter(self, ctx, other)
    }
    fn on_collision_exit(&mut self, ctx: &mut ComponentCtx, other: ObjectId) {
        TypedComponent::on_collision_exit(self, ctx, other)
    }
    fn on_trigger_enter(&mut self, ctx: &mut ComponentCtx, other: ObjectId) {
        TypedComponent::on_trigger_enter(self, ctx, other)
    }
    fn on_trigger_exit(&mut self, ctx: &mut ComponentCtx, other: ObjectId) {
        TypedComponent::on_trigger_exit(self, ctx, other)
    }
    fn on_animation_event(&mut self, ctx: &mut ComponentCtx, name: &str) {
        TypedComponent::on_animation_event(self, ctx, name)
    }
    fn on_event(&mut self, ctx: &mut ComponentCtx, event: &dyn std::any::Any) {
        TypedComponent::on_event(self, ctx, event)
    }
    fn subscribed_events(&self, sink: &mut dyn FnMut(std::any::TypeId)) {
        TypedComponent::subscribed_events(self, sink)
    }
    fn on_collision_enter_2d(&mut self, ctx: &mut ComponentCtx, other: ObjectId) {
        TypedComponent::on_collision_enter_2d(self, ctx, other)
    }
    fn on_collision_stay_2d(&mut self, ctx: &mut ComponentCtx, other: ObjectId) {
        TypedComponent::on_collision_stay_2d(self, ctx, other)
    }
    fn on_collision_exit_2d(&mut self, ctx: &mut ComponentCtx, other: ObjectId) {
        TypedComponent::on_collision_exit_2d(self, ctx, other)
    }
    fn on_trigger_enter_2d(&mut self, ctx: &mut ComponentCtx, other: ObjectId) {
        TypedComponent::on_trigger_enter_2d(self, ctx, other)
    }
    fn on_trigger_stay_2d(&mut self, ctx: &mut ComponentCtx, other: ObjectId) {
        TypedComponent::on_trigger_stay_2d(self, ctx, other)
    }
    fn on_trigger_exit_2d(&mut self, ctx: &mut ComponentCtx, other: ObjectId) {
        TypedComponent::on_trigger_exit_2d(self, ctx, other)
    }
    fn save(&self) -> String {
        ron::to_string(self).unwrap_or_default()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn predictable(&self) -> Option<&dyn Predictable> {
        TypedComponent::predictable(self)
    }
    fn predictable_mut(&mut self) -> Option<&mut dyn Predictable> {
        TypedComponent::predictable_mut(self)
    }
    fn actions(&self) -> &'static [ActionDesc] {
        TypedComponent::actions(self)
    }
    fn invoke_action(&mut self, ctx: &mut ActionCtx, name: &str, args: &[ArgValue]) -> Result<(), String> {
        TypedComponent::invoke_action(self, ctx, name, args)
    }
    fn properties(&self) -> &'static [PropertyDesc] {
        TypedComponent::properties(self)
    }
    fn read_property(&self, name: &str) -> Option<ArgValue> {
        TypedComponent::read_property(self, name)
    }
    fn instance_label(&self) -> Option<String> {
        TypedComponent::instance_label(self)
    }
}

struct ComponentInfo {
    aliases: &'static [&'static str],
    name: &'static str,
    net_role: NetRole,
    create: fn() -> Box<dyn Component>,
    load: fn(&str) -> Result<Box<dyn Component>, ron::error::SpannedError>,
}

/// Name → component factory/loader (runtime). One registry per app; plugins
/// extend it via their `zestengine_register` export.
#[derive(Default)]
pub struct ComponentRegistry {
    entries: Vec<ComponentInfo>,
    /// Project-wide role per type name, winning over the type's own default (NET-3). Kept apart
    /// from `entries` so a hot-reloaded re-registration does not lose it.
    role_overrides: std::collections::BTreeMap<String, NetRole>,
}

impl ComponentRegistry {
    pub fn with_builtins() -> Self {
        let mut registry = Self::default();
        registry.register::<CameraComponent>();
        registry.register::<LightComponent>();
        registry.register::<AreaLight>();
        registry.register::<LightProbeVolume>();
        registry.register::<FluxVoxelLight>();
        registry.register::<FluxVolume>();
        registry.register::<ReflectionProbe>();
        registry.register::<PlanarReflection>();
        registry.register::<PerInstance>();
        registry.register::<AudioSource>();
        registry.register::<AudioListener>();
        registry.register::<AudioVolume>();
        registry.register::<AcousticMaterial>();
        registry.register::<AudioProbeVolume>();
        registry.register::<VolumeComponent>();
        registry.register::<BoxCollider>();
        registry.register::<SphereCollider>();
        registry.register::<CapsuleCollider>();
        registry.register::<MeshCollider>();
        registry.register::<Terrain>();
        registry.register::<TerrainCollider>();
        registry.register::<TerrainFoliage>();
        registry.register::<DecalProjector>();
        registry.register::<BoneAttach>();
        registry.register::<WeaponHold>();
        registry.register::<Spin>();
        registry.register::<Bob>();
        registry.register::<LodComponent>();
        registry.register::<RigidBody>();
        registry.register::<Character>();
        registry.register::<AvatarHands>();
        registry.register::<PhysicsJoint>();
        registry.register::<Rigidbody2D>();
        registry.register::<BoxCollider2D>();
        registry.register::<CircleCollider2D>();
        registry.register::<CapsuleCollider2D>();
        registry.register::<PolygonCollider2D>();
        registry.register::<EdgeCollider2D>();
        registry.register::<CompositeCollider2D>();
        registry.register::<Joint2D>();
        registry.register::<Effector2D>();
        registry.register::<Grid>();
        registry.register::<Tilemap>();
        registry.register::<TilemapRenderer>();
        registry.register::<TilemapCollider2D>();
        registry.register::<TileObject>();
        registry.register::<SpriteRenderer>();
        registry.register::<SortingGroup>();
        registry.register::<SpriteMask>();
        registry.register::<SpriteBone>();
        registry.register::<SpriteSkin>();
        registry.register::<SpriteLibraryRef>();
        registry.register::<SpriteResolver>();
        registry.register::<IkSolver2D>();
        registry.register::<SpriteShapeRenderer>();
        registry.register::<NavMeshSurface>();
        registry.register::<NavAgent>();
        registry.register::<ParticleEmitter>();
        registry.register::<crate::ai::PerceptionComponent>();
        registry.register::<crate::ai::BehaviourTreeAgent>();
        registry.register::<ScenePortal>();
        registry.register::<Pawn>();
        registry.register::<Vehicle>();
        registry.register::<SpawnPoint>();
        registry.register::<Sync>();
        registry.register::<NetworkObject>();
        registry.register::<NetworkVariables>();
        registry.register::<NetworkAnimator>();
        registry.register::<NetworkAvatar>();
        registry.register::<NetworkManager>();
        registry.register::<PersistentState>();
        registry.register::<Animator>();
        registry.register::<HumanoidIk>();
        registry.register::<FaceTracking>();
        registry.register::<BlendShapes>();
        registry.register::<mesh_renderer::MeshRenderer>();
        registry.register::<mesh_renderer::SkinnedMeshRenderer>();
        registry.register::<Constraint>();
        registry.register::<CustomProperties>();
        registry.register::<LipSync>();
        registry.register::<VoiceChat>();
        registry.register::<Water>();
        registry.register::<SkeletalWaterInteraction>();
        registry.register::<Buoyancy>();
        registry.register::<Waterfall>();
        registry.register::<Whirlpool>();
        registry.register::<WaterContainer>();
        registry.register::<WaveRider>();
        registry.register::<BoneCloth>();
        registry.register::<ClothColliders>();
        registry.register::<ClothWind>();
        registry.register::<MeshCloth>();
        registry.register::<UiCanvas>();
        registry.register::<UiRect>();
        registry.register::<UiPanel>();
        registry.register::<UiImage>();
        registry.register::<UiText>();
        registry.register::<Text3D>();
        registry.register::<UiButton>();
        registry.register::<UiToggle>();
        registry.register::<UiSlider>();
        registry.register::<UiTextField>();
        registry.register::<UiProgressBar>();
        registry.register::<UiPauseMenu>();
        registry.register::<AvatarDescriptor>();
        registry.register::<AvatarParameters>();
        registry.register::<AvatarMenuItem>();
        registry.register::<MenuToggle>();
        registry.register::<MenuInstaller>();
        registry.register::<ObjectToggle>();
        registry.register::<MergeAnimator>();
        registry.register::<MergeArmature>();
        registry.register::<SkinRebound>();
        registry.register::<AvatarMenuData>();
        registry.register::<crate::script::Script>();
        registry.register::<crate::wasm::WasmScript>();
        kit::register_kit(&mut registry);
        registry
    }

    /// Register a component type. Re-registering a name replaces the entry;
    /// that's how hot-reloaded plugin components supersede their old build.
    pub fn register<T: TypedComponent>(&mut self) {
        let info = ComponentInfo {
            aliases: T::ALIASES,
            name: T::NAME,
            net_role: if T::NET_ROLE == NetRole::Both { net_role::builtin_default_role(T::NAME) } else { T::NET_ROLE },
            create: || Box::new(T::default()),
            load: |data| Ok(Box::new(ron::from_str::<T>(data)?) as Box<dyn Component>),
        };
        match self.entries.iter_mut().find(|e| e.name == T::NAME) {
            Some(entry) => {
                tracing::debug!("component {:?} re-registered (reload)", T::NAME);
                *entry = info;
            }
            None => self.entries.push(info),
        }
    }

    /// The default [`NetRole`] of a registered type; `Both` for unknown names
    /// (a plugin that is not loaded keeps its components on both sides).
    pub fn default_role(&self, name: &str) -> NetRole {
        let entry = self.entries.iter().find(|e| e.name == name || e.aliases.contains(&name));
        let override_role = entry.and_then(|e| self.role_overrides.get(e.name)
            .or_else(|| e.aliases.iter().find_map(|alias| self.role_overrides.get(*alias))))
            .or_else(|| self.role_overrides.get(name));
        if let Some(r) = override_role {
            return *r;
        }
        self.entries.iter().find(|e| e.name == name || e.aliases.contains(&name)).map_or(NetRole::Both, |e| e.net_role)
    }

    /// The role a type has before any project-wide override.
    pub fn type_role(&self, name: &str) -> NetRole {
        self.entries.iter().find(|e| e.name == name || e.aliases.contains(&name)).map_or(NetRole::Both, |e| e.net_role)
    }

    /// Replace the project-wide role overrides (`name -> role`; `Both` entries that equal the
    /// type default are harmless). Unknown names are kept: the plugin may load later.
    pub fn set_role_overrides(&mut self, overrides: impl IntoIterator<Item = (String, NetRole)>) {
        self.role_overrides = overrides.into_iter().collect();
    }

    /// Set (`Some`) or remove (`None`) one override; returns the previous value.
    pub fn set_role_override(&mut self, name: &str, role: Option<NetRole>) -> Option<NetRole> {
        let name = self.canonical_name(name).to_owned();
        let mut previous = self.role_overrides.remove(&name);
        if let Some(entry) = self.entries.iter().find(|e| e.name == name) {
            for alias in entry.aliases {
                let legacy = self.role_overrides.remove(*alias);
                previous = previous.or(legacy);
            }
        }
        if let Some(role) = role { self.role_overrides.insert(name, role); }
        previous
    }

    pub fn role_overrides(&self) -> &std::collections::BTreeMap<String, NetRole> {
        &self.role_overrides
    }

    /// Canonical component name (unknown plugin names are preserved).
    pub fn canonical_name<'a>(&self, name: &'a str) -> &'a str {
        self.entries.iter().find(|e| e.name == name || e.aliases.contains(&name)).map_or(name, |e| e.name)
    }

    /// Canonicalize a saved component key while retaining its duplicate suffix.
    pub fn canonical_key(&self, key: &str) -> String {
        match key.rsplit_once('#').filter(|(_, suffix)| suffix.parse::<usize>().is_ok()) {
            Some((name, suffix)) => format!("{}#{suffix}", self.canonical_name(name)),
            None => self.canonical_name(key).to_owned(),
        }
    }

    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.entries.iter().map(|e| e.name)
    }

    pub fn create(&self, name: &str) -> Option<Box<dyn Component>> {
        let entry = self.entries.iter().find(|e| e.name == name || e.aliases.contains(&name))?;
        Some((entry.create)())
    }

    /// Deserialize a saved component. Unknown names and broken data log a
    /// warning and return None so scene loads survive missing components.
    pub fn load(&self, name: &str, data: &str) -> Option<Box<dyn Component>> {
        let Some(entry) = self.entries.iter().find(|e| e.name == name || e.aliases.contains(&name)) else {
            tracing::warn!("unknown component {name:?}; dropping it");
            return None;
        };
        match (entry.load)(data) {
            Ok(component) => Some(component),
            Err(e) => {
                tracing::warn!("loading component {name:?}: {e}");
                None
            }
        }
    }
}

// ----------------------------------------------------------- builtin data

/// Camera properties (Unity-style). Attached automatically to camera objects;
/// the viewport draws a frustum widget from these values.
#[derive(Serialize, Deserialize)]
pub struct CameraComponent {
    /// Stable per-camera identity, saved in the `.scene`. The scene's "main"
    /// camera is the one with the smallest id. 0 = not yet assigned.
    #[serde(default)]
    pub id: u32,
    /// Vertical field of view, degrees.
    pub fov_deg: f32,
    pub near: f32,
    pub far: f32,
    /// Layer culling mask (Unity-style): a bit per layer; this camera only
    /// renders objects whose layer bit is set. Default = all layers.
    #[serde(default = "all_layers_mask")]
    pub culling_mask: u32,
    /// Render into a texture instead of the screen (CCTV / TV / minimap). A
    /// material samples it through the texture path `camera:<object id>`. Never
    /// the main camera.
    #[serde(default)]
    pub render_texture: bool,
    /// Render-texture size in pixels.
    #[serde(default = "default_render_texture_size")]
    pub texture_size: [u32; 2],
    /// Render-texture refreshes per second; 0 = every frame. Screens nobody
    /// can see are skipped regardless.
    #[serde(default)]
    pub update_hz: f32,
    /// Transient: the inspector's "Create Screen" button (spawns a plane showing
    /// this camera's feed). Consumed by the engine.
    #[serde(skip)]
    pub create_screen: bool,
    /// Perspective (uses `fov_deg`) or Orthographic (uses `ortho_size`).
    #[serde(default)]
    pub projection: CameraProjection,
    /// Orthographic half-height in world units (the visible height is twice
    /// this; the width follows the aspect ratio).
    #[serde(default = "default_ortho_size")]
    pub ortho_size: f32,
    /// Anti-aliasing for this camera; `Project` follows the Project Settings.
    #[serde(default)]
    pub anti_aliasing: CameraAntiAliasing,
}

/// Per-camera anti-aliasing override. Mirrors the renderer's `AntiAliasing`
/// (core does not depend on the render crate) plus `Project` = "inherit".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CameraAntiAliasing {
    /// Use the Project Settings value.
    #[default]
    Project,
    None,
    Fxaa,
    Taa,
    Msaa2,
    Msaa4,
    Fsr,
}

impl CameraAntiAliasing {
    pub const ALL: [CameraAntiAliasing; 7] = [Self::Project, Self::None, Self::Fxaa, Self::Taa, Self::Msaa2, Self::Msaa4, Self::Fsr];

    pub fn label(self) -> &'static str {
        match self {
            Self::Project => "Project setting",
            Self::None => "None",
            Self::Fxaa => "FXAA",
            Self::Taa => "TAA",
            Self::Msaa2 => "MSAA 2x",
            Self::Msaa4 => "MSAA 4x",
            Self::Fsr => "FSR 3",
        }
    }
}

fn default_render_texture_size() -> [u32; 2] {
    [512, 512]
}

fn default_ortho_size() -> f32 {
    5.0
}

/// How a camera projects the scene.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CameraProjection {
    #[default]
    Perspective,
    Orthographic,
}

impl CameraProjection {
    pub const ALL: [CameraProjection; 2] = [CameraProjection::Perspective, CameraProjection::Orthographic];

    pub fn label(self) -> &'static str {
        match self {
            CameraProjection::Perspective => "Perspective",
            CameraProjection::Orthographic => "Orthographic",
        }
    }
}

/// Right-handed orthographic projection (0..1 depth, like `perspective_rh`)
/// for a view `half_height` tall (world units) at `aspect` (width / height).
/// Unlike a perspective matrix, `near` may be negative to put the camera
/// inside the visible volume.
pub fn orthographic_projection(half_height: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let h = half_height.max(1e-4);
    let w = h * aspect.max(0.01);
    Mat4::orthographic_rh(-w, w, -h, h, near, far)
}

/// True when `proj` is an orthographic matrix (the last row of a perspective
/// matrix is `(0, 0, -1, 0)`, of an orthographic one `(0, 0, 0, 1)`).
pub fn is_orthographic(proj: &Mat4) -> bool {
    proj.w_axis.w.abs() > 0.5
}

/// World-space ray `(origin, direction)` through normalised device coords
/// `ndc` for a view/projection pair. Perspective rays start at `eye`;
/// orthographic rays are parallel and start on the near plane at the pointer.
pub fn ndc_ray(view: Mat4, proj: Mat4, ndc: glam::Vec2, eye: Vec3) -> Option<(Vec3, Vec3)> {
    let inv = (proj * view).inverse();
    let near = inv.project_point3(ndc.extend(0.0));
    let far = inv.project_point3(ndc.extend(1.0));
    let dir = (far - near).try_normalize()?;
    Some((if is_orthographic(&proj) { near } else { eye }, dir))
}

impl CameraComponent {
    /// Projection matrix for `aspect` (width / height).
    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        let near = self.near.max(0.001);
        let far = self.far.max(near + 0.001);
        match self.projection {
            CameraProjection::Perspective => Mat4::perspective_rh(self.fov_deg.to_radians(), aspect.max(0.01), near, far),
            CameraProjection::Orthographic => orthographic_projection(self.ortho_size, aspect, near, far),
        }
    }

    /// Half-width and half-height of the view volume at distance `d` in front
    /// of the camera (constant for orthographic cameras).
    pub fn half_extents_at(&self, aspect: f32, d: f32) -> (f32, f32) {
        let h = match self.projection {
            CameraProjection::Perspective => (self.fov_deg.to_radians() * 0.5).tan() * d,
            CameraProjection::Orthographic => self.ortho_size.max(1e-4),
        };
        (h * aspect, h)
    }
}

impl Default for CameraComponent {
    fn default() -> Self {
        Self {
            id: 0,
            projection: CameraProjection::Perspective,
            ortho_size: default_ortho_size(),
            fov_deg: 60.0,
            near: 0.1,
            far: 100.0,
            culling_mask: all_layers_mask(),
            render_texture: false,
            texture_size: default_render_texture_size(),
            update_hz: 0.0,
            create_screen: false,
            anti_aliasing: CameraAntiAliasing::Project,
        }
    }
}

/// Default layer mask: every layer visible / collidable.
pub fn all_layers_mask() -> u32 {
    u32::MAX
}

/// Box-projected (parallax-corrected) reflection direction — the **reference
/// implementation** the `standard.frag` reflection-probe block must match. Given a
/// surface at `world_pos` reflecting along `refl_dir`, and a probe whose box is
/// centred at `center` with half-extents `half` (and whose cubemap was captured
/// from `center`), it intersects the reflection ray with the box and re-vectors
/// from the capture centre, so a flat surface lines the reflection up with the
/// captured environment instead of treating the cube as infinitely distant.
///
/// Kept here (egui-free, glam-only) so the math is **unit-tested** independently of
/// the GPU — `cargo test -p zestengine-core box_projection` verifies the algorithm, and
/// any divergence between this and the shader is a bug in one of them.
pub fn box_project_reflection(world_pos: Vec3, refl_dir: Vec3, center: Vec3, half: Vec3) -> Vec3 {
    let bmin = center - half;
    let bmax = center + half;
    let inv_r = refl_dir.recip();
    let t1 = (bmax - world_pos) * inv_r;
    let t2 = (bmin - world_pos) * inv_r;
    let tmax = t1.max(t2);
    let dist = tmax.x.min(tmax.y).min(tmax.z);
    let hit = world_pos + refl_dir * dist;
    hit - center
}

/// Number of object layers (Unity-style: a 32-bit mask, one bit per layer).
pub const NUM_LAYERS: usize = 32;

/// Project-wide layer configuration: layer names + a symmetric layer-collision
/// matrix (Unity's "Layer Collision Matrix"). Drives both physics (which layers
/// collide) and rendering (per-camera culling masks). Serialized with the scene.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerSettings {
    /// Display name per layer index. Empty string = unnamed (shown as "Layer N").
    pub names: Vec<String>,
    /// `collision[i]` is the bitmask of layers that layer `i` collides with.
    /// Kept symmetric (`collide(a,b) == collide(b,a)`).
    pub collision: Vec<u32>,
    /// The 2D sorting layers (2D-3): ordered back to front, stable ids. Serde
    /// default = one "Default" layer, so older scene files load unchanged.
    #[serde(default)]
    pub sorting: SortingLayers,
}

impl Default for LayerSettings {
    fn default() -> Self {
        let mut names = vec![String::new(); NUM_LAYERS];
        names[0] = "Default".to_string();
        Self {
            names,
            collision: vec![u32::MAX; NUM_LAYERS],
            sorting: SortingLayers::default(),
        }
    }
}

impl LayerSettings {
    /// Whether two layers collide (defaults to true for out-of-range indices).
    pub fn collide(&self, a: u8, b: u8) -> bool {
        self.collision
            .get(a as usize)
            .is_none_or(|m| m & (1u32 << (b as u32 & 31)) != 0)
    }

    /// Set (symmetrically) whether layers `a` and `b` collide.
    pub fn set_collide(&mut self, a: u8, b: u8, on: bool) {
        for (x, y) in [(a, b), (b, a)] {
            if let Some(m) = self.collision.get_mut(x as usize) {
                let bit = 1u32 << (y as u32 & 31);
                if on {
                    *m |= bit;
                } else {
                    *m &= !bit;
                }
            }
        }
    }

    /// How many layers the editor UIs show. All 32 (`NUM_LAYERS`), CONSISTENTLY
    /// across the object layer dropdown, the camera culling mask, and the collision
    /// matrix — Unity-style (every layer is always addressable). A single source of
    /// truth so the three can never disagree (they previously showed 32 / 8 / 4).
    pub fn shown_count(&self) -> usize {
        NUM_LAYERS
    }

    /// Display name for a layer index ("Layer N" when unnamed).
    pub fn layer_name(&self, i: u8) -> String {
        self.names
            .get(i as usize)
            .filter(|s| !s.is_empty())
            .cloned()
            .unwrap_or_else(|| format!("Layer {i}"))
    }

    /// The collision filter mask for objects on `layer` (which layers it hits).
    pub fn collision_mask(&self, layer: u8) -> u32 {
        self.collision.get(layer as usize).copied().unwrap_or(u32::MAX)
    }

    /// Repair a deserialized value that's missing entries (short vecs from an
    /// older/edited file) so indexing 0..NUM_LAYERS is always safe.
    pub fn normalized(mut self) -> Self {
        self.names.resize(NUM_LAYERS, String::new());
        self.collision.resize(NUM_LAYERS, u32::MAX);
        if self.names[0].is_empty() {
            self.names[0] = "Default".to_string();
        }
        self.sorting = self.sorting.normalized();
        self
    }
}

impl TypedComponent for CameraComponent {
    const NAME: &'static str = "Camera";
}

/// What shape of light a [`LightComponent`] emits.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum LightKind {
    Directional,
    Point,
    Spot,
}

impl LightKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Directional => "Directional",
            Self::Point => "Point",
            Self::Spot => "Spot",
        }
    }
    pub const ALL: [LightKind; 3] = [Self::Directional, Self::Point, Self::Spot];
}

/// Whether a light is recomputed every frame or baked into the scene's lighting.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum LightMode {
    Realtime,
    Baked,
    Mixed,
}

impl LightMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Realtime => "Realtime",
            Self::Baked => "Baked (static)",
            Self::Mixed => "Mixed",
        }
    }
    pub const ALL: [LightMode; 3] = [Self::Realtime, Self::Baked, Self::Mixed];
}

/// How a light renders its shadows. `None` skips the shadow map entirely; `Hard`
/// does a single depth comparison (sharp, aliased edge); `Soft` runs the PCF
/// kernel for a smooth penumbra.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum ShadowType {
    None,
    Hard,
    Soft,
}

impl ShadowType {
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "No Shadows",
            Self::Hard => "Hard Shadows",
            Self::Soft => "Soft Shadows",
        }
    }
    pub const ALL: [ShadowType; 3] = [Self::None, Self::Hard, Self::Soft];
    /// Whether a shadow map should be allocated/rendered for this light.
    pub fn casts(self) -> bool {
        !matches!(self, Self::None)
    }
    /// Whether the PCF kernel runs (soft penumbra) vs a single hard tap.
    pub fn soft(self) -> bool {
        matches!(self, Self::Soft)
    }
}

/// How a rendered object takes part in shadows (Unity's MeshRenderer "Cast
/// Shadows"). `Off` keeps it out of every shadow map and the bake's shadow
/// rays; `ShadowsOnly` casts but is never drawn by a camera.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum ShadowCasting {
    #[default]
    On,
    Off,
    TwoSided,
    ShadowsOnly,
}

impl ShadowCasting {
    pub fn label(self) -> &'static str {
        match self {
            Self::On => "On",
            Self::Off => "Off",
            Self::TwoSided => "Two Sided",
            Self::ShadowsOnly => "Shadows Only",
        }
    }
    pub const ALL: [ShadowCasting; 4] = [Self::On, Self::Off, Self::TwoSided, Self::ShadowsOnly];
    /// The default; scene files skip writing it.
    pub fn is_on(&self) -> bool {
        *self == Self::On
    }
    /// Drawn into shadow maps.
    pub fn casts(self) -> bool {
        self != Self::Off
    }
    /// Drawn by cameras (main, prepass, captures, mirrors).
    pub fn visible(self) -> bool {
        self != Self::ShadowsOnly
    }
}

/// A scene light. The object's transform places/orients it; the engine gathers
/// every `LightComponent` each frame and feeds the renderer's multi-light path.
#[derive(Serialize, Deserialize)]
pub struct LightComponent {
    pub kind: LightKind,
    #[serde(default = "default_light_mode")]
    pub mode: LightMode,
    /// Linear RGB tint, 0..1.
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,
    /// Spot cone full angle at the outer edge, degrees.
    pub spot_angle: f32,
    /// 0 = hard cone edge, 1 = soft falloff.
    pub spot_blend: f32,
    #[serde(default = "default_shadow_type", alias = "cast_shadows")]
    pub shadow_type: ShadowType,
    #[serde(default = "default_shadow_bias")]
    pub shadow_bias: f32,
    /// Light source radius (world units) for baked soft shadows: the bake
    /// samples the shadow ray over a disc of this radius, so a larger value
    /// gives a wider, softer penumbra. 0 = hard shadow.
    #[serde(default = "default_light_radius")]
    pub radius: f32,
}

fn default_light_mode() -> LightMode {
    LightMode::Realtime
}
fn default_shadow_type() -> ShadowType {
    ShadowType::Soft
}
fn default_shadow_bias() -> f32 {
    0.002
}
fn default_light_radius() -> f32 {
    0.1
}

impl Default for LightComponent {
    fn default() -> Self {
        Self {
            kind: LightKind::Directional,
            mode: LightMode::Realtime,
            color: [1.0, 0.98, 0.92],
            intensity: 3.0,
            range: 10.0,
            spot_angle: 45.0,
            spot_blend: 0.15,
            shadow_type: ShadowType::Soft,
            shadow_bias: 0.002,
            radius: default_light_radius(),
        }
    }
}

impl LightComponent {
    /// Approximate photometric readout for the inspector. Our `intensity` isn't
    /// a physical unit (it's a radiance multiplier with a non-inverse-square
    /// falloff), so this treats `intensity` as candela-equivalent and converts
    /// to luminous flux the textbook way: point = 4π·I over the full sphere,
    /// spot = the cone's solid angle 2π(1-cos θ)·I. Directional has no flux
    /// (it's an infinite source) so it reports illuminance (lux ≈ intensity).
    /// Returns (value, unit) for display only; nothing reads it back.
    pub fn approx_photometric(&self) -> (f32, &'static str) {
        match self.kind {
            LightKind::Directional => (self.intensity, "lx"),
            LightKind::Point => (self.intensity * 4.0 * std::f32::consts::PI, "lm"),
            LightKind::Spot => {
                let half = (self.spot_angle * 0.5).to_radians();
                let solid = 2.0 * std::f32::consts::PI * (1.0 - half.cos());
                (self.intensity * solid, "lm")
            }
        }
    }
}

static LIGHT_ACTIONS: &[ActionDesc] = &[
    ActionDesc::new("SetIntensity", "Set Intensity", &[ParamDesc::float("intensity", 1.0)], true),
    ActionDesc::new("SetColor", "Set Color", &[ParamDesc::vec3("rgb", [1.0, 1.0, 1.0])], true),
];
static LIGHT_PROPERTIES: &[PropertyDesc] = &[
    PropertyDesc::new("intensity", ActionType::Float),
    PropertyDesc::new("range", ActionType::Float),
];

impl TypedComponent for LightComponent {
    const NAME: &'static str = "Light";

    fn actions(&self) -> &'static [ActionDesc] {
        LIGHT_ACTIONS
    }
    fn invoke_action(&mut self, _ctx: &mut ActionCtx, name: &str, args: &[ArgValue]) -> Result<(), String> {
        match name {
            "SetIntensity" => self.intensity = args[0].as_f32().max(0.0),
            "SetColor" => self.color = args[0].as_vec3().clamp(Vec3::ZERO, Vec3::ONE).to_array(),
            _ => return Err(actions::unknown_action(Self::NAME, name)),
        }
        Ok(())
    }
    fn properties(&self) -> &'static [PropertyDesc] {
        LIGHT_PROPERTIES
    }
    fn read_property(&self, name: &str) -> Option<ArgValue> {
        match name {
            "intensity" => Some(ArgValue::Float(self.intensity)),
            "range" => Some(ArgValue::Float(self.range)),
            _ => None,
        }
    }
}

/// Shape of an [`AreaLight`] emitter.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum AreaLightShape {
    #[default]
    Rectangle,
    Disc,
}

impl AreaLightShape {
    pub const ALL: [AreaLightShape; 2] = [Self::Rectangle, Self::Disc];
    pub fn label(self) -> &'static str {
        match self {
            Self::Rectangle => "Rectangle",
            Self::Disc => "Disc",
        }
    }
}

/// A rectangle or disc emitter shaded with linearly transformed cosines: soft diffuse light and
/// a specular highlight shaped like the emitter, with an optional emission texture (a video or
/// camera texture makes a screen light). It lights toward the object's -Z (like the other lights)
/// and casts **no shadows** (v1). At most 8 area lights, the nearest to the camera, are drawn.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AreaLight {
    pub shape: AreaLightShape,
    /// Full width and height in metres (the disc's diameters along local X / Y).
    pub size: [f32; 2],
    /// Linear RGB tint.
    pub color: [f32; 3],
    /// Emitted radiance: the brightness a white surface facing the emitter right up against it shows.
    pub intensity: f32,
    /// Distance at which the light reaches zero.
    pub range: f32,
    /// Also light the back.
    pub two_sided: bool,
    /// Project-relative emission texture (empty = plain colour). `camera:<id>` uses a camera's
    /// render texture, so a second camera can act as a screen light.
    pub texture: String,
}

impl Default for AreaLight {
    fn default() -> Self {
        Self {
            shape: AreaLightShape::Rectangle,
            size: [1.0, 1.0],
            color: [1.0, 1.0, 1.0],
            intensity: 4.0,
            range: 15.0,
            two_sided: false,
            texture: String::new(),
        }
    }
}

impl TypedComponent for AreaLight {
    const NAME: &'static str = "Area Light";
}

/// Rotate around a local axis at constant speed.
#[derive(Serialize, Deserialize)]
pub struct Spin {
    pub axis: [f32; 3],
    pub degrees_per_second: f32,
}

impl Default for Spin {
    fn default() -> Self {
        Self {
            axis: [0.0, 1.0, 0.0],
            degrees_per_second: 45.0,
        }
    }
}

impl TypedComponent for Spin {
    const NAME: &'static str = "Spin";
    fn update(&mut self, ctx: &mut ComponentCtx) {
        let axis = Vec3::from(self.axis).normalize_or(Vec3::Y);
        let step = Quat::from_axis_angle(axis, self.degrees_per_second.to_radians() * ctx.dt);
        *ctx.rotation = (step * *ctx.rotation).normalize();
    }
}

/// Sine-wave hover along an axis.
#[derive(Serialize, Deserialize)]
pub struct Bob {
    pub axis: [f32; 3],
    pub amplitude: f32,
    /// Cycles per second.
    pub frequency: f32,
    /// Offset currently applied, so each frame applies only the delta.
    #[serde(skip)]
    applied: Vec3,
}

impl Default for Bob {
    fn default() -> Self {
        Self {
            axis: [0.0, 1.0, 0.0],
            amplitude: 0.5,
            frequency: 0.5,
            applied: Vec3::ZERO,
        }
    }
}

impl TypedComponent for Bob {
    const NAME: &'static str = "Bob";
    fn update(&mut self, ctx: &mut ComponentCtx) {
        let phase = ctx.time * self.frequency * std::f32::consts::TAU;
        let offset = Vec3::from(self.axis) * (self.amplitude * phase.sin());
        *ctx.translation += offset - self.applied;
        self.applied = offset;
    }
}

/// Level-of-detail group (Unity `LODGroup`-style): swap the object's mesh for a
/// cheaper one as it gets farther from the camera, and optionally cull past the
/// last level. Authored PER-OBJECT in the Inspector (this is a component, not a
/// tool). The renderer reads the selected level via `select_lod`.
#[derive(Clone, Serialize, Deserialize)]
pub struct LodComponent {
    /// LOD levels, ascending by `max_distance`. Level 0 = highest detail (nearest).
    pub levels: Vec<LodLevel>,
    /// Beyond this camera distance the object is culled entirely (0 = never).
    pub cull_distance: f32,
    /// Select levels by screen size (`LodLevel::screen_size`) instead of
    /// `max_distance`. Needs no re-tuning per object size or camera FOV.
    #[serde(default)]
    pub by_screen_size: bool,
}

/// One LOD level: used while camera distance <= `max_distance`.
#[derive(Clone, Serialize, Deserialize)]
pub struct LodLevel {
    pub max_distance: f32,
    /// Project-relative model path for this level (empty = keep the base mesh).
    pub model: String,
    /// With no model: generate this level from the object's own model,
    /// keeping this share of its triangles (0 or 1 = the base mesh).
    #[serde(default)]
    pub keep: f32,
    /// Screen-size mode: the smallest share of the screen height the object's
    /// bounding sphere may cover while this level is used (levels descending).
    #[serde(default)]
    pub screen_size: f32,
}

impl Default for LodComponent {
    fn default() -> Self {
        Self {
            levels: vec![
                LodLevel { max_distance: 15.0, model: String::new(), keep: 0.0, screen_size: 0.0 },
                LodLevel { max_distance: 50.0, model: String::new(), keep: 0.5, screen_size: 0.0 },
            ],
            cull_distance: 0.0,
            by_screen_size: false,
        }
    }
}

impl LodComponent {
    /// The screen-size chain every model gets automatically (see [`lod`]).
    pub fn auto_chain() -> Self {
        Self {
            levels: lod::AUTO_LOD_STEPS.iter().map(|&(keep, size)| LodLevel::by_screen(keep, size)).collect(),
            cull_distance: 0.0,
            by_screen_size: true,
        }
    }

    /// Pick the LOD index for a screen coverage (see [`lod::screen_coverage`]),
    /// holding `previous` inside the hysteresis band. Screen-size mode only.
    pub fn select_by_screen(&self, coverage: f32, previous: Option<usize>) -> Option<usize> {
        let thresholds: Vec<f32> = self.levels.iter().map(|l| l.screen_size).collect();
        lod::select_by_coverage(&thresholds, coverage, previous)
    }

    /// Pick the LOD index for a camera distance, or `None` if culled.
    pub fn select_lod(&self, distance: f32) -> Option<usize> {
        if self.cull_distance > 0.0 && distance > self.cull_distance {
            return None;
        }
        // Past the last level: keep the coarsest one (culling is cull_distance's job).
        self.levels
            .iter()
            .position(|l| distance <= l.max_distance)
            .or(self.levels.len().checked_sub(1))
    }
}

impl TypedComponent for LodComponent {
    const NAME: &'static str = "LOD Group";
}

/// A box volume seeded with a regular grid of irradiance light probes.
///
/// Two modes. With `blend_distance` 0 (the default) it is a classic probe volume: the
/// finest volume covering a point wins and volumes fade into the next coarser one (the
/// cascades the realtime GI also uses). With `blend_distance` > 0 it is a **Light
/// Volume**: baked SH light for objects without a lightmap, weight-blended with the
/// other Light Volumes covering the point and fading out over `blend_distance` metres
/// towards its faces.
#[derive(Serialize, Deserialize)]
pub struct LightProbeVolume {
    /// Full box size in local meters (grid spans -size/2 .. +size/2).
    pub size: [f32; 3],
    /// Probes per meter along each axis.
    pub density: f32,
    /// Light Volume mode when > 0: how far in from a face the volume fades out (metres).
    #[serde(default)]
    pub blend_distance: f32,
    /// Light Volume blend weight (overlapping volumes share by weight x edge fade).
    #[serde(default = "default_volume_weight")]
    pub weight: f32,
}

fn default_volume_weight() -> f32 {
    1.0
}

impl Default for LightProbeVolume {
    fn default() -> Self {
        Self {
            size: [4.0, 3.0, 4.0],
            density: 1.0,
            blend_distance: 0.0,
            weight: 1.0,
        }
    }
}

impl LightProbeVolume {
    /// Probe count along each axis (>= 2 so a cell always has 8 corners).
    pub fn counts(&self) -> [usize; 3] {
        let mut out = [2usize; 3];
        for (i, c) in out.iter_mut().enumerate() {
            let n = (self.size[i].max(0.0) * self.density).round() as i64 + 1;
            *c = n.clamp(2, 256) as usize;
        }
        out
    }

    pub fn probe_count(&self) -> usize {
        let [x, y, z] = self.counts();
        x * y * z
    }

    /// Probe positions in local space, x-fastest then y then z.
    pub fn local_positions(&self) -> Vec<Vec3> {
        let [nx, ny, nz] = self.counts();
        let half = Vec3::from(self.size) * 0.5;
        let step = |n: usize, axis: usize| {
            if n > 1 {
                self.size[axis] / (n - 1) as f32
            } else {
                0.0
            }
        };
        let (sx, sy, sz) = (step(nx, 0), step(ny, 1), step(nz, 2));
        let mut out = Vec::with_capacity(nx * ny * nz);
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    out.push(Vec3::new(
                        -half.x + sx * ix as f32,
                        -half.y + sy * iy as f32,
                        -half.z + sz * iz as f32,
                    ));
                }
            }
        }
        out
    }
}

impl TypedComponent for LightProbeVolume {
    const NAME: &'static str = "Light Probe Volume";
}

/// Marks an object as a FluxVoxel light source. With the FluxVoxel GI backend active,
/// the object casts `color × intensity` into the FluxVoxel voxel light volume with
/// smooth distance falloff out to `range`, from the object's (possibly moving)
/// world position. Put it on a light or an emissive prop, static OR dynamic — a
/// dynamic FluxVoxel light mixes into the same voxel grid live, so it lights up
/// static objects sharing the volume. Only objects bearing this component feed
/// FluxVoxel, so you choose exactly which lights/props participate.
#[derive(Serialize, Deserialize)]
pub struct FluxVoxelLight {
    pub color: [f32; 3],
    pub intensity: f32,
    /// World-space distance at which the contribution reaches zero.
    pub range: f32,
}

impl Default for FluxVoxelLight {
    fn default() -> Self {
        Self {
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            range: 8.0,
        }
    }
}

impl TypedComponent for FluxVoxelLight {
    const NAME: &'static str = "FluxVoxel Light";
}

/// A FluxVoxel voxel light volume: a box (centred on the object, `size` local
/// meters) filled with a regular grid of SH-L1 light probes at `density` probes
/// per meter. FluxVoxel injects every FluxVoxel Light into these voxels (static ones
/// baked once, dynamic ones mixed live) and the standard shader reads them to
/// light static + dynamic meshes inside the box. Place one (or several) over the
/// play space; configurable size + density trade quality for memory/cost.
#[derive(Serialize, Deserialize)]
pub struct FluxVolume {
    /// Full box size in local meters (grid spans -size/2 .. +size/2).
    pub size: [f32; 3],
    /// Probes per meter along each axis.
    pub density: f32,
}

impl Default for FluxVolume {
    fn default() -> Self {
        Self {
            size: [8.0, 4.0, 8.0],
            density: 0.75,
        }
    }
}

impl FluxVolume {
    /// Probe count along each axis (>= 2 so a cell always has 8 corners),
    /// clamped so a huge/dense box can't explode the probe buffer.
    pub fn counts(&self) -> [u32; 3] {
        let mut out = [2u32; 3];
        for (i, c) in out.iter_mut().enumerate() {
            let n = (self.size[i].max(0.0) * self.density).round() as i64 + 1;
            *c = n.clamp(2, 64) as u32;
        }
        out
    }

    pub fn probe_count(&self) -> usize {
        let [x, y, z] = self.counts();
        (x * y * z) as usize
    }

    /// Voxel-grid positions in local space (x-fastest then y then z) — the
    /// points FluxVoxel injects light into.
    pub fn local_positions(&self) -> Vec<Vec3> {
        let [nx, ny, nz] = self.counts();
        let half = Vec3::from(self.size) * 0.5;
        let step = |n: u32, axis: usize| {
            if n > 1 {
                self.size[axis] / (n - 1) as f32
            } else {
                0.0
            }
        };
        let (sx, sy, sz) = (step(nx, 0), step(ny, 1), step(nz, 2));
        let mut out = Vec::with_capacity((nx * ny * nz) as usize);
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    out.push(Vec3::new(
                        -half.x + sx * ix as f32,
                        -half.y + sy * iy as f32,
                        -half.z + sz * iz as f32,
                    ));
                }
            }
        }
        out
    }
}

impl TypedComponent for FluxVolume {
    const NAME: &'static str = "Flux Volume";
}

/// A precomputed reflection capture: a box zone whose surroundings are captured
/// into a roughness-prefiltered cubemap (Unreal reflection-capture / Unity
/// reflection-probe style) and sampled at runtime for specular environment
/// reflection. The object's transform places it; `size` is the box influence
/// region (used for box-projected parallax + blend between probes).
/// How a reflection probe's cubemap is produced.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum ReflectionProbeMode {
    /// Captured live (recaptured on demand / scene changes); not saved to disk.
    #[default]
    Realtime,
    /// Baked to a `.reflprobe` sidecar and loaded from disk (static, cheapest).
    Baked,
}

#[derive(Serialize, Deserialize)]
pub struct ReflectionProbe {
    /// Full box influence size in local meters (centered on the object).
    pub size: [f32; 3],
    /// Box-center offset from the object position (world-axis meters, Unity's
    /// box offset). Decouples WHERE the cube is captured from (the object
    /// position — put it at eye height so interior objects reflect right) from
    /// WHERE the box sits (aligned to the room's walls/floor/ceiling).
    #[serde(default)]
    pub box_offset: [f32; 3],
    /// Capture-point offset from the object position (world-axis meters).
    /// Moves where the cube is rendered from without moving the object or its
    /// box (drag the capture handle in the viewport).
    #[serde(default)]
    pub capture_offset: [f32; 3],
    /// Keep the box FIXED IN WORLD SPACE while the object moves: `box_offset`
    /// auto-compensates each frame, so dragging the probe repositions only the
    /// capture point inside the room-aligned box. Off = box follows the object.
    #[serde(default)]
    pub lock_box: bool,
    /// Transient: last frame's world position, for `lock_box` compensation.
    #[serde(skip)]
    pub last_world_pos: Option<[f32; 3]>,
    /// Captured cubemap face resolution (per side): 32 … 8192.
    pub resolution: u32,
    /// Realtime (live capture) vs Baked (loaded from a `.reflprobe` sidecar).
    #[serde(default)]
    pub mode: ReflectionProbeMode,
    /// Transient: set by the inspector's "Bake This Probe" button, consumed by
    /// the engine to capture + save this probe. Never serialized.
    #[serde(skip)]
    pub bake_now: bool,
    /// Reflection strength multiplier applied when sampled.
    pub intensity: f32,
    /// Box-projected parallax correction (Unity-style): reproject the reflection
    /// ray onto the box so flat surfaces line up. Off = treat as distant (infinite).
    pub box_projection: bool,
    /// Blend-zone thickness (local meters) just inside the box edge. The probe's
    /// weight ramps 0→1 across this shell, so overlapping probes cross-fade and
    /// the edge fades to the skybox instead of popping. 0 = hard cutoff at the box.
    #[serde(default = "default_blend_distance")]
    pub blend_distance: f32,
    /// Overlap priority: higher-importance probes consume blend weight first, so a
    /// small specific probe wins over a large surrounding one where they overlap.
    #[serde(default)]
    pub importance: i32,
    /// Realtime mode: fresh cubes per second (0 = every frame).
    #[serde(default = "default_probe_update_hz")]
    pub update_hz: f32,
    /// Realtime mode: spread a capture over 7 frames (one face each, then the
    /// mip chain) instead of rendering all six faces in one frame. Cheaper per
    /// frame, but moving objects step in the reflection.
    #[serde(default)]
    pub time_slicing: bool,
    /// Realtime mode: also draw non-static objects into the cube. Off by
    /// default: box projection maps everything in the cube onto the box walls,
    /// so a moving object inside the room shows up as a ghost on far surfaces
    /// (SSR / ray-traced reflections handle moving objects instead).
    #[serde(default)]
    pub include_dynamic: bool,
}

fn default_blend_distance() -> f32 {
    1.0
}

fn default_probe_update_hz() -> f32 {
    30.0
}

impl Default for ReflectionProbe {
    fn default() -> Self {
        Self {
            size: [10.0, 6.0, 10.0],
            box_offset: [0.0; 3],
            capture_offset: [0.0; 3],
            lock_box: false,
            last_world_pos: None,
            resolution: 256,
            intensity: 1.0,
            box_projection: true,
            blend_distance: default_blend_distance(),
            importance: 0,
            update_hz: default_probe_update_hz(),
            time_slicing: false,
            include_dynamic: false,
            mode: ReflectionProbeMode::Realtime,
            bake_now: false,
        }
    }
}

impl TypedComponent for ReflectionProbe {
    const NAME: &'static str = "Reflection Probe";
}

/// A true mirror for a flat surface (floor, wall mirror): the scene is
/// re-rendered every frame from the camera reflected about the object's world
/// +Y plane and composited with per-pixel parallax — exact reflections that a
/// reflection probe (single capture point) cannot produce for mid-room
/// geometry. Each active surface costs a mirrored scene pass; at most
/// `MAX_PLANAR` (4) render per frame.
#[derive(Serialize, Deserialize)]
pub struct PlanarReflection {
    /// Roughness above which the planar reflection has fully faded back to the
    /// reflection probe/env cube (the mirror render is sharp-only).
    pub max_roughness: f32,
}

impl Default for PlanarReflection {
    fn default() -> Self {
        Self { max_roughness: 0.35 }
    }
}

impl TypedComponent for PlanarReflection {
    const NAME: &'static str = "Planar Reflection";
}

/// Per-instance material overrides (Unity per-instance properties). Read by
/// the forward pass only when the material opts in
/// (`MaterialFeatures::per_instance`); objects without one use the material's
/// own values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PerInstance {
    /// Per-instance base colour; `None` = use the material's base colour.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<[f32; 4]>,
    /// Per-instance material parameters (metallic, roughness, toon steps,
    /// toon/PBR blend); `None` = use the material's values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub param: Option<[f32; 4]>,
}

impl TypedComponent for PerInstance {
    const NAME: &'static str = "Per Instance";
}

/// How a spatial audio source quietens with distance.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum AudioRolloff {
    Linear,
    Logarithmic,
}

impl AudioRolloff {
    pub fn label(self) -> &'static str {
        match self {
            Self::Linear => "Linear",
            Self::Logarithmic => "Logarithmic",
        }
    }
    pub const ALL: [AudioRolloff; 2] = [Self::Linear, Self::Logarithmic];
}

/// A sound emitter. Spatial sources attenuate with distance to the
/// AudioListener; non-spatial play at a constant volume.
#[derive(Serialize, Deserialize)]
pub struct AudioSource {
    pub clip: String,
    pub play_on_start: bool,
    pub looping: bool,
    pub volume: f32,
    pub pitch: f32,
    pub spatial: bool,
    pub min_distance: f32,
    pub max_distance: f32,
    pub rolloff: AudioRolloff,
    /// Mixer bus (project Audio Mixer) this source plays through.
    #[serde(default = "default_bus")]
    pub bus: String,
    /// Spatial: render through the listener's HRTF (binaural) rather than
    /// plain stereo panning. Needs the `steam-audio` engine feature.
    #[serde(default = "default_true")]
    pub hrtf: bool,
    /// Spatial: high frequencies fade with distance.
    #[serde(default = "default_true")]
    pub air_absorption: bool,
    /// Spatial: scene geometry between source and listener muffles it.
    #[serde(default = "default_true")]
    pub occlusion: bool,
    /// 0 = omnidirectional, 0.5 = cardioid, 1 = dipole along the object's
    /// forward (-Z) axis.
    #[serde(default)]
    pub directivity: f32,
    /// How narrow the directivity lobe is (higher = tighter beam).
    #[serde(default = "default_one")]
    pub directivity_sharpness: f32,
    /// Spatial: simulate this source's own reflections (reverb that depends
    /// on where it is). Off = the listener-centric reverb, if enabled.
    #[serde(default)]
    pub reflections: bool,
    /// Spatial: sound bends around corners using baked pathing probes.
    #[serde(default)]
    pub pathing: bool,
    /// The clip is an ambisonic recording of this order (1..=3, ACN / N3D);
    /// 0 = a regular mono / stereo clip. Ambisonic clips rotate with the
    /// listener instead of being placed at the object.
    #[serde(default)]
    pub ambisonic_order: u8,
    /// Spatial: encode to ambisonics and decode, instead of direct HRTF.
    #[serde(default)]
    pub spatialize_via_ambisonics: bool,
}

fn default_bus() -> String {
    "sfx".into()
}

fn default_true() -> bool {
    true
}

fn default_one() -> f32 {
    1.0
}

impl Default for AudioSource {
    fn default() -> Self {
        Self {
            clip: String::new(),
            play_on_start: true,
            looping: false,
            volume: 1.0,
            pitch: 1.0,
            spatial: true,
            min_distance: 1.0,
            max_distance: 20.0,
            rolloff: AudioRolloff::Logarithmic,
            bus: default_bus(),
            hrtf: true,
            air_absorption: true,
            occlusion: true,
            directivity: 0.0,
            directivity_sharpness: 1.0,
            reflections: false,
            pathing: false,
            ambisonic_order: 0,
            spatialize_via_ambisonics: false,
        }
    }
}

static AUDIO_ACTIONS: &[ActionDesc] = &[
    ActionDesc::new("Play", "Play", &[], true),
    ActionDesc::new("Stop", "Stop", &[], true),
    ActionDesc::new("SetVolume", "Set Volume", &[ParamDesc::float("volume", 1.0)], true),
];

impl TypedComponent for AudioSource {
    const NAME: &'static str = "Audio Source";

    fn actions(&self) -> &'static [ActionDesc] {
        AUDIO_ACTIONS
    }
    fn invoke_action(&mut self, ctx: &mut ActionCtx, name: &str, args: &[ArgValue]) -> Result<(), String> {
        match name {
            "Play" => ctx.commands.push(ComponentCommand::PlayAudio(ctx.self_id)),
            "Stop" => ctx.commands.push(ComponentCommand::StopAudio(ctx.self_id)),
            "SetVolume" => self.volume = args[0].as_f32().max(0.0),
            _ => return Err(actions::unknown_action(Self::NAME, name)),
        }
        Ok(())
    }
}

/// A post-processing Volume (Unity-style): references a `.postfx` profile and
/// applies it globally or within a local box, blended by priority/weight. The
/// camera blends every volume affecting it into one effective profile.
#[derive(Serialize, Deserialize)]
pub struct VolumeComponent {
    /// Project-relative path to the `.postfx` profile this volume applies.
    pub profile: String,
    /// Global affects the whole scene; local only within the box (object
    /// transform + `half_extents`), faded in over `blend_distance`.
    pub global: bool,
    /// Higher priority blends last (wins ties).
    pub priority: f32,
    /// Maximum contribution, 0..1.
    pub weight: f32,
    /// Local: world-units over which the effect fades in approaching the box.
    pub blend_distance: f32,
    /// Local box half-size (object-local units).
    pub half_extents: [f32; 3],
}

impl Default for VolumeComponent {
    fn default() -> Self {
        Self {
            profile: String::new(),
            global: true,
            priority: 0.0,
            weight: 1.0,
            blend_distance: 1.0,
            half_extents: [5.0, 5.0, 5.0],
        }
    }
}

impl TypedComponent for VolumeComponent {
    const NAME: &'static str = "Volume (Post FX)";
}

/// Marks the object whose transform is the "ears" for spatial audio.
#[derive(Serialize, Deserialize, Default)]
pub struct AudioListener {}

impl TypedComponent for AudioListener {
    const NAME: &'static str = "Audio Listener";
}

/// Number of collision layers (mirrors the layer-collision matrix the physics
/// engine will consult).
pub const COLLISION_LAYERS: u32 = 32;

/// Axis-aligned (object space) box collision zone. Authoring data for the
/// physics engine.
#[derive(Serialize, Deserialize)]
pub struct BoxCollider {
    pub center: [f32; 3],
    pub size: [f32; 3],
    pub is_trigger: bool,
    pub layer: u32,
    /// Physics material (`.physmat`, project-relative; "" = the body's or project default).
    #[serde(default)]
    pub material: String,
}

impl Default for BoxCollider {
    fn default() -> Self {
        Self {
            center: [0.0; 3],
            size: [1.0; 3],
            is_trigger: false,
            layer: 0,
            material: String::new(),
        }
    }
}

impl TypedComponent for BoxCollider {
    const NAME: &'static str = "Box Collider";
}

/// Sphere collision zone (center offset + radius).
#[derive(Serialize, Deserialize)]
pub struct SphereCollider {
    pub center: [f32; 3],
    pub radius: f32,
    pub is_trigger: bool,
    pub layer: u32,
    /// Physics material (`.physmat`, project-relative; "" = the body's or project default).
    #[serde(default)]
    pub material: String,
}

impl Default for SphereCollider {
    fn default() -> Self {
        Self {
            center: [0.0; 3],
            radius: 0.5,
            is_trigger: false,
            layer: 0,
            material: String::new(),
        }
    }
}

impl TypedComponent for SphereCollider {
    const NAME: &'static str = "Sphere Collider";
}

/// Axis a capsule runs along (local).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ColliderAxis {
    X,
    #[default]
    Y,
    Z,
}

impl ColliderAxis {
    pub const ALL: [ColliderAxis; 3] = [ColliderAxis::X, ColliderAxis::Y, ColliderAxis::Z];
    pub fn label(self) -> &'static str {
        match self {
            ColliderAxis::X => "X",
            ColliderAxis::Y => "Y",
            ColliderAxis::Z => "Z",
        }
    }
    pub fn vector(self) -> Vec3 {
        match self {
            ColliderAxis::X => Vec3::X,
            ColliderAxis::Y => Vec3::Y,
            ColliderAxis::Z => Vec3::Z,
        }
    }
}

/// A capsule collision shape: `height` end to end (caps included) along
/// `direction`, `radius` thick.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CapsuleCollider {
    pub center: [f32; 3],
    pub radius: f32,
    pub height: f32,
    pub direction: ColliderAxis,
    pub is_trigger: bool,
    pub layer: u32,
    /// Physics material (`.physmat`, project-relative; "" = the body's or project default).
    pub material: String,
}

impl Default for CapsuleCollider {
    fn default() -> Self {
        Self { center: [0.0; 3], radius: 0.5, height: 2.0, direction: ColliderAxis::Y, is_trigger: false, layer: 0, material: String::new() }
    }
}

impl TypedComponent for CapsuleCollider {
    const NAME: &'static str = "Capsule Collider";
}

/// Collision from the object's own mesh (convex hull or exact triangles).
#[derive(Serialize, Deserialize)]
pub struct MeshCollider {
    /// One convex hull even when static (static meshes otherwise collide
    /// with their exact triangles).
    pub convex: bool,
    /// Moving bodies: split into convex pieces that follow a concave shape
    /// (a couch's seat and arms) instead of one hull around it all.
    #[serde(default)]
    pub follow_shape: bool,
    /// How closely hulls and pieces hug the mesh (more corners and pieces
    /// cost more to build and simulate).
    #[serde(default)]
    pub detail: ColliderDetail,
    /// Use collision pieces modelled with the mesh (`UCX_`, `UBX_`, `USP_`,
    /// `UCP_` objects) when it has any, instead of building from the mesh.
    #[serde(default = "default_true")]
    pub use_authored: bool,
    pub is_trigger: bool,
    pub layer: u32,
    /// Physics material (`.physmat`, project-relative; "" = the body's or project default).
    #[serde(default)]
    pub material: String,
}

/// Mesh Collider fidelity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ColliderDetail {
    Low,
    #[default]
    Medium,
    High,
}

impl ColliderDetail {
    pub const ALL: [ColliderDetail; 3] = [Self::Low, Self::Medium, Self::High];
    pub fn label(self) -> &'static str {
        match self {
            Self::Low => "Low",
            Self::Medium => "Medium",
            Self::High => "High",
        }
    }
}

impl Default for MeshCollider {
    fn default() -> Self {
        Self {
            convex: false,
            follow_shape: true,
            detail: ColliderDetail::Medium,
            use_authored: true,
            is_trigger: false,
            layer: 0,
            material: String::new(),
        }
    }
}

impl TypedComponent for MeshCollider {
    const NAME: &'static str = "Mesh Collider";
}

/// Heightfield terrain: a square, procedurally tessellated mesh whose vertex
/// heights come from a `(resolution + 1)²` grid. The engine builds the render
/// mesh and (optionally) a heightfield collider from this in Play; the
/// component itself stores only data, so it serializes cleanly.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct Terrain {
    /// World-space X/Z extent of the square, in meters.
    pub size: f32,
    /// Quads per side; the height grid has `(resolution + 1)²` entries.
    pub resolution: u32,
    /// Vertex heights in meters, row-major over Z then X
    /// (`heights[z * (resolution + 1) + x]`), length `(resolution + 1)²`.
    pub heights: Vec<f32>,
    /// Project-relative `.material` path ("" = project default).
    pub material: String,
    /// Project-relative heightmap image to import into `heights` (PNG, JPG,
    /// EXR or HDR; luminance is used). Empty = no source (sculpted heights).
    #[serde(default)]
    pub heightmap: String,
    /// Metres the heightmap's full 0..1 luminance range spans.
    #[serde(default = "default_heightmap_scale")]
    pub heightmap_scale: f32,
    /// Metres added to the imported height (0..1 black is at this height).
    #[serde(default)]
    pub heightmap_offset: f32,
    /// The `heightmap` source the heights were last imported from ("" = none),
    /// so changing the source re-imports. Not meant to be edited by hand.
    #[serde(default)]
    pub heightmap_imported_path: String,
    /// The scale/offset the last import applied; changing them re-imports.
    #[serde(default)]
    pub heightmap_imported_scale: f32,
    #[serde(default)]
    pub heightmap_imported_offset: f32,
    /// Per-vertex painted texture-layer weights (RGBA = up to 4 layers),
    /// row-major over Z then X like `heights`.
    /// Empty means "not painted yet" and defaults to layer 0 at runtime/editor.
    #[serde(default)]
    pub texture_weights: Vec<[f32; 4]>,
}

fn default_heightmap_scale() -> f32 {
    10.0
}

impl Default for Terrain {
    fn default() -> Self {
        let resolution = 32;
        Self {
            size: 100.0,
            resolution,
            heights: vec![0.0; (resolution as usize + 1).pow(2)],
            material: "materials/terrain_splat.material".into(),
            heightmap: String::new(),
            heightmap_scale: 10.0,
            heightmap_offset: 0.0,
            heightmap_imported_path: String::new(),
            heightmap_imported_scale: 0.0,
            heightmap_imported_offset: 0.0,
            texture_weights: Vec::new(),
        }
    }
}

impl TypedComponent for Terrain {
    const NAME: &'static str = "Terrain";
}

/// Collision from a sibling [`Terrain`] heightfield: exact triangles built
/// from the terrain's size, resolution and heights so physics walks on the
/// landscape. Add one alongside the Terrain component.
#[derive(Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct TerrainCollider {
    pub is_trigger: bool,
    pub layer: u32,
    /// Physics material (`.physmat`, project-relative; "" = the body's or project default).
    pub material: String,
}

impl TypedComponent for TerrainCollider {
    const NAME: &'static str = "Terrain Collider";
}

/// One foliage layer scattered over a terrain: instanced copies of a model,
/// placed deterministically from a seed.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FoliageLayer {
    /// Project-relative model path to instance.
    pub model: String,
    /// Target instances per square metre.
    pub density: f32,
    /// Per-instance uniform scale range.
    pub scale_min: f32,
    pub scale_max: f32,
    /// Random yaw range in radians (± this about the surface up).
    pub yaw: f32,
    /// Skip placements on slopes steeper than this (radians; 0 = any).
    pub max_slope: f32,
    /// Orient the instance's up axis along the surface normal.
    pub align_to_normal: bool,
}

impl Default for FoliageLayer {
    fn default() -> Self {
        Self {
            model: String::new(),
            density: 1.0,
            scale_min: 1.0,
            scale_max: 1.0,
            yaw: std::f32::consts::PI,
            max_slope: 0.0,
            align_to_normal: true,
        }
    }
}

/// Foliage scattered on a [`Terrain`] (a component on the terrain object).
#[derive(Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct TerrainFoliage {
    pub layers: Vec<FoliageLayer>,
    pub seed: u32,
}

impl TypedComponent for TerrainFoliage {
    const NAME: &'static str = "Terrain Foliage";
}

/// One placed foliage instance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FoliageInstance {
    pub position: [f32; 3],
    /// Unit quaternion `[x, y, z, w]`.
    pub rotation: [f32; 4],
    pub scale: f32,
}

/// Cap on scattered instances per layer (guards a large density × area).
const MAX_FOLIAGE_INSTANCES: usize = 1_000_000;

/// Deterministic xorshift64 RNG: a seed always scatters the same instances.
struct FoliageRng(u64);

impl FoliageRng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(0xBF58_476D_1CE4_E5B9))
    }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}

/// Bilinear height at continuous grid coords `(gx, gz)` (clamped to the grid).
fn foliage_height(heights: &[f32], n: usize, gx: f32, gz: f32) -> f32 {
    let x0 = gx.floor().clamp(0.0, (n - 1) as f32) as usize;
    let z0 = gz.floor().clamp(0.0, (n - 1) as f32) as usize;
    let x1 = (x0 + 1).min(n - 1);
    let z1 = (z0 + 1).min(n - 1);
    let (fx, fz) = (gx - x0 as f32, gz - z0 as f32);
    let h00 = heights[z0 * n + x0];
    let h10 = heights[z0 * n + x1];
    let h01 = heights[z1 * n + x0];
    let h11 = heights[z1 * n + x1];
    (h00 * (1.0 - fx) + h10 * fx) * (1.0 - fz) + (h01 * (1.0 - fx) + h11 * fx) * fz
}

/// Surface normal at the nearest grid point, matching `heightfield_mesh`'s
/// clamped central difference.
fn foliage_normal(heights: &[f32], n: usize, res: usize, xi: i64, zi: i64, step: f32) -> Vec3 {
    let sample = |x: i64, z: i64| {
        let x = x.clamp(0, res as i64) as usize;
        let z = z.clamp(0, res as i64) as usize;
        heights[z * n + x]
    };
    let dh_dx = sample(xi + 1, zi) - sample(xi - 1, zi);
    let dh_dz = sample(xi, zi + 1) - sample(xi, zi - 1);
    Vec3::new(-dh_dx, 2.0 * step, -dh_dz).normalize_or(Vec3::Y)
}

/// Scatter one foliage layer over `terrain`, deterministically from `seed`.
/// Returns the placed instances (fewer than the density target where the slope
/// limit rejects spots).
pub fn scatter_foliage(terrain: &Terrain, layer: &FoliageLayer, seed: u32) -> Vec<FoliageInstance> {
    let res = terrain.resolution as usize;
    let n = res + 1;
    if terrain.heights.len() != n * n || layer.density <= 0.0 {
        return Vec::new();
    }
    let step = terrain.size / terrain.resolution as f32;
    let half = terrain.size * 0.5;
    let count = ((layer.density * terrain.size * terrain.size).round() as usize).min(MAX_FOLIAGE_INSTANCES);
    let mut rng = FoliageRng::new(seed as u64);
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let wx = rng.range(-half, half);
        let wz = rng.range(-half, half);
        let gx = (wx + half) / step;
        let gz = (wz + half) / step;
        let height = foliage_height(&terrain.heights, n, gx, gz);
        let normal = foliage_normal(&terrain.heights, n, res, gx.round() as i64, gz.round() as i64, step);
        if layer.max_slope > 0.0 && normal.y.acos() > layer.max_slope {
            continue;
        }
        let scale = rng.range(layer.scale_min, layer.scale_max);
        let yaw = rng.range(-layer.yaw, layer.yaw);
        let up = if layer.align_to_normal { normal } else { Vec3::Y };
        // Yaw about the local up first, then tilt the up axis onto the surface,
        // so the instance keeps its yaw about the aligned normal.
        let rotation = Quat::from_rotation_arc(Vec3::Y, up) * Quat::from_rotation_y(yaw);
        out.push(FoliageInstance {
            position: [wx, height, wz],
            rotation: rotation.normalize().to_array(),
            scale,
        });
    }
    out
}

/// A decal projector: an orthographic box that projects a texture onto nearby
/// geometry (bullet holes, scorch marks, puddles). The box is the object's
/// local X/Y extents and a `depth` along its -Z (the projection direction).
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DecalProjector {
    /// Project-relative texture path (the decal image).
    pub texture: String,
    /// Orthographic box width/height in metres (local X/Y).
    pub size: [f32; 2],
    /// Projection depth along the projector's -Z (local), in metres.
    pub depth: f32,
    /// Fade the decal out over this fraction of `depth` at the far end.
    pub fade: f32,
    /// UV tiling over the box.
    pub tiling: [f32; 2],
    /// Linear RGBA tint multiplied with the decal texture.
    #[serde(default = "default_decal_tint")]
    pub tint: [f32; 4],
}

fn default_decal_tint() -> [f32; 4] {
    [1.0, 1.0, 1.0, 1.0]
}

impl Default for DecalProjector {
    fn default() -> Self {
        Self {
            texture: String::new(),
            size: [1.0, 1.0],
            depth: 1.0,
            fade: 0.1,
            tiling: [1.0, 1.0],
            tint: default_decal_tint(),
        }
    }
}

impl TypedComponent for DecalProjector {
    const NAME: &'static str = "Decal Projector";
}

/// Project a world point into a decal projector's UV space (0..1, origin at the
/// box's -X/-Y corner). `projector` is the object's world transform; `None`
/// when the point falls outside the orthographic box.
pub fn decal_uv(world: Vec3, projector: Mat4, size: [f32; 2], depth: f32) -> Option<[f32; 2]> {
    let local = projector.inverse().transform_point3(world);
    let hx = size[0] * 0.5;
    let hy = size[1] * 0.5;
    if local.x < -hx || local.x > hx || local.y < -hy || local.y > hy || local.z < -depth || local.z > 0.0 {
        return None;
    }
    Some([(local.x / size[0]) + 0.5, (local.y / size[1]) + 0.5])
}

/// Volumetric cloud settings: the density field a cloud raymarch pass samples.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VolumetricClouds {
    /// Cloud layer base altitude (metres, world Y).
    pub base: f32,
    /// Cloud layer thickness (metres).
    pub thickness: f32,
    /// Global coverage 0..1 (0 = clear sky).
    pub coverage: f32,
    /// Peak density inside a cloud body.
    pub density: f32,
    /// Wind direction (world XZ, used as a unit vector) and speed (m/s).
    pub wind: [f32; 3],
    /// Noise scale (larger = bigger, puffier clouds).
    pub scale: f32,
    /// Noise seed.
    pub seed: u32,
}

impl Default for VolumetricClouds {
    fn default() -> Self {
        Self {
            base: 1500.0,
            thickness: 800.0,
            coverage: 0.5,
            density: 1.0,
            wind: [1.0, 0.0, 0.0],
            scale: 0.0008,
            seed: 0,
        }
    }
}

/// Deterministic 3D value noise in [0,1].
fn cloud_noise(p: Vec3, seed: u32) -> f32 {
    let (xi, yi, zi) = (p.x.floor() as i32, p.y.floor() as i32, p.z.floor() as i32);
    let (xf, yf, zf) = (p.x - xi as f32, p.y - yi as f32, p.z - zi as f32);
    let smooth = |t: f32| t * t * (3.0 - 2.0 * t);
    let (u, v, w) = (smooth(xf), smooth(yf), smooth(zf));
    let hash = |x: i32, y: i32, z: i32| -> f32 {
        let mut n = (seed as i32)
            .wrapping_add(x.wrapping_mul(1619))
            .wrapping_add(y.wrapping_mul(31337))
            .wrapping_add(z.wrapping_mul(6971));
        n = (n ^ (n >> 13)).wrapping_mul(1274126177);
        n ^= n >> 16;
        (n & 0x7FFFFF) as f32 / 0x7FFFFF as f32
    };
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let c000 = hash(xi, yi, zi);
    let c100 = hash(xi + 1, yi, zi);
    let c010 = hash(xi, yi + 1, zi);
    let c110 = hash(xi + 1, yi + 1, zi);
    let c001 = hash(xi, yi, zi + 1);
    let c101 = hash(xi + 1, yi, zi + 1);
    let c011 = hash(xi, yi + 1, zi + 1);
    let c111 = hash(xi + 1, yi + 1, zi + 1);
    let x00 = lerp(c000, c100, u);
    let x10 = lerp(c010, c110, u);
    let x01 = lerp(c001, c101, u);
    let x11 = lerp(c011, c111, u);
    lerp(lerp(x00, x10, v), lerp(x01, x11, v), w)
}

/// Four octaves of value noise for puffy cloud structure, roughly in [0,1].
fn cloud_fbm(p: Vec3, seed: u32) -> f32 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    let mut q = p;
    for _ in 0..4 {
        sum += amp * cloud_noise(q, seed);
        amp *= 0.5;
        q *= 2.0;
    }
    sum / 0.9375
}

/// Sample the volumetric cloud density at a world point, in \[0,1\]. This is the
/// deterministic CPU reference the raymarch shader reproduces: a height-graded
/// layer shaped by coverage-modulated fBm noise, advected by wind.
pub fn cloud_density(point: Vec3, clouds: &VolumetricClouds, time: f32) -> f32 {
    let p = point - Vec3::from(clouds.wind) * time;
    let t = (p.y - clouds.base) / clouds.thickness.max(1.0);
    if t <= 0.0 || t >= 1.0 {
        return 0.0;
    }
    let height = 4.0 * t * (1.0 - t);
    let n = cloud_fbm(p * clouds.scale, clouds.seed);
    let lo = (1.0 - clouds.coverage).clamp(0.0, 1.0);
    let s = ((n - lo) / 0.06).clamp(0.0, 1.0);
    let shape = s * s * (3.0 - 2.0 * s);
    (shape * height * clouds.density).clamp(0.0, 1.0)
}

/// Places this object on a bone of an animated model every frame (a weapon in
/// a hand). `avatar` is the model or any ancestor of it: the first enabled
/// animated model under it with a joint named `bone` is used, so a switchable
/// avatar keeps the attachment.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BoneAttach {
    pub avatar: ObjectRef,
    pub bone: String,
    /// Offset in the bone's space, in meters.
    pub position: [f32; 3],
    /// Offset rotation in the bone's space (XYZ Euler, degrees).
    pub rotation: [f32; 3],
}

impl Default for BoneAttach {
    fn default() -> Self {
        Self { avatar: ObjectRef::NONE, bone: "hand_r".into(), position: [0.0; 3], rotation: [0.0; 3] }
    }
}

impl TypedComponent for BoneAttach {
    const NAME: &'static str = "Bone Attach";
}

/// Which stored pose of a [`WeaponHold`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeaponPoseSlot {
    #[default]
    Idle,
    Aim,
}

/// What a [`WeaponPose`] is relative to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeaponFollow {
    /// The anchor bone: sways with the torso (idle breathing, run bob).
    #[default]
    Bone,
    /// The avatar model: moves and turns with the player, no animation sway.
    Avatar,
    /// The world: stays put like a wall or a ledge; the hands reach for it.
    World,
}

/// A weapon placement for a [`WeaponHold`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WeaponPose {
    /// Meters, in the `follow` frame.
    pub position: [f32; 3],
    /// XYZ Euler, degrees, in the `follow` frame.
    pub rotation: [f32; 3],
    pub follow: WeaponFollow,
}

/// Editor request for a [`WeaponHold`], handled by the engine next frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeaponPoseRequest {
    /// Put the weapon at the pose being edited.
    Load,
    /// Save where the weapon is now as the pose being edited.
    Store,
}

/// Holds this object (a weapon) in both hands of an animated avatar. In Play
/// the weapon sits at the idle or aim pose (each relative to `anchor_bone`, the
/// avatar or the world, see [`WeaponFollow`]), blending on the avatar
/// Animator's `aim_parameter` bool.
/// Outside Play the weapon stays where it is moved, so it can be posed with
/// the transform gizmo and stored into a slot. Either way each arm reaches its
/// hand target: a child of the weapon whose transform is where the hand bone
/// goes.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WeaponHold {
    /// The animated model, or any ancestor of it (a switchable avatar).
    pub avatar: ObjectRef,
    /// Camera the weapon aims along in Play (points the gun at the crosshair).
    pub camera: ObjectRef,
    pub anchor_bone: String,
    pub idle: WeaponPose,
    pub aim: WeaponPose,
    pub aim_parameter: String,
    /// Seconds to blend between the idle and aim poses.
    pub blend_time: f32,
    pub right_hand: ObjectRef,
    pub left_hand: ObjectRef,
    /// The pose the editor's Load / Store work on.
    pub editing: WeaponPoseSlot,
    #[serde(skip)]
    pub request: Option<WeaponPoseRequest>,
    /// 0 = idle pose, 1 = aim pose.
    #[serde(skip)]
    pub blend: f32,
    /// The weapon has been put at a pose since loading (the editor starts it
    /// at the pose being edited, then leaves it where it is moved).
    #[serde(skip)]
    pub placed: bool,
}

impl Default for WeaponHold {
    fn default() -> Self {
        Self {
            avatar: ObjectRef::NONE,
            camera: ObjectRef::NONE,
            anchor_bone: "spine_03".into(),
            idle: WeaponPose::default(),
            aim: WeaponPose::default(),
            aim_parameter: "Aiming".into(),
            blend_time: 0.15,
            right_hand: ObjectRef::NONE,
            left_hand: ObjectRef::NONE,
            editing: WeaponPoseSlot::Idle,
            request: None,
            blend: 0.0,
            placed: false,
        }
    }
}

impl WeaponHold {
    pub fn pose(&self, slot: WeaponPoseSlot) -> WeaponPose {
        match slot {
            WeaponPoseSlot::Idle => self.idle,
            WeaponPoseSlot::Aim => self.aim,
        }
    }

    pub fn pose_mut(&mut self, slot: WeaponPoseSlot) -> &mut WeaponPose {
        match slot {
            WeaponPoseSlot::Idle => &mut self.idle,
            WeaponPoseSlot::Aim => &mut self.aim,
        }
    }
}

impl TypedComponent for WeaponHold {
    const NAME: &'static str = "Weapon Hold";
}

/// How the physics engine simulates a body. Dynamic bodies fall and react to
/// forces/collisions; Kinematic are moved by gameplay but push dynamics;
/// Fixed never move (static geometry / level colliders).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum BodyKind {
    Dynamic,
    Kinematic,
    Fixed,
}

impl BodyKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Dynamic => "Dynamic",
            Self::Kinematic => "Kinematic",
            Self::Fixed => "Fixed",
        }
    }
    pub const ALL: [BodyKind; 3] = [Self::Dynamic, Self::Kinematic, Self::Fixed];
}

/// Makes an object participate in the physics simulation (paired with a
/// collider for its shape). The engine builds a rapier rigid body from this on
/// Play and writes the simulated transform back each step.
#[derive(Serialize, Deserialize)]
pub struct RigidBody {
    pub kind: BodyKind,
    /// Mass in kg (Dynamic only; 0 = from the colliders' volume and the
    /// physics material's density, else 500 kg/m^3).
    pub mass: f32,
    /// Bounciness 0..1.
    pub restitution: f32,
    pub friction: f32,
    /// Per-body gravity multiplier (0 = floats).
    pub gravity_scale: f32,
    /// Physics material (`.physmat`, project-relative). When set it replaces
    /// restitution and friction above, and its colliders inherit it.
    #[serde(default)]
    pub material: String,
    /// Engine-filled in Play: the simulated mass (kg) and collider volume
    /// (m^3), so the inspector can show the body's real density.
    #[serde(skip)]
    pub measured: Option<(f32, f32)>,
}

impl Default for RigidBody {
    fn default() -> Self {
        Self {
            kind: BodyKind::Dynamic,
            mass: 0.0,
            restitution: 0.0,
            friction: 0.5,
            gravity_scale: 1.0,
            material: String::new(),
            measured: None,
        }
    }
}

static RIGID_BODY_ACTIONS: &[ActionDesc] = &[
    ActionDesc::new("ApplyImpulse", "Apply Impulse", &[ParamDesc::vec3("impulse", [0.0, 5.0, 0.0])], true),
    ActionDesc::new("SetVelocity", "Set Velocity", &[ParamDesc::vec3("velocity", [0.0; 3])], true),
];

impl TypedComponent for RigidBody {
    const NAME: &'static str = "Rigid Body";

    fn actions(&self) -> &'static [ActionDesc] {
        RIGID_BODY_ACTIONS
    }
    fn invoke_action(&mut self, ctx: &mut ActionCtx, name: &str, args: &[ArgValue]) -> Result<(), String> {
        let id = ctx.self_id;
        match name {
            "ApplyImpulse" => ctx.commands.push(ComponentCommand::AddImpulse { id, impulse: args[0].as_vec3() }),
            "SetVelocity" => ctx.commands.push(ComponentCommand::SetVelocity { id, linear: args[0].as_vec3() }),
            _ => return Err(actions::unknown_action(Self::NAME, name)),
        }
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum JointKind {
    /// Welds the two bodies together.
    Fixed,
    /// Rotation about one axis (doors, wheels, levers).
    Hinge,
    /// Free rotation about the anchor (ragdoll shoulders, chains).
    Ball,
    /// Translation along one axis (pistons, drawers).
    Slider,
    /// Pulls the anchors toward `rest_length` (soft connection).
    Spring,
    /// Keeps the anchors within `rest_length` (cables, tethers).
    Rope,
}

impl JointKind {
    pub const ALL: [JointKind; 6] = [Self::Fixed, Self::Hinge, Self::Ball, Self::Slider, Self::Spring, Self::Rope];
    pub fn label(self) -> &'static str {
        match self {
            Self::Fixed => "Fixed",
            Self::Hinge => "Hinge",
            Self::Ball => "Ball",
            Self::Slider => "Slider",
            Self::Spring => "Spring",
            Self::Rope => "Rope",
        }
    }
}

/// Connects this object's rigid body to another body (or to its start point
/// in the world when `connected` is unset).
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct PhysicsJoint {
    pub kind: JointKind,
    pub connected: ObjectRef,
    /// Anchor in this object's local space.
    pub anchor: [f32; 3],
    /// Hinge / slider axis in this object's local space.
    pub axis: [f32; 3],
    /// Hinge: degrees; slider: meters.
    pub limits_enabled: bool,
    pub limits: [f32; 2],
    /// Hinge / slider motor (deg/s or m/s).
    pub motor_enabled: bool,
    pub motor_speed: f32,
    pub motor_max_force: f32,
    /// Spring / rope length (0 = the distance at start).
    pub rest_length: f32,
    pub stiffness: f32,
    pub damping: f32,
    /// Let the two connected bodies collide with each other.
    pub collide_connected: bool,
}

impl Default for PhysicsJoint {
    fn default() -> Self {
        Self {
            kind: JointKind::Hinge,
            connected: ObjectRef::NONE,
            anchor: [0.0; 3],
            axis: [0.0, 1.0, 0.0],
            limits_enabled: false,
            limits: [-90.0, 90.0],
            motor_enabled: false,
            motor_speed: 90.0,
            motor_max_force: 100.0,
            rest_length: 0.0,
            stiffness: 50.0,
            damping: 2.0,
            collide_connected: false,
        }
    }
}

impl TypedComponent for PhysicsJoint {
    const NAME: &'static str = "Joint";
}

/// Emits camera-facing particles (smoke, sparks, magic, dust).
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct ParticleEmitter {
    /// Particles per second while playing.
    pub rate: f32,
    /// Particles emitted at once when the emitter starts.
    pub burst: u32,
    /// Seconds each particle lives [min, max].
    pub lifetime: [f32; 2],
    /// Initial speed [min, max] along the emit direction.
    pub speed: [f32; 2],
    /// Emit direction in local space and cone half-angle (degrees).
    pub direction: [f32; 3],
    pub cone_angle: f32,
    /// Spawn inside a sphere of this radius.
    pub shape_radius: f32,
    /// Multiplier on world gravity (negative floats upward).
    pub gravity: f32,
    pub drag: f32,
    pub size: [f32; 2],
    pub color_start: [f32; 4],
    pub color_end: [f32; 4],
    /// Project-relative texture (empty = soft round dot).
    pub texture: String,
    pub max_particles: u32,
    pub looping: bool,
    /// Seconds a non-looping emitter emits for.
    pub duration: f32,
    pub play_on_start: bool,
    /// Particles stay where they were emitted (else they follow the object).
    pub world_space: bool,
    #[serde(skip)]
    pub control: ParticleControl,
}

/// Gameplay requests to a running emitter (consumed by the engine).
#[derive(Clone, Copy, Debug, Default)]
pub struct ParticleControl {
    pub play: bool,
    pub stop: bool,
    pub clear: bool,
    pub emit: u32,
    /// Live particle count (filled in by the engine).
    pub alive: u32,
    pub playing: bool,
}

impl Default for ParticleEmitter {
    fn default() -> Self {
        Self {
            rate: 20.0,
            burst: 0,
            lifetime: [1.0, 2.0],
            speed: [1.0, 2.0],
            direction: [0.0, 1.0, 0.0],
            cone_angle: 20.0,
            shape_radius: 0.05,
            gravity: 0.0,
            drag: 0.0,
            size: [0.2, 0.05],
            color_start: [1.0, 0.8, 0.4, 1.0],
            color_end: [1.0, 0.2, 0.05, 0.0],
            texture: String::new(),
            max_particles: 500,
            looping: true,
            duration: 2.0,
            play_on_start: true,
            world_space: true,
            control: ParticleControl::default(),
        }
    }
}

impl TypedComponent for ParticleEmitter {
    const NAME: &'static str = "Particle Emitter";
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum PortalMode {
    /// Replace the current scene (level change).
    Switch,
    /// Stream the scene in alongside the current one.
    Additive,
    /// Stream a previously added scene back out.
    Unload,
}

impl PortalMode {
    pub const ALL: [PortalMode; 3] = [Self::Switch, Self::Additive, Self::Unload];
    pub fn label(self) -> &'static str {
        match self {
            Self::Switch => "Switch scene",
            Self::Additive => "Load additive",
            Self::Unload => "Unload",
        }
    }
}

/// Seamless scene transition: preloads `scene` in the background when the
/// activator comes within `preload_distance`, then switches / streams it when
/// the activator enters this object's trigger collider (add a Box/Sphere
/// Collider with "Is trigger"). Build the transition (a fall, a corridor, a
/// crack in the wall) so the activator spends the preload time inside it.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct ScenePortal {
    pub scene: String,
    pub mode: PortalMode,
    /// Object that activates the portal (unset = any object with a Pawn).
    pub activator: ObjectRef,
    /// Start preloading within this distance (0 = as soon as play starts).
    pub preload_distance: f32,
    /// Carry the activator into the new scene (Switch mode). It respawns at a
    /// matching SpawnPoint if its Pawn has a spawn tag.
    pub keep_activator: bool,
    /// Fire once, then disable.
    pub one_shot: bool,
    #[serde(skip)]
    preloaded: bool,
    #[serde(skip)]
    fired: bool,
}

impl Default for ScenePortal {
    fn default() -> Self {
        Self {
            scene: String::new(),
            mode: PortalMode::Switch,
            activator: ObjectRef::NONE,
            preload_distance: 30.0,
            keep_activator: true,
            one_shot: true,
            preloaded: false,
            fired: false,
        }
    }
}

impl ScenePortal {
    fn activator_index(&self, ctx: &ComponentCtx) -> Option<usize> {
        match self.activator.id() {
            Some(_) => ctx.resolve(self.activator),
            None => (0..ctx.object_ids.len()).find(|&i| {
                ctx.active.get(i).copied().unwrap_or(false)
                    && ctx.lookup.component(i, std::any::TypeId::of::<Pawn>()).is_some()
            }),
        }
    }
}

impl TypedComponent for ScenePortal {
    const NAME: &'static str = "Scene Portal";

    fn update(&mut self, ctx: &mut ComponentCtx) {
        if self.preloaded || self.scene.is_empty() || self.mode == PortalMode::Unload {
            return;
        }
        let near = self.preload_distance <= 0.0
            || self.activator_index(ctx).and_then(|i| ctx.object_position(i)).is_some_and(|p| {
                p.distance(ctx.self_transform().translation) <= self.preload_distance
            });
        if near {
            ctx.preload_scene(self.scene.clone());
            self.preloaded = true;
        }
    }

    fn on_trigger_enter(&mut self, ctx: &mut ComponentCtx, other: ObjectId) {
        if self.scene.is_empty() || (self.one_shot && self.fired) {
            return;
        }
        let Some(act) = self.activator_index(ctx) else {
            return;
        };
        if ctx.object_ids.get(act) != Some(&other) {
            return;
        }
        self.fired = true;
        match self.mode {
            PortalMode::Switch => {
                if self.keep_activator {
                    ctx.keep_on_scene_change(other, true);
                }
                ctx.load_scene(self.scene.clone());
            }
            PortalMode::Additive => ctx.load_scene_additive(self.scene.clone()),
            PortalMode::Unload => ctx.unload_scene(self.scene.clone()),
        }
    }
}

/// Bakes a walkable navigation grid from the physics colliders inside its box
/// (centered on the object) when play starts. Agents path over it.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct NavMeshSurface {
    /// Box size in meters (world axes, centered on the object).
    pub size: [f32; 3],
    pub cell_size: f32,
    pub agent_radius: f32,
    pub agent_height: f32,
    /// Steepest walkable slope in degrees.
    pub max_slope: f32,
    /// Tallest step an agent can climb between cells.
    pub step_height: f32,
    /// Layers whose colliders count as ground / obstacles.
    pub layer_mask: u32,
    /// Editor: overlay the baked walkable cells in the viewport.
    #[serde(skip)]
    pub show_preview: bool,
    /// Editor: re-bake the preview.
    #[serde(skip)]
    pub rebake_preview: bool,
}

impl Default for NavMeshSurface {
    fn default() -> Self {
        Self {
            size: [40.0, 10.0, 40.0],
            cell_size: 0.25,
            agent_radius: 0.35,
            agent_height: 1.8,
            max_slope: 45.0,
            step_height: 0.4,
            layer_mask: u32::MAX,
            show_preview: false,
            rebake_preview: false,
        }
    }
}

impl TypedComponent for NavMeshSurface {
    const NAME: &'static str = "NavMesh Surface";
}

/// Walks the object along navmesh paths. Set a destination with
/// `ctx.set_destination(agent, point)`; moves through a CharacterController
/// when the object has one.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct NavAgent {
    pub speed: f32,
    /// Turn rate in degrees per second (0 = don't rotate).
    pub turn_speed: f32,
    /// Stop when this close to the destination.
    pub stopping_distance: f32,
    #[serde(skip)]
    pub state: NavAgentState,
}

#[derive(Clone, Debug, Default)]
pub struct NavAgentState {
    pub destination: Option<Vec3>,
    /// Destination changed; the engine re-plans next update.
    pub replan: bool,
    pub path: Vec<Vec3>,
    pub next: usize,
    /// No path exists to the destination.
    pub unreachable: bool,
    pub velocity: Vec3,
}

impl Default for NavAgent {
    fn default() -> Self {
        Self {
            speed: 3.5,
            turn_speed: 540.0,
            stopping_distance: 0.1,
            state: NavAgentState::default(),
        }
    }
}

impl NavAgent {
    pub fn set_destination(&mut self, target: Vec3) {
        self.state.destination = Some(target);
        self.state.replan = true;
        self.state.unreachable = false;
    }

    pub fn stop(&mut self) {
        self.state.destination = None;
        self.state.path.clear();
        self.state.velocity = Vec3::ZERO;
    }

    /// Has a destination and reached it (or can't reach it).
    pub fn arrived(&self) -> bool {
        self.state.destination.is_none() || self.state.unreachable
    }

    /// Distance left along the current path.
    pub fn remaining_distance(&self, from: Vec3) -> f32 {
        let s = &self.state;
        if s.next >= s.path.len() {
            return 0.0;
        }
        let mut d = from.distance(s.path[s.next]);
        for w in s.path[s.next..].windows(2) {
            d += w[0].distance(w[1]);
        }
        d
    }
}

impl TypedComponent for NavAgent {
    const NAME: &'static str = "Nav Agent";
}

/// Kinematic capsule that walks: moved with `ctx.move_character(delta)`, it
/// slides along walls, climbs steps and slopes, and reports `ctx.is_grounded()`.
/// Gravity is up to the caller (a [`Pawn`] on the same object applies it).
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct Character {
    /// Create the default movement capsule. Disable for a custom collision/movement rig.
    pub use_default_capsule: bool,
    /// Let the movement capsule displace water. Disable when bone-attached colliders do that.
    pub capsule_affects_water: bool,
    pub radius: f32,
    /// Total capsule height, including both caps.
    pub height: f32,
    /// Capsule center relative to the object origin (default puts the feet at the origin).
    pub center: [f32; 3],
    /// Tallest step climbed automatically (0 disables autostep).
    pub step_height: f32,
    /// Steepest walkable slope in degrees.
    pub max_slope: f32,
    /// Keep the character glued to the ground when walking down slopes/steps (meters).
    pub snap_to_ground: f32,
}

impl Default for Character {
    fn default() -> Self {
        Self {
            use_default_capsule: true,
            capsule_affects_water: true,
            radius: 0.35,
            height: 1.8,
            center: [0.0, 0.9, 0.0],
            step_height: 0.3,
            max_slope: 45.0,
            snap_to_ground: 0.2,
        }
    }
}

impl TypedComponent for Character {
    const NAME: &'static str = "Character";
    const ALIASES: &'static [&'static str] = &["Character Controller"];
}

/// Compatibility name for gameplay plugins; new code should use [`Character`].
pub type CharacterController = Character;

/// Plays skeletal animation clips on this object's skinned meshes, with
/// cross-fades. Each animated object gets its own pose, so spawned copies of a
/// model animate independently. Drive it from gameplay with
/// `ctx.play_animation(target, "Run", 0.2)`, or give it an Animator Controller
/// (`.animctrl` state machine) and drive that with parameters
/// (`ctx.set_animator_trigger(target, "Jump")`). Clips may be imported model
/// clips or authored `.anim` assets, which can also animate object hierarchies.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct Animator {
    /// Project-relative `.animctrl` path. Empty = play `clip` / blend trees directly.
    pub controller: String,
    /// Clip, `.anim` path or blend tree to play on start (empty = the model's first clip).
    pub clip: String,
    pub speed: f32,
    pub looping: bool,
    pub play_on_start: bool,
    /// Named blend trees, played like clips (`play("Locomotion", 0.2)`), mixing
    /// clips by float parameters (`set_float("Speed", v)`).
    pub blend_trees: Vec<BlendTree>,
    /// Named markers on clips; reaching one calls `on_animation_event` on the
    /// object's components and emits an [`AnimationEvent`].
    pub events: Vec<AnimEvent>,
    /// Move the object by the root bone's horizontal motion (through its
    /// CharacterController when present) instead of letting the clip slide.
    pub root_motion: bool,
    /// Skin on the GPU (compute). Off = CPU skinning.
    pub gpu_skinning: bool,
    /// Parameters fed every frame from movement, physics or another Animator.
    pub drivers: Vec<ParamDriver>,
    /// Which clock drives the Animator.
    pub update_mode: AnimatorUpdateMode,
    /// What it skips while the object is not visible.
    pub culling: AnimatorCullingMode,
    #[serde(skip)]
    pub state: AnimatorState,
}

/// Which clock advances an [`Animator`].
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AnimatorUpdateMode {
    /// The frame's (time-scaled) delta.
    #[default]
    Normal,
    /// Whole fixed steps (the physics rate), holding the pose in between.
    Fixed,
    /// The real frame delta, ignoring the game's time scale.
    UnscaledTime,
}

impl AnimatorUpdateMode {
    pub const ALL: [AnimatorUpdateMode; 3] = [Self::Normal, Self::Fixed, Self::UnscaledTime];
    pub fn label(self) -> &'static str {
        match self {
            Self::Normal => "Normal",
            Self::Fixed => "Animate Physics",
            Self::UnscaledTime => "Unscaled Time",
        }
    }
}

/// What an [`Animator`] stops doing while its object is culled (none of its
/// renderers can be seen: camera, shadows, XR eyes). Skinning, blendshapes and
/// the GPU pose upload stop in every mode while nothing can see the object.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AnimatorCullingMode {
    /// The skeleton keeps being posed offscreen (bones, IK, root motion).
    AlwaysAnimate,
    /// The state machine and events keep running; posing, IK and writing
    /// transforms stop (the last pose stays). Bones something reads (a Bone
    /// Attach or Weapon Hold target) still update at a low rate, root motion
    /// and IK every frame.
    #[default]
    CullUpdateTransforms,
    /// Nothing runs.
    CullCompletely,
}

impl AnimatorCullingMode {
    pub const ALL: [AnimatorCullingMode; 3] = [Self::AlwaysAnimate, Self::CullUpdateTransforms, Self::CullCompletely];
    pub fn label(self) -> &'static str {
        match self {
            Self::AlwaysAnimate => "Always Animate",
            Self::CullUpdateTransforms => "Cull Update Transforms",
            Self::CullCompletely => "Cull Completely",
        }
    }
}

/// A two-bone IK goal for the Animator's IK pass: the limb ending at `tip`
/// (its parent and grandparent are the mid and root joints) reaches `target`.
/// Positions are in the rig's space (the skeleton root's space for skinned
/// models). Layers with `ik_pass` get their state behaviours' `on_state_ik`
/// to set goals; code can set them too.
#[derive(Clone, Debug, PartialEq)]
pub struct IkGoal {
    /// Name of the end joint (wrist, ankle).
    pub tip: String,
    pub target: Vec3,
    /// Where the middle joint (elbow, knee) should bend toward.
    pub pole: Option<Vec3>,
    /// 0..1 share of the solved pose.
    pub weight: f32,
}

/// Which callback a [`StateCallback`] is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateCallbackKind {
    /// The state became current (the frame its transition starts).
    Enter,
    /// A frame in the state, except the one it was entered on.
    Update,
    /// The state's transition away completed (or it was cut).
    Exit,
    /// The root-motion pass.
    Move,
    /// The IK pass.
    Ik,
}

/// A state callback waiting to be delivered to the state's behaviours.
#[derive(Clone, Debug, PartialEq)]
pub struct StateCallback {
    pub layer: usize,
    pub state: String,
    pub kind: StateCallbackKind,
    pub normalized_time: f32,
}

/// What a [`StateBehaviour`] callback sees and may change.
pub struct StateBehaviourCtx<'a> {
    pub layer: usize,
    pub state: &'a str,
    pub normalized_time: f32,
    /// The layer's weight (1 for the base layer).
    pub layer_weight: f32,
    /// The Animator's parameters: read and write.
    pub params: &'a mut HashMap<String, f32>,
    /// `on_state_move`: this frame's root motion (rig space), free to replace.
    pub root_delta: &'a mut Vec3,
    /// `on_state_ik`: set goals here (replace the one for a tip).
    pub ik_goals: &'a mut Vec<IkGoal>,
    /// This behaviour's settings on the state (`AnimState::behaviour_settings`,
    /// RON text; empty when none). Parse on change, not every call.
    pub settings: &'a str,
}

/// Rust callbacks for an Animator Controller state, named on the state
/// (`AnimState::behaviours`) and registered with [`register_state_behaviour`].
/// Each Animator gets its own instance per (layer, state, name), created the
/// first time it is needed. The state's `on_enter` / `on_exit` event strings
/// keep working alongside.
pub trait StateBehaviour: Send + std::marker::Sync {
    fn on_state_enter(&mut self, _ctx: &mut StateBehaviourCtx) {}
    fn on_state_update(&mut self, _ctx: &mut StateBehaviourCtx) {}
    /// Runs when the transition away completes, not when it starts.
    fn on_state_exit(&mut self, _ctx: &mut StateBehaviourCtx) {}
    fn on_state_move(&mut self, _ctx: &mut StateBehaviourCtx) {}
    fn on_state_ik(&mut self, _ctx: &mut StateBehaviourCtx) {}
}

type BehaviourFactory = Box<dyn Fn() -> Box<dyn StateBehaviour> + Send + std::marker::Sync>;

static STATE_BEHAVIOURS: std::sync::Mutex<Vec<(String, BehaviourFactory)>> = std::sync::Mutex::new(Vec::new());

/// Register (or replace) a state behaviour under `name`.
pub fn register_state_behaviour(name: &str, factory: impl Fn() -> Box<dyn StateBehaviour> + Send + std::marker::Sync + 'static) {
    let mut all = STATE_BEHAVIOURS.lock().unwrap_or_else(|e| e.into_inner());
    all.retain(|(n, _)| n != name);
    all.push((name.to_owned(), Box::new(factory)));
}

/// A new instance of the behaviour registered as `name`.
pub fn create_state_behaviour(name: &str) -> Option<Box<dyn StateBehaviour>> {
    let all = STATE_BEHAVIOURS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, f)) = all.iter().find(|(n, _)| n == name) {
        return Some(f());
    }
    // Behaviours that ship with the engine need no registration call.
    (name == PARAMETER_DRIVER).then(|| Box::new(ParameterDriver::new()) as Box<dyn StateBehaviour>)
}

/// Names of every state behaviour that can be created (registered ones, and
/// the engine's own), sorted, for the controller editor's picker.
pub fn state_behaviour_names() -> Vec<String> {
    let all = STATE_BEHAVIOURS.lock().unwrap_or_else(|e| e.into_inner());
    let mut names: Vec<String> = all.iter().map(|(n, _)| n.clone()).collect();
    if !names.iter().any(|n| n == PARAMETER_DRIVER) {
        names.push(PARAMETER_DRIVER.to_owned());
    }
    names.sort();
    names
}

/// An Animator's live behaviour instances. Cloning gives a fresh, empty set
/// (a copied Animator makes its own instances).
#[derive(Default)]
pub struct BehaviourInstances(pub HashMap<(usize, String, String), Box<dyn StateBehaviour>>);

impl Clone for BehaviourInstances {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl std::fmt::Debug for BehaviourInstances {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BehaviourInstances({})", self.0.len())
    }
}

/// Where a [`ParamDriver`] reads its value.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DriverSource {
    /// Horizontal movement speed (m/s) of the source object.
    #[default]
    Speed,
    VerticalSpeed,
    /// Signed speed along the source object's forward (-Z) axis.
    ForwardSpeed,
    /// Signed speed along the source object's right (+X) axis.
    RightSpeed,
    /// 1 while the source's Character Controller touches the ground.
    Grounded,
    /// A parameter of the source object's Animator.
    AnimatorParameter,
}

impl DriverSource {
    pub const ALL: [DriverSource; 6] = [
        Self::Speed,
        Self::VerticalSpeed,
        Self::ForwardSpeed,
        Self::RightSpeed,
        Self::Grounded,
        Self::AnimatorParameter,
    ];
    pub fn label(self) -> &'static str {
        match self {
            Self::Speed => "Speed",
            Self::VerticalSpeed => "Vertical speed",
            Self::ForwardSpeed => "Forward speed",
            Self::RightSpeed => "Right speed",
            Self::Grounded => "Grounded",
            Self::AnimatorParameter => "Animator parameter",
        }
    }
}

/// Links a controller parameter to a live value, so state machines react to
/// movement or to other state machines without code.
#[derive(Serialize, Deserialize, Clone, PartialEq)]
#[serde(default)]
pub struct ParamDriver {
    pub parameter: String,
    pub source: DriverSource,
    /// Source object (unset = this object).
    pub object: ObjectRef,
    /// For `AnimatorParameter`: the source Animator's parameter name.
    pub source_parameter: String,
    pub scale: f32,
}

impl Default for ParamDriver {
    fn default() -> Self {
        Self {
            parameter: String::new(),
            source: DriverSource::Speed,
            object: ObjectRef::NONE,
            source_parameter: String::new(),
            scale: 1.0,
        }
    }
}

/// How a [`BlendTree`] turns its parameters into child weights.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BlendTreeKind {
    /// One parameter; children sit at thresholds (`position[0]`) and blend as tents.
    Simple1D,
    /// Two parameters; one child per direction plus an optional child at the origin.
    SimpleDirectional2D,
    /// Two parameters; any number of children per direction, polar gradient bands.
    FreeformDirectional2D,
    /// Two parameters; any positions, Cartesian gradient bands. Trees saved
    /// before `kind` existed load as this (their 2D mode was a distance blend).
    #[default]
    FreeformCartesian2D,
    /// Every child weighted by its own float parameter.
    Direct,
}

impl BlendTreeKind {
    pub const ALL: [BlendTreeKind; 5] =
        [Self::Simple1D, Self::SimpleDirectional2D, Self::FreeformDirectional2D, Self::FreeformCartesian2D, Self::Direct];
    pub fn label(self) -> &'static str {
        match self {
            Self::Simple1D => "1D",
            Self::SimpleDirectional2D => "2D Simple Directional",
            Self::FreeformDirectional2D => "2D Freeform Directional",
            Self::FreeformCartesian2D => "2D Freeform Cartesian",
            Self::Direct => "Direct",
        }
    }
    pub fn is_2d(self) -> bool {
        matches!(self, Self::SimpleDirectional2D | Self::FreeformDirectional2D | Self::FreeformCartesian2D)
    }
}

/// Mixes clips (or nested trees) by float parameters. Every child plays on one
/// shared normalized time, so clips of different lengths stay in phase.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default)]
pub struct BlendTree {
    pub name: String,
    pub kind: BlendTreeKind,
    pub parameter: String,
    /// Second parameter of the 2D kinds. Empty with the default kind = a 1D
    /// tree (what files from before `kind` held).
    pub parameter_y: String,
    pub motions: Vec<BlendMotion>,
    /// Direct trees: divide the weights by their sum when it exceeds 1.
    pub normalize_direct: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct BlendMotion {
    /// Clip to play (ignored when `tree` is set).
    pub clip: String,
    /// A nested tree in place of a clip.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tree: Option<Box<BlendTree>>,
    /// Parameter value(s) where this child has full weight (y unused in 1D).
    pub position: [f32; 2],
    /// Playback speed factor: the child counts as `length / time_scale` long
    /// when the tree's length is worked out.
    pub time_scale: f32,
    /// Play the child mirrored (humanoid clips).
    pub mirror: bool,
    /// Phase shift in cycles added to the shared normalized time.
    pub cycle_offset: f32,
    /// Direct trees: the float parameter that weights this child.
    pub direct_parameter: String,
}

impl Default for BlendMotion {
    fn default() -> Self {
        Self {
            clip: String::new(),
            tree: None,
            position: [0.0, 0.0],
            time_scale: 1.0,
            mirror: false,
            cycle_offset: 0.0,
            direct_parameter: String::new(),
        }
    }
}

/// A clip reached through a (nested) blend tree.
#[derive(Clone, Debug, PartialEq)]
pub struct BlendLeaf<'a> {
    /// Position among all clip leaves of the tree in depth-first order,
    /// whether or not they carry weight. Stable while the tree is not edited.
    pub index: usize,
    pub clip: &'a str,
    /// Final weight: the product of the weights down the path.
    pub weight: f32,
    pub time_scale: f32,
    pub mirror: bool,
    pub cycle_offset: f32,
}

/// Deepest nesting walked (a guard, trees are finite).
const BLEND_TREE_MAX_DEPTH: usize = 16;
const BT_EPS: f32 = 1e-5;

fn bt_angle(a: [f32; 2], b: [f32; 2]) -> f32 {
    let cross = a[0] * b[1] - a[1] * b[0];
    let dot = a[0] * b[0] + a[1] * b[1];
    cross.atan2(dot)
}

fn bt_len(a: [f32; 2]) -> f32 {
    (a[0] * a[0] + a[1] * a[1]).sqrt()
}

/// 1D weights: a tent per child between its neighbours' thresholds.
pub fn blend_weights_1d(thresholds: &[f32], x: f32) -> Vec<f32> {
    let n = thresholds.len();
    let mut w = vec![0.0; n];
    if n == 0 {
        return w;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| thresholds[a].total_cmp(&thresholds[b]));
    let (first, last) = (order[0], order[n - 1]);
    let x = x.clamp(thresholds[first], thresholds[last]);
    if n == 1 {
        w[first] = 1.0;
        return w;
    }
    let last_window = n - 2;
    for (k, pair) in order.windows(2).enumerate() {
        let (a, b) = (pair[0], pair[1]);
        let (xa, xb) = (thresholds[a], thresholds[b]);
        if x < xb || k == last_window {
            // Equal thresholds: the upper child takes the full weight.
            let t = if xb - xa > 1e-6 { ((x - xa) / (xb - xa)).clamp(0.0, 1.0) } else { 1.0 };
            w[a] = 1.0 - t;
            w[b] = t;
            break;
        }
    }
    w
}

/// 2D Simple Directional: blend the two children angularly next to `p`
/// (largest dot product on each side of it) and give the remainder to the
/// child at the origin, or spread it over all children when there is none.
pub fn blend_weights_simple_directional(pos: &[[f32; 2]], p: [f32; 2]) -> Vec<f32> {
    let n = pos.len();
    let mut w = vec![0.0; n];
    if n == 0 {
        return w;
    }
    if n < 2 {
        w[0] = 1.0;
        return w;
    }
    let at_origin: Vec<usize> = (0..n).filter(|&i| bt_len(pos[i]) < BT_EPS).collect();
    if at_origin.len() > 1 {
        return w;
    }
    let centre = at_origin.first().copied();
    let pl = bt_len(p);
    if pl < BT_EPS {
        match centre {
            Some(c) => w[c] = 1.0,
            None => w.iter_mut().for_each(|v| *v = 1.0 / n as f32),
        }
        return w;
    }
    // Angular neighbours: the best-aligned child on each side of p.
    type Side = Option<(usize, f32)>;
    let (mut left, mut right): (Side, Side) = (None, None);
    for (i, pos_i) in pos.iter().enumerate() {
        let l = bt_len(*pos_i);
        if l < BT_EPS {
            continue;
        }
        let d = [pos_i[0] / l, pos_i[1] / l];
        let cross = d[0] * p[1] - d[1] * p[0];
        let dot = d[0] * p[0] + d[1] * p[1];
        let slot = if cross >= 0.0 { &mut left } else { &mut right };
        if slot.is_none_or(|(_, best)| dot > best) {
            *slot = Some((i, dot));
        }
    }
    let (mut a, mut b, mut c) = (0.0, 0.0, 1.0);
    let mut sides = None;
    if let (Some((l, _)), Some((r, _))) = (left, right) {
        let (pl_, pr) = (pos[l], pos[r]);
        let det = pl_[0] * pr[1] - pl_[1] * pr[0];
        if det.abs() > 1e-9 {
            let ca = (p[0] * pr[1] - p[1] * pr[0]) / det;
            let cb = (pl_[0] * p[1] - pl_[1] * p[0]) / det;
            sides = Some((l, r));
            a = ca.max(0.0);
            b = cb.max(0.0);
            c = 1.0 - a - b;
            if c < 0.0 {
                let s = a + b;
                if s > BT_EPS {
                    a /= s;
                    b /= s;
                }
                c = 0.0;
            } else if c > 1.0 {
                a = 0.0;
                b = 0.0;
                c = 1.0;
            }
        }
    }
    if let Some((l, r)) = sides {
        w[l] += a;
        w[r] += b;
    }
    match centre {
        Some(ci) => w[ci] += c,
        None => w.iter_mut().for_each(|v| *v += c / n as f32),
    }
    w
}

fn bt_normalize(w: &mut [f32], fallback: impl FnOnce() -> usize) {
    let sum: f32 = w.iter().sum();
    if sum > 1e-9 {
        w.iter_mut().for_each(|v| *v /= sum);
    } else if !w.is_empty() {
        w.iter_mut().for_each(|v| *v = 0.0);
        let k = fallback();
        w[k] = 1.0;
    }
}

fn nearest_child(pos: &[[f32; 2]], p: [f32; 2]) -> usize {
    let d = |i: usize| (pos[i][0] - p[0]).powi(2) + (pos[i][1] - p[1]).powi(2);
    (0..pos.len()).min_by(|&a, &b| d(a).total_cmp(&d(b))).unwrap_or(0)
}

/// 2D Freeform Cartesian: each child falls off along the line to every other
/// child (gradient bands), the minimum wins, and the weights are normalised.
pub fn blend_weights_freeform_cartesian(pos: &[[f32; 2]], p: [f32; 2]) -> Vec<f32> {
    let n = pos.len();
    let mut w = vec![0.0; n];
    if n < 2 {
        if n == 1 {
            w[0] = 1.0;
        }
        return w;
    }
    for i in 0..n {
        let mut wi = 1.0f32;
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = [pos[j][0] - pos[i][0], pos[j][1] - pos[i][1]];
            let dd = d[0] * d[0] + d[1] * d[1];
            if dd < 1e-12 {
                continue;
            }
            let dp = [p[0] - pos[i][0], p[1] - pos[i][1]];
            let h = 1.0 - (dp[0] * d[0] + dp[1] * d[1]) / dd;
            if h <= 0.0 {
                wi = 0.0;
                break;
            }
            wi = wi.min(h);
        }
        w[i] = wi;
    }
    bt_normalize(&mut w, || nearest_child(pos, p));
    w
}

/// 2D Freeform Directional: like Cartesian, but in polar terms (angle and
/// radius, the radius scaled by each pair's mean radius so they compare).
pub fn blend_weights_freeform_directional(pos: &[[f32; 2]], p: [f32; 2]) -> Vec<f32> {
    let n = pos.len();
    let mut w = vec![0.0; n];
    if n < 2 {
        if n == 1 {
            w[0] = 1.0;
        }
        return w;
    }
    let pr = bt_len(p);
    let rad: Vec<f32> = pos.iter().map(|q| bt_len(*q)).collect();
    let origin: Vec<bool> = rad.iter().map(|r| *r < BT_EPS).collect();
    // Offset of p from child i: (angle, radius).
    let alpha = |i: usize, via: Option<usize>| -> f32 {
        if pr < BT_EPS {
            return 0.0;
        }
        if !origin[i] {
            bt_angle(pos[i], p)
        } else if let Some(j) = via.filter(|&j| !origin[j]) {
            bt_angle(pos[j], p)
        } else {
            0.0
        }
    };
    for i in 0..n {
        if origin[i] && origin.iter().filter(|o| **o).count() > 1 {
            continue;
        }
        let cap_alpha = if origin[i] || pr < BT_EPS { 0.0 } else { bt_angle(pos[i], p) };
        let mut wi = 1.0 - cap_alpha.abs() / std::f32::consts::PI;
        let rho = pr - rad[i];
        for j in 0..n {
            if i == j || (origin[i] && origin[j]) {
                continue;
            }
            let s = 2.0 / (rad[i] + rad[j]).max(1e-9);
            let d_rho = (rad[j] - rad[i]) * s;
            let d_alpha = if origin[i] || origin[j] { 0.0 } else { bt_angle(pos[i], pos[j]) };
            let den = d_rho * d_rho + d_alpha * d_alpha;
            if den < 1e-12 {
                continue;
            }
            let h = 1.0 - (s * rho * d_rho + alpha(i, Some(j)) * d_alpha) / den;
            if h <= 0.0 {
                wi = 0.0;
                break;
            }
            wi = wi.min(h);
        }
        w[i] = wi.max(0.0);
    }
    bt_normalize(&mut w, || nearest_child(pos, p));
    w
}

/// Direct weights: each child's own (non-negative) parameter value, divided
/// by the sum of the raw values when `normalize` is on and the sum exceeds 1.
pub fn blend_weights_direct(values: &[f32], normalize: bool) -> Vec<f32> {
    let sum: f32 = values.iter().sum();
    let mut w: Vec<f32> = values.iter().map(|v| v.max(0.0)).collect();
    if normalize && sum > 1.0 {
        w.iter_mut().for_each(|v| *v /= sum);
    }
    w
}

impl BlendTree {
    /// The kind in effect: a 2D kind without a Y parameter is a 1D tree (that
    /// is how files from before `kind` existed read).
    pub fn effective_kind(&self) -> BlendTreeKind {
        if self.kind == BlendTreeKind::FreeformCartesian2D && self.parameter_y.is_empty() {
            BlendTreeKind::Simple1D
        } else {
            self.kind
        }
    }

    pub fn is_2d(&self) -> bool {
        self.effective_kind().is_2d()
    }

    /// Per-child weights for the parameter point `p`. `param` reads a named
    /// float (Direct trees); the others use `p`.
    pub fn weights_with(&self, p: [f32; 2], param: &dyn Fn(&str) -> f32) -> Vec<f32> {
        let n = self.motions.len();
        if n == 0 {
            return Vec::new();
        }
        let pos = || self.motions.iter().map(|m| m.position).collect::<Vec<_>>();
        match self.effective_kind() {
            BlendTreeKind::Simple1D => {
                let th: Vec<f32> = self.motions.iter().map(|m| m.position[0]).collect();
                blend_weights_1d(&th, p[0])
            }
            BlendTreeKind::SimpleDirectional2D => blend_weights_simple_directional(&pos(), p),
            BlendTreeKind::FreeformDirectional2D => blend_weights_freeform_directional(&pos(), p),
            BlendTreeKind::FreeformCartesian2D => blend_weights_freeform_cartesian(&pos(), p),
            BlendTreeKind::Direct => {
                let v: Vec<f32> = self.motions.iter().map(|m| param(&m.direct_parameter)).collect();
                blend_weights_direct(&v, self.normalize_direct)
            }
        }
    }

    /// Rename a parameter in this tree and its nested trees.
    pub fn rename_parameter(&mut self, old: &str, new: &str) {
        let fix = |s: &mut String| {
            if s == old {
                *s = new.to_owned();
            }
        };
        fix(&mut self.parameter);
        fix(&mut self.parameter_y);
        for m in &mut self.motions {
            fix(&mut m.direct_parameter);
            if let Some(t) = m.tree.as_deref_mut() {
                t.rename_parameter(old, new);
            }
        }
    }

    /// Per-child weights for `p` (Direct trees read zero parameters).
    pub fn weights(&self, p: [f32; 2]) -> Vec<f32> {
        self.weights_with(p, &|_| 0.0)
    }

    /// Per-child weights with the parameters read from `param`.
    pub fn weights_for(&self, param: &dyn Fn(&str) -> f32) -> Vec<f32> {
        self.weights_with([param(&self.parameter), param(&self.parameter_y)], param)
    }

    /// Every clip leaf of the tree (nested trees expanded) with its final
    /// weight and per-child playback settings, weight 0 included.
    pub fn leaves(&self, param: &dyn Fn(&str) -> f32) -> Vec<BlendLeaf<'_>> {
        let mut out = Vec::new();
        let mut index = 0;
        self.collect_leaves(param, 1.0, 1.0, false, 0.0, 0, &mut index, &mut out);
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn collect_leaves<'a>(
        &'a self,
        param: &dyn Fn(&str) -> f32,
        weight: f32,
        time_scale: f32,
        mirror: bool,
        cycle_offset: f32,
        depth: usize,
        index: &mut usize,
        out: &mut Vec<BlendLeaf<'a>>,
    ) {
        let w = self.weights_for(param);
        for (m, &mw) in self.motions.iter().zip(&w) {
            let (ts, mir, off) = (time_scale * m.time_scale, mirror ^ m.mirror, cycle_offset + m.cycle_offset);
            match &m.tree {
                Some(t) if depth < BLEND_TREE_MAX_DEPTH => t.collect_leaves(param, weight * mw, ts, mir, off, depth + 1, index, out),
                Some(_) => {}
                None => {
                    out.push(BlendLeaf { index: *index, clip: &m.clip, weight: weight * mw, time_scale: ts, mirror: mir, cycle_offset: off });
                    *index += 1;
                }
            }
        }
    }
}

/// A named marker at `time` seconds into `clip`.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[serde(default)]
pub struct AnimEvent {
    pub clip: String,
    pub time: f32,
    pub name: String,
}

/// The parameters a clip event carries (Unity `AnimationEvent`: one float, int,
/// string and object reference); unused ones stay at their defaults.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AnimEventArgs {
    pub float: f32,
    pub int: i32,
    pub string: String,
    /// An asset or object reference, by project path or name.
    pub object: String,
}

impl AnimEventArgs {
    /// Separates the name from the parameters in an Animator's `state.fired`
    /// entries (a name alone stays a plain string).
    const SEP: char = '\u{1}';

    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// `name` with these parameters, as one `state.fired` entry.
    pub fn encode(&self, name: &str) -> String {
        let clean = |s: &str| s.replace(Self::SEP, " ");
        let s = Self::SEP;
        format!("{}{s}{}{s}{}{s}{}{s}{}", clean(name), self.float, self.int, clean(&self.string), clean(&self.object))
    }

    /// Split a `state.fired` entry into the event name and its parameters
    /// (`None` for a plain name).
    pub fn decode(entry: &str) -> (&str, Option<AnimEventArgs>) {
        let mut parts = entry.splitn(5, Self::SEP);
        let name = parts.next().unwrap_or(entry);
        let (Some(f), Some(i), Some(s), Some(o)) = (parts.next(), parts.next(), parts.next(), parts.next()) else {
            return (name, None);
        };
        (name, Some(AnimEventArgs { float: f.parse().unwrap_or(0.0), int: i.parse().unwrap_or(0), string: s.to_owned(), object: o.to_owned() }))
    }
}

/// Broadcast when an animation event fires (readable via `ctx.events`).
#[derive(Clone, Debug)]
pub struct AnimationEvent {
    pub object: ObjectId,
    pub name: String,
    /// The parameters of a clip event that carries any.
    pub args: Option<AnimEventArgs>,
}

/// Runtime playback state of an [`Animator`] (not saved).
#[derive(Clone, Default, Debug)]
pub struct AnimatorState {
    pub current: Option<String>,
    /// Seconds into the current clip.
    pub time: f32,
    /// Clip being faded out and its time.
    pub previous: Option<(String, f32)>,
    pub fade_elapsed: f32,
    pub fade_duration: f32,
    pub playing: bool,
    pub started: bool,
    /// Length of the current clip (filled in by the engine).
    pub duration: f32,
    /// Clip names available on this object's model (filled in by the engine).
    pub clips: Vec<String>,
    /// Blend-tree parameters.
    pub params: HashMap<String, f32>,
    /// Events fired since the engine last delivered them.
    pub fired: Vec<String>,
    /// World-space root motion since it was last applied.
    pub root_delta: Vec3,
    /// Root motion turn (radians about the object's up axis) since it was last applied.
    pub root_yaw: f32,
    /// Root position (skeleton space) last frame, for root-motion deltas.
    pub last_root: Option<Vec3>,
    /// Controller playback per layer (filled in by the engine).
    pub layers: Vec<LayerPlayback>,
    /// Controller path `layers` was initialised for.
    pub controller_ready: Option<String>,
    /// `play_state` requests waiting for the next evaluation.
    pub requests: Vec<StateRequest>,
    /// Controller state changes `(layer, from, to)` since the engine last delivered them.
    pub state_changes: Vec<(usize, String, String)>,
    /// Blend weights of the playing motion's clips last frame (see
    /// [`LayerPlayback::leaf_weights`]), for the controller-less path.
    pub leaf_weights: Vec<f32>,
    /// The object is not visible (set by the renderer's visibility test);
    /// what [`AnimatorCullingMode`] reacts to.
    pub culled: bool,
    /// Time banked toward the next whole step in `Fixed` update mode.
    pub fixed_accum: f32,
    /// IK goals the IK pass solves (see [`IkGoal`]).
    pub ik_goals: Vec<IkGoal>,
    /// Callbacks raised for states that name behaviours, this frame.
    pub callbacks: Vec<StateCallback>,
    pub behaviours: BehaviourInstances,
    /// A state behaviour ran since the engine last cleared this: parameters
    /// may have changed under it (avatar parameters read them back).
    pub params_touched: bool,
    /// The last pose written, kept while culling holds it.
    pub last_output: Option<FrozenPose>,
}

/// Playback of one controller layer.
#[derive(Clone, Default, Debug)]
pub struct LayerPlayback {
    /// Current state index and name.
    pub state: usize,
    pub state_name: String,
    /// Seconds into the current state (speed applied).
    pub time: f32,
    /// `time` over the state's length (keeps growing past 1 while looping).
    pub normalized_time: f32,
    /// State being faded out and its time.
    pub previous: Option<(usize, f32)>,
    pub fade_elapsed: f32,
    pub fade_duration: f32,
    /// Transition currently blending (index into the layer's transitions).
    pub transition: Option<usize>,
    /// Pose a cross-fade fades out from when it interrupted another fade:
    /// the blend at the moment of the interruption, held still. `previous`
    /// then only names the state whose transition list owns the new fade.
    pub frozen: Option<FrozenPose>,
    /// The current state was entered this frame: its first event window
    /// reaches back a hair so events at the very start fire.
    pub just_entered: bool,
    /// Blend weights of the current state's clips as of the last frame (by
    /// leaf slot), so a clip that drops to weight 0 still gets one last
    /// event window. `src_leaf_weights` is the same for the fading-out state.
    pub leaf_weights: Vec<f32>,
    pub src_leaf_weights: Vec<f32>,
    /// The layer's pose from the last frame, kept while a state with Write
    /// Defaults off is involved.
    pub held: Option<FrozenPose>,
}

/// One animated object property (active, component enabled, blendshape,
/// material value) in a pose. `chan` indexes the rig's property table; `cover`
/// is how much of the final value this sample accounts for (1 = fully driven,
/// less while a fade or a partial layer leaves the rest to what the object
/// holds now).
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct PropSample {
    pub chan: u32,
    /// Rig node (the object the property lives on).
    pub node: u32,
    pub value: [f32; 4],
    pub cover: f32,
}

/// A pose snapshot (per-node TRS, scalar lane, driven-channel mask and
/// property samples).
#[derive(Clone, Default, Debug)]
pub struct FrozenPose {
    pub t: Vec<Vec3>,
    pub r: Vec<Quat>,
    pub s: Vec<Vec3>,
    pub scalar: Vec<f32>,
    pub touched: Vec<u8>,
    /// Sorted by `chan`.
    pub props: Vec<PropSample>,
}

impl LayerPlayback {
    pub fn in_transition(&self) -> bool {
        (self.previous.is_some() || self.frozen.is_some()) && self.fade_duration > 0.0
    }
    /// 0..1 progress of the active cross-fade (1 when none).
    pub fn transition_progress(&self) -> f32 {
        if self.in_transition() { (self.fade_elapsed / self.fade_duration).clamp(0.0, 1.0) } else { 1.0 }
    }
}

/// A forced state switch (`play_state`). `layer` None = the first layer with
/// a state of that name.
#[derive(Clone, Debug)]
pub struct StateRequest {
    pub layer: Option<usize>,
    pub state: String,
    pub fade: f32,
}

/// Broadcast when an Animator Controller layer enters a new state
/// (readable via `ctx.events::<AnimatorStateChanged>()`).
#[derive(Clone, Debug)]
pub struct AnimatorStateChanged {
    pub object: ObjectId,
    pub layer: usize,
    pub from: String,
    pub to: String,
}

impl Default for Animator {
    fn default() -> Self {
        Self {
            controller: String::new(),
            drivers: Vec::new(),
            update_mode: AnimatorUpdateMode::Normal,
            culling: AnimatorCullingMode::CullUpdateTransforms,
            clip: String::new(),
            speed: 1.0,
            looping: true,
            play_on_start: true,
            blend_trees: Vec::new(),
            events: Vec::new(),
            root_motion: false,
            gpu_skinning: true,
            state: AnimatorState::default(),
        }
    }
}

impl Animator {
    /// Switch to `clip`, cross-fading over `fade` seconds. A no-op when that
    /// clip is already playing. With a controller this forces the state of
    /// that name instead (see [`Animator::play_state`]).
    pub fn play(&mut self, clip: &str, fade: f32) {
        if !self.controller.is_empty() {
            self.play_state(clip, fade);
            return;
        }
        let s = &mut self.state;
        s.started = true;
        if s.playing && s.current.as_deref() == Some(clip) {
            return;
        }
        s.previous = if fade > 0.0 {
            s.current.take().map(|c| (c, s.time))
        } else {
            None
        };
        s.fade_elapsed = 0.0;
        s.fade_duration = fade.max(0.0);
        s.current = Some(clip.to_owned());
        s.time = 0.0;
        s.leaf_weights.clear();
        s.playing = true;
    }

    /// Set the IK goal for the limb ending at `tip` (replacing its previous one).
    pub fn set_ik_goal(&mut self, tip: &str, target: Vec3, pole: Option<Vec3>, weight: f32) {
        let goal = IkGoal { tip: tip.to_owned(), target, pole, weight };
        match self.state.ik_goals.iter_mut().find(|g| g.tip == tip) {
            Some(g) => *g = goal,
            None => self.state.ik_goals.push(goal),
        }
    }

    pub fn clear_ik_goals(&mut self) {
        self.state.ik_goals.clear();
    }

    /// Set a blend-tree parameter.
    pub fn set_float(&mut self, name: &str, value: f32) {
        self.state.params.insert(name.to_owned(), value);
    }

    pub fn get_float(&self, name: &str) -> f32 {
        self.state.params.get(name).copied().unwrap_or(0.0)
    }

    pub fn set_int(&mut self, name: &str, value: i32) {
        self.set_float(name, value as f32);
    }

    pub fn get_int(&self, name: &str) -> i32 {
        self.get_float(name).round() as i32
    }

    pub fn set_bool(&mut self, name: &str, value: bool) {
        self.set_float(name, if value { 1.0 } else { 0.0 });
    }

    pub fn get_bool(&self, name: &str) -> bool {
        self.get_float(name) > 0.5
    }

    /// Set a trigger: stays set until a transition consumes it.
    pub fn set_trigger(&mut self, name: &str) {
        self.set_float(name, 1.0);
    }

    pub fn reset_trigger(&mut self, name: &str) {
        self.set_float(name, 0.0);
    }

    /// Force a controller state by name (on the first layer that has it),
    /// cross-fading over `fade` seconds. States inside sub-state machines are
    /// named by path, `"Locomotion/Walk"` (and so reported by
    /// [`Self::current_state`] and state-change events).
    pub fn play_state(&mut self, state: &str, fade: f32) {
        self.state.requests.push(StateRequest { layer: None, state: state.to_owned(), fade });
    }

    pub fn play_state_on_layer(&mut self, layer: usize, state: &str, fade: f32) {
        self.state.requests.push(StateRequest { layer: Some(layer), state: state.to_owned(), fade });
    }

    /// Current controller state of a layer.
    pub fn current_state(&self, layer: usize) -> Option<&str> {
        self.state.layers.get(layer).map(|l| l.state_name.as_str())
    }

    pub fn is_in_state(&self, layer: usize, state: &str) -> bool {
        self.current_state(layer) == Some(state)
    }

    pub fn is_in_transition(&self, layer: usize) -> bool {
        self.state.layers.get(layer).is_some_and(|l| l.in_transition())
    }

    /// Normalized time of a layer's current state (1.0 = one full pass).
    pub fn state_normalized_time(&self, layer: usize) -> f32 {
        self.state.layers.get(layer).map(|l| l.normalized_time).unwrap_or(0.0)
    }

    /// Freeze on the current frame.
    pub fn stop(&mut self) {
        self.state.playing = false;
    }

    pub fn is_playing(&self, clip: &str) -> bool {
        self.state.playing && self.state.current.as_deref() == Some(clip)
    }

    /// A non-looping clip reached its end.
    pub fn finished(&self) -> bool {
        !self.looping && self.state.duration > 0.0 && self.state.time >= self.state.duration
    }
}

static ANIMATOR_ACTIONS: &[ActionDesc] = &[
    ActionDesc::new("SetFloat", "Set Float", &[ParamDesc::string("name"), ParamDesc::float("value", 0.0)], true),
    ActionDesc::new("SetInt", "Set Int", &[ParamDesc::string("name"), ParamDesc::int("value", 0)], true),
    ActionDesc::new("SetBool", "Set Bool", &[ParamDesc::string("name"), ParamDesc::bool("value", true)], true),
    ActionDesc::new("SetTrigger", "Set Trigger", &[ParamDesc::string("name")], true),
    ActionDesc::new("ResetTrigger", "Reset Trigger", &[ParamDesc::string("name")], true),
    ActionDesc::new("PlayState", "Play State", &[ParamDesc::string("state"), ParamDesc::float("fade", 0.1)], true),
];

impl TypedComponent for Animator {
    const NAME: &'static str = "Animator";

    fn actions(&self) -> &'static [ActionDesc] {
        ANIMATOR_ACTIONS
    }
    fn invoke_action(&mut self, _ctx: &mut ActionCtx, name: &str, args: &[ArgValue]) -> Result<(), String> {
        let n = args[0].as_str();
        match name {
            "SetFloat" => self.set_float(n, args[1].as_f32()),
            "SetInt" => self.set_int(n, args[1].as_i64() as i32),
            "SetBool" => self.set_bool(n, args[1].as_bool()),
            "SetTrigger" => self.set_trigger(n),
            "ResetTrigger" => self.reset_trigger(n),
            "PlayState" => self.play_state(n, args[1].as_f32()),
            _ => return Err(actions::unknown_action(Self::NAME, name)),
        }
        Ok(())
    }
}

// -------------------------------------------------- pawns & controllers (2A/2B)

/// How a [`Pawn`] reads input and places its camera.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum ControlMode {
    /// WASD + mouselook; camera at eye height, body yaws, camera pitches.
    FirstPerson,
    /// WASD relative to facing; orbit camera on a spring arm behind the body.
    ThirdPerson,
    /// WASD in world plane; body faces movement; fixed high camera angle.
    TopDown,
    /// Camera-only: WASD pans, wheel zooms; no body movement.
    Strategy,
}

impl ControlMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::FirstPerson => "First Person",
            Self::ThirdPerson => "Third Person",
            Self::TopDown => "Top Down",
            Self::Strategy => "Strategy",
        }
    }
    pub const ALL: [ControlMode; 4] = [
        Self::FirstPerson,
        Self::ThirdPerson,
        Self::TopDown,
        Self::Strategy,
    ];
}

/// A controllable entity (pawn + player-controller combined). When `possessed`,
/// it reads the action snapshot, moves itself, drives + activates its camera, and
/// (FP/TP) writes the child camera's pitch / spring-arm. With a
/// [`CharacterController`] on the same object it walks through the physics world
/// (walls, steps, slopes); without one it moves on a flat floor at its start height.
#[derive(Serialize, Deserialize)]
pub struct Pawn {
    pub mode: ControlMode,
    /// The local player controls this pawn (it receives input + owns the view).
    pub possessed: bool,
    /// Camera object this pawn drives/activates. For FP/TP make it a CHILD of the
    /// pawn so body yaw rotates it; the pawn writes its local pitch / arm offset.
    pub camera: ObjectRef,
    /// If set, the pawn teleports to a matching [`SpawnPoint`] on Play start.
    #[serde(default)]
    pub spawn_tag: String,
    pub move_speed: f32,
    pub accel: f32,
    pub decel: f32,
    pub jump_power: f32,
    pub gravity: f32,
    pub look_sensitivity: f32,
    pub eye_height: f32,
    /// Third-person / top-down camera distance.
    pub arm_length: f32,
    pub min_pitch: f32,
    pub max_pitch: f32,
    #[serde(skip)]
    vel: Vec3,
    #[serde(skip)]
    yaw: f32,
    #[serde(skip)]
    pitch: f32,
    #[serde(skip)]
    floor_y: f32,
    #[serde(skip)]
    grounded: bool,
    #[serde(skip)]
    started: bool,
}

impl Default for Pawn {
    fn default() -> Self {
        Self {
            mode: ControlMode::FirstPerson,
            possessed: true,
            camera: ObjectRef::NONE,
            spawn_tag: String::new(),
            move_speed: 6.0,
            accel: 40.0,
            decel: 30.0,
            jump_power: 6.0,
            gravity: 18.0,
            look_sensitivity: 0.15,
            eye_height: 1.6,
            arm_length: 5.0,
            min_pitch: -85.0,
            max_pitch: 85.0,
            vel: Vec3::ZERO,
            yaw: 0.0,
            pitch: -15.0,
            floor_y: 0.0,
            grounded: true,
            started: false,
        }
    }
}

impl Pawn {
    /// The body rotation this pawn writes (None in Strategy mode, which leaves it alone).
    pub fn body_rotation(&self) -> Option<Quat> {
        (!matches!(self.mode, ControlMode::Strategy)).then(|| Quat::from_rotation_y(self.yaw.to_radians()))
    }

    /// One movement step from an input frame (`dt` is the frame's scaled
    /// delta). `view` replaces the look input with a recorded (yaw, pitch).
    /// `physical` is `Some(grounded)` when a Character Controller moves the
    /// pawn: the returned world delta goes to `move_character`. Otherwise the
    /// pawn walks a flat floor and `pos` (local translation) moves in place.
    pub fn step(&mut self, input: &InputState, dt: f32, pos: &mut Vec3, physical: Option<bool>, view: Option<(f32, f32)>) -> Option<Vec3> {
        let dt = dt.min(0.1);
        let mv = input.axis2("Move");

        match view {
            Some((yaw, pitch)) => {
                self.yaw = yaw;
                // A networked view is the owner's claim: held to the pawn's pitch limits.
                self.pitch = pitch.clamp(self.min_pitch, self.max_pitch);
            }
            None => {
                // Look: mouse delta is per-frame (no dt); analog sticks pre-scaled.
                let look = input.axis2("Look");
                self.yaw -= look.x * self.look_sensitivity;
                if matches!(self.mode, ControlMode::FirstPerson | ControlMode::ThirdPerson) {
                    self.pitch = (self.pitch - look.y * self.look_sensitivity)
                        .clamp(self.min_pitch, self.max_pitch);
                }
            }
        }

        // Movement direction per mode.
        let yaw_rot = Quat::from_rotation_y(self.yaw.to_radians());
        let (dir, face_move) = match self.mode {
            ControlMode::FirstPerson | ControlMode::ThirdPerson => {
                let fwd = yaw_rot * Vec3::NEG_Z;
                let right = yaw_rot * Vec3::X;
                (right * mv.x + fwd * mv.y, false)
            }
            // World-plane movement; body faces the move direction.
            ControlMode::TopDown | ControlMode::Strategy => {
                (Vec3::new(mv.x, 0.0, -mv.y), true)
            }
        };
        let dir = dir.normalize_or_zero();

        let horizontal = if dir.length_squared() > 1e-4 {
            let target = dir * self.move_speed;
            let cur = Vec3::new(self.vel.x, 0.0, self.vel.z);
            let next = cur.lerp(target, (self.accel * dt).min(1.0));
            if face_move && self.mode == ControlMode::TopDown {
                self.yaw = (-dir.x).atan2(-dir.z).to_degrees();
            }
            next
        } else {
            let cur = Vec3::new(self.vel.x, 0.0, self.vel.z);
            cur.lerp(Vec3::ZERO, (self.decel * dt).min(1.0))
        };
        self.vel.x = horizontal.x;
        self.vel.z = horizontal.z;

        let walking = matches!(self.mode, ControlMode::FirstPerson | ControlMode::ThirdPerson);
        if let Some(grounded) = physical {
            self.grounded = grounded;
            if self.grounded && self.vel.y < 0.0 {
                self.vel.y = 0.0;
            }
        }
        if walking {
            if self.grounded && input.pressed("Jump") {
                self.vel.y = self.jump_power;
                self.grounded = false;
            }
            self.vel.y -= self.gravity * dt;
        } else {
            self.vel.y = 0.0;
        }

        if physical.is_some() {
            return Some(self.vel * dt);
        }
        *pos += self.vel * dt;
        if walking && pos.y <= self.floor_y {
            pos.y = self.floor_y;
            self.vel.y = 0.0;
            self.grounded = true;
        }
        None
    }

    /// One frame of movement through `host`: `update` runs exactly this, and
    /// so do networked prediction's replays.
    fn advance(&mut self, host: &mut dyn MoveHost, input: &InputState, dt: f32, view: Option<(f32, f32)>) {
        let physical = host.has_character().then(|| host.grounded());
        let mut pos = host.translation();
        match self.step(input, dt, &mut pos, physical, view) {
            Some(delta) => host.move_character(delta),
            None => host.set_translation(pos),
        }
        if let Some(r) = self.body_rotation() {
            host.set_rotation(r);
        }
    }

    fn place_camera(&self, ctx: &mut ComponentCtx) {
        let (trans, rot) = match self.mode {
            ControlMode::FirstPerson => (
                Vec3::new(0.0, self.eye_height, 0.0),
                Quat::from_rotation_x(self.pitch.to_radians()),
            ),
            ControlMode::ThirdPerson => (
                Vec3::new(0.0, self.eye_height, self.arm_length),
                Quat::from_rotation_x(self.pitch.to_radians()),
            ),
            ControlMode::TopDown | ControlMode::Strategy => (
                Vec3::new(0.0, self.arm_length, self.arm_length * 0.6),
                Quat::from_rotation_x((-60.0_f32).to_radians()),
            ),
        };
        ctx.set_local_transform(self.camera, Some(trans), Some(rot), None);
    }
}

impl TypedComponent for Pawn {
    const NAME: &'static str = "Pawn";

    fn predictable(&self) -> Option<&dyn Predictable> {
        Some(self)
    }
    fn predictable_mut(&mut self) -> Option<&mut dyn Predictable> {
        Some(self)
    }

    fn start(&mut self, ctx: &mut ComponentCtx) {
        self.vel = Vec3::ZERO;
        self.floor_y = ctx.self_transform().translation.y;
        // Spawn-point placement (2 — spawn points).
        if !self.spawn_tag.is_empty()
            && let Some(t) = ctx.spawn_point(&self.spawn_tag)
        {
            ctx.set_world_position(t.translation);
            self.floor_y = t.translation.y;
            self.yaw = t.rotation.to_euler(glam::EulerRot::YXZ).0.to_degrees();
        }
        if self.possessed {
            ctx.set_active_camera(self.camera);
            if matches!(self.mode, ControlMode::FirstPerson | ControlMode::ThirdPerson) {
                ctx.set_cursor_locked(true);
            }
        }
        *ctx.rotation = Quat::from_rotation_y(self.yaw.to_radians());
        self.place_camera(ctx);
        self.started = true;
    }

    fn update(&mut self, ctx: &mut ComponentCtx) {
        if !self.possessed {
            return;
        }
        if !self.started {
            TypedComponent::start(self, ctx);
        }
        let (input, dt) = (ctx.input, ctx.dt);
        self.advance(ctx, input, dt, None);
        // Strategy zoom on the wheel via the Look.y / wheel channel is omitted;
        // arm_length stays as authored.
        self.place_camera(ctx);
    }
}

impl Predictable for Pawn {
    fn motion(&self) -> MotionState {
        MotionState { vel: self.vel, yaw: self.yaw, pitch: self.pitch, grounded: self.grounded, floor_y: self.floor_y, ..Default::default() }
    }

    fn set_motion(&mut self, m: &MotionState, _host: &mut dyn MoveHost) {
        self.vel = m.vel;
        self.yaw = m.yaw;
        self.pitch = m.pitch;
        self.grounded = m.grounded;
        self.floor_y = m.floor_y;
    }

    fn frame_extra(&self) -> Vec<f32> {
        vec![self.yaw, self.pitch]
    }

    fn replay(&mut self, host: &mut dyn MoveHost, input: &InputState, extra: &[f32], dt: f32) {
        self.advance(host, input, dt, motion::view_of(extra));
    }

    fn locally_driven(&self) -> bool {
        self.possessed
    }
}

/// Marks an object as a spawn location. Pawns with a matching `spawn_tag` teleport
/// here on Play start; game code finds them via `ctx.spawn_point(tag)`.
#[derive(Serialize, Deserialize)]
pub struct SpawnPoint {
    /// Group this point belongs to: "player", "npc", a team name, etc.
    pub tag: String,
    /// Ordering hint within the tag (round-robin / slot index).
    #[serde(default)]
    pub index: u32,
}

impl Default for SpawnPoint {
    fn default() -> Self {
        Self {
            tag: "player".to_string(),
            index: 0,
        }
    }
}

impl TypedComponent for SpawnPoint {
    const NAME: &'static str = "Spawn Point";
}

// ------------------------------------------------------------- networking (2G)

/// Replicates the owning object's transform over the network. Whoever has
/// authority (see [`NetView`]) broadcasts its transform; everyone else receives
/// and applies it (optionally smoothed). `grabbable` lets the local player take
/// ownership by pressing `grab_action`, move it, and release it for others.
#[derive(Serialize, Deserialize)]
pub struct Sync {
    /// Any peer can grab authority by pressing `grab_action`.
    pub grabbable: bool,
    /// Action that toggles local ownership when `grabbable`.
    pub grab_action: String,
    /// Remote-update smoothing rate (0 = snap; higher = snappier lerp).
    pub smoothing: f32,
    #[serde(skip)]
    started: bool,
}

impl Default for Sync {
    fn default() -> Self {
        Self {
            grabbable: true,
            grab_action: "Grab".to_string(),
            smoothing: 12.0,
            started: false,
        }
    }
}

impl TypedComponent for Sync {
    const NAME: &'static str = "Sync";

    fn update(&mut self, ctx: &mut ComponentCtx) {
        self.started = true;
        if !ctx.net.connected || !self.grabbable || self.grab_action.is_empty() {
            return;
        }
        // Toggle authority on grab: take it if we don't own it, else release.
        if ctx.input.pressed(&self.grab_action) {
            if ctx.owns_self() {
                ctx.release_ownership();
            } else {
                ctx.request_ownership();
            }
        }
    }
}

#[cfg(test)]
mod component_toggle_tests {
    use super::*;

    #[test]
    fn a_component_switched_off_keeps_its_place_and_settings() {
        let mut slot: Box<dyn Component> = Box::new(CharacterController { radius: 0.5, ..Default::default() });
        assert!(component_enabled(slot.as_ref()));
        assert!(set_component_enabled(&mut slot, false));
        assert!(!component_enabled(slot.as_ref()) && slot.type_name() == "Character");
        assert!(slot.as_any().downcast_ref::<CharacterController>().is_none(), "systems looking it up by type don't see it");
        assert!(component_inner(slot.as_ref()).as_any().downcast_ref::<CharacterController>().is_some_and(|c| c.radius == 0.5));
        assert!(!set_component_enabled(&mut slot, false), "already off");
        assert!(set_component_enabled(&mut slot, true));
        assert!(slot.as_any().downcast_ref::<CharacterController>().is_some_and(|c| c.radius == 0.5), "back as it was");
    }
}

#[cfg(test)]
mod layer_tests {
    use super::*;

    #[test]
    fn default_layers_all_collide_and_named() {
        let s = LayerSettings::default();
        assert_eq!(s.layer_name(0), "Default");
        assert_eq!(s.layer_name(5), "Layer 5"); // unnamed fallback
        assert!(s.collide(0, 0));
        assert!(s.collide(3, 7));
        assert_eq!(s.collision_mask(0), u32::MAX);
    }

    #[test]
    fn set_collide_is_symmetric() {
        let mut s = LayerSettings::default();
        s.set_collide(1, 2, false);
        assert!(!s.collide(1, 2));
        assert!(!s.collide(2, 1)); // symmetric
        assert!(s.collide(1, 3)); // others unaffected
        s.set_collide(1, 2, true);
        assert!(s.collide(1, 2));
        assert!(s.collide(2, 1));
    }

    #[test]
    fn orthographic_camera_maps_the_view_box_to_clip_space() {
        let cam = CameraComponent { projection: CameraProjection::Orthographic, ortho_size: 4.0, near: 0.5, far: 20.0, ..Default::default() };
        let proj = cam.projection_matrix(2.0);
        assert!(is_orthographic(&proj));
        assert!(!is_orthographic(&CameraComponent::default().projection_matrix(1.5)));
        // Half-height 4, aspect 2: x in [-8, 8], y in [-4, 4], depth 0.5..20 in front.
        let p = |v: Vec3| proj.project_point3(v);
        assert!((p(Vec3::new(8.0, 4.0, -0.5)) - Vec3::new(1.0, 1.0, 0.0)).length() < 1e-5);
        assert!((p(Vec3::new(-8.0, -4.0, -20.0)) - Vec3::new(-1.0, -1.0, 1.0)).length() < 1e-5);
        // No perspective foreshortening: the same x at two depths lands on the same NDC x.
        assert!((p(Vec3::new(2.0, 0.0, -1.0)).x - p(Vec3::new(2.0, 0.0, -15.0)).x).abs() < 1e-6);
        assert_eq!(cam.half_extents_at(2.0, 1.0), (8.0, 4.0));
        assert_eq!(cam.half_extents_at(2.0, 10.0), (8.0, 4.0));
        let persp = CameraComponent { fov_deg: 90.0, ..Default::default() };
        let (w, h) = persp.half_extents_at(2.0, 3.0);
        assert!((h - 3.0).abs() < 1e-5 && (w - 6.0).abs() < 1e-5);
    }

    #[test]
    fn orthographic_rays_are_parallel_and_start_at_the_pointer() {
        let view = Mat4::look_to_rh(Vec3::new(0.0, 0.0, 10.0), Vec3::NEG_Z, Vec3::Y);
        let proj = orthographic_projection(5.0, 1.0, 0.1, 100.0);
        let eye = Vec3::new(0.0, 0.0, 10.0);
        let (o, d) = ndc_ray(view, proj, glam::Vec2::new(0.5, -0.5), eye).unwrap();
        assert!((d - Vec3::NEG_Z).length() < 1e-5);
        assert!((o.x - 2.5).abs() < 1e-4 && (o.y + 2.5).abs() < 1e-4, "origin offset by the pointer: {o}");
        // Perspective rays keep the eye as origin and fan out.
        let persp = Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0);
        let (o, d) = ndc_ray(view, persp, glam::Vec2::new(0.5, 0.0), eye).unwrap();
        assert_eq!(o, eye);
        assert!(d.x > 0.05);
    }

    #[test]
    fn camera_files_without_projection_load_as_perspective() {
        let cam: CameraComponent = ron::from_str("(id: 1, fov_deg: 50.0, near: 0.1, far: 10.0)").unwrap();
        assert_eq!(cam.projection, CameraProjection::Perspective);
        assert_eq!(cam.ortho_size, 5.0);
    }

    #[test]
    fn box_projection_at_center_is_identity() {
        // A fragment AT the capture centre: the box-projected vector points the
        // same way as the reflection ray (just scaled to the box face).
        let c = Vec3::ZERO;
        let h = Vec3::splat(5.0);
        for r in [Vec3::X, Vec3::Y, Vec3::Z, Vec3::new(1.0, 2.0, -3.0).normalize()] {
            let out = box_project_reflection(Vec3::ZERO, r, c, h).normalize();
            assert!(
                out.dot(r) > 0.999,
                "center reflection should be unchanged: {r:?} -> {out:?}"
            );
        }
    }

    #[test]
    fn box_projection_corrects_off_center_parallax() {
        // A fragment near the +X wall reflecting straight UP should hit the box
        // ceiling AHEAD of it (still +X-leaning), not straight up — that offset is
        // exactly the parallax correction a plain infinite cube would miss.
        let c = Vec3::ZERO;
        let h = Vec3::splat(5.0);
        let p = Vec3::new(4.0, 0.0, 0.0);
        let out = box_project_reflection(p, Vec3::Y, c, h);
        // Ray exits the top at (4, 5, 0); re-vectored from centre that's (4, 5, 0).
        assert!((out - Vec3::new(4.0, 5.0, 0.0)).length() < 1e-3, "got {out:?}");
        assert!(out.x > 0.0, "parallax must keep the +X lean: {out:?}");
    }

    #[test]
    fn box_projection_hits_facing_wall() {
        // Reflecting toward +X from the centre lands on the +X wall.
        let out = box_project_reflection(Vec3::ZERO, Vec3::X, Vec3::ZERO, Vec3::splat(5.0));
        assert!((out - Vec3::new(5.0, 0.0, 0.0)).length() < 1e-3, "got {out:?}");
    }

    #[test]
    fn normalized_repairs_short_vecs() {
        let s = LayerSettings { names: vec![], collision: vec![], sorting: SortingLayers::default() }.normalized();
        assert_eq!(s.names.len(), NUM_LAYERS);
        assert_eq!(s.collision.len(), NUM_LAYERS);
        assert_eq!(s.names[0], "Default");
        assert!(s.collide(10, 20)); // filled with full-collision default
    }
}

#[cfg(test)]
mod blend_tree_tests {
    use super::*;

    fn tree(kind: BlendTreeKind, ys: &str, motions: &[(&str, [f32; 2])]) -> BlendTree {
        BlendTree {
            name: "t".into(),
            kind,
            parameter: "x".into(),
            parameter_y: ys.into(),
            motions: motions.iter().map(|(c, p)| BlendMotion { clip: (*c).into(), position: *p, ..Default::default() }).collect(),
            normalize_direct: false,
        }
    }

    fn close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-4, "{got:?} vs {want:?}");
        }
    }

    const CROSS: [[f32; 2]; 5] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]];

    #[test]
    fn one_d_blends_neighbours_and_clamps() {
        let t = tree(BlendTreeKind::Simple1D, "", &[("Run", [6.0, 0.0]), ("Idle", [0.0, 0.0]), ("Walk", [2.0, 0.0])]);
        assert_eq!(t.weights([1.0, 0.0]), vec![0.0, 0.5, 0.5]);
        assert_eq!(t.weights([4.0, 0.0]), vec![0.5, 0.0, 0.5]);
        assert_eq!(t.weights([-3.0, 0.0]), vec![0.0, 1.0, 0.0]);
        assert_eq!(t.weights([9.0, 0.0]), vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn one_d_equal_thresholds_do_not_divide_by_zero() {
        close(&blend_weights_1d(&[0.0, 1.0, 1.0], 1.0), &[0.0, 0.0, 1.0]);
        close(&blend_weights_1d(&[0.0, 1.0, 1.0], 0.5), &[0.5, 0.5, 0.0]);
        close(&blend_weights_1d(&[2.0], 9.0), &[1.0]);
    }

    #[test]
    fn simple_directional_golden_table() {
        let w = |p: [f32; 2]| blend_weights_simple_directional(&CROSS, p);
        // [centre, up, right, down, left]
        close(&w([0.0, 0.0]), &[1.0, 0.0, 0.0, 0.0, 0.0]);
        close(&w([0.25, 0.25]), &[0.5, 0.25, 0.25, 0.0, 0.0]);
        close(&w([0.5, 0.5]), &[0.0, 0.5, 0.5, 0.0, 0.0]);
        close(&w([3.0, 3.0]), &[0.0, 0.5, 0.5, 0.0, 0.0]);
        close(&w([2.0, 0.0]), &[0.0, 0.0, 1.0, 0.0, 0.0]);
        close(&w([0.0, 0.5]), &[0.5, 0.5, 0.0, 0.0, 0.0]);
        close(&w([-0.3, -0.3]), &[0.4, 0.0, 0.0, 0.3, 0.3]);
    }

    #[test]
    fn simple_directional_without_a_centre_spreads_the_remainder() {
        let w = blend_weights_simple_directional(&CROSS[1..], [0.25, 0.25]);
        // [up, right, down, left]: a = b = 0.25, c = 0.5 spread 0.125 each.
        close(&w, &[0.375, 0.375, 0.125, 0.125]);
        close(&blend_weights_simple_directional(&CROSS[1..], [0.0, 0.0]), &[0.25; 4]);
        // Two children at the origin: an error, all zero.
        close(&blend_weights_simple_directional(&[[0.0; 2], [0.0; 2], [1.0, 0.0]], [0.5, 0.0]), &[0.0; 3]);
        close(&blend_weights_simple_directional(&[[1.0, 0.0]], [0.5, 0.0]), &[1.0]);
    }

    #[test]
    fn freeform_cartesian_golden_table() {
        let tri = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let w = |p: [f32; 2]| blend_weights_freeform_cartesian(&tri, p);
        close(&w([0.25, 0.25]), &[0.6, 0.2, 0.2]);
        close(&w([1.0, 0.0]), &[0.0, 1.0, 0.0]);
        close(&w([0.0, 0.0]), &[1.0, 0.0, 0.0]);
        close(&w([0.5, 0.0]), &[0.5, 0.5, 0.0]);
        // Far outside it still resolves to a sensible mix.
        let far = w([5.0, 5.0]);
        assert!((far.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn freeform_directional_golden_table() {
        let w = |p: [f32; 2]| blend_weights_freeform_directional(&CROSS, p);
        close(&w([0.0, 1.0]), &[0.0, 1.0, 0.0, 0.0, 0.0]);
        close(&w([0.0, 0.5]), &[0.5, 0.5, 0.0, 0.0, 0.0]);
        close(&w([0.0, 0.0]), &[1.0, 0.0, 0.0, 0.0, 0.0]);
        close(&w([1.0, 0.0]), &[0.0, 0.0, 1.0, 0.0, 0.0]);
        // Halfway round between up and right, on the circle: those two split it.
        let h = std::f32::consts::FRAC_1_SQRT_2;
        close(&w([h, h]), &[0.0, 0.5, 0.5, 0.0, 0.0]);
    }

    #[test]
    fn direct_weights_normalise_only_above_one() {
        close(&blend_weights_direct(&[0.25, 0.5], true), &[0.25, 0.5]);
        close(&blend_weights_direct(&[0.5, 0.8], true), &[0.5 / 1.3, 0.8 / 1.3]);
        close(&blend_weights_direct(&[0.5, 0.8], false), &[0.5, 0.8]);
        close(&blend_weights_direct(&[-1.0, 0.4], false), &[0.0, 0.4]);
    }

    #[test]
    fn direct_tree_reads_each_childs_parameter() {
        let mut t = tree(BlendTreeKind::Direct, "", &[("A", [0.0; 2]), ("B", [0.0; 2])]);
        t.motions[0].direct_parameter = "wa".into();
        t.motions[1].direct_parameter = "wb".into();
        let p = |n: &str| match n {
            "wa" => 0.25,
            "wb" => 0.75,
            _ => 0.0,
        };
        close(&t.weights_for(&p), &[0.25, 0.75]);
    }

    #[test]
    fn two_d_weights_normalize_and_snap() {
        let t = tree(
            BlendTreeKind::FreeformCartesian2D,
            "y",
            &[("F", [0.0, 1.0]), ("B", [0.0, -1.0]), ("L", [-1.0, 0.0]), ("R", [1.0, 0.0])],
        );
        assert_eq!(t.weights([0.0, 1.0]), vec![1.0, 0.0, 0.0, 0.0]);
        let w = t.weights([0.5, 0.5]);
        assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(w[0] > w[1] && w[3] > w[2]);
    }

    #[test]
    fn nested_trees_multiply_weights_and_inherit_settings() {
        let mut outer = tree(BlendTreeKind::Simple1D, "", &[("", [0.0, 0.0]), ("Run", [1.0, 0.0])]);
        outer.motions[0].tree = Some(Box::new(BlendTree {
            kind: BlendTreeKind::Simple1D,
            parameter: "y".into(),
            motions: vec![
                BlendMotion { clip: "Idle".into(), position: [0.0, 0.0], ..Default::default() },
                BlendMotion { clip: "Walk".into(), position: [1.0, 0.0], time_scale: 2.0, cycle_offset: 0.25, ..Default::default() },
            ],
            ..Default::default()
        }));
        outer.motions[0].time_scale = 0.5;
        outer.motions[0].mirror = true;
        let p = |n: &str| match n {
            "x" => 0.25,
            "y" => 0.5,
            _ => 0.0,
        };
        let leaves = outer.leaves(&p);
        assert_eq!(leaves.len(), 3);
        assert_eq!((leaves[0].clip, leaves[1].clip, leaves[2].clip), ("Idle", "Walk", "Run"));
        assert_eq!(leaves.iter().map(|l| l.index).collect::<Vec<_>>(), vec![0, 1, 2]);
        close(&leaves.iter().map(|l| l.weight).collect::<Vec<_>>(), &[0.375, 0.375, 0.25]);
        assert_eq!(leaves[1].time_scale, 1.0);
        assert!(leaves[1].mirror && !leaves[2].mirror);
        assert_eq!(leaves[1].cycle_offset, 0.25);
    }

    #[test]
    fn old_files_load_as_freeform_cartesian() {
        let old = r#"(name: "T", parameter: "x", parameter_y: "y", motions: [(clip: "A", position: (0.0, 0.0))])"#;
        let t: BlendTree = ron::from_str(old).expect("old tree");
        assert_eq!(t.kind, BlendTreeKind::FreeformCartesian2D);
        assert_eq!(t.motions[0].time_scale, 1.0);
        assert!(t.motions[0].tree.is_none());
        assert!(t.is_2d());
        // Without a Y parameter an old tree was 1D, and still is.
        let one = r#"(name: "T", parameter: "x", motions: [(clip: "A", position: (0.0, 0.0)), (clip: "B", position: (2.0, 0.0))])"#;
        let t: BlendTree = ron::from_str(one).expect("old 1D tree");
        assert_eq!(t.effective_kind(), BlendTreeKind::Simple1D);
        close(&t.weights([1.0, 0.0]), &[0.5, 0.5]);
        // And a saved tree round-trips.
        let mut n = t.clone();
        n.kind = BlendTreeKind::Direct;
        let back: BlendTree = ron::from_str(&ron::to_string(&n).expect("ser")).expect("de");
        assert_eq!(back, n);
    }
}

#[cfg(test)]
mod terrain_tests {
    use super::*;

    #[test]
    fn default_terrain_is_a_flat_32_quad_grid() {
        let t = Terrain::default();
        assert_eq!(t.size, 100.0);
        assert_eq!(t.resolution, 32);
        assert_eq!(t.heights.len(), 33 * 33);
        assert!(t.heights.iter().all(|h| *h == 0.0));
        assert!(t.texture_weights.is_empty());
    }

    #[test]
    fn terrain_round_trips_through_ron() {
        let t = Terrain {
            size: 64.0,
            resolution: 4,
            heights: (0..25).map(|i| i as f32).collect(),
            material: "ground.mat".into(),
            heightmap: "heightmap.png".into(),
            heightmap_scale: 25.0,
            heightmap_offset: -2.0,
            heightmap_imported_path: "heightmap.png".into(),
            heightmap_imported_scale: 25.0,
            heightmap_imported_offset: -2.0,
            texture_weights: (0..25)
                .map(|i| {
                    let t = i as f32 / 24.0;
                    [1.0 - t, t, 0.0, 0.0]
                })
                .collect(),
        };
        let s = ron::to_string(&t).expect("serialize");
        let back: Terrain = ron::from_str(&s).expect("deserialize");
        assert_eq!(back.size, t.size);
        assert_eq!(back.resolution, t.resolution);
        assert_eq!(back.heights, t.heights);
        assert_eq!(back.material, "ground.mat");
        assert_eq!(back.heightmap, "heightmap.png");
        assert_eq!(back.heightmap_scale, 25.0);
        assert_eq!(back.heightmap_offset, -2.0);
        assert_eq!(back.heightmap_imported_path, "heightmap.png");
        assert_eq!(back.heightmap_imported_scale, 25.0);
        assert_eq!(back.heightmap_imported_offset, -2.0);
        assert_eq!(back.texture_weights, t.texture_weights);
    }

    #[test]
    fn scatter_is_deterministic_and_counts_density() {
        let t = Terrain { size: 10.0, resolution: 4, heights: vec![0.0; 25], ..Default::default() };
        let layer = FoliageLayer { density: 1.0, ..Default::default() };
        let a = scatter_foliage(&t, &layer, 42);
        let b = scatter_foliage(&t, &layer, 42);
        assert_eq!(a.len(), 100, "1/m² over 100 m² = 100 instances");
        assert_eq!(a, b, "the same seed must scatter identically");
        assert!(a.iter().all(|i| i.position[1].abs() < 1e-5), "flat terrain -> on the surface");
    }

    #[test]
    fn scatter_skips_steep_slopes() {
        // A step: x < 4 flat, x >= 4 raised, so the cliff is steeper than 0.5 rad.
        let res = 8usize;
        let n = res + 1;
        let mut heights = vec![0.0f32; n * n];
        for z in 0..n {
            for x in 0..n {
                if x >= 4 {
                    heights[z * n + x] = 3.0;
                }
            }
        }
        let t = Terrain { size: 8.0, resolution: res as u32, heights, ..Default::default() };
        let layer = FoliageLayer { density: 2.0, max_slope: 0.5, ..Default::default() };
        let out = scatter_foliage(&t, &layer, 7);
        assert!(!out.is_empty());
        let step = t.size / t.resolution as f32;
        let half = t.size * 0.5;
        for i in &out {
            let gx = (i.position[0] + half) / step;
            let gz = (i.position[2] + half) / step;
            let normal = foliage_normal(&t.heights, n, res, gx.round() as i64, gz.round() as i64, step);
            assert!(normal.y.acos() <= layer.max_slope + 1e-4, "instance on a slope {}", normal.y.acos());
        }
    }

    #[test]
    fn scatter_aligns_instances_to_the_surface() {
        // A uniform +X slope so every surface normal tilts toward +X.
        let res = 4usize;
        let n = res + 1;
        let mut heights = vec![0.0f32; n * n];
        for z in 0..n {
            for x in 0..n {
                heights[z * n + x] = x as f32 * 0.5;
            }
        }
        let t = Terrain { size: 4.0, resolution: res as u32, heights, ..Default::default() };
        let layer = FoliageLayer { density: 4.0, align_to_normal: true, ..Default::default() };
        let out = scatter_foliage(&t, &layer, 7);
        assert!(!out.is_empty());
        for i in &out {
            let up = Quat::from_array(i.rotation) * Vec3::Y;
            assert!(up.y > 0.89, "aligned up should stay mostly vertical: {up}");
            assert!(up.x < -0.1, "up tilts toward -X for an uphill +X slope: {up}");
            assert!(up.z.abs() < 1e-3, "no Z tilt on a pure +X slope: {up}");
        }
    }
}

#[cfg(test)]
mod decal_tests {
    use super::*;

    #[test]
    fn decal_uv_centers_points_in_front() {
        let id = Mat4::IDENTITY;
        let size = [2.0, 2.0];
        let c = decal_uv(Vec3::new(0.0, 0.0, -0.5), id, size, 1.0).unwrap();
        assert!((c[0] - 0.5).abs() < 1e-5 && (c[1] - 0.5).abs() < 1e-5);
        let corner = decal_uv(Vec3::new(-1.0, 1.0, 0.0), id, size, 1.0).unwrap();
        assert!(corner[0].abs() < 1e-5 && (corner[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn decal_uv_rejects_outside_the_box() {
        let id = Mat4::IDENTITY;
        let size = [2.0, 2.0];
        assert!(decal_uv(Vec3::new(2.0, 0.0, -0.5), id, size, 1.0).is_none(), "outside X");
        assert!(decal_uv(Vec3::new(0.0, 0.0, 1.0), id, size, 1.0).is_none(), "behind the projector");
        assert!(decal_uv(Vec3::new(0.0, 0.0, -2.0), id, size, 1.0).is_none(), "past the depth");
    }

    #[test]
    fn decal_uv_follows_the_projector_transform() {
        let projector = Mat4::from_translation(Vec3::new(2.0, 0.0, 3.0));
        let uv = decal_uv(Vec3::new(2.0, 0.0, 2.5), projector, [1.0, 1.0], 1.0).unwrap();
        assert!((uv[0] - 0.5).abs() < 1e-5 && (uv[1] - 0.5).abs() < 1e-5);
        assert!(decal_uv(Vec3::new(0.0, 0.0, 2.5), projector, [1.0, 1.0], 1.0).is_none());
    }
}

#[cfg(test)]
mod area_light_tests {
    use super::*;

    /// The Area Light is a registered component; scene files written with only some of its
    /// fields fill the rest with defaults, and a Light Probe Volume saved before Light Volume
    /// mode loads as a classic cascade volume.
    #[test]
    fn area_light_is_registered_with_serde_defaults_and_old_probe_volumes_load_as_cascades() {
        let registry = ComponentRegistry::with_builtins();
        assert!(registry.create("Area Light").is_some(), "registered");
        let l: AreaLight = ron::from_str("(size: (2.0, 0.5), two_sided: true)").unwrap();
        assert_eq!((l.size, l.two_sided, l.shape, l.intensity, l.range), ([2.0, 0.5], true, AreaLightShape::Rectangle, 4.0, 15.0));
        assert!(l.texture.is_empty());
        let v: LightProbeVolume = ron::from_str("(size: (4.0, 3.0, 4.0), density: 1.0)").unwrap();
        assert_eq!((v.blend_distance, v.weight), (0.0, 1.0));
    }
}

#[cfg(test)]
mod cloud_tests {
    use super::*;

    #[test]
    fn cloud_density_is_zero_outside_the_layer() {
        let c = VolumetricClouds::default();
        assert_eq!(cloud_density(Vec3::new(0.0, c.base - 1.0, 0.0), &c, 0.0), 0.0);
        assert_eq!(cloud_density(Vec3::new(0.0, c.base + c.thickness + 1.0, 0.0), &c, 0.0), 0.0);
    }

    #[test]
    fn cloud_density_vanishes_at_zero_coverage() {
        let c = VolumetricClouds { coverage: 0.0, ..Default::default() };
        for y in [c.base, c.base + c.thickness * 0.25, c.base + c.thickness * 0.5] {
            for x in [0.0, 100.0, -250.0] {
                assert_eq!(cloud_density(Vec3::new(x, y, 0.0), &c, 0.0), 0.0);
            }
        }
    }

    #[test]
    fn cloud_density_advects_with_the_wind() {
        let c = VolumetricClouds::default();
        let p = Vec3::new(120.0, c.base + c.thickness * 0.5, -40.0);
        let t = 7.5;
        let a = cloud_density(p, &c, t);
        let b = cloud_density(p - Vec3::from(c.wind) * t, &c, 0.0);
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }

    #[test]
    fn cloud_density_is_deterministic_and_bounded() {
        let c = VolumetricClouds::default();
        let p = Vec3::new(33.0, c.base + c.thickness * 0.5, 17.0);
        assert_eq!(cloud_density(p, &c, 1.0), cloud_density(p, &c, 1.0));
        for i in 0..200 {
            let q = Vec3::new(i as f32 * 7.0, c.base + (i % 9) as f32 * c.thickness / 9.0, i as f32 * 3.0);
            let d = cloud_density(q, &c, 0.5);
            assert!((0.0..=1.0).contains(&d), "{d}");
        }
    }
}

mod avatar_hands;
pub use avatar_hands::{AvatarHands, FingerPose, HandPose};
