//! Top-down click-to-move and interactables.

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::{
    ActionCtx, ActionDesc, ActionType, ArgValue, ComponentCtx, EventActions, FireSource, NavAgent, NetValue, NetworkVariables,
    ObjectId, ObjectRef, PropertyDesc, RpcTarget, TypedComponent, fire_all,
};

/// An [`Interactable`] was used.
#[derive(Clone, Copy, Debug)]
pub struct InteractEvent {
    /// The interactable that was used.
    pub object: ObjectId,
    /// The player who used it.
    pub by: ObjectId,
    /// Total uses so far.
    pub uses: u32,
}

/// How an [`Interactable`] reacts to being used.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum InteractMode {
    /// Every use runs `on_interact` (a button, a pickup, a door that only opens).
    #[default]
    Trigger,
    /// Every use flips `on`: `on_on` runs when it turns on, `on_off` when it
    /// turns off (a lamp, a lever, a gate).
    Toggle,
    /// Use = press (`on_press`), `request_release` = release (`on_release`):
    /// a held lever, a pressure plate someone stands on.
    Hold,
}

impl InteractMode {
    /// Every mode, in declaration order.
    pub const ALL: [InteractMode; 3] = [Self::Trigger, Self::Toggle, Self::Hold];

    /// Inspector label of this mode.
    pub fn label(self) -> &'static str {
        match self {
            Self::Trigger => "Trigger",
            Self::Toggle => "Toggle",
            Self::Hold => "Hold",
        }
    }
}

/// Something the player can use: a switch, a door, a chest. Used by
/// [`ClickToMove`] (walk there and use it) or from code with
/// `ctx.modify::<Interactable>(id, |i| i.request(user))`.
///
/// What a use does is data: the `on_*` lists call exposed functions on any
/// component of any object (see [`crate::actions`]), with the user as the
/// instigator. The older `toggle` reference still works next to them.
///
/// Networking: with a `Network Variables` component on the same object the
/// toggle state `on` is replicated (key `"on"`), and the actions run only on
/// the peer with authority over the object; other peers mirror `on`, and a use
/// or release requested on another peer is sent to the authority as an RPC
/// (the user's object id travels with it, so the actions' instigator is right).
/// Without one, every peer runs its own uses locally as before.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct Interactable {
    /// The prompt shown while in range.
    pub prompt: String,
    /// How close the user must be.
    pub range: f32,
    /// How the interactable reacts to being used.
    pub mode: InteractMode,
    /// Toggle this object's active state on use (open a gate, switch a lamp).
    pub toggle: ObjectRef,
    /// Re-bake navmeshes after toggling (gates that open new paths).
    pub rebake_navmesh: bool,
    /// Sound played on use (empty = silent).
    pub sound: String,
    /// Seconds before it can be used again.
    pub cooldown: f32,
    /// Only usable once.
    pub single_use: bool,
    /// Toggle mode: the current state (saved, so a lamp stays lit).
    pub on: bool,
    /// Runs on every use (the press, in Hold mode).
    pub on_interact: EventActions,
    /// Toggle mode: runs when a use turns it on.
    pub on_on: EventActions,
    /// Toggle mode: runs when a use turns it off.
    pub on_off: EventActions,
    /// Hold mode: runs on the press.
    pub on_press: EventActions,
    /// Hold mode: runs on the release.
    pub on_release: EventActions,
    /// Total uses so far (runtime).
    #[serde(skip)]
    pub uses: u32,
    /// A use was requested by this player (runtime).
    #[serde(skip)]
    pub requested: Option<ObjectId>,
    /// A release was requested by this player (runtime).
    #[serde(skip)]
    pub release_requested: Option<ObjectId>,
    /// Hold mode: pressed and not yet released.
    #[serde(skip)]
    pub held: bool,
    /// Cooldown remaining (runtime).
    #[serde(skip)]
    pub cooldown_left: f32,
}

impl Default for Interactable {
    fn default() -> Self {
        Self {
            prompt: "Use".into(),
            range: 1.8,
            mode: InteractMode::Trigger,
            toggle: ObjectRef::NONE,
            rebake_navmesh: false,
            sound: String::new(),
            cooldown: 0.5,
            single_use: false,
            on: false,
            on_interact: Vec::new(),
            on_on: Vec::new(),
            on_off: Vec::new(),
            on_press: Vec::new(),
            on_release: Vec::new(),
            uses: 0,
            requested: None,
            release_requested: None,
            held: false,
            cooldown_left: 0.0,
        }
    }
}

impl Interactable {
    /// Ask to be used by `by` (applied on this component's next update).
    pub fn request(&mut self, by: ObjectId) {
        self.requested = Some(by);
    }

    /// Hold mode: `by` let go (applied on the next update).
    pub fn request_release(&mut self, by: ObjectId) {
        self.release_requested = Some(by);
    }

    /// Whether the interactable can be used right now (cooldown over, not
    /// spent if single-use).
    pub fn available(&self) -> bool {
        self.cooldown_left <= 0.0 && !(self.single_use && self.uses > 0)
    }

    /// Whether `on` is replicated and actions are authority-only here.
    fn networked(ctx: &ComponentCtx) -> bool {
        ctx.net.connected && ctx.get_self::<NetworkVariables>().is_some()
    }

    /// Hold mode: `by` let go. Runs on the peer that has authority.
    fn release(&mut self, ctx: &mut ComponentCtx, by: ObjectId) {
        if !self.held {
            return;
        }
        let me = ctx.self_id();
        self.held = false;
        let payload = [("instigator", ArgValue::Object(by.into())), ("object", ArgValue::Object(me.into()))];
        fire_all(&self.on_release, ctx, &FireSource { source: me, instigator: Some(by), payload: &payload });
    }

    /// `by` uses this. Runs on the peer that has authority.
    fn use_by(&mut self, ctx: &mut ComponentCtx, by: ObjectId, networked: bool) {
        let me = ctx.self_id();
        if !self.available() {
            return;
        }
        self.uses += 1;
        self.cooldown_left = self.cooldown;
        if self.toggle.is_set() {
            let on = ctx.is_active(self.toggle);
            ctx.set_active(self.toggle, !on);
            if self.rebake_navmesh {
                ctx.rebake_navmesh();
            }
        }
        if !self.sound.is_empty() {
            let at = ctx.self_transform().translation;
            ctx.play_sound_at(self.sound.clone(), at, 1.0);
        }
        if self.mode == InteractMode::Toggle {
            self.on = !self.on;
            if networked {
                ctx.set_net_var_self("on", self.on);
            }
        }
        if self.mode == InteractMode::Hold {
            self.held = true;
        }
        let payload = [
            ("instigator", ArgValue::Object(by.into())),
            ("object", ArgValue::Object(me.into())),
            ("uses", ArgValue::Int(self.uses as i64)),
            ("on", ArgValue::Bool(self.on)),
        ];
        let src = FireSource { source: me, instigator: Some(by), payload: &payload };
        fire_all(&self.on_interact, ctx, &src);
        match self.mode {
            InteractMode::Trigger => {}
            InteractMode::Toggle => fire_all(if self.on { &self.on_on } else { &self.on_off }, ctx, &src),
            InteractMode::Hold => fire_all(&self.on_press, ctx, &src),
        }
        ctx.emit(InteractEvent { object: me, by, uses: self.uses });
    }
}

/// RPC names a non-authority peer's use / release travels under (the argument
/// is the user's object id as 16 big-endian bytes), sent to the authority.
const RPC_USE: &str = "zestengine.interact";
const RPC_RELEASE: &str = "zestengine.interact_release";

static INTERACTABLE_ACTIONS: &[ActionDesc] = &[
    ActionDesc::new("Interact", "Interact", &[], true),
    ActionDesc::new("Release", "Release", &[], true),
];
static INTERACTABLE_PROPERTIES: &[PropertyDesc] = &[
    PropertyDesc::new("on", ActionType::Bool),
    PropertyDesc::new("held", ActionType::Bool),
    PropertyDesc::new("uses", ActionType::Int),
];

impl TypedComponent for Interactable {
    const NAME: &'static str = "Interactable";

    fn update(&mut self, ctx: &mut ComponentCtx) {
        self.cooldown_left = (self.cooldown_left - ctx.dt).max(0.0);
        let me = ctx.self_id();
        let networked = Self::networked(ctx);
        let authority = !networked || ctx.net.has_authority(me);
        if networked
            && !authority
            && let Some(NetValue::Bool(on)) = ctx.net_var_self("on")
        {
            self.on = *on;
        }
        if networked && !authority {
            // A use by this peer goes to the authority, which runs the actions
            // and replicates `on`; it is not answered locally.
            if let Some(by) = self.release_requested.take() {
                ctx.rpc_self(RPC_RELEASE, vec![NetValue::Bytes(by.raw().to_be_bytes().to_vec())], RpcTarget::Owner);
            }
            if let Some(by) = self.requested.take() {
                ctx.rpc_self(RPC_USE, vec![NetValue::Bytes(by.raw().to_be_bytes().to_vec())], RpcTarget::Owner);
            }
            return;
        }
        if let Some(by) = self.release_requested.take() {
            self.release(ctx, by);
        }
        if let Some(by) = self.requested.take() {
            self.use_by(ctx, by, networked);
        }
        if networked {
            // Uses other peers asked for, in arrival order.
            let asks: Vec<(bool, ObjectId)> = ctx
                .rpcs_for_self()
                .filter(|r| r.from != ctx.net.local_peer && (r.name == RPC_USE || r.name == RPC_RELEASE))
                .filter_map(|r| match r.args.first() {
                    Some(NetValue::Bytes(b)) => Some((r.name == RPC_RELEASE, ObjectId::from_raw(u128::from_be_bytes(b.as_slice().try_into().ok()?)))),
                    _ => None,
                })
                .collect();
            for (release, by) in asks {
                if release {
                    self.release(ctx, by);
                } else {
                    self.use_by(ctx, by, networked);
                }
            }
        }
    }

    fn actions(&self) -> &'static [ActionDesc] {
        INTERACTABLE_ACTIONS
    }
    fn invoke_action(&mut self, ctx: &mut ActionCtx, name: &str, _args: &[ArgValue]) -> Result<(), String> {
        let by = ctx.instigator.unwrap_or(ctx.self_id);
        match name {
            "Interact" => self.request(by),
            "Release" => self.request_release(by),
            _ => return Err(crate::actions::unknown_action(Self::NAME, name)),
        }
        Ok(())
    }
    fn properties(&self) -> &'static [PropertyDesc] {
        INTERACTABLE_PROPERTIES
    }
    fn read_property(&self, name: &str) -> Option<ArgValue> {
        match name {
            "on" => Some(ArgValue::Bool(self.on)),
            "held" => Some(ArgValue::Bool(self.held)),
            "uses" => Some(ArgValue::Int(self.uses as i64)),
            _ => None,
        }
    }
}

/// Click the ground to walk there over the navmesh (needs a [`NavAgent`] on
/// the same object and a baked `NavMesh Surface` in the scene); click an
/// [`Interactable`] to walk up to it and use it. Shows `marker` at the target.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct ClickToMove {
    /// Input action for clicking.
    pub click_action: String,
    /// Keep re-targeting while the button is held.
    pub hold_to_move: bool,
    /// Click-target marker (a root object; shown at the destination).
    pub marker: ObjectRef,
    /// How far a click ray may travel (m).
    pub max_ray_distance: f32,
    /// The clicked destination (runtime).
    #[serde(skip)]
    pub destination: Option<Vec3>,
    /// The interactable being walked to (runtime).
    #[serde(skip)]
    pub pending_interact: Option<ObjectId>,
    /// A Hold-mode Interactable this component pressed; released when the
    /// click button is up (the frame after the press at the earliest).
    #[serde(skip)]
    pub held: Option<ObjectId>,
    /// Seconds between destination updates while the button is held (runtime).
    #[serde(skip)]
    pub retarget: f32,
}

impl Default for ClickToMove {
    fn default() -> Self {
        Self {
            click_action: "Click".into(),
            hold_to_move: true,
            marker: ObjectRef::NONE,
            max_ray_distance: 500.0,
            destination: None,
            pending_interact: None,
            held: None,
            retarget: 0.0,
        }
    }
}

impl ClickToMove {
    /// The held Interactable to let go of now: the click button is up.
    pub(crate) fn take_release(&mut self, button_down: bool) -> Option<ObjectId> {
        if button_down { None } else { self.held.take() }
    }

    fn click(&mut self, ctx: &mut ComponentCtx) {
        let Some((origin, dir)) = ctx.pointer_ray() else {
            return;
        };
        let Some(hit) = ctx.raycast(origin, dir, self.max_ray_distance) else {
            return;
        };
        let interactable = ctx.get::<Interactable>(hit.object).map(|i| ctx.position_of(hit.object.into()).map(|p| (p, i.range)));
        let (dest, marker_at) = match interactable {
            Some(Some((pos, range))) => {
                self.pending_interact = Some(hit.object);
                let me = ctx.self_transform().translation;
                let back = (me - pos).with_y(0.0).normalize_or(Vec3::Z) * (range * 0.6);
                let stand = ctx.nearest_walkable(pos + back).unwrap_or(pos + back);
                (stand, pos)
            }
            _ => {
                self.pending_interact = None;
                let p = ctx.nearest_walkable(hit.point).unwrap_or(hit.point);
                (p, hit.point)
            }
        };
        self.destination = Some(dest);
        ctx.set_destination(ctx.self_id(), dest);
        if self.marker.is_set() {
            ctx.set_local_transform(self.marker, Some(marker_at + Vec3::Y * 0.03), None, None);
            ctx.set_active(self.marker, true);
        }
    }
}

impl TypedComponent for ClickToMove {
    const NAME: &'static str = "Click To Move";

    fn start(&mut self, ctx: &mut ComponentCtx) {
        if self.marker.is_set() {
            ctx.set_active(self.marker, false);
        }
    }

    fn update(&mut self, ctx: &mut ComponentCtx) {
        if ctx.dt <= 0.0 {
            return;
        }
        self.retarget -= ctx.dt;
        if let Some(target) = self.take_release(ctx.input.down(&self.click_action)) {
            let by = ctx.self_id();
            ctx.modify::<Interactable>(target, move |i| i.request_release(by));
        }
        let clicked = !ctx.pointer_over_ui()
            && (ctx.input.pressed(&self.click_action)
                || (self.hold_to_move && ctx.input.down(&self.click_action) && self.retarget <= 0.0));
        if clicked {
            self.retarget = 0.15;
            self.click(ctx);
            // The agent sees its new destination after this pass.
            return;
        }
        let me = ctx.self_transform().translation;
        if let Some(target) = self.pending_interact {
            let range = ctx.get::<Interactable>(target).map(|i| i.range);
            match (range, ctx.position_of(target.into())) {
                (Some(range), Some(pos)) if me.with_y(0.0).distance(pos.with_y(0.0)) <= range => {
                    let by = ctx.self_id();
                    ctx.modify::<Interactable>(target, move |i| i.request(by));
                    if ctx.get::<Interactable>(target).is_some_and(|i| i.mode == InteractMode::Hold) {
                        self.held = Some(target);
                    }
                    ctx.stop_agent(ctx.self_id());
                    self.pending_interact = None;
                    self.destination = None;
                }
                (None, _) | (_, None) => self.pending_interact = None,
                _ => {}
            }
        }
        let arrived = ctx.get_self::<NavAgent>().is_none_or(|a| a.arrived());
        if let Some(d) = self.destination
            && (arrived || me.with_y(0.0).distance(d.with_y(0.0)) < 0.3)
        {
            self.destination = None;
        }
        if self.destination.is_none() && self.pending_interact.is_none() && self.marker.is_set() && ctx.is_active(self.marker) {
            ctx.set_active(self.marker, false);
        }
    }
}
