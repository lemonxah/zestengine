//! Health, damage and hitscan weapons.

use glam::{Quat, Vec3};
use crate::ecs::{ObjectContext, Owner, Stage, SystemContext, SystemWorld};
use crate::MotionContext;
use serde::{Deserialize, Serialize};

use crate::{
    ActionCtx, ActionDesc, ActionType, ArgValue, ObjectId, ObjectRef, ParamDesc, PropertyDesc, RayFilter,
    TypedComponent,
};

/// Something took damage (broadcast by weapons; read with `ctx.events::<DamageEvent>()`).
#[derive(Clone, Copy, Debug)]
pub struct DamageEvent {
    /// The object that took damage.
    pub target: ObjectId,
    /// The object that dealt it.
    pub source: ObjectId,
    /// Damage dealt.
    pub amount: f32,
    /// World position of the hit.
    pub point: Vec3,
}

/// A [`Health`] reached zero.
#[derive(Clone, Copy, Debug)]
pub struct DeathEvent {
    /// The object whose `Health` reached zero.
    pub object: ObjectId,
}

/// Points scored (target dummies knocked down, objectives); the HUD's Score
/// binding sums these.
#[derive(Clone, Copy, Debug)]
pub struct ScoreEvent {
    /// Points scored.
    pub points: u32,
}

/// What happens to the object when its [`Health`] reaches zero.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum DeathAction {
    /// Stay in the scene (another component reacts, e.g. a target dummy).
    Nothing,
    /// Deactivate the object.
    Disable,
    /// Remove the object.
    Destroy,
}

impl DeathAction {
    /// Every action, in declaration order.
    pub const ALL: [DeathAction; 3] = [Self::Nothing, Self::Disable, Self::Destroy];
    /// Inspector label of this action.
    pub fn label(self) -> &'static str {
        match self {
            Self::Nothing => "Nothing",
            Self::Disable => "Disable object",
            Self::Destroy => "Destroy object",
        }
    }
}

/// Hit points. Native systems damage it through a checked mutable ECS borrow.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct Health {
    /// Maximum hit points.
    pub max: f32,
    /// Current hit points.
    pub current: f32,
    /// Hit points regained per second after `regen_delay` seconds without damage.
    pub regen_per_sec: f32,
    /// Seconds without damage before regeneration starts.
    pub regen_delay: f32,
    /// Take no damage.
    pub invulnerable: bool,
    /// What happens when `current` reaches zero.
    pub on_death: DeathAction,
    /// Seconds since the last hit (runtime).
    #[serde(skip)]
    pub since_damage: f32,
    /// The death event has fired (runtime).
    #[serde(skip)]
    pub death_handled: bool,
    /// Initialization completed for this component instance (runtime).
    #[serde(skip)]
    pub started: bool,
}

impl Default for Health {
    fn default() -> Self {
        Self {
            max: 100.0,
            current: 100.0,
            regen_per_sec: 0.0,
            regen_delay: 3.0,
            invulnerable: false,
            on_death: DeathAction::Nothing,
            since_damage: 0.0,
            death_handled: false,
            started: false,
        }
    }
}

impl Health {
    /// Subtract hit points. Returns true when this hit killed it.
    pub fn apply_damage(&mut self, amount: f32) -> bool {
        if self.invulnerable || self.is_dead() || amount <= 0.0 {
            return false;
        }
        self.current = (self.current - amount).max(0.0);
        self.since_damage = 0.0;
        self.is_dead()
    }

    /// Add hit points (capped at `max`).
    pub fn heal(&mut self, amount: f32) {
        if !self.is_dead() {
            self.current = (self.current + amount.max(0.0)).min(self.max);
        }
    }

    /// True when `current` is zero.
    pub fn is_dead(&self) -> bool {
        self.current <= 0.0
    }

    /// Back to full health (respawns, dummies standing back up).
    pub fn revive(&mut self) {
        self.current = self.max;
        self.death_handled = false;
        self.since_damage = 0.0;
    }

    /// Current / max, clamped to 0..1.
    pub fn fraction(&self) -> f32 {
        if self.max <= 0.0 { 0.0 } else { (self.current / self.max).clamp(0.0, 1.0) }
    }
}

static HEALTH_ACTIONS: &[ActionDesc] = &[
    ActionDesc::new("Damage", "Damage", &[ParamDesc::float("amount", 10.0)], false),
    ActionDesc::new("Heal", "Heal", &[ParamDesc::float("amount", 10.0)], true),
    ActionDesc::new("Revive", "Revive", &[], true),
    ActionDesc::new("SetInvulnerable", "Set Invulnerable", &[ParamDesc::bool("on", true)], true),
];
static HEALTH_PROPERTIES: &[PropertyDesc] = &[
    PropertyDesc::new("current", ActionType::Float),
    PropertyDesc::new("max", ActionType::Float),
    PropertyDesc::new("fraction", ActionType::Float),
    PropertyDesc::new("dead", ActionType::Bool),
];

impl TypedComponent for Health {
    const NAME: &'static str = "Health";

    fn actions(&self) -> &'static [ActionDesc] {
        HEALTH_ACTIONS
    }
    fn invoke_action(&mut self, _ctx: &mut ActionCtx, name: &str, args: &[ArgValue]) -> Result<(), String> {
        match name {
            "Damage" => {
                self.apply_damage(args[0].as_f32());
            }
            "Heal" => self.heal(args[0].as_f32()),
            "Revive" => self.revive(),
            "SetInvulnerable" => self.invulnerable = args[0].as_bool(),
            _ => return Err(crate::actions::unknown_action(Self::NAME, name)),
        }
        Ok(())
    }
    fn properties(&self) -> &'static [PropertyDesc] {
        HEALTH_PROPERTIES
    }
    fn read_property(&self, name: &str) -> Option<ArgValue> {
        match name {
            "current" => Some(ArgValue::Float(self.current)),
            "max" => Some(ArgValue::Float(self.max)),
            "fraction" => Some(ArgValue::Float(self.fraction())),
            "dead" => Some(ArgValue::Bool(self.is_dead())),
            _ => None,
        }
    }
}

impl Health {
    fn start_native(&mut self, _ctx: &mut ObjectContext<'_, '_>) {
        if self.current <= 0.0 || self.current > self.max {
            self.current = self.max;
        }
        self.started = true;
    }

    fn update_native(&mut self, ctx: &mut ObjectContext<'_, '_>) {
        if !self.started { self.start_native(ctx); }
        if self.is_dead() {
            if !self.death_handled {
                self.death_handled = true;
                let object = ctx.self_id();
                ctx.emit(DeathEvent { object });
                match self.on_death {
                    DeathAction::Nothing => {}
                    DeathAction::Disable => ctx.set_active(ctx.self_id(), false),
                    DeathAction::Destroy => ctx.destroy_self(),
                }
            }
            return;
        }
        self.since_damage += ctx.dt;
        if self.regen_per_sec > 0.0 && self.since_damage >= self.regen_delay {
            self.current = (self.current + self.regen_per_sec * ctx.dt).min(self.max);
        }
    }
}

/// What fires a [`HitscanWeapon`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum WeaponTrigger {
    /// Fires on an input action (desktop weapons).
    Action,
    /// Fires while held through a [`crate::Grabbable`] and its holder pulls the trigger (VR guns).
    Grabbed,
}

impl WeaponTrigger {
    /// Every trigger, in declaration order.
    pub const ALL: [WeaponTrigger; 2] = [Self::Action, Self::Grabbed];
    /// Inspector label of this trigger.
    pub fn label(self) -> &'static str {
        match self {
            Self::Action => "Input action",
            Self::Grabbed => "Held trigger (VR)",
        }
    }
}

/// Instant-hit gun: raycasts from `aim_from` (or the muzzle / this object),
/// damages any [`Health`] it hits, pushes rigid bodies and plays muzzle-flash
/// and impact particles. Magazine + reserve ammo with timed reloads.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct HitscanWeapon {
    /// What fires the weapon.
    pub trigger: WeaponTrigger,
    /// Input action that fires (Action trigger).
    pub fire_action: String,
    /// Input action that reloads.
    pub reload_action: String,
    /// Keep firing while the trigger is held.
    pub automatic: bool,
    /// Shots per second.
    pub fire_rate: f32,
    /// Damage per hit.
    pub damage: f32,
    /// Ray length (m).
    pub range: f32,
    /// Impulse applied to dynamic rigid bodies that are hit.
    pub impact_force: f32,
    /// Random cone half-angle in degrees.
    pub spread_deg: f32,
    /// Rounds per magazine.
    pub magazine_size: u32,
    /// Rounds in the magazine.
    pub ammo: u32,
    /// Rounds in reserve.
    pub reserve: u32,
    /// Never runs out of reserve ammo.
    pub infinite_reserve: bool,
    /// Seconds a reload takes.
    pub reload_time: f32,
    /// Ray origin + direction (forward, -Z). Usually the player camera.
    pub aim_from: ObjectRef,
    /// Muzzle-flash particle emitter.
    pub muzzle: ObjectRef,
    /// Muzzle particles per shot.
    pub muzzle_burst: u32,
    /// Impact particle emitter, moved to each world-space hit point.
    pub hit_effect: ObjectRef,
    /// Impact particles per hit.
    pub hit_burst: u32,
    /// Sound played on each shot (empty = silent).
    pub fire_sound: String,
    /// Sound played on each hit (empty = silent).
    pub hit_sound: String,
    /// Ray-cast layers (bit per layer).
    pub layer_mask: u32,
    /// Time until the next shot can fire (runtime).
    #[serde(skip)]
    pub cooldown: f32,
    /// Time left in the reload (runtime).
    #[serde(skip)]
    pub reload_timer: f32,
    /// Lifetime shots (runtime).
    #[serde(skip)]
    pub shots_fired: u32,
    /// Last hit position (runtime).
    #[serde(skip)]
    pub last_hit: Option<Vec3>,
    /// Trigger state last frame (runtime).
    #[serde(skip)]
    pub trigger_prev: bool,
    /// Frames since the last shot; > 0 keeps the weapon aimed at the crosshair.
    #[serde(skip)]
    pub fire_flash: f32,
    /// Spread RNG seed (runtime).
    #[serde(skip)]
    pub rng: u32,
    /// Initialization completed for this component instance (runtime).
    #[serde(skip)]
    pub started: bool,
}

impl Default for HitscanWeapon {
    fn default() -> Self {
        Self {
            trigger: WeaponTrigger::Action,
            fire_action: "Fire".into(),
            reload_action: "Reload".into(),
            automatic: true,
            fire_rate: 8.0,
            damage: 25.0,
            range: 200.0,
            impact_force: 4.0,
            spread_deg: 0.6,
            magazine_size: 30,
            ammo: 30,
            reserve: 120,
            infinite_reserve: false,
            reload_time: 1.4,
            aim_from: ObjectRef::NONE,
            muzzle: ObjectRef::NONE,
            muzzle_burst: 12,
            hit_effect: ObjectRef::NONE,
            hit_burst: 16,
            fire_sound: String::new(),
            hit_sound: String::new(),
            layer_mask: u32::MAX,
            cooldown: 0.0,
            reload_timer: 0.0,
            shots_fired: 0,
            last_hit: None,
            trigger_prev: false,
            fire_flash: 0.0,
            rng: 0x9E37_79B9,
            started: false,
        }
    }
}

/// `forward` rotated by a pseudo-random angle inside a cone of `spread_deg`.
/// `seed` advances on every call so successive shots differ.
pub fn spread_direction(forward: Vec3, spread_deg: f32, seed: &mut u32) -> Vec3 {
    let forward = forward.normalize_or(Vec3::NEG_Z);
    if spread_deg <= 0.0 {
        return forward;
    }
    let mut next = || {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 17;
        *seed ^= *seed << 5;
        (*seed as f32) / (u32::MAX as f32)
    };
    let angle = spread_deg.to_radians() * next().sqrt();
    let around = next() * std::f32::consts::TAU;
    let side = forward.any_orthonormal_vector();
    let axis = Quat::from_axis_angle(forward, around) * side;
    (Quat::from_axis_angle(axis, angle) * forward).normalize()
}

impl HitscanWeapon {
    /// True while a reload is running.
    pub fn reloading(&self) -> bool {
        self.reload_timer > 0.0
    }

    /// True briefly after a shot, so the held weapon keeps pointing at the
    /// crosshair while firing from the hip.
    pub fn firing(&self) -> bool {
        self.fire_flash > 0.0
    }

    fn start_reload(&mut self) {
        if self.reloading() || self.ammo >= self.magazine_size || (!self.infinite_reserve && self.reserve == 0) {
            return;
        }
        self.reload_timer = self.reload_time.max(0.01);
    }

    fn finish_reload(&mut self) {
        let need = self.magazine_size.saturating_sub(self.ammo);
        let take = if self.infinite_reserve { need } else { need.min(self.reserve) };
        self.ammo += take;
        if !self.infinite_reserve {
            self.reserve -= take;
        }
    }

    fn trigger_state(&self, ctx: &ObjectContext<'_, '_>) -> (bool, bool) {
        match self.trigger {
            WeaponTrigger::Action => (ctx.input.down(&self.fire_action), ctx.input.pressed(&self.fire_action)),
            WeaponTrigger::Grabbed => {
                let down = ctx.world.get::<crate::Grabbable>(ctx.owner).is_some_and(|g| g.held && g.trigger_down);
                (down, down && !self.trigger_prev)
            }
        }
    }

    fn fire(&mut self, ctx: &mut ObjectContext<'_, '_>) {
        let aim = ctx
            .transform_of(self.aim_from)
            .or_else(|| ctx.transform_of(self.muzzle))
            .unwrap_or_else(|| ctx.self_transform());
        let dir = spread_direction(aim.forward(), self.spread_deg, &mut self.rng);
        self.ammo = self.ammo.saturating_sub(1);
        self.shots_fired += 1;
        self.cooldown = 1.0 / self.fire_rate.max(0.01);
        self.fire_flash = 0.12;
        if self.muzzle.is_set() {
            ctx.emit_particles(self.muzzle, self.muzzle_burst);
        }
        if !self.fire_sound.is_empty() {
            ctx.play_sound_at(self.fire_sound.clone(), aim.translation, 1.0);
        }
        let filter = RayFilter { layer_mask: self.layer_mask, include_triggers: false, exclude: Some(ctx.self_id()) };
        let Some(hit) = ctx.raycast_filtered(aim.translation, dir, self.range, &filter) else {
            self.last_hit = None;
            return;
        };
        self.last_hit = Some(hit.point);
        let damage = self.damage;
        ctx.edit::<Health>(hit.object, move |h| {
            h.apply_damage(damage);
        });
        let source = ctx.self_id();
        ctx.emit(DamageEvent { target: hit.object, source, amount: damage, point: hit.point });
        if self.impact_force > 0.0 {
            ctx.add_impulse(hit.object, dir * self.impact_force);
        }
        if self.hit_effect.is_set() {
            let rot = crate::kit::look_rotation(hit.normal);
            ctx.set_world_transform_of(self.hit_effect, Some(hit.point + hit.normal * 0.02), Some(rot));
            ctx.emit_particles(self.hit_effect, self.hit_burst);
        }
        if !self.hit_sound.is_empty() {
            ctx.play_sound_at(self.hit_sound.clone(), hit.point, 0.8);
        }
    }
}

impl TypedComponent for HitscanWeapon {
    const NAME: &'static str = "Hitscan Weapon";
}

impl HitscanWeapon {
    fn start_native(&mut self, ctx: &mut ObjectContext<'_, '_>) {
        self.ammo = self.ammo.min(self.magazine_size);
        self.rng ^= (ctx.self_id().raw() as u32) | 1;
        self.started = true;
    }

    fn update_native(&mut self, ctx: &mut ObjectContext<'_, '_>) {
        if !self.started { self.start_native(ctx); }
        let (down, pressed) = self.trigger_state(ctx);
        self.trigger_prev = down;
        if ctx.dt <= 0.0 || ctx.ui_text_editing() {
            return;
        }
        self.cooldown = (self.cooldown - ctx.dt).max(0.0);
        self.fire_flash = (self.fire_flash - ctx.dt).max(0.0);
        if self.reloading() {
            self.reload_timer -= ctx.dt;
            if self.reload_timer <= 0.0 {
                self.reload_timer = 0.0;
                self.finish_reload();
            }
            return;
        }
        if !self.reload_action.is_empty() && ctx.input.pressed(&self.reload_action) {
            self.start_reload();
            return;
        }
        let wants = if self.automatic { down } else { pressed };
        if !wants || self.cooldown > 0.0 {
            return;
        }
        if self.ammo == 0 {
            self.start_reload();
            return;
        }
        self.fire(ctx);
    }
}

/// Shooting-range target: when its [`Health`] hits zero it tips over, scores,
/// and stands back up with full health after `respawn_delay` seconds.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct TargetDummy {
    /// Seconds knocked down before standing up.
    pub respawn_delay: f32,
    /// Tip-over speed in degrees per second.
    pub fall_speed: f32,
    /// Points scored when knocked down.
    pub points: u32,
    /// Knocked down (runtime).
    #[serde(skip)]
    pub knocked: bool,
    /// Time until standing up (runtime).
    #[serde(skip)]
    pub timer: f32,
    /// Tip angle (runtime).
    #[serde(skip)]
    pub angle: f32,
    /// Upright rotation (runtime).
    #[serde(skip)]
    pub base: Option<Quat>,
}

impl Default for TargetDummy {
    fn default() -> Self {
        Self { respawn_delay: 3.0, fall_speed: 300.0, points: 10, knocked: false, timer: 0.0, angle: 0.0, base: None }
    }
}

impl TargetDummy {
    /// True when the dummy is tipped over.
    pub fn knocked(&self) -> bool {
        self.knocked
    }
}

impl TypedComponent for TargetDummy {
    const NAME: &'static str = "Target Dummy";
}

impl TargetDummy {
    fn start_native(&mut self, ctx: &mut ObjectContext<'_, '_>) {
        self.base = Some(ctx.rotation());
    }

    fn update_native(&mut self, ctx: &mut ObjectContext<'_, '_>) {
        let base = *self.base.get_or_insert(ctx.rotation());
        let dead = ctx.world.get::<Health>(ctx.owner).is_some_and(|h| h.is_dead());
        if dead && !self.knocked {
            self.knocked = true;
            self.timer = self.respawn_delay;
            ctx.emit(ScoreEvent { points: self.points });
        }
        if self.knocked {
            self.timer -= ctx.dt;
            if self.timer <= 0.0 {
                self.knocked = false;
                ctx.edit::<Health>(ctx.self_id(), |h| h.revive());
            }
        }
        let goal = if self.knocked { 85.0 } else { 0.0 };
        let step = self.fall_speed * ctx.dt;
        self.angle += (goal - self.angle).clamp(-step, step);
        ctx.set_rotation(base * Quat::from_rotation_x(-self.angle.to_radians()));
    }
}

pub(crate) fn register_systems(registry: &mut crate::ComponentRegistry) {
    registry.register_ecs::<Health>();
    registry.register_ecs::<HitscanWeapon>();
    registry.register_ecs::<TargetDummy>();
    registry.register_system_if("zestengine::combat_start", Stage::Start, start, |world|
        world.query::<&Health>().iter().any(|(_, value)| !value.started)
        || world.query::<&HitscanWeapon>().iter().any(|(_, value)| !value.started)
        || world.query::<&TargetDummy>().iter().any(|(_, value)| value.base.is_none()));
    // Movement/tracking precedes shots, shots precede death handling, and
    // dummies see the resulting health before deciding whether to stand up.
    registry.register_system_for::<HitscanWeapon>("zestengine::hitscan", Stage::Gameplay, weapons);
    registry.register_system_for::<Health>("zestengine::health", Stage::Gameplay, health);
    registry.register_system_for::<TargetDummy>("zestengine::target_dummies", Stage::Gameplay, dummies);
}

fn start(world: &SystemWorld<'_>, context: &mut SystemContext<'_>) {
    macro_rules! initialize { ($t:ty, $pending:expr) => {
        let mut query = world.query::<(&Owner, &mut $t)>();
        let mut actors: Vec<_> = query.iter().filter(|(_, (owner, value))| ($pending)(value) && context.is_active(world, owner.0)).collect();
        actors.sort_unstable_by_key(|(_, (owner, _))| context.order_key(world, owner.0));
        for (_, (owner, value)) in actors { value.start_native(&mut ObjectContext::new(world, owner.0, context)); }
    }; }
    initialize!(Health, |value: &Health| !value.started);
    initialize!(HitscanWeapon, |value: &HitscanWeapon| !value.started);
    initialize!(TargetDummy, |value: &TargetDummy| value.base.is_none());
}

fn weapons(world: &SystemWorld<'_>, context: &mut SystemContext<'_>) {
    let mut query = world.query::<(&Owner, &mut HitscanWeapon)>();
    let mut actors: Vec<_> = query.iter().filter(|(_, (owner, _))| context.is_active(world, owner.0)).collect();
    actors.sort_unstable_by_key(|(_, (owner, _))| context.order_key(world, owner.0));
    for (_, (owner, value)) in actors { value.update_native(&mut ObjectContext::new(world, owner.0, context)); }
}

fn health(world: &SystemWorld<'_>, context: &mut SystemContext<'_>) {
    let mut query = world.query::<(&Owner, &mut Health)>();
    let mut actors: Vec<_> = query.iter().filter(|(_, (owner, _))| context.is_active(world, owner.0)).collect();
    actors.sort_unstable_by_key(|(_, (owner, _))| context.order_key(world, owner.0));
    for (_, (owner, value)) in actors { value.update_native(&mut ObjectContext::new(world, owner.0, context)); }
}

fn dummies(world: &SystemWorld<'_>, context: &mut SystemContext<'_>) {
    let mut query = world.query::<(&Owner, &mut TargetDummy)>();
    let mut actors: Vec<_> = query.iter().filter(|(_, (owner, _))| context.is_active(world, owner.0)).collect();
    actors.sort_unstable_by_key(|(_, (owner, _))| context.order_key(world, owner.0));
    for (_, (owner, value)) in actors { value.update_native(&mut ObjectContext::new(world, owner.0, context)); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_combat_reads_current_tracking_and_updates_damage_particles_and_next_frame_score() {
        use crate::{AnyBox, ComponentCommand, ComponentRegistry, Grabbable, HudBinding, HudValue, InputState, ParticleEmitter, PhysicsQuery, RayHit, ScreenInfo, Transform, UiText, VrCharacter, VrFrame, VrLocomotion, VrTurning};
        use crate::ecs::SceneWorld;
        use std::cell::RefCell;
        struct Hits { target: ObjectId, rays: RefCell<Vec<(Vec3, Vec3)>> }
        impl PhysicsQuery for Hits {
            fn raycast(&self, origin: Vec3, dir: Vec3, _: f32, _: &RayFilter) -> Option<RayHit> {
                self.rays.borrow_mut().push((origin, dir));
                Some(RayHit { object: self.target, point: Vec3::new(4.0, 2.0, -3.0), normal: Vec3::Z, distance: 5.0 })
            }
            fn overlap_sphere(&self, _: Vec3, _: f32, _: &RayFilter) -> Vec<ObjectId> { Vec::new() }
            fn velocity(&self, _: ObjectId) -> Option<Vec3> { None }
            fn grounded(&self, _: ObjectId) -> bool { true }
        }
        let registry = ComponentRegistry::with_builtins();
        let mut scene = SceneWorld::new();
        let parent = scene.spawn("Parent");
        *scene.get_mut::<Transform>(parent).unwrap() = Transform { translation: Vec3::new(8.0, 0.0, -1.0), rotation: Quat::from_rotation_y(0.7), scale: Vec3::splat(1.5) };
        let rig = scene.spawn("Rig");
        scene.set_parent(rig, Some(parent), false).unwrap();
        let hand = scene.spawn("Tracked hand");
        scene.set_parent(hand, Some(rig), false).unwrap();
        let hand_id = scene.object_id(hand).unwrap();
        scene.insert(rig, VrCharacter { right_hand: hand_id.into(), locomotion: VrLocomotion::Disabled, turning: VrTurning::Disabled, ..Default::default() }).unwrap();
        let muzzle = scene.spawn("Muzzle");
        scene.insert(muzzle, ParticleEmitter::default()).unwrap();
        let impact = scene.spawn("Parented impact");
        scene.set_parent(impact, Some(parent), false).unwrap();
        scene.insert(impact, ParticleEmitter::default()).unwrap();
        let weapon = scene.spawn("Held weapon");
        scene.insert(weapon, Grabbable { held: true, trigger_down: true, ..Default::default() }).unwrap();
        scene.insert(weapon, HitscanWeapon { trigger: WeaponTrigger::Grabbed, automatic: false, spread_deg: 0.0, aim_from: hand_id.into(),
            muzzle: scene.object_id(muzzle).unwrap().into(), hit_effect: scene.object_id(impact).unwrap().into(), ..Default::default() }).unwrap();
        let target = scene.spawn("Target");
        scene.insert(target, Health { max: 25.0, current: 25.0, ..Default::default() }).unwrap();
        scene.insert(target, TargetDummy { points: 7, respawn_delay: 0.3, ..Default::default() }).unwrap();
        let hud = scene.spawn("Score");
        scene.insert(hud, HudBinding { value: HudValue::Score, ..Default::default() }).unwrap();
        scene.insert(hud, UiText::default()).unwrap();
        scene.propagate().unwrap();
        let hits = Hits { target: scene.object_id(target).unwrap(), rays: RefCell::new(Vec::new()) };
        let screen = ScreenInfo { vr: VrFrame { active: true, right_hand: Some((Vec3::new(0.3, 1.3, -0.4), Quat::from_rotation_y(0.4))), ..Default::default() }, ..Default::default() };
        let input = InputState::default();
        let mut commands = Vec::new();
        {
            let mut context = SystemContext::new(&input, &mut commands);
            context.dt = 0.1;
            context.services.physics = Some(&hits);
            context.services.screen = Some(&screen);
            for stage in [Stage::Start, Stage::Update, Stage::Gameplay, Stage::LateUpdate] { registry.run_systems(stage, &scene, &mut context); }
        }
        let actual = hits.rays.borrow();
        assert_eq!(actual.len(), 1);
        scene.with_world(|world| {
            let hand = world.world_transform(hand).unwrap();
            assert!(actual[0].0.abs_diff_eq(hand.w_axis.truncate(), 1e-5), "shot origin uses this frame's tracked hand");
            assert!(actual[0].1.abs_diff_eq(hand.to_scale_rotation_translation().1 * Vec3::NEG_Z, 1e-5));
            let impact = world.world_transform(impact).unwrap();
            assert!(impact.w_axis.truncate().abs_diff_eq(Vec3::new(4.0, 2.0, -2.98), 1e-5));
        });
        assert!(scene.get::<Health>(target).unwrap().is_dead());
        assert!(scene.get::<TargetDummy>(target).unwrap().knocked());
        assert_eq!(scene.get::<ParticleEmitter>(muzzle).unwrap().control.emit, 12);
        assert_eq!(scene.get::<ParticleEmitter>(impact).unwrap().control.emit, 16);
        assert_eq!(scene.get::<UiText>(hud).unwrap().text, "0", "events are delivered next frame");
        assert!(!commands.iter().any(|command| matches!(command, ComponentCommand::Modify { .. } | ComponentCommand::SetLocalTransform { .. } | ComponentCommand::SetWorldTransform { .. })));
        let events: Vec<AnyBox> = commands.drain(..).filter_map(|command| match command { ComponentCommand::Emit(event) => Some(event), _ => None }).collect();
        assert_eq!(events.iter().filter(|event| event.0.is::<DeathEvent>()).count(), 1);
        assert_eq!(events.iter().filter(|event| event.0.is::<DamageEvent>()).count(), 1);
        let mut context = SystemContext::new(&input, &mut commands);
        context.services.events = &events;
        registry.run_systems(Stage::LateUpdate, &scene, &mut context);
        assert_eq!(scene.get::<UiText>(hud).unwrap().text, "7");
        assert_eq!(scene.get::<HudBinding>(hud).unwrap().score, 7);
        for _ in 0..3 {
            let mut context = SystemContext::new(&input, &mut commands);
            context.dt = 0.1;
            registry.run_systems(Stage::Gameplay, &scene, &mut context);
        }
        assert!(!scene.get::<TargetDummy>(target).unwrap().knocked());
        assert_eq!(scene.get::<Health>(target).unwrap().current, 25.0);
        assert!(!commands.iter().any(|command| matches!(command, ComponentCommand::Emit(event) if event.0.is::<DeathEvent>())));
    }

    #[test]
    fn native_health_initializes_once_and_queues_death_actions_once() {
        use crate::{ComponentCommand, ComponentRegistry, InputState};
        use crate::ecs::SceneWorld;
        let registry = ComponentRegistry::with_builtins();
        for action in [DeathAction::Disable, DeathAction::Destroy] {
            let mut scene = SceneWorld::new();
            let entity = scene.spawn("Health");
            let id = scene.object_id(entity).unwrap();
            scene.insert(entity, Health { max: 40.0, current: 0.0, on_death: action, ..Default::default() }).unwrap();
            let input = InputState::default();
            let mut commands = Vec::new();
            let mut context = SystemContext::new(&input, &mut commands);
            registry.run_systems(Stage::Start, &scene, &mut context);
            scene.get_mut::<Health>(entity).unwrap().apply_damage(100.0);
            registry.run_systems(Stage::Start, &scene, &mut context);
            for _ in 0..2 { registry.run_systems(Stage::Gameplay, &scene, &mut context); }
            assert!(scene.get::<Health>(entity).unwrap().is_dead(), "repeated startup must not revive a dead instance");
            assert_eq!(commands.iter().filter(|command| matches!(command, ComponentCommand::Emit(event) if event.0.is::<DeathEvent>())).count(), 1);
            assert_eq!(commands.iter().filter(|command| match command {
                ComponentCommand::SetActive(target, false) => *target == id && action == DeathAction::Disable,
                ComponentCommand::Destroy(target) => *target == id && action == DeathAction::Destroy,
                _ => false,
            }).count(), 1);
            scene.remove::<Health>(entity).unwrap();
            scene.insert(entity, Health { max: 9.0, current: 0.0, ..Default::default() }).unwrap();
            registry.run_systems(Stage::Start, &scene, &mut SystemContext::new(&input, &mut commands));
            assert_eq!(scene.get::<Health>(entity).unwrap().current, 9.0);
        }
    }

    #[test]
    fn damage_kills_once_and_revive_restores() {
        let mut h = Health { max: 50.0, current: 50.0, ..Default::default() };
        assert!(!h.apply_damage(20.0));
        assert_eq!(h.current, 30.0);
        assert!(h.apply_damage(40.0));
        assert!(h.is_dead());
        assert!(!h.apply_damage(10.0), "already dead");
        h.heal(10.0);
        assert!(h.is_dead(), "no healing the dead");
        h.revive();
        assert_eq!(h.current, 50.0);
        h.invulnerable = true;
        assert!(!h.apply_damage(100.0));
        assert_eq!(h.fraction(), 1.0);
    }

    #[test]
    fn reload_draws_from_reserve() {
        let mut w = HitscanWeapon { magazine_size: 10, ammo: 3, reserve: 5, ..Default::default() };
        w.start_reload();
        assert!(w.reloading());
        w.finish_reload();
        assert_eq!((w.ammo, w.reserve), (8, 0));
        w.reload_timer = 0.0;
        w.start_reload();
        assert!(!w.reloading(), "empty reserve can't reload");
        w.infinite_reserve = true;
        w.start_reload();
        w.finish_reload();
        assert_eq!(w.ammo, 10);
    }

    #[test]
    fn spread_stays_inside_cone() {
        let mut seed = 12345;
        for _ in 0..200 {
            let d = spread_direction(Vec3::NEG_Z, 5.0, &mut seed);
            assert!((d.length() - 1.0).abs() < 1e-4);
            assert!(d.angle_between(Vec3::NEG_Z).to_degrees() <= 5.01);
        }
        assert_eq!(spread_direction(Vec3::X, 0.0, &mut seed), Vec3::X);
    }
}
