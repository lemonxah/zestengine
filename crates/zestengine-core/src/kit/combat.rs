//! Health, damage and hitscan weapons.

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::{
    ActionCtx, ActionDesc, ActionType, ArgValue, ComponentCtx, ObjectId, ObjectRef, ParamDesc, PropertyDesc, RayFilter,
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

/// Hit points. Weapons damage it with `ctx.modify::<Health>(target, |h| { h.apply_damage(10.0); })`.
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

    fn start(&mut self, _ctx: &mut ComponentCtx) {
        if self.current <= 0.0 || self.current > self.max {
            self.current = self.max;
        }
    }

    fn update(&mut self, ctx: &mut ComponentCtx) {
        if self.is_dead() {
            if !self.death_handled {
                self.death_handled = true;
                ctx.emit(DeathEvent { object: ctx.self_id() });
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
    /// Impact particle emitter, moved to each hit point (must be a root object).
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

    fn trigger_state(&self, ctx: &ComponentCtx) -> (bool, bool) {
        match self.trigger {
            WeaponTrigger::Action => (ctx.input.down(&self.fire_action), ctx.input.pressed(&self.fire_action)),
            WeaponTrigger::Grabbed => {
                let down = ctx.get_self::<crate::Grabbable>().is_some_and(|g| g.held && g.trigger_down);
                (down, down && !self.trigger_prev)
            }
        }
    }

    fn fire(&mut self, ctx: &mut ComponentCtx) {
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
        ctx.modify::<Health>(hit.object, move |h| {
            h.apply_damage(damage);
        });
        ctx.emit(DamageEvent { target: hit.object, source: ctx.self_id(), amount: damage, point: hit.point });
        if self.impact_force > 0.0 {
            ctx.add_impulse(hit.object, dir * self.impact_force);
        }
        if self.hit_effect.is_set() {
            let rot = crate::kit::look_rotation(hit.normal);
            ctx.set_local_transform(self.hit_effect, Some(hit.point + hit.normal * 0.02), Some(rot), None);
            ctx.emit_particles(self.hit_effect, self.hit_burst);
        }
        if !self.hit_sound.is_empty() {
            ctx.play_sound_at(self.hit_sound.clone(), hit.point, 0.8);
        }
    }
}

impl TypedComponent for HitscanWeapon {
    const NAME: &'static str = "Hitscan Weapon";

    fn start(&mut self, ctx: &mut ComponentCtx) {
        self.ammo = self.ammo.min(self.magazine_size);
        self.rng ^= (ctx.self_id().raw() as u32) | 1;
    }

    fn update(&mut self, ctx: &mut ComponentCtx) {
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

    fn start(&mut self, ctx: &mut ComponentCtx) {
        self.base = Some(*ctx.rotation);
    }

    fn update(&mut self, ctx: &mut ComponentCtx) {
        let base = *self.base.get_or_insert(*ctx.rotation);
        let dead = ctx.get_self::<Health>().is_some_and(|h| h.is_dead());
        if dead && !self.knocked {
            self.knocked = true;
            self.timer = self.respawn_delay;
            ctx.emit(ScoreEvent { points: self.points });
        }
        if self.knocked {
            self.timer -= ctx.dt;
            if self.timer <= 0.0 {
                self.knocked = false;
                ctx.modify::<Health>(ctx.self_id(), |h| h.revive());
            }
        }
        let goal = if self.knocked { 85.0 } else { 0.0 };
        let step = self.fall_speed * ctx.dt;
        self.angle += (goal - self.angle).clamp(-step, step);
        *ctx.rotation = base * Quat::from_rotation_x(-self.angle.to_radians());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
