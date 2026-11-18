//! Wheeled-vehicle physics: wheel suspension and a simple longitudinal drive
//! model, plus the [`Vehicle`] component that reads the driving controls.

use serde::{Deserialize, Serialize};

use crate::actions::{ActionCtx, ActionDesc, ActionType, ArgValue, PropertyDesc};
use crate::ecs::{ObjectContext, Owner, Stage, SystemContext, SystemWorld, run_ordered};
use crate::{ComponentCommand, InputState, MotionContext, ObjectId, ObjectRef, TypedComponent};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VehicleEntered {
    pub vehicle: ObjectId,
    pub driver: ObjectId,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VehicleExited {
    pub vehicle: ObjectId,
    pub driver: ObjectId,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VehicleBoarded {
    pub vehicle: ObjectId,
    pub passenger: ObjectId,
    pub seat: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VehicleUnboarded {
    pub vehicle: ObjectId,
    pub passenger: ObjectId,
    pub seat: usize,
}

const SEAT_DEBOUNCE: f32 = 0.3;

/// One wheel of a wheeled vehicle. Positions are in the vehicle object's
/// local space (metres), forward = local -Z.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Wheel {
    /// Wheel centre in the vehicle's local space.
    pub anchor: [f32; 3],
    /// Suspension rest length (metres): how far the wheel hangs below `anchor`.
    pub rest_length: f32,
    /// Spring stiffness (N/m).
    pub spring: f32,
    /// Damping (N·s/m).
    pub damper: f32,
    /// Wheel radius (metres).
    pub radius: f32,
    /// How far the suspension can compress from rest (metres).
    pub travel: f32,
    /// This wheel steers.
    pub steer: bool,
    /// This wheel is driven by the engine and brakes.
    pub drive: bool,
    /// Tyre grip (0 = ice, 1 = road tyres).
    pub grip: f32,
}

impl Default for Wheel {
    fn default() -> Self {
        Self {
            anchor: [0.0, -0.45, 0.0],
            rest_length: 0.35,
            spring: 60_000.0,
            damper: 2_500.0,
            radius: 0.35,
            travel: 0.25,
            steer: false,
            drive: true,
            grip: 1.0,
        }
    }
}

/// Suspension force (N, up) for a wheel from its compression and compression
/// velocity: the spring pushes back against compression, and the damper resists
/// motion (adds force while compressing).
pub fn suspension_force(wheel: &Wheel, compression: f32, compression_velocity: f32) -> f32 {
    compression.max(0.0) * wheel.spring + compression_velocity * wheel.damper
}

/// Net longitudinal drive force (N). Throttle (0..1) accelerates; brake (0..1)
/// decelerates against the current travel direction, so the vehicle can also
/// reverse under brake.
pub fn drive_force(engine: f32, throttle: f32, brake: f32, forward_speed: f32) -> f32 {
    let accel = throttle.clamp(0.0, 1.0) * engine;
    let brake_dir = if forward_speed >= 0.0 { -1.0 } else { 1.0 };
    accel + brake.clamp(0.0, 1.0) * engine * brake_dir
}

/// A raycast-suspension wheeled vehicle. Put it on the same object as a
/// dynamic [`crate::RigidBody`] (and a chassis collider held above the wheel
/// contact points). Controls default to the First / Third Person Controller's
/// `Move` / `Jump` / `Sprint` actions, so the same WASD + Space + Shift scheme
/// drives it: `Move.y` throttles and brakes (reverse), `Move.x` steers, Jump
/// brakes and Sprint applies the handbrake.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct Vehicle {
    /// The wheels.
    pub wheels: Vec<Wheel>,
    /// Maximum steering angle of the steered wheels, degrees.
    pub steering_degrees: f32,
    /// Drive force (Newtons) split across the driven wheels; braking uses the
    /// same force.
    pub engine_force: f32,
    /// Forward speed (m/s) the engine force tapers off at.
    pub top_speed: f32,
    /// Downforce at `top_speed`, pushing the chassis into the road (Newtons).
    pub downforce: f32,
    /// 2D action: x = steering, y = throttle / brake.
    pub move_action: String,
    /// Button action held to brake.
    pub brake_action: String,
    /// Button action held to apply the handbrake to the non-steered wheels.
    pub handbrake_action: String,
    pub requires_driver: bool,
    pub hide_driver: bool,
    pub exit_action: String,
    pub exit_offset: [f32; 3],
    pub camera: ObjectRef,
    pub passenger_seats: Vec<[f32; 3]>,
    pub seat_driver: bool,
    pub driver_seat: [f32; 3],
    pub driver_figure: ObjectRef,
    pub passenger_figures: Vec<ObjectRef>,
    #[serde(skip)]
    pub figures_synced: bool,
    #[serde(skip)]
    pub passengers: Vec<Option<ObjectId>>,
    #[serde(skip)]
    pub pending_unboard: Vec<ObjectId>,
    #[serde(skip)]
    pub driver: Option<ObjectId>,
    #[serde(skip)]
    pub seated_for: f32,
    #[serde(skip)]
    pub pending_exit: bool,

    // Transient control state, written by the native fixed-step system and read by the
    // engine's vehicle step.
    /// Throttle input 0..1 (runtime, written each fixed step).
    #[serde(skip)]
    pub throttle: f32,
    /// Brake input 0..1 (runtime).
    #[serde(skip)]
    pub brake: f32,
    /// Steering input -1..1 (runtime).
    #[serde(skip)]
    pub steer: f32,
    /// Handbrake on (runtime).
    #[serde(skip)]
    pub handbrake: bool,
}

fn default_wheels() -> Vec<Wheel> {
    let mut wheels = Vec::with_capacity(4);
    for (x, z, steer) in [(-0.8, -1.2, true), (0.8, -1.2, true), (-0.8, 1.2, false), (0.8, 1.2, false)] {
        wheels.push(Wheel {
            anchor: [x, -0.45, z],
            steer,
            ..Default::default()
        });
    }
    wheels
}

impl Default for Vehicle {
    fn default() -> Self {
        Self {
            wheels: default_wheels(),
            steering_degrees: 30.0,
            engine_force: 5000.0,
            top_speed: 25.0,
            downforce: 800.0,
            move_action: "Move".into(),
            brake_action: "Jump".into(),
            handbrake_action: "Sprint".into(),
            requires_driver: false,
            hide_driver: true,
            exit_action: "Interact".into(),
            exit_offset: [-1.6, 0.0, 0.0],
            camera: ObjectRef::NONE,
            passenger_seats: Vec::new(),
            seat_driver: false,
            driver_seat: [0.0, 0.5, 0.0],
            driver_figure: ObjectRef::NONE,
            passenger_figures: Vec::new(),
            figures_synced: false,
            passengers: Vec::new(),
            pending_unboard: Vec::new(),
            driver: None,
            seated_for: 0.0,
            pending_exit: false,
            throttle: 0.0,
            brake: 0.0,
            steer: 0.0,
            handbrake: false,
        }
    }
}

pub const SEATED_OFF: [&str; 5] = ["Character", "First Person Character", "Third Person Character", "VR Character", "Click To Move"];

impl Vehicle {
    /// Read the driving controls from an input snapshot into the transient
    /// control state, so it can be tested without a runtime scene.
    pub fn read_input(&mut self, input: &InputState) {
        let mv = input.axis2(&self.move_action);
        self.steer = mv.x.clamp(-1.0, 1.0);
        self.throttle = mv.y.clamp(0.0, 1.0);
        self.brake = ((-mv.y).max(0.0) + input.down(&self.brake_action) as u8 as f32).clamp(0.0, 1.0);
        self.handbrake = input.down(&self.handbrake_action);
    }
}

impl Vehicle {
    pub fn drivable(&self) -> bool {
        !self.requires_driver || self.driver.is_some()
    }

    fn park(&mut self) {
        self.throttle = 0.0;
        self.brake = 0.0;
        self.steer = 0.0;
        self.handbrake = self.requires_driver;
    }

    pub fn enter(&mut self, vehicle: ObjectId, driver: ObjectId, commands: &mut Vec<ComponentCommand>) -> bool {
        if self.driver.is_some() || driver == vehicle {
            return false;
        }
        self.driver = Some(driver);
        self.seated_for = 0.0;
        self.pending_exit = false;
        if self.hide_driver {
            commands.push(ComponentCommand::SetActive(driver, false));
        } else if self.seat_driver {
            commands.push(ComponentCommand::SuspendComponents { id: driver, names: SEATED_OFF.iter().map(|name| name.to_string()).collect() });
        }
        if let Some(camera) = self.camera.id() {
            commands.push(ComponentCommand::SetActiveCamera(camera));
        }
        if let Some(figure) = self.driver_figure.id() {
            commands.push(ComponentCommand::SetActive(figure, true));
        }
        commands.push(ComponentCommand::Emit(crate::AnyBox(Box::new(VehicleEntered { vehicle, driver }))));
        true
    }

    pub fn rides_visibly(&self) -> bool {
        self.seat_driver && !self.hide_driver && self.driver.is_some()
    }

    pub fn has_figures(&self) -> bool {
        self.driver_figure.id().is_some() || self.passenger_figures.iter().any(|figure| figure.id().is_some())
    }

    fn passenger_figure(&self, seat: usize) -> Option<ObjectId> {
        self.passenger_figures.get(seat).and_then(ObjectRef::id)
    }

    fn sync_figures(&mut self, commands: &mut Vec<ComponentCommand>) {
        if self.figures_synced {
            return;
        }
        self.figures_synced = true;
        if let Some(figure) = self.driver_figure.id() {
            commands.push(ComponentCommand::SetActive(figure, self.driver.is_some()));
        }
        for seat in 0..self.passenger_figures.len() {
            if let Some(figure) = self.passenger_figure(seat) {
                commands.push(ComponentCommand::SetActive(figure, self.passengers.get(seat).is_some_and(Option::is_some)));
            }
        }
    }

    pub fn passenger_count(&self) -> usize {
        self.passengers.iter().flatten().count()
    }

    pub fn carries(&self, object: ObjectId) -> bool {
        self.driver == Some(object) || self.passengers.contains(&Some(object))
    }

    pub fn board(&mut self, vehicle: ObjectId, passenger: ObjectId, commands: &mut Vec<ComponentCommand>) -> bool {
        if passenger == vehicle || self.carries(passenger) {
            return false;
        }
        self.passengers.resize(self.passenger_seats.len(), None);
        let Some(seat) = self.passengers.iter().position(Option::is_none) else {
            return false;
        };
        self.passengers[seat] = Some(passenger);
        commands.push(ComponentCommand::SetActive(passenger, false));
        if let Some(figure) = self.passenger_figure(seat) {
            commands.push(ComponentCommand::SetActive(figure, true));
        }
        commands.push(ComponentCommand::Emit(crate::AnyBox(Box::new(VehicleBoarded { vehicle, passenger, seat }))));
        true
    }

    fn unboard_native(&mut self, obj: &mut ObjectContext<'_, '_>) {
        let leaving = std::mem::take(&mut self.pending_unboard);
        if leaving.is_empty() {
            return;
        }
        let vehicle = obj.self_id();
        let pose = obj.world.world_transform(obj.owner);
        for passenger in leaving {
            let Some(seat) = self.passengers.iter().position(|p| *p == Some(passenger)) else {
                continue;
            };
            self.passengers[seat] = None;
            if let Some(pose) = pose {
                let (_, rotation, _) = pose.to_scale_rotation_translation();
                let (yaw, _, _) = rotation.to_euler(glam::EulerRot::YXZ);
                let offset = self.passenger_seats.get(seat).copied().unwrap_or(self.exit_offset);
                obj.commands.push(ComponentCommand::SetWorldTransform {
                    id: passenger,
                    translation: Some(pose.transform_point3(glam::Vec3::from(offset))),
                    rotation: Some(glam::Quat::from_rotation_y(yaw)),
                });
            }
            obj.commands.push(ComponentCommand::SetActive(passenger, true));
            if let Some(figure) = self.passenger_figure(seat) {
                obj.commands.push(ComponentCommand::SetActive(figure, false));
            }
            obj.emit(VehicleUnboarded { vehicle, passenger, seat });
        }
    }

    fn seat_native(&mut self, obj: &mut ObjectContext<'_, '_>) {
        self.sync_figures(obj.commands);
        self.unboard_native(obj);
        let Some(driver) = self.driver else {
            self.pending_exit = false;
            return;
        };
        self.seated_for += obj.dt;
        if self.rides_visibly()
            && let Some(pose) = obj.world.world_transform(obj.owner)
        {
            let (_, rotation, _) = pose.to_scale_rotation_translation();
            obj.commands.push(ComponentCommand::SetWorldTransform {
                id: driver,
                translation: Some(pose.transform_point3(glam::Vec3::from(self.driver_seat))),
                rotation: Some(rotation),
            });
        }
        let typing = obj.services.screen.is_some_and(|screen| screen.ui_text_editing);
        let pressed = self.seated_for >= SEAT_DEBOUNCE && !typing && !self.exit_action.is_empty() && obj.input.pressed(&self.exit_action);
        if !self.pending_exit && !pressed {
            return;
        }
        self.pending_exit = false;
        self.driver = None;
        self.seated_for = 0.0;
        self.park();
        let vehicle = obj.self_id();
        if let Some(pose) = obj.world.world_transform(obj.owner) {
            let (_, rotation, _) = pose.to_scale_rotation_translation();
            let (yaw, _, _) = rotation.to_euler(glam::EulerRot::YXZ);
            obj.commands.push(ComponentCommand::SetWorldTransform {
                id: driver,
                translation: Some(pose.transform_point3(glam::Vec3::from(self.exit_offset))),
                rotation: Some(glam::Quat::from_rotation_y(yaw)),
            });
        }
        if self.hide_driver {
            obj.commands.push(ComponentCommand::SetActive(driver, true));
        } else if self.seat_driver {
            obj.commands.push(ComponentCommand::ResumeComponents(driver));
        }
        if let Some(camera) = self.camera.id() {
            obj.commands.push(ComponentCommand::ReleaseActiveCamera(camera));
        }
        if let Some(figure) = self.driver_figure.id() {
            obj.commands.push(ComponentCommand::SetActive(figure, false));
        }
        obj.emit(VehicleExited { vehicle, driver });
    }
}

static VEHICLE_ACTIONS: &[ActionDesc] = &[
    ActionDesc::new("Enter", "Enter", &[], true),
    ActionDesc::new("Exit", "Exit", &[], true),
    ActionDesc::new("Board", "Board As Passenger", &[], true),
    ActionDesc::new("Unboard", "Passenger Leaves", &[], true),
    ActionDesc::new("UnboardAll", "All Passengers Leave", &[], true),
];
static VEHICLE_PROPERTIES: &[PropertyDesc] = &[
    PropertyDesc::new("occupied", ActionType::Bool),
    PropertyDesc::new("passengers", ActionType::Int),
    PropertyDesc::new("free_seats", ActionType::Int),
    PropertyDesc::new("throttle", ActionType::Float),
    PropertyDesc::new("steer", ActionType::Float),
];

impl TypedComponent for Vehicle {
    const NAME: &'static str = "Vehicle";

    fn actions(&self) -> &'static [ActionDesc] {
        VEHICLE_ACTIONS
    }

    fn invoke_action(&mut self, ctx: &mut ActionCtx, name: &str, _args: &[ArgValue]) -> Result<(), String> {
        match name {
            "Enter" => {
                let driver = ctx.instigator.ok_or("Enter needs to know who is getting in (bind it to an event with an instigator, such as an Interactable's)")?;
                self.enter(ctx.self_id, driver, ctx.commands);
            }
            "Exit" => self.pending_exit = self.driver.is_some(),
            "Board" => {
                let passenger = ctx.instigator.ok_or("Board needs to know who is getting in (bind it to an event with an instigator)")?;
                if !self.board(ctx.self_id, passenger, ctx.commands) && !self.carries(passenger) {
                    return Err("no free passenger seat".into());
                }
            }
            "Unboard" => {
                let passenger = ctx.instigator.ok_or("Unboard needs to know who is getting out")?;
                if self.passengers.contains(&Some(passenger)) && !self.pending_unboard.contains(&passenger) {
                    self.pending_unboard.push(passenger);
                }
            }
            "UnboardAll" => self.pending_unboard = self.passengers.iter().flatten().copied().collect(),
            _ => return Err(crate::actions::unknown_action(Self::NAME, name)),
        }
        Ok(())
    }

    fn properties(&self) -> &'static [PropertyDesc] {
        VEHICLE_PROPERTIES
    }

    fn read_property(&self, name: &str) -> Option<ArgValue> {
        match name {
            "occupied" => Some(ArgValue::Bool(self.driver.is_some())),
            "passengers" => Some(ArgValue::Int(self.passenger_count() as i64)),
            "free_seats" => Some(ArgValue::Int(self.passenger_seats.len().saturating_sub(self.passenger_count()) as i64)),
            "throttle" => Some(ArgValue::Float(self.throttle)),
            "steer" => Some(ArgValue::Float(self.steer)),
            _ => None,
        }
    }
}

pub(crate) fn register_seat_system(registry: &mut crate::ComponentRegistry) {
    registry.register_system_if("zestengine::vehicle_seats", Stage::Update, |world, ctx| run_ordered(world, ctx, Vehicle::seat_native), |world| {
        world.query::<&Vehicle>().iter().any(|(_, vehicle)| vehicle.driver.is_some() || vehicle.pending_exit || !vehicle.pending_unboard.is_empty() || (!vehicle.figures_synced && vehicle.has_figures()))
    });
}

/// Native fixed-step controls. Data remains in the same typed Vehicle column
/// that suspension reads; remote authority and focused UI suppress local input.
pub(crate) fn controls(world: &SystemWorld<'_>, ctx: &mut SystemContext<'_>) {
    for (_, (owner, vehicle)) in world.query::<(&Owner, &mut Vehicle)>().iter() {
        if !ctx.is_active(world, owner.0) { continue; }
        let remote = ctx.services.net.is_some_and(|net| net.connected
            && (world.get::<crate::NetworkObject>(owner.0).is_some() || world.get::<crate::Sync>(owner.0).is_some())
            && !net.has_authority(world.object_id(owner.0).expect("component owner")));
        if !vehicle.drivable() {
            vehicle.park();
        } else if remote || ctx.services.screen.is_some_and(|screen| screen.ui_text_editing) {
            vehicle.throttle = 0.0;
            vehicle.brake = 0.0;
            vehicle.steer = 0.0;
            vehicle.handbrake = false;
        } else { vehicle.read_input(ctx.input); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ActionValue, InputState};

    fn input(actions: Vec<(String, ActionValue)>) -> InputState {
        InputState::from_actions(actions, glam::Vec2::ZERO)
    }

    fn emitted<E: Copy + 'static>(commands: &[ComponentCommand]) -> Vec<E> {
        commands
            .iter()
            .filter_map(|c| match c {
                ComponentCommand::Emit(event) => event.0.downcast_ref::<E>().copied(),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_vehicle_that_needs_a_driver_is_parked_until_someone_gets_in_and_out_again() {
        use crate::ecs::SceneWorld;
        use crate::ComponentRegistry;
        let mut scene = SceneWorld::new();
        let car = scene.spawn("Car");
        let player = scene.spawn("Player");
        let camera = scene.spawn("Chase camera");
        scene.get_mut::<crate::Transform>(car).unwrap().translation = glam::Vec3::new(10.0, 0.0, 5.0);
        let (car_id, player_id, camera_id) = (scene.object_id(car).unwrap(), scene.object_id(player).unwrap(), scene.object_id(camera).unwrap());
        scene.insert(car, Vehicle { requires_driver: true, camera: ObjectRef::new(camera_id), ..Default::default() }).unwrap();
        scene.propagate().unwrap();
        let forward = ActionValue { down: true, value: 1.0, axis2: glam::Vec2::new(0.0, 1.0), ..Default::default() };
        let exit = ActionValue { down: true, just_pressed: true, value: 1.0, ..Default::default() };
        let registry = ComponentRegistry::with_builtins();
        let step = |scene: &mut SceneWorld, stage: Stage, actions: Vec<(String, ActionValue)>, dt: f32| {
            let state = input(actions);
            let mut commands = Vec::new();
            let mut ctx = SystemContext::new(&state, &mut commands);
            ctx.dt = dt;
            registry.run_systems(stage, scene, &mut ctx);
            commands
        };

        step(&mut scene, Stage::FixedUpdate, vec![("Move".into(), forward)], 1.0 / 60.0);
        {
            let vehicle = scene.get::<Vehicle>(car).unwrap();
            assert_eq!((vehicle.throttle, vehicle.handbrake), (0.0, true), "nobody is driving: the player's input does not move it");
        }

        let mut commands = Vec::new();
        {
            let mut vehicle = scene.get_mut::<Vehicle>(car).unwrap();
            let mut ctx = ActionCtx { self_id: car_id, instigator: None, commands: &mut commands, reader: None };
            assert!(TypedComponent::invoke_action(&mut *vehicle, &mut ctx, "Enter", &[]).is_err(), "Enter needs an instigator");
            let mut ctx = ActionCtx { self_id: car_id, instigator: Some(player_id), commands: &mut commands, reader: None };
            TypedComponent::invoke_action(&mut *vehicle, &mut ctx, "Enter", &[]).unwrap();
            TypedComponent::invoke_action(&mut *vehicle, &mut ctx, "Enter", &[]).unwrap();
            assert_eq!(TypedComponent::read_property(&*vehicle, "occupied"), Some(ArgValue::Bool(true)));
        }
        assert_eq!(emitted::<VehicleEntered>(&commands), vec![VehicleEntered { vehicle: car_id, driver: player_id }], "a second Enter while occupied does nothing");
        assert!(commands.iter().any(|c| matches!(c, ComponentCommand::SetActive(id, false) if *id == player_id)), "the driver's own object is switched off");
        assert!(commands.iter().any(|c| matches!(c, ComponentCommand::SetActiveCamera(id) if *id == camera_id)));

        step(&mut scene, Stage::FixedUpdate, vec![("Move".into(), forward)], 1.0 / 60.0);
        assert!(scene.get::<Vehicle>(car).unwrap().throttle > 0.9, "the driver's input drives it");

        let early = step(&mut scene, Stage::Update, vec![("Interact".into(), exit)], 0.1);
        assert!(emitted::<VehicleExited>(&early).is_empty(), "the press that got the driver in does not throw them out again");
        let out = step(&mut scene, Stage::Update, vec![("Interact".into(), exit)], 0.3);
        assert_eq!(emitted::<VehicleExited>(&out), vec![VehicleExited { vehicle: car_id, driver: player_id }]);
        let placed = out.iter().find_map(|c| match c {
            ComponentCommand::SetWorldTransform { id, translation, .. } if *id == player_id => *translation,
            _ => None,
        });
        assert_eq!(placed, Some(glam::Vec3::new(8.4, 0.0, 5.0)), "the driver stands beside the door");
        assert!(out.iter().any(|c| matches!(c, ComponentCommand::SetActive(id, true) if *id == player_id)));
        assert!(out.iter().any(|c| matches!(c, ComponentCommand::ReleaseActiveCamera(id) if *id == camera_id)));
        let vehicle = scene.get::<Vehicle>(car).unwrap();
        assert!(vehicle.driver.is_none() && vehicle.handbrake && vehicle.throttle == 0.0);
    }

    #[test]
    fn a_visible_driver_rides_in_the_seat_with_their_controller_suspended() {
        use crate::ecs::SceneWorld;
        use crate::ComponentRegistry;
        let mut scene = SceneWorld::new();
        let car = scene.spawn("Car");
        let rider = scene.spawn("Rider");
        let (car_id, rider_id) = (scene.object_id(car).unwrap(), scene.object_id(rider).unwrap());
        scene.insert(car, Vehicle { requires_driver: true, hide_driver: false, seat_driver: true, driver_seat: [0.4, 0.6, 0.2], ..Default::default() }).unwrap();
        scene.get_mut::<crate::Transform>(car).unwrap().translation = glam::Vec3::new(10.0, 0.0, 0.0);
        scene.propagate().unwrap();
        let registry = ComponentRegistry::with_builtins();
        let mut commands = Vec::new();
        assert!(scene.get_mut::<Vehicle>(car).unwrap().enter(car_id, rider_id, &mut commands));
        assert!(commands.iter().any(|c| matches!(c, ComponentCommand::SuspendComponents { id, names } if *id == rider_id && names.iter().any(|n| n == "Character"))));
        assert!(!commands.iter().any(|c| matches!(c, ComponentCommand::SetActive(..))), "the rider stays visible");
        let state = input(Vec::new());
        let mut commands = Vec::new();
        let mut ctx = SystemContext::new(&state, &mut commands);
        ctx.dt = 0.016;
        registry.run_systems(Stage::Update, &mut scene, &mut ctx);
        let seated = commands.iter().find_map(|c| if let ComponentCommand::SetWorldTransform { id, translation: Some(t), .. } = c { (*id == rider_id).then_some(*t) } else { None });
        assert_eq!(seated, Some(glam::Vec3::new(10.4, 0.6, 0.2)), "the rider is held at the seat");
        {
            let mut vehicle = scene.get_mut::<Vehicle>(car).unwrap();
            let mut commands = Vec::new();
            let mut ctx = ActionCtx { self_id: car_id, instigator: None, commands: &mut commands, reader: None };
            TypedComponent::invoke_action(&mut *vehicle, &mut ctx, "Exit", &[]).unwrap();
        }
        let mut commands = Vec::new();
        let mut ctx = SystemContext::new(&state, &mut commands);
        ctx.dt = 0.016;
        registry.run_systems(Stage::Update, &mut scene, &mut ctx);
        let resume = commands.iter().position(|c| matches!(c, ComponentCommand::ResumeComponents(id) if *id == rider_id)).expect("controller resumes");
        let placed = commands.iter().rposition(|c| matches!(c, ComponentCommand::SetWorldTransform { id, .. } if *id == rider_id)).expect("placed beside the vehicle");
        assert!(placed < resume, "the rider is put down before the controller comes back");
        let old: Vehicle = ron::from_str("(hide_driver: false)").unwrap();
        assert!(!old.seat_driver, "older vehicles keep the walking rider");
    }

    #[test]
    fn seated_figures_show_only_while_their_seat_is_taken() {
        use crate::ecs::SceneWorld;
        use crate::ComponentRegistry;
        let mut scene = SceneWorld::new();
        let names = ["Car", "Player", "Friend", "Driver figure", "Passenger figure"];
        let [car, player, friend, driver_figure, passenger_figure] = names.map(|name| scene.spawn(name));
        let id = |scene: &SceneWorld, entity| scene.object_id(entity).unwrap();
        let (car_id, player_id, friend_id, driver_fig, passenger_fig) = (id(&scene, car), id(&scene, player), id(&scene, friend), id(&scene, driver_figure), id(&scene, passenger_figure));
        let vehicle = Vehicle {
            requires_driver: true,
            driver_figure: ObjectRef::new(driver_fig),
            passenger_seats: vec![[1.5, 0.0, 0.0], [1.5, 0.0, 1.0]],
            passenger_figures: vec![ObjectRef::new(passenger_fig)],
            ..Default::default()
        };
        scene.insert(car, vehicle).unwrap();
        scene.propagate().unwrap();
        let registry = ComponentRegistry::with_builtins();
        let state = input(Vec::new());
        let run = |scene: &mut SceneWorld| {
            let mut commands = Vec::new();
            let mut ctx = SystemContext::new(&state, &mut commands);
            ctx.dt = 0.016;
            registry.run_systems(Stage::Update, scene, &mut ctx);
            commands
        };
        let switches = |commands: &[ComponentCommand]| -> Vec<(ObjectId, bool)> {
            commands.iter().filter_map(|c| if let ComponentCommand::SetActive(id, on) = c { Some((*id, *on)) } else { None }).collect()
        };
        assert_eq!(switches(&run(&mut scene)), vec![(driver_fig, false), (passenger_fig, false)], "empty seats start with their figures hidden");
        assert!(switches(&run(&mut scene)).is_empty(), "only once");
        let mut commands = Vec::new();
        assert!(scene.get_mut::<Vehicle>(car).unwrap().enter(car_id, player_id, &mut commands));
        assert_eq!(switches(&commands), vec![(player_id, false), (driver_fig, true)], "the driver is swapped for the seated figure");
        let mut commands = Vec::new();
        assert!(scene.get_mut::<Vehicle>(car).unwrap().board(car_id, friend_id, &mut commands));
        assert_eq!(switches(&commands), vec![(friend_id, false), (passenger_fig, true)]);
        {
            let mut vehicle = scene.get_mut::<Vehicle>(car).unwrap();
            let mut commands = Vec::new();
            let mut ctx = ActionCtx { self_id: car_id, instigator: None, commands: &mut commands, reader: None };
            TypedComponent::invoke_action(&mut *vehicle, &mut ctx, "Exit", &[]).unwrap();
            TypedComponent::invoke_action(&mut *vehicle, &mut ctx, "UnboardAll", &[]).unwrap();
        }
        let after = switches(&run(&mut scene));
        assert!(after.contains(&(driver_fig, false)) && after.contains(&(passenger_fig, false)), "{after:?}");
        assert!(after.contains(&(player_id, true)) && after.contains(&(friend_id, true)));
        let text = ron::to_string(&Vehicle::default()).unwrap();
        assert!(!text.contains("figures_synced"));
        let old: Vehicle = ron::from_str("(requires_driver: true)").unwrap();
        assert!(!old.has_figures());
    }

    #[test]
    fn the_exit_action_works_without_the_button_and_old_vehicles_drive_as_before() {
        use crate::ecs::SceneWorld;
        use crate::ComponentRegistry;
        let mut scene = SceneWorld::new();
        let car = scene.spawn("Car");
        let rider = scene.spawn("Rider");
        let (car_id, rider_id) = (scene.object_id(car).unwrap(), scene.object_id(rider).unwrap());
        scene.insert(car, Vehicle { hide_driver: false, exit_action: String::new(), ..Default::default() }).unwrap();
        scene.propagate().unwrap();
        let registry = ComponentRegistry::with_builtins();
        let forward = ActionValue { down: true, value: 1.0, axis2: glam::Vec2::new(0.0, 1.0), ..Default::default() };
        let state = input(vec![("Move".into(), forward)]);
        let mut commands = Vec::new();
        let mut ctx = SystemContext::new(&state, &mut commands);
        registry.run_systems(Stage::FixedUpdate, &mut scene, &mut ctx);
        assert!(scene.get::<Vehicle>(car).unwrap().throttle > 0.9, "requires_driver is off by default");
        let mut commands = Vec::new();
        assert!(scene.get_mut::<Vehicle>(car).unwrap().enter(car_id, rider_id, &mut commands));
        assert!(!commands.iter().any(|c| matches!(c, ComponentCommand::SetActive(..) | ComponentCommand::SetActiveCamera(..))), "a visible rider and no camera: nothing is switched");
        {
            let mut vehicle = scene.get_mut::<Vehicle>(car).unwrap();
            let mut ctx = ActionCtx { self_id: car_id, instigator: None, commands: &mut commands, reader: None };
            TypedComponent::invoke_action(&mut *vehicle, &mut ctx, "Exit", &[]).unwrap();
        }
        let state = input(Vec::new());
        let mut commands = Vec::new();
        let mut ctx = SystemContext::new(&state, &mut commands);
        ctx.dt = 0.016;
        registry.run_systems(Stage::Update, &mut scene, &mut ctx);
        assert_eq!(emitted::<VehicleExited>(&commands).len(), 1);
        let text = crate::Component::save(&*scene.get::<Vehicle>(car).unwrap());
        assert!(!text.contains("seated_for") && !text.contains("pending_exit"));
        let old: Vehicle = ron::from_str("(engine_force: 4000.0)").unwrap();
        assert!(!old.requires_driver && old.hide_driver && old.exit_action == "Interact");
    }

    #[test]
    fn passengers_take_free_seats_and_leave_beside_their_own_seat() {
        use crate::ecs::SceneWorld;
        use crate::ComponentRegistry;
        let mut scene = SceneWorld::new();
        let bus = scene.spawn("Bus");
        scene.get_mut::<crate::Transform>(bus).unwrap().translation = glam::Vec3::new(0.0, 0.0, 20.0);
        let riders: Vec<ObjectId> = (0..3).map(|i| { let e = scene.spawn(&format!("Rider {i}")); scene.object_id(e).unwrap() }).collect();
        let bus_id = scene.object_id(bus).unwrap();
        scene.insert(bus, Vehicle { passenger_seats: vec![[1.6, 0.0, 0.0], [1.6, 0.0, 2.0]], ..Default::default() }).unwrap();
        scene.propagate().unwrap();
        let mut commands = Vec::new();
        {
            let mut vehicle = scene.get_mut::<Vehicle>(bus).unwrap();
            let act = |vehicle: &mut Vehicle, name: &str, who: Option<ObjectId>, commands: &mut Vec<ComponentCommand>| {
                let mut ctx = ActionCtx { self_id: bus_id, instigator: who, commands, reader: None };
                TypedComponent::invoke_action(vehicle, &mut ctx, name, &[])
            };
            assert!(act(&mut vehicle, "Board", None, &mut commands).is_err());
            act(&mut vehicle, "Board", Some(riders[0]), &mut commands).unwrap();
            act(&mut vehicle, "Board", Some(riders[0]), &mut commands).unwrap();
            act(&mut vehicle, "Board", Some(riders[1]), &mut commands).unwrap();
            assert!(act(&mut vehicle, "Board", Some(riders[2]), &mut commands).unwrap_err().contains("no free"));
            assert_eq!(TypedComponent::read_property(&*vehicle, "passengers"), Some(ArgValue::Int(2)));
            assert_eq!(TypedComponent::read_property(&*vehicle, "free_seats"), Some(ArgValue::Int(0)));
            assert!(vehicle.driver.is_none(), "passengers do not drive");
            act(&mut vehicle, "Unboard", Some(riders[1]), &mut commands).unwrap();
            act(&mut vehicle, "Unboard", Some(riders[2]), &mut commands).unwrap();
        }
        assert_eq!(emitted::<VehicleBoarded>(&commands).iter().map(|e| (e.passenger, e.seat)).collect::<Vec<_>>(), vec![(riders[0], 0), (riders[1], 1)]);
        assert_eq!(commands.iter().filter(|c| matches!(c, ComponentCommand::SetActive(_, false))).count(), 2);
        let registry = ComponentRegistry::with_builtins();
        let state = input(Vec::new());
        let mut out = Vec::new();
        let mut ctx = SystemContext::new(&state, &mut out);
        ctx.dt = 0.016;
        registry.run_systems(Stage::Update, &mut scene, &mut ctx);
        assert_eq!(emitted::<VehicleUnboarded>(&out), vec![VehicleUnboarded { vehicle: bus_id, passenger: riders[1], seat: 1 }]);
        let placed = out.iter().find_map(|c| match c {
            ComponentCommand::SetWorldTransform { id, translation, .. } if *id == riders[1] => *translation,
            _ => None,
        });
        assert_eq!(placed, Some(glam::Vec3::new(1.6, 0.0, 22.0)));
        assert!(out.iter().any(|c| matches!(c, ComponentCommand::SetActive(id, true) if *id == riders[1])));
        {
            let mut vehicle = scene.get_mut::<Vehicle>(bus).unwrap();
            assert_eq!(vehicle.passenger_count(), 1);
            let mut none = Vec::new();
            let mut ctx = ActionCtx { self_id: bus_id, instigator: None, commands: &mut none, reader: None };
            TypedComponent::invoke_action(&mut *vehicle, &mut ctx, "UnboardAll", &[]).unwrap();
        }
        let mut out = Vec::new();
        let mut ctx = SystemContext::new(&state, &mut out);
        registry.run_systems(Stage::Update, &mut scene, &mut ctx);
        assert_eq!(emitted::<VehicleUnboarded>(&out).len(), 1);
        assert_eq!(scene.get::<Vehicle>(bus).unwrap().passenger_count(), 0);
    }

    #[test]
    fn suspension_is_zero_at_rest_length() {
        let w = Wheel::default();
        assert_eq!(suspension_force(&w, 0.0, 0.0), 0.0);
        assert_eq!(suspension_force(&w, -0.1, 0.0), 0.0, "negative compression clamps");
    }

    #[test]
    fn suspension_spring_and_damper() {
        let w = Wheel::default();
        let f = suspension_force(&w, 0.1, 0.5);
        assert!((f - (0.1 * w.spring + 0.5 * w.damper)).abs() < 1e-3);
    }

    #[test]
    fn drive_throttles_and_brakes() {
        assert!(drive_force(1000.0, 1.0, 0.0, 0.0) > 0.0);
        assert!(drive_force(1000.0, 0.0, 1.0, 5.0) < 0.0, "brake opposes forward speed");
        assert!(drive_force(1000.0, 0.0, 1.0, -5.0) > 0.0, "brake opposes reverse speed");
        assert_eq!(drive_force(1000.0, 0.0, 0.0, 3.0), 0.0);
    }

    #[test]
    fn move_axis_steers_throttles_and_brakes() {
        let mut v = Vehicle::default();
        let mv = ActionValue {
            down: true,
            value: 1.0,
            axis2: glam::Vec2::new(0.6, 0.8),
            ..Default::default()
        };
        v.read_input(&input(vec![("Move".into(), mv)]));
        assert!((v.steer - 0.6).abs() < 1e-4);
        assert!((v.throttle - 0.8).abs() < 1e-4);
        assert_eq!(v.brake, 0.0);

        let back = ActionValue {
            down: true,
            value: 1.0,
            axis2: glam::Vec2::new(0.0, -1.0),
            ..Default::default()
        };
        v.read_input(&input(vec![("Move".into(), back)]));
        assert_eq!(v.throttle, 0.0);
        assert_eq!(v.brake, 1.0);
    }

    #[test]
    fn brake_and_handbrake_read_their_actions() {
        let mut v = Vehicle::default();
        let held = ActionValue { down: true, value: 1.0, ..Default::default() };
        v.read_input(&input(vec![
            ("Jump".into(), held),
            ("Sprint".into(), held),
        ]));
        assert_eq!(v.brake, 1.0);
        assert!(v.handbrake);
    }
}
