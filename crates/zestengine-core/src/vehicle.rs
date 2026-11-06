//! Wheeled-vehicle physics: wheel suspension and a simple longitudinal drive
//! model, plus the [`Vehicle`] component that reads the driving controls.

use serde::{Deserialize, Serialize};

use crate::{ComponentCtx, InputState, TypedComponent, driven_remotely};

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

    // Transient control state, written by `fixed_update` and read by the
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
            throttle: 0.0,
            brake: 0.0,
            steer: 0.0,
            handbrake: false,
        }
    }
}

impl Vehicle {
    /// Read the driving controls from an input snapshot into the transient
    /// control state, so it can be tested without a full [`ComponentCtx`].
    pub fn read_input(&mut self, input: &InputState) {
        let mv = input.axis2(&self.move_action);
        self.steer = mv.x.clamp(-1.0, 1.0);
        self.throttle = mv.y.clamp(0.0, 1.0);
        self.brake = ((-mv.y).max(0.0) + input.down(&self.brake_action) as u8 as f32).clamp(0.0, 1.0);
        self.handbrake = input.down(&self.handbrake_action);
    }
}

impl TypedComponent for Vehicle {
    const NAME: &'static str = "Vehicle";

    fn fixed_update(&mut self, ctx: &mut ComponentCtx) {
        if driven_remotely(ctx) {
            self.throttle = 0.0;
            self.brake = 0.0;
            self.steer = 0.0;
            self.handbrake = false;
            return;
        }
        if ctx.ui_text_editing() {
            self.throttle = 0.0;
            self.brake = 0.0;
            self.steer = 0.0;
            self.handbrake = false;
        } else {
            self.read_input(ctx.input);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ActionValue, InputState};

    fn input(actions: Vec<(String, ActionValue)>) -> InputState {
        InputState::from_actions(actions, glam::Vec2::ZERO)
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
