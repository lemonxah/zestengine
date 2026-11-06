//! Client-side prediction, server reconciliation and adaptive interpolation.
//!
//! Predicted objects (Network Object > Predicted, on a Pawn, First / Third
//! Person Controller or VR Player: any `zestengine_core::Predictable`) are
//! server-authoritative but feel local:
//!
//! 1. The owner's controller moves at once from local input, as always. Each
//!    frame becomes an [`InputCmd`] (sequence number, frame delta, the active
//!    input actions and what the controller took from elsewhere: view
//!    angles, headset pose), kept with the state it produced in a
//!    [`Predictor`]. Unacked commands go to the host every frame,
//!    redundantly, so a lost packet costs nothing.
//! 2. The host queues them per object ([`InputQueue`]: in order, gaps
//!    skipped after a short wait, a time budget against sped-up clients) and
//!    simulates the controller with exactly the same step (its `replay`,
//!    through the character controller). It replies with an [`AuthState`]:
//!    the resulting state and the last input sequence it processed.
//! 3. The owner compares that with what it predicted for the same sequence.
//!    On a mismatch it rewinds to the authoritative state, replays the
//!    unacknowledged inputs and folds the difference into a visual error
//!    offset that decays over [`SMOOTH_TIME`], so corrections glide instead
//!    of snapping (teleport-sized ones still snap).
//!
//! Everything else keeps interpolating snapshots; [`JitterMeter`] and
//! [`AdaptiveDelay`] size each object's interpolation delay from the measured
//! arrival jitter and loss.

use std::collections::{BTreeMap, VecDeque};

use zestengine_core::{ActionValue, Climb, InputState, MotionState};
use glam::{Quat, Vec2, Vec3};

/// Time constant (s) of the visual error decay after a correction.
pub const SMOOTH_TIME: f32 = 0.1;
/// Corrections larger than this (m) snap instead of gliding (teleports, respawns).
pub const SNAP_DISTANCE: f32 = 2.0;
/// Position disagreement (m) below which a prediction counts as correct.
pub const POS_EPSILON: f32 = 1e-3;
/// Unacked inputs kept (and resent) at most.
const MAX_HISTORY: usize = 240;
/// Commands carried by one input packet at most (newest last).
pub const MAX_CMDS_PER_PACKET: usize = 12;
/// A missing input is skipped after this long (s) when later ones are waiting.
const GAP_WAIT: f64 = 0.1;
/// Seconds of simulation time the host lets a client bank (bursts after jitter).
const MAX_BUDGET: f32 = 0.5;
/// Queued input beyond this (s) is dropped from the front (a flooding client).
const MAX_QUEUED: f32 = 1.0;

/// One frame of a player's input.
#[derive(Clone, Default)]
pub struct InputCmd {
    pub seq: u32,
    /// The frame's scaled delta time (the pawn clamps it the same way everywhere).
    pub dt: f32,
    pub actions: Vec<(String, ActionValue)>,
    pub mouse: Vec2,
    /// What the controller took from outside the actions this frame (view
    /// angles, headset pose, a teleport); see `Predictable::frame_extra`.
    pub extra: Vec<f32>,
}

impl std::fmt::Debug for InputCmd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<&str> = self.actions.iter().map(|(k, _)| k.as_str()).collect();
        f.debug_struct("InputCmd").field("seq", &self.seq).field("dt", &self.dt).field("actions", &names).finish()
    }
}

fn action_eq(a: &ActionValue, b: &ActionValue) -> bool {
    a.down == b.down && a.just_pressed == b.just_pressed && a.just_released == b.just_released && a.value == b.value && a.axis2 == b.axis2
}

impl InputCmd {
    /// Capture the active actions of a frame.
    pub fn capture(seq: u32, dt: f32, input: &InputState, extra: Vec<f32>) -> Self {
        let mut actions: Vec<(String, ActionValue)> = input.active_actions().map(|(k, v)| (k.to_owned(), v)).collect();
        actions.sort_by(|a, b| a.0.cmp(&b.0));
        Self { seq, dt, actions, mouse: input.mouse_delta, extra }
    }

    /// The frame as an input snapshot for the simulation.
    pub fn input(&self) -> InputState {
        InputState::from_actions(self.actions.iter().cloned(), self.mouse)
    }

    fn same_actions(&self, o: &InputCmd) -> bool {
        self.actions.len() == o.actions.len()
            && self.actions.iter().zip(&o.actions).all(|(a, b)| a.0 == b.0 && action_eq(&a.1, &b.1))
    }
}

/// A predicted object's simulation state.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MoveState {
    /// World position.
    pub pos: Vec3,
    pub rot: Quat,
    pub motion: MotionState,
    /// The character controller touched the ground on its last move.
    pub contact: bool,
}

impl MoveState {
    /// Close enough that the prediction counts as correct.
    fn agrees(&self, o: &MoveState) -> bool {
        let (a, b) = (&self.motion, &o.motion);
        self.pos.distance(o.pos) < POS_EPSILON
            && a.vel.distance(b.vel) < 0.01
            && a.grounded == b.grounded
            && self.contact == o.contact
            && a.crouched == b.crouched
            && a.crouch_capsule == b.crouch_capsule
            && a.swimming == b.swimming
            && a.climb.is_some() == b.climb.is_some()
    }
}

/// The host's state for a predicted object after processing input `ack`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AuthState {
    pub ack: u32,
    pub state: MoveState,
}

/// What reconciliation needs from the simulation.
pub trait Replay {
    /// Put the object back into `state`.
    fn rewind(&mut self, state: &MoveState);
    /// Simulate one input frame from the current state; returns the result.
    fn step(&mut self, cmd: &InputCmd) -> MoveState;
}

/// Owner-side prediction history for one object.
#[derive(Default)]
pub struct Predictor {
    next_seq: u32,
    history: VecDeque<(InputCmd, MoveState)>,
    pub last_ack: Option<u32>,
    /// Visual offset (world) still to be smoothed away.
    pub error: Vec3,
    /// Reconciliations that changed the state, and the size of the last one (m).
    pub corrections: u32,
    pub last_correction: f32,
}

impl Predictor {
    /// Record the frame just simulated locally; returns its command.
    pub fn record(&mut self, dt: f32, input: &InputState, extra: Vec<f32>, after: MoveState) -> &InputCmd {
        self.next_seq = self.next_seq.wrapping_add(1).max(1);
        self.history.push_back((InputCmd::capture(self.next_seq, dt, input, extra), after));
        while self.history.len() > MAX_HISTORY {
            self.history.pop_front();
        }
        &self.history.back().expect("just pushed").0
    }

    /// Commands the host hasn't acknowledged, oldest first.
    pub fn unacked(&self) -> impl Iterator<Item = &InputCmd> {
        self.history.iter().map(|(c, _)| c)
    }

    /// The newest predicted state.
    pub fn current(&self) -> Option<MoveState> {
        self.history.back().map(|h| h.1)
    }

    /// The newest input frame and the state it produced (what a new host
    /// continues from after a migration).
    pub fn newest(&self) -> Option<AuthState> {
        self.history.back().map(|(c, s)| AuthState { ack: c.seq, state: *s })
    }

    /// Reconcile with the host's state. Returns the correction applied to
    /// the simulation (the visual position stays continuous through `error`).
    pub fn reconcile(&mut self, auth: &AuthState, sim: &mut impl Replay) -> Option<Vec3> {
        if self.last_ack.is_some_and(|a| auth.ack <= a) {
            return None;
        }
        self.last_ack = Some(auth.ack);
        let at_ack = self.history.iter().position(|(c, _)| c.seq == auth.ack);
        let predicted = at_ack.map(|i| self.history[i].1);
        let before = self.current();
        if let Some(i) = at_ack {
            self.history.drain(..=i);
        } else {
            self.history.retain(|(c, _)| c.seq > auth.ack);
        }
        if predicted.is_some_and(|p| p.agrees(&auth.state)) {
            return None;
        }
        // The view (yaw / pitch) comes from local input only: keep ours, so
        // a correction never turns the camera.
        let mut base = auth.state;
        if let Some(p) = predicted {
            base.motion.yaw = p.motion.yaw;
            base.motion.pitch = p.motion.pitch;
            base.rot = p.rot;
        }
        sim.rewind(&base);
        let mut after = base;
        for (cmd, state) in self.history.iter_mut() {
            *state = sim.step(cmd);
            after = *state;
        }
        let before = before.map_or(after.pos, |b| b.pos);
        let delta = before - after.pos;
        self.error += delta;
        if self.error.length() > SNAP_DISTANCE {
            self.error = Vec3::ZERO;
        }
        self.corrections += 1;
        self.last_correction = delta.length();
        Some(-delta)
    }

    /// Let the visual offset decay.
    pub fn decay(&mut self, dt: f32) {
        self.error *= (-dt.max(0.0) / SMOOTH_TIME).exp();
        if self.error.length() < 1e-4 {
            self.error = Vec3::ZERO;
        }
    }
}

/// Host-side input queue for one predicted object.
#[derive(Default)]
pub struct InputQueue {
    pending: BTreeMap<u32, InputCmd>,
    pub last_done: Option<u32>,
    budget: f32,
    gap_since: Option<f64>,
}

impl InputQueue {
    pub fn push(&mut self, cmd: InputCmd) {
        if self.last_done.is_some_and(|d| cmd.seq <= d) || self.pending.contains_key(&cmd.seq) {
            return;
        }
        self.pending.insert(cmd.seq, cmd);
        let mut queued: f32 = self.pending.values().map(|c| c.dt.min(0.1)).sum();
        while queued > MAX_QUEUED
            && let Some((_, c)) = self.pending.pop_first()
        {
            queued -= c.dt.min(0.1);
            self.last_done = Some(c.seq);
        }
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// Continue after input `ack` (a new host taking the owner's state):
    /// older frames are dropped, later ones kept.
    pub fn rebase(&mut self, ack: u32) {
        self.last_done = Some(ack);
        self.pending.retain(|seq, _| *seq > ack);
        self.gap_since = None;
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// The commands to simulate this host frame, in order: as many as the
    /// time budget allows, skipping a missing one once later ones waited
    /// [`GAP_WAIT`].
    pub fn take_ready(&mut self, frame_dt: f32, now: f64) -> Vec<InputCmd> {
        self.budget = (self.budget + frame_dt.max(0.0)).min(MAX_BUDGET);
        let mut out = Vec::new();
        while self.budget > 0.0 {
            let Some((&first, _)) = self.pending.first_key_value() else {
                break;
            };
            let expected = self.last_done.map_or(first, |d| d + 1);
            if first != expected {
                let since = *self.gap_since.get_or_insert(now);
                if now - since < GAP_WAIT {
                    break;
                }
                // Give up on the missing ones: the owner gets corrected.
                self.last_done = Some(first - 1);
            }
            self.gap_since = None;
            let cmd = self.pending.pop_first().expect("non-empty").1;
            self.budget -= cmd.dt.min(0.1);
            self.last_done = Some(cmd.seq);
            out.push(cmd);
        }
        out
    }
}

/// Arrival jitter of one sender's packets (RFC 3550 style estimator).
#[derive(Clone, Copy, Debug, Default)]
pub struct JitterMeter {
    last: Option<(f64, f64)>,
    /// Smoothed jitter (s).
    pub jitter: f32,
}

impl JitterMeter {
    /// A packet stamped `sent` (sender clock) arrived at `arrival` (our clock).
    pub fn on_packet(&mut self, sent: f64, arrival: f64) {
        if let Some((la, ls)) = self.last {
            if sent <= ls {
                return;
            }
            let d = ((arrival - la) - (sent - ls)).abs() as f32;
            self.jitter += (d - self.jitter) / 16.0;
        }
        self.last = Some((arrival, sent));
    }
}

/// Interpolation delay that follows the link: one send interval, more when
/// packets go missing, plus three times the jitter. Grows fast (render time
/// slows to 75 %) and shrinks slowly (105 %), so playback never jumps.
#[derive(Clone, Copy, Debug, Default)]
pub struct AdaptiveDelay {
    pub current: f32,
    init: bool,
}

/// Largest adaptive delay (s).
pub const MAX_DELAY: f32 = 0.5;

pub fn target_delay(send_interval: f32, jitter: f32, loss: f32, floor: f32) -> f32 {
    (send_interval * (1.0 + 2.0 * loss.clamp(0.0, 0.5)) + 3.0 * jitter + 0.005).clamp(floor, MAX_DELAY.max(floor))
}

impl AdaptiveDelay {
    pub fn update(&mut self, target: f32, dt: f32) -> f32 {
        if !self.init {
            self.init = true;
            self.current = target;
        } else if target > self.current {
            self.current = (self.current + dt * 0.25).min(target);
        } else {
            self.current = (self.current - dt * 0.05).max(target);
        }
        self.current
    }
}

// ------------------------------------------------------------------ wire

const F_DOWN: u8 = 1;
const F_PRESSED: u8 = 2;
const F_RELEASED: u8 = 4;
const F_VALUE: u8 = 8;
const F_AXIS: u8 = 16;
/// Per command: the actions equal the previous command's.
const C_SAME: u8 = 1;
const C_MOUSE: u8 = 2;
/// Extra input follows / equals the previous command's.
const C_EXTRA: u8 = 4;
const C_EXTRA_SAME: u8 = 8;
/// Extra input values per command at most.
const MAX_EXTRA: usize = 64;

/// Encode commands (consecutive sequence numbers, oldest first) after the
/// packet header, stopping before `budget` bytes. Returns how many fit.
pub fn encode_cmds(b: &mut Vec<u8>, cmds: &[&InputCmd], budget: usize) -> usize {
    let Some(first) = cmds.first() else {
        return 0;
    };
    b.extend_from_slice(&first.seq.to_le_bytes());
    let count_at = b.len();
    b.push(0);
    let mut n = 0usize;
    let mut prev: Option<&InputCmd> = None;
    for c in cmds.iter().take(MAX_CMDS_PER_PACKET) {
        if c.seq != first.seq.wrapping_add(n as u32) {
            break;
        }
        let mut one = Vec::with_capacity(32);
        let same = prev.is_some_and(|p| p.same_actions(c));
        let extra = &c.extra[..c.extra.len().min(MAX_EXTRA)];
        let extra_flag = match prev {
            _ if extra.is_empty() => 0,
            Some(p) if p.extra == extra => C_EXTRA_SAME,
            _ => C_EXTRA,
        };
        let flags = if same { C_SAME } else { 0 } | if c.mouse != Vec2::ZERO { C_MOUSE } else { 0 } | extra_flag;
        one.push(flags);
        one.extend_from_slice(&c.dt.to_le_bytes());
        if c.mouse != Vec2::ZERO {
            one.extend_from_slice(&c.mouse.x.to_le_bytes());
            one.extend_from_slice(&c.mouse.y.to_le_bytes());
        }
        if extra_flag == C_EXTRA {
            one.push(extra.len() as u8);
            for v in extra {
                one.extend_from_slice(&v.to_le_bytes());
            }
        }
        if !same {
            one.push(c.actions.len().min(255) as u8);
            for (name, v) in c.actions.iter().take(255) {
                let name = &name.as_bytes()[..name.len().min(255)];
                one.push(name.len() as u8);
                one.extend_from_slice(name);
                let mut f = 0u8;
                f |= if v.down { F_DOWN } else { 0 };
                f |= if v.just_pressed { F_PRESSED } else { 0 };
                f |= if v.just_released { F_RELEASED } else { 0 };
                f |= if v.value != 0.0 { F_VALUE } else { 0 };
                f |= if v.axis2 != Vec2::ZERO { F_AXIS } else { 0 };
                one.push(f);
                if v.value != 0.0 {
                    one.extend_from_slice(&v.value.to_le_bytes());
                }
                if v.axis2 != Vec2::ZERO {
                    one.extend_from_slice(&v.axis2.x.to_le_bytes());
                    one.extend_from_slice(&v.axis2.y.to_le_bytes());
                }
            }
        }
        if b.len() + one.len() > budget {
            break;
        }
        b.extend_from_slice(&one);
        prev = Some(c);
        n += 1;
    }
    b[count_at] = n as u8;
    n
}

/// Decode what [`encode_cmds`] wrote.
pub fn decode_cmds(d: &[u8]) -> Option<Vec<InputCmd>> {
    let mut r = Rd { d, p: 0 };
    let first = r.u32()?;
    let n = r.u8()? as usize;
    let mut out: Vec<InputCmd> = Vec::with_capacity(n);
    for k in 0..n {
        let flags = r.u8()?;
        let dt = r.f32()?;
        if !dt.is_finite() || dt < 0.0 {
            return None;
        }
        let mouse = if flags & C_MOUSE != 0 { Vec2::new(r.f32()?, r.f32()?) } else { Vec2::ZERO };
        if !mouse.is_finite() {
            return None;
        }
        let extra = if flags & C_EXTRA != 0 {
            let n = (r.u8()? as usize).min(MAX_EXTRA);
            let mut e = Vec::with_capacity(n);
            for _ in 0..n {
                let v = r.f32()?;
                if !v.is_finite() {
                    return None;
                }
                e.push(v);
            }
            e
        } else if flags & C_EXTRA_SAME != 0 {
            out.last()?.extra.clone()
        } else {
            Vec::new()
        };
        let actions = if flags & C_SAME != 0 {
            out.last()?.actions.clone()
        } else {
            let m = r.u8()? as usize;
            let mut a = Vec::with_capacity(m);
            for _ in 0..m {
                let len = r.u8()? as usize;
                let name = String::from_utf8_lossy(r.take(len)?).into_owned();
                let f = r.u8()?;
                let value = if f & F_VALUE != 0 { r.f32()? } else { 0.0 };
                let axis2 = if f & F_AXIS != 0 { Vec2::new(r.f32()?, r.f32()?) } else { Vec2::ZERO };
                if !value.is_finite() || !axis2.is_finite() {
                    return None;
                }
                a.push((
                    name,
                    ActionValue {
                        down: f & F_DOWN != 0,
                        just_pressed: f & F_PRESSED != 0,
                        just_released: f & F_RELEASED != 0,
                        value,
                        axis2: axis2.clamp_length_max(1.0e3),
                    },
                ));
            }
            a
        };
        out.push(InputCmd { seq: first.wrapping_add(k as u32), dt, actions, mouse, extra });
    }
    Some(out)
}

const S_GROUNDED: u8 = 1;
const S_CONTACT: u8 = 2;
const S_CROUCHED: u8 = 4;
const S_CROUCH_CAPSULE: u8 = 8;
const S_SWIMMING: u8 = 16;
const S_BREATH: u8 = 32;
const S_CLIMB: u8 = 64;

pub fn encode_auth(b: &mut Vec<u8>, a: &AuthState) {
    b.extend_from_slice(&a.ack.to_le_bytes());
    let (s, m) = (&a.state, &a.state.motion);
    let mut f = 0u8;
    for (on, bit) in [
        (m.grounded, S_GROUNDED),
        (s.contact, S_CONTACT),
        (m.crouched, S_CROUCHED),
        (m.crouch_capsule, S_CROUCH_CAPSULE),
        (m.swimming, S_SWIMMING),
        (m.breath.is_some(), S_BREATH),
        (m.climb.is_some(), S_CLIMB),
    ] {
        if on {
            f |= bit;
        }
    }
    b.push(f);
    let fixed = [
        s.pos.x, s.pos.y, s.pos.z, s.rot.x, s.rot.y, s.rot.z, s.rot.w, m.vel.x, m.vel.y, m.vel.z, m.yaw, m.pitch, m.body_yaw, m.floor_y,
    ];
    for v in fixed.into_iter().chain(m.breath) {
        b.extend_from_slice(&v.to_le_bytes());
    }
    if let Some(c) = m.climb {
        for v in [c.from.x, c.from.y, c.from.z, c.to.x, c.to.y, c.to.z, c.t] {
            b.extend_from_slice(&v.to_le_bytes());
        }
    }
}

pub fn decode_auth(d: &[u8]) -> Option<AuthState> {
    let mut r = Rd { d, p: 0 };
    let ack = r.u32()?;
    let flags = r.u8()?;
    let mut next = || r.f32().filter(|v| v.is_finite());
    let mut f = [0f32; 14];
    for v in &mut f {
        *v = next()?;
    }
    let breath = if flags & S_BREATH != 0 { Some(next()?) } else { None };
    let climb = if flags & S_CLIMB != 0 {
        let mut c = [0f32; 7];
        for v in &mut c {
            *v = next()?;
        }
        Some(Climb { from: Vec3::new(c[0], c[1], c[2]), to: Vec3::new(c[3], c[4], c[5]), t: c[6] })
    } else {
        None
    };
    let on = |bit: u8| flags & bit != 0;
    Some(AuthState {
        ack,
        state: MoveState {
            pos: Vec3::new(f[0], f[1], f[2]),
            rot: Quat::from_xyzw(f[3], f[4], f[5], f[6]).normalize(),
            motion: MotionState {
                vel: Vec3::new(f[7], f[8], f[9]),
                yaw: f[10],
                pitch: f[11],
                body_yaw: f[12],
                floor_y: f[13],
                grounded: on(S_GROUNDED),
                crouched: on(S_CROUCHED),
                crouch_capsule: on(S_CROUCH_CAPSULE),
                swimming: on(S_SWIMMING),
                breath,
                climb,
            },
            contact: on(S_CONTACT),
        },
    })
}

struct Rd<'a> {
    d: &'a [u8],
    p: usize,
}

impl Rd<'_> {
    fn take(&mut self, n: usize) -> Option<&[u8]> {
        let s = self.d.get(self.p..self.p + n)?;
        self.p += n;
        Some(s)
    }
    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn f32(&mut self) -> Option<f32> {
        Some(f32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 60.0;

    fn walk(forward: f32) -> InputState {
        let v = ActionValue { axis2: Vec2::new(0.0, forward), ..Default::default() };
        InputState::from_actions([("Move".to_string(), v)], Vec2::ZERO)
    }

    /// A toy character: accelerates toward the input, blocked by walls the
    /// simulation knows about.
    #[derive(Clone)]
    struct Toy {
        state: MoveState,
        wall: Option<f32>,
        /// Host-only push applied once (a server event the client can't predict).
        push: Option<(u32, Vec3)>,
    }

    impl Replay for Toy {
        fn rewind(&mut self, s: &MoveState) {
            self.state = *s;
        }
        fn step(&mut self, cmd: &InputCmd) -> MoveState {
            let input = cmd.input();
            let target = input.axis2("Move").y * 4.0;
            let m = &mut self.state.motion;
            m.vel.x += (target - m.vel.x) * (10.0 * cmd.dt).min(1.0);
            self.state.pos.x += m.vel.x * cmd.dt;
            if let Some(w) = self.wall
                && self.state.pos.x > w
            {
                self.state.pos.x = w;
                self.state.motion.vel.x = 0.0;
            }
            if let Some((seq, p)) = self.push
                && seq == cmd.seq
            {
                self.state.pos += p;
            }
            self.state
        }
    }

    /// A one-way link with latency (frames) and periodic loss.
    struct Link<T> {
        queue: VecDeque<(u32, T)>,
        delay: u32,
        drop_every: u32,
        sent: u32,
    }

    impl<T> Link<T> {
        fn new(delay: u32, drop_every: u32) -> Self {
            Self { queue: VecDeque::new(), delay, drop_every, sent: 0 }
        }
        fn send(&mut self, frame: u32, v: T) {
            self.sent += 1;
            if self.drop_every > 0 && self.sent.is_multiple_of(self.drop_every) {
                return;
            }
            self.queue.push_back((frame + self.delay, v));
        }
        fn recv(&mut self, frame: u32) -> Vec<T> {
            let mut out = Vec::new();
            while self.queue.front().is_some_and(|(at, _)| *at <= frame) {
                out.push(self.queue.pop_front().unwrap().1);
            }
            out
        }
    }

    #[test]
    fn prediction_converges_smoothly_under_latency_and_loss() {
        // The client predicts without knowing about a wall and a server-side
        // push; 100 ms each way, every 7th packet lost in both directions.
        let mut client = Toy { state: MoveState::default(), wall: None, push: None };
        let mut host = Toy { state: MoveState::default(), wall: Some(6.0), push: Some((40, Vec3::new(0.3, 0.0, 0.0))) };
        let mut pred = Predictor::default();
        let mut queue = InputQueue::default();
        let mut up: Link<Vec<u8>> = Link::new(6, 7);
        let mut down: Link<AuthState> = Link::new(6, 7);
        let mut prev_visual = 0.0f32;
        let mut max_jump = 0.0f32;
        for frame in 0..600u32 {
            let input = walk(if frame < 300 { 1.0 } else { 0.0 });
            // Client: predict, record, send every unacked command.
            let cmd = InputCmd::capture(0, DT, &input, Vec::new());
            let after = client.step(&InputCmd { seq: pred.next_seq + 1, ..cmd });
            pred.record(DT, &input, Vec::new(), after);
            let unacked: Vec<&InputCmd> = pred.unacked().collect();
            let skip = unacked.len().saturating_sub(MAX_CMDS_PER_PACKET);
            let mut packet = Vec::new();
            encode_cmds(&mut packet, &unacked[skip..], 1200);
            up.send(frame, packet);
            // Host: consume what arrived in order, reply with the state.
            for p in up.recv(frame) {
                for c in decode_cmds(&p).unwrap() {
                    queue.push(c);
                }
            }
            let ready = queue.take_ready(DT, frame as f64 * DT as f64);
            for c in &ready {
                host.step(c);
            }
            if let Some(ack) = queue.last_done {
                down.send(frame, AuthState { ack, state: host.state });
            }
            // Client: reconcile, smooth.
            for a in down.recv(frame) {
                pred.reconcile(&a, &mut client);
            }
            pred.decay(DT);
            let visual = client.state.pos.x + pred.error.x;
            let speed_step = 4.0 * DT;
            max_jump = max_jump.max(((visual - prev_visual).abs() - speed_step).max(0.0));
            prev_visual = visual;
            if frame == 1 {
                assert!(client.state.pos.x > 0.0, "the client moved before any reply");
            }
        }
        assert!(pred.corrections > 0, "the wall and the push forced corrections");
        assert!((client.state.pos.x - host.state.pos.x).abs() < POS_EPSILON, "converged: {} vs {}", client.state.pos.x, host.state.pos.x);
        assert!(pred.error.length() < 1e-3, "visual offset decayed");
        // Unsmoothed, the 0.3 m push alone would jump in one frame.
        assert!(max_jump < 0.1, "no visible snap: largest extra per-frame jump {max_jump} m");
    }

    #[test]
    fn input_codec_round_trips_and_dedupes_repeats() {
        let mut p = Predictor::default();
        let s = MoveState::default();
        let jump = ActionValue { down: true, just_pressed: true, value: 1.0, ..Default::default() };
        p.record(DT, &walk(1.0), vec![10.0, -5.0], s);
        p.record(DT, &walk(1.0), vec![10.0, -5.0], s);
        let with_jump = InputState::from_actions(
            [("Move".to_string(), ActionValue { axis2: Vec2::new(0.5, 1.0), ..Default::default() }), ("Jump".to_string(), jump)],
            Vec2::new(3.0, -1.0),
        );
        p.record(DT * 2.0, &with_jump, vec![12.0, -5.0, 1.5], s);
        let cmds: Vec<&InputCmd> = p.unacked().collect();
        let mut b = Vec::new();
        assert_eq!(encode_cmds(&mut b, &cmds, 1200), 3);
        let back = decode_cmds(&b).unwrap();
        assert_eq!(back.len(), 3);
        for (a, b) in cmds.iter().zip(&back) {
            assert_eq!((a.seq, a.dt, a.mouse, &a.extra), (b.seq, b.dt, b.mouse, &b.extra));
            assert!(a.same_actions(b));
        }
        let input = back[2].input();
        assert!(input.pressed("Jump") && input.axis2("Move") == Vec2::new(0.5, 1.0));
        // A repeated frame costs 5 bytes.
        let mut one = Vec::new();
        encode_cmds(&mut one, &cmds[..1], 1200);
        let mut two = Vec::new();
        encode_cmds(&mut two, &cmds[..2], 1200);
        assert_eq!(two.len() - one.len(), 5);
        assert!(decode_cmds(&b[..b.len() - 3]).is_none(), "truncated packets are rejected");

        let motion = MotionState {
            vel: Vec3::X,
            yaw: 17.0,
            pitch: -3.0,
            body_yaw: 40.0,
            grounded: true,
            floor_y: 0.5,
            crouched: true,
            crouch_capsule: true,
            swimming: true,
            breath: Some(12.5),
            climb: Some(Climb { from: Vec3::ONE, to: Vec3::new(0.0, 2.0, -1.0), t: 0.25 }),
        };
        let a = AuthState { ack: 9, state: MoveState { pos: Vec3::new(1.0, 2.0, 3.0), rot: Quat::from_rotation_y(0.3), motion, contact: true } };
        let mut ab = Vec::new();
        encode_auth(&mut ab, &a);
        let d = decode_auth(&ab).unwrap();
        assert_eq!(d.ack, 9);
        assert!(d.state.pos == a.state.pos && d.state.motion == a.state.motion && d.state.contact);
        assert!(decode_auth(&ab[..ab.len() - 1]).is_none(), "truncated states are rejected");
    }

    #[test]
    fn input_queue_orders_skips_gaps_and_limits_speed() {
        let cmd = |seq| InputCmd { seq, dt: DT, ..Default::default() };
        let mut q = InputQueue::default();
        for s in [2, 1, 3, 3] {
            q.push(cmd(s));
        }
        let got: Vec<u32> = q.take_ready(0.1, 0.0).iter().map(|c| c.seq).collect();
        assert_eq!(got, vec![1, 2, 3]);
        q.push(cmd(2));
        assert!(q.is_empty(), "already processed");
        // 4 is lost: 5 and 6 wait a moment, then 4 is skipped.
        q.push(cmd(5));
        q.push(cmd(6));
        assert!(q.take_ready(DT, 1.0).is_empty());
        let got: Vec<u32> = q.take_ready(DT, 1.2).iter().map(|c| c.seq).collect();
        assert_eq!(got, vec![5, 6]);
        // A client claiming 0.1 s frames at 60 Hz gets one frame of
        // simulation per host frame's worth of budget, not six.
        let mut q = InputQueue::default();
        let mut simulated = 0.0;
        for k in 1..=120u32 {
            q.push(InputCmd { seq: k, dt: 0.1, ..Default::default() });
            simulated += q.take_ready(DT, k as f64 * DT as f64).iter().map(|c| c.dt).sum::<f32>();
        }
        assert!(simulated < 2.0 + MAX_BUDGET + 0.11, "simulated {simulated} s in 2 s of host time");
    }

    #[test]
    fn interpolation_delay_adapts_to_jitter_and_loss() {
        let interval = 0.05;
        let floor = 0.1;
        let mut jm = JitterMeter::default();
        let mut d = AdaptiveDelay::default();
        let mut rng = 0x1234_5678u32;
        // Calm LAN: no jitter keeps the delay at the floor.
        for k in 0..200 {
            let sent = k as f64 * interval as f64;
            jm.on_packet(sent, sent + 0.02);
            d.update(target_delay(interval, jm.jitter, 0.0, floor), interval);
        }
        assert!((d.current - floor).abs() < 1e-4, "{}", d.current);
        // A jittery internet link (0..120 ms) with 10% loss: the delay grows.
        for k in 200..600 {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            let jitter = (rng % 120) as f64 / 1000.0;
            let sent = k as f64 * interval as f64;
            jm.on_packet(sent, sent + 0.05 + jitter);
            d.update(target_delay(interval, jm.jitter, 0.1, floor), interval);
        }
        let grown = d.current;
        assert!(grown > 0.15 && grown <= MAX_DELAY, "grew to {grown}");
        // It changes slowly: never more than 25% of elapsed time per update.
        let before = d.current;
        d.update(MAX_DELAY, interval);
        assert!(d.current - before <= interval * 0.25 + 1e-6);
        // The link calms down: back to the floor over time, gently.
        for k in 600..1400 {
            let sent = k as f64 * interval as f64;
            jm.on_packet(sent, sent + 0.05);
            let prev = d.current;
            d.update(target_delay(interval, jm.jitter, 0.0, floor), interval);
            assert!(prev - d.current <= interval * 0.05 + 1e-6, "shrinks gently");
        }
        assert!((d.current - floor).abs() < 0.01, "back near the floor: {}", d.current);
    }
}
