use serde::{Deserialize, Serialize};

use crate::actions::{ActionCtx, ActionDesc, ActionType, ArgValue, ParamDesc, PropertyDesc};
use crate::{ObjectId, TypedComponent};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SequenceLoop {
    #[default]
    FromSequence,
    Once,
    Loop,
}

impl SequenceLoop {
    pub const ALL: [SequenceLoop; 3] = [Self::FromSequence, Self::Once, Self::Loop];

    pub fn label(self) -> &'static str {
        match self {
            Self::FromSequence => "As authored",
            Self::Once => "Once",
            Self::Loop => "Loop",
        }
    }

    pub fn resolve(self, authored: bool) -> bool {
        match self {
            Self::FromSequence => authored,
            Self::Once => false,
            Self::Loop => true,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SequenceFinished {
    pub object: ObjectId,
    pub sequence: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SequencePlayer {
    pub sequence: String,
    pub play_on_start: bool,
    pub looping: SequenceLoop,
    pub speed: f32,
    pub use_camera: bool,
    #[serde(skip)]
    pub time: f32,
    #[serde(skip)]
    pub playing: bool,
    #[serde(skip)]
    pub length: f32,
    #[serde(skip)]
    pub started: bool,
    #[serde(skip)]
    pub applied: Option<f32>,
    #[serde(skip)]
    pub camera_before: Option<Option<ObjectId>>,
    #[serde(skip)]
    pub camera_now: Option<ObjectId>,
    #[serde(skip)]
    pub release_camera: bool,
}

impl Default for SequencePlayer {
    fn default() -> Self {
        Self {
            sequence: String::new(),
            play_on_start: true,
            looping: SequenceLoop::FromSequence,
            speed: 1.0,
            use_camera: true,
            time: 0.0,
            playing: false,
            length: 0.0,
            started: false,
            applied: None,
            camera_before: None,
            camera_now: None,
            release_camera: false,
        }
    }
}

impl SequencePlayer {
    pub fn new(sequence: impl Into<String>) -> Self {
        Self { sequence: sequence.into(), ..Self::default() }
    }

    pub fn play(&mut self) {
        if !self.playing && self.length > 0.0 && self.time >= self.length && self.speed >= 0.0 {
            self.time = 0.0;
        }
        self.playing = true;
        self.started = true;
        self.release_camera = false;
    }

    pub fn pause(&mut self) {
        self.playing = false;
    }

    pub fn stop(&mut self) {
        self.playing = false;
        self.time = 0.0;
        self.release_camera = true;
    }

    pub fn restart(&mut self) {
        self.time = 0.0;
        self.applied = None;
        self.play();
    }

    pub fn seek(&mut self, time: f32) {
        let end = if self.length > 0.0 { self.length } else { f32::MAX };
        self.time = if time.is_finite() { time.clamp(0.0, end) } else { 0.0 };
    }

    pub fn progress(&self) -> f32 {
        if self.length > 0.0 { (self.time / self.length).clamp(0.0, 1.0) } else { 0.0 }
    }

    pub fn advance(&mut self, dt: f32, length: f32, authored_loop: bool) -> SequenceStep {
        self.length = length.max(0.0);
        let from = self.time;
        if !self.playing || self.length <= 0.0 {
            return SequenceStep { from, to: from, wrapped: false, finished: false };
        }
        let speed = if self.speed.is_finite() { self.speed } else { 1.0 };
        let next = from + dt.max(0.0) * speed;
        let looping = self.looping.resolve(authored_loop);
        let (to, wrapped, finished) = if next > self.length {
            if looping { (next.rem_euclid(self.length), true, false) } else { (self.length, false, true) }
        } else if next < 0.0 {
            if looping { (next.rem_euclid(self.length), true, false) } else { (0.0, false, true) }
        } else {
            (next, false, false)
        };
        self.time = to;
        if finished {
            self.playing = false;
            self.release_camera = true;
        }
        SequenceStep { from, to, wrapped, finished }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SequenceStep {
    pub from: f32,
    pub to: f32,
    pub wrapped: bool,
    pub finished: bool,
}

impl SequenceStep {
    pub fn moved_forward(&self) -> bool {
        self.wrapped || self.to > self.from
    }
}

static ACTIONS: &[ActionDesc] = &[
    ActionDesc::new("Play", "Play", &[], true),
    ActionDesc::new("Pause", "Pause", &[], true),
    ActionDesc::new("Stop", "Stop", &[], true),
    ActionDesc::new("Restart", "Restart", &[], true),
    ActionDesc::new("Seek", "Seek", &[ParamDesc::float("time", 0.0)], true),
    ActionDesc::new("SetSpeed", "Set Speed", &[ParamDesc::float("speed", 1.0)], true),
];

static PROPERTIES: &[PropertyDesc] = &[
    PropertyDesc::new("time", ActionType::Float),
    PropertyDesc::new("length", ActionType::Float),
    PropertyDesc::new("progress", ActionType::Float),
    PropertyDesc::new("playing", ActionType::Bool),
];

impl TypedComponent for SequencePlayer {
    const NAME: &'static str = "Sequence Player";

    fn actions(&self) -> &'static [ActionDesc] {
        ACTIONS
    }

    fn invoke_action(&mut self, _ctx: &mut ActionCtx, name: &str, args: &[ArgValue]) -> Result<(), String> {
        match name {
            "Play" => self.play(),
            "Pause" => self.pause(),
            "Stop" => self.stop(),
            "Restart" => self.restart(),
            "Seek" => self.seek(args.first().map_or(0.0, ArgValue::as_f32)),
            "SetSpeed" => self.speed = args.first().map_or(1.0, ArgValue::as_f32),
            _ => return Err(crate::actions::unknown_action(Self::NAME, name)),
        }
        Ok(())
    }

    fn properties(&self) -> &'static [PropertyDesc] {
        PROPERTIES
    }

    fn read_property(&self, name: &str) -> Option<ArgValue> {
        match name {
            "time" => Some(ArgValue::Float(self.time)),
            "length" => Some(ArgValue::Float(self.length)),
            "progress" => Some(ArgValue::Float(self.progress())),
            "playing" => Some(ArgValue::Bool(self.playing)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn act(player: &mut SequencePlayer, name: &str, args: &[ArgValue]) {
        let mut commands = Vec::new();
        let mut ctx = ActionCtx { self_id: ObjectId::new(), instigator: None, commands: &mut commands, reader: None };
        TypedComponent::invoke_action(player, &mut ctx, name, args).unwrap();
    }

    #[test]
    fn a_one_shot_plays_to_the_end_and_stops_there() {
        let mut p = SequencePlayer::new("cut.seq");
        p.play();
        let step = p.advance(1.5, 4.0, false);
        assert_eq!((step.from, step.to, step.finished), (0.0, 1.5, false));
        let step = p.advance(5.0, 4.0, false);
        assert_eq!((step.to, step.finished, step.wrapped), (4.0, true, false));
        assert!(!p.playing && p.release_camera);
        assert_eq!(p.progress(), 1.0);
        let idle = p.advance(1.0, 4.0, false);
        assert_eq!((idle.from, idle.to, idle.finished), (4.0, 4.0, false));
        p.play();
        assert_eq!(p.time, 0.0, "playing a finished sequence starts over");
        assert!(!p.release_camera);
    }

    #[test]
    fn looping_wraps_and_the_component_setting_overrides_the_sequence() {
        let mut p = SequencePlayer::new("loop.seq");
        p.play();
        let step = p.advance(5.0, 4.0, true);
        assert!(step.wrapped && !step.finished);
        assert!((step.to - 1.0).abs() < 1e-5);
        assert!(p.playing);
        p.looping = SequenceLoop::Once;
        assert!(p.advance(5.0, 4.0, true).finished);
        let mut forced = SequencePlayer { looping: SequenceLoop::Loop, ..SequencePlayer::new("x.seq") };
        forced.play();
        assert!(forced.advance(9.0, 4.0, false).wrapped);
    }

    #[test]
    fn reverse_playback_ends_at_the_start() {
        let mut p = SequencePlayer { speed: -1.0, ..SequencePlayer::new("back.seq") };
        p.length = 4.0;
        p.seek(1.0);
        p.play();
        let step = p.advance(3.0, 4.0, false);
        assert_eq!((step.to, step.finished), (0.0, true));
        assert!(!step.moved_forward());
    }

    #[test]
    fn actions_and_properties_drive_the_player() {
        let mut p = SequencePlayer::new("door.seq");
        p.length = 10.0;
        act(&mut p, "Play", &[]);
        assert_eq!(TypedComponent::read_property(&p, "playing"), Some(ArgValue::Bool(true)));
        p.advance(2.0, 10.0, false);
        act(&mut p, "Pause", &[]);
        assert_eq!(p.advance(2.0, 10.0, false).to, 2.0);
        act(&mut p, "Seek", &[ArgValue::Float(99.0)]);
        assert_eq!(TypedComponent::read_property(&p, "time"), Some(ArgValue::Float(10.0)));
        assert_eq!(TypedComponent::read_property(&p, "progress"), Some(ArgValue::Float(1.0)));
        act(&mut p, "SetSpeed", &[ArgValue::Float(2.0)]);
        act(&mut p, "Restart", &[]);
        assert_eq!(p.advance(1.0, 10.0, false).to, 2.0);
        act(&mut p, "Stop", &[]);
        assert_eq!((p.time, p.playing, p.release_camera), (0.0, false, true));
        let mut commands = Vec::new();
        let mut ctx = ActionCtx { self_id: ObjectId::new(), instigator: None, commands: &mut commands, reader: None };
        assert!(TypedComponent::invoke_action(&mut p, &mut ctx, "Explode", &[]).is_err());
    }

    #[test]
    fn only_authored_settings_are_saved() {
        let mut p = SequencePlayer { speed: 0.5, looping: SequenceLoop::Loop, ..SequencePlayer::new("intro.seq") };
        p.play();
        p.advance(1.0, 3.0, false);
        let text = ron::to_string(&p).unwrap();
        assert!(!text.contains("time") && !text.contains("playing"));
        let back: SequencePlayer = ron::from_str(&text).unwrap();
        assert_eq!((back.sequence.as_str(), back.speed, back.looping, back.time, back.playing), ("intro.seq", 0.5, SequenceLoop::Loop, 0.0, false));
        let old: SequencePlayer = ron::from_str("(sequence: \"a.seq\")").unwrap();
        assert!(old.play_on_start && old.use_camera && old.speed == 1.0);
    }
}
