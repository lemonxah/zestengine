//! Parameter drivers: a state behaviour that changes Animator parameters when
//! a state is entered (Set, Add, Random, Copy), so a controller can pick an
//! outfit, count steps, roll a random idle or mirror one parameter into
//! another without code.
//!
//! The settings are stored on the state (`AnimState::behaviour_settings`, under
//! the behaviour's name) as RON text, so the controller file keeps one stable
//! shape however many behaviour types exist; the behaviour parses them when
//! they change, not every frame. The driver is registered under
//! [`PARAMETER_DRIVER`] without any setup call.

use serde::{Deserialize, Serialize};

use crate::avatar_params::{AvatarParamKind, BuiltinParam};
use crate::{StateBehaviour, StateBehaviourCtx};

/// The name states list in `behaviours` to run a driver.
pub const PARAMETER_DRIVER: &str = "Parameter Driver";

/// Maps `[src_min, src_max]` onto `[dst_min, dst_max]` (clamped).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[serde(default)]
pub struct Remap {
    /// Remap input minimum.
    pub src_min: f32,
    /// Remap input maximum.
    pub src_max: f32,
    /// Remap output minimum.
    pub dst_min: f32,
    /// Remap output maximum.
    pub dst_max: f32,
}

impl Default for Remap {
    fn default() -> Self {
        Self { src_min: 0.0, src_max: 1.0, dst_min: 0.0, dst_max: 1.0 }
    }
}

impl Remap {
    /// Map `v` from the source range onto the destination range (clamped).
    pub fn apply(&self, v: f32) -> f32 {
        let span = self.src_max - self.src_min;
        let t = if span.abs() < 1e-9 { 0.0 } else { ((v - self.src_min) / span).clamp(0.0, 1.0) };
        self.dst_min + (self.dst_max - self.dst_min) * t
    }
}

/// What one entry does to its parameter when the state is entered.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum DriverOp {
    /// name = value
    Set {
        /// The value written.
        value: f32,
    },
    /// name += value (on a Bool: any non-zero result is true)
    Add {
        /// The value added.
        value: f32,
    },
    /// Int: whole number in [min, max]. Float: [min, max). Bool: true with
    /// probability `chance`.
    Random {
        /// Random range minimum.
        min: f32,
        /// Random range maximum.
        max: f32,
        /// Probability of true for a Bool (0..1).
        chance: f32,
    },
    /// name = another parameter, optionally remapped.
    Copy {
        /// The source parameter's name.
        source: String,
        /// Optional range remap applied to the copied value.
        remap: Option<Remap>,
    },
}

impl Default for DriverOp {
    fn default() -> Self {
        DriverOp::Set { value: 1.0 }
    }
}

impl DriverOp {
    /// Inspector label of this op.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Set { .. } => "Set",
            Self::Add { .. } => "Add",
            Self::Random { .. } => "Random",
            Self::Copy { .. } => "Copy",
        }
    }
    /// A fresh op of the same type as `label` (the editor's type picker).
    pub fn default_for(label: &str) -> DriverOp {
        match label {
            "Add" => Self::Add { value: 1.0 },
            "Random" => Self::Random { min: 0.0, max: 1.0, chance: 0.5 },
            "Copy" => Self::Copy { source: String::new(), remap: None },
            _ => Self::Set { value: 1.0 },
        }
    }
    /// The op labels, in picker order.
    pub const LABELS: [&'static str; 4] = ["Set", "Add", "Random", "Copy"];
}

/// One parameter change of a driver.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct DriverEntry {
    /// Parameter written.
    pub name: String,
    /// What the parameter holds: Bool results are 0/1, Int results rounded,
    /// and it decides how Random rolls.
    pub kind: AvatarParamKind,
    /// What the op does.
    pub op: DriverOp,
}

impl Default for DriverEntry {
    fn default() -> Self {
        Self { name: String::new(), kind: AvatarParamKind::Float, op: DriverOp::default() }
    }
}

/// Everything one driver does when its state is entered.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
#[serde(default)]
pub struct ParamDriverSettings {
    /// The entries, run in order.
    pub entries: Vec<DriverEntry>,
    /// Run only on the wearer's own copy (the `IsLocal` parameter), so Add and
    /// Random cannot diverge between players.
    pub local_only: bool,
}

/// Small xorshift generator: drivers need dice, not cryptography.
#[derive(Clone, Debug)]
pub struct DriverRng(u64);

impl DriverRng {
    /// Build a generator with a fixed seed (tests, replays).
    pub fn seeded(seed: u64) -> Self {
        // One splitmix64 round, so nearby seeds start far apart.
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        Self(if z == 0 { 0x9E37_79B9_7F4A_7C15 } else { z })
    }
    /// Build a generator seeded from OS entropy.
    pub fn from_entropy() -> Self {
        let mut b = [0u8; 8];
        if getrandom::fill(&mut b).is_err() {
            b = 0x1234_5678_9ABC_DEF1u64.to_le_bytes();
        }
        Self::seeded(u64::from_le_bytes(b))
    }
    /// Uniform in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        ((x >> 40) as f32) / (1u64 << 24) as f32
    }
}

impl ParamDriverSettings {
    /// Serialize the settings to RON.
    pub fn to_ron(&self) -> String {
        ron::to_string(self).unwrap_or_default()
    }
    /// Empty text is an empty driver; broken text is an error.
    pub fn from_ron(text: &str) -> Result<Self, String> {
        if text.trim().is_empty() {
            return Ok(Self::default());
        }
        ron::from_str(text).map_err(|e| e.to_string())
    }

    /// Follow a parameter rename in the entries' names and Copy sources.
    pub fn rename_parameter(&mut self, old: &str, new: &str) {
        for e in &mut self.entries {
            if e.name == old {
                e.name = new.to_owned();
            }
            if let DriverOp::Copy { source, .. } = &mut e.op
                && source == old
            {
                *source = new.to_owned();
            }
        }
    }

    /// Apply every entry, in order, to `params`. `is_local` is false on
    /// other players' copies. Each entry sees the writes before it.
    pub fn run(&self, params: &mut std::collections::HashMap<String, f32>, is_local: bool, rng: &mut DriverRng) {
        if self.local_only && !is_local {
            return;
        }
        for e in &self.entries {
            if e.name.is_empty() {
                continue;
            }
            let old = params.get(&e.name).copied().unwrap_or(0.0);
            let new = match &e.op {
                DriverOp::Set { value } => *value,
                DriverOp::Add { value } => old + *value,
                DriverOp::Random { min, max, chance } => match e.kind {
                    AvatarParamKind::Bool => (rng.next_f32() < chance.clamp(0.0, 1.0)) as u8 as f32,
                    AvatarParamKind::Int => {
                        let (lo, hi) = (min.min(*max).round(), min.max(*max).round());
                        (lo + (rng.next_f32() * (hi - lo + 1.0)).floor()).min(hi)
                    }
                    AvatarParamKind::Float => min + (max - min) * rng.next_f32(),
                },
                DriverOp::Copy { source, remap } => {
                    let v = params.get(source).copied().unwrap_or(0.0);
                    remap.map_or(v, |r| r.apply(v))
                }
            };
            let new = e.kind.coerce_local(new);
            match params.get_mut(&e.name) {
                Some(x) => *x = new,
                None => {
                    params.insert(e.name.clone(), new);
                }
            }
        }
    }
}

/// The registered behaviour: runs its settings when the state is entered.
pub struct ParameterDriver {
    settings: ParamDriverSettings,
    /// The text `settings` came from, to re-parse only when it changes.
    parsed_from: String,
    rng: DriverRng,
}

impl ParameterDriver {
    /// A driver with default settings and an entropy-seeded generator.
    pub fn new() -> Self {
        Self { settings: ParamDriverSettings::default(), parsed_from: String::new(), rng: DriverRng::from_entropy() }
    }
    /// Same, with a fixed dice seed (tests).
    pub fn with_seed(seed: u64) -> Self {
        Self { rng: DriverRng::seeded(seed), ..Self::new() }
    }
}

impl Default for ParameterDriver {
    fn default() -> Self {
        Self::new()
    }
}

impl StateBehaviour for ParameterDriver {
    fn on_state_enter(&mut self, ctx: &mut StateBehaviourCtx) {
        if self.parsed_from != ctx.settings {
            self.settings = match ParamDriverSettings::from_ron(ctx.settings) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("parameter driver settings on state '{}': {e}", ctx.state);
                    ParamDriverSettings::default()
                }
            };
            self.parsed_from.clear();
            self.parsed_from.push_str(ctx.settings);
        }
        // An Animator with no avatar parameters has no IsLocal: it is local.
        let is_local = ctx.params.get(BuiltinParam::IsLocal.name()).is_none_or(|v| *v != 0.0);
        self.settings.run(ctx.params, is_local, &mut self.rng);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn entry(name: &str, kind: AvatarParamKind, op: DriverOp) -> DriverEntry {
        DriverEntry { name: name.into(), kind, op }
    }

    fn run(entries: Vec<DriverEntry>, params: &mut HashMap<String, f32>) {
        ParamDriverSettings { entries, local_only: false }.run(params, true, &mut DriverRng::seeded(7));
    }

    #[test]
    fn set_and_add() {
        let mut p = HashMap::new();
        run(vec![entry("Outfit", AvatarParamKind::Int, DriverOp::Set { value: 3.0 })], &mut p);
        assert_eq!(p["Outfit"], 3.0);
        run(vec![entry("Outfit", AvatarParamKind::Int, DriverOp::Add { value: 2.0 })], &mut p);
        assert_eq!(p["Outfit"], 5.0);
        run(vec![entry("Outfit", AvatarParamKind::Int, DriverOp::Add { value: -1.0 })], &mut p);
        assert_eq!(p["Outfit"], 4.0);
        // Adding to a parameter that was never set starts from 0.
        run(vec![entry("Fresh", AvatarParamKind::Float, DriverOp::Add { value: 0.25 })], &mut p);
        assert_eq!(p["Fresh"], 0.25);
        // A Bool takes any non-zero result as true.
        run(vec![entry("Flag", AvatarParamKind::Bool, DriverOp::Add { value: 5.0 })], &mut p);
        assert_eq!(p["Flag"], 1.0);
        run(vec![entry("Flag", AvatarParamKind::Bool, DriverOp::Set { value: 0.0 })], &mut p);
        assert_eq!(p["Flag"], 0.0);
        // Ints round.
        run(vec![entry("Outfit", AvatarParamKind::Int, DriverOp::Set { value: 2.6 })], &mut p);
        assert_eq!(p["Outfit"], 3.0);
    }

    #[test]
    fn entries_run_in_order_and_see_earlier_writes() {
        let mut p = HashMap::new();
        run(
            vec![
                entry("A", AvatarParamKind::Float, DriverOp::Set { value: 0.5 }),
                entry("B", AvatarParamKind::Float, DriverOp::Copy { source: "A".into(), remap: None }),
                entry("", AvatarParamKind::Float, DriverOp::Set { value: 9.0 }),
            ],
            &mut p,
        );
        assert_eq!(p["B"], 0.5);
        assert_eq!(p.len(), 2, "unnamed entries are skipped");
    }

    #[test]
    fn copy_with_and_without_remap() {
        let mut p = HashMap::from([("Src".to_string(), 0.5)]);
        run(vec![entry("Dst", AvatarParamKind::Float, DriverOp::Copy { source: "Src".into(), remap: None })], &mut p);
        assert_eq!(p["Dst"], 0.5);
        let r = Remap { src_min: 0.0, src_max: 1.0, dst_min: -1.0, dst_max: 1.0 };
        run(vec![entry("Dst", AvatarParamKind::Float, DriverOp::Copy { source: "Src".into(), remap: Some(r) })], &mut p);
        assert_eq!(p["Dst"], 0.0);
        // Clamped at both ends.
        p.insert("Src".into(), 4.0);
        run(vec![entry("Dst", AvatarParamKind::Float, DriverOp::Copy { source: "Src".into(), remap: Some(r) })], &mut p);
        assert_eq!(p["Dst"], 1.0);
        // Source and destination kinds may differ: float to int, float to bool.
        p.insert("Src".into(), 0.6);
        run(vec![entry("Level", AvatarParamKind::Int, DriverOp::Copy { source: "Src".into(), remap: Some(Remap { dst_max: 10.0, ..Default::default() }) })], &mut p);
        assert_eq!(p["Level"], 6.0);
        run(vec![entry("On", AvatarParamKind::Bool, DriverOp::Copy { source: "Src".into(), remap: None })], &mut p);
        assert_eq!(p["On"], 1.0);
        // A missing source reads as 0; a degenerate source range does not divide by zero.
        run(vec![entry("Z", AvatarParamKind::Float, DriverOp::Copy { source: "Nope".into(), remap: None })], &mut p);
        assert_eq!(p["Z"], 0.0);
        let flat = Remap { src_min: 1.0, src_max: 1.0, dst_min: 2.0, dst_max: 3.0 };
        assert_eq!(flat.apply(5.0), 2.0);
    }

    #[test]
    fn random_int_float_bool_stay_in_range_and_cover_it() {
        let mut rng = DriverRng::seeded(42);
        let mut seen = [false; 4];
        for _ in 0..400 {
            let mut p = HashMap::new();
            ParamDriverSettings { entries: vec![entry("I", AvatarParamKind::Int, DriverOp::Random { min: 1.0, max: 4.0, chance: 0.5 })], local_only: false }
                .run(&mut p, true, &mut rng);
            let v = p["I"];
            assert!((1.0..=4.0).contains(&v) && v.fract() == 0.0, "{v}");
            seen[v as usize - 1] = true;
        }
        assert!(seen.iter().all(|s| *s), "inclusive upper bound is reachable: {seen:?}");

        let mut p = HashMap::new();
        for _ in 0..400 {
            ParamDriverSettings { entries: vec![entry("F", AvatarParamKind::Float, DriverOp::Random { min: -0.5, max: 0.5, chance: 0.0 })], local_only: false }
                .run(&mut p, true, &mut rng);
            assert!((-0.5..0.5).contains(&p["F"]), "{}", p["F"]);
        }

        let mut yes = 0;
        for _ in 0..1000 {
            ParamDriverSettings { entries: vec![entry("B", AvatarParamKind::Bool, DriverOp::Random { min: 0.0, max: 1.0, chance: 0.25 })], local_only: false }
                .run(&mut p, true, &mut rng);
            yes += p["B"] as i32;
        }
        assert!((150..350).contains(&yes), "chance 0.25 gave {yes}/1000");
        for (chance, want) in [(0.0, 0.0), (1.0, 1.0)] {
            ParamDriverSettings { entries: vec![entry("B", AvatarParamKind::Bool, DriverOp::Random { min: 0.0, max: 1.0, chance })], local_only: false }
                .run(&mut p, true, &mut rng);
            assert_eq!(p["B"], want);
        }
        // Swapped bounds still roll inside them.
        ParamDriverSettings { entries: vec![entry("S", AvatarParamKind::Int, DriverOp::Random { min: 5.0, max: 5.0, chance: 0.5 })], local_only: false }
            .run(&mut p, true, &mut rng);
        assert_eq!(p["S"], 5.0);
    }

    #[test]
    fn same_seed_same_rolls() {
        let s = ParamDriverSettings { entries: vec![entry("I", AvatarParamKind::Int, DriverOp::Random { min: 0.0, max: 100.0, chance: 0.5 })], local_only: false };
        let roll = |seed| {
            let mut p = HashMap::new();
            s.run(&mut p, true, &mut DriverRng::seeded(seed));
            p["I"]
        };
        assert_eq!(roll(3), roll(3));
        assert!((0..20).map(|i| roll(i + 1) as i32).collect::<std::collections::HashSet<_>>().len() > 3);
    }

    #[test]
    fn local_only_skips_remote_copies() {
        let s = ParamDriverSettings { entries: vec![entry("A", AvatarParamKind::Float, DriverOp::Add { value: 1.0 })], local_only: true };
        let mut p = HashMap::new();
        s.run(&mut p, false, &mut DriverRng::seeded(1));
        assert!(p.is_empty(), "a remote copy does not run a local-only driver");
        s.run(&mut p, true, &mut DriverRng::seeded(1));
        assert_eq!(p["A"], 1.0);
        let all = ParamDriverSettings { local_only: false, ..s };
        let mut q = HashMap::new();
        all.run(&mut q, false, &mut DriverRng::seeded(1));
        assert_eq!(q["A"], 1.0, "without the flag it runs everywhere");
    }

    #[test]
    fn settings_round_trip_and_tolerate_missing_fields() {
        let s = ParamDriverSettings {
            entries: vec![
                entry("A", AvatarParamKind::Int, DriverOp::Random { min: 0.0, max: 3.0, chance: 0.5 }),
                entry("B", AvatarParamKind::Float, DriverOp::Copy { source: "A".into(), remap: Some(Remap::default()) }),
            ],
            local_only: true,
        };
        assert_eq!(ParamDriverSettings::from_ron(&s.to_ron()).unwrap(), s);
        assert_eq!(ParamDriverSettings::from_ron("").unwrap(), ParamDriverSettings::default());
        assert_eq!(ParamDriverSettings::from_ron("()").unwrap(), ParamDriverSettings::default());
        let e = ParamDriverSettings::from_ron("(entries:[(name:\"X\")])").unwrap();
        assert_eq!(e.entries[0].op, DriverOp::Set { value: 1.0 });
        assert!(ParamDriverSettings::from_ron("(((").is_err());
    }

    #[test]
    fn behaviour_runs_on_enter_reads_is_local_and_reparses_on_change() {
        let settings = ParamDriverSettings {
            entries: vec![entry("N", AvatarParamKind::Int, DriverOp::Add { value: 1.0 })],
            local_only: true,
        }
        .to_ron();
        let mut b = ParameterDriver::with_seed(1);
        let mut params: HashMap<String, f32> = HashMap::new();
        let (mut delta, mut goals) = (glam::Vec3::ZERO, Vec::new());
        let mut enter = |b: &mut ParameterDriver, params: &mut HashMap<String, f32>, text: &str| {
            let mut ctx = StateBehaviourCtx {
                layer: 0,
                state: "S",
                normalized_time: 0.0,
                layer_weight: 1.0,
                params,
                root_delta: &mut delta,
                ik_goals: &mut goals,
                settings: text,
            };
            b.on_state_enter(&mut ctx);
        };
        enter(&mut b, &mut params, &settings);
        enter(&mut b, &mut params, &settings);
        assert_eq!(params["N"], 2.0, "no IsLocal parameter reads as local");
        params.insert("IsLocal".into(), 0.0);
        enter(&mut b, &mut params, &settings);
        assert_eq!(params["N"], 2.0, "remote copy skipped");
        params.insert("IsLocal".into(), 1.0);
        let changed = ParamDriverSettings { entries: vec![entry("N", AvatarParamKind::Int, DriverOp::Set { value: 9.0 })], local_only: false }.to_ron();
        enter(&mut b, &mut params, &changed);
        assert_eq!(params["N"], 9.0);
        // Broken settings do nothing rather than crash.
        enter(&mut b, &mut params, "(((");
        assert_eq!(params["N"], 9.0);
    }

    #[test]
    fn registered_without_setup() {
        assert!(crate::create_state_behaviour(PARAMETER_DRIVER).is_some());
        assert!(crate::create_state_behaviour("definitely not registered").is_none());
    }
}
