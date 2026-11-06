//! Structural humanoid auto-mapper.
//!
//! Maps a skeleton's joints to [`HumanBone`]s from hierarchy shape first and
//! names second. The search is driven by a goal table ([`Goal`]): for every
//! human bone it names the goal's parent bone, a window of hierarchy steps
//! below that parent, a direction prior and a length-ratio prior (both in the
//! character frame), an optional/side flag and keyword lists. A candidate joint
//! scores:
//!
//! * the sum of its best child goals (so a hips candidate is judged by how
//!   well a whole body maps beneath it),
//! * +20 for a keyword in its name, -20 when an ancestor inside the window
//!   already carries one (the keyword bone beats its child), -1000 for a
//!   rejected keyword,
//! * a direction term and +10 when the direction is right,
//! * a side term from `left`/`right` or an isolated `L`/`R` token
//!   (`_L`, `.R`, `L_`, ...): own side +15, other side or any side mark on a
//!   centre bone -1000,
//! * a length-ratio term against the parent bone's own length.
//!
//! Each goal keeps its top three candidates; sibling goals that picked joints
//! on the same path are resolved by trying every combination and keeping the
//! best total. The body pass runs in world axes and, when the legs and arms
//! show the rig is rotated (Z-up, facing away) by more than 20 degrees, again in
//! the detected character frame. Fingers then map in a per-hand pass in the
//! frame of the forearm; a hand that gains fewer than three finger bones loses
//! them all. 3ds Max Biped rigs map by their fixed name table instead.
//!
//! The canonical frame faces +Z with the avatar's left on +X and up on +Y.

use crate::human_rig::{ALL, HumanBone, REQUIRED_BONE_COUNT};
use crate::Skeleton;
use glam::{Mat3, Vec3};
use std::collections::HashMap;
use std::rc::Rc;

use HumanBone as B;

/// Which side of the body a goal belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Center,
    Left,
    Right,
}

/// One row of the goal table: where a human bone is searched for.
#[derive(Clone, Copy, Debug)]
pub struct Goal {
    pub bone: HumanBone,
    /// The goal this one hangs under (`None` = a root of the table).
    pub parent: Option<HumanBone>,
    /// Real-bone levels below the parent joint; `min_step 0` lets the bone
    /// coincide with its parent, which is how optional bones are skipped.
    pub min_step: u8,
    pub max_step: u8,
    /// Expected `|bone - parent| / |parent - grandparent|`; 0 = no prior.
    pub len_ratio: f32,
    /// Expected direction parent to bone in the character frame; zero = no prior.
    pub dir: [f32; 3],
    pub side: Side,
    pub optional: bool,
    pub always_include: bool,
    /// Needs a keyword hit; without one the candidate is rejected.
    pub strict: bool,
    /// Only helps score its parent candidate; never mapped.
    pub pseudo: bool,
    pub keywords: &'static [&'static str],
    pub rejects: &'static [&'static str],
}

impl Goal {
    const fn new(bone: HumanBone, parent: Option<HumanBone>, min_step: u8, max_step: u8, side: Side) -> Self {
        Self {
            bone,
            parent,
            min_step,
            max_step,
            len_ratio: 0.0,
            dir: [0.0; 3],
            side,
            optional: false,
            always_include: false,
            strict: false,
            pseudo: false,
            keywords: &[],
            rejects: &[],
        }
    }
    const fn len(mut self, r: f32) -> Self {
        self.len_ratio = r;
        self
    }
    const fn dir(mut self, x: f32, y: f32, z: f32) -> Self {
        self.dir = [x, y, z];
        self
    }
    const fn opt(mut self) -> Self {
        self.optional = true;
        self
    }
    const fn always(mut self) -> Self {
        self.always_include = true;
        self
    }
    const fn strict(mut self) -> Self {
        self.strict = true;
        self
    }
    const fn pseudo(mut self) -> Self {
        self.pseudo = true;
        self
    }
    const fn kw(mut self, k: &'static [&'static str]) -> Self {
        self.keywords = k;
        self
    }
    const fn rej(mut self, k: &'static [&'static str]) -> Self {
        self.rejects = k;
        self
    }
}

const END: &[&str] = &["end", "top", "nub"];
const HEAD_REJ: &[&str] = &["teeth", "tongue", "pony", "braid", "end", "top", "nub"];
const EYE_REJ: &[&str] = &["brow", "lid", "lash", "pony", "braid", "end", "top", "nub"];
const FINGER_REJ: &[&str] = &["palm", "wrist", "meta", "end", "top", "nub"];

/// Centre goals (spine, head, face).
const CENTER: &[Goal] = &[
    Goal::new(B::Hips, None, 1, 4, Side::Center).kw(&["hip", "pelvis"]),
    Goal::new(B::Spine, Some(B::Hips), 1, 3, Side::Center).dir(0.0, 1.0, 0.0),
    Goal::new(B::Chest, Some(B::Spine), 0, 3, Side::Center).len(1.4).dir(0.0, 1.0, 0.0).opt(),
    Goal::new(B::UpperChest, Some(B::Chest), 0, 3, Side::Center).len(1.4).dir(0.0, 1.0, 0.0).opt(),
    Goal::new(B::Neck, Some(B::UpperChest), 0, 3, Side::Center).len(1.8).dir(0.0, 1.0, 0.0).opt().kw(&["neck"]),
    Goal::new(B::Head, Some(B::Neck), 1, 3, Side::Center).len(0.3).dir(0.0, 1.0, 0.0).kw(&["head"]).rej(END),
    Goal::new(B::Jaw, Some(B::Head), 1, 2, Side::Center).dir(0.0, -0.5, 1.0).opt().strict().kw(&["jaw", "open"]).rej(HEAD_REJ),
];

/// Left-side goals; the right side is this table mirrored (`dir.x` negated,
/// `Left` bones swapped for `Right`).
const LEFT: &[Goal] = &[
    Goal::new(B::LeftUpperLeg, Some(B::Hips), 1, 2, Side::Left).dir(1.0, -0.4, 0.0).kw(&["up", "thigh"]),
    Goal::new(B::LeftLowerLeg, Some(B::LeftUpperLeg), 1, 2, Side::Left).len(3.0).dir(0.0, -1.0, 0.0).kw(&["low", "calf", "knee", "shin"]),
    Goal::new(B::LeftFoot, Some(B::LeftLowerLeg), 1, 2, Side::Left).len(1.0).dir(0.0, -1.0, 0.0).kw(&["foot", "ankle"]),
    Goal::new(B::LeftToes, Some(B::LeftFoot), 1, 2, Side::Left).len(0.5).dir(0.0, 0.0, 1.0).opt().always().kw(&["toe", "ball"]).rej(END),
    Goal::new(B::LeftShoulder, Some(B::UpperChest), 0, 3, Side::Left).dir(1.0, 0.0, 0.0).opt().kw(&["shoulder", "collar", "clavicle"]),
    Goal::new(B::LeftUpperArm, Some(B::LeftShoulder), 1, 3, Side::Left).len(0.5).dir(1.0, 0.0, 0.0).kw(&["up"]),
    Goal::new(B::LeftLowerArm, Some(B::LeftUpperArm), 1, 2, Side::Left).len(2.0).dir(1.0, 0.0, 0.0).kw(&["low", "fore", "elbow"]),
    Goal::new(B::LeftHand, Some(B::LeftLowerArm), 1, 2, Side::Left).len(1.0).dir(1.0, 0.0, 0.0).kw(&["hand", "wrist"]),
    Goal::new(B::LeftEye, Some(B::Head), 1, 2, Side::Left).dir(1.0, 1.0, 1.0).opt().strict().kw(&["eye"]).rej(EYE_REJ),
    // Scoring-only: a hand with a thumb and an index finger below it is likelier the real hand.
    Goal::new(B::LeftThumbProximal, Some(B::LeftHand), 1, 3, Side::Left).opt().strict().pseudo().kw(&["thu"]).rej(FINGER_REJ),
    Goal::new(B::LeftIndexProximal, Some(B::LeftHand), 1, 3, Side::Left).opt().strict().pseudo().kw(&["ind", "point"]).rej(FINGER_REJ),
];

/// Finger goals for the left hand: `(finger, proximal direction prior in the hand frame, keywords)`.
const FINGERS: [(&str, [f32; 3], &[&str]); 5] = [
    ("Thumb", [2.0, 0.0, 1.0], &["thu"]),
    ("Index", [4.0, 0.0, 1.0], &["ind", "point"]),
    ("Middle", [4.0, 0.0, 0.0], &["mid", "long"]),
    ("Ring", [4.0, 0.0, -1.0], &["rin"]),
    ("Little", [4.0, 0.0, -2.0], &["lit", "pin"]),
];

/// The Right bone for a Left bone (`LeftHand` -> `RightHand`); any other bone maps to itself.
pub fn mirror_bone(b: HumanBone) -> HumanBone {
    let n = b.name();
    match n.strip_prefix("Left") {
        Some(rest) => HumanBone::from_name(&format!("Right{rest}")).unwrap_or(b),
        None => b,
    }
}

fn mirror_goal(g: &Goal) -> Goal {
    let mut m = *g;
    m.bone = mirror_bone(g.bone);
    m.parent = g.parent.map(mirror_bone);
    m.dir[0] = -g.dir[0];
    m.side = match g.side {
        Side::Left => Side::Right,
        Side::Right => Side::Left,
        Side::Center => Side::Center,
    };
    m
}

/// The full-body goal table (centre goals plus both sides).
pub fn body_goals() -> Vec<Goal> {
    let mut out: Vec<Goal> = CENTER.to_vec();
    for g in LEFT {
        out.push(*g);
        out.push(mirror_goal(g));
    }
    out
}

fn finger_bone(left: bool, finger: usize, joint: usize) -> HumanBone {
    // Left fingers start at index 24, right at 39; each finger is three bones.
    let base = if left { 24 } else { 39 };
    ALL[base + finger * 3 + joint]
}

/// The finger goals of one hand (15 rows: proximal 1-3 steps below the hand,
/// intermediate and distal exactly one step each).
pub fn finger_goals(left: bool) -> Vec<Goal> {
    let hand = if left { B::LeftHand } else { B::RightHand };
    let side = if left { Side::Left } else { Side::Right };
    let mut out = Vec::new();
    for (f, (_, dir, kw)) in FINGERS.iter().enumerate() {
        let (p, i, d) = (finger_bone(left, f, 0), finger_bone(left, f, 1), finger_bone(left, f, 2));
        out.push(Goal::new(p, Some(hand), 1, 3, side).dir(dir[0], dir[1], dir[2]).opt().kw(kw).rej(FINGER_REJ));
        out.push(Goal::new(i, Some(p), 1, 1, side).opt().always().rej(FINGER_REJ));
        out.push(Goal::new(d, Some(i), 1, 1, side).opt().always().rej(FINGER_REJ));
    }
    out
}

/// Lower-case, namespace-stripped, camelCase-split name (`mixamorig:LeftUpLeg` -> `left up leg`).
pub fn nice_name(name: &str) -> String {
    let stripped = name.rsplit(':').next().unwrap_or(name);
    let chars: Vec<char> = stripped.chars().collect();
    let mut out = String::with_capacity(stripped.len() + 4);
    for (i, &c) in chars.iter().enumerate() {
        if matches!(c, '_' | '.' | '-' | ' ') {
            out.push(' ');
            continue;
        }
        if i > 0 && c.is_uppercase() {
            let p = chars[i - 1];
            let next_lower = chars.get(i + 1).is_some_and(|n| n.is_lowercase());
            if p.is_lowercase() || p.is_ascii_digit() || (p.is_uppercase() && next_lower) {
                out.push(' ');
            }
        }
        out.extend(c.to_lowercase());
    }
    out
}

/// The side a name marks: `left`/`right` anywhere, or an isolated `l`/`r` token.
pub fn side_mark(nice: &str) -> Option<Side> {
    let left = nice.contains("left");
    let right = nice.contains("right");
    match (left, right) {
        (true, false) => return Some(Side::Left),
        (false, true) => return Some(Side::Right),
        (true, true) => return None,
        _ => {}
    }
    let mut mark = None;
    for tok in nice.split(' ') {
        match tok {
            "l" => mark = Some(Side::Left),
            "r" => mark = Some(Side::Right),
            _ => {}
        }
    }
    mark
}

/// One candidate joint for a goal.
#[derive(Clone, Debug)]
struct Cand {
    joint: usize,
    /// The bone coincides with its parent (an optional bone left out).
    skipped: bool,
    /// Own score plus the best child goals.
    key: f32,
    /// Goal index -> joint for this candidate and everything chosen beneath it.
    assign: Vec<(usize, usize)>,
}

struct Mapper<'a> {
    skel: &'a Skeleton,
    pos: Vec<Vec3>,
    children: Vec<Vec<usize>>,
    roots: Vec<usize>,
    nice: Vec<String>,
    real: Vec<bool>,
    goals: Vec<Goal>,
    goal_children: Vec<Vec<usize>>,
    /// Character frame: columns are the character's right(left)/up/forward axes in world space.
    frame: Mat3,
    use_side: bool,
    memo: HashMap<(usize, usize, usize), Rc<Vec<Cand>>>,
}

const NONE: usize = usize::MAX;

impl<'a> Mapper<'a> {
    fn new(skel: &'a Skeleton, real: Vec<bool>) -> Self {
        let n = skel.joints.len();
        let pos = skel.joints.iter().map(|j| j.inverse_bind.inverse().w_axis.truncate()).collect();
        let mut children = vec![Vec::new(); n];
        let mut roots = Vec::new();
        for (i, j) in skel.joints.iter().enumerate() {
            match j.parent {
                Some(p) => children[p].push(i),
                None => roots.push(i),
            }
        }
        let nice = skel.joints.iter().map(|j| nice_name(&j.name)).collect();
        Self {
            skel,
            pos,
            children,
            roots,
            nice,
            real,
            goals: Vec::new(),
            goal_children: Vec::new(),
            frame: Mat3::IDENTITY,
            use_side: true,
            memo: HashMap::new(),
        }
    }

    fn set_goals(&mut self, goals: Vec<Goal>) {
        self.goal_children = goals
            .iter()
            .map(|g| goals.iter().enumerate().filter(|(_, c)| c.parent == Some(g.bone)).map(|(i, _)| i).collect())
            .collect();
        self.goals = goals;
        self.memo.clear();
    }

    fn set_frame(&mut self, frame: Mat3) {
        self.frame = frame;
        self.memo.clear();
    }

    fn is_ancestor(&self, a: usize, mut j: usize) -> bool {
        while let Some(p) = self.skel.joints[j].parent {
            if p == a {
                return true;
            }
            j = p;
        }
        false
    }

    /// Joints `min..=max` real levels below `p` (`None` = below the virtual root of all roots).
    fn window(&self, p: Option<usize>, min: u8, max: u8) -> Vec<usize> {
        let mut out = Vec::new();
        let mut stack: Vec<(usize, u8)> = match p {
            Some(p) => self.children[p].iter().map(|&c| (c, 0)).collect(),
            None => self.roots.iter().map(|&c| (c, 0)).collect(),
        };
        stack.reverse();
        while let Some((j, before)) = stack.pop() {
            let step = before + self.real[j] as u8;
            if self.real[j] && step >= min.max(1) && step <= max {
                out.push(j);
            }
            if step < max || !self.real[j] {
                for &c in self.children[j].iter().rev() {
                    stack.push((c, step));
                }
            }
        }
        out
    }

    /// Own score (`score`, `sibling`) of joint `c` for `goal` below parent joint `p`.
    fn own_score(&self, goal: &Goal, parent_side: Side, c: usize, p: Option<usize>, g: Option<usize>) -> (f32, f32) {
        let name = &self.nice[c];
        let (mut score, mut sib) = (0.0f32, 0.0f32);

        if goal.rejects.iter().any(|k| name.contains(k)) {
            score -= 1000.0;
        } else if !goal.keywords.is_empty() {
            let hit = |n: &str| goal.keywords.iter().any(|k| n.contains(k));
            let mut anc = self.skel.joints[c].parent;
            let mut ancestor_hit = false;
            while let Some(a) = anc {
                if Some(a) == p {
                    break;
                }
                if hit(&self.nice[a]) {
                    ancestor_hit = true;
                    break;
                }
                anc = self.skel.joints[a].parent;
            }
            if ancestor_hit {
                score -= 20.0;
            } else if hit(name) {
                score += 20.0;
            } else if goal.strict {
                score -= 1000.0;
            }
        } else if goal.always_include {
            score += 1.0;
        }

        let d_scale = if goal.optional { 5.0 } else { 10.0 };
        if let Some(p) = p {
            let v = self.pos[c] - self.pos[p];
            let len = v.length();
            if len < 1e-6 {
                score -= 1000.0;
            } else if goal.dir != [0.0; 3] {
                let local = self.frame.transpose() * (v / len);
                let d = local.dot(Vec3::from(goal.dir).normalize()) * d_scale;
                sib += d;
                if d > 0.0 {
                    score += 10.0;
                }
            }
        }

        if self.use_side {
            let pts = match (goal.side, side_mark(name)) {
                (_, None) => 0.0,
                (Side::Center, Some(_)) => -1000.0,
                (s, Some(m)) if s == m => 15.0,
                (_, Some(_)) => -1000.0,
            };
            if parent_side == Side::Center || pts < 0.0 {
                sib += pts;
            }
        }

        if score > 0.0 {
            if goal.optional {
                score += 5.0;
            }
            if !self.children[c].is_empty() && self.goal_children_of(goal.bone).is_empty() {
                score += 1.0;
            }
            if let (Some(p), Some(g)) = (p, g)
                && goal.len_ratio > 0.0
            {
                let denom = self.pos[p].distance(self.pos[g]);
                let num = self.pos[c].distance(self.pos[p]);
                if denom > 1e-6 && num > 1e-6 {
                    let off = ((num / denom).log2() - goal.len_ratio.log2()).abs();
                    score += 10.0 * (1.0 - 0.6 * off).clamp(0.0, 1.0);
                }
            }
        }
        (score, sib)
    }

    fn goal_children_of(&self, bone: HumanBone) -> Vec<usize> {
        self.goals.iter().enumerate().filter(|(_, c)| c.parent == Some(bone)).map(|(i, _)| i).collect()
    }

    /// Top candidates for goal `gi` below parent joint `p` (grandparent `g`).
    fn candidates(&mut self, gi: usize, p: Option<usize>, g: Option<usize>) -> Rc<Vec<Cand>> {
        let key = (gi, p.unwrap_or(NONE), g.unwrap_or(NONE));
        if let Some(c) = self.memo.get(&key) {
            return c.clone();
        }
        let goal = self.goals[gi];
        let parent_side = goal
            .parent
            .and_then(|pb| self.goals.iter().find(|x| x.bone == pb))
            .map(|x| x.side)
            .unwrap_or(Side::Center);
        let kids = self.goal_children[gi].clone();
        let mut out: Vec<Cand> = Vec::new();

        if goal.min_step == 0
            && let Some(p) = p
        {
            let (sum, assign) = self.combine(&kids, Some(p), g);
            out.push(Cand { joint: p, skipped: true, key: sum, assign });
        }

        let mut scored: Vec<Cand> = Vec::new();
        for c in self.window(p, goal.min_step, goal.max_step) {
            let (score, sib) = self.own_score(&goal, parent_side, c, p, g);
            if score + sib <= -500.0 {
                continue;
            }
            let (sum, mut assign) = self.combine(&kids, Some(c), p);
            let total = score + sum;
            if total + sib <= 0.0 {
                continue;
            }
            assign.push((gi, c));
            scored.push(Cand { joint: c, skipped: false, key: total + sib, assign });
        }
        scored.sort_by(|a, b| b.key.total_cmp(&a.key).then(a.joint.cmp(&b.joint)));
        scored.truncate(3);
        out.extend(scored);
        let rc = Rc::new(out);
        self.memo.insert(key, rc.clone());
        rc
    }

    /// Best conflict-free choice among sibling goals: tries every combination of
    /// their top candidates (or none) and keeps the highest total.
    fn combine(&mut self, goals: &[usize], p: Option<usize>, g: Option<usize>) -> (f32, Vec<(usize, usize)>) {
        if goals.is_empty() {
            return (0.0, Vec::new());
        }
        let lists: Vec<Rc<Vec<Cand>>> = goals.iter().map(|&gi| self.candidates(gi, p, g)).collect();
        let mut best = (0.0f32, Vec::new());
        let mut chosen: Vec<&Cand> = Vec::new();
        self.search(&lists, 0, &mut chosen, 0.0, &mut best);
        best
    }

    fn conflicts(&self, a: &Cand, b: &Cand) -> bool {
        if a.assign.iter().any(|(_, ja)| b.assign.iter().any(|(_, jb)| ja == jb)) {
            return true;
        }
        if !a.skipped && !b.skipped {
            return self.is_ancestor(a.joint, b.joint) || self.is_ancestor(b.joint, a.joint);
        }
        false
    }

    fn search<'c>(&self, lists: &'c [Rc<Vec<Cand>>], i: usize, chosen: &mut Vec<&'c Cand>, sum: f32, best: &mut (f32, Vec<(usize, usize)>)) {
        if i == lists.len() {
            if sum > best.0 || best.1.is_empty() && sum >= best.0 && !chosen.is_empty() {
                best.0 = sum;
                best.1 = chosen.iter().flat_map(|c| c.assign.iter().copied()).collect();
            }
            return;
        }
        // Leave this goal unmapped.
        self.search(lists, i + 1, chosen, sum, best);
        for c in lists[i].iter() {
            if chosen.iter().any(|o| self.conflicts(o, c)) {
                continue;
            }
            chosen.push(c);
            self.search(lists, i + 1, chosen, sum + c.key, best);
            chosen.pop();
        }
    }

    /// Run the table from the virtual root (body) or below a parent joint.
    fn solve(&mut self, roots: &[usize], p: Option<usize>, g: Option<usize>) -> Vec<(HumanBone, usize)> {
        let (_, assign) = self.combine(roots, p, g);
        let mut out: Vec<(HumanBone, usize)> = Vec::new();
        for (gi, j) in assign {
            let goal = &self.goals[gi];
            if goal.pseudo || out.iter().any(|(b, _)| *b == goal.bone) {
                continue;
            }
            out.push((goal.bone, j));
        }
        out
    }
}

/// Result of [`auto_map`].
#[derive(Clone, Debug, Default)]
pub struct AutoMap {
    /// Mapped bones with their joint index, in [`ALL`] order.
    pub bones: Vec<(HumanBone, usize)>,
    /// Mapped by the 3ds Max Biped name table.
    pub biped: bool,
}

impl AutoMap {
    pub fn joint(&self, bone: HumanBone) -> Option<usize> {
        self.bones.iter().find(|(b, _)| *b == bone).map(|(_, j)| *j)
    }

    /// Number of the 15 required bones that are mapped.
    pub fn required_count(&self) -> usize {
        self.bones.iter().filter(|(b, _)| b.is_required()).count()
    }

    /// Whether all 15 required bones are mapped.
    pub fn has_required(&self) -> bool {
        self.required_count() == REQUIRED_BONE_COUNT
    }

    /// The map as `(bone, joint name)` pairs, as stored in a `RigImport`.
    pub fn to_human(&self, skel: &Skeleton) -> Vec<(HumanBone, String)> {
        self.bones.iter().map(|(b, j)| (*b, skel.joints[*j].name.clone())).collect()
    }
}

fn sort_bones(v: &mut [(HumanBone, usize)]) {
    v.sort_by_key(|(b, _)| *b as usize);
}

/// Map `skel` to human bones: the Biped table when the rig is a 3ds Max Biped,
/// otherwise the structural search. Every joint counts as a real bone.
pub fn auto_map(skel: &Skeleton) -> AutoMap {
    auto_map_with(skel, None)
}

/// [`auto_map`] with a per-joint "real bone" mask (skinned, branching or
/// carrying geometry). Dummy joints add no hierarchy level; when the first
/// pass finds fewer than 15 bones the search reruns with every joint real.
pub fn auto_map_with(skel: &Skeleton, real: Option<&[bool]>) -> AutoMap {
    if skel.joints.len() < REQUIRED_BONE_COUNT {
        return AutoMap::default();
    }
    if let Some(m) = map_biped(skel) {
        return m;
    }
    let all = vec![true; skel.joints.len()];
    let mut map = match real.filter(|r| r.len() == skel.joints.len()) {
        Some(r) => {
            let m = map_structural(skel, r.to_vec());
            if m.required_count() < REQUIRED_BONE_COUNT { map_structural(skel, all) } else { m }
        }
        None => map_structural(skel, all),
    };
    sort_bones(&mut map.bones);
    map
}

fn map_structural(skel: &Skeleton, real: Vec<bool>) -> AutoMap {
    let mut m = Mapper::new(skel, real);
    m.set_goals(body_goals());
    let roots: Vec<usize> = (0..m.goals.len()).filter(|&i| m.goals[i].parent.is_none()).collect();
    let mut bones = m.solve(&roots, None, None);
    let mut frame = Mat3::IDENTITY;

    // A rotated or Z-up rig: detect the character frame and remap in it.
    if let Some(f) = orientation(&m, &bones)
        && frame_off_identity(&f)
    {
        m.set_frame(f);
        let again = m.solve(&roots, None, None);
        let count = |v: &[(HumanBone, usize)]| v.iter().filter(|(b, _)| b.is_required()).count();
        if count(&again) >= count(&bones) {
            bones = again;
            frame = f;
        } else {
            m.set_frame(Mat3::IDENTITY);
        }
    }

    // Chest unmapped but UpperChest mapped: move it down.
    let has = |bones: &[(HumanBone, usize)], b: HumanBone| bones.iter().any(|(x, _)| *x == b);
    if has(&bones, B::UpperChest) && !has(&bones, B::Chest) {
        for e in bones.iter_mut() {
            if e.0 == B::UpperChest {
                e.0 = B::Chest;
            }
        }
    }

    // Finger pass per hand.
    m.use_side = false;
    for left in [true, false] {
        let (hand_b, arm_b) = if left { (B::LeftHand, B::LeftLowerArm) } else { (B::RightHand, B::RightLowerArm) };
        let (Some(hand), Some(arm)) = (
            bones.iter().find(|(b, _)| *b == hand_b).map(|(_, j)| *j),
            bones.iter().find(|(b, _)| *b == arm_b).map(|(_, j)| *j),
        ) else {
            continue;
        };
        let along = m.pos[hand] - m.pos[arm];
        if along.length() < 1e-6 {
            continue;
        }
        let x = along.normalize();
        let up = frame.y_axis;
        let mut y = up - x * up.dot(x);
        if y.length() < 1e-4 {
            y = frame.z_axis - x * frame.z_axis.dot(x);
        }
        let y = y.normalize();
        let z = if left { x.cross(y) } else { y.cross(x) };
        m.set_goals(finger_goals(left));
        m.set_frame(Mat3::from_cols(x, y, z));
        let roots: Vec<usize> = (0..m.goals.len()).filter(|&i| m.goals[i].parent == Some(hand_b)).collect();
        let fingers = m.solve(&roots, Some(hand), Some(arm));
        if fingers.len() >= 3 {
            bones.extend(fingers);
        }
    }
    sort_bones(&mut bones);
    AutoMap { bones, biped: false }
}

/// The character frame from the legs and arms, if they are mapped.
fn orientation(m: &Mapper, bones: &[(HumanBone, usize)]) -> Option<Mat3> {
    let j = |b: HumanBone| bones.iter().find(|(x, _)| *x == b).map(|(_, j)| m.pos[*j]);
    let (lul, rul) = (j(B::LeftUpperLeg)?, j(B::RightUpperLeg)?);
    let mut across = lul - rul;
    let mut top = Vec3::ZERO;
    let mut n = 0.0;
    if let (Some(la), Some(ra)) = (j(B::LeftUpperArm), j(B::RightUpperArm)) {
        across += la - ra;
        top = (la + ra) * 0.5;
        n = 1.0;
    }
    if n == 0.0 {
        top = j(B::Head)?;
    }
    let x = across.try_normalize()?;
    let up = top - (lul + rul) * 0.5;
    let y = (up - x * up.dot(x)).try_normalize()?;
    Some(Mat3::from_cols(x, y, x.cross(y)))
}

/// More than 20 degrees off the world axes in up or forward.
fn frame_off_identity(f: &Mat3) -> bool {
    let limit = 20f32.to_radians().cos();
    f.y_axis.dot(Vec3::Y) < limit || f.z_axis.dot(Vec3::Z) < limit
}

// ---------------------------------------------------------------- biped

/// 3ds Max Biped name suffixes and their human bones (52 entries).
fn biped_table() -> Vec<(String, HumanBone)> {
    let mut t: Vec<(String, HumanBone)> = vec![
        ("Pelvis".into(), B::Hips),
        ("Spine".into(), B::Spine),
        ("Spine1".into(), B::Chest),
        ("Spine2".into(), B::UpperChest),
        ("Neck".into(), B::Neck),
        ("Head".into(), B::Head),
    ];
    for left in [true, false] {
        let s = if left { "L" } else { "R" };
        let pick = |l: HumanBone| if left { l } else { mirror_bone(l) };
        t.push((format!("{s} Clavicle"), pick(B::LeftShoulder)));
        t.push((format!("{s} UpperArm"), pick(B::LeftUpperArm)));
        t.push((format!("{s} Forearm"), pick(B::LeftLowerArm)));
        t.push((format!("{s} Hand"), pick(B::LeftHand)));
        t.push((format!("{s} Thigh"), pick(B::LeftUpperLeg)));
        t.push((format!("{s} Calf"), pick(B::LeftLowerLeg)));
        t.push((format!("{s} Foot"), pick(B::LeftFoot)));
        t.push((format!("{s} Toe0"), pick(B::LeftToes)));
        // Finger0 (thumb) has joints 0, 01, 02; Finger1..4 have 1, 11, 12 and so on.
        for f in 0..5 {
            for k in 0..3 {
                let name = if k == 0 { format!("{s} Finger{f}") } else { format!("{s} Finger{f}{}", k) };
                t.push((name, finger_bone(left, f, k)));
            }
        }
    }
    t
}

/// The Biped name-table mapping alone (the Configure menu's "Biped" entry);
/// `None` when the rig is not a 3ds Max Biped.
pub fn biped_map(skel: &Skeleton) -> Option<AutoMap> {
    map_biped(skel)
}

/// 3ds Max Biped rigs map by name suffix; `None` when a required bone is missing.
fn map_biped(skel: &Skeleton) -> Option<AutoMap> {
    // Cheap reject: every biped has a Pelvis joint.
    if !skel.joints.iter().any(|j| j.name.ends_with("Pelvis")) {
        return None;
    }
    let mut bones: Vec<(HumanBone, usize)> = Vec::new();
    for (suffix, bone) in biped_table() {
        if let Some(j) = skel.joints.iter().position(|j| j.name.ends_with(&suffix) && boundary(&j.name, &suffix)) {
            bones.push((bone, j));
        }
    }
    let mut map = AutoMap { bones, biped: true };
    if !map.has_required() {
        return None;
    }
    // Two-link spines: Spine1 is the chest of a rig without Spine2 already; one link has no chest.
    if map.joint(B::UpperChest).is_some() && map.joint(B::Chest).is_none() {
        for e in map.bones.iter_mut() {
            if e.0 == B::UpperChest {
                e.0 = B::Chest;
            }
        }
    }
    sort_bones(&mut map.bones);
    Some(map)
}

/// The suffix starts a word (`Bip001 L Hand`, not `xL Hand`).
fn boundary(name: &str, suffix: &str) -> bool {
    let head = &name[..name.len() - suffix.len()];
    head.is_empty() || head.ends_with(' ') || head.ends_with('_') || head.ends_with(':')
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Joint;
    use glam::{Mat4, Quat};

    /// A hand-built humanoid. `name` gives the joint name of each human bone;
    /// `skip` removes bones (their children attach to the nearest kept ancestor).
    struct Build<'a> {
        name: &'a dyn Fn(HumanBone) -> String,
        skip: &'a [HumanBone],
        fingers: bool,
        nubs: bool,
        twist: bool,
        /// Rotate the whole rig (world <- canonical).
        rot: Quat,
        /// Sort joints so the children come before an unrelated sibling (shuffles array order).
        reverse_children: bool,
        /// Drop fingers' bones down to this many per finger (0 = all 3).
        pose_a: bool,
    }

    impl<'a> Build<'a> {
        fn new(name: &'a dyn Fn(HumanBone) -> String) -> Self {
            Self { name, skip: &[], fingers: true, nubs: true, twist: false, rot: Quat::IDENTITY, reverse_children: false, pose_a: false }
        }
    }

    fn offset(b: HumanBone, pose_a: bool) -> Vec3 {
        let n = b.name();
        let x = if n.starts_with("Left") { 1.0 } else { -1.0 };
        let arm = |v: Vec3| if pose_a { Vec3::new(v.x * 0.7, -v.x.abs() * 0.7, 0.0) * Vec3::new(1.0, 1.0, 1.0) } else { v };
        match b {
            B::Hips => Vec3::new(0.0, 1.0, 0.0),
            B::Spine => Vec3::new(0.0, 0.1, 0.0),
            B::Chest => Vec3::new(0.0, 0.15, 0.0),
            B::UpperChest => Vec3::new(0.0, 0.12, 0.0),
            B::Neck => Vec3::new(0.0, 0.12, 0.0),
            B::Head => Vec3::new(0.0, 0.1, 0.0),
            B::LeftShoulder | B::RightShoulder => Vec3::new(0.04 * x, 0.1, 0.0),
            B::LeftUpperArm | B::RightUpperArm => Vec3::new(0.1 * x, 0.0, 0.0),
            B::LeftLowerArm | B::RightLowerArm => arm(Vec3::new(0.28 * x, 0.0, 0.0)),
            B::LeftHand | B::RightHand => arm(Vec3::new(0.26 * x, 0.0, 0.0)),
            B::LeftUpperLeg | B::RightUpperLeg => Vec3::new(0.09 * x, -0.05, 0.0),
            B::LeftLowerLeg | B::RightLowerLeg => Vec3::new(0.0, -0.42, 0.0),
            B::LeftFoot | B::RightFoot => Vec3::new(0.0, -0.42, 0.0),
            B::LeftToes | B::RightToes => Vec3::new(0.0, -0.05, 0.12),
            B::LeftEye | B::RightEye => Vec3::new(0.03 * x, 0.08, 0.08),
            B::Jaw => Vec3::new(0.0, -0.05, 0.05),
            _ => {
                // Fingers: proximal off the hand, then 0.035 along the finger.
                let s = n.replace("Left", "").replace("Right", "");
                let (root, dir) = if s.starts_with("Thumb") {
                    (Vec3::new(0.03, 0.0, 0.03), Vec3::new(0.5, 0.0, 0.85))
                } else if s.starts_with("Index") {
                    (Vec3::new(0.09, 0.0, 0.02), Vec3::X)
                } else if s.starts_with("Middle") {
                    (Vec3::new(0.09, 0.0, 0.0), Vec3::X)
                } else if s.starts_with("Ring") {
                    (Vec3::new(0.09, 0.0, -0.02), Vec3::X)
                } else {
                    (Vec3::new(0.08, 0.0, -0.04), Vec3::X)
                };
                let v = if s.ends_with("Proximal") { root } else { dir.normalize() * 0.035 };
                let v = Vec3::new(v.x * x, v.y, v.z);
                if pose_a { Vec3::new(v.x * 0.7, -v.x.abs() * 0.7, v.z) } else { v }
            }
        }
    }

    fn build(b: &Build) -> Skeleton {
        // World position of every present human bone.
        let mut world = [Vec3::ZERO; 55];
        let present = |h: HumanBone| !b.skip.contains(&h) && (b.fingers || !(h as usize >= 24 && h != B::UpperChest));
        let depth = |mut h: HumanBone| {
            let mut d = 0;
            while let Some(p) = h.parent() {
                d += 1;
                h = p;
            }
            d
        };
        let mut by_depth = ALL;
        by_depth.sort_by_key(|h| depth(*h));
        for h in by_depth {
            let p = h.parent().map(|p| world[p as usize]).unwrap_or(Vec3::ZERO);
            world[h as usize] = p + offset(h, b.pose_a);
        }
        let mut names: Vec<(String, Option<usize>, Vec3)> = Vec::new();
        let mut index: HashMap<HumanBone, usize> = HashMap::new();
        // Parents first: ALL puts UpperChest last, so order by depth.
        let mut order: Vec<HumanBone> = ALL.iter().copied().filter(|h| present(*h)).collect();
        order.sort_by_key(|h| (depth(*h), *h as usize));
        if b.reverse_children {
            order.sort_by_key(|h| (depth(*h), std::cmp::Reverse(*h as usize)));
        }
        for h in order {
            let parent = h.mapped_ancestor(|a| index.contains_key(&a)).map(|a| index[&a]);
            index.insert(h, names.len());
            names.push(((b.name)(h), parent, world[h as usize]));
        }
        let extra = |names: &mut Vec<(String, Option<usize>, Vec3)>, parent_bone: HumanBone, suffix: &str, off: Vec3| {
            let pi = index[&parent_bone];
            let pos = names[pi].2 + off;
            names.push((format!("{}{suffix}", names[pi].0), Some(pi), pos));
        };
        if b.nubs {
            extra(&mut names, B::Head, "_End", Vec3::new(0.0, 0.15, 0.0));
            if index.contains_key(&B::LeftToes) {
                extra(&mut names, B::LeftToes, "_End", Vec3::new(0.0, 0.0, 0.05));
                extra(&mut names, B::RightToes, "_End", Vec3::new(0.0, 0.0, 0.05));
            }
            if b.fingers {
                for h in [B::LeftIndexDistal, B::RightIndexDistal, B::LeftThumbDistal, B::RightThumbDistal] {
                    extra(&mut names, h, "_End", Vec3::new(0.0, 0.0, 0.02));
                }
            }
        }
        if b.twist {
            for (h, off) in [
                (B::LeftLowerArm, Vec3::new(0.14, 0.0, 0.0)),
                (B::RightLowerArm, Vec3::new(-0.14, 0.0, 0.0)),
                (B::LeftUpperArm, Vec3::new(0.14, 0.0, 0.0)),
                (B::RightUpperArm, Vec3::new(-0.14, 0.0, 0.0)),
                (B::LeftLowerLeg, Vec3::new(0.0, -0.21, 0.0)),
                (B::RightLowerLeg, Vec3::new(0.0, -0.21, 0.0)),
            ] {
                extra(&mut names, h, "Twist", off);
            }
        }
        let joints = names
            .iter()
            .map(|(n, p, w)| {
                let w = b.rot * *w;
                let pw = p.map(|p| b.rot * names[p].2).unwrap_or(Vec3::ZERO);
                Joint {
                    name: n.clone(),
                    parent: *p,
                    inverse_bind: Mat4::from_translation(-w),
                    rest_translation: w - pw,
                    rest_rotation: Quat::IDENTITY,
                    rest_scale: Vec3::ONE,
                }
            })
            .collect();
        Skeleton { joints, rig: None, ..Default::default() }
    }

    fn mapped(skel: &Skeleton, m: &AutoMap, b: HumanBone) -> Option<String> {
        m.joint(b).map(|j| skel.joints[j].name.clone())
    }

    fn assert_map(skel: &Skeleton, m: &AutoMap, name: &dyn Fn(HumanBone) -> String, expect: &[HumanBone]) {
        for b in expect {
            assert_eq!(mapped(skel, m, *b), Some(name(*b)), "{} in {:?}", b.name(), m.bones.iter().map(|(b, j)| (b.name(), &skel.joints[*j].name)).collect::<Vec<_>>());
        }
    }

    const BODY: [HumanBone; 20] = [
        B::Hips, B::Spine, B::Chest, B::Neck, B::Head, B::LeftShoulder, B::RightShoulder, B::LeftUpperArm,
        B::RightUpperArm, B::LeftLowerArm, B::RightLowerArm, B::LeftHand, B::RightHand, B::LeftUpperLeg,
        B::RightUpperLeg, B::LeftLowerLeg, B::RightLowerLeg, B::LeftFoot, B::RightFoot, B::LeftToes,
    ];

    fn finger_names(b: HumanBone, f: &dyn Fn(&str, &str, &str) -> String) -> Option<String> {
        let n = b.name();
        let side = if n.starts_with("Left") { "Left" } else if n.starts_with("Right") { "Right" } else { return None };
        let s = &n[side.len()..];
        for finger in ["Thumb", "Index", "Middle", "Ring", "Little"] {
            if let Some(joint) = s.strip_prefix(finger) {
                return Some(f(side, finger, joint));
            }
        }
        None
    }

    fn mixamo(b: HumanBone) -> String {
        let m = |s: &str| format!("mixamorig:{s}");
        if let Some(f) = finger_names(b, &|side, finger, joint| {
            let n = match joint { "Proximal" => 1, "Intermediate" => 2, _ => 3 };
            format!("mixamorig:{side}Hand{finger}{n}")
        }) {
            return f;
        }
        match b {
            B::Hips => m("Hips"),
            B::Spine => m("Spine"),
            B::Chest => m("Spine1"),
            B::UpperChest => m("Spine2"),
            B::Neck => m("Neck"),
            B::Head => m("Head"),
            B::LeftShoulder => m("LeftShoulder"),
            B::RightShoulder => m("RightShoulder"),
            B::LeftUpperArm => m("LeftArm"),
            B::RightUpperArm => m("RightArm"),
            B::LeftLowerArm => m("LeftForeArm"),
            B::RightLowerArm => m("RightForeArm"),
            B::LeftHand => m("LeftHand"),
            B::RightHand => m("RightHand"),
            B::LeftUpperLeg => m("LeftUpLeg"),
            B::RightUpperLeg => m("RightUpLeg"),
            B::LeftLowerLeg => m("LeftLeg"),
            B::RightLowerLeg => m("RightLeg"),
            B::LeftFoot => m("LeftFoot"),
            B::RightFoot => m("RightFoot"),
            B::LeftToes => m("LeftToeBase"),
            B::RightToes => m("RightToeBase"),
            B::LeftEye => m("LeftEye"),
            B::RightEye => m("RightEye"),
            B::Jaw => m("Jaw"),
            _ => unreachable!(),
        }
    }

    fn vrm(b: HumanBone) -> String {
        let s = |side: &str, rest: &str| format!("J_Bip_{side}_{rest}");
        let (side, rest) = if let Some(r) = b.name().strip_prefix("Left") { ("L", r) } else if let Some(r) = b.name().strip_prefix("Right") { ("R", r) } else { ("C", b.name()) };
        let out = match rest {
            "UpperArm" | "LowerArm" | "Hand" | "UpperLeg" | "LowerLeg" | "Foot" | "Shoulder" | "Toes" | "Eye" => s(side, rest),
            "Hips" | "Spine" | "Chest" | "UpperChest" | "Neck" | "Head" | "Jaw" => s("C", rest),
            r => {
                let (finger, joint) = ["Thumb", "Index", "Middle", "Ring", "Little"].iter().find_map(|f| r.strip_prefix(f).map(|j| (*f, j))).unwrap();
                let n = match joint { "Proximal" => 1, "Intermediate" => 2, _ => 3 };
                s(side, &format!("{finger}{n}"))
            }
        };
        if rest == "LowerArm" {
            // "LowerArm" is the canonical name in this mapper's output; no
            // substitution is needed.
        }
        out
    }

    fn unity(b: HumanBone) -> String {
        // Unity's own humanoid naming, as used by many store characters.
        b.name().to_string()
    }

    fn blender(b: HumanBone) -> String {
        let n = b.name();
        let (side, rest) = if let Some(r) = n.strip_prefix("Left") { (".L", r) } else if let Some(r) = n.strip_prefix("Right") { (".R", r) } else { ("", n) };
        let base = match rest {
            "UpperArm" => "upper_arm".to_string(),
            "LowerArm" => "forearm".to_string(),
            "UpperLeg" => "thigh".to_string(),
            "LowerLeg" => "shin".to_string(),
            "Hips" => "pelvis".to_string(),
            "Toes" => "toe".to_string(),
            "UpperChest" => "chest.001".to_string(),
            r if r.starts_with("Thumb") || r.starts_with("Index") || r.starts_with("Middle") || r.starts_with("Ring") || r.starts_with("Little") => {
                let (finger, joint) = ["Thumb", "Index", "Middle", "Ring", "Little"].iter().find_map(|f| r.strip_prefix(f).map(|j| (*f, j))).unwrap();
                let n = match joint { "Proximal" => 1, "Intermediate" => 2, _ => 3 };
                let f = if finger == "Little" { "pinky" } else { finger };
                return format!("f_{}.0{n}{side}", f.to_lowercase()).replace("f_thumb", "thumb");
            }
            r => r.to_lowercase(),
        };
        format!("{base}{side}")
    }

    fn ue(b: HumanBone) -> String {
        // Unreal mannequin style: lower-case with _l / _r suffix.
        let n = b.name();
        let (side, rest) = if let Some(r) = n.strip_prefix("Left") { ("_l", r) } else if let Some(r) = n.strip_prefix("Right") { ("_r", r) } else { ("", n) };
        let base = match rest {
            "Hips" => "pelvis".to_string(),
            "UpperArm" => "upperarm".to_string(),
            "LowerArm" => "lowerarm".to_string(),
            "UpperLeg" => "thigh".to_string(),
            "LowerLeg" => "calf".to_string(),
            "Toes" => "ball".to_string(),
            "Spine" => "spine_01".to_string(),
            "Chest" => "spine_02".to_string(),
            "UpperChest" => "spine_03".to_string(),
            "Shoulder" => "clavicle".to_string(),
            r if r.starts_with("Thumb") || r.starts_with("Index") || r.starts_with("Middle") || r.starts_with("Ring") || r.starts_with("Little") => {
                let (finger, joint) = ["Thumb", "Index", "Middle", "Ring", "Little"].iter().find_map(|f| r.strip_prefix(f).map(|j| (*f, j))).unwrap();
                let n = match joint { "Proximal" => 1, "Intermediate" => 2, _ => 3 };
                format!("{}_0{n}", finger.to_lowercase())
            }
            r => r.to_lowercase(),
        };
        format!("{base}{side}")
    }

    fn check_fingers(skel: &Skeleton, m: &AutoMap, name: &dyn Fn(HumanBone) -> String) {
        for b in &ALL[24..54] {
            assert_eq!(mapped(skel, m, *b), Some(name(*b)), "{}", b.name());
        }
    }

    #[test]
    fn nice_names_and_sides() {
        assert_eq!(nice_name("mixamorig:LeftUpLeg"), "left up leg");
        assert_eq!(nice_name("J_Bip_L_UpperArm"), "j bip l upper arm");
        assert_eq!(nice_name("upper_arm.R"), "upper arm r");
        assert_eq!(nice_name("LHand"), "l hand");
        assert_eq!(side_mark("upper arm r"), Some(Side::Right));
        assert_eq!(side_mark("bip001 l thigh"), Some(Side::Left));
        assert_eq!(side_mark("left hand"), Some(Side::Left));
        assert_eq!(side_mark("spine 1"), None);
        assert_eq!(side_mark("hair"), None);
    }

    #[test]
    fn maps_mixamo_with_fingers_toes_and_nubs() {
        let skel = build(&Build::new(&mixamo));
        let m = auto_map(&skel);
        assert!(!m.biped && m.has_required());
        assert_map(&skel, &m, &mixamo, &BODY);
        assert_map(&skel, &m, &mixamo, &[B::UpperChest, B::RightToes]);
        check_fingers(&skel, &m, &mixamo);
        assert!(mapped(&skel, &m, B::Head).unwrap() == "mixamorig:Head", "nub is not the head");
    }

    #[test]
    fn maps_mixamo_without_upper_chest_and_in_a_pose() {
        let skip = [B::UpperChest, B::LeftEye, B::RightEye, B::Jaw];
        let skel = build(&Build { skip: &skip, pose_a: true, ..Build::new(&mixamo) });
        let m = auto_map(&skel);
        assert!(m.has_required());
        assert_map(&skel, &m, &mixamo, &BODY);
        assert_eq!(m.joint(B::UpperChest), None);
        check_fingers(&skel, &m, &mixamo);
    }

    #[test]
    fn maps_vrm_including_face() {
        let skel = build(&Build::new(&vrm));
        let m = auto_map(&skel);
        assert!(m.has_required());
        assert_map(&skel, &m, &vrm, &BODY);
        assert_map(&skel, &m, &vrm, &[B::LeftEye, B::RightEye, B::Jaw, B::UpperChest]);
        check_fingers(&skel, &m, &vrm);
    }

    #[test]
    fn maps_unity_names_and_twist_bones() {
        let skel = build(&Build { twist: true, ..Build::new(&unity) });
        let m = auto_map(&skel);
        assert!(m.has_required());
        assert_map(&skel, &m, &unity, &BODY);
        assert_map(&skel, &m, &unity, &[B::LeftEye, B::Jaw]);
        check_fingers(&skel, &m, &unity);
    }

    #[test]
    fn maps_blender_dot_lr_names() {
        let skel = build(&Build::new(&blender));
        let m = auto_map(&skel);
        assert!(m.has_required(), "{:?}", m.bones.iter().map(|(b, j)| (b.name(), &skel.joints[*j].name)).collect::<Vec<_>>());
        assert_map(&skel, &m, &blender, &BODY);
        check_fingers(&skel, &m, &blender);
    }

    #[test]
    fn maps_unreal_style_underscore_names() {
        let skel = build(&Build { nubs: false, ..Build::new(&ue) });
        let m = auto_map(&skel);
        assert!(m.has_required());
        assert_map(&skel, &m, &ue, &BODY);
        check_fingers(&skel, &m, &ue);
    }

    #[test]
    fn a_saved_humanmap_reproduces_the_map() {
        // The golden-file pattern: auto-map, save as .humanmap, load it back onto the same rig.
        let skel = build(&Build::new(&mixamo));
        let human = auto_map(&skel).to_human(&skel);
        let map = crate::human_template::HumanMap::new("mixamo", &human);
        let text = ron::ser::to_string_pretty(&map, ron::ser::PrettyConfig::default()).unwrap();
        let back: crate::human_template::HumanMap = ron::from_str(&text).unwrap();
        assert_eq!(crate::human_template::apply_template(&skel, &back), human);
    }

    #[test]
    fn array_order_does_not_matter() {
        let a = build(&Build::new(&mixamo));
        let b = build(&Build { reverse_children: true, ..Build::new(&mixamo) });
        let (ma, mb) = (auto_map(&a), auto_map(&b));
        let named = |s: &Skeleton, m: &AutoMap| m.to_human(s);
        assert_eq!(named(&a, &ma), named(&b, &mb));
    }

    #[test]
    fn structure_beats_missing_names() {
        // Meaningless joint names: only the hierarchy and sides (by position) carry the map.
        let opaque = |b: HumanBone| format!("j{}", b as usize);
        let skip = [B::UpperChest, B::LeftEye, B::RightEye, B::Jaw];
        let skel = build(&Build { skip: &skip, nubs: false, fingers: false, ..Build::new(&opaque) });
        let m = auto_map(&skel);
        assert!(m.has_required(), "{} required", m.required_count());
        assert_map(&skel, &m, &opaque, &[B::Hips, B::Head, B::LeftHand, B::RightHand, B::LeftFoot, B::RightFoot, B::LeftUpperArm, B::RightLowerLeg]);
        // Left is +X (the avatar's left), so the direction prior fixes sides.
        assert_eq!(mapped(&skel, &m, B::LeftUpperLeg), Some(opaque(B::LeftUpperLeg)));
    }

    #[test]
    fn z_up_and_back_facing_rigs_are_remapped_in_their_frame() {
        let z_up = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
        for rot in [z_up, Quat::from_rotation_y(std::f32::consts::PI), Quat::from_rotation_z(0.9)] {
            let skel = build(&Build { rot, ..Build::new(&mixamo) });
            let m = auto_map(&skel);
            assert!(m.has_required(), "{rot:?}: {} required", m.required_count());
            assert_map(&skel, &m, &mixamo, &BODY);
        }
    }

    #[test]
    fn biped_names_map_by_table() {
        let biped = |b: HumanBone| -> String {
            let n = b.name();
            let (s, rest) = if let Some(r) = n.strip_prefix("Left") { ("L ", r) } else if let Some(r) = n.strip_prefix("Right") { ("R ", r) } else { ("", n) };
            let base = match rest {
                "Hips" => "Pelvis".to_string(),
                "Chest" => "Spine1".to_string(),
                "UpperChest" => "Spine2".to_string(),
                "Shoulder" => "Clavicle".to_string(),
                "LowerArm" => "Forearm".to_string(),
                "UpperLeg" => "Thigh".to_string(),
                "LowerLeg" => "Calf".to_string(),
                "Toes" => "Toe0".to_string(),
                r if r.starts_with("Thumb") || r.starts_with("Index") || r.starts_with("Middle") || r.starts_with("Ring") || r.starts_with("Little") => {
                    let (fi, joint) = ["Thumb", "Index", "Middle", "Ring", "Little"].iter().enumerate().find_map(|(i, f)| r.strip_prefix(f).map(|j| (i, j))).unwrap();
                    let k = match joint { "Proximal" => "", "Intermediate" => "1", _ => "2" };
                    format!("Finger{fi}{k}")
                }
                r => r.to_string(),
            };
            format!("Bip001 {s}{base}")
        };
        let skel = build(&Build { nubs: false, ..Build::new(&biped) });
        let m = auto_map(&skel);
        assert!(m.biped && m.has_required());
        assert_map(&skel, &m, &biped, &BODY);
        assert_map(&skel, &m, &biped, &[B::LeftIndexIntermediate, B::RightLittleDistal, B::LeftThumbProximal]);
        assert_eq!(biped_table().len(), 52);
    }

    #[test]
    fn partial_rigs_report_missing_required_bones() {
        let skel = build(&Build { skip: &[B::LeftLowerLeg, B::LeftFoot, B::LeftToes], fingers: false, nubs: false, ..Build::new(&unity) });
        let m = auto_map(&skel);
        assert!(!m.has_required());
        assert_eq!(mapped(&skel, &m, B::Head), Some("Head".into()));
        assert_eq!(m.joint(B::LeftLowerLeg), None);
    }

    #[test]
    fn hand_with_fewer_than_three_fingers_gets_none() {
        let skip = [
            B::LeftIndexProximal, B::LeftIndexIntermediate, B::LeftIndexDistal, B::LeftMiddleProximal, B::LeftMiddleIntermediate, B::LeftMiddleDistal,
            B::LeftRingProximal, B::LeftRingIntermediate, B::LeftRingDistal, B::LeftLittleProximal, B::LeftLittleIntermediate, B::LeftLittleDistal,
            B::LeftThumbDistal,
        ];
        let skel = build(&Build { skip: &skip, nubs: false, ..Build::new(&unity) });
        let m = auto_map(&skel);
        assert!(ALL.iter().filter(|b| b.name().starts_with("Left") && b.name().contains("Thumb") || b.name().contains("LeftIndex")).all(|b| m.joint(*b).is_none()));
        assert!(m.joint(B::RightIndexProximal).is_some(), "the other hand keeps its fingers");
    }
}
