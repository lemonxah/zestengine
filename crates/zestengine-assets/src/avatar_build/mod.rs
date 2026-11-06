//! Non-destructive avatar build: authoring components in, a finished avatar
//! out, on a clone.
//!
//! An avatar is authored with intent components (`Menu Toggle`, `Object
//! Toggle`, `Merge Animator`, `Merge Armature`, `Avatar Menu Item`, and
//! `Avatar Parameters` anywhere below the root). [`build_avatar`] runs ordered
//! passes over the avatar's subtree of a [`SceneFile`] and leaves:
//!
//! - one `Avatar Parameters` on the root holding every declared and generated
//!   parameter, plus an `Avatar Menu` (the finished menu tree);
//! - one `.animctrl` and the `.anim` clips it uses, referenced by the avatar's
//!   Animator, written to `.zest/generated/avatars/<id>/`;
//! - the authoring components removed.
//!
//! The caller passes a **clone** of the scene (package build, entering Play),
//! so the authored scene is never touched. The build is deterministic: names
//! come from hashes of object paths, collections are ordered, and output
//! bytes are produced by the RON writer, so building twice gives identical
//! files (and unchanged files are not rewritten).
//!
//! Pass order (the order the research settled on; later passes rely on
//! earlier ones):
//!
//! 1. `resolve` - find the avatar's Animator and base controller, object paths
//! 2. `scope_parameters` - subtree parameters, hashed internal names, conflicts
//! 3. `assign_menu_parameters` - menu items with reactive children get a parameter
//! 4. `generate_toggles` - `Menu Toggle`: parameter, menu entry, clip, layer
//! 5. `generate_reactive` - `Object Toggle`: follows the nearest menu item
//! 6. `exclusive_tags` - one layer per tag group switching the others off
//! 7. `merge_animators` - other controllers, parameters renamed through scopes
//! 8. `resting_state` - objects a toggle turns on start off (default-on: on)
//! 9. `merge_armature` - the only pass that moves objects (package build only)
//! 10. `defaults_layer` - layer 0 holding the rest value of every animated property
//! 11. `flatten_controller` - one controller: defaults, avatar, merged, generated
//! 12. `finalize_menu` - hierarchy menu items plus generated entries, paged
//! 13. `budget` - synced bits against the avatar's and the project's cap
//! 14. `emit` - write components and files, strip authoring components
//!
//! Also in the order above: `optimise_toggles` (after `exclusive_tags`) folds
//! toggles that animate the same properties into one layer, `lints` (before
//! `budget`) reports authoring mistakes, and `finalize_menu` installs `Menu
//! Installer` pages and downscales menu icons.

mod emit;
mod merge;
mod optimise;
mod skin;
mod toggles;
pub use emit::{ICON_MAX, downscale_icon};
pub use optimise::budget_breakdown;
pub use skin::SkinRebindReport;
#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use zestengine_core::{
    Animator, AvatarDescriptor, AvatarLayer, AvatarMenu, AvatarMenuData, AvatarMenuItem, AvatarMenuNode, AvatarParam, AvatarParamKind, AvatarParamScope,
    AvatarParamSync, AvatarParameters, BlendShapes, BuiltinParam, DriverEntry, DriverOp, MenuInstaller, MenuItemKind, MenuToggle, MergeAnimator, MergeArmature,
    ObjectToggle, PARAMETER_DRIVER, ParamBudget, ParamDriverSettings, ToggleAction, ToggleMode, TypedComponent, normalise_label,
};

use crate::{
    AnimClip, AnimCondition, AnimLayer, AnimParamKind, AnimParameter, AnimState, AnimTrack, AnimTransition, AnimatorController, CondOp, ComponentData,
    Keyframe, Motion, SceneEntry, SceneFile, StateMachine, TrackProperty, package::sha256_hex,
};

/// Where generated avatar assets go, under the project root.
pub const GENERATED_DIR: &str = ".zest/generated/avatars";
/// Threshold above which a slider counts as "on" when it is inactive at zero.
const SLIDER_ON: f32 = 0.001;
/// Length of a one-frame constant clip.
const CONSTANT_CLIP_LEN: f32 = 1.0 / 60.0;

/// What the build needs from its surroundings.
pub struct BuildEnv<'a> {
    /// Controllers, clips and menus are read from here.
    pub project_root: &'a Path,
    /// Write the generated files under [`GENERATED_DIR`] (they are always
    /// returned in [`AvatarBuild::files`]).
    pub write: bool,
    /// Allow passes that move or delete objects (`Merge Armature`). Off for
    /// the editor's Play, whose snapshot only restores components and flags.
    pub structural: bool,
    /// The project's synced-bit cap (`ContentLimits::max_param_bits`).
    pub budget_cap: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GeneratedLayer {
    pub name: String,
    /// Where it came from: an object path and component.
    pub source: String,
}

/// What a build produced besides the edited scene.
#[derive(Clone, Debug, Default)]
pub struct AvatarBuild {
    /// Project-relative folder the files live in.
    pub out_dir: String,
    /// Project-relative path -> bytes of every generated file.
    pub files: BTreeMap<String, Vec<u8>>,
    /// Every parameter the finished avatar declares (declared + generated).
    pub params: Vec<AvatarParam>,
    /// The parameters the build added (toggles, auto-assigned, drive targets).
    pub generated_params: Vec<String>,
    pub menu: AvatarMenu,
    /// Layers the build created or merged, in controller order.
    pub layers: Vec<GeneratedLayer>,
    pub budget: Option<ParamBudget>,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
    /// Passes that ran, in order.
    pub passes: Vec<&'static str>,
    /// Index of the avatar root in the edited scene (changes only when a
    /// structural pass reorders entries).
    pub root: usize,
    /// Hash of everything the build read (the avatar's entries, the controller
    /// and clips they reference, menu assets, limits). Written next to the
    /// outputs; equal hashes with all outputs present mean nothing changed.
    pub input_hash: String,
    /// The previous build used the same inputs and its files are all on disk,
    /// so no file was written or cleaned up.
    pub unchanged: bool,
    /// Layers folded into others by the toggle-table optimiser.
    pub layers_merged: usize,
    /// One entry per outfit model whose skin `Merge Armature` rebound onto the avatar's skeleton.
    pub skin_rebinds: Vec<SkinRebindReport>,
}

impl AvatarBuild {
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }
}

/// Every authoring component the build consumes.
pub fn authoring_kinds() -> [&'static str; 6] {
    [MenuToggle::NAME, ObjectToggle::NAME, MergeAnimator::NAME, MergeArmature::NAME, AvatarMenuItem::NAME, MenuInstaller::NAME]
}

/// Roots of the avatars in `scene`: parentless entries with an Avatar Descriptor.
pub fn avatar_roots(scene: &SceneFile) -> Vec<usize> {
    (0..scene.entries.len())
        .filter(|&i| scene.entries[i].parent.is_none() && scene.entries[i].components.iter().any(|c| c.kind == AvatarDescriptor::NAME))
        .collect()
}

/// Whether the avatar below `root` has anything for the build to do.
pub fn needs_build(scene: &SceneFile, root: usize) -> bool {
    let sub = subtree(scene, root);
    sub.iter().any(|&i| {
        scene.entries[i].components.iter().any(|c| authoring_kinds().contains(&c.kind.as_str()) || (i != root && c.kind == AvatarParameters::NAME))
    })
}

/// Build the avatar below `root` in place. Pass a clone of the authored scene.
pub fn build_avatar(scene: &mut SceneFile, root: usize, env: &BuildEnv) -> AvatarBuild {
    let mut b = Builder::new(scene, root, env);
    if root >= b.scene.entries.len() {
        b.out.errors.push("the avatar root does not exist".into());
        return b.out;
    }
    b.out.input_hash = input_hash(b.scene, root, env);
    macro_rules! run {
        ($($pass:ident),* $(,)?) => {$(
            b.$pass();
            b.out.passes.push(stringify!($pass));
        )*};
    }
    run!(
        resolve,
        scope_parameters,
        assign_menu_parameters,
        generate_toggles,
        generate_reactive,
        exclusive_tags,
        optimise_toggles,
        merge_animators,
        resting_state,
        merge_armature,
        defaults_layer,
        flatten_controller,
        finalize_menu,
        lints,
        budget,
        emit,
    );
    b.out.root = b.root;
    b.out
}

/// [`build_avatar`] on a copy: the authored scene stays as it is.
pub fn build_avatar_cloned(scene: &SceneFile, root: usize, env: &BuildEnv) -> (SceneFile, AvatarBuild) {
    let mut copy = scene.clone();
    let out = build_avatar(&mut copy, root, env);
    (copy, out)
}

// ------------------------------------------------------------------- helpers

/// Entries below `root` (inclusive) in hierarchy order: depth first, children
/// in file order.
pub fn subtree(scene: &SceneFile, root: usize) -> Vec<usize> {
    let mut kids: Vec<Vec<usize>> = vec![Vec::new(); scene.entries.len()];
    for (i, e) in scene.entries.iter().enumerate() {
        if let Some(p) = e.parent.filter(|&p| p < kids.len() && p != i) {
            kids[p].push(i);
        }
    }
    let mut out = Vec::new();
    let mut stack = vec![root];
    let mut seen = vec![false; scene.entries.len()];
    while let Some(n) = stack.pop() {
        if n >= seen.len() || std::mem::replace(&mut seen[n], true) {
            continue;
        }
        out.push(n);
        stack.extend(kids[n].iter().rev());
    }
    out
}

/// Hex digits of the hash of `s`, `n` of them.
fn short_hash(s: &str, n: usize) -> String {
    sha256_hex(s.as_bytes())[..n].to_owned()
}

/// File-name-safe version of `s` (letters, digits, `_`, `-`).
fn sanitise(s: &str) -> String {
    let t: String = s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' }).collect();
    let t = t.trim_matches('_');
    if t.is_empty() { "x".to_owned() } else { t.chars().take(40).collect() }
}

fn comps<T: TypedComponent>(e: &SceneEntry) -> Vec<T> {
    e.components.iter().filter(|c| c.kind == T::NAME && c.enabled).filter_map(|c| ron::from_str::<T>(&c.data).ok()).collect()
}

fn set_comp<T: TypedComponent>(e: &mut SceneEntry, v: &T) {
    let data = ron::to_string(v).unwrap_or_default();
    match e.components.iter_mut().find(|c| c.kind == T::NAME) {
        Some(c) => {
            c.data = data;
            c.enabled = true;
        }
        None => e.components.push(ComponentData::new(T::NAME, data)),
    }
}

fn clip_bytes(c: &AnimClip) -> Vec<u8> {
    ron::ser::to_string_pretty(c, ron::ser::PrettyConfig::default()).unwrap_or_default().into_bytes()
}

fn is_clip_path(s: &str) -> bool {
    !s.is_empty() && (s.starts_with('@') || s.to_ascii_lowercase().ends_with(".anim"))
}

/// One animated property of one object, for the defaults layer. `target` is
/// relative to the avatar's Animator.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct PropKey {
    target: String,
    property: u8,
    name: String,
    slot: u8,
}

fn property_of(id: u8) -> TrackProperty {
    const ALL: [TrackProperty; 10] = [
        TrackProperty::Translation,
        TrackProperty::Rotation,
        TrackProperty::Scale,
        TrackProperty::LightIntensity,
        TrackProperty::Active,
        TrackProperty::ComponentEnabled,
        TrackProperty::BlendShape,
        TrackProperty::MaterialColor,
        TrackProperty::MaterialFloat,
        TrackProperty::MaterialOffset,
    ];
    ALL[id as usize % ALL.len()]
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Cond {
    /// The parameter is true / non-zero.
    Truthy,
    Equals(f32),
    Above(f32),
}

impl Cond {
    fn on(self, p: &str) -> AnimCondition {
        match self {
            Cond::Truthy => AnimCondition { parameter: p.into(), op: CondOp::If, threshold: 0.0 },
            Cond::Equals(v) => AnimCondition { parameter: p.into(), op: CondOp::Equals, threshold: v },
            Cond::Above(t) => AnimCondition { parameter: p.into(), op: CondOp::Greater, threshold: t },
        }
    }
    fn off(self, p: &str) -> AnimCondition {
        match self {
            Cond::Truthy => AnimCondition { parameter: p.into(), op: CondOp::IfNot, threshold: 0.0 },
            Cond::Equals(v) => AnimCondition { parameter: p.into(), op: CondOp::NotEquals, threshold: v },
            Cond::Above(t) => AnimCondition { parameter: p.into(), op: CondOp::Less, threshold: t },
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SpecKind {
    Toggle,
    Reactive,
}

/// One toggle-like thing to build a layer for (a `Menu Toggle` or an
/// `Object Toggle` bound to a menu item).
struct Spec {
    kind: SpecKind,
    entry: usize,
    /// Layer name suffix and clip stem source.
    label: String,
    stem: String,
    actions: Vec<ToggleAction>,
    param: String,
    cond: Cond,
    mode: ToggleMode,
    default_on: bool,
    invert_rest: bool,
    drive_globals: Vec<String>,
    tags: Vec<String>,
    exclusive_off: bool,
    /// Only runs on the local player's avatar (gated on `IsLocal`).
    local: bool,
    fade_in: f32,
    fade_out: f32,
}

struct ScopeRemap {
    at: usize,
    from: String,
    to: String,
}

struct MergedSet {
    layer: AvatarLayer,
    layers: Vec<AnimLayer>,
}

struct Builder<'a> {
    scene: &'a mut SceneFile,
    env: &'a BuildEnv<'a>,
    root: usize,
    out_dir: String,
    /// Entries below the root in hierarchy order.
    sub: Vec<usize>,
    /// Path below the root per entry index.
    paths: Vec<String>,
    animator: Option<usize>,
    anim_prefix: String,
    base: AnimatorController,
    had_base: bool,
    replace_base: bool,
    params: Vec<AvatarParam>,
    generated_params: Vec<String>,
    remaps: Vec<ScopeRemap>,
    auto_params: BTreeMap<usize, String>,
    specs: Vec<Spec>,
    /// Menu entries the toggles generated.
    menu: AvatarMenu,
    gen_layers: Vec<(AnimLayer, String)>,
    /// Spec index -> index in `gen_layers` of the layer made for it.
    spec_layers: Vec<(usize, usize)>,
    merged: Vec<MergedSet>,
    ctrl_params: Vec<AnimParameter>,
    clips: BTreeMap<String, AnimClip>,
    keys: BTreeMap<PropKey, Option<[f32; 4]>>,
    defaults_layer: Option<AnimLayer>,
    root_budget: u32,
    controller: Option<AnimatorController>,
    renames: Vec<(String, String)>,
    files: BTreeMap<String, Vec<u8>>,
    out: AvatarBuild,
}

impl<'a> Builder<'a> {
    fn new(scene: &'a mut SceneFile, root: usize, env: &'a BuildEnv<'a>) -> Self {
        let id = scene
            .entries
            .get(root)
            .map(|e| if e.id.is_empty() { format!("avatar-{}", short_hash(&e.name, 12)) } else { sanitise(&e.id) })
            .unwrap_or_default();
        let out_dir = format!("{GENERATED_DIR}/{id}");
        Builder {
            scene,
            env,
            root,
            out_dir: out_dir.clone(),
            sub: Vec::new(),
            paths: Vec::new(),
            animator: None,
            anim_prefix: String::new(),
            base: AnimatorController { parameters: Vec::new(), layers: Vec::new() },
            had_base: false,
            replace_base: false,
            params: Vec::new(),
            generated_params: Vec::new(),
            remaps: Vec::new(),
            auto_params: BTreeMap::new(),
            specs: Vec::new(),
            menu: AvatarMenu::default(),
            gen_layers: Vec::new(),
            spec_layers: Vec::new(),
            merged: Vec::new(),
            ctrl_params: Vec::new(),
            clips: BTreeMap::new(),
            keys: BTreeMap::new(),
            defaults_layer: None,
            root_budget: zestengine_core::avatar_params::DEFAULT_PARAM_BUDGET_BITS,
            controller: None,
            renames: Vec::new(),
            files: BTreeMap::new(),
            out: AvatarBuild { out_dir, root, ..Default::default() },
        }
    }

    fn warn(&mut self, s: impl Into<String>) {
        self.out.warnings.push(s.into());
    }
    fn error(&mut self, s: impl Into<String>) {
        self.out.errors.push(s.into());
    }

    fn name_of(&self, i: usize) -> &str {
        &self.scene.entries[i].name
    }

    /// Recompute `sub` and `paths` (after objects moved).
    fn refresh_paths(&mut self) {
        self.sub = subtree(self.scene, self.root);
        self.paths = vec![String::new(); self.scene.entries.len()];
        for &i in &self.sub {
            if i == self.root {
                continue;
            }
            let parent = self.scene.entries[i].parent.unwrap_or(self.root);
            let pp = &self.paths[parent];
            self.paths[i] = if pp.is_empty() { self.scene.entries[i].name.clone() } else { format!("{pp}/{}", self.scene.entries[i].name) };
        }
    }

    /// The entry at `path` below the root.
    fn resolve_path(&self, path: &str) -> Option<usize> {
        if path.is_empty() {
            return Some(self.root);
        }
        self.sub.iter().copied().find(|&i| i != self.root && self.paths[i] == path)
    }

    /// Root-relative `path` as the Animator addresses it.
    fn to_anim(&self, path: &str) -> Option<String> {
        if self.anim_prefix.is_empty() {
            return Some(path.to_owned());
        }
        if path == self.anim_prefix {
            return Some(String::new());
        }
        path.strip_prefix(&format!("{}/", self.anim_prefix)).map(str::to_owned)
    }

    /// Animator-relative `target` as a root-relative path.
    #[allow(clippy::wrong_self_convention)]
    fn from_anim(&self, target: &str) -> String {
        match (self.anim_prefix.is_empty(), target.is_empty()) {
            (true, _) => target.to_owned(),
            (false, true) => self.anim_prefix.clone(),
            (false, false) => format!("{}/{target}", self.anim_prefix),
        }
    }

    /// A parameter name after the scopes in effect at `entry`, innermost first.
    fn scoped(&self, entry: usize, name: &str) -> String {
        let mut chain: Vec<&ScopeRemap> = self.remaps.iter().filter(|r| self.is_within(entry, r.at)).collect();
        // Deepest declaring object first.
        chain.sort_by_key(|r| std::cmp::Reverse(self.paths.get(r.at).map_or(0, |p| p.len())));
        let mut n = name.to_owned();
        for r in chain {
            if r.from == n {
                n = r.to.clone();
            }
        }
        n
    }

    /// `entry` is `ancestor` or below it.
    fn is_within(&self, entry: usize, ancestor: usize) -> bool {
        let mut k = Some(entry);
        let mut guard = 0;
        while let Some(i) = k {
            if i == ancestor {
                return true;
            }
            guard += 1;
            if guard > 256 {
                return false;
            }
            k = self.scene.entries.get(i).and_then(|e| e.parent);
        }
        false
    }

    fn declared(&self, name: &str) -> Option<&AvatarParam> {
        self.params.iter().find(|p| p.name == name)
    }

    /// Add a parameter, or merge into the one of that name: a different kind
    /// is an error, a different default a warning (the last one wins).
    fn add_param(&mut self, p: AvatarParam, generated: bool, who: &str) {
        if p.name.is_empty() || BuiltinParam::from_name(&p.name).is_some() {
            return;
        }
        match self.params.iter().position(|q| q.name == p.name) {
            Some(i) => {
                let q = &mut self.params[i];
                if q.kind != p.kind {
                    let msg = format!("{who}: parameter '{}' is {} here but {} elsewhere", p.name, p.kind.label(), q.kind.label());
                    self.out.errors.push(msg);
                    return;
                }
                if (q.default - p.default).abs() > 1e-6 {
                    let msg = format!("{who}: parameter '{}' has default {} here and {} elsewhere; the last one wins", p.name, p.default, q.default);
                    self.out.warnings.push(msg);
                    q.default = p.default;
                }
                q.saved |= p.saved;
                if p.is_synced() {
                    q.sync = AvatarParamSync::Synced;
                }
            }
            None => {
                if generated {
                    self.generated_params.push(p.name.clone());
                }
                self.params.push(AvatarParam { scope: AvatarParamScope::Global, ..p });
            }
        }
    }

    fn kind_of(&self, name: &str) -> AvatarParamKind {
        BuiltinParam::from_name(name).map(BuiltinParam::kind).or_else(|| self.declared(name).map(|p| p.kind)).unwrap_or(AvatarParamKind::Float)
    }

    // ------------------------------------------------------------- 1 resolve

    pub(super) fn resolve(&mut self) {
        self.refresh_paths();
        // The Animator on the root, else the first one below it.
        self.animator =
            self.sub.iter().copied().find(|&i| i == self.root && self.scene.entries[i].components.iter().any(|c| c.kind == Animator::NAME)).or_else(|| {
                self.sub.iter().copied().find(|&i| self.scene.entries[i].components.iter().any(|c| c.kind == Animator::NAME))
            });
        if let Some(a) = self.animator {
            self.anim_prefix = self.paths[a].clone();
            if let Some(anim) = comps::<Animator>(&self.scene.entries[a]).into_iter().next()
                && !anim.controller.is_empty()
            {
                let path = zestengine_core::asset_ref::resolve_path(self.env.project_root, &anim.controller);
                match crate::load_anim_controller(&path) {
                    Ok(c) => {
                        self.base = c;
                        self.had_base = true;
                    }
                    Err(e) => self.warn(format!("the avatar's controller {} could not be read ({e:#}); building without it", anim.controller)),
                }
            }
        }
    }

    // ----------------------------------------------------- 2 scope parameters

    pub(super) fn scope_parameters(&mut self) {
        let sub = self.sub.clone();
        let mut root_budget = None;
        for &i in &sub {
            let list = comps::<AvatarParameters>(&self.scene.entries[i]);
            for ap in list {
                let who = format!("{} (Avatar Parameters)", if self.paths[i].is_empty() { self.name_of(i).to_owned() } else { self.paths[i].clone() });
                if i == self.root {
                    root_budget = Some(ap.budget_bits);
                }
                for p in ap.params {
                    if p.name.is_empty() {
                        continue;
                    }
                    let final_name = if i == self.root {
                        p.name.clone()
                    } else {
                        match &p.scope {
                            AvatarParamScope::Global => p.name.clone(),
                            AvatarParamScope::Internal => format!("{}${}", p.name, short_hash(&self.paths[i], 12)),
                            AvatarParamScope::RenameTo(n) if !n.is_empty() => n.clone(),
                            AvatarParamScope::RenameTo(_) => p.name.clone(),
                        }
                    };
                    if final_name != p.name {
                        self.remaps.push(ScopeRemap { at: i, from: p.name.clone(), to: final_name.clone() });
                    }
                    self.add_param(AvatarParam { name: final_name, ..p }, false, &who);
                }
            }
        }
        if let Some(b) = root_budget {
            self.root_budget = b;
        }
    }
}

/// Hash of everything a build of the avatar below `root` reads: its entries
/// (as stored), the files its components point at, and the limits. Equal
/// hashes mean the same outputs, so a caller may skip writing them.
pub fn input_hash(scene: &SceneFile, root: usize, env: &BuildEnv) -> String {
    let mut text = String::new();
    let mut files: BTreeSet<String> = BTreeSet::new();
    for i in subtree(scene, root) {
        let e = &scene.entries[i];
        text.push_str(&format!("{}|{}|{}|{:?}|{:?}|{:?}|{}\n", e.name, e.parent.map_or(0, |p| p + 1), e.enabled, e.translation, e.rotation, e.scale, i));
        for c in &e.components {
            text.push_str(&format!("{}|{}|{}\n", c.kind, c.enabled, c.data));
            // Any quoted project path in the data that names an asset we read.
            for part in c.data.split('"') {
                let l = part.to_ascii_lowercase();
                if is_clip_path(part) || [".animctrl", ".avmenu", ".avparams", ".png", ".jpg", ".jpeg"].iter().any(|x| l.ends_with(x)) {
                    files.insert(part.to_owned());
                }
            }
        }
    }
    text.push_str(&format!("cap={}|structural={}\n", env.budget_cap, env.structural));
    for f in files {
        let path = zestengine_core::asset_ref::resolve_path(env.project_root, &f);
        let stamp = std::fs::read(&path).map(|b| sha256_hex(&b)).unwrap_or_else(|_| "missing".into());
        text.push_str(&format!("file {f}={stamp}\n"));
    }
    sha256_hex(text.as_bytes())
}
