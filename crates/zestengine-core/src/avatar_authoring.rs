//! Avatar authoring components and the finished expression menu.
//!
//! These components describe intent, not mechanism: a [`MenuToggle`] says "this
//! menu entry shows or hides these things"; the avatar build pass
//! (`zestengine_assets::avatar_build`) expands it into a parameter, a menu
//! entry, an animation clip and a controller layer, on a clone of the avatar,
//! and strips the authoring components from the result. Nothing here runs per
//! frame.
//!
//! Targets of actions are object paths below the avatar root ("" = the root
//! itself, "Body/Hat" = a descendant), the same addressing animation clips
//! use, so they survive renames done through the editor's rename helpers and
//! travel with prefabs.
//!
//! [`AvatarMenu`] is the radial menu's data: a tree of [`AvatarMenuItem`]s
//! with at most [`MAX_MENU_CONTROLS`] entries per page. The build pass
//! assembles it from the hierarchy's `Avatar Menu Item`s and the generated
//! toggles and stores it on the avatar root as [`AvatarMenuData`].

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::avatar_params::{AvatarMenuItem, MAX_MENU_CONTROLS, MenuItemKind};
use crate::TypedComponent;

/// File extension of authored `.avmenu` menu assets.
pub const AVATAR_MENU_EXTENSION: &str = "avmenu";

// ------------------------------------------------------------------- toggles

/// One thing a toggle changes while it is on. `target` is a path below the
/// avatar root.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum ToggleAction {
    /// The object is active (`on`) or inactive while the toggle is on.
    Active {
        /// Object path below the avatar root ("" = the root).
        target: String,
        /// The object's active state while the toggle is on.
        on: bool,
    },
    /// A component (by type name, "Light") is switched on or off.
    ComponentEnabled {
        /// Object path below the avatar root.
        target: String,
        /// Component type name.
        component: String,
        /// The component's enabled state while the toggle is on.
        on: bool,
    },
    /// A blendshape weight on the object's skinned mesh.
    BlendShape {
        /// Object path below the avatar root.
        target: String,
        /// Blendshape name.
        shape: String,
        /// Weight while the toggle is on (0..1).
        value: f32,
    },
    /// A material scalar of a render slot (`MaterialProp` names, "metallic").
    /// `off` is the value while the toggle is off.
    MaterialFloat {
        /// Object path below the avatar root.
        target: String,
        /// Render slot.
        slot: u8,
        /// Material property name.
        prop: String,
        /// Value while the toggle is on.
        value: f32,
        /// Value while the toggle is off.
        off: f32,
    },
    /// A material colour of a render slot ("base_color", "emission_color").
    MaterialColor {
        /// Object path below the avatar root.
        target: String,
        /// Render slot.
        slot: u8,
        /// Material property name.
        prop: String,
        /// Colour while the toggle is on (linear RGBA).
        value: [f32; 4],
        /// Colour while the toggle is off (linear RGBA).
        off: [f32; 4],
    },
    /// Every track of an authored `.anim` clip plays while the toggle is on.
    Clip {
        /// Project-relative `.anim` path.
        path: String,
    },
    /// A parameter is set when the toggle turns on.
    SetParam {
        /// Parameter name.
        name: String,
        /// Value written while the toggle is on.
        value: f32,
    },
}

impl ToggleAction {
    /// Inspector label of this action.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Active { .. } => "Object",
            Self::ComponentEnabled { .. } => "Component",
            Self::BlendShape { .. } => "Blend Shape",
            Self::MaterialFloat { .. } => "Material Value",
            Self::MaterialColor { .. } => "Material Colour",
            Self::Clip { .. } => "Clip",
            Self::SetParam { .. } => "Set Parameter",
        }
    }
    /// The object path this action touches (`None` for clips and parameters).
    pub fn target(&self) -> Option<&str> {
        match self {
            Self::Active { target, .. }
            | Self::ComponentEnabled { target, .. }
            | Self::BlendShape { target, .. }
            | Self::MaterialFloat { target, .. }
            | Self::MaterialColor { target, .. } => Some(target),
            Self::Clip { .. } | Self::SetParam { .. } => None,
        }
    }
    /// Mutable access to the object path, when the action touches one.
    pub fn target_mut(&mut self) -> Option<&mut String> {
        match self {
            Self::Active { target, .. }
            | Self::ComponentEnabled { target, .. }
            | Self::BlendShape { target, .. }
            | Self::MaterialFloat { target, .. }
            | Self::MaterialColor { target, .. } => Some(target),
            Self::Clip { .. } | Self::SetParam { .. } => None,
        }
    }
}

/// How a `MenuToggle` behaves when used.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Default)]
pub enum ToggleMode {
    /// Click on, click off.
    #[default]
    Toggle,
    /// On while the menu button is held.
    Hold,
    /// A radial puppet (0..1) scrubbing the on clip: 0 is off, 1 fully on.
    /// With `inactive_at_zero` the actions only apply above 0; otherwise they
    /// always apply, with the slider as the amount.
    Slider {
        /// The slider's default position.
        default: f32,
        /// Actions only apply above 0.
        inactive_at_zero: bool,
    },
}

impl ToggleMode {
    /// True for the Slider mode.
    pub fn is_slider(self) -> bool {
        matches!(self, Self::Slider { .. })
    }
}

/// A menu entry that shows or hides things. Put it on any object under the
/// avatar root; one component builds the whole chain (see module docs).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct MenuToggle {
    /// Where it sits in the menu: slashes make sub menus (`Outfit/Hat`),
    /// `\/` is a literal slash. Empty uses the object's name.
    pub menu_path: String,
    /// Project-relative icon shown in the menu.
    pub icon: String,
    /// What the toggle changes while on.
    pub actions: Vec<ToggleAction>,
    /// How the toggle behaves when used.
    pub mode: ToggleMode,
    /// Remember the setting between sessions.
    pub saved: bool,
    /// Send the setting to other players (costs 1 bit, a slider 8).
    pub synced: bool,
    /// Starts on.
    pub default_on: bool,
    /// Toggles sharing a tag switch each other off.
    pub exclusive_tags: Vec<String>,
    /// With a tag: this toggle turns itself on when every toggle it shares a
    /// tag with is off (a default outfit that cannot be deselected).
    pub exclusive_off_state: bool,
    /// Use this existing parameter instead of a generated one (a built-in
    /// name makes the toggle local only).
    pub global_param: String,
    /// Bool parameters set true while on and false while off.
    pub drive_globals: Vec<String>,
    /// Count the actions' effect as the avatar's resting look when the toggle
    /// is off (default) or on.
    pub invert_rest: bool,
    /// Toggles with the same non-empty selector are the options of one Int
    /// parameter (a radio group): option `n` (1, 2, ... in hierarchy order)
    /// is on while the parameter equals `n`, 0 is none. One Int costs 8 synced
    /// bits however many options it has, and exclusivity is built in.
    pub selector: String,
    /// Seconds the on / off transitions cross-fade (0 = instant).
    pub transition_in: f32,
    /// Seconds the off transition cross-fades (0 = instant).
    pub transition_out: f32,
    /// Actions applied only on the local player's own avatar (hide your own
    /// head, show a first-person arm). They get their own layer, gated on
    /// `IsLocal`; remote copies never run it.
    pub local_actions: Vec<ToggleAction>,
}

impl Default for MenuToggle {
    fn default() -> Self {
        Self {
            menu_path: String::new(),
            icon: String::new(),
            actions: Vec::new(),
            mode: ToggleMode::Toggle,
            saved: true,
            synced: true,
            default_on: false,
            exclusive_tags: Vec::new(),
            exclusive_off_state: false,
            global_param: String::new(),
            drive_globals: Vec::new(),
            invert_rest: false,
            selector: String::new(),
            transition_in: 0.0,
            transition_out: 0.0,
            local_actions: Vec::new(),
        }
    }
}

impl TypedComponent for MenuToggle {
    const NAME: &'static str = "Menu Toggle";
}

impl MenuToggle {
    /// The menu path split into segments (escapes resolved), falling back to
    /// `fallback` (the object's name) when empty.
    pub fn path_segments(&self, fallback: &str) -> Vec<String> {
        let p = if self.menu_path.trim().is_empty() { fallback } else { self.menu_path.as_str() };
        split_menu_path(p)
    }
    /// The parameter kind this toggle needs.
    pub fn param_kind(&self) -> crate::AvatarParamKind {
        if self.mode.is_slider() { crate::AvatarParamKind::Float } else { crate::AvatarParamKind::Bool }
    }
    /// Synced bits this toggle adds when it declares its own parameter.
    pub fn synced_bit_cost(&self) -> u32 {
        if !self.synced || !self.global_param.is_empty() || !self.selector.trim().is_empty() {
            0
        } else {
            self.param_kind().bits()
        }
    }
}

/// Installs an authored `.avmenu` page into the avatar's menu without any
/// manual wiring: drop it (with the prefab that owns the menu) anywhere under
/// the avatar root and the page appears at `install_to`. Like Modular Avatar's
/// install target, an empty `install_to` is the menu root.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
#[serde(default)]
pub struct MenuInstaller {
    /// Project path of the `.avmenu` asset to install.
    pub menu: String,
    /// Where the page goes: a sub menu path (`Outfits/Hats`, created if
    /// missing). Empty = the root page.
    pub install_to: String,
}

impl TypedComponent for MenuInstaller {
    const NAME: &'static str = "Menu Installer";
}

/// Split a menu path at unescaped slashes. Empty segments are dropped.
pub fn split_menu_path(path: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut chars = path.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' if chars.peek() == Some(&'/') => {
                cur.push('/');
                chars.next();
            }
            '/' => {
                if !cur.trim().is_empty() {
                    out.push(cur.trim().to_owned());
                }
                cur.clear();
            }
            c => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_owned());
    }
    out
}

/// One object a reactive toggle switches.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct ObjectToggleEntry {
    /// Path below the avatar root.
    pub target: String,
    /// The object is active while the condition holds.
    pub active: bool,
}

impl Default for ObjectToggleEntry {
    fn default() -> Self {
        Self { target: String::new(), active: true }
    }
}

/// Reactive object switch for prefab authors: it follows the nearest
/// `Avatar Menu Item` above it (or on its own object), so dropping a prefab
/// with a menu item and this component under an avatar is the whole install.
/// A switched-off object (or component) never reacts.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
#[serde(default)]
pub struct ObjectToggle {
    /// The objects the toggle switches.
    pub objects: Vec<ObjectToggleEntry>,
    /// Reverse every entry while the menu item is on.
    pub inverted: bool,
}

impl TypedComponent for ObjectToggle {
    const NAME: &'static str = "Object Toggle";
}

// ------------------------------------------------------------------- merging

/// The playable layer a merged controller belongs to. Layers are laid into the
/// avatar's single controller in this order (later layers win).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum AvatarLayer {
    /// The avatar's own ground layer.
    Base,
    /// Extra motion on top of the base.
    Additive,
    /// Gesture-driven motion.
    Gesture,
    /// Action-driven motion.
    Action,
    /// Effects on top of everything.
    #[default]
    Fx,
}

impl AvatarLayer {
    /// Every layer, in lay order.
    pub const ALL: [AvatarLayer; 5] = [Self::Base, Self::Additive, Self::Gesture, Self::Action, Self::Fx];
    /// Inspector label of this layer.
    pub fn label(self) -> &'static str {
        match self {
            Self::Base => "Base",
            Self::Additive => "Additive",
            Self::Gesture => "Gesture",
            Self::Action => "Action",
            Self::Fx => "FX",
        }
    }
}

/// How a merged animator's clip paths are interpreted.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MergePathMode {
    /// Clip paths are relative to this object (a prop's own hierarchy).
    #[default]
    Relative,
    /// Clip paths are already relative to the avatar root.
    Absolute,
}

/// How a merged animator is combined with the avatar's.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MergeMode {
    /// Add the controller's layers after the avatar's.
    #[default]
    Append,
    /// Drop the avatar's own layers first (all merges of this mode run
    /// before any Append, lowest priority first).
    Replace,
}

/// Merge another animator controller into the avatar's. Its parameters are
/// renamed through the enclosing `Avatar Parameters` scopes, and clip paths
/// are rebased onto the avatar.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct MergeAnimator {
    /// Project-relative `.animctrl`.
    pub controller: String,
    /// Which playable layer the merged controller lands in.
    pub layer: AvatarLayer,
    /// How clip paths are interpreted.
    pub path_mode: MergePathMode,
    /// Lower priority merges earlier (then hierarchy order).
    pub priority: i32,
    /// Append after, or replace, the avatar's layers.
    pub mode: MergeMode,
    /// Give the merged states the avatar's write-defaults setting.
    pub match_write_defaults: bool,
}

impl Default for MergeAnimator {
    fn default() -> Self {
        Self {
            controller: String::new(),
            layer: AvatarLayer::Fx,
            path_mode: MergePathMode::Relative,
            priority: 0,
            mode: MergeMode::Append,
            match_write_defaults: true,
        }
    }
}

impl TypedComponent for MergeAnimator {
    const NAME: &'static str = "Merge Animator";
}

/// Merge this object's bone hierarchy into the avatar's: bones whose names
/// match (after the prefix and suffix are removed) become one bone, the rest
/// are re-parented with their world pose kept. Runs at package build only
/// (it moves objects); in the editor's Play the outfit stays separate.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct MergeArmature {
    /// Path below the avatar root of the bone to merge into; empty = the
    /// avatar's first "Armature" child, else the avatar root.
    pub target: String,
    /// Prefix stripped from bone names before matching.
    pub prefix: String,
    /// Suffix stripped from bone names before matching.
    pub suffix: String,
    /// Rename bones that find no partner when their name is already taken.
    pub mangle_names: bool,
}

impl Default for MergeArmature {
    fn default() -> Self {
        Self { target: String::new(), prefix: String::new(), suffix: String::new(), mangle_names: true }
    }
}

impl TypedComponent for MergeArmature {
    const NAME: &'static str = "Merge Armature";
}

/// Written by the avatar build on every outfit mesh a `Merge Armature` merged: at load the
/// runtime re-skins that mesh onto the skeleton of the avatar object `avatar` (bone indices
/// rebound by humanoid bone, then name; bind poses corrected; extra bones appended or folded),
/// and the avatar's Animator poses it. The object keeps its own active flag, so a toggle that
/// hides the outfit still hides it. Not an authoring component: it survives the build.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct SkinRebound {
    /// Id (the scene entry's id) of the avatar object whose skeleton the mesh is bound to.
    pub avatar: String,
    /// Name affixes of the outfit's bones, as on the `Merge Armature`.
    pub prefix: String,
    /// Suffix stripped from bone names before matching.
    pub suffix: String,
    /// Rename bones that find no partner when their name is already taken.
    pub mangle_names: bool,
    /// Give the weights of bones with no partner to their nearest matched ancestor instead of
    /// adding the bones to the avatar's skeleton.
    pub fold_extras: bool,
    /// Align the outfit to the avatar (hips offset and body height ratio).
    pub auto_align: bool,
}

impl Default for SkinRebound {
    fn default() -> Self {
        Self { avatar: String::new(), prefix: String::new(), suffix: String::new(), mangle_names: true, fold_extras: false, auto_align: false }
    }
}

impl TypedComponent for SkinRebound {
    const NAME: &'static str = "Skin Rebound";
}

// ---------------------------------------------------------------------- menu

/// One entry of the finished menu with its page (Sub Menu only).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
#[serde(default)]
pub struct AvatarMenuNode {
    /// The entry.
    pub item: AvatarMenuItem,
    /// Its page (Sub Menu only).
    pub children: Vec<AvatarMenuNode>,
}

impl AvatarMenuNode {
    /// A node with no children.
    pub fn new(item: AvatarMenuItem) -> Self {
        Self { item, children: Vec::new() }
    }
}

/// The finished expression menu: the root page and everything below it.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
#[serde(default)]
pub struct AvatarMenu {
    /// The root page's entries.
    pub root: Vec<AvatarMenuNode>,
}

/// Text normalised for matching menu labels (case, markup tags, whitespace).
pub fn normalise_label(s: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' if in_tag => in_tag = false,
            _ if in_tag => {}
            c if c.is_whitespace() => {}
            c => out.extend(c.to_lowercase()),
        }
    }
    out
}

impl AvatarMenu {
    /// True when the root page is empty.
    pub fn is_empty(&self) -> bool {
        self.root.is_empty()
    }

    /// The page reached by following `path` (child indices) from the root.
    pub fn page(&self, path: &[usize]) -> Option<&[AvatarMenuNode]> {
        let mut page: &[AvatarMenuNode] = &self.root;
        for &i in path {
            page = &page.get(i)?.children;
        }
        Some(page)
    }

    /// The page at the sub menu path `segments`, created on the way, as a
    /// mutable list. An empty path is the root page.
    pub fn page_at_mut(&mut self, segments: &[String]) -> &mut Vec<AvatarMenuNode> {
        let mut path: Vec<usize> = Vec::new();
        for d in segments {
            let want = normalise_label(d);
            let page = self.page_mut(&path).expect("path built from this tree");
            let at = match page.iter().position(|n| n.item.kind == MenuItemKind::SubMenu && normalise_label(&n.item.label) == want) {
                Some(i) => i,
                None => {
                    page.push(AvatarMenuNode::new(AvatarMenuItem { label: d.clone(), kind: MenuItemKind::SubMenu, parameter: String::new(), ..Default::default() }));
                    page.len() - 1
                }
            };
            path.push(at);
        }
        self.page_mut(&path).expect("path built from this tree")
    }

    fn page_mut(&mut self, path: &[usize]) -> Option<&mut Vec<AvatarMenuNode>> {
        let mut page = &mut self.root;
        for &i in path {
            page = &mut page.get_mut(i)?.children;
        }
        Some(page)
    }

    /// Add `item` at `segments` (sub menus created on the way, matched by
    /// normalised label); the last segment is the item's own label.
    pub fn insert_at(&mut self, segments: &[String], mut item: AvatarMenuItem) {
        let mut path: Vec<usize> = Vec::new();
        if let Some((leaf, dirs)) = segments.split_last() {
            for d in dirs {
                let want = normalise_label(d);
                let page = self.page_mut(&path).expect("path built from this tree");
                let at = match page.iter().position(|n| n.item.kind == MenuItemKind::SubMenu && normalise_label(&n.item.label) == want) {
                    Some(i) => i,
                    None => {
                        page.push(AvatarMenuNode::new(AvatarMenuItem {
                            label: d.clone(),
                            kind: MenuItemKind::SubMenu,
                            parameter: String::new(),
                            ..Default::default()
                        }));
                        page.len() - 1
                    }
                };
                path.push(at);
            }
            item.label = leaf.clone();
        }
        self.page_mut(&path).expect("path built from this tree").push(AvatarMenuNode::new(item));
    }

    /// Pages longer than [`MAX_MENU_CONTROLS`] keep their first seven entries
    /// and move the rest into a "More" sub menu, repeatedly. Empty sub menus
    /// (no children and no asset) are dropped.
    pub fn finalize(&mut self) {
        fn fix(page: &mut Vec<AvatarMenuNode>) {
            for n in page.iter_mut() {
                fix(&mut n.children);
            }
            page.retain(|n| n.item.kind != MenuItemKind::SubMenu || !n.children.is_empty() || !n.item.sub_menu_asset.is_empty());
            if page.len() > MAX_MENU_CONTROLS {
                let mut rest = page.split_off(MAX_MENU_CONTROLS - 1);
                fix_overflow(&mut rest);
                page.push(AvatarMenuNode {
                    item: AvatarMenuItem { label: "More".into(), kind: MenuItemKind::SubMenu, parameter: String::new(), ..Default::default() },
                    children: rest,
                });
            }
        }
        fn fix_overflow(page: &mut Vec<AvatarMenuNode>) {
            if page.len() > MAX_MENU_CONTROLS {
                let mut rest = page.split_off(MAX_MENU_CONTROLS - 1);
                fix_overflow(&mut rest);
                page.push(AvatarMenuNode {
                    item: AvatarMenuItem { label: "More".into(), kind: MenuItemKind::SubMenu, parameter: String::new(), ..Default::default() },
                    children: rest,
                });
            }
        }
        fix(&mut self.root);
    }

    /// Visit every node depth first with its page depth.
    pub fn visit(&self, f: &mut dyn FnMut(&AvatarMenuNode, usize)) {
        fn go(page: &[AvatarMenuNode], depth: usize, f: &mut dyn FnMut(&AvatarMenuNode, usize)) {
            for n in page {
                f(n, depth);
                go(&n.children, depth + 1, f);
            }
        }
        go(&self.root, 0, f);
    }

    /// Total nodes, pages included.
    pub fn count(&self) -> usize {
        let mut n = 0;
        self.visit(&mut |_, _| n += 1);
        n
    }

    /// Every parameter name the menu reads or writes (items and puppet axes).
    pub fn parameters(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        self.visit(&mut |n, _| {
            for p in std::iter::once(&n.item.parameter).chain(&n.item.sub_parameters) {
                if !p.is_empty() && !out.contains(p) {
                    out.push(p.clone());
                }
            }
        });
        out
    }

    /// Rename a parameter everywhere in the menu.
    pub fn rename_parameter(&mut self, old: &str, new: &str) {
        fn go(page: &mut [AvatarMenuNode], old: &str, new: &str) {
            for n in page {
                for p in std::iter::once(&mut n.item.parameter).chain(n.item.sub_parameters.iter_mut()) {
                    if p == old {
                        *p = new.to_owned();
                    }
                }
                go(&mut n.children, old, new);
            }
        }
        go(&mut self.root, old, new);
    }

    /// Problems with the finished menu: undeclared parameters, shape errors.
    pub fn problems(&self, known: &dyn Fn(&str) -> bool) -> Vec<String> {
        let mut out = Vec::new();
        self.visit(&mut |n, _| out.extend(n.item.problems(known)));
        out
    }

    /// Serialize to RON.
    pub fn to_ron(&self) -> String {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default()).unwrap_or_default()
    }
    /// Parse RON.
    pub fn from_ron(text: &str) -> Result<Self, String> {
        ron::from_str(text).map_err(|e| e.to_string())
    }
    /// Read and parse from `path`.
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
        Self::from_ron(&text)
    }
    /// Write to `path` (creating parent directories).
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, self.to_ron())
    }
}

/// The avatar's finished menu, written to the avatar root by the build pass.
/// The radial menu reads it; there is nothing to edit here.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
#[serde(default)]
pub struct AvatarMenuData {
    /// The menu the build stored on the avatar.
    pub menu: AvatarMenu,
}

impl TypedComponent for AvatarMenuData {
    const NAME: &'static str = "Avatar Menu";
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(label: &str) -> AvatarMenuItem {
        AvatarMenuItem { label: label.into(), parameter: label.into(), ..Default::default() }
    }

    #[test]
    fn menu_paths_split_at_unescaped_slashes() {
        assert_eq!(split_menu_path("Outfit/Hat"), vec!["Outfit", "Hat"]);
        assert_eq!(split_menu_path(r"AC\/DC/Song"), vec!["AC/DC", "Song"]);
        assert_eq!(split_menu_path("//a//"), vec!["a"]);
        assert!(split_menu_path("  ").is_empty());
    }

    #[test]
    fn insert_builds_sub_menus_and_reuses_them_case_insensitively() {
        let mut m = AvatarMenu::default();
        m.insert_at(&["Outfit".into(), "Hat".into()], item("x"));
        m.insert_at(&["outfit".into(), "Scarf".into()], item("y"));
        assert_eq!(m.root.len(), 1);
        assert_eq!(m.root[0].item.kind, MenuItemKind::SubMenu);
        assert_eq!(m.root[0].children.iter().map(|n| n.item.label.as_str()).collect::<Vec<_>>(), vec!["Hat", "Scarf"]);
    }

    #[test]
    fn long_pages_spill_into_more() {
        let mut m = AvatarMenu::default();
        for i in 0..20 {
            m.insert_at(&[format!("T{i}")], item(&format!("T{i}")));
        }
        m.finalize();
        assert_eq!(m.root.len(), MAX_MENU_CONTROLS);
        assert_eq!(m.root.last().unwrap().item.label, "More");
        let more = &m.root.last().unwrap().children;
        assert_eq!(more.len(), MAX_MENU_CONTROLS);
        assert_eq!(more.last().unwrap().item.label, "More");
        assert_eq!(m.count(), 20 + 2, "every toggle survives, plus the two More pages");
        let mut leaves = 0;
        m.visit(&mut |n, _| leaves += (n.item.kind != MenuItemKind::SubMenu) as usize);
        assert_eq!(leaves, 20);
    }

    #[test]
    fn empty_sub_menus_are_dropped_and_the_menu_round_trips() {
        let mut m = AvatarMenu::default();
        m.root.push(AvatarMenuNode::new(AvatarMenuItem { kind: MenuItemKind::SubMenu, label: "Empty".into(), ..Default::default() }));
        m.insert_at(&["A".into()], item("a"));
        m.finalize();
        assert_eq!(m.root.len(), 1);
        let back = AvatarMenu::from_ron(&m.to_ron()).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn old_and_partial_files_load() {
        let t: MenuToggle = ron::from_str("(menu_path: \"Hat\")").unwrap();
        assert!(t.saved && t.synced && !t.default_on);
        assert_eq!(t.mode, ToggleMode::Toggle);
        let m: MergeAnimator = ron::from_str("()").unwrap();
        assert_eq!(m.layer, AvatarLayer::Fx);
    }

    #[test]
    fn toggle_cost_counts_only_its_own_synced_parameter() {
        let mut t = MenuToggle::default();
        assert_eq!(t.synced_bit_cost(), 1);
        t.mode = ToggleMode::Slider { default: 0.0, inactive_at_zero: true };
        assert_eq!(t.synced_bit_cost(), 8);
        t.synced = false;
        assert_eq!(t.synced_bit_cost(), 0);
        t.synced = true;
        t.global_param = "Existing".into();
        assert_eq!(t.synced_bit_cost(), 0);
    }
}
