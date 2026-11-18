use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use egui_dock::{DockState, Node, NodeIndex, Surface, TabIndex, Tree};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

pub const LAYOUT_FILE: &str = "layout.ron";
const SAVE_INTERVAL: Duration = Duration::from_secs(2);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SavedNode<T> {
    Leaf { tabs: Vec<T>, active: usize },
    Horizontal { fraction: f32, first: Box<SavedNode<T>>, second: Box<SavedNode<T>> },
    Vertical { fraction: f32, first: Box<SavedNode<T>>, second: Box<SavedNode<T>> },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SavedWindow<T> {
    pub rect: Option<[f32; 4]>,
    pub tree: SavedNode<T>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SavedLayout<T> {
    pub main: SavedNode<T>,
    pub windows: Vec<SavedWindow<T>>,
}

pub fn default_path() -> Option<PathBuf> {
    if std::env::var_os("ZESTENGINE_NO_LAYOUT").is_some() || std::env::var_os("ZESTENGINE_SHOT").is_some() {
        return None;
    }
    zestengine_engine::input_engine::user_config_dir().map(|d| d.join("ZestEngine").join(LAYOUT_FILE))
}

pub fn panel_windows_path() -> Option<PathBuf> {
    default_path().map(|layout| layout.with_file_name("panel_windows.txt"))
}

pub fn read_panel_windows(path: Option<&std::path::Path>) -> Vec<String> {
    path.and_then(|path| std::fs::read_to_string(path).ok())
        .map(|text| text.lines().map(str::trim).filter(|line| !line.is_empty()).map(str::to_owned).collect())
        .unwrap_or_default()
}

pub fn write_panel_windows(path: Option<&std::path::Path>, names: &[String]) {
    let Some(path) = path else { return };
    if names.is_empty() {
        let _ = std::fs::remove_file(path);
        return;
    }
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Err(e) = std::fs::write(path, names.join("\n") + "\n") {
        tracing::warn!("saving the panel window list: {e}");
    }
}

#[cfg(test)]
mod panel_window_list_tests {
    #[test]
    fn the_panel_window_list_round_trips_and_an_empty_list_removes_the_file() {
        let dir = std::env::temp_dir().join(format!("zestengine-panel-list-{}", std::process::id()));
        let path = dir.join("panel_windows.txt");
        super::write_panel_windows(Some(&path), &["Inspector".to_owned(), "Viewport".to_owned()]);
        assert_eq!(super::read_panel_windows(Some(&path)), ["Inspector", "Viewport"]);
        super::write_panel_windows(Some(&path), &[]);
        assert!(!path.exists());
        assert!(super::read_panel_windows(Some(&path)).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}

fn capture_node<T: Clone>(tree: &Tree<T>, at: NodeIndex, keep: &impl Fn(&T) -> bool) -> Option<SavedNode<T>> {
    if at.0 >= tree.len() {
        return None;
    }
    let pair = |fraction: f32, horizontal: bool| {
        let fraction = if fraction.is_finite() { fraction.clamp(0.02, 0.98) } else { 0.5 };
        match (capture_node(tree, at.left(), keep), capture_node(tree, at.right(), keep)) {
            (Some(first), Some(second)) if horizontal => Some(SavedNode::Horizontal { fraction, first: Box::new(first), second: Box::new(second) }),
            (Some(first), Some(second)) => Some(SavedNode::Vertical { fraction, first: Box::new(first), second: Box::new(second) }),
            (Some(only), None) | (None, Some(only)) => Some(only),
            (None, None) => None,
        }
    };
    match &tree[at] {
        Node::Empty => None,
        Node::Leaf(leaf) => {
            let active = leaf.tabs.get(leaf.active.0);
            let tabs: Vec<T> = leaf.tabs.iter().filter(|tab| keep(tab)).cloned().collect();
            if tabs.is_empty() {
                return None;
            }
            let active = active.filter(|tab| keep(tab)).map_or(0, |_| leaf.tabs[..leaf.active.0].iter().filter(|tab| keep(tab)).count());
            Some(SavedNode::Leaf { tabs, active })
        }
        Node::Horizontal(split) => pair(split.fraction, true),
        Node::Vertical(split) => pair(split.fraction, false),
    }
}

pub fn capture<T: Clone>(state: &DockState<T>, keep: impl Fn(&T) -> bool) -> Option<SavedLayout<T>> {
    let main = capture_node(state.main_surface(), NodeIndex::root(), &keep)?;
    let windows = state
        .iter_surfaces()
        .filter_map(|surface| match surface {
            Surface::Window(tree, window) => {
                let rect = window.rect();
                let rect = (rect.is_finite() && rect.width() > 1.0 && rect.height() > 1.0).then(|| [rect.min.x, rect.min.y, rect.width(), rect.height()]);
                capture_node(tree, NodeIndex::root(), &keep).map(|tree| SavedWindow { rect, tree })
            }
            _ => None,
        })
        .collect();
    Some(SavedLayout { main, windows })
}

fn first_tabs<T: Clone>(node: &SavedNode<T>) -> Vec<T> {
    match node {
        SavedNode::Leaf { tabs, .. } => tabs.clone(),
        SavedNode::Horizontal { first, .. } | SavedNode::Vertical { first, .. } => first_tabs(first),
    }
}

fn rebuild<T: Clone>(tree: &mut Tree<T>, node: &SavedNode<T>, at: NodeIndex, depth: usize) {
    match node {
        SavedNode::Leaf { active, .. } => {
            if let Node::Leaf(leaf) = &mut tree[at] {
                leaf.active = TabIndex((*active).min(leaf.tabs.len().saturating_sub(1)));
            }
        }
        SavedNode::Horizontal { fraction, first, second } | SavedNode::Vertical { fraction, first, second } => {
            if depth >= 16 {
                return;
            }
            let fraction = if fraction.is_finite() { fraction.clamp(0.02, 0.98) } else { 0.5 };
            let [a, b] = if matches!(node, SavedNode::Horizontal { .. }) {
                tree.split_right(at, fraction, first_tabs(second))
            } else {
                tree.split_below(at, fraction, first_tabs(second))
            };
            rebuild(tree, first, a, depth + 1);
            rebuild(tree, second, b, depth + 1);
        }
    }
}

fn well_formed<T>(node: &SavedNode<T>) -> bool {
    match node {
        SavedNode::Leaf { tabs, .. } => !tabs.is_empty(),
        SavedNode::Horizontal { first, second, .. } | SavedNode::Vertical { first, second, .. } => well_formed(first) && well_formed(second),
    }
}

pub fn restore<T: Clone>(saved: &SavedLayout<T>) -> Option<DockState<T>> {
    if !well_formed(&saved.main) || saved.windows.iter().any(|w| !well_formed(&w.tree)) {
        return None;
    }
    let mut state = DockState::new(first_tabs(&saved.main));
    rebuild(state.main_surface_mut(), &saved.main, NodeIndex::root(), 0);
    for window in &saved.windows {
        let surface = state.add_window(first_tabs(&window.tree));
        rebuild(&mut state[surface], &window.tree, NodeIndex::root(), 0);
        if let (Some([x, y, w, h]), Some(window_state)) = (window.rect, state.get_window_state_mut(surface)) {
            window_state.set_position(egui::pos2(x, y)).set_size(egui::vec2(w, h));
        }
    }
    Some(state)
}

pub fn encode<T: Serialize + Clone>(state: &DockState<T>, keep: impl Fn(&T) -> bool) -> Option<String> {
    ron::ser::to_string(&capture(state, keep)?).ok()
}

pub fn decode<T: DeserializeOwned + Clone>(text: &str, valid: impl Fn(&DockState<T>) -> bool) -> Option<DockState<T>> {
    let saved: SavedLayout<T> = ron::from_str(text).ok()?;
    restore(&saved).filter(|state| valid(state))
}

pub fn load<T: DeserializeOwned + Clone>(path: &Path, valid: impl Fn(&DockState<T>) -> bool) -> Option<DockState<T>> {
    let text = std::fs::read_to_string(path).ok()?;
    let state = decode(&text, valid);
    if state.is_none() {
        tracing::warn!("editor layout: ignoring {} (unreadable or missing the main panels)", path.display());
    }
    state
}

pub struct LayoutSaver {
    path: Option<PathBuf>,
    saved: Option<String>,
    checked: Instant,
}

impl LayoutSaver {
    pub fn new(path: Option<PathBuf>) -> Self {
        let saved = path.as_deref().and_then(|p| std::fs::read_to_string(p).ok());
        Self { path, saved, checked: Instant::now() }
    }

    pub fn due(&self) -> bool {
        self.checked.elapsed() >= SAVE_INTERVAL
    }

    pub fn tick<T: Serialize + Clone>(&mut self, state: &DockState<T>, keep: impl Fn(&T) -> bool) -> bool {
        if self.checked.elapsed() < SAVE_INTERVAL {
            return false;
        }
        self.save_now(state, keep)
    }

    pub fn save_now<T: Serialize + Clone>(&mut self, state: &DockState<T>, keep: impl Fn(&T) -> bool) -> bool {
        self.checked = Instant::now();
        let (Some(path), Some(text)) = (self.path.as_deref(), encode(state, keep)) else {
            return false;
        };
        if self.saved.as_deref() == Some(text.as_str()) {
            return false;
        }
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let tmp = path.with_extension("ron.tmp");
        match std::fs::write(&tmp, &text).and_then(|()| std::fs::rename(&tmp, path)) {
            Ok(()) => {
                self.saved = Some(text);
                true
            }
            Err(e) => {
                tracing::warn!("editor layout not saved to {}: {e}", path.display());
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    enum Tab {
        Viewport,
        Camera,
        Scene,
        Inspector,
        Log,
        Files,
        Code(String),
    }

    fn layout() -> DockState<Tab> {
        let mut state = DockState::new(vec![Tab::Viewport, Tab::Camera, Tab::Code("src/main.rs".into())]);
        let tree = state.main_surface_mut();
        let [viewport, _] = tree.split_right(NodeIndex::root(), 0.75, vec![Tab::Inspector]);
        let [viewport, _] = tree.split_below(viewport, 0.7, vec![Tab::Log, Tab::Code("notes.md".into())]);
        tree.split_left(viewport, 0.2, vec![Tab::Scene]);
        let window = state.add_window(vec![Tab::Files]);
        state.get_window_state_mut(window).unwrap().set_position(egui::pos2(40.0, 60.0)).set_size(egui::vec2(480.0, 320.0));
        state
    }

    fn shape(state: &DockState<Tab>) -> Option<SavedLayout<Tab>> {
        capture(state, |_| true)
    }

    fn has_viewport(state: &DockState<Tab>) -> bool {
        state.iter_all_tabs().any(|(_, tab)| *tab == Tab::Viewport)
    }

    #[test]
    fn splits_fractions_and_the_floating_window_survive_a_save_and_load() {
        let state = layout();
        let keep = |tab: &Tab| !matches!(tab, Tab::Code(_));
        let text = encode(&state, keep).unwrap();
        let back = decode::<Tab>(&text, has_viewport).expect("a saved layout loads");
        let saved = capture(&state, keep).unwrap();
        let mut again = shape(&back).unwrap();
        assert_eq!(again.main, saved.main, "the same tree of splits, fractions and tab groups");
        assert_eq!(again.windows.len(), 1);
        again.windows[0].rect = saved.windows[0].rect;
        assert_eq!(again.windows[0].tree, SavedNode::Leaf { tabs: vec![Tab::Files], active: 0 });
        assert!(!back.iter_all_tabs().any(|(_, tab)| matches!(tab, Tab::Code(_))), "document tabs are not part of the layout");
        let SavedNode::Horizontal { fraction, second, .. } = &saved.main else { panic!("the inspector split is the root: {:?}", saved.main) };
        assert!((fraction - 0.75).abs() < 1e-6);
        assert_eq!(**second, SavedNode::Leaf { tabs: vec![Tab::Inspector], active: 0 });
    }

    #[test]
    fn a_leaf_that_only_held_document_tabs_disappears_and_the_active_tab_follows() {
        let mut state = DockState::new(vec![Tab::Code("a.rs".into()), Tab::Viewport, Tab::Camera]);
        state.main_surface_mut().split_right(NodeIndex::root(), 0.5, vec![Tab::Code("b.rs".into())]);
        if let Node::Leaf(leaf) = &mut state.main_surface_mut()[NodeIndex::root().left()] {
            leaf.active = TabIndex(2);
        }
        let saved = capture(&state, |tab| !matches!(tab, Tab::Code(_))).unwrap();
        assert_eq!(saved.main, SavedNode::Leaf { tabs: vec![Tab::Viewport, Tab::Camera], active: 1 }, "the emptied split collapses into the remaining leaf");
        let back = restore(&saved).unwrap();
        let Some(Node::Leaf(leaf)) = back.main_surface().root_node() else { panic!() };
        assert_eq!(leaf.active, TabIndex(1));
        assert!(capture(&DockState::new(vec![Tab::Code("x".into())]), |tab| !matches!(tab, Tab::Code(_))).is_none());
    }

    #[test]
    fn a_broken_or_incomplete_layout_is_refused() {
        assert!(decode::<Tab>("not a layout", has_viewport).is_none());
        let only_log = encode(&DockState::new(vec![Tab::Log]), |_| true).unwrap();
        assert!(decode::<Tab>(&only_log, has_viewport).is_none(), "a layout without the viewport falls back to the default");
        let empty_leaf = "(main:Leaf(tabs:[],active:0),windows:[])";
        assert!(decode::<Tab>(empty_leaf, |_| true).is_none());
        let wild = "(main:Horizontal(fraction:NaN,first:Leaf(tabs:[Viewport],active:9),second:Leaf(tabs:[Log],active:0)),windows:[])";
        let state = decode::<Tab>(wild, has_viewport).expect("out-of-range numbers are clamped");
        assert_eq!(state.iter_all_tabs().count(), 2);
    }

    #[test]
    fn the_saver_writes_only_when_the_layout_changed() {
        let dir = std::env::temp_dir().join(format!("zest-layout-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("nested").join(LAYOUT_FILE);
        let mut saver = LayoutSaver::new(Some(path.clone()));
        let mut state = layout();
        assert!(!saver.tick(&state, |_| true), "not before the interval");
        assert!(saver.save_now(&state, |_| true));
        assert!(!saver.save_now(&state, |_| true), "unchanged");
        state.main_surface_mut().split_below(NodeIndex::root(), 0.5, vec![Tab::Log]);
        assert!(saver.save_now(&state, |_| true));
        assert_eq!(load::<Tab>(&path, has_viewport).unwrap().iter_all_tabs().count(), 9);
        let mut reopened = LayoutSaver::new(Some(path.clone()));
        assert!(!reopened.save_now(&state, |_| true), "a restart does not rewrite an identical file");
        assert!(!LayoutSaver::new(None).save_now(&state, |_| true));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
