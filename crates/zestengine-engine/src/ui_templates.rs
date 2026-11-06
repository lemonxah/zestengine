//! Ready-made UI layouts for the editor's UI Editor (Main Menu, Pause Menu,
//! HUD, ...). Each builds a canvas tree laid out for 1920x1080 that already
//! works in Play through button actions, `UiPauseMenu` and audio-bus sliders.
//! Also used by the project templates, so it compiles without the editor.

use zestengine_core::{
    Component, ControlsMenu, UiAction, UiAlign, UiButton, UiCanvas, UiFillDirection, UiPanel, UiPauseMenu, UiProgressBar,
    UiRect, UiSlider, UiText, UiToggle,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UiTemplate {
    MainMenu,
    PauseMenu,
    Settings,
    Controls,
    Hud,
    Overlay,
    Dialog,
    LoadingScreen,
}

impl UiTemplate {
    pub const ALL: [UiTemplate; 8] = [
        UiTemplate::MainMenu,
        UiTemplate::PauseMenu,
        UiTemplate::Settings,
        UiTemplate::Controls,
        UiTemplate::Hud,
        UiTemplate::Overlay,
        UiTemplate::Dialog,
        UiTemplate::LoadingScreen,
    ];

    pub fn label(self) -> &'static str {
        match self {
            UiTemplate::MainMenu => "Main Menu",
            UiTemplate::PauseMenu => "Pause Menu",
            UiTemplate::Settings => "Settings",
            UiTemplate::Controls => "Controls",
            UiTemplate::Hud => "HUD",
            UiTemplate::Overlay => "Overlay",
            UiTemplate::Dialog => "Dialog",
            UiTemplate::LoadingScreen => "Loading Screen",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            UiTemplate::MainMenu => "Title, Play / Settings / Quit, settings page with volume sliders",
            UiTemplate::PauseMenu => "Escape toggles it and pauses time; Resume / Settings / Main Menu / Quit",
            UiTemplate::Settings => "Volume sliders bound to the audio mixer, Subtitles and Invert Look toggles, Controls, Back",
            UiTemplate::Controls => "Rebind menu: click a binding, press the new input; scheme switch, paging, reset",
            UiTemplate::Hud => "Health and stamina bars, score, objective, ammo, crosshair",
            UiTemplate::Overlay => "Screen fade, notification toast, subtitles",
            UiTemplate::Dialog => "Modal confirm box with Yes / No",
            UiTemplate::LoadingScreen => "Background, title, progress bar, tip",
        }
    }

    pub fn build(self) -> Vec<TemplateNode> {
        let mut b = Builder::default();
        match self {
            UiTemplate::MainMenu => main_menu(&mut b),
            UiTemplate::PauseMenu => pause_menu(&mut b),
            UiTemplate::Settings => {
                let root = b.canvas("Settings Menu", 20);
                let page = settings_page(&mut b, root, "Settings Page", vec![UiAction::Hide("Settings Page".into())]);
                b.nodes[page].active = true;
            }
            UiTemplate::Controls => {
                let root = b.canvas("Controls Menu", 25);
                let page = controls_page(&mut b, root, "Controls Page", "Controls", vec![UiAction::Hide("Controls Page".into())]);
                b.nodes[page].active = true;
            }
            UiTemplate::Hud => hud(&mut b),
            UiTemplate::Overlay => overlay(&mut b),
            UiTemplate::Dialog => dialog(&mut b),
            UiTemplate::LoadingScreen => loading(&mut b),
        }
        b.nodes
    }
}

/// One object to spawn. `parent` indexes earlier nodes; node 0 is the canvas.
pub struct TemplateNode {
    pub name: String,
    pub parent: Option<usize>,
    pub active: bool,
    pub components: Vec<Box<dyn Component>>,
}

const BG: [f32; 4] = [0.05, 0.06, 0.08, 1.0];
const BOX: [f32; 4] = [0.09, 0.1, 0.13, 0.96];
const BORDER: [f32; 4] = [1.0, 1.0, 1.0, 0.12];
const ACCENT: [f32; 4] = [0.95, 0.75, 0.2, 1.0];
const DIM_TEXT: [f32; 4] = [0.75, 0.77, 0.8, 1.0];

fn at(anchor: [f32; 2], pivot: [f32; 2], position: [f32; 2], size: [f32; 2]) -> UiRect {
    UiRect { anchor_min: anchor, anchor_max: anchor, pivot, position, size }
}

/// Fill the parent, inset by `margin` reference pixels.
fn fill(margin: f32) -> UiRect {
    UiRect {
        anchor_min: [0.0, 0.0],
        anchor_max: [1.0, 1.0],
        pivot: [0.5, 0.5],
        position: [0.0, 0.0],
        size: [-2.0 * margin, -2.0 * margin],
    }
}

fn panel(color: [f32; 4], radius: f32) -> UiPanel {
    UiPanel { color, corner_radius: radius, ..Default::default() }
}

fn boxed() -> UiPanel {
    UiPanel { color: BOX, corner_radius: 12.0, border_width: 1.0, border_color: BORDER }
}

fn text(s: &str, size: f32, align_x: UiAlign) -> UiText {
    UiText { text: s.into(), size, align_x, ..Default::default() }
}

#[derive(Default)]
struct Builder {
    nodes: Vec<TemplateNode>,
}

impl Builder {
    fn add(&mut self, parent: Option<usize>, name: &str, components: Vec<Box<dyn Component>>) -> usize {
        self.nodes.push(TemplateNode { name: name.into(), parent, active: true, components });
        self.nodes.len() - 1
    }

    fn canvas(&mut self, name: &str, sort_order: i32) -> usize {
        self.add(None, name, vec![Box::new(UiCanvas { sort_order, ..Default::default() })])
    }

    fn hidden(&mut self, i: usize) -> usize {
        self.nodes[i].active = false;
        i
    }

    fn label(&mut self, parent: usize, name: &str, rect: UiRect, t: UiText) -> usize {
        self.add(Some(parent), name, vec![Box::new(rect), Box::new(t)])
    }

    fn button(&mut self, parent: usize, name: &str, caption: &str, rect: UiRect, on_click: Vec<UiAction>) -> usize {
        let i = self.add(
            Some(parent),
            name,
            vec![
                Box::new(rect),
                Box::new(panel([0.18, 0.18, 0.22, 0.95], 8.0)),
                Box::new(UiButton { on_click, ..Default::default() }),
            ],
        );
        self.label(i, "Label", fill(0.0), text(caption, 30.0, UiAlign::Center));
        i
    }

    /// Label on the left, slider on the right.
    fn slider_row(&mut self, parent: usize, caption: &str, y: f32, bus: &str) {
        let row = self.add(Some(parent), caption, vec![Box::new(at([0.5, 0.0], [0.5, 0.0], [0.0, y], [600.0, 50.0]))]);
        self.label(row, "Label", at([0.0, 0.5], [0.0, 0.5], [0.0, 0.0], [240.0, 50.0]), text(caption, 26.0, UiAlign::Start));
        self.add(
            Some(row),
            "Slider",
            vec![
                Box::new(at([1.0, 0.5], [1.0, 0.5], [0.0, 0.0], [340.0, 28.0])),
                Box::new(UiSlider { value: 1.0, audio_bus: bus.into(), ..Default::default() }),
            ],
        );
    }

    /// A button whose caption is its own `UiText` (so code can rewrite it).
    fn text_button(&mut self, parent: usize, name: &str, caption: &str, rect: UiRect, size: f32) -> usize {
        self.add(
            Some(parent),
            name,
            vec![
                Box::new(rect),
                Box::new(panel([0.18, 0.18, 0.22, 0.95], 8.0)),
                Box::new(UiButton::default()),
                Box::new(text(caption, size, UiAlign::Center)),
            ],
        )
    }

    fn toggle_row(&mut self, parent: usize, caption: &str, y: f32, toggle: UiToggle) {
        let row = self.add(
            Some(parent),
            caption,
            vec![Box::new(at([0.5, 0.0], [0.5, 0.0], [0.0, y], [600.0, 40.0])), Box::new(toggle)],
        );
        self.label(row, "Label", fill(0.0).with_left_inset(56.0), text(caption, 26.0, UiAlign::Start));
    }
}

trait Inset {
    fn with_left_inset(self, px: f32) -> Self;
}

impl Inset for UiRect {
    fn with_left_inset(mut self, px: f32) -> Self {
        self.size[0] -= px;
        self.position[0] += px * 0.5;
        self
    }
}

/// A centered settings box; `back` runs when Back is clicked. Starts hidden.
fn settings_page(b: &mut Builder, parent: usize, name: &str, back: Vec<UiAction>) -> usize {
    let page = b.add(
        Some(parent),
        name,
        vec![Box::new(at([0.5, 0.5], [0.5, 0.5], [0.0, 0.0], [720.0, 600.0])), Box::new(boxed())],
    );
    b.hidden(page);
    b.label(page, "Title", at([0.5, 0.0], [0.5, 0.0], [0.0, 30.0], [600.0, 70.0]), text("Settings", 48.0, UiAlign::Center));
    b.slider_row(page, "Master Volume", 130.0, "master");
    b.slider_row(page, "Music Volume", 200.0, "music");
    b.slider_row(page, "Effects Volume", 270.0, "sfx");
    b.toggle_row(page, "Subtitles", 350.0, UiToggle { on: true, ..Default::default() });
    // Inverts the Look action's vertical axis in the player's bindings.
    b.toggle_row(page, "Invert Look", 410.0, UiToggle { invert_y_action: "Look".into(), ..Default::default() });
    let controls = format!("{name} Controls");
    b.button(
        page,
        "Controls Button",
        "Controls",
        at([0.5, 1.0], [1.0, 1.0], [-12.0, -36.0], [260.0, 64.0]),
        vec![UiAction::Hide(name.into()), UiAction::Show(controls.clone())],
    );
    b.button(page, "Back Button", "Back", at([0.5, 1.0], [0.0, 1.0], [12.0, -36.0], [260.0, 64.0]), back);
    controls_page(
        b,
        parent,
        &controls,
        &controls,
        vec![UiAction::Hide(controls.clone()), UiAction::Show(name.into())],
    );
    page
}

/// A centered rebind page run by a [`ControlsMenu`] whose object names start
/// with `prefix`. Starts hidden.
fn controls_page(b: &mut Builder, parent: usize, name: &str, prefix: &str, back: Vec<UiAction>) -> usize {
    let menu = ControlsMenu { prefix: prefix.into(), ..Default::default() };
    let rows = menu.rows as usize;
    let page = b.add(
        Some(parent),
        name,
        vec![Box::new(at([0.5, 0.5], [0.5, 0.5], [0.0, 0.0], [960.0, 820.0])), Box::new(boxed()), Box::new(menu)],
    );
    b.hidden(page);
    let n = |s: &str| format!("{prefix} {s}");
    b.label(page, "Title", at([0.5, 0.0], [0.5, 0.0], [0.0, 24.0], [800.0, 64.0]), text("Controls", 48.0, UiAlign::Center));
    b.text_button(page, &n("Scheme"), "Scheme", at([0.5, 0.0], [0.5, 0.0], [0.0, 100.0], [520.0, 52.0]), 26.0);
    for i in 0..rows {
        let y = 172.0 + i as f32 * 54.0;
        let row = b.add(Some(page), &n(&format!("Row {}", i + 1)), vec![Box::new(at([0.5, 0.0], [0.5, 0.0], [0.0, y], [880.0, 48.0]))]);
        b.label(row, &n(&format!("Row {} Label", i + 1)), at([0.0, 0.5], [0.0, 0.5], [0.0, 0.0], [400.0, 48.0]), text("Action", 26.0, UiAlign::Start));
        for (col, x) in [(1, -244.0), (2, 0.0)] {
            let name = n(&format!("Row {} Bind {col}", i + 1));
            b.text_button(row, &name, "-", at([1.0, 0.5], [1.0, 0.5], [x, 0.0], [232.0, 44.0]), 24.0);
        }
    }
    let mut status = text("", 22.0, UiAlign::Center);
    status.color = DIM_TEXT;
    b.label(page, &n("Status"), at([0.5, 0.0], [0.5, 0.0], [0.0, 614.0], [880.0, 40.0]), status);
    let bottom = |x: f32, w: f32| at([0.0, 1.0], [0.0, 1.0], [x, -30.0], [w, 60.0]);
    b.button(page, &n("Prev"), "<", bottom(40.0, 100.0), Vec::new());
    b.label(page, &n("Page"), bottom(150.0, 120.0), text("1 / 1", 26.0, UiAlign::Center));
    b.button(page, &n("Next"), ">", bottom(280.0, 100.0), Vec::new());
    b.button(page, &n("Reset"), "Reset", at([1.0, 1.0], [1.0, 1.0], [-340.0, -30.0], [260.0, 60.0]), Vec::new());
    b.button(page, "Back Button", "Back", at([1.0, 1.0], [1.0, 1.0], [-40.0, -30.0], [280.0, 60.0]), back);
    page
}

fn main_menu(b: &mut Builder) {
    let root = b.canvas("Main Menu", 0);
    b.add(Some(root), "Background", vec![Box::new(fill(0.0)), Box::new(panel(BG, 0.0))]);
    let menu = b.add(Some(root), "Main Menu Panel", vec![Box::new(at([0.0, 0.5], [0.0, 0.5], [160.0, 0.0], [560.0, 640.0]))]);
    let mut title = text("GAME TITLE", 84.0, UiAlign::Start);
    title.color = ACCENT;
    b.label(menu, "Title", at([0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [560.0, 130.0]), title);
    let col = |y: f32| at([0.0, 0.0], [0.0, 0.0], [0.0, y], [380.0, 72.0]);
    b.button(menu, "Play Button", "Play", col(220.0), vec![UiAction::LoadScene(String::new())]);
    b.button(
        menu,
        "Settings Button",
        "Settings",
        col(312.0),
        vec![UiAction::Hide("Main Menu Panel".into()), UiAction::Show("Main Menu Settings".into())],
    );
    b.button(menu, "Quit Button", "Quit", col(404.0), vec![UiAction::Quit]);
    settings_page(
        b,
        root,
        "Main Menu Settings",
        vec![UiAction::Hide("Main Menu Settings".into()), UiAction::Show("Main Menu Panel".into())],
    );
    let mut version = text("v0.1", 22.0, UiAlign::End);
    version.color = DIM_TEXT;
    b.label(root, "Version", at([1.0, 1.0], [1.0, 1.0], [-30.0, -20.0], [300.0, 40.0]), version);
}

fn pause_menu(b: &mut Builder) {
    let root = b.add(
        None,
        "Pause Menu",
        vec![
            Box::new(UiCanvas { sort_order: 10, ..Default::default() }),
            Box::new(UiPauseMenu { panel: "Pause Panel".into(), ..Default::default() }),
        ],
    );
    let dim = b.add(Some(root), "Pause Panel", vec![Box::new(fill(0.0)), Box::new(panel([0.0, 0.0, 0.0, 0.6], 0.0))]);
    b.hidden(dim);
    let card = b.add(
        Some(dim),
        "Pause Box",
        vec![Box::new(at([0.5, 0.5], [0.5, 0.5], [0.0, 0.0], [520.0, 560.0])), Box::new(boxed())],
    );
    b.label(card, "Title", at([0.5, 0.0], [0.5, 0.0], [0.0, 30.0], [440.0, 80.0]), text("Paused", 56.0, UiAlign::Center));
    let col = |y: f32| at([0.5, 0.0], [0.5, 0.0], [0.0, y], [380.0, 72.0]);
    b.button(card, "Resume Button", "Resume", col(140.0), vec![UiAction::Hide("Pause Panel".into())]);
    b.button(
        card,
        "Settings Button",
        "Settings",
        col(228.0),
        vec![UiAction::Hide("Pause Box".into()), UiAction::Show("Pause Settings".into())],
    );
    b.button(card, "Main Menu Button", "Main Menu", col(316.0), vec![UiAction::LoadScene(String::new())]);
    b.button(card, "Quit Button", "Quit", col(404.0), vec![UiAction::Quit]);
    settings_page(
        b,
        dim,
        "Pause Settings",
        vec![UiAction::Hide("Pause Settings".into()), UiAction::Show("Pause Box".into())],
    );
}

fn hud(b: &mut Builder) {
    let root = b.canvas("HUD", 0);
    let bar = |b: &mut Builder, name: &str, y: f32, w: f32, h: f32, fill_color: [f32; 4]| {
        b.add(
            Some(root),
            name,
            vec![
                Box::new(at([0.0, 0.0], [0.0, 0.0], [40.0, y], [w, h])),
                Box::new(UiProgressBar { fill: fill_color, ..Default::default() }),
            ],
        )
    };
    let health = bar(b, "Health Bar", 40.0, 380.0, 30.0, [0.85, 0.25, 0.25, 1.0]);
    b.label(health, "Label", fill(0.0).with_left_inset(20.0), text("HEALTH", 20.0, UiAlign::Start));
    bar(b, "Stamina Bar", 80.0, 300.0, 14.0, [0.35, 0.8, 0.45, 1.0]);
    b.label(root, "Objective", at([0.5, 0.0], [0.5, 0.0], [0.0, 40.0], [900.0, 50.0]), text("Objective: find the exit", 30.0, UiAlign::Center));
    b.label(root, "Score", at([1.0, 0.0], [1.0, 0.0], [-40.0, 36.0], [420.0, 60.0]), text("Score: 0", 40.0, UiAlign::End));
    b.label(root, "Ammo", at([1.0, 1.0], [1.0, 1.0], [-40.0, -40.0], [320.0, 80.0]), text("30 / 90", 56.0, UiAlign::End));
    b.add(
        Some(root),
        "Crosshair",
        vec![Box::new(at([0.5, 0.5], [0.5, 0.5], [0.0, 0.0], [8.0, 8.0])), Box::new(panel([1.0, 1.0, 1.0, 0.9], 4.0))],
    );
}

fn overlay(b: &mut Builder) {
    let root = b.canvas("Overlay", 50);
    let toast = b.add(
        Some(root),
        "Toast",
        vec![Box::new(at([0.5, 0.0], [0.5, 0.0], [0.0, 120.0], [620.0, 84.0])), Box::new(boxed())],
    );
    b.label(toast, "Message", fill(16.0), text("Notification", 28.0, UiAlign::Center));
    let subs = b.add(
        Some(root),
        "Subtitles",
        vec![
            Box::new(at([0.5, 1.0], [0.5, 1.0], [0.0, -80.0], [1200.0, 96.0])),
            Box::new(panel([0.0, 0.0, 0.0, 0.55], 8.0)),
        ],
    );
    b.label(subs, "Line", fill(12.0), text("Subtitle text goes here.", 30.0, UiAlign::Center));
    // Last so it covers everything: fade by setting its panel alpha.
    b.add(Some(root), "Fade", vec![Box::new(fill(0.0)), Box::new(panel([0.0, 0.0, 0.0, 0.0], 0.0))]);
}

fn dialog(b: &mut Builder) {
    let root = b.canvas("Dialog", 30);
    let dim = b.add(Some(root), "Dialog Panel", vec![Box::new(fill(0.0)), Box::new(panel([0.0, 0.0, 0.0, 0.55], 0.0))]);
    let card = b.add(
        Some(dim),
        "Dialog Box",
        vec![Box::new(at([0.5, 0.5], [0.5, 0.5], [0.0, 0.0], [640.0, 320.0])), Box::new(boxed())],
    );
    b.label(card, "Title", at([0.5, 0.0], [0.5, 0.0], [0.0, 24.0], [560.0, 60.0]), text("Are you sure?", 40.0, UiAlign::Center));
    let mut msg = text("Unsaved progress will be lost.", 26.0, UiAlign::Center);
    msg.color = DIM_TEXT;
    b.label(card, "Message", at([0.5, 0.0], [0.5, 0.0], [0.0, 96.0], [560.0, 90.0]), msg);
    let close = vec![UiAction::Hide("Dialog Panel".into())];
    b.button(card, "Yes Button", "Yes", at([0.5, 1.0], [1.0, 1.0], [-12.0, -30.0], [240.0, 64.0]), close.clone());
    b.button(card, "No Button", "No", at([0.5, 1.0], [0.0, 1.0], [12.0, -30.0], [240.0, 64.0]), close);
}

fn loading(b: &mut Builder) {
    let root = b.canvas("Loading Screen", 100);
    b.add(Some(root), "Background", vec![Box::new(fill(0.0)), Box::new(panel(BG, 0.0))]);
    b.label(root, "Title", at([0.5, 0.5], [0.5, 0.5], [0.0, -40.0], [800.0, 90.0]), text("Loading...", 56.0, UiAlign::Center));
    b.add(
        Some(root),
        "Loading Bar",
        vec![
            Box::new(at([0.5, 1.0], [0.5, 1.0], [0.0, -140.0], [900.0, 14.0])),
            Box::new(UiProgressBar {
                value: 0.35,
                direction: UiFillDirection::LeftToRight,
                fill: ACCENT,
                corner_radius: 7.0,
                ..Default::default()
            }),
        ],
    );
    let mut tip = text("Tip: change this text from code with ctx.modify::<UiText>.", 26.0, UiAlign::Center);
    tip.color = DIM_TEXT;
    b.label(root, "Tip", at([0.5, 1.0], [0.5, 1.0], [0.0, -180.0], [1200.0, 50.0]), tip);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn templates_reference_existing_objects() {
        for t in UiTemplate::ALL {
            let nodes = t.build();
            assert!(nodes[0].parent.is_none() && nodes[0].components.iter().any(|c| c.as_any().is::<UiCanvas>()));
            for (i, n) in nodes.iter().enumerate() {
                assert!(n.parent.is_none_or(|p| p < i), "{} parent order", n.name);
                for c in &n.components {
                    let actions = c.as_any().downcast_ref::<UiButton>().map(|b| b.on_click.clone()).unwrap_or_default();
                    for a in actions {
                        if let UiAction::Show(name) | UiAction::Hide(name) | UiAction::Toggle(name) = a {
                            assert!(nodes.iter().any(|m| m.name == name), "{}: {name} missing", t.label());
                        }
                    }
                    if let Some(p) = c.as_any().downcast_ref::<UiPauseMenu>() {
                        assert!(nodes.iter().any(|m| m.name == p.panel));
                    }
                }
            }
        }
    }
}
