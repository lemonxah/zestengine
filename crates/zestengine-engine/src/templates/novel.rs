//! Visual Novel template: a fixed camera on a speaker and a dialogue/choice
//! UI. Desktop. The scene is the starter a story script drives; wire the
//! dialogue text, speaker name and choice buttons to a script or plugin.

use zestengine_assets::PrimitiveShape;
use zestengine_core::{Key, UiAction, UiAlign, UiButton, UiCanvas, UiPanel, UiRect, UiText};

use super::TemplateContent;
use super::builder::{self, Scene, button, keys};

fn choice_button(
    s: &mut Scene,
    parent: usize,
    name: &str,
    label: &str,
    action: UiAction,
    x: f32,
) -> usize {
    let i = s.empty(name, Some(parent), [0.0; 3]);
    s.with(
        i,
        UiRect {
            anchor_min: [0.0, 0.0],
            anchor_max: [0.0, 0.0],
            pivot: [0.5, 0.5],
            position: [x, 28.0],
            size: [220.0, 44.0],
        },
    );
    s.with(
        i,
        UiButton {
            on_click: vec![action],
            ..Default::default()
        },
    );
    let lbl = s.empty("Label", Some(i), [0.0; 3]);
    s.with(
        lbl,
        UiRect {
            anchor_min: [0.0, 0.0],
            anchor_max: [1.0, 1.0],
            pivot: [0.5, 0.5],
            position: [0.0, 0.0],
            size: [0.0, 0.0],
        },
    );
    s.with(
        lbl,
        UiText {
            text: label.into(),
            size: 18.0,
            align_x: UiAlign::Center,
            ..Default::default()
        },
    );
    i
}

pub fn build() -> TemplateContent {
    let mut s = Scene::new();
    builder::sun(&mut s, [-0.3, -1.0, -0.4], 3.0);

    s.terrain(12.0, 16, 0.0, "wall");
    s.block("Backdrop", None, [0.0, 2.5, -4.0], [10.0, 5.0, 0.3], "trim");
    let speaker = s.empty("Speaker", None, [0.0, 0.0, -1.2]);
    s.prim(
        "Body",
        PrimitiveShape::Capsule,
        Some(speaker),
        [0.0, 1.0, 0.0],
        [0.9, 1.6, 0.9],
        "player",
    );
    s.prim(
        "Head",
        PrimitiveShape::Sphere,
        Some(speaker),
        [0.0, 2.0, 0.0],
        [0.5; 3],
        "skin",
    );

    s.camera("Main Camera", None, [0.0, 1.7, 3.6], 50.0);

    let canvas = s.empty("UI", None, [0.0; 3]);
    s.with(canvas, UiCanvas::default());
    let dialogue = s.empty("Dialogue Box", Some(canvas), [0.0; 3]);
    s.with(
        dialogue,
        UiRect {
            anchor_min: [0.06, 0.04],
            anchor_max: [0.94, 0.34],
            pivot: [0.5, 0.0],
            position: [0.0, 0.0],
            size: [0.0, 0.0],
        },
    );
    s.with(
        dialogue,
        UiPanel {
            color: [0.02, 0.02, 0.08, 0.9],
            corner_radius: 12.0,
            ..Default::default()
        },
    );
    let name = s.empty("Speaker Name", Some(dialogue), [0.0; 3]);
    s.with(
        name,
        UiRect {
            anchor_min: [0.0, 1.0],
            anchor_max: [0.0, 1.0],
            pivot: [0.0, 1.0],
            position: [20.0, -16.0],
            size: [220.0, 30.0],
        },
    );
    s.with(
        name,
        UiText {
            text: "Narrator".into(),
            size: 24.0,
            align_x: UiAlign::Start,
            ..Default::default()
        },
    );
    let body = s.empty("Dialogue", Some(dialogue), [0.0; 3]);
    s.with(
        body,
        UiRect {
            anchor_min: [0.0, 0.0],
            anchor_max: [1.0, 1.0],
            pivot: [0.5, 0.5],
            position: [0.0, 8.0],
            size: [0.0, 0.0],
        },
    );
    s.with(
        body,
        UiText {
            text: "Your story starts here. Click a choice, or advance with Space.".into(),
            size: 20.0,
            align_x: UiAlign::Start,
            ..Default::default()
        },
    );
    choice_button(
        &mut s,
        dialogue,
        "Choice A",
        "Talk",
        UiAction::LoadScene(builder::MAIN_SCENE.into()),
        150.0,
    );
    choice_button(
        &mut s,
        dialogue,
        "Choice B",
        "Stay silent",
        UiAction::Quit,
        400.0,
    );

    builder::pause_menu(&mut s, builder::MAIN_SCENE, false);

    let bindings = builder::bindings(
        vec![
            button("Advance", keys(&[Key::Space])),
            button("ChoiceA", keys(&[Key::A])),
            button("ChoiceB", keys(&[Key::B])),
        ],
        vec![],
    );
    builder::content(s.finish(), Vec::new(), bindings, Vec::new())
}
