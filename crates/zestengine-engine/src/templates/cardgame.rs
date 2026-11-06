//! Multiplayer trading-card game template: a shared table, two player seats
//! and a hand/turn HUD, with the Network Manager ready for LAN multiplayer.
//! The card and turn rules are the natural next layer (a Rust plugin).

use zestengine_core::{
    Key, NetAutoStart, NetworkManager, NetworkObject, SpawnPoint, UiAlign, UiButton, UiCanvas,
    UiPanel, UiRect, UiText,
};

use super::TemplateContent;
use super::builder::{self, Scene, button, keys};

fn card_button() -> UiButton {
    UiButton {
        normal: [0.88, 0.88, 0.94, 1.0],
        hovered: [1.0, 1.0, 1.0, 1.0],
        pressed: [0.7, 0.7, 0.78, 1.0],
        disabled: [0.5, 0.5, 0.55, 0.5],
        ..Default::default()
    }
}

pub fn build() -> TemplateContent {
    let mut s = Scene::new();
    builder::sun(&mut s, [-0.2, -1.0, -0.5], 3.0);

    s.ground(10.0, "wood");
    let table = s.block("Table", None, [0.0, 1.0, 0.0], [3.0, 0.1, 2.0], "wood");
    s.with(table, NetworkObject::default());

    for (k, x) in [-1.2f32, 1.2f32].into_iter().enumerate() {
        let seat = s.empty(&format!("Seat {}", k + 1), None, [x, 0.0, 0.0]);
        s.with(
            seat,
            SpawnPoint {
                tag: "player".into(),
                index: k as u32,
            },
        );
    }

    let net = s.empty("Network", None, [0.0; 3]);
    s.with(
        net,
        NetworkManager {
            session_name: "ZestEngine Card Table".into(),
            port: 47600,
            max_players: 2,
            discovery: true,
            spawn_tag: "player".into(),
            auto_start: NetAutoStart::Off,
            ..Default::default()
        },
    );

    s.camera("Main Camera", None, [0.0, 2.4, 3.2], 50.0);

    let canvas = s.empty("UI", None, [0.0; 3]);
    s.with(canvas, UiCanvas::default());
    let turn = s.empty("Turn", Some(canvas), [0.0; 3]);
    s.with(
        turn,
        UiRect {
            anchor_min: [0.5, 0.9],
            anchor_max: [0.5, 0.9],
            pivot: [0.5, 0.5],
            position: [0.0, 0.0],
            size: [300.0, 50.0],
        },
    );
    s.with(
        turn,
        UiPanel {
            color: [0.05, 0.06, 0.1, 0.9],
            corner_radius: 8.0,
            ..Default::default()
        },
    );
    s.with(
        turn,
        UiText {
            text: "Your turn".into(),
            size: 24.0,
            align_x: UiAlign::Center,
            ..Default::default()
        },
    );

    let hand = s.empty("Hand", Some(canvas), [0.0; 3]);
    s.with(
        hand,
        UiRect {
            anchor_min: [0.1, 0.0],
            anchor_max: [0.9, 0.25],
            pivot: [0.5, 0.0],
            position: [0.0, 0.0],
            size: [0.0, 0.0],
        },
    );
    for k in 0..5 {
        let card = s.empty(&format!("Card {}", k + 1), Some(hand), [0.0; 3]);
        s.with(
            card,
            UiRect {
                anchor_min: [0.0, 0.0],
                anchor_max: [0.0, 0.0],
                pivot: [0.5, 0.5],
                position: [k as f32 * 110.0 - 220.0, 0.0],
                size: [100.0, 140.0],
            },
        );
        s.with(card, card_button());
        let label = s.empty("Label", Some(card), [0.0; 3]);
        s.with(
            label,
            UiRect {
                anchor_min: [0.0, 0.0],
                anchor_max: [1.0, 1.0],
                pivot: [0.5, 0.5],
                position: [0.0, 0.0],
                size: [0.0, 0.0],
            },
        );
        s.with(
            label,
            UiText {
                text: format!("{}", k + 1),
                size: 28.0,
                align_x: UiAlign::Center,
                ..Default::default()
            },
        );
    }

    builder::pause_menu(&mut s, builder::MAIN_SCENE, false);

    let bindings = builder::bindings(vec![button("Confirm", keys(&[Key::Space]))], vec![]);
    builder::content(s.finish(), Vec::new(), bindings, Vec::new())
}
