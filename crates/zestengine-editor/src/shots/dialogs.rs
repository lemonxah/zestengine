//! Headless screenshot of the New Project window (footer right-aligned through
//! `widgets::dialog_footer`). Pack import dialogs are in `pack_import.rs`.

use super::*;
use crate::shot::Shot;

#[test]
fn new_project_dialog() {
    let mut shot = Shot::new(720.0, 760.0, 1.2);
    let mut parent = String::from("/home/user/projects");
    let mut name = String::from("My Game");
    let mut template = String::new();
    let mut filter = None;
    let footer = std::cell::Cell::new(None::<(f32, f32)>);
    let img = shot.run(|ctx| {
        egui::Window::new("New Project").collapsible(false).resizable(false).anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
            let mut cancel = false;
            new_project_ui(ui, NewProjectState { parent: &mut parent, name: &mut name, template: &mut template, filter: &mut filter }, &mut cancel);
            footer.set(Some((ui.min_rect().right(), ui.max_rect().right())));
        });
    });
    shot.save_named(&img, "dialog_new_project");
    assert!(img.non_background() > 500);
    assert!(footer.get().is_some());
}
