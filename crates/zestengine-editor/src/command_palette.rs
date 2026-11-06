//! Ctrl K command palette: a filtered list of editor commands. The editor
//! supplies the commands each frame and runs the one picked.

use crate::theme;

#[derive(Default)]
pub struct CommandPalette {
    open: bool,
    query: String,
    selected: usize,
    focus: bool,
}

/// One palette entry. `group` shows dimmed on the right.
pub struct Command<T> {
    pub label: String,
    pub group: &'static str,
    pub id: T,
}

/// All query words appear in the label or group (case-insensitive).
fn matches(query: &str, label: &str, group: &str) -> bool {
    let hay = format!("{} {}", label.to_lowercase(), group.to_lowercase());
    query.split_whitespace().all(|w| hay.contains(&w.to_lowercase()))
}

impl CommandPalette {
    pub fn open(&mut self) {
        self.open = true;
        self.query.clear();
        self.selected = 0;
        self.focus = true;
    }

    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Show the palette (when open); returns the picked command's id.
    pub fn ui<T: Copy>(&mut self, ctx: &egui::Context, commands: &[Command<T>]) -> Option<T> {
        if !self.open {
            return None;
        }
        let shown: Vec<&Command<T>> = commands.iter().filter(|c| matches(&self.query, &c.label, c.group)).collect();
        self.selected = self.selected.min(shown.len().saturating_sub(1));
        let (up, down, enter, escape) = ctx.input(|i| {
            (
                i.key_pressed(egui::Key::ArrowUp),
                i.key_pressed(egui::Key::ArrowDown),
                i.key_pressed(egui::Key::Enter),
                i.key_pressed(egui::Key::Escape),
            )
        });
        if escape {
            self.open = false;
            return None;
        }
        if down && !shown.is_empty() {
            self.selected = (self.selected + 1) % shown.len();
        }
        if up && !shown.is_empty() {
            self.selected = (self.selected + shown.len() - 1) % shown.len();
        }
        let mut picked = enter.then(|| shown.get(self.selected).map(|c| c.id)).flatten();

        let screen = ctx.content_rect();
        let width = 520.0_f32.min(screen.width() - 32.0);
        let mut close = false;
        egui::Area::new(egui::Id::new("zestengine-command-palette"))
            .order(egui::Order::Foreground)
            .fixed_pos(egui::pos2(screen.center().x - width * 0.5, screen.top() + 72.0))
            .show(ctx, |ui| {
                egui::Frame::new()
                    .fill(theme::BG_PANEL)
                    .stroke(egui::Stroke::new(1.0_f32, theme::LINE_FIELD))
                    .corner_radius(egui::CornerRadius::same(theme::R_TOOLBAR))
                    .inner_margin(egui::Margin::same(8))
                    .show(ui, |ui| {
                        ui.set_width(width);
                        let field = crate::widgets::search_field(ui, &mut self.query, "Search commands", width);
                        if std::mem::take(&mut self.focus) {
                            field.request_focus();
                        }
                        if field.changed() {
                            self.selected = 0;
                        }
                        ui.add_space(6.0);
                        // Size from this frame's rows: the scroll area otherwise keeps the
                        // shorter height of an earlier search.
                        let rows = shown.len().max(1) as f32;
                        let height = (rows * (theme::H_TOOLBAR + ui.spacing().item_spacing.y)).min(360.0);
                        egui::ScrollArea::vertical()
                            .auto_shrink([false, false])
                            .min_scrolled_height(height)
                            .max_height(height)
                            .show(ui, |ui| {
                            if shown.is_empty() {
                                ui.label(egui::RichText::new("No matching commands").color(theme::TEXT_DIM));
                            }
                            for (k, c) in shown.iter().enumerate() {
                                let (rect, resp) =
                                    ui.allocate_exact_size(egui::vec2(width, theme::H_TOOLBAR), egui::Sense::click());
                                let selected = k == self.selected;
                                if selected || resp.hovered() {
                                    ui.painter().rect_filled(
                                        rect,
                                        egui::CornerRadius::same(theme::R_ROW),
                                        if selected { theme::accent_faint() } else { theme::BG_BUTTON },
                                    );
                                }
                                ui.painter().text(
                                    egui::pos2(rect.left() + 10.0, rect.center().y),
                                    egui::Align2::LEFT_CENTER,
                                    &c.label,
                                    egui::FontId::proportional(theme::SIZE_BODY),
                                    if selected { theme::TEXT_STRONG } else { theme::TEXT_SECONDARY },
                                );
                                ui.painter().text(
                                    egui::pos2(rect.right() - 10.0, rect.center().y),
                                    egui::Align2::RIGHT_CENTER,
                                    c.group,
                                    egui::FontId::proportional(theme::SIZE_META),
                                    theme::TEXT_DIM,
                                );
                                if selected && (up || down) {
                                    resp.scroll_to_me(None);
                                }
                                if resp.clicked() {
                                    picked = Some(c.id);
                                }
                            }
                        });
                    });
            });
        // Click outside closes it.
        if ctx.input(|i| i.pointer.any_pressed())
            && let Some(p) = ctx.input(|i| i.pointer.interact_pos())
            && ctx.layer_id_at(p).is_none_or(|l| l.id != egui::Id::new("zestengine-command-palette"))
        {
            close = true;
        }
        if picked.is_some() || close {
            self.open = false;
        }
        picked
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_words_match_label_or_group() {
        assert!(matches("ui main", "Add UI template: Main Menu", "UI"));
        assert!(matches("SAVE", "Save scene", "File"));
        assert!(!matches("bake x", "Bake lighting", "Lighting"));
        assert!(matches("", "anything", "x"));
    }
}
