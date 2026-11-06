//! The "Blendshapes" sub-section of the inspector's Mesh & Material: a filtered,
//! grouped, virtualised list of the model's morph targets with a weight slider
//! each. Blendshapes are data on the mesh (weights live in the object's `Blend
//! Shapes` component), never scene objects, so the Hierarchy stays clean.
//!
//! The list logic (grouping, filtering, row building) is plain functions so it
//! is tested without a UI; a face model with 800 shapes only lays out the rows
//! in view.

use std::collections::{HashMap, HashSet};

use egui::{RichText, Ui};
use zestengine_core::BlendShapes;

use crate::{icons::Icon, theme, widgets};

/// Models with more shapes than this start grouped (ARKit faces have 52, VRM
/// and full-body avatars hundreds).
pub const GROUP_MIN: usize = 24;
/// Group of the shapes with no sibling to share a group with.
const OTHER: &str = "Other";
/// Visible list height before it scrolls.
const LIST_HEIGHT: f32 = 264.0;

/// One line of the list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Row {
    /// A group header: its name, how many shapes it holds (after the filter) and whether it is open.
    Group { name: String, count: usize, open: bool },
    /// A shape, by index into the names.
    Shape(usize),
}

/// The group a blendshape sorts under, from its name: the prefix before a
/// `.` / `|`, the first two parts of `Fcl_BRW_Angry`, the lead word of
/// `mouthSmileLeft` / `JawOpen`. Names with no such structure have none.
pub fn group_of(name: &str) -> Option<String> {
    let name = name.trim();
    if let Some((head, rest)) = name.split_once(['.', '|'])
        && !head.is_empty()
        && !rest.is_empty()
    {
        return Some(head.to_owned());
    }
    let parts: Vec<&str> = name.split('_').filter(|p| !p.is_empty()).collect();
    if parts.len() >= 3 {
        return Some(format!("{}_{}", parts[0], parts[1]));
    }
    if parts.len() == 2 {
        return Some(parts[0].to_owned());
    }
    // camelCase / PascalCase: the first word, when there is a second.
    let chars: Vec<char> = name.chars().collect();
    let cut = (1..chars.len()).find(|&i| chars[i].is_uppercase() && !chars[i - 1].is_uppercase())?;
    Some(chars[..cut].iter().collect())
}

/// Every name's group; a group with a single member folds into "Other".
pub fn groups(names: &[String]) -> Vec<String> {
    let raw: Vec<Option<String>> = names.iter().map(|n| group_of(n)).collect();
    let mut count: HashMap<&str, usize> = HashMap::new();
    for g in raw.iter().flatten() {
        *count.entry(g.as_str()).or_default() += 1;
    }
    raw.iter().map(|g| g.as_ref().filter(|g| count[g.as_str()] > 1).cloned().unwrap_or_else(|| OTHER.to_owned())).collect()
}

/// Whether `name` holds every space separated word of `filter` (case blind).
pub fn matches(name: &str, filter: &str) -> bool {
    // No words (the usual case): no need to lowercase the name.
    let mut words = filter.split_whitespace().peekable();
    if words.peek().is_none() {
        return true;
    }
    let name = name.to_lowercase();
    words.all(|w| name.contains(&w.to_lowercase()))
}

/// `(base name, weight)` of an in-between frame name, `Mouth@50` = (`Mouth`, 0.5)
/// (the rule of the importer, which names the frames).
fn split_inbetween(name: &str) -> Option<(&str, f32)> {
    let (base, tail) = name.rsplit_once('@')?;
    let percent: f32 = tail.parse().ok()?;
    (!base.is_empty() && percent > 0.0 && percent < 100.0).then_some((base, percent / 100.0))
}

/// The in-between frames of each base shape that is in the list: base index to
/// `(frame index, weight)`, ascending by weight. A frame whose base is missing
/// stays a shape of its own.
pub fn inbetween_markers(names: &[String]) -> HashMap<usize, Vec<(usize, f32)>> {
    let index: HashMap<&str, usize> = names.iter().enumerate().map(|(i, n)| (n.as_str(), i)).collect();
    let mut out: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
    for (j, n) in names.iter().enumerate() {
        if let Some((base, w)) = split_inbetween(n)
            && let Some(&i) = index.get(base)
        {
            out.entry(i).or_default().push((j, w));
        }
    }
    for frames in out.values_mut() {
        frames.sort_by(|a, b| a.1.total_cmp(&b.1));
    }
    out
}

/// The rows to show. Blank names never show; `active_only` keeps shapes with a
/// non-zero weight; `grouped` adds a header per group (closed ones list no
/// shapes, unless a filter is typed, which opens everything it matched).
#[allow(dead_code)] // public API entry used by editor-side callers only
pub fn build_rows(names: &[String], weights: &HashMap<&str, f32>, filter: &str, active_only: bool, grouped: bool, collapsed: &HashSet<String>) -> Vec<Row> {
    build_rows_with(names, &Layout::of(names), weights, filter, active_only, grouped, collapsed)
}

/// What only depends on the names: groups and in-between markers. Built once per
/// model (cached in egui memory), not every frame.
#[derive(Clone, Default)]
pub struct Layout {
    pub groups: Vec<String>,
    pub markers: HashMap<usize, Vec<(usize, f32)>>,
    hidden: HashSet<usize>,
}

impl Layout {
    pub fn of(names: &[String]) -> Self {
        let markers = inbetween_markers(names);
        // An in-between frame (`Name@50`) is a marker on its base shape's row, not a row.
        let hidden = markers.values().flatten().map(|(i, _)| *i).collect();
        Self { groups: groups(names), markers, hidden }
    }
}

/// `build_rows` with the name-only data precomputed.
pub fn build_rows_with(names: &[String], layout: &Layout, weights: &HashMap<&str, f32>, filter: &str, active_only: bool, grouped: bool, collapsed: &HashSet<String>) -> Vec<Row> {
    let inbetweens = &layout.markers;
    let hidden = &layout.hidden;
    let keep: Vec<usize> = (0..names.len())
        .filter(|&i| !names[i].trim().is_empty() && !hidden.contains(&i))
        .filter(|&i| !active_only || weights.get(names[i].as_str()).is_some_and(|w| *w != 0.0))
        .filter(|&i| matches(&names[i], filter) || inbetweens.get(&i).is_some_and(|f| f.iter().any(|(j, _)| matches(&names[*j], filter))))
        .collect();
    if !grouped {
        return keep.into_iter().map(Row::Shape).collect();
    }
    let of = &layout.groups;
    let mut order: Vec<&str> = of.iter().map(String::as_str).collect::<HashSet<_>>().into_iter().collect();
    // "Other" last, the rest alphabetical.
    order.sort_by_key(|g| (*g == OTHER, g.to_lowercase()));
    let searching = !filter.trim().is_empty();
    let mut rows = Vec::new();
    for g in order {
        let members: Vec<usize> = keep.iter().copied().filter(|&i| of[i] == g).collect();
        if members.is_empty() {
            continue;
        }
        let open = searching || !collapsed.contains(g);
        rows.push(Row::Group { name: g.to_owned(), count: members.len(), open });
        if open {
            rows.extend(members.into_iter().map(Row::Shape));
        }
    }
    rows
}

/// Set one weight on the component; a zero drops the entry (same as never set).
pub fn set_weight(shapes: &mut BlendShapes, name: &str, weight: f32) {
    if weight == 0.0 {
        shapes.weights.retain(|(n, _)| n != name);
    } else {
        shapes.set(name, weight);
    }
}

/// How many shapes carry a non-zero weight.
pub fn active_count(shapes: &BlendShapes) -> usize {
    shapes.weights.iter().filter(|(_, w)| *w != 0.0).count()
}

/// The sub-section. `names` is every blendshape of the object's meshes; weights
/// are edited on the object's `Blend Shapes` component (added on the first
/// edit). Returns whether a weight changed.
pub fn blendshapes_section(ui: &mut Ui, names: &[String], components: &mut Vec<Box<dyn zestengine_core::Component>>) -> bool {
    if names.is_empty() {
        return false;
    }
    let mut changed = false;
    let id = ui.make_persistent_id("zestengine-blendshapes-section");
    let slot = components.iter().position(|c| c.as_any().is::<BlendShapes>());
    let active = slot.and_then(|i| components[i].as_any().downcast_ref::<BlendShapes>()).map_or(0, active_count);
    let mut reset_all = false;
    let (open, _) = widgets::section_header_with(ui, id, None, "Blendshapes", None, false, |ui| {
        if active > 0 {
            reset_all = widgets::icon_button(ui, Icon::Reset, "Reset all blendshape weights to 0", false).clicked();
        }
        let text = if active > 0 { format!("{active} active · {}", names.len()) } else { names.len().to_string() };
        ui.label(RichText::new(text).size(theme::SIZE_META).color(theme::TEXT_DIM));
    });
    if reset_all && let Some(i) = slot && let Some(b) = components[i].as_any_mut().downcast_mut::<BlendShapes>() {
        b.weights.clear();
        changed = true;
    }
    if !open {
        return changed;
    }

    let state_id = id.with("state");
    let mut filter: String = ui.data(|d| d.get_temp(state_id.with("filter"))).unwrap_or_default();
    let mut active_only = ui.data_mut(|d| *d.get_persisted_mut_or_default::<bool>(state_id.with("active")));
    let mut collapsed: Vec<String> = ui.data_mut(|d| d.get_persisted_mut_or_default::<Vec<String>>(state_id.with("collapsed")).clone());

    egui::Frame::new().inner_margin(egui::Margin { left: 8, right: 0, top: 0, bottom: 6 }).show(ui, |ui| {
        ui.horizontal(|ui| {
            let width = (ui.available_width() - 96.0).max(80.0);
            widgets::search_field(ui, &mut filter, &format!("Filter {} shapes", names.len()), width);
            ui.checkbox(&mut active_only, "Active").on_hover_text("Only shapes with a weight above 0");
        });

        if components.iter().any(|c| c.as_any().is::<zestengine_core::FaceTracking>()) {
            ui.label(RichText::new("Face Tracking overrides the shapes it drives while it tracks.").size(theme::SIZE_META).color(theme::TEXT_DIM));
        }
        let by_name: HashMap<&str, f32> = slot
            .and_then(|i| components[i].as_any().downcast_ref::<BlendShapes>())
            .map(|b| b.weights.iter().map(|(n, w)| (n.as_str(), *w)).collect())
            .unwrap_or_default();
        let grouped = names.len() > GROUP_MIN;
        // Name-only layout (groups, in-between markers): cached per name list, not rebuilt every frame.
        let layout_id = state_id.with(("layout", names.len(), names.iter().flat_map(|n| n.bytes()).fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))));
        let layout: std::sync::Arc<Layout> = ui.data_mut(|d| d.get_temp::<std::sync::Arc<Layout>>(layout_id)).unwrap_or_else(|| {
            let l = std::sync::Arc::new(Layout::of(names));
            ui.data_mut(|d| d.insert_temp(layout_id, l.clone()));
            l
        });
        let markers = &layout.markers;
        let closed: HashSet<String> = collapsed.iter().cloned().collect();
        let rows = build_rows_with(names, &layout, &by_name, &filter, active_only, grouped, &closed);
        if rows.is_empty() {
            ui.label(RichText::new("No blendshapes match").size(theme::SIZE_META).color(theme::TEXT_DIM));
            return;
        }
        let mut edit: Option<(String, f32)> = None;
        let mut toggle: Option<String> = None;
        egui::ScrollArea::vertical().id_salt(state_id.with("list")).max_height(LIST_HEIGHT).auto_shrink([false, true]).show_rows(ui, theme::H_ROW, rows.len(), |ui, range| {
            for row in &rows[range] {
                match row {
                    Row::Group { name, count, open } => {
                        let (rect, response) = ui.allocate_exact_size(egui::vec2(ui.available_width(), theme::H_ROW), egui::Sense::click());
                        widgets::paint_caret(ui.painter(), egui::Rect::from_center_size(egui::pos2(rect.left() + 8.0, rect.center().y), egui::Vec2::splat(12.0)), *open, theme::TEXT_DIM);
                        let font = theme::semibold(theme::SIZE_META);
                        ui.painter().text(egui::pos2(rect.left() + 20.0, rect.center().y), egui::Align2::LEFT_CENTER, format!("{name}  ·  {count}"), font, theme::TEXT_MUTED);
                        if response.clicked() {
                            toggle = Some(name.clone());
                        }
                    }
                    Row::Shape(i) => {
                        let name = &names[*i];
                        let mut w = by_name.get(name.as_str()).copied().unwrap_or(0.0);
                        // In-between frames ride on the row: "Mouth  ◇ 50".
                        let label = match markers.get(i) {
                            Some(frames) => format!("{name}  \u{25c7} {}", frames.iter().map(|(_, f)| format!("{:.0}", f * 100.0)).collect::<Vec<_>>().join(" \u{b7} ")),
                            None => name.clone(),
                        };
                        widgets::property_row_w(ui, label.as_str(), 120.0, |ui| {
                            let reset = ui.available_width().min(24.0);
                            ui.spacing_mut().slider_width = (ui.available_width() - reset - 64.0).max(40.0);
                            if widgets::slider(ui, egui::Slider::new(&mut w, 0.0..=1.0).show_value(true).fixed_decimals(2)).changed() {
                                edit = Some((name.clone(), w));
                            }
                            if w != 0.0 && widgets::icon_button(ui, Icon::Reset, "Reset to 0", false).clicked() {
                                edit = Some((name.clone(), 0.0));
                            }
                        });
                    }
                }
            }
        });
        if let Some(g) = toggle {
            match collapsed.iter().position(|c| *c == g) {
                Some(p) => {
                    collapsed.remove(p);
                }
                None => collapsed.push(g),
            }
        }
        if let Some((name, w)) = edit {
            let i = match slot {
                Some(i) => i,
                None => {
                    components.push(Box::new(BlendShapes::default()));
                    components.len() - 1
                }
            };
            if let Some(b) = components[i].as_any_mut().downcast_mut::<BlendShapes>() {
                set_weight(b, &name, w);
                changed = true;
            }
        }
    });

    ui.data_mut(|d| {
        d.insert_temp(state_id.with("filter"), filter);
        d.insert_persisted(state_id.with("active"), active_only);
        d.insert_persisted(state_id.with("collapsed"), collapsed);
    });
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn groups_follow_the_naming_conventions() {
        assert_eq!(group_of("mouthSmileLeft").as_deref(), Some("mouth"));
        assert_eq!(group_of("JawOpen").as_deref(), Some("Jaw"));
        assert_eq!(group_of("Fcl_BRW_Angry").as_deref(), Some("Fcl_BRW"));
        assert_eq!(group_of("v_aa").as_deref(), Some("v"));
        assert_eq!(group_of("vrc.v_sil").as_deref(), Some("vrc"));
        assert_eq!(group_of("smile"), None);
        assert_eq!(group_of("ARKIT"), None, "all caps has no word break");
    }

    #[test]
    fn a_group_of_one_folds_into_other() {
        let n = names(&["mouthOpen", "mouthSmile", "eyeBlink", "smile"]);
        assert_eq!(groups(&n), ["mouth", "mouth", "Other", "Other"]);
    }

    #[test]
    fn filter_needs_every_word_and_ignores_case() {
        assert!(matches("mouthSmileLeft", "smile left"));
        assert!(matches("mouthSmileLeft", "  "));
        assert!(!matches("mouthSmileLeft", "smile right"));
    }

    #[test]
    fn rows_group_filter_and_skip() {
        let n = names(&["mouthOpen", "mouthSmile", "eyeBlinkL", "eyeBlinkR", "", "smile"]);
        let none = HashMap::new();
        let rows = build_rows(&n, &none, "", false, true, &HashSet::new());
        assert_eq!(rows[0], Row::Group { name: "eye".into(), count: 2, open: true });
        // Blank names are skipped; "Other" comes last.
        assert!(!rows.contains(&Row::Shape(4)));
        assert!(matches!(rows.last(), Some(Row::Shape(5))));
        // A closed group lists no shapes.
        let closed: HashSet<String> = ["mouth".to_owned()].into();
        let rows = build_rows(&n, &none, "", false, true, &closed);
        assert!(!rows.contains(&Row::Shape(0)));
        assert!(rows.iter().any(|r| matches!(r, Row::Group { name, open: false, .. } if name == "mouth")));
        // Typing opens everything it matched.
        let rows = build_rows(&n, &none, "open", false, true, &closed);
        assert!(rows.contains(&Row::Shape(0)));
        // Ungrouped lists shapes only.
        assert_eq!(build_rows(&n, &none, "", false, false, &HashSet::new()), [0, 1, 2, 3, 5].map(Row::Shape));
    }

    #[test]
    fn active_only_keeps_non_zero_weights() {
        let n = names(&["a", "b", "c"]);
        let w: HashMap<&str, f32> = [("b", 0.5), ("c", 0.0)].into();
        assert_eq!(build_rows(&n, &w, "", true, false, &HashSet::new()), [Row::Shape(1)]);
    }

    #[test]
    fn a_zero_weight_drops_the_entry() {
        let mut b = BlendShapes::default();
        set_weight(&mut b, "Smile", 0.4);
        set_weight(&mut b, "Blink", 1.0);
        assert_eq!(active_count(&b), 2);
        set_weight(&mut b, "Smile", 0.0);
        assert_eq!(b.weights, [("Blink".to_owned(), 1.0)]);
        assert_eq!(b.get("Smile"), 0.0);
    }

    #[test]
    fn eight_hundred_shapes_build_rows_quickly() {
        let n: Vec<String> = (0..800).map(|i| format!("Fcl_G{}_Shape{i}", i % 20)).collect();
        let w: HashMap<&str, f32> = n.iter().step_by(7).map(|s| (s.as_str(), 0.3)).collect();
        let t = std::time::Instant::now();
        for _ in 0..20 {
            let rows = build_rows(&n, &w, "shape", false, true, &HashSet::new());
            assert_eq!(rows.len(), 800 + 20);
        }
        // Rebuilt every frame while open: stays far below a frame.
        assert!(t.elapsed().as_millis() / 20 < 40, "{:?}", t.elapsed() / 20);
    }

    #[test]
    fn in_between_frames_ride_on_their_base_shape_row() {
        let n = names(&["Mouth", "Mouth@50", "Mouth@25", "Orphan@50", "Smile"]);
        let m = inbetween_markers(&n);
        assert_eq!(m[&0], vec![(2, 0.25), (1, 0.5)], "frames ascend by weight");
        assert!(!m.contains_key(&3), "a frame without a base is not a marker");
        let rows = build_rows(&n, &HashMap::new(), "", false, false, &HashSet::new());
        assert_eq!(rows, vec![Row::Shape(0), Row::Shape(3), Row::Shape(4)], "frames are not rows");
        // Searching a frame finds its base.
        let found = build_rows(&n, &HashMap::new(), "@25", false, false, &HashSet::new());
        assert_eq!(found, vec![Row::Shape(0)]);
    }
}
