//! Avatar Mask transform tree for the animator layer inspector: a popup that
//! lists a skeleton as an indented tree with a checkbox per transform
//! (Alt-click takes the whole subtree along), drawn from the selected
//! object's rig or from another model's skeleton ("Use skeleton from").
//! Not verified visually.

use std::path::Path;
use std::sync::Arc;

use egui::Id;
use zestengine_assets::avatar_mask::{AvatarMask, tree_order};
use crate::icons::Icon;
use crate::{theme, widgets};

use crate::anim_controller_editor::scan_files;

/// A skeleton as `(name, parent)` per joint.
pub(crate) type SkeletonTree = Vec<(String, Option<usize>)>;

const MODEL_EXTENSIONS: [&str; 4] = ["fbx", "glb", "gltf", "vrm"];

/// The first skeleton of the model at `path` (project-relative) as a tree.
pub(crate) fn model_skeleton(root: &Path, path: &str) -> Option<SkeletonTree> {
    let scene = zestengine_assets::load_model_with_meta(root.join(path)).ok()?;
    let skel = scene.skeletons.first()?;
    Some(skel.joints.iter().map(|j| (j.name.clone(), j.parent)).collect())
}

/// The "Transforms" button and its popup. `own` is the skeleton of the
/// object being edited. Returns whether the mask changed.
pub(crate) fn transforms_button(ui: &mut egui::Ui, id: Id, mask: &mut AvatarMask, own: &[(String, Option<usize>)], root: &Path) -> bool {
    let mut edited = false;
    let button = widgets::secondary_button(ui, Some(Icon::Plus), "Transforms");
    egui::Popup::menu(&button).close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside).show(|ui| {
        widgets::menu_frame(ui);
        let src_key = id.with("skeleton-from");
        let mut source: String = ui.data_mut(|d| d.get_temp::<String>(src_key)).unwrap_or_default();
        let files_key = id.with("model-files");
        let files: Arc<Vec<String>> = match ui.data_mut(|d| d.get_temp::<Arc<Vec<String>>>(files_key)) {
            Some(f) => f,
            None => {
                let mut all: Vec<String> = MODEL_EXTENSIONS.iter().flat_map(|e| scan_files(root, e)).collect();
                all.sort();
                let all = Arc::new(all);
                ui.data_mut(|d| d.insert_temp(files_key, all.clone()));
                all
            }
        };
        widgets::property_row(ui, "Use skeleton from", |ui| {
            let shown = if source.is_empty() { "This object".to_owned() } else { source.clone() };
            egui::ComboBox::from_id_salt(id.with("from")).width(190.0).selected_text(shown).show_ui(ui, |ui| {
                widgets::menu_frame(ui);
                ui.selectable_value(&mut source, String::new(), "This object");
                for f in files.iter() {
                    ui.selectable_value(&mut source, f.clone(), f);
                }
            });
        });
        ui.data_mut(|d| d.insert_temp(src_key, source.clone()));
        // Another model's skeleton is read once per choice.
        let tree: SkeletonTree = if source.is_empty() {
            own.to_vec()
        } else {
            let key = id.with(("skeleton", source.clone()));
            match ui.data_mut(|d| d.get_temp::<Arc<SkeletonTree>>(key)) {
                Some(t) => (*t).clone(),
                None => {
                    let t = model_skeleton(root, &source).unwrap_or_default();
                    ui.data_mut(|d| d.insert_temp(key, Arc::new(t.clone())));
                    t
                }
            }
        };
        if tree.is_empty() {
            ui.label(egui::RichText::new("No skeleton to list: select an object with an Animator, or pick a model.").size(theme::SIZE_META).color(theme::TEXT_MUTED));
            return;
        }
        ui.label(egui::RichText::new("Alt-click a transform to switch its whole subtree.").size(theme::SIZE_META).color(theme::TEXT_DIM));
        let names: Vec<String> = tree.iter().map(|(n, _)| n.clone()).collect();
        let parents: Vec<Option<usize>> = tree.iter().map(|(_, p)| *p).collect();
        let propagate = ui.input(|i| i.modifiers.alt);
        egui::ScrollArea::vertical().max_height(320.0).show(ui, |ui| {
            for (node, depth) in tree_order(&parents) {
                ui.horizontal(|ui| {
                    ui.add_space(depth as f32 * 12.0);
                    let mut on = mask.is_active_element(&names[node]);
                    let label = if names[node].is_empty() { "(object)" } else { names[node].as_str() };
                    if ui.checkbox(&mut on, label).changed() {
                        mask.set_transform(&names, &parents, node, on, propagate);
                        edited = true;
                    }
                });
            }
        });
    });
    edited
}
